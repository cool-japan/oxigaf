//! `oxigaf report` — comparison reports across training runs.
//!
//! Glue over [`crate::experiment_report`]: each `--run NAME=PATH` pair is a
//! metrics stream (the same CSV / JSON Lines files `oxigaf monitor` reads),
//! turned into an [`crate::experiment_report::ExperimentMetrics`] curve and
//! compared side by side, optionally rendered as a self-contained HTML page.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde_json::json;

use crate::commands::monitor::{load_metric_records, MetricsInput};
use crate::commands::{emit, prepare_output, CmdContext};
use crate::experiment_report::{
    ExperimentComparison, ExperimentMetrics, HtmlReportConfig, HtmlReportGenerator,
};

/// `oxigaf report <command>`.
#[derive(Debug, Args)]
pub struct ReportArgs {
    #[command(subcommand)]
    pub command: ReportCommand,
}

/// Reporting subcommands.
#[derive(Debug, Subcommand)]
pub enum ReportCommand {
    /// Compare several training runs and optionally emit an HTML report.
    Experiment(ExperimentArgs),
}

/// Arguments for `oxigaf report experiment`.
#[derive(Debug, Args)]
pub struct ExperimentArgs {
    /// A run to include, as `NAME=PATH` where `PATH` is a metrics file.
    /// Repeat the flag once per run.
    #[arg(long = "run", required = true, num_args = 1.., value_name = "NAME=PATH")]
    pub runs: Vec<String>,

    /// Encoding of the metrics files.
    #[arg(long, value_enum, default_value = "auto")]
    pub format: MetricsInput,

    /// Write a self-contained HTML report here.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Title used in the HTML report.
    #[arg(long, default_value = "OxiGAF Experiment Comparison")]
    pub title: String,

    /// Omit the Gaussian-count chart from the HTML report.
    #[arg(long)]
    pub no_gaussians_chart: bool,

    /// Omit the hyperparameter table from the HTML report.
    #[arg(long)]
    pub no_hyperparams: bool,

    /// SVG chart width in pixels.
    #[arg(long, default_value = "900")]
    pub chart_width: usize,

    /// SVG chart height in pixels.
    #[arg(long, default_value = "350")]
    pub chart_height: usize,

    /// Overwrite the output file if it exists.
    #[arg(long)]
    pub force: bool,
}

/// Split a `NAME=PATH` run specification.
fn parse_run_spec(spec: &str) -> Result<(String, PathBuf)> {
    let (name, path) = spec
        .split_once('=')
        .ok_or_else(|| anyhow::anyhow!("Run specification {spec:?} must be NAME=PATH"))?;
    if name.trim().is_empty() {
        anyhow::bail!("Run specification {spec:?} has an empty name");
    }
    if path.trim().is_empty() {
        anyhow::bail!("Run specification {spec:?} has an empty path");
    }
    Ok((name.trim().to_string(), PathBuf::from(path.trim())))
}

/// Load one metrics file into an experiment curve.
///
/// Returns the curve plus whether the stream carried real PSNR values; a
/// stream without PSNR yields a flat zero curve, which is worth telling the
/// user about rather than presenting as a measurement.
fn load_experiment(
    name: &str,
    path: &Path,
    format: MetricsInput,
) -> Result<(ExperimentMetrics, bool)> {
    let records = load_metric_records(path, format)?;
    let mut experiment = ExperimentMetrics::new(name);
    experiment.description = path.display().to_string();

    let mut has_psnr = false;
    for record in &records {
        if record.psnr.is_some() {
            has_psnr = true;
        }
        experiment.add_step(
            record.step,
            record.psnr.unwrap_or(0.0),
            record.total_loss,
            record.num_gaussians,
        );
    }
    experiment.training_time_secs = records
        .last()
        .map(|record| record.elapsed_seconds)
        .unwrap_or(0.0);
    experiment.finalize();
    Ok((experiment, has_psnr))
}

/// Run the `report` family.
///
/// # Errors
///
/// Returns an error when a run specification is malformed, when a metrics
/// file cannot be read, or when the HTML report cannot be written.
pub fn run(args: ReportArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        ReportCommand::Experiment(experiment_args) => cmd_experiment(experiment_args, &ctx),
    }
}

fn cmd_experiment(args: ExperimentArgs, ctx: &CmdContext) -> Result<()> {
    if args.chart_width == 0 || args.chart_height == 0 {
        anyhow::bail!("--chart-width and --chart-height must both be positive");
    }

    let mut experiments = Vec::with_capacity(args.runs.len());
    let mut runs_without_psnr = Vec::new();
    for spec in &args.runs {
        let (name, path) = parse_run_spec(spec)?;
        let (experiment, has_psnr) = load_experiment(&name, &path, args.format)?;
        if !has_psnr {
            runs_without_psnr.push(name);
        }
        experiments.push(experiment);
    }

    let summaries: Vec<serde_json::Value> = experiments
        .iter()
        .map(|experiment| {
            json!({
                "name": experiment.name,
                "source": experiment.description,
                "steps": experiment.steps.len(),
                "best_psnr": experiment.best_psnr,
                "final_loss": experiment.final_loss,
                "auc_psnr": experiment.auc_psnr(),
                "stability_score": experiment.stability_score(),
                "convergence_step": experiment.convergence_step(),
                "training_time_secs": experiment.training_time_secs,
            })
        })
        .collect();

    let comparison = ExperimentComparison::new(experiments)?;
    let ranking: Vec<serde_json::Value> = comparison
        .psnr_ranking()
        .into_iter()
        .map(|(index, name, psnr)| json!({ "index": index, "name": name, "best_psnr": psnr }))
        .collect();
    let table = comparison.format_text_table();

    let config = HtmlReportConfig {
        title: args.title.clone(),
        include_gaussians_chart: !args.no_gaussians_chart,
        include_hyperparams_table: !args.no_hyperparams,
        chart_width: args.chart_width,
        chart_height: args.chart_height,
    };
    let generator = HtmlReportGenerator::new(comparison, config);

    let mut artifacts: Vec<(&str, &Path)> = Vec::new();
    if let Some(ref output) = args.output {
        if prepare_output(ctx, output, args.force)? {
            generator
                .save(output)
                .with_context(|| format!("Failed to write report: {}", output.display()))?;
            artifacts.push(("html-report", output.as_path()));
        }
    }

    let payload = json!({
        "experiments": summaries,
        "psnr_ranking": ranking,
        "runs_without_psnr": runs_without_psnr,
        "output": args.output.as_ref().map(|p| p.display().to_string()),
    });

    emit(ctx, "report experiment", payload, &artifacts, || {
        println!("{table}");
        if !runs_without_psnr.is_empty() {
            println!(
                "note: no `psnr` field in the metrics stream for: {}. \
                 Their PSNR curves are flat zero.",
                runs_without_psnr.join(", ")
            );
        }
        if let Some(ref output) = args.output {
            if ctx.dry_run {
                println!("[dry-run] would write {}", output.display());
            } else {
                println!("Wrote {}", output.display());
            }
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_specs_require_name_and_path() {
        assert!(parse_run_spec("baseline").is_err());
        assert!(parse_run_spec("=metrics.jsonl").is_err());
        assert!(parse_run_spec("baseline=").is_err());
        let (name, path) = parse_run_spec(" baseline = runs/a.jsonl ").expect("valid spec");
        assert_eq!(name, "baseline");
        assert_eq!(path, PathBuf::from("runs/a.jsonl"));
    }

    #[test]
    fn experiment_curve_flags_a_missing_psnr_stream() {
        let path = std::env::temp_dir().join("oxigaf_report_no_psnr.jsonl");
        std::fs::write(
            &path,
            "{\"iteration\":1,\"loss_total\":0.5,\"num_gaussians\":10,\"elapsed_seconds\":1.0}\n",
        )
        .expect("write metrics");
        let (experiment, has_psnr) =
            load_experiment("run", &path, MetricsInput::Auto).expect("curve loads");
        assert!(!has_psnr);
        assert_eq!(experiment.steps, vec![1]);
        assert!((experiment.final_loss - 0.5).abs() < 1e-6);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn html_report_is_written_and_self_contained() {
        let metrics = std::env::temp_dir().join("oxigaf_report_metrics.jsonl");
        std::fs::write(
            &metrics,
            "{\"iteration\":1,\"loss_total\":0.5,\"psnr\":25.0,\"num_gaussians\":10,\"elapsed_seconds\":1.0}\n\
             {\"iteration\":2,\"loss_total\":0.4,\"psnr\":27.0,\"num_gaussians\":12,\"elapsed_seconds\":2.0}\n",
        )
        .expect("write metrics");
        let out = std::env::temp_dir().join("oxigaf_report_experiment.html");
        let _ = std::fs::remove_file(&out);

        let ctx = CmdContext::new(crate::verbosity::Verbosity::Quiet, true, false);
        let args = ExperimentArgs {
            runs: vec![format!("baseline={}", metrics.display())],
            format: MetricsInput::Auto,
            output: Some(out.clone()),
            title: "Test".to_string(),
            no_gaussians_chart: false,
            no_hyperparams: false,
            chart_width: 400,
            chart_height: 200,
            force: true,
        };
        cmd_experiment(args, &ctx).expect("report generation succeeds");

        let html = std::fs::read_to_string(&out).expect("report exists");
        assert!(html.contains("<html"), "expected an HTML document");
        assert!(
            html.contains("baseline"),
            "the run name should appear in the report"
        );

        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_file(&metrics);
    }
}
