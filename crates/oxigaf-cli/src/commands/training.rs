//! `oxigaf training` — post-hoc analysis of a training run.
//!
//! | Subcommand | Library module |
//! |------------|----------------|
//! | `summary`   | [`crate::training_monitor`] |
//! | `smooth`    | [`crate::training_monitor`] |
//! | `report`    | [`crate::report_generator`] |
//! | `resume`    | [`crate::resume_analyzer`] |
//! | `telemetry` | [`crate::telemetry`] |
//!
//! # Relationship to `oxigaf monitor`
//!
//! `monitor dashboard` ([`crate::dashboard`]) *replays* a metrics stream as a
//! live terminal UI. This family *analyses* the same files after the fact:
//! it summarises the run, smooths the loss curve, renders a shareable report,
//! recommends a checkpoint to resume from, and digests a timing trace. Both
//! read the files `oxigaf train --metrics-output` writes, through the same
//! [`crate::commands::monitor::load_metric_records`] parser.
//!
//! # Telemetry input format
//!
//! [`crate::telemetry::TelemetryEvent`] has no serialisation of its own, so
//! this family defines the on-disk shape: a JSON array, or JSON Lines, of
//!
//! ```json
//! {"category": "step", "label": "train_step", "duration_us": 12345,
//!  "step": 10, "metadata": {"n_gaussians": 100000}}
//! ```
//!
//! `category` is one of `step`, `forward`, `backward`, `optimizer`,
//! `densification`, `render`, `data_load`, `checkpoint`, `loss`; anything
//! else becomes [`crate::telemetry::TelemetryCategory::Custom`].
//!
//! # Exit codes
//!
//! Unusable inputs are [`crate::error::CliError::InputInvalid`] /
//! [`crate::error::CliError::IoError`] → [`crate::error::EXIT_IO_ERROR`].
//! Argument-range errors are the catch-all status 1.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum, ValueHint};
use serde_json::{json, Value};

use crate::commands::image_io::input_invalid;
use crate::commands::monitor::{load_metric_records, MetricRecord, MetricsInput};
use crate::commands::{emit, prepare_output, CmdContext};
use crate::error::CliError;
use crate::progress_types::{OperationSpinner, TimingReport};
use crate::report_generator::{
    compute_trend, downsample_series, series_stats, write_report, ChartData, ChartSeries,
    MetricSummary, ReportBuilder, ReportFormat, ReportGeneratorConfig, ReportSection,
};
use crate::resume_analyzer::{
    analyze_checkpoints, CheckpointMetadata, CheckpointScanner, CheckpointScorer,
    ResumeRecommendation, ScoringWeights,
};
use crate::telemetry::{
    tel_detect_regression, tel_detect_spikes, tel_format_event, tel_format_report,
    tel_stats_by_category, tel_stats_by_label, LatencyStats, TelemetryCategory, TelemetryCollector,
    TelemetryConfig, TelemetryEvent, TelemetryReport, ThroughputTracker,
};
use crate::training_monitor::{
    compute_throughput, detect_divergence, ema_smooth, format_status_line, format_training_summary,
    loss_percentile, robust_smooth_loss, sma_smooth, summarize_training, MonitorConfig,
    TrainingEvent, TrainingMonitor,
};

/// `oxigaf training <command>`.
#[derive(Debug, Args)]
pub struct TrainingArgs {
    #[command(subcommand)]
    pub command: TrainingCommand,
}

/// Training-run analysis subcommands.
#[derive(Debug, Subcommand)]
pub enum TrainingCommand {
    /// Summarise a finished run: losses, throughput, stalls, divergence.
    Summary(SummaryArgs),

    /// Smooth a run's loss curve (EMA, SMA, or outlier-rejecting mean).
    Smooth(SmoothArgs),

    /// Render a run as a text, Markdown, or self-contained HTML report.
    Report(TrainingReportArgs),

    /// Recommend which checkpoint in a directory to resume training from.
    Resume(ResumeArgs),

    /// Digest a timing trace: per-category latency, spikes, regressions.
    Telemetry(TelemetryArgs),
}

/// Run the `training` family.
///
/// # Errors
///
/// Propagates unreadable metrics files, empty histories, and out-of-range
/// arguments.
pub fn run(args: TrainingArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        TrainingCommand::Summary(summary_args) => cmd_summary(summary_args, &ctx),
        TrainingCommand::Smooth(smooth_args) => cmd_smooth(smooth_args, &ctx),
        TrainingCommand::Report(report_args) => cmd_report(report_args, &ctx),
        TrainingCommand::Resume(resume_args) => cmd_resume(resume_args, &ctx),
        TrainingCommand::Telemetry(telemetry_args) => cmd_telemetry(telemetry_args, &ctx),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Render an `f32` for JSON, mapping non-finite values to `null`.
fn json_f32(value: f32) -> Value {
    if value.is_finite() {
        json!(value)
    } else {
        Value::Null
    }
}

/// Render an `f64` for JSON, mapping non-finite values to `null`.
fn json_f64(value: f64) -> Value {
    if value.is_finite() {
        json!(value)
    } else {
        Value::Null
    }
}

/// Turn parsed metric records into the monitor's event type.
///
/// The component losses, the Gaussian count and (when the run recorded one)
/// PSNR ride along as named extra metrics, so nothing the file carried is
/// dropped on the way in.
fn to_events(records: &[MetricRecord]) -> Vec<TrainingEvent> {
    records
        .iter()
        .map(|record| {
            let mut event = TrainingEvent::new(
                record.step,
                record.elapsed_seconds as f32,
                record.total_loss,
            )
            .with_metric("photometric_loss", record.photometric_loss)
            .with_metric("perceptual_loss", record.perceptual_loss)
            .with_metric("num_gaussians", record.num_gaussians as f32);
            if let Some(psnr) = record.psnr {
                event = event.with_metric("psnr", psnr);
            }
            event
        })
        .collect()
}

/// Load a metrics file and convert it to monitor events in one step.
fn load_events(
    path: &Path,
    format: MetricsInput,
) -> Result<(Vec<MetricRecord>, Vec<TrainingEvent>)> {
    if !path.is_file() {
        return Err(input_invalid(path, "not an existing metrics file"));
    }
    let records = load_metric_records(path, format)?;
    let events = to_events(&records);
    Ok((records, events))
}

// ---------------------------------------------------------------------------
// training summary
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf training summary`.
#[derive(Debug, Args)]
pub struct SummaryArgs {
    /// Metrics file written by `oxigaf train --metrics-output`.
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub metrics: PathBuf,

    /// Input encoding.
    #[arg(long, value_enum, default_value = "auto")]
    pub format: MetricsInput,

    /// Total planned steps, used for the progress bar and the ETA.
    #[arg(long)]
    pub total_steps: Option<usize>,

    /// Steps without improvement before a stall is declared.
    #[arg(long, default_value = "200")]
    pub patience: usize,

    /// Minimum relative improvement that counts as progress.
    #[arg(long, default_value = "0.001")]
    pub min_delta: f32,

    /// Events averaged for the smoothed loss.
    #[arg(long, default_value = "20")]
    pub smoothing_window: usize,

    /// Events averaged for the "recent" throughput figure.
    #[arg(long, default_value = "20")]
    pub recent_window: usize,

    /// Ratio of final to initial loss above which the run counts as diverged.
    #[arg(long, default_value = "2.0")]
    pub divergence_threshold: f32,
}

fn cmd_summary(args: SummaryArgs, ctx: &CmdContext) -> Result<()> {
    if args.patience == 0 || args.smoothing_window == 0 || args.recent_window == 0 {
        anyhow::bail!("--patience, --smoothing-window and --recent-window must all be at least 1");
    }

    let (records, events) = load_events(&args.metrics, args.format)?;
    let summary = summarize_training(&events, args.min_delta, args.patience)?;

    // `TrainingMonitor` trims its history to `max_history`, so the whole file
    // has to fit or the status line would describe only the tail. `validate`
    // additionally requires `eta_window <= max_history`, which the default of
    // 50 violates for any run shorter than 50 recorded steps.
    let max_history = events.len().max(1);
    let config = MonitorConfig {
        smoothing_window: args.smoothing_window,
        stall_patience: args.patience,
        improvement_threshold: args.min_delta,
        max_history,
        eta_window: args.recent_window.min(max_history),
    };
    let mut monitor = TrainingMonitor::new(config)?;
    if let Some(total) = args.total_steps {
        monitor = monitor.with_total_steps(total);
    }
    for event in &events {
        monitor.record(event.clone());
    }
    let snapshot = monitor.snapshot()?;
    let status_line = format_status_line(&snapshot, args.total_steps);

    // `compute_throughput` needs two events to form a rate; a single-record
    // file is reported as "unavailable" rather than as 0 steps/s.
    let throughput = compute_throughput(&events, args.recent_window).ok();

    let losses: Vec<f32> = events.iter().map(|event| event.loss).collect();
    let diverged = detect_divergence(&losses, args.divergence_threshold);
    let p50 = loss_percentile(&events, 50.0).ok();
    let p95 = loss_percentile(&events, 95.0).ok();

    emit(
        ctx,
        "training summary",
        json!({
            "metrics_file": args.metrics.display().to_string(),
            "records": records.len(),
            "summary": {
                "total_steps": summary.total_steps,
                "total_seconds": json_f32(summary.total_secs),
                "best_loss": json_f32(summary.best_loss),
                "best_step": summary.best_step,
                "final_loss": json_f32(summary.final_loss),
                "improvement_fraction": json_f32(summary.improvement_fraction),
                "mean_throughput": json_f32(summary.mean_throughput),
                "stall_count": summary.stall_count,
            },
            "snapshot": {
                "step": snapshot.step,
                "elapsed_seconds": json_f32(snapshot.elapsed_secs),
                "current_loss": json_f32(snapshot.current_loss),
                "smoothed_loss": json_f32(snapshot.smoothed_loss),
                "best_loss": json_f32(snapshot.best_loss),
                "steps_since_improvement": snapshot.steps_since_improvement,
                "is_stalled": snapshot.is_stalled,
                "steps_per_second": json_f32(snapshot.steps_per_second),
                "estimated_seconds_remaining": snapshot
                    .estimated_seconds_remaining
                    .map(json_f32)
                    .unwrap_or(Value::Null),
                "loss_improvement_rate": json_f32(snapshot.loss_improvement_rate),
            },
            "throughput": throughput
                .as_ref()
                .map(|stats| {
                    json!({
                        "mean_steps_per_second": json_f32(stats.mean_steps_per_second),
                        "peak_steps_per_second": json_f32(stats.peak_steps_per_second),
                        "current_steps_per_second": json_f32(stats.current_steps_per_second),
                        "recent_steps_per_second": json_f32(stats.recent_steps_per_second),
                    })
                })
                .unwrap_or(Value::Null),
            "diverged": diverged,
            "divergence_threshold": json_f32(args.divergence_threshold),
            "loss_p50": p50.map(json_f32).unwrap_or(Value::Null),
            "loss_p95": p95.map(json_f32).unwrap_or(Value::Null),
        }),
        &[],
        || {
            println!("{status_line}");
            println!();
            println!("{}", format_training_summary(&summary));
            if let Some(ref stats) = throughput {
                println!(
                    "Throughput:    mean {:.2} / peak {:.2} / recent {:.2} steps/s",
                    stats.mean_steps_per_second,
                    stats.peak_steps_per_second,
                    stats.recent_steps_per_second,
                );
            }
            if let (Some(p50), Some(p95)) = (p50, p95) {
                println!("Loss p50/p95:  {p50:.6} / {p95:.6}");
            }
            if diverged {
                println!(
                    "\nWARNING: the loss ended {:.1}x above where it started — this run diverged.",
                    args.divergence_threshold
                );
            }
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// training smooth
// ---------------------------------------------------------------------------

/// Loss-smoothing method.
#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
pub enum SmoothMethod {
    /// Exponential moving average over the whole series.
    #[default]
    Ema,
    /// Simple moving average over a fixed window.
    Sma,
    /// A single outlier-rejecting mean of the whole series.
    Robust,
}

/// Arguments for `oxigaf training smooth`.
#[derive(Debug, Args)]
pub struct SmoothArgs {
    /// Metrics file written by `oxigaf train --metrics-output`.
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub metrics: PathBuf,

    /// Input encoding.
    #[arg(long, value_enum, default_value = "auto")]
    pub format: MetricsInput,

    /// Smoothing method.
    #[arg(long, value_enum, default_value = "ema")]
    pub method: SmoothMethod,

    /// EMA weight of the current sample, in `(0, 1]`.
    #[arg(long, default_value = "0.1")]
    pub alpha: f32,

    /// SMA window length in steps.
    #[arg(long, default_value = "20")]
    pub window: usize,

    /// Robust mean: reject samples beyond this many standard deviations.
    #[arg(long, default_value = "2.0")]
    pub k: f32,

    /// Print every smoothed value instead of only the summary.
    #[arg(long)]
    pub print_series: bool,
}

fn cmd_smooth(args: SmoothArgs, ctx: &CmdContext) -> Result<()> {
    if args.alpha <= 0.0 || args.alpha > 1.0 {
        anyhow::bail!("--alpha must be within (0.0, 1.0] (got {})", args.alpha);
    }
    if args.k <= 0.0 {
        anyhow::bail!("--k must be above 0 (got {})", args.k);
    }

    let (_records, events) = load_events(&args.metrics, args.format)?;
    let losses: Vec<f32> = events.iter().map(|event| event.loss).collect();

    // `TimingReport` is what turns "the smooth took a while" into a number a
    // caller can act on; the series work is the only phase worth timing here.
    let mut timing = TimingReport::new();
    let smoothed: Vec<f32> = match args.method {
        SmoothMethod::Ema => timing.time("ema", || ema_smooth(&losses, args.alpha)),
        SmoothMethod::Sma => timing.time("sma", || sma_smooth(&losses, args.window))?,
        SmoothMethod::Robust => {
            let value = timing.time("robust", || robust_smooth_loss(&losses, args.k))?;
            vec![value]
        }
    };

    let stats = series_stats(&smoothed);
    let final_value = smoothed.last().copied();

    emit(
        ctx,
        "training smooth",
        json!({
            "metrics_file": args.metrics.display().to_string(),
            "method": format!("{:?}", args.method).to_lowercase(),
            "input_points": losses.len(),
            "output_points": smoothed.len(),
            "final_value": final_value.map(json_f32).unwrap_or(Value::Null),
            "mean": stats.map(|s| json_f32(s.0)).unwrap_or(Value::Null),
            "min": stats.map(|s| json_f32(s.1)).unwrap_or(Value::Null),
            "max": stats.map(|s| json_f32(s.2)).unwrap_or(Value::Null),
            "std": stats.map(|s| json_f32(s.3)).unwrap_or(Value::Null),
            "series": smoothed.iter().copied().map(json_f32).collect::<Vec<_>>(),
        }),
        &[],
        || {
            println!(
                "{:?} over {} loss samples → {} value(s)",
                args.method,
                losses.len(),
                smoothed.len()
            );
            if let Some((mean, min, max, std)) = stats {
                println!("  mean {mean:.6}  min {min:.6}  max {max:.6}  std {std:.6}");
            }
            if let Some(value) = final_value {
                println!("  final {value:.6}");
            }
            if args.print_series {
                for (index, value) in smoothed.iter().enumerate() {
                    println!("  {index:>6}  {value:.6}");
                }
            }
            if ctx.verbosity.show_timing() {
                println!("\n{}", timing.format_table());
            }
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// training report
// ---------------------------------------------------------------------------

/// Report output format.
#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
pub enum ReportOutputFormat {
    /// Self-contained HTML with embedded CSS and SVG charts.
    #[default]
    Html,
    /// Plain text with ASCII tables.
    Text,
    /// GitHub-flavoured Markdown.
    Markdown,
}

impl From<ReportOutputFormat> for ReportFormat {
    fn from(value: ReportOutputFormat) -> Self {
        match value {
            ReportOutputFormat::Html => ReportFormat::Html,
            ReportOutputFormat::Text => ReportFormat::PlainText,
            ReportOutputFormat::Markdown => ReportFormat::Markdown,
        }
    }
}

/// Arguments for `oxigaf training report`.
#[derive(Debug, Args)]
pub struct TrainingReportArgs {
    /// Metrics file written by `oxigaf train --metrics-output`.
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub metrics: PathBuf,

    /// Input encoding.
    #[arg(long, value_enum, default_value = "auto")]
    pub input_format: MetricsInput,

    /// Output format.
    #[arg(long, value_enum, default_value = "html")]
    pub format: ReportOutputFormat,

    /// Write the report here instead of to stdout.
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    pub output: Option<PathBuf>,

    /// Report title.
    #[arg(long, default_value = "OxiGAF Training Report")]
    pub title: String,

    /// Downsample every chart series to at most this many points.
    #[arg(long, default_value = "500")]
    pub max_points: usize,

    /// Omit the charts (they are skipped in text and Markdown regardless).
    #[arg(long)]
    pub no_charts: bool,

    /// Rows of the raw-record tail table.
    #[arg(long, default_value = "10")]
    pub tail_rows: usize,

    /// Overwrite `--output` if it already exists.
    #[arg(long)]
    pub force: bool,
}

/// Build one metric row from a series, choosing the trend direction from the
/// data rather than asserting one.
fn metric_row(name: &str, unit: &str, values: &[f32], higher_is_better: bool) -> MetricSummary {
    MetricSummary {
        name: name.to_string(),
        value: values.last().copied().unwrap_or(0.0),
        unit: unit.to_string(),
        trend: compute_trend(values),
        is_good_trend: higher_is_better,
    }
}

fn cmd_report(args: TrainingReportArgs, ctx: &CmdContext) -> Result<()> {
    if args.max_points == 0 {
        anyhow::bail!("--max-points must be at least 1");
    }

    let (records, _events) = load_events(&args.metrics, args.input_format)?;

    let steps: Vec<f32> = records.iter().map(|r| r.step as f32).collect();
    let total_loss: Vec<f32> = records.iter().map(|r| r.total_loss).collect();
    let photometric: Vec<f32> = records.iter().map(|r| r.photometric_loss).collect();
    let perceptual: Vec<f32> = records.iter().map(|r| r.perceptual_loss).collect();
    let gaussians: Vec<f32> = records.iter().map(|r| r.num_gaussians as f32).collect();
    // PSNR is not part of `TrainingMetrics`; a run that did not record it
    // gets no PSNR row rather than a fabricated one.
    let psnr: Vec<f32> = records.iter().filter_map(|r| r.psnr).collect();
    let has_psnr = psnr.len() == records.len() && !psnr.is_empty();

    let mut metrics = vec![
        metric_row("Loss (total)", "", &total_loss, false),
        metric_row("Loss (photometric)", "", &photometric, false),
        metric_row("Loss (perceptual)", "", &perceptual, false),
        metric_row("Gaussians", "", &gaussians, true),
    ];
    if has_psnr {
        metrics.push(metric_row("PSNR", "dB", &psnr, true));
    }

    let config = ReportGeneratorConfig {
        format: args.format.into(),
        title: args.title.clone(),
        show_charts: !args.no_charts,
        max_series_points: args.max_points,
        ..ReportGeneratorConfig::default()
    };

    let x_values = downsample_series(&steps, args.max_points);
    let mut builder = ReportBuilder::new(&args.title)
        .subtitle(format!(
            "{} records from {}  |  final loss {:.6}",
            records.len(),
            args.metrics.display(),
            total_loss.last().copied().unwrap_or(0.0),
        ))
        .config(config)
        .add_metrics("Metrics Summary", metrics);

    if !args.no_charts {
        builder = builder.add_chart(
            ReportSection::Charts,
            "Loss Curves",
            ChartData {
                title: "Training Loss".to_string(),
                x_label: "Step".to_string(),
                y_label: "Loss".to_string(),
                x_values: x_values.clone(),
                series: vec![
                    ChartSeries {
                        label: "total".to_string(),
                        values: downsample_series(&total_loss, args.max_points),
                    },
                    ChartSeries {
                        label: "photometric".to_string(),
                        values: downsample_series(&photometric, args.max_points),
                    },
                    ChartSeries {
                        label: "perceptual".to_string(),
                        values: downsample_series(&perceptual, args.max_points),
                    },
                ],
            },
        );
        builder = builder.add_chart(
            ReportSection::Charts,
            "Gaussian Count",
            ChartData {
                title: "Live Gaussians".to_string(),
                x_label: "Step".to_string(),
                y_label: "Count".to_string(),
                x_values: x_values.clone(),
                series: vec![ChartSeries {
                    label: "num_gaussians".to_string(),
                    values: downsample_series(&gaussians, args.max_points),
                }],
            },
        );
        if has_psnr {
            builder = builder.add_chart(
                ReportSection::Charts,
                "PSNR",
                ChartData {
                    title: "PSNR".to_string(),
                    x_label: "Step".to_string(),
                    y_label: "dB".to_string(),
                    x_values,
                    series: vec![ChartSeries {
                        label: "psnr".to_string(),
                        values: downsample_series(&psnr, args.max_points),
                    }],
                },
            );
        }
    }

    if args.tail_rows > 0 {
        let start = records.len().saturating_sub(args.tail_rows);
        let rows: Vec<Vec<String>> = records[start..]
            .iter()
            .map(|record| {
                vec![
                    record.step.to_string(),
                    format!("{:.6}", record.total_loss),
                    format!("{:.6}", record.photometric_loss),
                    format!("{:.6}", record.perceptual_loss),
                    record.num_gaussians.to_string(),
                    format!("{:.1}", record.elapsed_seconds),
                ]
            })
            .collect();
        builder = builder.add_table(
            ReportSection::Diagnostics,
            "Last Recorded Steps",
            vec![
                "step".to_string(),
                "loss_total".to_string(),
                "loss_photometric".to_string(),
                "loss_perceptual".to_string(),
                "num_gaussians".to_string(),
                "elapsed_s".to_string(),
            ],
            rows,
        );
    }

    let rendered = builder.render()?;
    let format_name = format!("{:?}", args.format).to_lowercase();

    let mut artifacts: Vec<(&str, &Path)> = Vec::new();
    if let Some(ref output) = args.output {
        if !prepare_output(ctx, output, args.force)? {
            emit(
                ctx,
                "training report",
                json!({
                    "dry_run": true,
                    "would_create": [output.display().to_string()],
                    "bytes": rendered.len(),
                }),
                &[],
                || println!("Would write report: {}", output.display()),
            );
            return Ok(());
        }
        write_report(&rendered, output)?;
        artifacts.push(("report", output.as_path()));
    }

    match args.output {
        Some(ref output) => emit(
            ctx,
            "training report",
            json!({
                "metrics_file": args.metrics.display().to_string(),
                "records": records.len(),
                "format": format_name,
                "output": output.display().to_string(),
                "bytes": rendered.len(),
            }),
            &artifacts,
            || {
                println!(
                    "Wrote {} byte {format_name} report to {}",
                    rendered.len(),
                    output.display()
                );
            },
        ),
        // Without `--output` the report *is* the output; under `--json` it
        // rides inside the document so stdout stays one JSON value.
        None => emit(
            ctx,
            "training report",
            json!({
                "metrics_file": args.metrics.display().to_string(),
                "records": records.len(),
                "format": format_name,
                "report": rendered,
            }),
            &[],
            || println!("{rendered}"),
        ),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// training resume
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf training resume`.
#[derive(Debug, Args)]
pub struct ResumeArgs {
    /// Directory of checkpoint files to rank.
    #[arg(value_hint = ValueHint::DirPath)]
    pub directory: PathBuf,

    /// File extensions treated as checkpoints.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "json,bin,safetensors,ckpt"
    )]
    pub extensions: Vec<String>,

    /// Ignore checkpoints below this training step.
    #[arg(long, default_value = "0")]
    pub min_step: usize,

    /// Ignore checkpoints above this training step.
    #[arg(long)]
    pub max_step: Option<usize>,

    /// Weight of PSNR in the composite score.
    #[arg(long, default_value = "0.5")]
    pub psnr_weight: f32,

    /// Weight of loss stability in the composite score.
    #[arg(long, default_value = "0.3")]
    pub loss_weight: f32,

    /// Weight of Gaussian-count stability in the composite score.
    #[arg(long, default_value = "0.2")]
    pub gaussian_weight: f32,
}

/// The JSON shape of one scanned checkpoint.
///
/// The `-1.0` / `0` sentinels the scanner uses for "not recorded" become
/// `null`, so a consumer never mistakes an unknown PSNR for a real one.
fn checkpoint_json(checkpoint: &CheckpointMetadata) -> Value {
    json!({
        "path": checkpoint.path.display().to_string(),
        "step": checkpoint.step,
        "psnr_db": if checkpoint.psnr >= 0.0 { json_f32(checkpoint.psnr) } else { Value::Null },
        "loss": if checkpoint.loss >= 0.0 { json_f32(checkpoint.loss) } else { Value::Null },
        "num_gaussians": if checkpoint.num_gaussians > 0 {
            json!(checkpoint.num_gaussians)
        } else {
            Value::Null
        },
        "timestamp_secs": checkpoint.timestamp_secs,
        "file_size_bytes": checkpoint.file_size_bytes,
    })
}

fn cmd_resume(args: ResumeArgs, ctx: &CmdContext) -> Result<()> {
    for (name, weight) in [
        ("--psnr-weight", args.psnr_weight),
        ("--loss-weight", args.loss_weight),
        ("--gaussian-weight", args.gaussian_weight),
    ] {
        if weight < 0.0 || !weight.is_finite() {
            anyhow::bail!("{name} must be a finite value at or above 0 (got {weight})");
        }
    }
    if args.psnr_weight + args.loss_weight + args.gaussian_weight <= 0.0 {
        anyhow::bail!("at least one scoring weight must be above 0");
    }
    if !args.directory.is_dir() {
        return Err(input_invalid(&args.directory, "not an existing directory"));
    }

    let max_step = args.max_step.unwrap_or(usize::MAX);
    if max_step < args.min_step {
        anyhow::bail!(
            "--max-step ({max_step}) must not be below --min-step ({})",
            args.min_step
        );
    }

    let scanner = CheckpointScanner::new()
        .with_extensions(args.extensions.clone())
        .with_step_range(args.min_step, max_step);
    let scorer = CheckpointScorer::with_weights(ScoringWeights {
        psnr_weight: args.psnr_weight,
        loss_stability_weight: args.loss_weight,
        gaussian_stability_weight: args.gaussian_weight,
    });

    // Scanning stats every checkpoint file in the directory, which is slow
    // enough on a long run to be worth a spinner.
    let spinner = if ctx.human() && ctx.verbosity.show_progress() {
        Some(OperationSpinner::new(format!(
            "Scanning {} for checkpoints…",
            args.directory.display()
        )))
    } else {
        None
    };
    let recommendation: ResumeRecommendation =
        match analyze_checkpoints(&args.directory, &scanner, &scorer) {
            Ok(recommendation) => {
                if let Some(ref spinner) = spinner {
                    spinner.finish_ok();
                }
                recommendation
            }
            Err(e) => {
                if let Some(ref spinner) = spinner {
                    spinner.fail(e.to_string());
                }
                return Err(e.into());
            }
        };

    emit(
        ctx,
        "training resume",
        json!({
            "directory": args.directory.display().to_string(),
            "total_scanned": recommendation.total_scanned,
            "best_score": json_f32(recommendation.best_score),
            "confidence": json_f32(recommendation.confidence),
            "reason": recommendation.reason,
            "best_checkpoint": checkpoint_json(&recommendation.best_checkpoint),
            "alternatives": recommendation
                .alternatives
                .iter()
                .map(checkpoint_json)
                .collect::<Vec<_>>(),
            "resume_command": format!(
                "oxigaf train --resume {} …",
                recommendation.best_checkpoint.path.display()
            ),
        }),
        &[],
        || {
            println!("{}", recommendation.format_report());
            println!(
                "\nResume with:\n  oxigaf train --resume {} …",
                recommendation.best_checkpoint.path.display()
            );
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// training telemetry
// ---------------------------------------------------------------------------

/// Arguments for `oxigaf training telemetry`.
#[derive(Debug, Args)]
pub struct TelemetryArgs {
    /// Timing trace (JSON array or JSON Lines); see the module docs.
    #[arg(value_hint = ValueHint::FilePath)]
    pub events: PathBuf,

    /// Restrict the digest to one category.
    #[arg(long)]
    pub category: Option<String>,

    /// Keep only every N-th step.
    #[arg(long, default_value = "1")]
    pub step_interval: usize,

    /// Ring-buffer capacity; older events are dropped past this.
    #[arg(long, default_value = "100000")]
    pub max_events: usize,

    /// Flag events slower than `mean + N × stddev`.
    #[arg(long, default_value = "3.0")]
    pub spike_sigma: f32,

    /// Earlier trace to compare against for a regression check.
    #[arg(long, value_hint = ValueHint::FilePath)]
    pub baseline: Option<PathBuf>,

    /// Ratio of new to baseline mean duration that counts as a regression.
    #[arg(long, default_value = "1.2")]
    pub regression_ratio: f32,

    /// Planned total steps, used to estimate the remaining time.
    #[arg(long)]
    pub target_steps: Option<usize>,
}

/// Map a category name onto the library enum.
///
/// Unknown names become [`TelemetryCategory::Custom`] rather than an error:
/// the trace format is open, and a project that instruments its own phase
/// should still get statistics for it.
fn category_from_str(name: &str) -> TelemetryCategory {
    match name {
        "step" => TelemetryCategory::Step,
        "forward" => TelemetryCategory::Forward,
        "backward" => TelemetryCategory::Backward,
        "optimizer" => TelemetryCategory::Optimizer,
        "densification" => TelemetryCategory::Densification,
        "render" => TelemetryCategory::Render,
        "data_load" => TelemetryCategory::DataLoad,
        "checkpoint" => TelemetryCategory::Checkpoint,
        "loss" => TelemetryCategory::Loss,
        other => TelemetryCategory::Custom(other.to_string()),
    }
}

/// Parse one trace record.
fn event_from_json(value: &Value, index: usize) -> Result<TelemetryEvent> {
    let label = value
        .get("label")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("record {index} has no string \"label\""))?
        .to_string();
    let duration_us = value
        .get("duration_us")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            anyhow::anyhow!("record {index} (\"{label}\") has no integer \"duration_us\"")
        })?;
    let category = value
        .get("category")
        .and_then(Value::as_str)
        .map(category_from_str)
        .unwrap_or_else(|| TelemetryCategory::Custom("unspecified".to_string()));
    let step = value
        .get("step")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .min(usize::MAX as u64) as usize;
    let metadata = value
        .get("metadata")
        .and_then(Value::as_object)
        .map(|map| {
            map.iter()
                .filter_map(|(key, entry)| entry.as_f64().map(|v| (key.clone(), v)))
                .collect::<Vec<(String, f64)>>()
        })
        .unwrap_or_default();

    Ok(TelemetryEvent {
        category,
        label,
        duration_us,
        step,
        metadata,
    })
}

/// Read a trace file as a JSON array or as JSON Lines.
fn load_telemetry_events(path: &Path) -> Result<Vec<TelemetryEvent>> {
    if !path.is_file() {
        return Err(input_invalid(path, "not an existing telemetry trace"));
    }
    let text = std::fs::read_to_string(path).map_err(|e| CliError::IoError {
        context: format!("Failed to read telemetry trace: {}", path.display()),
        source: e,
    })?;

    let values: Vec<Value> =
        if text.trim_start().starts_with('[') {
            serde_json::from_str(&text)
                .with_context(|| format!("{} is not a JSON array of records", path.display()))?
        } else {
            let mut parsed = Vec::new();
            for (line_no, line) in text.lines().enumerate() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let line_number = line_no + 1;
                parsed.push(serde_json::from_str(trimmed).with_context(|| {
                    format!("{}:{line_number} is not valid JSON", path.display())
                })?);
            }
            parsed
        };

    if values.is_empty() {
        return Err(input_invalid(path, "contains no telemetry records"));
    }

    values
        .iter()
        .enumerate()
        .map(|(index, value)| event_from_json(value, index))
        .collect()
}

/// The JSON shape of a [`LatencyStats`] group.
fn latency_json(stats: &LatencyStats) -> Value {
    json!({
        "label": stats.label,
        "category": stats.category.as_str(),
        "count": stats.count,
        "mean_us": json_f64(stats.mean_us),
        "std_us": json_f64(stats.std_us),
        "min_us": stats.min_us,
        "max_us": stats.max_us,
        "p50_us": stats.p50_us,
        "p95_us": stats.p95_us,
        "p99_us": stats.p99_us,
        "total_us": stats.total_us,
    })
}

/// Largest step a replay will walk the collector up to.
///
/// [`TelemetryCollector::advance_step`] moves the counter one step at a time
/// and there is no "seek", so a trace claiming step 10^12 would spin for
/// hours instead of failing. Real runs are six figures at most, so anything
/// past this is a malformed trace and is reported as one.
const MAX_REPLAY_STEP: usize = 10_000_000;

/// Replay a trace through a collector so `--category` and `--step-interval`
/// are applied by the library's own filter rather than re-implemented here.
///
/// The collector only ever moves its step counter forward, so the trace is
/// sorted by step first; ties keep their original order.
fn replay(events: Vec<TelemetryEvent>, config: TelemetryConfig) -> Result<TelemetryCollector> {
    let mut sorted = events;
    sorted.sort_by_key(|event| event.step);

    if let Some(last) = sorted.last() {
        if last.step > MAX_REPLAY_STEP {
            anyhow::bail!(
                "telemetry trace claims step {}, above the {MAX_REPLAY_STEP} replay ceiling; \
                 the \"step\" field is a training iteration index, not a timestamp",
                last.step
            );
        }
    }

    let mut collector = TelemetryCollector::new(config);
    for event in sorted {
        while collector.step() < event.step {
            collector.advance_step();
        }
        collector.record(
            event.category.clone(),
            &event.label,
            event.duration_us,
            event.metadata.clone(),
        )?;
    }
    Ok(collector)
}

fn cmd_telemetry(args: TelemetryArgs, ctx: &CmdContext) -> Result<()> {
    if args.step_interval == 0 {
        anyhow::bail!("--step-interval must be at least 1");
    }
    if args.max_events == 0 {
        anyhow::bail!("--max-events must be at least 1");
    }
    if args.spike_sigma <= 0.0 {
        anyhow::bail!("--spike-sigma must be above 0 (got {})", args.spike_sigma);
    }
    if args.regression_ratio <= 0.0 {
        anyhow::bail!(
            "--regression-ratio must be above 0 (got {})",
            args.regression_ratio
        );
    }

    let config = TelemetryConfig {
        max_events: args.max_events,
        enabled: true,
        track_categories: args
            .category
            .as_deref()
            .map(|name| vec![category_from_str(name)])
            .unwrap_or_default(),
        step_interval: args.step_interval,
    };

    let parsed = load_telemetry_events(&args.events)?;
    let parsed_count = parsed.len();
    let collector = replay(parsed, config)?;
    let events = collector.events().to_vec();
    if events.is_empty() {
        anyhow::bail!(
            "no records survived the filters (--category / --step-interval); \
             {parsed_count} record(s) were read from {}",
            args.events.display()
        );
    }

    let per_category_stats = tel_stats_by_category(&events);
    // `tel_stats_by_label` works one category at a time; walking the
    // categories the trace actually contains is what turns it into the
    // whole-trace label breakdown.
    let mut per_label_stats: Vec<LatencyStats> = Vec::new();
    let mut seen: BTreeMap<String, ()> = BTreeMap::new();
    for event in &events {
        if seen
            .insert(event.category.as_str().to_string(), ())
            .is_none()
        {
            per_label_stats.extend(tel_stats_by_label(&events, &event.category));
        }
    }

    let bottleneck_category = per_category_stats
        .iter()
        .max_by_key(|stats| stats.total_us)
        .map(|stats| stats.category.clone());

    // Throughput is derived from the recorded step durations, never from
    // wall-clock time: this command replays a trace long after the run
    // finished, so an elapsed-time figure would be meaningless.
    let mut tracker = ThroughputTracker::new(args.max_events.min(1024));
    let mut step_events = 0usize;
    for event in &events {
        if event.category == TelemetryCategory::Step {
            tracker.record_step(event.duration_us);
            step_events += 1;
        }
    }
    let instrumented_us: u64 = events.iter().map(|event| event.duration_us).sum();
    let total_steps = events.iter().map(|event| event.step).max().unwrap_or(0) + 1;
    let steps_per_second = if step_events > 0 {
        tracker.steps_per_second()
    } else if instrumented_us > 0 {
        total_steps as f64 / (instrumented_us as f64 / 1_000_000.0)
    } else {
        0.0
    };
    let eta_seconds = args
        .target_steps
        .filter(|_| step_events > 0)
        .map(|target| tracker.eta_seconds(total_steps, target));

    let report = TelemetryReport {
        total_steps,
        // Instrumented time, not wall clock — see the comment above.
        total_duration_s: instrumented_us as f64 / 1_000_000.0,
        per_category_stats,
        per_label_stats,
        bottleneck_category,
        steps_per_second,
    };

    let spikes = tel_detect_spikes(&events, args.spike_sigma);
    let spike_lines: Vec<String> = spikes
        .iter()
        .filter_map(|&index| events.get(index).map(tel_format_event))
        .collect();

    let regression = match args.baseline {
        Some(ref baseline_path) => {
            let baseline = load_telemetry_events(baseline_path)?;
            Some(tel_detect_regression(
                &baseline,
                &events,
                args.regression_ratio,
            ))
        }
        None => None,
    };

    emit(
        ctx,
        "training telemetry",
        json!({
            "trace": args.events.display().to_string(),
            "records_read": parsed_count,
            "records_kept": collector.n_events(),
            "records_recorded": collector.total_events_recorded(),
            "total_steps": report.total_steps,
            "instrumented_seconds": json_f64(report.total_duration_s),
            "steps_per_second": json_f64(report.steps_per_second),
            "eta_seconds": eta_seconds.map(json_f64).unwrap_or(Value::Null),
            "bottleneck_category": report
                .bottleneck_category
                .as_ref()
                .map(|category| json!(category.as_str()))
                .unwrap_or(Value::Null),
            "per_category": report
                .per_category_stats
                .iter()
                .map(latency_json)
                .collect::<Vec<_>>(),
            "per_label": report
                .per_label_stats
                .iter()
                .map(latency_json)
                .collect::<Vec<_>>(),
            "spike_indices": spikes,
            "spikes": spike_lines,
            "baseline": args.baseline.as_ref().map(|p| p.display().to_string()),
            "regressed": regression.map(Value::Bool).unwrap_or(Value::Null),
        }),
        &[],
        || {
            println!("{}", tel_format_report(&report));
            if let Some(eta) = eta_seconds {
                println!("Estimated seconds remaining: {eta:.1}");
            }
            if spike_lines.is_empty() {
                println!("\nNo events beyond {:.1}σ.", args.spike_sigma);
            } else {
                println!("\n--- Spikes beyond {:.1}σ ---", args.spike_sigma);
                for line in &spike_lines {
                    println!("{line}");
                }
            }
            if let Some(regressed) = regression {
                let verdict = if regressed {
                    "REGRESSED"
                } else {
                    "no regression"
                };
                println!(
                    "\nAgainst the baseline at {:.2}x: {verdict}",
                    args.regression_ratio
                );
            }
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::EXIT_IO_ERROR;
    use crate::verbosity::Verbosity;

    fn ctx() -> CmdContext {
        CmdContext::new(Verbosity::Quiet, true, false)
    }

    /// Write a small CSV metrics file and return its path.
    fn write_metrics(name: &str, rows: usize) -> PathBuf {
        let path = std::env::temp_dir().join(name);
        let mut text =
            String::from("iteration,loss_total,loss_l1,loss_ssim,num_gaussians,elapsed_seconds\n");
        for step in 0..rows {
            let loss = 1.0 - (step as f32) * 0.01;
            text.push_str(&format!(
                "{step},{loss:.4},{:.4},0.0100,{},{:.1}\n",
                loss * 0.8,
                1000 + step,
                step as f32 * 0.5
            ));
        }
        std::fs::write(&path, text).expect("temp metrics write");
        path
    }

    #[test]
    fn events_carry_every_recorded_component() {
        let path = write_metrics("oxigaf_training_events.csv", 4);
        let (records, events) = load_events(&path, MetricsInput::Csv).expect("metrics parse");
        assert_eq!(records.len(), 4);
        assert_eq!(events.len(), 4);
        let names: Vec<&str> = events[0]
            .extra_metrics
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        assert!(names.contains(&"photometric_loss"), "got {names:?}");
        assert!(names.contains(&"num_gaussians"), "got {names:?}");
        let _ = std::fs::remove_file(&path);
    }

    /// Regression: `MonitorConfig`'s default `eta_window` of 50 fails
    /// validation for any run shorter than 50 steps, so `summary` used to be
    /// impossible to run on a short file. The window is clamped to the
    /// history length instead.
    #[test]
    fn summary_works_on_a_run_shorter_than_the_eta_window() {
        let path = write_metrics("oxigaf_training_short.csv", 3);
        let args = SummaryArgs {
            metrics: path.clone(),
            format: MetricsInput::Csv,
            total_steps: Some(100),
            patience: 200,
            min_delta: 0.001,
            smoothing_window: 20,
            recent_window: 20,
            divergence_threshold: 2.0,
        };
        assert!(cmd_summary(args, &ctx()).is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn smooth_rejects_an_out_of_range_alpha() {
        let path = write_metrics("oxigaf_training_alpha.csv", 4);
        let args = SmoothArgs {
            metrics: path.clone(),
            format: MetricsInput::Csv,
            method: SmoothMethod::Ema,
            alpha: 0.0,
            window: 2,
            k: 2.0,
            print_series: false,
        };
        assert!(cmd_smooth(args, &ctx()).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_metrics_file_is_an_input_error() {
        let path = std::env::temp_dir().join("oxigaf_training_absent.csv");
        let _ = std::fs::remove_file(&path);
        let err = load_events(&path, MetricsInput::Auto)
            .expect_err("a missing metrics file must not load");
        assert_eq!(
            crate::commands::runtime::to_cli_error(err).exit_code(),
            EXIT_IO_ERROR
        );
    }

    #[test]
    fn telemetry_records_round_trip_through_the_collector() {
        let path = std::env::temp_dir().join("oxigaf_training_trace.jsonl");
        let text = "\
{\"category\":\"step\",\"label\":\"train_step\",\"duration_us\":1000,\"step\":0}
{\"category\":\"render\",\"label\":\"raster\",\"duration_us\":400,\"step\":0}
{\"category\":\"step\",\"label\":\"train_step\",\"duration_us\":1200,\"step\":1}
{\"category\":\"step\",\"label\":\"train_step\",\"duration_us\":90000,\"step\":2}
";
        std::fs::write(&path, text).expect("temp trace write");

        let events = load_telemetry_events(&path).expect("trace parse");
        assert_eq!(events.len(), 4);

        // The 90 ms outlier must be flagged; the ordinary steps must not.
        let spikes = tel_detect_spikes(&events, 1.5);
        assert!(spikes.contains(&3), "spikes were {spikes:?}");

        // `--category render` must be applied by the collector's own filter.
        let filtered = replay(
            events,
            TelemetryConfig {
                max_events: 100,
                enabled: true,
                track_categories: vec![TelemetryCategory::Render],
                step_interval: 1,
            },
        )
        .expect("replay");
        assert_eq!(filtered.n_events(), 1);

        let _ = std::fs::remove_file(&path);
    }

    /// A record without `duration_us` is a broken trace, and the message has
    /// to name the offending record — a bare "invalid JSON" is useless on a
    /// 100 000-line file.
    #[test]
    fn telemetry_reports_the_offending_record() {
        let value = json!({"label": "raster", "category": "render"});
        let err =
            event_from_json(&value, 41).expect_err("a record without duration_us must not parse");
        let message = format!("{err}");
        assert!(message.contains("41"), "message was: {message}");
        assert!(message.contains("raster"), "message was: {message}");
    }

    /// Regression: the collector has no seek, so a trace whose `step` looks
    /// like a timestamp used to walk the counter up one increment at a time
    /// and hang. It has to fail instead.
    #[test]
    fn absurd_step_values_are_refused_rather_than_walked() {
        let events = vec![TelemetryEvent {
            category: TelemetryCategory::Step,
            label: "train_step".to_string(),
            duration_us: 1000,
            step: MAX_REPLAY_STEP + 1,
            metadata: Vec::new(),
        }];
        let err = replay(events, TelemetryConfig::default())
            .err()
            .expect("an absurd step must be refused");
        assert!(
            format!("{err}").contains("replay ceiling"),
            "message was: {err}"
        );
    }

    #[test]
    fn unknown_categories_become_custom() {
        assert_eq!(category_from_str("step"), TelemetryCategory::Step);
        assert_eq!(
            category_from_str("my_phase"),
            TelemetryCategory::Custom("my_phase".to_string())
        );
    }

    #[test]
    fn report_renders_markdown_without_an_output_file() {
        let path = write_metrics("oxigaf_training_report.csv", 6);
        let args = TrainingReportArgs {
            metrics: path.clone(),
            input_format: MetricsInput::Csv,
            format: ReportOutputFormat::Markdown,
            output: None,
            title: "Test Report".to_string(),
            max_points: 100,
            no_charts: true,
            tail_rows: 3,
            force: false,
        };
        assert!(cmd_report(args, &ctx()).is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn resume_rejects_an_inverted_step_range() {
        let dir = std::env::temp_dir().join("oxigaf_training_resume_range");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let args = ResumeArgs {
            directory: dir.clone(),
            extensions: vec!["json".to_string()],
            min_step: 100,
            max_step: Some(10),
            psnr_weight: 0.5,
            loss_weight: 0.3,
            gaussian_weight: 0.2,
        };
        assert!(cmd_resume(args, &ctx()).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
