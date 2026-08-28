//! `oxigaf monitor` — render a training run's metrics stream as a dashboard.
//!
//! Glue over [`crate::dashboard`]. The metrics files written by
//! `oxigaf train --metrics-output` (CSV or JSON Lines, see
//! [`crate::metrics`]) are replayed through
//! [`crate::dashboard::DashboardRenderer`].
//!
//! The record parser here is deliberately schema-tolerant: it reads the
//! fields [`crate::metrics::TrainingMetrics`] writes, and additionally picks
//! up a `psnr` column/field when the producing run recorded one. PSNR is not
//! part of `TrainingMetrics`, so a plain `--metrics-output` file renders the
//! PSNR bar as `0.0`; that is reported honestly rather than derived from an
//! unrelated loss term.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use serde_json::json;

use crate::commands::{emit, CmdContext};
use crate::dashboard::{DashboardConfig, DashboardRenderer, DashboardState};

/// One parsed row of a training metrics stream.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetricRecord {
    /// Training iteration.
    pub step: usize,
    /// Total combined loss.
    pub total_loss: f32,
    /// Photometric (L1) loss component.
    pub photometric_loss: f32,
    /// Perceptual (SSIM + LPIPS) loss component.
    pub perceptual_loss: f32,
    /// Live Gaussian count.
    pub num_gaussians: usize,
    /// Wall-clock seconds since the run started.
    pub elapsed_seconds: f64,
    /// PSNR in dB when the stream carries one.
    pub psnr: Option<f32>,
}

/// Input encoding of a metrics file.
#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
pub enum MetricsInput {
    /// Detect from the file extension (`.csv` → CSV, otherwise JSON Lines).
    #[default]
    Auto,
    /// Comma-separated values with a header row.
    Csv,
    /// One JSON object per line.
    Jsonl,
}

/// `oxigaf monitor <command>`.
#[derive(Debug, Args)]
pub struct MonitorArgs {
    #[command(subcommand)]
    pub command: MonitorCommand,
}

/// Monitoring subcommands.
#[derive(Debug, Subcommand)]
pub enum MonitorCommand {
    /// Render a metrics file as the training dashboard.
    Dashboard(DashboardArgs),
}

/// Arguments for `oxigaf monitor dashboard`.
#[derive(Debug, Args)]
pub struct DashboardArgs {
    /// Metrics file written by `oxigaf train --metrics-output`.
    #[arg(long)]
    pub metrics: PathBuf,

    /// Input encoding.
    #[arg(long, value_enum, default_value = "auto")]
    pub format: MetricsInput,

    /// Total steps used for the progress bar. Defaults to the highest step
    /// present in the file.
    #[arg(long)]
    pub total_steps: Option<usize>,

    /// Animate every record in place instead of printing only the last frame.
    #[arg(long)]
    pub replay: bool,

    /// Delay between animated frames, in milliseconds.
    #[arg(long, default_value = "0")]
    pub frame_delay_ms: u64,

    /// Terminal columns used for the dashboard panel.
    #[arg(long, default_value = "80")]
    pub width: usize,
}

/// Read a metrics file into records.
///
/// # Errors
///
/// Returns an error when the file cannot be read, when a CSV header is
/// missing, or when no usable record is present.
pub fn load_metric_records(path: &Path, format: MetricsInput) -> Result<Vec<MetricRecord>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read metrics file: {}", path.display()))?;

    let resolved = match format {
        MetricsInput::Auto => {
            let is_csv = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("csv"))
                .unwrap_or(false);
            if is_csv {
                MetricsInput::Csv
            } else {
                MetricsInput::Jsonl
            }
        }
        other => other,
    };

    let records = match resolved {
        MetricsInput::Csv => parse_csv(&text)?,
        _ => parse_jsonl(&text)?,
    };

    if records.is_empty() {
        anyhow::bail!("No metrics records found in {}", path.display());
    }
    Ok(records)
}

fn parse_jsonl(text: &str) -> Result<Vec<MetricRecord>> {
    let mut records = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(trimmed)
            .with_context(|| format!("Line {} is not valid JSON", index + 1))?;
        records.push(record_from_json(&value));
    }
    Ok(records)
}

fn number(value: &serde_json::Value, key: &str) -> Option<f64> {
    value.get(key).and_then(serde_json::Value::as_f64)
}

fn record_from_json(value: &serde_json::Value) -> MetricRecord {
    let lpips = number(value, "loss_lpips").unwrap_or(0.0);
    let ssim = number(value, "loss_ssim").unwrap_or(0.0);
    MetricRecord {
        step: number(value, "iteration")
            .or_else(|| number(value, "step"))
            .unwrap_or(0.0) as usize,
        total_loss: number(value, "loss_total").unwrap_or(0.0) as f32,
        photometric_loss: number(value, "loss_l1").unwrap_or(0.0) as f32,
        perceptual_loss: (ssim + lpips) as f32,
        num_gaussians: number(value, "num_gaussians").unwrap_or(0.0) as usize,
        elapsed_seconds: number(value, "elapsed_seconds").unwrap_or(0.0),
        psnr: number(value, "psnr").map(|v| v as f32),
    }
}

fn parse_csv(text: &str) -> Result<Vec<MetricRecord>> {
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("CSV metrics file has no header row"))?;
    let columns: Vec<&str> = header.split(',').map(str::trim).collect();
    let index_of = |name: &str| columns.iter().position(|column| *column == name);

    let i_iteration = index_of("iteration").or_else(|| index_of("step"));
    let i_total = index_of("loss_total");
    let i_l1 = index_of("loss_l1");
    let i_ssim = index_of("loss_ssim");
    let i_lpips = index_of("loss_lpips");
    let i_gaussians = index_of("num_gaussians");
    let i_elapsed = index_of("elapsed_seconds");
    let i_psnr = index_of("psnr");

    let mut records = Vec::new();
    for line in lines {
        let cells: Vec<&str> = line.split(',').map(str::trim).collect();
        let cell = |index: Option<usize>| -> Option<f64> {
            index
                .and_then(|i| cells.get(i))
                .and_then(|raw| raw.parse::<f64>().ok())
        };
        let ssim = cell(i_ssim).unwrap_or(0.0);
        let lpips = cell(i_lpips).unwrap_or(0.0);
        records.push(MetricRecord {
            step: cell(i_iteration).unwrap_or(0.0) as usize,
            total_loss: cell(i_total).unwrap_or(0.0) as f32,
            photometric_loss: cell(i_l1).unwrap_or(0.0) as f32,
            perceptual_loss: (ssim + lpips) as f32,
            num_gaussians: cell(i_gaussians).unwrap_or(0.0) as usize,
            elapsed_seconds: cell(i_elapsed).unwrap_or(0.0),
            psnr: cell(i_psnr).map(|v| v as f32),
        });
    }
    Ok(records)
}

/// Run the `monitor` family.
///
/// # Errors
///
/// Returns an error when the metrics file cannot be read or parsed.
pub fn run(args: MonitorArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        MonitorCommand::Dashboard(dashboard_args) => cmd_dashboard(dashboard_args, &ctx),
    }
}

fn cmd_dashboard(args: DashboardArgs, ctx: &CmdContext) -> Result<()> {
    if args.width < 20 {
        anyhow::bail!("--width must be at least 20 columns");
    }
    let records = load_metric_records(&args.metrics, args.format)?;

    let total_steps = args.total_steps.unwrap_or_else(|| {
        records
            .iter()
            .map(|record| record.step)
            .max()
            .unwrap_or(0)
            .max(1)
    });

    let mut state = DashboardState::new(total_steps);
    let config = DashboardConfig {
        panel_width: args.width,
        ..DashboardConfig::default()
    };
    let mut renderer = DashboardRenderer::new(config);

    let animate = args.replay && ctx.human();
    let delay = std::time::Duration::from_millis(args.frame_delay_ms);

    for record in &records {
        apply_record(&mut state, record);
        if animate {
            print!("{}", renderer.render_frame(&state));
            if args.frame_delay_ms > 0 {
                std::thread::sleep(delay);
            }
        }
    }

    if animate {
        print!("{}", renderer.finish());
    }

    let payload = json!({
        "metrics_file": args.metrics.display().to_string(),
        "records": records.len(),
        "step": state.step,
        "total_steps": state.total_steps,
        "progress_fraction": state.progress_fraction(),
        "psnr": state.psnr,
        "total_loss": state.total_loss,
        "photometric_loss": state.photometric_loss,
        "perceptual_loss": state.perceptual_loss,
        "num_gaussians": state.num_gaussians,
        "iter_per_sec": state.iter_per_sec,
        "elapsed_secs": state.elapsed_secs,
        "eta_secs": state.eta_secs(),
        "has_psnr": records.iter().any(|record| record.psnr.is_some()),
    });

    let snapshot = renderer.render_frame_no_cursor(&state);
    let warn_missing_psnr = !records.iter().any(|record| record.psnr.is_some());

    emit(ctx, "monitor dashboard", payload, &[], || {
        if !animate {
            println!("{snapshot}");
        }
        if warn_missing_psnr {
            println!(
                "note: this metrics stream carries no `psnr` field, so the PSNR bar reads 0.0"
            );
        }
    });
    Ok(())
}

/// Fold one record into the dashboard state.
fn apply_record(state: &mut DashboardState, record: &MetricRecord) {
    state.update_step(
        record.step,
        record.psnr.unwrap_or(0.0),
        record.total_loss,
        record.photometric_loss,
        record.perceptual_loss,
        record.num_gaussians,
    );
    state.elapsed_secs = record.elapsed_seconds;
    state.iter_per_sec = if record.elapsed_seconds > 0.0 {
        (record.step as f64 / record.elapsed_seconds) as f32
    } else {
        0.0
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_records_parse_training_metrics_fields() {
        let text = concat!(
            r#"{"iteration":10,"loss_total":0.5,"loss_l1":0.3,"loss_ssim":0.1,"#,
            r#""loss_lpips":0.05,"loss_reg":0.05,"num_gaussians":1000,"#,
            r#""lr_position":0.0002,"lr_scaling":0.005,"lr_rotation":0.001,"#,
            r#""memory_mb":512,"elapsed_seconds":12.5}"#,
            "\n",
        );
        let records = parse_jsonl(text).expect("valid JSON lines");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].step, 10);
        assert!((records[0].total_loss - 0.5).abs() < 1e-6);
        assert!((records[0].perceptual_loss - 0.15).abs() < 1e-6);
        assert_eq!(records[0].num_gaussians, 1000);
        assert_eq!(records[0].psnr, None, "TrainingMetrics carries no PSNR");
    }

    #[test]
    fn csv_records_use_the_header_row_for_column_order() {
        let text = "loss_total,iteration,psnr\n0.25,7,31.5\n0.20,8,32.0\n";
        let records = parse_csv(text).expect("valid CSV");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].step, 7);
        assert_eq!(records[1].psnr, Some(32.0));
    }

    #[test]
    fn auto_format_picks_csv_by_extension() {
        let path = std::env::temp_dir().join("oxigaf_monitor_metrics.csv");
        std::fs::write(&path, "iteration,loss_total\n1,0.5\n").expect("write csv");
        let records = load_metric_records(&path, MetricsInput::Auto).expect("csv parses");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].step, 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn empty_metrics_files_are_rejected() {
        let path = std::env::temp_dir().join("oxigaf_monitor_empty.jsonl");
        std::fs::write(&path, "\n\n").expect("write empty file");
        assert!(load_metric_records(&path, MetricsInput::Auto).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn apply_record_tracks_throughput() {
        let mut state = DashboardState::new(100);
        apply_record(
            &mut state,
            &MetricRecord {
                step: 50,
                total_loss: 0.1,
                photometric_loss: 0.06,
                perceptual_loss: 0.04,
                num_gaussians: 42,
                elapsed_seconds: 10.0,
                psnr: Some(30.0),
            },
        );
        assert_eq!(state.step, 50);
        assert_eq!(state.num_gaussians, 42);
        assert!((state.iter_per_sec - 5.0).abs() < 1e-4);
        assert!((state.progress_fraction() - 0.5).abs() < 1e-4);
    }
}
