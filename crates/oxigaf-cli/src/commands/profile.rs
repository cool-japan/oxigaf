//! `oxigaf profile` — turn a phase-timing log into a bottleneck report.
//!
//! Wires [`crate::profiling_report`]. The training loop records one
//! [`PhaseRecord`] per phase per step; this command ingests those records
//! from a JSON file and produces the per-phase statistics, the bottleneck
//! ranking, and optionally a standalone HTML page.
//!
//! # Input format
//!
//! A JSON array of objects. `name` and `duration_ms` are required; `step`,
//! `memory_bytes` and `num_gaussians` default to zero.
//!
//! ```json
//! [
//!   {"name": "forward", "step": 0, "duration_ms": 12.5, "num_gaussians": 50000},
//!   {"name": "backward", "step": 0, "duration_ms": 21.0}
//! ]
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde_json::json;

use crate::commands::{emit, prepare_output, CmdContext};
use crate::profiling_report::{
    format_bytes, format_duration_ms, format_throughput, PhaseRecord, PhaseStats,
    ProfilingCollector, ProfilingConfig, ProfilingReport,
};

/// `oxigaf profile <command>`.
#[derive(Debug, Args)]
pub struct ProfileArgs {
    #[command(subcommand)]
    pub command: ProfileCommand,
}

/// Profiling subcommands.
#[derive(Debug, Subcommand)]
pub enum ProfileCommand {
    /// Summarise a phase-timing log.
    Report(ProfileReportArgs),
}

/// Run the `profile` family.
///
/// # Errors
///
/// Propagates unreadable or malformed record files and refused overwrites.
pub fn run(args: ProfileArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        ProfileCommand::Report(report_args) => cmd_report(report_args, &ctx),
    }
}

/// Arguments for `oxigaf profile report`.
#[derive(Debug, Args)]
pub struct ProfileReportArgs {
    /// JSON array of phase records.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Only keep these phases (comma-separated); empty keeps everything.
    #[arg(long, default_value = "")]
    pub phases: String,

    /// Ring-buffer capacity; older records are dropped once it is full.
    #[arg(long, default_value = "100000")]
    pub max_records: usize,

    /// First step of the reported range (requires `--end`).
    #[arg(long, requires = "end")]
    pub start: Option<usize>,

    /// Last step of the reported range, inclusive (requires `--start`).
    #[arg(long, requires = "start")]
    pub end: Option<usize>,

    /// Also break the run into windows of this many steps (0 disables).
    #[arg(long, default_value = "0")]
    pub report_interval: usize,

    /// How many phases the bottleneck ranking lists.
    #[arg(long, default_value = "5")]
    pub top: usize,

    /// Write a standalone HTML report here.
    #[arg(long)]
    pub html: Option<PathBuf>,

    /// Overwrite the HTML report if it exists.
    #[arg(long)]
    pub force: bool,
}

/// Parse the JSON record array into library records.
fn read_records(path: &Path) -> Result<Vec<PhaseRecord>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let document: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("Failed to parse {} as JSON", path.display()))?;
    let serde_json::Value::Array(entries) = document else {
        anyhow::bail!("{}: expected a JSON array of phase records", path.display());
    };

    let mut records = Vec::with_capacity(entries.len());
    for (position, entry) in entries.iter().enumerate() {
        let name = entry
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{}: record {position} has no string \"name\"",
                    path.display()
                )
            })?;
        let duration_ms = entry
            .get("duration_ms")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{}: record {position} has no numeric \"duration_ms\"",
                    path.display()
                )
            })?;
        if !duration_ms.is_finite() || duration_ms < 0.0 {
            anyhow::bail!(
                "{}: record {position} has a negative or non-finite duration ({duration_ms})",
                path.display()
            );
        }
        let step = entry
            .get("step")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;

        let mut record = PhaseRecord::new(name, step, duration_ms);
        if let Some(bytes) = entry
            .get("memory_bytes")
            .and_then(serde_json::Value::as_u64)
        {
            record = record.with_memory(bytes);
        }
        if let Some(count) = entry
            .get("num_gaussians")
            .and_then(serde_json::Value::as_u64)
        {
            record = record.with_gaussians(count as usize);
        }
        records.push(record);
    }
    Ok(records)
}

/// One phase's statistics as JSON.
fn phase_json(stats: &PhaseStats) -> serde_json::Value {
    json!({
        "name": stats.name,
        "count": stats.count,
        "mean_ms": stats.mean_ms,
        "min_ms": stats.min_ms,
        "max_ms": stats.max_ms,
        "std_ms": stats.std_ms,
        "p50_ms": stats.p50_ms,
        "p95_ms": stats.p95_ms,
        "p99_ms": stats.p99_ms,
        "mean_memory_bytes": stats.mean_memory_bytes,
        "mean_throughput_gps": stats.mean_throughput_gps,
        "fraction_of_total": stats.fraction_of_total,
    })
}

/// A whole report as JSON.
fn report_json(report: &ProfilingReport) -> serde_json::Value {
    json!({
        "num_steps": report.num_steps,
        "start_step": report.start_step,
        "end_step": report.end_step,
        "total_step_ms": report.total_step_ms,
        "phases": report.phases.iter().map(phase_json).collect::<Vec<_>>(),
    })
}

fn cmd_report(args: ProfileReportArgs, ctx: &CmdContext) -> Result<()> {
    if args.top == 0 {
        anyhow::bail!("--top must be at least 1");
    }
    let phases_to_track: Vec<String> = args
        .phases
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect();

    let config = ProfilingConfig {
        enabled: true,
        max_records: args.max_records,
        phases_to_track,
        report_interval_steps: args.report_interval,
    };
    config.validate()?;

    let records = read_records(&args.input)?;
    if records.is_empty() {
        anyhow::bail!("{} contains no phase records", args.input.display());
    }

    // `should_track` is the module's own filter, so `--phases` behaves
    // exactly like the in-process collector's phase allow-list.
    let mut collector = ProfilingCollector::new(config.max_records);
    let mut skipped = 0usize;
    for record in records {
        if config.should_track(&record.name) {
            collector.push(record);
        } else {
            skipped += 1;
        }
    }
    if collector.is_empty() {
        anyhow::bail!(
            "No records survived the --phases filter ({} skipped)",
            skipped
        );
    }

    let report = match (args.start, args.end) {
        (Some(start), Some(end)) => {
            if start > end {
                anyhow::bail!("--start {start} is after --end {end}");
            }
            collector.build_report_for_range(start, end)?
        }
        _ => collector.build_report()?,
    };

    // Optional per-window breakdown, driven by the config's own interval.
    let mut windows = Vec::new();
    if config.report_interval_steps > 0 && report.num_steps > 0 {
        let mut window_start = report.start_step;
        while window_start <= report.end_step {
            let window_end = window_start
                .saturating_add(config.report_interval_steps)
                .saturating_sub(1)
                .min(report.end_step);
            if let Ok(window) = collector.build_report_for_range(window_start, window_end) {
                windows.push(report_json(&window));
            }
            let Some(next) = window_end.checked_add(1) else {
                break;
            };
            window_start = next;
        }
    }

    let bottlenecks: Vec<serde_json::Value> = report
        .bottleneck_phases(args.top)
        .into_iter()
        .map(phase_json)
        .collect();

    let mut payload = json!({
        "input": args.input.display().to_string(),
        "records": collector.len(),
        "skipped": skipped,
        "phase_names": collector.phase_names(),
        "report": report_json(&report),
        "bottlenecks": bottlenecks,
        "windows": windows,
    });

    let mut artifacts: Vec<(&str, &Path)> = Vec::new();
    let mut written = false;
    if let Some(ref html_path) = args.html {
        if prepare_output(ctx, html_path, args.force)? {
            std::fs::write(html_path, report.to_html())
                .with_context(|| format!("Failed to write {}", html_path.display()))?;
            artifacts.push(("html", html_path.as_path()));
            written = true;
        }
        if let Some(map) = payload.as_object_mut() {
            map.insert("html".to_string(), json!(html_path.display().to_string()));
            map.insert("html_written".to_string(), json!(written));
        }
    }

    let table = report.format_table();
    let summary = report.format_summary();
    let slowest: Vec<String> = report
        .bottleneck_phases(args.top)
        .into_iter()
        .map(|stats| {
            format!(
                "  {} — mean {} ({:.1}% of the step), {}, {}",
                stats.name,
                format_duration_ms(stats.mean_ms),
                stats.fraction_of_total * 100.0,
                format_bytes(stats.mean_memory_bytes),
                format_throughput(stats.mean_throughput_gps),
            )
        })
        .collect();

    emit(ctx, "profile report", payload, &artifacts, || {
        println!("{summary}");
        println!("{table}");
        println!("Slowest phases:");
        for line in &slowest {
            println!("{line}");
        }
        if written {
            if let Some(ref html_path) = args.html {
                println!("Wrote {}", html_path.display());
            }
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verbosity::Verbosity;

    fn quiet_ctx() -> CmdContext {
        CmdContext::new(Verbosity::Quiet, true, false)
    }

    fn write_records(name: &str, body: &str) -> PathBuf {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, body).expect("write records");
        path
    }

    fn base_args(input: PathBuf) -> ProfileReportArgs {
        ProfileReportArgs {
            input,
            phases: String::new(),
            max_records: 1000,
            start: None,
            end: None,
            report_interval: 0,
            top: 3,
            html: None,
            force: false,
        }
    }

    #[test]
    fn records_parse_with_optional_fields() {
        let path = write_records(
            "oxigaf_profile_records.json",
            r#"[{"name":"forward","step":1,"duration_ms":10.0,"num_gaussians":100},
                {"name":"backward","duration_ms":20.0}]"#,
        );
        let records = read_records(&path).expect("records");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].step, 1);
        assert_eq!(records[0].num_gaussians, 100);
        assert_eq!(records[1].step, 0);
        assert_eq!(records[1].memory_bytes, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn records_reject_a_missing_duration() {
        let path = write_records("oxigaf_profile_bad.json", r#"[{"name":"forward"}]"#);
        assert!(read_records(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn records_reject_a_negative_duration() {
        let path = write_records(
            "oxigaf_profile_negative.json",
            r#"[{"name":"forward","duration_ms":-1.0}]"#,
        );
        assert!(read_records(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn records_reject_a_non_array_document() {
        let path = write_records("oxigaf_profile_object.json", r#"{"name":"forward"}"#);
        assert!(read_records(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn phase_filter_can_empty_the_collector() {
        let path = write_records(
            "oxigaf_profile_filter.json",
            r#"[{"name":"forward","duration_ms":10.0}]"#,
        );
        let mut args = base_args(path.clone());
        args.phases = "backward".to_string();
        assert!(cmd_report(args, &quiet_ctx()).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_inverted_range_is_rejected() {
        let path = write_records(
            "oxigaf_profile_range.json",
            r#"[{"name":"forward","step":0,"duration_ms":10.0}]"#,
        );
        let mut args = base_args(path.clone());
        args.start = Some(5);
        args.end = Some(1);
        assert!(cmd_report(args, &quiet_ctx()).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_report_ranks_the_slowest_phase_first() {
        let path = write_records(
            "oxigaf_profile_rank.json",
            r#"[{"name":"fast","step":0,"duration_ms":1.0},
                {"name":"slow","step":0,"duration_ms":50.0}]"#,
        );
        let args = base_args(path.clone());
        assert!(cmd_report(args, &quiet_ctx()).is_ok());

        let records = read_records(&path).expect("records");
        let mut collector = ProfilingCollector::new(100);
        for record in records {
            collector.push(record);
        }
        let report = collector.build_report().expect("report");
        let ranked = report.bottleneck_phases(1);
        assert_eq!(
            ranked.first().map(|stats| stats.name.as_str()),
            Some("slow")
        );
        let _ = std::fs::remove_file(&path);
    }
}
