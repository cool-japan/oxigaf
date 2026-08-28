//! `oxigaf perf` — micro-benchmark suite for the numeric kernels.
//!
//! Glue over [`crate::benchmark_suite`]. This is deliberately distinct from
//! the end-to-end `oxigaf benchmark` command: `benchmark` times the FLAME /
//! rasteriser / training pipeline on the GPU, while `perf suite` times the
//! CPU-side kernels with warmup, outlier filtering and throughput reporting.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use serde_json::json;

use crate::benchmark_suite::{
    bench_gaussian_bbox, bench_gaussian_centroid, bench_sort_f32, bench_vec_dot, bench_vec_sum,
    build_suite_report, format_suite_report, run_default_suite, run_suite_entry, BenchmarkConfig,
    BenchmarkResult, BenchmarkSuite, SuiteReport,
};
use crate::commands::{emit, CmdContext};

/// `oxigaf perf <command>`.
#[derive(Debug, Args)]
pub struct PerfArgs {
    #[command(subcommand)]
    pub command: PerfCommand,
}

/// Performance subcommands.
#[derive(Debug, Subcommand)]
pub enum PerfCommand {
    /// Run the CPU kernel benchmark suite.
    Suite(SuiteArgs),
}

/// Kernel selection for `perf suite`.
#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
pub enum PerfKernel {
    /// Every kernel below.
    #[default]
    All,
    /// Sum of a contiguous `f32` slice.
    VecSum,
    /// Dot product of two contiguous `f32` slices.
    VecDot,
    /// In-place sort of an `f32` slice.
    Sort,
    /// Centroid of a synthetic Gaussian cloud.
    GaussianCentroid,
    /// Axis-aligned bounding box of a synthetic Gaussian cloud.
    GaussianBbox,
}

/// Arguments for `oxigaf perf suite`.
#[derive(Debug, Args)]
pub struct SuiteArgs {
    /// Run the built-in default suite (fixed 1K/1M sizes) instead of a
    /// custom-sized run.
    #[arg(long)]
    pub default_suite: bool,

    /// Kernel to benchmark.
    #[arg(long, value_enum, default_value = "all")]
    pub kernel: PerfKernel,

    /// Problem size (elements or Gaussians) for the custom run.
    #[arg(long, default_value = "100000")]
    pub size: usize,

    /// Timed iterations per kernel.
    #[arg(long, default_value = "50")]
    pub iterations: usize,

    /// Warmup iterations discarded before timing.
    #[arg(long, default_value = "5")]
    pub warmup: usize,

    /// Keep timing beyond `--iterations` until this many milliseconds have
    /// elapsed (0 disables the floor).
    #[arg(long, default_value = "0")]
    pub min_time_ms: f64,

    /// Discard timings more than this many standard deviations from the mean.
    #[arg(long)]
    pub outlier_sigma: Option<f64>,

    /// Write the report (human text) to a file in addition to stdout.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

/// Run the `perf` family.
///
/// # Errors
///
/// Returns an error when the benchmark configuration is invalid, when a
/// kernel fails to run, or when the report file cannot be written.
pub fn run(args: PerfArgs, ctx: CmdContext) -> Result<()> {
    match args.command {
        PerfCommand::Suite(suite_args) => cmd_suite(suite_args, &ctx),
    }
}

fn cmd_suite(args: SuiteArgs, ctx: &CmdContext) -> Result<()> {
    if args.size == 0 {
        anyhow::bail!("--size must be at least 1");
    }
    if args.iterations == 0 {
        anyhow::bail!("--iterations must be at least 1");
    }
    if let Some(sigma) = args.outlier_sigma {
        if !(sigma.is_finite() && sigma > 0.0) {
            anyhow::bail!("--outlier-sigma must be a positive, finite number (got {sigma})");
        }
    }

    let suite = if args.default_suite {
        run_default_suite()?
    } else {
        run_custom_suite(&args)?
    };

    let report = build_suite_report(&suite)?;
    let rendered = format_suite_report(&report);

    if let Some(ref output) = args.output {
        if ctx.dry_run {
            if ctx.human() {
                println!("[dry-run] would write {}", output.display());
            }
        } else {
            if let Some(parent) = output.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).with_context(|| {
                        format!("Failed to create report directory: {}", parent.display())
                    })?;
                }
            }
            std::fs::write(output, rendered.as_bytes())
                .with_context(|| format!("Failed to write report: {}", output.display()))?;
        }
    }

    let payload = report_json(&report);
    let artifacts: Vec<(&str, &std::path::Path)> = match args.output {
        Some(ref output) if !ctx.dry_run => vec![("report", output.as_path())],
        _ => Vec::new(),
    };

    emit(ctx, "perf suite", payload, &artifacts, || {
        println!("{rendered}");
    });
    Ok(())
}

fn run_custom_suite(args: &SuiteArgs) -> Result<BenchmarkSuite> {
    let config = BenchmarkConfig {
        iterations: args.iterations,
        warmup: args.warmup,
        min_time_ms: args.min_time_ms,
        throughput_unit: "items".to_string(),
        throughput_count: args.size,
        outlier_sigma: args.outlier_sigma,
    };
    config.validate()?;

    let size = args.size;
    let mut suite = BenchmarkSuite::new();
    let selected = args.kernel;
    let wanted = |kernel: PerfKernel| selected == PerfKernel::All || selected == kernel;

    if wanted(PerfKernel::VecSum) {
        run_suite_entry(
            &mut suite,
            &format!("vec_sum_{size}"),
            "Sum of a contiguous f32 slice",
            || {
                std::hint::black_box(bench_vec_sum(size));
            },
            config.clone(),
        )?;
    }
    if wanted(PerfKernel::VecDot) {
        run_suite_entry(
            &mut suite,
            &format!("vec_dot_{size}"),
            "Dot product of two contiguous f32 slices",
            || {
                std::hint::black_box(bench_vec_dot(size));
            },
            config.clone(),
        )?;
    }
    if wanted(PerfKernel::Sort) {
        run_suite_entry(
            &mut suite,
            &format!("sort_f32_{size}"),
            "In-place sort of an f32 slice",
            || {
                std::hint::black_box(bench_sort_f32(size));
            },
            config.clone(),
        )?;
    }
    if wanted(PerfKernel::GaussianCentroid) {
        let mut centroid_config = config.clone();
        centroid_config.throughput_unit = "Gaussians".to_string();
        run_suite_entry(
            &mut suite,
            &format!("gaussian_centroid_{size}"),
            "Centroid of a synthetic Gaussian cloud",
            || {
                std::hint::black_box(bench_gaussian_centroid(size));
            },
            centroid_config,
        )?;
    }
    if wanted(PerfKernel::GaussianBbox) {
        let mut bbox_config = config.clone();
        bbox_config.throughput_unit = "Gaussians".to_string();
        run_suite_entry(
            &mut suite,
            &format!("gaussian_bbox_{size}"),
            "Axis-aligned bounding box of a synthetic Gaussian cloud",
            || {
                std::hint::black_box(bench_gaussian_bbox(size));
            },
            bbox_config,
        )?;
    }

    if suite.results.is_empty() {
        anyhow::bail!("No benchmark kernels selected");
    }
    Ok(suite)
}

fn result_json(result: &BenchmarkResult) -> serde_json::Value {
    json!({
        "name": result.name,
        "iterations": result.iterations,
        "mean_ns": result.mean_ns,
        "median_ns": result.median_ns,
        "min_ns": result.min_ns,
        "max_ns": result.max_ns,
        "std_ns": result.std_ns,
        "throughput": result.throughput,
        "throughput_unit": result.throughput_unit,
        "cv": result.cv,
    })
}

fn report_json(report: &SuiteReport) -> serde_json::Value {
    json!({
        "n_benchmarks": report.n_benchmarks,
        "fastest": report.fastest,
        "slowest": report.slowest,
        "total_time_ms": report.total_time_ms,
        "results": report.results.iter().map(result_json).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_args(kernel: PerfKernel) -> SuiteArgs {
        SuiteArgs {
            default_suite: false,
            kernel,
            size: 64,
            iterations: 2,
            warmup: 0,
            min_time_ms: 0.0,
            outlier_sigma: None,
            output: None,
        }
    }

    #[test]
    fn custom_suite_selects_a_single_kernel() {
        let suite = run_custom_suite(&tiny_args(PerfKernel::VecSum)).expect("suite should run");
        assert_eq!(suite.results.len(), 1);
        assert_eq!(suite.results[0].name, "vec_sum_64");
    }

    #[test]
    fn custom_suite_runs_every_kernel_for_all() {
        let suite = run_custom_suite(&tiny_args(PerfKernel::All)).expect("suite should run");
        assert_eq!(suite.results.len(), 5);
        let report = build_suite_report(&suite).expect("non-empty suite yields a report");
        assert_eq!(report.n_benchmarks, 5);
        let payload = report_json(&report);
        assert!(payload.get("results").is_some());
    }

    #[test]
    fn zero_iterations_are_rejected_by_config_validation() {
        // `cmd_suite` rejects `--iterations 0` before touching the suite; the
        // config validator is the second line of defence.
        let config = BenchmarkConfig {
            iterations: 0,
            ..BenchmarkConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
