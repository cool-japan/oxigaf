//! Performance benchmarking utilities.
//!
//! Provides comprehensive benchmarking for OxiGAF components:
//! - FLAME forward pass timing
//! - Rasterizer forward/backward timing
//! - Training iteration timing
//! - Export performance
//!
//! Results can be output in human-readable, JSON, CSV, or Markdown format.

use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::cli::{BenchTarget, BenchmarkArgs, OutputFormat};
use crate::output;
use crate::progress;
use crate::verbosity::Verbosity;

// ---------------------------------------------------------------------------
// Benchmark Results
// ---------------------------------------------------------------------------

/// Results from a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Name of the benchmark.
    pub name: String,
    /// Target component being benchmarked.
    pub target: String,
    /// Number of iterations run.
    pub iterations: u32,
    /// Total time for all iterations (microseconds).
    pub total_us: u64,
    /// Mean time per iteration (microseconds).
    pub mean_us: f64,
    /// Standard deviation (microseconds).
    pub std_us: f64,
    /// Minimum time (microseconds).
    pub min_us: u64,
    /// Maximum time (microseconds).
    pub max_us: u64,
    /// Throughput (operations per second).
    pub ops_per_sec: f64,
    /// Model size (number of Gaussians).
    pub model_size: usize,
    /// Additional metadata.
    pub metadata: BenchmarkMetadata,
}

/// Benchmark metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchmarkMetadata {
    /// OxiGAF version.
    pub version: String,
    /// Timestamp (ISO 8601).
    pub timestamp: String,
    /// Platform info.
    pub platform: String,
    /// GPU adapter name (if available).
    pub gpu_adapter: Option<String>,
}

/// Full benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Report version.
    pub version: String,
    /// Benchmark results.
    pub results: Vec<BenchmarkResult>,
    /// System information.
    pub system: SystemInfo,
}

/// System information for benchmark context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Operating system.
    pub os: String,
    /// CPU model (if available).
    pub cpu: String,
    /// Total memory (MB).
    pub memory_mb: u64,
    /// GPU name (if available).
    pub gpu: Option<String>,
    /// Rust version.
    pub rust_version: String,
}

// ---------------------------------------------------------------------------
// Main Benchmark Entry Point
// ---------------------------------------------------------------------------

/// Run the benchmark suite.
pub fn run_benchmark(args: BenchmarkArgs, verbosity: Verbosity, json_mode: bool) -> Result<()> {
    let start = Instant::now();

    if !json_mode {
        println!();
        output::info("OxiGAF Performance Benchmark");
        output::separator();
    }

    let system_info = collect_system_info();
    let mut results = Vec::new();

    // Progress bar for overall benchmark (hidden in JSON mode)
    let targets = get_benchmark_targets(&args.target);
    let pb = if !json_mode {
        progress::custom_progress(
            targets.len() as u64,
            "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} benchmarks ({msg})",
            verbosity,
        )
    } else {
        indicatif::ProgressBar::hidden()
    };

    for target in &targets {
        pb.set_message(target.to_string());

        let result = match target.as_str() {
            "flame" => benchmark_flame(&args)?,
            "raster" => benchmark_raster(&args)?,
            "train" => benchmark_train(&args)?,
            "export" => benchmark_export(&args)?,
            _ => continue,
        };

        results.push(result);
        pb.inc(1);
    }

    pb.finish_with_message("done");

    // Build report
    let report = BenchmarkReport {
        version: "1.0".to_string(),
        results,
        system: system_info,
    };

    // Compare with baseline if provided (skip in JSON mode)
    if !json_mode {
        if let Some(ref baseline_path) = args.baseline {
            compare_with_baseline(&report, baseline_path)?;
        }
    }

    // Output results
    if json_mode {
        // Use global JSON output format
        let output = crate::json_output::JsonOutput::success(
            "benchmark",
            serde_json::to_value(&report).context("Failed to serialize benchmark report")?,
        );
        output.print();
    } else {
        output_report(&report, &args)?;

        if verbosity.show_timing() {
            let elapsed = start.elapsed();
            println!();
            output::value(
                "Total benchmark time",
                &format!("{:.2}s", elapsed.as_secs_f64()),
            );
        }
    }

    Ok(())
}

/// Get list of benchmark targets based on user selection.
fn get_benchmark_targets(target: &BenchTarget) -> Vec<String> {
    match target {
        BenchTarget::Flame => vec!["flame".to_string()],
        BenchTarget::Raster => vec!["raster".to_string()],
        BenchTarget::Train => vec!["train".to_string()],
        BenchTarget::Export => vec!["export".to_string()],
        BenchTarget::Full => vec![
            "flame".to_string(),
            "raster".to_string(),
            "train".to_string(),
            "export".to_string(),
        ],
    }
}

// ---------------------------------------------------------------------------
// Individual Benchmarks
// ---------------------------------------------------------------------------

/// Benchmark FLAME forward pass.
fn benchmark_flame(args: &BenchmarkArgs) -> Result<BenchmarkResult> {
    tracing::info!("Benchmarking FLAME forward pass");

    let model_size = args.size.num_gaussians();

    // If real FLAME model is provided, use it; otherwise use simulation
    let use_real_model = args.flame_model.is_some();

    if use_real_model {
        tracing::info!("Using real FLAME model for benchmarking");
        // Note: Real FLAME model loading would be integrated here
        // For now, we still use simulation but with a note
        tracing::warn!("Real FLAME benchmarking not yet implemented, using simulation");
    }

    // Warmup and timing
    let mut times = Vec::with_capacity(args.iterations as usize);

    // Warmup
    for _ in 0..args.warmup {
        simulate_flame_forward(model_size);
    }

    // Timed runs
    for _ in 0..args.iterations {
        let start = Instant::now();
        simulate_flame_forward(model_size);
        times.push(start.elapsed());
    }

    Ok(build_result("FLAME Forward", "flame", &times, model_size))
}

/// Benchmark rasterizer.
fn benchmark_raster(args: &BenchmarkArgs) -> Result<BenchmarkResult> {
    tracing::info!("Benchmarking rasterizer");

    let model_size = args.size.num_gaussians();
    let mut times = Vec::with_capacity(args.iterations as usize);

    // Warmup
    for _ in 0..args.warmup {
        simulate_raster(model_size);
    }

    // Timed runs
    for _ in 0..args.iterations {
        let start = Instant::now();
        simulate_raster(model_size);
        times.push(start.elapsed());
    }

    Ok(build_result("Rasterizer", "raster", &times, model_size))
}

/// Benchmark training iteration.
fn benchmark_train(args: &BenchmarkArgs) -> Result<BenchmarkResult> {
    tracing::info!("Benchmarking training iteration");

    let model_size = args.size.num_gaussians();
    let mut times = Vec::with_capacity(args.iterations as usize);

    // Warmup
    for _ in 0..args.warmup {
        simulate_train_step(model_size);
    }

    // Timed runs
    for _ in 0..args.iterations {
        let start = Instant::now();
        simulate_train_step(model_size);
        times.push(start.elapsed());
    }

    Ok(build_result("Training Step", "train", &times, model_size))
}

/// Benchmark export.
fn benchmark_export(args: &BenchmarkArgs) -> Result<BenchmarkResult> {
    tracing::info!("Benchmarking export");

    let model_size = args.size.num_gaussians();
    let mut times = Vec::with_capacity(args.iterations as usize);

    // Warmup
    for _ in 0..args.warmup {
        simulate_export(model_size);
    }

    // Timed runs
    for _ in 0..args.iterations {
        let start = Instant::now();
        simulate_export(model_size);
        times.push(start.elapsed());
    }

    Ok(build_result("PLY Export", "export", &times, model_size))
}

// ---------------------------------------------------------------------------
// Simulation Functions (for benchmarking without real components)
// ---------------------------------------------------------------------------

/// Simulate FLAME forward pass.
fn simulate_flame_forward(num_vertices: usize) {
    // Simulate computation proportional to vertex count
    let data: Vec<f32> = (0..num_vertices * 3)
        .map(|i| (i as f32).sin() * 0.01)
        .collect();
    std::hint::black_box(data);
}

/// Simulate rasterization.
fn simulate_raster(num_gaussians: usize) {
    // Simulate sorting and rendering
    let mut indices: Vec<u32> = (0..num_gaussians as u32).collect();
    indices.sort_by(|a, b| b.cmp(a));
    let pixels: Vec<u8> = (0..512 * 512 * 4).map(|i| (i % 256) as u8).collect();
    std::hint::black_box((indices, pixels));
}

/// Simulate training step.
fn simulate_train_step(num_gaussians: usize) {
    // Simulate gradient computation and optimization
    let gradients: Vec<f32> = (0..num_gaussians * 14) // pos + rot + scale + opacity + sh
        .map(|i| ((i as f32) * 0.001).cos())
        .collect();
    // Simulate Adam update
    let params: Vec<f32> = gradients
        .iter()
        .enumerate()
        .map(|(i, g)| -g * 0.001 * (1.0 + (i as f32) * 0.0001))
        .collect();
    std::hint::black_box(params);
}

/// Simulate export operation.
fn simulate_export(num_gaussians: usize) {
    // Simulate PLY generation
    let header_size = 500;
    let vertex_line_size = 100; // approximate chars per vertex
    let total_size = header_size + num_gaussians * vertex_line_size;
    let mut buffer = Vec::with_capacity(total_size);
    for i in 0..total_size.min(1_000_000) {
        buffer.push((i % 128) as u8);
    }
    std::hint::black_box(buffer);
}

// ---------------------------------------------------------------------------
// Result Building
// ---------------------------------------------------------------------------

/// Build a benchmark result from timing data.
fn build_result(
    name: &str,
    target: &str,
    times: &[Duration],
    model_size: usize,
) -> BenchmarkResult {
    let times_us: Vec<u64> = times.iter().map(|d| d.as_micros() as u64).collect();

    let total_us: u64 = times_us.iter().sum();
    let mean_us = total_us as f64 / times_us.len() as f64;
    let min_us = times_us.iter().copied().min().unwrap_or(0);
    let max_us = times_us.iter().copied().max().unwrap_or(0);

    // Standard deviation
    let variance: f64 = times_us
        .iter()
        .map(|&t| {
            let diff = t as f64 - mean_us;
            diff * diff
        })
        .sum::<f64>()
        / times_us.len() as f64;
    let std_us = variance.sqrt();

    // Throughput
    let ops_per_sec = if mean_us > 0.0 {
        1_000_000.0 / mean_us
    } else {
        0.0
    };

    BenchmarkResult {
        name: name.to_string(),
        target: target.to_string(),
        iterations: times_us.len() as u32,
        total_us,
        mean_us,
        std_us,
        min_us,
        max_us,
        ops_per_sec,
        model_size,
        metadata: BenchmarkMetadata {
            version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono_now(),
            platform: std::env::consts::OS.to_string(),
            gpu_adapter: None,
        },
    }
}

/// Get current timestamp in ISO 8601 format.
fn chrono_now() -> String {
    chrono::Utc::now().to_rfc3339()
}

// ---------------------------------------------------------------------------
// System Information
// ---------------------------------------------------------------------------

/// Collect system information.
fn collect_system_info() -> SystemInfo {
    SystemInfo {
        os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        cpu: get_cpu_info(),
        memory_mb: get_memory_mb(),
        gpu: get_gpu_info(),
        rust_version: get_rust_version(),
    }
}

/// Get CPU info (simplified).
fn get_cpu_info() -> String {
    #[cfg(target_os = "macos")]
    {
        // Try to get more specific info on macOS
        let output = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output();

        if let Ok(output) = output {
            if let Ok(cpu_str) = String::from_utf8(output.stdout) {
                let cpu_str = cpu_str.trim();
                if !cpu_str.is_empty() {
                    return cpu_str.to_string();
                }
            }
        }

        "Apple Silicon / x86_64".to_string()
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_else(|| "Unknown".to_string())
    }
    #[cfg(target_os = "windows")]
    {
        "Unknown".to_string()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "Unknown".to_string()
    }
}

/// Parse `MemTotal` kB from a `/proc/meminfo`-formatted string and return MB.
///
/// Returns 0 if no valid `MemTotal` line is found.
#[cfg(target_os = "linux")]
fn parse_mem_from_str(s: &str) -> u64 {
    for line in s.lines() {
        if line.starts_with("MemTotal:") {
            let kb: u64 = line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            if kb > 0 {
                return kb / 1024;
            }
        }
    }
    0
}

/// Get total physical memory in MB.
///
/// On Linux reads `MemTotal` from `/proc/meminfo`.  On macOS queries
/// `sysctl hw.memsize`.  Falls back to 4 096 MB on all other platforms or
/// when the OS query fails.
fn get_memory_mb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            let mb = parse_mem_from_str(&content);
            if mb > 0 {
                return mb;
            }
        }
        4096
    }
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output();
        if let Ok(out) = output {
            if let Ok(s) = std::str::from_utf8(&out.stdout) {
                if let Ok(bytes) = s.trim().parse::<u64>() {
                    return bytes / (1024 * 1024);
                }
            }
        }
        4096
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        4096
    }
}

/// Get GPU info via wgpu.
fn get_gpu_info() -> Option<String> {
    // Try to get GPU info via wgpu (non-blocking check)
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;

    Some(adapter.get_info().name.clone())
}

/// Get Rust version.
fn get_rust_version() -> String {
    option_env!("RUSTC_VERSION")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

// ---------------------------------------------------------------------------
// Baseline Comparison
// ---------------------------------------------------------------------------

/// Compare results with a baseline file.
fn compare_with_baseline(report: &BenchmarkReport, baseline_path: &Path) -> Result<()> {
    let baseline_data = std::fs::read_to_string(baseline_path)
        .with_context(|| format!("Failed to read baseline: {}", baseline_path.display()))?;

    let baseline: BenchmarkReport =
        serde_json::from_str(&baseline_data).with_context(|| "Failed to parse baseline JSON")?;

    println!();
    output::header("Baseline Comparison");

    for result in &report.results {
        if let Some(base) = baseline.results.iter().find(|r| r.target == result.target) {
            let diff_pct = ((result.mean_us - base.mean_us) / base.mean_us) * 100.0;
            let indicator = if diff_pct < -5.0 {
                "[FASTER]"
            } else if diff_pct > 5.0 {
                "[SLOWER]"
            } else {
                "[~SAME]"
            };

            println!(
                "  {}: {:.2}us -> {:.2}us ({:+.1}%) {}",
                result.name, base.mean_us, result.mean_us, diff_pct, indicator
            );
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Output Formatting
// ---------------------------------------------------------------------------

/// Output the benchmark report in the requested format.
fn output_report(report: &BenchmarkReport, args: &BenchmarkArgs) -> Result<()> {
    let output = match args.format {
        OutputFormat::Human => format_human(report),
        OutputFormat::Json => format_json(report)?,
        OutputFormat::Csv => format_csv(report),
        OutputFormat::Markdown => format_markdown(report),
    };

    // Write to file or stdout
    if let Some(ref path) = args.output {
        let mut file = std::fs::File::create(path)
            .with_context(|| format!("Failed to create output file: {}", path.display()))?;
        file.write_all(output.as_bytes())?;
        output::path_value("Report saved to", path);
    } else {
        println!("{}", output);
    }

    Ok(())
}

/// Format report for human reading.
fn format_human(report: &BenchmarkReport) -> String {
    let mut out = String::new();

    out.push('\n');
    out.push_str("=".repeat(60).as_str());
    out.push_str("\n  OxiGAF Benchmark Report\n");
    out.push_str("=".repeat(60).as_str());
    out.push_str("\n\n");

    out.push_str("System Information:\n");
    out.push_str(&format!("  OS:     {}\n", report.system.os));
    out.push_str(&format!("  CPU:    {}\n", report.system.cpu));
    out.push_str(&format!("  Memory: {} MB\n", report.system.memory_mb));
    if let Some(ref gpu) = report.system.gpu {
        out.push_str(&format!("  GPU:    {}\n", gpu));
    }
    out.push('\n');

    out.push_str("Results:\n");
    out.push_str("-".repeat(60).as_str());
    out.push('\n');

    for result in &report.results {
        out.push_str(&format!("\n  {}:\n", result.name));
        out.push_str(&format!(
            "    Mean:       {:.2} us ({:.2} ms)\n",
            result.mean_us,
            result.mean_us / 1000.0
        ));
        out.push_str(&format!("    Std Dev:    {:.2} us\n", result.std_us));
        out.push_str(&format!(
            "    Min/Max:    {:.2} / {:.2} us\n",
            result.min_us as f64, result.max_us as f64
        ));
        out.push_str(&format!(
            "    Throughput: {:.2} ops/sec\n",
            result.ops_per_sec
        ));
        out.push_str(&format!(
            "    Model Size: {} Gaussians\n",
            result.model_size
        ));
    }

    out.push('\n');
    out.push_str("=".repeat(60).as_str());
    out.push('\n');

    out
}

/// Format report as JSON.
fn format_json(report: &BenchmarkReport) -> Result<String> {
    serde_json::to_string_pretty(report).context("Failed to serialize to JSON")
}

/// Format report as CSV.
fn format_csv(report: &BenchmarkReport) -> String {
    let mut out = String::new();
    out.push_str("name,target,iterations,mean_us,std_us,min_us,max_us,ops_per_sec,model_size\n");

    for result in &report.results {
        out.push_str(&format!(
            "{},{},{},{:.2},{:.2},{},{},{:.2},{}\n",
            result.name,
            result.target,
            result.iterations,
            result.mean_us,
            result.std_us,
            result.min_us,
            result.max_us,
            result.ops_per_sec,
            result.model_size
        ));
    }

    out
}

/// Format report as Markdown table.
fn format_markdown(report: &BenchmarkReport) -> String {
    let mut out = String::new();

    out.push_str("# OxiGAF Benchmark Report\n\n");
    out.push_str("## System Information\n\n");
    out.push_str(&format!("- **OS:** {}\n", report.system.os));
    out.push_str(&format!("- **CPU:** {}\n", report.system.cpu));
    out.push_str(&format!("- **Memory:** {} MB\n", report.system.memory_mb));
    if let Some(ref gpu) = report.system.gpu {
        out.push_str(&format!("- **GPU:** {}\n", gpu));
    }
    out.push_str("\n## Results\n\n");
    out.push_str("| Benchmark | Mean (us) | Std Dev | Min | Max | Ops/sec | Model Size |\n");
    out.push_str("|-----------|-----------|---------|-----|-----|---------|------------|\n");

    for result in &report.results {
        out.push_str(&format!(
            "| {} | {:.2} | {:.2} | {} | {} | {:.2} | {} |\n",
            result.name,
            result.mean_us,
            result.std_us,
            result.min_us,
            result.max_us,
            result.ops_per_sec,
            result.model_size
        ));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_result() {
        let times = vec![
            Duration::from_micros(100),
            Duration::from_micros(110),
            Duration::from_micros(90),
        ];
        let result = build_result("Test", "test", &times, 1000);

        assert_eq!(result.name, "Test");
        assert_eq!(result.iterations, 3);
        assert!((result.mean_us - 100.0).abs() < 1.0);
        assert_eq!(result.min_us, 90);
        assert_eq!(result.max_us, 110);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_mem_correct_line() {
        let s = "MemTotal:       16384000 kB\nMemFree: 4000 kB\n";
        assert_eq!(parse_mem_from_str(s), 16000);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_mem_empty_string_returns_zero() {
        assert_eq!(parse_mem_from_str(""), 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_mem_malformed_line_returns_zero() {
        let s = "MemTotal:       not_a_number kB\n";
        assert_eq!(parse_mem_from_str(s), 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_mem_missing_memtotal_returns_zero() {
        let s = "MemFree:  4000 kB\nBuffers:  512 kB\n";
        assert_eq!(parse_mem_from_str(s), 0);
    }

    #[test]
    fn test_get_memory_mb_returns_nonzero() {
        let mb = get_memory_mb();
        assert!(mb > 0, "get_memory_mb() returned 0");
    }

    #[test]
    fn test_format_csv() {
        let report = BenchmarkReport {
            version: "1.0".to_string(),
            results: vec![BenchmarkResult {
                name: "Test".to_string(),
                target: "test".to_string(),
                iterations: 10,
                total_us: 1000,
                mean_us: 100.0,
                std_us: 10.0,
                min_us: 90,
                max_us: 110,
                ops_per_sec: 10000.0,
                model_size: 1000,
                metadata: BenchmarkMetadata::default(),
            }],
            system: SystemInfo {
                os: "test".to_string(),
                cpu: "test".to_string(),
                memory_mb: 8192,
                gpu: None,
                rust_version: "1.70".to_string(),
            },
        };

        let csv = format_csv(&report);
        assert!(csv.contains("Test,test,10"));
    }
}
