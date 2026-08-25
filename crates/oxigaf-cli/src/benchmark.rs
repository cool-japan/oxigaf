//! Performance benchmarking utilities.
//!
//! Every benchmark target drives a **real** OxiGAF component — no simulated or
//! synthetic stand-in work is ever timed or reported:
//!
//! | Target   | What is measured                                                     |
//! |----------|----------------------------------------------------------------------|
//! | `flame`  | [`FlameModel::forward`] on a model loaded from `--flame-model`        |
//! | `raster` | `Rasterizer::forward` and `Rasterizer::backward` on a real wgpu device |
//! | `train`  | One full `Trainer::train_step` (render → loss → backward → Adam)      |
//! | `export` | `export::export_ply` writing a real PLY file to a temporary directory |
//!
//! ## Prerequisites and skipping
//!
//! * `flame` requires `--flame-model <DIR>` (a directory of `.npy` files).
//! * `raster` and `train` require a working GPU adapter.
//!
//! When a prerequisite is missing the target is **skipped** and the reason is
//! recorded in the report (`skipped` array in JSON, a dedicated section in the
//! human and Markdown formats). Fabricated numbers are never emitted. If a
//! single target was requested explicitly (`--target raster`), a missing
//! prerequisite is a hard error instead of a skip.
//!
//! ## Timing resolution
//!
//! Per-iteration durations are recorded with [`Instant`] and truncated to whole
//! microseconds ([`Duration::as_micros`]). Operations faster than 1 µs therefore
//! report `0.00 us`; increase `--size` if that happens.
//!
//! ## Output modes
//!
//! Results can be output in human-readable, JSON, CSV, or Markdown format.
//! `--output` and `--baseline` are honoured in every mode, including the global
//! `--json` flag: in JSON mode the file receives the bare report (so it can be
//! fed straight back in through `--baseline`) while stdout carries the usual
//! `JsonOutput` envelope, and the baseline comparison travels inside the report
//! as `baseline_comparison` rather than being printed.

use std::io::Write;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use nalgebra as na;
use serde::{Deserialize, Serialize};

use oxigaf::flame::{FlameModel, FlameParams};
use oxigaf::render::gaussian::{GaussianAttributes, GaussianModel};
use oxigaf::render::{RasterConfig, Rasterizer, RenderCamera};
use oxigaf::trainer::{TensorBoardConfig, Trainer, TrainingConfig};

use crate::cli::{BenchTarget, BenchmarkArgs, OutputFormat};
use crate::output;
use crate::progress;
use crate::verbosity::Verbosity;

/// Image width used for the rasterizer and training benchmarks.
const BENCH_IMAGE_WIDTH: u32 = 512;
/// Image height used for the rasterizer and training benchmarks.
const BENCH_IMAGE_HEIGHT: u32 = 512;

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
    /// Model size — number of Gaussians, or FLAME vertices for the `flame` target.
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

/// A benchmark target that could not be executed, and why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkippedTarget {
    /// Target identifier (`flame`, `raster`, `train`, `export`).
    pub target: String,
    /// Human-readable reason the target was skipped.
    pub reason: String,
}

/// One baseline-vs-current comparison entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    /// Target identifier the comparison applies to.
    pub target: String,
    /// Display name of the benchmark.
    pub name: String,
    /// Mean iteration time recorded in the baseline (microseconds).
    pub baseline_mean_us: f64,
    /// Mean iteration time of the current run (microseconds).
    pub current_mean_us: f64,
    /// Relative change in percent, or `None` when the baseline mean is zero
    /// (the ratio would be infinite).
    pub diff_pct: Option<f64>,
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
    /// Targets that were requested but could not run.
    ///
    /// `#[serde(default)]` keeps older baseline files (written before this
    /// field existed) loadable.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skipped: Vec<SkippedTarget>,
    /// Comparison against `--baseline`, when one was supplied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_comparison: Option<Vec<BaselineComparison>>,
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
    validate_args(&args)?;

    let start = Instant::now();

    if !json_mode {
        println!();
        output::info("OxiGAF Performance Benchmark");
        output::separator();
    }

    let system_info = collect_system_info();
    let gpu_adapter = system_info.gpu.clone();
    let mut results = Vec::new();
    let mut skipped: Vec<SkippedTarget> = Vec::new();

    // Progress bar for overall benchmark (hidden in JSON mode)
    let targets = get_benchmark_targets(&args.target);
    // An explicitly requested single target must fail loudly; the aggregate
    // `full` run records unavailable targets as skips instead.
    let single_target = !matches!(args.target, BenchTarget::Full);
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

        let outcome = match target.as_str() {
            "flame" => benchmark_flame(&args, gpu_adapter.as_deref()),
            "raster" => benchmark_raster(&args, gpu_adapter.as_deref()),
            "train" => benchmark_train(&args, gpu_adapter.as_deref()),
            "export" => benchmark_export(&args, gpu_adapter.as_deref()),
            other => Err(anyhow!("Unknown benchmark target: {other}")),
        };

        match outcome {
            Ok(mut target_results) => results.append(&mut target_results),
            Err(err) => {
                if single_target {
                    pb.finish_and_clear();
                    return Err(err);
                }
                let reason = format!("{err:#}");
                tracing::warn!(
                    benchmark_target = %target,
                    reason = %reason,
                    "Benchmark target skipped — no numbers will be reported for it"
                );
                skipped.push(SkippedTarget {
                    target: target.clone(),
                    reason,
                });
            }
        }

        pb.inc(1);
    }

    pb.finish_with_message("done");

    // Compare with baseline if provided — in every output mode, so that CI
    // jobs running `--json --baseline base.json` get the regression data too.
    let baseline_comparison = match args.baseline {
        Some(ref baseline_path) => {
            let baseline = load_baseline(baseline_path)?;
            Some(compute_baseline_comparison(&results, &baseline))
        }
        None => None,
    };

    // Build report
    let report = BenchmarkReport {
        version: "1.0".to_string(),
        results,
        system: system_info,
        skipped,
        baseline_comparison,
    };

    // Output results
    if json_mode {
        // `--output` is honoured in JSON mode as well; the file receives the
        // bare report so that it can be fed back in via `--baseline`.
        if let Some(ref path) = args.output {
            let rendered = format_json(&report)?;
            write_report_file(&rendered, path)?;
        }

        // Use global JSON output format
        let output = crate::json_output::JsonOutput::success(
            "benchmark",
            serde_json::to_value(&report).context("Failed to serialize benchmark report")?,
        );
        output.print();
    } else {
        // The comparison is embedded in the human / Markdown / JSON renderings.
        // Print it explicitly when it would otherwise be invisible: the report
        // was saved to a file, or CSV (a flat per-result table) was requested.
        let comparison_hidden = args.output.is_some() || matches!(args.format, OutputFormat::Csv);
        let hidden_comparison = report
            .baseline_comparison
            .as_deref()
            .filter(|_| comparison_hidden);
        if let Some(comparison) = hidden_comparison {
            print_baseline_comparison(comparison);
        }

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

/// Validate benchmark arguments before any work is done.
///
/// `--warmup 0` is legitimate (no warmup), but `--iterations 0` leaves the
/// timing vector empty, which would make every statistic undefined.
fn validate_args(args: &BenchmarkArgs) -> Result<()> {
    if args.iterations == 0 {
        return Err(anyhow!(
            "--iterations must be at least 1 (got 0): benchmark statistics are \
             undefined without a timed run"
        ));
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

/// Benchmark the real FLAME forward pass.
///
/// Requires `--flame-model <DIR>`; there is no way to synthesize a meaningful
/// FLAME model, so without one the target cannot be measured.
fn benchmark_flame(
    args: &BenchmarkArgs,
    gpu_adapter: Option<&str>,
) -> Result<Vec<BenchmarkResult>> {
    let model_path = args.flame_model.as_ref().ok_or_else(|| {
        anyhow!(
            "FLAME benchmark requires a model: pass --flame-model <DIR> \
             (directory of .npy files produced by scripts/convert_flame.py)"
        )
    })?;

    tracing::info!(
        path = %model_path.display(),
        "Benchmarking FLAME forward pass with a real model"
    );

    let model = FlameModel::load(model_path)
        .with_context(|| format!("Failed to load FLAME model from {}", model_path.display()))?;
    let num_vertices = model.num_vertices();
    let params = FlameParams::neutral();

    // Warmup
    for _ in 0..args.warmup {
        std::hint::black_box(model.forward(&params));
    }

    // Timed runs
    let mut times = Vec::with_capacity(args.iterations as usize);
    for _ in 0..args.iterations {
        let start = Instant::now();
        let mesh = model.forward(&params);
        times.push(start.elapsed());
        std::hint::black_box(mesh);
    }

    Ok(vec![build_result(
        "FLAME Forward",
        "flame",
        &times,
        num_vertices,
        gpu_adapter,
    )])
}

/// Benchmark the real GPU rasterizer (forward and backward passes).
fn benchmark_raster(
    args: &BenchmarkArgs,
    gpu_adapter: Option<&str>,
) -> Result<Vec<BenchmarkResult>> {
    let num_gaussians = args.size.num_gaussians();
    let config = RasterConfig::new().with_resolution(BENCH_IMAGE_WIDTH, BENCH_IMAGE_HEIGHT);

    tracing::info!(
        num_gaussians,
        width = config.image_width,
        height = config.image_height,
        "Benchmarking GPU rasterizer"
    );

    let model = synthetic_gaussian_model(num_gaussians, config.sh_degree);
    let camera = benchmark_camera(config.image_width, config.image_height);

    let mut rasterizer = pollster::block_on(Rasterizer::new(config.clone()))
        .map_err(|e| anyhow!("GPU rasterizer initialisation failed: {e}"))?;
    rasterizer.upload_gaussians(&model);

    // Non-zero incoming image gradient so the backward pass has real work.
    let grad_image = vec![1e-3_f32; (config.image_width * config.image_height * 4) as usize];

    // Warmup (shader compilation, buffer allocation, GPU clock ramp-up)
    for _ in 0..args.warmup {
        let output = rasterizer
            .forward(&model, &camera)
            .map_err(|e| anyhow!("Rasterizer forward pass failed: {e}"))?;
        let gradients = rasterizer
            .backward(&model, &grad_image)
            .map_err(|e| anyhow!("Rasterizer backward pass failed: {e}"))?;
        std::hint::black_box((output, gradients));
    }

    // Timed runs — forward and backward are measured separately.
    let mut forward_times = Vec::with_capacity(args.iterations as usize);
    let mut backward_times = Vec::with_capacity(args.iterations as usize);
    for _ in 0..args.iterations {
        let t_forward = Instant::now();
        let output = rasterizer
            .forward(&model, &camera)
            .map_err(|e| anyhow!("Rasterizer forward pass failed: {e}"))?;
        forward_times.push(t_forward.elapsed());

        let t_backward = Instant::now();
        let gradients = rasterizer
            .backward(&model, &grad_image)
            .map_err(|e| anyhow!("Rasterizer backward pass failed: {e}"))?;
        backward_times.push(t_backward.elapsed());

        std::hint::black_box((output, gradients));
    }

    Ok(vec![
        build_result(
            "Rasterizer Forward",
            "raster",
            &forward_times,
            num_gaussians,
            gpu_adapter,
        ),
        build_result(
            "Rasterizer Backward",
            "raster_backward",
            &backward_times,
            num_gaussians,
            gpu_adapter,
        ),
    ])
}

/// Benchmark one real training iteration (`Trainer::train_step`).
fn benchmark_train(
    args: &BenchmarkArgs,
    gpu_adapter: Option<&str>,
) -> Result<Vec<BenchmarkResult>> {
    let num_gaussians = args.size.num_gaussians();
    let raster_config = RasterConfig::new().with_resolution(BENCH_IMAGE_WIDTH, BENCH_IMAGE_HEIGHT);
    let model = synthetic_gaussian_model(num_gaussians, raster_config.sh_degree);

    tracing::info!(num_gaussians, "Benchmarking training iteration");

    let (device, queue) = request_benchmark_gpu_device()?;

    // TensorBoard logging is disabled explicitly: a benchmark must not write
    // event files as a side effect.
    let training_config = TrainingConfig {
        tensorboard: TensorBoardConfig {
            enabled: false,
            ..TensorBoardConfig::default()
        },
        ..TrainingConfig::default()
    };

    let mut trainer = Trainer::new(training_config, model, raster_config, device, queue, 42)
        .map_err(|e| anyhow!("Failed to create trainer: {e}"))?;

    // Warmup
    for _ in 0..args.warmup {
        let step = trainer
            .train_step()
            .map_err(|e| anyhow!("Training step failed: {e}"))?;
        std::hint::black_box(step);
    }

    // Timed runs
    let mut times = Vec::with_capacity(args.iterations as usize);
    for _ in 0..args.iterations {
        let start = Instant::now();
        let step = trainer
            .train_step()
            .map_err(|e| anyhow!("Training step failed: {e}"))?;
        times.push(start.elapsed());
        std::hint::black_box(step);
    }

    Ok(vec![build_result(
        "Training Step",
        "train",
        &times,
        num_gaussians,
        gpu_adapter,
    )])
}

/// Benchmark the real PLY export path.
fn benchmark_export(
    args: &BenchmarkArgs,
    gpu_adapter: Option<&str>,
) -> Result<Vec<BenchmarkResult>> {
    let num_gaussians = args.size.num_gaussians();
    let model = synthetic_gaussian_model(num_gaussians, RasterConfig::default().sh_degree);

    let dir = std::env::temp_dir().join(format!("oxigaf_benchmark_export_{}", unique_suffix()));
    std::fs::create_dir_all(&dir).with_context(|| {
        format!(
            "Failed to create temporary export directory: {}",
            dir.display()
        )
    })?;
    let path = dir.join("benchmark.ply");

    tracing::info!(
        num_gaussians,
        path = %path.display(),
        "Benchmarking PLY export"
    );

    let measured = time_ply_export(args, &model, &path, gpu_adapter);

    // Best-effort cleanup; a failure here must not mask the benchmark outcome.
    if let Err(err) = std::fs::remove_dir_all(&dir) {
        tracing::debug!(
            path = %dir.display(),
            error = %err,
            "Failed to remove temporary export directory"
        );
    }

    measured
}

/// Timed body of the export benchmark, split out so the caller can always
/// clean up its temporary directory.
fn time_ply_export(
    args: &BenchmarkArgs,
    model: &GaussianModel,
    path: &Path,
    gpu_adapter: Option<&str>,
) -> Result<Vec<BenchmarkResult>> {
    // Warmup
    for _ in 0..args.warmup {
        crate::export::export_ply(model, path)?;
    }

    // Timed runs
    let mut times = Vec::with_capacity(args.iterations as usize);
    for _ in 0..args.iterations {
        let start = Instant::now();
        crate::export::export_ply(model, path)?;
        times.push(start.elapsed());
    }

    Ok(vec![build_result(
        "PLY Export",
        "export",
        &times,
        model.len(),
        gpu_adapter,
    )])
}

// ---------------------------------------------------------------------------
// Synthetic scene construction (real data structures, deterministic content)
// ---------------------------------------------------------------------------

/// Deterministic xorshift64 PRNG returning a value in `[0, 1)`.
///
/// Used so that benchmark scenes are reproducible across runs without pulling
/// in an RNG dependency or depending on wall-clock state.
#[inline]
fn xorshift_unit(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    // Top 24 bits → exactly representable in f32.
    ((*state >> 40) as f32) / 16_777_216.0
}

/// Build a Gaussian cloud of `n` primitives filling roughly the unit cube.
///
/// This is a real [`GaussianModel`] — the same structure produced by training
/// and consumed by the rasterizer and the exporters. Only its *contents* are
/// synthetic, so that the benchmark does not require a trained avatar.
fn synthetic_gaussian_model(n: usize, sh_degree: u32) -> GaussianModel {
    let sh_per_gaussian = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
    let log_scale = 0.02_f32.ln();

    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut gaussians = Vec::with_capacity(n);
    let mut sh_coeffs = Vec::with_capacity(n * sh_per_gaussian);

    for _ in 0..n {
        let x = xorshift_unit(&mut state) * 2.0 - 1.0;
        let y = xorshift_unit(&mut state) * 2.0 - 1.0;
        let z = xorshift_unit(&mut state) * 2.0 - 1.0;

        gaussians.push(GaussianAttributes {
            position: [x, y, z],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [log_scale; 3],
            // Logit space: 0.0 → sigmoid(0) = 0.5 opacity.
            opacity: 0.0,
        });

        for k in 0..sh_per_gaussian {
            // DC term carries the colour; higher-order bands start small but
            // non-zero so the SH evaluation shaders do real work.
            let value = if k < 3 {
                xorshift_unit(&mut state)
            } else {
                xorshift_unit(&mut state) * 0.01
            };
            sh_coeffs.push(value);
        }
    }

    GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree,
        face_indices: vec![0_u32; n],
        barycentric: vec![[1.0_f32 / 3.0; 3]; n],
        local_offsets: vec![[0.0_f32; 3]; n],
        is_rigid: vec![true; n],
    }
}

/// Copy a column-major nalgebra 4×4 matrix into a flat `[f32; 16]`.
fn mat4_to_array(m: &na::Matrix4<f32>) -> [f32; 16] {
    let mut out = [0.0_f32; 16];
    for (dst, src) in out.iter_mut().zip(m.as_slice()) {
        *dst = *src;
    }
    out
}

/// Build the camera used by the rasterizer benchmark: a right-handed pinhole
/// camera on the +Z axis looking at the origin.
fn benchmark_camera(width: u32, height: u32) -> RenderCamera {
    let eye = na::Point3::new(0.0_f32, 0.0, 4.0);
    let view = na::Matrix4::look_at_rh(&eye, &na::Point3::origin(), &na::Vector3::y());

    let fov_y = 45.0_f32.to_radians();
    let aspect = width as f32 / height as f32;
    let proj = na::Matrix4::new_perspective(aspect, fov_y, 0.01, 100.0);

    // Square pixels: the horizontal focal length equals the vertical one for a
    // projection built from a vertical FoV plus the width/height aspect ratio.
    let focal_y = height as f32 / (2.0 * (fov_y / 2.0).tan());
    let focal_x = focal_y;

    RenderCamera {
        view_matrix: mat4_to_array(&view),
        proj_matrix: mat4_to_array(&proj),
        position: [eye.x, eye.y, eye.z],
        focal: [focal_x, focal_y],
    }
}

/// Request a wgpu device suitable for the training benchmark.
///
/// The storage-buffer limit is raised to 16 because the rasterizer backward
/// pass binds 13+ storage buffers in a single shader stage.
fn request_benchmark_gpu_device() -> Result<(wgpu::Device, wgpu::Queue)> {
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
    .map_err(|e| anyhow!("No suitable GPU adapter found: {e}"))?;

    tracing::info!(
        adapter = adapter.get_info().name,
        backend = ?adapter.get_info().backend,
        "Selected GPU adapter for benchmarking"
    );

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("oxigaf_benchmark"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits {
            max_storage_buffers_per_shader_stage: 16,
            ..wgpu::Limits::default()
        },
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: wgpu::ExperimentalFeatures::default(),
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| anyhow!("GPU device creation failed: {e}"))?;

    Ok((device, queue))
}

/// A process- and time-unique suffix for temporary paths.
fn unique_suffix() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}_{}", std::process::id(), nanos)
}

// ---------------------------------------------------------------------------
// Result Building
// ---------------------------------------------------------------------------

/// Build a benchmark result from timing data.
///
/// An empty `times` slice yields all-zero statistics rather than `NaN`, so the
/// report always serializes (serde_json rejects non-finite floats).
fn build_result(
    name: &str,
    target: &str,
    times: &[Duration],
    model_size: usize,
    gpu_adapter: Option<&str>,
) -> BenchmarkResult {
    let times_us: Vec<u64> = times.iter().map(|d| d.as_micros() as u64).collect();
    let count = times_us.len();

    let total_us: u64 = times_us.iter().sum();
    let min_us = times_us.iter().copied().min().unwrap_or(0);
    let max_us = times_us.iter().copied().max().unwrap_or(0);

    let mean_us = if count == 0 {
        0.0
    } else {
        total_us as f64 / count as f64
    };

    // Standard deviation
    let std_us = if count == 0 {
        0.0
    } else {
        let variance: f64 = times_us
            .iter()
            .map(|&t| {
                let diff = t as f64 - mean_us;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;
        variance.sqrt()
    };

    // Throughput
    let ops_per_sec = if mean_us > 0.0 {
        1_000_000.0 / mean_us
    } else {
        0.0
    };

    BenchmarkResult {
        name: name.to_string(),
        target: target.to_string(),
        iterations: count as u32,
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
            gpu_adapter: gpu_adapter.map(str::to_string),
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

/// Read and parse a baseline report from disk.
fn load_baseline(baseline_path: &Path) -> Result<BenchmarkReport> {
    let baseline_data = std::fs::read_to_string(baseline_path)
        .with_context(|| format!("Failed to read baseline: {}", baseline_path.display()))?;

    serde_json::from_str(&baseline_data)
        .with_context(|| format!("Failed to parse baseline JSON: {}", baseline_path.display()))
}

/// Compare current results against a baseline report.
///
/// Results without a matching baseline target are omitted. A baseline mean of
/// zero (or a non-finite value from a corrupt file) yields `diff_pct: None`
/// instead of an infinite percentage.
fn compute_baseline_comparison(
    results: &[BenchmarkResult],
    baseline: &BenchmarkReport,
) -> Vec<BaselineComparison> {
    let mut comparison = Vec::new();

    for result in results {
        if let Some(base) = baseline.results.iter().find(|r| r.target == result.target) {
            let diff_pct = if base.mean_us > 0.0 && base.mean_us.is_finite() {
                Some(((result.mean_us - base.mean_us) / base.mean_us) * 100.0)
            } else {
                None
            };

            comparison.push(BaselineComparison {
                target: result.target.clone(),
                name: result.name.clone(),
                baseline_mean_us: base.mean_us,
                current_mean_us: result.mean_us,
                diff_pct,
            });
        }
    }

    comparison
}

/// Render a single comparison entry as a human-readable line (newline included).
fn format_comparison_line(entry: &BaselineComparison) -> String {
    match entry.diff_pct {
        Some(diff_pct) => {
            let indicator = if diff_pct < -5.0 {
                "[FASTER]"
            } else if diff_pct > 5.0 {
                "[SLOWER]"
            } else {
                "[~SAME]"
            };
            format!(
                "  {}: {:.2}us -> {:.2}us ({:+.1}%) {}\n",
                entry.name, entry.baseline_mean_us, entry.current_mean_us, diff_pct, indicator
            )
        }
        None => format!(
            "  {}: {:.2}us -> {:.2}us (baseline mean is zero — no ratio) [N/A]\n",
            entry.name, entry.baseline_mean_us, entry.current_mean_us
        ),
    }
}

/// Print the baseline comparison to stdout.
fn print_baseline_comparison(comparison: &[BaselineComparison]) {
    println!();
    output::header("Baseline Comparison");

    if comparison.is_empty() {
        println!("  (no targets in common with the baseline)");
        return;
    }

    for entry in comparison {
        print!("{}", format_comparison_line(entry));
    }
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
        write_report_file(&output, path)?;
        output::path_value("Report saved to", path);
    } else {
        println!("{}", output);
    }

    Ok(())
}

/// Write a rendered report to `path`, creating parent directories as needed.
///
/// Prints nothing — callers decide what to report, so that JSON mode keeps a
/// clean stdout.
fn write_report_file(rendered: &str, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory: {}", parent.display()))?;
    }

    let mut file = std::fs::File::create(path)
        .with_context(|| format!("Failed to create output file: {}", path.display()))?;
    file.write_all(rendered.as_bytes())
        .with_context(|| format!("Failed to write output file: {}", path.display()))?;

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

    if report.results.is_empty() {
        out.push_str("\n  (no benchmark target could be measured)\n");
    }

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
            "    Model Size: {} {}\n",
            result.model_size,
            model_size_unit(&result.target)
        ));
    }

    if !report.skipped.is_empty() {
        out.push('\n');
        out.push_str("Skipped Targets (not measured — no numbers reported):\n");
        for skipped in &report.skipped {
            out.push_str(&format!("  {}: {}\n", skipped.target, skipped.reason));
        }
    }

    if let Some(ref comparison) = report.baseline_comparison {
        out.push('\n');
        out.push_str("Baseline Comparison:\n");
        if comparison.is_empty() {
            out.push_str("  (no targets in common with the baseline)\n");
        }
        for entry in comparison {
            out.push_str(&format_comparison_line(entry));
        }
    }

    out.push('\n');
    out.push_str("=".repeat(60).as_str());
    out.push('\n');

    out
}

/// Unit label for [`BenchmarkResult::model_size`] of a given target.
fn model_size_unit(target: &str) -> &'static str {
    if target == "flame" {
        "FLAME vertices"
    } else {
        "Gaussians"
    }
}

/// Format report as JSON.
fn format_json(report: &BenchmarkReport) -> Result<String> {
    serde_json::to_string_pretty(report).context("Failed to serialize to JSON")
}

/// Format report as CSV.
///
/// CSV is a flat per-result table: skipped targets and the baseline comparison
/// are not representable and are therefore omitted (use JSON, Markdown, or the
/// human format for those).
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
            "| {} | {:.2} | {:.2} | {} | {} | {:.2} | {} {} |\n",
            result.name,
            result.mean_us,
            result.std_us,
            result.min_us,
            result.max_us,
            result.ops_per_sec,
            result.model_size,
            model_size_unit(&result.target),
        ));
    }

    if !report.skipped.is_empty() {
        out.push_str("\n## Skipped Targets\n\n");
        out.push_str("These targets were **not measured**; no numbers are reported for them.\n\n");
        out.push_str("| Target | Reason |\n");
        out.push_str("|--------|--------|\n");
        for skipped in &report.skipped {
            out.push_str(&format!("| {} | {} |\n", skipped.target, skipped.reason));
        }
    }

    if let Some(ref comparison) = report.baseline_comparison {
        out.push_str("\n## Baseline Comparison\n\n");
        if comparison.is_empty() {
            out.push_str("_No targets in common with the baseline._\n");
        } else {
            out.push_str("| Benchmark | Baseline (us) | Current (us) | Change |\n");
            out.push_str("|-----------|---------------|--------------|--------|\n");
            for entry in comparison {
                let change = match entry.diff_pct {
                    Some(diff_pct) => format!("{diff_pct:+.1}%"),
                    None => "n/a".to_string(),
                };
                out.push_str(&format!(
                    "| {} | {:.2} | {:.2} | {} |\n",
                    entry.name, entry.baseline_mean_us, entry.current_mean_us, change
                ));
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::BenchSize;

    fn test_args(iterations: u32, warmup: u32) -> BenchmarkArgs {
        BenchmarkArgs {
            target: BenchTarget::Export,
            warmup,
            iterations,
            format: OutputFormat::Human,
            output: None,
            size: BenchSize::Tiny,
            flame_model: None,
            baseline: None,
        }
    }

    fn sample_result(target: &str, mean_us: f64) -> BenchmarkResult {
        BenchmarkResult {
            name: format!("{target} bench"),
            target: target.to_string(),
            iterations: 10,
            total_us: (mean_us * 10.0) as u64,
            mean_us,
            std_us: 1.0,
            min_us: 1,
            max_us: 2,
            ops_per_sec: 0.0,
            model_size: 100,
            metadata: BenchmarkMetadata::default(),
        }
    }

    fn sample_report(results: Vec<BenchmarkResult>) -> BenchmarkReport {
        BenchmarkReport {
            version: "1.0".to_string(),
            results,
            system: SystemInfo {
                os: "test".to_string(),
                cpu: "test".to_string(),
                memory_mb: 8192,
                gpu: None,
                rust_version: "1.70".to_string(),
            },
            skipped: Vec::new(),
            baseline_comparison: None,
        }
    }

    #[test]
    fn test_build_result() {
        let times = vec![
            Duration::from_micros(100),
            Duration::from_micros(110),
            Duration::from_micros(90),
        ];
        let result = build_result("Test", "test", &times, 1000, None);

        assert_eq!(result.name, "Test");
        assert_eq!(result.iterations, 3);
        assert!((result.mean_us - 100.0).abs() < 1.0);
        assert_eq!(result.min_us, 90);
        assert_eq!(result.max_us, 110);
    }

    /// Regression: `--iterations 0` used to produce `NaN` statistics, which
    /// serde_json refuses to serialize.
    #[test]
    fn test_build_result_empty_times_has_no_nan() {
        let result = build_result("Empty", "test", &[], 42, None);

        assert_eq!(result.iterations, 0);
        assert!(result.mean_us.is_finite(), "mean must not be NaN");
        assert!(result.std_us.is_finite(), "std must not be NaN");
        assert_eq!(result.mean_us, 0.0);
        assert_eq!(result.std_us, 0.0);
        assert_eq!(result.ops_per_sec, 0.0);

        // Must remain serializable (serde_json rejects non-finite floats).
        let json = serde_json::to_string(&result);
        assert!(json.is_ok(), "empty result must serialize");
    }

    #[test]
    fn test_build_result_records_gpu_adapter() {
        let times = vec![Duration::from_micros(10)];
        let result = build_result("Test", "raster", &times, 1, Some("Test Adapter"));
        assert_eq!(result.metadata.gpu_adapter.as_deref(), Some("Test Adapter"));
    }

    /// Regression: `--iterations 0` must be rejected up front.
    #[test]
    fn test_validate_args_rejects_zero_iterations() {
        let e = validate_args(&test_args(0, 3)).expect_err("iterations = 0 must be rejected");
        let message = format!("{e:#}");
        assert!(
            message.contains("--iterations"),
            "error must mention the offending flag, got: {message}"
        );
    }

    #[test]
    fn test_validate_args_accepts_zero_warmup() {
        assert!(validate_args(&test_args(1, 0)).is_ok());
    }

    #[test]
    fn test_compute_baseline_comparison_matches_by_target() {
        let baseline = sample_report(vec![sample_result("raster", 100.0)]);
        let current = vec![sample_result("raster", 150.0)];

        let comparison = compute_baseline_comparison(&current, &baseline);
        assert_eq!(comparison.len(), 1);
        assert_eq!(comparison[0].target, "raster");
        let diff = comparison[0].diff_pct.expect("expected a diff percentage");
        assert!((diff - 50.0).abs() < 1e-6, "unexpected diff: {diff}");
    }

    /// Regression: a zero baseline mean used to produce an infinite percentage.
    #[test]
    fn test_compute_baseline_comparison_zero_mean_guard() {
        let baseline = sample_report(vec![sample_result("train", 0.0)]);
        let current = vec![sample_result("train", 25.0)];

        let comparison = compute_baseline_comparison(&current, &baseline);
        assert_eq!(comparison.len(), 1);
        assert!(
            comparison[0].diff_pct.is_none(),
            "zero baseline mean must not yield a ratio"
        );

        // The rendered line must not contain `inf`.
        let line = format_comparison_line(&comparison[0]);
        assert!(!line.contains("inf"), "unexpected infinity in: {line}");
    }

    #[test]
    fn test_compute_baseline_comparison_ignores_unknown_targets() {
        let baseline = sample_report(vec![sample_result("flame", 10.0)]);
        let current = vec![sample_result("export", 20.0)];
        assert!(compute_baseline_comparison(&current, &baseline).is_empty());
    }

    /// Baseline files written before `skipped` / `baseline_comparison` existed
    /// must still deserialize.
    #[test]
    fn test_legacy_baseline_deserializes() {
        let legacy = r#"{
            "version": "1.0",
            "results": [],
            "system": {
                "os": "linux x86_64",
                "cpu": "test",
                "memory_mb": 1024,
                "gpu": null,
                "rust_version": "1.70"
            }
        }"#;

        let parsed: std::result::Result<BenchmarkReport, _> = serde_json::from_str(legacy);
        let report = parsed.expect("legacy baseline must parse");
        assert!(report.skipped.is_empty());
        assert!(report.baseline_comparison.is_none());
    }

    #[test]
    fn test_report_round_trip_through_file() {
        let mut report = sample_report(vec![sample_result("export", 5.0)]);
        report.skipped.push(SkippedTarget {
            target: "raster".to_string(),
            reason: "no GPU".to_string(),
        });

        let dir = std::env::temp_dir().join(format!("oxigaf_bench_report_{}", unique_suffix()));
        let path = dir.join("report.json");

        let rendered = format_json(&report).expect("format_json failed");
        write_report_file(&rendered, &path).expect("write_report_file failed");

        let loaded = load_baseline(&path).expect("load_baseline failed");
        assert_eq!(loaded.results.len(), 1);
        assert_eq!(loaded.skipped.len(), 1);
        assert_eq!(loaded.results[0].target, "export");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_synthetic_gaussian_model_is_well_formed() {
        let model = synthetic_gaussian_model(8, 3);
        assert_eq!(model.len(), 8);
        assert_eq!(model.sh_degree, 3);
        // (3+1)^2 * 3 = 48 coefficients per Gaussian
        assert_eq!(model.sh_coeffs.len(), 8 * 48);
        assert_eq!(model.face_indices.len(), 8);
        assert_eq!(model.barycentric.len(), 8);
        assert_eq!(model.local_offsets.len(), 8);
        assert_eq!(model.is_rigid.len(), 8);
        assert!(model
            .gaussians
            .iter()
            .all(|g| g.position.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn test_synthetic_gaussian_model_is_deterministic() {
        let a = synthetic_gaussian_model(16, 0);
        let b = synthetic_gaussian_model(16, 0);
        assert_eq!(a.sh_coeffs, b.sh_coeffs);
        for (ga, gb) in a.gaussians.iter().zip(b.gaussians.iter()) {
            assert_eq!(ga.position, gb.position);
        }
    }

    #[test]
    fn test_benchmark_camera_matrices_are_finite() {
        let camera = benchmark_camera(512, 512);
        assert!(camera.view_matrix.iter().all(|v| v.is_finite()));
        assert!(camera.proj_matrix.iter().all(|v| v.is_finite()));
        assert!(camera.focal[0] > 0.0 && camera.focal[1] > 0.0);
        // Right-handed look-at from +Z: the camera position is preserved.
        assert!((camera.position[2] - 4.0).abs() < 1e-6);
    }

    /// The export target must exercise the real PLY writer, not a synthetic
    /// byte buffer: the produced file has to be a parseable PLY header with the
    /// expected vertex count.
    #[test]
    fn test_export_path_writes_real_ply() {
        let model = synthetic_gaussian_model(4, 0);
        let dir = std::env::temp_dir().join(format!("oxigaf_bench_ply_{}", unique_suffix()));
        let path = dir.join("bench.ply");

        crate::export::export_ply(&model, &path).expect("export_ply failed");

        let content = std::fs::read_to_string(&path).expect("failed to read exported PLY");
        assert!(content.starts_with("ply"), "not a PLY file");
        assert!(content.contains("element vertex 4"), "wrong vertex count");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// End-to-end check of the export benchmark: it must report exactly the
    /// requested number of timed iterations and finite statistics.
    #[test]
    fn test_benchmark_export_measures_real_work() {
        let args = test_args(2, 1);
        let results = benchmark_export(&args, None).expect("benchmark_export failed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].target, "export");
        assert_eq!(results[0].iterations, 2);
        assert!(results[0].mean_us.is_finite());
    }

    /// A missing `--flame-model` must be an explicit error, never silently
    /// simulated numbers.
    #[test]
    fn test_benchmark_flame_requires_model() {
        let args = test_args(1, 0);
        let e = benchmark_flame(&args, None)
            .expect_err("flame benchmark must not run without --flame-model");
        let message = format!("{e:#}");
        assert!(
            message.contains("--flame-model"),
            "error must name the missing flag, got: {message}"
        );
    }

    #[test]
    fn test_get_benchmark_targets_full_covers_all() {
        let targets = get_benchmark_targets(&BenchTarget::Full);
        assert_eq!(targets, vec!["flame", "raster", "train", "export"]);
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
        let report = sample_report(vec![BenchmarkResult {
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
        }]);

        let csv = format_csv(&report);
        assert!(csv.contains("Test,test,10"));
    }

    #[test]
    fn test_format_human_lists_skipped_targets() {
        let mut report = sample_report(Vec::new());
        report.skipped.push(SkippedTarget {
            target: "raster".to_string(),
            reason: "No suitable GPU adapter found".to_string(),
        });

        let human = format_human(&report);
        assert!(human.contains("Skipped Targets"));
        assert!(human.contains("No suitable GPU adapter found"));
        assert!(human.contains("no benchmark target could be measured"));
    }

    #[test]
    fn test_format_markdown_includes_comparison() {
        let mut report = sample_report(vec![sample_result("export", 20.0)]);
        report.baseline_comparison = Some(vec![BaselineComparison {
            target: "export".to_string(),
            name: "PLY Export".to_string(),
            baseline_mean_us: 10.0,
            current_mean_us: 20.0,
            diff_pct: Some(100.0),
        }]);

        let markdown = format_markdown(&report);
        assert!(markdown.contains("## Baseline Comparison"));
        assert!(markdown.contains("+100.0%"));
    }
}
