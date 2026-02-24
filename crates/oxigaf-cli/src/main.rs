//! OxiGAF CLI — Gaussian Avatar Reconstruction from monocular video.
//!
//! Subcommands:
//! * `train` (alias `reconstruct`) — end-to-end avatar reconstruction pipeline
//! * `render` — render an existing avatar from novel viewpoints
//! * `export` — export an avatar to PLY, glTF, or safetensors
//! * `convert` — convert FLAME model files (.pkl to .npy format)
//! * `benchmark` — run performance benchmarks
//! * `doctor` — check system configuration and dependencies
//! * `setup` — download and cache required model weights
//! * `cache` — manage cached assets (list, clean, verify, path)
//! * `completions` — generate shell completion scripts

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![warn(clippy::panic)]

mod assets;
mod benchmark;
mod cache;
mod cli;
mod config;
mod convert;
mod dry_run;
mod error;
mod export;
mod interactive;
mod json_output;
mod log_rotation;
mod metrics;
mod output;
mod pipeline;
mod progress;
mod summary;
mod verbosity;

// Public exports for testing
pub use interactive::InteractiveController;

use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{CommandFactory, Parser};
use clap_complete::generate;
use indicatif::{ProgressBar, ProgressStyle};

use cli::{CacheCommands, Cli, Command, ExportFormat, ImageFormat, RenderMode};
use error::{CliError, EXIT_SUCCESS};
use verbosity::Verbosity;

// ---------------------------------------------------------------------------
// Logging initialization
// ---------------------------------------------------------------------------

/// Initialize tracing subscriber with the specified verbosity level and optional file logging.
///
/// Configures logging output format and filtering based on verbosity.
/// Higher verbosity levels include file names and line numbers.
/// If log file is specified, logs are written to both file and console.
fn init_logging(
    verbosity: Verbosity,
    log_file: Option<std::path::PathBuf>,
    log_rotation: cli::LogRotationStrategy,
    log_max_files: usize,
    log_format: cli::LogFormatType,
) -> Result<()> {
    use log_rotation::{LogConfig, LogFormat, LogRotation};

    let log_config = LogConfig {
        file_path: log_file.clone(),
        rotation: match log_rotation {
            cli::LogRotationStrategy::Never => LogRotation::Never,
            cli::LogRotationStrategy::Hourly => LogRotation::Hourly,
            cli::LogRotationStrategy::Daily => LogRotation::Daily,
        },
        max_files: log_max_files,
        format: match log_format {
            cli::LogFormatType::Json => LogFormat::Json,
            cli::LogFormatType::Pretty => LogFormat::Pretty,
            cli::LogFormatType::Compact => LogFormat::Compact,
        },
    };

    log_rotation::init_logging_with_file(log_config, verbosity)?;

    // Clean up old logs if file logging is enabled
    if let Some(ref log_path) = log_file {
        if let Some(parent) = log_path.parent() {
            if let Some(prefix) = log_path.file_stem().and_then(|s| s.to_str()) {
                // Ignore cleanup errors (non-critical)
                let _ = log_rotation::cleanup_old_logs(parent, prefix, log_max_files);
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> ExitCode {
    // Parse CLI args first to get verbosity level
    let cli = Cli::parse();
    let verbosity = cli.verbosity();
    let dry_run = cli.dry_run;
    let json_mode = cli.json;

    // Initialize logging based on verbosity (skip in JSON mode)
    if !json_mode {
        if let Err(e) = init_logging(
            verbosity,
            cli.log_file.clone(),
            cli.log_rotation,
            cli.log_max_files,
            cli.log_format,
        ) {
            eprintln!("Failed to initialize logging: {}", e);
            return ExitCode::from(1);
        }
    }

    let result = match cli.command {
        Command::Train(args) => cmd_train(args, verbosity, dry_run, json_mode),
        Command::Render(args) => cmd_render(args, verbosity, json_mode),
        Command::Export(args) => cmd_export(args, verbosity, dry_run, json_mode),
        Command::Convert(args) => cmd_convert(args, verbosity, dry_run, json_mode),
        Command::Benchmark(args) => cmd_benchmark(args, verbosity, json_mode),
        Command::Doctor(args) => cmd_doctor(args, verbosity, json_mode),
        Command::Setup(args) => cmd_setup(args, verbosity, json_mode),
        Command::Cache { command } => cmd_cache(command, verbosity, json_mode),
        Command::Completions { shell } => cmd_completions(shell),
    };

    match result {
        Ok(()) => ExitCode::from(EXIT_SUCCESS as u8),
        Err(err) => {
            // Convert anyhow::Error to CliError for proper exit code
            let cli_err: CliError = err.into();

            // Output error based on mode
            if json_mode {
                let output = json_output::JsonOutput::error("command", format!("{:#}", cli_err));
                output.print();
            } else {
                output::display_error(&cli_err);
                output::flush();
            }

            ExitCode::from(cli_err.exit_code() as u8)
        }
    }
}

// ---------------------------------------------------------------------------
// train (alias: reconstruct)
// ---------------------------------------------------------------------------

fn cmd_train(
    args: cli::TrainArgs,
    verbosity: Verbosity,
    dry_run: bool,
    json_mode: bool,
) -> Result<()> {
    let start = Instant::now();
    tracing::info!(?args.input, ?args.output, "Starting training pipeline");

    // Create interactive controller if requested
    let controller = if args.interactive {
        let ctrl = interactive::InteractiveController::new();
        ctrl.start_keyboard_listener();
        Some(ctrl)
    } else {
        None
    };

    // Dry-run validation
    if dry_run {
        let mut report = dry_run::DryRunReport::new();

        // Validate inputs
        if !args.input.exists() {
            anyhow::bail!("Input not found: {}", args.input.display());
        }
        if !json_mode {
            output::success(&format!("Input validated: {}", args.input.display()));
        }

        if !args.flame_model.exists() {
            anyhow::bail!("FLAME model not found: {}", args.flame_model.display());
        }
        if !json_mode {
            output::success(&format!(
                "FLAME model validated: {}",
                args.flame_model.display()
            ));
        }

        // Check output directory
        dry_run::check_writable(&args.output)?;
        report.add_create(format!("{}/", args.output.display()));
        report.add_create(format!("{}/checkpoints/", args.output.display()));
        report.add_create(format!("{}/preview/", args.output.display()));
        report.add_create(format!("{}/final_model.ply", args.output.display()));

        // GPU check
        dry_run::check_gpu()?;

        // Estimate resources
        report.resource_estimates.estimated_duration_sec = Some(3600); // 1 hour
        report.resource_estimates.estimated_vram_mb = Some(4096); // 4GB
        report.resource_estimates.estimated_disk_mb = Some(500); // 500MB

        if !json_mode {
            report.print_report();
        }
        return Ok(());
    }

    // 1. Load project configuration with hierarchical loading
    // Priority: CLI args > env vars > project config > user config > defaults
    let mut project_config = config::load_hierarchical_config(Some(&args.config), None)?;
    tracing::info!("Configuration loaded with hierarchical priority");
    tracing::debug!(
        "Config sources checked: env vars, {}, ~/.config/oxigaf/config.toml, defaults",
        args.config.display()
    );

    // Apply CLI overrides (highest priority)
    if let Some(max_iter) = args.max_iterations {
        project_config.training.total_iterations = max_iter;
        tracing::debug!("CLI override: total_iterations = {}", max_iter);
    }
    if let Some(ckpt_int) = args.checkpoint_interval {
        project_config.output.checkpoint_interval = ckpt_int;
        tracing::debug!("CLI override: checkpoint_interval = {}", ckpt_int);
    }

    // 2. Ensure output directory exists
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("Failed to create output dir: {}", args.output.display()))?;

    // 3. Run the full pipeline
    let pipeline_cfg = pipeline::PipelineConfig {
        flame_model_path: args.flame_model.clone(),
        flame_params_path: args.flame_params.clone(),
        input_path: args.input.clone(),
        output_dir: args.output.clone(),
        resume_checkpoint: args.resume.clone(),
        device_index: args.device,
        project_config,
        patience: args.patience,
        min_delta: args.min_delta,
        metrics_output: args.metrics_output.clone(),
        metrics_format: args.metrics_format,
        tensorboard: args.tensorboard,
        tensorboard_dir: args.tensorboard_dir.clone(),
    };

    let result = pipeline::run_reconstruction(pipeline_cfg, verbosity, controller.as_ref())?;

    // 4. Export the final model as PLY
    let ply_path = args.output.join("final_model.ply");
    export::export_ply(&result.model, &ply_path)?;

    // 5. Render a preview turntable (unless disabled)
    if !args.no_preview {
        let preview_dir = args.output.join("preview");
        std::fs::create_dir_all(&preview_dir)?;
        let cameras = pipeline::default_orbit_cameras(512, 512);
        for (i, cam) in cameras.iter().enumerate() {
            let img = export::render_point_cloud(&result.model, cam);
            img.save(preview_dir.join(format!("view_{i:03}.png")))?;
        }
        tracing::info!(
            "Saved {} preview images to {}",
            cameras.len(),
            preview_dir.display(),
        );
    }

    let elapsed = start.elapsed();

    // Cleanup interactive mode
    if controller.is_some() {
        let _ = crossterm::terminal::disable_raw_mode();
    }

    // Output based on mode
    if json_mode {
        let mut output = json_output::JsonOutput::success(
            "train",
            serde_json::json!({
                "num_gaussians": result.model.len(),
                "elapsed_seconds": elapsed.as_secs_f64()
            }),
        );

        // Add PLY artifact
        if ply_path.exists() {
            output.add_artifact("ply".to_string(), ply_path.clone());
        }

        // Add preview directory as artifact if created
        if !args.no_preview {
            let preview_dir = args.output.join("preview");
            if preview_dir.exists() {
                output.add_artifact("preview".to_string(), preview_dir);
            }
        }

        output.print();
    } else {
        // Calculate throughput
        let throughput = result.total_iterations as f32 / elapsed.as_secs_f32();

        // Prepare paths for summary
        let checkpoint_path = args.output.join("checkpoints/final.json");
        let preview_dir = if !args.no_preview {
            Some(args.output.join("preview"))
        } else {
            None
        };

        let training_summary = summary::TrainingSummary {
            total_iterations: result.total_iterations,
            final_loss: result.final_loss,
            num_gaussians: result.model.len() as u32,
            num_rigid: result.num_rigid as u32,
            num_flexible: result.num_flexible as u32,
            sh_degree: result.model.sh_degree,
            elapsed,
            throughput_iters_per_sec: throughput,
            checkpoint_path: Some(checkpoint_path.display().to_string()),
            ply_path: Some(ply_path.display().to_string()),
            preview_dir: preview_dir.map(|p| p.display().to_string()),
            peak_memory_mb: None, // TODO: Track memory usage
        };

        training_summary.print();
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------

fn cmd_render(args: cli::RenderArgs, verbosity: Verbosity, json_mode: bool) -> Result<()> {
    let start = Instant::now();
    tracing::info!(?args.model, ?args.output, "Rendering avatar");

    // 1. Load model
    let model = export::load_model(&args.model)
        .with_context(|| format!("Failed to load model: {}", args.model.display()))?;
    tracing::info!(
        "Model loaded: {} Gaussians, SH degree {}",
        model.len(),
        model.sh_degree
    );

    // 2. Apply quality preset if width/height are default values
    let (width, height) = if args.width == 512 && args.height == 512 {
        let (w, h) = args.quality.resolution();
        tracing::info!("Quality preset {:?} → resolution {}x{}", args.quality, w, h);
        (w, h)
    } else {
        (args.width, args.height)
    };

    // 3. Prepare output directory
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("Failed to create output dir: {}", args.output.display()))?;

    // 4. Build cameras based on mode
    let cameras = match args.mode {
        RenderMode::Frames => {
            if let Some(ref cam_path) = args.cameras {
                let json = std::fs::read_to_string(cam_path).with_context(|| {
                    format!("Failed to read cameras file: {}", cam_path.display())
                })?;
                let specs: Vec<pipeline::CameraSpec> =
                    serde_json::from_str(&json).with_context(|| {
                        format!("Failed to parse cameras JSON: {}", cam_path.display())
                    })?;
                specs
                    .iter()
                    .map(|s| {
                        pipeline::orbit_camera(s.azimuth, s.elevation, s.distance, width, height)
                    })
                    .collect::<Vec<_>>()
            } else {
                pipeline::default_orbit_cameras(width, height)
            }
        }
        RenderMode::Turntable => {
            // 360-degree turntable
            let step = 360.0 / args.num_frames as f32;
            (0..args.num_frames)
                .map(|i| pipeline::orbit_camera(i as f32 * step, 10.0, 0.6, width, height))
                .collect()
        }
        RenderMode::Orbit => {
            // Orbit with elevation variation
            let step = 360.0 / args.num_frames as f32;
            (0..args.num_frames)
                .map(|i| {
                    let az = i as f32 * step;
                    let el = 10.0 + 20.0 * (az.to_radians().sin());
                    pipeline::orbit_camera(az, el, 0.6, width, height)
                })
                .collect()
        }
        RenderMode::Dolly => {
            // Dolly zoom in/out
            (0..args.num_frames)
                .map(|i| {
                    let t = i as f32 / args.num_frames.max(1) as f32;
                    let dist = 0.3 + 0.6 * (1.0 - t);
                    pipeline::orbit_camera(0.0, 10.0, dist, width, height)
                })
                .collect()
        }
    };

    if let Some(ref flame_params_path) = args.flame_params {
        tracing::warn!(
            "FLAME-driven animation (--flame-params {}) is not yet supported in the \
             software renderer. Rendering static Gaussian positions.",
            flame_params_path.display(),
        );
    }

    // 5. Render each view with progress
    let pb = if !json_mode && verbosity.show_progress() {
        let pb = ProgressBar::new(cameras.len() as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} views rendered ({eta})",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=>-"),
        );
        pb
    } else {
        ProgressBar::hidden()
    };

    // Determine output extension based on format
    let ext = match args.format {
        ImageFormat::Png => "png",
        ImageFormat::Jpeg => "jpg",
        ImageFormat::Exr => "exr",
    };

    // Collect rendered file paths for JSON output
    let mut rendered_files = Vec::new();

    for (i, cam) in cameras.iter().enumerate() {
        let img = export::render_point_cloud(&model, cam);
        let out_path = args.output.join(format!("view_{i:03}.{ext}"));

        match args.format {
            ImageFormat::Png => {
                img.save(&out_path)
                    .with_context(|| format!("Failed to save image: {}", out_path.display()))?;
            }
            ImageFormat::Jpeg => {
                // Convert to JPEG with quality setting
                let jpeg_quality = 90u8;
                let file = std::fs::File::create(&out_path)
                    .with_context(|| format!("Failed to create file: {}", out_path.display()))?;
                let mut encoder =
                    image::codecs::jpeg::JpegEncoder::new_with_quality(file, jpeg_quality);
                encoder
                    .encode_image(&img)
                    .with_context(|| format!("Failed to encode JPEG: {}", out_path.display()))?;
            }
            ImageFormat::Exr => {
                // Convert RGB8 to RGB32F for EXR
                use image::{ImageBuffer, Rgb32FImage};
                let width = img.width();
                let height = img.height();
                let exr_img: Rgb32FImage = ImageBuffer::from_fn(width, height, |x, y| {
                    let pixel = img.get_pixel(x, y);
                    image::Rgb([
                        pixel[0] as f32 / 255.0,
                        pixel[1] as f32 / 255.0,
                        pixel[2] as f32 / 255.0,
                    ])
                });
                exr_img
                    .save(&out_path)
                    .with_context(|| format!("Failed to save EXR image: {}", out_path.display()))?;
            }
        }
        rendered_files.push(out_path);
        pb.inc(1);
    }

    pb.finish_with_message("done");

    let elapsed = start.elapsed();

    // Output based on mode
    if json_mode {
        let mut output = json_output::JsonOutput::success(
            "render",
            serde_json::json!({
                "num_views": cameras.len(),
                "width": width,
                "height": height,
                "mode": format!("{:?}", args.mode),
                "format": format!("{:?}", args.format),
                "output_dir": args.output.display().to_string()
            }),
        );

        // Add all rendered images as artifacts
        for file_path in rendered_files {
            if file_path.exists() {
                output.add_artifact("image".to_string(), file_path);
            }
        }

        output.print();
    } else {
        // Calculate FPS
        let fps = if elapsed.as_secs_f32() > 0.0 {
            Some(cameras.len() as f32 / elapsed.as_secs_f32())
        } else {
            None
        };

        let render_summary = summary::RenderSummary {
            num_views: cameras.len() as u32,
            resolution: (width, height),
            format: format!("{:?}", args.format),
            mode: format!("{:?}", args.mode),
            elapsed,
            output_dir: args.output.display().to_string(),
            fps,
        };

        render_summary.print();
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// export
// ---------------------------------------------------------------------------

fn cmd_export(
    args: cli::ExportArgs,
    _verbosity: Verbosity,
    dry_run: bool,
    json_mode: bool,
) -> Result<()> {
    let start = Instant::now();
    tracing::info!(?args.model, ?args.output, ?args.format, "Exporting avatar");

    // Dry-run validation
    if dry_run {
        let mut report = dry_run::DryRunReport::new();

        // Validate input
        if !args.model.exists() {
            anyhow::bail!("Input model not found: {}", args.model.display());
        }
        if !json_mode {
            output::success(&format!("Input model validated: {}", args.model.display()));
        }

        // Check for overwrite protection
        if args.output.exists() && !args.force {
            anyhow::bail!(
                "Output file already exists: {}. Use --force to overwrite.",
                args.output.display()
            );
        }

        // Check output
        dry_run::check_writable(&args.output)?;

        if args.output.exists() {
            report.add_modify(args.output.display().to_string());
        } else {
            report.add_create(args.output.display().to_string());
        }

        // Estimate size based on format
        report.resource_estimates.estimated_disk_mb = Some(100);

        if !json_mode {
            report.print_report();
        }
        return Ok(());
    }

    // Check for overwrite protection
    if args.output.exists() && !args.force {
        anyhow::bail!(
            "Output file already exists: {}. Use --force to overwrite.",
            args.output.display()
        );
    }

    // 1. Load model
    let model = export::load_model(&args.model)
        .with_context(|| format!("Failed to load model: {}", args.model.display()))?;
    tracing::info!(
        "Loaded model: {} Gaussians, SH degree {}",
        model.len(),
        model.sh_degree,
    );

    // 2. Export
    let format_name = match args.format {
        ExportFormat::Ply => {
            export::export_ply(&model, &args.output)?;
            "PLY"
        }
        ExportFormat::Safetensors => {
            export::export_safetensors(&model, &args.output)?;
            "safetensors"
        }
        ExportFormat::Gltf => {
            export::export_gltf(&model, &args.output, args.include_metadata)?;
            if args.include_metadata {
                "glTF 2.0 (with metadata)"
            } else {
                "glTF 2.0"
            }
        }
        ExportFormat::Json => {
            export::export_json_checkpoint(&model, &args.output)?;
            "JSON checkpoint"
        }
    };

    let elapsed = start.elapsed();

    // Get file size in MB
    let file_size_mb = std::fs::metadata(&args.output)
        .ok()
        .map(|m| m.len() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0);

    // Output based on mode
    if json_mode {
        let mut output = json_output::JsonOutput::success(
            "export",
            serde_json::json!({
                "format": format_name,
                "num_gaussians": model.len(),
                "sh_degree": model.sh_degree,
                "output_file": args.output.display().to_string()
            }),
        );

        // Add the exported file as an artifact
        if args.output.exists() {
            output.add_artifact("export".to_string(), args.output.clone());
        }

        output.print();
    } else {
        let export_summary = summary::ExportSummary {
            format: format_name.to_string(),
            input_file: args.model.display().to_string(),
            output_file: args.output.display().to_string(),
            file_size_mb,
            num_gaussians: model.len() as u32,
            elapsed,
        };

        export_summary.print();
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// convert
// ---------------------------------------------------------------------------

fn cmd_convert(
    args: cli::ConvertArgs,
    verbosity: Verbosity,
    dry_run: bool,
    json_mode: bool,
) -> Result<()> {
    convert::run_convert(args, verbosity, dry_run, json_mode)
}

// ---------------------------------------------------------------------------
// benchmark
// ---------------------------------------------------------------------------

fn cmd_benchmark(args: cli::BenchmarkArgs, verbosity: Verbosity, json_mode: bool) -> Result<()> {
    benchmark::run_benchmark(args, verbosity, json_mode)
}

// ---------------------------------------------------------------------------
// doctor
// ---------------------------------------------------------------------------

fn cmd_doctor(args: cli::DoctorArgs, verbosity: Verbosity, json_mode: bool) -> Result<()> {
    use serde_json::json;

    if !json_mode {
        println!();
        output::info("OxiGAF System Diagnostics");
        output::separator();
        println!();
    }

    let mut all_ok = true;
    let mut diagnostics = serde_json::Map::new();

    // 1. Check wgpu/GPU availability
    if !json_mode {
        output::header("GPU Configuration");
    }
    let gpu_check = check_gpu();
    match &gpu_check {
        Ok(info) => {
            if !json_mode {
                output::success(&format!("GPU adapter found: {}", info));
            }
            diagnostics.insert(
                "gpu".to_string(),
                json!({ "status": "ok", "adapter": info }),
            );
        }
        Err(e) => {
            if !json_mode {
                output::error(&format!("GPU not available: {}", e));
            }
            diagnostics.insert(
                "gpu".to_string(),
                json!({ "status": "error", "error": e.to_string() }),
            );
            all_ok = false;
        }
    }

    // 2. Check FLAME model (if path provided)
    if let Some(ref flame_path) = args.flame_model {
        if !json_mode {
            output::header("FLAME Model");
        }
        let flame_check = check_flame_model(flame_path);
        match &flame_check {
            Ok(info) => {
                if !json_mode {
                    output::success(&format!("FLAME model valid: {}", info));
                }
                diagnostics.insert("flame".to_string(), json!({ "status": "ok", "info": info }));
            }
            Err(e) => {
                if !json_mode {
                    output::error(&format!("FLAME model invalid: {}", e));
                }
                diagnostics.insert(
                    "flame".to_string(),
                    json!({ "status": "error", "error": e.to_string() }),
                );
                all_ok = false;
            }
        }
    }

    // 3. Check cache directory
    let cache_dir = args
        .cache_dir
        .as_ref()
        .map(|p| config::expand_tilde(p))
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            std::path::PathBuf::from(home).join(".cache/oxigaf")
        });

    if !json_mode {
        output::header("Asset Cache");
    }
    let cache_check = check_cache(&cache_dir);
    match &cache_check {
        Ok(info) => {
            if !json_mode {
                output::success(&format!("Cache directory: {}", info));
            }
            diagnostics.insert("cache".to_string(), json!({ "status": "ok", "info": info }));
        }
        Err(e) => {
            if !json_mode {
                output::warning(&format!("Cache issue: {}", e));
            }
            diagnostics.insert(
                "cache".to_string(),
                json!({ "status": "warning", "warning": e.to_string() }),
            );
        }
    }

    // 4. Version information
    if !json_mode {
        output::header("Version Information");
    }
    let version_info = get_version_info();
    if !json_mode {
        output::value("OxiGAF", &version_info.oxigaf);
        output::value("Rust", &version_info.rust);
        output::value("Platform", &version_info.platform);
    }
    diagnostics.insert(
        "version".to_string(),
        json!({
            "oxigaf": version_info.oxigaf,
            "rust": version_info.rust,
            "platform": version_info.platform,
        }),
    );

    // 5. Disk space check (if verbose)
    if !json_mode && verbosity >= Verbosity::Verbose {
        output::header("Disk Space");
        let disk_check = check_disk_space(&cache_dir);
        match &disk_check {
            Ok(info) => {
                output::value("Available Space", info);
            }
            Err(e) => {
                output::warning(&format!("Could not check disk space: {}", e));
            }
        }
    }

    // Output based on mode
    if json_mode {
        let output = json_output::JsonOutput::success("doctor", json!(diagnostics));
        output.print();
    } else {
        println!();
        output::separator();
        if all_ok {
            output::success("All checks passed! System is ready.");
        } else {
            output::warning("Some checks failed. See above for details.");
        }
    }

    Ok(())
}

/// Check GPU availability via wgpu.
fn check_gpu() -> Result<String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| anyhow::anyhow!("No GPU adapter found: {e}"))?;

    let info = adapter.get_info();
    Ok(format!("{} ({:?})", info.name, info.backend))
}

/// Check FLAME model directory.
fn check_flame_model(path: &std::path::Path) -> Result<String> {
    let expanded = config::expand_tilde(path);
    if !expanded.exists() {
        anyhow::bail!("Directory does not exist");
    }

    let required_files = ["v_template.npy", "shapedirs.npy", "faces.npy"];
    let mut found = 0;
    for file in &required_files {
        if expanded.join(file).exists() {
            found += 1;
        }
    }

    if found == required_files.len() {
        Ok(format!(
            "{} (all {} required files present)",
            expanded.display(),
            found
        ))
    } else {
        anyhow::bail!("{} of {} required files found", found, required_files.len())
    }
}

/// Check cache directory status.
fn check_cache(path: &std::path::Path) -> Result<String> {
    if !path.exists() {
        return Err(anyhow::anyhow!(
            "Cache directory does not exist. Run `oxigaf setup` to create it."
        ));
    }

    let assets = assets::expected_asset_paths(path);
    let mut cached = 0;
    for asset_path in &assets {
        if asset_path.exists() {
            cached += 1;
        }
    }

    Ok(format!(
        "{} ({}/{} assets cached)",
        path.display(),
        cached,
        assets.len()
    ))
}

/// Version information.
struct VersionInfo {
    oxigaf: String,
    rust: String,
    platform: String,
}

/// Get version information.
fn get_version_info() -> VersionInfo {
    VersionInfo {
        oxigaf: env!("CARGO_PKG_VERSION").to_string(),
        rust: option_env!("RUSTC_VERSION")
            .map(|s| s.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        platform: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
    }
}

/// Check available disk space.
fn check_disk_space(path: &std::path::Path) -> Result<String> {
    // Simple check - just verify the path is writable
    let _ = path;

    // On Unix systems, we could use statvfs, but that requires libc
    // For now, we just verify write access as a basic check
    if path.exists() {
        let test_file = path.join(".oxigaf_write_test");
        match std::fs::write(&test_file, b"test") {
            Ok(()) => {
                let _ = std::fs::remove_file(&test_file);
                Ok("Writable (space check not implemented)".to_string())
            }
            Err(e) => Err(anyhow::anyhow!("Not writable: {}", e)),
        }
    } else {
        Ok("Directory does not exist".to_string())
    }
}

// ---------------------------------------------------------------------------
// setup
// ---------------------------------------------------------------------------

fn cmd_setup(args: cli::SetupArgs, verbosity: Verbosity, json_mode: bool) -> Result<()> {
    // Check if HuggingFace Hub download is requested
    if let Some(hub_spec) = args.from_hub {
        // Parse the HuggingFace model specification
        let mut source = assets::HfModelSource::parse(&hub_spec)
            .context("Failed to parse HuggingFace model specification")?;

        // Override revision if specified via CLI flag
        if let Some(rev) = args.revision {
            source.revision = Some(rev);
        }

        // Override filename if specified via CLI flag
        if let Some(filename) = args.filename {
            source = source.with_filename(filename);
        }

        // Get authentication token (from CLI arg, environment, or config file)
        let token = args.hf_token.or_else(assets::get_hf_token);

        // Download the model
        tracing::info!(?hub_spec, ?source.revision, "Downloading from HuggingFace Hub");
        let downloaded_path = assets::download_with_progress(
            &source.repo_id,
            &source.filename,
            source.revision.as_deref(),
            token.as_deref(),
            verbosity,
        )
        .context("Failed to download model from HuggingFace Hub")?;

        // Output based on mode
        if json_mode {
            let output = json_output::JsonOutput::success(
                "setup",
                serde_json::json!({
                    "source": "huggingface_hub",
                    "repo_id": source.repo_id,
                    "filename": source.filename,
                    "revision": source.revision,
                    "path": downloaded_path.display().to_string()
                }),
            );
            output.print();
        } else {
            output::success(&format!(
                "Model downloaded successfully from HuggingFace Hub\nPath: {}",
                downloaded_path.display()
            ));
        }
    } else {
        // Standard asset download from manifest
        tracing::info!(?args.cache_dir, "Setting up model assets");
        assets::setup_cache(&args.cache_dir, verbosity, json_mode)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// cache
// ---------------------------------------------------------------------------

/// Manage cached assets.
fn cmd_cache(command: CacheCommands, _verbosity: Verbosity, json_mode: bool) -> Result<()> {
    // Get cache directory
    let cache_dir = get_cache_dir()?;

    // JSON mode not yet supported for cache commands
    if json_mode {
        anyhow::bail!("JSON mode not yet supported for cache commands");
    }

    match command {
        CacheCommands::List => cache::list_cache(&cache_dir),
        CacheCommands::Clean {
            max_age_days,
            dry_run,
        } => cache::clean_cache(&cache_dir, max_age_days, dry_run),
        CacheCommands::Verify => cache::verify_cache(&cache_dir),
        CacheCommands::Path => {
            println!("{}", cache_dir.display());
            Ok(())
        }
    }
}

/// Get the cache directory path.
fn get_cache_dir() -> Result<std::path::PathBuf> {
    dirs::cache_dir()
        .map(|p| p.join("oxigaf"))
        .ok_or_else(|| anyhow::anyhow!("Could not determine cache directory"))
}

// ---------------------------------------------------------------------------
// completions
// ---------------------------------------------------------------------------

/// Generate shell completion scripts.
fn cmd_completions(shell: clap_complete::Shell) -> Result<()> {
    let mut cmd = Cli::command();
    let bin_name = cmd.get_name().to_string();
    generate(shell, &mut cmd, bin_name, &mut std::io::stdout());
    Ok(())
}
