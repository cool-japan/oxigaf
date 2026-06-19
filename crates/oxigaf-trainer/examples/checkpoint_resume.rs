//! # Checkpoint Resume Example
//!
//! This example demonstrates checkpoint resume functionality for training:
//!
//! 1. **Initial Training Phase**: Train from scratch, save checkpoints periodically
//! 2. **Simulated Interruption**: Detect existing checkpoint
//! 3. **Resume Phase**: Load checkpoint, validate state, continue training seamlessly
//! 4. **Verification**: Show metrics history preserved, iteration counter continues
//!
//! ## Key Features
//!
//! - Periodic checkpoint saving during training
//! - Automatic checkpoint detection and resume
//! - Metrics history preservation across sessions
//! - Iteration counter continuity
//! - Optimizer state preservation (Adam momentum)
//!
//! ## Running
//!
//! First run (initial training):
//! ```bash
//! cargo run --example checkpoint_resume -- --iterations-per-phase 50
//! ```
//!
//! Second run (resume from checkpoint):
//! ```bash
//! cargo run --example checkpoint_resume -- --iterations-per-phase 50
//! ```
//!
//! ## Arguments
//!
//! - `--checkpoint-dir`: Directory for checkpoints (default: temp_dir/oxigaf_checkpoint_example)
//! - `--iterations-per-phase`: Iterations to run (default: 50)
//! - `--checkpoint-interval`: Save checkpoint every N iterations (default: 10)
//! - `--seed`: Random seed for reproducibility (default: 424242)
//! - `--clean`: Remove existing checkpoints before starting (default: false)

use std::path::{Path, PathBuf};

use clap::Parser;
use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};
use oxigaf_render::RasterConfig;
use oxigaf_trainer::config::TrainingConfig;
use oxigaf_trainer::{Trainer, TrainerError};

#[derive(Parser)]
#[command(name = "checkpoint_resume")]
#[command(about = "Demonstrate checkpoint resume functionality")]
struct Args {
    /// Directory for checkpoints
    #[arg(long)]
    checkpoint_dir: Option<PathBuf>,

    /// Iterations to run per phase
    #[arg(long, default_value = "50")]
    iterations_per_phase: u32,

    /// Save checkpoint every N iterations
    #[arg(long, default_value = "10")]
    checkpoint_interval: u32,

    /// Random seed for reproducibility
    #[arg(long, default_value = "424242")]
    seed: u64,

    /// Remove existing checkpoints before starting
    #[arg(long)]
    clean: bool,
}

/// Create a synthetic Gaussian model for demonstration.
fn create_demo_model(num_gaussians: usize, sh_degree: u32) -> GaussianModel {
    let mut gaussians = Vec::with_capacity(num_gaussians);
    let sh_per = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
    let mut sh_coeffs = Vec::with_capacity(num_gaussians * sh_per);

    // Simple LCG random number generator for demo
    let mut seed = 42u64;
    let mut random_f32 = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };

    for _ in 0..num_gaussians {
        // Random position in a sphere
        let x = random_f32() * 0.2;
        let y = random_f32() * 0.2;
        let z = random_f32() * 0.2;

        // Random rotation
        let rx = random_f32() * 0.1;
        let ry = random_f32() * 0.1;
        let rz = random_f32() * 0.1;
        let rw = 1.0;

        gaussians.push(GaussianAttributes {
            position: [x, y, z],
            _pad0: 0.0,
            rotation: [rx, ry, rz, rw],
            scale: [-5.0, -5.0, -5.0],
            opacity: -2.0,
        });

        for _ in 0..sh_per {
            sh_coeffs.push(random_f32() * 0.5);
        }
    }

    GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree,
        face_indices: vec![0; num_gaussians],
        barycentric: vec![[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]; num_gaussians],
        local_offsets: vec![[0.0, 0.0, 0.0]; num_gaussians],
        is_rigid: vec![true; num_gaussians],
    }
}

fn main() {
    println!("OxiGAF Checkpoint Resume Example");
    println!("=================================");
    println!();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    // Run the async training code
    match pollster::block_on(run_checkpoint_demo()) {
        Ok(()) => {
            println!();
            println!("Example completed successfully!");
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

async fn run_checkpoint_demo() -> Result<(), TrainerError> {
    let args = Args::parse();

    // Setup checkpoint directory
    let checkpoint_dir = args
        .checkpoint_dir
        .unwrap_or_else(|| std::env::temp_dir().join("oxigaf_checkpoint_example"));

    println!("Configuration:");
    println!("  Checkpoint dir: {}", checkpoint_dir.display());
    println!("  Iterations: {}", args.iterations_per_phase);
    println!("  Checkpoint interval: {}", args.checkpoint_interval);
    println!("  Seed: {}", args.seed);
    println!();

    // Clean existing checkpoints if requested
    if args.clean && checkpoint_dir.exists() {
        println!("Cleaning existing checkpoints...");
        std::fs::remove_dir_all(&checkpoint_dir).map_err(TrainerError::Io)?;
    }

    // Create checkpoint directory
    std::fs::create_dir_all(&checkpoint_dir).map_err(TrainerError::Io)?;

    let checkpoint_path = checkpoint_dir.join("training.json");

    // =========================================================================
    // Check if we're resuming or starting fresh
    // =========================================================================

    let is_resuming = checkpoint_path.exists();

    if is_resuming {
        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║   RESUMING FROM CHECKPOINT                                ║");
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!();
        println!("Found existing checkpoint: {}", checkpoint_path.display());
        println!();

        resume_training(&checkpoint_path, args.iterations_per_phase, args.seed).await?;
    } else {
        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║   STARTING FRESH TRAINING                                 ║");
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!();
        println!("No checkpoint found, starting from scratch...");
        println!();

        initial_training(
            &checkpoint_path,
            args.iterations_per_phase,
            args.checkpoint_interval,
            args.seed,
        )
        .await?;
    }

    // =========================================================================
    // Summary
    // =========================================================================

    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════════");
    println!();
    println!("Next steps:");
    if is_resuming {
        println!("  - Training continued from checkpoint");
        println!("  - Run again to continue further");
        println!("  - Use --clean to start fresh");
    } else {
        println!("  - Initial training completed and checkpoint saved");
        println!("  - Run again to resume from checkpoint");
        println!("  - Use --clean to restart from scratch");
    }
    println!();
    println!("Checkpoint location: {}", checkpoint_path.display());

    Ok(())
}

/// Initial training phase - train from scratch and save checkpoints.
async fn initial_training(
    checkpoint_path: &Path,
    total_iterations: u32,
    checkpoint_interval: u32,
    seed: u64,
) -> Result<(), TrainerError> {
    println!("Setting up trainer...");

    // Configure training
    let training_config = TrainingConfig {
        total_iterations,
        views_per_step: 2,
        checkpoint_interval,
        log_interval: 10,
        density_control_interval: 0, // Disable for determinism
        opacity_reset_interval: 0,   // Disable for determinism
        ..Default::default()
    };

    // Create model
    let model = create_demo_model(200, 0);
    println!("  Model: {} Gaussians", model.len());

    // Setup GPU
    let (device, queue) = setup_gpu().await?;

    // Create rasterizer config
    let raster_config = RasterConfig::new()
        .with_resolution(128, 128)
        .with_sh_degree(0)
        .with_background([1.0, 1.0, 1.0]);

    // Create trainer
    let mut trainer = Trainer::new(
        training_config.clone(),
        model,
        raster_config,
        device,
        queue,
        seed,
    )?;

    println!();
    println!("Starting initial training...");
    println!();

    // Training loop
    for _ in 0..total_iterations {
        let output = trainer.train_step()?;

        // Log progress
        if output.iteration % training_config.log_interval == 0 {
            println!(
                "  Iteration {:4} | Loss: {:.6} | Gaussians: {:4}",
                output.iteration, output.loss.total, output.num_gaussians,
            );
        }

        // Save checkpoint
        if output.iteration % checkpoint_interval == 0 {
            trainer.save_checkpoint(checkpoint_path)?;
            println!("  -> Checkpoint saved at iteration {}", output.iteration);
        }
    }

    // Save final checkpoint
    trainer.save_checkpoint(checkpoint_path)?;
    println!();
    println!("Initial training complete!");
    println!("  Final iteration: {}", trainer.iteration);
    println!("  Total Gaussians: {}", trainer.model.len());
    println!("  Metrics tracked: {}", trainer.metric_tracker.len());

    Ok(())
}

/// Resume training from checkpoint.
async fn resume_training(
    checkpoint_path: &Path,
    additional_iterations: u32,
    seed: u64,
) -> Result<(), TrainerError> {
    // Load checkpoint first to inspect state
    let checkpoint = oxigaf_trainer::checkpoint::load_checkpoint(checkpoint_path)?;

    println!("Checkpoint details:");
    println!("  Version: {}", checkpoint.version);
    println!("  Last iteration: {}", checkpoint.iteration);
    println!("  Gaussians: {}", checkpoint.positions.len());
    println!("  SH degree: {}", checkpoint.sh_degree);
    println!("  Optimizer groups: {}", checkpoint.optimizer_groups.len());
    println!("  Metrics history: {}", checkpoint.metrics_history.len());
    println!();

    // Validate checkpoint
    checkpoint.validate()?;
    println!("✓ Checkpoint validation passed");
    println!();

    // Configure training for resume
    let training_config = TrainingConfig {
        total_iterations: checkpoint.iteration + additional_iterations,
        views_per_step: 2,
        checkpoint_interval: 10,
        log_interval: 10,
        density_control_interval: 0,
        opacity_reset_interval: 0,
        ..Default::default()
    };

    // Setup GPU
    let (device, queue) = setup_gpu().await?;

    // Create rasterizer config
    let raster_config = RasterConfig::new()
        .with_resolution(128, 128)
        .with_sh_degree(checkpoint.sh_degree)
        .with_background([1.0, 1.0, 1.0]);

    // Resume trainer from checkpoint
    println!("Restoring trainer from checkpoint...");
    let mut trainer = Trainer::from_checkpoint(
        training_config.clone(),
        checkpoint_path,
        raster_config,
        device,
        queue,
        seed,
    )?;

    println!("✓ Trainer restored successfully");
    println!("  Resuming at iteration: {}", trainer.iteration);
    println!("  Model: {} Gaussians", trainer.model.len());
    println!("  Metrics history: {}", trainer.metric_tracker.len());
    println!();

    // Verify metrics history was preserved
    if let Some(latest_metric) = trainer.metric_tracker.latest() {
        println!("Latest metric before resume:");
        println!("  Iteration: {}", latest_metric.iteration);
        println!("  PSNR: {:.2} dB", latest_metric.psnr);
        println!("  SSIM: {:.4}", latest_metric.ssim);
        println!("  Loss: {:.6}", latest_metric.loss);
    }
    println!();

    println!("Continuing training...");
    println!();

    let start_iteration = trainer.iteration;

    // Continue training
    while trainer.iteration < training_config.total_iterations {
        let output = trainer.train_step()?;

        // Log progress
        if output.iteration % training_config.log_interval == 0 {
            println!(
                "  Iteration {:4} | Loss: {:.6} | Gaussians: {:4}",
                output.iteration, output.loss.total, output.num_gaussians,
            );
        }

        // Save checkpoint
        if output.iteration % training_config.checkpoint_interval == 0 {
            trainer.save_checkpoint(checkpoint_path)?;
            println!("  -> Checkpoint saved at iteration {}", output.iteration);
        }
    }

    // Save final checkpoint
    trainer.save_checkpoint(checkpoint_path)?;

    println!();
    println!("Resumed training complete!");
    println!("  Resumed from: iteration {}", start_iteration);
    println!("  Final iteration: {}", trainer.iteration);
    println!(
        "  Additional iterations: {}",
        trainer.iteration - start_iteration
    );
    println!("  Total Gaussians: {}", trainer.model.len());
    println!("  Total metrics tracked: {}", trainer.metric_tracker.len());

    Ok(())
}

/// Setup GPU device and queue.
async fn setup_gpu() -> Result<(wgpu::Device, wgpu::Queue), TrainerError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| {
            TrainerError::Render(oxigaf_render::RenderError::GpuInit(format!(
                "Failed to find GPU adapter: {}",
                e
            )))
        })?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("oxigaf_checkpoint_resume"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| {
            TrainerError::Render(oxigaf_render::RenderError::DeviceCreationFailed(format!(
                "Failed to create device: {}",
                e
            )))
        })?;

    Ok((device, queue))
}
