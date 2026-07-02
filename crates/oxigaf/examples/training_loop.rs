//! # Training Loop Example
//!
//! This example demonstrates a minimal training loop for Gaussian avatar optimization:
//!
//! 1. Set up training configuration
//! 2. Initialize Gaussian model (synthetic data for demo)
//! 3. Create the Trainer with GPU rasterizer
//! 4. Run training steps with logging
//! 5. Save/load checkpoints
//!
//! ## Training Pipeline Overview
//!
//! The trainer orchestrates:
//! - Random camera sampling for multi-view supervision
//! - GPU rasterization to render current Gaussians
//! - Loss computation (L1, SSIM, regularization)
//! - Backward pass through GPU shaders
//! - Adam optimizer with per-parameter learning rates
//! - Adaptive density control (split/clone/prune)
//! - Checkpoint management
//!
//! ## Running
//!
//! ```bash
//! cargo run --example training_loop
//! ```
//!
//! Note: Requires a GPU. For production training, use the CLI tool.

use std::path::Path;

use oxigaf::prelude::*;
use oxigaf::render::gaussian::{GaussianAttributes, GaussianModel};

/// Create a synthetic Gaussian model for demonstration.
///
/// In production, Gaussians would be initialized on a FLAME mesh surface
/// using `oxigaf_trainer::init::initialize_gaussians()`.
fn create_demo_model(num_gaussians: usize, sh_degree: u32) -> GaussianModel {
    let mut gaussians = Vec::with_capacity(num_gaussians);
    let sh_per = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
    let mut sh_coeffs = Vec::with_capacity(num_gaussians * sh_per);

    // Simple random number generator using linear congruential method
    // Avoids external dependencies for this demo
    let mut seed = 42u64;
    let mut random_f32 = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32) / (u32::MAX as f32 / 2.0) - 1.0
    };

    for _ in 0..num_gaussians {
        // Random position in a sphere of radius 0.2
        let x = random_f32() * 0.2;
        let y = random_f32() * 0.2;
        let z = random_f32() * 0.2;

        // Random rotation (not normalized, but close enough for demo)
        let rx = random_f32() * 0.1;
        let ry = random_f32() * 0.1;
        let rz = random_f32() * 0.1;
        let rw = 1.0;

        gaussians.push(GaussianAttributes {
            position: [x, y, z],
            _pad0: 0.0,
            rotation: [rx, ry, rz, rw],
            scale: [-5.0, -5.0, -5.0], // Log-scale: exp(-5) ~ 0.007
            opacity: -2.0,             // Sigmoid(-2) ~ 0.12
        });

        // Random SH coefficients for view-dependent color
        for _ in 0..sh_per {
            sh_coeffs.push(random_f32() * 0.5);
        }
    }

    GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree,
        // Demo binding to face 0 with uniform barycentric
        face_indices: vec![0; num_gaussians],
        barycentric: vec![[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]; num_gaussians],
        local_offsets: vec![[0.0, 0.0, 0.0]; num_gaussians],
        is_rigid: vec![true; num_gaussians],
    }
}

/// Main entry point.
fn main() {
    println!("OxiGAF Training Loop Example");
    println!("============================");
    println!();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    // Run the async training code
    match pollster::block_on(run_training()) {
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

/// Async training function.
async fn run_training() -> oxigaf::Result<()> {
    // =========================================================================
    // Step 1: Configure training
    // =========================================================================

    println!("Configuring training...");

    // Create a minimal training config for this demo
    // In production, load from TOML/JSON file
    let training_config = TrainingConfig {
        // Run only 100 iterations for demo (production: 15000+)
        total_iterations: 100,
        // Render 2 views per step (production: 4-8)
        views_per_step: 2,
        // Density control settings
        density_control_interval: 25,
        density_control_start: 20,
        density_control_end: 80,
        // Opacity reset
        opacity_reset_interval: 50,
        // Checkpointing and logging
        checkpoint_interval: 50,
        log_interval: 10,
        // Guidance scale (for diffusion distillation)
        guidance_scale_start: 7.5,
        guidance_scale_end: 3.0,
        guidance_anneal_steps: 100,
        // Sub-configs with defaults
        optimizer: OptimizerConfig::default(),
        loss: LossConfig::default(),
        density: DensityConfig::default(),
        init: InitConfig {
            num_rigid: 500, // Small for demo
            num_flexible: 500,
            initial_scale: -5.0,
            initial_opacity: -2.0,
            sh_degree: 0, // DC only for demo
        },
        // Use default for remaining fields (tensorboard, etc.)
        ..Default::default()
    };

    // Validate configuration
    if let Err(e) = training_config.validate() {
        eprintln!("Invalid training config: {}", e);
        return Err(oxigaf::OxigafError::Trainer(e));
    }

    println!("  Total iterations: {}", training_config.total_iterations);
    println!("  Views per step: {}", training_config.views_per_step);
    println!(
        "  Density control: {} - {} (every {})",
        training_config.density_control_start,
        training_config.density_control_end,
        training_config.density_control_interval
    );

    // =========================================================================
    // Step 2: Create Gaussian model
    // =========================================================================

    println!();
    println!("Creating initial Gaussian model...");

    let num_initial = training_config.init.num_rigid + training_config.init.num_flexible;
    let model = create_demo_model(num_initial, training_config.init.sh_degree);
    println!("  {} Gaussians, SH degree {}", model.len(), model.sh_degree);

    // =========================================================================
    // Step 3: Configure rasterizer
    // =========================================================================

    println!();
    println!("Configuring rasterizer...");

    let raster_config = RasterConfig::new()
        .with_resolution(256, 256) // Small for demo speed
        .with_sh_degree(training_config.init.sh_degree)
        .with_background([1.0, 1.0, 1.0]); // White background

    println!(
        "  Resolution: {}x{}",
        raster_config.image_width, raster_config.image_height
    );

    // =========================================================================
    // Step 4: Initialize Trainer
    // =========================================================================

    println!();
    println!("Initializing trainer...");

    // Request GPU device
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|e| {
            oxigaf::OxigafError::Render(oxigaf::render::RenderError::GpuInit(format!(
                "No suitable GPU adapter found: {}",
                e
            )))
        })?;

    let adapter_info = adapter.get_info();
    println!("  GPU: {} ({:?})", adapter_info.name, adapter_info.backend);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("oxigaf_trainer"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| {
            oxigaf::OxigafError::Render(oxigaf::render::RenderError::DeviceCreationFailed(
                e.to_string(),
            ))
        })?;

    // Create trainer with random seed for reproducibility
    let seed = 42u64;
    let mut trainer = Trainer::new(
        training_config.clone(),
        model,
        raster_config,
        device,
        queue,
        seed,
    )?;

    println!(
        "  Trainer initialized with {} Gaussians",
        trainer.model.len()
    );

    // =========================================================================
    // Step 5: Run training loop
    // =========================================================================

    println!();
    println!("Starting training...");
    println!();

    // Set up checkpoint directory
    let checkpoint_dir = std::env::temp_dir()
        .join("oxigaf_examples")
        .join("checkpoints");
    if let Err(e) = std::fs::create_dir_all(&checkpoint_dir) {
        eprintln!("Warning: Could not create checkpoint directory: {}", e);
    }

    // Training loop
    for _ in 0..training_config.total_iterations {
        // Execute single training step
        let output = trainer.train_step()?;

        // Log progress
        if output.iteration % training_config.log_interval == 0 {
            println!(
                "  Iteration {:4} | Loss: {:.6} | Gaussians: {:5} | L1: {:.4} SSIM: {:.4}",
                output.iteration,
                output.loss.total,
                output.num_gaussians,
                output.loss.l1,
                output.loss.ssim,
            );
        }

        // Save checkpoint
        if output.iteration % training_config.checkpoint_interval == 0 {
            let checkpoint_path = checkpoint_dir.join(format!("ckpt_{:06}.json", output.iteration));
            match trainer.save_checkpoint(&checkpoint_path) {
                Ok(()) => println!("  -> Saved checkpoint: {}", checkpoint_path.display()),
                Err(e) => eprintln!("  -> Checkpoint save failed: {}", e),
            }
        }
    }

    // =========================================================================
    // Step 6: Save final checkpoint
    // =========================================================================

    println!();
    println!("Training complete!");
    println!("  Final Gaussians: {}", trainer.model.len());

    let final_checkpoint = checkpoint_dir.join("ckpt_final.json");
    trainer.save_checkpoint(&final_checkpoint)?;
    println!("  Final checkpoint: {}", final_checkpoint.display());

    // =========================================================================
    // Step 7: Demonstrate checkpoint loading
    // =========================================================================

    println!();
    println!("Demonstrating checkpoint load...");

    // This shows how to restore training from a checkpoint
    demonstrate_checkpoint_load(&final_checkpoint)?;

    // =========================================================================
    // Summary
    // =========================================================================

    println!();
    println!("Key takeaways:");
    println!("  - TrainingConfig holds all hyperparameters");
    println!("  - Trainer::new() requires GPU device and queue");
    println!("  - train_step() runs one complete iteration");
    println!("  - save_checkpoint()/from_checkpoint() for persistence");
    println!("  - Density control automatically splits/prunes Gaussians");
    println!();
    println!("Checkpoints saved to: {}", checkpoint_dir.display());

    Ok(())
}

/// Demonstrate loading a checkpoint and inspecting its contents.
fn demonstrate_checkpoint_load(checkpoint_path: &Path) -> oxigaf::Result<()> {
    // Load checkpoint data
    let checkpoint = oxigaf::trainer::checkpoint::load_checkpoint(checkpoint_path)?;

    println!("  Loaded checkpoint:");
    println!("    Version: {}", checkpoint.version);
    println!("    Iteration: {}", checkpoint.iteration);
    println!("    Gaussians: {}", checkpoint.positions.len());
    println!("    SH degree: {}", checkpoint.sh_degree);
    println!(
        "    Optimizer groups: {}",
        checkpoint.optimizer_groups.len()
    );

    // Restore model from checkpoint
    let restored_model = oxigaf::trainer::checkpoint::restore_model(&checkpoint);
    println!("  Restored model: {} Gaussians", restored_model.len());

    // In production, you would:
    // 1. Create new device/queue
    // 2. Call Trainer::from_checkpoint() to resume training
    // 3. Continue with trainer.run() or trainer.train_step()

    Ok(())
}
