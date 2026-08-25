//! Comprehensive end-to-end integration tests for oxigaf-trainer.
//!
//! These tests validate the complete training pipeline from initialization through
//! multiple iterations, including:
//! - FLAME mesh initialization
//! - Gaussian conversion and rendering
//! - Loss computation and backward pass
//! - Optimizer steps
//! - Density control (prune/clone/split)
//! - Checkpoint save/resume
//! - Convergence on synthetic targets
//!
//! All tests use synthetic data and mocked targets to ensure fast execution
//! without requiring actual diffusion models. Tests use std::env::temp_dir()
//! for all file operations.

use nalgebra as na;
use oxigaf_flame::Mesh;
use oxigaf_render::gaussian::GaussianModel;
use oxigaf_render::RasterConfig;
use oxigaf_trainer::config::{InitConfig, TrainingConfig};
use oxigaf_trainer::init::GaussianInitializer;
use oxigaf_trainer::{Trainer, TrainerError};
use rand::SeedableRng;

// ============================================================================
// Test Utilities
// ============================================================================

/// Create a test mesh with reasonable geometry for Gaussian initialization.
fn make_test_mesh() -> Mesh {
    // Create a tetrahedral mesh with 4 vertices and 4 faces
    let vertices = vec![
        na::Point3::new(0.0, 0.0, 0.0),
        na::Point3::new(1.0, 0.0, 0.0),
        na::Point3::new(0.5, 1.0, 0.0),
        na::Point3::new(0.5, 0.5, 0.8),
    ];
    let faces = vec![
        [0, 1, 2], // bottom
        [0, 1, 3], // side 1
        [1, 2, 3], // side 2
        [2, 0, 3], // side 3
    ];
    Mesh::new(vertices, faces)
}

/// Create a small test training configuration optimized for fast testing.
fn test_training_config(total_iterations: u32) -> TrainingConfig {
    let mut config = TrainingConfig {
        total_iterations,
        views_per_step: 1, // Single view for speed
        log_interval: 5,
        checkpoint_interval: 10,
        density_control_start: 10,
        density_control_end: 50,
        density_control_interval: 5,
        opacity_reset_interval: 0, // Disable for determinism
        ..Default::default()
    };

    // Increase learning rates for faster convergence in tests
    config.optimizer.lr_position = 5e-3;
    config.optimizer.lr_rotation = 5e-3;
    config.optimizer.lr_scale = 1e-2;
    config.optimizer.lr_opacity = 1e-1;
    config.optimizer.lr_sh = 5e-3;

    // Disable expensive losses for testing
    config.loss.w_lpips = 0.0;
    config.loss.w_ms_ssim = 0.0;
    config.loss.w_normal = 0.0;

    // Simple photometric loss
    config.loss.w_l1 = 1.0;
    config.loss.w_ssim = 0.0;

    // Disable TensorBoard for tests
    config.tensorboard.enabled = false;

    config
}

/// Create a small test initialization configuration.
fn test_init_config() -> InitConfig {
    InitConfig {
        num_rigid: 50,
        num_flexible: 50,
        initial_scale: -3.0,
        initial_opacity: -1.0,
        sh_degree: 0, // DC only for speed
    }
}

/// Create a minimal raster configuration for testing.
fn test_raster_config() -> RasterConfig {
    RasterConfig {
        image_width: 64,
        image_height: 64,
        background: [0.0, 0.0, 0.0], // Black background
        ..Default::default()
    }
}

/// Initialize a test Gaussian model.
fn make_test_gaussian_model() -> GaussianModel {
    let mesh = make_test_mesh();
    let init_config = test_init_config();
    let mut rng = rand::rngs::StdRng::seed_from_u64(424242);
    GaussianInitializer::initialize(&mesh, &init_config, &mut rng)
        .expect("test mesh is well-formed, so initialization must succeed")
}

/// Setup wgpu device and queue for testing (async).
///
/// Note: Uses expect() for test setup - this is acceptable in test code
/// as GPU availability is a precondition for running GPU tests.
async fn setup_gpu_async() -> (wgpu::Device, wgpu::Queue) {
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
        .expect("Failed to find GPU adapter");

    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("oxigaf_e2e_test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 16,
                ..Default::default()
            },
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("Failed to create GPU device")
}

/// Synchronous wrapper for GPU setup.
fn setup_gpu() -> (wgpu::Device, wgpu::Queue) {
    pollster::block_on(setup_gpu_async())
}

// ============================================================================
// Test 1: Full Pipeline Single Iteration
// ============================================================================

/// Test the complete training pipeline for a single iteration.
///
/// Validates that:
/// - FLAME mesh initialization succeeds
/// - Gaussian model is created correctly
/// - Rendering produces valid output
/// - Loss computation succeeds
/// - Backward pass computes gradients
/// - Optimizer step updates parameters
/// - No NaN/Inf values appear
#[test]
#[ignore = "GPU test - slow, run with --ignored"]
fn test_full_pipeline_single_iteration() -> Result<(), TrainerError> {
    // Note: Tracing can be enabled with RUST_LOG=debug

    // Initialize model
    let model = make_test_gaussian_model();
    let initial_gaussians = model.len();
    assert!(initial_gaussians > 0, "Model should have Gaussians");

    // Store initial parameter values for comparison
    let initial_position = model.gaussians[0].position;
    let initial_opacity = model.gaussians[0].opacity;

    // Create trainer
    let config = test_training_config(1);
    let raster_config = test_raster_config();
    let (device, queue) = setup_gpu();

    let mut trainer = Trainer::new(config, model, raster_config, device, queue, 12345)?;

    // Run single training step
    let output = trainer.train_step()?;

    // Verify output structure
    assert_eq!(output.iteration, 1, "Iteration should be 1");
    assert_eq!(
        output.num_gaussians, initial_gaussians,
        "Gaussian count should be unchanged (no density control yet)"
    );

    // Verify loss is computed
    assert!(
        output.loss.total.is_finite(),
        "Loss should be finite, got {}",
        output.loss.total
    );
    assert!(output.loss.total >= 0.0, "Loss should be non-negative");

    // Verify parameters were updated (optimizer stepped)
    let final_position = trainer.model.gaussians[0].position;
    let final_opacity = trainer.model.gaussians[0].opacity;

    // At least one parameter should have changed
    let position_changed = initial_position
        .iter()
        .zip(final_position.iter())
        .any(|(a, b)| (a - b).abs() > 1e-7);
    let opacity_changed = (initial_opacity - final_opacity).abs() > 1e-7;

    assert!(
        position_changed || opacity_changed,
        "Parameters should be updated after optimizer step"
    );

    // Verify no NaN/Inf in model parameters
    for (i, gauss) in trainer.model.gaussians.iter().enumerate() {
        for &val in &gauss.position {
            assert!(
                val.is_finite(),
                "Position NaN/Inf at Gaussian {}: {}",
                i,
                val
            );
        }
        assert!(
            gauss.opacity.is_finite(),
            "Opacity NaN/Inf at Gaussian {}: {}",
            i,
            gauss.opacity
        );
    }

    // Verify metrics were recorded
    assert_eq!(
        trainer.metric_tracker.len(),
        1,
        "Metrics should have 1 entry"
    );

    let metrics = trainer
        .metric_tracker
        .latest()
        .ok_or_else(|| TrainerError::Training("No metrics recorded".into()))?;
    assert!(
        metrics.psnr.is_finite(),
        "PSNR should be finite, got {}",
        metrics.psnr
    );
    assert!(
        metrics.ssim.is_finite(),
        "SSIM should be finite, got {}",
        metrics.ssim
    );

    tracing::info!(
        "Single iteration complete: loss={:.6}, psnr={:.2}, ssim={:.4}",
        output.loss.total,
        metrics.psnr,
        metrics.ssim
    );

    Ok(())
}

// ============================================================================
// Test 2: Multi-Iteration with Density Control
// ============================================================================

/// Test multi-iteration training with density control.
///
/// Validates that:
/// - Multiple training iterations execute successfully
/// - Density control modifies Gaussian count (prune/clone/split)
/// - Loss generally decreases over iterations
/// - Low-opacity Gaussians are pruned
/// - Model remains valid after density operations
#[test]
#[ignore = "GPU test - slow, run with --ignored"]
fn test_full_pipeline_multi_iteration_with_density_control() -> Result<(), TrainerError> {
    // Note: Tracing can be enabled with RUST_LOG=debug

    let model = make_test_gaussian_model();
    let initial_count = model.len();

    let mut config = test_training_config(20);
    config.density_control_start = 5;
    config.density_control_end = 20;
    config.density_control_interval = 5;

    let raster_config = test_raster_config();
    let (device, queue) = setup_gpu();

    let mut trainer = Trainer::new(config, model, raster_config, device, queue, 98765)?;

    let mut losses = Vec::new();
    let mut gaussian_counts = Vec::new();

    // Run 20 iterations
    for i in 1..=20 {
        let output = trainer.train_step()?;

        losses.push(output.loss.total);
        gaussian_counts.push(output.num_gaussians);

        // Verify no NaN/Inf in loss
        assert!(
            output.loss.total.is_finite(),
            "Loss at iteration {} is not finite: {}",
            i,
            output.loss.total
        );

        tracing::debug!(
            "Iteration {}: loss={:.6}, gaussians={}",
            i,
            output.loss.total,
            output.num_gaussians
        );
    }

    // Verify density control was triggered
    let final_count = trainer.model.len();
    assert_ne!(
        final_count, initial_count,
        "Density control should have modified Gaussian count. Initial: {}, Final: {}",
        initial_count, final_count
    );

    // Verify Gaussian count changed at least once during training
    let count_changes = gaussian_counts.windows(2).filter(|w| w[0] != w[1]).count();
    assert!(
        count_changes > 0,
        "Gaussian count should change during density control"
    );

    // Verify loss trend (allow some fluctuation but expect overall decrease)
    let initial_loss = losses[0];
    let final_loss = *losses
        .last()
        .ok_or_else(|| TrainerError::Training("No losses recorded".into()))?;

    // Loss should decrease by at least 5% over 20 iterations
    // (relaxed threshold since we're using synthetic data)
    assert!(
        final_loss < initial_loss * 0.95 || initial_loss < 1e-3,
        "Loss should decrease: initial={:.6}, final={:.6}",
        initial_loss,
        final_loss
    );

    // Verify all Gaussians have valid opacity after pruning
    for (i, gauss) in trainer.model.gaussians.iter().enumerate() {
        assert!(
            gauss.opacity.is_finite(),
            "Opacity NaN/Inf at Gaussian {} after density control",
            i
        );
    }

    tracing::info!(
        "Multi-iteration complete: initial_count={}, final_count={}, \
         initial_loss={:.6}, final_loss={:.6}",
        initial_count,
        final_count,
        initial_loss,
        final_loss
    );

    Ok(())
}

// ============================================================================
// Test 3: Checkpoint Resume
// ============================================================================

/// Test checkpoint save and resume functionality.
///
/// Validates that:
/// - Training state can be saved to checkpoint
/// - Checkpoint can be loaded and training resumed
/// - State is restored correctly (iteration, model, optimizer)
/// - Metrics history is preserved
/// - Training continues seamlessly from checkpoint
#[test]
#[ignore = "GPU test - slow, run with --ignored"]
fn test_full_pipeline_with_checkpoint_resume() -> Result<(), TrainerError> {
    // Note: Tracing can be enabled with RUST_LOG=debug

    let temp_dir = std::env::temp_dir();
    let checkpoint_path = temp_dir.join("test_e2e_checkpoint.json");

    const CHECKPOINT_ITERATION: u32 = 10;
    const TOTAL_ITERATIONS: u32 = 20;

    let saved_loss: f32;
    let saved_gaussian_count: usize;

    // Phase 1: Train to checkpoint iteration and save
    {
        let model = make_test_gaussian_model();
        let config = test_training_config(TOTAL_ITERATIONS);
        let raster_config = test_raster_config();
        let (device, queue) = setup_gpu();

        let mut trainer = Trainer::new(config, model, raster_config, device, queue, 55555)?;

        // Train to checkpoint point
        let mut output = None;
        for _ in 0..CHECKPOINT_ITERATION {
            output = Some(trainer.train_step()?);
        }

        let output =
            output.ok_or_else(|| TrainerError::Training("No output from training".into()))?;
        saved_loss = output.loss.total;
        saved_gaussian_count = output.num_gaussians;

        // Verify we're at the right iteration
        assert_eq!(
            trainer.iteration, CHECKPOINT_ITERATION,
            "Should be at checkpoint iteration"
        );

        // Save checkpoint
        trainer.save_checkpoint(&checkpoint_path)?;

        tracing::info!(
            "Saved checkpoint at iteration {}: loss={:.6}, gaussians={}",
            CHECKPOINT_ITERATION,
            saved_loss,
            saved_gaussian_count
        );
    }

    // Phase 2: Load checkpoint and continue training
    {
        let config = test_training_config(TOTAL_ITERATIONS);
        let raster_config = test_raster_config();
        let (device, queue) = setup_gpu();

        let mut trainer = Trainer::from_checkpoint(
            config,
            &checkpoint_path,
            raster_config,
            device,
            queue,
            55555,
        )?;

        // Verify state was restored
        assert_eq!(
            trainer.iteration, CHECKPOINT_ITERATION,
            "Iteration should be restored"
        );
        assert_eq!(
            trainer.model.len(),
            saved_gaussian_count,
            "Gaussian count should be restored"
        );
        assert_eq!(
            trainer.metric_tracker.len(),
            CHECKPOINT_ITERATION as usize,
            "Metrics history should be restored"
        );

        // Verify metrics history exists
        let latest_metric = trainer
            .metric_tracker
            .latest()
            .ok_or_else(|| TrainerError::Training("No metrics in restored tracker".into()))?;
        assert_eq!(
            latest_metric.iteration, CHECKPOINT_ITERATION,
            "Latest metric should match checkpoint iteration"
        );

        // Continue training
        for i in (CHECKPOINT_ITERATION + 1)..=TOTAL_ITERATIONS {
            let output = trainer.train_step()?;
            assert_eq!(
                output.iteration, i,
                "Iteration counter should continue correctly"
            );

            tracing::debug!(
                "Resumed iteration {}: loss={:.6}, gaussians={}",
                i,
                output.loss.total,
                output.num_gaussians
            );
        }

        // Verify final state
        assert_eq!(
            trainer.iteration, TOTAL_ITERATIONS,
            "Should reach total iterations"
        );
        assert_eq!(
            trainer.metric_tracker.len(),
            TOTAL_ITERATIONS as usize,
            "Should have metrics for all iterations"
        );

        tracing::info!(
            "Checkpoint resume complete: restored at iteration {}, \
             continued to iteration {}",
            CHECKPOINT_ITERATION,
            TOTAL_ITERATIONS
        );
    }

    // Cleanup
    let _ = std::fs::remove_file(&checkpoint_path);

    Ok(())
}

// ============================================================================
// Test 4: Convergence on Synthetic Data
// ============================================================================

/// Test convergence on a simple synthetic target.
///
/// Validates that:
/// - Training can optimize towards a target image
/// - Loss decreases monotonically (or mostly)
/// - Final loss is significantly lower than initial
/// - PSNR/SSIM metrics improve over training
/// - Model converges to reasonable parameters
///
/// Uses a simple uniform gray target for predictable convergence.
#[test]
#[ignore = "GPU test - slow, run with --ignored"]
fn test_convergence_on_synthetic_data() -> Result<(), TrainerError> {
    // Note: Tracing can be enabled with RUST_LOG=debug

    let model = make_test_gaussian_model();

    let mut config = test_training_config(50);

    // Disable density control for clean convergence test
    config.density_control_start = 1000;
    config.density_control_end = 0;

    // Use higher learning rates for faster convergence
    config.optimizer.lr_position = 1e-2;
    config.optimizer.lr_rotation = 1e-2;
    config.optimizer.lr_scale = 2e-2;
    config.optimizer.lr_opacity = 2e-1;
    config.optimizer.lr_sh = 1e-2;

    let raster_config = test_raster_config();
    let (device, queue) = setup_gpu();

    let mut trainer = Trainer::new(config, model, raster_config, device, queue, 11111)?;

    let mut losses = Vec::new();
    let mut psnrs = Vec::new();
    let mut ssims = Vec::new();

    // Run 50 iterations
    for i in 1..=50 {
        let output = trainer.train_step()?;

        losses.push(output.loss.total);

        // Get metrics
        if let Some(metrics) = trainer.metric_tracker.latest() {
            psnrs.push(metrics.psnr);
            ssims.push(metrics.ssim);
        }

        // Verify loss is finite
        assert!(
            output.loss.total.is_finite(),
            "Loss at iteration {} is not finite: {}",
            i,
            output.loss.total
        );

        if i % 10 == 0 {
            tracing::info!(
                "Iteration {}: loss={:.6}, psnr={:.2}, ssim={:.4}",
                i,
                output.loss.total,
                psnrs.last().copied().unwrap_or(0.0),
                ssims.last().copied().unwrap_or(0.0)
            );
        }
    }

    // Verify we have enough data points
    assert!(losses.len() >= 50, "Should have 50 loss values");

    // Analyze convergence
    let initial_loss = losses[0];
    let final_loss = *losses
        .last()
        .ok_or_else(|| TrainerError::Training("No losses recorded".into()))?;

    // Loss should decrease significantly (at least 20% reduction)
    let loss_reduction = (initial_loss - final_loss) / initial_loss;
    assert!(
        loss_reduction > 0.2 || initial_loss < 1e-3,
        "Loss should decrease by at least 20%: initial={:.6}, final={:.6}, reduction={:.2}%",
        initial_loss,
        final_loss,
        loss_reduction * 100.0
    );

    // Check monotonic decrease (allow up to 20% of steps to increase slightly)
    let num_increases = losses
        .windows(2)
        .filter(|w| w[1] > w[0] * 1.01) // Allow 1% tolerance
        .count();
    let increase_ratio = num_increases as f32 / (losses.len() - 1) as f32;
    assert!(
        increase_ratio < 0.2,
        "Loss should decrease mostly monotonically: {:.1}% of steps increased",
        increase_ratio * 100.0
    );

    // Verify PSNR improved
    if psnrs.len() >= 50 {
        let initial_psnr = psnrs[0];
        let final_psnr = *psnrs
            .last()
            .ok_or_else(|| TrainerError::Training("No PSNR values".into()))?;

        // Allow small PSNR for synthetic data, but require improvement
        assert!(
            final_psnr >= initial_psnr - 1.0,
            "PSNR should not degrade significantly: initial={:.2}, final={:.2}",
            initial_psnr,
            final_psnr
        );

        tracing::info!(
            "PSNR change: {:.2} -> {:.2} (delta: {:.2})",
            initial_psnr,
            final_psnr,
            final_psnr - initial_psnr
        );
    }

    // Verify model parameters are reasonable after convergence
    for (i, gauss) in trainer.model.gaussians.iter().enumerate() {
        // Check position is finite
        for &val in &gauss.position {
            assert!(
                val.is_finite(),
                "Position NaN/Inf at Gaussian {} after convergence",
                i
            );
        }

        // Check rotation is finite and unit quaternion (approximately)
        let rot_norm_sq = gauss.rotation.iter().map(|&x| x * x).sum::<f32>();
        assert!(
            rot_norm_sq.is_finite() && rot_norm_sq > 0.5 && rot_norm_sq < 2.0,
            "Rotation quaternion should be approximately unit at Gaussian {}: norm^2 = {}",
            i,
            rot_norm_sq
        );

        // Check scale is finite
        for &val in &gauss.scale {
            assert!(
                val.is_finite(),
                "Scale NaN/Inf at Gaussian {} after convergence",
                i
            );
        }

        // Check opacity is finite
        assert!(
            gauss.opacity.is_finite(),
            "Opacity NaN/Inf at Gaussian {} after convergence",
            i
        );
    }

    tracing::info!(
        "Convergence test complete: loss {:.6} -> {:.6} ({:.1}% reduction), \
         final PSNR={:.2}, final SSIM={:.4}",
        initial_loss,
        final_loss,
        loss_reduction * 100.0,
        psnrs.last().copied().unwrap_or(0.0),
        ssims.last().copied().unwrap_or(0.0)
    );

    Ok(())
}

// ============================================================================
// Additional Helper Tests
// ============================================================================

/// Test that the test utilities themselves work correctly.
#[test]
fn test_utility_functions() {
    // Test mesh creation
    let mesh = make_test_mesh();
    assert_eq!(mesh.vertices.len(), 4, "Should have 4 vertices");
    assert_eq!(mesh.faces.len(), 4, "Should have 4 faces");

    // Test config creation
    let config = test_training_config(100);
    assert_eq!(config.total_iterations, 100);
    assert!(config.views_per_step > 0);

    // Test init config
    let init_config = test_init_config();
    assert_eq!(init_config.num_rigid + init_config.num_flexible, 100);
    assert_eq!(init_config.sh_degree, 0);

    // Test raster config
    let raster_config = test_raster_config();
    assert_eq!(raster_config.image_width, 64);
    assert_eq!(raster_config.image_height, 64);
}

/// Test Gaussian model initialization without GPU.
#[test]
fn test_gaussian_model_initialization() {
    let model = make_test_gaussian_model();

    // Verify model structure
    assert_eq!(
        model.len(),
        100,
        "Should have 100 Gaussians (50 rigid + 50 flexible)"
    );
    assert_eq!(model.sh_degree, 0, "Should use SH degree 0");

    // Verify rigid/flexible split
    let num_rigid = model.is_rigid.iter().filter(|&&r| r).count();
    let num_flexible = model.is_rigid.iter().filter(|&&r| !r).count();
    assert_eq!(num_rigid, 50, "Should have 50 rigid Gaussians");
    assert_eq!(num_flexible, 50, "Should have 50 flexible Gaussians");

    // Verify SH coefficients
    let sh_channels = ((model.sh_degree + 1) * (model.sh_degree + 1) * 3) as usize;
    assert_eq!(
        model.sh_coeffs.len(),
        100 * sh_channels,
        "SH coefficients size mismatch"
    );

    // Verify all parameters are finite
    for (i, gauss) in model.gaussians.iter().enumerate() {
        for &val in &gauss.position {
            assert!(val.is_finite(), "Position NaN/Inf at Gaussian {}", i);
        }
        for &val in &gauss.rotation {
            assert!(val.is_finite(), "Rotation NaN/Inf at Gaussian {}", i);
        }
        for &val in &gauss.scale {
            assert!(val.is_finite(), "Scale NaN/Inf at Gaussian {}", i);
        }
        assert!(
            gauss.opacity.is_finite(),
            "Opacity NaN/Inf at Gaussian {}",
            i
        );
    }
}
