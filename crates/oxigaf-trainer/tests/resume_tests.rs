//! Integration tests for checkpoint resume functionality.
//!
//! Verifies that resumed training:
//! - Preserves metrics history
//! - Continues iteration counter correctly
//! - Maintains optimizer state
//! - Produces deterministic results with same seed
//!
//! Uses std::env::temp_dir() for all file operations.

use nalgebra as na;
use oxigaf_flame::Mesh;
use oxigaf_render::gaussian::GaussianModel;
use oxigaf_render::RasterConfig;
use oxigaf_trainer::config::{InitConfig, TrainingConfig};
use oxigaf_trainer::init::GaussianInitializer;
use oxigaf_trainer::{Trainer, TrainerError};
use rand::SeedableRng;

/// Create a minimal test mesh (simple triangle).
fn make_test_mesh() -> Mesh {
    // Simple triangle mesh for testing
    let vertices = vec![
        na::Point3::new(0.0, 0.0, 0.0),
        na::Point3::new(1.0, 0.0, 0.0),
        na::Point3::new(0.0, 1.0, 0.0),
        na::Point3::new(0.5, 0.5, 1.0),
    ];
    let faces = vec![[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
    Mesh::new(vertices, faces)
}

/// Create a test training configuration with minimal iterations.
fn test_config() -> TrainingConfig {
    TrainingConfig {
        total_iterations: 100,
        log_interval: 10,
        checkpoint_interval: 20,
        density_control_start: 50,
        density_control_end: 80,
        density_control_interval: 10,
        opacity_reset_interval: 0, // Disable for determinism
        ..Default::default()
    }
}

/// Create a minimal Gaussian model for testing.
fn make_test_model() -> GaussianModel {
    let mesh = make_test_mesh();

    let init_config = InitConfig {
        num_rigid: 20,
        num_flexible: 10,
        initial_scale: -3.0,
        initial_opacity: -1.0,
        sh_degree: 0,
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(424242);
    GaussianInitializer::initialize(&mesh, &init_config, &mut rng)
}

/// Setup wgpu device and queue for testing.
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
        })
        .await
        .expect("Failed to find GPU adapter");

    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("oxigaf_trainer_test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                // Increase storage buffer limit for backward pass (needs 13+ buffers)
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

fn setup_gpu() -> (wgpu::Device, wgpu::Queue) {
    pollster::block_on(setup_gpu_async())
}

// ============================================================================
// Metrics History Tests
// ============================================================================

#[test]
#[ignore = "GPU test - slow, run with --ignored"]
fn test_resume_preserves_metrics_history() -> Result<(), TrainerError> {
    let temp_dir = std::env::temp_dir();
    let checkpoint_path = temp_dir.join("test_resume_metrics.json");

    // Phase 1: Train for a few iterations and save checkpoint
    {
        let model = make_test_model();
        let config = test_config();
        let raster_config = RasterConfig::default();
        let (device, queue) = setup_gpu();

        let mut trainer = Trainer::new(config.clone(), model, raster_config, device, queue, 12345)?;

        // Run 5 training steps
        for _ in 0..5 {
            trainer.train_step()?;
        }

        // Save checkpoint
        trainer.save_checkpoint(&checkpoint_path)?;

        // Verify we have metrics
        assert_eq!(trainer.metric_tracker.len(), 5);
    }

    // Phase 2: Resume and verify metrics history is preserved
    {
        let config = test_config();
        let raster_config = RasterConfig::default();
        let (device, queue) = setup_gpu();

        let trainer = Trainer::from_checkpoint(
            config,
            &checkpoint_path,
            raster_config,
            device,
            queue,
            12345,
        )?;

        // Verify metrics history was restored
        assert_eq!(trainer.metric_tracker.len(), 5);
        assert!(trainer.metric_tracker.latest().is_some());

        // Verify iteration counter was restored
        assert_eq!(trainer.iteration, 5);
    }

    // Cleanup
    let _ = std::fs::remove_file(&checkpoint_path);

    Ok(())
}

#[test]
#[ignore = "GPU test - slow, run with --ignored"]
fn test_resume_continues_iteration_counter() -> Result<(), TrainerError> {
    let temp_dir = std::env::temp_dir();
    let checkpoint_path = temp_dir.join("test_resume_iteration.json");

    // Phase 1: Train to iteration 10
    {
        let model = make_test_model();
        let config = test_config();
        let raster_config = RasterConfig::default();
        let (device, queue) = setup_gpu();

        let mut trainer = Trainer::new(config.clone(), model, raster_config, device, queue, 99999)?;

        // Run 10 steps
        for _ in 0..10 {
            trainer.train_step()?;
        }

        assert_eq!(trainer.iteration, 10);
        trainer.save_checkpoint(&checkpoint_path)?;
    }

    // Phase 2: Resume and continue training
    {
        let config = test_config();
        let raster_config = RasterConfig::default();
        let (device, queue) = setup_gpu();

        let mut trainer = Trainer::from_checkpoint(
            config,
            &checkpoint_path,
            raster_config,
            device,
            queue,
            99999,
        )?;

        assert_eq!(trainer.iteration, 10);

        // Run 5 more steps
        for _ in 0..5 {
            let output = trainer.train_step()?;
            // Verify iteration increments correctly
            assert!(output.iteration > 10);
            assert!(output.iteration <= 15);
        }

        assert_eq!(trainer.iteration, 15);
    }

    // Cleanup
    let _ = std::fs::remove_file(&checkpoint_path);

    Ok(())
}

// ============================================================================
// Optimizer State Tests
// ============================================================================

#[test]
#[ignore = "GPU test - slow, run with --ignored"]
fn test_resume_preserves_optimizer_state() -> Result<(), TrainerError> {
    let temp_dir = std::env::temp_dir();
    let checkpoint_path = temp_dir.join("test_resume_optimizer.json");

    // Phase 1: Train and accumulate optimizer momentum
    let initial_position_t: u32;
    {
        let model = make_test_model();
        let config = test_config();
        let raster_config = RasterConfig::default();
        let (device, queue) = setup_gpu();

        let mut trainer = Trainer::new(config.clone(), model, raster_config, device, queue, 77777)?;

        // Run several steps to accumulate momentum
        for _ in 0..20 {
            trainer.train_step()?;
        }

        initial_position_t = trainer.optimizer.position.t;
        assert!(
            initial_position_t > 0,
            "Optimizer should have non-zero timestep"
        );

        trainer.save_checkpoint(&checkpoint_path)?;
    }

    // Phase 2: Resume and verify optimizer state
    {
        let config = test_config();
        let raster_config = RasterConfig::default();
        let (device, queue) = setup_gpu();

        let trainer = Trainer::from_checkpoint(
            config,
            &checkpoint_path,
            raster_config,
            device,
            queue,
            77777,
        )?;

        // Verify optimizer timestep was restored
        assert_eq!(
            trainer.optimizer.position.t, initial_position_t,
            "Optimizer timestep should be preserved"
        );

        // Verify optimizer has non-zero momentum (m and v are not all zeros)
        let has_momentum = trainer.optimizer.position.m.iter().any(|&x| x.abs() > 1e-8);
        assert!(has_momentum, "Optimizer should have accumulated momentum");
    }

    // Cleanup
    let _ = std::fs::remove_file(&checkpoint_path);

    Ok(())
}

// ============================================================================
// Determinism Tests
// ============================================================================

#[test]
#[ignore = "GPU test - slow, run with --ignored"]
fn test_resume_deterministic() -> Result<(), TrainerError> {
    let temp_dir = std::env::temp_dir();
    let checkpoint_path = temp_dir.join("test_resume_deterministic.json");

    const SEED: u64 = 424242;

    // Phase 1: Continuous training for 30 steps
    let continuous_final_loss: f32;
    {
        let model = make_test_model();
        let config = test_config();
        let raster_config = RasterConfig::default();
        let (device, queue) = setup_gpu();

        let mut trainer = Trainer::new(config.clone(), model, raster_config, device, queue, SEED)?;

        // Run 30 steps continuously
        let mut final_output = None;
        for _ in 0..30 {
            final_output = Some(trainer.train_step()?);
        }

        continuous_final_loss = final_output.expect("Should have output").loss.total;
    }

    // Phase 2: Training with checkpoint resume at step 15
    let resumed_final_loss: f32;
    {
        // First segment: 15 steps
        {
            let model = make_test_model();
            let config = test_config();
            let raster_config = RasterConfig::default();
            let (device, queue) = setup_gpu();

            let mut trainer =
                Trainer::new(config.clone(), model, raster_config, device, queue, SEED)?;

            for _ in 0..15 {
                trainer.train_step()?;
            }

            trainer.save_checkpoint(&checkpoint_path)?;
        }

        // Second segment: resume and continue for 15 more steps
        {
            let config = test_config();
            let raster_config = RasterConfig::default();
            let (device, queue) = setup_gpu();

            let mut trainer = Trainer::from_checkpoint(
                config,
                &checkpoint_path,
                raster_config,
                device,
                queue,
                SEED,
            )?;

            let mut final_output = None;
            for _ in 0..15 {
                final_output = Some(trainer.train_step()?);
            }

            resumed_final_loss = final_output.expect("Should have output").loss.total;
        }
    }

    // Verify that resumed training produces similar results to continuous training
    // Note: Due to floating-point precision and potential GPU non-determinism,
    // we allow a small tolerance rather than requiring exact equality.
    let relative_error = (continuous_final_loss - resumed_final_loss).abs() / continuous_final_loss;
    assert!(
        relative_error < 0.01,
        "Resumed training should produce similar results: continuous={}, resumed={}, error={}",
        continuous_final_loss,
        resumed_final_loss,
        relative_error
    );

    // Cleanup
    let _ = std::fs::remove_file(&checkpoint_path);

    Ok(())
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_resume_with_corrupted_checkpoint() {
    let temp_dir = std::env::temp_dir();
    let corrupted_path = temp_dir.join("test_corrupted_resume.json");

    // Write invalid JSON
    std::fs::write(&corrupted_path, "{ this is not valid json }").expect("Failed to write file");

    let config = test_config();
    let raster_config = RasterConfig::default();
    let (device, queue) = setup_gpu();

    let result =
        Trainer::from_checkpoint(config, &corrupted_path, raster_config, device, queue, 12345);

    assert!(result.is_err(), "Should fail to load corrupted checkpoint");

    // Cleanup
    let _ = std::fs::remove_file(&corrupted_path);
}

#[test]
fn test_resume_with_nonexistent_checkpoint() {
    let temp_dir = std::env::temp_dir();
    let nonexistent_path = temp_dir.join("this_file_does_not_exist_xyz.json");

    let config = test_config();
    let raster_config = RasterConfig::default();
    let (device, queue) = setup_gpu();

    let result = Trainer::from_checkpoint(
        config,
        &nonexistent_path,
        raster_config,
        device,
        queue,
        12345,
    );

    assert!(
        result.is_err(),
        "Should fail to load nonexistent checkpoint"
    );
}

// ============================================================================
// Model Size Change Tests
// ============================================================================

#[test]
#[ignore = "GPU test - slow, run with --ignored"]
fn test_resume_after_density_control() -> Result<(), TrainerError> {
    let temp_dir = std::env::temp_dir();
    let checkpoint_path = temp_dir.join("test_resume_density.json");

    let initial_gaussian_count: usize;
    let post_densify_count: usize;

    // Phase 1: Train until density control modifies model
    {
        let model = make_test_model();
        initial_gaussian_count = model.len();

        let mut config = test_config();
        config.density_control_start = 5;
        config.density_control_end = 50;
        config.density_control_interval = 5;

        let raster_config = RasterConfig::default();
        let (device, queue) = setup_gpu();

        let mut trainer = Trainer::new(config.clone(), model, raster_config, device, queue, 55555)?;

        // Run enough steps to trigger density control
        for _ in 0..10 {
            trainer.train_step()?;
        }

        post_densify_count = trainer.model.len();
        trainer.save_checkpoint(&checkpoint_path)?;

        tracing::info!(
            "Gaussian count: initial={}, post-densify={}",
            initial_gaussian_count,
            post_densify_count
        );
    }

    // Phase 2: Resume and verify model size is preserved
    {
        let config = test_config();
        let raster_config = RasterConfig::default();
        let (device, queue) = setup_gpu();

        let trainer = Trainer::from_checkpoint(
            config,
            &checkpoint_path,
            raster_config,
            device,
            queue,
            55555,
        )?;

        assert_eq!(
            trainer.model.len(),
            post_densify_count,
            "Model size should be preserved after resume"
        );

        // Verify optimizer buffers match model size
        assert_eq!(
            trainer.optimizer.position.m.len(),
            post_densify_count * 3,
            "Optimizer position buffer should match model size"
        );
    }

    // Cleanup
    let _ = std::fs::remove_file(&checkpoint_path);

    Ok(())
}
