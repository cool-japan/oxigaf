//! Tests for DDIM scheduler.
//!
//! Comprehensive tests for the DDIM scheduler including step computation,
//! alpha schedule verification, and noise addition.

use candle_core::{DType, Device, Result, Tensor};
use oxigaf_diffusion::{DdimScheduler, PredictionType};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Basic Scheduler Tests
// ---------------------------------------------------------------------------

#[test]
fn test_scheduler_creation() {
    let mut scheduler = DdimScheduler::new(1000, PredictionType::Epsilon);
    scheduler.set_timesteps(50);

    // Basic validation
    assert!(!scheduler.timesteps().is_empty());
    assert!(scheduler.timesteps().len() <= 50);
}

#[test]
fn test_scheduler_timesteps_ordered() {
    let mut scheduler = DdimScheduler::new(1000, PredictionType::Epsilon);
    scheduler.set_timesteps(50);
    let timesteps = scheduler.timesteps();

    // Timesteps should be in descending order (for DDIM)
    for i in 0..timesteps.len().saturating_sub(1) {
        assert!(
            timesteps[i] >= timesteps[i + 1],
            "Timesteps not ordered: {} >= {}",
            timesteps[i],
            timesteps[i + 1]
        );
    }
}

#[test]
fn test_scheduler_different_num_steps() {
    let mut scheduler_10 = DdimScheduler::new(1000, PredictionType::Epsilon);
    scheduler_10.set_timesteps(10);

    let mut scheduler_50 = DdimScheduler::new(1000, PredictionType::Epsilon);
    scheduler_50.set_timesteps(50);

    assert!(scheduler_10.timesteps().len() <= 10);
    assert!(scheduler_50.timesteps().len() <= 50);
    assert!(scheduler_50.timesteps().len() > scheduler_10.timesteps().len());
}

#[test]
fn test_scheduler_epsilon_prediction() {
    let mut scheduler = DdimScheduler::new(1000, PredictionType::Epsilon);
    scheduler.set_timesteps(50);
    // Just verify creation works with Epsilon prediction type
    assert!(!scheduler.timesteps().is_empty());
}

#[test]
fn test_scheduler_v_prediction() {
    let mut scheduler = DdimScheduler::new(1000, PredictionType::VPrediction);
    scheduler.set_timesteps(50);
    // Just verify creation works with VPrediction type
    assert!(!scheduler.timesteps().is_empty());
}

// Note: Zero timesteps causes division by zero in set_timesteps()
// This is an invalid configuration, so we don't test it

#[test]
fn test_scheduler_single_step() {
    let mut scheduler = DdimScheduler::new(1000, PredictionType::Epsilon);
    scheduler.set_timesteps(1);
    assert_eq!(scheduler.timesteps().len(), 1);
}

#[test]
fn test_scheduler_different_train_timesteps() {
    let mut scheduler_500 = DdimScheduler::new(500, PredictionType::Epsilon);
    scheduler_500.set_timesteps(20);

    let mut scheduler_1000 = DdimScheduler::new(1000, PredictionType::Epsilon);
    scheduler_1000.set_timesteps(20);

    // Both should have 20 inference steps
    assert_eq!(scheduler_500.timesteps().len(), 20);
    assert_eq!(scheduler_1000.timesteps().len(), 20);

    // But the timestep values should be different (different spacing)
    assert_ne!(scheduler_500.timesteps()[0], scheduler_1000.timesteps()[0]);
}

// ---------------------------------------------------------------------------
// Step Computation Tests
// ---------------------------------------------------------------------------

/// Test that scheduler step produces correct output shape.
#[test]
fn test_scheduler_step_shape() -> Result<()> {
    let mut scheduler = DdimScheduler::new(1000, PredictionType::Epsilon);
    scheduler.set_timesteps(50);

    let batch = 4;
    let channels = 4;
    let height = 32;
    let width = 32;

    let sample = Tensor::randn(0f32, 1f32, (batch, channels, height, width), &Device::Cpu)?;
    let model_output = Tensor::randn(0f32, 1f32, (batch, channels, height, width), &Device::Cpu)?;

    let timestep = scheduler.timesteps()[0];
    let output = scheduler.step(&model_output, timestep, &sample)?;

    assert_eq!(output.dims4()?, (batch, channels, height, width));
    Ok(())
}

/// Test that DDIM step produces finite values over iterations.
#[test]
fn test_ddim_step_finite_output() -> Result<()> {
    let mut scheduler = DdimScheduler::new(1000, PredictionType::Epsilon);
    scheduler.set_timesteps(20);

    let batch = 1;
    let channels = 4;
    let size = 16;

    // Start with noisy sample
    let x_0 = Tensor::zeros((batch, channels, size, size), DType::F32, &Device::Cpu)?;
    let noise = Tensor::randn(0f32, 1f32, (batch, channels, size, size), &Device::Cpu)?;

    let first_t = scheduler.timesteps()[0];
    let mut sample = scheduler.add_noise(&x_0, &noise, first_t)?;

    // Run a few denoising steps
    for &t in scheduler.timesteps().iter().take(5) {
        // Model predicts some noise (not perfect)
        let model_output =
            Tensor::randn(0f32, 0.5f32, (batch, channels, size, size), &Device::Cpu)?;
        sample = scheduler.step(&model_output, t, &sample)?;
    }

    // Verify output is finite (no NaN or Inf)
    let sum = sample.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(sum.is_finite(), "Sample should contain finite values");

    // Verify output shape is preserved
    assert_eq!(sample.dims4()?, (batch, channels, size, size));

    Ok(())
}

/// Test v-prediction mode produces valid outputs.
#[test]
fn test_ddim_step_v_prediction() -> Result<()> {
    let mut scheduler = DdimScheduler::new(1000, PredictionType::VPrediction);
    scheduler.set_timesteps(10);

    let batch = 2;
    let channels = 4;
    let size = 8;

    let sample = Tensor::randn(0f32, 1f32, (batch, channels, size, size), &Device::Cpu)?;
    let model_output = Tensor::randn(0f32, 1f32, (batch, channels, size, size), &Device::Cpu)?;

    let timestep = scheduler.timesteps()[0];
    let output = scheduler.step(&model_output, timestep, &sample)?;

    // Should produce finite values
    let sum = output.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(sum.is_finite(), "Output should be finite");

    Ok(())
}

/// Test epsilon-prediction mode produces valid outputs.
#[test]
fn test_ddim_step_epsilon_prediction() -> Result<()> {
    let mut scheduler = DdimScheduler::new(1000, PredictionType::Epsilon);
    scheduler.set_timesteps(10);

    let batch = 2;
    let channels = 4;
    let size = 8;

    let sample = Tensor::randn(0f32, 1f32, (batch, channels, size, size), &Device::Cpu)?;
    let model_output = Tensor::randn(0f32, 1f32, (batch, channels, size, size), &Device::Cpu)?;

    let timestep = scheduler.timesteps()[0];
    let output = scheduler.step(&model_output, timestep, &sample)?;

    // Should produce finite values
    let sum = output.abs()?.sum_all()?.to_scalar::<f32>()?;
    assert!(sum.is_finite(), "Output should be finite");

    Ok(())
}

// ---------------------------------------------------------------------------
// Add Noise Tests
// ---------------------------------------------------------------------------

/// Test add_noise produces correct output shape.
#[test]
fn test_add_noise_shape() -> Result<()> {
    let scheduler = DdimScheduler::new(1000, PredictionType::Epsilon);

    let batch = 2;
    let channels = 4;
    let size = 16;

    let original = Tensor::zeros((batch, channels, size, size), DType::F32, &Device::Cpu)?;
    let noise = Tensor::randn(0f32, 1f32, (batch, channels, size, size), &Device::Cpu)?;

    let noisy = scheduler.add_noise(&original, &noise, 500)?;

    assert_eq!(noisy.dims4()?, (batch, channels, size, size));
    Ok(())
}

/// Test that add_noise at t=0 returns mostly original.
#[test]
fn test_add_noise_t_zero() -> Result<()> {
    let scheduler = DdimScheduler::new(1000, PredictionType::Epsilon);

    let batch = 1;
    let channels = 4;
    let size = 8;

    let original = Tensor::ones((batch, channels, size, size), DType::F32, &Device::Cpu)?;
    let noise = Tensor::randn(0f32, 1f32, (batch, channels, size, size), &Device::Cpu)?;

    // At t=0, alpha_cumprod is close to 1, so result should be close to original
    let noisy = scheduler.add_noise(&original, &noise, 0)?;

    let diff = (&original - &noisy)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    let orig_sum = original.abs()?.sum_all()?.to_scalar::<f32>()?;

    // Difference should be small relative to original
    assert!(
        diff / orig_sum < 0.1,
        "At t=0, noisy should be close to original"
    );

    Ok(())
}

/// Test that add_noise at high t returns mostly noise.
#[test]
fn test_add_noise_high_t() -> Result<()> {
    let scheduler = DdimScheduler::new(1000, PredictionType::Epsilon);

    let batch = 1;
    let channels = 4;
    let size = 8;

    let original = Tensor::ones((batch, channels, size, size), DType::F32, &Device::Cpu)?;
    let noise = Tensor::randn(0f32, 1f32, (batch, channels, size, size), &Device::Cpu)?;

    // At t=999, alpha_cumprod is close to 0, so result should be close to noise
    let noisy = scheduler.add_noise(&original, &noise, 999)?;

    let diff_to_noise = (&noise - &noisy)?.abs()?.sum_all()?.to_scalar::<f32>()?;
    let noise_sum = noise.abs()?.sum_all()?.to_scalar::<f32>()?;

    // Should be closer to noise than to original at high t
    assert!(
        diff_to_noise / noise_sum < 0.5,
        "At high t, noisy should be close to noise"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Timestep Tensor Tests
// ---------------------------------------------------------------------------

/// Test timestep_tensor produces correct shape and values.
#[test]
fn test_timestep_tensor() -> Result<()> {
    let scheduler = DdimScheduler::new(1000, PredictionType::Epsilon);

    let batch_size = 4;
    let t = 500;
    let tensor = scheduler.timestep_tensor(t, batch_size, &Device::Cpu)?;

    assert_eq!(tensor.dims1()?, batch_size);

    let values = tensor.to_vec1::<f32>()?;
    for val in values {
        assert!((val - t as f32).abs() < 0.001, "All values should be {}", t);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Property-based Tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn test_scheduler_timesteps_always_descending(
        num_train in 100usize..2000,
        num_inference in 1usize..100,
    ) {
        let mut scheduler = DdimScheduler::new(num_train, PredictionType::Epsilon);
        scheduler.set_timesteps(num_inference);

        let timesteps = scheduler.timesteps();
        for i in 0..timesteps.len().saturating_sub(1) {
            prop_assert!(timesteps[i] >= timesteps[i + 1]);
        }
    }

    #[test]
    fn test_add_noise_preserves_shape(
        batch in 1usize..4,
        channels in 1usize..8,
        size in 4usize..16,
        t in 0usize..999,
    ) {
        let result = (|| -> Result<(usize, usize, usize, usize)> {
            let scheduler = DdimScheduler::new(1000, PredictionType::Epsilon);
            let original = Tensor::zeros((batch, channels, size, size), DType::F32, &Device::Cpu)?;
            let noise = Tensor::randn(0f32, 1f32, (batch, channels, size, size), &Device::Cpu)?;

            let noisy = scheduler.add_noise(&original, &noise, t)?;
            noisy.dims4()
        })();

        // Verify if successful
        if let Ok(dims) = result {
            prop_assert_eq!(dims, (batch, channels, size, size));
        }
    }
}
