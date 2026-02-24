//! Tests for diffusion target generation and SDS loss integration.

use oxigaf_trainer::diffusion_target::{
    DiffusionTargetConfig, DiffusionTargetGenerator, SdsLoss, SdsWeighting, TemporalConsistency,
    ViewConsistencyLoss,
};

// ---------------------------------------------------------------------------
// DiffusionTargetConfig tests
// ---------------------------------------------------------------------------

#[test]
fn test_diffusion_target_config_default() {
    let config = DiffusionTargetConfig::default();

    assert_eq!(config.num_inference_steps, 50);
    assert_eq!(config.guidance_scale, 3.0);
    assert_eq!(config.warmup_iterations, 1000);
    assert_eq!(config.timestep_start, 1000);
    assert_eq!(config.timestep_end, 50);
    assert_eq!(config.timestep_anneal_steps, 10_000);
    assert_eq!(config.sds_weight, 0.5);
    assert!(config.enable_view_warping);
}

#[test]
fn test_diffusion_target_config_validation() {
    // Valid default config
    let config = DiffusionTargetConfig::default();
    assert!(config.validate().is_ok());

    // Invalid: zero inference steps
    let invalid = DiffusionTargetConfig {
        num_inference_steps: 0,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());

    // Invalid: negative guidance scale
    let invalid2 = DiffusionTargetConfig {
        guidance_scale: -1.0,
        ..Default::default()
    };
    assert!(invalid2.validate().is_err());

    // Invalid: NaN guidance scale
    let invalid3 = DiffusionTargetConfig {
        guidance_scale: f32::NAN,
        ..Default::default()
    };
    assert!(invalid3.validate().is_err());

    // Invalid: timestep_start <= timestep_end
    let invalid4 = DiffusionTargetConfig {
        timestep_start: 50,
        timestep_end: 100,
        ..Default::default()
    };
    assert!(invalid4.validate().is_err());
}

// ---------------------------------------------------------------------------
// Timestep annealing tests
// ---------------------------------------------------------------------------

#[test]
fn test_timestep_annealing_during_warmup() {
    let config = DiffusionTargetConfig {
        warmup_iterations: 100,
        timestep_start: 1000,
        timestep_end: 100,
        timestep_anneal_steps: 1000,
        ..Default::default()
    };

    // During warmup, timestep should be at max (start)
    assert_eq!(config.current_timestep(0), 1000);
    assert_eq!(config.current_timestep(50), 1000);
    assert_eq!(config.current_timestep(99), 1000);
}

#[test]
fn test_timestep_annealing_after_warmup() {
    let config = DiffusionTargetConfig {
        warmup_iterations: 100,
        timestep_start: 1000,
        timestep_end: 100,
        timestep_anneal_steps: 1000,
        ..Default::default()
    };

    // Just after warmup, should still be at start
    assert_eq!(config.current_timestep(100), 1000);

    // Midway through annealing (500 steps after warmup)
    let mid = config.current_timestep(600);
    assert!(
        mid < 1000 && mid > 100,
        "mid timestep = {}, expected between 100 and 1000",
        mid
    );

    // At end of annealing
    let end = config.current_timestep(1100);
    assert_eq!(end, 100);

    // Past annealing - should stay at end
    let past = config.current_timestep(5000);
    assert_eq!(past, 100);
}

#[test]
fn test_timestep_annealing_linear() {
    let config = DiffusionTargetConfig {
        warmup_iterations: 0,
        timestep_start: 1000,
        timestep_end: 0,
        timestep_anneal_steps: 1000,
        ..Default::default()
    };

    // Linear interpolation: at t=500, should be around 500
    let mid = config.current_timestep(500);
    assert!((mid as i32 - 500).abs() < 10, "expected ~500, got {}", mid);

    // At t=250, should be around 750
    let quarter = config.current_timestep(250);
    assert!(
        (quarter as i32 - 750).abs() < 10,
        "expected ~750, got {}",
        quarter
    );
}

// ---------------------------------------------------------------------------
// Warmup period tests
// ---------------------------------------------------------------------------

#[test]
fn test_warmup_period_detection() {
    let config = DiffusionTargetConfig {
        warmup_iterations: 500,
        ..Default::default()
    };
    let gen = DiffusionTargetGenerator::new(config);

    // During warmup
    assert!(gen.is_warmup(0));
    assert!(gen.is_warmup(250));
    assert!(gen.is_warmup(499));

    // After warmup
    assert!(!gen.is_warmup(500));
    assert!(!gen.is_warmup(1000));
}

#[test]
fn test_sds_weight_during_warmup() {
    let config = DiffusionTargetConfig {
        warmup_iterations: 100,
        sds_weight: 1.0,
        ..Default::default()
    };
    let gen = DiffusionTargetGenerator::new(config);

    // During warmup, SDS weight should be 0
    assert_eq!(gen.sds_weight(0), 0.0);
    assert_eq!(gen.sds_weight(50), 0.0);
    assert_eq!(gen.sds_weight(99), 0.0);
}

#[test]
fn test_sds_weight_ramp_after_warmup() {
    let config = DiffusionTargetConfig {
        warmup_iterations: 100,
        sds_weight: 1.0,
        ..Default::default()
    };
    let gen = DiffusionTargetGenerator::new(config);

    // Exactly at warmup boundary, ramp starts at 0
    assert_eq!(gen.sds_weight(100), 0.0);

    // Just after warmup (iteration 101), should have positive weight
    let w101 = gen.sds_weight(101);
    assert!(w101 > 0.0, "expected positive weight at 101, got {}", w101);

    // Should ramp up
    let w200 = gen.sds_weight(200);
    assert!(w200 > w101, "expected {} > {}", w200, w101);

    // After 500 iterations post-warmup, should be at full weight
    let w600 = gen.sds_weight(600);
    assert!((w600 - 1.0).abs() < 0.01, "expected ~1.0, got {}", w600);
}

// ---------------------------------------------------------------------------
// SDS Loss tests
// ---------------------------------------------------------------------------

#[test]
fn test_sds_loss_identical_views() {
    let loss = SdsLoss::default();
    let view = vec![0.5_f32; 100];

    let l = loss.compute(
        std::slice::from_ref(&view),
        std::slice::from_ref(&view),
        500,
    );
    assert!(
        l.abs() < 1e-6,
        "identical views should have ~0 loss, got {}",
        l
    );
}

#[test]
fn test_sds_loss_different_views() {
    let loss = SdsLoss::default();
    let view1 = vec![0.0_f32; 100];
    let view2 = vec![1.0_f32; 100];

    let l = loss.compute(&[view1], &[view2], 500);
    assert!(
        l > 0.0,
        "different views should have positive loss, got {}",
        l
    );
}

#[test]
fn test_sds_loss_empty_input() {
    let loss = SdsLoss::default();

    assert_eq!(loss.compute(&[], &[], 500), 0.0);
    assert_eq!(loss.compute(&[vec![0.5; 10]], &[], 500), 0.0);
    assert_eq!(loss.compute(&[], &[vec![0.5; 10]], 500), 0.0);
}

#[test]
fn test_sds_loss_multi_view() {
    let loss = SdsLoss::default();
    let rendered = vec![vec![0.5_f32; 100], vec![0.3_f32; 100]];
    let targets = vec![vec![0.6_f32; 100], vec![0.4_f32; 100]];

    let l = loss.compute(&rendered, &targets, 500);
    assert!(l > 0.0, "should have positive loss for different images");
}

#[test]
fn test_sds_loss_weighting() {
    // Test different weighting schemes
    let loss_uniform = SdsLoss {
        weighting: SdsWeighting::Uniform,
        max_timestep: 1000,
    };

    let loss_linear = SdsLoss {
        weighting: SdsWeighting::Linear,
        max_timestep: 1000,
    };

    let rendered = vec![vec![0.0_f32; 100]];
    let targets = vec![vec![1.0_f32; 100]];

    // At high timestep, linear should give higher weight
    let l_uniform = loss_uniform.compute(&rendered, &targets, 900);
    let l_linear = loss_linear.compute(&rendered, &targets, 900);

    // Linear weight at t=900 is 0.9, so loss should be scaled
    assert!(
        l_linear > 0.8 * l_uniform && l_linear < 1.1 * l_uniform,
        "linear={}, uniform={}",
        l_linear,
        l_uniform
    );
}

#[test]
fn test_sds_loss_timestep_dependency() {
    let loss = SdsLoss {
        weighting: SdsWeighting::Linear,
        max_timestep: 1000,
    };

    let rendered = vec![vec![0.0_f32; 100]];
    let targets = vec![vec![1.0_f32; 100]];

    // Higher timestep should give higher weighted loss
    let l_low = loss.compute(&rendered, &targets, 100);
    let l_high = loss.compute(&rendered, &targets, 900);

    assert!(
        l_high > l_low,
        "higher timestep should give higher loss: {} vs {}",
        l_high,
        l_low
    );
}

// ---------------------------------------------------------------------------
// View Consistency Loss tests
// ---------------------------------------------------------------------------

#[test]
fn test_view_consistency_empty() {
    let loss = ViewConsistencyLoss::default();

    assert_eq!(loss.compute(&[], &[], None, 64, 64), 0.0);
}

#[test]
fn test_view_consistency_single_view() {
    let loss = ViewConsistencyLoss::default();
    let view = vec![0.5_f32; 64 * 64 * 3];

    // Single view has no pairs to compare
    let l = loss.compute(&[view], &[], None, 64, 64);
    assert_eq!(l, 0.0);
}

#[test]
fn test_view_consistency_identical_views() {
    let loss = ViewConsistencyLoss::default();
    let view = vec![0.5_f32; 64 * 64 * 3];

    let camera1 = oxigaf_flame::Camera::default_front(64, 64);
    let camera2 = oxigaf_flame::Camera::default_front(64, 64);

    let l = loss.compute(&[view.clone(), view], &[camera1, camera2], None, 64, 64);
    // Identical views should have low consistency loss
    assert!(
        l < 0.01,
        "identical views should have near-zero consistency loss, got {}",
        l
    );
}

// ---------------------------------------------------------------------------
// Temporal Consistency tests
// ---------------------------------------------------------------------------

#[test]
fn test_temporal_consistency_empty() {
    let tc = TemporalConsistency::default();

    assert_eq!(tc.compute(&[], &[], None), 0.0);
}

#[test]
fn test_temporal_consistency_no_previous() {
    let tc = TemporalConsistency::default();
    let current = vec![0.5_f32; 100];

    let l = tc.compute(&current, &[], None);
    assert_eq!(l, 0.0);
}

#[test]
fn test_temporal_consistency_identical_frames() {
    let tc = TemporalConsistency::default();
    let current = vec![0.5_f32; 100];
    let previous = vec![0.5_f32; 100];

    let l = tc.compute(&current, &[&previous], None);
    assert!(
        l.abs() < 1e-6,
        "identical frames should have ~0 consistency loss"
    );
}

#[test]
fn test_temporal_consistency_different_frames() {
    let tc = TemporalConsistency::default();
    let current = vec![0.0_f32; 100];
    let previous = vec![1.0_f32; 100];

    let l = tc.compute(&current, &[&previous], None);
    assert!(
        l > 0.0,
        "different frames should have positive consistency loss"
    );
}

// ---------------------------------------------------------------------------
// Generator state tests
// ---------------------------------------------------------------------------

#[test]
fn test_generator_initial_state() {
    let config = DiffusionTargetConfig::default();
    let gen = DiffusionTargetGenerator::new(config);

    assert!(!gen.is_loaded());
}

#[test]
fn test_generator_with_custom_config() {
    let config = DiffusionTargetConfig {
        warmup_iterations: 500,
        timestep_start: 800,
        timestep_end: 100,
        sds_weight: 0.7,
        ..Default::default()
    };
    let gen = DiffusionTargetGenerator::new(config);

    // Check warmup detection with custom config
    assert!(gen.is_warmup(499));
    assert!(!gen.is_warmup(500));

    // SDS weight should use custom value after warmup ramp
    let w = gen.sds_weight(1000);
    assert!((w - 0.7).abs() < 0.01, "expected ~0.7, got {}", w);
}

#[test]
fn test_sds_gradient_computation() {
    let config = DiffusionTargetConfig {
        warmup_iterations: 0,
        sds_weight: 1.0,
        ..Default::default()
    };
    let gen = DiffusionTargetGenerator::new(config);

    let rendered = vec![1.0_f32, 0.5, 0.3];
    let target = vec![0.8_f32, 0.5, 0.4];

    let grad = gen.compute_sds_gradient(&rendered, &target, 500);

    assert_eq!(grad.len(), 3);
    // Gradient should be proportional to (rendered - target)
    assert!(
        grad[0] > 0.0,
        "first gradient should be positive (rendered > target)"
    );
    assert!(
        grad[1].abs() < 1e-6,
        "second gradient should be ~0 (same value)"
    );
    assert!(
        grad[2] < 0.0,
        "third gradient should be negative (rendered < target)"
    );
}

// ---------------------------------------------------------------------------
// Edge case tests
// ---------------------------------------------------------------------------

#[test]
fn test_zero_warmup() {
    let config = DiffusionTargetConfig {
        warmup_iterations: 0,
        sds_weight: 1.0,
        ..Default::default()
    };
    let gen = DiffusionTargetGenerator::new(config);

    // No warmup, so never in warmup period
    assert!(!gen.is_warmup(0));

    // At iteration 0, ramp starts at 0 (adjusted = 0)
    assert_eq!(gen.sds_weight(0), 0.0);

    // At iteration 1, ramp should be positive
    assert!(gen.sds_weight(1) > 0.0);
}

#[test]
fn test_very_long_warmup() {
    let config = DiffusionTargetConfig {
        warmup_iterations: 1_000_000,
        sds_weight: 1.0,
        ..Default::default()
    };
    let gen = DiffusionTargetGenerator::new(config);

    // Should be in warmup for a very long time
    assert!(gen.is_warmup(999_999));
    assert!(!gen.is_warmup(1_000_000));

    // SDS weight should be 0 during warmup
    assert_eq!(gen.sds_weight(500_000), 0.0);
}

#[test]
fn test_equal_timestep_start_end() {
    // This should fail validation
    let config = DiffusionTargetConfig {
        timestep_start: 500,
        timestep_end: 500,
        ..Default::default()
    };

    assert!(config.validate().is_err());
}
