//! Unit tests for the Gaussian optimizer.
//!
//! Tests Adam optimizer correctness, learning rate scheduling, and density control bookkeeping.

use oxigaf_render::gaussian::{GaussianAttributes, GaussianModel};
use oxigaf_trainer::config::OptimizerConfig;
use oxigaf_trainer::optimizer::{GaussianOptimizer, Gradients};

/// Create a minimal test model.
fn make_model(n: usize, sh_degree: u32) -> GaussianModel {
    let sh_per = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
    GaussianModel {
        gaussians: vec![
            GaussianAttributes {
                position: [0.0; 3],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [-5.0; 3],
                opacity: 0.0,
            };
            n
        ],
        sh_coeffs: vec![0.0; n * sh_per],
        sh_degree,
        face_indices: vec![0; n],
        barycentric: vec![[1.0, 0.0, 0.0]; n],
        local_offsets: vec![[0.0; 3]; n],
        is_rigid: vec![true; n],
    }
}

#[test]
fn adam_converges_for_quadratic() {
    // Minimize f(x) = x^2 where x is a position component.
    // Starting at x = 5.0, should converge towards 0.
    let mut model = make_model(1, 0);
    model.gaussians[0].position[0] = 5.0;

    let config = OptimizerConfig {
        lr_position: 0.1,
        lr_position_final: 0.1, // No decay for this test
        position_lr_decay_steps: 1000,
        ..OptimizerConfig::default()
    };

    let mut optimizer = GaussianOptimizer::new(&config, &model);
    let sh_channels = 3; // (0+1)^2 * 3

    for t in 1..=200 {
        // Gradient of x^2 is 2x
        let grad = 2.0 * model.gaussians[0].position[0];
        let mut gradients = Gradients::zeros(1, sh_channels);
        gradients.position[0] = grad;

        optimizer.step(&mut model, &gradients, t);
    }

    // Should be close to 0
    let final_x = model.gaussians[0].position[0];
    assert!(final_x.abs() < 0.1, "Expected x near 0, got {final_x}");
}

#[test]
fn zero_gradient_no_change() {
    let mut model = make_model(1, 0);
    model.gaussians[0].position = [1.0, 2.0, 3.0];
    let original_position = model.gaussians[0].position;

    let config = OptimizerConfig::default();
    let mut optimizer = GaussianOptimizer::new(&config, &model);

    // Zero gradients
    let gradients = Gradients::zeros(1, 3);

    for t in 1..=10 {
        optimizer.step(&mut model, &gradients, t);
    }

    // Position should not change with zero gradients
    assert_eq!(model.gaussians[0].position, original_position);
}

#[test]
fn position_lr_schedule_starts_at_initial() {
    let config = OptimizerConfig {
        lr_position: 1.0e-3,
        lr_position_final: 1.0e-5,
        position_lr_decay_steps: 10_000,
        ..OptimizerConfig::default()
    };

    let model = make_model(1, 0);
    let optimizer = GaussianOptimizer::new(&config, &model);

    let lr_at_0 = optimizer.position_lr(0);
    assert!(
        (lr_at_0 - config.lr_position).abs() < 1e-8,
        "Expected LR {} at iteration 0, got {}",
        config.lr_position,
        lr_at_0
    );
}

#[test]
fn position_lr_schedule_ends_at_final() {
    let config = OptimizerConfig {
        lr_position: 1.0e-3,
        lr_position_final: 1.0e-5,
        position_lr_decay_steps: 10_000,
        ..OptimizerConfig::default()
    };

    let model = make_model(1, 0);
    let optimizer = GaussianOptimizer::new(&config, &model);

    let lr_at_end = optimizer.position_lr(10_000);
    assert!(
        (lr_at_end - config.lr_position_final).abs() < 1e-8,
        "Expected LR {} at decay_steps, got {}",
        config.lr_position_final,
        lr_at_end
    );
}

#[test]
fn position_lr_interpolates_log_linearly() {
    let config = OptimizerConfig {
        lr_position: 1.0e-3,
        lr_position_final: 1.0e-5,
        position_lr_decay_steps: 100,
        ..OptimizerConfig::default()
    };

    let model = make_model(1, 0);
    let optimizer = GaussianOptimizer::new(&config, &model);

    // At midpoint (t=0.5), log-linear interpolation means:
    // log(lr) = (1-t)*log(start) + t*log(end)
    // lr = exp((1-0.5)*log(1e-3) + 0.5*log(1e-5))
    //    = exp(0.5*log(1e-3) + 0.5*log(1e-5))
    //    = sqrt(1e-3 * 1e-5) = sqrt(1e-8) = 1e-4
    let lr_at_half = optimizer.position_lr(50);
    let expected = 1.0e-4;

    assert!(
        (lr_at_half - expected).abs() / expected < 0.01,
        "Expected LR ~{expected} at midpoint, got {lr_at_half}"
    );
}

#[test]
fn handle_densify_compacts_correctly() {
    let model = make_model(5, 0);
    let config = OptimizerConfig::default();
    let mut optimizer = GaussianOptimizer::new(&config, &model);

    // Simulate some optimizer state
    optimizer.position.m[3] = 1.0; // Index 1, component 0
    optimizer.position.m[4] = 2.0; // Index 1, component 1
    optimizer.position.m[5] = 3.0; // Index 1, component 2

    // Keep mask: keep indices 0, 2, 4 (remove 1, 3)
    let keep_mask = vec![true, false, true, false, true];
    let num_added = 2;

    optimizer.handle_densify(&keep_mask, num_added);

    // After compaction: 3 kept + 2 added = 5 Gaussians
    // Position state: 5 * 3 = 15 elements
    assert_eq!(optimizer.position.m.len(), 5 * 3);
    assert_eq!(optimizer.position.v.len(), 5 * 3);

    // New Gaussians should have zeroed state
    assert_eq!(optimizer.position.m[9], 0.0); // First new Gaussian
    assert_eq!(optimizer.position.m[12], 0.0); // Second new Gaussian
}

#[test]
fn handle_densify_extends_for_new_gaussians() {
    let model = make_model(3, 0);
    let config = OptimizerConfig::default();
    let mut optimizer = GaussianOptimizer::new(&config, &model);

    // Keep all, add 2
    let keep_mask = vec![true, true, true];
    let num_added = 2;

    optimizer.handle_densify(&keep_mask, num_added);

    // 3 kept + 2 added = 5
    assert_eq!(optimizer.position.m.len(), 5 * 3);
    assert_eq!(optimizer.rotation.m.len(), 5 * 4);
    assert_eq!(optimizer.scale.m.len(), 5 * 3);
    assert_eq!(optimizer.opacity.m.len(), 5);
}

#[test]
fn optimizer_checkpoint_and_restore() {
    let model = make_model(2, 0);
    let config = OptimizerConfig::default();
    let mut optimizer = GaussianOptimizer::new(&config, &model);

    // Modify some state
    optimizer.position.m[0] = 1.5;
    optimizer.position.v[0] = 0.1;
    optimizer.position.t = 10;

    // Checkpoint
    let states = optimizer.checkpoint_states();

    // Create new optimizer and restore
    let mut optimizer2 = GaussianOptimizer::new(&config, &model);
    optimizer2.restore_states(&states);

    // Verify restoration
    assert!((optimizer2.position.m[0] - 1.5).abs() < 1e-8);
    assert!((optimizer2.position.v[0] - 0.1).abs() < 1e-8);
    assert_eq!(optimizer2.position.t, 10);
}

#[test]
fn constant_gradient_ema() {
    // With constant gradients, verify EMA behavior:
    // m_t = beta1 * m_{t-1} + (1 - beta1) * g
    // After many steps with constant g, m should converge to g

    let mut model = make_model(1, 0);
    let config = OptimizerConfig {
        beta1: 0.9,
        beta2: 0.999,
        lr_position: 0.001,
        lr_position_final: 0.001,
        position_lr_decay_steps: 10000,
        ..OptimizerConfig::default()
    };
    let mut optimizer = GaussianOptimizer::new(&config, &model);

    let constant_grad = 1.0;
    let mut gradients = Gradients::zeros(1, 3);
    gradients.position[0] = constant_grad;

    for t in 1..=100 {
        optimizer.step(&mut model, &gradients, t);
    }

    // After 100 steps, the first moment m should be close to the constant gradient
    // m = (1 - beta1^100) / (1 - beta1) * (1 - beta1) * g ≈ g for large t
    let m = optimizer.position.m[0];
    assert!(
        (m - constant_grad).abs() < 0.1,
        "Expected m close to {constant_grad}, got {m}"
    );
}
