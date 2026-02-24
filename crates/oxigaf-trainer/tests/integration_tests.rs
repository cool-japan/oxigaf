//! Basic integration tests for oxigaf-trainer.

use oxigaf_trainer::config::{DensityConfig, LossConfig, OptimizerConfig, TrainingConfig};

#[test]
fn test_optimizer_config_default() {
    let config = OptimizerConfig::default();

    // Verify default values are reasonable
    assert!(config.lr_position > 0.0);
    assert!(config.lr_rotation > 0.0);
    assert!(config.lr_scale > 0.0);
    assert!(config.lr_opacity > 0.0);
    assert!(config.lr_sh > 0.0);
    assert!(config.beta1 >= 0.0 && config.beta1 < 1.0);
    assert!(config.beta2 >= 0.0 && config.beta2 < 1.0);
    assert!(config.epsilon > 0.0);
}

#[test]
fn test_loss_config_default() {
    let config = LossConfig::default();

    assert!(config.w_l1 >= 0.0);
    assert!(config.w_ssim >= 0.0);
    assert!(config.w_l1 + config.w_ssim > 0.0);
}

#[test]
fn test_density_config_default() {
    let config = DensityConfig::default();

    assert!(config.grad_threshold > 0.0);
    assert!(config.min_opacity > 0.0);
}

#[test]
fn test_training_config_default() {
    let config = TrainingConfig::default();

    assert!(config.total_iterations > 0);
    assert!(config.log_interval > 0);
    assert!(config.checkpoint_interval > 0);
    assert!(config.density_control_interval > 0);
}

#[test]
fn test_config_creation() {
    // Just verify all configs can be created
    let _ = OptimizerConfig::default();
    let _ = LossConfig::default();
    let _ = DensityConfig::default();
    let _ = TrainingConfig::default();
}
