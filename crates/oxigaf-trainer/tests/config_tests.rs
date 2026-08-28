//! Unit tests for configuration validation.

use oxigaf_trainer::config::{
    DensityConfig, InitConfig, LossConfig, OptimizerConfig, TrainingConfig,
};
use oxigaf_trainer::TrainerError;

// ============================================================================
// TrainingConfig Tests
// ============================================================================

#[test]
fn default_config_is_valid() {
    let config = TrainingConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn zero_iterations_invalid() {
    let config = TrainingConfig {
        total_iterations: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(TrainerError::ParameterOutOfRange { .. })
    ));
}

#[test]
fn zero_views_per_step_invalid() {
    let config = TrainingConfig {
        views_per_step: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(TrainerError::ParameterOutOfRange { .. })
    ));
}

#[test]
fn density_control_start_after_end_invalid() {
    let config = TrainingConfig {
        density_control_start: 1000,
        density_control_end: 500, // Before start!
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(matches!(result, Err(TrainerError::InvalidConfig(_))));
}

// ============================================================================
// OptimizerConfig Tests
// ============================================================================

#[test]
fn default_optimizer_config_valid() {
    let config = OptimizerConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn negative_learning_rate_invalid() {
    let config = OptimizerConfig {
        lr_position: -0.001,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(TrainerError::ParameterOutOfRange { .. })
    ));
}

#[test]
fn zero_learning_rate_invalid() {
    let config = OptimizerConfig {
        lr_scale: 0.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn nan_learning_rate_invalid() {
    let config = OptimizerConfig {
        lr_opacity: f32::NAN,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn inf_learning_rate_invalid() {
    let config = OptimizerConfig {
        lr_sh: f32::INFINITY,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn beta1_zero_invalid() {
    let config = OptimizerConfig {
        beta1: 0.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
    assert!(matches!(
        result,
        Err(TrainerError::ParameterOutOfRange { .. })
    ));
}

#[test]
fn beta1_one_invalid() {
    let config = OptimizerConfig {
        beta1: 1.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn beta2_out_of_range_invalid() {
    let config = OptimizerConfig {
        beta2: 1.5,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn valid_custom_optimizer_config() {
    let config = OptimizerConfig {
        lr_position: 0.01,
        lr_position_final: 0.0001,
        lr_rotation: 0.001,
        lr_scale: 0.005,
        lr_opacity: 0.05,
        lr_sh: 0.0025,
        lr_offset: 0.0001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        position_lr_decay_steps: 10000,
    };
    assert!(config.validate().is_ok());
}

// ============================================================================
// LossConfig Tests
// ============================================================================

#[test]
fn default_loss_config_valid() {
    let config = LossConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn negative_weight_invalid() {
    let config = LossConfig {
        w_l1: -0.5,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn zero_weights_valid() {
    let config = LossConfig {
        w_l1: 0.0,
        w_ssim: 0.0,
        w_ms_ssim: 0.0,
        w_lpips: 0.0,
        w_position_reg: 0.0,
        w_scale_reg: 0.0,
        w_opacity_reg: 0.0,
        w_normal: 0.0,
        w_gradient_penalty: 0.0,
        gradient_penalty_threshold: 100.0,
        w_scale_reg_max_scale: oxigaf_trainer::loss::MAX_REASONABLE_WORLD_SCALE,
    };
    assert!(config.validate().is_ok());
}

#[test]
fn nan_weight_invalid() {
    let config = LossConfig {
        w_ssim: f32::NAN,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

// ============================================================================
// DensityConfig Tests
// ============================================================================

#[test]
fn default_density_config_valid() {
    let config = DensityConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn negative_grad_threshold_invalid() {
    let config = DensityConfig {
        grad_threshold: -0.001,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn min_opacity_out_of_range_invalid() {
    let config = DensityConfig {
        min_opacity: 1.5, // > 1
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());

    let config = DensityConfig {
        min_opacity: -0.1, // < 0
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn zero_max_gaussians_invalid() {
    let config = DensityConfig {
        max_gaussians: 0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn zero_max_screen_size_invalid() {
    let config = DensityConfig {
        max_screen_size: 0.0,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

// ============================================================================
// InitConfig Tests
// ============================================================================

#[test]
fn default_init_config_valid() {
    let config = InitConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn zero_gaussians_invalid() {
    let config = InitConfig {
        num_rigid: 0,
        num_flexible: 0,
        initial_scale: -5.0,
        initial_opacity: -2.0,
        sh_degree: 0,
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn only_rigid_valid() {
    let config = InitConfig {
        num_rigid: 1000,
        num_flexible: 0,
        initial_scale: -5.0,
        initial_opacity: -2.0,
        sh_degree: 0,
    };
    assert!(config.validate().is_ok());
}

#[test]
fn only_flexible_valid() {
    let config = InitConfig {
        num_rigid: 0,
        num_flexible: 500,
        initial_scale: -5.0,
        initial_opacity: -2.0,
        sh_degree: 0,
    };
    assert!(config.validate().is_ok());
}

#[test]
fn sh_degree_4_invalid() {
    let config = InitConfig {
        sh_degree: 4, // Max is 3
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn nan_initial_scale_invalid() {
    let config = InitConfig {
        initial_scale: f32::NAN,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn inf_initial_opacity_invalid() {
    let config = InitConfig {
        initial_opacity: f32::INFINITY,
        ..Default::default()
    };
    let result = config.validate();
    assert!(result.is_err());
}
