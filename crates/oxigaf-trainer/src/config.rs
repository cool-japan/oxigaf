//! Training configuration types.
//!
//! All structs derive [`serde::Serialize`] / [`serde::Deserialize`] so they can
//! be loaded from TOML/JSON configuration files.

use serde::{Deserialize, Serialize};

use crate::tensorboard::TensorBoardConfig;
use crate::TrainerError;

// ---------------------------------------------------------------------------
// TrainingConfig (top-level)
// ---------------------------------------------------------------------------

/// Full training configuration — embeds optimizer, loss, density, and init
/// sub-configs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Total number of training iterations.
    pub total_iterations: u32,
    /// Number of camera views rendered per training step.
    pub views_per_step: usize,
    /// Run density control every N iterations.
    pub density_control_interval: u32,
    /// Iteration at which to start density control.
    pub density_control_start: u32,
    /// Iteration at which to stop density control.
    pub density_control_end: u32,
    /// Reset all opacities to a low value every N iterations.
    pub opacity_reset_interval: u32,
    /// Save a checkpoint every N iterations.
    pub checkpoint_interval: u32,
    /// Log metrics every N iterations.
    pub log_interval: u32,
    /// Classifier-free guidance scale at the start of training.
    pub guidance_scale_start: f32,
    /// Classifier-free guidance scale at the end of annealing.
    pub guidance_scale_end: f32,
    /// Number of iterations over which to anneal guidance scale.
    pub guidance_anneal_steps: u32,

    /// Per-parameter-group optimizer settings.
    pub optimizer: OptimizerConfig,
    /// Loss function weights.
    pub loss: LossConfig,
    /// Adaptive density control settings.
    pub density: DensityConfig,
    /// Gaussian initialization settings.
    pub init: InitConfig,
    /// TensorBoard logging configuration.
    pub tensorboard: TensorBoardConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            total_iterations: 15_000,
            views_per_step: 4,
            density_control_interval: 500,
            density_control_start: 1_000,
            density_control_end: 12_000,
            opacity_reset_interval: 3_000,
            checkpoint_interval: 1_000,
            log_interval: 50,
            guidance_scale_start: 7.5,
            guidance_scale_end: 3.0,
            guidance_anneal_steps: 10_000,
            optimizer: OptimizerConfig::default(),
            loss: LossConfig::default(),
            density: DensityConfig::default(),
            init: InitConfig::default(),
            tensorboard: TensorBoardConfig::default(),
        }
    }
}

impl TrainingConfig {
    /// Validate the training configuration for consistency and correctness.
    ///
    /// Returns an error if any parameter is out of valid range.
    pub fn validate(&self) -> Result<(), TrainerError> {
        // Total iterations must be positive
        if self.total_iterations == 0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "total_iterations".into(),
                value: "0".into(),
                expected: "> 0".into(),
            });
        }

        // Views per step must be positive
        if self.views_per_step == 0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "views_per_step".into(),
                value: "0".into(),
                expected: "> 0".into(),
            });
        }

        // Density control start must be before end
        if self.density_control_start > self.density_control_end {
            return Err(TrainerError::InvalidConfig(format!(
                "density_control_start ({}) must be <= density_control_end ({})",
                self.density_control_start, self.density_control_end
            )));
        }

        // Validate optimizer config
        self.optimizer.validate()?;

        // Validate loss config
        self.loss.validate()?;

        // Validate density config
        self.density.validate()?;

        // Validate init config
        self.init.validate()?;

        // Validate TensorBoard config
        self.tensorboard.validate()?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// OptimizerConfig
// ---------------------------------------------------------------------------

/// Per-parameter-group learning rates and Adam hyper-parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Initial learning rate for **position** (exponential decay).
    pub lr_position: f32,
    /// Final learning rate for position after decay.
    pub lr_position_final: f32,
    /// Learning rate for **rotation** quaternions.
    pub lr_rotation: f32,
    /// Learning rate for **log-scale** parameters.
    pub lr_scale: f32,
    /// Learning rate for **inverse-sigmoid opacity**.
    pub lr_opacity: f32,
    /// Learning rate for **SH coefficients**.
    pub lr_sh: f32,
    /// Learning rate for **local offsets** from the mesh surface.
    pub lr_offset: f32,
    /// Adam β₁.
    pub beta1: f32,
    /// Adam β₂.
    pub beta2: f32,
    /// Adam ε (numerical stability).
    pub epsilon: f32,
    /// Iteration at which position LR reaches `lr_position_final`.
    pub position_lr_decay_steps: u32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            lr_position: 1.6e-4,
            lr_position_final: 1.6e-6,
            lr_rotation: 1e-3,
            lr_scale: 5e-3,
            lr_opacity: 5e-2,
            lr_sh: 2.5e-3,
            lr_offset: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-15,
            position_lr_decay_steps: 30_000,
        }
    }
}

impl OptimizerConfig {
    /// Validate optimizer configuration.
    pub fn validate(&self) -> Result<(), TrainerError> {
        // Helper for checking finite positive values
        let check_positive_finite = |name: &str, value: f32| {
            if !value.is_finite() || value <= 0.0 {
                Err(TrainerError::ParameterOutOfRange {
                    param: name.into(),
                    value: format!("{value}"),
                    expected: "> 0 and finite".into(),
                })
            } else {
                Ok(())
            }
        };

        check_positive_finite("lr_position", self.lr_position)?;
        check_positive_finite("lr_position_final", self.lr_position_final)?;
        check_positive_finite("lr_rotation", self.lr_rotation)?;
        check_positive_finite("lr_scale", self.lr_scale)?;
        check_positive_finite("lr_opacity", self.lr_opacity)?;
        check_positive_finite("lr_sh", self.lr_sh)?;
        check_positive_finite("lr_offset", self.lr_offset)?;
        check_positive_finite("epsilon", self.epsilon)?;

        // Beta values must be in (0, 1)
        if self.beta1 <= 0.0 || self.beta1 >= 1.0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "beta1".into(),
                value: format!("{}", self.beta1),
                expected: "in (0, 1)".into(),
            });
        }

        if self.beta2 <= 0.0 || self.beta2 >= 1.0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "beta2".into(),
                value: format!("{}", self.beta2),
                expected: "in (0, 1)".into(),
            });
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LossConfig
// ---------------------------------------------------------------------------

/// Weights for the individual loss terms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossConfig {
    /// Weight for the L1 photometric loss.
    pub w_l1: f32,
    /// Weight for the SSIM structural loss (used as `1 − SSIM`).
    pub w_ssim: f32,
    /// Weight for Multi-Scale SSIM loss.
    pub w_ms_ssim: f32,
    /// Weight for LPIPS perceptual loss (requires VGG weights).
    pub w_lpips: f32,
    /// Weight for position regularisation (offset from mesh surface).
    pub w_position_reg: f32,
    /// Weight for scale regularisation (penalise extreme scales).
    pub w_scale_reg: f32,
    /// Weight for opacity regularisation (encourage binary opacity).
    pub w_opacity_reg: f32,
    /// Weight for normal consistency loss.
    pub w_normal: f32,
    /// Weight for gradient penalty (training stability).
    pub w_gradient_penalty: f32,
    /// Gradient norm threshold above which penalty is applied.
    pub gradient_penalty_threshold: f32,
}

impl Default for LossConfig {
    fn default() -> Self {
        Self {
            w_l1: 0.8,
            w_ssim: 0.2,
            w_ms_ssim: 0.0, // Disabled by default (optional)
            w_lpips: 0.0,   // Disabled by default (requires weights)
            w_position_reg: 0.01,
            w_scale_reg: 0.01,
            w_opacity_reg: 0.001,
            w_normal: 0.05,
            w_gradient_penalty: 0.0, // Disabled by default
            gradient_penalty_threshold: 100.0,
        }
    }
}

impl LossConfig {
    /// Create a config with LPIPS enabled.
    pub fn with_lpips(lpips_weight: f32) -> Self {
        Self {
            w_lpips: lpips_weight,
            ..Default::default()
        }
    }

    /// Create a config with MS-SSIM enabled.
    pub fn with_ms_ssim(ms_ssim_weight: f32) -> Self {
        Self {
            w_ms_ssim: ms_ssim_weight,
            ..Default::default()
        }
    }

    /// Create a config with perceptual losses enabled.
    pub fn perceptual(lpips_weight: f32, ms_ssim_weight: f32) -> Self {
        Self {
            w_lpips: lpips_weight,
            w_ms_ssim: ms_ssim_weight,
            w_l1: 0.6,
            w_ssim: 0.1,
            ..Default::default()
        }
    }

    /// Validate loss configuration.
    pub fn validate(&self) -> Result<(), TrainerError> {
        // Helper for checking non-negative finite weights
        let check_weight = |name: &str, value: f32| {
            if !value.is_finite() || value < 0.0 {
                Err(TrainerError::ParameterOutOfRange {
                    param: name.into(),
                    value: format!("{value}"),
                    expected: ">= 0 and finite".into(),
                })
            } else {
                Ok(())
            }
        };

        check_weight("w_l1", self.w_l1)?;
        check_weight("w_ssim", self.w_ssim)?;
        check_weight("w_ms_ssim", self.w_ms_ssim)?;
        check_weight("w_lpips", self.w_lpips)?;
        check_weight("w_position_reg", self.w_position_reg)?;
        check_weight("w_scale_reg", self.w_scale_reg)?;
        check_weight("w_opacity_reg", self.w_opacity_reg)?;
        check_weight("w_normal", self.w_normal)?;
        check_weight("w_gradient_penalty", self.w_gradient_penalty)?;
        check_weight(
            "gradient_penalty_threshold",
            self.gradient_penalty_threshold,
        )?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DensityConfig
// ---------------------------------------------------------------------------

/// Parameters governing adaptive density control (split / clone / prune).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityConfig {
    /// Mean-gradient threshold above which a Gaussian is considered for
    /// densification.
    pub grad_threshold: f32,
    /// Minimum sigmoid-opacity; Gaussians below this are pruned.
    pub min_opacity: f32,
    /// Maximum screen-space extent (pixels); Gaussians above this are pruned.
    pub max_screen_size: f32,
    /// Log-scale threshold: above → **split**, below → **clone**.
    pub split_scale_threshold: f32,
    /// Hard cap on the total number of Gaussians.
    pub max_gaussians: usize,
}

impl Default for DensityConfig {
    fn default() -> Self {
        Self {
            grad_threshold: 0.0002,
            min_opacity: 0.005,
            max_screen_size: 20.0,
            split_scale_threshold: 0.01,
            max_gaussians: 500_000,
        }
    }
}

impl DensityConfig {
    /// Validate density control configuration.
    pub fn validate(&self) -> Result<(), TrainerError> {
        if !self.grad_threshold.is_finite() || self.grad_threshold < 0.0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "grad_threshold".into(),
                value: format!("{}", self.grad_threshold),
                expected: ">= 0 and finite".into(),
            });
        }

        if !self.min_opacity.is_finite() || self.min_opacity < 0.0 || self.min_opacity > 1.0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "min_opacity".into(),
                value: format!("{}", self.min_opacity),
                expected: "in [0, 1]".into(),
            });
        }

        if !self.max_screen_size.is_finite() || self.max_screen_size <= 0.0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "max_screen_size".into(),
                value: format!("{}", self.max_screen_size),
                expected: "> 0 and finite".into(),
            });
        }

        if self.max_gaussians == 0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "max_gaussians".into(),
                value: "0".into(),
                expected: "> 0".into(),
            });
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// InitConfig
// ---------------------------------------------------------------------------

/// Settings for initial Gaussian placement on the FLAME mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitConfig {
    /// Number of Gaussians bound **rigidly** to the mesh (head / scalp).
    pub num_rigid: usize,
    /// Number of **flexible** (deformable) Gaussians (jaw, eyes).
    pub num_flexible: usize,
    /// Initial log-scale value (`exp(−5) ≈ 0.0067`).
    pub initial_scale: f32,
    /// Initial inverse-sigmoid opacity (`σ(−2) ≈ 0.12`).
    pub initial_opacity: f32,
    /// SH degree (0–3).
    pub sh_degree: u32,
}

impl Default for InitConfig {
    fn default() -> Self {
        Self {
            num_rigid: 50_000,
            num_flexible: 50_000,
            initial_scale: -5.0,
            initial_opacity: -2.0,
            sh_degree: 3,
        }
    }
}

impl InitConfig {
    /// Validate initialization configuration.
    pub fn validate(&self) -> Result<(), TrainerError> {
        // Need at least some Gaussians
        if self.num_rigid == 0 && self.num_flexible == 0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "num_rigid + num_flexible".into(),
                value: "0".into(),
                expected: "> 0".into(),
            });
        }

        // Scale must be finite
        if !self.initial_scale.is_finite() {
            return Err(TrainerError::ParameterOutOfRange {
                param: "initial_scale".into(),
                value: format!("{}", self.initial_scale),
                expected: "finite".into(),
            });
        }

        // Opacity must be finite
        if !self.initial_opacity.is_finite() {
            return Err(TrainerError::ParameterOutOfRange {
                param: "initial_opacity".into(),
                value: format!("{}", self.initial_opacity),
                expected: "finite".into(),
            });
        }

        // SH degree must be 0-3
        if self.sh_degree > 3 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "sh_degree".into(),
                value: format!("{}", self.sh_degree),
                expected: "in [0, 3]".into(),
            });
        }

        Ok(())
    }
}
