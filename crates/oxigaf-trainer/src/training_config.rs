//! Training configuration presets for the OxiGAF optimization loop.
//!
//! Provides named training profile presets (Development, Standard, Production)
//! and a [`TrainingProfileConfig`] struct that holds all hyperparameters for
//! one training run.
//!
//! # Quick Start
//!
//! ```rust
//! use oxigaf_trainer::training_config::{TrainingProfile, TrainingProfileConfig};
//!
//! // Use a preset
//! let profile = TrainingProfile::standard();
//! let cfg = profile.config();
//!
//! // Validate and inspect resource estimates
//! cfg.validate().expect("standard config must be valid");
//! let mem_mb = cfg.total_memory_estimate_mb();
//! let hours  = cfg.estimated_training_hours(50.0); // 50 ms/iter
//! println!("{}", cfg.format_summary());
//! ```

use std::fmt::Write as FmtWrite;

use crate::config::{DensityConfig, InitConfig, LossConfig, OptimizerConfig, TrainingConfig};
use crate::TrainerError;

// ---------------------------------------------------------------------------
// TrainingProfileConfig
// ---------------------------------------------------------------------------

/// Full, flat training configuration for one OxiGAF training run.
///
/// All public fields have documented defaults for each named preset.  Use
/// [`TrainingProfile::config`] to get a preset instance, then mutate fields
/// freely before passing the config to the trainer.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingProfileConfig {
    // Training loop
    /// Maximum number of training iterations.
    /// dev: 1_000, std: 10_000, prod: 30_000
    pub max_iterations: u32,
    /// Number of warm-up iterations before full LR / loss scheduling takes effect.
    /// dev: 100, std: 500, prod: 1_000
    pub warmup_iterations: u32,
    /// Render target height in pixels.
    /// dev: 128, std: 256, prod: 512
    pub image_height: u32,
    /// Render target width in pixels.
    /// dev: 128, std: 256, prod: 512
    pub image_width: u32,
    /// Number of camera views processed per optimizer step.
    /// dev: 1, std: 2, prod: 4
    pub views_per_step: u32,

    // Gaussians
    /// Number of Gaussians at initialisation.
    /// dev: 1_000, std: 5_000, prod: 10_000
    pub initial_num_gaussians: u32,
    /// Spherical-harmonics degree (0 = colour only).
    /// dev: 0, std: 1, prod: 3
    pub sh_degree: u32,
    /// Hard upper limit on total Gaussians after densification.
    /// dev: 10_000, std: 50_000, prod: 200_000
    pub max_num_gaussians: u32,

    // Loss weights
    /// Weight for the photometric (L1) loss term.
    pub photometric_weight: f32,
    /// Weight for the SSIM structural loss term.
    pub ssim_weight: f32,
    /// Weight for the LPIPS perceptual loss term.
    /// dev: 0.0, std: 0.1, prod: 0.5
    pub lpips_weight: f32,
    /// L2 regularisation weight on Gaussian positions.
    pub position_reg_weight: f32,
    /// L2 regularisation weight on Gaussian scales.
    pub scale_reg_weight: f32,
    /// L2 regularisation weight on Gaussian opacities.
    pub opacity_reg_weight: f32,

    // Optimizer learning rates
    /// Learning rate for Gaussian positions.
    pub lr_position: f64,
    /// Learning rate for Gaussian scales.
    pub lr_scale: f64,
    /// Learning rate for Gaussian rotations.
    pub lr_rotation: f64,
    /// Learning rate for Gaussian opacities.
    pub lr_opacity: f64,
    /// Learning rate for spherical-harmonics coefficients.
    pub lr_sh: f64,

    // Density control
    /// Iteration at which adaptive densification begins.
    /// dev: 100, std: 500, prod: 500
    pub densify_from_iter: u32,
    /// Iteration at which adaptive densification stops.
    /// dev: 800, std: 8_000, prod: 20_000
    pub densify_until_iter: u32,
    /// Gradient-magnitude threshold for triggering a clone/split operation.
    pub densify_grad_threshold: f32,
    /// How often (in iterations) to reset low-opacity Gaussians.
    /// dev: 300, std: 3_000, prod: 3_000
    pub opacity_reset_interval: u32,

    // Checkpointing / logging
    /// Save a checkpoint every this many iterations.
    /// dev: 500, std: 1_000, prod: 2_000
    pub checkpoint_interval: u32,
    /// Log metrics every this many iterations.
    /// dev: 10, std: 50, prod: 100
    pub log_interval: u32,

    // Features
    /// Whether to maintain an Exponential Moving Average of model weights.
    /// dev: false, std: true, prod: true
    pub use_ema: bool,
    /// Whether to include the LPIPS loss in the objective.
    /// dev: false, std: false, prod: true
    pub use_lpips: bool,
}

impl TrainingProfileConfig {
    // ------------------------------------------------------------------
    // Validation
    // ------------------------------------------------------------------

    /// Validate all fields for internal consistency.
    ///
    /// Returns [`TrainerError::InvalidConfig`] if any constraint is violated.
    pub fn validate(&self) -> Result<(), TrainerError> {
        // Image dimensions
        if self.image_height == 0 {
            return Err(TrainerError::InvalidConfig(
                "image_height must be > 0".to_string(),
            ));
        }
        if self.image_width == 0 {
            return Err(TrainerError::InvalidConfig(
                "image_width must be > 0".to_string(),
            ));
        }

        // Iteration counts
        if self.max_iterations == 0 {
            return Err(TrainerError::InvalidConfig(
                "max_iterations must be > 0".to_string(),
            ));
        }
        if self.views_per_step == 0 {
            return Err(TrainerError::InvalidConfig(
                "views_per_step must be > 0".to_string(),
            ));
        }
        if self.initial_num_gaussians == 0 {
            return Err(TrainerError::InvalidConfig(
                "initial_num_gaussians must be > 0".to_string(),
            ));
        }
        if self.max_num_gaussians < self.initial_num_gaussians {
            return Err(TrainerError::InvalidConfig(format!(
                "max_num_gaussians ({}) must be >= initial_num_gaussians ({})",
                self.max_num_gaussians, self.initial_num_gaussians
            )));
        }
        if self.densify_until_iter <= self.densify_from_iter {
            return Err(TrainerError::InvalidConfig(format!(
                "densify_until_iter ({}) must be > densify_from_iter ({})",
                self.densify_until_iter, self.densify_from_iter
            )));
        }

        // Loss weights (must be non-negative)
        if self.photometric_weight < 0.0 {
            return Err(TrainerError::InvalidConfig(
                "photometric_weight must be >= 0".to_string(),
            ));
        }
        if self.ssim_weight < 0.0 {
            return Err(TrainerError::InvalidConfig(
                "ssim_weight must be >= 0".to_string(),
            ));
        }
        if self.lpips_weight < 0.0 {
            return Err(TrainerError::InvalidConfig(
                "lpips_weight must be >= 0".to_string(),
            ));
        }
        if self.position_reg_weight < 0.0 {
            return Err(TrainerError::InvalidConfig(
                "position_reg_weight must be >= 0".to_string(),
            ));
        }
        if self.scale_reg_weight < 0.0 {
            return Err(TrainerError::InvalidConfig(
                "scale_reg_weight must be >= 0".to_string(),
            ));
        }
        if self.opacity_reg_weight < 0.0 {
            return Err(TrainerError::InvalidConfig(
                "opacity_reg_weight must be >= 0".to_string(),
            ));
        }

        // Gradient threshold
        if self.densify_grad_threshold <= 0.0 {
            return Err(TrainerError::InvalidConfig(
                "densify_grad_threshold must be > 0".to_string(),
            ));
        }

        // Learning rates (must be strictly positive)
        if self.lr_position <= 0.0 {
            return Err(TrainerError::InvalidConfig(
                "lr_position must be > 0".to_string(),
            ));
        }
        if self.lr_scale <= 0.0 {
            return Err(TrainerError::InvalidConfig(
                "lr_scale must be > 0".to_string(),
            ));
        }
        if self.lr_rotation <= 0.0 {
            return Err(TrainerError::InvalidConfig(
                "lr_rotation must be > 0".to_string(),
            ));
        }
        if self.lr_opacity <= 0.0 {
            return Err(TrainerError::InvalidConfig(
                "lr_opacity must be > 0".to_string(),
            ));
        }
        if self.lr_sh <= 0.0 {
            return Err(TrainerError::InvalidConfig("lr_sh must be > 0".to_string()));
        }

        // SH degree: the renderer allocates `(d+1)^2*3` SH channels per
        // Gaussian; mirrors the identical bound in
        // `progressive_training::TrainingStage::validate`.
        if self.sh_degree > 3 {
            return Err(TrainerError::InvalidConfig(format!(
                "sh_degree ({}) must be <= 3",
                self.sh_degree
            )));
        }

        // Warmup must fit inside the run it's warming up.
        if self.warmup_iterations > self.max_iterations {
            return Err(TrainerError::InvalidConfig(format!(
                "warmup_iterations ({}) must be <= max_iterations ({})",
                self.warmup_iterations, self.max_iterations
            )));
        }

        // The trainer gates these with `iteration.is_multiple_of(interval)`,
        // which for `interval == 0` is true only at iteration 0 — a zero
        // interval silently disables checkpointing/logging/opacity-reset for
        // the entire run instead of erroring.
        if self.checkpoint_interval == 0 {
            return Err(TrainerError::InvalidConfig(
                "checkpoint_interval must be > 0 (0 silently disables checkpointing)".to_string(),
            ));
        }
        if self.log_interval == 0 {
            return Err(TrainerError::InvalidConfig(
                "log_interval must be > 0 (0 silently disables logging)".to_string(),
            ));
        }
        if self.opacity_reset_interval == 0 {
            return Err(TrainerError::InvalidConfig(
                "opacity_reset_interval must be > 0 (0 silently disables opacity reset)"
                    .to_string(),
            ));
        }

        // A nonzero `lpips_weight` with `use_lpips = false` is internally
        // inconsistent: the weight implies LPIPS should contribute to the
        // loss, but the flag means it never actually does (see
        // `TrainingProfile::Standard`, which had exactly this shape before
        // being fixed alongside this check).
        if self.lpips_weight > 0.0 && !self.use_lpips {
            return Err(TrainerError::InvalidConfig(format!(
                "lpips_weight ({}) is > 0 but use_lpips is false",
                self.lpips_weight
            )));
        }

        Ok(())
    }

    // ------------------------------------------------------------------
    // Resource estimates
    // ------------------------------------------------------------------

    /// Rough memory footprint estimate in MiB.
    ///
    /// Formula (very approximate):
    /// ```text
    /// (initial_num_gaussians * 200 + image_height * image_width * views_per_step * 16) / 2^20
    /// ```
    ///
    /// All arithmetic is performed in `f32` to avoid integer overflow at
    /// large resolutions.
    pub fn total_memory_estimate_mb(&self) -> f32 {
        let gaussian_bytes = self.initial_num_gaussians as f32 * 200.0_f32;
        let pixel_bytes = self.image_height as f32
            * self.image_width as f32
            * self.views_per_step as f32
            * 16.0_f32;
        (gaussian_bytes + pixel_bytes) / (1024.0_f32 * 1024.0_f32)
    }

    /// Estimated wall-clock training duration in hours.
    ///
    /// `ms_per_iter` — average measured time per iteration in milliseconds.
    pub fn estimated_training_hours(&self, ms_per_iter: f32) -> f32 {
        self.max_iterations as f32 * ms_per_iter / 3_600_000.0_f32
    }

    // ------------------------------------------------------------------
    // Formatting
    // ------------------------------------------------------------------

    /// Generate a compact multi-line summary table of the most important
    /// configuration parameters.
    pub fn format_summary(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "┌─────────────────────────────────────────┐");
        let _ = writeln!(out, "│       OxiGAF Training Profile Summary    │");
        let _ = writeln!(out, "├──────────────────────────┬──────────────┤");
        let _ = writeln!(
            out,
            "│ {:<24} │ {:>12} │",
            "max_iterations", self.max_iterations
        );
        let _ = writeln!(
            out,
            "│ {:<24} │ {:>12} │",
            "warmup_iterations", self.warmup_iterations
        );
        let _ = writeln!(
            out,
            "│ {:<24} │ {:>12} │",
            "image_size",
            format!("{}x{}", self.image_width, self.image_height)
        );
        let _ = writeln!(
            out,
            "│ {:<24} │ {:>12} │",
            "views_per_step", self.views_per_step
        );
        let _ = writeln!(
            out,
            "│ {:<24} │ {:>12} │",
            "initial_gaussians", self.initial_num_gaussians
        );
        let _ = writeln!(
            out,
            "│ {:<24} │ {:>12} │",
            "max_gaussians", self.max_num_gaussians
        );
        let _ = writeln!(out, "│ {:<24} │ {:>12} │", "sh_degree", self.sh_degree);
        let _ = writeln!(out, "├──────────────────────────┼──────────────┤");
        let _ = writeln!(
            out,
            "│ {:<24} │ {:>12.3} │",
            "photometric_weight", self.photometric_weight
        );
        let _ = writeln!(
            out,
            "│ {:<24} │ {:>12.3} │",
            "ssim_weight", self.ssim_weight
        );
        let _ = writeln!(
            out,
            "│ {:<24} │ {:>12.3} │",
            "lpips_weight", self.lpips_weight
        );
        let _ = writeln!(out, "├──────────────────────────┼──────────────┤");
        let _ = writeln!(
            out,
            "│ {:<24} │ {:>12.2e} │",
            "lr_position", self.lr_position
        );
        let _ = writeln!(out, "│ {:<24} │ {:>12.2e} │", "lr_scale", self.lr_scale);
        let _ = writeln!(
            out,
            "│ {:<24} │ {:>12.2e} │",
            "lr_rotation", self.lr_rotation
        );
        let _ = writeln!(out, "│ {:<24} │ {:>12.2e} │", "lr_opacity", self.lr_opacity);
        let _ = writeln!(out, "│ {:<24} │ {:>12.2e} │", "lr_sh", self.lr_sh);
        let _ = writeln!(out, "├──────────────────────────┼──────────────┤");
        let _ = writeln!(out, "│ {:<24} │ {:>12} │", "use_ema", self.use_ema);
        let _ = writeln!(out, "│ {:<24} │ {:>12} │", "use_lpips", self.use_lpips);
        let mem = self.total_memory_estimate_mb();
        let _ = writeln!(out, "│ {:<24} │ {:>9.1} MiB │", "est. memory", mem);
        let _ = writeln!(out, "└──────────────────────────┴──────────────┘");
        out
    }
}

impl From<TrainingProfileConfig> for TrainingConfig {
    /// Convert a flat preset config into the nested [`TrainingConfig`] that
    /// the trainer actually accepts, so a [`TrainingProfile`] preset can
    /// drive it directly instead of requiring a hand-written field-by-field
    /// copy at every call site.
    ///
    /// Every optimiser/loss/density/init field with a direct counterpart is
    /// carried over (learning rates narrow `f64 -> f32`; `sh_degree` and the
    /// checkpoint/log/opacity-reset intervals map 1:1). `TrainingConfig` has
    /// no equivalent for some `TrainingProfileConfig` fields, so this
    /// conversion is necessarily lossy for:
    /// - `image_height` / `image_width` — render resolution isn't part of
    ///   `TrainingConfig`; it's supplied to the renderer separately.
    /// - `warmup_iterations` — no LR/loss-schedule warmup concept exists on
    ///   `TrainingConfig`.
    /// - `use_ema` — the current trainer has no EMA support.
    ///
    /// `init.num_rigid` / `init.num_flexible` have no split in
    /// `TrainingProfileConfig` (only a single `initial_num_gaussians`), so
    /// the total is divided evenly, matching [`InitConfig::default`]'s own
    /// 50/50 split. Fields with no source at all (`guidance_*`,
    /// `density_control_interval`, `lr_position_final`, `lr_offset`, Adam
    /// betas/epsilon, `min_opacity`/`max_screen_size`/
    /// `split_scale_threshold`, `initial_scale`/`initial_opacity`,
    /// `tensorboard`, `precision`, `enable_profiling`) keep their
    /// `TrainingConfig` defaults.
    fn from(profile: TrainingProfileConfig) -> Self {
        let total_gaussians = profile.initial_num_gaussians as usize;
        let num_rigid = total_gaussians / 2;
        let num_flexible = total_gaussians - num_rigid;

        TrainingConfig {
            total_iterations: profile.max_iterations,
            views_per_step: profile.views_per_step as usize,
            density_control_start: profile.densify_from_iter,
            density_control_end: profile.densify_until_iter,
            opacity_reset_interval: profile.opacity_reset_interval,
            checkpoint_interval: profile.checkpoint_interval,
            log_interval: profile.log_interval,
            optimizer: OptimizerConfig {
                lr_position: profile.lr_position as f32,
                lr_rotation: profile.lr_rotation as f32,
                lr_scale: profile.lr_scale as f32,
                lr_opacity: profile.lr_opacity as f32,
                lr_sh: profile.lr_sh as f32,
                ..Default::default()
            },
            loss: LossConfig {
                w_l1: profile.photometric_weight,
                w_ssim: profile.ssim_weight,
                w_lpips: profile.lpips_weight,
                w_position_reg: profile.position_reg_weight,
                w_scale_reg: profile.scale_reg_weight,
                w_opacity_reg: profile.opacity_reg_weight,
                ..Default::default()
            },
            density: DensityConfig {
                grad_threshold: profile.densify_grad_threshold,
                max_gaussians: profile.max_num_gaussians as usize,
                ..Default::default()
            },
            init: InitConfig {
                num_rigid,
                num_flexible,
                sh_degree: profile.sh_degree,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// TrainingProfile
// ---------------------------------------------------------------------------

/// Named training profile preset.
///
/// Choose a preset appropriate for your workflow:
/// - [`Development`](TrainingProfile::Development): fast iteration, no LPIPS, low resolution.
/// - [`Standard`](TrainingProfile::Standard): balanced quality and speed.
/// - [`Production`](TrainingProfile::Production): full quality, all losses, maximum iterations.
/// - [`Custom`](TrainingProfile::Custom): supply your own [`TrainingProfileConfig`].
#[derive(Debug, Clone, PartialEq)]
pub enum TrainingProfile {
    /// Fast iteration for development (fewer steps, no LPIPS, lower resolution).
    Development,
    /// Balanced quality and speed.
    Standard,
    /// Full quality for final output (all losses, max iterations).
    Production,
    /// Custom profile (user-specified).
    Custom(Box<TrainingProfileConfig>),
}

impl TrainingProfile {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Return the Development preset.
    pub fn development() -> Self {
        Self::Development
    }

    /// Return the Standard preset.
    pub fn standard() -> Self {
        Self::Standard
    }

    /// Return the Production preset.
    pub fn production() -> Self {
        Self::Production
    }

    // ------------------------------------------------------------------
    // Config materialisation
    // ------------------------------------------------------------------

    /// Return the [`TrainingProfileConfig`] for this profile.
    ///
    /// For named presets a freshly-constructed config is returned.
    /// For `Custom` the inner config is cloned.
    pub fn config(&self) -> TrainingProfileConfig {
        match self {
            TrainingProfile::Development => TrainingProfileConfig {
                // Training loop
                max_iterations: 1_000,
                warmup_iterations: 100,
                image_height: 128,
                image_width: 128,
                views_per_step: 1,
                // Gaussians
                initial_num_gaussians: 1_000,
                sh_degree: 0,
                max_num_gaussians: 10_000,
                // Loss weights
                photometric_weight: 1.0,
                ssim_weight: 0.1,
                lpips_weight: 0.0,
                position_reg_weight: 1e-4,
                scale_reg_weight: 1e-3,
                opacity_reg_weight: 1e-2,
                // Optimizer
                lr_position: 1.6e-4,
                lr_scale: 5e-3,
                lr_rotation: 1e-3,
                lr_opacity: 5e-2,
                lr_sh: 2.5e-3,
                // Density control
                densify_from_iter: 100,
                densify_until_iter: 800,
                densify_grad_threshold: 2e-4,
                opacity_reset_interval: 300,
                // Checkpointing
                checkpoint_interval: 500,
                log_interval: 10,
                // Features
                use_ema: false,
                use_lpips: false,
            },
            TrainingProfile::Standard => TrainingProfileConfig {
                // Training loop
                max_iterations: 10_000,
                warmup_iterations: 500,
                image_height: 256,
                image_width: 256,
                views_per_step: 2,
                // Gaussians
                initial_num_gaussians: 5_000,
                sh_degree: 1,
                max_num_gaussians: 50_000,
                // Loss weights
                photometric_weight: 1.0,
                ssim_weight: 0.2,
                lpips_weight: 0.1,
                position_reg_weight: 1e-4,
                scale_reg_weight: 1e-3,
                opacity_reg_weight: 1e-2,
                // Optimizer
                lr_position: 1.6e-4,
                lr_scale: 5e-3,
                lr_rotation: 1e-3,
                lr_opacity: 5e-2,
                lr_sh: 2.5e-3,
                // Density control
                densify_from_iter: 500,
                densify_until_iter: 8_000,
                densify_grad_threshold: 2e-4,
                opacity_reset_interval: 3_000,
                // Checkpointing
                checkpoint_interval: 1_000,
                log_interval: 50,
                // Features
                use_ema: true,
                // `lpips_weight: 0.1` above only takes effect if this is
                // true; previously `false` here silently zeroed out LPIPS
                // for the "Standard" profile despite its nonzero weight
                // (an inconsistency `validate()` now rejects outright).
                use_lpips: true,
            },
            TrainingProfile::Production => TrainingProfileConfig {
                // Training loop
                max_iterations: 30_000,
                warmup_iterations: 1_000,
                image_height: 512,
                image_width: 512,
                views_per_step: 4,
                // Gaussians
                initial_num_gaussians: 10_000,
                sh_degree: 3,
                max_num_gaussians: 200_000,
                // Loss weights
                photometric_weight: 1.0,
                ssim_weight: 0.2,
                lpips_weight: 0.5,
                position_reg_weight: 1e-5,
                scale_reg_weight: 1e-4,
                opacity_reg_weight: 1e-3,
                // Optimizer
                lr_position: 1.6e-4,
                lr_scale: 5e-3,
                lr_rotation: 1e-3,
                lr_opacity: 5e-2,
                lr_sh: 2.5e-3,
                // Density control
                densify_from_iter: 500,
                densify_until_iter: 20_000,
                densify_grad_threshold: 2e-4,
                opacity_reset_interval: 3_000,
                // Checkpointing
                checkpoint_interval: 2_000,
                log_interval: 100,
                // Features
                use_ema: true,
                use_lpips: true,
            },
            TrainingProfile::Custom(cfg) => *cfg.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------ preset values

    #[test]
    fn test_development_profile() {
        let cfg = TrainingProfile::development().config();
        assert_eq!(cfg.max_iterations, 1_000);
        assert_eq!(cfg.image_height, 128);
        assert_eq!(cfg.image_width, 128);
        assert_eq!(cfg.sh_degree, 0);
        assert_eq!(cfg.initial_num_gaussians, 1_000);
        assert!(!cfg.use_ema);
        assert!(!cfg.use_lpips);
        assert!((cfg.lpips_weight - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_standard_profile() {
        let cfg = TrainingProfile::standard().config();
        assert_eq!(cfg.max_iterations, 10_000);
        assert_eq!(cfg.image_height, 256);
        assert_eq!(cfg.image_width, 256);
        assert_eq!(cfg.sh_degree, 1);
        assert_eq!(cfg.initial_num_gaussians, 5_000);
        assert!(cfg.use_ema);
        // `use_lpips` must agree with the nonzero `lpips_weight` this
        // preset sets — see the fix note at the `use_lpips: true` site.
        assert!(cfg.use_lpips);
        assert!(cfg.lpips_weight > 0.0);
    }

    #[test]
    fn test_production_profile() {
        let cfg = TrainingProfile::production().config();
        assert_eq!(cfg.max_iterations, 30_000);
        assert_eq!(cfg.image_height, 512);
        assert_eq!(cfg.image_width, 512);
        assert_eq!(cfg.sh_degree, 3);
        assert_eq!(cfg.initial_num_gaussians, 10_000);
        assert!(cfg.use_ema);
        assert!(cfg.use_lpips);
        assert!((cfg.lpips_weight - 0.5).abs() < 1e-6);
    }

    // ------------------------------------------------------------------ validate

    #[test]
    fn test_validate_valid_config() {
        let cfg = TrainingProfile::standard().config();
        assert!(cfg.validate().is_ok(), "standard config must be valid");
    }

    #[test]
    fn test_validate_invalid_lr_zero() {
        let mut cfg = TrainingProfile::development().config();
        cfg.lr_position = 0.0;
        let result = cfg.validate();
        assert!(result.is_err(), "lr_position = 0 must be invalid");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("lr_position"),
            "error must mention lr_position, got: {msg}"
        );
    }

    #[test]
    fn test_validate_invalid_lr_negative() {
        let mut cfg = TrainingProfile::development().config();
        cfg.lr_scale = -1e-3;
        let result = cfg.validate();
        assert!(result.is_err(), "negative lr_scale must be invalid");
    }

    #[test]
    fn test_validate_invalid_image_height_zero() {
        let mut cfg = TrainingProfile::development().config();
        cfg.image_height = 0;
        let result = cfg.validate();
        assert!(result.is_err(), "image_height = 0 must be invalid");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("image_height"),
            "error must mention image_height, got: {msg}"
        );
    }

    #[test]
    fn test_validate_invalid_image_width_zero() {
        let mut cfg = TrainingProfile::development().config();
        cfg.image_width = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_max_gaussians_less_than_initial() {
        let mut cfg = TrainingProfile::development().config();
        cfg.max_num_gaussians = 500; // less than initial_num_gaussians = 1000
        assert!(cfg.validate().is_err(), "max < initial must be invalid");
    }

    #[test]
    fn test_validate_densify_until_not_after_from() {
        let mut cfg = TrainingProfile::development().config();
        cfg.densify_until_iter = cfg.densify_from_iter; // equal, not strictly greater
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_sh_degree_too_high() {
        let mut cfg = TrainingProfile::development().config();
        cfg.sh_degree = 4; // renderer only supports 0..=3
        let result = cfg.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("sh_degree"));
    }

    #[test]
    fn test_validate_warmup_exceeds_max_iterations() {
        let mut cfg = TrainingProfile::development().config();
        cfg.warmup_iterations = cfg.max_iterations + 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_intervals_rejected() {
        for zero_field in ["checkpoint", "log", "opacity_reset"] {
            let mut cfg = TrainingProfile::development().config();
            match zero_field {
                "checkpoint" => cfg.checkpoint_interval = 0,
                "log" => cfg.log_interval = 0,
                _ => cfg.opacity_reset_interval = 0,
            }
            assert!(
                cfg.validate().is_err(),
                "{zero_field}_interval = 0 must be rejected (it silently disables the schedule, not runs it once)"
            );
        }
    }

    #[test]
    fn test_validate_lpips_weight_without_use_lpips_rejected() {
        let mut cfg = TrainingProfile::development().config();
        cfg.lpips_weight = 0.1;
        cfg.use_lpips = false;
        let result = cfg.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("lpips_weight"));
    }

    #[test]
    fn test_all_presets_are_internally_consistent() {
        // Regression: `TrainingProfile::Standard` used to pair a nonzero
        // `lpips_weight` with `use_lpips = false`; every built-in preset
        // must now pass its own `validate()`.
        for profile in [
            TrainingProfile::development(),
            TrainingProfile::standard(),
            TrainingProfile::production(),
        ] {
            let cfg = profile.config();
            assert!(cfg.validate().is_ok(), "{cfg:?} failed validate()");
        }
    }

    // ------------------------------------------------------------------ resource estimates

    #[test]
    fn test_memory_estimate_positive() {
        let cfg = TrainingProfile::production().config();
        let mem = cfg.total_memory_estimate_mb();
        assert!(mem > 0.0, "memory estimate must be positive, got {mem}");
    }

    #[test]
    fn test_memory_estimate_no_overflow() {
        // Ensure no integer overflow at production scales.
        let cfg = TrainingProfile::production().config();
        let mem = cfg.total_memory_estimate_mb();
        // Production: 10_000 * 200 + 512 * 512 * 4 * 16 = 2_000_000 + 16_777_216 bytes
        // = ~17.9 MiB
        assert!(
            mem.is_finite() && mem < 1_000.0,
            "memory estimate must be finite and sane, got {mem}"
        );
    }

    #[test]
    fn test_training_hours_estimate() {
        let cfg = TrainingProfile::standard().config(); // 10_000 iterations
        let hours = cfg.estimated_training_hours(36.0); // 36 ms/iter
                                                        // 10_000 * 36 ms = 360_000 ms = 0.1 hours
        let expected = 10_000.0_f32 * 36.0 / 3_600_000.0;
        assert!(
            (hours - expected).abs() < 1e-6,
            "expected {expected}, got {hours}"
        );
    }

    #[test]
    fn test_training_hours_estimate_zero_ms() {
        let cfg = TrainingProfile::development().config();
        let hours = cfg.estimated_training_hours(0.0);
        assert!((hours - 0.0).abs() < 1e-9, "0 ms/iter → 0 hours");
    }

    // ------------------------------------------------------------------ format_summary

    #[test]
    fn test_format_summary() {
        let cfg = TrainingProfile::standard().config();
        let summary = cfg.format_summary();
        assert!(
            !summary.is_empty(),
            "format_summary must return a non-empty string"
        );
        assert!(
            summary.contains("max_iterations"),
            "summary must contain max_iterations"
        );
        assert!(
            summary.contains("lr_position"),
            "summary must contain lr_position"
        );
        assert!(summary.contains("use_ema"), "summary must contain use_ema");
    }

    // ------------------------------------------------------------------ custom profile

    #[test]
    fn test_custom_profile() {
        let mut cfg = TrainingProfile::development().config();
        cfg.max_iterations = 5_000;
        cfg.use_ema = true;
        let profile = TrainingProfile::Custom(Box::new(cfg.clone()));
        let retrieved = profile.config();
        assert_eq!(retrieved.max_iterations, 5_000);
        assert!(retrieved.use_ema);
        // Verify PartialEq works on the config.
        assert_eq!(retrieved, cfg);
    }

    // ------------------------------------------------------------------ clone

    #[test]
    fn test_profile_config_clone() {
        let original = TrainingProfile::production().config();
        let cloned = original.clone();
        assert_eq!(original.max_iterations, cloned.max_iterations);
        assert_eq!(original.sh_degree, cloned.sh_degree);
        assert!((original.lpips_weight - cloned.lpips_weight).abs() < 1e-9);
    }

    #[test]
    fn test_profile_enum_clone() {
        let p = TrainingProfile::Production;
        let q = p.clone();
        assert_eq!(p, q);
    }

    // ------------------------------------------------------------------ all lr variants

    #[test]
    fn test_all_lr_must_be_positive() {
        let cfg = TrainingProfile::standard().config();
        assert!(cfg.lr_position > 0.0);
        assert!(cfg.lr_scale > 0.0);
        assert!(cfg.lr_rotation > 0.0);
        assert!(cfg.lr_opacity > 0.0);
        assert!(cfg.lr_sh > 0.0);
    }

    // ------------------------------------------------------------------ format_summary production

    // ------------------------------------------------------------------ From<TrainingProfileConfig>

    #[test]
    fn test_from_profile_maps_fields() {
        // Regression: `TrainingProfileConfig` previously had no conversion
        // to the type `Trainer::new` actually accepts.
        let profile = TrainingProfile::production().config();
        let cfg: TrainingConfig = profile.clone().into();

        assert_eq!(cfg.total_iterations, profile.max_iterations);
        assert_eq!(cfg.views_per_step, profile.views_per_step as usize);
        assert_eq!(cfg.density_control_start, profile.densify_from_iter);
        assert_eq!(cfg.density_control_end, profile.densify_until_iter);
        assert_eq!(cfg.checkpoint_interval, profile.checkpoint_interval);
        assert_eq!(cfg.log_interval, profile.log_interval);
        assert!((cfg.optimizer.lr_position - profile.lr_position as f32).abs() < 1e-12);
        assert!((cfg.loss.w_l1 - profile.photometric_weight).abs() < 1e-9);
        assert!((cfg.loss.w_lpips - profile.lpips_weight).abs() < 1e-9);
        assert_eq!(
            cfg.density.max_gaussians,
            profile.max_num_gaussians as usize
        );
        assert_eq!(cfg.init.sh_degree, profile.sh_degree);
        assert_eq!(
            cfg.init.num_rigid + cfg.init.num_flexible,
            profile.initial_num_gaussians as usize,
            "the rigid/flexible split must preserve the total Gaussian count"
        );
    }

    #[test]
    fn test_from_profile_produces_valid_training_config() {
        // Every built-in preset, once converted, must itself pass
        // `TrainingConfig::validate` — the whole point of the conversion is
        // to feed `Trainer::new` a usable config.
        for profile in [
            TrainingProfile::development(),
            TrainingProfile::standard(),
            TrainingProfile::production(),
        ] {
            let cfg: TrainingConfig = profile.config().into();
            assert!(cfg.validate().is_ok(), "converted config failed validate()");
        }
    }

    #[test]
    fn test_format_summary_production_contains_lpips() {
        let cfg = TrainingProfile::production().config();
        let summary = cfg.format_summary();
        assert!(
            summary.contains("use_lpips"),
            "production summary must mention use_lpips"
        );
        assert!(
            summary.contains("est. memory"),
            "summary must contain memory estimate"
        );
    }
}
