//! Training configuration types.
//!
//! All structs derive [`serde::Serialize`] / [`serde::Deserialize`] so they can
//! be loaded from TOML/JSON configuration files.

use serde::{Deserialize, Serialize};

use crate::gradient_clipping::ClipMode;
use crate::lr_scheduler::LrScheduler;
use crate::mixed_precision::TrainingPrecision;
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

    /// Floating-point precision mode for training (default: Float32).
    pub precision: TrainingPrecision,

    /// Enable per-phase profiling of the training loop (default: false).
    pub enable_profiling: bool,

    /// Learning-rate schedule applied on top of the per-group base rates.
    ///
    /// The schedule produces a **multiplier** (`1.0` at its peak), so it
    /// composes with the optimizer's own exponential position decay instead of
    /// replacing it.  Defaults to [`LrScheduleConfig::Fixed`], which is exactly
    /// the historical behaviour.
    #[serde(default)]
    pub lr_schedule: LrScheduleConfig,

    /// Gradient clipping applied immediately before the optimizer step.
    ///
    /// Defaults to [`GradientClipConfig::Disabled`].
    #[serde(default)]
    pub gradient_clip: GradientClipConfig,

    /// Number of `train_step` calls whose gradients are averaged into one
    /// optimizer update (micro-batching).  `1` (the default) steps every
    /// iteration, as before.
    #[serde(default = "default_accumulation_steps")]
    pub gradient_accumulation_steps: u32,

    /// Decay of the EMA shadow copy of the model, or `None` (the default) to
    /// keep no shadow weights.
    ///
    /// Must be in `(0, 1)`; `0.999` is a typical value.  When set, the trainer
    /// maintains an [`crate::ema::GaussianEma`] alongside the live model and
    /// checkpoints the **averaged** weights, which are usually the better
    /// evaluation target.
    #[serde(default)]
    pub ema_decay: Option<f32>,
}

/// Serde default for [`TrainingConfig::gradient_accumulation_steps`].
fn default_accumulation_steps() -> u32 {
    1
}

// ---------------------------------------------------------------------------
// LrScheduleConfig
// ---------------------------------------------------------------------------

/// Serialisable selection of an [`LrScheduler`] used as a learning-rate
/// *multiplier*.
///
/// [`crate::lr_scheduler`] schedules an absolute rate, but the trainer has six
/// independently configured per-group rates.  Building every schedule with
/// `base_lr = 1.0` turns it into a multiplicative factor that scales all groups
/// uniformly, which is what a warmup/cosine/step/cyclic schedule is for here.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LrScheduleConfig {
    /// No schedule: every group keeps its configured rate (times the
    /// optimizer's own position decay).
    #[default]
    Fixed,
    /// Linear warmup followed by cosine decay to `min_factor`.
    WarmupCosine {
        /// Steps spent ramping from ~0 to the full rate.
        warmup_steps: u32,
        /// Total schedule length; `0` means "use `total_iterations`".
        total_steps: u32,
        /// Floor of the multiplier, in `[0, 1)`.
        min_factor: f32,
    },
    /// Cosine decay from `1.0` to `min_factor` with no warmup.
    Cosine {
        /// Total schedule length; `0` means "use `total_iterations`".
        total_steps: u32,
        /// Floor of the multiplier, in `[0, 1)`.
        min_factor: f32,
    },
    /// Multiply by `decay_factor` every `step_size` iterations.
    Step {
        /// Per-drop factor, in `(0, 1)`.
        decay_factor: f32,
        /// Iterations between drops; must be `>= 1`.
        step_size: u32,
    },
    /// Multiply by `decay_rate` every iteration.
    Exponential {
        /// Per-iteration factor, in `(0, 1]`.
        decay_rate: f32,
    },
    /// Triangular cyclic schedule between `min_factor` and `max_factor`.
    Cyclic {
        /// Bottom of the cycle, in `[0, max_factor)`.
        min_factor: f32,
        /// Top of the cycle; must be `> min_factor`.
        max_factor: f32,
        /// Full-cycle length in iterations; must be `>= 1`.
        cycle_steps: u32,
    },
}

impl LrScheduleConfig {
    /// Build the concrete scheduler, or `None` for [`Self::Fixed`].
    ///
    /// `total_iterations` supplies the schedule length wherever a variant
    /// leaves `total_steps` at `0`.
    ///
    /// # Errors
    ///
    /// [`TrainerError::InvalidConfig`] wrapping the
    /// [`crate::lr_scheduler::LrSchedulerError`] for an out-of-range field —
    /// the scheduler constructors own those range checks, so there is exactly
    /// one place they live.
    pub fn build(&self, total_iterations: u32) -> Result<Option<LrScheduler>, TrainerError> {
        let resolve = |steps: u32| -> usize {
            if steps == 0 {
                total_iterations.max(1) as usize
            } else {
                steps as usize
            }
        };
        let to_err = |e: crate::lr_scheduler::LrSchedulerError| {
            TrainerError::InvalidConfig(format!("lr_schedule: {e}"))
        };

        let scheduler = match *self {
            LrScheduleConfig::Fixed => return Ok(None),
            LrScheduleConfig::WarmupCosine {
                warmup_steps,
                total_steps,
                min_factor,
            } => LrScheduler::warmup_cosine(
                warmup_steps as usize,
                1.0,
                min_factor as f64,
                resolve(total_steps),
            )
            .map_err(to_err)?,
            LrScheduleConfig::Cosine {
                total_steps,
                min_factor,
            } => LrScheduler::cosine_annealing(1.0, min_factor as f64, resolve(total_steps))
                .map_err(to_err)?,
            LrScheduleConfig::Step {
                decay_factor,
                step_size,
            } => LrScheduler::step_decay(1.0, decay_factor as f64, step_size as usize)
                .map_err(to_err)?,
            LrScheduleConfig::Exponential { decay_rate } => {
                LrScheduler::exponential(1.0, decay_rate as f64).map_err(to_err)?
            }
            LrScheduleConfig::Cyclic {
                min_factor,
                max_factor,
                cycle_steps,
            } => LrScheduler::cyclic(min_factor as f64, max_factor as f64, cycle_steps as usize)
                .map_err(to_err)?,
        };
        Ok(Some(scheduler))
    }

    /// Validate by building and discarding the scheduler.
    ///
    /// # Errors
    ///
    /// As [`build`](Self::build).
    pub fn validate(&self, total_iterations: u32) -> Result<(), TrainerError> {
        self.build(total_iterations).map(|_| ())
    }
}

// ---------------------------------------------------------------------------
// GradientClipConfig
// ---------------------------------------------------------------------------

/// Serialisable selection of a [`ClipMode`].
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum GradientClipConfig {
    /// No clipping (the default).
    #[default]
    Disabled,
    /// Clip the norm taken over all six parameter groups together.
    GlobalNorm {
        /// Maximum global L2 norm; must be `> 0`.
        max_norm: f32,
    },
    /// Clip each parameter group's norm independently.
    PerGroupNorm {
        /// Per-group maximum L2 norm; must be `> 0`.
        max_norm: f32,
    },
    /// Clamp every element into `[-max_value, max_value]`.
    Value {
        /// Element magnitude cap; must be `> 0`.
        max_value: f32,
    },
    /// Clip to `clip_factor ×` the EMA of the observed global norm.
    Adaptive {
        /// EMA smoothing factor, in `(0, 1)`.
        ema_factor: f32,
        /// Multiple of the EMA norm to clip at; must be `> 0`.
        clip_factor: f32,
    },
}

impl GradientClipConfig {
    /// The corresponding [`ClipMode`], or `None` when clipping is disabled.
    pub fn clip_mode(&self) -> Option<ClipMode> {
        match *self {
            GradientClipConfig::Disabled => None,
            GradientClipConfig::GlobalNorm { max_norm } => Some(ClipMode::GlobalNorm { max_norm }),
            GradientClipConfig::PerGroupNorm { max_norm } => {
                Some(ClipMode::PerGroupNorm { max_norm })
            }
            GradientClipConfig::Value { max_value } => {
                Some(ClipMode::ValueClip { max_val: max_value })
            }
            GradientClipConfig::Adaptive {
                ema_factor,
                clip_factor,
            } => Some(ClipMode::Adaptive {
                ema_factor,
                clip_factor,
            }),
        }
    }

    /// The configured magnitude threshold, where the mode has a fixed one.
    ///
    /// [`Self::Adaptive`] derives its threshold from the running norm EMA, so
    /// it has none up front and returns `None`.
    pub fn threshold(&self) -> Option<f32> {
        match *self {
            GradientClipConfig::Disabled | GradientClipConfig::Adaptive { .. } => None,
            GradientClipConfig::GlobalNorm { max_norm }
            | GradientClipConfig::PerGroupNorm { max_norm } => Some(max_norm),
            GradientClipConfig::Value { max_value } => Some(max_value),
        }
    }

    /// Validate the mode-specific thresholds.
    ///
    /// # Errors
    ///
    /// [`TrainerError::InvalidConfig`] wrapping the
    /// [`crate::gradient_clipping::ClipError`] — the range checks live in
    /// [`crate::gradient_clipping::GradientClipper::new`].
    pub fn validate(&self) -> Result<(), TrainerError> {
        match self.clip_mode() {
            None => Ok(()),
            Some(mode) => crate::gradient_clipping::GradientClipper::new(mode)
                .map(|_| ())
                .map_err(|e| TrainerError::InvalidConfig(format!("gradient_clip: {e}"))),
        }
    }
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
            precision: TrainingPrecision::Float32,
            enable_profiling: false,
            lr_schedule: LrScheduleConfig::Fixed,
            gradient_clip: GradientClipConfig::Disabled,
            gradient_accumulation_steps: 1,
            ema_decay: None,
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

        // NOTE: `density_control_end` is intentionally *not* required to be
        // `<= total_iterations` here. A window that outlives the run is
        // behaviourally inert (`should_densify` also checks
        // `iteration <= density_control_end`, and training simply stops at
        // `total_iterations` first) rather than a defect, and several
        // GPU-gated integration tests (`end_to_end_tests.rs`,
        // `#[ignore = "GPU test"]`) construct exactly this shape — e.g.
        // `test_training_config(1)` pairs `total_iterations: 1` with a fixed
        // `density_control_end: 50`. Rejecting it here would be inventing a
        // new constraint not required for correctness, at the cost of
        // breaking those fixtures the moment `validate()` is wired into a
        // constructor that calls it.

        // NOTE: `guidance_anneal_steps == 0` is intentionally *not* rejected
        // here. It reads as a division-by-zero hazard
        // (`t = iteration / guidance_anneal_steps`), but both actual readers
        // already guard it as a deliberate "annealing disabled, hold at
        // `guidance_scale_end`" sentinel:
        // `Trainer::current_guidance_scale` clamps the denominator to
        // `(guidance_anneal_steps as f32).max(1.0)`, and
        // `DiffusionTargetGenerator::annealed_guidance_scale` special-cases
        // `anneal_steps == 0` to `t = 1.0` directly — both are covered by
        // `guidance_schedule_reaches_the_diffusion_generator`, which
        // constructs `guidance_anneal_steps: 0` and asserts a finite result
        // rather than expecting an error. Rejecting `0` here would outlaw a
        // supported, tested configuration for a failure mode that no longer
        // exists in either reader.

        // Guidance scale endpoints feed straight into the lerp in both of
        // the readers above; a non-finite value would poison every annealed
        // scale regardless of `guidance_anneal_steps`.
        if !self.guidance_scale_start.is_finite() {
            return Err(TrainerError::ParameterOutOfRange {
                param: "guidance_scale_start".into(),
                value: format!("{}", self.guidance_scale_start),
                expected: "finite".into(),
            });
        }
        if !self.guidance_scale_end.is_finite() {
            return Err(TrainerError::ParameterOutOfRange {
                param: "guidance_scale_end".into(),
                value: format!("{}", self.guidance_scale_end),
                expected: "finite".into(),
            });
        }

        // The remaining interval fields gate `iteration.is_multiple_of(n)` /
        // `iteration % n` checks in the training loop. `0` is a valid,
        // deliberate "disable this schedule" sentinel for
        // `density_control_interval` and `opacity_reset_interval` (both are
        // explicitly guarded with `> 0` at the call site — see
        // `Trainer::should_densify` and the opacity-reset check in
        // `Trainer::train_step`), and is exercised that way by existing
        // tests. `checkpoint_interval` and `log_interval` have no such
        // caller-side `> 0` guard and are consulted unconditionally in
        // `Trainer::run` — `self.iteration` is incremented *before* either
        // check runs (never observed at `0`), so `n.is_multiple_of(0)` is
        // `false` for every reachable iteration and a `0` interval silently
        // disables logging/checkpointing for the entire run rather than
        // "once". That is unlikely to be the intent of a config that also
        // sets `total_iterations > 0`, so `0` is rejected for these two
        // instead of silently discarding all progress visibility.
        if self.checkpoint_interval == 0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "checkpoint_interval".into(),
                value: "0".into(),
                expected: "> 0".into(),
            });
        }
        if self.log_interval == 0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "log_interval".into(),
                value: "0".into(),
                expected: "> 0".into(),
            });
        }

        // A zero accumulation window would never step the optimizer at all.
        if self.gradient_accumulation_steps == 0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "gradient_accumulation_steps".into(),
                value: "0".into(),
                expected: ">= 1".into(),
            });
        }

        // `GaussianEma::new` takes the decay verbatim; outside `(0, 1)` the
        // shadow either never moves or diverges.
        if let Some(decay) = self.ema_decay {
            if !decay.is_finite() || decay <= 0.0 || decay >= 1.0 {
                return Err(TrainerError::ParameterOutOfRange {
                    param: "ema_decay".into(),
                    value: format!("{decay}"),
                    expected: "in (0, 1)".into(),
                });
            }
        }

        // Schedules own their own range checks; building them is the check.
        self.lr_schedule.validate(self.total_iterations)?;
        self.gradient_clip.validate()?;

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

        // `GaussianOptimizer::position_lr` divides by
        // `position_lr_decay_steps`; it clamps the denominator to 1 so a `0`
        // does not produce `NaN`, but the resulting schedule then jumps to
        // `lr_position_final` after a single step — silently *not* the decay
        // the user asked for.  Reject it here instead of relying on the
        // clamp, matching how every learning rate and epsilon is checked.
        if self.position_lr_decay_steps == 0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "position_lr_decay_steps".into(),
                value: "0".into(),
                expected: ">= 1".into(),
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
    /// World-space scale (post-`exp()`) above which the scale regulariser
    /// starts to charge — the `max_scale` of
    /// [`crate::loss::scale_reg_with_max`].
    ///
    /// Defaults to [`crate::loss::MAX_REASONABLE_WORLD_SCALE`].  Raising it
    /// tolerates larger Gaussians; lowering it fights growth sooner.  It is a
    /// world-space length, not a log-scale, and must be finite and `>= 0`.
    #[serde(default = "default_scale_reg_max_scale")]
    pub w_scale_reg_max_scale: f32,
}

/// Serde default for [`LossConfig::w_scale_reg_max_scale`].
///
/// Kept as a named function so configs serialised before the field existed
/// still deserialise to the constant the regulariser used back then.
fn default_scale_reg_max_scale() -> f32 {
    crate::loss::MAX_REASONABLE_WORLD_SCALE
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
            w_scale_reg_max_scale: crate::loss::MAX_REASONABLE_WORLD_SCALE,
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
        // Not a weight but the same domain: a negative or non-finite threshold
        // makes every Gaussian "oversized" (or NaN-poisons the term).
        check_weight("w_scale_reg_max_scale", self.w_scale_reg_max_scale)?;

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

        // Consumed at `density.rs` as `if max_scale > split_scale_threshold`
        // to choose between split and clone. Unlike `grad_threshold` this is
        // compared against an already-exponentiated scale (`s.exp()`, always
        // > 0), so a *negative* finite threshold is a legitimate (if
        // aggressive) "always split" setting — only non-finite values are
        // rejected, since a NaN threshold makes the comparison always false
        // (every Gaussian silently clones, never splits) with no diagnostic.
        if !self.split_scale_threshold.is_finite() {
            return Err(TrainerError::ParameterOutOfRange {
                param: "split_scale_threshold".into(),
                value: format!("{}", self.split_scale_threshold),
                expected: "finite".into(),
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizer_rejects_zero_position_lr_decay_steps() {
        // Regression (F139): every LR and epsilon was validated but
        // `position_lr_decay_steps` was not, so `0` survived validation and
        // only the optimizer's denominator clamp kept it finite — silently
        // collapsing the decay schedule to a single step.
        let mut config = OptimizerConfig::default();
        assert!(config.validate().is_ok());
        config.position_lr_decay_steps = 0;
        let err = config
            .validate()
            .expect_err("0 decay steps must be rejected");
        assert!(
            format!("{err}").contains("position_lr_decay_steps"),
            "error should name the field: {err}"
        );
        config.position_lr_decay_steps = 1;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn loss_config_validates_the_scale_reg_threshold() {
        // Regression (F289): the threshold used to be a hardcoded constant with
        // no field, hence no validation path at all.
        let mut config = LossConfig::default();
        assert_eq!(
            config.w_scale_reg_max_scale,
            crate::loss::MAX_REASONABLE_WORLD_SCALE
        );
        assert!(config.validate().is_ok());

        config.w_scale_reg_max_scale = -1.0;
        assert!(config.validate().is_err());
        config.w_scale_reg_max_scale = f32::NAN;
        assert!(config.validate().is_err());
        config.w_scale_reg_max_scale = 0.2;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn loss_config_deserialises_without_the_new_threshold_field() {
        // Configs written before `w_scale_reg_max_scale` existed must keep
        // loading, at the constant the regulariser used back then.
        let json = r#"{
            "w_l1": 0.8, "w_ssim": 0.2, "w_ms_ssim": 0.0, "w_lpips": 0.0,
            "w_position_reg": 0.01, "w_scale_reg": 0.01, "w_opacity_reg": 0.001,
            "w_normal": 0.05, "w_gradient_penalty": 0.0,
            "gradient_penalty_threshold": 100.0
        }"#;
        let config: LossConfig = serde_json::from_str(json).expect("legacy config must load");
        assert_eq!(
            config.w_scale_reg_max_scale,
            crate::loss::MAX_REASONABLE_WORLD_SCALE
        );
    }

    #[test]
    fn lr_schedule_config_builds_a_multiplier_schedule() {
        use crate::lr_scheduler::LrSchedule;

        assert!(LrScheduleConfig::Fixed
            .build(1_000)
            .expect("Fixed must build")
            .is_none());

        let schedule = LrScheduleConfig::WarmupCosine {
            warmup_steps: 100,
            total_steps: 1_000,
            min_factor: 0.0,
        }
        .build(1_000)
        .expect("valid schedule must build")
        .expect("WarmupCosine is not Fixed");

        // `base_lr = 1.0` makes the schedule a pure multiplier: it ramps up
        // through warmup and decays afterwards.
        let early = schedule.lr_at(10);
        let peak = schedule.lr_at(100);
        let late = schedule.lr_at(900);
        assert!(early < peak, "warmup must ramp: {early} !< {peak}");
        assert!(late < peak, "cosine must decay: {late} !< {peak}");
        assert!((peak - 1.0).abs() < 1e-9, "peak multiplier should be 1.0");

        // `total_steps == 0` falls back to `total_iterations`.
        let resolved = LrScheduleConfig::Cosine {
            total_steps: 0,
            min_factor: 0.1,
        }
        .build(500)
        .expect("valid schedule must build")
        .expect("Cosine is not Fixed");
        assert!(resolved.lr_at(500) <= resolved.lr_at(0));

        // Out-of-range fields are reported, not clamped.
        assert!(LrScheduleConfig::Step {
            decay_factor: 1.5,
            step_size: 10,
        }
        .build(100)
        .is_err());
    }

    #[test]
    fn gradient_clip_config_maps_to_clip_modes() {
        assert!(GradientClipConfig::Disabled.clip_mode().is_none());
        assert!(GradientClipConfig::Disabled.threshold().is_none());
        assert_eq!(
            GradientClipConfig::GlobalNorm { max_norm: 2.5 }.threshold(),
            Some(2.5)
        );
        assert_eq!(
            GradientClipConfig::Value { max_value: 0.5 }.clip_mode(),
            Some(ClipMode::ValueClip { max_val: 0.5 })
        );
        // Adaptive derives its threshold at runtime.
        assert!(GradientClipConfig::Adaptive {
            ema_factor: 0.9,
            clip_factor: 2.0,
        }
        .threshold()
        .is_none());

        assert!(GradientClipConfig::GlobalNorm { max_norm: 0.0 }
            .validate()
            .is_err());
        assert!(GradientClipConfig::GlobalNorm { max_norm: 1.0 }
            .validate()
            .is_ok());
    }

    #[test]
    fn training_config_validates_the_new_loop_fields() {
        let mut config = TrainingConfig::default();
        assert!(config.validate().is_ok());

        config.gradient_accumulation_steps = 0;
        assert!(config.validate().is_err());
        config.gradient_accumulation_steps = 4;
        assert!(config.validate().is_ok());

        config.ema_decay = Some(1.0);
        assert!(config.validate().is_err());
        config.ema_decay = Some(0.0);
        assert!(config.validate().is_err());
        config.ema_decay = Some(0.999);
        assert!(config.validate().is_ok());

        config.gradient_clip = GradientClipConfig::PerGroupNorm { max_norm: -1.0 };
        assert!(config.validate().is_err());
    }

    #[test]
    fn training_config_defaults_deserialise_from_a_legacy_document() {
        // The four new fields all carry serde defaults, so a config file
        // written before they existed must still round-trip.
        let full = serde_json::to_string(&TrainingConfig::default())
            .expect("default config must serialise");
        let mut value: serde_json::Value =
            serde_json::from_str(&full).expect("serialised config must parse");
        let object = value.as_object_mut().expect("config is a JSON object");
        for key in [
            "lr_schedule",
            "gradient_clip",
            "gradient_accumulation_steps",
            "ema_decay",
        ] {
            object.remove(key);
        }
        let legacy: TrainingConfig =
            serde_json::from_value(value).expect("legacy config must load");
        assert_eq!(legacy.lr_schedule, LrScheduleConfig::Fixed);
        assert_eq!(legacy.gradient_clip, GradientClipConfig::Disabled);
        assert_eq!(legacy.gradient_accumulation_steps, 1);
        assert!(legacy.ema_decay.is_none());
    }
}
