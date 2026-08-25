//! Main [`Trainer`] struct — orchestrates the GAF optimisation loop.
//!
//! Each iteration:
//! 1. Sample random camera views.
//! 2. Render the current Gaussian model via GPU rasterizer.
//! 3. Generate multi-view targets via diffusion (or self-supervised fallback).
//! 4. Compute photometric + distillation + regularisation losses.
//! 5. Backward pass → GPU rasterizer backward → per-Gaussian gradients.
//! 6. Adam optimiser step (fully functional).
//! 7. Adaptive density control at scheduled intervals.
//! 8. Periodic opacity resets and checkpointing.
//!
//! ## Gradients match the reported loss
//!
//! The image-space gradient handed to the backward pass is the analytic
//! derivative of exactly the objective [`LossComputer`] reports —
//! `w_l1·∂L1 + w_ssim·∂(1−SSIM) + w_ms_ssim·∂(1−MS-SSIM) + ∂L_distill`, all at
//! the same per-pixel/per-view normalisation — and the regularisers
//! (`w_position_reg`, `w_scale_reg`, `w_opacity_reg`) are differentiated
//! straight into the parameter gradients, since they never reach the renderer.
//! Changing a configured weight therefore changes what is optimised, not only
//! what is logged.
//!
//! Two configured terms contribute no gradient *by construction*: normal
//! consistency is reported once a mesh is installed with [`Trainer::set_mesh`]
//! (and stays a constant `0.0` without one) but is not differentiated into the
//! parameter gradients, and the gradient penalty is evaluated on an external
//! gradient buffer the trainer does not pass.
//!
//! ## Optional loop components
//!
//! [`TrainingConfig`] switches on four components that are otherwise inert:
//! a learning-rate multiplier schedule (`lr_schedule`), gradient clipping
//! (`gradient_clip`), micro-batch gradient accumulation
//! (`gradient_accumulation_steps`) and EMA shadow weights (`ema_decay`).  Each
//! defaults to off, so an unchanged config drives exactly the loop it did
//! before.
//!
//! ## Diffusion Integration
//!
//! The trainer distils multi-view diffusion output in **pixel space**: the
//! pipeline's decoded views are the pseudo ground truth, and the residual
//! against them — weighted by the variance-preserving `w(t)` at the annealed
//! timestep ([`DiffusionTargetGenerator::compute_sds_gradient`]) — is added to
//! the image-space gradient.  That is the Score Distillation Sampling descent
//! direction up to the denoiser Jacobian SDS drops by construction, without
//! needing the U-Net's internal `ε` prediction.  During training:
//!
//! 1. **Warmup Period** (first 1000 iterations by default): pure photometric
//!    training, letting the model reach a reasonable starting point.
//! 2. **Distillation Period**: the diffusion model generates pseudo-GT targets
//!    whose gradient is added to the photometric one.
//! 3. **Timestep Annealing**: the noise timestep starts high (more guidance)
//!    and anneals down to preserve fine details.

use std::path::Path;

use nalgebra as na;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use oxigaf_flame::{Camera, Mesh, NormalMapRenderer};
use oxigaf_render::gaussian::GaussianModel;
use oxigaf_render::{RasterConfig, Rasterizer, RenderCamera};

use crate::checkpoint;
use crate::config::{LossConfig, TrainingConfig};
use crate::density::DensityController;
use crate::diffusion_target::{DiffusionTargetConfig, DiffusionTargetGenerator, SdsLoss};
use crate::ema::GaussianEma;
use crate::gradient_accumulation::{AccumulationConfig, GradNormalization, GradientAccumulator};
use crate::gradient_clipping::{ClipStats, GradientClipper};
use crate::image_gradient::{photometric_pixel_gradient, PhotometricSpec};
use crate::loss::{scale_reg_axis_gradient, LossComputer, LossOutput};
use crate::lr_scheduler::{LrSchedule, LrScheduler};
use crate::metrics::{self, MetricTracker};
use crate::mixed_precision::MixedPrecisionTrainer;
use crate::optimizer::{GaussianOptimizer, Gradients};
use crate::profiler_integration::{TrainingPhase, TrainingProfiler};
use crate::tensorboard::{LearningRates, TrainingMetricsLogger};
use crate::TrainerError;

// ---------------------------------------------------------------------------
// StepOutput
// ---------------------------------------------------------------------------

/// Summary returned after every training step.
#[derive(Debug, Clone)]
pub struct StepOutput {
    pub iteration: u32,
    pub loss: LossOutput,
    pub num_gaussians: usize,
    /// SDS loss value (0.0 during warmup).
    pub sds_loss: f32,
    /// Whether this step used diffusion targets.
    pub used_diffusion: bool,
    /// Current diffusion timestep.
    pub diffusion_timestep: u32,
    /// Multi-view consistency penalty on the diffusion targets, already scaled
    /// by `DiffusionTargetConfig::view_consistency_weight` (`0.0` when no
    /// diffusion targets were produced this step).
    pub view_consistency: f32,
    /// Whether the optimizer actually stepped this iteration.
    ///
    /// `false` while a gradient-accumulation window is still filling, and on a
    /// mixed-precision gradient overflow.
    pub optimizer_stepped: bool,
    /// Learning-rate multiplier the configured schedule produced this step.
    pub lr_scale: f32,
    /// Clipping statistics, when gradient clipping is configured.
    pub clip_stats: Option<ClipStats>,
}

// ---------------------------------------------------------------------------
// LoopComponents
// ---------------------------------------------------------------------------

/// The optional per-iteration components [`TrainingConfig`] switches on.
///
/// Each is `None` for the historical default, so a config that sets none of
/// the new fields drives exactly the loop it did before.
struct LoopComponents {
    lr_scheduler: Option<LrScheduler>,
    gradient_clipper: Option<GradientClipper>,
    gradient_accumulator: Option<GradientAccumulator>,
    ema: Option<GaussianEma>,
}

impl LoopComponents {
    /// Build every configured component, or fail with the reason.
    ///
    /// # Errors
    ///
    /// [`TrainerError::InvalidConfig`] / [`TrainerError::ParameterOutOfRange`]
    /// for an out-of-range schedule, clip threshold, accumulation window or EMA
    /// decay — the same checks [`TrainingConfig::validate`] runs, applied here
    /// so a trainer can never be constructed with a component silently dropped.
    fn from_config(config: &TrainingConfig, model: &GaussianModel) -> Result<Self, TrainerError> {
        let lr_scheduler = config.lr_schedule.build(config.total_iterations)?;

        let gradient_clipper = match config.gradient_clip.clip_mode() {
            None => None,
            Some(mode) => Some(
                GradientClipper::new(mode)
                    .map_err(|e| TrainerError::InvalidConfig(format!("gradient_clip: {e}")))?,
            ),
        };

        let gradient_accumulator =
            if config.gradient_accumulation_steps <= 1 {
                None
            } else {
                let accumulation_config = AccumulationConfig {
                    accumulation_steps: config.gradient_accumulation_steps as usize,
                    // The trainer already averages over views inside one step, so
                    // the window average is the right micro-batch normalisation.
                    normalization: GradNormalization::MeanOverSteps,
                    auto_clear: true,
                };
                Some(GradientAccumulator::new(accumulation_config).map_err(|e| {
                    TrainerError::InvalidConfig(format!("gradient_accumulation: {e}"))
                })?)
            };

        let ema = match config.ema_decay {
            None => None,
            Some(decay) => {
                if !decay.is_finite() || decay <= 0.0 || decay >= 1.0 {
                    return Err(TrainerError::ParameterOutOfRange {
                        param: "ema_decay".into(),
                        value: format!("{decay}"),
                        expected: "in (0, 1)".into(),
                    });
                }
                Some(GaussianEma::new(model, decay))
            }
        };

        Ok(Self {
            lr_scheduler,
            gradient_clipper,
            gradient_accumulator,
            ema,
        })
    }
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

/// The main training driver.
///
/// Holds all mutable state: the Gaussian model, optimiser, density controller,
/// loss computer, and metric tracker.  Call [`Trainer::train_step`] in a loop
/// or use [`Trainer::run`] to execute the full schedule.
///
/// It optionally holds a [`DiffusionTargetGenerator`] for iterative denoising
/// distillation: after the warmup phase the diffusion model supplies pseudo-GT
/// targets whose distillation gradient joins the photometric one (see the
/// [module documentation](self)).
pub struct Trainer {
    pub config: TrainingConfig,
    pub model: GaussianModel,
    pub optimizer: GaussianOptimizer,
    pub density_controller: DensityController,
    pub loss_computer: LossComputer,
    pub metric_tracker: MetricTracker,
    pub raster_config: RasterConfig,
    pub rasterizer: Rasterizer,
    pub rng: StdRng,
    pub iteration: u32,

    // ---- Diffusion integration ----
    /// Diffusion target generator for pseudo-GT.
    pub diffusion_generator: DiffusionTargetGenerator,
    /// SDS loss computer.
    pub sds_loss: SdsLoss,
    /// Diffusion target configuration.
    pub diffusion_config: DiffusionTargetConfig,

    // ---- TensorBoard logging ----
    /// TensorBoard metrics logger.
    pub tensorboard_logger: TrainingMetricsLogger,

    // ---- Mixed precision / profiling ----
    /// Mixed-precision trainer (loss scaler + precision mode).
    pub mp_trainer: MixedPrecisionTrainer,
    /// Per-phase training profiler.
    pub profiler: TrainingProfiler,

    // ---- Configured loop components ----
    /// Learning-rate multiplier schedule, `None` for
    /// [`crate::config::LrScheduleConfig::Fixed`].
    pub lr_scheduler: Option<LrScheduler>,
    /// Gradient clipper run immediately before the optimizer step, `None` for
    /// [`crate::config::GradientClipConfig::Disabled`].
    pub gradient_clipper: Option<GradientClipper>,
    /// Micro-batch gradient accumulator, `None` when
    /// `TrainingConfig::gradient_accumulation_steps <= 1`.
    pub gradient_accumulator: Option<GradientAccumulator>,
    /// EMA shadow copy of the model, `None` when `TrainingConfig::ema_decay`
    /// is unset.  [`Trainer::save_checkpoint`] writes these weights when
    /// present — see [`Trainer::ema_model`].
    pub ema: Option<GaussianEma>,

    /// FLAME mesh the Gaussians are bound to, when the caller supplies one.
    ///
    /// `None` by default — [`Trainer::new`] takes a [`GaussianModel`] alone.
    /// Installing one with [`Trainer::set_mesh`] switches on the two terms that
    /// need the surface and are otherwise inert:
    ///
    /// * `LossConfig::w_normal` — normal consistency, a hardcoded `0.0` without
    ///   a mesh;
    /// * per-view normal-map conditioning of the diffusion targets
    ///   ([`DiffusionTargetGenerator::generate_targets_with_normals`]).
    pub mesh: Option<Mesh>,
}

impl Trainer {
    /// Create a fresh trainer with an already-initialised Gaussian model.
    ///
    /// Requires a `wgpu::Device` and `wgpu::Queue` for the GPU rasterizer.
    /// Use [`Rasterizer::new`] (async) or obtain them externally.
    ///
    /// The optional `diffusion_config` enables Score Distillation Sampling.
    /// If `None`, defaults are used but diffusion remains disabled until
    /// [`Trainer::load_diffusion_pipeline`] is called.
    pub fn new(
        config: TrainingConfig,
        model: GaussianModel,
        raster_config: RasterConfig,
        device: wgpu::Device,
        queue: wgpu::Queue,
        seed: u64,
    ) -> Result<Self, TrainerError> {
        Self::with_diffusion_config(
            config,
            model,
            raster_config,
            device,
            queue,
            seed,
            DiffusionTargetConfig::default(),
        )
    }

    /// Create a trainer with explicit diffusion configuration.
    ///
    /// This allows customizing the warmup period, timestep annealing, and SDS
    /// weights.  The classifier-free-guidance annealing schedule always comes
    /// from the top-level [`TrainingConfig`] (see `apply_guidance_schedule`).
    pub fn with_diffusion_config(
        config: TrainingConfig,
        model: GaussianModel,
        raster_config: RasterConfig,
        device: wgpu::Device,
        queue: wgpu::Queue,
        seed: u64,
        mut diffusion_config: DiffusionTargetConfig,
    ) -> Result<Self, TrainerError> {
        // Wire the CLI-facing guidance schedule into the config the generator
        // actually reads, then validate the merged result.
        apply_guidance_schedule(&config, &mut diffusion_config);
        diffusion_config.validate()?;

        let optimizer = GaussianOptimizer::new(&config.optimizer, &model);
        let density_controller = DensityController::new(config.density.clone(), model.len());
        let loss_computer = LossComputer::new(config.loss.clone());
        let metric_tracker = MetricTracker::new();
        let rng = StdRng::seed_from_u64(seed);

        let rasterizer = Rasterizer::from_device(device, queue, raster_config.clone())?;

        // Initialize diffusion target generator (pipeline loaded lazily)
        let diffusion_generator = DiffusionTargetGenerator::new(diffusion_config.clone());
        let sds_loss = SdsLoss::default();

        // Initialize TensorBoard logger
        let tensorboard_logger = TrainingMetricsLogger::new(config.tensorboard.clone())?;

        // Initialize mixed-precision trainer and profiler
        let mp_trainer = MixedPrecisionTrainer::new(config.precision);
        let profiler = TrainingProfiler::new(config.enable_profiling);
        let components = LoopComponents::from_config(&config, &model)?;

        tracing::info!(
            "Trainer created: {} Gaussians, {} total iterations, warmup={} iters, tensorboard={}, precision={}, profiling={}, lr_schedule={:?}, clip={:?}, accum={}, ema={:?}",
            model.len(),
            config.total_iterations,
            diffusion_config.warmup_iterations,
            tensorboard_logger.is_enabled(),
            mp_trainer.precision.label(),
            profiler.is_enabled(),
            config.lr_schedule,
            config.gradient_clip,
            config.gradient_accumulation_steps,
            config.ema_decay,
        );

        Ok(Self {
            config,
            model,
            optimizer,
            density_controller,
            loss_computer,
            metric_tracker,
            raster_config,
            rasterizer,
            rng,
            iteration: 0,
            diffusion_generator,
            sds_loss,
            diffusion_config,
            tensorboard_logger,
            mp_trainer,
            profiler,
            lr_scheduler: components.lr_scheduler,
            gradient_clipper: components.gradient_clipper,
            gradient_accumulator: components.gradient_accumulator,
            ema: components.ema,
            mesh: None,
        })
    }

    /// Load the diffusion pipeline from a weights directory.
    ///
    /// The directory should contain:
    /// - `unet/diffusion_pytorch_model.safetensors`
    /// - `vae/diffusion_pytorch_model.safetensors`
    /// - `image_encoder/model.safetensors`
    ///
    /// After loading, subsequent training steps will use diffusion targets
    /// for Score Distillation Sampling (after the warmup period).
    pub fn load_diffusion_pipeline(&mut self, weights_dir: &Path) -> Result<(), TrainerError> {
        self.diffusion_generator.load_pipeline(weights_dir)?;
        tracing::info!("Diffusion pipeline loaded, SDS training enabled");
        Ok(())
    }

    /// Check if the diffusion pipeline is loaded.
    pub fn is_diffusion_loaded(&self) -> bool {
        self.diffusion_generator.is_loaded()
    }

    /// Install the FLAME mesh the Gaussians are bound to.
    ///
    /// Without one, two configured terms are inert by construction:
    /// `LossConfig::w_normal` reports a constant `0.0`, and the diffusion
    /// targets are generated with no geometric conditioning.  Both switch on
    /// the moment a mesh is present — see [`Trainer::mesh`].
    ///
    /// The mesh is a *fixed* surface for the run: it is not re-posed per
    /// iteration, so pass the mesh in the pose the Gaussians were bound in.
    pub fn set_mesh(&mut self, mesh: Mesh) {
        tracing::info!(
            vertices = mesh.vertices.len(),
            faces = mesh.faces.len(),
            "FLAME mesh installed — normal consistency and normal-map \
             conditioning are now active"
        );
        self.mesh = Some(mesh);
    }

    /// Drop the installed mesh, returning to the mesh-free path.
    pub fn clear_mesh(&mut self) -> Option<Mesh> {
        self.mesh.take()
    }

    /// Check if we're currently in the warmup period.
    pub fn is_warmup(&self) -> bool {
        self.diffusion_generator.is_warmup(self.iteration)
    }

    /// Restore a trainer from a saved checkpoint.
    ///
    /// Optionally accepts a diffusion configuration. If `None`, defaults are used
    /// but diffusion remains disabled until [`Trainer::load_diffusion_pipeline`]
    /// is called.
    pub fn from_checkpoint(
        config: TrainingConfig,
        checkpoint_path: &Path,
        raster_config: RasterConfig,
        device: wgpu::Device,
        queue: wgpu::Queue,
        seed: u64,
    ) -> Result<Self, TrainerError> {
        Self::from_checkpoint_with_diffusion(
            config,
            checkpoint_path,
            raster_config,
            device,
            queue,
            seed,
            DiffusionTargetConfig::default(),
        )
    }

    /// Restore a trainer from a saved checkpoint with explicit diffusion config.
    ///
    /// Guidance annealing comes from [`TrainingConfig`], as in
    /// [`Trainer::with_diffusion_config`].
    pub fn from_checkpoint_with_diffusion(
        config: TrainingConfig,
        checkpoint_path: &Path,
        raster_config: RasterConfig,
        device: wgpu::Device,
        queue: wgpu::Queue,
        seed: u64,
        mut diffusion_config: DiffusionTargetConfig,
    ) -> Result<Self, TrainerError> {
        // Wire the CLI-facing guidance schedule into the config the generator
        // actually reads, then validate the merged result.
        apply_guidance_schedule(&config, &mut diffusion_config);
        diffusion_config.validate()?;

        let ckpt = checkpoint::load_checkpoint(checkpoint_path)?;

        let model = checkpoint::restore_model(&ckpt);
        let mut optimizer = GaussianOptimizer::new(&config.optimizer, &model);
        checkpoint::restore_optimizer(&ckpt, &mut optimizer);

        let density_controller = DensityController::new(config.density.clone(), model.len());
        let loss_computer = LossComputer::new(config.loss.clone());
        let metric_tracker = checkpoint::restore_metrics(&ckpt);
        let rng = StdRng::seed_from_u64(seed);
        let iteration = ckpt.iteration;

        let rasterizer = Rasterizer::from_device(device, queue, raster_config.clone())?;

        // Initialize diffusion target generator (pipeline loaded lazily)
        let diffusion_generator = DiffusionTargetGenerator::new(diffusion_config.clone());
        let sds_loss = SdsLoss::default();

        // Initialize TensorBoard logger
        let tensorboard_logger = TrainingMetricsLogger::new(config.tensorboard.clone())?;

        // Initialize mixed-precision trainer and profiler
        let mp_trainer = MixedPrecisionTrainer::new(config.precision);
        let profiler = TrainingProfiler::new(config.enable_profiling);
        let components = LoopComponents::from_config(&config, &model)?;

        tracing::info!(
            "Trainer restored from checkpoint at iteration {iteration}, {} Gaussians, warmup={}, tensorboard={}, precision={}, profiling={}",
            model.len(),
            diffusion_config.warmup_iterations,
            tensorboard_logger.is_enabled(),
            mp_trainer.precision.label(),
            profiler.is_enabled(),
        );

        Ok(Self {
            config,
            model,
            optimizer,
            density_controller,
            loss_computer,
            metric_tracker,
            raster_config,
            rasterizer,
            rng,
            iteration,
            diffusion_generator,
            sds_loss,
            diffusion_config,
            tensorboard_logger,
            mp_trainer,
            profiler,
            lr_scheduler: components.lr_scheduler,
            gradient_clipper: components.gradient_clipper,
            gradient_accumulator: components.gradient_accumulator,
            ema: components.ema,
            mesh: None,
        })
    }

    // -----------------------------------------------------------------------
    // Training loop
    // -----------------------------------------------------------------------

    /// Execute the full training schedule.
    pub fn run(&mut self, checkpoint_dir: Option<&Path>) -> Result<(), TrainerError> {
        let total = self.config.total_iterations;
        tracing::info!("Starting training for {total} iterations");

        while self.iteration < total {
            let output = self.train_step()?;

            // Logging.
            if self.iteration.is_multiple_of(self.config.log_interval) {
                tracing::info!(
                    "{}",
                    self.metric_tracker
                        .summary_string(self.config.log_interval as usize,),
                );
            }

            // Checkpointing.
            if let Some(dir) = checkpoint_dir {
                if self
                    .iteration
                    .is_multiple_of(self.config.checkpoint_interval)
                {
                    let t_ckpt = std::time::Instant::now();
                    let path = dir.join(format!("ckpt_{:06}.json", self.iteration));
                    let saved = self.save_checkpoint(&path);
                    self.profiler.record(
                        TrainingPhase::Checkpoint,
                        t_ckpt.elapsed().as_micros() as u64,
                    );
                    saved?;
                }
            }

            // Early log on first step for sanity.
            if output.iteration == 1 {
                tracing::info!(
                    "First step complete — loss {:.6}, {} Gaussians",
                    output.loss.total,
                    output.num_gaussians,
                );
            }
        }

        tracing::info!("Training complete after {total} iterations");
        Ok(())
    }

    /// Execute a **single** training step and return a summary.
    ///
    /// This implements the full training loop:
    /// 1. Sample random camera views
    /// 2. Render current Gaussian model
    /// 3. Generate diffusion targets (after warmup) or use self-supervised mode
    /// 4. Compute photometric + distillation losses
    /// 5. Backward pass through GPU rasterizer
    /// 6. Adam optimizer step
    /// 7. Adaptive density control
    /// 8. Periodic opacity reset
    /// 9. Record metrics
    ///
    /// # Errors
    ///
    /// A rasterizer failure in the forward or backward pass aborts the step with
    /// [`TrainerError::Render`] instead of optimising against a fabricated image
    /// or a silently down-weighted gradient.
    pub fn train_step(&mut self) -> Result<StepOutput, TrainerError> {
        // `TrainingPhase::Total` must wrap the *whole* iteration, gaps between
        // sub-phases included — `TrainingProfiler::iterations_per_second`
        // divides by it and reports `0.0` while it is never recorded.  Summing
        // the sub-phase EMAs instead would double-count nested scopes and miss
        // every gap, so the wrapper records the real wall clock, on the error
        // path too (a step that failed still consumed the time).
        let t_total = std::time::Instant::now();
        let result = self.train_step_inner();
        self.profiler
            .record(TrainingPhase::Total, t_total.elapsed().as_micros() as u64);
        result
    }

    /// The body of [`Trainer::train_step`], wrapped by the `Total` phase timer.
    fn train_step_inner(&mut self) -> Result<StepOutput, TrainerError> {
        self.iteration += 1;
        let iter = self.iteration;

        // 1. Get current timestep for SDS (used for logging and weighting).
        let current_timestep = self.diffusion_config.current_timestep(iter);
        let sds_weight = self.diffusion_generator.sds_weight(iter);

        // 2. Sample random cameras.
        let cameras = self.sample_cameras();

        // 3. Render current model from each camera  [Forward pass].
        let t_fwd = std::time::Instant::now();
        let rendered = self.render_views(&cameras);
        self.profiler
            .record(TrainingPhase::Forward, t_fwd.elapsed().as_micros() as u64);
        let rendered = rendered?;

        // 4. Generate diffusion targets (or fallback to rendered)  [DiffusionTarget].
        let t_diff = std::time::Instant::now();
        let (targets, used_diffusion) =
            self.generate_diffusion_targets(&cameras, &rendered, iter)?;
        self.profiler.record(
            TrainingPhase::DiffusionTarget,
            t_diff.elapsed().as_micros() as u64,
        );

        // 5. Compute photometric loss  [LossComputation].
        let t_loss = std::time::Instant::now();
        let mut loss_output = self.loss_computer.compute(
            &rendered,
            &targets,
            self.raster_config.image_width as usize,
            self.raster_config.image_height as usize,
            &self.model,
            self.mesh.as_ref(),
        );

        // 5b. Multi-view consistency of the *diffusion targets*.  Without this
        // the configured `view_consistency_weight` reached nothing at all: the
        // generator exposes the loss but the loop never called it.  It is a
        // penalty on the pseudo-GT, so it is meaningless on the self-supervised
        // fallback (where the targets ARE the renders).
        let view_consistency = if used_diffusion {
            self.diffusion_generator.view_consistency_loss().compute(
                &targets,
                &cameras,
                None,
                self.raster_config.image_width as usize,
                self.raster_config.image_height as usize,
            )
        } else {
            0.0
        };
        self.profiler.record(
            TrainingPhase::LossComputation,
            t_loss.elapsed().as_micros() as u64,
        );

        // 6. Compute the distillation (SDS) loss (only if using diffusion targets).
        // The reported value carries the same `sds_weight · w(t)` factors that
        // `DiffusionTargetGenerator::compute_sds_gradient` folds into the
        // image-space gradient, so loss and descent direction agree.
        let use_sds = used_diffusion && sds_weight > 0.0;
        let sds_loss_value = if use_sds {
            self.sds_loss.compute(&rendered, &targets, current_timestep) * sds_weight
        } else {
            0.0
        };
        // `LossOutput::sds` is documented as "set by the trainer for logging".
        loss_output.sds = sds_loss_value;

        // 7. Backward pass — GPU rasterizer backward  [Backward].
        // `compute_gradients` differentiates the *configured* loss (photometric
        // weights + distillation residual) and adds the analytic regularisation
        // gradients, so the optimiser descends the objective reported above.
        let t_bwd = std::time::Instant::now();
        let gradients = self.compute_gradients(&rendered, &targets, iter, use_sds);
        self.profiler
            .record(TrainingPhase::Backward, t_bwd.elapsed().as_micros() as u64);
        let mut gradients = gradients?;

        // 8+9. Mixed-precision gradient handling: scale (so an FP16/BF16-range
        // overflow becomes observable), unscale, and make ONE overflow
        // decision across all six groups, updating the dynamic loss scale.
        // `MixedPrecisionTrainer::process` is a no-op for `Float32`.
        let should_optimizer_step = self.mp_trainer.process(&mut gradients);
        if !should_optimizer_step {
            tracing::warn!(
                iteration = iter,
                precision = self.config.precision.label(),
                scale = self.mp_trainer.scaler.scale(),
                "Gradient overflow detected — skipping optimizer step"
            );
        }

        // 10. Optimiser step (skipped on gradient overflow, and while a
        // gradient-accumulation window is still filling)  [Optimize].
        let mut clip_stats = None;
        let mut optimizer_stepped = false;
        let lr_scale = self.scheduled_lr_scale(iter);
        if should_optimizer_step {
            let t_opt = std::time::Instant::now();
            // Accumulation returns the window mean once the window is full;
            // `None` means "keep accumulating, do not step yet".
            if let Some(mut effective) = self.stage_gradients(&gradients)? {
                clip_stats = self.clip_gradients(&mut effective)?;
                self.optimizer.set_lr_scale(lr_scale)?;
                self.optimizer.step(&mut self.model, &effective, iter)?;
                optimizer_stepped = true;
                if let Some(ema) = self.ema.as_mut() {
                    ema.update(&self.model);
                }
            }
            self.profiler
                .record(TrainingPhase::Optimize, t_opt.elapsed().as_micros() as u64);
        }

        // 11. Density control  [DensityControl].
        let t_density = std::time::Instant::now();
        self.density_controller.accumulate_gradients(&gradients);

        if self.should_densify(iter) {
            let result = self
                .density_controller
                .densify_and_prune(&mut self.model, &mut self.rng);
            self.optimizer
                .handle_densify(&result.keep_mask, result.num_added);
            // Every per-Gaussian buffer is now stale: the accumulator would
            // reject the next `accumulate` on a length mismatch, and the EMA
            // shadow would average unrelated Gaussians.  Note this must key on
            // whether the *membership* changed, not on the count: pruning K
            // and adding K leaves the length identical while replacing which
            // Gaussian sits at every index after the first prune.
            let pruned = result.keep_mask.iter().any(|kept| !kept);
            if pruned || result.num_added > 0 {
                self.reset_size_dependent_state();
            }
        }

        // 12. Opacity reset.
        if iter > 0
            && self.config.opacity_reset_interval > 0
            && iter.is_multiple_of(self.config.opacity_reset_interval)
        {
            DensityController::reset_opacity(&mut self.model, self.config.init.initial_opacity);
        }
        self.profiler.record(
            TrainingPhase::DensityControl,
            t_density.elapsed().as_micros() as u64,
        );

        // 13. Record metrics  [Metrics].
        let t_metrics = std::time::Instant::now();
        let psnr_val = if !rendered.is_empty() && !targets.is_empty() {
            metrics::psnr(&rendered[0], &targets[0])
        } else {
            0.0
        };
        let ssim_val = if !rendered.is_empty() && !targets.is_empty() {
            metrics::ssim(
                &rendered[0],
                &targets[0],
                self.raster_config.image_width as usize,
                self.raster_config.image_height as usize,
            )
        } else {
            0.0
        };
        self.metric_tracker.record(
            iter,
            psnr_val,
            ssim_val,
            loss_output.total + sds_loss_value + view_consistency,
        );

        // 14. TensorBoard logging.
        self.log_to_tensorboard(
            iter,
            &loss_output,
            sds_loss_value,
            psnr_val,
            ssim_val,
            &rendered,
            &gradients,
        )?;
        self.profiler.record(
            TrainingPhase::Metrics,
            t_metrics.elapsed().as_micros() as u64,
        );

        Ok(StepOutput {
            iteration: iter,
            loss: loss_output,
            num_gaussians: self.model.len(),
            sds_loss: sds_loss_value,
            used_diffusion,
            diffusion_timestep: current_timestep,
            view_consistency,
            optimizer_stepped,
            lr_scale,
            clip_stats,
        })
    }

    // -----------------------------------------------------------------------
    // Configured loop components
    // -----------------------------------------------------------------------

    /// The learning-rate multiplier the configured schedule produces at
    /// `iteration`, or `1.0` when no schedule is configured.
    ///
    /// Schedules are built with `base_lr = 1.0` (see
    /// [`crate::config::LrScheduleConfig`]) so their value *is* the multiplier;
    /// a non-finite or negative one would poison every parameter group, so it
    /// degrades to `1.0` with a warning rather than reaching the optimizer.
    fn scheduled_lr_scale(&self, iteration: u32) -> f32 {
        let Some(scheduler) = self.lr_scheduler.as_ref() else {
            return 1.0;
        };
        let raw = scheduler.lr_at(iteration as usize) as f32;
        if raw.is_finite() && raw >= 0.0 {
            raw
        } else {
            tracing::warn!(
                iteration,
                value = raw,
                "lr schedule produced a non-finite/negative multiplier — using 1.0"
            );
            1.0
        }
    }

    /// Feed this step's gradients through the accumulation window.
    ///
    /// Returns the gradients the optimizer should consume: `gradients`
    /// unchanged when accumulation is disabled, the window mean when the
    /// window just filled, and `None` while it is still filling.
    ///
    /// # Errors
    ///
    /// [`TrainerError::Training`] wrapping an
    /// [`crate::gradient_accumulation::AccumulationError`], and
    /// [`TrainerError::GradientSizeMismatch`] if the window's buffers do not
    /// match the current model — both mean the accumulated update would be
    /// wrong, which must not be silently applied.
    fn stage_gradients(
        &mut self,
        gradients: &Gradients,
    ) -> Result<Option<Gradients>, TrainerError> {
        let Some(accumulator) = self.gradient_accumulator.as_mut() else {
            return Ok(Some(gradients.clone()));
        };

        // A window always starts from buffers sized to the *current* model.
        // Density control clears the window when it changes the Gaussian set,
        // so `steps_accumulated == 0` is the point at which the new sizes are
        // adopted; mid-window the sizes cannot have changed, and
        // `accumulate` would report a `LengthMismatch` if they somehow had.
        if accumulator.steps_accumulated == 0 {
            accumulator
                .initialize(&gradients.group_sizes())
                .map_err(|e| TrainerError::Training(format!("gradient_accumulation: {e}")))?;
        }
        accumulator
            .accumulate(&gradients.to_group_vecs(), 1)
            .map_err(|e| TrainerError::Training(format!("gradient_accumulation: {e}")))?;

        if !accumulator.should_update() {
            return Ok(None);
        }
        let groups = accumulator
            .apply()
            .map_err(|e| TrainerError::Training(format!("gradient_accumulation: {e}")))?;
        let mut effective = gradients.clone();
        effective.set_from_group_vecs(&groups)?;
        Ok(Some(effective))
    }

    /// Clip `gradients` in place with the configured clipper, if any.
    ///
    /// # Errors
    ///
    /// [`TrainerError::GradientExplosion`] when the clipper reports a
    /// non-finite gradient norm: no threshold can rescue a `NaN`/`Inf` update,
    /// and stepping on one destroys the model silently.
    fn clip_gradients(
        &mut self,
        gradients: &mut Gradients,
    ) -> Result<Option<ClipStats>, TrainerError> {
        let configured_threshold = self.config.gradient_clip.threshold();
        let Some(clipper) = self.gradient_clipper.as_mut() else {
            return Ok(None);
        };
        // Read the EMA *before* the step: a non-finite norm poisons it, so the
        // post-step value would be the NaN we are trying to report about
        // rather than the threshold that was actually in force.
        let pre_step_ema = clipper.ema_norm();
        let mut groups = gradients.to_group_vecs();
        let stats = clipper
            .step(&mut groups)
            .map_err(|e| TrainerError::Training(format!("gradient_clip: {e}")))?;
        if !stats.original_norm.is_finite() {
            return Err(TrainerError::GradientExplosion {
                norm: stats.original_norm,
                // `Adaptive` has no fixed threshold; report the EMA it clips at.
                threshold: configured_threshold.unwrap_or(pre_step_ema),
            });
        }
        gradients.set_from_group_vecs(&groups)?;
        Ok(Some(stats))
    }

    /// Drop state whose buffers are indexed by Gaussian.
    ///
    /// Called after adaptive density control actually changed the model's
    /// membership (pruned or appended).  Both the accumulation window and the
    /// EMA shadow address Gaussians positionally, so after a prune every index
    /// refers to a different Gaussian even when the total count is unchanged;
    /// carrying either forward would blend unrelated parameters.  The caller
    /// skips this when nothing changed, so a decay near `1.0` is not restarted
    /// on every `density_control_interval` for no reason.
    fn reset_size_dependent_state(&mut self) {
        if let Some(accumulator) = self.gradient_accumulator.as_mut() {
            if accumulator.steps_accumulated > 0 {
                tracing::debug!(
                    steps = accumulator.steps_accumulated,
                    "densification resized the model — discarding the partial \
                     gradient-accumulation window"
                );
            }
            // `clear()` zeroes the buffers and resets `steps_accumulated`;
            // `stage_gradients` re-`initialize`s at the start of every window,
            // so the new sizes are picked up there.
            accumulator.clear();
        }
        if let Some(decay) = self.config.ema_decay {
            tracing::debug!(
                num_gaussians = self.model.len(),
                "density control changed the Gaussian set — restarting the EMA average"
            );
            self.ema = Some(GaussianEma::new(&self.model, decay));
        }
    }

    /// A copy of the model carrying the EMA shadow weights, when EMA is on.
    ///
    /// This is normally the better model to evaluate and to ship: it averages
    /// out the per-step noise the raw weights carry.
    ///
    /// **Caveat**: adaptive density control changes the Gaussian count, and the
    /// shadow is indexed by Gaussian, so it is restarted from the live weights
    /// whenever that happens.  Immediately after a densify/prune the "average"
    /// is therefore just the current model, and it needs on the order of
    /// `1/(1 − ema_decay)` further steps to become meaningful again.
    pub fn ema_model(&self) -> Option<GaussianModel> {
        let ema = self.ema.as_ref()?;
        let mut averaged = self.model.clone();
        ema.apply_to(&mut averaged);
        Some(averaged)
    }

    // -----------------------------------------------------------------------
    // Checkpoint helpers
    // -----------------------------------------------------------------------

    /// Save the current state to a JSON checkpoint file.
    ///
    /// This always writes the **live** weights, EMA configuration or not, so
    /// resuming continues the real trajectory.  For the averaged weights use
    /// [`Trainer::save_ema_checkpoint`] explicitly — silently substituting them
    /// here would be wrong whenever adaptive density control has recently
    /// restarted the average (see [`Trainer::ema_model`]).
    pub fn save_checkpoint(&self, path: &Path) -> Result<(), TrainerError> {
        let data = checkpoint::build_checkpoint(
            &self.model,
            &self.optimizer,
            self.iteration,
            &self.metric_tracker,
        );
        checkpoint::save_checkpoint(path, &data)
    }

    /// Save the **EMA shadow** weights to a JSON checkpoint file.
    ///
    /// The optimizer state written alongside is the live one, so the file is
    /// still resumable — but the weights are the average, which is usually the
    /// better model to evaluate or ship.
    ///
    /// # Errors
    ///
    /// [`TrainerError::InvalidConfig`] when no EMA is configured: writing the
    /// raw weights under this name would be a silent lie about what the file
    /// contains.  Also propagates any I/O or serialisation failure.
    pub fn save_ema_checkpoint(&self, path: &Path) -> Result<(), TrainerError> {
        let averaged = self.ema_model().ok_or_else(|| {
            TrainerError::InvalidConfig(
                "save_ema_checkpoint requires TrainingConfig::ema_decay to be set".into(),
            )
        })?;
        let data = checkpoint::build_checkpoint(
            &averaged,
            &self.optimizer,
            self.iteration,
            &self.metric_tracker,
        );
        checkpoint::save_checkpoint(path, &data)
    }

    /// Return a formatted profiling report for all training phases.
    ///
    /// Shows per-phase timing statistics (count, total, mean, min, max, EMA).
    /// Returns `"(profiler disabled)"` when profiling is not enabled.
    pub fn profiler_report(&self) -> String {
        self.profiler.format_report()
    }

    // -----------------------------------------------------------------------
    // TensorBoard logging
    // -----------------------------------------------------------------------

    /// Log training metrics to TensorBoard.
    ///
    /// Logs scalars (losses, metrics, learning rates) every step,
    /// and images/histograms at configured intervals.
    #[allow(clippy::too_many_arguments)]
    fn log_to_tensorboard(
        &mut self,
        iteration: u32,
        loss_output: &LossOutput,
        sds_loss: f32,
        psnr: f32,
        ssim: f32,
        rendered_images: &[Vec<f32>],
        gradients: &Gradients,
    ) -> Result<(), TrainerError> {
        if !self.tensorboard_logger.is_enabled() {
            return Ok(());
        }

        // Get current learning rates from optimizer config
        let lr = self.get_current_learning_rates(iteration);

        // Log step metrics (losses, PSNR, SSIM, learning rates)
        self.tensorboard_logger.log_step(
            iteration,
            loss_output.total,
            psnr,
            ssim,
            self.model.len(),
            &lr,
        )?;

        // Log individual loss components
        // Compute total regularization from individual components
        let total_reg = loss_output.position_reg
            + loss_output.scale_reg
            + loss_output.opacity_reg
            + loss_output.normal
            + loss_output.gradient_penalty;
        self.tensorboard_logger.log_losses(
            iteration,
            loss_output.l1,
            loss_output.ssim,
            loss_output.lpips,
            sds_loss,
            total_reg,
        )?;

        // Log rendered image (first view)
        if !rendered_images.is_empty() {
            self.tensorboard_logger.log_image(
                "render/view_0",
                &rendered_images[0],
                self.raster_config.image_width,
                self.raster_config.image_height,
                iteration,
            )?;
        }

        // Log gradient histograms
        self.tensorboard_logger.log_gradient_histogram(
            "gradients/position",
            &gradients.position,
            iteration,
        )?;
        self.tensorboard_logger.log_gradient_histogram(
            "gradients/rotation",
            &gradients.rotation,
            iteration,
        )?;
        self.tensorboard_logger.log_gradient_histogram(
            "gradients/scale",
            &gradients.scale,
            iteration,
        )?;
        self.tensorboard_logger.log_gradient_histogram(
            "gradients/opacity",
            &gradients.opacity,
            iteration,
        )?;

        Ok(())
    }

    /// Get the current learning rates for all parameter groups.
    ///
    /// The position rate comes straight from [`GaussianOptimizer::position_lr`]
    /// instead of re-deriving the decay schedule: the duplicate could disagree
    /// with the optimizer and divided by `position_lr_decay_steps` with no zero
    /// guard, logging `NaN` for `position_lr_decay_steps = 0`.
    fn get_current_learning_rates(&self, iteration: u32) -> LearningRates {
        let position_lr = self.optimizer.position_lr(iteration);

        LearningRates::from_config(
            position_lr,
            self.config.optimizer.lr_rotation,
            self.config.optimizer.lr_scale,
            self.config.optimizer.lr_opacity,
            self.config.optimizer.lr_sh,
        )
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Linearly-annealed classifier-free-guidance scale at the current iteration.
    ///
    /// This is the value the diffusion generator actually uses: the constructors
    /// mirror the [`TrainingConfig`] guidance fields into the
    /// [`DiffusionTargetConfig`] that
    /// [`DiffusionTargetGenerator::annealed_guidance_scale`] reads (see
    /// `apply_guidance_schedule`), so both schedules agree by construction.
    /// `guidance_anneal_steps == 0` means "no annealing": the denominator is
    /// clamped to `1` so the scale sits at the end value instead of being `NaN`.
    pub fn current_guidance_scale(&self) -> f32 {
        let denom = (self.config.guidance_anneal_steps as f32).max(1.0);
        let t = (self.iteration as f32 / denom).min(1.0);
        self.config.guidance_scale_start
            + t * (self.config.guidance_scale_end - self.config.guidance_scale_start)
    }

    /// Sample `views_per_step` random cameras on a sphere around the origin.
    fn sample_cameras(&mut self) -> Vec<Camera> {
        let n = self.config.views_per_step;
        let w = self.raster_config.image_width;
        let h = self.raster_config.image_height;
        let focal = w as f32 * 1.5;

        (0..n)
            .map(|_| {
                let theta: f32 = self.rng.random::<f32>() * 2.0 * std::f32::consts::PI;
                let phi: f32 = (self.rng.random::<f32>() * 2.0 - 1.0).acos();
                let radius: f32 = 0.6;

                let x = radius * phi.sin() * theta.cos();
                let y = radius * phi.sin() * theta.sin();
                let z = radius * phi.cos();

                let eye = na::Vector3::new(x, y, z);
                let forward = (-eye).normalize();
                let world_up = na::Vector3::new(0.0, 1.0, 0.0);
                let right = forward.cross(&world_up).normalize();
                // Re-derive up to guarantee orthonormality.
                let up = right.cross(&forward);

                let rotation = na::Matrix3::from_columns(&[right, up, -forward]).transpose();
                let translation = -(rotation * eye);

                Camera {
                    rotation,
                    translation,
                    focal_x: focal,
                    focal_y: focal,
                    cx: w as f32 / 2.0,
                    cy: h as f32 / 2.0,
                    width: w,
                    height: h,
                    near: 0.01,
                    far: 10.0,
                }
            })
            .collect()
    }

    /// Render the current Gaussian model from each camera using the GPU
    /// rasterizer.
    ///
    /// Returns one flat HWC `Vec<f32>` (RGB, length `W×H×3`) per view.
    ///
    /// # Errors
    ///
    /// Propagates any [`oxigaf_render::RenderError`] from the forward pass.  A
    /// failed render must *not* be replaced by a fabricated background image:
    /// the caller would build a loss and a full gradient from it and step the
    /// optimizer, so a transient device error would silently corrupt the model.
    fn render_views(&mut self, cameras: &[Camera]) -> Result<Vec<Vec<f32>>, TrainerError> {
        if cameras.is_empty() {
            return Ok(Vec::new());
        }

        let w = self.raster_config.image_width;
        let h = self.raster_config.image_height;
        let npx = (w as usize) * (h as usize);

        // Upload the latest model parameters to the GPU.
        self.rasterizer.upload_gaussians(&self.model);

        let mut views = Vec::with_capacity(cameras.len());
        for cam in cameras {
            let render_cam = camera_to_render_camera(cam, w, h);
            let output = self.rasterizer.forward(&self.model, &render_cam)?;

            // Convert RGBA [H*W*4] → RGB [H*W*3].  Chunked copies keep this a
            // single allocation with no per-channel `push` and no per-pixel
            // bounds checks; pixels missing from a short readback stay black.
            let mut rgb = vec![0.0_f32; npx * 3];
            for (dst, src) in rgb
                .chunks_exact_mut(3)
                .zip(output.color_data.chunks_exact(4))
            {
                dst.copy_from_slice(&src[..3]);
            }
            views.push(rgb);
        }

        Ok(views)
    }

    /// Generate multi-view diffusion targets.
    ///
    /// When a diffusion pipeline is loaded and we're past the warmup period,
    /// uses the diffusion model to generate pseudo-GT targets for SDS training.
    ///
    /// Otherwise, falls back to returning the rendered images themselves (self-
    /// supervised mode where the gradient comes from comparing against the
    /// current render).
    ///
    /// # Arguments
    /// * `cameras` - Camera views for target generation.
    /// * `rendered` - Current rendered images.
    /// * `iteration` - Current training iteration.
    ///
    /// # Returns
    /// Generated target images (or rendered images as fallback).
    fn generate_diffusion_targets(
        &mut self,
        cameras: &[Camera],
        rendered: &[Vec<f32>],
        iteration: u32,
    ) -> Result<(Vec<Vec<f32>>, bool), TrainerError> {
        if cameras.is_empty() || rendered.is_empty() {
            return Ok((Vec::new(), false));
        }

        // During warmup or if pipeline not loaded, use self-supervised fallback
        if self.diffusion_generator.is_warmup(iteration) || !self.diffusion_generator.is_loaded() {
            tracing::trace!(
                "generate_diffusion_targets: iteration {} (warmup={}, loaded={}), using self-supervised mode",
                iteration,
                self.diffusion_generator.is_warmup(iteration),
                self.diffusion_generator.is_loaded()
            );
            return Ok((rendered.to_vec(), false));
        }

        // Generate diffusion targets
        let width = self.raster_config.image_width;
        let height = self.raster_config.image_height;

        tracing::debug!(
            iteration,
            guidance_scale = self.current_guidance_scale(),
            "Generating diffusion targets with annealed classifier-free guidance"
        );

        // Per-view FLAME normal maps are the geometric conditioning the
        // multi-view U-Net expects.  Without an installed mesh there is
        // nothing to render, and the generator falls back to zero latents.
        let normal_maps = self.render_normal_maps(cameras);

        let targets = match self.diffusion_generator.generate_targets_with_normals(
            rendered,
            cameras,
            normal_maps.as_deref(),
            iteration,
            width,
            height,
        ) {
            Ok(targets) => targets,
            Err(e) => {
                tracing::warn!(
                    "Diffusion target generation failed: {}, falling back to rendered",
                    e
                );
                return Ok((rendered.to_vec(), false));
            }
        };

        // The pipeline decodes at `DiffusionConfig::image_size`, which is NOT
        // necessarily the rasterizer's resolution.  Every downstream consumer
        // (the loss, the image-space gradient, the SDS residual) indexes both
        // buffers by the *rasterizer's* pixel count, so a mismatch would be
        // absorbed per pixel and quietly corrupt the gradient.  A resolution
        // disagreement is a configuration error that recurs every step, so it
        // is reported rather than degraded into a silent self-supervised run.
        let expected = (width as usize) * (height as usize) * 3;
        for (idx, target) in targets.iter().enumerate() {
            if target.len() != expected {
                tracing::error!(
                    view = idx,
                    expected,
                    actual = target.len(),
                    raster_width = width,
                    raster_height = height,
                    "Diffusion target resolution does not match the rasterizer — \
                     set DiffusionConfig::image_size to the raster image size"
                );
                return Err(TrainerError::ImageDimensionMismatch {
                    expected,
                    actual: target.len(),
                });
            }
        }

        Ok((targets, true))
    }

    /// Render one FLAME normal map per camera, when a mesh is installed.
    ///
    /// Returns HWC RGB buffers in `[0, 1]` at the rasterizer's resolution —
    /// the encoding [`DiffusionTargetGenerator::generate_targets_with_normals`]
    /// documents.  `None` when no mesh is set, which is the honest signal that
    /// there is no geometric conditioning to hand over (as opposed to handing
    /// over a black image that would read as a valid, flat surface).
    fn render_normal_maps(&self, cameras: &[Camera]) -> Option<Vec<Vec<f32>>> {
        let mesh = self.mesh.as_ref()?;
        let maps = cameras
            .iter()
            .map(|camera| {
                let image = NormalMapRenderer::render(mesh, camera);
                image
                    .as_raw()
                    .iter()
                    .map(|&b| f32::from(b) / 255.0)
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>();
        Some(maps)
    }

    /// Compute parameter gradients via the GPU backward rasterization pass.
    ///
    /// For every view this builds ∂L/∂pixel of the **configured** loss — the
    /// weighted L1 / SSIM / MS-SSIM terms plus, when `use_sds` is set, the
    /// pixel-space distillation residual — dispatches the backward shader, and
    /// accumulates the per-Gaussian gradients into the [`Gradients`] struct the
    /// optimizer consumes.  The regularisers never reach the renderer and are
    /// differentiated analytically straight into the parameter gradients.
    ///
    /// # Errors
    ///
    /// Propagates a backward-pass [`oxigaf_render::RenderError`]: swallowing it
    /// leaves the surviving views normalised by the full `1/num_views`, i.e.
    /// silently scales the step down instead of reporting the failure.
    fn compute_gradients(
        &mut self,
        rendered: &[Vec<f32>],
        targets: &[Vec<f32>],
        iteration: u32,
        use_sds: bool,
    ) -> Result<Gradients, TrainerError> {
        let n = self.model.len();
        let sh_channels = ((self.model.sh_degree + 1) * (self.model.sh_degree + 1) * 3) as usize;

        // Differentiate the objective the *installed* `LossComputer` reports —
        // `loss_computer` is a public field, so its config, SSIM window and
        // MS-SSIM scale weights may differ from `self.config.loss` and the
        // library defaults.  Snapshotting them here also releases the borrow
        // before the rasterizer needs `&mut self`.
        let loss_config = self.loss_computer.config().clone();
        let ssim_kernel = self.loss_computer.ssim_kernel().to_vec();
        let ms_ssim_weights = *self.loss_computer.ms_ssim_weights();
        let sds_max_timestep = self.sds_loss.max_timestep;

        // Accumulate gradients across all views.
        let mut acc = Gradients::zeros(n, sh_channels);

        if rendered.is_empty() || targets.is_empty() {
            // No image term, but the regularisers still depend on the parameters.
            add_regularization_gradients(&loss_config, &self.model, &mut acc);
            return Ok(acc);
        }

        let w = self.raster_config.image_width as usize;
        let h = self.raster_config.image_height as usize;
        let npx = w * h;
        let num_views = rendered.len().min(targets.len());
        let spec = PhotometricSpec {
            config: &loss_config,
            ssim_kernel: &ssim_kernel,
            ms_ssim_weights: &ms_ssim_weights,
            num_views,
        };

        for v in 0..num_views {
            // ∂L/∂pixel of the configured photometric objective, averaged over
            // views exactly like `LossComputer::compute` averages the loss.
            let mut grad_rgb = photometric_pixel_gradient(&spec, &rendered[v], &targets[v], w, h);

            // Pixel-space distillation gradient w(t)·sds_weight·(render − target),
            // renormalised to the "mean over pixels, mean over views" scale the
            // reported SDS loss uses, so the two cannot drift apart.
            if use_sds {
                // Same DDPM horizon the reported `SdsLoss` weights with, so
                // `w(t)` is identical in the logged value and the descent
                // direction.
                let sds_grad = self.diffusion_generator.compute_sds_gradient_with_horizon(
                    &rendered[v],
                    &targets[v],
                    iteration,
                    sds_max_timestep,
                );
                let elems = rendered[v].len().max(1);
                let norm = 2.0 / (num_views as f32 * elems as f32);
                for (dst, g) in grad_rgb.iter_mut().zip(sds_grad.iter()) {
                    *dst += g * norm;
                }
            }

            // Scatter into the RGBA buffer the backward shader expects; the
            // alpha gradient stays 0 (the loss never sees alpha).
            let mut grad_image = vec![0.0_f32; npx * 4];
            for (dst, src) in grad_image.chunks_exact_mut(4).zip(grad_rgb.chunks_exact(3)) {
                dst[..3].copy_from_slice(src);
            }

            let gpu_grads = self.rasterizer.backward(&self.model, &grad_image)?;

            // Flatten GPU gradient arrays and accumulate into the CPU buffer.
            add_vec(&mut acc.position, &flatten(&gpu_grads.grad_positions));
            add_vec(&mut acc.rotation, &flatten(&gpu_grads.grad_rotations));
            add_vec(&mut acc.scale, &flatten(&gpu_grads.grad_scales));
            add_vec(&mut acc.opacity, &gpu_grads.grad_opacities);
            add_vec(&mut acc.sh, &gpu_grads.grad_sh_coeffs);
        }

        add_regularization_gradients(&loss_config, &self.model, &mut acc);

        Ok(acc)
    }

    /// Whether adaptive density control should run this iteration.
    fn should_densify(&self, iteration: u32) -> bool {
        iteration >= self.config.density_control_start
            && iteration <= self.config.density_control_end
            && self.config.density_control_interval > 0
            && iteration.is_multiple_of(self.config.density_control_interval)
    }
}

// ---------------------------------------------------------------------------
// Free-standing helpers
// ---------------------------------------------------------------------------

/// Clamp applied by [`crate::loss::opacity_reg`] to the sigmoid opacity.
///
/// Inside the clamped tails the entropy term is constant, so its derivative is
/// zero there.
const OPACITY_ENTROPY_CLAMP: f32 = 1e-6;

/// Flatten a slice of fixed-size per-Gaussian gradient tuples.
fn flatten<const N: usize>(values: &[[f32; N]]) -> Vec<f32> {
    values.iter().flat_map(|v| v.iter().copied()).collect()
}

/// Element-wise in-place addition: `dst[i] += src[i]`.
fn add_vec(dst: &mut [f32], src: &[f32]) {
    let len = dst.len().min(src.len());
    for i in 0..len {
        dst[i] += src[i];
    }
}

/// Logistic sigmoid — matches the one [`crate::loss`] applies to opacity logits.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Mirror the top-level guidance-annealing schedule into the diffusion config.
///
/// The CLI writes and serialises the [`TrainingConfig`] guidance fields, but the
/// scale actually handed to the pipeline is read from [`DiffusionTargetConfig`]
/// by [`DiffusionTargetGenerator::annealed_guidance_scale`].  Copying them across
/// at construction is what makes the configured annealing reach the pipeline; the
/// merged config is validated afterwards, so an out-of-range guidance scale is
/// reported instead of silently ignored.  The legacy static
/// [`DiffusionTargetConfig::guidance_scale`] is left alone: it is only a fallback
/// for callers that do not anneal.
///
/// The top-level schedule wins; a divergent one on the passed
/// [`DiffusionTargetConfig`] is logged rather than dropped in silence.
fn apply_guidance_schedule(config: &TrainingConfig, diffusion_config: &mut DiffusionTargetConfig) {
    if diffusion_config.guidance_anneal_steps != config.guidance_anneal_steps
        || diffusion_config.guidance_scale_start != config.guidance_scale_start
        || diffusion_config.guidance_scale_end != config.guidance_scale_end
    {
        tracing::debug!(
            start = config.guidance_scale_start,
            end = config.guidance_scale_end,
            anneal_steps = config.guidance_anneal_steps,
            "Guidance annealing taken from TrainingConfig, overriding the diffusion config"
        );
    }
    diffusion_config.guidance_scale_start = config.guidance_scale_start;
    diffusion_config.guidance_scale_end = config.guidance_scale_end;
    diffusion_config.guidance_anneal_steps = config.guidance_anneal_steps;
}

/// Add the analytic gradients of the configured regularisation terms.
///
/// These depend on the parameters only — they never reach the renderer — so
/// their derivatives are added straight to the parameter gradients:
///
/// | term | value ([`crate::loss`]) | gradient |
/// |---|---|---|
/// | `w_position_reg` | `(1/N)·Σ‖offset‖²` | `2·w·offset/N` → [`Gradients::offset`] |
/// | `w_scale_reg` | `(1/3N)·Σ relu(e^log_s − m)²` | `2·w·relu(e^log_s − m)·e^log_s/(3N)` → [`Gradients::scale`] |
/// | `w_opacity_reg` | `(1/N)·Σ H(σ(x))` | `−w·x·σ(1−σ)/N` → [`Gradients::opacity`] |
///
/// The scale term is the **asymmetric** one [`crate::loss::scale_reg_with_max`]
/// reports (its per-axis derivative lives in
/// [`crate::loss::scale_reg_axis_gradient`]), evaluated at
/// [`LossConfig::w_scale_reg_max_scale`].  It is
/// deliberately *not* the derivative of a symmetric `mean(log_s²)`: that older
/// form pulled every log-scale towards `0.0` (a 1.0 world-unit Gaussian), so
/// using its derivative here would have applied a constant outward pressure
/// the reported loss no longer charges for — descending an objective nobody
/// asked for, in the opposite direction for ordinary undersized Gaussians.
///
/// The opacity entropy is evaluated on a clamped sigmoid; inside the clamped
/// tails the term is constant, so its gradient is zero there.  `w_normal` and
/// `w_gradient_penalty` are absent by construction: the trainer keeps no FLAME
/// mesh (the normal term is a constant `0.0`) and passes no external gradient
/// buffer to the penalty.
///
/// The photometric loss contributes **no** offset gradient: the rasterizer
/// consumes `GaussianAttributes::position` and never reads
/// `GaussianModel::local_offsets`, so `∂L_photometric/∂offset` is exactly zero
/// for the current render path — the position-regularisation term above is the
/// entire offset gradient.  A photometric one needs `Rasterizer::backward` to
/// differentiate the FLAME binding.
fn add_regularization_gradients(cfg: &LossConfig, model: &GaussianModel, acc: &mut Gradients) {
    let n = model.len();
    if n == 0 {
        return;
    }
    let inv_n = 1.0 / n as f32;

    if cfg.w_position_reg > 0.0 {
        let k = 2.0 * cfg.w_position_reg * inv_n;
        for (offset, grad) in model
            .local_offsets
            .iter()
            .zip(acc.offset.chunks_exact_mut(3))
        {
            grad[0] += k * offset[0];
            grad[1] += k * offset[1];
            grad[2] += k * offset[2];
        }
    }

    if cfg.w_scale_reg > 0.0 {
        let k = cfg.w_scale_reg * inv_n / 3.0;
        let max_scale = cfg.w_scale_reg_max_scale;
        for (g, grad) in model.gaussians.iter().zip(acc.scale.chunks_exact_mut(3)) {
            for (axis, slot) in grad.iter_mut().enumerate() {
                *slot += k * scale_reg_axis_gradient(g.scale[axis], max_scale);
            }
        }
    }

    if cfg.w_opacity_reg > 0.0 {
        let k = cfg.w_opacity_reg * inv_n;
        for (g, grad) in model.gaussians.iter().zip(acc.opacity.iter_mut()) {
            let s = sigmoid(g.opacity);
            // Outside the clamp the entropy is constant → zero gradient.
            if s > OPACITY_ENTROPY_CLAMP && s < 1.0 - OPACITY_ENTROPY_CLAMP {
                // dH/dσ = −logit, dσ/dlogit = σ(1−σ)
                *grad += k * (-g.opacity) * s * (1.0 - s);
            }
        }
    }
}

/// Convert an [`oxigaf_flame::Camera`] to the GPU-facing [`RenderCamera`].
///
/// `_w` / `_h` are unused: the intrinsics already carry the resolution through
/// `Camera::width` / `Camera::height`.  Exposed so any other loop rendering
/// through the same rasterizer (e.g. [`crate::meta_learning_avatar`]) builds the
/// identical view/projection matrices rather than a second, divergent version.
pub fn camera_to_render_camera(cam: &Camera, _w: u32, _h: u32) -> RenderCamera {
    // Build a 4×4 view matrix (column-major f32 array) from the Camera's
    // rotation (3×3) and translation (3).
    let mut view = [0.0f32; 16];
    for c in 0..3 {
        for r in 0..3 {
            // nalgebra is column-major, which matches our [col*4+row] layout.
            view[c * 4 + r] = cam.rotation[(r, c)];
        }
    }
    view[3 * 4] = cam.translation.x;
    view[3 * 4 + 1] = cam.translation.y;
    view[3 * 4 + 2] = cam.translation.z;
    view[3 * 4 + 3] = 1.0;

    // Build a perspective projection matrix (column-major).
    let fx = cam.focal_x;
    let fy = cam.focal_y;
    let cx = cam.cx;
    let cy = cam.cy;
    let w = cam.width as f32;
    let h = cam.height as f32;
    let near = cam.near;
    let far = cam.far;

    // OpenGL-style NDC perspective from intrinsics.
    let mut proj = [0.0f32; 16];
    proj[0] = 2.0 * fx / w; // col 0, row 0
    proj[5] = 2.0 * fy / h; // col 1, row 1
    proj[8] = -(2.0 * cx / w - 1.0); // col 2, row 0
    proj[9] = -(2.0 * cy / h - 1.0); // col 2, row 1
    proj[10] = -(far + near) / (far - near);
    proj[11] = -1.0;
    proj[14] = -2.0 * far * near / (far - near);

    // Camera position in world space = −R^T t
    let eye_x = -(cam.rotation[(0, 0)] * cam.translation.x
        + cam.rotation[(1, 0)] * cam.translation.y
        + cam.rotation[(2, 0)] * cam.translation.z);
    let eye_y = -(cam.rotation[(0, 1)] * cam.translation.x
        + cam.rotation[(1, 1)] * cam.translation.y
        + cam.rotation[(2, 1)] * cam.translation.z);
    let eye_z = -(cam.rotation[(0, 2)] * cam.translation.x
        + cam.rotation[(1, 2)] * cam.translation.y
        + cam.rotation[(2, 2)] * cam.translation.z);

    RenderCamera {
        view_matrix: view,
        proj_matrix: proj,
        position: [eye_x, eye_y, eye_z],
        focal: [fx, fy],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OptimizerConfig;
    use crate::loss::{opacity_reg, position_reg, scale_reg};
    use oxigaf_render::gaussian::GaussianAttributes;

    /// A [`LossConfig`] carrying only the three regularisation weights; the
    /// image terms are differentiated (and tested) in [`crate::image_gradient`].
    fn reg_cfg(w_position_reg: f32, w_scale_reg: f32, w_opacity_reg: f32) -> LossConfig {
        LossConfig {
            w_l1: 0.0,
            w_ssim: 0.0,
            w_ms_ssim: 0.0,
            w_lpips: 0.0,
            w_position_reg,
            w_scale_reg,
            w_opacity_reg,
            w_normal: 0.0,
            w_gradient_penalty: 0.0,
            gradient_penalty_threshold: 100.0,
            w_scale_reg_max_scale: crate::loss::MAX_REASONABLE_WORLD_SCALE,
        }
    }

    fn relative_error(analytic: f32, numeric: f32) -> f32 {
        let denom = analytic.abs().max(numeric.abs()).max(1e-6);
        (analytic - numeric).abs() / denom
    }

    fn tiny_model(n: usize) -> GaussianModel {
        let attr = GaussianAttributes {
            position: [0.0; 3],
            _pad0: 0.0,
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [-1.5, -2.0, -2.5],
            opacity: 0.5,
        };
        GaussianModel {
            gaussians: vec![attr; n],
            sh_coeffs: vec![0.0; n * 3],
            sh_degree: 0,
            face_indices: vec![0; n],
            barycentric: vec![[1.0, 0.0, 0.0]; n],
            local_offsets: vec![[0.05, -0.1, 0.2]; n],
            is_rigid: vec![false; n],
        }
    }

    // ---- regularisation gradients -----------------------------------------

    #[test]
    fn regularisation_gradients_match_finite_differences() {
        let cfg = reg_cfg(0.01, 0.02, 0.05);
        let model = tiny_model(3);
        let mut grads = Gradients::zeros(model.len(), 3);
        add_regularization_gradients(&cfg, &model, &mut grads);

        let eps = 1e-2_f32;
        // Each term is differenced against **only its own** contribution.
        // Summing all three first would make the central difference measure a
        // ~1e-7 delta on top of a ~3e-2 constant, where f32 round-off alone is
        // 2% of the delta — the parameters a term does not depend on cancel
        // exactly anyway, so nothing is lost by isolating them.
        let position_loss = |m: &GaussianModel| cfg.w_position_reg * position_reg(m);
        let scale_loss = |m: &GaussianModel| cfg.w_scale_reg * scale_reg(m);
        let opacity_loss = |m: &GaussianModel| cfg.w_opacity_reg * opacity_reg(m);

        for j in 0..3 {
            // Offsets — the whole point of the `Offset` parameter group.
            let mut plus = model.clone();
            let mut minus = model.clone();
            plus.local_offsets[1][j] += eps;
            minus.local_offsets[1][j] -= eps;
            let numeric = (position_loss(&plus) - position_loss(&minus)) / (2.0 * eps);
            assert!(
                relative_error(grads.offset[3 + j], numeric) < 0.01,
                "offset[{j}]: analytic {} vs numeric {numeric}",
                grads.offset[3 + j]
            );

            // Log-scales.
            let mut plus = model.clone();
            let mut minus = model.clone();
            plus.gaussians[2].scale[j] += eps;
            minus.gaussians[2].scale[j] -= eps;
            let numeric = (scale_loss(&plus) - scale_loss(&minus)) / (2.0 * eps);
            assert!(
                relative_error(grads.scale[6 + j], numeric) < 0.02,
                "scale[{j}]: analytic {} vs numeric {numeric}",
                grads.scale[6 + j]
            );
        }

        // Opacity logits.
        let mut plus = model.clone();
        let mut minus = model.clone();
        plus.gaussians[0].opacity += eps;
        minus.gaussians[0].opacity -= eps;
        let numeric = (opacity_loss(&plus) - opacity_loss(&minus)) / (2.0 * eps);
        assert!(
            relative_error(grads.opacity[0], numeric) < 0.01,
            "opacity: analytic {} vs numeric {numeric}",
            grads.opacity[0]
        );
    }

    #[test]
    fn offset_regularisation_gradient_is_analytic() {
        // Regression: `Gradients::offset` was never written at all, so the whole
        // `ParameterGroup::Offset` Adam state updated with a zero gradient.  It
        // now carries the exact `∂(w_position_reg·position_reg)/∂offset`; note
        // that this vanishes at the all-zero offset initialisation, which is a
        // property of the objective, not of the plumbing.
        let model = tiny_model(2);
        let mut grads = Gradients::zeros(model.len(), 3);
        add_regularization_gradients(&reg_cfg(0.01, 0.0, 0.0), &model, &mut grads);
        assert!(grads.offset.iter().any(|g| g.abs() > 0.0));

        // Zero weights must still leave every group untouched.
        let mut zeroed = Gradients::zeros(model.len(), 3);
        add_regularization_gradients(&reg_cfg(0.0, 0.0, 0.0), &model, &mut zeroed);
        assert!(zeroed.offset.iter().all(|g| *g == 0.0));
        assert!(zeroed.scale.iter().all(|g| *g == 0.0));
        assert!(zeroed.opacity.iter().all(|g| *g == 0.0));
    }

    // ---- schedules ---------------------------------------------------------

    #[test]
    fn guidance_schedule_reaches_the_diffusion_generator() {
        let config = TrainingConfig {
            guidance_scale_start: 9.0,
            guidance_scale_end: 2.0,
            guidance_anneal_steps: 100,
            ..Default::default()
        };
        let mut diffusion = DiffusionTargetConfig::default();
        apply_guidance_schedule(&config, &mut diffusion);
        assert!(diffusion.validate().is_ok());

        let generator = DiffusionTargetGenerator::new(diffusion);
        assert!((generator.annealed_guidance_scale(0) - 9.0).abs() < 1e-6);
        assert!((generator.annealed_guidance_scale(50) - 5.5).abs() < 1e-5);
        assert!((generator.annealed_guidance_scale(100) - 2.0).abs() < 1e-6);
        assert!((generator.annealed_guidance_scale(1_000) - 2.0).abs() < 1e-6);

        // `guidance_anneal_steps = 0` must not divide by zero.
        let zero_anneal = TrainingConfig {
            guidance_anneal_steps: 0,
            ..Default::default()
        };
        let mut diffusion = DiffusionTargetConfig::default();
        apply_guidance_schedule(&zero_anneal, &mut diffusion);
        let generator = DiffusionTargetGenerator::new(diffusion);
        assert!(generator.annealed_guidance_scale(0).is_finite());
        assert!(generator.annealed_guidance_scale(5).is_finite());
    }

    #[test]
    fn position_learning_rate_survives_zero_decay_steps() {
        // Regression: the trainer re-derived this schedule and divided by
        // `position_lr_decay_steps` with no zero guard, logging NaN.
        let opt_config = OptimizerConfig {
            position_lr_decay_steps: 0,
            ..Default::default()
        };
        let optimizer = GaussianOptimizer::new(&opt_config, &tiny_model(1));
        for iteration in [0_u32, 1, 10_000] {
            let lr = optimizer.position_lr(iteration);
            assert!(lr.is_finite() && lr > 0.0, "lr at {iteration} is {lr}");
        }
    }

    #[test]
    fn scale_regularisation_gradient_follows_the_asymmetric_penalty() {
        // Regression: `scale_reg` became `mean(relu(e^log_s − m)²)` but this
        // gradient stayed the derivative of the old symmetric `mean(log_s²)`,
        // so for every ordinary (undersized) Gaussian it pushed the log-scale
        // in the OPPOSITE direction to the reported loss.
        let cfg = reg_cfg(0.0, 1.0, 0.0);
        let model = tiny_model(1);
        let mut grads = Gradients::zeros(model.len(), 3);
        add_regularization_gradients(&cfg, &model, &mut grads);

        // Oversized axes (e^-1.5 ≈ 0.223 > 0.05) push the scale DOWN, i.e. a
        // positive gradient; the old symmetric form gave a negative one.
        for (axis, g) in grads.scale.iter().enumerate() {
            assert!(*g > 0.0, "scale[{axis}] gradient {g} must be positive");
        }

        // Inside the clamp the penalty is exactly constant → zero gradient.
        let mut small = tiny_model(1);
        small.gaussians[0].scale = [-5.0, -5.0, -5.0]; // e^-5 ≈ 0.0067 < 0.05
        let mut small_grads = Gradients::zeros(small.len(), 3);
        add_regularization_gradients(&cfg, &small, &mut small_grads);
        assert!(small_grads.scale.iter().all(|g| *g == 0.0));
    }

    #[test]
    fn configured_max_scale_reaches_the_regularisation_gradient() {
        // Regression (F289): the threshold was a hardcoded constant, so
        // `LossConfig::w_scale_reg_max_scale` changed the reported loss but not
        // the descent direction.
        let mut model = tiny_model(1);
        model.gaussians[0].scale = [-3.0, -3.0, -3.0]; // e^-3 ≈ 0.0498

        // Default threshold (0.05) leaves it inside the clamp: no gradient.
        let mut default_grads = Gradients::zeros(model.len(), 3);
        add_regularization_gradients(&reg_cfg(0.0, 1.0, 0.0), &model, &mut default_grads);
        assert!(default_grads.scale.iter().all(|g| *g == 0.0));

        // A tighter threshold makes the same Gaussian oversized.
        let tight = LossConfig {
            w_scale_reg_max_scale: 0.01,
            ..reg_cfg(0.0, 1.0, 0.0)
        };
        let mut tight_grads = Gradients::zeros(model.len(), 3);
        add_regularization_gradients(&tight, &model, &mut tight_grads);
        assert!(tight_grads.scale.iter().all(|g| *g > 0.0));
    }

    // ---- configured loop components ----------------------------------------

    #[test]
    fn loop_components_are_off_by_default_and_on_when_configured() {
        let model = tiny_model(2);
        let off = LoopComponents::from_config(&TrainingConfig::default(), &model)
            .expect("default config must build");
        assert!(off.lr_scheduler.is_none());
        assert!(off.gradient_clipper.is_none());
        assert!(off.gradient_accumulator.is_none());
        assert!(off.ema.is_none());

        let config = TrainingConfig {
            lr_schedule: crate::config::LrScheduleConfig::WarmupCosine {
                warmup_steps: 10,
                total_steps: 100,
                min_factor: 0.1,
            },
            gradient_clip: crate::config::GradientClipConfig::GlobalNorm { max_norm: 1.0 },
            gradient_accumulation_steps: 4,
            ema_decay: Some(0.99),
            ..Default::default()
        };
        let on = LoopComponents::from_config(&config, &model).expect("valid config must build");
        assert!(on.lr_scheduler.is_some());
        assert!(on.gradient_clipper.is_some());
        assert!(on.gradient_accumulator.is_some());
        assert!(on.ema.is_some());

        // An out-of-range component is reported, never silently dropped.
        let bad = TrainingConfig {
            gradient_clip: crate::config::GradientClipConfig::GlobalNorm { max_norm: -1.0 },
            ..Default::default()
        };
        assert!(LoopComponents::from_config(&bad, &model).is_err());
    }
}
