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
//! consistency needs a FLAME mesh the trainer does not track (its value is a
//! constant `0.0` here), and the gradient penalty is evaluated on an external
//! gradient buffer the trainer does not pass.
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

use oxigaf_flame::Camera;
use oxigaf_render::gaussian::GaussianModel;
use oxigaf_render::{RasterConfig, Rasterizer, RenderCamera};

use crate::checkpoint;
use crate::config::{LossConfig, TrainingConfig};
use crate::density::DensityController;
use crate::diffusion_target::{DiffusionTargetConfig, DiffusionTargetGenerator, SdsLoss};
use crate::loss::{gaussian_kernel_1d, LossComputer, LossOutput};
use crate::metrics::{self, MetricTracker};
use crate::mixed_precision::{LossScaler, MixedPrecisionTrainer};
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

        tracing::info!(
            "Trainer created: {} Gaussians, {} total iterations, warmup={} iters, tensorboard={}, precision={}, profiling={}",
            model.len(),
            config.total_iterations,
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
            iteration: 0,
            diffusion_generator,
            sds_loss,
            diffusion_config,
            tensorboard_logger,
            mp_trainer,
            profiler,
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
                    let path = dir.join(format!("ckpt_{:06}.json", self.iteration));
                    self.save_checkpoint(&path)?;
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
            None, // Mesh not tracked by trainer; normal consistency loss will be 0.0
        );
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

        // 8. Mixed-precision loss scaling before the overflow check: simulates
        // scaling the loss ahead of the backward pass, enabling dynamic overflow
        // detection.  BF16 needs it as much as FP16 — see
        // `TrainingPrecision::requires_scaling`.
        if self.config.precision.requires_scaling() {
            gradients.scale(self.mp_trainer.scaler.scale());
        }

        // 9. Unscale gradients and check for overflow (scaled precisions only).
        // Skips optimizer step on overflow and updates the loss scale adaptively.
        let should_optimizer_step = if self.config.precision.requires_scaling() {
            let inv_scale = 1.0 / self.mp_trainer.scaler.scale();
            gradients.scale(inv_scale);
            let overflow = LossScaler::has_overflow(&gradients.position)
                || LossScaler::has_overflow(&gradients.rotation)
                || LossScaler::has_overflow(&gradients.scale)
                || LossScaler::has_overflow(&gradients.opacity)
                || LossScaler::has_overflow(&gradients.sh)
                || LossScaler::has_overflow(&gradients.offset);
            self.mp_trainer.scaler.update(overflow);
            if overflow {
                tracing::warn!(
                    iteration = iter,
                    precision = self.config.precision.label(),
                    scale = self.mp_trainer.scaler.scale(),
                    "Gradient overflow detected — skipping optimizer step"
                );
            }
            !overflow
        } else {
            true
        };

        // 10. Optimiser step (skipped on gradient overflow)  [Optimize].
        if should_optimizer_step {
            let t_opt = std::time::Instant::now();
            self.optimizer.step(&mut self.model, &gradients, iter);
            self.profiler
                .record(TrainingPhase::Optimize, t_opt.elapsed().as_micros() as u64);
        }

        // 11. Density control.
        self.density_controller.accumulate_gradients(&gradients);

        if self.should_densify(iter) {
            let result = self
                .density_controller
                .densify_and_prune(&mut self.model, &mut self.rng);
            self.optimizer
                .handle_densify(&result.keep_mask, result.num_added);
        }

        // 12. Opacity reset.
        if iter > 0
            && self.config.opacity_reset_interval > 0
            && iter.is_multiple_of(self.config.opacity_reset_interval)
        {
            DensityController::reset_opacity(&mut self.model, self.config.init.initial_opacity);
        }

        // 13. Record metrics.
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
        self.metric_tracker
            .record(iter, psnr_val, ssim_val, loss_output.total + sds_loss_value);

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

        Ok(StepOutput {
            iteration: iter,
            loss: loss_output,
            num_gaussians: self.model.len(),
            sds_loss: sds_loss_value,
            used_diffusion,
            diffusion_timestep: current_timestep,
        })
    }

    // -----------------------------------------------------------------------
    // Checkpoint helpers
    // -----------------------------------------------------------------------

    /// Save the current state to a JSON checkpoint file.
    pub fn save_checkpoint(&self, path: &Path) -> Result<(), TrainerError> {
        let data = checkpoint::build_checkpoint(
            &self.model,
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

        match self
            .diffusion_generator
            .generate_targets(rendered, cameras, iteration, width, height)
        {
            Ok(targets) => Ok((targets, true)),
            Err(e) => {
                tracing::warn!(
                    "Diffusion target generation failed: {}, falling back to rendered",
                    e
                );
                Ok((rendered.to_vec(), false))
            }
        }
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

        // Accumulate gradients across all views.
        let mut acc = Gradients::zeros(n, sh_channels);

        if rendered.is_empty() || targets.is_empty() {
            // No image term, but the regularisers still depend on the parameters.
            add_regularization_gradients(&self.config.loss, &self.model, &mut acc);
            return Ok(acc);
        }

        let w = self.raster_config.image_width as usize;
        let h = self.raster_config.image_height as usize;
        let npx = w * h;
        let num_views = rendered.len().min(targets.len());

        for v in 0..num_views {
            // ∂L/∂pixel of the configured photometric objective, averaged over
            // views exactly like `LossComputer::compute` averages the loss.
            let mut grad_rgb = photometric_pixel_gradient(
                &self.config.loss,
                &rendered[v],
                &targets[v],
                w,
                h,
                num_views,
                &LossComputer::DEFAULT_MS_SSIM_WEIGHTS,
            );

            // Pixel-space distillation gradient w(t)·sds_weight·(render − target),
            // renormalised to the "mean over pixels, mean over views" scale the
            // reported SDS loss uses, so the two cannot drift apart.
            if use_sds {
                let sds_grad = self.diffusion_generator.compute_sds_gradient(
                    &rendered[v],
                    &targets[v],
                    iteration,
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

        add_regularization_gradients(&self.config.loss, &self.model, &mut acc);

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

/// SSIM stabiliser `(K₁·L)²` with `K₁ = 0.01`, `L = 1` — as in [`crate::loss`].
const SSIM_C1: f32 = 0.01 * 0.01;
/// SSIM stabiliser `(K₂·L)²` with `K₂ = 0.03`, `L = 1` — as in [`crate::loss`].
const SSIM_C2: f32 = 0.03 * 0.03;

/// Taps of the SSIM Gaussian window used by [`crate::loss::ssim_loss`].
const SSIM_KERNEL_TAPS: usize = 11;
/// Sigma of the SSIM Gaussian window used by [`crate::loss::ssim_loss`].
const SSIM_KERNEL_SIGMA: f32 = 1.5;
/// Taps of the smaller window [`crate::loss::ms_ssim_loss`] uses per scale.
const MS_SSIM_KERNEL_TAPS: usize = 7;
/// Sigma of the smaller window [`crate::loss::ms_ssim_loss`] uses per scale.
const MS_SSIM_KERNEL_SIGMA: f32 = 1.0;
/// Smallest dimension [`crate::loss::ms_ssim_loss`] still builds a scale for.
const MS_SSIM_MIN_DIM: usize = 7;
/// Maximum number of MS-SSIM scales (one per weight).
const MS_SSIM_MAX_SCALES: usize = 5;

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

/// Sub-gradient of `|x|`, taking `0` at the kink.
#[inline]
fn sign_or_zero(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
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
/// | `w_scale_reg` | `(1/3N)·Σ log_s²` | `2·w·log_s/(3N)` → [`Gradients::scale`] |
/// | `w_opacity_reg` | `(1/N)·Σ H(σ(x))` | `−w·x·σ(1−σ)/N` → [`Gradients::opacity`] |
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
        let k = 2.0 * cfg.w_scale_reg * inv_n / 3.0;
        for (g, grad) in model.gaussians.iter().zip(acc.scale.chunks_exact_mut(3)) {
            grad[0] += k * g.scale[0];
            grad[1] += k * g.scale[1];
            grad[2] += k * g.scale[2];
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

// ---------------------------------------------------------------------------
// Image-space loss gradients
// ---------------------------------------------------------------------------

/// Image-space gradient `∂L/∂rendered` of the configured photometric loss for
/// one view, as a flat HWC RGB buffer of length `width·height·3`.
///
/// Every term is differentiated at exactly the normalisation
/// [`LossComputer::compute`] uses — mean over pixels, mean over views
/// (`num_views`), configured weight — and a zero-weight term is skipped.
/// LPIPS is absent on purpose: [`LossComputer::compute`] reports a hard `0.0`
/// for it, so it is not part of the objective the trainer optimises.
fn photometric_pixel_gradient(
    cfg: &LossConfig,
    rendered: &[f32],
    target: &[f32],
    width: usize,
    height: usize,
    num_views: usize,
    ms_ssim_weights: &[f32; 5],
) -> Vec<f32> {
    let npx = width * height;
    let mut grad = vec![0.0_f32; npx * 3];
    if npx == 0 || rendered.is_empty() || target.is_empty() || num_views == 0 {
        return grad;
    }
    let view_norm = 1.0 / num_views as f32;

    // ---- L1: mean absolute error over every element of the view ------------
    if cfg.w_l1 > 0.0 {
        // `crate::loss::l1_loss` divides by `pred.len()`.
        let k = cfg.w_l1 * view_norm / rendered.len() as f32;
        for ((g, r), t) in grad.iter_mut().zip(rendered.iter()).zip(target.iter()) {
            *g += k * sign_or_zero(r - t);
        }
    }

    // ---- SSIM dissimilarity (1 − SSIM) -------------------------------------
    if cfg.w_ssim > 0.0 {
        let kernel = gaussian_kernel_1d(SSIM_KERNEL_TAPS, SSIM_KERNEL_SIGMA);
        let g = ssim_pixel_gradient(rendered, target, width, height, &kernel);
        let k = cfg.w_ssim * view_norm;
        for (dst, src) in grad.iter_mut().zip(g.iter()) {
            *dst += k * src;
        }
    }

    // ---- MS-SSIM dissimilarity (1 − MS-SSIM) -------------------------------
    if cfg.w_ms_ssim > 0.0 {
        let g = ms_ssim_pixel_gradient(rendered, target, width, height, ms_ssim_weights);
        let k = cfg.w_ms_ssim * view_norm;
        for (dst, src) in grad.iter_mut().zip(g.iter()) {
            *dst += k * src;
        }
    }

    grad
}

/// Windowed first- and second-order statistics of an image-pair channel.
///
/// All five maps use the same separable Gaussian window as the forward SSIM, so
/// the gradient is the exact adjoint away from the border (replicate padding is
/// only its own approximate transpose).
struct LocalStats {
    mu_x: Vec<f32>,
    mu_y: Vec<f32>,
    sigma_x2: Vec<f32>,
    sigma_y2: Vec<f32>,
    sigma_xy: Vec<f32>,
}

impl LocalStats {
    /// Compute the windowed statistics of one channel pair (`len = width·height`).
    fn compute(x: &[f32], y: &[f32], width: usize, height: usize, kernel: &[f32]) -> Self {
        let n = width * height;
        let xx: Vec<f32> = x.iter().take(n).map(|v| v * v).collect();
        let yy: Vec<f32> = y.iter().take(n).map(|v| v * v).collect();
        let xy: Vec<f32> = x.iter().zip(y.iter()).take(n).map(|(a, b)| a * b).collect();

        let mu_x = convolve_separable(x, width, height, kernel);
        let mu_y = convolve_separable(y, width, height, kernel);
        let mut sigma_x2 = convolve_separable(&xx, width, height, kernel);
        let mut sigma_y2 = convolve_separable(&yy, width, height, kernel);
        let mut sigma_xy = convolve_separable(&xy, width, height, kernel);
        for i in 0..n {
            sigma_x2[i] -= mu_x[i] * mu_x[i];
            sigma_y2[i] -= mu_y[i] * mu_y[i];
            sigma_xy[i] -= mu_x[i] * mu_y[i];
        }

        Self {
            mu_x,
            mu_y,
            sigma_x2,
            sigma_y2,
            sigma_xy,
        }
    }

    /// SSIM luminance term `l = (2μxμy + C₁)/(μx² + μy² + C₁)` at pixel `q`.
    #[inline]
    fn luminance(&self, q: usize) -> f32 {
        let (mu_x, mu_y) = (self.mu_x[q], self.mu_y[q]);
        (2.0 * mu_x * mu_y + SSIM_C1) / (mu_x * mu_x + mu_y * mu_y + SSIM_C1)
    }

    /// SSIM contrast-structure term `cs = (2σxy + C₂)/(σx² + σy² + C₂)` at `q`.
    #[inline]
    fn contrast_structure(&self, q: usize) -> f32 {
        (2.0 * self.sigma_xy[q] + SSIM_C2) / (self.sigma_x2[q] + self.sigma_y2[q] + SSIM_C2)
    }

    /// Accumulate `∂/∂x Σ_q [ wl(q)·l(q) + wcs(q)·cs(q) ]` into `out`.
    ///
    /// Every SSIM-family term in [`crate::loss`] is a weighted sum of the
    /// luminance and contrast-structure maps, so one adjoint pass serves all of
    /// them.  With `∂μx(q)/∂x(p) = G(p−q)`,
    /// `∂σx²(q)/∂x(p) = 2·G(p−q)·(x(p) − μx(q))` and
    /// `∂σxy(q)/∂x(p) = G(p−q)·(y(p) − μy(q))`, the sum over windows collapses
    /// into five convolutions with the same (symmetric) window.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_gradient(
        &self,
        x: &[f32],
        y: &[f32],
        width: usize,
        height: usize,
        kernel: &[f32],
        wl: &[f32],
        wcs: &[f32],
        out: &mut [f32],
    ) {
        let n = width * height;
        let mut au = vec![0.0_f32; n];
        let mut av = vec![0.0_f32; n];
        let mut avmu = vec![0.0_f32; n];
        let mut az = vec![0.0_f32; n];
        let mut azmu = vec![0.0_f32; n];

        for q in 0..n {
            let mu_x = self.mu_x[q];
            let mu_y = self.mu_y[q];
            let a1 = 2.0 * mu_x * mu_y + SSIM_C1;
            let b1 = mu_x * mu_x + mu_y * mu_y + SSIM_C1;
            let a2 = 2.0 * self.sigma_xy[q] + SSIM_C2;
            let b2 = self.sigma_x2[q] + self.sigma_y2[q] + SSIM_C2;

            // ∂l/∂μx = 2(μy·B₁ − μx·A₁)/B₁²
            au[q] = wl[q] * 2.0 * (mu_y * b1 - mu_x * a1) / (b1 * b1);
            // ∂cs/∂σxy = 2/B₂ and ∂cs/∂σx² = −2·A₂/B₂²
            let v = wcs[q] * 2.0 / b2;
            let z = wcs[q] * -2.0 * a2 / (b2 * b2);
            av[q] = v;
            avmu[q] = v * mu_y;
            az[q] = z;
            azmu[q] = z * mu_x;
        }

        let gu = convolve_separable(&au, width, height, kernel);
        let gv = convolve_separable(&av, width, height, kernel);
        let gvmu = convolve_separable(&avmu, width, height, kernel);
        let gz = convolve_separable(&az, width, height, kernel);
        let gzmu = convolve_separable(&azmu, width, height, kernel);

        for p in 0..n {
            out[p] += gu[p] + y[p] * gv[p] - gvmu[p] + x[p] * gz[p] - gzmu[p];
        }
    }
}

/// Separable 2-D convolution with replicate-boundary padding.
///
/// Mirrors the private helper behind [`crate::loss::ssim_loss`] so forward loss
/// and backward gradient see the same window.
fn convolve_separable(src: &[f32], width: usize, height: usize, kernel: &[f32]) -> Vec<f32> {
    let n = width * height;
    let mut out = vec![0.0_f32; n];
    if n == 0 || src.len() < n || kernel.is_empty() {
        return out;
    }
    let half = (kernel.len() / 2) as isize;

    // Horizontal pass.
    let mut tmp = vec![0.0_f32; n];
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let mut sum = 0.0_f32;
            for (i, &kv) in kernel.iter().enumerate() {
                let ix = clamp_index(x as isize + i as isize - half, width);
                sum += src[row + ix] * kv;
            }
            tmp[row + x] = sum;
        }
    }

    // Vertical pass.
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0_f32;
            for (i, &kv) in kernel.iter().enumerate() {
                let iy = clamp_index(y as isize + i as isize - half, height);
                sum += tmp[iy * width + x] * kv;
            }
            out[y * width + x] = sum;
        }
    }

    out
}

/// Clamp a (possibly negative) index into `0..len`.  `len` must be non-zero.
#[inline]
fn clamp_index(idx: isize, len: usize) -> usize {
    idx.clamp(0, len as isize - 1) as usize
}

/// Extract one interleaved HWC channel as a dense plane, zero-filling any pixel
/// the source is too short for (the same tolerance the forward loss applies).
fn extract_channel(img: &[f32], n_pixels: usize, channel: usize) -> Vec<f32> {
    (0..n_pixels)
        .map(|p| img.get(p * 3 + channel).copied().unwrap_or(0.0))
        .collect()
}

/// `∂(1 − SSIM)/∂pred` for one HWC RGB image pair.
///
/// The forward term is `1 − (1/3)·Σ_c mean_q S_c(q)` with `S = l·cs`, so
/// `∂(mean S)/∂l = cs/N` and `∂(mean S)/∂cs = l/N`.
fn ssim_pixel_gradient(
    pred: &[f32],
    target: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
) -> Vec<f32> {
    let n = width * height;
    let mut grad = vec![0.0_f32; n * 3];
    // `crate::loss::ssim_loss` returns a constant 0.0 for undersized buffers.
    if n == 0 || pred.len() < n * 3 || target.len() < n * 3 {
        return grad;
    }

    let outer = -1.0 / (3.0 * n as f32);
    let mut wl = vec![0.0_f32; n];
    let mut wcs = vec![0.0_f32; n];
    let mut chan = vec![0.0_f32; n];

    for c in 0..3 {
        let x = extract_channel(pred, n, c);
        let y = extract_channel(target, n, c);
        let stats = LocalStats::compute(&x, &y, width, height, kernel);

        for q in 0..n {
            wl[q] = outer * stats.contrast_structure(q);
            wcs[q] = outer * stats.luminance(q);
        }

        chan.fill(0.0);
        stats.accumulate_gradient(&x, &y, width, height, kernel, &wl, &wcs, &mut chan);
        for (p, &g) in chan.iter().enumerate() {
            grad[p * 3 + c] = g;
        }
    }

    grad
}

/// `∂(1 − MS-SSIM)/∂pred` for one HWC RGB image pair.
///
/// [`crate::loss::ms_ssim_loss`] combines *scalar* per-scale means,
/// `P = Πⱼ cs̄ⱼ^wⱼ · l̄_M^w_M`, so `∂(1−P)/∂cs̄ⱼ = −P·wⱼ/cs̄ⱼ` (likewise for the
/// coarsest luminance), followed by the adjoint of the box-downsampling chain.
/// Outside the `[0, 1]` clamp the forward term is constant → zero gradient.
fn ms_ssim_pixel_gradient(
    pred: &[f32],
    target: &[f32],
    width: usize,
    height: usize,
    weights: &[f32; 5],
) -> Vec<f32> {
    let mut grad = vec![0.0_f32; width * height * 3];
    // Mirror the guards of the forward term: inside them it is a constant.
    if width < 16 || height < 16 {
        return grad;
    }
    if pred.len() < width * height * 3 || target.len() < width * height * 3 {
        return grad;
    }

    let kernel = gaussian_kernel_1d(MS_SSIM_KERNEL_TAPS, MS_SSIM_KERNEL_SIGMA);

    // Pyramid dimensions, exactly as the forward term derives them.
    let mut dims: Vec<(usize, usize)> = Vec::with_capacity(MS_SSIM_MAX_SCALES);
    let mut w = width;
    let mut h = height;
    for _ in 0..MS_SSIM_MAX_SCALES {
        if w < MS_SSIM_MIN_DIM || h < MS_SSIM_MIN_DIM {
            break;
        }
        dims.push((w, h));
        w /= 2;
        h /= 2;
    }
    let num_scales = dims.len();
    if num_scales == 0 {
        return grad;
    }
    let last = num_scales - 1;

    // Build the pyramid.
    let mut preds: Vec<Vec<f32>> = Vec::with_capacity(num_scales);
    let mut tgts: Vec<Vec<f32>> = Vec::with_capacity(num_scales);
    preds.push(pred[..width * height * 3].to_vec());
    tgts.push(target[..width * height * 3].to_vec());
    for idx in 1..num_scales {
        let (pw, ph) = dims[idx - 1];
        let down_pred = downsample_2x(&preds[idx - 1], pw, ph);
        let down_tgt = downsample_2x(&tgts[idx - 1], pw, ph);
        preds.push(down_pred);
        tgts.push(down_tgt);
    }

    // Per-scale scalar means, then the product the forward term reports.
    let mut l_means = Vec::with_capacity(num_scales);
    let mut cs_means = Vec::with_capacity(num_scales);
    for idx in 0..num_scales {
        let (sw, sh) = dims[idx];
        let (l, cs) = ssim_component_means(&preds[idx], &tgts[idx], sw, sh, &kernel);
        l_means.push(l);
        cs_means.push(cs);
    }

    let mut product = 1.0_f32;
    for idx in 0..num_scales {
        product *= cs_means[idx].max(0.0).powf(weights[idx]);
    }
    product *= l_means[last].max(0.0).powf(weights[last]);

    // `1 − clamp(P, 0, 1)`: on or outside the clamp the term is constant.
    if !product.is_finite() || product <= 0.0 || product >= 1.0 {
        return grad;
    }

    // Walk from the coarsest scale back up, folding in each scale's
    // contribution and applying the downsampling adjoint between scales.
    let (mut acc_w, mut acc_h) = dims[last];
    let mut acc = vec![0.0_f32; acc_w * acc_h * 3];
    for idx in (0..num_scales).rev() {
        let (sw, sh) = dims[idx];
        let n = sw * sh;

        let d_cs = if weights[idx] > 0.0 && cs_means[idx] > 0.0 {
            -product * weights[idx] / cs_means[idx]
        } else {
            0.0
        };
        let d_l = if idx == last && weights[last] > 0.0 && l_means[last] > 0.0 {
            -product * weights[last] / l_means[last]
        } else {
            0.0
        };

        if d_cs != 0.0 || d_l != 0.0 {
            // `ssim_components` averages over pixels *and* the three channels.
            let inv_count = 1.0 / (3 * n) as f32;
            let wl = vec![d_l * inv_count; n];
            let wcs = vec![d_cs * inv_count; n];
            let mut chan = vec![0.0_f32; n];
            for c in 0..3 {
                let x = extract_channel(&preds[idx], n, c);
                let y = extract_channel(&tgts[idx], n, c);
                let stats = LocalStats::compute(&x, &y, sw, sh, &kernel);
                chan.fill(0.0);
                stats.accumulate_gradient(&x, &y, sw, sh, &kernel, &wl, &wcs, &mut chan);
                for (p, &g) in chan.iter().enumerate() {
                    acc[p * 3 + c] += g;
                }
            }
        }

        if idx > 0 {
            let (fw, fh) = dims[idx - 1];
            acc = upsample_adjoint_2x(&acc, acc_w, acc_h, fw, fh);
            acc_w = fw;
            acc_h = fh;
        }
    }

    for (dst, src) in grad.iter_mut().zip(acc.iter()) {
        *dst += src;
    }

    grad
}

/// Mean SSIM luminance and contrast-structure over all pixels *and* channels —
/// the two scalars [`crate::loss::ms_ssim_loss`] combines per scale.
fn ssim_component_means(
    pred: &[f32],
    target: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
) -> (f32, f32) {
    let n = width * height;
    if n == 0 {
        return (0.0, 0.0);
    }

    let mut l_sum = 0.0_f32;
    let mut cs_sum = 0.0_f32;
    for c in 0..3 {
        let x = extract_channel(pred, n, c);
        let y = extract_channel(target, n, c);
        let stats = LocalStats::compute(&x, &y, width, height, kernel);
        for q in 0..n {
            l_sum += stats.luminance(q);
            cs_sum += stats.contrast_structure(q);
        }
    }

    let count = (3 * n) as f32;
    (l_sum / count, cs_sum / count)
}

/// 2× box downsample of an HWC RGB image — identical to the forward term's.
fn downsample_2x(image: &[f32], width: usize, height: usize) -> Vec<f32> {
    let new_w = width / 2;
    let new_h = height / 2;
    if new_w == 0 || new_h == 0 {
        return Vec::new();
    }

    let mut out = vec![0.0_f32; new_w * new_h * 3];
    for y in 0..new_h {
        for x in 0..new_w {
            for c in 0..3 {
                let mut sum = 0.0_f32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let idx = ((y * 2 + dy) * width + (x * 2 + dx)) * 3 + c;
                        if idx < image.len() {
                            sum += image[idx];
                        }
                    }
                }
                out[(y * new_w + x) * 3 + c] = sum / 4.0;
            }
        }
    }

    out
}

/// Adjoint of [`downsample_2x`]: spread each coarse gradient over the 2×2 block
/// it averaged, carrying the same `1/4` factor.
fn upsample_adjoint_2x(
    grad_coarse: &[f32],
    coarse_w: usize,
    coarse_h: usize,
    fine_w: usize,
    fine_h: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; fine_w * fine_h * 3];
    for y in 0..coarse_h {
        for x in 0..coarse_w {
            for c in 0..3 {
                let g = grad_coarse
                    .get((y * coarse_w + x) * 3 + c)
                    .copied()
                    .unwrap_or(0.0)
                    * 0.25;
                if g == 0.0 {
                    continue;
                }
                for dy in 0..2 {
                    for dx in 0..2 {
                        let fy = y * 2 + dy;
                        let fx = x * 2 + dx;
                        if fy < fine_h && fx < fine_w {
                            out[(fy * fine_w + fx) * 3 + c] += g;
                        }
                    }
                }
            }
        }
    }

    out
}

/// Convert an `oxigaf_flame::Camera` to the GPU-facing [`RenderCamera`].
fn camera_to_render_camera(cam: &Camera, _w: u32, _h: u32) -> RenderCamera {
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
    use crate::loss::{l1_loss, ms_ssim_loss, opacity_reg, position_reg, scale_reg, ssim_loss};
    use oxigaf_render::gaussian::GaussianAttributes;

    /// Deterministic, non-trivial image pair in `[0.1, 0.9]`.
    fn make_pair(width: usize, height: usize) -> (Vec<f32>, Vec<f32>) {
        let n = width * height * 3;
        let mut pred = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        for i in 0..n {
            let f = i as f32;
            pred.push(0.5 + 0.4 * (f * 0.137).sin());
            target.push(0.5 + 0.4 * (f * 0.091).cos());
        }
        (pred, target)
    }

    fn loss_cfg(w_l1: f32, w_ssim: f32, w_ms_ssim: f32) -> LossConfig {
        LossConfig {
            w_l1,
            w_ssim,
            w_ms_ssim,
            w_lpips: 0.0,
            w_position_reg: 0.0,
            w_scale_reg: 0.0,
            w_opacity_reg: 0.0,
            w_normal: 0.0,
            w_gradient_penalty: 0.0,
            gradient_penalty_threshold: 100.0,
        }
    }

    fn reg_cfg(w_position_reg: f32, w_scale_reg: f32, w_opacity_reg: f32) -> LossConfig {
        LossConfig {
            w_position_reg,
            w_scale_reg,
            w_opacity_reg,
            ..loss_cfg(0.0, 0.0, 0.0)
        }
    }

    /// `photometric_pixel_gradient` with the stock MS-SSIM scale weights.
    fn pixel_grad(c: &LossConfig, p: &[f32], t: &[f32], w: usize, h: usize, v: usize) -> Vec<f32> {
        photometric_pixel_gradient(c, p, t, w, h, v, &LossComputer::DEFAULT_MS_SSIM_WEIGHTS)
    }

    /// The scalar photometric objective `LossComputer::compute` reports for one
    /// view (its regularisation terms are constant w.r.t. the pixels).
    fn scalar_loss(cfg: &LossConfig, pred: &[f32], tgt: &[f32], w: usize, h: usize) -> f32 {
        let kernel = gaussian_kernel_1d(SSIM_KERNEL_TAPS, SSIM_KERNEL_SIGMA);
        cfg.w_l1 * l1_loss(pred, tgt)
            + cfg.w_ssim * ssim_loss(pred, tgt, w, h, &kernel)
            + cfg.w_ms_ssim * ms_ssim_loss(pred, tgt, w, h, &LossComputer::DEFAULT_MS_SSIM_WEIGHTS)
    }

    /// Central-difference derivative of [`scalar_loss`] w.r.t. `p[i]`.
    fn numeric_grad(c: &LossConfig, p: &[f32], t: &[f32], w: usize, h: usize, i: usize) -> f32 {
        let eps = 0.02_f32;
        let mut plus = p.to_vec();
        let mut minus = p.to_vec();
        plus[i] += eps;
        minus[i] -= eps;
        (scalar_loss(c, &plus, t, w, h) - scalar_loss(c, &minus, t, w, h)) / (2.0 * eps)
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

    // ---- photometric image-space gradient ---------------------------------

    #[test]
    fn l1_gradient_is_view_normalised_sign() {
        let (w, h) = (4, 4);
        let (pred, target) = make_pair(w, h);
        let cfg = loss_cfg(0.8, 0.0, 0.0);
        let grad = pixel_grad(&cfg, &pred, &target, w, h, 2);

        let k = 0.8 / 2.0 / pred.len() as f32;
        for i in 0..pred.len() {
            let expected = k * sign_or_zero(pred[i] - target[i]);
            assert!((grad[i] - expected).abs() < 1e-9, "L1 gradient at {i}");
        }
    }

    #[test]
    fn ssim_gradient_matches_finite_differences() {
        let (w, h) = (16, 16);
        let (pred, target) = make_pair(w, h);
        let cfg = loss_cfg(0.0, 1.0, 0.0);
        let grad = pixel_grad(&cfg, &pred, &target, w, h, 1);

        // Well-interior pixels only: replicate padding makes the window adjoint
        // approximate within `kernel_taps / 2` of the border.
        for &i in &[409_usize, 354, 317, 453, 255, 510] {
            let numeric = numeric_grad(&cfg, &pred, &target, w, h, i);
            assert!(
                relative_error(grad[i], numeric) < 0.1,
                "SSIM gradient at {i}: analytic {} vs numeric {numeric}",
                grad[i]
            );
        }
    }

    #[test]
    fn ms_ssim_gradient_matches_finite_differences() {
        let (w, h) = (64, 64);
        let (pred, target) = make_pair(w, h);
        let cfg = loss_cfg(0.0, 0.0, 1.0);
        let grad = pixel_grad(&cfg, &pred, &target, w, h, 1);

        for &i in &[6240_usize, 6241, 5484] {
            let numeric = numeric_grad(&cfg, &pred, &target, w, h, i);
            assert!(
                relative_error(grad[i], numeric) < 0.25,
                "MS-SSIM gradient at {i}: analytic {} vs numeric {numeric}",
                grad[i]
            );
        }

        // At the optimum (identical images) the clamped product pins the term.
        let flat = pixel_grad(&cfg, &pred, &pred, w, h, 1);
        assert!(flat.iter().all(|g| *g == 0.0));
    }

    #[test]
    fn configured_weights_reach_the_gradient() {
        // Regression: the image-space gradient used to be a hardcoded MSE that
        // ignored every configured loss weight.
        let (w, h) = (16, 16);
        let (pred, target) = make_pair(w, h);
        let g_l1 = pixel_grad(&loss_cfg(1.0, 0.0, 0.0), &pred, &target, w, h, 1);
        let g_ssim = pixel_grad(&loss_cfg(0.0, 1.0, 0.0), &pred, &target, w, h, 1);
        let g_double = pixel_grad(&loss_cfg(2.0, 0.0, 0.0), &pred, &target, w, h, 1);

        assert!(g_l1.iter().any(|g| g.abs() > 0.0));
        assert!(g_ssim.iter().any(|g| g.abs() > 0.0));
        // Two different objectives must not yield the same descent direction,
        // and doubling a weight must double that term's contribution.
        let zipped = g_l1.iter().zip(g_ssim.iter());
        assert!(zipped.map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max) > 1e-9);
        for (a, b) in g_double.iter().zip(g_l1.iter()) {
            assert!((a - 2.0 * b).abs() < 1e-9);
        }
    }

    #[test]
    fn downsample_and_adjoint_are_transposes() {
        // <D·x, y> == <x, Dᵀ·y> for the 2× box filter.
        let (w, h) = (4, 4);
        let x: Vec<f32> = (0..w * h * 3).map(|i| i as f32 * 0.01).collect();
        let y: Vec<f32> = (0..(w / 2) * (h / 2) * 3)
            .map(|i| 1.0 - i as f32 * 0.02)
            .collect();

        let dx = downsample_2x(&x, w, h);
        let dty = upsample_adjoint_2x(&y, w / 2, h / 2, w, h);
        let lhs: f32 = dx.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let rhs: f32 = x.iter().zip(dty.iter()).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-5, "lhs {lhs} rhs {rhs}");
    }

    // ---- regularisation gradients -----------------------------------------

    #[test]
    fn regularisation_gradients_match_finite_differences() {
        let cfg = reg_cfg(0.01, 0.02, 0.05);
        let model = tiny_model(3);
        let mut grads = Gradients::zeros(model.len(), 3);
        add_regularization_gradients(&cfg, &model, &mut grads);

        let eps = 1e-2_f32;
        let reg_loss = |m: &GaussianModel| {
            cfg.w_position_reg * position_reg(m)
                + cfg.w_scale_reg * scale_reg(m)
                + cfg.w_opacity_reg * opacity_reg(m)
        };

        for j in 0..3 {
            // Offsets — the whole point of the `Offset` parameter group.
            let mut plus = model.clone();
            let mut minus = model.clone();
            plus.local_offsets[1][j] += eps;
            minus.local_offsets[1][j] -= eps;
            let numeric = (reg_loss(&plus) - reg_loss(&minus)) / (2.0 * eps);
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
            let numeric = (reg_loss(&plus) - reg_loss(&minus)) / (2.0 * eps);
            assert!(
                relative_error(grads.scale[6 + j], numeric) < 0.01,
                "scale[{j}]: analytic {} vs numeric {numeric}",
                grads.scale[6 + j]
            );
        }

        // Opacity logits.
        let mut plus = model.clone();
        let mut minus = model.clone();
        plus.gaussians[0].opacity += eps;
        minus.gaussians[0].opacity -= eps;
        let numeric = (reg_loss(&plus) - reg_loss(&minus)) / (2.0 * eps);
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
}
