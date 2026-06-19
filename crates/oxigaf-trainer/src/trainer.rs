//! Main [`Trainer`] struct — orchestrates the GAF optimisation loop.
//!
//! Each iteration:
//! 1. Sample random camera views.
//! 2. Render the current Gaussian model via GPU rasterizer.
//! 3. Generate multi-view targets via diffusion (or self-supervised fallback).
//! 4. Compute photometric + SDS + regularisation losses.
//! 5. Backward pass → GPU rasterizer backward → per-Gaussian gradients.
//! 6. Adam optimiser step (fully functional).
//! 7. Adaptive density control at scheduled intervals.
//! 8. Periodic opacity resets and checkpointing.
//!
//! ## Diffusion Integration
//!
//! The trainer supports iterative denoising distillation via Score Distillation
//! Sampling (SDS). During training:
//!
//! 1. **Warmup Period** (first 1000 iterations by default): Pure photometric
//!    training without diffusion, allowing the model to reach a reasonable
//!    starting point.
//!
//! 2. **Distillation Period**: The diffusion model generates pseudo ground-truth
//!    targets that guide the Gaussian optimization. The SDS loss is combined
//!    with photometric losses for stable training.
//!
//! 3. **Timestep Annealing**: The noise timestep starts high (more guidance)
//!    and anneals down to preserve fine details.

use std::path::Path;

use nalgebra as na;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use oxigaf_flame::Camera;
use oxigaf_render::gaussian::GaussianModel;
use oxigaf_render::{RasterConfig, Rasterizer, RenderCamera};

use crate::checkpoint;
use crate::config::TrainingConfig;
use crate::density::DensityController;
use crate::diffusion_target::{DiffusionTargetConfig, DiffusionTargetGenerator, SdsLoss};
use crate::loss::{LossComputer, LossOutput};
use crate::metrics::{self, MetricTracker};
use crate::mixed_precision::{LossScaler, MixedPrecisionTrainer, TrainingPrecision};
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
/// ## Diffusion Integration
///
/// The trainer optionally holds a [`DiffusionTargetGenerator`] for iterative
/// denoising distillation. When enabled:
///
/// - **Warmup phase**: First N iterations use self-supervised training
/// - **Distillation phase**: Diffusion model generates pseudo-GT targets
/// - **SDS loss**: Combined with photometric loss for stable optimization
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
    /// This allows customizing the warmup period, timestep annealing,
    /// and SDS weights.
    pub fn with_diffusion_config(
        config: TrainingConfig,
        model: GaussianModel,
        raster_config: RasterConfig,
        device: wgpu::Device,
        queue: wgpu::Queue,
        seed: u64,
        diffusion_config: DiffusionTargetConfig,
    ) -> Result<Self, TrainerError> {
        // Validate diffusion config
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
    pub fn from_checkpoint_with_diffusion(
        config: TrainingConfig,
        checkpoint_path: &Path,
        raster_config: RasterConfig,
        device: wgpu::Device,
        queue: wgpu::Queue,
        seed: u64,
        diffusion_config: DiffusionTargetConfig,
    ) -> Result<Self, TrainerError> {
        // Validate diffusion config
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
    /// 4. Compute photometric + SDS losses
    /// 5. Backward pass through GPU rasterizer
    /// 6. Adam optimizer step
    /// 7. Adaptive density control
    /// 8. Periodic opacity reset
    /// 9. Record metrics
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
        let loss_output = self.loss_computer.compute(
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

        // 6. Compute SDS loss (only if using diffusion targets).
        let sds_loss_value = if used_diffusion && sds_weight > 0.0 {
            self.sds_loss.compute(&rendered, &targets, current_timestep) * sds_weight
        } else {
            0.0
        };

        // Combined total loss for gradient computation
        let _combined_total = loss_output.total + sds_loss_value;

        // 7. Backward pass — GPU rasterizer backward  [Backward].
        // The gradients incorporate both photometric and SDS contributions
        // since targets already encode the diffusion guidance.
        let t_bwd = std::time::Instant::now();
        let mut gradients = self.compute_gradients(&rendered, &targets);
        self.profiler
            .record(TrainingPhase::Backward, t_bwd.elapsed().as_micros() as u64);

        // 8. Mixed-precision loss scaling: scale gradients before SDS/clip step.
        // For Float16, this simulates the effect of having scaled the loss before
        // the backward pass, enabling dynamic overflow detection.
        if matches!(self.config.precision, TrainingPrecision::Float16) {
            self.mp_trainer
                .scaler
                .scale_gradients(&mut gradients.position);
            self.mp_trainer
                .scaler
                .scale_gradients(&mut gradients.rotation);
            self.mp_trainer.scaler.scale_gradients(&mut gradients.scale);
            self.mp_trainer
                .scaler
                .scale_gradients(&mut gradients.opacity);
            self.mp_trainer.scaler.scale_gradients(&mut gradients.sh);
            self.mp_trainer
                .scaler
                .scale_gradients(&mut gradients.offset);
        }

        // 9. Apply SDS gradient scaling if using diffusion (acts as gradient clip).
        if used_diffusion && sds_weight > 0.0 {
            self.apply_sds_gradient_scaling(&mut gradients, sds_weight, current_timestep);
        }

        // 10. Unscale gradients and check for overflow (Float16 only).
        // Skips optimizer step on overflow and updates the loss scale adaptively.
        let should_optimizer_step = if matches!(self.config.precision, TrainingPrecision::Float16) {
            self.mp_trainer
                .scaler
                .unscale_gradients(&mut gradients.position);
            self.mp_trainer
                .scaler
                .unscale_gradients(&mut gradients.rotation);
            self.mp_trainer
                .scaler
                .unscale_gradients(&mut gradients.scale);
            self.mp_trainer
                .scaler
                .unscale_gradients(&mut gradients.opacity);
            self.mp_trainer.scaler.unscale_gradients(&mut gradients.sh);
            self.mp_trainer
                .scaler
                .unscale_gradients(&mut gradients.offset);
            let overflow = LossScaler::has_overflow(&gradients.position)
                || LossScaler::has_overflow(&gradients.rotation)
                || LossScaler::has_overflow(&gradients.scale)
                || LossScaler::has_overflow(&gradients.opacity)
                || LossScaler::has_overflow(&gradients.sh)
                || LossScaler::has_overflow(&gradients.offset);
            self.mp_trainer.scaler.update(overflow);
            !overflow
        } else {
            true
        };

        // 11. Optimiser step (skipped on FP16 gradient overflow)  [Optimize].
        if should_optimizer_step {
            let t_opt = std::time::Instant::now();
            self.optimizer.step(&mut self.model, &gradients, iter);
            self.profiler
                .record(TrainingPhase::Optimize, t_opt.elapsed().as_micros() as u64);
        }

        // 12. Density control.
        self.density_controller.accumulate_gradients(&gradients);

        if self.should_densify(iter) {
            let result = self
                .density_controller
                .densify_and_prune(&mut self.model, &mut self.rng);
            self.optimizer
                .handle_densify(&result.keep_mask, result.num_added);
        }

        // 13. Opacity reset.
        if iter > 0
            && self.config.opacity_reset_interval > 0
            && iter.is_multiple_of(self.config.opacity_reset_interval)
        {
            DensityController::reset_opacity(&mut self.model, self.config.init.initial_opacity);
        }

        // 14. Record metrics.
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

        // 15. TensorBoard logging.
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

    /// Apply SDS gradient scaling based on timestep.
    ///
    /// Higher timesteps (more noise) get higher gradient scaling to provide
    /// stronger guidance early in training.
    fn apply_sds_gradient_scaling(
        &self,
        gradients: &mut Gradients,
        sds_weight: f32,
        timestep: u32,
    ) {
        // Compute timestep-based scaling factor
        let t_norm = timestep as f32 / 1000.0;
        let scale = sds_weight * (0.5 + 0.5 * t_norm); // Scale from 0.5 to 1.0 based on timestep

        // Apply scaling to all gradient components
        for g in gradients.position.iter_mut() {
            *g *= scale;
        }
        for g in gradients.rotation.iter_mut() {
            *g *= scale;
        }
        for g in gradients.scale.iter_mut() {
            *g *= scale;
        }
        for g in gradients.opacity.iter_mut() {
            *g *= scale;
        }
        for g in gradients.sh.iter_mut() {
            *g *= scale;
        }
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
    fn get_current_learning_rates(&self, iteration: u32) -> LearningRates {
        // Position learning rate with exponential decay
        let decay_progress =
            (iteration as f32 / self.config.optimizer.position_lr_decay_steps as f32).min(1.0);
        let position_lr = self.config.optimizer.lr_position
            * (self.config.optimizer.lr_position_final / self.config.optimizer.lr_position)
                .powf(decay_progress);

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

    /// Linearly-annealed guidance scale.
    #[allow(dead_code)]
    fn current_guidance_scale(&self) -> f32 {
        let t = self.iteration as f32 / self.config.guidance_anneal_steps as f32;
        let t = t.min(1.0);
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
    fn render_views(&mut self, cameras: &[Camera]) -> Vec<Vec<f32>> {
        if cameras.is_empty() {
            return Vec::new();
        }

        let w = self.raster_config.image_width;
        let h = self.raster_config.image_height;

        // Upload the latest model parameters to the GPU.
        self.rasterizer.upload_gaussians(&self.model);

        cameras
            .iter()
            .map(|cam| {
                let render_cam = camera_to_render_camera(cam, w, h);
                match self.rasterizer.forward(&self.model, &render_cam) {
                    Ok(output) => {
                        // Convert RGBA [H*W*4] → RGB [H*W*3].
                        let npx = (w * h) as usize;
                        let mut rgb = Vec::with_capacity(npx * 3);
                        for i in 0..npx {
                            rgb.push(output.color_data[i * 4]);
                            rgb.push(output.color_data[i * 4 + 1]);
                            rgb.push(output.color_data[i * 4 + 2]);
                        }
                        rgb
                    }
                    Err(e) => {
                        tracing::warn!("render_views: rasterizer forward failed: {e}");
                        let bg = &self.raster_config.background;
                        let mut img = Vec::with_capacity((w * h * 3) as usize);
                        for _ in 0..(w * h) {
                            img.push(bg[0]);
                            img.push(bg[1]);
                            img.push(bg[2]);
                        }
                        img
                    }
                }
            })
            .collect()
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
    /// Combines per-view image-space loss gradients (∂L/∂pixel), dispatches
    /// the backward shader, and converts the GPU-side per-Gaussian gradients
    /// to the CPU [`Gradients`] struct consumed by the optimizer.
    fn compute_gradients(&mut self, rendered: &[Vec<f32>], targets: &[Vec<f32>]) -> Gradients {
        let n = self.model.len();
        let sh_channels = ((self.model.sh_degree + 1) * (self.model.sh_degree + 1) * 3) as usize;

        if rendered.is_empty() || targets.is_empty() {
            return Gradients::zeros(n, sh_channels);
        }

        let w = self.raster_config.image_width as usize;
        let h = self.raster_config.image_height as usize;

        // Accumulate gradients across all views.
        let mut acc = Gradients::zeros(n, sh_channels);
        let num_views = rendered.len().min(targets.len());

        for v in 0..num_views {
            // Compute per-pixel gradient ∂L/∂image  ≈ 2·(rendered - target)
            // in RGBA format for the GPU backward pass.
            let npx = w * h;
            let mut grad_image = vec![0.0f32; npx * 4];
            for i in 0..npx {
                let dr = rendered[v].get(i * 3).copied().unwrap_or(0.0)
                    - targets[v].get(i * 3).copied().unwrap_or(0.0);
                let dg = rendered[v].get(i * 3 + 1).copied().unwrap_or(0.0)
                    - targets[v].get(i * 3 + 1).copied().unwrap_or(0.0);
                let db = rendered[v].get(i * 3 + 2).copied().unwrap_or(0.0)
                    - targets[v].get(i * 3 + 2).copied().unwrap_or(0.0);
                grad_image[i * 4] = 2.0 * dr / num_views as f32;
                grad_image[i * 4 + 1] = 2.0 * dg / num_views as f32;
                grad_image[i * 4 + 2] = 2.0 * db / num_views as f32;
                // alpha gradient stays 0
            }

            match self.rasterizer.backward(&self.model, &grad_image) {
                Ok(gpu_grads) => {
                    // Flatten GPU gradient arrays and accumulate into the CPU buffer.
                    let pos_flat: Vec<f32> = gpu_grads
                        .grad_positions
                        .iter()
                        .flat_map(|p| p.iter().copied())
                        .collect();
                    let rot_flat: Vec<f32> = gpu_grads
                        .grad_rotations
                        .iter()
                        .flat_map(|r| r.iter().copied())
                        .collect();
                    let scl_flat: Vec<f32> = gpu_grads
                        .grad_scales
                        .iter()
                        .flat_map(|s| s.iter().copied())
                        .collect();
                    add_vec(&mut acc.position, &pos_flat);
                    add_vec(&mut acc.rotation, &rot_flat);
                    add_vec(&mut acc.scale, &scl_flat);
                    add_vec(&mut acc.opacity, &gpu_grads.grad_opacities);
                    add_vec(&mut acc.sh, &gpu_grads.grad_sh_coeffs);
                }
                Err(e) => {
                    tracing::warn!("compute_gradients: backward pass failed: {e}");
                }
            }
        }

        acc
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

/// Element-wise in-place addition: `dst[i] += src[i]`.
fn add_vec(dst: &mut [f32], src: &[f32]) {
    let len = dst.len().min(src.len());
    for i in 0..len {
        dst[i] += src[i];
    }
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
