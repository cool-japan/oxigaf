//! Diffusion target generation for iterative denoising distillation.
//!
//! This module provides the core infrastructure for generating pseudo ground-truth
//! images from diffusion models during Gaussian avatar training. The main approach
//! is Score Distillation Sampling (SDS), which uses a pre-trained diffusion model
//! to guide the 3D Gaussian optimization.
//!
//! Distillation happens in **pixel space**: the pipeline's decoded multi-view
//! output is the pseudo ground truth, and the loss is the timestep-weighted MSE
//! against the current render.  The `‖ε̂ − ε‖²` latent-space form is not used —
//! see [`SdsLoss`] for why the two agree up to the denoiser Jacobian.
//!
//! Key components:
//! - [`DiffusionTargetGenerator`] — orchestrates pseudo-GT generation
//! - [`SdsLoss`] — Score Distillation Sampling loss computation
//! - [`ViewConsistencyLoss`] — ensures consistency across generated views

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use nalgebra as na;

use oxigaf_diffusion::{DiffusionConfig, DiffusionError, MultiViewDiffusionPipeline, Vae};
use oxigaf_flame::Camera;

use crate::TrainerError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Length of the DDPM **training** noise schedule.
///
/// The diffusion pipeline builds its sampler as `DdimScheduler::new(1000, …)`,
/// so every `alpha_cumprod` lookup in this module has to use the same horizon
/// or the SDS weighting would disagree with the model it distils from.  This
/// is *not* [`DiffusionTargetConfig::timestep_start`], which only says where
/// the training-time timestep annealing begins.
const DDPM_TRAIN_TIMESTEPS: u32 = 1000;

/// Input resolution of the CLIP ViT image encoder used for IP-Adapter tokens.
const CLIP_INPUT_SIZE: usize = 224;

/// Per-channel mean of the OpenAI CLIP image normalisation.
const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];

/// Per-channel standard deviation of the OpenAI CLIP image normalisation.
const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_1];

// ---------------------------------------------------------------------------
// DiffusionTargetConfig
// ---------------------------------------------------------------------------

/// Configuration for the diffusion target generator.
#[derive(Debug, Clone)]
pub struct DiffusionTargetConfig {
    /// Number of inference steps for diffusion denoising.
    pub num_inference_steps: usize,
    /// Classifier-free guidance scale (legacy field; kept for compatibility).
    ///
    /// When `guidance_scale_start` and `guidance_scale_end` are used, this
    /// field represents the initial/static value.  Prefer the annealing
    /// fields for new code.
    pub guidance_scale: f32,
    /// Guidance scale at the **start** of training (before annealing).
    pub guidance_scale_start: f32,
    /// Guidance scale at the **end** of annealing (held constant afterwards).
    pub guidance_scale_end: f32,
    /// Number of training steps over which guidance scale linearly decays
    /// from `guidance_scale_start` to `guidance_scale_end`.
    pub guidance_anneal_steps: u32,
    /// Weight for view consistency loss.
    pub view_consistency_weight: f32,
    /// Number of warmup iterations without diffusion.
    pub warmup_iterations: u32,
    /// Initial timestep for noise (annealed down during training).
    pub timestep_start: u32,
    /// Final timestep for noise.
    pub timestep_end: u32,
    /// Annealing steps for timestep.
    pub timestep_anneal_steps: u32,
    /// Weight for SDS loss vs photometric loss.
    pub sds_weight: f32,
    /// Enable view warping for consistency.
    pub enable_view_warping: bool,
}

impl Default for DiffusionTargetConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 50,
            guidance_scale: 3.0,
            guidance_scale_start: 7.5,
            guidance_scale_end: 3.0,
            guidance_anneal_steps: 10_000,
            view_consistency_weight: 0.1,
            warmup_iterations: 1000,
            timestep_start: 1000,
            timestep_end: 50,
            timestep_anneal_steps: 10_000,
            sds_weight: 0.5,
            enable_view_warping: true,
        }
    }
}

impl DiffusionTargetConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), TrainerError> {
        if self.num_inference_steps == 0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "num_inference_steps".into(),
                value: "0".into(),
                expected: "> 0".into(),
            });
        }
        if !self.guidance_scale.is_finite() || self.guidance_scale <= 0.0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "guidance_scale".into(),
                value: format!("{}", self.guidance_scale),
                expected: "> 0 and finite".into(),
            });
        }
        if !self.guidance_scale_start.is_finite() || self.guidance_scale_start <= 0.0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "guidance_scale_start".into(),
                value: format!("{}", self.guidance_scale_start),
                expected: "> 0 and finite".into(),
            });
        }
        if !self.guidance_scale_end.is_finite() || self.guidance_scale_end <= 0.0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "guidance_scale_end".into(),
                value: format!("{}", self.guidance_scale_end),
                expected: "> 0 and finite".into(),
            });
        }
        if self.timestep_start <= self.timestep_end {
            return Err(TrainerError::InvalidConfig(format!(
                "timestep_start ({}) must be > timestep_end ({})",
                self.timestep_start, self.timestep_end
            )));
        }
        Ok(())
    }

    /// Get the current timestep based on training iteration.
    pub fn current_timestep(&self, iteration: u32) -> u32 {
        if iteration < self.warmup_iterations {
            return self.timestep_start;
        }
        let adjusted_iter = iteration - self.warmup_iterations;
        let t = (adjusted_iter as f32) / (self.timestep_anneal_steps as f32).max(1.0);
        let t = t.min(1.0);

        let start = self.timestep_start as f32;
        let end = self.timestep_end as f32;
        ((1.0 - t) * start + t * end).round() as u32
    }
}

// ---------------------------------------------------------------------------
// DiffusionTargetGenerator
// ---------------------------------------------------------------------------

/// Generates pseudo ground-truth images using diffusion models.
///
/// During training, this generator:
/// 1. Takes the current rendered views from the Gaussian model
/// 2. Uses the first view as the CLIP identity reference and the caller's
///    per-view normal maps as geometric conditioning
/// 3. Runs the multi-view diffusion pipeline for
///    [`DiffusionTargetConfig::num_inference_steps`] DDIM steps at the guidance
///    scale returned by [`DiffusionTargetGenerator::annealed_guidance_scale`]
/// 4. Returns the generated views as pseudo-GT for loss computation
///
/// # Not (yet) an img2img loop
///
/// The denoising run still starts from seeded Gaussian noise rather than from
/// the *noised latents of the current render*: `MultiViewDiffusionPipeline`
/// exposes no entry point that accepts initial latents, so an SDEdit-style
/// `z_t = √ᾱ_t·z₀ + √(1−ᾱ_t)·ε` start is not reachable from here.  The annealed
/// timestep from [`DiffusionTargetConfig::current_timestep`] therefore weights
/// the distillation loss ([`DiffusionTargetGenerator::compute_sds_gradient`],
/// [`SdsLoss`]) but does **not** set a denoising start point.  Identity and
/// geometry conditioning are what tie the generated targets to the current
/// model state.
pub struct DiffusionTargetGenerator {
    /// Optional diffusion pipeline (loaded lazily).
    pipeline: Option<MultiViewDiffusionPipeline>,
    /// Diffusion configuration.
    diff_config: DiffusionConfig,
    /// Target generation configuration.
    target_config: DiffusionTargetConfig,
    /// Candle device for tensor operations.
    device: Device,
    /// Whether the pipeline is fully loaded.
    is_loaded: bool,
    /// Directory the pipeline weights were loaded from.
    ///
    /// Kept so the normal-map encoder can be built from the same VAE weights.
    weights_dir: Option<PathBuf>,
    /// VAE used to encode normal-map conditioning, built on first use.
    ///
    /// This is a *second* copy of the pipeline's VAE: `MultiViewDiffusionPipeline`
    /// keeps its own encoder private and exposes decoding only.  It is built
    /// lazily so runs that never supply normal maps never pay for it.
    normal_encoder: Option<Vae>,
    /// Whether the "no normal maps supplied" warning has already been emitted.
    warned_missing_normals: bool,
}

impl DiffusionTargetGenerator {
    /// Create a new generator with default CPU device.
    pub fn new(target_config: DiffusionTargetConfig) -> Self {
        Self::with_device(target_config, Device::Cpu)
    }

    /// Create a generator with a specific device.
    pub fn with_device(target_config: DiffusionTargetConfig, device: Device) -> Self {
        Self {
            pipeline: None,
            diff_config: DiffusionConfig::default(),
            target_config,
            device,
            is_loaded: false,
            weights_dir: None,
            normal_encoder: None,
            warned_missing_normals: false,
        }
    }

    /// Load the diffusion pipeline from weights directory.
    pub fn load_pipeline(&mut self, weights_dir: &Path) -> Result<(), TrainerError> {
        let pipeline =
            MultiViewDiffusionPipeline::load(self.diff_config.clone(), weights_dir, &self.device)?;
        self.pipeline = Some(pipeline);
        self.weights_dir = Some(weights_dir.to_path_buf());
        // A previously built encoder belongs to the old weights directory.
        self.normal_encoder = None;
        self.is_loaded = true;
        tracing::info!("Diffusion pipeline loaded from {:?}", weights_dir);
        Ok(())
    }

    /// The target-generation configuration this generator was built with.
    pub fn target_config(&self) -> &DiffusionTargetConfig {
        &self.target_config
    }

    /// The view-consistency loss described by this generator's configuration.
    ///
    /// Wires [`DiffusionTargetConfig::view_consistency_weight`] and
    /// [`DiffusionTargetConfig::enable_view_warping`] into a ready-to-use
    /// [`ViewConsistencyLoss`] so both knobs actually reach the loss the caller
    /// aggregates.
    pub fn view_consistency_loss(&self) -> ViewConsistencyLoss {
        ViewConsistencyLoss::from_config(&self.target_config)
    }

    /// Check if the diffusion pipeline is loaded.
    pub fn is_loaded(&self) -> bool {
        self.is_loaded
    }

    /// Check if we're in the warmup period (no diffusion yet).
    pub fn is_warmup(&self, iteration: u32) -> bool {
        iteration < self.target_config.warmup_iterations
    }

    /// Get the SDS weight based on iteration.
    ///
    /// During warmup, returns 0.0 (no SDS).
    /// After warmup, ramps up to the configured weight.
    pub fn sds_weight(&self, iteration: u32) -> f32 {
        if iteration < self.target_config.warmup_iterations {
            return 0.0;
        }

        // Ramp up SDS weight over 500 iterations after warmup
        let ramp_steps = 500.0_f32;
        let adjusted = (iteration - self.target_config.warmup_iterations) as f32;
        let factor = (adjusted / ramp_steps).min(1.0);

        self.target_config.sds_weight * factor
    }

    /// Compute the linearly-annealed guidance scale at a given training step.
    ///
    /// The scale decays linearly from `guidance_scale_start` to
    /// `guidance_scale_end` over `guidance_anneal_steps` steps.  After that
    /// point it remains clamped at `guidance_scale_end`.
    ///
    /// # Example
    /// ```text
    /// // At step 0, returns guidance_scale_start (e.g. 7.5).
    /// // At step guidance_anneal_steps, returns guidance_scale_end (e.g. 3.0).
    /// // Beyond that, stays at guidance_scale_end.
    /// ```
    pub fn annealed_guidance_scale(&self, step: u32) -> f32 {
        let anneal_steps = self.target_config.guidance_anneal_steps;
        // Clamp progress to [0, 1].
        let t = if anneal_steps == 0 {
            1.0_f32
        } else {
            (step as f32 / anneal_steps as f32).min(1.0)
        };
        let start = self.target_config.guidance_scale_start;
        let end = self.target_config.guidance_scale_end;
        start + t * (end - start)
    }

    /// Generate multi-view pseudo ground-truth targets **without** geometric
    /// conditioning.
    ///
    /// Convenience wrapper over [`Self::generate_targets_with_normals`] that
    /// passes no normal maps.  The pipeline's geometry channels are then
    /// zero-filled and a warning is emitted once per generator, because
    /// unconditioned targets are not tied to the current Gaussian geometry.
    /// Prefer the normals-taking variant wherever a FLAME mesh is available.
    pub fn generate_targets(
        &mut self,
        rendered: &[Vec<f32>],
        cameras: &[Camera],
        iteration: u32,
        width: u32,
        height: u32,
    ) -> Result<Vec<Vec<f32>>, TrainerError> {
        self.generate_targets_with_normals(rendered, cameras, None, iteration, width, height)
    }

    /// Generate multi-view pseudo ground-truth targets.
    ///
    /// If the pipeline is not loaded or we're in warmup, returns the rendered
    /// images as-is (self-supervised mode).
    ///
    /// Otherwise:
    /// 1. Builds the CLIP identity reference from the first rendered view
    ///    (bilinearly resampled to 224×224 and CLIP-normalised)
    /// 2. Encodes `normal_maps` through the VAE into the geometric conditioning
    ///    latents the U-Net concatenates onto its input
    /// 3. Runs [`DiffusionTargetConfig::num_inference_steps`] DDIM steps at the
    ///    guidance scale from [`Self::annealed_guidance_scale`]
    /// 4. Returns the decoded views as pseudo-GT
    ///
    /// `normal_maps` are per-view HWC RGB buffers in `[0, 1]` at `width` ×
    /// `height` (the encoding layer of [`oxigaf_flame::NormalMapRenderer`]);
    /// they are resampled to the pipeline's image size and tiled or truncated to
    /// exactly `num_views` entries, since the pipeline always denoises
    /// `DiffusionConfig::num_views` latents.
    ///
    /// The run starts from seeded noise, not from the noised latents of the
    /// current render — see the note on [`DiffusionTargetGenerator`] itself.
    pub fn generate_targets_with_normals(
        &mut self,
        rendered: &[Vec<f32>],
        cameras: &[Camera],
        normal_maps: Option<&[Vec<f32>]>,
        iteration: u32,
        width: u32,
        height: u32,
    ) -> Result<Vec<Vec<f32>>, TrainerError> {
        // During warmup or if pipeline not loaded, return rendered as targets
        if self.is_warmup(iteration) || !self.is_loaded {
            tracing::trace!(
                "generate_targets: iteration {} (warmup={}, loaded={}), returning rendered",
                iteration,
                self.is_warmup(iteration),
                self.is_loaded
            );
            return Ok(rendered.to_vec());
        }

        if cameras.is_empty() || rendered.is_empty() {
            return Ok(Vec::new());
        }

        // Read every scalar off `&self` before the pipeline is borrowed mutably.
        let timestep = self.target_config.current_timestep(iteration);
        let steps = self.target_config.num_inference_steps.max(1);
        let guidance = clamp_guidance_scale(self.annealed_guidance_scale(iteration));

        // Convert rendered images to tensor format
        // Each image is [H*W*3] HWC format, convert to [V, 3, H, W] NCHW in [0, 1]
        let rendered_tensor = images_to_tensor(rendered, width, height, &self.device)?;

        // Generate reference image for CLIP (use the first view)
        let ref_image = prepare_reference_image(&rendered_tensor)?;

        // Create camera pose tensor, matched to the pipeline's view count
        let camera_poses = cameras_to_tensor(cameras, self.diff_config.num_views, &self.device)?;

        // Encode the geometric conditioning (needs `&mut self` for the lazy VAE)
        let normal_latents = self.encode_normal_latents(normal_maps, width, height)?;

        let pipeline = self
            .pipeline
            .as_mut()
            .ok_or(TrainerError::DiffusionNotLoaded)?;

        // Thread the annealed guidance scale and the configured step count into
        // the run; `generate` would silently use the pipeline's own defaults.
        pipeline.set_guidance_scale(f64::from(guidance));
        let mut session = pipeline.begin_session_with_steps(
            &ref_image,
            &normal_latents,
            &camera_poses,
            u64::from(iteration),
            steps,
        )?;
        while pipeline.step_session(&mut session)? {}
        let output = pipeline.finish_session(&session)?;

        // Convert output back to Vec<Vec<f32>> in HWC format
        let mut targets = Vec::with_capacity(output.images.len());
        for img_tensor in &output.images {
            let hwc = tensor_to_hwc_image(img_tensor, output.width, output.height)?;
            targets.push(hwc);
        }

        tracing::trace!(
            "generate_targets: iteration {}, timestep {}, steps {}, guidance {:.3}, generated {} views",
            iteration,
            timestep,
            steps,
            guidance,
            targets.len()
        );

        Ok(targets)
    }

    /// Encode per-view normal maps into the pipeline's conditioning latents.
    ///
    /// Returns a `(num_views, latent_channels, latent_size, latent_size)`
    /// tensor.  When no normal maps are supplied the geometry channels are
    /// zero-filled and a one-shot warning is logged: the pipeline concatenates
    /// these latents onto its noise latents unconditionally, so there is no
    /// "absent" encoding to fall back to.
    fn encode_normal_latents(
        &mut self,
        normal_maps: Option<&[Vec<f32>]>,
        width: u32,
        height: u32,
    ) -> Result<Tensor, TrainerError> {
        let num_views = self.diff_config.num_views;
        let latent_channels = self.diff_config.latent_channels;
        let latent_size = self.diff_config.latent_size;
        let image_size = self.diff_config.image_size;

        let maps = match normal_maps {
            Some(maps) if !maps.is_empty() => maps,
            _ => {
                if !self.warned_missing_normals {
                    self.warned_missing_normals = true;
                    tracing::warn!(
                        "No normal maps supplied to the diffusion target generator: the \
                         geometry conditioning channels are zero-filled, so the generated \
                         pseudo-GT views are not constrained by the current geometry. Call \
                         `generate_targets_with_normals` with rendered FLAME normal maps."
                    );
                }
                return Tensor::zeros(
                    (num_views, latent_channels, latent_size, latent_size),
                    DType::F32,
                    &self.device,
                )
                .map_err(|e| {
                    TrainerError::from(DiffusionError::Inference(format!("normal latents: {e}")))
                });
            }
        };

        let pixels = normal_maps_to_tensor(
            maps,
            num_views,
            width as usize,
            height as usize,
            image_size,
            &self.device,
        )?;

        let latents = {
            let vae = self.normal_encoder()?;
            vae.encode(&pixels)
                .map_err(|e| DiffusionError::Inference(format!("normal VAE encode: {e}")))?
        };

        // The latents are concatenated channel-wise onto the noise latents, so a
        // spatial mismatch would only surface as an opaque `Tensor::cat` error.
        let dims = latents
            .dims4()
            .map_err(|e| DiffusionError::Inference(format!("normal latents dims: {e}")))?;
        if dims != (num_views, latent_channels, latent_size, latent_size) {
            return Err(TrainerError::InvalidConfig(format!(
                "normal latents have shape {dims:?}, expected \
                 ({num_views}, {latent_channels}, {latent_size}, {latent_size})"
            )));
        }

        Ok(latents)
    }

    /// The VAE used to encode normal maps, built from the pipeline weights on
    /// first use.
    fn normal_encoder(&mut self) -> Result<&Vae, TrainerError> {
        if self.normal_encoder.is_none() {
            let path = self
                .weights_dir
                .as_ref()
                .ok_or(TrainerError::DiffusionNotLoaded)?
                .join("vae/diffusion_pytorch_model.safetensors");

            // SAFETY: `from_mmaped_safetensors` inherits the `memmap2` contract
            // — the mapping is undefined behaviour if the backing file is
            // modified by *anything* while it stays mapped.  These are
            // read-only pipeline weights that the training run never writes,
            // the same assumption `MultiViewDiffusionPipeline` and
            // `lpips::VggFeatureExtractor::from_safetensors` already make.
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[&path], DType::F32, &self.device).map_err(
                    |e| {
                        TrainerError::Init(format!(
                            "normal-map VAE weights ({}): {e}",
                            path.display()
                        ))
                    },
                )?
            };
            let vae = Vae::new(
                vb,
                self.diff_config.latent_channels,
                self.diff_config.vae_scale_factor,
            )
            .map_err(|e| TrainerError::Init(format!("normal-map VAE build: {e}")))?;
            self.normal_encoder = Some(vae);
        }

        self.normal_encoder
            .as_ref()
            .ok_or(TrainerError::DiffusionNotLoaded)
    }

    /// Compute the per-pixel gradient of the pixel-space distillation loss.
    ///
    /// This generator distils in **pixel space**: `target` is the pseudo-GT
    /// produced by [`Self::generate_targets_with_normals`], not an `epsilon`
    /// prediction.  The gradient of `w(t) · sds_weight · ‖rendered − target‖²`
    /// with respect to the render is therefore
    ///
    /// ```text
    /// g = w(t) · sds_weight(iteration) · (rendered − target)
    /// ```
    ///
    /// with the constant factor 2 folded into the weights.  `w(t)` is the
    /// variance-preserving weighting [`sds_timestep_weight`] at the timestep
    /// [`DiffusionTargetConfig::current_timestep`] anneals to.
    ///
    /// - `rendered`: flattened HWC pixels of the current render.
    /// - `target`: flattened HWC pixels of the pseudo ground truth.
    /// - `iteration`: current training iteration.
    ///
    /// The returned vector is as long as the shorter of the two inputs.
    pub fn compute_sds_gradient(
        &self,
        rendered: &[f32],
        target: &[f32],
        iteration: u32,
    ) -> Vec<f32> {
        let timestep = self.target_config.current_timestep(iteration);
        let weight = sds_timestep_weight(timestep, DDPM_TRAIN_TIMESTEPS);
        let sds_w = self.sds_weight(iteration);

        rendered
            .iter()
            .zip(target.iter())
            .map(|(r, t)| weight * sds_w * (r - t))
            .collect()
    }
}

/// Clamp a guidance scale into the range the diffusion pipeline accepts.
///
/// `MultiViewDiffusionPipeline::begin_session_with_steps` rejects any scale
/// below `1.0`, while [`DiffusionTargetConfig::validate`] only demands a finite
/// positive value.  Annealing towards an endpoint below `1.0` must degrade to
/// "no guidance", not turn a valid configuration into a hard failure mid-run.
fn clamp_guidance_scale(scale: f32) -> f32 {
    if scale.is_finite() {
        scale.max(1.0)
    } else {
        1.0
    }
}

// ---------------------------------------------------------------------------
// SDS Loss
// ---------------------------------------------------------------------------

/// Score Distillation Sampling loss, evaluated in **pixel space**.
///
/// The canonical SDS objective compares the diffusion model's noise prediction
/// against the noise that was added, `w(t)·‖ε̂ − ε‖²`.  This implementation
/// distils one step further downstream: the pipeline's *decoded* output is used
/// as pseudo ground truth and the loss is the timestep-weighted mean squared
/// error between the render and that pseudo-GT,
///
/// ```text
/// L = w(t) · mean_pixels((rendered − target)²)
/// ```
///
/// where `w(t)` is the [`SdsWeighting`] evaluated at the current timestep.  The
/// two forms point the optimisation in the same direction up to the denoiser
/// Jacobian, which SDS drops by construction; the practical difference is that
/// this variant needs no access to the model's internal `epsilon` prediction.
#[derive(Debug, Clone)]
pub struct SdsLoss {
    /// Weighting function type.
    pub weighting: SdsWeighting,
    /// Length of the DDPM training schedule used for normalisation.
    ///
    /// Values below 2 are clamped: the beta schedule interpolates over
    /// `max_timestep - 1` steps and would otherwise divide by zero.  Use
    /// [`SdsLoss::new`] to reject such values up front instead.
    pub max_timestep: u32,
}

/// SDS weighting function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdsWeighting {
    /// Uniform weighting across all timesteps.
    Uniform,
    /// Linear decrease: w(t) = t / T
    Linear,
    /// Quadratic decrease: w(t) = (t / T)^2
    Quadratic,
    /// Sigma-based: w(t) = sigma(t)^2
    SigmaBased,
}

impl Default for SdsLoss {
    fn default() -> Self {
        Self {
            weighting: SdsWeighting::SigmaBased,
            max_timestep: DDPM_TRAIN_TIMESTEPS,
        }
    }
}

impl SdsLoss {
    /// Create a validated SDS loss.
    ///
    /// # Errors
    ///
    /// Returns [`TrainerError::ParameterOutOfRange`] when `max_timestep < 2`.
    /// The DDPM beta schedule interpolates over `max_timestep - 1` steps, so a
    /// shorter horizon divides by zero and yields a non-finite weighting.
    pub fn new(weighting: SdsWeighting, max_timestep: u32) -> Result<Self, TrainerError> {
        if max_timestep < 2 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "max_timestep".into(),
                value: max_timestep.to_string(),
                expected: ">= 2".into(),
            });
        }
        Ok(Self {
            weighting,
            max_timestep,
        })
    }

    /// Compute the SDS loss for a batch of views.
    ///
    /// - `rendered`: rendered images, `[num_views][H*W*3]` in HWC order.
    /// - `targets`: pseudo ground-truth images from the diffusion pipeline,
    ///   same layout.
    /// - `timestep`: current diffusion timestep; it only selects the weighting
    ///   `w(t)`, the residual itself is the pixel difference.
    ///
    /// Views are paired by index.  Surplus views on either side are ignored, and
    /// so are empty views — a failed render would otherwise contribute `0/0` and
    /// poison the whole loss curve with `NaN`.  Returns `0.0` when nothing is
    /// comparable.
    pub fn compute(&self, rendered: &[Vec<f32>], targets: &[Vec<f32>], timestep: u32) -> f32 {
        if rendered.is_empty() || targets.is_empty() {
            return 0.0;
        }

        let weight = self.weight(timestep);
        let num_views = rendered.len().min(targets.len());

        let mut loss_sum = 0.0_f32;
        let mut counted = 0_u32;
        for v in 0..num_views {
            let pixels = rendered[v].len().min(targets[v].len());
            if pixels == 0 {
                continue;
            }
            let view_loss: f32 = rendered[v]
                .iter()
                .zip(targets[v].iter())
                .map(|(r, t)| {
                    let diff = r - t;
                    diff * diff
                })
                .sum();
            loss_sum += view_loss / pixels as f32;
            counted += 1;
        }

        if counted == 0 {
            return 0.0;
        }

        weight * loss_sum / counted as f32
    }

    /// Get the weighting factor for a timestep.
    fn weight(&self, timestep: u32) -> f32 {
        // Clamped so a hand-built `SdsLoss { max_timestep: 0 | 1, .. }` cannot
        // produce a non-finite weight; `SdsLoss::new` rejects those up front.
        let max_timestep = self.max_timestep.max(2);
        let t_norm = (timestep as f32) / (max_timestep as f32);

        match self.weighting {
            SdsWeighting::Uniform => 1.0,
            SdsWeighting::Linear => t_norm,
            SdsWeighting::Quadratic => t_norm * t_norm,
            SdsWeighting::SigmaBased => {
                // Approximate sigma^2 weighting based on DDPM schedule
                let alpha_t = ddpm_alpha_cumprod(timestep, max_timestep);
                1.0 - alpha_t
            }
        }
    }
}

// ---------------------------------------------------------------------------
// View Consistency Loss
// ---------------------------------------------------------------------------

/// View consistency loss ensures multi-view coherence.
///
/// This loss penalizes differences between reprojected views, ensuring that
/// the generated targets are geometrically consistent.
#[derive(Debug, Clone)]
pub struct ViewConsistencyLoss {
    /// Weight for the consistency loss.
    pub weight: f32,
    /// Whether depth maps are used to reproject views before comparing them.
    ///
    /// Mirrors [`DiffusionTargetConfig::enable_view_warping`]: when `false` the
    /// loss uses the appearance-only comparison even if depth maps are handed
    /// in.  See [`ViewConsistencyLoss::from_config`].
    pub enable_warping: bool,
}

impl Default for ViewConsistencyLoss {
    fn default() -> Self {
        Self {
            weight: 0.1,
            enable_warping: true,
        }
    }
}

impl ViewConsistencyLoss {
    /// Build the loss from a [`DiffusionTargetConfig`].
    ///
    /// This is what makes the config's `view_consistency_weight` and
    /// `enable_view_warping` knobs reach the loss.
    pub fn from_config(config: &DiffusionTargetConfig) -> Self {
        Self {
            weight: config.view_consistency_weight,
            enable_warping: config.enable_view_warping,
        }
    }

    /// Compute view consistency loss across multiple views.
    ///
    /// For each pair of views, we:
    /// 1. Warp one view to the other using depth (when a depth map is available
    ///    for that view and [`Self::enable_warping`] is set)
    /// 2. Compute the photometric difference over the pixels the warp actually
    ///    covered
    ///
    /// A short (or absent) `depth_maps` slice is not an error: the pairs it does
    /// not cover fall back to the appearance-only comparison.
    pub fn compute(
        &self,
        views: &[Vec<f32>],
        cameras: &[Camera],
        depth_maps: Option<&[Vec<f32>]>,
        width: usize,
        height: usize,
    ) -> f32 {
        if views.len() < 2 || cameras.len() < 2 {
            return 0.0;
        }

        // `enable_warping == false` degrades to the appearance path.
        let depth_maps = depth_maps.filter(|_| self.enable_warping);

        let num_views = views.len().min(cameras.len());
        let mut total_loss = 0.0_f32;
        let mut pair_count = 0_u32;

        // Compute pairwise consistency
        for i in 0..num_views {
            for j in (i + 1)..num_views {
                // `depth_maps` may be shorter than `num_views`; indexing it
                // directly used to panic out of this public API.
                let loss = match depth_maps.and_then(|depths| depths.get(i)) {
                    Some(src_depth) => self.warped_consistency(
                        &views[i],
                        &cameras[i],
                        &views[j],
                        &cameras[j],
                        src_depth,
                        width,
                        height,
                    ),
                    // Fall back to simple appearance consistency
                    None => self.appearance_consistency(&views[i], &views[j]),
                };
                total_loss += loss;
                pair_count += 1;
            }
        }

        if pair_count == 0 {
            0.0
        } else {
            self.weight * total_loss / pair_count as f32
        }
    }

    /// Simple appearance consistency (no depth).
    fn appearance_consistency(&self, view1: &[f32], view2: &[f32]) -> f32 {
        // Compute normalized cross-correlation or simple L1
        if view1.len() != view2.len() || view1.is_empty() {
            return 0.0;
        }

        // Simple L1 as a baseline
        let sum: f32 = view1
            .iter()
            .zip(view2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        sum / view1.len() as f32
    }

    /// Depth-based warping consistency.
    ///
    /// Only the target pixels a source sample actually landed on are compared.
    /// Pixels no source projects onto are warp *holes*, not black geometry;
    /// averaging them in would turn this term into a measure of hole coverage.
    #[allow(clippy::too_many_arguments)]
    fn warped_consistency(
        &self,
        src_view: &[f32],
        src_cam: &Camera,
        tgt_view: &[f32],
        tgt_cam: &Camera,
        src_depth: &[f32],
        width: usize,
        height: usize,
    ) -> f32 {
        if src_view.len() < width * height * 3 || tgt_view.len() < width * height * 3 {
            return 0.0;
        }

        // Warp source view to target view using depth
        let warped = warp_view(src_view, src_cam, tgt_cam, src_depth, width, height);

        // Compute loss only for valid (visible) pixels
        let mut loss_sum = 0.0_f32;
        let mut valid_count = 0_u32;

        for i in 0..(width * height) {
            if !warped.mask.get(i).copied().unwrap_or(false) {
                continue;
            }

            let idx = i * 3;
            let (Some(&wr), Some(&wg), Some(&wb)) = (
                warped.pixels.get(idx),
                warped.pixels.get(idx + 1),
                warped.pixels.get(idx + 2),
            ) else {
                continue;
            };

            let diff_r = (wr - tgt_view[idx]).abs();
            let diff_g = (wg - tgt_view[idx + 1]).abs();
            let diff_b = (wb - tgt_view[idx + 2]).abs();

            if diff_r.is_finite() && diff_g.is_finite() && diff_b.is_finite() {
                loss_sum += diff_r + diff_g + diff_b;
                valid_count += 1;
            }
        }

        if valid_count == 0 {
            0.0
        } else {
            loss_sum / (valid_count * 3) as f32
        }
    }
}

// ---------------------------------------------------------------------------
// Temporal Consistency (for video/animation)
// ---------------------------------------------------------------------------

/// Temporal consistency for animated avatars.
///
/// Ensures smooth transitions between frames.
#[derive(Debug, Clone)]
pub struct TemporalConsistency {
    /// Weight for temporal loss.
    pub weight: f32,
    /// Buffer size for temporal history.
    pub buffer_size: usize,
}

impl Default for TemporalConsistency {
    fn default() -> Self {
        Self {
            weight: 0.05,
            buffer_size: 3,
        }
    }
}

impl TemporalConsistency {
    /// Compute temporal consistency loss.
    ///
    /// - `current`: current frame
    /// - `previous`: previous frame(s)
    /// - `optical_flow`: estimated optical flow between frames (optional)
    pub fn compute(
        &self,
        current: &[f32],
        previous: &[&[f32]],
        _optical_flow: Option<&[f32]>,
    ) -> f32 {
        if previous.is_empty() || current.is_empty() {
            return 0.0;
        }

        // Simple temporal smoothness: penalize differences from previous frame
        let prev = previous.last().copied().unwrap_or(&[]);
        if prev.len() != current.len() {
            return 0.0;
        }

        let diff: f32 = current
            .iter()
            .zip(prev.iter())
            .map(|(c, p)| (c - p).powi(2))
            .sum();

        self.weight * diff / current.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Convert HWC images to a batched tensor [N, C, H, W].
fn images_to_tensor(
    images: &[Vec<f32>],
    width: u32,
    height: u32,
    device: &Device,
) -> Result<Tensor, DiffusionError> {
    let n = images.len();
    let h = height as usize;
    let w = width as usize;

    // Create NCHW tensor
    let mut data = vec![0.0_f32; n * 3 * h * w];

    for (idx, img) in images.iter().enumerate() {
        for y in 0..h {
            for x in 0..w {
                let hwc_idx = (y * w + x) * 3;
                let r = img.get(hwc_idx).copied().unwrap_or(0.0);
                let g = img.get(hwc_idx + 1).copied().unwrap_or(0.0);
                let b = img.get(hwc_idx + 2).copied().unwrap_or(0.0);

                let base = idx * 3 * h * w;
                let channel_stride = h * w;
                data[base + y * w + x] = r;
                data[base + channel_stride + y * w + x] = g;
                data[base + 2 * channel_stride + y * w + x] = b;
            }
        }
    }

    Tensor::from_vec(data, (n, 3, h, w), device)
        .map_err(|e| DiffusionError::Inference(format!("images_to_tensor: {e}")))
}

/// Convert a tensor [C, H, W] to HWC Vec<f32>.
fn tensor_to_hwc_image(
    tensor: &Tensor,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, DiffusionError> {
    let h = height as usize;
    let w = width as usize;

    // Flatten and convert to Vec<f32>
    let data: Vec<f32> = tensor
        .flatten_all()
        .and_then(|t| t.to_vec1())
        .map_err(|e| DiffusionError::Inference(format!("tensor_to_hwc: {e}")))?;

    if data.len() < 3 * h * w {
        return Err(DiffusionError::Inference(format!(
            "tensor_to_hwc: data length {} < expected {}",
            data.len(),
            3 * h * w
        )));
    }

    // CHW to HWC
    let mut hwc = vec![0.0_f32; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let hwc_idx = (y * w + x) * 3;
            let channel_stride = h * w;
            let pixel_offset = y * w + x;
            hwc[hwc_idx] = data.get(pixel_offset).copied().unwrap_or(0.0);
            hwc[hwc_idx + 1] = data
                .get(channel_stride + pixel_offset)
                .copied()
                .unwrap_or(0.0);
            hwc[hwc_idx + 2] = data
                .get(2 * channel_stride + pixel_offset)
                .copied()
                .unwrap_or(0.0);
        }
    }

    Ok(hwc)
}

/// Convert cameras to a batched pose tensor `[num_views, 12]` (flattened 4x3
/// extrinsics).
///
/// Cameras are tiled cyclically when fewer than `num_views` are supplied and
/// dropped when more are, for the same reason the normal latents are (see
/// [`normal_maps_to_tensor`]): the pipeline denoises exactly
/// `DiffusionConfig::num_views` latents and every per-view conditioning tensor
/// has to line up with that batch dimension.
fn cameras_to_tensor(
    cameras: &[Camera],
    num_views: usize,
    device: &Device,
) -> Result<Tensor, DiffusionError> {
    if cameras.is_empty() || num_views == 0 {
        return Err(DiffusionError::Inference(format!(
            "cameras_to_tensor: {} camera(s), {num_views} view(s)",
            cameras.len()
        )));
    }

    let mut data = vec![0.0_f32; num_views * 12];

    for i in 0..num_views {
        let cam = &cameras[i % cameras.len()];
        // Flatten rotation (3x3) and translation (3)
        // Row-major: r00, r01, r02, r10, r11, r12, r20, r21, r22, tx, ty, tz
        for r in 0..3 {
            for c in 0..3 {
                data[i * 12 + r * 3 + c] = cam.rotation[(r, c)];
            }
        }
        data[i * 12 + 9] = cam.translation.x;
        data[i * 12 + 10] = cam.translation.y;
        data[i * 12 + 11] = cam.translation.z;
    }

    Tensor::from_vec(data, (num_views, 12), device)
        .map_err(|e| DiffusionError::Inference(format!("cameras_to_tensor: {e}")))
}

/// Half-pixel-centred bilinear sampling weights along one axis.
///
/// Maps destination index `dst` of a `dst_len`-long axis onto the source axis
/// with `src = (dst + 0.5) · src_len / dst_len − 0.5`, which covers the *whole*
/// source extent for both up- and down-scaling.  Returns `(lo, hi, frac)` so
/// the resampled value is `v[lo] + (v[hi] − v[lo]) · frac`.
fn bilinear_axis(dst: usize, dst_len: usize, src_len: usize) -> (usize, usize, f32) {
    if src_len == 0 || dst_len == 0 {
        return (0, 0, 0.0);
    }
    let scale = src_len as f32 / dst_len as f32;
    let pos = ((dst as f32 + 0.5) * scale - 0.5).clamp(0.0, (src_len - 1) as f32);
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(src_len - 1);
    (lo, hi, pos - lo as f32)
}

/// Bilinearly resample an interleaved HWC RGB image.
fn resample_hwc(src: &[f32], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; dst_w * dst_h * 3];
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return out;
    }

    for y in 0..dst_h {
        let (y0, y1, wy) = bilinear_axis(y, dst_h, src_h);
        for x in 0..dst_w {
            let (x0, x1, wx) = bilinear_axis(x, dst_w, src_w);
            for c in 0..3 {
                let p00 = src.get((y0 * src_w + x0) * 3 + c).copied().unwrap_or(0.0);
                let p01 = src.get((y0 * src_w + x1) * 3 + c).copied().unwrap_or(0.0);
                let p10 = src.get((y1 * src_w + x0) * 3 + c).copied().unwrap_or(0.0);
                let p11 = src.get((y1 * src_w + x1) * 3 + c).copied().unwrap_or(0.0);
                let top = p00 + (p01 - p00) * wx;
                let bottom = p10 + (p11 - p10) * wx;
                if let Some(slot) = out.get_mut((y * dst_w + x) * 3 + c) {
                    *slot = top + (bottom - top) * wy;
                }
            }
        }
    }

    out
}

/// Prepare the CLIP reference image from the first rendered view.
///
/// `images` is the `(V, 3, H, W)` render batch in `[0, 1]`.  The first view is
/// bilinearly resampled to 224×224 — covering the *whole* frame; the previous
/// integer-ratio box filter cropped the right/bottom edges and, for inputs
/// smaller than twice the target, degenerated into a plain top-left crop — and
/// then normalised with CLIP's own channel mean/std, which is what the ViT
/// image encoder behind the IP-Adapter tokens was trained on.
fn prepare_reference_image(images: &Tensor) -> Result<Tensor, DiffusionError> {
    let first = images
        .narrow(0, 0, 1)
        .map_err(|e| DiffusionError::Inference(format!("narrow: {e}")))?;

    let (_b, c, h, w) = first
        .dims4()
        .map_err(|e| DiffusionError::Inference(format!("dims4: {e}")))?;
    if c == 0 || h == 0 || w == 0 {
        return Err(DiffusionError::Inference(format!(
            "reference image has an empty dimension: ({c}, {h}, {w})"
        )));
    }

    let data: Vec<f32> = first
        .flatten_all()
        .and_then(|t| t.to_vec1())
        .map_err(|e| DiffusionError::Inference(format!("flatten: {e}")))?;

    let size = CLIP_INPUT_SIZE;
    // The axis weights are shared by every channel, so compute them once.
    let rows: Vec<(usize, usize, f32)> = (0..size).map(|y| bilinear_axis(y, size, h)).collect();
    let cols: Vec<(usize, usize, f32)> = (0..size).map(|x| bilinear_axis(x, size, w)).collect();

    let mut resized = vec![0.0_f32; c * size * size];
    for ch in 0..c {
        let src_plane = ch * h * w;
        let dst_plane = ch * size * size;
        let mean = CLIP_MEAN.get(ch).copied().unwrap_or(0.0);
        let std_dev = CLIP_STD.get(ch).copied().unwrap_or(1.0);
        for (y, &(y0, y1, wy)) in rows.iter().enumerate() {
            for (x, &(x0, x1, wx)) in cols.iter().enumerate() {
                let p00 = data.get(src_plane + y0 * w + x0).copied().unwrap_or(0.0);
                let p01 = data.get(src_plane + y0 * w + x1).copied().unwrap_or(0.0);
                let p10 = data.get(src_plane + y1 * w + x0).copied().unwrap_or(0.0);
                let p11 = data.get(src_plane + y1 * w + x1).copied().unwrap_or(0.0);
                let top = p00 + (p01 - p00) * wx;
                let bottom = p10 + (p11 - p10) * wx;
                let value = top + (bottom - top) * wy;
                if let Some(slot) = resized.get_mut(dst_plane + y * size + x) {
                    *slot = (value - mean) / std_dev;
                }
            }
        }
    }

    Tensor::from_vec(resized, (1, c, size, size), first.device())
        .map_err(|e| DiffusionError::Inference(format!("reference image: {e}")))
}

/// Build the `(num_views, 3, size, size)` pixel tensor the VAE encoder expects
/// from per-view HWC normal maps in `[0, 1]`.
///
/// Each map is bilinearly resampled to `size × size` and rescaled to the
/// `[-1, 1]` range the encoder was trained on.  Views are tiled cyclically when
/// fewer maps than views are supplied and dropped when more are: the pipeline
/// always denoises exactly `num_views` latents and concatenates these onto them
/// channel-wise, so the batch dimensions have to agree.
fn normal_maps_to_tensor(
    maps: &[Vec<f32>],
    num_views: usize,
    src_width: usize,
    src_height: usize,
    size: usize,
    device: &Device,
) -> Result<Tensor, DiffusionError> {
    if maps.is_empty() || num_views == 0 || size == 0 || src_width == 0 || src_height == 0 {
        return Err(DiffusionError::Inference(format!(
            "normal maps: {} map(s), {num_views} view(s), source {src_width}x{src_height}, \
             target {size}x{size}",
            maps.len()
        )));
    }

    let plane = size * size;
    let mut data = vec![0.0_f32; num_views * 3 * plane];

    for view in 0..num_views {
        let map = &maps[view % maps.len()];
        let resized = resample_hwc(map, src_width, src_height, size, size);
        let base = view * 3 * plane;
        for pixel in 0..plane {
            for c in 0..3 {
                let value = resized.get(pixel * 3 + c).copied().unwrap_or(0.0);
                if let Some(slot) = data.get_mut(base + c * plane + pixel) {
                    // [0, 1] → [-1, 1]
                    *slot = value.mul_add(2.0, -1.0);
                }
            }
        }
    }

    Tensor::from_vec(data, (num_views, 3, size, size), device)
        .map_err(|e| DiffusionError::Inference(format!("normal_maps_to_tensor: {e}")))
}

/// A source view forward-warped into a target view.
struct WarpedView {
    /// Warped RGB pixels in HWC order; entries whose `mask` is `false` are unset.
    pixels: Vec<f32>,
    /// `true` where at least one source pixel projected onto the target pixel.
    mask: Vec<bool>,
}

/// Warp a source view to a target view using depth.
///
/// This is a forward scatter, so it leaves holes wherever no source pixel lands;
/// `WarpedView::mask` records which target pixels were actually written so the
/// caller can skip the rest.  Competing sources are resolved with a per-target
/// depth buffer — nearest wins — instead of last-write-wins.
fn warp_view(
    src_view: &[f32],
    src_cam: &Camera,
    tgt_cam: &Camera,
    src_depth: &[f32],
    width: usize,
    height: usize,
) -> WarpedView {
    let mut warped = WarpedView {
        pixels: vec![0.0_f32; width * height * 3],
        mask: vec![false; width * height],
    };
    let mut depth_buffer = vec![f32::INFINITY; width * height];

    // For each pixel in source view
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let depth = src_depth.get(idx).copied().unwrap_or(0.0);

            if depth <= 0.0 || !depth.is_finite() {
                continue;
            }

            // Unproject to 3D
            let px = (x as f32 - src_cam.cx) / src_cam.focal_x;
            let py = (y as f32 - src_cam.cy) / src_cam.focal_y;
            let point_cam = na::Vector3::new(px * depth, py * depth, depth);

            // Transform to world space
            let r_inv = src_cam.rotation.transpose();
            let point_world = r_inv * (point_cam - src_cam.translation);

            // Project to target camera
            let point_tgt_cam = tgt_cam.rotation * point_world + tgt_cam.translation;

            if point_tgt_cam.z <= 0.0 || !point_tgt_cam.z.is_finite() {
                continue;
            }

            let tx = (point_tgt_cam.x / point_tgt_cam.z) * tgt_cam.focal_x + tgt_cam.cx;
            let ty = (point_tgt_cam.y / point_tgt_cam.z) * tgt_cam.focal_y + tgt_cam.cy;
            if !tx.is_finite() || !ty.is_finite() {
                continue;
            }

            let tx_i = tx.round() as i32;
            let ty_i = ty.round() as i32;

            if tx_i >= 0 && tx_i < width as i32 && ty_i >= 0 && ty_i < height as i32 {
                let tgt_idx = (ty_i as usize) * width + (tx_i as usize);

                // Z-buffer: keep the nearest source sample for this target pixel.
                let Some(slot) = depth_buffer.get_mut(tgt_idx) else {
                    continue;
                };
                if point_tgt_cam.z >= *slot {
                    continue;
                }
                *slot = point_tgt_cam.z;

                let src_hwc = idx * 3;
                let tgt_hwc = tgt_idx * 3;
                if let Some(dst) = warped.pixels.get_mut(tgt_hwc..tgt_hwc + 3) {
                    dst[0] = src_view.get(src_hwc).copied().unwrap_or(0.0);
                    dst[1] = src_view.get(src_hwc + 1).copied().unwrap_or(0.0);
                    dst[2] = src_view.get(src_hwc + 2).copied().unwrap_or(0.0);
                }
                if let Some(valid) = warped.mask.get_mut(tgt_idx) {
                    *valid = true;
                }
            }
        }
    }

    warped
}

/// SDS timestep weighting factor.
///
/// Higher timesteps (more noise) get higher weights.  `max_timestep` is the
/// length of the DDPM *training* schedule — pass [`DDPM_TRAIN_TIMESTEPS`] unless
/// the caller deliberately distils from a different schedule.
fn sds_timestep_weight(timestep: u32, max_timestep: u32) -> f32 {
    let alpha = ddpm_alpha_cumprod(timestep, max_timestep);
    let sigma_sq = 1.0 - alpha;

    // w(t) = sigma(t)^2 for variance-preserving weighting
    sigma_sq.max(0.001)
}

/// Approximate DDPM `alpha_cumprod` for a given timestep.
///
/// `max_timestep` is the length of the training schedule.  It is clamped to at
/// least 2 because the beta interpolation divides by `max_timestep - 1`, which
/// would otherwise yield `inf`/`NaN` and propagate straight into the SDS
/// weighting.  `timestep` is likewise clamped to the last schedule entry, so a
/// caller passing `t == max_timestep` stops at `beta_end` instead of
/// extrapolating past it.
fn ddpm_alpha_cumprod(timestep: u32, max_timestep: u32) -> f32 {
    // Scaled linear beta schedule (SD 2.1 style)
    let beta_start = 0.00085_f32.sqrt();
    let beta_end = 0.012_f32.sqrt();

    let last = max_timestep.max(2) - 1;
    let denom = last as f32;

    let mut alpha_cumprod = 1.0_f32;
    for t in 0..=timestep.min(last) {
        let beta = beta_start + (beta_end - beta_start) * (t as f32) / denom;
        let beta = beta * beta;
        let alpha = 1.0 - beta;
        alpha_cumprod *= alpha;
    }

    alpha_cumprod
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diffusion_config_default() {
        let config = DiffusionTargetConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.warmup_iterations, 1000);
        assert_eq!(config.timestep_start, 1000);
        assert_eq!(config.timestep_end, 50);
    }

    #[test]
    fn test_timestep_annealing() {
        let config = DiffusionTargetConfig {
            warmup_iterations: 100,
            timestep_start: 1000,
            timestep_end: 100,
            timestep_anneal_steps: 1000,
            ..Default::default()
        };

        // During warmup, timestep should be max
        assert_eq!(config.current_timestep(0), 1000);
        assert_eq!(config.current_timestep(50), 1000);
        assert_eq!(config.current_timestep(99), 1000);

        // After warmup, should anneal
        assert_eq!(config.current_timestep(100), 1000); // Just after warmup
        let mid = config.current_timestep(600); // ~500 steps after warmup
        assert!(mid < 1000 && mid > 100, "mid timestep = {}", mid);

        // At end of annealing
        let end = config.current_timestep(1100);
        assert_eq!(end, 100);
    }

    #[test]
    fn test_sds_weight_ramp() {
        let gen = DiffusionTargetGenerator::new(DiffusionTargetConfig {
            warmup_iterations: 100,
            sds_weight: 1.0,
            ..Default::default()
        });

        // During warmup
        assert_eq!(gen.sds_weight(0), 0.0);
        assert_eq!(gen.sds_weight(50), 0.0);
        assert_eq!(gen.sds_weight(99), 0.0);

        // Exactly at warmup boundary, ramp starts at 0
        assert_eq!(gen.sds_weight(100), 0.0);

        // After warmup, ramps up (iteration 101 has factor = 1/500)
        assert!(gen.sds_weight(101) > 0.0);
        assert!(gen.sds_weight(200) > gen.sds_weight(101));

        // Full weight after 500 steps post-warmup
        assert!((gen.sds_weight(600) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sds_loss_identical() {
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
    fn test_sds_loss_different() {
        let loss = SdsLoss::default();
        let view1 = vec![0.0_f32; 100];
        let view2 = vec![1.0_f32; 100];

        let l = loss.compute(&[view1], &[view2], 500);
        assert!(l > 0.0, "different views should have positive loss");
    }

    #[test]
    fn test_ddpm_alpha_cumprod() {
        // At t=0, alpha_cumprod should be close to 1
        let alpha_0 = ddpm_alpha_cumprod(0, 1000);
        assert!(alpha_0 > 0.99, "alpha at t=0 = {}", alpha_0);

        // At t=999, alpha_cumprod should be small
        let alpha_999 = ddpm_alpha_cumprod(999, 1000);
        assert!(alpha_999 < 0.1, "alpha at t=999 = {}", alpha_999);

        // Should be monotonically decreasing
        assert!(ddpm_alpha_cumprod(100, 1000) > ddpm_alpha_cumprod(500, 1000));
    }

    #[test]
    fn test_view_consistency_empty() {
        let loss = ViewConsistencyLoss::default();
        let l = loss.compute(&[], &[], None, 64, 64);
        assert_eq!(l, 0.0);
    }

    #[test]
    fn test_temporal_consistency_empty() {
        let tc = TemporalConsistency::default();
        let l = tc.compute(&[], &[], None);
        assert_eq!(l, 0.0);
    }

    // -----------------------------------------------------------------------
    // Guidance scale annealing tests
    // -----------------------------------------------------------------------

    fn make_annealing_generator(start: f32, end: f32, steps: u32) -> DiffusionTargetGenerator {
        DiffusionTargetGenerator::new(DiffusionTargetConfig {
            guidance_scale_start: start,
            guidance_scale_end: end,
            guidance_anneal_steps: steps,
            ..Default::default()
        })
    }

    #[test]
    fn guidance_annealing_at_step_zero_returns_start() {
        let gen = make_annealing_generator(7.5, 3.0, 10_000);
        let scale = gen.annealed_guidance_scale(0);
        assert!(
            (scale - 7.5).abs() < 1e-5,
            "at step 0 expected 7.5, got {scale}"
        );
    }

    #[test]
    fn guidance_annealing_at_full_steps_returns_end() {
        let gen = make_annealing_generator(7.5, 3.0, 10_000);
        let scale = gen.annealed_guidance_scale(10_000);
        assert!(
            (scale - 3.0).abs() < 1e-5,
            "at full steps expected 3.0, got {scale}"
        );
    }

    #[test]
    fn guidance_annealing_midpoint_is_midway() {
        let gen = make_annealing_generator(7.5, 3.0, 10_000);
        let scale = gen.annealed_guidance_scale(5_000);
        let expected = (7.5 + 3.0) / 2.0; // 5.25
        assert!(
            (scale - expected).abs() < 1e-4,
            "at midpoint expected {expected}, got {scale}"
        );
    }

    #[test]
    fn guidance_annealing_clamps_beyond_steps() {
        let gen = make_annealing_generator(7.5, 3.0, 10_000);
        // Well past the anneal period.
        let scale = gen.annealed_guidance_scale(50_000);
        assert!(
            (scale - 3.0).abs() < 1e-5,
            "beyond anneal steps expected 3.0, got {scale}"
        );
    }

    #[test]
    fn guidance_annealing_exact_one_step_boundary() {
        // With anneal_steps = 1, step 1 should return end exactly.
        let gen = make_annealing_generator(10.0, 2.0, 1);
        let scale = gen.annealed_guidance_scale(1);
        assert!(
            (scale - 2.0).abs() < 1e-5,
            "at boundary (steps=1) expected 2.0, got {scale}"
        );
    }

    #[test]
    fn guidance_annealing_quarter_point() {
        let gen = make_annealing_generator(8.0, 0.0, 1_000);
        // At 25 %, scale should be 8.0 - 0.25 * 8.0 = 6.0.
        let scale = gen.annealed_guidance_scale(250);
        assert!(
            (scale - 6.0).abs() < 1e-4,
            "at 25 %% expected 6.0, got {scale}"
        );
    }

    #[test]
    fn guidance_annealing_zero_anneal_steps_clamps_to_end() {
        // When guidance_anneal_steps == 0 every step should return `end`.
        let gen = make_annealing_generator(7.5, 3.0, 0);
        let at_zero = gen.annealed_guidance_scale(0);
        let at_hundred = gen.annealed_guidance_scale(100);
        assert!(
            (at_zero - 3.0).abs() < 1e-5,
            "zero steps: step 0 expected 3.0, got {at_zero}"
        );
        assert!(
            (at_hundred - 3.0).abs() < 1e-5,
            "zero steps: step 100 expected 3.0, got {at_hundred}"
        );
    }

    // -----------------------------------------------------------------------
    // Guidance / step-count wiring
    // -----------------------------------------------------------------------

    #[test]
    fn guidance_scale_is_clamped_to_the_pipeline_minimum() {
        // The pipeline rejects scales below 1.0 while `validate` only demands a
        // positive value, so annealing below 1.0 must degrade, not fail.
        assert!((clamp_guidance_scale(7.5) - 7.5).abs() < 1e-6);
        assert!((clamp_guidance_scale(0.5) - 1.0).abs() < 1e-6);
        assert!((clamp_guidance_scale(f32::NAN) - 1.0).abs() < 1e-6);

        let gen = make_annealing_generator(2.0, 0.25, 100);
        let annealed = clamp_guidance_scale(gen.annealed_guidance_scale(100));
        assert!((annealed - 1.0).abs() < 1e-6, "annealed = {annealed}");
    }

    #[test]
    fn generator_exposes_configured_view_consistency_loss() {
        let gen = DiffusionTargetGenerator::new(DiffusionTargetConfig {
            view_consistency_weight: 0.42,
            enable_view_warping: false,
            ..Default::default()
        });
        let loss = gen.view_consistency_loss();
        assert!(
            (loss.weight - 0.42).abs() < 1e-6,
            "weight = {}",
            loss.weight
        );
        assert!(!loss.enable_warping);
        assert_eq!(gen.target_config().view_consistency_weight, 0.42);
    }

    // -----------------------------------------------------------------------
    // Conditioning tensors
    // -----------------------------------------------------------------------

    #[test]
    fn normal_maps_tile_to_the_pipeline_view_count() {
        let device = Device::Cpu;
        let red = vec![1.0_f32, 0.0, 0.0];
        let green = vec![0.0_f32, 1.0, 0.0];
        let maps = vec![red, green];

        let tensor = normal_maps_to_tensor(&maps, 4, 1, 1, 2, &device).expect("normal tensor");
        assert_eq!(tensor.dims4().expect("dims"), (4, 3, 2, 2));

        let data: Vec<f32> = tensor
            .flatten_all()
            .and_then(|t| t.to_vec1())
            .expect("tensor data");
        let stride = 3 * 2 * 2;
        // View 0 is the red map: channel 0 saturated, mapped [0,1] → [-1,1].
        assert!((data[0] - 1.0).abs() < 1e-5, "view 0 ch 0 = {}", data[0]);
        // View 1 is the green map.
        assert!((data[stride] + 1.0).abs() < 1e-5);
        assert!((data[stride + 4] - 1.0).abs() < 1e-5);
        // View 2 wraps back onto the red map.
        assert!((data[2 * stride] - 1.0).abs() < 1e-5);

        assert!(normal_maps_to_tensor(&[], 4, 1, 1, 2, &device).is_err());
    }

    #[test]
    fn camera_poses_match_the_pipeline_view_count() {
        let device = Device::Cpu;
        let camera = Camera::default_front(8, 8);
        let poses = cameras_to_tensor(&[camera], 4, &device).expect("pose tensor");
        assert_eq!(poses.dims2().expect("dims"), (4, 12));
        assert!(cameras_to_tensor(&[], 4, &device).is_err());
    }

    // -----------------------------------------------------------------------
    // Reference-image resampling
    // -----------------------------------------------------------------------

    #[test]
    fn resample_hwc_reaches_the_last_source_pixel() {
        // 4×1 step edge → 2×1: the right output sample must come from the right
        // half of the source, which the old integer-ratio crop never reached.
        let src = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0_f32,
        ];
        let out = resample_hwc(&src, 4, 1, 2, 1);
        assert_eq!(out.len(), 6);
        assert!(out[0] < 0.25, "left sample = {}", out[0]);
        assert!(out[3] > 0.75, "right sample = {}", out[3]);
    }

    #[test]
    fn reference_image_covers_the_whole_frame_and_uses_clip_normalization() {
        let (channels, h, w) = (3_usize, 512_usize, 512_usize);
        let mut chw = vec![0.0_f32; channels * h * w];
        // Marker in the bottom-right corner: a 512² input used to be cropped to
        // its top-left 448², dropping this entirely.
        for ch in 0..channels {
            for y in (h - 32)..h {
                for x in (w - 32)..w {
                    chw[ch * h * w + y * w + x] = 1.0;
                }
            }
        }

        let device = Device::Cpu;
        let images = Tensor::from_vec(chw, (1, channels, h, w), &device).expect("input tensor");
        let reference = prepare_reference_image(&images).expect("reference image");
        assert_eq!(
            reference.dims4().expect("dims"),
            (1, channels, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE)
        );

        let data: Vec<f32> = reference
            .flatten_all()
            .and_then(|t| t.to_vec1())
            .expect("reference data");
        let plane = CLIP_INPUT_SIZE * CLIP_INPUT_SIZE;
        let origin = data[0];
        let corner = data[plane - 1];
        assert!(
            corner > origin + 1.0,
            "bottom-right marker lost: corner={corner}, origin={origin}"
        );
        // Black maps to -mean/std under CLIP's own normalization.
        let expected_black = -CLIP_MEAN[0] / CLIP_STD[0];
        assert!(
            (origin - expected_black).abs() < 1e-3,
            "expected {expected_black}, got {origin}"
        );
    }

    // -----------------------------------------------------------------------
    // View consistency
    // -----------------------------------------------------------------------

    #[test]
    fn view_consistency_tolerates_a_short_depth_slice() {
        let loss = ViewConsistencyLoss::default();
        let (w, h) = (4_usize, 4_usize);
        let views: Vec<Vec<f32>> = (0..4).map(|_| vec![0.5_f32; w * h * 3]).collect();
        let cameras: Vec<Camera> = (0..4)
            .map(|_| Camera::default_front(w as u32, h as u32))
            .collect();
        // Two depth maps for four views: this used to index out of bounds.
        let depths = vec![vec![1.0_f32; w * h], vec![1.0_f32; w * h]];

        let l = loss.compute(&views, &cameras, Some(&depths), w, h);
        assert!(l.is_finite(), "loss must stay finite, got {l}");
    }

    #[test]
    fn warp_view_marks_only_the_pixels_it_wrote() {
        let (w, h) = (4_usize, 4_usize);
        let camera = Camera::default_front(w as u32, h as u32);
        let src = vec![0.25_f32; w * h * 3];
        let mut depth = vec![0.0_f32; w * h];
        depth[5] = 2.0;

        let warped = warp_view(&src, &camera, &camera, &depth, w, h);
        assert_eq!(warped.mask.iter().filter(|valid| **valid).count(), 1);
        assert!(
            warped.mask[5],
            "identity warp must land on the source pixel"
        );
        assert!((warped.pixels[15] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn warp_view_keeps_the_nearest_contending_sample() {
        let (w, h) = (4_usize, 4_usize);
        let src_cam = Camera::default_front(w as u32, h as u32);
        // A near-zero focal length collapses the whole source frame onto the
        // principal point, so two source samples contend for one target pixel.
        let mut tgt_cam = src_cam.clone();
        tgt_cam.focal_x = 0.001;
        tgt_cam.focal_y = 0.001;

        let mut src = vec![0.0_f32; w * h * 3];
        for c in 0..3 {
            src[c] = 0.75; // source pixel 0 — near, scanned first
            src[3 + c] = 0.25; // source pixel 1 — far, scanned second
        }
        let mut depth = vec![0.0_f32; w * h];
        depth[0] = 1.0;
        depth[1] = 5.0;

        let warped = warp_view(&src, &src_cam, &tgt_cam, &depth, w, h);
        assert_eq!(warped.mask.iter().filter(|valid| **valid).count(), 1);
        assert!(
            warped.mask[10],
            "both samples must land on the centre pixel"
        );
        // Without the depth buffer the *later*, farther sample would overwrite
        // the nearer one — occluded geometry winning by scan order.
        assert!(
            (warped.pixels[30] - 0.75).abs() < 1e-6,
            "the nearer sample must win, got {}",
            warped.pixels[30]
        );
    }

    #[test]
    fn warped_consistency_ignores_warp_holes() {
        let loss = ViewConsistencyLoss::default();
        let (w, h) = (8_usize, 8_usize);
        let dark = vec![0.0_f32; w * h * 3];
        let bright = vec![1.0_f32; w * h * 3];
        let camera = Camera::default_front(w as u32, h as u32);
        // No positive depth ⇒ nothing warps ⇒ every target pixel is a hole. The
        // old code averaged those in as an L1 distance from black.
        let depths = vec![vec![0.0_f32; w * h], vec![0.0_f32; w * h]];

        let l = loss.compute(
            &[dark, bright],
            &[camera.clone(), camera],
            Some(&depths),
            w,
            h,
        );
        assert_eq!(l, 0.0, "warp holes must not be compared, got {l}");
    }

    #[test]
    fn view_consistency_from_config_disables_warping() {
        let config = DiffusionTargetConfig {
            view_consistency_weight: 0.25,
            enable_view_warping: false,
            ..Default::default()
        };
        let loss = ViewConsistencyLoss::from_config(&config);
        assert!((loss.weight - 0.25).abs() < 1e-6);
        assert!(!loss.enable_warping);

        let (w, h) = (4_usize, 4_usize);
        let dark = vec![0.0_f32; w * h * 3];
        let bright = vec![1.0_f32; w * h * 3];
        let camera = Camera::default_front(w as u32, h as u32);
        let depths = vec![vec![1.0_f32; w * h], vec![1.0_f32; w * h]];

        let with_depths = loss.compute(
            &[dark.clone(), bright.clone()],
            &[camera.clone(), camera.clone()],
            Some(&depths),
            w,
            h,
        );
        let without = loss.compute(&[dark, bright], &[camera.clone(), camera], None, w, h);
        assert!(
            (with_depths - without).abs() < 1e-6,
            "warping disabled must ignore depth maps: {with_depths} vs {without}"
        );
    }

    // -----------------------------------------------------------------------
    // SDS numerics
    // -----------------------------------------------------------------------

    #[test]
    fn sds_loss_skips_empty_views() {
        let loss = SdsLoss::default();
        // A single empty view used to produce 0/0 = NaN.
        assert_eq!(loss.compute(&[Vec::new()], &[Vec::new()], 500), 0.0);

        let rendered = vec![Vec::new(), vec![0.0_f32; 16]];
        let targets = vec![Vec::new(), vec![1.0_f32; 16]];
        let mixed = loss.compute(&rendered, &targets, 500);
        assert!(mixed.is_finite() && mixed > 0.0, "loss = {mixed}");

        // The skipped view must not dilute the mean either.
        let only = loss.compute(&[vec![0.0_f32; 16]], &[vec![1.0_f32; 16]], 500);
        assert!(
            (mixed - only).abs() < 1e-6,
            "mixed={mixed}, single={only}: empty views must not count"
        );
    }

    #[test]
    fn ddpm_alpha_cumprod_survives_degenerate_schedules() {
        for max_timestep in [0_u32, 1, 2] {
            for t in [0_u32, 1, 5] {
                let alpha = ddpm_alpha_cumprod(t, max_timestep);
                assert!(
                    alpha.is_finite() && alpha > 0.0 && alpha <= 1.0,
                    "alpha({t}, {max_timestep}) = {alpha}"
                );
            }
        }
        // `t` is clamped to the last schedule entry instead of extrapolating.
        let at_end = ddpm_alpha_cumprod(1000, 1000);
        let at_last = ddpm_alpha_cumprod(999, 1000);
        assert!((at_end - at_last).abs() < 1e-9, "{at_end} vs {at_last}");
    }

    #[test]
    fn sds_loss_weight_is_finite_for_degenerate_max_timestep() {
        let sigma = SdsLoss {
            weighting: SdsWeighting::SigmaBased,
            max_timestep: 1,
        };
        let w = sigma.weight(500);
        assert!(w.is_finite(), "sigma weight = {w}");

        let linear = SdsLoss {
            weighting: SdsWeighting::Linear,
            max_timestep: 0,
        };
        assert!(linear.weight(1).is_finite());
    }

    #[test]
    fn sds_loss_new_rejects_short_schedules() {
        assert!(SdsLoss::new(SdsWeighting::Linear, 0).is_err());
        assert!(SdsLoss::new(SdsWeighting::Linear, 1).is_err());
        let ok = SdsLoss::new(SdsWeighting::Linear, 2).expect("2 is the minimum schedule");
        assert_eq!(ok.max_timestep, 2);
        assert_eq!(SdsLoss::default().max_timestep, DDPM_TRAIN_TIMESTEPS);
    }
}
