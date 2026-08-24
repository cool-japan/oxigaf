//! Full multi-view diffusion pipeline.
//!
//! Orchestrates the CLIP encoder, U-Net, VAE, and DDIM scheduler to
//! generate multi-view images from a single reference photo and camera poses.
//!
//! # Reproducibility
//!
//! [`MultiViewDiffusionPipeline::generate`] derives its initial latent noise
//! from the `seed` argument through a dependency-free xorshift64 + Box-Muller
//! stream, so two runs with the same seed, config and inputs start from
//! bit-identical latents and (the DDIM sampler being deterministic) produce
//! bit-identical outputs.
//!
//! **Known gap**: the optional latent upsampler still draws its own
//! initialisation noise from candle's process-global RNG
//! (`upsampler.rs`), so runs with `upsampler_mode = Some(UpsamplerMode::SdX2)`
//! are not yet reproducible end-to-end.
//!
//! # Weight offloading
//!
//! [`DiffusionConfig::offload_strategy`] is honoured: with
//! [`OffloadStrategy::Sequential`] or [`OffloadStrategy::CacheOne`] the
//! components are built lazily from `weights_dir` right before the inference
//! phase that needs them and dropped again afterwards, so peak host/device
//! memory is bounded by the largest phase rather than by the sum of all
//! components. [`OffloadStrategy::AllInMemory`] (the default) keeps the
//! previous eager behaviour: everything is loaded by
//! [`MultiViewDiffusionPipeline::load`] and stays resident.
//!
//! # Streaming
//!
//! [`GenerationSession`] exposes the denoising loop one step at a time
//! (`begin_session` → `step_session`* → `finish_session`) so callers such as
//! [`crate::streaming`] can render partial results while generation runs.

use std::f32::consts::PI;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn as nn;

use crate::clip::{build_clip_encoder, ClipImageEncoder};
use crate::config::DiffusionConfig;
use crate::scheduler::{DdimScheduler, PredictionType};
use crate::unet::MultiViewUNet;
use crate::upsampler::{LatentUpsampler, UpsamplerMode};
use crate::vae::Vae;
use crate::weight_offload::{ComponentType, MemoryBudget, OffloadSchedule, OffloadStrategy};
use crate::DiffusionError;

// ---------------------------------------------------------------------------
// Deterministic noise sampling
// ---------------------------------------------------------------------------

/// Advance a 64-bit xorshift PRNG and return the new state.
///
/// The zero state is a fixed point of xorshift, so it is patched to `1`.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

/// Uniform `f32` in `[0, 1)` using 53 mantissa bits.
#[inline]
fn xorshift_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 11) as f32 / (1u64 << 53) as f32
}

/// Box-Muller transform: maps two uniform samples to a pair of standard normals.
#[inline]
fn box_muller(u1: f32, u2: f32) -> (f32, f32) {
    let r = (-2.0_f32 * u1.max(1e-10).ln()).sqrt();
    let theta = 2.0 * PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Draw `count` standard-normal samples from a stream seeded by `seed`.
///
/// Fully deterministic and device independent: the same `seed` always yields
/// the same values, on CPU as well as on GPU backends (candle's own
/// `Tensor::randn` uses a process-global RNG that cannot be seeded on CPU).
fn seeded_normal_values(count: usize, seed: u64) -> Vec<f32> {
    // xorshift64 requires a non-zero state; 0 is a legitimate user seed.
    let mut state = if seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        seed
    };
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let u1 = xorshift_f32(&mut state);
        let u2 = xorshift_f32(&mut state);
        let (z0, z1) = box_muller(u1, u2);
        out.push(z0);
        // Box-Muller produces samples in pairs; drop the second one when the
        // requested count is odd.
        if out.len() < count {
            out.push(z1);
        }
    }
    out
}

/// Build a `(v, c, h, w)` standard-normal tensor from the seeded stream.
fn seeded_normal_tensor(
    shape: (usize, usize, usize, usize),
    seed: u64,
    device: &Device,
) -> Result<Tensor, DiffusionError> {
    let (v, c, h, w) = shape;
    let values = seeded_normal_values(v * c * h * w, seed);
    Tensor::from_vec(values, shape, device)
        .map_err(|e| DiffusionError::Inference(format!("seeded noise init: {e}")))
}

// ---------------------------------------------------------------------------
// Classifier-free guidance helpers
// ---------------------------------------------------------------------------

/// Whether the unconditional U-Net pass is required for `guidance_scale`.
///
/// At `guidance_scale == 1.0` the CFG formula collapses to
/// `uncond + 1.0 * (cond - uncond) == cond`, so the unconditional pass is pure
/// waste — exactly half of the denoising compute.
#[inline]
fn needs_uncond_pass(guidance_scale: f64) -> bool {
    (guidance_scale - 1.0).abs() > f64::EPSILON
}

/// Apply `pred = uncond + guidance_scale * (cond - uncond)`.
fn combine_cfg(
    cond: &Tensor,
    uncond: &Tensor,
    guidance_scale: f64,
) -> Result<Tensor, DiffusionError> {
    let diff = (cond - uncond).map_err(|e| DiffusionError::Inference(format!("CFG diff: {e}")))?;
    (uncond + (diff * guidance_scale))
        .map_err(|e| DiffusionError::Inference(format!("CFG combine: {e}")))
}

// ---------------------------------------------------------------------------
// Chunking / offload helpers
// ---------------------------------------------------------------------------

/// Split `num_views` into `(start, len)` ranges of at most `chunk_size` views.
///
/// A `chunk_size` of `0` is treated as `1`.
fn chunk_ranges(num_views: usize, chunk_size: usize) -> Vec<(usize, usize)> {
    let chunk = chunk_size.max(1);
    let mut ranges = Vec::with_capacity(num_views.div_ceil(chunk));
    let mut start = 0;
    while start < num_views {
        let len = chunk.min(num_views - start);
        ranges.push((start, len));
        start += len;
    }
    ranges
}

/// Estimated resident weight size of a pipeline-owned component.
///
/// The pipeline holds a single [`Vae`] that owns both halves of the
/// autoencoder, so the decoder entry is charged for encoder + decoder.
fn component_size_mb(component: ComponentType) -> f32 {
    match component {
        ComponentType::VaeDecoder | ComponentType::VaeEncoder => {
            ComponentType::VaeDecoder.estimated_size_mb()
                + ComponentType::VaeEncoder.estimated_size_mb()
        }
        other => other.estimated_size_mb(),
    }
}

/// Weight memory (MB) charged for `component` given the configured upsampler.
///
/// Identical to [`component_size_mb`] except that a `BilinearVae` upsampler is
/// pure interpolation and holds no weights at all, so it costs nothing.
fn component_weight_mb(component: ComponentType, upsampler_mode: Option<UpsamplerMode>) -> f32 {
    if component == ComponentType::LatentUpsampler
        && upsampler_mode == Some(UpsamplerMode::BilinearVae)
    {
        return 0.0;
    }
    component_size_mb(component)
}

/// Whether `component` should be dropped once its inference phase is over.
fn component_should_release(strategy: OffloadStrategy, component: ComponentType) -> bool {
    match strategy {
        // Everything stays resident.
        OffloadStrategy::AllInMemory => false,
        // Nothing stays resident.
        OffloadStrategy::Sequential => true,
        // The U-Net is re-used every denoising step, so it stays cached.
        OffloadStrategy::CacheOne => component != ComponentType::MultiViewUNet,
    }
}

/// Decode `latents` through `vae` in chunks of at most `chunk_size` views.
///
/// Every operation in [`Vae::decode`] is per-sample — convolutions, group
/// normalisation (groups are taken over channels *within* a sample) and the
/// mid-block attention, which reshapes to `(batch, 3, channels, h*w)` and never
/// mixes across the batch dimension. Decoding view-chunks separately and
/// concatenating is therefore numerically identical to one batched decode,
/// while peak activation memory scales with `chunk_size` instead of the total
/// view count.
fn decode_chunked(
    vae: &Vae,
    latents: &Tensor,
    chunk_size: usize,
) -> Result<Tensor, DiffusionError> {
    let num_views = latents
        .dim(0)
        .map_err(|e| DiffusionError::Inference(format!("latents dim0: {e}")))?;
    let ranges = chunk_ranges(num_views, chunk_size);
    if ranges.len() <= 1 {
        return vae
            .decode(latents)
            .map_err(|e| DiffusionError::Inference(format!("VAE decode: {e}")));
    }

    let mut decoded_chunks: Vec<Tensor> = Vec::with_capacity(ranges.len());
    for (start, len) in ranges {
        let chunk = latents
            .narrow(0, start, len)
            .map_err(|e| DiffusionError::Inference(format!("latent chunk at {start}: {e}")))?;
        let decoded = vae
            .decode(&chunk)
            .map_err(|e| DiffusionError::Inference(format!("VAE decode chunk at {start}: {e}")))?;
        decoded_chunks.push(decoded);
    }

    Tensor::cat(&decoded_chunks, 0)
        .map_err(|e| DiffusionError::Inference(format!("sequential decode cat: {e}")))
}

/// Split a `(V, C, H, W)` batch into `V` tensors of shape `(C, H, W)`.
fn split_views(images: &Tensor) -> Result<Vec<Tensor>, DiffusionError> {
    let num_views = images
        .dim(0)
        .map_err(|e| DiffusionError::Inference(format!("images dim0: {e}")))?;
    let mut views = Vec::with_capacity(num_views);
    for i in 0..num_views {
        let img = images
            .narrow(0, i, 1)
            .and_then(|t| t.squeeze(0))
            .map_err(|e| DiffusionError::Inference(format!("split view {i}: {e}")))?;
        views.push(img);
    }
    Ok(views)
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// Output of the multi-view diffusion pipeline.
#[derive(Debug)]
pub struct MultiViewOutput {
    /// Generated images, one per view, as `(3, H, W)` tensors in `[0, 1]`.
    pub images: Vec<Tensor>,
    /// Width of each generated image.
    pub width: u32,
    /// Height of each generated image.
    pub height: u32,
}

// ---------------------------------------------------------------------------
// GenerationSession
// ---------------------------------------------------------------------------

/// A resumable multi-view denoising run.
///
/// Created by [`MultiViewDiffusionPipeline::begin_session`], advanced one DDIM
/// step at a time with [`MultiViewDiffusionPipeline::step_session`] and turned
/// into images by [`MultiViewDiffusionPipeline::finish_session`]. Between steps
/// the current latents can be decoded with
/// [`MultiViewDiffusionPipeline::preview_images`] for progressive display.
#[derive(Debug)]
pub struct GenerationSession {
    /// Current noisy latents, `(num_views, latent_channels, h, w)`.
    latents: Tensor,
    /// CLIP image tokens expanded to all views.
    ip_tokens: Tensor,
    /// Null text context (GAF conditions on images, not text).
    null_context: Tensor,
    /// Per-view flattened extrinsics.
    camera_poses: Tensor,
    /// Encoded normal maps concatenated to the model input each step.
    normal_map_latents: Tensor,
    /// Descending DDIM timesteps for this run.
    timesteps: Vec<usize>,
    /// Number of steps already applied.
    completed_steps: usize,
    /// Number of views denoised jointly.
    num_views: usize,
}

impl GenerationSession {
    /// Current latents, `(num_views, latent_channels, h, w)`.
    pub fn latents(&self) -> &Tensor {
        &self.latents
    }

    /// Number of views denoised jointly by this session.
    pub fn num_views(&self) -> usize {
        self.num_views
    }

    /// Total number of denoising steps in this run.
    pub fn total_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// Number of denoising steps already applied.
    pub fn completed_steps(&self) -> usize {
        self.completed_steps
    }

    /// `true` once every timestep has been applied.
    pub fn is_finished(&self) -> bool {
        self.completed_steps >= self.timesteps.len()
    }

    /// Fraction of the denoising run completed, in `[0.0, 1.0]`.
    ///
    /// Returns `1.0` for a degenerate empty schedule.
    pub fn progress(&self) -> f32 {
        if self.timesteps.is_empty() {
            return 1.0;
        }
        self.completed_steps as f32 / self.timesteps.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// The full multi-view diffusion pipeline.
pub struct MultiViewDiffusionPipeline {
    /// Denoising U-Net; `None` while offloaded.
    unet: Option<MultiViewUNet>,
    /// Image autoencoder; `None` while offloaded.
    vae: Option<Vae>,
    /// CLIP image encoder for IP-Adapter conditioning; `None` while offloaded.
    clip_encoder: Option<ClipImageEncoder>,
    scheduler: DdimScheduler,
    /// Latent upsampler; `None` when unconfigured **or** while offloaded —
    /// `config.upsampler_mode` is the authority on whether one is configured.
    upsampler: Option<LatentUpsampler>,
    config: DiffusionConfig,
    device: Device,
    /// Directory the component weights are (re-)loaded from.
    weights_dir: PathBuf,
    /// Optional VRAM budget enforced on every component load.
    budget: Option<MemoryBudget>,
}

impl MultiViewDiffusionPipeline {
    /// Load a pipeline from a directory of safetensors files.
    ///
    /// Expected files:
    /// - `unet/diffusion_pytorch_model.safetensors`
    /// - `vae/diffusion_pytorch_model.safetensors`
    /// - `image_encoder/model.safetensors`
    /// - `upsampler/diffusion_pytorch_model.safetensors` (optional, for SdX2 mode)
    ///
    /// With the default [`OffloadStrategy::AllInMemory`] every component is
    /// built here and stays resident. With [`OffloadStrategy::Sequential`] or
    /// [`OffloadStrategy::CacheOne`] only the *presence* of the weight files is
    /// verified; each component is built on demand by the phase that needs it
    /// and dropped again afterwards.
    pub fn load(
        config: DiffusionConfig,
        weights_dir: &Path,
        device: &Device,
    ) -> std::result::Result<Self, DiffusionError> {
        let mut pipeline = Self {
            unet: None,
            vae: None,
            clip_encoder: None,
            scheduler: DdimScheduler::new(1000, PredictionType::VPrediction),
            upsampler: None,
            config,
            device: device.clone(),
            weights_dir: weights_dir.to_path_buf(),
            budget: None,
        };

        let strategy = pipeline.config.offload_strategy;
        match strategy {
            OffloadStrategy::AllInMemory => {
                // Eager load, in the historical order so that a broken weights
                // directory reports the same component first as before.
                pipeline.load_unet()?;
                pipeline.load_vae()?;
                pipeline.load_clip()?;
                pipeline.load_upsampler()?;
            }
            OffloadStrategy::Sequential | OffloadStrategy::CacheOne => {
                pipeline.check_weight_files()?;
                tracing::debug!(
                    "Offloading enabled ({:?}): components are loaded on demand from {}",
                    pipeline.config.offload_strategy,
                    pipeline.weights_dir.display()
                );
            }
        }

        Ok(pipeline)
    }

    /// The configuration this pipeline was built with.
    pub fn config(&self) -> &DiffusionConfig {
        &self.config
    }

    /// The device this pipeline runs on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Override the classifier-free guidance scale used by later runs.
    ///
    /// The value is validated when a run starts: a scale below `1.0` makes
    /// [`Self::begin_session`] fail with `DiffusionError::Inference`.
    pub fn set_guidance_scale(&mut self, guidance_scale: f64) {
        self.config.guidance_scale = guidance_scale;
    }

    /// Attach a VRAM budget that every subsequent component load is checked
    /// against (`DiffusionError::InvalidConfig` when a component does not fit).
    ///
    /// The budget's `currently_loaded_mb` is re-synchronised with the set of
    /// components that are resident right now.
    pub fn set_memory_budget(&mut self, mut budget: MemoryBudget) {
        budget.currently_loaded_mb = self.resident_weight_mb();
        self.budget = Some(budget);
    }

    /// The attached VRAM budget, if any.
    pub fn memory_budget(&self) -> Option<&MemoryBudget> {
        self.budget.as_ref()
    }

    /// Estimated weight memory (MB) of the components currently resident.
    pub fn resident_weight_mb(&self) -> f32 {
        let mut total = 0.0_f32;
        if self.clip_encoder.is_some() {
            total += component_size_mb(ComponentType::ClipImageEncoder);
        }
        if self.unet.is_some() {
            total += component_size_mb(ComponentType::MultiViewUNet);
        }
        if self.vae.is_some() {
            total += component_size_mb(ComponentType::VaeDecoder);
        }
        if self.upsampler.is_some() {
            total += self.weight_mb(ComponentType::LatentUpsampler);
        }
        total
    }

    // -----------------------------------------------------------------------
    // Component load / unload
    // -----------------------------------------------------------------------

    fn unet_weights_path(&self) -> PathBuf {
        self.weights_dir
            .join("unet/diffusion_pytorch_model.safetensors")
    }

    fn vae_weights_path(&self) -> PathBuf {
        self.weights_dir
            .join("vae/diffusion_pytorch_model.safetensors")
    }

    fn clip_weights_path(&self) -> PathBuf {
        self.weights_dir.join("image_encoder/model.safetensors")
    }

    fn upsampler_weights_dir(&self) -> PathBuf {
        self.weights_dir.join("upsampler")
    }

    /// Verify the weight files exist without building any component.
    ///
    /// Keeps the fail-fast contract of [`Self::load`] for the lazy strategies.
    fn check_weight_files(&self) -> Result<(), DiffusionError> {
        for (label, path) in [
            ("U-Net", self.unet_weights_path()),
            ("VAE", self.vae_weights_path()),
            ("CLIP", self.clip_weights_path()),
        ] {
            if !path.exists() {
                return Err(DiffusionError::ModelLoad(format!(
                    "Failed to read {label} weights: {} does not exist",
                    path.display()
                )));
            }
        }
        // `BilinearVae` needs no weights at all, so only `SdX2` is checked.
        if self.config.upsampler_mode == Some(UpsamplerMode::SdX2) {
            let path = self
                .upsampler_weights_dir()
                .join("diffusion_pytorch_model.safetensors");
            if !path.exists() {
                return Err(DiffusionError::ModelLoad(format!(
                    "Failed to read upsampler weights: {} does not exist",
                    path.display()
                )));
            }
        }
        Ok(())
    }

    /// Weight memory (MB) charged for `component` in this pipeline.
    fn weight_mb(&self, component: ComponentType) -> f32 {
        component_weight_mb(component, self.config.upsampler_mode)
    }

    /// Reserve budget space for `component` before loading it.
    fn reserve_memory(&mut self, component: ComponentType) -> Result<(), DiffusionError> {
        let size_mb = self.weight_mb(component);
        if let Some(ref mut budget) = self.budget {
            budget.load(size_mb)?;
        }
        Ok(())
    }

    /// Give budget space back after dropping `component`.
    fn release_memory(&mut self, component: ComponentType) {
        let size_mb = self.weight_mb(component);
        if let Some(ref mut budget) = self.budget {
            budget.unload(size_mb);
        }
    }

    fn build_unet(&self) -> Result<MultiViewUNet, DiffusionError> {
        let path = self.unet_weights_path();
        let data = std::fs::read(&path)
            .map_err(|e| DiffusionError::ModelLoad(format!("Failed to read U-Net weights: {e}")))?;
        let vb = nn::VarBuilder::from_buffered_safetensors(data, DType::F32, &self.device)
            .map_err(|e| DiffusionError::ModelLoad(format!("U-Net VarBuilder: {e}")))?;
        MultiViewUNet::new(vb, &self.config)
            .map_err(|e| DiffusionError::ModelLoad(format!("U-Net build: {e}")))
    }

    fn build_vae(&self) -> Result<Vae, DiffusionError> {
        let path = self.vae_weights_path();
        let data = std::fs::read(&path)
            .map_err(|e| DiffusionError::ModelLoad(format!("Failed to read VAE weights: {e}")))?;
        let vb = nn::VarBuilder::from_buffered_safetensors(data, DType::F32, &self.device)
            .map_err(|e| DiffusionError::ModelLoad(format!("VAE VarBuilder: {e}")))?;
        Vae::new(
            vb,
            self.config.latent_channels,
            self.config.vae_scale_factor,
        )
        .map_err(|e| DiffusionError::ModelLoad(format!("VAE build: {e}")))
    }

    fn build_clip(&self) -> Result<ClipImageEncoder, DiffusionError> {
        let path = self.clip_weights_path();
        let data = std::fs::read(&path)
            .map_err(|e| DiffusionError::ModelLoad(format!("Failed to read CLIP weights: {e}")))?;
        let vb = nn::VarBuilder::from_buffered_safetensors(data, DType::F32, &self.device)
            .map_err(|e| DiffusionError::ModelLoad(format!("CLIP VarBuilder: {e}")))?;
        build_clip_encoder(vb, &self.config)
            .map_err(|e| DiffusionError::ModelLoad(format!("CLIP build: {e}")))
    }

    fn load_unet(&mut self) -> Result<(), DiffusionError> {
        if self.unet.is_some() {
            return Ok(());
        }
        self.reserve_memory(ComponentType::MultiViewUNet)?;
        match self.build_unet() {
            Ok(unet) => {
                self.unet = Some(unet);
                Ok(())
            }
            Err(e) => {
                self.release_memory(ComponentType::MultiViewUNet);
                Err(e)
            }
        }
    }

    fn load_vae(&mut self) -> Result<(), DiffusionError> {
        if self.vae.is_some() {
            return Ok(());
        }
        self.reserve_memory(ComponentType::VaeDecoder)?;
        match self.build_vae() {
            Ok(vae) => {
                self.vae = Some(vae);
                Ok(())
            }
            Err(e) => {
                self.release_memory(ComponentType::VaeDecoder);
                Err(e)
            }
        }
    }

    fn load_clip(&mut self) -> Result<(), DiffusionError> {
        if self.clip_encoder.is_some() {
            return Ok(());
        }
        self.reserve_memory(ComponentType::ClipImageEncoder)?;
        match self.build_clip() {
            Ok(clip) => {
                self.clip_encoder = Some(clip);
                Ok(())
            }
            Err(e) => {
                self.release_memory(ComponentType::ClipImageEncoder);
                Err(e)
            }
        }
    }

    fn load_upsampler(&mut self) -> Result<(), DiffusionError> {
        let mode = match self.config.upsampler_mode {
            Some(mode) => mode,
            // No upsampler configured — nothing to load.
            None => return Ok(()),
        };
        if self.upsampler.is_some() {
            return Ok(());
        }
        self.reserve_memory(ComponentType::LatentUpsampler)?;
        match LatentUpsampler::load(mode, &self.upsampler_weights_dir(), &self.device) {
            Ok(upsampler) => {
                self.upsampler = Some(upsampler);
                Ok(())
            }
            Err(e) => {
                self.release_memory(ComponentType::LatentUpsampler);
                Err(e)
            }
        }
    }

    /// Make sure `component` is resident, loading it from `weights_dir` if not.
    fn ensure_component(&mut self, component: ComponentType) -> Result<(), DiffusionError> {
        match component {
            ComponentType::ClipImageEncoder => self.load_clip(),
            ComponentType::MultiViewUNet => self.load_unet(),
            ComponentType::VaeEncoder | ComponentType::VaeDecoder => self.load_vae(),
            ComponentType::LatentUpsampler => self.load_upsampler(),
        }
    }

    /// Drop `component` and give its memory back to the budget.
    fn unload_component(&mut self, component: ComponentType) {
        let dropped = match component {
            ComponentType::ClipImageEncoder => self.clip_encoder.take().is_some(),
            ComponentType::MultiViewUNet => self.unet.take().is_some(),
            ComponentType::VaeEncoder | ComponentType::VaeDecoder => self.vae.take().is_some(),
            ComponentType::LatentUpsampler => self.upsampler.take().is_some(),
        };
        if dropped {
            let size_mb = self.weight_mb(component);
            self.release_memory(component);
            tracing::debug!(
                "Offloaded {} (~{:.0} MB), {:.0} MB still resident",
                component.display_name(),
                size_mb,
                self.resident_weight_mb()
            );
        }
    }

    /// Drop `component` if the configured strategy says it must not stay
    /// resident once its phase is over.
    fn release_after_phase(&mut self, component: ComponentType) {
        if component_should_release(self.config.offload_strategy, component) {
            self.unload_component(component);
        }
    }

    // -----------------------------------------------------------------------
    // Generation
    // -----------------------------------------------------------------------

    /// Generate multi-view images from a reference image and camera poses.
    ///
    /// - `reference_image`: `(1, 3, 224, 224)` normalised image for CLIP.
    /// - `normal_map_latents`: `(num_views, latent_channels, h, w)` encoded normal maps.
    /// - `camera_poses`: `(num_views, pose_dim)` flattened extrinsics per view.
    /// - `seed`: RNG seed for the initial latents. Two calls with the same seed
    ///   and inputs start from bit-identical noise; the DDIM sampler is
    ///   deterministic, so the whole run is reproducible (see the module-level
    ///   note about the upsampler).
    ///
    /// # Classifier-Free Guidance (CFG)
    ///
    /// This pipeline implements CFG for IP-Adapter conditioning:
    /// - **Conditional pass**: Uses IP tokens from CLIP-encoded reference image
    /// - **Unconditional pass**: Skips IP tokens (no reference conditioning)
    /// - **Formula**: `pred = uncond + guidance_scale * (cond - uncond)`
    ///
    /// At `guidance_scale == 1.0` the formula reduces to the conditional
    /// prediction, so the unconditional pass is skipped entirely and the run
    /// costs half the U-Net evaluations.
    ///
    /// The `guidance_scale` parameter (from config) controls the strength of
    /// conditioning. Typical values:
    /// - `1.0` = no guidance (pure conditional)
    /// - `3.0-7.5` = balanced (default: 3.0 for GAF)
    /// - `>10.0` = strong conditioning (may oversaturate)
    ///
    /// # Errors
    ///
    /// Returns `DiffusionError::Inference` if guidance_scale < 1.0 or if any
    /// tensor operation fails during generation, and
    /// `DiffusionError::ModelLoad` if an offloaded component cannot be
    /// re-loaded from the weights directory.
    pub fn generate(
        &mut self,
        reference_image: &Tensor,
        normal_map_latents: &Tensor,
        camera_poses: &Tensor,
        seed: u64,
    ) -> std::result::Result<MultiViewOutput, DiffusionError> {
        let mut session =
            self.begin_session(reference_image, normal_map_latents, camera_poses, seed)?;
        while self.step_session(&mut session)? {}
        self.finish_session(&session)
    }

    /// Start a resumable generation run using `config.num_inference_steps`.
    ///
    /// See [`Self::generate`] for the argument semantics.
    pub fn begin_session(
        &mut self,
        reference_image: &Tensor,
        normal_map_latents: &Tensor,
        camera_poses: &Tensor,
        seed: u64,
    ) -> std::result::Result<GenerationSession, DiffusionError> {
        let steps = self.config.num_inference_steps;
        self.begin_session_with_steps(
            reference_image,
            normal_map_latents,
            camera_poses,
            seed,
            steps,
        )
    }

    /// Start a resumable generation run with an explicit step count.
    ///
    /// Runs the CLIP conditioning phase, samples the seeded initial latents and
    /// configures the DDIM timesteps; no U-Net evaluation happens yet.
    ///
    /// # Errors
    ///
    /// - `DiffusionError::Inference` when `guidance_scale < 1.0`,
    ///   `num_inference_steps == 0`, or a tensor operation fails.
    /// - `DiffusionError::ModelLoad` when the CLIP encoder cannot be loaded.
    pub fn begin_session_with_steps(
        &mut self,
        reference_image: &Tensor,
        normal_map_latents: &Tensor,
        camera_poses: &Tensor,
        seed: u64,
        num_inference_steps: usize,
    ) -> std::result::Result<GenerationSession, DiffusionError> {
        let num_views = self.config.num_views;
        let latent_size = self.config.latent_size;
        let latent_ch = self.config.latent_channels;

        // Validate guidance_scale
        if self.config.guidance_scale < 1.0 {
            return Err(DiffusionError::Inference(format!(
                "guidance_scale must be >= 1.0, got {}",
                self.config.guidance_scale
            )));
        }
        if num_inference_steps == 0 {
            return Err(DiffusionError::Inference(
                "num_inference_steps must be >= 1".to_string(),
            ));
        }

        // Log the offload schedule for the configured strategy. The phases
        // below drive the actual component load/unload calls.
        let offload_schedule = OffloadSchedule::for_strategy(self.config.offload_strategy);
        tracing::debug!("Offload schedule:\n{}", offload_schedule.format_schedule());
        for (idx, phase) in offload_schedule.phases.iter().enumerate() {
            tracing::trace!(
                "Phase {}/{}: {} — peak {:.0} MB",
                idx + 1,
                offload_schedule.total_phases(),
                phase.name,
                phase.peak_memory_mb()
            );
        }

        // 1. Encode reference image with CLIP for IP-Adapter conditioning
        self.ensure_component(ComponentType::ClipImageEncoder)?;
        let ip_tokens = {
            let clip = self.clip_encoder.as_ref().ok_or_else(|| {
                DiffusionError::Inference("CLIP image encoder is not loaded".to_string())
            })?;
            clip.forward(reference_image)
                .map_err(|e| DiffusionError::Inference(format!("CLIP encode: {e}")))?
        };
        // Expand to all views: (1, seq, dim) -> (V, seq, dim)
        let ip_tokens = ip_tokens
            .repeat(&[num_views, 1, 1])
            .map_err(|e| DiffusionError::Inference(format!("IP token expand: {e}")))?;
        self.release_after_phase(ComponentType::ClipImageEncoder);

        // 2. Prepare null text embedding (GAF doesn't use text conditioning)
        let null_context = Tensor::zeros(
            (num_views, 77, self.config.cross_attention_dim),
            DType::F32,
            &self.device,
        )
        .map_err(|e| DiffusionError::Inference(format!("null context: {e}")))?;

        // 3. Prepare initial noise from the caller's seed
        let latents = seeded_normal_tensor(
            (num_views, latent_ch, latent_size, latent_size),
            seed,
            &self.device,
        )?;

        // 4. Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps);
        let timesteps = self.scheduler.timesteps().to_vec();

        Ok(GenerationSession {
            latents,
            ip_tokens,
            null_context,
            camera_poses: camera_poses.clone(),
            normal_map_latents: normal_map_latents.clone(),
            timesteps,
            completed_steps: 0,
            num_views,
        })
    }

    /// Apply one DDIM step to `session`.
    ///
    /// Returns `Ok(true)` when a step was applied and `Ok(false)` when the
    /// session was already finished.
    ///
    /// Uses two U-Net passes (conditional with IP-Adapter tokens, unconditional
    /// without) unless `guidance_scale == 1.0`, in which case only the
    /// conditional pass runs. The passes are kept separate rather than batched
    /// because IP-Adapter cross-attention takes different shapes for the
    /// conditional and unconditional halves.
    pub fn step_session(
        &mut self,
        session: &mut GenerationSession,
    ) -> std::result::Result<bool, DiffusionError> {
        let t = match session.timesteps.get(session.completed_steps) {
            Some(&t) => t,
            None => return Ok(false),
        };

        self.ensure_component(ComponentType::MultiViewUNet)?;

        // Concatenate noise latents with normal-map latents
        let model_input = Tensor::cat(&[&session.latents, &session.normal_map_latents], 1)
            .map_err(|e| DiffusionError::Inference(format!("concat: {e}")))?;

        let noise_pred = {
            let unet = self.unet.as_ref().ok_or_else(|| {
                DiffusionError::Inference("Multi-view U-Net is not loaded".to_string())
            })?;

            // Forward pass 1: Conditional (with IP-Adapter tokens)
            // This provides identity-preserving conditioning from the reference image
            let noise_pred_cond = unet.forward(
                &model_input,
                t,
                Some(&session.null_context),
                Some(&session.camera_poses),
                Some(&session.ip_tokens),
            )?;

            if needs_uncond_pass(self.config.guidance_scale) {
                // Forward pass 2: Unconditional (without IP-Adapter tokens)
                // This provides the baseline without reference conditioning
                let noise_pred_uncond = unet.forward(
                    &model_input,
                    t,
                    Some(&session.null_context),
                    Some(&session.camera_poses),
                    None, // Skip IP tokens for unconditional
                )?;
                combine_cfg(
                    &noise_pred_cond,
                    &noise_pred_uncond,
                    self.config.guidance_scale,
                )?
            } else {
                // guidance_scale == 1.0 ⇒ uncond + 1.0 * (cond - uncond) == cond
                noise_pred_cond
            }
        };

        // Scheduler step
        session.latents = self
            .scheduler
            .step(&noise_pred, t, &session.latents)
            .map_err(|e| DiffusionError::Inference(format!("scheduler step: {e}")))?;
        session.completed_steps += 1;

        if session.is_finished() {
            self.release_after_phase(ComponentType::MultiViewUNet);
        }

        Ok(true)
    }

    /// Upsample (when configured) and decode the session latents into images.
    pub fn finish_session(
        &mut self,
        session: &GenerationSession,
    ) -> std::result::Result<MultiViewOutput, DiffusionError> {
        let mut latents = session.latents.clone();

        // Upsample latents if configured (32×32 → 64×64)
        if self.config.upsampler_mode.is_some() {
            self.ensure_component(ComponentType::LatentUpsampler)?;
            let steps = self.config.upsampler_steps;
            latents = {
                let upsampler = self.upsampler.as_mut().ok_or_else(|| {
                    DiffusionError::Inference("Latent upsampler is not loaded".to_string())
                })?;
                upsampler
                    .upsample(&latents, steps)
                    .map_err(|e| DiffusionError::Inference(format!("Upsampler: {e}")))?
            };
            self.release_after_phase(ComponentType::LatentUpsampler);
        }

        let images = self.decode_latents(&latents)?;
        let view_images = split_views(&images)?;

        // Calculate output size based on whether upsampling was used.
        // `config.upsampler_mode` — not `self.upsampler` — is the authority:
        // the component may already have been offloaded again.
        let size = if self.config.upsampler_mode.is_some() {
            self.config.image_size as u32 * 2 // 512×512 with upsampling
        } else {
            self.config.image_size as u32 // 256×256 without upsampling
        };
        Ok(MultiViewOutput {
            images: view_images,
            width: size,
            height: size,
        })
    }

    /// Decode the session's *current* latents without upsampling.
    ///
    /// Intended for progressive previews while denoising is still running; the
    /// returned tensors are `(3, H, W)` in `[0, 1]`, one per view, at the
    /// pre-upsampling resolution.
    ///
    /// Each call runs a full VAE decode. Under any offloading strategy other
    /// than [`OffloadStrategy::AllInMemory`] the decoder is additionally
    /// re-loaded from disk on every call, so per-step previews are best
    /// combined with `AllInMemory`.
    pub fn preview_images(
        &mut self,
        session: &GenerationSession,
    ) -> std::result::Result<Vec<Tensor>, DiffusionError> {
        let images = self.decode_latents(session.latents())?;
        split_views(&images)
    }

    /// Decode `latents` into `(num_views, 3, H, W)` images in `[0, 1]`.
    ///
    /// When `config.sequential_vae` is set the decode runs in chunks of
    /// `config.vae_chunk_size` views (numerically identical, lower peak
    /// memory); otherwise all views are decoded in one batch.
    pub fn decode_latents(
        &mut self,
        latents: &Tensor,
    ) -> std::result::Result<Tensor, DiffusionError> {
        self.ensure_component(ComponentType::VaeDecoder)?;

        let decoded = {
            let vae = self
                .vae
                .as_ref()
                .ok_or_else(|| DiffusionError::Inference("VAE is not loaded".to_string()))?;
            if self.config.sequential_vae {
                decode_chunked(vae, latents, self.config.vae_chunk_size)?
            } else {
                vae.decode(latents)
                    .map_err(|e| DiffusionError::Inference(format!("VAE decode: {e}")))?
            }
        };
        self.release_after_phase(ComponentType::VaeDecoder);

        // Post-process: [-1, 1] → [0, 1]
        ((decoded + 1.0).map_err(|e| DiffusionError::Inference(format!("post +1: {e}")))? * 0.5)
            .map_err(|e| DiffusionError::Inference(format!("post *0.5: {e}")))?
            .clamp(0.0, 1.0)
            .map_err(|e| DiffusionError::Inference(format!("clamp: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Seeded noise (regression: `generate` used to ignore its seed) -------

    #[test]
    fn test_seeded_normal_values_is_deterministic() {
        let a = seeded_normal_values(64, 1234);
        let b = seeded_normal_values(64, 1234);
        assert_eq!(a, b, "same seed must produce bit-identical noise");
    }

    #[test]
    fn test_seeded_normal_values_differs_across_seeds() {
        let a = seeded_normal_values(64, 1);
        let b = seeded_normal_values(64, 2);
        assert_ne!(a, b, "different seeds must produce different noise");
    }

    #[test]
    fn test_seeded_normal_values_zero_seed_is_usable() {
        let values = seeded_normal_values(32, 0);
        assert_eq!(values.len(), 32);
        assert!(
            values.iter().any(|v| v.abs() > 1e-6),
            "seed 0 must not collapse the xorshift state to zeros"
        );
        assert!(values.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_seeded_normal_values_odd_count() {
        // Box-Muller yields pairs; an odd request must not over-fill.
        assert_eq!(seeded_normal_values(7, 99).len(), 7);
        assert_eq!(seeded_normal_values(1, 99).len(), 1);
        assert_eq!(seeded_normal_values(0, 99).len(), 0);
    }

    #[test]
    fn test_seeded_normal_values_are_roughly_standard_normal() {
        let values = seeded_normal_values(8192, 7);
        let n = values.len() as f32;
        let mean = values.iter().sum::<f32>() / n;
        let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        assert!(mean.abs() < 0.1, "mean {mean} should be near 0");
        assert!((var - 1.0).abs() < 0.15, "variance {var} should be near 1");
    }

    #[test]
    fn test_seeded_normal_tensor_shape_and_determinism() -> Result<(), DiffusionError> {
        let device = Device::Cpu;
        let a = seeded_normal_tensor((2, 4, 8, 8), 42, &device)?;
        let b = seeded_normal_tensor((2, 4, 8, 8), 42, &device)?;
        assert_eq!(a.dims(), &[2, 4, 8, 8]);
        let a_vec = a
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| DiffusionError::Inference(format!("{e}")))?;
        let b_vec = b
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| DiffusionError::Inference(format!("{e}")))?;
        assert_eq!(a_vec, b_vec);
        Ok(())
    }

    // -- CFG ----------------------------------------------------------------

    #[test]
    fn test_needs_uncond_pass() {
        assert!(
            !needs_uncond_pass(1.0),
            "scale 1.0 must skip the uncond pass"
        );
        assert!(needs_uncond_pass(3.0));
        assert!(needs_uncond_pass(7.5));
    }

    #[test]
    fn test_combine_cfg_matches_formula() -> Result<(), DiffusionError> {
        let device = Device::Cpu;
        let cond = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &device)
            .map_err(|e| DiffusionError::Inference(format!("{e}")))?;
        let uncond = Tensor::from_vec(vec![0.5f32, 0.5, 0.5, 0.5], (1, 4), &device)
            .map_err(|e| DiffusionError::Inference(format!("{e}")))?;
        let out = combine_cfg(&cond, &uncond, 2.0)?;
        let got = out
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| DiffusionError::Inference(format!("{e}")))?;
        // 0.5 + 2 * (c - 0.5)
        let expected = [1.5f32, 3.5, 5.5, 7.5];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-5, "got {g}, expected {e}");
        }
        Ok(())
    }

    // -- VAE chunking --------------------------------------------------------

    #[test]
    fn test_chunk_ranges_exact_split() {
        assert_eq!(chunk_ranges(4, 2), vec![(0, 2), (2, 2)]);
    }

    #[test]
    fn test_chunk_ranges_uneven_tail() {
        assert_eq!(chunk_ranges(5, 2), vec![(0, 2), (2, 2), (4, 1)]);
    }

    #[test]
    fn test_chunk_ranges_single_chunk() {
        assert_eq!(chunk_ranges(4, 8), vec![(0usize, 4usize)]);
        assert!(chunk_ranges(0, 4).is_empty());
    }

    #[test]
    fn test_chunk_ranges_zero_chunk_size_is_one() {
        assert_eq!(chunk_ranges(3, 0), vec![(0, 1), (1, 1), (2, 1)]);
    }

    #[test]
    fn test_chunk_ranges_cover_every_view_once() {
        for chunk in 1..=6 {
            let ranges = chunk_ranges(7, chunk);
            let covered: usize = ranges.iter().map(|(_, len)| len).sum();
            assert_eq!(covered, 7, "chunk {chunk} must cover all views");
            let mut next = 0;
            for (start, len) in ranges {
                assert_eq!(start, next);
                next += len;
            }
        }
    }

    // -- Offloading ----------------------------------------------------------

    #[test]
    fn test_component_size_charges_both_vae_halves() {
        let expected = ComponentType::VaeEncoder.estimated_size_mb()
            + ComponentType::VaeDecoder.estimated_size_mb();
        assert!((component_size_mb(ComponentType::VaeDecoder) - expected).abs() < 1e-3);
        assert!((component_size_mb(ComponentType::VaeEncoder) - expected).abs() < 1e-3);
        assert!(
            (component_size_mb(ComponentType::MultiViewUNet)
                - ComponentType::MultiViewUNet.estimated_size_mb())
            .abs()
                < 1e-3
        );
    }

    #[test]
    fn test_bilinear_upsampler_costs_no_weight_memory() {
        assert!(
            component_weight_mb(
                ComponentType::LatentUpsampler,
                Some(UpsamplerMode::BilinearVae)
            )
            .abs()
                < 1e-6,
            "BilinearVae holds no weights"
        );
        assert!(
            component_weight_mb(ComponentType::LatentUpsampler, Some(UpsamplerMode::SdX2)) > 0.0
        );
        // Non-upsampler components are unaffected by the mode.
        assert!(
            (component_weight_mb(
                ComponentType::MultiViewUNet,
                Some(UpsamplerMode::BilinearVae)
            ) - component_size_mb(ComponentType::MultiViewUNet))
            .abs()
                < 1e-3
        );
    }

    #[test]
    fn test_all_in_memory_never_releases() {
        for component in ComponentType::all_in_inference_order() {
            assert!(!component_should_release(
                OffloadStrategy::AllInMemory,
                *component
            ));
        }
    }

    #[test]
    fn test_sequential_releases_everything() {
        for component in ComponentType::all_in_inference_order() {
            assert!(component_should_release(
                OffloadStrategy::Sequential,
                *component
            ));
        }
    }

    #[test]
    fn test_cache_one_keeps_the_unet_resident() {
        assert!(!component_should_release(
            OffloadStrategy::CacheOne,
            ComponentType::MultiViewUNet
        ));
        for component in ComponentType::all_in_inference_order()
            .iter()
            .filter(|c| **c != ComponentType::MultiViewUNet)
        {
            assert!(component_should_release(
                OffloadStrategy::CacheOne,
                *component
            ));
        }
    }

    #[test]
    fn test_load_reports_missing_weights_for_lazy_strategies() {
        let dir = std::env::temp_dir().join("oxigaf_pipeline_missing_weights_test");
        let config = DiffusionConfig {
            offload_strategy: OffloadStrategy::Sequential,
            ..Default::default()
        };
        let result = MultiViewDiffusionPipeline::load(config, &dir, &Device::Cpu);
        assert!(
            matches!(result, Err(DiffusionError::ModelLoad(_))),
            "a lazy strategy must still fail fast on a missing weights directory"
        );
    }

    #[test]
    fn test_load_reports_missing_weights_for_eager_strategy() {
        let dir = std::env::temp_dir().join("oxigaf_pipeline_missing_weights_eager_test");
        let config = DiffusionConfig::default();
        assert_eq!(config.offload_strategy, OffloadStrategy::AllInMemory);
        let result = MultiViewDiffusionPipeline::load(config, &dir, &Device::Cpu);
        assert!(matches!(result, Err(DiffusionError::ModelLoad(_))));
    }

    // -- Session bookkeeping -------------------------------------------------

    #[test]
    fn test_split_views_shapes() -> Result<(), DiffusionError> {
        let device = Device::Cpu;
        let images = Tensor::zeros((3, 3, 4, 4), DType::F32, &device)
            .map_err(|e| DiffusionError::Inference(format!("{e}")))?;
        let views = split_views(&images)?;
        assert_eq!(views.len(), 3);
        for view in &views {
            assert_eq!(view.dims(), &[3, 4, 4]);
        }
        Ok(())
    }
}
