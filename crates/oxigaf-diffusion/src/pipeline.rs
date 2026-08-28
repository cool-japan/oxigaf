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
//! The optional latent upsampler is keyed off the same seed: its
//! initialisation noise comes from
//! [`LatentUpsampler::upsample_with_seed`][crate::upsampler::LatentUpsampler::upsample_with_seed]
//! (an independent sub-stream of the same generator), so runs with
//! `upsampler_mode = Some(UpsamplerMode::SdX2)` reproduce end-to-end too.
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

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use candle_nn as nn;

use crate::clip::{build_clip_encoder, ClipImageEncoder};
use crate::config::DiffusionConfig;
use crate::controlnet::ControlNetProcessor;
use crate::kv_cache::KVCache;
use crate::profiling::DiffusionProfiler;
use crate::scheduler::{DdimScheduler, PredictionType};
use crate::unet::MultiViewUNet;
use crate::upsampler::{LatentUpsampler, UpsamplerMode};
use crate::vae::Vae;
use crate::weight_offload::{ComponentType, MemoryBudget, OffloadSchedule, OffloadStrategy};
use crate::DiffusionError;

/// Channel count of the pixel-space tensors the VAE encodes and decodes (RGB).
const PIXEL_CHANNELS: usize = 3;

/// Training timestep count of the pipeline's DDIM scheduler.
///
/// Named rather than inlined so [`MultiViewDiffusionPipeline::scheduler`] and
/// the schedule bound `set_timesteps` enforces cannot drift apart.
const SCHEDULER_TRAIN_TIMESTEPS: usize = 1000;

mod support;

pub(crate) use support::seeded_normal_tensor;
use support::{
    combine_cfg, component_should_release, component_size_mb, component_weight_mb,
    conditioning_tag, decode_chunked, encode_chunked, needs_uncond_pass, profile_start,
    profile_stop, split_views, timesteps_from_start_step, unet_activation_estimate,
    validate_session_latents,
};

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
// SessionRequest
// ---------------------------------------------------------------------------

/// Everything [`MultiViewDiffusionPipeline::begin_session_from_latents`] needs
/// to start a run.
///
/// Groups the conditioning tensors with the schedule controls the way
/// [`crate::attention::SpatialTransformerSpec`] groups a transformer stage's
/// geometry, so the img2img entry point takes one value instead of seven
/// positional arguments — four of which are `&Tensor` and were therefore
/// trivially transposable at a call site (`clippy::too_many_arguments` flags
/// the shape, but silently swapping `normal_map_latents` and `camera_poses` is
/// the failure it is warning about).
///
/// The tensors are borrowed for the duration of the call; the session copies
/// what it needs out of them.
#[derive(Debug, Clone, Copy)]
pub struct SessionRequest<'a> {
    /// Reference photo to condition on, `(1, 3, H, W)`, as CLIP expects it.
    /// [`MultiViewDiffusionPipeline::begin_session_from_latents`] encodes it
    /// once and expands the resulting tokens to every view.
    pub reference_image: &'a Tensor,
    /// Encoded normal maps, `(num_views, latent_channels, h, w)`, concatenated
    /// to the latents along the channel axis at every denoising step. Must
    /// agree spatially with [`Self::latents`].
    pub normal_map_latents: &'a Tensor,
    /// Per-view flattened extrinsics, `(num_views, camera_pose_dim)`.
    pub camera_poses: &'a Tensor,
    /// Starting latents, `(num_views, latent_channels, h, w)`, already noised
    /// to [`MultiViewDiffusionPipeline::session_start_timestep`] for this
    /// `(num_inference_steps, start_step)` pair.
    pub latents: &'a Tensor,
    /// Keys whatever stochasticity remains once the starting latents are fixed
    /// — currently the latent upsampler's initialisation noise in
    /// [`MultiViewDiffusionPipeline::upsample_session_latents`]. Any stable
    /// value gives a reproducible run.
    pub seed: u64,
    /// How many timesteps to drop from the front of the descending schedule:
    /// the SDEdit strength control. `0` denoises from the noisiest timestep
    /// (equivalent to a from-scratch run); larger values start further down and
    /// preserve more of the input image. Must be `< num_inference_steps`.
    pub start_step: usize,
    /// Length of the full DDIM schedule before `start_step` is dropped. Must be
    /// non-zero and at most the scheduler's training timestep count.
    pub num_inference_steps: usize,
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
    /// Run seed. Keys every remaining stochastic step — currently only the
    /// latent upsampler's initial noise in
    /// [`MultiViewDiffusionPipeline::upsample_session_latents`].
    seed: u64,
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

    /// The seed this run was started with.
    ///
    /// Keys the initial latents (when they were sampled rather than supplied)
    /// and the latent upsampler's initialisation noise.
    pub fn seed(&self) -> u64 {
        self.seed
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
    /// Optional ControlNet conditioning applied inside the U-Net encoder.
    controlnet: Option<ControlNetProcessor>,
    /// Per-phase timing collector; `None` (the default) means no profiling.
    profiler: Option<DiffusionProfiler>,
    /// Shared cross-attention KV cache handed to the U-Net's IP-Adapter layers.
    kv_cache: Option<Arc<KVCache>>,
    /// Identity of the IP-Adapter tokens the cached projections belong to.
    ///
    /// Re-derived from the CLIP output at the start of every run so a new
    /// reference image can never hit another image's cached K/V.
    kv_conditioning_tag: u64,
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
            scheduler: DdimScheduler::new(SCHEDULER_TRAIN_TIMESTEPS, PredictionType::VPrediction),
            upsampler: None,
            config,
            device: device.clone(),
            weights_dir: weights_dir.to_path_buf(),
            budget: None,
            controlnet: None,
            profiler: None,
            kv_cache: None,
            kv_conditioning_tag: 0,
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

    /// Attach (or, with `None`, detach) ControlNet conditioning.
    ///
    /// Once attached, every [`Self::step_session`] runs the U-Net through
    /// [`MultiViewUNet::forward_with_control`], so the processor's conditions
    /// are injected into the trunk activation leaving each encoder stage that
    /// [`ControlNetConfig::injects_at`][crate::controlnet::ControlNetConfig::injects_at]
    /// accepts. Both the conditional and the unconditional CFG pass are
    /// conditioned, matching diffusers' `StableDiffusionControlNetPipeline`.
    ///
    /// Returns the previously attached processor, if any.
    ///
    /// Without a [`ZeroConv`][crate::controlnet::ZeroConv] registered for a
    /// stage the processor falls back to a channel-constant spatial bias
    /// rather than learned control features — this crate ships no trained
    /// ControlNet encoder. See the [`crate::controlnet`] module docs.
    pub fn set_controlnet(
        &mut self,
        controlnet: Option<ControlNetProcessor>,
    ) -> Option<ControlNetProcessor> {
        std::mem::replace(&mut self.controlnet, controlnet)
    }

    /// The attached ControlNet processor, if any.
    pub fn controlnet(&self) -> Option<&ControlNetProcessor> {
        self.controlnet.as_ref()
    }

    /// Mutable access to the attached ControlNet processor, for adding or
    /// clearing conditions between runs.
    pub fn controlnet_mut(&mut self) -> Option<&mut ControlNetProcessor> {
        self.controlnet.as_mut()
    }

    // -----------------------------------------------------------------------
    // Cross-attention KV cache
    // -----------------------------------------------------------------------

    /// Attach (or, with `None`, detach) a shared cross-attention KV cache.
    ///
    /// The IP-Adapter CLIP tokens are fixed for a whole run, so every
    /// `attn_ip` layer projects the *same* context at every denoising step.
    /// With a cache attached those `to_k`/`to_v` matmuls run once per layer per
    /// run instead of once per layer per step — the saving grows linearly with
    /// `num_inference_steps`.
    ///
    /// The cache is re-attached automatically after a weight-offload reload, so
    /// it survives [`OffloadStrategy::Sequential`] and
    /// [`OffloadStrategy::CacheOne`]. Entries are keyed by a hash of the
    /// reference image's CLIP tokens, so one cache can safely be shared across
    /// pipelines and reference images (as
    /// [`crate::batch_gen::BatchGenerator`] does); its
    /// [`KVCache::stats`] then report real hit/miss counts.
    pub fn set_kv_cache(&mut self, cache: Option<Arc<KVCache>>) {
        self.kv_cache = cache;
        self.apply_kv_cache();
    }

    /// The attached cross-attention KV cache, if any.
    pub fn kv_cache(&self) -> Option<&Arc<KVCache>> {
        self.kv_cache.as_ref()
    }

    /// Push the current `(cache, conditioning tag)` pair into the resident
    /// U-Net, if there is one.
    ///
    /// Called after every U-Net (re-)load and whenever either half changes, so
    /// an offloaded-and-reloaded U-Net does not silently lose its cache.
    fn apply_kv_cache(&mut self) {
        let cache = self.kv_cache.clone();
        let tag = self.kv_conditioning_tag;
        if let Some(unet) = self.unet.as_mut() {
            unet.set_kv_cache(cache, tag);
        }
    }

    // -----------------------------------------------------------------------
    // Profiling
    // -----------------------------------------------------------------------

    /// Start collecting per-phase timings into a [`DiffusionProfiler`].
    ///
    /// Profiling is off by default and is a runtime switch, not a compile-time
    /// one: when disabled the instrumentation costs a single `Option` check per
    /// phase. Once enabled, [`Self::begin_session_with_steps`],
    /// [`Self::step_session`], [`Self::encode_images`],
    /// [`Self::decode_latents`] and [`Self::finish_session`] record entries
    /// named `clip_encode`, `unet_forward`, `scheduler_step`, `vae_encode`,
    /// `vae_decode` and `latent_upsample`. The U-Net entries carry an
    /// activation-memory estimate from
    /// [`estimate_unet_memory_bytes`][crate::profiling::estimate_unet_memory_bytes].
    ///
    /// Calling this again clears any previously collected samples.
    pub fn enable_profiling(&mut self) {
        self.profiler = Some(DiffusionProfiler::new());
    }

    /// Stop collecting timings and return whatever was collected.
    pub fn disable_profiling(&mut self) -> Option<DiffusionProfiler> {
        self.profiler.take()
    }

    /// The active profiler, when [`Self::enable_profiling`] was called.
    ///
    /// Use [`DiffusionProfiler::format_report`] for a human-readable summary or
    /// [`DiffusionProfiler::top_slowest`] to find the dominant phase.
    pub fn profiler(&self) -> Option<&DiffusionProfiler> {
        self.profiler.as_ref()
    }

    /// Drop the collected samples but keep profiling enabled.
    pub fn reset_profiler(&mut self) {
        if let Some(profiler) = self.profiler.as_mut() {
            profiler.clear();
        }
    }

    /// Begin timing `name` when profiling is enabled; otherwise do nothing.
    fn profile_start(&mut self, name: &str, estimated_memory_bytes: usize) {
        profile_start(self.profiler.as_mut(), name, estimated_memory_bytes);
    }

    /// Close the timing opened by [`Self::profile_start`].
    fn profile_stop(&mut self, name: &str) {
        profile_stop(self.profiler.as_mut(), name);
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
                // A freshly built U-Net carries no cache; re-attach the
                // pipeline's, or a lazy offload strategy would silently lose it
                // after the first unload.
                self.apply_kv_cache();
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
        let latents = seeded_normal_tensor(
            (
                self.config.num_views,
                self.config.latent_channels,
                self.config.latent_size,
                self.config.latent_size,
            ),
            seed,
            &self.device,
        )?;
        self.begin_session_from_latents(SessionRequest {
            reference_image,
            normal_map_latents,
            camera_poses,
            latents: &latents,
            seed,
            start_step: 0,
            num_inference_steps,
        })
    }

    /// The first timestep a `(num_inference_steps, start_step)` session applies.
    ///
    /// An img2img caller must noise its encoded latents to exactly this
    /// timestep — e.g. with [`DdimScheduler::add_noise`] on
    /// [`Self::scheduler`] — before handing them to
    /// [`Self::begin_session_from_latents`], or the scheduler's alpha math will
    /// not match the noise level it is given.
    ///
    /// Takes `&mut self` because it configures the pipeline's *own* scheduler
    /// rather than a private copy: a copy would have to restate the schedule
    /// parameters, and would then silently disagree with the real one if those
    /// ever changed. The scheduler is left holding this exact schedule, which
    /// is also what the matching [`Self::begin_session_from_latents`] call
    /// re-derives, so nothing downstream depends on the transient state.
    ///
    /// # Errors
    ///
    /// [`DiffusionError::InvalidConfig`] under exactly the conditions
    /// [`Self::begin_session_from_latents`] would reject the same pair.
    pub fn session_start_timestep(
        &mut self,
        num_inference_steps: usize,
        start_step: usize,
    ) -> std::result::Result<usize, DiffusionError> {
        self.scheduler.set_timesteps(num_inference_steps)?;
        let timesteps = timesteps_from_start_step(self.scheduler.timesteps(), start_step)?;
        timesteps
            .first()
            .copied()
            .ok_or_else(|| DiffusionError::InvalidConfig("empty DDIM schedule".to_string()))
    }

    /// The pipeline's DDIM scheduler.
    ///
    /// Exposed so an img2img caller can reach [`DdimScheduler::add_noise`] with
    /// the *same* alpha table the denoising loop will use, instead of building
    /// a second scheduler that has to guess the schedule parameters.
    pub fn scheduler(&self) -> &DdimScheduler {
        &self.scheduler
    }

    /// Start a resumable generation run from **caller-supplied** latents.
    ///
    /// This is the img2img / SDS entry point: instead of sampling fresh noise
    /// from a seed, the caller hands in the starting latents — typically a real
    /// image encoded with [`Self::encode_images`] and then noised with
    /// [`DdimScheduler::add_noise`] to the timestep the run should start at.
    /// Everything else (CLIP conditioning, null context, DDIM schedule) matches
    /// [`Self::begin_session_with_steps`], which is implemented on top of this
    /// method with `start_step = 0`.
    ///
    /// The run is described by a single [`SessionRequest`]; see that type for
    /// what each field means and must satisfy. In particular
    /// [`SessionRequest::latents`] must be `(num_views, latent_channels, h, w)`
    /// where `num_views` and `latent_channels` come from the pipeline's
    /// [`DiffusionConfig`]. The spatial size may differ from
    /// `config.latent_size` (the U-Net is fully convolutional) but must match
    /// [`SessionRequest::normal_map_latents`], because [`Self::step_session`]
    /// concatenates the two along the channel axis.
    ///
    /// # Denoising strength (`start_step`)
    ///
    /// The full descending schedule of `num_inference_steps` timesteps is
    /// computed, then its first `start_step` entries are dropped. `start_step`
    /// therefore *is* the SDEdit strength control: `0` denoises from the
    /// noisiest timestep (equivalent to a from-scratch run), and larger values
    /// start further down the schedule, preserving more of the input image.
    ///
    /// **The caller must have noised the latents to the first timestep the
    /// session will apply** — [`Self::session_start_timestep`] reports which one
    /// that is for a given `(num_inference_steps, start_step)` pair. Handing in
    /// a lightly-noised latent while starting at `start_step = 0` makes the
    /// scheduler treat it as pure noise and over-denoise it.
    ///
    /// # Errors
    ///
    /// - [`DiffusionError::InvalidLatentShape`] when the latents are not 4-D,
    ///   have the wrong view or channel count, or disagree spatially with the
    ///   normal-map latents.
    /// - `DiffusionError::Inference` when `guidance_scale < 1.0`,
    ///   `num_inference_steps == 0`, or a tensor operation fails.
    /// - `DiffusionError::InvalidConfig` when `num_inference_steps` exceeds the
    ///   scheduler's training timestep count, or when `start_step` is not
    ///   strictly below `num_inference_steps` (which would leave the session
    ///   with nothing to do).
    /// - `DiffusionError::ModelLoad` when the CLIP encoder cannot be loaded.
    pub fn begin_session_from_latents(
        &mut self,
        request: SessionRequest<'_>,
    ) -> std::result::Result<GenerationSession, DiffusionError> {
        let SessionRequest {
            reference_image,
            normal_map_latents,
            camera_poses,
            latents,
            seed,
            start_step,
            num_inference_steps,
        } = request;
        let num_views = self.config.num_views;
        let latent_ch = self.config.latent_channels;

        // Validate the caller's latents before any weights are touched.
        let (_, _, lat_h, lat_w) = validate_session_latents(
            latents.dims(),
            normal_map_latents.dims(),
            num_views,
            latent_ch,
            self.config.latent_size,
        )?;
        if lat_h != self.config.latent_size || lat_w != self.config.latent_size {
            tracing::debug!(
                "generation session running at {lat_h}×{lat_w} latents, \
                 config.latent_size is {}",
                self.config.latent_size
            );
        }

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
        self.profile_start("clip_encode", 0);
        let ip_tokens = {
            let clip = self.clip_encoder.as_ref().ok_or_else(|| {
                DiffusionError::Inference("CLIP image encoder is not loaded".to_string())
            })?;
            clip.forward(reference_image)
                .map_err(|e| DiffusionError::Inference(format!("CLIP encode: {e}")))?
        };
        self.profile_stop("clip_encode");

        // Identify this conditioning for the cross-attention KV cache. Hashed
        // before the per-view expansion (the expansion is a pure function of
        // `num_views`, which is mixed in) so only 1/num_views of the data has
        // to be read back.
        if self.kv_cache.is_some() {
            self.kv_conditioning_tag = conditioning_tag(&ip_tokens, num_views)?;
            self.apply_kv_cache();
        }

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

        // 3. Set scheduler timesteps.
        //
        // `set_timesteps` rejects a step count of `0` (guarded above) and one
        // that exceeds the scheduler's 1000 training timesteps; propagating the
        // error keeps a mis-configured run from silently denoising on a
        // degenerate schedule of identical timesteps.
        self.scheduler.set_timesteps(num_inference_steps)?;
        let timesteps = timesteps_from_start_step(self.scheduler.timesteps(), start_step)?;

        Ok(GenerationSession {
            latents: latents.clone(),
            ip_tokens,
            null_context,
            camera_poses: camera_poses.clone(),
            normal_map_latents: normal_map_latents.clone(),
            timesteps,
            completed_steps: 0,
            num_views,
            seed,
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

        // Activation-memory estimate for this U-Net evaluation, recorded with
        // the timing so a profile shows both cost axes.
        let unet_memory_bytes = unet_activation_estimate(
            &self.config,
            session.num_views,
            model_input.dim(2).unwrap_or(self.config.latent_size),
        );
        self.profile_start("unet_forward", unet_memory_bytes);

        let noise_pred = {
            let unet = self.unet.as_ref().ok_or_else(|| {
                DiffusionError::Inference("Multi-view U-Net is not loaded".to_string())
            })?;
            // ControlNet conditions the structure of the output, so it is
            // applied to both CFG passes (as in diffusers'
            // StableDiffusionControlNetPipeline); it is `None` unless a caller
            // attached a processor with `set_controlnet`.
            let control = self.controlnet.as_ref();

            // Forward pass 1: Conditional (with IP-Adapter tokens)
            // This provides identity-preserving conditioning from the reference image
            let noise_pred_cond = unet.forward_with_control(
                &model_input,
                t,
                Some(&session.null_context),
                Some(&session.camera_poses),
                Some(&session.ip_tokens),
                control,
            )?;

            if needs_uncond_pass(self.config.guidance_scale) {
                // Forward pass 2: Unconditional (without IP-Adapter tokens)
                // This provides the baseline without reference conditioning
                let noise_pred_uncond = unet.forward_with_control(
                    &model_input,
                    t,
                    Some(&session.null_context),
                    Some(&session.camera_poses),
                    None, // Skip IP tokens for unconditional
                    control,
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

        self.profile_stop("unet_forward");

        // Scheduler step
        self.profile_start("scheduler_step", 0);
        let stepped = self
            .scheduler
            .step(&noise_pred, t, &session.latents)
            .map_err(|e| DiffusionError::Inference(format!("scheduler step: {e}")));
        self.profile_stop("scheduler_step");
        session.latents = stepped?;
        session.completed_steps += 1;

        if session.is_finished() {
            self.release_after_phase(ComponentType::MultiViewUNet);
        }

        Ok(true)
    }

    /// Upsample a session's latents, when an upsampler is configured.
    ///
    /// Returns the session's own latents untouched when
    /// `config.upsampler_mode` is `None`, and the 2×-upscaled latents
    /// (32×32 → 64×64 at the default `latent_size`) otherwise.
    ///
    /// Split out of [`Self::finish_session`] so that a caller wanting the
    /// latents themselves — to decode them elsewhere, to feed another stage, or
    /// to compare two runs — does not have to pay for a full VAE decode.
    ///
    /// # Reproducibility
    ///
    /// The upsampler's DDIM initialisation noise is keyed off
    /// [`GenerationSession::seed`], so this is a pure function of the session
    /// and the loaded weights: the same session upsamples to the same latents
    /// every time, and two sessions differing only in `seed` upsample
    /// differently. Before that wiring existed the noise came from
    /// `Tensor::randn` — candle's process-global RNG, unseedable on CPU — and
    /// an upsampled run was not reproducible at all.
    ///
    /// # Errors
    ///
    /// - `DiffusionError::ModelLoad` when an offloaded upsampler cannot be
    ///   re-loaded from the weights directory.
    /// - `DiffusionError::Inference` when the upsampler is not loaded, or when
    ///   the upsampling itself fails.
    pub fn upsample_session_latents(
        &mut self,
        session: &GenerationSession,
    ) -> std::result::Result<Tensor, DiffusionError> {
        let latents = session.latents.clone();
        if self.config.upsampler_mode.is_none() {
            return Ok(latents);
        }

        self.ensure_component(ComponentType::LatentUpsampler)?;
        let steps = self.config.upsampler_steps;
        self.profile_start("latent_upsample", 0);
        let upsampled = {
            let upsampler = self.upsampler.as_mut().ok_or_else(|| {
                DiffusionError::Inference("Latent upsampler is not loaded".to_string())
            })?;
            upsampler
                .upsample_with_seed(&latents, steps, session.seed)
                .map_err(|e| DiffusionError::Inference(format!("Upsampler: {e}")))
        };
        self.profile_stop("latent_upsample");
        let upsampled = upsampled?;
        self.release_after_phase(ComponentType::LatentUpsampler);
        Ok(upsampled)
    }

    /// Upsample (when configured) and decode the session latents into images.
    pub fn finish_session(
        &mut self,
        session: &GenerationSession,
    ) -> std::result::Result<MultiViewOutput, DiffusionError> {
        let latents = self.upsample_session_latents(session)?;
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

    /// Encode `images` into scaled latents with the pipeline's own VAE.
    ///
    /// The inverse of [`Self::decode_latents`], and the entry point for the
    /// img2img / score-distillation loop: encode real renders, noise the
    /// result, and hand it to [`Self::begin_session_from_latents`]. Before this
    /// existed, callers had to build a *second* [`Vae`] from the same
    /// `weights_dir` just to reach the encoder half.
    ///
    /// - `images`: `(B, 3, H, W)` in the **`[-1, 1]`** range the VAE was
    ///   trained on. [`Self::decode_latents`] returns `[0, 1]`, so a round trip
    ///   through both needs `x * 2.0 - 1.0` in between.
    /// - Returns `(B, latent_channels, H/8, W/8)` already multiplied by
    ///   `config.vae_scale_factor`, exactly the scaling
    ///   [`Self::decode_latents`] divides back out.
    ///
    /// When `config.sequential_vae` is set the encode runs in chunks of
    /// `config.vae_chunk_size` views (numerically identical, lower peak
    /// memory); otherwise all views are encoded in one batch.
    ///
    /// # Errors
    ///
    /// - [`DiffusionError::InvalidLatentShape`] when `images` is not a 4-D
    ///   3-channel tensor.
    /// - [`DiffusionError::ModelLoad`] when the VAE cannot be loaded.
    /// - [`DiffusionError::VaeEncodeFailed`] when the encoder itself fails.
    pub fn encode_images(
        &mut self,
        images: &Tensor,
    ) -> std::result::Result<Tensor, DiffusionError> {
        let dims = images.dims().to_vec();
        let (_, channels, _, _) =
            images
                .dims4()
                .map_err(|_| DiffusionError::InvalidLatentShape {
                    expected: vec![0, PIXEL_CHANNELS, 0, 0],
                    got: dims.clone(),
                })?;
        if channels != PIXEL_CHANNELS {
            return Err(DiffusionError::InvalidLatentShape {
                expected: vec![dims[0], PIXEL_CHANNELS, dims[2], dims[3]],
                got: dims,
            });
        }

        self.ensure_component(ComponentType::VaeEncoder)?;
        self.profile_start("vae_encode", 0);
        let encoded = {
            let vae = self
                .vae
                .as_ref()
                .ok_or_else(|| DiffusionError::Inference("VAE is not loaded".to_string()))?;
            if self.config.sequential_vae {
                encode_chunked(vae, images, self.config.vae_chunk_size)
            } else {
                vae.encode(images)
                    .map_err(|e| DiffusionError::VaeEncodeFailed(format!("{e}")))
            }
        };
        self.profile_stop("vae_encode");
        let encoded = encoded?;
        self.release_after_phase(ComponentType::VaeEncoder);
        Ok(encoded)
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

        self.profile_start("vae_decode", 0);
        let decoded = {
            let vae = self
                .vae
                .as_ref()
                .ok_or_else(|| DiffusionError::Inference("VAE is not loaded".to_string()))?;
            if self.config.sequential_vae {
                decode_chunked(vae, latents, self.config.vae_chunk_size)
            } else {
                vae.decode(latents)
                    .map_err(|e| DiffusionError::Inference(format!("VAE decode: {e}")))
            }
        };
        self.profile_stop("vae_decode");
        let decoded = decoded?;
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
    use crate::clip::ClipVisionConfig;

    // -- Offloading ----------------------------------------------------------

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

    // -- Reproducibility -----------------------------------------------------

    /// Side length, in pixels, of the reference image the tiny CLIP encoder in
    /// [`tiny_pipeline`] accepts. `ClipImageEncoder::forward` derives the patch
    /// count from its input and indexes the position embedding by it, so the
    /// reference tensor and the encoder's `image_size` have to agree.
    const TINY_CLIP_IMAGE: usize = 8;

    /// Fixed seed for the tiny run's reference image. Deterministic, so the
    /// conditioning cannot itself be a source of run-to-run variation.
    const TINY_REFERENCE_SEED: u64 = 0x0000_0000_0000_1234;

    /// Fixed seed for the tiny run's normal-map latents.
    const TINY_NORMALS_SEED: u64 = 0x0000_0000_0000_5678;

    /// Fixed seed for the latents a session is handed.
    const TINY_LATENT_SEED: u64 = 0x0000_0000_A5A5_A5A5;

    /// A [`nn::var_builder::SimpleBackend`] that fabricates weights instead of
    /// reading them from a checkpoint.
    ///
    /// The tests below need a network whose weights *vary* — a constant-weight
    /// network can collapse two different latents onto the same pixels — but
    /// not one that is trained. [`nn::VarMap`] would do, except that its
    /// initialisers draw every element individually through `rand_distr`, which
    /// costs around 100 s for the SD VAE's ~83 M parameters in a debug build.
    /// This fills the same tensors from a linear-congruential stream in a
    /// couple of seconds.
    ///
    /// Two properties are preserved deliberately:
    ///
    /// - [`nn::Init::Const`] requests are honoured exactly, so normalisation
    ///   scales stay at 1 and biases at 0. Fabricating those would break the
    ///   networks rather than merely detune them.
    /// - Everything else is uniform on `±1/√fan_in`, the same scale Kaiming and
    ///   LeCun initialisation use, so activations through the deep VAE decoder
    ///   neither vanish nor saturate against the final `clamp(0, 1)` in
    ///   [`MultiViewDiffusionPipeline::decode_latents`].
    ///
    /// The stream is keyed by the tensor's name, so the weights are the same on
    /// every run of the test and on every platform.
    struct FabricatedWeights;

    impl FabricatedWeights {
        /// FNV-1a over the tensor name, forced non-zero: the LCG seed.
        fn seed_for(name: &str) -> u64 {
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            for byte in name.as_bytes() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            hash | 1
        }
    }

    impl nn::var_builder::SimpleBackend for FabricatedWeights {
        fn get(
            &self,
            s: candle_core::Shape,
            name: &str,
            h: nn::Init,
            dtype: DType,
            dev: &Device,
        ) -> candle_core::Result<Tensor> {
            if let nn::Init::Const(value) = h {
                return Tensor::full(value as f32, s, dev)?.to_dtype(dtype);
            }

            // Kaiming/LeCun fan-in: everything but the output dimension.
            let count = s.elem_count();
            let fan_in = match s.dims() {
                [] | [_] => 1,
                dims => count / dims[0].max(1),
            };
            let bound = 1.0 / (fan_in.max(1) as f32).sqrt();

            let mut state = Self::seed_for(name);
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                // Top 24 bits → a uniform in [0, 1), then rescaled to ±bound.
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
                values.push((unit * 2.0 - 1.0) * bound);
            }
            Tensor::from_vec(values, s, dev)?.to_dtype(dtype)
        }

        fn get_unchecked(
            &self,
            name: &str,
            _dtype: DType,
            _dev: &Device,
        ) -> candle_core::Result<Tensor> {
            // Every model in this crate asks through `get`, which carries the
            // shape. Without one there is nothing to fabricate.
            Err(candle_core::Error::Msg(format!(
                "FabricatedWeights cannot supply {name} without a shape"
            )))
        }

        fn contains_tensor(&self, _name: &str) -> bool {
            true
        }
    }

    /// A `VarBuilder` over [`FabricatedWeights`].
    fn fabricated(device: &Device) -> nn::VarBuilder<'static> {
        nn::VarBuilder::from_backend(Box::new(FabricatedWeights), DType::F32, device.clone())
    }

    /// A miniature but structurally faithful pipeline config.
    ///
    /// Sized down everywhere it is free to be: one view, one denoising step,
    /// one upsampler step, and `guidance_scale = 1.0` (which skips the
    /// unconditional U-Net pass entirely). The shapes that are *not* free:
    ///
    /// - `latent_channels = 4`: [`LatentUpsampler`] validates against its own
    ///   4-channel constant and rejects anything else.
    /// - `unet_in_channels = 8`: [`MultiViewDiffusionPipeline::step_session`]
    ///   concatenates the latents with the normal-map latents.
    /// - `clip_embed_dim = 80`: the CLIP vision tower's own hidden width is a
    ///   knob of its own — the encoder projects its output down to
    ///   [`DiffusionConfig::ip_adapter_context_dim`], which is what the U-Net
    ///   builds `attn_ip` for, so the two widths are free to differ. 80 is the
    ///   narrowest tower [`DiffusionConfig::validate`] accepts (ViT-H/14's head
    ///   width), which keeps this config valid; `tiny_pipeline` builds a
    ///   matching 80-wide, 1-layer tower directly rather than going through
    ///   [`build_clip_encoder`]'s 32-layer ViT-H/14 geometry.
    /// - Channel widths stay multiples of `norm_num_groups` (32), and
    ///   `channel_mult = [1, 2]` halves the latent grid once, so `latent_size`
    ///   cannot go below 2 without the decoder's skip connections disagreeing
    ///   with the upsampled feature maps.
    fn tiny_config() -> DiffusionConfig {
        DiffusionConfig {
            num_views: 1,
            guidance_scale: 1.0,
            num_inference_steps: 1,
            upsampler_steps: 1,
            image_size: 16,
            latent_size: 2,
            latent_channels: 4,
            unet_in_channels: 8,
            unet_out_channels: 4,
            cross_attention_dim: 32,
            clip_embed_dim: 80,
            time_embed_dim: 64,
            base_channels: 32,
            channel_mult: vec![1, 2],
            layers_per_block: 1,
            attention_head_dim: vec![2, 4],
            transformer_layers_per_block: vec![1, 1],
            upsampler_mode: Some(UpsamplerMode::SdX2),
            ..DiffusionConfig::default()
        }
    }

    /// The test fixture must satisfy the same invariants a real config does —
    /// otherwise it stops standing in for one. In particular `clip_embed_dim`
    /// has to be a multiple of ViT-H/14's head width now that
    /// [`build_clip_encoder`] sizes the vision tower from it.
    #[test]
    fn tiny_config_is_a_valid_config() {
        tiny_config()
            .validate()
            .expect("the pipeline test fixture must be a valid DiffusionConfig");
    }

    /// Shape of the latents a `tiny_config` session carries.
    fn tiny_latent_shape(config: &DiffusionConfig) -> (usize, usize, usize, usize) {
        (
            config.num_views,
            config.latent_channels,
            config.latent_size,
            config.latent_size,
        )
    }

    /// A pipeline holding **nothing but** an `SdX2` latent upsampler.
    ///
    /// [`MultiViewDiffusionPipeline::upsample_session_latents`] touches no other
    /// component, so leaving the U-Net, VAE and CLIP encoder unbuilt keeps the
    /// seed-plumbing test down to a fraction of a second.
    fn upsampler_only_pipeline(
        device: &Device,
    ) -> std::result::Result<MultiViewDiffusionPipeline, DiffusionError> {
        let config = tiny_config();
        let upsampler = LatentUpsampler::sdx2_from_var_builder(fabricated(device), device)?;
        Ok(MultiViewDiffusionPipeline {
            unet: None,
            vae: None,
            clip_encoder: None,
            scheduler: DdimScheduler::new(SCHEDULER_TRAIN_TIMESTEPS, PredictionType::VPrediction),
            upsampler: Some(upsampler),
            config,
            device: device.clone(),
            weights_dir: std::env::temp_dir().join("oxigaf_upsampler_only_pipeline"),
            budget: None,
            controlnet: None,
            profiler: None,
            kv_cache: None,
            kv_conditioning_tag: 0,
        })
    }

    /// Assemble a complete, tiny, **in-memory** pipeline.
    ///
    /// Every component is built from [`FabricatedWeights`] instead of a
    /// safetensors file, so no weights directory is ever opened. The weights
    /// are keyed by tensor name and each component gets its own builder, so the
    /// U-Net and the upsampler — which both ask for a `conv_in.weight` — do not
    /// collide.
    ///
    /// Built **once** per pipeline: every run through the returned pipeline
    /// shares the same weights, so only explicitly seeded noise can differ
    /// between runs.
    fn tiny_pipeline(
        device: &Device,
    ) -> std::result::Result<MultiViewDiffusionPipeline, DiffusionError> {
        let config = tiny_config();
        let built = |what: &str, e: candle_core::Error| {
            DiffusionError::ModelLoad(format!("tiny {what}: {e}"))
        };

        let unet =
            MultiViewUNet::new(fabricated(device), &config).map_err(|e| built("U-Net", e))?;
        let vae = Vae::new(
            fabricated(device),
            config.latent_channels,
            config.vae_scale_factor,
        )
        .map_err(|e| built("VAE", e))?;

        // ViT-H/14's *ratios* at `tiny_config`'s width, but one layer instead
        // of 32 and an 8×8 input instead of 224² — `build_clip_encoder`'s full
        // geometry is far too large to instantiate in a unit test.
        let clip_config = ClipVisionConfig {
            embed_dim: config.clip_embed_dim,
            num_heads: 2,
            num_layers: 1,
            intermediate_size: config.clip_embed_dim * 2,
            image_size: TINY_CLIP_IMAGE,
            patch_size: 4,
        };
        let clip_encoder = ClipImageEncoder::new(
            fabricated(device),
            &clip_config,
            Some(config.cross_attention_dim),
        )
        .map_err(|e| built("CLIP encoder", e))?;

        let upsampler = LatentUpsampler::sdx2_from_var_builder(fabricated(device), device)?;

        Ok(MultiViewDiffusionPipeline {
            unet: Some(unet),
            vae: Some(vae),
            clip_encoder: Some(clip_encoder),
            scheduler: DdimScheduler::new(SCHEDULER_TRAIN_TIMESTEPS, PredictionType::VPrediction),
            upsampler: Some(upsampler),
            config,
            device: device.clone(),
            // `AllInMemory` never re-loads or releases a component, so nothing
            // here ever opens this path.
            weights_dir: std::env::temp_dir().join("oxigaf_tiny_in_memory_pipeline"),
            budget: None,
            controlnet: None,
            profiler: None,
            kv_cache: None,
            kv_conditioning_tag: 0,
        })
    }

    /// A session carrying `latents` and `seed`.
    ///
    /// [`MultiViewDiffusionPipeline::upsample_session_latents`] reads exactly
    /// those two fields, so the conditioning tensors are placeholders and the
    /// schedule is empty (the session is already finished).
    fn finished_session(
        latents: &Tensor,
        seed: u64,
        device: &Device,
    ) -> std::result::Result<GenerationSession, DiffusionError> {
        let placeholder = Tensor::zeros((1usize, 1usize), DType::F32, device)
            .map_err(|e| DiffusionError::Inference(format!("placeholder: {e}")))?;
        Ok(GenerationSession {
            latents: latents.clone(),
            ip_tokens: placeholder.clone(),
            null_context: placeholder.clone(),
            camera_poses: placeholder.clone(),
            normal_map_latents: placeholder,
            timesteps: Vec::new(),
            completed_steps: 0,
            num_views: 1,
            seed,
        })
    }

    /// Read a tensor back as host floats.
    fn floats(tensor: &Tensor) -> std::result::Result<Vec<f32>, DiffusionError> {
        tensor
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| DiffusionError::Inference(format!("readback: {e}")))
    }

    /// Flatten the first view of an output into host floats.
    fn first_view(output: &MultiViewOutput) -> std::result::Result<Vec<f32>, DiffusionError> {
        let view = output
            .images
            .first()
            .ok_or_else(|| DiffusionError::Inference("run produced no views".to_string()))?;
        floats(view)
    }

    /// The fixed conditioning inputs a tiny run takes.
    fn tiny_inputs(
        config: &DiffusionConfig,
        device: &Device,
    ) -> std::result::Result<(Tensor, Tensor, Tensor), DiffusionError> {
        let reference = seeded_normal_tensor(
            (1, PIXEL_CHANNELS, TINY_CLIP_IMAGE, TINY_CLIP_IMAGE),
            TINY_REFERENCE_SEED,
            device,
        )?;
        let normals = seeded_normal_tensor(tiny_latent_shape(config), TINY_NORMALS_SEED, device)?;
        let poses = Tensor::zeros(
            (config.num_views, config.camera_pose_dim),
            DType::F32,
            device,
        )
        .map_err(|e| DiffusionError::Inference(format!("tiny poses: {e}")))?;
        Ok((reference, normals, poses))
    }

    /// Regression: [`MultiViewDiffusionPipeline::finish_session`] used to call
    /// `LatentUpsampler::upsample`, which drew its DDIM initialisation noise
    /// from `Tensor::randn` — candle's process-global RNG, which cannot be
    /// seeded on CPU. A run configured with
    /// `upsampler_mode = Some(UpsamplerMode::SdX2)` was therefore irreproducible
    /// even though every other stage was deterministic, and the run seed had no
    /// influence whatsoever on the upsampling stage.
    ///
    /// Both halves are asserted, and the *inequality* is the one that pins the
    /// wiring. The latents are held fixed and only `seed` varies, so nothing but
    /// the upsampler's noise can account for a difference. A `generate()`-level
    /// test cannot make that claim: it varies the initial latents with the seed
    /// too, so it would still pass if the seed were dropped on the floor again
    /// before it reached the upsampler.
    #[test]
    fn test_upsampled_latents_are_keyed_by_the_session_seed(
    ) -> std::result::Result<(), DiffusionError> {
        let device = Device::Cpu;
        let mut pipeline = upsampler_only_pipeline(&device)?;
        let shape = tiny_latent_shape(&pipeline.config);
        let latents = seeded_normal_tensor(shape, TINY_LATENT_SEED, &device)?;

        let upsample = |pipeline: &mut MultiViewDiffusionPipeline, seed: u64| {
            let session = finished_session(&latents, seed, &device)?;
            let out = pipeline.upsample_session_latents(&session)?;
            floats(&out)
        };

        let first = upsample(&mut pipeline, 7)?;
        let again = upsample(&mut pipeline, 7)?;
        assert_eq!(first, again, "the same seed must upsample bit-identically");

        let other = upsample(&mut pipeline, 8)?;
        assert_ne!(
            first, other,
            "the session seed must reach the upsampler's initialisation noise"
        );

        // 2× the latent grid, and a real signal rather than a collapsed one.
        assert_eq!(first.len(), shape.0 * shape.1 * shape.2 * 2 * shape.3 * 2);
        assert!(first.iter().all(|v| v.is_finite()));
        Ok(())
    }

    /// Without an upsampler configured, the session's latents pass straight
    /// through — the seed has nothing left to key, and no upsampler is loaded.
    #[test]
    fn test_upsample_session_latents_is_a_no_op_without_an_upsampler(
    ) -> std::result::Result<(), DiffusionError> {
        let device = Device::Cpu;
        let mut pipeline = upsampler_only_pipeline(&device)?;
        pipeline.config.upsampler_mode = None;
        pipeline.upsampler = None;

        let latents = seeded_normal_tensor(tiny_latent_shape(&pipeline.config), 3, &device)?;
        let session = finished_session(&latents, 11, &device)?;
        let passed_through = pipeline.upsample_session_latents(&session)?;

        assert_eq!(passed_through.dims(), latents.dims());
        assert_eq!(floats(&passed_through)?, floats(&latents)?);
        Ok(())
    }

    /// The end-to-end promise of [`MultiViewDiffusionPipeline::generate`]: same
    /// seed, same inputs, same pixels — with the SdX2 upsampler in the path,
    /// which is the configuration that used to break it.
    ///
    /// Deliberately an equality-only test. It runs the whole CLIP → U-Net →
    /// upsampler → VAE stack, which is what makes it worth its runtime; the
    /// sharper "a different seed must change the output" claim is asserted on
    /// the latents above, where it is isolated from everything else.
    #[test]
    fn test_generate_is_reproducible_with_the_sdx2_upsampler(
    ) -> std::result::Result<(), DiffusionError> {
        let device = Device::Cpu;
        let mut pipeline = tiny_pipeline(&device)?;
        let config = pipeline.config.clone();
        let (reference, normals, poses) = tiny_inputs(&config, &device)?;

        let first = pipeline.generate(&reference, &normals, &poses, 99)?;
        let again = pipeline.generate(&reference, &normals, &poses, 99)?;

        // Upsampling doubles the reported edge, and `latent_size` × 8 × 2 is
        // exactly the decoded size.
        assert_eq!(first.width, config.image_size as u32 * 2);
        assert_eq!(first.height, first.width);
        assert_eq!(first.images.len(), config.num_views);

        let pixels = first_view(&first)?;
        assert!(
            pixels.iter().all(|v| v.is_finite()),
            "a decoded view must not contain NaNs"
        );
        // Equality alone would also hold for a stack that saturated to a
        // constant image — a constant is trivially reproducible. Requiring
        // signal makes this assert that the pipeline *computed* something.
        let first_pixel = pixels.first().copied().unwrap_or_default();
        assert!(
            pixels.iter().any(|v| (v - first_pixel).abs() > 1e-6),
            "a decoded view must carry signal, not a single constant value"
        );
        assert_eq!(
            pixels,
            first_view(&again)?,
            "two generate() runs with the same seed must produce identical pixels"
        );
        Ok(())
    }

    /// The img2img entry point is driven by a [`SessionRequest`], and every
    /// field has to land where its name says.
    ///
    /// `start_step` is the one with observable consequences: it drops that many
    /// timesteps off the front of the descending schedule, so `total_steps()`
    /// is `num_inference_steps - start_step` and the run opens on exactly the
    /// timestep [`MultiViewDiffusionPipeline::session_start_timestep`] reports
    /// for the same pair. The caller's latents are carried in verbatim — this
    /// entry point never re-samples them from the seed.
    #[test]
    fn test_session_request_fields_reach_the_schedule_and_latents(
    ) -> std::result::Result<(), DiffusionError> {
        let device = Device::Cpu;
        let mut pipeline = tiny_pipeline(&device)?;
        let config = pipeline.config.clone();
        let (reference, normals, poses) = tiny_inputs(&config, &device)?;
        let latents = seeded_normal_tensor(tiny_latent_shape(&config), TINY_LATENT_SEED, &device)?;

        let expected_first = pipeline.session_start_timestep(4, 1)?;
        let session = pipeline.begin_session_from_latents(SessionRequest {
            reference_image: &reference,
            normal_map_latents: &normals,
            camera_poses: &poses,
            latents: &latents,
            seed: 5,
            start_step: 1,
            num_inference_steps: 4,
        })?;

        assert_eq!(
            session.total_steps(),
            3,
            "start_step = 1 must drop one of the four scheduled timesteps"
        );
        assert_eq!(session.completed_steps(), 0);
        assert_eq!(
            session.timesteps.first().copied(),
            Some(expected_first),
            "the session must open on session_start_timestep's answer"
        );
        assert_eq!(
            floats(session.latents())?,
            floats(&latents)?,
            "the caller's latents must be carried in unchanged"
        );
        assert_eq!(session.num_views(), config.num_views);

        // `start_step` must stay strictly below `num_inference_steps`, or the
        // session would be created with nothing left to do.
        assert!(
            pipeline
                .begin_session_from_latents(SessionRequest {
                    reference_image: &reference,
                    normal_map_latents: &normals,
                    camera_poses: &poses,
                    latents: &latents,
                    seed: 5,
                    start_step: 4,
                    num_inference_steps: 4,
                })
                .is_err(),
            "start_step == num_inference_steps must be rejected"
        );
        Ok(())
    }
}
