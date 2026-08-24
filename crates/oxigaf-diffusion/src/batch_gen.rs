//! Batch generation API for multi-view diffusion inference.
//!
//! Provides a queue-based interface for processing multiple reference images
//! through [`MultiViewDiffusionPipeline`], with a shared [`KVCache`] for
//! cross-attention acceleration.
//!
//! # Model weights are required
//!
//! Generation runs the real CLIP → U-Net → VAE pipeline, so a
//! [`BatchGenerator`] must be built with [`BatchGenerator::with_pipeline`],
//! pointing at a directory of safetensors weights. A generator built with
//! [`BatchGenerator::new`] can still queue, inspect and clear requests, but
//! [`BatchGenerator::process_one`] returns [`DiffusionError::ModelLoad`] rather
//! than fabricating images.
//!
//! # Example
//!
//! Queue management works without weights; generation does not:
//!
//! ```rust
//! use oxigaf_diffusion::batch_gen::{BatchGenerator, BatchGenConfig, GenerationRequest};
//!
//! let gen = BatchGenerator::new(BatchGenConfig::default());
//!
//! let request = GenerationRequest {
//!     id: "img-001".into(),
//!     reference_image: vec![128u8; 256 * 256 * 3],
//!     image_width: 256,
//!     image_height: 256,
//!     num_views: 2,
//!     guidance_scale: None,
//!     num_steps: None,
//!     seed: None,
//! };
//!
//! gen.queue(request.clone()).expect("queueing failed");
//! assert_eq!(gen.queue_len(), 1);
//!
//! // No weights were loaded, so the generator refuses to invent output.
//! assert!(gen.process_one(request).is_err());
//! ```
//!
//! With weights on disk:
//!
//! ```no_run
//! use std::path::Path;
//!
//! use candle_core::Device;
//! use oxigaf_diffusion::batch_gen::{BatchGenerator, BatchGenConfig};
//! use oxigaf_diffusion::DiffusionConfig;
//!
//! let gen = BatchGenerator::with_pipeline(
//!     BatchGenConfig::default(),
//!     DiffusionConfig::default(),
//!     Path::new("weights/gaf"),
//!     &Device::Cpu,
//! )
//! .expect("failed to load model weights");
//! assert!(gen.has_pipeline());
//! ```

use std::path::Path;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Tensor};

use crate::config::DiffusionConfig;
use crate::image_preprocessing::{
    normalize_image, resize_image, ImageDims, NormalizationMode, ResizeFilter,
};
use crate::kv_cache::{KVCache, KVCacheConfig};
use crate::pipeline::MultiViewDiffusionPipeline;
use crate::DiffusionError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Channel count of reference images and generated views (RGB).
const RGB_CHANNELS: usize = 3;

/// Spatial size the CLIP image encoder expects (ViT-H/14 → 224×224).
const CLIP_INPUT_SIZE: usize = 224;

/// Per-channel means used by CLIP image preprocessing (RGB order).
const CLIP_MEAN: [f32; 3] = [0.4814547, 0.4578275, 0.4082107];

/// Per-channel standard deviations used by CLIP image preprocessing (RGB order).
const CLIP_STD: [f32; 3] = [0.2686295, 0.2613026, 0.2757771];

/// Largest reference-image edge accepted, in pixels.
///
/// Reference images are resized down to [`CLIP_INPUT_SIZE`] anyway, so anything
/// beyond this is a caller error rather than a useful input; rejecting it keeps
/// the buffer-size arithmetic far away from any overflow boundary.
const MAX_IMAGE_DIMENSION: u32 = 16_384;

/// Distance from the origin at which orbit cameras are placed, in world units.
const ORBIT_RADIUS: f32 = 2.7;

// ---------------------------------------------------------------------------
// GenerationRequest
// ---------------------------------------------------------------------------

/// Request for generating views of one reference image.
#[derive(Debug, Clone)]
pub struct GenerationRequest {
    /// Unique request ID (for matching responses).
    pub id: String,

    /// Reference image data (RGB bytes, flat row-major, interleaved as `RGBRGB…`).
    ///
    /// Must contain exactly `image_width * image_height * 3` bytes.
    pub reference_image: Vec<u8>,

    /// Width of the reference image in pixels.
    pub image_width: u32,

    /// Height of the reference image in pixels.
    pub image_height: u32,

    /// Number of output views to generate.
    ///
    /// Must equal the loaded pipeline's `DiffusionConfig::num_views`: the U-Net
    /// generates all views jointly with cross-view attention, so the count is
    /// baked into the model configuration rather than chosen per request.
    pub num_views: usize,

    /// Guidance scale (overrides config default when `Some`).
    pub guidance_scale: Option<f32>,

    /// Number of denoising steps (overrides config default when `Some`).
    pub num_steps: Option<usize>,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

// ---------------------------------------------------------------------------
// GeneratedView
// ---------------------------------------------------------------------------

/// Output for one generated view.
#[derive(Debug, Clone)]
pub struct GeneratedView {
    /// Zero-based index of this view in the result set.
    pub view_index: usize,

    /// Pixel data for this view (RGB bytes, flat row-major, interleaved).
    pub image_data: Vec<u8>,

    /// Width of the generated image in pixels.
    ///
    /// This is the pipeline's output resolution (`DiffusionConfig::image_size`,
    /// doubled when a latent upsampler is configured) — *not* the reference
    /// image's width.
    pub width: u32,

    /// Height of the generated image in pixels. See [`GeneratedView::width`].
    pub height: u32,

    /// Wall-clock time in milliseconds attributed to this view.
    ///
    /// All views of a request are denoised jointly in a single pipeline call, so
    /// this is that call's elapsed time divided evenly across the views.
    pub generation_time_ms: f64,
}

// ---------------------------------------------------------------------------
// GenerationResult
// ---------------------------------------------------------------------------

/// Result for one [`GenerationRequest`].
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Matches the `id` from the corresponding [`GenerationRequest`].
    pub id: String,

    /// All generated views, in order.
    pub views: Vec<GeneratedView>,

    /// Total wall-clock time in milliseconds for the entire request.
    pub total_time_ms: f64,

    /// Cumulative number of cross-attention KV lookups served from the shared
    /// [`KVCache`].
    ///
    /// The cache is owned and configured by the generator but is not yet
    /// consulted inside the attention layers, so this currently reports `0`.
    /// See [`BatchGenerator::kv_cache`].
    pub num_cached_kv: usize,
}

impl GenerationResult {
    /// Returns `true` if at least one view was generated.
    pub fn all_views_generated(&self) -> bool {
        !self.views.is_empty()
    }

    /// Throughput in views per second.
    ///
    /// Returns `0.0` when `total_time_ms` is zero (to avoid division by zero).
    pub fn throughput_views_per_sec(&self) -> f64 {
        if self.total_time_ms > 0.0 {
            self.views.len() as f64 / (self.total_time_ms / 1000.0)
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// BatchGenConfig
// ---------------------------------------------------------------------------

/// Configuration for [`BatchGenerator`].
#[derive(Debug, Clone)]
pub struct BatchGenConfig {
    /// Maximum requests to process simultaneously.
    ///
    /// Default: `4`.
    pub max_batch_size: usize,

    /// Maximum output views per request.
    ///
    /// Default: `4`.
    pub max_views_per_request: usize,

    /// Default guidance scale applied when a request omits `guidance_scale`.
    ///
    /// Copied into the pipeline's `DiffusionConfig::guidance_scale` by
    /// [`BatchGenerator::with_pipeline`], so it must be `>= 1.0`.
    ///
    /// Default: `3.0`.
    pub guidance_scale: f32,

    /// Default number of denoising steps applied when a request omits `num_steps`.
    ///
    /// Copied into the pipeline's `DiffusionConfig::num_inference_steps` by
    /// [`BatchGenerator::with_pipeline`], so it must be `> 0`.
    ///
    /// Default: `20`.
    pub num_steps: usize,

    /// Whether to use KV cache for cross-attention.
    ///
    /// Sets [`KVCacheConfig::enabled`] on the generator's shared cache; when
    /// `false` the cache never stores or serves anything.
    ///
    /// Default: `true`.
    pub use_kv_cache: bool,

    /// Whether callers process synchronously (blocking) or through the queue.
    ///
    /// This is a caller-facing hint, not a switch the generator acts on: both
    /// modes are already expressed by the API surface —
    /// [`BatchGenerator::process_one`] is the immediate/blocking path and
    /// [`BatchGenerator::queue`] + [`BatchGenerator::process_batch`] is the
    /// queued path. Both run on the calling thread; the diffusion pipeline is
    /// serialised behind a mutex, so a request never proceeds concurrently with
    /// another.
    ///
    /// Default: `true`.
    pub synchronous: bool,
}

impl Default for BatchGenConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 4,
            max_views_per_request: 4,
            guidance_scale: 3.0,
            num_steps: 20,
            use_kv_cache: true,
            synchronous: true,
        }
    }
}

// ---------------------------------------------------------------------------
// BatchStats
// ---------------------------------------------------------------------------

/// Cumulative statistics for a [`BatchGenerator`].
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total number of generation requests processed **successfully**.
    pub total_requests: u64,

    /// Total number of individual views generated across all requests.
    pub total_views_generated: u64,

    /// Cumulative wall-clock time in milliseconds across all requests.
    pub total_time_ms: f64,

    /// Number of cross-attention KV lookups that were served from cache.
    ///
    /// Mirrors [`crate::kv_cache::CacheStats::hits`] on the generator's shared
    /// cache as of the last successful request. The cache is not yet consulted
    /// inside the attention layers, so this currently stays `0`.
    pub cache_hits: u64,

    /// Number of cross-attention KV lookups that were cache misses.
    ///
    /// Mirrors [`crate::kv_cache::CacheStats::misses`]; see
    /// [`BatchStats::cache_hits`].
    pub cache_misses: u64,
}

impl BatchStats {
    /// Average wall-clock time per generated view in milliseconds.
    ///
    /// Returns `0.0` when no views have been generated yet.
    pub fn average_time_per_view_ms(&self) -> f64 {
        if self.total_views_generated > 0 {
            self.total_time_ms / self.total_views_generated as f64
        } else {
            0.0
        }
    }

    /// Fraction of cache lookups that were hits.
    ///
    /// Returns `0.0` when no lookups have been performed.
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total > 0 {
            self.cache_hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// BatchGenerator
// ---------------------------------------------------------------------------

/// Batch generator for processing multiple reference images.
///
/// Requests can be queued with [`BatchGenerator::queue`] and flushed in one
/// call via [`BatchGenerator::process_batch`], or processed immediately with
/// [`BatchGenerator::process_one`].
///
/// Generation requires model weights — see the module documentation.
pub struct BatchGenerator {
    config: BatchGenConfig,
    kv_cache: Arc<KVCache>,
    /// Loaded diffusion pipeline; `None` until [`BatchGenerator::with_pipeline`]
    /// supplies weights. Serialised because `generate` takes `&mut self`.
    pipeline: Option<Mutex<MultiViewDiffusionPipeline>>,
    /// The configuration the pipeline was loaded with (the pipeline keeps its
    /// own private copy).
    diffusion_config: Option<DiffusionConfig>,
    device: Device,
    request_queue: Mutex<Vec<GenerationRequest>>,
    stats: Mutex<BatchStats>,
}

impl BatchGenerator {
    /// Create a new batch generator **without** model weights.
    ///
    /// Queue management and statistics work; [`BatchGenerator::process_one`]
    /// returns [`DiffusionError::ModelLoad`]. Use
    /// [`BatchGenerator::with_pipeline`] to generate images.
    pub fn new(config: BatchGenConfig) -> Self {
        let kv_cache = Arc::new(KVCache::new(KVCacheConfig {
            enabled: config.use_kv_cache,
            ..KVCacheConfig::default()
        }));
        Self {
            config,
            kv_cache,
            pipeline: None,
            diffusion_config: None,
            device: Device::Cpu,
            request_queue: Mutex::new(Vec::new()),
            stats: Mutex::new(BatchStats::default()),
        }
    }

    /// Create a batch generator backed by a real diffusion pipeline.
    ///
    /// `weights_dir` must contain the safetensors layout documented on
    /// [`MultiViewDiffusionPipeline::load`]. The batch config's
    /// `guidance_scale` and `num_steps` are copied into `diffusion_config`
    /// before loading, so they become the pipeline's defaults.
    ///
    /// # Errors
    ///
    /// - [`DiffusionError::InvalidConfig`] when `config.guidance_scale < 1.0`,
    ///   `config.num_steps == 0`, `config.max_batch_size == 0` or
    ///   `config.max_views_per_request == 0`.
    /// - Anything [`MultiViewDiffusionPipeline::load`] reports (missing or
    ///   corrupt weight files, shape mismatches, …).
    pub fn with_pipeline(
        config: BatchGenConfig,
        diffusion_config: DiffusionConfig,
        weights_dir: &Path,
        device: &Device,
    ) -> Result<Self, DiffusionError> {
        if config.max_batch_size == 0 {
            return Err(DiffusionError::InvalidConfig(
                "max_batch_size must be > 0".into(),
            ));
        }
        if config.max_views_per_request == 0 {
            return Err(DiffusionError::InvalidConfig(
                "max_views_per_request must be > 0".into(),
            ));
        }
        if config.num_steps == 0 {
            return Err(DiffusionError::InvalidConfig(
                "num_steps must be > 0".into(),
            ));
        }
        if !config.guidance_scale.is_finite() || config.guidance_scale < 1.0 {
            return Err(DiffusionError::InvalidConfig(format!(
                "guidance_scale must be a finite value >= 1.0, got {}",
                config.guidance_scale
            )));
        }

        let mut diffusion_config = diffusion_config;
        diffusion_config.guidance_scale = config.guidance_scale as f64;
        diffusion_config.num_inference_steps = config.num_steps;

        let pipeline =
            MultiViewDiffusionPipeline::load(diffusion_config.clone(), weights_dir, device)?;

        let kv_cache = Arc::new(KVCache::new(KVCacheConfig {
            enabled: config.use_kv_cache,
            ..KVCacheConfig::default()
        }));

        Ok(Self {
            config,
            kv_cache,
            pipeline: Some(Mutex::new(pipeline)),
            diffusion_config: Some(diffusion_config),
            device: device.clone(),
            request_queue: Mutex::new(Vec::new()),
            stats: Mutex::new(BatchStats::default()),
        })
    }

    /// Returns `true` when model weights have been loaded and generation is
    /// possible.
    pub fn has_pipeline(&self) -> bool {
        self.pipeline.is_some()
    }

    /// The shared cross-attention KV cache.
    ///
    /// Exposed so the cache can be handed to the attention layers once they
    /// accept one; its statistics are mirrored into [`BatchStats`].
    pub fn kv_cache(&self) -> Arc<KVCache> {
        Arc::clone(&self.kv_cache)
    }

    /// The `DiffusionConfig` the pipeline was loaded with, when there is one.
    pub fn diffusion_config(&self) -> Option<&DiffusionConfig> {
        self.diffusion_config.as_ref()
    }

    // -----------------------------------------------------------------------
    // Queue management
    // -----------------------------------------------------------------------

    /// Add a request to the processing queue.
    ///
    /// # Errors
    ///
    /// - [`DiffusionError::InvalidConfig`] when the queue already holds
    ///   [`BatchGenConfig::max_batch_size`] pending requests, or when the
    ///   request fails the checks described on [`BatchGenerator::process_one`].
    /// - [`DiffusionError::ShapeMismatch`] when `reference_image` does not hold
    ///   `image_width * image_height * 3` bytes.
    pub fn queue(&self, request: GenerationRequest) -> Result<(), DiffusionError> {
        self.validate_request(&request)?;

        let mut queue = self.request_queue.lock().unwrap_or_else(|e| e.into_inner());
        if queue.len() >= self.config.max_batch_size {
            return Err(DiffusionError::InvalidConfig(format!(
                "queue is full: {} pending requests (max {})",
                queue.len(),
                self.config.max_batch_size
            )));
        }
        queue.push(request);
        Ok(())
    }

    /// Number of requests currently in the queue.
    pub fn queue_len(&self) -> usize {
        self.request_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Remove all pending requests from the queue without processing them.
    pub fn clear_queue(&self) {
        self.request_queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    // -----------------------------------------------------------------------
    // Processing
    // -----------------------------------------------------------------------

    /// Process all queued requests and return results in queue order.
    ///
    /// The queue is drained before processing begins, so any requests added
    /// concurrently during processing will not be included in this batch — and
    /// a failure part-way through does not leave stale requests queued.
    ///
    /// # Errors
    ///
    /// Propagates the first error from [`BatchGenerator::process_one`].
    pub fn process_batch(&self) -> Result<Vec<GenerationResult>, DiffusionError> {
        // Drain the queue atomically before any processing.
        let requests: Vec<GenerationRequest> = {
            let mut queue = self.request_queue.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *queue)
        };

        let mut results = Vec::with_capacity(requests.len());
        for req in requests {
            let result = self.process_one(req)?;
            results.push(result);
        }
        Ok(results)
    }

    /// Process a single request immediately, bypassing the queue.
    ///
    /// Runs the full pipeline: the reference image is resized to 224×224 and
    /// CLIP-normalised, camera extrinsics are laid out on a circular orbit, and
    /// [`MultiViewDiffusionPipeline::generate`] denoises all views jointly
    /// before the VAE decodes them into RGB bytes.
    ///
    /// # Per-request overrides
    ///
    /// `guidance_scale` and `num_steps` are pipeline-level settings fixed when
    /// the pipeline was loaded (see [`BatchGenerator::with_pipeline`]). A
    /// request that asks for a *different* value logs a `tracing` warning and is
    /// generated with the loaded configuration instead of silently pretending
    /// the override took effect.
    ///
    /// # Errors
    ///
    /// - [`DiffusionError::InvalidConfig`] when `request.num_views == 0`,
    ///   `num_views > max_views_per_request`, or either image dimension is `0`
    ///   or above 16384.
    /// - [`DiffusionError::ShapeMismatch`] when `reference_image` does not hold
    ///   `image_width * image_height * 3` bytes.
    /// - [`DiffusionError::ModelLoad`] when no model weights were loaded.
    /// - [`DiffusionError::InvalidViewCount`] when `request.num_views` differs
    ///   from the pipeline's configured view count.
    /// - [`DiffusionError::ImageProcessingError`] / [`DiffusionError::Inference`]
    ///   for preprocessing and inference failures.
    pub fn process_one(
        &self,
        request: GenerationRequest,
    ) -> Result<GenerationResult, DiffusionError> {
        let start = std::time::Instant::now();

        self.validate_request(&request)?;

        let (pipeline, diffusion_config) = match (&self.pipeline, &self.diffusion_config) {
            (Some(pipeline), Some(config)) => (pipeline, config),
            _ => {
                return Err(DiffusionError::ModelLoad(
                    "BatchGenerator has no diffusion pipeline: build it with \
                     BatchGenerator::with_pipeline(config, diffusion_config, weights_dir, device) \
                     so real model weights are loaded before generating"
                        .to_string(),
                ));
            }
        };

        if request.num_views != diffusion_config.num_views {
            return Err(DiffusionError::InvalidViewCount {
                expected: diffusion_config.num_views,
                got: request.num_views,
            });
        }

        // --- Resolve per-request overrides ---------------------------------
        let guidance = request.guidance_scale.unwrap_or(self.config.guidance_scale);
        let steps = request.num_steps.unwrap_or(self.config.num_steps);
        if (guidance as f64 - diffusion_config.guidance_scale).abs() > 1e-6 {
            tracing::warn!(
                request_id = %request.id,
                requested = %guidance,
                effective = %diffusion_config.guidance_scale,
                "per-request guidance_scale override ignored: the loaded pipeline's \
                 guidance scale is fixed at load time"
            );
        }
        if steps != diffusion_config.num_inference_steps {
            tracing::warn!(
                request_id = %request.id,
                requested = %steps,
                effective = %diffusion_config.num_inference_steps,
                "per-request num_steps override ignored: the loaded pipeline's step count \
                 is fixed at load time"
            );
        }

        // --- Preprocess inputs ---------------------------------------------
        let reference = self.encode_reference(&request)?;

        let latent_shape = (
            request.num_views,
            diffusion_config.latent_channels,
            diffusion_config.latent_size,
            diffusion_config.latent_size,
        );
        // GAF conditions the U-Net on per-view normal-map latents. The batch API
        // takes only a reference photo, so the normal-map channels are left at
        // zero (the "no geometric prior" case) rather than invented.
        let normal_map_latents = Tensor::zeros(latent_shape, DType::F32, &self.device)
            .map_err(|e| DiffusionError::Inference(format!("normal-map latents: {e}")))?;

        let pose_dim = diffusion_config.camera_pose_dim;
        let poses = orbit_camera_poses(request.num_views, pose_dim);
        let camera_poses = Tensor::from_vec(poses, (request.num_views, pose_dim), &self.device)
            .map_err(|e| DiffusionError::Inference(format!("camera poses: {e}")))?;

        // --- Run the diffusion pipeline ------------------------------------
        let generate_start = std::time::Instant::now();
        let output = {
            let mut guard = pipeline.lock().unwrap_or_else(|e| e.into_inner());
            guard.generate(
                &reference,
                &normal_map_latents,
                &camera_poses,
                request.seed.unwrap_or(0),
            )?
        };
        let generate_ms = generate_start.elapsed().as_secs_f64() * 1000.0;
        let per_view_ms = generate_ms / output.images.len().max(1) as f64;

        // --- Decode views into RGB bytes -----------------------------------
        let mut views = Vec::with_capacity(output.images.len());
        for (view_index, image) in output.images.iter().enumerate() {
            views.push(GeneratedView {
                view_index,
                image_data: tensor_to_rgb8(image)?,
                width: output.width,
                height: output.height,
                generation_time_ms: per_view_ms,
            });
        }

        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        let cache_stats = self.kv_cache.stats();

        // --- Update cumulative statistics ----------------------------------
        {
            let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
            stats.total_requests += 1;
            stats.total_views_generated += views.len() as u64;
            stats.total_time_ms += total_ms;
            stats.cache_hits = cache_stats.hits;
            stats.cache_misses = cache_stats.misses;
        }

        Ok(GenerationResult {
            id: request.id,
            views,
            total_time_ms: total_ms,
            num_cached_kv: cache_stats.hits as usize,
        })
    }

    // -----------------------------------------------------------------------
    // Input handling
    // -----------------------------------------------------------------------

    /// Validate the caller-supplied parts of a request.
    fn validate_request(&self, request: &GenerationRequest) -> Result<(), DiffusionError> {
        if request.num_views == 0 {
            return Err(DiffusionError::InvalidConfig(
                "num_views must be > 0".into(),
            ));
        }
        if request.num_views > self.config.max_views_per_request {
            return Err(DiffusionError::InvalidConfig(format!(
                "num_views {} exceeds max {}",
                request.num_views, self.config.max_views_per_request
            )));
        }
        if request.image_width == 0 || request.image_height == 0 {
            return Err(DiffusionError::InvalidConfig(format!(
                "image dimensions must be > 0, got {}×{}",
                request.image_width, request.image_height
            )));
        }
        if request.image_width > MAX_IMAGE_DIMENSION || request.image_height > MAX_IMAGE_DIMENSION {
            return Err(DiffusionError::InvalidConfig(format!(
                "image dimensions {}×{} exceed the maximum of {} per edge",
                request.image_width, request.image_height, MAX_IMAGE_DIMENSION
            )));
        }

        // Widen before multiplying: `u32 * u32` overflows for large edges.
        let expected_len =
            request.image_width as usize * request.image_height as usize * RGB_CHANNELS;
        if request.reference_image.len() != expected_len {
            return Err(DiffusionError::ShapeMismatch {
                op: "GenerationRequest::reference_image".to_string(),
                expected: vec![
                    request.image_height as usize,
                    request.image_width as usize,
                    RGB_CHANNELS,
                ],
                got: vec![request.reference_image.len()],
            });
        }

        Ok(())
    }

    /// Turn the raw reference bytes into the `(1, 3, 224, 224)` CLIP input the
    /// pipeline expects.
    fn encode_reference(&self, request: &GenerationRequest) -> Result<Tensor, DiffusionError> {
        let height = request.image_height as usize;
        let width = request.image_width as usize;

        // u8 → f32 in [0, 1], still HWC-interleaved.
        let source: Vec<f32> = request
            .reference_image
            .iter()
            .map(|&byte| byte as f32 / 255.0)
            .collect();

        let (resized, resized_dims) = resize_image(
            &source,
            ImageDims::new(height, width, RGB_CHANNELS),
            CLIP_INPUT_SIZE,
            CLIP_INPUT_SIZE,
            ResizeFilter::Bilinear,
        )
        .map_err(|e| DiffusionError::ImageProcessingError(format!("reference resize: {e}")))?;

        let normalized = normalize_image(
            &resized,
            resized_dims,
            NormalizationMode::ZeroMean {
                mean: CLIP_MEAN,
                std: CLIP_STD,
            },
        )
        .map_err(|e| DiffusionError::ImageProcessingError(format!("reference normalize: {e}")))?;

        // HWC → CHW.
        let plane = CLIP_INPUT_SIZE * CLIP_INPUT_SIZE;
        let mut chw = vec![0.0f32; RGB_CHANNELS * plane];
        for y in 0..CLIP_INPUT_SIZE {
            for x in 0..CLIP_INPUT_SIZE {
                let pixel = y * CLIP_INPUT_SIZE + x;
                for channel in 0..RGB_CHANNELS {
                    chw[channel * plane + pixel] = normalized
                        .get(pixel * RGB_CHANNELS + channel)
                        .copied()
                        .unwrap_or(0.0);
                }
            }
        }

        Tensor::from_vec(
            chw,
            (1, RGB_CHANNELS, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE),
            &self.device,
        )
        .map_err(|e| DiffusionError::Inference(format!("reference tensor: {e}")))
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Return a snapshot of cumulative generation statistics.
    pub fn stats(&self) -> BatchStats {
        self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Reset cumulative statistics to zero.
    ///
    /// The shared [`KVCache`]'s own counters are untouched, so the next
    /// successful request restores [`BatchStats::cache_hits`] and
    /// [`BatchStats::cache_misses`] from the cache.
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        *stats = BatchStats::default();
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

/// Builds `num_views` camera extrinsics evenly spaced on a circular orbit.
///
/// Cameras sit on the world XZ circle of radius [`ORBIT_RADIUS`] at zero
/// elevation, all looking at the origin. Each pose is the 3×4 world-to-camera
/// matrix `[R | t]` flattened row-major (12 values) in the OpenCV convention:
/// camera `+X` right, `+Y` down, `+Z` forward, `t = -R · C`.
///
/// `pose_dim` is the model's expected pose width: entries past the 12 matrix
/// values are left at zero, and a `pose_dim` below 12 truncates the matrix.
fn orbit_camera_poses(num_views: usize, pose_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; num_views.saturating_mul(pose_dim)];
    if num_views == 0 || pose_dim == 0 {
        return out;
    }

    for view in 0..num_views {
        let azimuth = std::f32::consts::TAU * view as f32 / num_views as f32;
        let (sin_a, cos_a) = azimuth.sin_cos();

        // Camera centre and the unit forward vector pointing at the origin.
        let centre = [ORBIT_RADIUS * sin_a, 0.0, ORBIT_RADIUS * cos_a];
        let forward = [-sin_a, 0.0, -cos_a];
        // right = normalize(cross(forward, world_up)) with world_up = +Y.
        let right = [-forward[2], 0.0, forward[0]];
        // down = cross(forward, right) — +Y down, as OpenCV expects.
        let down = [
            forward[1] * right[2] - forward[2] * right[1],
            forward[2] * right[0] - forward[0] * right[2],
            forward[0] * right[1] - forward[1] * right[0],
        ];

        let rows = [right, down, forward];
        let base = view * pose_dim;
        for (row_idx, row) in rows.iter().enumerate() {
            // Translation component: t = -R · C.
            let translation = -(row[0] * centre[0] + row[1] * centre[1] + row[2] * centre[2]);
            for (col_idx, &value) in row.iter().enumerate() {
                if let Some(slot) = out.get_mut(base + row_idx * 4 + col_idx) {
                    *slot = value;
                }
            }
            if let Some(slot) = out.get_mut(base + row_idx * 4 + 3) {
                *slot = translation;
            }
        }
    }

    out
}

/// Converts a `(C, H, W)` image tensor with values in `[0, 1]` into
/// row-major interleaved RGB bytes.
///
/// Missing channels are read as `0.0` so a single-channel or malformed tensor
/// degrades instead of panicking.
fn tensor_to_rgb8(image: &Tensor) -> Result<Vec<u8>, DiffusionError> {
    let (channels, height, width) = image
        .dims3()
        .map_err(|e| DiffusionError::Inference(format!("generated view dims: {e}")))?;

    let data = image
        .flatten_all()
        .and_then(|flat| flat.to_vec1::<f32>())
        .map_err(|e| DiffusionError::Inference(format!("generated view readback: {e}")))?;

    let plane = height * width;
    let mut out = vec![0u8; plane * RGB_CHANNELS];
    let last_channel = channels.saturating_sub(1);

    for pixel in 0..plane {
        for channel in 0..RGB_CHANNELS {
            let source = channel.min(last_channel) * plane + pixel;
            let value = data.get(source).copied().unwrap_or(0.0);
            out[pixel * RGB_CHANNELS + channel] = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::KVEntry;

    fn make_request(id: &str, num_views: usize) -> GenerationRequest {
        GenerationRequest {
            id: id.to_string(),
            reference_image: vec![128u8; 64 * 64 * 3],
            image_width: 64,
            image_height: 64,
            num_views,
            guidance_scale: None,
            num_steps: None,
            seed: None,
        }
    }

    // -----------------------------------------------------------------------
    // BatchGenConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_gen_config_default_max_batch_size() {
        assert_eq!(BatchGenConfig::default().max_batch_size, 4);
    }

    #[test]
    fn test_batch_gen_config_default_max_views_per_request() {
        assert_eq!(BatchGenConfig::default().max_views_per_request, 4);
    }

    #[test]
    fn test_batch_gen_config_default_guidance_scale() {
        let cfg = BatchGenConfig::default();
        assert!((cfg.guidance_scale - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_batch_gen_config_default_num_steps() {
        assert_eq!(BatchGenConfig::default().num_steps, 20);
    }

    #[test]
    fn test_batch_gen_config_default_use_kv_cache() {
        assert!(BatchGenConfig::default().use_kv_cache);
    }

    #[test]
    fn test_batch_gen_config_default_synchronous() {
        assert!(BatchGenConfig::default().synchronous);
    }

    // -----------------------------------------------------------------------
    // GenerationRequest construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_generation_request_can_be_constructed() {
        let req = GenerationRequest {
            id: "req-1".into(),
            reference_image: vec![0u8; 256 * 256 * 3],
            image_width: 256,
            image_height: 256,
            num_views: 4,
            guidance_scale: Some(5.0),
            num_steps: Some(30),
            seed: Some(42),
        };
        assert_eq!(req.id, "req-1");
        assert_eq!(req.num_views, 4);
    }

    // -----------------------------------------------------------------------
    // Queue length
    // -----------------------------------------------------------------------

    #[test]
    fn test_queue_len_starts_at_zero() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        assert_eq!(gen.queue_len(), 0);
    }

    #[test]
    fn test_queue_increases_queue_length() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        gen.queue(make_request("a", 1)).expect("queue failed");
        assert_eq!(gen.queue_len(), 1);
        gen.queue(make_request("b", 1)).expect("queue failed");
        assert_eq!(gen.queue_len(), 2);
    }

    #[test]
    fn test_queue_rejects_invalid_request() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        assert!(gen.queue(make_request("bad", 0)).is_err());
        assert_eq!(gen.queue_len(), 0);
    }

    #[test]
    fn test_queue_full_returns_err() {
        let gen = BatchGenerator::new(BatchGenConfig {
            max_batch_size: 1,
            ..Default::default()
        });
        gen.queue(make_request("a", 1)).expect("queue failed");
        assert!(gen.queue(make_request("b", 1)).is_err());
    }

    #[test]
    fn test_clear_queue_resets_to_zero() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        gen.queue(make_request("a", 1)).expect("queue failed");
        gen.queue(make_request("b", 1)).expect("queue failed");
        assert_eq!(gen.queue_len(), 2);
        gen.clear_queue();
        assert_eq!(gen.queue_len(), 0);
    }

    // -----------------------------------------------------------------------
    // process_one — validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_process_one_num_views_zero_returns_err() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        let result = gen.process_one(make_request("x", 0));
        assert!(result.is_err(), "num_views=0 should be an error");
    }

    #[test]
    fn test_process_one_num_views_exceeds_max_returns_err() {
        let config = BatchGenConfig {
            max_views_per_request: 2,
            ..Default::default()
        };
        let gen = BatchGenerator::new(config);
        let result = gen.process_one(make_request("x", 5));
        assert!(result.is_err(), "num_views > max should be an error");
    }

    #[test]
    fn test_process_one_rejects_zero_image_dimensions() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        let req = GenerationRequest {
            image_width: 0,
            image_height: 0,
            reference_image: Vec::new(),
            ..make_request("zero", 1)
        };
        assert!(gen.process_one(req).is_err());
    }

    // Regression: `image_width * image_height * 3` was computed in `u32`, so
    // 40000×40000 panicked in debug builds ("attempt to multiply with overflow")
    // and wrapped to 505,032,704 in release builds, allocating a buffer whose
    // size contradicted the reported width/height.
    #[test]
    fn test_process_one_huge_dimensions_are_rejected_without_overflow() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        let req = GenerationRequest {
            image_width: 40_000,
            image_height: 40_000,
            reference_image: Vec::new(),
            ..make_request("huge", 1)
        };
        let result = gen.process_one(req);
        assert!(result.is_err(), "oversized dimensions must be rejected");
        match result {
            Err(DiffusionError::InvalidConfig(msg)) => {
                assert!(msg.contains("maximum"), "unexpected message: {msg}");
            }
            other => panic!("Expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn test_reference_image_length_is_validated() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        let req = GenerationRequest {
            reference_image: vec![0u8; 10],
            ..make_request("short", 1)
        };
        let result = gen.process_one(req);
        match result {
            Err(DiffusionError::ShapeMismatch { op, got, .. }) => {
                assert!(op.contains("reference_image"), "unexpected op: {op}");
                assert_eq!(got, vec![10]);
            }
            other => panic!("Expected ShapeMismatch, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // process_one — honest failure without weights
    // -----------------------------------------------------------------------

    #[test]
    fn test_process_one_without_weights_returns_model_load_error() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        assert!(!gen.has_pipeline());
        let result = gen.process_one(make_request("no-weights", 2));
        match result {
            Err(DiffusionError::ModelLoad(msg)) => {
                assert!(
                    msg.contains("with_pipeline"),
                    "error should point at the real constructor: {msg}"
                );
            }
            other => panic!("Expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn test_failed_request_does_not_touch_stats() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        let _ = gen.process_one(make_request("no-weights", 2));
        let stats = gen.stats();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.total_views_generated, 0);
        assert!(stats.total_time_ms.abs() < f64::EPSILON);
    }

    #[test]
    fn test_with_pipeline_missing_weights_returns_error() {
        let missing = std::env::temp_dir().join("oxigaf-batch-gen-missing-weights");
        let result = BatchGenerator::with_pipeline(
            BatchGenConfig::default(),
            DiffusionConfig::default(),
            &missing,
            &Device::Cpu,
        );
        assert!(result.is_err(), "missing weights must not silently succeed");
    }

    #[test]
    fn test_with_pipeline_rejects_guidance_below_one() {
        let dir = std::env::temp_dir().join("oxigaf-batch-gen-guidance");
        let result = BatchGenerator::with_pipeline(
            BatchGenConfig {
                guidance_scale: 0.5,
                ..Default::default()
            },
            DiffusionConfig::default(),
            &dir,
            &Device::Cpu,
        );
        match result {
            Err(DiffusionError::InvalidConfig(msg)) => {
                assert!(msg.contains("guidance_scale"), "unexpected message: {msg}");
            }
            other => panic!("Expected InvalidConfig, got {:?}", other.err()),
        }
    }

    #[test]
    fn test_with_pipeline_rejects_zero_steps() {
        let dir = std::env::temp_dir().join("oxigaf-batch-gen-steps");
        let result = BatchGenerator::with_pipeline(
            BatchGenConfig {
                num_steps: 0,
                ..Default::default()
            },
            DiffusionConfig::default(),
            &dir,
            &Device::Cpu,
        );
        assert!(result.is_err(), "num_steps=0 must be rejected");
    }

    // -----------------------------------------------------------------------
    // KV cache wiring
    // -----------------------------------------------------------------------

    #[test]
    fn test_use_kv_cache_true_enables_the_shared_cache() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        let cache = gen.kv_cache();
        cache
            .insert(
                "layer0".to_string(),
                KVEntry::new(vec![0.0; 4], vec![0.0; 4], 1, 1, 2, 2),
            )
            .expect("insert failed");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_use_kv_cache_false_disables_the_shared_cache() {
        let gen = BatchGenerator::new(BatchGenConfig {
            use_kv_cache: false,
            ..Default::default()
        });
        let cache = gen.kv_cache();
        cache
            .insert(
                "layer0".to_string(),
                KVEntry::new(vec![0.0; 4], vec![0.0; 4], 1, 1, 2, 2),
            )
            .expect("insert failed");
        assert!(cache.is_empty(), "a disabled cache must not store entries");
        assert!(!cache.contains("layer0"));
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_start_at_zero() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        let stats = gen.stats();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.total_views_generated, 0);
        assert!((stats.cache_hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_average_time_zero_for_empty() {
        let stats = BatchStats::default();
        assert!((stats.average_time_per_view_ms() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_average_time_per_view() {
        let stats = BatchStats {
            total_views_generated: 4,
            total_time_ms: 200.0,
            ..Default::default()
        };
        assert!((stats.average_time_per_view_ms() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_cache_hit_rate() {
        let stats = BatchStats {
            cache_hits: 3,
            cache_misses: 1,
            ..Default::default()
        };
        assert!((stats.cache_hit_rate() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_reset_stats_clears_counters() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        gen.reset_stats();
        let stats = gen.stats();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.total_views_generated, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
    }

    // -----------------------------------------------------------------------
    // process_batch
    // -----------------------------------------------------------------------

    #[test]
    fn test_process_batch_empty_queue_returns_empty_vec() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        let results = gen.process_batch().expect("process_batch failed");
        assert!(results.is_empty());
    }

    #[test]
    fn test_process_batch_without_weights_returns_err() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        gen.queue(make_request("q1", 2)).expect("queue failed");
        assert!(gen.process_batch().is_err());
    }

    #[test]
    fn test_process_batch_drains_queue_even_on_failure() {
        let gen = BatchGenerator::new(BatchGenConfig::default());
        gen.queue(make_request("q1", 1)).expect("queue failed");
        gen.queue(make_request("q2", 1)).expect("queue failed");
        let _ = gen.process_batch();
        assert_eq!(gen.queue_len(), 0);
    }

    // -----------------------------------------------------------------------
    // GenerationResult helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_views_generated() {
        let empty = GenerationResult {
            id: "x".into(),
            views: Vec::new(),
            total_time_ms: 1.0,
            num_cached_kv: 0,
        };
        assert!(!empty.all_views_generated());

        let populated = GenerationResult {
            id: "y".into(),
            views: vec![GeneratedView {
                view_index: 0,
                image_data: vec![0u8; 12],
                width: 2,
                height: 2,
                generation_time_ms: 1.0,
            }],
            total_time_ms: 1.0,
            num_cached_kv: 0,
        };
        assert!(populated.all_views_generated());
    }

    #[test]
    fn test_throughput_zero_when_time_zero() {
        let result = GenerationResult {
            id: "x".into(),
            views: vec![],
            total_time_ms: 0.0,
            num_cached_kv: 0,
        };
        assert!((result.throughput_views_per_sec() - 0.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // orbit_camera_poses
    // -----------------------------------------------------------------------

    #[test]
    fn test_orbit_camera_poses_shape() {
        let poses = orbit_camera_poses(4, 12);
        assert_eq!(poses.len(), 48);
        assert!(poses.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_orbit_camera_poses_rotation_is_orthonormal() {
        let poses = orbit_camera_poses(4, 12);
        for view in 0..4 {
            let base = view * 12;
            let rows = [
                [poses[base], poses[base + 1], poses[base + 2]],
                [poses[base + 4], poses[base + 5], poses[base + 6]],
                [poses[base + 8], poses[base + 9], poses[base + 10]],
            ];
            for (i, row) in rows.iter().enumerate() {
                let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
                assert!((norm - 1.0).abs() < 1e-5, "row {i} norm {norm}");
            }
            // Rows must be mutually orthogonal.
            for (i, a) in rows.iter().enumerate() {
                for b in rows.iter().skip(i + 1) {
                    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                    assert!(dot.abs() < 1e-5, "rows not orthogonal: {dot}");
                }
            }
        }
    }

    #[test]
    fn test_orbit_camera_poses_look_at_origin() {
        // t = -R · C, and the cameras look at the origin from ORBIT_RADIUS away,
        // so the forward-row translation is exactly the orbit radius.
        let poses = orbit_camera_poses(3, 12);
        for view in 0..3 {
            let base = view * 12;
            assert!(poses[base + 3].abs() < 1e-4, "t_x should be 0");
            assert!(poses[base + 7].abs() < 1e-4, "t_y should be 0");
            assert!(
                (poses[base + 11] - ORBIT_RADIUS).abs() < 1e-4,
                "t_z should be the orbit radius, got {}",
                poses[base + 11]
            );
        }
    }

    #[test]
    fn test_orbit_camera_poses_pads_and_truncates_pose_dim() {
        let padded = orbit_camera_poses(1, 16);
        assert_eq!(padded.len(), 16);
        assert!(padded[12..].iter().all(|v| v.abs() < f32::EPSILON));

        let truncated = orbit_camera_poses(1, 6);
        assert_eq!(truncated.len(), 6);

        assert!(orbit_camera_poses(0, 12).is_empty());
        assert!(orbit_camera_poses(4, 0).is_empty());
    }

    // -----------------------------------------------------------------------
    // tensor_to_rgb8
    // -----------------------------------------------------------------------

    #[test]
    fn test_tensor_to_rgb8_interleaves_channels() {
        // (3, 1, 2): R plane = [0.0, 1.0], G plane = [0.5, 0.5], B plane = [1.0, 0.0]
        let data = vec![0.0f32, 1.0, 0.5, 0.5, 1.0, 0.0];
        let tensor =
            Tensor::from_vec(data, (3usize, 1usize, 2usize), &Device::Cpu).expect("tensor");
        let rgb = tensor_to_rgb8(&tensor).expect("conversion failed");
        assert_eq!(rgb.len(), 6);
        assert_eq!(rgb[0], 0);
        assert_eq!(rgb[1], 128);
        assert_eq!(rgb[2], 255);
        assert_eq!(rgb[3], 255);
        assert_eq!(rgb[4], 128);
        assert_eq!(rgb[5], 0);
    }

    #[test]
    fn test_tensor_to_rgb8_clamps_out_of_range_values() {
        let data = vec![-1.0f32, 2.0, 0.5];
        let tensor =
            Tensor::from_vec(data, (3usize, 1usize, 1usize), &Device::Cpu).expect("tensor");
        let rgb = tensor_to_rgb8(&tensor).expect("conversion failed");
        assert_eq!(rgb, vec![0u8, 255, 128]);
    }

    #[test]
    fn test_tensor_to_rgb8_single_channel_replicates() {
        let tensor = Tensor::from_vec(vec![1.0f32, 0.0], (1usize, 1usize, 2usize), &Device::Cpu)
            .expect("tensor");
        let rgb = tensor_to_rgb8(&tensor).expect("conversion failed");
        assert_eq!(rgb, vec![255u8, 255, 255, 0, 0, 0]);
    }
}
