//! Streaming inference API for multi-view diffusion.
//!
//! Drives a [`MultiViewDiffusionPipeline`] one denoising step at a time and
//! yields the partially denoised views as they become available, which is what
//! progressive UIs and latency-sensitive pipelines need.
//!
//! # Attaching a model
//!
//! Real pixels require real weights. Build the engine with
//! [`StreamingInference::load`] (or [`StreamingInference::with_pipeline`] when
//! you already hold a pipeline) and every yielded [`StreamingStep`] carries a
//! VAE-decoded RGB frame of the current latents:
//!
//! ```no_run
//! use candle_core::{DType, Device, Tensor};
//! use oxigaf_diffusion::streaming::{StreamingConfig, StreamingInference, StreamingInputs};
//! use oxigaf_diffusion::DiffusionConfig;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let device = Device::Cpu;
//! let diffusion_config = DiffusionConfig::default();
//! let inputs = StreamingInputs {
//!     reference_image: Tensor::zeros((1, 3, 224, 224), DType::F32, &device)?,
//!     normal_map_latents: Tensor::zeros((4, 4, 32, 32), DType::F32, &device)?,
//!     camera_poses: Tensor::zeros((4, 12), DType::F32, &device)?,
//!     seed: 42,
//! };
//! let si = StreamingInference::load(
//!     StreamingConfig::default(),
//!     diffusion_config,
//!     std::path::Path::new("weights/"),
//!     &device,
//!     &inputs,
//! )?;
//! for step in si.step_iter(4) {
//!     // `partial_image` is a decoded RGB frame of the current latents.
//!     let _bytes = step.partial_image.len();
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Without a model
//!
//! [`StreamingInference::new`] builds a **schedule-only** engine: the step
//! bookkeeping (view index, step index, progress, `is_final`) is real, but no
//! model is attached, so `partial_image` is left **empty** rather than filled
//! with fabricated pixels, and a warning is logged when the iterator is
//! created.
//!
//! ```rust
//! use oxigaf_diffusion::streaming::{StreamingInference, StreamingConfig};
//!
//! let config = StreamingConfig::default();
//! let si = StreamingInference::new(config);
//!
//! let steps: Vec<_> = si.step_iter(2).collect();
//! // 2 views × num_steps steps each
//! assert_eq!(steps.len(), 2 * StreamingConfig::default().num_steps);
//! ```
//!
//! # Ordering
//!
//! The U-Net denoises every view jointly, so steps are yielded **step-major**:
//! for each denoising step, one [`StreamingStep`] per view.
//!
//! # Cost
//!
//! Every denoising step VAE-decodes the current latents to build the preview
//! frames, which is the price of progressive output. Keep the pipeline's
//! `offload_strategy` at `AllInMemory` while streaming: the other strategies
//! re-load the decoder from disk for each preview.

use std::path::Path;
use std::sync::Mutex;

use candle_core::{Device, Tensor};

use crate::config::DiffusionConfig;
use crate::pipeline::{GenerationSession, MultiViewDiffusionPipeline};
use crate::DiffusionError;

/// Number of channels in the emitted RGB frames.
const RGB_CHANNELS: usize = 3;

// ---------------------------------------------------------------------------
// StreamingConfig
// ---------------------------------------------------------------------------

/// Configuration for [`StreamingInference`].
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Output image width in pixels.
    ///
    /// When a pipeline is attached this is re-synchronised with the pipeline's
    /// pre-upsampling image size, so it always describes the frames actually
    /// emitted.
    ///
    /// Default: `256`.
    pub image_width: u32,

    /// Output image height in pixels.
    ///
    /// When a pipeline is attached this is re-synchronised with the pipeline's
    /// pre-upsampling image size, so it always describes the frames actually
    /// emitted.
    ///
    /// Default: `256`.
    pub image_height: u32,

    /// Classifier-free guidance scale.
    ///
    /// Applied to the attached pipeline when the engine is built.
    ///
    /// Default: `3.0`.
    pub guidance_scale: f32,

    /// Number of denoising steps per view.
    ///
    /// Default: `20`.
    pub num_steps: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            image_width: 256,
            image_height: 256,
            guidance_scale: 3.0,
            num_steps: 20,
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingInputs
// ---------------------------------------------------------------------------

/// Conditioning tensors required to drive a real streaming run.
///
/// Mirrors the arguments of [`MultiViewDiffusionPipeline::generate`].
#[derive(Debug, Clone)]
pub struct StreamingInputs {
    /// `(1, 3, 224, 224)` normalised reference image for CLIP.
    pub reference_image: Tensor,
    /// `(num_views, latent_channels, h, w)` encoded normal maps.
    pub normal_map_latents: Tensor,
    /// `(num_views, pose_dim)` flattened extrinsics per view.
    pub camera_poses: Tensor,
    /// Seed for the initial latents (reproducible across runs).
    pub seed: u64,
}

// ---------------------------------------------------------------------------
// StreamingStep
// ---------------------------------------------------------------------------

/// One step of streaming output from [`StreamingIterator`].
#[derive(Debug, Clone)]
pub struct StreamingStep {
    /// Zero-based view index this step belongs to.
    pub view_index: usize,

    /// Zero-based denoising step index within the current view.
    pub step_index: usize,

    /// Total number of denoising steps per view.
    pub total_steps: usize,

    /// Partial denoising result at this step (may be noisy).
    ///
    /// RGB bytes, flat row-major, length = `width × height × 3`.
    ///
    /// **Empty** when the engine was built without a model
    /// ([`StreamingInference::new`]): no weights, no pixels.
    pub partial_image: Vec<u8>,

    /// `true` when this is the last step for the current view.
    pub is_final: bool,
}

impl StreamingStep {
    /// Fraction of denoising complete for the current view, in `[0.0, 1.0)`.
    ///
    /// The last step has `progress_fraction = (num_steps - 1) / num_steps`,
    /// not `1.0`, because `is_final` signals completion instead.
    ///
    /// Returns `0.0` when `total_steps == 0` (degenerate configuration).
    pub fn progress_fraction(&self) -> f32 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.step_index as f32 / self.total_steps as f32
    }
}

// ---------------------------------------------------------------------------
// StreamingInference
// ---------------------------------------------------------------------------

/// A pipeline plus the in-flight generation session it is driving.
struct StreamingModel {
    pipeline: MultiViewDiffusionPipeline,
    session: GenerationSession,
}

/// Streaming multi-view inference engine.
///
/// Produces an [`Iterator`] over [`StreamingStep`] values so callers can
/// process partial results as each denoising step completes.
pub struct StreamingInference {
    config: StreamingConfig,
    /// `None` for a schedule-only engine. The mutex lets `step_iter(&self)`
    /// drive the model without forcing callers to hold `&mut`.
    model: Option<Mutex<StreamingModel>>,
    /// Seed of the attached run (`0` when no model is attached).
    seed: u64,
}

impl StreamingInference {
    /// Create a **schedule-only** engine with no model attached.
    ///
    /// Step bookkeeping is real; `StreamingStep::partial_image` stays empty
    /// because decoding pixels requires trained weights. Use
    /// [`Self::load`] or [`Self::with_pipeline`] for real frames.
    pub fn new(config: StreamingConfig) -> Self {
        Self {
            config,
            model: None,
            seed: 0,
        }
    }

    /// Attach an already-loaded pipeline and start a generation session.
    ///
    /// `config.guidance_scale` is applied to the pipeline and
    /// `config.num_steps` becomes the session's DDIM step count.
    /// `config.image_width` / `image_height` are re-synchronised with the
    /// pipeline's pre-upsampling image size so they describe the frames that
    /// are actually emitted.
    ///
    /// # Errors
    ///
    /// - `DiffusionError::InvalidConfig` when `config.num_steps == 0`.
    /// - Anything [`MultiViewDiffusionPipeline::begin_session_with_steps`]
    ///   reports (invalid guidance scale, CLIP load or encode failure).
    pub fn with_pipeline(
        config: StreamingConfig,
        pipeline: MultiViewDiffusionPipeline,
        inputs: &StreamingInputs,
    ) -> Result<Self, DiffusionError> {
        let mut config = config;
        let mut pipeline = pipeline;

        if config.num_steps == 0 {
            return Err(DiffusionError::InvalidConfig(
                "StreamingConfig::num_steps must be >= 1".to_string(),
            ));
        }

        pipeline.set_guidance_scale(config.guidance_scale as f64);

        // Frames are decoded from the pre-upsampling latents, so the emitted
        // resolution is the pipeline's base image size.
        let frame_size = pipeline.config().image_size as u32;
        if config.image_width != frame_size || config.image_height != frame_size {
            tracing::warn!(
                "StreamingConfig requests {}×{} frames but the pipeline decodes {}×{}; using the pipeline size",
                config.image_width,
                config.image_height,
                frame_size,
                frame_size
            );
            config.image_width = frame_size;
            config.image_height = frame_size;
        }

        let session = pipeline.begin_session_with_steps(
            &inputs.reference_image,
            &inputs.normal_map_latents,
            &inputs.camera_poses,
            inputs.seed,
            config.num_steps,
        )?;

        Ok(Self {
            config,
            model: Some(Mutex::new(StreamingModel { pipeline, session })),
            seed: inputs.seed,
        })
    }

    /// Load a pipeline from `weights_dir` and start a streaming session.
    ///
    /// # Errors
    ///
    /// Anything [`MultiViewDiffusionPipeline::load`] or [`Self::with_pipeline`]
    /// reports.
    pub fn load(
        config: StreamingConfig,
        diffusion_config: DiffusionConfig,
        weights_dir: &Path,
        device: &Device,
        inputs: &StreamingInputs,
    ) -> Result<Self, DiffusionError> {
        let pipeline = MultiViewDiffusionPipeline::load(diffusion_config, weights_dir, device)?;
        Self::with_pipeline(config, pipeline, inputs)
    }

    /// The configuration in force (possibly adjusted to the attached pipeline).
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// `true` when a real model drives the produced frames.
    pub fn has_model(&self) -> bool {
        self.model.is_some()
    }

    /// Seed of the attached run; `0` for a schedule-only engine.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Number of views the attached session denoises jointly, if any.
    pub fn model_num_views(&self) -> Option<usize> {
        let model = self.model.as_ref()?;
        let guard = model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Some(guard.session.num_views())
    }

    /// Build an iterator that yields one [`StreamingStep`] per view per
    /// denoising step.
    ///
    /// Total steps yielded = `num_views × config.num_steps`. When a model is
    /// attached, `num_views` is capped at the session's view count (the U-Net
    /// denoises a fixed number of views jointly).
    ///
    /// The generation session is shared: iterators returned by repeated calls
    /// continue the *same* run rather than restarting it, and one opened after
    /// the run finished ends immediately.
    pub fn step_iter(&self, num_views: usize) -> StreamingIterator<'_> {
        let effective_views = match self.model_num_views() {
            Some(session_views) if num_views > session_views => {
                tracing::warn!(
                    "requested {} views but the pipeline generates {}; capping",
                    num_views,
                    session_views
                );
                session_views
            }
            _ => num_views,
        };

        if self.model.is_none() {
            tracing::warn!(
                "StreamingInference has no pipeline attached: steps will carry empty \
                 partial_image buffers (use StreamingInference::load for real frames)"
            );
        }

        StreamingIterator {
            config: self.config.clone(),
            model: self.model.as_ref(),
            num_views: effective_views,
            current_view: 0,
            current_step: 0,
            frames: Vec::new(),
            finished: false,
            seed: self.seed,
        }
    }
}

// ---------------------------------------------------------------------------
// StreamingIterator
// ---------------------------------------------------------------------------

/// Iterator over streaming denoising steps for multi-view generation.
///
/// Produced by [`StreamingInference::step_iter`]. Steps are yielded
/// step-major: all views of denoising step 0, then all views of step 1, …
pub struct StreamingIterator<'a> {
    config: StreamingConfig,
    model: Option<&'a Mutex<StreamingModel>>,
    num_views: usize,
    current_view: usize,
    current_step: usize,
    /// Decoded RGB frames for the denoising step currently being emitted.
    frames: Vec<Vec<u8>>,
    /// Set when the stream must stop early (model failure).
    finished: bool,
    /// Seed of the run this iterator streams.
    seed: u64,
}

impl StreamingIterator<'_> {
    /// Seed of the run this iterator streams.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Advance the attached model by one denoising step and decode a preview
    /// frame per view.
    ///
    /// Returns `Ok(None)` when the underlying session has no step left (for
    /// example when a second iterator is opened on a finished run), and an
    /// empty vector when no model is attached at all.
    fn advance_model(&mut self) -> Result<Option<Vec<Vec<u8>>>, DiffusionError> {
        let model = match self.model {
            Some(model) => model,
            None => return Ok(Some(Vec::new())),
        };

        let mut guard = model
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let StreamingModel { pipeline, session } = &mut *guard;

        if !pipeline.step_session(session)? {
            // The session is exhausted: stop rather than re-emitting the last
            // frame over and over.
            return Ok(None);
        }
        let views = pipeline.preview_images(session)?;

        let mut frames = Vec::with_capacity(views.len());
        for view in &views {
            frames.push(tensor_to_rgb8(view)?);
        }
        Ok(Some(frames))
    }
}

impl Iterator for StreamingIterator<'_> {
    type Item = StreamingStep;

    fn next(&mut self) -> Option<Self::Item> {
        // Degenerate configurations and early termination.
        if self.finished || self.num_views == 0 || self.config.num_steps == 0 {
            return None;
        }
        if self.current_step >= self.config.num_steps {
            return None;
        }

        // The U-Net denoises all views jointly, so the model advances once per
        // denoising step — right before the first view of that step is emitted.
        if self.current_view == 0 {
            match self.advance_model() {
                Ok(Some(frames)) => self.frames = frames,
                Ok(None) => {
                    // Denoising already complete for the shared session.
                    self.finished = true;
                    return None;
                }
                Err(e) => {
                    tracing::error!("streaming denoising step failed: {e}");
                    self.finished = true;
                    return None;
                }
            }
        }

        let step_index = self.current_step;
        let view_index = self.current_view;
        let is_final = step_index + 1 == self.config.num_steps;
        let partial_image = self.frames.get(view_index).cloned().unwrap_or_default();

        // Advance internal state (step-major).
        self.current_view += 1;
        if self.current_view >= self.num_views {
            self.current_view = 0;
            self.current_step += 1;
        }

        Some(StreamingStep {
            view_index,
            step_index,
            total_steps: self.config.num_steps,
            partial_image,
            is_final,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.finished {
            return (0, Some(0));
        }
        let emitted = self.current_step * self.num_views + self.current_view;
        let total = self.num_views * self.config.num_steps;
        let remaining = total.saturating_sub(emitted);
        if self.model.is_some() {
            // A denoising step can fail or the shared session can run out, so
            // only the upper bound is guaranteed.
            (0, Some(remaining))
        } else {
            (remaining, Some(remaining))
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor → RGB8
// ---------------------------------------------------------------------------

/// Convert a `(C, H, W)` tensor in `[0, 1]` into interleaved RGB bytes.
///
/// Missing channels are replicated from the last available one, so single
/// channel previews come out as grey rather than failing.
fn tensor_to_rgb8(image: &Tensor) -> Result<Vec<u8>, DiffusionError> {
    let (channels, height, width) = image
        .dims3()
        .map_err(|e| DiffusionError::Inference(format!("streaming frame dims: {e}")))?;

    let data = image
        .flatten_all()
        .and_then(|flat| flat.to_vec1::<f32>())
        .map_err(|e| DiffusionError::Inference(format!("streaming frame readback: {e}")))?;

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
    use candle_core::DType;

    // -----------------------------------------------------------------------
    // StreamingConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_streaming_config_default_image_width() {
        assert_eq!(StreamingConfig::default().image_width, 256);
    }

    #[test]
    fn test_streaming_config_default_image_height() {
        assert_eq!(StreamingConfig::default().image_height, 256);
    }

    #[test]
    fn test_streaming_config_default_guidance_scale() {
        let cfg = StreamingConfig::default();
        assert!((cfg.guidance_scale - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_streaming_config_default_num_steps() {
        assert_eq!(StreamingConfig::default().num_steps, 20);
    }

    // -----------------------------------------------------------------------
    // StreamingIterator step count
    // -----------------------------------------------------------------------

    #[test]
    fn test_step_iter_total_count_equals_views_times_steps() {
        let config = StreamingConfig::default();
        let si = StreamingInference::new(config.clone());
        let count = si.step_iter(2).count();
        assert_eq!(count, 2 * config.num_steps);
    }

    #[test]
    fn test_step_iter_zero_views_yields_nothing() {
        let si = StreamingInference::new(StreamingConfig::default());
        assert_eq!(si.step_iter(0).count(), 0);
    }

    #[test]
    fn test_step_iter_one_view_yields_num_steps_items() {
        let config = StreamingConfig {
            num_steps: 5,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        assert_eq!(si.step_iter(1).count(), 5);
    }

    /// Regression: a zero-step configuration used to increment `current_view`
    /// and return `None`, killing the whole iteration instead of yielding an
    /// empty stream.
    #[test]
    fn test_step_iter_zero_steps_yields_nothing_and_terminates() {
        let config = StreamingConfig {
            num_steps: 0,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        let mut iter = si.step_iter(3);
        assert!(iter.next().is_none());
        // Still terminated on a second poll (no state left half-advanced).
        assert!(iter.next().is_none());
        assert_eq!(si.step_iter(3).count(), 0);
    }

    // -----------------------------------------------------------------------
    // Ordering: step-major (all views share a denoising step)
    // -----------------------------------------------------------------------

    #[test]
    fn test_steps_are_emitted_step_major() {
        let config = StreamingConfig {
            num_steps: 3,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        let steps: Vec<_> = si.step_iter(2).collect();
        assert_eq!(steps.len(), 6);

        let observed: Vec<(usize, usize)> =
            steps.iter().map(|s| (s.step_index, s.view_index)).collect();
        assert_eq!(
            observed,
            vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        );
    }

    #[test]
    fn test_every_view_appears_once_per_step() {
        let config = StreamingConfig {
            num_steps: 4,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        let steps: Vec<_> = si.step_iter(3).collect();
        for step_index in 0..4 {
            let mut views: Vec<usize> = steps
                .iter()
                .filter(|s| s.step_index == step_index)
                .map(|s| s.view_index)
                .collect();
            views.sort_unstable();
            assert_eq!(views, vec![0, 1, 2]);
        }
    }

    // -----------------------------------------------------------------------
    // First and last step properties
    // -----------------------------------------------------------------------

    #[test]
    fn test_final_flag_only_on_last_denoising_step() {
        let config = StreamingConfig {
            num_steps: 4,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        let steps: Vec<_> = si.step_iter(3).collect();

        for step in &steps {
            assert_eq!(
                step.is_final,
                step.step_index == 3,
                "is_final must mark the last denoising step only"
            );
        }
        assert_eq!(steps.iter().filter(|s| s.is_final).count(), 3);
    }

    // -----------------------------------------------------------------------
    // progress_fraction
    // -----------------------------------------------------------------------

    #[test]
    fn test_progress_fraction_first_step_is_zero() {
        let config = StreamingConfig {
            num_steps: 10,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        let first = si
            .step_iter(1)
            .next()
            .expect("should have at least one step");
        assert!(
            (first.progress_fraction() - 0.0).abs() < f32::EPSILON,
            "step_index=0 should give progress_fraction=0.0"
        );
    }

    #[test]
    fn test_progress_fraction_last_step_is_n_minus_1_over_n() {
        let num_steps = 10;
        let config = StreamingConfig {
            num_steps,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        let last = si
            .step_iter(1)
            .last()
            .expect("should have at least one step");
        let expected = (num_steps - 1) as f32 / num_steps as f32;
        assert!(
            (last.progress_fraction() - expected).abs() < f32::EPSILON,
            "last step progress_fraction should be {expected}, got {}",
            last.progress_fraction()
        );
    }

    #[test]
    fn test_progress_fraction_zero_total_steps_returns_zero() {
        let step = StreamingStep {
            view_index: 0,
            step_index: 0,
            total_steps: 0,
            partial_image: vec![],
            is_final: true,
        };
        assert!((step.progress_fraction() - 0.0).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Honest empty frames without a model
    // -----------------------------------------------------------------------

    /// Regression: the iterator used to fabricate a grey ramp
    /// (`vec![255 * step / num_steps; w * h * 3]`) that looked like model
    /// output. Without weights there is nothing to decode, so the buffer must
    /// be empty instead.
    #[test]
    fn test_partial_image_is_empty_without_a_model() {
        let config = StreamingConfig {
            num_steps: 5,
            image_width: 8,
            image_height: 8,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        assert!(!si.has_model());
        for step in si.step_iter(2) {
            assert!(
                step.partial_image.is_empty(),
                "no model attached ⇒ no fabricated pixels"
            );
        }
    }

    #[test]
    fn test_seed_is_readable() {
        let si = StreamingInference::new(StreamingConfig::default());
        assert_eq!(si.seed(), 0);
        assert_eq!(si.step_iter(1).seed(), 0);
    }

    #[test]
    fn test_model_num_views_is_none_without_a_model() {
        let si = StreamingInference::new(StreamingConfig::default());
        assert_eq!(si.model_num_views(), None);
    }

    // -----------------------------------------------------------------------
    // total_steps / size_hint
    // -----------------------------------------------------------------------

    #[test]
    fn test_total_steps_field_matches_config() {
        let config = StreamingConfig {
            num_steps: 7,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        for step in si.step_iter(1) {
            assert_eq!(step.total_steps, 7);
        }
    }

    #[test]
    fn test_size_hint_counts_remaining_steps() {
        let config = StreamingConfig {
            num_steps: 3,
            ..Default::default()
        };
        let si = StreamingInference::new(config);
        let mut iter = si.step_iter(2);
        // Exact for a schedule-only engine: nothing can fail mid-stream.
        assert_eq!(iter.size_hint(), (6, Some(6)));
        let _ = iter.next();
        assert_eq!(iter.size_hint(), (5, Some(5)));
        let remaining = iter.count();
        assert_eq!(remaining, 5);
    }

    // -----------------------------------------------------------------------
    // Frame conversion
    // -----------------------------------------------------------------------

    #[test]
    fn test_tensor_to_rgb8_maps_unit_range_to_bytes() -> Result<(), DiffusionError> {
        let device = Device::Cpu;
        // 3 channels, 1×2 image: R=0, G=0.5, B=1 for both pixels.
        let data = vec![0.0f32, 0.0, 0.5, 0.5, 1.0, 1.0];
        let image = Tensor::from_vec(data, (3, 1, 2), &device)
            .map_err(|e| DiffusionError::Inference(format!("{e}")))?;
        let rgb = tensor_to_rgb8(&image)?;
        assert_eq!(rgb.len(), 2 * RGB_CHANNELS);
        assert_eq!(rgb[0], 0);
        assert_eq!(rgb[1], 128);
        assert_eq!(rgb[2], 255);
        Ok(())
    }

    #[test]
    fn test_tensor_to_rgb8_replicates_single_channel() -> Result<(), DiffusionError> {
        let device = Device::Cpu;
        let image = Tensor::zeros((1, 2, 2), DType::F32, &device)
            .map_err(|e| DiffusionError::Inference(format!("{e}")))?;
        let rgb = tensor_to_rgb8(&image)?;
        assert_eq!(rgb.len(), 2 * 2 * RGB_CHANNELS);
        assert!(rgb.iter().all(|&b| b == 0));
        Ok(())
    }

    #[test]
    fn test_tensor_to_rgb8_rejects_non_chw() -> Result<(), DiffusionError> {
        let device = Device::Cpu;
        let image = Tensor::zeros((2, 3, 4, 4), DType::F32, &device)
            .map_err(|e| DiffusionError::Inference(format!("{e}")))?;
        assert!(
            tensor_to_rgb8(&image).is_err(),
            "4-D input must be rejected"
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Model attachment errors
    // -----------------------------------------------------------------------

    #[test]
    fn test_load_without_weights_reports_an_error() -> Result<(), DiffusionError> {
        let device = Device::Cpu;
        let inputs = StreamingInputs {
            reference_image: Tensor::zeros((1, 3, 224, 224), DType::F32, &device)
                .map_err(|e| DiffusionError::Inference(format!("{e}")))?,
            normal_map_latents: Tensor::zeros((4, 4, 32, 32), DType::F32, &device)
                .map_err(|e| DiffusionError::Inference(format!("{e}")))?,
            camera_poses: Tensor::zeros((4, 12), DType::F32, &device)
                .map_err(|e| DiffusionError::Inference(format!("{e}")))?,
            seed: 7,
        };
        let config = StreamingConfig {
            num_steps: 0,
            ..Default::default()
        };
        // No weights on disk, so no model can be attached: the engine must
        // report the failure instead of silently falling back to fake frames.
        let missing = std::env::temp_dir().join("oxigaf_streaming_missing_weights");
        let result = StreamingInference::load(
            config,
            DiffusionConfig::default(),
            &missing,
            &device,
            &inputs,
        );
        assert!(result.is_err());
        Ok(())
    }
}
