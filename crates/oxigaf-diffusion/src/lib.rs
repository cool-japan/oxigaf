//! # oxigaf-diffusion
//!
//! Multi-view diffusion model inference for GAF.
//!
//! Implements the full pipeline: CLIP image encoding → multi-view U-Net
//! denoising with camera-conditioned cross-view attention → VAE decoding.
//!
//! ## Cargo Features
//!
//! This crate supports the following feature flags (see `Cargo.toml`'s
//! `[features]` section, which this list mirrors):
//!
//! - **`default`** = `[]`:
//!   No optional features on by default, keeping a plain `cargo build`
//!   minimal (CPU-only, no flash attention).
//!
//! - **`flash_attention`** (off by default):
//!   Memory-efficient attention with O(N) complexity instead of O(N²)
//!   - Reduces memory usage by 2-4× for large images
//!   - Tiled computation for better cache locality
//!   - Gates the `flash_attention` module and its re-exports (`FlashAttention`,
//!     `flash_attention`, `flash_attention_with_config`, ...); `unet.rs`
//!     constructs a `FlashAttention` when `DiffusionConfig::use_flash_attention`
//!     is set.
//!
//! - **`mixed_precision`**:
//!   The [`mixed_precision`] module (FP32 ↔ BF16/FP16 conversion and
//!   simulation utilities) is unconditionally compiled either way; this flag
//!   only changes the *default* value of `MixedPrecisionConfig::mode` (to
//!   `BFloat16` instead of `Float32`) — see the module's own docs for
//!   details on what is (and, as of this writing, is not) wired into the
//!   inference path.
//!
//! - **`gpu_debug`**:
//!   Enables NaN/Inf debug hooks (`DebugConfig::default()` sets `enabled =
//!   true` when this feature is active).
//!
//! Platform-specific GPU/BLAS backends (`accelerate`, `metal`, `cuda`) are
//! **not** feature flags of this crate — support for them was removed. To
//! enable one, configure the `candle-core`/`candle-nn` dependency directly in
//! your own `Cargo.toml`, e.g. via the COOLJAPAN `oxicandle-core` fork:
//! ```toml
//! candle-core = { package = "oxicandle-core", version = "0.11.0", features = ["accelerate"] }  # macOS BLAS
//! candle-core = { package = "oxicandle-core", version = "0.11.0", features = ["metal"] }       # macOS GPU
//! candle-core = { package = "oxicandle-core", version = "0.11.0", features = ["cuda"] }        # NVIDIA GPU
//! ```
//!
//! Example usage:
//! ```toml
//! # In Cargo.toml
//! # CPU-only, minimal build (the default)
//! oxigaf-diffusion = { version = "0.1" }
//!
//! # With flash attention
//! oxigaf-diffusion = { version = "0.1", features = ["flash_attention"] }
//! ```

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
// This crate carries no `allow(clippy::...)` attributes, crate-level or local.
// It used to blanket-allow `field_reassign_with_default`, `manual_range_contains`
// and `identity_op` on the theory that hundreds of test blocks depended on
// them; when the suppressions were finally lifted the whole crate produced
// nine hits, all mechanical. Keep it that way: fix the lint rather than
// re-adding a suppression.

pub mod adaptive_sampling;
pub mod attention;
pub mod attention_masking;
pub mod attention_viz;
pub mod avatar_conditioning;
pub mod batch_gen;
pub mod camera;
pub mod cfg_guidance;
pub mod classifier_guidance;
pub mod clip;
pub mod config;
pub mod consistency_model;
pub mod controlnet;
pub mod cross_frame_consistency;
pub mod ddpm_sampler;
pub mod debug_hooks;
pub mod denoising_trajectory;
pub mod denoising_viz;
pub mod distillation_loss;
pub mod dynamic_views;
#[cfg(feature = "flash_attention")]
pub mod flash_attention;
pub mod flow_matching;
pub mod fused_attention;
pub mod guidance_rescaling;
pub mod identity_conditioning;
pub mod image_editing;
pub mod image_preprocessing;
pub mod image_variations;
pub mod inversion;
pub mod kv_cache;
pub mod latent_blend;
pub mod latent_interp;
pub mod latent_space_analysis;
pub mod latent_walk;
pub mod lora_adapter;
pub mod mixed_precision;
pub mod model_variants;
pub mod multi_step_sampler;
pub mod multi_view_consistency;
pub mod noise_prediction_analysis;
pub mod noise_schedule_analysis;
pub mod numerics;
pub mod pipeline;
pub mod prior_preservation;
pub mod profiling;
pub mod prompt_scheduler;
pub mod prompt_weighting;
pub mod quantization;
pub mod rectified_flow;
pub mod resolution;
pub mod scheduler;
pub mod score_distillation;
pub mod score_matching;
pub mod sequential_vae;
pub mod sliced_attention;
pub mod step_scheduler;
pub mod stochastic_interpolant;
pub mod streaming;
pub mod style_transfer;
pub mod text_conditioning;
pub mod token_merging;
pub mod uncond_dropout;
pub mod unet;
pub mod upsampler;
pub mod vae;
pub mod weight_offload;
pub use rectified_flow::{
    rf_compute_stats, rf_euler_integrate, rf_euler_step, rf_flow_matching_loss, rf_format_config,
    rf_format_stats, rf_greedy_ot_match, rf_heun_step, rf_interpolate, rf_linspace,
    rf_logit_normal_t, rf_loss_per_sample, rf_make_batch, rf_midpoint_step, rf_reflow_pair,
    rf_rk4_step, rf_sample_t, rf_straight_path_length, rf_straightness_ratio, rf_target_velocity,
    rf_trajectory_curvature, rf_trajectory_length, rf_weighted_loss, CouplingMode,
    RectifiedFlowConfig, RectifiedFlowError, RfBatch, RfCoupling, RfOdeSolver, RfSolverKind,
    RfStats, RfTrajectory,
};
pub use stochastic_interpolant::{
    si_check_boundary_conditions, si_compute_stats, si_euler_step_ode, si_euler_step_sde,
    si_eval_coefficients, si_format_config, si_format_stats, si_interpolate, si_kind_name,
    si_linspace, si_make_batch, si_ode_integrate, si_score_loss, si_velocity, si_velocity_loss,
    si_velocity_loss_per_sample, InterpolantConfig, InterpolantKind, SiBatch, SiCoefficients,
    SiDrift, SiStats, StochasticInterpolantError,
};

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during diffusion model operations.
#[derive(Debug, Error)]
pub enum DiffusionError {
    // -------------------------------------------------------------------------
    // Model Loading Errors
    // -------------------------------------------------------------------------
    /// Generic model loading error with context message.
    #[error("Model loading error: {0}")]
    ModelLoad(String),

    /// Weight file not found at expected path.
    #[error("Weight not found: layer '{layer}', expected shape {expected_shape:?}")]
    WeightNotFound {
        layer: String,
        expected_shape: Vec<usize>,
    },

    /// Weight shape does not match expected dimensions.
    #[error("Weight shape mismatch: layer '{layer}', expected {expected:?}, got {got:?}")]
    WeightShapeMismatch {
        layer: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Safetensors file is corrupted or invalid.
    #[error("Safetensors corrupt: {path:?}, reason: {reason}")]
    SafetensorsCorrupt { path: PathBuf, reason: String },

    // -------------------------------------------------------------------------
    // Tensor Operation Errors
    // -------------------------------------------------------------------------
    /// Tensor shape mismatch during operation.
    #[error("Shape mismatch in '{op}': expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        op: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Data type mismatch between tensors.
    #[error("Dtype mismatch: expected {expected}, got {got}")]
    DtypeMismatch { expected: String, got: String },

    /// Device mismatch between tensors.
    #[error("Device mismatch: expected {expected}, got {got}")]
    DeviceMismatch { expected: String, got: String },

    // -------------------------------------------------------------------------
    // Numerical Errors
    // -------------------------------------------------------------------------
    /// Error from the numerics module (e.g., empty input, length mismatch).
    #[error("Numerics error: {0}")]
    Numerics(#[from] numerics::NumericsError),

    /// NaN detected in tensor during computation.
    #[error("NaN detected in layer '{layer}' at timestep {timestep:?}")]
    NanDetected {
        layer: String,
        timestep: Option<usize>,
    },

    /// Infinity detected in tensor during computation.
    #[error("Inf detected in layer '{layer}' at timestep {timestep:?}")]
    InfDetected {
        layer: String,
        timestep: Option<usize>,
    },

    /// NaN or Inf values detected in a named tensor slice (from debug hooks).
    ///
    /// Produced by [`debug_hooks::assert_finite`] when the tensor contains any
    /// non-finite values.
    #[error(
        "NaN/Inf detected in '{name}': {nan_count} NaN, {inf_count} Inf (first at index {first_index})"
    )]
    NanInfDetected {
        name: String,
        nan_count: usize,
        inf_count: usize,
        first_index: usize,
    },

    /// General numerical instability.
    #[error("Numerical instability: {context}")]
    NumericalInstability { context: String },

    /// Invalid configuration parameter.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    // -------------------------------------------------------------------------
    // Inference Errors
    // -------------------------------------------------------------------------
    /// Generic inference error with context.
    #[error("Inference error: {0}")]
    Inference(String),

    /// Invalid timestep value.
    #[error("Invalid timestep: {value}, max allowed: {max}")]
    InvalidTimestep { value: usize, max: usize },

    /// Invalid number of views provided.
    #[error("Invalid view count: expected {expected}, got {got}")]
    InvalidViewCount { expected: usize, got: usize },

    /// Invalid latent tensor shape.
    #[error("Invalid latent shape: expected {expected:?}, got {got:?}")]
    InvalidLatentShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// Skip connection underflow during U-Net forward pass.
    #[error(
        "Skip connection underflow: expected {expected} connections, only {available} available"
    )]
    SkipConnectionUnderflow { expected: usize, available: usize },

    // -------------------------------------------------------------------------
    // Pipeline Errors
    // -------------------------------------------------------------------------
    /// Scheduler not initialized before use.
    #[error("Scheduler not initialized: call set_timesteps() first")]
    SchedulerNotInitialized,

    /// CLIP encoding failed.
    #[error("CLIP encoding failed: {0}")]
    ClipEncodingFailed(String),

    /// VAE encoding failed.
    #[error("VAE encoding failed: {0}")]
    VaeEncodeFailed(String),

    /// VAE decoding failed.
    #[error("VAE decoding failed: {0}")]
    VaeDecodeFailed(String),

    /// U-Net forward pass failed.
    #[error("U-Net forward failed at timestep {timestep}: {reason}")]
    UnetForwardFailed { timestep: usize, reason: String },

    // -------------------------------------------------------------------------
    // I/O Errors
    // -------------------------------------------------------------------------
    /// I/O error during file operations.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Image processing error.
    #[error("Image processing error: {0}")]
    ImageProcessingError(String),

    // -------------------------------------------------------------------------
    // Candle Backend Errors
    // -------------------------------------------------------------------------
    /// Error from candle tensor operations.
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

/// Result type for diffusion operations.
pub type DiffusionResult<T> = std::result::Result<T, DiffusionError>;

// CFG guidance re-exports
pub use cfg_guidance::{
    apply_cfg, apply_cfg_multi_guidance, apply_cfg_multi_view, blend_unconditional, compute_mean,
    compute_std, rescale_cfg_result, CfgError, CfgGuidance, CfgScaleSchedule, CfgScheduleKind,
    ConstantCfgSchedule, CosineCfgSchedule, DynamicThresholdCfg, LinearCfgSchedule,
    ThresholdCfgSchedule,
};

// Classifier guidance re-exports
pub use classifier_guidance::{
    annealed_guidance_scale, apply_guidance_step, clip_gradient, fd_gradient, fd_gradient_forward,
    run_guidance_steps, stochastic_fd_gradient, ClassifierGuidanceError, CompositeScoreFunction,
    GuidanceConfig, GuidanceStats, L2Regularizer, MeanMaximizer, ScoreFunction, TargetProximity,
};

// Re-exports
//
// `AttentionSpec` bundles the head geometry with the kernel selection
// (`AttentionBackend`, from `config`) that `CrossAttention::with_spec` and
// `MultiViewSpatialTransformer` consume; `SpatialTransformerSpec` bundles an
// `AttentionSpec` with the rest of one transformer stage's geometry.
// `mask_to_bias_tensor` converts an `attention_masking::AttentionMask` into
// the additive bias tensor `CrossAttention::forward_masked` accepts. All
// four were previously reachable only via their defining module's full
// path.
pub use attention::{
    mask_to_bias_tensor, AttentionSpec, CrossAttention, MultiViewSpatialTransformer,
    MultiViewTransformerBlock, SpatialTransformerSpec,
};
pub use camera::{timestep_embedding, CameraEmbedding, TimestepEmbedding};
pub use clip::{build_clip_encoder, ClipImageEncoder, ClipVisionConfig};
pub use config::{AttentionBackend, DiffusionConfig};
pub use pipeline::{MultiViewDiffusionPipeline, MultiViewOutput, SessionRequest};
pub use scheduler::{DdimScheduler, PredictionType};
pub use step_scheduler::{
    compare_step_counts, recommend_inference_steps, DdimStepScheduler, StepCountComparison,
    StepPattern, StepScheduleConfig,
};
pub use unet::MultiViewUNet;
pub use upsampler::{LatentUpsampler, UpsamplerMode};
pub use vae::Vae;

// Sliced attention exports
//
// NOTE: as of this writing, nothing in `unet.rs` or `pipeline.rs` selects
// `SlicedAttention` — there is no config switch analogous to
// `DiffusionConfig::use_flash_attention` for it, so this memory-optimization
// path (chunked attention scores to trade compute for peak memory) is
// exercised only by this module's own tests. Wiring it in requires adding an
// attention-backend selector to `DiffusionConfig` and dispatching on it in
// `attention.rs`; consider that a prerequisite before relying on this type
// to actually reduce memory in a real pipeline run.
pub use sliced_attention::{SlicedAttention, SlicedAttentionConfig};

// Flash attention exports (only when feature is enabled)
// Note: `DEFAULT_SCORE_MATRIX_BUDGET` is the score-matrix size below which
// `FlashAttention::forward` falls back to the plain kernel; it is exported so
// callers can reason about (or reproduce) that threshold without duplicating
// the constant.
#[cfg(feature = "flash_attention")]
pub use flash_attention::{
    flash_attention, flash_attention_with_config, FlashAttention, FlashAttentionConfig,
    DEFAULT_SCORE_MATRIX_BUDGET,
};

// Debug hooks re-exports
pub use debug_hooks::{
    all_finite, assert_finite, check_tensor_health, DebugConfig, DebugHooks, TensorHealth,
};

// Numerics re-exports
pub use numerics::{
    count_subnormal, gelu, gelu_slice, is_subnormal, layer_norm, log_softmax, rms_norm, silu,
    silu_slice, softmax, softmax_inplace, AttentionPrecision, NumericsError,
};

// Mixed-precision re-exports
pub use mixed_precision::{
    apply_precision, bf16_to_f32, f16_to_f32, f32_to_bf16, f32_to_f16, simulate_bf16, simulate_f16,
    MixedPrecisionConfig, OpType, PrecisionMode, PrecisionStats,
};

// KV-cache re-exports
pub use kv_cache::{CacheKeyBuilder, CacheStats, EvictionPolicy, KVCache, KVCacheConfig, KVEntry};

// Batch generation re-exports
pub use batch_gen::{
    BatchGenConfig, BatchGenerator, BatchStats, GeneratedView, GenerationRequest, GenerationResult,
};

// Streaming inference re-exports
// Note: `StreamingInputs` carries the conditioning tensors a real (model-backed)
// streaming run needs, so it belongs on the crate root next to the engine types.
pub use streaming::{
    StreamingConfig, StreamingInference, StreamingInputs, StreamingIterator, StreamingStep,
};

// Style transfer re-exports
pub use style_transfer::{
    adain, channel_statistics, color_latent, compute_style_stats, histogram_match,
    interpolate_style, latent_mean, latent_variance, per_channel_mean, per_channel_std,
    whiten_latent, StyleConfig, StylePalette, StyleTransferError, StyleTransferStats,
};

// Sequential VAE re-exports
// Note: the `*_streaming` variants are the ones that actually hold peak memory
// at `O(chunk_size · H · W)` — the non-streaming `encode_sequential` /
// `decode_sequential` still accumulate every view in the returned value.
pub use sequential_vae::{
    batch_memory_bytes, decode_sequential, decode_sequential_streaming, encode_sequential,
    encode_sequential_streaming, memory_reduction_ratio, peak_memory_bytes, DecodedViews,
    EncodedViews, SequentialVaeConfig,
};

// Quantization re-exports
pub use quantization::{
    compute_quantization_metrics, estimate_model_compression, plan_layer_quantization,
    AbsmaxQuantizer, LayerQuantizationPlan, MinMaxQuantizer, QuantizationMetrics, QuantizationMode,
    QuantizationScope, QuantizedTensor,
};

// Dynamic views re-exports
pub use dynamic_views::{
    estimate_memory, recommend_view_count, validate_view_config, MultiViewPositionEmbedding,
    SupportedViewCount, ViewAttentionMask, ViewCountConfig, ViewCountMemoryEstimate,
};

// Denoising visualization re-exports
pub use denoising_viz::{
    DenoisingStep, DenoisingTimeline, DenoisingVisualizer, DenoisingVizConfig, LatentColormap,
};

// Denoising trajectory re-exports
pub use denoising_trajectory::{
    compare_trajectories, compute_latent_statistics, compute_step_sizes, compute_trajectory_stats,
    detect_trajectory_anomalies, format_anomalies, format_trajectory_stats,
    interpolate_trajectory_step, latent_cosine_sim, latent_l1_distance, latent_l2_distance,
    latent_norm, resample_trajectory, smooth_trajectory, trajectory_divergence,
    DenoisingTrajectory, TrajectoryAnomaly, TrajectoryComparison, TrajectoryError, TrajectoryStats,
    TrajectoryStep,
};

// Weight offload re-exports
pub use weight_offload::{
    recommend_strategy, ComponentType, InferencePhase, MemoryBudget, OffloadSchedule,
    OffloadStrategy,
};

// Profiling re-exports
//
// `DiffusionProfiler` records per-layer timings (`LayerProfile`) around a
// forward pass; the two `estimate_*_memory_bytes` helpers are pure functions
// that supply the memory estimate `DiffusionProfiler::start_with_memory`
// attaches to each record.
pub use profiling::{
    estimate_attention_memory_bytes, estimate_unet_memory_bytes, DiffusionProfiler, LayerProfile,
};

// Attention visualization re-exports
pub use attention_viz::{
    apply_colormap, attention_map_to_image, format_attention_stats, heatmap_to_rgba,
    normalize_attention, overlay_attention_on_image, AttentionColormap, AttentionMap,
    AttentionVizConfig, CrossViewAttentionMap,
};

// Fused attention re-exports
//
// NOTE: same reachability gap as `sliced_attention` above — nothing in
// `unet.rs` or `pipeline.rs` constructs a `FusedQKV` or calls
// `scaled_dot_product_attention` on the default inference path; this is a
// standalone, independently-tested fused-projection utility rather than
// something the crate's own pipeline currently exercises.
//
// `scaled_dot_product_attention_with_scale` is the full form of
// `scaled_dot_product_attention`: an explicit softmax scale (e.g.
// `FusedAttentionConfig::scale`), optional causal masking, and optional
// training-time attention dropout.
// `HeadShape` carries the `(seq_q, seq_k, head_dim)` extents both take, and
// has to travel with them.
pub use fused_attention::{
    fused_qkv_projection, scaled_dot_product_attention, scaled_dot_product_attention_with_scale,
    softmax_over_dim, FusedAttentionConfig, FusedQKV, HeadShape,
};

// Guidance rescaling re-exports
// Note: `GuidanceStats` is already re-exported from `classifier_guidance`, so
// we alias it to `RescalingGuidanceStats` to avoid a name conflict.
// Note: `dynamic_threshold` is already re-exported from `ddpm_sampler` (the
// DDPM variant takes a ratio, not a percentile); we re-export this module's
// version under the alias `rescaling_dynamic_threshold`.
pub use guidance_rescaling::{
    abs_percentile, adaptive_guidance_scale, annealed_phi, apply_cfg_guidance,
    apply_rescaled_guidance, box_blur_guidance, compute_guidance_stats,
    dynamic_threshold as rescaling_dynamic_threshold, effective_scale_with_phi, l2_norm_vec,
    l2_normalize, phi_rescale, self_attention_guidance, std_dev, GuidanceRescalingError,
    GuidanceStats as RescalingGuidanceStats, RescalingConfig,
};

// ControlNet re-exports
// Note: `ZeroConv` is the zero-initialised 1×1 injection projection used by
// every ControlNet branch, and `MAX_INJECTION_LAYERS` is the hard bound that
// `ControlNetConfig::injection_layers` entries are validated against; both are
// part of the public ControlNet surface and are exported here so callers do
// not have to reach into the `controlnet` module path.
pub use controlnet::{
    apply_edge_enhancement, condition_images_from_multi_view, ControlFeature, ControlNetCondition,
    ControlNetConfig, ControlNetProcessor, ControlSignalType, ZeroConv, MAX_INJECTION_LAYERS,
};

// Text conditioning re-exports
pub use text_conditioning::{
    SimpleTokenizer, TextBatch, TextConditioner, TextConditioningConfig, TextEmbedding,
};

// Distillation loss re-exports
// Note: `lpips_proxy_loss_planar` is the layout-aware form of
// `lpips_proxy_loss` — it takes a `LatentDims` (`C × H × W`) and evaluates the
// patch metric per channel plane instead of treating the flat buffer as one
// square image; prefer it for real diffusion latents. `vp_coefficients`,
// `edm_sigma`, `midpoint_alpha_cumprod` and `ddim_sample_at` are the
// noise-schedule helpers progressive-distillation callers need to build their
// own teacher/student steps.
pub use distillation_loss::{
    compute_distillation_loss, ddim_sample_at, edm_sigma, huber_loss, l2_distillation_loss,
    lpips_proxy_loss, lpips_proxy_loss_planar, midpoint_alpha_cumprod, vp_coefficients,
    DistillationConfig, DistillationLossResult, DistillationMode, DistillationStep, EmaTeacher,
    LatentDims, NoisePrediction, TeacherStudentPair,
};

// Image preprocessing re-exports
pub use image_preprocessing::{
    crop_image, flip_horizontal, flip_vertical, gaussian_blur, normalize_image, pad_to_square,
    resize_image, CropMode, ImageDims, ImageStats, NormalizationMode, PreprocessError,
    PreprocessingPipeline, PreprocessingStep, ResizeFilter,
};

// Image editing re-exports
//
// NOTE: `image_editing` exposes two parallel, overlapping APIs whose
// similarly-named functions/types are easy to conflate:
//   - The `EditLatent`-based API (`EditConfig`, `EditStats`, `EditMask`,
//     `add_edit_noise`, `blend_with_mask`, `interpolate_edit_latents`,
//     `edit_distance`, ...) from the module's top level.
//   - The `sdedit` slice-based API (`ImageEditingConfig`, `SdeditStats`,
//     `edit_add_noise`, `edit_blend_with_mask`, `edit_lerp_latents`,
//     `edit_latent_distance`, ...) from the `image_editing::sdedit`
//     submodule, following the standard DDPM/SDEdit alpha-bar
//     forward-noising convention (see `edit_cosine_alpha_bars` /
//     `edit_linear_alpha_bars`).
// The two APIs are **not** interchangeable — their input layouts and
// forward-process conventions differ — but they now share one error type,
// `ImageEditingError` (there is no separate `sdedit`-only error enum), so a
// caller mixing both needs only `?` at every fallible call and no
// conversion between error types. Prefer the `sdedit` (`edit_*`-prefixed)
// family for new code, and check each function's own rustdoc for its exact
// forward-process convention before mixing the two.
pub use image_editing::{
    add_edit_noise,
    apply_inpaint_mask,
    blend_with_mask,
    compute_edit_stats,
    compute_sdedit_stats,
    edit_add_noise,
    edit_blend_with_mask,
    edit_cosine_alpha_bars,
    edit_cosine_similarity,
    edit_distance,
    edit_expand_mask_to_channels,
    edit_latent_distance,
    edit_lerp_latents,
    edit_linear_alpha_bars,
    edit_mean_latent,
    edit_normalize_latent,
    edit_project_out,
    edit_sample_noise,
    edit_slerp_latents,
    edit_start_timestep,
    edit_variance_map,
    format_sdedit_stats,
    interpolate_edit_latents,
    prepare_edit_input,
    sample_edit_at_levels,
    sdedit_noise_step,
    sdedit_perturb,
    // Mask-aware SDEdit: honours `EditConfig::use_mask` (plain
    // `sdedit_perturb` ignores that field entirely).
    sdedit_perturb_with_mask,
    EditConfig,
    EditHistory,
    EditLatent,
    EditMask,
    EditMode,
    EditStats,
    // SDEdit slice-based API (sdedit submodule); shares `ImageEditingError`
    // with the `EditLatent`-based API below instead of its own error type.
    ImageEditingConfig,
    ImageEditingError,
    SdeditStats,
};

// Latent interpolation re-exports
pub use latent_interp::{
    latent_pca_2d, lerp, nearest_k_neighbors, nearest_neighbor, slerp, InterpError,
    InterpolationMethod, InterpolationPath, LatentArithmetic, LatentInterpolationConfig,
    LatentVector,
};

// Latent walk re-exports
pub use latent_walk::{
    analyze_walk_path,
    circular_walk,
    // Semantic editing functions
    directional_walk,
    // Core walk functions
    linear_walk,
    multi_direction_walk,
    random_walk,
    resample_path,
    run_walk,
    sample_path,
    spherical_walk,
    // Direction type
    LatentDirection,
    // Interactive explorer
    LatentExplorer,
    // Error type
    LatentWalkError,
    WalkConfig,
    // Walk parameterization
    WalkMode,
    // Path analysis
    WalkPathStats,
};

// Latent space analysis re-exports
pub use latent_space_analysis::{
    assign_to_clusters, compute_dataset_entropy, compute_dim_stats, compute_inertia,
    compute_interpolation_quality, compute_latent_dataset_stats, compute_pca,
    estimate_intrinsic_dimensionality, format_cluster_summary, format_pca_summary, kmeans_init,
    power_iteration_step, run_kmeans, update_centroids, InterpolationQuality, LatentAnalysisConfig,
    LatentAnalysisError, LatentClusterResult, LatentDatasetStats, LatentDimStats, PcaResult,
};

// Noise schedule analysis re-exports
pub use noise_schedule_analysis::{
    analyze_ddim_sampling, analyze_schedule, compare_schedules, ddim_timesteps,
    format_comparison_table, format_schedule_summary, format_snr_curve_ascii, DdimSamplingAnalysis,
    NoiseSchedule, ScheduleAnalysis, ScheduleAnalysisError, ScheduleComparison, ScheduleType,
};

// Noise prediction analysis re-exports
// Note: `format_noise_report` is re-exported as `format_noise_prediction_report` to avoid
// potential future collisions with other format_* functions.
pub use noise_prediction_analysis::{
    analyze_noise_prediction, channel_means, compare_predictions, compute_alpha_bar,
    compute_noise_floor, compute_prediction_stats, compute_spatial_error_map, compute_timestep_snr,
    detect_failure_mode, error_map_percentile,
    format_noise_report as format_noise_prediction_report, vec_std, weighted_noise_loss,
    FailureMode, NoiseAnalysisConfig, NoiseAnalysisError, NoisePredictionReport, PredictionHistory,
    PredictionStats, SpatialErrorMap, TimestepSnr,
};

// Prompt scheduler re-exports
pub use prompt_scheduler::{
    augment_embedding, embedding_diversity, interpolate_embeddings, make_cyclic_schedule,
    make_strengthening_schedule, mix_prompts, summarize_schedule, InterpolationMode,
    PromptEmbedding, PromptKeyframe, PromptScheduler, PromptSchedulerError, ScheduleSummary,
    WeightedPrompt,
};

// Model variant re-exports
pub use model_variants::{
    are_configs_compatible, config_diff, fits_in_vram, sd15_config, sd21_config, sdxl_config,
    zero123_config, ModelFamily, ModelVariantError, ModelVariantRegistry, MultiViewAdapterConfig,
    UNetVariantConfig,
};

// Attention masking re-exports
pub use attention_masking::{
    angular_proximity_mask, build_layer_mask, build_mask, causal_mask, cross_view_mask, full_mask,
    local_mask, neighbor_view_mask, reference_view_mask, ring_view_mask, self_view_mask,
    AttentionMask, AttentionMaskError, LayerMaskConfig, MaskPattern,
};

// DDIM inversion re-exports
pub use inversion::{
    blend_at_noise_level, constant_noise_predictor, ddim_denoise_step, ddim_inversion_step,
    mix_trajectories, null_noise_predictor, run_ddim_inversion, run_ddim_reconstruction,
    DdimInversionConfig, InversionError, InversionSchedule, InversionTrajectory,
    NoiseScheduleCoeffs,
};

// Token merging re-exports
// Note: `tome_merge_with_keys` is the entry point the ToMe paper actually
// describes — the partition is decided by the attention *keys* rather than the
// token values; `tome_merge` is the values-only convenience wrapper.
pub use token_merging::{
    attention_speedup_factor, bipartite_soft_matching, compute_similarity_matrix,
    compute_tome_stats, merge_states_compatible, merge_tokens, progressive_merge,
    progressive_unmerge, token_cosine_similarity, token_l2_norm, tome_merge, tome_merge_with_keys,
    tome_unmerge, unmerge_tokens, MergeMode, MergeState, ToMeConfig, ToMeStats, TokenMergeError,
};

// Unconditional dropout re-exports
pub use uncond_dropout::{
    // Backward-compat helpers
    compute_dropout_weights,
    normalize_weights,
    uncond_cfg_combine,
    uncond_cfg_combine_batch,
    uncond_format_config,
    uncond_format_stats,
    // Standalone functions
    uncond_update_stats,
    // Mask
    DropoutMask,
    ModalityDropout,
    // Multi-modal dropout
    MultiModalDropout,
    // Null embedding
    NullEmbedding,
    NullEmbeddingType,
    UncondDropout,
    // Config and core struct
    UncondDropoutConfig,
    // Error
    UncondDropoutError,
    // Stats
    UncondDropoutStats,
    // Schedules
    UncondSchedule,
};

// Image variation re-exports
// Note: the 3-D latent type is called `VarLatentVector` to avoid a name collision
// with `latent_interp::LatentVector`; likewise `var_channel_statistics` avoids
// clashing with `style_transfer::channel_statistics`.
pub use image_variations::{
    add_latent_noise, clamp_latent, compute_variation_stats, generate_guided_variations,
    generate_variations, interpolate_latents, latent_cosine_similarity, latent_distance,
    project_to_sphere, rank_by_diversity, spherical_interpolate_latents, var_channel_statistics,
    ImageVariationConfig, ImageVariationError, ImageVariationStats, VarLatentVector,
    VariationExplorer, VariationMode,
};

// Prompt weighting re-exports
// Note: `WeightedPrompt` is intentionally omitted — `prompt_scheduler` already
// exports a same-named type (embedding-based). Use `prompt_weighting::WeightedPrompt`
// for the token-based variant.
// Note: `normalize_weights` is intentionally omitted — `uncond_dropout` already
// exports a same-named function. Use `prompt_weighting::normalize_weights` directly.
pub use prompt_weighting::{
    apply_embedding_weights, blend_weighted_embeddings, clamp_weights, compute_weight_stats,
    merge_prompts, parse_weighted_prompt, prompt_to_arrays, scale_weights, tokenize,
    weight_to_attention_bias, weighted_average_embedding, PromptWeightingError, WeightScaleMode,
    WeightStats, WeightedToken, WeightingConfig,
};

// DDPM sampler re-exports
pub use ddpm_sampler::{
    clip_sample, compute_schedule_stats, cosine_beta_schedule, dynamic_threshold, generate_noise,
    inference_timesteps, linear_beta_schedule, p_sample, predict_x_0_from_noise,
    q_posterior_mean_variance, q_sample, sample, sigmoid_beta_schedule, snr_weighted_loss,
    BetaSchedule, DdpmError, DdpmSamplerConfig, DdpmSchedule, ScheduleStats,
};

// LoRA adapter re-exports
pub use lora_adapter::{
    apply_lora_dropout, compute_lora_stats, deserialize_adapter, init_lora_a, init_lora_b,
    interpolate_adapters, lora_backward, lora_sgd_step, lora_weight_norm, matmul,
    merge_lora_weights, serialize_adapter, LoraAdapter, LoraConfig, LoraError, LoraLayer,
    LoraStats,
};

// Score distillation re-exports
// Note: `VsdState` owns the per-particle LoRA adapters that `vsd_gradient`
// scores, so VSD callers need it alongside `VsdConfig`.
pub use score_distillation::{
    add_sds_noise, annealed_timestep_range, backprop_sds_to_gaussians, classifier_free_guidance,
    clip_sds_gradient, compute_sds_step_stats, compute_weight, sample_timestep, sds_grad_norm,
    sds_gradient, sds_loss, vsd_gradient, ScoreDistillationError, ScoreWeighting, SdsConfig,
    SdsNoiseSchedule, SdsStepStats, VsdConfig, VsdState,
};

// Multi-view consistency re-exports
pub use multi_view_consistency::{
    aggregate_view_gradients, compute_consistency_stats, compute_cross_view_correlation,
    compute_pairwise_distances, consistency_improvement, consistency_loss, consistency_map,
    find_anchor_view, format_consistency_report, l1_distance, l2_distance, luminance_correlation,
    mean_view, median_view, penalized_loss, weighted_mean_view, ConsistencyConfig,
    ConsistencyError, ConsistencyLossType, ConsistencyStats, CrossViewCorrelation, ViewBundle,
};

// Avatar conditioning re-exports
// Note: `l2_normalize` is intentionally omitted — `guidance_rescaling` already
// re-exports a function of the same name.  Use `avatar_conditioning::normalize_conditioning`
// or `guidance_rescaling::l2_normalize` for the respective variants.
pub use avatar_conditioning::{
    batch_condition_from_flame_params, blend_conditioning, compute_conditioning_stats,
    condition_from_flame_params, conditioning_similarity, create_neutral_conditioning,
    encode_expression_params, encode_pose_params, encode_shape_params, encode_translation,
    interpolate_conditioning_sequence, normalize_conditioning, ConditioningError,
    ConditioningStats, ConditioningVector, FlameConditioningConfig, LinearEmbedding,
    SinusoidalEncoding,
};

// Adaptive sampling re-exports
pub use adaptive_sampling::{
    box_muller, compute_effective_snr_range, compute_snr_sampling_weights,
    compute_timestep_histogram, format_sampling_stats, sample_cdf, sample_log_normal,
    sample_logit_normal, sigma_to_alpha_bar, timestep_to_sigma, uniform_f32, xorshift64,
    AdaptiveSamplingError, ContinuousSampler, ImportanceSampler, LogNormalSampler, MinSnrWeighter,
    SamplingStats, TimestepHistory, UniformSampler,
};

// Multi-step sampler re-exports
// Note: `SamplingNoiseSchedule` is used (instead of `NoiseSchedule`) to avoid
// collision with `noise_schedule_analysis::NoiseSchedule` already exported above.
// `sampler_apply_cfg` avoids collision with `cfg_guidance::apply_cfg`.
// `dpm_step_size` is the log-SNR step size `h` that `dpm_plus_plus_2m_step`
// consumes; callers driving that step themselves need it to carry the previous
// step's `h` forward for the second-order correction.
pub use multi_step_sampler::{
    compute_timestep_schedule, ddim_step, dpm_plus_plus_2m_step, dpm_step_size,
    format_sampler_stats, plms_step, predict_x0, sampler_apply_cfg, MultiStepSampler,
    MultiStepSamplerConfig, SamplerError, SamplerKind, SamplingNoiseSchedule,
};

// Consistency model re-exports
// Note: types are prefixed with `Cm` to avoid collisions with
// `multi_view_consistency::{ConsistencyError, ConsistencyConfig}` already
// exported above.
pub use consistency_model::{
    cm_add_noise, cm_compute_target, cm_loss_weight, cm_multi_step_inference, cm_pseudo_huber_loss,
    cm_sample_noise, cm_single_step_inference, cm_weighted_loss, format_cm_config, format_cm_stats,
    lcm_guided_output, lcm_lambda, lcm_skipped_timesteps, CmConsistencyConfig, CmConsistencyError,
    CmNoiseSchedule, ConsistencyPreconditioning, ConsistencySkip, LcmConfig, LcmLossWeight,
};

// Prior preservation re-exports
pub use prior_preservation::{
    format_prior_stats, prior_combined_loss, prior_cosine_similarity, prior_diverse_sample,
    prior_effective_weight, prior_kl_divergence, prior_per_sample_losses, prior_preservation_loss,
    prior_soft_loss, prior_softmax, prior_update_stats, xorshift64 as prior_xorshift64,
    xorshift_f32 as prior_xorshift_f32, ClassPriorBuffer, PriorError, PriorPreservationConfig,
    PriorPreservationTracker, PriorStats,
};

// Flow matching re-exports
pub use flow_matching::{
    compute_flow_stats, fm_euler_step, fm_heun_step, fm_inference_timesteps, fm_interpolate,
    fm_loss, fm_loss_weight, fm_reconstruct_x0, fm_sample_noise, fm_sample_pair, fm_sample_time,
    fm_stochastic_interpolant, fm_target_velocity, format_flow_config, format_flow_stats,
    FlowIntegrator, FlowLossWeight, FlowMatchingConfig, FlowMatchingError, FlowPath, FlowStats,
    TimeSchedule,
};

// Identity conditioning re-exports
// Note: `lerp`, `slerp`, `l2_normalize` are already exported from other modules;
// identity conditioning functions use the `ident_` prefix to avoid collisions.
pub use identity_conditioning::{
    ident_compute_stats, ident_cosine_similarity, ident_format_config, ident_format_stats,
    ident_hash_feature, ident_lerp, ident_mean_embedding, ident_normalize, ident_similarity_matrix,
    ident_slerp, IdentityCache, IdentityConditioningError, IdentityConfig, IdentityEncoder,
    IdentityFeature, IdentityStats, LinearLayer,
};

// Latent blend re-exports.
// All standalone functions carry the `lb_` prefix to avoid collisions with
// `lerp`/`slerp` (latent_interp), `fm_interpolate` (flow_matching), and other
// trajectory/statistics names in this crate.
pub use latent_blend::{
    lb_add_scaled,
    lb_blend_multi,
    lb_blend_trajectories,
    lb_channel_stats,
    lb_circular_mask,
    lb_compute_stats,
    lb_denormalize,
    lb_dilate_mask,
    lb_format_stats,
    lb_frequency_blend,
    // Frequency-domain blending
    lb_frequency_separate,
    lb_gradient_mask,
    // Statistical harmonization
    lb_harmonize_statistics,
    lb_l2_distance,
    // Basic blending
    lb_lerp,
    // Spatial mask blending
    lb_mask_blend,
    lb_normalize,
    lb_slerp,
    lb_smooth_mask,
    lb_trajectory_length,
    lb_trajectory_velocity,
    // Error type
    LatentBlendError,
    // Statistics and formatting
    LatentBlendStats,
    // Core type
    LatentTensor,
    LatentTrajectory,
    // Trajectory types and functions
    LatentTrajectoryStep,
};

// Cross-frame consistency re-exports.
// Note: `ConsistencyError` is aliased to `CfcConsistencyError` to avoid
// collision with `multi_view_consistency::ConsistencyError` already exported
// above. Likewise `ConsistencyLossConfig` is aliased to `CfcLossConfig` and
// `ConsistencyLoss` to `CfcConsistencyLoss` for clarity.
pub use cross_frame_consistency::{
    cfc_bilinear_sample,
    cfc_compute_flow,
    cfc_consistency_loss,
    cfc_format_loss,
    // Formatting
    cfc_format_pair_consistency,
    cfc_format_report,
    cfc_frame_difference,
    cfc_frame_pair_consistency,
    cfc_mae,
    // Per-frame metrics
    cfc_psnr,
    cfc_rmse,
    cfc_sequence_consistency,
    cfc_ssim,
    cfc_to_grayscale,
    cfc_warp_frame,
    // Error type (aliased to avoid collision with multi_view_consistency)
    ConsistencyError as CfcConsistencyError,
    ConsistencyLoss as CfcConsistencyLoss,
    // Training loss (aliased to avoid future collision with multi_view_consistency names)
    ConsistencyLossConfig as CfcLossConfig,
    // Optical flow
    FlowConfig,
    // Frame representation
    Frame,
    // Frame-pair metrics
    FramePairConsistency,
    // Sequence metrics
    SequenceConsistencyReport,
};

// Score matching re-exports.
// Note: `ScoreFunction` is already exported from `classifier_guidance`; the
// score_matching variant is aliased as `SmScoreFunction` to avoid a collision.
pub use score_matching::{
    sm_add_noise,
    sm_c_in,
    sm_c_noise,
    sm_c_out,
    // Karras preconditioning (prefixed to avoid future collisions)
    sm_c_skip,
    sm_compute_stats,
    sm_dsm_loss,
    sm_dsm_loss_per_sample,
    sm_format_config,
    sm_format_stats,
    sm_geometric_sigmas,
    sm_hutchinson_trace_estimate,
    sm_ism_loss,
    sm_loss_weight,
    sm_precondition_input,
    sm_precondition_output,
    sm_sample_sigma,
    sm_sliced_score_matching_loss,
    NoisyBatch,
    ScoreFunction as SmScoreFunction,
    ScoreMatchingConfig,
    ScoreMatchingError,
    ScoreMatchingStats,
    SmWeighting,
};

// ---------------------------------------------------------------------------
// Crate-root re-export regression tests
// ---------------------------------------------------------------------------

/// Pins the crate-root public surface for items that were previously reachable
/// only through their defining module path.
///
/// Every item below is referenced through a `crate::…` path, so dropping (or
/// forgetting to re-add) one of the corresponding `pub use` lines above becomes
/// a compile error here instead of a silent breaking change for downstream
/// callers.
#[cfg(test)]
mod crate_root_reexports {
    /// `controlnet::{ZeroConv, MAX_INJECTION_LAYERS}`.
    #[test]
    fn controlnet_injection_items_are_reachable() {
        assert_eq!(crate::MAX_INJECTION_LAYERS, 8);

        let conv = crate::ZeroConv::zeros(2, 3);
        assert!(conv.is_zero(), "a fresh ZeroConv is at its zero init");
        assert_eq!(conv.in_channels(), 2);
        assert_eq!(conv.out_channels(), 3);
        assert_eq!(conv.project(&[1.0, 2.0]), vec![0.0, 0.0, 0.0]);
    }

    /// `distillation_loss::{vp_coefficients, edm_sigma, midpoint_alpha_cumprod,
    /// ddim_sample_at}`.
    #[test]
    fn distillation_schedule_helpers_are_reachable() {
        let (alpha, sigma) = crate::vp_coefficients(1.0);
        assert!(
            (alpha - 1.0).abs() < 1e-6,
            "alpha at abar=1 is 1, got {alpha}"
        );
        assert!(sigma.abs() < 1e-6, "sigma at abar=1 is 0, got {sigma}");

        // sigma = sqrt((1 - abar) / abar) = 1 at abar = 0.5.
        let sigma_half = crate::edm_sigma(0.5);
        assert!((sigma_half - 1.0).abs() < 1e-6, "got {sigma_half}");

        let mid = crate::midpoint_alpha_cumprod(0.2, 0.8);
        assert!(
            mid > 0.2 && mid < 0.8,
            "log-SNR midpoint lies between its endpoints, got {mid}"
        );

        // At abar = 1 the DDIM step reproduces x0 exactly (sigma = 0).
        let x0 = [1.0_f32, -2.0, 3.0];
        let eps = [10.0_f32, 10.0, 10.0];
        assert_eq!(crate::ddim_sample_at(&x0, &eps, 1.0), vec![1.0, -2.0, 3.0]);
    }

    /// `distillation_loss::{lpips_proxy_loss_planar, LatentDims}`.
    #[test]
    fn planar_lpips_proxy_is_reachable() {
        let dims = crate::LatentDims::new(2, 2, 2);
        assert_eq!(dims.len(), 8);
        assert!(!dims.is_empty());

        let latent = vec![0.5_f32; 8];
        let loss = crate::lpips_proxy_loss_planar(&latent, &latent, 2, dims)
            .expect("identical planar latents must score");
        assert!(loss.abs() < 1e-6, "self-distance must be 0, got {loss}");
    }

    /// `flash_attention::DEFAULT_SCORE_MATRIX_BUDGET` (feature-gated, so this
    /// test is gated the same way as the `pub use` it guards).
    #[cfg(feature = "flash_attention")]
    #[test]
    fn flash_attention_score_budget_is_reachable() {
        assert_eq!(crate::DEFAULT_SCORE_MATRIX_BUDGET, 64 * 1024 * 1024);
    }

    /// `multi_step_sampler::dpm_step_size`.
    #[test]
    fn dpm_step_size_is_reachable() {
        let schedule = crate::SamplingNoiseSchedule::cosine(10);

        let h = crate::dpm_step_size(9, 5, &schedule)
            .expect("an interior DPM++ step has a finite log-SNR step size");
        assert!(h > 0.0, "denoising increases log-SNR, got h = {h}");

        assert!(
            crate::dpm_step_size(9, 0, &schedule).is_none(),
            "the clean boundary (sigma = 0) has no finite step size"
        );
    }

    /// `streaming::StreamingInputs` — constructing it needs three candle
    /// tensors, so only the path is pinned here.
    #[test]
    fn streaming_inputs_type_is_reachable() {
        let inputs: Option<crate::StreamingInputs> = None;
        assert!(inputs.is_none());
    }

    /// `token_merging::tome_merge_with_keys`.
    #[test]
    fn tome_merge_with_keys_is_reachable() {
        let tokens: Vec<f32> = (0..16).map(|x| x as f32).collect(); // 4 tokens x 4 dims
        let keys: Vec<f32> = (0..8).map(|x| x as f32).collect(); // 4 tokens x 2 dims
        let config = crate::ToMeConfig::default();

        let (merged, state) = crate::tome_merge_with_keys(&tokens, &keys, 4, 4, 2, &config)
            .expect("well-formed token/key tensors must merge");
        assert_eq!(merged.len(), state.merged_n_tokens * 4);
        assert_eq!(state.original_n_tokens, 4);
    }

    /// `image_editing::sdedit_perturb_with_mask`.
    #[test]
    fn sdedit_perturb_with_mask_is_reachable() {
        let config = crate::EditConfig {
            strength: 0.5,
            n_timesteps: 10,
            guidance_scale: 7.5,
            use_mask: true,
        };
        let alpha_bars = crate::edit_linear_alpha_bars(10, 1e-4, 0.02);
        let latent = vec![0.25_f32; 4];
        // All-zero mask: every pixel is preserved, so the perturbation is a
        // no-op even though noise was sampled.
        let mask = crate::EditMask::new(2, 2, 0.0);
        let mut rng_state = 7_u64;

        let (out, t_start) = crate::sdedit_perturb_with_mask(
            &latent,
            &config,
            &alpha_bars,
            &mut rng_state,
            Some(&mask),
        )
        .expect("masked SDEdit perturbation must succeed");

        assert_eq!(t_start, 5, "t_start = floor(n_timesteps * strength)");
        for (got, want) in out.iter().zip(latent.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "an all-zero mask preserves the latent: got {got}, want {want}"
            );
        }
    }

    /// `fused_attention::scaled_dot_product_attention_with_scale`.
    #[test]
    fn sdpa_with_scale_is_reachable() {
        // One query, one key: softmax over a single logit is 1, so the output
        // is exactly `v` regardless of the scale.
        let out = crate::scaled_dot_product_attention_with_scale(
            &[1.0],
            &[1.0],
            &[2.0],
            crate::HeadShape::square(1, 1),
            0.5,
            false,
            None,
        )
        .expect("single-position attention must succeed");
        assert_eq!(out.len(), 1);
        assert!((out[0] - 2.0).abs() < 1e-6, "got {}", out[0]);
    }

    /// `profiling::{DiffusionProfiler, LayerProfile,
    /// estimate_attention_memory_bytes, estimate_unet_memory_bytes}`.
    #[test]
    fn profiling_items_are_reachable() {
        let attention_bytes = crate::estimate_attention_memory_bytes(1, 4, 2, 8);
        assert!(attention_bytes > 0);
        assert!(crate::estimate_unet_memory_bytes(1, 4, 16, 2) > attention_bytes);

        let mut profiler = crate::DiffusionProfiler::new();
        profiler.start_with_memory("attn", attention_bytes);
        assert!(profiler.stop("attn").is_some(), "a started layer stops");

        let recorded: &[crate::LayerProfile] = profiler.profiles();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].name, "attn");
        assert_eq!(recorded[0].estimated_memory_bytes, attention_bytes);
    }

    /// `sequential_vae::{encode_sequential_streaming, decode_sequential_streaming}`.
    #[test]
    fn sequential_vae_streaming_fns_are_reachable() {
        let config = crate::SequentialVaeConfig::new(2, 4, 16, 16, 0.18215);
        let images = vec![vec![0.5_f32; 3 * 16 * 16]; 3];

        let mut encoded_chunk_sizes = Vec::new();
        let encoded_total =
            crate::encode_sequential_streaming(&images, &config, |chunk: Vec<Vec<f32>>| {
                encoded_chunk_sizes.push(chunk.len());
                Ok(())
            })
            .expect("streaming encode of well-formed images must succeed");
        assert_eq!(encoded_total, 3);
        assert_eq!(
            encoded_chunk_sizes,
            vec![2, 1],
            "chunk_size = 2 splits 3 views as 2 + 1"
        );

        let encoded =
            crate::encode_sequential(&images, &config).expect("batch encode must succeed");
        let mut decoded_chunk_sizes = Vec::new();
        let decoded_total =
            crate::decode_sequential_streaming(&encoded, &config, |chunk: Vec<Vec<f32>>| {
                decoded_chunk_sizes.push(chunk.len());
                Ok(())
            })
            .expect("streaming decode of well-formed latents must succeed");
        assert_eq!(decoded_total, 3);
        assert_eq!(decoded_chunk_sizes, vec![2, 1]);
    }

    /// `score_distillation::VsdState` — building one allocates per-particle
    /// LoRA adapters, so only the path is pinned here.
    #[test]
    fn vsd_state_type_is_reachable() {
        let state: Option<crate::VsdState> = None;
        assert!(state.is_none());
    }

    /// `config::AttentionBackend` and `attention::{AttentionSpec,
    /// SpatialTransformerSpec, mask_to_bias_tensor}` — the only members of
    /// their modules' public surface that were previously reachable solely
    /// through the full `crate::config::…` / `crate::attention::…` path.
    #[test]
    fn attention_backend_and_spec_types_are_reachable() {
        assert_eq!(
            crate::AttentionBackend::default(),
            crate::AttentionBackend::Standard
        );
        assert!(!crate::AttentionBackend::Standard.needs_flash_feature());
        assert!(crate::AttentionBackend::Flash.needs_flash_feature());

        let attention = crate::AttentionSpec::standard(8, 64);
        assert_eq!(attention.heads, 8);
        assert_eq!(attention.dim_head, 64);
        assert_eq!(attention.backend, crate::AttentionBackend::Standard);

        let transformer = crate::SpatialTransformerSpec {
            in_channels: 320,
            depth: 1,
            context_dim: 1024,
            ip_dim: 1024,
            num_views: 4,
            num_groups: 32,
            use_linear_projection: false,
            attention,
        };
        assert_eq!(transformer.attention.heads, 8);
        assert_eq!(transformer.num_views, 4);

        let mask = crate::full_mask(3);
        let bias = crate::mask_to_bias_tensor(&mask, &candle_core::Device::Cpu)
            .expect("an all-true mask must convert to a bias tensor");
        assert_eq!(bias.dims(), &[1, 1, 3, 3]);
    }
}
