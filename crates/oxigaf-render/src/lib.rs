//! # oxigaf-render
//!
//! Differentiable 3D Gaussian Splatting rasterizer using wgpu compute shaders.
//!
//! The rasterizer implements the full 3DGS pipeline:
//! - **Forward**: project → sort → rasterize (per-tile alpha-blending)
//! - **Backward**: reverse-order gradient computation through the rasterizer
//!
//! FLAME mesh binding allows Gaussians to be anchored to a parametric head model.
//!
//! ## Cargo Features
//!
//! This crate supports the following feature flags:
//!
//! - **`default`** = `[]`:
//!   Minimal configuration with no extra features
//!
//! - **`gpu_debug`**:
//!   Enables GPU debug mode with validation layers:
//!   - Vulkan validation layers on Linux/Windows
//!   - Metal API validation on macOS
//!   - DirectX debug layer on Windows
//!   - Enhanced error messages and warnings
//!   - **Warning**: Adds significant runtime overhead (10-100× slower)
//!
//! The `gpu_debug` feature is useful for:
//! - Debugging shader errors
//! - Validating buffer usage
//! - Catching GPU API misuse
//! - Performance profiling with detailed traces
//!
//! Example usage:
//! ```toml
//! # In Cargo.toml
//! # For production use (fast, minimal validation)
//! oxigaf-render = { version = "0.1" }
//!
//! # For development/debugging (slow, extensive validation)
//! oxigaf-render = { version = "0.1", features = ["gpu_debug"] }
//! ```

#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]

pub mod ambient_occlusion;
pub mod antialiasing;
pub mod background;
pub mod binding;
pub mod bloom;
pub mod bloom_simple;
pub mod buffers;
pub mod bvh;
pub mod camera_path;
pub mod chrom_post_effect;
pub mod chromatic_aberration;
pub mod color_calibration;
pub mod color_grading;
pub mod colorspace;
pub mod compression;
pub mod config;
pub mod cov2d_backward;
pub mod cpu_reference;
pub mod debug_readback;
pub mod deform;
pub mod denoising;
pub mod density;
pub mod depth_map;
pub mod depth_of_field;
pub mod dof;
pub mod edge_detection;
pub mod environment;
pub mod exposure;
pub mod film_grain;
pub mod gaussian;
pub mod gaussian_culling;
pub mod gaussian_stats;
pub mod hdr_tone_mapping;
pub mod image_compositor;
pub mod image_resize;
pub mod init;
pub mod lens_distortion;
pub mod lens_effects;
pub mod lod;
pub mod mb_pipeline;
pub mod mip_splatting;
pub mod model_pruning;
pub mod motion_blur;
pub mod motion_blur_image;
pub mod multi_view;
pub mod normal_estimation;
pub mod panoramic;
pub mod picking;
pub mod pipeline;
pub mod pool;
pub mod post_processing;
pub mod profiler;
pub mod rasterizer;
pub mod render_graph;
pub mod render_metrics;
pub mod scene_compositor;
pub mod sharpening;
pub mod silhouette;
pub mod sort;
pub mod spherical_harmonics;
pub mod ssao;
pub mod stereo;
pub mod temporal;
pub mod temporal_aa;
pub mod tile_stats;
pub mod tone_curve;
pub mod validation;
pub mod vignetting;
pub mod workgroup;

pub use background::{
    generate_background, standard_augmentor, BackgroundAugmentor, BackgroundColor,
    BackgroundEnvMap, BackgroundError, BackgroundImage, BackgroundType,
};
pub use color_grading::{
    hsv_to_rgb_grading, luminance, rgb_to_hsv_grading, ColorGradingError, ColorGradingPipeline,
    ContrastAdjust, ExposureAdjust, GradingStep, Lut3D, RgbHistogram, SaturationAdjust, ToneCurve,
};
pub use hdr_tone_mapping::{
    aces_filmic,
    apply_exposure,
    apply_gamma,
    apply_gamma_image,
    // New per-pixel helpers
    apply_operator,
    auto_exposure,
    compute_hdr_stats,
    estimate_scene_key,
    filmic,
    format_tone_config,
    gamma_correct,
    hable,
    hdr_image_stats,
    // New linear↔sRGB (prefixed to avoid conflict with colorspace::*)
    hdr_linear_to_srgb,
    // New white balance (prefixed to avoid conflict)
    hdr_white_balance,
    image_hdr_linear_to_srgb,
    inverse_srgb_gamma,
    lottes_approx,
    // New analysis
    luminance_histogram,
    preset_aces,
    preset_filmic,
    preset_photography,
    preset_reinhard,
    recommend_exposure,
    reinhard,
    reinhard_extended,
    srgb_gamma,
    srgb_to_linear_hdr,
    tone_luminance,
    // New image-level pipeline
    tone_map,
    tone_map_image,
    tone_map_inplace,
    tone_map_rgba_image,
    HdrImageStats,
    HdrStats,
    ToneMapConfig,
    // New types and operators
    ToneMapOperator,
    ToneMappingConfig,
    ToneMappingError,
    ToneMappingOperator,
};
pub use image_compositor::{
    apply_blend_mode, blend_pixel, composite_image_layers, composite_over, composite_psnr,
    compute_image_composite_stats, extract_alpha, flatten_to_rgb, premultiply_alpha, replace_alpha,
    scale_alpha, solid_color, transparency_checkerboard, unpremultiply_alpha, BlendMode,
    CompositingLayer, CompositorConfig, ImageCompositeStats, ImageCompositorError,
};
pub use image_resize::{
    center_crop, crop, fit_dimensions, flip_horizontal, flip_vertical, image_pyramid, pad_image,
    pad_to_square, resize, resize_bicubic, resize_bilinear, resize_box, resize_nearest, rotate_180,
    rotate_90_ccw, rotate_90_cw, scale_by_factor, thumbnail,
    validate_buffer as validate_image_buffer, ResizeError, ResizeFilter,
};

pub use ambient_occlusion::{
    ao_apply_to_image, ao_bilateral_blur, ao_compute, ao_compute_stats, ao_cross, ao_dot,
    ao_noise_texture, ao_normalize, ao_sample_depth, ao_sample_kernel, ao_smoothstep, apply_ssao,
    format_ao_config, format_ao_stats, AoConfig, AoError, AoStats, SsaoResult,
};
pub use antialiasing::{
    aa_bilinear_sample,
    aa_edge_count,
    aa_edge_map,
    aa_luminance,
    aa_luminance_map,
    aa_quality_estimate,
    aa_sample_pixel,
    apply_antialiasing,
    apply_fxaa,
    apply_image_aa,
    apply_smaa_lite,
    apply_supersampling_aa,
    apply_temporal_aa,
    compute_scale_compensation,
    compute_screen_radius_px,
    format_aa_config,
    format_aa_stats,
    opacity_scale_from_screen_radius,
    AaConfig,
    // Post-processing AA
    AaError,
    AaMethod,
    AaStats,
    AliasingStats,
    MipSplatConfig,
};
pub use bvh::{extract_frustum_planes, Aabb, BvhNode, GaussianBvh};
pub use camera_path::{
    dolly_path, keyframe_to_render_camera, spiral_path, turntable_path, CameraKeyframe, CameraPath,
    PathInterpolation,
};
pub use color_calibration::{
    apply_calibration,
    apply_color_correction_matrix,
    apply_gamma_correction_image,
    apply_gamma_f32,
    apply_saturation_image,
    apply_srgb_decoding,
    apply_srgb_encoding,
    apply_white_balance_image,
    compute_color_stats,
    compute_correction_matrix,
    d65_white_balance,
    gray_world_white_balance,
    histogram_stretch_matrix,
    kelvin_to_rgb_multipliers,
    // Renamed to avoid clash with colorspace::{linear_to_srgb, srgb_to_linear}
    linear_to_srgb_calib,
    rgb_to_luminance,
    saturation_matrix,
    srgb_to_linear_calib,
    white_balance_matrix,
    CalibrationError,
    ColorCalibrationConfig,
    ColorMatrix,
    ColorStats,
    WhiteBalance,
};
pub use colorspace::{
    apply_tone_mapping_image, apply_white_balance, color_linear_to_srgb, color_srgb_to_linear,
    convert_color, convert_image_colorspace, hsl_to_linear_rgb, hsv_to_linear_rgb, lab_to_xyz,
    linear_rgb_to_hsl, linear_rgb_to_hsv, linear_rgb_to_xyz, linear_to_srgb, srgb_to_linear,
    white_balance_from_temperature, xyz_to_lab, xyz_to_linear_rgb, Color, ColorError, ColorSpace,
    ToneMapping,
};
pub use compression::{
    compress_gaussians, decompress_gaussians, CompressedGaussianModel, CompressedOpacities,
    CompressedPositions, CompressedRotations, CompressedScales, CompressionConfig,
    CompressionError, CompressionStats, DecompressedArrays, SceneBounds, ShCodebook, ShCompressed,
};
pub use config::RasterConfig;
pub use cpu_reference::{CpuCamera, CpuRasterizer, CpuRenderOutput};
pub use debug_readback::{DebugReadbackBuilder, RasterizationSnapshot, RasterizationStats};
pub use denoising::{
    bilateral_filter, denoise_adaptive, estimate_noise, gaussian_denoise, joint_bilateral_filter,
    median_filter, non_local_means, BilateralConfig, DenoisingError, DenoisingMethod,
    DenoisingPipeline, GaussianDenoiseConfig, MedianConfig, NlmConfig, NoiseStats,
};
pub use density::{DensityConfig, DensityController, GradientAccumulator};
pub use depth_map::{
    depth_map_to_pointcloud, depth_to_disparity, render_depth_map, render_depth_maps, DepthCamera,
    DepthMap, DepthMapError, DepthMapStats, DepthMode, DepthSample, GaussianDepthData,
};
pub use depth_of_field::{
    // Full pipeline (renamed to avoid collision with dof::apply_dof)
    apply_depth_of_field as apply_bokeh_depth_of_field,
    dof_bilinear_sample,
    dof_circular_kernel,
    dof_composite_layers,
    // dof_-prefixed functions (no collision risk)
    dof_compute_coc,
    dof_compute_stats,
    dof_format_config,
    dof_format_stats,
    dof_gather,
    dof_hexagonal_kernel,
    dof_make_kernel,
    dof_pentagonal_kernel,
    dof_separate_layers,
    dof_square_kernel,
    // Non-conflicting types
    BokehShape,
    DofCocBuffer,
    // Rename conflicting types to avoid collision with dof::{DofConfig, DofError, DofStats}
    DofConfig as BokehDofConfig,
    DofError as BokehDofError,
    DofResult,
    DofStats as BokehDofStats,
};
pub use dof::{
    apply_dof, compute_coc, generate_dof_kernel, DofConfig, DofError, DofKernelShape, DofStats,
};
pub use edge_detection::{
    build_edge_gaussian_kernel, canny_edges, compute_edge_stats, detect_edges_log,
    detect_edges_sobel, edge_convolve, edge_gaussian_blur, hysteresis_threshold, log_edges,
    non_max_suppress, prewitt_edges, rgba_to_luminance, sobel_edges, CannyConfig,
    EdgeDetectionError, EdgeMap, EdgeStats, SobelResult,
};
pub use environment::{
    evaluate_gaussian_sh, project_environment_to_sh, sh_basis_up_to_l2, EnvironmentMap,
    SphericalHarmonicsLight,
};
pub use exposure::{
    apply_ev, apply_exposure as apply_exposure_u8, apply_gamma_correction, apply_lift_gain_offset,
    auto_expose, clahe, compute_histogram, compute_luminance_map, exposure_bracket,
    histogram_equalize, hsl_to_rgb, meter_exposure, rgb_to_hsl, scene_key, ExposureConfig,
    ExposureError, ExposureMetering, LuminanceHistogram,
};
pub use film_grain::{
    apply_film_grain, apply_film_grain_frame, apply_film_grain_rgba, apply_film_grain_sequence,
    compute_grain_stats, film_luminance, generate_grain_map, grain_scale_fn, FilmGrainConfig,
    FilmGrainError, GrainStats,
};
pub use gaussian::{GaussianAttributes, GaussianModel};
pub use gaussian_culling::{
    compute_cull_stats, compute_screen_bounds, cull_gaussians, project_radius_ndc, transform_point,
    CullStats, CullingConfig, CullingError, CullingResult, FrustumPlane, GaussianCullData,
    ScreenSpaceBounds, ViewFrustum,
};
pub use gaussian_stats::{
    compute_scalar_stats, detect_anomalies, format_gaussian_report, GaussianHistograms,
    GaussianStats, Histogram, ScalarStats,
};
pub use init::{GaussianInitConfig, GaussianInitializer};
pub use lod::{
    opacity_scale_from_radius, AdaptiveLodConfig, GaussianLodInfo, LodCamera, LodConfig, LodError,
    LodFilter, LodLevel, LodManager, LodSelector, LodStats, LodTransition,
};
pub use mb_pipeline::{
    // apply_motion_blur renamed to avoid conflict with motion_blur_image::apply_motion_blur
    apply_motion_blur as mb_apply_full_pipeline,
    mb_apply,
    mb_bilinear_sample,
    mb_compute_stats,
    mb_depth_weight,
    mb_dilate_velocity,
    mb_format_config,
    mb_format_stats,
    mb_jitter_samples,
    mb_smooth_velocity,
    mb_triangle_weight,
    mb_velocity_from_camera_motion,
    mb_velocity_from_depth,
    mb_velocity_rotational,
    MbConfig,
    MbStats,
    MotionBlurResult,
    VelocityBuffer,
};
pub use mip_splatting::{
    adjust_scales_for_mip, compute_ewa_filter, compute_filter_variance, compute_mip_level,
    compute_mip_levels_batch, compute_mip_stats, FilterMode as MipFilterMode, MipCamera, MipConfig,
    MipSplattingError, MipSplattingStats,
};
pub use model_pruning::{
    apply_pruning_mask, compute_pruning_mask, prune_gaussian_arrays, ImportanceScorer,
    PrunedArrays, PruningConfig, PruningCriteria, PruningError, PruningResult, BYTES_PER_GAUSSIAN,
};
pub use motion_blur::{
    accumulate_frames, apply_velocity_blur, compute_motion_stats, lerp_position, lerp_quaternion,
    normalize_quaternion, subframe_t, AccumulatedBlur, AccumulationConfig, CameraMotion,
    MotionBlurError, MotionBlurStats, VelocityBlurConfig, VelocityField,
};
pub use motion_blur_image::{
    accumulate_motion_blur, apply_linear_motion_blur, apply_motion_blur,
    apply_per_pixel_motion_blur, apply_radial_blur, apply_rotational_blur,
    compute_image_motion_stats, compute_motion_from_frames, format_motion_stats, BlurType,
    ImageMotionField, MotionBlurConfig, MotionStats,
};
pub use multi_view::{MultiViewConfig, MultiViewRenderer};
pub use normal_estimation::{
    compute_normal_stats, estimate_curvature, estimate_normals_cross_product,
    estimate_normals_sobel, normal_consistency_loss, smooth_normals, NormalError, NormalMap,
    NormalStats,
};
pub use panoramic::{
    camera_from_angles, compute_panoramic_stats, direction_to_equirect, equirect_to_cube_face,
    equirect_to_cubemap, equirect_to_direction, fibonacci_sphere_views, perspective_to_equirect,
    stitch_to_equirect, CubeFace, PanoramicCamera, PanoramicError, PanoramicStats, PerspectiveView,
};
pub use picking::{
    compute_pick_stats, pick_all, pick_closest_approach, pick_nearest, pick_region,
    ray_point_distance, ray_sphere_intersect, GaussianPickData, PickCamera, PickConfig, PickHit,
    PickStats, PickingError, Ray, RegionPickResult,
};
pub use pool::{BufferPool, PoolStats, PooledBuffer};
pub use post_processing::{
    // apply_bloom renamed to avoid collision with bloom::apply_bloom
    apply_bloom as post_apply_bloom,
    apply_chromatic_aberration,
    apply_sharpening,
    apply_vignette,
    // BloomConfig renamed to avoid collision with bloom::BloomConfig
    BloomConfig as PostBloomConfig,
    ChromaticAberrationConfig,
    PostEffect,
    PostImage,
    PostProcessError,
    PostProcessingPipeline,
    SharpenConfig,
    VignetteConfig,
};
pub use profiler::{PassProfiler, PassRecord, PassStats, ProfileScope};
pub use rasterizer::{GaussianGradients, Rasterizer, RenderCamera, RenderOutput};
pub use render_graph::{
    build_standard_3dgs_graph, CompiledRenderGraph, PassDesc, PassType, RenderGraph,
    RenderGraphError, ResourceDesc, ResourceFormat, ResourceId, ResourceLifetime,
};
pub use render_metrics::{
    compare_renders, compute_ms_ssim, compute_psnr, compute_ssim, MetricThresholds,
    RenderQualityMetrics,
};
pub use scene_compositor::{
    apply_trimap_matting, blend_pixels, composite_layers, compute_composite_stats, dilate_mask,
    erode_mask, feather_mask, mask_and, mask_not, mask_or, BlendMode as CompositorBlendMode,
    CompositeStats, CompositorError, Layer, RgbaImage,
};
pub use sharpening::{
    adaptive_sharpen,
    apply_sharpening as apply_sharpen_filter,
    compute_sharpening_stats,
    gaussian_blur_rgb as sharpen_blur_rgb,
    gaussian_kernel_1d as sharpen_kernel_1d,
    high_boost_filter,
    high_pass_sharpen,
    laplacian_sharpen,
    laplacian_variance,
    local_contrast_enhance,
    richardson_lucy_sharpen,
    sobel_magnitude_map,
    unsharp_mask,
    // Renamed to avoid collision with post_processing::{SharpenConfig, apply_sharpening}
    // and bloom::{gaussian_kernel_1d, gaussian_blur_rgb}.
    SharpenConfig as SharpenFilterConfig,
    SharpenMethod,
    SharpenStats,
    SharpeningError,
};
pub use silhouette::{
    render_silhouette, render_silhouette_antialiased, render_silhouettes, silhouette_bce_loss,
    silhouette_iou, GaussianSilData, SilhouetteCamera, SilhouetteError, SilhouetteMask,
    SilhouetteMode, SilhouetteStats,
};
pub use spherical_harmonics::{
    color_to_sh_dc, rotate_sh_coeffs_z, sh_band_energy, sh_dc_to_color, sh_downsample, sh_eval,
    sh_eval_degree0, sh_eval_degree1, sh_eval_degree2, sh_eval_degree3, sh_num_coeffs,
    sh_project_monte_carlo, sh_to_color, sh_upsample, ShBasis, ShError, SH_C0, SH_C1, SH_C2, SH_C3,
};
pub use ssao::{
    apply_ao_to_image, blur_ssao, compute_ssao, generate_noise_texture, SsaoConfig, SsaoError,
    SsaoKernel, SsaoStats,
};
pub use stereo::{
    compose_anaglyph, compose_side_by_side, compose_top_bottom, compute_disparity_sad,
    compute_stereo_stats, disparity_to_depth, disparity_to_image, split_side_by_side, AnaglyphMode,
    EyeOffsetMode, StereoConfig, StereoError, StereoImage, StereoStats,
};
pub use temporal::{
    MotionVector, MotionVectorField, TaaConfig, TaaError, TemporalAccumulator, TemporalStats,
};
pub use temporal_aa::{
    accumulate_taa, clip_to_variance, compute_taa_stats, halton, jitter_offset, local_color_stats,
    sharpen_image, TaaAccumulator, TaaHistory, TaaStats,
};
pub use tile_stats::{
    compute_tile_stats, HeatmapMode, TileAnalysisReport, TileHeatmap, TileStats, TileStatsError,
    TileStatsGrid,
};
pub use tone_curve::{
    analyze_curve,
    apply_bezier_curve,
    apply_channel_curves,
    apply_lut_256,
    apply_tone_curve,
    blend_curves,
    compose_curves,
    compute_tangents,
    curve_contrast,
    histogram_tone_curve,
    invert_curve,
    monotone_cubic_interp,
    BezierCurve,
    ChannelCurves,
    CurveStats,
    // ToneCurve renamed to avoid collision with color_grading::ToneCurve
    ToneCurve as ParametricToneCurve,
    ToneCurveError,
};
pub use validation::*;
pub use workgroup::{
    WorkgroupBenchResult, WorkgroupBenchmarker, WorkgroupConfig, WorkgroupProfile, WorkgroupSize,
};
pub mod subsurface_scatter;
pub use subsurface_scatter::{
    sss_apply_profile, sss_apply_transmittance, sss_blur_2d, sss_blur_depth_aware,
    sss_blur_horizontal, sss_blur_vertical, sss_compute_stats, sss_format_config,
    sss_format_material, sss_format_stats, sss_gaussian_kernel_1d, sss_integrate_irradiance,
    sss_monte_carlo_integral, sss_radial_irradiance, sss_transmittance, DiffusionProfile,
    SssConfig, SssError, SssMaterial, SssResult, SssStats,
};
pub mod volumetric_render;
pub use volumetric_render::{
    vr_build_occupancy_grid, vr_can_skip, vr_compute_stats, vr_format_config, vr_format_stats,
    vr_gaussians_to_volume, vr_march_ray, vr_ray_aabb_intersect, vr_render_image,
    vr_render_image_u8, RayMarchResult, TransferFunction, TransferPoint, VolumeGrid,
    VolumetricCamera, VolumetricIntegration, VolumetricRay, VolumetricRenderConfig,
    VolumetricRenderError, VolumetricStats,
};

pub use bloom::{
    apply_bloom,
    apply_hdr_bloom,
    apply_hdr_bloom_and_tonemap,
    apply_hdr_bloom_rgba,
    bloom_accumulate_mip_chain,
    bloom_build_mip_chain,
    bloom_composite,
    bloom_convolve_horizontal,
    bloom_convolve_vertical,
    bloom_coverage,
    bloom_downsample,
    bloom_extract_bright,
    bloom_gaussian_blur,
    bloom_luminance_map,
    bloom_make_kernel,
    bloom_upsample,
    build_mip_pyramid,
    compute_hdr_bloom_stats,
    downsample_2x,
    extract_bright,
    format_bloom_config,
    gaussian_blur_rgb,
    gaussian_kernel_1d,
    soft_knee_weight,
    upsample_2x,
    BloomConfig,
    // New simplified bloom API
    BloomError,
    HdrBloomConfig,
    HdrBloomError,
    HdrBloomStats,
};
pub use chromatic_aberration::{
    aberration_map,
    apply_barrel_distortion,
    apply_barrel_distortion_rgba,
    apply_chrom_aberration,
    apply_chromatic_aberration as apply_full_chromatic_aberration,
    apply_chromatic_aberration_rgba,
    apply_lateral_ca,
    apply_lens_effects,
    apply_longitudinal_ca,
    apply_radial_chromatic_aberration,
    apply_vignette_effect,
    apply_vignetting_f32,
    bilinear_sample_channel as chrom_bilinear_sample,
    compute_chromatic_stats,
    compute_lens_effect_stats,
    compute_radial_distances,
    compute_vignette_mask,
    estimate_aberration_strength,
    format_chrom_config,
    fringe_visualization,
    nearest_sample_channel as chrom_nearest_sample,
    radial_factor,
    shift_channel,
    undistort_pixel,
    ChromAberrationConfig,
    // f32 RGB post-processing API
    ChromAberrationError,
    ChromInterpolation,
    ChromaticError,
    ChromaticStats,
    DistortionConfig,
    LateralChromaticConfig,
    // RGBA u8 lens-effects API (re-exported via chromatic_aberration with renamed aliases)
    LensEffectError,
    LensEffectStats,
    LensEffectsConfig,
    LensVignetteConfig,
    LongitudinalChromaticConfig,
    RgbaChromAberrationConfig,
};
pub use lens_distortion::{
    barrel_distort_image, compute_distortion_map, compute_distortion_stats, distort_image,
    distort_point, estimate_radial_coefficients, remap_image, undistort_image,
    undistort_point_iterative, CameraIntrinsics, DistortionModel, DistortionStats,
    LensDistortionError,
};
pub use vignetting::{
    animated_vignetting_strength, apply_vignetting, apply_vignetting_rgba,
    apply_vignetting_sequence, compute_vignetting_stats, generate_vignetting_mask, pixel_radius,
    VignettingConfig, VignettingError, VignettingModel, VignettingStats,
};

use thiserror::Error;

/// Reason for GPU device being lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceLostReason {
    /// Device was explicitly destroyed.
    Destroyed,
    /// Unknown reason.
    Unknown,
    /// Device was disconnected (e.g., GPU unplugged).
    DeviceDisconnected,
    /// Driver was updated while rendering.
    DriverUpdate,
    /// Out of GPU memory.
    OutOfMemory,
}

impl std::fmt::Display for DeviceLostReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Destroyed => write!(f, "device destroyed"),
            Self::Unknown => write!(f, "unknown reason"),
            Self::DeviceDisconnected => write!(f, "device disconnected"),
            Self::DriverUpdate => write!(f, "driver update"),
            Self::OutOfMemory => write!(f, "out of memory"),
        }
    }
}

/// Errors that can occur during rendering operations.
#[derive(Debug, Error)]
pub enum RenderError {
    // --- GPU initialization ---
    /// General GPU initialization error.
    #[error("GPU initialization error: {0}")]
    GpuInit(String),

    /// No suitable GPU adapter found.
    #[error("No suitable GPU adapter found")]
    AdapterNotFound,

    /// Failed to create GPU device.
    #[error("Failed to create GPU device: {0}")]
    DeviceCreationFailed(String),

    /// GPU device was lost during operation.
    #[error("GPU device lost ({reason}): {message}")]
    DeviceLost {
        /// Reason for device loss.
        reason: DeviceLostReason,
        /// Additional message.
        message: String,
    },

    // --- Shader compilation ---
    /// Shader compilation failed.
    #[error("Shader compilation failed for '{shader_name}': {error}")]
    ShaderCompilation {
        /// Name of the shader that failed to compile.
        shader_name: String,
        /// Compilation error message.
        error: String,
    },

    /// Shader validation failed.
    #[error("Shader validation failed for '{shader_name}': {error}")]
    ShaderValidation {
        /// Name of the shader that failed validation.
        shader_name: String,
        /// Validation error message.
        error: String,
    },

    // --- Buffer operations ---
    /// Buffer allocation failed.
    #[error("Buffer allocation failed for '{buffer_name}' (requested {requested_size} bytes)")]
    BufferAllocation {
        /// Name of the buffer.
        buffer_name: String,
        /// Requested size in bytes.
        requested_size: u64,
    },

    /// Buffer overflow.
    #[error("Buffer overflow for '{buffer_name}' (max: {max_size}, requested: {requested})")]
    BufferOverflow {
        /// Name of the buffer.
        buffer_name: String,
        /// Maximum size in bytes.
        max_size: u64,
        /// Requested size in bytes.
        requested: u64,
    },

    /// Buffer mapping failed.
    #[error("Buffer map failed for '{buffer_name}': {error}")]
    BufferMapFailed {
        /// Name of the buffer.
        buffer_name: String,
        /// Error message.
        error: String,
    },

    // --- Rasterization ---
    /// General rasterization error.
    #[error("Rasterization error: {0}")]
    Rasterize(String),

    /// Compute dispatch limit exceeded.
    #[error("Dispatch limit exceeded for {dimension}: requested {requested}, max {max}")]
    DispatchLimitExceeded {
        /// Which dimension (x, y, or z).
        dimension: String,
        /// Requested workgroup count.
        requested: u32,
        /// Maximum allowed.
        max: u32,
    },

    /// Too many Gaussians for allocated buffers.
    #[error("Too many Gaussians: {count} exceeds maximum {max}")]
    TooManyGaussians {
        /// Actual count.
        count: u32,
        /// Maximum allowed.
        max: u32,
    },

    /// Too many tile-Gaussian pairs.
    #[error("Too many tile pairs: {count} exceeds allocated {allocated}")]
    TooManyTilePairs {
        /// Actual count.
        count: u32,
        /// Allocated capacity.
        allocated: u32,
    },

    // --- Validation ---
    /// Invalid Gaussian data.
    #[error("Invalid Gaussian at index {index}: {reason}")]
    InvalidGaussian {
        /// Index of the invalid Gaussian.
        index: usize,
        /// Reason for invalidity.
        reason: String,
    },

    /// Invalid quaternion (not normalized or zero).
    #[error("Invalid quaternion at index {index}: norm = {norm}")]
    InvalidQuaternion {
        /// Index of the Gaussian with invalid quaternion.
        index: usize,
        /// Norm of the quaternion.
        norm: f32,
    },

    /// Invalid scale values.
    #[error("Invalid scale at index {index}: values = {values:?}")]
    InvalidScale {
        /// Index of the Gaussian with invalid scale.
        index: usize,
        /// Scale values.
        values: [f32; 3],
    },

    /// Mismatched buffer sizes.
    #[error("Mismatched buffer sizes: expected {expected}, got {actual}")]
    MismatchedBufferSizes {
        /// Expected size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },

    // --- I/O ---
    /// PLY I/O error.
    #[error("PLY I/O error: {0}")]
    PlyIo(String),

    /// SafeTensors I/O error.
    #[error("SafeTensors I/O error: {0}")]
    SafetensorsIo(String),

    /// Image save failed.
    #[error("Image save failed: {0}")]
    ImageSaveFailed(String),

    /// Gradient readback failed.
    #[error("Gradient readback failed: {0}")]
    GradientReadbackFailed(String),

    /// Channel receive error.
    #[error("Channel receive error: {0}")]
    ChannelRecvError(String),

    /// Camera path error.
    #[error("Camera path error: {0}")]
    CameraPath(String),

    /// General validation error (NaN/Inf checks, count mismatches, etc.)
    #[error("Validation error: {0}")]
    ValidationError(String),
}

impl From<wgpu::RequestDeviceError> for RenderError {
    fn from(err: wgpu::RequestDeviceError) -> Self {
        RenderError::DeviceCreationFailed(err.to_string())
    }
}

impl From<std::sync::mpsc::RecvError> for RenderError {
    fn from(err: std::sync::mpsc::RecvError) -> Self {
        RenderError::ChannelRecvError(err.to_string())
    }
}

impl From<image::ImageError> for RenderError {
    fn from(err: image::ImageError) -> Self {
        RenderError::ImageSaveFailed(err.to_string())
    }
}
