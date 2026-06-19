//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Configuration for Mip-Splatting anti-aliasing.
///
/// Three presets are available:
/// - [`MipSplatConfig::default`] — balanced settings suitable for most scenes.
/// - [`MipSplatConfig::aggressive`] — stronger filtering, trades more opacity
///   reduction for fewer aliasing artifacts on distant Gaussians.
/// - [`MipSplatConfig::conservative`] — minimal filtering, preserves more
///   detail at the cost of some sub-pixel aliasing.
#[derive(Debug, Clone)]
pub struct MipSplatConfig {
    /// Minimum 2-D Gaussian extent in pixels (clamps projected size from below).
    ///
    /// Gaussians whose projected screen-space radius is smaller than this value
    /// will have their 3-D scale multiplied by a compensation factor so that
    /// their screen-space radius reaches this minimum.  Default: `0.3`.
    pub min_2d_radius_px: f32,
    /// Maximum distance-based scale multiplier (currently reserved for future
    /// distance-LOD extensions).  Default: `1.0`.
    pub max_distance_scale: f32,
    /// Opacity ramp start: Gaussians with screen radius below this value will
    /// have their opacity linearly reduced.  Default: `0.5` px.
    pub opacity_ramp_min_px: f32,
    /// Opacity ramp end: Gaussians with screen radius at or above this value
    /// retain full opacity.  Default: `2.0` px.
    pub opacity_ramp_max_px: f32,
    /// Whether to apply distance-based LOD scaling (reserved for future use).
    pub use_distance_lod: bool,
    /// Reference distance at which 3-D scales are "correct" (no modification
    /// applied).  Default: `1.0`.
    pub reference_distance: f32,
}
impl MipSplatConfig {
    /// Aggressive preset — stronger filtering for aliasing-prone scenes.
    ///
    /// Applies a larger minimum projected radius and a wider opacity ramp,
    /// trading higher opacity reduction for fewer aliasing artifacts.
    pub fn aggressive() -> Self {
        Self {
            min_2d_radius_px: 0.5,
            opacity_ramp_min_px: 0.3,
            opacity_ramp_max_px: 3.0,
            ..Self::default()
        }
    }
    /// Conservative preset — minimal filtering for detail-preserving use.
    ///
    /// Uses a small minimum projected radius and a narrow opacity ramp,
    /// preserving fine Gaussian detail at the cost of some sub-pixel aliasing.
    pub fn conservative() -> Self {
        Self {
            min_2d_radius_px: 0.1,
            opacity_ramp_min_px: 0.1,
            opacity_ramp_max_px: 1.0,
            ..Self::default()
        }
    }
}
/// Statistics collected by `apply_antialiasing` describing how
/// anti-aliasing affected each Gaussian in the model.
#[derive(Debug, Clone)]
pub struct AliasingStats {
    /// Total number of Gaussians processed.
    pub num_gaussians: usize,
    /// Gaussians that needed a scale compensation (projected radius < minimum).
    pub num_scaled_up: usize,
    /// Gaussians whose opacity was reduced (screen radius in the ramp zone).
    pub num_faded: usize,
    /// Gaussians whose opacity was driven to 0 (screen radius below ramp start).
    pub num_culled: usize,
    /// Mean scale compensation factor across all Gaussians (1.0 = no change).
    pub mean_scale_compensation: f32,
    /// Mean fractional opacity reduction: `1.0 - opacity_scale`.
    pub mean_opacity_reduction: f32,
}
/// Quality-improvement statistics returned by `aa_quality_estimate`.
#[derive(Debug, Clone)]
pub struct AaStats {
    /// Number of edge pixels in the original image.
    pub edge_pixels_original: usize,
    /// Number of edge pixels remaining after anti-aliasing.
    pub edge_pixels_after: usize,
    /// Fraction of edge pixels that were smoothed: `1 - after/original`.
    pub smoothing_ratio: f32,
    /// Mean absolute per-channel difference between original and AA output.
    pub mean_difference: f32,
    /// Maximum per-channel difference between original and AA output.
    pub max_difference: f32,
}
/// Which anti-aliasing algorithm to use.
#[derive(Debug, Clone)]
pub enum AaMethod {
    /// Fast Approximate Anti-Aliasing (luminance-space edge blending).
    Fxaa,
    /// Simplified Morphological Anti-Aliasing (Roberts-cross edge detection).
    Smaa,
    /// Temporal AA: blend current frame with a previous frame.
    Temporal {
        /// Weight given to the previous frame `[0, 1]`.
        blend_factor: f32,
    },
    /// Box-filter downsampling (factor must be 2 or 4).
    Supersampling {
        /// Downsampling factor (`2` or `4`).
        factor: u32,
    },
}
/// Errors from image-based anti-aliasing operations.
#[derive(Debug, thiserror::Error)]
pub enum AaError {
    /// Buffer length does not match width × height × channels.
    #[error("Image size {got} != {width}×{height}×{channels}")]
    SizeMismatch {
        got: usize,
        width: usize,
        height: usize,
        channels: usize,
    },
    /// A configuration parameter is out of its valid range.
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
    /// Image is smaller than the 4×4 minimum required for AA passes.
    #[error("Image too small: {width}×{height} (minimum 4×4)")]
    ImageTooSmall { width: usize, height: usize },
}
/// Configuration for image-space anti-aliasing passes.
#[derive(Debug, Clone)]
pub struct AaConfig {
    /// Algorithm to apply.
    pub method: AaMethod,
    /// FXAA: minimum relative edge contrast to process (default `0.0833`).
    pub edge_threshold: f32,
    /// FXAA: absolute luminance floor for edge detection (default `0.0312`).
    pub edge_threshold_min: f32,
    /// FXAA: subpixel AA strength in `[0, 1]` (default `0.75`).
    pub subpixel_quality: f32,
    /// FXAA: maximum search steps when walking along an edge (default `12`).
    pub search_steps: usize,
}
