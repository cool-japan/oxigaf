//! HDR tone mapping for 3DGS rendered images.
//!
//! Converts high-dynamic-range float images (values may exceed \[0,1\]) to
//! low-dynamic-range displayable images with values in \[0,1\].
//!
//! This is essential when rendering 3DGS scenes with physically-based lighting
//! or when accumulating multi-view images.
//!
//! # Operators
//! - **Reinhard** / **ReinhardExtended**: simple photographic tone mapping
//! - **ACES Filmic**: cinematic S-curve approximation (Narkowicz 2015)
//! - **Hable / Uncharted 2**: game-proven filmic curve
//! - **Filmic**: John Hable's simpler film simulation
//! - **Linear**: passthrough with \[0,1\] clamp
//! - **Custom**: parameterised shadow/midtone/highlight split

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by HDR tone mapping operations.
#[derive(Debug, Error)]
pub enum ToneMappingError {
    /// Image slice has incorrect length.
    #[error("Invalid image: {0}")]
    InvalidImage(String),

    /// A configuration parameter is out of valid range.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// The image slice is empty (no pixels to process).
    #[error("Empty image")]
    EmptyImage,

    /// Buffer size does not match width×height×channels.
    #[error("Image size mismatch: buffer {got} != {width}x{height}x{channels}")]
    SizeMismatch {
        /// Actual buffer length.
        got: usize,
        /// Image width.
        width: usize,
        /// Image height.
        height: usize,
        /// Number of channels.
        channels: usize,
    },

    /// A named parameter is invalid.
    #[error("Invalid parameter {name}: {reason}")]
    InvalidParam {
        /// Parameter name.
        name: String,
        /// Reason the value is invalid.
        reason: String,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Primitive tone mapping operators (per-channel scalar functions)
// ─────────────────────────────────────────────────────────────────────────────

/// Simple Reinhard: `out = x / (1 + x)`.
///
/// Maps `[0, ∞)` to `[0, 1)`.
#[inline]
pub fn reinhard(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    x / (1.0 + x)
}

/// Extended Reinhard: `out = x * (1 + x/white²) / (1 + x)`.
///
/// `white` is the luminance that should map to exactly 1.  Values of `white`
/// ≤ 0 are treated as 1 to avoid division by zero.
#[inline]
pub fn reinhard_extended(x: f32, white: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let w = if white > 0.0 { white } else { 1.0 };
    let numer = x * (1.0 + x / (w * w));
    let denom = 1.0 + x;
    (numer / denom).clamp(0.0, 1.0)
}

/// ACES filmic curve approximation (Narkowicz 2015).
///
/// `f(x) = (x * (a*x + b)) / (x * (c*x + d) + e)`
/// with `a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14`.
///
/// Output is clamped to `[0, 1]`.
#[inline]
pub fn aces_filmic(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let a = 2.51_f32;
    let b = 0.03_f32;
    let c = 2.43_f32;
    let d = 0.59_f32;
    let e = 0.14_f32;
    let numer = x * (a * x + b);
    let denom = x * (c * x + d) + e;
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    (numer / denom).clamp(0.0, 1.0)
}

/// Internal Hable partial function `U(x)`.
///
/// `U(x) = ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F)) - E/F`
/// with `A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30`.
#[inline]
fn hable_partial(x: f32) -> f32 {
    let a = 0.15_f32;
    let b = 0.50_f32;
    let c = 0.10_f32;
    let d = 0.20_f32;
    let e = 0.02_f32;
    let f = 0.30_f32;
    let numer = x * (a * x + c * b) + d * e;
    let denom = x * (a * x + b) + d * f;
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    numer / denom - e / f
}

/// Hable "Uncharted 2" filmic tone mapping.
///
/// `result = U(exposure * x) / U(W)` where `W = 11.2` is the white point.
///
/// Output is clamped to `[0, 1]`.
#[inline]
pub fn hable(x: f32, exposure: f32) -> f32 {
    const WHITE: f32 = 11.2;
    let w_partial = hable_partial(WHITE);
    if w_partial.abs() < 1e-12 {
        return 0.0;
    }
    let mapped = hable_partial(exposure * x) / w_partial;
    mapped.clamp(0.0, 1.0)
}

/// Simplified filmic curve (John Hable).
///
/// Step 1: `x = max(0, x - 0.004)`
/// Step 2: `y = (x*(6.2*x + 0.5)) / (x*(6.2*x + 1.7) + 0.06)`
///
/// Negative inputs map to 0. No output clamping is necessary; the formula
/// is naturally bounded in `[0, 1]` for non-negative input.
#[inline]
pub fn filmic(x: f32) -> f32 {
    let x = (x - 0.004_f32).max(0.0);
    if x < 1e-12 {
        return 0.0;
    }
    let numer = x * (6.2 * x + 0.5);
    let denom = x * (6.2 * x + 1.7) + 0.06;
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    (numer / denom).clamp(0.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Gamma / sRGB
// ─────────────────────────────────────────────────────────────────────────────

/// Apply gamma correction: `out = clamp(x, 0, 1) ^ (1 / gamma)`.
///
/// `gamma` values ≤ 0 are clamped to a small positive value to avoid
/// division by zero.
#[inline]
pub fn gamma_correct(x: f32, gamma: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    let g = gamma.max(1e-6);
    x.powf(1.0 / g).clamp(0.0, 1.0)
}

/// Proper piecewise sRGB gamma encoding (IEC 61966-2-1).
///
/// - `x <= 0.0031308`: `12.92 * x`
/// - `x > 0.0031308`: `1.055 * x^(1/2.4) - 0.055`
///
/// Input is clamped to `[0, 1]` before encoding.
#[inline]
pub fn srgb_gamma(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.003_130_8_f32 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// Inverse sRGB gamma (linearise): sRGB encoded → linear light.
///
/// - `x <= 0.04045`: `x / 12.92`
/// - `x > 0.04045`: `((x + 0.055) / 1.055)^2.4`
///
/// Input is clamped to `[0, 1]` before decoding.
#[inline]
pub fn inverse_srgb_gamma(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.04045_f32 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Exposure adjustment
// ─────────────────────────────────────────────────────────────────────────────

/// Apply EV exposure to every channel in the image slice.
///
/// `output[i] = input[i] * 2^stops`
///
/// No clamping is applied; the returned values may exceed `[0, 1]`.
pub fn apply_exposure(image: &[f32], stops: f32) -> Vec<f32> {
    let scale = (2.0_f32).powf(stops);
    image.iter().map(|&v| v * scale).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// ToneMappingOperator enum
// ─────────────────────────────────────────────────────────────────────────────

/// Available HDR tone mapping operators.
#[derive(Debug, Clone)]
pub enum ToneMappingOperator {
    /// Simple Reinhard: `x / (1 + x)`.
    Reinhard,

    /// Extended Reinhard with configurable white point.
    ReinhardExtended {
        /// Luminance value that maps to 1.0.
        white: f32,
    },

    /// ACES filmic approximation (Narkowicz 2015); output clamped to \[0,1\].
    AcesFilmic,

    /// Hable "Uncharted 2" filmic curve with configurable pre-exposure.
    Hable {
        /// Pre-exposure multiplier (applied before the curve, not in EV stops).
        exposure: f32,
    },

    /// Simplified John Hable filmic S-curve.
    Filmic,

    /// Pass-through: only clamp to `[0, 1]`.
    Linear,

    /// Custom parameterised shadow/midtone/highlight operator.
    Custom {
        /// Power applied in the shadow region.
        shadow_gamma: f32,
        /// Linear scale applied in the midtone region.
        midtone_scale: f32,
        /// Roll-off strength in the highlight region (> 1).
        highlight_rolloff: f32,
    },
}

impl ToneMappingOperator {
    /// Apply this operator to a single HDR channel value.
    ///
    /// Returns a value nominally in `[0, 1]`.
    pub fn apply_channel(&self, x: f32) -> f32 {
        match self {
            ToneMappingOperator::Reinhard => reinhard(x),
            ToneMappingOperator::ReinhardExtended { white } => reinhard_extended(x, *white),
            ToneMappingOperator::AcesFilmic => aces_filmic(x),
            ToneMappingOperator::Hable { exposure } => hable(x, *exposure),
            ToneMappingOperator::Filmic => filmic(x),
            ToneMappingOperator::Linear => x.clamp(0.0, 1.0),
            ToneMappingOperator::Custom {
                shadow_gamma,
                midtone_scale,
                highlight_rolloff,
            } => {
                let x_adj = x * midtone_scale;
                if x_adj <= 0.0 {
                    0.0
                } else if x_adj < 1.0 {
                    x_adj.powf(*shadow_gamma).clamp(0.0, 1.0)
                } else {
                    // Soft highlight roll-off: 1 - (1 - 1/x_adj) * highlight_rolloff
                    // Continuous at x_adj=1: gives exactly 1.0
                    let rolloff = 1.0 - (1.0 - 1.0 / x_adj) * highlight_rolloff;
                    rolloff.clamp(0.0, 1.0)
                }
            }
        }
    }

    /// Human-readable name for this operator.
    pub fn name(&self) -> &str {
        match self {
            ToneMappingOperator::Reinhard => "reinhard",
            ToneMappingOperator::ReinhardExtended { .. } => "reinhard_extended",
            ToneMappingOperator::AcesFilmic => "aces_filmic",
            ToneMappingOperator::Hable { .. } => "hable",
            ToneMappingOperator::Filmic => "filmic",
            ToneMappingOperator::Linear => "linear",
            ToneMappingOperator::Custom { .. } => "custom",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ToneMappingConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Full tone mapping pipeline configuration.
#[derive(Debug, Clone)]
pub struct ToneMappingConfig {
    /// The tone mapping operator to apply.
    pub operator: ToneMappingOperator,

    /// Pre-tone-mapping exposure in EV stops.
    pub exposure_stops: f32,

    /// Post-tone-mapping gamma correction exponent.
    ///
    /// `1.0` means no correction; `2.2` approximates sRGB.
    /// This is ignored when `use_srgb_gamma` is `true`.
    pub gamma: f32,

    /// If `true`, apply the proper piecewise sRGB curve instead of simple
    /// power-law gamma.
    pub use_srgb_gamma: bool,

    /// Channel saturation multiplier applied after tone mapping.
    ///
    /// `1.0` = neutral; `0.0` = greyscale; `> 1.0` = boosted saturation.
    pub saturation: f32,
}

impl Default for ToneMappingConfig {
    fn default() -> Self {
        Self {
            operator: ToneMappingOperator::AcesFilmic,
            exposure_stops: 0.0,
            gamma: 1.0,
            use_srgb_gamma: false,
            saturation: 1.0,
        }
    }
}

impl ToneMappingConfig {
    /// Validate configuration parameters.
    ///
    /// Returns [`ToneMappingError::InvalidConfig`] if any parameter is out of
    /// its valid range.
    pub fn validate(&self) -> Result<(), ToneMappingError> {
        if self.gamma <= 0.0 {
            return Err(ToneMappingError::InvalidConfig(format!(
                "gamma must be > 0, got {}",
                self.gamma
            )));
        }
        if self.saturation < 0.0 {
            return Err(ToneMappingError::InvalidConfig(format!(
                "saturation must be >= 0, got {}",
                self.saturation
            )));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Image-level tone mapping
// ─────────────────────────────────────────────────────────────────────────────

/// Apply tone mapping to a flat RGB image (row-major, 3 f32 values per pixel).
///
/// Returns an LDR image with values in `[0, 1]`.
///
/// # Steps per pixel
/// 1. Apply exposure (`2^stops`)
/// 2. Apply tone mapping operator per channel
/// 3. Apply saturation (luminance-based interpolation)
/// 4. Apply gamma or sRGB gamma
/// 5. Clamp to `[0, 1]`
///
/// # Errors
/// - [`ToneMappingError::EmptyImage`] if `image` is empty
/// - [`ToneMappingError::InvalidImage`] if `image.len()` is not a multiple of 3
/// - [`ToneMappingError::InvalidConfig`] if config fails validation
pub fn tone_map_image(
    image: &[f32],
    config: &ToneMappingConfig,
) -> Result<Vec<f32>, ToneMappingError> {
    if image.is_empty() {
        return Err(ToneMappingError::EmptyImage);
    }
    if !image.len().is_multiple_of(3) {
        return Err(ToneMappingError::InvalidImage(format!(
            "RGB image length must be a multiple of 3, got {}",
            image.len()
        )));
    }
    config.validate()?;

    let exposure_scale = (2.0_f32).powf(config.exposure_stops);
    let mut out = Vec::with_capacity(image.len());

    for pixel in image.chunks_exact(3) {
        // 1. Exposure
        let r = pixel[0] * exposure_scale;
        let g = pixel[1] * exposure_scale;
        let b = pixel[2] * exposure_scale;

        // 2. Tone mapping operator
        let r = config.operator.apply_channel(r);
        let g = config.operator.apply_channel(g);
        let b = config.operator.apply_channel(b);

        // 3. Saturation (luminance-preserving)
        let (r, g, b) = apply_saturation_rgb(r, g, b, config.saturation);

        // 4. Gamma / sRGB curve
        let (r, g, b) = if config.use_srgb_gamma {
            (srgb_gamma(r), srgb_gamma(g), srgb_gamma(b))
        } else if (config.gamma - 1.0).abs() > 1e-6 {
            (
                gamma_correct(r, config.gamma),
                gamma_correct(g, config.gamma),
                gamma_correct(b, config.gamma),
            )
        } else {
            (r, g, b)
        };

        // 5. Clamp
        out.push(r.clamp(0.0, 1.0));
        out.push(g.clamp(0.0, 1.0));
        out.push(b.clamp(0.0, 1.0));
    }

    Ok(out)
}

/// Apply tone mapping to a flat RGBA image (4 f32 values per pixel).
///
/// The alpha channel is passed through unchanged (not clamped or modified).
///
/// # Errors
/// Same as [`tone_map_image`] except the length check requires a multiple of 4.
pub fn tone_map_rgba_image(
    image: &[f32],
    config: &ToneMappingConfig,
) -> Result<Vec<f32>, ToneMappingError> {
    if image.is_empty() {
        return Err(ToneMappingError::EmptyImage);
    }
    if !image.len().is_multiple_of(4) {
        return Err(ToneMappingError::InvalidImage(format!(
            "RGBA image length must be a multiple of 4, got {}",
            image.len()
        )));
    }
    config.validate()?;

    let exposure_scale = (2.0_f32).powf(config.exposure_stops);
    let mut out = Vec::with_capacity(image.len());

    for pixel in image.chunks_exact(4) {
        let alpha = pixel[3];

        // 1. Exposure
        let r = pixel[0] * exposure_scale;
        let g = pixel[1] * exposure_scale;
        let b = pixel[2] * exposure_scale;

        // 2. Tone mapping operator
        let r = config.operator.apply_channel(r);
        let g = config.operator.apply_channel(g);
        let b = config.operator.apply_channel(b);

        // 3. Saturation
        let (r, g, b) = apply_saturation_rgb(r, g, b, config.saturation);

        // 4. Gamma / sRGB curve
        let (r, g, b) = if config.use_srgb_gamma {
            (srgb_gamma(r), srgb_gamma(g), srgb_gamma(b))
        } else if (config.gamma - 1.0).abs() > 1e-6 {
            (
                gamma_correct(r, config.gamma),
                gamma_correct(g, config.gamma),
                gamma_correct(b, config.gamma),
            )
        } else {
            (r, g, b)
        };

        // 5. Clamp RGB, passthrough alpha
        out.push(r.clamp(0.0, 1.0));
        out.push(g.clamp(0.0, 1.0));
        out.push(b.clamp(0.0, 1.0));
        out.push(alpha);
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helper: saturation
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a saturation multiplier to an RGB triplet.
///
/// Computes BT.709 luminance, then interpolates each channel between the
/// luminance value and its original:
/// `channel_out = luma + saturation * (channel - luma)`
#[inline]
fn apply_saturation_rgb(r: f32, g: f32, b: f32, saturation: f32) -> (f32, f32, f32) {
    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let r_out = luma + saturation * (r - luma);
    let g_out = luma + saturation * (g - luma);
    let b_out = luma + saturation * (b - luma);
    (r_out, g_out, b_out)
}

// ─────────────────────────────────────────────────────────────────────────────
// HDR statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics summarising the dynamic range of an HDR image.
#[derive(Debug, Clone)]
pub struct HdrStats {
    /// Minimum channel value across the image.
    pub min_value: f32,
    /// Maximum channel value across the image.
    pub max_value: f32,
    /// Mean luminance (BT.709) across all pixels.
    pub mean_luminance: f32,
    /// Log-mean luminance: `exp(mean(log(L + 1e-6)))`.
    pub log_mean_luminance: f32,
    /// 99th percentile of per-pixel luminance values.
    pub percentile_99: f32,
    /// Fraction of pixels whose maximum channel exceeds 1.
    pub fraction_clipped: f32,
    /// Dynamic range in EV: `log2(max / min)`.
    pub dynamic_range_ev: f32,
}

/// Compute HDR image statistics from a flat RGB image (multiple-of-3 length).
///
/// # Errors
/// - [`ToneMappingError::EmptyImage`] if the slice is empty
/// - [`ToneMappingError::InvalidImage`] if `image.len()` is not a multiple of 3
pub fn compute_hdr_stats(image: &[f32]) -> Result<HdrStats, ToneMappingError> {
    if image.is_empty() {
        return Err(ToneMappingError::EmptyImage);
    }
    if !image.len().is_multiple_of(3) {
        return Err(ToneMappingError::InvalidImage(format!(
            "RGB image length must be a multiple of 3, got {}",
            image.len()
        )));
    }

    let n_pixels = image.len() / 3;

    let mut min_value = f32::INFINITY;
    let mut max_value = f32::NEG_INFINITY;
    let mut sum_luma = 0.0_f64;
    let mut sum_log_luma = 0.0_f64;
    let mut clipped_count = 0_usize;
    let mut luminances = Vec::with_capacity(n_pixels);

    for pixel in image.chunks_exact(3) {
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];

        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        luminances.push(luma);

        sum_luma += luma as f64;
        sum_log_luma += (luma + 1e-6_f32).ln() as f64;

        let min_ch = r.min(g).min(b);
        let max_ch = r.max(g).max(b);

        if min_ch < min_value {
            min_value = min_ch;
        }
        if max_ch > max_value {
            max_value = max_ch;
        }
        if max_ch > 1.0 {
            clipped_count += 1;
        }
    }

    let mean_luminance = (sum_luma / n_pixels as f64) as f32;
    let log_mean_luminance = (sum_log_luma / n_pixels as f64).exp() as f32;
    let fraction_clipped = clipped_count as f32 / n_pixels as f32;

    // 99th percentile
    luminances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p99_idx = if n_pixels > 1 {
        (n_pixels - 1) * 99 / 100
    } else {
        0
    };
    let percentile_99 = luminances[p99_idx];

    // Dynamic range in EV — guard against zero or negative min
    let safe_min = min_value.max(1e-10);
    let safe_max = max_value.max(safe_min);
    let dynamic_range_ev = (safe_max / safe_min).log2().clamp(0.0, 100.0);

    Ok(HdrStats {
        min_value,
        max_value,
        mean_luminance,
        log_mean_luminance,
        percentile_99,
        fraction_clipped,
        dynamic_range_ev,
    })
}

/// Recommend an exposure adjustment (in EV stops) using the key-value method.
///
/// `stops = log2(0.18 / log_mean_luminance)`, clamped to `[-10, 10]`.
pub fn recommend_exposure(stats: &HdrStats) -> f32 {
    let lml = stats.log_mean_luminance.max(1e-6);
    let stops = (0.18_f32 / lml).log2();
    stops.clamp(-10.0, 10.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Presets
// ─────────────────────────────────────────────────────────────────────────────

/// Simple Reinhard tone mapping with neutral gamma.
pub fn preset_reinhard() -> ToneMappingConfig {
    ToneMappingConfig {
        operator: ToneMappingOperator::Reinhard,
        exposure_stops: 0.0,
        gamma: 1.0,
        use_srgb_gamma: false,
        saturation: 1.0,
    }
}

/// ACES filmic tone mapping with proper sRGB gamma encoding.
pub fn preset_aces() -> ToneMappingConfig {
    ToneMappingConfig {
        operator: ToneMappingOperator::AcesFilmic,
        exposure_stops: 0.0,
        gamma: 1.0,
        use_srgb_gamma: true,
        saturation: 1.0,
    }
}

/// Simplified filmic curve with neutral gamma.
pub fn preset_filmic() -> ToneMappingConfig {
    ToneMappingConfig {
        operator: ToneMappingOperator::Filmic,
        exposure_stops: 0.0,
        gamma: 1.0,
        use_srgb_gamma: false,
        saturation: 1.0,
    }
}

/// Hable "Uncharted 2" filmic curve with a slight saturation boost.
///
/// Simulates a photographic look suitable for outdoor 3DGS scenes.
pub fn preset_photography() -> ToneMappingConfig {
    ToneMappingConfig {
        operator: ToneMappingOperator::Hable { exposure: 2.0 },
        exposure_stops: 0.0,
        gamma: 1.0,
        use_srgb_gamma: true,
        saturation: 1.1,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// New operator enum (ToneMapOperator) — distinct from ToneMappingOperator
// ─────────────────────────────────────────────────────────────────────────────

/// Simple HDR tone mapping operators for use with [`apply_operator`] and
/// [`tone_map`] / [`tone_map_inplace`].
#[derive(Debug, Clone)]
pub enum ToneMapOperator {
    /// Global Reinhard: `Y / (1 + Y)` per luminance, preserving hue.
    Reinhard,
    /// Extended Reinhard with a configurable maximum luminance.
    ReinhardExtended {
        /// Maximum luminance value (maps to 1.0).
        max_luminance: f32,
    },
    /// Hable / Uncharted-2 filmic curve (exposure bias = 2.0, white = 11.2).
    Filmic,
    /// ACES filmic approximation (Narkowicz 2015), per-channel.
    Aces,
    /// Timothy Lottes' tone mapper approximation (per-channel).
    Lottes,
    /// Simple exposure adjustment: `channel * 2^stops`.
    Exposure {
        /// EV stops to apply.
        stops: f32,
    },
    /// Linear rescale from `[min, max]` to `[0, 1]`.
    Linear {
        /// Input minimum.
        min: f32,
        /// Input maximum.
        max: f32,
    },
}

/// Configuration for the new [`tone_map`] / [`tone_map_inplace`] pipeline.
#[derive(Debug, Clone)]
pub struct ToneMapConfig {
    /// The tone mapping operator.
    pub operator: ToneMapOperator,
    /// Display gamma exponent (default 2.2).
    pub gamma: f32,
    /// Whether to apply gamma correction after tone mapping.
    pub apply_gamma: bool,
    /// Whether to clip output to `[0, 1]`.
    pub clip: bool,
}

impl Default for ToneMapConfig {
    fn default() -> Self {
        Self {
            operator: ToneMapOperator::Aces,
            gamma: 2.2,
            apply_gamma: true,
            clip: true,
        }
    }
}

/// Per-channel and luminance statistics for an HDR image.
#[derive(Debug, Clone)]
pub struct HdrImageStats {
    /// Minimum luminance across all pixels.
    pub min_luminance: f32,
    /// Maximum luminance across all pixels.
    pub max_luminance: f32,
    /// Arithmetic mean luminance.
    pub mean_luminance: f32,
    /// Log-average luminance: `exp(mean(log(L + 1e-6)))`.
    pub log_mean_luminance: f32,
    /// Estimated scene key value (same as `log_mean_luminance`).
    pub key_value: f32,
    /// Ratio `max / (min + 1e-6)`.
    pub dynamic_range: f32,
    /// Mean value of each channel `[R, G, B]`.
    pub per_channel_mean: [f32; 3],
    /// Maximum value of each channel `[R, G, B]`.
    pub per_channel_max: [f32; 3],
}

// ─────────────────────────────────────────────────────────────────────────────
// Lottes approximation
// ─────────────────────────────────────────────────────────────────────────────

/// Lottes tone mapping approximation: `x^a / (x^a + 1)` with `a = 1.6`.
#[inline]
pub fn lottes_approx(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let a = 1.6_f32;
    let xa = x.powf(a);
    (xa / (xa + 1.0)).clamp(0.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_operator — single RGB triple
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a [`ToneMapOperator`] to a single RGB triple `(r, g, b)`.
///
/// Returns the tone-mapped `(r, g, b)`.  Values may still exceed `[0, 1]` for
/// the `Exposure` variant — clip with [`ToneMapConfig::clip`] if needed.
pub fn apply_operator(r: f32, g: f32, b: f32, op: &ToneMapOperator) -> (f32, f32, f32) {
    match op {
        ToneMapOperator::Reinhard => {
            let lum = tone_luminance(r, g, b);
            if lum < 1e-10 {
                return (0.0, 0.0, 0.0);
            }
            let lum_out = lum / (1.0 + lum);
            let scale = lum_out / lum;
            (r * scale, g * scale, b * scale)
        }
        ToneMapOperator::ReinhardExtended { max_luminance } => {
            let lum = tone_luminance(r, g, b);
            if lum < 1e-10 {
                return (0.0, 0.0, 0.0);
            }
            let w = max_luminance.max(1e-6);
            let lum_out = (lum * (1.0 + lum / (w * w))) / (1.0 + lum);
            let scale = (lum_out / lum).clamp(0.0, 10.0);
            (
                (r * scale).clamp(0.0, 1.0),
                (g * scale).clamp(0.0, 1.0),
                (b * scale).clamp(0.0, 1.0),
            )
        }
        ToneMapOperator::Filmic => {
            // Hable/Uncharted 2: apply per-channel with exposure bias = 2.0
            (hable(r, 2.0), hable(g, 2.0), hable(b, 2.0))
        }
        ToneMapOperator::Aces => (aces_filmic(r), aces_filmic(g), aces_filmic(b)),
        ToneMapOperator::Lottes => (lottes_approx(r), lottes_approx(g), lottes_approx(b)),
        ToneMapOperator::Exposure { stops } => {
            let scale = (2.0_f32).powf(*stops);
            (r * scale, g * scale, b * scale)
        }
        ToneMapOperator::Linear { min, max } => {
            let range = (max - min).abs() + 1e-7;
            let map = |v: f32| (v - min) / range;
            (map(r), map(g), map(b))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_gamma / apply_gamma_image
// ─────────────────────────────────────────────────────────────────────────────

/// Apply display gamma encoding: `value.max(0.0).powf(1.0 / gamma)`.
#[inline]
pub fn apply_gamma(value: f32, gamma: f32) -> f32 {
    let g = gamma.max(1e-6);
    value.max(0.0).powf(1.0 / g)
}

/// Apply gamma correction to every element of an image slice.
pub fn apply_gamma_image(img: &[f32], gamma: f32) -> Vec<f32> {
    img.iter().map(|&v| apply_gamma(v, gamma)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// tone_luminance — BT.709 luminance (avoids conflict with color_grading::luminance)
// ─────────────────────────────────────────────────────────────────────────────

/// BT.709 luminance: `0.2126*r + 0.7152*g + 0.0722*b`.
///
/// Named `tone_luminance` to avoid conflict with the re-exported
/// `color_grading::luminance` in the crate root.
#[inline]
pub fn tone_luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

// ─────────────────────────────────────────────────────────────────────────────
// luminance_histogram
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a 256-bin luminance histogram for a flat RGB image.
///
/// Luminance values are computed with BT.709 coefficients and clamped to
/// `[0, 1]` before binning.
///
/// # Errors
/// - [`ToneMappingError::EmptyImage`] if `img` is empty
/// - [`ToneMappingError::SizeMismatch`] if `img.len() != width * height * 3`
pub fn luminance_histogram(
    img: &[f32],
    width: usize,
    height: usize,
) -> Result<Vec<u32>, ToneMappingError> {
    let expected = width * height * 3;
    if img.is_empty() {
        return Err(ToneMappingError::EmptyImage);
    }
    if img.len() != expected {
        return Err(ToneMappingError::SizeMismatch {
            got: img.len(),
            width,
            height,
            channels: 3,
        });
    }
    let mut bins = vec![0u32; 256];
    for pixel in img.chunks_exact(3) {
        let lum = tone_luminance(pixel[0], pixel[1], pixel[2]).clamp(0.0, 1.0);
        let bin = (lum * 255.0).round() as usize;
        let bin = bin.min(255);
        bins[bin] = bins[bin].saturating_add(1);
    }
    Ok(bins)
}

// ─────────────────────────────────────────────────────────────────────────────
// estimate_scene_key
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the scene key value (log-average luminance).
///
/// `key = exp(mean(log(L_i + δ)))` where `δ = 1e-6`.
///
/// # Errors
/// - [`ToneMappingError::EmptyImage`] if `img` is empty
/// - [`ToneMappingError::SizeMismatch`] if `img.len() != width * height * 3`
pub fn estimate_scene_key(
    img: &[f32],
    width: usize,
    height: usize,
) -> Result<f32, ToneMappingError> {
    let expected = width * height * 3;
    if img.is_empty() {
        return Err(ToneMappingError::EmptyImage);
    }
    if img.len() != expected {
        return Err(ToneMappingError::SizeMismatch {
            got: img.len(),
            width,
            height,
            channels: 3,
        });
    }
    let n_pixels = width * height;
    let delta = 1e-6_f32;
    let mut sum_log = 0.0_f64;
    for pixel in img.chunks_exact(3) {
        let lum = tone_luminance(pixel[0], pixel[1], pixel[2]);
        sum_log += (lum + delta).ln() as f64;
    }
    let log_avg = (sum_log / n_pixels as f64).exp() as f32;
    Ok(log_avg)
}

// ─────────────────────────────────────────────────────────────────────────────
// hdr_image_stats
// ─────────────────────────────────────────────────────────────────────────────

/// Compute comprehensive statistics for an HDR image.
///
/// # Errors
/// - [`ToneMappingError::EmptyImage`] if `img` is empty
/// - [`ToneMappingError::SizeMismatch`] if `img.len() != width * height * 3`
pub fn hdr_image_stats(
    img: &[f32],
    width: usize,
    height: usize,
) -> Result<HdrImageStats, ToneMappingError> {
    let expected = width * height * 3;
    if img.is_empty() {
        return Err(ToneMappingError::EmptyImage);
    }
    if img.len() != expected {
        return Err(ToneMappingError::SizeMismatch {
            got: img.len(),
            width,
            height,
            channels: 3,
        });
    }
    let n_pixels = (width * height) as f64;
    let delta = 1e-6_f32;

    let mut min_lum = f32::INFINITY;
    let mut max_lum = f32::NEG_INFINITY;
    let mut sum_lum = 0.0_f64;
    let mut sum_log_lum = 0.0_f64;
    let mut sum_ch = [0.0_f64; 3];
    let mut max_ch = [f32::NEG_INFINITY; 3];

    for pixel in img.chunks_exact(3) {
        let r = pixel[0];
        let g = pixel[1];
        let b = pixel[2];
        let lum = tone_luminance(r, g, b);
        if lum < min_lum {
            min_lum = lum;
        }
        if lum > max_lum {
            max_lum = lum;
        }
        sum_lum += lum as f64;
        sum_log_lum += (lum + delta).ln() as f64;
        sum_ch[0] += r as f64;
        sum_ch[1] += g as f64;
        sum_ch[2] += b as f64;
        if r > max_ch[0] {
            max_ch[0] = r;
        }
        if g > max_ch[1] {
            max_ch[1] = g;
        }
        if b > max_ch[2] {
            max_ch[2] = b;
        }
    }

    let mean_luminance = (sum_lum / n_pixels) as f32;
    let log_mean_luminance = (sum_log_lum / n_pixels).exp() as f32;
    let dynamic_range = max_lum / (min_lum + 1e-6);

    Ok(HdrImageStats {
        min_luminance: min_lum,
        max_luminance: max_lum,
        mean_luminance,
        log_mean_luminance,
        key_value: log_mean_luminance,
        dynamic_range,
        per_channel_mean: [
            (sum_ch[0] / n_pixels) as f32,
            (sum_ch[1] / n_pixels) as f32,
            (sum_ch[2] / n_pixels) as f32,
        ],
        per_channel_max: max_ch,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// auto_exposure
// ─────────────────────────────────────────────────────────────────────────────

/// Compute exposure stops to map the scene's log-average luminance to 0.18
/// (middle grey).
///
/// `stops = log2(0.18 / scene_key)`
///
/// # Errors
/// Same as [`estimate_scene_key`].
pub fn auto_exposure(img: &[f32], width: usize, height: usize) -> Result<f32, ToneMappingError> {
    let key = estimate_scene_key(img, width, height)?;
    let target = 0.18_f32;
    let stops = (target / key.max(1e-10)).log2();
    Ok(stops.clamp(-20.0, 20.0))
}

// ─────────────────────────────────────────────────────────────────────────────
// hdr_white_balance
// ─────────────────────────────────────────────────────────────────────────────

/// Apply white balance by multiplying each channel by its scale factor.
///
/// The output is not clamped; apply clipping separately if needed.
/// Named `hdr_white_balance` to avoid conflict with `colorspace::apply_white_balance`.
pub fn hdr_white_balance(img: &[f32], r_scale: f32, g_scale: f32, b_scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(img.len());
    for pixel in img.chunks_exact(3) {
        out.push(pixel[0] * r_scale);
        out.push(pixel[1] * g_scale);
        out.push(pixel[2] * b_scale);
    }
    // Handle any trailing elements that don't form a complete pixel
    let full_pixels = (img.len() / 3) * 3;
    for &v in &img[full_pixels..] {
        out.push(v);
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// hdr_linear_to_srgb / srgb_to_linear_hdr
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a single linear-light value to sRGB (proper piecewise IEC 61966-2-1).
///
/// Named `hdr_linear_to_srgb` to avoid conflict with `colorspace::linear_to_srgb`.
#[inline]
pub fn hdr_linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8_f32 {
        12.92 * v
    } else {
        (1.055 * v.powf(1.0 / 2.4) - 0.055).clamp(0.0, 1.0)
    }
}

/// Convert entire image from linear light to sRGB encoding.
pub fn image_hdr_linear_to_srgb(img: &[f32]) -> Vec<f32> {
    img.iter().map(|&v| hdr_linear_to_srgb(v)).collect()
}

/// Convert a single sRGB-encoded value to linear light.
#[inline]
pub fn srgb_to_linear_hdr(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.04045_f32 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// tone_map / tone_map_inplace
// ─────────────────────────────────────────────────────────────────────────────

/// Apply tone mapping to a linear HDR image (flat `H×W×3` f32 slice).
///
/// Input values may exceed 1.0 (HDR).  With default config, output is in
/// `[0, 1]` (LDR).
///
/// # Errors
/// - [`ToneMappingError::EmptyImage`] if `img` is empty
/// - [`ToneMappingError::SizeMismatch`] if `img.len() != width * height * 3`
/// - [`ToneMappingError::InvalidParam`] if gamma is non-positive
pub fn tone_map(
    img: &[f32],
    width: usize,
    height: usize,
    config: &ToneMapConfig,
) -> Result<Vec<f32>, ToneMappingError> {
    let expected = width * height * 3;
    if img.is_empty() {
        return Err(ToneMappingError::EmptyImage);
    }
    if img.len() != expected {
        return Err(ToneMappingError::SizeMismatch {
            got: img.len(),
            width,
            height,
            channels: 3,
        });
    }
    if config.gamma <= 0.0 {
        return Err(ToneMappingError::InvalidParam {
            name: "gamma".to_string(),
            reason: format!("must be > 0, got {}", config.gamma),
        });
    }
    let mut out = Vec::with_capacity(img.len());
    for pixel in img.chunks_exact(3) {
        let (mut r, mut g, mut b) = apply_operator(pixel[0], pixel[1], pixel[2], &config.operator);
        if config.apply_gamma {
            r = apply_gamma(r, config.gamma);
            g = apply_gamma(g, config.gamma);
            b = apply_gamma(b, config.gamma);
        }
        if config.clip {
            r = r.clamp(0.0, 1.0);
            g = g.clamp(0.0, 1.0);
            b = b.clamp(0.0, 1.0);
        }
        out.push(r);
        out.push(g);
        out.push(b);
    }
    Ok(out)
}

/// Apply tone mapping in-place to a linear HDR image.
///
/// # Errors
/// Same as [`tone_map`].
pub fn tone_map_inplace(
    img: &mut [f32],
    width: usize,
    height: usize,
    config: &ToneMapConfig,
) -> Result<(), ToneMappingError> {
    let mapped = tone_map(img, width, height, config)?;
    img.copy_from_slice(&mapped);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// format_tone_config
// ─────────────────────────────────────────────────────────────────────────────

/// Format a [`ToneMapConfig`] as a human-readable string.
pub fn format_tone_config(config: &ToneMapConfig) -> String {
    let op = match &config.operator {
        ToneMapOperator::Reinhard => "reinhard".to_string(),
        ToneMapOperator::ReinhardExtended { max_luminance } => {
            format!("reinhard_extended(max_lum={max_luminance})")
        }
        ToneMapOperator::Filmic => "filmic".to_string(),
        ToneMapOperator::Aces => "aces".to_string(),
        ToneMapOperator::Lottes => "lottes".to_string(),
        ToneMapOperator::Exposure { stops } => format!("exposure(stops={stops})"),
        ToneMapOperator::Linear { min, max } => format!("linear(min={min},max={max})"),
    };
    format!(
        "ToneMapConfig {{ op={op}, gamma={}, apply_gamma={}, clip={} }}",
        config.gamma, config.apply_gamma, config.clip
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    // ── 1. reinhard: 0 → 0, large x → approaches 1 ──────────────────────────
    #[test]
    fn test_reinhard_zero() {
        assert!((reinhard(0.0) - 0.0).abs() < EPS);
    }

    #[test]
    fn test_reinhard_large_approaches_one() {
        let out = reinhard(1_000_000.0_f32);
        assert!(out > 0.999_99, "expected > 0.99999, got {out}");
        assert!(out <= 1.0, "expected <= 1.0, got {out}");
    }

    // ── 2. reinhard: 1.0 → 0.5 ───────────────────────────────────────────────
    #[test]
    fn test_reinhard_one_half() {
        let out = reinhard(1.0);
        assert!((out - 0.5).abs() < EPS, "expected 0.5, got {out}");
    }

    // ── 3. reinhard_extended: white point semantics ───────────────────────────
    #[test]
    fn test_reinhard_extended_at_white() {
        // At x=white the extended Reinhard gives (w*(1+1))/(1+w) = 2w/(1+w).
        // For w=1: 2/2 = 1.0 exactly (maps white → 1).
        let out = reinhard_extended(1.0, 1.0);
        assert!(out <= 1.0 + EPS, "expected <= 1, got {out}");
        assert!(out > 0.0, "expected > 0, got {out}");
    }

    // ── 4. aces_filmic: 0 → 0, 1 → reasonable ────────────────────────────────
    #[test]
    fn test_aces_filmic_zero() {
        assert!((aces_filmic(0.0) - 0.0).abs() < EPS);
    }

    #[test]
    fn test_aces_filmic_one_in_range() {
        let out = aces_filmic(1.0);
        assert!(out > 0.0 && out <= 1.0, "expected in (0,1], got {out}");
    }

    // ── 5. aces_filmic: output always clamped to [0,1] ───────────────────────
    #[test]
    fn test_aces_filmic_clamped() {
        for &v in &[-1.0_f32, 0.0, 0.5, 1.0, 10.0, 100.0] {
            let out = aces_filmic(v);
            assert!(
                (0.0..=1.0).contains(&out),
                "aces_filmic({v}) = {out} out of [0,1]"
            );
        }
    }

    // ── 6. hable: 0 → 0, large → approaches 1 ────────────────────────────────
    #[test]
    fn test_hable_zero() {
        assert!((hable(0.0, 2.0) - 0.0).abs() < EPS);
    }

    #[test]
    fn test_hable_large_approaches_one() {
        let out = hable(1000.0, 2.0);
        assert!(out > 0.9, "expected > 0.9, got {out}");
        assert!(out <= 1.0, "expected <= 1.0, got {out}");
    }

    // ── 7. filmic: negative input → 0 ────────────────────────────────────────
    #[test]
    fn test_filmic_negative_is_zero() {
        assert!((filmic(-5.0) - 0.0).abs() < EPS);
        assert!((filmic(-0.001) - 0.0).abs() < EPS);
    }

    // ── 8. gamma_correct: x=0 → 0, x=1 → 1 ──────────────────────────────────
    #[test]
    fn test_gamma_correct_endpoints() {
        assert!((gamma_correct(0.0, 2.2) - 0.0).abs() < EPS);
        assert!((gamma_correct(1.0, 2.2) - 1.0).abs() < EPS);
    }

    // ── 9. gamma_correct: gamma=2 → sqrt ─────────────────────────────────────
    #[test]
    fn test_gamma_correct_sqrt() {
        let out = gamma_correct(0.25, 2.0);
        let expected = 0.5_f32;
        assert!(
            (out - expected).abs() < EPS,
            "expected {expected}, got {out}"
        );
    }

    // ── 10. srgb_gamma: 0 → 0, 1 → 1 ────────────────────────────────────────
    #[test]
    fn test_srgb_gamma_endpoints() {
        assert!((srgb_gamma(0.0) - 0.0).abs() < EPS);
        assert!((srgb_gamma(1.0) - 1.0).abs() < EPS);
    }

    // ── 11. srgb_gamma: piecewise threshold at 0.0031308 ─────────────────────
    #[test]
    fn test_srgb_gamma_piecewise() {
        // Below threshold: linear branch
        let x_low = 0.001_f32;
        let out_low = srgb_gamma(x_low);
        let expected_low = 12.92 * x_low;
        assert!(
            (out_low - expected_low).abs() < EPS,
            "linear branch: expected {expected_low}, got {out_low}"
        );

        // Above threshold: power branch
        let x_high = 0.5_f32;
        let out_high = srgb_gamma(x_high);
        let expected_high = 1.055 * x_high.powf(1.0 / 2.4) - 0.055;
        assert!(
            (out_high - expected_high).abs() < EPS,
            "power branch: expected {expected_high}, got {out_high}"
        );
    }

    // ── 12. inverse_srgb_gamma: roundtrip ────────────────────────────────────
    #[test]
    fn test_inverse_srgb_gamma_roundtrip() {
        for &x in &[0.0_f32, 0.001, 0.02, 0.2, 0.5, 0.9, 1.0] {
            let encoded = srgb_gamma(x);
            let decoded = inverse_srgb_gamma(encoded);
            assert!(
                (decoded - x).abs() < 1e-4,
                "roundtrip failed for {x}: encoded={encoded}, decoded={decoded}"
            );
        }
    }

    // ── 13. apply_exposure: stops=0 → unchanged ───────────────────────────────
    #[test]
    fn test_apply_exposure_zero_stops() {
        let img = vec![0.1_f32, 0.5, 0.9, 0.3, 0.7, 1.5];
        let out = apply_exposure(&img, 0.0);
        for (o, i) in out.iter().zip(img.iter()) {
            assert!((o - i).abs() < EPS, "expected {i}, got {o}");
        }
    }

    // ── 14. apply_exposure: stops=1 → doubled ────────────────────────────────
    #[test]
    fn test_apply_exposure_one_stop() {
        let img = vec![0.25_f32, 0.5, 1.0];
        let out = apply_exposure(&img, 1.0);
        let expected = [0.5_f32, 1.0, 2.0];
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < EPS, "expected {e}, got {o}");
        }
    }

    // ── 15. tone_map_image: all-zero HDR → all-zero LDR ──────────────────────
    #[test]
    fn test_tone_map_image_all_zero() {
        let img = vec![0.0_f32; 12]; // 4 pixels
        let config = ToneMappingConfig::default();
        let out = tone_map_image(&img, &config).expect("tone mapping should succeed");
        for v in &out {
            assert!(v.abs() < EPS, "expected 0.0, got {v}");
        }
    }

    // ── 16. tone_map_image: HDR > 1 → clamped to [0,1] ───────────────────────
    #[test]
    fn test_tone_map_image_hdr_clamped() {
        let img = vec![5.0_f32, 10.0, 20.0]; // one bright pixel
        let config = ToneMappingConfig::default();
        let out = tone_map_image(&img, &config).expect("tone mapping should succeed");
        for v in &out {
            assert!(
                *v >= 0.0 && *v <= 1.0,
                "output {v} not in [0,1] after tone mapping"
            );
        }
    }

    // ── 17. tone_map_image: wrong length → Err ───────────────────────────────
    #[test]
    fn test_tone_map_image_wrong_length() {
        let img = vec![0.5_f32, 0.5]; // 2 values — not multiple of 3
        let config = ToneMappingConfig::default();
        let result = tone_map_image(&img, &config);
        assert!(
            matches!(result, Err(ToneMappingError::InvalidImage(_))),
            "expected InvalidImage error"
        );
    }

    // ── 18. ToneMappingConfig::validate: gamma=0 → Err ────────────────────────
    #[test]
    fn test_config_validate_gamma_zero() {
        let config = ToneMappingConfig {
            gamma: 0.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(
            matches!(result, Err(ToneMappingError::InvalidConfig(_))),
            "expected InvalidConfig error for gamma=0"
        );
    }

    // ── 19. ToneMappingConfig::validate: saturation=-1 → Err ─────────────────
    #[test]
    fn test_config_validate_saturation_negative() {
        let config = ToneMappingConfig {
            saturation: -1.0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(
            matches!(result, Err(ToneMappingError::InvalidConfig(_))),
            "expected InvalidConfig error for saturation=-1"
        );
    }

    // ── 20. compute_hdr_stats: all-zero → min=max=0 ───────────────────────────
    #[test]
    fn test_compute_hdr_stats_all_zero() {
        let img = vec![0.0_f32; 9]; // 3 pixels
        let stats = compute_hdr_stats(&img).expect("stats should succeed");
        assert!((stats.min_value - 0.0).abs() < EPS);
        assert!((stats.max_value - 0.0).abs() < EPS);
        assert!((stats.mean_luminance - 0.0).abs() < EPS);
    }

    // ── 21. compute_hdr_stats: mixed values → correct percentile ──────────────
    #[test]
    fn test_compute_hdr_stats_percentile() {
        // 100 pixels, luminance 0 through 0.99
        let n = 100_usize;
        let mut img = Vec::with_capacity(n * 3);
        for i in 0..n {
            let v = i as f32 / n as f32;
            img.push(v);
            img.push(v);
            img.push(v);
        }
        let stats = compute_hdr_stats(&img).expect("stats should succeed");
        // 99th percentile index = 99 * 99 / 100 = 98 → luma ≈ 0.98
        assert!(
            stats.percentile_99 >= 0.95 && stats.percentile_99 <= 1.0,
            "unexpected percentile_99 = {}",
            stats.percentile_99
        );
    }

    // ── 22. recommend_exposure: high luminance → negative stops ───────────────
    #[test]
    fn test_recommend_exposure_high_luminance() {
        // Build a bright image so log_mean_luminance >> 0.18
        let img: Vec<f32> = (0..30)
            .map(|i| if i % 3 == 0 { 10.0 } else { 9.0 })
            .collect();
        let stats = compute_hdr_stats(&img).expect("stats should succeed");
        let stops = recommend_exposure(&stats);
        assert!(
            stops < 0.0,
            "expected negative stops for bright image, got {stops}"
        );
    }

    // ── 23. preset_aces: produces a valid config ───────────────────────────────
    #[test]
    fn test_preset_aces_valid() {
        let config = preset_aces();
        config
            .validate()
            .expect("preset_aces config should be valid");
        assert!(config.use_srgb_gamma);
        assert!(matches!(config.operator, ToneMappingOperator::AcesFilmic));
    }

    // ── 24. tone_map_rgba_image: alpha channel is passed through unchanged ─────
    #[test]
    fn test_tone_map_rgba_alpha_passthrough() {
        // Pixels with distinctive alpha values
        let img = vec![
            0.5_f32, 0.3, 0.8, 0.42, // pixel 0, alpha = 0.42
            2.0, 5.0, 0.1, 0.99, // pixel 1, alpha = 0.99 (HDR rgb)
        ];
        let config = ToneMappingConfig::default();
        let out = tone_map_rgba_image(&img, &config).expect("rgba tone mapping should succeed");
        assert_eq!(out.len(), 8);
        // Check alphas are exactly preserved
        assert!((out[3] - 0.42).abs() < EPS, "alpha[0] changed: {}", out[3]);
        assert!((out[7] - 0.99).abs() < EPS, "alpha[1] changed: {}", out[7]);
        // RGB values for pixel 1 must be in [0,1] after tone mapping
        for &v in &[out[4], out[5], out[6]] {
            assert!((0.0..=1.0).contains(&v), "HDR rgb not clamped: {v}");
        }
    }

    // ── 25. tone_map_image: all presets validate and produce LDR output ────────
    #[test]
    fn test_all_presets_produce_ldr() {
        let img: Vec<f32> = (0..30)
            .map(|i| if i % 3 == 0 { 3.5 } else { 0.7 })
            .collect();
        let presets = [
            preset_reinhard(),
            preset_aces(),
            preset_filmic(),
            preset_photography(),
        ];
        for preset in &presets {
            preset.validate().expect("preset should be valid");
            let out = tone_map_image(&img, preset)
                .unwrap_or_else(|e| panic!("preset '{}' failed: {e}", preset.operator.name()));
            for (i, &v) in out.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "preset '{}' output[{i}] = {v} not in [0,1]",
                    preset.operator.name()
                );
            }
        }
    }

    // ── 26. custom operator: continuity at x_adj = 1 ─────────────────────────
    #[test]
    fn test_custom_operator_continuity() {
        let op = ToneMappingOperator::Custom {
            shadow_gamma: 1.0,
            midtone_scale: 1.0,
            highlight_rolloff: 1.0,
        };
        // x = 1 → x_adj = 1 → shadow branch: pow(1, 1) = 1
        let at_one = op.apply_channel(1.0);
        // x slightly > 1 → x_adj slightly > 1 → highlight branch
        let just_above = op.apply_channel(1.0001);
        assert!(
            (at_one - 1.0).abs() < EPS,
            "at x=1 expected 1.0, got {at_one}"
        );
        assert!(
            (just_above - 1.0).abs() < 0.01,
            "just above 1 expected close to 1.0, got {just_above}"
        );
    }

    // ── 27. dynamic_range_ev: positive for non-uniform image ─────────────────
    #[test]
    fn test_dynamic_range_ev_positive() {
        let img = vec![0.001_f32, 0.001, 0.001, 10.0, 10.0, 10.0];
        let stats = compute_hdr_stats(&img).expect("stats");
        assert!(
            stats.dynamic_range_ev > 0.0,
            "expected positive dynamic range, got {}",
            stats.dynamic_range_ev
        );
    }

    // ── 28. tone_map_image: empty → EmptyImage error ─────────────────────────
    #[test]
    fn test_tone_map_image_empty() {
        let config = ToneMappingConfig::default();
        let result = tone_map_image(&[], &config);
        assert!(matches!(result, Err(ToneMappingError::EmptyImage)));
    }

    // ── 29. ToneMappingOperator::name() returns correct strings ───────────────
    #[test]
    fn test_operator_names() {
        assert_eq!(ToneMappingOperator::Reinhard.name(), "reinhard");
        assert_eq!(ToneMappingOperator::AcesFilmic.name(), "aces_filmic");
        assert_eq!(ToneMappingOperator::Filmic.name(), "filmic");
        assert_eq!(ToneMappingOperator::Linear.name(), "linear");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // New tests (30–64): covering the new functions added in this pass
    // ─────────────────────────────────────────────────────────────────────────

    // ── 30. tone_luminance: pure red → 0.2126 ────────────────────────────────
    #[test]
    fn test_tone_luminance_pure_red() {
        let l = tone_luminance(1.0, 0.0, 0.0);
        assert!((l - 0.2126).abs() < EPS, "expected 0.2126, got {l}");
    }

    // ── 31. tone_luminance: pure green → 0.7152 ──────────────────────────────
    #[test]
    fn test_tone_luminance_pure_green() {
        let l = tone_luminance(0.0, 1.0, 0.0);
        assert!((l - 0.7152).abs() < EPS, "expected 0.7152, got {l}");
    }

    // ── 32. tone_luminance: white → 1.0 ──────────────────────────────────────
    #[test]
    fn test_tone_luminance_white() {
        let l = tone_luminance(1.0, 1.0, 1.0);
        assert!((l - 1.0).abs() < EPS, "expected 1.0, got {l}");
    }

    // ── 33. apply_gamma: value=1.0 → 1.0 regardless of gamma ─────────────────
    #[test]
    fn test_apply_gamma_one() {
        let out = apply_gamma(1.0, 2.2);
        assert!((out - 1.0).abs() < EPS, "expected 1.0, got {out}");
    }

    // ── 34. apply_gamma: value=0.0 → 0.0 ─────────────────────────────────────
    #[test]
    fn test_apply_gamma_zero() {
        let out = apply_gamma(0.0, 2.2);
        assert!(out.abs() < EPS, "expected 0.0, got {out}");
    }

    // ── 35. apply_gamma: gamma=2.0, value=0.25 → 0.5 ────────────────────────
    #[test]
    fn test_apply_gamma_half() {
        let out = apply_gamma(0.25, 2.0);
        assert!((out - 0.5).abs() < EPS, "expected 0.5, got {out}");
    }

    // ── 36. hdr_linear_to_srgb: 0 → 0, 1 → 1 ────────────────────────────────
    #[test]
    fn test_hdr_linear_to_srgb_endpoints() {
        assert!(hdr_linear_to_srgb(0.0).abs() < EPS);
        assert!((hdr_linear_to_srgb(1.0) - 1.0).abs() < EPS);
    }

    // ── 37. hdr_linear_to_srgb: piecewise boundary at 0.0031308 ──────────────
    #[test]
    fn test_hdr_linear_to_srgb_piecewise() {
        let x_low = 0.001_f32;
        let out_low = hdr_linear_to_srgb(x_low);
        let expected_low = 12.92 * x_low;
        assert!(
            (out_low - expected_low).abs() < EPS,
            "linear branch: expected {expected_low}, got {out_low}"
        );

        let x_high = 0.5_f32;
        let out_high = hdr_linear_to_srgb(x_high);
        let expected_high = (1.055 * x_high.powf(1.0 / 2.4) - 0.055).clamp(0.0, 1.0);
        assert!(
            (out_high - expected_high).abs() < EPS,
            "power branch: expected {expected_high}, got {out_high}"
        );
    }

    // ── 38. srgb_to_linear_hdr round-trip ─────────────────────────────────────
    #[test]
    fn test_srgb_to_linear_hdr_roundtrip() {
        for &x in &[0.0_f32, 0.001, 0.02, 0.2, 0.5, 0.9, 1.0] {
            let encoded = hdr_linear_to_srgb(x);
            let decoded = srgb_to_linear_hdr(encoded);
            assert!(
                (decoded - x).abs() < 1e-4,
                "round-trip failed for {x}: encoded={encoded}, decoded={decoded}"
            );
        }
    }

    // ── 39. apply_operator Reinhard: black stays black ────────────────────────
    #[test]
    fn test_apply_operator_reinhard_black() {
        let (r, g, b) = apply_operator(0.0, 0.0, 0.0, &ToneMapOperator::Reinhard);
        assert!(r.abs() < EPS && g.abs() < EPS && b.abs() < EPS);
    }

    // ── 40. apply_operator Reinhard: large → approaches (1,1,1) ──────────────
    #[test]
    fn test_apply_operator_reinhard_large() {
        let (r, g, b) = apply_operator(1000.0, 1000.0, 1000.0, &ToneMapOperator::Reinhard);
        assert!(r > 0.999 && g > 0.999 && b > 0.999, "r={r} g={g} b={b}");
    }

    // ── 41. apply_operator ReinhardExtended: max_luminance controls limit ─────
    #[test]
    fn test_apply_operator_reinhard_extended() {
        let op = ToneMapOperator::ReinhardExtended { max_luminance: 2.0 };
        let (r, _g, _b) = apply_operator(2.0, 2.0, 2.0, &op);
        // At lum = max_lum: extended Reinhard gives ≤ 1.0 and > reinhard(1)
        assert!(r <= 1.0 + EPS, "expected <= 1, got {r}");
        assert!(r > 0.0, "expected > 0, got {r}");
    }

    // ── 42. apply_operator Filmic: in plausible [0,1] range for positive inputs
    #[test]
    fn test_apply_operator_filmic_range() {
        for v in [0.0_f32, 0.5, 1.0, 2.0, 5.0] {
            let (r, g, b) = apply_operator(v, v, v, &ToneMapOperator::Filmic);
            assert!((0.0..=1.0).contains(&r), "filmic r={r} for input {v}");
            assert!((0.0..=1.0).contains(&g), "filmic g={g} for input {v}");
            assert!((0.0..=1.0).contains(&b), "filmic b={b} for input {v}");
        }
    }

    // ── 43. apply_operator Aces: clamped to [0,1] ────────────────────────────
    #[test]
    fn test_apply_operator_aces_clamped() {
        for v in [-1.0_f32, 0.0, 0.5, 1.0, 10.0, 100.0] {
            let (r, g, b) = apply_operator(v, v, v, &ToneMapOperator::Aces);
            assert!((0.0..=1.0).contains(&r), "aces r={r} for input {v}");
            assert!((0.0..=1.0).contains(&g), "aces g={g} for input {v}");
            assert!((0.0..=1.0).contains(&b), "aces b={b} for input {v}");
        }
    }

    // ── 44. apply_operator Lottes: monotonically non-decreasing for positive ──
    #[test]
    fn test_apply_operator_lottes_monotonic() {
        let mut prev = 0.0_f32;
        for i in 0..=20u32 {
            let v = i as f32 * 0.5;
            let (r, _g, _b) = apply_operator(v, v, v, &ToneMapOperator::Lottes);
            assert!(
                r >= prev - EPS,
                "Lottes not monotonic at {v}: r={r}, prev={prev}"
            );
            prev = r;
        }
    }

    // ── 45. apply_operator Exposure stops=0 → identity (no clamp applied) ─────
    #[test]
    fn test_apply_operator_exposure_zero_stops() {
        let (r, g, b) = apply_operator(0.5, 0.3, 0.7, &ToneMapOperator::Exposure { stops: 0.0 });
        assert!((r - 0.5).abs() < EPS && (g - 0.3).abs() < EPS && (b - 0.7).abs() < EPS);
    }

    // ── 46. apply_operator Exposure stops=1 → doubled ────────────────────────
    #[test]
    fn test_apply_operator_exposure_one_stop() {
        let (r, g, b) = apply_operator(0.5, 0.25, 0.1, &ToneMapOperator::Exposure { stops: 1.0 });
        assert!((r - 1.0).abs() < EPS, "r={r}");
        assert!((g - 0.5).abs() < EPS, "g={g}");
        assert!((b - 0.2).abs() < EPS, "b={b}");
    }

    // ── 47. apply_operator Linear min=0 max=1 → identity ─────────────────────
    #[test]
    fn test_apply_operator_linear_identity() {
        let op = ToneMapOperator::Linear { min: 0.0, max: 1.0 };
        let (r, g, b) = apply_operator(0.5, 0.3, 0.8, &op);
        assert!((r - 0.5).abs() < 1e-4, "r={r}");
        assert!((g - 0.3).abs() < 1e-4, "g={g}");
        assert!((b - 0.8).abs() < 1e-4, "b={b}");
    }

    // ── 48. apply_operator Linear min=0 max=2 → halves ───────────────────────
    #[test]
    fn test_apply_operator_linear_halves() {
        let op = ToneMapOperator::Linear { min: 0.0, max: 2.0 };
        let (r, _g, _b) = apply_operator(1.0, 0.0, 0.0, &op);
        // (1.0 - 0.0) / (2.0 + 1e-7) ≈ 0.5
        assert!((r - 0.5).abs() < 1e-4, "expected ~0.5, got {r}");
    }

    // ── 49. tone_map: size mismatch error ─────────────────────────────────────
    #[test]
    fn test_tone_map_size_mismatch() {
        let img = vec![0.5_f32; 7]; // 7 ≠ 2*2*3 = 12
        let config = ToneMapConfig::default();
        let result = tone_map(&img, 2, 2, &config);
        assert!(matches!(result, Err(ToneMappingError::SizeMismatch { .. })));
    }

    // ── 50. tone_map: empty image error ──────────────────────────────────────
    #[test]
    fn test_tone_map_empty() {
        let config = ToneMapConfig::default();
        let result = tone_map(&[], 0, 0, &config);
        assert!(matches!(result, Err(ToneMappingError::EmptyImage)));
    }

    // ── 51. tone_map: HDR values → LDR after clipping ────────────────────────
    #[test]
    fn test_tone_map_hdr_to_ldr() {
        let img = vec![5.0_f32, 10.0, 20.0, 3.0, 0.5, 0.1];
        let config = ToneMapConfig::default();
        let out = tone_map(&img, 2, 1, &config).expect("tone_map should succeed");
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "output {v} not in [0,1]");
        }
    }

    // ── 52. tone_map_inplace: same result as tone_map ─────────────────────────
    #[test]
    fn test_tone_map_inplace_matches_tone_map() {
        let img = vec![0.5_f32, 0.8, 2.0, 0.1, 3.0, 0.2];
        let config = ToneMapConfig::default();
        let expected = tone_map(&img, 2, 1, &config).expect("tone_map");
        let mut buf = img.clone();
        tone_map_inplace(&mut buf, 2, 1, &config).expect("tone_map_inplace");
        for (a, b) in buf.iter().zip(expected.iter()) {
            assert!((a - b).abs() < EPS, "inplace={a}, expected={b}");
        }
    }

    // ── 53. apply_gamma_image: all pixels correctly adjusted ──────────────────
    #[test]
    fn test_apply_gamma_image() {
        let img = vec![0.0_f32, 0.25, 1.0];
        let out = apply_gamma_image(&img, 2.0);
        assert!(out[0].abs() < EPS, "expected 0, got {}", out[0]);
        assert!((out[1] - 0.5).abs() < EPS, "expected 0.5, got {}", out[1]);
        assert!((out[2] - 1.0).abs() < EPS, "expected 1.0, got {}", out[2]);
    }

    // ── 54. image_hdr_linear_to_srgb: all pixels processed ───────────────────
    #[test]
    fn test_image_hdr_linear_to_srgb() {
        let img = vec![0.0_f32, 0.5, 1.0, 0.002];
        let out = image_hdr_linear_to_srgb(&img);
        assert_eq!(out.len(), img.len());
        for (i, (&v, &o)) in img.iter().zip(out.iter()).enumerate() {
            let expected = hdr_linear_to_srgb(v);
            assert!(
                (o - expected).abs() < EPS,
                "pixel {i}: expected {expected}, got {o}"
            );
        }
    }

    // ── 55. hdr_white_balance: scales channels correctly ─────────────────────
    #[test]
    fn test_hdr_white_balance_channels() {
        let img = vec![1.0_f32, 1.0, 1.0, 0.5, 0.5, 0.5];
        let out = hdr_white_balance(&img, 1.5, 0.8, 1.0);
        assert!((out[0] - 1.5).abs() < EPS, "R scale: {}", out[0]);
        assert!((out[1] - 0.8).abs() < EPS, "G scale: {}", out[1]);
        assert!((out[2] - 1.0).abs() < EPS, "B scale: {}", out[2]);
        assert!((out[3] - 0.75).abs() < EPS, "R2: {}", out[3]);
        assert!((out[4] - 0.4).abs() < EPS, "G2: {}", out[4]);
        assert!((out[5] - 0.5).abs() < EPS, "B2: {}", out[5]);
    }

    // ── 56. luminance_histogram: all-gray image fills one bin ────────────────
    #[test]
    fn test_luminance_histogram_single_bin() {
        // Gray pixel (0.5, 0.5, 0.5) → lum = 0.5 → bin = round(0.5 * 255) = 128
        let n_pixels = 10_usize;
        let img = vec![0.5_f32; n_pixels * 3];
        let hist = luminance_histogram(&img, n_pixels, 1).expect("histogram");
        assert_eq!(hist.len(), 256);
        let total: u32 = hist.iter().sum();
        assert_eq!(
            total as usize, n_pixels,
            "total bin count should equal n_pixels"
        );
        // Only one bin should be non-zero
        let non_zero: Vec<usize> = hist
            .iter()
            .enumerate()
            .filter(|(_, &v)| v > 0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(non_zero.len(), 1, "expected exactly 1 non-zero bin");
    }

    // ── 57. estimate_scene_key: constant luminance image ─────────────────────
    #[test]
    fn test_estimate_scene_key_constant() {
        let lum = 0.5_f32;
        // gray image: r=g=b=v where v gives lum = 0.5
        // lum = 0.2126*v + 0.7152*v + 0.0722*v = v → v = 0.5
        let img = vec![lum; 3 * 4]; // 4 pixels
        let key = estimate_scene_key(&img, 4, 1).expect("scene key");
        let expected = (lum + 1e-6).ln().exp();
        assert!(
            (key - expected).abs() < 1e-4,
            "expected ~{expected}, got {key}"
        );
    }

    // ── 58. hdr_image_stats: known image ─────────────────────────────────────
    #[test]
    fn test_hdr_image_stats_known() {
        // Two pixels: (1,0,0) lum=0.2126, (0,1,0) lum=0.7152
        let img = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let stats = hdr_image_stats(&img, 2, 1).expect("stats");
        assert!(
            (stats.min_luminance - 0.2126).abs() < EPS,
            "min_lum={}",
            stats.min_luminance
        );
        assert!(
            (stats.max_luminance - 0.7152).abs() < EPS,
            "max_lum={}",
            stats.max_luminance
        );
        let expected_mean = (0.2126 + 0.7152) / 2.0;
        assert!(
            (stats.mean_luminance - expected_mean).abs() < EPS,
            "mean_lum={}",
            stats.mean_luminance
        );
    }

    // ── 59. auto_exposure: scene with lum≈0.18 → stops≈0 ────────────────────
    #[test]
    fn test_auto_exposure_middle_gray() {
        // Build image where luminance ≈ 0.18 = 0.2126*r+0.7152*g+0.0722*b
        // Use gray: r=g=b=v, lum = v. Want lum ≈ 0.18, so v ≈ 0.18
        let img = vec![0.18_f32; 3 * 4];
        let stops = auto_exposure(&img, 4, 1).expect("auto_exposure");
        // scene_key ≈ 0.18 + 1e-6, target = 0.18 → stops ≈ 0
        assert!(stops.abs() < 0.01, "expected stops≈0, got {stops}");
    }

    // ── 60. format_tone_config: non-empty string ──────────────────────────────
    #[test]
    fn test_format_tone_config_nonempty() {
        let config = ToneMapConfig::default();
        let s = format_tone_config(&config);
        assert!(!s.is_empty(), "format_tone_config returned empty string");
        assert!(s.contains("aces"), "expected 'aces' in output: {s}");
    }

    // ── 61. tone_map all-black image ──────────────────────────────────────────
    #[test]
    fn test_tone_map_all_black() {
        let img = vec![0.0_f32; 12]; // 2x2x3
        let config = ToneMapConfig::default();
        let out = tone_map(&img, 2, 2, &config).expect("tone_map");
        for &v in &out {
            assert!(v.abs() < EPS, "expected 0 for black image, got {v}");
        }
    }

    // ── 62. tone_map all-white image → all ones ───────────────────────────────
    #[test]
    fn test_tone_map_all_white() {
        let img = vec![1.0_f32; 3];
        let config = ToneMapConfig {
            operator: ToneMapOperator::Reinhard,
            gamma: 1.0,
            apply_gamma: false,
            clip: true,
        };
        let out = tone_map(&img, 1, 1, &config).expect("tone_map");
        // Reinhard(1.0) = 0.5, not 1.0 — just check in [0,1]
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "v={v}");
        }
    }

    // ── 63. tone_map extreme HDR value=100 ───────────────────────────────────
    #[test]
    fn test_tone_map_extreme_hdr() {
        let img = vec![100.0_f32, 100.0, 100.0];
        let config = ToneMapConfig::default();
        let out = tone_map(&img, 1, 1, &config).expect("tone_map");
        for &v in &out {
            assert!(
                (0.0..=1.0).contains(&v),
                "extreme HDR not mapped to LDR: {v}"
            );
        }
    }

    // ── 64. lottes_approx: monotonically non-decreasing ──────────────────────
    #[test]
    fn test_lottes_approx_monotonic() {
        let mut prev = 0.0_f32;
        for i in 0..=30u32 {
            let x = i as f32 * 0.3;
            let y = lottes_approx(x);
            assert!(
                y >= prev - EPS,
                "lottes_approx not monotonic at {x}: y={y}, prev={prev}"
            );
            prev = y;
        }
    }

    // ── 65. hdr_image_stats: size mismatch error ──────────────────────────────
    #[test]
    fn test_hdr_image_stats_size_mismatch() {
        let img = vec![0.5_f32; 7]; // 7 ≠ 2*2*3 = 12
        let result = hdr_image_stats(&img, 2, 2);
        assert!(matches!(result, Err(ToneMappingError::SizeMismatch { .. })));
    }
}
