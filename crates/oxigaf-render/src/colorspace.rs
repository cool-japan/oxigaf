//! Color space conversion and tone mapping utilities.
//!
//! Provides:
//! - Gamma encoding/decoding (sRGB ↔ linear)
//! - Color space conversions: LinearRgb ↔ sRGB, XYZ D65, CIE L\*a\*b\*, HSV, HSL
//! - Universal `convert_color` routing through LinearRgb pivot
//! - Tone mapping operators: Clamp, Reinhard, ReinhardExtended, ACES, Uncharted2 (Hable)
//! - White balance correction from color temperature (Kelvin)
//! - Batch image operations (HWC layout, channels=3)

/// A color stored as 3 f32 components in some color space.
#[derive(Debug, Clone, Copy)]
pub struct Color {
    /// The three components of the color (interpretation depends on color space).
    pub c: [f32; 3],
}

impl Color {
    /// Construct a color from three component values.
    #[inline]
    pub fn new(c0: f32, c1: f32, c2: f32) -> Self {
        Self { c: [c0, c1, c2] }
    }

    /// Black in any additive/luminance-based color space.
    #[inline]
    pub fn black() -> Self {
        Self { c: [0.0, 0.0, 0.0] }
    }

    /// White in any normalized additive color space.
    #[inline]
    pub fn white() -> Self {
        Self { c: [1.0, 1.0, 1.0] }
    }
}

/// Supported color spaces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorSpace {
    /// Linear light RGB, components in [0, 1].
    LinearRgb,
    /// Gamma-encoded sRGB, components in [0, 1].
    SRgb,
    /// Hue [0, 360), Saturation [0, 1], Value [0, 1].
    Hsv,
    /// Hue [0, 360), Saturation [0, 1], Lightness [0, 1].
    Hsl,
    /// CIE XYZ with D65 illuminant. Roughly [0, 1] for visible colors.
    XyzD65,
    /// CIE L\*a\*b\*: L in [0, 100], a and b roughly in [-128, 127].
    Lab,
}

// ─────────────────────────────────────────────────────────────
// Gamma conversion
// ─────────────────────────────────────────────────────────────

/// Apply sRGB gamma encoding: linear light → sRGB.
///
/// Follows IEC 61966-2-1:
/// - `12.92 * c` for `c <= 0.0031308`
/// - `1.055 * c^(1/2.4) - 0.055` otherwise
///
/// Input is clamped to [0, 1] before encoding.
#[inline]
pub fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Apply sRGB gamma decoding: sRGB → linear light.
///
/// - `c / 12.92` for `c <= 0.04045`
/// - `((c + 0.055) / 1.055)^2.4` otherwise
///
/// Input is clamped to [0, 1] before decoding.
#[inline]
pub fn srgb_to_linear(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Apply sRGB gamma encoding to all three components of a color.
#[inline]
pub fn color_linear_to_srgb(c: Color) -> Color {
    Color::new(
        linear_to_srgb(c.c[0]),
        linear_to_srgb(c.c[1]),
        linear_to_srgb(c.c[2]),
    )
}

/// Apply sRGB gamma decoding to all three components of a color.
#[inline]
pub fn color_srgb_to_linear(c: Color) -> Color {
    Color::new(
        srgb_to_linear(c.c[0]),
        srgb_to_linear(c.c[1]),
        srgb_to_linear(c.c[2]),
    )
}

// ─────────────────────────────────────────────────────────────
// Linear RGB ↔ CIE XYZ D65
// ─────────────────────────────────────────────────────────────

/// Forward matrix: linear sRGB → CIE XYZ D65 (D65 whitepoint, IEC 61966-2-1).
///
/// ```text
/// [0.4124564, 0.3575761, 0.1804375]
/// [0.2126729, 0.7151522, 0.0721750]
/// [0.0193339, 0.1191920, 0.9503041]
/// ```
pub fn linear_rgb_to_xyz(c: Color) -> Color {
    let r = c.c[0];
    let g = c.c[1];
    let b = c.c[2];
    Color::new(
        0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b,
        0.212_672_9 * r + 0.715_152_2 * g + 0.072_175_0 * b,
        0.019_333_9 * r + 0.119_192 * g + 0.950_304_1 * b,
    )
}

/// Inverse matrix: CIE XYZ D65 → linear sRGB.
///
/// ```text
/// [ 3.2404542, -1.5371385, -0.4985314]
/// [-0.9692660,  1.8760108,  0.0415560]
/// [ 0.0556434, -0.2040259,  1.0572252]
/// ```
pub fn xyz_to_linear_rgb(c: Color) -> Color {
    let x = c.c[0];
    let y = c.c[1];
    let z = c.c[2];
    Color::new(
        3.240_454_2 * x - 1.537_138_5 * y - 0.498_531_4 * z,
        -0.969_266 * x + 1.876_010_8 * y + 0.041_556_0 * z,
        0.055_643_4 * x - 0.204_025_9 * y + 1.057_225_2 * z,
    )
}

// ─────────────────────────────────────────────────────────────
// CIE XYZ D65 ↔ CIE L*a*b*
// ─────────────────────────────────────────────────────────────

/// D65 reference white point for CIE L\*a\*b\* computation.
const D65_XN: f32 = 0.950_47;
const D65_YN: f32 = 1.000_00;
const D65_ZN: f32 = 1.088_83;

/// Threshold for the f(t) piecewise function.
/// (6/29)^3 ≈ 0.008856
const LAB_EPSILON: f32 = 0.008_856;
/// (29/6)^2 / 3 ≈ 7.787037
const LAB_KAPPA: f32 = 7.787_037;
/// 4/29
const LAB_DELTA: f32 = 4.0 / 29.0;

/// CIE f(t) function for XYZ → Lab conversion.
#[inline]
fn lab_f(t: f32) -> f32 {
    if t > LAB_EPSILON {
        t.cbrt()
    } else {
        LAB_KAPPA * t + LAB_DELTA
    }
}

/// Inverse CIE f(t) function for Lab → XYZ conversion.
#[inline]
fn lab_f_inv(t: f32) -> f32 {
    let t3 = t * t * t;
    if t3 > LAB_EPSILON {
        t3
    } else {
        (t - LAB_DELTA) / LAB_KAPPA
    }
}

/// Convert CIE XYZ D65 to CIE L\*a\*b\*.
///
/// Reference white: Xn=0.95047, Yn=1.0, Zn=1.08883.
pub fn xyz_to_lab(c: Color) -> Color {
    let fx = lab_f(c.c[0] / D65_XN);
    let fy = lab_f(c.c[1] / D65_YN);
    let fz = lab_f(c.c[2] / D65_ZN);
    Color::new(116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

/// Convert CIE L\*a\*b\* to CIE XYZ D65.
pub fn lab_to_xyz(c: Color) -> Color {
    let l = c.c[0];
    let a = c.c[1];
    let b = c.c[2];
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    Color::new(
        D65_XN * lab_f_inv(fx),
        D65_YN * lab_f_inv(fy),
        D65_ZN * lab_f_inv(fz),
    )
}

// ─────────────────────────────────────────────────────────────
// Linear RGB ↔ HSV
// ─────────────────────────────────────────────────────────────

/// Convert linear RGB to HSV (Hue in [0, 360), Saturation in [0, 1], Value in [0, 1]).
pub fn linear_rgb_to_hsv(c: Color) -> Color {
    let r = c.c[0];
    let g = c.c[1];
    let b = c.c[2];
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;

    let s = if max < f32::EPSILON { 0.0 } else { delta / max };

    let h = if delta < f32::EPSILON {
        0.0
    } else if (max - r).abs() < f32::EPSILON {
        // max is r
        let h = 60.0 * ((g - b) / delta);
        if h < 0.0 {
            h + 360.0
        } else {
            h
        }
    } else if (max - g).abs() < f32::EPSILON {
        // max is g
        60.0 * ((b - r) / delta + 2.0)
    } else {
        // max is b
        60.0 * ((r - g) / delta + 4.0)
    };

    let h = h.rem_euclid(360.0);
    Color::new(h, s, v)
}

/// Convert HSV (Hue in [0, 360), Saturation in [0, 1], Value in [0, 1]) to linear RGB.
pub fn hsv_to_linear_rgb(c: Color) -> Color {
    let h = c.c[0];
    let s = c.c[1];
    let v = c.c[2];

    if s < f32::EPSILON {
        return Color::new(v, v, v);
    }

    let h = h.rem_euclid(360.0);
    let sector = (h / 60.0) as i32;
    let frac = h / 60.0 - sector as f32;

    let p = v * (1.0 - s);
    let q = v * (1.0 - s * frac);
    let t = v * (1.0 - s * (1.0 - frac));

    match sector % 6 {
        0 => Color::new(v, t, p),
        1 => Color::new(q, v, p),
        2 => Color::new(p, v, t),
        3 => Color::new(p, q, v),
        4 => Color::new(t, p, v),
        _ => Color::new(v, p, q),
    }
}

// ─────────────────────────────────────────────────────────────
// Linear RGB ↔ HSL
// ─────────────────────────────────────────────────────────────

/// Convert linear RGB to HSL (Hue in [0, 360), Saturation in [0, 1], Lightness in [0, 1]).
pub fn linear_rgb_to_hsl(c: Color) -> Color {
    let r = c.c[0];
    let g = c.c[1];
    let b = c.c[2];
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let l = (max + min) / 2.0;

    let s = if delta < f32::EPSILON {
        0.0
    } else {
        let denom = 1.0 - (2.0 * l - 1.0).abs();
        if denom < f32::EPSILON {
            0.0
        } else {
            delta / denom
        }
    };

    let h = if delta < f32::EPSILON {
        0.0
    } else if (max - r).abs() < f32::EPSILON {
        let h = 60.0 * ((g - b) / delta);
        if h < 0.0 {
            h + 360.0
        } else {
            h
        }
    } else if (max - g).abs() < f32::EPSILON {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };

    let h = h.rem_euclid(360.0);
    Color::new(h, s, l)
}

/// Convert HSL (Hue in [0, 360), Saturation in [0, 1], Lightness in [0, 1]) to linear RGB.
pub fn hsl_to_linear_rgb(c: Color) -> Color {
    let h = c.c[0];
    let s = c.c[1];
    let l = c.c[2];

    if s < f32::EPSILON {
        return Color::new(l, l, l);
    }

    let chroma = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h_prime = h.rem_euclid(360.0) / 60.0;
    let x = chroma * (1.0 - (h_prime % 2.0 - 1.0).abs());

    let (r1, g1, b1) = match h_prime as i32 {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };

    let m = l - chroma / 2.0;
    Color::new(r1 + m, g1 + m, b1 + m)
}

// ─────────────────────────────────────────────────────────────
// Universal converter
// ─────────────────────────────────────────────────────────────

/// Convert `from` color space to LinearRgb.
fn to_linear_rgb(c: Color, from: ColorSpace) -> Color {
    match from {
        ColorSpace::LinearRgb => c,
        ColorSpace::SRgb => color_srgb_to_linear(c),
        ColorSpace::Hsv => hsv_to_linear_rgb(c),
        ColorSpace::Hsl => hsl_to_linear_rgb(c),
        ColorSpace::XyzD65 => xyz_to_linear_rgb(c),
        ColorSpace::Lab => {
            let xyz = lab_to_xyz(c);
            xyz_to_linear_rgb(xyz)
        }
    }
}

/// Convert LinearRgb to the target color space.
fn from_linear_rgb(c: Color, to: ColorSpace) -> Color {
    match to {
        ColorSpace::LinearRgb => c,
        ColorSpace::SRgb => color_linear_to_srgb(c),
        ColorSpace::Hsv => linear_rgb_to_hsv(c),
        ColorSpace::Hsl => linear_rgb_to_hsl(c),
        ColorSpace::XyzD65 => linear_rgb_to_xyz(c),
        ColorSpace::Lab => {
            let xyz = linear_rgb_to_xyz(c);
            xyz_to_lab(xyz)
        }
    }
}

/// Convert a color from one color space to another, using LinearRgb as the pivot.
///
/// Identity is preserved when `from == to`.
pub fn convert_color(c: Color, from: ColorSpace, to: ColorSpace) -> Color {
    if from == to {
        return c;
    }
    let linear = to_linear_rgb(c, from);
    from_linear_rgb(linear, to)
}

// ─────────────────────────────────────────────────────────────
// Tone mapping
// ─────────────────────────────────────────────────────────────

/// Tone mapping operators for HDR → LDR conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToneMapping {
    /// Clamp values to [0, 1].
    Clamp,
    /// Reinhard operator: `x / (1 + x)`.
    Reinhard,
    /// Extended Reinhard: `x * (1 + x / white²) / (1 + x)`.
    ReinhardExtended {
        /// White point luminance.
        white: f32,
    },
    /// ACES filmic approximation (Narkowicz 2015).
    Aces,
    /// Uncharted 2 / Hable filmic tone mapping.
    Uncharted2,
}

/// Hable partial function used in Uncharted 2 tone mapping.
#[inline]
fn hable_partial(x: f32) -> f32 {
    let shoulder_strength = 0.15_f32;
    let linear_strength = 0.50_f32;
    let linear_angle = 0.10_f32;
    let toe_strength = 0.20_f32;
    let toe_numerator = 0.02_f32;
    let toe_denominator = 0.30_f32;

    let a = shoulder_strength;
    let bv = linear_strength;
    let cv = linear_angle;
    let dv = toe_strength;
    let ev = toe_numerator;
    let fv = toe_denominator;

    ((x * (a * x + cv * bv) + dv * ev) / (x * (a * x + bv) + dv * fv)) - ev / fv
}

impl ToneMapping {
    /// Apply tone mapping to a single linear-light channel value.
    ///
    /// Output is clamped to [0, 1].
    pub fn apply_channel(&self, x: f32) -> f32 {
        let result = match self {
            Self::Clamp => x.clamp(0.0, 1.0),
            Self::Reinhard => x / (1.0 + x),
            Self::ReinhardExtended { white } => {
                let w = *white;
                (x * (1.0 + x / (w * w))) / (1.0 + x)
            }
            Self::Aces => {
                // Narkowicz 2015 approximation
                let a = 2.51_f32;
                let b = 0.03_f32;
                let cv = 2.43_f32;
                let d = 0.59_f32;
                let e = 0.14_f32;
                (x * (a * x + b)) / (x * (cv * x + d) + e)
            }
            Self::Uncharted2 => {
                let white_scale = hable_partial(11.2);
                hable_partial(x * 2.0) / white_scale
            }
        };
        result.clamp(0.0, 1.0)
    }

    /// Apply tone mapping to all three components of a Color (per-channel).
    pub fn apply(&self, c: Color) -> Color {
        Color::new(
            self.apply_channel(c.c[0]),
            self.apply_channel(c.c[1]),
            self.apply_channel(c.c[2]),
        )
    }
}

// ─────────────────────────────────────────────────────────────
// White balance
// ─────────────────────────────────────────────────────────────

/// Compute the CIE 1931 chromaticity x for a given color temperature in Kelvin.
///
/// Valid range: 1667 K – 25000 K (Kang et al. 2002).
fn ccx_from_temp(t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    if t <= 4000.0 {
        -0.266_123_9e9 / t3 - 2.343_58e5 / t2 + 0.877_695_6e3 / t + 0.179_910
    } else {
        -3.025_846_9e9 / t3 + 2.107_038e6 / t2 + 0.222_634_7e3 / t + 0.240_390
    }
}

/// Compute the CIE 1931 chromaticity y for a given color temperature in Kelvin.
///
/// Uses Kang et al. 2002 polynomial segmented by x.
fn ccy_from_temp(t: f32, x: f32) -> f32 {
    let x2 = x * x;
    let x3 = x2 * x;
    if t <= 2222.0 {
        -1.106_381_4 * x3 - 1.348_110_2 * x2 + 2.185_558_3 * x - 0.202_196_83
    } else if t <= 4000.0 {
        -0.954_947_6 * x3 - 1.374_185_9 * x2 + 2.091_37 * x - 0.167_488_67
    } else {
        3.081_758 * x3 - 5.873_387 * x2 + 3.751_129_9 * x - 0.370_014_83
    }
}

/// Compute per-channel white balance correction factors for a given color temperature (Kelvin).
///
/// The returned `[f32; 3]` is a per-channel scale factor to apply to linear RGB pixels.
/// Values near `[1, 1, 1]` mean the input already matches D65 (≈ 6500 K).
///
/// Valid temperature range: approximately 1667 K – 25000 K.
pub fn white_balance_from_temperature(kelvin: f32) -> [f32; 3] {
    let t = kelvin.clamp(1667.0, 25000.0);
    let x = ccx_from_temp(t);
    let y = ccy_from_temp(t, x);

    // Avoid division by zero for degenerate y
    let y_safe = if y.abs() < f32::EPSILON {
        f32::EPSILON
    } else {
        y
    };

    // XYZ from chromaticity: Y=1, X=x/y, Z=(1-x-y)/y
    let xn = x / y_safe;
    let yn = 1.0_f32;
    let zn = (1.0 - x - y) / y_safe;

    // Convert computed white point to linear RGB
    let wp_xyz = Color::new(xn, yn, zn);
    let wp_rgb = xyz_to_linear_rgb(wp_xyz);

    // D65 white in linear RGB is [1, 1, 1] by definition of the sRGB matrix.
    // Correction factor = D65_white_rgb / computed_white_rgb = 1.0 / wp_rgb[i]
    let safe_r = if wp_rgb.c[0].abs() < f32::EPSILON {
        1.0
    } else {
        wp_rgb.c[0]
    };
    let safe_g = if wp_rgb.c[1].abs() < f32::EPSILON {
        1.0
    } else {
        wp_rgb.c[1]
    };
    let safe_b = if wp_rgb.c[2].abs() < f32::EPSILON {
        1.0
    } else {
        wp_rgb.c[2]
    };

    [1.0 / safe_r, 1.0 / safe_g, 1.0 / safe_b]
}

/// Apply white balance correction factors to a linear RGB Color.
///
/// Each channel is multiplied by the corresponding correction factor and clamped to [0, ∞).
pub fn apply_white_balance(c: Color, correction: &[f32; 3]) -> Color {
    Color::new(
        (c.c[0] * correction[0]).max(0.0),
        (c.c[1] * correction[1]).max(0.0),
        (c.c[2] * correction[2]).max(0.0),
    )
}

// ─────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────

/// Errors produced by batch image operations.
#[derive(Debug, thiserror::Error)]
pub enum ColorError {
    /// The pixel buffer has a size that is inconsistent with the declared dimensions.
    #[error("invalid pixel count: expected {expected}, got {got}")]
    InvalidPixelCount {
        /// Expected number of values (`height * width * 3`).
        expected: usize,
        /// Actual number of values in the buffer.
        got: usize,
    },
    /// The number of channels per pixel is not supported (only 3 is supported).
    #[error("unsupported channel count: {channels} (only 3 is supported)")]
    UnsupportedChannelCount {
        /// Number of channels found.
        channels: usize,
    },
    /// The image buffer is empty.
    #[error("empty image buffer")]
    EmptyImage,
}

// ─────────────────────────────────────────────────────────────
// Batch image operations
// ─────────────────────────────────────────────────────────────

/// Apply tone mapping to a flat f32 image in HWC format (channels = 3).
///
/// If `output_gamma` is `true`, the result is gamma-encoded to sRGB after tone mapping.
/// Returns a new `Vec<f32>` with the same dimensions.
pub fn apply_tone_mapping_image(
    pixels: &[f32],
    height: usize,
    width: usize,
    tone_mapping: ToneMapping,
    output_gamma: bool,
) -> Result<Vec<f32>, ColorError> {
    if pixels.is_empty() {
        return Err(ColorError::EmptyImage);
    }
    let expected = height * width * 3;
    if pixels.len() != expected {
        return Err(ColorError::InvalidPixelCount {
            expected,
            got: pixels.len(),
        });
    }

    let mut out = Vec::with_capacity(pixels.len());
    for chunk in pixels.chunks(3) {
        let c = Color::new(chunk[0], chunk[1], chunk[2]);
        let mapped = tone_mapping.apply(c);
        if output_gamma {
            let encoded = color_linear_to_srgb(mapped);
            out.extend_from_slice(&encoded.c);
        } else {
            out.extend_from_slice(&mapped.c);
        }
    }
    Ok(out)
}

/// Convert a flat f32 image (HWC, channels = 3) from one color space to another.
///
/// Returns a new `Vec<f32>` with the same dimensions.
pub fn convert_image_colorspace(
    pixels: &[f32],
    height: usize,
    width: usize,
    from: ColorSpace,
    to: ColorSpace,
) -> Result<Vec<f32>, ColorError> {
    if pixels.is_empty() {
        return Err(ColorError::EmptyImage);
    }
    let expected = height * width * 3;
    if pixels.len() != expected {
        return Err(ColorError::InvalidPixelCount {
            expected,
            got: pixels.len(),
        });
    }

    let mut out = Vec::with_capacity(pixels.len());
    for chunk in pixels.chunks(3) {
        let c = Color::new(chunk[0], chunk[1], chunk[2]);
        let converted = convert_color(c, from, to);
        out.extend_from_slice(&converted.c);
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-3;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn color_approx_eq(a: Color, b: Color, eps: f32) -> bool {
        approx_eq(a.c[0], b.c[0], eps)
            && approx_eq(a.c[1], b.c[1], eps)
            && approx_eq(a.c[2], b.c[2], eps)
    }

    // ── Gamma ──────────────────────────────────────────────────────

    #[test]
    fn test_linear_to_srgb_midpoint() {
        // 0.5 linear ≈ 0.7354 sRGB
        let got = linear_to_srgb(0.5);
        assert!(
            approx_eq(got, 0.7354, 1e-3),
            "linear_to_srgb(0.5) = {got}, expected ≈ 0.7354"
        );
    }

    #[test]
    fn test_srgb_to_linear_midpoint() {
        // 0.5 sRGB → linear ≈ 0.2140
        let got = srgb_to_linear(0.5);
        let expected = ((0.5 + 0.055) / 1.055_f32).powf(2.4);
        assert!(
            approx_eq(got, expected, 1e-5),
            "srgb_to_linear(0.5) = {got}, expected ≈ {expected}"
        );
    }

    #[test]
    fn test_gamma_roundtrip() {
        for val in [0.0, 0.01, 0.1, 0.5, 0.9, 1.0] {
            let roundtrip = linear_to_srgb(srgb_to_linear(val));
            assert!(
                approx_eq(roundtrip, val, EPS),
                "gamma roundtrip failed for {val}: got {roundtrip}"
            );
        }
    }

    // ── XYZ ────────────────────────────────────────────────────────

    #[test]
    fn test_linear_rgb_to_xyz_white() {
        // White [1,1,1] linear RGB → D65 white in XYZ ≈ [0.9505, 1.0, 1.0883]
        let white = Color::white();
        let xyz = linear_rgb_to_xyz(white);
        assert!(
            approx_eq(xyz.c[0], 0.9505, EPS),
            "X = {}, expected ≈ 0.9505",
            xyz.c[0]
        );
        assert!(
            approx_eq(xyz.c[1], 1.0, EPS),
            "Y = {}, expected ≈ 1.0",
            xyz.c[1]
        );
        assert!(
            approx_eq(xyz.c[2], 1.0883, EPS),
            "Z = {}, expected ≈ 1.0883",
            xyz.c[2]
        );
    }

    #[test]
    fn test_xyz_to_linear_rgb_roundtrip() {
        let original = Color::new(0.3, 0.5, 0.7);
        let xyz = linear_rgb_to_xyz(original);
        let recovered = xyz_to_linear_rgb(xyz);
        assert!(
            color_approx_eq(original, recovered, EPS),
            "XYZ roundtrip failed: original={original:?}, recovered={recovered:?}"
        );
    }

    // ── Lab ────────────────────────────────────────────────────────

    #[test]
    fn test_xyz_to_lab_white() {
        // D65 white in XYZ → L=100, a≈0, b≈0
        let white_xyz = Color::new(D65_XN, D65_YN, D65_ZN);
        let lab = xyz_to_lab(white_xyz);
        assert!(
            approx_eq(lab.c[0], 100.0, EPS),
            "L = {}, expected ≈ 100",
            lab.c[0]
        );
        assert!(
            approx_eq(lab.c[1], 0.0, EPS),
            "a = {}, expected ≈ 0",
            lab.c[1]
        );
        assert!(
            approx_eq(lab.c[2], 0.0, EPS),
            "b = {}, expected ≈ 0",
            lab.c[2]
        );
    }

    #[test]
    fn test_lab_to_xyz_roundtrip() {
        let original_xyz = Color::new(0.2, 0.4, 0.6);
        let lab = xyz_to_lab(original_xyz);
        let recovered_xyz = lab_to_xyz(lab);
        assert!(
            color_approx_eq(original_xyz, recovered_xyz, EPS),
            "Lab roundtrip failed: original={original_xyz:?}, recovered={recovered_xyz:?}"
        );
    }

    // ── HSV ────────────────────────────────────────────────────────

    #[test]
    fn test_linear_rgb_to_hsv_red() {
        // [1, 0, 0] → H=0°, S=1, V=1
        let red = Color::new(1.0, 0.0, 0.0);
        let hsv = linear_rgb_to_hsv(red);
        assert!(
            approx_eq(hsv.c[0], 0.0, EPS) || approx_eq(hsv.c[0], 360.0, EPS),
            "H = {}, expected ≈ 0",
            hsv.c[0]
        );
        assert!(
            approx_eq(hsv.c[1], 1.0, EPS),
            "S = {}, expected ≈ 1",
            hsv.c[1]
        );
        assert!(
            approx_eq(hsv.c[2], 1.0, EPS),
            "V = {}, expected ≈ 1",
            hsv.c[2]
        );
    }

    #[test]
    fn test_hsv_to_linear_rgb_roundtrip() {
        for (r, g, b) in [
            (0.8, 0.2, 0.5),
            (0.0, 1.0, 0.5),
            (0.3, 0.3, 0.3),
            (1.0, 0.0, 0.0),
        ] {
            let original = Color::new(r, g, b);
            let hsv = linear_rgb_to_hsv(original);
            let recovered = hsv_to_linear_rgb(hsv);
            assert!(
                color_approx_eq(original, recovered, EPS),
                "HSV roundtrip failed: original={original:?}, recovered={recovered:?}"
            );
        }
    }

    // ── HSL ────────────────────────────────────────────────────────

    #[test]
    fn test_linear_rgb_to_hsl_red() {
        // [1, 0, 0] → H=0°, S=1, L=0.5
        let red = Color::new(1.0, 0.0, 0.0);
        let hsl = linear_rgb_to_hsl(red);
        assert!(
            approx_eq(hsl.c[0], 0.0, EPS) || approx_eq(hsl.c[0], 360.0, EPS),
            "H = {}, expected ≈ 0",
            hsl.c[0]
        );
        assert!(
            approx_eq(hsl.c[1], 1.0, EPS),
            "S = {}, expected ≈ 1",
            hsl.c[1]
        );
        assert!(
            approx_eq(hsl.c[2], 0.5, EPS),
            "L = {}, expected ≈ 0.5",
            hsl.c[2]
        );
    }

    #[test]
    fn test_hsl_to_linear_rgb_roundtrip() {
        for (r, g, b) in [
            (0.8, 0.2, 0.5),
            (0.0, 1.0, 0.5),
            (0.3, 0.3, 0.3),
            (1.0, 0.0, 0.0),
        ] {
            let original = Color::new(r, g, b);
            let hsl = linear_rgb_to_hsl(original);
            let recovered = hsl_to_linear_rgb(hsl);
            assert!(
                color_approx_eq(original, recovered, EPS),
                "HSL roundtrip failed: original={original:?}, recovered={recovered:?}"
            );
        }
    }

    // ── convert_color ──────────────────────────────────────────────

    #[test]
    fn test_convert_color_identity() {
        let c = Color::new(0.4, 0.6, 0.8);
        for space in [
            ColorSpace::LinearRgb,
            ColorSpace::SRgb,
            ColorSpace::Hsv,
            ColorSpace::Hsl,
        ] {
            let result = convert_color(c, space, space);
            assert!(
                color_approx_eq(result, c, EPS),
                "identity failed for {space:?}: got {result:?}"
            );
        }
    }

    // ── Tone mapping ───────────────────────────────────────────────

    #[test]
    fn test_tone_mapping_reinhard_half() {
        // Reinhard(0.5) = 0.5 / (1 + 0.5) = 1/3
        let tm = ToneMapping::Reinhard;
        let got = tm.apply_channel(0.5);
        assert!(
            approx_eq(got, 1.0 / 3.0, EPS),
            "Reinhard(0.5) = {got}, expected ≈ 1/3"
        );
    }

    #[test]
    fn test_tone_mapping_clamp() {
        let tm = ToneMapping::Clamp;
        assert!(approx_eq(tm.apply_channel(2.0), 1.0, EPS));
        assert!(approx_eq(tm.apply_channel(-0.5), 0.0, EPS));
        assert!(approx_eq(tm.apply_channel(0.7), 0.7, EPS));
    }

    #[test]
    fn test_tone_mapping_aces() {
        let tm = ToneMapping::Aces;
        // ACES output is in [0, 1]
        for x in [0.0_f32, 0.5, 1.0, 2.0, 5.0] {
            let y = tm.apply_channel(x);
            assert!((0.0..=1.0).contains(&y), "ACES({x}) = {y} out of [0,1]");
        }
        // At x=0, output should be 0
        assert!(approx_eq(tm.apply_channel(0.0), 0.0, EPS));
        // Monotone: higher input → higher output (for positive inputs)
        assert!(tm.apply_channel(1.0) > tm.apply_channel(0.5));
    }

    #[test]
    fn test_tone_mapping_uncharted2() {
        let tm = ToneMapping::Uncharted2;
        // Output must be in [0, 1] for non-negative inputs
        for x in [0.0_f32, 0.5, 1.0, 2.0, 5.0, 11.2] {
            let y = tm.apply_channel(x);
            assert!(
                (0.0..=1.0).contains(&y),
                "Uncharted2({x}) = {y} out of [0,1]"
            );
        }
    }

    // ── White balance ──────────────────────────────────────────────

    #[test]
    fn test_white_balance_6500k_near_unity() {
        // 6500 K ≈ D65 → correction factors should be near [1, 1, 1]
        let correction = white_balance_from_temperature(6500.0);
        // Use loose tolerance; polynomial approximation won't be perfect at 6500 K
        let tol = 0.15;
        assert!(
            approx_eq(correction[0], 1.0, tol),
            "R correction = {}, expected ≈ 1.0",
            correction[0]
        );
        assert!(
            approx_eq(correction[1], 1.0, tol),
            "G correction = {}, expected ≈ 1.0",
            correction[1]
        );
        assert!(
            approx_eq(correction[2], 1.0, tol),
            "B correction = {}, expected ≈ 1.0",
            correction[2]
        );
    }

    // ── Batch operations ───────────────────────────────────────────

    #[test]
    fn test_apply_tone_mapping_image() {
        // 2×2 white image (all ones, HDR = 1.0 linear)
        let pixels = vec![1.0_f32; 4 * 3]; // 4 pixels × 3 channels
        let result = apply_tone_mapping_image(&pixels, 2, 2, ToneMapping::Reinhard, false)
            .expect("tone mapping failed");
        assert_eq!(result.len(), 4 * 3);
        // Reinhard(1.0) = 0.5
        for &v in &result {
            assert!(approx_eq(v, 0.5, EPS), "expected 0.5, got {v}");
        }
    }

    #[test]
    fn test_apply_tone_mapping_image_with_gamma() {
        // single pixel, value 0.5 linear
        let pixels = vec![0.5_f32, 0.5, 0.5];
        let result = apply_tone_mapping_image(&pixels, 1, 1, ToneMapping::Clamp, true)
            .expect("tone mapping failed");
        assert_eq!(result.len(), 3);
        // Clamp(0.5) = 0.5, then sRGB ≈ 0.7354
        for &v in &result {
            assert!(approx_eq(v, 0.7354, 1e-3), "expected ≈ 0.7354, got {v}");
        }
    }

    #[test]
    fn test_apply_tone_mapping_image_empty_error() {
        let result = apply_tone_mapping_image(&[], 0, 0, ToneMapping::Clamp, false);
        assert!(matches!(result, Err(ColorError::EmptyImage)));
    }

    #[test]
    fn test_apply_tone_mapping_image_size_mismatch() {
        let pixels = vec![0.5_f32; 5]; // wrong size for 2×1 = 6 values
        let result = apply_tone_mapping_image(&pixels, 2, 1, ToneMapping::Clamp, false);
        assert!(matches!(result, Err(ColorError::InvalidPixelCount { .. })));
    }

    #[test]
    fn test_convert_image_colorspace() {
        // Single white pixel in LinearRgb → sRGB should give [1, 1, 1]
        let pixels = vec![1.0_f32, 1.0, 1.0];
        let result =
            convert_image_colorspace(&pixels, 1, 1, ColorSpace::LinearRgb, ColorSpace::SRgb)
                .expect("conversion failed");
        assert_eq!(result.len(), 3);
        for &v in &result {
            assert!(approx_eq(v, 1.0, EPS), "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_convert_image_colorspace_roundtrip() {
        // A 1×2 image: one pixel red, one pixel green
        let pixels = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let srgb = convert_image_colorspace(&pixels, 1, 2, ColorSpace::LinearRgb, ColorSpace::SRgb)
            .expect("to srgb failed");
        let recovered =
            convert_image_colorspace(&srgb, 1, 2, ColorSpace::SRgb, ColorSpace::LinearRgb)
                .expect("back to linear failed");
        for (orig, rec) in pixels.iter().zip(recovered.iter()) {
            assert!(
                approx_eq(*orig, *rec, EPS),
                "roundtrip failed: original={orig}, recovered={rec}"
            );
        }
    }
}
