//! Gamma and sRGB encode/decode helpers.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

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
