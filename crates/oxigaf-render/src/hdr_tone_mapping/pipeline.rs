//! Image-level tone mapping entry points.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::config::{ToneMapConfig, ToneMappingConfig};
use super::error::ToneMappingError;
use super::gamma::{apply_gamma, gamma_correct, srgb_gamma};
use super::operator::apply_operator;

/// Apply EV exposure to every channel in the image slice.
///
/// `output[i] = input[i] * 2^stops`
///
/// No clamping is applied; the returned values may exceed `[0, 1]`.
pub fn apply_exposure(image: &[f32], stops: f32) -> Vec<f32> {
    let scale = (2.0_f32).powf(stops);
    image.iter().map(|&v| v * scale).collect()
}

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

/// Validate the arguments shared by [`tone_map`] and [`tone_map_inplace`].
fn validate_tone_map_input(
    img_len: usize,
    width: usize,
    height: usize,
    config: &ToneMapConfig,
) -> Result<(), ToneMappingError> {
    let expected = width * height * 3;
    if img_len == 0 {
        return Err(ToneMappingError::EmptyImage);
    }
    if img_len != expected {
        return Err(ToneMappingError::SizeMismatch {
            got: img_len,
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
    Ok(())
}

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
    validate_tone_map_input(img.len(), width, height, config)?;
    let mut out = img.to_vec();
    tone_map_inplace(&mut out, width, height, config)?;
    Ok(out)
}

/// Apply tone mapping in-place to a linear HDR image.
///
/// Processes each pixel directly in `img` — unlike calling [`tone_map`] and
/// copying the result back, this allocates no intermediate output buffer.
///
/// # Errors
/// Same as [`tone_map`].
pub fn tone_map_inplace(
    img: &mut [f32],
    width: usize,
    height: usize,
    config: &ToneMapConfig,
) -> Result<(), ToneMappingError> {
    validate_tone_map_input(img.len(), width, height, config)?;

    for pixel in img.chunks_exact_mut(3) {
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
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
    }
    Ok(())
}
