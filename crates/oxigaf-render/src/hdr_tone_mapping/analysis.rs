//! HDR image statistics, histograms, exposure estimation and white balance.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::curves::tone_luminance;
use super::error::ToneMappingError;

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
