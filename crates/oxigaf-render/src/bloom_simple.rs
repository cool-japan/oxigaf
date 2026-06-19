//! Simplified bloom/glow post-processing for 3DGS rendered images.
//!
//! This module provides a straightforward bloom pipeline operating on flat
//! `Vec<f32>` RGB images (H×W×3, row-major). It complements the HDR pyramid
//! bloom in `bloom.rs` with a more ergonomic API.
//!
//! # Pipeline (`apply_bloom`)
//!
//! 1. Extract bright pixels above a luminance threshold (soft quadratic knee).
//! 2. Build a Gaussian mip chain from the bright image.
//! 3. Blur each mip level.
//! 4. Accumulate (progressive upsample + blend).
//! 5. Composite: `result = original + strength * bloom * color_tint`.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the simplified bloom API.
#[derive(Debug, Error)]
pub enum BloomError {
    /// Image buffer length does not match width × height × channels.
    #[error("Image size {got} does not match {width}×{height}×{channels}")]
    SizeMismatch {
        got: usize,
        width: usize,
        height: usize,
        channels: usize,
    },
    /// Threshold must be non-negative.
    #[error("Invalid threshold {0}: must be in [0, ∞)")]
    InvalidThreshold(f32),
    /// Kernel size must be odd and >= 3.
    #[error("Invalid kernel size {0}: must be odd and >= 3")]
    InvalidKernelSize(usize),
    /// Generic invalid parameter.
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
    /// Mip level index out of range.
    #[error("Mip level {0} out of range")]
    InvalidMipLevel(usize),
}

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the simplified bloom post-processing pass.
#[derive(Debug, Clone)]
pub struct BloomConfig {
    /// Luminance threshold for bloom extraction (default 0.8).
    pub threshold: f32,
    /// Bloom intensity added to original (default 0.04).
    pub strength: f32,
    /// Bloom spread as fraction of image width (default 0.1).
    pub radius: f32,
    /// Number of downsample levels (default 5).
    pub n_mip_levels: usize,
    /// Soft-threshold knee width (default 0.1).
    pub knee: f32,
    /// Per-channel tint for bloom color (default white).
    pub color_tint: [f32; 3],
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            threshold: 0.8,
            strength: 0.04,
            radius: 0.1,
            n_mip_levels: 5,
            knee: 0.1,
            color_tint: [1.0, 1.0, 1.0],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that `img` has exactly `width * height * 3` elements.
#[inline]
fn check_rgb_size(img: &[f32], width: usize, height: usize) -> Result<(), BloomError> {
    let expected = width * height * 3;
    if img.len() != expected {
        return Err(BloomError::SizeMismatch {
            got: img.len(),
            width,
            height,
            channels: 3,
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Public functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute per-pixel BT.709 luminance map (single channel).
///
/// # Errors
/// [`BloomError::SizeMismatch`] if `img.len() != width * height * 3`.
pub fn bloom_luminance_map(
    img: &[f32],
    width: usize,
    height: usize,
) -> Result<Vec<f32>, BloomError> {
    check_rgb_size(img, width, height)?;
    let n = width * height;
    let mut lum = Vec::with_capacity(n);
    for i in 0..n {
        let b = i * 3;
        lum.push(0.2126 * img[b] + 0.7152 * img[b + 1] + 0.0722 * img[b + 2]);
    }
    Ok(lum)
}

/// Compute fraction of pixels whose BT.709 luminance exceeds `threshold`.
///
/// # Errors
/// [`BloomError::SizeMismatch`] on bad input size.
pub fn bloom_coverage(
    img: &[f32],
    width: usize,
    height: usize,
    threshold: f32,
) -> Result<f32, BloomError> {
    let lum = bloom_luminance_map(img, width, height)?;
    let n = lum.len();
    if n == 0 {
        return Ok(0.0);
    }
    let above = lum.iter().filter(|&&l| l > threshold).count();
    Ok(above as f32 / n as f32)
}

/// Extract bright regions using a soft quadratic knee.
///
/// For each pixel with BT.709 luminance `L`:
/// - `L < threshold − knee`: contribution = 0
/// - `threshold − knee <= L < threshold`: contribution = pixel × rq²
///   where `rq = (L − (threshold − knee)) / knee`
/// - `L >= threshold`: contribution = pixel (full)
///
/// # Errors
/// [`BloomError::SizeMismatch`] on bad input, [`BloomError::InvalidThreshold`] if
/// `threshold < 0`.
pub fn bloom_extract_bright(
    img: &[f32],
    width: usize,
    height: usize,
    threshold: f32,
    knee: f32,
) -> Result<Vec<f32>, BloomError> {
    if threshold < 0.0 {
        return Err(BloomError::InvalidThreshold(threshold));
    }
    check_rgb_size(img, width, height)?;
    let n = width * height;
    let knee_start = threshold - knee.max(0.0);
    let mut out = vec![0.0_f32; n * 3];
    for i in 0..n {
        let b = i * 3;
        let (r, g, bv) = (img[b], img[b + 1], img[b + 2]);
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * bv;
        if lum <= 0.0 {
            continue;
        }
        let scale = if knee <= 0.0 || knee_start >= threshold {
            if lum >= threshold {
                1.0
            } else {
                0.0
            }
        } else if lum < knee_start {
            0.0
        } else if lum < threshold {
            let rq = (lum - knee_start) / knee;
            rq * rq
        } else {
            1.0
        };
        if scale > 0.0 {
            out[b] = r * scale;
            out[b + 1] = g * scale;
            out[b + 2] = bv * scale;
        }
    }
    Ok(out)
}

/// Build a 1-D Gaussian kernel of length `kernel_size` (must be odd, >= 3).
///
/// # Errors
/// [`BloomError::InvalidKernelSize`] if `kernel_size < 3` or `kernel_size` is even.
pub fn bloom_make_kernel(kernel_size: usize, sigma: f32) -> Result<Vec<f32>, BloomError> {
    if kernel_size < 3 || kernel_size.is_multiple_of(2) {
        return Err(BloomError::InvalidKernelSize(kernel_size));
    }
    let center = (kernel_size / 2) as f32;
    let sigma_sq = if sigma <= f32::EPSILON {
        1.0
    } else {
        sigma * sigma
    };
    let mut k: Vec<f32> = (0..kernel_size)
        .map(|i| {
            let x = i as f32 - center;
            (-0.5 * x * x / sigma_sq).exp()
        })
        .collect();
    let sum: f32 = k.iter().sum();
    if sum > f32::EPSILON {
        k.iter_mut().for_each(|v| *v /= sum);
    } else {
        let mid = kernel_size / 2;
        k = vec![0.0; kernel_size];
        k[mid] = 1.0;
    }
    Ok(k)
}

/// Apply 1-D separable convolution horizontally (clamp-to-edge).
pub fn bloom_convolve_horizontal(
    img: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
) -> Vec<f32> {
    let half = (kernel.len() / 2) as isize;
    let mut out = vec![0.0_f32; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let mut acc = [0.0_f32; 3];
            for (ki, &kw) in kernel.iter().enumerate() {
                let sx = ((x as isize + ki as isize - half).clamp(0, width as isize - 1)) as usize;
                let b = (y * width + sx) * 3;
                acc[0] += img[b] * kw;
                acc[1] += img[b + 1] * kw;
                acc[2] += img[b + 2] * kw;
            }
            let ob = (y * width + x) * 3;
            out[ob] = acc[0];
            out[ob + 1] = acc[1];
            out[ob + 2] = acc[2];
        }
    }
    out
}

/// Apply 1-D separable convolution vertically (clamp-to-edge).
pub fn bloom_convolve_vertical(
    img: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
) -> Vec<f32> {
    let half = (kernel.len() / 2) as isize;
    let mut out = vec![0.0_f32; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let mut acc = [0.0_f32; 3];
            for (ki, &kw) in kernel.iter().enumerate() {
                let sy = ((y as isize + ki as isize - half).clamp(0, height as isize - 1)) as usize;
                let b = (sy * width + x) * 3;
                acc[0] += img[b] * kw;
                acc[1] += img[b + 1] * kw;
                acc[2] += img[b + 2] * kw;
            }
            let ob = (y * width + x) * 3;
            out[ob] = acc[0];
            out[ob + 1] = acc[1];
            out[ob + 2] = acc[2];
        }
    }
    out
}

/// Apply separable Gaussian blur with the given `kernel_size` and `sigma`.
///
/// # Errors
/// [`BloomError::InvalidKernelSize`] if `kernel_size` is invalid.
/// [`BloomError::SizeMismatch`] on bad image size.
pub fn bloom_gaussian_blur(
    img: &[f32],
    width: usize,
    height: usize,
    sigma: f32,
    kernel_size: usize,
) -> Result<Vec<f32>, BloomError> {
    check_rgb_size(img, width, height)?;
    let kernel = bloom_make_kernel(kernel_size, sigma)?;
    let h_pass = bloom_convolve_horizontal(img, width, height, &kernel);
    Ok(bloom_convolve_vertical(&h_pass, width, height, &kernel))
}

/// Downsample by 2× using box filter (average 2×2 blocks).
///
/// Returns `(downsampled_pixels, new_width, new_height)`.
/// New dimensions: `(width + 1) / 2`, `(height + 1) / 2` (ceiling division).
pub fn bloom_downsample(img: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let nw = width.div_ceil(2).max(1);
    let nh = height.div_ceil(2).max(1);
    let mut out = vec![0.0_f32; nw * nh * 3];
    for oy in 0..nh {
        for ox in 0..nw {
            let sx0 = (ox * 2).min(width.saturating_sub(1));
            let sx1 = (ox * 2 + 1).min(width.saturating_sub(1));
            let sy0 = (oy * 2).min(height.saturating_sub(1));
            let sy1 = (oy * 2 + 1).min(height.saturating_sub(1));
            let b00 = (sy0 * width + sx0) * 3;
            let b01 = (sy0 * width + sx1) * 3;
            let b10 = (sy1 * width + sx0) * 3;
            let b11 = (sy1 * width + sx1) * 3;
            let ob = (oy * nw + ox) * 3;
            for c in 0..3 {
                out[ob + c] = (img[b00 + c] + img[b01 + c] + img[b10 + c] + img[b11 + c]) * 0.25;
            }
        }
    }
    (out, nw, nh)
}

/// Upsample to `target_w × target_h` using bilinear interpolation.
pub fn bloom_upsample(
    img: &[f32],
    width: usize,
    height: usize,
    target_w: usize,
    target_h: usize,
) -> Vec<f32> {
    if width == 0 || height == 0 || target_w == 0 || target_h == 0 {
        return vec![0.0_f32; target_w * target_h * 3];
    }
    let mut out = vec![0.0_f32; target_w * target_h * 3];
    let sx_scale = width as f32 / target_w as f32;
    let sy_scale = height as f32 / target_h as f32;
    for ty in 0..target_h {
        for tx in 0..target_w {
            let sx = tx as f32 * sx_scale;
            let sy = ty as f32 * sy_scale;
            let x0 = (sx.floor() as usize).min(width.saturating_sub(1));
            let x1 = (x0 + 1).min(width.saturating_sub(1));
            let y0 = (sy.floor() as usize).min(height.saturating_sub(1));
            let y1 = (y0 + 1).min(height.saturating_sub(1));
            let tfx = sx - sx.floor();
            let tfy = sy - sy.floor();
            let ob = (ty * target_w + tx) * 3;
            for c in 0..3 {
                let v00 = img[(y0 * width + x0) * 3 + c];
                let v01 = img[(y0 * width + x1) * 3 + c];
                let v10 = img[(y1 * width + x0) * 3 + c];
                let v11 = img[(y1 * width + x1) * 3 + c];
                out[ob + c] = (v00 * (1.0 - tfx) + v01 * tfx) * (1.0 - tfy)
                    + (v10 * (1.0 - tfx) + v11 * tfx) * tfy;
            }
        }
    }
    out
}

/// Build a mip chain: each level is half the previous, starting from `img`.
pub fn bloom_build_mip_chain(
    img: &[f32],
    width: usize,
    height: usize,
    n_levels: usize,
) -> Vec<(Vec<f32>, usize, usize)> {
    let mut chain = Vec::with_capacity(n_levels.max(1));
    chain.push((img.to_vec(), width, height));
    for _ in 1..n_levels {
        let last = chain.last().map(|(d, w, h)| (d.clone(), *w, *h));
        let (prev, pw, ph) = match last {
            Some(v) => v,
            None => break,
        };
        if pw <= 1 && ph <= 1 {
            break;
        }
        let (next, nw, nh) = bloom_downsample(&prev, pw, ph);
        chain.push((next, nw, nh));
    }
    chain
}

/// Accumulate bloom from a mip chain (progressive upsample + blend).
///
/// Starts from the lowest-resolution mip, upsamples to the next level size and
/// adds, repeating until the chain is exhausted. Finally upsamples to
/// `target_w × target_h`.
///
/// # Errors
/// [`BloomError::InvalidMipLevel`] if `mip_chain` is empty.
pub fn bloom_accumulate_mip_chain(
    mip_chain: &[(Vec<f32>, usize, usize)],
    target_w: usize,
    target_h: usize,
) -> Result<Vec<f32>, BloomError> {
    if mip_chain.is_empty() {
        return Err(BloomError::InvalidMipLevel(0));
    }
    let n = mip_chain.len();
    let (last_data, last_w, last_h) = &mip_chain[n - 1];
    let mut acc = last_data.clone();
    let mut acc_w = *last_w;
    let mut acc_h = *last_h;
    for i in (0..n.saturating_sub(1)).rev() {
        let (level_data, lw, lh) = &mip_chain[i];
        acc = bloom_upsample(&acc, acc_w, acc_h, *lw, *lh);
        acc_w = *lw;
        acc_h = *lh;
        let len = (acc_w * acc_h * 3).min(level_data.len());
        for j in 0..len {
            acc[j] += level_data[j];
        }
    }
    if acc_w != target_w || acc_h != target_h {
        acc = bloom_upsample(&acc, acc_w, acc_h, target_w, target_h);
    }
    Ok(acc)
}

/// Composite: `result = original + strength * bloom_layer * color_tint`.
///
/// # Errors
/// [`BloomError::SizeMismatch`] if `original` and `bloom_layer` differ in length.
pub fn bloom_composite(
    original: &[f32],
    bloom_layer: &[f32],
    strength: f32,
    color_tint: &[f32; 3],
) -> Result<Vec<f32>, BloomError> {
    if original.len() != bloom_layer.len() {
        return Err(BloomError::SizeMismatch {
            got: bloom_layer.len(),
            width: original.len() / 3,
            height: 1,
            channels: 3,
        });
    }
    let n = original.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ch = i % 3;
        out.push(original[i] + strength * bloom_layer[i] * color_tint[ch]);
    }
    Ok(out)
}

/// Apply bloom post-processing to an RGB image (H×W×3, f32).
///
/// Pipeline:
/// 1. Extract bright regions with soft knee.
/// 2. Build mip chain.
/// 3. Blur each mip level with Gaussian.
/// 4. Accumulate mip chain.
/// 5. Composite: original + strength * bloom * tint.
///
/// # Errors
/// [`BloomError::SizeMismatch`] on bad input, [`BloomError::InvalidThreshold`] if
/// `config.threshold < 0`.
pub fn apply_bloom(
    img: &[f32],
    width: usize,
    height: usize,
    config: &BloomConfig,
) -> Result<Vec<f32>, BloomError> {
    if config.threshold < 0.0 {
        return Err(BloomError::InvalidThreshold(config.threshold));
    }
    check_rgb_size(img, width, height)?;
    let bright = bloom_extract_bright(img, width, height, config.threshold, config.knee)?;
    let n_levels = config.n_mip_levels.max(1);
    let mip_chain = bloom_build_mip_chain(&bright, width, height, n_levels);
    let sigma_base = (config.radius * width as f32).max(1.0);
    let kernel_size = {
        let ks = (sigma_base as usize * 2 + 1).max(3);
        if ks.is_multiple_of(2) {
            ks + 1
        } else {
            ks
        }
    };
    let blurred_chain: Vec<(Vec<f32>, usize, usize)> = mip_chain
        .iter()
        .enumerate()
        .map(|(idx, (data, w, h))| {
            let sigma = (sigma_base / (idx as f32 + 1.0)).max(0.5);
            let ks = (kernel_size / (idx + 1)).max(3);
            let ks = if ks % 2 == 0 { ks + 1 } else { ks };
            let blurred =
                bloom_gaussian_blur(data, *w, *h, sigma, ks).unwrap_or_else(|_| data.clone());
            (blurred, *w, *h)
        })
        .collect();
    let bloom_layer = bloom_accumulate_mip_chain(&blurred_chain, width, height)?;
    bloom_composite(img, &bloom_layer, config.strength, &config.color_tint)
}

/// Format a `BloomConfig` as a human-readable string.
pub fn format_bloom_config(config: &BloomConfig) -> String {
    format!(
        "BloomConfig {{ threshold: {:.3}, strength: {:.4}, radius: {:.3}, \
         n_mip_levels: {}, knee: {:.3}, color_tint: [{:.2}, {:.2}, {:.2}] }}",
        config.threshold,
        config.strength,
        config.radius,
        config.n_mip_levels,
        config.knee,
        config.color_tint[0],
        config.color_tint[1],
        config.color_tint[2],
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── bloom_make_kernel ────────────────────────────────────────────────────

    #[test]
    fn test_bloom_make_kernel_odd_size_ok() {
        let k = bloom_make_kernel(5, 1.0).unwrap();
        assert_eq!(k.len(), 5);
    }

    #[test]
    fn test_bloom_make_kernel_even_size_err() {
        assert!(bloom_make_kernel(4, 1.0).is_err());
    }

    #[test]
    fn test_bloom_make_kernel_too_small_err() {
        assert!(bloom_make_kernel(1, 1.0).is_err());
        assert!(bloom_make_kernel(0, 1.0).is_err());
        assert!(bloom_make_kernel(2, 1.0).is_err());
    }

    #[test]
    fn test_bloom_make_kernel_minimum_size_ok() {
        let k = bloom_make_kernel(3, 1.0).unwrap();
        assert_eq!(k.len(), 3);
    }

    #[test]
    fn test_bloom_make_kernel_normalized() {
        let k = bloom_make_kernel(7, 2.0).unwrap();
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "kernel sum = {sum}");
    }

    #[test]
    fn test_bloom_make_kernel_center_is_max() {
        let k = bloom_make_kernel(9, 2.0).unwrap();
        let center = k[4];
        for &v in &k {
            assert!(v <= center + 1e-6);
        }
    }

    #[test]
    fn test_bloom_make_kernel_symmetric() {
        let k = bloom_make_kernel(7, 1.5).unwrap();
        let n = k.len();
        for i in 0..n / 2 {
            assert!((k[i] - k[n - 1 - i]).abs() < 1e-5, "asymmetry at {i}");
        }
    }

    // ── bloom_convolve_horizontal ────────────────────────────────────────────

    #[test]
    fn test_bloom_convolve_horizontal_impulse_identity() {
        let img: Vec<f32> = (0..4 * 4 * 3).map(|i| i as f32 * 0.01).collect();
        let k = vec![1.0_f32];
        let out = bloom_convolve_horizontal(&img, 4, 4, &k);
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_bloom_convolve_horizontal_uniform_stays_uniform() {
        let v = 0.7_f32;
        let img = vec![v; 6 * 6 * 3];
        let k = bloom_make_kernel(5, 1.0).unwrap();
        let out = bloom_convolve_horizontal(&img, 6, 6, &k);
        for &x in &out {
            assert!((x - v).abs() < 1e-4, "got {x}");
        }
    }

    #[test]
    fn test_bloom_convolve_horizontal_output_size() {
        let img = vec![0.5_f32; 8 * 6 * 3];
        let k = bloom_make_kernel(3, 1.0).unwrap();
        let out = bloom_convolve_horizontal(&img, 8, 6, &k);
        assert_eq!(out.len(), 8 * 6 * 3);
    }

    // ── bloom_convolve_vertical ──────────────────────────────────────────────

    #[test]
    fn test_bloom_convolve_vertical_impulse_identity() {
        let img: Vec<f32> = (0..4 * 4 * 3).map(|i| i as f32 * 0.01).collect();
        let k = vec![1.0_f32];
        let out = bloom_convolve_vertical(&img, 4, 4, &k);
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_bloom_convolve_vertical_uniform_stays_uniform() {
        let v = 0.6_f32;
        let img = vec![v; 6 * 6 * 3];
        let k = bloom_make_kernel(5, 1.0).unwrap();
        let out = bloom_convolve_vertical(&img, 6, 6, &k);
        for &x in &out {
            assert!((x - v).abs() < 1e-4, "got {x}");
        }
    }

    #[test]
    fn test_bloom_convolve_vertical_output_size() {
        let img = vec![0.5_f32; 6 * 8 * 3];
        let k = bloom_make_kernel(3, 1.0).unwrap();
        let out = bloom_convolve_vertical(&img, 6, 8, &k);
        assert_eq!(out.len(), 6 * 8 * 3);
    }

    // ── bloom_gaussian_blur ──────────────────────────────────────────────────

    #[test]
    fn test_bloom_gaussian_blur_uniform_stays_uniform() {
        let v = 0.5_f32;
        let img = vec![v; 8 * 8 * 3];
        let out = bloom_gaussian_blur(&img, 8, 8, 1.5, 5).unwrap();
        for &x in &out {
            assert!((x - v).abs() < 1e-4, "got {x}");
        }
    }

    #[test]
    fn test_bloom_gaussian_blur_invalid_kernel_err() {
        let img = vec![0.5_f32; 4 * 4 * 3];
        assert!(bloom_gaussian_blur(&img, 4, 4, 1.0, 4).is_err());
        assert!(bloom_gaussian_blur(&img, 4, 4, 1.0, 2).is_err());
    }

    #[test]
    fn test_bloom_gaussian_blur_reduces_max_variation() {
        let w = 8usize;
        let h = 8usize;
        let mut img = vec![0.0_f32; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
                let b = (y * w + x) * 3;
                img[b] = v;
                img[b + 1] = v;
                img[b + 2] = v;
            }
        }
        let out = bloom_gaussian_blur(&img, w, h, 2.0, 7).unwrap();
        let max_in = img.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_in = img.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_out = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_out = out.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            max_out - min_out < max_in - min_in,
            "blur should reduce variation"
        );
    }

    #[test]
    fn test_bloom_gaussian_blur_size_mismatch_err() {
        let img = vec![0.5_f32; 10];
        assert!(bloom_gaussian_blur(&img, 4, 4, 1.0, 3).is_err());
    }

    // ── bloom_extract_bright ─────────────────────────────────────────────────

    #[test]
    fn test_bloom_extract_bright_below_threshold_zeros() {
        let img = vec![0.3_f32; 4 * 4 * 3];
        let out = bloom_extract_bright(&img, 4, 4, 0.8, 0.1).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_bloom_extract_bright_above_threshold_full() {
        let img = vec![1.0_f32; 2 * 2 * 3];
        let out = bloom_extract_bright(&img, 2, 2, 0.5, 0.0).unwrap();
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-6, "got {v}");
        }
    }

    #[test]
    fn test_bloom_extract_bright_knee_transition_smooth() {
        // lum = 0.2126*r + 0.7152*g + 0.0722*b = 1.0 * v when r=g=b=v
        // threshold=0.8, knee=0.4 → knee_start=0.4; v=0.6 is in [0.4, 0.8]
        let threshold = 0.8_f32;
        let knee = 0.4_f32;
        let v = 0.6_f32;
        let img = vec![v, v, v];
        let out = bloom_extract_bright(&img, 1, 1, threshold, knee).unwrap();
        // rq = (0.6 - 0.4) / 0.4 = 0.5; scale = 0.25 → out[0] = 0.15
        assert!(out[0] > 0.0, "out[0] = {}", out[0]);
        assert!(out[0] < v, "out[0] = {} should be < {}", out[0], v);
    }

    #[test]
    fn test_bloom_extract_bright_negative_threshold_err() {
        let img = vec![0.5_f32; 4 * 4 * 3];
        assert!(bloom_extract_bright(&img, 4, 4, -0.1, 0.0).is_err());
    }

    #[test]
    fn test_bloom_extract_bright_size_mismatch_err() {
        let img = vec![0.5_f32; 10];
        assert!(bloom_extract_bright(&img, 4, 4, 0.5, 0.0).is_err());
    }

    // ── bloom_downsample ─────────────────────────────────────────────────────

    #[test]
    fn test_bloom_downsample_dimensions_halved() {
        let img = vec![1.0_f32; 8 * 8 * 3];
        let (_, nw, nh) = bloom_downsample(&img, 8, 8);
        assert_eq!(nw, 4);
        assert_eq!(nh, 4);
    }

    #[test]
    fn test_bloom_downsample_odd_dimensions() {
        let img = vec![1.0_f32; 5 * 5 * 3];
        let (_, nw, nh) = bloom_downsample(&img, 5, 5);
        assert_eq!(nw, 3);
        assert_eq!(nh, 3);
    }

    #[test]
    fn test_bloom_downsample_uniform_stays_uniform() {
        let v = 0.7_f32;
        let img = vec![v; 6 * 6 * 3];
        let (out, _, _) = bloom_downsample(&img, 6, 6);
        for &x in &out {
            assert!((x - v).abs() < 1e-5, "got {x}");
        }
    }

    #[test]
    fn test_bloom_downsample_1x1_stays_1x1() {
        let img = vec![0.5_f32, 0.3_f32, 0.1_f32];
        let (out, nw, nh) = bloom_downsample(&img, 1, 1);
        assert_eq!(nw, 1);
        assert_eq!(nh, 1);
        assert_eq!(out.len(), 3);
    }

    // ── bloom_upsample ───────────────────────────────────────────────────────

    #[test]
    fn test_bloom_upsample_dimensions_doubled() {
        let img = vec![0.5_f32; 4 * 4 * 3];
        let out = bloom_upsample(&img, 4, 4, 8, 8);
        assert_eq!(out.len(), 8 * 8 * 3);
    }

    #[test]
    fn test_bloom_upsample_uniform_stays_uniform() {
        let v = 0.6_f32;
        let img = vec![v; 3 * 3 * 3];
        let out = bloom_upsample(&img, 3, 3, 6, 6);
        for &x in &out {
            assert!((x - v).abs() < 1e-4, "got {x}");
        }
    }

    #[test]
    fn test_bloom_upsample_1x1_expands_correctly() {
        let img = vec![1.0_f32, 0.5_f32, 0.25_f32];
        let out = bloom_upsample(&img, 1, 1, 3, 3);
        assert_eq!(out.len(), 3 * 3 * 3);
        for i in 0..9 {
            assert!((out[i * 3] - 1.0).abs() < 1e-6);
            assert!((out[i * 3 + 1] - 0.5).abs() < 1e-6);
            assert!((out[i * 3 + 2] - 0.25).abs() < 1e-6);
        }
    }

    // ── bloom_luminance_map ──────────────────────────────────────────────────

    #[test]
    fn test_bloom_luminance_map_pure_red() {
        let img = vec![1.0_f32, 0.0, 0.0];
        let lum = bloom_luminance_map(&img, 1, 1).unwrap();
        assert!((lum[0] - 0.2126).abs() < 1e-5, "got {}", lum[0]);
    }

    #[test]
    fn test_bloom_luminance_map_pure_white() {
        let img = vec![1.0_f32, 1.0, 1.0];
        let lum = bloom_luminance_map(&img, 1, 1).unwrap();
        assert!((lum[0] - 1.0).abs() < 1e-5, "got {}", lum[0]);
    }

    #[test]
    fn test_bloom_luminance_map_size_mismatch_err() {
        let img = vec![0.5_f32; 5];
        assert!(bloom_luminance_map(&img, 2, 2).is_err());
    }

    #[test]
    fn test_bloom_luminance_map_output_length() {
        let img = vec![0.5_f32; 4 * 3 * 3];
        let lum = bloom_luminance_map(&img, 4, 3).unwrap();
        assert_eq!(lum.len(), 4 * 3);
    }

    // ── bloom_coverage ───────────────────────────────────────────────────────

    #[test]
    fn test_bloom_coverage_all_dark() {
        let img = vec![0.1_f32; 4 * 4 * 3];
        let c = bloom_coverage(&img, 4, 4, 0.5).unwrap();
        assert_eq!(c, 0.0);
    }

    #[test]
    fn test_bloom_coverage_all_bright() {
        let img = vec![1.0_f32; 4 * 4 * 3];
        let c = bloom_coverage(&img, 4, 4, 0.5).unwrap();
        assert_eq!(c, 1.0);
    }

    #[test]
    fn test_bloom_coverage_mixed() {
        let mut img = vec![0.0_f32; 8 * 3];
        for i in 0..4 {
            img[i * 3] = 1.0;
            img[i * 3 + 1] = 1.0;
            img[i * 3 + 2] = 1.0;
        }
        let c = bloom_coverage(&img, 8, 1, 0.5).unwrap();
        assert!((c - 0.5).abs() < 1e-5, "got {c}");
    }

    // ── bloom_composite ──────────────────────────────────────────────────────

    #[test]
    fn test_bloom_composite_strength_zero_unchanged() {
        let orig = vec![0.5_f32; 4 * 4 * 3];
        let bloom_l = vec![1.0_f32; 4 * 4 * 3];
        let out = bloom_composite(&orig, &bloom_l, 0.0, &[1.0, 1.0, 1.0]).unwrap();
        for (a, b) in orig.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_bloom_composite_adds_bloom() {
        let orig = vec![0.5_f32; 3];
        let bloom_l = vec![0.2_f32; 3];
        let out = bloom_composite(&orig, &bloom_l, 1.0, &[1.0, 1.0, 1.0]).unwrap();
        for &v in &out {
            assert!((v - 0.7).abs() < 1e-5, "got {v}");
        }
    }

    #[test]
    fn test_bloom_composite_tint_applied() {
        let orig = vec![0.0_f32; 3];
        let bloom_l = vec![1.0_f32; 3];
        let tint = [2.0_f32, 0.5, 0.0];
        let out = bloom_composite(&orig, &bloom_l, 1.0, &tint).unwrap();
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!((out[1] - 0.5).abs() < 1e-5);
        assert!((out[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_bloom_composite_size_mismatch_err() {
        let orig = vec![0.5_f32; 12];
        let bloom_l = vec![0.1_f32; 9];
        assert!(bloom_composite(&orig, &bloom_l, 1.0, &[1.0, 1.0, 1.0]).is_err());
    }

    // ── bloom_build_mip_chain ────────────────────────────────────────────────

    #[test]
    fn test_bloom_build_mip_chain_level_count() {
        let img = vec![0.5_f32; 16 * 16 * 3];
        let chain = bloom_build_mip_chain(&img, 16, 16, 4);
        assert_eq!(chain.len(), 4);
    }

    #[test]
    fn test_bloom_build_mip_chain_dimensions_halving() {
        let img = vec![0.5_f32; 8 * 8 * 3];
        let chain = bloom_build_mip_chain(&img, 8, 8, 3);
        assert_eq!(chain[0].1, 8);
        assert_eq!(chain[0].2, 8);
        assert_eq!(chain[1].1, 4);
        assert_eq!(chain[1].2, 4);
    }

    #[test]
    fn test_bloom_build_mip_chain_one_level() {
        let img = vec![0.3_f32; 4 * 4 * 3];
        let chain = bloom_build_mip_chain(&img, 4, 4, 1);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].1, 4);
    }

    // ── bloom_accumulate_mip_chain ───────────────────────────────────────────

    #[test]
    fn test_bloom_accumulate_mip_chain_empty_err() {
        let chain: Vec<(Vec<f32>, usize, usize)> = vec![];
        assert!(bloom_accumulate_mip_chain(&chain, 4, 4).is_err());
    }

    #[test]
    fn test_bloom_accumulate_mip_chain_correct_output_size() {
        let img = vec![0.5_f32; 8 * 8 * 3];
        let chain = bloom_build_mip_chain(&img, 8, 8, 3);
        let out = bloom_accumulate_mip_chain(&chain, 8, 8).unwrap();
        assert_eq!(out.len(), 8 * 8 * 3);
    }

    #[test]
    fn test_bloom_accumulate_single_level_matches_target() {
        let img = vec![0.4_f32; 4 * 4 * 3];
        let chain = bloom_build_mip_chain(&img, 4, 4, 1);
        let out = bloom_accumulate_mip_chain(&chain, 8, 8).unwrap();
        assert_eq!(out.len(), 8 * 8 * 3);
    }

    // ── apply_bloom ──────────────────────────────────────────────────────────

    #[test]
    fn test_apply_bloom_dark_image_no_bloom() {
        let img = vec![0.1_f32; 8 * 8 * 3];
        let cfg = BloomConfig {
            threshold: 0.8,
            strength: 1.0,
            knee: 0.0,
            ..BloomConfig::default()
        };
        let out = apply_bloom(&img, 8, 8, &cfg).unwrap();
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-4, "dark: got {b}");
        }
    }

    #[test]
    fn test_apply_bloom_bright_image_adds_bloom() {
        let img = vec![1.5_f32; 8 * 8 * 3];
        let cfg = BloomConfig {
            threshold: 0.8,
            strength: 0.5,
            ..BloomConfig::default()
        };
        let out = apply_bloom(&img, 8, 8, &cfg).unwrap();
        let sum_in: f32 = img.iter().sum();
        let sum_out: f32 = out.iter().sum();
        assert!(sum_out >= sum_in, "bloom should add light");
    }

    #[test]
    fn test_apply_bloom_size_mismatch_err() {
        let img = vec![0.5_f32; 10];
        assert!(apply_bloom(&img, 4, 4, &BloomConfig::default()).is_err());
    }

    #[test]
    fn test_apply_bloom_invalid_threshold_err() {
        let img = vec![0.5_f32; 4 * 4 * 3];
        let cfg = BloomConfig {
            threshold: -1.0,
            ..BloomConfig::default()
        };
        assert!(apply_bloom(&img, 4, 4, &cfg).is_err());
    }

    #[test]
    fn test_apply_bloom_output_same_length() {
        let img = vec![0.5_f32; 6 * 6 * 3];
        let out = apply_bloom(&img, 6, 6, &BloomConfig::default()).unwrap();
        assert_eq!(out.len(), 6 * 6 * 3);
    }

    #[test]
    fn test_apply_bloom_all_black() {
        let img = vec![0.0_f32; 4 * 4 * 3];
        let out = apply_bloom(&img, 4, 4, &BloomConfig::default()).unwrap();
        for &v in &out {
            assert!(v.abs() < 1e-6, "black in → black out");
        }
    }

    // ── BloomConfig::default ─────────────────────────────────────────────────

    #[test]
    fn test_bloom_config_default_values() {
        let cfg = BloomConfig::default();
        assert!((cfg.threshold - 0.8).abs() < 1e-6);
        assert!((cfg.strength - 0.04).abs() < 1e-6);
        assert!((cfg.radius - 0.1).abs() < 1e-6);
        assert_eq!(cfg.n_mip_levels, 5);
        assert!((cfg.knee - 0.1).abs() < 1e-6);
        assert_eq!(cfg.color_tint, [1.0, 1.0, 1.0]);
    }

    // ── format_bloom_config ──────────────────────────────────────────────────

    #[test]
    fn test_format_bloom_config_non_empty() {
        let cfg = BloomConfig::default();
        let s = format_bloom_config(&cfg);
        assert!(!s.is_empty());
        assert!(s.contains("BloomConfig"));
    }

    // ── edge cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_apply_bloom_1x1_image() {
        let img = vec![1.0_f32, 1.0, 1.0];
        let out = apply_bloom(&img, 1, 1, &BloomConfig::default()).unwrap();
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_bloom_downsample_2x4_image() {
        let img = vec![0.5_f32; 2 * 4 * 3];
        let (out, nw, nh) = bloom_downsample(&img, 2, 4);
        assert_eq!(nw, 1);
        assert_eq!(nh, 2);
        assert_eq!(out.len(), 2 * 3);
    }

    #[test]
    fn test_bloom_upsample_asymmetric() {
        let img = vec![0.5_f32; 2 * 3 * 3];
        let out = bloom_upsample(&img, 2, 3, 4, 6);
        assert_eq!(out.len(), 4 * 6 * 3);
    }

    #[test]
    fn test_bloom_luminance_map_black_pixel() {
        let img = vec![0.0_f32; 3];
        let lum = bloom_luminance_map(&img, 1, 1).unwrap();
        assert_eq!(lum[0], 0.0);
    }

    #[test]
    fn test_bloom_composite_with_zero_tint() {
        let orig = vec![0.5_f32; 3];
        let bloom_l = vec![1.0_f32; 3];
        let out = bloom_composite(&orig, &bloom_l, 1.0, &[0.0, 0.0, 0.0]).unwrap();
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-6, "got {v}");
        }
    }
}
