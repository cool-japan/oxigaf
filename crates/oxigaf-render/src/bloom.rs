//! HDR Bloom post-processing for 3DGS rendered outputs.
//!
//! Bloom is a physical phenomenon where extremely bright light scatters and
//! creates a soft glow around intense light sources. This module implements
//! a GPU-inspired Gaussian pyramid bloom that:
//!
//! 1. **Extracts** bright pixels above a luminance threshold (hard or soft-knee).
//! 2. **Blurs** them with separable Gaussian convolution at multiple pyramid levels.
//! 3. **Composites** the accumulated bloom back onto the original HDR image.
//!
//! All operations work on flat `Vec<f32>` RGB (or RGBA) images in row-major
//! order, making them easy to chain with other CPU post-processing passes.
//!
//! # Example
//!
//! ```
//! use oxigaf_render::{HdrBloomConfig, apply_hdr_bloom};
//!
//! let width  = 4;
//! let height = 4;
//! // Solid mid-grey image (no bright pixels → bloom should add nothing).
//! let image: Vec<f32> = vec![0.5_f32; width * height * 3];
//! let config = HdrBloomConfig::default();
//! let result = apply_hdr_bloom(&image, width, height, &config).unwrap();
//! assert_eq!(result.len(), width * height * 3);
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by HDR bloom operations.
#[derive(Debug, Error)]
pub enum HdrBloomError {
    /// Bad configuration parameter.
    #[error("Invalid bloom configuration: {0}")]
    InvalidConfig(String),
    /// Image buffer length does not match the declared dimensions.
    #[error("Invalid image: {0}")]
    InvalidImage(String),
    /// Image has zero pixels.
    #[error("Empty image (zero pixels)")]
    EmptyImage,
}

// ─────────────────────────────────────────────────────────────────────────────
// Luminance helper (BT.709)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute BT.709 luminance from linear-light RGB.
#[inline]
fn luminance_bt709(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Validate that `image` has `width * height * 3` samples and neither dimension
/// is zero.
fn validate_rgb(image: &[f32], width: usize, height: usize) -> Result<(), HdrBloomError> {
    if width == 0 || height == 0 {
        return Err(HdrBloomError::EmptyImage);
    }
    let expected = width * height * 3;
    if image.len() != expected {
        return Err(HdrBloomError::InvalidImage(format!(
            "expected {} values for {}×{}×3, got {}",
            expected,
            width,
            height,
            image.len()
        )));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Threshold extraction
// ─────────────────────────────────────────────────────────────────────────────

/// Extract pixels above a luminance threshold (hard threshold).
///
/// Returns a same-size RGB image where pixels whose BT.709 luminance exceeds
/// `threshold` are copied as-is and all others are zeroed.
///
/// # Errors
///
/// - [`HdrBloomError::EmptyImage`] if `width` or `height` is zero.
/// - [`HdrBloomError::InvalidImage`] if `image.len() != width * height * 3`.
pub fn extract_bright(
    image: &[f32],
    width: usize,
    height: usize,
    threshold: f32,
) -> Result<Vec<f32>, HdrBloomError> {
    validate_rgb(image, width, height)?;
    let n_pixels = width * height;
    let mut out = vec![0.0_f32; n_pixels * 3];
    for i in 0..n_pixels {
        let base = i * 3;
        let (r, g, b) = (image[base], image[base + 1], image[base + 2]);
        let lum = luminance_bt709(r, g, b);
        if lum > threshold {
            out[base] = r;
            out[base + 1] = g;
            out[base + 2] = b;
        }
    }
    Ok(out)
}

/// Compute the soft-knee bloom weight for a given luminance value.
///
/// Provides a smooth transition from 0 to 1 around the threshold:
/// - Below `threshold − knee`: returns `0.0`.
/// - Above `threshold + knee`: returns `1.0`.
/// - Between those bounds: Hermite smoothstep `t² (3 − 2t)`.
///
/// When `knee == 0.0` the function degenerates to a hard step at `threshold`.
pub fn soft_knee_weight(luminance: f32, threshold: f32, knee: f32) -> f32 {
    if knee <= 0.0 {
        // Hard threshold fallback.
        return if luminance > threshold { 1.0 } else { 0.0 };
    }
    let lo = threshold - knee;
    let hi = threshold + knee;
    if luminance <= lo {
        return 0.0;
    }
    if luminance >= hi {
        return 1.0;
    }
    let t = (luminance - lo) / (2.0 * knee);
    // Hermite smoothstep: t² (3 − 2t)
    t * t * (3.0 - 2.0 * t)
}

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian kernel & separable blur
// ─────────────────────────────────────────────────────────────────────────────

/// Build a 1-D Gaussian kernel of half-width `radius`.
///
/// Returns a kernel of length `2 * radius + 1` normalised so its coefficients
/// sum to 1.  `sigma` controls the spread; a very small `sigma` approximates a
/// Dirac delta (identity blur).
///
/// # Panics
///
/// Does not panic; if `sigma` rounds to 0 only the centre tap is 1.0.
pub fn gaussian_kernel_1d(radius: usize, sigma: f32) -> Vec<f32> {
    let len = 2 * radius + 1;
    let center = radius as f32;
    let mut kernel: Vec<f32> = (0..len)
        .map(|i| {
            let x = i as f32 - center;
            if sigma <= f32::EPSILON {
                if i == radius {
                    1.0
                } else {
                    0.0
                }
            } else {
                (-(x * x) / (2.0 * sigma * sigma)).exp()
            }
        })
        .collect();
    let sum: f32 = kernel.iter().sum();
    if sum > f32::EPSILON {
        kernel.iter_mut().for_each(|v| *v /= sum);
    } else {
        // Fallback: unit impulse.
        let mid = radius;
        kernel = vec![0.0; len];
        kernel[mid] = 1.0;
    }
    kernel
}

/// Apply separable Gaussian blur to an RGB image (two-pass horizontal → vertical).
///
/// Uses clamp-to-edge boundary conditions so edge pixels are not darkened.
///
/// # Errors
///
/// - [`HdrBloomError::EmptyImage`] / [`HdrBloomError::InvalidImage`] on bad input.
pub fn gaussian_blur_rgb(
    image: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    sigma: f32,
) -> Result<Vec<f32>, HdrBloomError> {
    validate_rgb(image, width, height)?;
    let kernel = gaussian_kernel_1d(radius, sigma);

    // ── Horizontal pass ──────────────────────────────────────────────────────
    let mut h_pass = vec![0.0_f32; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let mut acc = [0.0_f32; 3];
            for (k, &w) in kernel.iter().enumerate() {
                let sx = (x as isize + k as isize - radius as isize).clamp(0, width as isize - 1)
                    as usize;
                let base = (y * width + sx) * 3;
                acc[0] += image[base] * w;
                acc[1] += image[base + 1] * w;
                acc[2] += image[base + 2] * w;
            }
            let out_base = (y * width + x) * 3;
            h_pass[out_base] = acc[0];
            h_pass[out_base + 1] = acc[1];
            h_pass[out_base + 2] = acc[2];
        }
    }

    // ── Vertical pass ────────────────────────────────────────────────────────
    let mut v_pass = vec![0.0_f32; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let mut acc = [0.0_f32; 3];
            for (k, &w) in kernel.iter().enumerate() {
                let sy = (y as isize + k as isize - radius as isize).clamp(0, height as isize - 1)
                    as usize;
                let base = (sy * width + x) * 3;
                acc[0] += h_pass[base] * w;
                acc[1] += h_pass[base + 1] * w;
                acc[2] += h_pass[base + 2] * w;
            }
            let out_base = (y * width + x) * 3;
            v_pass[out_base] = acc[0];
            v_pass[out_base + 1] = acc[1];
            v_pass[out_base + 2] = acc[2];
        }
    }

    Ok(v_pass)
}

// ─────────────────────────────────────────────────────────────────────────────
// Pyramid helpers: downsample / upsample
// ─────────────────────────────────────────────────────────────────────────────

/// Downsample an RGB image by 2× using a box (2×2 average) filter.
///
/// Returns `(pixels, out_width, out_height)`.  Minimum output size is 1×1.
/// For odd input dimensions the last row/column is handled by clamping.
/// Returns `(Vec::new(), 0, 0)` if `width == 0 || height == 0` rather than
/// indexing into the (necessarily empty) input.
pub fn downsample_2x(image: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    if width == 0 || height == 0 {
        return (Vec::new(), 0, 0);
    }
    let ow = (width / 2).max(1);
    let oh = (height / 2).max(1);
    let mut out = vec![0.0_f32; ow * oh * 3];
    for oy in 0..oh {
        for ox in 0..ow {
            // Source 2×2 block, clamped so odd dimensions don't overflow.
            let sx0 = (ox * 2).min(width.saturating_sub(1));
            let sx1 = (ox * 2 + 1).min(width.saturating_sub(1));
            let sy0 = (oy * 2).min(height.saturating_sub(1));
            let sy1 = (oy * 2 + 1).min(height.saturating_sub(1));

            let b00 = (sy0 * width + sx0) * 3;
            let b01 = (sy0 * width + sx1) * 3;
            let b10 = (sy1 * width + sx0) * 3;
            let b11 = (sy1 * width + sx1) * 3;

            let out_base = (oy * ow + ox) * 3;
            for c in 0..3 {
                out[out_base + c] =
                    (image[b00 + c] + image[b01 + c] + image[b10 + c] + image[b11 + c]) * 0.25;
            }
        }
    }
    (out, ow, oh)
}

/// Upsample an RGB image by 2× using bilinear interpolation.
///
/// Output size is `(width * 2) × (height * 2)`. Uses pixel-centre-aligned
/// sampling (`(ox + 0.5) * 0.5 - 0.5`, not `ox * 0.5`): mapping output pixel
/// *centres* to source pixel *centres* keeps a resample's centre of mass
/// fixed. Corner-aligned sampling instead shifts the whole image by a
/// quarter of a source pixel toward the origin, which compounds visibly
/// across a multi-level mip chain.
pub fn upsample_2x(image: &[f32], width: usize, height: usize) -> Vec<f32> {
    let ow = width * 2;
    let oh = height * 2;
    let mut out = vec![0.0_f32; ow * oh * 3];

    // Sample source at (sx, sy) with bilinear interpolation, clamped.
    let sample = |sx: f32, sy: f32, c: usize| -> f32 {
        let x0 = (sx.floor() as isize).clamp(0, width as isize - 1) as usize;
        let x1 = (sx.ceil() as isize).clamp(0, width as isize - 1) as usize;
        let y0 = (sy.floor() as isize).clamp(0, height as isize - 1) as usize;
        let y1 = (sy.ceil() as isize).clamp(0, height as isize - 1) as usize;
        let tx = sx - sx.floor();
        let ty = sy - sy.floor();
        let v00 = image[(y0 * width + x0) * 3 + c];
        let v01 = image[(y0 * width + x1) * 3 + c];
        let v10 = image[(y1 * width + x0) * 3 + c];
        let v11 = image[(y1 * width + x1) * 3 + c];
        (v00 * (1.0 - tx) + v01 * tx) * (1.0 - ty) + (v10 * (1.0 - tx) + v11 * tx) * ty
    };

    for oy in 0..oh {
        for ox in 0..ow {
            let sx = ((ox as f32 + 0.5) * 0.5 - 0.5).max(0.0);
            let sy = ((oy as f32 + 0.5) * 0.5 - 0.5).max(0.0);
            let out_base = (oy * ow + ox) * 3;
            for c in 0..3 {
                out[out_base + c] = sample(sx, sy, c);
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian pyramid
// ─────────────────────────────────────────────────────────────────────────────

/// Build a Gaussian mip-pyramid by alternating blur + 2× downsample.
///
/// Returns a `Vec` of `(pixels, level_width, level_height)` entries where
/// index 0 is the original size and index `k` is at half the resolution of
/// index `k−1`.  Stops early if any dimension would reach 1×1.
///
/// # Errors
///
/// - [`HdrBloomError::InvalidConfig`] if `num_levels == 0`.
/// - [`HdrBloomError::EmptyImage`] / [`HdrBloomError::InvalidImage`] on bad input.
pub fn build_mip_pyramid(
    image: &[f32],
    width: usize,
    height: usize,
    num_levels: usize,
) -> Result<Vec<(Vec<f32>, usize, usize)>, HdrBloomError> {
    if num_levels == 0 {
        return Err(HdrBloomError::InvalidConfig(
            "num_levels must be at least 1".to_string(),
        ));
    }
    validate_rgb(image, width, height)?;

    let mut levels: Vec<(Vec<f32>, usize, usize)> = Vec::with_capacity(num_levels);
    levels.push((image.to_vec(), width, height));

    for _ in 1..num_levels {
        // Borrow the previous level's pixels only for the blur call itself
        // (no clone of the, potentially large, pixel buffer just to satisfy
        // the borrow checker) and propagate a blur failure instead of
        // silently falling back to unblurred data — `gaussian_blur_rgb` can
        // only fail on a dimension mismatch, which would indicate a real
        // bug in how this level's dimensions were tracked and is worth
        // surfacing rather than hiding.
        let blurred_level = match levels.last().map(|l| (&l.0, l.1, l.2)) {
            Some((prev_pixels, prev_w, prev_h)) => {
                let blurred = gaussian_blur_rgb(prev_pixels, prev_w, prev_h, 1, 1.0)?;
                (blurred, prev_w, prev_h)
            }
            None => break,
        };
        let (blurred, prev_w, prev_h) = blurred_level;

        let (down_pixels, down_w, down_h) = downsample_2x(&blurred, prev_w, prev_h);
        levels.push((down_pixels, down_w, down_h));

        if down_w == 1 && down_h == 1 {
            break;
        }
    }

    Ok(levels)
}

// ─────────────────────────────────────────────────────────────────────────────
// BloomConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for HDR pyramid bloom.
#[derive(Debug, Clone)]
pub struct HdrBloomConfig {
    /// Luminance threshold for bright pixel extraction (BT.709).
    /// Values above 1.0 are typical for HDR content.
    pub threshold: f32,
    /// Soft knee half-width around threshold.  `0.0` = hard threshold.
    pub knee: f32,
    /// Number of Gaussian pyramid levels (more = wider, softer bloom).
    pub num_levels: usize,
    /// Gaussian blur radius per level (half-kernel size).
    pub blur_radius: usize,
    /// Gaussian blur sigma per level.
    pub blur_sigma: f32,
    /// Bloom intensity: scale applied to accumulated bloom before adding to original.
    pub intensity: f32,
    /// Per-level intensity weights.  Empty = uniform `1 / num_levels` each.
    pub level_weights: Vec<f32>,
}

impl Default for HdrBloomConfig {
    fn default() -> Self {
        Self {
            threshold: 1.0,
            knee: 0.1,
            num_levels: 5,
            blur_radius: 3,
            blur_sigma: 1.5,
            intensity: 0.5,
            level_weights: vec![],
        }
    }
}

impl HdrBloomConfig {
    /// Validate configuration, returning a descriptive error on bad values.
    ///
    /// # Errors
    ///
    /// [`HdrBloomError::InvalidConfig`] when any parameter is out of range.
    pub fn validate(&self) -> Result<(), HdrBloomError> {
        if self.threshold < 0.0 {
            return Err(HdrBloomError::InvalidConfig(
                "threshold must be >= 0".to_string(),
            ));
        }
        if self.knee < 0.0 {
            return Err(HdrBloomError::InvalidConfig(
                "knee must be >= 0".to_string(),
            ));
        }
        if self.num_levels == 0 {
            return Err(HdrBloomError::InvalidConfig(
                "num_levels must be >= 1".to_string(),
            ));
        }
        if self.blur_radius == 0 {
            return Err(HdrBloomError::InvalidConfig(
                "blur_radius must be >= 1".to_string(),
            ));
        }
        if self.blur_sigma <= 0.0 {
            return Err(HdrBloomError::InvalidConfig(
                "blur_sigma must be > 0".to_string(),
            ));
        }
        if self.intensity < 0.0 {
            return Err(HdrBloomError::InvalidConfig(
                "intensity must be >= 0".to_string(),
            ));
        }
        if !self.level_weights.is_empty() && self.level_weights.len() != self.num_levels {
            return Err(HdrBloomError::InvalidConfig(format!(
                "level_weights length {} must equal num_levels {}",
                self.level_weights.len(),
                self.num_levels
            )));
        }
        Ok(())
    }

    /// Return the effective (normalised) weight for pyramid level `i`.
    ///
    /// - If `level_weights` is empty: uniform `1.0 / num_levels`.
    /// - Otherwise: `level_weights[i] / sum(level_weights)`.
    pub fn level_weight(&self, i: usize) -> f32 {
        if self.level_weights.is_empty() {
            return 1.0 / self.num_levels as f32;
        }
        let sum: f32 = self.level_weights.iter().sum();
        if sum <= f32::EPSILON {
            return 1.0 / self.num_levels as f32;
        }
        let w = self.level_weights.get(i).copied().unwrap_or(0.0);
        w / sum
    }

    /// Wide bloom preset: many levels, large radius, high intensity.
    pub fn wide() -> Self {
        Self {
            num_levels: 7,
            blur_radius: 5,
            blur_sigma: 2.5,
            intensity: 0.7,
            ..Self::default()
        }
    }

    /// Tight bloom preset: few levels, small radius, low intensity.
    pub fn tight() -> Self {
        Self {
            num_levels: 3,
            blur_radius: 2,
            blur_sigma: 1.0,
            intensity: 0.3,
            ..Self::default()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core bloom
// ─────────────────────────────────────────────────────────────────────────────

/// Apply HDR pyramid bloom to an RGB image.
///
/// The image may contain values above 1.0 (HDR content).  The output also
/// may exceed 1.0 — tone-map afterwards if you need LDR output.
///
/// # Algorithm
///
/// 1. Extract bright pixels via soft-knee or hard threshold.
/// 2. Build a Gaussian mip-pyramid from the bright image.
/// 3. Blur each pyramid level.
/// 4. Accumulate: upsample each level to full resolution and weight-sum them.
/// 5. Scale by `config.intensity`.
/// 6. Add to original: `output[i] = image[i] + bloom[i]`.
///
/// # Errors
///
/// - [`HdrBloomError::InvalidConfig`] if `config.validate()` fails.
/// - [`HdrBloomError::EmptyImage`] / [`HdrBloomError::InvalidImage`] on bad input.
pub fn apply_hdr_bloom(
    image: &[f32],
    width: usize,
    height: usize,
    config: &HdrBloomConfig,
) -> Result<Vec<f32>, HdrBloomError> {
    config.validate()?;
    validate_rgb(image, width, height)?;

    let n = width * height;

    // 1. Extract bright pixels (with optional soft knee).
    let mut bright = vec![0.0_f32; n * 3];
    for i in 0..n {
        let base = i * 3;
        let (r, g, b) = (image[base], image[base + 1], image[base + 2]);
        let lum = luminance_bt709(r, g, b);
        let weight = soft_knee_weight(lum, config.threshold, config.knee);
        if weight > 0.0 {
            bright[base] = r * weight;
            bright[base + 1] = g * weight;
            bright[base + 2] = b * weight;
        }
    }

    // 2. Build mip-pyramid from bright image.
    let pyramid = build_mip_pyramid(&bright, width, height, config.num_levels)?;

    // 3. Blur each level and accumulate with weight, upsampled to original size.
    let mut bloom_acc = vec![0.0_f32; n * 3];

    for (level_idx, (level_pixels, level_w, level_h)) in pyramid.iter().enumerate() {
        let level_weight = config.level_weight(level_idx);
        if level_weight <= 0.0 {
            continue;
        }

        // Blur this pyramid level.
        let blurred = gaussian_blur_rgb(
            level_pixels,
            *level_w,
            *level_h,
            config.blur_radius,
            config.blur_sigma,
        )?;

        // Upsample directly to the original size with a single size-targeted
        // bilinear resample (rather than a chain of `upsample_2x` doublings:
        // for a non-power-of-two `width`/`height`, `uw * 2` can overshoot
        // `width`, and there is no dimension at which the chain's real
        // buffer size (always exactly double the previous step) matches the
        // clamped `width × height` target — a discrepancy previous code
        // detected but never actually corrected before compositing the
        // wrongly-shaped buffer, producing a diagonally row-skewed bloom
        // layer). `bloom_simple::bloom_upsample` resamples straight from
        // `(level_w, level_h)` to `(width, height)` and always returns
        // exactly `width * height * 3` samples.
        let up = crate::bloom_simple::bloom_upsample(&blurred, *level_w, *level_h, width, height);
        for i in 0..n * 3 {
            bloom_acc[i] += up[i] * level_weight;
        }
    }

    // 4. Scale by intensity and add to original.
    let mut out = vec![0.0_f32; n * 3];
    for i in 0..n * 3 {
        out[i] = image[i] + bloom_acc[i] * config.intensity;
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// ACES + sRGB tone mapping
// ─────────────────────────────────────────────────────────────────────────────

/// ACES filmic approximation (Stephen Hill's curve).
///
/// Maps an HDR linear value to [0, 1]: `(x*(2.51x+0.03)) / (x*(2.43x+0.59)+0.14)`.
#[inline]
fn aces_filmic(x: f32) -> f32 {
    let x = x.max(0.0);
    ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
}

/// sRGB gamma encode (simple power-law `x^(1/2.2)`).
#[inline]
fn srgb_gamma(x: f32) -> f32 {
    x.max(0.0).powf(1.0 / 2.2)
}

/// Apply bloom, then ACES tone mapping, then sRGB gamma in a single pass.
///
/// Returns an LDR image with values in `[0, 1]`.
///
/// # Errors
///
/// Same as [`apply_hdr_bloom`].
pub fn apply_hdr_bloom_and_tonemap(
    image: &[f32],
    width: usize,
    height: usize,
    config: &HdrBloomConfig,
) -> Result<Vec<f32>, HdrBloomError> {
    let bloomed = apply_hdr_bloom(image, width, height, config)?;
    let out: Vec<f32> = bloomed
        .iter()
        .map(|&v| srgb_gamma(aces_filmic(v)))
        .collect();
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// RGBA support
// ─────────────────────────────────────────────────────────────────────────────

/// Apply HDR bloom to an RGBA image, passing alpha through unchanged.
///
/// `image` must be in RGBA row-major order (`len = width * height * 4`).
///
/// # Errors
///
/// - [`HdrBloomError::EmptyImage`] if `width` or `height` is zero.
/// - [`HdrBloomError::InvalidImage`] if `image.len() != width * height * 4`.
/// - [`HdrBloomError::InvalidConfig`] if `config.validate()` fails.
pub fn apply_hdr_bloom_rgba(
    image: &[f32],
    width: usize,
    height: usize,
    config: &HdrBloomConfig,
) -> Result<Vec<f32>, HdrBloomError> {
    if width == 0 || height == 0 {
        return Err(HdrBloomError::EmptyImage);
    }
    let n_pixels = width * height;
    let expected = n_pixels * 4;
    if image.len() != expected {
        return Err(HdrBloomError::InvalidImage(format!(
            "expected {} values for {}×{}×4, got {}",
            expected,
            width,
            height,
            image.len()
        )));
    }
    config.validate()?;

    // De-interleave: RGB + alpha.
    let mut rgb = vec![0.0_f32; n_pixels * 3];
    let mut alpha = vec![0.0_f32; n_pixels];
    for (i, alpha_val) in alpha.iter_mut().enumerate().take(n_pixels) {
        let src = i * 4;
        let dst = i * 3;
        rgb[dst] = image[src];
        rgb[dst + 1] = image[src + 1];
        rgb[dst + 2] = image[src + 2];
        *alpha_val = image[src + 3];
    }

    // Apply bloom to RGB channels.
    let bloomed_rgb = apply_hdr_bloom(&rgb, width, height, config)?;

    // Re-interleave with original alpha.
    let mut out = vec![0.0_f32; n_pixels * 4];
    for (i, &alpha_val) in alpha.iter().enumerate().take(n_pixels) {
        let src = i * 3;
        let dst = i * 4;
        out[dst] = bloomed_rgb[src];
        out[dst + 1] = bloomed_rgb[src + 1];
        out[dst + 2] = bloomed_rgb[src + 2];
        out[dst + 3] = alpha_val;
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// BloomStats
// ─────────────────────────────────────────────────────────────────────────────

/// Diagnostic statistics for a bloom pass.
#[derive(Debug, Clone)]
pub struct HdrBloomStats {
    /// Fraction of pixels whose luminance exceeded the threshold.
    pub bright_fraction: f32,
    /// Mean BT.709 luminance added by the bloom pass.
    pub mean_bloom_luminance: f32,
    /// Maximum BT.709 luminance added by the bloom pass.
    pub max_bloom_luminance: f32,
    /// Ratio of mean bloom luminance to mean original luminance.
    pub bloom_to_original_ratio: f32,
}

/// Compute [`HdrBloomStats`] by comparing an original and a bloomed image.
///
/// # Errors
///
/// - [`HdrBloomError::EmptyImage`] / [`HdrBloomError::InvalidImage`] if either
///   buffer does not match `width * height * 3`.
pub fn compute_hdr_bloom_stats(
    original: &[f32],
    bloomed: &[f32],
    width: usize,
    height: usize,
    threshold: f32,
) -> Result<HdrBloomStats, HdrBloomError> {
    validate_rgb(original, width, height)?;
    if bloomed.len() != original.len() {
        return Err(HdrBloomError::InvalidImage(format!(
            "bloomed image length {} does not match original {}",
            bloomed.len(),
            original.len()
        )));
    }

    let n_pixels = width * height;
    let mut bright_count = 0usize;
    let mut sum_orig_lum = 0.0_f32;
    let mut sum_bloom_lum = 0.0_f32;
    let mut max_bloom_lum = 0.0_f32;

    for i in 0..n_pixels {
        let base = i * 3;
        let (or_, og, ob) = (original[base], original[base + 1], original[base + 2]);
        let (br, bg, bb) = (bloomed[base], bloomed[base + 1], bloomed[base + 2]);

        let orig_lum = luminance_bt709(or_, og, ob);
        let bloom_lum = luminance_bt709(br - or_, bg - og, bb - ob);
        let bloom_lum = bloom_lum.max(0.0);

        if orig_lum > threshold {
            bright_count += 1;
        }
        sum_orig_lum += orig_lum;
        sum_bloom_lum += bloom_lum;
        if bloom_lum > max_bloom_lum {
            max_bloom_lum = bloom_lum;
        }
    }

    let n = n_pixels as f32;
    let mean_orig_lum = sum_orig_lum / n;
    let mean_bloom_lum = sum_bloom_lum / n;

    Ok(HdrBloomStats {
        bright_fraction: bright_count as f32 / n,
        mean_bloom_luminance: mean_bloom_lum,
        max_bloom_luminance: max_bloom_lum,
        bloom_to_original_ratio: mean_bloom_lum / (mean_orig_lum + 1e-8),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Simplified bloom API (delegated to bloom_simple module)
// ─────────────────────────────────────────────────────────────────────────────

pub use crate::bloom_simple::{
    apply_bloom, bloom_accumulate_mip_chain, bloom_build_mip_chain, bloom_composite,
    bloom_convolve_horizontal, bloom_convolve_vertical, bloom_coverage, bloom_downsample,
    bloom_extract_bright, bloom_gaussian_blur, bloom_luminance_map, bloom_make_kernel,
    bloom_upsample, format_bloom_config, BloomConfig, BloomError,
};

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── gaussian_kernel_1d ───────────────────────────────────────────────────

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        let k = gaussian_kernel_1d(5, 2.0);
        let sum: f32 = k.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "kernel sum should be 1.0, got {sum}"
        );
    }

    #[test]
    fn test_gaussian_kernel_radius_zero() {
        let k = gaussian_kernel_1d(0, 1.0);
        assert_eq!(k.len(), 1);
        assert!((k[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_kernel_length() {
        let radius = 3;
        let k = gaussian_kernel_1d(radius, 1.5);
        assert_eq!(k.len(), 2 * radius + 1);
    }

    #[test]
    fn test_gaussian_kernel_symmetric() {
        let k = gaussian_kernel_1d(4, 2.0);
        let n = k.len();
        for i in 0..n / 2 {
            assert!(
                (k[i] - k[n - 1 - i]).abs() < 1e-6,
                "kernel not symmetric at {i}"
            );
        }
    }

    // ── soft_knee_weight ─────────────────────────────────────────────────────

    #[test]
    fn test_soft_knee_below_threshold_returns_zero() {
        let w = soft_knee_weight(0.5, 1.0, 0.2);
        assert_eq!(w, 0.0);
    }

    #[test]
    fn test_soft_knee_above_threshold_returns_one() {
        let w = soft_knee_weight(2.0, 1.0, 0.2);
        assert_eq!(w, 1.0);
    }

    #[test]
    fn test_soft_knee_at_midpoint_approx_half() {
        // Midpoint of the knee band: luminance = threshold (= lo + knee).
        let w = soft_knee_weight(1.0, 1.0, 0.2);
        // t = (1.0 - 0.8) / 0.4 = 0.5  →  smoothstep(0.5) = 0.5
        assert!((w - 0.5).abs() < 1e-5, "expected ~0.5 at midpoint, got {w}");
    }

    #[test]
    fn test_soft_knee_hard_threshold_fallback() {
        assert_eq!(soft_knee_weight(0.9, 1.0, 0.0), 0.0);
        assert_eq!(soft_knee_weight(1.1, 1.0, 0.0), 1.0);
    }

    // ── extract_bright ───────────────────────────────────────────────────────

    #[test]
    fn test_extract_bright_all_dark() {
        // All pixels at luminance 0.3 < threshold 1.0.
        let img = vec![0.3_f32; 4 * 4 * 3];
        let out = extract_bright(&img, 4, 4, 1.0).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_extract_bright_all_bright() {
        // All pixels well above threshold.
        let img: Vec<f32> = (0..2 * 2 * 3).map(|_| 2.0).collect();
        let out = extract_bright(&img, 2, 2, 1.0).unwrap();
        assert_eq!(out, img);
    }

    #[test]
    fn test_extract_bright_mixed() {
        // 2 pixels: [dark, bright]
        let mut img = vec![0.0_f32; 2 * 3];
        // Pixel 0: luminance ≈ 0.1 (dark)
        img[0] = 0.5;
        img[1] = 0.0;
        img[2] = 0.0;
        // Pixel 1: luminance ≈ 1.4 (bright, R channel dominant)
        img[3] = 6.0;
        img[4] = 0.5;
        img[5] = 0.0;
        let out = extract_bright(&img, 2, 1, 1.0).unwrap();
        // First pixel should be zeroed.
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 0.0);
        // Second pixel preserved.
        assert_eq!(out[3], img[3]);
        assert_eq!(out[4], img[4]);
        assert_eq!(out[5], img[5]);
    }

    #[test]
    fn test_extract_bright_error_on_mismatch() {
        let img = vec![0.5_f32; 10]; // wrong length for 2×2×3 = 12
        assert!(extract_bright(&img, 2, 2, 1.0).is_err());
    }

    // ── gaussian_blur_rgb ────────────────────────────────────────────────────

    #[test]
    fn test_gaussian_blur_uniform_stays_uniform() {
        let v = 0.5_f32;
        let img = vec![v; 8 * 8 * 3];
        let out = gaussian_blur_rgb(&img, 8, 8, 2, 1.0).unwrap();
        for &x in &out {
            assert!(
                (x - v).abs() < 1e-4,
                "uniform image should stay uniform after blur, got {x}"
            );
        }
    }

    #[test]
    fn test_gaussian_blur_identity_with_tiny_sigma() {
        // radius=0, sigma=tiny → kernel is [1.0] → output == input
        let img: Vec<f32> = (0..3 * 3 * 3).map(|i| (i as f32) * 0.1).collect();
        let out = gaussian_blur_rgb(&img, 3, 3, 0, 0.00001).unwrap();
        for (a, b) in img.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "identity blur should leave image unchanged"
            );
        }
    }

    // ── downsample_2x ────────────────────────────────────────────────────────

    #[test]
    fn test_downsample_2x_4x4_to_2x2() {
        let img = vec![1.0_f32; 4 * 4 * 3];
        let (out, ow, oh) = downsample_2x(&img, 4, 4);
        assert_eq!(ow, 2);
        assert_eq!(oh, 2);
        assert_eq!(out.len(), 2 * 2 * 3);
    }

    #[test]
    fn test_downsample_2x_uniform_stays_uniform() {
        let v = 0.7_f32;
        let img = vec![v; 6 * 6 * 3];
        let (out, _, _) = downsample_2x(&img, 6, 6);
        for &x in &out {
            assert!(
                (x - v).abs() < 1e-5,
                "uniform downsample should remain uniform"
            );
        }
    }

    #[test]
    fn test_downsample_2x_zero_width_no_panic() {
        // Regression: `(ox * 2).min(width - 1)` used to underflow `width - 1`
        // as a `usize` subtraction when `width == 0`, which either panics in
        // debug builds or wraps to `usize::MAX` and then panics on the
        // resulting out-of-bounds slice index in release builds.
        let (out, ow, oh) = downsample_2x(&[], 0, 4);
        assert_eq!((out.len(), ow, oh), (0, 0, 0));
    }

    #[test]
    fn test_downsample_2x_zero_height_no_panic() {
        let (out, ow, oh) = downsample_2x(&[], 4, 0);
        assert_eq!((out.len(), ow, oh), (0, 0, 0));
    }

    #[test]
    fn test_downsample_2x_zero_width_and_height_no_panic() {
        let (out, ow, oh) = downsample_2x(&[], 0, 0);
        assert_eq!((out.len(), ow, oh), (0, 0, 0));
    }

    // ── upsample_2x ──────────────────────────────────────────────────────────

    #[test]
    fn test_upsample_2x_2x2_to_4x4() {
        let img = vec![0.5_f32; 2 * 2 * 3];
        let out = upsample_2x(&img, 2, 2);
        assert_eq!(out.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_upsample_2x_single_pixel_expands() {
        // 1×1 → 2×2, all the same value.
        let img = vec![0.8_f32, 0.4_f32, 0.2_f32];
        let out = upsample_2x(&img, 1, 1);
        assert_eq!(out.len(), 2 * 2 * 3);
        for i in 0..4 {
            let b = i * 3;
            assert!((out[b] - 0.8).abs() < 1e-6);
            assert!((out[b + 1] - 0.4).abs() < 1e-6);
            assert!((out[b + 2] - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn test_upsample_2x_pixel_center_aligned() {
        // 4x1 horizontal ramp (R channel = column index; G, B = 0). With
        // the pre-fix corner-aligned sampling (`sx = ox * 0.5`) the whole
        // image is shifted by a quarter source pixel toward the origin;
        // with pixel-centre alignment the ramp's centre of mass is
        // preserved and these exact values result (hand-derived from the
        // bilinear formula).
        let width = 4;
        let height = 1;
        let mut img = vec![0.0_f32; width * height * 3];
        for x in 0..width {
            img[x * 3] = x as f32;
        }
        let out = upsample_2x(&img, width, height);
        assert_eq!(out.len(), 8 * 2 * 3);

        let expected_r = [0.0_f32, 0.25, 0.75, 1.25];
        let corner_aligned_r = [0.0_f32, 0.5, 1.0, 1.5];
        for (ox, &exp) in expected_r.iter().enumerate() {
            let r = out[ox * 3];
            assert!(
                (r - exp).abs() < 1e-4,
                "pixel-centre-aligned upsample at ox={ox}: expected r={exp}, got {r} \
                 (corner-aligned sampling would give {})",
                corner_aligned_r[ox]
            );
        }
    }

    // ── build_mip_pyramid ────────────────────────────────────────────────────

    #[test]
    fn test_build_mip_pyramid_level_count() {
        let img = vec![0.5_f32; 16 * 16 * 3];
        let levels = build_mip_pyramid(&img, 16, 16, 4).unwrap();
        // Should have ≤ 4 levels (stops at 1×1).
        assert!(levels.len() <= 4);
        assert!(!levels.is_empty());
    }

    #[test]
    fn test_build_mip_pyramid_level0_matches_input() {
        let img: Vec<f32> = (0..8 * 8 * 3).map(|i| (i as f32) * 0.01).collect();
        let levels = build_mip_pyramid(&img, 8, 8, 3).unwrap();
        let (ref pixels, w, h) = levels[0];
        assert_eq!(w, 8);
        assert_eq!(h, 8);
        assert_eq!(pixels, &img);
    }

    #[test]
    fn test_build_mip_pyramid_zero_levels_error() {
        let img = vec![0.5_f32; 4 * 4 * 3];
        assert!(build_mip_pyramid(&img, 4, 4, 0).is_err());
    }

    #[test]
    fn test_build_mip_pyramid_level1_reflects_blur() {
        // Regression guard for the borrow-avoiding refactor of the per-level
        // blur call: level 1 must actually be blurred-then-downsampled, not
        // just downsampled (which is what a silently-swallowed blur error
        // used to fall back to).
        let mut img = vec![0.0_f32; 8 * 8 * 3];
        let center = (4 * 8 + 4) * 3;
        img[center] = 1.0;
        img[center + 1] = 1.0;
        img[center + 2] = 1.0;

        let levels = build_mip_pyramid(&img, 8, 8, 2).expect("pyramid build failed");
        assert_eq!(levels.len(), 2);
        let (ref level1_pixels, w1, h1) = levels[1];
        assert_eq!((w1, h1), (4, 4));

        // A plain downsample of the *unblurred* image, for comparison: if
        // the per-level blur genuinely ran, level 1 must differ from this.
        let (unblurred_down, _, _) = downsample_2x(&img, 8, 8);
        assert_ne!(
            level1_pixels, &unblurred_down,
            "level 1 should reflect the blur pass, not a plain downsample of \
             the unblurred image"
        );
    }

    // ── HdrBloomConfig ───────────────────────────────────────────────────────

    #[test]
    fn test_bloom_config_validate_valid() {
        assert!(HdrBloomConfig::default().validate().is_ok());
    }

    #[test]
    fn test_bloom_config_validate_negative_threshold() {
        let cfg = HdrBloomConfig {
            threshold: -0.1,
            ..HdrBloomConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_bloom_config_validate_zero_num_levels() {
        let cfg = HdrBloomConfig {
            num_levels: 0,
            ..HdrBloomConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_bloom_config_validate_mismatched_weights() {
        let cfg = HdrBloomConfig {
            num_levels: 3,
            level_weights: vec![1.0, 1.0], // length 2, not 3
            ..HdrBloomConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_bloom_config_level_weight_uniform() {
        let cfg = HdrBloomConfig::default(); // no level_weights
        let expected = 1.0 / cfg.num_levels as f32;
        for i in 0..cfg.num_levels {
            assert!((cfg.level_weight(i) - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_bloom_config_level_weight_custom_normalised() {
        let cfg = HdrBloomConfig {
            num_levels: 3,
            level_weights: vec![1.0, 2.0, 3.0],
            ..HdrBloomConfig::default()
        };
        let sum: f32 = (0..3).map(|i| cfg.level_weight(i)).sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights should sum to 1, got {sum}"
        );
    }

    // ── apply_hdr_bloom ──────────────────────────────────────────────────────

    #[test]
    fn test_apply_bloom_dark_image_unchanged() {
        // All pixels below threshold → bloom adds nothing.
        let img = vec![0.3_f32; 8 * 8 * 3];
        let cfg = HdrBloomConfig {
            threshold: 1.0,
            knee: 0.0,
            intensity: 1.0,
            num_levels: 3,
            blur_radius: 1,
            blur_sigma: 1.0,
            level_weights: vec![],
        };
        let out = apply_hdr_bloom(&img, 8, 8, &cfg).unwrap();
        for (a, b) in img.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "dark image should not be changed by bloom"
            );
        }
    }

    #[test]
    fn test_apply_bloom_adds_positive_values_for_bright_pixels() {
        // Image with one very bright pixel in the middle.
        let mut img = vec![0.0_f32; 8 * 8 * 3];
        let cx = 4;
        let cy = 4;
        let base = (cy * 8 + cx) * 3;
        img[base] = 10.0;
        img[base + 1] = 10.0;
        img[base + 2] = 10.0;

        let cfg = HdrBloomConfig {
            threshold: 1.0,
            knee: 0.1,
            intensity: 1.0,
            num_levels: 3,
            blur_radius: 2,
            blur_sigma: 1.5,
            level_weights: vec![],
        };
        let out = apply_hdr_bloom(&img, 8, 8, &cfg).unwrap();
        // The sum of the output must exceed the sum of the input.
        let sum_in: f32 = img.iter().sum();
        let sum_out: f32 = out.iter().sum();
        assert!(sum_out >= sum_in, "bloom should add light, not remove it");
    }

    #[test]
    fn test_apply_hdr_bloom_non_power_of_two_stays_centered() {
        // Regression for a row-skewed upsample chain: on non-power-of-two
        // dimensions, the old chained-doubling upsample tracked clamped
        // dimensions that diverged from the real (always-doubling) buffer
        // size, and never actually re-cropped the mismatch, so the
        // composited bloom was a diagonally row-skewed reinterpretation of
        // the buffer instead of a soft glow around the source pixel.
        //
        // width/height are both non-power-of-two and mutually coprime-ish
        // so every pyramid level stays non-power-of-two too.
        let width = 37;
        let height = 23;
        let n = width * height;
        let mut img = vec![0.0_f32; n * 3];
        let cx = 20;
        let cy = 12;
        let base = (cy * width + cx) * 3;
        img[base] = 4.0;
        img[base + 1] = 4.0;
        img[base + 2] = 4.0;

        let cfg = HdrBloomConfig {
            threshold: 1.0,
            knee: 0.1,
            intensity: 1.0,
            num_levels: 4,
            blur_radius: 2,
            blur_sigma: 1.5,
            level_weights: vec![],
        };
        let out = apply_hdr_bloom(&img, width, height, &cfg).expect("bloom failed");
        assert_eq!(out.len(), n * 3);
        for &v in &out {
            assert!(v.is_finite(), "bloom output must stay finite, got {v}");
        }

        // Find the pixel with the largest bloom-only contribution
        // (out - image summed over channels) and check it lands at (or
        // very near) the source bright pixel rather than somewhere the
        // buffer's true stride happened to place it.
        let mut best_idx = 0usize;
        let mut best_val = f32::MIN;
        for i in 0..n {
            let b = i * 3;
            let contribution =
                (out[b] - img[b]) + (out[b + 1] - img[b + 1]) + (out[b + 2] - img[b + 2]);
            if contribution > best_val {
                best_val = contribution;
                best_idx = i;
            }
        }
        let peak_x = best_idx % width;
        let peak_y = best_idx / width;
        let dx = (peak_x as isize - cx as isize).abs();
        let dy = (peak_y as isize - cy as isize).abs();
        assert!(
            dx <= 2 && dy <= 2,
            "bloom peak should stay near the source pixel ({cx},{cy}); got \
             ({peak_x},{peak_y}) — a row-skewed buffer would displace this \
             far more than a couple of pixels"
        );
    }

    // ── apply_hdr_bloom_and_tonemap ──────────────────────────────────────────

    #[test]
    fn test_bloom_and_tonemap_returns_values_in_01() {
        let img: Vec<f32> = (0..4 * 4 * 3).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let cfg = HdrBloomConfig::tight();
        let out = apply_hdr_bloom_and_tonemap(&img, 4, 4, &cfg).unwrap();
        for &v in &out {
            assert!(
                (0.0..=1.0).contains(&v),
                "tone-mapped output must be in [0,1], got {v}"
            );
        }
    }

    // ── apply_hdr_bloom_rgba ─────────────────────────────────────────────────

    #[test]
    fn test_bloom_rgba_alpha_passthrough() {
        let mut img = vec![0.0_f32; 4 * 4 * 4];
        // Set alpha of each pixel to a distinctive value.
        for i in 0..16 {
            img[i * 4] = 2.0; // R (bright)
            img[i * 4 + 1] = 2.0; // G
            img[i * 4 + 2] = 2.0; // B
            img[i * 4 + 3] = 0.42 * (i as f32 / 16.0); // A (unique per pixel)
        }
        let cfg = HdrBloomConfig::default();
        let out = apply_hdr_bloom_rgba(&img, 4, 4, &cfg).unwrap();
        for i in 0..16 {
            let expected_alpha = img[i * 4 + 3];
            let actual_alpha = out[i * 4 + 3];
            assert!(
                (actual_alpha - expected_alpha).abs() < 1e-6,
                "alpha[{i}] should be unchanged: expected {expected_alpha}, got {actual_alpha}"
            );
        }
    }

    // ── compute_hdr_bloom_stats ──────────────────────────────────────────────

    #[test]
    fn test_bloom_stats_dark_image_bright_fraction_zero() {
        let img = vec![0.3_f32; 4 * 4 * 3];
        let cfg = HdrBloomConfig {
            threshold: 1.0,
            knee: 0.0,
            ..HdrBloomConfig::default()
        };
        let bloomed = apply_hdr_bloom(&img, 4, 4, &cfg).unwrap();
        let stats = compute_hdr_bloom_stats(&img, &bloomed, 4, 4, 1.0).unwrap();
        assert_eq!(
            stats.bright_fraction, 0.0,
            "no pixels above threshold → bright_fraction must be 0"
        );
    }

    #[test]
    fn test_bloom_stats_bright_image_nonzero_fraction() {
        // All pixels far above threshold.
        let img = vec![2.0_f32; 4 * 4 * 3];
        let cfg = HdrBloomConfig {
            threshold: 1.0,
            ..HdrBloomConfig::default()
        };
        let bloomed = apply_hdr_bloom(&img, 4, 4, &cfg).unwrap();
        let stats = compute_hdr_bloom_stats(&img, &bloomed, 4, 4, 1.0).unwrap();
        assert_eq!(
            stats.bright_fraction, 1.0,
            "all pixels above threshold → bright_fraction must be 1"
        );
        assert!(
            stats.mean_bloom_luminance >= 0.0,
            "mean bloom luminance should be non-negative"
        );
    }
}
