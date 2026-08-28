//! f32-RGB post-processing chromatic aberration API.
//!
//! This module provides a higher-level, opinionated API for simulating
//! camera chromatic aberration as a post-processing pass on f32 RGB images
//! stored in row-major HxWx3 layout.
//!
//! For the lower-level lateral/longitudinal API see
//! [`crate::chromatic_aberration`].

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the f32-RGB post-processing chromatic aberration API.
#[derive(Debug, Error, PartialEq)]
pub enum ChromAberrationError {
    /// Buffer size does not match declared dimensions × channels.
    #[error("Image dimensions mismatch: buffer size {got} != {width}×{height}×{channels}")]
    SizeMismatch {
        /// Actual buffer length.
        got: usize,
        /// Declared width.
        width: usize,
        /// Declared height.
        height: usize,
        /// Expected channel count.
        channels: usize,
    },
    /// Strength value is outside `[0, 1]`.
    #[error("Invalid strength value {0}: must be in [0, 1]")]
    InvalidStrength(f32),
    /// Generic invalid parameter.
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Config types
// ─────────────────────────────────────────────────────────────────────────────

/// Interpolation mode used when sampling channel data at fractional pixel positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromInterpolation {
    /// Return the value at the nearest integer pixel coordinate.
    Nearest,
    /// Bilinear (4-sample weighted) interpolation.
    Bilinear,
}

/// Configuration for the f32-RGB chromatic aberration post-processing effect.
#[derive(Debug, Clone, PartialEq)]
pub struct ChromAberrationConfig {
    /// Radial shift strength (0.0 = none, 1.0 = maximum).
    pub strength: f32,
    /// Lateral shift for the red channel in pixels `[dx, dy]`.
    pub red_shift: [f32; 2],
    /// Lateral shift for the blue channel in pixels `[dx, dy]`.
    pub blue_shift: [f32; 2],
    /// Apply barrel distortion before channel shifting.
    pub barrel_distortion: bool,
    /// Barrel/pincushion distortion coefficient k1.
    /// Negative → barrel; positive → pincushion.
    pub barrel_k1: f32,
    /// Higher-order barrel distortion coefficient k2.
    pub barrel_k2: f32,
    /// Pixel-sampling interpolation mode.
    pub interpolation: ChromInterpolation,
}

impl Default for ChromAberrationConfig {
    fn default() -> Self {
        Self {
            strength: 0.02,
            red_shift: [2.0, 0.0],
            blue_shift: [-2.0, 0.0],
            barrel_distortion: false,
            barrel_k1: -0.1,
            barrel_k2: 0.0,
            interpolation: ChromInterpolation::Bilinear,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that the image buffer matches `width × height × 3`.
#[inline]
fn check_rgb_buffer(img: &[f32], width: usize, height: usize) -> Result<(), ChromAberrationError> {
    let expected = width * height * 3;
    if img.len() != expected {
        return Err(ChromAberrationError::SizeMismatch {
            got: img.len(),
            width,
            height,
            channels: 3,
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Public sampling helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Bilinear sample a single channel from an interleaved f32 image.
///
/// `img` is row-major HWC with `channels` channels per pixel.
/// `x` and `y` are fractional pixel coordinates, clamped to image bounds.
pub fn bilinear_sample_channel(
    img: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    channel: usize,
    x: f32,
    y: f32,
) -> f32 {
    if width == 0 || height == 0 || channels == 0 || channel >= channels {
        return 0.0;
    }
    let xc = x.clamp(0.0, width as f32 - 1.0);
    let yc = y.clamp(0.0, height as f32 - 1.0);
    let x0 = xc.floor() as usize;
    let y0 = yc.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = xc - x0 as f32;
    let fy = yc - y0 as f32;

    let get = |row: usize, col: usize| -> f32 {
        img.get(row * width * channels + col * channels + channel)
            .copied()
            .unwrap_or(0.0)
    };
    let c00 = get(y0, x0);
    let c10 = get(y0, x1);
    let c01 = get(y1, x0);
    let c11 = get(y1, x1);

    let top = c00 * (1.0 - fx) + c10 * fx;
    let bot = c01 * (1.0 - fx) + c11 * fx;
    top * (1.0 - fy) + bot * fy
}

/// Nearest-neighbour sample a single channel from an interleaved f32 image.
///
/// `img` is row-major HWC with `channels` channels per pixel.
/// `x` and `y` are pixel coordinates, clamped to image bounds.
pub fn nearest_sample_channel(
    img: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    channel: usize,
    x: f32,
    y: f32,
) -> f32 {
    if width == 0 || height == 0 || channels == 0 || channel >= channels {
        return 0.0;
    }
    let xi = (x.round() as isize).clamp(0, width as isize - 1) as usize;
    let yi = (y.round() as isize).clamp(0, height as isize - 1) as usize;
    img.get(yi * width * channels + xi * channels + channel)
        .copied()
        .unwrap_or(0.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Core post-processing functions
// ─────────────────────────────────────────────────────────────────────────────

/// Shift a single colour channel by `(dx, dy)` pixels using the given interpolation.
///
/// `img` is row-major HWC RGB (3 channels). Returns a new buffer of the same
/// dimensions with only the target channel shifted; the other two channels are
/// copied verbatim.
///
/// # Errors
/// Returns [`ChromAberrationError::SizeMismatch`] if `img.len() != width * height * 3`.
pub fn shift_channel(
    img: &[f32],
    width: usize,
    height: usize,
    channel: usize,
    dx: f32,
    dy: f32,
    interp: ChromInterpolation,
) -> Result<Vec<f32>, ChromAberrationError> {
    check_rgb_buffer(img, width, height)?;
    if channel >= 3 {
        return Err(ChromAberrationError::InvalidParam(format!(
            "channel index {channel} out of range (0..2)"
        )));
    }
    let mut out = img.to_vec();

    for py in 0..height {
        for px in 0..width {
            let src_x = px as f32 - dx;
            let src_y = py as f32 - dy;
            let val = match interp {
                ChromInterpolation::Bilinear => {
                    bilinear_sample_channel(img, width, height, 3, channel, src_x, src_y)
                }
                ChromInterpolation::Nearest => {
                    nearest_sample_channel(img, width, height, 3, channel, src_x, src_y)
                }
            };
            if let Some(slot) = out.get_mut((py * width + px) * 3 + channel) {
                *slot = val;
            }
        }
    }
    Ok(out)
}

/// Apply barrel/pincushion lens distortion to a f32 RGB image.
///
/// The Brown–Conrady model defines the *forward* (undistorted → distorted)
/// mapping `r_dst = r_src × (1 + k1·r_src² + k2·r_src⁴)`. For each output
/// (distorted-space) pixel this function needs the *inverse* — which
/// source radius produced this destination radius — and approximates it in
/// one step as `r_src ≈ r_dst / (1 + k1·r_dst² + k2·r_dst⁴)` (substituting
/// `r_dst` for the unknown `r_src` inside the correction term). This is
/// exact only when `k1`/`k2` are small; for strong distortion or when exact
/// convergence matters, use [`crate::lens_distortion::undistort_point_iterative`],
/// which Newton-iterates to the true inverse instead. The image centre is
/// taken as `((width - 1) / 2, (height - 1) / 2)` (pixel-index convention,
/// matching e.g. OpenCV's `cx`/`cy`).
///
/// Pixels outside the source image boundary are clamped to the edge.
///
/// # Errors
/// Returns [`ChromAberrationError::SizeMismatch`] if `img.len() != width * height * 3`.
pub fn apply_barrel_distortion(
    img: &[f32],
    width: usize,
    height: usize,
    k1: f32,
    k2: f32,
) -> Result<Vec<f32>, ChromAberrationError> {
    check_rgb_buffer(img, width, height)?;
    if width == 0 || height == 0 {
        return Ok(img.to_vec());
    }
    let cx = (width as f32 - 1.0) * 0.5;
    let cy = (height as f32 - 1.0) * 0.5;
    // Normalisation radius: half-diagonal.
    let norm_r = (cx * cx + cy * cy).sqrt().max(1.0);

    let mut out = vec![0.0f32; width * height * 3];

    for py in 0..height {
        for px in 0..width {
            let nx = (px as f32 - cx) / norm_r;
            let ny = (py as f32 - cy) / norm_r;
            let r2 = nx * nx + ny * ny;
            let factor = 1.0 + k1 * r2 + k2 * r2 * r2;
            // Guard against degenerate factor values.
            let safe_factor = if factor.abs() < 1e-6 {
                1e-6f32.copysign(factor)
            } else {
                factor
            };
            let src_nx = nx / safe_factor;
            let src_ny = ny / safe_factor;
            let src_x = src_nx * norm_r + cx;
            let src_y = src_ny * norm_r + cy;

            let pixel_out = (py * width + px) * 3;
            for ch in 0..3 {
                let val = bilinear_sample_channel(img, width, height, 3, ch, src_x, src_y);
                if let Some(slot) = out.get_mut(pixel_out + ch) {
                    *slot = val;
                }
            }
        }
    }
    Ok(out)
}

/// Apply chromatic aberration to a f32 RGB image (HxWx3 layout, values in `[0,1]`).
///
/// 1. If `config.barrel_distortion` is `true`, the barrel warp is applied first.
/// 2. Green channel is copied verbatim (reference channel).
/// 3. Red channel is shifted by `red_shift × strength`.
/// 4. Blue channel is shifted by `blue_shift × strength`.
///
/// # Errors
/// - [`ChromAberrationError::SizeMismatch`] — buffer length mismatch.
/// - [`ChromAberrationError::InvalidStrength`] — strength outside `[0, 1]`.
pub fn apply_chrom_aberration(
    img: &[f32],
    width: usize,
    height: usize,
    config: &ChromAberrationConfig,
) -> Result<Vec<f32>, ChromAberrationError> {
    check_rgb_buffer(img, width, height)?;
    if !(0.0..=1.0).contains(&config.strength) {
        return Err(ChromAberrationError::InvalidStrength(config.strength));
    }

    // Optional barrel warp first.
    let base: Vec<f32>;
    let src = if config.barrel_distortion {
        base = apply_barrel_distortion(img, width, height, config.barrel_k1, config.barrel_k2)?;
        &base
    } else {
        img
    };

    let rdx = config.red_shift[0] * config.strength;
    let rdy = config.red_shift[1] * config.strength;
    let bdx = config.blue_shift[0] * config.strength;
    let bdy = config.blue_shift[1] * config.strength;

    let mut out = src.to_vec();

    // Shift red channel.
    for py in 0..height {
        for px in 0..width {
            let src_x = px as f32 - rdx;
            let src_y = py as f32 - rdy;
            let val = match config.interpolation {
                ChromInterpolation::Bilinear => {
                    bilinear_sample_channel(src, width, height, 3, 0, src_x, src_y)
                }
                ChromInterpolation::Nearest => {
                    nearest_sample_channel(src, width, height, 3, 0, src_x, src_y)
                }
            };
            if let Some(slot) = out.get_mut((py * width + px) * 3) {
                *slot = val;
            }
        }
    }

    // Shift blue channel.
    for py in 0..height {
        for px in 0..width {
            let src_x = px as f32 - bdx;
            let src_y = py as f32 - bdy;
            let val = match config.interpolation {
                ChromInterpolation::Bilinear => {
                    bilinear_sample_channel(src, width, height, 3, 2, src_x, src_y)
                }
                ChromInterpolation::Nearest => {
                    nearest_sample_channel(src, width, height, 3, 2, src_x, src_y)
                }
            };
            if let Some(slot) = out.get_mut((py * width + px) * 3 + 2) {
                *slot = val;
            }
        }
    }

    Ok(out)
}

/// Apply radial chromatic aberration (shift increases with distance from centre).
///
/// For each pixel at normalised position `(nx, ny)` in `[-1, 1]`:
/// - Red channel: sample at `(nx, ny) × (1 + strength × r)` (pushed outward).
/// - Blue channel: sample at `(nx, ny) × (1 − strength × r)` (pulled inward).
/// - Green channel: unchanged.
///
/// `center` is the normalised centre in `[0, 1]²` (default `[0.5, 0.5]` = image centre).
///
/// # Errors
/// - [`ChromAberrationError::SizeMismatch`] — buffer length mismatch.
/// - [`ChromAberrationError::InvalidStrength`] — strength outside `[0, 1]`.
pub fn apply_radial_chromatic_aberration(
    img: &[f32],
    width: usize,
    height: usize,
    strength: f32,
    center: [f32; 2],
) -> Result<Vec<f32>, ChromAberrationError> {
    check_rgb_buffer(img, width, height)?;
    if !(0.0..=1.0).contains(&strength) {
        return Err(ChromAberrationError::InvalidStrength(strength));
    }
    if width == 0 || height == 0 {
        return Ok(img.to_vec());
    }

    let cx = center[0] * (width as f32 - 1.0);
    let cy = center[1] * (height as f32 - 1.0);
    // Half-diagonal as normalisation radius, derived from the same
    // pixel-index centre (`cx`, `cy`) used above rather than a separate
    // `width * 0.5` pixel-extent quantity — mixing the two conventions
    // meant `r` never quite reached the value a corner pixel should map to.
    // Matches `apply_barrel_distortion` and `compute_radial_distances`.
    let hnorm = (cx * cx + cy * cy).sqrt().max(1.0);

    let mut out = img.to_vec();

    for py in 0..height {
        for px in 0..width {
            let nx = (px as f32 - cx) / hnorm;
            let ny = (py as f32 - cy) / hnorm;
            let r = (nx * nx + ny * ny).sqrt();

            // Red: push outward.
            let r_scale = 1.0 + strength * r;
            let r_src_x = cx + nx * r_scale * hnorm;
            let r_src_y = cy + ny * r_scale * hnorm;
            let r_val = bilinear_sample_channel(img, width, height, 3, 0, r_src_x, r_src_y);

            // Blue: pull inward.
            let b_scale = (1.0 - strength * r).max(0.0);
            let b_src_x = cx + nx * b_scale * hnorm;
            let b_src_y = cy + ny * b_scale * hnorm;
            let b_val = bilinear_sample_channel(img, width, height, 3, 2, b_src_x, b_src_y);

            let pixel_out = (py * width + px) * 3;
            if let Some(slot) = out.get_mut(pixel_out) {
                *slot = r_val;
            }
            // Green (index 1) is already copied verbatim from img.to_vec().
            if let Some(slot) = out.get_mut(pixel_out + 2) {
                *slot = b_val;
            }
        }
    }
    Ok(out)
}

/// Compute the normalised radial distance from `center` for every pixel.
///
/// Returns a flat `width × height` slice. Values range from `0.0` at the
/// specified centre to `1.0` at the farthest corner of the image.
///
/// `center` is normalised: `[0.5, 0.5]` = exact image centre.
pub fn compute_radial_distances(width: usize, height: usize, center: [f32; 2]) -> Vec<f32> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let cx = center[0] * (width as f32 - 1.0);
    let cy = center[1] * (height as f32 - 1.0);

    // Find maximum distance (to a corner) for normalisation.
    let corners = [
        (0.0f32, 0.0f32),
        (width as f32 - 1.0, 0.0),
        (0.0, height as f32 - 1.0),
        (width as f32 - 1.0, height as f32 - 1.0),
    ];
    let max_dist = corners
        .iter()
        .map(|(cx2, cy2)| ((cx2 - cx).powi(2) + (cy2 - cy).powi(2)).sqrt())
        .fold(0.0f32, f32::max)
        .max(1.0);

    let mut out = vec![0.0f32; width * height];
    for py in 0..height {
        for px in 0..width {
            let dx = px as f32 - cx;
            let dy = py as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt() / max_dist;
            if let Some(slot) = out.get_mut(py * width + px) {
                *slot = dist;
            }
        }
    }
    out
}

/// Apply lens vignetting (darkening toward corners) to a f32 RGB image.
///
/// - `strength` in `[0, 1]`: 0 = no effect, 1 = fully black corners.
/// - `falloff` is the exponent of the vignette curve (2.0 = quadratic).
///
/// # Errors
/// - [`ChromAberrationError::SizeMismatch`] — buffer length mismatch.
/// - [`ChromAberrationError::InvalidStrength`] — strength outside `[0, 1]`.
/// - [`ChromAberrationError::InvalidParam`] — falloff ≤ 0.
pub fn apply_vignetting_f32(
    img: &[f32],
    width: usize,
    height: usize,
    strength: f32,
    falloff: f32,
) -> Result<Vec<f32>, ChromAberrationError> {
    check_rgb_buffer(img, width, height)?;
    if !(0.0..=1.0).contains(&strength) {
        return Err(ChromAberrationError::InvalidStrength(strength));
    }
    if falloff <= 0.0 {
        return Err(ChromAberrationError::InvalidParam(format!(
            "falloff must be > 0, got {falloff}"
        )));
    }
    let mask = compute_vignette_mask(width, height, strength, falloff);
    let mut out = img.to_vec();
    for py in 0..height {
        for px in 0..width {
            let m = mask.get(py * width + px).copied().unwrap_or(1.0);
            let pixel = (py * width + px) * 3;
            for ch in 0..3 {
                if let Some(slot) = out.get_mut(pixel + ch) {
                    *slot *= m;
                }
            }
        }
    }
    Ok(out)
}

/// Compute the vignette mask: 1.0 at centre, decreasing toward corners.
///
/// `strength` ∈ `[0, 1]` controls how dark the corners become.
/// `falloff` is the power-law exponent (2.0 = quadratic falloff).
pub fn compute_vignette_mask(width: usize, height: usize, strength: f32, falloff: f32) -> Vec<f32> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let dists = compute_radial_distances(width, height, [0.5, 0.5]);
    dists
        .iter()
        .map(|&d| {
            let factor = 1.0 - strength * d.powf(falloff);
            factor.clamp(0.0, 1.0)
        })
        .collect()
}

/// Estimate chromatic aberration strength from an image using channel-difference analysis.
///
/// Returns a value in `[0, ∞)` proportional to the mean absolute difference
/// between the red/blue channels and the green reference channel, normalised by
/// the overall image variance. A perfectly neutral image returns `0.0`.
///
/// # Errors
/// Returns [`ChromAberrationError::SizeMismatch`] if `img.len() != width * height * 3`.
pub fn estimate_aberration_strength(
    img: &[f32],
    width: usize,
    height: usize,
) -> Result<f32, ChromAberrationError> {
    check_rgb_buffer(img, width, height)?;
    let n_px = width * height;
    if n_px == 0 {
        return Ok(0.0);
    }

    let mut sum_rg = 0.0f64;
    let mut sum_bg = 0.0f64;
    let mut sum_var = 0.0f64;
    let mut mean = [0.0f64; 3];

    for i in 0..n_px {
        let r = img.get(i * 3).copied().unwrap_or(0.0) as f64;
        let g = img.get(i * 3 + 1).copied().unwrap_or(0.0) as f64;
        let b = img.get(i * 3 + 2).copied().unwrap_or(0.0) as f64;
        mean[0] += r;
        mean[1] += g;
        mean[2] += b;
        sum_rg += (r - g).abs();
        sum_bg += (b - g).abs();
    }
    for m in mean.iter_mut() {
        *m /= n_px as f64;
    }

    // Overall variance as sum of per-channel variances.
    for i in 0..n_px {
        let r = img.get(i * 3).copied().unwrap_or(0.0) as f64;
        let g = img.get(i * 3 + 1).copied().unwrap_or(0.0) as f64;
        let b = img.get(i * 3 + 2).copied().unwrap_or(0.0) as f64;
        sum_var += (r - mean[0]).powi(2) + (g - mean[1]).powi(2) + (b - mean[2]).powi(2);
    }
    let variance = sum_var / (n_px as f64 * 3.0);

    let mean_diff = (sum_rg + sum_bg) / (2.0 * n_px as f64);
    let normalised = if variance > 1e-10 {
        mean_diff / (variance.sqrt() + 1e-10)
    } else {
        0.0
    };
    Ok(normalised as f32)
}

/// Format a [`ChromAberrationConfig`] for human-readable display.
pub fn format_chrom_config(config: &ChromAberrationConfig) -> String {
    format!(
        "ChromAberrationConfig {{ strength: {:.4}, red_shift: [{:.2}, {:.2}], \
         blue_shift: [{:.2}, {:.2}], barrel_distortion: {}, barrel_k1: {:.4}, \
         barrel_k2: {:.4}, interpolation: {} }}",
        config.strength,
        config.red_shift[0],
        config.red_shift[1],
        config.blue_shift[0],
        config.blue_shift[1],
        config.barrel_distortion,
        config.barrel_k1,
        config.barrel_k2,
        match config.interpolation {
            ChromInterpolation::Nearest => "Nearest",
            ChromInterpolation::Bilinear => "Bilinear",
        },
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn make_rgb_gradient(w: usize, h: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; w * h * 3];
        for py in 0..h {
            for px in 0..w {
                let r = px as f32 / w.max(1) as f32;
                let g = py as f32 / h.max(1) as f32;
                let b = 1.0 - r;
                let idx = (py * w + px) * 3;
                out[idx] = r;
                out[idx + 1] = g;
                out[idx + 2] = b;
            }
        }
        out
    }

    // ── 1. compute_radial_distances: centre pixel is 0.0 ─────────────────────

    #[test]
    fn test_radial_distances_center_is_zero() {
        let w = 11usize;
        let h = 11usize;
        let dists = compute_radial_distances(w, h, [0.5, 0.5]);
        assert_eq!(dists.len(), w * h);
        let cx = w / 2;
        let cy = h / 2;
        let centre = dists[cy * w + cx];
        assert!(
            approx_eq(centre, 0.0, 1e-5),
            "Centre distance should be 0.0, got {centre}"
        );
    }

    // ── 2. compute_radial_distances: corners have distance ≈ 1.0 ─────────────

    #[test]
    fn test_radial_distances_corners_near_one() {
        let w = 11usize;
        let h = 11usize;
        let dists = compute_radial_distances(w, h, [0.5, 0.5]);
        let corner = dists[0]; // top-left
        assert!(
            corner > 0.9 && corner <= 1.0,
            "Corner distance should be near 1.0, got {corner}"
        );
    }

    // ── 3. compute_radial_distances: all values in [0, 1] ────────────────────

    #[test]
    fn test_radial_distances_in_range() {
        let dists = compute_radial_distances(20, 15, [0.5, 0.5]);
        for &d in &dists {
            assert!((0.0..=1.0).contains(&d), "Distance out of range: {d}");
        }
    }

    // ── 4. compute_radial_distances: zero dimensions → empty ─────────────────

    #[test]
    fn test_radial_distances_zero_dims() {
        assert!(compute_radial_distances(0, 10, [0.5, 0.5]).is_empty());
        assert!(compute_radial_distances(10, 0, [0.5, 0.5]).is_empty());
    }

    // ── 5. bilinear_sample_channel: integer position returns exact value ───────

    #[test]
    fn test_bilinear_sample_integer_exact() {
        let img = vec![
            1.0f32, 0.5, 0.0, 0.2, 0.4, 0.6, 0.3, 0.7, 0.1, 0.9, 0.0, 0.5,
        ];
        let val = bilinear_sample_channel(&img, 2, 2, 3, 0, 0.0, 0.0);
        assert!(approx_eq(val, 1.0, 1e-6), "Expected 1.0, got {val}");
        let val2 = bilinear_sample_channel(&img, 2, 2, 3, 1, 1.0, 0.0);
        assert!(approx_eq(val2, 0.4, 1e-6), "Expected 0.4, got {val2}");
    }

    // ── 6. bilinear_sample_channel: midpoint is average of neighbours ──────────

    #[test]
    fn test_bilinear_sample_midpoint_average() {
        let img = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
        let val = bilinear_sample_channel(&img, 2, 1, 3, 0, 0.5, 0.0);
        assert!(
            approx_eq(val, 0.5, 1e-5),
            "Midpoint should be 0.5, got {val}"
        );
    }

    // ── 7. bilinear_sample_channel: clamping at border ────────────────────────

    #[test]
    fn test_bilinear_sample_clamping() {
        let img = vec![0.5f32; 4 * 4 * 3];
        let val = bilinear_sample_channel(&img, 4, 4, 3, 0, -1.0, -1.0);
        assert!(
            approx_eq(val, 0.5, 1e-5),
            "Clamped sample should be 0.5, got {val}"
        );
        let val2 = bilinear_sample_channel(&img, 4, 4, 3, 0, 100.0, 100.0);
        assert!(
            approx_eq(val2, 0.5, 1e-5),
            "Clamped sample out-of-bounds should be 0.5, got {val2}"
        );
    }

    // ── 8. nearest_sample_channel: rounds to nearest integer pixel ────────────

    #[test]
    fn test_nearest_sample_channel_basic() {
        let img = vec![0.2f32, 0.0, 0.0, 0.8, 0.0, 0.0];
        let v0 = nearest_sample_channel(&img, 2, 1, 3, 0, 0.1, 0.0);
        assert!(approx_eq(v0, 0.2, 1e-6), "Should round to px 0: got {v0}");
        let v1 = nearest_sample_channel(&img, 2, 1, 3, 0, 0.9, 0.0);
        assert!(approx_eq(v1, 0.8, 1e-6), "Should round to px 1: got {v1}");
    }

    // ── 9. nearest_sample_channel: out-of-bounds clamped ─────────────────────

    #[test]
    fn test_nearest_sample_channel_clamp() {
        let img = vec![0.3f32; 3 * 3 * 3];
        let val = nearest_sample_channel(&img, 3, 3, 3, 0, -5.0, -5.0);
        assert!(approx_eq(val, 0.3, 1e-6), "Clamped nearest sample: {val}");
    }

    // ── 10. shift_channel: shift by 0 → identical ─────────────────────────────

    #[test]
    fn test_shift_channel_zero_shift_identical() {
        let img = make_rgb_gradient(8, 8);
        let out = shift_channel(&img, 8, 8, 0, 0.0, 0.0, ChromInterpolation::Bilinear)
            .expect("shift_channel failed");
        for (a, b) in img.iter().zip(out.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-5),
                "Zero shift should be identity: {a} vs {b}"
            );
        }
    }

    // ── 11. shift_channel: shift by 1 pixel moves channel ─────────────────────

    #[test]
    fn test_shift_channel_one_pixel() {
        let w = 4usize;
        let h = 4usize;
        let img = make_rgb_gradient(w, h);
        let out = shift_channel(&img, w, h, 0, 1.0, 0.0, ChromInterpolation::Bilinear)
            .expect("shift_channel failed");
        let original_px0_r = img[0];
        let shifted_px1_r = out[3];
        assert!(
            approx_eq(original_px0_r, shifted_px1_r, 1e-4),
            "Shifted R at px=1 should match original at px=0: {original_px0_r} vs {shifted_px1_r}"
        );
    }

    // ── 12. shift_channel: size mismatch error ────────────────────────────────

    #[test]
    fn test_shift_channel_size_mismatch() {
        let err = shift_channel(&[0.5f32; 5], 4, 4, 0, 0.0, 0.0, ChromInterpolation::Nearest);
        assert!(matches!(
            err,
            Err(ChromAberrationError::SizeMismatch { .. })
        ));
    }

    // ── 13. shift_channel: out-of-bounds channel index → error ───────────────

    #[test]
    fn test_shift_channel_invalid_channel() {
        let img = vec![0.5f32; 4 * 4 * 3];
        let err = shift_channel(&img, 4, 4, 5, 0.0, 0.0, ChromInterpolation::Nearest);
        assert!(matches!(err, Err(ChromAberrationError::InvalidParam(_))));
    }

    // ── 14. apply_barrel_distortion: k1=0, k2=0 → identity ──────────────────

    #[test]
    fn test_barrel_distortion_identity() {
        let img = make_rgb_gradient(10, 10);
        let out =
            apply_barrel_distortion(&img, 10, 10, 0.0, 0.0).expect("barrel_distortion failed");
        for (a, b) in img.iter().zip(out.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-4),
                "k1=k2=0 should be identity: {a} vs {b}"
            );
        }
    }

    // ── 15. apply_barrel_distortion: non-zero k1 changes image ───────────────

    #[test]
    fn test_barrel_distortion_nonzero_changes_image() {
        let img = make_rgb_gradient(12, 12);
        let out =
            apply_barrel_distortion(&img, 12, 12, -0.2, 0.0).expect("barrel_distortion failed");
        let n_diff = img
            .iter()
            .zip(out.iter())
            .filter(|&(a, b)| (a - b).abs() > 1e-4)
            .count();
        assert!(n_diff > 0, "Non-zero k1 should change some pixels");
    }

    // ── 16. apply_barrel_distortion: size mismatch error ─────────────────────

    #[test]
    fn test_barrel_distortion_size_mismatch() {
        let err = apply_barrel_distortion(&[0.5f32; 7], 4, 4, 0.0, 0.0);
        assert!(matches!(
            err,
            Err(ChromAberrationError::SizeMismatch { .. })
        ));
    }

    // ── 17. apply_barrel_distortion: output size equals input size ────────────

    #[test]
    fn test_barrel_distortion_output_size() {
        let img = make_rgb_gradient(8, 6);
        let out = apply_barrel_distortion(&img, 8, 6, -0.1, 0.0).expect("barrel_distortion failed");
        assert_eq!(out.len(), img.len());
    }

    // ── 18. apply_barrel_distortion: centre pixel near-unchanged ─────────────

    #[test]
    fn test_barrel_distortion_center_unchanged() {
        let w = 9usize;
        let h = 9usize;
        let img = make_rgb_gradient(w, h);
        let out = apply_barrel_distortion(&img, w, h, -0.2, 0.0).expect("barrel distortion failed");
        let cx = w / 2;
        let cy = h / 2;
        let orig = &img[(cy * w + cx) * 3..(cy * w + cx) * 3 + 3];
        let result = &out[(cy * w + cx) * 3..(cy * w + cx) * 3 + 3];
        for (a, b) in orig.iter().zip(result.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-3),
                "Centre pixel should be near-unchanged: {a} vs {b}"
            );
        }
    }

    // ── 19. apply_vignetting_f32: strength=0 → unchanged ─────────────────────

    #[test]
    fn test_vignetting_zero_strength_unchanged() {
        let img = make_rgb_gradient(8, 8);
        let out = apply_vignetting_f32(&img, 8, 8, 0.0, 2.0).expect("vignetting failed");
        for (a, b) in img.iter().zip(out.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-5),
                "Strength=0 should be identity: {a} vs {b}"
            );
        }
    }

    // ── 20. apply_vignetting_f32: strength=1 → corners darkened ──────────────

    #[test]
    fn test_vignetting_full_strength_corners_dark() {
        let w = 11usize;
        let h = 11usize;
        let img = vec![1.0f32; w * h * 3];
        let out = apply_vignetting_f32(&img, w, h, 1.0, 2.0).expect("vignetting failed");
        let corner_r = out[0];
        let cx = w / 2;
        let cy = h / 2;
        let centre_r = out[(cy * w + cx) * 3];
        assert!(
            centre_r > corner_r,
            "Centre ({centre_r}) should be brighter than corner ({corner_r})"
        );
        assert!(
            approx_eq(centre_r, 1.0, 1e-5),
            "Centre should be 1.0 at strength=1: {centre_r}"
        );
    }

    // ── 21. compute_vignette_mask: centre = 1.0 ──────────────────────────────

    #[test]
    fn test_vignette_mask_center_is_one() {
        let w = 9usize;
        let h = 9usize;
        let mask = compute_vignette_mask(w, h, 0.8, 2.0);
        let cx = w / 2;
        let cy = h / 2;
        let val = mask[cy * w + cx];
        assert!(approx_eq(val, 1.0, 1e-5), "Centre should be 1.0, got {val}");
    }

    // ── 22. compute_vignette_mask: corners < 1.0 for strength > 0 ─────────────

    #[test]
    fn test_vignette_mask_corners_less_than_one() {
        let mask = compute_vignette_mask(10, 10, 0.8, 2.0);
        let corner = mask[0];
        assert!(corner < 1.0, "Corner should be < 1.0, got {corner}");
    }

    // ── 23. compute_vignette_mask: all values in [0, 1] ──────────────────────

    #[test]
    fn test_vignette_mask_values_in_range() {
        let mask = compute_vignette_mask(16, 12, 0.7, 2.5);
        for &v in &mask {
            assert!((0.0..=1.0).contains(&v), "Mask value out of range: {v}");
        }
    }

    // ── 24. apply_vignetting_f32: invalid falloff → error ─────────────────────

    #[test]
    fn test_vignetting_invalid_falloff() {
        let img = vec![0.5f32; 4 * 4 * 3];
        let err = apply_vignetting_f32(&img, 4, 4, 0.5, 0.0);
        assert!(matches!(err, Err(ChromAberrationError::InvalidParam(_))));
    }

    // ── 25. apply_vignetting_f32: size mismatch error ─────────────────────────

    #[test]
    fn test_vignetting_size_mismatch() {
        let err = apply_vignetting_f32(&[0.5f32; 5], 4, 4, 0.5, 2.0);
        assert!(matches!(
            err,
            Err(ChromAberrationError::SizeMismatch { .. })
        ));
    }

    // ── 26. apply_chrom_aberration: zero strength → unchanged ─────────────────

    #[test]
    fn test_chrom_aberration_zero_strength() {
        let img = make_rgb_gradient(10, 10);
        let config = ChromAberrationConfig {
            strength: 0.0,
            ..ChromAberrationConfig::default()
        };
        let out =
            apply_chrom_aberration(&img, 10, 10, &config).expect("apply_chrom_aberration failed");
        for (a, b) in img.iter().zip(out.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-5),
                "Zero strength should preserve image: {a} vs {b}"
            );
        }
    }

    // ── 27. apply_chrom_aberration: nonzero strength → R, B differ ────────────

    #[test]
    fn test_chrom_aberration_nonzero_differs() {
        let img = make_rgb_gradient(16, 16);
        let config = ChromAberrationConfig {
            strength: 1.0,
            ..ChromAberrationConfig::default()
        };
        let out =
            apply_chrom_aberration(&img, 16, 16, &config).expect("apply_chrom_aberration failed");
        let r_changed = (0..16 * 16).any(|i| (out[i * 3] - img[i * 3]).abs() > 1e-5);
        let b_changed = (0..16 * 16).any(|i| (out[i * 3 + 2] - img[i * 3 + 2]).abs() > 1e-5);
        assert!(r_changed, "Red channel should be shifted");
        assert!(b_changed, "Blue channel should be shifted");
    }

    // ── 28. apply_chrom_aberration: invalid strength → error ──────────────────

    #[test]
    fn test_chrom_aberration_invalid_strength() {
        let img = vec![0.5f32; 4 * 4 * 3];
        let config = ChromAberrationConfig {
            strength: 1.5,
            ..ChromAberrationConfig::default()
        };
        let err = apply_chrom_aberration(&img, 4, 4, &config);
        assert!(matches!(err, Err(ChromAberrationError::InvalidStrength(_))));
    }

    // ── 29. apply_chrom_aberration: size mismatch → error ─────────────────────

    #[test]
    fn test_chrom_aberration_size_mismatch() {
        let config = ChromAberrationConfig::default();
        let err = apply_chrom_aberration(&[0.5f32; 5], 4, 4, &config);
        assert!(matches!(
            err,
            Err(ChromAberrationError::SizeMismatch { .. })
        ));
    }

    // ── 30. apply_chrom_aberration: 1×1 image (edge case) ────────────────────

    #[test]
    fn test_chrom_aberration_one_by_one() {
        let img = vec![0.7f32, 0.3, 0.5];
        let config = ChromAberrationConfig {
            strength: 1.0,
            ..ChromAberrationConfig::default()
        };
        let out = apply_chrom_aberration(&img, 1, 1, &config).expect("1×1 chrom aberration failed");
        assert_eq!(out.len(), 3);
        for (a, b) in img.iter().zip(out.iter()) {
            assert!(approx_eq(*a, *b, 1e-5), "1×1 should clamp: {a} vs {b}");
        }
    }

    // ── 31. apply_chrom_aberration: with barrel distortion ───────────────────

    #[test]
    fn test_chrom_aberration_with_barrel() {
        let img = make_rgb_gradient(12, 12);
        let config = ChromAberrationConfig {
            strength: 0.5,
            barrel_distortion: true,
            barrel_k1: -0.1,
            ..ChromAberrationConfig::default()
        };
        let out = apply_chrom_aberration(&img, 12, 12, &config)
            .expect("apply_chrom_aberration with barrel failed");
        assert_eq!(out.len(), img.len());
    }

    // ── 32. apply_chrom_aberration: nearest interpolation mode ───────────────

    #[test]
    fn test_chrom_aberration_nearest_interp() {
        let img = make_rgb_gradient(8, 8);
        let config = ChromAberrationConfig {
            strength: 0.5,
            interpolation: ChromInterpolation::Nearest,
            ..ChromAberrationConfig::default()
        };
        let out = apply_chrom_aberration(&img, 8, 8, &config).expect("nearest interp failed");
        assert_eq!(out.len(), img.len());
    }

    // ── 33. apply_radial_chromatic_aberration: centre pixels least affected ────

    #[test]
    fn test_radial_ca_center_least_affected() {
        let w = 13usize;
        let h = 13usize;
        let img = make_rgb_gradient(w, h);
        let out = apply_radial_chromatic_aberration(&img, w, h, 0.5, [0.5, 0.5])
            .expect("radial CA failed");
        let cx = w / 2;
        let cy = h / 2;
        let center_diff = (0..3)
            .map(|ch| (out[(cy * w + cx) * 3 + ch] - img[(cy * w + cx) * 3 + ch]).abs())
            .fold(0.0f32, f32::max);
        let corner_diff = (0..3)
            .map(|ch| (out[ch] - img[ch]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            center_diff <= corner_diff + 1e-5,
            "Centre ({center_diff}) should be less affected than corner ({corner_diff})"
        );
    }

    // ── 34. apply_radial_chromatic_aberration: invalid strength → error ────────

    #[test]
    fn test_radial_ca_invalid_strength() {
        let img = vec![0.5f32; 4 * 4 * 3];
        let err = apply_radial_chromatic_aberration(&img, 4, 4, -0.1, [0.5, 0.5]);
        assert!(matches!(err, Err(ChromAberrationError::InvalidStrength(_))));
    }

    // ── 35. apply_radial_chromatic_aberration: output same size as input ───────

    #[test]
    fn test_radial_ca_output_size() {
        let img = make_rgb_gradient(10, 8);
        let out = apply_radial_chromatic_aberration(&img, 10, 8, 0.1, [0.5, 0.5])
            .expect("radial CA failed");
        assert_eq!(out.len(), img.len());
    }

    // ── 35b. apply_radial_chromatic_aberration: r reaches exactly 1.0 at the
    //         farthest corner (consistent centre/normalisation convention) ────

    #[test]
    fn test_radial_ca_corner_reaches_full_normalized_radius() {
        // Regression for mixing a (width-1)-based centre with a
        // width*0.5-based normalisation radius: with `hnorm` derived from
        // the same (width-1)-based `cx`/`cy` as the centre, a centred
        // effect's normalised radius r is *exactly* 1.0 at the farthest
        // corner (odd width/height put the centre exactly on a pixel, so
        // this is exact, not approximate). At strength=1.0 the blue
        // channel's pull-inward scale `(1 - strength*r).max(0.0)` is then
        // exactly 0 at the corner, collapsing its sample position to
        // exactly the image centre. Under the pre-fix mismatched
        // normalisation, r at the corner was strictly < 1.0, so this exact
        // collapse never happened and the corner kept some of its own
        // (unpulled) blue value instead.
        let w = 11usize;
        let h = 7usize;
        let img = make_rgb_gradient(w, h);
        let out = apply_radial_chromatic_aberration(&img, w, h, 1.0, [0.5, 0.5])
            .expect("radial CA failed");

        let center_blue = img[((h / 2) * w + (w / 2)) * 3 + 2];
        let corner_out_blue = out[2]; // pixel (0,0), blue channel
        assert!(
            (corner_out_blue - center_blue).abs() < 1e-4,
            "at strength=1.0 the farthest corner's blue channel should be \
             pulled exactly to the image centre's blue value ({center_blue}), \
             got {corner_out_blue}"
        );
    }

    // ── 36. apply_radial_chromatic_aberration: green unchanged at centre ───────

    #[test]
    fn test_radial_ca_green_unchanged_at_center() {
        let w = 11usize;
        let h = 11usize;
        let img = make_rgb_gradient(w, h);
        let out = apply_radial_chromatic_aberration(&img, w, h, 0.5, [0.5, 0.5])
            .expect("radial CA failed");
        let cx = w / 2;
        let cy = h / 2;
        let orig_g = img[(cy * w + cx) * 3 + 1];
        let out_g = out[(cy * w + cx) * 3 + 1];
        assert!(
            approx_eq(orig_g, out_g, 1e-5),
            "Green at centre should be unchanged: {orig_g} vs {out_g}"
        );
    }

    // ── 37. estimate_aberration_strength: known shifted image → positive ──────

    #[test]
    fn test_estimate_aberration_strength_positive() {
        let w = 20usize;
        let h = 20usize;
        let base = make_rgb_gradient(w, h);
        let config = ChromAberrationConfig {
            strength: 1.0,
            red_shift: [2.0, 0.0],
            blue_shift: [-2.0, 0.0],
            ..ChromAberrationConfig::default()
        };
        let aberrated =
            apply_chrom_aberration(&base, w, h, &config).expect("apply_chrom_aberration failed");
        let score = estimate_aberration_strength(&aberrated, w, h).expect("estimate failed");
        assert!(
            score > 0.0,
            "Aberrated image should have positive score, got {score}"
        );
    }

    // ── 38. estimate_aberration_strength: size mismatch → error ──────────────

    #[test]
    fn test_estimate_aberration_strength_size_mismatch() {
        let err = estimate_aberration_strength(&[0.5f32; 5], 4, 4);
        assert!(matches!(
            err,
            Err(ChromAberrationError::SizeMismatch { .. })
        ));
    }

    // ── 39. estimate_aberration_strength: uniform image → near zero ───────────

    #[test]
    fn test_estimate_aberration_strength_uniform_zero() {
        let img = vec![0.5f32; 8 * 8 * 3];
        let score = estimate_aberration_strength(&img, 8, 8).expect("estimate failed");
        assert!(
            score <= 1e-5,
            "Uniform image should have near-zero aberration score, got {score}"
        );
    }

    // ── 40. format_chrom_config: non-empty string ─────────────────────────────

    #[test]
    fn test_format_chrom_config_non_empty() {
        let config = ChromAberrationConfig::default();
        let s = format_chrom_config(&config);
        assert!(
            !s.is_empty(),
            "format_chrom_config should return a non-empty string"
        );
        assert!(
            s.contains("strength"),
            "String should mention 'strength': {s}"
        );
    }

    // ── 41. format_chrom_config: contains interpolation mode ──────────────────

    #[test]
    fn test_format_chrom_config_contains_interp() {
        let config = ChromAberrationConfig {
            interpolation: ChromInterpolation::Nearest,
            ..ChromAberrationConfig::default()
        };
        let s = format_chrom_config(&config);
        assert!(s.contains("Nearest"), "Should mention 'Nearest': {s}");
    }

    // ── 42. ChromAberrationError::SizeMismatch display ───────────────────────

    #[test]
    fn test_chrom_aberration_error_display() {
        let err = ChromAberrationError::SizeMismatch {
            got: 5,
            width: 4,
            height: 4,
            channels: 3,
        };
        let s = err.to_string();
        assert!(
            s.contains("5"),
            "Error message should mention actual size: {s}"
        );
    }

    // ── 43. shift_channel: large shift clamps to border value ────────────────

    #[test]
    fn test_shift_channel_large_shift_clamps() {
        let w = 4usize;
        let h = 4usize;
        let img = make_rgb_gradient(w, h);
        let out = shift_channel(&img, w, h, 0, -1000.0, 0.0, ChromInterpolation::Bilinear)
            .expect("shift failed");
        let last_px_r = img[(h / 2 * w + (w - 1)) * 3];
        let shifted_mid_r = out[(h / 2 * w) * 3];
        assert!(
            approx_eq(shifted_mid_r, last_px_r, 1e-4),
            "Large shift should clamp: expected {last_px_r}, got {shifted_mid_r}"
        );
    }

    // ── 44. nearest_sample_channel: 1×1 image returns single value ───────────

    #[test]
    fn test_nearest_sample_one_by_one() {
        let img = vec![0.42f32, 0.0, 0.0];
        let val = nearest_sample_channel(&img, 1, 1, 3, 0, 0.0, 0.0);
        assert!(approx_eq(val, 0.42, 1e-6));
        let val2 = nearest_sample_channel(&img, 1, 1, 3, 0, 5.0, 5.0);
        assert!(approx_eq(val2, 0.42, 1e-6));
    }

    // ── 45. compute_radial_distances: non-centre center parameter ─────────────

    #[test]
    fn test_radial_distances_off_center() {
        let w = 10usize;
        let h = 10usize;
        let dists = compute_radial_distances(w, h, [0.0, 0.0]);
        assert_eq!(dists.len(), w * h);
        let top_left = dists[0];
        assert!(
            approx_eq(top_left, 0.0, 1e-5),
            "Top-left should be 0.0 for center=[0,0]: {top_left}"
        );
        let bottom_right = dists[(h - 1) * w + (w - 1)];
        assert!(
            bottom_right > 0.9,
            "Bottom-right should be far from [0,0] center: {bottom_right}"
        );
    }

    // ── 46. apply_radial_chromatic_aberration: size mismatch → error ──────────

    #[test]
    fn test_radial_ca_size_mismatch() {
        let err = apply_radial_chromatic_aberration(&[0.5f32; 5], 4, 4, 0.1, [0.5, 0.5]);
        assert!(matches!(
            err,
            Err(ChromAberrationError::SizeMismatch { .. })
        ));
    }

    // ── 47. apply_vignetting_f32: invalid strength → error ────────────────────

    #[test]
    fn test_vignetting_invalid_strength() {
        let img = vec![0.5f32; 4 * 4 * 3];
        let err = apply_vignetting_f32(&img, 4, 4, -0.1, 2.0);
        assert!(matches!(err, Err(ChromAberrationError::InvalidStrength(_))));
    }

    // ── 48. apply_chrom_aberration: output has same size as input ─────────────

    #[test]
    fn test_chrom_aberration_output_size() {
        let img = make_rgb_gradient(15, 10);
        let config = ChromAberrationConfig::default();
        let out =
            apply_chrom_aberration(&img, 15, 10, &config).expect("apply_chrom_aberration failed");
        assert_eq!(out.len(), img.len());
    }

    // ── 49. bilinear_sample_channel: out-of-range channel → 0.0 ──────────────

    #[test]
    fn test_bilinear_sample_invalid_channel() {
        let img = vec![0.5f32; 4 * 4 * 3];
        let val = bilinear_sample_channel(&img, 4, 4, 3, 5, 0.0, 0.0);
        assert!(
            approx_eq(val, 0.0, 1e-6),
            "Invalid channel should return 0.0: {val}"
        );
    }

    // ── 50. nearest_sample_channel: out-of-range channel → 0.0 ──────────────

    #[test]
    fn test_nearest_sample_invalid_channel() {
        let img = vec![0.5f32; 4 * 4 * 3];
        let val = nearest_sample_channel(&img, 4, 4, 3, 5, 0.0, 0.0);
        assert!(
            approx_eq(val, 0.0, 1e-6),
            "Invalid channel should return 0.0: {val}"
        );
    }

    // ── 51. compute_vignette_mask: zero dimensions → empty ───────────────────

    #[test]
    fn test_vignette_mask_zero_dims() {
        assert!(compute_vignette_mask(0, 10, 0.5, 2.0).is_empty());
        assert!(compute_vignette_mask(10, 0, 0.5, 2.0).is_empty());
    }

    // ── 52. apply_barrel_distortion: k2 also works ───────────────────────────

    #[test]
    fn test_barrel_distortion_with_k2() {
        let img = make_rgb_gradient(10, 10);
        let out = apply_barrel_distortion(&img, 10, 10, -0.1, 0.05)
            .expect("barrel_distortion with k2 failed");
        assert_eq!(out.len(), img.len());
        let n_diff = img
            .iter()
            .zip(out.iter())
            .filter(|&(a, b)| (a - b).abs() > 1e-4)
            .count();
        assert!(n_diff > 0, "k2 should also produce changes");
    }

    // ── 53. apply_chrom_aberration: strength clamped at 0 and 1 boundaries ────

    #[test]
    fn test_chrom_aberration_boundary_strength() {
        let img = make_rgb_gradient(8, 8);
        for s in &[0.0f32, 1.0f32] {
            let config = ChromAberrationConfig {
                strength: *s,
                ..ChromAberrationConfig::default()
            };
            let out =
                apply_chrom_aberration(&img, 8, 8, &config).expect("boundary strength failed");
            assert_eq!(out.len(), img.len());
        }
    }

    // ── 54. estimate_aberration_strength: returns finite value ───────────────

    #[test]
    fn test_estimate_aberration_strength_finite() {
        let img = make_rgb_gradient(12, 12);
        let score = estimate_aberration_strength(&img, 12, 12).expect("estimate failed");
        assert!(score.is_finite(), "Score should be finite: {score}");
        assert!(score >= 0.0, "Score should be non-negative: {score}");
    }

    // ── 55. shift_channel: all three channels can be shifted ─────────────────

    #[test]
    fn test_shift_all_channels() {
        let img = make_rgb_gradient(6, 6);
        for ch in 0..3 {
            let out = shift_channel(&img, 6, 6, ch, 1.0, 0.0, ChromInterpolation::Bilinear)
                .unwrap_or_else(|_| panic!("shift_channel failed for channel {ch}"));
            assert_eq!(out.len(), img.len());
        }
    }

    // ── 56. format_chrom_config: barrel_distortion reflected in output ────────

    #[test]
    fn test_format_chrom_config_barrel_flag() {
        let config = ChromAberrationConfig {
            barrel_distortion: true,
            ..ChromAberrationConfig::default()
        };
        let s = format_chrom_config(&config);
        assert!(
            s.contains("true"),
            "Should show barrel_distortion=true: {s}"
        );
    }

    // ── 57. compute_radial_distances: 1×1 image → single zero element ─────────

    #[test]
    fn test_radial_distances_one_by_one() {
        let dists = compute_radial_distances(1, 1, [0.5, 0.5]);
        assert_eq!(dists.len(), 1);
        // Only pixel is the "centre"; distance should be 0.
        assert!(
            approx_eq(dists[0], 0.0, 1e-5),
            "1×1 centre should be 0.0: {}",
            dists[0]
        );
    }

    // ── 58. apply_barrel_distortion: 1×1 image returns same pixel ────────────

    #[test]
    fn test_barrel_distortion_one_by_one() {
        let img = vec![0.4f32, 0.6, 0.8];
        let out = apply_barrel_distortion(&img, 1, 1, -0.5, 0.1).expect("barrel 1×1 failed");
        assert_eq!(out.len(), 3);
        for (a, b) in img.iter().zip(out.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-5),
                "1×1 barrel should be identity: {a} vs {b}"
            );
        }
    }

    // ── 59. apply_radial_chromatic_aberration: zero strength → same as input ───

    #[test]
    fn test_radial_ca_zero_strength_unchanged() {
        let img = make_rgb_gradient(8, 8);
        let out = apply_radial_chromatic_aberration(&img, 8, 8, 0.0, [0.5, 0.5])
            .expect("radial CA zero strength failed");
        for (a, b) in img.iter().zip(out.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-5),
                "Zero strength should preserve image: {a} vs {b}"
            );
        }
    }

    // ── 60. apply_vignetting_f32: quadratic vs linear falloff differ ──────────

    #[test]
    fn test_vignetting_falloff_differences() {
        // Use a larger image so that non-corner pixels have intermediate distances.
        let w = 21usize;
        let h = 21usize;
        let img = vec![1.0f32; w * h * 3];
        let out2 = apply_vignetting_f32(&img, w, h, 0.8, 1.0).expect("linear vignetting failed");
        let out4 = apply_vignetting_f32(&img, w, h, 0.8, 3.0).expect("cubic vignetting failed");
        // A pixel halfway to the corner should differ between falloff=1 and falloff=3.
        // The half-way pixel is not exactly at the corner so d < 1.0, so d^1 != d^3.
        let half_px = (h / 2 * w + w / 4) * 3; // midway between centre and edge
        let v1 = out2.get(half_px).copied().unwrap_or(0.0);
        let v3 = out4.get(half_px).copied().unwrap_or(0.0);
        assert!(
            (v1 - v3).abs() > 1e-5,
            "Different falloff should produce different mid-image values: {v1} vs {v3}"
        );
    }
}
