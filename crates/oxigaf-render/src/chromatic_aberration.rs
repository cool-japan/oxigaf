//! Chromatic aberration post-processing for rendered 3DGS images.
//!
//! Chromatic aberration is a lens defect where different wavelengths of light
//! focus at different distances or positions, causing colour fringing at edges.
//!
//! # Types
//!
//! - **Lateral (transverse)**: different wavelengths are displaced radially from
//!   the image centre.  Modelled here as a per-channel zoom / radial distortion.
//! - **Longitudinal (axial)**: different wavelengths have different focus planes,
//!   resulting in per-channel defocus blur.
//!
//! # Quick start
//!
//! ```rust
//! use oxigaf_render::chromatic_aberration::{
//!     LateralChromaticConfig, LongitudinalChromaticConfig,
//!     apply_chromatic_aberration,
//! };
//!
//! let width  = 4usize;
//! let height = 4usize;
//! let image  = vec![0.5f32; width * height * 3];
//! let lateral      = LateralChromaticConfig::default();
//! let longitudinal = LongitudinalChromaticConfig::default();
//! let output = apply_chromatic_aberration(&image, width, height, &lateral, &longitudinal)
//!     .expect("CA failed");
//! assert_eq!(output.len(), width * height * 3);
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by chromatic-aberration operations.
#[derive(Debug, Error, PartialEq)]
pub enum ChromaticError {
    /// Invalid configuration parameter.
    #[error("Invalid chromatic aberration config: {0}")]
    InvalidConfig(String),

    /// Image slice length does not match declared dimensions.
    #[error("Wrong pixel count: got {got}, expected {expected} ({w}×{h}×3)")]
    InvalidImage {
        /// Actual length of the slice.
        got: usize,
        /// Expected length.
        expected: usize,
        /// Image width.
        w: usize,
        /// Image height.
        h: usize,
    },

    /// Image slice (or dimension) is empty.
    #[error("Image is empty")]
    EmptyImage,
}

// ─────────────────────────────────────────────────────────────────────────────
// Radial distortion helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the radial distortion factor for a given normalised radius.
///
/// The factor is `1 + k1·r² + k2·r⁴`, matching the Brown–Conrady model used
/// in camera calibration.
///
/// # Parameters
/// - `r`  – Normalised radius from the image centre (0 at centre, ≈1 at corner).
/// - `k1` – First radial distortion coefficient.
/// - `k2` – Second radial distortion coefficient.
#[inline]
pub fn radial_factor(r: f32, k1: f32, k2: f32) -> f32 {
    let r2 = r * r;
    1.0 + k1 * r2 + k2 * r2 * r2
}

/// Map a pixel from distorted to undistorted coordinates using a simple
/// radial distortion model.
///
/// # Parameters
/// - `px`, `py` – Source pixel coordinates.
/// - `cx`, `cy` – Image centre in pixels.
/// - `k1`, `k2` – Radial distortion coefficients (positive → barrel).
///
/// # Returns
/// `(undistorted_x, undistorted_y)` in the same pixel coordinate system.
pub fn undistort_pixel(px: f32, py: f32, cx: f32, cy: f32, k1: f32, k2: f32) -> (f32, f32) {
    let dx = px - cx;
    let dy = py - cy;
    // Normalise radius by the larger of cx, cy (half-diagonal approximation).
    let norm = cx.max(cy).max(1.0);
    let r = (dx * dx + dy * dy).sqrt() / norm;
    let factor = radial_factor(r, k1, k2);
    (cx + dx * factor, cy + dy * factor)
}

// ─────────────────────────────────────────────────────────────────────────────
// LateralChromaticConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for lateral (transverse) chromatic aberration.
///
/// Each colour channel is sampled from a slightly different zoom level.
/// A scale > 1 zooms the channel out (samples further from centre); < 1 zooms in.
#[derive(Debug, Clone, PartialEq)]
pub struct LateralChromaticConfig {
    /// Zoom scale for the red channel (1.0 = no shift).
    pub red_scale: f32,
    /// Zoom scale for the green channel (reference, usually 1.0).
    pub green_scale: f32,
    /// Zoom scale for the blue channel.
    pub blue_scale: f32,
    /// Radial distortion coefficient k1 for the red channel.
    pub red_k1: f32,
    /// Radial distortion coefficient k1 for the green channel.
    pub green_k1: f32,
    /// Radial distortion coefficient k1 for the blue channel.
    pub blue_k1: f32,
}

impl Default for LateralChromaticConfig {
    fn default() -> Self {
        Self {
            red_scale: 1.005,
            green_scale: 1.0,
            blue_scale: 0.995,
            red_k1: 0.0,
            green_k1: 0.0,
            blue_k1: 0.0,
        }
    }
}

impl LateralChromaticConfig {
    /// Validate that all scale factors are strictly positive.
    ///
    /// # Errors
    ///
    /// Returns [`ChromaticError::InvalidConfig`] if any scale is ≤ 0.
    pub fn validate(&self) -> Result<(), ChromaticError> {
        if self.red_scale <= 0.0 {
            return Err(ChromaticError::InvalidConfig(format!(
                "red_scale must be > 0, got {}",
                self.red_scale
            )));
        }
        if self.green_scale <= 0.0 {
            return Err(ChromaticError::InvalidConfig(format!(
                "green_scale must be > 0, got {}",
                self.green_scale
            )));
        }
        if self.blue_scale <= 0.0 {
            return Err(ChromaticError::InvalidConfig(format!(
                "blue_scale must be > 0, got {}",
                self.blue_scale
            )));
        }
        Ok(())
    }

    /// Strong chromatic aberration preset.
    pub fn strong() -> Self {
        Self {
            red_scale: 1.015,
            green_scale: 1.0,
            blue_scale: 0.985,
            ..Self::default()
        }
    }

    /// Subtle chromatic aberration preset.
    pub fn subtle() -> Self {
        Self {
            red_scale: 1.002,
            green_scale: 1.0,
            blue_scale: 0.998,
            ..Self::default()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LongitudinalChromaticConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for longitudinal (axial) chromatic aberration.
///
/// Each colour channel is blurred by a different amount to simulate the
/// different focus planes of different wavelengths.
#[derive(Debug, Clone, PartialEq)]
pub struct LongitudinalChromaticConfig {
    /// Gaussian blur sigma for the red channel (pixels).
    pub red_blur: f32,
    /// Gaussian blur sigma for the green channel (pixels).
    pub green_blur: f32,
    /// Gaussian blur sigma for the blue channel (pixels).
    pub blue_blur: f32,
    /// Blend weight between original and blurred channels
    /// (0.0 = no effect, 1.0 = fully blurred fringe).
    pub fringe_strength: f32,
}

impl Default for LongitudinalChromaticConfig {
    fn default() -> Self {
        Self {
            red_blur: 0.0,
            green_blur: 0.0,
            blue_blur: 0.5,
            fringe_strength: 0.3,
        }
    }
}

impl LongitudinalChromaticConfig {
    /// Validate that all blur values and fringe_strength are non-negative.
    ///
    /// # Errors
    ///
    /// Returns [`ChromaticError::InvalidConfig`] for any negative value.
    pub fn validate(&self) -> Result<(), ChromaticError> {
        if self.red_blur < 0.0 {
            return Err(ChromaticError::InvalidConfig(format!(
                "red_blur must be >= 0, got {}",
                self.red_blur
            )));
        }
        if self.green_blur < 0.0 {
            return Err(ChromaticError::InvalidConfig(format!(
                "green_blur must be >= 0, got {}",
                self.green_blur
            )));
        }
        if self.blue_blur < 0.0 {
            return Err(ChromaticError::InvalidConfig(format!(
                "blue_blur must be >= 0, got {}",
                self.blue_blur
            )));
        }
        if self.fringe_strength < 0.0 {
            return Err(ChromaticError::InvalidConfig(format!(
                "fringe_strength must be >= 0, got {}",
                self.fringe_strength
            )));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ChromaticStats
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics for the aberration introduced by a [`LateralChromaticConfig`].
#[derive(Debug, Clone, PartialEq)]
pub struct ChromaticStats {
    /// Maximum per-pixel red-channel shift in pixels.
    pub max_red_shift_pixels: f32,
    /// Maximum per-pixel blue-channel shift in pixels.
    pub max_blue_shift_pixels: f32,
    /// Mean shift magnitude (red + blue channels averaged) across all pixels.
    pub mean_shift_pixels: f32,
    /// Fraction of pixels where the total shift exceeds 2 pixels.
    pub strong_fringe_fraction: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Bilinear sample of a single interleaved channel (`stride` channels, `channel_idx` ∈ [0, stride)).
///
/// Coordinates are clamped to `[0, width-1] × [0, height-1]` (clamp-to-edge).
fn bilinear_sample_channel_plane(
    channel: &[f32],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
) -> f32 {
    // Guard degenerate images.
    if width == 0 || height == 0 || channel.is_empty() {
        return 0.0;
    }

    let w = width as f32;
    let h = height as f32;

    // Clamp continuous coordinates.
    let xc = x.clamp(0.0, w - 1.0);
    let yc = y.clamp(0.0, h - 1.0);

    let x0 = xc.floor() as usize;
    let y0 = yc.floor() as usize;
    // Clamp integer coordinates to valid range.
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let fx = xc - x0 as f32;
    let fy = yc - y0 as f32;

    // Pixel index helpers.
    let idx = |row: usize, col: usize| -> f32 {
        // channel slice is already a single-channel plane: width * height floats.
        channel.get(row * width + col).copied().unwrap_or(0.0)
    };

    let c00 = idx(y0, x0);
    let c10 = idx(y0, x1);
    let c01 = idx(y1, x0);
    let c11 = idx(y1, x1);

    // Bilinear blend.
    let top = c00 * (1.0 - fx) + c10 * fx;
    let bottom = c01 * (1.0 - fx) + c11 * fx;
    top * (1.0 - fy) + bottom * fy
}

/// Separable 1-D Gaussian blur of a single-channel plane.
///
/// If `sigma < 0.1` the input is returned unchanged (no-op).
fn gaussian_blur_channel(channel: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.1 || width == 0 || height == 0 {
        return channel.to_vec();
    }

    // Build normalised 1-D kernel.
    let radius = (sigma * 3.0).ceil() as usize;
    let kernel_len = 2 * radius + 1;
    let mut kernel = vec![0.0f32; kernel_len];
    let two_sigma2 = 2.0 * sigma * sigma;
    let mut sum = 0.0f32;
    for i in 0..kernel_len {
        let d = (i as f32) - (radius as f32);
        let w = (-d * d / two_sigma2).exp();
        // Safe: index is in bounds by construction.
        if let Some(slot) = kernel.get_mut(i) {
            *slot = w;
        }
        sum += w;
    }
    // Normalise.
    if sum > 0.0 {
        for w in kernel.iter_mut() {
            *w /= sum;
        }
    }

    // Horizontal pass.
    let mut tmp = vec![0.0f32; width * height];
    for row in 0..height {
        for col in 0..width {
            let mut acc = 0.0f32;
            for (ki, &kw) in kernel.iter().enumerate() {
                let src_col = (col as isize + ki as isize - radius as isize)
                    .clamp(0, width as isize - 1) as usize;
                let px_val = channel.get(row * width + src_col).copied().unwrap_or(0.0);
                acc += kw * px_val;
            }
            if let Some(slot) = tmp.get_mut(row * width + col) {
                *slot = acc;
            }
        }
    }

    // Vertical pass.
    let mut out = vec![0.0f32; width * height];
    for row in 0..height {
        for col in 0..width {
            let mut acc = 0.0f32;
            for (ki, &kw) in kernel.iter().enumerate() {
                let src_row = (row as isize + ki as isize - radius as isize)
                    .clamp(0, height as isize - 1) as usize;
                let px_val = tmp.get(src_row * width + col).copied().unwrap_or(0.0);
                acc += kw * px_val;
            }
            if let Some(slot) = out.get_mut(row * width + col) {
                *slot = acc;
            }
        }
    }

    out
}

/// Validate common image arguments; returns `(cx, cy)` image centre.
fn validate_image(
    image: &[f32],
    width: usize,
    height: usize,
) -> Result<(f32, f32), ChromaticError> {
    if width == 0 || height == 0 || image.is_empty() {
        return Err(ChromaticError::EmptyImage);
    }
    let expected = width * height * 3;
    if image.len() != expected {
        return Err(ChromaticError::InvalidImage {
            got: image.len(),
            expected,
            w: width,
            h: height,
        });
    }
    Ok(((width as f32 - 1.0) * 0.5, (height as f32 - 1.0) * 0.5))
}

// ─────────────────────────────────────────────────────────────────────────────
// Core public functions
// ─────────────────────────────────────────────────────────────────────────────

/// Apply lateral (transverse) chromatic aberration.
///
/// Each colour channel is sampled from a slightly different zoom level around
/// the image centre. Pixels are bilinearly interpolated with clamp-to-edge.
///
/// # Parameters
/// - `image`  – RGB f32 slice in row-major HWC order (len = `width * height * 3`).
/// - `width`  – Image width in pixels.
/// - `height` – Image height in pixels.
/// - `config` – Lateral CA configuration.
///
/// # Errors
///
/// - [`ChromaticError::EmptyImage`] if any dimension is zero or the slice is empty.
/// - [`ChromaticError::InvalidImage`] if the slice length mismatches dimensions.
pub fn apply_lateral_ca(
    image: &[f32],
    width: usize,
    height: usize,
    config: &LateralChromaticConfig,
) -> Result<Vec<f32>, ChromaticError> {
    let (cx, cy) = validate_image(image, width, height)?;
    config.validate()?;

    // Split interleaved image into per-channel planes.
    let n_pixels = width * height;
    let mut r_plane = vec![0.0f32; n_pixels];
    let mut g_plane = vec![0.0f32; n_pixels];
    let mut b_plane = vec![0.0f32; n_pixels];

    for i in 0..n_pixels {
        r_plane[i] = image.get(i * 3).copied().unwrap_or(0.0);
        g_plane[i] = image.get(i * 3 + 1).copied().unwrap_or(0.0);
        b_plane[i] = image.get(i * 3 + 2).copied().unwrap_or(0.0);
    }

    let scales = [config.red_scale, config.green_scale, config.blue_scale];
    let planes = [r_plane.as_slice(), g_plane.as_slice(), b_plane.as_slice()];

    let mut output = vec![0.0f32; n_pixels * 3];

    for py in 0..height {
        for px in 0..width {
            let fx = px as f32;
            let fy = py as f32;
            let pixel_idx = py * width + px;

            for (ch, (&scale, plane)) in scales.iter().zip(planes.iter()).enumerate() {
                // Source coordinates: divide displacement by scale so that a
                // scale > 1 zooms the channel out (samples further from centre).
                let src_x = cx + (fx - cx) / scale;
                let src_y = cy + (fy - cy) / scale;

                let sampled = bilinear_sample_channel_plane(plane, width, height, src_x, src_y);
                if let Some(slot) = output.get_mut(pixel_idx * 3 + ch) {
                    *slot = sampled;
                }
            }
        }
    }

    Ok(output)
}

/// Apply longitudinal (axial) chromatic aberration.
///
/// Each colour channel is blurred by a different Gaussian sigma, then blended
/// back with the original according to `fringe_strength`.
///
/// # Parameters
/// - `image`  – RGB f32 slice in row-major HWC order.
/// - `width`  – Image width in pixels.
/// - `height` – Image height in pixels.
/// - `config` – Longitudinal CA configuration.
///
/// # Errors
///
/// - [`ChromaticError::EmptyImage`] if any dimension is zero or the slice is empty.
/// - [`ChromaticError::InvalidImage`] if the slice length mismatches dimensions.
pub fn apply_longitudinal_ca(
    image: &[f32],
    width: usize,
    height: usize,
    config: &LongitudinalChromaticConfig,
) -> Result<Vec<f32>, ChromaticError> {
    let _ = validate_image(image, width, height)?;
    config.validate()?;

    let n_pixels = width * height;

    // Split into per-channel planes.
    let mut r_orig = vec![0.0f32; n_pixels];
    let mut g_orig = vec![0.0f32; n_pixels];
    let mut b_orig = vec![0.0f32; n_pixels];

    for i in 0..n_pixels {
        r_orig[i] = image.get(i * 3).copied().unwrap_or(0.0);
        g_orig[i] = image.get(i * 3 + 1).copied().unwrap_or(0.0);
        b_orig[i] = image.get(i * 3 + 2).copied().unwrap_or(0.0);
    }

    // Blur each channel by its respective sigma.
    let r_blurred = gaussian_blur_channel(&r_orig, width, height, config.red_blur);
    let g_blurred = gaussian_blur_channel(&g_orig, width, height, config.green_blur);
    let b_blurred = gaussian_blur_channel(&b_orig, width, height, config.blue_blur);

    let fs = config.fringe_strength.clamp(0.0, 1.0);
    let fs_inv = 1.0 - fs;

    // Blend original with blurred and recombine.
    let mut output = vec![0.0f32; n_pixels * 3];
    for i in 0..n_pixels {
        let r = fs_inv * r_orig.get(i).copied().unwrap_or(0.0)
            + fs * r_blurred.get(i).copied().unwrap_or(0.0);
        let g = fs_inv * g_orig.get(i).copied().unwrap_or(0.0)
            + fs * g_blurred.get(i).copied().unwrap_or(0.0);
        let b = fs_inv * b_orig.get(i).copied().unwrap_or(0.0)
            + fs * b_blurred.get(i).copied().unwrap_or(0.0);

        if let Some(slot) = output.get_mut(i * 3) {
            *slot = r;
        }
        if let Some(slot) = output.get_mut(i * 3 + 1) {
            *slot = g;
        }
        if let Some(slot) = output.get_mut(i * 3 + 2) {
            *slot = b;
        }
    }

    Ok(output)
}

/// Apply both lateral and longitudinal chromatic aberration in sequence.
///
/// Applies lateral CA first, then longitudinal CA on the result.
///
/// # Errors
///
/// Propagates errors from [`apply_lateral_ca`] and [`apply_longitudinal_ca`].
pub fn apply_chromatic_aberration(
    image: &[f32],
    width: usize,
    height: usize,
    lateral: &LateralChromaticConfig,
    longitudinal: &LongitudinalChromaticConfig,
) -> Result<Vec<f32>, ChromaticError> {
    let after_lateral = apply_lateral_ca(image, width, height, lateral)?;
    apply_longitudinal_ca(&after_lateral, width, height, longitudinal)
}

// ─────────────────────────────────────────────────────────────────────────────
// Aberration map & fringe visualisation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a per-pixel aberration magnitude map.
///
/// For each pixel the map contains the distance (in pixels) between the source
/// coordinates of the red and green channels under the lateral CA model.
/// The output is normalised to `[0, 1]` by the maximum magnitude in the image.
///
/// Returns a `width * height` float slice.
pub fn aberration_map(width: usize, height: usize, config: &LateralChromaticConfig) -> Vec<f32> {
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let cx = (width as f32 - 1.0) * 0.5;
    let cy = (height as f32 - 1.0) * 0.5;
    let n = width * height;
    let mut magnitudes = vec![0.0f32; n];
    let mut max_mag = 0.0f32;

    for py in 0..height {
        for px in 0..width {
            let fx = px as f32;
            let fy = py as f32;

            // Red-channel source.
            let r_src_x = cx + (fx - cx) / config.red_scale;
            let r_src_y = cy + (fy - cy) / config.red_scale;
            // Green-channel source.
            let g_src_x = cx + (fx - cx) / config.green_scale;
            let g_src_y = cy + (fy - cy) / config.green_scale;

            let dx = r_src_x - g_src_x;
            let dy = r_src_y - g_src_y;
            let mag = (dx * dx + dy * dy).sqrt();

            if let Some(slot) = magnitudes.get_mut(py * width + px) {
                *slot = mag;
            }
            if mag > max_mag {
                max_mag = mag;
            }
        }
    }

    // Normalise.
    if max_mag > 0.0 {
        for m in magnitudes.iter_mut() {
            *m /= max_mag;
        }
    }

    magnitudes
}

/// Compute a per-pixel fringe colour visualisation.
///
/// Returns an RGB image (`width * height * 3` floats in `[0, 1]`) encoding the
/// colour-shift direction:
/// - Red channel:   normalised red shift relative to green (> 0.5 = red fringe)
/// - Green channel: always 0.5 (neutral reference)
/// - Blue channel:  normalised blue shift relative to green (< 0.5 = blue fringe)
pub fn fringe_visualization(
    width: usize,
    height: usize,
    config: &LateralChromaticConfig,
) -> Vec<f32> {
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let cx = (width as f32 - 1.0) * 0.5;
    let cy = (height as f32 - 1.0) * 0.5;
    let n = width * height;
    let mut output = vec![0.0f32; n * 3];

    // First pass: compute raw shifts and find extremes for normalisation.
    let mut red_shifts = vec![0.0f32; n];
    let mut blue_shifts = vec![0.0f32; n];
    let mut max_abs = 0.0f32;

    for py in 0..height {
        for px in 0..width {
            let fx = px as f32;
            let fy = py as f32;

            let r_src_x = cx + (fx - cx) / config.red_scale;
            let r_src_y = cy + (fy - cy) / config.red_scale;
            let g_src_x = cx + (fx - cx) / config.green_scale;
            let g_src_y = cy + (fy - cy) / config.green_scale;
            let b_src_x = cx + (fx - cx) / config.blue_scale;
            let b_src_y = cy + (fy - cy) / config.blue_scale;

            // Signed shift magnitude (project onto the radial direction).
            let r_shift = (r_src_x - g_src_x).hypot(r_src_y - g_src_y)
                * if (fx - cx) * (r_src_x - g_src_x) + (fy - cy) * (r_src_y - g_src_y) >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
            let b_shift = (b_src_x - g_src_x).hypot(b_src_y - g_src_y)
                * if (fx - cx) * (b_src_x - g_src_x) + (fy - cy) * (b_src_y - g_src_y) >= 0.0 {
                    1.0
                } else {
                    -1.0
                };

            let idx = py * width + px;
            if let Some(slot) = red_shifts.get_mut(idx) {
                *slot = r_shift;
            }
            if let Some(slot) = blue_shifts.get_mut(idx) {
                *slot = b_shift;
            }
            let candidate = r_shift.abs().max(b_shift.abs());
            if candidate > max_abs {
                max_abs = candidate;
            }
        }
    }

    // Normalise [-max_abs, max_abs] → [0, 1].
    let range = if max_abs > 0.0 { max_abs } else { 1.0 };

    for i in 0..n {
        let r_norm =
            (red_shifts.get(i).copied().unwrap_or(0.0) / range * 0.5 + 0.5).clamp(0.0, 1.0);
        let b_norm =
            (blue_shifts.get(i).copied().unwrap_or(0.0) / range * 0.5 + 0.5).clamp(0.0, 1.0);
        if let Some(slot) = output.get_mut(i * 3) {
            *slot = r_norm;
        }
        if let Some(slot) = output.get_mut(i * 3 + 1) {
            *slot = 0.5; // neutral green
        }
        if let Some(slot) = output.get_mut(i * 3 + 2) {
            *slot = b_norm;
        }
    }

    output
}

// ─────────────────────────────────────────────────────────────────────────────
// Stats
// ─────────────────────────────────────────────────────────────────────────────

/// Compute aberration statistics for a given lateral CA configuration.
///
/// Scans every pixel, measures the red/blue shift from the green reference, and
/// returns summary statistics.
pub fn compute_chromatic_stats(
    width: usize,
    height: usize,
    config: &LateralChromaticConfig,
) -> ChromaticStats {
    if width == 0 || height == 0 {
        return ChromaticStats {
            max_red_shift_pixels: 0.0,
            max_blue_shift_pixels: 0.0,
            mean_shift_pixels: 0.0,
            strong_fringe_fraction: 0.0,
        };
    }

    let cx = (width as f32 - 1.0) * 0.5;
    let cy = (height as f32 - 1.0) * 0.5;

    let mut max_r = 0.0f32;
    let mut max_b = 0.0f32;
    let mut sum_shifts = 0.0f32;
    let mut strong_count = 0usize;
    let n = width * height;

    for py in 0..height {
        for px in 0..width {
            let fx = px as f32;
            let fy = py as f32;

            let r_src_x = cx + (fx - cx) / config.red_scale;
            let r_src_y = cy + (fy - cy) / config.red_scale;
            let g_src_x = cx + (fx - cx) / config.green_scale;
            let g_src_y = cy + (fy - cy) / config.green_scale;
            let b_src_x = cx + (fx - cx) / config.blue_scale;
            let b_src_y = cy + (fy - cy) / config.blue_scale;

            let r_shift = ((r_src_x - g_src_x).powi(2) + (r_src_y - g_src_y).powi(2)).sqrt();
            let b_shift = ((b_src_x - g_src_x).powi(2) + (b_src_y - g_src_y).powi(2)).sqrt();
            let total_shift = r_shift + b_shift;

            if r_shift > max_r {
                max_r = r_shift;
            }
            if b_shift > max_b {
                max_b = b_shift;
            }
            sum_shifts += total_shift;
            if total_shift > 2.0 {
                strong_count += 1;
            }
        }
    }

    let mean_shift_pixels = if n > 0 {
        sum_shifts / (2.0 * n as f32) // average across red + blue per pixel
    } else {
        0.0
    };
    let strong_fringe_fraction = if n > 0 {
        strong_count as f32 / n as f32
    } else {
        0.0
    };

    ChromaticStats {
        max_red_shift_pixels: max_r,
        max_blue_shift_pixels: max_b,
        mean_shift_pixels,
        strong_fringe_fraction,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lens Effects — RGBA u8 API (delegated to lens_effects module)
// ─────────────────────────────────────────────────────────────────────────────

pub use crate::lens_effects::{
    // Renamed to avoid conflict with the f32-RGB apply_barrel_distortion defined below.
    apply_barrel_distortion as apply_barrel_distortion_rgba,
    apply_chromatic_aberration_rgba,
    apply_lens_effects,
    apply_vignette_effect,
    compute_lens_effect_stats,
    // Renamed to avoid conflict with the f32-RGB ChromAberrationConfig defined below.
    ChromAberrationConfig as RgbaChromAberrationConfig,
    DistortionConfig,
    LensEffectError,
    LensEffectStats,
    LensEffectsConfig,
    LensVignetteConfig,
};

// ─────────────────────────────────────────────────────────────────────────────
// Post-processing API — f32 RGB (HxWx3 layout)
// (Types and functions live in crate::chrom_post_effect to keep this file under 2000 lines)
// ─────────────────────────────────────────────────────────────────────────────

pub use crate::chrom_post_effect::{
    apply_barrel_distortion, apply_chrom_aberration, apply_radial_chromatic_aberration,
    apply_vignetting_f32, bilinear_sample_channel, compute_radial_distances, compute_vignette_mask,
    estimate_aberration_strength, format_chrom_config, nearest_sample_channel, shift_channel,
    ChromAberrationConfig, ChromAberrationError, ChromInterpolation,
};

#[cfg(test)]
mod tests {
    use super::*;

    // ── Utility ──────────────────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    /// Flat RGB image filled with a constant value.
    fn uniform_image(w: usize, h: usize, val: f32) -> Vec<f32> {
        vec![val; w * h * 3]
    }

    /// Simple gradient image.
    fn gradient_image(w: usize, h: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; w * h * 3];
        for py in 0..h {
            for px in 0..w {
                let idx = (py * w + px) * 3;
                let frac_x = px as f32 / w.max(1) as f32;
                let frac_y = py as f32 / h.max(1) as f32;
                out[idx] = frac_x; // R
                out[idx + 1] = frac_y; // G
                out[idx + 2] = 1.0 - frac_x; // B
            }
        }
        out
    }

    // ── 1. radial_factor at r=0 → 1.0 ────────────────────────────────────────

    #[test]
    fn test_radial_factor_zero_radius() {
        let result = radial_factor(0.0, 0.5, 0.1);
        assert!(
            approx_eq(result, 1.0, 1e-6),
            "radial_factor(0, ...) should be 1.0, got {result}"
        );
    }

    // ── 2. radial_factor at r=1 with known k1 ─────────────────────────────────

    #[test]
    fn test_radial_factor_known_value() {
        // r=1, k1=0.2, k2=0.05 → 1 + 0.2 + 0.05 = 1.25
        let result = radial_factor(1.0, 0.2, 0.05);
        assert!(approx_eq(result, 1.25, 1e-6), "Expected 1.25, got {result}");
    }

    // ── 3. radial_factor: only k2 ─────────────────────────────────────────────

    #[test]
    fn test_radial_factor_only_k2() {
        // r=1, k1=0, k2=0.1 → 1 + 0 + 0.1 = 1.1
        let result = radial_factor(1.0, 0.0, 0.1);
        assert!(approx_eq(result, 1.1, 1e-6), "Expected 1.1, got {result}");
    }

    // ── 4. undistort_pixel at centre → unchanged ──────────────────────────────

    #[test]
    fn test_undistort_pixel_at_center() {
        let cx = 50.0;
        let cy = 37.5;
        let (rx, ry) = undistort_pixel(cx, cy, cx, cy, 0.1, 0.05);
        assert!(
            approx_eq(rx, cx, 1e-5),
            "Centre x should be unchanged: {rx} vs {cx}"
        );
        assert!(
            approx_eq(ry, cy, 1e-5),
            "Centre y should be unchanged: {ry} vs {cy}"
        );
    }

    // ── 5. undistort_pixel with zero distortion → identity ────────────────────

    #[test]
    fn test_undistort_pixel_zero_distortion() {
        let (rx, ry) = undistort_pixel(30.0, 20.0, 50.0, 50.0, 0.0, 0.0);
        assert!(approx_eq(rx, 30.0, 1e-5));
        assert!(approx_eq(ry, 20.0, 1e-5));
    }

    // ── 6. LateralChromaticConfig::validate all scales > 0 ───────────────────

    #[test]
    fn test_lateral_config_validate_positive_scales() {
        let cfg = LateralChromaticConfig::default();
        assert!(cfg.validate().is_ok(), "Default config should be valid");
    }

    // ── 7. LateralChromaticConfig::validate rejects zero scale ────────────────

    #[test]
    fn test_lateral_config_validate_zero_red_scale() {
        let cfg = LateralChromaticConfig {
            red_scale: 0.0,
            ..LateralChromaticConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    // ── 8. LateralChromaticConfig::validate rejects negative blue scale ───────

    #[test]
    fn test_lateral_config_validate_negative_blue_scale() {
        let cfg = LateralChromaticConfig {
            blue_scale: -1.0,
            ..LateralChromaticConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    // ── 9. LateralChromaticConfig::strong() vs subtle() ──────────────────────

    #[test]
    fn test_lateral_config_strong_vs_subtle() {
        let strong = LateralChromaticConfig::strong();
        let subtle = LateralChromaticConfig::subtle();
        assert!(
            strong.red_scale > subtle.red_scale,
            "Strong should have larger red scale"
        );
        assert!(
            strong.blue_scale < subtle.blue_scale,
            "Strong should have smaller blue scale"
        );
        assert!(strong.validate().is_ok());
        assert!(subtle.validate().is_ok());
    }

    // ── 10. apply_lateral_ca: empty image → error ─────────────────────────────

    #[test]
    fn test_apply_lateral_ca_empty_image() {
        let cfg = LateralChromaticConfig::default();
        let result = apply_lateral_ca(&[], 0, 0, &cfg);
        assert!(result.is_err());
        assert!(matches!(result, Err(ChromaticError::EmptyImage)));
    }

    // ── 11. apply_lateral_ca: wrong length → error ────────────────────────────

    #[test]
    fn test_apply_lateral_ca_wrong_length() {
        let cfg = LateralChromaticConfig::default();
        // Supply only 5 floats for a 2×2×3 = 12 image.
        let result = apply_lateral_ca(&[0.5; 5], 2, 2, &cfg);
        assert!(matches!(result, Err(ChromaticError::InvalidImage { .. })));
    }

    // ── 12. apply_lateral_ca: all scales 1.0 → output ≈ input ────────────────

    #[test]
    fn test_apply_lateral_ca_identity_scale() {
        let cfg = LateralChromaticConfig {
            red_scale: 1.0,
            green_scale: 1.0,
            blue_scale: 1.0,
            red_k1: 0.0,
            green_k1: 0.0,
            blue_k1: 0.0,
        };
        let image = gradient_image(8, 8);
        let out = apply_lateral_ca(&image, 8, 8, &cfg).expect("apply_lateral_ca failed");
        for (a, b) in image.iter().zip(out.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-4),
                "Identity scale should preserve values: {a} vs {b}"
            );
        }
    }

    // ── 13. apply_lateral_ca: output has same dimensions as input ─────────────

    #[test]
    fn test_apply_lateral_ca_output_dimensions() {
        let cfg = LateralChromaticConfig::default();
        let image = gradient_image(16, 12);
        let out = apply_lateral_ca(&image, 16, 12, &cfg).expect("apply_lateral_ca failed");
        assert_eq!(out.len(), 16 * 12 * 3);
    }

    // ── 14. apply_lateral_ca: centre pixel unchanged ──────────────────────────

    #[test]
    fn test_apply_lateral_ca_center_pixel_unchanged() {
        // For any scale, cx + (cx - cx)/scale = cx, so centre maps to centre.
        let cfg = LateralChromaticConfig::default();
        let w = 9usize;
        let h = 9usize;
        let image = gradient_image(w, h);
        let out = apply_lateral_ca(&image, w, h, &cfg).expect("apply_lateral_ca failed");
        // Centre pixel index.
        let cx = w / 2;
        let cy = h / 2;
        let pixel_idx = (cy * w + cx) * 3;
        for ch in 0..3 {
            let orig = image[pixel_idx + ch];
            let result = out[pixel_idx + ch];
            assert!(
                approx_eq(orig, result, 1e-4),
                "Centre pixel channel {ch} should be unchanged: {orig} vs {result}"
            );
        }
    }

    // ── 15. apply_lateral_ca: non-uniform scale → output differs from input ───

    #[test]
    fn test_apply_lateral_ca_non_identity_differs() {
        let cfg = LateralChromaticConfig::default(); // red_scale ≠ blue_scale
        let image = gradient_image(16, 16);
        let out = apply_lateral_ca(&image, 16, 16, &cfg).expect("apply_lateral_ca failed");
        let n_different = image
            .iter()
            .zip(out.iter())
            .filter(|&(a, b)| (a - b).abs() > 1e-5)
            .count();
        assert!(
            n_different > 0,
            "Non-uniform scale should change at least some pixels"
        );
    }

    // ── 16. apply_longitudinal_ca: uniform image → unchanged ─────────────────

    #[test]
    fn test_apply_longitudinal_ca_uniform_image() {
        let cfg = LongitudinalChromaticConfig {
            blue_blur: 1.0,
            ..Default::default()
        };
        let image = uniform_image(8, 8, 0.7);
        let out = apply_longitudinal_ca(&image, 8, 8, &cfg).expect("apply_longitudinal_ca failed");
        for (a, b) in image.iter().zip(out.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-4),
                "Uniform image blurred should be unchanged: {a} vs {b}"
            );
        }
    }

    // ── 17. apply_longitudinal_ca: all blur=0 → same as input ────────────────

    #[test]
    fn test_apply_longitudinal_ca_zero_blur_identity() {
        let cfg = LongitudinalChromaticConfig {
            red_blur: 0.0,
            green_blur: 0.0,
            blue_blur: 0.0,
            fringe_strength: 0.5,
        };
        let image = gradient_image(10, 10);
        let out =
            apply_longitudinal_ca(&image, 10, 10, &cfg).expect("apply_longitudinal_ca failed");
        for (a, b) in image.iter().zip(out.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-5),
                "Zero blur should be identity: {a} vs {b}"
            );
        }
    }

    // ── 18. apply_longitudinal_ca: preserves image dimensions ────────────────

    #[test]
    fn test_apply_longitudinal_ca_dimensions() {
        let cfg = LongitudinalChromaticConfig::default();
        let image = gradient_image(20, 15);
        let out =
            apply_longitudinal_ca(&image, 20, 15, &cfg).expect("apply_longitudinal_ca failed");
        assert_eq!(out.len(), 20 * 15 * 3);
    }

    // ── 19. LongitudinalChromaticConfig::validate: negative blur → error ──────

    #[test]
    fn test_longitudinal_config_validate_negative_blur() {
        let cfg = LongitudinalChromaticConfig {
            red_blur: -0.1,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
        let cfg2 = LongitudinalChromaticConfig {
            green_blur: -0.5,
            ..Default::default()
        };
        assert!(cfg2.validate().is_err());
        let cfg3 = LongitudinalChromaticConfig {
            blue_blur: -1.0,
            ..Default::default()
        };
        assert!(cfg3.validate().is_err());
    }

    // ── 20. apply_chromatic_aberration: basic smoke test ─────────────────────

    #[test]
    fn test_apply_chromatic_aberration_smoke() {
        let lateral = LateralChromaticConfig::default();
        let longitudinal = LongitudinalChromaticConfig::default();
        let image = gradient_image(12, 12);
        let out = apply_chromatic_aberration(&image, 12, 12, &lateral, &longitudinal)
            .expect("apply_chromatic_aberration failed");
        assert_eq!(out.len(), 12 * 12 * 3);
    }

    // ── 21. aberration_map: centre → near 0 ──────────────────────────────────

    #[test]
    fn test_aberration_map_center_near_zero() {
        let cfg = LateralChromaticConfig::default();
        let w = 11usize;
        let h = 11usize;
        let map = aberration_map(w, h, &cfg);
        assert_eq!(map.len(), w * h);
        let cx = w / 2;
        let cy = h / 2;
        let centre_val = map[cy * w + cx];
        // Map is normalised; centre should be 0.
        assert!(
            approx_eq(centre_val, 0.0, 1e-5),
            "Aberration map centre should be ~0.0, got {centre_val}"
        );
    }

    // ── 22. aberration_map: corners larger than centre ────────────────────────

    #[test]
    fn test_aberration_map_corners_larger() {
        let cfg = LateralChromaticConfig::default();
        let w = 10usize;
        let h = 10usize;
        let map = aberration_map(w, h, &cfg);
        let centre = map[4 * w + 4]; // rough centre
        let corner = map[0]; // top-left corner
        assert!(
            corner > centre,
            "Corner aberration ({corner}) should exceed centre ({centre})"
        );
    }

    // ── 23. aberration_map: dimensions correct ────────────────────────────────

    #[test]
    fn test_aberration_map_dimensions() {
        let cfg = LateralChromaticConfig::default();
        let map = aberration_map(16, 9, &cfg);
        assert_eq!(map.len(), 16 * 9);
    }

    // ── 24. fringe_visualization: dimensions correct ──────────────────────────

    #[test]
    fn test_fringe_visualization_dimensions() {
        let cfg = LateralChromaticConfig::default();
        let vis = fringe_visualization(8, 6, &cfg);
        assert_eq!(vis.len(), 8 * 6 * 3);
    }

    // ── 25. fringe_visualization: all values in [0, 1] ───────────────────────

    #[test]
    fn test_fringe_visualization_values_in_range() {
        let cfg = LateralChromaticConfig::default();
        let vis = fringe_visualization(10, 8, &cfg);
        for &v in &vis {
            assert!(
                (0.0..=1.0).contains(&v),
                "fringe_visualization value out of [0,1]: {v}"
            );
        }
    }

    // ── 26. compute_chromatic_stats: max_red_shift > 0 for non-unity scale ───

    #[test]
    fn test_chromatic_stats_non_unity_scale() {
        let cfg = LateralChromaticConfig::default();
        let stats = compute_chromatic_stats(32, 32, &cfg);
        assert!(
            stats.max_red_shift_pixels > 0.0,
            "Expected positive red shift for non-unity scale, got {}",
            stats.max_red_shift_pixels
        );
        assert!(
            stats.max_blue_shift_pixels > 0.0,
            "Expected positive blue shift for non-unity scale, got {}",
            stats.max_blue_shift_pixels
        );
    }

    // ── 27. compute_chromatic_stats: zero shift for all equal scales ──────────

    #[test]
    fn test_chromatic_stats_equal_scales_zero_shift() {
        let cfg = LateralChromaticConfig {
            red_scale: 1.0,
            green_scale: 1.0,
            blue_scale: 1.0,
            ..Default::default()
        };
        let stats = compute_chromatic_stats(16, 16, &cfg);
        assert!(approx_eq(stats.max_red_shift_pixels, 0.0, 1e-6));
        assert!(approx_eq(stats.max_blue_shift_pixels, 0.0, 1e-6));
    }

    // ── 28. gaussian_blur_channel (via longitudinal CA): small blur smooths ───

    #[test]
    fn test_longitudinal_ca_small_blur_smooths() {
        // A sharp gradient image should become smoother after longitudinal CA.
        let w = 20usize;
        let h = 20usize;
        let cfg = LongitudinalChromaticConfig {
            red_blur: 2.0,
            green_blur: 0.0,
            blue_blur: 0.0,
            fringe_strength: 1.0, // fully blurred
        };
        let mut image = vec![0.0f32; w * h * 3];
        // Set R channel to a hard step at column 10.
        for py in 0..h {
            for px in 0..w {
                let idx = (py * w + px) * 3;
                image[idx] = if px >= 10 { 1.0 } else { 0.0 };
            }
        }
        let out = apply_longitudinal_ca(&image, w, h, &cfg).expect("longitudinal CA failed");
        // The blurred R channel should have values strictly between 0 and 1 near the edge.
        // Check a pixel just left of and just right of the step.
        let left_idx = (h / 2 * w + 9) * 3; // px=9, R channel
        let right_idx = (h / 2 * w + 10) * 3; // px=10, R channel
        let left_val = out[left_idx];
        let right_val = out[right_idx];
        // After blurring, neither should be exactly 0 or 1.
        assert!(
            left_val > 0.0,
            "Left of step should be > 0 after blur, got {left_val}"
        );
        assert!(
            right_val < 1.0,
            "Right of step should be < 1 after blur, got {right_val}"
        );
    }

    // ── 29. apply_lateral_ca: output values in reasonable range ──────────────

    #[test]
    fn test_apply_lateral_ca_output_range() {
        let cfg = LateralChromaticConfig::strong();
        let image = gradient_image(16, 16);
        let out = apply_lateral_ca(&image, 16, 16, &cfg).expect("lateral CA failed");
        for &v in &out {
            assert!(
                (0.0..=1.0).contains(&v),
                "Lateral CA output out of [0,1]: {v}"
            );
        }
    }

    // ── 30. aberration_map: zero-dimension guard ──────────────────────────────

    #[test]
    fn test_aberration_map_zero_dimensions() {
        let cfg = LateralChromaticConfig::default();
        let map = aberration_map(0, 0, &cfg);
        assert!(map.is_empty());
    }

    // Tests 31-60 live in chrom_post_effect::tests (see src/chrom_post_effect.rs)
}
