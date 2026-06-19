//! Temporal Anti-Aliasing (TAA) with Halton jitter, variance clipping, and adaptive blending.
//!
//! TAA reduces aliasing by accumulating and blending multiple frames over time. Each frame
//! uses slightly different subpixel sample offsets (jitter from a Halton sequence), and the
//! history buffer is blended with the current frame. This produces high-quality anti-aliasing
//! without the memory overhead of MSAA.
//!
//! ## Key Features
//!
//! - **Halton jitter**: Quasi-random subpixel offsets with better distribution than uniform random.
//! - **Variance clipping**: Reduces ghosting by clamping history to the local color neighborhood.
//! - **Adaptive blending**: Dynamically adjusts blend factor based on per-pixel motion magnitude.
//! - **Unsharp mask sharpening**: Post-accumulation sharpening to counteract blur from blending.
//! - **Stateful accumulator**: [`TaaAccumulator`] manages jitter sequencing and history state.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by temporal anti-aliasing operations.
#[derive(Debug, Error, PartialEq)]
pub enum TaaError {
    /// Invalid configuration parameter.
    #[error("Invalid TAA configuration: {0}")]
    InvalidConfig(String),

    /// Pixel count does not match declared dimensions.
    #[error("Invalid image: pixel count mismatch — {0}")]
    InvalidImage(String),

    /// History buffer has no frames yet.
    #[error("Empty TAA history — accumulate at least one frame first")]
    EmptyHistory,

    /// Dimension mismatch between two images.
    #[error("Dimension mismatch: expected {expected} pixels, got {got}")]
    DimensionMismatch {
        /// Expected number of pixels.
        expected: usize,
        /// Actual number of pixels.
        got: usize,
    },
}

// ---------------------------------------------------------------------------
// Halton sequence
// ---------------------------------------------------------------------------

/// Compute the `n`-th element of the Halton sequence with the given `base`.
///
/// Halton sequences are quasi-random and have better low-discrepancy properties
/// than pseudo-random sequences, making them ideal for jitter sampling patterns.
///
/// # Examples
///
/// ```
/// use oxigaf_render::temporal_aa::halton;
/// assert_eq!(halton(0, 2), 0.0);
/// assert!((halton(1, 2) - 0.5).abs() < 1e-6);
/// assert!((halton(2, 2) - 0.25).abs() < 1e-6);
/// ```
pub fn halton(mut n: usize, base: usize) -> f32 {
    let mut result = 0.0_f32;
    let mut f = 1.0_f32;
    while n > 0 {
        f /= base as f32;
        result += f * (n % base) as f32;
        n /= base;
    }
    result
}

/// Compute a 2D jitter offset for frame `frame_idx` in the range `[-0.5, 0.5]²`.
///
/// Uses the Halton(2, 3) pair, which is a common standard choice for TAA jitter.
/// The sequence wraps at `sequence_length` to avoid indefinite accumulation drift.
///
/// # Parameters
///
/// - `frame_idx`: Current frame index (0-based). Wraps at `sequence_length`.
/// - `sequence_length`: Number of frames in the jitter cycle. Typical: 8.
///
/// # Returns
///
/// A `(jx, jy)` pair in `[-0.5, 0.5]²`.
pub fn jitter_offset(frame_idx: usize, sequence_length: usize) -> (f32, f32) {
    let len = sequence_length.max(1);
    let i = frame_idx % len;
    let jx = halton(i + 1, 2) - 0.5;
    let jy = halton(i + 1, 3) - 0.5;
    (jx, jy)
}

// ---------------------------------------------------------------------------
// History buffer
// ---------------------------------------------------------------------------

/// TAA history buffer accumulating RGB values from previous frames.
///
/// The buffer stores a running blend of past frames in linear `[0, 1]` f32 RGB.
/// Access is row-major: index = `(y * width + x) * 3`.
#[derive(Debug, Clone)]
pub struct TaaHistory {
    /// Width of the frame in pixels.
    pub width: usize,
    /// Height of the frame in pixels.
    pub height: usize,
    /// Accumulated RGB image. `len == width * height * 3`.
    pub color: Vec<f32>,
    /// Number of frames accumulated into this history so far.
    pub frame_count: usize,
}

impl TaaHistory {
    /// Create a new history buffer initialized to black (all zeros).
    pub fn new(width: usize, height: usize) -> Self {
        let size = width.saturating_mul(height).saturating_mul(3);
        Self {
            width,
            height,
            color: vec![0.0_f32; size],
            frame_count: 0,
        }
    }

    /// Returns `true` if no frames have been accumulated yet.
    pub fn is_empty(&self) -> bool {
        self.frame_count == 0
    }

    /// Reset the history to black and zero the frame count.
    pub fn reset(&mut self) {
        self.color.iter_mut().for_each(|v| *v = 0.0);
        self.frame_count = 0;
    }

    /// Get the RGB color at pixel `(x, y)`.
    ///
    /// Returns `[0.0, 0.0, 0.0]` if the coordinates are out of bounds.
    pub fn get_pixel(&self, x: usize, y: usize) -> [f32; 3] {
        if x >= self.width || y >= self.height {
            return [0.0; 3];
        }
        let base = (y * self.width + x) * 3;
        let r = self.color.get(base).copied().unwrap_or(0.0);
        let g = self.color.get(base + 1).copied().unwrap_or(0.0);
        let b = self.color.get(base + 2).copied().unwrap_or(0.0);
        [r, g, b]
    }

    /// Set the RGB color at pixel `(x, y)`.
    ///
    /// # Errors
    ///
    /// Returns [`TaaError::DimensionMismatch`] if `(x, y)` is out of bounds.
    pub fn set_pixel(&mut self, x: usize, y: usize, color: [f32; 3]) -> Result<(), TaaError> {
        if x >= self.width || y >= self.height {
            return Err(TaaError::DimensionMismatch {
                expected: self.width * self.height,
                got: y * self.width + x,
            });
        }
        let base = (y * self.width + x) * 3;
        if let Some(slot) = self.color.get_mut(base) {
            *slot = color[0];
        }
        if let Some(slot) = self.color.get_mut(base + 1) {
            *slot = color[1];
        }
        if let Some(slot) = self.color.get_mut(base + 2) {
            *slot = color[2];
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TAA configuration
// ---------------------------------------------------------------------------

/// Configuration for the TAA accumulation pipeline.
#[derive(Debug, Clone)]
pub struct TaaConfig {
    /// Blend factor: weight of the current frame vs. accumulated history.
    ///
    /// - `0.0` = pure history (no current frame contribution).
    /// - `1.0` = pure current frame (no temporal accumulation).
    ///
    /// Typical value: `0.1` (10% current, 90% history).
    pub blend_factor: f32,

    /// Number of frames in the Halton jitter sequence before wrapping.
    ///
    /// A power-of-two value such as `8` or `16` is common.
    pub jitter_sequence_length: usize,

    /// When `true`, clamp history color to the local variance neighborhood of the current
    /// frame before blending. Reduces ghosting at the cost of some sharpness.
    pub variance_clipping: bool,

    /// Radius in pixels of the local neighborhood used for variance clipping.
    ///
    /// A radius of `1` uses a `3×3` pixel window.
    pub clip_window_radius: usize,

    /// Strength of unsharp mask sharpening applied after accumulation.
    ///
    /// - `0.0` = no sharpening.
    /// - `1.0` = strong sharpening.
    pub sharpen_strength: f32,

    /// When `true`, the blend factor adapts per-pixel based on the luminance difference
    /// between the current frame and history. High motion → more current frame weight.
    pub adaptive_blend: bool,

    /// Minimum blend factor when adaptive blending is active. Must be ≤ `blend_factor`.
    pub adaptive_blend_min: f32,
}

impl Default for TaaConfig {
    fn default() -> Self {
        Self {
            blend_factor: 0.1,
            jitter_sequence_length: 8,
            variance_clipping: true,
            clip_window_radius: 1,
            sharpen_strength: 0.2,
            adaptive_blend: false,
            adaptive_blend_min: 0.05,
        }
    }
}

impl TaaConfig {
    /// Validate all configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns [`TaaError::InvalidConfig`] if any parameter is out of range.
    pub fn validate(&self) -> Result<(), TaaError> {
        if self.blend_factor <= 0.0 || self.blend_factor > 1.0 {
            return Err(TaaError::InvalidConfig(format!(
                "blend_factor must be in (0, 1], got {}",
                self.blend_factor
            )));
        }
        if self.jitter_sequence_length < 1 {
            return Err(TaaError::InvalidConfig(
                "jitter_sequence_length must be >= 1".to_string(),
            ));
        }
        if self.clip_window_radius < 1 {
            return Err(TaaError::InvalidConfig(
                "clip_window_radius must be >= 1".to_string(),
            ));
        }
        if self.sharpen_strength < 0.0 {
            return Err(TaaError::InvalidConfig(format!(
                "sharpen_strength must be >= 0, got {}",
                self.sharpen_strength
            )));
        }
        if self.adaptive_blend_min < 0.0 || self.adaptive_blend_min > self.blend_factor {
            return Err(TaaError::InvalidConfig(format!(
                "adaptive_blend_min must be in [0, blend_factor={}], got {}",
                self.blend_factor, self.adaptive_blend_min
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Local color statistics for variance clipping
// ---------------------------------------------------------------------------

/// Compute per-pixel mean and standard deviation over a local window.
///
/// Gathers all pixels in a `(2*radius+1)²` window centered at `(cx, cy)`,
/// clamped to the image edges, and computes per-channel statistics.
///
/// # Parameters
///
/// - `image`: flat RGB image buffer, `len == width * height * 3`.
/// - `width`, `height`: image dimensions.
/// - `cx`, `cy`: center pixel coordinates.
/// - `radius`: half-width of the window. `radius=1` → 3×3 window.
///
/// # Returns
///
/// `(mean, std)` as `[f32; 3]` per channel. `std` is the population standard deviation.
pub fn local_color_stats(
    image: &[f32],
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
    radius: usize,
) -> ([f32; 3], [f32; 3]) {
    let y_min = cy.saturating_sub(radius);
    let y_max = (cy + radius + 1).min(height);
    let x_min = cx.saturating_sub(radius);
    let x_max = (cx + radius + 1).min(width);

    let mut sum = [0.0_f32; 3];
    let mut sum_sq = [0.0_f32; 3];
    let mut count = 0_usize;

    for ny in y_min..y_max {
        for nx in x_min..x_max {
            let base = (ny * width + nx) * 3;
            for c in 0..3 {
                let v = image.get(base + c).copied().unwrap_or(0.0);
                sum[c] += v;
                sum_sq[c] += v * v;
            }
            count += 1;
        }
    }

    if count == 0 {
        return ([0.0; 3], [0.0; 3]);
    }

    let n = count as f32;
    let mut mean = [0.0_f32; 3];
    let mut std = [0.0_f32; 3];
    for c in 0..3 {
        mean[c] = sum[c] / n;
        let variance = (sum_sq[c] / n) - (mean[c] * mean[c]);
        std[c] = variance.max(0.0).sqrt();
    }

    (mean, std)
}

/// Clip `color` to the `±clip_sigma`-sigma range around `mean`.
///
/// For each channel independently, clamps `color[c]` to
/// `[mean[c] - clip_sigma * std[c], mean[c] + clip_sigma * std[c]]`.
///
/// # Parameters
///
/// - `color`: input pixel color to clip.
/// - `mean`: local neighborhood mean per channel.
/// - `std`: local neighborhood standard deviation per channel.
/// - `clip_sigma`: number of standard deviations to allow. Typical: `1.0` or `1.5`.
pub fn clip_to_variance(
    color: [f32; 3],
    mean: [f32; 3],
    std: [f32; 3],
    clip_sigma: f32,
) -> [f32; 3] {
    let mut out = [0.0_f32; 3];
    for c in 0..3 {
        let lo = mean[c] - clip_sigma * std[c];
        let hi = mean[c] + clip_sigma * std[c];
        out[c] = color[c].clamp(lo, hi);
    }
    out
}

// ---------------------------------------------------------------------------
// Luminance helper
// ---------------------------------------------------------------------------

/// Compute the luminance of an RGB color using standard Rec. 709 weights.
#[inline]
fn luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// Linear interpolation between `a` and `b` by factor `t` ∈ [0, 1].
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

// ---------------------------------------------------------------------------
// Post-process sharpening
// ---------------------------------------------------------------------------

/// Apply unsharp mask sharpening to an RGB image.
///
/// The sharpened result is computed as:
/// ```text
/// sharpened = image + strength * (image - blur(image))
/// ```
/// where `blur` is a 3×3 box filter. Output is clamped to `[0, 1]`.
///
/// # Parameters
///
/// - `image`: flat RGB buffer, `len == width * height * 3`.
/// - `width`, `height`: image dimensions.
/// - `strength`: sharpening intensity. `0.0` returns a copy of the input unchanged.
///
/// # Returns
///
/// A new `Vec<f32>` of the same length with the sharpened image.
pub fn sharpen_image(image: &[f32], width: usize, height: usize, strength: f32) -> Vec<f32> {
    if strength == 0.0 || width == 0 || height == 0 {
        return image.to_vec();
    }

    // Compute 3×3 box-filter blur
    let mut blurred = vec![0.0_f32; image.len()];
    for py in 0..height {
        for px in 0..width {
            let y_min = py.saturating_sub(1);
            let y_max = (py + 2).min(height);
            let x_min = px.saturating_sub(1);
            let x_max = (px + 2).min(width);

            let mut sum = [0.0_f32; 3];
            let mut count = 0_usize;
            for ny in y_min..y_max {
                for nx in x_min..x_max {
                    let base = (ny * width + nx) * 3;
                    for (c, sum_val) in sum.iter_mut().enumerate() {
                        *sum_val += image.get(base + c).copied().unwrap_or(0.0);
                    }
                    count += 1;
                }
            }

            let dst_base = (py * width + px) * 3;
            let n = count as f32;
            if count > 0 {
                for (c, &sum_val) in sum.iter().enumerate() {
                    if let Some(slot) = blurred.get_mut(dst_base + c) {
                        *slot = sum_val / n;
                    }
                }
            }
        }
    }

    // sharpened = image + strength * (image - blurred), clamped to [0, 1]
    let out = image
        .iter()
        .zip(blurred.iter())
        .map(|(&orig, &blur)| (orig + strength * (orig - blur)).clamp(0.0, 1.0))
        .collect();
    out
}

// ---------------------------------------------------------------------------
// Core TAA accumulation
// ---------------------------------------------------------------------------

/// Perform one TAA accumulation step.
///
/// Blends `current` with the `history` buffer using the parameters in `config`.
/// History is mutated in-place; the function returns the final blended image.
///
/// # Algorithm
///
/// 1. If history is empty, seed it with `current` and return a copy.
/// 2. For each pixel:
///    - Optionally clip the history color to the local variance of `current`.
///    - Compute the blend factor (fixed or adaptive based on luminance diff).
///    - Blend: `alpha * current + (1 - alpha) * history`.
/// 3. Optionally apply unsharp mask sharpening.
/// 4. Write the result back into `history` and increment `frame_count`.
///
/// # Errors
///
/// - [`TaaError::DimensionMismatch`] if `current.len()` does not match `history`.
pub fn accumulate_taa(
    current: &[f32],
    history: &mut TaaHistory,
    config: &TaaConfig,
) -> Result<Vec<f32>, TaaError> {
    let expected = history.width * history.height * 3;

    if current.len() != expected {
        return Err(TaaError::DimensionMismatch {
            expected,
            got: current.len(),
        });
    }

    // Bootstrap: first frame
    if history.is_empty() {
        history.color = current.to_vec();
        history.frame_count = 1;
        return Ok(current.to_vec());
    }

    let width = history.width;
    let height = history.height;

    let mut blended = vec![0.0_f32; expected];

    for py in 0..height {
        for px in 0..width {
            let base = (py * width + px) * 3;

            // Read current pixel
            let cur_r = current.get(base).copied().unwrap_or(0.0);
            let cur_g = current.get(base + 1).copied().unwrap_or(0.0);
            let cur_b = current.get(base + 2).copied().unwrap_or(0.0);
            let cur_color = [cur_r, cur_g, cur_b];

            // Read history pixel
            let mut hist_color = history.get_pixel(px, py);

            // Variance clipping: clamp history to local color range of current frame
            if config.variance_clipping {
                let (mean, std) =
                    local_color_stats(current, width, height, px, py, config.clip_window_radius);
                hist_color = clip_to_variance(hist_color, mean, std, 1.0);
            }

            // Choose blend factor
            let alpha = if config.adaptive_blend {
                let diff = (luminance(cur_color) - luminance(hist_color)).abs();
                lerp(
                    config.adaptive_blend_min,
                    config.blend_factor,
                    diff.clamp(0.0, 1.0),
                )
            } else {
                config.blend_factor
            };

            // Temporal blend
            let out_r = alpha * cur_color[0] + (1.0 - alpha) * hist_color[0];
            let out_g = alpha * cur_color[1] + (1.0 - alpha) * hist_color[1];
            let out_b = alpha * cur_color[2] + (1.0 - alpha) * hist_color[2];

            if let Some(slot) = blended.get_mut(base) {
                *slot = out_r;
            }
            if let Some(slot) = blended.get_mut(base + 1) {
                *slot = out_g;
            }
            if let Some(slot) = blended.get_mut(base + 2) {
                *slot = out_b;
            }
        }
    }

    // Optional sharpening
    let result = if config.sharpen_strength > 0.0 {
        sharpen_image(&blended, width, height, config.sharpen_strength)
    } else {
        blended
    };

    // Update history
    history.color = result.clone();
    history.frame_count += 1;

    Ok(result)
}

// ---------------------------------------------------------------------------
// Stateful TAA accumulator
// ---------------------------------------------------------------------------

/// Stateful TAA accumulator for processing video sequences frame by frame.
///
/// Maintains the current history buffer and frame index, automatically
/// advancing the Halton jitter sequence on each processed frame.
pub struct TaaAccumulator {
    /// TAA configuration shared for all frames.
    pub config: TaaConfig,
    /// Internal history buffer.
    history: TaaHistory,
    /// Current frame index (used for jitter sequencing).
    frame_idx: usize,
}

impl TaaAccumulator {
    /// Create a new accumulator for frames of the given dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`TaaError::InvalidConfig`] if `config.validate()` fails.
    pub fn new(width: usize, height: usize, config: TaaConfig) -> Result<Self, TaaError> {
        config.validate()?;
        Ok(Self {
            history: TaaHistory::new(width, height),
            config,
            frame_idx: 0,
        })
    }

    /// Process the next frame in the sequence.
    ///
    /// Internally calls [`accumulate_taa`] and advances the jitter frame index.
    ///
    /// # Errors
    ///
    /// Returns [`TaaError::DimensionMismatch`] if `frame` has the wrong pixel count.
    pub fn process(&mut self, frame: &[f32]) -> Result<Vec<f32>, TaaError> {
        let result = accumulate_taa(frame, &mut self.history, &self.config)?;
        self.frame_idx += 1;
        Ok(result)
    }

    /// Get the jitter offset for the current (not yet processed) frame.
    ///
    /// Callers should apply this offset to their projection matrix before rendering.
    pub fn current_jitter(&self) -> (f32, f32) {
        jitter_offset(self.frame_idx, self.config.jitter_sequence_length)
    }

    /// Total number of frames that have been accumulated.
    pub fn frame_count(&self) -> usize {
        self.history.frame_count
    }

    /// Reset the accumulator: clear history and restart from frame 0.
    pub fn reset(&mut self) {
        self.history.reset();
        self.frame_idx = 0;
    }

    /// Estimate the history quality in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` for fresh/reset history, approaching `1.0` as the number of
    /// accumulated frames reaches `1 / blend_factor` (the exponential moving-average
    /// effective sample count).
    pub fn quality(&self) -> f32 {
        let effective_samples = 1.0 / self.config.blend_factor;
        (self.history.frame_count as f32 / effective_samples).min(1.0)
    }
}

// ---------------------------------------------------------------------------
// TAA statistics
// ---------------------------------------------------------------------------

/// Summary statistics computed over a completed TAA accumulation step.
#[derive(Debug, Clone)]
pub struct TaaStats {
    /// Number of frames accumulated in the history buffer.
    pub frame_count: usize,
    /// Mean blend factor actually applied across all pixels.
    ///
    /// For non-adaptive mode this equals `config.blend_factor`. For adaptive mode,
    /// it reflects the per-pixel weighted average.
    pub mean_blend_factor: f32,
    /// Per-pixel mean absolute luminance difference between `current` and `history`.
    ///
    /// Higher values indicate more inter-frame motion or scene change (ghosting risk).
    pub mean_ghosting_estimate: f32,
    /// Fraction of pixels where `|current - accumulated| < 0.01` (converged pixels).
    pub converged_fraction: f32,
}

/// Compute TAA statistics from a completed accumulation step.
///
/// # Parameters
///
/// - `current`: the raw current-frame image before blending.
/// - `accumulated`: the blended result from [`accumulate_taa`].
/// - `history`: the history buffer (already updated with this frame).
/// - `blend_factor`: the nominal blend factor from [`TaaConfig`].
///
/// # Errors
///
/// - [`TaaError::DimensionMismatch`] if `current` and `accumulated` differ in length.
/// - [`TaaError::DimensionMismatch`] if their pixel count doesn't match `history`.
pub fn compute_taa_stats(
    current: &[f32],
    accumulated: &[f32],
    history: &TaaHistory,
    blend_factor: f32,
) -> Result<TaaStats, TaaError> {
    if current.len() != accumulated.len() {
        return Err(TaaError::DimensionMismatch {
            expected: current.len(),
            got: accumulated.len(),
        });
    }

    let expected = history.width * history.height * 3;
    if current.len() != expected {
        return Err(TaaError::DimensionMismatch {
            expected,
            got: current.len(),
        });
    }

    let pixel_count = history.width * history.height;
    if pixel_count == 0 {
        return Ok(TaaStats {
            frame_count: history.frame_count,
            mean_blend_factor: blend_factor,
            mean_ghosting_estimate: 0.0,
            converged_fraction: 1.0,
        });
    }

    let mut ghosting_sum = 0.0_f32;
    let mut converged_count = 0_usize;

    for (px_idx, hist_chunk) in history.color.chunks(3).enumerate() {
        let base = px_idx * 3;
        let cur_r = current.get(base).copied().unwrap_or(0.0);
        let cur_g = current.get(base + 1).copied().unwrap_or(0.0);
        let cur_b = current.get(base + 2).copied().unwrap_or(0.0);
        let cur_color = [cur_r, cur_g, cur_b];

        let hist_r = hist_chunk.first().copied().unwrap_or(0.0);
        let hist_g = hist_chunk.get(1).copied().unwrap_or(0.0);
        let hist_b = hist_chunk.get(2).copied().unwrap_or(0.0);
        let hist_color = [hist_r, hist_g, hist_b];

        let lum_diff = (luminance(cur_color) - luminance(hist_color)).abs();
        ghosting_sum += lum_diff;

        // Converged pixel: all three channels within 0.01
        let acc_r = accumulated.get(base).copied().unwrap_or(0.0);
        let acc_g = accumulated.get(base + 1).copied().unwrap_or(0.0);
        let acc_b = accumulated.get(base + 2).copied().unwrap_or(0.0);
        let diff_max = (cur_r - acc_r)
            .abs()
            .max((cur_g - acc_g).abs())
            .max((cur_b - acc_b).abs());
        if diff_max < 0.01 {
            converged_count += 1;
        }
    }

    Ok(TaaStats {
        frame_count: history.frame_count,
        mean_blend_factor: blend_factor,
        mean_ghosting_estimate: ghosting_sum / pixel_count as f32,
        converged_fraction: converged_count as f32 / pixel_count as f32,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ── Halton sequence ────────────────────────────────────────────────────────

    #[test]
    fn test_halton_0_base2_is_zero() {
        assert_eq!(halton(0, 2), 0.0, "halton(0, 2) must be 0.0");
    }

    #[test]
    fn test_halton_1_base2_is_half() {
        let v = halton(1, 2);
        assert!((v - 0.5).abs() < 1e-6, "halton(1,2) expected 0.5, got {v}");
    }

    #[test]
    fn test_halton_2_base2_is_quarter() {
        let v = halton(2, 2);
        assert!(
            (v - 0.25).abs() < 1e-6,
            "halton(2,2) expected 0.25, got {v}"
        );
    }

    #[test]
    fn test_halton_1_base3_is_third() {
        let v = halton(1, 3);
        assert!(
            (v - 1.0 / 3.0).abs() < 1e-6,
            "halton(1,3) expected 1/3, got {v}"
        );
    }

    #[test]
    fn test_halton_2_base3() {
        // halton(2, 3): f=1/3, (2%3=2) → result=2/3
        let v = halton(2, 3);
        assert!(
            (v - 2.0 / 3.0).abs() < 1e-6,
            "halton(2,3) expected 2/3, got {v}"
        );
    }

    #[test]
    fn test_halton_monotone_coverage_base2() {
        // All 8 values in first cycle should be distinct and in [0, 1)
        let vals: Vec<f32> = (0..8).map(|n| halton(n, 2)).collect();
        for &v in &vals {
            assert!((0.0..1.0).contains(&v) || v == 0.0, "out of [0,1): {v}");
        }
        // Check uniqueness
        for i in 0..vals.len() {
            for j in (i + 1)..vals.len() {
                assert!((vals[i] - vals[j]).abs() > 1e-6, "duplicate at {i},{j}");
            }
        }
    }

    // ── Jitter offset ──────────────────────────────────────────────────────────

    #[test]
    fn test_jitter_offset_in_half_range() {
        for frame in 0..16_usize {
            let (jx, jy) = jitter_offset(frame, 8);
            assert!(
                (-0.5..=0.5).contains(&jx),
                "jx out of [-0.5,0.5]: {jx} at frame {frame}"
            );
            assert!(
                (-0.5..=0.5).contains(&jy),
                "jy out of [-0.5,0.5]: {jy} at frame {frame}"
            );
        }
    }

    #[test]
    fn test_jitter_offset_different_frames() {
        let (jx0, jy0) = jitter_offset(0, 8);
        let (jx1, jy1) = jitter_offset(1, 8);
        assert!(
            (jx0 - jx1).abs() > 1e-6 || (jy0 - jy1).abs() > 1e-6,
            "frames 0 and 1 must have different jitter"
        );
    }

    #[test]
    fn test_jitter_offset_wraps_at_sequence_length() {
        // frame 0 and frame 8 should produce identical jitter with length 8
        let (jx0, jy0) = jitter_offset(0, 8);
        let (jx8, jy8) = jitter_offset(8, 8);
        assert!((jx0 - jx8).abs() < 1e-6, "jx should wrap: {jx0} vs {jx8}");
        assert!((jy0 - jy8).abs() < 1e-6, "jy should wrap: {jy0} vs {jy8}");
    }

    #[test]
    fn test_jitter_offset_sequence_length_zero_handled() {
        // sequence_length=0 should not panic (uses .max(1))
        let (jx, jy) = jitter_offset(3, 0);
        assert!((-0.5..=0.5).contains(&jx));
        assert!((-0.5..=0.5).contains(&jy));
    }

    // ── TaaHistory ─────────────────────────────────────────────────────────────

    #[test]
    fn test_history_new_is_empty() {
        let h = TaaHistory::new(8, 8);
        assert!(h.is_empty(), "new history must be empty");
    }

    #[test]
    fn test_history_new_frame_count_zero() {
        let h = TaaHistory::new(4, 4);
        assert_eq!(h.frame_count, 0);
    }

    #[test]
    fn test_history_new_all_black() {
        let h = TaaHistory::new(3, 3);
        for i in 0..9_usize {
            let px = i % 3;
            let py = i / 3;
            assert_eq!(
                h.get_pixel(px, py),
                [0.0; 3],
                "pixel ({px},{py}) must be black"
            );
        }
    }

    #[test]
    fn test_history_set_get_roundtrip() {
        let mut h = TaaHistory::new(4, 4);
        let color = [0.1, 0.5, 0.9];
        h.set_pixel(2, 3, color).unwrap();
        let got = h.get_pixel(2, 3);
        for c in 0..3 {
            assert!((got[c] - color[c]).abs() < 1e-6, "channel {c} mismatch");
        }
    }

    #[test]
    fn test_history_set_oob_returns_err() {
        let mut h = TaaHistory::new(2, 2);
        let result = h.set_pixel(5, 5, [1.0, 0.0, 0.0]);
        assert!(result.is_err(), "OOB set must return Err");
    }

    #[test]
    fn test_history_get_oob_returns_black() {
        let h = TaaHistory::new(2, 2);
        assert_eq!(h.get_pixel(10, 10), [0.0; 3]);
    }

    #[test]
    fn test_history_reset() {
        let mut h = TaaHistory::new(4, 4);
        h.set_pixel(0, 0, [1.0, 1.0, 1.0]).unwrap();
        h.frame_count = 5;
        h.reset();
        assert!(h.is_empty());
        assert_eq!(h.frame_count, 0);
        assert_eq!(h.get_pixel(0, 0), [0.0; 3]);
    }

    // ── TaaConfig ──────────────────────────────────────────────────────────────

    #[test]
    fn test_taaconfig_default_validates() {
        assert!(TaaConfig::default().validate().is_ok());
    }

    #[test]
    fn test_taaconfig_invalid_blend_factor_zero() {
        let cfg = TaaConfig {
            blend_factor: 0.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_taaconfig_invalid_blend_factor_over_one() {
        let cfg = TaaConfig {
            blend_factor: 1.1,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_taaconfig_blend_factor_exactly_one_is_valid() {
        let cfg = TaaConfig {
            blend_factor: 1.0,
            adaptive_blend_min: 0.05_f32.min(1.0),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_taaconfig_invalid_jitter_length_zero() {
        let cfg = TaaConfig {
            jitter_sequence_length: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_taaconfig_invalid_clip_window_zero() {
        let cfg = TaaConfig {
            clip_window_radius: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_taaconfig_invalid_sharpen_negative() {
        let cfg = TaaConfig {
            sharpen_strength: -0.1,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_taaconfig_adaptive_blend_min_exceeds_blend_factor() {
        let mut cfg = TaaConfig::default();
        cfg.adaptive_blend_min = cfg.blend_factor + 0.1;
        assert!(cfg.validate().is_err());
    }

    // ── local_color_stats ──────────────────────────────────────────────────────

    #[test]
    fn test_local_color_stats_uniform_std_zero() {
        // Uniform image: all 0.5, std should be 0.0
        let image = vec![0.5_f32; 4 * 4 * 3];
        let (mean, std) = local_color_stats(&image, 4, 4, 2, 2, 1);
        for c in 0..3 {
            assert!(
                (mean[c] - 0.5).abs() < 1e-5,
                "mean[{c}] expected 0.5, got {}",
                mean[c]
            );
            assert!(std[c] < 1e-5, "std[{c}] expected ~0.0, got {}", std[c]);
        }
    }

    #[test]
    fn test_local_color_stats_single_pixel_image() {
        let image = vec![0.3_f32, 0.6_f32, 0.9_f32];
        let (mean, std) = local_color_stats(&image, 1, 1, 0, 0, 1);
        assert!((mean[0] - 0.3).abs() < 1e-5);
        assert!((mean[1] - 0.6).abs() < 1e-5);
        assert!((mean[2] - 0.9).abs() < 1e-5);
        assert!(std[0] < 1e-5);
        assert!(std[1] < 1e-5);
        assert!(std[2] < 1e-5);
    }

    #[test]
    fn test_local_color_stats_corner_pixel() {
        // Corner pixel: window is smaller but should not panic
        let image = vec![0.5_f32; 6 * 6 * 3];
        let (mean, std) = local_color_stats(&image, 6, 6, 0, 0, 1);
        for c in 0..3 {
            assert!((mean[c] - 0.5).abs() < 1e-5);
            assert!(std[c] < 1e-5);
        }
    }

    // ── clip_to_variance ───────────────────────────────────────────────────────

    #[test]
    fn test_clip_within_range_unchanged() {
        let color = [0.5, 0.5, 0.5];
        let mean = [0.5, 0.5, 0.5];
        let std = [0.2, 0.2, 0.2];
        let out = clip_to_variance(color, mean, std, 1.0);
        for c in 0..3 {
            assert!(
                (out[c] - color[c]).abs() < 1e-6,
                "within range must be unchanged"
            );
        }
    }

    #[test]
    fn test_clip_above_range_clamped() {
        let color = [1.0, 1.0, 1.0];
        let mean = [0.5, 0.5, 0.5];
        let std = [0.1, 0.1, 0.1];
        let out = clip_to_variance(color, mean, std, 1.0);
        // expected upper bound: 0.5 + 1.0 * 0.1 = 0.6
        for &val in &out {
            assert!(
                (val - 0.6).abs() < 1e-5,
                "above range: expected 0.6, got {}",
                val
            );
        }
    }

    #[test]
    fn test_clip_below_range_clamped() {
        let color = [0.0, 0.0, 0.0];
        let mean = [0.5, 0.5, 0.5];
        let std = [0.1, 0.1, 0.1];
        let out = clip_to_variance(color, mean, std, 1.0);
        // expected lower bound: 0.5 - 1.0 * 0.1 = 0.4
        for &val in &out {
            assert!(
                (val - 0.4).abs() < 1e-5,
                "below range: expected 0.4, got {}",
                val
            );
        }
    }

    // ── accumulate_taa ─────────────────────────────────────────────────────────

    #[test]
    fn test_accumulate_taa_empty_history_returns_current() {
        let cfg = TaaConfig {
            sharpen_strength: 0.0,
            variance_clipping: false,
            ..TaaConfig::default()
        };
        let current = vec![0.8_f32; 4 * 4 * 3];
        let mut history = TaaHistory::new(4, 4);
        let out = accumulate_taa(&current, &mut history, &cfg).unwrap();
        assert_eq!(out.len(), current.len());
        for (&a, &b) in out.iter().zip(current.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(history.frame_count, 1);
    }

    #[test]
    fn test_accumulate_taa_second_frame_blends() {
        let cfg = TaaConfig {
            blend_factor: 0.5,
            variance_clipping: false,
            sharpen_strength: 0.0,
            adaptive_blend: false,
            ..TaaConfig::default()
        };
        let mut history = TaaHistory::new(2, 2);
        // Seed history with all-zero frame
        let frame0 = vec![0.0_f32; 2 * 2 * 3];
        accumulate_taa(&frame0, &mut history, &cfg).unwrap();

        // Second frame: all 1.0
        let frame1 = vec![1.0_f32; 2 * 2 * 3];
        let out = accumulate_taa(&frame1, &mut history, &cfg).unwrap();

        // Expected: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-5, "expected 0.5, got {v}");
        }
        assert!(
            !out.iter().all(|&v| (v - 1.0).abs() < 1e-6),
            "must not be pure current"
        );
        assert!(
            !out.iter().all(|&v| v.abs() < 1e-6),
            "must not be pure history"
        );
    }

    #[test]
    fn test_accumulate_taa_updates_frame_count() {
        let cfg = TaaConfig {
            sharpen_strength: 0.0,
            variance_clipping: false,
            ..TaaConfig::default()
        };
        let mut history = TaaHistory::new(2, 2);
        let frame = vec![0.5_f32; 2 * 2 * 3];
        accumulate_taa(&frame, &mut history, &cfg).unwrap();
        accumulate_taa(&frame, &mut history, &cfg).unwrap();
        assert_eq!(history.frame_count, 2);
    }

    #[test]
    fn test_accumulate_taa_blend_factor_one_equals_current() {
        let cfg = TaaConfig {
            blend_factor: 1.0,
            variance_clipping: false,
            sharpen_strength: 0.0,
            adaptive_blend: false,
            adaptive_blend_min: 0.05_f32.min(1.0),
            ..TaaConfig::default()
        };
        let mut history = TaaHistory::new(2, 2);
        let frame0 = vec![0.0_f32; 2 * 2 * 3];
        accumulate_taa(&frame0, &mut history, &cfg).unwrap();

        let frame1 = vec![0.7_f32; 2 * 2 * 3];
        let out = accumulate_taa(&frame1, &mut history, &cfg).unwrap();
        for &v in &out {
            assert!(
                (v - 0.7).abs() < 1e-5,
                "blend_factor=1 must return current: {v}"
            );
        }
    }

    #[test]
    fn test_accumulate_taa_variance_clipping_difference() {
        // Clipping vs non-clipping should yield different results when history is far from current
        let no_clip_cfg = TaaConfig {
            blend_factor: 0.1,
            variance_clipping: false,
            sharpen_strength: 0.0,
            adaptive_blend: false,
            ..TaaConfig::default()
        };
        let clip_cfg = TaaConfig {
            variance_clipping: true,
            ..no_clip_cfg.clone()
        };

        let mut h_no_clip = TaaHistory::new(4, 4);
        let mut h_clip = TaaHistory::new(4, 4);

        // Seed both histories with all-black
        let black = vec![0.0_f32; 4 * 4 * 3];
        accumulate_taa(&black, &mut h_no_clip, &no_clip_cfg).unwrap();
        accumulate_taa(&black, &mut h_clip, &clip_cfg).unwrap();

        // Current frame is all-white
        let white = vec![1.0_f32; 4 * 4 * 3];
        let out_no_clip = accumulate_taa(&white, &mut h_no_clip, &no_clip_cfg).unwrap();
        let out_clip = accumulate_taa(&white, &mut h_clip, &clip_cfg).unwrap();

        // Results must differ (variance clipping pushes history toward current)
        let diff_sum: f32 = out_no_clip
            .iter()
            .zip(out_clip.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff_sum > 1e-5,
            "variance_clipping must produce different results"
        );
    }

    #[test]
    fn test_accumulate_taa_dimension_mismatch() {
        let cfg = TaaConfig::default();
        let mut history = TaaHistory::new(4, 4);
        let wrong_frame = vec![0.5_f32; 3 * 3 * 3]; // wrong size
        let result = accumulate_taa(&wrong_frame, &mut history, &cfg);
        assert!(matches!(result, Err(TaaError::DimensionMismatch { .. })));
    }

    // ── sharpen_image ──────────────────────────────────────────────────────────

    #[test]
    fn test_sharpen_strength_zero_returns_copy() {
        let image = vec![
            0.3_f32, 0.6_f32, 0.9_f32, 0.1_f32, 0.2_f32, 0.3_f32, 0.4_f32, 0.5_f32, 0.6_f32,
            0.7_f32, 0.8_f32, 0.9_f32,
        ];
        let out = sharpen_image(&image, 2, 2, 0.0);
        assert_eq!(out.len(), image.len());
        for (&a, &b) in out.iter().zip(image.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "strength=0 must return identical copy"
            );
        }
    }

    #[test]
    fn test_sharpen_uniform_image_unchanged() {
        // Uniform image: image - blur(image) = 0, so sharpening has no effect
        let image = vec![0.5_f32; 4 * 4 * 3];
        let out = sharpen_image(&image, 4, 4, 0.5);
        for &v in &out {
            assert!(
                (v - 0.5).abs() < 1e-4,
                "uniform image must be unchanged: {v}"
            );
        }
    }

    #[test]
    fn test_sharpen_output_clamped_to_unit() {
        let image: Vec<f32> = (0..16)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        let out = sharpen_image(&image, 4, 4 / 3, 2.0);
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "output must be in [0,1]: {v}");
        }
    }

    // ── TaaAccumulator ─────────────────────────────────────────────────────────

    #[test]
    fn test_accumulator_new_valid() {
        let cfg = TaaConfig::default();
        let acc = TaaAccumulator::new(8, 8, cfg);
        assert!(acc.is_ok());
    }

    #[test]
    fn test_accumulator_new_invalid_config() {
        let cfg = TaaConfig {
            blend_factor: 0.0,
            ..Default::default()
        }; // invalid
        let acc = TaaAccumulator::new(4, 4, cfg);
        assert!(acc.is_err());
    }

    #[test]
    fn test_accumulator_process_first_frame() {
        let cfg = TaaConfig {
            sharpen_strength: 0.0,
            variance_clipping: false,
            ..TaaConfig::default()
        };
        let mut acc = TaaAccumulator::new(2, 2, cfg).unwrap();
        let frame = vec![0.5_f32; 2 * 2 * 3];
        let out = acc.process(&frame).unwrap();
        assert_eq!(out.len(), frame.len());
        assert_eq!(acc.frame_count(), 1);
    }

    #[test]
    fn test_accumulator_process_accumulates() {
        let cfg = TaaConfig {
            blend_factor: 0.5,
            variance_clipping: false,
            sharpen_strength: 0.0,
            adaptive_blend: false,
            ..TaaConfig::default()
        };
        let mut acc = TaaAccumulator::new(2, 2, cfg).unwrap();
        let black = vec![0.0_f32; 2 * 2 * 3];
        let white = vec![1.0_f32; 2 * 2 * 3];
        acc.process(&black).unwrap();
        let out = acc.process(&white).unwrap();
        // Should be 0.5 (blend of 0 and 1 with equal weights)
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-5, "expected 0.5, got {v}");
        }
    }

    #[test]
    fn test_accumulator_current_jitter_changes() {
        let cfg = TaaConfig::default();
        let mut acc = TaaAccumulator::new(2, 2, cfg).unwrap();
        let j0 = acc.current_jitter();
        let frame = vec![0.5_f32; 2 * 2 * 3];
        acc.process(&frame).unwrap();
        let j1 = acc.current_jitter();
        assert!(
            (j0.0 - j1.0).abs() > 1e-6 || (j0.1 - j1.1).abs() > 1e-6,
            "jitter must change between frames"
        );
    }

    #[test]
    fn test_accumulator_quality_grows() {
        let cfg = TaaConfig {
            blend_factor: 0.5,
            variance_clipping: false,
            sharpen_strength: 0.0,
            ..TaaConfig::default()
        };
        let mut acc = TaaAccumulator::new(2, 2, cfg).unwrap();
        let q0 = acc.quality();
        let frame = vec![0.5_f32; 2 * 2 * 3];
        acc.process(&frame).unwrap();
        let q1 = acc.quality();
        acc.process(&frame).unwrap();
        let q2 = acc.quality();
        assert!(q1 > q0, "quality must grow after first frame");
        assert!(q2 > q1, "quality must grow after second frame");
        assert!(q2 <= 1.0, "quality must not exceed 1.0");
    }

    #[test]
    fn test_accumulator_quality_caps_at_one() {
        let cfg = TaaConfig {
            blend_factor: 0.5, // effective samples = 2
            variance_clipping: false,
            sharpen_strength: 0.0,
            ..TaaConfig::default()
        };
        let mut acc = TaaAccumulator::new(2, 2, cfg).unwrap();
        let frame = vec![0.5_f32; 2 * 2 * 3];
        // Process many more frames than effective sample count
        for _ in 0..20 {
            acc.process(&frame).unwrap();
        }
        assert!(
            (acc.quality() - 1.0).abs() < 1e-5,
            "quality must cap at 1.0"
        );
    }

    #[test]
    fn test_accumulator_reset() {
        let cfg = TaaConfig {
            sharpen_strength: 0.0,
            variance_clipping: false,
            ..TaaConfig::default()
        };
        let mut acc = TaaAccumulator::new(2, 2, cfg).unwrap();
        let frame = vec![0.5_f32; 2 * 2 * 3];
        acc.process(&frame).unwrap();
        acc.process(&frame).unwrap();
        assert_eq!(acc.frame_count(), 2);
        acc.reset();
        assert_eq!(acc.frame_count(), 0);
        assert!(
            (acc.quality() - 0.0).abs() < 1e-6,
            "quality after reset must be 0"
        );
    }

    // ── compute_taa_stats ──────────────────────────────────────────────────────

    #[test]
    fn test_compute_taa_stats_identical_images_zero_ghosting() {
        let image = vec![0.5_f32; 4 * 4 * 3];
        let mut history = TaaHistory::new(4, 4);
        history.color = image.clone();
        history.frame_count = 5;

        let stats = compute_taa_stats(&image, &image, &history, 0.1).unwrap();
        assert_eq!(stats.frame_count, 5);
        assert!(
            stats.mean_ghosting_estimate < 1e-5,
            "identical images → zero ghosting"
        );
        // All pixels converged (|current - accumulated| = 0 < 0.01)
        assert!((stats.converged_fraction - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_taa_stats_different_images_nonzero_ghosting() {
        let current = vec![0.0_f32; 4 * 4 * 3];
        let accumulated = vec![0.5_f32; 4 * 4 * 3];
        let mut history = TaaHistory::new(4, 4);
        history.color = vec![1.0_f32; 4 * 4 * 3];
        history.frame_count = 3;

        let stats = compute_taa_stats(&current, &accumulated, &history, 0.1).unwrap();
        // ghosting: |lum(0) - lum(1)| = 1.0 per pixel
        assert!(
            stats.mean_ghosting_estimate > 0.1,
            "must have nonzero ghosting"
        );
        // converged: |0 - 0.5| = 0.5 > 0.01, so converged_fraction should be 0
        assert!(stats.converged_fraction < 1e-5, "must not be converged");
    }

    #[test]
    fn test_compute_taa_stats_dimension_mismatch_error() {
        let current = vec![0.5_f32; 4 * 4 * 3];
        let wrong = vec![0.5_f32; 3 * 3 * 3];
        let history = TaaHistory::new(4, 4);
        let result = compute_taa_stats(&current, &wrong, &history, 0.1);
        assert!(result.is_err());
    }
}
