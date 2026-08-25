//! Post-process sharpening and the core TAA accumulation step.

use super::clipping::{clip_to_variance, lerp, luminance, RowWindowStats};
use super::config::{TaaConfig, TaaError};
use super::history::TaaHistory;
use super::jitter::jitter_offset;

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
/// 4. Write the result back into `history`, increment `frame_count`, and
///    record the measured mean of the per-pixel blend factors in
///    [`crate::temporal_aa::TaaHistory::last_mean_blend_factor`] for [`crate::temporal_aa::compute_taa_stats`].
///
/// # Performance
///
/// With `variance_clipping` enabled the local mean/σ of `current` comes from
/// a row-sliding running sum (`O(1)` per pixel) rather than a per-pixel call
/// to [`crate::temporal_aa::local_color_stats`] (`O((2r+1)²)` per pixel). The windows are
/// identical; the running sums are accumulated in `f64`, so results agree
/// with [`crate::temporal_aa::local_color_stats`] up to `f32` summation-order noise.
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
        // Nothing was blended: the returned image is 100% current frame.
        history.last_mean_blend_factor = if expected == 0 { None } else { Some(1.0) };
        return Ok(current.to_vec());
    }

    let width = history.width;
    let height = history.height;

    let mut blended = vec![0.0_f32; expected];

    // Variance clipping needs the local mean/σ of `current` around every
    // pixel. `local_color_stats` re-sums the whole window per pixel
    // (`O(w · h · r²)` per frame); `RowWindowStats` keeps running column sums
    // and slides them, which is `O(w · h)` for the same windows.
    let mut clip_stats = if config.variance_clipping {
        Some(RowWindowStats::new(
            width,
            height,
            config.clip_window_radius,
        ))
    } else {
        None
    };

    // Ground truth for `TaaStats::mean_blend_factor`: the per-pixel alpha is
    // only knowable here (it depends on the *pre-blend* history, which is
    // overwritten below), so `accumulate_taa` records it rather than letting
    // the stats reader re-assert the nominal config value.
    let mut alpha_sum = 0.0_f64;

    for py in 0..height {
        if let Some(stats) = clip_stats.as_mut() {
            stats.advance_to_row(current, py);
        }
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
            if let Some(stats) = clip_stats.as_ref() {
                let (mean, std) = stats.pixel(px);
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
            alpha_sum += alpha as f64;

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
    let pixel_count = width * height;
    history.last_mean_blend_factor = if pixel_count == 0 {
        None
    } else {
        Some((alpha_sum / pixel_count as f64) as f32)
    };

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_aa::compute_taa_stats;

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
        let out =
            accumulate_taa(&current, &mut history, &cfg).expect("accumulate_taa must succeed");
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
        accumulate_taa(&frame0, &mut history, &cfg).expect("accumulate_taa must succeed");

        // Second frame: all 1.0
        let frame1 = vec![1.0_f32; 2 * 2 * 3];
        let out = accumulate_taa(&frame1, &mut history, &cfg).expect("accumulate_taa must succeed");

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
        accumulate_taa(&frame, &mut history, &cfg).expect("accumulate_taa must succeed");
        accumulate_taa(&frame, &mut history, &cfg).expect("accumulate_taa must succeed");
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
        accumulate_taa(&frame0, &mut history, &cfg).expect("accumulate_taa must succeed");

        let frame1 = vec![0.7_f32; 2 * 2 * 3];
        let out = accumulate_taa(&frame1, &mut history, &cfg).expect("accumulate_taa must succeed");
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
        accumulate_taa(&black, &mut h_no_clip, &no_clip_cfg).expect("accumulate_taa must succeed");
        accumulate_taa(&black, &mut h_clip, &clip_cfg).expect("accumulate_taa must succeed");

        // Current frame is all-white
        let white = vec![1.0_f32; 4 * 4 * 3];
        let out_no_clip = accumulate_taa(&white, &mut h_no_clip, &no_clip_cfg)
            .expect("accumulate_taa must succeed");
        let out_clip =
            accumulate_taa(&white, &mut h_clip, &clip_cfg).expect("accumulate_taa must succeed");

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
        let mut acc = TaaAccumulator::new(2, 2, cfg).expect("TaaAccumulator::new must succeed");
        let frame = vec![0.5_f32; 2 * 2 * 3];
        let out = acc.process(&frame).expect("process must succeed");
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
        let mut acc = TaaAccumulator::new(2, 2, cfg).expect("TaaAccumulator::new must succeed");
        let black = vec![0.0_f32; 2 * 2 * 3];
        let white = vec![1.0_f32; 2 * 2 * 3];
        acc.process(&black).expect("process must succeed");
        let out = acc.process(&white).expect("process must succeed");
        // Should be 0.5 (blend of 0 and 1 with equal weights)
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-5, "expected 0.5, got {v}");
        }
    }

    #[test]
    fn test_accumulator_current_jitter_changes() {
        let cfg = TaaConfig::default();
        let mut acc = TaaAccumulator::new(2, 2, cfg).expect("TaaAccumulator::new must succeed");
        let j0 = acc.current_jitter();
        let frame = vec![0.5_f32; 2 * 2 * 3];
        acc.process(&frame).expect("process must succeed");
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
        let mut acc = TaaAccumulator::new(2, 2, cfg).expect("TaaAccumulator::new must succeed");
        let q0 = acc.quality();
        let frame = vec![0.5_f32; 2 * 2 * 3];
        acc.process(&frame).expect("process must succeed");
        let q1 = acc.quality();
        acc.process(&frame).expect("process must succeed");
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
        let mut acc = TaaAccumulator::new(2, 2, cfg).expect("TaaAccumulator::new must succeed");
        let frame = vec![0.5_f32; 2 * 2 * 3];
        // Process many more frames than effective sample count
        for _ in 0..20 {
            acc.process(&frame).expect("process must succeed");
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
        let mut acc = TaaAccumulator::new(2, 2, cfg).expect("TaaAccumulator::new must succeed");
        let frame = vec![0.5_f32; 2 * 2 * 3];
        acc.process(&frame).expect("process must succeed");
        acc.process(&frame).expect("process must succeed");
        assert_eq!(acc.frame_count(), 2);
        acc.reset();
        assert_eq!(acc.frame_count(), 0);
        assert!(
            (acc.quality() - 0.0).abs() < 1e-6,
            "quality after reset must be 0"
        );
    }

    // ── measured mean blend factor ────────────────────────────────────────────

    #[test]
    fn test_accumulate_taa_records_measured_blend_factor() {
        // Regression (F260): `TaaStats::mean_blend_factor` must be measured,
        // not echoed back from the config. With adaptive blending on, the
        // real mean lies strictly between `adaptive_blend_min` and
        // `blend_factor`, so echoing either one is detectable.
        let (w, h) = (6_usize, 6_usize);
        let frame0 = vec![0.0_f32; w * h * 3];
        let mut frame1 = vec![0.0_f32; w * h * 3];
        // Half the pixels are pure white → large luminance difference → alpha
        // near `blend_factor`; the other half stay black → alpha near
        // `adaptive_blend_min`.
        for px in 0..(w * h) {
            if px.is_multiple_of(2) {
                for c in 0..3 {
                    frame1[px * 3 + c] = 1.0;
                }
            }
        }

        let config = TaaConfig {
            variance_clipping: false,
            sharpen_strength: 0.0,
            adaptive_blend: true,
            blend_factor: 0.8,
            adaptive_blend_min: 0.05,
            ..TaaConfig::default()
        };
        config.validate().expect("config must be valid");

        let mut history = TaaHistory::new(w, h);
        assert_eq!(
            history.last_mean_blend_factor, None,
            "fresh history must carry no measurement"
        );

        accumulate_taa(&frame0, &mut history, &config).expect("bootstrap");
        assert_eq!(
            history.last_mean_blend_factor,
            Some(1.0),
            "the bootstrap frame is returned unblended"
        );

        let accumulated = accumulate_taa(&frame1, &mut history, &config).expect("blend");
        let measured = history
            .last_mean_blend_factor
            .expect("accumulate_taa must record a measurement");
        assert!(
            measured > config.adaptive_blend_min + 1e-3 && measured < config.blend_factor - 1e-3,
            "measured mean α {measured} must lie strictly between {} and {}",
            config.adaptive_blend_min,
            config.blend_factor
        );

        let stats =
            compute_taa_stats(&frame1, &accumulated, &history, config.blend_factor).expect("stats");
        assert!(
            (stats.mean_blend_factor - measured).abs() < 1e-6,
            "stats must report the measured α, got {}",
            stats.mean_blend_factor
        );
    }

    #[test]
    fn test_accumulate_taa_measured_blend_factor_non_adaptive() {
        // Non-adaptive mode: every pixel uses `blend_factor`, so the measured
        // mean must reproduce it exactly.
        let (w, h) = (4_usize, 3_usize);
        let frame0 = vec![0.2_f32; w * h * 3];
        let frame1 = vec![0.9_f32; w * h * 3];
        let config = TaaConfig {
            variance_clipping: false,
            sharpen_strength: 0.0,
            adaptive_blend: false,
            blend_factor: 0.3,
            ..TaaConfig::default()
        };

        let mut history = TaaHistory::new(w, h);
        accumulate_taa(&frame0, &mut history, &config).expect("bootstrap");
        accumulate_taa(&frame1, &mut history, &config).expect("blend");
        let measured = history.last_mean_blend_factor.expect("recorded");
        assert!(
            (measured - 0.3).abs() < 1e-6,
            "expected 0.3, got {measured}"
        );
    }

    #[test]
    fn test_taa_history_reset_drops_measurement() {
        // Regression: a measurement from before a reset describes a run that
        // no longer exists; leaking it would re-introduce the stale statistic.
        let (w, h) = (3_usize, 3_usize);
        let config = TaaConfig {
            variance_clipping: false,
            sharpen_strength: 0.0,
            ..TaaConfig::default()
        };
        let mut history = TaaHistory::new(w, h);
        accumulate_taa(&vec![0.1_f32; w * h * 3], &mut history, &config).expect("bootstrap");
        accumulate_taa(&vec![0.6_f32; w * h * 3], &mut history, &config).expect("blend");
        assert!(history.last_mean_blend_factor.is_some());

        history.reset();
        assert_eq!(
            history.last_mean_blend_factor, None,
            "reset must drop the recorded blend factor"
        );

        // With no measurement, stats fall back to the nominal value.
        let image = vec![0.0_f32; w * h * 3];
        let stats = compute_taa_stats(&image, &image, &history, 0.17).expect("stats");
        assert!(
            (stats.mean_blend_factor - 0.17).abs() < 1e-6,
            "expected the nominal fallback, got {}",
            stats.mean_blend_factor
        );
    }
}
