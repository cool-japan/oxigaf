//! Summary statistics for a completed TAA accumulation step.

use super::clipping::luminance;
use super::config::TaaError;
use super::history::TaaHistory;

// ---------------------------------------------------------------------------
// TAA statistics
// ---------------------------------------------------------------------------

/// Summary statistics computed over a completed TAA accumulation step.
#[derive(Debug, Clone)]
pub struct TaaStats {
    /// Number of frames accumulated in the history buffer.
    pub frame_count: usize,
    /// Mean blend factor actually applied across all pixels of the
    /// accumulation step recorded in the `history` passed to
    /// [`compute_taa_stats`].
    ///
    /// This is the measurement [`crate::temporal_aa::accumulate_taa`] took while blending (see
    /// [`TaaHistory::last_mean_blend_factor`]), so with `adaptive_blend`
    /// enabled it is the true per-pixel average rather than the nominal
    /// config value. It falls back to the caller-supplied nominal
    /// `blend_factor` only when the history carries no recorded step — a
    /// fresh or reset buffer, or one whose fields were assembled by hand.
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
/// - `accumulated`: the blended result from [`crate::temporal_aa::accumulate_taa`].
/// - `history`: the history buffer (already updated with this frame). Its
///   [`TaaHistory::last_mean_blend_factor`] supplies the *measured* mean α
///   reported in [`TaaStats::mean_blend_factor`].
/// - `blend_factor`: the nominal blend factor from [`crate::temporal_aa::TaaConfig`], used only
///   as a fallback when `history` carries no recorded accumulation step.
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
            mean_blend_factor: history.last_mean_blend_factor.unwrap_or(blend_factor),
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
        mean_blend_factor: history.last_mean_blend_factor.unwrap_or(blend_factor),
        mean_ghosting_estimate: ghosting_sum / pixel_count as f32,
        converged_fraction: converged_count as f32 / pixel_count as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_aa::TaaHistory;

    // ── compute_taa_stats ──────────────────────────────────────────────────────

    #[test]
    fn test_compute_taa_stats_identical_images_zero_ghosting() {
        let image = vec![0.5_f32; 4 * 4 * 3];
        let mut history = TaaHistory::new(4, 4);
        history.color = image.clone();
        history.frame_count = 5;

        let stats = compute_taa_stats(&image, &image, &history, 0.1)
            .expect("compute_taa_stats must succeed");
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

        let stats = compute_taa_stats(&current, &accumulated, &history, 0.1)
            .expect("compute_taa_stats must succeed");
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
