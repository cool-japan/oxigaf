//! The TAA history buffer: accumulated color plus the bookkeeping that
//! [`crate::temporal_aa::compute_taa_stats`] reports on.

use super::config::TaaError;

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
    /// Mean of the per-pixel blend factors the most recent [`crate::temporal_aa::accumulate_taa`]
    /// call actually applied, recorded by that call.
    ///
    /// - `None` on a fresh or [`TaaHistory::reset`] buffer — no accumulation
    ///   step has run, so there is nothing to report.
    /// - `Some(1.0)` after the bootstrap frame, which is returned unblended
    ///   (100 % current frame).
    /// - Otherwise the measured mean α. With `adaptive_blend` disabled this
    ///   equals `TaaConfig::blend_factor`; with it enabled it is the true
    ///   per-pixel average, which the config alone cannot predict.
    ///
    /// Read by [`crate::temporal_aa::compute_taa_stats`] to populate
    /// [`crate::temporal_aa::TaaStats::mean_blend_factor`].
    pub last_mean_blend_factor: Option<f32>,
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
            last_mean_blend_factor: None,
        }
    }

    /// Returns `true` if no frames have been accumulated yet.
    pub fn is_empty(&self) -> bool {
        self.frame_count == 0
    }

    /// Reset the history to black and zero the frame count.
    ///
    /// Also drops the recorded [`TaaHistory::last_mean_blend_factor`]: a
    /// measurement from before the reset describes an accumulation run that
    /// no longer exists, and reporting it would be exactly the kind of stale
    /// statistic the field is meant to replace.
    pub fn reset(&mut self) {
        self.color.iter_mut().for_each(|v| *v = 0.0);
        self.frame_count = 0;
        self.last_mean_blend_factor = None;
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

#[cfg(test)]
mod tests {
    use super::*;

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
        h.set_pixel(2, 3, color).expect("set_pixel must succeed");
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
        h.set_pixel(0, 0, [1.0, 1.0, 1.0])
            .expect("set_pixel must succeed");
        h.frame_count = 5;
        h.reset();
        assert!(h.is_empty());
        assert_eq!(h.frame_count, 0);
        assert_eq!(h.get_pixel(0, 0), [0.0; 3]);
    }
}
