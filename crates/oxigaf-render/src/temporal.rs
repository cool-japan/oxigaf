//! Temporal Anti-Aliasing (TAA) frame accumulation utilities.
//!
//! Provides motion vector fields, frame accumulators, and temporal blending
//! for ghosting-reduced TAA using neighborhood clamping and disocclusion detection.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during temporal anti-aliasing operations.
#[derive(Debug, Error)]
pub enum TaaError {
    /// No history frame available (accumulate at least one frame first).
    #[error("No history frame available")]
    NoHistory,

    /// Frame dimension mismatch between current frame and stored history.
    #[error(
        "Dimension mismatch: current frame has {current} pixels, history has {history} pixels"
    )]
    DimensionMismatch {
        /// Size of current frame (pixels * channels).
        current: usize,
        /// Size of history frame (pixels * channels).
        history: usize,
    },

    /// Motion field pixel count does not match frame dimensions.
    #[error("Invalid motion field: expected {expected_pixels} pixels, got {got}")]
    InvalidMotionField {
        /// Expected number of motion vectors.
        expected_pixels: usize,
        /// Actual number of motion vectors provided.
        got: usize,
    },

    /// Width or height is zero.
    #[error("Zero dimension: width and height must be > 0")]
    ZeroDimension,

    /// Channel count is zero or unsupported.
    #[error("Invalid channels: must be > 0")]
    InvalidChannels,
}

// ---------------------------------------------------------------------------
// Motion vector
// ---------------------------------------------------------------------------

/// Per-pixel 2D screen-space motion vector (displacement in pixels).
#[derive(Debug, Clone, Copy, Default)]
pub struct MotionVector {
    /// Horizontal displacement (positive = right).
    pub dx: f32,
    /// Vertical displacement (positive = down).
    pub dy: f32,
}

impl MotionVector {
    /// Euclidean magnitude of the motion vector.
    pub fn magnitude(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Returns `true` when magnitude is below `threshold` (pixel is effectively static).
    pub fn is_static(&self, threshold: f32) -> bool {
        self.magnitude() < threshold
    }
}

// ---------------------------------------------------------------------------
// Motion vector field
// ---------------------------------------------------------------------------

/// Motion vector field covering an entire frame (row-major, y*width+x).
pub struct MotionVectorField {
    /// Per-pixel motion vectors stored in row-major order.
    pub vectors: Vec<MotionVector>,
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
}

impl MotionVectorField {
    /// Create a new field with all vectors set to zero.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            vectors: vec![MotionVector::default(); width * height],
            width,
            height,
        }
    }

    /// Convenience alias — identical to [`Self::new`].
    pub fn zero_motion(width: usize, height: usize) -> Self {
        Self::new(width, height)
    }

    /// Return the motion vector at `(x, y)`, or `None` if out of bounds.
    pub fn get(&self, x: usize, y: usize) -> Option<&MotionVector> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.vectors.get(y * self.width + x)
    }

    /// Set the motion vector at `(x, y)`. Returns `false` if out of bounds.
    pub fn set(&mut self, x: usize, y: usize, mv: MotionVector) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        let idx = y * self.width + x;
        if let Some(slot) = self.vectors.get_mut(idx) {
            *slot = mv;
            true
        } else {
            false
        }
    }

    /// Average magnitude across all motion vectors.
    pub fn mean_magnitude(&self) -> f32 {
        if self.vectors.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.vectors.iter().map(|mv| mv.magnitude()).sum();
        sum / self.vectors.len() as f32
    }

    /// Maximum magnitude across all motion vectors.
    pub fn max_magnitude(&self) -> f32 {
        self.vectors
            .iter()
            .map(|mv| mv.magnitude())
            .fold(0.0_f32, f32::max)
    }
}

// ---------------------------------------------------------------------------
// TAA configuration
// ---------------------------------------------------------------------------

/// Configuration parameters for temporal anti-aliasing.
#[derive(Debug, Clone)]
pub struct TaaConfig {
    /// Blend weight for the current frame (0.0–1.0); the complement (1 - weight) goes to history.
    pub current_weight: f32,
    /// Disocclusion threshold: if max-component color distance from history exceeds this,
    /// treat the pixel as disoccluded and prefer the current frame.
    pub disocclusion_threshold: f32,
    /// When enabled, clamp warped history to the 3×3 AABB of current pixel's neighbors.
    pub enable_neighborhood_clamp: bool,
    /// Current-frame weight to use when disocclusion is detected (higher = less ghosting).
    pub disocclusion_blend: f32,
    /// Number of frames after which the accumulator is considered converged.
    pub convergence_frames: usize,
}

impl Default for TaaConfig {
    fn default() -> Self {
        Self {
            current_weight: 0.1,
            disocclusion_threshold: 0.1,
            enable_neighborhood_clamp: true,
            disocclusion_blend: 0.5,
            convergence_frames: 8,
        }
    }
}

// ---------------------------------------------------------------------------
// Bilinear sampling helper
// ---------------------------------------------------------------------------

/// Sample `image` (HWC layout) at fractional pixel position `(x, y)` using bilinear
/// interpolation with edge clamping. Returns a `Vec<f32>` of length `channels`.
fn sample_bilinear(
    image: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    x: f32,
    y: f32,
) -> Vec<f32> {
    // Clamp to valid range
    let x0f = x.clamp(0.0, (width.saturating_sub(1)) as f32);
    let y0f = y.clamp(0.0, (height.saturating_sub(1)) as f32);

    let x0 = x0f.floor() as usize;
    let y0 = y0f.floor() as usize;
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));

    let fx = x0f - x0f.floor();
    let fy = y0f - y0f.floor();

    let mut out = vec![0.0_f32; channels];
    for (c, out_val) in out.iter_mut().enumerate() {
        let p00 = image
            .get((y0 * width + x0) * channels + c)
            .copied()
            .unwrap_or(0.0);
        let p10 = image
            .get((y0 * width + x1) * channels + c)
            .copied()
            .unwrap_or(0.0);
        let p01 = image
            .get((y1 * width + x0) * channels + c)
            .copied()
            .unwrap_or(0.0);
        let p11 = image
            .get((y1 * width + x1) * channels + c)
            .copied()
            .unwrap_or(0.0);

        *out_val = (1.0 - fy) * ((1.0 - fx) * p00 + fx * p10) + fy * ((1.0 - fx) * p01 + fx * p11);
    }
    out
}

// ---------------------------------------------------------------------------
// Neighborhood AABB clamp helper
// ---------------------------------------------------------------------------

/// Compute the per-channel min/max over the 3×3 neighborhood (including center) at `(px, py)`.
///
/// Returns `(min_per_channel, max_per_channel)`.
fn neighborhood_minmax(
    image: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    px: usize,
    py: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut min_c = vec![f32::INFINITY; channels];
    let mut max_c = vec![f32::NEG_INFINITY; channels];

    let y_start = py.saturating_sub(1);
    let y_end = (py + 2).min(height);
    let x_start = px.saturating_sub(1);
    let x_end = (px + 2).min(width);

    for ny in y_start..y_end {
        for nx in x_start..x_end {
            let base = (ny * width + nx) * channels;
            for c in 0..channels {
                let v = image.get(base + c).copied().unwrap_or(0.0);
                if v < min_c[c] {
                    min_c[c] = v;
                }
                if v > max_c[c] {
                    max_c[c] = v;
                }
            }
        }
    }

    // Guard against empty neighborhood (zero-size image edge case)
    for c in 0..channels {
        if min_c[c] == f32::INFINITY {
            min_c[c] = 0.0;
            max_c[c] = 0.0;
        }
    }

    (min_c, max_c)
}

// ---------------------------------------------------------------------------
// Frame accumulator
// ---------------------------------------------------------------------------

/// Temporal frame accumulator implementing TAA blending with motion-vector warping.
pub struct TemporalAccumulator {
    config: TaaConfig,
    /// Accumulated history buffer (HWC layout, length = width * height * channels).
    history: Option<Vec<f32>>,
    /// Frame width in pixels.
    pub width: usize,
    /// Frame height in pixels.
    pub height: usize,
    /// Number of channels per pixel (3 for RGB, 4 for RGBA).
    pub channels: usize,
    /// Total number of frames accumulated so far.
    pub frame_count: usize,
}

impl TemporalAccumulator {
    /// Create a new accumulator with no history.
    pub fn new(width: usize, height: usize, channels: usize, config: TaaConfig) -> Self {
        Self {
            config,
            history: None,
            width,
            height,
            channels,
            frame_count: 0,
        }
    }

    /// Returns `true` if history has been initialized.
    pub fn has_history(&self) -> bool {
        self.history.is_some()
    }

    /// Returns `true` when frame count has reached the configured convergence threshold.
    pub fn is_converged(&self) -> bool {
        self.frame_count >= self.config.convergence_frames
    }

    /// Return the configuration used by this accumulator.
    pub fn config(&self) -> &TaaConfig {
        &self.config
    }

    /// Reset the accumulator, clearing history and frame count.
    pub fn reset(&mut self) {
        self.history = None;
        self.frame_count = 0;
    }

    /// Accumulate `current_frame` using `motion_vectors` for history warping.
    ///
    /// # Algorithm
    ///
    /// 1. If no history exists, initialise history from the current frame and return it.
    /// 2. For each pixel:
    ///    a. Warp history by looking it up at `(x - mv.dx, y - mv.dy)` via bilinear sampling.
    ///    b. Optionally clamp warped history to the 3×3 AABB of current pixel's neighborhood.
    ///    c. Compute max-component color distance between current and warped history.
    ///    d. Choose per-pixel blend weight based on disocclusion detection.
    ///    e. Blend `current * weight + warped_history * (1 - weight)`.
    /// 3. Update history; increment frame count.
    pub fn accumulate(
        &mut self,
        current_frame: &[f32],
        motion_vectors: &MotionVectorField,
    ) -> Result<Vec<f32>, TaaError> {
        if self.width == 0 || self.height == 0 {
            return Err(TaaError::ZeroDimension);
        }
        if self.channels == 0 {
            return Err(TaaError::InvalidChannels);
        }

        let expected_frame = self.width * self.height * self.channels;
        if current_frame.len() != expected_frame {
            return Err(TaaError::DimensionMismatch {
                current: current_frame.len(),
                history: expected_frame,
            });
        }

        let expected_pixels = self.width * self.height;
        if motion_vectors.vectors.len() != expected_pixels
            || motion_vectors.width != self.width
            || motion_vectors.height != self.height
        {
            return Err(TaaError::InvalidMotionField {
                expected_pixels,
                got: motion_vectors.vectors.len(),
            });
        }

        // First frame — bootstrap history.
        if self.history.is_none() {
            self.history = Some(current_frame.to_vec());
            self.frame_count = 1;
            return Ok(current_frame.to_vec());
        }

        let history = match self.history.as_deref() {
            Some(h) => h,
            None => return Err(TaaError::NoHistory),
        };

        let w = self.width;
        let h = self.height;
        let ch = self.channels;

        let mut blended = vec![0.0_f32; expected_frame];

        for py in 0..h {
            for px in 0..w {
                let mv = motion_vectors.get(px, py).copied().unwrap_or_default();

                // Warp: sample history at (x - dx, y - dy)
                let src_x = px as f32 - mv.dx;
                let src_y = py as f32 - mv.dy;
                let mut warped = sample_bilinear(history, w, h, ch, src_x, src_y);

                // Neighborhood clamp
                if self.config.enable_neighborhood_clamp {
                    let (n_min, n_max) = neighborhood_minmax(current_frame, w, h, ch, px, py);
                    for c in 0..ch {
                        warped[c] = warped[c].clamp(n_min[c], n_max[c]);
                    }
                }

                // Color distance (max component difference)
                let base_cur = (py * w + px) * ch;
                let mut color_dist = 0.0_f32;
                for (c, &warped_val) in warped.iter().enumerate().take(ch) {
                    let cur_val = current_frame.get(base_cur + c).copied().unwrap_or(0.0);
                    let diff = (cur_val - warped_val).abs();
                    if diff > color_dist {
                        color_dist = diff;
                    }
                }

                // Choose blend weight
                let weight = if color_dist > self.config.disocclusion_threshold {
                    self.config.disocclusion_blend
                } else {
                    self.config.current_weight
                };

                // Blend
                let base_out = base_cur;
                for c in 0..ch {
                    let cur_val = current_frame.get(base_cur + c).copied().unwrap_or(0.0);
                    blended[base_out + c] = weight * cur_val + (1.0 - weight) * warped[c];
                }
            }
        }

        self.history = Some(blended.clone());
        self.frame_count += 1;
        Ok(blended)
    }

    /// Return the current history frame, or an error if not yet initialised.
    pub fn history(&self) -> Result<&[f32], TaaError> {
        match self.history.as_deref() {
            Some(h) => Ok(h),
            None => Err(TaaError::NoHistory),
        }
    }
}

// ---------------------------------------------------------------------------
// Temporal statistics
// ---------------------------------------------------------------------------

/// Summary statistics for a temporal accumulation pass.
#[derive(Debug, Clone, Default)]
pub struct TemporalStats {
    /// Number of frames accumulated so far.
    pub frame_count: usize,
    /// Mean motion magnitude across the motion field (pixels per frame).
    pub mean_motion_magnitude: f32,
    /// Fraction of pixels detected as disoccluded (0.0–1.0).
    pub mean_disocclusion_fraction: f32,
    /// Estimated convergence level (0.0–1.0, clamped at 1.0 once converged).
    pub convergence_estimate: f32,
}

impl TemporalStats {
    /// Compute statistics from the current accumulator state and a motion field.
    pub fn compute(accumulator: &TemporalAccumulator, motion_field: &MotionVectorField) -> Self {
        let frame_count = accumulator.frame_count;
        let convergence_frames = accumulator.config().convergence_frames;

        let convergence_estimate = if convergence_frames == 0 {
            1.0
        } else {
            (frame_count as f32 / convergence_frames as f32).min(1.0)
        };

        let mean_motion_magnitude = motion_field.mean_magnitude();

        // Estimate disocclusion fraction from history if available.
        let mean_disocclusion_fraction =
            if let (Some(history), true) = (&accumulator.history, accumulator.has_history()) {
                let w = accumulator.width;
                let h = accumulator.height;
                let ch = accumulator.channels;
                let threshold = accumulator.config().disocclusion_threshold;
                let total_pixels = w * h;

                if total_pixels == 0 || ch == 0 {
                    0.0
                } else {
                    let disoccluded: usize = (0..h)
                        .flat_map(|py| (0..w).map(move |px| (px, py)))
                        .filter(|&(px, py)| {
                            let base = (py * w + px) * ch;
                            let max_diff = (0..ch)
                                .map(|c| {
                                    let h_val = history.get(base + c).copied().unwrap_or(0.0);
                                    h_val.abs()
                                })
                                .fold(0.0_f32, f32::max);
                            max_diff > threshold
                        })
                        .count();
                    disoccluded as f32 / total_pixels as f32
                }
            } else {
                0.0
            };

        Self {
            frame_count,
            mean_motion_magnitude,
            mean_disocclusion_fraction,
            convergence_estimate,
        }
    }

    /// Format a human-readable one-line summary.
    pub fn format_summary(&self) -> String {
        format!(
            "frames={} convergence={:.1}% mean_motion={:.2}px disocclusion={:.1}%",
            self.frame_count,
            self.convergence_estimate * 100.0,
            self.mean_motion_magnitude,
            self.mean_disocclusion_fraction * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a flat (uniform value) HWC frame.
    fn make_frame(w: usize, h: usize, ch: usize, val: f32) -> Vec<f32> {
        vec![val; w * h * ch]
    }

    /// Helper: make a zero-motion field matching (w, h).
    fn zero_field(w: usize, h: usize) -> MotionVectorField {
        MotionVectorField::zero_motion(w, h)
    }

    #[test]
    fn test_motion_vector_magnitude() {
        let mv = MotionVector { dx: 3.0, dy: 4.0 };
        let mag = mv.magnitude();
        assert!((mag - 5.0).abs() < 1e-5, "expected 5.0, got {mag}");
    }

    #[test]
    fn test_motion_vector_is_static() {
        let mv_static = MotionVector { dx: 0.05, dy: 0.0 };
        let mv_moving = MotionVector { dx: 1.0, dy: 0.0 };
        assert!(mv_static.is_static(0.1), "should be static");
        assert!(!mv_moving.is_static(0.1), "should not be static");
    }

    #[test]
    fn test_motion_field_new_zeros() {
        let field = MotionVectorField::new(4, 4);
        assert_eq!(field.vectors.len(), 16);
        for mv in &field.vectors {
            assert_eq!(mv.dx, 0.0);
            assert_eq!(mv.dy, 0.0);
        }
    }

    #[test]
    fn test_motion_field_set_get() {
        let mut field = MotionVectorField::new(5, 5);
        let mv = MotionVector { dx: 2.5, dy: -1.0 };
        let ok = field.set(2, 3, mv);
        assert!(ok, "set should succeed");
        if let Some(got) = field.get(2, 3) {
            assert!((got.dx - 2.5).abs() < 1e-6);
            assert!((got.dy - (-1.0)).abs() < 1e-6);
        } else {
            panic!("expected Some, got None");
        }
        // Out-of-bounds
        assert!(!field.set(10, 10, mv), "OOB set should return false");
        assert!(field.get(10, 10).is_none(), "OOB get should return None");
    }

    #[test]
    fn test_accumulator_first_frame_returns_current() {
        let mut acc = TemporalAccumulator::new(2, 2, 3, TaaConfig::default());
        let frame = make_frame(2, 2, 3, 0.8);
        let field = zero_field(2, 2);
        let result = acc.accumulate(&frame, &field);
        match result {
            Ok(out) => {
                assert_eq!(out.len(), frame.len());
                for (a, b) in out.iter().zip(frame.iter()) {
                    assert!((a - b).abs() < 1e-6, "first frame should equal input");
                }
            }
            Err(e) => panic!("unexpected error: {e:?}"),
        }
        assert!(acc.has_history());
        assert_eq!(acc.frame_count, 1);
    }

    #[test]
    fn test_accumulator_second_frame_blends() {
        let cfg = TaaConfig {
            current_weight: 0.5,
            enable_neighborhood_clamp: false,
            ..TaaConfig::default()
        };
        let mut acc = TemporalAccumulator::new(2, 2, 1, cfg);
        let field = zero_field(2, 2);

        // First frame: all 0.0
        let frame0 = make_frame(2, 2, 1, 0.0);
        let _ = acc.accumulate(&frame0, &field);

        // Second frame: all 1.0
        // Expected blend: 0.5 * 1.0 + 0.5 * 0.0 = 0.5, but disocclusion might fire.
        // With large color_distance (1.0 > threshold 0.1), disocclusion_blend (0.5) is used.
        let frame1 = make_frame(2, 2, 1, 1.0);
        let result = acc.accumulate(&frame1, &field);
        match result {
            Ok(out) => {
                // Each pixel: disocclusion fires → weight = disocclusion_blend = 0.5
                // blend = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
                for &v in &out {
                    assert!((v - 0.5).abs() < 1e-5, "expected 0.5, got {v}");
                }
            }
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    #[test]
    fn test_accumulator_uniform_image_stable() {
        // Uniform image + zero motion: output should equal input (all same value).
        let cfg = TaaConfig {
            enable_neighborhood_clamp: false,
            ..TaaConfig::default()
        };
        let mut acc = TemporalAccumulator::new(4, 4, 3, cfg);
        let field = zero_field(4, 4);
        let frame = make_frame(4, 4, 3, 0.5);

        // First frame
        let _ = acc.accumulate(&frame, &field);
        // Second frame: history == current → color_distance = 0, normal weight applies.
        // blend = 0.1 * 0.5 + 0.9 * 0.5 = 0.5
        let result = acc.accumulate(&frame, &field);
        match result {
            Ok(out) => {
                for &v in &out {
                    assert!((v - 0.5).abs() < 1e-5, "uniform stable, got {v}");
                }
            }
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    #[test]
    fn test_accumulator_convergence_after_n_frames() {
        let cfg = TaaConfig {
            convergence_frames: 3,
            ..TaaConfig::default()
        };
        let mut acc = TemporalAccumulator::new(2, 2, 1, cfg);
        let field = zero_field(2, 2);
        let frame = make_frame(2, 2, 1, 0.5);

        assert!(!acc.is_converged());
        let _ = acc.accumulate(&frame, &field);
        let _ = acc.accumulate(&frame, &field);
        assert!(!acc.is_converged());
        let _ = acc.accumulate(&frame, &field);
        assert!(acc.is_converged(), "should converge after 3 frames");
    }

    #[test]
    fn test_accumulator_reset() {
        let mut acc = TemporalAccumulator::new(2, 2, 3, TaaConfig::default());
        let field = zero_field(2, 2);
        let frame = make_frame(2, 2, 3, 0.3);
        let _ = acc.accumulate(&frame, &field);
        assert!(acc.has_history());
        acc.reset();
        assert!(!acc.has_history());
        assert_eq!(acc.frame_count, 0);
    }

    #[test]
    fn test_accumulator_disocclusion_large_change() {
        // Very large color change should trigger disocclusion path.
        let cfg = TaaConfig {
            current_weight: 0.1,
            disocclusion_threshold: 0.05,
            disocclusion_blend: 0.9,
            enable_neighborhood_clamp: false,
            ..TaaConfig::default()
        };
        let mut acc = TemporalAccumulator::new(1, 1, 1, cfg);
        let field = zero_field(1, 1);

        let _ = acc.accumulate(&[0.0], &field);
        let result = acc.accumulate(&[1.0], &field);
        match result {
            Ok(out) => {
                // disocclusion fires: weight = 0.9
                // 0.9 * 1.0 + 0.1 * 0.0 = 0.9
                assert!(!out.is_empty());
                assert!((out[0] - 0.9).abs() < 1e-5, "expected 0.9, got {}", out[0]);
            }
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    #[test]
    fn test_neighborhood_clamp_edge_pixel() {
        // Ensure neighborhood_minmax works at corner pixel (0,0).
        let image: Vec<f32> = (0..9).map(|i| i as f32 * 0.1).collect(); // 3x3x1
        let (mn, mx) = neighborhood_minmax(&image, 3, 3, 1, 0, 0);
        assert!(!mn.is_empty());
        assert!(!mx.is_empty());
        assert!(mn[0] <= mx[0]);
    }

    #[test]
    fn test_bilinear_sample_integer_coords() {
        // At integer coords bilinear should match direct lookup.
        let image: Vec<f32> = (0..9).map(|i| i as f32).collect(); // 3x3, 1 channel
        let v = sample_bilinear(&image, 3, 3, 1, 1.0, 1.0);
        // pixel (1,1) in row-major = index 4
        assert_eq!(v.len(), 1);
        assert!((v[0] - 4.0).abs() < 1e-5, "expected 4.0, got {}", v[0]);
    }

    #[test]
    fn test_temporal_stats_convergence() {
        let cfg = TaaConfig {
            convergence_frames: 10,
            ..TaaConfig::default()
        };
        let mut acc = TemporalAccumulator::new(2, 2, 3, cfg);
        let field = zero_field(2, 2);
        let frame = make_frame(2, 2, 3, 0.5);

        let _ = acc.accumulate(&frame, &field);
        let _ = acc.accumulate(&frame, &field);

        let stats = TemporalStats::compute(&acc, &field);
        assert_eq!(stats.frame_count, 2);
        assert!((stats.convergence_estimate - 0.2).abs() < 1e-5);
        let summary = stats.format_summary();
        assert!(summary.contains("frames=2"));
    }

    #[test]
    fn test_taa_config_default() {
        let cfg = TaaConfig::default();
        assert!((cfg.current_weight - 0.1).abs() < 1e-6);
        assert!((cfg.disocclusion_threshold - 0.1).abs() < 1e-6);
        assert!(cfg.enable_neighborhood_clamp);
        assert!((cfg.disocclusion_blend - 0.5).abs() < 1e-6);
        assert_eq!(cfg.convergence_frames, 8);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let mut acc = TemporalAccumulator::new(4, 4, 3, TaaConfig::default());
        let field = zero_field(4, 4);
        // Wrong size frame (too small)
        let bad_frame = make_frame(2, 2, 3, 0.5); // 12 elements instead of 48
        let result = acc.accumulate(&bad_frame, &field);
        match result {
            Err(TaaError::DimensionMismatch {
                current,
                history: _,
            }) => {
                assert_eq!(current, 12);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_no_history_error() {
        let acc = TemporalAccumulator::new(2, 2, 3, TaaConfig::default());
        match acc.history() {
            Err(TaaError::NoHistory) => {}
            other => panic!("expected NoHistory, got {other:?}"),
        }
    }
}
