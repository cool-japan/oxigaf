//! Local neighbourhood statistics and variance clipping.
//!
//! [`local_color_stats`] is the reference formulation; [`RowWindowStats`] is
//! the row-sliding running-sum form used by
//! [`crate::temporal_aa::accumulate_taa`] for the same windows at `O(1)` per
//! pixel.

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

/// Row-sliding provider of the same local mean/σ that [`local_color_stats`]
/// computes, at `O(1)` per pixel instead of `O((2r+1)²)`.
///
/// [`local_color_stats`] re-sums the whole window from scratch for every
/// pixel, so a full-frame variance-clipping pass costs `O(w · h · r²)`. This
/// walks the image in row-major order and keeps, for each column, the running
/// sums of `v` and `v²` over the current *vertical* window; advancing one row
/// adds the row entering the window and subtracts the one leaving it. A
/// second running sum slides horizontally across the columns of one row.
/// Total cost is `O(w · h)` per frame with `O(w)` extra memory.
///
/// The window bounds are identical to [`local_color_stats`]: the half-open
/// ranges `[cy - r, cy + r + 1) ∩ [0, height)` and
/// `[cx - r, cx + r + 1) ∩ [0, width)`, i.e. clipped (not clamp-replicated)
/// at the borders, with `count` equal to the number of pixels actually inside.
///
/// Sums are accumulated in `f64` so that the incremental add/subtract does
/// not drift across a full image sweep; the published values are `f32`. The
/// result therefore agrees with [`local_color_stats`] to within `f32`
/// summation-order noise rather than bit-exactly.
pub(super) struct RowWindowStats {
    width: usize,
    height: usize,
    radius: usize,
    /// Per-column running sums of `v` over the current vertical window,
    /// `width * 3` entries laid out as `[x * 3 + channel]`.
    col_sum: Vec<f64>,
    /// Per-column running sums of `v²`, same layout as `col_sum`.
    col_sum_sq: Vec<f64>,
    /// Half-open vertical window `[y_min, y_max)` currently held in the
    /// column sums. Equal when nothing has been accumulated yet.
    y_min: usize,
    y_max: usize,
    /// Per-channel means for the row most recently produced (`width * 3`).
    mean_row: Vec<f32>,
    /// Per-channel population σ for that row (`width * 3`).
    std_row: Vec<f32>,
}

impl RowWindowStats {
    /// Allocate the running-sum scratch for a `width × height` image.
    pub(super) fn new(width: usize, height: usize, radius: usize) -> Self {
        let stride = width.saturating_mul(3);
        Self {
            width,
            height,
            radius,
            col_sum: vec![0.0_f64; stride],
            col_sum_sq: vec![0.0_f64; stride],
            y_min: 0,
            y_max: 0,
            mean_row: vec![0.0_f32; stride],
            std_row: vec![0.0_f32; stride],
        }
    }

    /// Add (`sign = 1.0`) or remove (`sign = -1.0`) image row `y` from the
    /// per-column sums.
    fn accumulate_row(&mut self, image: &[f32], y: usize, sign: f64) {
        let row_base = y * self.width * 3;
        for idx in 0..(self.width * 3) {
            let v = image.get(row_base + idx).copied().unwrap_or(0.0) as f64;
            if let Some(slot) = self.col_sum.get_mut(idx) {
                *slot += sign * v;
            }
            if let Some(slot) = self.col_sum_sq.get_mut(idx) {
                *slot += sign * v * v;
            }
        }
    }

    /// Move the vertical window to the one centred on row `cy`, then run the
    /// horizontal sliding pass that fills `mean_row` / `std_row`.
    ///
    /// Rows are expected in increasing order (the natural `accumulate_taa`
    /// order); any other order falls back to rebuilding the column sums,
    /// which is still correct, just not incremental.
    pub(super) fn advance_to_row(&mut self, image: &[f32], cy: usize) {
        if self.width == 0 || self.height == 0 {
            return;
        }

        let new_y_min = cy.saturating_sub(self.radius);
        let new_y_max = (cy + self.radius + 1).min(self.height);

        if new_y_min < self.y_min || new_y_max < self.y_max {
            // Non-monotonic call order: rebuild from scratch.
            self.col_sum.iter_mut().for_each(|v| *v = 0.0);
            self.col_sum_sq.iter_mut().for_each(|v| *v = 0.0);
            self.y_min = new_y_min;
            self.y_max = new_y_min;
        }
        for y in self.y_max..new_y_max {
            self.accumulate_row(image, y, 1.0);
        }
        for y in self.y_min..new_y_min {
            self.accumulate_row(image, y, -1.0);
        }
        self.y_min = new_y_min;
        self.y_max = new_y_max;

        self.fill_row();
    }

    /// Horizontal sliding pass over the current column sums.
    fn fill_row(&mut self) {
        let v_count = self.y_max - self.y_min;
        let mut run_sum = [0.0_f64; 3];
        let mut run_sum_sq = [0.0_f64; 3];
        let mut cur_min = 0_usize;
        let mut cur_max = 0_usize;

        for cx in 0..self.width {
            let x_min = cx.saturating_sub(self.radius);
            let x_max = (cx + self.radius + 1).min(self.width);

            while cur_max < x_max {
                for (c, (sum, sum_sq)) in run_sum.iter_mut().zip(run_sum_sq.iter_mut()).enumerate()
                {
                    let idx = cur_max * 3 + c;
                    *sum += self.col_sum.get(idx).copied().unwrap_or(0.0);
                    *sum_sq += self.col_sum_sq.get(idx).copied().unwrap_or(0.0);
                }
                cur_max += 1;
            }
            while cur_min < x_min {
                for (c, (sum, sum_sq)) in run_sum.iter_mut().zip(run_sum_sq.iter_mut()).enumerate()
                {
                    let idx = cur_min * 3 + c;
                    *sum -= self.col_sum.get(idx).copied().unwrap_or(0.0);
                    *sum_sq -= self.col_sum_sq.get(idx).copied().unwrap_or(0.0);
                }
                cur_min += 1;
            }

            let count = v_count * (x_max - x_min);
            let out_base = cx * 3;
            if count == 0 {
                for c in 0..3 {
                    if let Some(slot) = self.mean_row.get_mut(out_base + c) {
                        *slot = 0.0;
                    }
                    if let Some(slot) = self.std_row.get_mut(out_base + c) {
                        *slot = 0.0;
                    }
                }
                continue;
            }

            let n = count as f64;
            for c in 0..3 {
                let mean = run_sum[c] / n;
                let variance = (run_sum_sq[c] / n) - (mean * mean);
                if let Some(slot) = self.mean_row.get_mut(out_base + c) {
                    *slot = mean as f32;
                }
                if let Some(slot) = self.std_row.get_mut(out_base + c) {
                    *slot = variance.max(0.0).sqrt() as f32;
                }
            }
        }
    }

    /// `(mean, std)` for pixel `cx` of the row last passed to
    /// [`Self::advance_to_row`]. Out-of-range columns yield zeros.
    pub(super) fn pixel(&self, cx: usize) -> ([f32; 3], [f32; 3]) {
        let base = cx * 3;
        let mut mean = [0.0_f32; 3];
        let mut std = [0.0_f32; 3];
        for c in 0..3 {
            mean[c] = self.mean_row.get(base + c).copied().unwrap_or(0.0);
            std[c] = self.std_row.get(base + c).copied().unwrap_or(0.0);
        }
        (mean, std)
    }
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
pub(super) fn luminance(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// Linear interpolation between `a` and `b` by factor `t` ∈ [0, 1].
#[inline]
pub(super) fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_aa::{accumulate_taa, TaaConfig, TaaHistory};

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

    // ── RowWindowStats (sliding local mean/σ) ─────────────────────────────────

    /// Naive `f64` window reference with exactly the bounds
    /// `local_color_stats` documents. Used as the equivalence oracle for
    /// [`RowWindowStats`], because the public `f32` implementation loses
    /// precision in `E[v²] - E[v]²` on near-uniform patches and would make a
    /// tight comparison meaningless.
    fn local_color_stats_f64_reference(
        image: &[f32],
        width: usize,
        height: usize,
        cx: usize,
        cy: usize,
        radius: usize,
    ) -> ([f64; 3], [f64; 3]) {
        let y_min = cy.saturating_sub(radius);
        let y_max = (cy + radius + 1).min(height);
        let x_min = cx.saturating_sub(radius);
        let x_max = (cx + radius + 1).min(width);

        let mut sum = [0.0_f64; 3];
        let mut sum_sq = [0.0_f64; 3];
        let mut count = 0_usize;
        for ny in y_min..y_max {
            for nx in x_min..x_max {
                let base = (ny * width + nx) * 3;
                for (c, (s, sq)) in sum.iter_mut().zip(sum_sq.iter_mut()).enumerate() {
                    let v = image.get(base + c).copied().unwrap_or(0.0) as f64;
                    *s += v;
                    *sq += v * v;
                }
                count += 1;
            }
        }
        if count == 0 {
            return ([0.0; 3], [0.0; 3]);
        }
        let n = count as f64;
        let mut mean = [0.0_f64; 3];
        let mut std = [0.0_f64; 3];
        for c in 0..3 {
            mean[c] = sum[c] / n;
            std[c] = ((sum_sq[c] / n) - mean[c] * mean[c]).max(0.0).sqrt();
        }
        (mean, std)
    }

    /// Deterministic pseudo-random test image in `[0, 1]`.
    fn pseudo_image(width: usize, height: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..(width * height * 3))
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / 16_777_216.0
            })
            .collect()
    }

    #[test]
    fn test_row_window_stats_matches_naive_f64_reference() {
        // Regression (F264): the sliding running-sum optimisation must
        // reproduce the exact windows of `local_color_stats`, including the
        // clipped (not clamp-replicated) borders. Bounds bookkeeping is where
        // sliding windows break, so sweep degenerate dimensions and radii
        // larger than the image.
        for &(w, h) in &[
            (1_usize, 1_usize),
            (1, 5),
            (5, 1),
            (2, 2),
            (7, 4),
            (4, 7),
            (9, 9),
        ] {
            let image = pseudo_image(w, h, 0x5EED_1234 ^ (w as u64) << 8 ^ h as u64);
            for radius in [0_usize, 1, 2, 3, 12] {
                let mut sliding = RowWindowStats::new(w, h, radius);
                for cy in 0..h {
                    sliding.advance_to_row(&image, cy);
                    for cx in 0..w {
                        let (mean, std) = sliding.pixel(cx);
                        let (ref_mean, ref_std) =
                            local_color_stats_f64_reference(&image, w, h, cx, cy, radius);
                        for c in 0..3 {
                            assert!(
                                (mean[c] as f64 - ref_mean[c]).abs() < 1e-6,
                                "mean drift at {w}x{h} r={radius} ({cx},{cy}) c={c}: \
                                 {} vs {}",
                                mean[c],
                                ref_mean[c]
                            );
                            assert!(
                                (std[c] as f64 - ref_std[c]).abs() < 1e-6,
                                "std drift at {w}x{h} r={radius} ({cx},{cy}) c={c}: \
                                 {} vs {}",
                                std[c],
                                ref_std[c]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_row_window_stats_matches_public_local_color_stats() {
        // Looser tolerance than the f64 oracle above on purpose: the public
        // `local_color_stats` accumulates in f32, so `E[v²] - E[v]²` there
        // carries cancellation error the f64 sliding path does not.
        let (w, h) = (11_usize, 6_usize);
        let image = pseudo_image(w, h, 0xABCD_EF01);
        for radius in [1_usize, 2] {
            let mut sliding = RowWindowStats::new(w, h, radius);
            for cy in 0..h {
                sliding.advance_to_row(&image, cy);
                for cx in 0..w {
                    let (mean, std) = sliding.pixel(cx);
                    let (ref_mean, ref_std) = local_color_stats(&image, w, h, cx, cy, radius);
                    for c in 0..3 {
                        assert!(
                            (mean[c] - ref_mean[c]).abs() < 1e-5,
                            "mean mismatch r={radius} ({cx},{cy}) c={c}"
                        );
                        assert!(
                            (std[c] - ref_std[c]).abs() < 1e-4,
                            "std mismatch r={radius} ({cx},{cy}) c={c}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_row_window_stats_rebuilds_on_backwards_row() {
        // The incremental path assumes increasing rows; a backwards jump must
        // fall back to a rebuild rather than reporting a stale window.
        let (w, h) = (5_usize, 5_usize);
        let image = pseudo_image(w, h, 0x1357_9BDF);
        let mut sliding = RowWindowStats::new(w, h, 1);
        for cy in 0..h {
            sliding.advance_to_row(&image, cy);
        }
        sliding.advance_to_row(&image, 1);
        for cx in 0..w {
            let (mean, std) = sliding.pixel(cx);
            let (ref_mean, ref_std) = local_color_stats_f64_reference(&image, w, h, cx, 1, 1);
            for c in 0..3 {
                assert!(
                    (mean[c] as f64 - ref_mean[c]).abs() < 1e-6,
                    "stale mean after backwards row at cx={cx} c={c}"
                );
                assert!(
                    (std[c] as f64 - ref_std[c]).abs() < 1e-6,
                    "stale std after backwards row at cx={cx} c={c}"
                );
            }
        }
    }

    #[test]
    fn test_accumulate_taa_clipping_matches_per_pixel_reference() {
        // End-to-end: swapping `local_color_stats` for the sliding window
        // inside `accumulate_taa` must not change the blended image.
        let (w, h) = (9_usize, 5_usize);
        let frame0 = pseudo_image(w, h, 0x2468_ACE0);
        let frame1 = pseudo_image(w, h, 0x1122_3344);
        let config = TaaConfig {
            variance_clipping: true,
            clip_window_radius: 2,
            sharpen_strength: 0.0,
            blend_factor: 0.25,
            ..TaaConfig::default()
        };

        let mut history = TaaHistory::new(w, h);
        accumulate_taa(&frame0, &mut history, &config).expect("bootstrap");
        let got = accumulate_taa(&frame1, &mut history, &config).expect("blend");

        // Reference: the original per-pixel formulation.
        let mut expected = vec![0.0_f32; w * h * 3];
        for py in 0..h {
            for px in 0..w {
                let base = (py * w + px) * 3;
                let cur = [frame1[base], frame1[base + 1], frame1[base + 2]];
                let hist = [frame0[base], frame0[base + 1], frame0[base + 2]];
                let (mean, std) =
                    local_color_stats(&frame1, w, h, px, py, config.clip_window_radius);
                let clipped = clip_to_variance(hist, mean, std, 1.0);
                for c in 0..3 {
                    expected[base + c] =
                        config.blend_factor * cur[c] + (1.0 - config.blend_factor) * clipped[c];
                }
            }
        }

        for (i, (&a, &b)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "pixel component {i}: sliding {a} vs per-pixel reference {b}"
            );
        }
    }
}
