//! Halton low-discrepancy sequence and per-frame subpixel jitter offsets.

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

#[cfg(test)]
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
}
