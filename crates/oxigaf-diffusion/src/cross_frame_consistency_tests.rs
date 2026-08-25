//! Test suite for [`crate::cross_frame_consistency`], split into its own
//! file (via a `#[path]`-attributed `mod tests;` declaration) to keep the
//! production-code file under the COOLJAPAN 2000-line policy limit.

use super::*;

// ── helpers ───────────────────────────────────────────────────────────────

fn uniform_frame(w: usize, h: usize, r: f32, g: f32, b: f32) -> Frame {
    let pixels: Vec<f32> = (0..w * h).flat_map(|_| [r, g, b]).collect();
    Frame {
        pixels,
        width: w,
        height: h,
    }
}

fn checkerboard_frame(w: usize, h: usize) -> Frame {
    let mut pixels = Vec::with_capacity(w * h * 3);
    for row in 0..h {
        for col in 0..w {
            let v = if (row + col) % 2 == 0 {
                0.0_f32
            } else {
                1.0_f32
            };
            pixels.push(v);
            pixels.push(v);
            pixels.push(v);
        }
    }
    Frame {
        pixels,
        width: w,
        height: h,
    }
}

fn gradient_frame(w: usize, h: usize) -> Frame {
    let mut pixels = Vec::with_capacity(w * h * 3);
    for row in 0..h {
        for col in 0..w {
            let v = col as f32 / w.max(1) as f32;
            let u = row as f32 / h.max(1) as f32;
            pixels.push(v);
            pixels.push(u);
            pixels.push(0.5_f32);
        }
    }
    Frame {
        pixels,
        width: w,
        height: h,
    }
}

/// A frame whose luminance varies only with column (Iy = 0 exactly),
/// avoiding the aperture-problem ambiguity a 2D gradient introduces for
/// optical flow: the per-pixel Horn-Schunck constraint `Ix*u + It = 0`
/// alone determines `u`.
fn x_ramp_frame(w: usize, h: usize) -> Frame {
    let mut pixels = Vec::with_capacity(w * h * 3);
    for _row in 0..h {
        for col in 0..w {
            let v = col as f32 / w.max(1) as f32;
            pixels.push(v);
            pixels.push(v);
            pixels.push(v);
        }
    }
    Frame {
        pixels,
        width: w,
        height: h,
    }
}

/// Shift a frame's content right by `shift` pixels, clamping at the
/// left edge (column 0 repeats).
fn shift_right_frame(f: &Frame, shift: usize) -> Frame {
    let mut pixels = vec![0.0_f32; f.pixels.len()];
    for row in 0..f.height {
        for col in 0..f.width {
            let src_col = col.saturating_sub(shift);
            let src_idx = (row * f.width + src_col) * 3;
            let dst_idx = (row * f.width + col) * 3;
            pixels[dst_idx] = f.pixels[src_idx];
            pixels[dst_idx + 1] = f.pixels[src_idx + 1];
            pixels[dst_idx + 2] = f.pixels[src_idx + 2];
        }
    }
    Frame {
        pixels,
        width: f.width,
        height: f.height,
    }
}

// ── Frame construction ────────────────────────────────────────────────────

#[test]
fn test_frame_new_all_zeros() {
    let f = Frame::new(4, 4);
    assert!(f.pixels.iter().all(|&v| v == 0.0));
}

#[test]
fn test_frame_new_correct_dimensions() {
    let f = Frame::new(8, 6);
    assert_eq!(f.width, 8);
    assert_eq!(f.height, 6);
    assert_eq!(f.pixels.len(), 8 * 6 * 3);
}

#[test]
fn test_frame_n_pixels() {
    let f = Frame::new(10, 5);
    assert_eq!(f.n_pixels(), 50);
}

#[test]
fn test_frame_from_pixels_ok() -> Result<(), ConsistencyError> {
    let pixels = vec![0.5_f32; 4 * 4 * 3];
    let f = Frame::from_pixels(pixels, 4, 4)?;
    assert_eq!(f.width, 4);
    assert_eq!(f.height, 4);
    Ok(())
}

#[test]
fn test_frame_from_pixels_wrong_size_error() {
    let pixels = vec![0.0_f32; 10]; // wrong length
    assert!(matches!(
        Frame::from_pixels(pixels, 4, 4),
        Err(ConsistencyError::InvalidConfig(_))
    ));
}

// ── pixel_at ─────────────────────────────────────────────────────────────

#[test]
fn test_pixel_at_correct_indexing() -> Result<(), ConsistencyError> {
    let f = uniform_frame(4, 4, 0.1, 0.2, 0.3);
    let px = f.pixel_at(2, 1)?;
    assert!((px[0] - 0.1).abs() < 1e-6);
    assert!((px[1] - 0.2).abs() < 1e-6);
    assert!((px[2] - 0.3).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_pixel_at_out_of_bounds_error() {
    let f = Frame::new(4, 4);
    assert!(matches!(
        f.pixel_at(10, 0),
        Err(ConsistencyError::InvalidConfig(_))
    ));
    assert!(matches!(
        f.pixel_at(0, 10),
        Err(ConsistencyError::InvalidConfig(_))
    ));
}

// ── mean_brightness ───────────────────────────────────────────────────────

#[test]
fn test_mean_brightness_zero_frame() {
    let f = Frame::new(8, 8);
    assert_eq!(f.mean_brightness(), 0.0);
}

#[test]
fn test_mean_brightness_uniform() {
    let f = uniform_frame(4, 4, 0.5, 0.5, 0.5);
    assert!((f.mean_brightness() - 0.5).abs() < 1e-6);
}

#[test]
fn test_mean_brightness_half_white() {
    // Half pixels at 1.0, half at 0.0 (RGB all same)
    let mut pixels = vec![0.0_f32; 4 * 4 * 3];
    for p in pixels.iter_mut().take(8 * 3) {
        *p = 1.0;
    }
    let f = Frame {
        pixels,
        width: 4,
        height: 4,
    };
    let mb = f.mean_brightness();
    assert!((mb - 0.5).abs() < 1e-5, "got {mb}");
}

// ── variance ─────────────────────────────────────────────────────────────

#[test]
fn test_variance_constant_frame_zero() {
    let f = uniform_frame(8, 8, 0.5, 0.5, 0.5);
    assert!(f.variance().abs() < 1e-8);
}

#[test]
fn test_variance_varying_frame_positive() {
    let f = checkerboard_frame(8, 8);
    assert!(f.variance() > 0.0);
}

// ── cfc_to_grayscale ──────────────────────────────────────────────────────

#[test]
fn test_to_grayscale_white() {
    let f = uniform_frame(4, 4, 1.0, 1.0, 1.0);
    let g = cfc_to_grayscale(&f);
    assert_eq!(g.len(), 16);
    for &v in &g {
        assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
    }
}

#[test]
fn test_to_grayscale_black() {
    let f = Frame::new(4, 4);
    let g = cfc_to_grayscale(&f);
    for &v in &g {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_to_grayscale_length() {
    let f = Frame::new(6, 7);
    assert_eq!(cfc_to_grayscale(&f).len(), 6 * 7);
}

#[test]
fn test_to_grayscale_known_value() {
    // Pure red pixel → 0.2126
    let f = uniform_frame(1, 1, 1.0, 0.0, 0.0);
    let g = cfc_to_grayscale(&f);
    assert!((g[0] - 0.2126).abs() < 1e-5, "got {}", g[0]);
}

// ── cfc_bilinear_sample ───────────────────────────────────────────────────

#[test]
fn test_bilinear_exact_pixel() {
    let f = uniform_frame(4, 4, 0.3, 0.6, 0.9);
    let rgb = cfc_bilinear_sample(&f, 2.0, 1.0);
    assert!((rgb[0] - 0.3).abs() < 1e-5);
    assert!((rgb[1] - 0.6).abs() < 1e-5);
    assert!((rgb[2] - 0.9).abs() < 1e-5);
}

#[test]
fn test_bilinear_clamp_out_of_bounds() {
    let f = uniform_frame(4, 4, 0.7, 0.8, 0.9);
    // Coordinates way outside → clamped to border value
    let rgb = cfc_bilinear_sample(&f, -5.0, 100.0);
    assert!((rgb[0] - 0.7).abs() < 1e-5);
}

#[test]
fn test_bilinear_midpoint_uniform() {
    // A uniform frame → any fractional coord returns the same color
    let f = uniform_frame(8, 8, 0.4, 0.5, 0.6);
    let rgb = cfc_bilinear_sample(&f, 3.7, 2.3);
    assert!((rgb[0] - 0.4).abs() < 1e-5);
    assert!((rgb[1] - 0.5).abs() < 1e-5);
    assert!((rgb[2] - 0.6).abs() < 1e-5);
}

#[test]
fn test_bilinear_sample_zero_dimension_frame_no_panic() {
    let f = Frame::new(0, 0);
    assert_eq!(cfc_bilinear_sample(&f, 5.0, 5.0), [0.0, 0.0, 0.0]);

    let f2 = Frame::new(0, 4);
    assert_eq!(cfc_bilinear_sample(&f2, 1.0, 1.0), [0.0, 0.0, 0.0]);

    let f3 = Frame::new(4, 0);
    assert_eq!(cfc_bilinear_sample(&f3, 1.0, 1.0), [0.0, 0.0, 0.0]);
}

// ── cfc_compute_flow ──────────────────────────────────────────────────────

#[test]
fn test_flow_identical_frames_near_zero() -> Result<(), ConsistencyError> {
    let f = checkerboard_frame(16, 16);
    let config = FlowConfig {
        n_iterations: 5,
        ..Default::default()
    };
    let (fx, fy) = cfc_compute_flow(&f, &f, &config)?;
    let max_u: f32 = fx.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let max_v: f32 = fy.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_u.abs() < 5.0,
        "large flow_x for identical frames: {max_u}"
    );
    assert!(
        max_v.abs() < 5.0,
        "large flow_y for identical frames: {max_v}"
    );
    Ok(())
}

#[test]
fn test_flow_dimension_mismatch_error() {
    let a = Frame::new(8, 8);
    let b = Frame::new(4, 4);
    assert!(matches!(
        cfc_compute_flow(&a, &b, &FlowConfig::default()),
        Err(ConsistencyError::FrameDimensionMismatch { .. })
    ));
}

#[test]
fn test_flow_output_length() -> Result<(), ConsistencyError> {
    let f = Frame::new(8, 8);
    let config = FlowConfig {
        n_iterations: 2,
        ..Default::default()
    };
    let (fx, fy) = cfc_compute_flow(&f, &f, &config)?;
    assert_eq!(fx.len(), 64);
    assert_eq!(fy.len(), 64);
    Ok(())
}

// ── cfc_warp_frame ────────────────────────────────────────────────────────

#[test]
fn test_warp_zero_flow_same_frame() -> Result<(), ConsistencyError> {
    let f = gradient_frame(8, 8);
    let flow = vec![0.0_f32; 64];
    let warped = cfc_warp_frame(&f, &flow, &flow)?;
    for (a, b) in f.pixels.iter().zip(warped.pixels.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "zero-flow warp changed pixel: {a} vs {b}"
        );
    }
    Ok(())
}

#[test]
fn test_warp_wrong_flow_length_error() {
    let f = Frame::new(4, 4);
    let bad_flow = vec![0.0_f32; 5]; // wrong length
    assert!(matches!(
        cfc_warp_frame(&f, &bad_flow, &bad_flow),
        Err(ConsistencyError::InvalidConfig(_))
    ));
}

#[test]
fn test_optical_flow_warp_uses_negated_flow_not_raw() -> Result<(), ConsistencyError> {
    // cfc_compute_flow estimates the FORWARD (A->B) flow; cfc_warp_frame
    // performs a backward warp and needs the negated flow. Using an
    // x-only ramp avoids the aperture-problem ambiguity a 2D gradient
    // introduces, and a small alpha (relative to the ramp's spatial
    // gradient, ~1/24) lets the data term dominate so a translating
    // scene actually recovers u ~= 1.
    let a = x_ramp_frame(24, 24);
    let b = shift_right_frame(&a, 1);
    let config = FlowConfig {
        alpha: 0.005,
        n_iterations: 100,
        scale: 1,
    };

    let (fx, fy) = cfc_compute_flow(&a, &b, &config)?;
    let raw_warped = cfc_warp_frame(&a, &fx, &fy)?;
    let raw_error = cfc_mae(&raw_warped, &b)?;

    let (neg_fx, neg_fy) = negate_flow(&fx, &fy);
    let negated_warped = cfc_warp_frame(&a, &neg_fx, &neg_fy)?;
    let negated_error = cfc_mae(&negated_warped, &b)?;

    assert!(
        negated_error < raw_error,
        "negated (fixed) pairing should reduce warp error relative to \
         the raw (buggy) pairing: negated={negated_error} raw={raw_error}"
    );
    Ok(())
}

#[test]
fn test_frame_pair_consistency_beats_raw_pairing_baseline() -> Result<(), ConsistencyError> {
    let a = x_ramp_frame(24, 24);
    let b = shift_right_frame(&a, 1);
    let config = FlowConfig {
        alpha: 0.005,
        n_iterations: 100,
        scale: 1,
    };

    // Baseline: what the OLD (buggy) code computed — raw, un-negated flow.
    let (fx, fy) = cfc_compute_flow(&a, &b, &config)?;
    let raw_error = cfc_mae(&cfc_warp_frame(&a, &fx, &fy)?, &b)?;

    let paired = cfc_frame_pair_consistency(&a, &b, &config)?;

    assert!(
        paired.mean_warp_error < raw_error,
        "cfc_frame_pair_consistency should use the negated (fixed) flow \
         pairing, beating the raw (buggy) pairing baseline: fixed={} raw={}",
        paired.mean_warp_error,
        raw_error
    );
    Ok(())
}

#[test]
fn test_consistency_loss_warp_term_beats_raw_pairing_baseline() -> Result<(), ConsistencyError> {
    let a = x_ramp_frame(24, 24);
    let b = shift_right_frame(&a, 1);
    let flow_cfg = FlowConfig {
        alpha: 0.005,
        n_iterations: 100,
        scale: 1,
    };

    let (fx, fy) = cfc_compute_flow(&a, &b, &flow_cfg)?;
    let raw_error = cfc_mae(&cfc_warp_frame(&a, &fx, &fy)?, &b)?;

    let cfg = ConsistencyLossConfig {
        l1_weight: 0.0,
        warp_weight: 1.0,
        temporal_smooth_weight: 0.0,
        use_optical_flow: true,
        psnr_weight: 0.0,
    };
    let loss = cfc_consistency_loss(&[a, b], &cfg, &flow_cfg)?;

    assert!(
        loss.warp_term < raw_error,
        "cfc_consistency_loss's warp term should use the negated (fixed) \
         flow pairing, beating the raw (buggy) pairing baseline: fixed={} raw={}",
        loss.warp_term,
        raw_error
    );
    Ok(())
}

// ── cfc_psnr ─────────────────────────────────────────────────────────────

#[test]
fn test_psnr_identical_frames_infinity() -> Result<(), ConsistencyError> {
    let f = uniform_frame(8, 8, 0.5, 0.5, 0.5);
    let psnr = cfc_psnr(&f, &f)?;
    assert_eq!(psnr, f32::INFINITY);
    Ok(())
}

#[test]
fn test_psnr_different_frames_finite() -> Result<(), ConsistencyError> {
    let a = Frame::new(8, 8);
    let b = uniform_frame(8, 8, 1.0, 1.0, 1.0);
    let psnr = cfc_psnr(&a, &b)?;
    assert!(psnr.is_finite());
    // MSE = 1.0, PSNR = 10*log10(1/1) = 0 dB
    assert!((psnr - 0.0).abs() < 1e-4, "got {psnr}");
    Ok(())
}

#[test]
fn test_psnr_dimension_mismatch() {
    let a = Frame::new(4, 4);
    let b = Frame::new(8, 8);
    assert!(matches!(
        cfc_psnr(&a, &b),
        Err(ConsistencyError::FrameDimensionMismatch { .. })
    ));
}

// ── cfc_ssim ─────────────────────────────────────────────────────────────

#[test]
fn test_ssim_identical_near_one() -> Result<(), ConsistencyError> {
    let f = checkerboard_frame(16, 16);
    let s = cfc_ssim(&f, &f)?;
    assert!(
        s > 0.99,
        "SSIM of identical frames should be near 1.0, got {s}"
    );
    Ok(())
}

#[test]
fn test_ssim_very_different_less_than_identical() -> Result<(), ConsistencyError> {
    let a = Frame::new(16, 16);
    let b = uniform_frame(16, 16, 1.0, 1.0, 1.0);
    let sa = cfc_ssim(&a, &a)?;
    let sab = cfc_ssim(&a, &b)?;
    assert!(
        sab < sa,
        "SSIM of different frames should be lower: {sab} vs {sa}"
    );
    Ok(())
}

#[test]
fn test_ssim_dimension_mismatch() {
    let a = Frame::new(4, 4);
    let b = Frame::new(8, 8);
    assert!(matches!(
        cfc_ssim(&a, &b),
        Err(ConsistencyError::FrameDimensionMismatch { .. })
    ));
}

#[test]
fn test_ssim_near_saturated_values_stays_in_range() -> Result<(), ConsistencyError> {
    // Near-saturated luminance (~0.999) is where the naive one-pass
    // variance E[x^2] - E[x]^2 can go slightly negative in f32 due to
    // catastrophic cancellation, which used to be able to flip the
    // SSIM denominator's sign and produce values outside a sane range.
    let w = 8;
    let h = 8;
    let mut pixels_a = Vec::with_capacity(w * h * 3);
    let mut pixels_b = Vec::with_capacity(w * h * 3);
    for i in 0..(w * h) {
        let jitter = if i % 2 == 0 { 0.0005 } else { -0.0005 };
        let va = (0.999_f32 + jitter).clamp(0.0, 1.0);
        let vb = (0.998_f32 - jitter).clamp(0.0, 1.0);
        pixels_a.extend_from_slice(&[va, va, va]);
        pixels_b.extend_from_slice(&[vb, vb, vb]);
    }
    let a = Frame::from_pixels(pixels_a, w, h)?;
    let b = Frame::from_pixels(pixels_b, w, h)?;
    let s = cfc_ssim(&a, &b)?;
    assert!(
        s.is_finite() && (-1.0..=1.0 + 1e-3).contains(&s),
        "SSIM should stay in a sane range even for near-saturated input, got {s}"
    );
    Ok(())
}

// ── cfc_mae ───────────────────────────────────────────────────────────────

#[test]
fn test_mae_identical_zero() -> Result<(), ConsistencyError> {
    let f = checkerboard_frame(8, 8);
    assert_eq!(cfc_mae(&f, &f)?, 0.0);
    Ok(())
}

#[test]
fn test_mae_known_difference() -> Result<(), ConsistencyError> {
    let a = Frame::new(4, 4); // all 0
    let b = uniform_frame(4, 4, 0.5, 0.5, 0.5); // all 0.5
    let mae = cfc_mae(&a, &b)?;
    assert!((mae - 0.5).abs() < 1e-6, "got {mae}");
    Ok(())
}

#[test]
fn test_mae_dimension_mismatch() {
    let a = Frame::new(4, 4);
    let b = Frame::new(8, 8);
    assert!(matches!(
        cfc_mae(&a, &b),
        Err(ConsistencyError::FrameDimensionMismatch { .. })
    ));
}

// ── cfc_rmse ──────────────────────────────────────────────────────────────

#[test]
fn test_rmse_identical_zero() -> Result<(), ConsistencyError> {
    let f = gradient_frame(8, 8);
    assert_eq!(cfc_rmse(&f, &f)?, 0.0);
    Ok(())
}

#[test]
fn test_rmse_known_value() -> Result<(), ConsistencyError> {
    let a = Frame::new(4, 4); // all 0
    let b = uniform_frame(4, 4, 1.0, 1.0, 1.0); // all 1
    let rmse = cfc_rmse(&a, &b)?;
    assert!((rmse - 1.0).abs() < 1e-6, "got {rmse}");
    Ok(())
}

// ── cfc_frame_difference ──────────────────────────────────────────────────

#[test]
fn test_frame_difference_identical_zero_warp_error() -> Result<(), ConsistencyError> {
    let f = checkerboard_frame(8, 8);
    let c = cfc_frame_difference(&f, &f)?;
    assert_eq!(c.mean_warp_error, 0.0);
    assert_eq!(c.psnr, f32::INFINITY);
    Ok(())
}

#[test]
fn test_frame_difference_dimension_mismatch() {
    let a = Frame::new(4, 4);
    let b = Frame::new(8, 8);
    assert!(matches!(
        cfc_frame_difference(&a, &b),
        Err(ConsistencyError::FrameDimensionMismatch { .. })
    ));
}

#[test]
fn test_frame_difference_ssim_range() -> Result<(), ConsistencyError> {
    let a = uniform_frame(16, 16, 0.2, 0.3, 0.4);
    let b = uniform_frame(16, 16, 0.5, 0.6, 0.7);
    let c = cfc_frame_difference(&a, &b)?;
    assert!(
        c.ssim >= -1.0 && c.ssim <= 1.0 + 1e-5,
        "ssim out of range: {}",
        c.ssim
    );
    Ok(())
}

// ── cfc_frame_pair_consistency ────────────────────────────────────────────

#[test]
fn test_frame_pair_consistency_identical_low_error() -> Result<(), ConsistencyError> {
    let f = gradient_frame(16, 16);
    let config = FlowConfig {
        n_iterations: 3,
        ..Default::default()
    };
    let c = cfc_frame_pair_consistency(&f, &f, &config)?;
    assert!(
        c.mean_warp_error < 0.05,
        "warp error too large for identical frames: {}",
        c.mean_warp_error
    );
    Ok(())
}

#[test]
fn test_frame_pair_consistency_psnr_finite_or_inf() -> Result<(), ConsistencyError> {
    let a = uniform_frame(16, 16, 0.3, 0.4, 0.5);
    let b = uniform_frame(16, 16, 0.3, 0.4, 0.5);
    let config = FlowConfig {
        n_iterations: 2,
        ..Default::default()
    };
    let c = cfc_frame_pair_consistency(&a, &b, &config)?;
    assert!(c.psnr.is_infinite() || c.psnr > 20.0, "psnr={}", c.psnr);
    Ok(())
}

// ── cfc_sequence_consistency ──────────────────────────────────────────────

#[test]
fn test_sequence_two_frames_one_pair() -> Result<(), ConsistencyError> {
    let fa = uniform_frame(8, 8, 0.3, 0.3, 0.3);
    let fb = uniform_frame(8, 8, 0.4, 0.4, 0.4);
    let report = cfc_sequence_consistency(&[fa, fb], false, &FlowConfig::default())?;
    assert_eq!(report.per_pair_psnr.len(), 1);
    assert_eq!(report.per_pair_ssim.len(), 1);
    Ok(())
}

#[test]
fn test_sequence_empty_error() {
    let result = cfc_sequence_consistency(&[], false, &FlowConfig::default());
    assert!(matches!(result, Err(ConsistencyError::EmptySequence)));
}

#[test]
fn test_sequence_single_frame_error() {
    let f = Frame::new(8, 8);
    let result = cfc_sequence_consistency(&[f], false, &FlowConfig::default());
    assert!(matches!(result, Err(ConsistencyError::TooShort { .. })));
}

#[test]
fn test_sequence_worst_best_valid_indices() -> Result<(), ConsistencyError> {
    let frames: Vec<Frame> = (0..5)
        .map(|i| uniform_frame(8, 8, i as f32 * 0.1, 0.0, 0.0))
        .collect();
    let report = cfc_sequence_consistency(&frames, false, &FlowConfig::default())?;
    assert!(report.worst_frame_pair < frames.len() - 1);
    assert!(report.best_frame_pair < frames.len() - 1);
    Ok(())
}

#[test]
fn test_sequence_n_frames_matches() -> Result<(), ConsistencyError> {
    let frames: Vec<Frame> = (0..4).map(|_| Frame::new(8, 8)).collect();
    let report = cfc_sequence_consistency(&frames, false, &FlowConfig::default())?;
    assert_eq!(report.n_frames, 4);
    assert_eq!(report.per_pair_psnr.len(), 3);
    Ok(())
}

#[test]
fn test_sequence_per_pair_psnr_length() -> Result<(), ConsistencyError> {
    let frames: Vec<Frame> = (0..6).map(|_| gradient_frame(8, 8)).collect();
    let report = cfc_sequence_consistency(&frames, false, &FlowConfig::default())?;
    assert_eq!(report.per_pair_psnr.len(), report.n_frames - 1);
    Ok(())
}

// ── psnr_trend ────────────────────────────────────────────────────────────

#[test]
fn test_psnr_trend_increasing_quality() -> Result<(), ConsistencyError> {
    // Build a sequence where consecutive-pair diffs shrink, so PSNR rises.
    // Frame values: 0.0, 0.4, 0.7, 0.85, 0.925, 1.0  (each step halves residual)
    // Differences: 0.4, 0.3, 0.15, 0.075, 0.075 — generally decreasing
    // → PSNR trend should be positive.
    let values = [0.0_f32, 0.4, 0.7, 0.85, 0.93, 1.0];
    let frames: Vec<Frame> = values
        .iter()
        .map(|&v| uniform_frame(8, 8, v, 0.0, 0.0))
        .collect();
    let report = cfc_sequence_consistency(&frames, false, &FlowConfig::default())?;
    // Consecutive diffs: 0.4, 0.3, 0.15, 0.08, 0.07 — strictly shrinking
    // PSNR increases monotonically, so trend slope must be positive.
    assert!(
        report.psnr_trend > 0.0,
        "expected positive psnr_trend, got {}",
        report.psnr_trend
    );
    Ok(())
}

// ── cfc_consistency_loss ──────────────────────────────────────────────────

#[test]
fn test_consistency_loss_constant_sequence_l1_zero() -> Result<(), ConsistencyError> {
    let frames: Vec<Frame> = (0..4).map(|_| uniform_frame(8, 8, 0.5, 0.5, 0.5)).collect();
    let cfg = ConsistencyLossConfig {
        use_optical_flow: false,
        ..Default::default()
    };
    let loss = cfc_consistency_loss(&frames, &cfg, &FlowConfig::default())?;
    assert!(
        loss.l1_term < 1e-6,
        "l1_term should be 0 for constant sequence: {}",
        loss.l1_term
    );
    Ok(())
}

#[test]
fn test_consistency_loss_single_frame_error() {
    let f = Frame::new(8, 8);
    let cfg = ConsistencyLossConfig::default();
    let result = cfc_consistency_loss(&[f], &cfg, &FlowConfig::default());
    assert!(matches!(result, Err(ConsistencyError::TooShort { .. })));
}

#[test]
fn test_consistency_loss_two_frames_smooth_term_error() {
    // With smooth term weight > 0 and only 2 frames → TooShort
    let fa = Frame::new(8, 8);
    let fb = Frame::new(8, 8);
    let cfg = ConsistencyLossConfig {
        temporal_smooth_weight: 0.1,
        ..Default::default()
    };
    let result = cfc_consistency_loss(&[fa, fb], &cfg, &FlowConfig::default());
    assert!(matches!(result, Err(ConsistencyError::TooShort { .. })));
}

#[test]
fn test_consistency_loss_all_l1_weight() -> Result<(), ConsistencyError> {
    let fa = uniform_frame(8, 8, 0.0, 0.0, 0.0);
    let fb = uniform_frame(8, 8, 1.0, 1.0, 1.0);
    let fc = uniform_frame(8, 8, 0.5, 0.5, 0.5);
    let cfg = ConsistencyLossConfig {
        l1_weight: 1.0,
        warp_weight: 0.0,
        temporal_smooth_weight: 0.0,
        ..Default::default()
    };
    let loss = cfc_consistency_loss(&[fa, fb, fc], &cfg, &FlowConfig::default())?;
    assert!(loss.warp_term == 0.0, "warp_term should be 0 when weight=0");
    assert!(
        loss.smooth_term == 0.0,
        "smooth_term should be 0 when weight=0"
    );
    assert!(loss.total > 0.0);
    Ok(())
}

#[test]
fn test_consistency_loss_warp_weight_applied() -> Result<(), ConsistencyError> {
    // use_optical_flow must be true for warp_weight to have any effect
    // (see test_consistency_loss_warp_term_zero_when_flow_disabled for
    // the case where it is not).
    let fa = uniform_frame(8, 8, 0.0, 0.0, 0.0);
    let fb = uniform_frame(8, 8, 1.0, 1.0, 1.0);
    let fc = uniform_frame(8, 8, 0.5, 0.5, 0.5);
    let cfg_no_warp = ConsistencyLossConfig {
        warp_weight: 0.0,
        temporal_smooth_weight: 0.0,
        use_optical_flow: true,
        ..Default::default()
    };
    let cfg_with_warp = ConsistencyLossConfig {
        warp_weight: 1.0,
        temporal_smooth_weight: 0.0,
        use_optical_flow: true,
        ..Default::default()
    };
    let frames_no: Vec<Frame> = [
        fa.pixels().to_vec(),
        fb.pixels().to_vec(),
        fc.pixels().to_vec(),
    ]
    .into_iter()
    .zip([(8usize, 8usize), (8, 8), (8, 8)])
    .map(|(p, (w, h))| Frame::from_pixels(p, w, h).expect("valid pixel buffer"))
    .collect();
    let loss_no = cfc_consistency_loss(&frames_no, &cfg_no_warp, &FlowConfig::default())?;
    let loss_with = cfc_consistency_loss(&[fa, fb, fc], &cfg_with_warp, &FlowConfig::default())?;
    // Both should be positive; with-warp should differ from no-warp
    assert!(loss_no.total > 0.0);
    assert!(loss_with.total > 0.0);
    assert_ne!(loss_no.total, loss_with.total);
    Ok(())
}

#[test]
fn test_consistency_loss_warp_term_zero_when_flow_disabled() -> Result<(), ConsistencyError> {
    // Regression test: warp_term must be exactly 0 when use_optical_flow
    // is false, regardless of warp_weight — it must not silently
    // degenerate into a duplicate of l1_term.
    let fa = uniform_frame(8, 8, 0.0, 0.0, 0.0);
    let fb = uniform_frame(8, 8, 1.0, 1.0, 1.0);
    let cfg = ConsistencyLossConfig {
        warp_weight: 5.0, // large — would previously have dominated the loss
        use_optical_flow: false,
        temporal_smooth_weight: 0.0,
        ..Default::default()
    };
    let loss = cfc_consistency_loss(&[fa, fb], &cfg, &FlowConfig::default())?;
    assert_eq!(
        loss.warp_term, 0.0,
        "warp_term must be 0 when optical flow is disabled, got {}",
        loss.warp_term
    );
    Ok(())
}

// ── ConsistencyLossConfig default ─────────────────────────────────────────

#[test]
fn test_consistency_loss_config_default() {
    let cfg = ConsistencyLossConfig::default();
    assert_eq!(cfg.psnr_weight, 0.0);
    assert_eq!(cfg.l1_weight, 1.0);
    assert_eq!(cfg.warp_weight, 0.5);
    assert_eq!(cfg.temporal_smooth_weight, 0.1);
    assert!(!cfg.use_optical_flow);
}

#[test]
fn test_consistency_loss_psnr_weight_applied() -> Result<(), ConsistencyError> {
    // Regression test: psnr_weight must actually affect `total`.
    let cfg_zero = ConsistencyLossConfig {
        l1_weight: 0.0,
        warp_weight: 0.0,
        temporal_smooth_weight: 0.0,
        psnr_weight: 0.0,
        use_optical_flow: false,
    };
    let cfg_nonzero = ConsistencyLossConfig {
        psnr_weight: 2.0,
        ..cfg_zero.clone()
    };
    let frames_zero = vec![
        uniform_frame(8, 8, 0.0, 0.0, 0.0),
        uniform_frame(8, 8, 0.2, 0.2, 0.2),
    ];
    let frames_nonzero = vec![
        uniform_frame(8, 8, 0.0, 0.0, 0.0),
        uniform_frame(8, 8, 0.2, 0.2, 0.2),
    ];
    let loss_zero = cfc_consistency_loss(&frames_zero, &cfg_zero, &FlowConfig::default())?;
    let loss_nonzero = cfc_consistency_loss(&frames_nonzero, &cfg_nonzero, &FlowConfig::default())?;
    assert_eq!(
        loss_zero.total, 0.0,
        "all weights zero should give zero loss"
    );
    assert!(
        loss_nonzero.psnr_term > 0.0,
        "psnr_term should be nonzero for differing frames"
    );
    assert!(
        (loss_nonzero.total - cfg_nonzero.psnr_weight * loss_nonzero.psnr_term).abs() < 1e-6,
        "total should equal psnr_weight * psnr_term when all other weights are 0"
    );
    Ok(())
}

// ── FlowConfig default ───────────────────────────────────────────────────

#[test]
fn test_flow_config_default() {
    let cfg = FlowConfig::default();
    assert_eq!(cfg.alpha, 100.0);
    assert_eq!(cfg.n_iterations, 20);
    assert_eq!(cfg.scale, 2);
}

// ── Formatting ────────────────────────────────────────────────────────────

#[test]
fn test_format_pair_consistency_nonempty() {
    let c = FramePairConsistency {
        psnr: 30.5,
        ssim: 0.95,
        mean_warp_error: 0.01,
        mean_flow_magnitude: 2.3,
        occlusion_ratio: 0.05,
    };
    let s = cfc_format_pair_consistency(&c);
    assert!(!s.is_empty());
    assert!(s.contains("psnr") || s.contains("PSNR") || s.contains("30.50"));
}

#[test]
fn test_format_report_contains_frames_and_psnr() {
    let report = SequenceConsistencyReport {
        n_frames: 10,
        mean_psnr: 35.0,
        mean_ssim: 0.92,
        mean_warp_error: 0.005,
        temporal_variance: 1.2,
        worst_frame_pair: 2,
        best_frame_pair: 7,
        psnr_trend: 0.3,
        per_pair_psnr: vec![35.0; 9],
        per_pair_ssim: vec![0.92; 9],
    };
    let s = cfc_format_report(&report);
    assert!(s.contains("frames") || s.contains("10"));
    assert!(s.contains("PSNR") || s.contains("35.00") || s.contains("psnr"));
}

#[test]
fn test_format_loss_contains_total() {
    let loss = ConsistencyLoss {
        total: 0.123,
        l1_term: 0.1,
        warp_term: 0.02,
        smooth_term: 0.003,
        psnr_term: 0.0,
    };
    let s = cfc_format_loss(&loss);
    assert!(!s.is_empty());
    assert!(s.contains("total") || s.contains("0.123000"));
}

// ── Additional edge-case tests ────────────────────────────────────────────

#[test]
fn test_warp_uniform_frame_stays_uniform() -> Result<(), ConsistencyError> {
    let f = uniform_frame(8, 8, 0.4, 0.5, 0.6);
    let n = 64;
    let fx: Vec<f32> = (0..n).map(|_| 0.7_f32).collect();
    let fy: Vec<f32> = (0..n).map(|_| 0.3_f32).collect();
    let warped = cfc_warp_frame(&f, &fx, &fy)?;
    // Uniform frame: any shift should return the same color
    for chunk in warped.pixels.chunks(3) {
        assert!((chunk[0] - 0.4).abs() < 1e-5);
    }
    Ok(())
}

#[test]
fn test_grayscale_weighted_correctly() {
    // Pure green → 0.7152
    let f = uniform_frame(1, 1, 0.0, 1.0, 0.0);
    let g = cfc_to_grayscale(&f);
    assert!((g[0] - 0.7152).abs() < 1e-5, "got {}", g[0]);
}

#[test]
fn test_sequence_all_identical_psnr_infinite() -> Result<(), ConsistencyError> {
    let frames: Vec<Frame> = (0..3).map(|_| uniform_frame(8, 8, 0.5, 0.5, 0.5)).collect();
    let report = cfc_sequence_consistency(&frames, false, &FlowConfig::default())?;
    assert_eq!(report.mean_psnr, f32::INFINITY);
    Ok(())
}

#[test]
fn test_frame_difference_zero_occlusion_identical() -> Result<(), ConsistencyError> {
    let f = uniform_frame(16, 16, 0.5, 0.5, 0.5);
    let c = cfc_frame_difference(&f, &f)?;
    assert_eq!(c.occlusion_ratio, 0.0);
    Ok(())
}

#[test]
fn test_consistency_loss_smooth_term_nonzero() -> Result<(), ConsistencyError> {
    let fa = uniform_frame(8, 8, 0.0, 0.0, 0.0);
    let fb = uniform_frame(8, 8, 0.5, 0.5, 0.5);
    let fc = uniform_frame(8, 8, 0.0, 0.0, 0.0);
    // fc - 2*fb + fa = 0 - 1 + 0 = -1; |·| = 1 per channel
    let cfg = ConsistencyLossConfig {
        l1_weight: 0.0,
        warp_weight: 0.0,
        temporal_smooth_weight: 1.0,
        ..Default::default()
    };
    let loss = cfc_consistency_loss(&[fa, fb, fc], &cfg, &FlowConfig::default())?;
    assert!(
        loss.smooth_term > 0.0,
        "smooth_term should be nonzero: {}",
        loss.smooth_term
    );
    assert!((loss.total - loss.smooth_term).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_psnr_known_mse() -> Result<(), ConsistencyError> {
    // MSE = 0.01 → PSNR = 10*log10(1/0.01) = 10*2 = 20 dB
    // Build two frames where every channel differs by exactly sqrt(0.01)
    let diff = 0.1_f32; // diff^2 = 0.01
    let a = Frame::new(4, 4);
    let b = uniform_frame(4, 4, diff, diff, diff);
    let psnr = cfc_psnr(&a, &b)?;
    assert!((psnr - 20.0).abs() < 0.01, "expected 20 dB, got {psnr}");
    Ok(())
}
