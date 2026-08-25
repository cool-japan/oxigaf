//! Unit tests for [`crate::multi_resolution_loss`].
//!
//! Split out of the parent module to keep every file under the 2000-line cap.

use super::*;

// Helper: create a uniform flat image
fn uniform_image(w: usize, h: usize, c: usize, val: f32) -> Vec<f32> {
    vec![val; w * h * c]
}

// Helper: create a checkerboard pattern (0.0 and 1.0)
fn checkerboard(w: usize, h: usize, c: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; w * h * c];
    for y in 0..h {
        for x in 0..w {
            let val = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
            for ch in 0..c {
                v[(y * w + x) * c + ch] = val;
            }
        }
    }
    v
}

// Helper: create a horizontal edge image (top half = 0, bottom half = 1)
fn edge_image(w: usize, h: usize, c: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; w * h * c];
    for y in (h / 2)..h {
        for x in 0..w {
            for ch in 0..c {
                v[(y * w + x) * c + ch] = 1.0;
            }
        }
    }
    v
}

// ---------------------------------------------------------------------------
// mr_downsample
// ---------------------------------------------------------------------------

#[test]
fn test_downsample_4x4_to_2x2() {
    let img: Vec<f32> = (0..16).map(|i| i as f32).collect();
    // 4×4 single channel → 2×2
    let (out, ow, oh) = mr_downsample(&img, 4, 4, 1).expect("downsample failed");
    assert_eq!(ow, 2);
    assert_eq!(oh, 2);
    assert_eq!(out.len(), 4);
    // Top-left 2×2 block: [0,1,4,5] → mean = 2.5
    assert!((out[0] - 2.5).abs() < 1e-5, "TL={}", out[0]);
    // Top-right: [2,3,6,7] → 4.5
    assert!((out[1] - 4.5).abs() < 1e-5, "TR={}", out[1]);
}

#[test]
fn test_downsample_uniform_stays_uniform() {
    let img = uniform_image(8, 8, 3, 0.7);
    let (out, ow, oh) = mr_downsample(&img, 8, 8, 3).expect("downsample");
    assert_eq!(ow, 4);
    assert_eq!(oh, 4);
    for &v in &out {
        assert!((v - 0.7).abs() < 1e-5);
    }
}

#[test]
fn test_downsample_odd_size() {
    let img = uniform_image(5, 5, 1, 1.0);
    let (out, ow, oh) = mr_downsample(&img, 5, 5, 1).expect("downsample");
    assert_eq!(ow, 3);
    assert_eq!(oh, 3);
    assert_eq!(out.len(), 9);
}

#[test]
fn test_downsample_too_small_error() {
    let img = vec![1.0_f32];
    let res = mr_downsample(&img, 1, 1, 1);
    assert!(matches!(res, Err(MultiResLossError::ImageTooSmall(_))));
}

#[test]
fn test_downsample_multi_channel() {
    let img = uniform_image(4, 4, 3, 0.5);
    let (out, ow, oh) = mr_downsample(&img, 4, 4, 3).expect("downsample");
    assert_eq!(ow, 2);
    assert_eq!(oh, 2);
    assert_eq!(out.len(), 2 * 2 * 3);
    for &v in &out {
        assert!((v - 0.5).abs() < 1e-5);
    }
}

// ---------------------------------------------------------------------------
// mr_upsample
// ---------------------------------------------------------------------------

#[test]
fn test_upsample_2x2_to_4x4() {
    // 2×2 uniform → 4×4
    let img = uniform_image(2, 2, 1, 0.6);
    let out = mr_upsample(&img, 2, 2, 1, 4, 4).expect("upsample");
    assert_eq!(out.len(), 16);
    for &v in &out {
        assert!((v - 0.6).abs() < 1e-5);
    }
}

#[test]
fn test_upsample_uniform_stays_uniform() {
    let img = uniform_image(3, 3, 2, 0.3);
    let out = mr_upsample(&img, 3, 3, 2, 5, 5).expect("upsample");
    assert_eq!(out.len(), 5 * 5 * 2);
    for &v in &out {
        assert!((v - 0.3).abs() < 1e-5);
    }
}

#[test]
fn test_upsample_identity_target() {
    // Upsample to same size should return identical (or very close)
    let img = checkerboard(4, 4, 1);
    let out = mr_upsample(&img, 4, 4, 1, 4, 4).expect("upsample");
    assert_eq!(out.len(), img.len());
    for (a, b) in img.iter().zip(out.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn test_upsample_roundtrip_size() {
    // Downsample 5×5 → 3×3 → upsample back to 5×5
    let img = uniform_image(5, 5, 1, 1.0);
    let (down, dw, dh) = mr_downsample(&img, 5, 5, 1).expect("down");
    let up = mr_upsample(&down, dw, dh, 1, 5, 5).expect("upsample");
    assert_eq!(up.len(), 5 * 5);
}

#[test]
fn test_upsample_size_mismatch_returns_error_not_panic() {
    // Regression: mr_upsample used to index `img` without validating its
    // length against width*height*channels, panicking on a mismatch
    // instead of returning `SizeMismatch`.
    let img = vec![0.0_f32; 10];
    let res = mr_upsample(&img, 8, 8, 1, 4, 4);
    assert!(matches!(res, Err(MultiResLossError::SizeMismatch { .. })));
}

// ---------------------------------------------------------------------------
// mr_gaussian_blur_3x3
// ---------------------------------------------------------------------------

#[test]
fn test_gaussian_blur_uniform_unchanged() {
    let img = uniform_image(6, 6, 1, 0.5);
    let out = mr_gaussian_blur_3x3(&img, 6, 6, 1).expect("blur");
    assert_eq!(out.len(), img.len());
    for &v in &out {
        assert!((v - 0.5).abs() < 1e-5);
    }
}

#[test]
fn test_gaussian_blur_edge_pixels_handled() {
    let img = checkerboard(8, 8, 1);
    let out = mr_gaussian_blur_3x3(&img, 8, 8, 1).expect("blur");
    // Just verify no panic and output size is correct
    assert_eq!(out.len(), img.len());
    // All values should be in [0, 1]
    for &v in &out {
        assert!((0.0..=1.0 + 1e-5).contains(&v), "v={}", v);
    }
}

#[test]
fn test_gaussian_blur_multi_channel() {
    let img = uniform_image(4, 4, 3, 0.8);
    let out = mr_gaussian_blur_3x3(&img, 4, 4, 3).expect("blur");
    assert_eq!(out.len(), 4 * 4 * 3);
    for &v in &out {
        assert!((v - 0.8).abs() < 1e-5);
    }
}

#[test]
fn test_gaussian_blur_size_mismatch_returns_error_not_panic() {
    // Regression: mr_gaussian_blur_3x3(&vec![0.0; 10], 8, 8, 1) used to
    // index out of bounds and panic instead of returning `SizeMismatch`.
    let img = vec![0.0_f32; 10];
    let res = mr_gaussian_blur_3x3(&img, 8, 8, 1);
    assert!(matches!(res, Err(MultiResLossError::SizeMismatch { .. })));
}

// ---------------------------------------------------------------------------
// mr_l1_loss
// ---------------------------------------------------------------------------

#[test]
fn test_l1_identical_is_zero() {
    let img = checkerboard(8, 8, 3);
    let loss = mr_l1_loss(&img, &img).expect("l1");
    assert!(loss.abs() < 1e-7);
}

#[test]
fn test_l1_known_difference() {
    let pred = vec![0.0_f32, 1.0, 0.0, 1.0];
    let gt = vec![1.0_f32, 0.0, 1.0, 0.0];
    let loss = mr_l1_loss(&pred, &gt).expect("l1");
    assert!((loss - 1.0).abs() < 1e-6);
}

#[test]
fn test_l1_partial_difference() {
    let pred = vec![0.0_f32, 0.0];
    let gt = vec![0.5_f32, 0.5];
    let loss = mr_l1_loss(&pred, &gt).expect("l1");
    assert!((loss - 0.5).abs() < 1e-6);
}

#[test]
fn test_l1_shape_mismatch_error() {
    let res = mr_l1_loss(&[1.0_f32, 2.0], &[1.0_f32]);
    assert!(matches!(res, Err(MultiResLossError::ShapeMismatch)));
}

// ---------------------------------------------------------------------------
// mr_l2_loss
// ---------------------------------------------------------------------------

#[test]
fn test_l2_identical_is_zero() {
    let img = uniform_image(4, 4, 1, 0.5);
    let loss = mr_l2_loss(&img, &img).expect("l2");
    assert!(loss.abs() < 1e-7);
}

#[test]
fn test_l2_known_difference() {
    // Mean of (1-0)^2 = 1.0
    let pred = vec![0.0_f32; 4];
    let gt = vec![1.0_f32; 4];
    let loss = mr_l2_loss(&pred, &gt).expect("l2");
    assert!((loss - 1.0).abs() < 1e-6);
}

#[test]
fn test_l2_shape_mismatch_error() {
    let res = mr_l2_loss(&[1.0_f32], &[1.0_f32, 2.0]);
    assert!(matches!(res, Err(MultiResLossError::ShapeMismatch)));
}

// ---------------------------------------------------------------------------
// mr_ssim_loss
// ---------------------------------------------------------------------------

#[test]
fn test_ssim_identical_is_zero() {
    let img = checkerboard(8, 8, 1);
    let loss = mr_ssim_loss(&img, &img, 8, 8).expect("ssim");
    assert!(loss < 1e-5, "loss={}", loss);
}

#[test]
fn test_ssim_different_is_positive() {
    let pred = uniform_image(8, 8, 1, 0.0);
    let gt = uniform_image(8, 8, 1, 1.0);
    let loss = mr_ssim_loss(&pred, &gt, 8, 8).expect("ssim");
    assert!(loss > 0.0, "Expected positive SSIM loss");
}

#[test]
fn test_ssim_small_image_no_panic() {
    // 2×2 images — the window shrinks to 1 tap rather than bailing out to an
    // L1 stand-in, so this is still a real SSIM evaluation.
    let pred = vec![0.0_f32, 1.0, 0.0, 1.0];
    let gt = vec![1.0_f32, 0.0, 1.0, 0.0];
    let loss = mr_ssim_loss(&pred, &gt, 2, 2).expect("ssim small");
    assert!(loss >= 0.0);
    // Anti-correlated 2x2 patches are structurally dissimilar, so the loss
    // must be substantial rather than the old `l1.min(1.0)` coincidence.
    assert!(loss > 0.5, "loss={loss}");
    // Identical small images still score a perfect match.
    let same = mr_ssim_loss(&pred, &pred, 2, 2).expect("ssim small identical");
    assert!(same < 1e-5, "identical 2x2 loss={same}");
}

// ── Regression (F295): mr_ssim_loss agrees with the unified SSIM ─────────────
// It used to run a hand-rolled "simplified 3x3 window" box-filter SSIM, a
// genuine divergence from the 11x11 Gaussian window used by
// `crate::loss::ssim_loss` / `crate::view_synthesis_eval::eval_ssim`, and it
// bailed out to an L1 proxy on images smaller than 3x3.
#[test]
fn test_ssim_matches_crate_loss_ssim_on_rgb() {
    // 16x16 >= 11 in both dimensions, so both implementations use the full
    // 11-tap sigma=1.5 window and must agree.
    let (w, h) = (16usize, 16usize);
    let mut pred = vec![0.0_f32; w * h * 3];
    let mut gt = vec![0.0_f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let i = (y * w + x) * 3 + c;
                let fx = x as f32 / w as f32;
                let fy = y as f32 / h as f32;
                pred[i] = (fx * 6.0).sin() * 0.25 + fy * 0.5 + c as f32 * 0.05;
                gt[i] = (fx * 6.0 + 0.3).sin() * 0.25 + fy * 0.45 + c as f32 * 0.05;
            }
        }
    }

    let kernel = crate::loss::gaussian_kernel_1d(11, 1.5);
    let reference = crate::loss::ssim_loss(&pred, &gt, w, h, &kernel);
    let ours = mr_ssim_loss(&pred, &gt, w, h).expect("ssim");
    assert!(
        (ours - reference).abs() < 1e-4,
        "multi-resolution SSIM {ours} diverged from crate::loss::ssim_loss {reference}"
    );
    assert!(ours > 0.0, "the two images differ, so the loss must be > 0");

    // And identical inputs agree on a perfect match in both implementations.
    let ref_same = crate::loss::ssim_loss(&pred, &pred, w, h, &kernel);
    let ours_same = mr_ssim_loss(&pred, &pred, w, h).expect("ssim identical");
    assert!(ref_same.abs() < 1e-4 && ours_same.abs() < 1e-4);
}

#[test]
fn test_ssim_window_shrinks_with_the_pyramid_level() {
    // A 4x4 level cannot host an 11-tap window; the estimator still runs and
    // still reports a perfect match for identical inputs and a positive loss
    // for different ones.
    let a = checkerboard(4, 4, 1);
    let b = uniform_image(4, 4, 1, 0.5);
    assert!(mr_ssim_loss(&a, &a, 4, 4).expect("ssim") < 1e-5);
    let differ = mr_ssim_loss(&a, &b, 4, 4).expect("ssim");
    assert!(differ > 0.0, "loss={differ}");
    assert!(differ.is_finite());
}

#[test]
fn test_ssim_multichannel_is_mean_over_channels() {
    // A 3-channel image whose channels are independent copies of the same
    // single-channel pair must score exactly the single-channel loss.
    let w = 12usize;
    let h = 12usize;
    let mono_pred = checkerboard(w, h, 1);
    let mono_gt = uniform_image(w, h, 1, 0.25);
    let mut rgb_pred = vec![0.0_f32; w * h * 3];
    let mut rgb_gt = vec![0.0_f32; w * h * 3];
    for i in 0..(w * h) {
        for c in 0..3 {
            rgb_pred[i * 3 + c] = mono_pred[i];
            rgb_gt[i * 3 + c] = mono_gt[i];
        }
    }
    let mono = mr_ssim_loss(&mono_pred, &mono_gt, w, h).expect("mono ssim");
    let rgb = mr_ssim_loss(&rgb_pred, &rgb_gt, w, h).expect("rgb ssim");
    assert!(
        (mono - rgb).abs() < 1e-5,
        "mono={mono} rgb={rgb} must match for replicated channels"
    );
}

#[test]
fn test_ssim_shape_mismatch() {
    let res = mr_ssim_loss(&[1.0_f32, 2.0], &[1.0_f32], 1, 1);
    assert!(matches!(res, Err(MultiResLossError::ShapeMismatch)));
}

// ---------------------------------------------------------------------------
// mr_gradient_l1_loss
// ---------------------------------------------------------------------------

#[test]
fn test_gradient_l1_identical_is_zero() {
    let img = edge_image(8, 8, 1);
    let loss = mr_gradient_l1_loss(&img, &img, 8, 8).expect("grad");
    assert!(loss.abs() < 1e-7);
}

#[test]
fn test_gradient_l1_edge_vs_flat_positive() {
    let pred = edge_image(8, 8, 1);
    let gt = uniform_image(8, 8, 1, 0.5);
    let loss = mr_gradient_l1_loss(&pred, &gt, 8, 8).expect("grad");
    assert!(loss > 0.0);
}

#[test]
fn test_gradient_l1_shape_mismatch() {
    let res = mr_gradient_l1_loss(&[1.0_f32], &[1.0_f32, 2.0], 1, 1);
    assert!(matches!(res, Err(MultiResLossError::ShapeMismatch)));
}

// ---------------------------------------------------------------------------
// mr_sobel_magnitude
// ---------------------------------------------------------------------------

#[test]
fn test_sobel_flat_image_is_zero() {
    let img = uniform_image(6, 6, 1, 0.5);
    let mag = mr_sobel_magnitude(&img, 6, 6, 1).expect("sobel");
    assert_eq!(mag.len(), 36);
    // Interior pixels should all be 0 for a uniform image
    for y in 1..5 {
        for x in 1..5 {
            assert!(
                mag[y * 6 + x].abs() < 1e-5,
                "mag[{},{}]={}",
                y,
                x,
                mag[y * 6 + x]
            );
        }
    }
}

#[test]
fn test_sobel_edge_image_nonzero() {
    let img = edge_image(8, 8, 1);
    let mag = mr_sobel_magnitude(&img, 8, 8, 1).expect("sobel");
    assert_eq!(mag.len(), 64);
    // Should have nonzero gradient at the edge row
    let max_mag = mag.iter().cloned().fold(0.0_f32, f32::max);
    assert!(max_mag > 0.0, "Expected nonzero gradient");
}

#[test]
fn test_sobel_border_pixels_are_zero() {
    let img = checkerboard(6, 6, 1);
    let mag = mr_sobel_magnitude(&img, 6, 6, 1).expect("sobel");
    // Border pixels should be 0
    for x in 0..6 {
        assert_eq!(mag[x], 0.0, "top border x={}", x);
        assert_eq!(mag[5 * 6 + x], 0.0, "bottom border x={}", x);
    }
    for y in 0..6 {
        assert_eq!(mag[y * 6], 0.0, "left border y={}", y);
        assert_eq!(mag[y * 6 + 5], 0.0, "right border y={}", y);
    }
}

#[test]
fn test_sobel_multi_channel() {
    let img = edge_image(8, 8, 3);
    let mag = mr_sobel_magnitude(&img, 8, 8, 3).expect("sobel");
    // Output has one value per pixel
    assert_eq!(mag.len(), 64);
}

#[test]
fn test_sobel_size_mismatch_returns_error_not_panic() {
    // Regression: mr_sobel_magnitude used to index `img` without
    // validating its length, panicking on a mismatch instead of
    // returning `SizeMismatch`.
    let img = vec![0.0_f32; 10];
    let res = mr_sobel_magnitude(&img, 8, 8, 1);
    assert!(matches!(res, Err(MultiResLossError::SizeMismatch { .. })));
}

// ---------------------------------------------------------------------------
// mr_laplacian_pyramid
// ---------------------------------------------------------------------------

#[test]
fn test_laplacian_pyramid_length() {
    let img = checkerboard(16, 16, 1);
    let lap = mr_laplacian_pyramid(&img, 16, 16, 1, 4).expect("lap");
    assert_eq!(lap.len(), 4);
}

#[test]
fn test_laplacian_pyramid_level_sizes() {
    let img = checkerboard(8, 8, 1);
    let lap = mr_laplacian_pyramid(&img, 8, 8, 1, 3).expect("lap");
    // lap[0] should be same size as original (8×8×1 = 64)
    assert_eq!(lap[0].len(), 64);
}

#[test]
fn test_laplacian_pyramid_no_levels_error() {
    let img = vec![1.0_f32];
    let res = mr_laplacian_pyramid(&img, 1, 1, 1, 0);
    assert!(matches!(res, Err(MultiResLossError::NoLevels)));
}

#[test]
fn test_laplacian_pyramid_single_level() {
    let img = uniform_image(4, 4, 1, 0.5);
    let lap = mr_laplacian_pyramid(&img, 4, 4, 1, 1).expect("lap");
    assert_eq!(lap.len(), 1);
}

// ---------------------------------------------------------------------------
// mr_laplacian_loss
// ---------------------------------------------------------------------------

#[test]
fn test_laplacian_loss_identical_near_zero() {
    let img = checkerboard(16, 16, 1);
    let loss = mr_laplacian_loss(&img, &img, 16, 16).expect("lap_loss");
    assert!(loss < 1e-5, "loss={}", loss);
}

#[test]
fn test_laplacian_loss_different_positive() {
    let pred = uniform_image(8, 8, 1, 0.0);
    let gt = checkerboard(8, 8, 1);
    let loss = mr_laplacian_loss(&pred, &gt, 8, 8).expect("lap_loss");
    assert!(loss >= 0.0);
}

#[test]
fn test_laplacian_loss_shape_mismatch() {
    let res = mr_laplacian_loss(&[1.0_f32, 2.0], &[1.0_f32], 1, 1);
    assert!(matches!(res, Err(MultiResLossError::ShapeMismatch)));
}

// ---------------------------------------------------------------------------
// ImagePyramid
// ---------------------------------------------------------------------------

#[test]
fn test_pyramid_build_correct_levels() {
    let img = checkerboard(16, 16, 1);
    let pyr = ImagePyramid::build(&img, 16, 16, 1, 4).expect("build");
    assert_eq!(pyr.n_levels(), 4);
}

#[test]
fn test_pyramid_scale_halves() {
    let img = checkerboard(16, 16, 1);
    let pyr = ImagePyramid::build(&img, 16, 16, 1, 4).expect("build");
    assert!((pyr.levels[0].scale - 1.0).abs() < 1e-6);
    assert!((pyr.levels[1].scale - 0.5).abs() < 1e-6);
    assert!((pyr.levels[2].scale - 0.25).abs() < 1e-6);
    assert!((pyr.levels[3].scale - 0.125).abs() < 1e-6);
}

#[test]
fn test_pyramid_original_is_level_0() {
    let img = checkerboard(8, 8, 3);
    let pyr = ImagePyramid::build(&img, 8, 8, 3, 3).expect("build");
    let original = pyr.original().expect("build() always inserts level 0");
    assert_eq!(original.width, 8);
    assert_eq!(original.height, 8);
    assert_eq!(original.data.len(), 8 * 8 * 3);
}

#[test]
fn test_pyramid_original_none_when_hand_constructed_empty() {
    // `ImagePyramid`'s fields are public, so a hand-constructed empty
    // pyramid is possible without going through `build()`. `original()`
    // must return `None` rather than panicking.
    let pyr = ImagePyramid {
        levels: Vec::new(),
        channels: 3,
    };
    assert!(pyr.original().is_none());
}

#[test]
fn test_pyramid_level_out_of_range_returns_none() {
    let img = checkerboard(8, 8, 1);
    let pyr = ImagePyramid::build(&img, 8, 8, 1, 3).expect("build");
    assert!(pyr.level(100).is_none());
    assert!(pyr.level(3).is_none());
}

#[test]
fn test_pyramid_level_in_range() {
    let img = checkerboard(8, 8, 1);
    let pyr = ImagePyramid::build(&img, 8, 8, 1, 3).expect("build");
    assert!(pyr.level(0).is_some());
    assert!(pyr.level(2).is_some());
}

#[test]
fn test_pyramid_size_mismatch_error() {
    let img = vec![1.0_f32; 10]; // wrong size
    let res = ImagePyramid::build(&img, 4, 4, 1, 2);
    assert!(matches!(res, Err(MultiResLossError::SizeMismatch { .. })));
}

#[test]
fn test_pyramid_no_levels_error() {
    let img = uniform_image(4, 4, 1, 0.5);
    let res = ImagePyramid::build(&img, 4, 4, 1, 0);
    assert!(matches!(res, Err(MultiResLossError::NoLevels)));
}

#[test]
fn test_pyramid_stops_when_too_small() {
    // A 2×2 image can only have 1 meaningful level + 1 downsampled
    let img = uniform_image(2, 2, 1, 0.5);
    let pyr = ImagePyramid::build(&img, 2, 2, 1, 10).expect("build");
    // Should not have 10 levels — image becomes too small
    assert!(pyr.n_levels() < 10);
}

// ---------------------------------------------------------------------------
// mr_compute_loss
// ---------------------------------------------------------------------------

#[test]
fn test_compute_loss_uniform_near_zero() {
    let pred = uniform_image(8, 8, 3, 0.5);
    let gt = uniform_image(8, 8, 3, 0.5);
    let config = MultiResLossConfig::default();
    let result = mr_compute_loss(&pred, &gt, 8, 8, 3, &config).expect("compute");
    assert!(result.total_loss < 1e-5, "total={}", result.total_loss);
}

#[test]
fn test_compute_loss_different_images_positive() {
    let pred = uniform_image(8, 8, 3, 0.0);
    let gt = uniform_image(8, 8, 3, 1.0);
    let config = MultiResLossConfig::default();
    let result = mr_compute_loss(&pred, &gt, 8, 8, 3, &config).expect("compute");
    assert!(result.total_loss > 0.0, "Expected positive loss");
}

#[test]
fn test_compute_loss_non_negative() {
    let pred = checkerboard(8, 8, 3);
    let gt = edge_image(8, 8, 3);
    let config = MultiResLossConfig::default();
    let result = mr_compute_loss(&pred, &gt, 8, 8, 3, &config).expect("compute");
    assert!(result.total_loss >= 0.0);
    for &l in &result.per_level_losses {
        assert!(l >= 0.0);
    }
    for &w in &result.weighted_level_losses {
        assert!(w >= 0.0);
    }
}

#[test]
fn test_compute_loss_size_mismatch_error() {
    let pred = uniform_image(8, 8, 3, 0.5);
    let gt = uniform_image(8, 8, 3, 0.5);
    let config = MultiResLossConfig::default();
    // Pass wrong size
    let res = mr_compute_loss(&pred[..10], &gt, 8, 8, 3, &config);
    assert!(matches!(res, Err(MultiResLossError::SizeMismatch { .. })));
}

#[test]
fn test_compute_loss_with_all_loss_types() {
    let pred = edge_image(16, 16, 1);
    let gt = checkerboard(16, 16, 1);
    let config = MultiResLossConfig {
        n_levels: 3,
        level_weights: vec![1.0, 0.5, 0.25],
        loss_types: vec![
            MultiResLossType::L1,
            MultiResLossType::L2,
            MultiResLossType::Ssim,
            MultiResLossType::Laplacian,
            MultiResLossType::GradientL1,
        ],
        normalize_weights: true,
    };
    let result = mr_compute_loss(&pred, &gt, 16, 16, 1, &config).expect("compute all types");
    assert!(result.total_loss >= 0.0);
    assert_eq!(result.per_type_losses.len(), 5);
}

#[test]
fn test_compute_loss_per_level_count() {
    let pred = checkerboard(16, 16, 1);
    let gt = edge_image(16, 16, 1);
    let config = MultiResLossConfig {
        n_levels: 3,
        level_weights: vec![1.0, 0.5, 0.25],
        loss_types: vec![MultiResLossType::L1],
        normalize_weights: false,
    };
    let result = mr_compute_loss(&pred, &gt, 16, 16, 1, &config).expect("compute");
    assert_eq!(result.per_level_losses.len(), 3);
    assert_eq!(result.weighted_level_losses.len(), 3);
}

// ---------------------------------------------------------------------------
// mr_level_loss
// ---------------------------------------------------------------------------

#[test]
fn test_level_loss_l1() {
    let lvl_pred = PyramidLevel {
        data: vec![0.0_f32; 16],
        width: 4,
        height: 4,
        scale: 1.0,
    };
    let lvl_gt = PyramidLevel {
        data: vec![1.0_f32; 16],
        width: 4,
        height: 4,
        scale: 1.0,
    };
    let loss = mr_level_loss(&lvl_pred, &lvl_gt, &MultiResLossType::L1).expect("ll");
    assert!((loss - 1.0).abs() < 1e-6);
}

#[test]
fn test_level_loss_l2() {
    let lvl_pred = PyramidLevel {
        data: vec![0.0_f32; 16],
        width: 4,
        height: 4,
        scale: 1.0,
    };
    let lvl_gt = PyramidLevel {
        data: vec![1.0_f32; 16],
        width: 4,
        height: 4,
        scale: 1.0,
    };
    let loss = mr_level_loss(&lvl_pred, &lvl_gt, &MultiResLossType::L2).expect("ll");
    assert!((loss - 1.0).abs() < 1e-6);
}

#[test]
fn test_level_loss_ssim() {
    let pred_data = checkerboard(8, 8, 1);
    let lvl_pred = PyramidLevel {
        data: pred_data.clone(),
        width: 8,
        height: 8,
        scale: 1.0,
    };
    let lvl_gt = PyramidLevel {
        data: pred_data.clone(),
        width: 8,
        height: 8,
        scale: 1.0,
    };
    let loss = mr_level_loss(&lvl_pred, &lvl_gt, &MultiResLossType::Ssim).expect("ll ssim");
    assert!(loss < 1e-5, "loss={}", loss);
}

#[test]
fn test_level_loss_laplacian() {
    let data = checkerboard(8, 8, 1);
    let lvl_pred = PyramidLevel {
        data: data.clone(),
        width: 8,
        height: 8,
        scale: 1.0,
    };
    let lvl_gt = PyramidLevel {
        data: data.clone(),
        width: 8,
        height: 8,
        scale: 1.0,
    };
    let loss = mr_level_loss(&lvl_pred, &lvl_gt, &MultiResLossType::Laplacian).expect("ll lap");
    assert!(loss < 1e-5, "loss={}", loss);
}

#[test]
fn test_level_loss_gradient_l1() {
    let data = edge_image(8, 8, 1);
    let lvl_pred = PyramidLevel {
        data: data.clone(),
        width: 8,
        height: 8,
        scale: 1.0,
    };
    let lvl_gt = PyramidLevel {
        data: data.clone(),
        width: 8,
        height: 8,
        scale: 1.0,
    };
    let loss = mr_level_loss(&lvl_pred, &lvl_gt, &MultiResLossType::GradientL1).expect("ll grad");
    assert!(loss.abs() < 1e-7);
}

#[test]
fn test_level_loss_shape_mismatch() {
    let lvl_pred = PyramidLevel {
        data: vec![0.0_f32; 4],
        width: 2,
        height: 2,
        scale: 1.0,
    };
    let lvl_gt = PyramidLevel {
        data: vec![0.0_f32; 9],
        width: 3,
        height: 3,
        scale: 1.0,
    };
    let res = mr_level_loss(&lvl_pred, &lvl_gt, &MultiResLossType::L1);
    assert!(matches!(res, Err(MultiResLossError::ShapeMismatch)));
}

// ---------------------------------------------------------------------------
// mr_compute_stats
// ---------------------------------------------------------------------------

#[test]
fn test_stats_quality_score() {
    let result = MultiResLossResult {
        total_loss: 0.0,
        per_level_losses: vec![0.0],
        per_type_losses: vec![0.0],
        weighted_level_losses: vec![0.0],
    };
    let stats = mr_compute_stats(&result);
    assert!((stats.quality_score - 1.0).abs() < 1e-6);
}

#[test]
fn test_stats_quality_decreases_with_loss() {
    let result_low = MultiResLossResult {
        total_loss: 0.1,
        per_level_losses: vec![0.1],
        per_type_losses: vec![0.1],
        weighted_level_losses: vec![0.1],
    };
    let result_high = MultiResLossResult {
        total_loss: 0.9,
        per_level_losses: vec![0.9],
        per_type_losses: vec![0.9],
        weighted_level_losses: vec![0.9],
    };
    let stats_low = mr_compute_stats(&result_low);
    let stats_high = mr_compute_stats(&result_high);
    assert!(stats_low.quality_score > stats_high.quality_score);
}

#[test]
fn test_stats_dominant_scale() {
    let result = MultiResLossResult {
        total_loss: 1.0,
        per_level_losses: vec![0.1, 0.5, 0.3],
        per_type_losses: vec![0.5],
        weighted_level_losses: vec![0.1, 0.5, 0.3],
    };
    let stats = mr_compute_stats(&result);
    assert_eq!(
        stats.dominant_scale, 1,
        "Highest weighted loss is at index 1"
    );
}

#[test]
fn test_stats_improvement_ratio() {
    let result = MultiResLossResult {
        total_loss: 0.5,
        per_level_losses: vec![0.2, 0.4],
        per_type_losses: vec![0.3],
        weighted_level_losses: vec![0.2, 0.2],
    };
    let stats = mr_compute_stats(&result);
    // fine(0.2) / coarse(0.4) = 0.5
    assert!((stats.loss_improvement_ratio - 0.5).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// MultiResLossConfig
// ---------------------------------------------------------------------------

#[test]
fn test_config_default_n_levels() {
    let cfg = MultiResLossConfig::default();
    assert_eq!(cfg.n_levels, 4);
}

#[test]
fn test_config_default_weights() {
    let cfg = MultiResLossConfig::default();
    assert_eq!(cfg.level_weights.len(), 4);
    assert!((cfg.level_weights[0] - 1.0).abs() < 1e-6);
    assert!((cfg.level_weights[1] - 0.5).abs() < 1e-6);
    assert!((cfg.level_weights[2] - 0.25).abs() < 1e-6);
    assert!((cfg.level_weights[3] - 0.125).abs() < 1e-6);
}

#[test]
fn test_config_default_loss_types() {
    let cfg = MultiResLossConfig::default();
    assert_eq!(cfg.loss_types.len(), 2);
    assert!(cfg.loss_types.contains(&MultiResLossType::L1));
    assert!(cfg.loss_types.contains(&MultiResLossType::Ssim));
}

#[test]
fn test_config_invalid_weight_error() {
    let cfg = MultiResLossConfig {
        n_levels: 2,
        level_weights: vec![1.0, -0.5],
        loss_types: vec![MultiResLossType::L1],
        normalize_weights: false,
    };
    assert!(matches!(
        cfg.validate(),
        Err(MultiResLossError::InvalidWeight(_))
    ));
}

#[test]
fn test_config_zero_weight_error() {
    let cfg = MultiResLossConfig {
        n_levels: 2,
        level_weights: vec![0.0, 1.0],
        loss_types: vec![MultiResLossType::L1],
        normalize_weights: false,
    };
    assert!(matches!(
        cfg.validate(),
        Err(MultiResLossError::InvalidWeight(_))
    ));
}

#[test]
fn test_config_normalize_weights() {
    let cfg = MultiResLossConfig {
        n_levels: 2,
        level_weights: vec![2.0, 2.0],
        loss_types: vec![MultiResLossType::L1],
        normalize_weights: true,
    };
    let weights = cfg.effective_weights(2);
    assert!((weights[0] - 0.5).abs() < 1e-6);
    assert!((weights[1] - 0.5).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// format functions
// ---------------------------------------------------------------------------

#[test]
fn test_format_mr_result_non_empty() {
    let result = MultiResLossResult {
        total_loss: 0.42,
        per_level_losses: vec![0.1, 0.2],
        per_type_losses: vec![0.15],
        weighted_level_losses: vec![0.1, 0.1],
    };
    let s = format_mr_result(&result);
    assert!(!s.is_empty());
    assert!(s.contains("0.42") || s.contains("MultiRes"));
}

#[test]
fn test_format_mr_stats_non_empty() {
    let result = MultiResLossResult {
        total_loss: 0.5,
        per_level_losses: vec![0.3, 0.7],
        per_type_losses: vec![0.5],
        weighted_level_losses: vec![0.3, 0.7],
    };
    let stats = mr_compute_stats(&result);
    let s = format_mr_stats(&stats);
    assert!(!s.is_empty());
    assert!(s.contains("quality") || s.contains("MultiRes"));
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

#[test]
fn test_error_image_too_small() {
    let img = vec![1.0_f32];
    let res = mr_downsample(&img, 1, 1, 1);
    assert!(matches!(res, Err(MultiResLossError::ImageTooSmall(_))));
}

#[test]
fn test_error_no_levels() {
    let img = uniform_image(4, 4, 1, 0.5);
    let res = ImagePyramid::build(&img, 4, 4, 1, 0);
    assert!(matches!(res, Err(MultiResLossError::NoLevels)));
}

#[test]
fn test_error_shape_mismatch_gt() {
    let pred = uniform_image(8, 8, 3, 0.5);
    let gt = uniform_image(4, 4, 3, 0.5); // different size
    let config = MultiResLossConfig::default();
    let res = mr_compute_loss(&pred, &gt, 8, 8, 3, &config);
    assert!(matches!(res, Err(MultiResLossError::SizeMismatch { .. })));
}

#[test]
fn test_error_invalid_weight() {
    let cfg = MultiResLossConfig {
        n_levels: 1,
        level_weights: vec![f32::NAN],
        loss_types: vec![MultiResLossType::L1],
        normalize_weights: false,
    };
    assert!(matches!(
        cfg.validate(),
        Err(MultiResLossError::InvalidWeight(_))
    ));
}
