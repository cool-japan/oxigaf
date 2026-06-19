//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::functions::{logit, sigmoid};
use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    #[test]
    fn test_config_default() {
        let cfg = MipSplatConfig::default();
        assert!((cfg.min_2d_radius_px - 0.3).abs() < 1e-6);
        assert!((cfg.max_distance_scale - 1.0).abs() < 1e-6);
        assert!((cfg.opacity_ramp_min_px - 0.5).abs() < 1e-6);
        assert!((cfg.opacity_ramp_max_px - 2.0).abs() < 1e-6);
        assert!(!cfg.use_distance_lod);
        assert!((cfg.reference_distance - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_config_aggressive() {
        let cfg = MipSplatConfig::aggressive();
        assert!((cfg.min_2d_radius_px - 0.5).abs() < 1e-6);
        assert!((cfg.opacity_ramp_min_px - 0.3).abs() < 1e-6);
        assert!((cfg.opacity_ramp_max_px - 3.0).abs() < 1e-6);
    }
    #[test]
    fn test_config_conservative() {
        let cfg = MipSplatConfig::conservative();
        assert!((cfg.min_2d_radius_px - 0.1).abs() < 1e-6);
        assert!((cfg.opacity_ramp_min_px - 0.1).abs() < 1e-6);
        assert!((cfg.opacity_ramp_max_px - 1.0).abs() < 1e-6);
    }
    #[test]
    fn test_compute_screen_radius_basic() {
        let r = compute_screen_radius_px(1.0, 10.0, 500.0);
        assert!((r - 50.0).abs() < 1e-4, "expected 50.0 px, got {r}");
    }
    #[test]
    fn test_compute_screen_radius_close() {
        let r = compute_screen_radius_px(1.0, 1e-10, 500.0);
        assert_eq!(r, 10_000.0, "expected clamped max, got {r}");
    }
    #[test]
    fn test_screen_radius_clamp() {
        let r = compute_screen_radius_px(100.0, 0.0, 100.0);
        assert_eq!(r, 10_000.0, "zero distance should give clamped max {r}");
    }
    #[test]
    fn test_compute_screen_radius_proportional() {
        let r1 = compute_screen_radius_px(1.0, 5.0, 400.0);
        let r2 = compute_screen_radius_px(1.0, 10.0, 400.0);
        assert!(
            (r1 - 2.0 * r2).abs() < 1e-4,
            "r1={r1}, r2={r2}; expected r1 = 2 * r2"
        );
    }
    #[test]
    fn test_opacity_scale_below_ramp() {
        let cfg = MipSplatConfig::default();
        let scale = opacity_scale_from_screen_radius(0.2, &cfg);
        assert_eq!(scale, 0.0, "below ramp_min should return 0.0, got {scale}");
    }
    #[test]
    fn test_opacity_scale_above_ramp() {
        let cfg = MipSplatConfig::default();
        let scale = opacity_scale_from_screen_radius(5.0, &cfg);
        assert_eq!(scale, 1.0, "above ramp_max should return 1.0, got {scale}");
    }
    #[test]
    fn test_opacity_scale_midpoint() {
        let cfg = MipSplatConfig::default();
        let mid = (cfg.opacity_ramp_min_px + cfg.opacity_ramp_max_px) / 2.0;
        let scale = opacity_scale_from_screen_radius(mid, &cfg);
        assert!(
            (scale - 0.5).abs() < 1e-5,
            "midpoint should give 0.5, got {scale}"
        );
    }
    #[test]
    fn test_opacity_scale_at_ramp_min() {
        let cfg = MipSplatConfig::default();
        let scale = opacity_scale_from_screen_radius(cfg.opacity_ramp_min_px, &cfg);
        assert!(
            scale.abs() < 1e-5,
            "at ramp_min t should be 0.0, got {scale}"
        );
    }
    #[test]
    fn test_opacity_scale_at_ramp_max() {
        let cfg = MipSplatConfig::default();
        let scale = opacity_scale_from_screen_radius(cfg.opacity_ramp_max_px, &cfg);
        assert!(
            (scale - 1.0).abs() < 1e-5,
            "at ramp_max should give 1.0, got {scale}"
        );
    }
    #[test]
    fn test_scale_compensation_no_need() {
        let cfg = MipSplatConfig::default();
        let comp = compute_scale_compensation(1.0, 5.0, 500.0, &cfg);
        assert!(
            (comp - 1.0).abs() < 1e-6,
            "no compensation needed, got {comp}"
        );
    }
    #[test]
    fn test_scale_compensation_small_gaussian() {
        let cfg = MipSplatConfig::default();
        let comp = compute_scale_compensation(0.001, 100.0, 500.0, &cfg);
        assert!(comp > 1.0, "compensation should be > 1.0, got {comp}");
        assert!(
            comp <= 10.0,
            "compensation should be clamped to 10.0, got {comp}"
        );
        assert!(
            (comp - 10.0).abs() < 1e-5,
            "should be clamped to 10.0, got {comp}"
        );
    }
    #[test]
    fn test_scale_compensation_max_clamp() {
        let cfg = MipSplatConfig::default();
        let comp = compute_scale_compensation(1e-7, 1000.0, 1000.0, &cfg);
        assert!(
            (comp - 10.0).abs() < 1e-5,
            "should clamp to 10.0, got {comp}"
        );
    }
    #[test]
    fn test_scale_compensation_exact_minimum() {
        let cfg = MipSplatConfig::default();
        let scale_3d = 0.003_f32;
        let comp = compute_scale_compensation(scale_3d, 10.0, 1000.0, &cfg);
        assert!(
            (comp - 1.0).abs() < 1e-5,
            "exact minimum should give 1.0, got {comp}"
        );
    }
    #[test]
    fn test_apply_antialiasing_basic() {
        let cfg = MipSplatConfig::default();
        let positions = vec![[0.0_f32, 0.0, 10.0]];
        let scales = vec![[0.0_f32, 0.0, 0.0]];
        let opacities = vec![0.0_f32];
        let (new_ops, comps, stats) =
            apply_antialiasing(&positions, &scales, &opacities, [0.0; 3], 500.0, &cfg);
        assert_eq!(new_ops.len(), 1);
        assert_eq!(comps.len(), 1);
        assert!(
            (new_ops[0] - 0.0).abs() < 1e-5,
            "opacity should be unchanged, got {}",
            new_ops[0]
        );
        assert!((comps[0] - 1.0).abs() < 1e-5);
        assert_eq!(stats.num_gaussians, 1);
        assert_eq!(stats.num_culled, 0);
        assert_eq!(stats.num_faded, 0);
    }
    #[test]
    fn test_apply_antialiasing_far_gaussians_faded() {
        let cfg = MipSplatConfig::default();
        let positions = vec![[0.0_f32, 0.0, 1000.0]];
        let scales = vec![[-11.51_f32, -11.51, -11.51]];
        let opacities = vec![0.0_f32];
        let (new_ops, _comps, stats) =
            apply_antialiasing(&positions, &scales, &opacities, [0.0; 3], 500.0, &cfg);
        assert!(
            new_ops[0].is_infinite() && new_ops[0] < 0.0,
            "culled Gaussian should have NEG_INFINITY opacity, got {}",
            new_ops[0]
        );
        assert_eq!(stats.num_culled, 1);
    }
    #[test]
    fn test_apply_antialiasing_close_gaussians_unchanged() {
        let cfg = MipSplatConfig::default();
        let positions = vec![[0.0_f32, 0.0, 1.0]];
        let scales = vec![[2.0_f32, 2.0, 2.0]];
        let opacities = vec![1.5_f32];
        let (new_ops, comps, stats) =
            apply_antialiasing(&positions, &scales, &opacities, [0.0; 3], 200.0, &cfg);
        assert!(
            (new_ops[0] - 1.5).abs() < 1e-5,
            "opacity unchanged for close large gaussian"
        );
        assert!(
            (comps[0] - 1.0).abs() < 1e-5,
            "no compensation for large close gaussian"
        );
        assert_eq!(stats.num_faded, 0);
        assert_eq!(stats.num_culled, 0);
    }
    #[test]
    fn test_apply_antialiasing_stats_counts() {
        let cfg = MipSplatConfig::default();
        let positions = vec![
            [0.0_f32, 0.0, 1.0],
            [0.0_f32, 0.0, 10.0],
            [0.0_f32, 0.0, 100.0],
        ];
        let scales = vec![
            [0.0_f32, 0.0, 0.0],
            [-1.0_f32, -1.0, -1.0],
            [-10.0_f32, -10.0, -10.0],
        ];
        let opacities = vec![0.0_f32; 3];
        let (_new_ops, _comps, stats) =
            apply_antialiasing(&positions, &scales, &opacities, [0.0; 3], 30.0, &cfg);
        assert_eq!(stats.num_gaussians, 3);
        assert_eq!(stats.num_culled, 1, "one Gaussian should be culled");
        assert_eq!(stats.num_faded, 1, "one Gaussian should be faded");
        assert_eq!(stats.num_gaussians, 3);
    }
    #[test]
    fn test_apply_antialiasing_empty_model() {
        let cfg = MipSplatConfig::default();
        let (new_ops, comps, stats) = apply_antialiasing(&[], &[], &[], [0.0; 3], 500.0, &cfg);
        assert!(new_ops.is_empty());
        assert!(comps.is_empty());
        assert_eq!(stats.num_gaussians, 0);
        assert_eq!(stats.num_culled, 0);
        assert_eq!(stats.num_faded, 0);
        assert_eq!(stats.num_scaled_up, 0);
    }
    #[test]
    fn test_apply_antialiasing_mismatched_lengths() {
        let cfg = MipSplatConfig::default();
        let positions = vec![[0.0_f32; 3]; 3];
        let scales = vec![[0.0_f32; 3]; 2];
        let opacities = vec![0.0_f32; 3];
        let (new_ops, comps, stats) =
            apply_antialiasing(&positions, &scales, &opacities, [0.0; 3], 500.0, &cfg);
        assert!(new_ops.is_empty());
        assert!(comps.is_empty());
        assert_eq!(stats.num_gaussians, 0);
    }
    #[test]
    fn test_sigmoid_logit_roundtrip() {
        for &x in &[-5.0_f32, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0] {
            let p = sigmoid(x);
            let x_recovered = logit(p);
            assert!(
                (x_recovered - x).abs() < 1e-4,
                "roundtrip failed for x={x}: got {x_recovered}"
            );
        }
    }
    #[test]
    fn test_sigmoid_range() {
        for &x in &[
            f32::NEG_INFINITY,
            -100.0_f32,
            -1.0,
            0.0,
            1.0,
            100.0,
            f32::INFINITY,
        ] {
            let p = sigmoid(x);
            assert!((0.0..=1.0).contains(&p), "sigmoid({x}) = {p} out of [0,1]");
        }
    }
    #[test]
    fn test_logit_finite_for_clamped_input() {
        assert!(
            logit(0.0_f32).is_finite(),
            "logit(0) must be finite after clamping"
        );
        assert!(
            logit(1.0_f32).is_finite(),
            "logit(1) must be finite after clamping"
        );
        assert!(logit(0.5_f32).is_finite());
    }
    #[test]
    fn test_opacity_scale_degenerate_config() {
        let cfg = MipSplatConfig {
            opacity_ramp_min_px: 1.0,
            opacity_ramp_max_px: 1.0,
            ..MipSplatConfig::default()
        };
        let below = opacity_scale_from_screen_radius(0.5, &cfg);
        assert_eq!(below, 0.0);
        let at = opacity_scale_from_screen_radius(1.0, &cfg);
        assert!((at - 1.0).abs() < 1e-5, "degenerate ramp at midpoint: {at}");
        let above = opacity_scale_from_screen_radius(2.0, &cfg);
        assert!((above - 1.0).abs() < 1e-5);
    }
    #[test]
    fn test_aa_luminance_pure_red() {
        let l = aa_luminance(1.0, 0.0, 0.0);
        assert!((l - 0.2126_f32).abs() < 1e-6, "pure red luma={l}");
    }
    #[test]
    fn test_aa_luminance_pure_green() {
        let l = aa_luminance(0.0, 1.0, 0.0);
        assert!((l - 0.7152_f32).abs() < 1e-6, "pure green luma={l}");
    }
    #[test]
    fn test_aa_luminance_pure_blue() {
        let l = aa_luminance(0.0, 0.0, 1.0);
        assert!((l - 0.0722_f32).abs() < 1e-6, "pure blue luma={l}");
    }
    #[test]
    fn test_aa_luminance_white() {
        let l = aa_luminance(1.0, 1.0, 1.0);
        assert!((l - 1.0_f32).abs() < 1e-5, "white luma={l}");
    }
    #[test]
    fn test_aa_luminance_black() {
        let l = aa_luminance(0.0, 0.0, 0.0);
        assert!(l.abs() < 1e-8, "black luma={l}");
    }
    #[test]
    fn test_aa_luminance_map_correct_size() {
        let w = 8;
        let h = 6;
        let img = vec![0.5_f32; w * h * 3];
        let luma = aa_luminance_map(&img, w, h).expect("luminance map");
        assert_eq!(luma.len(), w * h);
    }
    #[test]
    fn test_aa_luminance_map_size_mismatch() {
        let img = vec![0.0_f32; 10];
        let result = aa_luminance_map(&img, 4, 4);
        assert!(matches!(result, Err(AaError::SizeMismatch { .. })));
    }
    #[test]
    fn test_aa_luminance_map_values() {
        let w = 4;
        let h = 4;
        let img = vec![0.5_f32; w * h * 3];
        let luma = aa_luminance_map(&img, w, h).expect("ok");
        for &v in &luma {
            assert!((v - 0.5).abs() < 1e-5, "expected 0.5, got {v}");
        }
    }
    #[test]
    fn test_aa_sample_pixel_in_bounds() {
        let w = 4;
        let h = 4;
        let mut img = vec![0.0_f32; w * h * 3];
        img[(w + 2) * 3] = 0.8;
        img[(w + 2) * 3 + 1] = 0.5;
        img[(w + 2) * 3 + 2] = 0.2;
        let px = aa_sample_pixel(&img, w, h, 2, 1);
        assert!((px[0] - 0.8).abs() < 1e-6);
        assert!((px[1] - 0.5).abs() < 1e-6);
        assert!((px[2] - 0.2).abs() < 1e-6);
    }
    #[test]
    fn test_aa_sample_pixel_out_of_bounds_returns_black() {
        let img = vec![1.0_f32; 4 * 4 * 3];
        let px = aa_sample_pixel(&img, 4, 4, -1, 0);
        assert_eq!(px, [0.0, 0.0, 0.0]);
        let px2 = aa_sample_pixel(&img, 4, 4, 4, 2);
        assert_eq!(px2, [0.0, 0.0, 0.0]);
        let px3 = aa_sample_pixel(&img, 4, 4, 0, -1);
        assert_eq!(px3, [0.0, 0.0, 0.0]);
    }
    #[test]
    fn test_aa_bilinear_integer_coords_exact() {
        let w = 4;
        let h = 4;
        let mut img = vec![0.0_f32; w * h * 3];
        img[0] = 0.9;
        img[1] = 0.1;
        img[2] = 0.4;
        let px = aa_bilinear_sample(&img, w, h, 0.0, 0.0);
        assert!((px[0] - 0.9).abs() < 1e-5);
        assert!((px[1] - 0.1).abs() < 1e-5);
        assert!((px[2] - 0.4).abs() < 1e-5);
    }
    #[test]
    fn test_aa_bilinear_midpoint_average() {
        let w = 4;
        let h = 4;
        let mut img = vec![0.0_f32; w * h * 3];
        img[0] = 1.0;
        img[3] = 0.0;
        img[4] = 0.0;
        img[3] = 0.0;
        img[4] = 1.0;
        img[5] = 0.0;
        let px = aa_bilinear_sample(&img, w, h, 0.5, 0.0);
        assert!((px[0] - 0.5).abs() < 1e-5, "r={}", px[0]);
        assert!((px[1] - 0.5).abs() < 1e-5, "g={}", px[1]);
    }
    #[test]
    fn test_aa_edge_map_flat_image_zero() {
        let w = 8;
        let h = 8;
        let luma = vec![0.5_f32; w * h];
        let edges = aa_edge_map(&luma, w, h, 0.01);
        for &e in &edges {
            assert_eq!(e, 0.0, "flat image should have no edges");
        }
    }
    #[test]
    fn test_aa_edge_map_step_edge_nonzero() {
        let w = 8;
        let h = 8;
        let mut luma = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 4..w {
                luma[y * w + x] = 1.0;
            }
        }
        let edges = aa_edge_map(&luma, w, h, 0.01);
        let boundary_edge = edges[3] > 0.0 || edges[4] > 0.0;
        assert!(
            boundary_edge,
            "edge pixel at step boundary should be nonzero"
        );
    }
    #[test]
    fn test_aa_edge_count_flat_zero() {
        let w = 8;
        let h = 8;
        let luma = vec![0.3_f32; w * h];
        let count = aa_edge_count(&luma, w, h, 0.01);
        assert_eq!(count, 0, "flat image → 0 edge pixels");
    }
    #[test]
    fn test_aa_edge_count_checkerboard_nonzero() {
        let w = 8;
        let h = 8;
        let mut luma = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..w {
                luma[y * w + x] = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        let count = aa_edge_count(&luma, w, h, 0.01);
        assert!(count > 0, "checkerboard should have many edge pixels");
    }
    fn make_flat_image(w: usize, h: usize, r: f32, g: f32, b: f32) -> Vec<f32> {
        let mut img = Vec::with_capacity(w * h * 3);
        for _ in 0..(w * h) {
            img.push(r);
            img.push(g);
            img.push(b);
        }
        img
    }
    fn make_checkerboard(w: usize, h: usize) -> Vec<f32> {
        let mut img = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y) % 2 == 0 { 1.0_f32 } else { 0.0_f32 };
                img.push(v);
                img.push(v);
                img.push(v);
            }
        }
        img
    }
    #[test]
    fn test_apply_fxaa_flat_unchanged() {
        let w = 8;
        let h = 8;
        let img = make_flat_image(w, h, 0.5, 0.5, 0.5);
        let cfg = AaConfig::default();
        let out = apply_fxaa(&img, w, h, &cfg).expect("fxaa ok");
        assert_eq!(out.len(), img.len());
        for (a, b) in img.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "flat image should be unchanged by FXAA"
            );
        }
    }
    #[test]
    fn test_apply_fxaa_checkerboard_changes_pixels() {
        let w = 8;
        let h = 8;
        let img = make_checkerboard(w, h);
        let cfg = AaConfig::default();
        let out = apply_fxaa(&img, w, h, &cfg).expect("fxaa ok");
        let changed = img
            .iter()
            .zip(out.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            changed,
            "FXAA should smooth at least one pixel on a checkerboard"
        );
    }
    #[test]
    fn test_apply_fxaa_size_mismatch() {
        let cfg = AaConfig::default();
        let wrong = vec![0.5_f32; 5];
        let result = apply_fxaa(&wrong, 4, 4, &cfg);
        assert!(matches!(result, Err(AaError::SizeMismatch { .. })));
    }
    #[test]
    fn test_apply_fxaa_too_small() {
        let cfg = AaConfig::default();
        let tiny = vec![0.5_f32; 3 * 3 * 3];
        let result = apply_fxaa(&tiny, 3, 3, &cfg);
        assert!(matches!(result, Err(AaError::ImageTooSmall { .. })));
    }
    #[test]
    fn test_apply_fxaa_output_size_same() {
        let w = 8;
        let h = 8;
        let img = make_checkerboard(w, h);
        let cfg = AaConfig::default();
        let out = apply_fxaa(&img, w, h, &cfg).expect("ok");
        assert_eq!(out.len(), w * h * 3);
    }
    #[test]
    fn test_apply_fxaa_all_black() {
        let w = 8;
        let h = 8;
        let img = make_flat_image(w, h, 0.0, 0.0, 0.0);
        let cfg = AaConfig::default();
        let out = apply_fxaa(&img, w, h, &cfg).expect("ok");
        for v in &out {
            assert!(v.abs() < 1e-6, "all-black → all-black after FXAA");
        }
    }
    #[test]
    fn test_apply_fxaa_all_white() {
        let w = 8;
        let h = 8;
        let img = make_flat_image(w, h, 1.0, 1.0, 1.0);
        let cfg = AaConfig::default();
        let out = apply_fxaa(&img, w, h, &cfg).expect("ok");
        for v in &out {
            assert!((v - 1.0).abs() < 1e-6, "all-white → all-white after FXAA");
        }
    }
    #[test]
    fn test_apply_fxaa_minimum_4x4() {
        let w = 4;
        let h = 4;
        let img = make_flat_image(w, h, 0.5, 0.3, 0.7);
        let cfg = AaConfig::default();
        let out = apply_fxaa(&img, w, h, &cfg).expect("4×4 minimum should work");
        assert_eq!(out.len(), w * h * 3);
    }
    #[test]
    fn test_apply_smaa_lite_flat_unchanged() {
        let w = 8;
        let h = 8;
        let img = make_flat_image(w, h, 0.4, 0.6, 0.2);
        let cfg = AaConfig::default();
        let out = apply_smaa_lite(&img, w, h, &cfg).expect("smaa ok");
        for (a, b) in img.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "flat image should not change with SMAA"
            );
        }
    }
    #[test]
    fn test_apply_smaa_lite_checkerboard_smoother() {
        let w = 8;
        let h = 8;
        let img = make_checkerboard(w, h);
        let cfg = AaConfig {
            method: AaMethod::Smaa,
            edge_threshold: 0.01,
            ..AaConfig::default()
        };
        let out = apply_smaa_lite(&img, w, h, &cfg).expect("smaa ok");
        let changed = img
            .iter()
            .zip(out.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            changed,
            "SMAA should change at least one pixel on a checkerboard"
        );
    }
    #[test]
    fn test_apply_smaa_lite_size_mismatch() {
        let cfg = AaConfig::default();
        let wrong = vec![0.0_f32; 7];
        let result = apply_smaa_lite(&wrong, 4, 4, &cfg);
        assert!(matches!(result, Err(AaError::SizeMismatch { .. })));
    }
    #[test]
    fn test_apply_temporal_aa_blend_zero_equals_current() {
        let w = 8;
        let h = 8;
        let current = make_flat_image(w, h, 0.8, 0.3, 0.1);
        let previous = make_flat_image(w, h, 0.0, 0.0, 0.0);
        let out = apply_temporal_aa(&current, &previous, w, h, 0.0).expect("ok");
        for (a, b) in current.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6, "blend=0 should equal current");
        }
    }
    #[test]
    fn test_apply_temporal_aa_blend_one_equals_previous() {
        let w = 8;
        let h = 8;
        let current = make_flat_image(w, h, 0.0, 0.0, 0.0);
        let previous = make_flat_image(w, h, 0.9, 0.5, 0.3);
        let out = apply_temporal_aa(&current, &previous, w, h, 1.0).expect("ok");
        for (a, b) in previous.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6, "blend=1 should equal previous");
        }
    }
    #[test]
    fn test_apply_temporal_aa_blend_half_midpoint() {
        let w = 8;
        let h = 8;
        let current = make_flat_image(w, h, 0.0, 0.0, 0.0);
        let previous = make_flat_image(w, h, 1.0, 1.0, 1.0);
        let out = apply_temporal_aa(&current, &previous, w, h, 0.5).expect("ok");
        for v in &out {
            assert!(
                (v - 0.5).abs() < 1e-5,
                "blend=0.5 should give midpoint, got {v}"
            );
        }
    }
    #[test]
    fn test_apply_temporal_aa_invalid_blend_factor() {
        let w = 8;
        let h = 8;
        let img = make_flat_image(w, h, 0.5, 0.5, 0.5);
        let result = apply_temporal_aa(&img, &img, w, h, 1.5);
        assert!(matches!(result, Err(AaError::InvalidParam(_))));
    }
    #[test]
    fn test_apply_temporal_aa_negative_blend_factor() {
        let w = 8;
        let h = 8;
        let img = make_flat_image(w, h, 0.5, 0.5, 0.5);
        let result = apply_temporal_aa(&img, &img, w, h, -0.1);
        assert!(matches!(result, Err(AaError::InvalidParam(_))));
    }
    #[test]
    fn test_apply_supersampling_aa_factor2_dims() {
        let w = 8;
        let h = 8;
        let img = make_flat_image(w, h, 0.5, 0.5, 0.5);
        let out = apply_supersampling_aa(&img, w, h, 2).expect("ok");
        assert_eq!(out.len(), (w / 2) * (h / 2) * 3);
    }
    #[test]
    fn test_apply_supersampling_aa_factor4_dims() {
        let w = 8;
        let h = 8;
        let img = make_flat_image(w, h, 0.5, 0.5, 0.5);
        let out = apply_supersampling_aa(&img, w, h, 4).expect("ok");
        assert_eq!(out.len(), (w / 4) * (h / 4) * 3);
    }
    #[test]
    fn test_apply_supersampling_aa_invalid_factor() {
        let w = 8;
        let h = 8;
        let img = make_flat_image(w, h, 0.5, 0.5, 0.5);
        let result = apply_supersampling_aa(&img, w, h, 3);
        assert!(matches!(result, Err(AaError::InvalidParam(_))));
    }
    #[test]
    fn test_apply_supersampling_aa_flat_image_values_preserved() {
        let w = 8;
        let h = 8;
        let img = make_flat_image(w, h, 0.7, 0.3, 0.1);
        let out = apply_supersampling_aa(&img, w, h, 2).expect("ok");
        for i in (0..out.len()).step_by(3) {
            assert!((out[i] - 0.7).abs() < 1e-5, "r channel should be preserved");
            assert!(
                (out[i + 1] - 0.3).abs() < 1e-5,
                "g channel should be preserved"
            );
            assert!(
                (out[i + 2] - 0.1).abs() < 1e-5,
                "b channel should be preserved"
            );
        }
    }
    #[test]
    fn test_apply_image_aa_dispatches_fxaa() {
        let w = 8;
        let h = 8;
        let img = make_flat_image(w, h, 0.5, 0.5, 0.5);
        let cfg = AaConfig {
            method: AaMethod::Fxaa,
            ..AaConfig::default()
        };
        let out = apply_image_aa(&img, w, h, &cfg).expect("fxaa dispatch ok");
        assert_eq!(out.len(), img.len());
    }
    #[test]
    fn test_apply_image_aa_dispatches_smaa() {
        let w = 8;
        let h = 8;
        let img = make_flat_image(w, h, 0.5, 0.5, 0.5);
        let cfg = AaConfig {
            method: AaMethod::Smaa,
            ..AaConfig::default()
        };
        let out = apply_image_aa(&img, w, h, &cfg).expect("smaa dispatch ok");
        assert_eq!(out.len(), img.len());
    }
    #[test]
    fn test_apply_image_aa_dispatches_temporal() {
        let w = 8;
        let h = 8;
        let img = make_flat_image(w, h, 0.5, 0.5, 0.5);
        let cfg = AaConfig {
            method: AaMethod::Temporal { blend_factor: 0.5 },
            ..AaConfig::default()
        };
        let out = apply_image_aa(&img, w, h, &cfg).expect("temporal dispatch ok");
        assert_eq!(out.len(), img.len());
    }
    #[test]
    fn test_apply_image_aa_dispatches_supersampling() {
        let w = 8;
        let h = 8;
        let img = make_flat_image(w, h, 0.5, 0.5, 0.5);
        let cfg = AaConfig {
            method: AaMethod::Supersampling { factor: 2 },
            ..AaConfig::default()
        };
        let out = apply_image_aa(&img, w, h, &cfg).expect("supersampling dispatch ok");
        assert_eq!(out.len(), (w / 2) * (h / 2) * 3);
    }
    #[test]
    fn test_aa_quality_estimate_identical_images() {
        let w = 8;
        let h = 8;
        let img = make_checkerboard(w, h);
        let stats = aa_quality_estimate(&img, &img, w, h).expect("ok");
        assert!((stats.mean_difference).abs() < 1e-6);
        assert!((stats.max_difference).abs() < 1e-6);
        assert!((stats.smoothing_ratio).abs() < 1e-6);
    }
    #[test]
    fn test_aa_quality_estimate_smoother_image() {
        let w = 8;
        let h = 8;
        let orig = make_checkerboard(w, h);
        let smooth = make_flat_image(w, h, 0.5, 0.5, 0.5);
        let stats = aa_quality_estimate(&orig, &smooth, w, h).expect("ok");
        assert!(stats.mean_difference > 0.0, "smoothed image should differ");
        assert!(stats.max_difference > 0.0);
        assert!(
            stats.smoothing_ratio >= 0.0,
            "smoothing_ratio should be non-negative, got {}",
            stats.smoothing_ratio
        );
    }
    #[test]
    fn test_aa_quality_estimate_size_mismatch() {
        let w = 8;
        let h = 8;
        let img = make_flat_image(w, h, 0.5, 0.5, 0.5);
        let wrong = vec![0.5_f32; 10];
        let result = aa_quality_estimate(&img, &wrong, w, h);
        assert!(matches!(result, Err(AaError::SizeMismatch { .. })));
    }
    #[test]
    fn test_format_aa_config_non_empty() {
        let cfg = AaConfig::default();
        let s = format_aa_config(&cfg);
        assert!(
            !s.is_empty(),
            "format_aa_config should return non-empty string"
        );
        assert!(s.contains("FXAA"), "should mention the method");
    }
    #[test]
    fn test_format_aa_stats_non_empty() {
        let stats = AaStats {
            edge_pixels_original: 10,
            edge_pixels_after: 5,
            smoothing_ratio: 0.5,
            mean_difference: 0.01,
            max_difference: 0.1,
        };
        let s = format_aa_stats(&stats);
        assert!(
            !s.is_empty(),
            "format_aa_stats should return non-empty string"
        );
    }
    #[test]
    fn test_format_aa_config_smaa() {
        let cfg = AaConfig {
            method: AaMethod::Smaa,
            ..AaConfig::default()
        };
        let s = format_aa_config(&cfg);
        assert!(s.contains("SMAA"), "should mention SMAA");
    }
    #[test]
    fn test_format_aa_config_supersampling() {
        let cfg = AaConfig {
            method: AaMethod::Supersampling { factor: 4 },
            ..AaConfig::default()
        };
        let s = format_aa_config(&cfg);
        assert!(s.contains("Supersampling"), "should mention Supersampling");
    }
    #[test]
    fn test_format_aa_config_temporal() {
        let cfg = AaConfig {
            method: AaMethod::Temporal { blend_factor: 0.9 },
            ..AaConfig::default()
        };
        let s = format_aa_config(&cfg);
        assert!(s.contains("Temporal"), "should mention Temporal");
    }
}
