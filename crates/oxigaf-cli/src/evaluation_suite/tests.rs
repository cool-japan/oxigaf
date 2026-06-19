//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    /// Create a uniform RGB image of given dimensions and value.
    fn uniform_rgb(width: usize, height: usize, value: f32) -> Vec<f32> {
        vec![value; width * height * 3]
    }
    /// Create a 2×2 test image with distinct pixels.
    fn test_img_2x2() -> Vec<f32> {
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0, 0.1, 0.2]
    }
    /// A reasonably large (48×48) uniform image for MS-SSIM tests.
    fn uniform_48x48(value: f32) -> Vec<f32> {
        vec![value; 48 * 48 * 3]
    }
    #[test]
    fn test_metric_kind_names_non_empty() {
        let kinds = [
            EvalMetricKind::Psnr,
            EvalMetricKind::Ssim,
            EvalMetricKind::LpipsApprox,
            EvalMetricKind::Mae,
            EvalMetricKind::Rmse,
            EvalMetricKind::SsimMs,
        ];
        for k in &kinds {
            assert!(!k.name().is_empty(), "{k:?} should have non-empty name");
        }
    }
    #[test]
    fn test_metric_kind_higher_is_better_psnr() {
        assert!(EvalMetricKind::Psnr.higher_is_better());
    }
    #[test]
    fn test_metric_kind_higher_is_better_ssim() {
        assert!(EvalMetricKind::Ssim.higher_is_better());
    }
    #[test]
    fn test_metric_kind_higher_is_better_ssim_ms() {
        assert!(EvalMetricKind::SsimMs.higher_is_better());
    }
    #[test]
    fn test_metric_kind_lower_is_better_lpips() {
        assert!(!EvalMetricKind::LpipsApprox.higher_is_better());
    }
    #[test]
    fn test_metric_kind_lower_is_better_mae() {
        assert!(!EvalMetricKind::Mae.higher_is_better());
    }
    #[test]
    fn test_metric_kind_lower_is_better_rmse() {
        assert!(!EvalMetricKind::Rmse.higher_is_better());
    }
    #[test]
    fn test_metric_kind_display() {
        let s = format!("{}", EvalMetricKind::Psnr);
        assert!(!s.is_empty());
    }
    #[test]
    fn test_psnr_identical_images_infinity() {
        let img = test_img_2x2();
        let psnr = eval_psnr(&img, &img).expect("psnr should not error");
        assert!(psnr.is_infinite(), "identical images → PSNR should be ∞");
    }
    #[test]
    fn test_psnr_known_mse() {
        let pred = vec![0.0_f32; 4];
        let gt = vec![0.1_f32; 4];
        let psnr = eval_psnr(&pred, &gt).expect("psnr ok");
        let expected = 10.0_f32 * (100.0_f32).log10();
        assert!(
            (psnr - expected).abs() < 0.01,
            "PSNR mismatch: {psnr} vs {expected}"
        );
    }
    #[test]
    fn test_psnr_empty_error() {
        let result = eval_psnr(&[], &[]);
        assert!(result.is_err(), "empty slice should return error");
    }
    #[test]
    fn test_psnr_mismatched_lengths_error() {
        let pred = vec![0.5_f32; 4];
        let gt = vec![0.5_f32; 3];
        let result = eval_psnr(&pred, &gt);
        assert!(matches!(result, Err(EvalError::DimensionMismatch { .. })));
    }
    #[test]
    fn test_psnr_different_images_finite() {
        let pred = uniform_rgb(4, 4, 0.0);
        let gt = uniform_rgb(4, 4, 1.0);
        let psnr = eval_psnr(&pred, &gt).expect("psnr ok");
        assert!(
            psnr.is_finite() && psnr < 1.0,
            "completely different → low finite PSNR"
        );
    }
    #[test]
    fn test_mae_identical() {
        let img = test_img_2x2();
        let mae = eval_mae(&img, &img).expect("mae ok");
        assert_eq!(mae, 0.0, "identical → MAE = 0");
    }
    #[test]
    fn test_mae_known_diff() {
        let pred = vec![0.0_f32; 4];
        let gt = vec![0.2_f32; 4];
        let mae = eval_mae(&pred, &gt).expect("mae ok");
        assert!((mae - 0.2).abs() < 1e-6, "MAE should be 0.2, got {mae}");
    }
    #[test]
    fn test_mae_empty_error() {
        assert!(eval_mae(&[], &[]).is_err());
    }
    #[test]
    fn test_mae_mismatched_error() {
        let result = eval_mae(&[0.1_f32; 4], &[0.1_f32; 5]);
        assert!(matches!(result, Err(EvalError::DimensionMismatch { .. })));
    }
    #[test]
    fn test_rmse_identical() {
        let img = test_img_2x2();
        let rmse = eval_rmse(&img, &img).expect("rmse ok");
        assert_eq!(rmse, 0.0, "identical → RMSE = 0");
    }
    #[test]
    fn test_rmse_known_diff() {
        let pred = vec![0.0_f32; 6];
        let gt = vec![0.3_f32; 6];
        let rmse = eval_rmse(&pred, &gt).expect("rmse ok");
        assert!((rmse - 0.3).abs() < 1e-6, "RMSE should be 0.3, got {rmse}");
    }
    #[test]
    fn test_rmse_empty_error() {
        assert!(eval_rmse(&[], &[]).is_err());
    }
    #[test]
    fn test_rmse_mismatched_error() {
        let result = eval_rmse(&[0.0_f32; 4], &[0.0_f32; 3]);
        assert!(matches!(result, Err(EvalError::DimensionMismatch { .. })));
    }
    #[test]
    fn test_ssim_identical_large() {
        let img = uniform_rgb(16, 16, 0.5);
        let ssim = eval_ssim(&img, &img, 16, 16).expect("ssim ok");
        assert!(
            ssim > 0.99,
            "identical uniform 16×16 → SSIM ≈ 1.0, got {ssim}"
        );
    }
    #[test]
    fn test_ssim_different_images_less_than_one() {
        let pred = uniform_rgb(16, 16, 0.0);
        let gt = uniform_rgb(16, 16, 1.0);
        let ssim = eval_ssim(&pred, &gt, 16, 16).expect("ssim ok");
        assert!(ssim < 1.0, "very different → SSIM < 1.0");
    }
    #[test]
    fn test_ssim_tiny_image_in_range() {
        let pred = uniform_rgb(4, 4, 0.3);
        let gt = uniform_rgb(4, 4, 0.7);
        let ssim = eval_ssim(&pred, &gt, 4, 4).expect("ssim ok");
        assert!(
            (-1.0..=1.0).contains(&ssim),
            "SSIM must be in [-1, 1], got {ssim}"
        );
    }
    #[test]
    fn test_ssim_length_mismatch() {
        let result = eval_ssim(&[0.5_f32; 12], &[0.5_f32; 9], 2, 2);
        assert!(result.is_err());
    }
    #[test]
    fn test_ssim_buffer_length_mismatch() {
        let result = eval_ssim(&[0.5_f32; 12], &[0.5_f32; 12], 3, 2);
        assert!(result.is_err());
    }
    #[test]
    fn test_ssim_ms_identical() {
        let img = uniform_48x48(0.5);
        let ssim_ms = eval_ssim_ms(&img, &img, 48, 48).expect("ms-ssim ok");
        assert!(
            ssim_ms > 0.99,
            "identical 48×48 → MS-SSIM ≈ 1.0, got {ssim_ms}"
        );
    }
    #[test]
    fn test_ssim_ms_different_images() {
        let pred = uniform_48x48(0.0);
        let gt = uniform_48x48(1.0);
        let ssim_ms = eval_ssim_ms(&pred, &gt, 48, 48).expect("ms-ssim ok");
        assert!(ssim_ms < 1.0, "different images → MS-SSIM < 1.0");
    }
    #[test]
    fn test_ssim_ms_3_scales() {
        let pred = uniform_48x48(0.3);
        let gt = uniform_48x48(0.6);
        let ssim_ms = eval_ssim_ms(&pred, &gt, 48, 48).expect("ms-ssim ok");
        assert!(ssim_ms.is_finite(), "MS-SSIM should be finite");
        assert!((-1.0..=1.0).contains(&ssim_ms));
    }
    #[test]
    fn test_ssim_ms_length_mismatch() {
        let result = eval_ssim_ms(&[0.5_f32; 12], &[0.5_f32; 9], 4, 4);
        assert!(result.is_err());
    }
    #[test]
    fn test_lpips_identical_approx_zero() {
        let img = uniform_rgb(8, 8, 0.5);
        let lpips = eval_lpips_approx(&img, &img, 8, 8).expect("lpips ok");
        assert!(lpips < 1e-6, "identical → LPIPS ≈ 0, got {lpips}");
    }
    #[test]
    fn test_lpips_different_images_positive() {
        let w = 8;
        let h = 8;
        let mut pred = vec![0.0_f32; w * h * 3];
        let gt = vec![0.5_f32; w * h * 3];
        for y in 0..h {
            for x in 0..w / 2 {
                for c in 0..3 {
                    pred[(y * w + x) * 3 + c] = 1.0;
                }
            }
        }
        let lpips = eval_lpips_approx(&pred, &gt, w, h).expect("lpips ok");
        assert!(lpips > 0.0, "different images → LPIPS > 0, got {lpips}");
    }
    #[test]
    fn test_lpips_length_mismatch() {
        let result = eval_lpips_approx(&[0.5_f32; 12], &[0.5_f32; 6], 2, 2);
        assert!(result.is_err());
    }
    #[test]
    fn test_single_view_identical() {
        let img = uniform_rgb(16, 16, 0.5);
        let result = eval_single_view(&img, &img, 16, 16, "view_0").expect("ok");
        assert!(result.psnr.is_infinite());
        assert!(result.ssim > 0.99);
        assert_eq!(result.mae, 0.0);
        assert_eq!(result.rmse, 0.0);
        assert!(result.lpips_approx < 1e-6);
        assert_eq!(result.view_id, "view_0");
    }
    #[test]
    fn test_single_view_dimension_mismatch() {
        let pred = uniform_rgb(4, 4, 0.5);
        let gt = uniform_rgb(8, 8, 0.5);
        let result = eval_single_view(&pred, &gt, 4, 4, "v0");
        assert!(matches!(result, Err(EvalError::DimensionMismatch { .. })));
    }
    #[test]
    fn test_single_view_metrics_populated() {
        let pred = uniform_rgb(16, 16, 0.3);
        let gt = uniform_rgb(16, 16, 0.7);
        let result = eval_single_view(&pred, &gt, 16, 16, "v1").expect("ok");
        assert!(result.psnr.is_finite());
        assert!(result.ssim.is_finite());
        assert!(result.mae > 0.0);
        assert!(result.rmse > 0.0);
        assert_eq!(result.width, 16);
        assert_eq!(result.height, 16);
    }
    #[test]
    fn test_gaussian_kernel_11_length() {
        let k = eval_gaussian_kernel_11();
        assert_eq!(k.len(), 121);
    }
    #[test]
    fn test_gaussian_kernel_11_sums_to_one() {
        let k = eval_gaussian_kernel_11();
        let total: f32 = k.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "kernel sum should be ≈1.0, got {total}"
        );
    }
    #[test]
    fn test_gaussian_kernel_11_positive() {
        let k = eval_gaussian_kernel_11();
        assert!(
            k.iter().all(|&v| v > 0.0),
            "all kernel values should be positive"
        );
    }
    #[test]
    fn test_gaussian_kernel_11_center_max() {
        let k = eval_gaussian_kernel_11();
        let center = k[5 * 11 + 5];
        assert!(
            k.iter().all(|&v| v <= center + 1e-8),
            "center should be the maximum"
        );
    }
    #[test]
    fn test_convolve_length_preserved() {
        let img = vec![0.5_f32; 6 * 4];
        let k = eval_gaussian_kernel_11();
        let out = eval_convolve(&img, 6, 4, &k, 11);
        assert_eq!(out.len(), 6 * 4, "output length must match input");
    }
    #[test]
    fn test_convolve_uniform_with_gaussian() {
        let value = 0.7_f32;
        let img = vec![value; 24 * 24];
        let k = eval_gaussian_kernel_11();
        let out = eval_convolve(&img, 24, 24, &k, 11);
        let interior = out[12 * 24 + 12];
        assert!(
            (interior - value).abs() < 1e-5,
            "uniform image convolved → same value, got {interior}"
        );
    }
    #[test]
    fn test_convolve_identity_kernel() {
        let img = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let identity = [1.0_f32];
        let out = eval_convolve(&img, 3, 3, &identity, 1);
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
    #[test]
    fn test_downsample_2x_even_dimensions() {
        let img = uniform_rgb(8, 8, 0.5);
        let (_, w, h) = eval_downsample_2x(&img, 8, 8);
        assert_eq!(w, 4);
        assert_eq!(h, 4);
    }
    #[test]
    fn test_downsample_2x_odd_width() {
        let img = uniform_rgb(7, 6, 0.5);
        let (_, w, h) = eval_downsample_2x(&img, 7, 6);
        assert_eq!(w, 4);
        assert_eq!(h, 3);
    }
    #[test]
    fn test_downsample_2x_odd_height() {
        let img = uniform_rgb(6, 7, 0.5);
        let (_, w, h) = eval_downsample_2x(&img, 6, 7);
        assert_eq!(w, 3);
        assert_eq!(h, 4);
    }
    #[test]
    fn test_downsample_2x_odd_both() {
        let img = uniform_rgb(5, 5, 0.5);
        let (_, w, h) = eval_downsample_2x(&img, 5, 5);
        assert_eq!(w, 3);
        assert_eq!(h, 3);
    }
    #[test]
    fn test_downsample_2x_uniform_image() {
        let value = 0.4_f32;
        let img = uniform_rgb(8, 8, value);
        let (out, w, h) = eval_downsample_2x(&img, 8, 8);
        assert_eq!(out.len(), w * h * 3);
        for v in &out {
            assert!(
                (v - value).abs() < 1e-5,
                "uniform → same value after downsampling, got {v}"
            );
        }
    }
    #[test]
    fn test_downsample_2x_output_length() {
        let img = uniform_rgb(10, 6, 0.3);
        let (out, w, h) = eval_downsample_2x(&img, 10, 6);
        assert_eq!(out.len(), w * h * 3);
    }
    #[test]
    fn test_sobel_uniform_image_all_zeros() {
        let w = 8usize;
        let h = 8usize;
        let img = uniform_rgb(w, h, 0.5);
        let mag = eval_sobel(&img, w, h);
        let mut interior_max = 0.0_f32;
        for row in 1..h - 1 {
            for col in 1..w - 1 {
                let v = mag[row * w + col];
                if v > interior_max {
                    interior_max = v;
                }
            }
        }
        assert!(
            interior_max < 1e-5,
            "uniform image → zero Sobel magnitude (interior), got {interior_max}"
        );
    }
    #[test]
    fn test_sobel_output_length() {
        let img = uniform_rgb(6, 5, 0.5);
        let mag = eval_sobel(&img, 6, 5);
        assert_eq!(
            mag.len(),
            6 * 5,
            "Sobel output is grayscale (single channel)"
        );
    }
    #[test]
    fn test_sobel_detects_edge() {
        let w = 8;
        let h = 8;
        let mut img = vec![0.0_f32; w * h * 3];
        for y in 0..h {
            for x in w / 2..w {
                for c in 0..3 {
                    img[(y * w + x) * 3 + c] = 1.0;
                }
            }
        }
        let mag = eval_sobel(&img, w, h);
        let edge_col = w / 2;
        let max_at_edge = (0..h)
            .map(|y| mag[y * w + edge_col])
            .fold(0.0_f32, f32::max);
        assert!(
            max_at_edge > 0.1,
            "edge column should have high Sobel magnitude, got {max_at_edge}"
        );
    }
    #[test]
    fn test_eval_config_default_fields() {
        let cfg = EvalConfig::default();
        assert!(!cfg.metrics.is_empty());
        assert_eq!(cfg.n_worst_views, 5);
        assert_eq!(cfg.n_best_views, 5);
        assert!(!cfg.save_per_view_results);
    }
    #[test]
    fn test_eval_suite_empty_error() {
        let cfg = EvalConfig::default();
        let result = eval_suite(&[], &cfg);
        assert!(matches!(result, Err(EvalError::EmptyTestSet)));
    }
    #[test]
    fn test_eval_suite_single_item_mean_equals_view() {
        let img = uniform_rgb(16, 16, 0.5);
        let item = EvalTestItem {
            view_id: "view_0".to_string(),
            pred: img.clone(),
            gt: img,
            width: 16,
            height: 16,
        };
        let cfg = EvalConfig::default();
        let result = eval_suite(&[item], &cfg).expect("ok");
        assert!(
            result.mean_psnr.is_infinite(),
            "single identical view → mean PSNR = ∞"
        );
        assert_eq!(result.n_views, 1);
    }
    #[test]
    fn test_eval_suite_multiple_items_mean_psnr() {
        let cfg = EvalConfig::default();
        let pred1 = vec![0.0_f32; 16 * 16 * 3];
        let gt1 = vec![0.1_f32; 16 * 16 * 3];
        let pred2 = pred1.clone();
        let gt2 = vec![0.1_f32; 16 * 16 * 3];
        let items = vec![
            EvalTestItem {
                view_id: "v0".to_string(),
                pred: pred1,
                gt: gt1,
                width: 16,
                height: 16,
            },
            EvalTestItem {
                view_id: "v1".to_string(),
                pred: pred2,
                gt: gt2,
                width: 16,
                height: 16,
            },
        ];
        let result = eval_suite(&items, &cfg).expect("ok");
        assert_eq!(result.n_views, 2);
        assert!(result.mean_psnr.is_finite());
        let expected = eval_psnr(&[0.0_f32; 16 * 16 * 3], &[0.1_f32; 16 * 16 * 3]).expect("ok");
        assert!(
            (result.mean_psnr - expected).abs() < 0.01,
            "mean should equal per-view PSNR"
        );
    }
    #[test]
    fn test_eval_suite_n_views_correct() {
        let cfg = EvalConfig::default();
        let items: Vec<EvalTestItem> = (0..7)
            .map(|i| {
                let img = uniform_rgb(16, 16, i as f32 / 10.0);
                EvalTestItem {
                    view_id: format!("view_{i}"),
                    pred: img.clone(),
                    gt: img,
                    width: 16,
                    height: 16,
                }
            })
            .collect();
        let result = eval_suite(&items, &cfg).expect("ok");
        assert_eq!(result.n_views, 7);
    }
    #[test]
    fn test_eval_suite_worst_best_views() {
        let cfg = EvalConfig {
            n_worst_views: 1,
            n_best_views: 1,
            ..EvalConfig::default()
        };
        let pred_bad = uniform_rgb(16, 16, 0.0);
        let gt_bad = uniform_rgb(16, 16, 1.0);
        let img_good = uniform_rgb(16, 16, 0.5);
        let items = vec![
            EvalTestItem {
                view_id: "view_a".to_string(),
                pred: pred_bad,
                gt: gt_bad,
                width: 16,
                height: 16,
            },
            EvalTestItem {
                view_id: "view_b".to_string(),
                pred: img_good.clone(),
                gt: img_good,
                width: 16,
                height: 16,
            },
        ];
        let result = eval_suite(&items, &cfg).expect("ok");
        assert_eq!(result.worst_views.len(), 1);
        assert_eq!(result.best_views.len(), 1);
        assert_eq!(
            result.worst_views[0], "view_a",
            "view_a should be the worst"
        );
        assert_eq!(result.best_views[0], "view_b", "view_b should be the best");
    }
    #[test]
    fn test_eval_suite_is_worst_flag() {
        let cfg = EvalConfig {
            n_worst_views: 1,
            n_best_views: 1,
            ..EvalConfig::default()
        };
        let pred_bad = uniform_rgb(16, 16, 0.0);
        let gt_bad = uniform_rgb(16, 16, 1.0);
        let img_good = uniform_rgb(16, 16, 0.5);
        let items = vec![
            EvalTestItem {
                view_id: "bad".to_string(),
                pred: pred_bad,
                gt: gt_bad,
                width: 16,
                height: 16,
            },
            EvalTestItem {
                view_id: "good".to_string(),
                pred: img_good.clone(),
                gt: img_good,
                width: 16,
                height: 16,
            },
        ];
        let result = eval_suite(&items, &cfg).expect("ok");
        let bad_view = result
            .per_view
            .iter()
            .find(|v| v.view_id == "bad")
            .expect("found");
        let good_view = result
            .per_view
            .iter()
            .find(|v| v.view_id == "good")
            .expect("found");
        assert!(bad_view.is_worst);
        assert!(!bad_view.is_best);
        assert!(good_view.is_best);
        assert!(!good_view.is_worst);
    }
    #[test]
    fn test_eval_compare_identical_results() {
        let img = uniform_rgb(16, 16, 0.5);
        let item = EvalTestItem {
            view_id: "v0".to_string(),
            pred: img.clone(),
            gt: img,
            width: 16,
            height: 16,
        };
        let cfg = EvalConfig::default();
        let result = eval_suite(&[item], &cfg).expect("ok");
        let cmp = eval_compare(&result, &result).expect("compare ok");
        assert!((cmp.delta_psnr).abs() < 1e-6);
        assert!((cmp.delta_ssim).abs() < 1e-6);
        assert_eq!(cmp.n_views_improved, 0);
        assert_eq!(cmp.n_views_degraded, 0);
    }
    #[test]
    fn test_eval_compare_better_candidate() {
        let cfg = EvalConfig::default();
        let pred_bad = uniform_rgb(16, 16, 0.0);
        let gt = uniform_rgb(16, 16, 1.0);
        let pred_good = uniform_rgb(16, 16, 0.9);
        let baseline_items = vec![EvalTestItem {
            view_id: "v0".to_string(),
            pred: pred_bad,
            gt: gt.clone(),
            width: 16,
            height: 16,
        }];
        let candidate_items = vec![EvalTestItem {
            view_id: "v0".to_string(),
            pred: pred_good,
            gt,
            width: 16,
            height: 16,
        }];
        let baseline = eval_suite(&baseline_items, &cfg).expect("ok");
        let candidate = eval_suite(&candidate_items, &cfg).expect("ok");
        let cmp = eval_compare(&baseline, &candidate).expect("ok");
        assert!(cmp.is_candidate_better, "candidate should be better");
        assert!(cmp.delta_psnr > 0.0);
    }
    #[test]
    fn test_eval_compare_mismatched_n_views() {
        let cfg = EvalConfig::default();
        let img = uniform_rgb(16, 16, 0.5);
        let item1 = EvalTestItem {
            view_id: "v0".to_string(),
            pred: img.clone(),
            gt: img.clone(),
            width: 16,
            height: 16,
        };
        let _item2 = EvalTestItem {
            view_id: "v1".to_string(),
            pred: img.clone(),
            gt: img.clone(),
            width: 16,
            height: 16,
        };
        let r1 = eval_suite(&[item1], &cfg).expect("ok");
        let r2_items = vec![
            EvalTestItem {
                view_id: "v0".to_string(),
                pred: img.clone(),
                gt: img.clone(),
                width: 16,
                height: 16,
            },
            EvalTestItem {
                view_id: "v1".to_string(),
                pred: img.clone(),
                gt: img,
                width: 16,
                height: 16,
            },
        ];
        let r2 = eval_suite(&r2_items, &cfg).expect("ok");
        let cmp = eval_compare(&r1, &r2);
        assert!(cmp.is_ok(), "mismatched view counts should not error");
    }
    #[test]
    fn test_psnr_histogram_bin_count() {
        let cfg = EvalConfig::default();
        let items: Vec<EvalTestItem> = (0..10)
            .map(|i| {
                let pred = uniform_rgb(16, 16, i as f32 / 10.0);
                let gt = uniform_rgb(16, 16, 0.5);
                EvalTestItem {
                    view_id: format!("v{i}"),
                    pred,
                    gt,
                    width: 16,
                    height: 16,
                }
            })
            .collect();
        let result = eval_suite(&items, &cfg).expect("ok");
        let n_bins = 5;
        let (edges, counts) = eval_psnr_histogram(&result, n_bins);
        assert_eq!(
            edges.len(),
            n_bins + 1,
            "edges should have n_bins+1 entries"
        );
        assert_eq!(counts.len(), n_bins, "counts should have n_bins entries");
    }
    #[test]
    fn test_psnr_histogram_counts_sum_to_n_views() {
        let cfg = EvalConfig::default();
        let items: Vec<EvalTestItem> = (0..8)
            .map(|i| {
                let pred = uniform_rgb(16, 16, i as f32 / 8.0);
                let gt = uniform_rgb(16, 16, 0.5);
                EvalTestItem {
                    view_id: format!("v{i}"),
                    pred,
                    gt,
                    width: 16,
                    height: 16,
                }
            })
            .collect();
        let result = eval_suite(&items, &cfg).expect("ok");
        let (_, counts) = eval_psnr_histogram(&result, 4);
        let total: usize = counts.iter().sum();
        assert_eq!(
            total, result.n_views,
            "histogram counts must sum to n_views"
        );
    }
    #[test]
    fn test_psnr_percentiles_length() {
        let img = uniform_rgb(16, 16, 0.5);
        let item = EvalTestItem {
            view_id: "v0".to_string(),
            pred: img.clone(),
            gt: img,
            width: 16,
            height: 16,
        };
        let cfg = EvalConfig::default();
        let result = eval_suite(&[item], &cfg).expect("ok");
        let percs = eval_psnr_percentiles(&result);
        assert_eq!(percs.len(), 5, "should return 5 percentile values");
    }
    #[test]
    fn test_psnr_percentiles_ordered() {
        let cfg = EvalConfig::default();
        let items: Vec<EvalTestItem> = (0..20)
            .map(|i| {
                let pred = uniform_rgb(16, 16, i as f32 / 20.0);
                let gt = uniform_rgb(16, 16, 0.5);
                EvalTestItem {
                    view_id: format!("v{i}"),
                    pred,
                    gt,
                    width: 16,
                    height: 16,
                }
            })
            .collect();
        let result = eval_suite(&items, &cfg).expect("ok");
        let [p5, p25, p50, p75, p95] = eval_psnr_percentiles(&result);
        assert!(p5 <= p25, "P5 <= P25");
        assert!(p25 <= p50, "P25 <= P50");
        assert!(p50 <= p75, "P50 <= P75");
        assert!(p75 <= p95, "P75 <= P95");
    }
    #[test]
    fn test_psnr_percentiles_p50_approx_median() {
        let cfg = EvalConfig::default();
        let preds_gts: Vec<(f32, f32)> = vec![
            (0.0, 0.1),
            (0.1, 0.2),
            (0.2, 0.3),
            (0.3, 0.4),
            (0.4, 0.5),
            (0.5, 0.6),
            (0.6, 0.7),
            (0.7, 0.8),
            (0.8, 0.9),
            (0.9, 1.0),
        ];
        let items: Vec<EvalTestItem> = preds_gts
            .iter()
            .enumerate()
            .map(|(i, &(p, g))| {
                let pred = uniform_rgb(16, 16, p);
                let gt = uniform_rgb(16, 16, g);
                EvalTestItem {
                    view_id: format!("v{i}"),
                    pred,
                    gt,
                    width: 16,
                    height: 16,
                }
            })
            .collect();
        let result = eval_suite(&items, &cfg).expect("ok");
        let [_, _, p50, _, _] = eval_psnr_percentiles(&result);
        assert!(p50.is_finite(), "P50 should be finite");
        assert!(p50 >= result.min_psnr - 0.01);
        assert!(p50 <= result.max_psnr + 0.01);
    }
    #[test]
    fn test_format_view_result_non_empty() {
        let result = ViewEvalResult {
            view_id: "test_view".to_string(),
            psnr: 32.5,
            ssim: 0.95,
            lpips_approx: 0.02,
            mae: 0.01,
            rmse: 0.02,
            ssim_ms: 0.96,
            width: 512,
            height: 512,
            is_worst: false,
            is_best: false,
        };
        let s = eval_format_view_result(&result);
        assert!(!s.is_empty());
        assert!(s.contains("32.50"), "should contain PSNR value");
    }
    #[test]
    fn test_format_suite_result_contains_mean_psnr() {
        let img = uniform_rgb(16, 16, 0.4);
        let gt = uniform_rgb(16, 16, 0.6);
        let items = vec![EvalTestItem {
            view_id: "v0".to_string(),
            pred: img,
            gt,
            width: 16,
            height: 16,
        }];
        let cfg = EvalConfig::default();
        let result = eval_suite(&items, &cfg).expect("ok");
        let s = eval_format_suite_result(&result);
        assert!(!s.is_empty());
        assert!(s.contains("PSNR"), "should mention PSNR");
        assert!(s.contains("SSIM"), "should mention SSIM");
    }
    #[test]
    fn test_format_comparison_shows_delta() {
        let img = uniform_rgb(16, 16, 0.5);
        let item = EvalTestItem {
            view_id: "v0".to_string(),
            pred: img.clone(),
            gt: img,
            width: 16,
            height: 16,
        };
        let cfg = EvalConfig::default();
        let result = eval_suite(&[item], &cfg).expect("ok");
        let cmp = eval_compare(&result, &result).expect("ok");
        let s = eval_format_comparison(&cmp);
        assert!(!s.is_empty());
        assert!(
            s.contains("ΔPSNR")
                || s.contains("delta")
                || s.contains("Delta")
                || s.to_lowercase().contains("psnr")
        );
    }
    #[test]
    fn test_format_view_result_infinity_psnr() {
        let result = ViewEvalResult {
            view_id: "identical".to_string(),
            psnr: f32::INFINITY,
            ssim: 1.0,
            lpips_approx: 0.0,
            mae: 0.0,
            rmse: 0.0,
            ssim_ms: 1.0,
            width: 16,
            height: 16,
            is_worst: false,
            is_best: true,
        };
        let s = eval_format_view_result(&result);
        assert!(!s.is_empty());
        assert!(s.contains("[BEST]"), "best view should be marked");
    }
    #[test]
    fn test_format_view_result_worst_flag() {
        let result = ViewEvalResult {
            view_id: "bad_view".to_string(),
            psnr: 10.0,
            ssim: 0.3,
            lpips_approx: 0.8,
            mae: 0.3,
            rmse: 0.4,
            ssim_ms: 0.4,
            width: 8,
            height: 8,
            is_worst: true,
            is_best: false,
        };
        let s = eval_format_view_result(&result);
        assert!(s.contains("[WORST]"), "worst view should be marked");
    }
}
