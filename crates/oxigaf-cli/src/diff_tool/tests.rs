//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn make_snapshot(name: &str, step: usize, n: usize, fill: f32) -> ModelSnapshot {
        ModelSnapshot::new(
            name,
            step,
            vec![fill; n * 3],
            vec![fill; n],
            vec![fill; n * 3],
            vec![fill; n * 3],
        )
        .expect("valid snapshot")
    }

    fn make_snapshot_vals(
        name: &str,
        step: usize,
        positions: Vec<f32>,
        opacities: Vec<f32>,
        scales: Vec<f32>,
        colors: Vec<f32>,
    ) -> ModelSnapshot {
        ModelSnapshot::new(name, step, positions, opacities, scales, colors)
            .expect("valid snapshot")
    }

    // ------------------------------------------------------------------
    // ModelSnapshot::new
    // ------------------------------------------------------------------

    #[test]
    fn test_snapshot_new_valid() {
        let s = ModelSnapshot::new(
            "a",
            0,
            vec![0.0; 6],
            vec![0.0; 2],
            vec![0.0; 6],
            vec![0.0; 6],
        );
        assert!(s.is_ok());
        let s = s.expect("ok");
        assert_eq!(s.n_gaussians, 2);
    }

    #[test]
    fn test_snapshot_new_positions_not_divisible_by_3() {
        let r = ModelSnapshot::new(
            "a",
            0,
            vec![0.0; 5],
            vec![0.0; 1],
            vec![0.0; 3],
            vec![0.0; 3],
        );
        assert!(matches!(r, Err(DiffError::DimensionError(_))));
    }

    #[test]
    fn test_snapshot_new_opacities_mismatch() {
        let r = ModelSnapshot::new(
            "a",
            0,
            vec![0.0; 6],
            vec![0.0; 3],
            vec![0.0; 6],
            vec![0.0; 6],
        );
        assert!(matches!(r, Err(DiffError::DimensionError(_))));
    }

    #[test]
    fn test_snapshot_new_scales_mismatch() {
        let r = ModelSnapshot::new(
            "a",
            0,
            vec![0.0; 6],
            vec![0.0; 2],
            vec![0.0; 5],
            vec![0.0; 6],
        );
        assert!(matches!(r, Err(DiffError::DimensionError(_))));
    }

    #[test]
    fn test_snapshot_new_colors_mismatch() {
        let r = ModelSnapshot::new(
            "a",
            0,
            vec![0.0; 6],
            vec![0.0; 2],
            vec![0.0; 6],
            vec![0.0; 5],
        );
        assert!(matches!(r, Err(DiffError::DimensionError(_))));
    }

    // ------------------------------------------------------------------
    // ModelSnapshot::activated_opacity
    // ------------------------------------------------------------------

    #[test]
    fn test_activated_opacity_zero_logit() {
        let s = make_snapshot("a", 0, 2, 0.0);
        // sigmoid(0) = 0.5
        let opa = s.activated_opacity(0);
        assert!((opa - 0.5).abs() < 1e-6, "expected 0.5, got {}", opa);
    }

    #[test]
    fn test_activated_opacity_large_positive() {
        let s = make_snapshot_vals(
            "a",
            0,
            vec![0.0; 3],
            vec![100.0],
            vec![0.0; 3],
            vec![0.0; 3],
        );
        // sigmoid(100) ≈ 1.0
        assert!(s.activated_opacity(0) > 0.999);
    }

    #[test]
    fn test_activated_opacity_large_negative() {
        let s = make_snapshot_vals(
            "a",
            0,
            vec![0.0; 3],
            vec![-100.0],
            vec![0.0; 3],
            vec![0.0; 3],
        );
        // sigmoid(-100) ≈ 0.0
        assert!(s.activated_opacity(0) < 1e-3);
    }

    // ------------------------------------------------------------------
    // ModelSnapshot::activated_scale
    // ------------------------------------------------------------------

    #[test]
    fn test_activated_scale_zero_log() {
        let s = make_snapshot("a", 0, 2, 0.0);
        // exp(0) = 1.0
        let sc = s.activated_scale(0, 0);
        assert!((sc - 1.0).abs() < 1e-6, "expected 1.0, got {}", sc);
    }

    #[test]
    fn test_activated_scale_log_one() {
        let s = make_snapshot_vals(
            "a",
            0,
            vec![0.0; 3],
            vec![0.0],
            vec![1.0, 2.0, 3.0],
            vec![0.0; 3],
        );
        assert!((s.activated_scale(0, 0) - 1.0f32.exp()).abs() < 1e-5);
        assert!((s.activated_scale(0, 1) - 2.0f32.exp()).abs() < 1e-5);
        assert!((s.activated_scale(0, 2) - 3.0f32.exp()).abs() < 1e-5);
    }

    // ------------------------------------------------------------------
    // compute_field_diff
    // ------------------------------------------------------------------

    #[test]
    fn test_field_diff_identical() {
        let a = vec![1.0f32, 2.0, 3.0];
        let d = compute_field_diff(&a, &a, "test", 1e-6).expect("ok");
        assert!((d.mean_change).abs() < 1e-7);
        assert!((d.l2_distance).abs() < 1e-7);
        assert!((d.cosine_similarity - 1.0).abs() < 1e-6);
        assert_eq!(d.fraction_changed, 0.0);
    }

    #[test]
    fn test_field_diff_known_difference() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 1.0, 1.0];
        let d = compute_field_diff(&a, &b, "test", 1e-6).expect("ok");
        assert!((d.mean_change - 1.0).abs() < 1e-6);
        assert!((d.rms_change - 1.0).abs() < 1e-6);
        assert!((d.l2_distance - 3.0f32.sqrt()).abs() < 1e-6);
        assert_eq!(d.fraction_changed, 1.0);
    }

    #[test]
    fn test_field_diff_both_zero_vectors_cosine() {
        let a = vec![0.0f32; 4];
        let d = compute_field_diff(&a, &a, "zeros", 1e-6).expect("ok");
        assert!((d.cosine_similarity - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_field_diff_one_zero_vector_cosine() {
        let a = vec![0.0f32; 4];
        let b = vec![1.0f32; 4];
        let d = compute_field_diff(&a, &b, "t", 1e-6).expect("ok");
        assert!((d.cosine_similarity).abs() < 1e-7);
    }

    #[test]
    fn test_field_diff_length_mismatch() {
        let a = vec![1.0f32; 3];
        let b = vec![1.0f32; 4];
        assert!(matches!(
            compute_field_diff(&a, &b, "x", 1e-6),
            Err(DiffError::DimensionError(_))
        ));
    }

    // ------------------------------------------------------------------
    // diff_models
    // ------------------------------------------------------------------

    #[test]
    fn test_diff_models_identical() {
        let a = make_snapshot("a", 0, 10, 0.5);
        let config = DiffConfig::default();
        let d = diff_models(&a, &a, &config).expect("ok");
        assert!((d.position_diff.rms_change).abs() < 1e-7);
        assert!((d.opacity_diff.rms_change).abs() < 1e-7);
        assert!((d.scale_diff.rms_change).abs() < 1e-7);
        assert!((d.color_diff.rms_change).abs() < 1e-7);
        assert!(d.summary_score < 1e-6);
    }

    #[test]
    fn test_diff_models_size_mismatch() {
        let a = make_snapshot("a", 0, 5, 0.0);
        let b = make_snapshot("b", 1, 7, 0.0);
        let config = DiffConfig::default();
        assert!(matches!(
            diff_models(&a, &b, &config),
            Err(DiffError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn test_diff_models_empty_a() {
        let a = ModelSnapshot {
            name: "a".into(),
            step: 0,
            n_gaussians: 0,
            positions: vec![],
            opacities: vec![],
            scales: vec![],
            colors: vec![],
        };
        let b = make_snapshot("b", 1, 3, 0.0);
        let config = DiffConfig::default();
        assert!(matches!(
            diff_models(&a, &b, &config),
            Err(DiffError::EmptyModelA)
        ));
    }

    #[test]
    fn test_diff_models_empty_b() {
        let a = make_snapshot("a", 0, 3, 0.0);
        let b = ModelSnapshot {
            name: "b".into(),
            step: 1,
            n_gaussians: 0,
            positions: vec![],
            opacities: vec![],
            scales: vec![],
            colors: vec![],
        };
        let config = DiffConfig::default();
        assert!(matches!(
            diff_models(&a, &b, &config),
            Err(DiffError::EmptyModelB)
        ));
    }

    #[test]
    fn test_diff_models_n_compared_matches_default_include_inactive() {
        // include_inactive: true (the default) -- every Gaussian counts as
        // compared, matching n_gaussians.
        let a = make_snapshot("a", 0, 10, 0.5);
        let config = DiffConfig::default();
        let d = diff_models(&a, &a, &config).expect("ok");
        assert_eq!(d.n_gaussians, 10);
        assert_eq!(d.n_compared, 10);
    }

    // Regression coverage for: two models where every Gaussian is inactive
    // (activated opacity < 0.1) used to report `cosine_similarity: 1.0` and
    // all-zero change statistics -- indistinguishable from "these models
    // are identical" -- with nothing indicating that zero Gaussians were
    // actually compared.
    #[test]
    fn test_diff_models_all_inactive_reports_zero_compared_not_identical() {
        let n = 5;
        // sigmoid(-10) ~= 4.5e-5, well under the 0.1 activation threshold:
        // every Gaussian in both snapshots is "inactive".
        let very_negative = vec![-10.0_f32; n];
        let a = make_snapshot_vals(
            "a",
            0,
            vec![0.0; n * 3],
            very_negative.clone(),
            vec![0.0; n * 3],
            vec![0.0; n * 3],
        );
        // Genuinely different positions from `a`, so an *honest* comparison
        // (if these Gaussians had been included) would not be "identical".
        let b = make_snapshot_vals(
            "b",
            1,
            vec![5.0; n * 3],
            very_negative,
            vec![0.0; n * 3],
            vec![0.0; n * 3],
        );
        let config = DiffConfig {
            include_inactive: false,
            ..DiffConfig::default()
        };
        let diff = diff_models(&a, &b, &config).expect("diff ok");

        assert_eq!(
            diff.n_gaussians, n,
            "total Gaussian count is still reported"
        );
        assert_eq!(
            diff.n_compared, 0,
            "no Gaussian passed the activation threshold, so none were compared"
        );

        let text = format_model_diff(&diff);
        assert!(
            text.contains("WARNING"),
            "output must warn that the comparison is vacuous: {text}"
        );
        assert!(
            text.contains("Compared: 0"),
            "output must surface n_compared: {text}"
        );
    }

    #[test]
    fn test_diff_models_n_compared_reflects_partial_activity() {
        let n = 4;
        // Gaussians 0,1 active (opacity logit 5.0 -> sigmoid ~0.993);
        // Gaussians 2,3 inactive (opacity logit -10.0 -> sigmoid ~4.5e-5).
        let opacities = vec![5.0, 5.0, -10.0, -10.0];
        let a = make_snapshot_vals(
            "a",
            0,
            vec![0.0; n * 3],
            opacities.clone(),
            vec![0.0; n * 3],
            vec![0.0; n * 3],
        );
        let b = make_snapshot_vals(
            "b",
            1,
            vec![1.0; n * 3],
            opacities,
            vec![0.0; n * 3],
            vec![0.0; n * 3],
        );
        let config = DiffConfig {
            include_inactive: false,
            ..DiffConfig::default()
        };
        let diff = diff_models(&a, &b, &config).expect("diff ok");
        assert_eq!(diff.n_gaussians, 4);
        assert_eq!(
            diff.n_compared, 2,
            "only the 2 active Gaussians should be compared"
        );
        // The 2 active Gaussians moved by 1.0 on every axis, so this must
        // not report "no change" the way an all-inactive comparison would.
        assert!(diff.position_diff.rms_change > 0.5);
    }

    // ------------------------------------------------------------------
    // format_model_diff
    // ------------------------------------------------------------------

    #[test]
    fn test_format_model_diff_contains_step_info() {
        let a = make_snapshot("snap_100", 100, 5, 0.0);
        let b = make_snapshot("snap_200", 200, 5, 1.0);
        let config = DiffConfig::default();
        let diff = diff_models(&a, &b, &config).expect("ok");
        let text = format_model_diff(&diff);
        assert!(text.contains("100"), "should contain step_a");
        assert!(text.contains("200"), "should contain step_b");
        assert!(text.contains("snap_100"));
        assert!(text.contains("snap_200"));
    }

    // ------------------------------------------------------------------
    // largest_position_changes
    // ------------------------------------------------------------------

    #[test]
    fn test_largest_position_changes_k1() {
        // Gaussian 1 moves 10 units, others stay.
        let mut pos_b = vec![0.0f32; 9]; // 3 Gaussians
        pos_b[3] = 10.0; // Gaussian 1 moves on x.
        let a = make_snapshot_vals(
            "a",
            0,
            vec![0.0; 9],
            vec![0.0; 3],
            vec![0.0; 9],
            vec![0.0; 9],
        );
        let b = make_snapshot_vals("b", 1, pos_b, vec![0.0; 3], vec![0.0; 9], vec![0.0; 9]);
        let result = largest_position_changes(&a, &b, 1).expect("ok");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 1); // Gaussian index 1
        assert!((result[0].1 - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_largest_position_changes_k_greater_than_n() {
        // k=100 with only 3 Gaussians → returns all 3.
        let a = make_snapshot("a", 0, 3, 0.0);
        let b = make_snapshot("b", 1, 3, 1.0);
        let result = largest_position_changes(&a, &b, 100).expect("ok");
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_largest_position_changes_size_mismatch() {
        let a = make_snapshot("a", 0, 3, 0.0);
        let b = make_snapshot("b", 1, 5, 0.0);
        assert!(matches!(
            largest_position_changes(&a, &b, 1),
            Err(DiffError::SizeMismatch { .. })
        ));
    }

    // ------------------------------------------------------------------
    // opacity_changes
    // ------------------------------------------------------------------

    #[test]
    fn test_opacity_changes_all_below_threshold() {
        // All logits very negative → all below 0.5 threshold.
        let a = make_snapshot_vals(
            "a",
            0,
            vec![0.0; 3],
            vec![-10.0],
            vec![0.0; 3],
            vec![0.0; 3],
        );
        let b = make_snapshot_vals(
            "b",
            1,
            vec![0.0; 3],
            vec![-10.0],
            vec![0.0; 3],
            vec![0.0; 3],
        );
        let (active, inactive) = opacity_changes(&a, &b, 0.5).expect("ok");
        assert!(active.is_empty());
        assert!(inactive.is_empty());
    }

    #[test]
    fn test_opacity_changes_crossing() {
        // Gaussian 0: A=inactive (logit -10), B=active (logit +10).
        // Gaussian 1: A=active (logit +10), B=inactive (logit -10).
        let a = make_snapshot_vals(
            "a",
            0,
            vec![0.0; 6],
            vec![-10.0, 10.0],
            vec![0.0; 6],
            vec![0.0; 6],
        );
        let b = make_snapshot_vals(
            "b",
            1,
            vec![0.0; 6],
            vec![10.0, -10.0],
            vec![0.0; 6],
            vec![0.0; 6],
        );
        let (active, inactive) = opacity_changes(&a, &b, 0.5).expect("ok");
        assert_eq!(active, vec![0]);
        assert_eq!(inactive, vec![1]);
    }

    // ------------------------------------------------------------------
    // per_gaussian_change_score
    // ------------------------------------------------------------------

    #[test]
    fn test_per_gaussian_change_score_identical() {
        let a = make_snapshot("a", 0, 5, 1.0);
        let scores = per_gaussian_change_score(&a, &a).expect("ok");
        assert_eq!(scores.len(), 5);
        for s in &scores {
            assert!(s.abs() < 1e-6, "expected ~0, got {}", s);
        }
    }

    #[test]
    fn test_per_gaussian_change_score_different() {
        let a = make_snapshot("a", 0, 4, 0.0);
        let b = make_snapshot("b", 1, 4, 1.0);
        let scores = per_gaussian_change_score(&a, &b).expect("ok");
        for s in &scores {
            assert!(*s > 0.0, "expected positive score");
        }
    }

    #[test]
    fn test_per_gaussian_change_score_size_mismatch() {
        let a = make_snapshot("a", 0, 3, 0.0);
        let b = make_snapshot("b", 1, 4, 0.0);
        assert!(matches!(
            per_gaussian_change_score(&a, &b),
            Err(DiffError::SizeMismatch { .. })
        ));
    }

    // ------------------------------------------------------------------
    // change_score_histogram
    // ------------------------------------------------------------------

    #[test]
    fn test_change_score_histogram_bin_count() {
        let scores = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
        let (edges, counts) = change_score_histogram(&scores, 4).expect("ok");
        assert_eq!(edges.len(), 5); // bins + 1
        assert_eq!(counts.len(), 4);
    }

    #[test]
    fn test_change_score_histogram_sum_equals_n() {
        let scores: Vec<f32> = (0..20).map(|i| i as f32 * 0.05).collect();
        let (_edges, counts) = change_score_histogram(&scores, 5).expect("ok");
        let total: usize = counts.iter().sum();
        assert_eq!(total, 20);
    }

    #[test]
    fn test_change_score_histogram_bins_zero_error() {
        let scores = vec![1.0f32; 5];
        assert!(matches!(
            change_score_histogram(&scores, 0),
            Err(DiffError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_change_score_histogram_empty_error() {
        assert!(matches!(
            change_score_histogram(&[], 5),
            Err(DiffError::InvalidConfig(_))
        ));
    }

    // ------------------------------------------------------------------
    // snapshots_approximately_equal
    // ------------------------------------------------------------------

    #[test]
    fn test_snapshots_approximately_equal_identical() {
        let a = make_snapshot("a", 0, 10, 0.5);
        let eq = snapshots_approximately_equal(&a, &a, 1e-5, 1e-8).expect("ok");
        assert!(eq);
    }

    #[test]
    fn test_snapshots_approximately_equal_large_diff() {
        let a = make_snapshot("a", 0, 5, 0.0);
        let b = make_snapshot("b", 1, 5, 100.0);
        let eq = snapshots_approximately_equal(&a, &b, 1e-5, 1e-8).expect("ok");
        assert!(!eq);
    }

    #[test]
    fn test_snapshots_approximately_equal_within_atol() {
        let a = make_snapshot("a", 0, 3, 1.0);
        let b = make_snapshot_vals(
            "b",
            1,
            vec![1.0 + 1e-7; 9],
            vec![1.0 + 1e-7; 3],
            vec![1.0 + 1e-7; 9],
            vec![1.0 + 1e-7; 9],
        );
        let eq = snapshots_approximately_equal(&a, &b, 1e-5, 1e-6).expect("ok");
        assert!(eq);
    }

    // ------------------------------------------------------------------
    // diff_sequence
    // ------------------------------------------------------------------

    #[test]
    fn test_diff_sequence_empty_error() {
        let config = DiffConfig::default();
        assert!(matches!(
            diff_sequence(&[], &config),
            Err(DiffError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_diff_sequence_single_error() {
        let a = make_snapshot("a", 0, 5, 0.0);
        let config = DiffConfig::default();
        assert!(matches!(
            diff_sequence(&[a], &config),
            Err(DiffError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_diff_sequence_two_snapshots() {
        let a = make_snapshot("a", 0, 5, 0.0);
        let b = make_snapshot("b", 10, 5, 1.0);
        let config = DiffConfig::default();
        let diffs = diff_sequence(&[a, b], &config).expect("ok");
        assert_eq!(diffs.len(), 1);
    }

    #[test]
    fn test_diff_sequence_three_snapshots() {
        let a = make_snapshot("a", 0, 5, 0.0);
        let b = make_snapshot("b", 10, 5, 0.5);
        let c = make_snapshot("c", 20, 5, 1.0);
        let config = DiffConfig::default();
        let diffs = diff_sequence(&[a, b, c], &config).expect("ok");
        assert_eq!(diffs.len(), 2);
    }

    // ------------------------------------------------------------------
    // detect_regression
    // ------------------------------------------------------------------

    #[test]
    fn test_detect_regression_identical_no_regression() {
        let a = make_snapshot("a", 0, 5, 0.5);
        let config = DiffConfig::default();
        let diff = diff_models(&a, &a, &config).expect("ok");
        let report = detect_regression(&diff, 0.05, 0.1, 0.01);
        assert!(!report.overall_regression);
        assert!(report.details.is_empty());
    }

    #[test]
    fn test_detect_regression_scale_increased() {
        // Model B has much larger scales (log-scale increased from 0 to 5).
        let a = make_snapshot("a", 0, 5, 0.0);
        let b = make_snapshot_vals(
            "b",
            10,
            vec![0.0; 15],
            vec![0.0; 5],
            vec![5.0; 15], // scales increased
            vec![0.0; 15],
        );
        let config = DiffConfig::default();
        let diff = diff_models(&a, &b, &config).expect("ok");
        // mean_change of scale should be 5.0 > scale_threshold 0.1.
        let report = detect_regression(&diff, 0.05, 0.1, 100.0); // high pos threshold
        assert!(report.scale_regressed);
        assert!(report.overall_regression);
    }

    #[test]
    fn test_detect_regression_position_unstable() {
        let a = make_snapshot("a", 0, 5, 0.0);
        let b = make_snapshot_vals(
            "b",
            1,
            vec![10.0; 15], // large position shift
            vec![0.0; 5],
            vec![0.0; 15],
            vec![0.0; 15],
        );
        let config = DiffConfig::default();
        let diff = diff_models(&a, &b, &config).expect("ok");
        let report = detect_regression(&diff, 100.0, 100.0, 0.01);
        assert!(report.position_unstable);
        assert!(report.overall_regression);
    }

    // ------------------------------------------------------------------
    // summarize_progress
    // ------------------------------------------------------------------

    #[test]
    fn test_summarize_progress_empty_error() {
        assert!(matches!(
            summarize_progress(&[], 1e-4),
            Err(DiffError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_summarize_progress_one_diff() {
        let a = make_snapshot("a", 0, 5, 0.0);
        let b = make_snapshot("b", 100, 5, 1.0);
        let config = DiffConfig::default();
        let diff = diff_models(&a, &b, &config).expect("ok");
        let summary = summarize_progress(&[diff], 1e-4).expect("ok");
        assert_eq!(summary.n_steps, 1);
        assert_eq!(summary.total_steps, 100);
    }

    #[test]
    fn test_summarize_progress_converging() {
        // Three diffs with decreasing summary_scores → converging.
        let make_diff = |score: f32, step_a: usize, step_b: usize| {
            // Build a ModelDiff with the desired summary_score.
            // We can construct a diff from identical + small perturbation.
            let a = make_snapshot("a", step_a, 5, 0.0);
            let b = make_snapshot_vals(
                "b",
                step_b,
                vec![score; 15], // positions differ by 'score'
                vec![0.0; 5],
                vec![0.0; 15],
                vec![0.0; 15],
            );
            let config = DiffConfig::default();
            diff_models(&a, &b, &config).expect("ok")
        };
        let d1 = make_diff(3.0, 0, 10);
        let d2 = make_diff(2.0, 10, 20);
        let d3 = make_diff(1.0, 20, 30);
        let summary = summarize_progress(&[d1, d2, d3], 1e-4).expect("ok");
        assert!(summary.converging);
        assert_eq!(summary.total_steps, 30);
    }

    #[test]
    fn test_summarize_progress_stalled() {
        // All diffs have near-zero change → stalled.
        let a = make_snapshot("a", 0, 5, 0.5);
        let config = DiffConfig::default();
        let d1 = diff_models(&a, &a, &config).expect("ok");
        let d2 = diff_models(&a, &a, &config).expect("ok");
        let d3 = diff_models(&a, &a, &config).expect("ok");
        let summary = summarize_progress(&[d1, d2, d3], 1e-3).expect("ok");
        assert!(summary.stalled);
    }

    // ------------------------------------------------------------------
    // DiffConfig::validate
    // ------------------------------------------------------------------

    #[test]
    fn test_diff_config_validate_negative_epsilon() {
        let config = DiffConfig {
            epsilon: -1.0,
            ..DiffConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(DiffError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_diff_config_validate_default_ok() {
        assert!(DiffConfig::default().validate().is_ok());
    }

    // ------------------------------------------------------------------
    // diff_models_variable
    // ------------------------------------------------------------------

    /// Helper: build a snapshot from a list of (x,y,z) positions with
    /// uniform opacities/scales/colors.
    fn make_snapshot_positions(name: &str, step: usize, positions: &[[f32; 3]]) -> ModelSnapshot {
        let n = positions.len();
        let pos_flat: Vec<f32> = positions.iter().flat_map(|p| p.iter().copied()).collect();
        ModelSnapshot::new(
            name,
            step,
            pos_flat,
            vec![0.5_f32; n],
            vec![0.0_f32; n * 3],
            vec![0.3_f32; n * 3],
        )
        .expect("valid snapshot")
    }

    #[test]
    fn test_diff_variable_identical_models() {
        // Same model diffed against itself → 0 added, 0 removed, all matched.
        let positions: Vec<[f32; 3]> = (0..20).map(|i| [i as f32 * 2.0, 0.0, 0.0]).collect();
        let a = make_snapshot_positions("a", 0, &positions);
        let config = DiffConfig::default();
        let diff = diff_models_variable(&a, &a, &config).expect("ok");
        assert_eq!(diff.added_gaussians, 0, "no added gaussians");
        assert_eq!(diff.removed_gaussians, 0, "no removed gaussians");
        assert_eq!(diff.n_gaussians, 20, "all 20 matched");
    }

    #[test]
    fn test_diff_variable_b_larger() {
        // A has 10 Gaussians on a line, B has those same 10 plus 10 extra far away.
        let base: Vec<[f32; 3]> = (0..10).map(|i| [i as f32 * 2.0, 0.0, 0.0]).collect();
        let extra: Vec<[f32; 3]> = (0..10).map(|i| [i as f32 * 2.0, 1000.0, 0.0]).collect();
        let a_positions = base.clone();
        let mut b_positions = base.clone();
        b_positions.extend_from_slice(&extra);

        let a = make_snapshot_positions("a", 0, &a_positions);
        let b = make_snapshot_positions("b", 1, &b_positions);
        let config = DiffConfig::default();
        let diff = diff_models_variable(&a, &b, &config).expect("ok");
        assert_eq!(diff.added_gaussians, 10, "10 extra B gaussians are added");
        assert_eq!(diff.removed_gaussians, 0, "no A gaussians removed");
    }

    #[test]
    fn test_diff_variable_a_larger() {
        // A has 10 base + 10 extra far away, B has only the 10 base.
        let base: Vec<[f32; 3]> = (0..10).map(|i| [i as f32 * 2.0, 0.0, 0.0]).collect();
        let extra: Vec<[f32; 3]> = (0..10).map(|i| [i as f32 * 2.0, 1000.0, 0.0]).collect();
        let mut a_positions = base.clone();
        a_positions.extend_from_slice(&extra);
        let b_positions = base.clone();

        let a = make_snapshot_positions("a", 0, &a_positions);
        let b = make_snapshot_positions("b", 1, &b_positions);
        let config = DiffConfig::default();
        let diff = diff_models_variable(&a, &b, &config).expect("ok");
        assert_eq!(diff.added_gaussians, 0, "no B gaussians are added");
        assert_eq!(diff.removed_gaussians, 10, "10 A gaussians have no match");
    }

    #[test]
    fn test_diff_variable_all_moved_far() {
        // All B positions shifted by 100.0 on x — far beyond the default 0.5 match_radius.
        let a_positions: Vec<[f32; 3]> = (0..5).map(|i| [i as f32, 0.0, 0.0]).collect();
        let b_positions: Vec<[f32; 3]> = (0..5).map(|i| [i as f32 + 100.0, 0.0, 0.0]).collect();

        let a = make_snapshot_positions("a", 0, &a_positions);
        let b = make_snapshot_positions("b", 1, &b_positions);
        let config = DiffConfig::default();
        let diff = diff_models_variable(&a, &b, &config).expect("ok");
        assert_eq!(diff.added_gaussians, 5, "all B gaussians are new");
        assert_eq!(diff.removed_gaussians, 5, "all A gaussians are gone");
        assert_eq!(diff.n_gaussians, 0, "no matched pairs");
    }

    #[test]
    fn test_diff_variable_partial_match() {
        // A has 10 Gaussians; B has 8 close-matches + 4 far-away extras.
        // A[8] and A[9] have no B match → 2 removed.
        // B[8..12] are far away → 4 added.
        let a_positions: Vec<[f32; 3]> = (0..10).map(|i| [i as f32 * 3.0, 0.0, 0.0]).collect();
        // B[0..8]: very close to A[0..8] (offset 0.01 — within 0.5 radius)
        let mut b_positions: Vec<[f32; 3]> =
            (0..8).map(|i| [i as f32 * 3.0 + 0.01, 0.0, 0.0]).collect();
        // B[8..12]: far from any A position (y = 500)
        for j in 0..4 {
            b_positions.push([j as f32, 500.0, 0.0]);
        }

        let a = make_snapshot_positions("a", 0, &a_positions);
        let b = make_snapshot_positions("b", 1, &b_positions);
        let config = DiffConfig::default();
        let diff = diff_models_variable(&a, &b, &config).expect("ok");
        assert_eq!(diff.added_gaussians, 4, "4 far B gaussians are added");
        assert_eq!(diff.removed_gaussians, 2, "A[8] and A[9] have no match");
        assert_eq!(diff.n_gaussians, 8, "8 matched pairs");
    }

    #[test]
    fn test_diff_variable_empty_a() {
        let a = ModelSnapshot {
            name: "a".into(),
            step: 0,
            n_gaussians: 0,
            positions: vec![],
            opacities: vec![],
            scales: vec![],
            colors: vec![],
        };
        let b = make_snapshot_positions("b", 1, &[[0.0, 0.0, 0.0]]);
        let config = DiffConfig::default();
        assert!(matches!(
            diff_models_variable(&a, &b, &config),
            Err(DiffError::EmptyModelA)
        ));
    }

    #[test]
    fn test_diff_variable_empty_b() {
        let a = make_snapshot_positions("a", 0, &[[0.0, 0.0, 0.0]]);
        let b = ModelSnapshot {
            name: "b".into(),
            step: 1,
            n_gaussians: 0,
            positions: vec![],
            opacities: vec![],
            scales: vec![],
            colors: vec![],
        };
        let config = DiffConfig::default();
        assert!(matches!(
            diff_models_variable(&a, &b, &config),
            Err(DiffError::EmptyModelB)
        ));
    }

    #[test]
    fn test_diff_variable_format_output() {
        // Ensure format_model_diff produces output containing "Added:" text.
        let a_positions: Vec<[f32; 3]> = (0..5).map(|i| [i as f32 * 3.0, 0.0, 0.0]).collect();
        let mut b_positions: Vec<[f32; 3]> = (0..5).map(|i| [i as f32 * 3.0, 0.0, 0.0]).collect();
        // Add 3 extra Gaussians far away.
        for k in 0..3 {
            b_positions.push([k as f32, 999.0, 0.0]);
        }
        let a = make_snapshot_positions("snap_a", 0, &a_positions);
        let b = make_snapshot_positions("snap_b", 10, &b_positions);
        let config = DiffConfig::default();
        let diff = diff_models_variable(&a, &b, &config).expect("ok");
        let text = format_model_diff(&diff);
        assert!(
            text.contains("Added:"),
            "formatted diff should contain 'Added:' but got:\n{}",
            text
        );
        assert_eq!(diff.added_gaussians, 3);
        assert_eq!(diff.removed_gaussians, 0);
    }
}
