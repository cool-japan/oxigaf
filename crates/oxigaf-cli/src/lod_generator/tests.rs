//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::functions::{opacity_descending_permutation, select_indices_by_rank};
use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;

    // ------------------------------------------------------------------
    // Shared test utilities
    // ------------------------------------------------------------------

    type CloudArrays = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    fn make_cloud(n: usize, sh_c: usize) -> CloudArrays {
        let positions: Vec<f32> = (0..n * 3).map(|i| i as f32 * 0.01).collect();
        let rotations: Vec<f32> = (0..n * 4)
            .map(|i| if i % 4 == 3 { 1.0 } else { 0.0 })
            .collect();
        let scales: Vec<f32> = vec![-1.0f32; n * 3];
        // Logit opacities from -3.0 to 3.0 across n Gaussians.
        let opacities: Vec<f32> = (0..n)
            .map(|i| (i as f32 / (n as f32).max(1.0)) * 6.0 - 3.0)
            .collect();
        let sh_coefficients: Vec<f32> = vec![0.1f32; n * sh_c];
        (positions, rotations, scales, opacities, sh_coefficients)
    }

    fn make_level(n: usize, val: f32, level_idx: usize) -> LodLevel {
        LodLevel {
            level: level_idx,
            n_gaussians: n,
            reduction_factor: 1.0,
            positions: vec![val; n * 3],
            rotations: vec![val; n * 4],
            scales: vec![val; n * 3],
            opacities: vec![val; n],
            sh_coefficients: vec![val; n * 9],
        }
    }

    // ------------------------------------------------------------------
    // LodConfig tests
    // ------------------------------------------------------------------

    #[test]
    fn test_lod_config_default_n_levels() {
        let cfg = LodConfig::default();
        assert_eq!(cfg.n_levels, 4);
    }

    #[test]
    fn test_lod_config_default_ratios() {
        let cfg = LodConfig::default();
        assert_eq!(cfg.reduction_ratios.len(), 4);
        assert!((cfg.reduction_ratios[0] - 1.0).abs() < 1e-6);
        assert!((cfg.reduction_ratios[1] - 0.5).abs() < 1e-6);
        assert!((cfg.reduction_ratios[2] - 0.25).abs() < 1e-6);
        assert!((cfg.reduction_ratios[3] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_lod_config_default_strategy() {
        let cfg = LodConfig::default();
        assert_eq!(cfg.strategy, LodStrategy::TopOpacity);
    }

    #[test]
    fn test_lod_config_default_sort_by_opacity() {
        let cfg = LodConfig::default();
        assert!(cfg.sort_by_opacity);
    }

    #[test]
    fn test_lod_config_validate_ok() {
        assert!(LodConfig::default().validate().is_ok());
    }

    #[test]
    fn test_lod_config_validate_zero_ratio() {
        let cfg = LodConfig {
            n_levels: 2,
            reduction_ratios: vec![1.0, 0.0],
            strategy: LodStrategy::TopOpacity,
            sort_by_opacity: true,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_lod_config_validate_ratio_above_one() {
        let cfg = LodConfig {
            n_levels: 2,
            reduction_ratios: vec![1.5, 0.5],
            strategy: LodStrategy::TopOpacity,
            sort_by_opacity: true,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_lod_config_validate_ascending_ratios() {
        let cfg = LodConfig {
            n_levels: 3,
            reduction_ratios: vec![0.1, 0.5, 1.0],
            strategy: LodStrategy::TopOpacity,
            sort_by_opacity: true,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_lod_config_validate_mismatched_length() {
        let cfg = LodConfig {
            n_levels: 4,
            reduction_ratios: vec![1.0, 0.5],
            strategy: LodStrategy::TopOpacity,
            sort_by_opacity: true,
        };
        assert!(cfg.validate().is_err());
    }

    // ------------------------------------------------------------------
    // compute_opacity_values tests
    // ------------------------------------------------------------------

    #[test]
    fn test_compute_opacity_values_zero_logit() {
        let probs = compute_opacity_values(&[0.0]);
        assert!((probs[0] - 0.5).abs() < 1e-6, "sigmoid(0) must equal 0.5");
    }

    #[test]
    fn test_compute_opacity_values_large_positive() {
        let probs = compute_opacity_values(&[20.0]);
        assert!(
            probs[0] > 0.99,
            "sigmoid(20) must be close to 1.0, got {}",
            probs[0]
        );
    }

    #[test]
    fn test_compute_opacity_values_large_negative() {
        let probs = compute_opacity_values(&[-20.0]);
        assert!(
            probs[0] < 0.01,
            "sigmoid(-20) must be close to 0.0, got {}",
            probs[0]
        );
    }

    #[test]
    fn test_compute_opacity_values_preserves_monotone_order() {
        let logits = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let probs = compute_opacity_values(&logits);
        for i in 1..probs.len() {
            assert!(
                probs[i] > probs[i - 1],
                "sigmoid must be monotone increasing"
            );
        }
    }

    // ------------------------------------------------------------------
    // select_top_opacity_indices tests
    // ------------------------------------------------------------------

    #[test]
    fn test_select_top_opacity_indices_selects_highest() {
        // Logit values: index 9 is highest opacity.
        let opacities: Vec<f32> = (0..10).map(|i| i as f32 - 5.0).collect();
        let indices = select_top_opacity_indices(10, &opacities, 3);
        assert_eq!(indices.len(), 3);
        // The top 3 by logit are indices 7, 8, 9.
        assert!(indices.contains(&7) || indices.contains(&8) || indices.contains(&9));
        assert_eq!(indices.iter().filter(|&&i| i >= 7).count(), 3);
    }

    #[test]
    fn test_select_top_opacity_indices_k_equals_n() {
        let opacities = vec![1.0, 2.0, 3.0];
        let indices = select_top_opacity_indices(3, &opacities, 3);
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_select_top_opacity_indices_k_exceeds_n() {
        let opacities = vec![1.0, 2.0];
        let indices = select_top_opacity_indices(2, &opacities, 10);
        assert_eq!(indices.len(), 2);
    }

    #[test]
    fn test_select_top_opacity_indices_sorted_output() {
        let opacities: Vec<f32> = (0..20).rev().map(|i| i as f32).collect();
        let indices = select_top_opacity_indices(20, &opacities, 5);
        // Output must be in ascending index order.
        for w in indices.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    // ------------------------------------------------------------------
    // select_uniform_indices tests
    // ------------------------------------------------------------------

    #[test]
    fn test_select_uniform_indices_count() {
        let indices = select_uniform_indices(100, 10);
        assert_eq!(indices.len(), 10);
    }

    #[test]
    fn test_select_uniform_indices_includes_first() {
        let indices = select_uniform_indices(100, 5);
        assert!(indices.contains(&0));
    }

    #[test]
    fn test_select_uniform_indices_includes_last() {
        let indices = select_uniform_indices(100, 5);
        assert!(indices.contains(&99), "expected 99 in {:?}", indices);
    }

    #[test]
    fn test_select_uniform_indices_k_ge_n() {
        let indices = select_uniform_indices(5, 10);
        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_select_uniform_indices_k_zero() {
        assert!(select_uniform_indices(100, 0).is_empty());
    }

    #[test]
    fn test_select_uniform_indices_specific_known() {
        // n=10, k=5 → raw: 0,2,4,6,8 → last becomes 9 → result: 0,2,4,6,9
        let indices = select_uniform_indices(10, 5);
        assert!(indices.contains(&0), "must contain 0, got {:?}", indices);
        assert!(indices.contains(&9), "must contain 9, got {:?}", indices);
        assert_eq!(indices.len(), 5);
    }

    // ------------------------------------------------------------------
    // select_random_indices tests
    // ------------------------------------------------------------------

    #[test]
    fn test_select_random_indices_count() {
        assert_eq!(select_random_indices(100, 30, 12345).len(), 30);
    }

    #[test]
    fn test_select_random_indices_reproducible() {
        let a = select_random_indices(100, 30, 42);
        let b = select_random_indices(100, 30, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_select_random_indices_different_seeds_differ() {
        let a = select_random_indices(100, 50, 1);
        let b = select_random_indices(100, 50, 9999);
        assert_ne!(a, b);
    }

    #[test]
    fn test_select_random_indices_k_exceeds_n() {
        assert_eq!(select_random_indices(5, 20, 1).len(), 5);
    }

    #[test]
    fn test_select_random_indices_k_zero() {
        assert!(select_random_indices(100, 0, 1).is_empty());
    }

    // ------------------------------------------------------------------
    // select_spatial_grid_indices tests
    // ------------------------------------------------------------------

    #[test]
    fn test_select_spatial_grid_indices_count_limit() {
        let n = 1000usize;
        let positions: Vec<f32> = (0..n * 3).map(|i| (i as f32).sin()).collect();
        let indices = select_spatial_grid_indices(&positions, 50).expect("grid selection failed");
        // Regression: the grid pass alone (occupied-cell-only) used to
        // undershoot k for real point clouds (e.g. only ~25-30 of a 4×4×4
        // grid's 64 cells occupied); the top-up pass must reach exactly k.
        assert_eq!(
            indices.len(),
            50,
            "expected exactly the requested count, not a grid-occupancy-limited undershoot"
        );
    }

    #[test]
    fn test_select_spatial_grid_indices_top_up_reaches_k_when_grid_sparse() {
        // All Gaussians share one point: every one of them falls in the
        // same grid cell, so the grid pass alone can select only 1. The
        // top-up pass must still reach the full k=20.
        let n = 100usize;
        let mut positions: Vec<f32> = Vec::with_capacity(n * 3);
        for _ in 0..n {
            positions.extend_from_slice(&[1.0f32, 2.0, 3.0]);
        }
        let indices = select_spatial_grid_indices(&positions, 20).expect("grid selection failed");
        assert_eq!(indices.len(), 20);
        // No duplicate indices.
        let mut sorted = indices.clone();
        sorted.dedup();
        assert_eq!(sorted.len(), indices.len());
    }

    #[test]
    fn test_select_spatial_grid_indices_no_duplicates_general() {
        let n = 200usize;
        let positions: Vec<f32> = (0..n * 3).map(|i| (i as f32 * 0.7).cos()).collect();
        let indices = select_spatial_grid_indices(&positions, 77).expect("grid selection failed");
        assert_eq!(indices.len(), 77);
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), indices.len(), "indices must be unique");
        assert!(indices.iter().all(|&i| i < n));
    }

    #[test]
    fn test_select_spatial_grid_indices_empty_error() {
        assert!(select_spatial_grid_indices(&[], 10).is_err());
    }

    #[test]
    fn test_select_spatial_grid_indices_k_zero() {
        let positions = vec![0.0f32; 30];
        let indices = select_spatial_grid_indices(&positions, 0).expect("k=0 must not error");
        assert!(indices.is_empty());
    }

    #[test]
    fn test_select_spatial_grid_indices_k_ge_n() {
        let n = 5usize;
        let positions: Vec<f32> = (0..n * 3).map(|i| i as f32).collect();
        let indices = select_spatial_grid_indices(&positions, 100).expect("k>=n must not error");
        assert_eq!(indices.len(), n);
    }

    // ------------------------------------------------------------------
    // extract_subset tests
    // ------------------------------------------------------------------

    #[test]
    fn test_extract_subset_stride_3_positions() {
        // n=4 Gaussians, stride=3. Source rows: [0,1,2], [3,4,5], [6,7,8], [9,10,11].
        let source: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let out = extract_subset(&source, 4, &[0, 2]).expect("extract failed");
        assert_eq!(out.len(), 6);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 2.0);
        assert_eq!(out[3], 6.0);
        assert_eq!(out[4], 7.0);
        assert_eq!(out[5], 8.0);
    }

    #[test]
    fn test_extract_subset_stride_4_rotations() {
        // n=3 Gaussians, stride=4. Row 1 = [4,5,6,7].
        let source: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let out = extract_subset(&source, 3, &[1]).expect("extract failed");
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], 4.0);
        assert_eq!(out[3], 7.0);
    }

    #[test]
    fn test_extract_subset_empty_indices() {
        let source = vec![1.0f32, 2.0, 3.0];
        assert!(extract_subset(&source, 1, &[])
            .expect("extract failed")
            .is_empty());
    }

    #[test]
    fn test_extract_subset_index_out_of_range_errors_instead_of_panicking() {
        // Regression: n_gaussians=4 but an index of 10 used to compute
        // `start = 10 * stride` past `source.len()`, producing a `start >
        // end` range that panics after the old `.min(source.len())` clamp
        // only clamped the end, not the start.
        let source: Vec<f32> = (0..12).map(|i| i as f32).collect(); // n=4, stride=3
        let result = extract_subset(&source, 4, &[0, 10]);
        assert!(matches!(
            result,
            Err(LodError::IndexOutOfRange { index: 10, .. })
        ));
    }

    #[test]
    fn test_extract_subset_not_multiple_of_n_gaussians_errors() {
        let source = vec![0.0f32; 10];
        let result = extract_subset(&source, 3, &[0]);
        assert!(matches!(result, Err(LodError::ArrayLengthMismatch { .. })));
    }

    // ------------------------------------------------------------------
    // generate_lod_level tests
    // ------------------------------------------------------------------

    #[test]
    fn test_generate_lod_level_top_opacity_count() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let config = LodConfig::default();
        let level = generate_lod_level(
            LodInputSlices {
                n_gaussians: 100,
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
            },
            50,
            &config,
            0,
        )
        .expect("generate_lod_level failed");
        assert_eq!(level.n_gaussians, 50);
    }

    #[test]
    fn test_generate_lod_level_arrays_consistent() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let config = LodConfig::default();
        let level = generate_lod_level(
            LodInputSlices {
                n_gaussians: 100,
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
            },
            50,
            &config,
            0,
        )
        .expect("generate_lod_level failed");
        assert_eq!(level.positions.len(), 50 * 3);
        assert_eq!(level.rotations.len(), 50 * 4);
        assert_eq!(level.scales.len(), 50 * 3);
        assert_eq!(level.opacities.len(), 50);
        assert_eq!(level.sh_coefficients.len(), 50 * 9);
    }

    #[test]
    fn test_generate_lod_level_level_index_stored() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let config = LodConfig::default();
        let level = generate_lod_level(
            LodInputSlices {
                n_gaussians: 100,
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
            },
            50,
            &config,
            2,
        )
        .expect("generate_lod_level failed");
        assert_eq!(level.level, 2);
    }

    #[test]
    fn test_generate_lod_level_uniform_strategy() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let config = LodConfig {
            n_levels: 2,
            reduction_ratios: vec![1.0, 0.5],
            strategy: LodStrategy::Uniform,
            sort_by_opacity: false,
        };
        let level = generate_lod_level(
            LodInputSlices {
                n_gaussians: 100,
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
            },
            40,
            &config,
            1,
        )
        .expect("uniform generate failed");
        assert!(level.n_gaussians <= 40);
    }

    #[test]
    fn test_generate_lod_level_insufficient_gaussians_error() {
        let (pos, rot, sc, op, sh) = make_cloud(10, 9);
        let config = LodConfig::default();
        let result = generate_lod_level(
            LodInputSlices {
                n_gaussians: 10,
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
            },
            20,
            &config,
            0,
        );
        assert!(matches!(
            result,
            Err(LodError::InsufficientGaussians { .. })
        ));
    }

    #[test]
    fn test_generate_lod_level_direct_call_rejects_mismatched_positions() {
        // Regression: calling `generate_lod_level` directly (bypassing
        // `generate_lod_chain`, which used to be the only validated entry
        // point) with n_gaussians=100 but a 200-element positions array
        // (a multiple of 100, but the wrong stride: 2 instead of 3) must
        // error instead of silently extracting misaligned rows.
        let n = 100usize;
        let positions = vec![0.0f32; n * 2]; // wrong: should be n * 3
        let rotations = vec![0.0f32; n * 4];
        let scales = vec![0.0f32; n * 3];
        let opacities = vec![0.0f32; n];
        let sh = vec![0.0f32; n * 9];
        let config = LodConfig::default();
        let result = generate_lod_level(
            LodInputSlices {
                n_gaussians: n,
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_coefficients: &sh,
            },
            50,
            &config,
            0,
        );
        assert!(matches!(result, Err(LodError::ArrayLengthMismatch { .. })));
    }

    #[test]
    fn test_generate_lod_level_direct_call_rejects_short_rotations() {
        // Regression: a rotations array shorter than n_gaussians*4 used to
        // reach `extract_subset` unchecked and could panic (or, before that
        // fix, `idx * stride` could exceed the array and slice out of
        // bounds). It must now be rejected up front.
        let n = 20usize;
        let positions = vec![0.0f32; n * 3];
        let rotations = vec![0.0f32; n * 4 - 1]; // one short
        let scales = vec![0.0f32; n * 3];
        let opacities = vec![0.0f32; n];
        let sh: Vec<f32> = Vec::new();
        let config = LodConfig::default();
        let result = generate_lod_level(
            LodInputSlices {
                n_gaussians: n,
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_coefficients: &sh,
            },
            10,
            &config,
            0,
        );
        assert!(matches!(result, Err(LodError::ArrayLengthMismatch { .. })));
    }

    #[test]
    fn test_generate_lod_level_uniform_sort_by_opacity_changes_selection() {
        // n=10, opacity strictly increasing with index → descending-opacity
        // permutation is exactly the reverse index order (perm[r] = 9 - r).
        let n = 10usize;
        let positions: Vec<f32> = (0..n * 3).map(|i| i as f32).collect();
        let rotations: Vec<f32> = (0..n * 4)
            .map(|i| if i % 4 == 3 { 1.0 } else { 0.0 })
            .collect();
        let scales = vec![-1.0f32; n * 3];
        let opacities: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let sh = vec![0.0f32; n];

        let unsorted_config = LodConfig {
            n_levels: 2,
            reduction_ratios: vec![1.0, 0.5],
            strategy: LodStrategy::Uniform,
            sort_by_opacity: false,
        };
        let sorted_config = LodConfig {
            sort_by_opacity: true,
            ..unsorted_config.clone()
        };

        let unsorted = generate_lod_level(
            LodInputSlices {
                n_gaussians: n,
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_coefficients: &sh,
            },
            5,
            &unsorted_config,
            1,
        )
        .expect("unsorted level");
        let sorted = generate_lod_level(
            LodInputSlices {
                n_gaussians: n,
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_coefficients: &sh,
            },
            5,
            &sorted_config,
            1,
        )
        .expect("sorted level");

        // Recover which original indices were kept: positions[i] == i*3.
        let kept = |level: &LodLevel| -> Vec<usize> {
            level
                .positions
                .chunks_exact(3)
                .map(|p| (p[0] / 3.0).round() as usize)
                .collect()
        };
        // select_uniform_indices(10, 5) picks ranks [0, 2, 4, 6, 9].
        assert_eq!(kept(&unsorted), vec![0, 2, 4, 6, 9]);
        // Mapped through the descending-opacity permutation (perm[r]=9-r)
        // and re-sorted ascending: {9,7,5,3,0} -> [0,3,5,7,9].
        assert_eq!(kept(&sorted), vec![0, 3, 5, 7, 9]);
    }

    #[test]
    fn test_generate_lod_level_random_sort_by_opacity_still_returns_target_n() {
        let (pos, rot, sc, op, sh) = make_cloud(50, 9);
        let config = LodConfig {
            n_levels: 2,
            reduction_ratios: vec![1.0, 0.4],
            strategy: LodStrategy::Random,
            sort_by_opacity: true,
        };
        let level = generate_lod_level(
            LodInputSlices {
                n_gaussians: 50,
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
            },
            20,
            &config,
            1,
        )
        .expect("random+sort_by_opacity level");
        assert_eq!(level.n_gaussians, 20);
    }

    // ------------------------------------------------------------------
    // opacity_descending_permutation / select_indices_by_rank tests
    // ------------------------------------------------------------------

    #[test]
    fn test_opacity_descending_permutation_orders_highest_first() {
        let opacities: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let perm = opacity_descending_permutation(&opacities);
        assert_eq!(perm, vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn test_select_indices_by_rank_passthrough_when_disabled() {
        let opacities = vec![0.0f32; 5];
        let ranks = vec![0, 2, 4];
        let out = select_indices_by_rank(ranks.clone(), false, &opacities);
        assert_eq!(out, ranks);
    }

    #[test]
    fn test_select_indices_by_rank_maps_through_opacity_permutation() {
        let opacities: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let ranks = vec![0, 2, 4, 6, 9];
        let out = select_indices_by_rank(ranks, true, &opacities);
        assert_eq!(out, vec![0, 3, 5, 7, 9]);
    }

    // ------------------------------------------------------------------
    // generate_lod_chain tests
    // ------------------------------------------------------------------

    #[test]
    fn test_generate_lod_chain_n_levels() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain generation failed");
        assert_eq!(chain.levels.len(), 4);
    }

    #[test]
    fn test_generate_lod_chain_level0_full_resolution() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain generation failed");
        assert_eq!(chain.levels[0].n_gaussians, 100);
    }

    #[test]
    fn test_generate_lod_chain_decreasing_sizes() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain generation failed");
        for i in 1..chain.levels.len() {
            assert!(
                chain.levels[i].n_gaussians <= chain.levels[i - 1].n_gaussians,
                "level {} ({}) must be ≤ level {} ({})",
                i,
                chain.levels[i].n_gaussians,
                i - 1,
                chain.levels[i - 1].n_gaussians
            );
        }
    }

    #[test]
    fn test_generate_lod_chain_empty_error() {
        let result = generate_lod_chain(&[], &[], &[], &[], &[], &LodConfig::default());
        assert!(matches!(result, Err(LodError::EmptyCloud)));
    }

    #[test]
    fn test_generate_lod_chain_invalid_position_length() {
        let result = generate_lod_chain(&[1.0, 2.0], &[], &[], &[], &[], &LodConfig::default());
        assert!(matches!(
            result,
            Err(LodError::InvalidPositionLength { .. })
        ));
    }

    #[test]
    fn test_generate_lod_chain_array_mismatch_rotations() {
        let result = generate_lod_chain(
            &[0.0, 0.0, 0.0], // n=1
            &[0.0; 8],        // should be 4, error
            &[0.0; 3],
            &[0.0; 1],
            &[],
            &LodConfig::default(),
        );
        assert!(matches!(result, Err(LodError::ArrayLengthMismatch { .. })));
    }

    #[test]
    fn test_generate_lod_chain_original_n_gaussians() {
        let (pos, rot, sc, op, sh) = make_cloud(80, 3);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        assert_eq!(chain.original_n_gaussians, 80);
    }

    // ------------------------------------------------------------------
    // compute_lod_stats tests
    // ------------------------------------------------------------------

    #[test]
    fn test_compute_lod_stats_n_levels() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        assert_eq!(compute_lod_stats(&chain).n_levels, 4);
    }

    #[test]
    fn test_compute_lod_stats_level_sizes_length() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        assert_eq!(compute_lod_stats(&chain).level_sizes.len(), 4);
    }

    #[test]
    fn test_compute_lod_stats_level0_size() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        assert_eq!(compute_lod_stats(&chain).level_sizes[0], 100);
    }

    #[test]
    fn test_compute_lod_stats_memory_level0() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        let stats = compute_lod_stats(&chain);
        // 100 × (3+4+3+1+9) × 4 = 100 × 20 × 4 = 8000
        assert_eq!(stats.memory_estimates[0], 8000);
    }

    #[test]
    fn test_compute_lod_stats_total_memory_correct() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        let stats = compute_lod_stats(&chain);
        let expected: usize = stats.memory_estimates.iter().sum();
        assert_eq!(stats.total_memory, expected);
    }

    // ------------------------------------------------------------------
    // merge_lod_levels tests
    // ------------------------------------------------------------------

    #[test]
    fn test_merge_lod_levels_weight_zero_is_pure_a() {
        let a = make_level(10, 1.0, 0);
        let b = make_level(10, 2.0, 1);
        let merged = merge_lod_levels(&a, &b, 0.0).expect("merge failed");
        assert!((merged.opacities[0] - 1.0).abs() < 1e-6);
        assert!((merged.positions[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_merge_lod_levels_weight_one_is_pure_b() {
        let a = make_level(10, 1.0, 0);
        let b = make_level(10, 2.0, 1);
        let merged = merge_lod_levels(&a, &b, 1.0).expect("merge failed");
        assert!((merged.opacities[0] - 2.0).abs() < 1e-6);
        assert!((merged.positions[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_merge_lod_levels_weight_half_midpoint() {
        let a = make_level(10, 0.0, 0);
        let b = make_level(10, 2.0, 1);
        let merged = merge_lod_levels(&a, &b, 0.5).expect("merge failed");
        assert!((merged.opacities[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_merge_lod_levels_size_mismatch_error() {
        let a = make_level(10, 1.0, 0);
        let b = make_level(5, 2.0, 1);
        assert!(merge_lod_levels(&a, &b, 0.5).is_err());
    }

    #[test]
    fn test_merge_lod_levels_preserves_level_a_index() {
        let a = make_level(10, 0.0, 3);
        let b = make_level(10, 1.0, 5);
        let merged = merge_lod_levels(&a, &b, 0.5).expect("merge failed");
        assert_eq!(merged.level, 3);
    }

    #[test]
    fn test_merge_lod_levels_sh_length_mismatch_error() {
        // Same n_gaussians on both sides, but a different SH degree, so the
        // mismatch is invisible to the n_gaussians check alone.
        let mut a = make_level(5, 1.0, 0);
        let mut b = make_level(5, 2.0, 0);
        a.sh_coefficients = vec![1.0; 5 * 9];
        b.sh_coefficients = vec![2.0; 5 * 3];
        let result = merge_lod_levels(&a, &b, 0.5);
        assert!(matches!(result, Err(LodError::ArrayLengthMismatch { .. })));
    }

    #[test]
    fn test_merge_lod_levels_rotations_are_unit_length() {
        let mut a = make_level(2, 0.0, 0);
        let mut b = make_level(2, 0.0, 0);
        // Two already-unit quaternions per Gaussian.
        a.rotations = vec![0.0, 0.0, 0.0, 1.0, 0.6, 0.0, 0.0, 0.8];
        b.rotations = vec![
            0.0,
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
            0.0,
            0.6,
            0.0,
            0.8,
        ];
        let merged = merge_lod_levels(&a, &b, 0.5).expect("merge failed");
        for q in merged.rotations.chunks_exact(4) {
            let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "quaternion not unit length: {q:?} (norm={norm})"
            );
        }
    }

    #[test]
    fn test_merge_lod_levels_rotation_shortest_path_avoids_zero_quaternion() {
        // `b`'s quaternion is the negation of `a`'s: same rotation, opposite
        // sign (dot(a,b) = -1 < 0). A naive component-wise LERP at t=0.5
        // would average to the zero quaternion (a degenerate rotation);
        // NLERP's sign correction must instead recognize these as the same
        // rotation and return a unit quaternion equal (up to sign) to `a`.
        let mut a = make_level(1, 0.0, 0);
        let mut b = make_level(1, 0.0, 0);
        a.rotations = vec![0.0, 0.0, 0.0, 1.0];
        b.rotations = vec![0.0, 0.0, 0.0, -1.0];
        let merged = merge_lod_levels(&a, &b, 0.5).expect("merge failed");
        let q = &merged.rotations;
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "expected a unit quaternion, got {q:?} (norm={norm})"
        );
        assert!(
            q[3].abs() > 0.99,
            "expected the w component to dominate (same rotation as input), got {q:?}"
        );
    }

    #[test]
    fn test_merge_lod_levels_rotation_weight_extremes_match_inputs() {
        let mut a = make_level(1, 0.0, 0);
        let mut b = make_level(1, 0.0, 0);
        a.rotations = vec![0.6, 0.0, 0.0, 0.8];
        b.rotations = vec![0.0, 0.6, 0.0, 0.8];
        let at_a = merge_lod_levels(&a, &b, 0.0).expect("merge failed");
        assert!((at_a.rotations[0] - 0.6).abs() < 1e-5);
        assert!((at_a.rotations[3] - 0.8).abs() < 1e-5);
        let at_b = merge_lod_levels(&a, &b, 1.0).expect("merge failed");
        assert!((at_b.rotations[1] - 0.6).abs() < 1e-5);
        assert!((at_b.rotations[3] - 0.8).abs() < 1e-5);
    }

    // ------------------------------------------------------------------
    // LodChain::select tests
    // ------------------------------------------------------------------

    #[test]
    fn test_lod_chain_select_distance_zero_gives_level0() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        let sel = LodSelector::default();
        let level = chain.select(0.0, &sel).expect("select failed");
        assert_eq!(level.level, 0);
    }

    #[test]
    fn test_lod_chain_select_large_distance_gives_lowest() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        let sel = LodSelector::default();
        let level = chain.select(1000.0, &sel).expect("select failed");
        assert_eq!(level.level, chain.levels.len() - 1);
    }

    #[test]
    fn test_lod_chain_select_mid_distance_gives_level1() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        // thresholds=[1.0, 3.0, 7.0]; distance=2.0 → beyond [0]=1.0, below [1]=3.0 → level 1
        let sel = LodSelector::new(vec![1.0, 3.0, 7.0]);
        let level = chain.select(2.0, &sel).expect("select failed");
        assert_eq!(level.level, 1);
    }

    // ------------------------------------------------------------------
    // LodChain::get_level tests
    // ------------------------------------------------------------------

    #[test]
    fn test_lod_chain_get_level_in_range() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        let level = chain.get_level(2).expect("level 2 should exist");
        assert_eq!(level.level, 2);
    }

    #[test]
    fn test_lod_chain_get_level_out_of_range_errors() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        let result = chain.get_level(999);
        assert!(matches!(
            result,
            Err(LodError::InvalidLodLevel {
                level: 999,
                n_levels: 4
            })
        ));
    }

    // ------------------------------------------------------------------
    // format_lod_stats tests
    // ------------------------------------------------------------------

    #[test]
    fn test_format_lod_stats_is_non_empty() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        assert!(!format_lod_stats(&compute_lod_stats(&chain)).is_empty());
    }

    #[test]
    fn test_format_lod_stats_contains_lod() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        let s = format_lod_stats(&compute_lod_stats(&chain));
        assert!(s.contains("LOD"), "expected 'LOD' in '{}'", s);
    }

    #[test]
    fn test_format_lod_stats_contains_counts() {
        let stats = LodStats {
            n_levels: 3,
            original_gaussians: 500,
            level_sizes: vec![500, 250, 50],
            memory_estimates: vec![40000, 20000, 4000],
            total_memory: 64000,
        };
        let s = format_lod_stats(&stats);
        assert!(s.contains("500"));
        assert!(s.contains("3"));
    }

    // ------------------------------------------------------------------
    // estimate_lod_memory tests
    // ------------------------------------------------------------------

    #[test]
    fn test_estimate_lod_memory_with_sh() {
        // n=100, sh_len=900 → sh_per=9, params=20, bytes=4*20*100=8000
        assert_eq!(estimate_lod_memory(100, 900), 8000);
    }

    #[test]
    fn test_estimate_lod_memory_no_sh() {
        // n=10, sh_len=0, params=11, bytes=4*11*10=440
        assert_eq!(estimate_lod_memory(10, 0), 440);
    }

    #[test]
    fn test_estimate_lod_memory_zero_gaussians() {
        assert_eq!(estimate_lod_memory(0, 0), 0);
    }

    // ------------------------------------------------------------------
    // find_optimal_reduction_ratios tests
    // ------------------------------------------------------------------

    #[test]
    fn test_find_optimal_reduction_ratios_returns_n_levels() {
        let ratios = find_optimal_reduction_ratios(1000, 1_000_000, 4, 9).expect("search failed");
        assert_eq!(ratios.len(), 4);
    }

    #[test]
    fn test_find_optimal_reduction_ratios_first_is_one() {
        let ratios = find_optimal_reduction_ratios(1000, 1_000_000, 4, 9).expect("search failed");
        assert!(
            (ratios[0] - 1.0).abs() < 1e-6,
            "first ratio must be 1.0, got {}",
            ratios[0]
        );
    }

    #[test]
    fn test_find_optimal_reduction_ratios_non_ascending() {
        let ratios = find_optimal_reduction_ratios(500, 500_000, 4, 9).expect("search failed");
        for i in 1..ratios.len() {
            assert!(
                ratios[i] <= ratios[i - 1] + 1e-5,
                "ratios must be non-ascending"
            );
        }
    }

    #[test]
    fn test_find_optimal_reduction_ratios_n_levels_one() {
        let ratios = find_optimal_reduction_ratios(100, 100_000, 1, 9).expect("search failed");
        assert_eq!(ratios.len(), 1);
        assert!((ratios[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_optimal_reduction_ratios_n_levels_zero() {
        assert!(find_optimal_reduction_ratios(100, 100_000, 0, 9)
            .expect("search failed")
            .is_empty());
    }

    #[test]
    fn test_find_optimal_reduction_ratios_infeasible_budget_errors() {
        // Even at r=0.1, level 0 alone (pinned to ratio 1.0) already costs
        // 1,000,000 * 80 = 80,000,000 bytes, vastly more than a 1-byte
        // budget. The old implementation returned a chain overshooting the
        // budget with no indication; it must now report the shortfall.
        let result = find_optimal_reduction_ratios(1_000_000, 1, 4, 9);
        assert!(
            matches!(
                &result,
                Err(LodError::MemoryBudgetExceeded { minimum_bytes, target_bytes })
                    if *target_bytes == 1 && *minimum_bytes > *target_bytes
            ),
            "expected MemoryBudgetExceeded with target_bytes=1 and minimum_bytes > target_bytes, got {result:?}"
        );
    }

    #[test]
    fn test_find_optimal_reduction_ratios_feasible_budget_does_not_error() {
        // Sanity check alongside the infeasible case above: a generous
        // budget must still succeed.
        let result = find_optimal_reduction_ratios(1000, 1_000_000, 4, 9);
        assert!(result.is_ok());
    }

    // ------------------------------------------------------------------
    // LodError variant display tests
    // ------------------------------------------------------------------

    #[test]
    fn test_lod_error_empty_cloud_display() {
        let e = LodError::EmptyCloud;
        assert!(format!("{}", e).contains("Empty"));
    }

    #[test]
    fn test_lod_error_invalid_position_length_display() {
        let e = LodError::InvalidPositionLength { len: 7 };
        assert!(format!("{}", e).contains("7"));
    }

    #[test]
    fn test_lod_error_array_length_mismatch_display() {
        let e = LodError::ArrayLengthMismatch {
            n_gaussians: 10,
            field: "rotations".to_string(),
            actual: 30,
        };
        assert!(format!("{}", e).contains("rotations"));
    }

    #[test]
    fn test_lod_error_invalid_lod_level_display() {
        let e = LodError::InvalidLodLevel {
            level: 5,
            n_levels: 4,
        };
        assert!(format!("{}", e).contains("5"));
    }

    #[test]
    fn test_lod_error_insufficient_gaussians_display() {
        let e = LodError::InsufficientGaussians { k: 100, n: 50 };
        let s = format!("{}", e);
        assert!(s.contains("100") && s.contains("50"));
    }

    // ------------------------------------------------------------------
    // LodLevel::validate tests
    // ------------------------------------------------------------------

    #[test]
    fn test_lod_level_validate_ok() {
        assert!(make_level(10, 0.5, 0).validate().is_ok());
    }

    #[test]
    fn test_lod_level_validate_positions_extra_element() {
        let mut level = make_level(10, 0.5, 0);
        level.positions.push(0.0);
        assert!(level.validate().is_err());
    }

    #[test]
    fn test_lod_level_validate_opacities_missing_element() {
        let mut level = make_level(10, 0.5, 0);
        level.opacities.pop();
        assert!(level.validate().is_err());
    }

    // ------------------------------------------------------------------
    // LodStrategy variant tests
    // ------------------------------------------------------------------

    #[test]
    fn test_lod_strategy_all_variants_distinct() {
        assert_ne!(LodStrategy::TopOpacity, LodStrategy::Uniform);
        assert_ne!(LodStrategy::Uniform, LodStrategy::SpatialGrid);
        assert_ne!(LodStrategy::SpatialGrid, LodStrategy::Random);
        assert_ne!(LodStrategy::Random, LodStrategy::TopOpacity);
    }

    #[test]
    fn test_lod_strategy_random_chain() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let config = LodConfig {
            n_levels: 2,
            reduction_ratios: vec![1.0, 0.5],
            strategy: LodStrategy::Random,
            sort_by_opacity: false,
        };
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &config).expect("chain failed");
        assert_eq!(chain.levels[0].n_gaussians, 100);
        assert!(chain.levels[1].n_gaussians <= 50);
    }

    #[test]
    fn test_lod_strategy_spatial_grid_chain() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let config = LodConfig {
            n_levels: 2,
            reduction_ratios: vec![1.0, 0.5],
            strategy: LodStrategy::SpatialGrid,
            sort_by_opacity: false,
        };
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &config).expect("chain failed");
        assert_eq!(chain.levels.len(), 2);
    }

    // ------------------------------------------------------------------
    // LodSelector tests
    // ------------------------------------------------------------------

    #[test]
    fn test_lod_selector_default_three_thresholds() {
        let sel = LodSelector::default();
        assert_eq!(sel.thresholds.len(), 3);
    }

    #[test]
    fn test_lod_selector_default_values() {
        let sel = LodSelector::default();
        assert!((sel.thresholds[0] - 0.5).abs() < 1e-6);
        assert!((sel.thresholds[1] - 2.0).abs() < 1e-6);
        assert!((sel.thresholds[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_lod_selector_new() {
        let sel = LodSelector::new(vec![1.0, 10.0]);
        assert_eq!(sel.thresholds, vec![1.0, 10.0]);
    }

    // ------------------------------------------------------------------
    // n_params_per_gaussian tests
    // ------------------------------------------------------------------

    #[test]
    fn test_n_params_per_gaussian_with_sh9() {
        let level = make_level(10, 0.0, 0);
        // 3+4+3+1+9 = 20
        assert_eq!(level.n_params_per_gaussian(), 20);
    }

    #[test]
    fn test_n_params_per_gaussian_no_sh() {
        let mut level = make_level(10, 0.0, 0);
        level.sh_coefficients.clear();
        // 3+4+3+1+0 = 11
        assert_eq!(level.n_params_per_gaussian(), 11);
    }
}
