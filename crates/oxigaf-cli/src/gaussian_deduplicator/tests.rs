//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_config() -> DedupConfig {
        DedupConfig {
            position_threshold: 0.01,
            opacity_threshold: 0.1,
            scale_threshold: 0.2,
            color_threshold: 0.2,
            keep_policy: DedupKeepPolicy::KeepHighestOpacity,
            use_spatial_hash: false,
            cell_size: 0.02,
        }
    }

    type GaussianArrays = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    /// Build minimal flat arrays for `n` Gaussians at distinct positions.
    fn make_distinct(n: usize) -> GaussianArrays {
        let positions: Vec<f32> = (0..n).flat_map(|i| [i as f32, 0.0, 0.0]).collect();
        let rotations: Vec<f32> = (0..n).flat_map(|_| [0.0f32, 0.0, 0.0, 1.0]).collect();
        let scales: Vec<f32> = (0..n).flat_map(|_| [1.0f32, 1.0, 1.0]).collect();
        let opacities: Vec<f32> = vec![0.5f32; n];
        let sh: Vec<f32> = vec![0.0f32; n * 3];
        (positions, rotations, scales, opacities, sh)
    }

    /// Build arrays for n Gaussians, all at the same position.
    fn make_identical(n: usize) -> GaussianArrays {
        let positions: Vec<f32> = vec![0.0f32; n * 3];
        let rotations: Vec<f32> = (0..n).flat_map(|_| [0.0f32, 0.0, 0.0, 1.0]).collect();
        let scales: Vec<f32> = (0..n).flat_map(|_| [1.0f32, 1.0, 1.0]).collect();
        let opacities: Vec<f32> = vec![0.5f32; n];
        let sh: Vec<f32> = vec![0.0f32; n * 3];
        (positions, rotations, scales, opacities, sh)
    }

    // -----------------------------------------------------------------------
    // gd_hash_cell
    // -----------------------------------------------------------------------

    #[test]
    fn hash_cell_same_coords_same_hash() {
        let h1 = gd_hash_cell(3, 7, -2, 1024);
        let h2 = gd_hash_cell(3, 7, -2, 1024);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_cell_origin_in_range() {
        let h = gd_hash_cell(0, 0, 0, 1024);
        assert!(h < 1024);
    }

    #[test]
    fn hash_cell_different_coords_likely_different() {
        // Not guaranteed but extremely unlikely to collide for adjacent cells.
        let h1 = gd_hash_cell(0, 0, 0, 65536);
        let h2 = gd_hash_cell(1, 0, 0, 65536);
        let h3 = gd_hash_cell(0, 1, 0, 65536);
        // At least two should differ (probabilistic, but primes make this essentially certain).
        assert!(h1 != h2 || h1 != h3);
    }

    #[test]
    fn hash_cell_large_coords_no_panic() {
        // Should not panic even with extreme values.
        let _ = gd_hash_cell(i32::MAX, i32::MIN, i32::MAX, 1024);
    }

    #[test]
    fn hash_cell_single_bucket() {
        let h = gd_hash_cell(100, -50, 200, 1);
        assert_eq!(h, 0);
    }

    // -----------------------------------------------------------------------
    // gd_world_to_cell
    // -----------------------------------------------------------------------

    #[test]
    fn world_to_cell_at_origin() {
        let cell = gd_world_to_cell([0.0, 0.0, 0.0], 1.0, [0.0, 0.0, 0.0]);
        assert_eq!(cell, [0, 0, 0]);
    }

    #[test]
    fn world_to_cell_at_one_cell() {
        let cell = gd_world_to_cell([1.0, 0.0, 0.0], 1.0, [0.0, 0.0, 0.0]);
        assert_eq!(cell, [1, 0, 0]);
    }

    #[test]
    fn world_to_cell_fractional() {
        let cell = gd_world_to_cell([0.5, 0.5, 0.5], 1.0, [0.0, 0.0, 0.0]);
        assert_eq!(cell, [0, 0, 0]);
    }

    #[test]
    fn world_to_cell_negative_offset() {
        // Position at bounds_min should give cell [0,0,0].
        let bounds_min = [-5.0, -3.0, -1.0];
        let cell = gd_world_to_cell(bounds_min, 1.0, bounds_min);
        assert_eq!(cell, [0, 0, 0]);
    }

    #[test]
    fn world_to_cell_with_bounds_offset() {
        let bounds_min = [10.0, 0.0, 0.0];
        let cell = gd_world_to_cell([11.0, 0.0, 0.0], 1.0, bounds_min);
        assert_eq!(cell, [1, 0, 0]);
    }

    // -----------------------------------------------------------------------
    // SpatialHashMap
    // -----------------------------------------------------------------------

    #[test]
    fn spatial_hash_map_new_empty_n0() {
        let map = SpatialHashMap::new(64, 0.1, &[], 0);
        assert!(map.is_ok());
    }

    #[test]
    fn spatial_hash_map_new_invalid_cell_size() {
        let positions = vec![0.0f32; 3];
        let err = SpatialHashMap::new(64, -0.1, &positions, 1);
        assert!(matches!(
            err,
            Err(DeduplicatorError::InvalidCellSize { .. })
        ));
    }

    #[test]
    fn spatial_hash_map_query_finds_inserted_point() {
        let positions = vec![0.0f32, 0.0, 0.0, 5.0, 5.0, 5.0];
        let map = SpatialHashMap::new(64, 1.0, &positions, 2).unwrap();
        let neighbors = map.query_neighbors([0.0, 0.0, 0.0]);
        assert!(neighbors.contains(&0), "Should find index 0 near origin");
    }

    #[test]
    fn spatial_hash_map_query_does_not_find_distant() {
        let positions = vec![0.0f32, 0.0, 0.0, 100.0, 100.0, 100.0];
        let map = SpatialHashMap::new(64, 1.0, &positions, 2).unwrap();
        let near_origin = map.query_neighbors([0.0, 0.0, 0.0]);
        // Index 1 is far away; it might land in same bucket due to hash collision,
        // but it should NOT appear in the 3×3×3 cell neighborhood.
        // (We check that 0 is in the results at minimum.)
        assert!(near_origin.contains(&0));
    }

    // -----------------------------------------------------------------------
    // gd_are_duplicates
    // -----------------------------------------------------------------------

    fn default_sh() -> Vec<f32> {
        vec![]
    }

    #[test]
    fn are_duplicates_identical_positions() {
        let pos = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let op = vec![0.5f32, 0.5];
        let sc = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = make_config();
        assert!(gd_are_duplicates(
            0,
            1,
            GdSceneSlices {
                positions: &pos,
                opacities: &op,
                scales: &sc,
                sh_coeffs: &default_sh(),
                sh_channels: 0
            },
            &cfg
        ));
    }

    #[test]
    fn are_duplicates_far_apart_not_duplicate() {
        let pos = vec![0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0];
        let op = vec![0.5f32, 0.5];
        let sc = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = make_config();
        assert!(!gd_are_duplicates(
            0,
            1,
            GdSceneSlices {
                positions: &pos,
                opacities: &op,
                scales: &sc,
                sh_coeffs: &default_sh(),
                sh_channels: 0
            },
            &cfg
        ));
    }

    #[test]
    fn are_duplicates_same_pos_different_opacity() {
        let pos = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let op = vec![0.0f32, 1.0]; // diff = 1.0 >> threshold 0.1
        let sc = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = make_config();
        assert!(!gd_are_duplicates(
            0,
            1,
            GdSceneSlices {
                positions: &pos,
                opacities: &op,
                scales: &sc,
                sh_coeffs: &default_sh(),
                sh_channels: 0
            },
            &cfg
        ));
    }

    #[test]
    fn are_duplicates_within_position_threshold() {
        let pos = vec![0.0f32, 0.0, 0.0, 0.005, 0.0, 0.0]; // dist = 0.005 < 0.01
        let op = vec![0.5f32, 0.5];
        let sc = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = make_config();
        assert!(gd_are_duplicates(
            0,
            1,
            GdSceneSlices {
                positions: &pos,
                opacities: &op,
                scales: &sc,
                sh_coeffs: &default_sh(),
                sh_channels: 0
            },
            &cfg
        ));
    }

    #[test]
    fn are_duplicates_beyond_position_threshold() {
        let pos = vec![0.0f32, 0.0, 0.0, 0.02, 0.0, 0.0]; // dist = 0.02 > 0.01
        let op = vec![0.5f32, 0.5];
        let sc = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = make_config();
        assert!(!gd_are_duplicates(
            0,
            1,
            GdSceneSlices {
                positions: &pos,
                opacities: &op,
                scales: &sc,
                sh_coeffs: &default_sh(),
                sh_channels: 0
            },
            &cfg
        ));
    }

    #[test]
    fn are_duplicates_opacity_within_threshold() {
        let pos = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let op = vec![0.5f32, 0.55]; // diff = 0.05 < 0.1
        let sc = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = make_config();
        assert!(gd_are_duplicates(
            0,
            1,
            GdSceneSlices {
                positions: &pos,
                opacities: &op,
                scales: &sc,
                sh_coeffs: &default_sh(),
                sh_channels: 0
            },
            &cfg
        ));
    }

    #[test]
    fn are_duplicates_scale_within_threshold() {
        let pos = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let op = vec![0.5f32, 0.5];
        // max_scale(0) = 1.0, max_scale(1) = 1.1; rel = 0.1/1.1 ≈ 0.09 < 0.2
        let sc = vec![1.0f32, 1.0, 1.0, 1.1, 1.0, 1.0];
        let cfg = make_config();
        assert!(gd_are_duplicates(
            0,
            1,
            GdSceneSlices {
                positions: &pos,
                opacities: &op,
                scales: &sc,
                sh_coeffs: &default_sh(),
                sh_channels: 0
            },
            &cfg
        ));
    }

    #[test]
    fn are_duplicates_color_filter() {
        let pos = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let op = vec![0.5f32, 0.5];
        let sc = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        // sh_channels=3; colors are very different
        let sh = vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut cfg = make_config();
        cfg.color_threshold = 0.1;
        assert!(!gd_are_duplicates(
            0,
            1,
            GdSceneSlices {
                positions: &pos,
                opacities: &op,
                scales: &sc,
                sh_coeffs: &sh,
                sh_channels: 3
            },
            &cfg
        ));
    }

    // -----------------------------------------------------------------------
    // gd_find_duplicates_brute
    // -----------------------------------------------------------------------

    #[test]
    fn brute_two_identical_one_group() {
        let (pos, _, sc, op, sh) = make_identical(2);
        let cfg = make_config();
        let groups = gd_find_duplicates_brute(&pos, &op, &sc, &sh, 3, 2, &cfg).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2);
    }

    #[test]
    fn brute_all_distinct_no_groups() {
        let (pos, _, sc, op, sh) = make_distinct(5);
        let cfg = make_config();
        let groups = gd_find_duplicates_brute(&pos, &op, &sc, &sh, 3, 5, &cfg).unwrap();
        assert_eq!(groups.len(), 0);
    }

    #[test]
    fn brute_three_identical_one_group_of_three() {
        let (pos, _, sc, op, sh) = make_identical(3);
        let cfg = make_config();
        let groups = gd_find_duplicates_brute(&pos, &op, &sc, &sh, 3, 3, &cfg).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
    }

    #[test]
    fn brute_n0_returns_empty() {
        let cfg = make_config();
        let groups = gd_find_duplicates_brute(&[], &[], &[], &[], 0, 0, &cfg).unwrap();
        assert_eq!(groups.len(), 0);
    }

    #[test]
    fn brute_length_mismatch_error() {
        let cfg = make_config();
        // positions only has 3 floats but n=2 expects 6.
        let err =
            gd_find_duplicates_brute(&[0.0, 0.0, 0.0], &[0.5, 0.5], &[1.0; 6], &[], 0, 2, &cfg);
        assert!(matches!(
            err,
            Err(DeduplicatorError::PositionLengthMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // gd_find_duplicates_spatial matches brute force
    // -----------------------------------------------------------------------

    #[test]
    fn spatial_matches_brute_two_identical() {
        let (pos, _, sc, op, sh) = make_identical(2);
        let mut cfg = make_config();
        cfg.cell_size = 0.02;
        let brute = gd_find_duplicates_brute(&pos, &op, &sc, &sh, 3, 2, &cfg).unwrap();
        let spatial = gd_find_duplicates_spatial(&pos, &op, &sc, &sh, 3, 2, &cfg).unwrap();
        assert_eq!(brute.len(), spatial.len(), "Group counts should match");
        assert_eq!(brute[0].len(), spatial[0].len(), "Group sizes should match");
    }

    #[test]
    fn spatial_matches_brute_all_distinct() {
        let (pos, _, sc, op, sh) = make_distinct(6);
        let mut cfg = make_config();
        cfg.cell_size = 0.02;
        let brute = gd_find_duplicates_brute(&pos, &op, &sc, &sh, 3, 6, &cfg).unwrap();
        let spatial = gd_find_duplicates_spatial(&pos, &op, &sc, &sh, 3, 6, &cfg).unwrap();
        assert_eq!(brute.len(), 0);
        assert_eq!(spatial.len(), 0);
    }

    // -----------------------------------------------------------------------
    // gd_pick_representative
    // -----------------------------------------------------------------------

    #[test]
    fn pick_representative_highest_opacity() {
        let opacities = vec![0.2f32, 0.8, 0.5];
        let scales = vec![1.0f32; 9];
        let group = vec![0usize, 1, 2];
        let rep = gd_pick_representative(
            &group,
            &opacities,
            &scales,
            &DedupKeepPolicy::KeepHighestOpacity,
        )
        .expect("group is non-empty");
        assert_eq!(rep, 1);
    }

    #[test]
    fn pick_representative_keep_first() {
        let opacities = vec![0.5f32; 3];
        let scales = vec![1.0f32; 9];
        let group = vec![2usize, 0, 1];
        let rep = gd_pick_representative(&group, &opacities, &scales, &DedupKeepPolicy::KeepFirst)
            .expect("group is non-empty");
        assert_eq!(rep, 0);
    }

    #[test]
    fn pick_representative_keep_last() {
        let opacities = vec![0.5f32; 3];
        let scales = vec![1.0f32; 9];
        let group = vec![0usize, 1, 2];
        let rep = gd_pick_representative(&group, &opacities, &scales, &DedupKeepPolicy::KeepLast)
            .expect("group is non-empty");
        assert_eq!(rep, 2);
    }

    #[test]
    fn pick_representative_largest_scale() {
        let opacities = vec![0.5f32; 3];
        // max_scale(0)=1, max_scale(1)=3, max_scale(2)=2
        let scales = vec![1.0f32, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0, 1.0];
        let group = vec![0usize, 1, 2];
        let rep = gd_pick_representative(
            &group,
            &opacities,
            &scales,
            &DedupKeepPolicy::KeepLargestScale,
        )
        .expect("group is non-empty");
        assert_eq!(rep, 1);
    }

    #[test]
    fn pick_representative_smallest_scale() {
        let opacities = vec![0.5f32; 3];
        let scales = vec![1.0f32, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0, 1.0];
        let group = vec![0usize, 1, 2];
        let rep = gd_pick_representative(
            &group,
            &opacities,
            &scales,
            &DedupKeepPolicy::KeepSmallestScale,
        )
        .expect("group is non-empty");
        assert_eq!(rep, 0);
    }

    #[test]
    fn pick_representative_empty_group_returns_none_not_panic() {
        // KeepFirst/KeepLast used to evaluate group[0] as an eager
        // unwrap_or fallback, panicking on an empty group.
        let opacities: Vec<f32> = vec![];
        let scales: Vec<f32> = vec![];
        let group: Vec<usize> = vec![];
        for policy in [
            DedupKeepPolicy::KeepHighestOpacity,
            DedupKeepPolicy::KeepLargestScale,
            DedupKeepPolicy::KeepSmallestScale,
            DedupKeepPolicy::KeepFirst,
            DedupKeepPolicy::KeepLast,
        ] {
            assert_eq!(
                gd_pick_representative(&group, &opacities, &scales, &policy),
                None,
                "empty group must return None for {policy:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // gd_build_remove_mask
    // -----------------------------------------------------------------------

    #[test]
    fn build_remove_mask_group_of_two_keep_first() {
        let opacities = vec![0.5f32, 0.5];
        let scales = vec![1.0f32; 6];
        let groups = vec![vec![0usize, 1]];
        let mask =
            gd_build_remove_mask(&groups, 2, &opacities, &scales, &DedupKeepPolicy::KeepFirst);
        assert!(!mask[0], "index 0 kept");
        assert!(mask[1], "index 1 removed");
    }

    #[test]
    fn build_remove_mask_no_groups() {
        let opacities = vec![0.5f32; 3];
        let scales = vec![1.0f32; 9];
        let groups: Vec<Vec<usize>> = vec![];
        let mask =
            gd_build_remove_mask(&groups, 3, &opacities, &scales, &DedupKeepPolicy::KeepFirst);
        assert!(mask.iter().all(|&r| !r), "Nothing removed if no groups");
    }

    // -----------------------------------------------------------------------
    // gd_apply_mask / gd_apply_scalar_mask
    // -----------------------------------------------------------------------

    #[test]
    fn apply_mask_stride3_removes_correct_rows() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mask = vec![false, true, false]; // remove middle row
        let result = gd_apply_mask(&data, &mask, 3).expect("lengths match");
        assert_eq!(result, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn apply_mask_keeps_all_when_no_removal() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = vec![false, false];
        let result = gd_apply_mask(&data, &mask, 3).expect("lengths match");
        assert_eq!(result, data);
    }

    #[test]
    fn apply_scalar_mask_removes_elements() {
        let data = vec![0.1f32, 0.5, 0.9];
        let mask = vec![true, false, false];
        let result = gd_apply_scalar_mask(&data, &mask).expect("lengths match");
        assert_eq!(result, vec![0.5, 0.9]);
    }

    #[test]
    fn apply_scalar_mask_removes_all() {
        let data = vec![1.0f32, 2.0, 3.0];
        let mask = vec![true, true, true];
        let result = gd_apply_scalar_mask(&data, &mask).expect("lengths match");
        assert!(result.is_empty());
    }

    #[test]
    fn apply_mask_length_mismatch_is_error_not_panic() {
        // A short `data` used to panic ("slice index starts at ... but ends
        // at ..."); a length between row boundaries silently emitted a
        // truncated, misaligned row. Both must now error up front.
        let short = gd_apply_mask(&[1.0f32, 2.0, 3.0], &[false, false], 3);
        assert!(matches!(
            short,
            Err(DeduplicatorError::AttributeLengthMismatch { .. })
        ));
        let partial_row = gd_apply_mask(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[false, false], 3);
        assert!(matches!(
            partial_row,
            Err(DeduplicatorError::AttributeLengthMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // gd_deduplicate
    // -----------------------------------------------------------------------

    #[test]
    fn deduplicate_two_identical_removes_one() {
        let (pos, rot, sc, op, sh) = make_identical(2);
        let cfg = make_config();
        let result = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 2,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(result.n_before, 2);
        assert_eq!(result.n_after, 1);
        assert_eq!(result.n_removed, 1);
    }

    #[test]
    fn deduplicate_all_distinct_no_removal() {
        let (pos, rot, sc, op, sh) = make_distinct(5);
        let cfg = make_config();
        let result = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 5,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(result.n_after, 5);
        assert_eq!(result.n_removed, 0);
    }

    #[test]
    fn deduplicate_preserves_array_lengths() {
        let (pos, rot, sc, op, sh) = make_identical(4);
        let cfg = make_config();
        let res = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 4,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(res.positions.len(), res.n_after * 3);
        assert_eq!(res.rotations.len(), res.n_after * 4);
        assert_eq!(res.scales.len(), res.n_after * 3);
        assert_eq!(res.opacities.len(), res.n_after);
        assert_eq!(res.sh_coefficients.len(), res.n_after * 3);
    }

    #[test]
    fn deduplicate_length_mismatch_error() {
        let pos = vec![0.0f32; 5]; // wrong: should be 2*3=6
        let rot = vec![0.0f32; 8];
        let sc = vec![1.0f32; 6];
        let op = vec![0.5f32; 2];
        let sh = vec![0.0f32; 6];
        let cfg = make_config();
        let err = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 2,
            },
            &cfg,
        );
        assert!(matches!(
            err,
            Err(DeduplicatorError::PositionLengthMismatch { .. })
        ));
    }

    #[test]
    fn deduplicate_empty_scene_error() {
        let cfg = make_config();
        let err = gd_deduplicate(
            GdDeduplicateInput {
                positions: &[],
                rotations: &[],
                scales: &[],
                opacities: &[],
                sh_coefficients: &[],
                sh_channels: 0,
                n_gaussians: 0,
            },
            &cfg,
        );
        assert!(matches!(err, Err(DeduplicatorError::EmptyScene)));
    }

    #[test]
    fn deduplicate_10_gaussians_3_exact_duplicates() {
        // Gaussians 0-6 are distinct; 7,8,9 are copies of 0.
        let mut pos = vec![0.0f32; 10 * 3];
        let mut rot = vec![0.0f32; 10 * 4];
        let sc = vec![1.0f32; 10 * 3];
        let op = vec![0.5f32; 10];
        let sh = vec![0.0f32; 10 * 3];

        // Distinct positions for 0..7
        for i in 0..7 {
            pos[i * 3] = i as f32 * 10.0;
        }
        // rot w=1 for all
        for i in 0..10 {
            rot[i * 4 + 3] = 1.0;
        }
        // 7,8,9 copy position of 0 (already 0.0)
        // → group [0,7,8,9]
        let cfg = make_config();
        let result = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 10,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(result.n_removed, 3, "3 duplicates of index 0 removed");
        assert_eq!(result.n_after, 7);
        // group_sizes must reflect the real group (0,7,8,9), size 4.
        assert_eq!(result.group_sizes.len(), result.n_groups);
        assert_eq!(result.group_sizes.iter().copied().max(), Some(4));
        let _ = (rot, sc, op, sh); // suppress unused warnings
    }

    #[test]
    fn deduplicate_multiple_groups_three_pairs() {
        // 3 pairs: (0,1), (2,3), (4,5); all at same pos within pair.
        let mut pos = vec![0.0f32; 6 * 3];
        let _rot = (0..6)
            .flat_map(|_| [0.0f32, 0.0, 0.0, 1.0])
            .collect::<Vec<_>>();
        let sc = vec![1.0f32; 6 * 3];
        let op = vec![0.5f32; 6];
        let sh = vec![0.0f32; 6 * 3];

        // Pair centroids far apart.
        pos[0] = 0.0;
        pos[3] = 0.0;
        pos[6] = 10.0;
        pos[9] = 10.0;
        pos[12] = 20.0;
        pos[15] = 20.0;

        let cfg = make_config();
        let groups = gd_find_duplicates_brute(&pos, &op, &sc, &sh, 3, 6, &cfg).unwrap();
        assert_eq!(groups.len(), 3, "Three groups expected");
    }

    // -----------------------------------------------------------------------
    // gd_analyze_duplicates
    // -----------------------------------------------------------------------

    #[test]
    fn analyze_duplicates_centroid_correct() {
        // Two identical Gaussians at (0,0,0) → centroid=(0,0,0)
        let (pos, _, sc, op, sh) = make_identical(2);
        let cfg = make_config();
        let groups = gd_analyze_duplicates(&pos, &op, &sc, &sh, 3, 2, &cfg).unwrap();
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert!((g.centroid[0]).abs() < 1e-6);
        assert!((g.centroid[1]).abs() < 1e-6);
        assert!((g.centroid[2]).abs() < 1e-6);
    }

    #[test]
    fn analyze_duplicates_spread_zero_for_identical() {
        let (pos, _, sc, op, sh) = make_identical(3);
        let cfg = make_config();
        let groups = gd_analyze_duplicates(&pos, &op, &sc, &sh, 3, 3, &cfg).unwrap();
        assert_eq!(groups.len(), 1);
        assert!(groups[0].max_position_spread < 1e-6);
    }

    #[test]
    fn analyze_duplicates_empty_scene_error() {
        let cfg = make_config();
        let err = gd_analyze_duplicates(&[], &[], &[], &[], 0, 0, &cfg);
        assert!(matches!(err, Err(DeduplicatorError::EmptyScene)));
    }

    #[test]
    fn analyze_duplicates_mismatched_opacities_is_error_not_panic() {
        // Never validated anything beyond n_gaussians == 0 (not even
        // positions); relied on gd_are_duplicates' unchecked indexing.
        let positions = vec![0.0f32; 3 * 3];
        let scales = vec![1.0f32; 3 * 3];
        let opacities = vec![0.5f32]; // too short
        let sh = vec![0.0f32; 3 * 3];
        let cfg = make_config();
        let err = gd_analyze_duplicates(&positions, &opacities, &scales, &sh, 3, 3, &cfg);
        assert!(
            matches!(err, Err(DeduplicatorError::AttributeLengthMismatch { .. })),
            "expected AttributeLengthMismatch, got {err:?}"
        );
    }

    #[test]
    fn deduplicate_short_sh_coeffs_with_3_channels_is_error_not_panic() {
        // The sh_channels >= 3 guard only checks !sh_coeffs.is_empty(),
        // which a short-but-nonempty array passes and then indexes past.
        let (pos, rot, sc, op, _) = make_identical(100);
        let sh = vec![0.0f32; 3]; // far too short for n=100, sh_channels=3
        let cfg = make_config();
        let err = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 100,
            },
            &cfg,
        );
        assert!(
            matches!(err, Err(DeduplicatorError::AttributeLengthMismatch { .. })),
            "expected AttributeLengthMismatch, got {err:?}"
        );
    }

    #[test]
    fn analyze_duplicates_mean_opacity_correct() {
        // All 4 Gaussians at same position with very similar opacities → one group.
        // Opacities differ by <= 0.02 < threshold 0.1.
        let pos = vec![0.0f32; 4 * 3];
        let sc = vec![1.0f32; 4 * 3];
        let op = vec![0.48f32, 0.50, 0.50, 0.52];
        let sh = vec![0.0f32; 4 * 3];
        let cfg = make_config();
        let groups = gd_analyze_duplicates(&pos, &op, &sc, &sh, 3, 4, &cfg).unwrap();
        assert_eq!(groups.len(), 1);
        let mean = groups[0].mean_opacity;
        let expected = (0.48f32 + 0.50 + 0.50 + 0.52) / 4.0; // 0.5
        assert!(
            (mean - expected).abs() < 1e-5,
            "mean opacity should be {}, got {}",
            expected,
            mean
        );
    }

    // -----------------------------------------------------------------------
    // gd_compute_stats
    // -----------------------------------------------------------------------

    #[test]
    fn compute_stats_reduction_percent() {
        let result = DedupResult {
            positions: vec![],
            rotations: vec![],
            scales: vec![],
            opacities: vec![],
            sh_coefficients: vec![],
            n_before: 100,
            n_after: 75,
            n_removed: 25,
            n_groups: 5,
            group_sizes: vec![6, 6, 6, 6, 6], // 5 groups of 6 => removes 5 each = 25
        };
        let stats = gd_compute_stats(&result, 3);
        assert!((stats.reduction_percent - 25.0).abs() < 1e-3);
    }

    #[test]
    fn compute_stats_memory_saved_bytes() {
        let result = DedupResult {
            positions: vec![],
            rotations: vec![],
            scales: vec![],
            opacities: vec![],
            sh_coefficients: vec![],
            n_before: 10,
            n_after: 8,
            n_removed: 2,
            n_groups: 1,
            group_sizes: vec![3], // 1 group of 3 => removes 2, keeps 1
        };
        // bytes_per_gaussian = (3+4+3+1+3)*4 = 56
        let stats = gd_compute_stats(&result, 3);
        assert_eq!(stats.memory_saved_bytes, 2 * 56);
    }

    #[test]
    fn compute_stats_zero_groups_zero_mean() {
        let result = DedupResult {
            positions: vec![],
            rotations: vec![],
            scales: vec![],
            opacities: vec![],
            sh_coefficients: vec![],
            n_before: 10,
            n_after: 10,
            n_removed: 0,
            n_groups: 0,
            group_sizes: vec![],
        };
        let stats = gd_compute_stats(&result, 3);
        assert_eq!(stats.mean_group_size, 0.0);
        assert_eq!(stats.max_group_size, 0);
    }

    #[test]
    fn compute_stats_max_group_size_is_actual_max_not_total_members() {
        // Groups of 2 and 5: old formula (n_removed+n_groups).max(2)=7 — no
        // group has 7 members; true max is 5.
        let result = DedupResult {
            positions: vec![],
            rotations: vec![],
            scales: vec![],
            opacities: vec![],
            sh_coefficients: vec![],
            n_before: 7,
            n_after: 2,
            n_removed: 5,
            n_groups: 2,
            group_sizes: vec![2, 5],
        };
        let stats = gd_compute_stats(&result, 3);
        assert_eq!(
            stats.max_group_size, 5,
            "max group size must be the largest actual group, not total members"
        );
        let expected_mean = (2 + 5) as f32 / 2.0;
        assert!((stats.mean_group_size - expected_mean).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // gd_format_stats / gd_format_report
    // -----------------------------------------------------------------------

    #[test]
    fn format_stats_nonempty_string() {
        let result = DedupResult {
            positions: vec![],
            rotations: vec![],
            scales: vec![],
            opacities: vec![],
            sh_coefficients: vec![],
            n_before: 50,
            n_after: 40,
            n_removed: 10,
            n_groups: 2,
            group_sizes: vec![6, 6], // 2 groups of 6 => removes 5 each = 10
        };
        let stats = gd_compute_stats(&result, 3);
        let s = gd_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("50"));
    }

    #[test]
    fn format_report_nonempty_contains_group_count() {
        let result = DedupResult {
            positions: vec![],
            rotations: vec![],
            scales: vec![],
            opacities: vec![],
            sh_coefficients: vec![],
            n_before: 10,
            n_after: 8,
            n_removed: 2,
            n_groups: 1,
            group_sizes: vec![3],
        };
        let groups: Vec<DuplicateGroup> = vec![DuplicateGroup {
            indices: vec![0, 1],
            centroid: [0.0; 3],
            mean_opacity: 0.5,
            max_position_spread: 0.0,
        }];
        let cfg = make_config();
        let report = gd_build_report(&result, groups, &cfg, 3);
        let s = gd_format_report(&report);
        assert!(!s.is_empty());
        assert!(s.contains("1"), "Should mention 1 group");
    }

    #[test]
    fn format_config_nonempty() {
        let cfg = make_config();
        let s = gd_format_config(&cfg);
        assert!(!s.is_empty());
        assert!(s.contains("KeepHighestOpacity"));
    }

    // -----------------------------------------------------------------------
    // Spatial hash pipeline with use_spatial_hash=true
    // -----------------------------------------------------------------------

    #[test]
    fn deduplicate_spatial_two_identical() {
        let (pos, rot, sc, op, sh) = make_identical(2);
        let mut cfg = make_config();
        cfg.use_spatial_hash = true;
        cfg.cell_size = 0.02;
        let result = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 2,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(result.n_removed, 1);
    }

    #[test]
    fn deduplicate_spatial_all_distinct_no_removal() {
        let (pos, rot, sc, op, sh) = make_distinct(8);
        let mut cfg = make_config();
        cfg.use_spatial_hash = true;
        cfg.cell_size = 0.5;
        let result = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 8,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(result.n_removed, 0);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn deduplicate_single_gaussian_no_groups() {
        let pos = vec![1.0f32, 2.0, 3.0];
        let rot = vec![0.0f32, 0.0, 0.0, 1.0];
        let sc = vec![1.0f32, 1.0, 1.0];
        let op = vec![0.5f32];
        let sh = vec![0.1f32, 0.2, 0.3];
        let cfg = make_config();
        let result = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 1,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(result.n_after, 1);
        assert_eq!(result.n_removed, 0);
    }

    #[test]
    fn deduplicate_no_sh_channels() {
        let pos = vec![0.0f32; 4 * 3];
        let rot = [0.0f32, 0.0, 0.0, 1.0].repeat(4);
        let sc = vec![1.0f32; 4 * 3];
        let op = vec![0.5f32; 4];
        let sh: Vec<f32> = vec![];
        let cfg = make_config();
        let result = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 0,
                n_gaussians: 4,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(result.n_removed, 3);
        assert!(result.sh_coefficients.is_empty());
    }

    #[test]
    fn pick_representative_highest_opacity_among_three() {
        // [0.2, 0.8, 0.5] → index 1 has highest
        let op = vec![0.2f32, 0.8, 0.5];
        let sc = vec![1.0f32; 9];
        let group = vec![0usize, 1, 2];
        let rep = gd_pick_representative(&group, &op, &sc, &DedupKeepPolicy::KeepHighestOpacity)
            .expect("group is non-empty");
        assert_eq!(rep, 1);
    }

    #[test]
    fn world_to_cell_negative_position_handled() {
        // Position below bounds_min should give cell < 0, which is fine for i32.
        let bounds_min = [0.0f32, 0.0, 0.0];
        // A position at -0.5 with cell_size 1.0 → floor(-0.5) = -1
        let cell = gd_world_to_cell([-0.5, 0.0, 0.0], 1.0, bounds_min);
        assert_eq!(cell[0], -1);
    }
}
