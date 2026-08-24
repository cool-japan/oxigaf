//! Test suite for the parent `scene_optimizer` module, split into its own
//! file to keep `scene_optimizer.rs` itself under the workspace file-size policy.

use super::*;

// --- so_compute_snapshot ---

#[test]
fn test_snapshot_single_gaussian_values() {
    let positions = vec![1.0f32, 2.0, 3.0];
    let scales = vec![0.1f32, 0.2, 0.3];
    let opacities = vec![0.0f32]; // sigmoid(0) = 0.5
    let snap = so_compute_snapshot(&positions, &scales, &opacities, 1, 3, OpacitySpace::Logit);
    assert_eq!(snap.n_gaussians, 1);
    assert!((snap.mean_opacity - 0.5).abs() < 1e-5, "mean_opacity should be ~0.5");
    assert!((snap.max_opacity - 0.5).abs() < 1e-5, "max_opacity should be ~0.5");
    assert!((snap.mean_scale - 0.2).abs() < 1e-5, "mean_scale should be ~0.2");
    assert!((snap.max_scale - 0.3).abs() < 1e-5, "max_scale should be 0.3");
}

#[test]
fn test_snapshot_memory_bytes() {
    let n = 10;
    let sh_channels = 9;
    let positions = vec![0.0f32; n * 3];
    let scales = vec![1.0f32; n * 3];
    let opacities = vec![0.0f32; n];
    let snap =
        so_compute_snapshot(&positions, &scales, &opacities, n, sh_channels, OpacitySpace::Logit);
    let expected = (3 + 4 + 3 + 1 + sh_channels) * 4 * n;
    assert_eq!(snap.memory_bytes, expected);
}

#[test]
fn test_snapshot_empty() {
    let snap = so_compute_snapshot(&[], &[], &[], 0, 3, OpacitySpace::Logit);
    assert_eq!(snap.n_gaussians, 0);
    assert_eq!(snap.memory_bytes, 0);
}

#[test]
fn test_snapshot_bounds() {
    let positions = vec![1.0f32, 2.0, 3.0, -1.0, -2.0, -3.0];
    let scales = vec![0.1f32; 6];
    let opacities = vec![0.0f32; 2];
    let snap = so_compute_snapshot(&positions, &scales, &opacities, 2, 0, OpacitySpace::Logit);
    assert!((snap.bounds_min[0] - (-1.0)).abs() < 1e-5);
    assert!((snap.bounds_max[0] - 1.0).abs() < 1e-5);
}

// --- so_prune_by_opacity ---

#[test]
fn test_prune_high_logit_kept() {
    let mask = so_prune_by_opacity(&[10.0f32], 1, 0.5, OpacitySpace::Logit);
    assert!(mask[0], "logit=10 (sigmoid≈1) should be kept with threshold=0.5");
}

#[test]
fn test_prune_low_logit_removed() {
    let mask = so_prune_by_opacity(&[-10.0f32], 1, 0.5, OpacitySpace::Logit);
    assert!(!mask[0], "logit=-10 (sigmoid≈0) should be removed with threshold=0.5");
}

#[test]
fn test_prune_all_above_threshold() {
    let opacities = vec![5.0f32, 3.0, 2.0];
    let mask = so_prune_by_opacity(&opacities, 3, 0.5, OpacitySpace::Logit);
    assert!(mask.iter().all(|&v| v));
}

#[test]
fn test_prune_threshold_zero_keeps_all() {
    // sigmoid(x) > 0 for all finite x, so threshold=0 keeps everything
    let opacities = vec![-100.0f32, -10.0, 0.0, 10.0];
    let mask = so_prune_by_opacity(&opacities, 4, 0.0, OpacitySpace::Logit);
    assert!(mask.iter().all(|&v| v), "threshold=0 must keep all Gaussians");
}

#[test]
fn test_prune_mixed() {
    let opacities = vec![5.0f32, -5.0]; // sigmoid(5)≈0.993, sigmoid(-5)≈0.007
    let mask = so_prune_by_opacity(&opacities, 2, 0.5, OpacitySpace::Logit);
    assert!(mask[0]);
    assert!(!mask[1]);
}

// --- so_deduplicate_near ---

#[test]
fn test_dedup_identical_positions_keeps_first() {
    let positions = vec![1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0];
    let mask = so_deduplicate_near(&positions, 2, 0.01);
    assert!(mask[0], "first should be kept");
    assert!(!mask[1], "duplicate should be removed");
}

#[test]
fn test_dedup_distinct_positions_all_kept() {
    let positions = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let mask = so_deduplicate_near(&positions, 3, 0.01);
    assert!(mask.iter().all(|&v| v));
}

#[test]
fn test_dedup_radius_zero_keeps_all() {
    let positions = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mask = so_deduplicate_near(&positions, 2, 0.0);
    assert!(mask.iter().all(|&v| v), "radius=0 must keep all");
}

#[test]
fn test_dedup_single_gaussian() {
    let positions = vec![0.0f32, 0.0, 0.0];
    let mask = so_deduplicate_near(&positions, 1, 0.1);
    assert!(mask[0]);
}

// --- so_clamp_scales ---

#[test]
fn test_clamp_above_max() {
    let mut scales = vec![0.5f32, 1.0, 2.0];
    so_clamp_scales(&mut scales, 1, 0.0, 0.3);
    // Only first Gaussian (3 components), all above max=0.3
    assert!(scales[0] <= 0.3);
    assert!(scales[1] <= 0.3);
    assert!(scales[2] <= 0.3);
}

#[test]
fn test_clamp_below_min() {
    let mut scales = vec![0.001f32, 0.002, 0.003];
    so_clamp_scales(&mut scales, 1, 0.01, 1.0);
    assert!(scales[0] >= 0.01);
}

#[test]
fn test_clamp_in_range_unchanged() {
    let mut scales = vec![0.05f32, 0.06, 0.07];
    so_clamp_scales(&mut scales, 1, 0.01, 0.1);
    assert!((scales[0] - 0.05).abs() < 1e-7);
    assert!((scales[1] - 0.06).abs() < 1e-7);
    assert!((scales[2] - 0.07).abs() < 1e-7);
}

#[test]
fn test_clamp_multiple_gaussians() {
    let mut scales = vec![0.2f32, 0.3, 0.4, 0.001, 0.002, 0.003];
    so_clamp_scales(&mut scales, 2, 0.01, 0.15);
    for &v in &scales {
        assert!(v >= 0.01 && v <= 0.15);
    }
}

// --- so_sort_morton ---

#[test]
fn test_sort_morton_non_decreasing_codes() {
    let positions = vec![
        0.5f32, 0.5, 0.5,
        0.1, 0.1, 0.1,
        0.9, 0.9, 0.9,
        0.3, 0.3, 0.3,
    ];
    let indices = so_sort_morton(&positions, 4);
    assert_eq!(indices.len(), 4);

    // Compute bounds
    let bounds_min = [0.1f32; 3];
    let bounds_max = [0.9f32; 3];

    let codes: Vec<u32> = indices
        .iter()
        .map(|&i| {
            let pos = [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
            let [qx, qy, qz] = so_quantize_position(pos, bounds_min, bounds_max, 10);
            so_morton_code(qx, qy, qz)
        })
        .collect();

    for w in codes.windows(2) {
        assert!(w[0] <= w[1], "Morton codes must be non-decreasing");
    }
}

#[test]
fn test_sort_morton_single() {
    let indices = so_sort_morton(&[0.0f32, 0.0, 0.0], 1);
    assert_eq!(indices, vec![0]);
}

#[test]
fn test_sort_morton_empty() {
    let indices = so_sort_morton(&[], 0);
    assert!(indices.is_empty());
}

// --- so_top_n_by_opacity ---

#[test]
fn test_top_n_keeps_highest() {
    let opacities = vec![-5.0f32, 10.0, -3.0]; // sigmoid: ~0.007, ~1.0, ~0.047
    let mask = so_top_n_by_opacity(&opacities, 3, 1, OpacitySpace::Logit);
    assert!(!mask[0]);
    assert!(mask[1], "highest opacity (index 1) must be kept");
    assert!(!mask[2]);
}

#[test]
fn test_top_n_keep_all() {
    let opacities = vec![1.0f32, 2.0, 3.0];
    let mask = so_top_n_by_opacity(&opacities, 3, 3, OpacitySpace::Logit);
    assert!(mask.iter().all(|&v| v));
}

#[test]
fn test_top_n_keep_more_than_total() {
    let opacities = vec![1.0f32, 2.0];
    let mask = so_top_n_by_opacity(&opacities, 2, 100, OpacitySpace::Logit);
    assert!(mask.iter().all(|&v| v));
}

#[test]
fn test_top_n_count_correct() {
    let opacities = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let mask = so_top_n_by_opacity(&opacities, 5, 2, OpacitySpace::Logit);
    let kept = mask.iter().filter(|&&v| v).count();
    assert_eq!(kept, 2);
}

// --- so_normalize_opacity ---

#[test]
fn test_normalize_zero_logit() {
    let result = so_normalize_opacity(&[0.0f32], 1);
    assert!((result[0] - 0.5).abs() < 1e-5, "sigmoid(0) must be 0.5");
}

#[test]
fn test_normalize_large_positive_logit() {
    let result = so_normalize_opacity(&[100.0f32], 1);
    assert!(result[0] > 0.999, "sigmoid(100) must be ≈1.0");
}

#[test]
fn test_normalize_large_negative_logit() {
    let result = so_normalize_opacity(&[-100.0f32], 1);
    assert!(result[0] < 0.001, "sigmoid(-100) must be ≈0.0");
}

#[test]
fn test_normalize_all_in_range() {
    let opacities = vec![-5.0f32, -1.0, 0.0, 1.0, 5.0];
    let result = so_normalize_opacity(&opacities, 5);
    for &v in &result {
        assert!(v > 0.0 && v < 1.0);
    }
}

// --- so_clip_to_sphere ---

#[test]
fn test_clip_sphere_center_kept() {
    let positions = vec![0.0f32, 0.0, 0.0];
    let mask = so_clip_to_sphere(&positions, 1, [0.0, 0.0, 0.0], 1.0);
    assert!(mask[0]);
}

#[test]
fn test_clip_sphere_outside_removed() {
    let positions = vec![10.0f32, 0.0, 0.0];
    let mask = so_clip_to_sphere(&positions, 1, [0.0, 0.0, 0.0], 1.0);
    assert!(!mask[0]);
}

#[test]
fn test_clip_sphere_on_boundary_kept() {
    let positions = vec![1.0f32, 0.0, 0.0];
    let mask = so_clip_to_sphere(&positions, 1, [0.0, 0.0, 0.0], 1.0);
    assert!(mask[0], "point exactly on boundary should be kept");
}

// --- so_clip_to_aabb ---

#[test]
fn test_clip_aabb_inside_kept() {
    let positions = vec![0.5f32, 0.5, 0.5];
    let mask = so_clip_to_aabb(&positions, 1, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    assert!(mask[0]);
}

#[test]
fn test_clip_aabb_outside_removed() {
    let positions = vec![2.0f32, 0.5, 0.5];
    let mask = so_clip_to_aabb(&positions, 1, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    assert!(!mask[0]);
}

#[test]
fn test_clip_aabb_mixed() {
    let positions = vec![
        0.5f32, 0.5, 0.5, // inside
        2.0, 0.5, 0.5, // outside x
    ];
    let mask = so_clip_to_aabb(&positions, 2, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    assert!(mask[0]);
    assert!(!mask[1]);
}

// --- mask helpers ---

#[test]
fn test_apply_keep_mask_nd_stride3() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mask = vec![true, false];
    let result = so_apply_keep_mask_nd(&data, &mask, 2, 3);
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_apply_keep_mask_1d_scalar() {
    let data = vec![10.0f32, 20.0, 30.0];
    let mask = vec![true, false, true];
    let result = so_apply_keep_mask_1d(&data, &mask, 3);
    assert_eq!(result, vec![10.0, 30.0]);
}

#[test]
fn test_apply_keep_mask_3d() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mask = vec![false, true, false];
    let result = so_apply_keep_mask_3d(&data, &mask, 3);
    assert_eq!(result, vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_apply_keep_mask_4d() {
    let data = vec![
        1.0f32, 2.0, 3.0, 4.0, // row 0
        5.0, 6.0, 7.0, 8.0, // row 1
    ];
    let mask = vec![false, true];
    let result = so_apply_keep_mask_4d(&data, &mask, 2);
    assert_eq!(result, vec![5.0, 6.0, 7.0, 8.0]);
}

// --- so_reorder_by_indices ---

#[test]
fn test_reorder_identity() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices = vec![0, 1];
    let result = so_reorder_by_indices(&data, &indices, 2, 3);
    assert_eq!(result, data);
}

#[test]
fn test_reorder_reversal() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices = vec![1, 0];
    let result = so_reorder_by_indices(&data, &indices, 2, 3);
    assert_eq!(result, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
}

// --- so_morton_interleave ---

#[test]
fn test_morton_interleave_zero() {
    assert_eq!(so_morton_interleave(0), 0);
}

#[test]
fn test_morton_interleave_one() {
    assert_eq!(so_morton_interleave(1), 1);
}

#[test]
fn test_morton_interleave_two() {
    // 2 = 0b10 → bit 1 spreads to position 3 → 0b1000 = 8
    assert_eq!(so_morton_interleave(2), 8);
}

// --- so_morton_code ---

#[test]
fn test_morton_code_origin() {
    assert_eq!(so_morton_code(0, 0, 0), 0);
}

#[test]
fn test_morton_code_x1() {
    // (1,0,0): interleave(1)=1, rest=0 → 1
    assert_eq!(so_morton_code(1, 0, 0), 1);
}

#[test]
fn test_morton_code_y1() {
    // (0,1,0): interleave(1)<<1 = 2
    assert_eq!(so_morton_code(0, 1, 0), 2);
}

#[test]
fn test_morton_code_z1() {
    // (0,0,1): interleave(1)<<2 = 4
    assert_eq!(so_morton_code(0, 0, 1), 4);
}

// --- so_quantize_position ---

#[test]
fn test_quantize_min_maps_to_zero() {
    let [qx, qy, qz] =
        so_quantize_position([0.0, 0.0, 0.0], [0.0; 3], [1.0; 3], 10);
    assert_eq!(qx, 0);
    assert_eq!(qy, 0);
    assert_eq!(qz, 0);
}

#[test]
fn test_quantize_max_maps_to_2pow_bits_minus1() {
    let [qx, qy, qz] =
        so_quantize_position([1.0, 1.0, 1.0], [0.0; 3], [1.0; 3], 10);
    assert_eq!(qx, 1023);
    assert_eq!(qy, 1023);
    assert_eq!(qz, 1023);
}

#[test]
fn test_quantize_midpoint() {
    let [qx, _, _] =
        so_quantize_position([0.5, 0.0, 0.0], [0.0; 3], [1.0; 3], 10);
    // 0.5 * 1023 = 511.5 → rounds to 512
    assert_eq!(qx, 512);
}

// --- OptimizationPipeline::run ---

fn make_scene(n: usize, sh_ch: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let positions: Vec<f32> = (0..n * 3).map(|i| i as f32 * 0.01).collect();
    let rotations: Vec<f32> = (0..n)
        .flat_map(|_| [0.0f32, 0.0, 0.0, 1.0])
        .collect();
    let scales: Vec<f32> = vec![0.05f32; n * 3];
    let opacities: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 2.0).collect();
    let sh: Vec<f32> = vec![0.0f32; n * sh_ch];
    (positions, rotations, scales, opacities, sh)
}

#[test]
fn test_pipeline_empty_scene_error() {
    let config = SceneOptimizerConfig {
        steps: vec![OptimizationStep::PruneByOpacity { threshold: 0.5 }],
        sh_channels: 9,
        seed: 0,
    };
    let pipeline = OptimizationPipeline::new(config);
    let result = pipeline.run(&[], &[], &[], &[], &[], 0);
    assert!(matches!(result, Err(OptimizerError::EmptyScene)));
}

#[test]
fn test_pipeline_single_prune_step() {
    let n = 10;
    let sh_ch = 3;
    let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
    let config = SceneOptimizerConfig {
        steps: vec![OptimizationStep::PruneByOpacity { threshold: 0.5 }],
        sh_channels: sh_ch,
        seed: 0,
    };
    let pipeline = OptimizationPipeline::new(config);
    let result = pipeline.run(&pos, &rot, &scl, &op, &sh, n);
    assert!(result.is_ok());
    let (scene, report) = result.expect("pipeline should succeed");
    assert!(scene.n_gaussians <= n);
    assert_eq!(report.step_results.len(), 1);
}

#[test]
fn test_pipeline_n_after_le_n_before() {
    let n = 20;
    let sh_ch = 9;
    let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
    let config = SceneOptimizerConfig {
        steps: vec![
            OptimizationStep::PruneByOpacity { threshold: 0.3 },
            OptimizationStep::DeduplicateNear { position_radius: 0.05 },
        ],
        sh_channels: sh_ch,
        seed: 0,
    };
    let pipeline = OptimizationPipeline::new(config);
    let (scene, report) = pipeline.run(&pos, &rot, &scl, &op, &sh, n)
        .expect("pipeline should succeed");
    for step_result in &report.step_results {
        assert!(step_result.n_after <= step_result.n_before);
    }
    assert!(scene.n_gaussians <= n);
}

#[test]
fn test_pipeline_step_results_count() {
    let n = 5;
    let sh_ch = 3;
    let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
    let steps = vec![
        OptimizationStep::PruneByOpacity { threshold: 0.1 },
        OptimizationStep::ClampScales { min_scale: 0.01, max_scale: 0.5 },
        OptimizationStep::SortMorton,
    ];
    let config = SceneOptimizerConfig {
        steps: steps.clone(),
        sh_channels: sh_ch,
        seed: 0,
    };
    let pipeline = OptimizationPipeline::new(config);
    let (_, report) = pipeline.run(&pos, &rot, &scl, &op, &sh, n)
        .expect("pipeline should succeed");
    assert_eq!(report.step_results.len(), steps.len());
}

#[test]
fn test_pipeline_snapshot_after_matches_scene() {
    let n = 10;
    let sh_ch = 3;
    let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
    let config = SceneOptimizerConfig {
        steps: vec![OptimizationStep::PruneByOpacity { threshold: 0.5 }],
        sh_channels: sh_ch,
        seed: 0,
    };
    let pipeline = OptimizationPipeline::new(config);
    let (scene, report) = pipeline.run(&pos, &rot, &scl, &op, &sh, n)
        .expect("pipeline should succeed");
    assert_eq!(report.snapshot_after.n_gaussians, scene.n_gaussians);
}

#[test]
fn test_report_total_removed() {
    let n = 10;
    let sh_ch = 3;
    let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
    let config = SceneOptimizerConfig {
        steps: vec![OptimizationStep::PruneByOpacity { threshold: 0.5 }],
        sh_channels: sh_ch,
        seed: 0,
    };
    let pipeline = OptimizationPipeline::new(config);
    let (scene, report) = pipeline.run(&pos, &rot, &scl, &op, &sh, n)
        .expect("pipeline should succeed");
    assert_eq!(
        report.total_removed,
        n - scene.n_gaussians,
        "total_removed must equal n_before - n_after"
    );
}

#[test]
fn test_report_memory_saved() {
    let n = 10;
    let sh_ch = 9;
    let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
    let config = SceneOptimizerConfig {
        steps: vec![OptimizationStep::PruneByOpacity { threshold: 0.5 }],
        sh_channels: sh_ch,
        seed: 0,
    };
    let pipeline = OptimizationPipeline::new(config);
    let (_, report) = pipeline.run(&pos, &rot, &scl, &op, &sh, n)
        .expect("pipeline should succeed");
    let expected_saved = report.snapshot_before.memory_bytes
        .saturating_sub(report.snapshot_after.memory_bytes);
    assert_eq!(report.memory_saved_bytes, expected_saved);
}

// --- so_profile_config ---

#[test]
fn test_profile_quality_only_dedup() {
    let config = so_profile_config(OptimizationProfile::Quality, 9);
    assert_eq!(config.steps.len(), 1);
    assert!(
        matches!(config.steps[0], OptimizationStep::DeduplicateNear { .. }),
        "Quality profile should only contain DeduplicateNear"
    );
}

#[test]
fn test_profile_performance_includes_topn() {
    let config = so_profile_config(OptimizationProfile::Performance, 9);
    let has_topn = config.steps.iter().any(|s| matches!(s, OptimizationStep::TopNByOpacity { .. }));
    assert!(has_topn, "Performance profile must include TopNByOpacity");
}

#[test]
fn test_profile_streaming_includes_sort_morton() {
    let config = so_profile_config(OptimizationProfile::Streaming, 9);
    let has_sort = config.steps.iter().any(|s| matches!(s, OptimizationStep::SortMorton));
    assert!(has_sort, "Streaming profile must include SortMorton");
}

#[test]
fn test_profile_balanced_has_multiple_steps() {
    let config = so_profile_config(OptimizationProfile::Balanced, 9);
    assert!(config.steps.len() >= 2, "Balanced profile needs multiple steps");
}

#[test]
fn test_profile_balanced_clamp_scales_is_log_space() {
    // Bounds must be `ln` of the intended linear [1e-5, 0.1] world-unit
    // bounds, not the linear values (which would clamp log-scale into
    // [1e-5, 0.1], i.e. linear sizes ~[1.00001, 1.105]).
    let config = so_profile_config(OptimizationProfile::Balanced, 3);
    let (min_scale, max_scale) = config
        .steps
        .iter()
        .find_map(|s| match s {
            OptimizationStep::ClampScales { min_scale, max_scale } => Some((*min_scale, *max_scale)),
            _ => None,
        })
        .expect("Balanced profile must include a ClampScales step");

    assert!((min_scale - (1e-5f32).ln()).abs() < 1e-3, "got {min_scale}");
    assert!((max_scale - (0.1f32).ln()).abs() < 1e-3, "got {max_scale}");
}

// --- so_quick_optimize ---

#[test]
fn test_quick_optimize_quality_no_error() {
    let n = 5;
    let sh_ch = 3;
    let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
    let result = so_quick_optimize(&pos, &rot, &scl, &op, &sh, n, sh_ch, OptimizationProfile::Quality);
    assert!(result.is_ok(), "quick_optimize with Quality profile must succeed");
}

#[test]
fn test_quick_optimize_balanced_no_error() {
    let n = 5;
    let sh_ch = 3;
    let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
    let result = so_quick_optimize(&pos, &rot, &scl, &op, &sh, n, sh_ch, OptimizationProfile::Balanced);
    assert!(result.is_ok());
}

// --- formatting ---

#[test]
fn test_format_report_nonempty_with_step_count() {
    let n = 5;
    let sh_ch = 3;
    let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
    let config = SceneOptimizerConfig {
        steps: vec![OptimizationStep::PruneByOpacity { threshold: 0.5 }],
        sh_channels: sh_ch,
        seed: 0,
    };
    let pipeline = OptimizationPipeline::new(config);
    let (_, report) = pipeline.run(&pos, &rot, &scl, &op, &sh, n)
        .expect("pipeline should succeed");
    let s = so_format_report(&report);
    assert!(!s.is_empty());
    assert!(s.contains("Steps:"), "report must mention step count");
}

#[test]
fn test_format_snapshot_nonempty() {
    let snap =
        so_compute_snapshot(&[0.0f32, 0.0, 0.0], &[0.1f32, 0.1, 0.1], &[0.0f32], 1, 3, OpacitySpace::Logit);
    let s = so_format_snapshot(&snap);
    assert!(!s.is_empty());
}

#[test]
fn test_format_config_nonempty() {
    let config = so_profile_config(OptimizationProfile::Performance, 9);
    let s = so_format_config(&config);
    assert!(!s.is_empty());
}

#[test]
fn test_format_step_result_nonempty() {
    let result = OptimizationStepResult {
        step_name: "PruneByOpacity".to_string(),
        n_before: 100,
        n_after: 80,
        n_removed: 20,
        duration_hint: "fast".to_string(),
        notes: "threshold=0.5".to_string(),
    };
    let s = so_format_step_result(&result);
    assert!(!s.is_empty());
}

// --- error cases ---

#[test]
fn test_clamp_scales_invalid_min_gt_max_pipeline_error() {
    let n = 3;
    let sh_ch = 3;
    let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
    let config = SceneOptimizerConfig {
        steps: vec![OptimizationStep::ClampScales {
            min_scale: 1.0,
            max_scale: 0.01, // min > max → error
        }],
        sh_channels: sh_ch,
        seed: 0,
    };
    let pipeline = OptimizationPipeline::new(config);
    let result = pipeline.run(&pos, &rot, &scl, &op, &sh, n);
    assert!(
        matches!(result, Err(OptimizerError::InvalidThreshold { .. })),
        "min_scale > max_scale must produce InvalidThreshold"
    );
}

// --- additional coverage ---

#[test]
fn test_pipeline_sort_morton_preserves_count() {
    let n = 8;
    let sh_ch = 3;
    let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
    let config = SceneOptimizerConfig {
        steps: vec![OptimizationStep::SortMorton],
        sh_channels: sh_ch,
        seed: 0,
    };
    let pipeline = OptimizationPipeline::new(config);
    let (scene, _) = pipeline.run(&pos, &rot, &scl, &op, &sh, n)
        .expect("Morton sort pipeline should succeed");
    assert_eq!(scene.n_gaussians, n, "SortMorton must not remove any Gaussians");
}

#[test]
fn test_pipeline_normalize_opacity_preserves_count() {
    let n = 5;
    let sh_ch = 3;
    let (pos, rot, scl, op, sh) = make_scene(n, sh_ch);
    let config = SceneOptimizerConfig {
        steps: vec![OptimizationStep::NormalizeOpacity],
        sh_channels: sh_ch,
        seed: 0,
    };
    let pipeline = OptimizationPipeline::new(config);
    let (scene, _) = pipeline.run(&pos, &rot, &scl, &op, &sh, n)
        .expect("NormalizeOpacity pipeline should succeed");
    assert_eq!(scene.n_gaussians, n);
}

#[test]
fn test_pipeline_normalize_then_prune_then_renormalize_is_single_sigmoid() {
    // Logits [-3,0,3] activate once to [0.0474,0.5,0.9526]. Double-
    // sigmoiding (either in Prune, or via a non-idempotent trailing
    // NormalizeOpacity) would push the survivor to sigmoid(sigmoid(3))
    // ≈0.7217 instead of sigmoid(3)≈0.9526, and/or keep all 3 in Prune.
    let n = 3;
    let positions = vec![0.0f32; n * 3];
    let rotations: Vec<f32> = (0..n).flat_map(|_| [0.0f32, 0.0, 0.0, 1.0]).collect();
    let scales = vec![0.01f32; n * 3];
    let opacities = vec![-3.0f32, 0.0, 3.0];
    let sh: Vec<f32> = Vec::new();

    let config = SceneOptimizerConfig {
        steps: vec![
            OptimizationStep::NormalizeOpacity,
            OptimizationStep::PruneByOpacity { threshold: 0.5 },
            OptimizationStep::NormalizeOpacity,
        ],
        sh_channels: 0,
        seed: 0,
    };
    let pipeline = OptimizationPipeline::new(config);
    let (optimized, report) = pipeline
        .run(&positions, &rotations, &scales, &opacities, &sh, n)
        .expect("pipeline should succeed");

    assert_eq!(
        optimized.n_gaussians, 1,
        "only logit=3.0 (sigmoid≈0.953>0.5) should survive; \
         double-sigmoiding would incorrectly keep all 3"
    );
    assert!((optimized.opacities[0] - so_sigmoid(3.0)).abs() < 1e-4);
    assert!((report.snapshot_after.mean_opacity - so_sigmoid(3.0)).abs() < 1e-4);
}

#[test]
fn test_pipeline_clip_sphere_removes_distant() {
    // Put 3 Gaussians far from origin, 2 at origin
    let positions = vec![
        0.0f32, 0.0, 0.0, // inside
        0.0, 0.1, 0.0, // inside
        5.0, 0.0, 0.0, // outside
        0.0, 5.0, 0.0, // outside
        0.0, 0.0, 5.0, // outside
    ];
    let n = 5;
    let rotations: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0].into_iter().cycle().take(n * 4).collect();
    let scales = vec![0.05f32; n * 3];
    let opacities = vec![0.0f32; n];
    let sh: Vec<f32> = vec![];
    let config = SceneOptimizerConfig {
        steps: vec![OptimizationStep::ClipToSphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        }],
        sh_channels: 0,
        seed: 0,
    };
    let pipeline = OptimizationPipeline::new(config);
    let (scene, _) = pipeline.run(&positions, &rotations, &scales, &opacities, &sh, n)
        .expect("ClipToSphere pipeline should succeed");
    assert_eq!(scene.n_gaussians, 2, "only 2 Gaussians should survive sphere clip");
}

#[test]
fn test_pipeline_clip_aabb_removes_outside() {
    let positions = vec![
        0.5f32, 0.5, 0.5,  // inside
        2.0, 0.5, 0.5,     // outside
    ];
    let n = 2;
    let rotations: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let scales = vec![0.05f32; 6];
    let opacities = vec![0.0f32; 2];
    let sh: Vec<f32> = vec![];
    let config = SceneOptimizerConfig {
        steps: vec![OptimizationStep::ClipToAabb {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        }],
        sh_channels: 0,
        seed: 0,
    };
    let pipeline = OptimizationPipeline::new(config);
    let (scene, _) = pipeline.run(&positions, &rotations, &scales, &opacities, &sh, n)
        .expect("ClipToAabb pipeline should succeed");
    assert_eq!(scene.n_gaussians, 1);
}
