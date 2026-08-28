use super::*;

fn make_snap(name: &str, step: usize, params: Vec<f32>) -> ParamSnapshot {
    ParamSnapshot {
        name: name.to_string(),
        step,
        params,
    }
}

// ── lerp_params ──────────────────────────────────────────────────────────

#[test]
fn test_lerp_t0_returns_a() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let r = lerp_params(&a, &b, 0.0).unwrap();
    assert!((r[0] - 1.0).abs() < 1e-6);
    assert!((r[1] - 2.0).abs() < 1e-6);
    assert!((r[2] - 3.0).abs() < 1e-6);
}

#[test]
fn test_lerp_t1_returns_b() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let r = lerp_params(&a, &b, 1.0).unwrap();
    assert!((r[0] - 4.0).abs() < 1e-6);
    assert!((r[1] - 5.0).abs() < 1e-6);
    assert!((r[2] - 6.0).abs() < 1e-6);
}

#[test]
fn test_lerp_t05_midpoint() {
    let a = vec![0.0, 0.0];
    let b = vec![2.0, 4.0];
    let r = lerp_params(&a, &b, 0.5).unwrap();
    assert!((r[0] - 1.0).abs() < 1e-6);
    assert!((r[1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_lerp_invalid_t_negative() {
    let a = vec![1.0];
    let b = vec![2.0];
    let err = lerp_params(&a, &b, -0.1).unwrap_err();
    assert!(matches!(err, InterpolationError::InvalidT { .. }));
}

#[test]
fn test_lerp_invalid_t_gt1() {
    let a = vec![1.0];
    let b = vec![2.0];
    let err = lerp_params(&a, &b, 1.1).unwrap_err();
    assert!(matches!(err, InterpolationError::InvalidT { .. }));
}

#[test]
fn test_lerp_length_mismatch() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0, 2.0, 3.0];
    let err = lerp_params(&a, &b, 0.5).unwrap_err();
    assert!(matches!(err, InterpolationError::LengthMismatch { .. }));
}

// ── slerp_quat ───────────────────────────────────────────────────────────

#[test]
fn test_slerp_quat_t0_returns_a() {
    let qa = [0.0, 0.0, 0.0, 1.0f32];
    let qb = [0.0, 1.0, 0.0, 0.0f32];
    let r = slerp_quat(&qa, &qb, 0.0);
    assert!((r[3] - 1.0).abs() < 1e-5, "t=0 should return qa");
}

#[test]
fn test_slerp_quat_t1_returns_b() {
    let qa = [0.0, 0.0, 0.0, 1.0f32];
    let qb = [0.0, 1.0, 0.0, 0.0f32];
    let r = slerp_quat(&qa, &qb, 1.0);
    assert!((r[1] - 1.0).abs() < 1e-5, "t=1 should return qb");
}

#[test]
fn test_slerp_quat_identity_same() {
    let q = [0.0, 0.0, 0.0, 1.0f32];
    let r = slerp_quat(&q, &q, 0.5);
    let norm_sq: f32 = r.iter().map(|v| v * v).sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-5,
        "Result must be unit quaternion"
    );
    assert!((r[3] - 1.0).abs() < 1e-5);
}

#[test]
fn test_slerp_quat_180_rotation() {
    // 90-degree rotation around Z: (0, 0, sin(45°), cos(45°))
    let qa = [0.0, 0.0, 0.0, 1.0f32];
    let qb = [
        0.0,
        0.0,
        std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    ];
    let r = slerp_quat(&qa, &qb, 0.5);
    let norm_sq: f32 = r.iter().map(|v| v * v).sum();
    assert!((norm_sq - 1.0).abs() < 1e-5);
}

#[test]
fn test_slerp_quat_result_normalized() {
    let qa = [0.5, 0.5, 0.5, 0.5f32];
    let qb = [0.0, 0.0, 0.0, 1.0f32];
    let r = slerp_quat(&qa, &qb, 0.3);
    let norm_sq: f32 = r.iter().map(|v| v * v).sum();
    assert!((norm_sq - 1.0).abs() < 1e-5);
}

// ── normalize_quaternion ─────────────────────────────────────────────────

#[test]
fn test_normalize_quaternion_unit_norm() {
    let mut q = [2.0f32, 0.0, 0.0, 0.0];
    normalize_quaternion(&mut q);
    let norm_sq: f32 = q.iter().map(|v| v * v).sum();
    assert!((norm_sq - 1.0).abs() < 1e-6);
}

#[test]
fn test_normalize_quaternion_already_unit() {
    let mut q = [0.0f32, 0.0, 0.0, 1.0];
    normalize_quaternion(&mut q);
    assert!((q[3] - 1.0).abs() < 1e-6);
}

// ── params_l2_distance ───────────────────────────────────────────────────

#[test]
fn test_params_l2_distance_known() {
    let a = vec![0.0, 0.0];
    let b = vec![3.0, 4.0];
    let d = params_l2_distance(&a, &b).unwrap();
    assert!((d - 5.0).abs() < 1e-5);
}

#[test]
fn test_params_l2_distance_same_is_zero() {
    let a = vec![1.0, 2.0, 3.0];
    let d = params_l2_distance(&a, &a).unwrap();
    assert!(d.abs() < 1e-6);
}

#[test]
fn test_params_l2_distance_mismatch() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0];
    let err = params_l2_distance(&a, &b).unwrap_err();
    assert!(matches!(err, InterpolationError::LengthMismatch { .. }));
}

// ── params_cosine_similarity ─────────────────────────────────────────────

#[test]
fn test_cosine_similarity_same_is_one() {
    let a = vec![1.0, 2.0, 3.0];
    let s = params_cosine_similarity(&a, &a).unwrap();
    assert!((s - 1.0).abs() < 1e-5);
}

#[test]
fn test_cosine_similarity_opposite_is_minus_one() {
    let a = vec![1.0, 0.0];
    let b = vec![-1.0, 0.0];
    let s = params_cosine_similarity(&a, &b).unwrap();
    assert!((s - (-1.0)).abs() < 1e-5);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let s = params_cosine_similarity(&a, &b).unwrap();
    assert!(s.abs() < 1e-5);
}

#[test]
fn test_cosine_similarity_zero_vector() {
    let a = vec![0.0, 0.0];
    let b = vec![1.0, 0.0];
    let s = params_cosine_similarity(&a, &b).unwrap();
    assert_eq!(s, 0.0);
}

// ── params_l2_norm ───────────────────────────────────────────────────────

#[test]
fn test_params_l2_norm_known() {
    let v = vec![3.0, 4.0];
    assert!((params_l2_norm(&v) - 5.0).abs() < 1e-5);
}

#[test]
fn test_params_l2_norm_unit() {
    let v = vec![1.0, 0.0, 0.0];
    assert!((params_l2_norm(&v) - 1.0).abs() < 1e-5);
}

#[test]
fn test_params_l2_norm_zero() {
    let v = vec![0.0, 0.0, 0.0];
    assert!(params_l2_norm(&v).abs() < 1e-6);
}

// ── params_mean ──────────────────────────────────────────────────────────

#[test]
fn test_params_mean_known() {
    let v = vec![1.0, 2.0, 3.0, 4.0];
    assert!((params_mean(&v) - 2.5).abs() < 1e-5);
}

#[test]
fn test_params_mean_empty() {
    assert_eq!(params_mean(&[]), 0.0);
}

// ── params_std ───────────────────────────────────────────────────────────

#[test]
fn test_params_std_known() {
    let v = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let s = params_std(&v);
    // Population std ≈ 2.0
    assert!((s - 2.0).abs() < 1e-4);
}

#[test]
fn test_params_std_single_element() {
    assert_eq!(params_std(&[42.0]), 0.0);
}

#[test]
fn test_params_std_constant() {
    let v = vec![5.0; 10];
    assert!(params_std(&v).abs() < 1e-6);
}

// ── linear_interpolate ───────────────────────────────────────────────────

#[test]
fn test_linear_interpolate_t05() {
    let a = vec![0.0, 10.0];
    let b = vec![10.0, 0.0];
    let r = linear_interpolate(&a, &b, 0.5).unwrap();
    assert!((r[0] - 5.0).abs() < 1e-5);
    assert!((r[1] - 5.0).abs() < 1e-5);
}

// ── weighted_average_params ──────────────────────────────────────────────

#[test]
fn test_weighted_average_single_weight_one() {
    let snap = make_snap("a", 0, vec![1.0, 2.0, 3.0]);
    let result = weighted_average_params(&[snap], &[1.0]).unwrap();
    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[1] - 2.0).abs() < 1e-5);
    assert!((result[2] - 3.0).abs() < 1e-5);
}

#[test]
fn test_weighted_average_two_checkpoints_equal() {
    let a = make_snap("a", 0, vec![0.0, 0.0]);
    let b = make_snap("b", 1, vec![2.0, 4.0]);
    let r = weighted_average_params(&[a, b], &[0.5, 0.5]).unwrap();
    assert!((r[0] - 1.0).abs() < 1e-5);
    assert!((r[1] - 2.0).abs() < 1e-5);
}

#[test]
fn test_weighted_average_empty_error() {
    let err = weighted_average_params(&[], &[]).unwrap_err();
    assert!(matches!(err, InterpolationError::EmptyCheckpointList));
}

#[test]
fn test_weighted_average_weight_mismatch_error() {
    let a = make_snap("a", 0, vec![1.0]);
    let err = weighted_average_params(&[a], &[0.5, 0.5]).unwrap_err();
    assert!(matches!(
        err,
        InterpolationError::WeightCountMismatch { .. }
    ));
}

#[test]
fn test_weighted_average_bad_sum_error() {
    let a = make_snap("a", 0, vec![1.0]);
    let err = weighted_average_params(&[a], &[0.5]).unwrap_err();
    assert!(matches!(
        err,
        InterpolationError::InvalidBlendWeights { .. }
    ));
}

#[test]
fn test_weighted_average_negative_weight_error() {
    let a = make_snap("a", 0, vec![1.0]);
    let b = make_snap("b", 1, vec![2.0]);
    // sum = 1.0 but one weight is negative — the error must name the
    // offending index/value (index 0, value -0.2), not just repeat the
    // (still ~1.0) sum. `value` is checked with a tolerance rather than
    // `assert_eq!`-style exact float equality (the current
    // implementation passes the input weight through unchanged, but a
    // tolerance check is the more robust pattern regardless).
    let err = weighted_average_params(&[a, b], &[-0.2, 1.2]).unwrap_err();
    match err {
        InterpolationError::NegativeBlendWeight { index, value } => {
            assert_eq!(index, 0);
            assert!((value - (-0.2)).abs() < 1e-6, "value={value}");
        }
        other => panic!("expected NegativeBlendWeight, got {other:?}"),
    }
}

// Regression: weights summing to 0.99/1.01 (within tolerance) must be
// renormalized before being applied, not used as-is — otherwise every
// merged parameter would carry a systematic multiplicative bias.
#[test]
fn test_weighted_average_renormalizes_near_unity_sum() {
    let a = make_snap("a", 0, vec![10.0, 20.0]);
    let b = make_snap("b", 1, vec![30.0, 40.0]);
    // True proportions 0.5/0.5, but the sum is 0.99, not 1.0.
    let r = weighted_average_params(&[a, b], &[0.495, 0.495]).unwrap();
    // Renormalized: effectively [0.5, 0.5] -> (10+30)/2=20, (20+40)/2=30.
    assert!((r[0] - 20.0).abs() < 1e-4, "r[0]={}", r[0]);
    assert!((r[1] - 30.0).abs() < 1e-4, "r[1]={}", r[1]);
}

// ── uniform_average_params ───────────────────────────────────────────────

#[test]
fn test_uniform_average_smoke() {
    let a = make_snap("a", 0, vec![0.0, 0.0]);
    let b = make_snap("b", 1, vec![2.0, 4.0]);
    let r = uniform_average_params(&[a, b]).unwrap();
    assert!((r[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_uniform_average_three() {
    let a = make_snap("a", 0, vec![0.0]);
    let b = make_snap("b", 1, vec![3.0]);
    let c = make_snap("c", 2, vec![6.0]);
    let r = uniform_average_params(&[a, b, c]).unwrap();
    assert!((r[0] - 3.0).abs() < 1e-5);
}

#[test]
fn test_uniform_average_empty_error() {
    let err = uniform_average_params(&[]).unwrap_err();
    assert!(matches!(err, InterpolationError::EmptyCheckpointList));
}

// ── interpolation_sequence ───────────────────────────────────────────────

#[test]
fn test_interpolation_sequence_n2_endpoints() {
    let a = vec![0.0];
    let b = vec![1.0];
    let seq = interpolation_sequence(&a, &b, 2).unwrap();
    assert_eq!(seq.len(), 2);
    assert!((seq[0][0] - 0.0).abs() < 1e-5);
    assert!((seq[1][0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_interpolation_sequence_n5_correct_length() {
    let a = vec![0.0, 0.0];
    let b = vec![1.0, 1.0];
    let seq = interpolation_sequence(&a, &b, 5).unwrap();
    assert_eq!(seq.len(), 5);
    // Check t=0.5 is at index 2
    assert!((seq[2][0] - 0.5).abs() < 1e-5);
}

#[test]
fn test_interpolation_sequence_invalid_steps() {
    let a = vec![0.0];
    let b = vec![1.0];
    let err = interpolation_sequence(&a, &b, 1).unwrap_err();
    assert!(matches!(err, InterpolationError::InvalidStepCount { .. }));
}

#[test]
fn test_interpolation_sequence_length_mismatch() {
    let a = vec![0.0, 1.0];
    let b = vec![1.0];
    let err = interpolation_sequence(&a, &b, 3).unwrap_err();
    assert!(matches!(err, InterpolationError::LengthMismatch { .. }));
}

// ── build_checkpoint_path ────────────────────────────────────────────────

#[test]
fn test_build_checkpoint_path_two_snapshots() {
    let a = make_snap("a", 0, vec![0.0, 0.0]);
    let b = make_snap("b", 1, vec![3.0, 4.0]);
    let path = build_checkpoint_path(vec![a, b]).unwrap();
    assert_eq!(path.snapshots.len(), 2);
    assert_eq!(path.step_sizes.len(), 1);
    assert!((path.step_sizes[0] - 5.0).abs() < 1e-4);
    assert!((path.total_length - 5.0).abs() < 1e-4);
}

#[test]
fn test_build_checkpoint_path_empty_error() {
    let err = build_checkpoint_path(vec![]).unwrap_err();
    assert!(matches!(err, InterpolationError::EmptyCheckpointList));
}

#[test]
fn test_build_checkpoint_path_single_snapshot() {
    // Single snapshot: no step sizes, total_length = 0
    let a = make_snap("a", 0, vec![1.0, 2.0]);
    let path = build_checkpoint_path(vec![a]).unwrap();
    assert_eq!(path.snapshots.len(), 1);
    assert_eq!(path.step_sizes.len(), 0);
    assert_eq!(path.total_length, 0.0);
}

#[test]
fn test_build_checkpoint_path_three_snapshots() {
    let a = make_snap("a", 0, vec![0.0, 0.0]);
    let b = make_snap("b", 1, vec![3.0, 4.0]);
    let c = make_snap("c", 2, vec![3.0, 4.0]);
    let path = build_checkpoint_path(vec![a, b, c]).unwrap();
    assert_eq!(path.step_sizes.len(), 2);
    assert!((path.step_sizes[0] - 5.0).abs() < 1e-4);
    assert!(path.step_sizes[1].abs() < 1e-4);
}

// ── interpolate_along_path ───────────────────────────────────────────────

#[test]
fn test_interpolate_along_path_t0_first() {
    let a = make_snap("a", 0, vec![0.0, 0.0]);
    let b = make_snap("b", 1, vec![1.0, 1.0]);
    let path = build_checkpoint_path(vec![a, b]).unwrap();
    let r = interpolate_along_path(&path, 0.0).unwrap();
    assert!((r[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_interpolate_along_path_t1_last() {
    let a = make_snap("a", 0, vec![0.0, 0.0]);
    let b = make_snap("b", 1, vec![1.0, 1.0]);
    let path = build_checkpoint_path(vec![a, b]).unwrap();
    let r = interpolate_along_path(&path, 1.0).unwrap();
    assert!((r[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_interpolate_along_path_t05_midpoint() {
    let a = make_snap("a", 0, vec![0.0, 0.0]);
    let b = make_snap("b", 1, vec![2.0, 4.0]);
    let path = build_checkpoint_path(vec![a, b]).unwrap();
    let r = interpolate_along_path(&path, 0.5).unwrap();
    assert!((r[0] - 1.0).abs() < 1e-4);
    assert!((r[1] - 2.0).abs() < 1e-4);
}

#[test]
fn test_interpolate_along_path_invalid_t() {
    let a = make_snap("a", 0, vec![0.0]);
    let path = build_checkpoint_path(vec![a]).unwrap();
    let err = interpolate_along_path(&path, 1.5).unwrap_err();
    assert!(matches!(err, InterpolationError::InvalidT { .. }));
}

// ── model_soup ───────────────────────────────────────────────────────────

#[test]
fn test_model_soup_identical_checkpoints() {
    let a = make_snap("a", 0, vec![1.0, 2.0, 3.0]);
    let b = make_snap("b", 1, vec![1.0, 2.0, 3.0]);
    let c = make_snap("c", 2, vec![1.0, 2.0, 3.0]);
    let r = model_soup(&[a, b, c]).unwrap();
    assert!((r[0] - 1.0).abs() < 1e-5);
    assert!((r[1] - 2.0).abs() < 1e-5);
    assert!((r[2] - 3.0).abs() < 1e-5);
}

#[test]
fn test_model_soup_empty_error() {
    let err = model_soup(&[]).unwrap_err();
    assert!(matches!(err, InterpolationError::EmptyCheckpointList));
}

#[test]
fn test_model_soup_two_diverse() {
    let a = make_snap("a", 0, vec![0.0, 0.0]);
    let b = make_snap("b", 1, vec![4.0, 8.0]);
    let r = model_soup(&[a, b]).unwrap();
    assert!((r[0] - 2.0).abs() < 1e-5);
    assert!((r[1] - 4.0).abs() < 1e-5);
}

// ── linear_mode_connectivity ─────────────────────────────────────────────

#[test]
fn test_linear_mode_connectivity_correct_length() {
    let a = vec![0.0; 8];
    let b = vec![1.0; 8];
    let pairs = linear_mode_connectivity(&a, &b, 5, &|p: &[f32]| p.iter().sum::<f32>());
    assert_eq!(pairs.len(), 5);
}

#[test]
fn test_linear_mode_connectivity_t_in_range() {
    let a = vec![0.0; 4];
    let b = vec![1.0; 4];
    let pairs = linear_mode_connectivity(&a, &b, 10, &|_: &[f32]| 0.0);
    for (t, _loss) in &pairs {
        assert!(*t >= 0.0 && *t <= 1.0);
    }
}

#[test]
fn test_linear_mode_connectivity_mismatch_returns_empty() {
    let a = vec![0.0, 1.0];
    let b = vec![1.0];
    let pairs = linear_mode_connectivity(&a, &b, 5, &|_: &[f32]| 0.0);
    assert!(pairs.is_empty());
}

#[test]
fn test_linear_mode_connectivity_loss_monotone() {
    // Loss = sum of params; at t=0 sum=0, at t=1 sum=8
    let a = vec![0.0; 8];
    let b = vec![1.0; 8];
    let pairs = linear_mode_connectivity(&a, &b, 5, &|p: &[f32]| p.iter().sum::<f32>());
    // Losses should be monotone increasing
    for w in pairs.windows(2) {
        assert!(w[1].1 >= w[0].1 - 1e-5);
    }
}

// ── compute_interpolation_stats ──────────────────────────────────────────

#[test]
fn test_interpolation_stats_correct_min_max() {
    let params = vec![1.0, 3.0, -2.0, 5.0];
    let stats = compute_interpolation_stats(&params);
    assert!((stats.min - (-2.0)).abs() < 1e-5);
    assert!((stats.max - 5.0).abs() < 1e-5);
    assert_eq!(stats.param_dim, 4);
}

#[test]
fn test_interpolation_stats_correct_mean() {
    let params = vec![1.0, 2.0, 3.0, 4.0];
    let stats = compute_interpolation_stats(&params);
    assert!((stats.mean - 2.5).abs() < 1e-5);
}

#[test]
fn test_interpolation_stats_empty() {
    let stats = compute_interpolation_stats(&[]);
    assert_eq!(stats.param_dim, 0);
    assert_eq!(stats.mean, 0.0);
}

#[test]
fn test_interpolation_stats_l2_norm() {
    let params = vec![3.0, 4.0];
    let stats = compute_interpolation_stats(&params);
    assert!((stats.l2_norm - 5.0).abs() < 1e-5);
}

// ── find_optimal_blend ───────────────────────────────────────────────────

#[test]
fn test_find_optimal_blend_identity_loss() {
    // Loss is constant — any weights work; just verify no error and that
    // the returned vector has one weight per checkpoint (length 2), not
    // one entry per parameter (which would also happen to be 2 here —
    // see test_find_optimal_blend_returns_weights_not_blended_params for
    // an unambiguous check).
    let a = make_snap("a", 0, vec![1.0, 0.0]);
    let b = make_snap("b", 1, vec![0.0, 1.0]);
    let result = find_optimal_blend(&[a, b], &[0.5, 0.5], &|_: &[f32]| 0.0, 10);
    assert!(result.is_ok());
    let weights = result.unwrap();
    assert_eq!(weights.len(), 2);
}

#[test]
fn test_find_optimal_blend_empty_error() {
    let err = find_optimal_blend(&[], &[], &|_: &[f32]| 0.0, 10).unwrap_err();
    assert!(matches!(err, InterpolationError::EmptyCheckpointList));
}

#[test]
fn test_find_optimal_blend_single_checkpoint() {
    let a = make_snap("a", 0, vec![1.0, 2.0, 3.0]);
    let result = find_optimal_blend(
        &[a],
        &[],
        &|p: &[f32]| {
            // Minimize L2 norm
            p.iter().map(|x| x * x).sum::<f32>()
        },
        5,
    );
    assert!(result.is_ok());
    // One checkpoint -> one weight, trivially [1.0].
    assert_eq!(result.unwrap().len(), 1);
}

#[test]
fn test_find_optimal_blend_converges_toward_minimum() {
    // Loss = distance from target [0,0,0,0]
    // Checkpoint a=[2,2,2,2], b=[0,0,0,0]
    // Optimal blend should prefer checkpoint b (weight close to 1.0 for b).
    let a = make_snap("a", 0, vec![2.0, 2.0, 2.0, 2.0]);
    let b = make_snap("b", 1, vec![0.0, 0.0, 0.0, 0.0]);
    let checkpoints = [a, b];
    let target = vec![0.0, 0.0, 0.0, 0.0];
    let weights = find_optimal_blend(
        &checkpoints,
        &target,
        &|p: &[f32]| p.iter().map(|x| x * x).sum::<f32>(),
        50,
    )
    .unwrap();
    assert_eq!(weights.len(), 2, "one weight per checkpoint");
    // The blended result, reconstructed from the *returned weights*,
    // should be close to zero (weight on `b` dominant).
    let blended = weighted_average_params(&checkpoints, &weights).unwrap();
    let norm: f32 = blended.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(norm < 2.0, "Blend should reduce loss, got norm={}", norm);
}

// Regression: find_optimal_blend must return the weight vector (length
// == checkpoints.len()), not the blended parameter vector (length ==
// param_len). Use checkpoint/param lengths that differ so the two
// possible return shapes are distinguishable.
#[test]
fn test_find_optimal_blend_returns_weights_not_blended_params() {
    let a = make_snap("a", 0, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = make_snap("b", 1, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    let c = make_snap("c", 2, vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    // 3 checkpoints, 5 params each — a bug returning blended params
    // would yield length 5, not 3.
    let weights = find_optimal_blend(&[a, b, c], &[], &|_: &[f32]| 0.0, 3).unwrap();
    assert_eq!(
        weights.len(),
        3,
        "expected one weight per checkpoint (3), got {}",
        weights.len()
    );
    // Weights must remain on the simplex: non-negative and summing to 1.
    assert!(weights.iter().all(|&w| w >= 0.0));
    let sum: f32 = weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "weights sum to {sum}, expected 1.0"
    );
}

// ── quantize_params ──────────────────────────────────────────────────────

#[test]
fn test_quantize_8bit_stays_within_range() {
    let params = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let q = quantize_params(&params, 8);
    assert_eq!(q.len(), params.len());
    for &v in &q {
        assert!((0.0..=1.0).contains(&v));
    }
}

#[test]
fn test_quantize_empty() {
    let q = quantize_params(&[], 8);
    assert!(q.is_empty());
}

#[test]
fn test_quantize_constant_unchanged() {
    let params = vec![5.0, 5.0, 5.0];
    let q = quantize_params(&params, 8);
    assert!((q[0] - 5.0).abs() < 1e-5);
}

#[test]
fn test_quantize_endpoints_preserved() {
    let params = vec![0.0, 1.0];
    let q = quantize_params(&params, 8);
    assert!((q[0] - 0.0).abs() < 1e-4);
    assert!((q[1] - 1.0).abs() < 1e-4);
}

// Regression: bits=0 must not produce NaN (old code computed
// `levels = (1 << 0) - 1 = 0`, then divided by zero).
#[test]
fn test_quantize_bits_zero_does_not_produce_nan() {
    let params = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let q = quantize_params(&params, 0);
    assert_eq!(q.len(), params.len());
    for &v in &q {
        assert!(!v.is_nan(), "bits=0 produced NaN");
        assert!((0.0..=1.0).contains(&v), "bits=0 produced out-of-range {v}");
    }
}

// Regression: bits>=64 must not panic on `1u64 << bits` shift overflow.
// `bits` is `u8` (max value 255), which already exceeds the 64-bit
// shift-overflow threshold this guards against.
#[test]
fn test_quantize_bits_overflow_does_not_panic() {
    let params = vec![0.0, 0.5, 1.0];
    for bits in [64u8, 100, 255] {
        let q = quantize_params(&params, bits);
        assert_eq!(q.len(), params.len());
        for &v in &q {
            assert!(!v.is_nan());
        }
    }
}

// ── dequantize_error ─────────────────────────────────────────────────────

#[test]
fn test_dequantize_error_same_is_zero() {
    let params = vec![1.0, 2.0, 3.0];
    let err = dequantize_error(&params, &params).unwrap();
    assert!(err.abs() < 1e-6);
}

#[test]
fn test_dequantize_error_positive() {
    let original = vec![0.0, 0.5, 1.0];
    let quantized = vec![0.0, 0.502, 1.0];
    let err = dequantize_error(&original, &quantized).unwrap();
    assert!(err > 0.0);
}

#[test]
fn test_dequantize_error_mismatch() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0];
    let err = dequantize_error(&a, &b).unwrap_err();
    assert!(matches!(err, InterpolationError::LengthMismatch { .. }));
}

#[test]
fn test_dequantize_error_empty_is_zero() {
    let err = dequantize_error(&[], &[]).unwrap();
    assert_eq!(err, 0.0);
}

// ── format_checkpoint_path ───────────────────────────────────────────────

#[test]
fn test_format_checkpoint_path_non_empty() {
    let a = make_snap("a", 0, vec![0.0]);
    let b = make_snap("b", 1, vec![1.0]);
    let path = build_checkpoint_path(vec![a, b]).unwrap();
    let s = format_checkpoint_path(&path);
    assert!(!s.is_empty());
    assert!(s.contains("2 snapshots"));
    assert!(s.contains("total_length"));
}

#[test]
fn test_format_checkpoint_path_single() {
    let a = make_snap("only", 0, vec![1.0, 2.0]);
    let path = build_checkpoint_path(vec![a]).unwrap();
    let s = format_checkpoint_path(&path);
    assert!(s.contains("1 snapshots"));
    assert!(s.contains("0.0000"));
}

// ── Error variant tests ──────────────────────────────────────────────────

#[test]
fn test_error_display_length_mismatch() {
    let err = InterpolationError::LengthMismatch { len_a: 3, len_b: 5 };
    let msg = err.to_string();
    assert!(msg.contains("3") && msg.contains("5"));
}

#[test]
fn test_error_display_invalid_t() {
    let err = InterpolationError::InvalidT { t: 1.5 };
    let msg = err.to_string();
    assert!(msg.contains("1.5"));
}

#[test]
fn test_error_display_invalid_step_count() {
    let err = InterpolationError::InvalidStepCount { n: 1 };
    let msg = err.to_string();
    assert!(msg.contains("1"));
}

#[test]
fn test_error_display_empty_checkpoint_list() {
    let err = InterpolationError::EmptyCheckpointList;
    assert!(!err.to_string().is_empty());
}

#[test]
fn test_error_display_weight_count_mismatch() {
    let err = InterpolationError::WeightCountMismatch {
        weights_len: 2,
        n_checkpoints: 3,
    };
    let msg = err.to_string();
    assert!(msg.contains("2") && msg.contains("3"));
}

// ── Struct field tests ───────────────────────────────────────────────────

#[test]
fn test_param_snapshot_fields() {
    let snap = ParamSnapshot {
        name: "test_snap".to_string(),
        step: 500,
        params: vec![1.0, 2.0],
    };
    assert_eq!(snap.name, "test_snap");
    assert_eq!(snap.step, 500);
    assert_eq!(snap.params.len(), 2);
}

#[test]
fn test_interpolation_config_default() {
    let cfg = InterpolationConfig::default();
    assert!(!cfg.use_slerp_for_rotations);
    assert!(cfg.normalize_rotations);
    assert_eq!(cfg.n_quaternion_params, 0);
}

// ── interpolate_with_config ──────────────────────────────────────────────
//
// Regression coverage for: InterpolationConfig was previously entirely
// dead (never consulted by any public function), and slerp_quat /
// normalize_quaternion were #[cfg(test)]-gated (unreachable in a release
// build). These tests exercise the config-aware entry point that wires
// them together.

#[test]
fn test_interpolate_with_config_default_matches_plain_lerp() {
    // Default config (n_quaternion_params: 0) must behave exactly like
    // linear_interpolate — no quaternion region is defined.
    let a = vec![0.0, 0.0, 0.0, 1.0, 10.0];
    let b = vec![0.0, 1.0, 0.0, 0.0, 20.0];
    let cfg = InterpolationConfig::default();
    let via_config = interpolate_with_config(&a, &b, 0.5, &cfg).unwrap();
    let via_plain = linear_interpolate(&a, &b, 0.5).unwrap();
    assert_eq!(via_config, via_plain);
}

#[test]
fn test_interpolate_with_config_normalizes_lerped_quaternion_block() {
    // Two orthogonal unit quaternions lerped at t=0.5 without
    // normalization would have norm < 1 (component-wise lerp of unit
    // vectors is not itself unit length); with normalize_rotations set
    // (and use_slerp_for_rotations left false), the result must be
    // renormalized back to unit length.
    let a = vec![0.0, 0.0, 0.0, 1.0]; // identity quaternion
    let b = vec![0.0, 1.0, 0.0, 0.0]; // 180° rotation about Y
    let cfg = InterpolationConfig {
        use_slerp_for_rotations: false,
        normalize_rotations: true,
        n_quaternion_params: 4,
    };
    let r = interpolate_with_config(&a, &b, 0.5, &cfg).unwrap();
    let norm_sq: f32 = r.iter().map(|v| v * v).sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-5,
        "expected unit quaternion, got norm_sq={norm_sq}"
    );
}

#[test]
fn test_interpolate_with_config_slerp_matches_slerp_quat() {
    let a = vec![0.0, 0.0, 0.0, 1.0];
    let b = vec![0.0, 1.0, 0.0, 0.0];
    let cfg = InterpolationConfig {
        use_slerp_for_rotations: true,
        normalize_rotations: true,
        n_quaternion_params: 4,
    };
    let r = interpolate_with_config(&a, &b, 0.3, &cfg).unwrap();
    let qa = [a[0], a[1], a[2], a[3]];
    let qb = [b[0], b[1], b[2], b[3]];
    let expected = slerp_quat(&qa, &qb, 0.3);
    for c in 0..4 {
        assert!((r[c] - expected[c]).abs() < 1e-5, "channel {c}");
    }
}

#[test]
fn test_interpolate_with_config_only_first_block_is_quaternion_aware() {
    // Elements after the quaternion region must always be plain-lerped
    // regardless of config.
    let a = vec![0.0, 0.0, 0.0, 1.0, /* trailing scalar */ 2.0];
    let b = vec![0.0, 1.0, 0.0, 0.0, /* trailing scalar */ 8.0];
    let cfg = InterpolationConfig {
        use_slerp_for_rotations: true,
        normalize_rotations: true,
        n_quaternion_params: 4,
    };
    let r = interpolate_with_config(&a, &b, 0.5, &cfg).unwrap();
    assert!(
        (r[4] - 5.0).abs() < 1e-5,
        "trailing scalar should be plain-lerped: {}",
        r[4]
    );
}

#[test]
fn test_interpolate_with_config_invalid_region_not_multiple_of_4() {
    let a = vec![0.0; 6];
    let b = vec![0.0; 6];
    let cfg = InterpolationConfig {
        n_quaternion_params: 3, // not a multiple of 4
        ..InterpolationConfig::default()
    };
    let err = interpolate_with_config(&a, &b, 0.5, &cfg).unwrap_err();
    assert!(matches!(
        err,
        InterpolationError::InvalidQuaternionRegion { .. }
    ));
}

#[test]
fn test_interpolate_with_config_invalid_region_exceeds_len() {
    let a = vec![0.0; 4];
    let b = vec![0.0; 4];
    let cfg = InterpolationConfig {
        n_quaternion_params: 8, // exceeds param_len of 4
        ..InterpolationConfig::default()
    };
    let err = interpolate_with_config(&a, &b, 0.5, &cfg).unwrap_err();
    assert!(matches!(
        err,
        InterpolationError::InvalidQuaternionRegion { .. }
    ));
}

#[test]
fn test_interpolation_sequence_with_config_endpoints() {
    let a = vec![0.0, 0.0, 0.0, 1.0];
    let b = vec![0.0, 1.0, 0.0, 0.0];
    let cfg = InterpolationConfig {
        use_slerp_for_rotations: true,
        normalize_rotations: true,
        n_quaternion_params: 4,
    };
    let seq = interpolation_sequence_with_config(&a, &b, 3, &cfg).unwrap();
    assert_eq!(seq.len(), 3);
    // t=0 -> a, t=1 -> b (SLERP endpoints match input exactly).
    for c in 0..4 {
        assert!((seq[0][c] - a[c]).abs() < 1e-5);
        assert!((seq[2][c] - b[c]).abs() < 1e-5);
    }
    // Midpoint must remain unit-length.
    let norm_sq: f32 = seq[1].iter().map(|v| v * v).sum();
    assert!((norm_sq - 1.0).abs() < 1e-5);
}

#[test]
fn test_interpolate_along_path_with_config_midpoint_unit_quaternion() {
    let a = make_snap("a", 0, vec![0.0, 0.0, 0.0, 1.0]);
    let b = make_snap("b", 1, vec![0.0, 1.0, 0.0, 0.0]);
    let path = build_checkpoint_path(vec![a, b]).unwrap();
    let cfg = InterpolationConfig {
        use_slerp_for_rotations: true,
        normalize_rotations: true,
        n_quaternion_params: 4,
    };
    let r = interpolate_along_path_with_config(&path, 0.5, &cfg).unwrap();
    let norm_sq: f32 = r.iter().map(|v| v * v).sum();
    assert!((norm_sq - 1.0).abs() < 1e-5);
}

#[test]
fn test_checkpoint_path_fields() {
    let a = make_snap("a", 0, vec![0.0, 0.0]);
    let b = make_snap("b", 1, vec![1.0, 0.0]);
    let path = build_checkpoint_path(vec![a, b]).unwrap();
    assert_eq!(path.snapshots.len(), 2);
    assert_eq!(path.step_sizes.len(), 1);
    assert!(path.total_length > 0.0);
}

#[test]
fn test_interpolation_stats_fields() {
    let stats = compute_interpolation_stats(&[1.0, 2.0, 3.0]);
    assert_eq!(stats.param_dim, 3);
    assert!(stats.mean > 0.0);
    assert!(stats.std >= 0.0);
    assert!(stats.l2_norm > 0.0);
}

// ── Edge cases ───────────────────────────────────────────────────────────

#[test]
fn test_interpolate_along_path_degenerate_zero_length() {
    // All snapshots identical → total_length = 0 → returns first snapshot
    let a = make_snap("a", 0, vec![3.0, 7.0]);
    let b = make_snap("b", 1, vec![3.0, 7.0]);
    let path = build_checkpoint_path(vec![a, b]).unwrap();
    let r = interpolate_along_path(&path, 0.5).unwrap();
    assert!((r[0] - 3.0).abs() < 1e-5);
    assert!((r[1] - 7.0).abs() < 1e-5);
}

#[test]
fn test_interpolation_sequence_all_same() {
    let a = vec![1.0, 1.0];
    let seq = interpolation_sequence(&a, &a, 5).unwrap();
    for v in &seq {
        assert!((v[0] - 1.0).abs() < 1e-5);
    }
}

#[test]
fn test_slerp_quat_antiparallel_short_path() {
    // When qa and qb are antiparallel, dot < 0, qb should be negated
    let qa = [0.0, 0.0, 0.0, 1.0f32];
    let qb = [0.0, 0.0, 0.0, -1.0f32]; // antipodal
    let r = slerp_quat(&qa, &qb, 0.0);
    // t=0 should return qa (short path ensures we start at qa)
    let norm_sq: f32 = r.iter().map(|v| v * v).sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-5,
        "result must be unit quaternion"
    );
}
