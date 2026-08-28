// ─── Tests ───────────────────────────────────────────────────────────────────

use super::svd::*;
use super::*;
use kiddo::KdTree;

// ── Helper ───────────────────────────────────────────────────────────────

/// Build rotation matrix for angle θ around Z-axis.
fn rot_z(theta: f32) -> [f32; 9] {
    let c = theta.cos();
    let s = theta.sin();
    [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]
}

/// Apply rot + t + scale to a flat point array (for test setup).
fn apply_transform_flat(pts: &[f32], r: &[f32; 9], t: [f32; 3], s: f32) -> Vec<f32> {
    let xf = SimilarityTransform {
        rotation: *r,
        translation: t,
        scale: s,
    };
    xf.apply_flat(pts)
}

/// Simple flat point set: a tetrahedron.
fn tetra() -> Vec<f32> {
    vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}

fn assert_float_eq(a: f32, b: f32, tol: f32, label: &str) {
    assert!(
        (a - b).abs() <= tol,
        "{label}: expected {b}, got {a} (tol {tol})"
    );
}

// ── SimilarityTransform::identity ─────────────────────────────────────

#[test]
fn test_identity_apply_is_noop() {
    let id = SimilarityTransform::identity();
    let p = [1.0f32, 2.0, 3.0];
    let q = id.apply(p);
    assert_float_eq(q[0], 1.0, 1e-6, "x");
    assert_float_eq(q[1], 2.0, 1e-6, "y");
    assert_float_eq(q[2], 3.0, 1e-6, "z");
}

#[test]
fn test_identity_scale_one() {
    assert_float_eq(SimilarityTransform::identity().scale, 1.0, 1e-9, "scale");
}

// ── SimilarityTransform::apply ────────────────────────────────────────

#[test]
fn test_apply_known_rotation_translation() {
    // 90° around Z: (1,0,0) → (0,1,0), then translate +1 on x
    use std::f32::consts::FRAC_PI_2;
    let xf = SimilarityTransform {
        rotation: rot_z(FRAC_PI_2),
        translation: [1.0, 0.0, 0.0],
        scale: 1.0,
    };
    let q = xf.apply([1.0, 0.0, 0.0]);
    assert_float_eq(q[0], 1.0, 1e-5, "x");
    assert_float_eq(q[1], 1.0, 1e-5, "y");
    assert_float_eq(q[2], 0.0, 1e-5, "z");
}

#[test]
fn test_apply_scale() {
    let xf = SimilarityTransform {
        rotation: mat3_identity(),
        translation: [0.0; 3],
        scale: 3.0,
    };
    let q = xf.apply([2.0, 1.0, 0.5]);
    assert_float_eq(q[0], 6.0, 1e-6, "x");
    assert_float_eq(q[1], 3.0, 1e-6, "y");
    assert_float_eq(q[2], 1.5, 1e-6, "z");
}

// ── SimilarityTransform::apply_flat ───────────────────────────────────

#[test]
fn test_apply_flat_output_shape() {
    let id = SimilarityTransform::identity();
    let pts = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let out = id.apply_flat(&pts);
    assert_eq!(out.len(), 6);
}

#[test]
fn test_apply_flat_identity_values() {
    let id = SimilarityTransform::identity();
    let pts = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let out = id.apply_flat(&pts);
    for (a, b) in pts.iter().zip(out.iter()) {
        assert_float_eq(*b, *a, 1e-6, "flat id");
    }
}

#[test]
fn test_apply_flat_translation() {
    let xf = SimilarityTransform {
        rotation: mat3_identity(),
        translation: [10.0, 0.0, 0.0],
        scale: 1.0,
    };
    let pts = vec![1.0f32, 0.0, 0.0, 2.0, 0.0, 0.0];
    let out = xf.apply_flat(&pts);
    assert_float_eq(out[0], 11.0, 1e-6, "p0.x");
    assert_float_eq(out[3], 12.0, 1e-6, "p1.x");
}

// ── SimilarityTransform::inverse ─────────────────────────────────────

#[test]
fn test_inverse_roundtrip() {
    use std::f32::consts::FRAC_PI_4;
    let xf = SimilarityTransform {
        rotation: rot_z(FRAC_PI_4),
        translation: [1.0, -2.0, 3.0],
        scale: 2.5,
    };
    let inv = xf.inverse();
    let p = [3.0f32, -1.0, 0.5];
    let q = xf.apply(p);
    let r = inv.apply(q);
    assert_float_eq(r[0], p[0], 1e-4, "inv x");
    assert_float_eq(r[1], p[1], 1e-4, "inv y");
    assert_float_eq(r[2], p[2], 1e-4, "inv z");
}

#[test]
fn test_inverse_identity_is_identity() {
    let id = SimilarityTransform::identity();
    let inv = id.inverse();
    assert_float_eq(inv.scale, 1.0, 1e-7, "inv id scale");
    for i in 0..3 {
        assert_float_eq(inv.translation[i], 0.0, 1e-7, "inv id t");
    }
}

// ── SimilarityTransform::compose ─────────────────────────────────────

#[test]
fn test_compose_with_identity_left() {
    let id = SimilarityTransform::identity();
    use std::f32::consts::FRAC_PI_6;
    let xf = SimilarityTransform {
        rotation: rot_z(FRAC_PI_6),
        translation: [1.0, 2.0, 3.0],
        scale: 1.5,
    };
    let composed = id.compose(&xf);
    let p = [1.0f32, 0.0, 0.0];
    let a = composed.apply(p);
    let b = xf.apply(p);
    assert_float_eq(a[0], b[0], 1e-5, "compose left x");
    assert_float_eq(a[1], b[1], 1e-5, "compose left y");
}

#[test]
fn test_compose_with_identity_right() {
    let id = SimilarityTransform::identity();
    use std::f32::consts::FRAC_PI_6;
    let xf = SimilarityTransform {
        rotation: rot_z(FRAC_PI_6),
        translation: [1.0, 2.0, 3.0],
        scale: 1.5,
    };
    let composed = xf.compose(&id);
    let p = [1.0f32, 0.0, 0.0];
    let a = composed.apply(p);
    let b = xf.apply(p);
    assert_float_eq(a[0], b[0], 1e-5, "compose right x");
    assert_float_eq(a[1], b[1], 1e-5, "compose right y");
}

#[test]
fn test_compose_scales_multiply() {
    let xf1 = SimilarityTransform {
        rotation: mat3_identity(),
        translation: [0.0; 3],
        scale: 2.0,
    };
    let xf2 = SimilarityTransform {
        rotation: mat3_identity(),
        translation: [0.0; 3],
        scale: 3.0,
    };
    let composed = xf1.compose(&xf2);
    assert_float_eq(composed.scale, 6.0, 1e-6, "compose scale");
}

#[test]
fn test_compose_is_sequential() {
    let t1 = SimilarityTransform {
        rotation: mat3_identity(),
        translation: [1.0, 0.0, 0.0],
        scale: 1.0,
    };
    let t2 = SimilarityTransform {
        rotation: mat3_identity(),
        translation: [0.0, 2.0, 0.0],
        scale: 1.0,
    };
    let p = [0.0f32; 3];
    let composed = t1.compose(&t2);
    let q = composed.apply(p);
    // p → t1 → (1,0,0) → t2 → (1,2,0)
    assert_float_eq(q[0], 1.0, 1e-6, "seq x");
    assert_float_eq(q[1], 2.0, 1e-6, "seq y");
}

// ── SimilarityTransform::rotation_matrix ──────────────────────────────

#[test]
fn test_rotation_matrix_identity() {
    let id = SimilarityTransform::identity();
    let rm = id.rotation_matrix();
    for (r, row) in rm.iter().enumerate() {
        for (c, &val) in row.iter().enumerate() {
            let expected = if r == c { 1.0 } else { 0.0 };
            assert_float_eq(val, expected, 1e-9, "rot mat id");
        }
    }
}

// ── align_procrustes: identical sets ─────────────────────────────────

#[test]
fn test_procrustes_identical_rmse_zero() {
    let pts = tetra();
    let xf = align_procrustes(&pts, &pts).expect("procrustes failed");
    let rmse = align_rmse(&pts, &pts, &xf).expect("rmse failed");
    assert!(
        rmse < 1e-4,
        "expected RMSE~0 for identical sets, got {rmse}"
    );
}

#[test]
fn test_procrustes_identical_scale_one() {
    let pts = tetra();
    let xf = align_procrustes(&pts, &pts).expect("procrustes failed");
    assert_float_eq(xf.scale, 1.0, 0.01, "identity scale");
}

// ── align_procrustes: translation ─────────────────────────────────────

#[test]
fn test_procrustes_translation_recovery() {
    let src = tetra();
    let tgt = apply_transform_flat(&src, &mat3_identity(), [5.0, -3.0, 1.0], 1.0);
    let xf = align_procrustes(&src, &tgt).expect("proc failed");
    let rmse = align_rmse(&src, &tgt, &xf).expect("rmse");
    assert!(rmse < 1e-3, "translation rmse {rmse}");
    assert_float_eq(xf.translation[0], 5.0, 1e-2, "tx");
    assert_float_eq(xf.translation[1], -3.0, 1e-2, "ty");
    assert_float_eq(xf.translation[2], 1.0, 1e-2, "tz");
}

// ── align_procrustes: scale ───────────────────────────────────────────

#[test]
fn test_procrustes_scale_recovery() {
    let src = tetra();
    let tgt = apply_transform_flat(&src, &mat3_identity(), [0.0; 3], 2.5);
    let xf = align_procrustes(&src, &tgt).expect("proc failed");
    let rmse = align_rmse(&src, &tgt, &xf).expect("rmse");
    assert!(rmse < 1e-3, "scale rmse {rmse}");
    assert_float_eq(xf.scale, 2.5, 0.05, "recovered scale");
}

// ── align_procrustes: rotation ────────────────────────────────────────

#[test]
fn test_procrustes_rotation_recovery_rmse() {
    use std::f32::consts::FRAC_PI_3;
    let src = tetra();
    let r = rot_z(FRAC_PI_3);
    let tgt = apply_transform_flat(&src, &r, [0.0; 3], 1.0);
    let xf = align_procrustes(&src, &tgt).expect("proc failed");
    let rmse = align_rmse(&src, &tgt, &xf).expect("rmse");
    assert!(rmse < 1e-3, "rotation rmse {rmse}");
}

// ── align_procrustes: error cases ─────────────────────────────────────

#[test]
fn test_procrustes_too_few_points_error() {
    // 2 points → not enough
    let src = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
    let tgt = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
    assert!(align_procrustes(&src, &tgt).is_err());
}

#[test]
fn test_procrustes_mismatched_count_error() {
    let src = tetra(); // 4 points
    let tgt = vec![0.0f32; 9]; // 3 points
    assert!(matches!(
        align_procrustes(&src, &tgt),
        Err(AlignmentError::PointCountMismatch { .. })
    ));
}

#[test]
fn test_procrustes_empty_error() {
    let empty: Vec<f32> = vec![];
    assert!(matches!(
        align_procrustes(&empty, &empty),
        Err(AlignmentError::EmptyInput)
    ));
}

// ── align_procrustes_rigid ─────────────────────────────────────────────

#[test]
fn test_procrustes_rigid_scale_is_one() {
    let src = tetra();
    let tgt = apply_transform_flat(&src, &mat3_identity(), [0.0; 3], 3.0);
    let xf = align_procrustes_rigid(&src, &tgt).expect("rigid proc");
    assert_float_eq(xf.scale, 1.0, 1e-9, "rigid scale must be 1");
}

#[test]
fn test_procrustes_rigid_rotation_rmse() {
    use std::f32::consts::FRAC_PI_4;
    let src = tetra();
    let r = rot_z(FRAC_PI_4);
    let tgt = apply_transform_flat(&src, &r, [0.0; 3], 1.0);
    let xf = align_procrustes_rigid(&src, &tgt).expect("rigid proc");
    let rmse = align_rmse(&src, &tgt, &xf).expect("rmse");
    assert!(rmse < 1e-3, "rigid rotation rmse {rmse}");
}

#[test]
fn test_procrustes_rigid_translation() {
    let src = tetra();
    let tgt = apply_transform_flat(&src, &mat3_identity(), [2.0, -1.0, 0.5], 1.0);
    let xf = align_procrustes_rigid(&src, &tgt).expect("rigid proc");
    let rmse = align_rmse(&src, &tgt, &xf).expect("rmse");
    assert!(rmse < 1e-3, "rigid translation rmse {rmse}");
}

// ── align_rmse ────────────────────────────────────────────────────────

#[test]
fn test_rmse_identical_zero() {
    let pts = tetra();
    let id = SimilarityTransform::identity();
    let rmse = align_rmse(&pts, &pts, &id).expect("rmse");
    assert!(rmse < 1e-6, "rmse of identical pts {rmse}");
}

#[test]
fn test_rmse_known_error() {
    // Source: origin; target: (1,0,0) — RMSE = 1
    let src = vec![0.0f32, 0.0, 0.0];
    let tgt = vec![1.0f32, 0.0, 0.0];
    let id = SimilarityTransform::identity();
    let rmse = align_rmse(&src, &tgt, &id).expect("rmse");
    assert_float_eq(rmse, 1.0, 1e-5, "known rmse");
}

#[test]
fn test_rmse_mismatch_error() {
    let src = vec![0.0f32; 6];
    let tgt = vec![0.0f32; 9];
    assert!(matches!(
        align_rmse(&src, &tgt, &SimilarityTransform::identity()),
        Err(AlignmentError::PointCountMismatch { .. })
    ));
}

#[test]
fn test_rmse_after_procrustes_low() {
    let src = tetra();
    let r = rot_z(0.6283); // ~π/5 in radians
    let tgt = apply_transform_flat(&src, &r, [1.0, 2.0, 3.0], 1.5);
    let xf = align_procrustes(&src, &tgt).expect("proc");
    let rmse = align_rmse(&src, &tgt, &xf).expect("rmse");
    assert!(rmse < 0.05, "rmse after procrustes {rmse}");
}

// ── align_nearest_neighbors ───────────────────────────────────────────

#[test]
fn test_nearest_neighbors_simple() {
    // source: two points; target: three points
    let src = vec![0.1f32, 0.0, 0.0, 2.0, 0.0, 0.0];
    let tgt = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let idx = align_nearest_neighbors(&src, &tgt).expect("nn");
    assert_eq!(idx[0], 0); // 0.1 nearest to 0.0
    assert_eq!(idx[1], 2); // 2.0 nearest to 2.0
}

#[test]
fn test_nearest_neighbors_empty_source_error() {
    let tgt = vec![0.0f32; 9];
    assert!(align_nearest_neighbors(&[], &tgt).is_err());
}

#[test]
fn test_nearest_neighbors_empty_target_error() {
    let src = vec![0.0f32; 9];
    assert!(align_nearest_neighbors(&src, &[]).is_err());
}

#[test]
fn test_nearest_neighbors_count() {
    let src = tetra();
    let tgt = tetra();
    let idx = align_nearest_neighbors(&src, &tgt).expect("nn");
    assert_eq!(idx.len(), 4);
}

// ── align_nearest_neighbors_filtered ─────────────────────────────────

#[test]
fn test_nn_filtered_reject_far() {
    let src = vec![10.0f32, 0.0, 0.0]; // far from target
    let tgt = vec![0.0f32, 0.0, 0.0];
    let (idx, dist) = align_nearest_neighbors_filtered(&src, &tgt, 1.0).expect("nn_f");
    assert_eq!(idx[0], usize::MAX);
    assert!(dist[0] > 1.0);
}

#[test]
fn test_nn_filtered_accept_near() {
    let src = vec![0.5f32, 0.0, 0.0];
    let tgt = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
    let (idx, dist) = align_nearest_neighbors_filtered(&src, &tgt, 2.0).expect("nn_f");
    assert_ne!(idx[0], usize::MAX);
    assert!(dist[0] < 2.0);
}

#[test]
fn test_nn_filtered_infinity_accepts_all() {
    let src = vec![100.0f32, 0.0, 0.0];
    let tgt = vec![0.0f32, 0.0, 0.0];
    let (idx, _) = align_nearest_neighbors_filtered(&src, &tgt, f32::INFINITY).expect("nn_f");
    assert_eq!(idx[0], 0);
}

// ── IcpConfig ─────────────────────────────────────────────────────────

#[test]
fn test_icp_config_default_fields() {
    let cfg = IcpConfig::default();
    assert_eq!(cfg.max_iterations, 50);
    assert!(cfg.convergence_threshold < 1e-4);
    assert!(!cfg.use_scale);
    assert!(cfg.max_correspondence_dist.is_infinite());
}

// ── align_icp ─────────────────────────────────────────────────────────

#[test]
fn test_icp_identical_rmse_zero() {
    let pts = tetra();
    let cfg = IcpConfig::default();
    let result = align_icp(&pts, &pts, &cfg).expect("icp");
    assert!(
        result.final_rmse < 1e-4,
        "icp identity rmse {}",
        result.final_rmse
    );
}

#[test]
fn test_icp_identical_converged() {
    let pts = tetra();
    let cfg = IcpConfig::default();
    let result = align_icp(&pts, &pts, &cfg).expect("icp");
    assert!(result.converged);
}

#[test]
fn test_icp_translation_recovery() {
    // Use a point set where all nearest-neighbour correspondences are unambiguous
    // (large translation relative to inter-point spacing would break NN,
    // so we use a small translation with a rotationally-symmetric-free layout).
    let src = vec![
        0.0f32, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0,
    ];
    // Translate by [0.1, 0, 0] — small compared to inter-point spacing (3.0)
    let tgt = apply_transform_flat(&src, &mat3_identity(), [0.1, 0.0, 0.0], 1.0);
    let cfg = IcpConfig {
        max_iterations: 100,
        convergence_threshold: 1e-6,
        ..Default::default()
    };
    let result = align_icp(&src, &tgt, &cfg).expect("icp");
    let aligned = result.transform.apply_flat(&src);
    let rmse = align_rmse(&aligned, &tgt, &SimilarityTransform::identity()).expect("rmse");
    assert!(rmse < 0.05, "icp translation rmse {rmse}");
}

#[test]
fn test_icp_too_few_points_error() {
    let src = vec![0.0f32; 6]; // 2 points
    let tgt = vec![0.0f32; 9];
    let cfg = IcpConfig::default();
    assert!(align_icp(&src, &tgt, &cfg).is_err());
}

#[test]
fn test_icp_zero_iterations_error() {
    let pts = tetra();
    let cfg = IcpConfig {
        max_iterations: 0,
        ..Default::default()
    };
    assert!(align_icp(&pts, &pts, &cfg).is_err());
}

#[test]
fn test_icp_rmse_history_nonempty() {
    let pts = tetra();
    let cfg = IcpConfig::default();
    let result = align_icp(&pts, &pts, &cfg).expect("icp");
    assert!(!result.rmse_history.is_empty());
}

#[test]
fn test_icp_converged_flag_true_when_close() {
    let pts = tetra();
    let tgt = apply_transform_flat(&pts, &mat3_identity(), [0.001, 0.0, 0.0], 1.0);
    let cfg = IcpConfig {
        max_iterations: 200,
        convergence_threshold: 1e-4,
        ..Default::default()
    };
    let result = align_icp(&pts, &tgt, &cfg).expect("icp");
    // convergence is expected for such a tiny offset
    assert!(result.converged || result.final_rmse < 0.01);
}

// ── align_by_landmarks ────────────────────────────────────────────────

#[test]
fn test_landmarks_same_as_procrustes() {
    let src = tetra();
    let tgt = apply_transform_flat(&src, &mat3_identity(), [1.0, 2.0, 3.0], 1.0);
    let xf_lm = align_by_landmarks(&src, &tgt).expect("lm");
    let xf_pr = align_procrustes(&src, &tgt).expect("pr");
    assert_float_eq(xf_lm.scale, xf_pr.scale, 1e-5, "landmark scale");
    for i in 0..3 {
        assert_float_eq(xf_lm.translation[i], xf_pr.translation[i], 1e-4, "lm t");
    }
}

#[test]
fn test_landmarks_rmse_low() {
    use std::f32::consts::FRAC_PI_3;
    let src = tetra();
    let r = rot_z(FRAC_PI_3);
    let tgt = apply_transform_flat(&src, &r, [0.5, -0.5, 0.0], 1.2);
    let xf = align_by_landmarks(&src, &tgt).expect("lm");
    let rmse = align_rmse(&src, &tgt, &xf).expect("rmse");
    assert!(rmse < 0.05, "landmark rmse {rmse}");
}

// ── align_by_weighted_landmarks ───────────────────────────────────────

#[test]
fn test_weighted_landmarks_uniform_weights() {
    // Uniform weights ≡ unweighted Procrustes
    let src = tetra();
    let tgt = apply_transform_flat(&src, &mat3_identity(), [1.0, 0.0, 0.0], 1.0);
    let weights = vec![1.0f32; 4];
    let xf_w = align_by_weighted_landmarks(&src, &tgt, &weights).expect("w_lm");
    let xf_u = align_procrustes(&src, &tgt).expect("pr");
    assert_float_eq(xf_w.scale, xf_u.scale, 1e-4, "uniform w scale");
    for i in 0..3 {
        assert_float_eq(
            xf_w.translation[i],
            xf_u.translation[i],
            1e-3,
            "uniform w t",
        );
    }
}

#[test]
fn test_weighted_landmarks_zero_weight_negligible_effect() {
    // Three good landmarks + one zero-weight outlier
    let mut src = tetra();
    let mut tgt = apply_transform_flat(&tetra(), &mat3_identity(), [2.0, 0.0, 0.0], 1.0);
    // Add a wildly wrong 4th point pair; zero-weight it
    // (tetra already has 4 points — set last to something far)
    src[9] = 100.0;
    tgt[9] = -100.0;
    let weights = vec![1.0f32, 1.0, 1.0, 0.0];
    let xf = align_by_weighted_landmarks(&src, &tgt, &weights).expect("w_lm_z");
    // Should still recover approximately translation=[2,0,0]
    assert_float_eq(xf.translation[0], 2.0, 0.3, "zero-w outlier tx");
}

#[test]
fn test_weighted_landmarks_mismatch_error() {
    let src = tetra();
    let tgt = tetra();
    let weights = vec![1.0f32; 3]; // should be 4
    assert!(matches!(
        align_by_weighted_landmarks(&src, &tgt, &weights),
        Err(AlignmentError::WeightLengthMismatch { .. })
    ));
}

#[test]
fn test_weighted_landmarks_count_mismatch_error() {
    let src = tetra(); // 4 pts
    let tgt = vec![0.0f32; 9]; // 3 pts
    let weights = vec![1.0f32; 4];
    assert!(matches!(
        align_by_weighted_landmarks(&src, &tgt, &weights),
        Err(AlignmentError::PointCountMismatch { .. })
    ));
}

#[test]
fn test_weighted_landmarks_zero_sum_weights_error() {
    let src = tetra();
    let tgt = tetra();
    let weights = vec![0.0f32; 4];
    assert!(align_by_weighted_landmarks(&src, &tgt, &weights).is_err());
}

// ── align_compute_stats ───────────────────────────────────────────────

#[test]
fn test_stats_improvement_ratio_gt_one() {
    let src = tetra();
    let tgt = apply_transform_flat(&src, &mat3_identity(), [5.0, 0.0, 0.0], 1.0);
    let xf = align_procrustes(&src, &tgt).expect("proc");
    let stats = align_compute_stats(&src, &tgt, &xf).expect("stats");
    assert!(
        stats.improvement_ratio > 1.0 || stats.rmse_after < 1e-3,
        "ratio {} rmse_after {}",
        stats.improvement_ratio,
        stats.rmse_after
    );
}

#[test]
fn test_stats_rmse_before_and_after() {
    let src = tetra();
    let tgt = apply_transform_flat(&src, &mat3_identity(), [3.0, 0.0, 0.0], 1.0);
    let xf = align_procrustes(&src, &tgt).expect("proc");
    let stats = align_compute_stats(&src, &tgt, &xf).expect("stats");
    // Before should be nonzero; after should be small
    assert!(stats.rmse_before > 0.1, "rmse_before should be nonzero");
    assert!(stats.rmse_after < stats.rmse_before, "after < before");
}

#[test]
fn test_stats_mismatch_error() {
    let src = vec![0.0f32; 6];
    let tgt = vec![0.0f32; 9];
    assert!(align_compute_stats(&src, &tgt, &SimilarityTransform::identity()).is_err());
}

// ── align_format_stats ────────────────────────────────────────────────

#[test]
fn test_format_stats_nonempty() {
    let src = tetra();
    let tgt = apply_transform_flat(&src, &mat3_identity(), [1.0, 0.0, 0.0], 1.0);
    let xf = align_procrustes(&src, &tgt).expect("proc");
    let stats = align_compute_stats(&src, &tgt, &xf).expect("stats");
    let s = align_format_stats(&stats);
    assert!(!s.is_empty());
    assert!(s.contains("RMSE"), "format contains RMSE");
}

#[test]
fn test_format_stats_contains_values() {
    let src = tetra();
    let tgt = tetra();
    let xf = SimilarityTransform::identity();
    let stats = align_compute_stats(&src, &tgt, &xf).expect("stats");
    let s = align_format_stats(&stats);
    assert!(s.contains("Improvement"), "format contains Improvement");
}

// ── align_format_icp_result ───────────────────────────────────────────

#[test]
fn test_format_icp_nonempty() {
    let pts = tetra();
    let cfg = IcpConfig::default();
    let result = align_icp(&pts, &pts, &cfg).expect("icp");
    let s = align_format_icp_result(&result);
    assert!(!s.is_empty());
    assert!(
        s.contains("Converged") || s.contains("RMSE"),
        "format info present"
    );
}

#[test]
fn test_format_icp_contains_scale() {
    let pts = tetra();
    let cfg = IcpConfig::default();
    let result = align_icp(&pts, &pts, &cfg).expect("icp");
    let s = align_format_icp_result(&result);
    assert!(s.contains("Scale"), "format contains scale");
}

// ── mat3_mul ──────────────────────────────────────────────────────────

#[test]
fn test_mat3_mul_identity_left() {
    let id = mat3_identity();
    let m = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let r = mat3_mul(&id, &m);
    for i in 0..9 {
        assert_float_eq(r[i], m[i], 1e-6, "id*M");
    }
}

#[test]
fn test_mat3_mul_identity_right() {
    let id = mat3_identity();
    let m = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let r = mat3_mul(&m, &id);
    for i in 0..9 {
        assert_float_eq(r[i], m[i], 1e-6, "M*id");
    }
}

#[test]
fn test_mat3_mul_known() {
    // [[1,0],[0,-1]] * [[0,1],[-1,0]] = [[0,1],[1,0]] (2D embedded in 3D)
    let a = [1.0f32, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0];
    let b = [0.0f32, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let r = mat3_mul(&a, &b);
    assert_float_eq(r[0], 0.0, 1e-6, "r[0]");
    assert_float_eq(r[1], 1.0, 1e-6, "r[1]");
    assert_float_eq(r[3], 1.0, 1e-6, "r[3]");
    assert_float_eq(r[4], 0.0, 1e-6, "r[4]");
}

// ── mat3_det ──────────────────────────────────────────────────────────

#[test]
fn test_mat3_det_identity_is_one() {
    let id = mat3_identity();
    assert_float_eq(mat3_det(&id), 1.0, 1e-6, "det(I)");
}

#[test]
fn test_mat3_det_rotation_is_one() {
    use std::f32::consts::FRAC_PI_4;
    let r = rot_z(FRAC_PI_4);
    assert_float_eq(mat3_det(&r), 1.0, 1e-5, "det(R)");
}

#[test]
fn test_mat3_det_known() {
    // [[1,2,3],[0,1,4],[5,6,0]] → det = 1*(0-24) - 2*(0-20) + 3*(0-5) = -24+40-15 = 1
    let m = [1.0f32, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0];
    assert_float_eq(mat3_det(&m), 1.0, 1e-5, "det known");
}

// ── svd_3x3 ───────────────────────────────────────────────────────────

#[test]
fn test_svd_reconstruction_identity() {
    let id = mat3_identity();
    let (u, s, vt) = svd_3x3(&id);
    let diag_s = [
        s[0] * u[0] + s[1] * u[1] + s[2] * u[2], // wrong — below is correct
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ];
    let _ = diag_s;
    // Check U * diag(S) * Vt ≈ I
    let mut ds = [0.0f32; 9]; // diag(S) * Vt
    for row in 0..3 {
        for col in 0..3 {
            ds[row * 3 + col] = s[row] * vt[row * 3 + col];
        }
    }
    let recon = mat3_mul(&u, &ds);
    for i in 0..9 {
        assert_float_eq(recon[i], id[i], 1e-4, &format!("svd id recon[{i}]"));
    }
}

#[test]
fn test_svd_reconstruction_general() {
    let m = [3.0f32, 1.0, 0.5, -1.0, 2.0, 0.7, 0.2, -0.3, 1.5];
    let (u, s, vt) = svd_3x3(&m);
    let mut ds = [0.0f32; 9];
    for row in 0..3 {
        for col in 0..3 {
            ds[row * 3 + col] = s[row] * vt[row * 3 + col];
        }
    }
    let recon = mat3_mul(&u, &ds);
    for i in 0..9 {
        assert_float_eq(recon[i], m[i], 1e-3, &format!("svd gen recon[{i}]"));
    }
}

#[test]
fn test_svd_singular_values_nonneg() {
    let m = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let (_u, s, _vt) = svd_3x3(&m);
    for (i, &si) in s.iter().enumerate() {
        assert!(si >= 0.0, "s[{i}] = {si} should be non-negative");
    }
}

#[test]
fn test_svd_singular_values_sorted_desc() {
    let m = [3.0f32, 1.0, 0.5, -1.0, 2.0, 0.7, 0.2, -0.3, 1.5];
    let (_u, s, _vt) = svd_3x3(&m);
    assert!(s[0] >= s[1], "s[0]={} >= s[1]={}", s[0], s[1]);
    assert!(s[1] >= s[2], "s[1]={} >= s[2]={}", s[1], s[2]);
}

#[test]
fn test_svd_u_orthogonal() {
    let m = [3.0f32, 1.0, 0.5, -1.0, 2.0, 0.7, 0.2, -0.3, 1.5];
    let (u, _s, _vt) = svd_3x3(&m);
    let utu = mat3_mul(&mat3_transpose(&u), &u);
    let id = mat3_identity();
    for i in 0..9 {
        assert_float_eq(utu[i], id[i], 1e-4, &format!("U^T U [{i}]"));
    }
}

// ── Extra coverage ────────────────────────────────────────────────────

#[test]
fn test_icp_with_scale() {
    let src = tetra();
    let tgt = apply_transform_flat(&src, &mat3_identity(), [0.3, 0.0, 0.0], 1.2);
    let cfg = IcpConfig {
        use_scale: true,
        max_iterations: 50,
        ..Default::default()
    };
    let result = align_icp(&src, &tgt, &cfg).expect("icp scale");
    let aligned = result.transform.apply_flat(&src);
    let rmse = align_rmse(&aligned, &tgt, &SimilarityTransform::identity()).expect("rmse");
    assert!(rmse < 0.5, "icp with scale rmse {rmse}");
}

#[test]
fn test_stats_n_correspondences() {
    let src = tetra();
    let tgt = tetra();
    let xf = SimilarityTransform::identity();
    let stats = align_compute_stats(&src, &tgt, &xf).expect("stats");
    assert_eq!(stats.n_correspondences, 4);
}

#[test]
fn test_apply_batch_matches_flat() {
    let xf = SimilarityTransform {
        rotation: rot_z(0.5),
        translation: [1.0, -1.0, 2.0],
        scale: 1.5,
    };
    let pts_arr: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let flat: Vec<f32> = pts_arr.iter().flat_map(|p| *p).collect();
    let batch_out = xf.apply_batch(&pts_arr);
    let flat_out = xf.apply_flat(&flat);
    for i in 0..2 {
        assert_float_eq(batch_out[i][0], flat_out[i * 3], 1e-6, "batch/flat x");
        assert_float_eq(batch_out[i][1], flat_out[i * 3 + 1], 1e-6, "batch/flat y");
        assert_float_eq(batch_out[i][2], flat_out[i * 3 + 2], 1e-6, "batch/flat z");
    }
}

#[test]
fn test_procrustes_then_inverse_cancels() {
    let src = tetra();
    let tgt = apply_transform_flat(&src, &mat3_identity(), [2.0, 1.0, -1.0], 1.0);
    let xf = align_procrustes(&src, &tgt).expect("proc");
    let inv = xf.inverse();
    let composed = xf.compose(&inv);
    // compose(xf, inv) ≈ identity
    let p = [1.5f32, 0.5, 0.25];
    let q = composed.apply(p);
    assert_float_eq(q[0], p[0], 1e-4, "cancel x");
    assert_float_eq(q[1], p[1], 1e-4, "cancel y");
    assert_float_eq(q[2], p[2], 1e-4, "cancel z");
}

// ── Regression: ICP's KD-tree correspondence search ──────────────────

#[test]
fn test_kdtree_nn_matches_brute_force() {
    let src = vec![0.1f32, 0.0, 0.0, 2.0, 0.0, 0.0, 50.0, 0.0, 0.0];
    let tgt = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let mut tree: KdTree<f32, 3> = KdTree::with_capacity(3);
    for i in 0..3 {
        tree.add(&[tgt[i * 3], tgt[i * 3 + 1], tgt[i * 3 + 2]], i as u64);
    }
    let kd_idx = nearest_neighbors_kdtree(&tree, &src, 10.0);
    let (bf_idx, _) = align_nearest_neighbors_filtered(&src, &tgt, 10.0).expect("bf");
    assert_eq!(
        kd_idx, bf_idx,
        "KD-tree correspondences must match brute-force"
    );
}

// ── Regression: Umeyama scale reflection-sign correction ─────────────

#[test]
fn test_procrustes_reflected_points_uses_corrected_scale() {
    // target = source reflected through Z=0: for a non-degenerate
    // tetrahedron this makes det(cross-covariance) < 0, exercising the
    // reflection-sign correction on the scale.
    let src = tetra();
    let tgt: Vec<f32> = src
        .chunks_exact(3)
        .flat_map(|p| [p[0], p[1], -p[2]])
        .collect();

    let mu_s = centroid(&src, 4);
    let var_s = point_variance(&src, 4, mu_s);
    let sigma = cross_covariance(&src, &tgt, 4, mu_s, centroid(&tgt, 4));
    assert!(mat3_det(&sigma) < 0.0, "setup: det(Sigma) must be negative");
    let (_u, sv, _vt) = svd_3x3(&sigma);
    let corrected_scale = (sv[0] + sv[1] - sv[2]) / var_s;
    let buggy_scale = (sv[0] + sv[1] + sv[2]) / var_s;
    assert!(
        (corrected_scale - buggy_scale).abs() > 1e-3,
        "setup: corrected and buggy scale must actually differ"
    );

    let xf = align_procrustes(&src, &tgt).expect("proc failed");
    assert_float_eq(
        xf.scale,
        corrected_scale,
        1e-3,
        "reflection-corrected scale",
    );
}

// ── Regression: svd_3x3 reconstruction under the reflection correction ─

#[test]
fn test_svd_reflection_case_reconstructs_with_correction() {
    // det(M) < 0: U * diag(S) * Vt alone does NOT reconstruct M here;
    // U * diag(1,1,-1) * diag(S) * Vt does (see svd_3x3's doc).
    let m = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
    assert!(mat3_det(&m) < 0.0, "setup: det(M) must be negative");
    let (u, s, vt) = svd_3x3(&m);
    assert_float_eq(mat3_det(&u), 1.0, 1e-5, "det(U) must be +1");

    let mut ds = [0.0f32; 9]; // diag(1,1,-1) * diag(S) * Vt
    for col in 0..3 {
        ds[col] = s[0] * vt[col];
        ds[3 + col] = s[1] * vt[3 + col];
        ds[6 + col] = -s[2] * vt[6 + col];
    }
    let recon = mat3_mul(&u, &ds);
    for i in 0..9 {
        assert_float_eq(recon[i], m[i], 1e-4, &format!("reflected recon[{i}]"));
    }
}
