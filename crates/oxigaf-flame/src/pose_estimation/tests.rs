// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

use super::math::*;
use super::*;
use std::f32::consts::PI;

// -- Landmark constructors --

#[test]
fn test_landmark2d_new() {
    let lm = Landmark2D::new(10.0, 20.0);
    assert!((lm.u - 10.0).abs() < 1e-6);
    assert!((lm.v - 20.0).abs() < 1e-6);
    assert!((lm.confidence - 1.0).abs() < 1e-6);
}

#[test]
fn test_landmark2d_with_confidence() {
    let lm = Landmark2D::with_confidence(5.0, 8.0, 0.75);
    assert!((lm.confidence - 0.75).abs() < 1e-6);
}

#[test]
fn test_landmark3d_new() {
    let lm = Landmark3D::new(1.0, 2.0, 3.0);
    assert!((lm.x - 1.0).abs() < 1e-6);
    assert!((lm.y - 2.0).abs() < 1e-6);
    assert!((lm.z - 3.0).abs() < 1e-6);
}

// -- PosePinholeCamera --

#[test]
fn test_pinhole_from_image_size_proportions() {
    let cam = PosePinholeCamera::from_image_size(640, 480);
    // focal = max(640, 480) = 640
    assert!((cam.focal_length - 640.0).abs() < 1e-6);
    assert!((cam.cx - 320.0).abs() < 1e-6);
    assert!((cam.cy - 240.0).abs() < 1e-6);
}

#[test]
fn test_pinhole_project_behind_camera_returns_none() {
    let cam = PosePinholeCamera::new(500.0, 320.0, 240.0);
    assert!(cam.project(0.1, 0.1, -1.0).is_none());
    assert!(cam.project(0.1, 0.1, 0.0).is_none());
}

#[test]
fn test_pinhole_project_forward_point_in_image() {
    // A point along the optical axis projects to the principal point
    let cam = PosePinholeCamera::new(500.0, 320.0, 240.0);
    let (u, v) = cam.project(0.0, 0.0, 1.0).expect("should project");
    assert!((u - 320.0).abs() < 1e-4);
    assert!((v - 240.0).abs() < 1e-4);
}

#[test]
fn test_pinhole_project_offset_point() {
    let cam = PosePinholeCamera::new(500.0, 320.0, 240.0);
    // x > 0 should yield u > cx
    let (u, _v) = cam.project(0.1, 0.0, 1.0).expect("should project");
    assert!(u > 320.0);
    // y > 0 should yield v < cy (y is flipped)
    let (_u, v) = cam.project(0.0, 0.1, 1.0).expect("should project");
    assert!(v < 240.0);
}

#[test]
fn test_pinhole_unproject_roundtrip() {
    let cam = PosePinholeCamera::new(500.0, 320.0, 240.0);
    let orig_3d = (0.2_f32, -0.15_f32, 2.0_f32);
    let (u, v) = cam
        .project(orig_3d.0, orig_3d.1, orig_3d.2)
        .expect("should project");
    let (rx, ry, rz) = cam.unproject(u, v, orig_3d.2);
    assert!((rx - orig_3d.0).abs() < 1e-4, "x: {rx} vs {}", orig_3d.0);
    assert!((ry - orig_3d.1).abs() < 1e-4, "y: {ry} vs {}", orig_3d.1);
    assert!((rz - orig_3d.2).abs() < 1e-6);
}

// -- HeadPose --

#[test]
fn test_headpose_transform_identity_rotation() {
    let pose = HeadPose::new(mat3_identity(), [1.0, 2.0, 3.0]);
    let pt = Landmark3D::new(4.0, 5.0, 6.0);
    let (cx, cy, cz) = pose.transform(pt);
    assert!((cx - 5.0).abs() < 1e-5);
    assert!((cy - 7.0).abs() < 1e-5);
    assert!((cz - 9.0).abs() < 1e-5);
}

#[test]
fn test_headpose_euler_angles_identity() {
    let pose = HeadPose::new(mat3_identity(), [0.0, 0.0, 1.0]);
    let [yaw, pitch, roll] = pose.euler_angles();
    assert!(yaw.abs() < 1e-5);
    assert!(pitch.abs() < 1e-5);
    assert!(roll.abs() < 1e-5);
}

#[test]
fn test_headpose_rotation_axis_angle_identity() {
    let pose = HeadPose::new(mat3_identity(), [0.0, 0.0, 1.0]);
    assert!(vec3_norm(pose.rotation_axis_angle()) < 1e-5);
}

#[test]
fn test_headpose_rotation_axis_angle_known() {
    // 90° rotation around z-axis
    let r = [[0.0_f32, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let pose = HeadPose::new(r, [0.0, 0.0, 1.0]);
    let aa = pose.rotation_axis_angle();
    // axis should be [0, 0, π/2]
    assert!(aa[0].abs() < 1e-4, "ax={}", aa[0]);
    assert!(aa[1].abs() < 1e-4, "ay={}", aa[1]);
    assert!((aa[2] - PI / 2.0).abs() < 1e-4, "az={}", aa[2]);
}

// -- PoseConfig --

#[test]
fn test_pose_config_validate_valid() {
    let cfg = PoseConfig::default();
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_pose_config_validate_rejects_bad_fields() {
    let bad = [
        PoseConfig {
            min_correspondences: 3,
            ..Default::default()
        },
        PoseConfig {
            max_reprojection_error: -1.0,
            ..Default::default()
        },
        PoseConfig {
            min_confidence: 1.5,
            ..Default::default()
        },
    ];
    for cfg in &bad {
        assert!(matches!(
            cfg.validate(),
            Err(PoseEstimationError::InvalidConfig(_))
        ));
    }
}

// -- estimate_pose_weak_perspective --

#[test]
fn test_estimate_pose_weak_perspective_insufficient_points() {
    let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
    let config = PoseConfig::default();
    // Only 2 correspondences
    let corrs: Vec<PointCorrespondence> = (0..2)
        .map(|i| {
            let fi = i as f32;
            PointCorrespondence::new(
                Landmark3D::new(fi, fi, 1.0),
                Landmark2D::new(320.0 + fi * 10.0, 240.0),
            )
        })
        .collect();
    let result = estimate_pose_weak_perspective(&corrs, &camera, &config);
    assert!(matches!(
        result,
        Err(PoseEstimationError::InsufficientPoints { .. })
    ));
}

#[test]
fn test_estimate_pose_weak_perspective_front_facing() {
    // Front-facing head: cross of model points at z=0 observed at unit
    // depth (u = cx + f*x, v = cy - f*y), so t_z ≈ 1 and R ≈ I.
    let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
    let config = PoseConfig {
        min_confidence: 0.0,
        ..Default::default()
    };
    let model_pts = [
        Landmark3D::new(-0.1, 0.0, 0.0),
        Landmark3D::new(0.1, 0.0, 0.0),
        Landmark3D::new(0.0, -0.1, 0.0),
        Landmark3D::new(0.0, 0.1, 0.0),
    ];
    let corrs: Vec<PointCorrespondence> = model_pts
        .iter()
        .map(|p| {
            let u = camera.cx + camera.focal_length * p.x;
            let v = camera.cy - camera.focal_length * p.y;
            PointCorrespondence::new(*p, Landmark2D::new(u, v))
        })
        .collect();

    let pose = estimate_pose_weak_perspective(&corrs, &camera, &config)
        .expect("front-facing solve should succeed");
    let (tx, ty) = (pose.translation[0], pose.translation[1]);
    assert!(tx.abs() < 0.1 && ty.abs() < 0.1, "tx={tx} ty={ty}");
    let err = pose.reprojection_error;
    assert!(err < 5.0, "reprojection_error={err}");
}

#[test]
fn test_estimate_pose_weak_perspective_degenerate_coincident_points() {
    // All 3D points are at the same location → scale cannot be estimated
    let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
    let config = PoseConfig {
        min_confidence: 0.0,
        ..Default::default()
    };
    let corrs: Vec<PointCorrespondence> = (0..4)
        .map(|_| {
            PointCorrespondence::new(
                Landmark3D::new(0.0, 0.0, 1.0),
                Landmark2D::new(320.0, 240.0),
            )
        })
        .collect();
    let result = estimate_pose_weak_perspective(&corrs, &camera, &config);
    assert!(matches!(
        result,
        Err(PoseEstimationError::NumericalError(_))
    ));
}

// -- estimate_yaw_from_symmetry --

#[test]
fn test_estimate_yaw_symmetric_is_zero() {
    // Perfectly symmetric left/right landmarks → yaw ≈ 0
    let left = vec![[100.0_f32, 200.0]];
    let right = vec![[300.0_f32, 200.0]];
    let yaw = estimate_yaw_from_symmetry(&left, &right).expect("should succeed");
    assert!(yaw.abs() < 0.1, "yaw={yaw}");
}

#[test]
fn test_estimate_yaw_asymmetric_nonzero() {
    // Left side wider than right → face turned toward the right (positive yaw).
    let left = vec![[80.0_f32, 200.0], [150.0, 200.0]]; // spread = 70
    let right = vec![[300.0_f32, 200.0], [320.0, 200.0]]; // spread = 20
    let yaw = estimate_yaw_from_symmetry(&left, &right).expect("should succeed");
    // yaw = asin((70-20)/(70+20)) = asin(50/90) ≈ asin(0.556) ≈ 0.59 rad
    assert!(yaw > 0.1, "expected positive nonzero yaw, got {yaw}");
}

#[test]
fn test_estimate_yaw_asymmetry_sign() {
    // Frontal face: both sides have equal internal spread (20 px each).
    let left_frontal = [[90.0_f32, 200.0], [110.0, 200.0]];
    let right_frontal = [[290.0_f32, 200.0], [310.0, 200.0]];
    let yaw_frontal =
        estimate_yaw_from_symmetry(&left_frontal, &right_frontal).expect("should succeed");

    // Turned right: the left side expands (spread 50) while the right stays.
    let left_turned = [[80.0_f32, 200.0], [130.0, 200.0]];
    let yaw_turned =
        estimate_yaw_from_symmetry(&left_turned, &right_frontal).expect("should succeed");

    assert!(yaw_frontal.abs() < 0.01, "frontal yaw={yaw_frontal}");
    assert!(
        yaw_turned.abs() > yaw_frontal.abs(),
        "frontal={yaw_frontal}, turned={yaw_turned}"
    );
}

#[test]
fn test_estimate_yaw_mismatched_lengths_error() {
    let left = vec![[100.0_f32, 200.0], [110.0, 210.0]];
    let right = vec![[300.0_f32, 200.0]];
    let result = estimate_yaw_from_symmetry(&left, &right);
    assert!(matches!(
        result,
        Err(PoseEstimationError::InsufficientPoints { .. })
    ));
}

#[test]
fn test_estimate_yaw_empty_input_error() {
    let result = estimate_yaw_from_symmetry(&[], &[]);
    assert!(matches!(
        result,
        Err(PoseEstimationError::InsufficientPoints { .. })
    ));
}

// -- estimate_pitch_from_vertical --

/// Upper group 0.12 above and 0.03 in front of the lower group, at 250 px
/// per model unit.
fn pitch_reference() -> PitchReference {
    PitchReference::new(
        Landmark3D::new(0.0, 0.06, 0.05),
        Landmark3D::new(0.0, -0.06, 0.02),
        250.0,
    )
}

/// Weak-perspective projection (`v = −scale·y_cam`; the constant `cy`
/// cancels because only the difference is used) of the two reference group
/// centroids after a pitch rotation of `theta` about the camera x-axis.
fn project_pitched_groups(theta: f32, reference: &PitchReference) -> ([f32; 2], [f32; 2]) {
    let rot = |p: Landmark3D| p.y * theta.cos() - p.z * theta.sin();
    let upper_y = rot(reference.upper_3d);
    let lower_y = rot(reference.lower_3d);
    (
        [10.0, -reference.scale * upper_y],
        [10.0, -reference.scale * lower_y],
    )
}

#[test]
fn test_estimate_pitch_empty_upper_error() {
    let result = estimate_pitch_from_vertical(&[], &[[200.0_f32, 350.0]], &pitch_reference());
    assert!(matches!(
        result,
        Err(PoseEstimationError::InsufficientPoints { .. })
    ));
}

#[test]
fn test_estimate_pitch_empty_lower_error() {
    let result = estimate_pitch_from_vertical(&[[200.0_f32, 100.0]], &[], &pitch_reference());
    assert!(matches!(
        result,
        Err(PoseEstimationError::InsufficientPoints { .. })
    ));
}

// -- reprojection_error --

#[test]
fn test_reprojection_error_empty() {
    let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
    let pose = HeadPose::new(mat3_identity(), [0.0, 0.0, 1.0]);
    let err = reprojection_error(&[], &pose, &camera);
    assert!((err - 0.0).abs() < 1e-9);
}

#[test]
fn test_reprojection_error_known_pose() {
    let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
    // Identity pose with z=1: model point (0,0,0) → camera (0,0,1) → image cx,cy
    let pose = HeadPose::new(mat3_identity(), [0.0, 0.0, 1.0]);
    let pt3d = Landmark3D::new(0.0, 0.0, 0.0);
    let (pu, pv) = camera.project(0.0, 0.0, 1.0).expect("should project");
    let corr = PointCorrespondence::new(pt3d, Landmark2D::new(pu, pv));
    let err = reprojection_error(&[corr], &pose, &camera);
    assert!(err < 1e-4, "err={err}");
}

// -- count_inliers --

#[test]
fn test_count_inliers_threshold_split() {
    let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
    let pose = HeadPose::new(mat3_identity(), [0.0, 0.0, 1.0]);
    let pt3d = Landmark3D::new(0.0, 0.0, 0.0);
    let (pu, pv) = camera.project(0.0, 0.0, 1.0).expect("should project");
    // Perfect correspondence → 0 error → inlier
    let good = PointCorrespondence::new(pt3d, Landmark2D::new(pu, pv));
    assert_eq!(count_inliers(&[good], &pose, &camera, 1.0), 1);
    // Observation far from the projection (cx, cy) → outlier
    let bad = PointCorrespondence::new(pt3d, Landmark2D::new(0.0, 0.0));
    assert_eq!(count_inliers(&[bad], &pose, &camera, 1.0), 0);
}

// -- PoseTracker --

#[test]
fn test_pose_tracker_new_has_no_pose() {
    let tracker = PoseTracker::new(0.7);
    assert!(!tracker.has_pose());
    assert!(tracker.current_pose().is_none());
}

#[test]
fn test_pose_tracker_update_single_pose() {
    let mut tracker = PoseTracker::new(0.7);
    let pose = HeadPose::new(mat3_identity(), [1.0, 2.0, 3.0]);
    tracker.update(&pose);
    assert!(tracker.has_pose());
    let current = tracker.current_pose().expect("should have pose");
    assert!((current.translation[0] - 1.0).abs() < 1e-5);
    assert!((current.translation[1] - 2.0).abs() < 1e-5);
    assert!((current.translation[2] - 3.0).abs() < 1e-5);
}

#[test]
fn test_pose_tracker_update_ema_smoothing() {
    let mut tracker = PoseTracker::new(0.7);
    let pose1 = HeadPose::new(mat3_identity(), [0.0, 0.0, 1.0]);
    let pose2 = HeadPose::new(mat3_identity(), [10.0, 10.0, 10.0]);
    tracker.update(&pose1);
    tracker.update(&pose2);

    let current = tracker.current_pose().expect("should have pose");
    // t_x = 0.7 * 0.0 + 0.3 * 10.0 = 3.0
    assert!(
        (current.translation[0] - 3.0).abs() < 1e-4,
        "tx={}",
        current.translation[0]
    );
}

#[test]
fn test_pose_tracker_reset() {
    let mut tracker = PoseTracker::new(0.7);
    let pose = HeadPose::new(mat3_identity(), [1.0, 0.0, 1.0]);
    tracker.update(&pose);
    assert!(tracker.has_pose());
    tracker.reset();
    assert!(!tracker.has_pose());
    assert!(tracker.current_pose().is_none());
}

// -- mat3 helpers (internal sanity) --

#[test]
fn test_mat3_mul_and_transpose() {
    let id = mat3_identity();
    let m = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let product = mat3_mul(&id, &m);
    let transposed = mat3_transpose(&m);
    for (i, row) in m.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            assert!((product[i][j] - val).abs() < 1e-6, "I*M != M");
            assert!((transposed[j][i] - val).abs() < 1e-6, "transpose");
        }
    }
}

#[test]
fn test_mat3_flat_roundtrip_and_pseudo_inverse() {
    let m = [[2.0_f32, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]];
    assert_eq!(mat3_from_flat(&mat3_to_flat(&m)), m);
    // Rank-2 input: the pseudo-inverse inverts the non-zero directions and
    // leaves the degenerate one at zero.
    let pinv = mat3_pseudo_inverse(&m);
    assert!((pinv[0][0] - 0.5).abs() < 1e-5, "{pinv:?}");
    assert!((pinv[1][1] - 0.25).abs() < 1e-5, "{pinv:?}");
    assert!(pinv[2][2].abs() < 1e-5, "{pinv:?}");
}

// -- estimate_pitch_from_vertical (geometric foreshortening solve) --

/// Smallest distance between `theta` and either returned candidate.
fn pitch_candidate_error(candidates: [f32; 2], theta: f32) -> f32 {
    (candidates[0] - theta)
        .abs()
        .min((candidates[1] - theta).abs())
}

/// Synthetically rotated landmarks must be recovered at known angles
/// (including the frontal case), with the candidates sorted ascending.
#[test]
fn test_pitch_recovers_known_rotation_both_signs() {
    let reference = pitch_reference();
    for &theta in &[-0.45_f32, -0.3, -0.1, 0.0, 0.1, 0.3, 0.45] {
        let (upper, lower) = project_pitched_groups(theta, &reference);
        let cands = estimate_pitch_from_vertical(&[upper], &[lower], &reference)
            .expect("pitch solve should succeed");
        let err = pitch_candidate_error(cands, theta);
        assert!(err < 1e-3, "theta={theta}: candidates={cands:?} err={err}");
        assert!(cands[0] <= cands[1], "unsorted candidates: {cands:?}");
    }
}

/// Regression for the cardinality bug: the previous implementation returned
/// exactly 0 whenever both groups had the same number of points, whatever
/// the actual geometry, and otherwise measured only the size imbalance.
#[test]
fn test_pitch_uses_geometry_not_group_cardinality() {
    let reference = pitch_reference();
    let theta = 0.25_f32;
    let (upper_c, lower_c) = project_pitched_groups(theta, &reference);

    // Two points per group, spread symmetrically about each centroid.
    let upper = [[8.0_f32, upper_c[1] - 3.0], [12.0, upper_c[1] + 3.0]];
    let lower = [[8.0_f32, lower_c[1] - 5.0], [12.0, lower_c[1] + 5.0]];
    let equal = estimate_pitch_from_vertical(&upper, &lower, &reference)
        .expect("pitch solve should succeed");
    assert!(
        pitch_candidate_error(equal, theta) < 1e-3,
        "equal-size groups must still recover theta={theta}, got {equal:?}"
    );
    assert!(
        equal.iter().all(|c| c.abs() > 1e-2),
        "a rotated face must not report a zero pitch: {equal:?}"
    );

    // Same centroids, different group sizes → identical answer.
    let lower_many = [
        lower_c,
        [lower_c[0], lower_c[1] - 7.0],
        [lower_c[0], lower_c[1] + 7.0],
    ];
    let uneven = estimate_pitch_from_vertical(&[upper_c], &lower_many, &reference)
        .expect("pitch solve should succeed");
    assert!(
        (equal[0] - uneven[0]).abs() < 1e-4 && (equal[1] - uneven[1]).abs() < 1e-4,
        "group cardinality must not change the estimate: {equal:?} vs {uneven:?}"
    );
}

#[test]
fn test_pitch_invalid_reference_errors() {
    let (upper, lower) = ([[0.0_f32, 0.0]], [[0.0_f32, 30.0]]);
    let up = Landmark3D::new(0.0, 0.06, 0.05);
    let low = Landmark3D::new(0.0, -0.06, 0.02);
    for bad in [
        PitchReference::new(up, low, 0.0),  // non-positive scale
        PitchReference::new(up, up, 250.0), // coincident centroids
    ] {
        assert!(matches!(
            estimate_pitch_from_vertical(&upper, &lower, &bad),
            Err(PoseEstimationError::InvalidConfig(_))
        ));
    }
}

#[test]
fn test_select_pitch_candidate_prefers_prior() {
    let candidates = [-0.79_f32, 0.3];
    assert!((select_pitch_candidate(candidates, 0.35) - 0.3).abs() < 1e-6);
    assert!((select_pitch_candidate(candidates, -0.9) + 0.79).abs() < 1e-6);
}

// -- rotation recovery (weak-perspective solver) --

/// Build a rotation about the model y-axis.
fn rot_y(theta: f32) -> [[f32; 3]; 3] {
    let (s, c) = theta.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

/// Non-planar model point set (full-rank scatter matrix), centred at origin.
fn spatial_model() -> [Landmark3D; 6] {
    [
        Landmark3D::new(-0.1, 0.0, 0.0),
        Landmark3D::new(0.1, 0.0, 0.0),
        Landmark3D::new(0.0, -0.1, 0.0),
        Landmark3D::new(0.0, 0.1, 0.0),
        Landmark3D::new(0.0, 0.0, 0.08),
        Landmark3D::new(0.0, 0.0, -0.08),
    ]
}

#[test]
fn test_weak_perspective_recovers_rotation_not_identity() {
    let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
    let config = PoseConfig {
        min_confidence: 0.0,
        ..Default::default()
    };
    let theta = 25.0_f32.to_radians();
    let r_true = rot_y(theta);
    let t_true = [0.0_f32, 0.0, 2.0];
    let scale = camera.focal_length / t_true[2];

    // Weak-perspective observations of the rotated model.
    let corrs: Vec<PointCorrespondence> = spatial_model()
        .iter()
        .map(|p| {
            let rp = mat3_vec3_mul(&r_true, [p.x, p.y, p.z]);
            let u = camera.cx + scale * (rp[0] + t_true[0]);
            let v = camera.cy - scale * (rp[1] + t_true[1]);
            PointCorrespondence::new(*p, Landmark2D::new(u, v))
        })
        .collect();

    let pose =
        estimate_pose_weak_perspective(&corrs, &camera, &config).expect("solve should succeed");

    for (i, row) in r_true.iter().enumerate() {
        for (j, &expected) in row.iter().enumerate() {
            assert!(
                (pose.rotation[i][j] - expected).abs() < 2e-3,
                "R[{i}][{j}] = {} expected {expected}",
                pose.rotation[i][j]
            );
        }
    }
    // The out-of-plane term must be non-zero — the old solver returned identity.
    let out_of_plane = pose.rotation[0][2].abs();
    assert!(
        out_of_plane > 0.3,
        "rotation still ~identity: {out_of_plane}"
    );
    let t_z = pose.translation[2];
    assert!((t_z - 2.0).abs() < 5e-3, "t_z={t_z}");
    // The observations are weak-perspective while the error uses the exact
    // pinhole projection, so a sub-pixel residual is expected.
    let err = pose.reprojection_error;
    assert!(err < 2.0, "reprojection_error={err}");
}

#[test]
fn test_weak_perspective_translation_with_depth_and_offset_centroid() {
    // Model centroid at (0.05, 0, 0.05) — deliberately not the origin — and
    // a true depth of 1.95 so that t_z != 1.  All points share one depth,
    // making the perspective projection exactly weak-perspective.
    let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
    let config = PoseConfig {
        min_confidence: 0.0,
        ..Default::default()
    };
    let model = [
        Landmark3D::new(-0.05, 0.0, 0.05),
        Landmark3D::new(0.15, 0.0, 0.05),
        Landmark3D::new(0.05, -0.1, 0.05),
        Landmark3D::new(0.05, 0.1, 0.05),
    ];
    let t_true = [0.4_f32, -0.2, 1.95];

    let corrs: Vec<PointCorrespondence> = model
        .iter()
        .map(|p| {
            let cam = [p.x + t_true[0], p.y + t_true[1], p.z + t_true[2]];
            let (u, v) = camera
                .project(cam[0], cam[1], cam[2])
                .expect("model point must be in front of the camera");
            PointCorrespondence::new(*p, Landmark2D::new(u, v))
        })
        .collect();

    let pose =
        estimate_pose_weak_perspective(&corrs, &camera, &config).expect("solve should succeed");

    // The pre-fix formula dropped the t_z factor and the model centroid,
    // which yielded t_x = 0.2 / t_y = -0.1 here instead of 0.4 / -0.2.
    for (axis, (&got, &want)) in pose.translation.iter().zip(t_true.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "t[{axis}]={got}, expected {want}"
        );
    }
    let err = pose.reprojection_error;
    assert!(err < 1.0, "reprojection_error={err}");
}

// -- RANSAC --

#[test]
fn test_ransac_rejects_gross_outliers() {
    let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
    let t_true = [0.0_f32, 0.0, 2.0];
    let scale = camera.focal_length / t_true[2];

    // Ten clean correspondences (identity rotation) on a non-planar ring …
    let mut corrs: Vec<PointCorrespondence> = Vec::new();
    for i in 0..10 {
        let angle = i as f32 * std::f32::consts::TAU / 10.0;
        let p = Landmark3D::new(
            0.08 * angle.cos(),
            0.08 * angle.sin(),
            0.02 * (2.0 * angle).cos(),
        );
        let u = camera.cx + scale * p.x;
        let v = camera.cy - scale * p.y;
        corrs.push(PointCorrespondence::new(p, Landmark2D::new(u, v)));
    }
    // … plus three wildly wrong observations.
    for i in 0..3 {
        let p = Landmark3D::new(0.05, -0.05 + i as f32 * 0.01, 0.0);
        corrs.push(PointCorrespondence::new(
            p,
            Landmark2D::new(40.0 + i as f32 * 12.0, 430.0),
        ));
    }

    let plain = PoseConfig {
        min_confidence: 0.0,
        ..Default::default()
    };
    let robust = PoseConfig {
        min_confidence: 0.0,
        ransac_iterations: 64,
        max_reprojection_error: 5.0,
        ..Default::default()
    };

    let plain_err = estimate_pose_weak_perspective(&corrs, &camera, &plain)
        .expect("solve should succeed")
        .reprojection_error;
    let pose_robust =
        estimate_pose_weak_perspective(&corrs, &camera, &robust).expect("solve should succeed");
    let robust_err = pose_robust.reprojection_error;
    let robust_tz = pose_robust.translation[2];

    assert!(robust_err < plain_err, "RANSAC {robust_err} vs {plain_err}");
    assert!(
        (robust_tz - t_true[2]).abs() < 0.1,
        "robust t_z={robust_tz}"
    );
    // Deterministic: the sampler is seeded, so repeated runs agree exactly.
    let again =
        estimate_pose_weak_perspective(&corrs, &camera, &robust).expect("solve should succeed");
    assert!((again.reprojection_error - robust_err).abs() < 1e-6);
}

// -- euler gimbal lock --

/// Build `R = Rz(yaw)·Ry(pitch)·Rx(roll)` (the convention `euler_angles` inverts).
fn rot_zyx(yaw: f32, pitch: f32, roll: f32) -> [[f32; 3]; 3] {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sr, cr) = roll.sin_cos();
    [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
}

/// At both gimbal-lock poles the recovered yaw must keep its sign; the
/// pre-fix code mirrored it at pitch = −π/2 (a +30° turn read as −30°).
#[test]
fn test_euler_gimbal_lock_yaw_sign_both_poles() {
    let yaw_true = 30.0_f32.to_radians();
    for &pitch_true in &[PI / 2.0, -PI / 2.0] {
        let r = rot_zyx(yaw_true, pitch_true, 0.0);
        let [yaw, pitch, roll] = HeadPose::new(r, [0.0, 0.0, 1.0]).euler_angles();
        assert!((pitch - pitch_true).abs() < 1e-3, "pitch={pitch}");
        assert!((yaw - yaw_true).abs() < 1e-3, "yaw={yaw} at {pitch_true}");
        assert!(roll.abs() < 1e-6);
    }
}

// -- PoseTracker rotation stays in SO(3) --

fn mat3_det(m: &[[f32; 3]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

#[test]
fn test_pose_tracker_smoothed_rotation_is_orthonormal() {
    let mut tracker = PoseTracker::new(0.6);
    let poses = [
        HeadPose::new(mat3_identity(), [0.0, 0.0, 2.0]),
        HeadPose::new(rot_y(40.0_f32.to_radians()), [0.1, 0.0, 2.0]),
        HeadPose::new(rot_y(-35.0_f32.to_radians()), [0.0, 0.1, 2.0]),
        HeadPose::new(rot_y(80.0_f32.to_radians()), [0.0, 0.0, 2.1]),
        // The last two are >180° apart as quaternions (dot < 0), which
        // exercises the antipodal branch of the slerp.
        HeadPose::new(rot_y(170.0_f32.to_radians()), [0.0, 0.0, 2.0]),
        HeadPose::new(rot_y(-170.0_f32.to_radians()), [0.0, 0.0, 2.0]),
    ];

    for pose in &poses {
        tracker.update(pose);
        let current = tracker.current_pose().expect("tracker has a pose");
        let r = current.rotation;
        let rrt = mat3_mul(&r, &mat3_transpose(&r));
        for (i, row) in rrt.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((value - expected).abs() < 1e-5, "RRᵀ[{i}][{j}]={value}");
            }
        }
        let det = mat3_det(&r);
        assert!((det - 1.0).abs() < 1e-5, "det(R)={det}");
        // Euler extraction must stay in range for a proper rotation.
        let [_yaw, pitch, _roll] = current.euler_angles();
        assert!(pitch.is_finite(), "pitch must be finite, got {pitch}");
    }
}

#[test]
fn test_pose_tracker_slerp_moves_toward_observation() {
    let mut tracker = PoseTracker::new(0.5);
    tracker.update(&HeadPose::new(mat3_identity(), [0.0, 0.0, 2.0]));
    tracker.update(&HeadPose::new(
        rot_y(60.0_f32.to_radians()),
        [0.0, 0.0, 2.0],
    ));

    let smoothed = tracker.current_pose().expect("tracker has a pose");
    let aa = smoothed.rotation_axis_angle();
    let magnitude = vec3_norm(aa);
    // Half-way between 0° and 60° on the quaternion sphere.
    assert!(
        (magnitude - 30.0_f32.to_radians()).abs() < 1e-3,
        "expected ~30°, got {magnitude} rad"
    );
}
