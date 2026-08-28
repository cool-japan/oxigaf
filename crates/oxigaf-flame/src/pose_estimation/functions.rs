//! Pose-estimation algorithms: the weak-perspective (+ RANSAC) solver, the
//! symmetry-based yaw heuristic, vertical-foreshortening pitch estimation,
//! and reprojection-error utilities.

use super::math::{
    mat3_pseudo_inverse, mat3_vec3_mul, orthonormalize_rotation, vec3_norm, SplitMix64, RANSAC_SEED,
};
use super::types::{
    HeadPose, PitchReference, PointCorrespondence, PoseConfig, PoseEstimationError,
    PosePinholeCamera,
};

// ---------------------------------------------------------------------------
// Pose solvers
// ---------------------------------------------------------------------------

/// Estimate head pose using a weak-perspective (scaled orthographic) solve.
///
/// This approach works well when the face occupies a moderate field of view
/// (its depth extent is small compared to its distance from the camera).
/// At least `config.min_correspondences` (≥ 4) correspondences with confidence
/// ≥ `config.min_confidence` are required.
///
/// # Algorithm
///
/// With `x̃ᵢ` the centered model points and `(ũᵢ, ṽᵢ)` the centered
/// observations, the weak-perspective projection is `ũᵢ = s·(r₁·x̃ᵢ)` and
/// `ṽᵢ = −s·(r₂·x̃ᵢ)` (image +v points down), where `r₁`, `r₂` are the first two
/// rotation rows and `s = f / t_z`.
///
/// 1. Compute the model and image centroids.
/// 2. Solve the two 3-parameter least-squares problems for `s·r₁` and `−s·r₂`
///    (pseudo-inverse of `Σ x̃ᵢ x̃ᵢᵀ`, well behaved for coplanar landmark sets).
/// 3. Recover the scale as the mean of the two row norms.
/// 4. Complete the rotation with `r₃ = r₁ × r₂` and project onto `SO(3)` via SVD.
/// 5. Derive the translation from the centroid correspondence: `t_z = f / s`,
///    `t' = ((ū − cx)·t_z/f, −(v̄ − cy)·t_z/f, t_z)` and `t = t' − R·c₃`
///    (`c₃` = model centroid), so an off-origin model centroid is handled.
/// 6. Compute the confidence-weighted reprojection error.
///
/// When `config.ransac_iterations > 0` the solve is wrapped in RANSAC: minimal
/// subsets are sampled, scored with [`count_inliers`] against
/// `config.max_reprojection_error`, and the best consensus set is refit — in
/// which case `reprojection_error` is reported over the final inlier set.
///
/// # Errors
///
/// [`PoseEstimationError::InvalidConfig`] for an invalid configuration or
/// non-positive focal length, [`PoseEstimationError::InsufficientPoints`] when
/// too few correspondences pass the confidence filter, and
/// [`PoseEstimationError::NumericalError`] for degenerate correspondence sets.
pub fn estimate_pose_weak_perspective(
    correspondences: &[PointCorrespondence],
    camera: &PosePinholeCamera,
    config: &PoseConfig,
) -> Result<HeadPose, PoseEstimationError> {
    config.validate()?;

    // Filter by minimum confidence
    let filtered: Vec<PointCorrespondence> = correspondences
        .iter()
        .copied()
        .filter(|c| c.point_2d.confidence >= config.min_confidence)
        .collect();

    let n = filtered.len();
    if n < config.min_correspondences {
        return Err(PoseEstimationError::InsufficientPoints {
            got: n,
            required: config.min_correspondences,
        });
    }

    if config.ransac_iterations > 0 {
        solve_weak_perspective_ransac(&filtered, camera, config)
    } else {
        solve_weak_perspective(&filtered, camera)
    }
}

/// Core weak-perspective solve over **all** supplied correspondences (no
/// configuration handling, no confidence filtering).
fn solve_weak_perspective(
    points: &[PointCorrespondence],
    camera: &PosePinholeCamera,
) -> Result<HeadPose, PoseEstimationError> {
    let n = points.len();
    if n < 4 {
        return Err(PoseEstimationError::InsufficientPoints {
            got: n,
            required: 4,
        });
    }
    if camera.focal_length <= 0.0 || !camera.focal_length.is_finite() {
        return Err(PoseEstimationError::InvalidConfig(format!(
            "focal_length must be > 0, got {}",
            camera.focal_length
        )));
    }

    let nf = n as f32;

    // Step 1: centroids
    let centroid_3d = [
        points.iter().map(|c| c.point_3d.x).sum::<f32>() / nf,
        points.iter().map(|c| c.point_3d.y).sum::<f32>() / nf,
        points.iter().map(|c| c.point_3d.z).sum::<f32>() / nf,
    ];
    let centroid_2d_u = points.iter().map(|c| c.point_2d.u).sum::<f32>() / nf;
    let centroid_2d_v = points.iter().map(|c| c.point_2d.v).sum::<f32>() / nf;

    // Step 2: normal equations for the two scaled rotation rows.
    let mut normal_mat = [[0.0f32; 3]; 3];
    let mut rhs_u = [0.0f32; 3];
    let mut rhs_v = [0.0f32; 3];

    for c in points {
        let x = [
            c.point_3d.x - centroid_3d[0],
            c.point_3d.y - centroid_3d[1],
            c.point_3d.z - centroid_3d[2],
        ];
        let du = c.point_2d.u - centroid_2d_u;
        let dv = c.point_2d.v - centroid_2d_v;

        for (i, row) in normal_mat.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell += x[i] * x[j];
            }
        }
        for (i, r) in rhs_u.iter_mut().enumerate() {
            *r += du * x[i];
        }
        for (i, r) in rhs_v.iter_mut().enumerate() {
            *r += dv * x[i];
        }
    }

    let pinv = mat3_pseudo_inverse(&normal_mat);
    let row_u = mat3_vec3_mul(&pinv, rhs_u); //  s · r₁
    let row_v = mat3_vec3_mul(&pinv, rhs_v); // −s · r₂

    let norm_u = vec3_norm(row_u);
    let norm_v = vec3_norm(row_v);
    if norm_u < 1e-9 || norm_v < 1e-9 {
        return Err(PoseEstimationError::NumericalError(
            "Degenerate correspondence set; weak-perspective scale cannot be estimated".to_string(),
        ));
    }

    // Step 3: scale = mean of the two scaled-row norms
    let scale = 0.5 * (norm_u + norm_v);

    // Step 4: rotation rows, completed and re-orthonormalized
    let r1 = [row_u[0] / norm_u, row_u[1] / norm_u, row_u[2] / norm_u];
    let r2 = [-row_v[0] / norm_v, -row_v[1] / norm_v, -row_v[2] / norm_v];
    let r3 = [
        r1[1] * r2[2] - r1[2] * r2[1],
        r1[2] * r2[0] - r1[0] * r2[2],
        r1[0] * r2[1] - r1[1] * r2[0],
    ];
    let rotation = orthonormalize_rotation(&[r1, r2, r3]);

    // Step 5: translation from the centroid correspondence
    let t_z = camera.focal_length / scale;
    let t_prime = [
        (centroid_2d_u - camera.cx) * t_z / camera.focal_length,
        -(centroid_2d_v - camera.cy) * t_z / camera.focal_length,
        t_z,
    ];
    let rotated_centroid = mat3_vec3_mul(&rotation, centroid_3d);
    let translation = [
        t_prime[0] - rotated_centroid[0],
        t_prime[1] - rotated_centroid[1],
        t_prime[2] - rotated_centroid[2],
    ];

    let mut pose = HeadPose::new(rotation, translation);

    // Step 6: reprojection error over the supplied correspondences
    pose.reprojection_error = reprojection_error(points, &pose, camera);

    Ok(pose)
}

/// RANSAC wrapper around [`solve_weak_perspective`]: samples
/// `config.ransac_iterations` minimal subsets, scores each with
/// [`count_inliers`] at `config.max_reprojection_error`, then refits on the best
/// consensus set.
fn solve_weak_perspective_ransac(
    points: &[PointCorrespondence],
    camera: &PosePinholeCamera,
    config: &PoseConfig,
) -> Result<HeadPose, PoseEstimationError> {
    let n = points.len();
    let sample_size = config.min_correspondences.max(4).min(n);
    if sample_size < 4 {
        return Err(PoseEstimationError::InsufficientPoints {
            got: n,
            required: 4,
        });
    }

    let mut rng = SplitMix64::new(RANSAC_SEED);
    let mut order: Vec<usize> = (0..n).collect();
    let mut best_pose: Option<HeadPose> = None;
    let mut best_inliers = 0usize;

    for _ in 0..config.ransac_iterations {
        // Partial Fisher–Yates shuffle: the first `sample_size` entries of
        // `order` become a uniform subset drawn without replacement.
        for i in 0..sample_size {
            let j = i + rng.next_index(n - i);
            order.swap(i, j);
        }
        let subset: Vec<PointCorrespondence> =
            order[..sample_size].iter().map(|&i| points[i]).collect();

        let Ok(candidate) = solve_weak_perspective(&subset, camera) else {
            continue;
        };

        let inliers = count_inliers(points, &candidate, camera, config.max_reprojection_error);
        if inliers > best_inliers {
            best_inliers = inliers;
            best_pose = Some(candidate);
        }
    }

    // No candidate could be fitted at all (every sampled subset was degenerate):
    // fall back to a plain fit over the full correspondence set.
    let Some(mut best) = best_pose else {
        return solve_weak_perspective(points, camera);
    };

    let inlier_points: Vec<PointCorrespondence> = points
        .iter()
        .copied()
        .filter(|c| is_inlier(c, &best, camera, config.max_reprojection_error))
        .collect();

    if inlier_points.len() < config.min_correspondences.max(4) {
        // Consensus set too small to refit: keep the best sampled model but
        // report its error over every correspondence.
        let error = reprojection_error(points, &best, camera);
        best.reprojection_error = error;
        return Ok(best);
    }

    solve_weak_perspective(&inlier_points, camera)
}

/// Estimate yaw angle from the horizontal asymmetry of symmetric landmark pairs.
///
/// Each element of `left_points` and `right_points` is a `[u, v]` image
/// coordinate; the arrays must have the same length and at least one element.
/// Returns the estimated yaw in radians (positive = turned right).
///
/// Standalone heuristic for callers who only have the two landmark groups —
/// [`estimate_pose_weak_perspective`] recovers yaw as part of a full rotation
/// and needs no initial guess from here.
///
/// # Errors
///
/// [`PoseEstimationError::InsufficientPoints`] if either slice is empty or the
/// slices differ in length.
pub fn estimate_yaw_from_symmetry(
    left_points: &[[f32; 2]],
    right_points: &[[f32; 2]],
) -> Result<f32, PoseEstimationError> {
    if left_points.is_empty() || right_points.is_empty() {
        return Err(PoseEstimationError::InsufficientPoints {
            got: 0,
            required: 1,
        });
    }
    if left_points.len() != right_points.len() {
        return Err(PoseEstimationError::InsufficientPoints {
            got: left_points.len().min(right_points.len()),
            required: left_points.len().max(right_points.len()),
        });
    }

    // Compare per-side spread: for a frontal face both sides have equal spread;
    // when the face turns, the far side compresses (smaller spread) relative to
    // the near side.
    //
    // left_spread  = max(left_u) - min(left_u)
    // right_spread = max(right_u) - min(right_u)
    //
    // yaw ≈ asin((left_spread - right_spread) / (left_spread + right_spread))
    // Positive yaw = face turned right (right side compressed in image).
    //
    // If both spreads are near zero (single-point inputs), we cannot determine
    // yaw from spread alone; return 0.0 (frontal assumption).
    let left_max_u = left_points
        .iter()
        .map(|p| p[0])
        .fold(f32::NEG_INFINITY, f32::max);
    let left_min_u = left_points
        .iter()
        .map(|p| p[0])
        .fold(f32::INFINITY, f32::min);
    let right_max_u = right_points
        .iter()
        .map(|p| p[0])
        .fold(f32::NEG_INFINITY, f32::max);
    let right_min_u = right_points
        .iter()
        .map(|p| p[0])
        .fold(f32::INFINITY, f32::min);

    let left_spread = left_max_u - left_min_u;
    let right_spread = right_max_u - right_min_u;

    let total_spread = left_spread + right_spread;
    if total_spread < 1e-6 {
        // Single-point pairs (or all coincident): no asymmetry observable
        return Ok(0.0);
    }

    let sin_yaw = ((left_spread - right_spread) / total_spread).clamp(-1.0, 1.0);
    let yaw = sin_yaw.asin();

    Ok(yaw)
}

/// Estimate head pitch from the vertical foreshortening of two landmark groups.
///
/// `upper_points` and `lower_points` are `[u, v]` image coordinates (v
/// increases downward); only the group **centroids** are used, so the two
/// groups may have different sizes without biasing the result.
///
/// ## Geometry
///
/// Let `Δy = upper.y − lower.y`, `Δz = upper.z − lower.z` be the model-space
/// offset between the group centroids and `θ` a rotation about the camera
/// x-axis (`y' = y·cos θ − z·sin θ`).  The observed camera-space separation is
///
/// ```text
/// Δy_cam = (v_lower − v_upper) / scale = Δy·cos θ − Δz·sin θ = A·cos(θ + φ)
/// ```
///
/// with `A = hypot(Δy, Δz)`, `φ = atan2(Δz, Δy)`, so `θ = ±acos(Δy_cam/A) − φ`.
///
/// ## Return value
///
/// The two candidate pitch angles in radians, ascending.  A single
/// foreshortening measurement genuinely cannot distinguish them (they collapse
/// to `±θ` when both groups sit at the same model depth, `Δz = 0`), so both are
/// returned rather than one being picked silently.  Use
/// [`select_pitch_candidate`] with a prior (e.g. the previous frame's pitch), or
/// [`estimate_pose_weak_perspective`] on the full landmark set for a fully
/// disambiguated rotation.
///
/// # Errors
///
/// [`PoseEstimationError::InsufficientPoints`] if either slice is empty, and
/// [`PoseEstimationError::InvalidConfig`] when the reference scale is not
/// positive and finite or the two reference centroids coincide.
pub fn estimate_pitch_from_vertical(
    upper_points: &[[f32; 2]],
    lower_points: &[[f32; 2]],
    reference: &PitchReference,
) -> Result<[f32; 2], PoseEstimationError> {
    if upper_points.is_empty() || lower_points.is_empty() {
        return Err(PoseEstimationError::InsufficientPoints {
            got: upper_points.len().min(lower_points.len()),
            required: 1,
        });
    }
    if !reference.scale.is_finite() || reference.scale <= 0.0 {
        return Err(PoseEstimationError::InvalidConfig(format!(
            "PitchReference::scale must be a positive finite value, got {}",
            reference.scale
        )));
    }

    let n_upper = upper_points.len() as f32;
    let n_lower = lower_points.len() as f32;
    let upper_v = upper_points.iter().map(|p| p[1]).sum::<f32>() / n_upper;
    let lower_v = lower_points.iter().map(|p| p[1]).sum::<f32>() / n_lower;

    // Image v grows downward while camera y grows upward, hence the flip.
    let dy_cam = (lower_v - upper_v) / reference.scale;

    let dy = reference.upper_3d.y - reference.lower_3d.y;
    let dz = reference.upper_3d.z - reference.lower_3d.z;
    let amplitude = dy.hypot(dz);
    if amplitude < 1e-6 {
        return Err(PoseEstimationError::InvalidConfig(
            "PitchReference upper_3d and lower_3d coincide; pitch is unobservable".to_string(),
        ));
    }

    let phi = dz.atan2(dy);
    let base = (dy_cam / amplitude).clamp(-1.0, 1.0).acos();

    let first = base - phi;
    let second = -base - phi;
    Ok(if first <= second {
        [first, second]
    } else {
        [second, first]
    })
}

/// Pick the pitch candidate closest to `prior` (radians).
///
/// Companion to [`estimate_pitch_from_vertical`]: pass the previous frame's
/// pitch when tracking, or `0.0` to prefer the solution nearest frontal.
#[must_use]
pub fn select_pitch_candidate(candidates: [f32; 2], prior: f32) -> f32 {
    if (candidates[0] - prior).abs() <= (candidates[1] - prior).abs() {
        candidates[0]
    } else {
        candidates[1]
    }
}

// ---------------------------------------------------------------------------
// Reprojection utilities
// ---------------------------------------------------------------------------

/// Compute the confidence-weighted mean reprojection error (in pixels) for a
/// set of correspondences under the given pose and camera.
///
/// Returns `0.0` if `correspondences` is empty or no points project forward.
#[must_use]
pub fn reprojection_error(
    correspondences: &[PointCorrespondence],
    pose: &HeadPose,
    camera: &PosePinholeCamera,
) -> f32 {
    if correspondences.is_empty() {
        return 0.0;
    }

    let mut total_error = 0.0_f32;
    let mut total_weight = 0.0_f32;

    for c in correspondences {
        if let Some((proj_u, proj_v)) = pose.project_point(c.point_3d, camera) {
            let du = proj_u - c.point_2d.u;
            let dv = proj_v - c.point_2d.v;
            let dist = (du * du + dv * dv).sqrt();
            let w = c.point_2d.confidence;
            total_error += dist * w;
            total_weight += w;
        }
    }

    if total_weight < 1e-9 {
        return 0.0;
    }

    total_error / total_weight
}

/// Return `true` when the correspondence reprojects within `threshold` pixels
/// (points that do not project in front of the camera are never inliers).
fn is_inlier(
    correspondence: &PointCorrespondence,
    pose: &HeadPose,
    camera: &PosePinholeCamera,
    threshold: f32,
) -> bool {
    pose.project_point(correspondence.point_3d, camera)
        .is_some_and(|(proj_u, proj_v)| {
            let du = proj_u - correspondence.point_2d.u;
            let dv = proj_v - correspondence.point_2d.v;
            (du * du + dv * dv).sqrt() < threshold
        })
}

/// Count the number of correspondences whose reprojection error is below
/// `threshold` (in pixels).
///
/// Used by [`estimate_pose_weak_perspective`] to score RANSAC hypotheses, and
/// available to callers running their own robust loop.
#[must_use]
pub fn count_inliers(
    correspondences: &[PointCorrespondence],
    pose: &HeadPose,
    camera: &PosePinholeCamera,
    threshold: f32,
) -> usize {
    correspondences
        .iter()
        .filter(|c| is_inlier(c, pose, camera, threshold))
        .count()
}
