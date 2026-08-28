//! The ICP loop itself: the closed-form per-step estimator, one iteration, the
//! driver that runs iterations to convergence, and the transform/reporting
//! helpers that surround them.

use super::correspondence::{
    filter_correspondences, find_correspondences_with_tree, validate_cloud_pair,
};
use super::kdtree::KdTree;
use super::math::{
    largest_eigenvector_sym4, mat3_vec, quat_normalize, quat_to_mat3, vec3_dot, vec3_len,
    vec3_scale, vec3_sub,
};
use super::types::{
    RegistrationConfig, RegistrationError, RegistrationResult, RegistrationStats,
    RegistrationTransform,
};

/// Estimate the best-fit transform aligning `source_pts` to `target_pts` with
/// the closed-form Umeyama (1991) solution.
///
/// `source_pts` and `target_pts` are flat `[x0,y0,z0, x1,y1,z1, ...]` arrays
/// of the same length (already matched by correspondence).
///
/// Both clouds are centred, the 3×3 cross-covariance `Σ = Σᵢ xsᵢ · xtᵢᵀ` is
/// accumulated, and the rotation is read off the eigenvector belonging to the
/// largest eigenvalue of Horn's symmetric 4×4 quaternion matrix built from `Σ`.
/// That maximises `Σᵢ ⟨R·xsᵢ, xtᵢ⟩` over *proper* rotations, so the result is
/// always orthonormal with `det(R) = +1`: the same answer as the SVD form with
/// its `diag(1, 1, det(U·Vᵀ))` reflection correction, without needing an SVD.
///
/// When `allow_scale` is `true` the optimal uniform scale
/// `s = Σᵢ ⟨R·xsᵢ, xtᵢ⟩ / Σᵢ ⟨xsᵢ, xsᵢ⟩` is estimated as well; when it is
/// `false` scale is fixed at 1.0 and no scale term is computed at all.
///
/// # Errors
///
/// Returns [`RegistrationError::SizeMismatch`] when the two arrays differ in
/// length and [`RegistrationError::InvalidPositionLength`] when that length is
/// not a multiple of 3. Empty (but valid) input yields the identity transform.
pub fn estimate_transform_umeyama(
    source_pts: &[f32],
    target_pts: &[f32],
    allow_scale: bool,
) -> Result<RegistrationTransform, RegistrationError> {
    if source_pts.len() != target_pts.len() {
        return Err(RegistrationError::SizeMismatch {
            src: source_pts.len(),
            tgt: target_pts.len(),
        });
    }
    if !source_pts.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength {
            len: source_pts.len(),
        });
    }
    let n = source_pts.len() / 3;
    if n == 0 {
        return Ok(RegistrationTransform::identity());
    }

    // Centroids of both clouds.
    let mut cs = [0.0f32; 3];
    let mut ct = [0.0f32; 3];
    for (s, t) in source_pts.chunks_exact(3).zip(target_pts.chunks_exact(3)) {
        cs = [cs[0] + s[0], cs[1] + s[1], cs[2] + s[2]];
        ct = [ct[0] + t[0], ct[1] + t[1], ct[2] + t[2]];
    }
    let inv_n = 1.0 / n as f32;
    let cs = vec3_scale(cs, inv_n);
    let ct = vec3_scale(ct, inv_n);

    // Cross-covariance sigma[a * 3 + b] = Σᵢ (xsᵢ)ₐ · (xtᵢ)_b, plus the source
    // variance the optimal scale is normalised by.
    let mut sigma = [0.0f32; 9];
    let mut var_src = 0.0f32;
    for (s, t) in source_pts.chunks_exact(3).zip(target_pts.chunks_exact(3)) {
        let xs = vec3_sub([s[0], s[1], s[2]], cs);
        let xt = vec3_sub([t[0], t[1], t[2]], ct);
        var_src += vec3_dot(xs, xs);
        for (a, &xa) in xs.iter().enumerate() {
            sigma[a * 3] += xa * xt[0];
            sigma[a * 3 + 1] += xa * xt[1];
            sigma[a * 3 + 2] += xa * xt[2];
        }
    }

    // Horn's symmetric 4×4 matrix N, whose quadratic form qᵀ·N·q equals
    // Σᵢ ⟨R(q)·xsᵢ, xtᵢ⟩ for unit q ordered as (w, x, y, z).
    let [sxx, sxy, sxz, syx, syy, syz, szx, szy, szz] = sigma;
    let trace = sxx + syy + szz;
    let n_mat = [
        trace,
        syz - szy,
        szx - sxz,
        sxy - syx,
        syz - szy,
        sxx - syy - szz,
        sxy + syx,
        szx + sxz,
        szx - sxz,
        sxy + syx,
        -sxx + syy - szz,
        syz + szy,
        sxy - syx,
        szx + sxz,
        syz + szy,
        -sxx - syy + szz,
    ];
    let q = largest_eigenvector_sym4(&n_mat);
    let (qx, qy, qz, qw) = quat_normalize(q[1], q[2], q[3], q[0]);
    let rotation = quat_to_mat3(qx, qy, qz, qw);

    let scale = if allow_scale && var_src > 0.0 {
        // Σᵢ ⟨R·xsᵢ, xtᵢ⟩ = trace(R · Σ).
        let mut num = 0.0f32;
        for a in 0..3 {
            for b in 0..3 {
                num += rotation[a * 3 + b] * sigma[b * 3 + a];
            }
        }
        (num / var_src).max(1e-6)
    } else {
        1.0
    };

    // translation = ct - scale * R * cs
    let translation = vec3_sub(ct, vec3_scale(mat3_vec(rotation, cs), scale));

    Ok(RegistrationTransform {
        rotation,
        translation,
        scale,
    })
}

/// Reject a scale factor that would silently turn a cloud into garbage.
///
/// Shared by [`apply_registration_transform`] and the pre-flight check in
/// [`icp_step`] so the two can never drift apart.
fn validate_scale(scale: f32) -> Result<(), RegistrationError> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(RegistrationError::InvalidScale { scale });
    }
    Ok(())
}

/// Reject inputs that [`icp_step_with_tree`] would only fail on deeper in the
/// call chain, reporting them in exactly that chain's order: the source cloud
/// and the transform applied to it first, then the target cloud.
///
/// Running this before the target's k-d tree is built keeps `icp_step` from
/// paying `O(n_tgt log n_tgt)` for a call that cannot succeed.
fn validate_icp_inputs(
    source: &[f32],
    target: &[f32],
    transform: &RegistrationTransform,
) -> Result<(), RegistrationError> {
    if source.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !source.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength { len: source.len() });
    }
    validate_scale(transform.scale)?;
    if target.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !target.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength { len: target.len() });
    }
    Ok(())
}

/// Perform one ICP iteration:
/// 1. Transform source by the current transform.
/// 2. Find correspondences between transformed source and target.
/// 3. Filter outlier correspondences.
/// 4. Estimate a delta transform from the matched pairs.
/// 5. Compose the delta with the current transform.
/// 6. Compute and return the RMSE of the final correspondences.
///
/// Returns `(new_transform, rmse, n_correspondences)`.
///
/// A k-d tree over `target` is built for this single call. Iterating against a
/// fixed target — which is what [`register_point_clouds`] does — reuses one
/// tree across every iteration instead, so prefer that driver over calling this
/// in a loop of your own.
pub fn icp_step(
    source: &[f32],
    target: &[f32],
    transform: RegistrationTransform,
    config: &RegistrationConfig,
) -> Result<(RegistrationTransform, f32, usize), RegistrationError> {
    validate_icp_inputs(source, target, &transform)?;
    let tree = KdTree::build_all(target);
    icp_step_with_tree(source, target, &tree, transform, config)
}

/// [`icp_step`] against a k-d tree that already exists.
///
/// # Invariant
///
/// `tree` must have been built over exactly the `target` slice passed here
/// (`KdTree::build_all(target)`). The tree holds indices into that slice and
/// the type system cannot tie the two together, so pairing a tree with another
/// cloud silently matches against the cloud it was built from.
fn icp_step_with_tree(
    source: &[f32],
    target: &[f32],
    tree: &KdTree,
    transform: RegistrationTransform,
    config: &RegistrationConfig,
) -> Result<(RegistrationTransform, f32, usize), RegistrationError> {
    // Apply current transform to source
    let transformed = apply_registration_transform(source, &transform)?;

    // Find correspondences against the pre-built target index
    let corr_raw =
        find_correspondences_with_tree(&transformed, target, tree, config.max_correspondence_dist)?;
    let corr = filter_correspondences(corr_raw, config.outlier_fraction);

    if corr.is_empty() {
        // No correspondences found — return unchanged
        return Ok((transform, f32::MAX, 0));
    }

    // Extract matched point pairs (flat arrays)
    let mut src_pts = Vec::with_capacity(corr.len() * 3);
    let mut tgt_pts = Vec::with_capacity(corr.len() * 3);
    for c in &corr {
        let si = c.source_idx;
        let ti = c.target_idx;
        src_pts.push(transformed[si * 3]);
        src_pts.push(transformed[si * 3 + 1]);
        src_pts.push(transformed[si * 3 + 2]);
        tgt_pts.push(target[ti * 3]);
        tgt_pts.push(target[ti * 3 + 1]);
        tgt_pts.push(target[ti * 3 + 2]);
    }

    // Estimate the delta transform on the matched subsets: the exact closed
    // form, so no inner iteration is needed here.
    let delta = estimate_transform_umeyama(&src_pts, &tgt_pts, config.allow_scale)?;

    // Compose: apply delta on top of the existing transform
    let new_transform = delta.compose(&transform);

    // Compute RMSE: re-apply new transform to source and measure against target
    let recheck = apply_registration_transform(source, &new_transform)?;
    let mut sq_sum = 0.0f32;
    for c in &corr {
        let si = c.source_idx;
        let ti = c.target_idx;
        let rp = [recheck[si * 3], recheck[si * 3 + 1], recheck[si * 3 + 2]];
        let tp = [target[ti * 3], target[ti * 3 + 1], target[ti * 3 + 2]];
        let diff = vec3_sub(rp, tp);
        sq_sum += vec3_dot(diff, diff);
    }
    let rmse = (sq_sum / corr.len() as f32).sqrt();

    Ok((new_transform, rmse, corr.len()))
}

/// Register source point cloud to target using iterative closest point (ICP).
///
/// Returns the best transform together with convergence information.
///
/// The target never changes across iterations, so its k-d tree — a pure
/// function of the target cloud — is built once here and reused by every
/// iteration rather than rebuilt inside each step.
pub fn register_point_clouds(
    source: &[f32],
    target: &[f32],
    config: &RegistrationConfig,
) -> Result<RegistrationResult, RegistrationError> {
    // Validate inputs
    validate_cloud_pair(source, target)?;
    let n_src = source.len() / 3;
    if n_src < 3 {
        return Err(RegistrationError::InsufficientPoints {
            need: 3,
            got: n_src,
        });
    }

    // Subsample source if requested
    let working_source: Vec<f32> = if config.subsample_rate > 1 {
        subsample_positions(source, config.subsample_rate)?
    } else {
        source.to_vec()
    };

    // A run capped at zero iterations reports the identity without ever
    // querying the target, so it must not pay for an index either.
    if config.max_iterations == 0 {
        return Ok(RegistrationResult {
            transform: RegistrationTransform::identity(),
            final_rmse: f32::MAX,
            n_iterations: 0,
            converged: false,
            n_correspondences: 0,
            rmse_history: Vec::new(),
        });
    }

    // One index over the target, shared by every iteration below.
    let tree = KdTree::build_all(target);

    let mut transform = RegistrationTransform::identity();
    let mut prev_rmse = f32::MAX;
    let mut converged = false;
    let mut n_correspondences = 0usize;
    let mut rmse_history: Vec<f32> = Vec::with_capacity(config.max_iterations);

    let mut iter = 0usize;
    loop {
        if iter >= config.max_iterations {
            break;
        }

        let (new_transform, rmse, n_corr) =
            icp_step_with_tree(&working_source, target, &tree, transform, config)?;
        transform = new_transform;
        n_correspondences = n_corr;
        rmse_history.push(rmse);

        let delta = (prev_rmse - rmse).abs();
        if delta < config.tolerance && rmse < f32::MAX {
            converged = true;
            prev_rmse = rmse;
            iter += 1;
            break;
        }
        prev_rmse = rmse;
        iter += 1;
    }

    let final_rmse = prev_rmse;
    // `f32::MAX` is the sentinel `icp_step` returns when it could not match a
    // single pair, so a run that never improved on it produced nothing usable.
    if config.max_iterations > 0 && (final_rmse.is_nan() || final_rmse >= f32::MAX) {
        return Err(RegistrationError::Diverged {
            iters: iter,
            rmse: final_rmse,
        });
    }

    Ok(RegistrationResult {
        transform,
        final_rmse,
        n_iterations: iter,
        converged,
        n_correspondences,
        rmse_history,
    })
}

/// Apply a registration transform to every point in a flat positions array.
///
/// Returns a new flat positions array of the same length.
///
/// # Errors
///
/// Fails on an empty or mis-sized array, and on a transform whose scale is not
/// finite and positive (which would silently turn the cloud into garbage).
pub fn apply_registration_transform(
    positions: &[f32],
    transform: &RegistrationTransform,
) -> Result<Vec<f32>, RegistrationError> {
    if positions.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !positions.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength {
            len: positions.len(),
        });
    }
    validate_scale(transform.scale)?;
    let mut out = Vec::with_capacity(positions.len());
    for chunk in positions.chunks_exact(3) {
        let p = [chunk[0], chunk[1], chunk[2]];
        let q = transform.apply(p);
        out.push(q[0]);
        out.push(q[1]);
        out.push(q[2]);
    }
    Ok(out)
}

/// Compute summary statistics from a completed registration result.
///
/// `initial_rmse` should be the RMSE before any iterations were applied.
pub fn compute_registration_stats(
    result: &RegistrationResult,
    initial_rmse: f32,
) -> RegistrationStats {
    let final_rmse = result.final_rmse;
    let improvement_factor = if final_rmse > 0.0 && final_rmse < f32::MAX {
        initial_rmse / final_rmse
    } else {
        1.0
    };

    // Extract rotation angle from trace: angle = arccos((trace(R)-1)/2)
    let r = &result.transform.rotation;
    let trace = r[0] + r[4] + r[8];
    let cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    let rotation_angle_deg = cos_angle.acos() * (180.0 / std::f32::consts::PI);

    let transform_magnitude = vec3_len(result.transform.translation);
    let scale_change = (result.transform.scale - 1.0).abs();

    RegistrationStats {
        initial_rmse,
        final_rmse,
        improvement_factor,
        transform_magnitude,
        rotation_angle_deg,
        scale_change,
    }
}

/// Format a registration result and stats as a human-readable summary string.
pub fn format_registration_result(
    result: &RegistrationResult,
    stats: &RegistrationStats,
) -> String {
    format!(
        "ICP: converged={}, iter={}, RMSE: {:.4e} -> {:.4e} (x{:.2}), rot={:.2}°, t={:.4}m",
        result.converged,
        result.n_iterations,
        stats.initial_rmse,
        stats.final_rmse,
        stats.improvement_factor,
        stats.rotation_angle_deg,
        stats.transform_magnitude,
    )
}

/// Extract every `stride`-th point from a flat positions array.
///
/// `stride=1` returns all points; `stride=2` returns every other point, etc.
pub fn subsample_positions(
    positions: &[f32],
    stride: usize,
) -> Result<Vec<f32>, RegistrationError> {
    if positions.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !positions.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength {
            len: positions.len(),
        });
    }
    let stride = stride.max(1);
    let n = positions.len() / 3;
    let out_n = n.div_ceil(stride);
    let mut out = Vec::with_capacity(out_n * 3);
    for i in (0..n).step_by(stride) {
        out.push(positions[i * 3]);
        out.push(positions[i * 3 + 1]);
        out.push(positions[i * 3 + 2]);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud_registration::compute_centroid_3d;
    use crate::cloud_registration::test_support::{
        approx_eq, close3, grid_positions, mat3_det, pseudo_cloud,
    };

    /// Bit-exact comparison of two transforms — used where two code paths must
    /// perform the identical sequence of floating-point operations.
    fn same_transform_bits(a: &RegistrationTransform, b: &RegistrationTransform) -> bool {
        a.rotation
            .iter()
            .zip(b.rotation.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
            && a.translation
                .iter()
                .zip(b.translation.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
            && a.scale.to_bits() == b.scale.to_bits()
    }

    /// Target cloud produced by rotating `src` about its own centre and
    /// translating it, so ICP has a real transform to recover.
    fn rotated_target(src: &[f32], angle_deg: f32, shift: [f32; 3]) -> Vec<f32> {
        let angle = angle_deg.to_radians();
        let (sa, ca) = (angle.sin(), angle.cos());
        let rot = [ca, -sa, 0.0, sa, ca, 0.0, 0.0, 0.0, 1.0];
        let centre = compute_centroid_3d(src).unwrap_or([0.0; 3]);
        let mut tgt = Vec::with_capacity(src.len());
        for p in src.chunks_exact(3) {
            let q = mat3_vec(rot, vec3_sub([p[0], p[1], p[2]], centre));
            tgt.extend_from_slice(&[
                q[0] + centre[0] + shift[0],
                q[1] + centre[1] + shift[1],
                q[2] + centre[2] + shift[2],
            ]);
        }
        tgt
    }

    // -----------------------------------------------------------------------
    // estimate_transform_umeyama
    // -----------------------------------------------------------------------

    #[test]
    fn test_umeyama_same_cloud_identity() {
        let pts = vec![
            0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let t = estimate_transform_umeyama(&pts, &pts, false).unwrap();
        // The closed form is exact here: every point maps onto itself.
        for chunk in pts.chunks_exact(3) {
            let p_in = [chunk[0], chunk[1], chunk[2]];
            assert!(close3(t.apply(p_in), p_in), "point moved: {:?}", p_in);
        }
    }

    #[test]
    fn test_umeyama_pure_translation() {
        let src = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let tgt: Vec<f32> = src
            .iter()
            .enumerate()
            .map(|(i, &v)| if i.is_multiple_of(3) { v + 5.0 } else { v })
            .collect();
        let t = estimate_transform_umeyama(&src, &tgt, false).unwrap();
        // Exactly (5, 0, 0) with a proper identity rotation.
        let tr = t.translation;
        assert!(close3(tr, [5.0, 0.0, 0.0]), "{:?}", tr);
        assert!(approx_eq(mat3_det(t.rotation), 1.0, 1e-5));
    }

    #[test]
    fn test_umeyama_empty_returns_identity() {
        let t = estimate_transform_umeyama(&[], &[], false).unwrap();
        assert!(approx_eq(t.scale, 1.0, 1e-6));
        assert!(close3(t.translation, [0.0; 3]));
    }

    #[test]
    fn test_umeyama_rejects_mismatched_inputs() {
        let six = vec![0.0f32; 6];
        let three = vec![0.0f32; 3];
        assert!(matches!(
            estimate_transform_umeyama(&six, &three, false),
            Err(RegistrationError::SizeMismatch { src: 6, tgt: 3 })
        ));
        let four = vec![0.0f32; 4];
        assert!(matches!(
            estimate_transform_umeyama(&four, &four, false),
            Err(RegistrationError::InvalidPositionLength { len: 4 })
        ));
    }

    #[test]
    fn test_umeyama_recovers_known_rigid_transform() {
        // 30° about the normalised axis (1, 2, 3), plus a translation.
        let inv = 1.0 / 14.0f32.sqrt();
        let half = 15.0f32.to_radians();
        let (sn, cs) = (half.sin(), half.cos());
        let r_known = quat_to_mat3(inv * sn, 2.0 * inv * sn, 3.0 * inv * sn, cs);
        let t_known = [0.7f32, -1.3, 2.1];
        let src = grid_positions(64, 1.0);
        let mut tgt = Vec::with_capacity(src.len());
        for p in src.chunks_exact(3) {
            let q = mat3_vec(r_known, [p[0], p[1], p[2]]);
            tgt.extend_from_slice(&[q[0] + t_known[0], q[1] + t_known[1], q[2] + t_known[2]]);
        }
        let est = estimate_transform_umeyama(&src, &tgt, false).unwrap();
        for (i, (got, want)) in est.rotation.iter().zip(r_known.iter()).enumerate() {
            assert!(
                approx_eq(*got, *want, 1e-3),
                "R[{}]: {} vs {}",
                i,
                got,
                want
            );
        }
        for (k, (got, want)) in est.translation.iter().zip(t_known.iter()).enumerate() {
            assert!(
                approx_eq(*got, *want, 1e-3),
                "t[{}]: {} vs {}",
                k,
                got,
                want
            );
        }
        assert!(approx_eq(est.scale, 1.0, 1e-6));
        // A proper rotation, never a reflection.
        assert!(approx_eq(mat3_det(est.rotation), 1.0, 1e-4));
    }

    #[test]
    fn test_umeyama_scale_allowed() {
        let src = vec![
            0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ];
        // Scale target by 2.0
        let tgt: Vec<f32> = src.iter().map(|&v| v * 2.0).collect();
        let t = estimate_transform_umeyama(&src, &tgt, true).unwrap();
        // The closed form recovers the factor exactly.
        let sc = t.scale;
        assert!(approx_eq(sc, 2.0, 1e-4), "expected 2.0, got {}", sc);
    }

    // -----------------------------------------------------------------------
    // icp_step
    // -----------------------------------------------------------------------

    #[test]
    fn test_icp_step_same_cloud() {
        let pts = grid_positions(8, 1.0);
        let cfg = RegistrationConfig::default();
        let id = RegistrationTransform::identity();
        let (new_t, rmse, n) = icp_step(&pts, &pts, id, &cfg).unwrap();
        assert!(
            rmse < 1e-5,
            "RMSE on identical clouds should vanish, got {}",
            rmse
        );
        assert!(n > 0);
        // Scale should remain close to 1.0
        assert!(approx_eq(new_t.scale, 1.0, 0.1));
    }

    #[test]
    fn test_icp_step_returns_correspondences() {
        let src = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let tgt = vec![0.1f32, 0.0, 0.0, 1.1, 0.0, 0.0, 0.1, 1.0, 0.0];
        let cfg = RegistrationConfig::default();
        let id = RegistrationTransform::identity();
        let (_t, _rmse, n) = icp_step(&src, &tgt, id, &cfg).unwrap();
        assert_eq!(n, 3);
    }

    #[test]
    fn test_icp_step_reports_source_before_target() {
        // The pre-flight check must not reorder the errors the inner path
        // reports: a malformed source wins over a malformed target.
        let cfg = RegistrationConfig::default();
        let id = RegistrationTransform::identity();
        assert!(matches!(
            icp_step(&[1.0, 2.0], &[], id, &cfg),
            Err(RegistrationError::InvalidPositionLength { len: 2 })
        ));
        assert!(matches!(
            icp_step(&[1.0, 2.0, 3.0], &[1.0, 2.0], id, &cfg),
            Err(RegistrationError::InvalidPositionLength { len: 2 })
        ));
        let mut bad_scale = RegistrationTransform::identity();
        bad_scale.scale = -1.0;
        assert!(matches!(
            icp_step(&[1.0, 2.0, 3.0], &[], bad_scale, &cfg),
            Err(RegistrationError::InvalidScale { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // icp_step_with_tree — the hoisted index must change nothing
    // -----------------------------------------------------------------------

    #[test]
    fn test_icp_step_with_tree_matches_public_step() {
        // Same inputs, same tree contents: the two entry points must agree
        // bit for bit, or hoisting the tree out of the loop would alter the
        // trajectory ICP takes.
        let src = pseudo_cloud(300, 2_027, 6.0);
        let tgt = rotated_target(&src, 7.0, [0.2, -0.1, 0.05]);
        let cfg = RegistrationConfig {
            max_iterations: 4,
            tolerance: 1e-6,
            max_correspondence_dist: 5.0,
            allow_scale: true,
            subsample_rate: 1,
            outlier_fraction: 0.1,
        };
        let start = RegistrationTransform {
            rotation: RegistrationTransform::identity().rotation,
            translation: [0.05, 0.0, -0.02],
            scale: 1.01,
        };
        let tree = KdTree::build_all(&tgt);
        let (t_pub, rmse_pub, n_pub) = icp_step(&src, &tgt, start, &cfg).unwrap();
        let (t_tree, rmse_tree, n_tree) =
            icp_step_with_tree(&src, &tgt, &tree, start, &cfg).unwrap();
        assert_eq!(n_pub, n_tree);
        assert_eq!(rmse_pub.to_bits(), rmse_tree.to_bits());
        assert!(
            same_transform_bits(&t_pub, &t_tree),
            "{:?} vs {:?}",
            t_pub,
            t_tree
        );
    }

    #[test]
    fn test_register_matches_manual_public_icp_loop() {
        // `register_point_clouds` builds the target tree once and reuses it.
        // Driving the public, tree-per-call `icp_step` by hand with the same
        // control flow must reproduce the run exactly.
        let src = pseudo_cloud(400, 55, 8.0);
        let tgt = rotated_target(&src, 4.0, [0.15, 0.1, -0.05]);
        let cfg = RegistrationConfig {
            max_iterations: 6,
            tolerance: 1e-9,
            ..Default::default()
        };
        let result = register_point_clouds(&src, &tgt, &cfg).unwrap();

        let mut transform = RegistrationTransform::identity();
        let mut prev_rmse = f32::MAX;
        let mut history: Vec<f32> = Vec::new();
        let mut iter = 0usize;
        loop {
            if iter >= cfg.max_iterations {
                break;
            }
            let (new_transform, rmse, _n) = icp_step(&src, &tgt, transform, &cfg).unwrap();
            transform = new_transform;
            history.push(rmse);
            let delta = (prev_rmse - rmse).abs();
            prev_rmse = rmse;
            iter += 1;
            if delta < cfg.tolerance && rmse < f32::MAX {
                break;
            }
        }

        assert_eq!(result.n_iterations, iter);
        assert_eq!(result.rmse_history.len(), history.len());
        for (a, b) in result.rmse_history.iter().zip(history.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(result.final_rmse.to_bits(), prev_rmse.to_bits());
        assert!(
            same_transform_bits(&result.transform, &transform),
            "{:?} vs {:?}",
            result.transform,
            transform
        );
    }

    // -----------------------------------------------------------------------
    // register_point_clouds
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_same_cloud() {
        let pts = grid_positions(27, 1.0);
        let cfg = RegistrationConfig {
            max_iterations: 10,
            tolerance: 1e-3,
            ..Default::default()
        };
        let result = register_point_clouds(&pts, &pts, &cfg).unwrap();
        // The closed-form estimator returns the identity on the first pass, so
        // the residual collapses immediately instead of creeping down.
        assert!(
            result.final_rmse < 1e-4,
            "RMSE should vanish for same cloud: {}",
            result.final_rmse
        );
    }

    #[test]
    fn test_register_recovers_small_rigid_transform() {
        // 5° about z through the cloud centre plus a sub-cell translation moves
        // every point by at most 2 * 2.83 * sin(2.5°) + 0.13 ≈ 0.37, well under
        // half the 1.0 grid spacing, so the very first correspondence set is the
        // true one and ICP must land on the exact transform.
        let src = grid_positions(125, 1.0);
        let angle = 5.0f32.to_radians();
        let (sa, ca) = (angle.sin(), angle.cos());
        let r_known = [ca, -sa, 0.0, sa, ca, 0.0, 0.0, 0.0, 1.0];
        let t_known = [0.1f32, -0.05, 0.05];
        let tgt = rotated_target(&src, 5.0, t_known);
        let cfg = RegistrationConfig {
            max_iterations: 20,
            tolerance: 1e-6,
            ..Default::default()
        };
        let result = register_point_clouds(&src, &tgt, &cfg).unwrap();
        assert!(
            result.final_rmse < 1e-3,
            "RMSE should collapse, got {}",
            result.final_rmse
        );
        for (i, (got, want)) in result
            .transform
            .rotation
            .iter()
            .zip(r_known.iter())
            .enumerate()
        {
            assert!(
                approx_eq(*got, *want, 1e-3),
                "R[{}]: {} vs {}",
                i,
                got,
                want
            );
        }
        // Orthonormal, proper rotation (composed over every iteration).
        assert!(approx_eq(mat3_det(result.transform.rotation), 1.0, 1e-4));
    }

    #[test]
    fn test_register_diverges_without_correspondences() {
        let src = grid_positions(8, 1.0);
        let tgt: Vec<f32> = src
            .iter()
            .enumerate()
            .map(|(i, &v)| if i.is_multiple_of(3) { v + 100.0 } else { v })
            .collect();
        let cfg = RegistrationConfig {
            max_iterations: 3,
            max_correspondence_dist: 1.0,
            ..Default::default()
        };
        let result = register_point_clouds(&src, &tgt, &cfg);
        assert!(matches!(result, Err(RegistrationError::Diverged { .. })));
    }

    #[test]
    fn test_register_translated_cloud() {
        let src: Vec<f32> = grid_positions(8, 1.0);
        // Translate target by (3, 0, 0)
        let tgt: Vec<f32> = src
            .iter()
            .enumerate()
            .map(|(i, &v)| if i.is_multiple_of(3) { v + 3.0 } else { v })
            .collect();
        let cfg = RegistrationConfig {
            max_iterations: 30,
            tolerance: 1e-4,
            ..Default::default()
        };
        let result = register_point_clouds(&src, &tgt, &cfg).unwrap();
        // Translation should be approximately (3, 0, 0)
        assert!(
            result.transform.translation[0] > 1.5,
            "expected tx > 1.5, got {}",
            result.transform.translation[0]
        );
    }

    #[test]
    fn test_register_empty_source_error() {
        let result = register_point_clouds(&[], &[0.0, 0.0, 0.0], &RegistrationConfig::default());
        assert!(matches!(result, Err(RegistrationError::EmptyCloud)));
    }

    #[test]
    fn test_register_invalid_length_error() {
        let result = register_point_clouds(
            &[1.0, 2.0],
            &[1.0, 2.0, 3.0],
            &RegistrationConfig::default(),
        );
        assert!(matches!(
            result,
            Err(RegistrationError::InvalidPositionLength { len: 2 })
        ));
    }

    #[test]
    fn test_register_insufficient_points() {
        let result = register_point_clouds(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            &RegistrationConfig::default(),
        );
        assert!(matches!(
            result,
            Err(RegistrationError::InsufficientPoints { .. })
        ));
    }

    #[test]
    fn test_register_result_fields() {
        let pts = grid_positions(27, 1.0);
        let cfg = RegistrationConfig {
            max_iterations: 5,
            ..Default::default()
        };
        let result = register_point_clouds(&pts, &pts, &cfg).unwrap();
        assert!(result.n_iterations <= 5);
        assert!(!result.rmse_history.is_empty());
    }

    #[test]
    fn test_register_with_subsample() {
        let pts = grid_positions(27, 1.0);
        let cfg = RegistrationConfig {
            max_iterations: 5,
            subsample_rate: 3,
            ..Default::default()
        };
        let result = register_point_clouds(&pts, &pts, &cfg).unwrap();
        assert!(result.final_rmse < 1.0);
    }

    #[test]
    fn test_register_with_outlier_fraction() {
        let pts = grid_positions(27, 1.0);
        let cfg = RegistrationConfig {
            max_iterations: 5,
            outlier_fraction: 0.2,
            ..Default::default()
        };
        let result = register_point_clouds(&pts, &pts, &cfg).unwrap();
        assert!(result.final_rmse < 1.0);
    }

    #[test]
    fn test_register_zero_iterations_is_a_no_op() {
        // The early-out that skips building the target index must report the
        // same result the loop-that-never-runs did: identity, no history, and
        // the `f32::MAX` RMSE sentinel without a `Diverged` error.
        let pts = grid_positions(27, 1.0);
        // Both with and without subsampling: the early-out sits behind the
        // subsample step, so that step still runs (and can still fail) first.
        for subsample_rate in [1usize, 3] {
            let cfg = RegistrationConfig {
                max_iterations: 0,
                subsample_rate,
                ..Default::default()
            };
            let result = register_point_clouds(&pts, &pts, &cfg).unwrap();
            assert_eq!(result.n_iterations, 0);
            assert!(!result.converged);
            assert_eq!(result.n_correspondences, 0);
            assert!(result.rmse_history.is_empty());
            assert_eq!(result.final_rmse.to_bits(), f32::MAX.to_bits());
            assert!(same_transform_bits(
                &result.transform,
                &RegistrationTransform::identity()
            ));
            // Validation still runs ahead of the early-out.
            assert!(matches!(
                register_point_clouds(&[1.0, 2.0], &pts, &cfg),
                Err(RegistrationError::InvalidPositionLength { len: 2 })
            ));
            assert!(matches!(
                register_point_clouds(&[0.0, 0.0, 0.0], &pts, &cfg),
                Err(RegistrationError::InsufficientPoints { need: 3, got: 1 })
            ));
        }
    }

    #[test]
    fn test_register_convergence_flag() {
        let pts = grid_positions(27, 1.0);
        let cfg = RegistrationConfig {
            max_iterations: 50,
            tolerance: 1e-2,
            ..Default::default()
        };
        let result = register_point_clouds(&pts, &pts, &cfg).unwrap();
        // Same cloud should converge
        assert!(result.converged || result.n_iterations == 50);
    }

    // -----------------------------------------------------------------------
    // apply_registration_transform
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_transform_correct_length() {
        let pts = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = RegistrationTransform::identity();
        let out = apply_registration_transform(&pts, &t).unwrap();
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn test_apply_transform_identity_unchanged() {
        let pts = vec![1.0f32, 2.0, 3.0, -1.0, 0.5, 0.0];
        let t = RegistrationTransform::identity();
        let out = apply_registration_transform(&pts, &t).unwrap();
        for (a, b) in pts.iter().zip(out.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_apply_transform_known_translation() {
        let pts = vec![0.0f32, 0.0, 0.0];
        let t = RegistrationTransform {
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            translation: [5.0, 6.0, 7.0],
            scale: 1.0,
        };
        let out = apply_registration_transform(&pts, &t).unwrap();
        assert!(approx_eq(out[0], 5.0, 1e-6));
        assert!(approx_eq(out[1], 6.0, 1e-6));
        assert!(approx_eq(out[2], 7.0, 1e-6));
    }

    #[test]
    fn test_apply_transform_empty_error() {
        let t = RegistrationTransform::identity();
        let result = apply_registration_transform(&[], &t);
        assert!(matches!(result, Err(RegistrationError::EmptyCloud)));
    }

    #[test]
    fn test_apply_transform_invalid_length() {
        let t = RegistrationTransform::identity();
        let result = apply_registration_transform(&[1.0, 2.0], &t);
        assert!(matches!(
            result,
            Err(RegistrationError::InvalidPositionLength { len: 2 })
        ));
    }

    #[test]
    fn test_apply_transform_rejects_non_positive_scale() {
        let mut t = RegistrationTransform::identity();
        t.scale = 0.0;
        let err = apply_registration_transform(&[1.0, 2.0, 3.0], &t);
        assert!(matches!(err, Err(RegistrationError::InvalidScale { .. })));
        t.scale = f32::NAN;
        let err = apply_registration_transform(&[1.0, 2.0, 3.0], &t);
        assert!(matches!(err, Err(RegistrationError::InvalidScale { .. })));
    }

    // -----------------------------------------------------------------------
    // compute_registration_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_improvement_factor() {
        let result = RegistrationResult {
            transform: RegistrationTransform::identity(),
            final_rmse: 0.5,
            n_iterations: 10,
            converged: true,
            n_correspondences: 100,
            rmse_history: vec![2.0, 1.0, 0.5],
        };
        let stats = compute_registration_stats(&result, 2.0);
        assert!(approx_eq(stats.improvement_factor, 4.0, 1e-5));
        assert!(approx_eq(stats.initial_rmse, 2.0, 1e-6));
        assert!(approx_eq(stats.final_rmse, 0.5, 1e-6));
    }

    #[test]
    fn test_stats_identity_rotation_angle() {
        let result = RegistrationResult {
            transform: RegistrationTransform::identity(),
            final_rmse: 0.1,
            n_iterations: 5,
            converged: true,
            n_correspondences: 50,
            rmse_history: vec![0.1],
        };
        let stats = compute_registration_stats(&result, 1.0);
        assert!(approx_eq(stats.rotation_angle_deg, 0.0, 1e-4));
        assert!(approx_eq(stats.scale_change, 0.0, 1e-6));
        assert!(approx_eq(stats.transform_magnitude, 0.0, 1e-6));
    }

    #[test]
    fn test_stats_zero_final_rmse() {
        let result = RegistrationResult {
            transform: RegistrationTransform::identity(),
            final_rmse: 0.0,
            n_iterations: 3,
            converged: true,
            n_correspondences: 10,
            rmse_history: vec![0.0],
        };
        let stats = compute_registration_stats(&result, 1.0);
        // When final_rmse = 0, improvement_factor should be 1.0 (no divide by zero)
        assert!(approx_eq(stats.improvement_factor, 1.0, 1e-6));
    }

    // -----------------------------------------------------------------------
    // format_registration_result
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_result_non_empty() {
        let result = RegistrationResult {
            transform: RegistrationTransform::identity(),
            final_rmse: 0.01,
            n_iterations: 20,
            converged: true,
            n_correspondences: 100,
            rmse_history: vec![0.01],
        };
        let stats = compute_registration_stats(&result, 1.0);
        let s = format_registration_result(&result, &stats);
        assert!(!s.is_empty());
        assert!(s.contains("ICP:"));
        assert!(s.contains("converged=true"));
    }

    #[test]
    fn test_format_result_contains_iter_count() {
        let result = RegistrationResult {
            transform: RegistrationTransform::identity(),
            final_rmse: 0.1,
            n_iterations: 42,
            converged: false,
            n_correspondences: 30,
            rmse_history: vec![0.1],
        };
        let stats = compute_registration_stats(&result, 0.5);
        let s = format_registration_result(&result, &stats);
        assert!(s.contains("42"), "Expected iter count 42 in: {}", s);
    }

    // -----------------------------------------------------------------------
    // subsample_positions
    // -----------------------------------------------------------------------

    #[test]
    fn test_subsample_strides() {
        let pts = grid_positions(8, 1.0);
        assert_eq!(subsample_positions(&pts, 1).unwrap().len(), pts.len());
        // 4 of 8 points survive, 3 floats each.
        assert_eq!(subsample_positions(&pts, 2).unwrap().len(), 12);
        let four = vec![
            0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0,
        ];
        // stride=4 → only first point
        let out = subsample_positions(&four, 4).unwrap();
        assert_eq!(out.len(), 3);
        assert!(approx_eq(out[0], 0.0, 1e-6));
    }

    #[test]
    fn test_subsample_errors() {
        let empty = subsample_positions(&[], 2);
        assert!(matches!(empty, Err(RegistrationError::EmptyCloud)));
        assert!(matches!(
            subsample_positions(&[1.0, 2.0], 1),
            Err(RegistrationError::InvalidPositionLength { len: 2 })
        ));
    }

    #[test]
    fn test_subsample_zero_stride_treated_as_one() {
        let pts = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = subsample_positions(&pts, 0).unwrap();
        assert_eq!(out.len(), pts.len());
    }
}
