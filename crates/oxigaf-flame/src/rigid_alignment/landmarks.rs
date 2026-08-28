//! Landmark-based (optionally weighted) alignment, and alignment
//! statistics/formatting.

use super::icp::align_nearest_neighbors_filtered;
use super::procrustes::{align_procrustes, align_rmse, parse_points};
use super::svd::{mat3_det, mat3_mul, svd_3x3};
use super::types::{AlignmentError, AlignmentStats, IcpResult, SimilarityTransform};

// ─── Landmark-based alignment ────────────────────────────────────────────────

/// Align using a sparse set of facial landmarks.
///
/// Delegates to [`align_procrustes`].
///
/// # Errors
/// Returns [`AlignmentError`] for invalid inputs.
pub fn align_by_landmarks(
    source_landmarks: &[f32],
    target_landmarks: &[f32],
) -> Result<SimilarityTransform, AlignmentError> {
    align_procrustes(source_landmarks, target_landmarks)
}

/// Weighted landmark alignment: minimise `Σ_i w_i ‖R·s_i + t - t_i‖²`.
///
/// Uses weighted Procrustes: weighted centroids and weighted cross-covariance.
///
/// # Errors
/// Returns [`AlignmentError`] for invalid inputs or size mismatches.
pub fn align_by_weighted_landmarks(
    source_landmarks: &[f32],
    target_landmarks: &[f32],
    weights: &[f32],
) -> Result<SimilarityTransform, AlignmentError> {
    let n = parse_points(source_landmarks, 3, "source_landmarks")?;
    let nt = parse_points(target_landmarks, 3, "target_landmarks")?;
    if n != nt {
        return Err(AlignmentError::PointCountMismatch { src: n, tgt: nt });
    }
    if weights.len() != n {
        return Err(AlignmentError::WeightLengthMismatch {
            w: weights.len(),
            n,
        });
    }

    // Effective weight sum
    let w_sum: f32 = weights.iter().sum();
    if w_sum < 1e-12 {
        return Err(AlignmentError::InvalidConfig(
            "sum of weights is zero or near-zero".to_string(),
        ));
    }
    let inv_w = 1.0 / w_sum;

    // Weighted centroids
    let mut mu_s = [0.0f32; 3];
    let mut mu_t = [0.0f32; 3];
    for i in 0..n {
        let w = weights[i];
        mu_s[0] += w * source_landmarks[i * 3];
        mu_s[1] += w * source_landmarks[i * 3 + 1];
        mu_s[2] += w * source_landmarks[i * 3 + 2];
        mu_t[0] += w * target_landmarks[i * 3];
        mu_t[1] += w * target_landmarks[i * 3 + 1];
        mu_t[2] += w * target_landmarks[i * 3 + 2];
    }
    mu_s = [mu_s[0] * inv_w, mu_s[1] * inv_w, mu_s[2] * inv_w];
    mu_t = [mu_t[0] * inv_w, mu_t[1] * inv_w, mu_t[2] * inv_w];

    // Weighted cross-covariance and weighted variance
    let mut sigma = [0.0f32; 9];
    let mut var_s = 0.0f32;
    for i in 0..n {
        let w = weights[i] * inv_w;
        let sx = source_landmarks[i * 3] - mu_s[0];
        let sy = source_landmarks[i * 3 + 1] - mu_s[1];
        let sz = source_landmarks[i * 3 + 2] - mu_s[2];
        let tx = target_landmarks[i * 3] - mu_t[0];
        let ty = target_landmarks[i * 3 + 1] - mu_t[1];
        let tz = target_landmarks[i * 3 + 2] - mu_t[2];
        sigma[0] += w * tx * sx;
        sigma[1] += w * tx * sy;
        sigma[2] += w * tx * sz;
        sigma[3] += w * ty * sx;
        sigma[4] += w * ty * sy;
        sigma[5] += w * ty * sz;
        sigma[6] += w * tz * sx;
        sigma[7] += w * tz * sy;
        sigma[8] += w * tz * sz;
        var_s += w * (sx * sx + sy * sy + sz * sz);
    }

    let (u, sv, vt) = svd_3x3(&sigma);

    // svd_3x3 guarantees det(U)=+1 and det(V)=+1, so R = U * Vt is always proper.
    let rotation = mat3_mul(&u, &vt);

    // Reflection-sign correction: see align_procrustes.
    let d = if mat3_det(&sigma) < 0.0 { -1.0 } else { 1.0 };
    let trace_s = sv[0] + sv[1] + d * sv[2];
    let scale = if var_s > 1e-12 { trace_s / var_s } else { 1.0 };

    let r = &rotation;
    let translation = [
        mu_t[0] - scale * (r[0] * mu_s[0] + r[1] * mu_s[1] + r[2] * mu_s[2]),
        mu_t[1] - scale * (r[3] * mu_s[0] + r[4] * mu_s[1] + r[5] * mu_s[2]),
        mu_t[2] - scale * (r[6] * mu_s[0] + r[7] * mu_s[1] + r[8] * mu_s[2]),
    ];

    Ok(SimilarityTransform {
        rotation,
        translation,
        scale,
    })
}

/// Compute alignment statistics for `source` aligned to `target`.
///
/// # Errors
/// Returns [`AlignmentError`] for invalid inputs.
pub fn align_compute_stats(
    source: &[f32],
    target: &[f32],
    transform: &SimilarityTransform,
) -> Result<AlignmentStats, AlignmentError> {
    let n = parse_points(source, 1, "source")?;
    let nt = parse_points(target, 1, "target")?;
    if n != nt {
        return Err(AlignmentError::PointCountMismatch { src: n, tgt: nt });
    }

    // RMSE before (identity transform)
    let identity = SimilarityTransform::identity();
    let rmse_before = align_rmse(source, target, &identity)?;
    let rmse_after = align_rmse(source, target, transform)?;

    // Nearest-neighbour distances on aligned source
    let aligned: Vec<f32> = transform.apply_flat(source);
    let (_, distances) = align_nearest_neighbors_filtered(&aligned, target, f32::INFINITY)?;
    let mean_dist = if distances.is_empty() {
        0.0
    } else {
        distances.iter().sum::<f32>() / distances.len() as f32
    };
    let max_dist = distances.iter().copied().fold(0.0f32, f32::max);

    let improvement_ratio = if rmse_after > 1e-12 {
        rmse_before / rmse_after
    } else {
        f32::INFINITY
    };

    Ok(AlignmentStats {
        rmse_before,
        rmse_after,
        improvement_ratio,
        mean_correspondence_dist: mean_dist,
        max_correspondence_dist: max_dist,
        n_correspondences: n,
    })
}

/// Format alignment statistics as a human-readable string.
#[must_use]
pub fn align_format_stats(stats: &AlignmentStats) -> String {
    format!(
        "Alignment Stats:\n  RMSE before : {:.6}\n  RMSE after  : {:.6}\n  Improvement : {:.2}×\n  Mean corr.  : {:.6}\n  Max corr.   : {:.6}\n  N corr.     : {}",
        stats.rmse_before,
        stats.rmse_after,
        stats.improvement_ratio,
        stats.mean_correspondence_dist,
        stats.max_correspondence_dist,
        stats.n_correspondences,
    )
}

/// Format ICP result as a human-readable string.
#[must_use]
pub fn align_format_icp_result(result: &IcpResult) -> String {
    format!(
        "ICP Result:\n  Converged   : {}\n  Iterations  : {}\n  Final RMSE  : {:.6}\n  Scale       : {:.6}",
        result.converged,
        result.n_iterations,
        result.final_rmse,
        result.transform.scale,
    )
}
