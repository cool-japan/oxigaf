//! Closed-form Procrustes (Umeyama) fitting: similarity and rigid-only
//! variants, plus RMSE.

use super::svd::{mat3_det, mat3_mul, svd_3x3};
use super::types::{AlignmentError, SimilarityTransform};

// ─── Procrustes alignment ────────────────────────────────────────────────────

/// Parse a flat N×3 slice into N points; validates length and count.
pub(super) fn parse_points(
    data: &[f32],
    min_n: usize,
    label: &str,
) -> Result<usize, AlignmentError> {
    if data.is_empty() {
        return Err(AlignmentError::EmptyInput);
    }
    if !data.len().is_multiple_of(3) {
        return Err(AlignmentError::InvalidConfig(format!(
            "{label}: slice length {} is not divisible by 3",
            data.len()
        )));
    }
    let n = data.len() / 3;
    if n < min_n {
        return Err(AlignmentError::NotEnoughPoints {
            needed: min_n,
            got: n,
        });
    }
    Ok(n)
}

/// Compute unweighted centroid of flat N×3 data.
pub(super) fn centroid(data: &[f32], n: usize) -> [f32; 3] {
    let mut c = [0.0f32; 3];
    for i in 0..n {
        c[0] += data[i * 3];
        c[1] += data[i * 3 + 1];
        c[2] += data[i * 3 + 2];
    }
    let inv = 1.0 / n as f32;
    [c[0] * inv, c[1] * inv, c[2] * inv]
}

/// Compute variance of centered points (used for Umeyama scale).
pub(super) fn point_variance(data: &[f32], n: usize, mu: [f32; 3]) -> f32 {
    let mut v = 0.0f32;
    for i in 0..n {
        let dx = data[i * 3] - mu[0];
        let dy = data[i * 3 + 1] - mu[1];
        let dz = data[i * 3 + 2] - mu[2];
        v += dx * dx + dy * dy + dz * dz;
    }
    v / n as f32
}

/// Compute cross-covariance Σ = (1/N) * `centered_target^T` * `centered_source` (3×3, row-major).
/// Here `centered_target`[i] = target[i] - `mu_t`, etc.
pub(super) fn cross_covariance(
    source: &[f32],
    target: &[f32],
    n: usize,
    mu_s: [f32; 3],
    mu_t: [f32; 3],
) -> [f32; 9] {
    let mut sigma = [0.0f32; 9];
    let inv = 1.0 / n as f32;
    for i in 0..n {
        let sx = source[i * 3] - mu_s[0];
        let sy = source[i * 3 + 1] - mu_s[1];
        let sz = source[i * 3 + 2] - mu_s[2];
        let tx = target[i * 3] - mu_t[0];
        let ty = target[i * 3 + 1] - mu_t[1];
        let tz = target[i * 3 + 2] - mu_t[2];
        // Σ = Σ_i (t_i * s_i^T) * inv — 3×3 outer product sum
        // Row 0
        sigma[0] += tx * sx;
        sigma[1] += tx * sy;
        sigma[2] += tx * sz;
        // Row 1
        sigma[3] += ty * sx;
        sigma[4] += ty * sy;
        sigma[5] += ty * sz;
        // Row 2
        sigma[6] += tz * sx;
        sigma[7] += tz * sy;
        sigma[8] += tz * sz;
    }
    for s in &mut sigma {
        *s *= inv;
    }
    sigma
}

/// Procrustes analysis: find similarity transform that best aligns `source` to `target`.
///
/// `source`, `target`: N×3 point sets (flat: [x0,y0,z0, x1,y1,z1, …]).
/// Uses closed-form Umeyama algorithm via 3×3 Jacobi SVD.
///
/// # Errors
/// Returns [`AlignmentError`] for under-determined inputs or mismatched sizes.
pub fn align_procrustes(
    source: &[f32],
    target: &[f32],
) -> Result<SimilarityTransform, AlignmentError> {
    let n = parse_points(source, 3, "source")?;
    let nt = parse_points(target, 3, "target")?;
    if n != nt {
        return Err(AlignmentError::PointCountMismatch { src: n, tgt: nt });
    }

    let mu_s = centroid(source, n);
    let mu_t = centroid(target, n);
    let var_s = point_variance(source, n, mu_s);
    let sigma = cross_covariance(source, target, n, mu_s, mu_t);

    let (u, sv, vt) = svd_3x3(&sigma);

    // svd_3x3 guarantees det(U)=+1 and det(V)=+1, so R = U * Vt is always proper.
    let rotation = mat3_mul(&u, &vt);

    // Scale = trace(diag(1,1,d)*S)/var(source); d=-1 matches the reflection
    // correction above (det(Σ)<0) per Umeyama's closed form.
    let d = if mat3_det(&sigma) < 0.0 { -1.0 } else { 1.0 };
    let trace_s = sv[0] + sv[1] + d * sv[2];
    let scale = if var_s > 1e-12 { trace_s / var_s } else { 1.0 };

    // t = mu_t - scale * R * mu_s
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

/// Orthogonal Procrustes (no scale — rotation only).
///
/// Finds the rotation R that minimises `‖R*S - T‖_F`.
///
/// # Errors
/// Returns [`AlignmentError`] for invalid inputs.
pub fn align_procrustes_rigid(
    source: &[f32],
    target: &[f32],
) -> Result<SimilarityTransform, AlignmentError> {
    let n = parse_points(source, 3, "source")?;
    let nt = parse_points(target, 3, "target")?;
    if n != nt {
        return Err(AlignmentError::PointCountMismatch { src: n, tgt: nt });
    }

    let mu_s = centroid(source, n);
    let mu_t = centroid(target, n);
    let sigma = cross_covariance(source, target, n, mu_s, mu_t);

    let (u, _sv, vt) = svd_3x3(&sigma);

    // svd_3x3 guarantees det(U)=+1 and det(V)=+1, so R = U * Vt is always proper.
    let rotation = mat3_mul(&u, &vt);
    let r = &rotation;

    let translation = [
        mu_t[0] - (r[0] * mu_s[0] + r[1] * mu_s[1] + r[2] * mu_s[2]),
        mu_t[1] - (r[3] * mu_s[0] + r[4] * mu_s[1] + r[5] * mu_s[2]),
        mu_t[2] - (r[6] * mu_s[0] + r[7] * mu_s[1] + r[8] * mu_s[2]),
    ];

    Ok(SimilarityTransform {
        rotation,
        translation,
        scale: 1.0,
    })
}

/// Compute RMSE between transformed `source` and `target`.
///
/// # Errors
/// Returns [`AlignmentError`] for size mismatches.
pub fn align_rmse(
    source: &[f32],
    target: &[f32],
    transform: &SimilarityTransform,
) -> Result<f32, AlignmentError> {
    let n = parse_points(source, 1, "source")?;
    let nt = parse_points(target, 1, "target")?;
    if n != nt {
        return Err(AlignmentError::PointCountMismatch { src: n, tgt: nt });
    }
    let mut mse = 0.0f32;
    for i in 0..n {
        let p = [source[i * 3], source[i * 3 + 1], source[i * 3 + 2]];
        let q = transform.apply(p);
        let dx = q[0] - target[i * 3];
        let dy = q[1] - target[i * 3 + 1];
        let dz = q[2] - target[i * 3 + 2];
        mse += dx * dx + dy * dy + dz * dz;
    }
    Ok((mse / n as f32).sqrt())
}
