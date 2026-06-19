//! # Rigid and Similarity Alignment
//!
//! Implements rigid (and similarity) alignment of 3D head meshes via:
//! - Closed-form Procrustes (Umeyama algorithm) using 3×3 Jacobi SVD
//! - Iterative Closest Point (ICP) alignment
//! - Landmark-based (optionally weighted) alignment
//! - Nearest-neighbour search (brute-force, pure Rust)
//!
//! All matrix operations are manual (no ndarray). PRNG is xorshift64 (no rand crate).
//!
//! ## References
//! - Umeyama, "Least-squares estimation of transformation parameters …", 1991.
//! - Besl & `McKay`, "A method for registration of 3-D shapes", 1992.

// ─── Error type ─────────────────────────────────────────────────────────────

/// Errors returned by rigid-alignment functions.
#[derive(Debug, thiserror::Error)]
pub enum AlignmentError {
    /// Fewer points than the minimum required.
    #[error("not enough points: need at least {needed}, got {got}")]
    NotEnoughPoints { needed: usize, got: usize },

    /// Source and target have different point counts.
    #[error("dimension mismatch: source has {src} points, target has {tgt}")]
    PointCountMismatch { src: usize, tgt: usize },

    /// Matrix inversion failed (numerically singular).
    #[error("singular matrix: cannot invert")]
    Singular,

    /// ICP did not converge within the iteration budget.
    #[error("did not converge after {0} iterations")]
    DidNotConverge(usize),

    /// Invalid configuration parameter.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// Empty slice was passed where at least one element is required.
    #[error("empty input")]
    EmptyInput,

    /// Lengths of weights and landmarks disagree.
    #[error("weight count {w} does not match landmark count {n}")]
    WeightLengthMismatch { w: usize, n: usize },
}

// ─── Xorshift64 PRNG (no rand crate) ────────────────────────────────────────

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

#[inline]
#[allow(dead_code)]
fn xorshift_f32(state: &mut u64) -> f32 {
    xorshift64(state) as f32 / u64::MAX as f32
}

// ─── 3×3 matrix utilities (row-major) ───────────────────────────────────────

/// Multiply two 3×3 row-major matrices.
#[inline]
fn mat3_mul(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let mut c = [0.0f32; 9];
    for row in 0..3 {
        for col in 0..3 {
            let mut s = 0.0f32;
            for k in 0..3 {
                s += a[row * 3 + k] * b[k * 3 + col];
            }
            c[row * 3 + col] = s;
        }
    }
    c
}

/// Transpose a 3×3 row-major matrix.
#[inline]
fn mat3_transpose(m: &[f32; 9]) -> [f32; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

/// Determinant of a 3×3 row-major matrix.
#[inline]
fn mat3_det(m: &[f32; 9]) -> f32 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Identity 3×3 matrix.
#[inline]
fn mat3_identity() -> [f32; 9] {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}

/// 3D cross product.
#[inline]
fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// 3D dot product.
#[inline]
fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Euclidean norm of a 3D vector.
#[inline]
fn vec3_norm(a: [f32; 3]) -> f32 {
    vec3_dot(a, a).sqrt()
}

/// Scale a column of a 3×3 row-major matrix by a scalar.
/// Column `col` ∈ {0,1,2}.
#[inline]
fn mat3_scale_col(m: &mut [f32; 9], col: usize, s: f32) {
    m[col] *= s;
    m[3 + col] *= s;
    m[6 + col] *= s;
}

/// Extract column `col` from a 3×3 row-major matrix as `[f32;3]`.
#[inline]
fn mat3_col(m: &[f32; 9], col: usize) -> [f32; 3] {
    [m[col], m[3 + col], m[6 + col]]
}

/// Set column `col` of a 3×3 row-major matrix.
#[inline]
fn mat3_set_col(m: &mut [f32; 9], col: usize, v: [f32; 3]) {
    m[col] = v[0];
    m[3 + col] = v[1];
    m[6 + col] = v[2];
}

// ─── 3×3 Jacobi SVD (symmetric eigendecomposition of MᵀM) ──────────────────

// ── Jacobi SVD helpers ────────────────────────────────────────────────────────

const SVD_SWEEPS: usize = 20;
const SVD_PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

/// Run Jacobi eigendecomposition sweeps on the symmetric matrix `bsym`.
/// Accumulates eigenvectors in `eigvec` (columns).  Modifies `bsym` in-place
/// until it is approximately diagonal.
fn svd_jacobi_sweep(bsym: &mut [f32; 9], eigvec: &mut [f32; 9]) {
    for _ in 0..SVD_SWEEPS {
        for &(pivot_p, pivot_q) in &SVD_PAIRS {
            let bpq = bsym[pivot_p * 3 + pivot_q];
            if bpq.abs() < 1e-14 {
                continue;
            }
            let bpp = bsym[pivot_p * 3 + pivot_p];
            let bqq = bsym[pivot_q * 3 + pivot_q];
            let theta = 0.5 * (bqq - bpp) / bpq;
            let jacobi_t = if theta >= 0.0 {
                1.0 / (theta + (1.0 + theta * theta).sqrt())
            } else {
                1.0 / (theta - (1.0 + theta * theta).sqrt())
            };
            let jacobi_c = 1.0 / (1.0 + jacobi_t * jacobi_t).sqrt();
            let jacobi_s = jacobi_t * jacobi_c;
            bsym[pivot_p * 3 + pivot_p] = bpp - jacobi_t * bpq;
            bsym[pivot_q * 3 + pivot_q] = bqq + jacobi_t * bpq;
            bsym[pivot_p * 3 + pivot_q] = 0.0;
            bsym[pivot_q * 3 + pivot_p] = 0.0;
            // Third row (the one that is neither pivot_p nor pivot_q)
            let third_row = 3 - pivot_p - pivot_q; // works for pairs (0,1),(0,2),(1,2)
            let brp = bsym[third_row * 3 + pivot_p];
            let brq = bsym[third_row * 3 + pivot_q];
            bsym[third_row * 3 + pivot_p] = jacobi_c * brp - jacobi_s * brq;
            bsym[pivot_p * 3 + third_row] = bsym[third_row * 3 + pivot_p];
            bsym[third_row * 3 + pivot_q] = jacobi_s * brp + jacobi_c * brq;
            bsym[pivot_q * 3 + third_row] = bsym[third_row * 3 + pivot_q];
            // Accumulate eigenvectors: V ← V * J(pivot_p,pivot_q,θ)
            for ev_row in 0..3 {
                let vrp = eigvec[ev_row * 3 + pivot_p];
                let vrq = eigvec[ev_row * 3 + pivot_q];
                eigvec[ev_row * 3 + pivot_p] = jacobi_c * vrp - jacobi_s * vrq;
                eigvec[ev_row * 3 + pivot_q] = jacobi_s * vrp + jacobi_c * vrq;
            }
        }
    }
}

/// Sort the three singular values in `sigma_unsorted` descending and permute
/// the corresponding columns of `eigvec`.  Returns `(v_sorted, sigma_sorted)`.
/// Ensures `det(v_sorted) = +1` by negating the last column if needed.
fn svd_sort_and_correct_v(eigvec: &[f32; 9], sigma_unsorted: [f32; 3]) -> ([f32; 9], [f32; 3]) {
    let mut idx = [0usize, 1, 2];
    for ins in 1..3 {
        let mut pos = ins;
        while pos > 0 && sigma_unsorted[idx[pos - 1]] < sigma_unsorted[idx[pos]] {
            idx.swap(pos - 1, pos);
            pos -= 1;
        }
    }
    let sigma = [
        sigma_unsorted[idx[0]],
        sigma_unsorted[idx[1]],
        sigma_unsorted[idx[2]],
    ];
    let mut v_sorted = mat3_identity();
    mat3_set_col(&mut v_sorted, 0, mat3_col(eigvec, idx[0]));
    mat3_set_col(&mut v_sorted, 1, mat3_col(eigvec, idx[1]));
    mat3_set_col(&mut v_sorted, 2, mat3_col(eigvec, idx[2]));
    if mat3_det(&v_sorted) < 0.0 {
        mat3_scale_col(&mut v_sorted, 2, -1.0);
    }
    (v_sorted, sigma)
}

/// Compute U columns from `mat_m * v_sorted[:,i] / sigma[i]`.
/// Fills degenerate columns via cross-product; ensures `det(U) = +1`.
fn svd_compute_u(mat_m: &[f32; 9], v_sorted: &[f32; 9], sigma: [f32; 3]) -> [f32; 9] {
    let mut u_mat = mat3_identity();
    for (col_idx, &sig_val) in sigma.iter().enumerate() {
        if sig_val > 1e-10 {
            let vi = mat3_col(v_sorted, col_idx);
            let ui = [
                mat_m[0] * vi[0] + mat_m[1] * vi[1] + mat_m[2] * vi[2],
                mat_m[3] * vi[0] + mat_m[4] * vi[1] + mat_m[5] * vi[2],
                mat_m[6] * vi[0] + mat_m[7] * vi[1] + mat_m[8] * vi[2],
            ];
            let ui_norm = vec3_norm(ui);
            if ui_norm > 1e-10 {
                let inv_norm = 1.0 / ui_norm;
                mat3_set_col(
                    &mut u_mat,
                    col_idx,
                    [ui[0] * inv_norm, ui[1] * inv_norm, ui[2] * inv_norm],
                );
            }
        }
    }
    // Fill degenerate columns via cross product
    if sigma[0] <= 1e-10 {
        u_mat = mat3_identity();
    } else if sigma[1] <= 1e-10 {
        let u0 = mat3_col(&u_mat, 0);
        let perp = if u0[0].abs() < 0.9 {
            [1.0f32, 0.0, 0.0]
        } else {
            [0.0f32, 1.0, 0.0]
        };
        let u1_raw = vec3_cross(u0, perp);
        let norm1 = vec3_norm(u1_raw);
        let u1 = if norm1 > 1e-10 {
            [u1_raw[0] / norm1, u1_raw[1] / norm1, u1_raw[2] / norm1]
        } else {
            [0.0, 1.0, 0.0]
        };
        let u2 = vec3_cross(u0, u1);
        mat3_set_col(&mut u_mat, 1, u1);
        mat3_set_col(&mut u_mat, 2, u2);
    } else if sigma[2] <= 1e-10 {
        let u2 = vec3_cross(mat3_col(&u_mat, 0), mat3_col(&u_mat, 1));
        mat3_set_col(&mut u_mat, 2, u2);
    }
    if mat3_det(&u_mat) < 0.0 {
        mat3_scale_col(&mut u_mat, 2, -1.0);
    }
    u_mat
}

/// Approximate 3×3 SVD using Jacobi iterations on `MᵀM`.
///
/// Returns `(U, S, Vt)` such that `M ≈ U * diag(S) * Vt`.
/// * `U` and `Vt` are 3×3 row-major orthogonal matrices (both det = +1).
/// * `S` is `[s0, s1, s2]` with non-negative singular values in descending order.
///
/// Algorithm:
/// 1. Form B = `MᵀM` (symmetric 3×3)
/// 2. Jacobi eigendecomposition of B → eigenvalues `λ_i`, eigenvectors V (20 sweeps)
/// 3. `σ_i` = sqrt(max(0, `λ_i`))
/// 4. Sort descending by σ; correct `det(V_sorted)` = +1 (flip last col if needed)
/// 5. U[:,i] = M * V[:,i] / `σ_i`  (fill zero-σ columns via cross-product)
/// 6. Correct det(U) = +1 (flip last col if needed)
pub(crate) fn svd_3x3(mat_m: &[f32; 9]) -> ([f32; 9], [f32; 3], [f32; 9]) {
    // Step 1: B = Mᵀ M
    let normal_mat = mat3_mul(&mat3_transpose(mat_m), mat_m);

    // Step 2: Jacobi eigen-decomposition of symmetric B
    let mut bsym = normal_mat;
    let mut eigvec = mat3_identity();
    svd_jacobi_sweep(&mut bsym, &mut eigvec);

    // Step 3: eigenvalues → singular values
    let sigma_unsorted = [
        bsym[0].max(0.0).sqrt(),
        bsym[4].max(0.0).sqrt(),
        bsym[8].max(0.0).sqrt(),
    ];

    // Steps 4–4b: sort + ensure det(V) = +1
    let (v_sorted, sigma) = svd_sort_and_correct_v(&eigvec, sigma_unsorted);

    // Steps 5–6: compute U, fill degenerate cols, ensure det(U) = +1
    let u_mat = svd_compute_u(mat_m, &v_sorted, sigma);

    let vt = mat3_transpose(&v_sorted);
    (u_mat, sigma, vt)
}

// ─── SimilarityTransform ────────────────────────────────────────────────────

/// A rigid similarity transform: `x' = scale * R * x + t`.
///
/// `rotation` is stored as a 3×3 row-major matrix with `det = +1`.
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarityTransform {
    /// 3×3 rotation matrix (row-major, det = +1).
    pub rotation: [f32; 9],
    /// Translation vector.
    pub translation: [f32; 3],
    /// Uniform scale factor.
    pub scale: f32,
}

impl SimilarityTransform {
    /// Identity transform (no rotation, no translation, scale = 1).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            rotation: mat3_identity(),
            translation: [0.0; 3],
            scale: 1.0,
        }
    }

    /// Apply to a single 3D point: `x' = scale * R * x + t`.
    #[must_use]
    pub fn apply(&self, point: [f32; 3]) -> [f32; 3] {
        let r = &self.rotation;
        let s = self.scale;
        let t = self.translation;
        [
            s * (r[0] * point[0] + r[1] * point[1] + r[2] * point[2]) + t[0],
            s * (r[3] * point[0] + r[4] * point[1] + r[5] * point[2]) + t[1],
            s * (r[6] * point[0] + r[7] * point[1] + r[8] * point[2]) + t[2],
        ]
    }

    /// Apply to a batch of points (array of `[f32;3]`).
    #[must_use]
    pub fn apply_batch(&self, points: &[[f32; 3]]) -> Vec<[f32; 3]> {
        points.iter().map(|&p| self.apply(p)).collect()
    }

    /// Apply to a flat N×3 slice (length must be divisible by 3).
    #[must_use]
    pub fn apply_flat(&self, positions: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(positions.len());
        for chunk in positions.chunks_exact(3) {
            let p = [chunk[0], chunk[1], chunk[2]];
            let q = self.apply(p);
            out.extend_from_slice(&q);
        }
        // handle any leftover (degenerate case — just copy)
        let rem = positions.len() % 3;
        if rem > 0 {
            out.extend_from_slice(&positions[positions.len() - rem..]);
        }
        out
    }

    /// Return the inverse transform such that `inv.apply(self.apply(x)) ≈ x`.
    ///
    /// * `scale_inv` = 1 / scale
    /// * `rotation_inv` = Rᵀ
    /// * `translation_inv` = -(1/scale) * Rᵀ * t
    #[must_use]
    pub fn inverse(&self) -> SimilarityTransform {
        let s_inv = if self.scale.abs() > f32::EPSILON {
            1.0 / self.scale
        } else {
            0.0
        };
        let rt = mat3_transpose(&self.rotation);
        let t = self.translation;
        // -(1/s) * Rᵀ * t
        let ti = [
            -s_inv * (rt[0] * t[0] + rt[1] * t[1] + rt[2] * t[2]),
            -s_inv * (rt[3] * t[0] + rt[4] * t[1] + rt[5] * t[2]),
            -s_inv * (rt[6] * t[0] + rt[7] * t[1] + rt[8] * t[2]),
        ];
        SimilarityTransform {
            rotation: rt,
            translation: ti,
            scale: s_inv,
        }
    }

    /// Compose `self` then `other`: `other.apply(self.apply(x))`.
    ///
    /// * scale = s1 * s2
    /// * rotation = R2 * R1
    /// * translation = s2 * R2 * t1 + t2
    #[must_use]
    pub fn compose(&self, other: &SimilarityTransform) -> SimilarityTransform {
        let r_new = mat3_mul(&other.rotation, &self.rotation);
        let s_new = self.scale * other.scale;
        let t1 = self.translation;
        let r2 = &other.rotation;
        let s2 = other.scale;
        let t2 = other.translation;
        let t_new = [
            s2 * (r2[0] * t1[0] + r2[1] * t1[1] + r2[2] * t1[2]) + t2[0],
            s2 * (r2[3] * t1[0] + r2[4] * t1[1] + r2[5] * t1[2]) + t2[1],
            s2 * (r2[6] * t1[0] + r2[7] * t1[1] + r2[8] * t1[2]) + t2[2],
        ];
        SimilarityTransform {
            rotation: r_new,
            translation: t_new,
            scale: s_new,
        }
    }

    /// Return the rotation as a `[[f32;3];3]` array.
    #[must_use]
    pub fn rotation_matrix(&self) -> [[f32; 3]; 3] {
        let r = &self.rotation;
        [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]]
    }
}

// ─── Procrustes alignment ────────────────────────────────────────────────────

/// Parse a flat N×3 slice into N points; validates length and count.
fn parse_points(data: &[f32], min_n: usize, label: &str) -> Result<usize, AlignmentError> {
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
fn centroid(data: &[f32], n: usize) -> [f32; 3] {
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
fn point_variance(data: &[f32], n: usize, mu: [f32; 3]) -> f32 {
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
fn cross_covariance(
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

    // Scale = trace(diag(S)) / var(source)
    let trace_s = sv[0] + sv[1] + sv[2];
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

// ─── Nearest-neighbour search ────────────────────────────────────────────────

/// Find nearest point in `target` for each point in `source`.
///
/// Returns indices: `result[i]` = index in target nearest to `source[i]`.
///
/// # Errors
/// Returns [`AlignmentError`] for empty or malformed inputs.
pub fn align_nearest_neighbors(
    source: &[f32],
    target: &[f32],
) -> Result<Vec<usize>, AlignmentError> {
    let n = parse_points(source, 1, "source")?;
    let m = parse_points(target, 1, "target")?;

    let mut indices = Vec::with_capacity(n);
    for i in 0..n {
        let sx = source[i * 3];
        let sy = source[i * 3 + 1];
        let sz = source[i * 3 + 2];
        let mut best_idx = 0usize;
        let mut best_d2 = f32::INFINITY;
        for j in 0..m {
            let dx = sx - target[j * 3];
            let dy = sy - target[j * 3 + 1];
            let dz = sz - target[j * 3 + 2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < best_d2 {
                best_d2 = d2;
                best_idx = j;
            }
        }
        indices.push(best_idx);
    }
    Ok(indices)
}

/// Find nearest points with distance filtering.
///
/// Returns `(indices, distances)`. Source points with nearest-target distance
/// exceeding `max_dist` get index `usize::MAX`.
///
/// # Errors
/// Returns [`AlignmentError`] for empty or malformed inputs.
pub fn align_nearest_neighbors_filtered(
    source: &[f32],
    target: &[f32],
    max_dist: f32,
) -> Result<(Vec<usize>, Vec<f32>), AlignmentError> {
    let n = parse_points(source, 1, "source")?;
    let m = parse_points(target, 1, "target")?;
    let max_d2 = max_dist * max_dist;

    let mut indices = Vec::with_capacity(n);
    let mut distances = Vec::with_capacity(n);
    for i in 0..n {
        let sx = source[i * 3];
        let sy = source[i * 3 + 1];
        let sz = source[i * 3 + 2];
        let mut best_idx = 0usize;
        let mut best_d2 = f32::INFINITY;
        for j in 0..m {
            let dx = sx - target[j * 3];
            let dy = sy - target[j * 3 + 1];
            let dz = sz - target[j * 3 + 2];
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 < best_d2 {
                best_d2 = d2;
                best_idx = j;
            }
        }
        if best_d2 <= max_d2 {
            indices.push(best_idx);
            distances.push(best_d2.sqrt());
        } else {
            indices.push(usize::MAX);
            distances.push(best_d2.sqrt());
        }
    }
    Ok((indices, distances))
}

// ─── ICP ─────────────────────────────────────────────────────────────────────

/// Configuration for Iterative Closest Point.
#[derive(Debug, Clone)]
pub struct IcpConfig {
    /// Maximum number of ICP iterations.
    pub max_iterations: usize,
    /// RMSE change below which we declare convergence.
    pub convergence_threshold: f32,
    /// Reject correspondences farther than this distance.
    pub max_correspondence_dist: f32,
    /// Whether to allow scale in addition to rigid alignment.
    pub use_scale: bool,
}

impl Default for IcpConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            convergence_threshold: 1e-5,
            max_correspondence_dist: f32::INFINITY,
            use_scale: false,
        }
    }
}

/// Result of ICP alignment.
#[derive(Debug, Clone)]
pub struct IcpResult {
    /// Accumulated similarity transform.
    pub transform: SimilarityTransform,
    /// Final RMSE between aligned source and target correspondences.
    pub final_rmse: f32,
    /// Number of iterations performed.
    pub n_iterations: usize,
    /// Whether convergence criterion was met.
    pub converged: bool,
    /// RMSE at each iteration.
    pub rmse_history: Vec<f32>,
}

/// Iterative Closest Point alignment.
///
/// Aligns `source` (N×3 flat) to `target` (M×3 flat) by alternating between
/// nearest-neighbour correspondence and Procrustes fitting.
///
/// # Errors
/// Returns [`AlignmentError`] for invalid inputs or if `max_iterations == 0`.
pub fn align_icp(
    source: &[f32],
    target: &[f32],
    config: &IcpConfig,
) -> Result<IcpResult, AlignmentError> {
    let n = parse_points(source, 3, "source")?;
    parse_points(target, 3, "target")?;

    if config.max_iterations == 0 {
        return Err(AlignmentError::InvalidConfig(
            "max_iterations must be > 0".to_string(),
        ));
    }

    // Work on a mutable copy of source
    let mut current: Vec<f32> = source.to_vec();
    let mut accumulated = SimilarityTransform::identity();
    let mut rmse_history = Vec::with_capacity(config.max_iterations);
    let mut prev_rmse = f32::INFINITY;
    let mut converged = false;

    for iter in 0..config.max_iterations {
        // 1. Find correspondences with distance filter
        let (nn_idx, nn_dist) =
            align_nearest_neighbors_filtered(&current, target, config.max_correspondence_dist)?;

        // 2. Build filtered correspondence point sets
        let mut src_corr: Vec<f32> = Vec::new();
        let mut tgt_corr: Vec<f32> = Vec::new();
        for i in 0..n {
            if nn_idx[i] != usize::MAX {
                let j = nn_idx[i];
                src_corr.extend_from_slice(&current[i * 3..i * 3 + 3]);
                tgt_corr.extend_from_slice(&target[j * 3..j * 3 + 3]);
            }
        }

        if src_corr.len() < 9 {
            // fewer than 3 correspondences — cannot fit
            return Err(AlignmentError::NotEnoughPoints {
                needed: 3,
                got: src_corr.len() / 3,
            });
        }

        // 3. Compute RMSE over valid correspondences
        let n_corr = src_corr.len() / 3;
        let mut mse = 0.0f32;
        for i in 0..n_corr {
            let dx = src_corr[i * 3] - tgt_corr[i * 3];
            let dy = src_corr[i * 3 + 1] - tgt_corr[i * 3 + 1];
            let dz = src_corr[i * 3 + 2] - tgt_corr[i * 3 + 2];
            mse += dx * dx + dy * dy + dz * dz;
        }
        let rmse = (mse / n_corr as f32).sqrt();
        rmse_history.push(rmse);

        // 4. Check convergence
        if (prev_rmse - rmse).abs() < config.convergence_threshold {
            converged = true;
            prev_rmse = rmse;
            // still run the fit for the final transform, then break
            let step = if config.use_scale {
                align_procrustes(&src_corr, &tgt_corr)?
            } else {
                align_procrustes_rigid(&src_corr, &tgt_corr)?
            };
            accumulated = accumulated.compose(&step);
            break;
        }
        prev_rmse = rmse;

        // 5. Fit Procrustes transform on correspondences
        let step = if config.use_scale {
            align_procrustes(&src_corr, &tgt_corr)?
        } else {
            align_procrustes_rigid(&src_corr, &tgt_corr)?
        };

        // 6. Apply step transform to current source
        let next: Vec<f32> = (0..n)
            .flat_map(|i| {
                let p = [current[i * 3], current[i * 3 + 1], current[i * 3 + 2]];
                let q = step.apply(p);
                [q[0], q[1], q[2]]
            })
            .collect();
        current = next;

        // 7. Accumulate: accumulated = accumulated ∘ step
        accumulated = accumulated.compose(&step);

        // Use nn_dist for last-iteration check but drop it now
        let _ = nn_dist;

        if iter + 1 == config.max_iterations {
            // last iteration — converged stays false
        }
    }

    Ok(IcpResult {
        transform: accumulated,
        final_rmse: prev_rmse,
        n_iterations: rmse_history.len(),
        converged,
        rmse_history,
    })
}

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

    let trace_s = sv[0] + sv[1] + sv[2];
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

// ─── Alignment statistics ─────────────────────────────────────────────────────

/// Summary statistics for an alignment operation.
#[derive(Debug, Clone)]
pub struct AlignmentStats {
    /// RMSE before applying the transform.
    pub rmse_before: f32,
    /// RMSE after applying the transform.
    pub rmse_after: f32,
    /// `rmse_before / rmse_after` — larger is better.
    pub improvement_ratio: f32,
    /// Mean nearest-neighbour distance post-alignment.
    pub mean_correspondence_dist: f32,
    /// Maximum nearest-neighbour distance post-alignment.
    pub max_correspondence_dist: f32,
    /// Number of correspondences used.
    pub n_correspondences: usize,
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
}
