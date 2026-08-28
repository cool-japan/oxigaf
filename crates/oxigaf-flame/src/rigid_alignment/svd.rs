//! Private 3×3 matrix/vector utilities and the Jacobi-sweep 3×3 SVD
//! (`svd_3x3`) they build up to. `svd_3x3` is `pub(crate)`: it is also
//! used directly by `crate::pose_estimation`.

// ─── 3×3 matrix utilities (row-major) ───────────────────────────────────────

/// Multiply two 3×3 row-major matrices.
#[inline]
pub(super) fn mat3_mul(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
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
pub(super) fn mat3_transpose(m: &[f32; 9]) -> [f32; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

/// Determinant of a 3×3 row-major matrix.
#[inline]
pub(super) fn mat3_det(m: &[f32; 9]) -> f32 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Identity 3×3 matrix.
#[inline]
pub(super) fn mat3_identity() -> [f32; 9] {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}

/// 3D cross product.
#[inline]
pub(super) fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// 3D dot product.
#[inline]
pub(super) fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Euclidean norm of a 3D vector.
#[inline]
pub(super) fn vec3_norm(a: [f32; 3]) -> f32 {
    vec3_dot(a, a).sqrt()
}

/// Scale a column of a 3×3 row-major matrix by a scalar.
/// Column `col` ∈ {0,1,2}.
#[inline]
pub(super) fn mat3_scale_col(m: &mut [f32; 9], col: usize, s: f32) {
    m[col] *= s;
    m[3 + col] *= s;
    m[6 + col] *= s;
}

/// Extract column `col` from a 3×3 row-major matrix as `[f32;3]`.
#[inline]
pub(super) fn mat3_col(m: &[f32; 9], col: usize) -> [f32; 3] {
    [m[col], m[3 + col], m[6 + col]]
}

/// Set column `col` of a 3×3 row-major matrix.
#[inline]
pub(super) fn mat3_set_col(m: &mut [f32; 9], col: usize, v: [f32; 3]) {
    m[col] = v[0];
    m[3 + col] = v[1];
    m[6 + col] = v[2];
}

// ─── 3×3 Jacobi SVD (symmetric eigendecomposition of MᵀM) ──────────────────

// ── Jacobi SVD helpers ────────────────────────────────────────────────────────

pub(super) const SVD_SWEEPS: usize = 20;
pub(super) const SVD_PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

/// Run Jacobi eigendecomposition sweeps on the symmetric matrix `bsym`.
/// Accumulates eigenvectors in `eigvec` (columns).  Modifies `bsym` in-place
/// until it is approximately diagonal.
pub(super) fn svd_jacobi_sweep(bsym: &mut [f32; 9], eigvec: &mut [f32; 9]) {
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
pub(super) fn svd_sort_and_correct_v(
    eigvec: &[f32; 9],
    sigma_unsorted: [f32; 3],
) -> ([f32; 9], [f32; 3]) {
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
pub(super) fn svd_compute_u(mat_m: &[f32; 9], v_sorted: &[f32; 9], sigma: [f32; 3]) -> [f32; 9] {
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
/// Returns `(U, S, Vt)` with `det(U) = det(V) = +1` always (the Umeyama
/// reflection correction, so `U * Vt` is always a proper rotation) — hence
/// reconstruction is conditional, not the unconditional `M ≈ U * diag(S) *
/// Vt` a general SVD would give: that holds only when `det(M) >= 0`. When
/// `det(M) < 0`, it instead takes `M ≈ U * diag(1, 1, -1) * diag(S) * Vt`.
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
