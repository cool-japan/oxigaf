//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    ExpressionState, ExpressionVarianceStats, LinearExpressionRetargeter, RetargetConfig,
    RetargetError, RetargetPair, RetargetStats,
};

/// Compute per-dimension variance statistics across a set of expression states.
///
/// Returns [`RetargetError::EmptySequence`] if `states` is empty.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn retar_compute_variance(
    states: &[ExpressionState],
) -> Result<ExpressionVarianceStats, RetargetError> {
    if states.is_empty() {
        return Err(RetargetError::EmptySequence);
    }
    let dim = states[0].expr_dim;
    let n = states.len() as f32;
    let mut mean = vec![0.0_f32; dim];
    for s in states {
        for (m, &v) in mean.iter_mut().zip(s.expression_params.iter()) {
            *m += v;
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    let mut variance = vec![0.0_f32; dim];
    for s in states {
        for (var, (&v, &mu)) in variance
            .iter_mut()
            .zip(s.expression_params.iter().zip(mean.iter()))
        {
            let d = v - mu;
            *var += d * d;
        }
    }
    for var in &mut variance {
        *var /= n;
    }
    let total_variance = variance.iter().sum();
    let mut indices: Vec<usize> = (0..dim).collect();
    indices.sort_unstable_by(|&a, &b| {
        variance[b]
            .partial_cmp(&variance[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(ExpressionVarianceStats {
        per_dim_variance: variance,
        total_variance,
        top_k_dims: indices,
    })
}
/// Normalize expression params by per-dimension standard deviation.
///
/// Dimensions with near-zero variance are left as-is (no division by zero).
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn retar_standardize(
    state: &ExpressionState,
    variance_stats: &ExpressionVarianceStats,
) -> Result<Vec<f32>, RetargetError> {
    if state.expr_dim != variance_stats.per_dim_variance.len() {
        return Err(RetargetError::DimensionMismatch {
            src_dim: variance_stats.per_dim_variance.len(),
            got: state.expr_dim,
        });
    }
    let result = state
        .expression_params
        .iter()
        .zip(variance_stats.per_dim_variance.iter())
        .map(|(&v, &var)| {
            let std = var.sqrt();
            if std > 1e-8_f32 {
                v / std
            } else {
                v
            }
        })
        .collect();
    Ok(result)
}
/// Invert standardization back to original scale.
///
/// `expr_dim` is used to rebuild an [`ExpressionState`] from the flat slice.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn retar_unstandardize(
    standardized: &[f32],
    variance_stats: &ExpressionVarianceStats,
    expr_dim: usize,
) -> Result<ExpressionState, RetargetError> {
    if standardized.len() != expr_dim {
        return Err(RetargetError::DimensionMismatch {
            src_dim: expr_dim,
            got: standardized.len(),
        });
    }
    if expr_dim != variance_stats.per_dim_variance.len() {
        return Err(RetargetError::DimensionMismatch {
            src_dim: variance_stats.per_dim_variance.len(),
            got: expr_dim,
        });
    }
    let params = standardized
        .iter()
        .zip(variance_stats.per_dim_variance.iter())
        .map(|(&v, &var)| {
            let std = var.sqrt();
            if std > 1e-8_f32 {
                v * std
            } else {
                v
            }
        })
        .collect();
    Ok(ExpressionState::from_params(params))
}
/// Solve the ridge regression system (A^T A + λI) x = A^T b.
///
/// `a` is stored row-major as `[n_samples × dim]`.
/// Returns the solution vector `x` of length `dim`.
///
/// Uses Cholesky decomposition of the symmetric positive-definite matrix
/// (A^T A + λI).  Fails with [`RetargetError::Singular`] if any pivot is
/// non-positive.
pub(super) fn retar_solve_ridge(
    matrix_a: &[f32],
    rhs_b: &[f32],
    n_samples: usize,
    dim: usize,
    lambda: f32,
) -> Result<Vec<f32>, RetargetError> {
    let mut ata = vec![0.0_f32; dim * dim];
    for row_idx in 0..n_samples {
        for col in 0..dim {
            let ai_c = matrix_a[row_idx * dim + col];
            if ai_c == 0.0_f32 {
                continue;
            }
            for row in 0..dim {
                ata[row * dim + col] += matrix_a[row_idx * dim + row] * ai_c;
            }
        }
    }
    for diag in 0..dim {
        ata[diag * dim + diag] += lambda;
    }
    let mut atb = vec![0.0_f32; dim];
    for row_idx in 0..n_samples {
        let bi = rhs_b[row_idx];
        for diag in 0..dim {
            atb[diag] += matrix_a[row_idx * dim + diag] * bi;
        }
    }
    let mut chol_l = vec![0.0_f32; dim * dim];
    for ii in 0..dim {
        for jj in 0..=ii {
            let mut sum_val = ata[ii * dim + jj];
            for kk in 0..jj {
                sum_val -= chol_l[ii * dim + kk] * chol_l[jj * dim + kk];
            }
            if ii == jj {
                if sum_val <= 0.0_f32 {
                    return Err(RetargetError::Singular);
                }
                chol_l[ii * dim + ii] = sum_val.sqrt();
            } else {
                chol_l[ii * dim + jj] = sum_val / chol_l[jj * dim + jj];
            }
        }
    }
    let mut fwd_y = vec![0.0_f32; dim];
    for ii in 0..dim {
        let mut sum_val = atb[ii];
        for jj in 0..ii {
            sum_val -= chol_l[ii * dim + jj] * fwd_y[jj];
        }
        fwd_y[ii] = sum_val / chol_l[ii * dim + ii];
    }
    let mut sol_x = vec![0.0_f32; dim];
    for ii in (0..dim).rev() {
        let mut sum_val = fwd_y[ii];
        for jj in (ii + 1)..dim {
            sum_val -= chol_l[jj * dim + ii] * sol_x[jj];
        }
        sol_x[ii] = sum_val / chol_l[ii * dim + ii];
    }
    Ok(sol_x)
}
/// Compute per-frame velocity (first difference) of an expression sequence.
///
/// Returns a sequence of length `n - 1`.  Returns [`RetargetError::EmptySequence`]
/// if `sequence` has fewer than 2 frames.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn retar_expression_velocity(
    sequence: &[ExpressionState],
) -> Result<Vec<ExpressionState>, RetargetError> {
    if sequence.len() < 2 {
        return Err(RetargetError::EmptySequence);
    }
    let mut result = Vec::with_capacity(sequence.len() - 1);
    for w in sequence.windows(2) {
        let dim = w[0].expr_dim;
        let expr: Vec<f32> = w[0]
            .expression_params
            .iter()
            .zip(w[1].expression_params.iter())
            .map(|(&a, &b)| b - a)
            .collect();
        let jaw = [
            w[1].jaw_pose[0] - w[0].jaw_pose[0],
            w[1].jaw_pose[1] - w[0].jaw_pose[1],
            w[1].jaw_pose[2] - w[0].jaw_pose[2],
        ];
        result.push(ExpressionState {
            expression_params: expr,
            jaw_pose: jaw,
            expr_dim: dim,
        });
    }
    Ok(result)
}
/// Compute per-frame acceleration (second difference) of a sequence.
///
/// Returns a sequence of length `n - 2`.  Returns [`RetargetError::EmptySequence`]
/// if `sequence` has fewer than 3 frames.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn retar_expression_acceleration(
    sequence: &[ExpressionState],
) -> Result<Vec<ExpressionState>, RetargetError> {
    if sequence.len() < 3 {
        return Err(RetargetError::EmptySequence);
    }
    let vel = retar_expression_velocity(sequence)?;
    retar_expression_velocity(&vel)
}
/// Gaussian smoothing of an expression sequence.
///
/// `sigma` is the smoothing strength in frames.  The kernel half-width is
/// `ceil(3 * sigma)`, **clamped to `len - 1`**.  Boundary frames are padded by
/// repetition (clamp-to-edge), so a tap further than `len - 1` away from the
/// current frame can only ever re-read the first or last frame; the clamp keeps
/// the kernel allocation and the convolution proportional to the sequence
/// instead of to `sigma`, and keeps `2 * half + 1` far away from `i64`
/// overflow.  A consequence is that very large `sigma` values all saturate to
/// the same edge-dominated, near-uniform average rather than growing without
/// bound.
///
/// `sigma <= 0` returns the sequence unchanged.
///
/// # Errors
///
/// - [`RetargetError::EmptySequence`] if `sequence` is empty.
/// - [`RetargetError::InvalidConfig`] if `sigma` is not finite (NaN or ±∞).
pub fn retar_smooth_sequence(
    sequence: &[ExpressionState],
    sigma: f32,
) -> Result<Vec<ExpressionState>, RetargetError> {
    if sequence.is_empty() {
        return Err(RetargetError::EmptySequence);
    }
    if !sigma.is_finite() {
        return Err(RetargetError::InvalidConfig(format!(
            "smoothing sigma must be finite, got {sigma}"
        )));
    }
    if sigma <= 0.0_f32 {
        return Ok(sequence.to_vec());
    }
    let n = i64::try_from(sequence.len()).unwrap_or(i64::MAX);
    // Widest useful half-width: beyond `n - 1` every extra tap is clamped onto
    // a frame the kernel already covers.
    let max_half = n.saturating_sub(1).max(1);
    let half = ((3.0_f32 * sigma).ceil() as i64).clamp(0, max_half);
    let dim = sequence[0].expr_dim;
    let kernel_len = (2 * half + 1) as usize;
    let mut kernel: Vec<f32> = (0..kernel_len)
        .map(|k| {
            let x = k as f32 - half as f32;
            (-0.5_f32 * (x / sigma).powi(2)).exp()
        })
        .collect();
    let kernel_sum: f32 = kernel.iter().sum();
    for w in &mut kernel {
        *w /= kernel_sum;
    }
    let mut result = Vec::with_capacity(sequence.len());
    for i in 0..n {
        let mut expr = vec![0.0_f32; dim];
        let mut jaw = [0.0_f32; 3];
        for (ki, k_offset) in (-half..=half).enumerate() {
            let j = (i + k_offset).clamp(0, n - 1) as usize;
            let w = kernel[ki];
            for (expr_val, &src_val) in expr.iter_mut().zip(sequence[j].expression_params.iter()) {
                *expr_val += w * src_val;
            }
            jaw[0] += w * sequence[j].jaw_pose[0];
            jaw[1] += w * sequence[j].jaw_pose[1];
            jaw[2] += w * sequence[j].jaw_pose[2];
        }
        result.push(ExpressionState {
            expression_params: expr,
            jaw_pose: jaw,
            expr_dim: dim,
        });
    }
    Ok(result)
}
/// Resample an expression sequence to `target_len` frames via linear interpolation.
///
/// Returns [`RetargetError::InvalidConfig`] if `target_len == 0`.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn retar_resample_sequence(
    sequence: &[ExpressionState],
    target_len: usize,
) -> Result<Vec<ExpressionState>, RetargetError> {
    if target_len == 0 {
        return Err(RetargetError::InvalidConfig(
            "target_len must be > 0".to_string(),
        ));
    }
    if sequence.is_empty() {
        return Err(RetargetError::EmptySequence);
    }
    if target_len == 1 {
        return Ok(vec![sequence[0].clone()]);
    }
    if sequence.len() == 1 {
        return Ok(vec![sequence[0].clone(); target_len]);
    }
    let src_n = sequence.len();
    let mut result = Vec::with_capacity(target_len);
    for out_i in 0..target_len {
        let t = out_i as f32 / (target_len - 1) as f32 * (src_n - 1) as f32;
        let lo = (t as usize).min(src_n - 2);
        let hi = lo + 1;
        let frac = t - lo as f32;
        let blended = sequence[lo].blend(&sequence[hi], frac)?;
        result.push(blended);
    }
    Ok(result)
}
/// Cosine similarity of the expression parameter vectors (jaw is excluded).
///
/// Returns `0.0` if either vector is zero.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn retar_expression_similarity(
    a: &ExpressionState,
    b: &ExpressionState,
) -> Result<f32, RetargetError> {
    if a.expr_dim != b.expr_dim {
        return Err(RetargetError::DimensionMismatch {
            src_dim: a.expr_dim,
            got: b.expr_dim,
        });
    }
    let dot: f32 = a
        .expression_params
        .iter()
        .zip(b.expression_params.iter())
        .map(|(&x, &y)| x * y)
        .sum();
    let na: f32 = a
        .expression_params
        .iter()
        .map(|&x| x * x)
        .sum::<f32>()
        .sqrt();
    let nb: f32 = b
        .expression_params
        .iter()
        .map(|&x| x * x)
        .sum::<f32>()
        .sqrt();
    if na < 1e-8_f32 || nb < 1e-8_f32 {
        return Ok(0.0_f32);
    }
    Ok((dot / (na * nb)).clamp(-1.0_f32, 1.0_f32))
}
/// Find the index of the most neutral frame (closest to the zero expression vector).
///
/// Returns [`RetargetError::EmptySequence`] if `sequence` is empty.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn retar_find_neutral_frame(sequence: &[ExpressionState]) -> Result<usize, RetargetError> {
    if sequence.is_empty() {
        return Err(RetargetError::EmptySequence);
    }
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;
    for (i, s) in sequence.iter().enumerate() {
        let dist: f32 = s
            .expression_params
            .iter()
            .map(|&v| v * v)
            .sum::<f32>()
            .sqrt();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    Ok(best_idx)
}
/// Cholesky factorization `A = L Lᵀ` of a symmetric positive-definite matrix.
///
/// `matrix` is `[dim × dim]` row-major; only the lower triangle is read.  The
/// returned `L` is `[dim × dim]` row-major with a zero upper triangle.
fn cholesky_factor(matrix: &[f32], dim: usize) -> Result<Vec<f32>, RetargetError> {
    let mut chol = vec![0.0_f32; dim * dim];
    for ii in 0..dim {
        for jj in 0..=ii {
            let mut sum_val = matrix[ii * dim + jj];
            for kk in 0..jj {
                sum_val -= chol[ii * dim + kk] * chol[jj * dim + kk];
            }
            if ii == jj {
                if sum_val <= 0.0_f32 {
                    return Err(RetargetError::Singular);
                }
                chol[ii * dim + ii] = sum_val.sqrt();
            } else {
                chol[ii * dim + jj] = sum_val / chol[jj * dim + jj];
            }
        }
    }
    Ok(chol)
}

/// Solve `L Lᵀ x = rhs` given the Cholesky factor `L` from [`cholesky_factor`].
fn cholesky_solve(chol: &[f32], rhs: &[f32], dim: usize) -> Vec<f32> {
    // Forward substitution: L y = rhs
    let mut fwd_y = vec![0.0_f32; dim];
    for ii in 0..dim {
        let mut sum_val = rhs[ii];
        for jj in 0..ii {
            sum_val -= chol[ii * dim + jj] * fwd_y[jj];
        }
        fwd_y[ii] = sum_val / chol[ii * dim + ii];
    }
    // Back substitution: Lᵀ x = y
    let mut sol_x = vec![0.0_f32; dim];
    for ii in (0..dim).rev() {
        let mut sum_val = fwd_y[ii];
        for jj in (ii + 1)..dim {
            sum_val -= chol[jj * dim + ii] * sol_x[jj];
        }
        sol_x[ii] = sum_val / chol[ii * dim + ii];
    }
    sol_x
}

/// Build the coefficient-space left-right mirror matrix `M` for a FLAME
/// expression basis.
///
/// Mirroring a FLAME face is a **vertex-space** operation, not a coefficient
/// relabelling: reflect every vertex displacement across the mid-sagittal plane
/// (`x → −x`) *and* swap every vertex with its bilateral counterpart.  Written
/// as matrices that is `P · S · B`, where
///
/// - `B` is the expression basis, `[3N × K]`,
/// - `S` negates the x component of each vertex displacement, and
/// - `P` is the vertex permutation described by `symmetry_map`.
///
/// The mirrored displacement field is projected back onto the basis by least
/// squares, giving a `K × K` matrix in coefficient space:
///
/// `M = (BᵀB + λI)⁻¹ · (Bᵀ P S B)`
///
/// with a tiny ridge term `λ` (`1e-6` of the mean Gram diagonal) so that a
/// rank-deficient basis still factorizes.  Feed the result to
/// [`retar_mirror_expression`].
///
/// # Arguments
///
/// - `expression_basis`: FLAME `expressiondirs` flattened exactly as
///   `ndarray`'s `[N, 3, K]` row-major layout, i.e. element `(vertex, axis, k)`
///   lives at `((vertex * 3) + axis) * K + k`.  Length must be
///   `symmetry_map.len() * 3 * expr_dim`.
/// - `symmetry_map`: for every vertex, the index of its bilateral counterpart
///   (midline vertices map to themselves) — see
///   [`crate::symmetry::SymmetryMap`].
/// - `expr_dim`: number of expression components `K`.
///
/// # Performance
///
/// Cost is `O(3N · K²)`; for the standard FLAME model (`N = 5023`, `K = 100`)
/// this is a few hundred million multiply-adds.  Build the matrix once and
/// reuse it across frames.
///
/// # Errors
///
/// - [`RetargetError::InvalidConfig`] if `expr_dim` is zero, `symmetry_map` is
///   empty, or a symmetry-map entry is out of bounds.
/// - [`RetargetError::DimensionMismatch`] if `expression_basis` does not have
///   `symmetry_map.len() * 3 * expr_dim` elements.
/// - [`RetargetError::Singular`] if `BᵀB + λI` is not positive definite.
pub fn retar_build_expression_mirror_matrix(
    expression_basis: &[f32],
    symmetry_map: &[usize],
    expr_dim: usize,
) -> Result<Vec<f32>, RetargetError> {
    if expr_dim == 0 {
        return Err(RetargetError::InvalidConfig(
            "expr_dim must be > 0".to_string(),
        ));
    }
    let num_vertices = symmetry_map.len();
    if num_vertices == 0 {
        return Err(RetargetError::InvalidConfig(
            "symmetry map must not be empty".to_string(),
        ));
    }
    let expected = num_vertices * 3 * expr_dim;
    if expression_basis.len() != expected {
        return Err(RetargetError::DimensionMismatch {
            src_dim: expected,
            got: expression_basis.len(),
        });
    }
    for (vertex, &mapped) in symmetry_map.iter().enumerate() {
        if mapped >= num_vertices {
            return Err(RetargetError::InvalidConfig(format!(
                "symmetry map entry {vertex} points at vertex {mapped}, out of bounds for {num_vertices} vertices"
            )));
        }
    }

    // gram  = BᵀB          cross = Bᵀ (P S B)
    let mut gram = vec![0.0_f32; expr_dim * expr_dim];
    let mut cross = vec![0.0_f32; expr_dim * expr_dim];
    for (vertex, &mirrored) in symmetry_map.iter().enumerate() {
        for axis in 0..3_usize {
            // S negates the x component (axis 0) of the displacement.
            let sign = if axis == 0 { -1.0_f32 } else { 1.0_f32 };
            let row = (vertex * 3 + axis) * expr_dim;
            // P reads the displacement of the bilateral counterpart.
            let mirror_row = (mirrored * 3 + axis) * expr_dim;
            for i in 0..expr_dim {
                let b_i = expression_basis[row + i];
                if b_i == 0.0_f32 {
                    continue;
                }
                let signed_b_i = sign * b_i;
                for j in 0..expr_dim {
                    gram[i * expr_dim + j] += b_i * expression_basis[row + j];
                    cross[i * expr_dim + j] += signed_b_i * expression_basis[mirror_row + j];
                }
            }
        }
    }

    // Ridge term scaled to the basis energy so it stays negligible but rescues
    // a rank-deficient (or duplicated-component) basis.
    let trace: f32 = (0..expr_dim).map(|d| gram[d * expr_dim + d]).sum();
    let lambda = (trace / expr_dim as f32).abs() * 1e-6_f32 + f32::EPSILON;
    for d in 0..expr_dim {
        gram[d * expr_dim + d] += lambda;
    }

    let chol = cholesky_factor(&gram, expr_dim)?;

    // Solve gram · M[:, j] = cross[:, j] independently for every column j.
    let mut mirror = vec![0.0_f32; expr_dim * expr_dim];
    let mut rhs = vec![0.0_f32; expr_dim];
    for j in 0..expr_dim {
        for (i, slot) in rhs.iter_mut().enumerate() {
            *slot = cross[i * expr_dim + j];
        }
        let column = cholesky_solve(&chol, &rhs, expr_dim);
        for (i, &value) in column.iter().enumerate() {
            mirror[i * expr_dim + j] = value;
        }
    }
    Ok(mirror)
}

/// Mirror an expression state across the mid-sagittal plane.
///
/// The expression coefficients are transformed by the coefficient-space mirror
/// matrix produced by [`retar_build_expression_mirror_matrix`]:
/// `ψ' = M · ψ`.  A FLAME expression basis is a PCA basis over vertex
/// displacements ordered by explained variance — its components carry no
/// alternating symmetric/antisymmetric structure — so a mirror **cannot** be
/// expressed as a fixed sign pattern on the coefficients and genuinely requires
/// the basis geometry that `M` encodes.
///
/// The jaw pose is mirrored analytically: reflecting across `x = 0` conjugates
/// the jaw rotation by `diag(−1, 1, 1)`, which maps the axis-angle vector
/// `(rx, ry, rz)` to `(rx, −ry, −rz)`.
///
/// # Errors
///
/// Returns [`RetargetError::DimensionMismatch`] if `state.expression_params`
/// does not have `state.expr_dim` entries, or if `mirror_matrix` is not
/// `expr_dim × expr_dim`.
pub fn retar_mirror_expression(
    state: &ExpressionState,
    mirror_matrix: &[f32],
) -> Result<ExpressionState, RetargetError> {
    let dim = state.expr_dim;
    if state.expression_params.len() != dim {
        return Err(RetargetError::DimensionMismatch {
            src_dim: dim,
            got: state.expression_params.len(),
        });
    }
    if mirror_matrix.len() != dim * dim {
        return Err(RetargetError::DimensionMismatch {
            src_dim: dim * dim,
            got: mirror_matrix.len(),
        });
    }
    let mut params = vec![0.0_f32; dim];
    for (i, out) in params.iter_mut().enumerate() {
        let row = &mirror_matrix[i * dim..(i + 1) * dim];
        *out = row
            .iter()
            .zip(state.expression_params.iter())
            .map(|(&m, &v)| m * v)
            .sum();
    }
    Ok(ExpressionState {
        expression_params: params,
        jaw_pose: [state.jaw_pose[0], -state.jaw_pose[1], -state.jaw_pose[2]],
        expr_dim: dim,
    })
}
/// Weighted blend of multiple expression states.
///
/// Weights need not sum to 1; they are normalized internally.
/// Returns an error if `states` or `weights` are empty, or if their lengths
/// differ, or if any weight is negative.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn retar_blend_states(
    states: &[ExpressionState],
    weights: &[f32],
) -> Result<ExpressionState, RetargetError> {
    if states.is_empty() {
        return Err(RetargetError::EmptySequence);
    }
    if states.len() != weights.len() {
        return Err(RetargetError::DimensionMismatch {
            src_dim: states.len(),
            got: weights.len(),
        });
    }
    for &w in weights {
        if w < 0.0_f32 {
            return Err(RetargetError::InvalidWeight(w));
        }
    }
    let weight_sum: f32 = weights.iter().sum();
    if weight_sum < 1e-8_f32 {
        return Ok(ExpressionState::neutral(states[0].expr_dim));
    }
    let dim = states[0].expr_dim;
    let mut expr = vec![0.0_f32; dim];
    let mut jaw = [0.0_f32; 3];
    for (s, &w) in states.iter().zip(weights.iter()) {
        let nw = w / weight_sum;
        for (e, &v) in expr.iter_mut().zip(s.expression_params.iter()) {
            *e += nw * v;
        }
        jaw[0] += nw * s.jaw_pose[0];
        jaw[1] += nw * s.jaw_pose[1];
        jaw[2] += nw * s.jaw_pose[2];
    }
    Ok(ExpressionState {
        expression_params: expr,
        jaw_pose: jaw,
        expr_dim: dim,
    })
}
/// Spherical linear interpolation (SLERP) of two expression states.
///
/// The expression vector is treated as a single point in `expr_dim`-dimensional
/// space: its **direction** is interpolated along the great circle joining the
/// unit vectors of `a` and `b`, while its **magnitude** is interpolated
/// linearly between `‖a‖` and `‖b‖`.  This is one N-dimensional arc, *not* N
/// independent per-coefficient arcs — every output coefficient therefore
/// depends on every input coefficient.
///
/// Fallbacks:
///
/// - If either expression vector is near zero (`‖·‖ < 1e-8`) the whole vector
///   is linearly interpolated instead (the direction of a zero vector is
///   undefined).
/// - If the angle between the two directions is below `1e-6` rad the directions
///   are linearly interpolated and renormalized, which avoids dividing by
///   `sin θ ≈ 0`.
///
/// The jaw pose is **always** interpolated linearly, never slerped.
///
/// `t` is clamped to `[0, 1]`; `t = 0` reproduces `a` and `t = 1` reproduces `b`.
///
/// # Errors
///
/// Returns [`RetargetError::DimensionMismatch`] if `a` and `b` have different
/// expression dimensionality.
pub fn retar_slerp_states(
    a: &ExpressionState,
    b: &ExpressionState,
    t: f32,
) -> Result<ExpressionState, RetargetError> {
    if a.expr_dim != b.expr_dim {
        return Err(RetargetError::DimensionMismatch {
            src_dim: a.expr_dim,
            got: b.expr_dim,
        });
    }
    let t = t.clamp(0.0_f32, 1.0_f32);
    let na: f32 = a
        .expression_params
        .iter()
        .map(|&v| v * v)
        .sum::<f32>()
        .sqrt();
    let nb: f32 = b
        .expression_params
        .iter()
        .map(|&v| v * v)
        .sum::<f32>()
        .sqrt();
    let expr = if na < 1e-8_f32 || nb < 1e-8_f32 {
        a.expression_params
            .iter()
            .zip(b.expression_params.iter())
            .map(|(&av, &bv)| av + t * (bv - av))
            .collect()
    } else {
        let a_unit: Vec<f32> = a.expression_params.iter().map(|&v| v / na).collect();
        let b_unit: Vec<f32> = b.expression_params.iter().map(|&v| v / nb).collect();
        let cos_theta: f32 = a_unit
            .iter()
            .zip(b_unit.iter())
            .map(|(&x, &y)| x * y)
            .sum::<f32>()
            .clamp(-1.0_f32, 1.0_f32);
        let theta = cos_theta.acos();
        let magnitude = na + t * (nb - na);
        if theta.abs() < 1e-6_f32 {
            let dir: Vec<f32> = a_unit
                .iter()
                .zip(b_unit.iter())
                .map(|(&av, &bv)| av + t * (bv - av))
                .collect();
            let dn: f32 = dir.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-8_f32);
            dir.iter().map(|&v| v / dn * magnitude).collect()
        } else {
            let sin_theta = theta.sin();
            let wa = ((1.0_f32 - t) * theta).sin() / sin_theta;
            let wb = (t * theta).sin() / sin_theta;
            let dir: Vec<f32> = a_unit
                .iter()
                .zip(b_unit.iter())
                .map(|(&av, &bv)| wa * av + wb * bv)
                .collect();
            let dn: f32 = dir.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-8_f32);
            dir.iter().map(|&v| v / dn * magnitude).collect()
        }
    };
    let jaw = [
        a.jaw_pose[0] + t * (b.jaw_pose[0] - a.jaw_pose[0]),
        a.jaw_pose[1] + t * (b.jaw_pose[1] - a.jaw_pose[1]),
        a.jaw_pose[2] + t * (b.jaw_pose[2] - a.jaw_pose[2]),
    ];
    Ok(ExpressionState {
        expression_params: expr,
        jaw_pose: jaw,
        expr_dim: a.expr_dim,
    })
}
/// Compute retargeting statistics on a set of pairs.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn retar_compute_stats(
    retargeter: &LinearExpressionRetargeter,
    pairs: &[RetargetPair],
) -> Result<RetargetStats, RetargetError> {
    if pairs.is_empty() {
        return Err(RetargetError::EmptySequence);
    }
    let expr_dim = retargeter.config.expr_dim;
    let mut total_error = 0.0_f32;
    let mut max_error = 0.0_f32;
    let mut per_dim_abs: Vec<f32> = vec![0.0_f32; expr_dim];
    for pair in pairs {
        let predicted = retargeter.retarget(&pair.source)?;
        let err: f32 = predicted
            .expression_params
            .iter()
            .zip(pair.target.expression_params.iter())
            .map(|(&p, &t)| (p - t).powi(2))
            .sum::<f32>()
            .sqrt();
        total_error += err;
        if err > max_error {
            max_error = err;
        }
        for (d, (&p, &t)) in predicted
            .expression_params
            .iter()
            .zip(pair.target.expression_params.iter())
            .enumerate()
        {
            per_dim_abs[d] += (p - t).abs();
        }
    }
    let n = pairs.len() as f32;
    let mean_error = total_error / n;
    for v in &mut per_dim_abs {
        *v /= n;
    }
    let mapping_frobenius: f32 = retargeter
        .mapping
        .iter()
        .map(|&v| v * v)
        .sum::<f32>()
        .sqrt();
    Ok(RetargetStats {
        mean_error,
        max_error,
        per_dim_error: per_dim_abs,
        mapping_frobenius,
    })
}
/// Format retargeting statistics as a human-readable string.
#[must_use]
pub fn retar_format_stats(stats: &RetargetStats) -> String {
    format!(
        "RetargetStats {{ mean_error: {:.4}, max_error: {:.4}, mapping_frobenius: {:.4}, per_dim_dims: {} }}",
        stats.mean_error, stats.max_error, stats.mapping_frobenius, stats.per_dim_error
        .len(),
    )
}
/// Format a [`RetargetConfig`] as a human-readable string.
#[must_use]
pub fn retar_format_config(config: &RetargetConfig) -> String {
    format!(
        "RetargetConfig {{ expr_dim: {}, regularization: {:.2e}, scale_by_variance: {}, include_jaw: {}, smoothing_sigma: {:.1} }}",
        config.expr_dim, config.regularization, config.scale_by_variance, config
        .include_jaw, config.smoothing_sigma,
    )
}

// ---------------------------------------------------------------------------
// Regression tests for the mirror / smoothing / SLERP fixes
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-vertex, one-component basis whose single component is a pure
    /// x-displacement of magnitude `x0` on vertex 0 and `x1` on vertex 1.
    ///
    /// Layout matches `ndarray`'s `[N, 3, K]` row-major order with `N = 2`,
    /// `K = 1`: `((vertex * 3) + axis) * 1 + 0`.
    fn two_vertex_x_basis(x0: f32, x1: f32) -> Vec<f32> {
        vec![x0, 0.0, 0.0, x1, 0.0, 0.0]
    }

    #[test]
    fn mirror_matrix_negates_a_symmetric_x_component() {
        // Both vertices displace along +x by the same amount.  Reflecting x and
        // swapping the two vertices maps the field onto its own negative, so the
        // coefficient-space mirror must be −1.
        let basis = two_vertex_x_basis(1.0, 1.0);
        let symmetry_map = vec![1_usize, 0_usize];
        let mirror = retar_build_expression_mirror_matrix(&basis, &symmetry_map, 1)
            .expect("mirror matrix for a well-conditioned basis");
        assert_eq!(mirror.len(), 1);
        assert!(
            (mirror[0] + 1.0).abs() < 1e-4,
            "expected M ≈ -1, got {}",
            mirror[0]
        );
    }

    #[test]
    fn mirror_matrix_preserves_an_antisymmetric_x_component() {
        // Vertex 0 displaces +x, vertex 1 displaces −x.  Reflecting x and
        // swapping the vertices reproduces the field exactly, so M must be +1.
        let basis = two_vertex_x_basis(1.0, -1.0);
        let symmetry_map = vec![1_usize, 0_usize];
        let mirror = retar_build_expression_mirror_matrix(&basis, &symmetry_map, 1)
            .expect("mirror matrix for a well-conditioned basis");
        assert!(
            (mirror[0] - 1.0).abs() < 1e-4,
            "expected M ≈ +1, got {}",
            mirror[0]
        );
    }

    #[test]
    fn mirror_matrix_rejects_wrong_basis_length() {
        let symmetry_map = vec![1_usize, 0_usize];
        assert!(matches!(
            retar_build_expression_mirror_matrix(&[1.0, 0.0, 0.0], &symmetry_map, 1),
            Err(RetargetError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn mirror_matrix_rejects_out_of_bounds_symmetry_map() {
        let basis = two_vertex_x_basis(1.0, 1.0);
        assert!(matches!(
            retar_build_expression_mirror_matrix(&basis, &[9_usize, 0_usize], 1),
            Err(RetargetError::InvalidConfig(_))
        ));
    }

    #[test]
    fn mirror_expression_applies_the_matrix_and_flips_jaw_yaw_roll() {
        // Identity-free 2×2 matrix so a wrong (transposed) application is visible.
        let mirror = vec![0.0_f32, 1.0, 2.0, 0.0];
        let state = ExpressionState::from_params_and_jaw(vec![3.0, 5.0], [0.1, 0.2, 0.3]);
        let mirrored =
            retar_mirror_expression(&state, &mirror).expect("2×2 matrix on a 2-dim state");
        // row 0 = [0, 1] · [3, 5] = 5 ; row 1 = [2, 0] · [3, 5] = 6
        assert!((mirrored.expression_params[0] - 5.0).abs() < 1e-6);
        assert!((mirrored.expression_params[1] - 6.0).abs() < 1e-6);
        // Reflecting across x = 0 keeps the pitch axis and flips yaw and roll.
        assert!((mirrored.jaw_pose[0] - 0.1).abs() < 1e-6);
        assert!((mirrored.jaw_pose[1] + 0.2).abs() < 1e-6);
        assert!((mirrored.jaw_pose[2] + 0.3).abs() < 1e-6);
    }

    #[test]
    fn mirror_expression_twice_is_the_identity_for_a_real_basis() {
        let basis = two_vertex_x_basis(1.0, 1.0);
        let symmetry_map = vec![1_usize, 0_usize];
        let mirror =
            retar_build_expression_mirror_matrix(&basis, &symmetry_map, 1).expect("mirror matrix");
        let state = ExpressionState::from_params_and_jaw(vec![0.75], [0.1, 0.2, 0.3]);
        let once = retar_mirror_expression(&state, &mirror).expect("first mirror");
        let twice = retar_mirror_expression(&once, &mirror).expect("second mirror");
        assert!(
            (twice.expression_params[0] - 0.75).abs() < 1e-4,
            "double mirror should be the identity, got {}",
            twice.expression_params[0]
        );
        assert_eq!(twice.jaw_pose, [0.1, 0.2, 0.3]);
    }

    #[test]
    fn mirror_expression_rejects_wrong_matrix_size() {
        let state = ExpressionState::from_params(vec![1.0, 2.0]);
        assert!(matches!(
            retar_mirror_expression(&state, &[1.0, 0.0, 0.0]),
            Err(RetargetError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn smooth_sequence_survives_enormous_sigma() {
        // Before the half-width clamp this overflowed `2 * half + 1` (panic in
        // debug) or tried to allocate a kernel of astronomic length.
        let sequence = vec![
            ExpressionState::from_params(vec![0.0, 1.0]),
            ExpressionState::from_params(vec![1.0, 0.0]),
            ExpressionState::from_params(vec![2.0, -1.0]),
        ];
        let smoothed =
            retar_smooth_sequence(&sequence, 1e18_f32).expect("huge sigma must not panic");
        assert_eq!(smoothed.len(), 3);
        for frame in &smoothed {
            for value in &frame.expression_params {
                assert!(value.is_finite(), "smoothed value {value} is not finite");
            }
        }
    }

    #[test]
    fn smooth_sequence_rejects_non_finite_sigma() {
        let sequence = vec![ExpressionState::from_params(vec![0.0])];
        assert!(matches!(
            retar_smooth_sequence(&sequence, f32::NAN),
            Err(RetargetError::InvalidConfig(_))
        ));
        assert!(matches!(
            retar_smooth_sequence(&sequence, f32::INFINITY),
            Err(RetargetError::InvalidConfig(_))
        ));
    }

    #[test]
    fn smooth_sequence_moderate_sigma_still_smooths() {
        let sequence = vec![
            ExpressionState::from_params(vec![0.0]),
            ExpressionState::from_params(vec![10.0]),
            ExpressionState::from_params(vec![0.0]),
        ];
        let smoothed = retar_smooth_sequence(&sequence, 1.0_f32).expect("finite sigma");
        assert_eq!(smoothed.len(), 3);
        assert!(
            smoothed[1].expression_params[0] < 10.0,
            "the spike should be attenuated, got {}",
            smoothed[1].expression_params[0]
        );
    }

    #[test]
    fn slerp_endpoints_reproduce_the_inputs() {
        let a = ExpressionState::from_params_and_jaw(vec![1.0, 0.0], [0.0, 0.0, 0.0]);
        let b = ExpressionState::from_params_and_jaw(vec![0.0, 2.0], [1.0, 1.0, 1.0]);
        let at_zero = retar_slerp_states(&a, &b, 0.0).expect("slerp at t=0");
        let at_one = retar_slerp_states(&a, &b, 1.0).expect("slerp at t=1");
        assert!((at_zero.expression_params[0] - 1.0).abs() < 1e-5);
        assert!(at_zero.expression_params[1].abs() < 1e-5);
        assert!(at_one.expression_params[0].abs() < 1e-5);
        assert!((at_one.expression_params[1] - 2.0).abs() < 1e-5);
        // The jaw pose is linearly interpolated, never slerped.
        assert!((at_zero.jaw_pose[1]).abs() < 1e-6);
        assert!((at_one.jaw_pose[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn slerp_couples_all_coefficients() {
        // Documented behaviour: the whole vector shares one arc, so changing a
        // single coefficient of `a` moves *every* interpolated coefficient.
        let b = ExpressionState::from_params(vec![0.0, 1.0]);
        let a1 = ExpressionState::from_params(vec![1.0, 0.0]);
        let a2 = ExpressionState::from_params(vec![2.0, 0.0]);
        let mid1 = retar_slerp_states(&a1, &b, 0.5).expect("slerp");
        let mid2 = retar_slerp_states(&a2, &b, 0.5).expect("slerp");
        assert!(
            (mid1.expression_params[1] - mid2.expression_params[1]).abs() > 1e-4,
            "coefficient 1 should react to a change in coefficient 0"
        );
    }
}
