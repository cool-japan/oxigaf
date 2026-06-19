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
/// `sigma` is the smoothing strength in frames.  Kernel half-width is
/// `ceil(3 * sigma)`.  Boundary frames are padded by repetition (clamp).
///
/// Returns [`RetargetError::EmptySequence`] if `sequence` is empty.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn retar_smooth_sequence(
    sequence: &[ExpressionState],
    sigma: f32,
) -> Result<Vec<ExpressionState>, RetargetError> {
    if sequence.is_empty() {
        return Err(RetargetError::EmptySequence);
    }
    if sigma <= 0.0_f32 {
        return Ok(sequence.to_vec());
    }
    let half = (3.0_f32 * sigma).ceil() as i64;
    let n = i64::try_from(sequence.len()).unwrap_or(i64::MAX);
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
/// Mirror an expression by negating odd-indexed expression dimensions.
///
/// FLAME expression dimensions alternate between bilateral-symmetric components;
/// negating the odd-indexed ones performs a left-right reflection.
#[must_use]
pub fn retar_mirror_expression(state: &ExpressionState) -> ExpressionState {
    let params: Vec<f32> = state
        .expression_params
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 2 == 1 { -v } else { v })
        .collect();
    ExpressionState {
        expression_params: params,
        jaw_pose: state.jaw_pose,
        expr_dim: state.expr_dim,
    }
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
/// Component-wise spherical linear interpolation (SLERP) of two expression states.
///
/// Each expression coefficient is interpolated on a 1-D arc. `t = 0` returns
/// a clone of `a`; `t = 1` returns a clone of `b`.
///
/// # Errors
///
/// Returns an error if the operation fails.
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
