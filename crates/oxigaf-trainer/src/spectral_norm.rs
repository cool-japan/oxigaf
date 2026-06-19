//! Spectral normalization for controlling the Lipschitz constant of weight matrices.
//!
//! This module implements spectral normalization via power iteration, providing:
//! - Estimation of the largest singular value (spectral norm) of weight matrices
//! - Tracking of singular vectors across training steps (warm-start)
//! - Gradient Jacobian conditioning analysis
//! - Lipschitz constant bounds for sequences of matrix multiplications
//!
//! Spectral normalization constrains the Lipschitz constant of linear maps,
//! preventing gradient explosion and improving training stability for 3DGS avatars.
//!
//! # Example
//! ```rust
//! use oxigaf_trainer::spectral_norm::{spectral_norm, PowerIterationConfig};
//!
//! let matrix = vec![2.0f32, 0.0, 0.0, 3.0]; // 2×2 diagonal
//! let config = PowerIterationConfig::default();
//! let sigma = spectral_norm(&matrix, 2, 2, &config).unwrap();
//! assert!((sigma - 3.0).abs() < 0.01);
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by spectral normalization operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum SpectralNormError {
    /// Matrix element count doesn't match n_rows × n_cols.
    #[error("Matrix dimensions mismatch: {n_rows}×{n_cols} matrix has {actual} elements")]
    DimensionMismatch {
        n_rows: usize,
        n_cols: usize,
        actual: usize,
    },

    /// At least one dimension is zero.
    #[error("Empty matrix: n_rows={n_rows}, n_cols={n_cols}")]
    EmptyMatrix { n_rows: usize, n_cols: usize },

    /// A vector supplied to a function has the wrong length.
    #[error("Vector length mismatch: expected {expected}, got {actual}")]
    VectorLengthMismatch { expected: usize, actual: usize },

    /// Power iteration did not converge within the iteration budget.
    #[error("Power iteration failed to converge after {max_iter} iterations")]
    ConvergenceFailed { max_iter: usize },

    /// Matrices in a batch have inconsistent sizes.
    #[error("Batch size mismatch: matrices have different sizes")]
    BatchSizeMismatch,
}

// ─────────────────────────────────────────────────────────────────────────────
// Random number generation (no external crate)
// ─────────────────────────────────────────────────────────────────────────────

/// Xorshift64 PRNG — advances `state` and returns the new pseudo-random u64.
/// Ensures the state never becomes zero (the degenerate fixed point).
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state = (*state).max(1);
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the power iteration algorithm.
#[derive(Debug, Clone)]
pub struct PowerIterationConfig {
    /// Maximum number of iterations before declaring failure.
    pub max_iter: usize,
    /// Convergence tolerance: stop when relative change in sigma drops below this.
    pub tolerance: f32,
    /// Seed for the xorshift64 PRNG used to initialise the random starting vector.
    pub seed: u64,
}

impl Default for PowerIterationConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tolerance: 1e-6,
            seed: 42,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result types
// ─────────────────────────────────────────────────────────────────────────────

/// Estimated largest singular value together with the associated singular vectors.
#[derive(Debug, Clone)]
pub struct SingularValueEstimate {
    /// Estimated largest singular value.
    pub sigma: f32,
    /// Left singular vector (length n_rows).
    pub u: Vec<f32>,
    /// Right singular vector (length n_cols).
    pub v: Vec<f32>,
    /// Whether the iteration converged within the tolerance.
    pub converged: bool,
    /// Number of iterations actually performed.
    pub iterations: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Free-standing linear-algebra helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Row-major matrix–vector product: `y = A * x` where A is n_rows × n_cols.
///
/// Element (i, j) of A is stored at `matrix[i * n_cols + j]`.
pub fn mat_vec_mul(
    matrix: &[f32],
    n_rows: usize,
    n_cols: usize,
    x: &[f32],
) -> Result<Vec<f32>, SpectralNormError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(SpectralNormError::EmptyMatrix { n_rows, n_cols });
    }
    if matrix.len() != n_rows * n_cols {
        return Err(SpectralNormError::DimensionMismatch {
            n_rows,
            n_cols,
            actual: matrix.len(),
        });
    }
    if x.len() != n_cols {
        return Err(SpectralNormError::VectorLengthMismatch {
            expected: n_cols,
            actual: x.len(),
        });
    }

    let mut result = vec![0.0f32; n_rows];
    for (i, res) in result.iter_mut().enumerate() {
        let row_start = i * n_cols;
        *res = matrix[row_start..row_start + n_cols]
            .iter()
            .zip(x.iter())
            .map(|(&m, &xi)| m * xi)
            .sum();
    }
    Ok(result)
}

/// Transpose matrix–vector product: `y = A^T * x` where A is n_rows × n_cols.
///
/// The result has length n_cols.
pub fn mat_transpose_vec_mul(
    matrix: &[f32],
    n_rows: usize,
    n_cols: usize,
    x: &[f32],
) -> Result<Vec<f32>, SpectralNormError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(SpectralNormError::EmptyMatrix { n_rows, n_cols });
    }
    if matrix.len() != n_rows * n_cols {
        return Err(SpectralNormError::DimensionMismatch {
            n_rows,
            n_cols,
            actual: matrix.len(),
        });
    }
    if x.len() != n_rows {
        return Err(SpectralNormError::VectorLengthMismatch {
            expected: n_rows,
            actual: x.len(),
        });
    }

    let mut result = vec![0.0f32; n_cols];
    for (&xi, mat_row) in x.iter().zip(matrix.chunks(n_cols)) {
        for (res_j, &mij) in result.iter_mut().zip(mat_row.iter()) {
            *res_j += mij * xi;
        }
    }
    Ok(result)
}

/// L2 norm of a vector (returns 0 for an empty slice).
pub fn spectral_l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Normalise a vector to unit L2 norm in-place.
///
/// If the norm is below `1e-12` the vector is left unchanged (all-zero vectors
/// remain all-zero, avoiding division by near-zero).
pub fn spectral_normalize(v: &mut [f32]) {
    let norm = spectral_l2_norm(v);
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core power-iteration algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Run power iteration to estimate the largest singular value and singular vectors
/// of a row-major matrix.
///
/// # Algorithm
/// 1. Initialise `v` with a random unit vector from the xorshift64 PRNG.
/// 2. Repeatedly apply `u = A*v / ||A*v||` and `v = A^T*u / ||A^T*u||`.
/// 3. Estimate `sigma = u^T A v` and check for convergence.
pub fn power_iteration(
    matrix: &[f32],
    n_rows: usize,
    n_cols: usize,
    config: &PowerIterationConfig,
) -> Result<SingularValueEstimate, SpectralNormError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(SpectralNormError::EmptyMatrix { n_rows, n_cols });
    }
    if matrix.len() != n_rows * n_cols {
        return Err(SpectralNormError::DimensionMismatch {
            n_rows,
            n_cols,
            actual: matrix.len(),
        });
    }

    // Initialise random unit vector v
    let mut state = config.seed;
    let mut v = vec![0.0f32; n_cols];
    for vi in v.iter_mut() {
        *vi = (xorshift64(&mut state) % 10_000) as f32 / 10_000.0 - 0.5;
    }
    spectral_normalize(&mut v);

    let mut u = vec![0.0f32; n_rows];
    let mut sigma = 0.0f32;
    let mut converged = false;
    let mut iterations = 0usize;

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        // u = A * v, normalise
        u = mat_vec_mul(matrix, n_rows, n_cols, &v)?;
        spectral_normalize(&mut u);

        // v_new = A^T * u, normalise
        let mut v_new = mat_transpose_vec_mul(matrix, n_rows, n_cols, &u)?;
        spectral_normalize(&mut v_new);

        // sigma = u^T * (A * v_new)
        let av = mat_vec_mul(matrix, n_rows, n_cols, &v_new)?;
        let sigma_new: f32 = u.iter().zip(av.iter()).map(|(a, b)| a * b).sum();

        // Convergence check
        let rel_change = (sigma_new - sigma).abs() / (sigma.abs() + 1e-8);
        v = v_new;
        sigma = sigma_new;

        if rel_change < config.tolerance {
            converged = true;
            break;
        }
    }

    Ok(SingularValueEstimate {
        sigma,
        u,
        v,
        converged,
        iterations,
    })
}

/// Compute the spectral norm (largest singular value) of a row-major matrix.
pub fn spectral_norm(
    matrix: &[f32],
    n_rows: usize,
    n_cols: usize,
    config: &PowerIterationConfig,
) -> Result<f32, SpectralNormError> {
    let est = power_iteration(matrix, n_rows, n_cols, config)?;
    Ok(est.sigma)
}

/// Normalise a matrix by its spectral norm.
///
/// Returns `(normalised_matrix, sigma)`. If `sigma < 1e-8` the original matrix
/// is returned unchanged and `sigma` is reported as `0.0`.
pub fn normalize_by_spectral_norm(
    matrix: &[f32],
    n_rows: usize,
    n_cols: usize,
    config: &PowerIterationConfig,
) -> Result<(Vec<f32>, f32), SpectralNormError> {
    let sigma = spectral_norm(matrix, n_rows, n_cols, config)?;
    if sigma < 1e-8 {
        return Ok((matrix.to_vec(), 0.0));
    }
    let normalised: Vec<f32> = matrix.iter().map(|x| x / sigma).collect();
    Ok((normalised, sigma))
}

// ─────────────────────────────────────────────────────────────────────────────
// Single warm-start step
// ─────────────────────────────────────────────────────────────────────────────

/// One power-iteration step with warm-start singular vectors.
///
/// Returns `(new_u, new_v, new_sigma)`. Useful for tracking spectral norms
/// across training steps without running full iteration every step.
pub fn power_iteration_step(
    matrix: &[f32],
    n_rows: usize,
    n_cols: usize,
    u: &[f32],
    v: &[f32],
) -> Result<(Vec<f32>, Vec<f32>, f32), SpectralNormError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(SpectralNormError::EmptyMatrix { n_rows, n_cols });
    }
    if matrix.len() != n_rows * n_cols {
        return Err(SpectralNormError::DimensionMismatch {
            n_rows,
            n_cols,
            actual: matrix.len(),
        });
    }
    if u.len() != n_rows {
        return Err(SpectralNormError::VectorLengthMismatch {
            expected: n_rows,
            actual: u.len(),
        });
    }
    if v.len() != n_cols {
        return Err(SpectralNormError::VectorLengthMismatch {
            expected: n_cols,
            actual: v.len(),
        });
    }

    // u_new = A * v, normalise
    let mut u_new = mat_vec_mul(matrix, n_rows, n_cols, v)?;
    spectral_normalize(&mut u_new);

    // v_new = A^T * u_new, normalise
    let mut v_new = mat_transpose_vec_mul(matrix, n_rows, n_cols, &u_new)?;
    spectral_normalize(&mut v_new);

    // sigma = u_new^T * (A * v_new)
    let av_new = mat_vec_mul(matrix, n_rows, n_cols, &v_new)?;
    let sigma: f32 = u_new.iter().zip(av_new.iter()).map(|(a, b)| a * b).sum();

    Ok((u_new, v_new, sigma))
}

// ─────────────────────────────────────────────────────────────────────────────
// SpectralNormTracker
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks singular vectors across training steps to enable warm-start power iteration.
///
/// By reusing the previous singular vector estimates, a single iteration per training
/// step is usually sufficient to track the spectral norm accurately.
pub struct SpectralNormTracker {
    /// Number of rows in the tracked matrix.
    pub n_rows: usize,
    /// Number of columns in the tracked matrix.
    pub n_cols: usize,
    /// Cached left singular vector estimate.
    u: Vec<f32>,
    /// Cached right singular vector estimate.
    v: Vec<f32>,
    /// Power-iteration configuration (seed used only for resets).
    pub config: PowerIterationConfig,
    /// Most recently computed spectral norm estimate.
    pub last_sigma: f32,
}

impl SpectralNormTracker {
    /// Create a new tracker for matrices of shape n_rows × n_cols.
    ///
    /// The initial singular vectors are set to random unit vectors via the
    /// xorshift64 PRNG seeded from `config.seed`.
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        config: PowerIterationConfig,
    ) -> Result<Self, SpectralNormError> {
        if n_rows == 0 || n_cols == 0 {
            return Err(SpectralNormError::EmptyMatrix { n_rows, n_cols });
        }

        let mut state = config.seed;
        let mut u = vec![0.0f32; n_rows];
        for ui in u.iter_mut() {
            *ui = (xorshift64(&mut state) % 10_000) as f32 / 10_000.0 - 0.5;
        }
        spectral_normalize(&mut u);

        let mut v = vec![0.0f32; n_cols];
        for vi in v.iter_mut() {
            *vi = (xorshift64(&mut state) % 10_000) as f32 / 10_000.0 - 0.5;
        }
        spectral_normalize(&mut v);

        Ok(Self {
            n_rows,
            n_cols,
            u,
            v,
            config,
            last_sigma: 0.0,
        })
    }

    /// Update the tracker with a new matrix, performing a single warm-start step.
    ///
    /// Returns the updated spectral norm estimate.
    pub fn update(&mut self, matrix: &[f32]) -> Result<f32, SpectralNormError> {
        if matrix.len() != self.n_rows * self.n_cols {
            return Err(SpectralNormError::DimensionMismatch {
                n_rows: self.n_rows,
                n_cols: self.n_cols,
                actual: matrix.len(),
            });
        }

        let (new_u, new_v, sigma) =
            power_iteration_step(matrix, self.n_rows, self.n_cols, &self.u, &self.v)?;

        self.u = new_u;
        self.v = new_v;
        self.last_sigma = sigma;
        Ok(sigma)
    }

    /// Reset the cached singular vectors, forcing fresh estimation on the next `update`.
    pub fn reset(&mut self) {
        let mut state = self.config.seed.wrapping_add(0xDEAD_BEEF);
        for ui in self.u.iter_mut() {
            *ui = (xorshift64(&mut state) % 10_000) as f32 / 10_000.0 - 0.5;
        }
        spectral_normalize(&mut self.u);

        for vi in self.v.iter_mut() {
            *vi = (xorshift64(&mut state) % 10_000) as f32 / 10_000.0 - 0.5;
        }
        spectral_normalize(&mut self.v);

        self.last_sigma = 0.0;
    }

    /// Return the most recently computed spectral norm estimate.
    pub fn current_sigma(&self) -> f32 {
        self.last_sigma
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix norms and derived quantities
// ─────────────────────────────────────────────────────────────────────────────

/// Frobenius norm: `||A||_F = sqrt(sum_ij A_ij^2)`.
pub fn frobenius_norm(matrix: &[f32]) -> f32 {
    matrix.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Stable rank: `||A||_F^2 / ||A||_2^2`.
///
/// A value close to 1 indicates a rank-1 matrix; higher values indicate
/// a more uniformly distributed spectrum (effectively higher rank).
pub fn stable_rank(
    matrix: &[f32],
    n_rows: usize,
    n_cols: usize,
    config: &PowerIterationConfig,
) -> Result<f32, SpectralNormError> {
    let sigma = spectral_norm(matrix, n_rows, n_cols, config)?;
    if sigma < 1e-12 {
        return Ok(0.0);
    }
    let frob_sq = frobenius_norm(matrix).powi(2);
    Ok(frob_sq / (sigma * sigma))
}

// ─────────────────────────────────────────────────────────────────────────────
// Conditioning
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the condition number (sigma_max / sigma_min) of a matrix.
///
/// Uses the simplified lower-bound approximation: the smallest singular value
/// is lower-bounded by the minimum column norm. If any column has near-zero norm
/// the condition number is `f32::INFINITY`.
pub fn estimate_condition_number(
    matrix: &[f32],
    n_rows: usize,
    n_cols: usize,
    config: &PowerIterationConfig,
) -> Result<f32, SpectralNormError> {
    if n_rows == 0 || n_cols == 0 {
        return Err(SpectralNormError::EmptyMatrix { n_rows, n_cols });
    }
    if matrix.len() != n_rows * n_cols {
        return Err(SpectralNormError::DimensionMismatch {
            n_rows,
            n_cols,
            actual: matrix.len(),
        });
    }

    let sigma_max = spectral_norm(matrix, n_rows, n_cols, config)?;

    // Lower bound for sigma_min: minimum column L2 norm
    let mut min_col_norm = f32::INFINITY;
    for j in 0..n_cols {
        let col_norm_sq: f32 = (0..n_rows).map(|i| matrix[i * n_cols + j].powi(2)).sum();
        let col_norm = col_norm_sq.sqrt();
        if col_norm < min_col_norm {
            min_col_norm = col_norm;
        }
    }

    if min_col_norm < 1e-10 {
        return Ok(f32::INFINITY);
    }

    Ok(sigma_max / min_col_norm)
}

/// Returns `true` if the condition number of the matrix is below `threshold`.
pub fn is_well_conditioned(
    matrix: &[f32],
    n_rows: usize,
    n_cols: usize,
    threshold: f32,
    config: &PowerIterationConfig,
) -> Result<bool, SpectralNormError> {
    let cond = estimate_condition_number(matrix, n_rows, n_cols, config)?;
    Ok(cond.is_finite() && cond < threshold)
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch and sequence operations
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the spectral norm for each matrix in a batch (all must be n_rows × n_cols).
///
/// Returns a `Vec<f32>` with one sigma value per matrix.
pub fn batch_spectral_norm(
    matrices: &[Vec<f32>],
    n_rows: usize,
    n_cols: usize,
    config: &PowerIterationConfig,
) -> Result<Vec<f32>, SpectralNormError> {
    let expected_len = n_rows * n_cols;
    let mut sigmas = Vec::with_capacity(matrices.len());

    for mat in matrices {
        if mat.len() != expected_len {
            return Err(SpectralNormError::BatchSizeMismatch);
        }
        let sigma = spectral_norm(mat, n_rows, n_cols, config)?;
        sigmas.push(sigma);
    }

    Ok(sigmas)
}

/// Estimate an upper bound on the Lipschitz constant of a composition of linear maps.
///
/// For a product `A_k * ... * A_2 * A_1` the Lipschitz constant is bounded above
/// by the product of individual spectral norms.
///
/// `matrices` is a slice of `(matrix_data, n_rows, n_cols)` tuples, one per layer.
pub fn sequence_lipschitz_bound(
    matrices: &[(&[f32], usize, usize)],
    config: &PowerIterationConfig,
) -> Result<f32, SpectralNormError> {
    let mut bound = 1.0f32;
    for &(matrix, n_rows, n_cols) in matrices {
        let sigma = spectral_norm(matrix, n_rows, n_cols, config)?;
        bound *= sigma;
    }
    Ok(bound)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn default_config() -> PowerIterationConfig {
        PowerIterationConfig::default()
    }

    /// Absolute tolerance for floating-point comparisons in tests.
    const EPS: f32 = 1e-3;

    // ── mat_vec_mul ───────────────────────────────────────────────────────────

    #[test]
    fn test_mat_vec_mul_identity_2x2() {
        // I₂ * [1, 2]^T = [1, 2]^T
        let i2 = vec![1.0f32, 0.0, 0.0, 1.0];
        let x = vec![1.0f32, 2.0];
        let y = mat_vec_mul(&i2, 2, 2, &x).unwrap();
        assert_eq!(y, vec![1.0, 2.0]);
    }

    #[test]
    fn test_mat_vec_mul_3x2() {
        // [[1,2],[3,4],[5,6]] * [1,1]^T = [3,7,11]
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0f32, 1.0];
        let y = mat_vec_mul(&a, 3, 2, &x).unwrap();
        assert!((y[0] - 3.0).abs() < EPS);
        assert!((y[1] - 7.0).abs() < EPS);
        assert!((y[2] - 11.0).abs() < EPS);
    }

    #[test]
    fn test_mat_vec_mul_dimension_error_empty() {
        let err = mat_vec_mul(&[], 0, 2, &[1.0, 2.0]);
        assert!(matches!(err, Err(SpectralNormError::EmptyMatrix { .. })));
    }

    #[test]
    fn test_mat_vec_mul_dimension_mismatch() {
        // 2×2 matrix (4 elements) but only 3 elements provided
        let err = mat_vec_mul(&[1.0f32, 2.0, 3.0], 2, 2, &[1.0, 2.0]);
        assert!(matches!(
            err,
            Err(SpectralNormError::DimensionMismatch { .. })
        ));
    }

    // ── mat_transpose_vec_mul ────────────────────────────────────────────────

    #[test]
    fn test_mat_transpose_vec_mul_identity_2x2() {
        // I₂^T * [1, 2]^T = [1, 2]^T
        let i2 = vec![1.0f32, 0.0, 0.0, 1.0];
        let x = vec![1.0f32, 2.0];
        let y = mat_transpose_vec_mul(&i2, 2, 2, &x).unwrap();
        assert_eq!(y, vec![1.0, 2.0]);
    }

    #[test]
    fn test_mat_transpose_vec_mul_3x2() {
        // [[1,2],[3,4],[5,6]]^T * [1,1,1]^T = [9,12]
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0f32, 1.0, 1.0];
        let y = mat_transpose_vec_mul(&a, 3, 2, &x).unwrap();
        assert!((y[0] - 9.0).abs() < EPS);
        assert!((y[1] - 12.0).abs() < EPS);
    }

    #[test]
    fn test_mat_transpose_vec_mul_empty_error() {
        let err = mat_transpose_vec_mul(&[], 0, 3, &[]);
        assert!(matches!(err, Err(SpectralNormError::EmptyMatrix { .. })));
    }

    // ── spectral_l2_norm ─────────────────────────────────────────────────────

    #[test]
    fn test_spectral_l2_norm_zero_vec() {
        assert_eq!(spectral_l2_norm(&[0.0f32, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_spectral_l2_norm_unit_vec() {
        assert!((spectral_l2_norm(&[1.0f32, 0.0, 0.0]) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_spectral_l2_norm_known() {
        // [3, 4] -> 5
        assert!((spectral_l2_norm(&[3.0f32, 4.0]) - 5.0).abs() < EPS);
    }

    // ── spectral_normalize ───────────────────────────────────────────────────

    #[test]
    fn test_spectral_normalize_unit_unchanged() {
        let mut v = vec![1.0f32, 0.0, 0.0];
        spectral_normalize(&mut v);
        assert!((spectral_l2_norm(&v) - 1.0).abs() < EPS);
        assert!((v[0] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_spectral_normalize_scales_to_unit() {
        let mut v = vec![3.0f32, 4.0];
        spectral_normalize(&mut v);
        assert!((spectral_l2_norm(&v) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_spectral_normalize_zero_vector() {
        // All-zero should remain all-zero (no divide by near-zero)
        let mut v = vec![0.0f32, 0.0, 0.0];
        spectral_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    // ── power_iteration ──────────────────────────────────────────────────────

    #[test]
    fn test_power_iteration_identity_2x2() {
        // σ_max(I₂) = 1
        let i2 = vec![1.0f32, 0.0, 0.0, 1.0];
        let est = power_iteration(&i2, 2, 2, &default_config()).unwrap();
        assert!((est.sigma - 1.0).abs() < 0.01, "sigma={}", est.sigma);
    }

    #[test]
    fn test_power_iteration_diagonal_max_eigenvalue() {
        // diag(2, 5, 3) → σ_max = 5
        let d = vec![2.0f32, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 3.0];
        let est = power_iteration(&d, 3, 3, &default_config()).unwrap();
        assert!((est.sigma - 5.0).abs() < 0.01, "sigma={}", est.sigma);
    }

    #[test]
    fn test_power_iteration_singular_vectors_unit_norm() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let est = power_iteration(&a, 2, 2, &default_config()).unwrap();
        assert!((spectral_l2_norm(&est.u) - 1.0).abs() < EPS);
        assert!((spectral_l2_norm(&est.v) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_power_iteration_empty_error() {
        let err = power_iteration(&[], 0, 3, &default_config());
        assert!(matches!(err, Err(SpectralNormError::EmptyMatrix { .. })));
    }

    #[test]
    fn test_power_iteration_dimension_mismatch_error() {
        let err = power_iteration(&[1.0f32, 2.0], 2, 2, &default_config());
        assert!(matches!(
            err,
            Err(SpectralNormError::DimensionMismatch { .. })
        ));
    }

    // ── spectral_norm ────────────────────────────────────────────────────────

    #[test]
    fn test_spectral_norm_rotation_2x2() {
        // Rotation matrix has σ_max = 1
        let theta: f32 = 0.7;
        let rot = vec![theta.cos(), -theta.sin(), theta.sin(), theta.cos()];
        let sigma = spectral_norm(&rot, 2, 2, &default_config()).unwrap();
        assert!((sigma - 1.0).abs() < 0.01, "sigma={}", sigma);
    }

    #[test]
    fn test_spectral_norm_scaling_matrix() {
        // diag(4, 1) → σ_max = 4
        let s = vec![4.0f32, 0.0, 0.0, 1.0];
        let sigma = spectral_norm(&s, 2, 2, &default_config()).unwrap();
        assert!((sigma - 4.0).abs() < 0.05, "sigma={}", sigma);
    }

    #[test]
    fn test_spectral_norm_identity_3x3() {
        let i3: Vec<f32> = (0..9).map(|k| if k % 4 == 0 { 1.0 } else { 0.0 }).collect();
        let sigma = spectral_norm(&i3, 3, 3, &default_config()).unwrap();
        assert!((sigma - 1.0).abs() < 0.01, "sigma={}", sigma);
    }

    #[test]
    fn test_spectral_norm_rectangular() {
        // 1×2 matrix [3, 4] → σ_max = 5
        let a = vec![3.0f32, 4.0];
        let sigma = spectral_norm(&a, 1, 2, &default_config()).unwrap();
        assert!((sigma - 5.0).abs() < 0.01, "sigma={}", sigma);
    }

    // ── normalize_by_spectral_norm ───────────────────────────────────────────

    #[test]
    fn test_normalize_by_spectral_norm_gives_unit_sigma() {
        let a = vec![2.0f32, 0.0, 0.0, 3.0];
        let (norm_a, sigma) = normalize_by_spectral_norm(&a, 2, 2, &default_config()).unwrap();
        assert!((sigma - 3.0).abs() < 0.05, "sigma={}", sigma);
        // σ_max of the normalised matrix should be ~1
        let sigma_check = spectral_norm(&norm_a, 2, 2, &default_config()).unwrap();
        assert!(
            (sigma_check - 1.0).abs() < 0.05,
            "sigma_check={}",
            sigma_check
        );
    }

    #[test]
    fn test_normalize_by_spectral_norm_zero_matrix() {
        let zeros = vec![0.0f32; 4];
        let (out, sigma) = normalize_by_spectral_norm(&zeros, 2, 2, &default_config()).unwrap();
        assert_eq!(sigma, 0.0);
        assert_eq!(out, zeros);
    }

    #[test]
    fn test_normalize_by_spectral_norm_unchanged_when_sigma_zero() {
        let zeros = vec![0.0f32; 9];
        let (out, _) = normalize_by_spectral_norm(&zeros, 3, 3, &default_config()).unwrap();
        for v in &out {
            assert_eq!(*v, 0.0);
        }
    }

    // ── power_iteration_step ─────────────────────────────────────────────────

    #[test]
    fn test_power_iteration_step_basic() {
        // After one step from a good warm start, sigma should be close to max
        let a = vec![3.0f32, 0.0, 0.0, 1.0]; // diag(3,1) → σ=3
        let u = vec![1.0f32, 0.0];
        let v = vec![1.0f32, 0.0];
        let (new_u, new_v, sigma) = power_iteration_step(&a, 2, 2, &u, &v).unwrap();
        assert!((spectral_l2_norm(&new_u) - 1.0).abs() < EPS);
        assert!((spectral_l2_norm(&new_v) - 1.0).abs() < EPS);
        assert!((sigma - 3.0).abs() < 0.1, "sigma={}", sigma);
    }

    #[test]
    fn test_power_iteration_step_improves_estimate() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let config = default_config();
        let full = power_iteration(&a, 2, 2, &config).unwrap();
        // Starting from the converged u/v, another step should keep sigma stable
        let (_, _, sigma2) = power_iteration_step(&a, 2, 2, &full.u, &full.v).unwrap();
        assert!((sigma2 - full.sigma).abs() / (full.sigma + 1e-8) < 0.01);
    }

    #[test]
    fn test_power_iteration_step_vector_length_error() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2×2
        let u_bad = vec![1.0f32]; // wrong length
        let v = vec![1.0f32, 0.0];
        let err = power_iteration_step(&a, 2, 2, &u_bad, &v);
        assert!(matches!(
            err,
            Err(SpectralNormError::VectorLengthMismatch { .. })
        ));
    }

    // ── frobenius_norm ───────────────────────────────────────────────────────

    #[test]
    fn test_frobenius_norm_zeros() {
        assert_eq!(frobenius_norm(&[0.0f32; 9]), 0.0);
    }

    #[test]
    fn test_frobenius_norm_identity_2x2() {
        // ||I₂||_F = sqrt(2)
        let i2 = vec![1.0f32, 0.0, 0.0, 1.0];
        assert!((frobenius_norm(&i2) - 2.0f32.sqrt()).abs() < EPS);
    }

    #[test]
    fn test_frobenius_norm_known() {
        // [3, 4] → 5
        assert!((frobenius_norm(&[3.0f32, 4.0]) - 5.0).abs() < EPS);
    }

    // ── stable_rank ──────────────────────────────────────────────────────────

    #[test]
    fn test_stable_rank_full_rank_identity() {
        // ||I_n||_F^2 / σ_max^2 = n / 1 = n
        let i3: Vec<f32> = (0..9).map(|k| if k % 4 == 0 { 1.0 } else { 0.0 }).collect();
        let sr = stable_rank(&i3, 3, 3, &default_config()).unwrap();
        // Expected ≈ 3 (within tolerance due to power iteration)
        assert!(sr > 2.5 && sr < 3.5, "stable_rank={}", sr);
    }

    #[test]
    fn test_stable_rank_rank1_matrix() {
        // Rank-1 matrix: ||A||_F^2 = σ_max^2 → stable_rank ≈ 1
        let a = vec![3.0f32, 0.0, 0.0, 0.0]; // 2×2 rank-1
        let sr = stable_rank(&a, 2, 2, &default_config()).unwrap();
        assert!((sr - 1.0).abs() < 0.05, "stable_rank={}", sr);
    }

    #[test]
    fn test_stable_rank_zero_matrix() {
        let zeros = vec![0.0f32; 4];
        let sr = stable_rank(&zeros, 2, 2, &default_config()).unwrap();
        assert_eq!(sr, 0.0);
    }

    // ── batch_spectral_norm ──────────────────────────────────────────────────

    #[test]
    fn test_batch_spectral_norm_three_matrices() {
        let config = default_config();
        let m1 = vec![1.0f32, 0.0, 0.0, 1.0]; // σ=1
        let m2 = vec![2.0f32, 0.0, 0.0, 2.0]; // σ=2
        let m3 = vec![3.0f32, 0.0, 0.0, 3.0]; // σ=3
        let sigmas = batch_spectral_norm(&[m1, m2, m3], 2, 2, &config).unwrap();
        assert_eq!(sigmas.len(), 3);
        assert!((sigmas[0] - 1.0).abs() < 0.05, "s0={}", sigmas[0]);
        assert!((sigmas[1] - 2.0).abs() < 0.05, "s1={}", sigmas[1]);
        assert!((sigmas[2] - 3.0).abs() < 0.05, "s2={}", sigmas[2]);
    }

    #[test]
    fn test_batch_spectral_norm_empty_batch() {
        let sigmas = batch_spectral_norm(&[], 2, 2, &default_config()).unwrap();
        assert!(sigmas.is_empty());
    }

    #[test]
    fn test_batch_spectral_norm_size_mismatch() {
        let m1 = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2 = 4 elements
        let m2 = vec![1.0f32, 0.0, 0.0]; // 3 elements — wrong
        let err = batch_spectral_norm(&[m1, m2], 2, 2, &default_config());
        assert!(matches!(err, Err(SpectralNormError::BatchSizeMismatch)));
    }

    // ── sequence_lipschitz_bound ─────────────────────────────────────────────

    #[test]
    fn test_sequence_lipschitz_bound_product_of_rotations() {
        // Product of two rotation matrices: Lipschitz bound = 1 × 1 = 1
        let theta: f32 = 0.5;
        let rot = vec![theta.cos(), -theta.sin(), theta.sin(), theta.cos()];
        let config = default_config();
        let bound = sequence_lipschitz_bound(&[(&rot, 2, 2), (&rot, 2, 2)], &config).unwrap();
        // Bound should be ≥ 1.0 (product of two sigmas each ≈ 1)
        assert!(bound >= 0.98, "bound={}", bound);
    }

    #[test]
    fn test_sequence_lipschitz_bound_single_matrix() {
        let a = vec![2.0f32, 0.0, 0.0, 2.0]; // σ=2
        let config = default_config();
        let bound = sequence_lipschitz_bound(&[(&a, 2, 2)], &config).unwrap();
        assert!((bound - 2.0).abs() < 0.05, "bound={}", bound);
    }

    #[test]
    fn test_sequence_lipschitz_bound_empty_sequence() {
        let bound = sequence_lipschitz_bound(&[], &default_config()).unwrap();
        assert_eq!(bound, 1.0); // empty product = 1
    }

    // ── SpectralNormTracker ──────────────────────────────────────────────────

    #[test]
    fn test_tracker_new_valid() {
        let tracker = SpectralNormTracker::new(3, 4, default_config());
        assert!(tracker.is_ok());
        let t = tracker.unwrap();
        assert_eq!(t.n_rows, 3);
        assert_eq!(t.n_cols, 4);
        assert_eq!(t.last_sigma, 0.0);
    }

    #[test]
    fn test_tracker_new_empty_error() {
        let err = SpectralNormTracker::new(0, 4, default_config());
        assert!(err.is_err());
    }

    #[test]
    fn test_tracker_update_converges() {
        let mut tracker = SpectralNormTracker::new(2, 2, default_config()).unwrap();
        // Run a few warm-start updates on diag(5, 2)
        let a = vec![5.0f32, 0.0, 0.0, 2.0];
        for _ in 0..20 {
            tracker.update(&a).unwrap();
        }
        let sigma = tracker.current_sigma();
        assert!((sigma - 5.0).abs() < 0.2, "sigma={}", sigma);
    }

    #[test]
    fn test_tracker_reset_clears_sigma() {
        let mut tracker = SpectralNormTracker::new(2, 2, default_config()).unwrap();
        let a = vec![5.0f32, 0.0, 0.0, 2.0];
        tracker.update(&a).unwrap();
        assert!(tracker.current_sigma() > 0.0);
        tracker.reset();
        assert_eq!(tracker.current_sigma(), 0.0);
    }

    // ── is_well_conditioned ──────────────────────────────────────────────────

    #[test]
    fn test_is_well_conditioned_identity() {
        // Identity is perfectly conditioned (cond = 1)
        let i3: Vec<f32> = (0..9).map(|k| if k % 4 == 0 { 1.0 } else { 0.0 }).collect();
        let result = is_well_conditioned(&i3, 3, 3, 10.0, &default_config()).unwrap();
        assert!(result);
    }

    #[test]
    fn test_is_well_conditioned_ill_conditioned() {
        // diag(1000, 1) → condition ≈ 1000
        let a = vec![1000.0f32, 0.0, 0.0, 1.0];
        let result = is_well_conditioned(&a, 2, 2, 10.0, &default_config()).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_is_well_conditioned_threshold_boundary() {
        // Near-identity: cond should be 1.0 → well-conditioned for threshold=2
        let a = vec![1.0f32, 0.0, 0.0, 1.0];
        let result = is_well_conditioned(&a, 2, 2, 2.0, &default_config()).unwrap();
        assert!(result);
    }

    // ── estimate_condition_number ─────────────────────────────────────────────

    #[test]
    fn test_estimate_condition_number_identity() {
        let i2 = vec![1.0f32, 0.0, 0.0, 1.0];
        let cond = estimate_condition_number(&i2, 2, 2, &default_config()).unwrap();
        // Both σ_max and min_col_norm = 1, so cond ≈ 1
        assert!((cond - 1.0).abs() < 0.05, "cond={}", cond);
    }

    #[test]
    fn test_estimate_condition_number_scaled_diagonal() {
        // diag(6, 2) → σ_max ≈ 6, min_col_norm = 2, cond ≈ 3
        let a = vec![6.0f32, 0.0, 0.0, 2.0];
        let cond = estimate_condition_number(&a, 2, 2, &default_config()).unwrap();
        assert!((cond - 3.0).abs() < 0.1, "cond={}", cond);
    }

    #[test]
    fn test_estimate_condition_number_singular_infinity() {
        // Singular matrix: one column all zeros → cond = INFINITY
        let a = vec![1.0f32, 0.0, 2.0, 0.0]; // column 1 is [0, 0]
        let cond = estimate_condition_number(&a, 2, 2, &default_config()).unwrap();
        assert!(cond.is_infinite(), "cond={}", cond);
    }

    // ── Additional edge-case tests ────────────────────────────────────────────

    #[test]
    fn test_spectral_norm_1x1() {
        let a = vec![7.0f32];
        let sigma = spectral_norm(&a, 1, 1, &default_config()).unwrap();
        assert!((sigma - 7.0).abs() < 0.05, "sigma={}", sigma);
    }

    #[test]
    fn test_spectral_l2_norm_empty_slice() {
        assert_eq!(spectral_l2_norm(&[]), 0.0);
    }

    #[test]
    fn test_power_iteration_step_empty_error() {
        let err = power_iteration_step(&[], 0, 2, &[], &[1.0, 2.0]);
        assert!(matches!(err, Err(SpectralNormError::EmptyMatrix { .. })));
    }

    #[test]
    fn test_batch_spectral_norm_single_matrix() {
        let m = vec![1.0f32, 0.0, 0.0, 1.0];
        let sigmas = batch_spectral_norm(&[m], 2, 2, &default_config()).unwrap();
        assert_eq!(sigmas.len(), 1);
        assert!((sigmas[0] - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_tracker_warm_start_fewer_iterations_than_cold() {
        // Verify warm-start converges fast: after 5 warm-start steps sigma is close
        let a = vec![4.0f32, 0.0, 0.0, 1.0]; // σ_max = 4
        let mut tracker = SpectralNormTracker::new(2, 2, default_config()).unwrap();
        for _ in 0..5 {
            tracker.update(&a).unwrap();
        }
        assert!(
            (tracker.current_sigma() - 4.0).abs() < 0.5,
            "sigma={}",
            tracker.current_sigma()
        );
    }

    #[test]
    fn test_normalize_by_spectral_norm_rotation() {
        // Rotation matrix already has σ=1, so normalised ≈ original
        let theta: f32 = 1.2;
        let rot = vec![theta.cos(), -theta.sin(), theta.sin(), theta.cos()];
        let (norm_rot, sigma) = normalize_by_spectral_norm(&rot, 2, 2, &default_config()).unwrap();
        assert!((sigma - 1.0).abs() < 0.01, "sigma={}", sigma);
        for (a, b) in rot.iter().zip(norm_rot.iter()) {
            assert!((a - b).abs() < 0.05, "element mismatch a={} b={}", a, b);
        }
    }
}
