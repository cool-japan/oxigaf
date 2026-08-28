//! Statistical Shape Model (SSM) for FLAME head shapes.
//!
//! Implements PCA-based statistical shape modelling: learn principal modes of
//! variation from a training set of head meshes, then fit or generate novel
//! shapes within the learned subspace.
//!
//! Shapes are stored as flat `Vec<f32>` of length `n_vertices * 3`
//! (interleaved x, y, z coordinates).

use std::fmt::Write as FmtWrite;

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the Statistical Shape Model routines.
#[derive(Debug, Error)]
pub enum SsmError {
    /// Training set has fewer than 2 shapes.
    #[error("not enough training shapes: need at least 2, got {0}")]
    NotEnoughShapes(usize),
    /// Two shapes in the training set have different vertex counts.
    #[error("dimension mismatch: shapes have different vertex counts ({a} vs {b})")]
    DimensionMismatch { a: usize, b: usize },
    /// Requested component count is zero or otherwise invalid.
    #[error("invalid number of components: {0}")]
    InvalidComponents(usize),
    /// A component index is out of range.
    #[error("component index out of bounds: {index} >= {n_components}")]
    ComponentOutOfBounds { index: usize, n_components: usize },
    /// Matrix is singular or power iteration failed to converge.
    #[error("singular matrix or non-convergence")]
    Singular,
    /// Empty input was supplied where non-empty data was required.
    #[error("empty input")]
    EmptyInput,
}

// ─────────────────────────────────────────────────────────────────────────────
// PRNG (no rand crate)
// ─────────────────────────────────────────────────────────────────────────────

/// Xorshift64 step.  `state` must not be zero (guarded internally).
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

/// Draw a uniform f32 in [0, 1) from the xorshift64 PRNG.
///
/// Built directly from the mantissa bits of a `[1.0, 2.0)`-range float
/// (top 23 bits of the xorshift output become the mantissa, exponent fixed
/// at 127), then shifted down by 1.0. A `u64 as f32` division (the
/// previous implementation) rounds `u64::MAX` itself up to `2^64` in f32,
/// so a draw whose top ~41 bits are all set divides to exactly `1.0` --
/// this construction can never round up to `1.0`.
#[inline]
fn xorshift_f32(state: &mut u64) -> f32 {
    let bits = xorshift64(state);
    let mantissa = (bits >> 41) as u32;
    let float_bits: u32 = 0x3f80_0000u32 | mantissa;
    f32::from_bits(float_bits) - 1.0_f32
}

/// Draw one sample from the standard normal distribution `N(0, 1)` via the
/// Box-Muller transform, using two draws from [`xorshift_f32`].
#[inline]
fn standard_normal(state: &mut u64) -> f32 {
    use std::f32::consts::PI;
    // `xorshift_f32` can legitimately return exactly 0.0 (when its top 23
    // bits happen to all be zero); floor it away from zero so `ln(u1)`
    // never produces -inf (which would otherwise make this -inf/NaN).
    let u1 = xorshift_f32(state).max(1e-30);
    let u2 = xorshift_f32(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

// ─────────────────────────────────────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────────────────────────────────────

/// PCA-based statistical shape model.
///
/// Shapes are represented as flat `Vec<f32>` of length `n_vertices * 3`
/// (`x0, y0, z0, x1, y1, z1, …`).
#[derive(Debug, Clone)]
pub struct StatisticalShapeModel {
    /// Number of training shapes used to build the model.
    pub n_shapes: usize,
    /// Number of vertices per shape.
    pub n_vertices: usize,
    /// Mean shape: length `n_vertices * 3`.
    pub mean_shape: Vec<f32>,
    /// Principal component directions (row-major): `[n_components][n_vertices*3]`.
    /// Each row has unit L2 norm.
    pub components: Vec<Vec<f32>>,
    /// Explained variance (eigenvalues): length `n_components`.
    pub variances: Vec<f32>,
    /// Cumulative explained variance ratio (relative to `total_variance`,
    /// the TRUE total variance of the training data -- not merely the sum
    /// of the retained `variances`): length `n_components`.
    pub explained_ratio: Vec<f32>,
    /// Total variance of the (centered) training data, i.e. the sum of
    /// ALL non-trivial eigenvalues of the data's covariance, not just the
    /// `n_components` that were retained. `explained_ratio` is normalized
    /// against this, so `explained_ratio.last()` reaches `1.0` only when
    /// every true mode of variation was retained.
    pub total_variance: f32,
}

/// Shape coefficients in the principal-component subspace.
#[derive(Debug, Clone)]
pub struct ShapeParameters {
    /// Projection weights; length `n_components`.
    pub params: Vec<f32>,
    /// Number of vertices in the associated model.
    pub n_vertices: usize,
}

impl ShapeParameters {
    /// Clamp each coefficient to `±n_sigma * sqrt(variance[k])`.
    #[must_use]
    pub fn clamp_to_sigma(&self, model: &StatisticalShapeModel, n_sigma: f32) -> ShapeParameters {
        let clamped = self
            .params
            .iter()
            .zip(model.variances.iter())
            .map(|(&p, &v)| {
                let bound = n_sigma * v.max(0.0).sqrt();
                p.clamp(-bound, bound)
            })
            .collect();
        ShapeParameters {
            params: clamped,
            n_vertices: self.n_vertices,
        }
    }

    /// L2 norm of the parameter vector (Mahalanobis-like distance from mean).
    #[must_use]
    pub fn mahalanobis_distance(&self, _model: &StatisticalShapeModel) -> f32 {
        self.params.iter().map(|&p| p * p).sum::<f32>().sqrt()
    }

    /// Normalised parameters: `params[k] / sqrt(variance[k])` (z-scores).
    ///
    /// Returns [`SsmError::Singular`] if any variance is non-positive.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn z_scores(&self, model: &StatisticalShapeModel) -> Result<Vec<f32>, SsmError> {
        self.params
            .iter()
            .zip(model.variances.iter())
            .map(|(&p, &v)| {
                if v <= 0.0 {
                    Err(SsmError::Singular)
                } else {
                    Ok(p / v.sqrt())
                }
            })
            .collect()
    }
}

/// Summary statistics for a fitted `StatisticalShapeModel`.
#[derive(Debug, Clone)]
pub struct SsmStats {
    /// Number of training shapes.
    pub n_shapes: usize,
    /// Number of vertices per shape.
    pub n_vertices: usize,
    /// Number of principal components.
    pub n_components: usize,
    /// True total variance of the training data (sum of ALL non-trivial
    /// eigenvalues of its covariance, not just the retained components --
    /// see [`StatisticalShapeModel::total_variance`]).
    pub total_variance: f32,
    /// Per-component explained variance ratio.
    pub explained_variance_ratio: Vec<f32>,
    /// Cumulative explained variance ratio.
    pub cumulative_explained_ratio: Vec<f32>,
    /// Mean RMSE over the training shapes.
    pub mean_reconstruction_error: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// L2 norm of a slice.
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Normalise a vector in-place; return `false` if the norm is effectively zero.
#[inline]
fn normalize_inplace(v: &mut [f32]) -> bool {
    let n = l2_norm(v);
    if n < 1e-30 {
        return false;
    }
    for x in v.iter_mut() {
        *x /= n;
    }
    true
}

/// Build the Gram matrix `G[i][j] = dot(c[i], c[j]) / n_shapes`.
/// Returns a flat row-major `n × n` matrix.
fn build_gram_matrix(centered: &[Vec<f32>], n_shapes: usize) -> Vec<f32> {
    let n = n_shapes;
    let mut g = vec![0.0f32; n * n];
    let scale = 1.0 / n as f32;
    for i in 0..n {
        for j in i..n {
            let v = dot(&centered[i], &centered[j]) * scale;
            g[i * n + j] = v;
            g[j * n + i] = v;
        }
    }
    g
}

/// Power-iteration to find the dominant eigenvector of the symmetric matrix `g`
/// (flat row-major, size `n × n`).
///
/// Returns `(eigenvalue, eigenvector)` or `SsmError::Singular` if it does not converge.
fn power_iteration(g: &[f32], n: usize, seed: u64) -> Result<(f32, Vec<f32>), SsmError> {
    const MAX_ITER: usize = 200;
    const TOL: f32 = 1e-8;

    // Initialise with a deterministic pseudo-random unit vector.
    let mut rng = seed;
    let mut v: Vec<f32> = (0..n).map(|_| xorshift_f32(&mut rng) * 2.0 - 1.0).collect();
    if !normalize_inplace(&mut v) {
        // All-zero init — fall back to standard basis.
        v[0] = 1.0;
    }

    let mut lambda = 0.0f32;
    for _ in 0..MAX_ITER {
        // w = G * v  (dense matvec)
        let mut w = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += g[i * n + j] * v[j];
            }
        }
        let lambda_new = l2_norm(&w);
        if lambda_new < 1e-12 {
            // Eigenvalue is numerically zero; signal via Singular.
            return Err(SsmError::Singular);
        }
        let w_norm: Vec<f32> = w.iter().map(|&x| x / lambda_new).collect();
        if (lambda_new - lambda).abs() < TOL {
            return Ok((lambda_new, w_norm));
        }
        lambda = lambda_new;
        v = w_norm;
    }
    Ok((lambda, v))
}

/// Deflate the Gram matrix: `G ← G - λ * g_k * g_k^T`.
fn deflate(g: &mut [f32], n: usize, lambda: f32, evec: &[f32]) {
    for i in 0..n {
        for j in 0..n {
            g[i * n + j] -= lambda * evec[i] * evec[j];
        }
    }
}

/// Map a Gram-space eigenvector `g_k` back to vertex space (without normalising).
///
/// `v_k = X^T g_k`  where `X` rows are the centered shapes.
/// Normalisation is deferred so that Gram-Schmidt can run before the final
/// re-normalisation step.
fn gram_to_vertex_space_raw(centered: &[Vec<f32>], g_evec: &[f32], dim: usize) -> Vec<f32> {
    let n_shapes = centered.len();
    let mut v = vec![0.0f32; dim];
    for k in 0..n_shapes {
        let coef = g_evec[k];
        for d in 0..dim {
            v[d] += coef * centered[k][d];
        }
    }
    v
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API – building the model
// ─────────────────────────────────────────────────────────────────────────────

/// Build a `StatisticalShapeModel` from a collection of training shapes via PCA.
///
/// Each element of `shapes` is a flat `Vec<f32>` of length `n_vertices * 3`.
/// `n_components` controls how many principal components to retain;
/// it is capped at `n_shapes - 1` automatically.
///
/// PCA uses the dual (Gram-matrix) formulation so that complexity scales with
/// the number of training shapes rather than with the (much larger) vertex
/// dimension.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn ssm_build(
    shapes: &[Vec<f32>],
    n_components: usize,
) -> Result<StatisticalShapeModel, SsmError> {
    if shapes.len() < 2 {
        return Err(SsmError::NotEnoughShapes(shapes.len()));
    }
    if n_components == 0 {
        return Err(SsmError::InvalidComponents(0));
    }

    let n_shapes = shapes.len();
    let dim = shapes[0].len();
    if dim == 0 {
        return Err(SsmError::EmptyInput);
    }

    // Validate uniform dimensionality.
    for (_i, s) in shapes.iter().enumerate().skip(1) {
        if s.len() != dim {
            return Err(SsmError::DimensionMismatch { a: dim, b: s.len() });
        }
    }

    let n_vertices = dim / 3;

    // 1. Mean shape.
    let mut mean_shape = vec![0.0f32; dim];
    for s in shapes {
        for (m, &x) in mean_shape.iter_mut().zip(s.iter()) {
            *m += x;
        }
    }
    let inv_n = 1.0 / n_shapes as f32;
    for m in &mut mean_shape {
        *m *= inv_n;
    }

    // 2. Centred shapes.
    let centered: Vec<Vec<f32>> = shapes
        .iter()
        .map(|s| {
            s.iter()
                .zip(mean_shape.iter())
                .map(|(&x, &m)| x - m)
                .collect()
        })
        .collect();

    // 3. Gram matrix G[i,j] = dot(c_i, c_j) / n_shapes.
    let mut gram = build_gram_matrix(&centered, n_shapes);

    // True total variance of the training data = trace(Gram) = mean
    // squared norm of the centered shapes. The dual (Gram-matrix) and
    // primal (covariance-matrix) PCA formulations share the same nonzero
    // eigenvalues, so trace(Gram) equals the sum of ALL of them -- not
    // just the `n_components` retained below. Must be captured now, before
    // deflation mutates `gram`.
    let total_variance: f32 = (0..n_shapes).map(|i| gram[i * n_shapes + i]).sum();

    // 4. Power iteration + deflation to extract top-k eigenpairs.
    let max_components = (n_shapes - 1).min(n_components);
    let mut components: Vec<Vec<f32>> = Vec::with_capacity(max_components);
    let mut variances: Vec<f32> = Vec::with_capacity(max_components);

    // Adaptive threshold: first eigenvalue determines the scale.
    let mut lambda_first = 0.0f32;

    for k in 0..max_components {
        let seed = (k as u64) + 1; // non-zero seed
        match power_iteration(&gram, n_shapes, seed) {
            Ok((lambda, g_evec)) => {
                // Hard eigenvalue threshold: stop if nearly zero.
                if lambda < 1e-10 {
                    break;
                }
                // Relative threshold once we know the scale of the first eigenvalue.
                if k == 0 {
                    lambda_first = lambda;
                } else if lambda < 1e-8 * lambda_first {
                    break;
                }

                // Map back to vertex space.
                let mut pc = gram_to_vertex_space_raw(&centered, &g_evec, dim);

                // Gram-Schmidt orthogonalisation against previously extracted PCs.
                for prev in &components {
                    let proj = dot(&pc, prev);
                    for (p, &q) in pc.iter_mut().zip(prev.iter()) {
                        *p -= proj * q;
                    }
                }
                // If the residual is negligible, this direction is a duplicate — skip it.
                if !normalize_inplace(&mut pc) {
                    // Deflate anyway so the next iteration sees a clean residual.
                    deflate(&mut gram, n_shapes, lambda, &g_evec);
                    continue;
                }

                // Deflate Gram matrix.
                deflate(&mut gram, n_shapes, lambda, &g_evec);
                components.push(pc);
                variances.push(lambda.max(0.0));
            }
            Err(SsmError::Singular) => {
                // Remaining eigenvalues are numerically zero — stop early.
                break;
            }
            Err(e) => return Err(e),
        }
    }

    if components.is_empty() {
        return Err(SsmError::Singular);
    }

    // 5. Explained variance ratio (cumulative), relative to the TRUE total
    // variance -- not the sum of retained `variances` -- so it correctly
    // reads < 1.0 whenever fewer components than true modes are kept.
    let explained_ratio: Vec<f32> = if total_variance > 1e-20 {
        let mut cum = 0.0f32;
        variances
            .iter()
            .map(|&v| {
                cum += v;
                (cum / total_variance).min(1.0)
            })
            .collect()
    } else {
        vec![1.0f32; variances.len()]
    };

    Ok(StatisticalShapeModel {
        n_shapes,
        n_vertices,
        mean_shape,
        components,
        variances,
        explained_ratio,
        total_variance,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Reconstruction, projection, fitting
// ─────────────────────────────────────────────────────────────────────────────

/// Reconstruct a shape from its shape parameters.
///
/// `shape = mean + Σ_k params[k] * components[k]`
///
/// `params` must have length equal to `model.components.len()`.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn ssm_reconstruct(
    model: &StatisticalShapeModel,
    params: &[f32],
) -> Result<Vec<f32>, SsmError> {
    let n_comp = model.components.len();
    if params.len() != n_comp {
        return Err(SsmError::ComponentOutOfBounds {
            index: params.len(),
            n_components: n_comp,
        });
    }

    let dim = model.n_vertices * 3;
    let mut shape = model.mean_shape.clone();
    for (k, &p) in params.iter().enumerate() {
        if p == 0.0 {
            continue;
        }
        let comp = &model.components[k];
        for (s, &c) in shape.iter_mut().zip(comp.iter()) {
            *s += p * c;
        }
    }
    // Ensure the output length matches dim (handles edge cases).
    shape.truncate(dim);
    Ok(shape)
}

/// Project a target shape onto the principal subspace.
///
/// Computes: `params[k] = dot(target - mean, components[k])`.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn ssm_project(
    model: &StatisticalShapeModel,
    target_shape: &[f32],
) -> Result<ShapeParameters, SsmError> {
    let dim = model.n_vertices * 3;
    if target_shape.len() != dim {
        return Err(SsmError::DimensionMismatch {
            a: dim,
            b: target_shape.len(),
        });
    }

    // Centred target.
    let centered: Vec<f32> = target_shape
        .iter()
        .zip(model.mean_shape.iter())
        .map(|(&t, &m)| t - m)
        .collect();

    let params = model
        .components
        .iter()
        .map(|comp| dot(&centered, comp))
        .collect();

    Ok(ShapeParameters {
        params,
        n_vertices: model.n_vertices,
    })
}

/// Fit a target shape to the model by projecting onto the subspace and
/// reconstructing.  Returns the best approximation within the SSM subspace.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn ssm_fit(model: &StatisticalShapeModel, target_shape: &[f32]) -> Result<Vec<f32>, SsmError> {
    let sp = ssm_project(model, target_shape)?;
    ssm_reconstruct(model, &sp.params)
}

/// Compute the RMSE between `target_shape` and its SSM approximation.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn ssm_reconstruction_error(
    model: &StatisticalShapeModel,
    target_shape: &[f32],
) -> Result<f32, SsmError> {
    let fitted = ssm_fit(model, target_shape)?;
    let n = fitted.len() as f32;
    if n == 0.0 {
        return Err(SsmError::EmptyInput);
    }
    let mse: f32 = fitted
        .iter()
        .zip(target_shape.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum::<f32>()
        / n;
    Ok(mse.sqrt())
}

// ─────────────────────────────────────────────────────────────────────────────
// Shape generation and interpolation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a random shape by sampling the model's learned Gaussian in
/// coefficient space, `c_k ~ N(0, variance[k])`, clamped to `±n_sigma *
/// sqrt(variance[k])`.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn ssm_random_shape(
    model: &StatisticalShapeModel,
    n_sigma: f32,
    rng_state: &mut u64,
) -> Result<Vec<f32>, SsmError> {
    let params: Vec<f32> = model
        .variances
        .iter()
        .map(|&v| {
            let std = v.max(0.0).sqrt();
            // `.abs()` keeps the bound valid (min <= max) regardless of
            // `n_sigma`'s sign, matching the previous implementation's
            // tolerance of a negative `n_sigma`.
            let bound = (n_sigma * std).abs();
            let z = standard_normal(rng_state);
            (z * std).clamp(-bound, bound)
        })
        .collect();
    ssm_reconstruct(model, &params)
}

/// Linearly interpolate between two shapes in the SSM subspace.
///
/// Both shapes are projected, their coefficients are linearly interpolated, and
/// the result is reconstructed.  `t = 0` → `shape_a`, `t = 1` → `shape_b`.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn ssm_interpolate(
    model: &StatisticalShapeModel,
    shape_a: &[f32],
    shape_b: &[f32],
    t: f32,
) -> Result<Vec<f32>, SsmError> {
    let pa = ssm_project(model, shape_a)?;
    let pb = ssm_project(model, shape_b)?;
    let params: Vec<f32> = pa
        .params
        .iter()
        .zip(pb.params.iter())
        .map(|(&a, &b)| a * (1.0 - t) + b * t)
        .collect();
    ssm_reconstruct(model, &params)
}

/// Generate `n_steps` shapes uniformly sweeping component `component` from
/// `-n_sigma * sqrt(var[component])` to `+n_sigma * sqrt(var[component])`,
/// with all other components set to zero.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn ssm_component_sweep(
    model: &StatisticalShapeModel,
    component: usize,
    n_sigma: f32,
    n_steps: usize,
) -> Result<Vec<Vec<f32>>, SsmError> {
    let n_comp = model.components.len();
    if component >= n_comp {
        return Err(SsmError::ComponentOutOfBounds {
            index: component,
            n_components: n_comp,
        });
    }
    if n_steps == 0 {
        return Ok(Vec::new());
    }

    let std = model.variances[component].max(0.0).sqrt();
    let range = n_sigma * std;

    (0..n_steps)
        .map(|i| {
            let t = if n_steps == 1 {
                0.0
            } else {
                (i as f32) / (n_steps as f32 - 1.0) * 2.0 - 1.0 // [-1, +1]
            };
            let mut params = vec![0.0f32; n_comp];
            params[component] = t * range;
            ssm_reconstruct(model, &params)
        })
        .collect()
}

/// Number of components required to explain at least `target_ratio` of variance
/// (0.0–1.0).  Returns 0 for `target_ratio ≤ 0` and `n_components` for
/// `target_ratio ≥ 1` (or when the full explained ratio never reaches the
/// target).
#[must_use]
pub fn ssm_components_for_variance(model: &StatisticalShapeModel, target_ratio: f32) -> usize {
    if target_ratio <= 0.0 {
        return 0;
    }
    for (k, &cum) in model.explained_ratio.iter().enumerate() {
        if cum >= target_ratio {
            return k + 1;
        }
    }
    model.components.len()
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics and analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Compute SSM summary statistics including mean reconstruction error on the
/// supplied training shapes.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn ssm_compute_stats(
    model: &StatisticalShapeModel,
    training_shapes: &[Vec<f32>],
) -> Result<SsmStats, SsmError> {
    if training_shapes.is_empty() {
        return Err(SsmError::EmptyInput);
    }

    // The model's TRUE total variance (not the sum of retained
    // `variances` -- see `StatisticalShapeModel::total_variance`).
    let total_variance = model.total_variance;
    let n_components = model.components.len();

    // Per-component ratio (not cumulative).
    let per_ratio: Vec<f32> = if total_variance > 0.0 {
        model
            .variances
            .iter()
            .map(|&v| v / total_variance)
            .collect()
    } else {
        vec![0.0f32; n_components]
    };

    let mut mean_err = 0.0f32;
    let mut count = 0usize;
    for s in training_shapes {
        if s.len() == model.n_vertices * 3 {
            if let Ok(e) = ssm_reconstruction_error(model, s) {
                mean_err += e;
                count += 1;
            }
        }
    }
    if count > 0 {
        mean_err /= count as f32;
    }

    Ok(SsmStats {
        n_shapes: model.n_shapes,
        n_vertices: model.n_vertices,
        n_components,
        total_variance,
        explained_variance_ratio: per_ratio,
        cumulative_explained_ratio: model.explained_ratio.clone(),
        mean_reconstruction_error: mean_err,
    })
}

/// Compute vertex-wise standard deviation across a set of shapes.
///
/// Returns a `Vec<f32>` of length `n_vertices * 3` where each element is the
/// standard deviation across shapes for that coordinate.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn ssm_vertex_std(shapes: &[Vec<f32>]) -> Result<Vec<f32>, SsmError> {
    if shapes.is_empty() {
        return Err(SsmError::EmptyInput);
    }

    let dim = shapes[0].len();
    for (_i, s) in shapes.iter().enumerate().skip(1) {
        if s.len() != dim {
            return Err(SsmError::DimensionMismatch { a: dim, b: s.len() });
        }
    }

    let n = shapes.len() as f32;
    // Compute mean.
    let mut mean = vec![0.0f32; dim];
    for s in shapes {
        for (m, &x) in mean.iter_mut().zip(s.iter()) {
            *m += x;
        }
    }
    for m in &mut mean {
        *m /= n;
    }

    // Compute variance then sqrt.
    let mut var = vec![0.0f32; dim];
    for s in shapes {
        for (v, (&x, &m)) in var.iter_mut().zip(s.iter().zip(mean.iter())) {
            let d = x - m;
            *v += d * d;
        }
    }
    for v in &mut var {
        *v = (*v / n).sqrt();
    }

    Ok(var)
}

/// Return the top-`k` most variable vertices as `(vertex_index, total_std)` pairs,
/// sorted in descending order by total displacement.
///
/// The "total std" for vertex `i` is `sqrt(std_x² + std_y² + std_z²)` where
/// `std_{x,y,z}` come from the PCA component vectors (each column of the
/// components matrix summed in quadrature).
#[must_use]
pub fn ssm_most_variable_vertices(model: &StatisticalShapeModel, k: usize) -> Vec<(usize, f32)> {
    let n_v = model.n_vertices;
    // Compute per-vertex variability from component vectors weighted by variance.
    let mut vertex_var = vec![0.0f32; n_v];
    for (comp, &var) in model.components.iter().zip(model.variances.iter()) {
        for vi in 0..n_v {
            let dx = comp[vi * 3];
            let dy = comp[vi * 3 + 1];
            let dz = comp[vi * 3 + 2];
            vertex_var[vi] += var * (dx * dx + dy * dy + dz * dz);
        }
    }
    let mut pairs: Vec<(usize, f32)> = vertex_var
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.sqrt()))
        .collect();

    // Sort descending.
    pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(k);
    pairs
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Format `SsmStats` as a human-readable multi-line string.
#[must_use]
pub fn ssm_format_stats(stats: &SsmStats) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "SSM Stats: {} shapes, {} vertices, {} components",
        stats.n_shapes, stats.n_vertices, stats.n_components
    );
    let _ = writeln!(s, "  Total variance: {:.6}", stats.total_variance);
    let _ = writeln!(
        s,
        "  Mean reconstruction error (training): {:.6}",
        stats.mean_reconstruction_error
    );
    for (k, (&r, &cum)) in stats
        .explained_variance_ratio
        .iter()
        .zip(stats.cumulative_explained_ratio.iter())
        .enumerate()
    {
        let _ = writeln!(
            s,
            "  PC {:3}: {:.4} explained  ({:.4} cumulative)",
            k + 1,
            r,
            cum
        );
    }
    s
}

/// Format a `StatisticalShapeModel` as a human-readable summary string.
#[must_use]
pub fn ssm_format_model(model: &StatisticalShapeModel) -> String {
    let n_comp = model.components.len();
    let cum_last = model.explained_ratio.last().copied().unwrap_or(0.0);
    format!(
        "StatisticalShapeModel {{ n_shapes: {}, n_vertices: {}, n_components: {}, \
         explained_ratio_last: {:.4}, variances_sum: {:.6} }}",
        model.n_shapes,
        model.n_vertices,
        n_comp,
        cum_last,
        model.variances.iter().sum::<f32>(),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Build a simple set of rank-1 training shapes: `n_shapes` shapes with
    /// `n_vertices` vertices each.  Shape `i` = mean + coeff[i] * mode,
    /// where `mode` is a fixed direction in vertex space.
    ///
    /// This data lives in a 1-dimensional subspace; `ssm_build` will return
    /// exactly 1 principal component.
    fn make_shapes(n_shapes: usize, n_vertices: usize) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
        let dim = n_vertices * 3;
        // Mean shape: constant 1.0 per coordinate.
        let mean: Vec<f32> = (0..dim).map(|i| 1.0 + (i as f32) * 0.01).collect();
        // Primary mode: uniform unit vector.
        let mut mode = vec![0.0f32; dim];
        let norm = (dim as f32).sqrt();
        for m in &mut mode {
            *m = 1.0 / norm;
        }
        // Coefficients spread uniformly around zero.
        let shapes: Vec<Vec<f32>> = (0..n_shapes)
            .map(|i| {
                let coeff = (i as f32) - (n_shapes as f32 - 1.0) * 0.5;
                mean.iter()
                    .zip(mode.iter())
                    .map(|(&m, &d)| m + coeff * d)
                    .collect()
            })
            .collect();
        (shapes, mean, mode)
    }

    /// Build a set of *full-rank* pseudo-random training shapes using xorshift64.
    ///
    /// Use this helper when tests need multiple meaningful principal components
    /// (orthogonality, variance ordering with multiple PCs, round-trip accuracy,
    /// random-shape bounds, etc.).
    fn make_random_shapes(n_shapes: usize, n_vertices: usize, seed: u64) -> Vec<Vec<f32>> {
        let dim = n_vertices * 3;
        let mut rng = seed;
        (0..n_shapes)
            .map(|_| (0..dim).map(|_| xorshift_f32(&mut rng)).collect())
            .collect()
    }

    // ── SsmError display ─────────────────────────────────────────────────────

    #[test]
    fn test_error_not_enough_shapes_display() {
        let e = SsmError::NotEnoughShapes(1);
        assert!(!e.to_string().is_empty());
        assert!(e.to_string().contains('1'));
    }

    #[test]
    fn test_error_dimension_mismatch_display() {
        let e = SsmError::DimensionMismatch { a: 10, b: 20 };
        assert!(!e.to_string().is_empty());
        assert!(e.to_string().contains("10"));
        assert!(e.to_string().contains("20"));
    }

    #[test]
    fn test_error_invalid_components_display() {
        let e = SsmError::InvalidComponents(0);
        assert!(!e.to_string().is_empty());
        assert!(e.to_string().contains('0'));
    }

    #[test]
    fn test_error_component_out_of_bounds_display() {
        let e = SsmError::ComponentOutOfBounds {
            index: 5,
            n_components: 3,
        };
        let s = e.to_string();
        assert!(!s.is_empty());
        assert!(s.contains('5'));
        assert!(s.contains('3'));
    }

    #[test]
    fn test_error_singular_display() {
        let e = SsmError::Singular;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_error_empty_input_display() {
        let e = SsmError::EmptyInput;
        assert!(!e.to_string().is_empty());
    }

    // ── ssm_build errors ─────────────────────────────────────────────────────

    #[test]
    fn test_build_not_enough_shapes() {
        let shapes = vec![vec![1.0f32; 9]];
        assert!(matches!(
            ssm_build(&shapes, 1),
            Err(SsmError::NotEnoughShapes(1))
        ));
    }

    #[test]
    fn test_build_zero_shapes() {
        let shapes: Vec<Vec<f32>> = vec![];
        assert!(matches!(
            ssm_build(&shapes, 1),
            Err(SsmError::NotEnoughShapes(0))
        ));
    }

    #[test]
    fn test_build_zero_components() {
        let shapes = vec![vec![1.0f32; 9], vec![2.0f32; 9]];
        assert!(matches!(
            ssm_build(&shapes, 0),
            Err(SsmError::InvalidComponents(0))
        ));
    }

    #[test]
    fn test_build_dimension_mismatch() {
        let shapes = vec![vec![1.0f32; 9], vec![1.0f32; 12]];
        assert!(matches!(
            ssm_build(&shapes, 1),
            Err(SsmError::DimensionMismatch { a: 9, b: 12 })
        ));
    }

    #[test]
    fn test_build_identical_shapes_zero_variance() {
        let shape = vec![1.0f32; 9];
        let shapes = vec![shape.clone(), shape.clone(), shape.clone()];
        // With identical shapes the Gram matrix has zero eigenvalues → should error
        // or return no valid components.
        let result = ssm_build(&shapes, 2);
        // Either Singular or very small variances.
        match result {
            Err(SsmError::Singular) => {}
            Ok(model) => {
                for &v in &model.variances {
                    assert!(v < 1e-10, "variance should be ~0 for identical shapes");
                }
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    // ── ssm_build success ────────────────────────────────────────────────────

    #[test]
    fn test_build_5_shapes_3_components() {
        let (shapes, _mean, _mode) = make_shapes(5, 10);
        let model = ssm_build(&shapes, 3).expect("build failed");
        assert_eq!(model.n_shapes, 5);
        assert_eq!(model.n_vertices, 10);
        // At most min(n_shapes-1, requested) = min(4, 3) = 3 components.
        assert!(model.components.len() <= 3);
        assert_eq!(model.mean_shape.len(), 30);
    }

    #[test]
    fn test_build_n_components_capped_at_n_shapes_minus_1() {
        let (shapes, _, _) = make_shapes(4, 5);
        let model = ssm_build(&shapes, 10).expect("build failed");
        // Max usable components = n_shapes - 1 = 3.
        assert!(model.components.len() <= 3);
    }

    #[test]
    fn test_build_variances_non_increasing() {
        let (shapes, _, _) = make_shapes(8, 6);
        let model = ssm_build(&shapes, 5).expect("build failed");
        for w in model.variances.windows(2) {
            assert!(
                w[0] >= w[1] - 1e-5,
                "variances not non-increasing: {} < {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_build_components_unit_norm() {
        let (shapes, _, _) = make_shapes(6, 8);
        let model = ssm_build(&shapes, 4).expect("build failed");
        for (k, comp) in model.components.iter().enumerate() {
            let n = l2_norm(comp);
            assert!(
                (n - 1.0).abs() < 1e-5,
                "component {k} has norm {n:.6}, expected ~1"
            );
        }
    }

    #[test]
    fn test_build_components_orthogonal() {
        // Full-rank random data ensures multiple meaningful PCs.
        let shapes = make_random_shapes(8, 10, 42);
        let model = ssm_build(&shapes, 6).expect("build failed");
        let n = model.components.len();
        assert!(n >= 2, "need at least 2 PCs to test orthogonality, got {n}");
        for i in 0..n {
            for j in (i + 1)..n {
                let d = dot(&model.components[i], &model.components[j]);
                assert!(
                    d.abs() < 1e-4,
                    "PC {i} and PC {j} not orthogonal: dot={d:.6}"
                );
            }
        }
    }

    #[test]
    fn test_build_explained_ratio_monotone() {
        let (shapes, _, _) = make_shapes(5, 7);
        let model = ssm_build(&shapes, 4).expect("build failed");
        for w in model.explained_ratio.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "explained_ratio not non-decreasing");
        }
    }

    #[test]
    fn test_build_explained_ratio_last_leq_1() {
        let (shapes, _, _) = make_shapes(6, 5);
        let model = ssm_build(&shapes, 5).expect("build failed");
        let last = *model.explained_ratio.last().expect("non-empty");
        assert!(last <= 1.0 + 1e-5, "cumulative ratio > 1: {last}");
    }

    // Regression test: 11 full-rank random shapes span up to n_shapes-1=10
    // non-trivial modes. Retaining only 3 components must NOT claim to
    // explain ~100% of the variance (the old bug normalized against the
    // sum of RETAINED eigenvalues only, so `explained_ratio.last()` was
    // always exactly 1.0 regardless of how many true modes existed).
    #[test]
    fn test_build_explained_ratio_reflects_true_total_variance() {
        let shapes = make_random_shapes(11, 20, 123);
        let model = ssm_build(&shapes, 3).expect("build failed");
        assert_eq!(model.components.len(), 3);
        let last = *model.explained_ratio.last().expect("non-empty");
        assert!(
            last < 0.99,
            "3 of ~10 true modes should not explain ~100% of variance: {last}"
        );
        let retained: f32 = model.variances.iter().sum();
        assert!(
            model.total_variance > retained + 1e-6,
            "total_variance ({}) should exceed retained variance ({retained}) \
             when fewer components than true modes are kept",
            model.total_variance
        );
    }

    #[test]
    fn test_build_mean_shape_correct() {
        let n_shapes = 5;
        let n_vertices = 4;
        let dim = n_vertices * 3;
        let shapes: Vec<Vec<f32>> = (0..n_shapes)
            .map(|i| (0..dim).map(|d| (i * dim + d) as f32).collect())
            .collect();
        let model = ssm_build(&shapes, 2).expect("build");
        let expected_mean: Vec<f32> = (0..dim)
            .map(|d| shapes.iter().map(|s| s[d]).sum::<f32>() / n_shapes as f32)
            .collect();
        for (a, b) in model.mean_shape.iter().zip(expected_mean.iter()) {
            assert!((a - b).abs() < 1e-5, "mean mismatch: {a} vs {b}");
        }
    }

    // ── ssm_reconstruct ───────────────────────────────────────────────────────

    #[test]
    fn test_reconstruct_zero_params_gives_mean() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        let n_comp = model.components.len();
        let params = vec![0.0f32; n_comp];
        let rec = ssm_reconstruct(&model, &params).expect("reconstruct");
        for (a, b) in rec.iter().zip(model.mean_shape.iter()) {
            assert!((a - b).abs() < 1e-6, "expected mean shape");
        }
    }

    #[test]
    fn test_reconstruct_wrong_params_length() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        let params = vec![0.0f32; 99];
        assert!(matches!(
            ssm_reconstruct(&model, &params),
            Err(SsmError::ComponentOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_reconstruct_output_length() {
        let (shapes, _, _) = make_shapes(5, 8);
        let model = ssm_build(&shapes, 3).expect("build");
        let n_comp = model.components.len();
        let params = vec![1.0f32; n_comp];
        let rec = ssm_reconstruct(&model, &params).expect("reconstruct");
        assert_eq!(rec.len(), model.n_vertices * 3);
    }

    // ── ssm_project ───────────────────────────────────────────────────────────

    #[test]
    fn test_project_mean_shape_zero_params() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        let sp = ssm_project(&model, &model.mean_shape.clone()).expect("project");
        for &p in &sp.params {
            assert!(p.abs() < 1e-5, "mean shape should project to ~0, got {p}");
        }
    }

    #[test]
    fn test_project_wrong_length() {
        let (shapes, _, _) = make_shapes(4, 5);
        let model = ssm_build(&shapes, 2).expect("build");
        let wrong = vec![0.0f32; 9999];
        assert!(matches!(
            ssm_project(&model, &wrong),
            Err(SsmError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_project_output_length() {
        let (shapes, _, _) = make_shapes(5, 7);
        let model = ssm_build(&shapes, 3).expect("build");
        let sp = ssm_project(&model, &shapes[0]).expect("project");
        assert_eq!(sp.params.len(), model.components.len());
        assert_eq!(sp.n_vertices, model.n_vertices);
    }

    // ── ssm_fit & ssm_reconstruction_error ───────────────────────────────────

    #[test]
    fn test_fit_mean_shape_low_error() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 4).expect("build");
        let err = ssm_reconstruction_error(&model, &model.mean_shape.clone()).expect("error");
        assert!(err < 1e-5, "RMSE of mean shape should be ~0, got {err}");
    }

    #[test]
    fn test_fit_training_shape_low_error() {
        // Use full-rank random data with n_components = n_shapes - 1 = 4.
        // All training shapes should be recoverable with low RMSE.
        let shapes = make_random_shapes(5, 6, 7);
        let model = ssm_build(&shapes, 4).expect("build");
        for (i, s) in shapes.iter().enumerate() {
            let err = ssm_reconstruction_error(&model, s).expect("error");
            assert!(
                err < 0.05,
                "training shape {i} reconstruction RMSE = {err:.6}"
            );
        }
    }

    #[test]
    fn test_fit_output_length() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        let fitted = ssm_fit(&model, &shapes[0]).expect("fit");
        assert_eq!(fitted.len(), model.n_vertices * 3);
    }

    #[test]
    fn test_reconstruction_error_mean_shape_near_zero() {
        let (shapes, _, _) = make_shapes(6, 5);
        let model = ssm_build(&shapes, 5).expect("build");
        let err = ssm_reconstruction_error(&model, &model.mean_shape.clone()).expect("err");
        assert!(err < 1e-4);
    }

    // ── round-trip test ───────────────────────────────────────────────────────

    #[test]
    fn test_round_trip_project_reconstruct() {
        // Full-rank random data: with n_components = n_shapes - 1 we span the
        // entire training-set subspace and recover each training shape exactly.
        let shapes = make_random_shapes(5, 8, 13);
        let model = ssm_build(&shapes, 4).expect("build");
        for (i, s) in shapes.iter().enumerate() {
            let sp = ssm_project(&model, s).expect("project");
            let rec = ssm_reconstruct(&model, &sp.params).expect("reconstruct");
            let err: f32 = rec
                .iter()
                .zip(s.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f32>()
                / (rec.len() as f32);
            assert!(
                err.sqrt() < 0.05,
                "round-trip RMSE for shape {i} = {:.6}",
                err.sqrt()
            );
        }
    }

    // ── ShapeParameters ───────────────────────────────────────────────────────

    #[test]
    fn test_clamp_to_sigma_clamps_large_params() {
        let (shapes, _, _) = make_shapes(5, 4);
        let model = ssm_build(&shapes, 3).expect("build");
        let n_comp = model.components.len();
        // Very large parameters.
        let params: Vec<f32> = (0..n_comp).map(|i| 1000.0 * (i as f32 + 1.0)).collect();
        let sp = ShapeParameters {
            params,
            n_vertices: model.n_vertices,
        };
        let clamped = sp.clamp_to_sigma(&model, 2.0);
        for (k, (&p, &v)) in clamped
            .params
            .iter()
            .zip(model.variances.iter())
            .enumerate()
        {
            let bound = 2.0 * v.max(0.0).sqrt();
            assert!(
                p.abs() <= bound + 1e-5,
                "component {k}: |{p}| > bound {bound}"
            );
        }
    }

    #[test]
    fn test_clamp_to_sigma_zero_params_unchanged() {
        let (shapes, _, _) = make_shapes(5, 4);
        let model = ssm_build(&shapes, 3).expect("build");
        let n_comp = model.components.len();
        let sp = ShapeParameters {
            params: vec![0.0f32; n_comp],
            n_vertices: model.n_vertices,
        };
        let clamped = sp.clamp_to_sigma(&model, 2.0);
        for &p in &clamped.params {
            assert_eq!(p, 0.0);
        }
    }

    #[test]
    fn test_mahalanobis_distance_zero_params() {
        let (shapes, _, _) = make_shapes(5, 4);
        let model = ssm_build(&shapes, 3).expect("build");
        let n_comp = model.components.len();
        let sp = ShapeParameters {
            params: vec![0.0f32; n_comp],
            n_vertices: model.n_vertices,
        };
        assert!((sp.mahalanobis_distance(&model)).abs() < 1e-9);
    }

    #[test]
    fn test_mahalanobis_distance_nonzero() {
        let (shapes, _, _) = make_shapes(5, 4);
        let model = ssm_build(&shapes, 3).expect("build");
        let n_comp = model.components.len();
        let sp = ShapeParameters {
            params: vec![1.0f32; n_comp],
            n_vertices: model.n_vertices,
        };
        assert!(sp.mahalanobis_distance(&model) > 0.0);
    }

    #[test]
    fn test_z_scores_correct_normalisation() {
        let (shapes, _, _) = make_shapes(5, 4);
        let model = ssm_build(&shapes, 3).expect("build");
        let n_comp = model.components.len();
        // params[k] = sqrt(variance[k]) → z_score[k] should be 1.0
        let params: Vec<f32> = model.variances.iter().map(|&v| v.max(0.0).sqrt()).collect();
        let sp = ShapeParameters {
            params,
            n_vertices: model.n_vertices,
        };
        let zs = sp.z_scores(&model).expect("z_scores");
        assert_eq!(zs.len(), n_comp);
        for (k, &z) in zs.iter().enumerate() {
            assert!(
                (z - 1.0).abs() < 1e-5,
                "z_score[{k}] = {z:.6}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_z_scores_singular_when_zero_variance() {
        // Build a model with forced zero variance and check z_scores returns Singular.
        let (shapes, _, _) = make_shapes(5, 4);
        let mut model = ssm_build(&shapes, 3).expect("build");
        // Force a zero variance.
        model.variances[0] = 0.0;
        let n_comp = model.components.len();
        let sp = ShapeParameters {
            params: vec![1.0f32; n_comp],
            n_vertices: model.n_vertices,
        };
        assert!(matches!(sp.z_scores(&model), Err(SsmError::Singular)));
    }

    // ── ssm_random_shape ─────────────────────────────────────────────────────

    #[test]
    fn test_random_shape_output_length() {
        let (shapes, _, _) = make_shapes(5, 8);
        let model = ssm_build(&shapes, 3).expect("build");
        let mut rng = 42u64;
        let s = ssm_random_shape(&model, 2.0, &mut rng).expect("random");
        assert_eq!(s.len(), model.n_vertices * 3);
    }

    #[test]
    fn test_random_shape_within_sigma_bounds() {
        // Use full-rank random shapes so that projection retrieves the generated
        // coefficients faithfully.  With a generous tolerance (1e-3) to cover
        // floating-point round-trip error.
        let shapes = make_random_shapes(6, 8, 55);
        let model = ssm_build(&shapes, 5).expect("build");
        let n_sigma = 3.0f32;
        let mut rng = 1234u64;
        for _ in 0..10 {
            let s = ssm_random_shape(&model, n_sigma, &mut rng).expect("random");
            let sp = ssm_project(&model, &s).expect("project");
            for (k, (&p, &v)) in sp.params.iter().zip(model.variances.iter()).enumerate() {
                // Generous tolerance: projection is not exact for non-orthonormal
                // bases relative to the generation distribution.
                let bound = n_sigma * v.max(0.0).sqrt() * 1.1 + 1e-3;
                assert!(p.abs() <= bound, "random param {k} = {p} exceeds ±{bound}");
            }
        }
    }

    // Regression test: the old implementation drew coefficients uniformly
    // over [-n_sigma*std, +n_sigma*std], whose variance is (n_sigma*std)^2/3
    // -- about 1/3 of the model's learned variance for large n_sigma (where
    // clamping rarely triggers). Sampling the learned Gaussian instead
    // should produce a sample variance close to the model's variance.
    #[test]
    fn test_random_shape_sample_variance_matches_model_not_uniform_third() {
        let shapes = make_random_shapes(9, 12, 7);
        let model = ssm_build(&shapes, 4).expect("build failed");
        let n_sigma = 6.0f32; // wide enough that clamping is negligible
        let mut rng = 4242u64;

        let n_draws = 4000;
        let mut sum_sq = vec![0.0f64; model.components.len()];
        for _ in 0..n_draws {
            let s = ssm_random_shape(&model, n_sigma, &mut rng).expect("random");
            let sp = ssm_project(&model, &s).expect("project");
            for (acc, &p) in sum_sq.iter_mut().zip(sp.params.iter()) {
                *acc += f64::from(p) * f64::from(p);
            }
        }

        for (k, (&acc, &v)) in sum_sq.iter().zip(model.variances.iter()).enumerate() {
            let sample_var = (acc / f64::from(n_draws)) as f32;
            let ratio = sample_var / v.max(1e-12);
            assert!(
                ratio > 0.5,
                "component {k}: sample variance {sample_var} is only {ratio:.2}x the \
                 model variance {v} -- looks uniform (~1/3), not Gaussian (~1x)"
            );
        }
    }

    #[test]
    fn test_random_shape_reproducible_with_same_seed() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        let mut rng1 = 99u64;
        let mut rng2 = 99u64;
        let s1 = ssm_random_shape(&model, 2.0, &mut rng1).expect("r1");
        let s2 = ssm_random_shape(&model, 2.0, &mut rng2).expect("r2");
        assert_eq!(s1, s2);
    }

    // ── ssm_interpolate ───────────────────────────────────────────────────────

    #[test]
    fn test_interpolate_t0_gives_shape_a_approx() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 4).expect("build");
        let a = shapes[0].clone();
        let b = shapes[4].clone();
        let interp = ssm_interpolate(&model, &a, &b, 0.0).expect("interp");
        // Project a through the SSM and compare.
        let a_fit = ssm_fit(&model, &a).expect("fit_a");
        let err: f32 = interp
            .iter()
            .zip(a_fit.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(err < 1e-4, "t=0 should ≈ ssm_fit(shape_a), err={err}");
    }

    #[test]
    fn test_interpolate_t1_gives_shape_b_approx() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 4).expect("build");
        let a = shapes[0].clone();
        let b = shapes[4].clone();
        let interp = ssm_interpolate(&model, &a, &b, 1.0).expect("interp");
        let b_fit = ssm_fit(&model, &b).expect("fit_b");
        let err: f32 = interp
            .iter()
            .zip(b_fit.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(err < 1e-4, "t=1 should ≈ ssm_fit(shape_b), err={err}");
    }

    #[test]
    fn test_interpolate_t_half() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 4).expect("build");
        let a = shapes[0].clone();
        let b = shapes[4].clone();
        let interp = ssm_interpolate(&model, &a, &b, 0.5).expect("interp");
        assert_eq!(interp.len(), model.n_vertices * 3);
    }

    // ── ssm_component_sweep ───────────────────────────────────────────────────

    #[test]
    fn test_component_sweep_returns_n_steps_shapes() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        let sweep = ssm_component_sweep(&model, 0, 2.0, 7).expect("sweep");
        assert_eq!(sweep.len(), 7);
    }

    #[test]
    fn test_component_sweep_middle_shape_near_mean() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        let n_steps = 5usize;
        let sweep = ssm_component_sweep(&model, 0, 2.0, n_steps).expect("sweep");
        // The middle step (index 2) has t=0, so params[0] = 0 → should equal mean.
        let mid = &sweep[n_steps / 2];
        let err: f32 = mid
            .iter()
            .zip(model.mean_shape.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(err < 1e-4, "middle sweep shape should ≈ mean, err={err}");
    }

    #[test]
    fn test_component_sweep_empty_for_zero_steps() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        let sweep = ssm_component_sweep(&model, 0, 2.0, 0).expect("sweep");
        assert!(sweep.is_empty());
    }

    #[test]
    fn test_component_sweep_out_of_bounds() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        assert!(matches!(
            ssm_component_sweep(&model, 999, 2.0, 5),
            Err(SsmError::ComponentOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_component_sweep_shape_lengths() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        let sweep = ssm_component_sweep(&model, 0, 2.0, 4).expect("sweep");
        for s in &sweep {
            assert_eq!(s.len(), model.n_vertices * 3);
        }
    }

    // ── ssm_components_for_variance ───────────────────────────────────────────

    #[test]
    fn test_components_for_variance_zero_ratio() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        assert_eq!(ssm_components_for_variance(&model, 0.0), 0);
    }

    #[test]
    fn test_components_for_variance_full_ratio() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 4).expect("build");
        let n = ssm_components_for_variance(&model, 1.0);
        assert!(n <= model.components.len());
    }

    #[test]
    fn test_components_for_variance_monotone() {
        let (shapes, _, _) = make_shapes(6, 7);
        let model = ssm_build(&shapes, 5).expect("build");
        let n1 = ssm_components_for_variance(&model, 0.5);
        let n2 = ssm_components_for_variance(&model, 0.9);
        assert!(n1 <= n2, "more components needed for higher target");
    }

    #[test]
    fn test_components_for_variance_negative_ratio() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        assert_eq!(ssm_components_for_variance(&model, -0.1), 0);
    }

    // ── ssm_compute_stats ─────────────────────────────────────────────────────

    #[test]
    fn test_compute_stats_low_training_error() {
        // Use full-rank random shapes so that max components span the training set.
        let shapes = make_random_shapes(5, 6, 17);
        let model = ssm_build(&shapes, 4).expect("build");
        let stats = ssm_compute_stats(&model, &shapes).expect("stats");
        assert!(
            stats.mean_reconstruction_error < 0.1,
            "mean RMSE on training data should be low: {}",
            stats.mean_reconstruction_error
        );
    }

    #[test]
    fn test_compute_stats_fields() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        let stats = ssm_compute_stats(&model, &shapes).expect("stats");
        assert_eq!(stats.n_shapes, model.n_shapes);
        assert_eq!(stats.n_vertices, model.n_vertices);
        assert_eq!(stats.n_components, model.components.len());
        assert_eq!(stats.explained_variance_ratio.len(), model.components.len());
        assert_eq!(
            stats.cumulative_explained_ratio.len(),
            model.components.len()
        );
    }

    #[test]
    fn test_compute_stats_empty_training() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        assert!(matches!(
            ssm_compute_stats(&model, &[]),
            Err(SsmError::EmptyInput)
        ));
    }

    // ── ssm_vertex_std ───────────────────────────────────────────────────────

    #[test]
    fn test_vertex_std_identical_shapes_all_zero() {
        let shape = vec![1.0f32; 15];
        let shapes = vec![shape.clone(), shape.clone(), shape.clone()];
        let std = ssm_vertex_std(&shapes).expect("vertex_std");
        for &v in &std {
            assert!(v.abs() < 1e-6, "std should be 0 for identical shapes");
        }
    }

    #[test]
    fn test_vertex_std_output_length() {
        let (shapes, _, _) = make_shapes(5, 7);
        let std = ssm_vertex_std(&shapes).expect("vertex_std");
        assert_eq!(std.len(), shapes[0].len());
    }

    #[test]
    fn test_vertex_std_empty_input() {
        let shapes: Vec<Vec<f32>> = vec![];
        assert!(matches!(ssm_vertex_std(&shapes), Err(SsmError::EmptyInput)));
    }

    #[test]
    fn test_vertex_std_dimension_mismatch() {
        let shapes = vec![vec![1.0f32; 9], vec![1.0f32; 12]];
        assert!(matches!(
            ssm_vertex_std(&shapes),
            Err(SsmError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_vertex_std_nonzero_for_varying_shapes() {
        let (shapes, _, _) = make_shapes(5, 6);
        let std = ssm_vertex_std(&shapes).expect("vertex_std");
        let total: f32 = std.iter().sum();
        assert!(total > 0.0, "std should be non-zero for varying shapes");
    }

    // ── ssm_most_variable_vertices ────────────────────────────────────────────

    #[test]
    fn test_most_variable_vertices_k_zero() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        let res = ssm_most_variable_vertices(&model, 0);
        assert!(res.is_empty());
    }

    #[test]
    fn test_most_variable_vertices_k_exceeds_n() {
        let (shapes, _, _) = make_shapes(5, 4);
        let model = ssm_build(&shapes, 3).expect("build");
        let res = ssm_most_variable_vertices(&model, 1000);
        // Should return all vertices.
        assert_eq!(res.len(), model.n_vertices);
    }

    #[test]
    fn test_most_variable_vertices_sorted_descending() {
        let (shapes, _, _) = make_shapes(5, 8);
        let model = ssm_build(&shapes, 3).expect("build");
        let res = ssm_most_variable_vertices(&model, 5);
        for w in res.windows(2) {
            assert!(
                w[0].1 >= w[1].1 - 1e-6,
                "not sorted descending: {:.6} < {:.6}",
                w[0].1,
                w[1].1
            );
        }
    }

    #[test]
    fn test_most_variable_vertices_valid_indices() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        let res = ssm_most_variable_vertices(&model, 4);
        for (idx, _) in &res {
            assert!(*idx < model.n_vertices);
        }
    }

    // ── formatting ────────────────────────────────────────────────────────────

    #[test]
    fn test_format_stats_nonempty() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        let stats = ssm_compute_stats(&model, &shapes).expect("stats");
        let s = ssm_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("SSM Stats"));
    }

    #[test]
    fn test_format_model_nonempty() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        let s = ssm_format_model(&model);
        assert!(!s.is_empty());
        assert!(s.contains("StatisticalShapeModel"));
    }

    // ── xorshift PRNG ─────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_nonzero() {
        let mut state = 42u64;
        for _ in 0..100 {
            let v = xorshift64(&mut state);
            assert_ne!(v, 0);
        }
    }

    #[test]
    fn test_xorshift_f32_range() {
        let mut state = 1u64;
        for _ in 0..1000 {
            let v = xorshift_f32(&mut state);
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    // ── additional variance / edge-case tests ─────────────────────────────────

    #[test]
    fn test_build_two_shapes_one_component() {
        let (shapes, _, _) = make_shapes(2, 3);
        let model = ssm_build(&shapes, 1).expect("build");
        assert_eq!(model.n_shapes, 2);
        assert!(!model.components.is_empty());
    }

    #[test]
    fn test_component_sweep_single_step() {
        let (shapes, _, _) = make_shapes(5, 6);
        let model = ssm_build(&shapes, 3).expect("build");
        let sweep = ssm_component_sweep(&model, 0, 2.0, 1).expect("sweep");
        assert_eq!(sweep.len(), 1);
    }

    #[test]
    fn test_variance_ordering_strict_decrease() {
        // Use shapes with clear variance ordering.
        let n_shapes = 10;
        let n_vertices = 5;
        let dim = n_vertices * 3;
        let mut shapes: Vec<Vec<f32>> = Vec::new();
        let mut rng = 7u64;
        for _ in 0..n_shapes {
            let s: Vec<f32> = (0..dim).map(|_| xorshift_f32(&mut rng)).collect();
            shapes.push(s);
        }
        let model = ssm_build(&shapes, 5).expect("build");
        for w in model.variances.windows(2) {
            assert!(
                w[0] >= w[1] - 1e-5,
                "variance not non-increasing: {:.6} < {:.6}",
                w[0],
                w[1]
            );
        }
    }
}
