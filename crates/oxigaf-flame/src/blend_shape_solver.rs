//! Blend shape solver: inverse FLAME optimization.
//!
//! Given a target vertex set, find the blend shape (expression/shape) coefficients
//! that minimize vertex-wise reconstruction error via gradient descent.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the blend shape solver.
#[derive(Debug, Error)]
pub enum BlendSolverError {
    /// Target mesh contains no vertices.
    #[error("Empty target mesh: no vertices")]
    EmptyTarget,

    /// Template and target vertex counts differ.
    #[error("Vertex count mismatch: template has {template}, target has {target}")]
    VertexCountMismatch { template: usize, target: usize },

    /// Basis displacement vectors are not all the same length.
    #[error("Basis has {n_basis} vectors but each has {n_verts} vertices (expected {expected})")]
    BasisDimensionMismatch {
        n_basis: usize,
        n_verts: usize,
        expected: usize,
    },

    /// Weight bounds are logically invalid (min >= max).
    #[error("Invalid constraint: weight bound [{min}, {max}] is invalid")]
    InvalidWeightBound { min: f32, max: f32 },

    /// Solver failed to converge and the residual is unacceptably large.
    #[error("Solver diverged after {iters} iterations (residual: {residual:.4e})")]
    SolverDiverged { iters: usize, residual: f32 },

    /// No basis vectors were provided.
    #[error("No basis vectors provided")]
    EmptyBasis,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the blend shape solver.
#[derive(Debug, Clone)]
pub struct BlendSolverConfig {
    /// Maximum optimization iterations (default: 200).
    pub max_iter: usize,
    /// Convergence tolerance on residual change (default: 1e-5).
    pub tolerance: f32,
    /// Gradient descent step size (default: 0.01).
    pub step_size: f32,
    /// Minimum blend weight (default: -3.0).
    pub weight_min: f32,
    /// Maximum blend weight (default: 3.0).
    pub weight_max: f32,
    /// L2 regularization on weights (default: 1e-3).
    pub regularization: f32,
    /// Project weights to >= 0 after each step (default: false).
    pub enforce_nonneg: bool,
}

impl Default for BlendSolverConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tolerance: 1e-5,
            step_size: 0.01,
            weight_min: -3.0,
            weight_max: 3.0,
            regularization: 1e-3,
            enforce_nonneg: false,
        }
    }
}

// ---------------------------------------------------------------------------
// BlendBasis
// ---------------------------------------------------------------------------

/// A linear blend shape basis.
///
/// Each basis vector is a displacement from the template mesh stored as a flat
/// `Vec<f32>` of length `n_vertices * 3` (x0, y0, z0, x1, y1, z1, …).
#[derive(Debug, Clone)]
pub struct BlendBasis {
    /// Number of basis vectors.
    pub n_basis: usize,
    /// Number of vertices per basis displacement.
    pub n_vertices: usize,
    /// `n_basis` displacement vectors, each of length `n_vertices * 3`.
    pub displacements: Vec<Vec<f32>>,
}

impl BlendBasis {
    /// Construct a `BlendBasis` from a collection of displacement vectors.
    ///
    /// All displacement vectors must have the same length and that length must
    /// be a multiple of 3.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn new(displacements: Vec<Vec<f32>>) -> Result<Self, BlendSolverError> {
        if displacements.is_empty() {
            return Err(BlendSolverError::EmptyBasis);
        }
        let expected_len = displacements[0].len();
        let n_vertices = expected_len / 3;
        let n_basis = displacements.len();

        for (idx, disp) in displacements.iter().enumerate() {
            if disp.len() != expected_len {
                return Err(BlendSolverError::BasisDimensionMismatch {
                    n_basis,
                    n_verts: disp.len() / 3,
                    expected: n_vertices,
                });
            }
            let _ = idx; // used implicitly
        }

        Ok(Self {
            n_basis,
            n_vertices,
            displacements,
        })
    }

    /// Compute the weighted sum of displacement vectors.
    ///
    /// Returns a flat `Vec<f32>` of length `n_vertices * 3` representing the
    /// total displacement (not the final vertex positions — add to template).
    #[must_use]
    pub fn apply(&self, weights: &[f32]) -> Vec<f32> {
        let n = self.n_vertices * 3;
        let mut result = vec![0.0f32; n];
        for (b, disp) in self.displacements.iter().enumerate() {
            let w = if b < weights.len() { weights[b] } else { 0.0 };
            for i in 0..n {
                result[i] += w * disp[i];
            }
        }
        result
    }

    /// Validate that all displacement vectors have the same length.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn validate(&self) -> Result<(), BlendSolverError> {
        if self.displacements.is_empty() {
            return Err(BlendSolverError::EmptyBasis);
        }
        let expected = self.displacements[0].len();
        for disp in &self.displacements {
            if disp.len() != expected {
                return Err(BlendSolverError::BasisDimensionMismatch {
                    n_basis: self.n_basis,
                    n_verts: disp.len() / 3,
                    expected: expected / 3,
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// WeightConstraints
// ---------------------------------------------------------------------------

/// Per-weight bounds and optional symmetry pairs.
#[derive(Debug, Clone)]
pub struct WeightConstraints {
    /// Per-basis minimum weight (length = `n_basis`).
    pub min_weights: Vec<f32>,
    /// Per-basis maximum weight (length = `n_basis`).
    pub max_weights: Vec<f32>,
    /// Index pairs `(i, j)` that must share the same weight (their average).
    pub symmetry_pairs: Vec<(usize, usize)>,
}

impl WeightConstraints {
    /// Create uniform constraints (same bounds for every basis vector).
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn uniform(n_basis: usize, min: f32, max: f32) -> Result<Self, BlendSolverError> {
        if min >= max {
            return Err(BlendSolverError::InvalidWeightBound { min, max });
        }
        Ok(Self {
            min_weights: vec![min; n_basis],
            max_weights: vec![max; n_basis],
            symmetry_pairs: Vec::new(),
        })
    }

    /// Create unconstrained bounds (-3.0 … 3.0) with no symmetry pairs.
    #[must_use]
    pub fn unconstrained(n_basis: usize) -> Self {
        Self {
            min_weights: vec![-3.0; n_basis],
            max_weights: vec![3.0; n_basis],
            symmetry_pairs: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// BlendSolverResult
// ---------------------------------------------------------------------------

/// Result of a blend shape solve.
#[derive(Debug, Clone)]
pub struct BlendSolverResult {
    /// Optimized blend weights.
    pub weights: Vec<f32>,
    /// Final vertex-wise RMSE.
    pub residual: f32,
    /// Number of iterations actually performed.
    pub n_iterations: usize,
    /// Whether the solver met the convergence criterion.
    pub converged: bool,
    /// Per-vertex Euclidean error magnitude.
    pub vertex_errors: Vec<f32>,
}

// ---------------------------------------------------------------------------
// SolverStats
// ---------------------------------------------------------------------------

/// Aggregated statistics about a solver result.
#[derive(Debug, Clone)]
pub struct SolverStats {
    /// Mean per-vertex error.
    pub mean_vertex_error: f32,
    /// Maximum per-vertex error.
    pub max_vertex_error: f32,
    /// L2 norm of the weight vector.
    pub weight_l2_norm: f32,
    /// Fraction of weights with |w| < 0.01.
    pub weight_sparsity: f32,
    /// Number of weights with |w| > 0.01.
    pub n_active_weights: usize,
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Apply blend displacements to a template vertex array.
///
/// `template_verts`: flat N×3 vertex positions.
/// `displacements`: `n_basis` flat N×3 displacement vectors.
/// `weights`: `n_basis` blend weights.
///
/// Returns `template + sum(weight[b] * displacements[b])`.
#[must_use]
pub fn apply_blend_displacements(
    template_verts: &[f32],
    displacements: &[Vec<f32>],
    weights: &[f32],
) -> Vec<f32> {
    let mut result = template_verts.to_vec();
    for (b, disp) in displacements.iter().enumerate() {
        let w = if b < weights.len() { weights[b] } else { 0.0 };
        let len = result.len().min(disp.len());
        for i in 0..len {
            result[i] += w * disp[i];
        }
    }
    result
}

/// Compute per-vertex Euclidean distances between two flat N×3 vertex arrays.
///
/// Both slices must have the same length (a multiple of 3).
#[must_use]
pub fn compute_vertex_errors(pred_verts: &[f32], target_verts: &[f32]) -> Vec<f32> {
    let n_verts = pred_verts.len() / 3;
    let mut errors = Vec::with_capacity(n_verts);
    for v in 0..n_verts {
        let dx = pred_verts[v * 3] - target_verts[v * 3];
        let dy = pred_verts[v * 3 + 1] - target_verts[v * 3 + 1];
        let dz = pred_verts[v * 3 + 2] - target_verts[v * 3 + 2];
        errors.push((dx * dx + dy * dy + dz * dz).sqrt());
    }
    errors
}

/// Root-mean-squared vertex error (RMSE).
#[must_use]
pub fn compute_residual(pred_verts: &[f32], target_verts: &[f32]) -> f32 {
    let n_verts = pred_verts.len() / 3;
    if n_verts == 0 {
        return 0.0;
    }
    let mut sum_sq = 0.0f32;
    for v in 0..n_verts {
        let dx = pred_verts[v * 3] - target_verts[v * 3];
        let dy = pred_verts[v * 3 + 1] - target_verts[v * 3 + 1];
        let dz = pred_verts[v * 3 + 2] - target_verts[v * 3 + 2];
        sum_sq += dx * dx + dy * dy + dz * dz;
    }
    (sum_sq / n_verts as f32).sqrt()
}

/// Gradient of the MSE loss w.r.t. each blend weight.
///
/// `dL/dw_b = (2/N) * sum_v( dot(pred_v - target_v, displacements[b][v]) )`
///
/// Returns a `Vec<f32>` of length `basis.n_basis`.
#[must_use]
pub fn gradient_wrt_weights(
    basis: &BlendBasis,
    pred_verts: &[f32],
    target_verts: &[f32],
) -> Vec<f32> {
    let n_verts = basis.n_vertices;
    let scale = 2.0 / n_verts.max(1) as f32;
    let mut grad = vec![0.0f32; basis.n_basis];

    for (b, disp) in basis.displacements.iter().enumerate() {
        let mut acc = 0.0f32;
        for v in 0..n_verts {
            let base = v * 3;
            let ex = pred_verts[base] - target_verts[base];
            let ey = pred_verts[base + 1] - target_verts[base + 1];
            let ez = pred_verts[base + 2] - target_verts[base + 2];
            acc += ex * disp[base] + ey * disp[base + 1] + ez * disp[base + 2];
        }
        grad[b] = scale * acc;
    }
    grad
}

/// Project weights into the feasible region defined by `constraints`.
///
/// 1. Clamp each weight to `[min_weights[i], max_weights[i]]`.
/// 2. For symmetry pairs `(i, j)`, set both to their mean.
#[must_use]
pub fn project_weights(weights: &[f32], constraints: &WeightConstraints) -> Vec<f32> {
    let mut w = weights.to_vec();
    // Clamp
    for (i, wi) in w.iter_mut().enumerate() {
        if i < constraints.min_weights.len() {
            *wi = wi.max(constraints.min_weights[i]);
        }
        if i < constraints.max_weights.len() {
            *wi = wi.min(constraints.max_weights[i]);
        }
    }
    // Symmetry averaging
    for &(a, b) in &constraints.symmetry_pairs {
        if a < w.len() && b < w.len() {
            let mean = (w[a] + w[b]) * 0.5;
            w[a] = mean;
            w[b] = mean;
        }
    }
    w
}

/// Run gradient descent to find blend weights that minimize vertex-wise RMSE.
///
/// # Arguments
/// - `template_verts`: flat N×3 positions of the undeformed template mesh.
/// - `target_verts`: flat N×3 positions of the target mesh.
/// - `basis`: the blend shape basis.
/// - `constraints`: per-weight bounds and symmetry pairs.
/// - `config`: solver hyper-parameters.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn solve_blend_shapes(
    template_verts: &[f32],
    target_verts: &[f32],
    basis: &BlendBasis,
    constraints: &WeightConstraints,
    config: &BlendSolverConfig,
) -> Result<BlendSolverResult, BlendSolverError> {
    // --- Validate inputs -------------------------------------------------------
    if target_verts.is_empty() {
        return Err(BlendSolverError::EmptyTarget);
    }
    let n_template = template_verts.len() / 3;
    let n_target = target_verts.len() / 3;
    if n_template != n_target {
        return Err(BlendSolverError::VertexCountMismatch {
            template: n_template,
            target: n_target,
        });
    }
    basis.validate()?;

    // --- Initialise -----------------------------------------------------------
    let mut weights = vec![0.0f32; basis.n_basis];
    let mut residual = f32::MAX;
    let mut n_iter = 0usize;
    let mut converged = false;

    // Build an effective constraints struct that enforces non-neg if requested
    let effective_constraints = if config.enforce_nonneg {
        let min_w = constraints
            .min_weights
            .iter()
            .map(|&m| m.max(0.0))
            .collect::<Vec<_>>();
        WeightConstraints {
            min_weights: min_w,
            max_weights: constraints.max_weights.clone(),
            symmetry_pairs: constraints.symmetry_pairs.clone(),
        }
    } else {
        constraints.clone()
    };

    // --- Gradient descent loop ------------------------------------------------
    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        // Forward: apply displacements
        let pred = apply_blend_displacements(template_verts, &basis.displacements, &weights);

        // Gradient of MSE loss w.r.t. weights
        let mut grad = gradient_wrt_weights(basis, &pred, target_verts);

        // L2 regularisation
        for (b, g) in grad.iter_mut().enumerate() {
            *g += 2.0 * config.regularization * weights[b];
        }

        // Weight update
        for (b, w) in weights.iter_mut().enumerate() {
            *w -= config.step_size * grad[b];
        }

        // Projection
        weights = project_weights(&weights, &effective_constraints);

        // Compute new residual
        let pred_new = apply_blend_displacements(template_verts, &basis.displacements, &weights);
        let new_residual = compute_residual(&pred_new, target_verts);

        // Check convergence
        if (residual - new_residual).abs() < config.tolerance {
            residual = new_residual;
            converged = true;
            break;
        }
        residual = new_residual;
    }

    // --- Divergence check -----------------------------------------------------
    if !converged && residual > 1.0 {
        return Err(BlendSolverError::SolverDiverged {
            iters: n_iter,
            residual,
        });
    }

    // Compute final per-vertex errors
    let pred_final = apply_blend_displacements(template_verts, &basis.displacements, &weights);
    let vertex_errors = compute_vertex_errors(&pred_final, target_verts);

    Ok(BlendSolverResult {
        weights,
        residual,
        n_iterations: n_iter,
        converged,
        vertex_errors,
    })
}

/// Compute aggregated statistics from a solver result.
#[must_use]
pub fn compute_solver_stats(result: &BlendSolverResult) -> SolverStats {
    let n_weights = result.weights.len();

    // Weight statistics
    let weight_l2_norm = result.weights.iter().map(|&w| w * w).sum::<f32>().sqrt();

    let n_near_zero = result.weights.iter().filter(|&&w| w.abs() < 0.01).count();
    let weight_sparsity = if n_weights > 0 {
        n_near_zero as f32 / n_weights as f32
    } else {
        0.0
    };
    let n_active_weights = n_weights - n_near_zero;

    // Vertex error statistics
    let n_verts = result.vertex_errors.len();
    let mean_vertex_error = if n_verts > 0 {
        result.vertex_errors.iter().sum::<f32>() / n_verts as f32
    } else {
        0.0
    };
    let max_vertex_error = result.vertex_errors.iter().copied().fold(0.0f32, f32::max);

    SolverStats {
        mean_vertex_error,
        max_vertex_error,
        weight_l2_norm,
        weight_sparsity,
        n_active_weights,
    }
}

/// Solve with non-negativity constraint (all weights >= 0.0).
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn nonneg_solve_blend_shapes(
    template_verts: &[f32],
    target_verts: &[f32],
    basis: &BlendBasis,
    config: &BlendSolverConfig,
) -> Result<BlendSolverResult, BlendSolverError> {
    let mut cfg = config.clone();
    cfg.enforce_nonneg = true;
    let constraints = WeightConstraints {
        min_weights: vec![0.0; basis.n_basis],
        max_weights: vec![cfg.weight_max; basis.n_basis],
        symmetry_pairs: Vec::new(),
    };
    solve_blend_shapes(template_verts, target_verts, basis, &constraints, &cfg)
}

/// Fit FLAME expression coefficients to a target vertex set.
///
/// `expression_basis`: each element is a flat N×3 displacement shape.
/// Returns the optimized weight vector (expression coefficients).
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn fit_expression_coefficients(
    template_verts: &[f32],
    target_verts: &[f32],
    expression_basis: Vec<Vec<f32>>,
    n_coeffs: usize,
    config: &BlendSolverConfig,
) -> Result<Vec<f32>, BlendSolverError> {
    // Trim basis to requested coefficient count
    let basis_vecs: Vec<Vec<f32>> = expression_basis.into_iter().take(n_coeffs).collect();
    let basis = BlendBasis::new(basis_vecs)?;
    let constraints = WeightConstraints::unconstrained(basis.n_basis);
    let result = solve_blend_shapes(template_verts, target_verts, &basis, &constraints, config)?;
    Ok(result.weights)
}

/// Compute the residual for each weight vector in a sequence.
///
/// Useful for visualising convergence curves.
#[must_use]
pub fn blend_solver_residual_curve(
    template_verts: &[f32],
    target_verts: &[f32],
    basis: &BlendBasis,
    weights_sequence: &[Vec<f32>],
) -> Vec<f32> {
    weights_sequence
        .iter()
        .map(|w| {
            let pred = apply_blend_displacements(template_verts, &basis.displacements, w);
            compute_residual(&pred, target_verts)
        })
        .collect()
}

/// Format a solver result and its stats as a human-readable string.
#[must_use]
pub fn format_solver_result(result: &BlendSolverResult, stats: &SolverStats) -> String {
    format!(
        "Blend solve: converged={}, residual={:.6}, active_weights={}/{}",
        result.converged,
        result.residual,
        stats.n_active_weights,
        result.weights.len(),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a flat vertex array for `n` vertices all at the given xyz.
    #[allow(clippy::many_single_char_names)]
    fn flat_verts(n: usize, x: f32, y: f32, z: f32) -> Vec<f32> {
        let mut v = Vec::with_capacity(n * 3);
        for _ in 0..n {
            v.push(x);
            v.push(y);
            v.push(z);
        }
        v
    }

    // Helper: create a single-displacement basis that shifts every vertex by (dx, dy, dz).
    fn uniform_basis(n_verts: usize, dx: f32, dy: f32, dz: f32) -> Vec<Vec<f32>> {
        let mut disp = Vec::with_capacity(n_verts * 3);
        for _ in 0..n_verts {
            disp.push(dx);
            disp.push(dy);
            disp.push(dz);
        }
        vec![disp]
    }

    // -----------------------------------------------------------------------
    // BlendSolverError variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_empty_target_display() {
        let e = BlendSolverError::EmptyTarget;
        assert!(!format!("{e}").is_empty());
    }

    #[test]
    fn test_error_vertex_count_mismatch_display() {
        let e = BlendSolverError::VertexCountMismatch {
            template: 100,
            target: 50,
        };
        let s = format!("{e}");
        assert!(s.contains("100") && s.contains("50"));
    }

    #[test]
    fn test_error_basis_dimension_mismatch_display() {
        let e = BlendSolverError::BasisDimensionMismatch {
            n_basis: 5,
            n_verts: 200,
            expected: 300,
        };
        let s = format!("{e}");
        assert!(s.contains('5') && s.contains("200") && s.contains("300"));
    }

    #[test]
    fn test_error_invalid_weight_bound_display() {
        let e = BlendSolverError::InvalidWeightBound { min: 1.0, max: 0.0 };
        assert!(!format!("{e}").is_empty());
    }

    #[test]
    fn test_error_solver_diverged_display() {
        let e = BlendSolverError::SolverDiverged {
            iters: 50,
            residual: 2.5,
        };
        let s = format!("{e}");
        assert!(s.contains("50"));
    }

    #[test]
    fn test_error_empty_basis_display() {
        let e = BlendSolverError::EmptyBasis;
        assert!(!format!("{e}").is_empty());
    }

    // -----------------------------------------------------------------------
    // BlendBasis::new
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_basis_new_valid() {
        let disps = vec![vec![0.0f32; 30], vec![1.0f32; 30]];
        let basis = BlendBasis::new(disps).unwrap();
        assert_eq!(basis.n_basis, 2);
        assert_eq!(basis.n_vertices, 10);
    }

    #[test]
    fn test_blend_basis_new_empty_basis_error() {
        let result = BlendBasis::new(vec![]);
        assert!(matches!(result, Err(BlendSolverError::EmptyBasis)));
    }

    #[test]
    fn test_blend_basis_new_mismatched_size_error() {
        // First displacement: 30 floats (10 verts), second: 15 floats (5 verts).
        let disps = vec![vec![0.0f32; 30], vec![1.0f32; 15]];
        let result = BlendBasis::new(disps);
        assert!(matches!(
            result,
            Err(BlendSolverError::BasisDimensionMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // BlendBasis::apply
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_basis_apply_zero_weights() {
        let disps = vec![vec![1.0f32; 9]]; // 3 vertices
        let basis = BlendBasis::new(disps).unwrap();
        let result = basis.apply(&[0.0]);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_blend_basis_apply_identity() {
        // One basis vector of all-1.0; weight=2.0 should give all-2.0.
        let disps = vec![vec![1.0f32; 9]];
        let basis = BlendBasis::new(disps).unwrap();
        let result = basis.apply(&[2.0]);
        assert!(result.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_blend_basis_apply_two_basis_vectors() {
        // Two basis vectors: all-1.0 and all-0.5; weights = [1.0, 2.0] → each float = 1*1 + 2*0.5 = 2.0
        let disps = vec![vec![1.0f32; 9], vec![0.5f32; 9]];
        let basis = BlendBasis::new(disps).unwrap();
        let result = basis.apply(&[1.0, 2.0]);
        assert!(result.iter().all(|&v| (v - 2.0).abs() < 1e-5));
    }

    #[test]
    fn test_blend_basis_apply_empty_weights() {
        let disps = vec![vec![1.0f32; 9]];
        let basis = BlendBasis::new(disps).unwrap();
        // Weights shorter than n_basis — missing weights treated as 0.
        let result = basis.apply(&[]);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // -----------------------------------------------------------------------
    // BlendBasis::validate
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_basis_validate_ok() {
        let disps = vec![vec![0.0f32; 12], vec![0.0f32; 12]];
        let basis = BlendBasis::new(disps).unwrap();
        assert!(basis.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // WeightConstraints::uniform
    // -----------------------------------------------------------------------

    #[test]
    fn test_weight_constraints_uniform_valid() {
        let c = WeightConstraints::uniform(5, -1.0, 1.0).unwrap();
        assert_eq!(c.min_weights.len(), 5);
        assert!(c.min_weights.iter().all(|&v| (v + 1.0).abs() < 1e-6));
        assert!(c.max_weights.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_weight_constraints_uniform_invalid_bounds() {
        let result = WeightConstraints::uniform(5, 2.0, 1.0);
        assert!(matches!(
            result,
            Err(BlendSolverError::InvalidWeightBound { .. })
        ));
    }

    #[test]
    fn test_weight_constraints_uniform_equal_bounds_invalid() {
        let result = WeightConstraints::uniform(3, 0.5, 0.5);
        assert!(matches!(
            result,
            Err(BlendSolverError::InvalidWeightBound { .. })
        ));
    }

    #[test]
    fn test_weight_constraints_unconstrained() {
        let c = WeightConstraints::unconstrained(4);
        assert_eq!(c.min_weights.len(), 4);
        assert!(c.min_weights.iter().all(|&v| (v + 3.0).abs() < 1e-6));
        assert!(c.max_weights.iter().all(|&v| (v - 3.0).abs() < 1e-6));
        assert!(c.symmetry_pairs.is_empty());
    }

    // -----------------------------------------------------------------------
    // apply_blend_displacements
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_blend_displacements_known_value() {
        // template = [1, 2, 3], basis = [[0.5, 0.5, 0.5]], weight = [2.0]
        // result = [1+1, 2+1, 3+1] = [2, 3, 4]
        let template = vec![1.0f32, 2.0, 3.0];
        let disps = vec![vec![0.5f32, 0.5, 0.5]];
        let result = apply_blend_displacements(&template, &disps, &[2.0]);
        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 3.0).abs() < 1e-6);
        assert!((result[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_blend_displacements_zero_basis() {
        let template = vec![1.0f32, 2.0, 3.0];
        let disps: Vec<Vec<f32>> = vec![];
        let result = apply_blend_displacements(&template, &disps, &[]);
        assert_eq!(result, template);
    }

    #[test]
    fn test_apply_blend_displacements_zero_weight() {
        let template = vec![1.0f32, 0.0, 0.0];
        let disps = vec![vec![10.0f32, 10.0, 10.0]];
        let result = apply_blend_displacements(&template, &disps, &[0.0]);
        assert_eq!(result, template);
    }

    // -----------------------------------------------------------------------
    // compute_vertex_errors
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_vertex_errors_coincident() {
        let v = flat_verts(5, 1.0, 2.0, 3.0);
        let errors = compute_vertex_errors(&v, &v);
        assert!(errors.iter().all(|&e| e.abs() < 1e-6));
    }

    #[test]
    fn test_compute_vertex_errors_known_distance() {
        // Single vertex: pred=(1,0,0), target=(0,0,0) → distance=1
        let pred = vec![1.0f32, 0.0, 0.0];
        let target = vec![0.0f32, 0.0, 0.0];
        let errors = compute_vertex_errors(&pred, &target);
        assert_eq!(errors.len(), 1);
        assert!((errors[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_vertex_errors_multiple_verts() {
        // Two vertices: (3,4,0)→(0,0,0)=5, (0,0,0)→(0,0,0)=0
        let pred = vec![3.0f32, 4.0, 0.0, 0.0, 0.0, 0.0];
        let target = vec![0.0f32; 6];
        let errors = compute_vertex_errors(&pred, &target);
        assert!((errors[0] - 5.0).abs() < 1e-5);
        assert!(errors[1].abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // compute_residual
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_residual_zero_error() {
        let v = flat_verts(4, 0.5, -1.0, 2.0);
        let r = compute_residual(&v, &v);
        assert!(r.abs() < 1e-6);
    }

    #[test]
    fn test_compute_residual_known_value() {
        // Two vertices: error vectors (1,0,0) and (0,1,0); squared errors 1 and 1; RMSE = 1.
        let pred = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let target = vec![0.0f32; 6];
        let r = compute_residual(&pred, &target);
        assert!((r - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_residual_empty() {
        let r = compute_residual(&[], &[]);
        assert!(r.abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // gradient_wrt_weights
    // -----------------------------------------------------------------------

    #[test]
    fn test_gradient_wrt_weights_direction() {
        // template=zeros, single basis = all-1. pred is above target in all dims.
        // If weight=0, pred=template=zeros, target = all-1 → error = -1 per component.
        // gradient = (2/N) * sum dot(pred-target, disp) = (2/N) * sum(-1 * 1) < 0.
        // A negative gradient means we should increase the weight → correct.
        let n = 3usize;
        let template = flat_verts(n, 0.0, 0.0, 0.0);
        let target = flat_verts(n, 1.0, 1.0, 1.0);
        let disps = uniform_basis(n, 1.0, 1.0, 1.0);
        let basis = BlendBasis::new(disps).unwrap();
        let pred = apply_blend_displacements(&template, &basis.displacements, &[0.0]);
        let grad = gradient_wrt_weights(&basis, &pred, &target);
        assert_eq!(grad.len(), 1);
        // Gradient is negative → step in -grad direction → increase weight → reduces error.
        assert!(grad[0] < 0.0);
    }

    #[test]
    fn test_gradient_wrt_weights_zero_error() {
        // If pred == target, gradient should be zero.
        let n = 5usize;
        let verts = flat_verts(n, 1.0, 2.0, 3.0);
        let disps = uniform_basis(n, 1.0, 0.0, 0.0);
        let basis = BlendBasis::new(disps).unwrap();
        let grad = gradient_wrt_weights(&basis, &verts, &verts);
        assert!(grad.iter().all(|&g| g.abs() < 1e-6));
    }

    // -----------------------------------------------------------------------
    // project_weights
    // -----------------------------------------------------------------------

    #[test]
    fn test_project_weights_clamping() {
        let constraints = WeightConstraints::uniform(3, -1.0, 1.0).unwrap();
        let weights = vec![-5.0, 0.5, 3.0];
        let projected = project_weights(&weights, &constraints);
        assert!((projected[0] + 1.0).abs() < 1e-6); // clamped to -1
        assert!((projected[1] - 0.5).abs() < 1e-6); // unchanged
        assert!((projected[2] - 1.0).abs() < 1e-6); // clamped to 1
    }

    #[test]
    fn test_project_weights_symmetry() {
        let mut constraints = WeightConstraints::unconstrained(4);
        constraints.symmetry_pairs = vec![(0, 2)];
        let weights = vec![1.0, 0.5, 3.0, 0.0];
        let projected = project_weights(&weights, &constraints);
        // indices 0 and 2 should both equal their mean: (1+3)/2 = 2.0
        assert!((projected[0] - 2.0).abs() < 1e-6);
        assert!((projected[2] - 2.0).abs() < 1e-6);
        // indices 1 and 3 unchanged
        assert!((projected[1] - 0.5).abs() < 1e-6);
        assert!((projected[3] - 0.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // solve_blend_shapes
    // -----------------------------------------------------------------------

    #[test]
    fn test_solve_blend_shapes_target_equals_template() {
        // When target == template, optimal weights should be near zero.
        let n = 10usize;
        let template = flat_verts(n, 1.0, 0.5, -0.3);
        let basis = BlendBasis::new(uniform_basis(n, 1.0, 0.0, 0.0)).unwrap();
        let constraints = WeightConstraints::unconstrained(1);
        let config = BlendSolverConfig::default();
        let result =
            solve_blend_shapes(&template, &template, &basis, &constraints, &config).unwrap();
        assert!(result.residual < 1e-3);
        assert!(result.weights.iter().all(|&w| w.abs() < 0.1));
    }

    #[test]
    fn test_solve_blend_shapes_single_basis_recovery() {
        // target = template + 1.0 * disp → solver should recover weight ≈ 1.0
        let n = 5usize;
        let template = flat_verts(n, 0.0, 0.0, 0.0);
        let disp = vec![1.0f32; n * 3];
        let target = disp.clone(); // template + 1*disp
        let basis = BlendBasis::new(vec![disp]).unwrap();
        let constraints = WeightConstraints::unconstrained(1);
        let config = BlendSolverConfig {
            max_iter: 500,
            step_size: 0.05,
            tolerance: 1e-7,
            ..Default::default()
        };
        let result = solve_blend_shapes(&template, &target, &basis, &constraints, &config).unwrap();
        assert!(
            (result.weights[0] - 1.0).abs() < 0.15,
            "weight={}",
            result.weights[0]
        );
    }

    #[test]
    fn test_solve_blend_shapes_empty_target_error() {
        let template = flat_verts(3, 0.0, 0.0, 0.0);
        let basis = BlendBasis::new(uniform_basis(3, 1.0, 0.0, 0.0)).unwrap();
        let constraints = WeightConstraints::unconstrained(1);
        let config = BlendSolverConfig::default();
        let result = solve_blend_shapes(&template, &[], &basis, &constraints, &config);
        assert!(matches!(result, Err(BlendSolverError::EmptyTarget)));
    }

    #[test]
    fn test_solve_blend_shapes_vertex_count_mismatch_error() {
        let template = flat_verts(5, 0.0, 0.0, 0.0);
        let target = flat_verts(3, 0.0, 0.0, 0.0);
        let basis = BlendBasis::new(uniform_basis(5, 1.0, 0.0, 0.0)).unwrap();
        let constraints = WeightConstraints::unconstrained(1);
        let config = BlendSolverConfig::default();
        let result = solve_blend_shapes(&template, &target, &basis, &constraints, &config);
        assert!(matches!(
            result,
            Err(BlendSolverError::VertexCountMismatch { .. })
        ));
    }

    #[test]
    fn test_solve_blend_shapes_residual_zero_after_identity() {
        // Single basis that exactly spans the difference → residual should drop significantly.
        let n = 4usize;
        let template = flat_verts(n, 0.0, 0.0, 0.0);
        let disp = vec![0.5f32; n * 3];
        // Target = 0.5 * basis → weight 1.0 is optimal.
        let target = disp.clone();
        let basis = BlendBasis::new(vec![disp]).unwrap();
        let constraints = WeightConstraints::unconstrained(1);
        let config = BlendSolverConfig {
            max_iter: 1000,
            step_size: 0.1,
            tolerance: 1e-8,
            ..Default::default()
        };
        let result = solve_blend_shapes(&template, &target, &basis, &constraints, &config).unwrap();
        assert!(result.residual < 0.1, "residual={}", result.residual);
    }

    // -----------------------------------------------------------------------
    // compute_solver_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_solver_stats_all_zero_weights() {
        let result = BlendSolverResult {
            weights: vec![0.0f32; 5],
            residual: 0.0,
            n_iterations: 1,
            converged: true,
            vertex_errors: vec![0.0f32; 3],
        };
        let stats = compute_solver_stats(&result);
        assert_eq!(stats.n_active_weights, 0);
        assert!((stats.weight_sparsity - 1.0).abs() < 1e-6);
        assert!((stats.weight_l2_norm).abs() < 1e-6);
    }

    #[test]
    fn test_compute_solver_stats_active_weights() {
        let result = BlendSolverResult {
            weights: vec![0.5, 0.0, 1.0, -0.005, 2.0],
            residual: 0.1,
            n_iterations: 10,
            converged: true,
            vertex_errors: vec![0.05, 0.1, 0.15],
        };
        let stats = compute_solver_stats(&result);
        // Active: |w| > 0.01 → indices 0, 2, 4 → 3 active
        assert_eq!(stats.n_active_weights, 3);
        // Sparsity: 2 near-zero / 5 total = 0.4
        assert!((stats.weight_sparsity - 0.4).abs() < 1e-5);
        assert!((stats.mean_vertex_error - 0.1).abs() < 1e-5);
        assert!((stats.max_vertex_error - 0.15).abs() < 1e-5);
    }

    #[test]
    fn test_compute_solver_stats_l2_norm() {
        let result = BlendSolverResult {
            weights: vec![3.0, 4.0],
            residual: 0.0,
            n_iterations: 1,
            converged: true,
            vertex_errors: vec![],
        };
        let stats = compute_solver_stats(&result);
        assert!((stats.weight_l2_norm - 5.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // nonneg_solve_blend_shapes
    // -----------------------------------------------------------------------

    #[test]
    fn test_nonneg_solve_blend_shapes_all_nonneg() {
        let n = 5usize;
        let template = flat_verts(n, 0.0, 0.0, 0.0);
        let disp = vec![1.0f32; n * 3];
        let target = flat_verts(n, 0.5, 0.5, 0.5);
        let basis = BlendBasis::new(vec![disp]).unwrap();
        let config = BlendSolverConfig::default();
        let result = nonneg_solve_blend_shapes(&template, &target, &basis, &config).unwrap();
        assert!(result.weights.iter().all(|&w| w >= -1e-6));
    }

    #[test]
    fn test_nonneg_solve_blend_shapes_does_not_go_negative() {
        // When target is "below" template, unconstrained would go negative;
        // nonneg should clamp to 0.
        let n = 3usize;
        let template = flat_verts(n, 2.0, 2.0, 2.0);
        let disp = vec![1.0f32; n * 3]; // displacement pushes values up
        let target = flat_verts(n, 1.0, 1.0, 1.0); // target is below template
        let basis = BlendBasis::new(vec![disp]).unwrap();
        let config = BlendSolverConfig::default();
        let result = nonneg_solve_blend_shapes(&template, &target, &basis, &config).unwrap();
        assert!(result.weights.iter().all(|&w| w >= -1e-6));
    }

    // -----------------------------------------------------------------------
    // fit_expression_coefficients
    // -----------------------------------------------------------------------

    #[test]
    fn test_fit_expression_coefficients_smoke() {
        let n = 6usize;
        let template = flat_verts(n, 0.0, 0.0, 0.0);
        let target = flat_verts(n, 0.1, 0.2, 0.0);
        let expr_basis = vec![vec![1.0f32; n * 3], vec![0.0f32; n * 3]];
        let config = BlendSolverConfig::default();
        let coeffs =
            fit_expression_coefficients(&template, &target, expr_basis, 2, &config).unwrap();
        assert_eq!(coeffs.len(), 2);
    }

    #[test]
    fn test_fit_expression_coefficients_n_coeffs_trims_basis() {
        let n = 4usize;
        let template = flat_verts(n, 0.0, 0.0, 0.0);
        let target = flat_verts(n, 0.0, 0.0, 0.0);
        let expr_basis = vec![
            vec![1.0f32; n * 3],
            vec![2.0f32; n * 3],
            vec![3.0f32; n * 3],
        ];
        let config = BlendSolverConfig::default();
        // Request only 2 coefficients from 3-vector basis
        let coeffs =
            fit_expression_coefficients(&template, &target, expr_basis, 2, &config).unwrap();
        assert_eq!(coeffs.len(), 2);
    }

    // -----------------------------------------------------------------------
    // blend_solver_residual_curve
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_solver_residual_curve_length() {
        let n = 3usize;
        let template = flat_verts(n, 0.0, 0.0, 0.0);
        let target = flat_verts(n, 1.0, 0.0, 0.0);
        let basis = BlendBasis::new(uniform_basis(n, 1.0, 0.0, 0.0)).unwrap();
        let weights_seq: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32 * 0.25]).collect();
        let curve = blend_solver_residual_curve(&template, &target, &basis, &weights_seq);
        assert_eq!(curve.len(), 5);
    }

    #[test]
    fn test_blend_solver_residual_curve_decreasing_toward_target() {
        // weights [0.0, 0.5, 1.0] applied to basis that spans the target distance.
        let n = 2usize;
        let template = flat_verts(n, 0.0, 0.0, 0.0);
        let target = flat_verts(n, 1.0, 0.0, 0.0);
        let basis = BlendBasis::new(uniform_basis(n, 1.0, 0.0, 0.0)).unwrap();
        let weights_seq = vec![vec![0.0], vec![0.5], vec![1.0]];
        let curve = blend_solver_residual_curve(&template, &target, &basis, &weights_seq);
        // residual should decrease as weight approaches 1.0
        assert!(curve[0] > curve[1]);
        assert!(curve[1] > curve[2]);
        assert!(curve[2] < 1e-5);
    }

    #[test]
    fn test_blend_solver_residual_curve_empty_sequence() {
        let template = flat_verts(2, 0.0, 0.0, 0.0);
        let target = flat_verts(2, 1.0, 0.0, 0.0);
        let basis = BlendBasis::new(uniform_basis(2, 1.0, 0.0, 0.0)).unwrap();
        let curve = blend_solver_residual_curve(&template, &target, &basis, &[]);
        assert!(curve.is_empty());
    }

    // -----------------------------------------------------------------------
    // format_solver_result
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_solver_result_non_empty() {
        let result = BlendSolverResult {
            weights: vec![0.5, 0.0, 1.0],
            residual: 0.023,
            n_iterations: 42,
            converged: true,
            vertex_errors: vec![0.01, 0.02],
        };
        let stats = compute_solver_stats(&result);
        let s = format_solver_result(&result, &stats);
        assert!(!s.is_empty());
        assert!(s.contains("converged=true"));
    }

    #[test]
    fn test_format_solver_result_contains_active_weights() {
        let result = BlendSolverResult {
            weights: vec![1.0, 0.0],
            residual: 0.05,
            n_iterations: 10,
            converged: false,
            vertex_errors: vec![0.05],
        };
        let stats = compute_solver_stats(&result);
        let s = format_solver_result(&result, &stats);
        // Should mention active_weights=1/2
        assert!(s.contains("1/2"));
    }

    // -----------------------------------------------------------------------
    // BlendSolverResult fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_solver_result_fields() {
        let result = BlendSolverResult {
            weights: vec![0.1, 0.2],
            residual: 0.05,
            n_iterations: 7,
            converged: true,
            vertex_errors: vec![0.03, 0.07],
        };
        assert_eq!(result.weights.len(), 2);
        assert!((result.residual - 0.05).abs() < 1e-6);
        assert_eq!(result.n_iterations, 7);
        assert!(result.converged);
        assert_eq!(result.vertex_errors.len(), 2);
    }

    // -----------------------------------------------------------------------
    // SolverStats fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_solver_stats_fields() {
        let stats = SolverStats {
            mean_vertex_error: 0.1,
            max_vertex_error: 0.5,
            weight_l2_norm: 1.4,
            weight_sparsity: 0.3,
            n_active_weights: 7,
        };
        assert!((stats.mean_vertex_error - 0.1).abs() < 1e-6);
        assert!((stats.max_vertex_error - 0.5).abs() < 1e-6);
        assert_eq!(stats.n_active_weights, 7);
    }

    // -----------------------------------------------------------------------
    // Additional edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_blend_displacements_multiple_basis_independent() {
        // Two basis vectors: one shifts x, one shifts y.
        let template = vec![0.0f32, 0.0, 0.0];
        let disp_x = vec![1.0f32, 0.0, 0.0];
        let disp_y = vec![0.0f32, 1.0, 0.0];
        let result = apply_blend_displacements(&template, &[disp_x, disp_y], &[3.0, 5.0]);
        assert!((result[0] - 3.0).abs() < 1e-6);
        assert!((result[1] - 5.0).abs() < 1e-6);
        assert!(result[2].abs() < 1e-6);
    }

    #[test]
    fn test_compute_vertex_errors_empty() {
        let errors = compute_vertex_errors(&[], &[]);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_solve_blend_shapes_multi_basis() {
        // Two basis vectors, target is 0.5*disp0 + 0.5*disp1.
        let n = 4usize;
        let template = flat_verts(n, 0.0, 0.0, 0.0);
        let disp0: Vec<f32> = (0..n * 3)
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let disp1: Vec<f32> = (0..n * 3)
            .map(|i| if i % 3 == 1 { 1.0 } else { 0.0 })
            .collect();
        let target: Vec<f32> = (0..n * 3)
            .map(|i| if i % 3 < 2 { 0.5 } else { 0.0 })
            .collect();
        let basis = BlendBasis::new(vec![disp0, disp1]).unwrap();
        let constraints = WeightConstraints::unconstrained(2);
        let config = BlendSolverConfig {
            max_iter: 1000,
            step_size: 0.05,
            tolerance: 1e-8,
            ..Default::default()
        };
        let result = solve_blend_shapes(&template, &target, &basis, &constraints, &config).unwrap();
        // Both weights should be near 0.5
        assert!(
            (result.weights[0] - 0.5).abs() < 0.2,
            "w0={}",
            result.weights[0]
        );
        assert!(
            (result.weights[1] - 0.5).abs() < 0.2,
            "w1={}",
            result.weights[1]
        );
    }

    #[test]
    fn test_solver_stats_zero_weights() {
        let result = BlendSolverResult {
            weights: vec![],
            residual: 0.0,
            n_iterations: 0,
            converged: true,
            vertex_errors: vec![],
        };
        let stats = compute_solver_stats(&result);
        assert_eq!(stats.n_active_weights, 0);
        assert_eq!(stats.weight_sparsity, 0.0);
    }

    #[test]
    fn test_project_weights_no_symmetry() {
        let constraints = WeightConstraints::unconstrained(3);
        let weights = vec![0.1, 0.2, 0.3];
        let projected = project_weights(&weights, &constraints);
        assert_eq!(projected, weights);
    }
}
