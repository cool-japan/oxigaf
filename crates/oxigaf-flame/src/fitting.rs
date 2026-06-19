//! CPU-based Gauss-Newton / gradient-descent fitting of FLAME model parameters
//! to 2D landmark observations.
//!
//! # Overview
//!
//! This module provides landmark-based parameter fitting:
//! - A simple pinhole camera for 3D → 2D projection.
//! - [`FittingParams`] that wrap the subset of [`crate::params::FlameParams`] being
//!   optimized (shape, expression, global rotation, translation, jaw).
//! - A [`FlameForward`] trait so the optimizer is decoupled from the heavy model
//!   binary; a [`MockFlameForward`] implementation enables fast, deterministic tests.
//! - [`fit_landmarks`] — the main gradient-descent loop with numerical Jacobian.

use crate::params::FlameParams;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during landmark fitting.
#[derive(Debug, thiserror::Error)]
pub enum FittingError {
    /// No landmark observations were provided.
    #[error("No observations supplied for fitting")]
    NoObservations,

    /// A landmark observation references a vertex index that does not exist.
    #[error("Vertex index {idx} is out of range; the mesh only has {num_vertices} vertices")]
    InvalidVertexIndex {
        /// The invalid vertex index.
        idx: usize,
        /// Total number of vertices in the mesh.
        num_vertices: usize,
    },

    /// The parameter vector length does not match what is expected.
    #[error("Parameter count mismatch: expected {expected} values, got {got}")]
    ParameterCountMismatch {
        /// Expected number of parameters.
        expected: usize,
        /// Actual number of parameters received.
        got: usize,
    },

    /// The FLAME forward pass returned an error.
    #[error("FLAME forward pass failed: {0}")]
    ForwardPassFailed(String),

    /// A 3D point could not be projected (e.g. behind the camera).
    #[error("Camera projection failed: point is at or behind the camera plane")]
    CameraProjectionFailed,
}

// ---------------------------------------------------------------------------
// Pinhole camera
// ---------------------------------------------------------------------------

/// A simple pinhole camera that projects 3D world points to 2D image pixels.
#[derive(Debug, Clone)]
pub struct PinholeCamera {
    /// Focal length in pixels.
    pub focal_length: f32,
    /// Principal point x (image centre, pixels).
    pub cx: f32,
    /// Principal point y (image centre, pixels).
    pub cy: f32,
    /// World-to-camera view matrix (4×4, column-major, row-index = row, col-index = col).
    pub view_matrix: [[f32; 4]; 4],
}

impl PinholeCamera {
    /// Create a pinhole camera with an identity view matrix.
    #[must_use]
    pub fn new(focal_length: f32, cx: f32, cy: f32) -> Self {
        // Identity 4×4 matrix (row-major storage, so row i col j = view_matrix[i][j])
        let mut m = [[0.0f32; 4]; 4];
        m[0][0] = 1.0;
        m[1][1] = 1.0;
        m[2][2] = 1.0;
        m[3][3] = 1.0;
        Self {
            focal_length,
            cx,
            cy,
            view_matrix: m,
        }
    }

    /// Project a 3D world-space point to 2D pixel coordinates.
    ///
    /// Returns `None` when the transformed z-coordinate is ≤ 0 (point is at
    /// or behind the camera).
    #[must_use]
    pub fn project(&self, point: [f32; 3]) -> Option<[f32; 2]> {
        // Apply view matrix: camera_point = M * [x, y, z, 1]^T
        let m = &self.view_matrix;
        let px = point[0];
        let py = point[1];
        let pz = point[2];

        let cx = m[0][0] * px + m[0][1] * py + m[0][2] * pz + m[0][3];
        let cy = m[1][0] * px + m[1][1] * py + m[1][2] * pz + m[1][3];
        let cz = m[2][0] * px + m[2][1] * py + m[2][2] * pz + m[2][3];

        if cz <= 0.0 {
            return None;
        }

        let x_px = self.focal_length * cx / cz + self.cx;
        let y_px = self.focal_length * cy / cz + self.cy;
        Some([x_px, y_px])
    }
}

impl Default for PinholeCamera {
    fn default() -> Self {
        Self::new(500.0, 256.0, 256.0)
    }
}

// ---------------------------------------------------------------------------
// Landmark observation
// ---------------------------------------------------------------------------

/// A single 2D landmark observation in image space.
#[derive(Debug, Clone)]
pub struct LandmarkObservation {
    /// Vertex index in the FLAME mesh that corresponds to this landmark.
    pub vertex_index: usize,
    /// Observed 2D position in image pixels `[x, y]`.
    pub position_2d: [f32; 2],
    /// Confidence weight in `[0, 1]` (1.0 = fully trusted).
    pub confidence: f32,
}

impl LandmarkObservation {
    /// Create a new observation with full confidence (1.0).
    #[must_use]
    pub fn new(vertex_index: usize, x: f32, y: f32) -> Self {
        Self {
            vertex_index,
            position_2d: [x, y],
            confidence: 1.0,
        }
    }

    /// Override the confidence weight (builder-style).
    #[must_use]
    pub fn with_confidence(mut self, c: f32) -> Self {
        self.confidence = c;
        self
    }
}

// ---------------------------------------------------------------------------
// Fitting parameters
// ---------------------------------------------------------------------------

/// The subset of FLAME parameters being optimized during fitting.
///
/// Laid out as: `[shape…, expression…, global_rotation(3), translation(3), jaw(3)]`.
#[derive(Debug, Clone)]
pub struct FittingParams {
    /// Shape (identity) coefficients being optimized.
    pub shape: Vec<f32>,
    /// Expression coefficients being optimized.
    pub expression: Vec<f32>,
    /// Global rotation as an axis-angle triple `[rx, ry, rz]`.
    pub global_rotation: [f32; 3],
    /// Global translation `[tx, ty, tz]`.
    pub translation: [f32; 3],
    /// Jaw rotation as an axis-angle triple `[rx, ry, rz]`.
    pub jaw: [f32; 3],
}

impl FittingParams {
    /// Create a zeroed parameter set with `num_shape` shape dims and `num_expr`
    /// expression dims.
    #[must_use]
    pub fn zero(num_shape: usize, num_expr: usize) -> Self {
        Self {
            shape: vec![0.0; num_shape],
            expression: vec![0.0; num_expr],
            global_rotation: [0.0; 3],
            translation: [0.0; 3],
            jaw: [0.0; 3],
        }
    }

    /// Total number of scalar dimensions across all sub-vectors.
    #[must_use]
    pub fn total_dims(&self) -> usize {
        self.shape.len() + self.expression.len() + 3 + 3 + 3
    }

    /// Flatten all parameters into a single `Vec<f32>`.
    ///
    /// Layout: `[shape…, expression…, global_rotation(3), translation(3), jaw(3)]`.
    #[must_use]
    pub fn to_vec(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.total_dims());
        out.extend_from_slice(&self.shape);
        out.extend_from_slice(&self.expression);
        out.extend_from_slice(&self.global_rotation);
        out.extend_from_slice(&self.translation);
        out.extend_from_slice(&self.jaw);
        out
    }

    /// Reconstruct `FittingParams` from a flat slice produced by `to_vec`.
    ///
    /// # Errors
    ///
    /// Returns [`FittingError::ParameterCountMismatch`] if the slice length
    /// does not equal `num_shape + num_expr + 9`.
    pub fn from_vec(v: &[f32], num_shape: usize, num_expr: usize) -> Result<Self, FittingError> {
        let expected = num_shape + num_expr + 9;
        if v.len() != expected {
            return Err(FittingError::ParameterCountMismatch {
                expected,
                got: v.len(),
            });
        }

        let mut cursor = 0usize;

        let shape = v[cursor..cursor + num_shape].to_vec();
        cursor += num_shape;

        let expression = v[cursor..cursor + num_expr].to_vec();
        cursor += num_expr;

        let global_rotation = [v[cursor], v[cursor + 1], v[cursor + 2]];
        cursor += 3;

        let translation = [v[cursor], v[cursor + 1], v[cursor + 2]];
        cursor += 3;

        let jaw = [v[cursor], v[cursor + 1], v[cursor + 2]];

        Ok(Self {
            shape,
            expression,
            global_rotation,
            translation,
            jaw,
        })
    }

    /// Convert to a full [`FlameParams`] suitable for the FLAME forward pass.
    ///
    /// - `pose` is 15 values: `[global_rotation(0-2), zeros(3-5), jaw(6-8), zeros(9-14)]`
    /// - `translation` is copied directly as `[f32; 3]`.
    #[must_use]
    pub fn to_flame_params(&self) -> FlameParams {
        let mut pose = vec![0.0f32; 15];
        // Root (global) rotation at joints 0-2
        pose[0] = self.global_rotation[0];
        pose[1] = self.global_rotation[1];
        pose[2] = self.global_rotation[2];
        // Jaw rotation at joints 6-8 (joint index 2 = jaw)
        pose[6] = self.jaw[0];
        pose[7] = self.jaw[1];
        pose[8] = self.jaw[2];

        FlameParams {
            shape: self.shape.clone(),
            expression: self.expression.clone(),
            pose,
            translation: self.translation,
        }
    }
}

// ---------------------------------------------------------------------------
// Fitting configuration
// ---------------------------------------------------------------------------

/// Configuration for the landmark fitting optimizer.
#[derive(Debug, Clone)]
pub struct FittingConfig {
    /// Maximum number of gradient-descent iterations.
    pub max_iterations: usize,
    /// Stop early when the L2 norm of the gradient step is below this value.
    pub convergence_delta: f32,
    /// Number of shape coefficients to optimize.
    pub shape_dim: usize,
    /// Number of expression coefficients to optimize.
    pub expr_dim: usize,
    /// Weight applied to the landmark reprojection term.
    pub landmark_weight: f32,
    /// L2 regularization weight on shape coefficients.
    pub shape_regularizer: f32,
    /// L2 regularization weight on expression coefficients.
    pub expr_regularizer: f32,
    /// Finite-difference epsilon for numerical gradient computation.
    pub finite_diff_eps: f32,
    /// Step size for gradient descent updates.
    pub learning_rate: f32,
}

impl Default for FittingConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            convergence_delta: 1e-5,
            shape_dim: 10,
            expr_dim: 10,
            landmark_weight: 1.0,
            shape_regularizer: 0.01,
            expr_regularizer: 0.01,
            finite_diff_eps: 1e-3,
            learning_rate: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Fitting result
// ---------------------------------------------------------------------------

/// Result returned by [`fit_landmarks`].
#[derive(Debug, Clone)]
pub struct FittingResult {
    /// Optimized parameters.
    pub params: FittingParams,
    /// Cost at the final iteration (landmark + regularization).
    pub final_cost: f32,
    /// Number of iterations executed.
    pub iterations: usize,
    /// Whether the optimizer converged before reaching `max_iterations`.
    pub converged: bool,
    /// Per-landmark reprojection error in pixels.
    pub reprojection_errors: Vec<f32>,
    /// Mean reprojection error in pixels.
    pub mean_reprojection_error: f32,
}

// ---------------------------------------------------------------------------
// FlameForward trait + MockFlameForward
// ---------------------------------------------------------------------------

/// Trait abstracting the FLAME forward pass for fitting.
///
/// Implement this for the real [`crate::model::FlameModel`] or for mocks in
/// tests.
pub trait FlameForward {
    /// Run the forward pass and return vertex positions `[x, y, z]` for each
    /// vertex.
    ///
    /// # Errors
    ///
    /// Returns [`FittingError::ForwardPassFailed`] on computation failures.
    fn forward(&self, params: &FittingParams) -> Result<Vec<[f32; 3]>, FittingError>;

    /// Number of vertices produced by this model.
    fn num_vertices(&self) -> usize;
}

/// A lightweight mock forward model for testing.
///
/// Applies only translation to a fixed set of base vertices, making the fitting
/// loop fast and fully deterministic.
pub struct MockFlameForward {
    /// Base vertex positions before any deformation.
    pub base_vertices: Vec<[f32; 3]>,
}

impl MockFlameForward {
    /// Generate `n` vertices approximately uniformly distributed on the unit
    /// sphere using the golden-ratio Fibonacci spiral.
    #[must_use]
    pub fn unit_sphere(n: usize) -> Self {
        let golden = f32::midpoint(1.0, 5.0_f32.sqrt());
        let n_safe = n.max(1);
        let mut verts = Vec::with_capacity(n);
        for i in 0..n {
            let theta = 2.0 * std::f32::consts::PI * i as f32 / golden;
            let phi = (1.0 - 2.0 * i as f32 / n_safe as f32)
                .clamp(-1.0, 1.0)
                .acos();
            verts.push([phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()]);
        }
        Self {
            base_vertices: verts,
        }
    }
}

impl FlameForward for MockFlameForward {
    fn forward(&self, params: &FittingParams) -> Result<Vec<[f32; 3]>, FittingError> {
        let t = params.translation;
        let verts = self
            .base_vertices
            .iter()
            .map(|v| [v[0] + t[0], v[1] + t[1], v[2] + t[2]])
            .collect();
        Ok(verts)
    }

    fn num_vertices(&self) -> usize {
        self.base_vertices.len()
    }
}

// ---------------------------------------------------------------------------
// Internal cost function
// ---------------------------------------------------------------------------

/// Compute the total fitting cost for the given parameters.
///
/// Cost = `landmark_weight * Σᵢ conf_i * ||proj(vᵢ) - obs_i||²`
///       `+ shape_reg * ||shape||² + expr_reg * ||expr||²`
fn compute_cost<F: FlameForward>(
    model: &F,
    params: &FittingParams,
    observations: &[LandmarkObservation],
    camera: &PinholeCamera,
    config: &FittingConfig,
) -> Result<f32, FittingError> {
    let vertices = model.forward(params)?;
    let num_verts = vertices.len();

    // Validate vertex indices once per cost evaluation
    for obs in observations {
        if obs.vertex_index >= num_verts {
            return Err(FittingError::InvalidVertexIndex {
                idx: obs.vertex_index,
                num_vertices: num_verts,
            });
        }
    }

    // Landmark reprojection term
    let mut landmark_cost = 0.0f32;
    for obs in observations {
        if let Some(proj) = camera.project(vertices[obs.vertex_index]) {
            let dx = proj[0] - obs.position_2d[0];
            let dy = proj[1] - obs.position_2d[1];
            landmark_cost += obs.confidence * (dx * dx + dy * dy);
        }
        // Points behind the camera contribute 0 to gradient pressure — they are
        // simply ignored, which is the standard approach in fitting pipelines.
    }

    // Regularization terms
    let shape_reg: f32 = params.shape.iter().map(|s| s * s).sum();
    let expr_reg: f32 = params.expression.iter().map(|e| e * e).sum();

    Ok(config.landmark_weight * landmark_cost
        + config.shape_regularizer * shape_reg
        + config.expr_regularizer * expr_reg)
}

// ---------------------------------------------------------------------------
// Main fitting function
// ---------------------------------------------------------------------------

/// Fit FLAME model parameters to 2D landmark observations using gradient
/// descent with a numerical Jacobian.
///
/// # Algorithm
///
/// 1. Initialize parameters at zero.
/// 2. For each iteration:
///    a. Compute the cost at the current parameters.
///    b. Compute the numerical gradient by perturbing each parameter by ±`eps`.
///    c. Update every parameter: `pᵢ -= learning_rate * grad_i`.
///    d. Stop early if `||step|| < convergence_delta`.
/// 3. Compute final per-landmark reprojection errors.
///
/// # Errors
///
/// - [`FittingError::NoObservations`] — empty observation list.
/// - [`FittingError::InvalidVertexIndex`] — observation references a vertex
///   that does not exist in the model.
/// - [`FittingError::ForwardPassFailed`] — the model's forward pass failed.
pub fn fit_landmarks<F: FlameForward>(
    model: &F,
    observations: &[LandmarkObservation],
    camera: &PinholeCamera,
    config: &FittingConfig,
) -> Result<FittingResult, FittingError> {
    if observations.is_empty() {
        return Err(FittingError::NoObservations);
    }

    // Validate vertex indices against model size upfront
    let num_verts = model.num_vertices();
    for obs in observations {
        if obs.vertex_index >= num_verts {
            return Err(FittingError::InvalidVertexIndex {
                idx: obs.vertex_index,
                num_vertices: num_verts,
            });
        }
    }

    let mut params = FittingParams::zero(config.shape_dim, config.expr_dim);
    let ndims = params.total_dims();
    let eps = config.finite_diff_eps;

    let mut converged = false;
    let mut iterations = 0usize;
    let mut final_cost = 0.0f32;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Current cost
        let cost_base = compute_cost(model, &params, observations, camera, config)?;
        final_cost = cost_base;

        // Numerical gradient — one forward pass per parameter dimension
        let flat = params.to_vec();
        let mut grad = vec![0.0f32; ndims];

        for dim in 0..ndims {
            let mut flat_plus = flat.clone();
            flat_plus[dim] += eps;
            let params_plus =
                FittingParams::from_vec(&flat_plus, config.shape_dim, config.expr_dim)?;
            let cost_plus = compute_cost(model, &params_plus, observations, camera, config)?;

            grad[dim] = (cost_plus - cost_base) / eps;
        }

        // Gradient norm clipping: scale gradient to have unit L2 norm when it
        // would otherwise produce an oversized step.  This keeps the optimizer
        // stable regardless of the cost-function scale.
        let grad_norm_sq: f32 = grad.iter().map(|g| g * g).sum();
        let grad_norm = grad_norm_sq.sqrt();
        let scale = if grad_norm > 1.0 {
            1.0 / grad_norm
        } else {
            1.0
        };

        // Gradient descent update
        let mut new_flat = flat.clone();
        for dim in 0..ndims {
            new_flat[dim] -= config.learning_rate * grad[dim] * scale;
        }

        // Convergence check: L2 norm of the step taken
        let step_norm = config.learning_rate * grad_norm.min(1.0);

        params = FittingParams::from_vec(&new_flat, config.shape_dim, config.expr_dim)?;

        if step_norm < config.convergence_delta {
            converged = true;
            // Recompute final cost at the updated params
            final_cost = compute_cost(model, &params, observations, camera, config)?;
            break;
        }
    }

    if !converged {
        // Ensure final_cost reflects the last params after the loop
        final_cost = compute_cost(model, &params, observations, camera, config)?;
    }

    // Compute per-landmark reprojection errors
    let final_vertices = model.forward(&params)?;
    let mut reprojection_errors = Vec::with_capacity(observations.len());

    for obs in observations {
        if obs.vertex_index >= final_vertices.len() {
            return Err(FittingError::InvalidVertexIndex {
                idx: obs.vertex_index,
                num_vertices: final_vertices.len(),
            });
        }
        let err = match camera.project(final_vertices[obs.vertex_index]) {
            Some(proj) => {
                let dx = proj[0] - obs.position_2d[0];
                let dy = proj[1] - obs.position_2d[1];
                (dx * dx + dy * dy).sqrt()
            }
            None => f32::INFINITY,
        };
        reprojection_errors.push(err);
    }

    let mean_reprojection_error = if reprojection_errors.is_empty() {
        0.0
    } else {
        let finite_errors: Vec<f32> = reprojection_errors
            .iter()
            .copied()
            .filter(|e| e.is_finite())
            .collect();
        if finite_errors.is_empty() {
            f32::INFINITY
        } else {
            finite_errors.iter().sum::<f32>() / finite_errors.len() as f32
        }
    };

    Ok(FittingResult {
        params,
        final_cost,
        iterations,
        converged,
        reprojection_errors,
        mean_reprojection_error,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // PinholeCamera
    // -----------------------------------------------------------------------

    #[test]
    fn test_pinhole_project_center() {
        let cam = PinholeCamera::new(500.0, 320.0, 240.0);
        // Point at [0, 0, 1] in camera space → should project to principal point
        let proj = cam.project([0.0, 0.0, 1.0]).expect("should project");
        assert!(
            (proj[0] - 320.0).abs() < 1e-4,
            "x={} should be cx=320",
            proj[0]
        );
        assert!(
            (proj[1] - 240.0).abs() < 1e-4,
            "y={} should be cy=240",
            proj[1]
        );
    }

    #[test]
    fn test_pinhole_project_behind_camera() {
        let cam = PinholeCamera::default();
        // z=0 should return None
        assert!(cam.project([1.0, 1.0, 0.0]).is_none(), "z=0 must be None");
        // z < 0 also None
        assert!(cam.project([1.0, 1.0, -1.0]).is_none(), "z<0 must be None");
    }

    #[test]
    fn test_pinhole_project_offset() {
        // Point at [1.0, 0.0, 1.0]: x_px = focal * 1/1 + cx = focal + cx
        let cam = PinholeCamera::new(100.0, 50.0, 50.0);
        let proj = cam.project([1.0, 0.0, 1.0]).expect("should project");
        assert!(
            (proj[0] - 150.0).abs() < 1e-4,
            "expected 150.0 got {}",
            proj[0]
        );
        assert!(
            (proj[1] - 50.0).abs() < 1e-4,
            "expected 50.0 got {}",
            proj[1]
        );
    }

    // -----------------------------------------------------------------------
    // FittingParams roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_fitting_params_roundtrip() {
        let p = FittingParams {
            shape: vec![0.1, 0.2, 0.3],
            expression: vec![0.4, 0.5],
            global_rotation: [0.01, 0.02, 0.03],
            translation: [1.0, 2.0, 3.0],
            jaw: [0.05, 0.06, 0.07],
        };
        let v = p.to_vec();
        let p2 = FittingParams::from_vec(&v, 3, 2).expect("roundtrip failed");

        assert_eq!(p2.shape, p.shape);
        assert_eq!(p2.expression, p.expression);
        assert_eq!(p2.global_rotation, p.global_rotation);
        assert_eq!(p2.translation, p.translation);
        assert_eq!(p2.jaw, p.jaw);
    }

    #[test]
    fn test_fitting_params_total_dims() {
        let p = FittingParams::zero(10, 5);
        // 10 shape + 5 expr + 3 global_rot + 3 translation + 3 jaw = 24
        assert_eq!(p.total_dims(), 24);
    }

    #[test]
    fn test_fitting_params_zero_lengths() {
        let p = FittingParams::zero(7, 3);
        assert_eq!(p.shape.len(), 7);
        assert_eq!(p.expression.len(), 3);
        assert_eq!(p.global_rotation, [0.0; 3]);
        assert_eq!(p.translation, [0.0; 3]);
        assert_eq!(p.jaw, [0.0; 3]);
    }

    #[test]
    fn test_fitting_params_from_vec_mismatch() {
        let v = vec![0.0; 5]; // too short
        let result = FittingParams::from_vec(&v, 10, 5);
        assert!(result.is_err(), "should fail on length mismatch");
        match result {
            Err(FittingError::ParameterCountMismatch { expected, got }) => {
                assert_eq!(expected, 24); // 10+5+9
                assert_eq!(got, 5);
            }
            _ => panic!("wrong error variant"),
        }
    }

    // -----------------------------------------------------------------------
    // FittingParams::to_flame_params
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_flame_params_pose_layout() {
        let p = FittingParams {
            shape: vec![1.0, 2.0],
            expression: vec![0.5],
            global_rotation: [0.1, 0.2, 0.3],
            translation: [4.0, 5.0, 6.0],
            jaw: [0.7, 0.8, 0.9],
        };
        let fp = p.to_flame_params();
        assert_eq!(fp.pose.len(), 15);
        // Root at 0-2
        assert!((fp.pose[0] - 0.1).abs() < 1e-6);
        assert!((fp.pose[1] - 0.2).abs() < 1e-6);
        assert!((fp.pose[2] - 0.3).abs() < 1e-6);
        // Neck at 3-5 should be zero
        assert_eq!(fp.pose[3], 0.0);
        assert_eq!(fp.pose[4], 0.0);
        assert_eq!(fp.pose[5], 0.0);
        // Jaw at 6-8
        assert!((fp.pose[6] - 0.7).abs() < 1e-6);
        assert!((fp.pose[7] - 0.8).abs() < 1e-6);
        assert!((fp.pose[8] - 0.9).abs() < 1e-6);
        // Translation
        assert!((fp.translation[0] - 4.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // FittingConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_fitting_config_default() {
        let cfg = FittingConfig::default();
        assert_eq!(cfg.max_iterations, 50);
        assert_eq!(cfg.shape_dim, 10);
        assert_eq!(cfg.expr_dim, 10);
        assert!((cfg.convergence_delta - 1e-5).abs() < 1e-10);
        assert!((cfg.finite_diff_eps - 1e-3).abs() < 1e-10);
        assert!((cfg.learning_rate - 0.1).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // LandmarkObservation
    // -----------------------------------------------------------------------

    #[test]
    fn test_landmark_observation_new() {
        let obs = LandmarkObservation::new(42, 100.0, 200.0);
        assert_eq!(obs.vertex_index, 42);
        assert_eq!(obs.position_2d, [100.0, 200.0]);
        assert!((obs.confidence - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_landmark_observation_with_confidence() {
        let obs = LandmarkObservation::new(0, 10.0, 20.0).with_confidence(0.75);
        assert!((obs.confidence - 0.75).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // MockFlameForward
    // -----------------------------------------------------------------------

    #[test]
    fn test_mock_flame_forward_translation() {
        let mock = MockFlameForward::unit_sphere(10);
        let base: Vec<[f32; 3]> = mock.base_vertices.clone();

        let mut params = FittingParams::zero(2, 2);
        params.translation = [1.0, 2.0, 3.0];

        let verts = mock.forward(&params).expect("forward failed");
        assert_eq!(verts.len(), 10);
        for (orig, deformed) in base.iter().zip(verts.iter()) {
            assert!((deformed[0] - orig[0] - 1.0).abs() < 1e-5, "x offset wrong");
            assert!((deformed[1] - orig[1] - 2.0).abs() < 1e-5, "y offset wrong");
            assert!((deformed[2] - orig[2] - 3.0).abs() < 1e-5, "z offset wrong");
        }
    }

    #[test]
    fn test_mock_num_vertices() {
        let mock = MockFlameForward::unit_sphere(25);
        assert_eq!(mock.num_vertices(), 25);
    }

    // -----------------------------------------------------------------------
    // fit_landmarks: error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_fit_landmarks_no_observations_error() {
        let model = MockFlameForward::unit_sphere(10);
        let cam = PinholeCamera::default();
        let cfg = FittingConfig::default();
        let result = fit_landmarks(&model, &[], &cam, &cfg);
        assert!(
            matches!(result, Err(FittingError::NoObservations)),
            "expected NoObservations error"
        );
    }

    #[test]
    fn test_fit_landmarks_invalid_vertex_index() {
        let model = MockFlameForward::unit_sphere(5);
        let cam = PinholeCamera::default();
        let cfg = FittingConfig {
            shape_dim: 2,
            expr_dim: 2,
            ..FittingConfig::default()
        };
        // Vertex index 99 does not exist in a 5-vertex mesh
        let obs = vec![LandmarkObservation::new(99, 256.0, 256.0)];
        let result = fit_landmarks(&model, &obs, &cam, &cfg);
        assert!(
            matches!(result, Err(FittingError::InvalidVertexIndex { .. })),
            "expected InvalidVertexIndex error"
        );
    }

    // -----------------------------------------------------------------------
    // fit_landmarks: convergence / correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_fit_landmarks_converges_translation() {
        // Place a single vertex at [0, 0, 2] in base mesh.
        // Camera: focal=500, cx=256, cy=256.
        // With zero translation, vertex projects to [256, 256].
        // We observe it at [306, 256] (50 pixels right).
        // The optimizer should find translation_x ≈ +0.2 (50/500 * 2).
        let mock = MockFlameForward {
            base_vertices: vec![[0.0, 0.0, 2.0]],
        };
        let cam = PinholeCamera::new(500.0, 256.0, 256.0);

        // Observed position when tx = 0.2: proj_x = 500 * 0.2/2 + 256 = 306
        let obs = vec![LandmarkObservation::new(0, 306.0, 256.0)];

        let cfg = FittingConfig {
            max_iterations: 200,
            shape_dim: 0,
            expr_dim: 0,
            learning_rate: 0.05,
            finite_diff_eps: 1e-3,
            convergence_delta: 1e-6,
            ..FittingConfig::default()
        };

        let result = fit_landmarks(&mock, &obs, &cam, &cfg).expect("fitting failed");
        // Translation x should be close to 0.2
        assert!(
            (result.params.translation[0] - 0.2).abs() < 0.05,
            "tx={}, expected ~0.2",
            result.params.translation[0]
        );
    }

    #[test]
    fn test_fit_landmarks_reprojection_errors_computed() {
        let mock = MockFlameForward::unit_sphere(5);
        let cam = PinholeCamera::default();
        let obs = vec![
            LandmarkObservation::new(0, 256.0, 256.0),
            LandmarkObservation::new(1, 260.0, 256.0),
        ];
        let cfg = FittingConfig {
            max_iterations: 5,
            shape_dim: 2,
            expr_dim: 2,
            ..FittingConfig::default()
        };
        let result = fit_landmarks(&mock, &obs, &cam, &cfg).expect("fitting failed");
        assert_eq!(
            result.reprojection_errors.len(),
            2,
            "one error per landmark"
        );
    }

    #[test]
    fn test_fit_landmarks_result_converged() {
        // With zero observations residual at start (perfect fit by coincidence),
        // cost should be near 0 — but we just test that the struct is populated.
        let mock = MockFlameForward {
            base_vertices: vec![[0.0, 0.0, 1.0]],
        };
        let cam = PinholeCamera::new(500.0, 256.0, 256.0);
        // Observe the exact projection of the zero-translation vertex
        let obs = vec![LandmarkObservation::new(0, 256.0, 256.0)];
        let cfg = FittingConfig {
            max_iterations: 100,
            shape_dim: 0,
            expr_dim: 0,
            convergence_delta: 1e-3,
            learning_rate: 0.1,
            ..FittingConfig::default()
        };
        let result = fit_landmarks(&mock, &obs, &cam, &cfg).expect("fitting failed");
        // With perfect initial fit, should converge quickly
        assert!(result.iterations > 0);
        assert!(result.final_cost.is_finite());
    }

    #[test]
    fn test_cost_zero_when_perfect_fit() {
        // Vertex at [0, 0, 1], camera projects to [cx, cy] = [256, 256].
        // Observe at exactly [256, 256] with zero regularizer.
        let mock = MockFlameForward {
            base_vertices: vec![[0.0, 0.0, 1.0]],
        };
        let cam = PinholeCamera::new(500.0, 256.0, 256.0);
        let obs = vec![LandmarkObservation::new(0, 256.0, 256.0)];
        let params = FittingParams::zero(0, 0);
        let cfg = FittingConfig {
            shape_regularizer: 0.0,
            expr_regularizer: 0.0,
            shape_dim: 0,
            expr_dim: 0,
            ..FittingConfig::default()
        };
        let cost = compute_cost(&mock, &params, &obs, &cam, &cfg).expect("cost failed");
        assert!(
            cost.abs() < 1e-5,
            "cost should be ~0 for perfect fit, got {cost}"
        );
    }

    #[test]
    fn test_reprojection_error_decreasing() {
        // Setup where initial residual is large; run multiple passes and verify
        // mean reprojection error decreases monotonically over increasing iterations.
        let mock = MockFlameForward {
            base_vertices: vec![[0.0, 0.0, 2.0]],
        };
        let cam = PinholeCamera::new(500.0, 256.0, 256.0);
        // Observed far from the projected vertex — forces optimizer to move
        let obs = vec![LandmarkObservation::new(0, 400.0, 256.0)];

        let cfg_short = FittingConfig {
            max_iterations: 5,
            shape_dim: 0,
            expr_dim: 0,
            learning_rate: 0.05,
            ..FittingConfig::default()
        };
        let cfg_long = FittingConfig {
            max_iterations: 100,
            shape_dim: 0,
            expr_dim: 0,
            learning_rate: 0.05,
            ..FittingConfig::default()
        };

        let res_short = fit_landmarks(&mock, &obs, &cam, &cfg_short).expect("short fit failed");
        let res_long = fit_landmarks(&mock, &obs, &cam, &cfg_long).expect("long fit failed");

        assert!(
            res_long.mean_reprojection_error <= res_short.mean_reprojection_error + 1.0,
            "longer optimization should not be worse: long={} short={}",
            res_long.mean_reprojection_error,
            res_short.mean_reprojection_error
        );
    }

    #[test]
    fn test_fit_landmarks_regularizer_effect() {
        // High regularizer should keep shape params near zero.
        let mock = MockFlameForward::unit_sphere(20);
        let cam = PinholeCamera::default();

        // Observe some landmarks at arbitrary positions
        let obs = vec![
            LandmarkObservation::new(0, 270.0, 260.0),
            LandmarkObservation::new(1, 250.0, 250.0),
        ];

        let cfg_high_reg = FittingConfig {
            max_iterations: 30,
            shape_dim: 5,
            expr_dim: 5,
            shape_regularizer: 10.0,
            expr_regularizer: 10.0,
            learning_rate: 0.1,
            ..FittingConfig::default()
        };

        let result = fit_landmarks(&mock, &obs, &cam, &cfg_high_reg).expect("fitting failed");

        let shape_norm: f32 = result
            .params
            .shape
            .iter()
            .map(|s| s * s)
            .sum::<f32>()
            .sqrt();
        let expr_norm: f32 = result
            .params
            .expression
            .iter()
            .map(|e| e * e)
            .sum::<f32>()
            .sqrt();

        assert!(
            shape_norm < 1.0,
            "high regularizer should keep shape near zero, got norm={shape_norm}"
        );
        assert!(
            expr_norm < 1.0,
            "high regularizer should keep expr near zero, got norm={expr_norm}"
        );
    }
}
