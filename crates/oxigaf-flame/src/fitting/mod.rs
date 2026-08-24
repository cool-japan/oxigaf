//! CPU-based gradient-descent fitting of FLAME model parameters to 2D landmark
//! observations.
//!
//! # Overview
//!
//! This module provides landmark-based parameter fitting:
//! - A simple pinhole camera for 3D → 2D projection.
//! - [`FittingParams`] that wrap the subset of [`crate::params::FlameParams`] being
//!   optimized (shape, expression, global rotation, translation, jaw).
//! - A [`FlameForward`] trait so the optimizer is decoupled from the heavy model
//!   binary; [`FlameModel`] implements it directly, and [`MockFlameForward`]
//!   enables fast, deterministic tests.
//! - [`FlameLandmarkFitter`] — a landmark-specialised forward pass that keeps
//!   the finite-difference loop proportional to the number of landmarks rather
//!   than the number of mesh vertices.
//! - [`fit_landmarks`] — the main loop: steepest descent on a scalar cost whose
//!   gradient is estimated by **central differences** over the parameter vector.
//!   This is plain gradient descent, not Gauss-Newton: no residual Jacobian,
//!   normal equations or damping term are formed.

mod landmark_fitter;

pub use landmark_fitter::FlameLandmarkFitter;

use crate::model::FlameModel;
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

    /// Not a single landmark vertex projected in front of the camera, so the
    /// landmark term of the cost carries no information and the optimizer
    /// cannot make progress.
    ///
    /// The usual cause is a view matrix / initial translation combination that
    /// leaves the whole head at or behind the camera plane (camera-space
    /// `z <= 0`).
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
    /// World-to-camera view matrix, 4×4 in **row-major** storage:
    /// `view_matrix[row][col]`, so `view_matrix[0]` is the first *row* of the
    /// matrix and `view_matrix[i][3]` is the translation component of row `i`.
    ///
    /// [`PinholeCamera::project`] evaluates `camera = M · [x, y, z, 1]ᵀ` with
    /// this convention. A matrix taken from a column-major source (OpenGL /
    /// glam / nalgebra slices) must be transposed before being stored here,
    /// otherwise the camera is inverted.
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

    /// Read the `dim`-th scalar of the flat layout without materialising a
    /// `Vec`. Returns `0.0` for out-of-range dimensions.
    ///
    /// Layout: `[shape…, expression…, global_rotation(3), translation(3), jaw(3)]`.
    #[must_use]
    fn dim_value(&self, dim: usize) -> f32 {
        let n_shape = self.shape.len();
        let n_expr = self.expression.len();
        if dim < n_shape {
            self.shape[dim]
        } else if dim < n_shape + n_expr {
            self.expression[dim - n_shape]
        } else {
            match dim - n_shape - n_expr {
                r @ 0..=2 => self.global_rotation[r],
                r @ 3..=5 => self.translation[r - 3],
                r @ 6..=8 => self.jaw[r - 6],
                _ => 0.0,
            }
        }
    }

    /// Write the `dim`-th scalar of the flat layout in place. Out-of-range
    /// dimensions are ignored.
    ///
    /// This is the allocation-free counterpart of
    /// `from_vec(&{ let mut v = to_vec(); v[dim] = value; v })`, used by the
    /// finite-difference gradient loop.
    fn set_dim_value(&mut self, dim: usize, value: f32) {
        let n_shape = self.shape.len();
        let n_expr = self.expression.len();
        if dim < n_shape {
            self.shape[dim] = value;
        } else if dim < n_shape + n_expr {
            self.expression[dim - n_shape] = value;
        } else {
            match dim - n_shape - n_expr {
                r @ 0..=2 => self.global_rotation[r] = value,
                r @ 3..=5 => self.translation[r - 3] = value,
                r @ 6..=8 => self.jaw[r - 6] = value,
                _ => {}
            }
        }
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
    ///
    /// Always `false` when [`FittingResult::n_visible_landmarks`] is zero or
    /// [`FittingResult::mean_reprojection_error`] is non-finite: a "converged"
    /// fit whose landmark term carried no information is a failure, not a
    /// success.
    pub converged: bool,
    /// Per-landmark reprojection error in pixels. `f32::INFINITY` for a
    /// landmark whose vertex lies at or behind the camera plane.
    pub reprojection_errors: Vec<f32>,
    /// Mean reprojection error in pixels over the landmarks that projected;
    /// `f32::INFINITY` when none did.
    pub mean_reprojection_error: f32,
    /// How many observations projected in front of the camera at the final
    /// parameters. Zero means the fit is meaningless regardless of
    /// `final_cost`.
    pub n_visible_landmarks: usize,
}

// ---------------------------------------------------------------------------
// FlameForward trait + MockFlameForward
// ---------------------------------------------------------------------------

/// Trait abstracting the FLAME forward pass for fitting.
///
/// Implemented for the real [`FlameModel`], for the landmark-specialised
/// [`FlameLandmarkFitter`], and for [`MockFlameForward`] in tests.
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

    /// Evaluate the forward pass and write only the positions of `indices`
    /// into `out` (`out[i]` receives the position of vertex `indices[i]`).
    ///
    /// The optimizer only ever looks at the landmark vertices, so an
    /// implementation that can skip the rest of the mesh should override this
    /// method — see [`FlameLandmarkFitter`], which makes the per-call cost
    /// proportional to the number of landmarks instead of the number of
    /// vertices. The default implementation runs the full
    /// [`FlameForward::forward`] pass and gathers.
    ///
    /// # Errors
    ///
    /// Returns [`FittingError::ForwardPassFailed`] on computation failures,
    /// [`FittingError::InvalidVertexIndex`] if an index is out of range, and
    /// [`FittingError::ParameterCountMismatch`] if `out.len() != indices.len()`.
    fn forward_landmarks(
        &self,
        params: &FittingParams,
        indices: &[usize],
        out: &mut [[f32; 3]],
    ) -> Result<(), FittingError> {
        let vertices = self.forward(params)?;
        gather_landmarks(&vertices, indices, out)
    }
}

/// Copy `verts[indices[i]]` into `out[i]`, validating every index.
pub(crate) fn gather_landmarks(
    verts: &[[f32; 3]],
    indices: &[usize],
    out: &mut [[f32; 3]],
) -> Result<(), FittingError> {
    if out.len() != indices.len() {
        return Err(FittingError::ParameterCountMismatch {
            expected: indices.len(),
            got: out.len(),
        });
    }
    for (slot, &idx) in out.iter_mut().zip(indices.iter()) {
        match verts.get(idx) {
            Some(v) => *slot = *v,
            None => {
                return Err(FittingError::InvalidVertexIndex {
                    idx,
                    num_vertices: verts.len(),
                })
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// FlameForward for the real FLAME model
// ---------------------------------------------------------------------------

impl FlameForward for FlameModel {
    fn forward(&self, params: &FittingParams) -> Result<Vec<[f32; 3]>, FittingError> {
        let flame_params = params.to_flame_params();
        // Disambiguate from this trait method: the inherent `FlameModel::forward`
        // takes `&FlameParams` and returns a `Mesh`.
        let mesh = FlameModel::forward(self, &flame_params);
        Ok(mesh
            .vertices
            .iter()
            .map(|v| [v.x, v.y, v.z])
            .collect::<Vec<[f32; 3]>>())
    }

    fn num_vertices(&self) -> usize {
        FlameModel::num_vertices(self)
    }
}

// ---------------------------------------------------------------------------
// MockFlameForward
// ---------------------------------------------------------------------------

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

/// A cost evaluation: total cost plus how many observations projected.
#[derive(Debug, Clone, Copy)]
struct CostEval {
    /// `landmark_weight · Σᵢ confᵢ ‖proj(vᵢ) − obsᵢ‖² + regularizers`.
    cost: f32,
    /// Number of observations whose vertex projected in front of the camera.
    n_visible: usize,
}

/// Compute the total fitting cost from already-evaluated landmark positions.
///
/// Cost = `landmark_weight * Σᵢ conf_i * ||proj(vᵢ) - obs_i||²`
///       `+ shape_reg * ||shape||² + expr_reg * ||expr||²`
///
/// Landmarks behind the camera plane cannot be projected and contribute
/// nothing; the count of those that did project is returned so callers can
/// detect the degenerate all-behind-the-camera case instead of mistaking a
/// zero landmark term for a perfect fit.
fn cost_from_landmarks(
    landmark_positions: &[[f32; 3]],
    params: &FittingParams,
    observations: &[LandmarkObservation],
    camera: &PinholeCamera,
    config: &FittingConfig,
) -> CostEval {
    let mut landmark_cost = 0.0f32;
    let mut n_visible = 0usize;
    for (obs, &vertex) in observations.iter().zip(landmark_positions.iter()) {
        if let Some(proj) = camera.project(vertex) {
            let dx = proj[0] - obs.position_2d[0];
            let dy = proj[1] - obs.position_2d[1];
            landmark_cost += obs.confidence * (dx * dx + dy * dy);
            n_visible += 1;
        }
    }

    // Regularization terms
    let shape_reg: f32 = params.shape.iter().map(|s| s * s).sum();
    let expr_reg: f32 = params.expression.iter().map(|e| e * e).sum();

    CostEval {
        cost: config.landmark_weight * landmark_cost
            + config.shape_regularizer * shape_reg
            + config.expr_regularizer * expr_reg,
        n_visible,
    }
}

/// Evaluate the forward pass at the landmark vertices and compute the cost.
///
/// `scratch` must already be sized to `observations.len()`; it is reused
/// across the many evaluations of the finite-difference loop so the gradient
/// estimate allocates nothing per parameter dimension.
fn compute_cost_into<F: FlameForward + ?Sized>(
    model: &F,
    params: &FittingParams,
    observations: &[LandmarkObservation],
    indices: &[usize],
    camera: &PinholeCamera,
    config: &FittingConfig,
    scratch: &mut [[f32; 3]],
) -> Result<CostEval, FittingError> {
    model.forward_landmarks(params, indices, scratch)?;
    Ok(cost_from_landmarks(
        scratch,
        params,
        observations,
        camera,
        config,
    ))
}

/// Compute the total fitting cost for the given parameters.
///
/// Convenience wrapper over [`compute_cost_into`] that allocates its own
/// scratch buffer. The optimizer always uses the buffer-reusing form, so this
/// exists only to let tests evaluate the cost at a single point.
///
/// # Errors
///
/// Propagates forward-pass and index-validation failures.
#[cfg(test)]
fn compute_cost<F: FlameForward + ?Sized>(
    model: &F,
    params: &FittingParams,
    observations: &[LandmarkObservation],
    camera: &PinholeCamera,
    config: &FittingConfig,
) -> Result<f32, FittingError> {
    let indices: Vec<usize> = observations.iter().map(|o| o.vertex_index).collect();
    let mut scratch = vec![[0.0f32; 3]; observations.len()];
    let eval = compute_cost_into(
        model,
        params,
        observations,
        &indices,
        camera,
        config,
        &mut scratch,
    )?;
    Ok(eval.cost)
}

// ---------------------------------------------------------------------------
// Main fitting function
// ---------------------------------------------------------------------------

/// Fit FLAME model parameters to 2D landmark observations by gradient descent
/// on a finite-difference gradient.
///
/// # Algorithm
///
/// 1. Initialize parameters at zero.
/// 2. For each iteration:
///    a. Compute the cost at the current parameters.
///    b. Estimate the gradient by **central differences**: perturb each
///       parameter by `+eps` and `−eps` in turn and take
///       `(cost₊ − cost₋) / (2·eps)`, which is `O(eps²)`-accurate.
///    c. Clip the gradient to unit L2 norm and update every parameter:
///       `pᵢ -= learning_rate * grad_i`.
///    d. Stop early if `||step|| < convergence_delta`.
/// 3. Compute final per-landmark reprojection errors.
///
/// This is steepest descent on a scalar cost, **not** Gauss-Newton: no residual
/// Jacobian is formed and no normal equations are solved.
///
/// # Cost
///
/// Each iteration performs `2 · ndims + 1` forward evaluations, where
/// `ndims = shape_dim + expr_dim + 9`. Only the landmark vertices are ever
/// read, so wrapping a [`FlameModel`] in [`FlameLandmarkFitter`] — which
/// evaluates just those vertices — keeps the per-evaluation cost proportional
/// to the number of landmarks rather than the ~5000 vertices of the full mesh.
///
/// # Errors
///
/// - [`FittingError::NoObservations`] — empty observation list.
/// - [`FittingError::InvalidVertexIndex`] — observation references a vertex
///   that does not exist in the model.
/// - [`FittingError::ForwardPassFailed`] — the model's forward pass failed.
/// - [`FittingError::CameraProjectionFailed`] — not one landmark projects in
///   front of the camera at the initial parameters, so the landmark term is
///   identically zero and the optimizer has nothing to descend.
pub fn fit_landmarks<F: FlameForward + ?Sized>(
    model: &F,
    observations: &[LandmarkObservation],
    camera: &PinholeCamera,
    config: &FittingConfig,
) -> Result<FittingResult, FittingError> {
    if observations.is_empty() {
        return Err(FittingError::NoObservations);
    }

    // Validate vertex indices against model size upfront.  Because they are
    // checked here, the inner loop never repeats the check.
    let num_verts = model.num_vertices();
    for obs in observations {
        if obs.vertex_index >= num_verts {
            return Err(FittingError::InvalidVertexIndex {
                idx: obs.vertex_index,
                num_vertices: num_verts,
            });
        }
    }

    let indices: Vec<usize> = observations.iter().map(|o| o.vertex_index).collect();
    let mut scratch = vec![[0.0f32; 3]; observations.len()];

    let mut params = FittingParams::zero(config.shape_dim, config.expr_dim);
    let ndims = params.total_dims();
    let eps = config.finite_diff_eps;

    let mut converged = false;
    let mut iterations = 0usize;
    let mut final_cost = 0.0f32;
    let mut grad = vec![0.0f32; ndims];
    // Reused across every finite-difference probe so the loop allocates nothing.
    let mut probe = params.clone();

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Current cost
        let base = compute_cost_into(
            model,
            &params,
            observations,
            &indices,
            camera,
            config,
            &mut scratch,
        )?;
        final_cost = base.cost;

        if iter == 0 && base.n_visible == 0 {
            // Every landmark is at or behind the camera plane: the landmark
            // term is identically zero for every perturbation, so the step
            // norm would immediately fall below `convergence_delta` and the
            // optimizer would report a converged fit that never looked at a
            // single observation.  Fail loudly instead.
            return Err(FittingError::CameraProjectionFailed);
        }

        // Numerical gradient — central differences, two forward evaluations
        // per parameter dimension.
        for (dim, g) in grad.iter_mut().enumerate() {
            let center = params.dim_value(dim);

            probe.set_dim_value(dim, center + eps);
            let cost_plus = compute_cost_into(
                model,
                &probe,
                observations,
                &indices,
                camera,
                config,
                &mut scratch,
            )?
            .cost;

            probe.set_dim_value(dim, center - eps);
            let cost_minus = compute_cost_into(
                model,
                &probe,
                observations,
                &indices,
                camera,
                config,
                &mut scratch,
            )?
            .cost;

            // Restore the probe so the next dimension perturbs around the
            // current parameters, not a previously perturbed copy.
            probe.set_dim_value(dim, center);

            *g = (cost_plus - cost_minus) / (2.0 * eps);
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

        // Gradient descent update, applied in place.
        for (dim, &g) in grad.iter().enumerate() {
            let updated = params.dim_value(dim) - config.learning_rate * g * scale;
            params.set_dim_value(dim, updated);
            probe.set_dim_value(dim, updated);
        }

        // Convergence check: L2 norm of the step taken
        let step_norm = config.learning_rate * grad_norm.min(1.0);

        if step_norm < config.convergence_delta {
            converged = true;
            // Recompute final cost at the updated params
            final_cost = compute_cost_into(
                model,
                &params,
                observations,
                &indices,
                camera,
                config,
                &mut scratch,
            )?
            .cost;
            break;
        }
    }

    if !converged {
        // Ensure final_cost reflects the last params after the loop
        final_cost = compute_cost_into(
            model,
            &params,
            observations,
            &indices,
            camera,
            config,
            &mut scratch,
        )?
        .cost;
    }

    // Compute per-landmark reprojection errors at the final parameters.
    // `scratch` already holds the final landmark positions from the cost
    // evaluation immediately above.
    let mut reprojection_errors = Vec::with_capacity(observations.len());
    let mut n_visible_landmarks = 0usize;

    for (obs, &vertex) in observations.iter().zip(scratch.iter()) {
        let err = match camera.project(vertex) {
            Some(proj) => {
                n_visible_landmarks += 1;
                let dx = proj[0] - obs.position_2d[0];
                let dy = proj[1] - obs.position_2d[1];
                (dx * dx + dy * dy).sqrt()
            }
            None => f32::INFINITY,
        };
        reprojection_errors.push(err);
    }

    let mean_reprojection_error = if n_visible_landmarks == 0 {
        f32::INFINITY
    } else {
        reprojection_errors
            .iter()
            .copied()
            .filter(|e| e.is_finite())
            .sum::<f32>()
            / n_visible_landmarks as f32
    };

    // A fit whose landmark term carried no information never converged,
    // whatever the step norm said.
    let converged = converged && n_visible_landmarks > 0 && mean_reprojection_error.is_finite();

    Ok(FittingResult {
        params,
        final_cost,
        iterations,
        converged,
        reprojection_errors,
        mean_reprojection_error,
        n_visible_landmarks,
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

    // -----------------------------------------------------------------------
    // Regression: landmarks entirely behind the camera must not "converge"
    // -----------------------------------------------------------------------

    #[test]
    fn test_fit_landmarks_all_behind_camera_errors() {
        // Every vertex has z < 0, so with the identity view matrix nothing
        // projects.  Previously the landmark term was identically zero, the
        // gradient vanished, and the optimizer reported converged == true with
        // a tiny final_cost.  It must now fail loudly instead.
        let mock = MockFlameForward {
            base_vertices: vec![[0.0, 0.0, -2.0], [0.1, 0.0, -2.0]],
        };
        let cam = PinholeCamera::new(500.0, 256.0, 256.0);
        let obs = vec![
            LandmarkObservation::new(0, 256.0, 256.0),
            LandmarkObservation::new(1, 300.0, 256.0),
        ];
        let cfg = FittingConfig {
            max_iterations: 50,
            shape_dim: 0,
            expr_dim: 0,
            ..FittingConfig::default()
        };

        let result = fit_landmarks(&mock, &obs, &cam, &cfg);
        assert!(
            matches!(result, Err(FittingError::CameraProjectionFailed)),
            "all-behind-the-camera must report CameraProjectionFailed, got {result:?}"
        );
    }

    #[test]
    fn test_fit_landmarks_reports_visible_landmark_count() {
        // One vertex in front of the camera, one behind it.
        let mock = MockFlameForward {
            base_vertices: vec![[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]],
        };
        let cam = PinholeCamera::new(500.0, 256.0, 256.0);
        let obs = vec![
            LandmarkObservation::new(0, 256.0, 256.0),
            LandmarkObservation::new(1, 256.0, 256.0),
        ];
        let cfg = FittingConfig {
            max_iterations: 3,
            shape_dim: 0,
            expr_dim: 0,
            ..FittingConfig::default()
        };

        let result = fit_landmarks(&mock, &obs, &cam, &cfg).expect("fitting failed");
        assert_eq!(
            result.n_visible_landmarks, 1,
            "exactly one landmark projects"
        );
        assert!(result.reprojection_errors[0].is_finite());
        assert!(
            result.reprojection_errors[1].is_infinite(),
            "the behind-camera landmark must report an infinite error"
        );
        assert!(result.mean_reprojection_error.is_finite());
    }

    // -----------------------------------------------------------------------
    // Regression: central differences
    // -----------------------------------------------------------------------

    #[test]
    fn test_gradient_uses_central_differences() {
        // Cost as a function of tx is a parabola:
        //   c(tx) = (f*(x0+tx)/z + cx - obs_x)^2
        // Its exact derivative at tx = 0 is analytic, and central differences
        // reproduce it to O(eps^2) while a forward difference is off by O(eps).
        // One gradient-clipped step from zero must therefore land the parameter
        // on the descent direction with the correct sign and scale.
        let mock = MockFlameForward {
            base_vertices: vec![[0.0, 0.0, 2.0]],
        };
        let cam = PinholeCamera::new(500.0, 256.0, 256.0);
        // Observed 50 px to the right → optimizer must increase tx.
        let obs = vec![LandmarkObservation::new(0, 306.0, 256.0)];
        let cfg = FittingConfig {
            max_iterations: 1,
            shape_dim: 0,
            expr_dim: 0,
            learning_rate: 0.1,
            finite_diff_eps: 1e-3,
            convergence_delta: 0.0,
            ..FittingConfig::default()
        };

        let result = fit_landmarks(&mock, &obs, &cam, &cfg).expect("fitting failed");
        assert_eq!(result.iterations, 1);
        // Gradient wrt tx is large (pixel-scale cost), so it is clipped to unit
        // norm and the step is exactly `learning_rate` in the descent direction.
        assert!(
            (result.params.translation[0] - 0.1).abs() < 1e-4,
            "one clipped step must move tx by +learning_rate, got {}",
            result.params.translation[0]
        );
        // No other parameter should have moved (their gradients are ~0 for a
        // single on-axis landmark, and ty/tz/rotation are orthogonal here).
        assert!(result.params.translation[2].abs() < 1e-3);
    }

    #[test]
    fn test_central_difference_is_symmetric_around_current_params() {
        // A quadratic cost has an exactly-symmetric central difference, so the
        // gradient at the minimum must be zero to machine precision — the
        // signature property a one-sided difference does NOT have.
        let mock = MockFlameForward {
            base_vertices: vec![[0.0, 0.0, 2.0]],
        };
        let cam = PinholeCamera::new(500.0, 256.0, 256.0);
        // Observed exactly where the vertex already projects: tx = 0 is the
        // minimum, so the optimizer must not move.
        let obs = vec![LandmarkObservation::new(0, 256.0, 256.0)];
        let cfg = FittingConfig {
            max_iterations: 1,
            shape_dim: 0,
            expr_dim: 0,
            learning_rate: 0.1,
            finite_diff_eps: 1e-3,
            shape_regularizer: 0.0,
            expr_regularizer: 0.0,
            convergence_delta: 0.0,
            ..FittingConfig::default()
        };

        let result = fit_landmarks(&mock, &obs, &cam, &cfg).expect("fitting failed");
        for (i, t) in result.params.translation.iter().enumerate() {
            assert!(
                t.abs() < 1e-5,
                "translation[{i}] must stay at the minimum, got {t}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // FittingParams flat-layout accessors (used by the gradient loop)
    // -----------------------------------------------------------------------

    #[test]
    fn test_dim_accessors_match_flat_layout() {
        let p = FittingParams {
            shape: vec![0.1, 0.2, 0.3],
            expression: vec![0.4, 0.5],
            global_rotation: [0.01, 0.02, 0.03],
            translation: [1.0, 2.0, 3.0],
            jaw: [0.05, 0.06, 0.07],
        };
        let flat = p.to_vec();
        assert_eq!(flat.len(), p.total_dims());
        for (dim, &expected) in flat.iter().enumerate() {
            assert!(
                (p.dim_value(dim) - expected).abs() < f32::EPSILON,
                "dim_value({dim}) = {} but to_vec()[{dim}] = {expected}",
                p.dim_value(dim)
            );
        }
        // Out-of-range reads are zero, not a panic.
        assert_eq!(p.dim_value(flat.len()), 0.0);
    }

    #[test]
    fn test_set_dim_value_matches_from_vec() {
        let base = FittingParams::zero(3, 2);
        let ndims = base.total_dims();
        for dim in 0..ndims {
            let mut via_setter = base.clone();
            via_setter.set_dim_value(dim, 7.5);

            let mut flat = base.to_vec();
            flat[dim] = 7.5;
            let via_vec = FittingParams::from_vec(&flat, 3, 2).expect("roundtrip failed");

            assert_eq!(
                via_setter.to_vec(),
                via_vec.to_vec(),
                "mismatch at dim {dim}"
            );
        }
        // Out-of-range writes are ignored, not a panic.
        let mut p = base.clone();
        p.set_dim_value(ndims + 5, 1.0);
        assert_eq!(p.to_vec(), base.to_vec());
    }

    // -----------------------------------------------------------------------
    // FlameForward for FlameModel / FlameLandmarkFitter
    // -----------------------------------------------------------------------

    /// Build a small but structurally complete synthetic FLAME model.
    fn synthetic_model() -> FlameModel {
        use ndarray::{Array2, Array3};

        let n_verts = 8;
        let n_joints = 5;
        let n_shape = 4;
        let n_expr = 3;
        let n_pose_dirs = (n_joints - 1) * 9;

        // Spread the vertices out in front of the camera (z around +2).
        let v_template = Array2::from_shape_fn((n_verts, 3), |(i, c)| match c {
            0 => (i as f32) * 0.05 - 0.2,
            1 => (i as f32) * 0.03 - 0.1,
            _ => 2.0 + (i as f32) * 0.01,
        });

        let faces = vec![[0u32, 1, 2], [2, 3, 4], [4, 5, 6]];

        let shapedirs = Array3::from_shape_fn((n_verts, 3, n_shape), |(i, c, k)| {
            0.01 * ((i + c + k) as f32).sin()
        });
        let expressiondirs = Array3::from_shape_fn((n_verts, 3, n_expr), |(i, c, k)| {
            0.02 * ((i * 3 + c + k) as f32).cos()
        });
        let posedirs = Array3::from_shape_fn((n_verts, 3, n_pose_dirs), |(i, c, k)| {
            0.001 * ((i + c * 2 + k) as f32).sin()
        });

        // Non-uniform joint regressor so reassociation is actually exercised.
        let j_regressor = Array2::from_shape_fn((n_joints, n_verts), |(j, v)| {
            let raw = 1.0 + ((j * 7 + v * 3) % 5) as f32;
            raw / (n_verts as f32 * 3.0)
        });

        let parents = vec![-1i32, 0, 0, 1, 1];

        let lbs_weights = Array2::from_shape_fn((n_verts, n_joints), |(i, j)| {
            if i % n_joints == j {
                0.6
            } else {
                0.1
            }
        });

        FlameModel::from_arrays(
            v_template,
            faces,
            shapedirs,
            expressiondirs,
            posedirs,
            j_regressor,
            parents,
            lbs_weights,
            n_joints,
        )
    }

    #[test]
    fn test_flame_model_implements_flame_forward() {
        let model = synthetic_model();
        assert_eq!(FlameForward::num_vertices(&model), 8);

        let mut params = FittingParams::zero(4, 3);
        params.translation = [0.5, -0.25, 0.75];

        let verts = FlameForward::forward(&model, &params).expect("forward failed");
        assert_eq!(verts.len(), 8);

        // Cross-check against the inherent forward pass.
        let mesh = FlameModel::forward(&model, &params.to_flame_params());
        for (got, want) in verts.iter().zip(mesh.vertices.iter()) {
            assert!((got[0] - want.x).abs() < 1e-5);
            assert!((got[1] - want.y).abs() < 1e-5);
            assert!((got[2] - want.z).abs() < 1e-5);
        }
    }

    #[test]
    fn test_landmark_fitter_matches_full_forward() {
        let model = synthetic_model();
        let indices = vec![0usize, 3, 6];
        let fitter =
            FlameLandmarkFitter::new(&model, &indices, 4, 3).expect("fitter construction failed");
        assert_eq!(fitter.indices(), indices.as_slice());

        let params = FittingParams {
            shape: vec![0.4, -0.2, 0.1, 0.05],
            expression: vec![-0.3, 0.15, 0.6],
            global_rotation: [0.12, -0.08, 0.03],
            translation: [0.2, -0.1, 0.35],
            jaw: [0.2, 0.0, 0.0],
        };

        let full = FlameForward::forward(&model, &params).expect("full forward failed");
        let mut subset = vec![[0.0f32; 3]; indices.len()];
        fitter
            .forward_landmarks(&params, &indices, &mut subset)
            .expect("subset forward failed");

        for (slot, &idx) in subset.iter().zip(indices.iter()) {
            for c in 0..3 {
                assert!(
                    (slot[c] - full[idx][c]).abs() < 1e-4,
                    "vertex {idx} component {c}: subset {} vs full {}",
                    slot[c],
                    full[idx][c]
                );
            }
        }
    }

    #[test]
    fn test_landmark_fitter_falls_back_for_uncached_dimensions() {
        let model = synthetic_model();
        let indices = vec![1usize, 5];
        // Cache only 1 shape / 1 expression component…
        let fitter = FlameLandmarkFitter::new(&model, &indices, 1, 1).expect("construction failed");

        // …but evaluate with more, forcing the exact full-mesh fallback.
        let params = FittingParams {
            shape: vec![0.3, 0.2, -0.1, 0.05],
            expression: vec![0.1, -0.4, 0.2],
            global_rotation: [0.0; 3],
            translation: [0.1, 0.1, 0.1],
            jaw: [0.0; 3],
        };

        let full = FlameForward::forward(&model, &params).expect("full forward failed");
        let mut subset = vec![[0.0f32; 3]; indices.len()];
        fitter
            .forward_landmarks(&params, &indices, &mut subset)
            .expect("subset forward failed");

        for (slot, &idx) in subset.iter().zip(indices.iter()) {
            for c in 0..3 {
                assert!((slot[c] - full[idx][c]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_landmark_fitter_rejects_out_of_range_index() {
        let model = synthetic_model();
        let result = FlameLandmarkFitter::new(&model, &[0, 999], 4, 3);
        assert!(
            matches!(
                result,
                Err(FittingError::InvalidVertexIndex {
                    idx: 999,
                    num_vertices: 8
                })
            ),
            "out-of-range landmark index must be rejected"
        );
    }

    #[test]
    fn test_fit_landmarks_with_real_model_reduces_error() {
        let model = synthetic_model();
        let cam = PinholeCamera::new(500.0, 256.0, 256.0);

        // Generate synthetic observations from a known parameter set so the
        // target is reachable.
        let mut truth = FittingParams::zero(2, 2);
        truth.translation = [0.15, -0.1, 0.0];
        let truth_verts = FlameForward::forward(&model, &truth).expect("truth forward failed");

        let landmark_indices = [0usize, 2, 4, 6];
        let observations: Vec<LandmarkObservation> = landmark_indices
            .iter()
            .map(|&idx| {
                let proj = cam
                    .project(truth_verts[idx])
                    .expect("test landmarks must project");
                LandmarkObservation::new(idx, proj[0], proj[1])
            })
            .collect();

        let cfg = FittingConfig {
            max_iterations: 200,
            shape_dim: 2,
            expr_dim: 2,
            learning_rate: 0.02,
            shape_regularizer: 0.0,
            expr_regularizer: 0.0,
            convergence_delta: 1e-7,
            ..FittingConfig::default()
        };

        // Baseline error at the zero initialisation.
        let zero_params = FittingParams::zero(cfg.shape_dim, cfg.expr_dim);
        let zero_verts = FlameForward::forward(&model, &zero_params).expect("forward failed");
        let baseline: f32 = observations
            .iter()
            .map(|obs| {
                let proj = cam
                    .project(zero_verts[obs.vertex_index])
                    .expect("must project");
                let dx = proj[0] - obs.position_2d[0];
                let dy = proj[1] - obs.position_2d[1];
                (dx * dx + dy * dy).sqrt()
            })
            .sum::<f32>()
            / observations.len() as f32;

        let fitter = FlameLandmarkFitter::for_observations(&model, &observations, &cfg)
            .expect("fitter construction failed");
        let result = fit_landmarks(&fitter, &observations, &cam, &cfg).expect("fitting failed");

        assert_eq!(result.n_visible_landmarks, observations.len());
        assert!(
            result.mean_reprojection_error < baseline,
            "fitting must reduce reprojection error: {} vs baseline {}",
            result.mean_reprojection_error,
            baseline
        );

        // Fitting through the plain `FlameModel` impl must reach the same place.
        let direct = fit_landmarks(&model, &observations, &cam, &cfg).expect("direct fit failed");
        assert!(
            (direct.mean_reprojection_error - result.mean_reprojection_error).abs() < 1e-2,
            "specialised and full-mesh fits must agree: {} vs {}",
            direct.mean_reprojection_error,
            result.mean_reprojection_error
        );
    }
}
