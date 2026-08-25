//! [`FlameLandmarkFitter`] — a [`FlameForward`] implementation that evaluates a
//! [`FlameModel`] at a fixed set of landmark vertices only.
//!
//! The optimizer in [`super::fit_landmarks`] never reads any vertex other than
//! the landmarks, so running the full 5023-vertex FLAME pipeline `2 · ndims`
//! times per iteration is almost entirely wasted work. This module restructures
//! the identical arithmetic so that the per-evaluation cost scales with the
//! landmark count instead.

use nalgebra as na;

use crate::model::{rodrigues, FlameModel};

use super::{
    gather_landmarks, FittingConfig, FittingError, FittingParams, FlameForward, LandmarkObservation,
};

/// A [`FlameForward`] adapter that evaluates a [`FlameModel`] at a fixed set of
/// landmark vertices only.
///
/// # Why this exists
///
/// [`super::fit_landmarks`] estimates the cost gradient by finite differences, which
/// costs `2 × ndims` forward passes per iteration. Running the full FLAME
/// pipeline for each of those — 5023 vertices of blend shapes, skinning, mesh
/// allocation and ~10 000 faces of normal recomputation — is wasted work when
/// only ~68 landmark vertices are ever read.
///
/// This adapter restructures the same computation so that the per-call cost is
/// proportional to the number of landmarks rather than the number of vertices:
///
/// - The template row, the shape/expression/pose blend-shape directions and the
///   skinning weights are sliced down to the landmark vertices once, at
///   construction time.
/// - Joint regression is reassociated. Because
///   `J · (v_template + Σₖ βₖ Sₖ) = J·v_template + Σₖ βₖ (J·Sₖ)`, the products
///   `J·Sₖ` are precomputed once and each forward pass reduces to a
///   `n_joints × 3` accumulation instead of an `n_joints × N × 3` matrix
///   product. Note the identity involves the **shape** directions only:
///   [`FlameModel::forward`] regresses the joint pivots from the shape-only
///   rest mesh and adds expression afterwards, so no expression term belongs
///   here.
/// - Mesh assembly and normal recomputation are skipped entirely.
///
/// The arithmetic is otherwise identical to the model's own forward pass: same
/// blend shapes, same pose correctives, same kinematic chain, same linear blend
/// skinning, same trailing translation.
///
/// # Example
///
/// ```rust,no_run
/// use oxigaf_flame::fitting::FlameLandmarkFitter;
/// use oxigaf_flame::{
///     fit_landmarks, FittingConfig, FlameModel, LandmarkObservation, PinholeCamera,
/// };
///
/// let model = FlameModel::load("path/to/flame")?;
/// let observations = vec![LandmarkObservation::new(0, 260.0, 250.0)];
/// let config = FittingConfig::default();
///
/// let fitter = FlameLandmarkFitter::for_observations(&model, &observations, &config)?;
/// let result = fit_landmarks(&fitter, &observations, &PinholeCamera::default(), &config)?;
/// println!("mean reprojection error: {}", result.mean_reprojection_error);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct FlameLandmarkFitter<'a> {
    /// The model being evaluated; used for the full-mesh fallback path.
    model: &'a FlameModel,
    /// Vertex indices this fitter is specialised for, in `out` order.
    indices: Vec<usize>,
    /// `v_template` rows at `indices`.
    template_subset: Vec<[f32; 3]>,
    /// `shapedirs[:, :, k]` rows at `indices`, per shape component.
    shape_subset: Vec<Vec<[f32; 3]>>,
    /// `expressiondirs[:, :, k]` rows at `indices`, per expression component.
    expr_subset: Vec<Vec<[f32; 3]>>,
    /// `posedirs[:, :, k]` rows at `indices`, per pose feature.
    pose_subset: Vec<Vec<[f32; 3]>>,
    /// `lbs_weights` rows at `indices` (`[landmark][joint]`).
    lbs_subset: Vec<Vec<f32>>,
    /// `j_regressor · v_template` (`[joint]`).
    joints_template: Vec<[f32; 3]>,
    /// `j_regressor · shapedirs[:, :, k]` (`[shape_component][joint]`).
    ///
    /// There is deliberately no expression counterpart: FLAME regresses the
    /// joint pivots from the shape-only rest mesh, so the expression
    /// coefficients never move the joints.
    joints_shape: Vec<Vec<[f32; 3]>>,
    /// Number of skeleton joints.
    n_joints: usize,
    /// Number of vertices in the underlying model.
    num_vertices: usize,
}

/// Blend-shape coefficients below this magnitude contribute nothing; the same
/// threshold `FlameModel`'s internal `apply_blend_shapes` uses.
const COEFF_EPS: f32 = 1e-12;

/// Extract `dirs[idx, :, k]` for every `idx` in `indices`.
fn slice_dirs(dirs: &ndarray::Array3<f32>, indices: &[usize], k: usize) -> Vec<[f32; 3]> {
    indices
        .iter()
        .map(|&v| [dirs[[v, 0, k]], dirs[[v, 1, k]], dirs[[v, 2, k]]])
        .collect()
}

/// Multiply the joint regressor by a per-vertex 3-column array supplied as a
/// `(vertex, component) -> value` accessor, yielding one `[x, y, z]` per joint.
fn regress_columns(
    j_regressor: &ndarray::Array2<f32>,
    n_joints: usize,
    num_vertices: usize,
    column: impl Fn(usize, usize) -> f32,
) -> Vec<[f32; 3]> {
    (0..n_joints)
        .map(|j| {
            let mut acc = [0.0f32; 3];
            for v in 0..num_vertices {
                let w = j_regressor[[j, v]];
                if w != 0.0 {
                    acc[0] += w * column(v, 0);
                    acc[1] += w * column(v, 1);
                    acc[2] += w * column(v, 2);
                }
            }
            acc
        })
        .collect()
}

/// Validate that `model`'s joint chain is well-formed for `n_joints`: a
/// nonzero count, a `parents` array long enough to cover it, and a
/// topological order (`parent < child`) throughout.
fn validate_joint_chain(model: &FlameModel, n_joints: usize) -> Result<(), FittingError> {
    if n_joints == 0 {
        return Err(FittingError::ForwardPassFailed(
            "model has zero joints".to_owned(),
        ));
    }
    if model.parents.len() < n_joints {
        return Err(FittingError::ForwardPassFailed(format!(
            "parents has {} entries but the model has {n_joints} joints",
            model.parents.len()
        )));
    }
    for (j, &parent) in model.parents.iter().enumerate().take(n_joints) {
        if parent >= 0 {
            let p = parent as usize;
            if p >= j {
                return Err(FittingError::ForwardPassFailed(format!(
                    "joint {j} has parent {p}, which is not earlier in the chain"
                )));
            }
        }
    }
    Ok(())
}

/// Validate that the joint regressor, skinning weights and vertex template
/// are all shaped for `n_joints` joints and `num_vertices` vertices.
fn validate_regressor_shapes(
    model: &FlameModel,
    n_joints: usize,
    num_vertices: usize,
) -> Result<(), FittingError> {
    if model.j_regressor.nrows() < n_joints || model.j_regressor.ncols() < num_vertices {
        return Err(FittingError::ForwardPassFailed(format!(
            "j_regressor is {}×{} but {n_joints}×{num_vertices} is required",
            model.j_regressor.nrows(),
            model.j_regressor.ncols()
        )));
    }
    if model.lbs_weights.nrows() < num_vertices || model.lbs_weights.ncols() < n_joints {
        return Err(FittingError::ForwardPassFailed(format!(
            "lbs_weights is {}×{} but {num_vertices}×{n_joints} is required",
            model.lbs_weights.nrows(),
            model.lbs_weights.ncols()
        )));
    }
    if model.v_template.ncols() < 3 {
        return Err(FittingError::ForwardPassFailed(format!(
            "v_template has {} columns, expected 3",
            model.v_template.ncols()
        )));
    }
    Ok(())
}

/// Validate that every entry of `landmark_indices` is a valid vertex index.
fn validate_landmark_indices(
    landmark_indices: &[usize],
    num_vertices: usize,
) -> Result<(), FittingError> {
    for &idx in landmark_indices {
        if idx >= num_vertices {
            return Err(FittingError::InvalidVertexIndex { idx, num_vertices });
        }
    }
    Ok(())
}

/// The `(shapedirs, expressiondirs, posedirs)` array shapes, each as
/// returned by `ndarray::Array3::shape`.
type BlendShapeDims<'a> = (&'a [usize], &'a [usize], &'a [usize]);

/// Validate that the blend-shape direction arrays (`shapedirs`,
/// `expressiondirs`, `posedirs`) are all at least `num_vertices × 3 × K`,
/// returning their shapes for the caller to read `K` from.
fn validate_blend_shape_dims(
    model: &FlameModel,
    num_vertices: usize,
) -> Result<BlendShapeDims<'_>, FittingError> {
    let shape_dims = model.shapedirs.shape();
    let expr_dims = model.expressiondirs.shape();
    let pose_dims = model.posedirs.shape();
    for (name, dims) in [
        ("shapedirs", shape_dims),
        ("expressiondirs", expr_dims),
        ("posedirs", pose_dims),
    ] {
        if dims[0] < num_vertices || dims[1] < 3 {
            return Err(FittingError::ForwardPassFailed(format!(
                "{name} is {}×{}×{} but {num_vertices}×3×K is required",
                dims[0], dims[1], dims[2]
            )));
        }
    }
    Ok((shape_dims, expr_dims, pose_dims))
}

impl<'a> FlameLandmarkFitter<'a> {
    /// Build a fitter specialised for `landmark_indices`, caching the first
    /// `n_shape` shape and `n_expr` expression components.
    ///
    /// Both counts are clamped to what the model actually provides. Parameter
    /// vectors longer than the cached counts fall back to the full-mesh path,
    /// so the result is always correct, only slower.
    ///
    /// # Errors
    ///
    /// - [`FittingError::InvalidVertexIndex`] if a landmark index is out of range.
    /// - [`FittingError::ForwardPassFailed`] if the model's arrays are
    ///   structurally inconsistent (mismatched joint counts, regressor width,
    ///   skinning-weight shape, or a non-topological parent chain).
    pub fn new(
        model: &'a FlameModel,
        landmark_indices: &[usize],
        n_shape: usize,
        n_expr: usize,
    ) -> Result<Self, FittingError> {
        let num_vertices = model.num_vertices();
        let n_joints = model.n_joints;

        validate_joint_chain(model, n_joints)?;
        validate_regressor_shapes(model, n_joints, num_vertices)?;
        validate_landmark_indices(landmark_indices, num_vertices)?;
        let (shape_dims, expr_dims, pose_dims) = validate_blend_shape_dims(model, num_vertices)?;

        let n_shape = n_shape.min(shape_dims[2]);
        let n_expr = n_expr.min(expr_dims[2]);
        let n_pose = pose_dims[2];

        // --- Slice the per-vertex arrays down to the landmark subset --------
        let template_subset: Vec<[f32; 3]> = landmark_indices
            .iter()
            .map(|&v| {
                [
                    model.v_template[[v, 0]],
                    model.v_template[[v, 1]],
                    model.v_template[[v, 2]],
                ]
            })
            .collect();

        let shape_subset: Vec<Vec<[f32; 3]>> = (0..n_shape)
            .map(|k| slice_dirs(&model.shapedirs, landmark_indices, k))
            .collect();
        let expr_subset: Vec<Vec<[f32; 3]>> = (0..n_expr)
            .map(|k| slice_dirs(&model.expressiondirs, landmark_indices, k))
            .collect();
        let pose_subset: Vec<Vec<[f32; 3]>> = (0..n_pose)
            .map(|k| slice_dirs(&model.posedirs, landmark_indices, k))
            .collect();

        let lbs_subset: Vec<Vec<f32>> = landmark_indices
            .iter()
            .map(|&v| (0..n_joints).map(|j| model.lbs_weights[[v, j]]).collect())
            .collect();

        // --- Precompute the joint-regression products -----------------------
        // joints(β) = J·v_template + Σₖ βₖ (J·Sₖ)
        //
        // Expression is absent on purpose. `FlameModel::forward` regresses the
        // joints from the SHAPE-ONLY rest mesh (`apply_shape_only`) and only
        // then adds expression on top for the posing/skinning stage, so the
        // expression coefficients must not move the joint pivots. Adding a
        // `Σₖ ψₖ (J·Eₖ)` term here silently shifts every skinning transform
        // and makes this fast path disagree with the full forward pass.
        let joints_template =
            regress_columns(&model.j_regressor, n_joints, num_vertices, |v, c| {
                model.v_template[[v, c]]
            });
        let joints_shape: Vec<Vec<[f32; 3]>> = (0..n_shape)
            .map(|k| {
                regress_columns(&model.j_regressor, n_joints, num_vertices, |v, c| {
                    model.shapedirs[[v, c, k]]
                })
            })
            .collect();

        Ok(Self {
            model,
            indices: landmark_indices.to_vec(),
            template_subset,
            shape_subset,
            expr_subset,
            pose_subset,
            lbs_subset,
            joints_template,
            joints_shape,
            n_joints,
            num_vertices,
        })
    }

    /// Build a fitter for the vertices referenced by `observations`, sized for
    /// the shape and expression dimensions `config` optimizes.
    ///
    /// # Errors
    ///
    /// Same as [`FlameLandmarkFitter::new`].
    pub fn for_observations(
        model: &'a FlameModel,
        observations: &[LandmarkObservation],
        config: &FittingConfig,
    ) -> Result<Self, FittingError> {
        let indices: Vec<usize> = observations.iter().map(|o| o.vertex_index).collect();
        Self::new(model, &indices, config.shape_dim, config.expr_dim)
    }

    /// The landmark vertex indices this fitter is specialised for.
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// `true` when `params` stays within the cached shape/expression
    /// components, so the fast path applies.
    fn can_specialise(&self, params: &FittingParams) -> bool {
        params.shape.len() <= self.shape_subset.len()
            && params.expression.len() <= self.expr_subset.len()
    }

    /// The specialised forward pass: landmark positions only.
    fn evaluate_subset(
        &self,
        params: &FittingParams,
        out: &mut [[f32; 3]],
    ) -> Result<(), FittingError> {
        if out.len() != self.indices.len() {
            return Err(FittingError::ParameterCountMismatch {
                expected: self.indices.len(),
                got: out.len(),
            });
        }

        // 1. Shape + expression blend shapes at the landmark vertices.
        let mut verts = self.template_subset.clone();
        accumulate_dirs(&mut verts, &self.shape_subset, &params.shape);
        accumulate_dirs(&mut verts, &self.expr_subset, &params.expression);

        // 2. Joint positions, via the precomputed regressor products.
        //    Shape only — `FlameModel::forward` regresses the pivots from the
        //    shape-only rest mesh, before expression is added.
        let mut joints = self.joints_template.clone();
        accumulate_dirs(&mut joints, &self.joints_shape, &params.shape);

        // 3. Per-joint rotation matrices (Rodrigues).  `to_flame_params` lays
        //    the pose out as [root(3), neck(3), jaw(3), l_eye(3), r_eye(3)],
        //    so joint 0 takes the global rotation, joint 2 the jaw, and the
        //    rest are identity — read directly to avoid rebuilding a
        //    `FlameParams` (and its three Vec allocations) per probe.
        let rot_mats: Vec<na::Matrix3<f32>> = (0..self.n_joints)
            .map(|j| {
                let [rx, ry, rz] = match j {
                    0 => params.global_rotation,
                    2 => params.jaw,
                    _ => [0.0; 3],
                };
                rodrigues(rx, ry, rz)
            })
            .collect();

        // 4. Pose corrective blend shapes: flatten (R_j − I) column-major for
        //    every non-root joint, exactly as `FlameModel::forward` does.
        let identity = na::Matrix3::<f32>::identity();
        let mut pose_feature = Vec::with_capacity((self.n_joints - 1) * 9);
        for rot in rot_mats.iter().skip(1) {
            let diff = rot - identity;
            for c in 0..3 {
                for r in 0..3 {
                    pose_feature.push(diff[(r, c)]);
                }
            }
        }
        accumulate_dirs(&mut verts, &self.pose_subset, &pose_feature);

        // 5. Kinematic-chain skinning transforms.
        let skinning = self.compute_skinning(&rot_mats, &joints);

        // 6. Linear blend skinning at the landmark vertices, plus translation.
        let [tx, ty, tz] = params.translation;
        for ((slot, v_posed), weights) in
            out.iter_mut().zip(verts.iter()).zip(self.lbs_subset.iter())
        {
            let mut blended = na::Matrix4::<f32>::zeros();
            for (transform, &w) in skinning.iter().zip(weights.iter()) {
                if w.abs() > COEFF_EPS {
                    blended += w * transform;
                }
            }
            let v = na::Vector4::new(v_posed[0], v_posed[1], v_posed[2], 1.0);
            let r = blended * v;
            *slot = [r[0] + tx, r[1] + ty, r[2] + tz];
        }

        Ok(())
    }

    /// Build the per-joint skinning transforms, mirroring
    /// `FlameModel::compute_skinning_transforms`.
    fn compute_skinning(
        &self,
        rot_mats: &[na::Matrix3<f32>],
        joints: &[[f32; 3]],
    ) -> Vec<na::Matrix4<f32>> {
        let nj = self.n_joints;
        let mut global = vec![na::Matrix4::<f32>::identity(); nj];

        for j in 0..nj {
            let j_pos = joints[j];
            let parent = self.model.parents[j];

            let mut local = na::Matrix4::<f32>::identity();
            for r in 0..3 {
                for c in 0..3 {
                    local[(r, c)] = rot_mats[j][(r, c)];
                }
            }

            if parent < 0 {
                // Root joint: absolute position.
                local[(0, 3)] = j_pos[0];
                local[(1, 3)] = j_pos[1];
                local[(2, 3)] = j_pos[2];
                global[j] = local;
            } else {
                // Child joint: relative to parent. `new` guarantees p < j.
                let p = parent as usize;
                let p_pos = joints[p];
                local[(0, 3)] = j_pos[0] - p_pos[0];
                local[(1, 3)] = j_pos[1] - p_pos[1];
                local[(2, 3)] = j_pos[2] - p_pos[2];
                global[j] = global[p] * local;
            }
        }

        // Remove rest-pose joint translations: A_j = G_j − pad(G_j · [J_j, 0]ᵀ).
        for (transform, joint) in global.iter_mut().zip(joints.iter()) {
            let j_homo = na::Vector4::new(joint[0], joint[1], joint[2], 0.0);
            let correction = *transform * j_homo;
            transform[(0, 3)] -= correction[0];
            transform[(1, 3)] -= correction[1];
            transform[(2, 3)] -= correction[2];
        }

        global
    }
}

/// Accumulate `Σₖ coeffs[k] · dirs[k][i]` into `acc[i]`.
///
/// Components beyond `dirs.len()` are ignored, matching `FlameModel`'s
/// truncation behaviour, and coefficients below [`COEFF_EPS`] are skipped.
fn accumulate_dirs(acc: &mut [[f32; 3]], dirs: &[Vec<[f32; 3]>], coeffs: &[f32]) {
    let k_max = coeffs.len().min(dirs.len());
    for (dir, &coeff) in dirs.iter().zip(coeffs.iter()).take(k_max) {
        if coeff.abs() > COEFF_EPS {
            for (slot, d) in acc.iter_mut().zip(dir.iter()) {
                slot[0] += coeff * d[0];
                slot[1] += coeff * d[1];
                slot[2] += coeff * d[2];
            }
        }
    }
}

impl FlameForward for FlameLandmarkFitter<'_> {
    fn forward(&self, params: &FittingParams) -> Result<Vec<[f32; 3]>, FittingError> {
        FlameForward::forward(self.model, params)
    }

    fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    fn forward_landmarks(
        &self,
        params: &FittingParams,
        indices: &[usize],
        out: &mut [[f32; 3]],
    ) -> Result<(), FittingError> {
        if indices == self.indices.as_slice() && self.can_specialise(params) {
            return self.evaluate_subset(params, out);
        }
        // Different landmark set or more coefficients than were cached: fall
        // back to the exact full-mesh pass.
        let vertices = FlameForward::forward(self.model, params)?;
        gather_landmarks(&vertices, indices, out)
    }
}
