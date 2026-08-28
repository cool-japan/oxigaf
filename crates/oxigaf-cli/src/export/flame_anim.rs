//! FLAME-driven animation of a bound [`GaussianModel`].
//!
//! `oxigaf render --flame-params <seq.json>` asks for the avatar to be posed
//! per frame instead of rendered statically.  The Gaussians carry the binding
//! needed for that — [`GaussianModel::face_indices`] and
//! [`GaussianModel::barycentric`] name the FLAME triangle each Gaussian was
//! sampled on — but nothing consumed it, so the flag could only be reported as
//! unimplemented.  This module implements the deformation.
//!
//! # How a Gaussian follows its triangle
//!
//! Every Gaussian is attached to one mesh triangle through a *triangle frame*:
//!
//! * **origin** `T` — the Gaussian's own barycentric point on the triangle,
//! * **rotation** `R` — an orthonormal basis built from the triangle's first
//!   edge, its normal, and their cross product,
//! * **scale** `s` — `√area`, so `s` tracks the triangle's linear size.
//!
//! Posing computes that frame twice, once on the *rest* mesh the model was
//! trained against and once on the *posed* mesh, and applies the **difference**
//! to the values already stored on the Gaussian:
//!
//! ```text
//! Δ R = R_posed · R_restᵀ            k = s_posed / s_rest
//! p'  = T_posed + k · ΔR · (p − T_rest)
//! q'  = ΔR · q
//! log_scale' = log_scale + ln k
//! ```
//!
//! Working from the *difference* rather than re-deriving the position from the
//! barycentric coordinates matters: after training, a Gaussian's position has
//! drifted away from its exact barycentric point, and re-deriving would throw
//! that trained refinement away.  With the delta form, a Gaussian that never
//! moved lands exactly on its barycentric point on the posed mesh, and one that
//! did keeps its offset — rotated and scaled with the triangle.
//!
//! Scales are stored in **log space** (the initialiser writes `-5.0`), so the
//! scale *ratio* is an **additive** `ln k`, not a multiplication.
//!
//! # Rigid vs flexible Gaussians
//!
//! [`GaussianModel::is_rigid`] marks the Gaussians that should "move with the
//! head bone only" — they are not supposed to follow the expression and jaw
//! blendshapes.  [`animate_model`] honours that by evaluating FLAME twice per
//! frame: once with the full parameters (which drives the flexible Gaussians)
//! and once with [`rigid_pose_params`] applied (which drives the rigid ones).

use std::path::Path;

use anyhow::{Context, Result};
use nalgebra as na;
use serde::Deserialize;

use oxigaf::flame::{FlameModel, FlameParams, Mesh};
use oxigaf::render::gaussian::GaussianModel;

/// Below this, a triangle edge / area / quaternion norm is treated as
/// degenerate and the affected Gaussian is left untouched.
const EPS: f32 = 1e-12;

/// Barycentric coordinates this close to `1/3` each, combined with an all-zero
/// `face_indices`, are the placeholder [`crate::export::load_ply`] writes for a
/// PLY that carries no FLAME binding at all.
const PLACEHOLDER_BARY_TOLERANCE: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Parameter sequences
// ---------------------------------------------------------------------------

/// One frame of a FLAME parameter sequence, as written by the exporters this
/// CLI can read.
///
/// Every field defaults, because FLAME treats missing trailing coefficients as
/// zero; `expr` is accepted as an alias for `expression`.
#[derive(Debug, Deserialize)]
struct FrameSpec {
    #[serde(default)]
    shape: Vec<f32>,
    #[serde(default, alias = "expr")]
    expression: Vec<f32>,
    #[serde(default)]
    pose: Vec<f32>,
    #[serde(default)]
    translation: Option<[f32; 3]>,
}

/// The two document shapes `--flame-params` is written in.
///
/// Variant order is load-bearing for `#[serde(untagged)]`: every [`FrameSpec`]
/// field defaults, so a `{ "frames": [...] }` document would also match the
/// single-frame variant if that came first.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SequenceDoc {
    /// A bare array — the shape `oxigaf train --flame-params` reads.
    Bare(Vec<FrameSpec>),
    /// `{ "fps": …, "frames": [ … ] }` — the shape `FlameSequence::from_json`
    /// reads.
    Wrapped { frames: Vec<FrameSpec> },
    /// A single frame object, for a one-pose render.
    Single(FrameSpec),
}

impl FrameSpec {
    fn into_params(self, index: usize) -> Result<FlameParams> {
        let check = |name: &str, values: &[f32]| -> Result<()> {
            if let Some(pos) = values.iter().position(|v| !v.is_finite()) {
                anyhow::bail!("FLAME frame {index}: {name}[{pos}] is not a finite number");
            }
            Ok(())
        };
        check("shape", &self.shape)?;
        check("expression", &self.expression)?;
        check("pose", &self.pose)?;

        let translation = self.translation.unwrap_or([0.0; 3]);
        check("translation", &translation)?;

        let max_pose = FlameParams::NUM_JOINTS * 3;
        anyhow::ensure!(
            self.pose.len() <= max_pose,
            "FLAME frame {index}: pose has {} values, expected at most {max_pose} \
             (root, neck, jaw, left eye, right eye — 3 axis-angle values each)",
            self.pose.len()
        );
        let mut pose = self.pose;
        pose.resize(max_pose, 0.0);

        Ok(FlameParams {
            shape: self.shape,
            expression: self.expression,
            pose,
            translation,
        })
    }
}

/// Load a per-frame FLAME parameter sequence from JSON.
///
/// Three document shapes are accepted, so the same file works for `train
/// --flame-params` (which reads a bare array) and for exporters that wrap the
/// frames:
///
/// * `[{ "shape": [...], "expression": [...], "pose": [...] }, …]`
/// * `{ "fps": 30.0, "frames": [ … ] }`
/// * a single frame object, for a one-pose render.
///
/// # Errors
///
/// Returns an error when the file cannot be read, is not one of the three
/// shapes above, holds no frames, or carries a non-finite / over-long
/// parameter vector.
pub fn load_flame_param_sequence(path: &Path) -> Result<Vec<FlameParams>> {
    let json = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read FLAME params: {}", path.display()))?;

    let doc: SequenceDoc = serde_json::from_str(&json).with_context(|| {
        format!(
            "Failed to parse FLAME params: {} — expected a JSON array of frames, a \
             {{\"frames\": [...]}} object, or a single frame object",
            path.display()
        )
    })?;

    let specs = match doc {
        SequenceDoc::Bare(frames) | SequenceDoc::Wrapped { frames } => frames,
        SequenceDoc::Single(frame) => vec![frame],
    };
    anyhow::ensure!(
        !specs.is_empty(),
        "FLAME params file {} contains no frames",
        path.display()
    );

    let frames = specs
        .into_iter()
        .enumerate()
        .map(|(i, spec)| spec.into_params(i))
        .collect::<Result<Vec<_>>>()
        .with_context(|| format!("Invalid FLAME params in {}", path.display()))?;

    tracing::info!(
        "Loaded {} frame(s) of FLAME parameters from {}",
        frames.len(),
        path.display(),
    );
    Ok(frames)
}

/// Strip everything but head-bone motion from a frame's parameters.
///
/// Expression coefficients and the jaw / eye joints are zeroed; the identity
/// (`shape`), the root and neck joints, and the global translation are kept.
/// This is the pose the [`GaussianModel::is_rigid`] Gaussians follow — see the
/// module docs.
#[must_use]
pub fn rigid_pose_params(params: &FlameParams) -> FlameParams {
    // Pose layout: [root(3), neck(3), jaw(3), left_eye(3), right_eye(3)].
    const HEAD_POSE_VALUES: usize = 6;
    let mut pose = params.pose.clone();
    for value in pose.iter_mut().skip(HEAD_POSE_VALUES) {
        *value = 0.0;
    }
    FlameParams {
        shape: params.shape.clone(),
        expression: Vec::new(),
        pose,
        translation: params.translation,
    }
}

// ---------------------------------------------------------------------------
// Triangle frames
// ---------------------------------------------------------------------------

/// The local frame a Gaussian is attached to on one mesh triangle.
#[derive(Debug, Clone, Copy)]
struct TriangleFrame {
    origin: na::Point3<f32>,
    basis: na::Matrix3<f32>,
    scale: f32,
}

/// Build the triangle frame for `face` at barycentric coordinates `bary`.
///
/// Returns `None` for a degenerate triangle (zero-length first edge or zero
/// area), whose basis and scale are not defined.
fn triangle_frame(mesh: &Mesh, face: &[u32; 3], bary: [f32; 3]) -> Option<TriangleFrame> {
    let v0 = mesh.vertices.get(face[0] as usize)?;
    let v1 = mesh.vertices.get(face[1] as usize)?;
    let v2 = mesh.vertices.get(face[2] as usize)?;

    let origin = na::Point3::from(v0.coords * bary[0] + v1.coords * bary[1] + v2.coords * bary[2]);

    let e0 = v1 - v0;
    let e1 = v2 - v0;
    let normal = e0.cross(&e1);
    let double_area = normal.norm();
    let edge_len = e0.norm();
    if double_area <= EPS || edge_len <= EPS {
        return None;
    }

    let x = e0 / edge_len;
    let z = normal / double_area;
    let y = z.cross(&x);

    Some(TriangleFrame {
        origin,
        basis: na::Matrix3::from_columns(&[x, y, z]),
        scale: (0.5 * double_area).sqrt(),
    })
}

/// Per-call diagnostics from a rebind, so a caller can report how much of the
/// model actually moved.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RebindStats {
    /// Gaussians whose position, rotation and scale were transformed.
    pub transformed: usize,
    /// Gaussians left untouched because their rest or posed triangle was
    /// degenerate.
    pub degenerate: usize,
}

// ---------------------------------------------------------------------------
// Binding validation
// ---------------------------------------------------------------------------

/// Reject a model whose FLAME binding cannot drive an animation.
///
/// Two failures matter in practice:
///
/// * mismatched or out-of-range `face_indices` — a corrupt or foreign
///   checkpoint,
/// * the all-zero / `[1/3, 1/3, 1/3]` placeholder that
///   [`crate::export::load_ply`] fills in, because a PLY carries no binding at
///   all.  Animating that would silently collapse the whole avatar onto face 0.
///
/// # Errors
///
/// Returns an error describing which of the two cases was hit.
pub fn ensure_flame_binding(model: &GaussianModel, mesh: &Mesh) -> Result<()> {
    let n = model.len();
    anyhow::ensure!(
        model.face_indices.len() == n && model.barycentric.len() == n && model.is_rigid.len() == n,
        "model has {n} Gaussians but {} face indices, {} barycentric coordinates and {} \
         rigid flags; the FLAME binding is inconsistent",
        model.face_indices.len(),
        model.barycentric.len(),
        model.is_rigid.len(),
    );
    if n == 0 {
        return Ok(());
    }

    let num_faces = mesh.faces.len();
    anyhow::ensure!(num_faces > 0, "FLAME mesh has no faces to bind against");

    if let Some((i, face)) = model
        .face_indices
        .iter()
        .enumerate()
        .find(|(_, &f)| f as usize >= num_faces)
    {
        anyhow::bail!(
            "Gaussian {i} is bound to face {face} but the FLAME mesh has only {num_faces} \
             faces; the model was trained against a different FLAME topology"
        );
    }

    // `num_faces > 1` matters: on a single-triangle mesh, face 0 is the only
    // legal index and a centroid binding is perfectly real, so the
    // placeholder signature is only meaningful where an alternative existed.
    let third = 1.0 / 3.0;
    let placeholder = n > 1
        && num_faces > 1
        && model.face_indices.iter().all(|&f| f == 0)
        && model.barycentric.iter().all(|b| {
            b.iter()
                .all(|c| (c - third).abs() < PLACEHOLDER_BARY_TOLERANCE)
        });
    anyhow::ensure!(
        !placeholder,
        "this model carries no FLAME binding: all {n} Gaussians report face 0 with \
         barycentric [1/3, 1/3, 1/3], which is the placeholder written when a model is \
         loaded from PLY. Animation needs the binding stored in a .json checkpoint — \
         re-export the model as a checkpoint, or render without --flame-params"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Rebinding
// ---------------------------------------------------------------------------

/// Pose `model` by moving every Gaussian from its `rest_mesh` triangle frame to
/// the matching frame on `posed_mesh`.
///
/// # Errors
///
/// Propagates [`ensure_flame_binding`].
pub fn rebind_to_mesh(
    model: &GaussianModel,
    rest_mesh: &Mesh,
    posed_mesh: &Mesh,
) -> Result<(GaussianModel, RebindStats)> {
    rebind_to_meshes(model, rest_mesh, posed_mesh, posed_mesh)
}

/// Pose `model` with a separate target mesh for flexible and rigid Gaussians.
///
/// `flexible_mesh` drives the Gaussians whose [`GaussianModel::is_rigid`] flag
/// is `false`; `rigid_mesh` drives the rest.  Passing the same mesh twice is
/// [`rebind_to_mesh`].
///
/// # Errors
///
/// Propagates [`ensure_flame_binding`] for each target mesh, and fails when the
/// three meshes do not share a topology.
pub fn rebind_to_meshes(
    model: &GaussianModel,
    rest_mesh: &Mesh,
    flexible_mesh: &Mesh,
    rigid_mesh: &Mesh,
) -> Result<(GaussianModel, RebindStats)> {
    ensure_flame_binding(model, rest_mesh)?;
    anyhow::ensure!(
        rest_mesh.faces.len() == flexible_mesh.faces.len()
            && rest_mesh.faces.len() == rigid_mesh.faces.len()
            && rest_mesh.vertices.len() == flexible_mesh.vertices.len()
            && rest_mesh.vertices.len() == rigid_mesh.vertices.len(),
        "rest, flexible and rigid meshes must share a topology: {}/{} vs {}/{} vs {}/{} \
         (vertices/faces)",
        rest_mesh.vertices.len(),
        rest_mesh.faces.len(),
        flexible_mesh.vertices.len(),
        flexible_mesh.faces.len(),
        rigid_mesh.vertices.len(),
        rigid_mesh.faces.len(),
    );

    let mut posed = model.clone();
    let mut stats = RebindStats::default();

    for i in 0..model.len() {
        let (Some(face_index), Some(bary), Some(is_rigid)) = (
            model.face_indices.get(i).copied(),
            model.barycentric.get(i).copied(),
            model.is_rigid.get(i).copied(),
        ) else {
            // `ensure_flame_binding` already proved the three arrays are as
            // long as `model.len()`; this arm cannot be reached.
            stats.degenerate += 1;
            continue;
        };
        let Some(face) = rest_mesh.faces.get(face_index as usize) else {
            stats.degenerate += 1;
            continue;
        };
        let target = if is_rigid { rigid_mesh } else { flexible_mesh };

        let (Some(rest_frame), Some(posed_frame)) = (
            triangle_frame(rest_mesh, face, bary),
            triangle_frame(target, face, bary),
        ) else {
            stats.degenerate += 1;
            continue;
        };

        let Some(gaussian) = posed.gaussians.get_mut(i) else {
            stats.degenerate += 1;
            continue;
        };

        let delta_rotation = posed_frame.basis * rest_frame.basis.transpose();
        let ratio = posed_frame.scale / rest_frame.scale;

        // --- position ---
        let p = na::Point3::new(
            gaussian.position[0],
            gaussian.position[1],
            gaussian.position[2],
        );
        let offset = delta_rotation * ((p - rest_frame.origin) * ratio);
        let moved = posed_frame.origin + offset;
        gaussian.position = [moved.x, moved.y, moved.z];

        // --- rotation: ΔR · q, with the stored (x, y, z, w) convention ---
        let stored = na::Quaternion::new(
            gaussian.rotation[3],
            gaussian.rotation[0],
            gaussian.rotation[1],
            gaussian.rotation[2],
        );
        if stored.norm() > EPS {
            let delta = na::UnitQuaternion::from_rotation_matrix(
                &na::Rotation3::from_matrix_unchecked(delta_rotation),
            );
            let rotated = delta * na::UnitQuaternion::from_quaternion(stored);
            gaussian.rotation = [rotated.i, rotated.j, rotated.k, rotated.w];
        }

        // --- scale: stored in log space, so the ratio is additive ---
        let log_ratio = ratio.ln();
        if log_ratio.is_finite() {
            for axis in &mut gaussian.scale {
                *axis += log_ratio;
            }
        }

        stats.transformed += 1;
    }

    if stats.degenerate > 0 {
        tracing::warn!(
            "{} of {} Gaussians were left in their rest pose: their bound triangle is \
             degenerate on the rest or posed mesh",
            stats.degenerate,
            model.len(),
        );
    }

    Ok((posed, stats))
}

/// A loaded FLAME model plus the parameter sequence to animate it with.
///
/// This is what `oxigaf render --flame-params` needs in hand before it can
/// call [`AnimationSource::frame`] per frame.  Building it here — rather than
/// in the command handler — keeps the *rest-pose convention* in one place: the
/// reconstruction pipeline initialises its Gaussians on
/// `flame.forward(params[0])`, falling back to [`FlameParams::neutral`] when no
/// sequence was supplied, and [`AnimationSource::rest_params`] reproduces
/// exactly that choice.  A caller that picked a different rest pose would bake
/// a constant offset into every frame.
pub struct AnimationSource {
    flame: FlameModel,
    /// The rest pose, held separately so it is always available without an
    /// index — `frames` is non-empty by construction, but keeping this field
    /// means no accessor needs to prove that again.
    rest: FlameParams,
    frames: Vec<FlameParams>,
}

impl AnimationSource {
    /// Load a FLAME model directory and a `--flame-params` sequence.
    ///
    /// # Errors
    ///
    /// Returns an error when the FLAME model directory cannot be loaded (see
    /// [`FlameModel::load`]) or when the parameter file cannot be read (see
    /// [`load_flame_param_sequence`]).
    pub fn load(flame_model_dir: &Path, params_path: &Path) -> Result<Self> {
        let flame = FlameModel::load(flame_model_dir).map_err(|e| {
            anyhow::anyhow!(
                "Failed to load FLAME model from {}: {e}",
                flame_model_dir.display()
            )
        })?;
        let frames = load_flame_param_sequence(params_path)?;
        // `load_flame_param_sequence` rejects an empty sequence, so `first`
        // is present; cloning it here removes the last indexing operation
        // from the per-frame path.
        let rest = frames
            .first()
            .cloned()
            .context("FLAME parameter sequence is empty")?;
        Ok(Self {
            flame,
            rest,
            frames,
        })
    }

    /// Number of parameter frames available.
    #[must_use]
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// The frame index [`Self::frame`] will actually use for `index`.
    ///
    /// Exposed so a caller can report the hold — "rendered 60 frames from a
    /// 30-frame sequence, holding the last pose" — rather than leaving it
    /// silent.
    #[must_use]
    pub fn clamped_index(&self, index: usize) -> usize {
        index.min(self.frames.len().saturating_sub(1))
    }

    /// The rest pose the bound Gaussians were trained against.
    ///
    /// This is the first frame of the sequence, matching what
    /// `pipeline::run_reconstruction` used to initialise them.
    #[must_use]
    pub fn rest_params(&self) -> &FlameParams {
        &self.rest
    }

    /// Pose `model` for frame `index`.
    ///
    /// Indices at or past the end of the sequence hold on the last frame,
    /// which is what a render of `--num-frames 60` against a 30-frame sequence
    /// should do rather than failing halfway through.
    ///
    /// # Errors
    ///
    /// Propagates [`animate_model`] — most importantly the explicit error for
    /// a model that carries no FLAME binding.
    pub fn frame(
        &self,
        model: &GaussianModel,
        index: usize,
    ) -> Result<(GaussianModel, RebindStats)> {
        let clamped = self.clamped_index(index);
        let frame_params = self
            .frames
            .get(clamped)
            .context("FLAME parameter sequence is empty")?;
        animate_model(&self.flame, model, self.rest_params(), frame_params)
    }
}

/// Pose `model` for one frame of FLAME parameters.
///
/// `rest_params` are the parameters the model was **trained** against — the
/// reconstruction pipeline initialises Gaussians on `flame.forward(params[0])`
/// (falling back to [`FlameParams::neutral`] when no sequence was given), so
/// the first frame of the same sequence is normally the right value.  Passing
/// the wrong rest pose does not fail; it bakes a constant offset into every
/// frame, which is why the caller must choose rather than this function
/// guessing.
///
/// Rigid Gaussians follow [`rigid_pose_params`] of `frame_params`; flexible
/// ones follow `frame_params` in full.  FLAME is only evaluated for the rigid
/// pose when the model actually contains a rigid Gaussian.
///
/// # Errors
///
/// Propagates [`ensure_flame_binding`].
pub fn animate_model(
    flame: &FlameModel,
    model: &GaussianModel,
    rest_params: &FlameParams,
    frame_params: &FlameParams,
) -> Result<(GaussianModel, RebindStats)> {
    let rest_mesh = flame.forward(rest_params);
    let flexible_mesh = flame.forward(frame_params);

    let has_rigid = model.is_rigid.iter().any(|&r| r);
    let rigid_mesh = if has_rigid {
        Some(flame.forward(&rigid_pose_params(frame_params)))
    } else {
        None
    };

    rebind_to_meshes(
        model,
        &rest_mesh,
        &flexible_mesh,
        rigid_mesh.as_ref().unwrap_or(&flexible_mesh),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxigaf::render::gaussian::GaussianAttributes;

    fn unit_triangle() -> Mesh {
        Mesh::new(
            vec![
                na::Point3::new(0.0, 0.0, 0.0),
                na::Point3::new(1.0, 0.0, 0.0),
                na::Point3::new(0.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2]],
        )
    }

    /// The same triangle, uniformly scaled by `k` about the origin and then
    /// translated by `t`.
    fn scaled_translated(k: f32, t: [f32; 3]) -> Mesh {
        Mesh::new(
            vec![
                na::Point3::new(t[0], t[1], t[2]),
                na::Point3::new(k + t[0], t[1], t[2]),
                na::Point3::new(t[0], k + t[1], t[2]),
            ],
            vec![[0, 1, 2]],
        )
    }

    /// The same triangle rotated +90° about the Z axis.
    fn rotated_90_z() -> Mesh {
        Mesh::new(
            vec![
                na::Point3::new(0.0, 0.0, 0.0),
                na::Point3::new(0.0, 1.0, 0.0),
                na::Point3::new(-1.0, 0.0, 0.0),
            ],
            vec![[0, 1, 2]],
        )
    }

    fn bound_model(position: [f32; 3], bary: [f32; 3], is_rigid: bool) -> GaussianModel {
        GaussianModel {
            gaussians: vec![GaussianAttributes {
                position,
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [-1.0; 3],
                opacity: 1.0,
            }],
            sh_coeffs: vec![0.0, 0.0, 0.0],
            sh_degree: 0,
            face_indices: vec![0],
            barycentric: vec![bary],
            local_offsets: vec![[0.0; 3]],
            is_rigid: vec![is_rigid],
        }
    }

    #[test]
    fn rebind_moves_a_centroid_gaussian_onto_the_posed_centroid() {
        let third = 1.0 / 3.0;
        let rest = unit_triangle();
        let posed = scaled_translated(2.0, [0.0, 0.0, 1.0]);
        let model = bound_model([third, third, 0.0], [third; 3], false);

        let (out, stats) = rebind_to_mesh(&model, &rest, &posed).expect("rebind ok");
        assert_eq!(stats.transformed, 1);
        assert_eq!(stats.degenerate, 0);

        let p = out.gaussians[0].position;
        assert!((p[0] - 2.0 * third).abs() < 1e-5, "x = {}", p[0]);
        assert!((p[1] - 2.0 * third).abs() < 1e-5, "y = {}", p[1]);
        assert!((p[2] - 1.0).abs() < 1e-5, "z = {}", p[2]);
    }

    #[test]
    fn rebind_adds_the_log_scale_ratio_instead_of_multiplying() {
        // Regression: scales are stored in log space, so doubling the
        // triangle must ADD ln(2) — multiplying by 2 would give -2.0 here.
        let third = 1.0 / 3.0;
        let rest = unit_triangle();
        let posed = scaled_translated(2.0, [0.0; 3]);
        let model = bound_model([third, third, 0.0], [third; 3], false);

        let (out, _) = rebind_to_mesh(&model, &rest, &posed).expect("rebind ok");
        let expected = -1.0 + 2.0_f32.ln();
        for axis in out.gaussians[0].scale {
            assert!(
                (axis - expected).abs() < 1e-5,
                "log-scale was {axis}, expected {expected}"
            );
        }
    }

    #[test]
    fn rebind_carries_a_trained_offset_through_the_rotation() {
        // A Gaussian that drifted off its barycentric point during training
        // must keep that offset, rotated with the triangle — not snap back
        // onto the barycentric point.
        let rest = unit_triangle();
        let posed = rotated_90_z();
        // Barycentric origin (1, 0, 0) is vertex 0; the Gaussian sits 0.5
        // along +X from it, which +90° about Z sends to +Y.
        let model = bound_model([0.5, 0.0, 0.0], [1.0, 0.0, 0.0], false);

        let (out, _) = rebind_to_mesh(&model, &rest, &posed).expect("rebind ok");
        let p = out.gaussians[0].position;
        assert!((p[0] - 0.0).abs() < 1e-5, "x = {}", p[0]);
        assert!((p[1] - 0.5).abs() < 1e-5, "y = {}", p[1]);

        // The stored quaternion must pick up the same +90° about Z.
        let q = out.gaussians[0].rotation;
        let unit = na::UnitQuaternion::from_quaternion(na::Quaternion::new(q[3], q[0], q[1], q[2]));
        let turned = unit * na::Vector3::new(1.0, 0.0, 0.0);
        assert!((turned.x - 0.0).abs() < 1e-5, "rotated x = {}", turned.x);
        assert!((turned.y - 1.0).abs() < 1e-5, "rotated y = {}", turned.y);
    }

    #[test]
    fn rebind_is_the_identity_when_the_mesh_does_not_move() {
        let third = 1.0 / 3.0;
        let rest = unit_triangle();
        let model = bound_model([0.2, 0.3, 0.4], [third; 3], false);
        let (out, _) = rebind_to_mesh(&model, &rest, &rest).expect("rebind ok");
        for k in 0..3 {
            assert!(
                (out.gaussians[0].position[k] - model.gaussians[0].position[k]).abs() < 1e-6,
                "component {k} moved"
            );
            assert!((out.gaussians[0].scale[k] - (-1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn rigid_and_flexible_gaussians_follow_different_meshes() {
        let third = 1.0 / 3.0;
        let rest = unit_triangle();
        let flexible = scaled_translated(1.0, [0.0, 0.0, 1.0]);
        let rigid = scaled_translated(1.0, [0.0, 0.0, -1.0]);

        let mut model = bound_model([third, third, 0.0], [third; 3], false);
        model.gaussians.push(model.gaussians[0]);
        model.sh_coeffs.extend_from_slice(&[0.0, 0.0, 0.0]);
        model.face_indices.push(0);
        model.barycentric.push([third; 3]);
        model.local_offsets.push([0.0; 3]);
        model.is_rigid.push(true);

        let (out, stats) = rebind_to_meshes(&model, &rest, &flexible, &rigid).expect("rebind ok");
        assert_eq!(stats.transformed, 2);
        assert!(
            (out.gaussians[0].position[2] - 1.0).abs() < 1e-5,
            "flexible Gaussian must follow the expression mesh"
        );
        assert!(
            (out.gaussians[1].position[2] + 1.0).abs() < 1e-5,
            "rigid Gaussian must follow the head-bone-only mesh"
        );
    }

    #[test]
    fn ply_placeholder_binding_is_rejected_instead_of_collapsing_the_avatar() {
        // Regression: `load_ply` fills `face_indices` with zeros and
        // `barycentric` with [1/3, 1/3, 1/3]. Animating that would drag every
        // Gaussian onto face 0.
        let mesh = Mesh::new(
            vec![
                na::Point3::new(0.0, 0.0, 0.0),
                na::Point3::new(1.0, 0.0, 0.0),
                na::Point3::new(0.0, 1.0, 0.0),
                na::Point3::new(1.0, 1.0, 0.0),
            ],
            vec![[0, 1, 2], [1, 3, 2]],
        );
        let third = 1.0 / 3.0;
        let mut model = bound_model([0.0; 3], [third; 3], false);
        model.gaussians.push(model.gaussians[0]);
        model.sh_coeffs.extend_from_slice(&[0.0, 0.0, 0.0]);
        model.face_indices.push(0);
        model.barycentric.push([third; 3]);
        model.local_offsets.push([0.0; 3]);
        model.is_rigid.push(false);

        let err = ensure_flame_binding(&model, &mesh)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            err.contains("no FLAME binding"),
            "error was: {err:?} — the PLY placeholder must be an explicit error"
        );
        assert!(rebind_to_mesh(&model, &mesh, &mesh).is_err());
    }

    #[test]
    fn a_single_face_mesh_is_not_mistaken_for_the_ply_placeholder() {
        // Companion to the test above: on a one-triangle mesh, face 0 is the
        // only legal index, so an all-zero `face_indices` is real binding, not
        // the placeholder. Rejecting it would break every single-face caller.
        let third = 1.0 / 3.0;
        let mesh = unit_triangle();
        let mut model = bound_model([0.0; 3], [third; 3], false);
        model.gaussians.push(model.gaussians[0]);
        model.sh_coeffs.extend_from_slice(&[0.0, 0.0, 0.0]);
        model.face_indices.push(0);
        model.barycentric.push([third; 3]);
        model.local_offsets.push([0.0; 3]);
        model.is_rigid.push(false);

        ensure_flame_binding(&model, &mesh).expect("a single-face binding must be accepted");
    }

    #[test]
    fn out_of_range_face_index_is_an_error() {
        let mesh = unit_triangle();
        let mut model = bound_model([0.0; 3], [1.0, 0.0, 0.0], false);
        model.face_indices[0] = 7;
        let err = ensure_flame_binding(&model, &mesh)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("face 7"), "error was: {err:?}");
    }

    #[test]
    fn degenerate_triangles_leave_their_gaussians_untouched() {
        let rest = Mesh::new(
            vec![
                na::Point3::new(0.0, 0.0, 0.0),
                na::Point3::new(0.0, 0.0, 0.0),
                na::Point3::new(0.0, 0.0, 0.0),
            ],
            vec![[0, 1, 2]],
        );
        let posed = unit_triangle();
        let model = bound_model([0.4, 0.5, 0.6], [1.0, 0.0, 0.0], false);
        let (out, stats) = rebind_to_mesh(&model, &rest, &posed).expect("rebind ok");
        assert_eq!(stats.degenerate, 1);
        assert_eq!(stats.transformed, 0);
        assert_eq!(out.gaussians[0].position, [0.4, 0.5, 0.6]);
    }

    #[test]
    fn rigid_pose_params_keeps_identity_and_head_bone_only() {
        let params = FlameParams {
            shape: vec![0.5, -0.25],
            expression: vec![0.9, 0.9],
            pose: vec![
                0.1, 0.2, 0.3, // root
                0.4, 0.5, 0.6, // neck
                0.7, 0.8, 0.9, // jaw
                1.0, 1.1, 1.2, // left eye
                1.3, 1.4, 1.5, // right eye
            ],
            translation: [0.01, 0.02, 0.03],
        };
        let rigid = rigid_pose_params(&params);
        assert_eq!(rigid.shape, params.shape, "identity must survive");
        assert!(rigid.expression.is_empty(), "expression must be zeroed");
        assert_eq!(
            &rigid.pose[..6],
            &params.pose[..6],
            "head bone must survive"
        );
        assert!(
            rigid.pose[6..].iter().all(|v| *v == 0.0),
            "jaw and eyes must be zeroed, got {:?}",
            &rigid.pose[6..]
        );
        assert_eq!(rigid.translation, params.translation);
    }

    fn temp_json(name: &str, body: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "oxigaf_flame_anim_{name}_{}.json",
            std::process::id()
        ));
        std::fs::write(&path, body).expect("write temp json");
        path
    }

    #[test]
    fn sequence_loader_accepts_a_bare_array() {
        let path = temp_json(
            "bare",
            r#"[{"shape":[0.1],"expression":[0.2],"pose":[0.0,0.0,0.0]},
                {"shape":[0.1],"expression":[0.3],"pose":[0.0,0.0,0.0]}]"#,
        );
        let frames = load_flame_param_sequence(&path).expect("load ok");
        let _ = std::fs::remove_file(&path);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[1].expression, vec![0.3]);
        assert_eq!(
            frames[0].pose.len(),
            FlameParams::NUM_JOINTS * 3,
            "a short pose must be zero-padded to the full joint set"
        );
    }

    #[test]
    fn sequence_loader_accepts_the_frames_wrapper_and_the_expr_alias() {
        let path = temp_json(
            "wrapped",
            r#"{"fps":30.0,"frames":[{"shape":[],"expr":[0.4],"pose":[]}]}"#,
        );
        let frames = load_flame_param_sequence(&path).expect("load ok");
        let _ = std::fs::remove_file(&path);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].expression, vec![0.4]);
    }

    /// Build an `AnimationSource`-shaped index clamp without needing a real
    /// FLAME model directory on disk. Only the frame-selection rule is under
    /// test here; the deformation itself is covered by the `rebind_*` tests.
    fn clamp(num_frames: usize, index: usize) -> usize {
        index.min(num_frames.saturating_sub(1))
    }

    #[test]
    fn frame_index_holds_on_the_last_pose_past_the_end() {
        // Rendering `--num-frames 60` against a 30-frame sequence must hold
        // the final pose rather than failing or wrapping back to frame 0.
        assert_eq!(clamp(30, 0), 0);
        assert_eq!(clamp(30, 29), 29);
        assert_eq!(clamp(30, 59), 29);
        assert_eq!(clamp(1, 100), 0, "a single-frame sequence is a static pose");
    }

    #[test]
    fn sequence_loader_rejects_empty_and_non_finite_frames() {
        let empty = temp_json("empty", "[]");
        assert!(load_flame_param_sequence(&empty).is_err());
        let _ = std::fs::remove_file(&empty);

        let nan = temp_json("nan", r#"[{"shape":[1e999],"expression":[],"pose":[]}]"#);
        assert!(load_flame_param_sequence(&nan).is_err());
        let _ = std::fs::remove_file(&nan);

        let long = temp_json(
            "longpose",
            r#"[{"shape":[],"expression":[],"pose":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}]"#,
        );
        assert!(load_flame_param_sequence(&long).is_err());
        let _ = std::fs::remove_file(&long);
    }
}
