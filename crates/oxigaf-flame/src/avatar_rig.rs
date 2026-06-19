//! High-level avatar rig that maps intuitive control parameters to FLAME parameters.
//!
//! This module provides a user-friendly animation interface that converts human-readable
//! controls (smile, brow raise, head tilt, etc.) into FLAME expression and pose parameter
//! deltas. It also provides keyframe animation, interpolation, and statistics utilities.
//!
//! # Quick Start
//!
//! ```rust
//! use oxigaf_flame::avatar_rig::{standard_face_rig, apply_rig_to_params, FlameRigParams};
//!
//! let mut rig = standard_face_rig(50);
//! rig.set("smile", 0.8).expect("set smile");
//! let base = FlameRigParams::neutral(50, 15);
//! let modified = apply_rig_to_params(&rig, &base).expect("apply rig");
//! println!("Expression[0] = {}", modified.expression[0]);
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors arising from avatar rig operations.
#[derive(Debug, Error)]
pub enum RigError {
    /// A control value is invalid (NaN, Inf, or min > max configuration error).
    #[error("Invalid control value: {control} = {value}, expected [{min}, {max}]")]
    InvalidControlValue {
        /// Name of the control.
        control: String,
        /// The invalid value.
        value: f32,
        /// Minimum allowed value.
        min: f32,
        /// Maximum allowed value.
        max: f32,
    },

    /// The requested control does not exist in the rig.
    #[error("Unknown control: {0}")]
    UnknownControl(String),

    /// Configuration error in rig setup.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// The rig has no controls defined.
    #[error("Empty rig")]
    EmptyRig,

    /// Attempting to add a control that already exists.
    #[error("Control already exists: {0}")]
    DuplicateControl(String),

    /// A vector dimension does not match the expected size.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension provided.
        actual: usize,
    },
}

// ---------------------------------------------------------------------------
// RigControl
// ---------------------------------------------------------------------------

/// A named control with a defined range and current value.
///
/// Controls represent single scalar animation parameters such as "smile" or "`head_tilt`".
/// Values are clamped to `[min, max]` on assignment.
#[derive(Debug, Clone)]
pub struct RigControl {
    /// Unique name for this control.
    pub name: String,
    /// Current value, always within `[min, max]`.
    pub value: f32,
    /// Minimum allowed value (default `-1.0`).
    pub min: f32,
    /// Maximum allowed value (default `1.0`).
    pub max: f32,
    /// The default/reset value (default `0.0`).
    pub default: f32,
    /// Human-readable description of what this control does.
    pub description: String,
}

impl RigControl {
    /// Create a new control with the given name, range, and default value.
    ///
    /// The current value is set to `default`.
    #[must_use]
    pub fn new(name: impl Into<String>, min: f32, max: f32, default: f32) -> Self {
        let name = name.into();
        let value = default.clamp(min, max);
        Self {
            name,
            value,
            min,
            max,
            default,
            description: String::new(),
        }
    }

    /// Returns the current value normalized to `[0, 1]` within `[min, max]`.
    ///
    /// If `min == max`, returns `0.0` to avoid division by zero.
    #[must_use]
    pub fn normalized_value(&self) -> f32 {
        let range = self.max - self.min;
        if range.abs() < f32::EPSILON {
            return 0.0;
        }
        (self.value - self.min) / range
    }

    /// Set the current value, clamping it to `[min, max]`.
    ///
    /// Returns `Err(RigError::InvalidControlValue)` if the value is non-finite.
    ///
    /// # Errors
    ///
    /// Returns an error if `v` is NaN or infinite.
    pub fn set_value(&mut self, v: f32) -> Result<(), RigError> {
        if !v.is_finite() {
            return Err(RigError::InvalidControlValue {
                control: self.name.clone(),
                value: v,
                min: self.min,
                max: self.max,
            });
        }
        self.value = v.clamp(self.min, self.max);
        Ok(())
    }

    /// Reset the current value to the default.
    pub fn reset(&mut self) {
        self.value = self.default;
    }
}

// ---------------------------------------------------------------------------
// BlendTarget
// ---------------------------------------------------------------------------

/// Maps a control's value to deltas in FLAME expression and pose parameters.
///
/// When evaluated, each affected parameter receives `control_value * weight[i]`
/// added to it.
#[derive(Debug, Clone)]
pub struct BlendTarget {
    /// Name of the control this target responds to.
    pub control_name: String,
    /// Indices into the expression parameter vector to affect.
    pub expression_indices: Vec<usize>,
    /// Scaling factors for each expression index (one per index).
    pub weights: Vec<f32>,
    /// Indices into the pose parameter vector to affect.
    pub pose_indices: Vec<usize>,
    /// Scaling factors for each pose index (one per index).
    pub pose_weights: Vec<f32>,
}

// ---------------------------------------------------------------------------
// AvatarRig
// ---------------------------------------------------------------------------

/// A complete avatar rig: a set of named controls and their blend targets.
///
/// The rig accumulates deltas by evaluating each blend target against the
/// current control values and summing the contributions.
#[derive(Debug, Clone)]
pub struct AvatarRig {
    /// All controls registered in this rig.
    pub controls: Vec<RigControl>,
    /// All blend targets (mappings from controls to parameters).
    pub blend_targets: Vec<BlendTarget>,
    /// Number of FLAME expression parameters (e.g., 50).
    pub expression_dim: usize,
    /// Number of FLAME pose parameters (e.g., 15).
    pub pose_dim: usize,
}

impl AvatarRig {
    /// Create an empty rig with the given parameter dimensions.
    #[must_use]
    pub fn new(expression_dim: usize, pose_dim: usize) -> Self {
        Self {
            controls: Vec::new(),
            blend_targets: Vec::new(),
            expression_dim,
            pose_dim,
        }
    }

    /// Add a control to the rig.
    ///
    /// # Errors
    ///
    /// Returns [`RigError::DuplicateControl`] if a control with the same name already exists.
    pub fn add_control(&mut self, control: RigControl) -> Result<(), RigError> {
        if self.controls.iter().any(|c| c.name == control.name) {
            return Err(RigError::DuplicateControl(control.name));
        }
        self.controls.push(control);
        Ok(())
    }

    /// Add a blend target to the rig.
    ///
    /// # Errors
    ///
    /// Returns [`RigError::InvalidConfig`] if `expression_indices` and `weights` have different
    /// lengths, or if `pose_indices` and `pose_weights` have different lengths.
    pub fn add_blend_target(&mut self, target: BlendTarget) -> Result<(), RigError> {
        if target.expression_indices.len() != target.weights.len() {
            return Err(RigError::InvalidConfig(format!(
                "blend target '{}': expression_indices len {} != weights len {}",
                target.control_name,
                target.expression_indices.len(),
                target.weights.len()
            )));
        }
        if target.pose_indices.len() != target.pose_weights.len() {
            return Err(RigError::InvalidConfig(format!(
                "blend target '{}': pose_indices len {} != pose_weights len {}",
                target.control_name,
                target.pose_indices.len(),
                target.pose_weights.len()
            )));
        }
        self.blend_targets.push(target);
        Ok(())
    }

    /// Return a shared reference to the named control, or `None` if not found.
    #[must_use]
    pub fn get_control(&self, name: &str) -> Option<&RigControl> {
        self.controls.iter().find(|c| c.name == name)
    }

    /// Return a mutable reference to the named control, or `None` if not found.
    #[must_use]
    pub fn get_control_mut(&mut self, name: &str) -> Option<&mut RigControl> {
        self.controls.iter_mut().find(|c| c.name == name)
    }

    /// Set a control value by name.
    ///
    /// # Errors
    ///
    /// - [`RigError::UnknownControl`] if no control with this name exists.
    /// - [`RigError::InvalidControlValue`] if the value is non-finite.
    pub fn set(&mut self, name: &str, value: f32) -> Result<(), RigError> {
        let control = self
            .controls
            .iter_mut()
            .find(|c| c.name == name)
            .ok_or_else(|| RigError::UnknownControl(name.to_owned()))?;
        control.set_value(value)
    }

    /// Get the current value of a control by name.
    ///
    /// # Errors
    ///
    /// Returns [`RigError::UnknownControl`] if no control with this name exists.
    pub fn get(&self, name: &str) -> Result<f32, RigError> {
        self.controls
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.value)
            .ok_or_else(|| RigError::UnknownControl(name.to_owned()))
    }

    /// Reset all controls to their default values.
    pub fn reset_all(&mut self) {
        for control in &mut self.controls {
            control.reset();
        }
    }

    /// Return the number of controls in the rig.
    #[must_use]
    pub fn n_controls(&self) -> usize {
        self.controls.len()
    }

    /// Evaluate the current control state, returning expression and pose parameter deltas.
    ///
    /// Returns `(expression_deltas, pose_deltas)` where each vector has length
    /// `expression_dim` and `pose_dim` respectively.
    ///
    /// Each blend target contributes `control_value * weight` to the affected indices.
    ///
    /// # Errors
    ///
    /// - [`RigError::EmptyRig`] if there are no controls.
    /// - [`RigError::DimensionMismatch`] if a blend target references an out-of-bounds index.
    pub fn evaluate(&self) -> Result<(Vec<f32>, Vec<f32>), RigError> {
        if self.controls.is_empty() {
            return Err(RigError::EmptyRig);
        }

        let mut expression_deltas = vec![0.0_f32; self.expression_dim];
        let mut pose_deltas = vec![0.0_f32; self.pose_dim];

        for target in &self.blend_targets {
            // Find the current value of the referenced control
            let control_value = match self.controls.iter().find(|c| c.name == target.control_name) {
                Some(c) => c.value,
                None => continue, // Blend target for missing control — skip silently
            };

            // Accumulate expression deltas
            for (idx_pos, &idx) in target.expression_indices.iter().enumerate() {
                if idx >= self.expression_dim {
                    return Err(RigError::DimensionMismatch {
                        expected: self.expression_dim,
                        actual: idx + 1,
                    });
                }
                expression_deltas[idx] += control_value * target.weights[idx_pos];
            }

            // Accumulate pose deltas
            for (idx_pos, &idx) in target.pose_indices.iter().enumerate() {
                if idx >= self.pose_dim {
                    return Err(RigError::DimensionMismatch {
                        expected: self.pose_dim,
                        actual: idx + 1,
                    });
                }
                pose_deltas[idx] += control_value * target.pose_weights[idx_pos];
            }
        }

        Ok((expression_deltas, pose_deltas))
    }
}

// ---------------------------------------------------------------------------
// standard_face_rig factory
// ---------------------------------------------------------------------------

/// Build a standard FLAME face rig with 12 intuitive controls.
///
/// Controls and their blend targets are tuned for the first ~10 FLAME expression
/// PCA components and standard pose layout (root 0-2, neck 3-5, jaw 6-8, `left_eye` 9-11,
/// `right_eye` 12-14).
///
/// | Control | Range | Description |
/// |---|---|---|
/// | `smile` | [-1, 1] | Happy (+) / sad (-) corners |
/// | `mouth_open` | [0, 1] | Jaw opening |
/// | `brow_raise` | [-1, 1] | Brows up (+) / down (-) |
/// | `brow_furrow` | [0, 1] | Concerned furrowed brows |
/// | `eye_squint` | [0, 1] | Squinting eyelids |
/// | `cheek_puff` | [0, 1] | Puffed cheeks |
/// | `lip_pucker` | [0, 1] | Pursed / puckered lips |
/// | `head_tilt` | [-1, 1] | Head roll angle |
/// | `head_nod` | [-1, 1] | Head pitch (up/down) |
/// | `head_turn` | [-1, 1] | Head yaw (left/right) |
/// | `eye_look_left` | [-1, 1] | Eye gaze horizontal |
/// | `eye_look_up` | [-1, 1] | Eye gaze vertical |
#[must_use]
pub fn standard_face_rig(expression_dim: usize) -> AvatarRig {
    // pose_dim: root(3) + neck(3) + jaw(3) + left_eye(3) + right_eye(3) = 15
    let pose_dim = 15_usize;
    let mut rig = AvatarRig::new(expression_dim, pose_dim);
    add_facial_expression_controls(&mut rig, expression_dim);
    add_head_pose_controls(&mut rig);
    add_eye_gaze_controls(&mut rig);
    rig
}

/// Static descriptor for a single expression control + blend-target registration.
struct ExprControlSpec {
    name: &'static str,
    min: f32,
    max: f32,
    default: f32,
    description: &'static str,
    /// `(expression_index, weight)` pairs; indices exceeding `expression_dim` are filtered out.
    iw_pairs: &'static [(usize, f32)],
    /// `(pose_index, pose_weight)` pairs (empty for pure expression controls).
    pose_iw_pairs: &'static [(usize, f32)],
}

/// Register one control + blend-target described by `spec`, clamping expression indices to
/// those that are valid for `expression_dim`.
fn add_expr_control(rig: &mut AvatarRig, expression_dim: usize, spec: &ExprControlSpec) {
    let mut ctrl = RigControl::new(spec.name, spec.min, spec.max, spec.default);
    spec.description
        .to_owned()
        .clone_into(&mut ctrl.description);
    rig.add_control(ctrl).ok();
    let (expr_i, weights): (Vec<usize>, Vec<f32>) = spec
        .iw_pairs
        .iter()
        .filter_map(|&(i, w)| {
            if i < expression_dim {
                Some((i, w))
            } else {
                None
            }
        })
        .unzip();
    let (pose_i, pose_w): (Vec<usize>, Vec<f32>) = spec.pose_iw_pairs.iter().copied().unzip();
    rig.add_blend_target(BlendTarget {
        control_name: spec.name.to_owned(),
        expression_indices: expr_i,
        weights,
        pose_indices: pose_i,
        pose_weights: pose_w,
    })
    .ok();
}

/// Table of all facial expression controls for the standard face rig.
static FACIAL_EXPR_SPECS: &[ExprControlSpec] = &[
    ExprControlSpec {
        name: "smile",
        min: -1.0,
        max: 1.0,
        default: 0.0,
        description: "Happy (+1) to sad (-1) expression",
        iw_pairs: &[(0, 0.50_f32), (1, -0.20_f32), (2, 0.10_f32), (3, 0.08_f32)],
        pose_iw_pairs: &[],
    },
    ExprControlSpec {
        name: "mouth_open",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Jaw opening (0 = closed, 1 = fully open)",
        iw_pairs: &[],
        pose_iw_pairs: &[(6, 0.35_f32)],
    },
    ExprControlSpec {
        name: "brow_raise",
        min: -1.0,
        max: 1.0,
        default: 0.0,
        description: "Brows up (+1) or down (-1)",
        iw_pairs: &[(4, 0.35_f32), (5, 0.25_f32), (6, -0.10_f32)],
        pose_iw_pairs: &[],
    },
    ExprControlSpec {
        name: "brow_furrow",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Furrowed/concerned brows (0 = neutral, 1 = max furrow)",
        iw_pairs: &[(5, 0.30_f32), (4, -0.15_f32), (7, 0.20_f32)],
        pose_iw_pairs: &[],
    },
    ExprControlSpec {
        name: "eye_squint",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Squinting eyelids (0 = open, 1 = max squint)",
        iw_pairs: &[(8, 0.40_f32), (3, 0.15_f32)],
        pose_iw_pairs: &[],
    },
    ExprControlSpec {
        name: "cheek_puff",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Cheek puffing (0 = neutral, 1 = max puff)",
        iw_pairs: &[(9, 0.45_f32), (2, 0.10_f32)],
        pose_iw_pairs: &[],
    },
    ExprControlSpec {
        name: "lip_pucker",
        min: 0.0,
        max: 1.0,
        default: 0.0,
        description: "Lip puckering/pursing (0 = neutral, 1 = max pucker)",
        iw_pairs: &[(1, 0.30_f32), (0, -0.10_f32), (2, 0.20_f32)],
        pose_iw_pairs: &[],
    },
];

/// Register facial expression blend-targets (smile, mouth, brows, squint, cheeks, lips).
fn add_facial_expression_controls(rig: &mut AvatarRig, expression_dim: usize) {
    for spec in FACIAL_EXPR_SPECS {
        add_expr_control(rig, expression_dim, spec);
    }
}

/// Register head-pose blend-targets (tilt/roll, nod/pitch, turn/yaw).
fn add_head_pose_controls(rig: &mut AvatarRig) {
    // --- head_tilt [-1, 1]: roll — root pose z-axis (index 2) ---
    let mut ctrl = RigControl::new("head_tilt", -1.0, 1.0, 0.0);
    "Head roll angle (+1 = tilt right, -1 = tilt left)"
        .to_owned()
        .clone_into(&mut ctrl.description);
    rig.add_control(ctrl).ok();
    rig.add_blend_target(BlendTarget {
        control_name: "head_tilt".to_owned(),
        expression_indices: vec![],
        weights: vec![],
        pose_indices: vec![2],
        pose_weights: vec![0.30],
    })
    .ok();

    // --- head_nod [-1, 1]: pitch — root pose x-axis (index 0) ---
    let mut ctrl = RigControl::new("head_nod", -1.0, 1.0, 0.0);
    "Head pitch (+1 = nod down, -1 = look up)"
        .to_owned()
        .clone_into(&mut ctrl.description);
    rig.add_control(ctrl).ok();
    rig.add_blend_target(BlendTarget {
        control_name: "head_nod".to_owned(),
        expression_indices: vec![],
        weights: vec![],
        pose_indices: vec![0],
        pose_weights: vec![0.30],
    })
    .ok();

    // --- head_turn [-1, 1]: yaw — root pose y-axis (index 1) ---
    let mut ctrl = RigControl::new("head_turn", -1.0, 1.0, 0.0);
    "Head yaw (+1 = turn right, -1 = turn left)"
        .to_owned()
        .clone_into(&mut ctrl.description);
    rig.add_control(ctrl).ok();
    rig.add_blend_target(BlendTarget {
        control_name: "head_turn".to_owned(),
        expression_indices: vec![],
        weights: vec![],
        pose_indices: vec![1],
        pose_weights: vec![0.30],
    })
    .ok();
}

/// Register eye-gaze blend-targets (horizontal look-left, vertical look-up).
fn add_eye_gaze_controls(rig: &mut AvatarRig) {
    // --- eye_look_left [-1, 1]: horizontal — left eye y (10), right eye y (13) ---
    let mut ctrl = RigControl::new("eye_look_left", -1.0, 1.0, 0.0);
    "Eye gaze horizontal (+1 = look left, -1 = look right)"
        .to_owned()
        .clone_into(&mut ctrl.description);
    rig.add_control(ctrl).ok();
    rig.add_blend_target(BlendTarget {
        control_name: "eye_look_left".to_owned(),
        expression_indices: vec![],
        weights: vec![],
        pose_indices: vec![10, 13],
        pose_weights: vec![0.20, 0.20],
    })
    .ok();

    // --- eye_look_up [-1, 1]: vertical — left eye x (9), right eye x (12) ---
    let mut ctrl = RigControl::new("eye_look_up", -1.0, 1.0, 0.0);
    "Eye gaze vertical (+1 = look up, -1 = look down)"
        .to_owned()
        .clone_into(&mut ctrl.description);
    rig.add_control(ctrl).ok();
    rig.add_blend_target(BlendTarget {
        control_name: "eye_look_up".to_owned(),
        expression_indices: vec![],
        weights: vec![],
        pose_indices: vec![9, 12],
        pose_weights: vec![0.20, 0.20],
    })
    .ok();
}

// ---------------------------------------------------------------------------
// interpolate_rigs
// ---------------------------------------------------------------------------

/// Linearly interpolate all matching controls between two rigs at parameter `t` in `[0, 1]`.
///
/// Controls present in `rig_a` but not in `rig_b` retain `rig_a` values.
/// The resulting rig has `rig_a`'s blend targets, `expression_dim`, and `pose_dim`.
///
/// # Errors
///
/// Returns [`RigError::InvalidConfig`] if `t` is non-finite.
pub fn interpolate_rigs(
    rig_a: &AvatarRig,
    rig_b: &AvatarRig,
    t: f32,
) -> Result<AvatarRig, RigError> {
    if !t.is_finite() {
        return Err(RigError::InvalidConfig(format!(
            "interpolation t must be finite, got {t}"
        )));
    }

    let t_clamped = t.clamp(0.0, 1.0);
    let mut result = rig_a.clone();

    for ctrl in &mut result.controls {
        if let Some(ctrl_b) = rig_b.get_control(&ctrl.name) {
            let lerped = ctrl.value + (ctrl_b.value - ctrl.value) * t_clamped;
            // Clamp to result control's own range
            ctrl.value = lerped.clamp(ctrl.min, ctrl.max);
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// FlameRigParams
// ---------------------------------------------------------------------------

/// Simplified FLAME parameter struct used by the rig system.
///
/// Decoupled from the full `FlameParams` to avoid circular dependency and to
/// carry only the fields relevant to rig output (expression and pose).
#[derive(Debug, Clone)]
pub struct FlameRigParams {
    /// Expression blend-shape coefficients.
    pub expression: Vec<f32>,
    /// Pose axis-angle parameters (root + neck + jaw + `left_eye` + `right_eye`).
    pub pose: Vec<f32>,
}

impl FlameRigParams {
    /// Create a neutral (all-zero) parameter set with the given dimensions.
    #[must_use]
    pub fn neutral(expression_dim: usize, pose_dim: usize) -> Self {
        Self {
            expression: vec![0.0; expression_dim],
            pose: vec![0.0; pose_dim],
        }
    }

    /// Add a delta vector to the expression parameters.
    ///
    /// # Errors
    ///
    /// Returns [`RigError::DimensionMismatch`] if `delta.len() != self.expression.len()`.
    pub fn add_expression_delta(&mut self, delta: &[f32]) -> Result<(), RigError> {
        if delta.len() != self.expression.len() {
            return Err(RigError::DimensionMismatch {
                expected: self.expression.len(),
                actual: delta.len(),
            });
        }
        for (e, d) in self.expression.iter_mut().zip(delta.iter()) {
            *e += d;
        }
        Ok(())
    }

    /// Add a delta vector to the pose parameters.
    ///
    /// # Errors
    ///
    /// Returns [`RigError::DimensionMismatch`] if `delta.len() != self.pose.len()`.
    pub fn add_pose_delta(&mut self, delta: &[f32]) -> Result<(), RigError> {
        if delta.len() != self.pose.len() {
            return Err(RigError::DimensionMismatch {
                expected: self.pose.len(),
                actual: delta.len(),
            });
        }
        for (p, d) in self.pose.iter_mut().zip(delta.iter()) {
            *p += d;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// apply_rig_to_params
// ---------------------------------------------------------------------------

/// Apply the rig's current evaluation to a set of base FLAME rig parameters.
///
/// Returns a new `FlameRigParams` with expression and pose deltas added.
/// The base parameters are not mutated.
///
/// # Errors
///
/// Propagates any errors from [`AvatarRig::evaluate`] or the delta-add methods.
pub fn apply_rig_to_params(
    rig: &AvatarRig,
    base_params: &FlameRigParams,
) -> Result<FlameRigParams, RigError> {
    let (expr_delta, pose_delta) = rig.evaluate()?;

    let mut result = base_params.clone();

    // Expression delta may be shorter/longer than base; use min-length addition
    for (base_val, &delta_val) in result.expression.iter_mut().zip(expr_delta.iter()) {
        *base_val += delta_val;
    }

    // Pose delta
    for (base_val, &delta_val) in result.pose.iter_mut().zip(pose_delta.iter()) {
        *base_val += delta_val;
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// RigKeyframe / RigAnimation
// ---------------------------------------------------------------------------

/// A single keyframe in a rig animation: a time stamp and a set of control values.
#[derive(Debug, Clone)]
pub struct RigKeyframe {
    /// Normalized time in `[0, 1]`.
    pub time: f32,
    /// Control values as `(name, value)` pairs.
    pub control_values: Vec<(String, f32)>,
}

/// A keyframe animation built on an `AvatarRig`.
///
/// Keyframes are kept sorted by time. Evaluating at any time `t` interpolates
/// between the two surrounding keyframes.
pub struct RigAnimation {
    /// The underlying rig whose controls are animated.
    pub rig: AvatarRig,
    /// Sorted list of keyframes.
    pub keyframes: Vec<RigKeyframe>,
}

impl RigAnimation {
    /// Create a new animation wrapping the given rig with no keyframes.
    #[must_use]
    pub fn new(rig: AvatarRig) -> Self {
        Self {
            rig,
            keyframes: Vec::new(),
        }
    }

    /// Add a keyframe, keeping the list sorted by time.
    ///
    /// # Errors
    ///
    /// Returns [`RigError::InvalidConfig`] if `keyframe.time` is non-finite.
    pub fn add_keyframe(&mut self, keyframe: RigKeyframe) -> Result<(), RigError> {
        if !keyframe.time.is_finite() {
            return Err(RigError::InvalidConfig(format!(
                "keyframe time must be finite, got {}",
                keyframe.time
            )));
        }
        // Insert sorted by time
        let pos = self.keyframes.partition_point(|k| k.time <= keyframe.time);
        self.keyframes.insert(pos, keyframe);
        Ok(())
    }

    /// Return the number of keyframes.
    #[must_use]
    pub fn n_keyframes(&self) -> usize {
        self.keyframes.len()
    }

    /// Evaluate the animation at normalized time `t` in `[0, 1]`.
    ///
    /// Applies the interpolated control values to the internal rig and returns
    /// `(expression_deltas, pose_deltas)`.
    ///
    /// - If there are no keyframes, returns [`RigError::EmptyRig`].
    /// - `t <= first_keyframe.time` uses the first keyframe.
    /// - `t >= last_keyframe.time` uses the last keyframe.
    /// - Otherwise linearly interpolates between the two surrounding keyframes.
    ///
    /// # Errors
    ///
    /// Returns errors from rig evaluation or keyframe lookup.
    pub fn evaluate_at(&mut self, t: f32) -> Result<(Vec<f32>, Vec<f32>), RigError> {
        if self.keyframes.is_empty() {
            return Err(RigError::EmptyRig);
        }

        // First keyframe boundary
        if t <= self.keyframes[0].time {
            apply_keyframe_to_rig(&mut self.rig, &self.keyframes[0])?;
            return self.rig.evaluate();
        }

        // Last keyframe boundary
        let last_idx = self.keyframes.len() - 1;
        if t >= self.keyframes[last_idx].time {
            let kf = self.keyframes[last_idx].clone();
            apply_keyframe_to_rig(&mut self.rig, &kf)?;
            return self.rig.evaluate();
        }

        // Find surrounding keyframes
        let next_idx = self.keyframes.partition_point(|k| k.time <= t);
        let prev_idx = next_idx - 1;

        let kf_prev = self.keyframes[prev_idx].clone();
        let kf_next = self.keyframes[next_idx].clone();

        let span = kf_next.time - kf_prev.time;
        let local_t = if span.abs() < f32::EPSILON {
            0.0
        } else {
            (t - kf_prev.time) / span
        };

        // Apply interpolated values to rig
        apply_keyframe_to_rig(&mut self.rig, &kf_prev)?;
        // Lerp each control toward next keyframe value
        for (name, next_val) in &kf_next.control_values {
            let prev_val = self.rig.get(name).unwrap_or(0.0);
            let lerped = prev_val + (next_val - prev_val) * local_t;
            self.rig.set(name, lerped)?;
        }

        self.rig.evaluate()
    }
}

/// Internal helper: apply all control values from a keyframe to a rig.
fn apply_keyframe_to_rig(rig: &mut AvatarRig, keyframe: &RigKeyframe) -> Result<(), RigError> {
    for (name, value) in &keyframe.control_values {
        if rig.get_control(name).is_some() {
            rig.set(name, *value)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// RigStats / compute_rig_stats
// ---------------------------------------------------------------------------

/// Statistical summary of the current rig state.
#[derive(Debug, Clone)]
pub struct RigStats {
    /// Total number of controls.
    pub n_controls: usize,
    /// Number of controls whose value differs from their default.
    pub n_active: usize,
    /// Name of the control with the largest deviation from default (`|value - default|`).
    pub most_active: Option<String>,
    /// L2 norm of the expression parameter deltas.
    pub expression_magnitude: f32,
    /// L2 norm of the pose parameter deltas.
    pub pose_magnitude: f32,
}

/// Compute statistics for the current rig state.
///
/// # Errors
///
/// Propagates errors from [`AvatarRig::evaluate`].
pub fn compute_rig_stats(rig: &AvatarRig) -> Result<RigStats, RigError> {
    let n_controls = rig.n_controls();

    let mut n_active = 0_usize;
    let mut max_deviation = 0.0_f32;
    let mut most_active: Option<String> = None;

    for ctrl in &rig.controls {
        let deviation = (ctrl.value - ctrl.default).abs();
        if deviation > f32::EPSILON {
            n_active += 1;
        }
        if deviation > max_deviation {
            max_deviation = deviation;
            most_active = Some(ctrl.name.clone());
        }
    }

    // Only report most_active if there is genuinely an active control
    if max_deviation <= f32::EPSILON {
        most_active = None;
    }

    let (expr_delta, pose_delta) = rig.evaluate()?;

    let expression_magnitude = expr_delta.iter().map(|x| x * x).sum::<f32>().sqrt();
    let pose_magnitude = pose_delta.iter().map(|x| x * x).sum::<f32>().sqrt();

    Ok(RigStats {
        n_controls,
        n_active,
        most_active,
        expression_magnitude,
        pose_magnitude,
    })
}

// ---------------------------------------------------------------------------
// generate_talking_animation
// ---------------------------------------------------------------------------

/// Generate a simple "talking" animation that oscillates the `mouth_open` control.
///
/// Uses a deterministic splitmix64 PRNG seeded with `seed` to produce varied
/// jaw-open amplitudes and timing, avoiding any dependency on the `rand` crate.
///
/// # Arguments
///
/// - `rig`: The base rig (must contain a `mouth_open` control).
/// - `n_frames`: Number of keyframes to generate.
/// - `mouth_open_amp`: Peak amplitude for jaw-open motion in `[0, 1]`.
/// - `seed`: PRNG seed for reproducible variation.
///
/// # Errors
///
/// - [`RigError::InvalidConfig`] if `n_frames == 0` or `rig` lacks `mouth_open`.
/// - [`RigError::InvalidControlValue`] if `mouth_open_amp` is non-finite.
pub fn generate_talking_animation(
    rig: AvatarRig,
    n_frames: usize,
    mouth_open_amp: f32,
    seed: u64,
) -> Result<RigAnimation, RigError> {
    if n_frames == 0 {
        return Err(RigError::InvalidConfig("n_frames must be > 0".to_owned()));
    }
    if !mouth_open_amp.is_finite() {
        return Err(RigError::InvalidControlValue {
            control: "mouth_open_amp".to_owned(),
            value: mouth_open_amp,
            min: 0.0,
            max: 1.0,
        });
    }
    if rig.get_control("mouth_open").is_none() {
        return Err(RigError::UnknownControl("mouth_open".to_owned()));
    }

    let amp = mouth_open_amp.clamp(0.0, 1.0);
    let mut animation = RigAnimation::new(rig);

    let mut prng_state = seed.wrapping_add(1);

    for frame_idx in 0..n_frames {
        let t = if n_frames == 1 {
            0.0
        } else {
            frame_idx as f32 / (n_frames - 1) as f32
        };

        // splitmix64 step
        prng_state = splitmix64_step(prng_state);
        let random_fraction = (prng_state >> 11) as f32 / (1u64 << 53) as f32;

        // Sinusoidal base + slight random variation
        let base_open = (t * std::f32::consts::PI * 4.0).sin().abs();
        let jaw_value = (base_open * amp * (0.7 + 0.3 * random_fraction)).clamp(0.0, 1.0);

        let keyframe = RigKeyframe {
            time: t,
            control_values: vec![("mouth_open".to_owned(), jaw_value)],
        };
        animation.add_keyframe(keyframe)?;
    }

    Ok(animation)
}

/// One step of the splitmix64 algorithm — a fast, high-quality 64-bit PRNG.
#[inline]
fn splitmix64_step(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // RigControl tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rig_control_new_defaults() {
        let ctrl = RigControl::new("smile", -1.0, 1.0, 0.0);
        assert_eq!(ctrl.name, "smile");
        assert!((ctrl.value - 0.0).abs() < 1e-6);
        assert!((ctrl.min - (-1.0)).abs() < 1e-6);
        assert!((ctrl.max - 1.0).abs() < 1e-6);
        assert!((ctrl.default - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rig_control_default_value_clamped() {
        // Default outside range should be clamped on construction
        let ctrl = RigControl::new("test", 0.0, 1.0, 2.0);
        assert!((ctrl.value - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rig_control_set_value_clamped_no_error() {
        let mut ctrl = RigControl::new("smile", -1.0, 1.0, 0.0);
        // Out of range → clamped but returns Ok
        ctrl.set_value(5.0)
            .expect("set_value should not error for finite values");
        assert!((ctrl.value - 1.0).abs() < 1e-6);
        ctrl.set_value(-5.0)
            .expect("set_value should not error for finite values");
        assert!((ctrl.value - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_rig_control_set_value_nan_error() {
        let mut ctrl = RigControl::new("test", -1.0, 1.0, 0.0);
        let result = ctrl.set_value(f32::NAN);
        assert!(result.is_err());
        assert!(matches!(result, Err(RigError::InvalidControlValue { .. })));
    }

    #[test]
    fn test_rig_control_set_value_inf_error() {
        let mut ctrl = RigControl::new("test", -1.0, 1.0, 0.0);
        let result = ctrl.set_value(f32::INFINITY);
        assert!(result.is_err());
    }

    #[test]
    fn test_rig_control_normalized_value() {
        let mut ctrl = RigControl::new("x", 0.0, 2.0, 0.0);
        ctrl.set_value(1.0).expect("set_value");
        let norm = ctrl.normalized_value();
        assert!((norm - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_rig_control_normalized_value_at_min() {
        let ctrl = RigControl::new("x", -1.0, 1.0, -1.0);
        let norm = ctrl.normalized_value();
        assert!(norm.abs() < 1e-5);
    }

    #[test]
    fn test_rig_control_normalized_value_at_max() {
        let mut ctrl = RigControl::new("x", -1.0, 1.0, 0.0);
        ctrl.set_value(1.0).expect("set_value");
        let norm = ctrl.normalized_value();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rig_control_normalized_value_zero_range() {
        let ctrl = RigControl::new("x", 0.5, 0.5, 0.5);
        let norm = ctrl.normalized_value();
        assert!(norm.abs() < 1e-5);
    }

    #[test]
    fn test_rig_control_reset() {
        let mut ctrl = RigControl::new("smile", -1.0, 1.0, 0.0);
        ctrl.set_value(0.8).expect("set_value");
        assert!((ctrl.value - 0.8).abs() < 1e-5);
        ctrl.reset();
        assert!(ctrl.value.abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // AvatarRig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_avatar_rig_new() {
        let rig = AvatarRig::new(50, 15);
        assert_eq!(rig.expression_dim, 50);
        assert_eq!(rig.pose_dim, 15);
        assert_eq!(rig.n_controls(), 0);
    }

    #[test]
    fn test_avatar_rig_add_control() {
        let mut rig = AvatarRig::new(50, 15);
        let ctrl = RigControl::new("smile", -1.0, 1.0, 0.0);
        rig.add_control(ctrl).expect("add_control");
        assert_eq!(rig.n_controls(), 1);
    }

    #[test]
    fn test_avatar_rig_add_duplicate_control_error() {
        let mut rig = AvatarRig::new(50, 15);
        rig.add_control(RigControl::new("smile", -1.0, 1.0, 0.0))
            .expect("first add");
        let result = rig.add_control(RigControl::new("smile", -1.0, 1.0, 0.0));
        assert!(matches!(result, Err(RigError::DuplicateControl(_))));
    }

    #[test]
    fn test_avatar_rig_get_control() {
        let mut rig = AvatarRig::new(50, 15);
        rig.add_control(RigControl::new("smile", -1.0, 1.0, 0.0))
            .expect("add");
        assert!(rig.get_control("smile").is_some());
        assert!(rig.get_control("nonexistent").is_none());
    }

    #[test]
    fn test_avatar_rig_get_control_mut() {
        let mut rig = AvatarRig::new(50, 15);
        rig.add_control(RigControl::new("smile", -1.0, 1.0, 0.0))
            .expect("add");
        let ctrl = rig.get_control_mut("smile").expect("get_control_mut");
        ctrl.set_value(0.5).expect("set");
        assert!((ctrl.value - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_avatar_rig_set_get() {
        let mut rig = AvatarRig::new(50, 15);
        rig.add_control(RigControl::new("smile", -1.0, 1.0, 0.0))
            .expect("add");
        rig.set("smile", 0.7).expect("set");
        let v = rig.get("smile").expect("get");
        assert!((v - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_avatar_rig_set_unknown_error() {
        let mut rig = AvatarRig::new(50, 15);
        rig.add_control(RigControl::new("smile", -1.0, 1.0, 0.0))
            .expect("add");
        let result = rig.set("brow_raise", 0.5);
        assert!(matches!(result, Err(RigError::UnknownControl(_))));
    }

    #[test]
    fn test_avatar_rig_get_unknown_error() {
        let rig = AvatarRig::new(50, 15);
        let result = rig.get("nonexistent");
        assert!(matches!(result, Err(RigError::UnknownControl(_))));
    }

    #[test]
    fn test_avatar_rig_reset_all() {
        let mut rig = AvatarRig::new(50, 15);
        rig.add_control(RigControl::new("smile", -1.0, 1.0, 0.0))
            .expect("add");
        rig.add_control(RigControl::new("brow_raise", -1.0, 1.0, 0.0))
            .expect("add");
        rig.set("smile", 0.8).expect("set");
        rig.set("brow_raise", -0.5).expect("set");
        rig.reset_all();
        assert!(rig.get("smile").expect("get").abs() < 1e-5);
        assert!(rig.get("brow_raise").expect("get").abs() < 1e-5);
    }

    #[test]
    fn test_avatar_rig_add_blend_target_valid() {
        let mut rig = AvatarRig::new(50, 15);
        rig.add_control(RigControl::new("smile", -1.0, 1.0, 0.0))
            .expect("add");
        let target = BlendTarget {
            control_name: "smile".to_owned(),
            expression_indices: vec![0, 1],
            weights: vec![0.5, -0.2],
            pose_indices: vec![],
            pose_weights: vec![],
        };
        rig.add_blend_target(target).expect("add_blend_target");
        assert_eq!(rig.blend_targets.len(), 1);
    }

    #[test]
    fn test_avatar_rig_add_blend_target_mismatch_error() {
        let mut rig = AvatarRig::new(50, 15);
        let target = BlendTarget {
            control_name: "smile".to_owned(),
            expression_indices: vec![0, 1, 2],
            weights: vec![0.5], // length mismatch
            pose_indices: vec![],
            pose_weights: vec![],
        };
        let result = rig.add_blend_target(target);
        assert!(matches!(result, Err(RigError::InvalidConfig(_))));
    }

    #[test]
    fn test_avatar_rig_evaluate_neutral_zero_deltas() {
        let mut rig = AvatarRig::new(10, 15);
        rig.add_control(RigControl::new("smile", -1.0, 1.0, 0.0))
            .expect("add");
        let target = BlendTarget {
            control_name: "smile".to_owned(),
            expression_indices: vec![0, 1],
            weights: vec![0.5, -0.2],
            pose_indices: vec![],
            pose_weights: vec![],
        };
        rig.add_blend_target(target).expect("add_blend_target");

        // Neutral (smile = 0.0) → all deltas zero
        let (expr, pose) = rig.evaluate().expect("evaluate");
        assert!(expr.iter().all(|&x| x.abs() < 1e-6));
        assert!(pose.iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn test_avatar_rig_evaluate_smile_nonzero() {
        let mut rig = AvatarRig::new(10, 15);
        rig.add_control(RigControl::new("smile", -1.0, 1.0, 0.0))
            .expect("add");
        let target = BlendTarget {
            control_name: "smile".to_owned(),
            expression_indices: vec![0, 1],
            weights: vec![0.5, -0.2],
            pose_indices: vec![],
            pose_weights: vec![],
        };
        rig.add_blend_target(target).expect("add_blend_target");
        rig.set("smile", 1.0).expect("set");

        let (expr, _pose) = rig.evaluate().expect("evaluate");
        assert!((expr[0] - 0.5).abs() < 1e-5);
        assert!((expr[1] - (-0.2)).abs() < 1e-5);
    }

    #[test]
    fn test_avatar_rig_evaluate_empty_rig_error() {
        let rig = AvatarRig::new(10, 15);
        let result = rig.evaluate();
        assert!(matches!(result, Err(RigError::EmptyRig)));
    }

    // -----------------------------------------------------------------------
    // standard_face_rig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_standard_face_rig_has_12_controls() {
        let rig = standard_face_rig(50);
        assert_eq!(
            rig.n_controls(),
            12,
            "Standard rig should have exactly 12 controls"
        );
    }

    #[test]
    fn test_standard_face_rig_all_controls_present() {
        let rig = standard_face_rig(50);
        let expected = [
            "smile",
            "mouth_open",
            "brow_raise",
            "brow_furrow",
            "eye_squint",
            "cheek_puff",
            "lip_pucker",
            "head_tilt",
            "head_nod",
            "head_turn",
            "eye_look_left",
            "eye_look_up",
        ];
        for name in &expected {
            assert!(rig.get_control(name).is_some(), "Missing control: {name}");
        }
    }

    #[test]
    fn test_standard_face_rig_small_expr_dim() {
        // With expression_dim=3, some targets will have fewer indices
        let rig = standard_face_rig(3);
        assert_eq!(rig.n_controls(), 12);
        // Should still evaluate without errors
        let _result = rig.evaluate().expect("evaluate with small expr_dim");
    }

    // -----------------------------------------------------------------------
    // interpolate_rigs tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpolate_rigs_t0_equals_rig_a() {
        let mut rig_a = AvatarRig::new(10, 15);
        rig_a
            .add_control(RigControl::new("smile", -1.0, 1.0, 0.0))
            .expect("add");
        rig_a.set("smile", 0.8).expect("set");

        let mut rig_b = AvatarRig::new(10, 15);
        rig_b
            .add_control(RigControl::new("smile", -1.0, 1.0, 0.0))
            .expect("add");
        rig_b.set("smile", -0.4).expect("set");

        let result = interpolate_rigs(&rig_a, &rig_b, 0.0).expect("interpolate");
        let val = result.get("smile").expect("get");
        assert!((val - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_interpolate_rigs_t1_equals_rig_b() {
        let mut rig_a = AvatarRig::new(10, 15);
        rig_a
            .add_control(RigControl::new("smile", -1.0, 1.0, 0.0))
            .expect("add");
        rig_a.set("smile", 0.8).expect("set");

        let mut rig_b = AvatarRig::new(10, 15);
        rig_b
            .add_control(RigControl::new("smile", -1.0, 1.0, 0.0))
            .expect("add");
        rig_b.set("smile", -0.4).expect("set");

        let result = interpolate_rigs(&rig_a, &rig_b, 1.0).expect("interpolate");
        let val = result.get("smile").expect("get");
        assert!((val - (-0.4)).abs() < 1e-5);
    }

    #[test]
    fn test_interpolate_rigs_midpoint() {
        let mut rig_a = AvatarRig::new(10, 15);
        rig_a
            .add_control(RigControl::new("smile", -1.0, 1.0, 0.0))
            .expect("add");
        rig_a.set("smile", 0.0).expect("set");

        let mut rig_b = AvatarRig::new(10, 15);
        rig_b
            .add_control(RigControl::new("smile", -1.0, 1.0, 0.0))
            .expect("add");
        rig_b.set("smile", 1.0).expect("set");

        let result = interpolate_rigs(&rig_a, &rig_b, 0.5).expect("interpolate");
        let val = result.get("smile").expect("get");
        assert!((val - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_interpolate_rigs_nan_error() {
        let rig_a = AvatarRig::new(10, 15);
        let rig_b = AvatarRig::new(10, 15);
        let result = interpolate_rigs(&rig_a, &rig_b, f32::NAN);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // FlameRigParams tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_flame_rig_params_neutral_all_zeros() {
        let p = FlameRigParams::neutral(50, 15);
        assert_eq!(p.expression.len(), 50);
        assert_eq!(p.pose.len(), 15);
        assert!(p.expression.iter().all(|&x| x == 0.0));
        assert!(p.pose.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_flame_rig_params_add_expression_delta() {
        let mut p = FlameRigParams::neutral(5, 15);
        let delta = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        p.add_expression_delta(&delta)
            .expect("add_expression_delta");
        for (i, &expected) in delta.iter().enumerate() {
            assert!((p.expression[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_flame_rig_params_add_expression_delta_mismatch_error() {
        let mut p = FlameRigParams::neutral(5, 15);
        let delta = vec![0.1, 0.2]; // wrong length
        let result = p.add_expression_delta(&delta);
        assert!(matches!(result, Err(RigError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_flame_rig_params_add_pose_delta() {
        let mut p = FlameRigParams::neutral(50, 3);
        let delta = vec![0.1, 0.2, 0.3];
        p.add_pose_delta(&delta).expect("add_pose_delta");
        for (i, &expected) in delta.iter().enumerate() {
            assert!((p.pose[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_flame_rig_params_add_pose_delta_mismatch_error() {
        let mut p = FlameRigParams::neutral(50, 15);
        let delta = vec![0.1]; // wrong length
        let result = p.add_pose_delta(&delta);
        assert!(matches!(result, Err(RigError::DimensionMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // apply_rig_to_params tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_rig_neutral_params_unchanged() {
        let rig = standard_face_rig(50);
        let base = FlameRigParams::neutral(50, 15);
        let result = apply_rig_to_params(&rig, &base).expect("apply_rig_to_params");
        // All controls at default (0) → all zeros
        assert!(result.expression.iter().all(|&x| x.abs() < 1e-6));
        assert!(result.pose.iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn test_apply_rig_smile_modifies_expression() {
        let mut rig = standard_face_rig(50);
        rig.set("smile", 1.0).expect("set smile");
        let base = FlameRigParams::neutral(50, 15);
        let result = apply_rig_to_params(&rig, &base).expect("apply_rig_to_params");
        // e[0] should be positive with smile=1.0
        assert!(result.expression[0] > 0.0);
    }

    #[test]
    fn test_apply_rig_mouth_open_modifies_pose() {
        let mut rig = standard_face_rig(50);
        rig.set("mouth_open", 1.0).expect("set mouth_open");
        let base = FlameRigParams::neutral(50, 15);
        let result = apply_rig_to_params(&rig, &base).expect("apply_rig_to_params");
        // jaw x-axis (index 6) should be > 0
        assert!(result.pose[6] > 0.0);
    }

    // -----------------------------------------------------------------------
    // RigAnimation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rig_animation_new() {
        let rig = standard_face_rig(50);
        let anim = RigAnimation::new(rig);
        assert_eq!(anim.n_keyframes(), 0);
    }

    #[test]
    fn test_rig_animation_add_keyframe() {
        let rig = standard_face_rig(50);
        let mut anim = RigAnimation::new(rig);
        let kf = RigKeyframe {
            time: 0.0,
            control_values: vec![("smile".to_owned(), 0.5)],
        };
        anim.add_keyframe(kf).expect("add_keyframe");
        assert_eq!(anim.n_keyframes(), 1);
    }

    #[test]
    fn test_rig_animation_n_keyframes() {
        let rig = standard_face_rig(50);
        let mut anim = RigAnimation::new(rig);
        for i in 0..5 {
            let kf = RigKeyframe {
                time: i as f32 / 4.0,
                control_values: vec![],
            };
            anim.add_keyframe(kf).expect("add_keyframe");
        }
        assert_eq!(anim.n_keyframes(), 5);
    }

    #[test]
    fn test_rig_animation_add_keyframe_invalid_time() {
        let rig = standard_face_rig(50);
        let mut anim = RigAnimation::new(rig);
        let kf = RigKeyframe {
            time: f32::NAN,
            control_values: vec![],
        };
        let result = anim.add_keyframe(kf);
        assert!(result.is_err());
    }

    #[test]
    fn test_rig_animation_evaluate_at_t0_uses_first_keyframe() {
        let rig = standard_face_rig(50);
        let mut anim = RigAnimation::new(rig);

        anim.add_keyframe(RigKeyframe {
            time: 0.0,
            control_values: vec![("smile".to_owned(), 1.0)],
        })
        .expect("add kf 0");
        anim.add_keyframe(RigKeyframe {
            time: 1.0,
            control_values: vec![("smile".to_owned(), -1.0)],
        })
        .expect("add kf 1");

        let (expr, _) = anim.evaluate_at(0.0).expect("evaluate_at 0");
        // smile=1.0 → e[0] should be positive
        assert!(expr[0] > 0.0);
    }

    #[test]
    fn test_rig_animation_evaluate_at_t1_uses_last_keyframe() {
        let rig = standard_face_rig(50);
        let mut anim = RigAnimation::new(rig);

        anim.add_keyframe(RigKeyframe {
            time: 0.0,
            control_values: vec![("smile".to_owned(), 1.0)],
        })
        .expect("add kf 0");
        anim.add_keyframe(RigKeyframe {
            time: 1.0,
            control_values: vec![("smile".to_owned(), -1.0)],
        })
        .expect("add kf 1");

        let (expr, _) = anim.evaluate_at(1.0).expect("evaluate_at 1");
        // smile=-1.0 → e[0] should be negative
        assert!(expr[0] < 0.0);
    }

    #[test]
    fn test_rig_animation_evaluate_empty_error() {
        let rig = standard_face_rig(50);
        let mut anim = RigAnimation::new(rig);
        let result = anim.evaluate_at(0.5);
        assert!(matches!(result, Err(RigError::EmptyRig)));
    }

    // -----------------------------------------------------------------------
    // RigStats / compute_rig_stats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_rig_stats_neutral() {
        let rig = standard_face_rig(50);
        let stats = compute_rig_stats(&rig).expect("compute_rig_stats");
        assert_eq!(stats.n_controls, 12);
        assert_eq!(stats.n_active, 0);
        assert!(stats.most_active.is_none());
        assert!(stats.expression_magnitude.abs() < 1e-6);
        assert!(stats.pose_magnitude.abs() < 1e-6);
    }

    #[test]
    fn test_compute_rig_stats_active() {
        let mut rig = standard_face_rig(50);
        rig.set("smile", 0.9).expect("set");
        rig.set("brow_raise", 0.5).expect("set");
        let stats = compute_rig_stats(&rig).expect("compute_rig_stats");
        assert!(stats.n_active >= 2);
        assert!(stats.most_active.is_some());
        assert!(stats.expression_magnitude > 0.0);
    }

    #[test]
    fn test_compute_rig_stats_most_active_identifies_largest() {
        let mut rig = standard_face_rig(50);
        rig.set("smile", 0.3).expect("set smile");
        rig.set("brow_raise", 0.9).expect("set brow_raise");
        let stats = compute_rig_stats(&rig).expect("stats");
        // brow_raise has larger deviation from 0.0
        assert_eq!(stats.most_active.as_deref(), Some("brow_raise"));
    }

    // -----------------------------------------------------------------------
    // generate_talking_animation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_talking_animation_correct_n_keyframes() {
        let rig = standard_face_rig(50);
        let anim = generate_talking_animation(rig, 10, 0.7, 42).expect("generate");
        assert_eq!(anim.n_keyframes(), 10);
    }

    #[test]
    fn test_generate_talking_animation_jaw_control_present() {
        let rig = standard_face_rig(50);
        let anim = generate_talking_animation(rig, 5, 0.5, 1).expect("generate");
        // All keyframes should target mouth_open
        for kf in &anim.keyframes {
            let has_mouth = kf.control_values.iter().any(|(n, _)| n == "mouth_open");
            assert!(has_mouth, "keyframe should contain mouth_open");
        }
    }

    #[test]
    fn test_generate_talking_animation_zero_frames_error() {
        let rig = standard_face_rig(50);
        let result = generate_talking_animation(rig, 0, 0.5, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_talking_animation_values_in_range() {
        let rig = standard_face_rig(50);
        let anim = generate_talking_animation(rig, 20, 0.8, 99).expect("generate");
        for kf in &anim.keyframes {
            for (_, &v) in kf.control_values.iter().map(|(n, v)| (n, v)) {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "mouth_open value {v} out of range [0,1]"
                );
            }
        }
    }

    #[test]
    fn test_generate_talking_animation_deterministic() {
        let rig_a = standard_face_rig(50);
        let rig_b = standard_face_rig(50);
        let anim_a = generate_talking_animation(rig_a, 8, 0.6, 777).expect("gen a");
        let anim_b = generate_talking_animation(rig_b, 8, 0.6, 777).expect("gen b");
        for (kf_a, kf_b) in anim_a.keyframes.iter().zip(anim_b.keyframes.iter()) {
            assert!((kf_a.time - kf_b.time).abs() < 1e-6);
            for ((_, va), (_, vb)) in kf_a.control_values.iter().zip(kf_b.control_values.iter()) {
                assert!((va - vb).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_generate_talking_animation_missing_control_error() {
        // Rig without mouth_open
        let mut rig = AvatarRig::new(10, 15);
        rig.add_control(RigControl::new("smile", -1.0, 1.0, 0.0))
            .expect("add");
        let result = generate_talking_animation(rig, 5, 0.5, 1);
        assert!(matches!(result, Err(RigError::UnknownControl(_))));
    }
}
