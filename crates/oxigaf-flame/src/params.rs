//! FLAME model parameters for a single frame / pose.
//!
//! ## Parameter Ranges
//!
//! FLAME parameters typically lie in specific ranges for realistic results:
//!
//! - **Shape**: Each coefficient typically in range `[-3.0, 3.0]` (3 standard deviations)
//! - **Expression**: Each coefficient typically in range `[-2.0, 2.0]`
//! - **Pose** (axis-angle): Each component typically in range `[-pi, pi]` radians
//!   - Root rotation: Full 3D rotation of the head
//!   - Neck: Limited range for natural neck movement
//!   - Jaw: Typically `[-0.5, 0.2]` radians (opening range)
//!   - Eyes: Typically `[-0.3, 0.3]` radians per axis
//! - **Translation**: In meters, typically `[-1.0, 1.0]` for each axis
//!
//! ## Example
//!
//! ```rust
//! use oxigaf_flame::FlameParams;
//!
//! // Neutral face (zero deformation)
//! let neutral = FlameParams::neutral();
//!
//! // Smiling face with slight head tilt
//! let smiling = FlameParams {
//!     shape: vec![0.0; 10],  // Use neutral identity shape
//!     expression: vec![0.5, 0.3, -0.2],  // Smile expression
//!     pose: vec![0.1, 0.0, 0.0,  // Slight head tilt
//!                0.0, 0.0, 0.0,  // No neck rotation
//!                0.1, 0.0, 0.0,  // Slight jaw opening
//!                0.0, 0.0, 0.0,  // Neutral left eye
//!                0.0, 0.0, 0.0], // Neutral right eye
//!     translation: [0.0, 0.0, 0.0],
//! };
//! ```

use crate::error::FlameError;
use crate::params_builder::FlameParamsBuilder;
use serde::{Deserialize, Serialize};

/// FLAME model parameters for a single frame.
///
/// All vectors can be shorter than the maximum; missing trailing coefficients
/// are treated as zero during the forward pass.
///
/// See module-level documentation for parameter ranges and examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameParams {
    /// Shape (identity) blend-shape coefficients. Up to 300, typically 100.
    pub shape: Vec<f32>,

    /// Expression blend-shape coefficients. Up to 100, typically 50.
    pub expression: Vec<f32>,

    /// Joint pose as axis-angle vectors, concatenated.
    /// 5 joints x 3 = 15 values.
    /// Order: `[root(3), neck(3), jaw(3), left_eye(3), right_eye(3)]`
    pub pose: Vec<f32>,

    /// Global translation applied after posing.
    pub translation: [f32; 3],
}

impl FlameParams {
    /// Number of FLAME joints.
    pub const NUM_JOINTS: usize = 5;

    /// Create a neutral (zero) parameter set.
    #[must_use]
    pub fn neutral() -> Self {
        Self {
            shape: Vec::new(),
            expression: Vec::new(),
            pose: vec![0.0; Self::NUM_JOINTS * 3],
            translation: [0.0; 3],
        }
    }

    /// Start building `FlameParams` with a builder pattern.
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxigaf_flame::FlameParams;
    ///
    /// let params = FlameParams::builder()
    ///     .shape(vec![0.5, -0.3, 0.2])
    ///     .expression(vec![0.8, 0.5])
    ///     .jaw_rotation(0.1)
    ///     .translation([0.0, 0.1, 0.0])
    ///     .build();
    /// ```
    #[must_use]
    pub fn builder() -> FlameParamsBuilder {
        FlameParamsBuilder::default()
    }

    /// Return the axis-angle triple for joint `j` (0-indexed), or zeros if
    /// the pose vector is too short.
    #[must_use]
    pub fn joint_pose(&self, j: usize) -> [f32; 3] {
        let off = j * 3;
        if off + 2 < self.pose.len() {
            [self.pose[off], self.pose[off + 1], self.pose[off + 2]]
        } else {
            [0.0; 3]
        }
    }

    /// Validate that parameters are within typical ranges.
    ///
    /// Returns `true` if all parameters are finite AND within reasonable bounds:
    /// - Shape: [-3.0, 3.0]
    /// - Expression: [-2.0, 2.0]
    /// - Pose: [-pi, pi]
    /// - Translation: [-1.0, 1.0]
    ///
    /// `NaN` and `±inf` always fail validation: IEEE-754 comparisons against
    /// `NaN` are false, so a naive `.abs() > limit` check alone would let
    /// `NaN` silently pass.
    #[must_use]
    pub fn validate(&self) -> bool {
        use std::f32::consts::PI;

        // A value fails if it's non-finite (NaN or ±inf) OR outside [-lim, lim].
        let out_of_range = |v: &f32, lim: f32| !v.is_finite() || v.abs() > lim;

        // Check shape coefficients
        if self.shape.iter().any(|s| out_of_range(s, 3.0)) {
            return false;
        }

        // Check expression coefficients
        if self.expression.iter().any(|e| out_of_range(e, 2.0)) {
            return false;
        }

        // Check pose angles
        if self.pose.iter().any(|p| out_of_range(p, PI)) {
            return false;
        }

        // Check translation
        if self.translation.iter().any(|t| out_of_range(t, 1.0)) {
            return false;
        }

        true
    }

    /// Linearly interpolate between `self` and `other`.
    ///
    /// - `t = 0.0` returns a clone of `self`.
    /// - `t = 1.0` returns a clone of `other`.
    /// - `shape`, `expression`, and `translation` are interpolated element-wise (linear).
    /// - `pose` is interpolated using quaternion slerp (axis-angle → quaternion → slerp →
    ///   axis-angle) so that rotational paths stay on the rotation manifold.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if `self` and `other` have mismatched
    /// parameter vector lengths (shape, expression, or pose).
    pub fn lerp(&self, other: &Self, t: f32) -> Result<Self, FlameError> {
        // Validate dimensions before doing any work.
        if self.shape.len() != other.shape.len() {
            return Err(FlameError::InvalidParams(format!(
                "shape length mismatch: {} vs {}",
                self.shape.len(),
                other.shape.len()
            )));
        }
        if self.expression.len() != other.expression.len() {
            return Err(FlameError::InvalidParams(format!(
                "expression length mismatch: {} vs {}",
                self.expression.len(),
                other.expression.len()
            )));
        }
        if self.pose.len() != other.pose.len() {
            return Err(FlameError::InvalidParams(format!(
                "pose length mismatch: {} vs {}",
                self.pose.len(),
                other.pose.len()
            )));
        }
        if !self.pose.len().is_multiple_of(3) {
            return Err(FlameError::InvalidParams(format!(
                "pose length {} is not a multiple of 3 (axis-angle triples required)",
                self.pose.len()
            )));
        }
        // A NaN `t` slips past both boundary fast paths below (`NaN <= 0.0`
        // and `NaN >= 1.0` are both false), so without this guard it would
        // fall through to the interpolation arithmetic and silently
        // produce an all-NaN result. `±inf` is deliberately NOT rejected
        // here: `t <= 0.0` / `t >= 1.0` already handle it correctly
        // (`NEG_INFINITY <= 0.0` and `INFINITY >= 1.0` are both true), so
        // rejecting it too would only narrow previously-correct behavior.
        if t.is_nan() {
            return Err(FlameError::InvalidParams(
                "lerp t must not be NaN".to_string(),
            ));
        }

        // Fast paths for t at the boundary.
        if t <= 0.0 {
            return Ok(self.clone());
        }
        if t >= 1.0 {
            return Ok(other.clone());
        }

        // Linearly interpolate shape.
        let shape = self
            .shape
            .iter()
            .zip(other.shape.iter())
            .map(|(&a, &b)| a + (b - a) * t)
            .collect();

        // Linearly interpolate expression.
        let expression = self
            .expression
            .iter()
            .zip(other.expression.iter())
            .map(|(&a, &b)| a + (b - a) * t)
            .collect();

        // Spherically interpolate pose (axis-angle groups of 3).
        let pose = slerp_pose_flat(&self.pose, &other.pose, t);

        // Linearly interpolate translation.
        let translation = [
            self.translation[0] + (other.translation[0] - self.translation[0]) * t,
            self.translation[1] + (other.translation[1] - self.translation[1]) * t,
            self.translation[2] + (other.translation[2] - self.translation[2]) * t,
        ];

        Ok(Self {
            shape,
            expression,
            pose,
            translation,
        })
    }

    /// Interpolate only the `pose` field (axis-angle, grouped by 3) using quaternion slerp.
    ///
    /// This is a convenience wrapper around [`lerp`] that borrows only the pose slice
    /// and returns the interpolated flat `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if pose lengths differ or are not
    /// multiples of 3.
    ///
    /// [`lerp`]: FlameParams::lerp
    pub fn slerp_pose(&self, other: &Self, t: f32) -> Result<Vec<f32>, FlameError> {
        if self.pose.len() != other.pose.len() {
            return Err(FlameError::InvalidParams(format!(
                "pose length mismatch: {} vs {}",
                self.pose.len(),
                other.pose.len()
            )));
        }
        if !self.pose.len().is_multiple_of(3) {
            return Err(FlameError::InvalidParams(format!(
                "pose length {} is not a multiple of 3",
                self.pose.len()
            )));
        }
        Ok(slerp_pose_flat(&self.pose, &other.pose, t))
    }
}

// ---------------------------------------------------------------------------
// Private helpers — axis-angle ↔ quaternion conversion + slerp
// ---------------------------------------------------------------------------

/// Interpolate two flat pose vectors (each group of 3 = one axis-angle rotation)
/// using quaternion slerp.  Assumes `a.len() == b.len()` and divisible by 3.
fn slerp_pose_flat(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let n = a.len() / 3;
    let mut out = Vec::with_capacity(a.len());
    for i in 0..n {
        let aa: [f32; 3] = [a[i * 3], a[i * 3 + 1], a[i * 3 + 2]];
        let ab: [f32; 3] = [b[i * 3], b[i * 3 + 1], b[i * 3 + 2]];
        let qa = axis_angle_to_quat(aa);
        let qb = axis_angle_to_quat(ab);
        let qr = quat_slerp(qa, qb, t);
        let result = quat_to_axis_angle(qr);
        out.extend_from_slice(&result);
    }
    out
}

/// Convert an axis-angle vector (magnitude = rotation angle in radians) to a
/// unit quaternion `[x, y, z, w]`.
///
/// When the rotation angle is near zero the function returns the identity
/// quaternion `[0, 0, 0, 1]` to avoid division by zero.
fn axis_angle_to_quat(aa: [f32; 3]) -> [f32; 4] {
    const EPSILON: f32 = 1e-8;
    let angle_sq = aa[0] * aa[0] + aa[1] * aa[1] + aa[2] * aa[2];
    if angle_sq < EPSILON * EPSILON {
        // Identity rotation.
        return [0.0, 0.0, 0.0, 1.0];
    }
    let angle = angle_sq.sqrt();
    let inv = 1.0 / angle;
    let half = angle * 0.5;
    let s = half.sin() * inv;
    [aa[0] * s, aa[1] * s, aa[2] * s, half.cos()]
}

/// Spherically interpolate two unit quaternions `q1` and `q2` by parameter `t`.
///
/// Handles the antipodal case (dot < 0) by negating `q2` so the interpolation
/// takes the shorter arc.  Falls back to linear interpolation (normalised) when
/// the quaternions are very close to avoid numerical issues in `acos`.
fn quat_slerp(q1: [f32; 4], mut q2: [f32; 4], t: f32) -> [f32; 4] {
    const SLERP_THRESHOLD: f32 = 0.9995;

    let dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3];

    // Choose the shorter arc: if dot < 0 negate q2.
    let (mut dot, negate) = if dot < 0.0 {
        (-dot, true)
    } else {
        (dot, false)
    };
    if negate {
        q2 = [-q2[0], -q2[1], -q2[2], -q2[3]];
    }

    // Clamp to valid range for acos.
    dot = dot.min(1.0);

    // When quaternions are very close use normalised linear interpolation (nlerp)
    // to avoid precision loss in sin/acos near 0.
    if dot > SLERP_THRESHOLD {
        let rx = q1[0] + t * (q2[0] - q1[0]);
        let ry = q1[1] + t * (q2[1] - q1[1]);
        let rz = q1[2] + t * (q2[2] - q1[2]);
        let rw = q1[3] + t * (q2[3] - q1[3]);
        let norm_sq = rx * rx + ry * ry + rz * rz + rw * rw;
        if norm_sq < 1e-16 {
            // Degenerate; return identity.
            return [0.0, 0.0, 0.0, 1.0];
        }
        let inv = norm_sq.sqrt().recip();
        return [rx * inv, ry * inv, rz * inv, rw * inv];
    }

    let omega = dot.acos();
    let sin_omega = omega.sin();
    let s1 = ((1.0 - t) * omega).sin() / sin_omega;
    let s2 = (t * omega).sin() / sin_omega;

    [
        s1 * q1[0] + s2 * q2[0],
        s1 * q1[1] + s2 * q2[1],
        s1 * q1[2] + s2 * q2[2],
        s1 * q1[3] + s2 * q2[3],
    ]
}

/// Convert a unit quaternion `[x, y, z, w]` back to an axis-angle vector
/// (magnitude = rotation angle in radians).
///
/// When the rotation angle is near zero the function returns `[0, 0, 0]`.
///
/// The formula:
/// - `angle = 2 * acos(clamp(w, -1, 1))` — always in `[0, π]` for the
///   principal representation (the shortest arc).
/// - `axis  = xyz / sin(angle/2)` — requires `sin(angle/2) ≠ 0`.
/// - Result vector = `axis * angle` (magnitude encodes rotation amount).
fn quat_to_axis_angle(q: [f32; 4]) -> [f32; 3] {
    const EPSILON: f32 = 1e-8;

    // Normalise to the canonical hemisphere (w >= 0) so that the reconstructed
    // angle lies in [0, π].  The rotation q and -q are identical; choosing the
    // one with w >= 0 gives the shorter (principal) arc.
    let sign = if q[3] < 0.0 { -1.0f32 } else { 1.0f32 };
    let (qx, qy, qz, w_raw) = (q[0] * sign, q[1] * sign, q[2] * sign, q[3] * sign);

    // Clamp to [-1, 1] to guard against floating-point overshoot.
    let w = w_raw.clamp(-1.0, 1.0);

    // θ/2 = acos(w), angle = θ = 2·acos(w) ∈ [0, π].
    let half_angle = w.acos();
    let angle = 2.0 * half_angle;
    let sin_half = half_angle.sin();

    if sin_half.abs() < EPSILON {
        // Near-identity rotation: axis is indeterminate, return zero vector.
        return [0.0, 0.0, 0.0];
    }

    // axis = (qx, qy, qz) / sin(θ/2);  axis_angle = axis * θ
    let scale = angle / sin_half;
    [qx * scale, qy * scale, qz * scale]
}

impl Default for FlameParams {
    fn default() -> Self {
        Self::neutral()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    // ------ helper ------

    fn make_params(
        shape: Vec<f32>,
        expression: Vec<f32>,
        pose: Vec<f32>,
        t: [f32; 3],
    ) -> FlameParams {
        FlameParams {
            shape,
            expression,
            pose,
            translation: t,
        }
    }

    fn assert_f32_close(a: f32, b: f32, tol: f32, msg: &str) {
        assert!((a - b).abs() < tol, "{msg}: {a} vs {b} (tol {tol})");
    }

    fn assert_slice_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (ai - bi).abs() < tol,
                "{msg}: index {i} — {ai} vs {bi} (tol {tol})"
            );
        }
    }

    // ------ validate: NaN/inf must always fail ------

    #[test]
    fn validate_accepts_in_range_finite_params() {
        let p = make_params(vec![1.0], vec![0.5], vec![0.1, 0.0, 0.0], [0.1, 0.0, 0.0]);
        assert!(p.validate(), "in-range finite params must be valid");
    }

    #[test]
    fn validate_rejects_nan_shape() {
        let p = make_params(vec![f32::NAN], vec![], vec![], [0.0; 3]);
        assert!(!p.validate(), "NaN shape coefficient must fail validation");
    }

    #[test]
    fn validate_rejects_nan_expression() {
        let p = make_params(vec![], vec![f32::NAN], vec![], [0.0; 3]);
        assert!(
            !p.validate(),
            "NaN expression coefficient must fail validation"
        );
    }

    #[test]
    fn validate_rejects_nan_pose() {
        let p = make_params(vec![], vec![], vec![f32::NAN, 0.0, 0.0], [0.0; 3]);
        assert!(!p.validate(), "NaN pose component must fail validation");
    }

    #[test]
    fn validate_rejects_nan_translation() {
        let p = make_params(vec![], vec![], vec![], [f32::NAN, 0.0, 0.0]);
        assert!(
            !p.validate(),
            "NaN translation component must fail validation"
        );
    }

    #[test]
    fn validate_rejects_infinite_shape() {
        let p = make_params(vec![f32::INFINITY], vec![], vec![], [0.0; 3]);
        assert!(
            !p.validate(),
            "infinite shape coefficient must fail validation"
        );
    }

    // ------ lerp boundary conditions ------

    #[test]
    fn lerp_t0_returns_self() {
        let a = make_params(
            vec![1.0, 2.0],
            vec![0.5],
            vec![0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
        );
        let b = make_params(
            vec![3.0, 4.0],
            vec![1.5],
            vec![0.4, 0.5, 0.6],
            [0.4, 0.5, 0.6],
        );
        let r = a.lerp(&b, 0.0).expect("lerp failed");
        assert_slice_close(&r.shape, &a.shape, 1e-6, "shape");
        assert_slice_close(&r.expression, &a.expression, 1e-6, "expression");
        assert_slice_close(&r.translation, &a.translation, 1e-6, "translation");
    }

    #[test]
    fn lerp_t1_returns_other() {
        let a = make_params(
            vec![1.0, 2.0],
            vec![0.5],
            vec![0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
        );
        let b = make_params(
            vec![3.0, 4.0],
            vec![1.5],
            vec![0.4, 0.5, 0.6],
            [0.4, 0.5, 0.6],
        );
        let r = a.lerp(&b, 1.0).expect("lerp failed");
        assert_slice_close(&r.shape, &b.shape, 1e-6, "shape");
        assert_slice_close(&r.expression, &b.expression, 1e-6, "expression");
        assert_slice_close(&r.translation, &b.translation, 1e-6, "translation");
    }

    #[test]
    fn lerp_rejects_nan_t() {
        // A NaN `t` slips past both `t <= 0.0` and `t >= 1.0` (both false
        // for NaN), so without an explicit NaN guard it would fall through
        // to the interpolation arithmetic and produce a silently all-NaN
        // result instead of a diagnosable error.
        let a = make_params(vec![1.0], vec![0.5], vec![0.1, 0.2, 0.3], [0.1, 0.2, 0.3]);
        let b = make_params(vec![3.0], vec![1.5], vec![0.4, 0.5, 0.6], [0.4, 0.5, 0.6]);
        assert!(
            a.lerp(&b, f32::NAN).is_err(),
            "lerp with NaN t must return an error, not a NaN-filled result"
        );
    }

    #[test]
    fn lerp_infinite_t_still_clamps_to_boundary() {
        // Unlike NaN, ±inf is NOT rejected: `t <= 0.0` / `t >= 1.0` already
        // handle it correctly via the existing boundary fast paths, so
        // this behavior must be preserved exactly as before the NaN guard
        // was added.
        let a = make_params(vec![1.0], vec![0.5], vec![0.1, 0.2, 0.3], [0.1, 0.2, 0.3]);
        let b = make_params(vec![3.0], vec![1.5], vec![0.4, 0.5, 0.6], [0.4, 0.5, 0.6]);
        let r_pos = a
            .lerp(&b, f32::INFINITY)
            .expect("+inf must clamp, not error");
        assert_slice_close(&r_pos.shape, &b.shape, 1e-6, "shape (+inf -> other)");
        let r_neg = a
            .lerp(&b, f32::NEG_INFINITY)
            .expect("-inf must clamp, not error");
        assert_slice_close(&r_neg.shape, &a.shape, 1e-6, "shape (-inf -> self)");
    }

    // ------ midpoint for shape / expression / translation ------

    #[test]
    fn lerp_t05_midpoint_shape() {
        let a = make_params(vec![0.0, 0.0], vec![], vec![], [0.0, 0.0, 0.0]);
        let b = make_params(vec![2.0, 4.0], vec![], vec![], [0.0, 0.0, 0.0]);
        let r = a.lerp(&b, 0.5).expect("lerp failed");
        assert_slice_close(&r.shape, &[1.0, 2.0], 1e-6, "shape midpoint");
    }

    #[test]
    fn lerp_t05_midpoint_expression() {
        let a = make_params(vec![], vec![0.0, 0.0, 0.0], vec![], [0.0; 3]);
        let b = make_params(vec![], vec![1.0, 2.0, 3.0], vec![], [0.0; 3]);
        let r = a.lerp(&b, 0.5).expect("lerp failed");
        assert_slice_close(&r.expression, &[0.5, 1.0, 1.5], 1e-6, "expression midpoint");
    }

    #[test]
    fn lerp_t05_midpoint_translation() {
        let a = make_params(vec![], vec![], vec![], [0.0, 0.0, 0.0]);
        let b = make_params(vec![], vec![], vec![], [2.0, 4.0, 6.0]);
        let r = a.lerp(&b, 0.5).expect("lerp failed");
        assert_slice_close(
            &r.translation,
            &[1.0, 2.0, 3.0],
            1e-6,
            "translation midpoint",
        );
    }

    // ------ pose slerp: 90° around Z at t=0.5 → 45° ------

    #[test]
    fn pose_slerp_t05_half_rotation() {
        // 90° around Z axis = axis-angle [0, 0, π/2]
        let a = make_params(vec![], vec![], vec![0.0, 0.0, 0.0], [0.0; 3]);
        let b = make_params(vec![], vec![], vec![0.0, 0.0, FRAC_PI_2], [0.0; 3]);
        let r = a.lerp(&b, 0.5).expect("lerp failed");
        // Expected: 45° around Z = [0, 0, π/4]
        let expected_angle = FRAC_PI_2 / 2.0;
        let result_angle =
            (r.pose[0] * r.pose[0] + r.pose[1] * r.pose[1] + r.pose[2] * r.pose[2]).sqrt();
        assert_f32_close(result_angle, expected_angle, 1e-5, "pose slerp angle");
        // Axis should point along Z.
        if result_angle > 1e-6 {
            let axis_z = r.pose[2] / result_angle;
            assert_f32_close(axis_z, 1.0, 1e-5, "pose slerp Z axis");
        }
    }

    // ------ slerp identity quaternions ------

    #[test]
    fn slerp_identity_quaternions() {
        // Interpolating two identities at any t → identity quaternion.
        let id = [0.0f32, 0.0, 0.0, 1.0];
        let q = quat_slerp(id, id, 0.5);
        assert_f32_close(q[3], 1.0, 1e-6, "w must be 1 for identity slerp");
        assert_f32_close(
            q[0] * q[0] + q[1] * q[1] + q[2] * q[2],
            0.0,
            1e-6,
            "xyz must be 0",
        );
    }

    // ------ slerp antiparallel quaternions ------

    #[test]
    fn slerp_antiparallel_quaternions() {
        // q and -q represent the same rotation.  slerp(q, -q) should still produce
        // a unit quaternion that lies on the shorter arc.
        let q1 = [
            0.0f32,
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        ]; // 90° around Z
        let q2 = [-q1[0], -q1[1], -q1[2], -q1[3]]; // antipodal
        let q = quat_slerp(q1, q2, 0.5);
        let norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        assert_f32_close(norm_sq, 1.0, 1e-5, "result must be unit quaternion");
    }

    // ------ error on mismatched dimensions ------

    #[test]
    fn lerp_error_shape_mismatch() {
        let a = make_params(vec![1.0, 2.0], vec![], vec![], [0.0; 3]);
        let b = make_params(vec![1.0], vec![], vec![], [0.0; 3]);
        assert!(a.lerp(&b, 0.5).is_err(), "should error on shape mismatch");
    }

    #[test]
    fn lerp_error_expression_mismatch() {
        let a = make_params(vec![], vec![1.0, 2.0], vec![], [0.0; 3]);
        let b = make_params(vec![], vec![1.0], vec![], [0.0; 3]);
        assert!(
            a.lerp(&b, 0.5).is_err(),
            "should error on expression mismatch"
        );
    }

    #[test]
    fn lerp_error_pose_mismatch() {
        let a = make_params(vec![], vec![], vec![0.0, 0.0, 0.0], [0.0; 3]);
        let b = make_params(vec![], vec![], vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0; 3]);
        assert!(a.lerp(&b, 0.5).is_err(), "should error on pose mismatch");
    }

    // ------ axis-angle ↔ quaternion roundtrip ------

    #[test]
    fn roundtrip_axis_angle_quat_axis_angle() {
        // Pick a non-trivial rotation: 73° around [1,1,1] (normalised).
        let angle = 73.0_f32.to_radians();
        let inv_sqrt3 = 1.0_f32 / 3.0_f32.sqrt();
        let aa = [angle * inv_sqrt3, angle * inv_sqrt3, angle * inv_sqrt3];
        let q = axis_angle_to_quat(aa);
        let aa2 = quat_to_axis_angle(q);
        assert_slice_close(&aa, &aa2, 1e-5, "axis-angle roundtrip");
    }

    #[test]
    fn roundtrip_identity_axis_angle() {
        // Zero rotation roundtrips to zero.
        let q = axis_angle_to_quat([0.0, 0.0, 0.0]);
        assert_f32_close(q[3], 1.0, 1e-7, "identity quaternion w");
        let aa = quat_to_axis_angle(q);
        assert_slice_close(&aa, &[0.0, 0.0, 0.0], 1e-7, "identity axis-angle");
    }

    // ------ slerp_pose method ------

    #[test]
    fn slerp_pose_method_returns_correct_length() {
        let a = make_params(vec![], vec![], vec![0.0; 15], [0.0; 3]);
        let b = make_params(vec![], vec![], vec![0.1; 15], [0.0; 3]);
        let r = a.slerp_pose(&b, 0.5).expect("slerp_pose failed");
        assert_eq!(r.len(), 15, "output pose must have same length");
    }

    #[test]
    fn slerp_pose_method_error_on_mismatch() {
        let a = make_params(vec![], vec![], vec![0.0; 15], [0.0; 3]);
        let b = make_params(vec![], vec![], vec![0.0; 12], [0.0; 3]);
        assert!(a.slerp_pose(&b, 0.5).is_err());
    }

    // ------ proptest: quaternion slerp result is always unit quaternion ------

    proptest::proptest! {
        #[test]
        fn prop_quat_slerp_is_unit(
            ax in -1.0f32..1.0f32,
            ay in -1.0f32..1.0f32,
            az in -1.0f32..1.0f32,
            bx in -1.0f32..1.0f32,
            by in -1.0f32..1.0f32,
            bz in -1.0f32..1.0f32,
            t in 0.0f32..=1.0f32,
        ) {
            let aa1 = [ax, ay, az];
            let aa2 = [bx, by, bz];
            let q1 = axis_angle_to_quat(aa1);
            let q2 = axis_angle_to_quat(aa2);
            let q = quat_slerp(q1, q2, t);
            let norm_sq = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
            proptest::prop_assert!((norm_sq - 1.0).abs() < 1e-4,
                "slerp result not unit: norm_sq={}", norm_sq);
        }
    }

    proptest::proptest! {
        #[test]
        fn prop_lerp_t0_is_self(
            s0 in -3.0f32..3.0f32,
            s1 in -3.0f32..3.0f32,
            e0 in -2.0f32..2.0f32,
            p0 in -std::f32::consts::PI..std::f32::consts::PI,
            p1 in -std::f32::consts::PI..std::f32::consts::PI,
            p2 in -std::f32::consts::PI..std::f32::consts::PI,
        ) {
            let a = make_params(vec![s0, s1], vec![e0], vec![p0, p1, p2], [0.0; 3]);
            let b = make_params(vec![s0 * 0.5, s1 * 0.5], vec![e0 * 0.5], vec![0.0, 0.0, 0.0], [0.0; 3]);
            let r = a.lerp(&b, 0.0).expect("lerp t=0");
            proptest::prop_assert_eq!(r.shape[0], a.shape[0]);
            proptest::prop_assert_eq!(r.shape[1], a.shape[1]);
        }
    }
}
