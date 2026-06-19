//! Pose prior for the FLAME parametric head model.
//!
//! This module provides anatomically-constrained pose scoring and projection for the five
//! FLAME joints (global rotation, neck, jaw, left eye, right eye). It combines hard joint
//! limits with a Gaussian prior to produce a differentiable pose score suitable for
//! gradient-based optimisation.
//!
//! # Joint Layout
//!
//! The full 15-dimensional pose vector is laid out as:
//! ```text
//! pose[0..3]   – Joint 0: global rotation (axis-angle)
//! pose[3..6]   – Joint 1: neck rotation
//! pose[6..9]   – Joint 2: jaw rotation
//! pose[9..12]  – Joint 3: left eye rotation
//! pose[12..15] – Joint 4: right eye rotation
//! ```
//!
//! All rotations are expressed in the axis-angle convention: the direction of the vector
//! defines the rotation axis and its magnitude defines the rotation angle in radians.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when working with the pose prior.
#[derive(Debug, thiserror::Error)]
pub enum PosePriorError {
    /// Pose vector has wrong length.
    #[error("Pose vector length {0}, expected 15 (5 joints × 3)")]
    InvalidPoseLength(usize),

    /// Joint index is out of the valid range.
    #[error("Joint index {0} out of range [0, 4]")]
    InvalidJointIndex(usize),

    /// PCA basis has wrong number of rows.
    #[error("PCA basis shape mismatch: got {got} rows, expected {expected}")]
    PcaBasisMismatch {
        /// Actual number of rows received.
        got: usize,
        /// Number of rows that was expected.
        expected: usize,
    },

    /// Covariance matrix is not positive definite.
    #[error("Invalid covariance: must be positive definite (dim {0})")]
    InvalidCovariance(usize),
}

// ---------------------------------------------------------------------------
// JointLimits
// ---------------------------------------------------------------------------

/// Anatomical joint rotation limits expressed in terms of the three axis-angle components.
///
/// Each component limit is `[min, max]` in radians. Note that the axis-angle vector
/// components do *not* correspond to Euler angles in general; the limits serve as soft
/// bounding boxes in axis-angle space that match the anatomically plausible range.
#[derive(Debug, Clone)]
pub struct JointLimits {
    /// Per-component `[min, max]` bounds: `[[min_x, max_x], [min_y, max_y], [min_z, max_z]]`.
    pub component_limits: [[f32; 2]; 3],
}

impl JointLimits {
    /// Create new joint limits from explicit per-component bounds.
    #[must_use]
    pub fn new(limits: [[f32; 2]; 3]) -> Self {
        Self {
            component_limits: limits,
        }
    }

    /// Return `true` when all three axis-angle components are within the configured bounds.
    #[must_use]
    pub fn is_valid(&self, aa: [f32; 3]) -> bool {
        for (component, &limit_pair) in aa.iter().zip(self.component_limits.iter()) {
            let [lo, hi] = limit_pair;
            if *component < lo || *component > hi {
                return false;
            }
        }
        true
    }

    /// Return the axis-angle vector clamped so that every component is within its bounds.
    #[must_use]
    pub fn clamp(&self, aa: [f32; 3]) -> [f32; 3] {
        let mut out = aa;
        for (out_val, &limit_pair) in out.iter_mut().zip(self.component_limits.iter()) {
            let [lo, hi] = limit_pair;
            *out_val = out_val.clamp(lo, hi);
        }
        out
    }

    /// Return the L2 norm of the per-component bound violations.
    ///
    /// Returns `0.0` when the vector is within limits.
    #[must_use]
    pub fn violation_norm(&self, aa: [f32; 3]) -> f32 {
        let mut sq_sum = 0.0_f32;
        for (component, &limit_pair) in aa.iter().zip(self.component_limits.iter()) {
            let [lo, hi] = limit_pair;
            let violation = if *component < lo {
                lo - *component
            } else if *component > hi {
                *component - hi
            } else {
                0.0
            };
            sq_sum += violation * violation;
        }
        sq_sum.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Default limits
// ---------------------------------------------------------------------------

/// Return the default anatomical joint limits for all five FLAME joints.
///
/// The limits are:
/// - **Joint 0** (global rotation): ±90° yaw, ±30° pitch, ±30° roll
/// - **Joint 1** (neck): ±45° yaw, ±30° pitch, ±15° roll
/// - **Joint 2** (jaw): 0–30° opening, ±3° yaw/roll
/// - **Joint 3** (left eye): ±25° all axes
/// - **Joint 4** (right eye): ±25° all axes
#[must_use]
pub fn default_joint_limits() -> [JointLimits; 5] {
    [
        // Joint 0: global rotation — head turns freely
        JointLimits::new([
            [-1.57, 1.57], // yaw (y-axis): ±90°
            [-0.52, 0.52], // pitch (x-axis): ±30°
            [-0.52, 0.52], // roll (z-axis): ±30°
        ]),
        // Joint 1: neck — more restricted than global
        JointLimits::new([
            [-0.79, 0.79], // yaw: ±45°
            [-0.52, 0.52], // pitch: ±30°
            [-0.26, 0.26], // roll: ±15°
        ]),
        // Joint 2: jaw — opens only (positive x), minimal yaw/roll
        JointLimits::new([
            [0.0, 0.52],   // opening: 0–30°
            [-0.05, 0.05], // minimal yaw
            [-0.05, 0.05], // minimal roll
        ]),
        // Joint 3: left eye — ±25° all axes
        JointLimits::new([[-0.44, 0.44], [-0.44, 0.44], [-0.44, 0.44]]),
        // Joint 4: right eye — ±25° all axes
        JointLimits::new([[-0.44, 0.44], [-0.44, 0.44], [-0.44, 0.44]]),
    ]
}

// ---------------------------------------------------------------------------
// GaussianPosePrior
// ---------------------------------------------------------------------------

/// Diagonal multivariate Gaussian pose prior.
///
/// The prior density is:
/// ```text
/// P(pose) ∝ exp(-0.5 · Σᵢ (pose_i − mean_i)² / exp(log_diag_cov_i))
/// ```
///
/// Using the log-diagonal form avoids numerical issues when covariance values span
/// many orders of magnitude, and makes the representation amenable to gradient-based
/// learning.
#[derive(Debug, Clone)]
pub struct GaussianPosePrior {
    /// 15-dimensional mean pose (neutral/rest position).
    pub mean: Vec<f32>,
    /// Log of the diagonal covariance elements (15-dimensional).
    pub log_diag_cov: Vec<f32>,
}

impl GaussianPosePrior {
    /// Create a Gaussian prior from an explicit mean and log-diagonal covariance.
    ///
    /// Returns [`PosePriorError::InvalidPoseLength`] when either vector does not have
    /// exactly 15 elements.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn new(mean: Vec<f32>, log_diag_cov: Vec<f32>) -> Result<Self, PosePriorError> {
        if mean.len() != 15 {
            return Err(PosePriorError::InvalidPoseLength(mean.len()));
        }
        if log_diag_cov.len() != 15 {
            return Err(PosePriorError::InvalidPoseLength(log_diag_cov.len()));
        }
        Ok(Self { mean, log_diag_cov })
    }

    /// Create a neutral prior: zero mean with moderate, isotropic variance.
    ///
    /// The log-covariance is initialised to `log(0.1) ≈ −2.303` for every dimension,
    /// corresponding to a standard deviation of ~0.316 rad (~18°) per component.
    #[must_use]
    pub fn neutral() -> Self {
        let log_cov = (0.1_f32).ln();
        Self {
            mean: vec![0.0_f32; 15],
            log_diag_cov: vec![log_cov; 15],
        }
    }

    /// Compute the negative log-likelihood of a pose under this prior.
    ///
    /// ```text
    /// NLL = 0.5 · Σᵢ (pose_i − mean_i)² / exp(log_diag_cov_i)
    /// ```
    ///
    /// Returns [`PosePriorError::InvalidPoseLength`] when `pose.len() != 15`.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn neg_log_likelihood(&self, pose: &[f32]) -> Result<f32, PosePriorError> {
        if pose.len() != 15 {
            return Err(PosePriorError::InvalidPoseLength(pose.len()));
        }
        let nll = pose
            .iter()
            .zip(self.mean.iter())
            .zip(self.log_diag_cov.iter())
            .map(|((p, m), lc)| {
                let diff = p - m;
                0.5 * diff * diff / lc.exp()
            })
            .sum();
        Ok(nll)
    }

    /// Compute the gradient of the NLL with respect to each pose component.
    ///
    /// ```text
    /// ∂NLL/∂pose_i = (pose_i − mean_i) / exp(log_diag_cov_i)
    /// ```
    ///
    /// Returns [`PosePriorError::InvalidPoseLength`] when `pose.len() != 15`.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn nll_gradient(&self, pose: &[f32]) -> Result<Vec<f32>, PosePriorError> {
        if pose.len() != 15 {
            return Err(PosePriorError::InvalidPoseLength(pose.len()));
        }
        let grad = pose
            .iter()
            .zip(self.mean.iter())
            .zip(self.log_diag_cov.iter())
            .map(|((p, m), lc)| (p - m) / lc.exp())
            .collect();
        Ok(grad)
    }
}

// ---------------------------------------------------------------------------
// PoseValidityReport
// ---------------------------------------------------------------------------

/// Detailed validity report for a FLAME pose vector.
#[derive(Debug, Clone)]
pub struct PoseValidityReport {
    /// `true` when every joint is within its anatomical limits.
    pub is_valid: bool,
    /// Per-joint violation norm (0.0 when the joint is within limits).
    pub joint_violations: [f32; 5],
    /// Sum of per-joint violation norms.
    pub total_violation: f32,
    /// Negative log-likelihood under the Gaussian prior (lower = more likely).
    pub prior_score: f32,
    /// Human-readable summary of the validity check.
    pub summary: String,
}

// ---------------------------------------------------------------------------
// PoseScorer
// ---------------------------------------------------------------------------

/// Combined pose scorer that integrates hard joint-limit enforcement with a Gaussian prior.
///
/// The total score is:
/// ```text
/// score = limit_weight × total_violation + prior_weight × neg_log_likelihood
/// ```
pub struct PoseScorer {
    /// Anatomical limits for each of the five FLAME joints.
    pub joint_limits: [JointLimits; 5],
    /// Gaussian prior over the full 15-dimensional pose vector.
    pub prior: GaussianPosePrior,
    /// Penalty weight applied to joint-limit violations.
    pub limit_weight: f32,
    /// Weight applied to the Gaussian prior NLL term.
    pub prior_weight: f32,
}

impl PoseScorer {
    /// Create a scorer with explicitly supplied components.
    #[must_use]
    pub fn new(
        joint_limits: [JointLimits; 5],
        prior: GaussianPosePrior,
        limit_weight: f32,
        prior_weight: f32,
    ) -> Self {
        Self {
            joint_limits,
            prior,
            limit_weight,
            prior_weight,
        }
    }

    /// Create the default FLAME scorer with anatomical limits and a neutral Gaussian prior.
    #[must_use]
    pub fn default_flame() -> Self {
        Self::new(
            default_joint_limits(),
            GaussianPosePrior::neutral(),
            10.0,
            1.0,
        )
    }

    /// Compute the total score for a pose.
    ///
    /// ```text
    /// score = limit_weight × total_violation + prior_weight × neg_log_likelihood
    /// ```
    ///
    /// Returns [`PosePriorError::InvalidPoseLength`] when `pose.len() != 15`.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn score(&self, pose: &[f32]) -> Result<f32, PosePriorError> {
        if pose.len() != 15 {
            return Err(PosePriorError::InvalidPoseLength(pose.len()));
        }
        let total_violation = self.compute_total_violation(pose);
        let nll = self.prior.neg_log_likelihood(pose)?;
        Ok(self.limit_weight * total_violation + self.prior_weight * nll)
    }

    /// Produce a detailed validity report for the given pose.
    ///
    /// Returns [`PosePriorError::InvalidPoseLength`] when `pose.len() != 15`.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn validate(&self, pose: &[f32]) -> Result<PoseValidityReport, PosePriorError> {
        if pose.len() != 15 {
            return Err(PosePriorError::InvalidPoseLength(pose.len()));
        }
        let mut joint_violations = [0.0_f32; 5];
        for (joint_idx, viol) in joint_violations.iter_mut().enumerate() {
            let aa = get_joint(pose, joint_idx)?;
            *viol = self.joint_limits[joint_idx].violation_norm(aa);
        }
        let total_violation: f32 = joint_violations.iter().sum();
        let is_valid = total_violation == 0.0;
        let prior_score = self.prior.neg_log_likelihood(pose)?;

        let summary = if is_valid {
            format!("Pose is valid. Prior NLL: {prior_score:.4}")
        } else {
            let violating: Vec<String> = joint_violations
                .iter()
                .enumerate()
                .filter(|(_, v)| **v > 0.0)
                .map(|(i, v)| format!("joint {i} ({v:.4})"))
                .collect();
            format!(
                "Pose INVALID — violations: {}. Prior NLL: {prior_score:.4}",
                violating.join(", ")
            )
        };

        Ok(PoseValidityReport {
            is_valid,
            joint_violations,
            total_violation,
            prior_score,
            summary,
        })
    }

    /// Return a copy of the pose with all per-joint components clamped to their limits.
    ///
    /// This enforces hard anatomical limits but does not optimise the Gaussian prior.
    ///
    /// Returns [`PosePriorError::InvalidPoseLength`] when `pose.len() != 15`.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn clamp_to_limits(&self, pose: &[f32]) -> Result<Vec<f32>, PosePriorError> {
        if pose.len() != 15 {
            return Err(PosePriorError::InvalidPoseLength(pose.len()));
        }
        let mut out = pose.to_vec();
        for j in 0..5 {
            let aa = get_joint(&out, j)?;
            let clamped = self.joint_limits[j].clamp(aa);
            set_joint(&mut out, j, clamped)?;
        }
        Ok(out)
    }

    /// Compute the gradient of the total score with respect to each pose component.
    ///
    /// The gradient has two terms:
    /// - **Prior gradient**: `prior_weight × ∂NLL/∂pose_i`
    /// - **Violation gradient**: `limit_weight × 2 × (pose_i − clamp(pose_i, lo_i, hi_i))`
    ///
    /// Returns [`PosePriorError::InvalidPoseLength`] when `pose.len() != 15`.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn score_gradient(&self, pose: &[f32]) -> Result<Vec<f32>, PosePriorError> {
        if pose.len() != 15 {
            return Err(PosePriorError::InvalidPoseLength(pose.len()));
        }
        let prior_grad = self.prior.nll_gradient(pose)?;
        let clamped = self.clamp_to_limits(pose)?;
        let grad: Vec<f32> = pose
            .iter()
            .zip(clamped.iter())
            .zip(prior_grad.iter())
            .map(|((p, c), pg)| {
                let violation_grad = (p - c) * 2.0;
                self.prior_weight * pg + self.limit_weight * violation_grad
            })
            .collect();
        Ok(grad)
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Compute the sum of per-joint violation norms.
    ///
    /// Caller must ensure `pose.len() == 15`.
    fn compute_total_violation(&self, pose: &[f32]) -> f32 {
        (0..5)
            .map(|j| {
                let start = j * 3;
                let aa = [pose[start], pose[start + 1], pose[start + 2]];
                self.joint_limits[j].violation_norm(aa)
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Extract the three axis-angle components of joint `joint` (0–4) from a flat 15-dim pose.
///
/// Returns [`PosePriorError::InvalidPoseLength`] when `pose.len() != 15` and
/// [`PosePriorError::InvalidJointIndex`] when `joint > 4`.
///
/// # Errors
///
/// Returns an error if the operation fails.
#[must_use = "returns the extracted joint components"]
pub fn get_joint(pose: &[f32], joint: usize) -> Result<[f32; 3], PosePriorError> {
    if pose.len() != 15 {
        return Err(PosePriorError::InvalidPoseLength(pose.len()));
    }
    if joint > 4 {
        return Err(PosePriorError::InvalidJointIndex(joint));
    }
    let s = joint * 3;
    Ok([pose[s], pose[s + 1], pose[s + 2]])
}

/// Overwrite the axis-angle components of joint `joint` (0–4) in a flat 15-dim pose.
///
/// Returns [`PosePriorError::InvalidPoseLength`] when `pose.len() != 15` and
/// [`PosePriorError::InvalidJointIndex`] when `joint > 4`.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn set_joint(pose: &mut [f32], joint: usize, aa: [f32; 3]) -> Result<(), PosePriorError> {
    if pose.len() != 15 {
        return Err(PosePriorError::InvalidPoseLength(pose.len()));
    }
    if joint > 4 {
        return Err(PosePriorError::InvalidJointIndex(joint));
    }
    let s = joint * 3;
    pose[s] = aa[0];
    pose[s + 1] = aa[1];
    pose[s + 2] = aa[2];
    Ok(())
}

/// Compute the magnitude (rotation angle in radians) of an axis-angle vector.
#[must_use]
pub fn aa_magnitude(aa: [f32; 3]) -> f32 {
    (aa[0] * aa[0] + aa[1] * aa[1] + aa[2] * aa[2]).sqrt()
}

/// Return an approximate Euler-angle representation of an axis-angle vector.
///
/// For display purposes: if the rotation magnitude exceeds π the function maps it to the
/// equivalent rotation `(2π − θ)` around the negated axis, keeping each component in
/// `(−π, π]`.
#[must_use]
pub fn aa_to_euler_approx(aa: [f32; 3]) -> [f32; 3] {
    let theta = aa_magnitude(aa);
    if theta < 1e-8 {
        return [0.0, 0.0, 0.0];
    }
    if theta > PI {
        // Equivalent rotation: angle = 2π − θ, axis = −axis / θ
        let new_theta = 2.0 * PI - theta;
        let scale = -new_theta / theta;
        [aa[0] * scale, aa[1] * scale, aa[2] * scale]
    } else {
        aa
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // JointLimits
    // -----------------------------------------------------------------------

    #[test]
    fn joint_limits_is_valid_zero_vector() {
        let limits = default_joint_limits();
        // Every joint should accept the zero vector.
        for lim in &limits {
            assert!(lim.is_valid([0.0, 0.0, 0.0]));
        }
    }

    #[test]
    fn joint_limits_is_valid_out_of_range() {
        let lim = JointLimits::new([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]);
        // 5.0 >> 1.0 → must be invalid
        assert!(!lim.is_valid([5.0, 0.0, 0.0]));
    }

    #[test]
    fn joint_limits_clamp_brings_to_limit() {
        let lim = JointLimits::new([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]);
        let clamped = lim.clamp([5.0, -5.0, 0.5]);
        assert!(
            (clamped[0] - 1.0).abs() < 1e-6,
            "x should be clamped to 1.0"
        );
        assert!(
            (clamped[1] + 1.0).abs() < 1e-6,
            "y should be clamped to -1.0"
        );
        assert!((clamped[2] - 0.5).abs() < 1e-6, "z within range, unchanged");
    }

    #[test]
    fn joint_limits_violation_norm_valid_is_zero() {
        let lim = JointLimits::new([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]);
        let v = lim.violation_norm([0.0, 0.5, -0.5]);
        assert!(v.abs() < 1e-6, "expected 0.0 for valid pose, got {v}");
    }

    #[test]
    fn joint_limits_violation_norm_invalid_is_positive() {
        let lim = JointLimits::new([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]);
        let v = lim.violation_norm([2.0, 0.0, 0.0]);
        assert!(v > 0.0, "expected positive violation, got {v}");
    }

    // -----------------------------------------------------------------------
    // default_joint_limits
    // -----------------------------------------------------------------------

    #[test]
    fn default_joint_limits_returns_five_entries() {
        let limits = default_joint_limits();
        assert_eq!(limits.len(), 5);
    }

    // -----------------------------------------------------------------------
    // GaussianPosePrior
    // -----------------------------------------------------------------------

    #[test]
    fn gaussian_prior_neutral_has_correct_dims() {
        let prior = GaussianPosePrior::neutral();
        assert_eq!(prior.mean.len(), 15);
        assert_eq!(prior.log_diag_cov.len(), 15);
    }

    #[test]
    fn gaussian_prior_nll_at_mean_is_zero() {
        let prior = GaussianPosePrior::neutral();
        let mean_pose = prior.mean.clone();
        let nll = prior.neg_log_likelihood(&mean_pose).expect("valid pose");
        assert!(nll.abs() < 1e-6, "NLL at mean should be 0.0, got {nll}");
    }

    #[test]
    fn gaussian_prior_nll_away_from_mean_is_positive() {
        let prior = GaussianPosePrior::neutral();
        let pose: Vec<f32> = vec![0.1; 15];
        let nll = prior.neg_log_likelihood(&pose).expect("valid pose");
        assert!(
            nll > 0.0,
            "NLL away from mean should be positive, got {nll}"
        );
    }

    #[test]
    fn gaussian_prior_gradient_at_mean_is_zero() {
        let prior = GaussianPosePrior::neutral();
        let mean_pose = prior.mean.clone();
        let grad = prior.nll_gradient(&mean_pose).expect("valid pose");
        for (i, g) in grad.iter().enumerate() {
            assert!(g.abs() < 1e-6, "gradient[{i}] at mean should be 0, got {g}");
        }
    }

    #[test]
    fn gaussian_prior_gradient_length_is_fifteen() {
        let prior = GaussianPosePrior::neutral();
        let pose = vec![0.1_f32; 15];
        let grad = prior.nll_gradient(&pose).expect("valid pose");
        assert_eq!(grad.len(), 15);
    }

    #[test]
    fn gaussian_prior_new_length_mismatch_errors() {
        let result = GaussianPosePrior::new(vec![0.0; 10], vec![0.0; 15]);
        assert!(
            matches!(result, Err(PosePriorError::InvalidPoseLength(10))),
            "expected InvalidPoseLength(10)"
        );
    }

    // -----------------------------------------------------------------------
    // get_joint / set_joint
    // -----------------------------------------------------------------------

    #[test]
    fn get_joint_index_zero_returns_first_three() {
        let pose: Vec<f32> = (0..15).map(|i| i as f32).collect();
        let aa = get_joint(&pose, 0).expect("valid");
        assert_eq!(aa, [0.0, 1.0, 2.0]);
    }

    #[test]
    fn get_joint_index_four_returns_last_three() {
        let pose: Vec<f32> = (0..15).map(|i| i as f32).collect();
        let aa = get_joint(&pose, 4).expect("valid");
        assert_eq!(aa, [12.0, 13.0, 14.0]);
    }

    #[test]
    fn get_joint_index_five_is_error() {
        let pose = vec![0.0_f32; 15];
        let result = get_joint(&pose, 5);
        assert!(
            matches!(result, Err(PosePriorError::InvalidJointIndex(5))),
            "expected InvalidJointIndex(5)"
        );
    }

    #[test]
    fn set_joint_updates_correct_elements() {
        let mut pose = vec![0.0_f32; 15];
        set_joint(&mut pose, 2, [1.0, 2.0, 3.0]).expect("valid");
        assert_eq!(&pose[6..9], &[1.0, 2.0, 3.0]);
        // Other elements must be untouched.
        assert_eq!(pose[0], 0.0);
        assert_eq!(pose[14], 0.0);
    }

    // -----------------------------------------------------------------------
    // PoseScorer
    // -----------------------------------------------------------------------

    #[test]
    fn pose_scorer_default_flame_creates_valid_scorer() {
        let scorer = PoseScorer::default_flame();
        // Should not panic; check weights are as documented.
        assert!((scorer.limit_weight - 10.0).abs() < 1e-6);
        assert!((scorer.prior_weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pose_scorer_score_neutral_pose_is_low() {
        let scorer = PoseScorer::default_flame();
        let pose = vec![0.0_f32; 15];
        let s = scorer.score(&pose).expect("valid");
        // Neutral pose is at the prior mean and has no limit violations.
        assert!(s.abs() < 1e-6, "neutral score should be ~0, got {s}");
    }

    #[test]
    fn pose_scorer_score_invalid_pose_is_high() {
        let scorer = PoseScorer::default_flame();
        // Jaw wide open: component[6] = 5.0 (way beyond 0.52 limit)
        let mut pose = vec![0.0_f32; 15];
        pose[6] = 5.0;
        let s = scorer.score(&pose).expect("valid");
        assert!(s > 1.0, "invalid pose should have high score, got {s}");
    }

    #[test]
    fn pose_scorer_validate_neutral_is_valid() {
        let scorer = PoseScorer::default_flame();
        let pose = vec![0.0_f32; 15];
        let report = scorer.validate(&pose).expect("valid");
        assert!(report.is_valid, "neutral pose should be valid");
    }

    #[test]
    fn pose_scorer_validate_out_of_limit_is_invalid() {
        let scorer = PoseScorer::default_flame();
        // Global rotation way out of range.
        let mut pose = vec![0.0_f32; 15];
        pose[0] = 10.0;
        let report = scorer.validate(&pose).expect("valid");
        assert!(!report.is_valid, "out-of-limit pose should be invalid");
    }

    #[test]
    fn pose_scorer_clamp_neutral_unchanged() {
        let scorer = PoseScorer::default_flame();
        let pose = vec![0.0_f32; 15];
        let clamped = scorer.clamp_to_limits(&pose).expect("valid");
        for (a, b) in pose.iter().zip(clamped.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "neutral pose must not change under clamping"
            );
        }
    }

    #[test]
    fn pose_scorer_clamp_jaw_gets_clamped() {
        let scorer = PoseScorer::default_flame();
        // Jaw component: pose[6] too negative (below 0.0 limit for jaw x-component)
        let mut pose = vec![0.0_f32; 15];
        pose[6] = -1.0; // jaw x must be ≥ 0.0
        let clamped = scorer.clamp_to_limits(&pose).expect("valid");
        assert!(
            clamped[6] >= 0.0,
            "clamped jaw x should be ≥ 0.0, got {}",
            clamped[6]
        );
    }

    #[test]
    fn pose_scorer_score_gradient_length_is_fifteen() {
        let scorer = PoseScorer::default_flame();
        let pose = vec![0.1_f32; 15];
        let grad = scorer.score_gradient(&pose).expect("valid");
        assert_eq!(grad.len(), 15);
    }

    // -----------------------------------------------------------------------
    // aa_magnitude
    // -----------------------------------------------------------------------

    #[test]
    fn aa_magnitude_zero_vector_is_zero() {
        assert!(aa_magnitude([0.0, 0.0, 0.0]).abs() < 1e-8);
    }

    #[test]
    fn aa_magnitude_unit_vector_is_one() {
        assert!((aa_magnitude([1.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);
    }
}
