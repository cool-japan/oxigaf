//! Public data types of the registration module: the error enum, the
//! transform, the tuning knobs, and the result/statistics carriers.

use thiserror::Error;

use super::math::{mat3_mul, mat3_vec};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by registration operations.
#[derive(Debug, Error)]
pub enum RegistrationError {
    /// No positions were provided.
    #[error("Empty point cloud: no positions")]
    EmptyCloud,

    /// Positions length is not divisible by 3.
    #[error("Positions length {len} is not divisible by 3")]
    InvalidPositionLength { len: usize },

    /// Source and target have different numbers of points.
    #[error("Source and target have different sizes: {src} vs {tgt}")]
    SizeMismatch { src: usize, tgt: usize },

    /// Registration did not converge.
    #[error("Registration diverged after {iters} iterations (RMSE: {rmse:.4e})")]
    Diverged { iters: usize, rmse: f32 },

    /// Too few points for a meaningful registration.
    #[error("Insufficient points: need at least {need}, got {got}")]
    InsufficientPoints { need: usize, got: usize },

    /// Not a single valid source/target pair could be established.
    #[error("No valid correspondences: every distance was non-finite or above the cutoff")]
    NoCorrespondences,

    /// Scale factor is not positive.
    #[error("Invalid scale factor {scale}: must be > 0")]
    InvalidScale { scale: f32 },
}

// ---------------------------------------------------------------------------
// RegistrationTransform
// ---------------------------------------------------------------------------

/// Rigid body + uniform scale transform for registration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegistrationTransform {
    /// 3×3 rotation matrix in row-major order.
    pub rotation: [f32; 9],
    /// Translation vector `[tx, ty, tz]`.
    pub translation: [f32; 3],
    /// Uniform scale factor (1.0 = no scale change).
    pub scale: f32,
}

impl RegistrationTransform {
    /// Identity transform: no rotation, no translation, unit scale.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            translation: [0.0, 0.0, 0.0],
            scale: 1.0,
        }
    }

    /// Apply this transform to a single point.
    ///
    /// `p_out[i] = scale * (R[i,0]*p[0] + R[i,1]*p[1] + R[i,2]*p[2]) + t[i]`
    #[must_use]
    pub fn apply(&self, point: [f32; 3]) -> [f32; 3] {
        let r = &self.rotation;
        let rotated = [
            r[0] * point[0] + r[1] * point[1] + r[2] * point[2],
            r[3] * point[0] + r[4] * point[1] + r[5] * point[2],
            r[6] * point[0] + r[7] * point[1] + r[8] * point[2],
        ];
        [
            self.scale * rotated[0] + self.translation[0],
            self.scale * rotated[1] + self.translation[1],
            self.scale * rotated[2] + self.translation[2],
        ]
    }

    /// Compose two transforms: result = self after other.
    ///
    /// Applying the result is equivalent to first applying `other`, then `self`.
    ///
    /// - `scale_out = self.scale * other.scale`
    /// - `R_out = self.R * other.R`
    /// - `t_out = self.scale * self.R * other.t + self.t`
    #[must_use]
    pub fn compose(&self, other: &RegistrationTransform) -> RegistrationTransform {
        let r_out = mat3_mul(self.rotation, other.rotation);
        let rot_other_t = mat3_vec(self.rotation, other.translation);
        let t_out = [
            self.scale * rot_other_t[0] + self.translation[0],
            self.scale * rot_other_t[1] + self.translation[1],
            self.scale * rot_other_t[2] + self.translation[2],
        ];
        RegistrationTransform {
            rotation: r_out,
            translation: t_out,
            scale: self.scale * other.scale,
        }
    }
}

// ---------------------------------------------------------------------------
// RegistrationConfig
// ---------------------------------------------------------------------------

/// Configuration for iterative closest point (ICP) registration.
#[derive(Debug, Clone)]
pub struct RegistrationConfig {
    /// Maximum number of ICP iterations.
    pub max_iterations: usize,
    /// Convergence threshold: stop when |prev_rmse - rmse| < tolerance.
    pub tolerance: f32,
    /// Maximum distance for a valid correspondence (`f32::MAX` = accept all).
    pub max_correspondence_dist: f32,
    /// Allow estimating a uniform scale factor in addition to R and t.
    pub allow_scale: bool,
    /// Use every `subsample_rate`-th source point (1 = use all).
    pub subsample_rate: usize,
    /// Fraction of worst correspondences to discard as outliers (0.0 = none).
    pub outlier_fraction: f32,
}

impl Default for RegistrationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-5,
            max_correspondence_dist: f32::MAX,
            allow_scale: false,
            subsample_rate: 1,
            outlier_fraction: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// RegistrationResult
// ---------------------------------------------------------------------------

/// Full result of an ICP registration run.
#[derive(Debug, Clone)]
pub struct RegistrationResult {
    /// The estimated transform aligning source to target.
    pub transform: RegistrationTransform,
    /// Final RMSE of the correspondences.
    pub final_rmse: f32,
    /// Total number of ICP iterations performed.
    pub n_iterations: usize,
    /// Whether the algorithm converged within tolerance.
    pub converged: bool,
    /// Number of point correspondences used in the final iteration.
    pub n_correspondences: usize,
    /// RMSE after each iteration (for convergence plotting).
    pub rmse_history: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Correspondence
// ---------------------------------------------------------------------------

/// A matched pair between a source point and the nearest target point.
#[derive(Debug, Clone, Copy)]
pub struct Correspondence {
    /// Index (in units of points, not flat array) of the source point.
    pub source_idx: usize,
    /// Index (in units of points) of the nearest target point.
    pub target_idx: usize,
    /// Euclidean distance between the pair.
    pub distance: f32,
}

// ---------------------------------------------------------------------------
// RegistrationStats
// ---------------------------------------------------------------------------

/// Derived statistics computed from a completed registration.
#[derive(Debug, Clone, Copy)]
pub struct RegistrationStats {
    /// RMSE before any registration was applied.
    pub initial_rmse: f32,
    /// RMSE after registration.
    pub final_rmse: f32,
    /// `initial_rmse / final_rmse` (1.0 when final_rmse is zero or no improvement).
    pub improvement_factor: f32,
    /// L2 norm of the translation vector.
    pub transform_magnitude: f32,
    /// Rotation angle extracted from the rotation matrix, in degrees.
    pub rotation_angle_deg: f32,
    /// `|scale - 1.0|`.
    pub scale_change: f32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud_registration::test_support::{approx_eq, close3};

    // -----------------------------------------------------------------------
    // RegistrationTransform::identity
    // -----------------------------------------------------------------------

    #[test]
    fn test_identity_transform() {
        let t = RegistrationTransform::identity();
        assert_eq!(t.scale, 1.0);
        for p in [[0.0f32, 0.0, 0.0], [3.0, -2.0, 1.5]] {
            let q = t.apply(p);
            for k in 0..3 {
                assert!(approx_eq(q[k], p[k], 1e-6), "coord {}", k);
            }
        }
    }

    // -----------------------------------------------------------------------
    // RegistrationTransform::apply — known transforms
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_known_transforms() {
        let id = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mk = |rotation: [f32; 9], translation: [f32; 3], scale: f32| RegistrationTransform {
            rotation,
            translation,
            scale,
        };
        // Pure translation.
        let t = mk(id, [1.0, 2.0, 3.0], 1.0);
        assert!(close3(t.apply([0.0, 0.0, 0.0]), [1.0, 2.0, 3.0]));
        // Pure scale.
        let t = mk(id, [0.0; 3], 2.0);
        assert!(close3(t.apply([1.0, 1.0, 1.0]), [2.0, 2.0, 2.0]));
        // 90° rotation around z: x->y, y->-x, z->z
        let rot_z90 = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let t = mk(rot_z90, [0.0; 3], 1.0);
        assert!(close3(t.apply([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]));
    }

    // -----------------------------------------------------------------------
    // RegistrationTransform::compose
    // -----------------------------------------------------------------------

    #[test]
    fn test_compose() {
        let id = RegistrationTransform::identity();
        let mk = |translation: [f32; 3], scale: f32| RegistrationTransform {
            rotation: id.rotation,
            translation,
            scale,
        };
        // identity ∘ t should equal t
        let t = mk([1.0, 2.0, 3.0], 1.5);
        let p = [0.5, -0.5, 1.0];
        assert!(close3(id.compose(&t).apply(p), t.apply(p)));
        // Translations add (the outer scale applies to the inner one) and
        // scales multiply.
        let composed = mk([1.0, 0.0, 0.0], 2.0).compose(&mk([0.0, 1.0, 0.0], 3.0));
        assert!(approx_eq(composed.scale, 6.0, 1e-6));
        assert!(close3(composed.apply([0.0; 3]), [1.0, 2.0, 0.0]));
    }

    // -----------------------------------------------------------------------
    // RegistrationError variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_displays() {
        use RegistrationError as E;
        let rmse = 0.5;
        let cases: Vec<(E, &str)> = vec![
            (E::EmptyCloud, "Empty"),
            (E::InvalidPositionLength { len: 7 }, "7"),
            (E::SizeMismatch { src: 10, tgt: 20 }, "20"),
            (E::Diverged { iters: 42, rmse }, "42"),
            (E::InsufficientPoints { need: 3, got: 2 }, "2"),
            (E::InvalidScale { scale: -1.0 }, "-1"),
            (E::NoCorrespondences, "correspondence"),
        ];
        for (e, needle) in cases {
            let rendered = format!("{}", e);
            assert!(rendered.contains(needle), "{:?} -> {}", e, rendered);
        }
    }

    // -----------------------------------------------------------------------
    // Plain data carriers
    // -----------------------------------------------------------------------

    #[test]
    fn test_result_and_correspondence_fields() {
        let c = Correspondence {
            source_idx: 5,
            target_idx: 3,
            distance: 0.25,
        };
        assert_eq!(c.source_idx, 5);
        assert_eq!(c.target_idx, 3);
        assert!(approx_eq(c.distance, 0.25, 1e-7));
        let result = RegistrationResult {
            transform: RegistrationTransform::identity(),
            final_rmse: 0.05,
            n_iterations: 15,
            converged: true,
            n_correspondences: 200,
            rmse_history: vec![1.0, 0.5, 0.05],
        };
        assert!(approx_eq(result.final_rmse, 0.05, 1e-6));
        assert_eq!(result.n_iterations, 15);
        assert!(result.converged);
        assert_eq!(result.n_correspondences, 200);
        assert_eq!(result.rmse_history.len(), 3);
    }

    // -----------------------------------------------------------------------
    // RegistrationConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_defaults() {
        let cfg = RegistrationConfig::default();
        assert_eq!(cfg.max_iterations, 100);
        assert!(approx_eq(cfg.tolerance, 1e-5, 1e-10));
        assert_eq!(cfg.max_correspondence_dist, f32::MAX);
        assert!(!cfg.allow_scale);
        assert_eq!(cfg.subsample_rate, 1);
        assert!(approx_eq(cfg.outlier_fraction, 0.0, 1e-10));
    }
}
