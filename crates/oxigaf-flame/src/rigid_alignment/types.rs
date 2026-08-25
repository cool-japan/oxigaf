//! Data types for rigid alignment: the error type, the [`SimilarityTransform`]
//! result type, ICP configuration/result, and alignment statistics.
//!
//! Split out of the former monolithic `rigid_alignment.rs` to stay under the
//! workspace's 2000-line-per-file policy; see [`super::svd`] for the 3×3
//! Jacobi SVD, [`super::procrustes`] for closed-form Procrustes fitting,
//! [`super::icp`] for nearest-neighbour search and ICP, and
//! [`super::landmarks`] for landmark-based alignment and statistics.

use super::svd::{mat3_identity, mat3_mul, mat3_transpose};

// ─── Error type ─────────────────────────────────────────────────────────────

/// Errors returned by rigid-alignment functions.
#[derive(Debug, thiserror::Error)]
pub enum AlignmentError {
    /// Fewer points than the minimum required.
    #[error("not enough points: need at least {needed}, got {got}")]
    NotEnoughPoints { needed: usize, got: usize },

    /// Source and target have different point counts.
    #[error("dimension mismatch: source has {src} points, target has {tgt}")]
    PointCountMismatch { src: usize, tgt: usize },

    /// Matrix inversion failed (numerically singular).
    #[error("singular matrix: cannot invert")]
    Singular,

    /// ICP did not converge within the iteration budget.
    #[error("did not converge after {0} iterations")]
    DidNotConverge(usize),

    /// Invalid configuration parameter.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// Empty slice was passed where at least one element is required.
    #[error("empty input")]
    EmptyInput,

    /// Lengths of weights and landmarks disagree.
    #[error("weight count {w} does not match landmark count {n}")]
    WeightLengthMismatch { w: usize, n: usize },
}

// ─── SimilarityTransform ────────────────────────────────────────────────────

/// A rigid similarity transform: `x' = scale * R * x + t`.
///
/// `rotation` is stored as a 3×3 row-major matrix with `det = +1`.
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarityTransform {
    /// 3×3 rotation matrix (row-major, det = +1).
    pub rotation: [f32; 9],
    /// Translation vector.
    pub translation: [f32; 3],
    /// Uniform scale factor.
    pub scale: f32,
}

impl SimilarityTransform {
    /// Identity transform (no rotation, no translation, scale = 1).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            rotation: mat3_identity(),
            translation: [0.0; 3],
            scale: 1.0,
        }
    }

    /// Apply to a single 3D point: `x' = scale * R * x + t`.
    #[must_use]
    pub fn apply(&self, point: [f32; 3]) -> [f32; 3] {
        let r = &self.rotation;
        let s = self.scale;
        let t = self.translation;
        [
            s * (r[0] * point[0] + r[1] * point[1] + r[2] * point[2]) + t[0],
            s * (r[3] * point[0] + r[4] * point[1] + r[5] * point[2]) + t[1],
            s * (r[6] * point[0] + r[7] * point[1] + r[8] * point[2]) + t[2],
        ]
    }

    /// Apply to a batch of points (array of `[f32;3]`).
    #[must_use]
    pub fn apply_batch(&self, points: &[[f32; 3]]) -> Vec<[f32; 3]> {
        points.iter().map(|&p| self.apply(p)).collect()
    }

    /// Apply to a flat N×3 slice (length must be divisible by 3).
    #[must_use]
    pub fn apply_flat(&self, positions: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(positions.len());
        for chunk in positions.chunks_exact(3) {
            let p = [chunk[0], chunk[1], chunk[2]];
            let q = self.apply(p);
            out.extend_from_slice(&q);
        }
        // handle any leftover (degenerate case — just copy)
        let rem = positions.len() % 3;
        if rem > 0 {
            out.extend_from_slice(&positions[positions.len() - rem..]);
        }
        out
    }

    /// Return the inverse transform such that `inv.apply(self.apply(x)) ≈ x`.
    ///
    /// * `scale_inv` = 1 / scale
    /// * `rotation_inv` = Rᵀ
    /// * `translation_inv` = -(1/scale) * Rᵀ * t
    #[must_use]
    pub fn inverse(&self) -> SimilarityTransform {
        let s_inv = if self.scale.abs() > f32::EPSILON {
            1.0 / self.scale
        } else {
            0.0
        };
        let rt = mat3_transpose(&self.rotation);
        let t = self.translation;
        // -(1/s) * Rᵀ * t
        let ti = [
            -s_inv * (rt[0] * t[0] + rt[1] * t[1] + rt[2] * t[2]),
            -s_inv * (rt[3] * t[0] + rt[4] * t[1] + rt[5] * t[2]),
            -s_inv * (rt[6] * t[0] + rt[7] * t[1] + rt[8] * t[2]),
        ];
        SimilarityTransform {
            rotation: rt,
            translation: ti,
            scale: s_inv,
        }
    }

    /// Compose `self` then `other`: `other.apply(self.apply(x))`.
    ///
    /// * scale = s1 * s2
    /// * rotation = R2 * R1
    /// * translation = s2 * R2 * t1 + t2
    #[must_use]
    pub fn compose(&self, other: &SimilarityTransform) -> SimilarityTransform {
        let r_new = mat3_mul(&other.rotation, &self.rotation);
        let s_new = self.scale * other.scale;
        let t1 = self.translation;
        let r2 = &other.rotation;
        let s2 = other.scale;
        let t2 = other.translation;
        let t_new = [
            s2 * (r2[0] * t1[0] + r2[1] * t1[1] + r2[2] * t1[2]) + t2[0],
            s2 * (r2[3] * t1[0] + r2[4] * t1[1] + r2[5] * t1[2]) + t2[1],
            s2 * (r2[6] * t1[0] + r2[7] * t1[1] + r2[8] * t1[2]) + t2[2],
        ];
        SimilarityTransform {
            rotation: r_new,
            translation: t_new,
            scale: s_new,
        }
    }

    /// Return the rotation as a `[[f32;3];3]` array.
    #[must_use]
    pub fn rotation_matrix(&self) -> [[f32; 3]; 3] {
        let r = &self.rotation;
        [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]]
    }
}

/// Configuration for Iterative Closest Point.
#[derive(Debug, Clone)]
pub struct IcpConfig {
    /// Maximum number of ICP iterations.
    pub max_iterations: usize,
    /// RMSE change below which we declare convergence.
    pub convergence_threshold: f32,
    /// Reject correspondences farther than this distance.
    pub max_correspondence_dist: f32,
    /// Whether to allow scale in addition to rigid alignment.
    pub use_scale: bool,
}

impl Default for IcpConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            convergence_threshold: 1e-5,
            max_correspondence_dist: f32::INFINITY,
            use_scale: false,
        }
    }
}

/// Result of ICP alignment.
#[derive(Debug, Clone)]
pub struct IcpResult {
    /// Accumulated similarity transform.
    pub transform: SimilarityTransform,
    /// Final RMSE between aligned source and target correspondences.
    pub final_rmse: f32,
    /// Number of iterations performed.
    pub n_iterations: usize,
    /// Whether convergence criterion was met.
    pub converged: bool,
    /// RMSE at each iteration.
    pub rmse_history: Vec<f32>,
}

// ─── Alignment statistics ─────────────────────────────────────────────────────

/// Summary statistics for an alignment operation.
#[derive(Debug, Clone)]
pub struct AlignmentStats {
    /// RMSE before applying the transform.
    pub rmse_before: f32,
    /// RMSE after applying the transform.
    pub rmse_after: f32,
    /// `rmse_before / rmse_after` — larger is better.
    pub improvement_ratio: f32,
    /// Mean nearest-neighbour distance post-alignment.
    pub mean_correspondence_dist: f32,
    /// Maximum nearest-neighbour distance post-alignment.
    pub max_correspondence_dist: f32,
    /// Number of correspondences used.
    pub n_correspondences: usize,
}
