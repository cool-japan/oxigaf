//! Point cloud registration: align two Gaussian clouds via ICP, with each step
//! solved in closed form by the Umeyama similarity estimator.
//!
//! Finds the optimal rigid transform (rotation + translation + optional uniform scale)
//! that aligns a source cloud to a target cloud.
//!
//! # Example
//! ```rust,no_run
//! use oxigaf_cli::cloud_registration::{register_point_clouds, RegistrationConfig};
//!
//! let source = vec![0.0f32, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0];
//! let target = vec![1.0f32, 0.0, 0.0,  2.0, 0.0, 0.0,  1.0, 1.0, 0.0];
//! let cfg = RegistrationConfig::default();
//! match register_point_clouds(&source, &target, &cfg) {
//!     Ok(result) => println!("RMSE: {:.4e}", result.final_rmse),
//!     Err(e) => eprintln!("registration failed: {e}"),
//! }
//! ```

use rayon::prelude::*;
use thiserror::Error;

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
// Private math helpers
// ---------------------------------------------------------------------------

/// 3×3 row-major matrix multiply: C = A * B.
#[inline]
fn mat3_mul(a: [f32; 9], b: [f32; 9]) -> [f32; 9] {
    let mut c = [0.0f32; 9];
    for row in 0..3 {
        for col in 0..3 {
            let mut sum = 0.0f32;
            for k in 0..3 {
                sum += a[row * 3 + k] * b[k * 3 + col];
            }
            c[row * 3 + col] = sum;
        }
    }
    c
}

/// Multiply 3×3 row-major matrix by a 3-vector.
#[inline]
fn mat3_vec(m: [f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

/// Element-wise vector subtraction.
#[inline]
fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Scalar multiplication of a vector.
#[inline]
fn vec3_scale(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Dot product of two 3-vectors.
#[inline]
fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// L2 norm of a 3-vector.
#[inline]
fn vec3_len(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Convert a unit quaternion `(qx, qy, qz, qw)` to a row-major 3×3 rotation matrix.
///
/// Assumes the input is already normalized.
#[inline]
fn quat_to_mat3(qx: f32, qy: f32, qz: f32, qw: f32) -> [f32; 9] {
    let x2 = qx * qx;
    let y2 = qy * qy;
    let z2 = qz * qz;
    let xy = qx * qy;
    let xz = qx * qz;
    let yz = qy * qz;
    let wx = qw * qx;
    let wy = qw * qy;
    let wz = qw * qz;

    [
        1.0 - 2.0 * (y2 + z2),
        2.0 * (xy - wz),
        2.0 * (xz + wy),
        2.0 * (xy + wz),
        1.0 - 2.0 * (x2 + z2),
        2.0 * (yz - wx),
        2.0 * (xz - wy),
        2.0 * (yz + wx),
        1.0 - 2.0 * (x2 + y2),
    ]
}

/// Normalize a quaternion to unit length. Returns `(0, 0, 0, 1)` if near-zero.
#[inline]
fn quat_normalize(qx: f32, qy: f32, qz: f32, qw: f32) -> (f32, f32, f32, f32) {
    let len = (qx * qx + qy * qy + qz * qz + qw * qw).sqrt();
    if len < 1e-10 {
        return (0.0, 0.0, 0.0, 1.0);
    }
    (qx / len, qy / len, qz / len, qw / len)
}

// ---------------------------------------------------------------------------
// Nearest-neighbour acceleration
// ---------------------------------------------------------------------------

/// Source-cloud size from which nearest-neighbour queries are run on the rayon
/// pool. Below it the pool hand-off costs more than the queries themselves.
const PAR_QUERY_THRESHOLD: usize = 1024;

/// Balanced k-d tree over a target cloud, stored as a permuted index array.
///
/// Every subtree occupies one contiguous slice laid out as
/// `[left | median | right]` and is split on axis `depth % 3`, so the whole tree
/// is a single `Vec<usize>` with no per-node allocation. Queries prune with the
/// squared distance to the splitting plane, which turns the per-point cost from
/// the `O(n_tgt)` of a brute-force scan into `O(log n_tgt)` on average while
/// returning exactly the same match (ties resolved towards the lower index).
struct KdTree {
    /// Target point indices, permuted into k-d tree order.
    order: Vec<usize>,
}

impl KdTree {
    /// Build the tree over the first `n` points of `target` (flat xyz).
    fn build(target: &[f32], n: usize) -> Self {
        let mut order: Vec<usize> = (0..n).collect();
        Self::split(&mut order, target, 0);
        Self { order }
    }

    /// Recursively partition `idx` around its median along axis `depth % 3`.
    fn split(idx: &mut [usize], target: &[f32], depth: usize) {
        if idx.len() <= 1 {
            return;
        }
        let axis = depth % 3;
        let mid = idx.len() / 2;
        let (left, _median, right) = idx.select_nth_unstable_by(mid, |&a, &b| {
            target[a * 3 + axis]
                .partial_cmp(&target[b * 3 + axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self::split(left, target, depth + 1);
        Self::split(right, target, depth + 1);
    }

    /// Nearest target point to `p`, as `(index, squared distance)`.
    ///
    /// Returns `None` when the cloud is empty or no finite distance exists
    /// (non-finite coordinates compare as "not better" and are skipped).
    fn nearest(&self, target: &[f32], p: [f32; 3]) -> Option<(usize, f32)> {
        let mut best = (usize::MAX, f32::MAX);
        Self::search(&self.order, target, p, 0, &mut best);
        (best.0 != usize::MAX).then_some(best)
    }

    /// Depth-first search with splitting-plane pruning.
    ///
    /// The far subtree is only skipped when the splitting plane is strictly
    /// farther than the incumbent — and never on a non-finite comparison — so
    /// every point at exactly the minimum distance is still visited and the
    /// lower-index tie-break matches a brute-force scan.
    fn search(idx: &[usize], target: &[f32], p: [f32; 3], depth: usize, best: &mut (usize, f32)) {
        if idx.is_empty() {
            return;
        }
        let axis = depth % 3;
        let mid = idx.len() / 2;
        let ti = idx[mid];
        let tp = [target[ti * 3], target[ti * 3 + 1], target[ti * 3 + 2]];
        let diff = vec3_sub(p, tp);
        let sq = vec3_dot(diff, diff);
        if sq < best.1 || (sq == best.1 && ti < best.0) {
            *best = (ti, sq);
        }
        let delta = p[axis] - tp[axis];
        let (near, far) = if delta < 0.0 {
            (&idx[..mid], &idx[mid + 1..])
        } else {
            (&idx[mid + 1..], &idx[..mid])
        };
        Self::search(near, target, p, depth + 1, best);
        let plane_sq = delta * delta;
        if plane_sq.is_nan() || plane_sq <= best.1 {
            Self::search(far, target, p, depth + 1, best);
        }
    }
}

// ---------------------------------------------------------------------------
// Core registration functions
// ---------------------------------------------------------------------------

/// Compute the mean position (centroid) of a flat position array.
///
/// Returns `[mean_x, mean_y, mean_z]`.
pub fn compute_centroid_3d(positions: &[f32]) -> Result<[f32; 3], RegistrationError> {
    if positions.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !positions.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength {
            len: positions.len(),
        });
    }
    let n = (positions.len() / 3) as f32;
    let mut sum = [0.0f32; 3];
    for chunk in positions.chunks_exact(3) {
        sum[0] += chunk[0];
        sum[1] += chunk[1];
        sum[2] += chunk[2];
    }
    Ok([sum[0] / n, sum[1] / n, sum[2] / n])
}

/// For each source point, find the nearest target point.
///
/// A balanced k-d tree is built over the target cloud once per call, so the
/// search costs `O(n_tgt log n_tgt + n_src log n_tgt)` instead of the
/// `O(n_src n_tgt)` of a brute-force scan, and returns the identical matches.
/// Only correspondences with distance < `max_dist` are included.
pub fn find_correspondences(
    source: &[f32],
    target: &[f32],
    max_dist: f32,
) -> Result<Vec<Correspondence>, RegistrationError> {
    if source.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !source.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength { len: source.len() });
    }
    if target.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !target.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength { len: target.len() });
    }

    let n_src = source.len() / 3;
    let n_tgt = target.len() / 3;
    let tree = KdTree::build(target, n_tgt);

    let match_one = |source_idx: usize| -> Option<Correspondence> {
        let sp = [
            source[source_idx * 3],
            source[source_idx * 3 + 1],
            source[source_idx * 3 + 2],
        ];
        let (target_idx, sq) = tree.nearest(target, sp)?;
        let distance = sq.sqrt();
        (distance < max_dist).then_some(Correspondence {
            source_idx,
            target_idx,
            distance,
        })
    };

    // Queries are independent, so they parallelise once the cloud is large
    // enough to outweigh the thread-pool hand-off. `collect` keeps source order.
    Ok(if n_src >= PAR_QUERY_THRESHOLD {
        (0..n_src).into_par_iter().filter_map(match_one).collect()
    } else {
        (0..n_src).filter_map(match_one).collect()
    })
}

/// Remove the worst `outlier_fraction` of correspondences (by distance).
///
/// If `outlier_fraction` is 0.0, returns all correspondences unchanged.
pub fn filter_correspondences(
    correspondences: Vec<Correspondence>,
    outlier_fraction: f32,
) -> Vec<Correspondence> {
    if outlier_fraction <= 0.0 || correspondences.is_empty() {
        return correspondences;
    }
    let mut sorted = correspondences;
    sorted.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let keep = ((sorted.len() as f32) * (1.0 - outlier_fraction.min(1.0))).ceil() as usize;
    let keep = keep.max(1).min(sorted.len());
    sorted.truncate(keep);
    sorted
}

/// Eigenvector belonging to the largest eigenvalue of a symmetric 4×4 matrix.
///
/// Cyclic Jacobi rotations: every sweep zeroes the six off-diagonal entries in
/// turn while accumulating the rotations in `v`, whose columns converge to the
/// eigenvectors. Four sweeps normally suffice for a 4×4; a few more are run so
/// that near-degenerate inputs also settle. Entries that are already negligible
/// against their diagonals are skipped, so converged sweeps cost nothing.
fn largest_eigenvector_sym4(input: &[f32; 16]) -> [f32; 4] {
    let mut a = *input;
    let mut v = [0.0f32; 16];
    v[0] = 1.0;
    v[5] = 1.0;
    v[10] = 1.0;
    v[15] = 1.0;

    for _sweep in 0..16 {
        for p in 0..3 {
            for q in (p + 1)..4 {
                let apq = a[p * 4 + q];
                let app = a[p * 4 + p];
                let aqq = a[q * 4 + q];
                if apq.abs() <= 1e-12 * (app.abs() + aqq.abs() + 1e-12) {
                    continue;
                }
                // Rotation that annihilates a[p][q]: t = tan(theta) chosen as the
                // smaller root so that the transform stays well conditioned.
                let theta = (aqq - app) / (2.0 * apq);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                a[p * 4 + p] = app - t * apq;
                a[q * 4 + q] = aqq + t * apq;
                a[p * 4 + q] = 0.0;
                a[q * 4 + p] = 0.0;
                for k in 0..4 {
                    if k != p && k != q {
                        let akp = a[k * 4 + p];
                        let akq = a[k * 4 + q];
                        a[k * 4 + p] = c * akp - s * akq;
                        a[k * 4 + q] = s * akp + c * akq;
                        a[p * 4 + k] = a[k * 4 + p];
                        a[q * 4 + k] = a[k * 4 + q];
                    }
                    let vkp = v[k * 4 + p];
                    let vkq = v[k * 4 + q];
                    v[k * 4 + p] = c * vkp - s * vkq;
                    v[k * 4 + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let mut best = 0usize;
    for i in 1..4 {
        if a[i * 5] > a[best * 5] {
            best = i;
        }
    }
    [v[best], v[4 + best], v[8 + best], v[12 + best]]
}

/// Estimate the best-fit transform aligning `source_pts` to `target_pts` with
/// the closed-form Umeyama (1991) solution.
///
/// `source_pts` and `target_pts` are flat `[x0,y0,z0, x1,y1,z1, ...]` arrays
/// of the same length (already matched by correspondence).
///
/// Both clouds are centred, the 3×3 cross-covariance `Σ = Σᵢ xsᵢ · xtᵢᵀ` is
/// accumulated, and the rotation is read off the eigenvector belonging to the
/// largest eigenvalue of Horn's symmetric 4×4 quaternion matrix built from `Σ`.
/// That maximises `Σᵢ ⟨R·xsᵢ, xtᵢ⟩` over *proper* rotations, so the result is
/// always orthonormal with `det(R) = +1`: the same answer as the SVD form with
/// its `diag(1, 1, det(U·Vᵀ))` reflection correction, without needing an SVD.
///
/// When `allow_scale` is `true` the optimal uniform scale
/// `s = Σᵢ ⟨R·xsᵢ, xtᵢ⟩ / Σᵢ ⟨xsᵢ, xsᵢ⟩` is estimated as well; when it is
/// `false` scale is fixed at 1.0 and no scale term is computed at all.
///
/// # Errors
///
/// Returns [`RegistrationError::SizeMismatch`] when the two arrays differ in
/// length and [`RegistrationError::InvalidPositionLength`] when that length is
/// not a multiple of 3. Empty (but valid) input yields the identity transform.
pub fn estimate_transform_umeyama_approx(
    source_pts: &[f32],
    target_pts: &[f32],
    allow_scale: bool,
) -> Result<RegistrationTransform, RegistrationError> {
    if source_pts.len() != target_pts.len() {
        return Err(RegistrationError::SizeMismatch {
            src: source_pts.len(),
            tgt: target_pts.len(),
        });
    }
    if !source_pts.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength {
            len: source_pts.len(),
        });
    }
    let n = source_pts.len() / 3;
    if n == 0 {
        return Ok(RegistrationTransform::identity());
    }

    // Centroids of both clouds.
    let mut cs = [0.0f32; 3];
    let mut ct = [0.0f32; 3];
    for (s, t) in source_pts.chunks_exact(3).zip(target_pts.chunks_exact(3)) {
        cs = [cs[0] + s[0], cs[1] + s[1], cs[2] + s[2]];
        ct = [ct[0] + t[0], ct[1] + t[1], ct[2] + t[2]];
    }
    let inv_n = 1.0 / n as f32;
    let cs = vec3_scale(cs, inv_n);
    let ct = vec3_scale(ct, inv_n);

    // Cross-covariance sigma[a * 3 + b] = Σᵢ (xsᵢ)ₐ · (xtᵢ)_b, plus the source
    // variance the optimal scale is normalised by.
    let mut sigma = [0.0f32; 9];
    let mut var_src = 0.0f32;
    for (s, t) in source_pts.chunks_exact(3).zip(target_pts.chunks_exact(3)) {
        let xs = vec3_sub([s[0], s[1], s[2]], cs);
        let xt = vec3_sub([t[0], t[1], t[2]], ct);
        var_src += vec3_dot(xs, xs);
        for (a, &xa) in xs.iter().enumerate() {
            sigma[a * 3] += xa * xt[0];
            sigma[a * 3 + 1] += xa * xt[1];
            sigma[a * 3 + 2] += xa * xt[2];
        }
    }

    // Horn's symmetric 4×4 matrix N, whose quadratic form qᵀ·N·q equals
    // Σᵢ ⟨R(q)·xsᵢ, xtᵢ⟩ for unit q ordered as (w, x, y, z).
    let [sxx, sxy, sxz, syx, syy, syz, szx, szy, szz] = sigma;
    let trace = sxx + syy + szz;
    let n_mat = [
        trace,
        syz - szy,
        szx - sxz,
        sxy - syx,
        syz - szy,
        sxx - syy - szz,
        sxy + syx,
        szx + sxz,
        szx - sxz,
        sxy + syx,
        -sxx + syy - szz,
        syz + szy,
        sxy - syx,
        szx + sxz,
        syz + szy,
        -sxx - syy + szz,
    ];
    let q = largest_eigenvector_sym4(&n_mat);
    let (qx, qy, qz, qw) = quat_normalize(q[1], q[2], q[3], q[0]);
    let rotation = quat_to_mat3(qx, qy, qz, qw);

    let scale = if allow_scale && var_src > 0.0 {
        // Σᵢ ⟨R·xsᵢ, xtᵢ⟩ = trace(R · Σ).
        let mut num = 0.0f32;
        for a in 0..3 {
            for b in 0..3 {
                num += rotation[a * 3 + b] * sigma[b * 3 + a];
            }
        }
        (num / var_src).max(1e-6)
    } else {
        1.0
    };

    // translation = ct - scale * R * cs
    let translation = vec3_sub(ct, vec3_scale(mat3_vec(rotation, cs), scale));

    Ok(RegistrationTransform {
        rotation,
        translation,
        scale,
    })
}

/// Perform one ICP iteration:
/// 1. Transform source by the current transform.
/// 2. Find correspondences between transformed source and target.
/// 3. Filter outlier correspondences.
/// 4. Estimate a delta transform from the matched pairs.
/// 5. Compose the delta with the current transform.
/// 6. Compute and return the RMSE of the final correspondences.
///
/// Returns `(new_transform, rmse, n_correspondences)`.
pub fn icp_step(
    source: &[f32],
    target: &[f32],
    transform: RegistrationTransform,
    config: &RegistrationConfig,
) -> Result<(RegistrationTransform, f32, usize), RegistrationError> {
    // Apply current transform to source
    let transformed = apply_registration_transform(source, &transform)?;

    // Find correspondences
    let corr_raw = find_correspondences(&transformed, target, config.max_correspondence_dist)?;
    let corr = filter_correspondences(corr_raw, config.outlier_fraction);

    if corr.is_empty() {
        // No correspondences found — return unchanged
        return Ok((transform, f32::MAX, 0));
    }

    // Extract matched point pairs (flat arrays)
    let mut src_pts = Vec::with_capacity(corr.len() * 3);
    let mut tgt_pts = Vec::with_capacity(corr.len() * 3);
    for c in &corr {
        let si = c.source_idx;
        let ti = c.target_idx;
        src_pts.push(transformed[si * 3]);
        src_pts.push(transformed[si * 3 + 1]);
        src_pts.push(transformed[si * 3 + 2]);
        tgt_pts.push(target[ti * 3]);
        tgt_pts.push(target[ti * 3 + 1]);
        tgt_pts.push(target[ti * 3 + 2]);
    }

    // Estimate delta transform on the matched subsets
    let delta = estimate_transform_umeyama_approx(&src_pts, &tgt_pts, config.allow_scale)?;

    // Compose: apply delta on top of the existing transform
    let new_transform = delta.compose(&transform);

    // Compute RMSE: re-apply new transform to source and measure against target
    let recheck = apply_registration_transform(source, &new_transform)?;
    let mut sq_sum = 0.0f32;
    for c in &corr {
        let si = c.source_idx;
        let ti = c.target_idx;
        let rp = [recheck[si * 3], recheck[si * 3 + 1], recheck[si * 3 + 2]];
        let tp = [target[ti * 3], target[ti * 3 + 1], target[ti * 3 + 2]];
        let diff = vec3_sub(rp, tp);
        sq_sum += vec3_dot(diff, diff);
    }
    let rmse = (sq_sum / corr.len() as f32).sqrt();

    Ok((new_transform, rmse, corr.len()))
}

/// Register source point cloud to target using iterative closest point (ICP).
///
/// Returns the best transform together with convergence information.
pub fn register_point_clouds(
    source: &[f32],
    target: &[f32],
    config: &RegistrationConfig,
) -> Result<RegistrationResult, RegistrationError> {
    // Validate inputs
    if source.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !source.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength { len: source.len() });
    }
    if target.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !target.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength { len: target.len() });
    }
    let n_src = source.len() / 3;
    if n_src < 3 {
        return Err(RegistrationError::InsufficientPoints {
            need: 3,
            got: n_src,
        });
    }

    // Subsample source if requested
    let working_source: Vec<f32> = if config.subsample_rate > 1 {
        subsample_positions(source, config.subsample_rate)?
    } else {
        source.to_vec()
    };

    let mut transform = RegistrationTransform::identity();
    let mut prev_rmse = f32::MAX;
    let mut converged = false;
    let mut n_correspondences = 0usize;
    let mut rmse_history: Vec<f32> = Vec::with_capacity(config.max_iterations);

    let mut iter = 0usize;
    loop {
        if iter >= config.max_iterations {
            break;
        }

        let (new_transform, rmse, n_corr) = icp_step(&working_source, target, transform, config)?;
        transform = new_transform;
        n_correspondences = n_corr;
        rmse_history.push(rmse);

        let delta = (prev_rmse - rmse).abs();
        if delta < config.tolerance && rmse < f32::MAX {
            converged = true;
            prev_rmse = rmse;
            iter += 1;
            break;
        }
        prev_rmse = rmse;
        iter += 1;
    }

    let final_rmse = prev_rmse;
    // `f32::MAX` is the sentinel `icp_step` returns when it could not match a
    // single pair, so a run that never improved on it produced nothing usable.
    if config.max_iterations > 0 && (final_rmse.is_nan() || final_rmse >= f32::MAX) {
        return Err(RegistrationError::Diverged {
            iters: iter,
            rmse: final_rmse,
        });
    }

    Ok(RegistrationResult {
        transform,
        final_rmse,
        n_iterations: iter,
        converged,
        n_correspondences,
        rmse_history,
    })
}

/// Apply a registration transform to every point in a flat positions array.
///
/// Returns a new flat positions array of the same length.
///
/// # Errors
///
/// Fails on an empty or mis-sized array, and on a transform whose scale is not
/// finite and positive (which would silently turn the cloud into garbage).
pub fn apply_registration_transform(
    positions: &[f32],
    transform: &RegistrationTransform,
) -> Result<Vec<f32>, RegistrationError> {
    if positions.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !positions.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength {
            len: positions.len(),
        });
    }
    if !transform.scale.is_finite() || transform.scale <= 0.0 {
        return Err(RegistrationError::InvalidScale {
            scale: transform.scale,
        });
    }
    let mut out = Vec::with_capacity(positions.len());
    for chunk in positions.chunks_exact(3) {
        let p = [chunk[0], chunk[1], chunk[2]];
        let q = transform.apply(p);
        out.push(q[0]);
        out.push(q[1]);
        out.push(q[2]);
    }
    Ok(out)
}

/// Compute summary statistics from a completed registration result.
///
/// `initial_rmse` should be the RMSE before any iterations were applied.
pub fn compute_registration_stats(
    result: &RegistrationResult,
    initial_rmse: f32,
) -> RegistrationStats {
    let final_rmse = result.final_rmse;
    let improvement_factor = if final_rmse > 0.0 && final_rmse < f32::MAX {
        initial_rmse / final_rmse
    } else {
        1.0
    };

    // Extract rotation angle from trace: angle = arccos((trace(R)-1)/2)
    let r = &result.transform.rotation;
    let trace = r[0] + r[4] + r[8];
    let cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    let rotation_angle_deg = cos_angle.acos() * (180.0 / std::f32::consts::PI);

    let transform_magnitude = vec3_len(result.transform.translation);
    let scale_change = (result.transform.scale - 1.0).abs();

    RegistrationStats {
        initial_rmse,
        final_rmse,
        improvement_factor,
        transform_magnitude,
        rotation_angle_deg,
        scale_change,
    }
}

/// Compute the initial RMSE between source and target using the identity transform.
///
/// Finds correspondences without applying any transform, then computes RMSE.
///
/// # Errors
///
/// Fails on empty or mis-sized clouds, and with
/// [`RegistrationError::NoCorrespondences`] when not a single pair could be
/// matched — reporting `0.0` there would read as a perfect alignment.
pub fn compute_initial_rmse(source: &[f32], target: &[f32]) -> Result<f32, RegistrationError> {
    if source.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !source.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength { len: source.len() });
    }
    if target.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !target.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength { len: target.len() });
    }

    let corr = find_correspondences(source, target, f32::MAX)?;
    if corr.is_empty() {
        return Err(RegistrationError::NoCorrespondences);
    }

    let mut sq_sum = 0.0f32;
    for c in &corr {
        let sp = [
            source[c.source_idx * 3],
            source[c.source_idx * 3 + 1],
            source[c.source_idx * 3 + 2],
        ];
        let tp = [
            target[c.target_idx * 3],
            target[c.target_idx * 3 + 1],
            target[c.target_idx * 3 + 2],
        ];
        let diff = vec3_sub(sp, tp);
        sq_sum += vec3_dot(diff, diff);
    }
    Ok((sq_sum / corr.len() as f32).sqrt())
}

/// Format a registration result and stats as a human-readable summary string.
pub fn format_registration_result(
    result: &RegistrationResult,
    stats: &RegistrationStats,
) -> String {
    format!(
        "ICP: converged={}, iter={}, RMSE: {:.4e} -> {:.4e} (x{:.2}), rot={:.2}°, t={:.4}m",
        result.converged,
        result.n_iterations,
        stats.initial_rmse,
        stats.final_rmse,
        stats.improvement_factor,
        stats.rotation_angle_deg,
        stats.transform_magnitude,
    )
}

/// Extract every `stride`-th point from a flat positions array.
///
/// `stride=1` returns all points; `stride=2` returns every other point, etc.
pub fn subsample_positions(
    positions: &[f32],
    stride: usize,
) -> Result<Vec<f32>, RegistrationError> {
    if positions.is_empty() {
        return Err(RegistrationError::EmptyCloud);
    }
    if !positions.len().is_multiple_of(3) {
        return Err(RegistrationError::InvalidPositionLength {
            len: positions.len(),
        });
    }
    let stride = stride.max(1);
    let n = positions.len() / 3;
    let out_n = n.div_ceil(stride);
    let mut out = Vec::with_capacity(out_n * 3);
    for i in (0..n).step_by(stride) {
        out.push(positions[i * 3]);
        out.push(positions[i * 3 + 1]);
        out.push(positions[i * 3 + 2]);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: assert f32 approximately equal
    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // Helper: assert two 3-vectors approximately equal
    fn close3(a: [f32; 3], b: [f32; 3]) -> bool {
        vec3_len(vec3_sub(a, b)) <= 1e-5
    }

    // Helper: determinant of a row-major 3×3 matrix
    fn mat3_det(m: [f32; 9]) -> f32 {
        m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6])
    }

    // Helper: deterministic pseudo-random cloud of N points in [0, span)³
    fn pseudo_cloud(n: usize, seed: u32, span: f32) -> Vec<f32> {
        let mut state = seed;
        (0..n * 3)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / 16_777_216.0) * span
            })
            .collect()
    }

    // Helper: build a grid of N points centred near origin
    fn grid_positions(n: usize, spacing: f32) -> Vec<f32> {
        let side = (n as f32).cbrt().ceil() as usize;
        let mut pts = Vec::with_capacity(n * 3);
        'outer: for ix in 0..side {
            for iy in 0..side {
                for iz in 0..side {
                    if pts.len() / 3 >= n {
                        break 'outer;
                    }
                    pts.push(ix as f32 * spacing);
                    pts.push(iy as f32 * spacing);
                    pts.push(iz as f32 * spacing);
                }
            }
        }
        pts
    }

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
    // mat3_mul
    // -----------------------------------------------------------------------

    #[test]
    fn test_mat3_mul_identity() {
        let id = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let m = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let r = mat3_mul(id, m);
        for i in 0..9 {
            assert!(approx_eq(r[i], m[i], 1e-6));
        }
    }

    #[test]
    fn test_mat3_mul_known() {
        // [[1,0],[0,1]] × [[1,2],[3,4]] but 3×3 version
        let a = [2.0f32, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0];
        let b = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 5.0];
        let r = mat3_mul(a, b);
        assert!(approx_eq(r[0], 2.0, 1e-6));
        assert!(approx_eq(r[4], 3.0, 1e-6));
        assert!(approx_eq(r[8], 5.0, 1e-6));
    }

    // -----------------------------------------------------------------------
    // quat_to_mat3
    // -----------------------------------------------------------------------

    #[test]
    fn test_quat_to_mat3_identity() {
        let r = quat_to_mat3(0.0, 0.0, 0.0, 1.0);
        let id = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for i in 0..9 {
            assert!(
                approx_eq(r[i], id[i], 1e-6),
                "idx {}: {} vs {}",
                i,
                r[i],
                id[i]
            );
        }
    }

    #[test]
    fn test_quat_to_mat3_90deg_z() {
        // quat for 90° around z: (0, 0, sin(45°), cos(45°))
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let r = quat_to_mat3(0.0, 0.0, s, s);
        // Should map x → y, y → -x
        let xp = mat3_vec(r, [1.0, 0.0, 0.0]);
        assert!(approx_eq(xp[0], 0.0, 1e-5));
        assert!(approx_eq(xp[1], 1.0, 1e-5));
    }

    #[test]
    fn test_quat_to_mat3_180deg_x() {
        // 180° around x: y→-y, z→-z
        let r = quat_to_mat3(1.0, 0.0, 0.0, 0.0);
        let yp = mat3_vec(r, [0.0, 1.0, 0.0]);
        assert!(approx_eq(yp[1], -1.0, 1e-5));
    }

    // -----------------------------------------------------------------------
    // quat_normalize
    // -----------------------------------------------------------------------

    #[test]
    fn test_quat_normalize_unit() {
        let (x, y, z, w) = quat_normalize(0.0, 0.0, 0.0, 1.0);
        assert!(approx_eq(w, 1.0, 1e-7));
        assert!(approx_eq(x, 0.0, 1e-7));
        assert!(approx_eq(y, 0.0, 1e-7));
        assert!(approx_eq(z, 0.0, 1e-7));
    }

    #[test]
    fn test_quat_normalize_arbitrary() {
        let (x, y, z, w) = quat_normalize(1.0, 1.0, 1.0, 1.0);
        let len = (x * x + y * y + z * z + w * w).sqrt();
        assert!(approx_eq(len, 1.0, 1e-6));
    }

    #[test]
    fn test_quat_normalize_zero() {
        let (x, y, z, w) = quat_normalize(0.0, 0.0, 0.0, 0.0);
        assert!(approx_eq(x, 0.0, 1e-7));
        assert!(approx_eq(y, 0.0, 1e-7));
        assert!(approx_eq(z, 0.0, 1e-7));
        assert!(approx_eq(w, 1.0, 1e-7));
    }

    // -----------------------------------------------------------------------
    // compute_centroid_3d
    // -----------------------------------------------------------------------

    #[test]
    fn test_centroid_single_point() {
        let pts = vec![3.0f32, -1.0, 2.0];
        let c = compute_centroid_3d(&pts).unwrap();
        assert!(approx_eq(c[0], 3.0, 1e-6));
        assert!(approx_eq(c[1], -1.0, 1e-6));
        assert!(approx_eq(c[2], 2.0, 1e-6));
    }

    #[test]
    fn test_centroid_two_points() {
        let pts = vec![0.0f32, 0.0, 0.0, 2.0, 2.0, 2.0];
        let c = compute_centroid_3d(&pts).unwrap();
        assert!(approx_eq(c[0], 1.0, 1e-6));
        assert!(approx_eq(c[1], 1.0, 1e-6));
        assert!(approx_eq(c[2], 1.0, 1e-6));
    }

    #[test]
    fn test_centroid_empty_error() {
        let result = compute_centroid_3d(&[]);
        assert!(matches!(result, Err(RegistrationError::EmptyCloud)));
    }

    #[test]
    fn test_centroid_invalid_length_error() {
        let result = compute_centroid_3d(&[1.0, 2.0]);
        assert!(matches!(
            result,
            Err(RegistrationError::InvalidPositionLength { len: 2 })
        ));
    }

    // -----------------------------------------------------------------------
    // find_correspondences
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_correspondences_basic() {
        let source = vec![0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0];
        let target = vec![0.1f32, 0.0, 0.0, 10.1, 0.0, 0.0, 100.0, 0.0, 0.0];
        let corr = find_correspondences(&source, &target, f32::MAX).unwrap();
        assert_eq!(corr.len(), 2);
        assert_eq!(corr[0].source_idx, 0);
        assert_eq!(corr[0].target_idx, 0);
        assert_eq!(corr[1].source_idx, 1);
        assert_eq!(corr[1].target_idx, 1);
    }

    #[test]
    fn test_find_correspondences_max_dist_filter() {
        let source = vec![0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0];
        let target = vec![0.1f32, 0.0, 0.0, 100.0, 0.0, 0.0];
        // Second source point is 90 units from its nearest target
        let corr = find_correspondences(&source, &target, 1.0).unwrap();
        assert_eq!(corr.len(), 1);
        assert_eq!(corr[0].source_idx, 0);
    }

    #[test]
    fn test_find_correspondences_empty_errors() {
        let no_source = find_correspondences(&[], &[0.0, 0.0, 0.0], f32::MAX);
        assert!(matches!(no_source, Err(RegistrationError::EmptyCloud)));
        let no_target = find_correspondences(&[0.0, 0.0, 0.0], &[], f32::MAX);
        assert!(matches!(no_target, Err(RegistrationError::EmptyCloud)));
    }

    #[test]
    fn test_find_correspondences_matches_brute_force() {
        // The k-d tree must return exactly what an exhaustive scan would.
        let src = pseudo_cloud(60, 12_345, 10.0);
        let tgt = pseudo_cloud(80, 987, 10.0);
        let corr = find_correspondences(&src, &tgt, f32::MAX).unwrap();
        assert_eq!(corr.len(), 60);
        for c in &corr {
            let sp = [
                src[c.source_idx * 3],
                src[c.source_idx * 3 + 1],
                src[c.source_idx * 3 + 2],
            ];
            // Same rule as the tree: smallest squared distance, lowest index.
            let mut best = (usize::MAX, f32::MAX);
            for ti in 0..80 {
                let tp = [tgt[ti * 3], tgt[ti * 3 + 1], tgt[ti * 3 + 2]];
                let diff = vec3_sub(sp, tp);
                let sq = vec3_dot(diff, diff);
                if sq < best.1 {
                    best = (ti, sq);
                }
            }
            assert_eq!(c.target_idx, best.0, "source {}", c.source_idx);
            assert!(approx_eq(c.distance, best.1.sqrt(), 1e-5));
        }
    }

    #[test]
    fn test_find_correspondences_parallel_path_keeps_order() {
        // Above the threshold the queries run on the rayon pool; the output
        // must still be one correspondence per source point, in source order.
        let n = PAR_QUERY_THRESHOLD + 16;
        let src = pseudo_cloud(n, 7, 25.0);
        let tgt = pseudo_cloud(n, 99, 25.0);
        let corr = find_correspondences(&src, &tgt, f32::MAX).unwrap();
        assert_eq!(corr.len(), n);
        for (i, c) in corr.iter().enumerate() {
            assert_eq!(c.source_idx, i);
        }
    }

    // -----------------------------------------------------------------------
    // filter_correspondences
    // -----------------------------------------------------------------------

    #[test]
    fn test_filter_correspondences_edge_cases() {
        let corr: Vec<Correspondence> = [5.0f32, 1.0, 9.0]
            .iter()
            .enumerate()
            .map(|(i, &distance)| Correspondence {
                source_idx: i,
                target_idx: i,
                distance,
            })
            .collect();
        // A zero fraction keeps everything.
        assert_eq!(filter_correspondences(corr.clone(), 0.0).len(), 3);
        // Discarding everything still keeps one correspondence.
        assert!(!filter_correspondences(corr, 1.0).is_empty());
        assert!(filter_correspondences(vec![], 0.5).is_empty());
    }

    #[test]
    fn test_filter_correspondences_half() {
        let corr: Vec<Correspondence> = (0..10)
            .map(|i| Correspondence {
                source_idx: i,
                target_idx: i,
                distance: i as f32,
            })
            .collect();
        let filtered = filter_correspondences(corr, 0.5);
        // Keep the 5 closest (distance 0..4) – ceil(10 * 0.5) = 5
        assert!(filtered.len() <= 6);
        // All remaining should have distance <= 5.0
        for c in &filtered {
            assert!(
                c.distance <= 5.0,
                "expected distance <= 5.0, got {}",
                c.distance
            );
        }
    }

    // -----------------------------------------------------------------------
    // estimate_transform_umeyama_approx
    // -----------------------------------------------------------------------

    #[test]
    fn test_umeyama_same_cloud_identity() {
        let pts = vec![
            0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let t = estimate_transform_umeyama_approx(&pts, &pts, false).unwrap();
        // The closed form is exact here: every point maps onto itself.
        for chunk in pts.chunks_exact(3) {
            let p_in = [chunk[0], chunk[1], chunk[2]];
            assert!(close3(t.apply(p_in), p_in), "point moved: {:?}", p_in);
        }
    }

    #[test]
    fn test_umeyama_pure_translation() {
        let src = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let tgt: Vec<f32> = src
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 3 == 0 { v + 5.0 } else { v })
            .collect();
        let t = estimate_transform_umeyama_approx(&src, &tgt, false).unwrap();
        // Exactly (5, 0, 0) with a proper identity rotation.
        let tr = t.translation;
        assert!(close3(tr, [5.0, 0.0, 0.0]), "{:?}", tr);
        assert!(approx_eq(mat3_det(t.rotation), 1.0, 1e-5));
    }

    #[test]
    fn test_umeyama_empty_returns_identity() {
        let t = estimate_transform_umeyama_approx(&[], &[], false).unwrap();
        assert!(approx_eq(t.scale, 1.0, 1e-6));
        assert!(close3(t.translation, [0.0; 3]));
    }

    #[test]
    fn test_umeyama_rejects_mismatched_inputs() {
        let six = vec![0.0f32; 6];
        let three = vec![0.0f32; 3];
        assert!(matches!(
            estimate_transform_umeyama_approx(&six, &three, false),
            Err(RegistrationError::SizeMismatch { src: 6, tgt: 3 })
        ));
        let four = vec![0.0f32; 4];
        assert!(matches!(
            estimate_transform_umeyama_approx(&four, &four, false),
            Err(RegistrationError::InvalidPositionLength { len: 4 })
        ));
    }

    #[test]
    fn test_umeyama_recovers_known_rigid_transform() {
        // 30° about the normalised axis (1, 2, 3), plus a translation.
        let inv = 1.0 / 14.0f32.sqrt();
        let half = 15.0f32.to_radians();
        let (sn, cs) = (half.sin(), half.cos());
        let r_known = quat_to_mat3(inv * sn, 2.0 * inv * sn, 3.0 * inv * sn, cs);
        let t_known = [0.7f32, -1.3, 2.1];
        let src = grid_positions(64, 1.0);
        let mut tgt = Vec::with_capacity(src.len());
        for p in src.chunks_exact(3) {
            let q = mat3_vec(r_known, [p[0], p[1], p[2]]);
            tgt.extend_from_slice(&[q[0] + t_known[0], q[1] + t_known[1], q[2] + t_known[2]]);
        }
        let est = estimate_transform_umeyama_approx(&src, &tgt, false).unwrap();
        for i in 0..9 {
            assert!(
                approx_eq(est.rotation[i], r_known[i], 1e-3),
                "R[{}]: {} vs {}",
                i,
                est.rotation[i],
                r_known[i]
            );
        }
        for k in 0..3 {
            assert!(approx_eq(est.translation[k], t_known[k], 1e-3), "t[{}]", k);
        }
        assert!(approx_eq(est.scale, 1.0, 1e-6));
        // A proper rotation, never a reflection.
        assert!(approx_eq(mat3_det(est.rotation), 1.0, 1e-4));
    }

    // -----------------------------------------------------------------------
    // icp_step
    // -----------------------------------------------------------------------

    #[test]
    fn test_icp_step_same_cloud() {
        let pts = grid_positions(8, 1.0);
        let cfg = RegistrationConfig::default();
        let id = RegistrationTransform::identity();
        let (new_t, rmse, n) = icp_step(&pts, &pts, id, &cfg).unwrap();
        assert!(
            rmse < 1e-5,
            "RMSE on identical clouds should vanish, got {}",
            rmse
        );
        assert!(n > 0);
        // Scale should remain close to 1.0
        assert!(approx_eq(new_t.scale, 1.0, 0.1));
    }

    #[test]
    fn test_icp_step_returns_correspondences() {
        let src = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let tgt = vec![0.1f32, 0.0, 0.0, 1.1, 0.0, 0.0, 0.1, 1.0, 0.0];
        let cfg = RegistrationConfig::default();
        let id = RegistrationTransform::identity();
        let (_t, _rmse, n) = icp_step(&src, &tgt, id, &cfg).unwrap();
        assert_eq!(n, 3);
    }

    // -----------------------------------------------------------------------
    // register_point_clouds
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_same_cloud() {
        let pts = grid_positions(27, 1.0);
        let cfg = RegistrationConfig {
            max_iterations: 10,
            tolerance: 1e-3,
            ..Default::default()
        };
        let result = register_point_clouds(&pts, &pts, &cfg).unwrap();
        // The closed-form estimator returns the identity on the first pass, so
        // the residual collapses immediately instead of creeping down.
        assert!(
            result.final_rmse < 1e-4,
            "RMSE should vanish for same cloud: {}",
            result.final_rmse
        );
    }

    #[test]
    fn test_register_recovers_small_rigid_transform() {
        // 5° about z through the cloud centre plus a sub-cell translation moves
        // every point by at most 2 * 2.83 * sin(2.5°) + 0.13 ≈ 0.37, well under
        // half the 1.0 grid spacing, so the very first correspondence set is the
        // true one and ICP must land on the exact transform.
        let src = grid_positions(125, 1.0);
        let angle = 5.0f32.to_radians();
        let (sa, ca) = (angle.sin(), angle.cos());
        let r_known = [ca, -sa, 0.0, sa, ca, 0.0, 0.0, 0.0, 1.0];
        let t_known = [0.1f32, -0.05, 0.05];
        let centre = compute_centroid_3d(&src).unwrap();
        let mut tgt = Vec::with_capacity(src.len());
        for p in src.chunks_exact(3) {
            let q = mat3_vec(r_known, vec3_sub([p[0], p[1], p[2]], centre));
            tgt.extend_from_slice(&[
                q[0] + centre[0] + t_known[0],
                q[1] + centre[1] + t_known[1],
                q[2] + centre[2] + t_known[2],
            ]);
        }
        let cfg = RegistrationConfig {
            max_iterations: 20,
            tolerance: 1e-6,
            ..Default::default()
        };
        let result = register_point_clouds(&src, &tgt, &cfg).unwrap();
        assert!(
            result.final_rmse < 1e-3,
            "RMSE should collapse, got {}",
            result.final_rmse
        );
        for i in 0..9 {
            assert!(
                approx_eq(result.transform.rotation[i], r_known[i], 1e-3),
                "R[{}]: {} vs {}",
                i,
                result.transform.rotation[i],
                r_known[i]
            );
        }
        // Orthonormal, proper rotation (composed over every iteration).
        assert!(approx_eq(mat3_det(result.transform.rotation), 1.0, 1e-4));
    }

    #[test]
    fn test_register_diverges_without_correspondences() {
        let src = grid_positions(8, 1.0);
        let tgt: Vec<f32> = src
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 3 == 0 { v + 100.0 } else { v })
            .collect();
        let cfg = RegistrationConfig {
            max_iterations: 3,
            max_correspondence_dist: 1.0,
            ..Default::default()
        };
        let result = register_point_clouds(&src, &tgt, &cfg);
        assert!(matches!(result, Err(RegistrationError::Diverged { .. })));
    }

    #[test]
    fn test_register_translated_cloud() {
        let src: Vec<f32> = grid_positions(8, 1.0);
        // Translate target by (3, 0, 0)
        let tgt: Vec<f32> = src
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 3 == 0 { v + 3.0 } else { v })
            .collect();
        let cfg = RegistrationConfig {
            max_iterations: 30,
            tolerance: 1e-4,
            ..Default::default()
        };
        let result = register_point_clouds(&src, &tgt, &cfg).unwrap();
        // Translation should be approximately (3, 0, 0)
        assert!(
            result.transform.translation[0] > 1.5,
            "expected tx > 1.5, got {}",
            result.transform.translation[0]
        );
    }

    #[test]
    fn test_register_empty_source_error() {
        let result = register_point_clouds(&[], &[0.0, 0.0, 0.0], &RegistrationConfig::default());
        assert!(matches!(result, Err(RegistrationError::EmptyCloud)));
    }

    #[test]
    fn test_register_invalid_length_error() {
        let result = register_point_clouds(
            &[1.0, 2.0],
            &[1.0, 2.0, 3.0],
            &RegistrationConfig::default(),
        );
        assert!(matches!(
            result,
            Err(RegistrationError::InvalidPositionLength { len: 2 })
        ));
    }

    #[test]
    fn test_register_insufficient_points() {
        let result = register_point_clouds(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            &RegistrationConfig::default(),
        );
        assert!(matches!(
            result,
            Err(RegistrationError::InsufficientPoints { .. })
        ));
    }

    #[test]
    fn test_register_result_fields() {
        let pts = grid_positions(27, 1.0);
        let cfg = RegistrationConfig {
            max_iterations: 5,
            ..Default::default()
        };
        let result = register_point_clouds(&pts, &pts, &cfg).unwrap();
        assert!(result.n_iterations <= 5);
        assert!(!result.rmse_history.is_empty());
    }

    #[test]
    fn test_register_with_subsample() {
        let pts = grid_positions(27, 1.0);
        let cfg = RegistrationConfig {
            max_iterations: 5,
            subsample_rate: 3,
            ..Default::default()
        };
        let result = register_point_clouds(&pts, &pts, &cfg).unwrap();
        assert!(result.final_rmse < 1.0);
    }

    #[test]
    fn test_register_with_outlier_fraction() {
        let pts = grid_positions(27, 1.0);
        let cfg = RegistrationConfig {
            max_iterations: 5,
            outlier_fraction: 0.2,
            ..Default::default()
        };
        let result = register_point_clouds(&pts, &pts, &cfg).unwrap();
        assert!(result.final_rmse < 1.0);
    }

    // -----------------------------------------------------------------------
    // apply_registration_transform
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_transform_correct_length() {
        let pts = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = RegistrationTransform::identity();
        let out = apply_registration_transform(&pts, &t).unwrap();
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn test_apply_transform_identity_unchanged() {
        let pts = vec![1.0f32, 2.0, 3.0, -1.0, 0.5, 0.0];
        let t = RegistrationTransform::identity();
        let out = apply_registration_transform(&pts, &t).unwrap();
        for (a, b) in pts.iter().zip(out.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_apply_transform_known_translation() {
        let pts = vec![0.0f32, 0.0, 0.0];
        let t = RegistrationTransform {
            rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            translation: [5.0, 6.0, 7.0],
            scale: 1.0,
        };
        let out = apply_registration_transform(&pts, &t).unwrap();
        assert!(approx_eq(out[0], 5.0, 1e-6));
        assert!(approx_eq(out[1], 6.0, 1e-6));
        assert!(approx_eq(out[2], 7.0, 1e-6));
    }

    #[test]
    fn test_apply_transform_empty_error() {
        let t = RegistrationTransform::identity();
        let result = apply_registration_transform(&[], &t);
        assert!(matches!(result, Err(RegistrationError::EmptyCloud)));
    }

    #[test]
    fn test_apply_transform_invalid_length() {
        let t = RegistrationTransform::identity();
        let result = apply_registration_transform(&[1.0, 2.0], &t);
        assert!(matches!(
            result,
            Err(RegistrationError::InvalidPositionLength { len: 2 })
        ));
    }

    #[test]
    fn test_apply_transform_rejects_non_positive_scale() {
        let mut t = RegistrationTransform::identity();
        t.scale = 0.0;
        let err = apply_registration_transform(&[1.0, 2.0, 3.0], &t);
        assert!(matches!(err, Err(RegistrationError::InvalidScale { .. })));
    }

    // -----------------------------------------------------------------------
    // compute_registration_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_improvement_factor() {
        let result = RegistrationResult {
            transform: RegistrationTransform::identity(),
            final_rmse: 0.5,
            n_iterations: 10,
            converged: true,
            n_correspondences: 100,
            rmse_history: vec![2.0, 1.0, 0.5],
        };
        let stats = compute_registration_stats(&result, 2.0);
        assert!(approx_eq(stats.improvement_factor, 4.0, 1e-5));
        assert!(approx_eq(stats.initial_rmse, 2.0, 1e-6));
        assert!(approx_eq(stats.final_rmse, 0.5, 1e-6));
    }

    #[test]
    fn test_stats_identity_rotation_angle() {
        let result = RegistrationResult {
            transform: RegistrationTransform::identity(),
            final_rmse: 0.1,
            n_iterations: 5,
            converged: true,
            n_correspondences: 50,
            rmse_history: vec![0.1],
        };
        let stats = compute_registration_stats(&result, 1.0);
        assert!(approx_eq(stats.rotation_angle_deg, 0.0, 1e-4));
        assert!(approx_eq(stats.scale_change, 0.0, 1e-6));
        assert!(approx_eq(stats.transform_magnitude, 0.0, 1e-6));
    }

    #[test]
    fn test_stats_zero_final_rmse() {
        let result = RegistrationResult {
            transform: RegistrationTransform::identity(),
            final_rmse: 0.0,
            n_iterations: 3,
            converged: true,
            n_correspondences: 10,
            rmse_history: vec![0.0],
        };
        let stats = compute_registration_stats(&result, 1.0);
        // When final_rmse = 0, improvement_factor should be 1.0 (no divide by zero)
        assert!(approx_eq(stats.improvement_factor, 1.0, 1e-6));
    }

    // -----------------------------------------------------------------------
    // compute_initial_rmse
    // -----------------------------------------------------------------------

    #[test]
    fn test_initial_rmse_same_cloud() {
        let pts = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
        let rmse = compute_initial_rmse(&pts, &pts).unwrap();
        assert!(approx_eq(rmse, 0.0, 1e-6));
    }

    #[test]
    fn test_initial_rmse_offset() {
        let src = vec![0.0f32, 0.0, 0.0];
        let tgt = vec![1.0f32, 0.0, 0.0];
        let rmse = compute_initial_rmse(&src, &tgt).unwrap();
        assert!(approx_eq(rmse, 1.0, 1e-5));
    }

    #[test]
    fn test_initial_rmse_empty_error() {
        let result = compute_initial_rmse(&[], &[1.0, 2.0, 3.0]);
        assert!(matches!(result, Err(RegistrationError::EmptyCloud)));
    }

    #[test]
    fn test_initial_rmse_without_correspondences_errors() {
        // A non-finite target yields no valid pair; 0.0 would read as a
        // perfect alignment, so an error must come back instead.
        let src = vec![0.0f32, 0.0, 0.0];
        let tgt = vec![f32::NAN, 0.0, 0.0];
        let result = compute_initial_rmse(&src, &tgt);
        assert!(matches!(result, Err(RegistrationError::NoCorrespondences)));
    }

    // -----------------------------------------------------------------------
    // format_registration_result
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_result_non_empty() {
        let result = RegistrationResult {
            transform: RegistrationTransform::identity(),
            final_rmse: 0.01,
            n_iterations: 20,
            converged: true,
            n_correspondences: 100,
            rmse_history: vec![0.01],
        };
        let stats = compute_registration_stats(&result, 1.0);
        let s = format_registration_result(&result, &stats);
        assert!(!s.is_empty());
        assert!(s.contains("ICP:"));
        assert!(s.contains("converged=true"));
    }

    #[test]
    fn test_format_result_contains_iter_count() {
        let result = RegistrationResult {
            transform: RegistrationTransform::identity(),
            final_rmse: 0.1,
            n_iterations: 42,
            converged: false,
            n_correspondences: 30,
            rmse_history: vec![0.1],
        };
        let stats = compute_registration_stats(&result, 0.5);
        let s = format_registration_result(&result, &stats);
        assert!(s.contains("42"), "Expected iter count 42 in: {}", s);
    }

    // -----------------------------------------------------------------------
    // subsample_positions
    // -----------------------------------------------------------------------

    #[test]
    fn test_subsample_strides() {
        let pts = grid_positions(8, 1.0);
        assert_eq!(subsample_positions(&pts, 1).unwrap().len(), pts.len());
        // 4 of 8 points survive, 3 floats each.
        assert_eq!(subsample_positions(&pts, 2).unwrap().len(), 12);
        let four = vec![
            0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0,
        ];
        // stride=4 → only first point
        let out = subsample_positions(&four, 4).unwrap();
        assert_eq!(out.len(), 3);
        assert!(approx_eq(out[0], 0.0, 1e-6));
    }

    #[test]
    fn test_subsample_errors() {
        let empty = subsample_positions(&[], 2);
        assert!(matches!(empty, Err(RegistrationError::EmptyCloud)));
        assert!(matches!(
            subsample_positions(&[1.0, 2.0], 1),
            Err(RegistrationError::InvalidPositionLength { len: 2 })
        ));
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

    // -----------------------------------------------------------------------
    // Math helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_vector_helpers() {
        let ex = [1.0f32, 0.0, 0.0];
        let ey = [0.0f32, 1.0, 0.0];
        let v = [1.0f32, 2.0, 3.0];
        let ones = [1.0f32, 1.0, 1.0];
        assert!(close3(vec3_sub([3.0, 2.0, 1.0], ones), [2.0, 1.0, 0.0]));
        assert!(close3(vec3_scale(v, 2.0), [2.0, 4.0, 6.0]));
        assert!(approx_eq(vec3_dot(ex, ey), 0.0, 1e-6));
        assert!(approx_eq(vec3_dot(v, v), 14.0, 1e-5));
        assert!(approx_eq(vec3_len([3.0, 4.0, 0.0]), 5.0, 1e-5));
        let diag = [2.0f32, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
        assert!(close3(mat3_vec(diag, [1.0, 2.0, 3.0]), [2.0, 6.0, 12.0]));
        assert!(approx_eq(mat3_det(diag), 24.0, 1e-4));
    }

    // -----------------------------------------------------------------------
    // Edge cases / robustness
    // -----------------------------------------------------------------------

    #[test]
    fn test_register_convergence_flag() {
        let pts = grid_positions(27, 1.0);
        let cfg = RegistrationConfig {
            max_iterations: 50,
            tolerance: 1e-2,
            ..Default::default()
        };
        let result = register_point_clouds(&pts, &pts, &cfg).unwrap();
        // Same cloud should converge
        assert!(result.converged || result.n_iterations == 50);
    }

    #[test]
    fn test_find_correspondences_self_nearest() {
        // Each point is its own nearest neighbour
        let pts = vec![0.0f32, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0, 0.0];
        let corr = find_correspondences(&pts, &pts, f32::MAX).unwrap();
        for c in &corr {
            assert_eq!(c.source_idx, c.target_idx);
            assert!(approx_eq(c.distance, 0.0, 1e-6));
        }
    }

    #[test]
    fn test_subsample_zero_stride_treated_as_one() {
        let pts = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = subsample_positions(&pts, 0).unwrap();
        assert_eq!(out.len(), pts.len());
    }

    #[test]
    fn test_umeyama_scale_allowed() {
        let src = vec![
            0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        ];
        // Scale target by 2.0
        let tgt: Vec<f32> = src.iter().map(|&v| v * 2.0).collect();
        let t = estimate_transform_umeyama_approx(&src, &tgt, true).unwrap();
        // The closed form recovers the factor exactly.
        let sc = t.scale;
        assert!(approx_eq(sc, 2.0, 1e-4), "expected 2.0, got {}", sc);
    }
}
