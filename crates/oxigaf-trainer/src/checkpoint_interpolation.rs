//! Checkpoint Interpolation for OxiGAF Gaussian Avatar Training
//!
//! This module implements weight-space interpolation between model checkpoints,
//! enabling model merging, ensembling, and smooth trajectory generation between
//! training states.
//!
//! # Overview
//!
//! Gaussian model parameters are stored as flat `Vec<f32>` arrays:
//! - positions: N×3
//! - rotations: N×4 quaternion [qx, qy, qz, qw]
//! - scales: N×3 (log-scale)
//! - opacities: N (logit-space)
//! - sh_coefficients: N×C
//!
//! # Quick Start
//!
//! ```rust
//! use oxigaf_trainer::checkpoint_interpolation::{
//!     ParamSnapshot, linear_interpolate, interpolation_sequence,
//!     model_soup, compute_interpolation_stats,
//! };
//!
//! // Create two snapshots
//! let snap_a = ParamSnapshot { name: "step_0".into(), step: 0, params: vec![0.0; 16] };
//! let snap_b = ParamSnapshot { name: "step_1".into(), step: 1, params: vec![1.0; 16] };
//!
//! // Interpolate at t=0.5
//! let mid = linear_interpolate(&snap_a.params, &snap_b.params, 0.5).unwrap();
//! assert!((mid[0] - 0.5).abs() < 1e-6);
//!
//! // Model soup: uniform average
//! let soup = model_soup(&[snap_a, snap_b]).unwrap();
//! assert!((soup[0] - 0.5).abs() < 1e-6);
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by checkpoint interpolation operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum InterpolationError {
    #[error("Parameter vectors have different lengths: {len_a} vs {len_b}")]
    LengthMismatch { len_a: usize, len_b: usize },

    #[error("Empty parameter vector")]
    EmptyParams,

    #[error("Invalid interpolation parameter t={t}: must be in [0, 1]")]
    InvalidT { t: f32 },

    #[error("Invalid step count {n}: must be >= 2")]
    InvalidStepCount { n: usize },

    #[error("Checkpoint list is empty")]
    EmptyCheckpointList,

    #[error("Blend weights length {weights_len} does not match checkpoint count {n_checkpoints}")]
    WeightCountMismatch {
        weights_len: usize,
        n_checkpoints: usize,
    },

    #[error("Blend weights must be non-negative and sum to 1.0, got sum={sum:.4}")]
    InvalidBlendWeights { sum: f32 },

    #[error("Blend weight at index {index} is negative: {value}")]
    NegativeBlendWeight { index: usize, value: f32 },

    #[error(
        "n_quaternion_params={n_quaternion_params} is invalid for a {param_len}-element \
         parameter vector: must be a multiple of 4 and <= param_len"
    )]
    InvalidQuaternionRegion {
        n_quaternion_params: usize,
        param_len: usize,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Core data structures
// ─────────────────────────────────────────────────────────────────────────────

/// A named parameter snapshot (checkpoint).
#[derive(Debug, Clone)]
pub struct ParamSnapshot {
    /// Identifier (e.g., "step_1000")
    pub name: String,
    /// Training step at which this snapshot was taken.
    pub step: usize,
    /// Flat parameter vector.
    pub params: Vec<f32>,
}

/// Configuration for checkpoint interpolation.
#[derive(Debug, Clone)]
pub struct InterpolationConfig {
    /// Use SLERP for rotation params (default: false — use lerp).
    pub use_slerp_for_rotations: bool,
    /// Re-normalize quaternions after interpolation (default: true).
    pub normalize_rotations: bool,
    /// Number of rotation params for normalization. 0 = auto-detect.
    pub n_quaternion_params: usize,
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            use_slerp_for_rotations: false,
            normalize_rotations: true,
            n_quaternion_params: 0,
        }
    }
}

/// A path through checkpoint space.
#[derive(Debug, Clone)]
pub struct CheckpointPath {
    /// Ordered snapshots along the path.
    pub snapshots: Vec<ParamSnapshot>,
    /// L2 distance between consecutive snapshots.
    pub step_sizes: Vec<f32>,
    /// Sum of all step sizes.
    pub total_length: f32,
}

/// Statistics about an interpolated parameter set.
#[derive(Debug, Clone)]
pub struct InterpolationStats {
    /// Dimensionality of the parameter vector.
    pub param_dim: usize,
    /// Mean of all parameters.
    pub mean: f32,
    /// Population standard deviation.
    pub std: f32,
    /// Minimum parameter value.
    pub min: f32,
    /// Maximum parameter value.
    pub max: f32,
    /// L2 norm of the parameter vector.
    pub l2_norm: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Private math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Element-wise linear interpolation: `(1 - t) * a[i] + t * b[i]`.
///
/// Returns an error if lengths differ or `t` is not in [0, 1].
fn lerp_params(a: &[f32], b: &[f32], t: f32) -> Result<Vec<f32>, InterpolationError> {
    if !(0.0..=1.0).contains(&t) {
        return Err(InterpolationError::InvalidT { t });
    }
    if a.len() != b.len() {
        return Err(InterpolationError::LengthMismatch {
            len_a: a.len(),
            len_b: b.len(),
        });
    }
    let one_minus_t = 1.0 - t;
    Ok(a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| one_minus_t * ai + t * bi)
        .collect())
}

/// Spherical linear interpolation between two unit quaternions.
///
/// - Handles the short-path (dot < 0) by negating qb.
/// - Falls back to lerp + normalize when quaternions are nearly parallel.
///
/// Used by [`interpolate_with_config`] when
/// [`InterpolationConfig::use_slerp_for_rotations`] is set.
fn slerp_quat(qa: &[f32; 4], qb: &[f32; 4], t: f32) -> [f32; 4] {
    let dot = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3];

    // Take the short path
    let (dot, qb_effective) = if dot < 0.0 {
        (-dot, [-qb[0], -qb[1], -qb[2], -qb[3]])
    } else {
        (dot, *qb)
    };

    // Clamp to avoid numerical issues with acos
    let dot = dot.clamp(-1.0, 1.0);

    const SLERP_THRESHOLD: f32 = 0.9995;

    let mut result = if dot > SLERP_THRESHOLD {
        // Nearly parallel — use lerp + normalize
        [
            (1.0 - t) * qa[0] + t * qb_effective[0],
            (1.0 - t) * qa[1] + t * qb_effective[1],
            (1.0 - t) * qa[2] + t * qb_effective[2],
            (1.0 - t) * qa[3] + t * qb_effective[3],
        ]
    } else {
        let theta = dot.acos();
        let sin_theta = theta.sin();
        let scale_a = ((1.0 - t) * theta).sin() / sin_theta;
        let scale_b = (t * theta).sin() / sin_theta;
        [
            scale_a * qa[0] + scale_b * qb_effective[0],
            scale_a * qa[1] + scale_b * qb_effective[1],
            scale_a * qa[2] + scale_b * qb_effective[2],
            scale_a * qa[3] + scale_b * qb_effective[3],
        ]
    };

    normalize_quaternion(&mut result);
    result
}

/// Normalize a 4-element quaternion slice in-place.
///
/// Used by [`interpolate_with_config`] when
/// [`InterpolationConfig::normalize_rotations`] is set.
fn normalize_quaternion(q: &mut [f32]) {
    let norm_sq = q.iter().map(|v| v * v).sum::<f32>();
    if norm_sq > 1e-24 {
        let inv_norm = norm_sq.sqrt().recip();
        for v in q.iter_mut() {
            *v *= inv_norm;
        }
    }
}

/// L2 norm of a flat parameter vector.
pub fn params_l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Mean of all elements in a flat parameter vector.
pub fn params_mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f32>() / v.len() as f32
}

/// Population standard deviation of a flat parameter vector.
pub fn params_std(v: &[f32]) -> f32 {
    if v.len() < 2 {
        return 0.0;
    }
    let mean = params_mean(v);
    let variance = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32;
    variance.sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Public distance / similarity functions
// ─────────────────────────────────────────────────────────────────────────────

/// L2 distance between two flat parameter vectors.
///
/// Returns `sqrt(Σ (a[i] - b[i])²)`.
///
/// Note: Named `params_l2_distance` to distinguish from
/// `weights_l2_distance` in the `weight_averaging` module.
///
/// # Errors
///
/// Returns [`InterpolationError::LengthMismatch`] if lengths differ.
pub fn params_l2_distance(a: &[f32], b: &[f32]) -> Result<f32, InterpolationError> {
    if a.len() != b.len() {
        return Err(InterpolationError::LengthMismatch {
            len_a: a.len(),
            len_b: b.len(),
        });
    }
    let sum_sq: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| (ai - bi) * (ai - bi))
        .sum();
    Ok(sum_sq.sqrt())
}

/// Cosine similarity between two flat parameter vectors.
///
/// Returns `dot(a, b) / (norm(a) * norm(b))`.
/// Returns `0.0` if either norm is approximately zero.
///
/// Note: Named `params_cosine_similarity` to distinguish from
/// `weights_cosine_similarity` in the `weight_averaging` module.
///
/// # Errors
///
/// Returns [`InterpolationError::LengthMismatch`] if lengths differ.
pub fn params_cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, InterpolationError> {
    if a.len() != b.len() {
        return Err(InterpolationError::LengthMismatch {
            len_a: a.len(),
            len_b: b.len(),
        });
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
    let norm_a = params_l2_norm(a);
    let norm_b = params_l2_norm(b);
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return Ok(0.0);
    }
    Ok(dot / (norm_a * norm_b))
}

// ─────────────────────────────────────────────────────────────────────────────
// Public interpolation functions
// ─────────────────────────────────────────────────────────────────────────────

/// Linear interpolation between two flat parameter vectors.
///
/// Equivalent to `(1 - t) * a + t * b` element-wise.
///
/// # Errors
///
/// - [`InterpolationError::InvalidT`] if `t` is not in [0, 1].
/// - [`InterpolationError::LengthMismatch`] if `a.len() != b.len()`.
pub fn linear_interpolate(a: &[f32], b: &[f32], t: f32) -> Result<Vec<f32>, InterpolationError> {
    lerp_params(a, b, t)
}

/// Interpolate between two flat parameter vectors, honoring
/// [`InterpolationConfig`]'s quaternion-handling options.
///
/// With the default config (`n_quaternion_params: 0`) this is exactly
/// equivalent to [`linear_interpolate`] — plain element-wise lerp — since
/// there is no reliable way to locate a rotation sub-range within an opaque
/// flat `Vec<f32>` without additional layout information, so auto-detect
/// deliberately declines rather than guessing.
///
/// When `config.n_quaternion_params` is set to a positive multiple of 4, the
/// **first** `n_quaternion_params` elements of `a`/`b` are treated as a
/// contiguous block of `[qx, qy, qz, qw]` quaternions (matching this
/// module's documented parameter layout, where rotations are the first
/// per-Gaussian block after positions — callers whose flat vector places
/// rotations elsewhere should slice around that region themselves):
///
/// - If [`InterpolationConfig::use_slerp_for_rotations`] is set, each
///   4-tuple in that block is interpolated via spherical linear
///   interpolation (SLERP) instead of a plain lerp — SLERP always returns a
///   unit quaternion, so `normalize_rotations` has no additional effect on
///   this block in that case.
/// - Otherwise, the block is lerped normally (same as the rest of the
///   vector), and if [`InterpolationConfig::normalize_rotations`] is also
///   set, each 4-tuple is renormalized to unit length afterward — a plain
///   component-wise lerp between two unit quaternions is not itself
///   unit-length in general.
/// - Elements outside the quaternion block are always lerped normally,
///   regardless of `config`.
///
/// # Errors
///
/// - [`InterpolationError::InvalidT`] if `t` is not in [0, 1].
/// - [`InterpolationError::LengthMismatch`] if `a.len() != b.len()`.
/// - [`InterpolationError::InvalidQuaternionRegion`] if
///   `config.n_quaternion_params` is not a multiple of 4, or exceeds the
///   parameter vector length.
pub fn interpolate_with_config(
    a: &[f32],
    b: &[f32],
    t: f32,
    config: &InterpolationConfig,
) -> Result<Vec<f32>, InterpolationError> {
    let mut result = lerp_params(a, b, t)?;
    apply_quaternion_region(&mut result, a, b, t, config)?;
    Ok(result)
}

/// Shared helper: re-processes the quaternion sub-range of an
/// already-lerped `result` (computed from `a`/`b`/`t`) according to
/// `config`. `result`, `a`, and `b` are all assumed to have equal length
/// (the caller must have already validated this, e.g. via [`lerp_params`]).
fn apply_quaternion_region(
    result: &mut [f32],
    a: &[f32],
    b: &[f32],
    t: f32,
    config: &InterpolationConfig,
) -> Result<(), InterpolationError> {
    let n_quat = config.n_quaternion_params;
    if n_quat == 0 {
        return Ok(());
    }
    if !n_quat.is_multiple_of(4) || n_quat > result.len() {
        return Err(InterpolationError::InvalidQuaternionRegion {
            n_quaternion_params: n_quat,
            param_len: result.len(),
        });
    }

    let mut i = 0;
    while i < n_quat {
        if config.use_slerp_for_rotations {
            let qa: [f32; 4] = [a[i], a[i + 1], a[i + 2], a[i + 3]];
            let qb: [f32; 4] = [b[i], b[i + 1], b[i + 2], b[i + 3]];
            let q = slerp_quat(&qa, &qb, t);
            result[i..i + 4].copy_from_slice(&q);
        } else if config.normalize_rotations {
            normalize_quaternion(&mut result[i..i + 4]);
        }
        i += 4;
    }
    Ok(())
}

/// Compute a weighted average of checkpoint parameter vectors.
///
/// `result[i] = Σ (weights[j] / Σweights) * checkpoints[j].params[i]`
///
/// Weights are renormalized by their actual sum before being applied (see
/// below), so `result` is always a true convex combination even though the
/// sum-to-1.0 check has tolerance.
///
/// # Errors
///
/// - [`InterpolationError::EmptyCheckpointList`] if no checkpoints are provided.
/// - [`InterpolationError::WeightCountMismatch`] if weight count differs.
/// - [`InterpolationError::InvalidBlendWeights`] if weights don't sum to ~1.0.
/// - [`InterpolationError::NegativeBlendWeight`] if any weight is negative.
/// - [`InterpolationError::LengthMismatch`] if param vectors have different lengths.
pub fn weighted_average_params(
    checkpoints: &[ParamSnapshot],
    weights: &[f32],
) -> Result<Vec<f32>, InterpolationError> {
    if checkpoints.is_empty() {
        return Err(InterpolationError::EmptyCheckpointList);
    }
    if weights.len() != checkpoints.len() {
        return Err(InterpolationError::WeightCountMismatch {
            weights_len: weights.len(),
            n_checkpoints: checkpoints.len(),
        });
    }

    // Validate all weights are non-negative, reporting the offending index
    // and value rather than just the (still-valid-looking) sum.
    for (index, &w) in weights.iter().enumerate() {
        if w < 0.0 {
            return Err(InterpolationError::NegativeBlendWeight { index, value: w });
        }
    }

    let sum: f32 = weights.iter().sum();
    // Tolerance is intentionally tight: `weighted_average_params` applies
    // weights *as given* after renormalizing by this exact `sum`, so the
    // tolerance only bounds how far `sum` may be from 1.0 before we refuse
    // to guess the caller's intent — it does not bound any residual bias in
    // the output (renormalization below removes that regardless of how
    // close `sum` was to 1.0).
    if (sum - 1.0_f32).abs() > 0.01 {
        return Err(InterpolationError::InvalidBlendWeights { sum });
    }

    let param_len = checkpoints[0].params.len();
    if param_len == 0 {
        return Err(InterpolationError::EmptyParams);
    }

    // Validate all param vectors have same length
    for (idx, snap) in checkpoints.iter().enumerate().skip(1) {
        if snap.params.len() != param_len {
            return Err(InterpolationError::LengthMismatch {
                len_a: param_len,
                len_b: snap.params.len(),
            });
        }
        let _ = idx;
    }

    // Renormalize by the *actual* sum (e.g. 0.99 or 1.01, within the 0.01
    // tolerance above) so the applied weights always sum to exactly 1.0.
    // Without this, a sum of 0.99 would silently shrink every merged
    // parameter by 1% — a systematic multiplicative bias in the merged
    // model (and, for `uniform_average_params`, `1.0 / n` repeated `n`
    // times can itself drift from 1.0 in f32 for large `n`).
    let inv_sum = 1.0_f32 / sum;

    let mut result = vec![0.0f32; param_len];
    for (snap, &w) in checkpoints.iter().zip(weights.iter()) {
        let w = w * inv_sum;
        for (r, &p) in result.iter_mut().zip(snap.params.iter()) {
            *r += w * p;
        }
    }
    Ok(result)
}

/// Compute a uniform average of all checkpoint parameter vectors.
///
/// Equivalent to `model_soup` but takes `&[ParamSnapshot]` slice directly
/// and is the building block for `model_soup`.
///
/// # Errors
///
/// - [`InterpolationError::EmptyCheckpointList`] if no checkpoints are provided.
/// - [`InterpolationError::LengthMismatch`] if param vectors have different lengths.
pub fn uniform_average_params(
    checkpoints: &[ParamSnapshot],
) -> Result<Vec<f32>, InterpolationError> {
    if checkpoints.is_empty() {
        return Err(InterpolationError::EmptyCheckpointList);
    }
    let n = checkpoints.len();
    let weights = vec![1.0_f32 / n as f32; n];
    weighted_average_params(checkpoints, &weights)
}

/// Generate `n_steps` evenly-spaced interpolated parameter vectors from `a` to `b`.
///
/// `t` values: 0, 1/(n_steps-1), 2/(n_steps-1), ..., 1
/// Includes both endpoints.
///
/// # Errors
///
/// - [`InterpolationError::InvalidStepCount`] if `n_steps < 2`.
/// - [`InterpolationError::LengthMismatch`] if `a.len() != b.len()`.
pub fn interpolation_sequence(
    a: &[f32],
    b: &[f32],
    n_steps: usize,
) -> Result<Vec<Vec<f32>>, InterpolationError> {
    if n_steps < 2 {
        return Err(InterpolationError::InvalidStepCount { n: n_steps });
    }
    if a.len() != b.len() {
        return Err(InterpolationError::LengthMismatch {
            len_a: a.len(),
            len_b: b.len(),
        });
    }
    let mut result = Vec::with_capacity(n_steps);
    for i in 0..n_steps {
        let t = i as f32 / (n_steps - 1) as f32;
        result.push(lerp_params(a, b, t)?);
    }
    Ok(result)
}

/// [`interpolation_sequence`], honoring [`InterpolationConfig`]'s
/// quaternion-handling options via [`interpolate_with_config`] at each step.
/// See [`interpolate_with_config`] for exactly what the config controls.
///
/// # Errors
///
/// Same as [`interpolation_sequence`], plus
/// [`InterpolationError::InvalidQuaternionRegion`] under the same condition
/// as [`interpolate_with_config`].
pub fn interpolation_sequence_with_config(
    a: &[f32],
    b: &[f32],
    n_steps: usize,
    config: &InterpolationConfig,
) -> Result<Vec<Vec<f32>>, InterpolationError> {
    if n_steps < 2 {
        return Err(InterpolationError::InvalidStepCount { n: n_steps });
    }
    if a.len() != b.len() {
        return Err(InterpolationError::LengthMismatch {
            len_a: a.len(),
            len_b: b.len(),
        });
    }
    let mut result = Vec::with_capacity(n_steps);
    for i in 0..n_steps {
        let t = i as f32 / (n_steps - 1) as f32;
        result.push(interpolate_with_config(a, b, t, config)?);
    }
    Ok(result)
}

/// Build a [`CheckpointPath`] from an ordered sequence of snapshots.
///
/// Computes L2 distances between consecutive snapshots and total path length.
///
/// # Errors
///
/// - [`InterpolationError::EmptyCheckpointList`] if fewer than 1 snapshot provided.
/// - [`InterpolationError::LengthMismatch`] if consecutive snapshots have different param lengths.
pub fn build_checkpoint_path(
    snapshots: Vec<ParamSnapshot>,
) -> Result<CheckpointPath, InterpolationError> {
    if snapshots.is_empty() {
        return Err(InterpolationError::EmptyCheckpointList);
    }
    let mut step_sizes = Vec::with_capacity(snapshots.len().saturating_sub(1));
    for pair in snapshots.windows(2) {
        let dist = params_l2_distance(&pair[0].params, &pair[1].params)?;
        step_sizes.push(dist);
    }
    let total_length: f32 = step_sizes.iter().sum();
    Ok(CheckpointPath {
        snapshots,
        step_sizes,
        total_length,
    })
}

/// Interpolate along a [`CheckpointPath`] at parameter `t ∈ [0, 1]`.
///
/// `t = 0` returns the first snapshot's params; `t = 1` returns the last.
/// Values in between perform linear interpolation within the appropriate segment.
///
/// # Errors
///
/// - [`InterpolationError::InvalidT`] if `t` is not in [0, 1].
/// - [`InterpolationError::EmptyCheckpointList`] if the path has no snapshots.
pub fn interpolate_along_path(
    path: &CheckpointPath,
    t: f32,
) -> Result<Vec<f32>, InterpolationError> {
    if !(0.0..=1.0).contains(&t) {
        return Err(InterpolationError::InvalidT { t });
    }
    if path.snapshots.is_empty() {
        return Err(InterpolationError::EmptyCheckpointList);
    }
    // Edge cases
    if t <= 0.0 || path.snapshots.len() == 1 {
        return Ok(path.snapshots[0].params.clone());
    }
    if t >= 1.0 {
        return Ok(path.snapshots[path.snapshots.len() - 1].params.clone());
    }
    // Degenerate case: zero total length
    if path.total_length < 1e-30 {
        return Ok(path.snapshots[0].params.clone());
    }

    let target_dist = t * path.total_length;
    let mut cumulative = 0.0_f32;

    for (i, &seg_len) in path.step_sizes.iter().enumerate() {
        let seg_end = cumulative + seg_len;
        if target_dist <= seg_end {
            // We are within segment i (from snapshot i to snapshot i+1)
            let seg_t = if seg_len < 1e-30 {
                0.0
            } else {
                (target_dist - cumulative) / seg_len
            };
            return lerp_params(
                &path.snapshots[i].params,
                &path.snapshots[i + 1].params,
                seg_t,
            );
        }
        cumulative = seg_end;
    }

    // Should not reach here for t < 1.0, but return last snapshot as safety
    Ok(path.snapshots[path.snapshots.len() - 1].params.clone())
}

/// [`interpolate_along_path`], honoring [`InterpolationConfig`]'s
/// quaternion-handling options via [`interpolate_with_config`] for the
/// within-segment interpolation step. See [`interpolate_with_config`] for
/// exactly what the config controls.
///
/// # Errors
///
/// Same as [`interpolate_along_path`], plus
/// [`InterpolationError::InvalidQuaternionRegion`] under the same condition
/// as [`interpolate_with_config`].
pub fn interpolate_along_path_with_config(
    path: &CheckpointPath,
    t: f32,
    config: &InterpolationConfig,
) -> Result<Vec<f32>, InterpolationError> {
    if !(0.0..=1.0).contains(&t) {
        return Err(InterpolationError::InvalidT { t });
    }
    if path.snapshots.is_empty() {
        return Err(InterpolationError::EmptyCheckpointList);
    }
    // Edge cases
    if t <= 0.0 || path.snapshots.len() == 1 {
        return Ok(path.snapshots[0].params.clone());
    }
    if t >= 1.0 {
        return Ok(path.snapshots[path.snapshots.len() - 1].params.clone());
    }
    // Degenerate case: zero total length
    if path.total_length < 1e-30 {
        return Ok(path.snapshots[0].params.clone());
    }

    let target_dist = t * path.total_length;
    let mut cumulative = 0.0_f32;

    for (i, &seg_len) in path.step_sizes.iter().enumerate() {
        let seg_end = cumulative + seg_len;
        if target_dist <= seg_end {
            let seg_t = if seg_len < 1e-30 {
                0.0
            } else {
                (target_dist - cumulative) / seg_len
            };
            return interpolate_with_config(
                &path.snapshots[i].params,
                &path.snapshots[i + 1].params,
                seg_t,
                config,
            );
        }
        cumulative = seg_end;
    }

    Ok(path.snapshots[path.snapshots.len() - 1].params.clone())
}

/// Model Soups technique: uniform average of all checkpoint params.
///
/// Reference: "Model Soups: averaging weights of multiple fine-tuned models
/// improves accuracy without increasing inference time" (Wortsman et al., 2022).
///
/// # Errors
///
/// - [`InterpolationError::EmptyCheckpointList`] if no checkpoints provided.
/// - [`InterpolationError::LengthMismatch`] if param vectors have different lengths.
pub fn model_soup(checkpoints: &[ParamSnapshot]) -> Result<Vec<f32>, InterpolationError> {
    uniform_average_params(checkpoints)
}

/// Evaluate loss along the linear path from `a` to `b`.
///
/// Returns `n_eval` `(t, loss)` pairs with `t` values evenly spaced in [0, 1].
///
/// If `a` and `b` have different lengths, returns an empty vector.
pub fn linear_mode_connectivity(
    a: &[f32],
    b: &[f32],
    n_eval: usize,
    loss_fn: &dyn Fn(&[f32]) -> f32,
) -> Vec<(f32, f32)> {
    if a.len() != b.len() || n_eval == 0 {
        return Vec::new();
    }
    let steps = n_eval;
    let mut result = Vec::with_capacity(steps);
    for i in 0..steps {
        let t = if steps <= 1 {
            0.0_f32
        } else {
            i as f32 / (steps - 1) as f32
        };
        // lerp_params with valid t cannot fail when lengths match
        let params = lerp_params(a, b, t).unwrap_or_else(|_| a.to_vec());
        let loss = loss_fn(&params);
        result.push((t, loss));
    }
    result
}

/// Compute statistics about a parameter vector.
///
/// Returns zeros for all stats if the input is empty.
pub fn compute_interpolation_stats(params: &[f32]) -> InterpolationStats {
    if params.is_empty() {
        return InterpolationStats {
            param_dim: 0,
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            l2_norm: 0.0,
        };
    }
    let mean = params_mean(params);
    let std = params_std(params);
    let l2_norm = params_l2_norm(params);
    let mut min = params[0];
    let mut max = params[0];
    for &v in &params[1..] {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    InterpolationStats {
        param_dim: params.len(),
        mean,
        std,
        min,
        max,
        l2_norm,
    }
}

/// Find blend weights that minimize `loss_fn(weighted_average(checkpoints, weights))`.
///
/// Uses gradient descent on the probability simplex (weights sum to 1, each >= 0).
///
/// The `target` parameter is included for API compatibility; the optimization
/// loop uses only `loss_fn`.
///
/// Returns the optimized weight vector (length `checkpoints.len()`), matching
/// the function name and this doc comment. Use [`weighted_average_params`] (or
/// the `compute_blend` pattern shown in the tests) if you also need the
/// resulting blended parameters — call it with the returned weights.
///
/// # Errors
///
/// - [`InterpolationError::EmptyCheckpointList`] if no checkpoints provided.
/// - [`InterpolationError::EmptyParams`] if the checkpoints' parameter vectors are empty.
/// - [`InterpolationError::LengthMismatch`] if param vectors have different lengths.
pub fn find_optimal_blend(
    checkpoints: &[ParamSnapshot],
    _target: &[f32],
    loss_fn: &dyn Fn(&[f32]) -> f32,
    iters: usize,
) -> Result<Vec<f32>, InterpolationError> {
    if checkpoints.is_empty() {
        return Err(InterpolationError::EmptyCheckpointList);
    }

    let n = checkpoints.len();
    let param_len = checkpoints[0].params.len();
    if param_len == 0 {
        return Err(InterpolationError::EmptyParams);
    }

    // Validate all param vectors have the same length
    for snap in checkpoints.iter().skip(1) {
        if snap.params.len() != param_len {
            return Err(InterpolationError::LengthMismatch {
                len_a: param_len,
                len_b: snap.params.len(),
            });
        }
    }

    // Init: uniform weights
    let mut weights = vec![1.0_f32 / n as f32; n];

    let eps = 1e-4_f32;
    let lr = 0.01_f32;

    let compute_blend = |w: &[f32]| -> Vec<f32> {
        let mut blended = vec![0.0f32; param_len];
        for (snap, &wi) in checkpoints.iter().zip(w.iter()) {
            for (b, &p) in blended.iter_mut().zip(snap.params.iter()) {
                *b += wi * p;
            }
        }
        blended
    };

    for _ in 0..iters {
        let blended = compute_blend(&weights);
        let loss_base = loss_fn(&blended);

        let mut grad = vec![0.0f32; n];
        for i in 0..n {
            let old_wi = weights[i];
            weights[i] += eps;
            let blended_perturbed = compute_blend(&weights);
            let loss_perturbed = loss_fn(&blended_perturbed);
            grad[i] = (loss_perturbed - loss_base) / eps;
            weights[i] = old_wi;
        }

        // Update: gradient descent step
        for i in 0..n {
            weights[i] -= lr * grad[i];
        }

        // Project to simplex: clip negatives to 0, renormalize
        for w in weights.iter_mut() {
            if *w < 0.0 {
                *w = 0.0;
            }
        }
        let w_sum: f32 = weights.iter().sum();
        if w_sum < 1e-12 {
            // Fallback: uniform weights
            for w in weights.iter_mut() {
                *w = 1.0 / n as f32;
            }
        } else {
            for w in weights.iter_mut() {
                *w /= w_sum;
            }
        }
    }

    // Return the optimal blend *weights* — matching the function name and
    // rustdoc ("Find blend weights that minimize..."). The previous
    // implementation discarded `weights` (length `n`) here and returned
    // `compute_blend(&weights)` (length `param_len`) instead, so a caller
    // trusting the doc had no way to obtain the weights the function was
    // named for, and would silently misread arbitrary parameter values as
    // weights (or panic) if they indexed the result as if it had `n` entries.
    Ok(weights)
}

/// Quantize parameter values to `bits`-bit precision.
///
/// Simulates the effect of quantization by rounding to the nearest grid point
/// within the observed [min, max] range of the input.
///
/// For 8 bits: `levels = 255` grid points; each value is rounded to
/// the nearest multiple of `(max - min) / levels`.
///
/// `bits` is clamped to `1..=32` before use: `bits == 0` would make
/// `levels == 0` and every output NaN (division by zero, silently — under a
/// name that promises quantized values), and `bits >= 64` would overflow the
/// `1u64 << bits` shift (a panic in debug builds). f32 only has 24 bits of
/// mantissa precision, so 32 levels-bits is already far more than a float
/// can meaningfully resolve; clamping there is a safe, generous ceiling.
///
/// Returns the original values unchanged if min == max.
pub fn quantize_params(params: &[f32], bits: u8) -> Vec<f32> {
    if params.is_empty() {
        return Vec::new();
    }
    let bits = bits.clamp(1, 32);
    let levels = (1u64 << bits as u64) - 1;
    let mut min = params[0];
    let mut max = params[0];
    for &v in &params[1..] {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    let range = max - min;
    if range < 1e-30 {
        return params.to_vec();
    }
    let step = range / levels as f32;
    params
        .iter()
        .map(|&v| {
            let quantized_idx = ((v - min) / step).round();
            min + quantized_idx * step
        })
        .collect()
}

/// Mean absolute error between original and quantized parameter vectors.
///
/// # Errors
///
/// - [`InterpolationError::LengthMismatch`] if lengths differ.
pub fn dequantize_error(original: &[f32], quantized: &[f32]) -> Result<f32, InterpolationError> {
    if original.len() != quantized.len() {
        return Err(InterpolationError::LengthMismatch {
            len_a: original.len(),
            len_b: quantized.len(),
        });
    }
    if original.is_empty() {
        return Ok(0.0);
    }
    let mae: f32 = original
        .iter()
        .zip(quantized.iter())
        .map(|(&o, &q)| (o - q).abs())
        .sum::<f32>()
        / original.len() as f32;
    Ok(mae)
}

/// Format a human-readable summary of a [`CheckpointPath`].
///
/// Returns a string like:
/// `"CheckpointPath[N snapshots]: total_length=X.XXXX, step_sizes=[...]"`
pub fn format_checkpoint_path(path: &CheckpointPath) -> String {
    let n = path.snapshots.len();
    let sizes: Vec<String> = path
        .step_sizes
        .iter()
        .map(|s| format!("{:.4}", s))
        .collect();
    format!(
        "CheckpointPath[{} snapshots]: total_length={:.4}, step_sizes=[{}]",
        n,
        path.total_length,
        sizes.join(", ")
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "checkpoint_interpolation/tests.rs"]
mod tests;
