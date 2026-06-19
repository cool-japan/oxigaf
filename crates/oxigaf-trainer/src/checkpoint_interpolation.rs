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
#[cfg(test)]
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
#[cfg(test)]
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

/// Compute a weighted average of checkpoint parameter vectors.
///
/// `result[i] = Σ weights[j] * checkpoints[j].params[i]`
///
/// # Errors
///
/// - [`InterpolationError::EmptyCheckpointList`] if no checkpoints are provided.
/// - [`InterpolationError::WeightCountMismatch`] if weight count differs.
/// - [`InterpolationError::InvalidBlendWeights`] if weights don't sum to ~1.0.
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

    let sum: f32 = weights.iter().sum();
    if (sum - 1.0_f32).abs() > 0.01 {
        return Err(InterpolationError::InvalidBlendWeights { sum });
    }

    // Validate all weights are non-negative
    for &w in weights {
        if w < 0.0 {
            return Err(InterpolationError::InvalidBlendWeights { sum });
        }
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

    let mut result = vec![0.0f32; param_len];
    for (snap, &w) in checkpoints.iter().zip(weights.iter()) {
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
/// # Errors
///
/// - [`InterpolationError::EmptyCheckpointList`] if no checkpoints provided.
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

    // Return the optimal blended params
    let result = compute_blend(&weights);
    Ok(result)
}

/// Quantize parameter values to `bits`-bit precision.
///
/// Simulates the effect of quantization by rounding to the nearest grid point
/// within the observed [min, max] range of the input.
///
/// For 8 bits: `levels = 255` grid points; each value is rounded to
/// the nearest multiple of `(max - min) / levels`.
///
/// Returns the original values unchanged if min == max.
pub fn quantize_params(params: &[f32], bits: u8) -> Vec<f32> {
    if params.is_empty() {
        return Vec::new();
    }
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
mod tests {
    use super::*;

    fn make_snap(name: &str, step: usize, params: Vec<f32>) -> ParamSnapshot {
        ParamSnapshot {
            name: name.to_string(),
            step,
            params,
        }
    }

    // ── lerp_params ──────────────────────────────────────────────────────────

    #[test]
    fn test_lerp_t0_returns_a() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let r = lerp_params(&a, &b, 0.0).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!((r[1] - 2.0).abs() < 1e-6);
        assert!((r[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_lerp_t1_returns_b() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let r = lerp_params(&a, &b, 1.0).unwrap();
        assert!((r[0] - 4.0).abs() < 1e-6);
        assert!((r[1] - 5.0).abs() < 1e-6);
        assert!((r[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_lerp_t05_midpoint() {
        let a = vec![0.0, 0.0];
        let b = vec![2.0, 4.0];
        let r = lerp_params(&a, &b, 0.5).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!((r[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_lerp_invalid_t_negative() {
        let a = vec![1.0];
        let b = vec![2.0];
        let err = lerp_params(&a, &b, -0.1).unwrap_err();
        assert!(matches!(err, InterpolationError::InvalidT { .. }));
    }

    #[test]
    fn test_lerp_invalid_t_gt1() {
        let a = vec![1.0];
        let b = vec![2.0];
        let err = lerp_params(&a, &b, 1.1).unwrap_err();
        assert!(matches!(err, InterpolationError::InvalidT { .. }));
    }

    #[test]
    fn test_lerp_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let err = lerp_params(&a, &b, 0.5).unwrap_err();
        assert!(matches!(err, InterpolationError::LengthMismatch { .. }));
    }

    // ── slerp_quat ───────────────────────────────────────────────────────────

    #[test]
    fn test_slerp_quat_t0_returns_a() {
        let qa = [0.0, 0.0, 0.0, 1.0f32];
        let qb = [0.0, 1.0, 0.0, 0.0f32];
        let r = slerp_quat(&qa, &qb, 0.0);
        assert!((r[3] - 1.0).abs() < 1e-5, "t=0 should return qa");
    }

    #[test]
    fn test_slerp_quat_t1_returns_b() {
        let qa = [0.0, 0.0, 0.0, 1.0f32];
        let qb = [0.0, 1.0, 0.0, 0.0f32];
        let r = slerp_quat(&qa, &qb, 1.0);
        assert!((r[1] - 1.0).abs() < 1e-5, "t=1 should return qb");
    }

    #[test]
    fn test_slerp_quat_identity_same() {
        let q = [0.0, 0.0, 0.0, 1.0f32];
        let r = slerp_quat(&q, &q, 0.5);
        let norm_sq: f32 = r.iter().map(|v| v * v).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-5,
            "Result must be unit quaternion"
        );
        assert!((r[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_slerp_quat_180_rotation() {
        // 90-degree rotation around Z: (0, 0, sin(45°), cos(45°))
        let qa = [0.0, 0.0, 0.0, 1.0f32];
        let qb = [
            0.0,
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        ];
        let r = slerp_quat(&qa, &qb, 0.5);
        let norm_sq: f32 = r.iter().map(|v| v * v).sum();
        assert!((norm_sq - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_slerp_quat_result_normalized() {
        let qa = [0.5, 0.5, 0.5, 0.5f32];
        let qb = [0.0, 0.0, 0.0, 1.0f32];
        let r = slerp_quat(&qa, &qb, 0.3);
        let norm_sq: f32 = r.iter().map(|v| v * v).sum();
        assert!((norm_sq - 1.0).abs() < 1e-5);
    }

    // ── normalize_quaternion ─────────────────────────────────────────────────

    #[test]
    fn test_normalize_quaternion_unit_norm() {
        let mut q = [2.0f32, 0.0, 0.0, 0.0];
        normalize_quaternion(&mut q);
        let norm_sq: f32 = q.iter().map(|v| v * v).sum();
        assert!((norm_sq - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_quaternion_already_unit() {
        let mut q = [0.0f32, 0.0, 0.0, 1.0];
        normalize_quaternion(&mut q);
        assert!((q[3] - 1.0).abs() < 1e-6);
    }

    // ── params_l2_distance ───────────────────────────────────────────────────

    #[test]
    fn test_params_l2_distance_known() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let d = params_l2_distance(&a, &b).unwrap();
        assert!((d - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_params_l2_distance_same_is_zero() {
        let a = vec![1.0, 2.0, 3.0];
        let d = params_l2_distance(&a, &a).unwrap();
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn test_params_l2_distance_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        let err = params_l2_distance(&a, &b).unwrap_err();
        assert!(matches!(err, InterpolationError::LengthMismatch { .. }));
    }

    // ── params_cosine_similarity ─────────────────────────────────────────────

    #[test]
    fn test_cosine_similarity_same_is_one() {
        let a = vec![1.0, 2.0, 3.0];
        let s = params_cosine_similarity(&a, &a).unwrap();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_opposite_is_minus_one() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let s = params_cosine_similarity(&a, &b).unwrap();
        assert!((s - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let s = params_cosine_similarity(&a, &b).unwrap();
        assert!(s.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        let s = params_cosine_similarity(&a, &b).unwrap();
        assert_eq!(s, 0.0);
    }

    // ── params_l2_norm ───────────────────────────────────────────────────────

    #[test]
    fn test_params_l2_norm_known() {
        let v = vec![3.0, 4.0];
        assert!((params_l2_norm(&v) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_params_l2_norm_unit() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((params_l2_norm(&v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_params_l2_norm_zero() {
        let v = vec![0.0, 0.0, 0.0];
        assert!(params_l2_norm(&v).abs() < 1e-6);
    }

    // ── params_mean ──────────────────────────────────────────────────────────

    #[test]
    fn test_params_mean_known() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert!((params_mean(&v) - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_params_mean_empty() {
        assert_eq!(params_mean(&[]), 0.0);
    }

    // ── params_std ───────────────────────────────────────────────────────────

    #[test]
    fn test_params_std_known() {
        let v = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = params_std(&v);
        // Population std ≈ 2.0
        assert!((s - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_params_std_single_element() {
        assert_eq!(params_std(&[42.0]), 0.0);
    }

    #[test]
    fn test_params_std_constant() {
        let v = vec![5.0; 10];
        assert!(params_std(&v).abs() < 1e-6);
    }

    // ── linear_interpolate ───────────────────────────────────────────────────

    #[test]
    fn test_linear_interpolate_t05() {
        let a = vec![0.0, 10.0];
        let b = vec![10.0, 0.0];
        let r = linear_interpolate(&a, &b, 0.5).unwrap();
        assert!((r[0] - 5.0).abs() < 1e-5);
        assert!((r[1] - 5.0).abs() < 1e-5);
    }

    // ── weighted_average_params ──────────────────────────────────────────────

    #[test]
    fn test_weighted_average_single_weight_one() {
        let snap = make_snap("a", 0, vec![1.0, 2.0, 3.0]);
        let result = weighted_average_params(&[snap], &[1.0]).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 2.0).abs() < 1e-5);
        assert!((result[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_weighted_average_two_checkpoints_equal() {
        let a = make_snap("a", 0, vec![0.0, 0.0]);
        let b = make_snap("b", 1, vec![2.0, 4.0]);
        let r = weighted_average_params(&[a, b], &[0.5, 0.5]).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-5);
        assert!((r[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_weighted_average_empty_error() {
        let err = weighted_average_params(&[], &[]).unwrap_err();
        assert!(matches!(err, InterpolationError::EmptyCheckpointList));
    }

    #[test]
    fn test_weighted_average_weight_mismatch_error() {
        let a = make_snap("a", 0, vec![1.0]);
        let err = weighted_average_params(&[a], &[0.5, 0.5]).unwrap_err();
        assert!(matches!(
            err,
            InterpolationError::WeightCountMismatch { .. }
        ));
    }

    #[test]
    fn test_weighted_average_bad_sum_error() {
        let a = make_snap("a", 0, vec![1.0]);
        let err = weighted_average_params(&[a], &[0.5]).unwrap_err();
        assert!(matches!(
            err,
            InterpolationError::InvalidBlendWeights { .. }
        ));
    }

    #[test]
    fn test_weighted_average_negative_weight_error() {
        let a = make_snap("a", 0, vec![1.0]);
        let b = make_snap("b", 1, vec![2.0]);
        // sum = 1.0 but one weight is negative
        let err = weighted_average_params(&[a, b], &[-0.2, 1.2]).unwrap_err();
        assert!(matches!(
            err,
            InterpolationError::InvalidBlendWeights { .. }
        ));
    }

    // ── uniform_average_params ───────────────────────────────────────────────

    #[test]
    fn test_uniform_average_smoke() {
        let a = make_snap("a", 0, vec![0.0, 0.0]);
        let b = make_snap("b", 1, vec![2.0, 4.0]);
        let r = uniform_average_params(&[a, b]).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_uniform_average_three() {
        let a = make_snap("a", 0, vec![0.0]);
        let b = make_snap("b", 1, vec![3.0]);
        let c = make_snap("c", 2, vec![6.0]);
        let r = uniform_average_params(&[a, b, c]).unwrap();
        assert!((r[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_uniform_average_empty_error() {
        let err = uniform_average_params(&[]).unwrap_err();
        assert!(matches!(err, InterpolationError::EmptyCheckpointList));
    }

    // ── interpolation_sequence ───────────────────────────────────────────────

    #[test]
    fn test_interpolation_sequence_n2_endpoints() {
        let a = vec![0.0];
        let b = vec![1.0];
        let seq = interpolation_sequence(&a, &b, 2).unwrap();
        assert_eq!(seq.len(), 2);
        assert!((seq[0][0] - 0.0).abs() < 1e-5);
        assert!((seq[1][0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_interpolation_sequence_n5_correct_length() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        let seq = interpolation_sequence(&a, &b, 5).unwrap();
        assert_eq!(seq.len(), 5);
        // Check t=0.5 is at index 2
        assert!((seq[2][0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_interpolation_sequence_invalid_steps() {
        let a = vec![0.0];
        let b = vec![1.0];
        let err = interpolation_sequence(&a, &b, 1).unwrap_err();
        assert!(matches!(err, InterpolationError::InvalidStepCount { .. }));
    }

    #[test]
    fn test_interpolation_sequence_length_mismatch() {
        let a = vec![0.0, 1.0];
        let b = vec![1.0];
        let err = interpolation_sequence(&a, &b, 3).unwrap_err();
        assert!(matches!(err, InterpolationError::LengthMismatch { .. }));
    }

    // ── build_checkpoint_path ────────────────────────────────────────────────

    #[test]
    fn test_build_checkpoint_path_two_snapshots() {
        let a = make_snap("a", 0, vec![0.0, 0.0]);
        let b = make_snap("b", 1, vec![3.0, 4.0]);
        let path = build_checkpoint_path(vec![a, b]).unwrap();
        assert_eq!(path.snapshots.len(), 2);
        assert_eq!(path.step_sizes.len(), 1);
        assert!((path.step_sizes[0] - 5.0).abs() < 1e-4);
        assert!((path.total_length - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_build_checkpoint_path_empty_error() {
        let err = build_checkpoint_path(vec![]).unwrap_err();
        assert!(matches!(err, InterpolationError::EmptyCheckpointList));
    }

    #[test]
    fn test_build_checkpoint_path_single_snapshot() {
        // Single snapshot: no step sizes, total_length = 0
        let a = make_snap("a", 0, vec![1.0, 2.0]);
        let path = build_checkpoint_path(vec![a]).unwrap();
        assert_eq!(path.snapshots.len(), 1);
        assert_eq!(path.step_sizes.len(), 0);
        assert_eq!(path.total_length, 0.0);
    }

    #[test]
    fn test_build_checkpoint_path_three_snapshots() {
        let a = make_snap("a", 0, vec![0.0, 0.0]);
        let b = make_snap("b", 1, vec![3.0, 4.0]);
        let c = make_snap("c", 2, vec![3.0, 4.0]);
        let path = build_checkpoint_path(vec![a, b, c]).unwrap();
        assert_eq!(path.step_sizes.len(), 2);
        assert!((path.step_sizes[0] - 5.0).abs() < 1e-4);
        assert!(path.step_sizes[1].abs() < 1e-4);
    }

    // ── interpolate_along_path ───────────────────────────────────────────────

    #[test]
    fn test_interpolate_along_path_t0_first() {
        let a = make_snap("a", 0, vec![0.0, 0.0]);
        let b = make_snap("b", 1, vec![1.0, 1.0]);
        let path = build_checkpoint_path(vec![a, b]).unwrap();
        let r = interpolate_along_path(&path, 0.0).unwrap();
        assert!((r[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_interpolate_along_path_t1_last() {
        let a = make_snap("a", 0, vec![0.0, 0.0]);
        let b = make_snap("b", 1, vec![1.0, 1.0]);
        let path = build_checkpoint_path(vec![a, b]).unwrap();
        let r = interpolate_along_path(&path, 1.0).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_interpolate_along_path_t05_midpoint() {
        let a = make_snap("a", 0, vec![0.0, 0.0]);
        let b = make_snap("b", 1, vec![2.0, 4.0]);
        let path = build_checkpoint_path(vec![a, b]).unwrap();
        let r = interpolate_along_path(&path, 0.5).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-4);
        assert!((r[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_interpolate_along_path_invalid_t() {
        let a = make_snap("a", 0, vec![0.0]);
        let path = build_checkpoint_path(vec![a]).unwrap();
        let err = interpolate_along_path(&path, 1.5).unwrap_err();
        assert!(matches!(err, InterpolationError::InvalidT { .. }));
    }

    // ── model_soup ───────────────────────────────────────────────────────────

    #[test]
    fn test_model_soup_identical_checkpoints() {
        let a = make_snap("a", 0, vec![1.0, 2.0, 3.0]);
        let b = make_snap("b", 1, vec![1.0, 2.0, 3.0]);
        let c = make_snap("c", 2, vec![1.0, 2.0, 3.0]);
        let r = model_soup(&[a, b, c]).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-5);
        assert!((r[1] - 2.0).abs() < 1e-5);
        assert!((r[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_model_soup_empty_error() {
        let err = model_soup(&[]).unwrap_err();
        assert!(matches!(err, InterpolationError::EmptyCheckpointList));
    }

    #[test]
    fn test_model_soup_two_diverse() {
        let a = make_snap("a", 0, vec![0.0, 0.0]);
        let b = make_snap("b", 1, vec![4.0, 8.0]);
        let r = model_soup(&[a, b]).unwrap();
        assert!((r[0] - 2.0).abs() < 1e-5);
        assert!((r[1] - 4.0).abs() < 1e-5);
    }

    // ── linear_mode_connectivity ─────────────────────────────────────────────

    #[test]
    fn test_linear_mode_connectivity_correct_length() {
        let a = vec![0.0; 8];
        let b = vec![1.0; 8];
        let pairs = linear_mode_connectivity(&a, &b, 5, &|p: &[f32]| p.iter().sum::<f32>());
        assert_eq!(pairs.len(), 5);
    }

    #[test]
    fn test_linear_mode_connectivity_t_in_range() {
        let a = vec![0.0; 4];
        let b = vec![1.0; 4];
        let pairs = linear_mode_connectivity(&a, &b, 10, &|_: &[f32]| 0.0);
        for (t, _loss) in &pairs {
            assert!(*t >= 0.0 && *t <= 1.0);
        }
    }

    #[test]
    fn test_linear_mode_connectivity_mismatch_returns_empty() {
        let a = vec![0.0, 1.0];
        let b = vec![1.0];
        let pairs = linear_mode_connectivity(&a, &b, 5, &|_: &[f32]| 0.0);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_linear_mode_connectivity_loss_monotone() {
        // Loss = sum of params; at t=0 sum=0, at t=1 sum=8
        let a = vec![0.0; 8];
        let b = vec![1.0; 8];
        let pairs = linear_mode_connectivity(&a, &b, 5, &|p: &[f32]| p.iter().sum::<f32>());
        // Losses should be monotone increasing
        for w in pairs.windows(2) {
            assert!(w[1].1 >= w[0].1 - 1e-5);
        }
    }

    // ── compute_interpolation_stats ──────────────────────────────────────────

    #[test]
    fn test_interpolation_stats_correct_min_max() {
        let params = vec![1.0, 3.0, -2.0, 5.0];
        let stats = compute_interpolation_stats(&params);
        assert!((stats.min - (-2.0)).abs() < 1e-5);
        assert!((stats.max - 5.0).abs() < 1e-5);
        assert_eq!(stats.param_dim, 4);
    }

    #[test]
    fn test_interpolation_stats_correct_mean() {
        let params = vec![1.0, 2.0, 3.0, 4.0];
        let stats = compute_interpolation_stats(&params);
        assert!((stats.mean - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_interpolation_stats_empty() {
        let stats = compute_interpolation_stats(&[]);
        assert_eq!(stats.param_dim, 0);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn test_interpolation_stats_l2_norm() {
        let params = vec![3.0, 4.0];
        let stats = compute_interpolation_stats(&params);
        assert!((stats.l2_norm - 5.0).abs() < 1e-5);
    }

    // ── find_optimal_blend ───────────────────────────────────────────────────

    #[test]
    fn test_find_optimal_blend_identity_loss() {
        // Loss is constant — any weights work; just verify no error
        let a = make_snap("a", 0, vec![1.0, 0.0]);
        let b = make_snap("b", 1, vec![0.0, 1.0]);
        let result = find_optimal_blend(&[a, b], &[0.5, 0.5], &|_: &[f32]| 0.0, 10);
        assert!(result.is_ok());
        let blended = result.unwrap();
        assert_eq!(blended.len(), 2);
    }

    #[test]
    fn test_find_optimal_blend_empty_error() {
        let err = find_optimal_blend(&[], &[], &|_: &[f32]| 0.0, 10).unwrap_err();
        assert!(matches!(err, InterpolationError::EmptyCheckpointList));
    }

    #[test]
    fn test_find_optimal_blend_single_checkpoint() {
        let a = make_snap("a", 0, vec![1.0, 2.0, 3.0]);
        let result = find_optimal_blend(
            &[a],
            &[],
            &|p: &[f32]| {
                // Minimize L2 norm
                p.iter().map(|x| x * x).sum::<f32>()
            },
            5,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_find_optimal_blend_converges_toward_minimum() {
        // Loss = distance from target [0,0,0,0]
        // Checkpoint a=[2,2,2,2], b=[0,0,0,0]
        // Optimal blend should prefer checkpoint b (weight close to 1.0 for b)
        let a = make_snap("a", 0, vec![2.0, 2.0, 2.0, 2.0]);
        let b = make_snap("b", 1, vec![0.0, 0.0, 0.0, 0.0]);
        let target = vec![0.0, 0.0, 0.0, 0.0];
        let result = find_optimal_blend(
            &[a, b],
            &target,
            &|p: &[f32]| p.iter().map(|x| x * x).sum::<f32>(),
            50,
        )
        .unwrap();
        // Blended result should be close to 0
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm < 2.0, "Blend should reduce loss, got norm={}", norm);
    }

    // ── quantize_params ──────────────────────────────────────────────────────

    #[test]
    fn test_quantize_8bit_stays_within_range() {
        let params = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let q = quantize_params(&params, 8);
        assert_eq!(q.len(), params.len());
        for &v in &q {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_quantize_empty() {
        let q = quantize_params(&[], 8);
        assert!(q.is_empty());
    }

    #[test]
    fn test_quantize_constant_unchanged() {
        let params = vec![5.0, 5.0, 5.0];
        let q = quantize_params(&params, 8);
        assert!((q[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_quantize_endpoints_preserved() {
        let params = vec![0.0, 1.0];
        let q = quantize_params(&params, 8);
        assert!((q[0] - 0.0).abs() < 1e-4);
        assert!((q[1] - 1.0).abs() < 1e-4);
    }

    // ── dequantize_error ─────────────────────────────────────────────────────

    #[test]
    fn test_dequantize_error_same_is_zero() {
        let params = vec![1.0, 2.0, 3.0];
        let err = dequantize_error(&params, &params).unwrap();
        assert!(err.abs() < 1e-6);
    }

    #[test]
    fn test_dequantize_error_positive() {
        let original = vec![0.0, 0.5, 1.0];
        let quantized = vec![0.0, 0.502, 1.0];
        let err = dequantize_error(&original, &quantized).unwrap();
        assert!(err > 0.0);
    }

    #[test]
    fn test_dequantize_error_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        let err = dequantize_error(&a, &b).unwrap_err();
        assert!(matches!(err, InterpolationError::LengthMismatch { .. }));
    }

    #[test]
    fn test_dequantize_error_empty_is_zero() {
        let err = dequantize_error(&[], &[]).unwrap();
        assert_eq!(err, 0.0);
    }

    // ── format_checkpoint_path ───────────────────────────────────────────────

    #[test]
    fn test_format_checkpoint_path_non_empty() {
        let a = make_snap("a", 0, vec![0.0]);
        let b = make_snap("b", 1, vec![1.0]);
        let path = build_checkpoint_path(vec![a, b]).unwrap();
        let s = format_checkpoint_path(&path);
        assert!(!s.is_empty());
        assert!(s.contains("2 snapshots"));
        assert!(s.contains("total_length"));
    }

    #[test]
    fn test_format_checkpoint_path_single() {
        let a = make_snap("only", 0, vec![1.0, 2.0]);
        let path = build_checkpoint_path(vec![a]).unwrap();
        let s = format_checkpoint_path(&path);
        assert!(s.contains("1 snapshots"));
        assert!(s.contains("0.0000"));
    }

    // ── Error variant tests ──────────────────────────────────────────────────

    #[test]
    fn test_error_display_length_mismatch() {
        let err = InterpolationError::LengthMismatch { len_a: 3, len_b: 5 };
        let msg = err.to_string();
        assert!(msg.contains("3") && msg.contains("5"));
    }

    #[test]
    fn test_error_display_invalid_t() {
        let err = InterpolationError::InvalidT { t: 1.5 };
        let msg = err.to_string();
        assert!(msg.contains("1.5"));
    }

    #[test]
    fn test_error_display_invalid_step_count() {
        let err = InterpolationError::InvalidStepCount { n: 1 };
        let msg = err.to_string();
        assert!(msg.contains("1"));
    }

    #[test]
    fn test_error_display_empty_checkpoint_list() {
        let err = InterpolationError::EmptyCheckpointList;
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn test_error_display_weight_count_mismatch() {
        let err = InterpolationError::WeightCountMismatch {
            weights_len: 2,
            n_checkpoints: 3,
        };
        let msg = err.to_string();
        assert!(msg.contains("2") && msg.contains("3"));
    }

    // ── Struct field tests ───────────────────────────────────────────────────

    #[test]
    fn test_param_snapshot_fields() {
        let snap = ParamSnapshot {
            name: "test_snap".to_string(),
            step: 500,
            params: vec![1.0, 2.0],
        };
        assert_eq!(snap.name, "test_snap");
        assert_eq!(snap.step, 500);
        assert_eq!(snap.params.len(), 2);
    }

    #[test]
    fn test_interpolation_config_default() {
        let cfg = InterpolationConfig::default();
        assert!(!cfg.use_slerp_for_rotations);
        assert!(cfg.normalize_rotations);
        assert_eq!(cfg.n_quaternion_params, 0);
    }

    #[test]
    fn test_checkpoint_path_fields() {
        let a = make_snap("a", 0, vec![0.0, 0.0]);
        let b = make_snap("b", 1, vec![1.0, 0.0]);
        let path = build_checkpoint_path(vec![a, b]).unwrap();
        assert_eq!(path.snapshots.len(), 2);
        assert_eq!(path.step_sizes.len(), 1);
        assert!(path.total_length > 0.0);
    }

    #[test]
    fn test_interpolation_stats_fields() {
        let stats = compute_interpolation_stats(&[1.0, 2.0, 3.0]);
        assert_eq!(stats.param_dim, 3);
        assert!(stats.mean > 0.0);
        assert!(stats.std >= 0.0);
        assert!(stats.l2_norm > 0.0);
    }

    // ── Edge cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_interpolate_along_path_degenerate_zero_length() {
        // All snapshots identical → total_length = 0 → returns first snapshot
        let a = make_snap("a", 0, vec![3.0, 7.0]);
        let b = make_snap("b", 1, vec![3.0, 7.0]);
        let path = build_checkpoint_path(vec![a, b]).unwrap();
        let r = interpolate_along_path(&path, 0.5).unwrap();
        assert!((r[0] - 3.0).abs() < 1e-5);
        assert!((r[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_interpolation_sequence_all_same() {
        let a = vec![1.0, 1.0];
        let seq = interpolation_sequence(&a, &a, 5).unwrap();
        for v in &seq {
            assert!((v[0] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_slerp_quat_antiparallel_short_path() {
        // When qa and qb are antiparallel, dot < 0, qb should be negated
        let qa = [0.0, 0.0, 0.0, 1.0f32];
        let qb = [0.0, 0.0, 0.0, -1.0f32]; // antipodal
        let r = slerp_quat(&qa, &qb, 0.0);
        // t=0 should return qa (short path ensures we start at qa)
        let norm_sq: f32 = r.iter().map(|v| v * v).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-5,
            "result must be unit quaternion"
        );
    }
}
