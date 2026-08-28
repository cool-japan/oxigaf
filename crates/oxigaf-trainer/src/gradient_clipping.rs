//! Gradient clipping utilities for stable training of 3D Gaussian Splatting models.
//!
//! Provides four clipping strategies (global norm, per-group norm, value clamp, adaptive),
//! a stateful [`GradientClipper`] that tracks EMA of norms, gradient health diagnostics,
//! and a sliding-window norm history for adaptive thresholding.
//!
//! # Quick start
//! ```rust
//! use oxigaf_trainer::gradient_clipping::{GradientClipper, ClipMode};
//!
//! let mut clipper = GradientClipper::new(ClipMode::GlobalNorm { max_norm: 1.0 }).unwrap();
//! let mut grads = vec![vec![3.0_f32, 4.0]]; // norm = 5 → will be clipped to 1
//! let stats = clipper.step(&mut grads).unwrap();
//! assert!(stats.was_clipped);
//! ```

use std::collections::VecDeque;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by gradient-clipping operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ClipError {
    /// The threshold parameter is non-positive (max_norm ≤ 0, max_val ≤ 0, etc.).
    #[error("Invalid threshold: {0}")]
    InvalidThreshold(String),

    /// The gradient list is empty (no parameter groups provided).
    #[error("No gradient groups provided (empty gradients)")]
    EmptyGradients,

    /// A gradient slice has the wrong number of elements.
    #[error("Length mismatch: expected {expected} elements, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },
}

// ─────────────────────────────────────────────────────────────────────────────
// ClipMode
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy used by [`GradientClipper`] (and the standalone clip functions).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClipMode {
    /// Clip global gradient norm across all parameters.
    GlobalNorm { max_norm: f32 },
    /// Clip each parameter group's gradient norm independently.
    PerGroupNorm { max_norm: f32 },
    /// Clip individual gradient values to `[-max_val, max_val]`.
    ValueClip { max_val: f32 },
    /// Adaptive: compute EMA of gradient norm, clip to `ema_norm * clip_factor`.
    Adaptive { ema_factor: f32, clip_factor: f32 },
}

impl ClipMode {
    /// Validate mode-specific parameters, returning an error if any are invalid.
    fn validate(&self) -> Result<(), ClipError> {
        match *self {
            ClipMode::GlobalNorm { max_norm } => {
                if max_norm <= 0.0 {
                    return Err(ClipError::InvalidThreshold(format!(
                        "GlobalNorm max_norm must be > 0, got {max_norm}"
                    )));
                }
            }
            ClipMode::PerGroupNorm { max_norm } => {
                if max_norm <= 0.0 {
                    return Err(ClipError::InvalidThreshold(format!(
                        "PerGroupNorm max_norm must be > 0, got {max_norm}"
                    )));
                }
            }
            ClipMode::ValueClip { max_val } => {
                if max_val <= 0.0 {
                    return Err(ClipError::InvalidThreshold(format!(
                        "ValueClip max_val must be > 0, got {max_val}"
                    )));
                }
            }
            ClipMode::Adaptive {
                ema_factor,
                clip_factor,
            } => {
                if !(0.0 < ema_factor && ema_factor < 1.0) {
                    return Err(ClipError::InvalidThreshold(format!(
                        "Adaptive ema_factor must be in (0, 1), got {ema_factor}"
                    )));
                }
                if clip_factor <= 0.0 {
                    return Err(ClipError::InvalidThreshold(format!(
                        "Adaptive clip_factor must be > 0, got {clip_factor}"
                    )));
                }
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone utility functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the L2 norm of a flat gradient slice.
///
/// Returns `0.0` for an empty slice.
#[inline]
pub fn l2_norm(grad: &[f32]) -> f32 {
    grad.iter().map(|&v| v * v).sum::<f32>().sqrt()
}

/// Compute the global L2 norm across multiple gradient tensors.
///
/// Equivalent to flattening all tensors into a single vector and computing
/// its L2 norm. Returns `0.0` when `gradients` is empty or all slices are empty.
pub fn global_gradient_norm(gradients: &[Vec<f32>]) -> f32 {
    let sum_sq: f32 = gradients
        .iter()
        .flat_map(|g| g.iter())
        .map(|&v| v * v)
        .sum();
    sum_sq.sqrt()
}

/// Clip a list of gradient tensors so their combined L2 norm is at most `max_norm`.
///
/// If `total_norm <= max_norm`, the gradients are returned unchanged.
/// Otherwise all gradients are scaled by `max_norm / total_norm`.
///
/// Returns the **original** total norm (before any clipping).
///
/// # Errors
/// - [`ClipError::EmptyGradients`] if `gradients` is empty.
/// - [`ClipError::InvalidThreshold`] if `max_norm <= 0`.
pub fn clip_by_global_norm(gradients: &mut [Vec<f32>], max_norm: f32) -> Result<f32, ClipError> {
    if max_norm <= 0.0 {
        return Err(ClipError::InvalidThreshold(format!(
            "max_norm must be > 0, got {max_norm}"
        )));
    }
    if gradients.is_empty() {
        return Err(ClipError::EmptyGradients);
    }

    let total_norm = global_gradient_norm(gradients);
    if total_norm > max_norm {
        let scale = max_norm / total_norm;
        for group in gradients.iter_mut() {
            for v in group.iter_mut() {
                *v *= scale;
            }
        }
    }
    Ok(total_norm)
}

/// Clip each gradient tensor independently by `max_norm`.
///
/// Returns a `Vec` of the original norm for each group (one per group).
///
/// # Errors
/// - [`ClipError::EmptyGradients`] if `gradients` is empty.
/// - [`ClipError::InvalidThreshold`] if `max_norm <= 0`.
pub fn clip_by_per_group_norm(
    gradients: &mut [Vec<f32>],
    max_norm: f32,
) -> Result<Vec<f32>, ClipError> {
    if max_norm <= 0.0 {
        return Err(ClipError::InvalidThreshold(format!(
            "max_norm must be > 0, got {max_norm}"
        )));
    }
    if gradients.is_empty() {
        return Err(ClipError::EmptyGradients);
    }

    let mut original_norms = Vec::with_capacity(gradients.len());
    for group in gradients.iter_mut() {
        let norm = l2_norm(group);
        original_norms.push(norm);
        if norm > max_norm {
            let scale = max_norm / norm;
            for v in group.iter_mut() {
                *v *= scale;
            }
        }
    }
    Ok(original_norms)
}

/// Scale `gradients` in place by `max_norm / known_norm` if `known_norm > max_norm`.
///
/// This is the same scaling `clip_by_global_norm` performs, but takes the
/// pre-clip global norm as a parameter instead of recomputing it internally.
/// `GradientClipper::step` already computes `original_norm` once per call
/// (needed for `ClipStats` and the EMA update regardless of mode), so its
/// `GlobalNorm`/`Adaptive` arms use this instead of the public
/// `clip_by_global_norm` to avoid a second full O(n) pass over the gradient
/// set just to recompute the identical norm.
///
/// No validation: callers are expected to have already checked
/// `max_norm > 0` and non-empty `gradients` (as `GradientClipper::step` does
/// via `ClipMode::validate` at construction time and its own empty check).
#[inline]
fn scale_to_norm(gradients: &mut [Vec<f32>], known_norm: f32, max_norm: f32) {
    if known_norm > max_norm && known_norm > 0.0 {
        let scale = max_norm / known_norm;
        for group in gradients.iter_mut() {
            for v in group.iter_mut() {
                *v *= scale;
            }
        }
    }
}

/// Clamp every gradient element to `[-max_val, max_val]`.
///
/// Returns the number of elements that were clamped (changed value).
///
/// # Errors
/// - [`ClipError::EmptyGradients`] if `gradients` is empty.
/// - [`ClipError::InvalidThreshold`] if `max_val <= 0`.
pub fn clip_by_value(gradients: &mut [Vec<f32>], max_val: f32) -> Result<usize, ClipError> {
    if max_val <= 0.0 {
        return Err(ClipError::InvalidThreshold(format!(
            "max_val must be > 0, got {max_val}"
        )));
    }
    if gradients.is_empty() {
        return Err(ClipError::EmptyGradients);
    }

    let mut num_clipped = 0usize;
    let neg_max = -max_val;
    for group in gradients.iter_mut() {
        for v in group.iter_mut() {
            let orig = *v;
            *v = v.clamp(neg_max, max_val);
            if *v != orig {
                num_clipped += 1;
            }
        }
    }
    Ok(num_clipped)
}

// ─────────────────────────────────────────────────────────────────────────────
// ClipStats
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics produced by one call to [`GradientClipper::step`].
#[derive(Debug, Clone)]
pub struct ClipStats {
    /// Original global norm before clipping.
    pub original_norm: f32,
    /// Global norm after clipping (≤ original_norm).
    pub clipped_norm: f32,
    /// Whether clipping was applied (norm exceeded the threshold).
    pub was_clipped: bool,
    /// Number of elements clipped (non-zero only for `ValueClip` mode).
    pub num_elements_clipped: usize,
    /// Current EMA norm after this step.
    pub ema_norm: f32,
}

impl ClipStats {
    /// Ratio `clipped_norm / original_norm`, floored to avoid division by near-zero.
    ///
    /// Returns `1.0` when the original norm is effectively zero.
    pub fn clip_ratio(&self) -> f32 {
        self.clipped_norm / self.original_norm.max(1e-8)
    }

    /// Human-readable summary: `"norm: 1.23 → 1.00 (clipped: true)"`.
    pub fn format(&self) -> String {
        format!(
            "norm: {:.2} → {:.2} (clipped: {})",
            self.original_norm, self.clipped_norm, self.was_clipped
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GradientClipper
// ─────────────────────────────────────────────────────────────────────────────

/// Default EMA factor used for modes other than `Adaptive`.
const DEFAULT_EMA_FACTOR: f32 = 0.9;

/// Default number of recent global norms [`GradientClipper`] retains.
///
/// The history is a *sliding window*, not a full log: a training run is
/// hundreds of thousands of steps long, so an unbounded `Vec` would grow for
/// the entire run while only its tail is ever read (see
/// [`GradientClipper::recent_history`]).
pub const DEFAULT_NORM_HISTORY_CAPACITY: usize = 1024;

/// Stateful gradient clipper that tracks history and supports adaptive clipping.
#[derive(Debug)]
pub struct GradientClipper {
    /// The clipping strategy.
    pub mode: ClipMode,
    /// Exponential moving average of the global gradient norm.
    ema_norm: f32,
    /// Sliding window of the most recently observed global norms (one entry
    /// per `step` call), bounded by `history_capacity`.
    norm_history: VecDeque<f32>,
    /// Maximum number of entries retained in `norm_history`.
    history_capacity: usize,
    /// Number of times clipping was actually applied (norm exceeded threshold).
    pub clip_count: usize,
    /// Total number of `step` calls.
    pub step_count: usize,
}

impl GradientClipper {
    /// Create a new clipper for the given mode, retaining
    /// [`DEFAULT_NORM_HISTORY_CAPACITY`] recent norms.
    ///
    /// Returns `Err` if the mode parameters are invalid.
    pub fn new(mode: ClipMode) -> Result<Self, ClipError> {
        Self::with_history_capacity(mode, DEFAULT_NORM_HISTORY_CAPACITY)
    }

    /// Create a new clipper that retains at most `history_capacity` recent
    /// global norms.
    ///
    /// The window is bounded so a long training run cannot grow the history
    /// without limit; once it is full, pushing a new norm evicts the oldest.
    ///
    /// # Errors
    /// - `Err` if the mode parameters are invalid.
    /// - [`ClipError::InvalidThreshold`] if `history_capacity` is zero.
    pub fn with_history_capacity(
        mode: ClipMode,
        history_capacity: usize,
    ) -> Result<Self, ClipError> {
        mode.validate()?;
        if history_capacity == 0 {
            return Err(ClipError::InvalidThreshold(
                "history_capacity must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            mode,
            ema_norm: 0.0,
            norm_history: VecDeque::with_capacity(history_capacity),
            history_capacity,
            clip_count: 0,
            step_count: 0,
        })
    }

    /// Apply clipping to `gradients` according to `self.mode`.
    ///
    /// Updates internal EMA, history, and counters. Returns [`ClipStats`].
    pub fn step(&mut self, gradients: &mut [Vec<f32>]) -> Result<ClipStats, ClipError> {
        if gradients.is_empty() {
            return Err(ClipError::EmptyGradients);
        }

        let original_norm = global_gradient_norm(gradients);

        // Initialise EMA to the first observed norm to avoid a cold-start cliff.
        if self.step_count == 0 {
            self.ema_norm = original_norm;
        }

        let (was_clipped, num_elements_clipped, clipped_norm) = match self.mode {
            ClipMode::GlobalNorm { max_norm } => {
                // Use the already-computed `original_norm` directly instead
                // of calling the public `clip_by_global_norm` (which would
                // recompute the identical norm internally before deciding
                // whether to scale).
                scale_to_norm(gradients, original_norm, max_norm);
                let clipped = original_norm > max_norm;
                // Exact by construction: clipping scales the norm down to
                // exactly `max_norm` when exceeded, otherwise it is
                // unchanged -- no need to re-scan the (possibly huge)
                // gradient set to measure it again.
                let clipped_norm = original_norm.min(max_norm);
                (clipped, 0usize, clipped_norm)
            }
            ClipMode::PerGroupNorm { max_norm } => {
                let norms = clip_by_per_group_norm(gradients, max_norm)?;
                let clipped = norms.iter().any(|&n| n > max_norm);
                // Exact by construction: each group's norm is scaled down to
                // exactly `max_norm` when exceeded; the global norm is the
                // L2 combination of the (now-capped) per-group norms, which
                // `clip_by_per_group_norm` already computed and returned --
                // no need for another full pass over every element.
                let clipped_norm = norms
                    .iter()
                    .map(|&n| {
                        let capped = n.min(max_norm);
                        capped * capped
                    })
                    .sum::<f32>()
                    .sqrt();
                (clipped, 0usize, clipped_norm)
            }
            ClipMode::ValueClip { max_val } => {
                let n = clip_by_value(gradients, max_val)?;
                let clipped = n > 0;
                // Per-element clamping has no closed-form norm update, so
                // this is the one mode that must re-measure directly.
                let clipped_norm = global_gradient_norm(gradients);
                (clipped, n, clipped_norm)
            }
            ClipMode::Adaptive {
                ema_factor,
                clip_factor,
            } => {
                // Update EMA with this step's norm (before clipping).
                self.ema_norm = ema_factor * self.ema_norm + (1.0 - ema_factor) * original_norm;
                let threshold = (self.ema_norm * clip_factor).max(1e-6);
                // Same reasoning as `GlobalNorm` above: `original_norm` was
                // already measured before this threshold was derived from
                // it, so reuse it instead of letting `clip_by_global_norm`
                // recompute the same pre-clip norm a second time.
                scale_to_norm(gradients, original_norm, threshold);
                let clipped = original_norm > threshold;
                let clipped_norm = original_norm.min(threshold);
                (clipped, 0usize, clipped_norm)
            }
        };

        // For non-Adaptive modes, update EMA with default factor.
        if !matches!(self.mode, ClipMode::Adaptive { .. }) {
            self.ema_norm =
                DEFAULT_EMA_FACTOR * self.ema_norm + (1.0 - DEFAULT_EMA_FACTOR) * original_norm;
        }

        if self.norm_history.len() == self.history_capacity {
            self.norm_history.pop_front();
        }
        self.norm_history.push_back(original_norm);
        self.step_count += 1;
        if was_clipped {
            self.clip_count += 1;
        }

        Ok(ClipStats {
            original_norm,
            clipped_norm,
            was_clipped,
            num_elements_clipped,
            ema_norm: self.ema_norm,
        })
    }

    /// Current EMA norm (updated after each `step`).
    pub fn ema_norm(&self) -> f32 {
        self.ema_norm
    }

    /// Fraction of steps where clipping was applied: `clip_count / step_count`.
    ///
    /// Returns `0.0` before any steps have been taken.
    pub fn clip_fraction(&self) -> f32 {
        if self.step_count == 0 {
            return 0.0;
        }
        self.clip_count as f32 / self.step_count as f32
    }

    /// The last `n` entries of the retained norm window, oldest first.
    ///
    /// If `n` is at least the retained length, the whole window is returned.
    /// Note the window itself is bounded by
    /// [`history_capacity`](Self::history_capacity), so this can never return
    /// more than that many entries however large `n` is, and entries older
    /// than the window have already been evicted.
    pub fn recent_history(&self, n: usize) -> Vec<f32> {
        let len = self.norm_history.len();
        let skip = len.saturating_sub(n);
        self.norm_history.iter().skip(skip).copied().collect()
    }

    /// Maximum number of recent norms retained by this clipper.
    pub fn history_capacity(&self) -> usize {
        self.history_capacity
    }

    /// Number of norms currently retained (at most
    /// [`history_capacity`](Self::history_capacity)).
    pub fn history_len(&self) -> usize {
        self.norm_history.len()
    }

    /// Reset all counters and EMA, but keep the mode and history capacity.
    pub fn reset_stats(&mut self) {
        self.ema_norm = 0.0;
        self.norm_history.clear();
        self.clip_count = 0;
        self.step_count = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient health diagnostics
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics describing the health of a set of gradient tensors.
#[derive(Debug, Clone)]
pub struct GradientHealth {
    /// Global L2 norm across all gradient tensors.
    pub global_norm: f32,
    /// Largest absolute value found across all elements.
    pub max_abs_value: f32,
    /// Smallest non-zero absolute value, or `None` if all elements are zero.
    pub min_abs_nonzero: Option<f32>,
    /// Whether any `NaN` value is present.
    pub has_nan: bool,
    /// Whether any `Inf` (positive or negative) value is present.
    pub has_inf: bool,
    /// Count of exactly-zero elements.
    pub num_zero: usize,
    /// Total element count across all tensors.
    pub num_total: usize,
    /// Fraction of zero elements: `num_zero / num_total`.
    pub sparsity: f32,
}

/// Analyse the health of a set of gradient tensors.
///
/// Iterates over every element exactly once and fills in [`GradientHealth`].
pub fn check_gradient_health(gradients: &[Vec<f32>]) -> GradientHealth {
    let mut sum_sq = 0.0_f32;
    let mut max_abs = 0.0_f32;
    let mut min_abs_nonzero: Option<f32> = None;
    let mut has_nan = false;
    let mut has_inf = false;
    let mut num_zero = 0usize;
    let mut num_total = 0usize;

    for group in gradients.iter() {
        for &v in group.iter() {
            num_total += 1;
            if v.is_nan() {
                has_nan = true;
                // Skip further arithmetic for NaN elements.
                continue;
            }
            if v.is_infinite() {
                has_inf = true;
                // Inf elements skew the norm; skip from sum_sq but track max.
                let abs = v.abs();
                if abs > max_abs {
                    max_abs = abs;
                }
                continue;
            }
            let abs = v.abs();
            sum_sq += v * v;
            if abs > max_abs {
                max_abs = abs;
            }
            if abs == 0.0 {
                num_zero += 1;
            } else {
                min_abs_nonzero = Some(match min_abs_nonzero {
                    None => abs,
                    Some(prev) => prev.min(abs),
                });
            }
        }
    }

    let global_norm = sum_sq.sqrt();
    let sparsity = if num_total == 0 {
        0.0
    } else {
        num_zero as f32 / num_total as f32
    };

    GradientHealth {
        global_norm,
        max_abs_value: max_abs,
        min_abs_nonzero,
        has_nan,
        has_inf,
        num_zero,
        num_total,
        sparsity,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdaptiveNormHistory
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks gradient norm statistics over a sliding window.
///
/// Keeps only the most recent `window_size` observations and provides
/// descriptive statistics (mean, standard deviation, percentiles) that
/// can inform adaptive clipping thresholds.
#[derive(Debug)]
pub struct AdaptiveNormHistory {
    /// Maximum number of norm values retained.
    pub window_size: usize,
    history: VecDeque<f32>,
}

impl AdaptiveNormHistory {
    /// Create a new history with the given `window_size`.
    ///
    /// # Errors
    /// Returns [`ClipError::InvalidThreshold`] if `window_size` is zero.
    pub fn new(window_size: usize) -> Result<Self, ClipError> {
        if window_size == 0 {
            return Err(ClipError::InvalidThreshold(
                "window_size must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            window_size,
            history: VecDeque::with_capacity(window_size),
        })
    }

    /// Add a new norm observation, evicting the oldest if the window is full.
    pub fn push(&mut self, norm: f32) {
        if self.history.len() == self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(norm);
    }

    /// Arithmetic mean of the window. Returns `0.0` if the window is empty.
    pub fn mean(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().sum();
        sum / self.history.len() as f32
    }

    /// Population standard deviation of the window. Returns `0.0` if empty.
    pub fn std(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let m = self.mean();
        let variance = self.history.iter().map(|&v| (v - m) * (v - m)).sum::<f32>()
            / self.history.len() as f32;
        variance.sqrt()
    }

    /// The value at quantile `p ∈ [0, 1]` using nearest-rank interpolation.
    ///
    /// Specifically returns `sorted[floor(p * (n - 1))]`.
    /// Returns `0.0` if the window is empty.
    pub fn percentile(&self, p: f32) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f32> = self.history.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let idx = ((p * (n - 1) as f32).floor() as usize).min(n - 1);
        // Safety: idx is clamped to [0, n-1] and sorted.len() == n >= 1.
        sorted[idx]
    }

    /// Current number of entries in the window.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Returns `true` if no entries have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Adaptive clip threshold: `mean + k * std`.
    ///
    /// A `k` of `3.0` implements the 3-sigma rule, accepting ~99.7 % of values.
    pub fn adaptive_threshold(&self, k: f32) -> f32 {
        self.mean() + k * self.std()
    }

    /// Returns `true` if `norm` is an outlier by the `k`-sigma criterion.
    ///
    /// Specifically, returns `norm > mean + k * std`.
    pub fn is_outlier(&self, norm: f32, k: f32) -> bool {
        norm > self.adaptive_threshold(k)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. l2_norm: [3, 4] → 5.0 ─────────────────────────────────────────────
    #[test]
    fn test_l2_norm_basic() {
        let norm = l2_norm(&[3.0, 4.0]);
        assert!((norm - 5.0).abs() < 1e-5, "expected 5.0, got {norm}");
    }

    // ── 2. l2_norm: empty → 0.0 ──────────────────────────────────────────────
    #[test]
    fn test_l2_norm_empty() {
        assert_eq!(l2_norm(&[]), 0.0);
    }

    // ── 3. global_gradient_norm: two tensors, correct combined norm ───────────
    #[test]
    fn test_global_gradient_norm_two_tensors() {
        // ||[3,4]||² + ||[0,5]||² = 25 + 25 = 50; norm = sqrt(50)
        let grads = vec![vec![3.0_f32, 4.0], vec![0.0, 5.0]];
        let norm = global_gradient_norm(&grads);
        let expected = 50.0_f32.sqrt();
        assert!(
            (norm - expected).abs() < 1e-5,
            "expected {expected}, got {norm}"
        );
    }

    // ── 4. clip_by_global_norm: norm below threshold → no change ─────────────
    #[test]
    fn test_clip_global_norm_below_threshold() {
        let mut grads = vec![vec![3.0_f32, 4.0]]; // norm = 5
        let original = clip_by_global_norm(&mut grads, 10.0).unwrap();
        assert!((original - 5.0).abs() < 1e-5);
        // Values unchanged
        assert!((grads[0][0] - 3.0).abs() < 1e-5);
        assert!((grads[0][1] - 4.0).abs() < 1e-5);
    }

    // ── 5. clip_by_global_norm: norm above threshold → scales down ────────────
    #[test]
    fn test_clip_global_norm_above_threshold() {
        let mut grads = vec![vec![3.0_f32, 4.0]]; // norm = 5
        let original = clip_by_global_norm(&mut grads, 1.0).unwrap();
        assert!((original - 5.0).abs() < 1e-5, "original norm should be 5");
        let new_norm = global_gradient_norm(&grads);
        assert!(
            new_norm <= 1.0 + 1e-5,
            "clipped norm {new_norm} should be ≤ 1.0"
        );
    }

    // ── 6. clip_by_global_norm: empty gradients → EmptyGradients Err ─────────
    #[test]
    fn test_clip_global_norm_empty() {
        let mut grads: Vec<Vec<f32>> = vec![];
        let err = clip_by_global_norm(&mut grads, 1.0).unwrap_err();
        assert!(matches!(err, ClipError::EmptyGradients));
    }

    // ── 7. clip_by_global_norm: max_norm=0 → Err ─────────────────────────────
    #[test]
    fn test_clip_global_norm_zero_threshold() {
        let mut grads = vec![vec![1.0_f32, 2.0]];
        let err = clip_by_global_norm(&mut grads, 0.0).unwrap_err();
        assert!(matches!(err, ClipError::InvalidThreshold(_)));
    }

    // ── 8. clip_by_per_group_norm: each group clipped independently ───────────
    #[test]
    fn test_clip_per_group_norm_clipped() {
        // Both groups have norm = 5, clipped to 2
        let mut grads = vec![vec![3.0_f32, 4.0], vec![3.0, 4.0]];
        let norms = clip_by_per_group_norm(&mut grads, 2.0).unwrap();
        assert_eq!(norms.len(), 2);
        for &n in &norms {
            assert!((n - 5.0).abs() < 1e-5);
        }
        for group in &grads {
            let new_n = l2_norm(group);
            assert!(new_n <= 2.0 + 1e-5, "group norm {new_n} > 2.0");
        }
    }

    // ── 9. clip_by_per_group_norm: group below threshold → unchanged ──────────
    #[test]
    fn test_clip_per_group_norm_unchanged() {
        let mut grads = vec![vec![0.5_f32, 0.5]]; // norm ≈ 0.707
        let norms = clip_by_per_group_norm(&mut grads, 10.0).unwrap();
        // Original values preserved
        assert!((grads[0][0] - 0.5).abs() < 1e-6);
        assert!((grads[0][1] - 0.5).abs() < 1e-6);
        assert!(norms[0] < 1.0);
    }

    // ── 10. clip_by_value: values outside range clamped ──────────────────────
    #[test]
    fn test_clip_by_value_clamps() {
        // [-5, 0, 5, 3, -3] with max_val=2:
        //   -5 → -2 (clipped), 0 → 0, 5 → 2 (clipped), 3 → 2 (clipped), -3 → -2 (clipped)
        // 4 elements clipped.
        let mut grads = vec![vec![-5.0_f32, 0.0, 5.0, 3.0, -3.0]];
        let n = clip_by_value(&mut grads, 2.0).unwrap();
        assert_eq!(n, 4, "expected 4 elements clipped, got {n}");
        assert!((grads[0][0] - (-2.0)).abs() < 1e-6);
        assert!((grads[0][2] - 2.0).abs() < 1e-6);
        assert!((grads[0][3] - 2.0).abs() < 1e-6);
        assert!((grads[0][4] - (-2.0)).abs() < 1e-6);
    }

    // ── 11. clip_by_value: values inside range → 0 clipped ───────────────────
    #[test]
    fn test_clip_by_value_no_clip() {
        let mut grads = vec![vec![0.5_f32, -0.5, 0.9, -0.9]];
        let n = clip_by_value(&mut grads, 1.0).unwrap();
        assert_eq!(n, 0);
    }

    // ── 12. GradientClipper::new: GlobalNorm valid → Ok ──────────────────────
    #[test]
    fn test_gradient_clipper_new_valid() {
        let clipper = GradientClipper::new(ClipMode::GlobalNorm { max_norm: 1.0 });
        assert!(clipper.is_ok());
    }

    // ── 13. GradientClipper::new: invalid max_norm → Err ─────────────────────
    #[test]
    fn test_gradient_clipper_new_invalid() {
        let err = GradientClipper::new(ClipMode::GlobalNorm { max_norm: -1.0 }).unwrap_err();
        assert!(matches!(err, ClipError::InvalidThreshold(_)));
    }

    // ── 14. GradientClipper::step: GlobalNorm clips correctly ─────────────────
    #[test]
    fn test_gradient_clipper_step_clips() {
        let mut clipper = GradientClipper::new(ClipMode::GlobalNorm { max_norm: 1.0 }).unwrap();
        let mut grads = vec![vec![3.0_f32, 4.0]]; // norm = 5 → clipped to 1
        let stats = clipper.step(&mut grads).unwrap();
        assert!(stats.was_clipped);
        assert!((stats.original_norm - 5.0).abs() < 1e-5);
        assert!(stats.clipped_norm <= 1.0 + 1e-5);
    }

    // ── 15. GradientClipper::step: increments step_count ─────────────────────
    #[test]
    fn test_gradient_clipper_step_count() {
        let mut clipper = GradientClipper::new(ClipMode::GlobalNorm { max_norm: 10.0 }).unwrap();
        let mut grads = vec![vec![1.0_f32]];
        clipper.step(&mut grads).unwrap();
        clipper.step(&mut grads).unwrap();
        assert_eq!(clipper.step_count, 2);
    }

    // ── 16. GradientClipper::step: increments clip_count only when clipped ────
    #[test]
    fn test_gradient_clipper_clip_count() {
        let mut clipper = GradientClipper::new(ClipMode::GlobalNorm { max_norm: 1.0 }).unwrap();
        // norm = 5 → clipped
        let mut grads_big = vec![vec![3.0_f32, 4.0]];
        clipper.step(&mut grads_big).unwrap();
        // norm = 0.1 → not clipped
        let mut grads_small = vec![vec![0.1_f32]];
        clipper.step(&mut grads_small).unwrap();
        assert_eq!(clipper.clip_count, 1);
        assert_eq!(clipper.step_count, 2);
    }

    // ── 17. GradientClipper::clip_fraction: correct ratio ────────────────────
    #[test]
    fn test_gradient_clipper_clip_fraction() {
        let mut clipper = GradientClipper::new(ClipMode::GlobalNorm { max_norm: 1.0 }).unwrap();
        let mut big = vec![vec![3.0_f32, 4.0]]; // clipped
        let mut small = vec![vec![0.1_f32]]; // not clipped
        clipper.step(&mut big).unwrap();
        clipper.step(&mut small).unwrap();
        clipper.step(&mut small).unwrap();
        // 1 clipped out of 3 = 1/3
        let frac = clipper.clip_fraction();
        assert!((frac - 1.0 / 3.0).abs() < 1e-5, "fraction was {frac}");
    }

    // ── 18. GradientClipper::ema_norm: updates after steps ───────────────────
    #[test]
    fn test_gradient_clipper_ema_norm_updates() {
        let mut clipper = GradientClipper::new(ClipMode::GlobalNorm { max_norm: 100.0 }).unwrap();
        assert_eq!(clipper.ema_norm(), 0.0);
        // First step initialises EMA to current norm (5.0)
        let mut grads = vec![vec![3.0_f32, 4.0]]; // norm = 5
        clipper.step(&mut grads).unwrap();
        let ema_after_first = clipper.ema_norm();
        // After first step: initialised to 5.0 then updated: 0.9*5 + 0.1*5 = 5
        assert!(
            ema_after_first > 0.0,
            "EMA should be non-zero after first step"
        );
        // Second step with different magnitude
        let mut grads2 = vec![vec![0.0_f32]];
        clipper.step(&mut grads2).unwrap();
        let ema_after_second = clipper.ema_norm();
        // 0.9 * 5 + 0.1 * 0 = 4.5
        assert!(ema_after_second < ema_after_first, "EMA should decrease");
    }

    // ── 19. ClipStats::clip_ratio: 1.0 when not clipped, < 1.0 when clipped ──
    #[test]
    fn test_clip_stats_ratio() {
        let not_clipped = ClipStats {
            original_norm: 1.0,
            clipped_norm: 1.0,
            was_clipped: false,
            num_elements_clipped: 0,
            ema_norm: 1.0,
        };
        assert!((not_clipped.clip_ratio() - 1.0).abs() < 1e-6);

        let clipped = ClipStats {
            original_norm: 5.0,
            clipped_norm: 1.0,
            was_clipped: true,
            num_elements_clipped: 0,
            ema_norm: 1.0,
        };
        assert!(clipped.clip_ratio() < 1.0);
        assert!((clipped.clip_ratio() - 0.2).abs() < 1e-5);
    }

    // ── 20. check_gradient_health: detects NaN ────────────────────────────────
    #[test]
    fn test_check_gradient_health_nan() {
        let grads = vec![vec![1.0_f32, f32::NAN, 2.0]];
        let health = check_gradient_health(&grads);
        assert!(health.has_nan);
    }

    // ── 21. check_gradient_health: correct sparsity ───────────────────────────
    #[test]
    fn test_check_gradient_health_sparsity() {
        // 2 zeros out of 4 elements → sparsity = 0.5
        let grads = vec![vec![0.0_f32, 1.0], vec![0.0, 2.0]];
        let health = check_gradient_health(&grads);
        assert_eq!(health.num_zero, 2);
        assert_eq!(health.num_total, 4);
        assert!((health.sparsity - 0.5).abs() < 1e-6);
    }

    // ── 22. AdaptiveNormHistory::new: window_size=0 → Err ────────────────────
    #[test]
    fn test_adaptive_norm_history_zero_window() {
        let err = AdaptiveNormHistory::new(0).unwrap_err();
        assert!(matches!(err, ClipError::InvalidThreshold(_)));
    }

    // ── 23. AdaptiveNormHistory::push + percentile ────────────────────────────
    #[test]
    fn test_adaptive_norm_history_percentile() {
        let mut hist = AdaptiveNormHistory::new(10).unwrap();
        for &v in &[1.0_f32, 2.0, 3.0, 4.0, 5.0] {
            hist.push(v);
        }
        // p=0.0 → sorted[0] = 1.0
        assert!((hist.percentile(0.0) - 1.0).abs() < 1e-5);
        // p=1.0 → sorted[4] = 5.0
        assert!((hist.percentile(1.0) - 5.0).abs() < 1e-5);
        // p=0.5 → sorted[floor(0.5*4)] = sorted[2] = 3.0
        assert!((hist.percentile(0.5) - 3.0).abs() < 1e-5);
    }

    // ── 24. AdaptiveNormHistory::adaptive_threshold: mean + k*std ────────────
    #[test]
    fn test_adaptive_norm_history_threshold() {
        let mut hist = AdaptiveNormHistory::new(10).unwrap();
        for &v in &[1.0_f32, 2.0, 3.0] {
            hist.push(v);
        }
        let m = hist.mean();
        let s = hist.std();
        let k = 2.0_f32;
        let threshold = hist.adaptive_threshold(k);
        assert!((threshold - (m + k * s)).abs() < 1e-5);
    }

    // ── 25. Adaptive mode: clips at ema * clip_factor ─────────────────────────
    #[test]
    fn test_adaptive_mode_clips() {
        // ema_factor=0.9, clip_factor=0.5
        // First step: EMA initialised to norm, threshold = norm * 0.5 → always clips.
        let mut clipper = GradientClipper::new(ClipMode::Adaptive {
            ema_factor: 0.9,
            clip_factor: 0.5,
        })
        .unwrap();
        let mut grads = vec![vec![3.0_f32, 4.0]]; // norm = 5
                                                  // On first step: ema_norm is initialised to 5.0, then updated:
                                                  //   ema_norm = 0.9 * 5 + 0.1 * 5 = 5.0
                                                  //   threshold = 5 * 0.5 = 2.5
                                                  // norm (5.0) > threshold (2.5) → clipped
        let stats = clipper.step(&mut grads).unwrap();
        assert!(
            stats.was_clipped,
            "adaptive mode should clip when norm > ema*factor"
        );
        assert!(
            stats.clipped_norm <= 2.5 + 1e-4,
            "clipped norm {} should be ≤ 2.5",
            stats.clipped_norm
        );
    }

    // ── PerGroupNorm: analytical clipped_norm matches direct measurement ─────
    #[test]
    fn test_step_per_group_norm_clipped_norm_matches_direct_computation() {
        let mut clipper = GradientClipper::new(ClipMode::PerGroupNorm { max_norm: 1.0 }).unwrap();
        // Group 0: norm = 5 (3,4) -> clipped to 1.0
        // Group 1: norm = 0.5 (0.3,0.4) -> unclipped
        let mut grads = vec![vec![3.0_f32, 4.0], vec![0.3_f32, 0.4]];
        let stats = clipper.step(&mut grads).unwrap();
        assert!(stats.was_clipped);
        // Expected global norm after clipping: sqrt(1.0^2 + 0.5^2)
        let expected = (1.0f32 * 1.0 + 0.5 * 0.5).sqrt();
        assert!(
            (stats.clipped_norm - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            stats.clipped_norm
        );
        // Cross-check against directly re-measuring the (now-mutated)
        // gradients, to confirm the analytical shortcut is exact.
        let direct = global_gradient_norm(&grads);
        assert!(
            (stats.clipped_norm - direct).abs() < 1e-4,
            "analytical clipped_norm {} should match direct measurement {direct}",
            stats.clipped_norm
        );
    }

    // ── GlobalNorm/Adaptive: analytical clipped_norm matches direct measurement ──
    // Regression for the `scale_to_norm` refactor (replaces the internal
    // `clip_by_global_norm` call in `step`, which used to recompute the same
    // pre-clip norm `step` had already measured as `original_norm`): confirms
    // the now-single-pass norm computation still produces gradients whose
    // post-clip norm exactly matches the analytically-derived `clipped_norm`.
    #[test]
    fn test_step_global_norm_clipped_norm_matches_direct_computation() {
        let mut clipper = GradientClipper::new(ClipMode::GlobalNorm { max_norm: 1.0 }).unwrap();
        let mut grads = vec![vec![3.0_f32, 4.0], vec![0.3_f32, 0.4]]; // norm = sqrt(25+0.25)
        let stats = clipper.step(&mut grads).unwrap();
        assert!(stats.was_clipped);
        let direct = global_gradient_norm(&grads);
        assert!(
            (stats.clipped_norm - direct).abs() < 1e-4,
            "analytical clipped_norm {} should match direct measurement {direct}",
            stats.clipped_norm
        );
        assert!(
            (stats.clipped_norm - 1.0).abs() < 1e-4,
            "clipped_norm should be exactly max_norm=1.0, got {}",
            stats.clipped_norm
        );
    }

    #[test]
    fn test_step_adaptive_clipped_norm_matches_direct_computation() {
        let mut clipper = GradientClipper::new(ClipMode::Adaptive {
            ema_factor: 0.9,
            clip_factor: 0.5,
        })
        .unwrap();
        let mut grads = vec![vec![3.0_f32, 4.0]]; // norm = 5 -> threshold = 5*0.5 = 2.5
        let stats = clipper.step(&mut grads).unwrap();
        assert!(stats.was_clipped);
        let direct = global_gradient_norm(&grads);
        assert!(
            (stats.clipped_norm - direct).abs() < 1e-4,
            "analytical clipped_norm {} should match direct measurement {direct}",
            stats.clipped_norm
        );
    }

    // ── Bonus: clip_fraction is 0 before any steps ────────────────────────────
    #[test]
    fn test_clip_fraction_zero_steps() {
        let clipper = GradientClipper::new(ClipMode::GlobalNorm { max_norm: 1.0 }).unwrap();
        assert_eq!(clipper.clip_fraction(), 0.0);
    }

    // ── Bonus: recent_history returns last n entries ───────────────────────────
    #[test]
    fn test_recent_history() {
        let mut clipper = GradientClipper::new(ClipMode::GlobalNorm { max_norm: 100.0 }).unwrap();
        for i in 1..=5 {
            let mut g = vec![vec![i as f32]];
            clipper.step(&mut g).unwrap();
        }
        let last_3 = clipper.recent_history(3);
        assert_eq!(last_3.len(), 3);
        // Last 3 norms should be 3.0, 4.0, 5.0
        for (j, &v) in last_3.iter().enumerate() {
            assert!((v - (j + 3) as f32).abs() < 1e-5, "entry {j} was {v}");
        }
    }

    // ── Regression (F283): the norm history is a bounded sliding window ──────
    // It used to be an unbounded `Vec<f32>` pushed once per `step`, so a
    // multi-hundred-thousand-step training run grew it for the entire run
    // while only its tail was ever read.
    #[test]
    fn test_norm_history_is_bounded_and_evicts_oldest() {
        let mut clipper =
            GradientClipper::with_history_capacity(ClipMode::GlobalNorm { max_norm: 1000.0 }, 4)
                .expect("capacity 4 is valid");
        assert_eq!(clipper.history_capacity(), 4);

        for i in 1..=100 {
            let mut g = vec![vec![i as f32]];
            clipper.step(&mut g).expect("step must succeed");
        }

        // 100 steps recorded, but only the last 4 norms are retained.
        assert_eq!(clipper.step_count, 100);
        assert_eq!(clipper.history_len(), 4, "history must stay bounded");

        // Asking for more than the capacity still yields only the window.
        let all = clipper.recent_history(usize::MAX);
        assert_eq!(all.len(), 4);
        // Oldest-first ordering, and the *newest* norms survived.
        for (j, &v) in all.iter().enumerate() {
            let expected = (97 + j) as f32;
            assert!(
                (v - expected).abs() < 1e-5,
                "entry {j} was {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_with_history_capacity_zero_is_rejected() {
        let err = GradientClipper::with_history_capacity(ClipMode::GlobalNorm { max_norm: 1.0 }, 0)
            .expect_err("zero capacity must be rejected");
        assert!(matches!(err, ClipError::InvalidThreshold(_)));
    }

    #[test]
    fn test_default_history_capacity_is_bounded() {
        let clipper =
            GradientClipper::new(ClipMode::GlobalNorm { max_norm: 1.0 }).expect("valid mode");
        assert_eq!(clipper.history_capacity(), DEFAULT_NORM_HISTORY_CAPACITY);
        assert_eq!(clipper.history_len(), 0);
    }

    // ── Bonus: reset_stats clears state ───────────────────────────────────────
    #[test]
    fn test_reset_stats() {
        let mut clipper = GradientClipper::new(ClipMode::GlobalNorm { max_norm: 1.0 }).unwrap();
        let mut g = vec![vec![3.0_f32, 4.0]];
        clipper.step(&mut g).unwrap();
        assert!(clipper.step_count > 0);
        clipper.reset_stats();
        assert_eq!(clipper.step_count, 0);
        assert_eq!(clipper.clip_count, 0);
        assert_eq!(clipper.ema_norm(), 0.0);
    }

    // ── Bonus: ClipStats::format contains expected substrings ─────────────────
    #[test]
    fn test_clip_stats_format() {
        let stats = ClipStats {
            original_norm: 5.0,
            clipped_norm: 1.0,
            was_clipped: true,
            num_elements_clipped: 0,
            ema_norm: 1.0,
        };
        let s = stats.format();
        assert!(s.contains("5.00"), "format: {s}");
        assert!(s.contains("1.00"), "format: {s}");
        assert!(s.contains("true"), "format: {s}");
    }

    // ── Bonus: check_gradient_health detects Inf ─────────────────────────────
    #[test]
    fn test_check_gradient_health_inf() {
        let grads = vec![vec![1.0_f32, f32::INFINITY]];
        let health = check_gradient_health(&grads);
        assert!(health.has_inf);
    }

    // ── Bonus: AdaptiveNormHistory sliding window evicts old entries ──────────
    #[test]
    fn test_adaptive_norm_history_window_eviction() {
        let mut hist = AdaptiveNormHistory::new(3).unwrap();
        for &v in &[1.0_f32, 2.0, 3.0, 100.0] {
            hist.push(v);
        }
        // Window = [2.0, 3.0, 100.0]; mean should NOT be close to 1.5
        assert_eq!(hist.len(), 3);
        let m = hist.mean();
        assert!(m > 30.0, "mean {m} should reflect window without 1.0");
    }
}
