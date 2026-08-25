//! Dynamic per-sample loss reweighting for Gaussian avatar training.
//!
//! Implements multiple strategies for computing per-sample importance weights,
//! enabling the training loop to focus on hard examples (focal weighting,
//! hardness weighting) or to follow a curriculum (easy-first to hard).
//!
//! This module is **distinct** from `adaptive_loss` (which adjusts weights
//! *between* loss components like photometric/perceptual/structural) and from
//! `ohem` (which selects a hard-example *subset*). Here, every sample gets a
//! scalar weight; no sample is dropped.
//!
//! # Quick start
//! ```rust
//! use oxigaf_trainer::loss_reweighting::{
//!     SampleWeightingStrategy, HardnessConfig, compute_hardness_weights,
//!     build_sample_weights,
//! };
//!
//! let losses = vec![0.1_f32, 0.5, 0.9, 0.3];
//! let config = HardnessConfig::default();
//! let weights = compute_hardness_weights(&losses, &config).expect("non-empty losses");
//! let sw = build_sample_weights(weights, SampleWeightingStrategy::Hardness);
//! println!("{}", oxigaf_trainer::loss_reweighting::format_weight_summary(&sw));
//! ```

use std::collections::VecDeque;

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// ReweightingError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the per-sample reweighting subsystem.
#[derive(Debug, Error)]
pub enum ReweightingError {
    #[error("Empty losses: cannot compute weights for empty sample set")]
    EmptyLosses,

    #[error("Losses and predictions have different lengths: {losses_len} vs {pred_len}")]
    LengthMismatch { losses_len: usize, pred_len: usize },

    #[error("Invalid gamma {gamma}: must be >= 0")]
    InvalidGamma { gamma: f32 },

    #[error("Invalid temperature {temp}: must be > 0")]
    InvalidTemperature { temp: f32 },

    #[error("Invalid beta {beta}: must be in [0, 1]")]
    InvalidBeta { beta: f32 },
}

// ─────────────────────────────────────────────────────────────────────────────
// SampleWeightingStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for computing per-sample loss weights.
///
/// In Gaussian avatar training a "sample" can be a rendered pixel, a training
/// view, or a diffusion time step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleWeightingStrategy {
    /// All samples receive equal weight (no reweighting).
    Uniform,
    /// Hard samples receive higher weight following the focal loss formulation.
    FocalLoss,
    /// Higher loss → higher weight (proportional / softmax-based hardness).
    Hardness,
    /// Lower loss → higher weight (easy samples first; curriculum start phase).
    InverseHardness,
    /// Exponential scaling of loss values into weights.
    Exponential,
    /// Weight by rank: the sample with the highest loss gets the highest weight.
    Rank,
}

// ─────────────────────────────────────────────────────────────────────────────
// FocalLossConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for focal-loss per-sample weighting.
///
/// The focal-loss weight for sample *i* with predicted probability *p_i* is:
///
/// `w_i = alpha * (1 - p_i)^gamma`
///
/// After normalization the weights sum to `n_samples`.
#[derive(Debug, Clone)]
pub struct FocalLossConfig {
    /// Focusing parameter ≥ 0.  Larger values suppress well-classified samples
    /// more aggressively.  Default: 2.0.
    pub gamma: f32,
    /// Class-balance factor in \[0, 1\].  Default: 0.5.
    pub alpha: f32,
}

impl Default for FocalLossConfig {
    fn default() -> Self {
        Self {
            gamma: 2.0,
            alpha: 0.5,
        }
    }
}

impl FocalLossConfig {
    /// Returns an error if the configuration is invalid.
    pub fn validate(&self) -> Result<(), ReweightingError> {
        if self.gamma < 0.0 {
            return Err(ReweightingError::InvalidGamma { gamma: self.gamma });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HardnessConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for hardness-based per-sample weighting.
#[derive(Debug, Clone)]
pub struct HardnessConfig {
    /// Softmax temperature: lower → sharper weight distribution.  Default: 1.0.
    pub temperature: f32,
    /// Maximum allowed weight as a multiple of the mean weight.  Weights
    /// exceeding this ratio are clipped and the distribution is re-normalised.
    /// Default: 10.0.
    pub max_weight_ratio: f32,
}

impl Default for HardnessConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            max_weight_ratio: 10.0,
        }
    }
}

impl HardnessConfig {
    /// Returns an error if the configuration is invalid.
    pub fn validate(&self) -> Result<(), ReweightingError> {
        if self.temperature <= 0.0 {
            return Err(ReweightingError::InvalidTemperature {
                temp: self.temperature,
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CurriculumWeightConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for curriculum-aware per-sample weighting.
///
/// Interpolates linearly from `start_strategy` to `end_strategy` over
/// `warmup_steps` training steps.
#[derive(Debug, Clone)]
pub struct CurriculumWeightConfig {
    /// Strategy used at the beginning of training (step 0).  Default:
    /// [`SampleWeightingStrategy::InverseHardness`] — easy samples first.
    pub start_strategy: SampleWeightingStrategy,
    /// Strategy used at the end of the warmup.  Default:
    /// [`SampleWeightingStrategy::Hardness`] — hard samples prioritised.
    pub end_strategy: SampleWeightingStrategy,
    /// Number of steps over which to transition.  Default: 1000.
    pub warmup_steps: usize,
}

impl Default for CurriculumWeightConfig {
    fn default() -> Self {
        Self {
            start_strategy: SampleWeightingStrategy::InverseHardness,
            end_strategy: SampleWeightingStrategy::Hardness,
            warmup_steps: 1000,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SampleWeights
// ─────────────────────────────────────────────────────────────────────────────

/// A set of computed per-sample weights, together with summary statistics.
///
/// The weights are normalised so that `mean(weights) == 1.0`
/// (i.e. `sum(weights) == n_samples`).
#[derive(Debug, Clone)]
pub struct SampleWeights {
    /// Per-sample scalar weights (length = n_samples).
    pub weights: Vec<f32>,
    /// The strategy that produced these weights.
    pub strategy: SampleWeightingStrategy,
    /// Arithmetic mean of the weights (≈ 1.0 by construction).
    pub mean_weight: f32,
    /// Maximum weight in the distribution.
    pub max_weight: f32,
    /// Minimum weight in the distribution.
    pub min_weight: f32,
    /// Standard deviation of the weight distribution.
    pub weight_std: f32,
    /// Effective sample size: `n² / sum(w²)`.  Equals `n` for uniform weights;
    /// smaller values indicate high weight concentration.
    pub effective_sample_size: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// WeightTracker
// ─────────────────────────────────────────────────────────────────────────────

/// Rolling history of per-step weight statistics for post-hoc analysis.
#[derive(Debug, Clone)]
pub struct WeightTracker {
    /// Circular history: `(step, mean_weight, max_weight)`. Backed by a
    /// `VecDeque` so eviction at capacity is O(1) via `pop_front()` instead
    /// of the O(n) memmove that `Vec::remove(0)` would incur on every
    /// recorded step.
    pub history: VecDeque<(usize, f32, f32)>,
    /// Maximum number of entries retained.
    pub capacity: usize,
}

impl WeightTracker {
    /// Create a new tracker with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Record statistics for the current training step.
    ///
    /// If the history is at capacity, the oldest entry is evicted.
    pub fn record(&mut self, step: usize, mean_weight: f32, max_weight: f32) {
        if self.capacity == 0 {
            return;
        }
        if self.history.len() >= self.capacity {
            self.history.pop_front();
        }
        self.history.push_back((step, mean_weight, max_weight));
    }

    /// Estimate the linear trend of `max_weight` over the last `window` entries.
    ///
    /// Returns the slope (positive = weights growing, negative = shrinking).
    /// Returns `None` if there are fewer than 2 entries in the window.
    pub fn recent_max_weight_trend(&self, window: usize) -> Option<f32> {
        let n = self.history.len();
        if n < 2 || window < 2 {
            return None;
        }
        let start = n.saturating_sub(window);
        let count = n - start;
        if count < 2 {
            return None;
        }
        // Simple linear regression slope on (step, max_weight) pairs.
        // `VecDeque` has no `Index<Range<usize>>`, so use `.range()` instead
        // of slicing directly (as in `GradientFlowTracker::analyze_group`).
        let len = count as f32;
        let sum_x: f32 = self.history.range(start..).map(|&(s, _, _)| s as f32).sum();
        let sum_y: f32 = self.history.range(start..).map(|&(_, _, m)| m).sum();
        let mean_x = sum_x / len;
        let mean_y = sum_y / len;

        let ss_xx: f32 = self
            .history
            .range(start..)
            .map(|&(s, _, _)| {
                let dx = s as f32 - mean_x;
                dx * dx
            })
            .sum();
        let ss_xy: f32 = self
            .history
            .range(start..)
            .map(|&(s, _, m)| (s as f32 - mean_x) * (m - mean_y))
            .sum();

        if ss_xx.abs() < 1e-12 {
            return None;
        }
        Some(ss_xy / ss_xx)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core weighting functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute focal-loss per-sample weights.
///
/// `w_i = alpha_i * (1 - predictions[i])^gamma`, then normalised so
/// `mean(w) == 1.0`, where `alpha_i = alpha` when `predictions[i] >= 0.5`
/// (treated as the "positive"/in-class prediction) and `alpha_i = 1 -
/// alpha` otherwise — the standard Lin et al. 2017 class-balance split,
/// applied per-sample from the prediction itself since this function has
/// no separate ground-truth label input to split on.
///
/// A UNIFORM (non-per-class) `alpha` — the previous implementation — has
/// NO effect on the returned weights: scaling every raw weight by the
/// same constant before mean-normalisation cancels out exactly. Splitting
/// `alpha` by predicted class is what gives it an actual, testable
/// effect; `alpha = 0.5` puts BOTH branches at the same 0.5 multiplier,
/// so it remains the alpha-independent midpoint (matching the previous
/// behaviour at that one value).
///
/// Before normalisation: a perfect "positive" prediction (`p_i = 1.0`)
/// yields raw weight `alpha * 0 = 0`; a completely wrong "negative"
/// prediction (`p_i = 0.0`) yields raw weight `(1 - alpha) * 1`.
///
/// # Errors
/// - [`ReweightingError::EmptyLosses`] if `predictions` is empty.
/// - [`ReweightingError::InvalidGamma`] if `gamma < 0`.
pub fn compute_focal_weights(
    predictions: &[f32],
    gamma: f32,
    alpha: f32,
) -> Result<Vec<f32>, ReweightingError> {
    if predictions.is_empty() {
        return Err(ReweightingError::EmptyLosses);
    }
    if gamma < 0.0 {
        return Err(ReweightingError::InvalidGamma { gamma });
    }
    let alpha = alpha.clamp(0.0, 1.0);

    let raw: Vec<f32> = predictions
        .iter()
        .map(|&p| {
            let p_clamped = p.clamp(0.0, 1.0);
            let alpha_i = if p_clamped >= 0.5 { alpha } else { 1.0 - alpha };
            alpha_i * (1.0 - p_clamped).powf(gamma)
        })
        .collect();

    normalize_to_mean_one(raw)
}

/// Compute hardness-based per-sample weights.
///
/// Uses a numerically stable softmax with `temperature`:
/// `w_i ∝ exp((losses[i] - max_loss) / temperature)`.
///
/// After computing softmax probabilities the result is scaled so that
/// `mean(w) == 1.0` and any weight exceeding `config.max_weight_ratio *
/// mean_weight` is clipped; ONLY the un-clipped mass is then redistributed
/// among the un-clipped weights (never rescaling an already-capped weight
/// back above the cap), iterating a few passes since redistributing mass
/// can itself push a previously-safe weight over the cap. The result
/// always satisfies both `sum(weights) == n` (mean stays 1.0) and
/// `weights[i] <= config.max_weight_ratio` for every `i` — unless even the
/// full un-capped mass cannot be absorbed without exceeding the cap
/// itself, e.g. `max_weight_ratio < 1.0`, in which case the mean
/// invariant is preserved and the cap is honoured for whichever weights
/// were clipped, but the deficit cannot be fully redistributed onto
/// unclipped weights that are already at (or would need to exceed) the
/// same cap.
///
/// Higher loss → higher weight.
///
/// # Errors
/// - [`ReweightingError::EmptyLosses`] if `losses` is empty.
/// - [`ReweightingError::InvalidTemperature`] if `config.temperature <= 0`.
pub fn compute_hardness_weights(
    losses: &[f32],
    config: &HardnessConfig,
) -> Result<Vec<f32>, ReweightingError> {
    if losses.is_empty() {
        return Err(ReweightingError::EmptyLosses);
    }
    config.validate()?;

    let max_loss = losses.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Numerically stable softmax.
    let exps: Vec<f32> = losses
        .iter()
        .map(|&l| ((l - max_loss) / config.temperature).exp())
        .collect();

    let sum_exp: f32 = exps.iter().sum();
    let n = losses.len() as f32;

    // Scale from probabilities (sum = 1) to weights (mean = 1).
    let mut weights: Vec<f32> = exps.iter().map(|&e| e / sum_exp * n).collect();

    let cap = config.max_weight_ratio;

    // Clip weights above the cap, then redistribute ONLY the un-clipped
    // mass among the un-clipped weights — never touching an
    // already-clipped weight again. The previous implementation rescaled
    // ALL weights (including the just-clipped ones) by `n / clip_sum`;
    // since clipping only ever reduces the sum, that scale factor was
    // always > 1, which pushed the clipped weights right back above the
    // cap it had just enforced. Iterate a few passes since redistributing
    // the deficit can itself push a previously-uncapped weight over the
    // cap.
    let mut clipped = vec![false; weights.len()];
    for _pass in 0..8 {
        let mut any_new_clip = false;
        for (w, c) in weights.iter_mut().zip(clipped.iter_mut()) {
            if !*c && *w > cap {
                *w = cap;
                *c = true;
                any_new_clip = true;
            }
        }

        let clipped_total: f32 = weights
            .iter()
            .zip(clipped.iter())
            .filter(|&(_, &c)| c)
            .map(|(&w, _)| w)
            .sum();
        let uncapped_sum: f32 = weights
            .iter()
            .zip(clipped.iter())
            .filter(|&(_, &c)| !c)
            .map(|(&w, _)| w)
            .sum();
        let deficit = n - clipped_total;

        if uncapped_sum > 1e-12 && deficit > 0.0 {
            let scale = deficit / uncapped_sum;
            for (w, &c) in weights.iter_mut().zip(clipped.iter()) {
                if !c {
                    *w *= scale;
                }
            }
        }

        if !any_new_clip {
            break;
        }
    }

    Ok(weights)
}

/// Compute inverse-hardness per-sample weights.
///
/// Low loss → high weight (easy samples get more attention — curriculum start).
/// `w_i ∝ exp(-losses[i] / temperature)`, normalised to `mean == 1.0`.
///
/// # Errors
/// - [`ReweightingError::EmptyLosses`] if `losses` is empty.
/// - [`ReweightingError::InvalidTemperature`] if `temperature <= 0`.
pub fn compute_inverse_hardness_weights(
    losses: &[f32],
    temperature: f32,
) -> Result<Vec<f32>, ReweightingError> {
    if losses.is_empty() {
        return Err(ReweightingError::EmptyLosses);
    }
    if temperature <= 0.0 {
        return Err(ReweightingError::InvalidTemperature { temp: temperature });
    }

    // Numerically stable: subtract the minimum divided by temperature,
    // which corresponds to adding the maximum of (-l/T).
    let min_loss = losses.iter().copied().fold(f32::INFINITY, f32::min);

    let exps: Vec<f32> = losses
        .iter()
        .map(|&l| (-(l - min_loss) / temperature).exp())
        .collect();

    normalize_to_mean_one(exps)
}

/// Compute exponential per-sample weights.
///
/// `w_i = exp(beta * losses[i]) / mean(exp(beta * losses))`, scaled to
/// `mean == 1.0`.
///
/// `beta == 0` produces uniform weights.  Positive beta causes hard samples
/// to receive higher weight.
///
/// # Errors
/// - [`ReweightingError::EmptyLosses`] if `losses` is empty.
/// - [`ReweightingError::InvalidBeta`] if `beta` is not in \[0, 1\].
pub fn compute_exponential_weights(
    losses: &[f32],
    beta: f32,
) -> Result<Vec<f32>, ReweightingError> {
    if losses.is_empty() {
        return Err(ReweightingError::EmptyLosses);
    }
    if !(0.0..=1.0).contains(&beta) {
        return Err(ReweightingError::InvalidBeta { beta });
    }

    // Numerically stable: subtract max before exp when beta > 0.
    let max_loss = losses.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let exps: Vec<f32> = losses
        .iter()
        .map(|&l| (beta * (l - max_loss)).exp())
        .collect();

    normalize_to_mean_one(exps)
}

/// Compute rank-based per-sample weights.
///
/// Samples are ranked by ascending loss value: the sample with the lowest
/// loss receives rank 1, and the sample with the highest loss receives rank
/// `n`.  Ties are broken by index (earlier index → lower rank).
///
/// The weight is `rank / mean(ranks)`, normalised to `mean == 1.0`.
///
/// # Errors
/// - [`ReweightingError::EmptyLosses`] if `losses` is empty.
pub fn compute_rank_weights(losses: &[f32]) -> Result<Vec<f32>, ReweightingError> {
    if losses.is_empty() {
        return Err(ReweightingError::EmptyLosses);
    }

    let n = losses.len();

    // Build sorted order of indices: lowest loss first.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        losses[a]
            .partial_cmp(&losses[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b)) // tie-break by index
    });

    // Assign rank 1..=n.
    let mut ranks = vec![0usize; n];
    for (rank_minus_one, &original_idx) in order.iter().enumerate() {
        ranks[original_idx] = rank_minus_one + 1;
    }

    // mean rank = (1 + n) / 2.
    let mean_rank = (n + 1) as f32 / 2.0;
    let weights: Vec<f32> = ranks.iter().map(|&r| r as f32 / mean_rank).collect();

    Ok(weights)
}

/// Compute uniform weights (all equal to 1.0).
///
/// The identity weighting — equivalent to standard unweighted training.
pub fn compute_uniform_weights(n: usize) -> Vec<f32> {
    vec![1.0f32; n]
}

// ─────────────────────────────────────────────────────────────────────────────
// Build SampleWeights
// ─────────────────────────────────────────────────────────────────────────────

/// Compute summary statistics and package weights into a [`SampleWeights`].
pub fn build_sample_weights(weights: Vec<f32>, strategy: SampleWeightingStrategy) -> SampleWeights {
    let n = weights.len();
    if n == 0 {
        return SampleWeights {
            weights,
            strategy,
            mean_weight: 0.0,
            max_weight: 0.0,
            min_weight: 0.0,
            weight_std: 0.0,
            effective_sample_size: 0.0,
        };
    }

    let mean_weight = weights.iter().sum::<f32>() / n as f32;
    let max_weight = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_weight = weights.iter().copied().fold(f32::INFINITY, f32::min);

    let variance = weights
        .iter()
        .map(|&w| {
            let d = w - mean_weight;
            d * d
        })
        .sum::<f32>()
        / n as f32;
    let weight_std = variance.sqrt();

    // ESS = n² / sum(w²).
    let sum_w_sq: f32 = weights.iter().map(|&w| w * w).sum();
    let effective_sample_size = if sum_w_sq > 0.0 {
        (n as f32 * n as f32) / sum_w_sq
    } else {
        0.0
    };

    SampleWeights {
        weights,
        strategy,
        mean_weight,
        max_weight,
        min_weight,
        weight_std,
        effective_sample_size,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_sample_reweights — primary dispatch function
// ─────────────────────────────────────────────────────────────────────────────

/// Compute per-sample weights for an arbitrary strategy.
///
/// # Arguments
/// - `losses` — per-sample loss values.
/// - `predictions` — per-sample model confidence in \[0, 1\] (used only for
///   [`SampleWeightingStrategy::FocalLoss`]).  Pass `None` for other
///   strategies.  Passing `None` for focal loss returns
///   [`ReweightingError::LengthMismatch`].
/// - `strategy` — which weighting rule to apply.
/// - `config_focal` — focal-loss parameters (used when `strategy ==
///   FocalLoss`).
/// - `config_hardness` — hardness parameters (used when `strategy ==
///   Hardness`).
///
/// # Errors
/// Propagates errors from the underlying weighting function.
pub fn compute_sample_reweights(
    losses: &[f32],
    predictions: Option<&[f32]>,
    strategy: SampleWeightingStrategy,
    config_focal: &FocalLossConfig,
    config_hardness: &HardnessConfig,
) -> Result<SampleWeights, ReweightingError> {
    if losses.is_empty() {
        return Err(ReweightingError::EmptyLosses);
    }

    let weights = match strategy {
        SampleWeightingStrategy::Uniform => compute_uniform_weights(losses.len()),

        SampleWeightingStrategy::FocalLoss => {
            let preds = predictions.ok_or(ReweightingError::LengthMismatch {
                losses_len: losses.len(),
                pred_len: 0,
            })?;
            if preds.len() != losses.len() {
                return Err(ReweightingError::LengthMismatch {
                    losses_len: losses.len(),
                    pred_len: preds.len(),
                });
            }
            compute_focal_weights(preds, config_focal.gamma, config_focal.alpha)?
        }

        SampleWeightingStrategy::Hardness => compute_hardness_weights(losses, config_hardness)?,

        SampleWeightingStrategy::InverseHardness => {
            compute_inverse_hardness_weights(losses, config_hardness.temperature)?
        }

        SampleWeightingStrategy::Exponential => {
            // Use beta = clamp(1/temperature, 0, 1) as the exponential parameter
            // when dispatching through this generic interface.
            let beta = (1.0 / config_hardness.temperature).clamp(0.0, 1.0);
            compute_exponential_weights(losses, beta)?
        }

        SampleWeightingStrategy::Rank => compute_rank_weights(losses)?,
    };

    Ok(build_sample_weights(weights, strategy))
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_sample_weights / weighted_mean_loss
// ─────────────────────────────────────────────────────────────────────────────

/// Element-wise multiply each loss by the corresponding weight.
///
/// # Errors
/// - [`ReweightingError::LengthMismatch`] if lengths differ.
/// - [`ReweightingError::EmptyLosses`] if both are empty.
pub fn apply_sample_weights(losses: &[f32], weights: &[f32]) -> Result<Vec<f32>, ReweightingError> {
    if losses.is_empty() {
        return Err(ReweightingError::EmptyLosses);
    }
    if losses.len() != weights.len() {
        return Err(ReweightingError::LengthMismatch {
            losses_len: losses.len(),
            pred_len: weights.len(),
        });
    }
    Ok(losses
        .iter()
        .zip(weights.iter())
        .map(|(&l, &w)| l * w)
        .collect())
}

/// Weighted mean of a loss vector.
///
/// `result = sum(losses * weights) / sum(weights)`.
///
/// # Errors
/// - [`ReweightingError::EmptyLosses`] if either slice is empty.
/// - [`ReweightingError::LengthMismatch`] if lengths differ.
pub fn weighted_mean_loss(losses: &[f32], weights: &[f32]) -> Result<f32, ReweightingError> {
    if losses.is_empty() {
        return Err(ReweightingError::EmptyLosses);
    }
    if losses.len() != weights.len() {
        return Err(ReweightingError::LengthMismatch {
            losses_len: losses.len(),
            pred_len: weights.len(),
        });
    }

    let numerator: f32 = losses
        .iter()
        .zip(weights.iter())
        .map(|(&l, &w)| l * w)
        .sum();
    let denominator: f32 = weights.iter().sum();

    if denominator.abs() < f32::EPSILON {
        // All weights zero: fall back to simple mean.
        Ok(losses.iter().sum::<f32>() / losses.len() as f32)
    } else {
        Ok(numerator / denominator)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// interpolate_strategies / curriculum_weights
// ─────────────────────────────────────────────────────────────────────────────

/// Linearly interpolate two weight vectors.
///
/// `result[i] = (1 - t) * start[i] + t * end[i]`.
///
/// `t` is clamped to \[0, 1\].  If the vectors have different lengths, the
/// shorter one is padded with 1.0 (uniform weight).
pub fn interpolate_strategies(start_weights: &[f32], end_weights: &[f32], t: f32) -> Vec<f32> {
    let t = t.clamp(0.0, 1.0);
    let n = start_weights.len().max(end_weights.len());
    (0..n)
        .map(|i| {
            let s = start_weights.get(i).copied().unwrap_or(1.0);
            let e = end_weights.get(i).copied().unwrap_or(1.0);
            (1.0 - t) * s + t * e
        })
        .collect()
}

/// Compute curriculum-aware per-sample weights.
///
/// At step 0 the `start_strategy` weights are used; at step ≥ `warmup_steps`
/// the `end_strategy` weights are used.  Between those extremes the two weight
/// vectors are linearly interpolated.
///
/// The final weight vector is normalised to `mean == 1.0`.
///
/// # Errors
/// Propagates errors from the underlying weighting functions.
pub fn curriculum_weights(
    losses: &[f32],
    predictions: Option<&[f32]>,
    config: &CurriculumWeightConfig,
    current_step: usize,
) -> Result<Vec<f32>, ReweightingError> {
    if losses.is_empty() {
        return Err(ReweightingError::EmptyLosses);
    }

    let t = if config.warmup_steps == 0 {
        1.0f32
    } else {
        (current_step as f32 / config.warmup_steps as f32).min(1.0)
    };

    let focal_config = FocalLossConfig::default();
    let hardness_config = HardnessConfig::default();

    let start_weights = strategy_weights(
        losses,
        predictions,
        config.start_strategy,
        &focal_config,
        &hardness_config,
    )?;
    let end_weights = strategy_weights(
        losses,
        predictions,
        config.end_strategy,
        &focal_config,
        &hardness_config,
    )?;

    let interpolated = interpolate_strategies(&start_weights, &end_weights, t);

    // Normalise to mean == 1.0.
    normalize_to_mean_one(interpolated)
}

// ─────────────────────────────────────────────────────────────────────────────
// Analysis / utility functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Shannon entropy of a weight distribution.
///
/// Weights are converted to a probability distribution (`p_i = w_i / sum(w)`)
/// and the entropy `H = -sum(p_i * log2(p_i + ε))` is returned.
///
/// High entropy → more uniform weights; low entropy → concentrated weights.
pub fn compute_weight_entropy(weights: &[f32]) -> f32 {
    if weights.is_empty() {
        return 0.0;
    }
    let sum: f32 = weights.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    weights
        .iter()
        .map(|&w| {
            let p = w / sum;
            -p * (p + 1e-10_f32).log2()
        })
        .sum()
}

/// Format a human-readable summary of a [`SampleWeights`].
///
/// Format: `"SampleWeights[n=N, strategy=S]: mean=1.00, max=X.XX, ESS=YY.Y%"`.
pub fn format_weight_summary(sw: &SampleWeights) -> String {
    let n = sw.weights.len();
    let ess_pct = if n > 0 {
        sw.effective_sample_size / n as f32 * 100.0
    } else {
        0.0
    };
    let strategy_name = match sw.strategy {
        SampleWeightingStrategy::Uniform => "Uniform",
        SampleWeightingStrategy::FocalLoss => "FocalLoss",
        SampleWeightingStrategy::Hardness => "Hardness",
        SampleWeightingStrategy::InverseHardness => "InverseHardness",
        SampleWeightingStrategy::Exponential => "Exponential",
        SampleWeightingStrategy::Rank => "Rank",
    };
    format!(
        "SampleWeights[n={}, strategy={}]: mean={:.2}, max={:.2}, ESS={:.1}%",
        n, strategy_name, sw.mean_weight, sw.max_weight, ess_pct
    )
}

/// Detect whether weights have collapsed onto a small fraction of samples.
///
/// Returns `true` if the top 1% of samples (at least 1 sample) hold more than
/// `collapse_ratio` of the total weight sum.
///
/// `collapse_ratio` should be in \[0, 1\]; the default meaningful value is
/// `0.5` (50% of weight concentrated in 1% of samples).
pub fn detect_weight_collapse(weights: &[f32], collapse_ratio: f32) -> bool {
    if weights.is_empty() {
        return false;
    }

    let total: f32 = weights.iter().sum();
    if total <= 0.0 {
        return false;
    }

    let n = weights.len();
    let top_k = (n / 100).max(1);

    // Sort weights descending.
    let mut sorted = weights.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let top_sum: f32 = sorted.iter().take(top_k).sum();
    top_sum / total > collapse_ratio
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Normalise a weight vector so that `mean == 1.0` (i.e. `sum == n`).
///
/// If the sum is zero, returns uniform weights (all 1.0).
fn normalize_to_mean_one(mut weights: Vec<f32>) -> Result<Vec<f32>, ReweightingError> {
    let n = weights.len();
    if n == 0 {
        return Err(ReweightingError::EmptyLosses);
    }
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        let scale = n as f32 / sum;
        for w in &mut weights {
            *w *= scale;
        }
    } else {
        // All weights zero — return uniform.
        weights.fill(1.0);
    }
    Ok(weights)
}

/// Internal helper: compute raw weights for a single strategy.
fn strategy_weights(
    losses: &[f32],
    predictions: Option<&[f32]>,
    strategy: SampleWeightingStrategy,
    focal_config: &FocalLossConfig,
    hardness_config: &HardnessConfig,
) -> Result<Vec<f32>, ReweightingError> {
    match strategy {
        SampleWeightingStrategy::Uniform => Ok(compute_uniform_weights(losses.len())),

        SampleWeightingStrategy::FocalLoss => {
            let preds = predictions.ok_or(ReweightingError::LengthMismatch {
                losses_len: losses.len(),
                pred_len: 0,
            })?;
            if preds.len() != losses.len() {
                return Err(ReweightingError::LengthMismatch {
                    losses_len: losses.len(),
                    pred_len: preds.len(),
                });
            }
            compute_focal_weights(preds, focal_config.gamma, focal_config.alpha)
        }

        SampleWeightingStrategy::Hardness => compute_hardness_weights(losses, hardness_config),

        SampleWeightingStrategy::InverseHardness => {
            compute_inverse_hardness_weights(losses, hardness_config.temperature)
        }

        SampleWeightingStrategy::Exponential => {
            let beta = (1.0 / hardness_config.temperature).clamp(0.0, 1.0);
            compute_exponential_weights(losses, beta)
        }

        SampleWeightingStrategy::Rank => compute_rank_weights(losses),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn assert_approx(a: f32, b: f32, tol: f32, msg: &str) {
        assert!((a - b).abs() < tol, "{}: got {}, expected {}", msg, a, b);
    }

    fn assert_mean_one(weights: &[f32]) {
        let n = weights.len() as f32;
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - n).abs() < 1e-3,
            "weights should sum to n={}, got sum={}",
            n,
            sum
        );
    }

    // ── FocalLossConfig ───────────────────────────────────────────────────────

    // 1. Default values.
    #[test]
    fn test_focal_config_default() {
        let cfg = FocalLossConfig::default();
        assert_approx(cfg.gamma, 2.0, 1e-6, "default gamma");
        assert_approx(cfg.alpha, 0.5, 1e-6, "default alpha");
    }

    // 2. Validate succeeds for non-negative gamma.
    #[test]
    fn test_focal_config_validate_ok() {
        let cfg = FocalLossConfig {
            gamma: 0.0,
            alpha: 0.5,
        };
        assert!(cfg.validate().is_ok());
    }

    // 3. Validate returns error for negative gamma.
    #[test]
    fn test_focal_config_validate_negative_gamma() {
        let cfg = FocalLossConfig {
            gamma: -0.1,
            alpha: 0.5,
        };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, ReweightingError::InvalidGamma { .. }));
    }

    // ── HardnessConfig ────────────────────────────────────────────────────────

    // 4. Default temperature.
    #[test]
    fn test_hardness_config_default() {
        let cfg = HardnessConfig::default();
        assert_approx(cfg.temperature, 1.0, 1e-6, "default temperature");
        assert_approx(cfg.max_weight_ratio, 10.0, 1e-6, "default max_weight_ratio");
    }

    // 5. Validate returns error for temperature <= 0.
    #[test]
    fn test_hardness_config_validate_zero_temp() {
        let cfg = HardnessConfig {
            temperature: 0.0,
            max_weight_ratio: 10.0,
        };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, ReweightingError::InvalidTemperature { .. }));
    }

    // 6. Validate returns error for negative temperature.
    #[test]
    fn test_hardness_config_validate_negative_temp() {
        let cfg = HardnessConfig {
            temperature: -1.0,
            max_weight_ratio: 10.0,
        };
        assert!(cfg.validate().is_err());
    }

    // ── compute_focal_weights ─────────────────────────────────────────────────

    // 7. Perfect prediction → weight zero.
    #[test]
    fn test_focal_perfect_prediction_weight_zero() {
        // p=1.0 → (1-1)^2 = 0.
        let preds = vec![1.0_f32, 0.5, 0.5];
        let weights = compute_focal_weights(&preds, 2.0, 0.5).unwrap();
        // First sample should have the lowest weight (approaching 0).
        assert!(weights[0] < weights[1] * 0.01);
    }

    // 8. Zero prediction → weight equals alpha (before normalisation: after,
    //    still the maximum in a uniform set).
    #[test]
    fn test_focal_zero_prediction_max_weight() {
        let preds = vec![0.0_f32];
        let weights = compute_focal_weights(&preds, 2.0, 0.5).unwrap();
        // Single element → normalised to 1.0.
        assert_approx(weights[0], 1.0, 1e-4, "single sample weight");
    }

    // 9. Empty predictions → EmptyLosses error.
    #[test]
    fn test_focal_empty_predictions() {
        let err = compute_focal_weights(&[], 2.0, 0.5).unwrap_err();
        assert!(matches!(err, ReweightingError::EmptyLosses));
    }

    // 10. Negative gamma → InvalidGamma error.
    #[test]
    fn test_focal_negative_gamma_error() {
        let err = compute_focal_weights(&[0.5], -1.0, 0.5).unwrap_err();
        assert!(matches!(err, ReweightingError::InvalidGamma { .. }));
    }

    // 11. Weights are normalised (mean == 1).
    #[test]
    fn test_focal_weights_normalised() {
        let preds = vec![0.1, 0.5, 0.9, 0.3, 0.7];
        let weights = compute_focal_weights(&preds, 2.0, 0.5).unwrap();
        assert_mean_one(&weights);
    }

    // 11b. Regression: alpha must have a genuine (non-cancelling) effect
    //      on the normalised weight distribution when it differs from
    //      0.5 — this was previously a no-op because a uniform scalar
    //      multiplier cancels under mean-normalisation.
    #[test]
    fn test_focal_alpha_has_real_effect() {
        let preds = vec![0.9_f32, 0.1, 0.5];
        let gamma = 2.0;
        let w_alpha_02 = compute_focal_weights(&preds, gamma, 0.2).unwrap();
        let w_alpha_08 = compute_focal_weights(&preds, gamma, 0.8).unwrap();
        let max_diff = w_alpha_02
            .iter()
            .zip(w_alpha_08.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-2,
            "alpha=0.2 vs alpha=0.8 should give visibly different weights, \
             got alpha_0.2={w_alpha_02:?} alpha_0.8={w_alpha_08:?} (max_diff={max_diff})"
        );
    }

    // ── compute_hardness_weights ──────────────────────────────────────────────

    // 12. Higher loss → higher weight.
    #[test]
    fn test_hardness_higher_loss_higher_weight() {
        let losses = vec![0.1_f32, 0.5, 0.9];
        let cfg = HardnessConfig::default();
        let weights = compute_hardness_weights(&losses, &cfg).unwrap();
        assert!(
            weights[2] > weights[1],
            "loss 0.9 should get higher weight than 0.5"
        );
        assert!(
            weights[1] > weights[0],
            "loss 0.5 should get higher weight than 0.1"
        );
    }

    // 13. Equal losses → equal weights.
    #[test]
    fn test_hardness_equal_losses_equal_weights() {
        let losses = vec![0.5_f32; 5];
        let cfg = HardnessConfig::default();
        let weights = compute_hardness_weights(&losses, &cfg).unwrap();
        for &w in &weights {
            assert_approx(w, 1.0, 1e-4, "uniform loss should give uniform weight");
        }
    }

    // 14. Empty losses → error.
    #[test]
    fn test_hardness_empty_losses() {
        let cfg = HardnessConfig::default();
        let err = compute_hardness_weights(&[], &cfg).unwrap_err();
        assert!(matches!(err, ReweightingError::EmptyLosses));
    }

    // 15. Normalised (mean == 1).
    #[test]
    fn test_hardness_weights_normalised() {
        let losses = vec![0.2, 0.8, 0.4, 1.0, 0.6];
        let cfg = HardnessConfig::default();
        let weights = compute_hardness_weights(&losses, &cfg).unwrap();
        assert_mean_one(&weights);
    }

    // 15b. Regression: a clipped weight must never be rescaled back above
    //      the cap. The previous implementation rescaled ALL weights
    //      (including already-clipped ones) by `n / clip_sum`, and since
    //      clipping only reduces the sum, that factor was always > 1.
    #[test]
    fn test_hardness_weights_never_exceed_cap() {
        let losses = vec![0.0_f32, 0.0, 0.0, 0.0, 5.0];
        let cfg = HardnessConfig {
            temperature: 1.0,
            max_weight_ratio: 2.0,
        };
        let weights = compute_hardness_weights(&losses, &cfg).unwrap();
        let cap = cfg.max_weight_ratio;
        for (i, &w) in weights.iter().enumerate() {
            assert!(w <= cap + 1e-3, "weight[{i}]={w} exceeds cap={cap}");
        }
        // Mean should still be preserved at 1.0 (sum == n) — clipping
        // alone must not silently change the overall weight budget.
        let n = losses.len() as f32;
        let sum: f32 = weights.iter().sum();
        assert!((sum - n).abs() < 1e-2, "sum should stay ≈ n={n}, got {sum}");
    }

    // ── compute_inverse_hardness_weights ──────────────────────────────────────

    // 16. Lower loss → higher weight.
    #[test]
    fn test_inverse_hardness_lower_loss_higher_weight() {
        let losses = vec![0.1_f32, 0.5, 0.9];
        let weights = compute_inverse_hardness_weights(&losses, 1.0).unwrap();
        assert!(
            weights[0] > weights[1],
            "loss 0.1 should get higher weight than 0.5"
        );
        assert!(
            weights[1] > weights[2],
            "loss 0.5 should get higher weight than 0.9"
        );
    }

    // 17. Normalised.
    #[test]
    fn test_inverse_hardness_normalised() {
        let losses = vec![0.1, 0.5, 0.9, 0.3];
        let weights = compute_inverse_hardness_weights(&losses, 0.5).unwrap();
        assert_mean_one(&weights);
    }

    // 18. Invalid temperature → error.
    #[test]
    fn test_inverse_hardness_invalid_temperature() {
        let err = compute_inverse_hardness_weights(&[0.5], 0.0).unwrap_err();
        assert!(matches!(err, ReweightingError::InvalidTemperature { .. }));
    }

    // ── compute_exponential_weights ───────────────────────────────────────────

    // 19. beta=0 → all weights equal.
    #[test]
    fn test_exponential_beta_zero_uniform() {
        let losses = vec![0.1_f32, 0.5, 0.9];
        let weights = compute_exponential_weights(&losses, 0.0).unwrap();
        for &w in &weights {
            assert_approx(w, 1.0, 1e-5, "beta=0 should give uniform weights");
        }
    }

    // 20. Positive beta → higher loss = higher weight.
    #[test]
    fn test_exponential_positive_beta_hardness() {
        let losses = vec![0.1_f32, 0.5, 0.9];
        let weights = compute_exponential_weights(&losses, 0.5).unwrap();
        assert!(
            weights[2] > weights[0],
            "highest loss should get highest weight"
        );
    }

    // 21. Beta out of range → error.
    #[test]
    fn test_exponential_invalid_beta() {
        let err = compute_exponential_weights(&[0.5], 1.1).unwrap_err();
        assert!(matches!(err, ReweightingError::InvalidBeta { .. }));
    }

    // 22. Empty losses → error.
    #[test]
    fn test_exponential_empty() {
        let err = compute_exponential_weights(&[], 0.5).unwrap_err();
        assert!(matches!(err, ReweightingError::EmptyLosses));
    }

    // ── compute_rank_weights ──────────────────────────────────────────────────

    // 23. Lowest loss gets lowest weight, highest loss gets highest weight.
    #[test]
    fn test_rank_weights_ordering() {
        let losses = vec![0.9_f32, 0.1, 0.5]; // index 1 lowest, index 0 highest
        let weights = compute_rank_weights(&losses).unwrap();
        // index 0 has highest loss → highest rank → highest weight
        assert!(weights[0] > weights[2], "loss 0.9 should beat 0.5");
        assert!(weights[2] > weights[1], "loss 0.5 should beat 0.1");
    }

    // 24. Known rank assignment: sorted losses [0.1, 0.5, 0.9].
    #[test]
    fn test_rank_weights_known_values() {
        let losses = vec![0.1_f32, 0.5, 0.9]; // ranks: 1, 2, 3
        let weights = compute_rank_weights(&losses).unwrap();
        // mean rank = 2.0; weights = 0.5, 1.0, 1.5
        assert_approx(weights[0], 0.5, 1e-4, "rank 1/2");
        assert_approx(weights[1], 1.0, 1e-4, "rank 2/2");
        assert_approx(weights[2], 1.5, 1e-4, "rank 3/2");
    }

    // 25. Normalised.
    #[test]
    fn test_rank_weights_normalised() {
        let losses = vec![0.2, 0.8, 0.4, 1.0, 0.6];
        let weights = compute_rank_weights(&losses).unwrap();
        assert_mean_one(&weights);
    }

    // 26. Empty losses → error.
    #[test]
    fn test_rank_weights_empty() {
        let err = compute_rank_weights(&[]).unwrap_err();
        assert!(matches!(err, ReweightingError::EmptyLosses));
    }

    // ── compute_uniform_weights ───────────────────────────────────────────────

    // 27. All weights equal 1.0.
    #[test]
    fn test_uniform_weights_all_one() {
        let weights = compute_uniform_weights(5);
        assert_eq!(weights.len(), 5);
        for &w in &weights {
            assert_approx(w, 1.0, 1e-6, "uniform weight");
        }
    }

    // 28. Zero count → empty vec.
    #[test]
    fn test_uniform_weights_zero() {
        let weights = compute_uniform_weights(0);
        assert!(weights.is_empty());
    }

    // ── build_sample_weights ──────────────────────────────────────────────────

    // 29. Correct statistics.
    #[test]
    fn test_build_sample_weights_statistics() {
        let weights = vec![0.5_f32, 1.0, 1.5];
        let sw = build_sample_weights(weights, SampleWeightingStrategy::Rank);
        assert_approx(sw.mean_weight, 1.0, 1e-4, "mean weight");
        assert_approx(sw.max_weight, 1.5, 1e-4, "max weight");
        assert_approx(sw.min_weight, 0.5, 1e-4, "min weight");
    }

    // 30. ESS == n for uniform weights.
    #[test]
    fn test_build_sample_weights_ess_uniform() {
        let n = 10usize;
        let weights = compute_uniform_weights(n);
        let sw = build_sample_weights(weights, SampleWeightingStrategy::Uniform);
        // ESS = n² / sum(w²) = n² / n = n
        assert_approx(sw.effective_sample_size, n as f32, 1e-3, "ESS for uniform");
    }

    // 31. ESS < n for concentrated weights.
    #[test]
    fn test_build_sample_weights_ess_concentrated() {
        let mut weights = vec![0.0_f32; 10];
        weights[0] = 10.0; // all weight on one sample
        let sw = build_sample_weights(weights, SampleWeightingStrategy::Hardness);
        // ESS = 100 / 100 = 1.0 (only one effective sample)
        assert_approx(sw.effective_sample_size, 1.0, 1e-3, "ESS for concentrated");
    }

    // ── compute_sample_reweights ──────────────────────────────────────────────

    // 32. Uniform strategy.
    #[test]
    fn test_compute_sample_reweights_uniform() {
        let losses = vec![0.1, 0.5, 0.9];
        let sw = compute_sample_reweights(
            &losses,
            None,
            SampleWeightingStrategy::Uniform,
            &FocalLossConfig::default(),
            &HardnessConfig::default(),
        )
        .unwrap();
        assert_eq!(sw.strategy, SampleWeightingStrategy::Uniform);
        assert_approx(sw.mean_weight, 1.0, 1e-4, "mean weight");
    }

    // 33. Hardness strategy dispatches correctly.
    #[test]
    fn test_compute_sample_reweights_hardness() {
        let losses = vec![0.1, 0.5, 0.9];
        let sw = compute_sample_reweights(
            &losses,
            None,
            SampleWeightingStrategy::Hardness,
            &FocalLossConfig::default(),
            &HardnessConfig::default(),
        )
        .unwrap();
        assert_eq!(sw.strategy, SampleWeightingStrategy::Hardness);
        // Higher loss → higher weight.
        assert!(sw.weights[2] > sw.weights[0]);
    }

    // 34. FocalLoss without predictions → error.
    #[test]
    fn test_compute_sample_reweights_focal_no_predictions() {
        let losses = vec![0.1, 0.5, 0.9];
        let err = compute_sample_reweights(
            &losses,
            None,
            SampleWeightingStrategy::FocalLoss,
            &FocalLossConfig::default(),
            &HardnessConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(err, ReweightingError::LengthMismatch { .. }));
    }

    // 35. FocalLoss with predictions dispatches correctly.
    #[test]
    fn test_compute_sample_reweights_focal_with_predictions() {
        let losses = vec![0.1, 0.5, 0.9];
        let preds = vec![0.9, 0.5, 0.1];
        let sw = compute_sample_reweights(
            &losses,
            Some(&preds),
            SampleWeightingStrategy::FocalLoss,
            &FocalLossConfig::default(),
            &HardnessConfig::default(),
        )
        .unwrap();
        assert_eq!(sw.strategy, SampleWeightingStrategy::FocalLoss);
    }

    // 36. Rank strategy dispatches correctly.
    #[test]
    fn test_compute_sample_reweights_rank() {
        let losses = vec![0.1, 0.5, 0.9];
        let sw = compute_sample_reweights(
            &losses,
            None,
            SampleWeightingStrategy::Rank,
            &FocalLossConfig::default(),
            &HardnessConfig::default(),
        )
        .unwrap();
        assert_eq!(sw.strategy, SampleWeightingStrategy::Rank);
    }

    // ── apply_sample_weights ──────────────────────────────────────────────────

    // 37. Known multiplication.
    #[test]
    fn test_apply_sample_weights_known() {
        let losses = vec![1.0, 2.0, 3.0];
        let weights = vec![2.0, 0.5, 1.0];
        let result = apply_sample_weights(&losses, &weights).unwrap();
        assert_approx(result[0], 2.0, 1e-5, "1*2");
        assert_approx(result[1], 1.0, 1e-5, "2*0.5");
        assert_approx(result[2], 3.0, 1e-5, "3*1");
    }

    // 38. Length mismatch → error.
    #[test]
    fn test_apply_sample_weights_length_mismatch() {
        let err = apply_sample_weights(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(matches!(err, ReweightingError::LengthMismatch { .. }));
    }

    // 39. Empty losses → EmptyLosses error.
    #[test]
    fn test_apply_sample_weights_empty() {
        let err = apply_sample_weights(&[], &[]).unwrap_err();
        assert!(matches!(err, ReweightingError::EmptyLosses));
    }

    // ── weighted_mean_loss ────────────────────────────────────────────────────

    // 40. Uniform weights → regular mean.
    #[test]
    fn test_weighted_mean_uniform() {
        let losses = vec![0.2, 0.4, 0.6];
        let weights = vec![1.0, 1.0, 1.0];
        let mean = weighted_mean_loss(&losses, &weights).unwrap();
        assert_approx(mean, 0.4, 1e-5, "uniform-weighted mean");
    }

    // 41. Known values.
    #[test]
    fn test_weighted_mean_known() {
        let losses = vec![1.0, 2.0];
        let weights = vec![3.0, 1.0]; // 3*1 + 1*2 / (3+1) = 5/4
        let mean = weighted_mean_loss(&losses, &weights).unwrap();
        assert_approx(mean, 1.25, 1e-5, "weighted mean");
    }

    // 42. Empty losses → error.
    #[test]
    fn test_weighted_mean_empty() {
        let err = weighted_mean_loss(&[], &[]).unwrap_err();
        assert!(matches!(err, ReweightingError::EmptyLosses));
    }

    // ── interpolate_strategies ────────────────────────────────────────────────

    // 43. t=0 → start weights.
    #[test]
    fn test_interpolate_t_zero() {
        let start = vec![0.5_f32, 1.5];
        let end = vec![1.5_f32, 0.5];
        let result = interpolate_strategies(&start, &end, 0.0);
        assert_approx(result[0], 0.5, 1e-5, "t=0 result[0]");
        assert_approx(result[1], 1.5, 1e-5, "t=0 result[1]");
    }

    // 44. t=1 → end weights.
    #[test]
    fn test_interpolate_t_one() {
        let start = vec![0.5_f32, 1.5];
        let end = vec![1.5_f32, 0.5];
        let result = interpolate_strategies(&start, &end, 1.0);
        assert_approx(result[0], 1.5, 1e-5, "t=1 result[0]");
        assert_approx(result[1], 0.5, 1e-5, "t=1 result[1]");
    }

    // 45. t=0.5 → midpoint.
    #[test]
    fn test_interpolate_t_half() {
        let start = vec![0.0_f32, 2.0];
        let end = vec![2.0_f32, 0.0];
        let result = interpolate_strategies(&start, &end, 0.5);
        assert_approx(result[0], 1.0, 1e-5, "t=0.5 result[0]");
        assert_approx(result[1], 1.0, 1e-5, "t=0.5 result[1]");
    }

    // ── curriculum_weights ────────────────────────────────────────────────────

    // 46. At step=0 uses start strategy (InverseHardness).
    #[test]
    fn test_curriculum_step_zero_uses_start_strategy() {
        let losses = vec![0.1_f32, 0.9];
        let config = CurriculumWeightConfig {
            start_strategy: SampleWeightingStrategy::InverseHardness,
            end_strategy: SampleWeightingStrategy::Hardness,
            warmup_steps: 100,
        };
        let weights = curriculum_weights(&losses, None, &config, 0).unwrap();
        // InverseHardness: lower loss → higher weight.
        assert!(
            weights[0] > weights[1],
            "easy sample should dominate at step 0"
        );
    }

    // 47. At step >= warmup uses end strategy (Hardness).
    #[test]
    fn test_curriculum_step_warmup_uses_end_strategy() {
        let losses = vec![0.1_f32, 0.9];
        let config = CurriculumWeightConfig {
            start_strategy: SampleWeightingStrategy::InverseHardness,
            end_strategy: SampleWeightingStrategy::Hardness,
            warmup_steps: 100,
        };
        let weights = curriculum_weights(&losses, None, &config, 100).unwrap();
        // Hardness: higher loss → higher weight.
        assert!(
            weights[1] > weights[0],
            "hard sample should dominate at step >= warmup"
        );
    }

    // ── WeightTracker ─────────────────────────────────────────────────────────

    // 48. Record and evict oldest entry at capacity.
    #[test]
    fn test_weight_tracker_record_evict() {
        let mut tracker = WeightTracker::new(3);
        tracker.record(0, 1.0, 2.0);
        tracker.record(1, 1.0, 3.0);
        tracker.record(2, 1.0, 4.0);
        assert_eq!(tracker.history.len(), 3);
        // Record a 4th entry — oldest should be evicted.
        tracker.record(3, 1.0, 5.0);
        assert_eq!(tracker.history.len(), 3);
        // The first entry (step=0) should be gone.
        assert_eq!(tracker.history[0].0, 1);
    }

    // 48b. Regression test: eviction at capacity must use O(1) `pop_front`
    // (via `VecDeque`) rather than `Vec::remove(0)`, while preserving FIFO
    // order across many evictions.
    #[test]
    fn test_weight_tracker_record_evict_preserves_fifo_order() {
        let mut tracker = WeightTracker::new(5);
        for step in 0..100usize {
            tracker.record(step, 1.0, step as f32);
        }
        assert_eq!(tracker.history.len(), 5);
        // Oldest surviving step should be 95, newest 99, in order.
        let steps: Vec<usize> = tracker.history.iter().map(|&(s, _, _)| s).collect();
        assert_eq!(steps, vec![95, 96, 97, 98, 99]);
    }

    // 49. recent_max_weight_trend positive when weights are growing.
    #[test]
    fn test_weight_tracker_trend_positive() {
        let mut tracker = WeightTracker::new(10);
        for step in 0..5usize {
            tracker.record(step, 1.0, step as f32 * 0.1 + 1.0);
        }
        let trend = tracker.recent_max_weight_trend(5).unwrap();
        assert!(trend > 0.0, "growing max_weight should give positive slope");
    }

    // 50. recent_max_weight_trend negative when weights are shrinking.
    #[test]
    fn test_weight_tracker_trend_negative() {
        let mut tracker = WeightTracker::new(10);
        for step in 0..5usize {
            tracker.record(step, 1.0, 5.0 - step as f32 * 0.1);
        }
        let trend = tracker.recent_max_weight_trend(5).unwrap();
        assert!(
            trend < 0.0,
            "shrinking max_weight should give negative slope"
        );
    }

    // 51. recent_max_weight_trend None with fewer than 2 entries.
    #[test]
    fn test_weight_tracker_trend_none_single_entry() {
        let mut tracker = WeightTracker::new(10);
        tracker.record(0, 1.0, 2.0);
        assert!(tracker.recent_max_weight_trend(5).is_none());
    }

    // ── compute_weight_entropy ────────────────────────────────────────────────

    // 52. Uniform weights → maximum entropy.
    #[test]
    fn test_weight_entropy_uniform_is_max() {
        let n = 4usize;
        let uniform = vec![1.0_f32; n];
        let skewed = vec![3.5_f32, 0.1, 0.2, 0.2];
        let h_uniform = compute_weight_entropy(&uniform);
        let h_skewed = compute_weight_entropy(&skewed);
        assert!(
            h_uniform > h_skewed,
            "uniform entropy {} should exceed skewed entropy {}",
            h_uniform,
            h_skewed
        );
    }

    // 53. Single sample → entropy is 0 (all weight on one sample, p=1).
    #[test]
    fn test_weight_entropy_single_sample() {
        let weights = vec![5.0_f32];
        let h = compute_weight_entropy(&weights);
        // p=1, H = -1 * log2(1 + 1e-10) ≈ 0
        assert!(
            h.abs() < 0.01,
            "single sample entropy should be ~0, got {}",
            h
        );
    }

    // 54. Empty weights → 0.
    #[test]
    fn test_weight_entropy_empty() {
        assert_approx(compute_weight_entropy(&[]), 0.0, 1e-6, "empty entropy");
    }

    // ── format_weight_summary ─────────────────────────────────────────────────

    // 55. Non-empty string containing expected fields.
    #[test]
    fn test_format_weight_summary_contains_fields() {
        let sw = build_sample_weights(vec![0.5, 1.0, 1.5], SampleWeightingStrategy::Rank);
        let s = format_weight_summary(&sw);
        assert!(!s.is_empty());
        assert!(s.contains("n=3"), "should contain sample count: {}", s);
        assert!(s.contains("strategy="), "should contain strategy: {}", s);
        assert!(s.contains("mean="), "should contain mean: {}", s);
        assert!(s.contains("max="), "should contain max: {}", s);
        assert!(s.contains("ESS="), "should contain ESS: {}", s);
    }

    // ── detect_weight_collapse ────────────────────────────────────────────────

    // 56. Even distribution → not collapsed.
    #[test]
    fn test_detect_weight_collapse_even_false() {
        let weights = vec![1.0_f32; 200];
        assert!(!detect_weight_collapse(&weights, 0.5));
    }

    // 57. Extreme skew → collapsed.
    #[test]
    fn test_detect_weight_collapse_extreme_true() {
        let mut weights = vec![0.0_f32; 200];
        // Put 90% of weight on first two samples (1% of 200 = 2 samples).
        weights[0] = 45.0;
        weights[1] = 45.0;
        for w in weights.iter_mut().skip(2) {
            *w = 10.0 / 198.0;
        }
        assert!(detect_weight_collapse(&weights, 0.5));
    }

    // 58. Empty weights → false.
    #[test]
    fn test_detect_weight_collapse_empty() {
        assert!(!detect_weight_collapse(&[], 0.5));
    }

    // ── ReweightingError variants ─────────────────────────────────────────────

    // 59. EmptyLosses displays correctly.
    #[test]
    fn test_error_empty_losses_display() {
        let e = ReweightingError::EmptyLosses;
        assert!(!format!("{}", e).is_empty());
    }

    // 60. LengthMismatch contains both lengths.
    #[test]
    fn test_error_length_mismatch_display() {
        let e = ReweightingError::LengthMismatch {
            losses_len: 5,
            pred_len: 3,
        };
        let s = format!("{}", e);
        assert!(s.contains("5"), "should contain 5: {}", s);
        assert!(s.contains("3"), "should contain 3: {}", s);
    }

    // ── SampleWeightingStrategy variants ─────────────────────────────────────

    // 61. PartialEq works.
    #[test]
    fn test_strategy_partial_eq() {
        assert_eq!(
            SampleWeightingStrategy::Uniform,
            SampleWeightingStrategy::Uniform
        );
        assert_ne!(
            SampleWeightingStrategy::Uniform,
            SampleWeightingStrategy::Hardness
        );
    }

    // 62. Clone works.
    #[test]
    fn test_strategy_clone() {
        let s = SampleWeightingStrategy::FocalLoss;
        let s2 = s;
        assert_eq!(s, s2);
    }

    // ── SampleWeights fields ──────────────────────────────────────────────────

    // 63. weight_std is correct.
    #[test]
    fn test_sample_weights_std_correct() {
        // Weights [0.5, 1.0, 1.5]: mean=1, deviations [-0.5, 0, 0.5]
        // variance = (0.25 + 0 + 0.25) / 3 ≈ 0.1667, std ≈ 0.4082.
        let sw = build_sample_weights(vec![0.5, 1.0, 1.5], SampleWeightingStrategy::Rank);
        let expected_std = (2.0_f32 / (3.0 * 4.0)).sqrt(); // = sqrt(1/6)
        assert_approx(sw.weight_std, expected_std, 1e-4, "weight std");
    }

    // 64. InverseHardness in compute_sample_reweights dispatches correctly.
    #[test]
    fn test_compute_sample_reweights_inverse_hardness() {
        let losses = vec![0.1_f32, 0.9];
        let sw = compute_sample_reweights(
            &losses,
            None,
            SampleWeightingStrategy::InverseHardness,
            &FocalLossConfig::default(),
            &HardnessConfig::default(),
        )
        .unwrap();
        assert_eq!(sw.strategy, SampleWeightingStrategy::InverseHardness);
        assert!(
            sw.weights[0] > sw.weights[1],
            "low loss should get higher weight"
        );
    }

    // 65. Exponential in compute_sample_reweights dispatches correctly.
    #[test]
    fn test_compute_sample_reweights_exponential() {
        let losses = vec![0.1_f32, 0.5, 0.9];
        let sw = compute_sample_reweights(
            &losses,
            None,
            SampleWeightingStrategy::Exponential,
            &FocalLossConfig::default(),
            &HardnessConfig::default(),
        )
        .unwrap();
        assert_eq!(sw.strategy, SampleWeightingStrategy::Exponential);
        assert_mean_one(&sw.weights);
    }

    // 66. curriculum_weights normalised.
    #[test]
    fn test_curriculum_weights_normalised() {
        let losses = vec![0.1_f32, 0.3, 0.9, 0.5];
        let config = CurriculumWeightConfig::default();
        let weights = curriculum_weights(&losses, None, &config, 500).unwrap();
        assert_mean_one(&weights);
    }
}
