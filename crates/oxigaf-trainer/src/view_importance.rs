//! Per-view importance tracking for training view sampling.
//!
//! Computes and tracks importance scores per training view, enabling importance
//! sampling to prioritise views with high rendering error or high information
//! content. This is distinct from:
//!
//! - `camera_sampling` — geometric camera position generation
//! - `view_scheduler` — training-order scheduling of view angles
//! - `curriculum`     — training difficulty scheduling
//!
//! # Quick start
//! ```rust
//! use oxigaf_trainer::view_importance::{
//!     ViewImportanceConfig, ViewImportanceSampler, ImportanceStrategy,
//! };
//!
//! let config = ViewImportanceConfig {
//!     n_views: 4,
//!     strategy: ImportanceStrategy::ByLoss,
//!     ..ViewImportanceConfig::default()
//! };
//! let mut sampler = ViewImportanceSampler::new(config).expect("valid config");
//! sampler.update(0, 0.8, 1).expect("valid update");
//! sampler.update(1, 0.2, 1).expect("valid update");
//! let scores = sampler.compute_importance().expect("has views");
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// PRNG — xorshift64 (no rand crate)
// ─────────────────────────────────────────────────────────────────────────────

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = 1;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Returns a uniform f32 in [0, 1).
fn xorshift_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 40) as f32 / (1u64 << 24) as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the view-importance subsystem.
#[derive(Debug, Error)]
pub enum ViewImportanceError {
    #[error("No views registered — call register_view() first")]
    NoViews,

    #[error("View index {idx} out of range (n_views = {n_views})")]
    ViewIndexOutOfRange { idx: usize, n_views: usize },

    #[error("Loss history is empty for view {idx}")]
    EmptyViewHistory { idx: usize },

    #[error("Invalid temperature {temp}: must be > 0")]
    InvalidTemperature { temp: f32 },

    #[error("All importance weights are zero")]
    ZeroImportanceWeights,

    /// `history_window == 0` would make `update`'s eviction check
    /// (`hist.len() >= history_window`) true on an empty history, i.e.
    /// `Vec::remove(0)` on an empty vector.
    #[error("Invalid history_window {window}: must be >= 1")]
    InvalidHistoryWindow { window: usize },

    #[error("Invalid ema_decay {decay}: must be in (0.0, 1.0)")]
    InvalidEmaDecay { decay: f32 },
}

// ─────────────────────────────────────────────────────────────────────────────
// ImportanceStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for computing per-view importance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImportanceStrategy {
    /// All views equally important.
    Uniform,
    /// Higher loss → higher importance.
    ByLoss,
    /// High variance in loss → higher importance (unstable views).
    ByLossVariance,
    /// Views not sampled recently get higher importance.
    ByRecency,
    /// Weighted combination of `ByLoss` and `ByRecency`.
    Combined,
}

// ─────────────────────────────────────────────────────────────────────────────
// ViewImportanceConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for view importance tracking.
#[derive(Debug, Clone)]
pub struct ViewImportanceConfig {
    /// Total number of training views.
    pub n_views: usize,
    /// How many recent losses to track per view.
    pub history_window: usize,
    /// How to compute importance.
    pub strategy: ImportanceStrategy,
    /// Softmax temperature for sampling (must be > 0).
    pub temperature: f32,
    /// Weight for loss component in `Combined` strategy.
    pub loss_weight: f32,
    /// Weight for recency component in `Combined` strategy.
    pub recency_weight: f32,
    /// EMA decay for smoothing importance scores.
    pub ema_decay: f32,
}

impl Default for ViewImportanceConfig {
    fn default() -> Self {
        Self {
            n_views: 1,
            history_window: 20,
            strategy: ImportanceStrategy::ByLoss,
            temperature: 1.0,
            loss_weight: 0.7,
            recency_weight: 0.3,
            ema_decay: 0.9,
        }
    }
}

impl ViewImportanceConfig {
    /// Validate configuration, returning an error on invalid values.
    pub fn validate(&self) -> Result<(), ViewImportanceError> {
        if self.temperature <= 0.0 {
            return Err(ViewImportanceError::InvalidTemperature {
                temp: self.temperature,
            });
        }
        if self.n_views == 0 {
            return Err(ViewImportanceError::NoViews);
        }
        if self.history_window == 0 {
            return Err(ViewImportanceError::InvalidHistoryWindow {
                window: self.history_window,
            });
        }
        // The interval is OPEN at both ends, matching
        // `InvalidEmaDecay`'s "must be in (0.0, 1.0)" and every other EMA
        // decay in this crate (`online_learning`, `weight_averaging`,
        // `pose_conditioning`, `curriculum_learning`).  Both endpoints
        // degenerate `update`'s smoothing step
        // `ema = d·ema + (1 − d)·raw`: at `d == 0.0` there is no smoothing
        // at all (`ema == raw`, the EMA is pure noise), and at `d == 1.0`
        // the EMA is frozen at its uniform initial value and never sees a
        // single loss observation.  Written as "not strictly inside" rather
        // than `<= 0.0 || >= 1.0` so NaN is rejected too.
        let in_open_unit_interval = self.ema_decay > 0.0 && self.ema_decay < 1.0;
        if !in_open_unit_interval {
            return Err(ViewImportanceError::InvalidEmaDecay {
                decay: self.ema_decay,
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ViewImportanceScore
// ─────────────────────────────────────────────────────────────────────────────

/// Importance information for a single view.
#[derive(Debug, Clone)]
pub struct ViewImportanceScore {
    /// Index of this view.
    pub view_idx: usize,
    /// Raw importance score in [0, 1].
    pub importance: f32,
    /// Softmax-normalised sampling probability.
    pub sample_weight: f32,
    /// Mean loss over recent history.
    pub mean_loss: f32,
    /// Variance of the loss history.
    pub loss_variance: f32,
    /// Total times this view has been sampled.
    pub n_samples: usize,
    /// Step when last sampled (0 if never).
    pub last_sampled_step: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// ViewImportanceSampler
// ─────────────────────────────────────────────────────────────────────────────

/// Stateful tracker that assigns sampling importance to each training view.
pub struct ViewImportanceSampler {
    /// Configuration (immutable after construction).
    pub config: ViewImportanceConfig,
    /// Rolling loss history per view.
    loss_history: Vec<Vec<f32>>,
    /// Total number of samples recorded per view.
    n_samples: Vec<usize>,
    /// Step at which each view was last sampled (0 = never).
    last_sampled: Vec<usize>,
    /// EMA-smoothed importance per view.
    ema_importance: Vec<f32>,
    /// Current training step.
    current_step: usize,
}

impl ViewImportanceSampler {
    /// Construct a new sampler from the given configuration.
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: ViewImportanceConfig) -> Result<Self, ViewImportanceError> {
        config.validate()?;
        let n = config.n_views;
        let uniform = 1.0 / n as f32;
        Ok(Self {
            loss_history: vec![Vec::new(); n],
            n_samples: vec![0usize; n],
            last_sampled: vec![0usize; n],
            ema_importance: vec![uniform; n],
            current_step: 0,
            config,
        })
    }

    /// Record a loss observation for `view_idx` at training `step`.
    ///
    /// This evicts the oldest loss if the history window is exceeded,
    /// updates sample counts, and refreshes the EMA.
    pub fn update(
        &mut self,
        view_idx: usize,
        loss: f32,
        step: usize,
    ) -> Result<(), ViewImportanceError> {
        let n = self.config.n_views;
        if view_idx >= n {
            return Err(ViewImportanceError::ViewIndexOutOfRange {
                idx: view_idx,
                n_views: n,
            });
        }

        // Maintain rolling window. `config` is a `pub` field, so it can be
        // mutated to `history_window == 0` after construction even though
        // `validate()` now rejects that at `new()` time — a `while` (not
        // `if`) guarded by `!hist.is_empty()` is defence in depth against
        // that: `hist.len() >= 0` is always true, so it evicts down to
        // empty (net effect: a window of 1) instead of calling
        // `Vec::remove(0)` on an already-empty history, which panics.
        let hist = &mut self.loss_history[view_idx];
        while hist.len() >= self.config.history_window && !hist.is_empty() {
            hist.remove(0);
        }
        hist.push(loss);

        self.n_samples[view_idx] += 1;
        self.last_sampled[view_idx] = step;

        // Advance the step counter.
        if step > self.current_step {
            self.current_step = step;
        }

        // Update EMA importance.  The raw importance for this view is taken
        // as the (possibly new) mean loss; unsampled views stay at their
        // existing EMA value.
        let raw = compute_mean_loss(hist);
        let decay = self.config.ema_decay;
        self.ema_importance[view_idx] = decay * self.ema_importance[view_idx] + (1.0 - decay) * raw;

        Ok(())
    }

    /// Compute importance scores for all views using the configured strategy.
    ///
    /// Returns a `Vec<ViewImportanceScore>` sorted by `view_idx`.
    pub fn compute_importance(&self) -> Result<Vec<ViewImportanceScore>, ViewImportanceError> {
        let n = self.config.n_views;
        if n == 0 {
            return Err(ViewImportanceError::NoViews);
        }

        // Compute raw importances according to strategy.
        let raw: Vec<f32> = match self.config.strategy {
            ImportanceStrategy::Uniform => vec![1.0_f32; n],

            ImportanceStrategy::ByLoss => compute_loss_importance(&self.loss_history),

            ImportanceStrategy::ByLossVariance => compute_variance_importance(&self.loss_history),

            ImportanceStrategy::ByRecency => {
                compute_recency_importance(&self.last_sampled, self.current_step)
            }

            ImportanceStrategy::Combined => {
                let loss_imp = compute_loss_importance(&self.loss_history);
                let rec_imp = compute_recency_importance(&self.last_sampled, self.current_step);
                combine_importance(
                    &loss_imp,
                    &rec_imp,
                    self.config.loss_weight,
                    self.config.recency_weight,
                )
            }
        };

        // Normalise raw importances to [0, 1] (divides by max, or keeps as-is
        // if all are equal / zero — we avoid the ZeroImportanceWeights error
        // here and let softmax decide).
        let max_raw = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let norm_raw: Vec<f32> = if max_raw > 0.0 {
            raw.iter().map(|v| v / max_raw).collect()
        } else {
            vec![1.0 / n as f32; n]
        };

        // Softmax weights.
        let weights = importance_softmax(&norm_raw, self.config.temperature)?;

        // Build score structs.
        let scores = (0..n)
            .map(|i| ViewImportanceScore {
                view_idx: i,
                importance: norm_raw[i],
                sample_weight: weights[i],
                mean_loss: compute_mean_loss(&self.loss_history[i]),
                loss_variance: compute_loss_variance(&self.loss_history[i]),
                n_samples: self.n_samples[i],
                last_sampled_step: self.last_sampled[i],
            })
            .collect();

        Ok(scores)
    }

    /// Sample a single view index proportionally to current importance weights.
    ///
    /// Uses CDF inversion with xorshift64 PRNG — no `rand` crate required.
    pub fn sample_view(&mut self, rng_state: &mut u64) -> Result<usize, ViewImportanceError> {
        let scores = self.compute_importance()?;
        let weights: Vec<f32> = scores.iter().map(|s| s.sample_weight).collect();
        let chosen = sample_from_weights(&weights, rng_state)?;
        self.last_sampled[chosen] = self.current_step;
        Ok(chosen)
    }

    /// Return the indices of the top-`k` views by current importance score.
    pub fn top_k_views(&self, k: usize) -> Result<Vec<usize>, ViewImportanceError> {
        let scores = self.compute_importance()?;
        let mut indexed: Vec<(usize, f32)> =
            scores.iter().map(|s| (s.view_idx, s.importance)).collect();
        // Sort descending by importance.
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let take = k.min(indexed.len());
        Ok(indexed[..take].iter().map(|(idx, _)| *idx).collect())
    }

    /// Reset all history, counters, and EMA to the initial uniform state.
    pub fn reset(&mut self) {
        let n = self.config.n_views;
        let uniform = 1.0 / n as f32;
        for h in self.loss_history.iter_mut() {
            h.clear();
        }
        for v in self.n_samples.iter_mut() {
            *v = 0;
        }
        for v in self.last_sampled.iter_mut() {
            *v = 0;
        }
        for v in self.ema_importance.iter_mut() {
            *v = uniform;
        }
        self.current_step = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Free functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute importance proportional to mean loss.
///
/// Views with an empty history receive importance 1.0 (maximum) to promote
/// unsampled views.
pub fn compute_loss_importance(loss_history: &[Vec<f32>]) -> Vec<f32> {
    loss_history
        .iter()
        .map(|h| {
            if h.is_empty() {
                1.0_f32
            } else {
                compute_mean_loss(h)
            }
        })
        .collect()
}

/// Compute importance proportional to loss variance.
///
/// Views with fewer than 2 samples receive importance 1.0.
pub fn compute_variance_importance(loss_history: &[Vec<f32>]) -> Vec<f32> {
    loss_history
        .iter()
        .map(|h| {
            if h.len() < 2 {
                1.0_f32
            } else {
                compute_loss_variance(h)
            }
        })
        .collect()
}

/// Compute recency-based importance.
///
/// Importance = steps since last sample.  Views that have never been sampled
/// (`last_sampled == 0`) get `current_step + 1`.
pub fn compute_recency_importance(last_sampled: &[usize], current_step: usize) -> Vec<f32> {
    last_sampled
        .iter()
        .map(|&ls| {
            if ls == 0 {
                (current_step + 1) as f32
            } else {
                (current_step.saturating_sub(ls)) as f32
            }
        })
        .collect()
}

/// Combine loss and recency importances with configurable weights.
pub fn combine_importance(
    loss_imp: &[f32],
    recency_imp: &[f32],
    loss_weight: f32,
    recency_weight: f32,
) -> Vec<f32> {
    loss_imp
        .iter()
        .zip(recency_imp.iter())
        .map(|(&l, &r)| loss_weight * l + recency_weight * r)
        .collect()
}

/// Normalise a slice of importance values so they sum to 1.
///
/// Returns `ZeroImportanceWeights` if the sum is effectively zero.
pub fn normalize_importance(importance: &[f32]) -> Result<Vec<f32>, ViewImportanceError> {
    let sum: f32 = importance.iter().sum();
    if sum.abs() < 1e-12 {
        return Err(ViewImportanceError::ZeroImportanceWeights);
    }
    Ok(importance.iter().map(|v| v / sum).collect())
}

/// Compute numerically-stable softmax with a temperature parameter.
///
/// Returns `InvalidTemperature` if `temperature <= 0`.
pub fn importance_softmax(
    importance: &[f32],
    temperature: f32,
) -> Result<Vec<f32>, ViewImportanceError> {
    if temperature <= 0.0 {
        return Err(ViewImportanceError::InvalidTemperature { temp: temperature });
    }
    if importance.is_empty() {
        return Ok(Vec::new());
    }
    let max_val = importance.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = importance
        .iter()
        .map(|&v| ((v - max_val) / temperature).exp())
        .collect();
    let sum: f32 = exps.iter().sum();
    if sum < 1e-30 {
        // Fall back to uniform.
        let n = importance.len() as f32;
        return Ok(vec![1.0 / n; importance.len()]);
    }
    Ok(exps.iter().map(|e| e / sum).collect())
}

/// Sample one index from a probability weight vector using CDF inversion.
///
/// `weights` should be non-negative and sum to approximately 1, but this
/// function handles unnormalised weights as well by computing the CDF on
/// the fly.
pub fn sample_from_weights(
    weights: &[f32],
    rng_state: &mut u64,
) -> Result<usize, ViewImportanceError> {
    if weights.is_empty() {
        return Err(ViewImportanceError::NoViews);
    }
    let sum: f32 = weights.iter().sum();
    if sum < 1e-30 {
        return Err(ViewImportanceError::ZeroImportanceWeights);
    }
    let u = xorshift_f32(rng_state) * sum;
    let mut cumulative = 0.0_f32;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += w;
        if cumulative >= u {
            return Ok(i);
        }
    }
    // Floating-point edge: return last valid index.
    Ok(weights.len() - 1)
}

/// Compute the mean of a loss history slice.  Returns 0.0 for empty slices.
pub fn compute_mean_loss(history: &[f32]) -> f32 {
    if history.is_empty() {
        return 0.0;
    }
    history.iter().sum::<f32>() / history.len() as f32
}

/// Compute the population variance of a loss history slice.
///
/// Returns 0.0 when fewer than 2 values are present.
pub fn compute_loss_variance(history: &[f32]) -> f32 {
    if history.len() < 2 {
        return 0.0;
    }
    let mean = compute_mean_loss(history);
    let mean_sq: f32 = history.iter().map(|v| v * v).sum::<f32>() / history.len() as f32;
    (mean_sq - mean * mean).max(0.0)
}

/// Format a one-line summary of importance scores.
///
/// Output: `"Views[N]: top=view_{i}(w=0.45), bottom=view_{j}(w=0.01), mean_w=0.25"`
pub fn format_importance_summary(scores: &[ViewImportanceScore]) -> String {
    if scores.is_empty() {
        return "Views[0]: (no data)".to_string();
    }
    let n = scores.len();
    let top = scores
        .iter()
        .max_by(|a, b| {
            a.sample_weight
                .partial_cmp(&b.sample_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|s| (s.view_idx, s.sample_weight));
    let bottom = scores
        .iter()
        .min_by(|a, b| {
            a.sample_weight
                .partial_cmp(&b.sample_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|s| (s.view_idx, s.sample_weight));
    let mean_w = scores.iter().map(|s| s.sample_weight).sum::<f32>() / n as f32;

    match (top, bottom) {
        (Some((ti, tw)), Some((bi, bw))) => format!(
            "Views[{n}]: top=view_{ti}(w={tw:.2}), bottom=view_{bi}(w={bw:.2}), mean_w={mean_w:.2}"
        ),
        _ => format!("Views[{n}]: mean_w={mean_w:.2}"),
    }
}

/// Greedily select up to `budget` views by descending sample weight.
///
/// Returns a sorted list of view indices.
pub fn select_views_by_importance(scores: &[ViewImportanceScore], budget: usize) -> Vec<usize> {
    let mut sorted: Vec<&ViewImportanceScore> = scores.iter().collect();
    sorted.sort_by(|a, b| {
        b.sample_weight
            .partial_cmp(&a.sample_weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let take = budget.min(sorted.len());
    let mut indices: Vec<usize> = sorted[..take].iter().map(|s| s.view_idx).collect();
    indices.sort_unstable();
    indices
}

/// Shannon entropy of a weight distribution.
///
/// Measures how uniform the distribution is.  Maximum when uniform
/// (= log(n)), zero when all mass on a single view.
pub fn importance_entropy(weights: &[f32]) -> f32 {
    weights
        .iter()
        .map(|&w| {
            let v = w + 1e-10_f32;
            -w * v.ln()
        })
        .sum()
}

/// Compute (mean_importance, max_importance, min_importance) over a score slice.
pub fn view_importance_summary_stats(scores: &[ViewImportanceScore]) -> (f32, f32, f32) {
    if scores.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut sum = 0.0_f32;
    let mut max = f32::NEG_INFINITY;
    let mut min = f32::INFINITY;
    for s in scores {
        sum += s.importance;
        if s.importance > max {
            max = s.importance;
        }
        if s.importance < min {
            min = s.importance;
        }
    }
    let mean = sum / scores.len() as f32;
    (mean, max, min)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn default_sampler(n: usize) -> ViewImportanceSampler {
        let cfg = ViewImportanceConfig {
            n_views: n,
            ..ViewImportanceConfig::default()
        };
        ViewImportanceSampler::new(cfg).expect("valid config")
    }

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // ── ViewImportanceConfig::default ─────────────────────────────────────────

    #[test]
    fn test_config_default_n_views() {
        let cfg = ViewImportanceConfig::default();
        assert_eq!(cfg.n_views, 1);
    }

    #[test]
    fn test_config_default_history_window() {
        let cfg = ViewImportanceConfig::default();
        assert_eq!(cfg.history_window, 20);
    }

    #[test]
    fn test_config_default_strategy() {
        let cfg = ViewImportanceConfig::default();
        assert_eq!(cfg.strategy, ImportanceStrategy::ByLoss);
    }

    #[test]
    fn test_config_default_temperature() {
        let cfg = ViewImportanceConfig::default();
        assert!(approx_eq(cfg.temperature, 1.0, 1e-6));
    }

    #[test]
    fn test_config_default_loss_weight() {
        let cfg = ViewImportanceConfig::default();
        assert!(approx_eq(cfg.loss_weight, 0.7, 1e-6));
    }

    #[test]
    fn test_config_default_recency_weight() {
        let cfg = ViewImportanceConfig::default();
        assert!(approx_eq(cfg.recency_weight, 0.3, 1e-6));
    }

    #[test]
    fn test_config_default_ema_decay() {
        let cfg = ViewImportanceConfig::default();
        assert!(approx_eq(cfg.ema_decay, 0.9, 1e-6));
    }

    // ── ViewImportanceConfig::validate ────────────────────────────────────────

    #[test]
    fn test_config_validate_zero_temperature() {
        let cfg = ViewImportanceConfig {
            temperature: 0.0,
            ..ViewImportanceConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ViewImportanceError::InvalidTemperature { temp }) if temp == 0.0
        ));
    }

    #[test]
    fn test_config_validate_negative_temperature() {
        let cfg = ViewImportanceConfig {
            temperature: -1.0,
            ..ViewImportanceConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ViewImportanceError::InvalidTemperature { .. })
        ));
    }

    #[test]
    fn test_config_validate_zero_n_views() {
        let cfg = ViewImportanceConfig {
            n_views: 0,
            ..ViewImportanceConfig::default()
        };
        assert!(matches!(cfg.validate(), Err(ViewImportanceError::NoViews)));
    }

    #[test]
    fn test_config_validate_ok() {
        assert!(ViewImportanceConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_history_window() {
        let cfg = ViewImportanceConfig {
            history_window: 0,
            ..ViewImportanceConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ViewImportanceError::InvalidHistoryWindow { window: 0 })
        ));
    }

    #[test]
    fn test_config_validate_ema_decay_out_of_range() {
        for bad in [-0.1_f32, 0.0, 1.0, 1.1] {
            let cfg = ViewImportanceConfig {
                ema_decay: bad,
                ..ViewImportanceConfig::default()
            };
            assert!(
                matches!(
                    cfg.validate(),
                    Err(ViewImportanceError::InvalidEmaDecay { .. })
                ),
                "ema_decay={bad} should be rejected"
            );
        }
    }

    /// Regression: `validate` must implement the OPEN interval `(0, 1)` that
    /// `InvalidEmaDecay`'s message documents.  It previously used the
    /// half-open range `[0.0, 1.0)`, which accepted `ema_decay == 0.0` — a
    /// decay that disables smoothing entirely, since `update`'s
    /// `ema = d·ema + (1 − d)·raw` collapses to `ema = raw`.  NaN must be
    /// rejected as well: it is not inside any interval, and it would poison
    /// the EMA permanently on the first `update`.
    #[test]
    fn test_config_validate_ema_decay_open_interval_boundaries() {
        let with_decay = |d: f32| ViewImportanceConfig {
            ema_decay: d,
            ..ViewImportanceConfig::default()
        };

        for bad in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, 0.0, 1.0] {
            assert!(
                matches!(
                    with_decay(bad).validate(),
                    Err(ViewImportanceError::InvalidEmaDecay { .. })
                ),
                "ema_decay={bad} is outside (0, 1) and must be rejected"
            );
        }

        // Strictly interior values, including ones adjacent to the excluded
        // endpoints, stay accepted.
        for good in [f32::MIN_POSITIVE, 1e-6, 0.5, 0.9, 0.999_999] {
            assert!(
                with_decay(good).validate().is_ok(),
                "ema_decay={good} is inside (0, 1) and must be accepted"
            );
        }
    }

    /// A rejected `ema_decay` must also stop `ViewImportanceSampler::new`,
    /// which is the only way the invalid value could otherwise reach
    /// `update`'s smoothing step.
    #[test]
    fn test_sampler_new_rejects_zero_ema_decay() {
        let cfg = ViewImportanceConfig {
            n_views: 4,
            ema_decay: 0.0,
            ..ViewImportanceConfig::default()
        };
        assert!(matches!(
            ViewImportanceSampler::new(cfg),
            Err(ViewImportanceError::InvalidEmaDecay { decay }) if decay == 0.0
        ));
    }

    // ── ViewImportanceSampler::new ────────────────────────────────────────────

    #[test]
    fn test_sampler_new_initialises_uniform_ema() {
        let s = default_sampler(4);
        for &v in &s.ema_importance {
            assert!(approx_eq(v, 0.25, 1e-6));
        }
    }

    #[test]
    fn test_sampler_new_empty_histories() {
        let s = default_sampler(3);
        assert!(s.loss_history.iter().all(|h| h.is_empty()));
    }

    #[test]
    fn test_sampler_new_zero_sample_counts() {
        let s = default_sampler(5);
        assert!(s.n_samples.iter().all(|&c| c == 0));
    }

    // ── ViewImportanceSampler::update ─────────────────────────────────────────

    #[test]
    fn test_update_grows_history() {
        let mut s = default_sampler(3);
        s.update(0, 0.5, 1).expect("valid");
        s.update(0, 0.6, 2).expect("valid");
        assert_eq!(s.loss_history[0].len(), 2);
    }

    #[test]
    fn test_new_rejects_zero_history_window() {
        // Regression: `ViewImportanceSampler::new(ViewImportanceConfig {
        // n_views: 1, history_window: 0, ..Default::default() })` used to
        // succeed and then panic on the very first `update()` call
        // (`Vec::remove(0)` on an empty history). `new()` now rejects the
        // config outright instead of deferring the failure.
        let cfg = ViewImportanceConfig {
            n_views: 1,
            history_window: 0,
            ..ViewImportanceConfig::default()
        };
        assert!(matches!(
            ViewImportanceSampler::new(cfg),
            Err(ViewImportanceError::InvalidHistoryWindow { window: 0 })
        ));
    }

    #[test]
    fn test_update_does_not_panic_if_history_window_mutated_to_zero() {
        // `config` is a `pub` field, so `validate()` at construction time
        // cannot fully protect `update()` against `history_window == 0` —
        // this exercises the defence-in-depth `while` loop directly.
        let mut s = default_sampler(1);
        s.config.history_window = 0;
        s.update(0, 0.5, 1).expect("update must not panic");
        s.update(0, 0.6, 2).expect("update must not panic");
        // Degrades to keeping only the single most recent sample.
        assert_eq!(s.loss_history[0].len(), 1);
        assert!(approx_eq(s.loss_history[0][0], 0.6, 1e-6));
    }

    #[test]
    fn test_update_evicts_oldest_at_capacity() {
        let mut s = ViewImportanceSampler::new(ViewImportanceConfig {
            n_views: 2,
            history_window: 3,
            ..ViewImportanceConfig::default()
        })
        .expect("valid");
        s.update(0, 0.1, 1).expect("ok");
        s.update(0, 0.2, 2).expect("ok");
        s.update(0, 0.3, 3).expect("ok");
        s.update(0, 0.4, 4).expect("ok"); // evicts 0.1
        assert_eq!(s.loss_history[0].len(), 3);
        assert!(approx_eq(s.loss_history[0][0], 0.2, 1e-6));
    }

    #[test]
    fn test_update_increments_n_samples() {
        let mut s = default_sampler(2);
        s.update(1, 0.5, 1).expect("ok");
        s.update(1, 0.7, 2).expect("ok");
        assert_eq!(s.n_samples[1], 2);
    }

    #[test]
    fn test_update_out_of_range_error() {
        let mut s = default_sampler(2);
        assert!(matches!(
            s.update(99, 0.5, 1),
            Err(ViewImportanceError::ViewIndexOutOfRange { idx: 99, .. })
        ));
    }

    #[test]
    fn test_update_advances_current_step() {
        let mut s = default_sampler(2);
        s.update(0, 0.5, 10).expect("ok");
        assert_eq!(s.current_step, 10);
    }

    // ── ViewImportanceSampler::compute_importance ─────────────────────────────

    #[test]
    fn test_compute_importance_uniform_equal_weights() {
        let cfg = ViewImportanceConfig {
            n_views: 4,
            strategy: ImportanceStrategy::Uniform,
            ..ViewImportanceConfig::default()
        };
        let s = ViewImportanceSampler::new(cfg).expect("ok");
        let scores = s.compute_importance().expect("ok");
        for sc in &scores {
            assert!(approx_eq(sc.sample_weight, 0.25, 1e-5));
        }
    }

    #[test]
    fn test_compute_importance_by_loss_high_loss_higher_weight() {
        let cfg = ViewImportanceConfig {
            n_views: 2,
            strategy: ImportanceStrategy::ByLoss,
            temperature: 0.5,
            ..ViewImportanceConfig::default()
        };
        let mut s = ViewImportanceSampler::new(cfg).expect("ok");
        s.update(0, 0.9, 1).expect("ok"); // high loss
        s.update(1, 0.1, 1).expect("ok"); // low loss
        let scores = s.compute_importance().expect("ok");
        assert!(scores[0].sample_weight > scores[1].sample_weight);
    }

    #[test]
    fn test_compute_importance_returns_sorted_by_view_idx() {
        let s = default_sampler(5);
        let scores = s.compute_importance().expect("ok");
        for (i, sc) in scores.iter().enumerate() {
            assert_eq!(sc.view_idx, i);
        }
    }

    #[test]
    fn test_compute_importance_weights_sum_to_one() {
        let s = default_sampler(6);
        let scores = s.compute_importance().expect("ok");
        let total: f32 = scores.iter().map(|sc| sc.sample_weight).sum();
        assert!(approx_eq(total, 1.0, 1e-5));
    }

    // ── ViewImportanceSampler::sample_view ────────────────────────────────────

    #[test]
    fn test_sample_view_valid_index() {
        let mut s = default_sampler(4);
        let mut rng: u64 = 0xdeadbeef12345678;
        let idx = s.sample_view(&mut rng).expect("ok");
        assert!(idx < 4);
    }

    #[test]
    fn test_sample_view_multiple_valid() {
        let mut s = default_sampler(8);
        s.update(3, 1.0, 1).expect("ok");
        let mut rng: u64 = 0xabcdef1234567890;
        for _ in 0..20 {
            let idx = s.sample_view(&mut rng).expect("ok");
            assert!(idx < 8);
        }
    }

    // ── ViewImportanceSampler::top_k_views ────────────────────────────────────

    #[test]
    fn test_top_k_views_length() {
        let s = default_sampler(6);
        let top = s.top_k_views(3).expect("ok");
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_top_k_views_ordering_after_updates() {
        let cfg = ViewImportanceConfig {
            n_views: 3,
            strategy: ImportanceStrategy::ByLoss,
            temperature: 0.1,
            ..ViewImportanceConfig::default()
        };
        let mut s = ViewImportanceSampler::new(cfg).expect("ok");
        s.update(0, 0.9, 1).expect("ok");
        s.update(1, 0.3, 1).expect("ok");
        s.update(2, 0.6, 1).expect("ok");
        let top = s.top_k_views(1).expect("ok");
        assert_eq!(top[0], 0); // highest loss = most important
    }

    #[test]
    fn test_top_k_views_k_exceeds_n() {
        let s = default_sampler(3);
        let top = s.top_k_views(100).expect("ok");
        assert_eq!(top.len(), 3);
    }

    // ── ViewImportanceSampler::reset ──────────────────────────────────────────

    #[test]
    fn test_reset_clears_history() {
        let mut s = default_sampler(3);
        s.update(0, 0.5, 1).expect("ok");
        s.reset();
        assert!(s.loss_history[0].is_empty());
    }

    #[test]
    fn test_reset_clears_step_counter() {
        let mut s = default_sampler(2);
        s.update(0, 0.5, 42).expect("ok");
        s.reset();
        assert_eq!(s.current_step, 0);
    }

    #[test]
    fn test_reset_restores_uniform_ema() {
        let mut s = default_sampler(4);
        s.update(0, 0.9, 1).expect("ok");
        s.reset();
        for &v in &s.ema_importance {
            assert!(approx_eq(v, 0.25, 1e-6));
        }
    }

    // ── compute_loss_importance ───────────────────────────────────────────────

    #[test]
    fn test_compute_loss_importance_empty_history_is_one() {
        let history: Vec<Vec<f32>> = vec![vec![], vec![0.5]];
        let imp = compute_loss_importance(&history);
        assert!(approx_eq(imp[0], 1.0, 1e-6));
    }

    #[test]
    fn test_compute_loss_importance_known_values() {
        let history = vec![vec![0.8_f32], vec![0.2_f32]];
        let imp = compute_loss_importance(&history);
        assert!(approx_eq(imp[0], 0.8, 1e-6));
        assert!(approx_eq(imp[1], 0.2, 1e-6));
    }

    // ── compute_variance_importance ───────────────────────────────────────────

    #[test]
    fn test_compute_variance_importance_constant_history() {
        let history = vec![vec![0.5_f32, 0.5, 0.5]];
        let imp = compute_variance_importance(&history);
        assert!(approx_eq(imp[0], 0.0, 1e-6));
    }

    #[test]
    fn test_compute_variance_importance_single_sample_is_one() {
        let history: Vec<Vec<f32>> = vec![vec![0.7]];
        let imp = compute_variance_importance(&history);
        assert!(approx_eq(imp[0], 1.0, 1e-6));
    }

    #[test]
    fn test_compute_variance_importance_known_variance() {
        // values [0, 1]: mean = 0.5, var = 0.25
        let history = vec![vec![0.0_f32, 1.0]];
        let imp = compute_variance_importance(&history);
        assert!(approx_eq(imp[0], 0.25, 1e-5));
    }

    // ── compute_recency_importance ────────────────────────────────────────────

    #[test]
    fn test_compute_recency_importance_old_views_higher() {
        let last_sampled = vec![5usize, 10usize];
        let current_step = 15;
        let imp = compute_recency_importance(&last_sampled, current_step);
        assert!(imp[0] > imp[1]); // view 0 was sampled earlier
    }

    #[test]
    fn test_compute_recency_importance_never_sampled() {
        let last_sampled = vec![0usize, 5usize];
        let current_step = 10;
        let imp = compute_recency_importance(&last_sampled, current_step);
        assert!(approx_eq(imp[0], 11.0, 1e-5)); // current_step + 1
    }

    // ── combine_importance ────────────────────────────────────────────────────

    #[test]
    fn test_combine_importance_weighted_sum() {
        let loss_imp = vec![1.0_f32, 0.0];
        let rec_imp = vec![0.0_f32, 1.0];
        let combined = combine_importance(&loss_imp, &rec_imp, 0.6, 0.4);
        assert!(approx_eq(combined[0], 0.6, 1e-6));
        assert!(approx_eq(combined[1], 0.4, 1e-6));
    }

    #[test]
    fn test_combine_importance_length() {
        let a = vec![0.5_f32; 5];
        let b = vec![0.5_f32; 5];
        let c = combine_importance(&a, &b, 0.5, 0.5);
        assert_eq!(c.len(), 5);
    }

    // ── normalize_importance ──────────────────────────────────────────────────

    #[test]
    fn test_normalize_importance_sums_to_one() {
        let imp = vec![1.0_f32, 2.0, 3.0];
        let norm = normalize_importance(&imp).expect("ok");
        let s: f32 = norm.iter().sum();
        assert!(approx_eq(s, 1.0, 1e-6));
    }

    #[test]
    fn test_normalize_importance_zero_error() {
        let imp = vec![0.0_f32, 0.0];
        assert!(matches!(
            normalize_importance(&imp),
            Err(ViewImportanceError::ZeroImportanceWeights)
        ));
    }

    #[test]
    fn test_normalize_importance_correct_values() {
        let imp = vec![1.0_f32, 1.0];
        let norm = normalize_importance(&imp).expect("ok");
        assert!(approx_eq(norm[0], 0.5, 1e-6));
        assert!(approx_eq(norm[1], 0.5, 1e-6));
    }

    // ── importance_softmax ────────────────────────────────────────────────────

    #[test]
    fn test_importance_softmax_sums_to_one() {
        let imp = vec![0.2_f32, 0.5, 0.8, 0.1];
        let sw = importance_softmax(&imp, 1.0).expect("ok");
        let s: f32 = sw.iter().sum();
        assert!(approx_eq(s, 1.0, 1e-5));
    }

    #[test]
    fn test_importance_softmax_temperature_sharpening() {
        let imp = vec![0.2_f32, 0.8];
        let low_t = importance_softmax(&imp, 0.1).expect("ok");
        let high_t = importance_softmax(&imp, 10.0).expect("ok");
        // Low temperature → peakier distribution
        assert!(low_t[1] > high_t[1]);
    }

    #[test]
    fn test_importance_softmax_invalid_temperature() {
        let imp = vec![0.5_f32];
        assert!(matches!(
            importance_softmax(&imp, -1.0),
            Err(ViewImportanceError::InvalidTemperature { .. })
        ));
    }

    #[test]
    fn test_importance_softmax_empty_input() {
        let sw = importance_softmax(&[], 1.0).expect("ok");
        assert!(sw.is_empty());
    }

    // ── sample_from_weights ───────────────────────────────────────────────────

    #[test]
    fn test_sample_from_weights_valid_index() {
        let w = vec![0.25_f32, 0.25, 0.25, 0.25];
        let mut rng: u64 = 0x123456789abcdef0;
        let idx = sample_from_weights(&w, &mut rng).expect("ok");
        assert!(idx < 4);
    }

    #[test]
    fn test_sample_from_weights_zero_weights_error() {
        let w = vec![0.0_f32, 0.0];
        let mut rng: u64 = 1;
        assert!(matches!(
            sample_from_weights(&w, &mut rng),
            Err(ViewImportanceError::ZeroImportanceWeights)
        ));
    }

    #[test]
    fn test_sample_from_weights_empty_error() {
        let w: Vec<f32> = vec![];
        let mut rng: u64 = 1;
        assert!(matches!(
            sample_from_weights(&w, &mut rng),
            Err(ViewImportanceError::NoViews)
        ));
    }

    #[test]
    fn test_sample_from_weights_concentrated_mass() {
        // All mass on index 2 — should always return 2.
        let w = vec![0.0_f32, 0.0, 1.0, 0.0];
        let mut rng: u64 = 0x1111222233334444;
        for _ in 0..10 {
            let idx = sample_from_weights(&w, &mut rng).expect("ok");
            assert_eq!(idx, 2);
        }
    }

    // ── compute_mean_loss ─────────────────────────────────────────────────────

    #[test]
    fn test_compute_mean_loss_empty() {
        assert!(approx_eq(compute_mean_loss(&[]), 0.0, 1e-9));
    }

    #[test]
    fn test_compute_mean_loss_known() {
        assert!(approx_eq(compute_mean_loss(&[1.0, 2.0, 3.0]), 2.0, 1e-6));
    }

    // ── compute_loss_variance ─────────────────────────────────────────────────

    #[test]
    fn test_compute_loss_variance_constant() {
        assert!(approx_eq(
            compute_loss_variance(&[0.5, 0.5, 0.5]),
            0.0,
            1e-6
        ));
    }

    #[test]
    fn test_compute_loss_variance_empty() {
        assert!(approx_eq(compute_loss_variance(&[]), 0.0, 1e-9));
    }

    #[test]
    fn test_compute_loss_variance_single() {
        assert!(approx_eq(compute_loss_variance(&[0.7]), 0.0, 1e-9));
    }

    #[test]
    fn test_compute_loss_variance_known() {
        // [0, 1, 2]: mean=1, var = (0+1+4)/3 - 1 = 5/3 - 1 = 2/3
        let v = compute_loss_variance(&[0.0, 1.0, 2.0]);
        assert!(approx_eq(v, 2.0 / 3.0, 1e-5));
    }

    // ── format_importance_summary ─────────────────────────────────────────────

    #[test]
    fn test_format_importance_summary_non_empty() {
        let s = default_sampler(3);
        let scores = s.compute_importance().expect("ok");
        let summary = format_importance_summary(&scores);
        assert!(summary.contains("Views[3]"));
    }

    #[test]
    fn test_format_importance_summary_empty() {
        let summary = format_importance_summary(&[]);
        assert!(summary.contains("Views[0]"));
    }

    // ── select_views_by_importance ────────────────────────────────────────────

    #[test]
    fn test_select_views_by_importance_correct_count() {
        let s = default_sampler(6);
        let scores = s.compute_importance().expect("ok");
        let selected = select_views_by_importance(&scores, 3);
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_select_views_by_importance_sorted() {
        let s = default_sampler(5);
        let scores = s.compute_importance().expect("ok");
        let selected = select_views_by_importance(&scores, 5);
        for w in selected.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn test_select_views_by_importance_budget_exceeds_n() {
        let s = default_sampler(3);
        let scores = s.compute_importance().expect("ok");
        let selected = select_views_by_importance(&scores, 100);
        assert_eq!(selected.len(), 3);
    }

    // ── importance_entropy ────────────────────────────────────────────────────

    #[test]
    fn test_importance_entropy_uniform_maximum() {
        // Uniform: entropy = log(4)
        let n = 4;
        let w = vec![0.25_f32; n];
        let h = importance_entropy(&w);
        let expected = (n as f32).ln();
        // Allow some tolerance for the 1e-10 epsilon in the formula
        assert!((h - expected).abs() < 0.001);
    }

    #[test]
    fn test_importance_entropy_concentrated_near_zero() {
        // All mass on one view — entropy ≈ 0.
        let w = vec![0.0_f32, 0.0, 1.0, 0.0];
        let h = importance_entropy(&w);
        assert!(h < 0.01);
    }

    // ── view_importance_summary_stats ─────────────────────────────────────────

    #[test]
    fn test_view_importance_summary_stats_known() {
        let scores = vec![
            ViewImportanceScore {
                view_idx: 0,
                importance: 0.2,
                sample_weight: 0.3,
                mean_loss: 0.1,
                loss_variance: 0.0,
                n_samples: 1,
                last_sampled_step: 1,
            },
            ViewImportanceScore {
                view_idx: 1,
                importance: 0.8,
                sample_weight: 0.7,
                mean_loss: 0.9,
                loss_variance: 0.0,
                n_samples: 1,
                last_sampled_step: 1,
            },
        ];
        let (mean, max, min) = view_importance_summary_stats(&scores);
        assert!(approx_eq(mean, 0.5, 1e-6));
        assert!(approx_eq(max, 0.8, 1e-6));
        assert!(approx_eq(min, 0.2, 1e-6));
    }

    #[test]
    fn test_view_importance_summary_stats_empty() {
        let (mean, max, min) = view_importance_summary_stats(&[]);
        assert!(approx_eq(mean, 0.0, 1e-9));
        assert!(approx_eq(max, 0.0, 1e-9));
        assert!(approx_eq(min, 0.0, 1e-9));
    }

    // ── ViewImportanceError variants ──────────────────────────────────────────

    #[test]
    fn test_error_no_views_message() {
        let e = ViewImportanceError::NoViews;
        let s = e.to_string();
        assert!(s.contains("No views registered"));
    }

    #[test]
    fn test_error_view_index_out_of_range_message() {
        let e = ViewImportanceError::ViewIndexOutOfRange { idx: 5, n_views: 3 };
        let s = e.to_string();
        assert!(s.contains("5"));
        assert!(s.contains("3"));
    }

    #[test]
    fn test_error_zero_importance_weights_message() {
        let e = ViewImportanceError::ZeroImportanceWeights;
        assert!(e.to_string().contains("zero"));
    }

    // ── ImportanceStrategy variants ───────────────────────────────────────────

    #[test]
    fn test_strategy_eq() {
        assert_eq!(ImportanceStrategy::Uniform, ImportanceStrategy::Uniform);
        assert_ne!(ImportanceStrategy::ByLoss, ImportanceStrategy::ByRecency);
    }

    #[test]
    fn test_strategy_debug() {
        let s = format!("{:?}", ImportanceStrategy::Combined);
        assert!(s.contains("Combined"));
    }

    // ── ViewImportanceScore fields ────────────────────────────────────────────

    #[test]
    fn test_view_importance_score_fields_accessible() {
        let sc = ViewImportanceScore {
            view_idx: 7,
            importance: 0.42,
            sample_weight: 0.13,
            mean_loss: 0.55,
            loss_variance: 0.01,
            n_samples: 3,
            last_sampled_step: 99,
        };
        assert_eq!(sc.view_idx, 7);
        assert!(approx_eq(sc.importance, 0.42, 1e-6));
        assert_eq!(sc.n_samples, 3);
        assert_eq!(sc.last_sampled_step, 99);
    }

    // ── ByLossVariance and ByRecency strategies ───────────────────────────────

    #[test]
    fn test_compute_importance_by_loss_variance_strategy() {
        let cfg = ViewImportanceConfig {
            n_views: 2,
            strategy: ImportanceStrategy::ByLossVariance,
            ..ViewImportanceConfig::default()
        };
        let mut s = ViewImportanceSampler::new(cfg).expect("ok");
        // View 0: constant loss (zero variance)
        s.update(0, 0.5, 1).expect("ok");
        s.update(0, 0.5, 2).expect("ok");
        s.update(0, 0.5, 3).expect("ok");
        // View 1: varying loss (nonzero variance)
        s.update(1, 0.1, 1).expect("ok");
        s.update(1, 0.9, 2).expect("ok");
        let scores = s.compute_importance().expect("ok");
        assert!(scores[1].importance >= scores[0].importance);
    }

    #[test]
    fn test_compute_importance_by_recency_strategy() {
        let cfg = ViewImportanceConfig {
            n_views: 2,
            strategy: ImportanceStrategy::ByRecency,
            ..ViewImportanceConfig::default()
        };
        let mut s = ViewImportanceSampler::new(cfg).expect("ok");
        s.update(1, 0.5, 1).expect("ok"); // view 1 sampled recently
        s.update(0, 0.5, 10).expect("ok"); // view 0 sampled even more recently
                                           // Advance to step 20 by updating again
        s.update(1, 0.5, 20).expect("ok"); // view 1 sampled at step 20 now
        let scores = s.compute_importance().expect("ok");
        // view 0 last sampled at step 10, view 1 at step 20 → view 0 is older
        assert!(scores[0].importance >= scores[1].importance);
    }

    #[test]
    fn test_compute_importance_combined_strategy() {
        let cfg = ViewImportanceConfig {
            n_views: 2,
            strategy: ImportanceStrategy::Combined,
            loss_weight: 0.7,
            recency_weight: 0.3,
            ..ViewImportanceConfig::default()
        };
        let mut s = ViewImportanceSampler::new(cfg).expect("ok");
        s.update(0, 0.9, 1).expect("ok");
        s.update(1, 0.1, 1).expect("ok");
        let scores = s.compute_importance().expect("ok");
        let total: f32 = scores.iter().map(|sc| sc.sample_weight).sum();
        assert!(approx_eq(total, 1.0, 1e-5));
    }
}
