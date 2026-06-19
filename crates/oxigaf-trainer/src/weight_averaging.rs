//! Stochastic Weight Averaging (SWA), Model Soup, and Polyak Averaging.
//!
//! These techniques improve model quality without additional training by
//! combining weights from multiple checkpoints or maintaining running averages.
//!
//! # Quick start
//!
//! ```rust
//! use oxigaf_trainer::weight_averaging::{
//!     SwaConfig, StochasticWeightAverager, ModelSoup, PolyakAverager,
//!     count_params, weights_l2_distance,
//! };
//!
//! // Stochastic Weight Averaging
//! let config = SwaConfig::default();
//! let mut swa = StochasticWeightAverager::new(config).unwrap();
//!
//! // Model Soup
//! let soup = ModelSoup::new();
//!
//! // Polyak averaging
//! let polyak = PolyakAverager::new(0.999).unwrap();
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by weight averaging operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum WeightAveragingError {
    /// No models have been added yet.
    #[error("No weights available: no models have been added")]
    NoWeights,

    /// Weight vector length mismatch between models.
    #[error("Length mismatch: expected {expected} parameters, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    /// Invalid configuration parameter.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Negative mixing weight provided.
    #[error("Invalid mixing weight: {0}")]
    InvalidWeight(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelWeights type and free functions
// ─────────────────────────────────────────────────────────────────────────────

/// Flat representation of model weights for a single parameter group.
///
/// Each inner `Vec<f32>` corresponds to one parameter group (e.g., positions,
/// rotations, scales). The outer `Vec` holds all groups.
pub type ModelWeights = Vec<Vec<f32>>;

/// Compute total number of parameters across all groups.
pub fn count_params(weights: &ModelWeights) -> usize {
    weights.iter().map(|g| g.len()).sum()
}

/// Compute L2 distance between two `ModelWeights`.
///
/// Returns `sqrt(Σ (a_i - b_i)²)` summed across all elements of all groups.
///
/// # Errors
///
/// Returns [`WeightAveragingError::LengthMismatch`] if the group counts or
/// per-group lengths differ.
pub fn weights_l2_distance(
    a: &ModelWeights,
    b: &ModelWeights,
) -> Result<f32, WeightAveragingError> {
    let total_a = count_params(a);
    let total_b = count_params(b);
    if a.len() != b.len() || total_a != total_b {
        return Err(WeightAveragingError::LengthMismatch {
            expected: total_a,
            actual: total_b,
        });
    }
    // Verify per-group lengths.
    for (ga, gb) in a.iter().zip(b.iter()) {
        if ga.len() != gb.len() {
            return Err(WeightAveragingError::LengthMismatch {
                expected: ga.len(),
                actual: gb.len(),
            });
        }
    }

    let sum_sq: f32 = a
        .iter()
        .zip(b.iter())
        .flat_map(|(ga, gb)| ga.iter().zip(gb.iter()).map(|(x, y)| (x - y) * (x - y)))
        .sum();
    Ok(sum_sq.sqrt())
}

/// Compute cosine similarity between two `ModelWeights` (treating all groups
/// as one flattened vector).
///
/// Returns `dot(a_flat, b_flat) / (|a| * |b| + 1e-8)`.
///
/// # Errors
///
/// Returns [`WeightAveragingError::LengthMismatch`] if the shapes differ.
pub fn weights_cosine_similarity(
    a: &ModelWeights,
    b: &ModelWeights,
) -> Result<f32, WeightAveragingError> {
    let total_a = count_params(a);
    let total_b = count_params(b);
    if a.len() != b.len() || total_a != total_b {
        return Err(WeightAveragingError::LengthMismatch {
            expected: total_a,
            actual: total_b,
        });
    }
    for (ga, gb) in a.iter().zip(b.iter()) {
        if ga.len() != gb.len() {
            return Err(WeightAveragingError::LengthMismatch {
                expected: ga.len(),
                actual: gb.len(),
            });
        }
    }

    let dot: f32 = a
        .iter()
        .zip(b.iter())
        .flat_map(|(ga, gb)| ga.iter().zip(gb.iter()).map(|(x, y)| x * y))
        .sum();
    let norm_a: f32 = a
        .iter()
        .flat_map(|g| g.iter().map(|x| x * x))
        .sum::<f32>()
        .sqrt();
    let norm_b: f32 = b
        .iter()
        .flat_map(|g| g.iter().map(|x| x * x))
        .sum::<f32>()
        .sqrt();
    Ok(dot / (norm_a * norm_b + 1e-8))
}

// ─────────────────────────────────────────────────────────────────────────────
// SwaConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`StochasticWeightAverager`].
#[derive(Debug, Clone)]
pub struct SwaConfig {
    /// Training step to start SWA (after initial training stabilizes).
    pub start_step: usize,
    /// How often to update the SWA average.
    pub update_every: usize,
    /// Learning rate for SWA phase (typically lower than main LR).
    pub swa_lr: f64,
    /// Whether to use exponential moving average instead of uniform SWA.
    pub use_ema: bool,
    /// EMA decay (0.999 = slow, 0.9 = fast). Only used if `use_ema = true`.
    pub ema_decay: f32,
}

impl Default for SwaConfig {
    fn default() -> Self {
        Self {
            start_step: 50_000,
            update_every: 100,
            swa_lr: 1e-5,
            use_ema: false,
            ema_decay: 0.999,
        }
    }
}

impl SwaConfig {
    /// Validate configuration parameters.
    ///
    /// # Errors
    ///
    /// Returns [`WeightAveragingError::InvalidConfig`] if `update_every == 0`
    /// or `ema_decay` is not in the open interval `(0, 1)`.
    pub fn validate(&self) -> Result<(), WeightAveragingError> {
        if self.update_every == 0 {
            return Err(WeightAveragingError::InvalidConfig(
                "update_every must be >= 1".to_string(),
            ));
        }
        if self.ema_decay <= 0.0 || self.ema_decay >= 1.0 {
            return Err(WeightAveragingError::InvalidConfig(format!(
                "ema_decay must be in (0, 1), got {}",
                self.ema_decay
            )));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StochasticWeightAverager
// ─────────────────────────────────────────────────────────────────────────────

/// Maintains a running average of model weights during training.
///
/// Two averaging modes are supported:
///
/// - **Uniform SWA**: Equal-weight running mean of all snapshots.
///   `swa = swa * n/(n+1) + weights * 1/(n+1)`
///
/// - **EMA mode**: Exponential moving average with configurable decay.
///   `swa = swa * decay + weights * (1 - decay)`
#[derive(Debug, Clone)]
pub struct StochasticWeightAverager {
    /// Configuration controlling SWA behaviour.
    pub config: SwaConfig,
    /// Current SWA / EMA average weights. `None` before the first update.
    swa_weights: Option<ModelWeights>,
    /// Number of snapshots included in the average.
    pub num_snapshots: usize,
    /// Training steps seen so far.
    pub current_step: usize,
}

impl StochasticWeightAverager {
    /// Create a new averager, validating the config.
    ///
    /// # Errors
    ///
    /// Returns [`WeightAveragingError::InvalidConfig`] if the config fails
    /// validation.
    pub fn new(config: SwaConfig) -> Result<Self, WeightAveragingError> {
        config.validate()?;
        Ok(Self {
            config,
            swa_weights: None,
            num_snapshots: 0,
            current_step: 0,
        })
    }

    /// Check if SWA should be active at the current step.
    ///
    /// Returns `true` once `current_step >= start_step`.
    pub fn is_active(&self) -> bool {
        self.current_step >= self.config.start_step
    }

    /// Check if this step should trigger an SWA update.
    ///
    /// Returns `true` when active and the offset from `start_step` is a
    /// multiple of `update_every`.
    pub fn should_update(&self) -> bool {
        if !self.is_active() {
            return false;
        }
        let offset = self.current_step - self.config.start_step;
        offset.is_multiple_of(self.config.update_every)
    }

    /// Update the SWA / EMA running average with new model weights.
    ///
    /// - First call: initialises the running average to a clone of `weights`.
    /// - Subsequent calls (uniform SWA): `swa = swa * n/(n+1) + weights * 1/(n+1)`.
    /// - Subsequent calls (EMA): `swa = swa * decay + weights * (1 - decay)`.
    ///
    /// # Errors
    ///
    /// Returns [`WeightAveragingError::LengthMismatch`] if subsequent
    /// `weights` have a different shape than the first.
    pub fn update(&mut self, weights: &ModelWeights) -> Result<(), WeightAveragingError> {
        match &mut self.swa_weights {
            None => {
                // First update: copy the weights directly.
                self.swa_weights = Some(weights.clone());
                self.num_snapshots = 1;
            }
            Some(swa) => {
                // Validate dimensions.
                let expected = count_params(swa);
                let actual = count_params(weights);
                if swa.len() != weights.len() || expected != actual {
                    return Err(WeightAveragingError::LengthMismatch { expected, actual });
                }
                for (sg, wg) in swa.iter().zip(weights.iter()) {
                    if sg.len() != wg.len() {
                        return Err(WeightAveragingError::LengthMismatch {
                            expected: sg.len(),
                            actual: wg.len(),
                        });
                    }
                }

                if self.config.use_ema {
                    let d = self.config.ema_decay;
                    let one_minus_d = 1.0 - d;
                    for (sg, wg) in swa.iter_mut().zip(weights.iter()) {
                        for (s, w) in sg.iter_mut().zip(wg.iter()) {
                            *s = d * *s + one_minus_d * w;
                        }
                    }
                } else {
                    // Uniform SWA: swa = swa * n/(n+1) + weights * 1/(n+1)
                    let n = self.num_snapshots as f32;
                    let coeff_swa = n / (n + 1.0);
                    let coeff_new = 1.0 / (n + 1.0);
                    for (sg, wg) in swa.iter_mut().zip(weights.iter()) {
                        for (s, w) in sg.iter_mut().zip(wg.iter()) {
                            *s = coeff_swa * *s + coeff_new * w;
                        }
                    }
                }
                self.num_snapshots += 1;
            }
        }
        Ok(())
    }

    /// Get read-only access to the current SWA weights.
    pub fn swa_weights(&self) -> Option<&ModelWeights> {
        self.swa_weights.as_ref()
    }

    /// Advance the internal step counter by one.
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Linearly interpolate between `current` weights and the SWA weights.
    ///
    /// `alpha = 0.0` → returns `current`; `alpha = 1.0` → returns SWA.
    ///
    /// # Errors
    ///
    /// - [`WeightAveragingError::NoWeights`] if no SWA weights are available.
    /// - [`WeightAveragingError::LengthMismatch`] if shapes differ.
    pub fn interpolate(
        &self,
        current: &ModelWeights,
        alpha: f32,
    ) -> Result<ModelWeights, WeightAveragingError> {
        let swa = self
            .swa_weights
            .as_ref()
            .ok_or(WeightAveragingError::NoWeights)?;
        let expected = count_params(swa);
        let actual = count_params(current);
        if swa.len() != current.len() || expected != actual {
            return Err(WeightAveragingError::LengthMismatch { expected, actual });
        }
        for (sg, cg) in swa.iter().zip(current.iter()) {
            if sg.len() != cg.len() {
                return Err(WeightAveragingError::LengthMismatch {
                    expected: sg.len(),
                    actual: cg.len(),
                });
            }
        }

        let one_minus_alpha = 1.0 - alpha;
        let result: ModelWeights = current
            .iter()
            .zip(swa.iter())
            .map(|(cg, sg)| {
                cg.iter()
                    .zip(sg.iter())
                    .map(|(c, s)| one_minus_alpha * c + alpha * s)
                    .collect()
            })
            .collect();
        Ok(result)
    }

    /// Reset the running average, clearing accumulated state.
    ///
    /// The config is preserved; `num_snapshots` and `current_step` are zeroed.
    pub fn reset(&mut self) {
        self.swa_weights = None;
        self.num_snapshots = 0;
        self.current_step = 0;
    }

    /// Format a human-readable status string for logging.
    pub fn format_status(&self) -> String {
        if self.swa_weights.is_none() {
            return format!(
                "SWA: inactive (step={}, start={})",
                self.current_step, self.config.start_step
            );
        }
        format!(
            "SWA: active step={} snapshots={} mode={}",
            self.current_step,
            self.num_snapshots,
            if self.config.use_ema {
                "EMA"
            } else {
                "uniform"
            }
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelSoup
// ─────────────────────────────────────────────────────────────────────────────

/// Average multiple independently-trained model checkpoints.
///
/// Supports uniform blending, weighted blending, and the greedy soup algorithm.
#[derive(Debug, Clone)]
pub struct ModelSoup {
    /// Stored model weight snapshots.
    pub models: Vec<ModelWeights>,
    /// Per-model mixing weights (not necessarily normalised).
    pub weights: Vec<f32>,
}

impl ModelSoup {
    /// Create an empty soup.
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Add a model with equal weight (weight = 1.0).
    ///
    /// # Errors
    ///
    /// Returns [`WeightAveragingError::LengthMismatch`] if the model's shape
    /// differs from the first model.
    pub fn add(&mut self, weights: ModelWeights) -> Result<(), WeightAveragingError> {
        self.add_weighted(weights, 1.0)
    }

    /// Add a model with a specific mixing weight.
    ///
    /// # Errors
    ///
    /// - [`WeightAveragingError::InvalidWeight`] if `weight < 0.0`.
    /// - [`WeightAveragingError::LengthMismatch`] if the model shape differs.
    pub fn add_weighted(
        &mut self,
        model: ModelWeights,
        weight: f32,
    ) -> Result<(), WeightAveragingError> {
        if weight < 0.0 {
            return Err(WeightAveragingError::InvalidWeight(format!(
                "mixing weight must be >= 0.0, got {weight}"
            )));
        }
        if let Some(first) = self.models.first() {
            let expected = count_params(first);
            let actual = count_params(&model);
            if first.len() != model.len() || expected != actual {
                return Err(WeightAveragingError::LengthMismatch { expected, actual });
            }
            for (fg, mg) in first.iter().zip(model.iter()) {
                if fg.len() != mg.len() {
                    return Err(WeightAveragingError::LengthMismatch {
                        expected: fg.len(),
                        actual: mg.len(),
                    });
                }
            }
        }
        self.models.push(model);
        self.weights.push(weight);
        Ok(())
    }

    /// Compute the soup as a normalised weighted average of all stored models.
    ///
    /// # Errors
    ///
    /// Returns [`WeightAveragingError::NoWeights`] if no models have been added.
    pub fn blend(&self) -> Result<ModelWeights, WeightAveragingError> {
        if self.models.is_empty() {
            return Err(WeightAveragingError::NoWeights);
        }
        let weight_sum: f32 = self.weights.iter().sum();
        let norm_factor = if weight_sum > 1e-12 {
            1.0 / weight_sum
        } else {
            1.0 / self.models.len() as f32
        };

        let first = &self.models[0];
        let mut result: ModelWeights = first.iter().map(|g| vec![0.0f32; g.len()]).collect();

        for (model, &w) in self.models.iter().zip(self.weights.iter()) {
            let norm_w = w * norm_factor;
            for (rg, mg) in result.iter_mut().zip(model.iter()) {
                for (r, m) in rg.iter_mut().zip(mg.iter()) {
                    *r += norm_w * m;
                }
            }
        }
        Ok(result)
    }

    /// Uniform blend: equal weights for all models, ignoring stored weights.
    ///
    /// # Errors
    ///
    /// Returns [`WeightAveragingError::NoWeights`] if no models have been added.
    pub fn uniform_blend(&self) -> Result<ModelWeights, WeightAveragingError> {
        if self.models.is_empty() {
            return Err(WeightAveragingError::NoWeights);
        }
        let n = self.models.len() as f32;
        let first = &self.models[0];
        let mut result: ModelWeights = first.iter().map(|g| vec![0.0f32; g.len()]).collect();

        for model in &self.models {
            for (rg, mg) in result.iter_mut().zip(model.iter()) {
                for (r, m) in rg.iter_mut().zip(mg.iter()) {
                    *r += m / n;
                }
            }
        }
        Ok(result)
    }

    /// Number of models currently in the soup.
    pub fn num_models(&self) -> usize {
        self.models.len()
    }

    /// Greedy soup: include models one at a time if they improve quality.
    ///
    /// Algorithm:
    /// 1. Sort all models by `eval_fn` score (descending).
    /// 2. Start with the best single model.
    /// 3. For each remaining model: compute uniform average of current
    ///    selection + candidate. If the average improves the score → keep.
    ///
    /// # Errors
    ///
    /// Returns [`WeightAveragingError::NoWeights`] if no models have been added.
    pub fn greedy_soup<F>(&self, eval_fn: F) -> Result<ModelWeights, WeightAveragingError>
    where
        F: Fn(&ModelWeights) -> f32,
    {
        if self.models.is_empty() {
            return Err(WeightAveragingError::NoWeights);
        }

        // Score all models and sort indices by descending score.
        let mut indexed: Vec<(usize, f32)> = self
            .models
            .iter()
            .enumerate()
            .map(|(i, m)| (i, eval_fn(m)))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Start with the best model.
        let (best_idx, _) = indexed[0];
        let mut selected_indices: Vec<usize> = vec![best_idx];

        // Compute initial average (just the best model) and its score.
        let current_avg = self.models[best_idx].clone();
        let mut current_score = eval_fn(&current_avg);
        let mut current_soup = current_avg;

        // Greedy: try adding each remaining model.
        for &(idx, _) in indexed.iter().skip(1) {
            // Compute uniform average of selected + candidate.
            let candidate = &self.models[idx];
            let n = (selected_indices.len() + 1) as f32;
            let trial_avg: ModelWeights = current_soup
                .iter()
                .zip(candidate.iter())
                .map(|(sg, cg)| {
                    sg.iter()
                        .zip(cg.iter())
                        .map(|(s, c)| {
                            // Weighted combination: current soup * k/n + candidate * 1/n
                            let k = (n - 1.0) / n;
                            k * s + (1.0 / n) * c
                        })
                        .collect()
                })
                .collect();

            let trial_score = eval_fn(&trial_avg);
            if trial_score >= current_score {
                current_score = trial_score;
                current_soup = trial_avg;
                selected_indices.push(idx);
            }
        }

        Ok(current_soup)
    }

    /// Compute the pairwise L2 distance matrix between all stored models.
    ///
    /// Returns an `n × n` matrix where element `[i][j]` is the L2 distance
    /// between model `i` and model `j`. Diagonal entries are `0.0`.
    ///
    /// # Errors
    ///
    /// Returns [`WeightAveragingError::LengthMismatch`] if models have
    /// inconsistent shapes (should not occur with valid `add` usage).
    #[allow(clippy::needless_range_loop)]
    pub fn pairwise_distances(&self) -> Result<Vec<Vec<f32>>, WeightAveragingError> {
        let n = self.models.len();
        let mut matrix = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = weights_l2_distance(&self.models[i], &self.models[j])?;
                matrix[i][j] = d;
                matrix[j][i] = d;
            }
        }
        Ok(matrix)
    }
}

impl Default for ModelSoup {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PolyakAverager
// ─────────────────────────────────────────────────────────────────────────────

/// Simple Polyak averaging: maintain a running exponential mean.
///
/// Update rule: `average = decay * average + (1 - decay) * weights`
///
/// On the first call the average is initialised to zero, so the first update
/// produces `(1 - decay) * weights`.
#[derive(Debug, Clone)]
pub struct PolyakAverager {
    /// EMA decay factor (must be in (0, 1)).
    pub decay: f32,
    /// Current running average.
    average: Option<ModelWeights>,
    /// Number of updates applied.
    pub num_updates: usize,
}

impl PolyakAverager {
    /// Create a new Polyak averager.
    ///
    /// # Errors
    ///
    /// Returns [`WeightAveragingError::InvalidConfig`] if `decay` is not in
    /// the open interval `(0, 1)`.
    pub fn new(decay: f32) -> Result<Self, WeightAveragingError> {
        if decay <= 0.0 || decay >= 1.0 {
            return Err(WeightAveragingError::InvalidConfig(format!(
                "decay must be in (0, 1), got {decay}"
            )));
        }
        Ok(Self {
            decay,
            average: None,
            num_updates: 0,
        })
    }

    /// Apply one update: `average = decay * average + (1 - decay) * weights`.
    ///
    /// On the first call, the average starts from zero, so the result is
    /// `(1 - decay) * weights`.
    ///
    /// # Errors
    ///
    /// Returns [`WeightAveragingError::LengthMismatch`] if subsequent
    /// `weights` have a different shape than the first update.
    pub fn update(&mut self, weights: &ModelWeights) -> Result<(), WeightAveragingError> {
        let d = self.decay;
        let one_minus_d = 1.0 - d;
        match &mut self.average {
            None => {
                // Initialise to zero then apply: result = (1 - d) * weights.
                let init: ModelWeights = weights
                    .iter()
                    .map(|g| g.iter().map(|w| one_minus_d * w).collect())
                    .collect();
                self.average = Some(init);
            }
            Some(avg) => {
                let expected = count_params(avg);
                let actual = count_params(weights);
                if avg.len() != weights.len() || expected != actual {
                    return Err(WeightAveragingError::LengthMismatch { expected, actual });
                }
                for (ag, wg) in avg.iter().zip(weights.iter()) {
                    if ag.len() != wg.len() {
                        return Err(WeightAveragingError::LengthMismatch {
                            expected: ag.len(),
                            actual: wg.len(),
                        });
                    }
                }
                for (ag, wg) in avg.iter_mut().zip(weights.iter()) {
                    for (a, w) in ag.iter_mut().zip(wg.iter()) {
                        *a = d * *a + one_minus_d * w;
                    }
                }
            }
        }
        self.num_updates += 1;
        Ok(())
    }

    /// Get read-only access to the current running average.
    pub fn average(&self) -> Option<&ModelWeights> {
        self.average.as_ref()
    }

    /// Bias-corrected average to correct the cold-start under-estimation.
    ///
    /// Returns `average / (1 - decay^num_updates)`.
    ///
    /// Returns `None` if no updates have been applied.
    pub fn bias_corrected_average(&self) -> Option<ModelWeights> {
        let avg = self.average.as_ref()?;
        if self.num_updates == 0 {
            return None;
        }
        let correction = 1.0 - self.decay.powi(self.num_updates as i32);
        if correction < 1e-12 {
            return Some(avg.clone());
        }
        let corrected: ModelWeights = avg
            .iter()
            .map(|g| g.iter().map(|v| v / correction).collect())
            .collect();
        Some(corrected)
    }

    /// Reset the averager, clearing accumulated state.
    pub fn reset(&mut self) {
        self.average = None;
        self.num_updates = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics of a set of model weights.
#[derive(Debug, Clone)]
pub struct WeightStats {
    /// Total number of parameters.
    pub total_params: usize,
    /// L2 norm of all parameters.
    pub l2_norm: f32,
    /// Mean parameter value.
    pub mean: f32,
    /// Standard deviation of parameter values.
    pub std: f32,
    /// Maximum absolute value.
    pub max_abs: f32,
    /// Minimum absolute value.
    pub min_abs: f32,
    /// Number of parameters that are exactly zero.
    pub num_zero: usize,
}

/// Compute summary statistics for a set of model weights.
pub fn compute_weight_stats(weights: &ModelWeights) -> WeightStats {
    let total_params = count_params(weights);
    if total_params == 0 {
        return WeightStats {
            total_params: 0,
            l2_norm: 0.0,
            mean: 0.0,
            std: 0.0,
            max_abs: 0.0,
            min_abs: 0.0,
            num_zero: 0,
        };
    }

    let n = total_params as f32;
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut max_abs = 0.0f32;
    let mut min_abs = f32::INFINITY;
    let mut num_zero = 0usize;

    for group in weights.iter() {
        for &v in group.iter() {
            sum += v;
            sum_sq += v * v;
            let abs_v = v.abs();
            if abs_v > max_abs {
                max_abs = abs_v;
            }
            if abs_v < min_abs {
                min_abs = abs_v;
            }
            if v == 0.0 {
                num_zero += 1;
            }
        }
    }

    let mean = sum / n;
    let variance = (sum_sq / n) - (mean * mean);
    let std = if variance > 0.0 { variance.sqrt() } else { 0.0 };
    let l2_norm = sum_sq.sqrt();

    if min_abs == f32::INFINITY {
        min_abs = 0.0;
    }

    WeightStats {
        total_params,
        l2_norm,
        mean,
        std,
        max_abs,
        min_abs,
        num_zero,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_weights(groups: &[&[f32]]) -> ModelWeights {
        groups.iter().map(|g| g.to_vec()).collect()
    }

    // ── count_params ─────────────────────────────────────────────────────────

    #[test]
    fn test_count_params_correct_total() {
        let w = make_weights(&[&[1.0, 2.0, 3.0], &[4.0, 5.0]]);
        assert_eq!(count_params(&w), 5);
    }

    #[test]
    fn test_count_params_empty() {
        let w: ModelWeights = vec![];
        assert_eq!(count_params(&w), 0);
    }

    // ── weights_l2_distance ──────────────────────────────────────────────────

    #[test]
    fn test_l2_distance_identical_is_zero() {
        let w = make_weights(&[&[1.0, 2.0], &[3.0]]);
        let d = weights_l2_distance(&w, &w).expect("no error");
        assert!(d.abs() < 1e-6, "distance to self should be 0, got {d}");
    }

    #[test]
    fn test_l2_distance_known_value() {
        // a=[3,0], b=[0,4] → sqrt(9+16) = 5
        let a = make_weights(&[&[3.0, 0.0], &[0.0, 4.0]]);
        let b = make_weights(&[&[0.0, 0.0], &[0.0, 0.0]]);
        let d = weights_l2_distance(&a, &b).expect("no error");
        assert!((d - 5.0).abs() < 1e-5, "expected 5.0, got {d}");
    }

    #[test]
    fn test_l2_distance_length_mismatch_returns_err() {
        let a = make_weights(&[&[1.0, 2.0]]);
        let b = make_weights(&[&[1.0]]);
        assert!(weights_l2_distance(&a, &b).is_err());
    }

    // ── weights_cosine_similarity ────────────────────────────────────────────

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let w = make_weights(&[&[1.0, 0.0], &[0.0, 1.0]]);
        let sim = weights_cosine_similarity(&w, &w).expect("no error");
        // sim = dot(w,w) / (|w|^2 + 1e-8) ≈ 1.0
        assert!((sim - 1.0).abs() < 1e-5, "expected ~1.0, got {sim}");
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = make_weights(&[&[1.0, 0.0]]);
        let b = make_weights(&[&[0.0, 1.0]]);
        let sim = weights_cosine_similarity(&a, &b).expect("no error");
        assert!(sim.abs() < 1e-5, "orthogonal vectors → ~0.0, got {sim}");
    }

    // ── SwaConfig::validate ──────────────────────────────────────────────────

    #[test]
    fn test_swa_config_validate_update_every_zero_is_err() {
        let config = SwaConfig {
            update_every: 0,
            ..SwaConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_swa_config_validate_ema_decay_out_of_range_is_err() {
        let config = SwaConfig {
            ema_decay: 1.0,
            ..SwaConfig::default()
        };
        assert!(config.validate().is_err());

        let config2 = SwaConfig {
            ema_decay: 0.0,
            ..SwaConfig::default()
        };
        assert!(config2.validate().is_err());
    }

    // ── StochasticWeightAverager ─────────────────────────────────────────────

    #[test]
    fn test_swa_is_active_false_before_start_step() {
        let config = SwaConfig {
            start_step: 100,
            ..SwaConfig::default()
        };
        let swa = StochasticWeightAverager::new(config).expect("valid config");
        assert!(!swa.is_active());
    }

    #[test]
    fn test_swa_is_active_true_at_start_step() {
        let config = SwaConfig {
            start_step: 0,
            ..SwaConfig::default()
        };
        let swa = StochasticWeightAverager::new(config).expect("valid config");
        assert!(swa.is_active());
    }

    #[test]
    fn test_swa_first_update_sets_weights() {
        let config = SwaConfig::default();
        let mut swa = StochasticWeightAverager::new(config).expect("valid config");
        let weights = make_weights(&[&[1.0, 2.0, 3.0]]);
        swa.update(&weights).expect("update ok");
        assert!(swa.swa_weights().is_some());
        assert_eq!(swa.num_snapshots, 1);
    }

    #[test]
    fn test_swa_uniform_two_equal_weights_is_mean() {
        let config = SwaConfig {
            use_ema: false,
            ..SwaConfig::default()
        };
        let mut swa = StochasticWeightAverager::new(config).expect("valid config");
        let w1 = make_weights(&[&[0.0, 0.0]]);
        let w2 = make_weights(&[&[4.0, 8.0]]);
        swa.update(&w1).expect("update 1 ok");
        swa.update(&w2).expect("update 2 ok");
        let result = swa.swa_weights().expect("weights available");
        // Expected: mean([0,0], [4,8]) = [2,4]
        assert!(
            (result[0][0] - 2.0).abs() < 1e-5,
            "expected 2.0, got {}",
            result[0][0]
        );
        assert!(
            (result[0][1] - 4.0).abs() < 1e-5,
            "expected 4.0, got {}",
            result[0][1]
        );
    }

    #[test]
    fn test_swa_ema_converges_toward_input() {
        let config = SwaConfig {
            use_ema: true,
            ema_decay: 0.9,
            ..SwaConfig::default()
        };
        let mut swa = StochasticWeightAverager::new(config).expect("valid config");
        let w_start = make_weights(&[&[100.0]]);
        swa.update(&w_start).expect("init ok");

        let w_target = make_weights(&[&[0.0]]);
        for _ in 0..200 {
            swa.update(&w_target).expect("update ok");
        }
        let result = swa.swa_weights().expect("weights available");
        assert!(
            result[0][0] < 1.0,
            "EMA should converge toward 0.0, got {}",
            result[0][0]
        );
    }

    #[test]
    fn test_swa_interpolate_alpha_zero_is_current() {
        let config = SwaConfig::default();
        let mut swa = StochasticWeightAverager::new(config).expect("valid config");
        let swa_w = make_weights(&[&[10.0, 10.0]]);
        swa.update(&swa_w).expect("update ok");

        let current = make_weights(&[&[0.0, 0.0]]);
        let result = swa.interpolate(&current, 0.0).expect("interpolate ok");
        assert!((result[0][0] - 0.0).abs() < 1e-6);
        assert!((result[0][1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_swa_interpolate_alpha_one_is_swa() {
        let config = SwaConfig::default();
        let mut swa = StochasticWeightAverager::new(config).expect("valid config");
        let swa_w = make_weights(&[&[10.0, 20.0]]);
        swa.update(&swa_w).expect("update ok");

        let current = make_weights(&[&[0.0, 0.0]]);
        let result = swa.interpolate(&current, 1.0).expect("interpolate ok");
        assert!((result[0][0] - 10.0).abs() < 1e-6);
        assert!((result[0][1] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_swa_reset_clears_weights() {
        let config = SwaConfig::default();
        let mut swa = StochasticWeightAverager::new(config).expect("valid config");
        let w = make_weights(&[&[1.0, 2.0]]);
        swa.update(&w).expect("update ok");
        assert!(swa.swa_weights().is_some());
        swa.reset();
        assert!(swa.swa_weights().is_none());
        assert_eq!(swa.num_snapshots, 0);
    }

    // ── ModelSoup ────────────────────────────────────────────────────────────

    #[test]
    fn test_model_soup_add_first_model() {
        let mut soup = ModelSoup::new();
        let m = make_weights(&[&[1.0, 2.0]]);
        soup.add(m).expect("add ok");
        assert_eq!(soup.num_models(), 1);
    }

    #[test]
    fn test_model_soup_add_dimension_mismatch_is_err() {
        let mut soup = ModelSoup::new();
        soup.add(make_weights(&[&[1.0, 2.0]]))
            .expect("add first ok");
        let bad = make_weights(&[&[1.0]]); // different length
        assert!(soup.add(bad).is_err());
    }

    #[test]
    fn test_model_soup_uniform_blend_two_equal_models() {
        let mut soup = ModelSoup::new();
        soup.add(make_weights(&[&[0.0, 0.0]])).expect("add ok");
        soup.add(make_weights(&[&[4.0, 8.0]])).expect("add ok");
        let result = soup.uniform_blend().expect("blend ok");
        assert!((result[0][0] - 2.0).abs() < 1e-5);
        assert!((result[0][1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_model_soup_weighted_blend_correct() {
        let mut soup = ModelSoup::new();
        // model 0: [0.0], weight 1.0 → contribution: 0.0
        // model 1: [6.0], weight 3.0 → contribution: 4.5
        // normalized: 0*(1/4) + 6*(3/4) = 4.5
        soup.add_weighted(make_weights(&[&[0.0]]), 1.0)
            .expect("add ok");
        soup.add_weighted(make_weights(&[&[6.0]]), 3.0)
            .expect("add ok");
        let result = soup.blend().expect("blend ok");
        assert!(
            (result[0][0] - 4.5).abs() < 1e-5,
            "expected 4.5, got {}",
            result[0][0]
        );
    }

    #[test]
    fn test_model_soup_blend_empty_is_err() {
        let soup = ModelSoup::new();
        assert!(soup.blend().is_err());
    }

    #[test]
    fn test_model_soup_greedy_soup_best_model_selected() {
        let mut soup = ModelSoup::new();
        // Model 0: [1.0] → score 1.0
        // Model 1: [2.0] → score 2.0 (best)
        // Model 2: [3.0] → score 3.0 but average with model1 = 2.5, score 2.5 > 2.0 → add
        // After models 1 and 2: score 2.5
        soup.add(make_weights(&[&[1.0]])).expect("add ok");
        soup.add(make_weights(&[&[10.0]])).expect("add ok");
        soup.add(make_weights(&[&[0.0]])).expect("add ok");

        // eval_fn: returns the value (higher value = better)
        let result = soup.greedy_soup(|w| w[0][0]).expect("greedy ok");
        // Best single model is [10.0]. Adding [0.0] would lower to 5.0 < 10.0 → skip.
        // Adding [1.0] would lower to 5.5 < 10.0 → skip. So result should be [10.0].
        assert!(
            (result[0][0] - 10.0).abs() < 1e-5,
            "expected 10.0, got {}",
            result[0][0]
        );
    }

    #[test]
    fn test_model_soup_pairwise_distances_diagonal_is_zero() {
        let mut soup = ModelSoup::new();
        soup.add(make_weights(&[&[1.0, 0.0]])).expect("add ok");
        soup.add(make_weights(&[&[0.0, 1.0]])).expect("add ok");
        soup.add(make_weights(&[&[0.0, 0.0]])).expect("add ok");
        let matrix = soup.pairwise_distances().expect("distances ok");
        for (i, row) in matrix.iter().enumerate() {
            assert!(
                row[i].abs() < 1e-6,
                "diagonal[{i}] should be 0.0, got {}",
                row[i]
            );
        }
    }

    // ── PolyakAverager ───────────────────────────────────────────────────────

    #[test]
    fn test_polyak_averager_new_decay_zero_is_err() {
        assert!(PolyakAverager::new(0.0).is_err());
    }

    #[test]
    fn test_polyak_averager_new_decay_one_is_err() {
        assert!(PolyakAverager::new(1.0).is_err());
    }

    #[test]
    fn test_polyak_averager_first_update_equals_one_minus_decay_times_weights() {
        let decay = 0.9_f32;
        let mut pa = PolyakAverager::new(decay).expect("valid decay");
        let weights = make_weights(&[&[10.0, 20.0]]);
        pa.update(&weights).expect("update ok");
        let avg = pa.average().expect("average available");
        // First update from zero: (1 - 0.9) * 10.0 = 1.0
        assert!(
            (avg[0][0] - 1.0).abs() < 1e-5,
            "expected 1.0, got {}",
            avg[0][0]
        );
        assert!(
            (avg[0][1] - 2.0).abs() < 1e-5,
            "expected 2.0, got {}",
            avg[0][1]
        );
    }

    #[test]
    fn test_polyak_averager_bias_corrected_converges_to_true_average() {
        // After many updates with constant weights, bias-corrected average → weights.
        let decay = 0.999_f32;
        let mut pa = PolyakAverager::new(decay).expect("valid decay");
        let weights = make_weights(&[&[5.0]]);
        for _ in 0..5000 {
            pa.update(&weights).expect("update ok");
        }
        let corrected = pa.bias_corrected_average().expect("corrected available");
        assert!(
            (corrected[0][0] - 5.0).abs() < 0.1,
            "bias-corrected average should converge to 5.0, got {}",
            corrected[0][0]
        );
    }

    // ── compute_weight_stats ─────────────────────────────────────────────────

    #[test]
    fn test_compute_weight_stats_l2_norm_and_mean() {
        // weights: [3.0, 4.0] → l2_norm = 5.0, mean = 3.5
        let weights = make_weights(&[&[3.0, 4.0]]);
        let stats = compute_weight_stats(&weights);
        assert_eq!(stats.total_params, 2);
        assert!(
            (stats.l2_norm - 5.0).abs() < 1e-5,
            "l2_norm: expected 5.0, got {}",
            stats.l2_norm
        );
        assert!(
            (stats.mean - 3.5).abs() < 1e-5,
            "mean: expected 3.5, got {}",
            stats.mean
        );
    }

    #[test]
    fn test_compute_weight_stats_zero_params() {
        let weights: ModelWeights = vec![vec![]];
        let stats = compute_weight_stats(&weights);
        assert_eq!(stats.total_params, 0);
        assert_eq!(stats.l2_norm, 0.0);
    }

    #[test]
    fn test_compute_weight_stats_num_zero() {
        let weights = make_weights(&[&[0.0, 1.0, 0.0, 2.0]]);
        let stats = compute_weight_stats(&weights);
        assert_eq!(stats.num_zero, 2);
    }
}
