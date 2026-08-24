//! Hyperparameter sweep management for 3DGS training.
//!
//! Provides grid search, random search, and a simplified pseudo-Bayesian surrogate
//! search for tuning 3D Gaussian Splatting training hyperparameters.
//!
//! # Example
//! ```rust
//! use oxigaf_cli::parameter_sweep::{
//!     ParamSpec, SweepConfig, SweepStrategy, ParameterSweep,
//! };
//!
//! let config = SweepConfig {
//!     specs: vec![
//!         ParamSpec::Continuous { name: "lr".into(), low: 1e-4, high: 1e-2, log_scale: true },
//!         ParamSpec::Discrete { name: "n_sh".into(), values: vec![1.0, 4.0, 9.0] },
//!     ],
//!     strategy: SweepStrategy::Random,
//!     max_trials: 20,
//!     seed: 42,
//!     minimize: true,
//! };
//!
//! let mut sweep = ParameterSweep::new(config).expect("Failed to create sweep");
//! let trial = sweep.suggest().expect("Failed to suggest trial");
//! println!("Trial {}: {}", trial.id, oxigaf_cli::parameter_sweep::format_sweep_trial(&trial));
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// xorshift64 PRNG
// ---------------------------------------------------------------------------

fn xorshift64(state: &mut u64) -> u64 {
    (*state) ^= (*state) << 13;
    (*state) ^= (*state) >> 7;
    (*state) ^= (*state) << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

fn xorshift_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64
}

// ---------------------------------------------------------------------------
// SweepError
// ---------------------------------------------------------------------------

/// Errors that can occur during a hyperparameter sweep.
#[derive(Debug, Error)]
pub enum SweepError {
    /// Grid search not supported for Continuous specs.
    #[error("Grid search not supported for Continuous specs")]
    GridNotSupportedForContinuous,

    /// No parameter specs provided.
    #[error("Empty parameter spec list")]
    EmptySpecs,

    /// A score was reported for an unknown trial ID.
    #[error("Trial {0} not found")]
    TrialNotFound(usize),

    /// Invalid parameter value or range.
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),

    /// Discrete spec has no values.
    #[error("Discrete spec has no values")]
    EmptyDiscrete,

    /// All configured trials have been generated.
    #[error("Maximum number of trials ({max}) already reached")]
    MaxTrialsReached { max: usize },

    /// Grid index out of bounds.
    #[error("Grid index out of bounds for dimension sizes {dims:?} at trial index {trial_idx}")]
    GridIndexOutOfBounds { dims: Vec<usize>, trial_idx: usize },
}

// ---------------------------------------------------------------------------
// ParamSpec
// ---------------------------------------------------------------------------

/// A single hyperparameter dimension.
#[derive(Debug, Clone)]
pub enum ParamSpec {
    /// Continuous real-valued parameter with optional log scaling.
    Continuous {
        name: String,
        low: f64,
        high: f64,
        log_scale: bool,
    },
    /// Discrete numeric parameter chosen from a fixed list of values.
    Discrete { name: String, values: Vec<f64> },
    /// Categorical parameter chosen from a fixed list of strings.
    Categorical { name: String, choices: Vec<String> },
}

impl ParamSpec {
    /// Returns the name of this parameter.
    pub fn name(&self) -> &str {
        match self {
            ParamSpec::Continuous { name, .. } => name,
            ParamSpec::Discrete { name, .. } => name,
            ParamSpec::Categorical { name, .. } => name,
        }
    }

    /// Number of grid points for this spec (for Discrete/Categorical only).
    fn grid_size(&self) -> Option<usize> {
        match self {
            ParamSpec::Continuous { .. } => None,
            ParamSpec::Discrete { values, .. } => Some(values.len()),
            ParamSpec::Categorical { choices, .. } => Some(choices.len()),
        }
    }
}

// ---------------------------------------------------------------------------
// ParamValue
// ---------------------------------------------------------------------------

/// The resolved value of a single parameter in a trial.
#[derive(Debug, Clone)]
pub enum ParamValue {
    /// Floating-point value (from Continuous or Discrete specs).
    Float(f64),
    /// Integer value.
    Int(i64),
    /// Categorical string choice.
    Choice(String),
}

impl std::fmt::Display for ParamValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamValue::Float(v) => write!(f, "{:.6}", v),
            ParamValue::Int(v) => write!(f, "{}", v),
            ParamValue::Choice(s) => write!(f, "{}", s),
        }
    }
}

// ---------------------------------------------------------------------------
// Trial
// ---------------------------------------------------------------------------

/// A single trial with its parameter assignments and optional score.
#[derive(Debug, Clone)]
pub struct SweepTrial {
    /// Unique trial identifier.
    pub id: usize,
    /// Parameter name-value pairs for this trial.
    pub params: Vec<(String, ParamValue)>,
    /// Score reported after evaluation (lower is better for minimize=true).
    pub score: Option<f64>,
}

// ---------------------------------------------------------------------------
// SweepConfig
// ---------------------------------------------------------------------------

/// Configuration for a hyperparameter sweep.
#[derive(Debug, Clone)]
pub struct SweepConfig {
    /// Parameter specifications defining the search space.
    pub specs: Vec<ParamSpec>,
    /// Search strategy to use.
    pub strategy: SweepStrategy,
    /// Maximum number of trials to run.
    pub max_trials: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// If true, lower score is better (loss minimization).
    pub minimize: bool,
}

// ---------------------------------------------------------------------------
// SweepStrategy
// ---------------------------------------------------------------------------

/// Strategy for generating trial parameter combinations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SweepStrategy {
    /// Cartesian product over Discrete/Categorical dims (no Continuous allowed).
    Grid,
    /// Uniform random sampling over the search space.
    Random,
    /// KNN-surrogate guided search (falls back to random with no completed trials).
    Surrogate,
}

// ---------------------------------------------------------------------------
// SweepSummary
// ---------------------------------------------------------------------------

/// Aggregated statistics about a completed or in-progress sweep.
#[derive(Debug, Clone)]
pub struct SweepSummary {
    /// Total trials that have been generated.
    pub total_trials: usize,
    /// Trials that have had a score reported.
    pub completed_trials: usize,
    /// Best (lowest if minimize, highest otherwise) score seen.
    pub best_score: Option<f64>,
    /// Worst score seen.
    pub worst_score: Option<f64>,
    /// Mean score across completed trials.
    pub mean_score: Option<f64>,
    /// Standard deviation of scores across completed trials.
    pub std_score: Option<f64>,
    /// Per-parameter importance scores, summing to 1.0.
    pub param_importances: Vec<(String, f64)>,
}

// ---------------------------------------------------------------------------
// ParameterSweep
// ---------------------------------------------------------------------------

/// Manages the state of a hyperparameter sweep.
pub struct ParameterSweep {
    config: SweepConfig,
    trials: Vec<SweepTrial>,
    rng_state: u64,
    trial_counter: usize,
}

impl ParameterSweep {
    /// Create a new sweep from the given config.
    ///
    /// # Errors
    /// Returns `SweepError::EmptySpecs` if no specs are provided.
    /// Returns `SweepError::GridNotSupportedForContinuous` if Grid strategy
    /// is used with any Continuous spec.
    pub fn new(config: SweepConfig) -> Result<Self, SweepError> {
        if config.specs.is_empty() {
            return Err(SweepError::EmptySpecs);
        }
        if config.strategy == SweepStrategy::Grid {
            for spec in &config.specs {
                if matches!(spec, ParamSpec::Continuous { .. }) {
                    return Err(SweepError::GridNotSupportedForContinuous);
                }
            }
        }
        let seed = if config.seed == 0 { 1 } else { config.seed };
        Ok(Self {
            config,
            trials: Vec::new(),
            rng_state: seed,
            trial_counter: 0,
        })
    }

    /// Generate the next trial parameter assignment.
    ///
    /// # Errors
    /// Returns `SweepError::MaxTrialsReached` if all trials have been generated.
    pub fn suggest(&mut self) -> Result<SweepTrial, SweepError> {
        if self.trial_counter >= self.config.max_trials {
            return Err(SweepError::MaxTrialsReached {
                max: self.config.max_trials,
            });
        }
        let trial = match self.config.strategy {
            SweepStrategy::Grid => self.suggest_grid()?,
            SweepStrategy::Random => self.suggest_random()?,
            SweepStrategy::Surrogate => self.suggest_surrogate()?,
        };
        self.trials.push(trial.clone());
        self.trial_counter += 1;
        Ok(trial)
    }

    fn suggest_grid(&mut self) -> Result<SweepTrial, SweepError> {
        let dims: Vec<usize> = self
            .config
            .specs
            .iter()
            .map(|s| s.grid_size().unwrap_or(1))
            .collect();
        let indices = sweep_grid_indices(&dims, self.trial_counter);
        let mut params = Vec::with_capacity(self.config.specs.len());
        for (spec, &idx) in self.config.specs.iter().zip(indices.iter()) {
            let pv = match spec {
                ParamSpec::Continuous { .. } => {
                    return Err(SweepError::GridNotSupportedForContinuous);
                }
                ParamSpec::Discrete { name, values } => {
                    let v = values
                        .get(idx)
                        .copied()
                        .ok_or(SweepError::GridIndexOutOfBounds {
                            dims: dims.clone(),
                            trial_idx: self.trial_counter,
                        })?;
                    (name.clone(), ParamValue::Float(v))
                }
                ParamSpec::Categorical { name, choices } => {
                    let c = choices
                        .get(idx)
                        .cloned()
                        .ok_or(SweepError::GridIndexOutOfBounds {
                            dims: dims.clone(),
                            trial_idx: self.trial_counter,
                        })?;
                    (name.clone(), ParamValue::Choice(c))
                }
            };
            params.push(pv);
        }
        Ok(SweepTrial {
            id: self.trial_counter,
            params,
            score: None,
        })
    }

    fn suggest_random(&mut self) -> Result<SweepTrial, SweepError> {
        let params = sample_params_random(&self.config.specs, &mut self.rng_state)?;
        Ok(SweepTrial {
            id: self.trial_counter,
            params,
            score: None,
        })
    }

    fn suggest_surrogate(&mut self) -> Result<SweepTrial, SweepError> {
        let completed: Vec<&SweepTrial> =
            self.trials.iter().filter(|t| t.score.is_some()).collect();
        if completed.is_empty() {
            return self.suggest_random();
        }
        // Generate 10 random candidates and pick the best predicted score.
        let minimize = self.config.minimize;
        let mut best_params: Option<Vec<(String, ParamValue)>> = None;
        let mut best_predicted = f64::NAN;
        let all_trials: Vec<SweepTrial> = self.trials.clone();
        for _ in 0..10 {
            let candidate = sample_params_random(&self.config.specs, &mut self.rng_state)?;
            let predicted = sweep_surrogate_predict(&all_trials, &candidate, &self.config.specs);
            let better = if best_params.is_none() {
                true
            } else if minimize {
                predicted < best_predicted
            } else {
                predicted > best_predicted
            };
            if better {
                best_predicted = predicted;
                best_params = Some(candidate);
            }
        }
        let params = best_params.ok_or(SweepError::EmptySpecs)?;
        Ok(SweepTrial {
            id: self.trial_counter,
            params,
            score: None,
        })
    }

    /// Report a score for a previously suggested trial.
    ///
    /// # Errors
    /// Returns `SweepError::TrialNotFound` if the trial ID does not exist.
    pub fn report(&mut self, trial_id: usize, score: f64) -> Result<(), SweepError> {
        let trial = self
            .trials
            .iter_mut()
            .find(|t| t.id == trial_id)
            .ok_or(SweepError::TrialNotFound(trial_id))?;
        trial.score = Some(score);
        Ok(())
    }

    /// Return the best trial seen so far, according to the minimize setting.
    pub fn best_trial(&self) -> Option<&SweepTrial> {
        let scored: Vec<&SweepTrial> = self.trials.iter().filter(|t| t.score.is_some()).collect();
        if scored.is_empty() {
            return None;
        }
        let minimize = self.config.minimize;
        scored.into_iter().reduce(|best, t| {
            let bs = best.score.unwrap_or(f64::NAN);
            let ts = t.score.unwrap_or(f64::NAN);
            let t_better = if minimize { ts < bs } else { ts > bs };
            if t_better {
                t
            } else {
                best
            }
        })
    }

    /// Return the top `k` trials sorted from best to worst score.
    pub fn top_k_trials(&self, k: usize) -> Vec<&SweepTrial> {
        let mut scored: Vec<&SweepTrial> =
            self.trials.iter().filter(|t| t.score.is_some()).collect();
        let minimize = self.config.minimize;
        scored.sort_by(|a, b| {
            let sa = a.score.unwrap_or(f64::NAN);
            let sb = b.score.unwrap_or(f64::NAN);
            if minimize {
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            }
        });
        scored.into_iter().take(k).collect()
    }

    /// Number of trials with a reported score.
    pub fn trials_completed(&self) -> usize {
        self.trials.iter().filter(|t| t.score.is_some()).count()
    }

    /// Returns true if the maximum number of trials has been generated.
    pub fn is_done(&self) -> bool {
        self.trial_counter >= self.config.max_trials
    }

    /// Produce a summary of the current sweep state.
    pub fn summary(&self) -> SweepSummary {
        let scores: Vec<f64> = self.trials.iter().filter_map(|t| t.score).collect();
        let completed_trials = scores.len();
        let (best_score, worst_score, mean_score, std_score) = if scores.is_empty() {
            (None, None, None, None)
        } else {
            let minimize = self.config.minimize;
            let best = if minimize {
                scores.iter().cloned().fold(f64::INFINITY, f64::min)
            } else {
                scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            };
            let worst = if minimize {
                scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            } else {
                scores.iter().cloned().fold(f64::INFINITY, f64::min)
            };
            let n = scores.len() as f64;
            let mean = scores.iter().sum::<f64>() / n;
            let variance = scores.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / n;
            let std = variance.sqrt();
            (Some(best), Some(worst), Some(mean), Some(std))
        };
        let param_importances = sweep_param_importance(&self.trials, &self.config.specs);
        SweepSummary {
            total_trials: self.trial_counter,
            completed_trials,
            best_score,
            worst_score,
            mean_score,
            std_score,
            param_importances,
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Sample a continuous parameter value.
///
/// If `log_scale` is true, samples uniformly in log-space: `exp(uniform(log(low), log(high)))`.
///
/// # Errors
/// Returns `SweepError::InvalidParam` if `low >= high` or if `log_scale`
/// and `low <= 0.0`.
pub fn sweep_sample_continuous(
    low: f64,
    high: f64,
    log_scale: bool,
    state: &mut u64,
) -> Result<f64, SweepError> {
    if low >= high {
        return Err(SweepError::InvalidParam(format!(
            "low ({}) must be less than high ({})",
            low, high
        )));
    }
    if log_scale && low <= 0.0 {
        return Err(SweepError::InvalidParam(format!(
            "log_scale requires low > 0.0, got low={}",
            low
        )));
    }
    if log_scale {
        let log_low = low.ln();
        let log_high = high.ln();
        let t = xorshift_f64(state);
        Ok((log_low + t * (log_high - log_low)).exp())
    } else {
        let t = xorshift_f64(state);
        Ok(low + t * (high - low))
    }
}

/// Sample a value uniformly from a discrete list.
///
/// # Errors
/// Returns `SweepError::EmptyDiscrete` if `values` is empty.
pub fn sweep_sample_discrete(values: &[f64], state: &mut u64) -> Result<f64, SweepError> {
    if values.is_empty() {
        return Err(SweepError::EmptyDiscrete);
    }
    let idx = xorshift64(state) as usize % values.len();
    Ok(values[idx])
}

/// Compute the per-dimension indices for a mixed-radix grid counter.
///
/// Given dimension sizes `dims` and a trial index, returns the index into each
/// dimension using a mixed-radix (row-major) decomposition.
///
/// For dimension `i`, the index is:
/// `(trial_idx / product(dims[i+1..])) % dims[i]`
pub fn sweep_grid_indices(dims: &[usize], trial_idx: usize) -> Vec<usize> {
    let n = dims.len();
    let mut indices = vec![0usize; n];
    let remaining = trial_idx;
    // Compute suffix products: suffix[i] = product of dims[i+1..]
    let mut suffix = vec![1usize; n];
    for i in (0..n.saturating_sub(1)).rev() {
        suffix[i] = suffix[i + 1].saturating_mul(dims[i + 1]);
    }
    for i in 0..n {
        if dims[i] == 0 {
            indices[i] = 0;
        } else {
            indices[i] = (remaining / suffix[i]) % dims[i];
        }
    }
    indices
}

/// Predict the score for a candidate parameter set using inverse-distance weighted KNN.
///
/// Uses K = min(5, number of completed trials).
/// Distance is computed in normalized parameter space (each dimension scaled
/// to \[0,1\] using `specs`' declared bounds; see [`param_value_distance`]).
/// Weight = 1 / (dist^2 + 1e-8).
/// Returns 0.0 if there are no scored trials.
pub fn sweep_surrogate_predict(
    trials: &[SweepTrial],
    params: &[(String, ParamValue)],
    specs: &[ParamSpec],
) -> f64 {
    let scored: Vec<&SweepTrial> = trials.iter().filter(|t| t.score.is_some()).collect();
    if scored.is_empty() {
        return 0.0;
    }
    let k = scored.len().min(5);

    // Compute distances from candidate to all scored trials.
    let mut dist_scores: Vec<(f64, f64)> = scored
        .iter()
        .map(|trial| {
            let dist = param_distance(params, &trial.params, specs);
            let score = trial.score.unwrap_or(0.0);
            (dist, score)
        })
        .collect();

    // Sort by distance ascending.
    dist_scores.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Take K nearest.
    let nearest = &dist_scores[..k];

    // Inverse-distance weighting.
    let mut weight_sum = 0.0f64;
    let mut weighted_score = 0.0f64;
    for &(dist, score) in nearest {
        let w = 1.0 / (dist * dist + 1e-8);
        weight_sum += w;
        weighted_score += w * score;
    }
    if weight_sum < 1e-15 {
        return scored[0].score.unwrap_or(0.0);
    }
    weighted_score / weight_sum
}

/// Compute the importance of each parameter based on rank correlation with scores.
///
/// Uses Spearman rank correlation: `rho = 1 - 6*sum(d^2) / (n*(n^2-1))`.
/// Returns importances normalized to sum to 1.0.
/// Returns an empty vec if fewer than 2 scored trials exist.
pub fn sweep_param_importance(trials: &[SweepTrial], specs: &[ParamSpec]) -> Vec<(String, f64)> {
    let scored: Vec<&SweepTrial> = trials.iter().filter(|t| t.score.is_some()).collect();
    if scored.len() < 2 {
        return specs.iter().map(|s| (s.name().to_string(), 0.0)).collect();
    }
    let n = scored.len();

    // Score ranks.
    let scores: Vec<f64> = scored.iter().map(|t| t.score.unwrap_or(0.0)).collect();
    let score_ranks = compute_ranks(&scores);

    let mut raw_importances: Vec<(String, f64)> = Vec::with_capacity(specs.len());

    for spec in specs {
        let name = spec.name().to_string();
        // Extract numeric proxy for each trial for this param.
        let param_vals: Vec<f64> = scored
            .iter()
            .map(|trial| numeric_proxy_for_param(trial, spec))
            .collect();
        let param_ranks = compute_ranks(&param_vals);

        // Spearman rank correlation.
        let rho = if n <= 2 {
            0.0
        } else {
            let d_sq_sum: f64 = score_ranks
                .iter()
                .zip(param_ranks.iter())
                .map(|(r1, r2)| (r1 - r2).powi(2))
                .sum();
            let n_f = n as f64;
            1.0 - 6.0 * d_sq_sum / (n_f * (n_f * n_f - 1.0))
        };

        raw_importances.push((name, rho.abs()));
    }

    // Normalize to sum 1.0.
    let total: f64 = raw_importances.iter().map(|(_, v)| v).sum();
    if total < 1e-12 {
        let uniform = if specs.is_empty() {
            0.0
        } else {
            1.0 / specs.len() as f64
        };
        raw_importances.iter_mut().for_each(|(_, v)| *v = uniform);
    } else {
        raw_importances.iter_mut().for_each(|(_, v)| *v /= total);
    }
    raw_importances
}

/// Format a trial as a human-readable string.
pub fn format_sweep_trial(trial: &SweepTrial) -> String {
    let mut parts = Vec::with_capacity(trial.params.len() + 2);
    parts.push(format!("Trial #{}", trial.id));
    for (name, value) in &trial.params {
        parts.push(format!("{}={}", name, value));
    }
    if let Some(score) = trial.score {
        parts.push(format!("score={:.6}", score));
    } else {
        parts.push("score=pending".to_string());
    }
    parts.join(", ")
}

/// Format a sweep summary as a human-readable multi-line string.
pub fn format_sweep_summary(summary: &SweepSummary) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "Sweep Summary: {}/{} trials completed",
        summary.completed_trials, summary.total_trials
    ));
    if let Some(b) = summary.best_score {
        lines.push(format!("  Best score:  {:.6}", b));
    }
    if let Some(w) = summary.worst_score {
        lines.push(format!("  Worst score: {:.6}", w));
    }
    if let Some(m) = summary.mean_score {
        lines.push(format!("  Mean score:  {:.6}", m));
    }
    if let Some(s) = summary.std_score {
        lines.push(format!("  Std score:   {:.6}", s));
    }
    if !summary.param_importances.is_empty() {
        lines.push("  Parameter importances:".to_string());
        for (name, imp) in &summary.param_importances {
            lines.push(format!("    {}: {:.4}", name, imp));
        }
    }
    lines.join("\n")
}

/// Compute the Hyperband bracket: list of `(n_configs, budget)` per round.
///
/// Rounds are ordered from the innermost (fewest configs, largest budget)
/// to the outermost (most configs, smallest budget).
///
/// Formula (matches Li et al. 2016's published Hyperband algorithm):
/// - `s_max = floor(log_eta(max_iter))`
/// - Round `r` (0..=s_max): `n_r = ceil((s_max+1) / (s_max-r+1) * eta^(s_max-r))`,
///   `budget_r = floor(max_iter / eta^(s_max-r))`
/// - Returns rounds from `r=s_max` down to `r=0`.
///
/// The `/ (s_max-r+1)` factor was previously omitted, which over-allocated
/// configurations in the wide/cheap rounds: for `max_iter=81, eta=3` this
/// used to yield `5, 15, 45, 135, 405` configs per round instead of the
/// correct `5, 8, 15, 34, 81`, so a caller budgeting total work from these
/// counts planned several times the intended amount.
pub fn hyperband_bracket(max_iter: usize, eta: usize) -> Vec<(usize, usize)> {
    if max_iter == 0 || eta < 2 {
        return Vec::new();
    }
    // Use round-then-floor to avoid floating-point underflow on exact powers.
    let raw_log = (max_iter as f64).log(eta as f64);
    // If raw_log is within 1e-9 of an integer (e.g. log_3(243) ≈ 5.0 - eps),
    // round to that integer first so exact powers are handled correctly.
    let s_max = if (raw_log - raw_log.round()).abs() < 1e-9 {
        raw_log.round() as usize
    } else {
        raw_log.floor() as usize
    };
    let mut rounds = Vec::with_capacity(s_max + 1);

    // r iterates from s_max down to 0 (innermost first).
    for r_rev in 0..=s_max {
        let r = s_max - r_rev; // actual r value
        let power = s_max.saturating_sub(r) as u32;
        let eta_pow = (eta as f64).powi(power as i32);
        // `power` here is `s_max - r`, i.e. the paper's `s` in `eta^s`; the
        // published formula divides by `(s + 1)` = `(power + 1)`, not by
        // `(r + 1)` (this function's own, differently-scoped `r`).
        let n_configs = (((s_max + 1) as f64 / (power + 1) as f64) * eta_pow).ceil() as usize;
        let n_configs = n_configs.max(1);
        let budget = if eta_pow < 1.0 {
            0usize
        } else {
            ((max_iter as f64) / eta_pow).floor() as usize
        };
        rounds.push((n_configs, budget));
    }
    rounds
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Sample a full parameter set randomly for all specs.
fn sample_params_random(
    specs: &[ParamSpec],
    rng: &mut u64,
) -> Result<Vec<(String, ParamValue)>, SweepError> {
    let mut params = Vec::with_capacity(specs.len());
    for spec in specs {
        let pv = match spec {
            ParamSpec::Continuous {
                name,
                low,
                high,
                log_scale,
            } => {
                let v = sweep_sample_continuous(*low, *high, *log_scale, rng)?;
                (name.clone(), ParamValue::Float(v))
            }
            ParamSpec::Discrete { name, values } => {
                let v = sweep_sample_discrete(values, rng)?;
                (name.clone(), ParamValue::Float(v))
            }
            ParamSpec::Categorical { name, choices } => {
                if choices.is_empty() {
                    return Err(SweepError::EmptyDiscrete);
                }
                let idx = xorshift64(rng) as usize % choices.len();
                let c = choices[idx].clone();
                (name.clone(), ParamValue::Choice(c))
            }
        };
        params.push(pv);
    }
    Ok(params)
}

/// Compute the L2 distance between two parameter sets in normalized space.
///
/// Categorical: 0.0 if same choice, 1.0 if different.
/// Float: difference normalized to [0,1] using each parameter's declared
/// bounds in `specs` (see [`param_value_distance`]).
fn param_distance(
    a: &[(String, ParamValue)],
    b: &[(String, ParamValue)],
    specs: &[ParamSpec],
) -> f64 {
    let mut sq_sum = 0.0f64;
    for (name_a, val_a) in a {
        if let Some((_, val_b)) = b.iter().find(|(n, _)| n == name_a) {
            let spec = specs.iter().find(|s| s.name() == name_a);
            let d = param_value_distance(spec, val_a, val_b);
            sq_sum += d * d;
        }
    }
    sq_sum.sqrt()
}

/// Distance between two `ParamValue`s in \[0, 1\].
///
/// `spec` (matched by name to the parameter) normalizes a `Float` distance
/// by the parameter's actual declared range instead of the un-normalized
/// `diff / (diff + 1.0)` this used previously -- which put every `Float`
/// parameter on the same soft [0,1) scale regardless of magnitude (a
/// learning rate spanning `1e-4..1e-2` always scored near 0 next to a
/// `0..1000` parameter always near 1, making the KNN surrogate in
/// [`sweep_surrogate_predict`] effectively blind to small-range dimensions).
/// See [`normalized_float_distance`] for the normalization itself, used
/// when `spec` has no usable bounds (no match, or a degenerate range).
fn param_value_distance(spec: Option<&ParamSpec>, a: &ParamValue, b: &ParamValue) -> f64 {
    match (a, b) {
        (ParamValue::Choice(ca), ParamValue::Choice(cb)) => {
            if ca == cb {
                0.0
            } else {
                1.0
            }
        }
        (ParamValue::Float(fa), ParamValue::Float(fb)) => normalized_float_distance(spec, *fa, *fb),
        // No `ParamSpec` variant currently produces an `Int` (see
        // `ParamValue`'s doc); kept as a total fallback, not reachable today.
        (ParamValue::Int(ia), ParamValue::Int(ib)) => {
            let diff = (ia - ib).unsigned_abs() as f64;
            diff / (diff + 1.0)
        }
        // Mixed types: treat as maximally different.
        _ => 1.0,
    }
}

/// Normalize the distance between two `Float` parameter values to \[0, 1\]
/// using `spec`'s declared bounds (log-space for a log-scaled
/// [`ParamSpec::Continuous`], matching how [`sweep_sample_continuous`]
/// samples it; linear range for `Discrete`'s `values`), falling back to the
/// spec-agnostic `diff / (diff + 1.0)` when no usable bounds are found.
fn normalized_float_distance(spec: Option<&ParamSpec>, a: f64, b: f64) -> f64 {
    let bounds: Option<(f64, f64, bool)> = match spec {
        Some(ParamSpec::Continuous {
            low,
            high,
            log_scale,
            ..
        }) => Some((*low, *high, *log_scale)),
        Some(ParamSpec::Discrete { values, .. }) if !values.is_empty() => {
            let low = values.iter().copied().fold(f64::INFINITY, f64::min);
            let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            Some((low, high, false))
        }
        _ => None,
    };

    if let Some((low, high, log_scale)) = bounds {
        if log_scale && low > 0.0 && high > 0.0 {
            let log_low = low.ln();
            let range = high.ln() - log_low;
            if range.abs() > 1e-12 {
                let na = (a.max(f64::MIN_POSITIVE).ln() - log_low) / range;
                let nb = (b.max(f64::MIN_POSITIVE).ln() - log_low) / range;
                return (na - nb).abs().clamp(0.0, 1.0);
            }
        } else {
            let range = high - low;
            if range.abs() > 1e-12 {
                let na = (a - low) / range;
                let nb = (b - low) / range;
                return (na - nb).abs().clamp(0.0, 1.0);
            }
        }
    }

    // Fallback: no matching/usable spec (unmatched name, or a degenerate
    // zero-width / non-positive-for-log-scale range).
    let diff = (a - b).abs();
    diff / (diff + 1.0)
}

/// Extract a numeric proxy for a parameter value given its spec.
///
/// For Categorical, uses the index of the choice in the spec's choices list.
/// For Float/Int, uses the raw value.
fn numeric_proxy_for_param(trial: &SweepTrial, spec: &ParamSpec) -> f64 {
    let name = spec.name();
    let pv = trial.params.iter().find(|(n, _)| n == name).map(|(_, v)| v);
    match (spec, pv) {
        (ParamSpec::Categorical { choices, .. }, Some(ParamValue::Choice(c))) => {
            choices.iter().position(|ch| ch == c).unwrap_or(0) as f64
        }
        (_, Some(ParamValue::Float(f))) => *f,
        (_, Some(ParamValue::Int(i))) => *i as f64,
        _ => 0.0,
    }
}

/// Compute fractional ranks (1-indexed) for a slice of values.
///
/// Ties are broken by taking the average rank.
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    // Create (value, original_index) pairs sorted by value.
    let mut indexed: Vec<(f64, usize)> = values
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        // Find the run of equal values.
        while j < n && (indexed[j].0 - indexed[i].0).abs() < f64::EPSILON {
            j += 1;
        }
        // Average rank for the tie group (1-indexed).
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[indexed[k].1] = avg_rank;
        }
        i = j;
    }
    ranks
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- xorshift64 --------------------------------------------------------

    #[test]
    fn test_xorshift64_nonzero() {
        let mut state = 42u64;
        for _ in 0..100 {
            let v = xorshift64(&mut state);
            assert_ne!(v, 0);
        }
    }

    #[test]
    fn test_xorshift_f64_range() {
        let mut state = 1u64;
        for _ in 0..1000 {
            let v = xorshift_f64(&mut state);
            assert!((0.0..1.0).contains(&v), "v={} out of [0,1)", v);
        }
    }

    #[test]
    fn test_xorshift_f64_zero_state_recovers() {
        // If state becomes 0, it is reset to 1.
        let mut state = 0u64;
        // xorshift64 will first apply XOR ops that may leave state 0,
        // but the guard sets it to 1.
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
    }

    // ---- sample_continuous -------------------------------------------------

    #[test]
    fn test_sample_continuous_normal() {
        let mut state = 12345u64;
        for _ in 0..200 {
            let v = sweep_sample_continuous(0.0, 1.0, false, &mut state)
                .expect("sample_continuous failed");
            assert!((0.0..=1.0).contains(&v), "v={} out of [0,1]", v);
        }
    }

    #[test]
    fn test_sample_continuous_wider_range() {
        let mut state = 99u64;
        for _ in 0..200 {
            let v = sweep_sample_continuous(-10.0, 10.0, false, &mut state)
                .expect("sample_continuous failed");
            assert!((-10.0..=10.0).contains(&v), "v={}", v);
        }
    }

    #[test]
    fn test_sample_continuous_log_scale() {
        let mut state = 7u64;
        for _ in 0..200 {
            let v = sweep_sample_continuous(1e-4, 1e-1, true, &mut state)
                .expect("log scale sample failed");
            assert!((1e-4..=1e-1 + 1e-12).contains(&v), "v={} out of range", v);
        }
    }

    #[test]
    fn test_sample_continuous_log_scale_negative_low_error() {
        let mut state = 1u64;
        let err = sweep_sample_continuous(-1.0, 1.0, true, &mut state);
        assert!(err.is_err());
    }

    #[test]
    fn test_sample_continuous_low_ge_high_error() {
        let mut state = 1u64;
        let err = sweep_sample_continuous(1.0, 0.5, false, &mut state);
        assert!(err.is_err());
    }

    #[test]
    fn test_sample_continuous_low_eq_high_error() {
        let mut state = 1u64;
        let err = sweep_sample_continuous(1.0, 1.0, false, &mut state);
        assert!(err.is_err());
    }

    // ---- sample_discrete ---------------------------------------------------

    #[test]
    fn test_sample_discrete_valid() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut state = 42u64;
        for _ in 0..500 {
            let v = sweep_sample_discrete(&values, &mut state).expect("sample_discrete failed");
            assert!(
                values.contains(&v),
                "sampled value {} not in {:?}",
                v,
                values
            );
        }
    }

    #[test]
    fn test_sample_discrete_single() {
        let values = vec![std::f64::consts::PI];
        let mut state = 1u64;
        let v = sweep_sample_discrete(&values, &mut state).expect("single value");
        assert_eq!(v, std::f64::consts::PI);
    }

    #[test]
    fn test_sample_discrete_empty_error() {
        let mut state = 1u64;
        let err = sweep_sample_discrete(&[], &mut state);
        assert!(matches!(err, Err(SweepError::EmptyDiscrete)));
    }

    // ---- grid_indices -------------------------------------------------------

    #[test]
    fn test_grid_indices_single_dim() {
        let dims = vec![5];
        for i in 0..5 {
            let idx = sweep_grid_indices(&dims, i);
            assert_eq!(idx, vec![i], "trial_idx={}", i);
        }
    }

    #[test]
    fn test_grid_indices_two_dims() {
        // dims=[3,2]: total 6 combos (row-major)
        let dims = vec![3, 2];
        let expected = [
            vec![0, 0],
            vec![0, 1],
            vec![1, 0],
            vec![1, 1],
            vec![2, 0],
            vec![2, 1],
        ];
        for (i, exp) in expected.iter().enumerate() {
            let idx = sweep_grid_indices(&dims, i);
            assert_eq!(&idx, exp, "trial_idx={}", i);
        }
    }

    #[test]
    fn test_grid_indices_three_dims() {
        let dims = vec![2, 3, 2];
        // 12 total combos
        let idx = sweep_grid_indices(&dims, 0);
        assert_eq!(idx, vec![0, 0, 0]);
        let idx_last = sweep_grid_indices(&dims, 11);
        assert_eq!(idx_last, vec![1, 2, 1]);
    }

    #[test]
    fn test_grid_indices_wraps_modulo() {
        // trial_idx beyond total grid size should wrap per dim.
        let dims = vec![2, 2];
        // 4 total; index 4 should wrap to [0,0] again.
        let idx = sweep_grid_indices(&dims, 4);
        assert_eq!(idx, vec![0, 0]);
    }

    #[test]
    fn test_grid_indices_empty_dims() {
        let dims: Vec<usize> = vec![];
        let idx = sweep_grid_indices(&dims, 5);
        assert!(idx.is_empty());
    }

    // ---- param_value_distance / normalized_float_distance -----------------

    #[test]
    fn test_normalized_float_distance_fallback_endpoints_and_log_scale() {
        // No spec: preserves the pre-fix `diff/(diff+1)` behavior exactly.
        let diff = 0.4f64;
        assert!((normalized_float_distance(None, 0.0, 0.4) - diff / (diff + 1.0)).abs() < 1e-12);

        let spec = ParamSpec::Continuous {
            name: "lr".to_string(),
            low: 1e-4,
            high: 1e-2,
            log_scale: false,
        };
        // Range endpoints are exactly 0.0 and 1.0 apart.
        assert!((normalized_float_distance(Some(&spec), 1e-4, 1e-4) - 0.0).abs() < 1e-9);
        assert!((normalized_float_distance(Some(&spec), 1e-4, 1e-2) - 1.0).abs() < 1e-9);

        // log_scale computes distance in log-space (matching how
        // `sweep_sample_continuous` samples it): the *geometric* midpoint of
        // [1e-4, 1e-2] (1e-3) sits at normalized distance 0.5 from 1e-4.
        let log_spec = ParamSpec::Continuous {
            name: "lr".to_string(),
            low: 1e-4,
            high: 1e-2,
            log_scale: true,
        };
        let dist = normalized_float_distance(Some(&log_spec), 1e-4, 1e-3);
        assert!((dist - 0.5).abs() < 1e-6, "got {dist}");
    }

    #[test]
    fn test_normalized_float_distance_small_and_large_range_params_are_comparable() {
        // Regression for the finding: a 1e-4..1e-2 parameter and a 0..1000
        // parameter, each compared at the same *relative* position in their
        // own range, must now yield comparable (not wildly different)
        // normalized distances -- the whole point of normalizing by each
        // parameter's own declared range. Before this fix, the un-normalized
        // `diff/(diff+1)` distance for the same two pairs was wildly
        // different (~0.005 vs ~0.998), dominated by absolute scale.
        let small_spec = ParamSpec::Continuous {
            name: "lr".to_string(),
            low: 1e-4,
            high: 1e-2,
            log_scale: false,
        };
        let large_spec = ParamSpec::Continuous {
            name: "batch_scale".to_string(),
            low: 0.0,
            high: 1000.0,
            log_scale: false,
        };
        // Each pair spans exactly half of its own parameter's range.
        let small_dist = normalized_float_distance(Some(&small_spec), 1e-4, 1e-4 + 0.0099 / 2.0);
        let large_dist = normalized_float_distance(Some(&large_spec), 0.0, 500.0);
        assert!((small_dist - 0.5).abs() < 1e-6, "small_dist={small_dist}");
        assert!((large_dist - 0.5).abs() < 1e-6, "large_dist={large_dist}");

        // Choice/mixed-type handling is unaffected by the spec-normalization
        // change: same/different choices stay 0.0/1.0, mixed types stay 1.0.
        let adam = ParamValue::Choice("adam".into());
        assert_eq!(
            param_value_distance(None, &adam, &ParamValue::Choice("adam".into())),
            0.0
        );
        assert_eq!(
            param_value_distance(None, &adam, &ParamValue::Choice("sgd".into())),
            1.0
        );
        assert_eq!(
            param_value_distance(None, &ParamValue::Float(0.5), &adam),
            1.0
        );
    }

    // ---- surrogate_predict -------------------------------------------------

    #[test]
    fn test_surrogate_predict_no_trials() {
        let params = vec![("lr".to_string(), ParamValue::Float(0.001))];
        let result = sweep_surrogate_predict(&[], &params, &[]);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_surrogate_predict_exact_match() {
        let params = vec![("lr".to_string(), ParamValue::Float(0.001))];
        let trial = SweepTrial {
            id: 0,
            params: params.clone(),
            score: Some(0.5),
        };
        let result = sweep_surrogate_predict(&[trial], &params, &[]);
        // Exact match => distance=0 => weight is huge => should predict ~0.5.
        assert!((result - 0.5).abs() < 1e-6, "result={}", result);
    }

    #[test]
    fn test_surrogate_predict_two_trials() {
        let trial_a = SweepTrial {
            id: 0,
            params: vec![("x".to_string(), ParamValue::Float(0.0))],
            score: Some(1.0),
        };
        let trial_b = SweepTrial {
            id: 1,
            params: vec![("x".to_string(), ParamValue::Float(1.0))],
            score: Some(2.0),
        };
        // Query close to trial_a should predict closer to 1.0.
        let query = vec![("x".to_string(), ParamValue::Float(0.01))];
        // No matching spec for "x" -> falls back to the un-normalized
        // diff/(diff+1) distance, same as before this fix.
        let result = sweep_surrogate_predict(&[trial_a, trial_b], &query, &[]);
        assert!(result < 1.5, "Expected result < 1.5, got {}", result);
    }

    #[test]
    fn test_surrogate_predict_ignores_unscored() {
        let scored = SweepTrial {
            id: 0,
            params: vec![("x".to_string(), ParamValue::Float(0.5))],
            score: Some(3.0),
        };
        let unscored = SweepTrial {
            id: 1,
            params: vec![("x".to_string(), ParamValue::Float(0.5))],
            score: None,
        };
        let query = vec![("x".to_string(), ParamValue::Float(0.5))];
        let result = sweep_surrogate_predict(&[scored, unscored], &query, &[]);
        assert!((result - 3.0).abs() < 1e-6, "result={}", result);
    }

    #[test]
    fn test_surrogate_predict_normalizes_small_range_param_by_spec_bounds() {
        // End-to-end through the public API: `specs` reaches `param_distance`,
        // so a query near the top of a small (1e-4..1e-2) range predicts
        // close to the trial at the top of that range.
        let lr_spec = ParamSpec::Continuous {
            name: "lr".to_string(),
            low: 1e-4,
            high: 1e-2,
            log_scale: false,
        };
        let low = SweepTrial {
            id: 0,
            params: vec![("lr".to_string(), ParamValue::Float(1e-4))],
            score: Some(1.0),
        };
        let high = SweepTrial {
            id: 1,
            params: vec![("lr".to_string(), ParamValue::Float(1e-2))],
            score: Some(2.0),
        };
        let query = vec![("lr".to_string(), ParamValue::Float(9.9e-3))];
        let result = sweep_surrogate_predict(&[low, high], &query, &[lr_spec]);
        assert!((result - 2.0).abs() < 0.05, "got {result}");
    }

    // ---- compute_param_importance ------------------------------------------

    #[test]
    fn test_compute_param_importance_single_param() {
        let specs = vec![ParamSpec::Continuous {
            name: "lr".into(),
            low: 0.0,
            high: 1.0,
            log_scale: false,
        }];
        let trials = vec![
            SweepTrial {
                id: 0,
                params: vec![("lr".into(), ParamValue::Float(0.1))],
                score: Some(0.9),
            },
            SweepTrial {
                id: 1,
                params: vec![("lr".into(), ParamValue::Float(0.5))],
                score: Some(0.5),
            },
            SweepTrial {
                id: 2,
                params: vec![("lr".into(), ParamValue::Float(0.9))],
                score: Some(0.1),
            },
        ];
        let importances = sweep_param_importance(&trials, &specs);
        assert_eq!(importances.len(), 1);
        assert_eq!(importances[0].0, "lr");
        // Single param => importance=1.0.
        assert!((importances[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_param_importance_multi_param_sums_to_one() {
        let specs = vec![
            ParamSpec::Continuous {
                name: "lr".into(),
                low: 0.0,
                high: 1.0,
                log_scale: false,
            },
            ParamSpec::Discrete {
                name: "n_sh".into(),
                values: vec![1.0, 4.0, 9.0],
            },
        ];
        let trials = vec![
            SweepTrial {
                id: 0,
                params: vec![
                    ("lr".into(), ParamValue::Float(0.1)),
                    ("n_sh".into(), ParamValue::Float(1.0)),
                ],
                score: Some(0.8),
            },
            SweepTrial {
                id: 1,
                params: vec![
                    ("lr".into(), ParamValue::Float(0.5)),
                    ("n_sh".into(), ParamValue::Float(4.0)),
                ],
                score: Some(0.5),
            },
            SweepTrial {
                id: 2,
                params: vec![
                    ("lr".into(), ParamValue::Float(0.9)),
                    ("n_sh".into(), ParamValue::Float(9.0)),
                ],
                score: Some(0.2),
            },
        ];
        let importances = sweep_param_importance(&trials, &specs);
        let total: f64 = importances.iter().map(|(_, v)| v).sum();
        assert!((total - 1.0).abs() < 1e-10, "total={}", total);
    }

    #[test]
    fn test_compute_param_importance_fewer_than_two_trials() {
        let specs = vec![ParamSpec::Discrete {
            name: "x".into(),
            values: vec![1.0, 2.0],
        }];
        let trials = vec![SweepTrial {
            id: 0,
            params: vec![("x".into(), ParamValue::Float(1.0))],
            score: Some(0.5),
        }];
        let importances = sweep_param_importance(&trials, &specs);
        assert_eq!(importances[0].1, 0.0);
    }

    #[test]
    fn test_compute_param_importance_categorical() {
        let specs = vec![ParamSpec::Categorical {
            name: "opt".into(),
            choices: vec!["adam".into(), "sgd".into(), "rmsprop".into()],
        }];
        let trials = vec![
            SweepTrial {
                id: 0,
                params: vec![("opt".into(), ParamValue::Choice("adam".into()))],
                score: Some(0.1),
            },
            SweepTrial {
                id: 1,
                params: vec![("opt".into(), ParamValue::Choice("sgd".into()))],
                score: Some(0.5),
            },
            SweepTrial {
                id: 2,
                params: vec![("opt".into(), ParamValue::Choice("rmsprop".into()))],
                score: Some(0.9),
            },
        ];
        let importances = sweep_param_importance(&trials, &specs);
        assert_eq!(importances.len(), 1);
        // Single param => 1.0 (after normalization of a nonzero correlation).
        assert!(importances[0].1 >= 0.0 && importances[0].1 <= 1.0 + 1e-10);
    }

    // ---- hyperband_bracket -------------------------------------------------

    #[test]
    fn test_hyperband_bracket_basic() {
        // max_iter=81, eta=3: s_max=4
        let brackets = hyperband_bracket(81, 3);
        // Should have s_max+1 = 5 rounds.
        assert_eq!(brackets.len(), 5, "brackets={:?}", brackets);
        // All budgets should be > 0.
        for &(n, b) in &brackets {
            assert!(n >= 1, "n_configs must be >= 1");
            assert!(b > 0, "budget must be > 0");
        }
    }

    #[test]
    fn test_hyperband_bracket_matches_published_algorithm_max_iter_81_eta_3() {
        // Regression: pins the exact reference values from Li et al. 2016's
        // published Hyperband algorithm for the textbook max_iter=81, eta=3
        // example. The `n_configs` formula previously omitted the `/ (s+1)`
        // divisor, which yielded 5, 15, 45, 135, 405 instead of these.
        let brackets = hyperband_bracket(81, 3);
        let n_configs: Vec<usize> = brackets.iter().map(|&(n, _)| n).collect();
        assert_eq!(n_configs, vec![5, 8, 15, 34, 81], "brackets={:?}", brackets);
    }

    #[test]
    fn test_hyperband_bracket_eta_2() {
        let brackets = hyperband_bracket(16, 2);
        assert!(!brackets.is_empty());
        // Innermost round (first in list): fewest configs, highest budget.
        let (n0, b0) = brackets[0];
        if brackets.len() > 1 {
            let (n1, b1) = brackets[1];
            // Innermost should have fewer or equal configs.
            let _ = (n0, b0, n1, b1); // verify they exist
        }
    }

    #[test]
    fn test_hyperband_bracket_zero_max_iter() {
        let brackets = hyperband_bracket(0, 3);
        assert!(brackets.is_empty());
    }

    #[test]
    fn test_hyperband_bracket_eta_one_returns_empty() {
        let brackets = hyperband_bracket(10, 1);
        assert!(brackets.is_empty());
    }

    #[test]
    fn test_hyperband_bracket_large() {
        let brackets = hyperband_bracket(243, 3);
        // s_max=5, so 6 rounds.
        assert_eq!(brackets.len(), 6, "brackets={:?}", brackets);
    }

    // ---- ParameterSweep::new -----------------------------------------------

    #[test]
    fn test_sweep_new_empty_specs_error() {
        let config = SweepConfig {
            specs: vec![],
            strategy: SweepStrategy::Random,
            max_trials: 10,
            seed: 1,
            minimize: true,
        };
        let err = ParameterSweep::new(config);
        assert!(matches!(err, Err(SweepError::EmptySpecs)));
    }

    #[test]
    fn test_sweep_new_grid_with_continuous_error() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Continuous {
                name: "lr".into(),
                low: 0.0,
                high: 1.0,
                log_scale: false,
            }],
            strategy: SweepStrategy::Grid,
            max_trials: 10,
            seed: 1,
            minimize: true,
        };
        let err = ParameterSweep::new(config);
        assert!(matches!(
            err,
            Err(SweepError::GridNotSupportedForContinuous)
        ));
    }

    #[test]
    fn test_sweep_new_valid() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "n".into(),
                values: vec![1.0, 2.0],
            }],
            strategy: SweepStrategy::Grid,
            max_trials: 4,
            seed: 1,
            minimize: true,
        };
        assert!(ParameterSweep::new(config).is_ok());
    }

    // ---- ParameterSweep::suggest (Grid) ------------------------------------

    #[test]
    fn test_sweep_suggest_grid_produces_all_combos() {
        let config = SweepConfig {
            specs: vec![
                ParamSpec::Discrete {
                    name: "a".into(),
                    values: vec![1.0, 2.0],
                },
                ParamSpec::Categorical {
                    name: "b".into(),
                    choices: vec!["x".into(), "y".into()],
                },
            ],
            strategy: SweepStrategy::Grid,
            max_trials: 4,
            seed: 0,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        let mut trials = Vec::new();
        for _ in 0..4 {
            trials.push(sweep.suggest().expect("suggest"));
        }
        assert_eq!(trials.len(), 4);
        // All IDs unique.
        let ids: Vec<usize> = trials.iter().map(|t| t.id).collect();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_sweep_suggest_grid_max_trials_reached() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0, 2.0],
            }],
            strategy: SweepStrategy::Grid,
            max_trials: 2,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        sweep.suggest().expect("first");
        sweep.suggest().expect("second");
        let err = sweep.suggest();
        assert!(matches!(err, Err(SweepError::MaxTrialsReached { .. })));
    }

    // ---- ParameterSweep::suggest (Random) ----------------------------------

    #[test]
    fn test_sweep_suggest_random_continuous_in_range() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Continuous {
                name: "lr".into(),
                low: 0.001,
                high: 0.1,
                log_scale: false,
            }],
            strategy: SweepStrategy::Random,
            max_trials: 50,
            seed: 7,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        for _ in 0..50 {
            let trial = sweep.suggest().expect("suggest");
            if let ParamValue::Float(v) = &trial.params[0].1 {
                assert!(*v >= 0.001 && *v <= 0.1, "v={} out of range", v);
            }
        }
    }

    #[test]
    fn test_sweep_suggest_random_log_scale() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Continuous {
                name: "lr".into(),
                low: 1e-5,
                high: 1e-1,
                log_scale: true,
            }],
            strategy: SweepStrategy::Random,
            max_trials: 50,
            seed: 99,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        for _ in 0..50 {
            let trial = sweep.suggest().expect("suggest");
            if let ParamValue::Float(v) = &trial.params[0].1 {
                assert!(*v >= 1e-5 && *v <= 1e-1 + 1e-15, "v={}", v);
            }
        }
    }

    #[test]
    fn test_sweep_suggest_random_categorical() {
        let choices = vec!["adam".into(), "sgd".into()];
        let config = SweepConfig {
            specs: vec![ParamSpec::Categorical {
                name: "opt".into(),
                choices: choices.clone(),
            }],
            strategy: SweepStrategy::Random,
            max_trials: 30,
            seed: 5,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        for _ in 0..30 {
            let trial = sweep.suggest().expect("suggest");
            if let ParamValue::Choice(c) = &trial.params[0].1 {
                assert!(choices.contains(c), "unexpected choice: {}", c);
            }
        }
    }

    // ---- ParameterSweep::suggest (Surrogate) --------------------------------

    #[test]
    fn test_sweep_suggest_surrogate_no_completed_falls_back_to_random() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Continuous {
                name: "x".into(),
                low: 0.0,
                high: 1.0,
                log_scale: false,
            }],
            strategy: SweepStrategy::Surrogate,
            max_trials: 5,
            seed: 3,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        // Without any completed trials, suggest should succeed (random fallback).
        let trial = sweep.suggest().expect("surrogate fallback");
        assert!(trial.score.is_none());
    }

    #[test]
    fn test_sweep_suggest_surrogate_with_completed_trials() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Continuous {
                name: "x".into(),
                low: 0.0,
                high: 1.0,
                log_scale: false,
            }],
            strategy: SweepStrategy::Surrogate,
            max_trials: 10,
            seed: 11,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        // Bootstrap with random trial.
        let t0 = sweep.suggest().expect("t0");
        sweep.report(t0.id, 0.3).expect("report t0");
        // Now surrogate should use t0 as guidance.
        let t1 = sweep.suggest().expect("t1 surrogate");
        assert!(t1.score.is_none());
    }

    // ---- ParameterSweep::report --------------------------------------------

    #[test]
    fn test_report_valid() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 5,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        let trial = sweep.suggest().expect("suggest");
        sweep.report(trial.id, 0.42).expect("report");
        assert_eq!(sweep.trials_completed(), 1);
    }

    #[test]
    fn test_report_unknown_id_error() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 5,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        let err = sweep.report(999, 0.5);
        assert!(matches!(err, Err(SweepError::TrialNotFound(999))));
    }

    // ---- ParameterSweep::best_trial ----------------------------------------

    #[test]
    fn test_best_trial_minimize() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0, 2.0, 3.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 3,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        let t0 = sweep.suggest().expect("t0");
        sweep.report(t0.id, 0.5).expect("r0");
        let t1 = sweep.suggest().expect("t1");
        sweep.report(t1.id, 0.1).expect("r1");
        let t2 = sweep.suggest().expect("t2");
        sweep.report(t2.id, 0.8).expect("r2");
        let best = sweep.best_trial().expect("best");
        assert_eq!(best.id, t1.id);
    }

    #[test]
    fn test_best_trial_maximize() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0, 2.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 2,
            seed: 1,
            minimize: false,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        let t0 = sweep.suggest().expect("t0");
        sweep.report(t0.id, 0.3).expect("r0");
        let t1 = sweep.suggest().expect("t1");
        sweep.report(t1.id, 0.9).expect("r1");
        let best = sweep.best_trial().expect("best");
        assert_eq!(best.id, t1.id);
    }

    #[test]
    fn test_best_trial_no_completed() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 2,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        sweep.suggest().expect("suggest");
        assert!(sweep.best_trial().is_none());
    }

    // ---- ParameterSweep::top_k_trials --------------------------------------

    #[test]
    fn test_top_k_trials_ordering() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 5,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        let scores = [0.5, 0.1, 0.9, 0.3, 0.7];
        for &s in &scores {
            let t = sweep.suggest().expect("suggest");
            sweep.report(t.id, s).expect("report");
        }
        let top3 = sweep.top_k_trials(3);
        assert_eq!(top3.len(), 3);
        let top_scores: Vec<f64> = top3.iter().map(|t| t.score.unwrap()).collect();
        // Should be ascending (minimize=true).
        assert!(top_scores[0] <= top_scores[1]);
        assert!(top_scores[1] <= top_scores[2]);
        assert_eq!(top_scores[0], 0.1);
    }

    #[test]
    fn test_top_k_exceeds_completed() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0, 2.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 2,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        let t0 = sweep.suggest().expect("t0");
        sweep.report(t0.id, 0.4).expect("r0");
        // Request 10 but only 1 completed.
        let top = sweep.top_k_trials(10);
        assert_eq!(top.len(), 1);
    }

    // ---- ParameterSweep::is_done -------------------------------------------

    #[test]
    fn test_is_done() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0, 2.0],
            }],
            strategy: SweepStrategy::Grid,
            max_trials: 2,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        assert!(!sweep.is_done());
        sweep.suggest().expect("t0");
        assert!(!sweep.is_done());
        sweep.suggest().expect("t1");
        assert!(sweep.is_done());
    }

    // ---- format_trial / format_sweep_summary --------------------------------

    #[test]
    fn test_format_trial_pending() {
        let trial = SweepTrial {
            id: 5,
            params: vec![("lr".into(), ParamValue::Float(0.001234))],
            score: None,
        };
        let s = format_sweep_trial(&trial);
        assert!(s.contains("Trial #5"), "s={}", s);
        assert!(s.contains("lr="), "s={}", s);
        assert!(s.contains("pending"), "s={}", s);
    }

    #[test]
    fn test_format_trial_with_score() {
        let trial = SweepTrial {
            id: 0,
            params: vec![("lr".into(), ParamValue::Float(0.01))],
            score: Some(0.123456),
        };
        let s = format_sweep_trial(&trial);
        assert!(s.contains("score=0.123456"), "s={}", s);
    }

    #[test]
    fn test_format_trial_categorical() {
        let trial = SweepTrial {
            id: 2,
            params: vec![("opt".into(), ParamValue::Choice("adam".into()))],
            score: Some(0.5),
        };
        let s = format_sweep_trial(&trial);
        assert!(s.contains("opt=adam"), "s={}", s);
    }

    #[test]
    fn test_format_sweep_summary_no_trials() {
        let summary = SweepSummary {
            total_trials: 0,
            completed_trials: 0,
            best_score: None,
            worst_score: None,
            mean_score: None,
            std_score: None,
            param_importances: vec![("lr".into(), 1.0)],
        };
        let s = format_sweep_summary(&summary);
        assert!(s.contains("0/0"), "s={}", s);
    }

    #[test]
    fn test_format_sweep_summary_with_data() {
        let summary = SweepSummary {
            total_trials: 10,
            completed_trials: 8,
            best_score: Some(0.1),
            worst_score: Some(0.9),
            mean_score: Some(0.5),
            std_score: Some(0.2),
            param_importances: vec![("lr".into(), 0.7), ("n_sh".into(), 0.3)],
        };
        let s = format_sweep_summary(&summary);
        assert!(s.contains("8/10"), "s={}", s);
        assert!(s.contains("0.100000"), "s={}", s);
        assert!(s.contains("lr"), "s={}", s);
        assert!(s.contains("n_sh"), "s={}", s);
    }

    // ---- ParameterSweep::summary -------------------------------------------

    #[test]
    fn test_summary_empty() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 5,
            seed: 1,
            minimize: true,
        };
        let sweep = ParameterSweep::new(config).expect("new");
        let s = sweep.summary();
        assert_eq!(s.total_trials, 0);
        assert_eq!(s.completed_trials, 0);
        assert!(s.best_score.is_none());
    }

    #[test]
    fn test_summary_all_same_score() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0, 2.0, 3.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 3,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        for _ in 0..3 {
            let t = sweep.suggest().expect("suggest");
            sweep.report(t.id, 0.5).expect("report");
        }
        let s = sweep.summary();
        assert_eq!(s.best_score, Some(0.5));
        assert_eq!(s.worst_score, Some(0.5));
        assert_eq!(s.mean_score, Some(0.5));
        assert_eq!(s.std_score, Some(0.0));
    }

    // ---- Edge cases --------------------------------------------------------

    #[test]
    fn test_max_trials_zero_immediately_done() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 0,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        assert!(sweep.is_done());
        let err = sweep.suggest();
        assert!(matches!(err, Err(SweepError::MaxTrialsReached { .. })));
    }

    #[test]
    fn test_trials_completed_count() {
        let config = SweepConfig {
            specs: vec![ParamSpec::Discrete {
                name: "x".into(),
                values: vec![1.0, 2.0, 3.0],
            }],
            strategy: SweepStrategy::Random,
            max_trials: 3,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        assert_eq!(sweep.trials_completed(), 0);
        let t0 = sweep.suggest().expect("t0");
        assert_eq!(sweep.trials_completed(), 0);
        sweep.report(t0.id, 0.5).expect("r0");
        assert_eq!(sweep.trials_completed(), 1);
        let t1 = sweep.suggest().expect("t1");
        sweep.report(t1.id, 0.3).expect("r1");
        assert_eq!(sweep.trials_completed(), 2);
    }

    #[test]
    fn test_param_value_display_float() {
        let v = ParamValue::Float(std::f64::consts::PI);
        let s = format!("{}", v);
        assert!(s.contains("3.141593"), "s={}", s);
    }

    #[test]
    fn test_param_value_display_int() {
        let v = ParamValue::Int(42);
        assert_eq!(format!("{}", v), "42");
    }

    #[test]
    fn test_param_value_display_choice() {
        let v = ParamValue::Choice("adam".into());
        assert_eq!(format!("{}", v), "adam");
    }

    #[test]
    fn test_surrogate_multiple_params() {
        let config = SweepConfig {
            specs: vec![
                ParamSpec::Continuous {
                    name: "lr".into(),
                    low: 0.0001,
                    high: 0.01,
                    log_scale: false,
                },
                ParamSpec::Discrete {
                    name: "n_sh".into(),
                    values: vec![1.0, 4.0, 9.0],
                },
            ],
            strategy: SweepStrategy::Surrogate,
            max_trials: 15,
            seed: 42,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        // Run a few bootstrap trials.
        for _ in 0..3 {
            let t = sweep.suggest().expect("suggest");
            sweep.report(t.id, 0.5).expect("report");
        }
        // Surrogate suggestions should now succeed.
        for _ in 0..5 {
            sweep.suggest().expect("surrogate suggest");
        }
        assert_eq!(sweep.trials_completed(), 3);
    }

    #[test]
    fn test_grid_full_product_count() {
        // 2 * 3 * 2 = 12 total combos.
        let config = SweepConfig {
            specs: vec![
                ParamSpec::Discrete {
                    name: "a".into(),
                    values: vec![1.0, 2.0],
                },
                ParamSpec::Discrete {
                    name: "b".into(),
                    values: vec![10.0, 20.0, 30.0],
                },
                ParamSpec::Discrete {
                    name: "c".into(),
                    values: vec![0.1, 0.9],
                },
            ],
            strategy: SweepStrategy::Grid,
            max_trials: 12,
            seed: 1,
            minimize: true,
        };
        let mut sweep = ParameterSweep::new(config).expect("new");
        let mut all_params = Vec::new();
        for _ in 0..12 {
            let t = sweep.suggest().expect("suggest");
            all_params.push(t.params.clone());
        }
        // All 12 combos should be unique.
        let unique: std::collections::HashSet<String> = all_params
            .iter()
            .map(|p| {
                format!(
                    "{:?}",
                    p.iter().map(|(_, v)| format!("{}", v)).collect::<Vec<_>>()
                )
            })
            .collect();
        assert_eq!(
            unique.len(),
            12,
            "Expected 12 unique combos, got {}",
            unique.len()
        );
    }
}
