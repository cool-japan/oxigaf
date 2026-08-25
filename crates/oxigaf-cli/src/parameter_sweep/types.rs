//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use thiserror::Error;

use super::functions::{
    sample_params_random, sweep_grid_indices, sweep_param_importance, sweep_surrogate_predict,
};

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
/// Manages the state of a hyperparameter sweep.
#[derive(Debug)]
pub struct ParameterSweep {
    pub(super) config: SweepConfig,
    pub(super) trials: Vec<SweepTrial>,
    pub(super) rng_state: u64,
    pub(super) trial_counter: usize,
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
    pub(super) fn suggest_grid(&mut self) -> Result<SweepTrial, SweepError> {
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
    pub(super) fn suggest_random(&mut self) -> Result<SweepTrial, SweepError> {
        let params = sample_params_random(&self.config.specs, &mut self.rng_state)?;
        Ok(SweepTrial {
            id: self.trial_counter,
            params,
            score: None,
        })
    }
    pub(super) fn suggest_surrogate(&mut self) -> Result<SweepTrial, SweepError> {
        let completed: Vec<&SweepTrial> =
            self.trials.iter().filter(|t| t.score.is_some()).collect();
        if completed.is_empty() {
            return self.suggest_random();
        }
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
    pub(super) fn grid_size(&self) -> Option<usize> {
        match self {
            ParamSpec::Continuous { .. } => None,
            ParamSpec::Discrete { values, .. } => Some(values.len()),
            ParamSpec::Categorical { choices, .. } => Some(choices.len()),
        }
    }
}
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
