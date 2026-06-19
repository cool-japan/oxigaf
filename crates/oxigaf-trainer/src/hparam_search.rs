//! Hyperparameter search infrastructure for training configuration exploration.
//!
//! Provides grid, random, Latin-hypercube, and simple Bayesian search strategies
//! over user-defined search spaces.  All randomness uses a deterministic
//! xorshift64 PRNG — no external `rand` dependency.
//!
//! # Quick start
//! ```no_run
//! use oxigaf_trainer::hparam_search::{
//!     HparamRange, HparamSearcher, SearchSpace, SearchStrategy,
//! };
//!
//! let space = SearchSpace::new()
//!     .add("lr", HparamRange::Continuous { lo: 1e-4, hi: 1e-2, log_scale: true })
//!     .add("batch", HparamRange::Discrete { lo: 8, hi: 64 });
//!
//! let mut searcher = HparamSearcher::new(space, SearchStrategy::Random, 42);
//! let trial = searcher.suggest_next().expect("search space is non-empty");
//! println!("{}", trial.format_params());
//! ```

use std::collections::{HashMap, VecDeque};
use std::fmt::Write as FmtWrite;

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Xorshift64 PRNG
// ─────────────────────────────────────────────────────────────────────────────

/// Deterministic 64-bit xorshift PRNG.
pub(crate) struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0xcafe_babe_dead_beef
        } else {
            seed
        })
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform sample in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in [0, n).  Returns 0 when n == 0.
    fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next() % n as u64) as usize
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SearchError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the hyperparameter search subsystem.
#[derive(Debug, Error)]
pub enum SearchError {
    /// The search space contains no parameters.
    #[error("search space is empty")]
    EmptySearchSpace,

    /// The grid search has been fully exhausted.
    #[error("grid search exhausted")]
    GridExhausted,

    /// No trial with the given id exists in the history.
    #[error("trial {0} not found")]
    TrialNotFound(usize),
}

// ─────────────────────────────────────────────────────────────────────────────
// HparamRange
// ─────────────────────────────────────────────────────────────────────────────

/// The range / domain of a single hyperparameter.
#[derive(Debug, Clone)]
pub enum HparamRange {
    /// Continuous range [lo, hi].  When `log_scale` is true, sampling and
    /// grid-stepping operate in log space.
    Continuous { lo: f64, hi: f64, log_scale: bool },

    /// Discrete integer range [lo, hi] inclusive.
    Discrete { lo: i64, hi: i64 },

    /// Categorical choices.  The sampled value is the index cast to f64.
    Categorical(Vec<String>),

    /// Fixed value — not searched.
    Fixed(f64),
}

impl HparamRange {
    /// Number of discrete options.  Returns 0 for Continuous (infinite), 1 for Fixed.
    pub fn num_options(&self) -> usize {
        match self {
            HparamRange::Continuous { .. } => 0,
            HparamRange::Discrete { lo, hi } => {
                if hi >= lo {
                    (hi - lo + 1) as usize
                } else {
                    1
                }
            }
            HparamRange::Categorical(v) => v.len(),
            HparamRange::Fixed(_) => 1,
        }
    }

    /// Sample a random value.  For Categorical the return value is the index as f64.
    pub(crate) fn sample(&self, rng: &mut Xorshift64) -> f64 {
        match self {
            HparamRange::Continuous { lo, hi, log_scale } => {
                let u = rng.next_f64();
                if *log_scale && *lo > 0.0 && *hi > 0.0 {
                    let lo_log = lo.ln();
                    let hi_log = hi.ln();
                    (lo_log + u * (hi_log - lo_log)).exp()
                } else {
                    lo + u * (hi - lo)
                }
            }
            HparamRange::Discrete { lo, hi } => {
                let n = if hi >= lo { (hi - lo + 1) as usize } else { 1 };
                let idx = rng.next_usize(n);
                (lo + idx as i64) as f64
            }
            HparamRange::Categorical(v) => {
                let idx = if v.is_empty() {
                    0
                } else {
                    rng.next_usize(v.len())
                };
                idx as f64
            }
            HparamRange::Fixed(v) => *v,
        }
    }

    /// Return the i-th evenly-spaced value for grid/LHS mapping.
    ///
    /// `grid_size` is used only for `Continuous` (number of grid points).
    pub fn grid_sample(&self, idx: usize, grid_size: usize) -> f64 {
        match self {
            HparamRange::Continuous { lo, hi, log_scale } => {
                let n = grid_size.max(1);
                if n == 1 {
                    return *lo;
                }
                let t = idx.min(n - 1) as f64 / (n - 1) as f64;
                if *log_scale && *lo > 0.0 && *hi > 0.0 {
                    let lo_log = lo.ln();
                    let hi_log = hi.ln();
                    (lo_log + t * (hi_log - lo_log)).exp()
                } else {
                    lo + t * (hi - lo)
                }
            }
            HparamRange::Discrete { lo, hi } => {
                let max_idx = if hi >= lo { (hi - lo) as usize } else { 0 };
                let i = idx.min(max_idx);
                (lo + i as i64) as f64
            }
            HparamRange::Categorical(v) => {
                if v.is_empty() {
                    0.0
                } else {
                    (idx % v.len()) as f64
                }
            }
            HparamRange::Fixed(v) => *v,
        }
    }

    /// Map a unit value u ∈ [0, 1) to the parameter's native space.
    /// Used internally for Latin hypercube stratum sampling.
    fn unit_to_value(&self, u: f64, _grid_size: usize) -> f64 {
        match self {
            HparamRange::Continuous { lo, hi, log_scale } => {
                if *log_scale && *lo > 0.0 && *hi > 0.0 {
                    let lo_log = lo.ln();
                    let hi_log = hi.ln();
                    (lo_log + u * (hi_log - lo_log)).exp()
                } else {
                    lo + u * (hi - lo)
                }
            }
            HparamRange::Discrete { lo, hi } => {
                let n = if hi >= lo { (hi - lo + 1) as usize } else { 1 };
                let raw = (u * n as f64).floor() as i64;
                let clamped = raw.max(0).min((n as i64) - 1);
                (lo + clamped) as f64
            }
            HparamRange::Categorical(v) => {
                let n = v.len().max(1);
                let idx = (u * n as f64).floor() as usize;
                idx.min(n - 1) as f64
            }
            HparamRange::Fixed(v) => *v,
        }
    }

    /// Normalize a native value to [0, 1] for distance computation.
    fn normalize_value(&self, v: f64) -> f64 {
        match self {
            HparamRange::Continuous { lo, hi, log_scale } => {
                let range = hi - lo;
                if range == 0.0 {
                    return 0.0;
                }
                if *log_scale && *lo > 0.0 && *hi > 0.0 {
                    let lo_log = lo.ln();
                    let hi_log = hi.ln();
                    let log_range = hi_log - lo_log;
                    if log_range == 0.0 {
                        0.0
                    } else {
                        (v.max(*lo).ln() - lo_log) / log_range
                    }
                } else {
                    ((v - lo) / range).clamp(0.0, 1.0)
                }
            }
            HparamRange::Discrete { lo, hi } => {
                let n = if hi >= lo { (hi - lo) as f64 } else { 1.0 };
                if n == 0.0 {
                    0.0
                } else {
                    ((v - *lo as f64) / n).clamp(0.0, 1.0)
                }
            }
            HparamRange::Categorical(vs) => {
                let n = vs.len().max(1) as f64;
                (v / n).clamp(0.0, 1.0)
            }
            HparamRange::Fixed(_) => 0.0,
        }
    }

    /// Format a sampled f64 value as a human-readable string.
    pub fn format_value(&self, v: f64) -> String {
        match self {
            HparamRange::Continuous { .. } => {
                if v.abs() < 1e-3 || v.abs() >= 1e5 {
                    format!("{:.3e}", v)
                } else {
                    format!("{:.6}", v)
                }
            }
            HparamRange::Discrete { .. } => format!("{}", v as i64),
            HparamRange::Categorical(cats) => {
                let idx = (v as usize).min(cats.len().saturating_sub(1));
                cats.get(idx).cloned().unwrap_or_else(|| format!("{}", idx))
            }
            HparamRange::Fixed(fv) => format!("{:.6}", fv),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HparamDef / SearchSpace
// ─────────────────────────────────────────────────────────────────────────────

/// Definition of a single hyperparameter: name + range.
#[derive(Debug, Clone)]
pub struct HparamDef {
    pub name: String,
    pub range: HparamRange,
}

/// The full set of hyperparameters to search over.
#[derive(Debug, Clone)]
pub struct SearchSpace {
    pub params: Vec<HparamDef>,
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchSpace {
    /// Create an empty search space.
    pub fn new() -> Self {
        Self { params: Vec::new() }
    }

    /// Add a parameter definition (builder pattern).
    pub fn add(mut self, name: impl Into<String>, range: HparamRange) -> Self {
        self.params.push(HparamDef {
            name: name.into(),
            range,
        });
        self
    }

    /// Number of parameters in this space.
    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    /// Total number of grid combinations.
    ///
    /// For Continuous params, `grid_size` points are used.
    /// Discrete and Categorical use their actual option counts.
    /// Fixed contributes 1.
    pub fn total_grid_combinations(&self, grid_size: usize) -> usize {
        self.params.iter().fold(1usize, |acc, p| {
            let n = match &p.range {
                HparamRange::Continuous { .. } => grid_size.max(1),
                HparamRange::Fixed(_) => 1,
                r => r.num_options().max(1),
            };
            acc.saturating_mul(n)
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trial
// ─────────────────────────────────────────────────────────────────────────────

/// A single hyperparameter configuration to evaluate.
#[derive(Debug, Clone)]
pub struct Trial {
    /// Unique monotonically increasing identifier.
    pub id: usize,
    /// Parameter name → sampled value mapping.
    pub params: HashMap<String, f64>,
    /// Score assigned after evaluation (higher = better).  `None` if pending.
    pub score: Option<f32>,
    /// Training step at which this trial was recorded / scored.
    pub step: usize,
}

impl Trial {
    /// Get a parameter value as f64.
    pub fn get_f64(&self, name: &str) -> Option<f64> {
        self.params.get(name).copied()
    }

    /// Get a parameter value as i64 (truncates towards zero).
    pub fn get_i64(&self, name: &str) -> Option<i64> {
        self.get_f64(name).map(|v| v as i64)
    }

    /// Get a parameter value as usize.  Returns `None` if the i64 is negative.
    pub fn get_usize(&self, name: &str) -> Option<usize> {
        self.get_i64(name)
            .and_then(|v| if v >= 0 { Some(v as usize) } else { None })
    }

    /// Get the index of a Categorical parameter (as usize).
    pub fn get_str_index(&self, name: &str) -> Option<usize> {
        self.get_f64(name).map(|v| v as usize)
    }

    /// Format all parameters as a comma-separated string.
    pub fn format_params(&self) -> String {
        let mut pairs: Vec<(&String, &f64)> = self.params.iter().collect();
        pairs.sort_by_key(|(k, _)| k.as_str());
        let mut out = String::new();
        for (i, (k, v)) in pairs.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "{}={:.6}", k, v);
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TrialHistory
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulates all completed trials and tracks the current best.
pub struct TrialHistory {
    pub trials: Vec<Trial>,
    /// Index into `trials` of the best-scoring trial seen so far.
    pub best_trial: Option<usize>,
}

impl Default for TrialHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl TrialHistory {
    /// Create an empty history.
    pub fn new() -> Self {
        Self {
            trials: Vec::new(),
            best_trial: None,
        }
    }

    /// Append a trial.
    pub fn record(&mut self, trial: Trial) {
        self.trials.push(trial);
    }

    /// Scan all scored trials and update `best_trial`.
    pub fn update_best(&mut self) {
        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = None;
        for (idx, trial) in self.trials.iter().enumerate() {
            if let Some(s) = trial.score {
                if s > best_score {
                    best_score = s;
                    best_idx = Some(idx);
                }
            }
        }
        self.best_trial = best_idx;
    }

    /// Return a reference to the best trial, if any scored trial exists.
    pub fn best(&self) -> Option<&Trial> {
        self.best_trial.and_then(|i| self.trials.get(i))
    }

    /// All scored trials (score is Some).
    pub fn scored_trials(&self) -> Vec<&Trial> {
        self.trials.iter().filter(|t| t.score.is_some()).collect()
    }

    /// Scored trials sorted descending by score.
    pub fn sorted_by_score(&self) -> Vec<&Trial> {
        let mut scored = self.scored_trials();
        scored.sort_by(|a, b| {
            b.score
                .unwrap_or(f32::NEG_INFINITY)
                .partial_cmp(&a.score.unwrap_or(f32::NEG_INFINITY))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SearchStrategy / AcquisitionFunction
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy used to suggest the next trial.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SearchStrategy {
    /// Exhaustive grid over all parameters.
    Grid { grid_size: usize },
    /// Independent random sampling for each parameter.
    Random,
    /// Stratified random sampling (Latin Hypercube).
    LatinHypercube,
    /// UCB-based with simple GP surrogate (K-NN mean/std).
    BayesianSimple,
}

/// Acquisition function used by `BayesianSimple`.
#[derive(Debug, Clone, Copy)]
pub enum AcquisitionFunction {
    /// Upper Confidence Bound: μ + κ·σ.
    Ucb { kappa: f32 },
    /// Expected Improvement over current best, with exploration parameter ξ.
    ExpectedImprovement { xi: f32 },
}

impl Default for AcquisitionFunction {
    fn default() -> Self {
        AcquisitionFunction::Ucb { kappa: 2.0 }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Normal distribution helpers for EI
// ─────────────────────────────────────────────────────────────────────────────

/// Logistic approximation to the standard-normal CDF: Φ(x) ≈ 1/(1+exp(-1.7015·x)).
fn normal_cdf(x: f64) -> f64 {
    1.0 / (1.0 + (-1.7015 * x).exp())
}

/// Standard-normal PDF: φ(x) = exp(-x²/2) / √(2π).
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// HparamSearcher
// ─────────────────────────────────────────────────────────────────────────────

/// Orchestrates hyperparameter search over a [`SearchSpace`].
pub struct HparamSearcher {
    pub space: SearchSpace,
    pub strategy: SearchStrategy,
    pub history: TrialHistory,
    pub acquisition: AcquisitionFunction,
    rng: Xorshift64,
    next_trial_id: usize,
    /// Current position in the grid (Grid strategy only).
    grid_index: usize,
    /// Pre-shuffled stratum queues, one per parameter (LHS strategy).
    lhs_strata: Vec<VecDeque<f64>>,
    lhs_batch_size: usize,
}

impl HparamSearcher {
    /// Create a new searcher.
    pub fn new(space: SearchSpace, strategy: SearchStrategy, seed: u64) -> Self {
        let num_params = space.num_params();
        Self {
            space,
            strategy,
            history: TrialHistory::new(),
            acquisition: AcquisitionFunction::default(),
            rng: Xorshift64::new(seed),
            next_trial_id: 0,
            grid_index: 0,
            lhs_strata: vec![VecDeque::new(); num_params],
            lhs_batch_size: 20,
        }
    }

    /// Override the acquisition function (builder pattern).
    pub fn with_acquisition(mut self, acq: AcquisitionFunction) -> Self {
        self.acquisition = acq;
        self
    }

    /// Suggest the next trial to evaluate.
    pub fn suggest_next(&mut self) -> Result<Trial, SearchError> {
        if self.space.num_params() == 0 {
            return Err(SearchError::EmptySearchSpace);
        }

        let params = match self.strategy {
            SearchStrategy::Grid { grid_size } => self.suggest_grid(grid_size)?,
            SearchStrategy::Random => self.suggest_random(),
            SearchStrategy::LatinHypercube => self.suggest_lhs(),
            SearchStrategy::BayesianSimple => self.suggest_bayesian(),
        };

        let trial = Trial {
            id: self.next_trial_id,
            params,
            score: None,
            step: 0,
        };
        self.next_trial_id += 1;
        Ok(trial)
    }

    /// Record the evaluation result for a trial.
    ///
    /// Finds the trial in history and updates its score, or appends a new
    /// placeholder record if the trial has not been stored yet.
    pub fn record_result(&mut self, trial_id: usize, score: f32, step: usize) {
        // Try to update an existing history entry.
        for trial in &mut self.history.trials {
            if trial.id == trial_id {
                trial.score = Some(score);
                trial.step = step;
                self.history.update_best();
                return;
            }
        }
        // No existing entry — store a placeholder so best tracking works.
        let placeholder = Trial {
            id: trial_id,
            params: HashMap::new(),
            score: Some(score),
            step,
        };
        self.history.record(placeholder);
        self.history.update_best();
    }

    /// Get the best trial seen so far.
    pub fn best_trial(&self) -> Option<&Trial> {
        self.history.best()
    }

    /// Format a human-readable summary of search progress.
    pub fn format_summary(&self) -> String {
        let mut s = String::new();
        let total = self.next_trial_id;
        let scored = self.history.scored_trials().len();
        let _ = write!(s, "HparamSearch: strategy={:?}", self.strategy);
        let _ = write!(s, ", trials={total}, scored={scored}");

        if let Some(best) = self.history.best() {
            let best_score = best.score.unwrap_or(0.0);
            let _ = write!(s, ", best_score={:.4}", best_score);
            if !best.params.is_empty() {
                let _ = write!(s, " [{}]", best.format_params());
            }
        } else {
            let _ = write!(s, ", no scored trials yet");
        }
        s
    }

    /// Whether all grid points have been generated.
    pub fn is_grid_exhausted(&self) -> bool {
        match self.strategy {
            SearchStrategy::Grid { grid_size } => {
                self.grid_index >= self.space.total_grid_combinations(grid_size)
            }
            _ => false,
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    fn suggest_grid(&mut self, grid_size: usize) -> Result<HashMap<String, f64>, SearchError> {
        let total = self.space.total_grid_combinations(grid_size);
        if self.grid_index >= total {
            return Err(SearchError::GridExhausted);
        }

        // Mixed-radix decomposition: walk params left-to-right.
        let mut remaining = self.grid_index;
        let mut params = HashMap::new();
        for p in &self.space.params {
            let size = match &p.range {
                HparamRange::Continuous { .. } => grid_size.max(1),
                HparamRange::Fixed(_) => 1,
                r => r.num_options().max(1),
            };
            let local_idx = remaining % size;
            remaining /= size;
            let value = p.range.grid_sample(local_idx, grid_size);
            params.insert(p.name.clone(), value);
        }

        self.grid_index += 1;
        Ok(params)
    }

    fn suggest_random(&mut self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        // Borrow params immutably first to collect definitions.
        let defs: Vec<(String, HparamRange)> = self
            .space
            .params
            .iter()
            .map(|p| (p.name.clone(), p.range.clone()))
            .collect();
        for (name, range) in defs {
            let value = range.sample(&mut self.rng);
            params.insert(name, value);
        }
        params
    }

    /// Refill LHS strata for all params, then pop the front value.
    fn suggest_lhs(&mut self) -> HashMap<String, f64> {
        let grid_size = 10usize; // default strata count for Continuous
        let batch = self.lhs_batch_size;
        let num_params = self.space.num_params();

        // Ensure strata queues are sized correctly.
        if self.lhs_strata.len() != num_params {
            self.lhs_strata = vec![VecDeque::new(); num_params];
        }

        // Check if any queue is empty; if so, regenerate all.
        let needs_refill = self.lhs_strata.iter().any(|q| q.is_empty());
        if needs_refill {
            self.lhs_strata = vec![VecDeque::new(); num_params];

            let defs: Vec<HparamRange> =
                self.space.params.iter().map(|p| p.range.clone()).collect();

            for (param_idx, range) in defs.iter().enumerate() {
                // Number of strata M.
                let m = batch.max(1);
                // Generate stratum indices [0, m) and Fisher-Yates shuffle.
                let mut indices: Vec<usize> = (0..m).collect();
                for i in (1..m).rev() {
                    let j = self.rng.next_usize(i + 1);
                    indices.swap(i, j);
                }
                // Map each stratum k to a uniform sample within [k/M, (k+1)/M).
                for &k in &indices {
                    let lo_u = k as f64 / m as f64;
                    let hi_u = (k + 1) as f64 / m as f64;
                    let u = lo_u + self.rng.next_f64() * (hi_u - lo_u);
                    let value = range.unit_to_value(u, grid_size);
                    self.lhs_strata[param_idx].push_back(value);
                }
            }
        }

        // Pop the front value from each queue.
        let mut params = HashMap::new();
        let names: Vec<String> = self.space.params.iter().map(|p| p.name.clone()).collect();
        for (param_idx, name) in names.iter().enumerate() {
            let value = self.lhs_strata[param_idx].pop_front().unwrap_or_else(|| {
                // Fallback: shouldn't happen, but sample randomly.
                let range = self.space.params[param_idx].range.clone();
                range.sample(&mut self.rng)
            });
            params.insert(name.clone(), value);
        }
        params
    }

    fn suggest_bayesian(&mut self) -> HashMap<String, f64> {
        let scored = self.history.scored_trials();
        let min_scored = 2 * self.space.num_params().max(1);

        if scored.len() < min_scored {
            // Fall back to random until we have enough data.
            return self.suggest_random();
        }

        // Sample 50 candidates randomly and score via UCB/EI.
        const NUM_CANDIDATES: usize = 50;
        const K_NEIGHBORS: usize = 3;

        let best_score = self
            .history
            .best()
            .and_then(|t| t.score)
            .unwrap_or(f32::NEG_INFINITY) as f64;

        let mut best_acq = f64::NEG_INFINITY;
        let mut best_params: Option<HashMap<String, f64>> = None;

        // Pre-collect scored trials to avoid repeated allocation.
        let scored_owned: Vec<Trial> = scored.iter().map(|t| (*t).clone()).collect();

        let defs: Vec<(String, HparamRange)> = self
            .space
            .params
            .iter()
            .map(|p| (p.name.clone(), p.range.clone()))
            .collect();

        for _ in 0..NUM_CANDIDATES {
            // Sample candidate.
            let mut cand_params = HashMap::new();
            for (name, range) in &defs {
                cand_params.insert(name.clone(), range.sample(&mut self.rng));
            }

            // Compute normalized L2 distance from candidate to each scored trial.
            let mut distances: Vec<(f64, f32)> = scored_owned
                .iter()
                .map(|t| {
                    let dist = defs
                        .iter()
                        .map(|(name, range)| {
                            let v_cand = cand_params.get(name).copied().unwrap_or(0.0);
                            let v_trial = t.params.get(name).copied().unwrap_or(0.0);
                            let n_cand = range.normalize_value(v_cand);
                            let n_trial = range.normalize_value(v_trial);
                            let diff = n_cand - n_trial;
                            diff * diff
                        })
                        .sum::<f64>()
                        .sqrt();
                    (dist, t.score.unwrap_or(0.0))
                })
                .collect();

            // Sort by distance; take K nearest.
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let k = K_NEIGHBORS.min(distances.len());
            let neighbors = &distances[..k];

            // Compute mean and std of K neighbor scores.
            let scores_f64: Vec<f64> = neighbors.iter().map(|(_, s)| *s as f64).collect();
            let mean = scores_f64.iter().sum::<f64>() / k as f64;
            let variance = scores_f64
                .iter()
                .map(|s| (s - mean) * (s - mean))
                .sum::<f64>()
                / k as f64;
            let sigma = variance.sqrt().max(1e-8);

            let acq = match self.acquisition {
                AcquisitionFunction::Ucb { kappa } => mean + kappa as f64 * sigma,
                AcquisitionFunction::ExpectedImprovement { xi } => {
                    let delta = mean - best_score;
                    let z = (delta - xi as f64) / sigma;
                    (delta - xi as f64) * normal_cdf(z) + sigma * normal_pdf(z)
                }
            };

            if acq > best_acq {
                best_acq = acq;
                best_params = Some(cand_params);
            }
        }

        best_params.unwrap_or_else(|| self.suggest_random())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ── PRNG ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_not_zero() {
        let mut rng = Xorshift64::new(1);
        for _ in 0..1000 {
            assert_ne!(rng.next(), 0);
        }
    }

    #[test]
    fn test_xorshift64_f64_in_range() {
        let mut rng = Xorshift64::new(42);
        for _ in 0..10_000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "f64 out of range: {v}");
        }
    }

    // ── HparamRange sampling ─────────────────────────────────────────────────

    #[test]
    fn test_hparam_range_continuous_sample_in_bounds() {
        let range = HparamRange::Continuous {
            lo: 0.001,
            hi: 0.1,
            log_scale: false,
        };
        let mut rng = Xorshift64::new(7);
        for _ in 0..1000 {
            let v = range.sample(&mut rng);
            assert!((0.001..=0.1).contains(&v), "out of bounds: {v}");
        }
    }

    #[test]
    fn test_hparam_range_discrete_sample_in_bounds() {
        let range = HparamRange::Discrete { lo: 4, hi: 64 };
        let mut rng = Xorshift64::new(13);
        for _ in 0..1000 {
            let v = range.sample(&mut rng);
            let i = v as i64;
            assert!((4..=64).contains(&i), "out of range: {i}");
        }
    }

    #[test]
    fn test_hparam_range_categorical_sample_valid_index() {
        let range = HparamRange::Categorical(vec![
            "adam".to_string(),
            "sgd".to_string(),
            "rmsprop".to_string(),
        ]);
        let mut rng = Xorshift64::new(99);
        for _ in 0..1000 {
            let v = range.sample(&mut rng);
            let idx = v as usize;
            assert!(idx < 3, "invalid index: {idx}");
        }
    }

    #[test]
    fn test_hparam_range_log_scale_positive() {
        let range = HparamRange::Continuous {
            lo: 1e-5,
            hi: 1.0,
            log_scale: true,
        };
        let mut rng = Xorshift64::new(2025);
        for _ in 0..1000 {
            let v = range.sample(&mut rng);
            assert!(v > 0.0, "log-scale sample must be positive: {v}");
            assert!((1e-5..=1.0).contains(&v), "log-scale out of range: {v}");
        }
    }

    // ── SearchSpace ──────────────────────────────────────────────────────────

    #[test]
    fn test_search_space_total_grid_combinations() {
        let space = SearchSpace::new()
            .add(
                "lr",
                HparamRange::Continuous {
                    lo: 1e-4,
                    hi: 1e-1,
                    log_scale: true,
                },
            )
            .add("batch", HparamRange::Discrete { lo: 8, hi: 16 }) // 9 options
            .add(
                "opt",
                HparamRange::Categorical(vec!["adam".to_string(), "sgd".to_string()]),
            ) // 2
            .add("fixed", HparamRange::Fixed(0.9)); // 1

        // grid_size=5 → 5 * 9 * 2 * 1 = 90
        assert_eq!(space.total_grid_combinations(5), 90);
    }

    // ── Grid strategy ────────────────────────────────────────────────────────

    #[test]
    fn test_searcher_grid_exhaustion() {
        let space = SearchSpace::new()
            .add("a", HparamRange::Discrete { lo: 0, hi: 1 }) // 2
            .add("b", HparamRange::Discrete { lo: 0, hi: 2 }); // 3 → total=6

        let mut searcher = HparamSearcher::new(space, SearchStrategy::Grid { grid_size: 5 }, 0);
        // Should generate exactly 6 trials.
        for _ in 0..6 {
            assert!(searcher.suggest_next().is_ok());
        }
        let err = searcher.suggest_next().unwrap_err();
        assert!(matches!(err, SearchError::GridExhausted));
    }

    // ── Random strategy ──────────────────────────────────────────────────────

    #[test]
    fn test_searcher_random_suggest_count() {
        let space = SearchSpace::new()
            .add(
                "lr",
                HparamRange::Continuous {
                    lo: 1e-4,
                    hi: 1e-1,
                    log_scale: true,
                },
            )
            .add("batch", HparamRange::Discrete { lo: 8, hi: 64 });

        let mut searcher = HparamSearcher::new(space, SearchStrategy::Random, 0);
        for i in 0..50 {
            let trial = searcher.suggest_next().unwrap();
            assert_eq!(trial.id, i);
            assert_eq!(trial.params.len(), 2);
        }
    }

    // ── LHS strategy ─────────────────────────────────────────────────────────

    #[test]
    fn test_searcher_lhs_coverage() {
        // With 1 Continuous param and 20 LHS strata, each [0,1/20), [1/20, 2/20), …
        // should be covered exactly once in the first 20 suggestions.
        let space = SearchSpace::new().add(
            "lr",
            HparamRange::Continuous {
                lo: 0.0,
                hi: 1.0,
                log_scale: false,
            },
        );

        let mut searcher = HparamSearcher::new(space, SearchStrategy::LatinHypercube, 7);
        let mut values: Vec<f64> = Vec::new();
        for _ in 0..20 {
            let t = searcher.suggest_next().unwrap();
            let v = t.get_f64("lr").unwrap();
            values.push(v);
        }

        // Check that all 20 strata [k/20, (k+1)/20) are covered exactly once.
        let m = 20usize;
        let mut covered = vec![false; m];
        for &v in &values {
            let stratum = (v * m as f64).floor() as usize;
            let stratum = stratum.min(m - 1);
            assert!(!covered[stratum], "stratum {stratum} covered twice");
            covered[stratum] = true;
        }
        assert!(covered.iter().all(|&c| c), "not all strata covered");
    }

    // ── Record / best ────────────────────────────────────────────────────────

    #[test]
    fn test_record_result_and_best() {
        let space = SearchSpace::new().add(
            "lr",
            HparamRange::Continuous {
                lo: 1e-4,
                hi: 1e-1,
                log_scale: false,
            },
        );
        let mut searcher = HparamSearcher::new(space, SearchStrategy::Random, 0);

        let t0 = searcher.suggest_next().unwrap();
        let t1 = searcher.suggest_next().unwrap();
        let t2 = searcher.suggest_next().unwrap();

        searcher.history.record(t0.clone());
        searcher.history.record(t1.clone());
        searcher.history.record(t2.clone());

        searcher.record_result(t0.id, 0.5, 10);
        searcher.record_result(t1.id, 0.9, 20);
        searcher.record_result(t2.id, 0.3, 30);

        let best = searcher.best_trial().unwrap();
        assert_eq!(best.id, t1.id);
        assert!((best.score.unwrap() - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_history_sorted_by_score() {
        let mut history = TrialHistory::new();
        for (id, score) in [(0, 0.2f32), (1, 0.8), (2, 0.5), (3, 0.95), (4, 0.1)] {
            history.record(Trial {
                id,
                params: HashMap::new(),
                score: Some(score),
                step: 0,
            });
        }
        history.update_best();
        let sorted = history.sorted_by_score();
        let scores: Vec<f32> = sorted.iter().map(|t| t.score.unwrap()).collect();
        for w in scores.windows(2) {
            assert!(w[0] >= w[1], "not sorted descending");
        }
        assert!((scores[0] - 0.95).abs() < 1e-6);
    }

    // ── Bayesian fallback ────────────────────────────────────────────────────

    #[test]
    fn test_bayesian_fallback_to_random_insufficient() {
        // With 2 params, we need at least 4 scored trials before Bayesian kicks in.
        let space = SearchSpace::new()
            .add(
                "lr",
                HparamRange::Continuous {
                    lo: 1e-4,
                    hi: 1e-1,
                    log_scale: false,
                },
            )
            .add("batch", HparamRange::Discrete { lo: 4, hi: 32 });

        let mut searcher = HparamSearcher::new(space, SearchStrategy::BayesianSimple, 42);

        // Only 1 scored trial — should still produce a trial (random fallback).
        let t = searcher.suggest_next().unwrap();
        searcher.history.record(t.clone());
        searcher.record_result(t.id, 0.5, 1);

        let next = searcher.suggest_next();
        assert!(next.is_ok(), "Bayesian fallback should succeed");
    }

    #[test]
    fn test_bayesian_suggests_candidate() {
        let space = SearchSpace::new()
            .add(
                "lr",
                HparamRange::Continuous {
                    lo: 1e-4,
                    hi: 1e-1,
                    log_scale: false,
                },
            )
            .add("batch", HparamRange::Discrete { lo: 4, hi: 32 });

        let mut searcher = HparamSearcher::new(space, SearchStrategy::BayesianSimple, 99);

        // Feed 5 scored trials (>= 2*2=4).
        for score in [0.3f32, 0.6, 0.5, 0.7, 0.4] {
            let t = searcher.suggest_next().unwrap();
            searcher.history.record(t.clone());
            searcher.record_result(t.id, score, 1);
        }

        // Now Bayesian should engage.
        let trial = searcher.suggest_next().unwrap();
        assert_eq!(trial.params.len(), 2);
    }

    // ── Summary ──────────────────────────────────────────────────────────────

    #[test]
    fn test_format_summary() {
        let space = SearchSpace::new().add(
            "lr",
            HparamRange::Continuous {
                lo: 1e-4,
                hi: 1e-1,
                log_scale: false,
            },
        );
        let mut searcher = HparamSearcher::new(space, SearchStrategy::Random, 0);

        let summary_before = searcher.format_summary();
        assert!(summary_before.contains("scored=0"));

        let t = searcher.suggest_next().unwrap();
        searcher.history.record(t.clone());
        searcher.record_result(t.id, 0.77, 5);

        let summary_after = searcher.format_summary();
        assert!(
            summary_after.contains("best_score=0.7700"),
            "summary: {summary_after}"
        );
    }

    // ── Trial formatting ─────────────────────────────────────────────────────

    #[test]
    fn test_trial_format_params() {
        let mut params = HashMap::new();
        params.insert("lr".to_string(), 0.001);
        params.insert("batch".to_string(), 32.0);
        let trial = Trial {
            id: 0,
            params,
            score: None,
            step: 0,
        };
        let s = trial.format_params();
        // Keys are sorted alphabetically: batch then lr.
        assert!(s.starts_with("batch="), "got: {s}");
        assert!(s.contains("lr="), "got: {s}");
    }

    // ── grid_sample edge cases ───────────────────────────────────────────────

    #[test]
    fn test_grid_sample_fixed() {
        let range = HparamRange::Fixed(3.1);
        assert!((range.grid_sample(0, 10) - 3.1).abs() < 1e-12);
        assert!((range.grid_sample(99, 10) - 3.1).abs() < 1e-12);
    }

    #[test]
    fn test_grid_sample_continuous_bounds() {
        let range = HparamRange::Continuous {
            lo: 0.0,
            hi: 1.0,
            log_scale: false,
        };
        assert!((range.grid_sample(0, 5) - 0.0).abs() < 1e-12);
        assert!((range.grid_sample(4, 5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_format_value_categorical() {
        let range = HparamRange::Categorical(vec!["adam".to_string(), "sgd".to_string()]);
        assert_eq!(range.format_value(0.0), "adam");
        assert_eq!(range.format_value(1.0), "sgd");
    }
}
