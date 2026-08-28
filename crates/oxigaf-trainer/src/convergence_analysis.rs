//! Convergence analysis utilities for the OxiGAF training pipeline.
//!
//! Provides tools for analyzing training convergence — detecting plateaus,
//! computing convergence rates, classifying the current training phase, and
//! generating convergence reports.
//!
//! # Key types
//! - [`ConvergenceAnalyzer`]: stateful analyzer that ingests (step, loss) pairs
//! - [`ConvergenceConfig`]: tuning parameters for phase detection
//! - [`ConvergenceStats`]: snapshot of current convergence state
//! - [`ConvergenceReport`]: full diagnostic report with recommendation

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the convergence analysis subsystem.
#[derive(Debug, Error, PartialEq)]
pub enum ConvergenceError {
    #[error("Insufficient history: need at least {needed} points, got {got}")]
    InsufficientHistory { needed: usize, got: usize },

    #[error("Window size {window} is larger than history length {history_len}")]
    WindowTooLarge { window: usize, history_len: usize },

    #[error("Invalid window size {size}: must be >= 2")]
    InvalidWindow { size: usize },

    #[error("Empty loss history")]
    EmptyHistory,
}

// ---------------------------------------------------------------------------
// ConvergencePhase
// ---------------------------------------------------------------------------

/// Phase of training convergence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvergencePhase {
    /// Fewer than `min_history` points collected; analysis not yet meaningful.
    Initializing,
    /// Loss dropping fast (relative improvement rate > `rapid_threshold`).
    RapidDecline,
    /// Loss dropping slowly (rate in `(plateau_threshold, rapid_threshold)`).
    SlowImprovement,
    /// Very slow improvement (rate ≤ `plateau_threshold`).
    Plateau,
    /// Loss oscillating (high standard deviation of loss changes).
    Oscillating,
    /// Loss trending upward.
    Diverging,
}

// ---------------------------------------------------------------------------
// ConvergenceConfig
// ---------------------------------------------------------------------------

/// Configuration for convergence analysis.
#[derive(Debug, Clone)]
pub struct ConvergenceConfig {
    /// Rolling window for analysis (default: 100).
    pub window_size: usize,
    /// Minimum points before analysis (default: 50).
    pub min_history: usize,
    /// Rate below this is a plateau (default: 1e-4).
    pub plateau_threshold: f32,
    /// Rate above this is rapid decline (default: 1e-2).
    pub rapid_threshold: f32,
    /// Rate above this (upward) is diverging (default: 0.1).
    pub diverge_threshold: f32,
    /// Window for oscillation detection (default: 20).
    pub oscillation_window: usize,
    /// Decay for EMA smoothing (default: 0.95).
    pub ema_decay: f32,
    /// Threshold on the *relative* oscillation score — `std(successive loss
    /// diffs) / mean(|loss|)` over the oscillation window — above which the
    /// phase is classified [`ConvergencePhase::Oscillating`] (default: 0.3,
    /// i.e. step-to-step swings averaging >30% of the loss magnitude).
    ///
    /// This is intentionally a *separate* knob from `plateau_threshold`
    /// rather than derived from it: `plateau_threshold` calibrates a
    /// relative-improvement-per-window rate (typically ~1e-4, i.e. a
    /// fraction of a percent), while oscillation noise on a realistic loss
    /// curve routinely sits in the few-percent range even for healthy,
    /// non-oscillating runs — comparing the two on the same tiny scale
    /// would make rule 2 fire on essentially every call.
    pub oscillation_threshold: f32,
    /// Optional cap on the number of `(step, loss)` points retained in
    /// [`ConvergenceAnalyzer`]'s internal history buffer. `None` (default)
    /// keeps every point ever recorded, which [`detect_phase_transitions`]
    /// needs to see the whole run. `Some(n)` evicts the oldest points once
    /// more than `n` are held, bounding memory for very long runs at the
    /// cost of [`detect_phase_transitions`] only being able to see the most
    /// recent `n` points. [`ConvergenceAnalyzer::best_loss`] and
    /// [`ConvergenceAnalyzer::history_len`] are unaffected by eviction: both
    /// track the true full-run value regardless of this setting.
    pub max_history: Option<usize>,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            min_history: 50,
            plateau_threshold: 1e-4,
            rapid_threshold: 1e-2,
            diverge_threshold: 0.1,
            oscillation_window: 20,
            ema_decay: 0.95,
            oscillation_threshold: 0.3,
            max_history: None,
        }
    }
}

// ---------------------------------------------------------------------------
// ConvergenceStats
// ---------------------------------------------------------------------------

/// Convergence statistics computed over a rolling window.
#[derive(Debug, Clone)]
pub struct ConvergenceStats {
    /// Detected convergence phase.
    pub phase: ConvergencePhase,
    /// Raw loss value at the latest point.
    pub current_loss: f32,
    /// EMA-smoothed loss.
    pub smoothed_loss: f32,
    /// Linear regression slope over the window (negative = improving).
    pub loss_slope: f32,
    /// `(loss_start - loss_end) / |loss_start|`, negative when diverging.
    pub relative_improvement: f32,
    /// Standard deviation of loss-to-loss differences — higher = more oscillating.
    pub oscillation_score: f32,
    /// Minimum loss in the window.
    pub window_min: f32,
    /// Maximum loss in the window.
    pub window_max: f32,
    /// Number of points analysed.
    pub n_points: usize,
}

// ---------------------------------------------------------------------------
// ConvergenceReport
// ---------------------------------------------------------------------------

/// Full convergence diagnostic report.
#[derive(Debug, Clone)]
pub struct ConvergenceReport {
    /// Current convergence phase.
    pub phase: ConvergencePhase,
    /// Latest raw loss.
    pub current_loss: f32,
    /// Best (lowest) loss seen in history.
    pub best_loss: f32,
    /// Number of steps in history.
    pub steps_analyzed: usize,
    /// Estimated additional steps to reach target loss, if converging.
    pub estimated_steps_remaining: Option<usize>,
    /// Human-readable recommendation.
    pub recommendation: String,
}

// ---------------------------------------------------------------------------
// ConvergenceAnalyzer
// ---------------------------------------------------------------------------

/// Stateful convergence analyzer.
///
/// Feed training steps via [`update`](ConvergenceAnalyzer::update), then call
/// [`analyze`](ConvergenceAnalyzer::analyze) or
/// [`phase`](ConvergenceAnalyzer::phase) to inspect the current state.
pub struct ConvergenceAnalyzer {
    /// Tuning parameters.
    pub config: ConvergenceConfig,
    /// `(step, loss)` history. Unbounded by default (needed so
    /// [`detect_phase_transitions`] can see the whole run); capped to
    /// `config.max_history` most-recent points when that is `Some`. Use
    /// [`Self::best_loss`] / [`Self::history_len`] rather than scanning this
    /// directly — both remain correct for the *entire* run even when
    /// eviction is active.
    history: Vec<(usize, f32)>,
    /// Current EMA loss.
    ema_loss: f32,
    /// Total calls to `update`, independent of any `history` eviction.
    n_updates: usize,
    /// Lowest loss seen across every `update` call so far, tracked
    /// independently of `history` so it stays correct even after older
    /// entries (potentially including the one that set this minimum) have
    /// been evicted under `config.max_history`.
    best_loss_seen: Option<f32>,
}

impl ConvergenceAnalyzer {
    /// Create a new analyzer with the given configuration.
    pub fn new(config: ConvergenceConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            ema_loss: f32::MAX,
            n_updates: 0,
            best_loss_seen: None,
        }
    }

    /// Record a new training step.
    ///
    /// The EMA is seeded on the first call so there is no cold-start bias.
    pub fn update(&mut self, step: usize, loss: f32) {
        if self.n_updates == 0 {
            self.ema_loss = loss;
        } else {
            let d = self.config.ema_decay;
            self.ema_loss = d * self.ema_loss + (1.0 - d) * loss;
        }
        self.best_loss_seen = Some(match self.best_loss_seen {
            Some(best) => best.min(loss),
            None => loss,
        });
        self.history.push((step, loss));
        self.n_updates += 1;

        // Bound memory for very long runs when the caller has opted in.
        // Evicting from the *front* keeps the most recent points, which is
        // what `analyze()` (rolling `window_size`) and
        // `detect_phase_transitions()` (chunked from the front, so eviction
        // just means older chunks are no longer visible) both want.
        //
        // The effective cap is never allowed below `min_history`: `analyze`
        // requires at least `min_history` points to ever leave
        // `InsufficientHistory`, so a caller-misconfigured
        // `max_history < min_history` would otherwise permanently lock the
        // analyzer in `ConvergencePhase::Initializing` no matter how many
        // steps are recorded.
        if let Some(cap) = self.config.max_history {
            let effective_cap = cap.max(self.config.min_history);
            if self.history.len() > effective_cap {
                let excess = self.history.len() - effective_cap;
                self.history.drain(0..excess);
            }
        }
    }

    /// Compute [`ConvergenceStats`] over the most recent `window_size` points.
    ///
    /// Returns [`ConvergenceError::InsufficientHistory`] when fewer than
    /// `min_history` points have been recorded.
    pub fn analyze(&self) -> Result<ConvergenceStats, ConvergenceError> {
        let total = self.history.len();

        if total < self.config.min_history {
            return Err(ConvergenceError::InsufficientHistory {
                needed: self.config.min_history,
                got: total,
            });
        }

        let window = self.config.window_size.min(total);
        let window_pairs = &self.history[total - window..];
        let windowed: Vec<f32> = window_pairs.iter().map(|(_, l)| *l).collect();

        let current_loss = *windowed.last().ok_or(ConvergenceError::EmptyHistory)?;
        let window_min = windowed.iter().copied().fold(f32::INFINITY, f32::min);
        let window_max = windowed.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Regress against the *recorded step values*, not the positional
        // index within the window: callers that log every `log_interval`
        // steps (rather than every single step) would otherwise get a slope
        // in "loss per recorded point", silently off by a factor of
        // `log_interval` from the "loss per training step" the field is
        // documented (and used by `estimate_steps_to_convergence`) to mean.
        let loss_slope = compute_loss_slope_xy(window_pairs)?;
        let relative_improvement = compute_relative_improvement(&windowed)?;

        // For oscillation we use the smaller oscillation_window clamped to available data.
        let osc_win = self.config.oscillation_window.min(window);
        let osc_slice = &windowed[windowed.len() - osc_win..];
        let oscillation_score = compute_oscillation_score(osc_slice)?;
        // Scale for turning the absolute `oscillation_score` (loss units)
        // into a relative quantity comparable to `oscillation_threshold`.
        let osc_scale = osc_slice.iter().sum::<f32>() / osc_slice.len() as f32;

        let phase = detect_convergence_phase(
            loss_slope,
            relative_improvement,
            oscillation_score,
            osc_scale,
            &self.config,
        );

        Ok(ConvergenceStats {
            phase,
            current_loss,
            smoothed_loss: self.ema_loss,
            loss_slope,
            relative_improvement,
            oscillation_score,
            window_min,
            window_max,
            n_points: window,
        })
    }

    /// Quick phase query without building the full [`ConvergenceStats`].
    ///
    /// Returns [`ConvergencePhase::Initializing`] when insufficient history is
    /// available, instead of propagating an error.
    pub fn phase(&self) -> ConvergencePhase {
        if self.history.len() < self.config.min_history {
            return ConvergencePhase::Initializing;
        }
        // Reuse analyze; silently fall back to Initializing on any error.
        self.analyze()
            .map(|s| s.phase)
            .unwrap_or(ConvergencePhase::Initializing)
    }

    /// Clear all history and reset the EMA.
    pub fn reset(&mut self) {
        self.history.clear();
        self.ema_loss = f32::MAX;
        self.n_updates = 0;
        self.best_loss_seen = None;
    }

    /// Best (lowest) loss seen across every `update()` call so far.
    ///
    /// Tracked incrementally (O(1) per call) rather than scanned from
    /// `history`, so this remains the true whole-run minimum even when
    /// `config.max_history` has evicted the history entry that originally
    /// set it.
    pub fn best_loss(&self) -> Option<f32> {
        self.best_loss_seen
    }

    /// Total number of `update()` calls made so far.
    ///
    /// This is the true cumulative step count for the whole run, not the
    /// number of points currently retained in the (possibly
    /// `config.max_history`-capped) internal buffer.
    pub fn history_len(&self) -> usize {
        self.n_updates
    }
}

impl Default for ConvergenceAnalyzer {
    fn default() -> Self {
        Self::new(ConvergenceConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Compute the linear regression slope over a loss sequence.
///
/// Uses ordinary least squares: `slope = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)`
/// where x = 0, 1, …, n−1.
///
/// Negative slope means loss is improving; positive slope means it is worsening.
///
/// # Errors
/// - [`ConvergenceError::EmptyHistory`] if `losses` is empty.
/// - [`ConvergenceError::InvalidWindow`] if `losses` has fewer than 2 points.
pub fn compute_loss_slope(losses: &[f32]) -> Result<f32, ConvergenceError> {
    let n = losses.len();
    if n == 0 {
        return Err(ConvergenceError::EmptyHistory);
    }
    if n < 2 {
        return Err(ConvergenceError::InvalidWindow { size: n });
    }
    let nf = n as f64;
    let sum_x: f64 = (0..n).map(|i| i as f64).sum();
    let sum_y: f64 = losses.iter().map(|&v| v as f64).sum();
    let sum_xy: f64 = losses
        .iter()
        .enumerate()
        .map(|(i, &y)| i as f64 * y as f64)
        .sum();
    let sum_x2: f64 = (0..n).map(|i| (i as f64) * (i as f64)).sum();

    let denom = nf * sum_x2 - sum_x * sum_x;
    if denom.abs() < f64::EPSILON {
        return Ok(0.0);
    }
    Ok(((nf * sum_xy - sum_x * sum_y) / denom) as f32)
}

/// Compute the linear regression slope of `(step, loss)` pairs against the
/// actual recorded `step` values, rather than their positional index.
///
/// Identical OLS formula to [`compute_loss_slope`], but with `x` taken from
/// `points[i].0` instead of `i`. This matters whenever consecutive history
/// entries are not exactly one training step apart (e.g. a caller that only
/// records every `log_interval` steps): [`compute_loss_slope`] would then
/// return "loss change per *recorded point*", silently off by a factor of
/// the recording interval from "loss change per *training step*" —
/// [`ConvergenceStats::loss_slope`] and [`estimate_steps_to_convergence`]
/// are both specified (and consumed) in the latter unit.
///
/// # Errors
/// - [`ConvergenceError::EmptyHistory`] if `points` is empty.
/// - [`ConvergenceError::InvalidWindow`] if `points` has fewer than 2 points.
pub fn compute_loss_slope_xy(points: &[(usize, f32)]) -> Result<f32, ConvergenceError> {
    let n = points.len();
    if n == 0 {
        return Err(ConvergenceError::EmptyHistory);
    }
    if n < 2 {
        return Err(ConvergenceError::InvalidWindow { size: n });
    }
    let nf = n as f64;
    let sum_x: f64 = points.iter().map(|&(s, _)| s as f64).sum();
    let sum_y: f64 = points.iter().map(|&(_, l)| l as f64).sum();
    let sum_xy: f64 = points.iter().map(|&(s, l)| s as f64 * l as f64).sum();
    let sum_x2: f64 = points.iter().map(|&(s, _)| (s as f64) * (s as f64)).sum();

    let denom = nf * sum_x2 - sum_x * sum_x;
    if denom.abs() < f64::EPSILON {
        // All points share the same step (or n == 1, handled above) — no
        // well-defined per-step rate; matches `compute_loss_slope`'s own
        // degenerate-denominator fallback.
        return Ok(0.0);
    }
    Ok(((nf * sum_xy - sum_x * sum_y) / denom) as f32)
}

/// Compute relative improvement across a loss sequence.
///
/// Returns `(losses[0] - losses[last]) / |losses[0]|`, clamped to a minimum
/// denominator of `1e-8` to avoid division by zero.  A positive value means
/// the loss has decreased (improved); negative means it has increased.
///
/// # Errors
/// Returns [`ConvergenceError::EmptyHistory`] if `losses` is empty.
pub fn compute_relative_improvement(losses: &[f32]) -> Result<f32, ConvergenceError> {
    if losses.is_empty() {
        return Err(ConvergenceError::EmptyHistory);
    }
    let first = losses[0];
    let last = *losses.last().ok_or(ConvergenceError::EmptyHistory)?;
    let denom = first.abs().max(1e-8);
    Ok((first - last) / denom)
}

/// Compute the oscillation score (std dev of successive differences).
///
/// High values indicate the loss is swinging up and down rather than trending.
///
/// # Errors
/// - [`ConvergenceError::EmptyHistory`] if `losses` is empty.
/// - [`ConvergenceError::InvalidWindow`] if fewer than 2 points.
pub fn compute_oscillation_score(losses: &[f32]) -> Result<f32, ConvergenceError> {
    let n = losses.len();
    if n == 0 {
        return Err(ConvergenceError::EmptyHistory);
    }
    if n < 2 {
        return Err(ConvergenceError::InvalidWindow { size: n });
    }
    let diffs: Vec<f32> = losses.windows(2).map(|w| w[1] - w[0]).collect();
    let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
    let variance = diffs.iter().map(|d| (d - mean) * (d - mean)).sum::<f32>() / diffs.len() as f32;
    Ok(variance.sqrt())
}

/// Apply exponential moving average smoothing to a loss sequence.
///
/// The first output equals the first input (no cold-start bias).
/// `output[i] = decay * output[i-1] + (1 - decay) * losses[i]`
///
/// Returns a `Vec` of the same length as `losses`.  Returns an empty `Vec` for
/// empty input without error.
pub fn ema_smooth_losses(losses: &[f32], decay: f32) -> Vec<f32> {
    if losses.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(losses.len());
    out.push(losses[0]);
    for &val in &losses[1..] {
        let prev = *out.last().unwrap_or(&losses[0]);
        out.push(decay * prev + (1.0 - decay) * val);
    }
    out
}

/// Classify the convergence phase from pre-computed statistics.
///
/// Decision order (first matching rule wins):
/// 1. `slope > diverge_threshold` → [`ConvergencePhase::Diverging`]
/// 2. `oscillation_score / loss_scale > oscillation_threshold` → [`ConvergencePhase::Oscillating`]
/// 3. `|relative_improvement| < plateau_threshold` → [`ConvergencePhase::Plateau`]
/// 4. `relative_improvement > rapid_threshold` → [`ConvergencePhase::RapidDecline`]
/// 5. Otherwise → [`ConvergencePhase::SlowImprovement`]
///
/// `oscillation_score` (from [`compute_oscillation_score`]) is a standard
/// deviation expressed in absolute *loss units*, while every threshold in
/// `config` besides `oscillation_threshold` is a dimensionless relative
/// rate. `loss_scale` (typically the window's mean loss) converts the score
/// to the same relative scale before comparing it against
/// `config.oscillation_threshold`, so the two are actually comparable —
/// comparing the raw absolute score directly against a relative-rate
/// threshold such as `plateau_threshold` would make rule 2 fire on
/// essentially every realistically-scaled loss curve regardless of whether
/// it is actually oscillating.
pub fn detect_convergence_phase(
    slope: f32,
    relative_improvement: f32,
    oscillation_score: f32,
    loss_scale: f32,
    config: &ConvergenceConfig,
) -> ConvergencePhase {
    if slope > config.diverge_threshold {
        return ConvergencePhase::Diverging;
    }
    let relative_oscillation = oscillation_score / loss_scale.abs().max(1e-8);
    if relative_oscillation > config.oscillation_threshold {
        return ConvergencePhase::Oscillating;
    }
    if relative_improvement.abs() < config.plateau_threshold {
        return ConvergencePhase::Plateau;
    }
    if relative_improvement > config.rapid_threshold {
        return ConvergencePhase::RapidDecline;
    }
    ConvergencePhase::SlowImprovement
}

/// Estimate how many more steps are needed to reach `target_loss`.
///
/// Uses the current regression slope to extrapolate.
///
/// Returns `None` if:
/// - `analyze()` fails (too few data points)
/// - The model is not converging (`slope >= 0`)
/// - Current loss is already at or below `target_loss`
pub fn estimate_steps_to_convergence(
    analyzer: &ConvergenceAnalyzer,
    target_loss: f32,
) -> Option<usize> {
    let stats = analyzer.analyze().ok()?;
    if stats.loss_slope >= 0.0 {
        return None;
    }
    if stats.current_loss <= target_loss {
        return None;
    }
    let steps = (stats.current_loss - target_loss) / (-stats.loss_slope);
    if steps.is_finite() && steps >= 0.0 {
        Some(steps.ceil() as usize)
    } else {
        None
    }
}

/// Compute the `percentile`-th value (0.0 … 100.0) of the given loss sequence.
///
/// Uses linear interpolation between adjacent sorted values.
///
/// # Errors
/// Returns [`ConvergenceError::EmptyHistory`] if `losses` is empty.
pub fn compute_loss_percentile(losses: &[f32], percentile: f32) -> Result<f32, ConvergenceError> {
    if losses.is_empty() {
        return Err(ConvergenceError::EmptyHistory);
    }
    let mut sorted = losses.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let pct = percentile.clamp(0.0, 100.0);
    let float_idx = pct / 100.0 * (n - 1) as f32;
    let lo = float_idx.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = float_idx - lo as f32;
    Ok(sorted[lo] * (1.0 - frac) + sorted[hi] * frac)
}

/// Compute the rolling improvement rate over a sliding window.
///
/// For each position `i >= window`:
/// `rate[i] = (losses[i - window] - losses[i]) / |losses[i - window]|`
///
/// Returns a `Vec` of length `n - window`.
///
/// # Errors
/// - [`ConvergenceError::EmptyHistory`] if `losses` is empty.
/// - [`ConvergenceError::InvalidWindow`] if `window < 2`.
/// - [`ConvergenceError::WindowTooLarge`] if `window >= losses.len()`.
pub fn loss_improvement_rate(losses: &[f32], window: usize) -> Result<Vec<f32>, ConvergenceError> {
    if losses.is_empty() {
        return Err(ConvergenceError::EmptyHistory);
    }
    if window < 2 {
        return Err(ConvergenceError::InvalidWindow { size: window });
    }
    if window >= losses.len() {
        return Err(ConvergenceError::WindowTooLarge {
            window,
            history_len: losses.len(),
        });
    }
    let rates = losses[window..]
        .iter()
        .zip(losses.iter())
        .map(|(&current, &past)| {
            let denom = past.abs().max(1e-8);
            (past - current) / denom
        })
        .collect();
    Ok(rates)
}

/// Detect phase transitions in the full training history.
///
/// Splits the history into non-overlapping chunks of `config.window_size` and
/// runs phase detection on each chunk.  Returns `(step_index, new_phase)` pairs
/// for every chunk where the phase differs from the previous chunk.
///
/// The `step_index` is the *first step index* of the new-phase chunk.
pub fn detect_phase_transitions(analyzer: &ConvergenceAnalyzer) -> Vec<(usize, ConvergencePhase)> {
    let history = &analyzer.history;
    if history.len() < 2 {
        return Vec::new();
    }
    let ws = analyzer.config.window_size.max(2);
    let mut transitions = Vec::new();
    let mut prev_phase: Option<ConvergencePhase> = None;

    let chunks: Vec<&[(usize, f32)]> = history.chunks(ws).collect();
    for chunk in &chunks {
        let losses: Vec<f32> = chunk.iter().map(|(_, l)| *l).collect();
        if losses.len() < 2 {
            continue;
        }
        // Step-aware, matching `analyze()`: regressing against positional
        // index instead of the real `step` values would silently be wrong
        // whenever chunk entries are not exactly one training step apart.
        let slope = match compute_loss_slope_xy(chunk) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let rel_imp = match compute_relative_improvement(&losses) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let osc = match compute_oscillation_score(&losses) {
            Ok(o) => o,
            Err(_) => continue,
        };
        let osc_scale = losses.iter().sum::<f32>() / losses.len() as f32;
        let phase = detect_convergence_phase(slope, rel_imp, osc, osc_scale, &analyzer.config);
        let step_idx = chunk[0].0;

        match prev_phase {
            None => {
                transitions.push((step_idx, phase));
            }
            Some(p) if p != phase => {
                transitions.push((step_idx, phase));
            }
            _ => {}
        }
        prev_phase = Some(phase);
    }
    transitions
}

/// Format a human-readable convergence report from [`ConvergenceStats`].
pub fn format_convergence_report(stats: &ConvergenceStats) -> String {
    let phase_name = match stats.phase {
        ConvergencePhase::Initializing => "Initializing",
        ConvergencePhase::RapidDecline => "RapidDecline",
        ConvergencePhase::SlowImprovement => "SlowImprovement",
        ConvergencePhase::Plateau => "Plateau",
        ConvergencePhase::Oscillating => "Oscillating",
        ConvergencePhase::Diverging => "Diverging",
    };
    format!(
        "Phase: {phase_name} | loss={:.4e} (smoothed={:.4e}) | slope={:.4e} | \
         rel_improvement={:.4e} | oscillation={:.4e} | window=[{:.4e}, {:.4e}] | n={}",
        stats.current_loss,
        stats.smoothed_loss,
        stats.loss_slope,
        stats.relative_improvement,
        stats.oscillation_score,
        stats.window_min,
        stats.window_max,
        stats.n_points,
    )
}

/// Build a full [`ConvergenceReport`] from the analyzer state.
///
/// # Errors
/// Returns [`ConvergenceError::InsufficientHistory`] if the analyzer has too
/// few data points.
pub fn generate_convergence_report(
    analyzer: &ConvergenceAnalyzer,
    target_loss: f32,
) -> Result<ConvergenceReport, ConvergenceError> {
    let stats = analyzer.analyze()?;

    let best_loss = analyzer.best_loss().unwrap_or(stats.current_loss);

    let estimated_steps_remaining = estimate_steps_to_convergence(analyzer, target_loss);

    let recommendation = build_recommendation(&stats, estimated_steps_remaining);

    Ok(ConvergenceReport {
        phase: stats.phase,
        current_loss: stats.current_loss,
        best_loss,
        steps_analyzed: analyzer.history_len(),
        estimated_steps_remaining,
        recommendation,
    })
}

/// Internal helper: produce a short training recommendation from current stats.
fn build_recommendation(stats: &ConvergenceStats, steps_remaining: Option<usize>) -> String {
    match stats.phase {
        ConvergencePhase::Initializing => {
            "Collecting initial history — check back after more steps.".to_string()
        }
        ConvergencePhase::RapidDecline => {
            "Training is in rapid decline — maintain current configuration.".to_string()
        }
        ConvergencePhase::SlowImprovement => match steps_remaining {
            Some(n) => format!("Slow but steady improvement. Estimated ~{n} more steps to target."),
            None => "Slow but steady improvement. Continue training.".to_string(),
        },
        ConvergencePhase::Plateau => {
            "Loss has plateaued. Consider reducing the learning rate or adjusting densification."
                .to_string()
        }
        ConvergencePhase::Oscillating => {
            "Loss is oscillating. Consider reducing the learning rate or increasing gradient \
             clipping."
                .to_string()
        }
        ConvergencePhase::Diverging => {
            "Loss is diverging! Reduce learning rate immediately or roll back to a checkpoint."
                .to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // compute_loss_slope
    // ------------------------------------------------------------------

    #[test]
    fn slope_monotone_decrease_is_negative() {
        // Losses: 10, 9, 8, 7, 6 — strictly decreasing
        let losses: Vec<f32> = (0..5).map(|i| 10.0 - i as f32).collect();
        let slope = compute_loss_slope(&losses).expect("slope computation failed");
        assert!(slope < 0.0, "expected negative slope, got {slope}");
    }

    #[test]
    fn slope_monotone_increase_is_positive() {
        let losses: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let slope = compute_loss_slope(&losses).expect("slope computation failed");
        assert!(slope > 0.0, "expected positive slope, got {slope}");
    }

    #[test]
    fn slope_flat_is_near_zero() {
        let losses = vec![5.0f32; 10];
        let slope = compute_loss_slope(&losses).expect("slope computation failed");
        assert!(
            slope.abs() < 1e-5,
            "expected ~0 slope for flat sequence, got {slope}"
        );
    }

    #[test]
    fn slope_empty_returns_error() {
        let result = compute_loss_slope(&[]);
        assert_eq!(result, Err(ConvergenceError::EmptyHistory));
    }

    #[test]
    fn slope_single_element_returns_invalid_window() {
        let result = compute_loss_slope(&[1.0]);
        assert_eq!(result, Err(ConvergenceError::InvalidWindow { size: 1 }));
    }

    #[test]
    fn slope_two_elements_correct() {
        // losses = [0, 1] → slope should be 1.0
        let slope = compute_loss_slope(&[0.0, 1.0]).expect("slope computation failed");
        assert!(
            (slope - 1.0).abs() < 1e-4,
            "expected slope ≈ 1, got {slope}"
        );
    }

    // ------------------------------------------------------------------
    // compute_loss_slope_xy
    // ------------------------------------------------------------------

    #[test]
    fn slope_xy_matches_positional_slope_for_contiguous_steps() {
        // When step == positional index (the common case: one update() per
        // training step), compute_loss_slope_xy must agree exactly with
        // compute_loss_slope.
        let losses: Vec<f32> = (0..10).map(|i| 10.0 - i as f32).collect();
        let points: Vec<(usize, f32)> = losses.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        let plain = compute_loss_slope(&losses).expect("plain slope");
        let xy = compute_loss_slope_xy(&points).expect("xy slope");
        assert!(
            (plain - xy).abs() < 1e-4,
            "plain={plain}, xy={xy}: must agree when step == index"
        );
    }

    #[test]
    fn slope_xy_scales_inversely_with_step_spacing() {
        // Identical loss values, but recorded 10 real steps apart instead
        // of 1 → the per-step slope must be ~10x shallower.
        let losses: Vec<f32> = (0..10).map(|i| 10.0 - i as f32).collect();
        let dense: Vec<(usize, f32)> = losses.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        let sparse: Vec<(usize, f32)> = losses
            .iter()
            .enumerate()
            .map(|(i, &l)| (i * 10, l))
            .collect();
        let dense_slope = compute_loss_slope_xy(&dense).expect("dense slope");
        let sparse_slope = compute_loss_slope_xy(&sparse).expect("sparse slope");
        assert!(
            (dense_slope - sparse_slope * 10.0).abs() < 1e-3,
            "dense_slope={dense_slope} should equal ~10x sparse_slope={sparse_slope}"
        );
    }

    #[test]
    fn slope_xy_empty_returns_error() {
        let result = compute_loss_slope_xy(&[]);
        assert_eq!(result, Err(ConvergenceError::EmptyHistory));
    }

    #[test]
    fn slope_xy_single_element_returns_invalid_window() {
        let result = compute_loss_slope_xy(&[(0, 1.0)]);
        assert_eq!(result, Err(ConvergenceError::InvalidWindow { size: 1 }));
    }

    #[test]
    fn slope_xy_duplicate_steps_returns_zero_not_panic() {
        // Degenerate denominator (all x equal) must fall back to 0.0
        // exactly like compute_loss_slope's positional-index counterpart,
        // not divide by zero.
        let result = compute_loss_slope_xy(&[(5, 1.0), (5, 2.0), (5, 3.0)]);
        assert_eq!(result, Ok(0.0));
    }

    // ------------------------------------------------------------------
    // compute_relative_improvement
    // ------------------------------------------------------------------

    #[test]
    fn relative_improvement_known_values() {
        // first=2.0, last=1.0 → (2-1)/2 = 0.5
        let ri = compute_relative_improvement(&[2.0, 1.5, 1.0]).expect("ri failed");
        assert!((ri - 0.5).abs() < 1e-5, "expected 0.5, got {ri}");
    }

    #[test]
    fn relative_improvement_diverging_is_negative() {
        // first=1.0, last=2.0 → (1-2)/1 = -1.0
        let ri = compute_relative_improvement(&[1.0, 1.5, 2.0]).expect("ri failed");
        assert!(ri < 0.0, "expected negative ri for diverging, got {ri}");
    }

    #[test]
    fn relative_improvement_flat_is_zero() {
        let ri = compute_relative_improvement(&[1.0, 1.0, 1.0]).expect("ri failed");
        assert!(ri.abs() < 1e-6, "expected ~0 for flat, got {ri}");
    }

    #[test]
    fn relative_improvement_empty_returns_error() {
        assert_eq!(
            compute_relative_improvement(&[]),
            Err(ConvergenceError::EmptyHistory)
        );
    }

    #[test]
    fn relative_improvement_near_zero_first_uses_clamp() {
        // first ≈ 0 → denominator clamped to 1e-8, no divide by zero
        let ri = compute_relative_improvement(&[0.0, 1.0]).expect("ri with zero first failed");
        assert!(ri.is_finite(), "expected finite result, got {ri}");
    }

    // ------------------------------------------------------------------
    // compute_oscillation_score
    // ------------------------------------------------------------------

    #[test]
    fn oscillation_flat_sequence_is_near_zero() {
        let losses = vec![1.0f32; 10];
        let score = compute_oscillation_score(&losses).expect("osc score failed");
        assert!(
            score < 1e-5,
            "expected ~0 oscillation for flat, got {score}"
        );
    }

    #[test]
    fn oscillation_alternating_sequence_is_high() {
        // Alternating 0.0, 1.0, 0.0, 1.0, … — high oscillation
        let losses: Vec<f32> = (0..10)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        let score = compute_oscillation_score(&losses).expect("osc score failed");
        assert!(score > 0.1, "expected high oscillation score, got {score}");
    }

    #[test]
    fn oscillation_empty_returns_error() {
        assert_eq!(
            compute_oscillation_score(&[]),
            Err(ConvergenceError::EmptyHistory)
        );
    }

    #[test]
    fn oscillation_single_element_returns_invalid_window() {
        assert_eq!(
            compute_oscillation_score(&[1.0]),
            Err(ConvergenceError::InvalidWindow { size: 1 })
        );
    }

    // ------------------------------------------------------------------
    // ema_smooth_losses
    // ------------------------------------------------------------------

    #[test]
    fn ema_smooth_first_element_preserved() {
        let losses = vec![5.0f32, 3.0, 1.0];
        let out = ema_smooth_losses(&losses, 0.9);
        assert_eq!(out.len(), 3);
        assert!(
            (out[0] - 5.0).abs() < 1e-6,
            "first element should equal losses[0]"
        );
    }

    #[test]
    fn ema_smooth_converges_to_constant_input() {
        // Feed constant 2.0; starting from 100.0 seed, after many steps should be near 2.0
        let mut losses = vec![100.0f32];
        losses.extend(vec![2.0f32; 200]);
        let out = ema_smooth_losses(&losses, 0.9);
        let last = *out.last().expect("output should not be empty");
        assert!(
            (last - 2.0).abs() < 0.1,
            "EMA should converge to constant value, got {last}"
        );
    }

    #[test]
    fn ema_smooth_empty_returns_empty() {
        let out = ema_smooth_losses(&[], 0.9);
        assert!(out.is_empty());
    }

    #[test]
    fn ema_smooth_length_equals_input_length() {
        let losses: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let out = ema_smooth_losses(&losses, 0.95);
        assert_eq!(out.len(), losses.len());
    }

    // ------------------------------------------------------------------
    // detect_convergence_phase
    // ------------------------------------------------------------------

    #[test]
    fn phase_diverging_when_slope_high() {
        let config = ConvergenceConfig::default();
        // loss_scale = 1.0 throughout this block so relative_oscillation ==
        // oscillation_score, matching the values these tests were originally
        // written against.
        let phase = detect_convergence_phase(0.5, -0.1, 0.0, 1.0, &config);
        assert_eq!(phase, ConvergencePhase::Diverging);
    }

    #[test]
    fn phase_oscillating_when_high_osc_score() {
        let config = ConvergenceConfig::default();
        // slope below diverge threshold, high oscillation (1.0 relative to
        // loss_scale=1.0 is far above the default oscillation_threshold=0.3)
        let phase = detect_convergence_phase(0.0, 0.001, 1.0, 1.0, &config);
        assert_eq!(phase, ConvergencePhase::Oscillating);
    }

    #[test]
    fn phase_plateau_when_tiny_improvement() {
        let config = ConvergenceConfig::default();
        // slope < diverge, low osc, tiny improvement
        let phase = detect_convergence_phase(-1e-6, 1e-6, 0.0, 1.0, &config);
        assert_eq!(phase, ConvergencePhase::Plateau);
    }

    #[test]
    fn phase_rapid_decline_when_large_improvement() {
        let config = ConvergenceConfig::default();
        let phase = detect_convergence_phase(-0.05, 0.5, 0.0, 1.0, &config);
        assert_eq!(phase, ConvergencePhase::RapidDecline);
    }

    #[test]
    fn phase_slow_improvement_otherwise() {
        let config = ConvergenceConfig::default();
        // relative_improvement in (plateau, rapid) range
        let phase = detect_convergence_phase(-0.001, 0.005, 0.0, 1.0, &config);
        assert_eq!(phase, ConvergencePhase::SlowImprovement);
    }

    // ------------------------------------------------------------------
    // detect_convergence_phase — oscillation unit-mismatch regression
    // ------------------------------------------------------------------

    #[test]
    fn phase_not_oscillating_for_realistic_small_relative_noise() {
        // Regression for the absolute-vs-relative unit mismatch: a loss
        // around 0.05 with an absolute step-to-step std dev of 5e-4 (1% of
        // the loss magnitude -- ordinary minibatch noise on a converging
        // run) used to ALWAYS be classified Oscillating, because the raw
        // absolute oscillation_score (5e-4) was compared directly against
        // `2 * plateau_threshold` (2e-4), a relative-rate threshold two
        // orders of magnitude too tight for an absolute quantity at this
        // loss scale. With the score normalized by loss_scale (0.05) before
        // comparison, 5e-4/0.05 = 0.01 (1%) sits well under the default
        // relative `oscillation_threshold` (0.3), so a healthy
        // slow-improvement run must be able to reach a non-Oscillating
        // phase.
        let config = ConvergenceConfig::default();
        // relative_improvement = 0.005 lands strictly between
        // plateau_threshold (1e-4) and rapid_threshold (1e-2), i.e.
        // SlowImprovement once oscillation no longer short-circuits rule 2.
        let phase = detect_convergence_phase(-1e-5, 0.005, 5e-4, 0.05, &config);
        assert_ne!(
            phase,
            ConvergencePhase::Oscillating,
            "1% relative noise on a realistic loss scale must not be flagged as oscillating"
        );
        assert_eq!(phase, ConvergencePhase::SlowImprovement);
    }

    #[test]
    fn phase_still_oscillating_for_genuinely_large_relative_swings() {
        // Same realistic loss scale (0.05) but with the loss swinging by an
        // absolute std dev of 0.04 between steps -- 80% of the loss
        // magnitude, i.e. genuinely oscillating -- must still be detected.
        let config = ConvergenceConfig::default();
        let phase = detect_convergence_phase(0.0, 0.0, 0.04, 0.05, &config);
        assert_eq!(phase, ConvergencePhase::Oscillating);
    }

    // ------------------------------------------------------------------
    // ConvergenceAnalyzer
    // ------------------------------------------------------------------

    fn make_analyzer_with_data(n: usize, start: f32, end: f32) -> ConvergenceAnalyzer {
        let cfg = ConvergenceConfig {
            window_size: n,
            min_history: n / 2,
            ..Default::default()
        };
        let mut analyzer = ConvergenceAnalyzer::new(cfg);
        for i in 0..n {
            let loss = start + (end - start) * (i as f32 / (n - 1) as f32);
            analyzer.update(i, loss);
        }
        analyzer
    }

    #[test]
    fn analyzer_update_records_history() {
        let mut analyzer = ConvergenceAnalyzer::default();
        analyzer.update(0, 1.0);
        analyzer.update(1, 0.9);
        assert_eq!(analyzer.history_len(), 2);
    }

    #[test]
    fn analyzer_analyze_insufficient_history_returns_error() {
        let analyzer = ConvergenceAnalyzer::default(); // min_history=50, no data
        let result = analyzer.analyze();
        assert!(matches!(
            result,
            Err(ConvergenceError::InsufficientHistory { .. })
        ));
    }

    #[test]
    fn analyzer_analyze_after_sufficient_history_succeeds() {
        let analyzer = make_analyzer_with_data(100, 1.0, 0.1);
        let stats = analyzer
            .analyze()
            .expect("analyze should succeed with enough data");
        assert!(stats.current_loss > 0.0);
        assert!(stats.n_points > 0);
    }

    #[test]
    fn analyzer_phase_initializing_when_too_few_points() {
        let analyzer = ConvergenceAnalyzer::default();
        assert_eq!(analyzer.phase(), ConvergencePhase::Initializing);
    }

    #[test]
    fn analyzer_phase_not_initializing_after_enough_data() {
        let analyzer = make_analyzer_with_data(100, 1.0, 0.1);
        let phase = analyzer.phase();
        assert_ne!(phase, ConvergencePhase::Initializing);
    }

    #[test]
    fn analyzer_reset_clears_history() {
        let mut analyzer = make_analyzer_with_data(100, 1.0, 0.1);
        analyzer.reset();
        assert_eq!(analyzer.history_len(), 0);
        assert_eq!(analyzer.phase(), ConvergencePhase::Initializing);
    }

    #[test]
    fn analyzer_best_loss_returns_minimum() {
        let mut analyzer = ConvergenceAnalyzer::default();
        analyzer.update(0, 5.0);
        analyzer.update(1, 0.5);
        analyzer.update(2, 2.0);
        let best = analyzer.best_loss().expect("best_loss should be Some");
        assert!((best - 0.5).abs() < 1e-6, "expected best=0.5, got {best}");
    }

    #[test]
    fn analyzer_ema_starts_at_first_loss() {
        let mut analyzer = ConvergenceAnalyzer::default();
        analyzer.update(0, 3.1);
        // After first update, ema_loss should equal the first loss
        assert!((analyzer.ema_loss - 3.1).abs() < 1e-5);
    }

    // ------------------------------------------------------------------
    // estimate_steps_to_convergence
    // ------------------------------------------------------------------

    #[test]
    fn estimate_steps_returns_some_for_improving() {
        let analyzer = make_analyzer_with_data(100, 1.0, 0.1);
        let result = estimate_steps_to_convergence(&analyzer, 0.01);
        // Should be Some because loss is improving
        assert!(result.is_some(), "expected Some for improving loss");
    }

    #[test]
    fn estimate_steps_returns_none_for_diverging() {
        // Loss increases from 0.1 to 1.0 → slope is positive
        let analyzer = make_analyzer_with_data(100, 0.1, 1.0);
        let result = estimate_steps_to_convergence(&analyzer, 0.01);
        assert!(result.is_none(), "expected None for diverging loss");
    }

    #[test]
    fn estimate_steps_returns_none_when_already_at_target() {
        let analyzer = make_analyzer_with_data(100, 1.0, 0.001);
        // target higher than current loss → already reached
        let result = estimate_steps_to_convergence(&analyzer, 1.0);
        assert!(
            result.is_none(),
            "expected None when already at or below target"
        );
    }

    #[test]
    fn estimate_steps_returns_none_for_insufficient_history() {
        let analyzer = ConvergenceAnalyzer::default();
        let result = estimate_steps_to_convergence(&analyzer, 0.01);
        assert!(result.is_none());
    }

    #[test]
    fn estimate_steps_scales_with_real_step_spacing_not_index() {
        // Regression: `analyze()` previously regressed `loss_slope` against
        // the POSITIONAL index within the window (0, 1, 2, ...) instead of
        // the actually-recorded `step` values, so `estimate_steps_to_convergence`
        // silently reported "additional recorded points" mislabeled as
        // "additional training steps". Two analyzers fed the identical loss
        // trajectory but different real step spacing must therefore produce
        // DIFFERENT (spacing-scaled) slopes/estimates -- under the bug they
        // would have been identical, since positional index never saw the
        // step values at all.
        let n = 100usize;
        let make = |step_gap: usize| {
            let cfg = ConvergenceConfig {
                window_size: n,
                min_history: n / 2,
                ..Default::default()
            };
            let mut a = ConvergenceAnalyzer::new(cfg);
            for i in 0..n {
                let loss = 1.0 - 0.9 * (i as f32 / (n - 1) as f32);
                a.update(i * step_gap, loss);
            }
            a
        };

        let dense = make(1);
        let sparse = make(50);

        let dense_slope = dense.analyze().expect("dense analyze").loss_slope;
        let sparse_slope = sparse.analyze().expect("sparse analyze").loss_slope;

        // Same total loss change spread over 50x more real steps between
        // recordings -> the per-real-step slope must be proportionally
        // shallower (closer to zero), not identical to the dense case.
        assert!(
            dense_slope.abs() > sparse_slope.abs() * 20.0,
            "dense_slope={dense_slope}, sparse_slope={sparse_slope}: wider real step spacing \
             must yield a proportionally shallower per-step slope"
        );

        let target = 0.05;
        let dense_steps = estimate_steps_to_convergence(&dense, target).expect("dense estimate");
        let sparse_steps = estimate_steps_to_convergence(&sparse, target).expect("sparse estimate");

        // The wider-spaced run needs proportionally more *real steps* to
        // close the same absolute loss gap.
        assert!(
            sparse_steps > dense_steps * 20,
            "dense_steps={dense_steps}, sparse_steps={sparse_steps}: wider step spacing must \
             yield a proportionally larger step estimate, not the same positional-index one"
        );
    }

    // ------------------------------------------------------------------
    // ConvergenceAnalyzer — max_history / best_loss / history_len
    // ------------------------------------------------------------------

    #[test]
    fn analyzer_max_history_bounds_buffer_but_best_loss_and_len_stay_exact() {
        // Regression for the "bounded to avoid unbounded memory growth" doc
        // claim that the implementation never actually enforced.
        // `max_history` now genuinely bounds the internal buffer when set,
        // but `best_loss()` and `history_len()` must remain correct for the
        // WHOLE run rather than silently degrading to "best/length within
        // whatever happens to still be in the capped buffer".
        let cfg = ConvergenceConfig {
            window_size: 10,
            min_history: 5,
            max_history: Some(20),
            ..Default::default()
        };
        let mut analyzer = ConvergenceAnalyzer::new(cfg);
        // The true global minimum (0.01) is recorded early and would be
        // long evicted from a 20-entry capped buffer after 100 updates.
        for i in 0..100usize {
            let loss = if i == 3 { 0.01 } else { 1.0 };
            analyzer.update(i, loss);
        }

        assert_eq!(
            analyzer.history_len(),
            100,
            "history_len() must report the true cumulative update count, not the capped \
             buffer size"
        );
        let best = analyzer.best_loss().expect("best_loss should be Some");
        assert!(
            (best - 0.01).abs() < 1e-6,
            "best_loss() must still find the long-evicted global minimum, got {best}"
        );
        assert!(
            analyzer.analyze().is_ok(),
            "analyze() must still succeed once enough points remain post-cap"
        );
    }

    #[test]
    fn analyzer_max_history_never_locks_out_below_min_history() {
        // A misconfigured `max_history < min_history` must not permanently
        // strand the analyzer in `InsufficientHistory` / `Initializing`.
        let cfg = ConvergenceConfig {
            window_size: 10,
            min_history: 50,
            max_history: Some(5), // smaller than min_history
            ..Default::default()
        };
        let mut analyzer = ConvergenceAnalyzer::new(cfg);
        for i in 0..60usize {
            analyzer.update(i, 1.0);
        }
        assert!(
            analyzer.analyze().is_ok(),
            "analyze() must be reachable even when max_history < min_history"
        );
    }

    // ------------------------------------------------------------------
    // compute_loss_percentile
    // ------------------------------------------------------------------

    #[test]
    fn percentile_p0_equals_min() {
        let losses = vec![3.0f32, 1.0, 4.0, 1.5, 2.5];
        let p0 = compute_loss_percentile(&losses, 0.0).expect("percentile failed");
        assert!((p0 - 1.0).abs() < 1e-5, "p0 should equal min=1.0, got {p0}");
    }

    #[test]
    fn percentile_p100_equals_max() {
        let losses = vec![3.0f32, 1.0, 4.0, 1.5, 2.5];
        let p100 = compute_loss_percentile(&losses, 100.0).expect("percentile failed");
        assert!(
            (p100 - 4.0).abs() < 1e-5,
            "p100 should equal max=4.0, got {p100}"
        );
    }

    #[test]
    fn percentile_p50_near_median() {
        let losses = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let p50 = compute_loss_percentile(&losses, 50.0).expect("percentile failed");
        assert!(
            (p50 - 3.0).abs() < 0.5,
            "p50 should be near median=3.0, got {p50}"
        );
    }

    #[test]
    fn percentile_empty_returns_error() {
        assert_eq!(
            compute_loss_percentile(&[], 50.0),
            Err(ConvergenceError::EmptyHistory)
        );
    }

    #[test]
    fn percentile_single_element_returns_that_element() {
        let p = compute_loss_percentile(&[7.0], 50.0).expect("single element percentile failed");
        assert!((p - 7.0).abs() < 1e-5);
    }

    // ------------------------------------------------------------------
    // loss_improvement_rate
    // ------------------------------------------------------------------

    #[test]
    fn improvement_rate_known_sequence() {
        // losses: [1.0, 1.0, 0.5], window=2
        // rate[0] = (1.0 - 0.5) / 1.0 = 0.5
        let losses = vec![1.0f32, 1.0, 0.5];
        let rates = loss_improvement_rate(&losses, 2).expect("improvement rate failed");
        assert_eq!(rates.len(), 1);
        assert!(
            (rates[0] - 0.5).abs() < 1e-5,
            "expected 0.5, got {}",
            rates[0]
        );
    }

    #[test]
    fn improvement_rate_length_correct() {
        let losses: Vec<f32> = (0..10).map(|i| 10.0 - i as f32).collect();
        let rates = loss_improvement_rate(&losses, 3).expect("improvement rate failed");
        assert_eq!(rates.len(), losses.len() - 3);
    }

    #[test]
    fn improvement_rate_empty_returns_error() {
        assert_eq!(
            loss_improvement_rate(&[], 3),
            Err(ConvergenceError::EmptyHistory)
        );
    }

    #[test]
    fn improvement_rate_invalid_window_too_small() {
        assert_eq!(
            loss_improvement_rate(&[1.0, 2.0, 3.0], 1),
            Err(ConvergenceError::InvalidWindow { size: 1 })
        );
    }

    #[test]
    fn improvement_rate_window_too_large_returns_error() {
        assert_eq!(
            loss_improvement_rate(&[1.0, 2.0], 5),
            Err(ConvergenceError::WindowTooLarge {
                window: 5,
                history_len: 2,
            })
        );
    }

    // ------------------------------------------------------------------
    // detect_phase_transitions
    // ------------------------------------------------------------------

    #[test]
    fn phase_transitions_empty_history_returns_empty() {
        let analyzer = ConvergenceAnalyzer::default();
        let transitions = detect_phase_transitions(&analyzer);
        assert!(transitions.is_empty());
    }

    #[test]
    fn phase_transitions_constant_no_transitions() {
        // Constant loss → all chunks in Plateau → only one initial entry
        let cfg = ConvergenceConfig {
            window_size: 20,
            min_history: 2,
            ..Default::default()
        };
        let mut analyzer = ConvergenceAnalyzer::new(cfg);
        for i in 0..60 {
            analyzer.update(i, 1.0);
        }
        let transitions = detect_phase_transitions(&analyzer);
        // First chunk always adds an entry; subsequent same-phase chunks do not
        assert_eq!(
            transitions.len(),
            1,
            "constant loss should produce one phase entry"
        );
    }

    #[test]
    fn phase_transitions_detects_change() {
        let cfg = ConvergenceConfig {
            window_size: 20,
            min_history: 2,
            ..Default::default()
        };
        let mut analyzer = ConvergenceAnalyzer::new(cfg);
        // First 20 steps: rapid decrease
        for i in 0..20 {
            analyzer.update(i, 100.0 - i as f32 * 4.0);
        }
        // Next 20 steps: constant (plateau)
        for i in 20..40 {
            analyzer.update(i, 20.0);
        }
        let transitions = detect_phase_transitions(&analyzer);
        assert!(
            transitions.len() >= 2,
            "should detect at least 2 phase entries for clearly different regions"
        );
    }

    // ------------------------------------------------------------------
    // format_convergence_report
    // ------------------------------------------------------------------

    #[test]
    fn format_report_non_empty_string() {
        let stats = ConvergenceStats {
            phase: ConvergencePhase::SlowImprovement,
            current_loss: 0.1,
            smoothed_loss: 0.12,
            loss_slope: -0.001,
            relative_improvement: 0.05,
            oscillation_score: 0.002,
            window_min: 0.09,
            window_max: 0.15,
            n_points: 50,
        };
        let report = format_convergence_report(&stats);
        assert!(!report.is_empty());
        assert!(report.contains("SlowImprovement"));
    }

    #[test]
    fn format_report_contains_phase_name_for_all_phases() {
        let phases = [
            ConvergencePhase::Initializing,
            ConvergencePhase::RapidDecline,
            ConvergencePhase::SlowImprovement,
            ConvergencePhase::Plateau,
            ConvergencePhase::Oscillating,
            ConvergencePhase::Diverging,
        ];
        for phase in phases {
            let stats = ConvergenceStats {
                phase,
                current_loss: 0.5,
                smoothed_loss: 0.5,
                loss_slope: 0.0,
                relative_improvement: 0.0,
                oscillation_score: 0.0,
                window_min: 0.5,
                window_max: 0.5,
                n_points: 10,
            };
            let report = format_convergence_report(&stats);
            assert!(
                !report.is_empty(),
                "report for phase {phase:?} should not be empty"
            );
        }
    }

    // ------------------------------------------------------------------
    // generate_convergence_report
    // ------------------------------------------------------------------

    #[test]
    fn generate_report_returns_ok_with_sufficient_data() {
        let analyzer = make_analyzer_with_data(100, 1.0, 0.1);
        let report =
            generate_convergence_report(&analyzer, 0.01).expect("report generation failed");
        assert!(report.current_loss > 0.0);
        assert!(report.best_loss <= report.current_loss || report.best_loss >= 0.0);
        assert!(!report.recommendation.is_empty());
    }

    #[test]
    fn generate_report_insufficient_data_returns_error() {
        let analyzer = ConvergenceAnalyzer::default();
        let result = generate_convergence_report(&analyzer, 0.01);
        assert!(matches!(
            result,
            Err(ConvergenceError::InsufficientHistory { .. })
        ));
    }

    #[test]
    fn generate_report_steps_analyzed_correct() {
        let n = 100;
        let analyzer = make_analyzer_with_data(n, 1.0, 0.1);
        let report =
            generate_convergence_report(&analyzer, 0.01).expect("report generation failed");
        assert_eq!(report.steps_analyzed, n);
    }

    // ------------------------------------------------------------------
    // ConvergenceError variants
    // ------------------------------------------------------------------

    #[test]
    fn error_insufficient_history_display() {
        let err = ConvergenceError::InsufficientHistory {
            needed: 50,
            got: 10,
        };
        let msg = format!("{err}");
        assert!(msg.contains("50"));
        assert!(msg.contains("10"));
    }

    #[test]
    fn error_window_too_large_display() {
        let err = ConvergenceError::WindowTooLarge {
            window: 100,
            history_len: 5,
        };
        let msg = format!("{err}");
        assert!(msg.contains("100"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn error_invalid_window_display() {
        let err = ConvergenceError::InvalidWindow { size: 0 };
        let msg = format!("{err}");
        assert!(msg.contains("0"));
    }

    #[test]
    fn error_empty_history_display() {
        let err = ConvergenceError::EmptyHistory;
        let msg = format!("{err}");
        assert!(!msg.is_empty());
    }

    // ------------------------------------------------------------------
    // ConvergencePhase variants
    // ------------------------------------------------------------------

    #[test]
    fn phase_variants_all_distinct() {
        let phases = [
            ConvergencePhase::Initializing,
            ConvergencePhase::RapidDecline,
            ConvergencePhase::SlowImprovement,
            ConvergencePhase::Plateau,
            ConvergencePhase::Oscillating,
            ConvergencePhase::Diverging,
        ];
        // Verify each variant can be cloned and compared
        for phase in phases {
            assert_eq!(phase, phase.clone());
        }
    }

    #[test]
    fn phase_initializing_copy_semantics() {
        let p = ConvergencePhase::Initializing;
        let q = p; // Copy
        assert_eq!(p, q);
    }

    #[test]
    fn phase_plateau_debug_format() {
        let p = ConvergencePhase::Plateau;
        let s = format!("{p:?}");
        assert!(s.contains("Plateau"));
    }

    // ------------------------------------------------------------------
    // ConvergenceConfig default
    // ------------------------------------------------------------------

    #[test]
    fn config_default_values_are_sensible() {
        let cfg = ConvergenceConfig::default();
        assert_eq!(cfg.window_size, 100);
        assert_eq!(cfg.min_history, 50);
        assert!((cfg.plateau_threshold - 1e-4).abs() < 1e-8);
        assert!((cfg.rapid_threshold - 1e-2).abs() < 1e-8);
        assert!((cfg.diverge_threshold - 0.1).abs() < 1e-6);
        assert_eq!(cfg.oscillation_window, 20);
        assert!((cfg.ema_decay - 0.95).abs() < 1e-6);
        assert!((cfg.oscillation_threshold - 0.3).abs() < 1e-6);
        assert_eq!(cfg.max_history, None);
    }

    // ------------------------------------------------------------------
    // ema_smooth_losses: unwrap_or safety
    // ------------------------------------------------------------------

    #[test]
    fn ema_smooth_single_element() {
        let out = ema_smooth_losses(&[42.0], 0.9);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 42.0).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // Additional edge-case tests
    // ------------------------------------------------------------------

    #[test]
    fn slope_large_sequence_is_stable() {
        // 500 points linearly decreasing from 100 to 0 → slope should be ~-100/499
        let losses: Vec<f32> = (0..500)
            .map(|i| 100.0 - i as f32 * (100.0 / 499.0))
            .collect();
        let slope = compute_loss_slope(&losses).expect("slope on large sequence failed");
        let expected = -100.0 / 499.0;
        assert!(
            (slope - expected).abs() < 1e-3,
            "slope={slope}, expected≈{expected}"
        );
    }

    #[test]
    fn analyzer_detects_improving_phase_after_decrease() {
        let analyzer = make_analyzer_with_data(100, 1.0, 0.001);
        let phase = analyzer.phase();
        // Should not be Diverging or Oscillating
        assert_ne!(phase, ConvergencePhase::Diverging);
        assert_ne!(phase, ConvergencePhase::Initializing);
    }

    #[test]
    fn improvement_rate_all_positive_for_decreasing_losses() {
        let losses: Vec<f32> = (0..10).map(|i| 10.0 - i as f32).collect();
        let rates = loss_improvement_rate(&losses, 2).expect("improvement rate failed");
        for r in &rates {
            assert!(
                *r > 0.0,
                "rate should be positive for decreasing losses, got {r}"
            );
        }
    }

    #[test]
    fn percentile_clamped_above_100() {
        // percentile > 100 should clamp to 100 → max
        let losses = vec![1.0f32, 2.0, 3.0];
        let p = compute_loss_percentile(&losses, 999.0).expect("clamped percentile failed");
        assert!((p - 3.0).abs() < 1e-5);
    }

    #[test]
    fn percentile_clamped_below_0() {
        let losses = vec![1.0f32, 2.0, 3.0];
        let p = compute_loss_percentile(&losses, -5.0).expect("clamped percentile failed");
        assert!((p - 1.0).abs() < 1e-5);
    }
}
