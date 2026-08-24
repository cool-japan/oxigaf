//! Real-time monitoring utilities for long training runs.
//!
//! Provides:
//! - [`TrainingEvent`] — a single recorded training event with timing and metrics
//! - [`MonitorConfig`] — configuration for the monitor
//! - [`ImprovementTracker`] — tracks loss improvement over time
//! - [`TrainingMonitor`] — the main monitoring struct
//! - [`MonitorSnapshot`] — current state at a given moment
//! - Various utility functions for ETA estimation, smoothing, and status formatting

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors that can occur during training monitoring.
#[derive(Debug, Error)]
pub enum MonitorError {
    /// No training data is available yet.
    #[error("No data yet")]
    NoData,

    /// A configuration value is invalid.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// Smoothing window is larger than available history.
    #[error("Window too large: window {window} > history {history}")]
    WindowTooLarge { window: usize, history: usize },

    /// A division by zero was attempted.
    #[error("Division by zero: {0}")]
    DivisionByZero(String),

    /// A metric value is invalid or out of range.
    #[error("Invalid metric: {0}")]
    InvalidMetric(String),
}

// ---------------------------------------------------------------------------
// TrainingEvent
// ---------------------------------------------------------------------------

/// A single recorded training event with timing, loss, and extra metrics.
#[derive(Debug, Clone)]
pub struct TrainingEvent {
    /// The training step index.
    pub step: usize,
    /// Elapsed wall-clock time in seconds since training start.
    pub elapsed_secs: f32,
    /// Loss value at this step.
    pub loss: f32,
    /// Optional extra named metrics (name, value) pairs.
    pub extra_metrics: Vec<(String, f32)>,
}

impl TrainingEvent {
    /// Create a new `TrainingEvent` at `step` with `elapsed_secs` and `loss`.
    pub fn new(step: usize, elapsed_secs: f32, loss: f32) -> Self {
        Self {
            step,
            elapsed_secs,
            loss,
            extra_metrics: Vec::new(),
        }
    }

    /// Add a named metric value (builder-style).
    pub fn with_metric(mut self, name: impl Into<String>, value: f32) -> Self {
        self.extra_metrics.push((name.into(), value));
        self
    }

    /// Compute steps per second relative to a previous event.
    ///
    /// Returns 0.0 if the elapsed time difference is zero or negative.
    pub fn steps_per_second(&self, prev: &TrainingEvent) -> f32 {
        let dt = self.elapsed_secs - prev.elapsed_secs;
        if dt <= 0.0 {
            return 0.0;
        }
        let ds = self.step.saturating_sub(prev.step) as f32;
        ds / dt
    }
}

// ---------------------------------------------------------------------------
// MonitorConfig
// ---------------------------------------------------------------------------

/// Configuration for [`TrainingMonitor`].
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Number of events to average for smooth metrics (default 20).
    pub smoothing_window: usize,
    /// Steps without improvement before declaring a stall (default 200).
    pub stall_patience: usize,
    /// Minimum relative improvement to count as improvement (default 0.001 = 0.1%).
    pub improvement_threshold: f32,
    /// Maximum events to keep in history (default 1000).
    pub max_history: usize,
    /// Number of events to use for ETA estimation (default 50).
    pub eta_window: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            smoothing_window: 20,
            stall_patience: 200,
            improvement_threshold: 0.001,
            max_history: 1000,
            eta_window: 50,
        }
    }
}

impl MonitorConfig {
    /// Validate the configuration, returning an error if any field is invalid.
    pub fn validate(&self) -> Result<(), MonitorError> {
        if self.smoothing_window == 0 {
            return Err(MonitorError::InvalidConfig(
                "smoothing_window must be > 0".into(),
            ));
        }
        if self.stall_patience == 0 {
            return Err(MonitorError::InvalidConfig(
                "stall_patience must be > 0".into(),
            ));
        }
        if self.eta_window == 0 {
            return Err(MonitorError::InvalidConfig("eta_window must be > 0".into()));
        }
        if self.max_history == 0 {
            return Err(MonitorError::InvalidConfig(
                "max_history must be > 0".into(),
            ));
        }
        if self.eta_window > self.max_history {
            return Err(MonitorError::InvalidConfig(format!(
                "eta_window ({}) must not exceed max_history ({})",
                self.eta_window, self.max_history
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ImprovementTracker
// ---------------------------------------------------------------------------

/// Tracks loss improvement over a rolling window to detect stalls and trends.
#[derive(Debug, Clone)]
pub struct ImprovementTracker {
    /// Recent loss values within the tracker window.
    pub recent_losses: Vec<f32>,
    /// The window size (max number of recent losses retained).
    pub window: usize,
    /// Best (lowest) loss seen so far.
    pub best: f32,
    /// Number of consecutive steps without a qualifying improvement.
    pub steps_without_improvement: usize,
}

impl ImprovementTracker {
    /// Create a new `ImprovementTracker` with given `window`.
    pub fn new(window: usize) -> Self {
        Self {
            recent_losses: Vec::new(),
            window: window.max(1),
            best: f32::INFINITY,
            steps_without_improvement: 0,
        }
    }

    /// Update with a new loss value.
    ///
    /// Returns `true` if the loss improved by more than `improvement_threshold`
    /// relative to the current best.
    pub fn update(&mut self, loss: f32, improvement_threshold: f32) -> bool {
        // Maintain the rolling window.
        if self.recent_losses.len() >= self.window {
            self.recent_losses.remove(0);
        }
        self.recent_losses.push(loss);

        // Determine whether this counts as an improvement.
        let improved = if self.best.is_infinite() {
            // First data point always counts as improvement.
            true
        } else {
            let relative_improvement = (self.best - loss) / self.best.abs().max(f32::EPSILON);
            relative_improvement > improvement_threshold
        };

        if improved {
            self.best = loss.min(self.best);
            self.steps_without_improvement = 0;
        } else {
            self.steps_without_improvement += 1;
        }

        improved
    }

    /// Returns `true` if `steps_without_improvement >= patience`.
    pub fn is_stalled(&self, patience: usize) -> bool {
        self.steps_without_improvement >= patience
    }

    /// Compute the linear regression slope over `recent_losses`.
    ///
    /// A negative slope means loss is decreasing (improving).
    /// Returns 0.0 if there are fewer than 2 data points or regression fails.
    pub fn trend(&self) -> f32 {
        if self.recent_losses.len() < 2 {
            return 0.0;
        }
        let n = self.recent_losses.len();
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        match linear_regression(&x, &self.recent_losses) {
            Ok((slope, _)) => slope,
            Err(_) => 0.0,
        }
    }

    /// Mean of the recent losses, or `None` if empty.
    pub fn mean_recent(&self) -> Option<f32> {
        if self.recent_losses.is_empty() {
            return None;
        }
        let sum: f32 = self.recent_losses.iter().sum();
        Some(sum / self.recent_losses.len() as f32)
    }
}

// ---------------------------------------------------------------------------
// MonitorSnapshot
// ---------------------------------------------------------------------------

/// A snapshot of the training monitor's current state.
#[derive(Debug, Clone)]
pub struct MonitorSnapshot {
    /// Current training step.
    pub step: usize,
    /// Elapsed time in seconds since training started.
    pub elapsed_secs: f32,
    /// Raw current loss.
    pub current_loss: f32,
    /// Smoothed loss (moving average over `smoothing_window` events).
    pub smoothed_loss: f32,
    /// Best (lowest) loss seen so far.
    pub best_loss: f32,
    /// Number of steps since the last qualifying improvement.
    pub steps_since_improvement: usize,
    /// Whether training is considered stalled.
    pub is_stalled: bool,
    /// Recent throughput in steps per second.
    pub steps_per_second: f32,
    /// Estimated seconds remaining to reach `total_steps`, if known.
    pub estimated_seconds_remaining: Option<f32>,
    /// Change in smoothed loss per 100 steps (negative = improving).
    pub loss_improvement_rate: f32,
}

// ---------------------------------------------------------------------------
// TrainingMonitor
// ---------------------------------------------------------------------------

/// Real-time monitor for a training run.
pub struct TrainingMonitor {
    /// Configuration.
    pub config: MonitorConfig,
    history: Vec<TrainingEvent>,
    improvement_tracker: ImprovementTracker,
    best_loss: f32,
    total_steps: Option<usize>,
}

impl TrainingMonitor {
    /// Create a new `TrainingMonitor` with the given config.
    ///
    /// Returns an error if the config is invalid.
    pub fn new(config: MonitorConfig) -> Result<Self, MonitorError> {
        config.validate()?;
        let window = config.smoothing_window;
        Ok(Self {
            config,
            history: Vec::new(),
            improvement_tracker: ImprovementTracker::new(window),
            best_loss: f32::INFINITY,
            total_steps: None,
        })
    }

    /// Set the total number of training steps (enables ETA computation).
    pub fn with_total_steps(mut self, total: usize) -> Self {
        self.total_steps = Some(total);
        self
    }

    /// Record a new training event.
    ///
    /// Updates the improvement tracker and trims history if needed.
    pub fn record(&mut self, event: TrainingEvent) {
        self.improvement_tracker
            .update(event.loss, self.config.improvement_threshold);

        if event.loss < self.best_loss {
            self.best_loss = event.loss;
        }

        self.history.push(event);

        // Trim oldest events if over the limit.
        if self.history.len() > self.config.max_history {
            let excess = self.history.len() - self.config.max_history;
            self.history.drain(..excess);
        }
    }

    /// Build a snapshot of the current training state.
    pub fn snapshot(&self) -> Result<MonitorSnapshot, MonitorError> {
        let latest = self.history.last().ok_or(MonitorError::NoData)?;

        // Smoothed loss: mean over last `smoothing_window` events.
        let window = self.config.smoothing_window.min(self.history.len());
        let recent_slice = &self.history[self.history.len() - window..];
        let smoothed_loss = recent_slice.iter().map(|e| e.loss).sum::<f32>() / window as f32;

        // Steps per second: use last two events if available.
        let steps_per_second = if self.history.len() >= 2 {
            let n = self.history.len();
            latest.steps_per_second(&self.history[n - 2])
        } else {
            0.0
        };

        // ETA: use `eta_window` most recent events to estimate throughput.
        let estimated_seconds_remaining = self.total_steps.and_then(|total| {
            if total <= latest.step {
                return Some(0.0);
            }
            let remaining_steps = (total - latest.step) as f32;
            let eta_n = self.config.eta_window.min(self.history.len());
            if eta_n < 2 {
                return None;
            }
            let eta_slice = &self.history[self.history.len() - eta_n..];
            let first = eta_slice.first()?;
            let last = eta_slice.last()?;
            let dt = last.elapsed_secs - first.elapsed_secs;
            let ds = last.step.saturating_sub(first.step) as f32;
            if dt <= 0.0 || ds <= 0.0 {
                return None;
            }
            let rate = ds / dt; // steps per second
            Some(remaining_steps / rate)
        });

        // Improvement rate: slope per 100 steps using the improvement_tracker trend.
        let loss_improvement_rate = self.improvement_tracker.trend() * 100.0;

        Ok(MonitorSnapshot {
            step: latest.step,
            elapsed_secs: latest.elapsed_secs,
            current_loss: latest.loss,
            smoothed_loss,
            best_loss: self.best_loss,
            steps_since_improvement: self.improvement_tracker.steps_without_improvement,
            is_stalled: self
                .improvement_tracker
                .is_stalled(self.config.stall_patience),
            steps_per_second,
            estimated_seconds_remaining,
            loss_improvement_rate,
        })
    }

    /// Number of events in the history.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Best (lowest) loss observed.
    pub fn best_loss(&self) -> f32 {
        self.best_loss
    }

    /// Most recent recorded event, if any.
    pub fn latest(&self) -> Option<&TrainingEvent> {
        self.history.last()
    }

    /// Whether the training is considered stalled.
    pub fn is_stalled(&self) -> bool {
        self.improvement_tracker
            .is_stalled(self.config.stall_patience)
    }
}

// ---------------------------------------------------------------------------
// Smoothing utilities
// ---------------------------------------------------------------------------

/// Compute exponential moving average (EMA) over a slice.
///
/// `alpha` is the weight for the current value (0 < alpha <= 1).
/// Higher alpha tracks the raw signal more closely; lower alpha smooths more.
/// Returns an empty vector for empty input.
pub fn ema_smooth(values: &[f32], alpha: f32) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let alpha = alpha.clamp(f32::EPSILON, 1.0);
    let mut out = Vec::with_capacity(values.len());
    let mut ema = values[0];
    out.push(ema);
    for &v in &values[1..] {
        ema = alpha * v + (1.0 - alpha) * ema;
        out.push(ema);
    }
    out
}

/// Simple moving average (SMA) with given window size.
///
/// Returns `WindowTooLarge` if `window > values.len()` and `InvalidConfig` if
/// `window == 0`.
pub fn sma_smooth(values: &[f32], window: usize) -> Result<Vec<f32>, MonitorError> {
    if window == 0 {
        return Err(MonitorError::InvalidConfig("window must be > 0".into()));
    }
    if window > values.len() {
        return Err(MonitorError::WindowTooLarge {
            window,
            history: values.len(),
        });
    }
    let mut out = Vec::with_capacity(values.len() - window + 1);
    let mut running_sum: f32 = values[..window].iter().sum();
    out.push(running_sum / window as f32);
    for i in window..values.len() {
        running_sum += values[i];
        running_sum -= values[i - window];
        out.push(running_sum / window as f32);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Linear regression
// ---------------------------------------------------------------------------

/// Compute ordinary least-squares linear regression over paired (x, y) slices.
///
/// Returns `(slope, intercept)`. Returns `DivisionByZero` if x-values are
/// all identical (zero variance). Returns `InvalidMetric` if the slices have
/// different lengths or fewer than 2 points.
pub fn linear_regression(x: &[f32], y: &[f32]) -> Result<(f32, f32), MonitorError> {
    if x.len() != y.len() {
        return Err(MonitorError::InvalidMetric(format!(
            "x length {} != y length {}",
            x.len(),
            y.len()
        )));
    }
    let n = x.len();
    if n < 2 {
        return Err(MonitorError::InvalidMetric(
            "need at least 2 points for linear regression".into(),
        ));
    }
    let n_f = n as f32;
    let sum_x: f32 = x.iter().sum();
    let sum_y: f32 = y.iter().sum();
    let sum_xx: f32 = x.iter().map(|&xi| xi * xi).sum();
    let sum_xy: f32 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();

    let denom = n_f * sum_xx - sum_x * sum_x;
    if denom.abs() < f32::EPSILON {
        return Err(MonitorError::DivisionByZero(
            "x values have zero variance (all identical)".into(),
        ));
    }

    let slope = (n_f * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n_f;
    Ok((slope, intercept))
}

// ---------------------------------------------------------------------------
// Formatting utilities
// ---------------------------------------------------------------------------

/// Format a duration in seconds as a human-readable ETA string.
///
/// Examples: `"45s"`, `"3m 45s"`, `"2h 3m 45s"`.
pub fn format_eta(seconds: f32) -> String {
    let secs = seconds.max(0.0) as u64;
    let hours = secs / 3600;
    let mins = (secs % 3600) / 60;
    let s = secs % 60;
    if hours > 0 {
        format!("{}h {}m {}s", hours, mins, s)
    } else if mins > 0 {
        format!("{}m {}s", mins, s)
    } else {
        format!("{}s", s)
    }
}

/// Format an elapsed time in seconds as `"h:mm:ss"` style.
///
/// Examples: `"0:00:00"`, `"1:03:45"`.
pub fn format_elapsed(seconds: f32) -> String {
    let secs = seconds.max(0.0) as u64;
    let hours = secs / 3600;
    let mins = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{}:{:02}:{:02}", hours, mins, s)
}

/// Format training status as a single-line status bar string.
///
/// Example output:
/// `"Step 1500/5000 | Loss: 0.1234 (best: 0.1100) | 12.3 it/s | ETA: 5m 32s | [=====     ]"`
pub fn format_status_line(snapshot: &MonitorSnapshot, total_steps: Option<usize>) -> String {
    let step_part = match total_steps {
        Some(total) => format!("Step {}/{}", snapshot.step, total),
        None => format!("Step {}", snapshot.step),
    };

    let loss_part = format!(
        "Loss: {:.4} (best: {:.4})",
        snapshot.current_loss, snapshot.best_loss
    );

    let speed_part = format!("{:.1} it/s", snapshot.steps_per_second);

    let eta_part = match snapshot.estimated_seconds_remaining {
        Some(eta) => format!("ETA: {}", format_eta(eta)),
        None => "ETA: ?".to_string(),
    };

    // Progress bar
    let bar_part = match total_steps {
        Some(total) if total > 0 => {
            let bar_width = 10usize;
            let filled = ((snapshot.step as f64 / total as f64) * bar_width as f64) as usize;
            let filled = filled.min(bar_width);
            let empty = bar_width - filled;
            let bar: String = "=".repeat(filled) + &" ".repeat(empty);
            format!("[{}]", bar)
        }
        _ => String::new(),
    };

    let mut parts = vec![step_part, loss_part, speed_part, eta_part];
    if !bar_part.is_empty() {
        parts.push(bar_part);
    }
    parts.join(" | ")
}

// ---------------------------------------------------------------------------
// Divergence detection
// ---------------------------------------------------------------------------

/// Detect if training has diverged (loss increasing sharply).
///
/// Returns `true` if the last loss is more than `threshold` times the first loss.
/// Returns `false` for empty or single-element slices.
pub fn detect_divergence(recent_losses: &[f32], threshold: f32) -> bool {
    if recent_losses.len() < 2 {
        return false;
    }
    let first = recent_losses[0];
    // `recent_losses.len() >= 2` is guaranteed by the check above, so this
    // index is always in bounds; indexing directly avoids an
    // `Option::expect()` in a production code path (COOLJAPAN no-unwrap
    // policy).
    let last = recent_losses[recent_losses.len() - 1];
    if first <= 0.0 {
        return false;
    }
    last / first > threshold
}

// ---------------------------------------------------------------------------
// Loss percentile
// ---------------------------------------------------------------------------

/// Compute a percentile of the losses stored in `history`.
///
/// `percentile` must be in `0.0..=100.0`.
/// Returns `NoData` for empty history and `InvalidMetric` for out-of-range percentile.
pub fn loss_percentile(history: &[TrainingEvent], percentile: f32) -> Result<f32, MonitorError> {
    if history.is_empty() {
        return Err(MonitorError::NoData);
    }
    if !(0.0..=100.0).contains(&percentile) {
        return Err(MonitorError::InvalidMetric(format!(
            "percentile {percentile} out of range 0..=100"
        )));
    }
    let mut losses: Vec<f32> = history.iter().map(|e| e.loss).collect();
    losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = losses.len();
    if percentile <= 0.0 {
        return Ok(losses[0]);
    }
    if percentile >= 100.0 {
        return Ok(losses[n - 1]);
    }
    // Linear interpolation
    let rank = (percentile / 100.0) * (n - 1) as f32;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = rank - lo as f32;
    Ok(losses[lo] * (1.0 - frac) + losses[hi] * frac)
}

// ---------------------------------------------------------------------------
// Throughput statistics
// ---------------------------------------------------------------------------

/// Throughput statistics computed from event history.
#[derive(Debug, Clone)]
pub struct ThroughputStats {
    /// Mean steps per second over the full history.
    pub mean_steps_per_second: f32,
    /// Peak steps per second observed in any consecutive pair.
    pub peak_steps_per_second: f32,
    /// Most recent steps per second (last consecutive pair).
    pub current_steps_per_second: f32,
    /// Steps per second averaged over the last `recent_window` events.
    pub recent_steps_per_second: f32,
}

/// Compute throughput statistics from the event history.
///
/// Returns `NoData` if fewer than 2 events are available.
pub fn compute_throughput(
    history: &[TrainingEvent],
    recent_window: usize,
) -> Result<ThroughputStats, MonitorError> {
    if history.len() < 2 {
        return Err(MonitorError::NoData);
    }

    // Per-pair step rates.
    let rates: Vec<f32> = history
        .windows(2)
        .map(|w| w[1].steps_per_second(&w[0]))
        .collect();

    let mean_steps_per_second = rates.iter().sum::<f32>() / rates.len() as f32;
    let peak_steps_per_second = rates.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let current_steps_per_second = *rates.last().unwrap_or(&0.0);

    // Recent window rates.
    let recent_n = recent_window.min(rates.len());
    let recent_rates = &rates[rates.len() - recent_n..];
    let recent_steps_per_second = if recent_rates.is_empty() {
        0.0
    } else {
        recent_rates.iter().sum::<f32>() / recent_rates.len() as f32
    };

    Ok(ThroughputStats {
        mean_steps_per_second,
        peak_steps_per_second,
        current_steps_per_second,
        recent_steps_per_second,
    })
}

// ---------------------------------------------------------------------------
// Training summary
// ---------------------------------------------------------------------------

/// Summary of a complete training run.
#[derive(Debug, Clone)]
pub struct TrainingSummary {
    /// Total number of steps in the run.
    pub total_steps: usize,
    /// Total elapsed time in seconds.
    pub total_secs: f32,
    /// Best (lowest) loss achieved during training.
    pub best_loss: f32,
    /// Loss at the final step.
    pub final_loss: f32,
    /// Step at which `best_loss` was first achieved.
    pub best_step: usize,
    /// Relative improvement: `(initial_loss - best_loss) / initial_loss`.
    pub improvement_fraction: f32,
    /// Overall mean throughput (steps/second).
    pub mean_throughput: f32,
    /// Number of times the patience threshold was hit (stall events).
    pub stall_count: usize,
}

/// Summarize a full training run given its history.
///
/// Returns `NoData` if fewer than 1 event is available.
pub fn summarize_training(
    history: &[TrainingEvent],
    improvement_threshold: f32,
    patience: usize,
) -> Result<TrainingSummary, MonitorError> {
    if history.is_empty() {
        return Err(MonitorError::NoData);
    }

    let first = &history[0];
    let last = &history[history.len() - 1];

    let initial_loss = first.loss;
    let final_loss = last.loss;
    // Use the step *delta* across the recorded window rather than the
    // absolute final step index. `history` may be a trimmed window (see
    // `TrainingMonitor::record`, which drops the oldest events past
    // `max_history`) or start mid-run after a resume, in which case
    // `first.step` is not 0; dividing the absolute `last.step` by the
    // windowed `total_secs` below would wildly inflate `mean_throughput`.
    // This also matches the field's own documentation ("Total number of
    // steps in the run") and is a no-op for the common case where the
    // window starts at step 0.
    let total_steps = last.step.saturating_sub(first.step);
    let total_secs = last.elapsed_secs - first.elapsed_secs;

    // Find best loss and the step where it was first achieved.
    let mut best_loss = f32::INFINITY;
    let mut best_step = first.step;
    for event in history {
        if event.loss < best_loss {
            best_loss = event.loss;
            best_step = event.step;
        }
    }

    let improvement_fraction = if initial_loss.abs() > f32::EPSILON {
        (initial_loss - best_loss) / initial_loss
    } else {
        0.0
    };

    // Overall throughput.
    let mean_throughput = if total_secs > 0.0 {
        total_steps as f32 / total_secs
    } else {
        0.0
    };

    // Count stall events: count how many times consecutive_no_improve crosses patience.
    let mut stall_count = 0usize;
    let mut consecutive_no_improve = 0usize;
    let mut current_best = f32::INFINITY;
    let mut stalled = false;
    for event in history {
        let improved = if current_best.is_infinite() {
            true
        } else {
            let rel = (current_best - event.loss) / current_best.abs().max(f32::EPSILON);
            rel > improvement_threshold
        };

        if improved {
            if event.loss < current_best {
                current_best = event.loss;
            }
            consecutive_no_improve = 0;
            stalled = false;
        } else {
            consecutive_no_improve += 1;
            if !stalled && consecutive_no_improve >= patience {
                stall_count += 1;
                stalled = true;
            }
        }
    }

    Ok(TrainingSummary {
        total_steps,
        total_secs,
        best_loss,
        final_loss,
        best_step,
        improvement_fraction,
        mean_throughput,
        stall_count,
    })
}

/// Format a [`TrainingSummary`] as human-readable text.
pub fn format_training_summary(summary: &TrainingSummary) -> String {
    format!(
        "Training Summary\n\
         ----------------\n\
         Total steps:   {}\n\
         Total time:    {}\n\
         Best loss:     {:.6} (step {})\n\
         Final loss:    {:.6}\n\
         Improvement:   {:.2}%\n\
         Throughput:    {:.2} steps/s\n\
         Stall events:  {}",
        summary.total_steps,
        format_elapsed(summary.total_secs),
        summary.best_loss,
        summary.best_step,
        summary.final_loss,
        summary.improvement_fraction * 100.0,
        summary.mean_throughput,
        summary.stall_count,
    )
}

// ---------------------------------------------------------------------------
// Robust loss smoothing
// ---------------------------------------------------------------------------

/// Compute a robust smoothed loss by rejecting outliers beyond `k` standard
/// deviations from the mean, then averaging the remaining values.
///
/// Returns `NoData` for empty input, `InvalidConfig` if `k <= 0`, and `NoData`
/// if all values are rejected as outliers.
pub fn robust_smooth_loss(losses: &[f32], k: f32) -> Result<f32, MonitorError> {
    if losses.is_empty() {
        return Err(MonitorError::NoData);
    }
    if k <= 0.0 {
        return Err(MonitorError::InvalidConfig("k must be > 0".into()));
    }

    let n = losses.len() as f32;
    let mean = losses.iter().sum::<f32>() / n;
    let variance = losses.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let std_dev = variance.sqrt();

    let filtered: Vec<f32> = if std_dev < f32::EPSILON {
        // All values are essentially identical; no outliers possible.
        losses.to_vec()
    } else {
        losses
            .iter()
            .cloned()
            .filter(|&v| (v - mean).abs() <= k * std_dev)
            .collect()
    };

    if filtered.is_empty() {
        return Err(MonitorError::NoData);
    }

    let robust_mean = filtered.iter().sum::<f32>() / filtered.len() as f32;
    Ok(robust_mean)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // TrainingEvent
    // -----------------------------------------------------------------------

    #[test]
    fn training_event_new_sets_fields() {
        let e = TrainingEvent::new(42, 10.5, 0.123);
        assert_eq!(e.step, 42);
        assert!((e.elapsed_secs - 10.5).abs() < 1e-6);
        assert!((e.loss - 0.123).abs() < 1e-6);
        assert!(e.extra_metrics.is_empty());
    }

    #[test]
    fn training_event_with_metric_builder() {
        let e = TrainingEvent::new(1, 0.0, 0.5)
            .with_metric("psnr", 32.1)
            .with_metric("ssim", 0.98);
        assert_eq!(e.extra_metrics.len(), 2);
        assert_eq!(e.extra_metrics[0].0, "psnr");
        assert!((e.extra_metrics[1].1 - 0.98).abs() < 1e-5);
    }

    #[test]
    fn training_event_steps_per_second_normal() {
        let prev = TrainingEvent::new(100, 5.0, 0.5);
        let curr = TrainingEvent::new(110, 7.0, 0.4);
        let sps = curr.steps_per_second(&prev);
        // (110-100) / (7.0-5.0) = 5.0
        assert!((sps - 5.0).abs() < 1e-4, "Expected 5.0, got {sps}");
    }

    #[test]
    fn training_event_steps_per_second_zero_dt() {
        let prev = TrainingEvent::new(100, 5.0, 0.5);
        let curr = TrainingEvent::new(110, 5.0, 0.4);
        assert_eq!(curr.steps_per_second(&prev), 0.0);
    }

    // -----------------------------------------------------------------------
    // MonitorConfig
    // -----------------------------------------------------------------------

    #[test]
    fn monitor_config_default_is_valid() {
        let config = MonitorConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn monitor_config_validate_zero_smoothing_window() {
        let config = MonitorConfig {
            smoothing_window: 0,
            ..MonitorConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn monitor_config_validate_zero_stall_patience() {
        let config = MonitorConfig {
            stall_patience: 0,
            ..MonitorConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn monitor_config_validate_zero_eta_window() {
        let config = MonitorConfig {
            eta_window: 0,
            ..MonitorConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn monitor_config_validate_eta_exceeds_max_history() {
        let config = MonitorConfig {
            eta_window: 200,
            max_history: 100,
            ..MonitorConfig::default()
        };
        assert!(config.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // ImprovementTracker
    // -----------------------------------------------------------------------

    #[test]
    fn improvement_tracker_first_update_is_improvement() {
        let mut tracker = ImprovementTracker::new(10);
        let improved = tracker.update(0.5, 0.001);
        assert!(
            improved,
            "First data point should always count as improvement"
        );
    }

    #[test]
    fn improvement_tracker_update_improving() {
        let mut tracker = ImprovementTracker::new(10);
        tracker.update(0.5, 0.001);
        let improved = tracker.update(0.3, 0.001); // large improvement
        assert!(improved);
        assert_eq!(tracker.steps_without_improvement, 0);
    }

    #[test]
    fn improvement_tracker_update_plateau() {
        let mut tracker = ImprovementTracker::new(10);
        tracker.update(0.5, 0.001);
        let improved = tracker.update(0.4999, 0.001); // tiny, below threshold
        assert!(!improved);
        assert_eq!(tracker.steps_without_improvement, 1);
    }

    #[test]
    fn improvement_tracker_is_stalled() {
        let mut tracker = ImprovementTracker::new(10);
        tracker.update(0.5, 0.001);
        for _ in 0..5 {
            tracker.update(0.4999, 0.001);
        }
        assert!(tracker.is_stalled(5));
        assert!(!tracker.is_stalled(6));
    }

    #[test]
    fn improvement_tracker_trend_increasing() {
        let mut tracker = ImprovementTracker::new(20);
        for i in 0..10 {
            tracker.update(i as f32 * 0.1, 0.001);
        }
        // Losses are 0.0, 0.1, ... 0.9 — positive trend
        let t = tracker.trend();
        assert!(t > 0.0, "Expected positive trend, got {t}");
    }

    #[test]
    fn improvement_tracker_trend_flat() {
        let mut tracker = ImprovementTracker::new(20);
        for _ in 0..5 {
            tracker.update(0.5, 0.001);
        }
        // All identical → trend ≈ 0 (may return 0.0 from DivisionByZero guard)
        let t = tracker.trend();
        assert!(t.abs() < 1e-4, "Expected near-zero trend, got {t}");
    }

    #[test]
    fn improvement_tracker_mean_recent_some() {
        let mut tracker = ImprovementTracker::new(10);
        tracker.update(1.0, 0.001);
        tracker.update(3.0, 0.001);
        let m = tracker.mean_recent();
        assert!(m.is_some());
        assert!((m.unwrap() - 2.0).abs() < 1e-4);
    }

    #[test]
    fn improvement_tracker_mean_recent_none_when_empty() {
        let tracker = ImprovementTracker::new(10);
        assert!(tracker.mean_recent().is_none());
    }

    // -----------------------------------------------------------------------
    // TrainingMonitor
    // -----------------------------------------------------------------------

    #[test]
    fn training_monitor_new_valid_config() {
        let config = MonitorConfig::default();
        assert!(TrainingMonitor::new(config).is_ok());
    }

    #[test]
    fn training_monitor_new_invalid_config() {
        let config = MonitorConfig {
            smoothing_window: 0,
            ..MonitorConfig::default()
        };
        assert!(TrainingMonitor::new(config).is_err());
    }

    #[test]
    fn training_monitor_record_grows_history() {
        let mut monitor = TrainingMonitor::new(MonitorConfig::default()).unwrap();
        monitor.record(TrainingEvent::new(1, 1.0, 0.5));
        monitor.record(TrainingEvent::new(2, 2.0, 0.4));
        assert_eq!(monitor.history_len(), 2);
    }

    #[test]
    fn training_monitor_record_trims_history() {
        let config = MonitorConfig {
            max_history: 5,
            eta_window: 3,
            smoothing_window: 3,
            ..MonitorConfig::default()
        };
        let mut monitor = TrainingMonitor::new(config).unwrap();
        for i in 0..10 {
            monitor.record(TrainingEvent::new(i, i as f32, 0.5));
        }
        assert_eq!(monitor.history_len(), 5);
    }

    #[test]
    fn training_monitor_snapshot_no_data_error() {
        let monitor = TrainingMonitor::new(MonitorConfig::default()).unwrap();
        assert!(matches!(monitor.snapshot(), Err(MonitorError::NoData)));
    }

    #[test]
    fn training_monitor_snapshot_best_loss() {
        let mut monitor = TrainingMonitor::new(MonitorConfig::default()).unwrap();
        monitor.record(TrainingEvent::new(1, 1.0, 0.9));
        monitor.record(TrainingEvent::new(2, 2.0, 0.3));
        monitor.record(TrainingEvent::new(3, 3.0, 0.5));
        let snap = monitor.snapshot().unwrap();
        assert!((snap.best_loss - 0.3).abs() < 1e-5);
    }

    #[test]
    fn training_monitor_snapshot_steps_since_improvement() {
        let config = MonitorConfig {
            improvement_threshold: 0.001,
            ..MonitorConfig::default()
        };
        let mut monitor = TrainingMonitor::new(config).unwrap();
        monitor.record(TrainingEvent::new(1, 1.0, 1.0));
        monitor.record(TrainingEvent::new(2, 2.0, 0.5));
        // big improvement → steps_since_improvement reset to 0
        monitor.record(TrainingEvent::new(3, 3.0, 0.4999));
        // tiny (plateau) → steps_since_improvement increments
        let snap = monitor.snapshot().unwrap();
        assert!(snap.steps_since_improvement >= 1);
    }

    #[test]
    fn training_monitor_is_stalled_false_when_improving() {
        let config = MonitorConfig {
            stall_patience: 10,
            ..MonitorConfig::default()
        };
        let mut monitor = TrainingMonitor::new(config).unwrap();
        for i in 0..5 {
            monitor.record(TrainingEvent::new(i, i as f32, 1.0 / (i + 1) as f32));
        }
        assert!(!monitor.is_stalled());
    }

    #[test]
    fn training_monitor_with_total_steps_eta() {
        let config = MonitorConfig {
            eta_window: 3,
            ..MonitorConfig::default()
        };
        let mut monitor = TrainingMonitor::new(config).unwrap().with_total_steps(100);
        for i in 0..5 {
            monitor.record(TrainingEvent::new(i, i as f32, 0.5));
        }
        let snap = monitor.snapshot().unwrap();
        assert!(snap.estimated_seconds_remaining.is_some());
    }

    // -----------------------------------------------------------------------
    // ema_smooth
    // -----------------------------------------------------------------------

    #[test]
    fn ema_smooth_alpha_one_is_identity() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let result = ema_smooth(&values, 1.0);
        assert_eq!(result.len(), values.len());
        for (r, v) in result.iter().zip(values.iter()) {
            assert!((r - v).abs() < 1e-5, "Expected {v}, got {r}");
        }
    }

    #[test]
    fn ema_smooth_small_alpha_smooths() {
        let values: Vec<f32> = vec![10.0, 0.0, 10.0, 0.0, 10.0];
        let result = ema_smooth(&values, 0.01);
        // With tiny alpha, the series should vary much less than raw
        let range = result.iter().cloned().fold(f32::NAN, f32::max)
            - result.iter().cloned().fold(f32::NAN, f32::min);
        let raw_range = 10.0f32;
        assert!(
            range < raw_range,
            "EMA range {range} should be < raw range {raw_range}"
        );
    }

    #[test]
    fn ema_smooth_single_element() {
        let result = ema_smooth(&[42.0f32], 0.5);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 42.0).abs() < 1e-5);
    }

    #[test]
    fn ema_smooth_empty_returns_empty() {
        let result = ema_smooth(&[], 0.5);
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // sma_smooth
    // -----------------------------------------------------------------------

    #[test]
    fn sma_smooth_window_one_is_identity() {
        let values = vec![1.0f32, 2.0, 3.0];
        let result = sma_smooth(&values, 1).unwrap();
        assert_eq!(result.len(), values.len());
        for (r, v) in result.iter().zip(values.iter()) {
            assert!((r - v).abs() < 1e-5);
        }
    }

    #[test]
    fn sma_smooth_window_exceeds_length_error() {
        let values = vec![1.0f32, 2.0];
        assert!(matches!(
            sma_smooth(&values, 5),
            Err(MonitorError::WindowTooLarge { .. })
        ));
    }

    #[test]
    fn sma_smooth_window_zero_error() {
        let values = vec![1.0f32, 2.0];
        assert!(matches!(
            sma_smooth(&values, 0),
            Err(MonitorError::InvalidConfig(_))
        ));
    }

    #[test]
    fn sma_smooth_correct_output() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = sma_smooth(&values, 2).unwrap();
        // windows: [1,2] -> 1.5, [2,3] -> 2.5, [3,4] -> 3.5
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.5).abs() < 1e-5);
        assert!((result[1] - 2.5).abs() < 1e-5);
        assert!((result[2] - 3.5).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // linear_regression
    // -----------------------------------------------------------------------

    #[test]
    fn linear_regression_flat_slope_zero() {
        let x: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let y = vec![3.0f32; 5];
        let (slope, intercept) = linear_regression(&x, &y).unwrap();
        assert!(slope.abs() < 1e-4, "Expected slope ~0, got {slope}");
        assert!((intercept - 3.0).abs() < 1e-4);
    }

    #[test]
    fn linear_regression_increasing_positive_slope() {
        let x: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let y: Vec<f32> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let (slope, intercept) = linear_regression(&x, &y).unwrap();
        assert!((slope - 2.0).abs() < 1e-4, "Expected slope=2, got {slope}");
        assert!((intercept - 1.0).abs() < 1e-4);
    }

    #[test]
    fn linear_regression_length_mismatch_error() {
        let x = vec![1.0f32, 2.0, 3.0];
        let y = vec![1.0f32, 2.0];
        assert!(matches!(
            linear_regression(&x, &y),
            Err(MonitorError::InvalidMetric(_))
        ));
    }

    #[test]
    fn linear_regression_zero_variance_error() {
        // All x identical → denominator is zero
        let x = vec![5.0f32; 5];
        let y = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert!(matches!(
            linear_regression(&x, &y),
            Err(MonitorError::DivisionByZero(_))
        ));
    }

    #[test]
    fn linear_regression_too_few_points_error() {
        let x = vec![1.0f32];
        let y = vec![2.0f32];
        assert!(matches!(
            linear_regression(&x, &y),
            Err(MonitorError::InvalidMetric(_))
        ));
    }

    // -----------------------------------------------------------------------
    // format_eta
    // -----------------------------------------------------------------------

    #[test]
    fn format_eta_seconds_only() {
        let s = format_eta(45.0);
        assert_eq!(s, "45s");
    }

    #[test]
    fn format_eta_minutes_and_seconds() {
        let s = format_eta(3.0 * 60.0 + 45.0);
        assert_eq!(s, "3m 45s");
    }

    #[test]
    fn format_eta_hours() {
        let s = format_eta(2.0 * 3600.0 + 3.0 * 60.0 + 45.0);
        assert_eq!(s, "2h 3m 45s");
    }

    // -----------------------------------------------------------------------
    // format_elapsed
    // -----------------------------------------------------------------------

    #[test]
    fn format_elapsed_zero() {
        assert_eq!(format_elapsed(0.0), "0:00:00");
    }

    #[test]
    fn format_elapsed_one_hour() {
        assert_eq!(format_elapsed(3600.0), "1:00:00");
    }

    #[test]
    fn format_elapsed_complex() {
        // 1h 3m 45s = 3600 + 180 + 45 = 3825s
        assert_eq!(format_elapsed(3825.0), "1:03:45");
    }

    // -----------------------------------------------------------------------
    // format_status_line
    // -----------------------------------------------------------------------

    #[test]
    fn format_status_line_non_empty() {
        let snap = MonitorSnapshot {
            step: 1500,
            elapsed_secs: 60.0,
            current_loss: 0.1234,
            smoothed_loss: 0.1200,
            best_loss: 0.1100,
            steps_since_improvement: 5,
            is_stalled: false,
            steps_per_second: 12.3,
            estimated_seconds_remaining: Some(332.0),
            loss_improvement_rate: -0.01,
        };
        let line = format_status_line(&snap, Some(5000));
        assert!(!line.is_empty());
        assert!(line.contains("1500"), "Should contain current step");
        assert!(line.contains("5000"), "Should contain total steps");
    }

    #[test]
    fn format_status_line_contains_loss() {
        let snap = MonitorSnapshot {
            step: 100,
            elapsed_secs: 10.0,
            current_loss: 0.9876,
            smoothed_loss: 0.9800,
            best_loss: 0.9000,
            steps_since_improvement: 0,
            is_stalled: false,
            steps_per_second: 10.0,
            estimated_seconds_remaining: None,
            loss_improvement_rate: -0.05,
        };
        let line = format_status_line(&snap, None);
        assert!(line.contains("0.9876") || line.contains("Loss"));
    }

    // -----------------------------------------------------------------------
    // detect_divergence
    // -----------------------------------------------------------------------

    #[test]
    fn detect_divergence_flat_false() {
        let losses = vec![0.5f32, 0.5, 0.5, 0.5, 0.5];
        assert!(!detect_divergence(&losses, 2.0));
    }

    #[test]
    fn detect_divergence_tripling_true() {
        let losses = vec![0.5f32, 0.6, 0.8, 1.2, 1.7];
        // 1.7 / 0.5 = 3.4 > 2.0
        assert!(detect_divergence(&losses, 2.0));
    }

    #[test]
    fn detect_divergence_empty_false() {
        assert!(!detect_divergence(&[], 2.0));
    }

    #[test]
    fn detect_divergence_single_false() {
        assert!(!detect_divergence(&[1.0], 2.0));
    }

    #[test]
    fn detect_divergence_minimum_length_two_no_panic() {
        // Regression test: `detect_divergence` used to fetch the last
        // element via `.last().expect(...)`. This exercises the exact
        // boundary (len == 2) that the length guard is meant to cover,
        // ensuring the direct-indexing replacement neither panics nor
        // miscomputes the ratio.
        assert!(!detect_divergence(&[1.0, 1.5], 2.0));
        assert!(detect_divergence(&[1.0, 3.0], 2.0));
    }

    // -----------------------------------------------------------------------
    // loss_percentile
    // -----------------------------------------------------------------------

    #[test]
    fn loss_percentile_empty_error() {
        assert!(matches!(
            loss_percentile(&[], 50.0),
            Err(MonitorError::NoData)
        ));
    }

    #[test]
    fn loss_percentile_p0_is_min() {
        let history: Vec<TrainingEvent> = vec![0.1, 0.5, 0.3, 0.2, 0.4]
            .into_iter()
            .enumerate()
            .map(|(i, l)| TrainingEvent::new(i, i as f32, l))
            .collect();
        let p = loss_percentile(&history, 0.0).unwrap();
        assert!((p - 0.1).abs() < 1e-5, "Expected min=0.1, got {p}");
    }

    #[test]
    fn loss_percentile_p100_is_max() {
        let history: Vec<TrainingEvent> = vec![0.1, 0.5, 0.3, 0.2, 0.4]
            .into_iter()
            .enumerate()
            .map(|(i, l)| TrainingEvent::new(i, i as f32, l))
            .collect();
        let p = loss_percentile(&history, 100.0).unwrap();
        assert!((p - 0.5).abs() < 1e-5, "Expected max=0.5, got {p}");
    }

    #[test]
    fn loss_percentile_out_of_range_error() {
        let history = vec![TrainingEvent::new(0, 0.0, 0.5)];
        assert!(matches!(
            loss_percentile(&history, 101.0),
            Err(MonitorError::InvalidMetric(_))
        ));
        assert!(matches!(
            loss_percentile(&history, -1.0),
            Err(MonitorError::InvalidMetric(_))
        ));
    }

    // -----------------------------------------------------------------------
    // compute_throughput
    // -----------------------------------------------------------------------

    #[test]
    fn compute_throughput_single_event_error() {
        let history = vec![TrainingEvent::new(0, 0.0, 0.5)];
        assert!(matches!(
            compute_throughput(&history, 5),
            Err(MonitorError::NoData)
        ));
    }

    #[test]
    fn compute_throughput_valid_history() {
        let history: Vec<TrainingEvent> = (0..10)
            .map(|i| TrainingEvent::new(i, i as f32, 0.5))
            .collect();
        let stats = compute_throughput(&history, 3).unwrap();
        // Each pair has 1 step per 1 second = 1.0 sps
        assert!((stats.mean_steps_per_second - 1.0).abs() < 1e-4);
        assert!((stats.current_steps_per_second - 1.0).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // summarize_training
    // -----------------------------------------------------------------------

    #[test]
    fn summarize_training_empty_error() {
        assert!(matches!(
            summarize_training(&[], 0.001, 100),
            Err(MonitorError::NoData)
        ));
    }

    #[test]
    fn summarize_training_correct_best_step() {
        let history = vec![
            TrainingEvent::new(1, 1.0, 0.9),
            TrainingEvent::new(2, 2.0, 0.3),
            TrainingEvent::new(3, 3.0, 0.5),
        ];
        let summary = summarize_training(&history, 0.001, 5).unwrap();
        assert_eq!(summary.best_step, 2);
        assert!((summary.best_loss - 0.3).abs() < 1e-5);
    }

    #[test]
    fn summarize_training_improvement_fraction() {
        let history = vec![
            TrainingEvent::new(0, 0.0, 1.0),
            TrainingEvent::new(10, 10.0, 0.5),
        ];
        let summary = summarize_training(&history, 0.001, 5).unwrap();
        // (1.0 - 0.5) / 1.0 = 0.5
        assert!((summary.improvement_fraction - 0.5).abs() < 1e-4);
    }

    #[test]
    fn summarize_training_windowed_history_throughput_not_inflated() {
        // Regression test: `history` can be a trimmed window (see
        // `TrainingMonitor::record`, which drops events past
        // `max_history`) or start mid-run after a resume, so `first.step`
        // is not necessarily 0. `total_steps`/`mean_throughput` must be
        // derived from the step *delta* across the window, not from the
        // absolute final step index — otherwise dividing an absolute step
        // count by only the windowed duration wildly inflates throughput.
        //
        // Window: steps 5000..=5100 (100 steps) over 10 seconds.
        let history = vec![
            TrainingEvent::new(5000, 500.0, 0.2),
            TrainingEvent::new(5050, 505.0, 0.15),
            TrainingEvent::new(5100, 510.0, 0.1),
        ];
        let summary = summarize_training(&history, 0.001, 100).unwrap();

        assert_eq!(summary.total_steps, 100, "total_steps must be the delta across the window (5100 - 5000), not the absolute final step index 5100");
        assert!((summary.total_secs - 10.0).abs() < 1e-4);
        // 100 steps / 10s = 10 steps/s. The old buggy computation
        // (5100 absolute steps / 10s = 510 steps/s) would be off by ~51x.
        assert!(
            (summary.mean_throughput - 10.0).abs() < 1e-3,
            "mean_throughput should be ~10.0 steps/s, got {}",
            summary.mean_throughput
        );
    }

    #[test]
    fn summarize_training_non_windowed_history_unchanged() {
        // When the history starts at step 0 (the common, non-trimmed,
        // non-resumed case), total_steps must equal the previous
        // behavior (the final step index), since first.step == 0 makes
        // the delta and the absolute index identical.
        let history = vec![
            TrainingEvent::new(0, 0.0, 1.0),
            TrainingEvent::new(50, 5.0, 0.5),
        ];
        let summary = summarize_training(&history, 0.001, 100).unwrap();
        assert_eq!(summary.total_steps, 50);
    }

    #[test]
    fn summarize_training_stall_count() {
        // patience=3, threshold=0.001
        // Start at 1.0, plateau for 5 steps (stall detected once)
        let mut history = vec![TrainingEvent::new(0, 0.0, 1.0)];
        for i in 1..6 {
            history.push(TrainingEvent::new(i, i as f32, 0.9999)); // below threshold
        }
        let summary = summarize_training(&history, 0.001, 3).unwrap();
        assert!(summary.stall_count >= 1);
    }

    // -----------------------------------------------------------------------
    // format_training_summary
    // -----------------------------------------------------------------------

    #[test]
    fn format_training_summary_non_empty() {
        let summary = TrainingSummary {
            total_steps: 1000,
            total_secs: 60.0,
            best_loss: 0.1,
            final_loss: 0.15,
            best_step: 800,
            improvement_fraction: 0.5,
            mean_throughput: 16.7,
            stall_count: 0,
        };
        let s = format_training_summary(&summary);
        assert!(!s.is_empty());
        assert!(s.contains("1000") || s.contains("steps"));
    }

    // -----------------------------------------------------------------------
    // robust_smooth_loss
    // -----------------------------------------------------------------------

    #[test]
    fn robust_smooth_loss_empty_error() {
        assert!(matches!(
            robust_smooth_loss(&[], 2.0),
            Err(MonitorError::NoData)
        ));
    }

    #[test]
    fn robust_smooth_loss_single_value() {
        let result = robust_smooth_loss(&[0.5], 2.0).unwrap();
        assert!((result - 0.5).abs() < 1e-5);
    }

    #[test]
    fn robust_smooth_loss_with_outlier() {
        // Most values around 0.5, one outlier at 100.0
        let mut losses = vec![0.5f32; 10];
        losses.push(100.0);
        let result = robust_smooth_loss(&losses, 2.0).unwrap();
        // Should be close to 0.5 after removing the outlier
        assert!(result < 1.0, "Expected result near 0.5, got {result}");
    }

    #[test]
    fn robust_smooth_loss_invalid_k() {
        assert!(matches!(
            robust_smooth_loss(&[0.5, 0.4], 0.0),
            Err(MonitorError::InvalidConfig(_))
        ));
        assert!(matches!(
            robust_smooth_loss(&[0.5, 0.4], -1.0),
            Err(MonitorError::InvalidConfig(_))
        ));
    }
}
