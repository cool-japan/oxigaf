//! Training diagnostics — rich instrumentation for the OxiGAF optimization loop.
//!
//! Provides scalar EMA smoothing, gradient-norm tracking, density-control
//! event recording, and a central [`TrainingDiagnostics`] aggregator that
//! produces both human-readable status lines and full diagnostic reports.
//!
//! All structures are designed to be completely self-contained: no GPU,
//! no real model data, and no external crates beyond `std`.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::fmt::Write as FmtWrite;

// ---------------------------------------------------------------------------
// EmaTracker
// ---------------------------------------------------------------------------

/// Exponential moving average tracker for scalar values.
///
/// The update rule is:
/// ```text
/// ema_new = (1 − alpha) * ema_prev + alpha * raw
/// ```
/// A higher `alpha` tracks the raw signal more closely (less smoothing).
///
/// On the very first update the EMA is seeded with the raw value so that
/// there is no cold-start bias.
#[derive(Debug, Clone)]
pub struct EmaTracker {
    /// Smoothing factor `α ∈ (0, 1)`.  Higher = less smooth.
    pub alpha: f32,
    /// Current EMA value, or `None` until the first update.
    pub value: Option<f32>,
    /// Last raw value passed to [`update`](EmaTracker::update).
    pub raw_last: f32,
    /// Number of times [`update`](EmaTracker::update) has been called.
    pub update_count: u64,
}

impl EmaTracker {
    /// Create a new tracker with the given smoothing factor.
    ///
    /// # Panics (compile-time policy)
    /// Will *not* panic — `alpha` is used as-is; callers should keep it in
    /// `(0, 1)` for sensible behaviour.
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            value: None,
            raw_last: 0.0,
            update_count: 0,
        }
    }

    /// Update the EMA with a new raw observation.
    ///
    /// On the first call the EMA is seeded with `raw` directly.
    /// Returns the new EMA value.
    pub fn update(&mut self, raw: f32) -> f32 {
        self.raw_last = raw;
        self.update_count += 1;
        let new_ema = match self.value {
            None => raw,
            Some(prev) => (1.0 - self.alpha) * prev + self.alpha * raw,
        };
        self.value = Some(new_ema);
        new_ema
    }

    /// Return the current smoothed value.
    ///
    /// Returns `0.0` if [`update`](EmaTracker::update) has never been called.
    pub fn smoothed(&self) -> f32 {
        self.value.unwrap_or(0.0)
    }

    /// Reset the tracker to its initial (no-data) state.
    pub fn reset(&mut self) {
        self.value = None;
        self.raw_last = 0.0;
        self.update_count = 0;
    }
}

impl Default for EmaTracker {
    fn default() -> Self {
        Self::new(0.1)
    }
}

// ---------------------------------------------------------------------------
// GradientNormTracker
// ---------------------------------------------------------------------------

/// Tracks L2 gradient norms per parameter group over a rolling window.
///
/// Each named group stores a fixed-capacity [`VecDeque`] of recent norms.
/// Statistics (mean, max, spike detection) are computed over that window.
#[derive(Debug, Clone)]
pub struct GradientNormTracker {
    /// `group_name → recent L2 norms`.
    pub norms: HashMap<String, VecDeque<f32>>,
    /// Maximum number of entries kept per group.
    pub window_size: usize,
}

impl Default for GradientNormTracker {
    fn default() -> Self {
        Self::new(100)
    }
}

impl GradientNormTracker {
    /// Create a new tracker with the specified rolling-window capacity.
    pub fn new(window_size: usize) -> Self {
        Self {
            norms: HashMap::new(),
            window_size,
        }
    }

    /// Record the L2 gradient norm for a named parameter group.
    ///
    /// If the group does not yet exist it is created automatically.
    /// The oldest entry is dropped when the window is full.
    pub fn record(&mut self, group_name: impl Into<String>, l2_norm: f32) {
        let window = self
            .norms
            .entry(group_name.into())
            .or_insert_with(|| VecDeque::with_capacity(self.window_size));

        if window.len() >= self.window_size {
            window.pop_front();
        }
        window.push_back(l2_norm);
    }

    /// Mean gradient norm for a group over its current window.
    ///
    /// Returns `None` if the group has never been recorded.
    pub fn mean_norm(&self, group_name: &str) -> Option<f32> {
        let window = self.norms.get(group_name)?;
        if window.is_empty() {
            return None;
        }
        let sum: f32 = window.iter().sum();
        Some(sum / window.len() as f32)
    }

    /// Maximum gradient norm for a group over its current window.
    ///
    /// Returns `None` if the group has never been recorded.
    pub fn max_norm(&self, group_name: &str) -> Option<f32> {
        let window = self.norms.get(group_name)?;
        window.iter().copied().reduce(f32::max)
    }

    /// Returns `true` if the *latest* norm for a group is more than
    /// three times the mean of all previous norms in the window.
    ///
    /// Returns `false` if the group is unknown or has fewer than 2 entries.
    pub fn has_gradient_spike(&self, group_name: &str) -> bool {
        let window = match self.norms.get(group_name) {
            Some(w) => w,
            None => return false,
        };
        if window.len() < 2 {
            return false;
        }
        let latest = match window.back() {
            Some(&v) => v,
            None => return false,
        };
        // Mean over all entries *except* the latest.
        let history_len = window.len() - 1;
        let history_sum: f32 = window.iter().take(history_len).sum();
        let history_mean = history_sum / history_len as f32;
        if history_mean <= 0.0 {
            return false;
        }
        latest > 3.0 * history_mean
    }

    /// Return the names of all currently-tracked groups.
    pub fn groups(&self) -> Vec<&String> {
        self.norms.keys().collect()
    }

    /// Format a human-readable table of all groups.
    ///
    /// The table has columns: group | count | mean | max.
    /// Returns an empty string if no groups are tracked yet.
    pub fn format_table(&self) -> String {
        if self.norms.is_empty() {
            return String::new();
        }

        let mut out = String::new();
        // Sort group names for deterministic output.
        let mut names: Vec<&String> = self.norms.keys().collect();
        names.sort();

        let _ = writeln!(
            out,
            "{:<20} {:>6} {:>12} {:>12}",
            "group", "count", "mean", "max"
        );
        let _ = writeln!(out, "{}", "-".repeat(54));

        for name in names {
            let window = match self.norms.get(name) {
                Some(w) => w,
                None => continue,
            };
            let count = window.len();
            let mean = if count == 0 {
                0.0
            } else {
                window.iter().sum::<f32>() / count as f32
            };
            let max = window.iter().copied().reduce(f32::max).unwrap_or(0.0);
            let _ = writeln!(
                out,
                "{:<20} {:>6} {:>12.4e} {:>12.4e}",
                name, count, mean, max
            );
        }
        out
    }
}

// ---------------------------------------------------------------------------
// DensityStats
// ---------------------------------------------------------------------------

/// Statistics for a single density-control operation.
#[derive(Debug, Clone, Default)]
pub struct DensityStats {
    /// Training iteration at which this event occurred.
    pub iteration: u64,
    /// Number of Gaussians before the operation.
    pub num_gaussians_before: usize,
    /// Number of Gaussians after the operation.
    pub num_gaussians_after: usize,
    /// Gaussians added by cloning (under-reconstructed regions).
    pub num_cloned: usize,
    /// Gaussians added by splitting (over-reconstructed regions).
    pub num_split: usize,
    /// Gaussians removed by pruning (opacity / size threshold).
    pub num_pruned: usize,
    /// Gaussians whose opacity was reset to encourage diversity.
    pub num_opacity_reset: usize,
}

impl DensityStats {
    /// Net change in Gaussian count (positive = growth, negative = shrinkage).
    pub fn net_change(&self) -> i64 {
        self.num_gaussians_after as i64 - self.num_gaussians_before as i64
    }

    /// Returns `true` if this event included any cloning or splitting.
    pub fn was_densification(&self) -> bool {
        self.num_cloned > 0 || self.num_split > 0
    }

    /// Returns `true` if this event included any pruning.
    pub fn was_pruning(&self) -> bool {
        self.num_pruned > 0
    }
}

// ---------------------------------------------------------------------------
// LossTracker
// ---------------------------------------------------------------------------

/// Per-loss-component tracker with EMA smoothing and total-loss history.
#[derive(Debug, Clone)]
pub struct LossTracker {
    /// Component name → EMA tracker.
    pub components: HashMap<String, EmaTracker>,
    /// Rolling history of total loss values.
    pub total_loss_history: VecDeque<f32>,
    /// Maximum number of total-loss entries kept.
    pub history_size: usize,
    /// EMA smoothing factor shared across all component trackers.
    alpha: f32,
}

impl LossTracker {
    /// Create a new tracker.
    ///
    /// * `alpha` — EMA smoothing factor for component trackers.
    /// * `history_size` — rolling window length for total-loss history.
    pub fn new(alpha: f32, history_size: usize) -> Self {
        Self {
            components: HashMap::new(),
            total_loss_history: VecDeque::with_capacity(history_size),
            history_size,
            alpha,
        }
    }

    /// Record a value for the named loss component.
    ///
    /// The component tracker is created on first use.
    /// Returns the EMA-smoothed value.
    pub fn record(&mut self, component: impl Into<String>, value: f32) -> f32 {
        let alpha = self.alpha;
        let tracker = self
            .components
            .entry(component.into())
            .or_insert_with(|| EmaTracker::new(alpha));
        tracker.update(value)
    }

    /// Record the total loss for this iteration.
    ///
    /// Older entries are dropped when the history window is full.
    pub fn record_total(&mut self, total: f32) {
        if self.total_loss_history.len() >= self.history_size {
            self.total_loss_history.pop_front();
        }
        self.total_loss_history.push_back(total);
    }

    /// Return the current EMA-smoothed value for the named component.
    ///
    /// Returns `None` if the component has never been recorded.
    pub fn smoothed(&self, component: &str) -> Option<f32> {
        let tracker = self.components.get(component)?;
        tracker.value
    }

    /// Mean total loss over the current rolling window.
    ///
    /// Returns `None` if no total losses have been recorded yet.
    pub fn mean_total_loss(&self) -> Option<f32> {
        if self.total_loss_history.is_empty() {
            return None;
        }
        let sum: f32 = self.total_loss_history.iter().sum();
        Some(sum / self.total_loss_history.len() as f32)
    }

    /// Whether the training is converging.
    ///
    /// Converging is defined as the mean of the **second half** of the
    /// total-loss history being strictly lower than the mean of the first half.
    /// Returns `false` when fewer than 2 entries are in the history.
    pub fn is_converging(&self) -> bool {
        let n = self.total_loss_history.len();
        if n < 2 {
            return false;
        }
        let mid = n / 2;
        let first_half: Vec<f32> = self.total_loss_history.iter().take(mid).copied().collect();
        let second_half: Vec<f32> = self.total_loss_history.iter().skip(mid).copied().collect();

        let mean_first = first_half.iter().sum::<f32>() / first_half.len() as f32;
        let mean_second = second_half.iter().sum::<f32>() / second_half.len() as f32;

        mean_second < mean_first
    }

    /// Format a compact one-liner summarising tracked components.
    pub fn format_line(&self) -> String {
        let mut parts: Vec<String> = Vec::new();

        if let Some(total) = self.mean_total_loss() {
            parts.push(format!("total={:.4}", total));
        }

        let mut names: Vec<&String> = self.components.keys().collect();
        names.sort();
        for name in names {
            if let Some(tracker) = self.components.get(name) {
                if let Some(v) = tracker.value {
                    parts.push(format!("{}={:.4}", name, v));
                }
            }
        }

        if parts.is_empty() {
            return String::from("(no loss data)");
        }
        parts.join("  ")
    }

    /// Return the names of all tracked loss components.
    pub fn component_names(&self) -> Vec<&String> {
        self.components.keys().collect()
    }
}

// ---------------------------------------------------------------------------
// TrainingDiagnostics
// ---------------------------------------------------------------------------

/// Comprehensive training diagnostics aggregator.
///
/// Aggregates loss tracking, gradient-norm tracking, density-control events,
/// and timing information.  The entry point for console-level reporting
/// during a training loop.
pub struct TrainingDiagnostics {
    /// Smoothed per-component loss tracker.
    pub loss_tracker: LossTracker,
    /// Per-group gradient-norm tracker.
    pub grad_norms: GradientNormTracker,
    /// Recent density-control events.
    pub density_history: VecDeque<DensityStats>,
    /// Maximum number of density events to retain.
    pub density_history_size: usize,
    /// Current training iteration.
    pub current_iteration: u64,
    /// Time at which this diagnostics object was created.
    pub start_time: std::time::Instant,
}

impl TrainingDiagnostics {
    /// Create a fresh diagnostics object.
    ///
    /// Defaults: EMA alpha = 0.1, loss history size = 200,
    /// gradient-norm window = 100, density history size = 50.
    pub fn new() -> Self {
        Self {
            loss_tracker: LossTracker::new(0.1, 200),
            grad_norms: GradientNormTracker::new(100),
            density_history: VecDeque::with_capacity(50),
            density_history_size: 50,
            current_iteration: 0,
            start_time: std::time::Instant::now(),
        }
    }

    /// Advance the iteration counter by one.
    pub fn next_iteration(&mut self) {
        self.current_iteration += 1;
    }

    /// Record a batch of loss components for the current iteration.
    ///
    /// `components` is a slice of `(name, value)` pairs.
    /// The total loss is derived as the sum of all provided components.
    pub fn record_losses(&mut self, components: &[(&str, f32)]) {
        let mut total = 0.0_f32;
        for &(name, value) in components {
            self.loss_tracker.record(name, value);
            total += value;
        }
        if !components.is_empty() {
            self.loss_tracker.record_total(total);
        }
    }

    /// Record a batch of gradient norms.
    ///
    /// `norms` is a slice of `(group_name, l2_norm)` pairs.
    pub fn record_grad_norms(&mut self, norms: &[(&str, f32)]) {
        for &(name, norm) in norms {
            self.grad_norms.record(name, norm);
        }
    }

    /// Append a density-control event to the history.
    ///
    /// Oldest events are dropped when the history is at capacity.
    pub fn record_density_event(&mut self, stats: DensityStats) {
        if self.density_history.len() >= self.density_history_size {
            self.density_history.pop_front();
        }
        self.density_history.push_back(stats);
    }

    /// Most recent density-control event, or `None` if none have occurred.
    pub fn latest_density(&self) -> Option<&DensityStats> {
        self.density_history.back()
    }

    /// Wall-clock time since this diagnostics object was created.
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Iterations per second since creation.
    pub fn iterations_per_second(&self) -> f64 {
        let secs = self.elapsed().as_secs_f64();
        if secs > 0.0 {
            self.current_iteration as f64 / secs
        } else {
            0.0
        }
    }

    /// Generate a full multi-line diagnostic report.
    pub fn format_report(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "=== OxiGAF Training Diagnostics ===");
        let _ = writeln!(out, "Iteration : {}", self.current_iteration);
        let elapsed = self.elapsed();
        let _ = writeln!(
            out,
            "Elapsed   : {:.1}s  ({:.2} iter/s)",
            elapsed.as_secs_f64(),
            self.iterations_per_second()
        );

        // Loss section
        let _ = writeln!(out, "\n--- Losses ---");
        let _ = writeln!(out, "{}", self.loss_tracker.format_line());
        let converging = if self.loss_tracker.is_converging() {
            "yes"
        } else {
            "no"
        };
        let _ = writeln!(out, "Converging: {}", converging);

        // Gradient norms section
        if !self.grad_norms.norms.is_empty() {
            let _ = writeln!(out, "\n--- Gradient Norms ---");
            let _ = write!(out, "{}", self.grad_norms.format_table());
        }

        // Density section
        if let Some(d) = self.latest_density() {
            let _ = writeln!(out, "\n--- Last Density Event (iter {}) ---", d.iteration);
            let _ = writeln!(
                out,
                "  Gaussians: {} → {} (net {:+})",
                d.num_gaussians_before,
                d.num_gaussians_after,
                d.net_change()
            );
            let _ = writeln!(
                out,
                "  Cloned={} Split={} Pruned={} OpacityReset={}",
                d.num_cloned, d.num_split, d.num_pruned, d.num_opacity_reset
            );
        }

        out
    }

    /// Generate a compact one-line status string for console progress output.
    ///
    /// Guaranteed to fit within 120 characters under typical usage.
    pub fn format_status_line(&self) -> String {
        let iter = self.current_iteration;
        let its = self.iterations_per_second();
        let loss_part = self
            .loss_tracker
            .mean_total_loss()
            .map(|v| format!(" loss={:.4}", v))
            .unwrap_or_default();
        let conv = if self.loss_tracker.is_converging() {
            " ↓"
        } else {
            ""
        };
        format!("iter={} {:.1}it/s{}{}", iter, its, loss_part, conv)
    }
}

impl Default for TrainingDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------ EmaTracker

    #[test]
    fn ema_tracker_first_update_equals_raw() {
        let mut t = EmaTracker::new(0.1);
        let result = t.update(5.0);
        // First call seeds with raw value.
        assert!((result - 5.0).abs() < 1e-6, "expected 5.0, got {result}");
    }

    #[test]
    fn ema_tracker_second_update_converges_toward_new_value() {
        let mut t = EmaTracker::new(0.5);
        t.update(0.0);
        let v = t.update(10.0);
        // ema = 0.5 * 0.0 + 0.5 * 10.0 = 5.0
        assert!((v - 5.0).abs() < 1e-5, "expected 5.0, got {v}");
    }

    #[test]
    fn ema_tracker_smoothed_returns_last_ema() {
        let mut t = EmaTracker::new(0.3);
        let v = t.update(2.0);
        assert!((t.smoothed() - v).abs() < 1e-6);
    }

    #[test]
    fn ema_tracker_reset_clears_value() {
        let mut t = EmaTracker::new(0.1);
        t.update(99.0);
        t.reset();
        assert!(t.value.is_none(), "value should be None after reset");
        assert_eq!(t.update_count, 0);
    }

    #[test]
    fn ema_tracker_smoothed_returns_zero_when_no_data() {
        let t = EmaTracker::new(0.1);
        assert!((t.smoothed() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn ema_tracker_multiple_updates_converge() {
        let mut t = EmaTracker::new(0.2);
        // Seed at 100, then pull toward 0 with many updates.
        t.update(100.0);
        for _ in 0..200 {
            t.update(0.0);
        }
        let v = t.smoothed();
        assert!(v < 0.01, "expected near 0, got {v}");
    }

    // ------------------------------------------------------------------ GradientNormTracker

    #[test]
    fn grad_norm_tracker_record_adds_entry() {
        let mut g = GradientNormTracker::new(10);
        g.record("pos", 1.0);
        assert_eq!(g.norms["pos"].len(), 1);
    }

    #[test]
    fn grad_norm_tracker_mean_correct_after_three_records() {
        let mut g = GradientNormTracker::new(10);
        g.record("pos", 1.0);
        g.record("pos", 2.0);
        g.record("pos", 3.0);
        let mean = g.mean_norm("pos").unwrap_or(0.0);
        assert!((mean - 2.0).abs() < 1e-5, "expected mean 2.0, got {mean}");
    }

    #[test]
    fn grad_norm_tracker_spike_detected() {
        let mut g = GradientNormTracker::new(20);
        // Fill with stable baseline norms.
        for _ in 0..10 {
            g.record("grad", 1.0);
        }
        // Add a large spike.
        g.record("grad", 100.0);
        assert!(
            g.has_gradient_spike("grad"),
            "should detect spike when latest >> mean"
        );
    }

    #[test]
    fn grad_norm_tracker_no_spike_for_stable_gradients() {
        let mut g = GradientNormTracker::new(20);
        for _ in 0..10 {
            g.record("grad", 1.0);
        }
        g.record("grad", 1.05); // within 3× mean
        assert!(
            !g.has_gradient_spike("grad"),
            "should not detect spike for stable gradients"
        );
    }

    #[test]
    fn grad_norm_tracker_format_table_nonempty() {
        let mut g = GradientNormTracker::new(5);
        g.record("pos", 0.5);
        g.record("rot", 0.3);
        let table = g.format_table();
        assert!(!table.is_empty(), "format_table should be non-empty");
        assert!(table.contains("pos"));
        assert!(table.contains("rot"));
    }

    #[test]
    fn grad_norm_tracker_max_norm_correct() {
        let mut g = GradientNormTracker::new(10);
        g.record("sh", 0.5);
        g.record("sh", 2.5);
        g.record("sh", 1.0);
        let max = g.max_norm("sh");
        assert!(max.is_some());
        assert!((max.unwrap_or(0.0) - 2.5).abs() < 1e-5);
    }

    #[test]
    fn grad_norm_tracker_unknown_group_returns_none() {
        let g = GradientNormTracker::new(10);
        assert!(g.mean_norm("unknown").is_none());
        assert!(g.max_norm("unknown").is_none());
    }

    // ------------------------------------------------------------------ DensityStats

    #[test]
    fn density_stats_net_change_positive_when_more_gaussians() {
        let d = DensityStats {
            num_gaussians_before: 100,
            num_gaussians_after: 150,
            num_cloned: 50,
            ..Default::default()
        };
        assert_eq!(d.net_change(), 50);
    }

    #[test]
    fn density_stats_was_densification_true_when_cloned() {
        let d = DensityStats {
            num_cloned: 5,
            ..Default::default()
        };
        assert!(d.was_densification());
    }

    #[test]
    fn density_stats_was_densification_true_when_split() {
        let d = DensityStats {
            num_split: 3,
            ..Default::default()
        };
        assert!(d.was_densification());
    }

    #[test]
    fn density_stats_was_pruning_true_when_pruned() {
        let d = DensityStats {
            num_pruned: 10,
            ..Default::default()
        };
        assert!(d.was_pruning());
    }

    #[test]
    fn density_stats_was_densification_false_when_neither() {
        let d = DensityStats::default();
        assert!(!d.was_densification());
    }

    // ------------------------------------------------------------------ LossTracker

    #[test]
    fn loss_tracker_record_returns_ema_value() {
        let mut t = LossTracker::new(0.1, 100);
        let v = t.record("l1", 1.0);
        // First update: EMA = raw.
        assert!((v - 1.0).abs() < 1e-6);
    }

    #[test]
    fn loss_tracker_smoothed_none_for_unknown_component() {
        let t = LossTracker::new(0.1, 100);
        assert!(t.smoothed("nonexistent").is_none());
    }

    #[test]
    fn loss_tracker_mean_total_loss_none_for_empty() {
        let t = LossTracker::new(0.1, 50);
        assert!(t.mean_total_loss().is_none());
    }

    #[test]
    fn loss_tracker_mean_total_loss_correct() {
        let mut t = LossTracker::new(0.1, 10);
        t.record_total(1.0);
        t.record_total(3.0);
        let mean = t.mean_total_loss().unwrap_or(0.0);
        assert!((mean - 2.0).abs() < 1e-5, "expected 2.0, got {mean}");
    }

    #[test]
    fn loss_tracker_is_converging_true_when_decreasing() {
        let mut t = LossTracker::new(0.1, 20);
        // First half: higher losses.
        for i in (10..20usize).rev() {
            t.record_total(i as f32 * 0.1 + 1.0);
        }
        // Second half: lower losses.
        for i in 0..10usize {
            t.record_total(i as f32 * 0.01);
        }
        assert!(
            t.is_converging(),
            "should converge when second half < first half"
        );
    }

    #[test]
    fn loss_tracker_is_converging_false_when_constant() {
        let mut t = LossTracker::new(0.1, 10);
        for _ in 0..10 {
            t.record_total(1.0);
        }
        assert!(!t.is_converging(), "constant loss should not be converging");
    }

    #[test]
    fn loss_tracker_format_line_nonempty() {
        let mut t = LossTracker::new(0.1, 50);
        t.record("l1", 0.5);
        t.record_total(0.5);
        let line = t.format_line();
        assert!(!line.is_empty());
    }

    #[test]
    fn loss_tracker_component_names_returns_all() {
        let mut t = LossTracker::new(0.1, 50);
        t.record("l1", 1.0);
        t.record("ssim", 0.5);
        let mut names: Vec<&String> = t.component_names();
        names.sort();
        assert_eq!(names.len(), 2);
    }

    // ------------------------------------------------------------------ TrainingDiagnostics

    #[test]
    fn diagnostics_next_iteration_increments_counter() {
        let mut d = TrainingDiagnostics::new();
        assert_eq!(d.current_iteration, 0);
        d.next_iteration();
        assert_eq!(d.current_iteration, 1);
        d.next_iteration();
        assert_eq!(d.current_iteration, 2);
    }

    #[test]
    fn diagnostics_record_losses_updates_tracker() {
        let mut d = TrainingDiagnostics::new();
        d.record_losses(&[("l1", 0.5), ("ssim", 0.3)]);
        assert!(d.loss_tracker.smoothed("l1").is_some());
        assert!(d.loss_tracker.smoothed("ssim").is_some());
    }

    #[test]
    fn diagnostics_record_grad_norms_updates_tracker() {
        let mut d = TrainingDiagnostics::new();
        d.record_grad_norms(&[("position", 0.01), ("rotation", 0.005)]);
        assert!(d.grad_norms.mean_norm("position").is_some());
        assert!(d.grad_norms.mean_norm("rotation").is_some());
    }

    #[test]
    fn diagnostics_record_density_event_appends() {
        let mut d = TrainingDiagnostics::new();
        let stats = DensityStats {
            iteration: 100,
            num_gaussians_before: 500,
            num_gaussians_after: 600,
            num_cloned: 100,
            ..Default::default()
        };
        d.record_density_event(stats);
        let latest = d.latest_density();
        assert!(latest.is_some());
        assert_eq!(latest.unwrap_or(&DensityStats::default()).iteration, 100);
    }

    #[test]
    fn diagnostics_format_report_nonempty() {
        let mut d = TrainingDiagnostics::new();
        d.next_iteration();
        d.record_losses(&[("l1", 0.4)]);
        let report = d.format_report();
        assert!(!report.is_empty());
        assert!(report.contains("Iteration"));
    }

    #[test]
    fn diagnostics_format_status_line_under_120_chars() {
        let mut d = TrainingDiagnostics::new();
        for _ in 0..1000 {
            d.next_iteration();
        }
        d.record_losses(&[("l1", 0.123), ("ssim", 0.456)]);
        let line = d.format_status_line();
        assert!(
            line.len() < 120,
            "status line too long ({} chars): {}",
            line.len(),
            line
        );
    }

    #[test]
    fn diagnostics_density_history_capped() {
        let mut d = TrainingDiagnostics::new();
        d.density_history_size = 3;
        for i in 0..10u64 {
            d.record_density_event(DensityStats {
                iteration: i,
                ..Default::default()
            });
        }
        // Only the most recent 3 should be kept.
        assert_eq!(d.density_history.len(), 3);
        let latest_iter = d.latest_density().map(|s| s.iteration).unwrap_or(0);
        assert_eq!(latest_iter, 9);
    }

    #[test]
    fn diagnostics_iterations_per_second_nonnegative() {
        let d = TrainingDiagnostics::new();
        assert!(d.iterations_per_second() >= 0.0);
    }
}
