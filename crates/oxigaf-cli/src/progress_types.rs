//! Higher-level progress display types for CLI workflows.
//!
//! This module provides structured progress wrappers for training loops
//! ([`TrainingProgress`]), single-step indeterminate operations
//! ([`OperationSpinner`]), multi-step batch jobs ([`BatchProgress`]), and
//! per-component timing collection ([`TimingReport`]).
//!
//! These types are part of the library API exposed through `oxigaf-cli`'s
//! `lib.rs`, but are intentionally not included in the binary's module tree
//! so that the `dead_code` lint does not fire when they are only consumed
//! from integration tests.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

// ---------------------------------------------------------------------------
// TrainingProgress — multi-bar display for training loops
// ---------------------------------------------------------------------------

/// Multi-bar progress display for training.
///
/// Combines a main iteration bar, a loss display bar, and a timing bar
/// into a single cohesive terminal UI backed by indicatif's
/// [`MultiProgress`].
///
/// # Example
///
/// ```no_run
/// use oxigaf_cli::progress_types::TrainingProgress;
///
/// let tp = TrainingProgress::new(10_000);
/// for i in 0..10_000u64 {
///     tp.update(i + 1, 0.45, 0.30, 0.10, 12.5, 60_000);
/// }
/// tp.finish();
/// ```
pub struct TrainingProgress {
    multi: Arc<MultiProgress>,
    total_bar: ProgressBar,
    loss_bar: ProgressBar,
    timing_bar: ProgressBar,
    start_time: Instant,
    total_iterations: u64,
}

impl TrainingProgress {
    /// Create a new progress display for training with `total_iterations`.
    #[must_use]
    pub fn new(total_iterations: u64) -> Self {
        let multi = Arc::new(MultiProgress::new());

        let total_bar = multi.add(ProgressBar::new(total_iterations));
        total_bar.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=>-"),
        );

        let loss_bar = multi.add(ProgressBar::new_spinner());
        loss_bar.set_style(
            ProgressStyle::with_template("{spinner:.green} Loss: {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_spinner()),
        );

        let timing_bar = multi.add(ProgressBar::new_spinner());
        timing_bar.set_style(
            ProgressStyle::with_template("{spinner:.blue} {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_spinner()),
        );

        Self {
            multi,
            total_bar,
            loss_bar,
            timing_bar,
            start_time: Instant::now(),
            total_iterations,
        }
    }

    /// Update progress after an iteration.
    pub fn update(
        &self,
        iteration: u64,
        total_loss: f32,
        photometric_loss: f32,
        ssim_loss: f32,
        iter_duration_ms: f64,
        num_gaussians: usize,
    ) {
        self.total_bar.set_position(iteration);
        self.total_bar.set_message(format!(
            "loss={:.4} | gaussians={}",
            total_loss, num_gaussians
        ));

        self.loss_bar.set_message(format!(
            "total={:.4}  photo={:.4}  ssim={:.4}",
            total_loss, photometric_loss, ssim_loss
        ));

        let eta = self.eta(iteration, self.total_iterations);
        self.timing_bar.set_message(format!(
            "iter={:.1}ms  ETA={}s",
            iter_duration_ms,
            eta.as_secs()
        ));

        self.loss_bar.tick();
        self.timing_bar.tick();
    }

    /// Show a status message without disrupting bars.
    pub fn message(&self, msg: &str) {
        self.multi.println(msg).unwrap_or(());
    }

    /// Mark training as complete.
    pub fn finish(&self) {
        let elapsed = self.start_time.elapsed();
        self.total_bar.finish_with_message(format!(
            "Training complete in {:.1}s",
            elapsed.as_secs_f64()
        ));
        self.loss_bar.finish_and_clear();
        self.timing_bar.finish_and_clear();
    }

    /// Estimated time remaining.
    #[must_use]
    pub fn eta(&self, current: u64, total: u64) -> Duration {
        if current == 0 || current >= total {
            return Duration::ZERO;
        }
        let elapsed = self.start_time.elapsed();
        let rate = elapsed.as_secs_f64() / current as f64;
        let remaining = (total - current) as f64 * rate;
        Duration::from_secs_f64(remaining)
    }
}

// ---------------------------------------------------------------------------
// OperationSpinner — spinner for single-step operations
// ---------------------------------------------------------------------------

/// Spinner for a single-step operation of indeterminate length.
///
/// # Example
///
/// ```no_run
/// use oxigaf_cli::progress_types::OperationSpinner;
///
/// let sp = OperationSpinner::new("Loading model weights...");
/// // ... do work ...
/// sp.finish_ok();
/// ```
pub struct OperationSpinner {
    bar: ProgressBar,
    start: Instant,
}

impl OperationSpinner {
    /// Create a new spinner with `message`.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        let bar = ProgressBar::new_spinner();
        bar.set_style(
            ProgressStyle::with_template("{spinner:.blue} {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_spinner()),
        );
        bar.set_message(message.into());
        Self {
            bar,
            start: Instant::now(),
        }
    }

    /// Finish the spinner with a custom message.
    pub fn finish_with_message(&self, msg: impl Into<String>) {
        self.bar.finish_with_message(msg.into());
    }

    /// Finish indicating success (appends elapsed time).
    pub fn finish_ok(&self) {
        let elapsed = self.start.elapsed();
        self.bar
            .finish_with_message(format!("done ({:.2}s)", elapsed.as_secs_f64()));
    }

    /// Finish indicating failure with `msg`.
    pub fn fail(&self, msg: impl Into<String>) {
        self.bar
            .abandon_with_message(format!("FAILED: {}", msg.into()));
    }
}

// ---------------------------------------------------------------------------
// BatchProgress — progress bar for multi-step batch operations
// ---------------------------------------------------------------------------

/// Multi-step operation progress (e.g., multi-view render batch).
///
/// # Example
///
/// ```no_run
/// use oxigaf_cli::progress_types::BatchProgress;
///
/// let bp = BatchProgress::new(100, "Rendering frames");
/// for _ in 0..100 {
///     bp.increment();
/// }
/// bp.finish();
/// ```
pub struct BatchProgress {
    bar: ProgressBar,
}

impl BatchProgress {
    /// Create a new batch progress bar with `total` items and a `message`.
    #[must_use]
    pub fn new(total: u64, message: impl Into<String>) -> Self {
        let bar = ProgressBar::new(total);
        bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} | {msg}",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=>-"),
        );
        bar.set_message(message.into());
        Self { bar }
    }

    /// Increment progress by 1.
    pub fn increment(&self) {
        self.bar.inc(1);
    }

    /// Increment progress by `n`.
    pub fn increment_by(&self, n: u64) {
        self.bar.inc(n);
    }

    /// Finish the batch progress bar.
    pub fn finish(&self) {
        self.bar.finish_with_message("done");
    }

    /// Update the displayed message without changing position.
    pub fn set_message(&self, msg: impl Into<String>) {
        self.bar.set_message(msg.into());
    }
}

// ---------------------------------------------------------------------------
// TimingReport — component-level timing tracker
// ---------------------------------------------------------------------------

/// Timing tracker for performance reporting.
///
/// Records duration per named component and can format a summary table.
///
/// # Example
///
/// ```
/// use std::time::Duration;
/// use oxigaf_cli::progress_types::TimingReport;
///
/// let mut report = TimingReport::new();
/// report.record("rasterizer", Duration::from_millis(300));
/// report.record("loss", Duration::from_millis(100));
/// println!("{}", report.format_table());
/// ```
#[derive(Debug, Default)]
pub struct TimingReport {
    /// Per-component durations.
    pub component_times: HashMap<String, Duration>,
}

impl TimingReport {
    /// Create a new empty timing report.
    #[must_use]
    pub fn new() -> Self {
        Self {
            component_times: HashMap::new(),
        }
    }

    /// Record time for a named component.
    pub fn record(&mut self, name: impl Into<String>, duration: Duration) {
        self.component_times.insert(name.into(), duration);
    }

    /// Record a timed block by name using a closure.
    ///
    /// The closure's return value is passed through unchanged.
    pub fn time<T, F: FnOnce() -> T>(&mut self, name: impl Into<String>, f: F) -> T {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        self.record(name, duration);
        result
    }

    /// Format timing data as a human-readable table string.
    ///
    /// Components are sorted alphabetically for deterministic output.
    #[must_use]
    pub fn format_table(&self) -> String {
        if self.component_times.is_empty() {
            return String::new();
        }

        let total = self.total();
        let total_ms = total.as_secs_f64() * 1_000.0;

        let mut lines = Vec::new();
        lines.push(format!(
            "{:<20} {:>10} {:>8}",
            "Component", "Time (ms)", "% Total"
        ));
        lines.push(format!("{:-<40}", ""));

        let mut sorted: Vec<(&String, &Duration)> = self.component_times.iter().collect();
        sorted.sort_by_key(|(k, _)| k.as_str());

        for (name, dur) in &sorted {
            let ms = dur.as_secs_f64() * 1_000.0;
            let pct = if total_ms > 0.0 {
                ms / total_ms * 100.0
            } else {
                0.0
            };
            lines.push(format!("{:<20} {:>10.2} {:>7.1}%", name, ms, pct));
        }

        lines.push(format!("{:-<40}", ""));
        lines.push(format!("{:<20} {:>10.2}", "TOTAL", total_ms));

        lines.join("\n")
    }

    /// Total time across all components.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.component_times.values().sum()
    }

    /// Component as percentage of total (0.0–100.0).
    ///
    /// Returns 0.0 if the component is not found or if total is zero.
    #[must_use]
    pub fn percentage(&self, name: &str) -> f64 {
        let total = self.total();
        if total.is_zero() {
            return 0.0;
        }
        let component = match self.component_times.get(name) {
            Some(d) => d,
            None => return 0.0,
        };
        component.as_secs_f64() / total.as_secs_f64() * 100.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // TimingReport tests
    // -----------------------------------------------------------------------

    #[test]
    fn timing_report_new_starts_empty() {
        let report = TimingReport::new();
        assert!(report.component_times.is_empty());
    }

    #[test]
    fn timing_report_record_stores_duration() {
        let mut report = TimingReport::new();
        let dur = Duration::from_millis(250);
        report.record("rasterizer", dur);
        assert_eq!(report.component_times.get("rasterizer"), Some(&dur));
    }

    #[test]
    fn timing_report_time_returns_closure_result_and_stores_timing() {
        let mut report = TimingReport::new();
        let result = report.time("compute", || 42u32);
        assert_eq!(result, 42);
        assert!(report.component_times.contains_key("compute"));
    }

    #[test]
    fn timing_report_total_sums_all_components() {
        let mut report = TimingReport::new();
        report.record("a", Duration::from_millis(100));
        report.record("b", Duration::from_millis(200));
        report.record("c", Duration::from_millis(300));
        assert_eq!(report.total(), Duration::from_millis(600));
    }

    #[test]
    fn timing_report_percentage_in_range() {
        let mut report = TimingReport::new();
        report.record("fast", Duration::from_millis(100));
        report.record("slow", Duration::from_millis(400));
        let pct = report.percentage("fast");
        // fast is 100 out of 500 total = 20%
        assert!((pct - 20.0).abs() < 1e-6, "Expected ~20%, got {pct}");
        assert!((0.0..=100.0).contains(&pct));
    }

    #[test]
    fn timing_report_format_table_non_empty_for_non_empty_report() {
        let mut report = TimingReport::new();
        report.record("loader", Duration::from_millis(50));
        let table = report.format_table();
        assert!(
            !table.is_empty(),
            "format_table() should produce non-empty output"
        );
    }

    #[test]
    fn timing_report_format_table_contains_component_names() {
        let mut report = TimingReport::new();
        report.record("shader_pass", Duration::from_millis(300));
        report.record("sort", Duration::from_millis(100));
        let table = report.format_table();
        assert!(
            table.contains("shader_pass"),
            "Table should contain 'shader_pass'"
        );
        assert!(table.contains("sort"), "Table should contain 'sort'");
    }

    #[test]
    fn timing_report_percentage_returns_zero_for_unknown_component() {
        let mut report = TimingReport::new();
        report.record("known", Duration::from_millis(100));
        let pct = report.percentage("unknown");
        assert_eq!(pct, 0.0);
    }

    #[test]
    fn timing_report_percentage_returns_zero_when_all_zero() {
        let report = TimingReport::new();
        let pct = report.percentage("anything");
        assert_eq!(pct, 0.0);
    }

    // -----------------------------------------------------------------------
    // TrainingProgress tests
    // -----------------------------------------------------------------------

    #[test]
    fn training_progress_eta_returns_zero_when_complete() {
        let tp = TrainingProgress::new(1000);
        let eta = tp.eta(1000, 1000);
        assert_eq!(
            eta,
            Duration::ZERO,
            "ETA should be zero when current == total"
        );
    }

    #[test]
    fn training_progress_eta_returns_zero_for_zero_current() {
        let tp = TrainingProgress::new(1000);
        let eta = tp.eta(0, 1000);
        assert_eq!(
            eta,
            Duration::ZERO,
            "ETA should be zero when no progress made"
        );
    }

    // -----------------------------------------------------------------------
    // OperationSpinner tests
    // -----------------------------------------------------------------------

    #[test]
    fn operation_spinner_new_creates_without_panic() {
        let _sp = OperationSpinner::new("Loading...");
        // If we reach here, creation succeeded
    }

    #[test]
    fn operation_spinner_finish_ok_does_not_panic() {
        let sp = OperationSpinner::new("Working...");
        sp.finish_ok();
    }

    #[test]
    fn operation_spinner_fail_does_not_panic() {
        let sp = OperationSpinner::new("Working...");
        sp.fail("something went wrong");
    }

    // -----------------------------------------------------------------------
    // BatchProgress tests
    // -----------------------------------------------------------------------

    #[test]
    fn batch_progress_new_creates_without_panic() {
        let _bp = BatchProgress::new(100, "Processing frames");
        // If we reach here, creation succeeded
    }

    #[test]
    fn batch_progress_increment_does_not_panic() {
        let bp = BatchProgress::new(10, "test");
        bp.increment();
        bp.increment_by(3);
    }

    #[test]
    fn batch_progress_finish_does_not_panic() {
        let bp = BatchProgress::new(5, "test batch");
        bp.finish();
    }

    #[test]
    fn batch_progress_set_message_does_not_panic() {
        let bp = BatchProgress::new(5, "initial message");
        bp.set_message("updated message");
    }
}
