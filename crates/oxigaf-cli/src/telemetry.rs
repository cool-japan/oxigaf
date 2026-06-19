//! Performance telemetry for OxiGAF training runs.
//!
//! Provides systematic collection and analysis of timing, memory, and
//! throughput metrics. Used to identify bottlenecks, track performance
//! regressions, and generate profiling reports for optimization decisions.
//!
//! # Design
//!
//! Events are stored in a fixed-capacity ring buffer. When the buffer is
//! full, new events overwrite the oldest ones. All statistics are computed
//! on-demand — no background threads or allocations beyond the event buffer.
//!
//! Timing uses [`std::time::Instant`] only (no rand, no ndarray).

use std::time::Instant;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors that can occur during telemetry collection or analysis.
#[derive(Debug, Error)]
pub enum TelemetryError {
    /// An event with the given label was not found.
    #[error("event not found: {0}")]
    EventNotFound(String),

    /// A timer was started again for an already-running label.
    #[error("timer already started for: {0}")]
    TimerAlreadyStarted(String),

    /// A timer was stopped but never started for the given label.
    #[error("timer not started for: {0}")]
    TimerNotStarted(String),

    /// A configuration value is invalid.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// The ring buffer overflowed (informational — collector wraps automatically).
    #[error("buffer overflow: max {max} events, got {current}")]
    BufferOverflow { max: usize, current: usize },
}

// ---------------------------------------------------------------------------
// TelemetryCategory
// ---------------------------------------------------------------------------

/// Classifies what kind of work a telemetry event measures.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TelemetryCategory {
    /// Full training step.
    Step,
    /// Forward pass.
    Forward,
    /// Backward pass / gradient computation.
    Backward,
    /// Optimizer step.
    Optimizer,
    /// Gaussian clone / split / prune (densification).
    Densification,
    /// Rendering.
    Render,
    /// Data loading.
    DataLoad,
    /// Checkpoint save or load.
    Checkpoint,
    /// Loss computation.
    Loss,
    /// User-defined category.
    Custom(String),
}

impl TelemetryCategory {
    /// Return a human-readable string for this category.
    pub fn as_str(&self) -> &str {
        match self {
            TelemetryCategory::Step => "step",
            TelemetryCategory::Forward => "forward",
            TelemetryCategory::Backward => "backward",
            TelemetryCategory::Optimizer => "optimizer",
            TelemetryCategory::Densification => "densification",
            TelemetryCategory::Render => "render",
            TelemetryCategory::DataLoad => "data_load",
            TelemetryCategory::Checkpoint => "checkpoint",
            TelemetryCategory::Loss => "loss",
            TelemetryCategory::Custom(s) => s.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// TelemetryEvent
// ---------------------------------------------------------------------------

/// A single recorded timing event.
#[derive(Debug, Clone)]
pub struct TelemetryEvent {
    /// The category of work this event measures.
    pub category: TelemetryCategory,
    /// A short label identifying the specific operation.
    pub label: String,
    /// Duration of the event in microseconds.
    pub duration_us: u64,
    /// Training step at which this event was recorded.
    pub step: usize,
    /// Optional key-value metadata (e.g. `("n_gaussians", 100_000.0)`).
    pub metadata: Vec<(String, f64)>,
}

impl TelemetryEvent {
    /// Duration in milliseconds.
    pub fn duration_ms(&self) -> f64 {
        self.duration_us as f64 / 1_000.0
    }

    /// Duration in seconds.
    pub fn duration_s(&self) -> f64 {
        self.duration_us as f64 / 1_000_000.0
    }
}

// ---------------------------------------------------------------------------
// TelemetryConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`TelemetryCollector`].
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Ring buffer capacity (default `100_000`).
    pub max_events: usize,
    /// When `false` all operations become no-ops (default `true`).
    pub enabled: bool,
    /// Restrict recording to these categories; empty means record all.
    pub track_categories: Vec<TelemetryCategory>,
    /// Record only every `N`-th step (1 = every step, default `1`).
    pub step_interval: usize,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            max_events: 100_000,
            enabled: true,
            track_categories: Vec::new(),
            step_interval: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// TelemetryCollector
// ---------------------------------------------------------------------------

/// Collects telemetry events from a training run.
pub struct TelemetryCollector {
    config: TelemetryConfig,
    events: Vec<TelemetryEvent>,
    /// Write position for ring-buffer overwrite behavior.
    write_pos: usize,
    /// Total events ever recorded, including those overwritten.
    total_events: usize,
    /// Active timers: (label, start Instant).
    active_timers: Vec<(String, Instant)>,
    /// Current training step.
    step: usize,
    /// When collection started (used for relative timestamps in timers).
    collection_start: Instant,
}

impl TelemetryCollector {
    /// Create a new collector with the given configuration.
    pub fn new(config: TelemetryConfig) -> Self {
        let capacity = config.max_events;
        Self {
            config,
            events: Vec::with_capacity(capacity),
            write_pos: 0,
            total_events: 0,
            active_timers: Vec::new(),
            step: 0,
            collection_start: Instant::now(),
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn should_record(&self, category: &TelemetryCategory) -> bool {
        if !self.config.enabled {
            return false;
        }
        if self.config.step_interval > 1 && !self.step.is_multiple_of(self.config.step_interval) {
            return false;
        }
        if !self.config.track_categories.is_empty()
            && !self.config.track_categories.contains(category)
        {
            return false;
        }
        true
    }

    fn push_event(&mut self, event: TelemetryEvent) {
        self.total_events += 1;
        if self.events.len() < self.config.max_events {
            self.events.push(event);
        } else {
            let idx = self.write_pos % self.config.max_events;
            self.events[idx] = event;
            self.write_pos += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Record an event with a known duration.
    ///
    /// Returns `Ok(())` even when the collector is disabled or the category is
    /// filtered — those are silent no-ops, not errors.
    pub fn record(
        &mut self,
        category: TelemetryCategory,
        label: &str,
        duration_us: u64,
        metadata: Vec<(String, f64)>,
    ) -> Result<(), TelemetryError> {
        if !self.should_record(&category) {
            return Ok(());
        }
        let event = TelemetryEvent {
            category,
            label: label.to_string(),
            duration_us,
            step: self.step,
            metadata,
        };
        self.push_event(event);
        Ok(())
    }

    /// Start a named timer.
    ///
    /// Returns [`TelemetryError::TimerAlreadyStarted`] if a timer with the
    /// same label is already running.
    pub fn start_timer(&mut self, label: &str) -> Result<(), TelemetryError> {
        if self.active_timers.iter().any(|(l, _)| l == label) {
            return Err(TelemetryError::TimerAlreadyStarted(label.to_string()));
        }
        self.active_timers.push((label.to_string(), Instant::now()));
        Ok(())
    }

    /// Stop a named timer and record the elapsed time.
    ///
    /// Returns the elapsed time in microseconds, or
    /// [`TelemetryError::TimerNotStarted`] if no such timer is running.
    pub fn stop_timer(
        &mut self,
        category: TelemetryCategory,
        label: &str,
        metadata: Vec<(String, f64)>,
    ) -> Result<u64, TelemetryError> {
        let pos = self
            .active_timers
            .iter()
            .position(|(l, _)| l == label)
            .ok_or_else(|| TelemetryError::TimerNotStarted(label.to_string()))?;

        let (_, start) = self.active_timers.remove(pos);
        let elapsed_us = start.elapsed().as_micros() as u64;

        self.record(category, label, elapsed_us, metadata)?;
        Ok(elapsed_us)
    }

    /// Advance the internal step counter by one.
    pub fn advance_step(&mut self) {
        self.step += 1;
    }

    /// Current training step.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Number of events currently stored in the ring buffer.
    pub fn n_events(&self) -> usize {
        self.events.len()
    }

    /// Total number of events ever recorded (including overwritten ones).
    pub fn total_events_recorded(&self) -> usize {
        self.total_events
    }

    /// All events currently stored (may not be in chronological order after wrapping).
    pub fn events(&self) -> &[TelemetryEvent] {
        &self.events
    }

    /// Clear all stored events and reset counters.
    pub fn clear(&mut self) {
        self.events.clear();
        self.write_pos = 0;
        self.total_events = 0;
        self.active_timers.clear();
    }

    /// Return all events belonging to a specific category.
    pub fn events_by_category(&self, category: &TelemetryCategory) -> Vec<&TelemetryEvent> {
        self.events
            .iter()
            .filter(|e| &e.category == category)
            .collect()
    }

    /// Return all events whose label starts with `label` (prefix match).
    pub fn events_by_label(&self, label: &str) -> Vec<&TelemetryEvent> {
        self.events
            .iter()
            .filter(|e| e.label.starts_with(label))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// LatencyStats
// ---------------------------------------------------------------------------

/// Aggregated latency statistics for a group of events sharing the same label.
#[derive(Debug, Clone)]
pub struct LatencyStats {
    /// Label of the group.
    pub label: String,
    /// Category of the group.
    pub category: TelemetryCategory,
    /// Number of events in the group.
    pub count: usize,
    /// Mean duration in microseconds.
    pub mean_us: f64,
    /// Standard deviation in microseconds.
    pub std_us: f64,
    /// Minimum observed duration in microseconds.
    pub min_us: u64,
    /// Maximum observed duration in microseconds.
    pub max_us: u64,
    /// Median (50th percentile) in microseconds.
    pub p50_us: u64,
    /// 95th percentile in microseconds.
    pub p95_us: u64,
    /// 99th percentile in microseconds.
    pub p99_us: u64,
    /// Sum of all durations in microseconds.
    pub total_us: u64,
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

fn sorted_durations(events: &[&TelemetryEvent]) -> Vec<u64> {
    let mut vals: Vec<u64> = events.iter().map(|e| e.duration_us).collect();
    vals.sort_unstable();
    vals
}

fn percentile_sorted(sorted: &[u64], p: f64) -> u64 {
    let n = sorted.len();
    if n == 0 {
        return 0;
    }
    let idx = ((p / 100.0) * (n - 1) as f64).round() as usize;
    sorted[idx.min(n - 1)]
}

fn mean_f64(vals: &[u64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64
}

fn std_f64(vals: &[u64], mean: f64) -> f64 {
    if vals.len() < 2 {
        return 0.0;
    }
    let variance =
        vals.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / (vals.len() as f64);
    variance.sqrt()
}

// ---------------------------------------------------------------------------
// tel_compute_latency_stats
// ---------------------------------------------------------------------------

/// Compute latency statistics for a slice of event references that all share
/// the same label and category.
///
/// Returns `None` if the slice is empty.
pub fn tel_compute_latency_stats(events: &[&TelemetryEvent]) -> Option<LatencyStats> {
    if events.is_empty() {
        return None;
    }
    let first = events[0];
    let sorted = sorted_durations(events);
    let mean = mean_f64(&sorted);
    let std = std_f64(&sorted, mean);
    let total: u64 = sorted.iter().sum();

    Some(LatencyStats {
        label: first.label.clone(),
        category: first.category.clone(),
        count: events.len(),
        mean_us: mean,
        std_us: std,
        min_us: sorted[0],
        max_us: *sorted.last().unwrap_or(&0),
        p50_us: percentile_sorted(&sorted, 50.0),
        p95_us: percentile_sorted(&sorted, 95.0),
        p99_us: percentile_sorted(&sorted, 99.0),
        total_us: total,
    })
}

// ---------------------------------------------------------------------------
// tel_stats_by_label
// ---------------------------------------------------------------------------

/// Compute per-label statistics for events belonging to a specific category.
pub fn tel_stats_by_label(
    events: &[TelemetryEvent],
    category: &TelemetryCategory,
) -> Vec<LatencyStats> {
    // Collect events by label using a deterministic approach (preserve insertion order).
    let mut label_order: Vec<String> = Vec::new();
    let mut label_map: std::collections::HashMap<String, Vec<&TelemetryEvent>> =
        std::collections::HashMap::new();

    for event in events {
        if &event.category != category {
            continue;
        }
        if !label_map.contains_key(&event.label) {
            label_order.push(event.label.clone());
        }
        label_map
            .entry(event.label.clone())
            .or_default()
            .push(event);
    }

    label_order
        .into_iter()
        .filter_map(|label| {
            let group = label_map.get(&label)?;
            tel_compute_latency_stats(group)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// tel_stats_by_category
// ---------------------------------------------------------------------------

/// Compute aggregate statistics for each category present in `events`.
///
/// Each entry in the returned vector summarises all events for one category,
/// with the label set to the category's string name.
pub fn tel_stats_by_category(events: &[TelemetryEvent]) -> Vec<LatencyStats> {
    // Maintain insertion order.
    let mut cat_order: Vec<TelemetryCategory> = Vec::new();
    let mut cat_map: std::collections::HashMap<String, Vec<&TelemetryEvent>> =
        std::collections::HashMap::new();

    for event in events {
        let key = event.category.as_str().to_string();
        if !cat_map.contains_key(&key) {
            cat_order.push(event.category.clone());
        }
        cat_map.entry(key).or_default().push(event);
    }

    cat_order
        .into_iter()
        .filter_map(|cat| {
            let key = cat.as_str().to_string();
            let group = cat_map.get(&key)?;
            let first = group[0];
            let sorted = sorted_durations(group);
            let mean = mean_f64(&sorted);
            let std = std_f64(&sorted, mean);
            let total: u64 = sorted.iter().sum();
            Some(LatencyStats {
                label: cat.as_str().to_string(),
                category: first.category.clone(),
                count: group.len(),
                mean_us: mean,
                std_us: std,
                min_us: sorted[0],
                max_us: *sorted.last().unwrap_or(&0),
                p50_us: percentile_sorted(&sorted, 50.0),
                p95_us: percentile_sorted(&sorted, 95.0),
                p99_us: percentile_sorted(&sorted, 99.0),
                total_us: total,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// RollingWindow
// ---------------------------------------------------------------------------

/// Circular buffer for computing rolling statistics over the last `N` values.
pub struct RollingWindow {
    capacity: usize,
    values: Vec<u64>,
    write_pos: usize,
    count: usize,
}

impl RollingWindow {
    /// Create a new rolling window with the given capacity.
    ///
    /// # Panics
    ///
    /// Does not panic; a capacity of 0 is clamped to 1 internally to avoid
    /// division-by-zero later.
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            values: Vec::with_capacity(capacity),
            write_pos: 0,
            count: 0,
        }
    }

    /// Push a new value, overwriting the oldest when capacity is reached.
    pub fn push(&mut self, value: u64) {
        if self.values.len() < self.capacity {
            self.values.push(value);
        } else {
            self.values[self.write_pos % self.capacity] = value;
        }
        self.write_pos += 1;
        self.count = self.count.saturating_add(1);
    }

    fn current_slice(&self) -> &[u64] {
        &self.values
    }

    /// Arithmetic mean of stored values; returns `0.0` if empty.
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        mean_f64(self.current_slice())
    }

    /// Population standard deviation of stored values; returns `0.0` if fewer than 2.
    pub fn std(&self) -> f64 {
        if self.values.len() < 2 {
            return 0.0;
        }
        let m = self.mean();
        std_f64(self.current_slice(), m)
    }

    /// Minimum value; returns `None` if empty.
    pub fn min(&self) -> Option<u64> {
        self.values.iter().copied().min()
    }

    /// Maximum value; returns `None` if empty.
    pub fn max(&self) -> Option<u64> {
        self.values.iter().copied().max()
    }

    /// Percentile value where `p` ∈ [0, 100]; returns `None` if empty.
    pub fn percentile(&self, p: f64) -> Option<u64> {
        if self.values.is_empty() {
            return None;
        }
        let mut sorted = self.values.clone();
        sorted.sort_unstable();
        Some(percentile_sorted(&sorted, p))
    }

    /// Number of values currently stored.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// `true` if no values have been stored.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ThroughputTracker
// ---------------------------------------------------------------------------

/// Tracks step throughput over a rolling window.
pub struct ThroughputTracker {
    window: RollingWindow,
    steps_in_window: usize,
    start_time: Instant,
    total_steps: usize,
}

impl ThroughputTracker {
    /// Create a new tracker with the given rolling-window size.
    pub fn new(window_size: usize) -> Self {
        Self {
            window: RollingWindow::new(window_size),
            steps_in_window: 0,
            start_time: Instant::now(),
            total_steps: 0,
        }
    }

    /// Record the time taken by one training step (in microseconds).
    pub fn record_step(&mut self, step_time_us: u64) {
        self.window.push(step_time_us);
        self.steps_in_window += 1;
        self.total_steps += 1;
    }

    /// Steps per second based on the rolling window mean step time.
    ///
    /// Returns `0.0` if no steps have been recorded.
    pub fn steps_per_second(&self) -> f64 {
        let mean_us = self.window.mean();
        if mean_us <= 0.0 {
            return 0.0;
        }
        1_000_000.0 / mean_us
    }

    /// Estimated remaining time in seconds to reach `target_steps`.
    ///
    /// Returns `0.0` if `current_step >= target_steps`.
    pub fn eta_seconds(&self, current_step: usize, target_steps: usize) -> f64 {
        if current_step >= target_steps {
            return 0.0;
        }
        let remaining = (target_steps - current_step) as f64;
        let sps = self.steps_per_second();
        if sps <= 0.0 {
            return f64::INFINITY;
        }
        remaining / sps
    }

    /// Total elapsed seconds since this tracker was created.
    pub fn total_elapsed_s(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
}

// ---------------------------------------------------------------------------
// TelemetryReport
// ---------------------------------------------------------------------------

/// Aggregated summary of a telemetry collection session.
#[derive(Debug)]
pub struct TelemetryReport {
    /// Total number of training steps recorded.
    pub total_steps: usize,
    /// Total wall-clock time in seconds since collection started.
    pub total_duration_s: f64,
    /// Per-category latency statistics.
    pub per_category_stats: Vec<LatencyStats>,
    /// Per-label latency statistics across all categories.
    pub per_label_stats: Vec<LatencyStats>,
    /// The category with the highest cumulative time (the bottleneck).
    pub bottleneck_category: Option<TelemetryCategory>,
    /// Overall steps per second (total steps / total duration).
    pub steps_per_second: f64,
}

// ---------------------------------------------------------------------------
// Report generation
// ---------------------------------------------------------------------------

/// Build a [`TelemetryReport`] from a collector.
pub fn tel_generate_report(collector: &TelemetryCollector) -> TelemetryReport {
    let events = collector.events();
    let total_duration_s = collector.collection_start.elapsed().as_secs_f64();
    let total_steps = collector.step();

    let per_category_stats = tel_stats_by_category(events);

    // Per-label: gather all labels across all categories.
    let mut label_order: Vec<String> = Vec::new();
    let mut label_map: std::collections::HashMap<String, Vec<&TelemetryEvent>> =
        std::collections::HashMap::new();
    for event in events {
        if !label_map.contains_key(&event.label) {
            label_order.push(event.label.clone());
        }
        label_map
            .entry(event.label.clone())
            .or_default()
            .push(event);
    }
    let per_label_stats: Vec<LatencyStats> = label_order
        .into_iter()
        .filter_map(|label| {
            let group = label_map.get(&label)?;
            tel_compute_latency_stats(group)
        })
        .collect();

    // Bottleneck: category with highest total_us.
    let bottleneck_category = per_category_stats
        .iter()
        .max_by_key(|s| s.total_us)
        .map(|s| s.category.clone());

    let steps_per_second = if total_duration_s > 0.0 {
        total_steps as f64 / total_duration_s
    } else {
        0.0
    };

    TelemetryReport {
        total_steps,
        total_duration_s,
        per_category_stats,
        per_label_stats,
        bottleneck_category,
        steps_per_second,
    }
}

/// Format a [`TelemetryReport`] as a human-readable string.
pub fn tel_format_report(report: &TelemetryReport) -> String {
    let mut out = String::new();
    out.push_str("=== Telemetry Report ===\n");
    out.push_str(&format!(
        "Total steps: {}   Duration: {:.2}s   Steps/s: {:.2}\n",
        report.total_steps, report.total_duration_s, report.steps_per_second
    ));
    if let Some(cat) = &report.bottleneck_category {
        out.push_str(&format!("Bottleneck category: {}\n", cat.as_str()));
    }
    out.push_str("\n--- Per-Category Stats ---\n");
    for stats in &report.per_category_stats {
        out.push_str(&tel_format_latency_stats(stats));
        out.push('\n');
    }
    if !report.per_label_stats.is_empty() {
        out.push_str("\n--- Per-Label Stats ---\n");
        for stats in &report.per_label_stats {
            out.push_str(&tel_format_latency_stats(stats));
            out.push('\n');
        }
    }
    out
}

/// Format a single [`LatencyStats`] as a one-line summary.
pub fn tel_format_latency_stats(stats: &LatencyStats) -> String {
    format!(
        "[{}] count={} mean={:.1}ms p50={:.1}ms p95={:.1}ms p99={:.1}ms total={:.1}ms",
        stats.label,
        stats.count,
        stats.mean_us / 1_000.0,
        stats.p50_us as f64 / 1_000.0,
        stats.p95_us as f64 / 1_000.0,
        stats.p99_us as f64 / 1_000.0,
        stats.total_us as f64 / 1_000.0,
    )
}

/// Format a single [`TelemetryEvent`] as a short string.
pub fn tel_format_event(event: &TelemetryEvent) -> String {
    format!(
        "[step={} cat={} label={}] {:.3}ms",
        event.step,
        event.category.as_str(),
        event.label,
        event.duration_ms(),
    )
}

// ---------------------------------------------------------------------------
// Anomaly detection
// ---------------------------------------------------------------------------

/// Detect timing spikes: return indices of events whose duration exceeds
/// `mean + n_sigma * std`.
///
/// Returns an empty vector when all durations are uniform (std ≈ 0).
pub fn tel_detect_spikes(events: &[TelemetryEvent], n_sigma: f32) -> Vec<usize> {
    if events.is_empty() {
        return Vec::new();
    }
    let vals: Vec<u64> = events.iter().map(|e| e.duration_us).collect();
    let mean = mean_f64(&vals);
    let std = std_f64(&vals, mean);
    if std < 1.0 {
        return Vec::new();
    }
    let threshold = mean + n_sigma as f64 * std;
    events
        .iter()
        .enumerate()
        .filter_map(|(i, e)| {
            if e.duration_us as f64 > threshold {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

/// Detect regressions between two time windows.
///
/// Returns `true` when the mean duration in `late_events` exceeds
/// `threshold_ratio * mean(early_events)`.
pub fn tel_detect_regression(
    early_events: &[TelemetryEvent],
    late_events: &[TelemetryEvent],
    threshold_ratio: f32,
) -> bool {
    if early_events.is_empty() || late_events.is_empty() {
        return false;
    }
    let early_vals: Vec<u64> = early_events.iter().map(|e| e.duration_us).collect();
    let late_vals: Vec<u64> = late_events.iter().map(|e| e.duration_us).collect();
    let early_mean = mean_f64(&early_vals);
    if early_mean <= 0.0 {
        return false;
    }
    let late_mean = mean_f64(&late_vals);
    late_mean > threshold_ratio as f64 * early_mean
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_event(
        category: TelemetryCategory,
        label: &str,
        duration_us: u64,
        step: usize,
    ) -> TelemetryEvent {
        TelemetryEvent {
            category,
            label: label.to_string(),
            duration_us,
            step,
            metadata: Vec::new(),
        }
    }

    fn make_collector(max_events: usize) -> TelemetryCollector {
        TelemetryCollector::new(TelemetryConfig {
            max_events,
            ..Default::default()
        })
    }

    // -----------------------------------------------------------------------
    // TelemetryCategory
    // -----------------------------------------------------------------------

    #[test]
    fn test_category_as_str_all_variants_non_empty() {
        let variants = [
            TelemetryCategory::Step,
            TelemetryCategory::Forward,
            TelemetryCategory::Backward,
            TelemetryCategory::Optimizer,
            TelemetryCategory::Densification,
            TelemetryCategory::Render,
            TelemetryCategory::DataLoad,
            TelemetryCategory::Checkpoint,
            TelemetryCategory::Loss,
            TelemetryCategory::Custom("my_op".to_string()),
        ];
        for variant in &variants {
            assert!(
                !variant.as_str().is_empty(),
                "as_str was empty for {:?}",
                variant
            );
        }
    }

    #[test]
    fn test_category_as_str_known_values() {
        assert_eq!(TelemetryCategory::Step.as_str(), "step");
        assert_eq!(TelemetryCategory::Forward.as_str(), "forward");
        assert_eq!(TelemetryCategory::Backward.as_str(), "backward");
        assert_eq!(TelemetryCategory::Optimizer.as_str(), "optimizer");
        assert_eq!(TelemetryCategory::Densification.as_str(), "densification");
        assert_eq!(TelemetryCategory::Render.as_str(), "render");
        assert_eq!(TelemetryCategory::DataLoad.as_str(), "data_load");
        assert_eq!(TelemetryCategory::Checkpoint.as_str(), "checkpoint");
        assert_eq!(TelemetryCategory::Loss.as_str(), "loss");
        assert_eq!(TelemetryCategory::Custom("xyz".to_string()).as_str(), "xyz");
    }

    // -----------------------------------------------------------------------
    // TelemetryEvent
    // -----------------------------------------------------------------------

    #[test]
    fn test_event_duration_ms_conversion() {
        let e = make_event(TelemetryCategory::Step, "step", 5_000, 0);
        assert!((e.duration_ms() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_event_duration_s_conversion() {
        let e = make_event(TelemetryCategory::Step, "step", 2_000_000, 0);
        assert!((e.duration_s() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_event_duration_ms_zero() {
        let e = make_event(TelemetryCategory::Loss, "loss", 0, 0);
        assert_eq!(e.duration_ms(), 0.0);
        assert_eq!(e.duration_s(), 0.0);
    }

    // -----------------------------------------------------------------------
    // TelemetryCollector basics
    // -----------------------------------------------------------------------

    #[test]
    fn test_collector_new_starts_empty() {
        let c = make_collector(1000);
        assert_eq!(c.n_events(), 0);
        assert_eq!(c.step(), 0);
        assert_eq!(c.total_events_recorded(), 0);
    }

    #[test]
    fn test_collector_record_stores_event() {
        let mut c = make_collector(1000);
        c.record(TelemetryCategory::Step, "step", 1000, vec![])
            .unwrap();
        assert_eq!(c.n_events(), 1);
        assert_eq!(c.total_events_recorded(), 1);
        assert_eq!(c.events()[0].label, "step");
        assert_eq!(c.events()[0].duration_us, 1000);
    }

    #[test]
    fn test_collector_record_increments_n_events() {
        let mut c = make_collector(100);
        for i in 0..10u64 {
            c.record(TelemetryCategory::Forward, "fwd", i * 100, vec![])
                .unwrap();
        }
        assert_eq!(c.n_events(), 10);
    }

    #[test]
    fn test_collector_record_disabled_is_noop() {
        let mut c = TelemetryCollector::new(TelemetryConfig {
            enabled: false,
            ..Default::default()
        });
        c.record(TelemetryCategory::Step, "step", 1000, vec![])
            .unwrap();
        assert_eq!(c.n_events(), 0);
        assert_eq!(c.total_events_recorded(), 0);
    }

    #[test]
    fn test_collector_record_ring_buffer_wraps() {
        let mut c = make_collector(5);
        for i in 0..8u64 {
            c.record(TelemetryCategory::Step, "step", i * 100, vec![])
                .unwrap();
        }
        // Ring buffer capped at 5
        assert_eq!(c.n_events(), 5);
        // Total ever recorded is 8
        assert_eq!(c.total_events_recorded(), 8);
    }

    #[test]
    fn test_collector_ring_buffer_overwrites_old_events() {
        let mut c = make_collector(3);
        c.record(TelemetryCategory::Step, "s", 100, vec![]).unwrap();
        c.record(TelemetryCategory::Step, "s", 200, vec![]).unwrap();
        c.record(TelemetryCategory::Step, "s", 300, vec![]).unwrap();
        // Buffer now full [100, 200, 300]
        c.record(TelemetryCategory::Step, "s", 400, vec![]).unwrap();
        // One slot was overwritten; buffer still has 3 events
        assert_eq!(c.n_events(), 3);
        // 400 must be somewhere in the buffer
        let has_400 = c.events().iter().any(|e| e.duration_us == 400);
        assert!(has_400);
    }

    #[test]
    fn test_collector_advance_step_increments() {
        let mut c = make_collector(100);
        assert_eq!(c.step(), 0);
        c.advance_step();
        assert_eq!(c.step(), 1);
        c.advance_step();
        assert_eq!(c.step(), 2);
    }

    #[test]
    fn test_collector_clear_resets_events() {
        let mut c = make_collector(100);
        c.record(TelemetryCategory::Step, "s", 1000, vec![])
            .unwrap();
        c.record(TelemetryCategory::Step, "s", 2000, vec![])
            .unwrap();
        assert_eq!(c.n_events(), 2);
        c.clear();
        assert_eq!(c.n_events(), 0);
        assert_eq!(c.total_events_recorded(), 0);
    }

    // -----------------------------------------------------------------------
    // Timer API
    // -----------------------------------------------------------------------

    #[test]
    fn test_start_stop_timer_records_elapsed() {
        let mut c = make_collector(100);
        c.start_timer("render").unwrap();
        // Simulate some work
        std::thread::sleep(std::time::Duration::from_millis(5));
        let elapsed = c
            .stop_timer(TelemetryCategory::Render, "render", vec![])
            .unwrap();
        assert!(elapsed >= 4_000, "elapsed_us={elapsed} should be >= 4ms");
        assert_eq!(c.n_events(), 1);
    }

    #[test]
    fn test_stop_timer_not_started_returns_error() {
        let mut c = make_collector(100);
        let result = c.stop_timer(TelemetryCategory::Render, "render", vec![]);
        assert!(matches!(result, Err(TelemetryError::TimerNotStarted(_))));
    }

    #[test]
    fn test_start_timer_twice_returns_error() {
        let mut c = make_collector(100);
        c.start_timer("forward").unwrap();
        let result = c.start_timer("forward");
        assert!(matches!(
            result,
            Err(TelemetryError::TimerAlreadyStarted(_))
        ));
    }

    #[test]
    fn test_multiple_timers_independent() {
        let mut c = make_collector(100);
        c.start_timer("a").unwrap();
        c.start_timer("b").unwrap();
        c.stop_timer(TelemetryCategory::Forward, "b", vec![])
            .unwrap();
        c.stop_timer(TelemetryCategory::Backward, "a", vec![])
            .unwrap();
        assert_eq!(c.n_events(), 2);
    }

    // -----------------------------------------------------------------------
    // events_by_category / events_by_label
    // -----------------------------------------------------------------------

    #[test]
    fn test_events_by_category_filters_correctly() {
        let mut c = make_collector(100);
        c.record(TelemetryCategory::Forward, "fwd", 100, vec![])
            .unwrap();
        c.record(TelemetryCategory::Backward, "bwd", 200, vec![])
            .unwrap();
        c.record(TelemetryCategory::Forward, "fwd2", 150, vec![])
            .unwrap();
        let fwd = c.events_by_category(&TelemetryCategory::Forward);
        assert_eq!(fwd.len(), 2);
        let bwd = c.events_by_category(&TelemetryCategory::Backward);
        assert_eq!(bwd.len(), 1);
    }

    #[test]
    fn test_events_by_category_empty_when_no_match() {
        let mut c = make_collector(100);
        c.record(TelemetryCategory::Forward, "fwd", 100, vec![])
            .unwrap();
        let render = c.events_by_category(&TelemetryCategory::Render);
        assert!(render.is_empty());
    }

    #[test]
    fn test_events_by_label_prefix_match() {
        let mut c = make_collector(100);
        c.record(TelemetryCategory::Step, "forward_pass", 100, vec![])
            .unwrap();
        c.record(TelemetryCategory::Step, "forward_attn", 200, vec![])
            .unwrap();
        c.record(TelemetryCategory::Step, "backward", 300, vec![])
            .unwrap();
        let fwd = c.events_by_label("forward");
        assert_eq!(fwd.len(), 2);
        let back = c.events_by_label("backward");
        assert_eq!(back.len(), 1);
    }

    #[test]
    fn test_events_by_label_exact_match_included() {
        let mut c = make_collector(100);
        c.record(TelemetryCategory::Step, "foo", 100, vec![])
            .unwrap();
        let result = c.events_by_label("foo");
        assert_eq!(result.len(), 1);
    }

    // -----------------------------------------------------------------------
    // step_interval filter
    // -----------------------------------------------------------------------

    #[test]
    fn test_step_interval_skips_odd_steps() {
        let mut c = TelemetryCollector::new(TelemetryConfig {
            step_interval: 2,
            ..Default::default()
        });
        // step=0: should record (0 % 2 == 0)
        c.record(TelemetryCategory::Step, "s", 100, vec![]).unwrap();
        c.advance_step(); // step=1
                          // step=1: should NOT record (1 % 2 != 0)
        c.record(TelemetryCategory::Step, "s", 200, vec![]).unwrap();
        c.advance_step(); // step=2
                          // step=2: should record (2 % 2 == 0)
        c.record(TelemetryCategory::Step, "s", 300, vec![]).unwrap();
        assert_eq!(c.n_events(), 2);
    }

    // -----------------------------------------------------------------------
    // track_categories filter
    // -----------------------------------------------------------------------

    #[test]
    fn test_track_categories_filters_correctly() {
        let mut c = TelemetryCollector::new(TelemetryConfig {
            track_categories: vec![TelemetryCategory::Forward, TelemetryCategory::Backward],
            ..Default::default()
        });
        c.record(TelemetryCategory::Forward, "fwd", 100, vec![])
            .unwrap();
        c.record(TelemetryCategory::Render, "render", 200, vec![])
            .unwrap();
        c.record(TelemetryCategory::Backward, "bwd", 300, vec![])
            .unwrap();
        assert_eq!(c.n_events(), 2);
        assert!(c.events_by_category(&TelemetryCategory::Render).is_empty());
    }

    // -----------------------------------------------------------------------
    // tel_compute_latency_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_latency_stats_empty_returns_none() {
        let result = tel_compute_latency_stats(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_latency_stats_single_event() {
        let e = make_event(TelemetryCategory::Step, "step", 1000, 0);
        let refs = vec![&e];
        let stats = tel_compute_latency_stats(&refs).unwrap();
        assert_eq!(stats.count, 1);
        assert!((stats.mean_us - 1000.0).abs() < 1e-6);
        assert_eq!(stats.min_us, 1000);
        assert_eq!(stats.max_us, 1000);
        assert_eq!(stats.p50_us, 1000);
        assert_eq!(stats.p95_us, 1000);
        assert_eq!(stats.p99_us, 1000);
        assert_eq!(stats.std_us, 0.0);
    }

    #[test]
    fn test_latency_stats_known_values_mean() {
        // Events: 100, 200, 300, 400, 500 µs
        let events: Vec<TelemetryEvent> = (1..=5u64)
            .map(|i| make_event(TelemetryCategory::Forward, "fwd", i * 100, 0))
            .collect();
        let refs: Vec<&TelemetryEvent> = events.iter().collect();
        let stats = tel_compute_latency_stats(&refs).unwrap();
        assert_eq!(stats.count, 5);
        assert!(
            (stats.mean_us - 300.0).abs() < 1e-6,
            "mean_us={}",
            stats.mean_us
        );
        assert_eq!(stats.min_us, 100);
        assert_eq!(stats.max_us, 500);
        assert_eq!(stats.p50_us, 300);
        assert_eq!(stats.total_us, 1500);
    }

    #[test]
    fn test_latency_stats_p95_and_p99() {
        // 100 events from 1 to 100 µs
        let events: Vec<TelemetryEvent> = (1..=100u64)
            .map(|i| make_event(TelemetryCategory::Forward, "fwd", i, 0))
            .collect();
        let refs: Vec<&TelemetryEvent> = events.iter().collect();
        let stats = tel_compute_latency_stats(&refs).unwrap();
        // p95 index = round(0.95 * 99) = round(94.05) = 94, value = 95
        assert_eq!(stats.p95_us, 95, "p95_us={}", stats.p95_us);
        // p99 index = round(0.99 * 99) = round(98.01) = 98, value = 99
        assert_eq!(stats.p99_us, 99, "p99_us={}", stats.p99_us);
    }

    // -----------------------------------------------------------------------
    // tel_stats_by_label
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_by_label_groups_correctly() {
        let events = vec![
            make_event(TelemetryCategory::Forward, "fwd", 100, 0),
            make_event(TelemetryCategory::Forward, "fwd", 200, 1),
            make_event(TelemetryCategory::Forward, "attn", 50, 0),
        ];
        let stats = tel_stats_by_label(&events, &TelemetryCategory::Forward);
        assert_eq!(stats.len(), 2);
        let fwd_stat = stats.iter().find(|s| s.label == "fwd").unwrap();
        assert_eq!(fwd_stat.count, 2);
        let attn_stat = stats.iter().find(|s| s.label == "attn").unwrap();
        assert_eq!(attn_stat.count, 1);
    }

    #[test]
    fn test_stats_by_label_ignores_other_categories() {
        let events = vec![
            make_event(TelemetryCategory::Forward, "fwd", 100, 0),
            make_event(TelemetryCategory::Backward, "bwd", 200, 1),
        ];
        let stats = tel_stats_by_label(&events, &TelemetryCategory::Forward);
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].label, "fwd");
    }

    // -----------------------------------------------------------------------
    // tel_stats_by_category
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_by_category_one_entry_per_category() {
        let events = vec![
            make_event(TelemetryCategory::Forward, "fwd", 100, 0),
            make_event(TelemetryCategory::Forward, "fwd2", 150, 1),
            make_event(TelemetryCategory::Backward, "bwd", 200, 0),
        ];
        let stats = tel_stats_by_category(&events);
        assert_eq!(stats.len(), 2);
        let fwd = stats.iter().find(|s| s.label == "forward").unwrap();
        assert_eq!(fwd.count, 2);
        let bwd = stats.iter().find(|s| s.label == "backward").unwrap();
        assert_eq!(bwd.count, 1);
    }

    #[test]
    fn test_stats_by_category_total_us_sum() {
        let events = vec![
            make_event(TelemetryCategory::Render, "r1", 1000, 0),
            make_event(TelemetryCategory::Render, "r2", 2000, 1),
        ];
        let stats = tel_stats_by_category(&events);
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].total_us, 3000);
    }

    // -----------------------------------------------------------------------
    // RollingWindow
    // -----------------------------------------------------------------------

    #[test]
    fn test_rolling_window_new_empty() {
        let w = RollingWindow::new(10);
        assert!(w.is_empty());
        assert_eq!(w.len(), 0);
    }

    #[test]
    fn test_rolling_window_push_increases_len() {
        let mut w = RollingWindow::new(5);
        w.push(100);
        w.push(200);
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn test_rolling_window_wraps_at_capacity() {
        let mut w = RollingWindow::new(3);
        for v in [10u64, 20, 30, 40, 50] {
            w.push(v);
        }
        assert_eq!(w.len(), 3);
        // Latest 3 values must be present: 30, 40, 50
        let vals: Vec<u64> = w.current_slice().to_vec();
        assert!(vals.contains(&40));
        assert!(vals.contains(&50));
    }

    #[test]
    fn test_rolling_window_mean_empty_is_zero() {
        let w = RollingWindow::new(10);
        assert_eq!(w.mean(), 0.0);
    }

    #[test]
    fn test_rolling_window_mean_known_values() {
        let mut w = RollingWindow::new(4);
        for v in [100u64, 200, 300, 400] {
            w.push(v);
        }
        assert!((w.mean() - 250.0).abs() < 1e-6);
    }

    #[test]
    fn test_rolling_window_std_single_value() {
        let mut w = RollingWindow::new(5);
        w.push(500);
        assert_eq!(w.std(), 0.0);
    }

    #[test]
    fn test_rolling_window_min_max_empty() {
        let w = RollingWindow::new(10);
        assert!(w.min().is_none());
        assert!(w.max().is_none());
    }

    #[test]
    fn test_rolling_window_min_max_values() {
        let mut w = RollingWindow::new(5);
        for v in [300u64, 100, 200] {
            w.push(v);
        }
        assert_eq!(w.min(), Some(100));
        assert_eq!(w.max(), Some(300));
    }

    #[test]
    fn test_rolling_window_percentile_empty() {
        let w = RollingWindow::new(10);
        assert!(w.percentile(50.0).is_none());
    }

    #[test]
    fn test_rolling_window_percentile_p50_is_median() {
        let mut w = RollingWindow::new(5);
        for v in [10u64, 20, 30, 40, 50] {
            w.push(v);
        }
        // Sorted: [10, 20, 30, 40, 50]; p50 index = round(0.5*4)=2 => 30
        assert_eq!(w.percentile(50.0), Some(30));
    }

    #[test]
    fn test_rolling_window_percentile_p0_is_min() {
        let mut w = RollingWindow::new(5);
        for v in [50u64, 10, 30] {
            w.push(v);
        }
        assert_eq!(w.percentile(0.0), w.min());
    }

    #[test]
    fn test_rolling_window_percentile_p100_is_max() {
        let mut w = RollingWindow::new(5);
        for v in [50u64, 10, 30] {
            w.push(v);
        }
        assert_eq!(w.percentile(100.0), w.max());
    }

    // -----------------------------------------------------------------------
    // ThroughputTracker
    // -----------------------------------------------------------------------

    #[test]
    fn test_throughput_tracker_steps_per_second_after_recording() {
        let mut t = ThroughputTracker::new(10);
        // Record 10 steps each taking 1ms = 1000µs
        for _ in 0..10 {
            t.record_step(1_000);
        }
        let sps = t.steps_per_second();
        // Should be ~1000 steps/s
        assert!(sps > 900.0, "sps={sps}");
    }

    #[test]
    fn test_throughput_tracker_steps_per_second_zero_before_recording() {
        let t = ThroughputTracker::new(10);
        assert_eq!(t.steps_per_second(), 0.0);
    }

    #[test]
    fn test_throughput_tracker_eta_seconds_reasonable() {
        let mut t = ThroughputTracker::new(10);
        // 100µs per step => 10,000 steps/s
        for _ in 0..10 {
            t.record_step(100);
        }
        // From step 500 to 1000: 500 remaining at ~10k sps => ~0.05s
        let eta = t.eta_seconds(500, 1000);
        assert!(eta > 0.0, "eta={eta}");
        assert!(eta < 1.0, "eta={eta} should be much less than 1s");
    }

    #[test]
    fn test_throughput_tracker_eta_done_returns_zero() {
        let mut t = ThroughputTracker::new(10);
        t.record_step(1000);
        assert_eq!(t.eta_seconds(1000, 1000), 0.0);
    }

    #[test]
    fn test_throughput_tracker_total_elapsed_increases() {
        let t = ThroughputTracker::new(10);
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(t.total_elapsed_s() >= 0.004);
    }

    // -----------------------------------------------------------------------
    // tel_generate_report
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_report_correct_total_steps() {
        let mut c = make_collector(1000);
        c.record(TelemetryCategory::Step, "step", 1000, vec![])
            .unwrap();
        c.advance_step();
        c.record(TelemetryCategory::Step, "step", 2000, vec![])
            .unwrap();
        c.advance_step();
        let report = tel_generate_report(&c);
        assert_eq!(report.total_steps, 2);
    }

    #[test]
    fn test_generate_report_has_per_category_stats() {
        let mut c = make_collector(1000);
        c.record(TelemetryCategory::Forward, "fwd", 100, vec![])
            .unwrap();
        c.record(TelemetryCategory::Backward, "bwd", 200, vec![])
            .unwrap();
        let report = tel_generate_report(&c);
        assert!(!report.per_category_stats.is_empty());
        assert_eq!(report.per_category_stats.len(), 2);
    }

    #[test]
    fn test_generate_report_bottleneck_is_highest_total() {
        let mut c = make_collector(1000);
        c.record(TelemetryCategory::Forward, "fwd", 100, vec![])
            .unwrap();
        c.record(TelemetryCategory::Render, "render", 10_000, vec![])
            .unwrap();
        let report = tel_generate_report(&c);
        assert_eq!(
            report.bottleneck_category.as_ref().map(|c| c.as_str()),
            Some("render")
        );
    }

    // -----------------------------------------------------------------------
    // Formatting functions
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_report_non_empty_contains_steps() {
        let mut c = make_collector(1000);
        c.record(TelemetryCategory::Step, "s", 5000, vec![])
            .unwrap();
        c.advance_step();
        let report = tel_generate_report(&c);
        let text = tel_format_report(&report);
        assert!(!text.is_empty());
        assert!(text.contains("steps") || text.contains("ms"), "text={text}");
    }

    #[test]
    fn test_format_latency_stats_contains_mean_and_p95() {
        let events = [
            make_event(TelemetryCategory::Forward, "fwd", 1000, 0),
            make_event(TelemetryCategory::Forward, "fwd", 2000, 1),
        ];
        let refs: Vec<&TelemetryEvent> = events.iter().collect();
        let stats = tel_compute_latency_stats(&refs).unwrap();
        let text = tel_format_latency_stats(&stats);
        assert!(text.contains("mean"), "text={text}");
        assert!(text.contains("p95"), "text={text}");
    }

    #[test]
    fn test_format_event_contains_label_and_ms() {
        let e = make_event(TelemetryCategory::Render, "render_main", 3_500, 5);
        let text = tel_format_event(&e);
        assert!(text.contains("render_main"), "text={text}");
        assert!(text.contains("ms"), "text={text}");
    }

    // -----------------------------------------------------------------------
    // tel_detect_spikes
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_spikes_uniform_data_no_spikes() {
        let events: Vec<TelemetryEvent> = (0..20)
            .map(|i| make_event(TelemetryCategory::Step, "s", 1000, i))
            .collect();
        let spikes = tel_detect_spikes(&events, 2.0);
        assert!(spikes.is_empty(), "spikes={spikes:?}");
    }

    #[test]
    fn test_detect_spikes_detects_clear_spike() {
        let mut events: Vec<TelemetryEvent> = (0..19)
            .map(|i| make_event(TelemetryCategory::Step, "s", 1000, i))
            .collect();
        // Add a massive outlier
        events.push(make_event(TelemetryCategory::Step, "s", 1_000_000, 19));
        let spikes = tel_detect_spikes(&events, 2.0);
        assert!(!spikes.is_empty(), "Expected at least one spike");
        assert!(spikes.contains(&19));
    }

    #[test]
    fn test_detect_spikes_empty_events() {
        let spikes = tel_detect_spikes(&[], 2.0);
        assert!(spikes.is_empty());
    }

    // -----------------------------------------------------------------------
    // tel_detect_regression
    // -----------------------------------------------------------------------

    #[test]
    fn test_detect_regression_same_data_no_regression() {
        let events: Vec<TelemetryEvent> = (0..10)
            .map(|i| make_event(TelemetryCategory::Step, "s", 1000, i))
            .collect();
        let result = tel_detect_regression(&events, &events, 1.5);
        assert!(!result);
    }

    #[test]
    fn test_detect_regression_2x_slower_detects_regression() {
        let early: Vec<TelemetryEvent> = (0..10)
            .map(|i| make_event(TelemetryCategory::Step, "s", 1000, i))
            .collect();
        let late: Vec<TelemetryEvent> = (0..10)
            .map(|i| make_event(TelemetryCategory::Step, "s", 2100, i))
            .collect();
        let result = tel_detect_regression(&early, &late, 1.5);
        assert!(result, "2x slower should be a regression at 1.5x threshold");
    }

    #[test]
    fn test_detect_regression_empty_early_returns_false() {
        let late: Vec<TelemetryEvent> = vec![make_event(TelemetryCategory::Step, "s", 1000, 0)];
        assert!(!tel_detect_regression(&[], &late, 1.5));
    }

    #[test]
    fn test_detect_regression_empty_late_returns_false() {
        let early: Vec<TelemetryEvent> = vec![make_event(TelemetryCategory::Step, "s", 1000, 0)];
        assert!(!tel_detect_regression(&early, &[], 1.5));
    }

    // -----------------------------------------------------------------------
    // Metadata on events
    // -----------------------------------------------------------------------

    #[test]
    fn test_event_metadata_stored_correctly() {
        let mut c = make_collector(100);
        c.record(
            TelemetryCategory::Densification,
            "densify",
            5000,
            vec![("n_gaussians".to_string(), 100_000.0)],
        )
        .unwrap();
        let ev = &c.events()[0];
        assert_eq!(ev.metadata.len(), 1);
        assert_eq!(ev.metadata[0].0, "n_gaussians");
        assert!((ev.metadata[0].1 - 100_000.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // TelemetryError display
    // -----------------------------------------------------------------------

    #[test]
    fn test_telemetry_error_display_messages() {
        let err = TelemetryError::EventNotFound("foo".to_string());
        assert!(err.to_string().contains("foo"));

        let err = TelemetryError::TimerAlreadyStarted("bar".to_string());
        assert!(err.to_string().contains("bar"));

        let err = TelemetryError::TimerNotStarted("baz".to_string());
        assert!(err.to_string().contains("baz"));

        let err = TelemetryError::BufferOverflow {
            max: 100,
            current: 101,
        };
        assert!(err.to_string().contains("100"));
    }
}
