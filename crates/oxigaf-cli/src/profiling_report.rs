//! Performance profiling report for OxiGAF training runs.
//!
//! This module provides:
//! - [`PhaseRecord`] — timing record for a single named phase in one step
//! - [`PhaseStats`] — aggregated statistics across multiple steps for one phase
//! - [`ProfilingReport`] — full report with per-phase stats and totals
//! - [`ProfilingCollector`] — ring-buffer collector for training-loop integration
//! - [`ProfilingConfig`] — configuration for which phases to track
//!
//! # Design
//!
//! The collector stores records in a [`std::collections::VecDeque`] as a ring
//! buffer capped at `max_records`. When capacity is reached, the oldest record
//! is evicted (FIFO) in O(1) time. Profiling can be disabled at runtime so that
//! `push` calls become true no-ops with zero allocation.
//!
//! A collector built via [`ProfilingCollector::with_config`] owns its
//! [`ProfilingConfig`] settings and enforces them itself — the phase
//! allow-list is applied inside `push`, so an excluded phase never consumes a
//! ring-buffer slot and callers do not have to re-implement
//! [`ProfilingConfig::should_track`] at their own boundary.
//!
//! Statistics (mean, std, percentiles) are computed on demand when
//! [`ProfilingCollector::build_report`] is called. The module never uses
//! external statistical libraries, `unwrap`, or `rand`.

use std::collections::{HashMap, VecDeque};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors that can occur while collecting or formatting profiling data.
#[derive(Debug, Error)]
pub enum ProfilingError {
    /// No timing records are available to compute statistics.
    #[error("no profiling data available")]
    EmptyData,

    /// A configuration value is invalid.
    #[error("invalid profiling configuration: {0}")]
    InvalidConfig(String),

    /// Serialization of a report failed.
    #[error("serialization error: {0}")]
    SerializationError(String),

    /// A requested phase name does not exist in the collector.
    #[error("missing phase: {0}")]
    MissingPhase(String),
}

// ---------------------------------------------------------------------------
// PhaseRecord
// ---------------------------------------------------------------------------

/// Timing record for a single named phase in one training step.
#[derive(Debug, Clone)]
pub struct PhaseRecord {
    /// Name of the phase (e.g. `"render"`, `"optimizer"`, `"diffusion"`).
    pub name: String,
    /// Training step at which this measurement was taken.
    pub step: usize,
    /// Wall-clock duration in milliseconds for this phase.
    pub duration_ms: f64,
    /// Memory allocated during this phase in bytes. `0` if not tracked.
    pub memory_bytes: u64,
    /// Number of 3D Gaussians processed. `0` if not applicable.
    pub num_gaussians: usize,
}

impl PhaseRecord {
    /// Create a new `PhaseRecord` with the given name, step, and duration.
    ///
    /// `memory_bytes` and `num_gaussians` default to `0`.
    pub fn new(name: impl Into<String>, step: usize, duration_ms: f64) -> Self {
        Self {
            name: name.into(),
            step,
            duration_ms,
            memory_bytes: 0,
            num_gaussians: 0,
        }
    }

    /// Set the memory allocation for this record (builder-style).
    pub fn with_memory(mut self, bytes: u64) -> Self {
        self.memory_bytes = bytes;
        self
    }

    /// Set the Gaussian count for this record (builder-style).
    pub fn with_gaussians(mut self, n: usize) -> Self {
        self.num_gaussians = n;
        self
    }

    /// Throughput in Gaussians per second.
    ///
    /// Returns `0.0` if `num_gaussians` is zero or `duration_ms` is zero.
    pub fn throughput_gps(&self) -> f64 {
        if self.num_gaussians == 0 || self.duration_ms == 0.0 {
            return 0.0;
        }
        self.num_gaussians as f64 / (self.duration_ms / 1000.0)
    }
}

// ---------------------------------------------------------------------------
// PhaseStats
// ---------------------------------------------------------------------------

/// Aggregated statistics for a phase across multiple steps.
#[derive(Debug, Clone)]
pub struct PhaseStats {
    /// Name of the phase.
    pub name: String,
    /// Number of records contributing to these statistics.
    pub count: usize,
    /// Arithmetic mean of `duration_ms`.
    pub mean_ms: f64,
    /// Minimum `duration_ms` observed.
    pub min_ms: f64,
    /// Maximum `duration_ms` observed.
    pub max_ms: f64,
    /// Population standard deviation of `duration_ms` (zero for a single record).
    pub std_ms: f64,
    /// 50th percentile (median) of `duration_ms`.
    pub p50_ms: f64,
    /// 95th percentile of `duration_ms`.
    pub p95_ms: f64,
    /// 99th percentile of `duration_ms`.
    pub p99_ms: f64,
    /// Mean memory allocated during this phase, in bytes.
    pub mean_memory_bytes: u64,
    /// Mean throughput in Gaussians per second.
    pub mean_throughput_gps: f64,
    /// Fraction of the total per-step wall time occupied by this phase.
    /// Set to `0.0` until [`ProfilingReport`] assigns it after computing
    /// `total_step_ms`.
    pub fraction_of_total: f64,
}

impl PhaseStats {
    /// One-line summary suitable for log output.
    ///
    /// # Example output
    /// ```text
    /// render: 12.3ms ± 1.2ms (p95=14.5ms, 42.1% of total)
    /// ```
    pub fn format_summary(&self) -> String {
        format!(
            "{}: {} ± {} (p95={}, {:.1}% of total)",
            self.name,
            format_duration_ms(self.mean_ms),
            format_duration_ms(self.std_ms),
            format_duration_ms(self.p95_ms),
            self.fraction_of_total * 100.0,
        )
    }

    /// Multi-line detailed statistics block.
    pub fn format_detailed(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("Phase: {}\n", self.name));
        out.push_str(&format!("  Count        : {}\n", self.count));
        out.push_str(&format!(
            "  Mean         : {}\n",
            format_duration_ms(self.mean_ms)
        ));
        out.push_str(&format!(
            "  Std          : {}\n",
            format_duration_ms(self.std_ms)
        ));
        out.push_str(&format!(
            "  Min          : {}\n",
            format_duration_ms(self.min_ms)
        ));
        out.push_str(&format!(
            "  Max          : {}\n",
            format_duration_ms(self.max_ms)
        ));
        out.push_str(&format!(
            "  P50          : {}\n",
            format_duration_ms(self.p50_ms)
        ));
        out.push_str(&format!(
            "  P95          : {}\n",
            format_duration_ms(self.p95_ms)
        ));
        out.push_str(&format!(
            "  P99          : {}\n",
            format_duration_ms(self.p99_ms)
        ));
        out.push_str(&format!(
            "  Memory (mean): {}\n",
            format_bytes(self.mean_memory_bytes)
        ));
        out.push_str(&format!(
            "  Throughput   : {}\n",
            format_throughput(self.mean_throughput_gps)
        ));
        out.push_str(&format!(
            "  % of total   : {:.2}%\n",
            self.fraction_of_total * 100.0
        ));
        out
    }
}

// ---------------------------------------------------------------------------
// ProfilingReport
// ---------------------------------------------------------------------------

/// Full profiling report with per-phase aggregated statistics.
#[derive(Debug)]
pub struct ProfilingReport {
    /// Per-phase stats, sorted by `mean_ms` descending (slowest first).
    pub phases: Vec<PhaseStats>,
    /// Approximate mean total per-step wall time (sum of all phase `mean_ms`).
    pub total_step_ms: f64,
    /// Number of training steps covered by this report.
    pub num_steps: usize,
    /// Earliest step included.
    pub start_step: usize,
    /// Latest step included.
    pub end_step: usize,
}

impl ProfilingReport {
    /// Look up statistics for a named phase. Returns `None` if not present.
    pub fn get_phase(&self, name: &str) -> Option<&PhaseStats> {
        self.phases.iter().find(|s| s.name == name)
    }

    /// Return up to `top_n` phases sorted by `mean_ms` descending.
    ///
    /// Phases are already sorted in `self.phases`, so this is a simple slice.
    pub fn bottleneck_phases(&self, top_n: usize) -> Vec<&PhaseStats> {
        self.phases.iter().take(top_n).collect()
    }

    /// Format the report as a fixed-width ASCII table.
    ///
    /// Columns: `Phase | Count | Mean(ms) | Std(ms) | P50 | P95 | P99 | % Total`
    pub fn format_table(&self) -> String {
        // Compute column widths dynamically so long phase names don't truncate.
        let name_width = self
            .phases
            .iter()
            .map(|p| p.name.len())
            .max()
            .unwrap_or(5)
            .max(5);

        let header = format!(
            "{:<nw$} | {:>5} | {:>10} | {:>8} | {:>10} | {:>10} | {:>10} | {:>7}",
            "Phase",
            "Count",
            "Mean(ms)",
            "Std(ms)",
            "P50",
            "P95",
            "P99",
            "% Total",
            nw = name_width,
        );
        let separator = "-".repeat(header.len());

        let mut rows = Vec::with_capacity(self.phases.len() + 3);
        rows.push(header);
        rows.push(separator.clone());

        for p in &self.phases {
            rows.push(format!(
                "{:<nw$} | {:>5} | {:>10} | {:>8} | {:>10} | {:>10} | {:>10} | {:>6.1}%",
                p.name,
                p.count,
                format_duration_ms(p.mean_ms),
                format_duration_ms(p.std_ms),
                format_duration_ms(p.p50_ms),
                format_duration_ms(p.p95_ms),
                format_duration_ms(p.p99_ms),
                p.fraction_of_total * 100.0,
                nw = name_width,
            ));
        }

        rows.push(separator);
        rows.push(format!(
            "{:<nw$} | {:>5} | {:>10} | {:>8} | {:>10} | {:>10} | {:>10} | {:>6.1}%",
            "TOTAL",
            "",
            format_duration_ms(self.total_step_ms),
            "",
            "",
            "",
            "",
            100.0_f64,
            nw = name_width,
        ));

        rows.join("\n")
    }

    /// One-paragraph summary: top 5 phases and total.
    pub fn format_summary(&self) -> String {
        let mut out = format!(
            "Profiling Summary (steps {}-{}, {} steps)\n",
            self.start_step, self.end_step, self.num_steps,
        );
        out.push_str(&format!(
            "Total per-step time: {}\n",
            format_duration_ms(self.total_step_ms)
        ));
        out.push_str("Top phases by mean duration:\n");
        for (i, p) in self.phases.iter().take(5).enumerate() {
            out.push_str(&format!("  {}. {}\n", i + 1, p.format_summary()));
        }
        out
    }

    /// Generate a self-contained HTML report with inline CSS. No JavaScript.
    pub fn to_html(&self) -> String {
        let mut html = String::new();
        html.push_str("<!DOCTYPE html>\n");
        html.push_str("<html lang=\"en\">\n<head>\n");
        html.push_str("<meta charset=\"utf-8\">\n");
        html.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
        html.push_str("<title>OxiGAF Profiling Report</title>\n");
        html.push_str("<style>\n");
        html.push_str(
            "body{font-family:monospace,sans-serif;background:#1a1a2e;color:#e0e0e0;margin:2rem;}\n",
        );
        html.push_str("h1{color:#4ecdc4;border-bottom:1px solid #4ecdc4;padding-bottom:.4rem;}\n");
        html.push_str(
            "table{border-collapse:collapse;width:100%;margin-top:1rem;font-size:.9rem;}\n",
        );
        html.push_str(
            "th{background:#16213e;color:#4ecdc4;padding:.5rem .8rem;text-align:left;border:1px solid #0f3460;}\n",
        );
        html.push_str("td{padding:.4rem .8rem;border:1px solid #0f3460;}\n");
        html.push_str("tr:nth-child(even){background:#16213e;}\n");
        html.push_str("tr.total-row{background:#0f3460;font-weight:bold;color:#f7b731;}\n");
        html.push_str(".bar-container{width:120px;background:#0f3460;border-radius:3px;display:inline-block;vertical-align:middle;}\n");
        html.push_str(".bar-fill{height:12px;background:#4ecdc4;border-radius:3px;}\n");
        html.push_str(".meta{color:#a0a0c0;font-size:.85rem;margin-bottom:1rem;}\n");
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");
        html.push_str("<h1>OxiGAF Profiling Report</h1>\n");
        html.push_str(&format!(
            "<p class=\"meta\">Steps {}&ndash;{} &bull; {} steps &bull; Total per-step: {}</p>\n",
            self.start_step,
            self.end_step,
            self.num_steps,
            html_escape(&format_duration_ms(self.total_step_ms)),
        ));

        html.push_str("<table>\n<thead><tr>");
        for col in &[
            "Phase",
            "Count",
            "Mean",
            "Std",
            "Min",
            "Max",
            "P50",
            "P95",
            "P99",
            "Memory",
            "Throughput",
            "% Total",
        ] {
            html.push_str(&format!("<th>{}</th>", html_escape(col)));
        }
        html.push_str("</tr></thead>\n<tbody>\n");

        for p in &self.phases {
            // Bar width capped at 100px.
            let bar_pct = (p.fraction_of_total * 100.0).clamp(0.0, 100.0);
            let bar_px = (bar_pct * 1.2) as u32; // 120px max container

            html.push_str("<tr>");
            html.push_str(&format!("<td>{}</td>", html_escape(&p.name)));
            html.push_str(&format!("<td>{}</td>", p.count));
            html.push_str(&format!(
                "<td>{}</td>",
                html_escape(&format_duration_ms(p.mean_ms))
            ));
            html.push_str(&format!(
                "<td>{}</td>",
                html_escape(&format_duration_ms(p.std_ms))
            ));
            html.push_str(&format!(
                "<td>{}</td>",
                html_escape(&format_duration_ms(p.min_ms))
            ));
            html.push_str(&format!(
                "<td>{}</td>",
                html_escape(&format_duration_ms(p.max_ms))
            ));
            html.push_str(&format!(
                "<td>{}</td>",
                html_escape(&format_duration_ms(p.p50_ms))
            ));
            html.push_str(&format!(
                "<td>{}</td>",
                html_escape(&format_duration_ms(p.p95_ms))
            ));
            html.push_str(&format!(
                "<td>{}</td>",
                html_escape(&format_duration_ms(p.p99_ms))
            ));
            html.push_str(&format!(
                "<td>{}</td>",
                html_escape(&format_bytes(p.mean_memory_bytes))
            ));
            html.push_str(&format!(
                "<td>{}</td>",
                html_escape(&format_throughput(p.mean_throughput_gps))
            ));
            html.push_str(&format!(
                "<td><span class=\"bar-container\"><span class=\"bar-fill\" style=\"width:{}px\"></span></span>&nbsp;{:.1}%</td>",
                bar_px, bar_pct,
            ));
            html.push_str("</tr>\n");
        }

        html.push_str("</tbody>\n<tfoot>\n");
        html.push_str("<tr class=\"total-row\">");
        html.push_str("<td>TOTAL</td>");
        html.push_str(&format!("<td colspan=\"1\">{}</td>", self.num_steps));
        html.push_str(&format!(
            "<td>{}</td>",
            html_escape(&format_duration_ms(self.total_step_ms))
        ));
        html.push_str("<td colspan=\"9\"></td>");
        html.push_str("</tr>\n</tfoot>\n</table>\n");
        html.push_str("</body>\n</html>\n");
        html
    }
}

// ---------------------------------------------------------------------------
// ProfilingCollector
// ---------------------------------------------------------------------------

/// Collects [`PhaseRecord`] entries from the training loop.
///
/// Internally uses a [`VecDeque`] as a ring buffer with O(1) FIFO eviction.
/// When `enabled` is `false`, all operations are no-ops.
///
/// A collector built with [`ProfilingCollector::with_config`] carries its
/// [`ProfilingConfig`]'s phase allow-list and applies it inside [`push`], so
/// callers never have to re-implement [`ProfilingConfig::should_track`] at
/// their own boundary. Collectors built with [`ProfilingCollector::new`],
/// [`ProfilingCollector::disabled`], or `Default` carry an empty allow-list,
/// which tracks every phase.
///
/// [`push`]: ProfilingCollector::push
pub struct ProfilingCollector {
    records: VecDeque<PhaseRecord>,
    max_records: usize,
    enabled: bool,
    /// Phase allow-list copied from the originating [`ProfilingConfig`].
    /// Empty means "track every phase".
    phases_to_track: Vec<String>,
    /// Number of records rejected by `phases_to_track` since construction (or
    /// since the last [`ProfilingCollector::clear`]).
    skipped: usize,
}

impl Default for ProfilingCollector {
    fn default() -> Self {
        Self {
            records: VecDeque::new(),
            max_records: 100_000,
            enabled: true,
            phases_to_track: Vec::new(),
            skipped: 0,
        }
    }
}

impl ProfilingCollector {
    /// Create a new collector capped at `max_records` entries.
    ///
    /// The collector tracks every phase name. Use
    /// [`ProfilingCollector::with_config`] to apply a phase allow-list.
    pub fn new(max_records: usize) -> Self {
        Self {
            records: VecDeque::with_capacity(max_records.min(4096)),
            max_records,
            enabled: true,
            phases_to_track: Vec::new(),
            skipped: 0,
        }
    }

    /// Create a collector that honours every field of `config`.
    ///
    /// - `config.enabled == false` yields a disabled collector whose `push`
    ///   calls are no-ops.
    /// - `config.max_records` becomes the ring-buffer capacity.
    /// - `config.phases_to_track` becomes the allow-list consulted by
    ///   [`ProfilingCollector::push`] via [`ProfilingConfig::should_track`],
    ///   so records for other phases are dropped at the source instead of
    ///   occupying ring-buffer slots that a tracked phase could have used.
    ///
    /// `config.report_interval_steps` describes *when a caller should emit a
    /// report*, not what the collector stores, so it is deliberately not
    /// consulted here; drive [`ProfilingCollector::build_report_for_range`]
    /// with it.
    ///
    /// # Errors
    ///
    /// Returns [`ProfilingError::InvalidConfig`] when `config` fails
    /// [`ProfilingConfig::validate`] — that is, when `max_records` is zero.
    /// A disabled config is still validated, so a caller cannot smuggle an
    /// invalid capacity past this constructor by turning profiling off.
    pub fn with_config(config: &ProfilingConfig) -> Result<Self, ProfilingError> {
        config.validate()?;
        if !config.enabled {
            return Ok(Self::disabled());
        }
        Ok(Self {
            records: VecDeque::with_capacity(config.max_records.min(4096)),
            max_records: config.max_records,
            enabled: true,
            phases_to_track: config.phases_to_track.clone(),
            skipped: 0,
        })
    }

    /// Create a disabled collector. All `push` calls are no-ops.
    pub fn disabled() -> Self {
        Self {
            records: VecDeque::new(),
            max_records: 0,
            enabled: false,
            phases_to_track: Vec::new(),
            skipped: 0,
        }
    }

    /// Whether this collector stores records for `phase_name`.
    ///
    /// Mirrors [`ProfilingConfig::should_track`] against the allow-list this
    /// collector was constructed with; always `true` for a collector built
    /// without a config.
    pub fn tracks_phase(&self, phase_name: &str) -> bool {
        self.phases_to_track.is_empty() || self.phases_to_track.iter().any(|p| p == phase_name)
    }

    /// The phase allow-list this collector applies. Empty means "track all".
    pub fn phases_to_track(&self) -> &[String] {
        &self.phases_to_track
    }

    /// Number of records rejected by the phase allow-list since construction
    /// (or since the last [`ProfilingCollector::clear`]).
    ///
    /// Records dropped because the collector is disabled are not counted:
    /// disabling profiling is not a filtering decision about a phase.
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    /// Add a record. No-op if the collector is disabled.
    ///
    /// Records whose phase name is excluded by the collector's allow-list are
    /// dropped and counted in [`ProfilingCollector::skipped`].
    ///
    /// When at capacity, the oldest record is evicted (FIFO) before insertion.
    pub fn push(&mut self, record: PhaseRecord) {
        if !self.enabled {
            return;
        }
        if self.max_records == 0 {
            return;
        }
        if !self.tracks_phase(&record.name) {
            self.skipped = self.skipped.saturating_add(1);
            return;
        }
        if self.records.len() >= self.max_records {
            self.records.pop_front();
        }
        self.records.push_back(record);
    }

    /// Whether this collector is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Number of records currently stored.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the collector holds no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Remove all records and reset the [`ProfilingCollector::skipped`]
    /// counter. The phase allow-list is retained.
    pub fn clear(&mut self) {
        self.records.clear();
        self.skipped = 0;
    }

    /// All records for a specific phase name, in insertion order.
    pub fn records_for_phase(&self, name: &str) -> Vec<&PhaseRecord> {
        self.records.iter().filter(|r| r.name == name).collect()
    }

    /// All records whose step is in `[start, end]` (inclusive).
    pub fn records_in_steps(&self, start: usize, end: usize) -> Vec<&PhaseRecord> {
        self.records
            .iter()
            .filter(|r| r.step >= start && r.step <= end)
            .collect()
    }

    /// Sorted, deduplicated list of all phase names present in the collector.
    pub fn phase_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .records
            .iter()
            .map(|r| r.name.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        names.sort();
        names
    }

    /// Build aggregated [`PhaseStats`] for a single phase name.
    ///
    /// Returns [`ProfilingError::EmptyData`] if no records exist for that name.
    pub fn build_phase_stats(&self, name: &str) -> Result<PhaseStats, ProfilingError> {
        let records: Vec<&PhaseRecord> = self.records_for_phase(name);
        if records.is_empty() {
            return Err(ProfilingError::EmptyData);
        }
        compute_phase_stats(name, &records)
    }

    /// Build the full [`ProfilingReport`] from all collected records.
    ///
    /// Returns [`ProfilingError::EmptyData`] if the collector is empty.
    pub fn build_report(&self) -> Result<ProfilingReport, ProfilingError> {
        if self.records.is_empty() {
            return Err(ProfilingError::EmptyData);
        }
        build_report_from_records(self.records.iter().collect())
    }

    /// Build a [`ProfilingReport`] restricted to steps in `[start, end]`.
    ///
    /// Returns [`ProfilingError::EmptyData`] if no records fall in the range.
    pub fn build_report_for_range(
        &self,
        start: usize,
        end: usize,
    ) -> Result<ProfilingReport, ProfilingError> {
        let records: Vec<&PhaseRecord> = self.records_in_steps(start, end);
        if records.is_empty() {
            return Err(ProfilingError::EmptyData);
        }
        build_report_from_records(records)
    }
}

// ---------------------------------------------------------------------------
// Internal statistics helpers
// ---------------------------------------------------------------------------

/// Compute [`PhaseStats`] from a non-empty slice of record references.
///
/// Panics if `records` is empty (callers must pre-check).
fn compute_phase_stats(name: &str, records: &[&PhaseRecord]) -> Result<PhaseStats, ProfilingError> {
    let n = records.len();
    debug_assert!(n > 0, "compute_phase_stats called with empty slice");

    // Duration statistics
    let durations: Vec<f64> = records.iter().map(|r| r.duration_ms).collect();
    let mean_ms = durations.iter().sum::<f64>() / n as f64;
    let variance = durations.iter().map(|d| (d - mean_ms).powi(2)).sum::<f64>() / n as f64;
    let std_ms = variance.sqrt();
    let min_ms = durations.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms = durations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Percentiles: sort a copy, then index.
    let mut sorted = durations.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let percentile = |pct: f64| -> f64 {
        let idx = ((pct * (n - 1) as f64) as usize).min(n - 1);
        sorted[idx]
    };
    let p50_ms = percentile(0.50);
    let p95_ms = percentile(0.95);
    let p99_ms = percentile(0.99);

    // Memory statistics
    let mean_memory_bytes = if records.iter().all(|r| r.memory_bytes == 0) {
        0
    } else {
        let sum: u64 = records.iter().map(|r| r.memory_bytes).sum();
        sum / n as u64
    };

    // Throughput: mean of per-record throughput values
    let mean_throughput_gps = {
        let sum: f64 = records.iter().map(|r| r.throughput_gps()).sum();
        sum / n as f64
    };

    Ok(PhaseStats {
        name: name.to_string(),
        count: n,
        mean_ms,
        min_ms,
        max_ms,
        std_ms,
        p50_ms,
        p95_ms,
        p99_ms,
        mean_memory_bytes,
        mean_throughput_gps,
        fraction_of_total: 0.0, // assigned later by build_report_from_records
    })
}

/// Build a full [`ProfilingReport`] from an arbitrary slice of record references.
///
/// Groups by phase name, computes stats, sets `fraction_of_total`, and sorts.
fn build_report_from_records(
    records: Vec<&PhaseRecord>,
) -> Result<ProfilingReport, ProfilingError> {
    if records.is_empty() {
        return Err(ProfilingError::EmptyData);
    }

    // Group records by phase name.
    let mut by_name: HashMap<String, Vec<&PhaseRecord>> = HashMap::new();
    let mut min_step = usize::MAX;
    let mut max_step = 0_usize;
    let mut distinct_steps: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for rec in &records {
        by_name.entry(rec.name.clone()).or_default().push(rec);
        if rec.step < min_step {
            min_step = rec.step;
        }
        if rec.step > max_step {
            max_step = rec.step;
        }
        distinct_steps.insert(rec.step);
    }

    // Compute per-phase stats.
    let mut phases: Vec<PhaseStats> = by_name
        .iter()
        .map(|(name, recs)| compute_phase_stats(name, recs))
        .collect::<Result<Vec<_>, _>>()?;

    // Total wall time = sum of mean_ms across all phases (approximation).
    let total_step_ms: f64 = phases.iter().map(|p| p.mean_ms).sum();

    // Assign fraction_of_total.
    if total_step_ms > 0.0 {
        for p in &mut phases {
            p.fraction_of_total = p.mean_ms / total_step_ms;
        }
    }

    // Sort phases by mean_ms descending (slowest first).
    phases.sort_by(|a, b| {
        b.mean_ms
            .partial_cmp(&a.mean_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(ProfilingReport {
        phases,
        total_step_ms,
        num_steps: distinct_steps.len(),
        start_step: if min_step == usize::MAX { 0 } else { min_step },
        end_step: max_step,
    })
}

// ---------------------------------------------------------------------------
// Text formatting helpers
// ---------------------------------------------------------------------------

/// Format a duration in milliseconds to a human-readable string.
///
/// - Sub-millisecond: `"0.12ms"`
/// - Up to 1000 ms: `"12.3ms"`
/// - 1000 ms or more: `"1.23s"`
pub fn format_duration_ms(ms: f64) -> String {
    if ms < 1.0 {
        format!("{:.2}ms", ms)
    } else if ms < 1000.0 {
        format!("{:.1}ms", ms)
    } else {
        format!("{:.2}s", ms / 1000.0)
    }
}

/// Format memory in bytes to a human-readable string.
///
/// - < 1 KiB: `"123 B"`
/// - < 1 MiB: `"12.3 KB"`
/// - < 1 GiB: `"12.3 MB"`
/// - 1 GiB or more: `"1.23 GB"`
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes < KB {
        format!("{} B", bytes)
    } else if bytes < MB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else if bytes < GB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    }
}

/// Format throughput in Gaussians per second to a human-readable string.
///
/// - < 1000 G/s: `"123 G/s"`
/// - < 1,000,000 G/s: `"12.3 KG/s"`
/// - 1,000,000 G/s or more: `"1.23 MG/s"`
pub fn format_throughput(gps: f64) -> String {
    if gps < 1_000.0 {
        format!("{:.0} G/s", gps)
    } else if gps < 1_000_000.0 {
        format!("{:.1} KG/s", gps / 1_000.0)
    } else {
        format!("{:.2} MG/s", gps / 1_000_000.0)
    }
}

// ---------------------------------------------------------------------------
// HTML escaping helper (module-private)
// ---------------------------------------------------------------------------

/// Escape HTML special characters in `s`.
fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            _ => out.push(ch),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// ProfilingConfig
// ---------------------------------------------------------------------------

/// Configuration controlling which phases are tracked and when reports appear.
#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    /// Whether profiling is active.
    pub enabled: bool,
    /// Maximum records to store in the ring buffer.
    pub max_records: usize,
    /// Phase names to track. Empty means track all phases.
    pub phases_to_track: Vec<String>,
    /// Emit a report every this many steps. `0` disables automatic reporting.
    pub report_interval_steps: usize,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_records: 100_000,
            phases_to_track: Vec::new(),
            report_interval_steps: 0,
        }
    }
}

impl ProfilingConfig {
    /// Validate the configuration.
    ///
    /// Returns [`ProfilingError::InvalidConfig`] if `max_records` is zero.
    pub fn validate(&self) -> Result<(), ProfilingError> {
        if self.max_records < 1 {
            return Err(ProfilingError::InvalidConfig(
                "max_records must be at least 1".to_string(),
            ));
        }
        Ok(())
    }

    /// Whether the given phase name should be tracked under this configuration.
    ///
    /// Returns `true` if `phases_to_track` is empty (track all) or the name
    /// is explicitly listed.
    ///
    /// A [`ProfilingCollector`] built with
    /// [`ProfilingCollector::with_config`] applies this predicate itself on
    /// every [`ProfilingCollector::push`], so callers normally do not need to
    /// call it directly.
    pub fn should_track(&self, phase_name: &str) -> bool {
        if self.phases_to_track.is_empty() {
            return true;
        }
        self.phases_to_track.iter().any(|p| p == phase_name)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- PhaseRecord ---

    #[test]
    fn test_phase_record_new_defaults() {
        let r = PhaseRecord::new("render", 5, 12.5);
        assert_eq!(r.name, "render");
        assert_eq!(r.step, 5);
        assert!((r.duration_ms - 12.5).abs() < 1e-9);
        assert_eq!(r.memory_bytes, 0);
        assert_eq!(r.num_gaussians, 0);
    }

    #[test]
    fn test_phase_record_builder_methods() {
        let r = PhaseRecord::new("opt", 10, 5.0)
            .with_memory(1024 * 1024)
            .with_gaussians(500_000);
        assert_eq!(r.memory_bytes, 1024 * 1024);
        assert_eq!(r.num_gaussians, 500_000);
    }

    #[test]
    fn test_throughput_gps_zero_gaussians() {
        let r = PhaseRecord::new("x", 0, 10.0);
        assert_eq!(r.throughput_gps(), 0.0);
    }

    #[test]
    fn test_throughput_gps_zero_duration() {
        let r = PhaseRecord::new("x", 0, 0.0).with_gaussians(1000);
        assert_eq!(r.throughput_gps(), 0.0);
    }

    #[test]
    fn test_throughput_gps_normal() {
        // 1_000_000 gaussians in 1000ms = 1000 s^-1 * 1000 G = 1_000_000 G/s = 1 MG/s
        let r = PhaseRecord::new("render", 0, 1000.0).with_gaussians(1_000_000);
        let gps = r.throughput_gps();
        assert!((gps - 1_000_000.0).abs() < 1.0, "got {}", gps);
    }

    // --- PhaseStats formatting ---

    #[test]
    fn test_phase_stats_format_summary_smoke() {
        let stats = PhaseStats {
            name: "render".to_string(),
            count: 100,
            mean_ms: 12.3,
            min_ms: 10.0,
            max_ms: 20.0,
            std_ms: 1.2,
            p50_ms: 12.0,
            p95_ms: 14.5,
            p99_ms: 18.0,
            mean_memory_bytes: 0,
            mean_throughput_gps: 0.0,
            fraction_of_total: 0.421,
        };
        let s = stats.format_summary();
        assert!(s.contains("render"), "name missing: {}", s);
        assert!(s.contains("12.3ms"), "mean missing: {}", s);
        assert!(s.contains("14.5ms"), "p95 missing: {}", s);
        assert!(s.contains("42.1%"), "fraction missing: {}", s);
    }

    // --- format_duration_ms ---

    #[test]
    fn test_format_duration_ms_sub_ms() {
        let s = format_duration_ms(0.123);
        assert!(s.ends_with("ms"), "{}", s);
        assert!(s.contains("0.12"), "{}", s);
    }

    #[test]
    fn test_format_duration_ms_ms_range() {
        let s = format_duration_ms(12.345);
        assert_eq!(s, "12.3ms");
    }

    #[test]
    fn test_format_duration_ms_seconds_range() {
        let s = format_duration_ms(1234.5);
        assert!(s.ends_with('s') && !s.ends_with("ms"), "{}", s);
        assert!(s.contains("1.23"), "{}", s);
    }

    // --- format_bytes ---

    #[test]
    fn test_format_bytes_bytes() {
        assert_eq!(format_bytes(500), "500 B");
    }

    #[test]
    fn test_format_bytes_kb() {
        let s = format_bytes(12 * 1024 + 300); // ~12.3 KB
        assert!(s.contains("KB"), "{}", s);
    }

    #[test]
    fn test_format_bytes_mb() {
        let s = format_bytes(12 * 1024 * 1024 + 300 * 1024); // ~12.3 MB
        assert!(s.contains("MB"), "{}", s);
    }

    #[test]
    fn test_format_bytes_gb() {
        let s = format_bytes(2 * 1024 * 1024 * 1024);
        assert!(s.contains("GB"), "{}", s);
    }

    // --- format_throughput ---

    #[test]
    fn test_format_throughput_gps() {
        let s = format_throughput(500.0);
        assert!(s.contains("G/s"), "{}", s);
        assert!(!s.contains('K'), "{}", s);
    }

    #[test]
    fn test_format_throughput_kgps() {
        let s = format_throughput(12_345.0);
        assert!(s.contains("KG/s"), "{}", s);
    }

    #[test]
    fn test_format_throughput_mgps() {
        let s = format_throughput(2_000_000.0);
        assert!(s.contains("MG/s"), "{}", s);
    }

    // --- ProfilingCollector ---

    #[test]
    fn test_collector_default_is_enabled() {
        let c = ProfilingCollector::default();
        assert!(c.is_enabled());
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
    }

    #[test]
    fn test_collector_disabled_push_is_noop() {
        let mut c = ProfilingCollector::disabled();
        c.push(PhaseRecord::new("x", 0, 1.0));
        assert_eq!(c.len(), 0);
        assert!(!c.is_enabled());
    }

    #[test]
    fn test_collector_push_and_len() {
        let mut c = ProfilingCollector::new(10);
        for i in 0..5 {
            c.push(PhaseRecord::new("render", i, i as f64 + 1.0));
        }
        assert_eq!(c.len(), 5);
        assert!(!c.is_empty());
    }

    #[test]
    fn test_collector_push_past_capacity_evicts_oldest() {
        let mut c = ProfilingCollector::new(3);
        // Push 5 records; only the last 3 should remain.
        for i in 0..5_usize {
            c.push(PhaseRecord::new("p", i, i as f64));
        }
        assert_eq!(c.len(), 3);
        // The oldest evicted should be step 0 and 1; steps 2,3,4 remain.
        let steps: Vec<usize> = c.records.iter().map(|r| r.step).collect();
        assert_eq!(steps, vec![2, 3, 4]);
    }

    #[test]
    fn test_records_for_phase_no_records_returns_empty() {
        let c = ProfilingCollector::default();
        assert!(c.records_for_phase("render").is_empty());
    }

    #[test]
    fn test_records_for_phase_finds_correct_records() {
        let mut c = ProfilingCollector::new(20);
        c.push(PhaseRecord::new("render", 0, 10.0));
        c.push(PhaseRecord::new("optimizer", 0, 5.0));
        c.push(PhaseRecord::new("render", 1, 11.0));
        let found = c.records_for_phase("render");
        assert_eq!(found.len(), 2);
        assert!(found.iter().all(|r| r.name == "render"));
    }

    #[test]
    fn test_records_in_steps_range_filtering() {
        let mut c = ProfilingCollector::new(20);
        for i in 0..10_usize {
            c.push(PhaseRecord::new("p", i, i as f64));
        }
        let in_range = c.records_in_steps(3, 6);
        assert_eq!(in_range.len(), 4);
        assert!(in_range.iter().all(|r| r.step >= 3 && r.step <= 6));
    }

    #[test]
    fn test_phase_names_sorted_and_deduplicated() {
        let mut c = ProfilingCollector::new(20);
        c.push(PhaseRecord::new("render", 0, 1.0));
        c.push(PhaseRecord::new("optimizer", 0, 1.0));
        c.push(PhaseRecord::new("render", 1, 1.0));
        c.push(PhaseRecord::new("diffusion", 0, 1.0));
        let names = c.phase_names();
        assert_eq!(names, vec!["diffusion", "optimizer", "render"]);
    }

    #[test]
    fn test_build_phase_stats_no_records_returns_empty_data() {
        let c = ProfilingCollector::default();
        let result = c.build_phase_stats("missing");
        assert!(
            matches!(result, Err(ProfilingError::EmptyData)),
            "expected EmptyData, got {result:?}"
        );
    }

    #[test]
    fn test_build_phase_stats_single_record() {
        let mut c = ProfilingCollector::new(10);
        c.push(PhaseRecord::new("render", 5, 20.0));
        let stats = c.build_phase_stats("render").expect("stats");
        assert_eq!(stats.count, 1);
        assert!((stats.mean_ms - 20.0).abs() < 1e-9);
        assert!((stats.std_ms - 0.0).abs() < 1e-9);
        assert!((stats.p50_ms - 20.0).abs() < 1e-9);
        assert!((stats.p95_ms - 20.0).abs() < 1e-9);
        assert!((stats.p99_ms - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_build_phase_stats_multiple_records_mean_and_p50() {
        let mut c = ProfilingCollector::new(20);
        // Insert 5 records with durations 10,20,30,40,50 ms.
        let durations = [10.0_f64, 20.0, 30.0, 40.0, 50.0];
        for (i, &d) in durations.iter().enumerate() {
            c.push(PhaseRecord::new("render", i, d));
        }
        let stats = c.build_phase_stats("render").expect("stats");
        assert_eq!(stats.count, 5);
        // Mean = 30.0
        assert!(
            (stats.mean_ms - 30.0).abs() < 1e-6,
            "mean={}",
            stats.mean_ms
        );
        // Sorted: 10,20,30,40,50; p50 index = floor(0.5 * 4) = 2 → 30.0
        assert!((stats.p50_ms - 30.0).abs() < 1e-6, "p50={}", stats.p50_ms);
        // p95 index = floor(0.95 * 4) = 3 → 40.0
        assert!((stats.p95_ms - 40.0).abs() < 1e-6, "p95={}", stats.p95_ms);
    }

    #[test]
    fn test_build_report_empty_returns_empty_data() {
        let c = ProfilingCollector::default();
        let result = c.build_report();
        assert!(
            matches!(result, Err(ProfilingError::EmptyData)),
            "expected EmptyData, got {result:?}"
        );
    }

    #[test]
    fn test_build_report_phases_sorted_by_mean_desc() {
        let mut c = ProfilingCollector::new(100);
        // render is slow (mean ~50ms), optimizer is fast (mean ~5ms)
        for i in 0..5_usize {
            c.push(PhaseRecord::new("render", i, 50.0));
            c.push(PhaseRecord::new("optimizer", i, 5.0));
        }
        let report = c.build_report().expect("report");
        assert_eq!(report.phases.len(), 2);
        assert_eq!(report.phases[0].name, "render");
        assert_eq!(report.phases[1].name, "optimizer");
    }

    #[test]
    fn test_build_report_fraction_of_total_sums_to_one() {
        let mut c = ProfilingCollector::new(100);
        for i in 0..4_usize {
            c.push(PhaseRecord::new("a", i, 30.0));
            c.push(PhaseRecord::new("b", i, 20.0));
            c.push(PhaseRecord::new("c", i, 10.0));
        }
        let report = c.build_report().expect("report");
        let sum: f64 = report.phases.iter().map(|p| p.fraction_of_total).sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={}", sum);
    }

    #[test]
    fn test_build_report_start_end_steps() {
        let mut c = ProfilingCollector::new(100);
        c.push(PhaseRecord::new("p", 3, 1.0));
        c.push(PhaseRecord::new("p", 10, 2.0));
        c.push(PhaseRecord::new("p", 7, 3.0));
        let report = c.build_report().expect("report");
        assert_eq!(report.start_step, 3);
        assert_eq!(report.end_step, 10);
        assert_eq!(report.num_steps, 3);
    }

    #[test]
    fn test_build_report_for_range() {
        let mut c = ProfilingCollector::new(100);
        for i in 0..10_usize {
            c.push(PhaseRecord::new("p", i, (i + 1) as f64));
        }
        let report = c.build_report_for_range(3, 6).expect("report");
        assert_eq!(report.start_step, 3);
        assert_eq!(report.end_step, 6);
        assert_eq!(report.num_steps, 4);
    }

    // --- ProfilingReport ---

    #[test]
    fn test_profiling_report_get_phase_found() {
        let mut c = ProfilingCollector::new(20);
        c.push(PhaseRecord::new("render", 0, 15.0));
        c.push(PhaseRecord::new("optimizer", 0, 5.0));
        let report = c.build_report().expect("report");
        assert!(report.get_phase("render").is_some());
    }

    #[test]
    fn test_profiling_report_get_phase_not_found() {
        let mut c = ProfilingCollector::new(20);
        c.push(PhaseRecord::new("render", 0, 15.0));
        let report = c.build_report().expect("report");
        assert!(report.get_phase("nonexistent").is_none());
    }

    #[test]
    fn test_profiling_report_bottleneck_phases_top_n() {
        let mut c = ProfilingCollector::new(100);
        for name in &["a", "b", "c", "d", "e"] {
            for i in 0..3_usize {
                c.push(PhaseRecord::new(
                    *name,
                    i,
                    (name.as_bytes()[0] as f64) * 0.5,
                ));
            }
        }
        let report = c.build_report().expect("report");
        let top3 = report.bottleneck_phases(3);
        assert_eq!(top3.len(), 3);
        // Top 3 should be the slowest by mean_ms (already sorted descending)
        for w in top3.windows(2) {
            assert!(w[0].mean_ms >= w[1].mean_ms);
        }
    }

    #[test]
    fn test_profiling_report_format_table_has_header() {
        let mut c = ProfilingCollector::new(20);
        c.push(PhaseRecord::new("render", 0, 10.0));
        c.push(PhaseRecord::new("optimizer", 0, 5.0));
        let report = c.build_report().expect("report");
        let table = report.format_table();
        assert!(table.contains("Phase"), "no Phase header: {}", table);
        assert!(table.contains("Mean(ms)"), "no Mean(ms) header: {}", table);
        assert!(table.contains("% Total"), "no %Total header: {}", table);
        assert!(table.contains("render"), "no phase row: {}", table);
        assert!(table.contains("TOTAL"), "no total row: {}", table);
    }

    #[test]
    fn test_profiling_report_to_html_produces_html() {
        let mut c = ProfilingCollector::new(20);
        c.push(PhaseRecord::new("render", 0, 12.3));
        c.push(PhaseRecord::new("optimizer", 0, 4.5));
        let report = c.build_report().expect("report");
        let html = report.to_html();
        assert!(
            html.starts_with("<!DOCTYPE html>"),
            "bad html start: {}",
            &html[..40]
        );
        assert!(html.contains("<table>"), "no table tag");
        assert!(html.contains("render"), "phase name missing from html");
        assert!(html.contains("OxiGAF Profiling Report"), "title missing");
    }

    // --- ProfilingConfig ---

    #[test]
    fn test_profiling_config_default_values() {
        let cfg = ProfilingConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.max_records, 100_000);
        assert!(cfg.phases_to_track.is_empty());
        assert_eq!(cfg.report_interval_steps, 0);
    }

    #[test]
    fn test_profiling_config_validate_zero_max_records() {
        let cfg = ProfilingConfig {
            max_records: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_profiling_config_should_track_empty_tracks_all() {
        let cfg = ProfilingConfig::default();
        assert!(cfg.should_track("render"));
        assert!(cfg.should_track("anything"));
    }

    #[test]
    fn test_profiling_config_should_track_filtered() {
        let cfg = ProfilingConfig {
            phases_to_track: vec!["render".to_string(), "diffusion".to_string()],
            ..Default::default()
        };
        assert!(cfg.should_track("render"));
        assert!(cfg.should_track("diffusion"));
        assert!(!cfg.should_track("optimizer"));
    }

    // --- ProfilingCollector honours its own ProfilingConfig ---
    // Regression: `push` used to ignore `ProfilingConfig::should_track`
    // entirely, forcing every caller to re-apply the filter itself.

    #[test]
    fn test_with_config_push_applies_phase_allow_list() {
        let cfg = ProfilingConfig {
            phases_to_track: vec!["render".to_string(), "diffusion".to_string()],
            ..Default::default()
        };
        let mut c = ProfilingCollector::with_config(&cfg).expect("valid config");
        c.push(PhaseRecord::new("render", 0, 10.0));
        c.push(PhaseRecord::new("optimizer", 0, 5.0));
        c.push(PhaseRecord::new("diffusion", 0, 7.0));
        c.push(PhaseRecord::new("backward", 0, 3.0));

        assert_eq!(c.len(), 2, "only allow-listed phases may be stored");
        assert_eq!(c.skipped(), 2, "excluded records must be counted");
        assert_eq!(c.phase_names(), vec!["diffusion", "render"]);
        assert!(c.records_for_phase("optimizer").is_empty());
    }

    #[test]
    fn test_with_config_empty_allow_list_tracks_every_phase() {
        let cfg = ProfilingConfig::default();
        let mut c = ProfilingCollector::with_config(&cfg).expect("valid config");
        c.push(PhaseRecord::new("render", 0, 10.0));
        c.push(PhaseRecord::new("optimizer", 0, 5.0));
        assert_eq!(c.len(), 2);
        assert_eq!(c.skipped(), 0);
        assert!(c.tracks_phase("anything"));
        assert!(c.phases_to_track().is_empty());
    }

    #[test]
    fn test_with_config_disabled_yields_noop_collector() {
        let cfg = ProfilingConfig {
            enabled: false,
            phases_to_track: vec!["render".to_string()],
            ..Default::default()
        };
        let mut c = ProfilingCollector::with_config(&cfg).expect("valid config");
        assert!(!c.is_enabled());
        c.push(PhaseRecord::new("render", 0, 10.0));
        assert_eq!(c.len(), 0);
        // Disabling profiling is not a per-phase filtering decision, so it
        // must not inflate the skipped counter.
        assert_eq!(c.skipped(), 0);
    }

    #[test]
    fn test_with_config_rejects_invalid_config() {
        let cfg = ProfilingConfig {
            max_records: 0,
            ..Default::default()
        };
        let result = ProfilingCollector::with_config(&cfg);
        assert!(
            matches!(&result, Err(ProfilingError::InvalidConfig(_))),
            "expected InvalidConfig, got {:?}",
            result.map(|c| c.len())
        );
    }

    #[test]
    fn test_with_config_rejects_invalid_config_even_when_disabled() {
        let cfg = ProfilingConfig {
            enabled: false,
            max_records: 0,
            ..Default::default()
        };
        assert!(
            ProfilingCollector::with_config(&cfg).is_err(),
            "an invalid capacity must not slip through a disabled config"
        );
    }

    #[test]
    fn test_with_config_honours_max_records_capacity() {
        let cfg = ProfilingConfig {
            max_records: 3,
            ..Default::default()
        };
        let mut c = ProfilingCollector::with_config(&cfg).expect("valid config");
        for i in 0..5_usize {
            c.push(PhaseRecord::new("p", i, i as f64));
        }
        assert_eq!(c.len(), 3);
        let steps: Vec<usize> = c.records.iter().map(|r| r.step).collect();
        assert_eq!(steps, vec![2, 3, 4]);
    }

    #[test]
    fn test_filtered_records_do_not_consume_ring_buffer_slots() {
        // The point of filtering at the source: an excluded phase must not
        // evict a tracked one. With a 3-slot buffer and 3 tracked records
        // interleaved with 3 excluded ones, all three tracked records survive.
        let cfg = ProfilingConfig {
            max_records: 3,
            phases_to_track: vec!["render".to_string()],
            ..Default::default()
        };
        let mut c = ProfilingCollector::with_config(&cfg).expect("valid config");
        for i in 0..3_usize {
            c.push(PhaseRecord::new("render", i, i as f64));
            c.push(PhaseRecord::new("optimizer", i, 99.0));
        }
        assert_eq!(c.len(), 3);
        assert_eq!(c.skipped(), 3);
        let steps: Vec<usize> = c.records.iter().map(|r| r.step).collect();
        assert_eq!(
            steps,
            vec![0, 1, 2],
            "excluded records must never displace tracked ones"
        );
    }

    #[test]
    fn test_collector_new_tracks_all_phases_unchanged() {
        // `new` must keep its historical "no filter" behaviour so existing
        // callers are unaffected by the allow-list.
        let mut c = ProfilingCollector::new(10);
        c.push(PhaseRecord::new("render", 0, 1.0));
        c.push(PhaseRecord::new("anything-at-all", 0, 1.0));
        assert_eq!(c.len(), 2);
        assert_eq!(c.skipped(), 0);
        assert!(c.tracks_phase("unlisted"));
    }

    // --- Clear ---

    #[test]
    fn test_collector_clear() {
        let mut c = ProfilingCollector::new(20);
        for i in 0..5_usize {
            c.push(PhaseRecord::new("p", i, 1.0));
        }
        assert_eq!(c.len(), 5);
        c.clear();
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
    }

    #[test]
    fn test_collector_clear_resets_skipped_but_keeps_filter() {
        let cfg = ProfilingConfig {
            phases_to_track: vec!["render".to_string()],
            ..Default::default()
        };
        let mut c = ProfilingCollector::with_config(&cfg).expect("valid config");
        c.push(PhaseRecord::new("render", 0, 1.0));
        c.push(PhaseRecord::new("optimizer", 0, 1.0));
        assert_eq!(c.skipped(), 1);
        c.clear();
        assert_eq!(c.len(), 0);
        assert_eq!(c.skipped(), 0);
        assert!(
            !c.tracks_phase("optimizer"),
            "clear() must not drop the allow-list"
        );
    }
}
