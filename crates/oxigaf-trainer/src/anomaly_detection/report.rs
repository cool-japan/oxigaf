//! Aggregated [`AnomalyReport`] and human-readable formatting helpers.

use super::detector::AnomalyDetector;
use super::types::{AnomalyEvent, AnomalyKind};

// ─────────────────────────────────────────────────────────────────────────────
// AnomalyReport and helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics report for an anomaly detector.
#[derive(Debug, Clone)]
pub struct AnomalyReport {
    /// Number of training steps at which checks were performed.
    pub n_steps_checked: usize,
    /// Total Fatal events.
    pub n_fatal: usize,
    /// Total Critical events.
    pub n_critical: usize,
    /// Total Warning events.
    pub n_warning: usize,
    /// Total Info events.
    pub n_info: usize,
    /// The most severe anomaly kind still present in the detector's
    /// (possibly `max_history`-truncated) retained event buffer, if any.
    pub most_severe: Option<AnomalyKind>,
    /// Anomaly events per 100 steps **checked** (`n_steps_checked`), not
    /// per 100 elapsed training steps — these differ by a factor of
    /// `check_interval`.
    pub anomaly_rate: f32,
}

/// Generate a summary report from a detector's accumulated state.
pub fn anom_generate_report(detector: &AnomalyDetector) -> AnomalyReport {
    // Use the detector's monotone counters, not a recount of
    // `detector.events()` — that buffer is truncated to `max_history` and
    // would silently under-report on a long run.
    let n_fatal = detector.n_fatal();
    let n_critical = detector.n_critical();
    let n_warning = detector.n_warning();
    let n_info = detector.n_info();
    let total = n_fatal + n_critical + n_warning + n_info;

    // `checks_performed()` is the detector's own count of steps where
    // `should_check` was true (including step 0) — not an approximation
    // from elapsed steps, which both double-divides by `check_interval`
    // and misses the step-0 check.
    let n_steps_checked = detector.checks_performed();
    let anomaly_rate = if n_steps_checked > 0 {
        total as f32 / n_steps_checked as f32 * 100.0
    } else {
        0.0
    };

    // Most severe among the currently retained (possibly truncated) events.
    let most_severe = detector
        .events()
        .iter()
        .max_by_key(|e| e.severity)
        .map(|e| e.kind.clone());

    AnomalyReport {
        n_steps_checked,
        n_fatal,
        n_critical,
        n_warning,
        n_info,
        most_severe,
        anomaly_rate,
    }
}

/// Format a single anomaly event as a log-line string.
pub fn anom_format_event(event: &AnomalyEvent) -> String {
    format!(
        "[Step {:>6}] [{:>8}] {}",
        event.step,
        event.severity.label(),
        event.kind.description()
    )
}

/// Format a summary report as a multi-line string.
pub fn anom_format_report(report: &AnomalyReport) -> String {
    let total = report.n_fatal + report.n_critical + report.n_warning + report.n_info;
    let mut out = String::new();
    out.push_str("=== Anomaly Detection Report ===\n");
    out.push_str(&format!("  Steps checked : {}\n", report.n_steps_checked));
    out.push_str(&format!(
        "  Total events  : {} ({:.2} per 100 steps)\n",
        total, report.anomaly_rate
    ));
    out.push_str(&format!("  Fatal         : {}\n", report.n_fatal));
    out.push_str(&format!("  Critical      : {}\n", report.n_critical));
    out.push_str(&format!("  Warning       : {}\n", report.n_warning));
    out.push_str(&format!("  Info          : {}\n", report.n_info));
    if let Some(ref kind) = report.most_severe {
        out.push_str(&format!("  Most severe   : {}\n", kind.description()));
    }
    out
}
