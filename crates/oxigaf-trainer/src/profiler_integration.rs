//! Training phase profiler — lightweight per-phase timing for the OxiGAF
//! optimization loop.
//!
//! Provides:
//! - [`TrainingPhase`] — enum of all distinct phases in a training iteration.
//! - [`PhaseStats`] — statistics accumulated for a single phase.
//! - [`TrainingProfiler`] — thread-safe profiler that records elapsed time per phase.
//! - [`PhaseGuard`] — RAII guard that records elapsed time on drop.
//!
//! All state is protected by a [`Mutex`] so the profiler can be shared across
//! threads without external synchronisation.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::sync::Mutex;
use std::time::Instant;

// ---------------------------------------------------------------------------
// TrainingPhase
// ---------------------------------------------------------------------------

/// Distinct phases of a training iteration.
///
/// `Total` should wrap the entire iteration; individual sub-phases should be
/// timed separately within that.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrainingPhase {
    Initialize,
    DataLoading,
    /// Pseudo-GT generation from the diffusion pipeline.
    DiffusionTarget,
    /// Rasterization forward pass.
    Forward,
    LossComputation,
    /// Rasterization backward pass.
    Backward,
    /// Adam parameter update step.
    Optimize,
    /// Clone / split / prune Gaussians.
    DensityControl,
    Checkpoint,
    Metrics,
    Total,
}

impl TrainingPhase {
    /// Short label used in formatted output.
    pub fn label(self) -> &'static str {
        match self {
            TrainingPhase::Initialize => "init",
            TrainingPhase::DataLoading => "data",
            TrainingPhase::DiffusionTarget => "diff",
            TrainingPhase::Forward => "fwd",
            TrainingPhase::LossComputation => "loss",
            TrainingPhase::Backward => "bwd",
            TrainingPhase::Optimize => "opt",
            TrainingPhase::DensityControl => "dens",
            TrainingPhase::Checkpoint => "ckpt",
            TrainingPhase::Metrics => "metrics",
            TrainingPhase::Total => "total",
        }
    }

    /// Full human-readable name used in tabular reports.
    pub fn display_name(self) -> &'static str {
        match self {
            TrainingPhase::Initialize => "Initialize",
            TrainingPhase::DataLoading => "Data Loading",
            TrainingPhase::DiffusionTarget => "Diffusion Target",
            TrainingPhase::Forward => "Forward",
            TrainingPhase::LossComputation => "Loss Computation",
            TrainingPhase::Backward => "Backward",
            TrainingPhase::Optimize => "Optimize",
            TrainingPhase::DensityControl => "Density Control",
            TrainingPhase::Checkpoint => "Checkpoint",
            TrainingPhase::Metrics => "Metrics",
            TrainingPhase::Total => "Total",
        }
    }

    /// Ordering index used when sorting phases in canonical display order.
    fn canonical_order(self) -> u8 {
        match self {
            TrainingPhase::Total => 0,
            TrainingPhase::Initialize => 1,
            TrainingPhase::DataLoading => 2,
            TrainingPhase::DiffusionTarget => 3,
            TrainingPhase::Forward => 4,
            TrainingPhase::LossComputation => 5,
            TrainingPhase::Backward => 6,
            TrainingPhase::Optimize => 7,
            TrainingPhase::DensityControl => 8,
            TrainingPhase::Checkpoint => 9,
            TrainingPhase::Metrics => 10,
        }
    }
}

// ---------------------------------------------------------------------------
// PhaseStats
// ---------------------------------------------------------------------------

/// Statistics accumulated for a single training phase.
#[derive(Debug, Clone)]
pub struct PhaseStats {
    /// Number of recordings made for this phase.
    pub count: u64,
    /// Cumulative time spent in this phase (microseconds).
    pub total_us: u64,
    /// Minimum single-recording duration (microseconds).
    pub min_us: u64,
    /// Maximum single-recording duration (microseconds).
    pub max_us: u64,
    /// Exponential moving average of duration, α = 0.1 (microseconds).
    pub ema_us: f64,
}

impl PhaseStats {
    /// Mean duration per recording in microseconds.
    ///
    /// Returns `0.0` when count is zero.
    pub fn mean_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_us as f64 / self.count as f64
        }
    }

    /// Update the statistics with a new measurement.
    ///
    /// The EMA is seeded with the raw value on the very first call.
    fn record(&mut self, duration_us: u64) {
        if self.count == 0 {
            // First recording: seed all fields.
            self.count = 1;
            self.total_us = duration_us;
            self.min_us = duration_us;
            self.max_us = duration_us;
            self.ema_us = duration_us as f64;
        } else {
            self.count += 1;
            self.total_us = self.total_us.saturating_add(duration_us);
            if duration_us < self.min_us {
                self.min_us = duration_us;
            }
            if duration_us > self.max_us {
                self.max_us = duration_us;
            }
            // EMA with α = 0.1
            const ALPHA: f64 = 0.1;
            self.ema_us = (1.0 - ALPHA) * self.ema_us + ALPHA * duration_us as f64;
        }
    }
}

impl Default for PhaseStats {
    fn default() -> Self {
        Self {
            count: 0,
            total_us: 0,
            min_us: u64::MAX,
            max_us: 0,
            ema_us: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal guard state
// ---------------------------------------------------------------------------

struct ProfilerState {
    phases: HashMap<TrainingPhase, PhaseStats>,
}

impl ProfilerState {
    fn new() -> Self {
        Self {
            phases: HashMap::new(),
        }
    }

    fn record(&mut self, phase: TrainingPhase, duration_us: u64) {
        self.phases.entry(phase).or_default().record(duration_us);
    }

    fn stats(&self, phase: TrainingPhase) -> Option<PhaseStats> {
        self.phases.get(&phase).cloned()
    }

    fn all_stats_sorted_by_total_desc(&self) -> Vec<(TrainingPhase, PhaseStats)> {
        let mut pairs: Vec<(TrainingPhase, PhaseStats)> =
            self.phases.iter().map(|(&p, s)| (p, s.clone())).collect();
        pairs.sort_by_key(|a| std::cmp::Reverse(a.1.total_us));
        pairs
    }

    fn reset(&mut self) {
        self.phases.clear();
    }
}

// ---------------------------------------------------------------------------
// TrainingProfiler
// ---------------------------------------------------------------------------

/// Thread-safe training profiler that tracks time spent in each phase.
///
/// When `enabled` is `false` all recording paths are skipped, so the overhead
/// of calling [`time`](TrainingProfiler::time) is a single branch plus one
/// function call — effectively free.
pub struct TrainingProfiler {
    enabled: bool,
    state: Mutex<ProfilerState>,
}

impl TrainingProfiler {
    /// Create a new profiler.
    ///
    /// Pass `enabled = false` to create a no-op profiler that never records
    /// any timing data and never acquires the internal lock.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            state: Mutex::new(ProfilerState::new()),
        }
    }

    /// Convenience constructor for a disabled (no-op) profiler.
    pub fn disabled() -> Self {
        Self::new(false)
    }

    /// Whether this profiler is collecting data.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record a duration for a phase.
    ///
    /// Thread-safe — acquires the internal [`Mutex`].
    /// If the mutex is poisoned the recording is silently dropped.
    pub fn record(&self, phase: TrainingPhase, duration_us: u64) {
        if !self.enabled {
            return;
        }
        if let Ok(mut guard) = self.state.lock() {
            guard.record(phase, duration_us);
        }
    }

    /// Time a closure, recording its wall-clock duration for `phase`.
    ///
    /// If the profiler is disabled the closure is still called but no timing
    /// is performed, avoiding any Mutex overhead.
    pub fn time<F, T>(&self, phase: TrainingPhase, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        if !self.enabled {
            return f();
        }
        let start = Instant::now();
        let result = f();
        let elapsed_us = start.elapsed().as_micros() as u64;
        self.record(phase, elapsed_us);
        result
    }

    /// Create an RAII [`PhaseGuard`] that records elapsed time on drop.
    ///
    /// ```rust
    /// # use oxigaf_trainer::profiler_integration::{TrainingProfiler, TrainingPhase};
    /// let profiler = TrainingProfiler::new(true);
    /// {
    ///     let _guard = profiler.scope(TrainingPhase::Forward);
    ///     // ... perform forward pass ...
    /// } // elapsed recorded here
    /// ```
    pub fn scope(&self, phase: TrainingPhase) -> PhaseGuard<'_> {
        PhaseGuard {
            profiler: self,
            phase,
            start: Instant::now(),
        }
    }

    /// Retrieve current statistics for a phase.
    ///
    /// Returns `None` if the phase has never been recorded, or if the profiler
    /// is disabled.
    pub fn stats(&self, phase: TrainingPhase) -> Option<PhaseStats> {
        if !self.enabled {
            return None;
        }
        self.state.lock().ok().and_then(|g| g.stats(phase))
    }

    /// Return all recorded phase statistics sorted by total time descending.
    ///
    /// Returns an empty `Vec` if the profiler is disabled or no data has been
    /// recorded yet.
    pub fn all_stats(&self) -> Vec<(TrainingPhase, PhaseStats)> {
        if !self.enabled {
            return Vec::new();
        }
        self.state
            .lock()
            .map(|g| g.all_stats_sorted_by_total_desc())
            .unwrap_or_default()
    }

    /// Clear all accumulated statistics.
    pub fn reset(&self) {
        if let Ok(mut guard) = self.state.lock() {
            guard.reset();
        }
    }

    /// Generate a tabular profiling report.
    ///
    /// Columns: Phase | Count | Total(ms) | Mean(ms) | Min(ms) | Max(ms) | EMA(ms)
    ///
    /// Numbers are right-aligned in fixed-width columns.
    pub fn format_report(&self) -> String {
        let mut out = String::new();
        if !self.enabled {
            let _ = writeln!(out, "(profiler disabled)");
            return out;
        }

        let pairs = self.all_stats();
        if pairs.is_empty() {
            let _ = writeln!(out, "(no profiling data)");
            return out;
        }

        // Header
        let _ = writeln!(
            out,
            "{:<20} {:>8} {:>12} {:>10} {:>10} {:>10} {:>10}",
            "Phase", "Count", "Total(ms)", "Mean(ms)", "Min(ms)", "Max(ms)", "EMA(ms)"
        );
        let _ = writeln!(out, "{}", "-".repeat(84));

        for (phase, s) in &pairs {
            let total_ms = s.total_us as f64 / 1000.0;
            let mean_ms = s.mean_us() / 1000.0;
            let min_ms = if s.count == 0 {
                0.0
            } else {
                s.min_us as f64 / 1000.0
            };
            let max_ms = s.max_us as f64 / 1000.0;
            let ema_ms = s.ema_us / 1000.0;

            let _ = writeln!(
                out,
                "{:<20} {:>8} {:>12.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
                phase.display_name(),
                s.count,
                total_ms,
                mean_ms,
                min_ms,
                max_ms,
                ema_ms,
            );
        }
        out
    }

    /// Generate a compact single-line status string of key phase EMAs.
    ///
    /// Format: `"fwd=2.1ms bwd=3.4ms opt=0.8ms dens=12.1ms"`
    ///
    /// Only phases that have recorded data are included.  Result is guaranteed
    /// to be well under 100 characters for typical training loops.
    pub fn format_status_line(&self) -> String {
        if !self.enabled {
            return String::from("(profiler disabled)");
        }

        // Collect recorded phases in canonical display order (except Total).
        let pairs = match self.state.lock() {
            Ok(g) => g.all_stats_sorted_by_total_desc(),
            Err(_) => return String::from("(lock error)"),
        };

        // Sort by canonical order instead of total_us for the status line,
        // skipping Total so the line stays compact.
        let mut ordered: Vec<(TrainingPhase, PhaseStats)> = pairs
            .into_iter()
            .filter(|(p, _)| *p != TrainingPhase::Total)
            .collect();
        ordered.sort_by_key(|(p, _)| p.canonical_order());

        if ordered.is_empty() {
            return String::from("(no data)");
        }

        let parts: Vec<String> = ordered
            .iter()
            .map(|(phase, s)| {
                let ema_ms = s.ema_us / 1000.0;
                format!("{}={:.1}ms", phase.label(), ema_ms)
            })
            .collect();

        parts.join(" ")
    }

    /// Estimated training iterations per second based on the EMA of
    /// [`TrainingPhase::Total`].
    ///
    /// Returns `0.0` if no `Total` data has been recorded.
    pub fn iterations_per_second(&self) -> f64 {
        let stats = match self.stats(TrainingPhase::Total) {
            Some(s) if s.ema_us > 0.0 => s,
            _ => return 0.0,
        };
        1_000_000.0 / stats.ema_us
    }
}

impl Default for TrainingProfiler {
    fn default() -> Self {
        Self::new(true)
    }
}

// ---------------------------------------------------------------------------
// PhaseGuard — RAII guard
// ---------------------------------------------------------------------------

/// RAII guard that records elapsed time for a phase on drop.
///
/// Created via [`TrainingProfiler::scope`].
pub struct PhaseGuard<'a> {
    profiler: &'a TrainingProfiler,
    phase: TrainingPhase,
    start: Instant,
}

impl<'a> Drop for PhaseGuard<'a> {
    fn drop(&mut self) {
        let elapsed_us = self.start.elapsed().as_micros() as u64;
        self.profiler.record(self.phase, elapsed_us);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_new_profiler() {
        let p = TrainingProfiler::new(true);
        assert!(p.is_enabled());
        assert!(p.all_stats().is_empty());
    }

    #[test]
    fn test_disabled_profiler_no_overhead() {
        let p = TrainingProfiler::disabled();
        assert!(!p.is_enabled());
        // Record should silently do nothing.
        p.record(TrainingPhase::Forward, 1_000);
        // Stats should always be None / empty.
        assert!(p.stats(TrainingPhase::Forward).is_none());
        assert!(p.all_stats().is_empty());
        // time() should still call the closure.
        let result = p.time(TrainingPhase::Forward, || 42u32);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_record_phase() {
        let p = TrainingProfiler::new(true);
        p.record(TrainingPhase::Forward, 2_000);
        p.record(TrainingPhase::Forward, 4_000);

        let s = p
            .stats(TrainingPhase::Forward)
            .expect("stats must be present");
        assert_eq!(s.count, 2);
        assert_eq!(s.total_us, 6_000);
        assert_eq!(s.min_us, 2_000);
        assert_eq!(s.max_us, 4_000);
    }

    #[test]
    fn test_time_closure() {
        let p = TrainingProfiler::new(true);
        let val = p.time(TrainingPhase::LossComputation, || {
            // No actual sleep needed — just return a value.
            99u32
        });
        assert_eq!(val, 99);
        let s = p
            .stats(TrainingPhase::LossComputation)
            .expect("stats must exist after time()");
        assert_eq!(s.count, 1);
        // Duration may be 0 µs on fast hardware — just verify it recorded.
        assert!(s.total_us < 1_000_000); // sanity upper bound: < 1 s
    }

    #[test]
    fn test_ema_converges() {
        let p = TrainingProfiler::new(true);
        // Seed with 1000 µs, then many recordings of 100 µs.
        p.record(TrainingPhase::Backward, 1_000);
        for _ in 0..200 {
            p.record(TrainingPhase::Backward, 100);
        }
        let s = p.stats(TrainingPhase::Backward).expect("stats must exist");
        // After many pulls toward 100 the EMA should be very close to 100.
        assert!(
            s.ema_us < 110.0,
            "EMA should converge toward 100, got {:.2}",
            s.ema_us
        );
    }

    #[test]
    fn test_all_stats_sorted_by_total() {
        let p = TrainingProfiler::new(true);
        // Forward: 1 × 1000 µs = total 1000
        p.record(TrainingPhase::Forward, 1_000);
        // Backward: 1 × 5000 µs = total 5000
        p.record(TrainingPhase::Backward, 5_000);
        // Optimize: 1 × 200 µs = total 200
        p.record(TrainingPhase::Optimize, 200);

        let all = p.all_stats();
        assert_eq!(all.len(), 3);
        // First entry should be Backward (highest total).
        assert_eq!(all[0].0, TrainingPhase::Backward);
        // Last entry should be Optimize (lowest total).
        assert_eq!(all[2].0, TrainingPhase::Optimize);
    }

    #[test]
    fn test_format_report_contains_headers() {
        let p = TrainingProfiler::new(true);
        p.record(TrainingPhase::Forward, 1_000);
        let report = p.format_report();
        assert!(
            report.contains("Phase"),
            "report must contain 'Phase' header"
        );
        assert!(
            report.contains("Count"),
            "report must contain 'Count' header"
        );
        assert!(
            report.contains("Total(ms)"),
            "report must contain 'Total(ms)' header"
        );
        assert!(
            report.contains("EMA(ms)"),
            "report must contain 'EMA(ms)' header"
        );
        assert!(
            report.contains("Forward"),
            "report must contain the phase name"
        );
    }

    #[test]
    fn test_format_status_line_short() {
        let p = TrainingProfiler::new(true);
        p.record(TrainingPhase::Forward, 2_100);
        p.record(TrainingPhase::Backward, 3_400);
        p.record(TrainingPhase::Optimize, 800);
        p.record(TrainingPhase::DensityControl, 12_100);

        let line = p.format_status_line();
        assert!(
            line.len() < 100,
            "status line must be < 100 chars, got {} chars: {}",
            line.len(),
            line
        );
        assert!(line.contains("fwd"), "status line must contain 'fwd'");
        assert!(line.contains("bwd"), "status line must contain 'bwd'");
        // Total should not appear in status line.
        assert!(
            !line.contains("total"),
            "status line must not contain 'total'"
        );
    }

    #[test]
    fn test_iterations_per_second() {
        let p = TrainingProfiler::new(true);
        // EMA seeded at 100_000 µs (0.1 s) → ~10 iter/s.
        p.record(TrainingPhase::Total, 100_000);
        let its = p.iterations_per_second();
        assert!(
            (its - 10.0).abs() < 0.1,
            "expected ~10 iter/s, got {:.3}",
            its
        );
    }

    #[test]
    fn test_reset_clears_stats() {
        let p = TrainingProfiler::new(true);
        p.record(TrainingPhase::Forward, 1_000);
        assert!(p.stats(TrainingPhase::Forward).is_some());
        p.reset();
        assert!(
            p.stats(TrainingPhase::Forward).is_none(),
            "stats must be cleared after reset"
        );
        assert!(p.all_stats().is_empty());
    }

    #[test]
    fn test_phase_guard_raii() {
        let p = TrainingProfiler::new(true);
        {
            let _guard = p.scope(TrainingPhase::Optimize);
            // Scope exits here, guard drops, elapsed is recorded.
        }
        let s = p
            .stats(TrainingPhase::Optimize)
            .expect("stats must exist after PhaseGuard drop");
        assert_eq!(s.count, 1);
    }

    #[test]
    fn test_thread_safe_concurrent_recording() {
        let p = Arc::new(TrainingProfiler::new(true));
        let n_threads = 8;
        let recordings_per_thread = 100u64;

        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let p_clone = Arc::clone(&p);
                thread::spawn(move || {
                    for _ in 0..recordings_per_thread {
                        p_clone.record(TrainingPhase::Forward, 1_000);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread must not panic");
        }

        let s = p
            .stats(TrainingPhase::Forward)
            .expect("stats must exist after concurrent recording");
        assert_eq!(
            s.count,
            n_threads * recordings_per_thread,
            "all concurrent recordings must be captured"
        );
        assert_eq!(
            s.total_us,
            n_threads * recordings_per_thread * 1_000,
            "total_us must be consistent"
        );
    }

    #[test]
    fn test_phase_stats_min_max_tracking() {
        let p = TrainingProfiler::new(true);
        p.record(TrainingPhase::DataLoading, 500);
        p.record(TrainingPhase::DataLoading, 2_000);
        p.record(TrainingPhase::DataLoading, 1_000);

        let s = p
            .stats(TrainingPhase::DataLoading)
            .expect("stats must exist");
        assert_eq!(s.min_us, 500, "min should be 500");
        assert_eq!(s.max_us, 2_000, "max should be 2000");
    }

    #[test]
    fn test_disabled_profiler_format_methods() {
        let p = TrainingProfiler::disabled();
        let report = p.format_report();
        assert!(
            report.contains("disabled"),
            "disabled profiler report should indicate disabled state"
        );
        let status = p.format_status_line();
        assert!(
            status.contains("disabled"),
            "disabled profiler status should indicate disabled state"
        );
        assert_eq!(p.iterations_per_second(), 0.0);
    }

    // Ensure Duration 0 does not panic (fast hardware edge case).
    #[test]
    fn test_record_zero_duration() {
        let p = TrainingProfiler::new(true);
        p.record(TrainingPhase::Metrics, 0);
        let s = p.stats(TrainingPhase::Metrics).expect("stats must exist");
        assert_eq!(s.count, 1);
        assert_eq!(s.total_us, 0);
        assert_eq!(s.min_us, 0);
        assert_eq!(s.ema_us, 0.0);
    }

    // Verify that the unused import Duration does not cause issues and the
    // sleep-free test is valid.
    #[test]
    fn test_time_returns_closure_value() {
        let p = TrainingProfiler::new(true);
        // Verify generic return type forwarding.
        let s: String = p.time(TrainingPhase::Initialize, || String::from("hello"));
        assert_eq!(s, "hello");
    }
}
