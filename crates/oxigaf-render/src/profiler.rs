//! CPU-side pass profiler for measuring wall-clock time per compute pass.
//!
//! # Overview
//!
//! [`PassProfiler`] accumulates timing statistics across frames for named passes,
//! computing mean, min, max and exponential moving average (EMA) durations.
//!
//! # Example
//!
//! ```rust
//! use oxigaf_render::profiler::{PassProfiler, ProfileScope};
//! use std::time::Duration;
//!
//! let profiler = PassProfiler::new();
//!
//! // Time a named pass with a closure.
//! profiler.time("preprocess", || {
//!     // GPU preprocess dispatch would go here
//! });
//!
//! // Or use the RAII scope guard.
//! {
//!     let _scope = ProfileScope::new(&profiler, "sort");
//!     // GPU sort dispatch
//! } // timing recorded on drop
//!
//! profiler.next_frame();
//! println!("{}", profiler.format_report());
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A single timing record for one pass invocation.
#[derive(Debug, Clone)]
pub struct PassRecord {
    /// Name of the pass.
    pub pass_name: String,
    /// Wall-clock duration of the invocation.
    pub duration: Duration,
    /// Frame index at which this record was taken.
    pub frame_index: u64,
}

/// Running statistics for a named pass.
#[derive(Debug, Clone)]
pub struct PassStats {
    /// Name of the pass.
    pub pass_name: String,
    /// Total number of times this pass was invoked.
    pub invocation_count: u64,
    /// Sum of all recorded durations.
    pub total_duration: Duration,
    /// Minimum recorded duration (Duration::MAX if no invocations yet).
    pub min_duration: Duration,
    /// Maximum recorded duration.
    pub max_duration: Duration,
    /// Arithmetic mean duration (total / count).
    pub mean_duration: Duration,
    /// Exponential moving average in microseconds (alpha = 0.1).
    pub ema_duration_us: f64,
}

const EMA_ALPHA: f64 = 0.1;

impl PassStats {
    /// Create a new, empty stats entry for the given pass name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            pass_name: name.into(),
            invocation_count: 0,
            total_duration: Duration::ZERO,
            min_duration: Duration::MAX,
            max_duration: Duration::ZERO,
            mean_duration: Duration::ZERO,
            ema_duration_us: 0.0,
        }
    }

    /// Update statistics with a new duration sample.
    pub fn update(&mut self, duration: Duration) {
        let new_us = duration.as_micros() as f64;

        // First sample: seed the EMA directly rather than pulling from zero.
        if self.invocation_count == 0 {
            self.ema_duration_us = new_us;
        } else {
            self.ema_duration_us = EMA_ALPHA * new_us + (1.0 - EMA_ALPHA) * self.ema_duration_us;
        }

        self.invocation_count += 1;
        self.total_duration += duration;

        if duration < self.min_duration {
            self.min_duration = duration;
        }
        if duration > self.max_duration {
            self.max_duration = duration;
        }

        // Recompute mean.
        let mean_us = self.total_duration.as_micros() / self.invocation_count as u128;
        self.mean_duration = Duration::from_micros(mean_us as u64);
    }

    /// Mean duration in milliseconds.
    pub fn mean_ms(&self) -> f64 {
        self.mean_duration.as_secs_f64() * 1_000.0
    }

    /// Minimum duration in milliseconds.
    pub fn min_ms(&self) -> f64 {
        if self.invocation_count == 0 {
            0.0
        } else {
            self.min_duration.as_secs_f64() * 1_000.0
        }
    }

    /// Maximum duration in milliseconds.
    pub fn max_ms(&self) -> f64 {
        self.max_duration.as_secs_f64() * 1_000.0
    }

    /// EMA duration in milliseconds.
    pub fn ema_ms(&self) -> f64 {
        self.ema_duration_us / 1_000.0
    }

    /// Estimated throughput: 1.0 / mean_seconds.
    ///
    /// Returns `0.0` if no invocations have been recorded yet or if mean is zero.
    pub fn throughput_fps(&self) -> f64 {
        let mean_secs = self.mean_duration.as_secs_f64();
        if mean_secs == 0.0 {
            0.0
        } else {
            1.0 / mean_secs
        }
    }
}

/// CPU-side pass profiler. Thread-safe for use across render frames.
pub struct PassProfiler {
    stats: Mutex<HashMap<String, PassStats>>,
    frame_count: AtomicU64,
    enabled: AtomicBool,
}

impl PassProfiler {
    /// Create a new enabled profiler.
    pub fn new() -> Self {
        Self {
            stats: Mutex::new(HashMap::new()),
            frame_count: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
        }
    }

    /// Create a disabled profiler. `time` calls still return the closure result
    /// but do not record any statistics.
    pub fn new_disabled() -> Self {
        Self {
            stats: Mutex::new(HashMap::new()),
            frame_count: AtomicU64::new(0),
            enabled: AtomicBool::new(false),
        }
    }

    /// Returns `true` if this profiler is actively recording.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Time a closure as a named pass and return the closure's result.
    ///
    /// If the profiler is disabled, the closure is still executed but no
    /// statistics are recorded.
    pub fn time<T, F: FnOnce() -> T>(&self, pass_name: &str, f: F) -> T {
        if !self.is_enabled() {
            return f();
        }
        let start = Instant::now();
        let result = f();
        self.record(pass_name, start.elapsed());
        result
    }

    /// Record a duration for a named pass (used internally and by [`ProfileScope`]).
    pub(crate) fn record(&self, pass_name: &str, duration: Duration) {
        if !self.is_enabled() {
            return;
        }
        let mut guard = self.stats.lock().unwrap_or_else(|e| e.into_inner());

        let entry = guard
            .entry(pass_name.to_owned())
            .or_insert_with(|| PassStats::new(pass_name));
        entry.update(duration);
    }

    /// Get statistics for a specific pass, or `None` if the pass is unknown.
    pub fn pass_stats(&self, pass_name: &str) -> Option<PassStats> {
        let guard = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        guard.get(pass_name).cloned()
    }

    /// Get all pass statistics, sorted by total duration descending.
    pub fn all_stats(&self) -> Vec<PassStats> {
        let guard = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        let mut v: Vec<PassStats> = guard.values().cloned().collect();
        v.sort_by_key(|p| std::cmp::Reverse(p.total_duration));
        v
    }

    /// Reset all accumulated statistics and the frame counter.
    pub fn reset(&self) {
        let mut guard = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        guard.clear();
        self.frame_count.store(0, Ordering::Relaxed);
    }

    /// Increment the frame counter by one.
    pub fn next_frame(&self) {
        self.frame_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Current frame index (number of times `next_frame` has been called).
    pub fn frame_count(&self) -> u64 {
        self.frame_count.load(Ordering::Relaxed)
    }

    /// Format a human-readable performance report table.
    ///
    /// Columns: Pass | Count | Mean(ms) | Min(ms) | Max(ms) | EMA(ms)
    pub fn format_report(&self) -> String {
        let all = self.all_stats();
        if all.is_empty() {
            return String::from("PassProfiler: no data recorded.\n");
        }

        let header = format!(
            "{:<30} {:>8} {:>12} {:>10} {:>10} {:>10}\n",
            "Pass", "Count", "Mean(ms)", "Min(ms)", "Max(ms)", "EMA(ms)"
        );
        let separator = "-".repeat(header.len() - 1) + "\n";

        let mut report = String::new();
        report.push_str(&format!(
            "PassProfiler Report (frame {})\n",
            self.frame_count()
        ));
        report.push_str(&separator);
        report.push_str(&header);
        report.push_str(&separator);

        for s in &all {
            let min_ms = if s.invocation_count == 0 {
                0.0
            } else {
                s.min_ms()
            };
            report.push_str(&format!(
                "{:<30} {:>8} {:>12.4} {:>10.4} {:>10.4} {:>10.4}\n",
                s.pass_name,
                s.invocation_count,
                s.mean_ms(),
                min_ms,
                s.max_ms(),
                s.ema_ms(),
            ));
        }
        report.push_str(&separator);
        report
    }

    /// Estimate memory bandwidth for a named pass given the number of bytes
    /// transferred per call.
    ///
    /// Returns `Some(GB/s)` if the pass is known, `None` otherwise.
    /// Returns `Some(0.0)` if mean duration is zero.
    pub fn estimate_bandwidth_gbs(&self, pass_name: &str, bytes_per_call: u64) -> Option<f64> {
        let stats = self.pass_stats(pass_name)?;
        let mean_secs = stats.mean_duration.as_secs_f64();
        if mean_secs == 0.0 {
            return Some(0.0);
        }
        Some(bytes_per_call as f64 / mean_secs / 1e9)
    }
}

impl Default for PassProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard that times a named scope and records the elapsed duration
/// to the associated [`PassProfiler`] when dropped.
pub struct ProfileScope<'a> {
    profiler: &'a PassProfiler,
    pass_name: &'a str,
    start: Instant,
}

impl<'a> ProfileScope<'a> {
    /// Begin timing the named pass.  Timing ends and is recorded when this
    /// value is dropped.
    pub fn new(profiler: &'a PassProfiler, pass_name: &'a str) -> Self {
        Self {
            profiler,
            pass_name,
            start: Instant::now(),
        }
    }
}

impl<'a> Drop for ProfileScope<'a> {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        self.profiler.record(self.pass_name, elapsed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // -------------------------------------------------------------------------
    // PassProfiler::new() — starts with no stats
    // -------------------------------------------------------------------------
    #[test]
    fn test_new_profiler_has_no_stats() {
        let p = PassProfiler::new();
        assert!(p.all_stats().is_empty());
        assert_eq!(p.frame_count(), 0);
    }

    // -------------------------------------------------------------------------
    // time() — records a pass
    // -------------------------------------------------------------------------
    #[test]
    fn test_time_records_pass() {
        let p = PassProfiler::new();
        p.time("alpha", || thread::sleep(Duration::from_millis(10)));
        let stats = p.pass_stats("alpha");
        assert!(stats.is_some());
        let s = stats.unwrap_or_else(|| panic!("expected stats for 'alpha'"));
        assert_eq!(s.invocation_count, 1);
        // At least 5 ms recorded (generous tolerance for CI)
        assert!(
            s.total_duration >= Duration::from_millis(5),
            "expected >= 5 ms, got {:?}",
            s.total_duration
        );
    }

    // -------------------------------------------------------------------------
    // Multiple time() calls accumulate count
    // -------------------------------------------------------------------------
    #[test]
    fn test_multiple_time_calls_accumulate() {
        let p = PassProfiler::new();
        for _ in 0..5 {
            p.time("beta", || {});
        }
        let s = p
            .pass_stats("beta")
            .unwrap_or_else(|| panic!("expected stats for 'beta'"));
        assert_eq!(s.invocation_count, 5);
    }

    // -------------------------------------------------------------------------
    // pass_stats() returns None for unknown pass
    // -------------------------------------------------------------------------
    #[test]
    fn test_pass_stats_none_for_unknown() {
        let p = PassProfiler::new();
        assert!(p.pass_stats("nonexistent_pass").is_none());
    }

    // -------------------------------------------------------------------------
    // pass_stats() returns correct mean after N calls
    // -------------------------------------------------------------------------
    #[test]
    fn test_pass_stats_correct_mean() {
        let p = PassProfiler::new();
        // Record 3 synthetic durations by calling record directly
        p.record("gamma", Duration::from_millis(10));
        p.record("gamma", Duration::from_millis(20));
        p.record("gamma", Duration::from_millis(30));

        let s = p
            .pass_stats("gamma")
            .unwrap_or_else(|| panic!("expected stats for 'gamma'"));
        assert_eq!(s.invocation_count, 3);
        // Mean should be ~20 ms (integer division may give 19-20 ms)
        assert!(
            s.mean_ms() >= 19.0 && s.mean_ms() <= 21.0,
            "mean_ms = {}",
            s.mean_ms()
        );
    }

    // -------------------------------------------------------------------------
    // min_duration <= mean
    // -------------------------------------------------------------------------
    #[test]
    fn test_min_duration_lte_mean() {
        let p = PassProfiler::new();
        p.record("delta", Duration::from_millis(5));
        p.record("delta", Duration::from_millis(15));
        p.record("delta", Duration::from_millis(25));

        let s = p
            .pass_stats("delta")
            .unwrap_or_else(|| panic!("expected stats for 'delta'"));
        assert!(s.min_duration <= s.mean_duration);
    }

    // -------------------------------------------------------------------------
    // max_duration >= mean
    // -------------------------------------------------------------------------
    #[test]
    fn test_max_duration_gte_mean() {
        let p = PassProfiler::new();
        p.record("epsilon", Duration::from_millis(5));
        p.record("epsilon", Duration::from_millis(15));
        p.record("epsilon", Duration::from_millis(25));

        let s = p
            .pass_stats("epsilon")
            .unwrap_or_else(|| panic!("expected stats for 'epsilon'"));
        assert!(s.max_duration >= s.mean_duration);
    }

    // -------------------------------------------------------------------------
    // reset() clears all stats
    // -------------------------------------------------------------------------
    #[test]
    fn test_reset_clears_stats() {
        let p = PassProfiler::new();
        p.time("zeta", || {});
        assert!(!p.all_stats().is_empty());

        p.reset();
        assert!(p.all_stats().is_empty());
        assert_eq!(p.frame_count(), 0);
    }

    // -------------------------------------------------------------------------
    // next_frame() increments counter
    // -------------------------------------------------------------------------
    #[test]
    fn test_next_frame_increments() {
        let p = PassProfiler::new();
        assert_eq!(p.frame_count(), 0);
        p.next_frame();
        assert_eq!(p.frame_count(), 1);
        p.next_frame();
        assert_eq!(p.frame_count(), 2);
    }

    // -------------------------------------------------------------------------
    // frame_count() returns correct value
    // -------------------------------------------------------------------------
    #[test]
    fn test_frame_count_correct() {
        let p = PassProfiler::new();
        for _ in 0..7 {
            p.next_frame();
        }
        assert_eq!(p.frame_count(), 7);
    }

    // -------------------------------------------------------------------------
    // all_stats() sorted by total_duration descending
    // -------------------------------------------------------------------------
    #[test]
    fn test_all_stats_sorted_descending() {
        let p = PassProfiler::new();
        // "slow" gets more total time
        for _ in 0..3 {
            p.record("slow", Duration::from_millis(30));
        }
        // "fast" gets less
        for _ in 0..3 {
            p.record("fast", Duration::from_millis(5));
        }

        let all = p.all_stats();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].pass_name, "slow");
        assert_eq!(all[1].pass_name, "fast");
    }

    // -------------------------------------------------------------------------
    // format_report() contains pass names
    // -------------------------------------------------------------------------
    #[test]
    fn test_format_report_contains_pass_names() {
        let p = PassProfiler::new();
        p.record("preprocess", Duration::from_millis(10));
        p.record("sort", Duration::from_millis(5));
        p.record("rasterize", Duration::from_millis(20));

        let report = p.format_report();
        assert!(report.contains("preprocess"), "report: {}", report);
        assert!(report.contains("sort"), "report: {}", report);
        assert!(report.contains("rasterize"), "report: {}", report);
    }

    // -------------------------------------------------------------------------
    // format_report() is non-empty for non-empty profiler
    // -------------------------------------------------------------------------
    #[test]
    fn test_format_report_non_empty() {
        let p = PassProfiler::new();
        p.record("alpha", Duration::from_millis(1));

        let report = p.format_report();
        assert!(!report.is_empty());
        // Should contain the header and the pass name
        assert!(report.contains("Pass"));
        assert!(report.contains("Mean"));
    }

    // -------------------------------------------------------------------------
    // new_disabled() — time() still returns result but records nothing
    // -------------------------------------------------------------------------
    #[test]
    fn test_disabled_profiler_returns_result_records_nothing() {
        let p = PassProfiler::new_disabled();
        let result = p.time("alpha", || 42_u32);
        assert_eq!(result, 42);
        assert!(
            p.all_stats().is_empty(),
            "disabled profiler should not record"
        );
    }

    // -------------------------------------------------------------------------
    // estimate_bandwidth_gbs() returns None for unknown pass
    // -------------------------------------------------------------------------
    #[test]
    fn test_bandwidth_none_for_unknown() {
        let p = PassProfiler::new();
        assert!(p.estimate_bandwidth_gbs("ghost", 1024).is_none());
    }

    // -------------------------------------------------------------------------
    // estimate_bandwidth_gbs() returns Some(positive) for known pass
    // -------------------------------------------------------------------------
    #[test]
    fn test_bandwidth_some_positive_for_known_pass() {
        let p = PassProfiler::new();
        // Record 1 ms for "memcopy"
        p.record("memcopy", Duration::from_millis(1));

        let bw = p
            .estimate_bandwidth_gbs("memcopy", 1_000_000_000)
            .unwrap_or_else(|| panic!("expected Some bandwidth"));
        assert!(bw > 0.0, "bandwidth should be positive, got {}", bw);
        // 1 GB in 1 ms → 1000 GB/s
        assert!(
            (bw - 1000.0).abs() < 50.0,
            "expected ~1000 GB/s, got {}",
            bw
        );
    }

    // -------------------------------------------------------------------------
    // EMA converges: after many identical durations, EMA ≈ actual duration
    // -------------------------------------------------------------------------
    #[test]
    fn test_ema_converges_to_actual_duration() {
        let p = PassProfiler::new();
        let target = Duration::from_millis(10);

        // Run 60 iterations so EMA has time to converge from the first seed.
        for _ in 0..60 {
            p.record("ema_test", target);
        }

        let s = p
            .pass_stats("ema_test")
            .unwrap_or_else(|| panic!("expected stats for 'ema_test'"));
        let target_us = target.as_micros() as f64;
        let ratio = s.ema_duration_us / target_us;
        assert!(
            (0.5..=1.5).contains(&ratio),
            "EMA did not converge: ema={} us, target={} us, ratio={}",
            s.ema_duration_us,
            target_us,
            ratio
        );
    }

    // -------------------------------------------------------------------------
    // PassStats::throughput_fps() = 1.0 / mean_seconds
    // -------------------------------------------------------------------------
    #[test]
    fn test_throughput_fps_formula() {
        let p = PassProfiler::new();
        // Mean = 10 ms → fps = 100
        p.record("fps_test", Duration::from_millis(10));

        let s = p
            .pass_stats("fps_test")
            .unwrap_or_else(|| panic!("expected stats for 'fps_test'"));
        let fps = s.throughput_fps();
        // 1 / 0.01 = 100
        assert!((fps - 100.0).abs() < 1.0, "expected ~100 fps, got {}", fps);
    }

    // -------------------------------------------------------------------------
    // ProfileScope — records timing when dropped
    // -------------------------------------------------------------------------
    #[test]
    fn test_profile_scope_records_on_drop() {
        let p = PassProfiler::new();

        {
            let _scope = ProfileScope::new(&p, "scoped_pass");
            thread::sleep(Duration::from_millis(10));
        } // drop here records timing

        let s = p
            .pass_stats("scoped_pass")
            .unwrap_or_else(|| panic!("expected stats for 'scoped_pass'"));
        assert_eq!(s.invocation_count, 1);
        assert!(
            s.total_duration >= Duration::from_millis(5),
            "expected >= 5 ms, got {:?}",
            s.total_duration
        );
    }

    // -------------------------------------------------------------------------
    // PassStats::min_ms() == 0.0 when no invocations
    // -------------------------------------------------------------------------
    #[test]
    fn test_min_ms_zero_when_no_invocations() {
        let s = PassStats::new("empty");
        assert_eq!(s.min_ms(), 0.0);
    }

    // -------------------------------------------------------------------------
    // PassStats::throughput_fps() == 0.0 when mean is zero
    // -------------------------------------------------------------------------
    #[test]
    fn test_throughput_fps_zero_when_no_data() {
        let s = PassStats::new("empty");
        assert_eq!(s.throughput_fps(), 0.0);
    }

    // -------------------------------------------------------------------------
    // format_report() returns "no data" message for empty profiler
    // -------------------------------------------------------------------------
    #[test]
    fn test_format_report_empty_profiler() {
        let p = PassProfiler::new();
        let report = p.format_report();
        assert!(
            report.contains("no data"),
            "expected 'no data' message, got: {}",
            report
        );
    }

    // -------------------------------------------------------------------------
    // time() return value passes through correctly
    // -------------------------------------------------------------------------
    #[test]
    fn test_time_return_value_passthrough() {
        let p = PassProfiler::new();
        let v: Vec<u32> = p.time("compute", || vec![1, 2, 3]);
        assert_eq!(v, vec![1, 2, 3]);
    }

    // -------------------------------------------------------------------------
    // Multiple distinct passes tracked independently
    // -------------------------------------------------------------------------
    #[test]
    fn test_multiple_passes_tracked_independently() {
        let p = PassProfiler::new();
        p.record("pass_a", Duration::from_millis(5));
        p.record("pass_b", Duration::from_millis(50));
        p.record("pass_a", Duration::from_millis(5));

        let a = p
            .pass_stats("pass_a")
            .unwrap_or_else(|| panic!("expected stats for 'pass_a'"));
        let b = p
            .pass_stats("pass_b")
            .unwrap_or_else(|| panic!("expected stats for 'pass_b'"));
        assert_eq!(a.invocation_count, 2);
        assert_eq!(b.invocation_count, 1);
    }
}
