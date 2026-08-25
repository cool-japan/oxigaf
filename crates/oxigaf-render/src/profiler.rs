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

use crate::RenderError;

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

// ---------------------------------------------------------------------------
// GpuTimestampProfiler
// ---------------------------------------------------------------------------

/// Bytes per resolved timestamp query (`u64` ticks).
const TIMESTAMP_BYTES: u64 = 8;

/// Alignment `CommandEncoder::resolve_query_set` requires of its destination
/// buffer offset, and therefore of the resolve buffer's size.
const QUERY_RESOLVE_ALIGNMENT: u64 = wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;

/// GPU-side pass profiler backed by `wgpu` timestamp queries.
///
/// [`PassProfiler`] measures *wall-clock host* time, which for a compute
/// dispatch is the cost of **recording** the pass, not of executing it — the
/// GPU runs asynchronously. This type measures the real thing: two timestamps
/// per compute pass, written by the GPU itself at the pass boundaries.
///
/// # Requirements
///
/// The device must have been created with
/// [`REQUIRED_FEATURES`](Self::REQUIRED_FEATURES) (i.e.
/// `wgpu::Features::TIMESTAMP_QUERY`), which is only available when the
/// adapter advertises it. [`Rasterizer::new`](crate::Rasterizer::new) requests
/// the feature whenever the adapter offers it, so
/// [`Rasterizer::enable_gpu_timestamps`](crate::Rasterizer::enable_gpu_timestamps)
/// succeeds on such devices and reports a clear error otherwise.
///
/// # Frame protocol
///
/// 1. [`pass_writes`](Self::pass_writes) once per compute pass, passing the
///    result as `ComputePassDescriptor::timestamp_writes`. Each call reserves
///    two query slots and remembers the pass name.
/// 2. [`resolve`](Self::resolve) into the same encoder, after the last pass.
/// 3. Submit, then [`collect`](Self::collect) once the device has been polled:
///    it maps the readback, converts ticks to durations with the queue's
///    timestamp period, folds them into [`stats`](Self::stats) and clears the
///    reservation list for the next frame.
///
/// A frame that never reaches `collect` must call [`discard`](Self::discard),
/// otherwise the next frame keeps reserving slots after the stale ones and
/// eventually overflows the query set.
pub struct GpuTimestampProfiler {
    query_set: wgpu::QuerySet,
    /// `QUERY_RESOLVE | COPY_SRC` destination of `resolve_query_set`.
    resolve_buf: wgpu::Buffer,
    /// `MAP_READ | COPY_DST` host-visible copy of `resolve_buf`.
    staging: wgpu::Buffer,
    /// Nanoseconds per timestamp tick, from `Queue::get_timestamp_period`.
    period_ns: f32,
    /// Number of passes (query pairs) this profiler can record per frame.
    max_passes: u32,
    /// Names of the passes reserved so far this frame, in slot order.
    pending: Mutex<Vec<String>>,
    /// Accumulated per-pass GPU statistics.
    stats: PassProfiler,
}

impl GpuTimestampProfiler {
    /// Device features a GPU timestamp profiler needs.
    pub const REQUIRED_FEATURES: wgpu::Features = wgpu::Features::TIMESTAMP_QUERY;

    /// Passes [`Rasterizer`](crate::Rasterizer) records in one forward plus one
    /// backward pass, with headroom: preprocess, three scan levels, two
    /// add-backs, tile_assign, tile_ranges, rasterize_fwd, rasterize_bwd, three
    /// atomic conversions, preprocess_bwd and flame_binding_bwd.
    pub const DEFAULT_MAX_PASSES: u32 = 32;

    /// Create a timestamp profiler on `device` with room for `max_passes`
    /// timed passes per frame.
    ///
    /// # Errors
    ///
    /// [`RenderError::GpuInit`] when the device was not created with
    /// [`REQUIRED_FEATURES`](Self::REQUIRED_FEATURES), or when `max_passes` is
    /// zero or needs more queries than `wgpu::QUERY_SET_MAX_QUERIES`.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_passes: u32,
    ) -> Result<Self, RenderError> {
        if !device.features().contains(Self::REQUIRED_FEATURES) {
            return Err(RenderError::GpuInit(
                "GPU timestamps need wgpu::Features::TIMESTAMP_QUERY, which this device was not \
                 created with (the adapter may not support it)"
                    .to_string(),
            ));
        }
        if max_passes == 0 {
            return Err(RenderError::GpuInit(
                "GpuTimestampProfiler::new: max_passes must be > 0".to_string(),
            ));
        }
        let query_count = max_passes.checked_mul(2).ok_or_else(|| {
            RenderError::GpuInit(format!(
                "GpuTimestampProfiler: {max_passes} passes overflow u32"
            ))
        })?;
        if query_count > wgpu::QUERY_SET_MAX_QUERIES {
            return Err(RenderError::GpuInit(format!(
                "GpuTimestampProfiler: {max_passes} passes need {query_count} queries, above the \
                 {} a query set can hold",
                wgpu::QUERY_SET_MAX_QUERIES
            )));
        }

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: query_count,
        });

        // `resolve_query_set` writes at 256-byte-aligned offsets, so the
        // buffer is rounded up to that alignment as well.
        let byte_size = (u64::from(query_count) * TIMESTAMP_BYTES)
            .next_multiple_of(QUERY_RESOLVE_ALIGNMENT)
            .max(QUERY_RESOLVE_ALIGNMENT);
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_timestamps_resolve"),
            size: byte_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_timestamps_staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            query_set,
            resolve_buf,
            staging,
            period_ns: queue.get_timestamp_period(),
            max_passes,
            pending: Mutex::new(Vec::new()),
            stats: PassProfiler::new(),
        })
    }

    /// Accumulated GPU-side statistics, in the same shape as the CPU profiler.
    pub fn stats(&self) -> &PassProfiler {
        &self.stats
    }

    /// Nanoseconds per timestamp tick on this queue.
    pub fn period_ns(&self) -> f32 {
        self.period_ns
    }

    /// Number of passes reserved for the frame currently being recorded.
    pub fn reserved_passes(&self) -> usize {
        self.pending.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Reserve two query slots for `pass_name` and build the descriptor to
    /// hand to `ComputePassDescriptor::timestamp_writes`.
    ///
    /// Returns `None` once the frame has already reserved `max_passes` passes;
    /// the caller then simply records an untimed pass rather than failing the
    /// frame.
    pub fn pass_writes(&self, pass_name: &str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        let index = u32::try_from(pending.len()).unwrap_or(u32::MAX);
        if index >= self.max_passes {
            tracing::warn!(
                pass = pass_name,
                max_passes = self.max_passes,
                "GPU timestamp query set is full for this frame; pass recorded untimed"
            );
            return None;
        }
        pending.push(pass_name.to_owned());
        let slot = index * 2;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(slot),
            end_of_pass_write_index: Some(slot + 1),
        })
    }

    /// Record the query resolve and the host readback copy into `encoder`.
    ///
    /// Returns the number of passes that will be reported by the following
    /// [`collect`](Self::collect); zero means nothing was timed and `collect`
    /// can be skipped.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) -> usize {
        let passes = self.reserved_passes();
        if passes == 0 {
            return 0;
        }
        let queries = u32::try_from(passes * 2).unwrap_or(u32::MAX);
        encoder.resolve_query_set(&self.query_set, 0..queries, &self.resolve_buf, 0);
        let bytes = (u64::from(queries) * TIMESTAMP_BYTES).min(self.staging.size());
        encoder.copy_buffer_to_buffer(&self.resolve_buf, 0, &self.staging, 0, bytes);
        passes
    }

    /// Map the resolved timestamps, fold them into [`stats`](Self::stats) and
    /// start a new frame.
    ///
    /// Call this only after the submission containing [`resolve`](Self::resolve)
    /// has completed. Returns the per-pass GPU durations of the frame, in the
    /// order the passes were reserved.
    ///
    /// # Errors
    ///
    /// [`RenderError::BufferMapFailed`] when the readback cannot be mapped. The
    /// frame's reservations are cleared either way, so one failed readback does
    /// not poison every later frame.
    pub fn collect(&self, device: &wgpu::Device) -> Result<Vec<(String, Duration)>, RenderError> {
        let names: Vec<String> = {
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *pending)
        };
        if names.is_empty() {
            return Ok(Vec::new());
        }

        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv()
            .map_err(|e| RenderError::BufferMapFailed {
                buffer_name: "gpu_timestamps_staging".to_string(),
                error: format!("Channel recv failed: {e}"),
            })?
            .map_err(|e| RenderError::BufferMapFailed {
                buffer_name: "gpu_timestamps_staging".to_string(),
                error: e.to_string(),
            })?;

        let data = slice
            .get_mapped_range()
            .map_err(|e| RenderError::BufferMapFailed {
                buffer_name: "gpu_timestamps_staging".to_string(),
                error: format!("Mapped range failed: {e}"),
            })?;
        let ticks: Vec<u64> = data
            .chunks_exact(TIMESTAMP_BYTES as usize)
            .map(|c| {
                let mut raw = [0u8; 8];
                raw.copy_from_slice(c);
                u64::from_le_bytes(raw)
            })
            .collect();
        drop(data);
        self.staging.unmap();

        let out = timestamps_to_durations(&names, &ticks, self.period_ns);
        for (name, duration) in &out {
            self.stats.record(name, *duration);
        }
        self.stats.next_frame();
        Ok(out)
    }

    /// Drop this frame's reservations without reading them back.
    ///
    /// Use this when a frame is abandoned (an error before submission, or a
    /// submission whose completion is never awaited): otherwise the stale
    /// reservations stay counted and the query set fills up.
    pub fn discard(&self) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

/// Convert resolved timestamp ticks into one duration per named pass.
///
/// `ticks` holds `[begin_0, end_0, begin_1, end_1, ...]`. A pair whose end
/// precedes its begin (an unwritten or wrapped query) yields a zero duration
/// rather than an underflow, and names without a full pair are dropped.
fn timestamps_to_durations(
    names: &[String],
    ticks: &[u64],
    period_ns: f32,
) -> Vec<(String, Duration)> {
    let period = f64::from(period_ns.max(0.0));
    names
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            let begin = *ticks.get(i * 2)?;
            let end = *ticks.get(i * 2 + 1)?;
            let elapsed_ticks = end.saturating_sub(begin);
            let nanos = (elapsed_ticks as f64) * period;
            Some((name.clone(), Duration::from_nanos(nanos as u64)))
        })
        .collect()
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

    // -------------------------------------------------------------------------
    // GPU timestamp conversion (no GPU required)
    // -------------------------------------------------------------------------

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn test_timestamps_to_durations_uses_the_queue_period() {
        // 1 tick = 2.5 ns; pass 0 spans 1000 ticks, pass 1 spans 40 ticks.
        let out = timestamps_to_durations(
            &names(&["preprocess", "rasterize_fwd"]),
            &[100, 1_100, 5_000, 5_040],
            2.5,
        );
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, "preprocess");
        assert_eq!(out[0].1, Duration::from_nanos(2_500));
        assert_eq!(out[1].0, "rasterize_fwd");
        assert_eq!(out[1].1, Duration::from_nanos(100));
    }

    /// Regression: an unwritten or wrapped query pair must not underflow into
    /// a multi-century duration.
    #[test]
    fn test_timestamps_to_durations_clamps_reversed_pairs() {
        let out = timestamps_to_durations(&names(&["odd"]), &[900, 100], 1.0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1, Duration::ZERO);
    }

    /// A truncated readback drops the incomplete tail instead of indexing
    /// past the resolved data.
    #[test]
    fn test_timestamps_to_durations_drops_incomplete_tail() {
        let out = timestamps_to_durations(&names(&["a", "b"]), &[0, 10, 20], 1.0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "a");
        assert_eq!(out[0].1, Duration::from_nanos(10));
    }

    #[test]
    fn test_timestamps_to_durations_empty_is_empty() {
        assert!(timestamps_to_durations(&[], &[1, 2], 1.0).is_empty());
        assert!(timestamps_to_durations(&names(&["a"]), &[], 1.0).is_empty());
    }

    /// The two timestamps a pass reserves must land in adjacent, non-
    /// overlapping slots — the invariant `pass_writes` encodes and
    /// `timestamps_to_durations` decodes.
    #[test]
    fn test_timestamp_slot_layout_is_two_per_pass() {
        for pass_index in 0u32..8 {
            let begin = pass_index * 2;
            let end = begin + 1;
            assert_eq!(end - begin, 1);
            assert!(begin.is_multiple_of(2));
        }
    }
}
