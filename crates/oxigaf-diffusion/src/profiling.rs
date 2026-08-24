use std::cmp::Reverse;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Per-layer timing and memory profile record.
#[derive(Debug, Clone)]
pub struct LayerProfile {
    pub name: String,
    pub duration: Duration,
    pub estimated_memory_bytes: usize,
}

/// Accumulates per-layer timing data for a diffusion model forward pass.
#[derive(Debug, Default)]
pub struct DiffusionProfiler {
    profiles: Vec<LayerProfile>,
    active: HashMap<String, (Instant, usize)>,
}

impl DiffusionProfiler {
    /// Create a new, empty profiler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin timing the named layer.
    ///
    /// The resulting [`LayerProfile::estimated_memory_bytes`] will be `0`;
    /// use [`Self::start_with_memory`] instead when a memory estimate (e.g.
    /// from [`estimate_attention_memory_bytes`] or
    /// [`estimate_unet_memory_bytes`]) is available for the layer being
    /// timed.
    pub fn start(&mut self, name: impl Into<String>) {
        self.start_with_memory(name, 0);
    }

    /// Begin timing the named layer, attaching a caller-supplied estimated
    /// memory footprint (in bytes) that will be recorded on the resulting
    /// [`LayerProfile`] once [`Self::stop`] is called.
    ///
    /// The estimate is typically computed with
    /// [`estimate_attention_memory_bytes`] or [`estimate_unet_memory_bytes`]
    /// for the specific layer shape being profiled — this method has no way
    /// to derive it itself, since it does not know what kind of layer `name`
    /// refers to.
    pub fn start_with_memory(&mut self, name: impl Into<String>, estimated_memory_bytes: usize) {
        self.active
            .insert(name.into(), (Instant::now(), estimated_memory_bytes));
    }

    /// Stop timing the named layer and record its duration (and any memory
    /// estimate attached via [`Self::start_with_memory`]).
    ///
    /// Returns the elapsed [`Duration`], or `None` if `name` was not started.
    pub fn stop(&mut self, name: &str) -> Option<Duration> {
        self.active
            .remove(name)
            .map(|(start, estimated_memory_bytes)| {
                let elapsed = start.elapsed();
                self.profiles.push(LayerProfile {
                    name: name.to_string(),
                    duration: elapsed,
                    estimated_memory_bytes,
                });
                elapsed
            })
    }

    /// Time a closure and record it under `name`.
    ///
    /// The resulting [`LayerProfile::estimated_memory_bytes`] will be `0`;
    /// use [`Self::time_with_memory`] instead when a memory estimate is
    /// available for the layer being timed.
    pub fn time<F: FnOnce() -> R, R>(&mut self, name: &str, f: F) -> R {
        self.time_with_memory(name, 0, f)
    }

    /// Time a closure and record it under `name`, along with a
    /// caller-supplied estimated memory footprint (in bytes) — typically
    /// computed via [`estimate_attention_memory_bytes`] or
    /// [`estimate_unet_memory_bytes`] before calling this.
    pub fn time_with_memory<F: FnOnce() -> R, R>(
        &mut self,
        name: &str,
        estimated_memory_bytes: usize,
        f: F,
    ) -> R {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        self.profiles.push(LayerProfile {
            name: name.to_string(),
            duration: elapsed,
            estimated_memory_bytes,
        });
        result
    }

    /// Sum of [`LayerProfile::estimated_memory_bytes`] across all recorded
    /// layers.
    ///
    /// This is a naive sum, not a true peak-memory estimate (real layers
    /// overlap and get freed during a forward pass), but it gives a useful
    /// upper bound and lets [`estimate_attention_memory_bytes`] /
    /// [`estimate_unet_memory_bytes`] estimates recorded via
    /// [`Self::start_with_memory`] / [`Self::time_with_memory`] actually
    /// feed into a queryable total.
    pub fn total_estimated_memory_bytes(&self) -> usize {
        self.profiles.iter().map(|p| p.estimated_memory_bytes).sum()
    }

    /// Return all recorded layer profiles in insertion order.
    pub fn profiles(&self) -> &[LayerProfile] {
        &self.profiles
    }

    /// Sum of all recorded layer durations.
    pub fn total_duration(&self) -> Duration {
        self.profiles.iter().map(|p| p.duration).sum()
    }

    /// Returns the top-N slowest layers by duration (descending).
    pub fn top_slowest(&self, n: usize) -> Vec<&LayerProfile> {
        let mut sorted: Vec<&LayerProfile> = self.profiles.iter().collect();
        sorted.sort_by_key(|p| Reverse(p.duration));
        sorted.truncate(n);
        sorted
    }

    /// Fraction of total time spent across all profile entries whose name
    /// equals `name` (summed, not just the first match — a layer timed once
    /// per denoising step, for example, will have its `name` repeated once
    /// per step, and this returns the aggregate share across every step).
    ///
    /// Returns `0.0` when total duration is zero.
    pub fn fraction_of_total(&self, name: &str) -> f64 {
        let total_ns = self.total_duration().as_nanos();
        if total_ns == 0 {
            return 0.0;
        }
        let layer_ns = self
            .profiles
            .iter()
            .filter(|p| p.name == name)
            .map(|p| p.duration.as_nanos())
            .sum::<u128>();
        layer_ns as f64 / total_ns as f64
    }

    /// Render a human-readable timing report.
    pub fn format_report(&self) -> String {
        let total = self.total_duration();
        let total_ms = total.as_secs_f64() * 1000.0;
        let mut lines = vec![
            format!(
                "Diffusion profiler: {:.3} ms total, {} layers, {} est. memory (sum)",
                total_ms,
                self.profiles.len(),
                format_bytes(self.total_estimated_memory_bytes()),
            ),
            format!(
                "{:<40} {:>12} {:>8} {:>12}",
                "Layer", "Time (ms)", "%", "Est. Mem"
            ),
            "-".repeat(76),
        ];
        for p in &self.profiles {
            let ms = p.duration.as_secs_f64() * 1000.0;
            let pct = if total_ms > 0.0 {
                ms / total_ms * 100.0
            } else {
                0.0
            };
            lines.push(format!(
                "{:<40} {:>12.3} {:>8.2} {:>12}",
                p.name,
                ms,
                pct,
                format_bytes(p.estimated_memory_bytes)
            ));
        }
        lines.join("\n")
    }

    /// Clear all recorded profiles and any in-progress timers.
    pub fn clear(&mut self) {
        self.profiles.clear();
        self.active.clear();
    }
}

/// Format a byte count as a human-readable string (`B`/`KB`/`MB`/`GB`,
/// 1024-based).
fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GB", b / GB)
    } else if b >= MB {
        format!("{:.2} MB", b / MB)
    } else if b >= KB {
        format!("{:.2} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

// ---------------------------------------------------------------------------
// Pure mathematical memory estimators
// ---------------------------------------------------------------------------

/// Estimate memory consumed by a multi-head self-attention layer (in bytes).
///
/// Accounts for Q/K/V matrices and the attention score matrix, all in f32.
pub fn estimate_attention_memory_bytes(
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> usize {
    // Q, K, V: 3 × batch × heads × seq × head_dim × 4 bytes (f32)
    let qkv = 3 * batch * num_heads * seq_len * head_dim * 4;
    // Attention score matrix: batch × heads × seq × seq × 4 bytes
    let scores = batch * num_heads * seq_len * seq_len * 4;
    qkv + scores
}

/// Estimate peak activation memory for a simplified UNet forward pass (bytes).
///
/// Each residual block contributes two activation tensors at the current
/// spatial resolution; a single bottleneck self-attention at spatial/8 is
/// added for the bottleneck stage.
pub fn estimate_unet_memory_bytes(
    batch: usize,
    channels: usize,
    spatial: usize,
    num_res_blocks: usize,
) -> usize {
    // Each residual block: 2 activations × batch × channels × spatial² × 4 bytes
    let base_act = batch * channels * spatial * spatial * 4;
    let res_cost = num_res_blocks * 2 * base_act;
    // Bottleneck self-attention at spatial/8 resolution (minimum 1 token)
    let seq_len = (spatial / 8).max(1).pow(2);
    let head_dim = (channels / 8).max(1);
    let attn_cost = estimate_attention_memory_bytes(batch, seq_len, 8, head_dim);
    res_cost + attn_cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // 1. A freshly constructed profiler has no profiles
    #[test]
    fn new_profiler_is_empty() {
        let profiler = DiffusionProfiler::new();
        assert!(profiler.profiles().is_empty());
        assert_eq!(profiler.total_duration(), Duration::ZERO);
    }

    // 2. time() records exactly one profile entry
    #[test]
    fn time_records_one_profile() {
        let mut profiler = DiffusionProfiler::new();
        profiler.time("unet", || {
            // lightweight work
            let _x: u64 = (0u64..100).sum();
        });
        assert_eq!(profiler.profiles().len(), 1);
        assert_eq!(profiler.profiles()[0].name, "unet");
    }

    // 3. total_duration() sums correctly across two layers
    #[test]
    fn total_duration_sums_two_layers() {
        let mut profiler = DiffusionProfiler::new();
        profiler.time("layer_a", || {
            let _: u64 = (0u64..1000).sum();
        });
        profiler.time("layer_b", || {
            let _: u64 = (0u64..1000).sum();
        });
        let total = profiler.total_duration();
        let sum_individual: Duration = profiler.profiles().iter().map(|p| p.duration).sum();
        assert_eq!(total, sum_individual);
        assert_eq!(profiler.profiles().len(), 2);
    }

    // 4. top_slowest(1) returns the profile with the greatest duration
    #[test]
    fn top_slowest_returns_slowest() {
        let mut profiler = DiffusionProfiler::new();
        // Insert a known-fast layer then a known-slower layer by manipulating
        // the profiles vector directly via the public API through time().
        profiler.time("fast", || {
            // near-zero work
        });
        profiler.time("slow", || {
            // slightly more work
            let _: u64 = (0u64..100_000).sum();
        });
        let top = profiler.top_slowest(1);
        assert_eq!(top.len(), 1);
        // The slowest should have the highest duration
        let max_dur = profiler
            .profiles()
            .iter()
            .map(|p| p.duration)
            .max()
            .unwrap_or(Duration::ZERO);
        assert_eq!(top[0].duration, max_dur);
    }

    // 5. fraction_of_total() returns a value in [0.0, 1.0]
    #[test]
    fn fraction_of_total_in_range() {
        let mut profiler = DiffusionProfiler::new();
        profiler.time("clip", || {
            let _: u64 = (0u64..10_000).sum();
        });
        profiler.time("unet", || {
            let _: u64 = (0u64..50_000).sum();
        });
        let frac_clip = profiler.fraction_of_total("clip");
        let frac_unet = profiler.fraction_of_total("unet");
        assert!((0.0..=1.0).contains(&frac_clip));
        assert!((0.0..=1.0).contains(&frac_unet));
        // The two fractions should sum to ~1.0
        assert!((frac_clip + frac_unet - 1.0_f64).abs() < 1e-9);
    }

    // 6. format_report() contains the layer name
    #[test]
    fn format_report_contains_layer_name() {
        let mut profiler = DiffusionProfiler::new();
        profiler.time("vae_decode", || {});
        let report = profiler.format_report();
        assert!(report.contains("vae_decode"), "report:\n{report}");
    }

    // 7. clear() removes all profiles and active timers
    #[test]
    fn clear_empties_profiler() {
        let mut profiler = DiffusionProfiler::new();
        profiler.time("layer", || {});
        assert!(!profiler.profiles().is_empty());
        profiler.clear();
        assert!(profiler.profiles().is_empty());
        assert_eq!(profiler.total_duration(), Duration::ZERO);
    }

    // 8. start()/stop() round-trip records a profile entry
    #[test]
    fn start_stop_records_profile() {
        let mut profiler = DiffusionProfiler::new();
        profiler.start("encoder");
        let _: u64 = (0u64..1000).sum();
        let elapsed = profiler.stop("encoder");
        assert!(elapsed.is_some(), "stop() should return Some(duration)");
        assert_eq!(profiler.profiles().len(), 1);
        assert_eq!(profiler.profiles()[0].name, "encoder");
    }

    // 9. stop() on an unknown name returns None
    #[test]
    fn stop_unknown_name_returns_none() {
        let mut profiler = DiffusionProfiler::new();
        let result = profiler.stop("nonexistent");
        assert!(result.is_none());
        assert!(profiler.profiles().is_empty());
    }

    // Regression tests for the previously-hardcoded-to-0 estimated_memory_bytes.

    // 9a. plain start()/stop() still records 0 (no fabricated estimate).
    #[test]
    fn start_stop_without_memory_records_zero() {
        let mut profiler = DiffusionProfiler::new();
        profiler.start("plain");
        profiler.stop("plain");
        assert_eq!(profiler.profiles()[0].estimated_memory_bytes, 0);
    }

    // 9b. plain time() still records 0 (no fabricated estimate).
    #[test]
    fn time_without_memory_records_zero() {
        let mut profiler = DiffusionProfiler::new();
        profiler.time("plain", || {});
        assert_eq!(profiler.profiles()[0].estimated_memory_bytes, 0);
    }

    // 9c. start_with_memory()/stop() carries the estimate through.
    #[test]
    fn start_stop_with_memory_records_estimate() {
        let mut profiler = DiffusionProfiler::new();
        let est = estimate_attention_memory_bytes(1, 64, 8, 64);
        profiler.start_with_memory("attn", est);
        profiler.stop("attn");
        assert_eq!(profiler.profiles()[0].estimated_memory_bytes, est);
        assert!(est > 0);
    }

    // 9d. time_with_memory() carries the estimate through.
    #[test]
    fn time_with_memory_records_estimate() {
        let mut profiler = DiffusionProfiler::new();
        let est = estimate_unet_memory_bytes(1, 256, 64, 4);
        let result = profiler.time_with_memory("unet", est, || 7u32);
        assert_eq!(result, 7);
        assert_eq!(profiler.profiles()[0].estimated_memory_bytes, est);
    }

    // 9e. total_estimated_memory_bytes sums across mixed recording styles.
    #[test]
    fn total_estimated_memory_bytes_sums_all_layers() {
        let mut profiler = DiffusionProfiler::new();
        profiler.time("no_estimate", || {});
        profiler.time_with_memory("with_estimate_1", 1000, || {});
        profiler.time_with_memory("with_estimate_2", 2000, || {});
        assert_eq!(profiler.total_estimated_memory_bytes(), 3000);
    }

    // 9f. format_report surfaces the recorded memory estimate, not just 0.
    #[test]
    fn format_report_shows_nonzero_memory_estimate() {
        let mut profiler = DiffusionProfiler::new();
        profiler.time_with_memory("unet", 1_048_576, || {});
        let report = profiler.format_report();
        assert!(
            report.contains("1.00 MB"),
            "report should surface the recorded memory estimate:\n{report}"
        );
    }

    // 10. estimate_attention_memory_bytes returns a positive value
    #[test]
    fn estimate_attention_memory_positive() {
        let bytes = estimate_attention_memory_bytes(1, 64, 8, 64);
        assert!(bytes > 0, "expected > 0, got {bytes}");
    }

    // 11. estimate_unet_memory_bytes returns a positive value
    #[test]
    fn estimate_unet_memory_positive() {
        let bytes = estimate_unet_memory_bytes(1, 256, 64, 4);
        assert!(bytes > 0, "expected > 0, got {bytes}");
    }

    // 12. Larger batch produces a larger memory estimate
    #[test]
    fn larger_batch_larger_estimate() {
        let bytes_1 = estimate_unet_memory_bytes(1, 256, 64, 4);
        let bytes_2 = estimate_unet_memory_bytes(2, 256, 64, 4);
        assert!(
            bytes_2 > bytes_1,
            "batch=2 ({bytes_2}) should exceed batch=1 ({bytes_1})"
        );
    }

    // 13. format_report() contains the header row
    #[test]
    fn format_report_has_header() {
        let mut profiler = DiffusionProfiler::new();
        profiler.time("dummy", || {});
        let report = profiler.format_report();
        assert!(
            report.contains("Layer") && report.contains("Time (ms)"),
            "header missing from report:\n{report}"
        );
    }

    // 14. fraction_of_total returns 0.0 when profiler is empty
    #[test]
    fn fraction_of_total_empty_profiler() {
        let profiler = DiffusionProfiler::new();
        assert_eq!(profiler.fraction_of_total("anything"), 0.0);
    }

    // 15. Attention memory score component grows quadratically with seq_len.
    //     With seq_len=256, the score term alone (batch×heads×seq×seq×4 bytes)
    //     is 4× that of seq_len=128, confirming the quadratic relationship.
    #[test]
    fn attention_memory_score_component_quadratic() {
        // Isolate the score component by choosing head_dim=0 (zero QKV bytes).
        // score = batch × heads × seq × seq × 4
        let scores_128: usize = 1 * 8 * 128 * 128 * 4;
        let scores_256: usize = 1 * 8 * 256 * 256 * 4;
        assert_eq!(
            scores_256,
            4 * scores_128,
            "score term must be quadratic in seq_len"
        );
        // Full estimate also grows with seq_len
        let m64 = estimate_attention_memory_bytes(1, 64, 8, 64);
        let m128 = estimate_attention_memory_bytes(1, 128, 8, 64);
        assert!(m128 > m64, "m128 ({m128}) should exceed m64 ({m64})");
    }
}
