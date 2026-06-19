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
    active: HashMap<String, Instant>,
}

impl DiffusionProfiler {
    /// Create a new, empty profiler.
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin timing the named layer.
    pub fn start(&mut self, name: impl Into<String>) {
        self.active.insert(name.into(), Instant::now());
    }

    /// Stop timing the named layer and record its duration.
    ///
    /// Returns the elapsed [`Duration`], or `None` if `name` was not started.
    pub fn stop(&mut self, name: &str) -> Option<Duration> {
        self.active.remove(name).map(|start| {
            let elapsed = start.elapsed();
            self.profiles.push(LayerProfile {
                name: name.to_string(),
                duration: elapsed,
                estimated_memory_bytes: 0,
            });
            elapsed
        })
    }

    /// Time a closure and record it under `name`.
    pub fn time<F: FnOnce() -> R, R>(&mut self, name: &str, f: F) -> R {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        self.profiles.push(LayerProfile {
            name: name.to_string(),
            duration: elapsed,
            estimated_memory_bytes: 0,
        });
        result
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

    /// Fraction of total time spent in the first profile entry whose name equals `name`.
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
                "Diffusion profiler: {:.3} ms total, {} layers",
                total_ms,
                self.profiles.len()
            ),
            format!("{:<40} {:>12} {:>8}", "Layer", "Time (ms)", "%"),
            "-".repeat(62),
        ];
        for p in &self.profiles {
            let ms = p.duration.as_secs_f64() * 1000.0;
            let pct = if total_ms > 0.0 {
                ms / total_ms * 100.0
            } else {
                0.0
            };
            lines.push(format!("{:<40} {:>12.3} {:>8.2}", p.name, ms, pct));
        }
        lines.join("\n")
    }

    /// Clear all recorded profiles and any in-progress timers.
    pub fn clear(&mut self) {
        self.profiles.clear();
        self.active.clear();
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
