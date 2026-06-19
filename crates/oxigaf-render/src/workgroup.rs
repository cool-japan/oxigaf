//! Workgroup size configuration and benchmark-based auto-selection.
//!
//! This module provides a `WorkgroupConfig` system for choosing optimal GPU
//! dispatch sizes across different GPU architectures, along with a CPU-side
//! benchmarker that measures timing of user-supplied closures to recommend
//! the best configuration.
//!
//! # Example
//!
//! ```rust
//! use oxigaf_render::workgroup::{WorkgroupConfig, WorkgroupProfile, WorkgroupBenchmarker};
//!
//! // Use a preset profile
//! let config = WorkgroupConfig::balanced();
//! assert_eq!(config.profile, WorkgroupProfile::Balanced);
//!
//! // Adapt automatically to the Gaussian count
//! let config = WorkgroupConfig::adaptive(50_000);
//!
//! // Benchmark-driven recommendation
//! let benchmarker = WorkgroupBenchmarker::new();
//! let config = benchmarker.recommend(50_000, |ws| {
//!     // Simulate dispatch work proportional to workgroup total
//!     let start = std::time::Instant::now();
//!     let _ = ws.total(); // trivial placeholder
//!     start.elapsed()
//! });
//! ```

use crate::RenderError;

// ---------------------------------------------------------------------------
// WorkgroupSize
// ---------------------------------------------------------------------------

/// Three-dimensional workgroup size for compute shaders.
///
/// For 1-D compute shaders (most Gaussian processing passes) use
/// [`WorkgroupSize::linear`]. For 2-D tile operations use
/// [`WorkgroupSize::square`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkgroupSize {
    /// Workgroup threads in the X dimension.
    pub x: u32,
    /// Workgroup threads in the Y dimension.
    pub y: u32,
    /// Workgroup threads in the Z dimension.
    pub z: u32,
}

impl WorkgroupSize {
    /// Create a workgroup size with explicit x/y/z dimensions.
    #[inline]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Create a 1-D workgroup of `n` threads (y = z = 1).
    #[inline]
    pub const fn linear(n: u32) -> Self {
        Self::new(n, 1, 1)
    }

    /// Create a 2-D square workgroup of `n × n` threads (z = 1).
    #[inline]
    pub const fn square(n: u32) -> Self {
        Self::new(n, n, 1)
    }

    /// Total number of threads in this workgroup.
    #[inline]
    pub fn total(&self) -> u32 {
        self.x * self.y * self.z
    }

    /// Number of workgroups needed to cover `n` threads in the X dimension
    /// (rounds up to avoid leaving elements unprocessed).
    #[inline]
    pub fn dispatch_count_x(&self, n: u32) -> u32 {
        n.div_ceil(self.x)
    }

    /// Number of workgroups needed to cover `n` threads in the Y dimension
    /// (rounds up).
    #[inline]
    pub fn dispatch_count_y(&self, n: u32) -> u32 {
        n.div_ceil(self.y)
    }
}

// ---------------------------------------------------------------------------
// WorkgroupProfile
// ---------------------------------------------------------------------------

/// Preset workgroup size profiles targeting different GPU classes.
///
/// `WorkgroupProfile` is *workload-oriented* (small vs. large thread counts)
/// while [`crate::config::GpuPreset`] is *vendor-oriented*.  The two can be
/// used together or independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkgroupProfile {
    /// Mobile / integrated GPUs: 32 threads per workgroup.
    Mobile,
    /// Balanced default: 64 threads per workgroup (good for most GPUs).
    Balanced,
    /// High-throughput desktop GPUs: 256 threads per workgroup.
    HighThroughput,
    /// User-defined configuration; the profile field acts as a tag only.
    Custom,
}

impl WorkgroupProfile {
    /// Return the default 1-D workgroup size for this profile.
    #[must_use]
    pub fn default_size(&self) -> WorkgroupSize {
        match self {
            Self::Mobile => WorkgroupSize::linear(32),
            Self::Balanced => WorkgroupSize::linear(64),
            Self::HighThroughput => WorkgroupSize::linear(256),
            Self::Custom => WorkgroupSize::linear(64), // sensible fallback
        }
    }

    /// Short machine-readable name for this profile.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Mobile => "mobile",
            Self::Balanced => "balanced",
            Self::HighThroughput => "high_throughput",
            Self::Custom => "custom",
        }
    }

    /// Human-readable description of this profile.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Mobile => "Small workgroups (32 threads) for mobile/integrated GPUs",
            Self::Balanced => "Balanced workgroups (64 threads) suitable for most GPUs",
            Self::HighThroughput => {
                "Large workgroups (256 threads) for powerful desktop/server GPUs"
            }
            Self::Custom => "User-defined workgroup configuration",
        }
    }
}

// ---------------------------------------------------------------------------
// WorkgroupConfig
// ---------------------------------------------------------------------------

/// Complete workgroup configuration covering all rasterization passes.
///
/// Each pass can be tuned independently; use [`WorkgroupConfig::from_profile`]
/// to start from a preset and then override individual fields as needed.
#[derive(Debug, Clone)]
pub struct WorkgroupConfig {
    /// The profile that was used to create this configuration.
    pub profile: WorkgroupProfile,

    /// Preprocess pass (Gaussian projection, covariance computation).
    pub preprocess: WorkgroupSize,

    /// Sorting pass (radix sort over depth-keyed Gaussians).
    pub sort: WorkgroupSize,

    /// Forward rasterization pass (per-tile alpha-blending).
    pub rasterize: WorkgroupSize,

    /// Backward pass (gradient computation through rasterizer).
    pub backward: WorkgroupSize,

    /// Tile-based operations (2-D, typically square).
    pub tile: WorkgroupSize,
}

impl WorkgroupConfig {
    /// Build a `WorkgroupConfig` from a [`WorkgroupProfile`].
    ///
    /// All 1-D passes use the profile's default linear size; the tile pass
    /// always uses a 4 × 4 square (16 threads) as tile-based passes are
    /// inherently two-dimensional.
    #[must_use]
    pub fn from_profile(profile: WorkgroupProfile) -> Self {
        let size = profile.default_size();
        Self {
            profile,
            preprocess: size,
            sort: size,
            rasterize: size,
            backward: size,
            tile: WorkgroupSize::square(4), // 4×4 = 16 threads; tile is 2-D
        }
    }

    /// Preset configuration suitable for mobile / integrated GPUs.
    #[must_use]
    pub fn mobile() -> Self {
        Self::from_profile(WorkgroupProfile::Mobile)
    }

    /// Preset configuration suitable for most GPUs (default).
    #[must_use]
    pub fn balanced() -> Self {
        Self::from_profile(WorkgroupProfile::Balanced)
    }

    /// Preset configuration for high-throughput desktop/server GPUs.
    #[must_use]
    pub fn high_throughput() -> Self {
        Self::from_profile(WorkgroupProfile::HighThroughput)
    }

    /// Choose a profile adaptively based on the number of Gaussians.
    ///
    /// | Gaussian count   | Profile       |
    /// |-----------------|---------------|
    /// | < 10 000        | Mobile        |
    /// | 10 000 – 99 999 | Balanced      |
    /// | ≥ 100 000       | HighThroughput|
    #[must_use]
    pub fn adaptive(num_gaussians: usize) -> Self {
        if num_gaussians < 10_000 {
            Self::mobile()
        } else if num_gaussians < 100_000 {
            Self::balanced()
        } else {
            Self::high_throughput()
        }
    }

    /// Validate that all workgroup sizes are well-formed.
    ///
    /// Checks:
    /// - All dimensions are non-zero.
    /// - Total threads per workgroup do not exceed 1024 (safe WebGPU / Vulkan limit).
    /// - Each dimension is a power of two (required by most GPU architectures for
    ///   efficient scheduling).
    pub fn validate(&self) -> Result<(), RenderError> {
        let passes = [
            ("preprocess", self.preprocess),
            ("sort", self.sort),
            ("rasterize", self.rasterize),
            ("backward", self.backward),
            ("tile", self.tile),
        ];

        for (name, ws) in passes {
            if ws.x == 0 || ws.y == 0 || ws.z == 0 {
                return Err(RenderError::Rasterize(format!(
                    "workgroup '{}' has a zero dimension: {:?}",
                    name, ws
                )));
            }

            let total = ws.total();
            if total > 1024 {
                return Err(RenderError::Rasterize(format!(
                    "workgroup '{}' total threads {} exceeds maximum 1024",
                    name, total
                )));
            }

            if !is_power_of_two(ws.x) || !is_power_of_two(ws.y) || !is_power_of_two(ws.z) {
                return Err(RenderError::Rasterize(format!(
                    "workgroup '{}' dimensions must be powers of two, got {:?}",
                    name, ws
                )));
            }
        }

        Ok(())
    }
}

impl Default for WorkgroupConfig {
    fn default() -> Self {
        Self::balanced()
    }
}

/// Returns `true` if `n` is a power of two (and non-zero).
#[inline]
fn is_power_of_two(n: u32) -> bool {
    n != 0 && n.is_power_of_two()
}

// ---------------------------------------------------------------------------
// WorkgroupBenchResult
// ---------------------------------------------------------------------------

/// Result of benchmarking a single workgroup size.
#[derive(Debug, Clone)]
pub struct WorkgroupBenchResult {
    /// The workgroup size that was benchmarked.
    pub size: WorkgroupSize,
    /// Arithmetic mean of measured durations in microseconds.
    pub mean_duration_us: f64,
    /// Minimum measured duration across all samples, in microseconds.
    pub min_duration_us: f64,
    /// Number of measurement samples taken.
    pub samples: usize,
}

// ---------------------------------------------------------------------------
// WorkgroupBenchmarker
// ---------------------------------------------------------------------------

/// CPU-side benchmarker for selecting optimal workgroup sizes.
///
/// The benchmarker tests a set of candidate 1-D workgroup sizes by calling
/// a user-supplied closure and measuring wall-clock time.  The closure is
/// responsible for simulating or performing the dispatch work; the results
/// are used purely to rank the candidates.
///
/// # Example
///
/// ```rust
/// use std::time::Instant;
/// use oxigaf_render::workgroup::{WorkgroupBenchmarker, WorkgroupSize};
///
/// let benchmarker = WorkgroupBenchmarker::new()
///     .with_warmup(2)
///     .with_measure(5);
///
/// let results = benchmarker.benchmark(|ws| {
///     let start = Instant::now();
///     let _n = ws.total(); // placeholder for real work
///     start.elapsed()
/// });
///
/// let best = benchmarker.best_of(&results);
/// ```
pub struct WorkgroupBenchmarker {
    /// Candidate linear workgroup sizes to benchmark.
    candidates: Vec<u32>,
    /// Number of warm-up invocations (results discarded).
    warmup_rounds: usize,
    /// Number of measurement invocations per candidate.
    measure_rounds: usize,
}

impl WorkgroupBenchmarker {
    /// Create a benchmarker with sensible defaults:
    /// - Candidates: 32, 64, 128, 256
    /// - Warm-up rounds: 3
    /// - Measurement rounds: 10
    #[must_use]
    pub fn new() -> Self {
        Self {
            candidates: vec![32, 64, 128, 256],
            warmup_rounds: 3,
            measure_rounds: 10,
        }
    }

    /// Override the candidate workgroup sizes (linear, 1-D).
    ///
    /// Each value becomes a `WorkgroupSize::linear(n)` internally.
    #[must_use]
    pub fn with_candidates(mut self, sizes: Vec<u32>) -> Self {
        self.candidates = sizes;
        self
    }

    /// Set the number of warm-up rounds (results discarded).
    #[must_use]
    pub fn with_warmup(mut self, rounds: usize) -> Self {
        self.warmup_rounds = rounds;
        self
    }

    /// Set the number of measurement rounds per candidate.
    #[must_use]
    pub fn with_measure(mut self, rounds: usize) -> Self {
        self.measure_rounds = rounds;
        self
    }

    /// Benchmark `f` for every candidate workgroup size.
    ///
    /// For each candidate the closure is called `warmup_rounds` times
    /// (results discarded) and then `measure_rounds` times for the actual
    /// measurement.  Each [`WorkgroupBenchResult`] contains the mean and
    /// minimum measured durations in microseconds.
    pub fn benchmark<F>(&self, f: F) -> Vec<WorkgroupBenchResult>
    where
        F: Fn(WorkgroupSize) -> std::time::Duration,
    {
        let mut results = Vec::with_capacity(self.candidates.len());

        for &size_n in &self.candidates {
            let ws = WorkgroupSize::linear(size_n);

            // Warm-up: prime caches, JIT, branch predictors, etc.
            for _ in 0..self.warmup_rounds {
                let _ = f(ws);
            }

            // Measurement
            let mut durations_us: Vec<f64> = Vec::with_capacity(self.measure_rounds);
            for _ in 0..self.measure_rounds {
                let d = f(ws);
                durations_us.push(duration_to_us(d));
            }

            let n = durations_us.len();
            let mean = if n == 0 {
                0.0
            } else {
                durations_us.iter().copied().sum::<f64>() / n as f64
            };

            let min = durations_us.iter().copied().fold(f64::INFINITY, f64::min);

            results.push(WorkgroupBenchResult {
                size: ws,
                mean_duration_us: mean,
                min_duration_us: if min.is_infinite() { 0.0 } else { min },
                samples: n,
            });
        }

        results
    }

    /// Return the workgroup size with the lowest mean duration, or `None` if
    /// `results` is empty.
    #[must_use]
    pub fn best_of(&self, results: &[WorkgroupBenchResult]) -> Option<WorkgroupSize> {
        results
            .iter()
            .min_by(|a, b| {
                a.mean_duration_us
                    .partial_cmp(&b.mean_duration_us)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.size)
    }

    /// Run the benchmark and return a recommended [`WorkgroupConfig`].
    ///
    /// The best workgroup size found is applied to all passes (preprocess,
    /// sort, rasterize, backward) while the tile pass always uses a 4 × 4
    /// square.  If the benchmark yields no results (e.g., no candidates),
    /// falls back to [`WorkgroupConfig::adaptive`].
    #[must_use]
    pub fn recommend<F>(&self, num_gaussians: usize, f: F) -> WorkgroupConfig
    where
        F: Fn(WorkgroupSize) -> std::time::Duration,
    {
        let results = self.benchmark(f);

        match self.best_of(&results) {
            Some(best_size) => WorkgroupConfig {
                profile: WorkgroupProfile::Custom,
                preprocess: best_size,
                sort: best_size,
                rasterize: best_size,
                backward: best_size,
                tile: WorkgroupSize::square(4),
            },
            None => WorkgroupConfig::adaptive(num_gaussians),
        }
    }
}

impl Default for WorkgroupBenchmarker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a [`std::time::Duration`] to microseconds as `f64`.
#[inline]
fn duration_to_us(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1_000_000.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    // --- WorkgroupSize basic ---

    #[test]
    fn test_linear_total() {
        assert_eq!(WorkgroupSize::linear(64).total(), 64);
    }

    #[test]
    fn test_square_total() {
        assert_eq!(WorkgroupSize::square(8).total(), 64);
    }

    #[test]
    fn test_dispatch_count_x_rounds_up() {
        let ws = WorkgroupSize::linear(64);
        // 65 threads → needs 2 workgroups (128 threads dispatched)
        assert_eq!(ws.dispatch_count_x(65), 2);
        // 1 thread → needs 1 workgroup
        assert_eq!(ws.dispatch_count_x(1), 1);
    }

    #[test]
    fn test_dispatch_count_x_exact_multiple() {
        let ws = WorkgroupSize::linear(64);
        // 128 is exactly 2 × 64, should not round up
        assert_eq!(ws.dispatch_count_x(128), 2);
    }

    #[test]
    fn test_dispatch_count_y_rounds_up() {
        let ws = WorkgroupSize::square(8);
        assert_eq!(ws.dispatch_count_y(9), 2);
        assert_eq!(ws.dispatch_count_y(8), 1);
    }

    // --- WorkgroupProfile ---

    #[test]
    fn test_profile_mobile_total() {
        assert_eq!(WorkgroupProfile::Mobile.default_size().total(), 32);
    }

    #[test]
    fn test_profile_balanced_total() {
        assert_eq!(WorkgroupProfile::Balanced.default_size().total(), 64);
    }

    #[test]
    fn test_profile_high_throughput_total() {
        assert_eq!(WorkgroupProfile::HighThroughput.default_size().total(), 256);
    }

    #[test]
    fn test_profile_name() {
        assert_eq!(WorkgroupProfile::Mobile.name(), "mobile");
        assert_eq!(WorkgroupProfile::Balanced.name(), "balanced");
        assert_eq!(WorkgroupProfile::HighThroughput.name(), "high_throughput");
        assert_eq!(WorkgroupProfile::Custom.name(), "custom");
    }

    #[test]
    fn test_profile_description_non_empty() {
        for profile in [
            WorkgroupProfile::Mobile,
            WorkgroupProfile::Balanced,
            WorkgroupProfile::HighThroughput,
            WorkgroupProfile::Custom,
        ] {
            assert!(!profile.description().is_empty());
        }
    }

    // --- WorkgroupConfig construction ---

    #[test]
    fn test_mobile_preprocess_total() {
        assert_eq!(WorkgroupConfig::mobile().preprocess.total(), 32);
    }

    #[test]
    fn test_balanced_profile_tag() {
        assert_eq!(
            WorkgroupConfig::balanced().profile,
            WorkgroupProfile::Balanced
        );
    }

    #[test]
    fn test_adaptive_small() {
        let cfg = WorkgroupConfig::adaptive(5_000);
        assert_eq!(cfg.profile, WorkgroupProfile::Mobile);
    }

    #[test]
    fn test_adaptive_medium() {
        let cfg = WorkgroupConfig::adaptive(50_000);
        assert_eq!(cfg.profile, WorkgroupProfile::Balanced);
    }

    #[test]
    fn test_adaptive_large() {
        let cfg = WorkgroupConfig::adaptive(500_000);
        assert_eq!(cfg.profile, WorkgroupProfile::HighThroughput);
    }

    #[test]
    fn test_default_is_balanced() {
        let cfg = WorkgroupConfig::default();
        assert_eq!(cfg.profile, WorkgroupProfile::Balanced);
    }

    // --- Validation ---

    #[test]
    fn test_validate_balanced_ok() {
        assert!(WorkgroupConfig::balanced().validate().is_ok());
    }

    #[test]
    fn test_validate_mobile_ok() {
        assert!(WorkgroupConfig::mobile().validate().is_ok());
    }

    #[test]
    fn test_validate_high_throughput_ok() {
        assert!(WorkgroupConfig::high_throughput().validate().is_ok());
    }

    #[test]
    fn test_validate_zero_dimension_fails() {
        let mut cfg = WorkgroupConfig::balanced();
        cfg.preprocess = WorkgroupSize::new(0, 1, 1);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_exceeds_max_threads_fails() {
        let mut cfg = WorkgroupConfig::balanced();
        // 32 × 32 × 2 = 2048 > 1024
        cfg.sort = WorkgroupSize::new(32, 32, 2);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_non_power_of_two_fails() {
        let mut cfg = WorkgroupConfig::balanced();
        cfg.backward = WorkgroupSize::new(48, 1, 1); // 48 is not a power of 2
        assert!(cfg.validate().is_err());
    }

    // --- WorkgroupBenchmarker ---

    #[test]
    fn test_benchmarker_default_candidates() {
        let b = WorkgroupBenchmarker::new();
        assert!(!b.candidates.is_empty());
        assert!(b.candidates.contains(&32));
        assert!(b.candidates.contains(&64));
        assert!(b.candidates.contains(&128));
        assert!(b.candidates.contains(&256));
    }

    #[test]
    fn test_benchmark_result_count() {
        let b = WorkgroupBenchmarker::new().with_warmup(1).with_measure(2);
        let results = b.benchmark(|_ws| Duration::from_nanos(100));
        // Should produce one result per default candidate (4)
        assert_eq!(results.len(), b.candidates.len());
    }

    #[test]
    fn test_best_of_returns_lowest_mean() {
        let b = WorkgroupBenchmarker::new();

        let results = vec![
            WorkgroupBenchResult {
                size: WorkgroupSize::linear(32),
                mean_duration_us: 200.0,
                min_duration_us: 180.0,
                samples: 5,
            },
            WorkgroupBenchResult {
                size: WorkgroupSize::linear(64),
                mean_duration_us: 50.0,
                min_duration_us: 45.0,
                samples: 5,
            },
            WorkgroupBenchResult {
                size: WorkgroupSize::linear(128),
                mean_duration_us: 120.0,
                min_duration_us: 110.0,
                samples: 5,
            },
        ];

        let best = b.best_of(&results);
        assert_eq!(best, Some(WorkgroupSize::linear(64)));
    }

    #[test]
    fn test_best_of_empty_returns_none() {
        let b = WorkgroupBenchmarker::new();
        assert_eq!(b.best_of(&[]), None);
    }

    #[test]
    fn test_recommend_returns_config() {
        let b = WorkgroupBenchmarker::new().with_warmup(1).with_measure(2);

        let config = b.recommend(50_000, |_ws| Duration::from_nanos(1));
        // Should return a valid config
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_recommend_trivial_closure_works() {
        let b = WorkgroupBenchmarker::new().with_warmup(1).with_measure(3);

        // Trivial closure: time an instant capture
        let config = b.recommend(10_000, |ws| {
            let start = Instant::now();
            let _total = ws.total();
            start.elapsed()
        });

        // Result must be a well-formed config
        assert!(config.validate().is_ok());
        // Profile is Custom (benchmarker chose a winner) or Mobile (fallback)
        // Either way the config must have non-zero dimensions
        assert!(config.preprocess.total() > 0);
    }

    #[test]
    fn test_recommend_empty_candidates_fallback() {
        let b = WorkgroupBenchmarker::new()
            .with_candidates(vec![])
            .with_warmup(0)
            .with_measure(1);

        // No candidates → best_of returns None → falls back to adaptive
        let config = b.recommend(5_000, |_ws| Duration::from_nanos(1));
        assert_eq!(config.profile, WorkgroupProfile::Mobile);
    }

    #[test]
    fn test_custom_candidates_respected() {
        let b = WorkgroupBenchmarker::new()
            .with_candidates(vec![128])
            .with_warmup(0)
            .with_measure(2);

        let results = b.benchmark(|_ws| Duration::from_nanos(50));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].size, WorkgroupSize::linear(128));
    }

    #[test]
    fn test_duration_to_us_conversion() {
        let d = std::time::Duration::from_micros(42);
        assert!((duration_to_us(d) - 42.0).abs() < 0.01);
    }

    #[test]
    fn test_is_power_of_two() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(64));
        assert!(is_power_of_two(256));
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(48));
        assert!(!is_power_of_two(100));
    }

    #[test]
    fn test_tile_workgroup_size() {
        let cfg = WorkgroupConfig::balanced();
        // Tile is always 4×4 regardless of the main profile
        assert_eq!(cfg.tile, WorkgroupSize::square(4));
        assert_eq!(cfg.tile.total(), 16);
    }
}
