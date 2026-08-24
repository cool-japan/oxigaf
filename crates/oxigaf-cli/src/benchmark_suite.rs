//! CPU-side benchmarking framework for Gaussian cloud operations.
//!
//! This module provides a lightweight, pure-CPU benchmarking framework for
//! measuring the performance of Gaussian cloud operations such as filtering,
//! sorting, centroid computation, and bounding box calculation.
//!
//! It is distinct from `profiling_report` (which handles training-time profiling)
//! and is specifically designed for CLI-level benchmarking of pure-CPU operations.
//!
//! # Design
//!
//! - Wall-clock time measured via [`std::time::Instant`]
//! - No external RNG dependency — uses xorshift64 for synthetic data generation
//! - No `unwrap()` or `expect()` — all fallible paths return `Result`
//! - All statistics (mean, median, std) computed from raw `Vec<f64>` timings
//!
//! # Example
//!
//! ```rust
//! use oxigaf_cli::benchmark_suite::{BenchmarkConfig, run_benchmark, format_benchmark_result};
//!
//! let config = BenchmarkConfig::default();
//! let result = run_benchmark("my_bench", || {
//!     let _: f64 = (0..1000_usize).map(|i| i as f64).sum();
//! }, &config).expect("benchmark failed");
//!
//! println!("{}", format_benchmark_result(&result));
//! ```

use std::time::Instant;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors that can occur during benchmark execution or suite management.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum BenchmarkError {
    /// A benchmark function raised an error or produced invalid timing data.
    #[error("Benchmark '{name}' failed: {reason}")]
    BenchmarkFailed { name: String, reason: String },

    /// The suite has no registered benchmarks.
    #[error("No benchmarks registered in suite")]
    EmptySuite,

    /// Warmup count is not strictly less than iteration count.
    #[error("Invalid warmup count {warmup}: must be < iterations {iters}")]
    InvalidWarmup { warmup: usize, iters: usize },

    /// Iteration count is zero, which makes statistics undefined.
    #[error("Invalid iteration count {count}: must be >= 1")]
    InvalidIterCount { count: usize },

    /// A benchmark with the given name does not exist in the suite.
    #[error("Benchmark '{name}' not found in suite")]
    BenchmarkNotFound { name: String },
}

// ---------------------------------------------------------------------------
// BenchmarkConfig
// ---------------------------------------------------------------------------

/// Configuration controlling how a benchmark is run.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Total number of timed iterations to execute.
    pub iterations: usize,
    /// Number of warmup iterations whose timing is discarded.
    pub warmup: usize,
    /// Minimum total elapsed time (ms) before stopping early (0.0 = no minimum).
    ///
    /// When positive, [`time_fn`] keeps timing additional iterations beyond
    /// `iterations` (Criterion-style) until the cumulative measured elapsed
    /// time reaches this floor, so that very cheap operations still yield
    /// statistically meaningful timing data rather than being dominated by
    /// clock-resolution noise. Bounded by an internal safety cap so a
    /// near-zero-cost closure cannot spin indefinitely trying to reach it.
    pub min_time_ms: f64,
    /// Human-readable unit name for throughput reporting (e.g. `"items"`, `"Gaussians"`).
    pub throughput_unit: String,
    /// Number of logical items processed per iteration for throughput computation.
    pub throughput_count: usize,
    /// When `Some(sigma)`, [`build_benchmark_result`] discards timings more
    /// than `sigma` standard deviations from the mean (via
    /// [`filter_outliers`]) before computing statistics. `None` (default)
    /// disables outlier filtering.
    pub outlier_sigma: Option<f64>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 100,
            warmup: 10,
            min_time_ms: 0.0,
            throughput_unit: "items".to_string(),
            throughput_count: 1,
            outlier_sigma: None,
        }
    }
}

impl BenchmarkConfig {
    /// Validate the configuration, returning an error if any field is out of range.
    pub fn validate(&self) -> Result<(), BenchmarkError> {
        if self.iterations < 1 {
            return Err(BenchmarkError::InvalidIterCount {
                count: self.iterations,
            });
        }
        if self.warmup >= self.iterations {
            return Err(BenchmarkError::InvalidWarmup {
                warmup: self.warmup,
                iters: self.iterations,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BenchmarkResult
// ---------------------------------------------------------------------------

/// Statistics from a completed benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name identifying this benchmark.
    pub name: String,
    /// Number of timed iterations that contributed to these statistics.
    pub iterations: usize,
    /// Arithmetic mean of per-iteration timings in nanoseconds.
    pub mean_ns: f64,
    /// Median of per-iteration timings in nanoseconds.
    pub median_ns: f64,
    /// Minimum per-iteration timing in nanoseconds.
    pub min_ns: f64,
    /// Maximum per-iteration timing in nanoseconds.
    pub max_ns: f64,
    /// Population standard deviation of per-iteration timings in nanoseconds.
    pub std_ns: f64,
    /// Throughput in items per second: `throughput_count / (mean_ns / 1e9)`.
    pub throughput: f64,
    /// Coefficient of variation: `std / mean` (dimensionless, 0..∞).
    pub cv: f64,
    /// Human-readable unit name for `throughput`, carried over from
    /// [`BenchmarkConfig::throughput_unit`] (e.g. `"items"`, `"Gaussians"`).
    pub throughput_unit: String,
}

// ---------------------------------------------------------------------------
// BenchmarkEntry
// ---------------------------------------------------------------------------

/// Metadata for a registered benchmark (name, description, configuration).
///
/// The actual callable is not stored here; it is passed to
/// [`run_suite_entry`] at execution time.
#[derive(Debug, Clone)]
pub struct BenchmarkEntry {
    /// Unique name for this benchmark within the suite.
    pub name: String,
    /// Human-readable description of what is being measured.
    pub description: String,
    /// Configuration used for this benchmark.
    pub config: BenchmarkConfig,
}

// ---------------------------------------------------------------------------
// BenchmarkSuite
// ---------------------------------------------------------------------------

/// A collection of benchmark entries and their accumulated results.
#[derive(Debug, Default)]
pub struct BenchmarkSuite {
    /// Registered benchmark metadata (in registration order).
    pub entries: Vec<BenchmarkEntry>,
    /// Results accumulated by calls to [`run_suite_entry`].
    pub results: Vec<BenchmarkResult>,
}

impl BenchmarkSuite {
    /// Create a new, empty suite.
    pub fn new() -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// SuiteReport
// ---------------------------------------------------------------------------

/// Aggregated report across all benchmarks in a suite.
#[derive(Debug, Clone)]
pub struct SuiteReport {
    /// Total number of benchmarks included in this report.
    pub n_benchmarks: usize,
    /// Name of the benchmark with the lowest `mean_ns`.
    pub fastest: String,
    /// Name of the benchmark with the highest `mean_ns`.
    pub slowest: String,
    /// Approximate total wall-clock time in milliseconds for all benchmark runs.
    pub total_time_ms: f64,
    /// All individual benchmark results (same order as suite results).
    pub results: Vec<BenchmarkResult>,
}

// ---------------------------------------------------------------------------
// BenchmarkComparison
// ---------------------------------------------------------------------------

/// Pairwise comparison between two benchmark results.
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    /// Name of the first benchmark.
    pub name_a: String,
    /// Name of the second benchmark.
    pub name_b: String,
    /// Ratio `a.mean_ns / b.mean_ns`. Values > 1 indicate `a` is slower.
    pub speedup: f64,
    /// Name of the faster benchmark (lower `mean_ns`).
    pub faster: String,
}

// ---------------------------------------------------------------------------
// Internal RNG — xorshift64
// ---------------------------------------------------------------------------

/// Simple xorshift64 PRNG for synthetic data generation.
///
/// The caller passes in a mutable state seed; results are deterministic given
/// the same starting seed.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Convert a raw xorshift64 output to an `f32` in `[0, 1)`.
#[inline]
fn xorshift_f32(state: &mut u64) -> f32 {
    // Use upper 23 bits for mantissa
    let raw = xorshift64(state);
    (raw >> 41) as f32 / (1u64 << 23) as f32
}

// ---------------------------------------------------------------------------
// Statistical helpers (private)
// ---------------------------------------------------------------------------

/// Arithmetic mean of a non-empty slice. Returns 0.0 for empty slices.
fn vec_mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Median of a slice. Sorts a clone in place; does not modify the original.
///
/// For even-length slices the median is the average of the two middle values.
/// Returns 0.0 for empty slices.
fn vec_median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len() % 2 == 1 {
        v[mid]
    } else {
        (v[mid - 1] + v[mid]) / 2.0
    }
}

/// Population standard deviation given a pre-computed mean.
fn vec_std(v: &[f64], mean: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let variance = v.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
    variance.sqrt()
}

// ---------------------------------------------------------------------------
// Core timing machinery
// ---------------------------------------------------------------------------

/// Safety cap on the number of *extra* iterations [`time_fn`] will run to
/// satisfy [`BenchmarkConfig::min_time_ms`], on top of `config.iterations`.
/// Bounds worst-case runtime for a near-zero-cost closure whose measured
/// per-call time can round to ~0 ns at typical clock resolution.
const MAX_EXTRA_ITERATIONS_FOR_MIN_TIME: usize = 1_000_000;

/// Run a closure `f` according to `config` and collect per-iteration timings.
///
/// Warmup calls are not timed. The returned `Vec<f64>` has exactly
/// `config.iterations` entries, each in nanoseconds — unless
/// `config.min_time_ms > 0.0` and that many timed iterations complete
/// faster than the floor, in which case additional iterations are timed
/// (up to [`MAX_EXTRA_ITERATIONS_FOR_MIN_TIME`] beyond `config.iterations`)
/// until the cumulative elapsed time reaches `min_time_ms`. The returned
/// vector's length is always the *actual* number of timed iterations.
///
/// # Errors
///
/// Returns [`BenchmarkError::InvalidWarmup`] or [`BenchmarkError::InvalidIterCount`]
/// if the config is invalid.
pub fn time_fn<F: FnMut()>(mut f: F, config: &BenchmarkConfig) -> Result<Vec<f64>, BenchmarkError> {
    config.validate()?;

    // Warmup phase — results discarded.
    for _ in 0..config.warmup {
        f();
    }

    // Timed phase.
    let mut timings = Vec::with_capacity(config.iterations);
    let mut total_elapsed_ns = 0.0f64;
    for _ in 0..config.iterations {
        let t0 = Instant::now();
        f();
        let elapsed_ns = t0.elapsed().as_nanos() as f64;
        timings.push(elapsed_ns);
        total_elapsed_ns += elapsed_ns;
    }

    if config.min_time_ms > 0.0 {
        let min_elapsed_ns = config.min_time_ms * 1_000_000.0;
        let mut extra = 0usize;
        while total_elapsed_ns < min_elapsed_ns && extra < MAX_EXTRA_ITERATIONS_FOR_MIN_TIME {
            let t0 = Instant::now();
            f();
            let elapsed_ns = t0.elapsed().as_nanos() as f64;
            timings.push(elapsed_ns);
            total_elapsed_ns += elapsed_ns;
            extra += 1;
        }
    }

    Ok(timings)
}

/// Compute a [`BenchmarkResult`] from raw per-iteration nanosecond timings.
///
/// `timings` must be non-empty; the function does not validate `config` (call
/// `validate` before reaching here). If `config.outlier_sigma` is set,
/// timings are filtered via [`filter_outliers`] before statistics are
/// computed (see that function's fallback behaviour when filtering would
/// remove every timing).
pub fn build_benchmark_result(
    name: &str,
    timings: Vec<f64>,
    config: &BenchmarkConfig,
) -> BenchmarkResult {
    let timings = match config.outlier_sigma {
        Some(sigma) => filter_outliers(timings, sigma),
        None => timings,
    };
    let mut timings_sorted = timings.clone();
    let mean_ns = vec_mean(&timings);
    let median_ns = vec_median(&mut timings_sorted);
    let min_ns = timings.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ns = timings.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let std_ns = vec_std(&timings, mean_ns);
    let throughput = config.throughput_count as f64 / (mean_ns.max(1e-10) / 1e9);
    let cv = std_ns / mean_ns.max(1e-10);

    BenchmarkResult {
        name: name.to_string(),
        iterations: timings.len(),
        mean_ns,
        median_ns,
        min_ns,
        max_ns,
        std_ns,
        throughput,
        cv,
        throughput_unit: config.throughput_unit.clone(),
    }
}

/// Run a single named benchmark and return its [`BenchmarkResult`].
///
/// This is the primary entry point for one-shot benchmark execution.
///
/// # Errors
///
/// Propagates any error from [`time_fn`].
pub fn run_benchmark<F: FnMut()>(
    name: &str,
    f: F,
    config: &BenchmarkConfig,
) -> Result<BenchmarkResult, BenchmarkError> {
    let timings = time_fn(f, config)?;
    Ok(build_benchmark_result(name, timings, config))
}

// ---------------------------------------------------------------------------
// Suite management
// ---------------------------------------------------------------------------

/// Run a named benchmark and push its result into `suite`.
///
/// If an entry with the same name already exists in `suite.entries`, no
/// duplicate entry is added. The result is always appended to `suite.results`.
///
/// # Errors
///
/// Propagates any error from [`run_benchmark`].
pub fn run_suite_entry<F: FnMut()>(
    suite: &mut BenchmarkSuite,
    name: &str,
    description: &str,
    f: F,
    config: BenchmarkConfig,
) -> Result<(), BenchmarkError> {
    // Only register the entry once.
    if !suite.entries.iter().any(|e| e.name == name) {
        suite.entries.push(BenchmarkEntry {
            name: name.to_string(),
            description: description.to_string(),
            config: config.clone(),
        });
    }

    let result = run_benchmark(name, f, &config)?;
    suite.results.push(result);
    Ok(())
}

/// Build a [`SuiteReport`] summarising all results in a suite.
///
/// # Errors
///
/// Returns [`BenchmarkError::EmptySuite`] if `suite.results` is empty.
pub fn build_suite_report(suite: &BenchmarkSuite) -> Result<SuiteReport, BenchmarkError> {
    if suite.results.is_empty() {
        return Err(BenchmarkError::EmptySuite);
    }

    let fastest = suite
        .results
        .iter()
        .min_by(|a, b| {
            a.mean_ns
                .partial_cmp(&b.mean_ns)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.name.clone())
        .unwrap_or_default();

    let slowest = suite
        .results
        .iter()
        .max_by(|a, b| {
            a.mean_ns
                .partial_cmp(&b.mean_ns)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.name.clone())
        .unwrap_or_default();

    // Approximate total: sum of mean_ns * iterations, converted to ms.
    let total_time_ms = suite
        .results
        .iter()
        .map(|r| r.mean_ns * r.iterations as f64 / 1_000_000.0)
        .sum();

    Ok(SuiteReport {
        n_benchmarks: suite.results.len(),
        fastest,
        slowest,
        total_time_ms,
        results: suite.results.clone(),
    })
}

// ---------------------------------------------------------------------------
// Synthetic benchmark targets
// ---------------------------------------------------------------------------

/// Sum all elements in a `Vec<f32>` of length `size`.
///
/// Fills the vector with values derived from the element index (no RNG needed
/// for a simple sum benchmark). Returns the sum to prevent dead-code elimination.
pub fn bench_vec_sum(size: usize) -> f64 {
    let v: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
    v.iter().map(|&x| x as f64).sum()
}

/// Dot product of two `Vec<f32>` vectors of length `size`.
///
/// Uses a fixed xorshift64 seed to generate both vectors. Returns the dot
/// product as `f64` to prevent dead-code elimination.
pub fn bench_vec_dot(size: usize) -> f64 {
    let mut state: u64 = 0xDEAD_BEEF_CAFE_1234;
    let a: Vec<f32> = (0..size).map(|_| xorshift_f32(&mut state)).collect();
    let b: Vec<f32> = (0..size).map(|_| xorshift_f32(&mut state)).collect();
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum()
}

/// Sort a `Vec<f32>` of length `size` filled with xorshift64-derived values.
///
/// Returns the first element after sorting (which is the minimum), preventing
/// dead-code elimination.
pub fn bench_sort_f32(size: usize) -> f32 {
    let mut state: u64 = 0xFEED_FACE_DEAD_BEEF;
    let mut v: Vec<f32> = (0..size).map(|_| xorshift_f32(&mut state)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v.first().copied().unwrap_or(0.0)
}

/// Compute the centroid (mean x, y, z) of `n_gaussians` points.
///
/// Positions are filled with xorshift64-derived values in `[0, 1)`.
/// Returns `[mean_x, mean_y, mean_z]`.
pub fn bench_gaussian_centroid(n_gaussians: usize) -> [f32; 3] {
    if n_gaussians == 0 {
        return [0.0; 3];
    }
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    let mut sum = [0.0f64; 3];
    for _ in 0..n_gaussians {
        sum[0] += xorshift_f32(&mut state) as f64;
        sum[1] += xorshift_f32(&mut state) as f64;
        sum[2] += xorshift_f32(&mut state) as f64;
    }
    let n = n_gaussians as f64;
    [
        (sum[0] / n) as f32,
        (sum[1] / n) as f32,
        (sum[2] / n) as f32,
    ]
}

/// Compute the axis-aligned bounding box (AABB) of `n_gaussians` points.
///
/// Returns `(min_xyz, max_xyz)` where each component is the minimum/maximum
/// coordinate across all points.
pub fn bench_gaussian_bbox(n_gaussians: usize) -> ([f32; 3], [f32; 3]) {
    if n_gaussians == 0 {
        return ([0.0; 3], [0.0; 3]);
    }
    let mut state: u64 = 0xABCD_EF01_2345_6789;
    let mut min_xyz = [f32::INFINITY; 3];
    let mut max_xyz = [f32::NEG_INFINITY; 3];
    for _ in 0..n_gaussians {
        for axis in 0..3 {
            let v = xorshift_f32(&mut state);
            if v < min_xyz[axis] {
                min_xyz[axis] = v;
            }
            if v > max_xyz[axis] {
                max_xyz[axis] = v;
            }
        }
    }
    (min_xyz, max_xyz)
}

// ---------------------------------------------------------------------------
// Default suite factory
// ---------------------------------------------------------------------------

/// Create a [`BenchmarkSuite`] pre-populated with representative entry metadata.
///
/// The entries represent common Gaussian cloud operations at small (1 k) and
/// large (1 M) scales. Note that entries only hold metadata; to actually run
/// and record results, call [`run_suite_entry`] for each entry.
pub fn create_default_suite() -> BenchmarkSuite {
    let mut suite = BenchmarkSuite::new();

    let small_config = BenchmarkConfig {
        iterations: 50,
        warmup: 5,
        throughput_unit: "items".to_string(),
        throughput_count: 1_000,
        ..Default::default()
    };
    let large_config = BenchmarkConfig {
        iterations: 20,
        warmup: 2,
        throughput_unit: "items".to_string(),
        throughput_count: 1_000_000,
        ..Default::default()
    };
    let dot_config = BenchmarkConfig {
        iterations: 50,
        warmup: 5,
        throughput_unit: "items".to_string(),
        throughput_count: 1_000,
        ..Default::default()
    };
    let sort_config = BenchmarkConfig {
        iterations: 50,
        warmup: 5,
        throughput_unit: "items".to_string(),
        throughput_count: 1_000,
        ..Default::default()
    };
    let centroid_config = BenchmarkConfig {
        iterations: 50,
        warmup: 5,
        throughput_unit: "Gaussians".to_string(),
        throughput_count: 1_000,
        ..Default::default()
    };
    let bbox_config = BenchmarkConfig {
        iterations: 50,
        warmup: 5,
        throughput_unit: "Gaussians".to_string(),
        throughput_count: 1_000,
        ..Default::default()
    };

    suite.entries.push(BenchmarkEntry {
        name: "vec_sum_1k".to_string(),
        description: "Sum 1 000 f32 elements".to_string(),
        config: small_config.clone(),
    });
    suite.entries.push(BenchmarkEntry {
        name: "vec_sum_1m".to_string(),
        description: "Sum 1 000 000 f32 elements".to_string(),
        config: large_config,
    });
    suite.entries.push(BenchmarkEntry {
        name: "vec_dot_1k".to_string(),
        description: "Dot product of two 1 000-element f32 vectors".to_string(),
        config: dot_config,
    });
    suite.entries.push(BenchmarkEntry {
        name: "sort_1k".to_string(),
        description: "Sort 1 000 f32 values".to_string(),
        config: sort_config,
    });
    suite.entries.push(BenchmarkEntry {
        name: "gauss_centroid_1k".to_string(),
        description: "Centroid of 1 000 Gaussian positions".to_string(),
        config: centroid_config,
    });
    suite.entries.push(BenchmarkEntry {
        name: "gauss_bbox_1k".to_string(),
        description: "AABB of 1 000 Gaussian positions".to_string(),
        config: bbox_config,
    });

    suite
}

/// Build the default suite (via [`create_default_suite`]) and actually run
/// every entry, returning a suite whose `results` are populated.
///
/// [`BenchmarkEntry`] deliberately does not store a callable — the actual
/// function is passed to [`run_suite_entry`] at execution time — so
/// `create_default_suite()`'s entries alone cannot be mechanically executed;
/// this function resolves that name-to-closure mapping for the six built-in
/// entries, each backed by one of the `bench_*` functions in this module at
/// the problem size implied by its name (`_1k` = 1 000, `_1m` = 1 000 000).
///
/// # Errors
///
/// Returns [`BenchmarkError::BenchmarkNotFound`] if `create_default_suite`
/// is ever extended with an entry name this function does not know how to
/// run (keeping that error variant genuinely reachable rather than dead).
/// Propagates any error from [`run_suite_entry`].
pub fn run_default_suite() -> Result<BenchmarkSuite, BenchmarkError> {
    let mut suite = BenchmarkSuite::new();
    let planned = create_default_suite();

    for entry in planned.entries {
        let config = entry.config.clone();
        match entry.name.as_str() {
            "vec_sum_1k" => run_suite_entry(
                &mut suite,
                &entry.name,
                &entry.description,
                || {
                    std::hint::black_box(bench_vec_sum(1_000));
                },
                config,
            )?,
            "vec_sum_1m" => run_suite_entry(
                &mut suite,
                &entry.name,
                &entry.description,
                || {
                    std::hint::black_box(bench_vec_sum(1_000_000));
                },
                config,
            )?,
            "vec_dot_1k" => run_suite_entry(
                &mut suite,
                &entry.name,
                &entry.description,
                || {
                    std::hint::black_box(bench_vec_dot(1_000));
                },
                config,
            )?,
            "sort_1k" => run_suite_entry(
                &mut suite,
                &entry.name,
                &entry.description,
                || {
                    std::hint::black_box(bench_sort_f32(1_000));
                },
                config,
            )?,
            "gauss_centroid_1k" => run_suite_entry(
                &mut suite,
                &entry.name,
                &entry.description,
                || {
                    std::hint::black_box(bench_gaussian_centroid(1_000));
                },
                config,
            )?,
            "gauss_bbox_1k" => run_suite_entry(
                &mut suite,
                &entry.name,
                &entry.description,
                || {
                    std::hint::black_box(bench_gaussian_bbox(1_000));
                },
                config,
            )?,
            unknown => {
                return Err(BenchmarkError::BenchmarkNotFound {
                    name: unknown.to_string(),
                });
            }
        }
    }

    Ok(suite)
}

// ---------------------------------------------------------------------------
// Outlier filtering
// ---------------------------------------------------------------------------

/// Remove timings that deviate more than `sigma` standard deviations from the mean.
///
/// If filtering would leave zero elements, the original `timings` vector is
/// returned unchanged so that callers always have at least one data point.
pub fn filter_outliers(timings: Vec<f64>, sigma: f64) -> Vec<f64> {
    if timings.is_empty() {
        return timings;
    }
    let mean = vec_mean(&timings);
    let std = vec_std(&timings, mean);
    let threshold = sigma * std;
    let filtered: Vec<f64> = timings
        .iter()
        .copied()
        .filter(|&x| (x - mean).abs() <= threshold)
        .collect();
    if filtered.is_empty() {
        timings
    } else {
        filtered
    }
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/// Format a nanosecond duration into a human-readable string.
///
/// | Range | Example output |
/// |-------|---------------|
/// | < 1 000 ns | `"500.0 ns"` |
/// | < 1 000 000 ns | `"12.34 µs"` |
/// | < 1 000 000 000 ns | `"5.67 ms"` |
/// | ≥ 1 000 000 000 ns | `"1.23 s"` |
pub fn format_duration_ns(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{:.1} ns", ns)
    } else if ns < 1_000_000.0 {
        format!("{:.2} µs", ns / 1_000.0)
    } else if ns < 1_000_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else {
        format!("{:.2} s", ns / 1_000_000_000.0)
    }
}

/// Format a [`BenchmarkResult`] as a single-line summary string.
///
/// Example output:
/// ```text
/// my_bench: mean=1.23 µs, median=1.21 µs, min=1.10 µs, max=1.50 µs, throughput=8.1M items/s, cv=0.05
/// ```
pub fn format_benchmark_result(result: &BenchmarkResult) -> String {
    let throughput_str = if result.throughput >= 1_000_000.0 {
        format!("{:.1}M", result.throughput / 1_000_000.0)
    } else if result.throughput >= 1_000.0 {
        format!("{:.1}k", result.throughput / 1_000.0)
    } else {
        format!("{:.1}", result.throughput)
    };

    format!(
        "{}: mean={}, median={}, min={}, max={}, throughput={} {}/s, cv={:.4}",
        result.name,
        format_duration_ns(result.mean_ns),
        format_duration_ns(result.median_ns),
        format_duration_ns(result.min_ns),
        format_duration_ns(result.max_ns),
        throughput_str,
        result.throughput_unit,
        result.cv,
    )
}

/// Format a [`SuiteReport`] as a multi-line human-readable string.
pub fn format_suite_report(report: &SuiteReport) -> String {
    let mut out = String::new();

    out.push_str("=== Benchmark Suite Report ===\n");
    out.push_str(&format!("Benchmarks : {}\n", report.n_benchmarks));
    out.push_str(&format!("Total time : {:.2} ms\n", report.total_time_ms));
    out.push_str(&format!("Fastest    : {}\n", report.fastest));
    out.push_str(&format!("Slowest    : {}\n", report.slowest));
    out.push_str("---\n");

    for result in &report.results {
        out.push_str(&format_benchmark_result(result));
        out.push('\n');
    }

    out.push_str("==============================\n");
    out
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Compare two benchmark results pairwise.
///
/// `speedup` = `a.mean_ns / b.mean_ns`. A value > 1 indicates `a` is slower
/// than `b`. The `faster` field names the benchmark with lower `mean_ns`.
pub fn compare_benchmarks(a: &BenchmarkResult, b: &BenchmarkResult) -> BenchmarkComparison {
    let speedup = a.mean_ns / b.mean_ns.max(1e-10);
    let faster = if a.mean_ns <= b.mean_ns {
        a.name.clone()
    } else {
        b.name.clone()
    };

    BenchmarkComparison {
        name_a: a.name.clone(),
        name_b: b.name.clone(),
        speedup,
        faster,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // BenchmarkConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_defaults() {
        let cfg = BenchmarkConfig::default();
        assert_eq!(cfg.iterations, 100);
        assert_eq!(cfg.warmup, 10);
        assert_eq!(cfg.min_time_ms, 0.0);
        assert_eq!(cfg.throughput_unit, "items");
        assert_eq!(cfg.throughput_count, 1);
    }

    #[test]
    fn test_config_validate_ok() {
        let cfg = BenchmarkConfig {
            iterations: 20,
            warmup: 5,
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_warmup_eq_iters() {
        let cfg = BenchmarkConfig {
            iterations: 10,
            warmup: 10,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(BenchmarkError::InvalidWarmup {
                warmup: 10,
                iters: 10
            })
        ));
    }

    #[test]
    fn test_config_validate_warmup_gt_iters() {
        let cfg = BenchmarkConfig {
            iterations: 5,
            warmup: 10,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(BenchmarkError::InvalidWarmup { .. })
        ));
    }

    #[test]
    fn test_config_validate_zero_iterations() {
        let cfg = BenchmarkConfig {
            iterations: 0,
            warmup: 0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(BenchmarkError::InvalidIterCount { count: 0 })
        ));
    }

    #[test]
    fn test_config_validate_one_iter_no_warmup() {
        let cfg = BenchmarkConfig {
            iterations: 1,
            warmup: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    // -----------------------------------------------------------------------
    // Statistical helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_vec_mean_known() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((vec_mean(&v) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_vec_mean_empty() {
        let v: Vec<f64> = vec![];
        assert_eq!(vec_mean(&v), 0.0);
    }

    #[test]
    fn test_vec_mean_single() {
        let v = vec![42.0];
        assert!((vec_mean(&v) - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_vec_median_odd_length() {
        let mut v = vec![3.0, 1.0, 2.0];
        assert!((vec_median(&mut v) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_vec_median_even_length() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        assert!((vec_median(&mut v) - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_vec_median_single() {
        let mut v = vec![7.0];
        assert!((vec_median(&mut v) - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_vec_median_empty() {
        let mut v: Vec<f64> = vec![];
        assert_eq!(vec_median(&mut v), 0.0);
    }

    #[test]
    fn test_vec_std_constant() {
        let v = vec![5.0, 5.0, 5.0, 5.0];
        let mean = vec_mean(&v);
        assert!(vec_std(&v, mean) < 1e-9);
    }

    #[test]
    fn test_vec_std_known_variance() {
        // Values: 2, 4, 4, 4, 5, 5, 7, 9 — population std = 2.0
        let v = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mean = vec_mean(&v);
        let std = vec_std(&v, mean);
        assert!((std - 2.0).abs() < 1e-9, "expected 2.0, got {}", std);
    }

    #[test]
    fn test_vec_std_empty() {
        let v: Vec<f64> = vec![];
        assert_eq!(vec_std(&v, 0.0), 0.0);
    }

    // -----------------------------------------------------------------------
    // time_fn
    // -----------------------------------------------------------------------

    #[test]
    fn test_time_fn_returns_correct_count() {
        let cfg = BenchmarkConfig {
            iterations: 10,
            warmup: 2,
            ..Default::default()
        };
        let mut counter = 0usize;
        let timings = time_fn(|| counter += 1, &cfg).expect("time_fn failed");
        assert_eq!(timings.len(), 10);
        // warmup (2) + timed (10) = 12 total calls
        assert_eq!(counter, 12);
    }

    #[test]
    fn test_time_fn_all_values_positive() {
        let cfg = BenchmarkConfig {
            iterations: 5,
            warmup: 1,
            ..Default::default()
        };
        let timings = time_fn(
            || {
                // Minimal work so Instant doesn't collapse to 0 on fast machines.
                std::hint::black_box(vec![0u8; 64]);
            },
            &cfg,
        )
        .expect("time_fn failed");
        assert!(timings.iter().all(|&t| t >= 0.0));
    }

    #[test]
    fn test_time_fn_invalid_config_propagated() {
        let cfg = BenchmarkConfig {
            iterations: 0,
            warmup: 0,
            ..Default::default()
        };
        assert!(time_fn(|| {}, &cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // build_benchmark_result
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_result_mean() {
        let cfg = BenchmarkConfig::default();
        let timings = vec![100.0, 200.0, 300.0];
        let result = build_benchmark_result("test", timings, &cfg);
        assert!((result.mean_ns - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_result_median_odd() {
        let cfg = BenchmarkConfig::default();
        let timings = vec![300.0, 100.0, 200.0];
        let result = build_benchmark_result("test", timings, &cfg);
        assert!((result.median_ns - 200.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_result_min_max() {
        let cfg = BenchmarkConfig::default();
        let timings = vec![500.0, 100.0, 300.0, 200.0, 400.0];
        let result = build_benchmark_result("test", timings, &cfg);
        assert!((result.min_ns - 100.0).abs() < 1e-6);
        assert!((result.max_ns - 500.0).abs() < 1e-6);
    }

    #[test]
    fn test_build_result_throughput_formula() {
        let cfg = BenchmarkConfig {
            throughput_count: 1000,
            ..Default::default()
        };
        // mean_ns = 1_000_000 ns = 1 ms → throughput = 1000 / 0.001 = 1_000_000 items/s
        let timings = vec![1_000_000.0f64; 5];
        let result = build_benchmark_result("test", timings, &cfg);
        assert!(
            (result.throughput - 1_000_000.0).abs() < 1.0,
            "got {}",
            result.throughput
        );
    }

    #[test]
    fn test_build_result_cv() {
        let cfg = BenchmarkConfig::default();
        // All identical timings → std = 0 → cv = 0.
        let timings = vec![1000.0; 10];
        let result = build_benchmark_result("test", timings, &cfg);
        assert!(result.cv < 1e-9);
    }

    #[test]
    fn test_build_result_iterations_count() {
        let cfg = BenchmarkConfig::default();
        let timings: Vec<f64> = (0..17).map(|i| (i + 1) as f64 * 100.0).collect();
        let result = build_benchmark_result("test", timings, &cfg);
        assert_eq!(result.iterations, 17);
    }

    // -----------------------------------------------------------------------
    // run_benchmark
    // -----------------------------------------------------------------------

    #[test]
    fn test_run_benchmark_smoke() {
        let cfg = BenchmarkConfig {
            iterations: 5,
            warmup: 1,
            throughput_count: 100,
            ..Default::default()
        };
        let result = run_benchmark(
            "smoke",
            || {
                std::hint::black_box(42u64 * 7);
            },
            &cfg,
        );
        assert!(result.is_ok());
        let r = result.expect("should succeed");
        assert_eq!(r.name, "smoke");
        assert_eq!(r.iterations, 5);
    }

    #[test]
    fn test_run_benchmark_invalid_config_errors() {
        let cfg = BenchmarkConfig {
            iterations: 3,
            warmup: 5,
            ..Default::default()
        };
        assert!(run_benchmark("bad", || {}, &cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // Synthetic benchmark targets
    // -----------------------------------------------------------------------

    #[test]
    fn test_bench_vec_sum_zero_size() {
        let result = bench_vec_sum(0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_bench_vec_sum_known_value() {
        // Elements: 0*0.001, 1*0.001, 2*0.001 = 0.003 sum
        let result = bench_vec_sum(3);
        // sum = 0.0 + 0.001 + 0.002 = 0.003 (allow f32 rounding)
        assert!((result - 0.003).abs() < 1e-5, "got {}", result);
    }

    #[test]
    fn test_bench_vec_sum_positive() {
        // With any non-zero size the sum should be non-negative (all values ≥ 0).
        assert!(bench_vec_sum(1000) >= 0.0);
    }

    #[test]
    fn test_bench_vec_dot_zero_size() {
        let result = bench_vec_dot(0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_bench_vec_dot_positive_range() {
        // Both vectors have values in [0,1), so dot product must be ≥ 0.
        let result = bench_vec_dot(500);
        assert!(result >= 0.0);
    }

    #[test]
    fn test_bench_vec_dot_bounded() {
        // Max possible: size * 1.0 * 1.0 = size.
        let size = 200;
        let result = bench_vec_dot(size);
        assert!(result <= size as f64, "dot product exceeded upper bound");
    }

    #[test]
    fn test_bench_sort_f32_is_minimum() {
        // After sorting, first element must be the minimum.
        // xorshift values are in [0,1), so first element ≤ all others.
        let first = bench_sort_f32(100);
        // Must be in [0, 1) — not NaN.
        assert!((0.0..1.0).contains(&first));
    }

    #[test]
    fn test_bench_sort_f32_zero_size() {
        // Edge case: size=0 should return 0.0 without panic.
        let result = bench_sort_f32(0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_bench_sort_f32_deterministic() {
        // Same seed → same result every call.
        let a = bench_sort_f32(50);
        let b = bench_sort_f32(50);
        assert_eq!(a, b);
    }

    #[test]
    fn test_bench_gaussian_centroid_zero() {
        let c = bench_gaussian_centroid(0);
        assert_eq!(c, [0.0f32; 3]);
    }

    #[test]
    fn test_bench_gaussian_centroid_bounded() {
        // All positions in [0, 1) → centroid in [0, 1).
        let c = bench_gaussian_centroid(1000);
        for &v in &c {
            assert!((0.0..1.0).contains(&v), "centroid out of range: {}", v);
        }
    }

    #[test]
    fn test_bench_gaussian_centroid_near_half() {
        // With many points uniformly distributed, centroid should be near 0.5.
        let c = bench_gaussian_centroid(100_000);
        for &v in &c {
            assert!((v - 0.5).abs() < 0.05, "centroid far from 0.5: {}", v);
        }
    }

    #[test]
    fn test_bench_gaussian_bbox_zero() {
        let (mn, mx) = bench_gaussian_bbox(0);
        assert_eq!(mn, [0.0f32; 3]);
        assert_eq!(mx, [0.0f32; 3]);
    }

    #[test]
    fn test_bench_gaussian_bbox_min_le_max() {
        let (mn, mx) = bench_gaussian_bbox(500);
        for i in 0..3 {
            assert!(mn[i] <= mx[i], "min[{}] > max[{}]", i, i);
        }
    }

    #[test]
    fn test_bench_gaussian_bbox_in_unit_cube() {
        let (mn, mx) = bench_gaussian_bbox(500);
        for i in 0..3 {
            assert!(mn[i] >= 0.0);
            assert!(mx[i] < 1.0);
        }
    }

    // -----------------------------------------------------------------------
    // Suite management
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_default_suite_entry_count() {
        let suite = create_default_suite();
        assert_eq!(suite.entries.len(), 6);
    }

    #[test]
    fn test_create_default_suite_entry_names() {
        let suite = create_default_suite();
        let names: Vec<&str> = suite.entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"vec_sum_1k"));
        assert!(names.contains(&"vec_sum_1m"));
        assert!(names.contains(&"vec_dot_1k"));
        assert!(names.contains(&"sort_1k"));
        assert!(names.contains(&"gauss_centroid_1k"));
        assert!(names.contains(&"gauss_bbox_1k"));
    }

    #[test]
    fn test_create_default_suite_no_results() {
        // A freshly created suite has no results.
        let suite = create_default_suite();
        assert!(suite.results.is_empty());
    }

    #[test]
    fn test_run_suite_entry_pushes_result() {
        let mut suite = BenchmarkSuite::new();
        let cfg = BenchmarkConfig {
            iterations: 5,
            warmup: 1,
            ..Default::default()
        };
        run_suite_entry(
            &mut suite,
            "t1",
            "desc",
            || {
                std::hint::black_box(1u32 + 1);
            },
            cfg,
        )
        .expect("run_suite_entry failed");
        assert_eq!(suite.results.len(), 1);
        assert_eq!(suite.results[0].name, "t1");
    }

    #[test]
    fn test_run_suite_entry_no_duplicate_entries() {
        let mut suite = BenchmarkSuite::new();
        let cfg = BenchmarkConfig {
            iterations: 3,
            warmup: 1,
            ..Default::default()
        };
        run_suite_entry(&mut suite, "dup", "d1", || {}, cfg.clone()).expect("first run failed");
        run_suite_entry(&mut suite, "dup", "d2", || {}, cfg).expect("second run failed");
        // Two results but only one entry.
        assert_eq!(suite.entries.len(), 1);
        assert_eq!(suite.results.len(), 2);
    }

    // -----------------------------------------------------------------------
    // build_suite_report
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_suite_report_empty_is_err() {
        let suite = BenchmarkSuite::new();
        assert!(matches!(
            build_suite_report(&suite),
            Err(BenchmarkError::EmptySuite)
        ));
    }

    #[test]
    fn test_build_suite_report_fastest_slowest() {
        let mut suite = BenchmarkSuite::new();
        let cfg = BenchmarkConfig {
            iterations: 5,
            warmup: 1,
            ..Default::default()
        };
        // bench_vec_sum(1) should be fast; bench_vec_sum(100_000) slower.
        run_suite_entry(
            &mut suite,
            "fast_bench",
            "fast",
            || {
                std::hint::black_box(bench_vec_sum(1));
            },
            cfg.clone(),
        )
        .expect("fast bench failed");
        run_suite_entry(
            &mut suite,
            "slow_bench",
            "slow",
            || {
                std::hint::black_box(bench_vec_sum(100_000));
            },
            cfg,
        )
        .expect("slow bench failed");

        let report = build_suite_report(&suite).expect("report failed");
        assert_eq!(report.n_benchmarks, 2);
        // fastest and slowest must be valid names.
        assert!(report.fastest == "fast_bench" || report.fastest == "slow_bench");
        assert!(report.slowest == "fast_bench" || report.slowest == "slow_bench");
        assert_ne!(report.fastest, report.slowest);
    }

    #[test]
    fn test_build_suite_report_total_time_positive() {
        let mut suite = BenchmarkSuite::new();
        let cfg = BenchmarkConfig {
            iterations: 5,
            warmup: 1,
            ..Default::default()
        };
        run_suite_entry(
            &mut suite,
            "x",
            "x",
            || {
                std::hint::black_box(1u64);
            },
            cfg,
        )
        .expect("failed");
        let report = build_suite_report(&suite).expect("report failed");
        assert!(report.total_time_ms >= 0.0);
    }

    // -----------------------------------------------------------------------
    // format_duration_ns
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_duration_ns_ns_range() {
        let s = format_duration_ns(500.0);
        assert!(s.ends_with("ns"), "got: {}", s);
        assert!(s.contains("500"), "got: {}", s);
    }

    #[test]
    fn test_format_duration_ns_boundary_ns() {
        // Exactly 999.9 ns → still ns range.
        let s = format_duration_ns(999.9);
        assert!(s.ends_with("ns"), "got: {}", s);
    }

    #[test]
    fn test_format_duration_ns_us_range() {
        let s = format_duration_ns(5_000.0);
        assert!(s.contains("µs"), "got: {}", s);
        assert!(s.contains("5.00"), "got: {}", s);
    }

    #[test]
    fn test_format_duration_ns_ms_range() {
        let s = format_duration_ns(5_000_000.0);
        assert!(s.contains("ms"), "got: {}", s);
        assert!(s.contains("5.00"), "got: {}", s);
    }

    #[test]
    fn test_format_duration_ns_s_range() {
        let s = format_duration_ns(2_000_000_000.0);
        assert!(s.ends_with(" s"), "got: {}", s);
        assert!(s.contains("2.00"), "got: {}", s);
    }

    #[test]
    fn test_format_duration_ns_zero() {
        let s = format_duration_ns(0.0);
        assert!(s.ends_with("ns"), "got: {}", s);
    }

    // -----------------------------------------------------------------------
    // format_benchmark_result
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_benchmark_result_nonempty() {
        let r = BenchmarkResult {
            name: "my_bench".to_string(),
            iterations: 100,
            mean_ns: 1_234.0,
            median_ns: 1_210.0,
            min_ns: 1_100.0,
            max_ns: 1_500.0,
            std_ns: 50.0,
            throughput: 810_000.0,
            throughput_unit: "items".to_string(),
            cv:0.05,
        };
        let s = format_benchmark_result(&r);
        assert!(!s.is_empty());
        assert!(s.contains("my_bench"), "got: {}", s);
    }

    #[test]
    fn test_format_benchmark_result_contains_mean() {
        let r = BenchmarkResult {
            name: "x".to_string(),
            iterations: 10,
            mean_ns: 1_000_000.0,
            median_ns: 1_000_000.0,
            min_ns: 900_000.0,
            max_ns: 1_100_000.0,
            std_ns: 50_000.0,
            throughput: 1000.0,
            throughput_unit: "items".to_string(),
            cv:0.05,
        };
        let s = format_benchmark_result(&r);
        assert!(s.contains("mean="), "got: {}", s);
        assert!(s.contains("ms"), "got: {}", s);
    }

    #[test]
    fn test_format_benchmark_result_contains_throughput() {
        let r = BenchmarkResult {
            name: "t".to_string(),
            iterations: 10,
            mean_ns: 1_000.0,
            median_ns: 1_000.0,
            min_ns: 900.0,
            max_ns: 1_100.0,
            std_ns: 50.0,
            throughput: 5_000_000.0,
            throughput_unit: "items".to_string(),
            cv:0.02,
        };
        let s = format_benchmark_result(&r);
        assert!(s.contains("throughput="), "got: {}", s);
        assert!(s.contains("M"), "got: {}", s);
    }

    // -----------------------------------------------------------------------
    // format_suite_report
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_suite_report_nonempty() {
        let r = BenchmarkResult {
            name: "a".to_string(),
            iterations: 5,
            mean_ns: 1000.0,
            median_ns: 1000.0,
            min_ns: 900.0,
            max_ns: 1100.0,
            std_ns: 50.0,
            throughput: 1_000_000.0,
            throughput_unit: "items".to_string(),
            cv:0.05,
        };
        let report = SuiteReport {
            n_benchmarks: 1,
            fastest: "a".to_string(),
            slowest: "a".to_string(),
            total_time_ms: 0.005,
            results: vec![r],
        };
        let s = format_suite_report(&report);
        assert!(!s.is_empty());
        assert!(s.contains("Benchmark Suite"), "got: {}", s);
        assert!(s.contains('a'), "got: {}", s);
    }

    #[test]
    fn test_format_suite_report_contains_fastest_slowest() {
        let r1 = BenchmarkResult {
            name: "slow_one".to_string(),
            iterations: 5,
            mean_ns: 10_000.0,
            median_ns: 10_000.0,
            min_ns: 9_000.0,
            max_ns: 11_000.0,
            std_ns: 500.0,
            throughput: 100_000.0,
            throughput_unit: "items".to_string(),
            cv:0.05,
        };
        let r2 = BenchmarkResult {
            name: "fast_one".to_string(),
            iterations: 5,
            mean_ns: 100.0,
            median_ns: 100.0,
            min_ns: 90.0,
            max_ns: 110.0,
            std_ns: 5.0,
            throughput: 10_000_000.0,
            throughput_unit: "items".to_string(),
            cv:0.05,
        };
        let report = SuiteReport {
            n_benchmarks: 2,
            fastest: "fast_one".to_string(),
            slowest: "slow_one".to_string(),
            total_time_ms: 1.0,
            results: vec![r1, r2],
        };
        let s = format_suite_report(&report);
        assert!(s.contains("fast_one"), "got: {}", s);
        assert!(s.contains("slow_one"), "got: {}", s);
    }

    // -----------------------------------------------------------------------
    // compare_benchmarks
    // -----------------------------------------------------------------------

    #[test]
    fn test_compare_benchmarks_speedup_direction() {
        let fast = BenchmarkResult {
            name: "fast".to_string(),
            iterations: 10,
            mean_ns: 100.0,
            median_ns: 100.0,
            min_ns: 90.0,
            max_ns: 110.0,
            std_ns: 5.0,
            throughput: 1e7,
            throughput_unit: "items".to_string(),
            cv:0.05,
        };
        let slow = BenchmarkResult {
            name: "slow".to_string(),
            iterations: 10,
            mean_ns: 1000.0,
            median_ns: 1000.0,
            min_ns: 900.0,
            max_ns: 1100.0,
            std_ns: 50.0,
            throughput: 1e6,
            throughput_unit: "items".to_string(),
            cv:0.05,
        };
        // a=slow, b=fast → speedup = 1000/100 = 10 → a is 10× slower
        let cmp = compare_benchmarks(&slow, &fast);
        assert!((cmp.speedup - 10.0).abs() < 0.01, "got {}", cmp.speedup);
        assert_eq!(cmp.faster, "fast");
    }

    #[test]
    fn test_compare_benchmarks_equal_timings() {
        let r = BenchmarkResult {
            name: "eq".to_string(),
            iterations: 5,
            mean_ns: 500.0,
            median_ns: 500.0,
            min_ns: 490.0,
            max_ns: 510.0,
            std_ns: 5.0,
            throughput: 2e6,
            throughput_unit: "items".to_string(),
            cv:0.01,
        };
        let cmp = compare_benchmarks(&r, &r);
        assert!((cmp.speedup - 1.0).abs() < 0.01);
        // When equal, a is not slower so a is "faster".
        assert_eq!(cmp.faster, "eq");
    }

    #[test]
    fn test_compare_benchmarks_names_preserved() {
        let a = BenchmarkResult {
            name: "alpha".to_string(),
            iterations: 5,
            mean_ns: 200.0,
            median_ns: 200.0,
            min_ns: 180.0,
            max_ns: 220.0,
            std_ns: 10.0,
            throughput: 5e6,
            throughput_unit: "items".to_string(),
            cv:0.05,
        };
        let b = BenchmarkResult {
            name: "beta".to_string(),
            iterations: 5,
            mean_ns: 400.0,
            median_ns: 400.0,
            min_ns: 380.0,
            max_ns: 420.0,
            std_ns: 10.0,
            throughput: 2.5e6,
            throughput_unit: "items".to_string(),
            cv:0.025,
        };
        let cmp = compare_benchmarks(&a, &b);
        assert_eq!(cmp.name_a, "alpha");
        assert_eq!(cmp.name_b, "beta");
        assert_eq!(cmp.faster, "alpha");
    }

    // -----------------------------------------------------------------------
    // filter_outliers
    // -----------------------------------------------------------------------

    #[test]
    fn test_filter_outliers_removes_obvious_outlier() {
        // Values mostly around 100, with one at 10_000.
        let timings: Vec<f64> = {
            let mut v: Vec<f64> = (0..20).map(|_| 100.0).collect();
            v.push(10_000.0);
            v
        };
        let filtered = filter_outliers(timings, 2.0);
        // The 10_000 outlier should be removed.
        assert!(
            !filtered.contains(&10_000.0),
            "outlier should have been removed"
        );
        assert!(!filtered.is_empty());
    }

    #[test]
    fn test_filter_outliers_keeps_all_when_tight() {
        let timings: Vec<f64> = (0..10).map(|_| 500.0).collect();
        // All identical → std=0 → threshold=0 → all kept (within 0 of mean = 500).
        let filtered = filter_outliers(timings.clone(), 2.0);
        assert_eq!(filtered.len(), timings.len());
    }

    #[test]
    fn test_filter_outliers_empty_input() {
        let filtered = filter_outliers(vec![], 3.0);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_outliers_fallback_when_all_filtered() {
        // Contrived: sigma=0.0 would remove everything since all deviate 0 > 0*std.
        // Actually with sigma=0 and std=0, threshold=0, (x - mean).abs() = 0 ≤ 0 → kept.
        // Use a different approach: sigma extremely small with non-zero std.
        let timings = vec![1.0, 2.0, 3.0, 100.0];
        // All values should be filtered with sigma = 0.0 (threshold = 0 * std > 0 for any non-constant series).
        // Actually (x-mean).abs() <= 0 means only elements equal to mean survive.
        // mean = 26.5; no element equals 26.5 exactly → all filtered → fallback.
        let filtered = filter_outliers(timings.clone(), 0.0);
        // Fallback: must return original.
        assert!(!filtered.is_empty());
    }

    #[test]
    fn test_filter_outliers_single_element() {
        let filtered = filter_outliers(vec![42.0], 3.0);
        assert_eq!(filtered, vec![42.0]);
    }

    // -----------------------------------------------------------------------
    // BenchmarkError variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_benchmark_failed_display() {
        let e = BenchmarkError::BenchmarkFailed {
            name: "foo".to_string(),
            reason: "timeout".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("foo"));
        assert!(s.contains("timeout"));
    }

    #[test]
    fn test_error_empty_suite_display() {
        let e = BenchmarkError::EmptySuite;
        let s = e.to_string();
        assert!(!s.is_empty());
    }

    #[test]
    fn test_error_not_found_display() {
        let e = BenchmarkError::BenchmarkNotFound {
            name: "missing_bench".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("missing_bench"));
    }

    // -----------------------------------------------------------------------
    // BenchmarkResult field access
    // -----------------------------------------------------------------------

    #[test]
    fn test_result_fields_accessible() {
        let cfg = BenchmarkConfig {
            iterations: 3,
            warmup: 1,
            throughput_count: 50,
            ..Default::default()
        };
        let result = run_benchmark(
            "field_test",
            || {
                std::hint::black_box(vec![1u8; 32]);
            },
            &cfg,
        )
        .expect("failed");
        // Verify all fields are accessible and sensible.
        assert_eq!(result.iterations, 3);
        assert!(result.mean_ns >= 0.0);
        assert!(result.median_ns >= 0.0);
        assert!(result.min_ns <= result.mean_ns + 1.0 || result.min_ns >= 0.0);
        assert!(result.max_ns >= result.min_ns);
        assert!(result.std_ns >= 0.0);
        assert!(result.throughput > 0.0);
        assert!(result.cv >= 0.0);
    }

    // -----------------------------------------------------------------------
    // SuiteReport fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_suite_report_fields() {
        let r = BenchmarkResult {
            name: "r1".to_string(),
            iterations: 5,
            mean_ns: 250.0,
            median_ns: 248.0,
            min_ns: 230.0,
            max_ns: 270.0,
            std_ns: 12.0,
            throughput: 4e6,
            throughput_unit: "items".to_string(),
            cv:0.048,
        };
        let report = SuiteReport {
            n_benchmarks: 1,
            fastest: "r1".to_string(),
            slowest: "r1".to_string(),
            total_time_ms: 0.125,
            results: vec![r.clone()],
        };
        assert_eq!(report.n_benchmarks, 1);
        assert_eq!(report.fastest, "r1");
        assert_eq!(report.slowest, "r1");
        assert!((report.total_time_ms - 0.125).abs() < 1e-9);
        assert_eq!(report.results.len(), 1);
    }

    // -----------------------------------------------------------------------
    // BenchmarkComparison speedup direction
    // -----------------------------------------------------------------------

    #[test]
    fn test_comparison_speedup_a_faster() {
        let a = BenchmarkResult {
            name: "a".to_string(),
            iterations: 5,
            mean_ns: 50.0,
            median_ns: 50.0,
            min_ns: 45.0,
            max_ns: 55.0,
            std_ns: 2.0,
            throughput: 2e7,
            throughput_unit: "items".to_string(),
            cv:0.04,
        };
        let b = BenchmarkResult {
            name: "b".to_string(),
            iterations: 5,
            mean_ns: 200.0,
            median_ns: 200.0,
            min_ns: 190.0,
            max_ns: 210.0,
            std_ns: 5.0,
            throughput: 5e6,
            throughput_unit: "items".to_string(),
            cv:0.025,
        };
        let cmp = compare_benchmarks(&a, &b);
        // speedup = 50/200 = 0.25 → a is faster
        assert!((cmp.speedup - 0.25).abs() < 0.01);
        assert_eq!(cmp.faster, "a");
    }

    #[test]
    fn test_xorshift64_produces_nonzero() {
        let mut state: u64 = 12345;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0);
        // State must have changed.
        assert_ne!(state, 12345);
    }
}
