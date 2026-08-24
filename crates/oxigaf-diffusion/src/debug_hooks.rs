//! NaN/Inf detection hooks for the oxigaf-diffusion pipeline.
//!
//! Provides [`TensorHealth`], [`DebugHooks`], [`DebugConfig`], and helper
//! functions for detecting and recording numerical anomalies in f32 tensor
//! data anywhere in the diffusion pipeline.
//!
//! # Quick start
//!
//! ```rust
//! use oxigaf_diffusion::debug_hooks::{all_finite, assert_finite, check_tensor_health};
//!
//! let data = vec![0.0f32, 1.0, 2.0, f32::NAN];
//! let health = check_tensor_health("my_tensor", &data);
//! assert!(!health.is_healthy);
//! assert_eq!(health.nan_count, 1);
//! ```

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

use crate::DiffusionError;

// ---------------------------------------------------------------------------
// TensorHealth
// ---------------------------------------------------------------------------

/// Health status of a tensor after checking for NaN/Inf values.
///
/// Contains full diagnostics: counts of each anomaly type, the index of the
/// first anomaly, and summary statistics over finite-only values.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorHealth {
    /// Name / label given at check time.
    pub name: String,
    /// Total number of elements examined.
    pub total_elements: usize,
    /// Number of `NaN` values.
    pub nan_count: usize,
    /// Number of `+∞` values.
    pub pos_inf_count: usize,
    /// Number of `-∞` values.
    pub neg_inf_count: usize,
    /// Number of values that are finite (not NaN and not Inf).
    pub finite_count: usize,
    /// Index of the first element that is NaN or Inf, if any.
    pub first_bad_index: Option<usize>,
    /// Minimum of finite values, or `None` if no finite values exist.
    pub min_finite: Option<f32>,
    /// Maximum of finite values, or `None` if no finite values exist.
    pub max_finite: Option<f32>,
    /// Mean of finite values as f64 for numerical precision, or `None` if no
    /// finite values exist.
    pub mean_finite: Option<f64>,
    /// `true` iff `nan_count == 0 && pos_inf_count == 0 && neg_inf_count == 0`.
    pub is_healthy: bool,
}

// ---------------------------------------------------------------------------
// check_tensor_health
// ---------------------------------------------------------------------------

/// Scan a flat `f32` slice and produce a detailed [`TensorHealth`] report.
///
/// This is a single-pass O(N) scan that simultaneously computes all counts,
/// finds the first bad index, and accumulates statistics over finite values.
pub fn check_tensor_health(name: impl Into<String>, data: &[f32]) -> TensorHealth {
    let name = name.into();
    let total_elements = data.len();

    if total_elements == 0 {
        return TensorHealth {
            name,
            total_elements: 0,
            nan_count: 0,
            pos_inf_count: 0,
            neg_inf_count: 0,
            finite_count: 0,
            first_bad_index: None,
            min_finite: None,
            max_finite: None,
            mean_finite: None,
            is_healthy: true,
        };
    }

    let mut nan_count = 0usize;
    let mut pos_inf_count = 0usize;
    let mut neg_inf_count = 0usize;
    let mut finite_count = 0usize;
    let mut first_bad_index: Option<usize> = None;

    // Running statistics over finite values (Welford-style for mean).
    let mut min_finite = f32::MAX;
    let mut max_finite = f32::MIN;
    let mut sum_finite = 0f64;

    for (i, &v) in data.iter().enumerate() {
        if v.is_nan() {
            nan_count += 1;
            if first_bad_index.is_none() {
                first_bad_index = Some(i);
            }
        } else if v == f32::INFINITY {
            pos_inf_count += 1;
            if first_bad_index.is_none() {
                first_bad_index = Some(i);
            }
        } else if v == f32::NEG_INFINITY {
            neg_inf_count += 1;
            if first_bad_index.is_none() {
                first_bad_index = Some(i);
            }
        } else {
            // Finite value
            finite_count += 1;
            if v < min_finite {
                min_finite = v;
            }
            if v > max_finite {
                max_finite = v;
            }
            sum_finite += v as f64;
        }
    }

    let is_healthy = nan_count == 0 && pos_inf_count == 0 && neg_inf_count == 0;

    let (min_finite_opt, max_finite_opt, mean_finite_opt) = if finite_count > 0 {
        (
            Some(min_finite),
            Some(max_finite),
            Some(sum_finite / finite_count as f64),
        )
    } else {
        (None, None, None)
    };

    TensorHealth {
        name,
        total_elements,
        nan_count,
        pos_inf_count,
        neg_inf_count,
        finite_count,
        first_bad_index,
        min_finite: min_finite_opt,
        max_finite: max_finite_opt,
        mean_finite: mean_finite_opt,
        is_healthy,
    }
}

// ---------------------------------------------------------------------------
// Fast-path helpers
// ---------------------------------------------------------------------------

/// Returns `true` if every element of `data` is finite (not NaN, not Inf).
///
/// This is an O(N) early-exit fast path for hot loops. It avoids the overhead
/// of building a full [`TensorHealth`] report when no anomaly is expected.
#[inline]
pub fn all_finite(data: &[f32]) -> bool {
    data.iter().all(|v| v.is_finite())
}

/// Check a slice and return `Err` if any NaN or Inf is found.
///
/// Produces a detailed error variant with counts and the index of the first
/// bad element. Use this at pipeline boundaries where anomalies must be
/// treated as hard errors.
pub fn assert_finite(name: &str, data: &[f32]) -> Result<(), DiffusionError> {
    let health = check_tensor_health(name, data);
    if health.is_healthy {
        Ok(())
    } else {
        Err(DiffusionError::NanInfDetected {
            name: health.name.clone(),
            nan_count: health.nan_count,
            inf_count: health.pos_inf_count + health.neg_inf_count,
            first_index: health.first_bad_index.unwrap_or(0),
        })
    }
}

// ---------------------------------------------------------------------------
// DebugConfig
// ---------------------------------------------------------------------------

/// Configuration for the [`DebugHooks`] registry.
///
/// Controls whether checks are active, how to react to detected anomalies,
/// and how many bad-tensor records to retain.
#[derive(Debug, Clone)]
pub struct DebugConfig {
    /// When `false`, calls to [`DebugHooks::check`] become no-ops (the tensor
    /// is marked healthy without scanning). Useful to compile hooks into
    /// release builds but disable them at runtime.
    pub enabled: bool,
    /// If `true`, panic immediately upon detecting any NaN.
    ///
    /// Only takes effect when this crate is built with the `gpu_debug`
    /// Cargo feature; otherwise it is accepted but has no effect, so a
    /// default (production) build never contains a reachable `panic!` from
    /// this path.
    pub panic_on_nan: bool,
    /// If `true`, panic immediately upon detecting any Inf.
    ///
    /// Same `gpu_debug`-feature gating as [`DebugConfig::panic_on_nan`].
    pub panic_on_inf: bool,
    /// If `true`, log even healthy tensors via `tracing::trace!`.
    pub log_all_checks: bool,
    /// Maximum number of bad-tensor records to retain. Older records are
    /// dropped when this limit is reached (FIFO eviction). `0` retains no
    /// records at all.
    pub max_records: usize,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            enabled: cfg!(feature = "gpu_debug"),
            panic_on_nan: false,
            panic_on_inf: false,
            log_all_checks: false,
            max_records: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// DebugHooks
// ---------------------------------------------------------------------------

/// Thread-safe, global-registry of NaN/Inf checks and recorded anomalies.
///
/// Create one instance per pipeline (or use a `std::sync::OnceLock` for a
/// process-global instance). Each call to [`DebugHooks::check`] atomically
/// increments counters and, if an anomaly is detected, appends the report to
/// an internal ring buffer (bounded by [`DebugConfig::max_records`]).
pub struct DebugHooks {
    config: DebugConfig,
    records: Mutex<Vec<TensorHealth>>,
    check_count: AtomicU64,
    bad_count: AtomicU64,
}

impl DebugHooks {
    /// Create a new registry with the given configuration.
    pub fn new(config: DebugConfig) -> Self {
        Self {
            config,
            records: Mutex::new(Vec::new()),
            check_count: AtomicU64::new(0),
            bad_count: AtomicU64::new(0),
        }
    }

    /// Scan `data` for NaN/Inf and record the result if unhealthy.
    ///
    /// When [`DebugConfig::enabled`] is `false` the scan is skipped and a
    /// synthetic healthy report is returned immediately.
    ///
    /// When [`DebugConfig::panic_on_nan`] or [`DebugConfig::panic_on_inf`] is
    /// set AND this crate is built with the `gpu_debug` Cargo feature, the
    /// method panics after recording (giving you the full record before
    /// aborting). Without that feature the flags are accepted but inert,
    /// so a default (production) build never panics here.
    pub fn check(&self, name: impl Into<String>, data: &[f32]) -> TensorHealth {
        self.check_count.fetch_add(1, Ordering::Relaxed);

        if !self.config.enabled {
            // Fast no-op path: return a synthetic healthy report.
            let name = name.into();
            return TensorHealth {
                name,
                total_elements: data.len(),
                nan_count: 0,
                pos_inf_count: 0,
                neg_inf_count: 0,
                finite_count: data.len(),
                first_bad_index: None,
                min_finite: None,
                max_finite: None,
                mean_finite: None,
                is_healthy: true,
            };
        }

        let health = check_tensor_health(name, data);

        if self.config.log_all_checks || !health.is_healthy {
            tracing::trace!(
                name = %health.name,
                total = health.total_elements,
                nan = health.nan_count,
                pos_inf = health.pos_inf_count,
                neg_inf = health.neg_inf_count,
                healthy = health.is_healthy,
                "tensor health check"
            );
        }

        if !health.is_healthy {
            self.bad_count.fetch_add(1, Ordering::Relaxed);

            // Append to records with max_records eviction. max_records == 0
            // means "retain no records" rather than "evict from an empty
            // Vec", which would otherwise panic on Vec::remove(0).
            if self.config.max_records > 0 {
                let mut records = self.records.lock().unwrap_or_else(|e| e.into_inner());
                if records.len() >= self.config.max_records {
                    records.remove(0);
                }
                records.push(health.clone());
                drop(records);
            }

            // Panic after recording so the record is visible. Gated behind
            // the `gpu_debug` feature so a default (production) build never
            // contains a reachable panic!: panic_on_nan/panic_on_inf are
            // inert unless the crate is built with `--features gpu_debug`.
            #[cfg(feature = "gpu_debug")]
            if self.config.panic_on_nan && health.nan_count > 0 {
                panic!(
                    "DebugHooks: NaN detected in tensor '{}' ({} NaN values, first at index {:?})",
                    health.name, health.nan_count, health.first_bad_index
                );
            }
            #[cfg(feature = "gpu_debug")]
            if self.config.panic_on_inf && (health.pos_inf_count + health.neg_inf_count) > 0 {
                panic!(
                    "DebugHooks: Inf detected in tensor '{}' ({} +Inf, {} -Inf)",
                    health.name, health.pos_inf_count, health.neg_inf_count
                );
            }
        }

        health
    }

    /// Return a snapshot of all recorded bad-tensor reports.
    pub fn bad_records(&self) -> Vec<TensorHealth> {
        self.records
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Clear all recorded bad-tensor reports and reset the bad counter.
    ///
    /// The total check counter is NOT reset — it is monotonically increasing.
    pub fn clear(&self) {
        let mut records = self.records.lock().unwrap_or_else(|e| e.into_inner());
        records.clear();
        self.bad_count.store(0, Ordering::Relaxed);
    }

    /// Return `(total_checks, bad_checks)` since construction (or since last
    /// `clear()` for the bad counter).
    pub fn stats(&self) -> (u64, u64) {
        (
            self.check_count.load(Ordering::Relaxed),
            self.bad_count.load(Ordering::Relaxed),
        )
    }

    /// Returns `true` if at least one NaN/Inf anomaly has been recorded since
    /// the last [`DebugHooks::clear`] call.
    pub fn has_issues(&self) -> bool {
        self.bad_count.load(Ordering::Relaxed) > 0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // check_tensor_health: basic cases
    // ------------------------------------------------------------------

    #[test]
    fn test_all_finite_is_healthy() {
        let data = vec![0.0f32, 1.0, -1.0, 3.1, 1e-6, 1e6];
        let h = check_tensor_health("all_finite", &data);
        assert!(h.is_healthy);
        assert_eq!(h.nan_count, 0);
        assert_eq!(h.pos_inf_count, 0);
        assert_eq!(h.neg_inf_count, 0);
        assert_eq!(h.finite_count, 6);
        assert_eq!(h.total_elements, 6);
        assert!(h.first_bad_index.is_none());
    }

    #[test]
    fn test_single_nan_detected_correct_index() {
        let data = vec![1.0f32, 2.0, f32::NAN, 4.0];
        let h = check_tensor_health("single_nan", &data);
        assert!(!h.is_healthy);
        assert_eq!(h.nan_count, 1);
        assert_eq!(h.first_bad_index, Some(2));
    }

    #[test]
    fn test_multiple_nans_count_correct() {
        let data = vec![f32::NAN, 1.0, f32::NAN, f32::NAN, 5.0];
        let h = check_tensor_health("multi_nan", &data);
        assert!(!h.is_healthy);
        assert_eq!(h.nan_count, 3);
        assert_eq!(h.finite_count, 2);
        // First bad index is 0
        assert_eq!(h.first_bad_index, Some(0));
    }

    #[test]
    fn test_pos_inf_detected() {
        let data = vec![0.0f32, f32::INFINITY, 2.0];
        let h = check_tensor_health("pos_inf", &data);
        assert!(!h.is_healthy);
        assert_eq!(h.pos_inf_count, 1);
        assert_eq!(h.neg_inf_count, 0);
        assert_eq!(h.nan_count, 0);
        assert_eq!(h.first_bad_index, Some(1));
    }

    #[test]
    fn test_neg_inf_detected() {
        let data = vec![0.0f32, f32::NEG_INFINITY, 2.0];
        let h = check_tensor_health("neg_inf", &data);
        assert!(!h.is_healthy);
        assert_eq!(h.neg_inf_count, 1);
        assert_eq!(h.pos_inf_count, 0);
        assert_eq!(h.nan_count, 0);
        assert_eq!(h.first_bad_index, Some(1));
    }

    #[test]
    fn test_mixed_nan_and_inf_both_counted() {
        let data = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0, 2.0];
        let h = check_tensor_health("mixed", &data);
        assert!(!h.is_healthy);
        assert_eq!(h.nan_count, 1);
        assert_eq!(h.pos_inf_count, 1);
        assert_eq!(h.neg_inf_count, 1);
        assert_eq!(h.finite_count, 2);
        // First bad is index 0 (NaN)
        assert_eq!(h.first_bad_index, Some(0));
    }

    #[test]
    fn test_empty_tensor_is_healthy() {
        let data: Vec<f32> = vec![];
        let h = check_tensor_health("empty", &data);
        assert!(h.is_healthy);
        assert_eq!(h.total_elements, 0);
        assert_eq!(h.nan_count, 0);
        assert_eq!(h.finite_count, 0);
        assert!(h.first_bad_index.is_none());
        assert!(h.min_finite.is_none());
        assert!(h.max_finite.is_none());
        assert!(h.mean_finite.is_none());
    }

    #[test]
    fn test_all_nan_tensor_mean_is_none() {
        let data = vec![f32::NAN; 5];
        let h = check_tensor_health("all_nan", &data);
        assert!(!h.is_healthy);
        assert_eq!(h.nan_count, 5);
        assert_eq!(h.finite_count, 0);
        assert!(h.mean_finite.is_none());
        assert!(h.min_finite.is_none());
        assert!(h.max_finite.is_none());
    }

    #[test]
    fn test_statistics_min_max_mean_correct() {
        // finite values: 2.0, 4.0, 6.0 → min=2, max=6, mean=4
        let data = vec![2.0f32, f32::NAN, 4.0, f32::INFINITY, 6.0];
        let h = check_tensor_health("stats", &data);
        assert!(!h.is_healthy);
        assert_eq!(h.finite_count, 3);
        let min = h.min_finite.expect("min should be Some");
        let max = h.max_finite.expect("max should be Some");
        let mean = h.mean_finite.expect("mean should be Some");
        assert!((min - 2.0f32).abs() < 1e-6, "min={min}");
        assert!((max - 6.0f32).abs() < 1e-6, "max={max}");
        assert!((mean - 4.0f64).abs() < 1e-9, "mean={mean}");
    }

    #[test]
    fn test_first_bad_index_is_correct_when_bad_not_at_start() {
        // Index 0, 1, 2 are fine; index 3 is NaN
        let data = vec![10.0f32, 20.0, 30.0, f32::NAN, 50.0];
        let h = check_tensor_health("bad_at_3", &data);
        assert_eq!(h.first_bad_index, Some(3));
    }

    // ------------------------------------------------------------------
    // all_finite fast path
    // ------------------------------------------------------------------

    #[test]
    fn test_all_finite_fast_path_true() {
        let data = vec![0.0f32, 1.0, -1.0, 100.0];
        assert!(all_finite(&data));
    }

    #[test]
    fn test_all_finite_fast_path_false_nan() {
        let mut data = vec![0.0f32; 100];
        data[50] = f32::NAN;
        assert!(!all_finite(&data));
    }

    #[test]
    fn test_all_finite_fast_path_false_inf() {
        let data = vec![1.0f32, f32::INFINITY];
        assert!(!all_finite(&data));
    }

    // ------------------------------------------------------------------
    // assert_finite
    // ------------------------------------------------------------------

    #[test]
    fn test_assert_finite_ok_for_good_data() {
        let data = vec![0.0f32, 1.0, 2.0];
        assert!(assert_finite("good", &data).is_ok());
    }

    #[test]
    fn test_assert_finite_err_for_nan() {
        let data = vec![1.0f32, f32::NAN, 3.0];
        let result = assert_finite("with_nan", &data);
        assert!(result.is_err());
        match result {
            Err(DiffusionError::NanInfDetected {
                name,
                nan_count,
                inf_count,
                first_index,
            }) => {
                assert_eq!(name, "with_nan");
                assert_eq!(nan_count, 1);
                assert_eq!(inf_count, 0);
                assert_eq!(first_index, 1);
            }
            other => panic!("Expected NanInfDetected, got {:?}", other),
        }
    }

    #[test]
    fn test_assert_finite_err_for_inf() {
        let data = vec![f32::INFINITY, 2.0f32, f32::NEG_INFINITY];
        let result = assert_finite("with_inf", &data);
        assert!(result.is_err());
        match result {
            Err(DiffusionError::NanInfDetected {
                nan_count,
                inf_count,
                first_index,
                ..
            }) => {
                assert_eq!(nan_count, 0);
                assert_eq!(inf_count, 2);
                assert_eq!(first_index, 0);
            }
            other => panic!("Expected NanInfDetected, got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // DebugHooks
    // ------------------------------------------------------------------

    #[test]
    fn test_debug_hooks_accumulate_multiple_checks() {
        let hooks = DebugHooks::new(DebugConfig {
            enabled: true,
            ..Default::default()
        });

        let good = vec![1.0f32, 2.0, 3.0];
        let bad1 = vec![f32::NAN, 2.0f32];
        let bad2 = vec![f32::INFINITY, 1.0f32];

        hooks.check("good", &good);
        hooks.check("bad1", &bad1);
        hooks.check("bad2", &bad2);

        let records = hooks.bad_records();
        assert_eq!(records.len(), 2, "should have 2 bad records");
        assert_eq!(records[0].name, "bad1");
        assert_eq!(records[1].name, "bad2");
    }

    #[test]
    fn test_debug_hooks_stats_counts_correct() {
        let hooks = DebugHooks::new(DebugConfig {
            enabled: true,
            ..Default::default()
        });

        let good = vec![1.0f32, 2.0, 3.0];
        let bad = vec![f32::NAN, 2.0f32];

        hooks.check("g1", &good);
        hooks.check("g2", &good);
        hooks.check("b1", &bad);

        let (total, bad_count) = hooks.stats();
        assert_eq!(total, 3, "3 total checks");
        assert_eq!(bad_count, 1, "1 bad check");
    }

    #[test]
    fn test_debug_hooks_clear_resets_state() {
        let hooks = DebugHooks::new(DebugConfig {
            enabled: true,
            ..Default::default()
        });

        let bad = vec![f32::NAN, 2.0f32];
        hooks.check("b1", &bad);
        hooks.check("b2", &bad);

        assert!(hooks.has_issues());
        hooks.clear();

        assert!(!hooks.has_issues());
        let records = hooks.bad_records();
        assert!(records.is_empty(), "records should be empty after clear");

        let (total, bad_count) = hooks.stats();
        // Total check count is NOT reset
        assert_eq!(total, 2, "total checks preserved");
        assert_eq!(bad_count, 0, "bad count reset to 0");
    }

    #[test]
    fn test_debug_hooks_max_records_eviction() {
        let hooks = DebugHooks::new(DebugConfig {
            enabled: true,
            max_records: 3,
            ..Default::default()
        });

        let bad = vec![f32::NAN];
        for i in 0..5usize {
            hooks.check(format!("tensor_{i}"), &bad);
        }

        let records = hooks.bad_records();
        // Max 3 records kept, oldest evicted
        assert_eq!(records.len(), 3);
        // The retained names should be tensor_2, tensor_3, tensor_4
        assert_eq!(records[0].name, "tensor_2");
        assert_eq!(records[1].name, "tensor_3");
        assert_eq!(records[2].name, "tensor_4");
    }

    #[test]
    fn test_debug_hooks_max_records_zero_does_not_panic() {
        let hooks = DebugHooks::new(DebugConfig {
            enabled: true,
            max_records: 0,
            ..Default::default()
        });

        let bad = vec![f32::NAN];
        // Must not panic (max_records == 0 used to hit Vec::remove(0) on
        // an empty Vec).
        let h = hooks.check("t0", &bad);
        assert!(!h.is_healthy);

        let records = hooks.bad_records();
        assert!(records.is_empty(), "max_records=0 should retain no records");

        let (_, bad_count) = hooks.stats();
        assert_eq!(bad_count, 1, "bad_count should still increment");
    }

    #[test]
    #[cfg(not(feature = "gpu_debug"))]
    fn test_debug_hooks_panic_on_nan_inert_without_gpu_debug_feature() {
        // panic_on_nan/panic_on_inf only take effect when this crate is
        // built with the `gpu_debug` feature; in a default build (as tests
        // normally run) they must never cause check() to panic.
        let hooks = DebugHooks::new(DebugConfig {
            enabled: true,
            panic_on_nan: true,
            panic_on_inf: true,
            ..Default::default()
        });
        let bad = vec![f32::NAN, f32::INFINITY];
        // Must not panic.
        let h = hooks.check("should_not_panic", &bad);
        assert!(!h.is_healthy);
        assert_eq!(h.nan_count, 1);
    }

    #[test]
    fn test_debug_hooks_disabled_skips_scan() {
        let hooks = DebugHooks::new(DebugConfig {
            enabled: false,
            ..Default::default()
        });

        // Deliberately bad data — should be ignored when disabled
        let bad = vec![f32::NAN; 1000];
        let h = hooks.check("disabled", &bad);

        // Returns synthetic healthy report
        assert!(h.is_healthy);
        assert_eq!(h.nan_count, 0);
        assert!(!hooks.has_issues());
    }

    #[test]
    fn test_debug_hooks_has_issues_false_initially() {
        let hooks = DebugHooks::new(DebugConfig::default());
        assert!(!hooks.has_issues());
    }
}
