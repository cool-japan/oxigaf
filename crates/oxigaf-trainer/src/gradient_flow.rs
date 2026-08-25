//! Gradient flow tracking and analysis for 3D Gaussian Splatting training.
//!
//! This module monitors how gradients flow through each Gaussian parameter
//! group (positions, rotations, scales, opacities, SH-DC, SH-rest), detecting
//! vanishing/exploding gradients, tracking per-group learning signals, and
//! providing data for gradient flow visualization.
//!
//! # Key Distinctions
//! - **gradient_clipping.rs**: clips gradients
//! - **gradient_accumulation.rs**: accumulates across mini-batches
//! - **diagnostics.rs**: general training stats
//! - **gradient_flow.rs** (this module): flow analysis and health classification
//!
//! # Quick start
//! ```rust
//! use oxigaf_trainer::gradient_flow::{GradientFlowTracker, GradientFlowConfig};
//!
//! let config = GradientFlowConfig::default();
//! let mut tracker = GradientFlowTracker::new(config);
//! let positions = vec![0.001_f32; 300];
//! let opacities = vec![0.0001_f32; 100];
//! let snapshot = tracker.record(1, vec![
//!     ("positions".to_string(), positions.as_slice()),
//!     ("opacities".to_string(), opacities.as_slice()),
//! ]).unwrap();
//! println!("total_l2_norm = {:.4e}", snapshot.total_l2_norm);
//! ```

use std::collections::VecDeque;

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by gradient-flow tracking operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum GradientFlowError {
    /// The gradient buffer for a named parameter group was empty.
    #[error("Empty gradient buffer for parameter group '{group}'")]
    EmptyGradients { group: String },

    /// No snapshots have been recorded yet.
    #[error("History is empty — call record() at least once")]
    EmptyHistory,

    /// The requested window is larger than the available history.
    #[error("Window size {window} exceeds history length {len}")]
    WindowExceedsHistory { window: usize, len: usize },

    /// A named parameter group was not found in the registry.
    #[error("Parameter group '{name}' not found in registry")]
    UnknownGroup { name: String },
}

// ─────────────────────────────────────────────────────────────────────────────
// Core data structures
// ─────────────────────────────────────────────────────────────────────────────

/// One parameter group's gradient snapshot at a single training step.
#[derive(Debug, Clone)]
pub struct GroupGradSnapshot {
    /// Name of the Gaussian parameter group (e.g., "positions").
    pub group_name: String,
    /// Training step at which this snapshot was taken.
    pub step: usize,
    /// L2 norm of the gradient vector.
    pub l2_norm: f32,
    /// Maximum absolute value (L∞ norm).
    pub l_inf_norm: f32,
    /// Mean of |gradient|.
    pub mean_abs: f32,
    /// Population standard deviation of gradient values.
    pub std: f32,
    /// Number of gradient parameters.
    pub n_params: usize,
    /// Whether any gradient element is NaN.
    pub has_nan: bool,
    /// Whether any gradient element is infinite.
    pub has_inf: bool,
}

/// Multi-group gradient snapshot at a single training step.
#[derive(Debug, Clone)]
pub struct FlowSnapshot {
    /// Training step index.
    pub step: usize,
    /// One snapshot per parameter group.
    pub groups: Vec<GroupGradSnapshot>,
    /// L2 norm across ALL groups combined: sqrt(Σ l2_norm_i²).
    pub total_l2_norm: f32,
}

/// Classification of gradient flow health for one group.
///
/// Declaration order below (least to most severe) doubles as the derived
/// [`Ord`] ranking: `Healthy < Vanishing < Exploding < Dead < NanOrInf`.
/// [`worst_health`] and [`classify_flow_health`]'s documented priority both
/// derive from this single ordering, so they cannot drift apart again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FlowHealth {
    /// Gradient norms are in a reasonable range.
    Healthy,
    /// L2 norm is below the vanish threshold (but not exactly zero).
    Vanishing,
    /// L2 norm is above the explode threshold.
    Exploding,
    /// L2 norm is exactly zero.
    Dead,
    /// Contains NaN or Inf.
    NanOrInf,
}

/// Trend direction of gradient norms over a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradTrend {
    /// Norm variation is below the stable coefficient-of-variation threshold.
    Stable,
    /// Norms are trending upward (may indicate divergence).
    Increasing,
    /// Norms are trending downward (may indicate vanishing).
    Decreasing,
    /// High variance with no clear monotone trend.
    Oscillating,
}

/// Configuration for gradient flow tracking.
#[derive(Debug, Clone)]
pub struct GradientFlowConfig {
    /// Maximum number of steps to retain in history.  Default: 1000.
    pub history_capacity: usize,
    /// L2 norm below this → `Vanishing`.  Default: 1e-6.
    pub vanish_threshold: f32,
    /// L2 norm above this → `Exploding`.  Default: 100.0.
    pub explode_threshold: f32,
    /// Number of steps to look back when computing trends.  Default: 50.
    pub trend_window: usize,
    /// Coefficient-of-variation threshold for `Stable` classification.  Default: 0.1.
    pub stable_cv: f32,
}

impl Default for GradientFlowConfig {
    fn default() -> Self {
        Self {
            history_capacity: 1000,
            vanish_threshold: 1e-6,
            explode_threshold: 100.0,
            trend_window: 50,
            stable_cv: 0.1,
        }
    }
}

/// Per-group analysis report over a history window.
#[derive(Debug, Clone)]
pub struct GroupFlowReport {
    /// Name of the Gaussian parameter group.
    pub group_name: String,
    /// Health classification.
    pub health: FlowHealth,
    /// Trend direction.
    pub trend: GradTrend,
    /// Mean L2 norm over the window.
    pub mean_norm: f32,
    /// Maximum L2 norm in the window.
    pub peak_norm: f32,
    /// Minimum L2 norm in the window.
    pub min_norm: f32,
    /// Fraction of total gradient signal from this group
    /// (mean_norm / Σ all-group mean_norms).
    pub relative_signal: f32,
}

/// Full gradient flow analysis for all groups at a given step.
#[derive(Debug, Clone)]
pub struct GradientFlowReport {
    /// Training step.
    pub step: usize,
    /// Per-group reports.
    pub groups: Vec<GroupFlowReport>,
    /// Group with the highest `relative_signal`.
    pub dominant_group: String,
    /// Group with the lowest `relative_signal`.
    pub weakest_group: String,
    /// Worst health classification across all groups.
    pub overall_health: FlowHealth,
}

/// Stateful gradient flow tracker.
///
/// Maintains a bounded history of [`FlowSnapshot`]s and provides analysis
/// methods for gradient health and trend classification.
pub struct GradientFlowTracker {
    /// Configuration parameters.
    pub config: GradientFlowConfig,
    /// Chronological history of flow snapshots.
    history: VecDeque<FlowSnapshot>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Free-function norm utilities (prefixed with `flow_` to avoid name conflicts)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the L2 norm of a gradient slice: sqrt(Σ xᵢ²).
///
/// Returns `0.0` for an empty slice.
pub fn flow_l2_norm(gradients: &[f32]) -> f32 {
    let sum_sq: f32 = gradients.iter().map(|&x| x * x).sum();
    sum_sq.sqrt()
}

/// Compute the L∞ norm (max |xᵢ|) of a gradient slice.
///
/// Returns `0.0` for an empty slice.
pub fn flow_l_inf_norm(gradients: &[f32]) -> f32 {
    gradients.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max)
}

/// Compute the mean absolute value of a gradient slice: (Σ |xᵢ|) / n.
///
/// Returns `0.0` for an empty slice.
pub fn flow_mean_abs(gradients: &[f32]) -> f32 {
    if gradients.is_empty() {
        return 0.0;
    }
    let sum: f32 = gradients.iter().map(|&x| x.abs()).sum();
    sum / gradients.len() as f32
}

/// Compute the population standard deviation of a gradient slice.
///
/// Formula: sqrt(E\[x²\] - E\[x\]²), clamped to 0 before sqrt to avoid
/// floating-point negative due to catastrophic cancellation.
///
/// Returns `0.0` for an empty slice.
pub fn flow_std(gradients: &[f32]) -> f32 {
    if gradients.is_empty() {
        return 0.0;
    }
    let n = gradients.len() as f32;
    let mean_x: f32 = gradients.iter().sum::<f32>() / n;
    let mean_x2: f32 = gradients.iter().map(|&x| x * x).sum::<f32>() / n;
    // Clamp to avoid negative due to floating-point error.
    let variance = (mean_x2 - mean_x * mean_x).max(0.0);
    variance.sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Snapshot computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a [`GroupGradSnapshot`] from raw gradient values.
///
/// Returns [`GradientFlowError::EmptyGradients`] if `gradients` is empty.
pub fn compute_group_snapshot(
    group_name: &str,
    step: usize,
    gradients: &[f32],
) -> Result<GroupGradSnapshot, GradientFlowError> {
    if gradients.is_empty() {
        return Err(GradientFlowError::EmptyGradients {
            group: group_name.to_string(),
        });
    }

    let has_nan = gradients.iter().any(|x| x.is_nan());
    let has_inf = gradients.iter().any(|x| x.is_infinite());

    // When NaN or Inf is present, norm computations are unreliable,
    // but we still fill them in (they may themselves be NaN/Inf).
    let l2 = flow_l2_norm(gradients);
    let l_inf = flow_l_inf_norm(gradients);
    let mean_abs = flow_mean_abs(gradients);
    let std = flow_std(gradients);

    Ok(GroupGradSnapshot {
        group_name: group_name.to_string(),
        step,
        l2_norm: l2,
        l_inf_norm: l_inf,
        mean_abs,
        std,
        n_params: gradients.len(),
        has_nan,
        has_inf,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Health and trend classification
// ─────────────────────────────────────────────────────────────────────────────

/// Classify gradient flow health based on L2 norm and NaN/Inf flags.
///
/// Priority (highest to lowest): NanOrInf → Dead → Exploding → Vanishing → Healthy.
pub fn classify_flow_health(
    l2_norm: f32,
    has_nan: bool,
    has_inf: bool,
    config: &GradientFlowConfig,
) -> FlowHealth {
    if has_nan || has_inf {
        return FlowHealth::NanOrInf;
    }
    if l2_norm == 0.0 {
        return FlowHealth::Dead;
    }
    if l2_norm > config.explode_threshold {
        return FlowHealth::Exploding;
    }
    if l2_norm < config.vanish_threshold {
        return FlowHealth::Vanishing;
    }
    FlowHealth::Healthy
}

/// Return the worse of two [`FlowHealth`] values.
///
/// Ordering (worst first): NanOrInf > Dead > Exploding > Vanishing > Healthy,
/// matching the priority documented on [`classify_flow_health`] — both are
/// derived from [`FlowHealth`]'s single [`Ord`] impl (see its doc).
pub fn worst_health(a: FlowHealth, b: FlowHealth) -> FlowHealth {
    a.max(b)
}

/// Compute the linear regression slope for a sequence of values.
///
/// Uses indices `[0, 1, …, n-1]` as x-values.
/// slope = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
///
/// Returns `0.0` when the denominator is near zero (fewer than 2 points or
/// all x-values are the same, which cannot happen with sequential indices).
pub fn flow_linear_regression(values: &[f32]) -> f32 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f32;
    let sum_x: f32 = (0..n).map(|i| i as f32).sum();
    let sum_y: f32 = values.iter().sum();
    let sum_xy: f32 = values.iter().enumerate().map(|(i, &y)| i as f32 * y).sum();
    let sum_x2: f32 = (0..n).map(|i| (i as f32) * (i as f32)).sum();

    let denom = nf * sum_x2 - sum_x * sum_x;
    if denom.abs() < f32::EPSILON {
        return 0.0;
    }
    (nf * sum_xy - sum_x * sum_y) / denom
}

/// Classify the gradient norm trend over a time-ordered slice of L2 norms.
///
/// - Empty or single-element input → `Stable`
/// - coefficient of variation (std/mean) < `config.stable_cv` → `Stable`
/// - Otherwise, normalise the linear-regression slope into a window-relative
///   fractional change (`rel_slope`, see below); if its magnitude is still
///   below `config.stable_cv` → `Oscillating` (the variance is high but not
///   explained by a monotone trend — i.e. noise, not drift)
/// - `rel_slope > 0` → `Increasing`
/// - `rel_slope <= 0` → `Decreasing`
///
/// `rel_slope = slope * n / mean.abs()` estimates the total fractional
/// change predicted by the fitted line across the whole window, so it is on
/// the same (dimensionless) scale as `stable_cv` and does not require the
/// slope to be bit-exactly `0.0` to detect a genuinely trendless, noisy
/// sequence.
pub fn compute_grad_trend(norms: &[f32], config: &GradientFlowConfig) -> GradTrend {
    if norms.len() < 2 {
        return GradTrend::Stable;
    }

    let n = norms.len() as f32;
    let mean: f32 = norms.iter().sum::<f32>() / n;

    // Avoid division by zero when mean is essentially 0.
    let cv = if mean.abs() < f32::EPSILON {
        // All norms near zero → consider stable.
        0.0
    } else {
        let mean_sq: f32 = norms.iter().map(|&x| x * x).sum::<f32>() / n;
        let variance = (mean_sq - mean * mean).max(0.0);
        variance.sqrt() / mean.abs()
    };

    if cv < config.stable_cv {
        return GradTrend::Stable;
    }

    let slope = flow_linear_regression(norms);
    let rel_slope = if mean.abs() < f32::EPSILON {
        0.0
    } else {
        slope * n / mean.abs()
    };

    if rel_slope.abs() < config.stable_cv {
        GradTrend::Oscillating
    } else if rel_slope > 0.0 {
        GradTrend::Increasing
    } else {
        GradTrend::Decreasing
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GradientFlowTracker
// ─────────────────────────────────────────────────────────────────────────────

impl GradientFlowTracker {
    /// Create a new tracker with the given configuration.
    pub fn new(config: GradientFlowConfig) -> Self {
        Self {
            config,
            history: VecDeque::new(),
        }
    }

    /// Record gradient snapshots for all parameter groups at the given step.
    ///
    /// Each element of `groups` is `(group_name, gradient_slice)`.
    /// Returns an error if any group's gradient slice is empty.
    ///
    /// When history exceeds `config.history_capacity`, the oldest entry is evicted.
    pub fn record(
        &mut self,
        step: usize,
        groups: Vec<(String, &[f32])>,
    ) -> Result<FlowSnapshot, GradientFlowError> {
        let mut group_snaps = Vec::with_capacity(groups.len());

        for (name, grads) in &groups {
            let snap = compute_group_snapshot(name, step, grads)?;
            group_snaps.push(snap);
        }

        // total_l2_norm = sqrt(Σ l2_norms_i²)
        let total_l2_norm = {
            let sum_sq: f32 = group_snaps.iter().map(|s| s.l2_norm * s.l2_norm).sum();
            sum_sq.sqrt()
        };

        let snapshot = FlowSnapshot {
            step,
            groups: group_snaps,
            total_l2_norm,
        };

        // Evict oldest entry if at capacity. `VecDeque::pop_front` is O(1),
        // unlike `Vec::remove(0)` which shifts the entire backing buffer on
        // every eviction.
        if self.history.len() >= self.config.history_capacity {
            self.history.pop_front();
        }
        self.history.push_back(snapshot.clone());

        Ok(snapshot)
    }

    /// Return the number of snapshots currently stored in history.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Return the most recently recorded snapshot, or `None` if history is empty.
    pub fn latest_snapshot(&self) -> Option<&FlowSnapshot> {
        // `VecDeque` has no inherent `last()` (it does not `Deref<Target =
        // [T]>` the way `Vec` does) — `back()` is the O(1) equivalent for
        // the most-recently-`push_back`-ed element.
        self.history.back()
    }

    /// Analyze the gradient flow for one named parameter group over the last
    /// `window` recorded steps.
    ///
    /// Returns [`GradientFlowError::EmptyHistory`] if no snapshots exist,
    /// [`GradientFlowError::WindowExceedsHistory`] if `window > history_len()`,
    /// and [`GradientFlowError::UnknownGroup`] if the group name is not found
    /// in the relevant history window.
    pub fn analyze_group(
        &self,
        group_name: &str,
        window: usize,
    ) -> Result<GroupFlowReport, GradientFlowError> {
        if self.history.is_empty() {
            return Err(GradientFlowError::EmptyHistory);
        }
        if window > self.history.len() {
            return Err(GradientFlowError::WindowExceedsHistory {
                window,
                len: self.history.len(),
            });
        }

        let start = self.history.len() - window;
        // `VecDeque` does not implement `Index<Range<usize>>`, so collect
        // the window into a `Vec` of references via `range()` instead of
        // slicing directly.
        let window_snaps: Vec<&FlowSnapshot> = self.history.range(start..).collect();

        // Collect L2 norms for this group across the window.
        let mut norms: Vec<f32> = Vec::new();
        let mut any_nan = false;
        let mut any_inf = false;

        for snap in &window_snaps {
            for g in &snap.groups {
                if g.group_name == group_name {
                    norms.push(g.l2_norm);
                    if g.has_nan {
                        any_nan = true;
                    }
                    if g.has_inf {
                        any_inf = true;
                    }
                    break;
                }
            }
        }

        if norms.is_empty() {
            return Err(GradientFlowError::UnknownGroup {
                name: group_name.to_string(),
            });
        }

        let mean_norm: f32 = norms.iter().sum::<f32>() / norms.len() as f32;
        let peak_norm: f32 = norms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_norm: f32 = norms.iter().cloned().fold(f32::INFINITY, f32::min);

        let health = classify_flow_health(mean_norm, any_nan, any_inf, &self.config);
        let trend = compute_grad_trend(&norms, &self.config);

        // Compute relative_signal as this group's mean_norm vs the sum of all
        // group mean_norms.  We compute the latter from the same window.
        let total_mean: f32 = self.group_total_mean_norm_in_window(&window_snaps);
        let relative_signal = if total_mean > 0.0 {
            mean_norm / total_mean
        } else {
            0.0
        };

        Ok(GroupFlowReport {
            group_name: group_name.to_string(),
            health,
            trend,
            mean_norm,
            peak_norm,
            min_norm,
            relative_signal,
        })
    }

    /// Analyze gradient flow for ALL groups present in the last `window` steps.
    ///
    /// Derives group names from the most recent snapshot, then calls
    /// [`analyze_group`](Self::analyze_group) for each.
    pub fn analyze_all(&self, window: usize) -> Result<GradientFlowReport, GradientFlowError> {
        if self.history.is_empty() {
            return Err(GradientFlowError::EmptyHistory);
        }
        if window > self.history.len() {
            return Err(GradientFlowError::WindowExceedsHistory {
                window,
                len: self.history.len(),
            });
        }

        // Use group names from the latest snapshot as the canonical list.
        // `VecDeque` has no inherent `last()`; `back()` is the equivalent.
        let latest = self.history.back().ok_or(GradientFlowError::EmptyHistory)?;
        let group_names: Vec<String> = latest.groups.iter().map(|g| g.group_name.clone()).collect();

        let mut reports: Vec<GroupFlowReport> = Vec::with_capacity(group_names.len());
        for name in &group_names {
            let report = self.analyze_group(name, window)?;
            reports.push(report);
        }

        // Dominant = highest mean_norm; weakest = lowest mean_norm.
        let dominant_group = reports
            .iter()
            .max_by(|a, b| {
                a.mean_norm
                    .partial_cmp(&b.mean_norm)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.group_name.clone())
            .unwrap_or_default();

        let weakest_group = reports
            .iter()
            .min_by(|a, b| {
                a.mean_norm
                    .partial_cmp(&b.mean_norm)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|r| r.group_name.clone())
            .unwrap_or_default();

        let overall_health = reports
            .iter()
            .map(|r| r.health)
            .fold(FlowHealth::Healthy, worst_health);

        Ok(GradientFlowReport {
            step: latest.step,
            groups: reports,
            dominant_group,
            weakest_group,
            overall_health,
        })
    }

    /// Like [`analyze_group`](Self::analyze_group), but uses
    /// `config.trend_window` (clamped to the available history) as the
    /// window size instead of requiring the caller to pass one explicitly.
    pub fn analyze_group_default_window(
        &self,
        group_name: &str,
    ) -> Result<GroupFlowReport, GradientFlowError> {
        let window = self.config.trend_window.min(self.history.len());
        self.analyze_group(group_name, window)
    }

    /// Like [`analyze_all`](Self::analyze_all), but uses
    /// `config.trend_window` (clamped to the available history) as the
    /// window size instead of requiring the caller to pass one explicitly.
    pub fn analyze_all_default_window(&self) -> Result<GradientFlowReport, GradientFlowError> {
        let window = self.config.trend_window.min(self.history.len());
        self.analyze_all(window)
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Compute the sum of per-group mean L2 norms across the given window of snapshots.
    fn group_total_mean_norm_in_window(&self, window_snaps: &[&FlowSnapshot]) -> f32 {
        // Collect all unique group names from window.
        let mut group_names: Vec<String> = Vec::new();
        for snap in window_snaps {
            for g in &snap.groups {
                if !group_names.contains(&g.group_name) {
                    group_names.push(g.group_name.clone());
                }
            }
        }

        let mut total = 0.0_f32;
        for name in &group_names {
            let norms: Vec<f32> = window_snaps
                .iter()
                .flat_map(|snap| snap.groups.iter())
                .filter(|g| &g.group_name == name)
                .map(|g| g.l2_norm)
                .collect();
            if !norms.is_empty() {
                total += norms.iter().sum::<f32>() / norms.len() as f32;
            }
        }
        total
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Report formatting and utility functions
// ─────────────────────────────────────────────────────────────────────────────

/// Render a human-readable gradient flow report string.
///
/// Example output:
/// ```text
/// GradFlow [step 1000]: positions=Healthy(↔), opacities=Vanishing(↓), overall=Vanishing
/// ```
pub fn format_flow_report(report: &GradientFlowReport) -> String {
    let mut parts = Vec::with_capacity(report.groups.len());
    for g in &report.groups {
        let health_str = match g.health {
            FlowHealth::Healthy => "Healthy",
            FlowHealth::Vanishing => "Vanishing(!)",
            FlowHealth::Exploding => "Exploding(!)",
            FlowHealth::Dead => "Dead(!)",
            FlowHealth::NanOrInf => "NanOrInf(!)",
        };
        let trend_str = match g.trend {
            GradTrend::Stable => "↔",
            GradTrend::Increasing => "↑",
            GradTrend::Decreasing => "↓",
            GradTrend::Oscillating => "~",
        };
        parts.push(format!("{}={}({})", g.group_name, health_str, trend_str));
    }

    let overall_str = match report.overall_health {
        FlowHealth::Healthy => "Healthy",
        FlowHealth::Vanishing => "Vanishing",
        FlowHealth::Exploding => "Exploding",
        FlowHealth::Dead => "Dead",
        FlowHealth::NanOrInf => "NanOrInf",
    };

    format!(
        "GradFlow [step {}]: {}, overall={}",
        report.step,
        parts.join(", "),
        overall_str
    )
}

/// Return `(group_name, relative_signal)` pairs sorted descending by signal.
///
/// Useful for identifying which Gaussian parameter groups receive the most
/// gradient during training.
pub fn compare_group_signals(report: &GradientFlowReport) -> Vec<(String, f32)> {
    let mut pairs: Vec<(String, f32)> = report
        .groups
        .iter()
        .map(|g| (g.group_name.clone(), g.relative_signal))
        .collect();
    // Sort descending by relative_signal.
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // Helper: default config with tight thresholds for testing.
    fn test_config() -> GradientFlowConfig {
        GradientFlowConfig {
            history_capacity: 10,
            vanish_threshold: 1e-6,
            explode_threshold: 100.0,
            trend_window: 5,
            stable_cv: 0.1,
        }
    }

    // ─── flow_l2_norm ────────────────────────────────────────────────────────

    #[test]
    fn test_flow_l2_norm_empty() {
        assert_eq!(flow_l2_norm(&[]), 0.0);
    }

    #[test]
    fn test_flow_l2_norm_known() {
        // 3-4-5 triangle.
        let v = [3.0_f32, 4.0];
        let norm = flow_l2_norm(&v);
        assert!((norm - 5.0).abs() < 1e-5, "expected 5.0, got {norm}");
    }

    #[test]
    fn test_flow_l2_norm_ones() {
        let v = [1.0_f32; 4];
        let norm = flow_l2_norm(&v);
        assert!((norm - 2.0).abs() < 1e-5, "expected 2.0, got {norm}");
    }

    #[test]
    fn test_flow_l2_norm_negatives() {
        let v = [-3.0_f32, 4.0];
        let norm = flow_l2_norm(&v);
        assert!((norm - 5.0).abs() < 1e-5, "expected 5.0, got {norm}");
    }

    // ─── flow_l_inf_norm ─────────────────────────────────────────────────────

    #[test]
    fn test_flow_l_inf_norm_empty() {
        assert_eq!(flow_l_inf_norm(&[]), 0.0);
    }

    #[test]
    fn test_flow_l_inf_norm_known() {
        let v = [1.0_f32, -5.0, 3.0];
        assert!((flow_l_inf_norm(&v) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_flow_l_inf_norm_all_equal() {
        let v = [2.0_f32; 5];
        assert!((flow_l_inf_norm(&v) - 2.0).abs() < 1e-6);
    }

    // ─── flow_mean_abs ───────────────────────────────────────────────────────

    #[test]
    fn test_flow_mean_abs_empty() {
        assert_eq!(flow_mean_abs(&[]), 0.0);
    }

    #[test]
    fn test_flow_mean_abs_known() {
        // |1| + |-1| + |2| + |-2| = 6, / 4 = 1.5
        let v = [1.0_f32, -1.0, 2.0, -2.0];
        let ma = flow_mean_abs(&v);
        assert!((ma - 1.5).abs() < 1e-6, "expected 1.5, got {ma}");
    }

    #[test]
    fn test_flow_mean_abs_zeros() {
        let v = [0.0_f32; 10];
        assert_eq!(flow_mean_abs(&v), 0.0);
    }

    // ─── flow_std ────────────────────────────────────────────────────────────

    #[test]
    fn test_flow_std_empty() {
        assert_eq!(flow_std(&[]), 0.0);
    }

    #[test]
    fn test_flow_std_constant() {
        // All same → std = 0.
        let v = [3.0_f32; 10];
        let s = flow_std(&v);
        assert!(
            s.abs() < 1e-6,
            "constant values should yield std=0, got {s}"
        );
    }

    #[test]
    fn test_flow_std_known() {
        // Population std of [0, 2] = 1.0.
        let v = [0.0_f32, 2.0];
        let s = flow_std(&v);
        assert!((s - 1.0).abs() < 1e-5, "expected 1.0, got {s}");
    }

    #[test]
    fn test_flow_std_symmetric() {
        // [-1, 0, 1] → mean=0, E[x²]=2/3, std=sqrt(2/3).
        let v = [-1.0_f32, 0.0, 1.0];
        let s = flow_std(&v);
        let expected = (2.0_f32 / 3.0).sqrt();
        assert!((s - expected).abs() < 1e-5, "expected {expected}, got {s}");
    }

    // ─── compute_group_snapshot ──────────────────────────────────────────────

    #[test]
    fn test_compute_group_snapshot_empty_returns_error() {
        let result = compute_group_snapshot("positions", 1, &[]);
        assert!(matches!(
            result,
            Err(GradientFlowError::EmptyGradients { group }) if group == "positions"
        ));
    }

    #[test]
    fn test_compute_group_snapshot_basic_fields() {
        let grads = [3.0_f32, 4.0];
        let snap = compute_group_snapshot("positions", 42, &grads).unwrap();
        assert_eq!(snap.group_name, "positions");
        assert_eq!(snap.step, 42);
        assert!((snap.l2_norm - 5.0).abs() < 1e-5);
        assert!((snap.l_inf_norm - 4.0).abs() < 1e-5);
        assert!(!snap.has_nan);
        assert!(!snap.has_inf);
        assert_eq!(snap.n_params, 2);
    }

    #[test]
    fn test_compute_group_snapshot_nan_detection() {
        let grads = [1.0_f32, f32::NAN, 0.5];
        let snap = compute_group_snapshot("rotations", 0, &grads).unwrap();
        assert!(snap.has_nan);
        assert!(!snap.has_inf);
    }

    #[test]
    fn test_compute_group_snapshot_inf_detection() {
        let grads = [1.0_f32, f32::INFINITY];
        let snap = compute_group_snapshot("scales", 0, &grads).unwrap();
        assert!(!snap.has_nan);
        assert!(snap.has_inf);
    }

    #[test]
    fn test_compute_group_snapshot_neg_inf_detection() {
        let grads = [f32::NEG_INFINITY, 0.5];
        let snap = compute_group_snapshot("opacities", 0, &grads).unwrap();
        assert!(snap.has_inf);
    }

    // ─── classify_flow_health ────────────────────────────────────────────────

    #[test]
    fn test_classify_flow_health_nan() {
        let cfg = test_config();
        assert_eq!(
            classify_flow_health(0.5, true, false, &cfg),
            FlowHealth::NanOrInf
        );
    }

    #[test]
    fn test_classify_flow_health_inf() {
        let cfg = test_config();
        assert_eq!(
            classify_flow_health(0.5, false, true, &cfg),
            FlowHealth::NanOrInf
        );
    }

    #[test]
    fn test_classify_flow_health_dead() {
        let cfg = test_config();
        assert_eq!(
            classify_flow_health(0.0, false, false, &cfg),
            FlowHealth::Dead
        );
    }

    #[test]
    fn test_classify_flow_health_exploding() {
        let cfg = test_config();
        assert_eq!(
            classify_flow_health(200.0, false, false, &cfg),
            FlowHealth::Exploding
        );
    }

    #[test]
    fn test_classify_flow_health_vanishing() {
        let cfg = test_config();
        assert_eq!(
            classify_flow_health(1e-8, false, false, &cfg),
            FlowHealth::Vanishing
        );
    }

    #[test]
    fn test_classify_flow_health_healthy() {
        let cfg = test_config();
        assert_eq!(
            classify_flow_health(0.5, false, false, &cfg),
            FlowHealth::Healthy
        );
    }

    // ─── worst_health ────────────────────────────────────────────────────────

    #[test]
    fn test_worst_health_same() {
        assert_eq!(
            worst_health(FlowHealth::Healthy, FlowHealth::Healthy),
            FlowHealth::Healthy
        );
        assert_eq!(
            worst_health(FlowHealth::NanOrInf, FlowHealth::NanOrInf),
            FlowHealth::NanOrInf
        );
    }

    #[test]
    fn test_worst_health_nan_beats_all() {
        assert_eq!(
            worst_health(FlowHealth::NanOrInf, FlowHealth::Exploding),
            FlowHealth::NanOrInf
        );
        assert_eq!(
            worst_health(FlowHealth::Healthy, FlowHealth::NanOrInf),
            FlowHealth::NanOrInf
        );
    }

    #[test]
    fn test_worst_health_exploding_beats_vanishing() {
        assert_eq!(
            worst_health(FlowHealth::Exploding, FlowHealth::Vanishing),
            FlowHealth::Exploding
        );
    }

    #[test]
    fn test_worst_health_dead_beats_vanishing() {
        // Regression test: Dead (exactly-zero gradient, no learning signal
        // at all) must outrank Vanishing (merely small but nonzero), per
        // classify_flow_health's documented priority. worst_health's rank
        // previously disagreed and ranked Vanishing above Dead.
        assert_eq!(
            worst_health(FlowHealth::Dead, FlowHealth::Vanishing),
            FlowHealth::Dead
        );
        assert_eq!(
            worst_health(FlowHealth::Vanishing, FlowHealth::Dead),
            FlowHealth::Dead
        );
    }

    #[test]
    fn test_worst_health_dead_beats_exploding() {
        // Full documented ordering check: Dead also outranks Exploding.
        assert_eq!(
            worst_health(FlowHealth::Dead, FlowHealth::Exploding),
            FlowHealth::Dead
        );
    }

    #[test]
    fn test_worst_health_dead_beats_healthy() {
        assert_eq!(
            worst_health(FlowHealth::Dead, FlowHealth::Healthy),
            FlowHealth::Dead
        );
    }

    // ─── flow_linear_regression ──────────────────────────────────────────────

    #[test]
    fn test_flow_linear_regression_empty_returns_zero() {
        assert_eq!(flow_linear_regression(&[]), 0.0);
    }

    #[test]
    fn test_flow_linear_regression_single_returns_zero() {
        assert_eq!(flow_linear_regression(&[5.0]), 0.0);
    }

    #[test]
    fn test_flow_linear_regression_flat() {
        let v = [2.0_f32; 10];
        let slope = flow_linear_regression(&v);
        assert!(
            slope.abs() < 1e-5,
            "flat sequence should give slope≈0, got {slope}"
        );
    }

    #[test]
    fn test_flow_linear_regression_increasing() {
        // y = x → slope = 1.
        let v: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let slope = flow_linear_regression(&v);
        assert!(
            slope > 0.0,
            "monotone increase should give positive slope, got {slope}"
        );
    }

    #[test]
    fn test_flow_linear_regression_decreasing() {
        // y = -x → slope = -1.
        let v: Vec<f32> = (0..5).map(|i| -(i as f32)).collect();
        let slope = flow_linear_regression(&v);
        assert!(
            slope < 0.0,
            "monotone decrease should give negative slope, got {slope}"
        );
    }

    #[test]
    fn test_flow_linear_regression_known_slope() {
        // y = 2x + 1 → slope should be 2.
        let v: Vec<f32> = (0..5).map(|i| 2.0 * i as f32 + 1.0).collect();
        let slope = flow_linear_regression(&v);
        assert!(
            (slope - 2.0).abs() < 1e-4,
            "expected slope≈2.0, got {slope}"
        );
    }

    // ─── compute_grad_trend ──────────────────────────────────────────────────

    #[test]
    fn test_compute_grad_trend_empty() {
        let cfg = test_config();
        assert_eq!(compute_grad_trend(&[], &cfg), GradTrend::Stable);
    }

    #[test]
    fn test_compute_grad_trend_single() {
        let cfg = test_config();
        assert_eq!(compute_grad_trend(&[1.0], &cfg), GradTrend::Stable);
    }

    #[test]
    fn test_compute_grad_trend_stable() {
        let cfg = test_config();
        // Small variance relative to mean → Stable.
        let v = [1.0_f32, 1.001, 1.002, 1.001, 1.003];
        assert_eq!(compute_grad_trend(&v, &cfg), GradTrend::Stable);
    }

    #[test]
    fn test_compute_grad_trend_increasing() {
        let cfg = test_config();
        // Norms rising sharply.
        let v = [0.1_f32, 1.0, 2.0, 5.0, 10.0];
        assert_eq!(compute_grad_trend(&v, &cfg), GradTrend::Increasing);
    }

    #[test]
    fn test_compute_grad_trend_decreasing() {
        let cfg = test_config();
        // Norms falling sharply.
        let v = [10.0_f32, 5.0, 2.0, 0.5, 0.1];
        assert_eq!(compute_grad_trend(&v, &cfg), GradTrend::Decreasing);
    }

    // ─── GradientFlowTracker::new ────────────────────────────────────────────

    #[test]
    fn test_tracker_new_empty_history() {
        let tracker = GradientFlowTracker::new(test_config());
        assert_eq!(tracker.history_len(), 0);
        assert!(tracker.latest_snapshot().is_none());
    }

    // ─── GradientFlowTracker::record ─────────────────────────────────────────

    #[test]
    fn test_tracker_record_single_group() {
        let mut tracker = GradientFlowTracker::new(test_config());
        let grads = vec![3.0_f32, 4.0];
        let snap = tracker
            .record(1, vec![("positions".to_string(), grads.as_slice())])
            .unwrap();
        assert_eq!(snap.step, 1);
        assert_eq!(snap.groups.len(), 1);
        assert!((snap.groups[0].l2_norm - 5.0).abs() < 1e-5);
        assert!((snap.total_l2_norm - 5.0).abs() < 1e-5);
        assert_eq!(tracker.history_len(), 1);
    }

    #[test]
    fn test_tracker_record_multiple_groups() {
        let mut tracker = GradientFlowTracker::new(test_config());
        let pos = vec![3.0_f32, 4.0]; // norm=5
        let rot = vec![1.0_f32, 0.0, 0.0, 0.0]; // norm=1
        let snap = tracker
            .record(
                2,
                vec![
                    ("positions".to_string(), pos.as_slice()),
                    ("rotations".to_string(), rot.as_slice()),
                ],
            )
            .unwrap();
        assert_eq!(snap.groups.len(), 2);
        // total_l2_norm = sqrt(5² + 1²) = sqrt(26)
        let expected = (26.0_f32).sqrt();
        assert!(
            (snap.total_l2_norm - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            snap.total_l2_norm
        );
    }

    #[test]
    fn test_tracker_record_empty_group_returns_error() {
        let mut tracker = GradientFlowTracker::new(test_config());
        let result = tracker.record(1, vec![("positions".to_string(), &[] as &[f32])]);
        assert!(matches!(
            result,
            Err(GradientFlowError::EmptyGradients { .. })
        ));
    }

    #[test]
    fn test_tracker_record_updates_latest_snapshot() {
        let mut tracker = GradientFlowTracker::new(test_config());
        let grads = vec![1.0_f32];
        tracker
            .record(10, vec![("g".to_string(), grads.as_slice())])
            .unwrap();
        let latest = tracker.latest_snapshot().unwrap();
        assert_eq!(latest.step, 10);
    }

    // ─── History capacity eviction ────────────────────────────────────────────

    #[test]
    fn test_tracker_history_capacity_eviction() {
        let cfg = GradientFlowConfig {
            history_capacity: 3,
            ..test_config()
        };
        let mut tracker = GradientFlowTracker::new(cfg);
        let grads = vec![1.0_f32];
        for step in 0..5 {
            tracker
                .record(step, vec![("g".to_string(), grads.as_slice())])
                .unwrap();
        }
        // Should only retain last 3.
        assert_eq!(tracker.history_len(), 3);
        // Latest step should be 4.
        assert_eq!(tracker.latest_snapshot().unwrap().step, 4);
        // Oldest retained step should be 2.
        assert_eq!(tracker.history[0].step, 2);
    }

    // ─── latest_snapshot ─────────────────────────────────────────────────────

    #[test]
    fn test_latest_snapshot_none_when_empty() {
        let tracker = GradientFlowTracker::new(test_config());
        assert!(tracker.latest_snapshot().is_none());
    }

    #[test]
    fn test_latest_snapshot_some_after_record() {
        let mut tracker = GradientFlowTracker::new(test_config());
        let grads = vec![0.5_f32];
        tracker
            .record(99, vec![("x".to_string(), grads.as_slice())])
            .unwrap();
        assert!(tracker.latest_snapshot().is_some());
        assert_eq!(tracker.latest_snapshot().unwrap().step, 99);
    }

    // ─── analyze_group ───────────────────────────────────────────────────────

    #[test]
    fn test_analyze_group_empty_history_error() {
        let tracker = GradientFlowTracker::new(test_config());
        let result = tracker.analyze_group("positions", 1);
        assert!(matches!(result, Err(GradientFlowError::EmptyHistory)));
    }

    #[test]
    fn test_analyze_group_window_exceeds_history_error() {
        let mut tracker = GradientFlowTracker::new(test_config());
        let grads = vec![1.0_f32];
        tracker
            .record(1, vec![("g".to_string(), grads.as_slice())])
            .unwrap();
        let result = tracker.analyze_group("g", 5);
        assert!(matches!(
            result,
            Err(GradientFlowError::WindowExceedsHistory { window: 5, len: 1 })
        ));
    }

    #[test]
    fn test_analyze_group_unknown_group_error() {
        let mut tracker = GradientFlowTracker::new(test_config());
        let grads = vec![1.0_f32];
        tracker
            .record(1, vec![("positions".to_string(), grads.as_slice())])
            .unwrap();
        let result = tracker.analyze_group("opacities", 1);
        assert!(matches!(
            result,
            Err(GradientFlowError::UnknownGroup { name }) if name == "opacities"
        ));
    }

    #[test]
    fn test_analyze_group_healthy_case() {
        let mut tracker = GradientFlowTracker::new(test_config());
        // norm = 5, healthy range.
        let grads = vec![3.0_f32, 4.0];
        tracker
            .record(1, vec![("positions".to_string(), grads.as_slice())])
            .unwrap();
        let report = tracker.analyze_group("positions", 1).unwrap();
        assert_eq!(report.health, FlowHealth::Healthy);
        assert!((report.mean_norm - 5.0).abs() < 1e-4);
        assert!((report.peak_norm - 5.0).abs() < 1e-4);
        assert!((report.min_norm - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_analyze_group_vanishing_case() {
        let cfg = GradientFlowConfig {
            vanish_threshold: 1e-3,
            ..test_config()
        };
        let mut tracker = GradientFlowTracker::new(cfg);
        // norm = 1e-6 < 1e-3 → Vanishing.
        let tiny = vec![1e-6_f32];
        tracker
            .record(1, vec![("opacities".to_string(), tiny.as_slice())])
            .unwrap();
        let report = tracker.analyze_group("opacities", 1).unwrap();
        assert_eq!(report.health, FlowHealth::Vanishing);
    }

    #[test]
    fn test_analyze_group_relative_signal_one_group() {
        let mut tracker = GradientFlowTracker::new(test_config());
        let grads = vec![1.0_f32];
        tracker
            .record(1, vec![("g".to_string(), grads.as_slice())])
            .unwrap();
        let report = tracker.analyze_group("g", 1).unwrap();
        // Only group → relative_signal should be 1.0.
        assert!((report.relative_signal - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_analyze_group_relative_signal_two_groups() {
        let mut tracker = GradientFlowTracker::new(test_config());
        // positions norm=4 (exactly), opacities norm=0 (dead).
        let _pos = [4.0_f32];
        let _ops = [0.0_f32]; // norm=0 → Dead, but should still work here
                              // We'll use non-zero for opacities to avoid Dead classification.
        let ops = vec![0.0_f32];
        let _ = ops; // suppress unused warning
        let pos2 = vec![3.0_f32, 4.0]; // norm=5
        let ops2 = vec![1.0_f32]; // norm=1
        tracker
            .record(
                1,
                vec![
                    ("positions".to_string(), pos2.as_slice()),
                    ("opacities".to_string(), ops2.as_slice()),
                ],
            )
            .unwrap();
        let rep_pos = tracker.analyze_group("positions", 1).unwrap();
        let rep_ops = tracker.analyze_group("opacities", 1).unwrap();
        // total = 5 + 1 = 6; positions signal = 5/6, opacities = 1/6.
        assert!((rep_pos.relative_signal - 5.0 / 6.0).abs() < 1e-4);
        assert!((rep_ops.relative_signal - 1.0 / 6.0).abs() < 1e-4);
    }

    // ─── analyze_all ─────────────────────────────────────────────────────────

    #[test]
    fn test_analyze_all_empty_history_error() {
        let tracker = GradientFlowTracker::new(test_config());
        assert!(matches!(
            tracker.analyze_all(1),
            Err(GradientFlowError::EmptyHistory)
        ));
    }

    #[test]
    fn test_analyze_all_dominant_weakest() {
        let mut tracker = GradientFlowTracker::new(test_config());
        let pos = vec![3.0_f32, 4.0]; // norm=5
        let rot = vec![0.1_f32]; // norm=0.1
        tracker
            .record(
                1,
                vec![
                    ("positions".to_string(), pos.as_slice()),
                    ("rotations".to_string(), rot.as_slice()),
                ],
            )
            .unwrap();
        let report = tracker.analyze_all(1).unwrap();
        assert_eq!(report.dominant_group, "positions");
        assert_eq!(report.weakest_group, "rotations");
    }

    #[test]
    fn test_analyze_all_overall_health_worst() {
        let cfg = GradientFlowConfig {
            vanish_threshold: 1e-3,
            ..test_config()
        };
        let mut tracker = GradientFlowTracker::new(cfg);
        let pos = vec![3.0_f32, 4.0]; // Healthy
        let ops = vec![1e-6_f32]; // Vanishing
        tracker
            .record(
                1,
                vec![
                    ("positions".to_string(), pos.as_slice()),
                    ("opacities".to_string(), ops.as_slice()),
                ],
            )
            .unwrap();
        let report = tracker.analyze_all(1).unwrap();
        assert_eq!(report.overall_health, FlowHealth::Vanishing);
    }

    #[test]
    fn test_analyze_all_step_matches_latest() {
        let mut tracker = GradientFlowTracker::new(test_config());
        let grads = vec![1.0_f32];
        tracker
            .record(77, vec![("g".to_string(), grads.as_slice())])
            .unwrap();
        let report = tracker.analyze_all(1).unwrap();
        assert_eq!(report.step, 77);
    }

    // ─── format_flow_report ──────────────────────────────────────────────────

    #[test]
    fn test_format_flow_report_non_empty() {
        let mut tracker = GradientFlowTracker::new(test_config());
        let grads = vec![1.0_f32];
        tracker
            .record(5, vec![("g".to_string(), grads.as_slice())])
            .unwrap();
        let report = tracker.analyze_all(1).unwrap();
        let s = format_flow_report(&report);
        assert!(!s.is_empty());
        assert!(s.contains("GradFlow [step 5]"));
    }

    #[test]
    fn test_format_flow_report_contains_group_name() {
        let mut tracker = GradientFlowTracker::new(test_config());
        let grads = vec![0.5_f32];
        tracker
            .record(1, vec![("positions".to_string(), grads.as_slice())])
            .unwrap();
        let report = tracker.analyze_all(1).unwrap();
        let s = format_flow_report(&report);
        assert!(s.contains("positions"), "report: {s}");
    }

    #[test]
    fn test_format_flow_report_contains_overall() {
        let mut tracker = GradientFlowTracker::new(test_config());
        let grads = vec![1.0_f32];
        tracker
            .record(1, vec![("g".to_string(), grads.as_slice())])
            .unwrap();
        let report = tracker.analyze_all(1).unwrap();
        let s = format_flow_report(&report);
        assert!(
            s.contains("overall="),
            "report should contain 'overall=': {s}"
        );
    }

    // ─── compare_group_signals ───────────────────────────────────────────────

    #[test]
    fn test_compare_group_signals_sorted_descending() {
        let mut tracker = GradientFlowTracker::new(test_config());
        let pos = vec![3.0_f32, 4.0]; // norm=5
        let rot = vec![1.0_f32]; // norm=1
        tracker
            .record(
                1,
                vec![
                    ("positions".to_string(), pos.as_slice()),
                    ("rotations".to_string(), rot.as_slice()),
                ],
            )
            .unwrap();
        let report = tracker.analyze_all(1).unwrap();
        let signals = compare_group_signals(&report);
        assert_eq!(signals.len(), 2);
        assert_eq!(signals[0].0, "positions");
        assert_eq!(signals[1].0, "rotations");
        assert!(signals[0].1 >= signals[1].1);
    }

    #[test]
    fn test_compare_group_signals_single_group() {
        let mut tracker = GradientFlowTracker::new(test_config());
        let grads = vec![1.0_f32];
        tracker
            .record(1, vec![("only".to_string(), grads.as_slice())])
            .unwrap();
        let report = tracker.analyze_all(1).unwrap();
        let signals = compare_group_signals(&report);
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].0, "only");
    }

    // ─── GradientFlowError variants ──────────────────────────────────────────

    #[test]
    fn test_error_empty_gradients_display() {
        let e = GradientFlowError::EmptyGradients {
            group: "sh_dc".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("sh_dc"), "display: {s}");
    }

    #[test]
    fn test_error_empty_history_display() {
        let e = GradientFlowError::EmptyHistory;
        let s = e.to_string();
        assert!(!s.is_empty());
    }

    #[test]
    fn test_error_window_exceeds_history_display() {
        let e = GradientFlowError::WindowExceedsHistory { window: 10, len: 3 };
        let s = e.to_string();
        assert!(s.contains("10") && s.contains("3"), "display: {s}");
    }

    #[test]
    fn test_error_unknown_group_display() {
        let e = GradientFlowError::UnknownGroup {
            name: "sh_rest".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("sh_rest"), "display: {s}");
    }

    // ─── GradTrend variants ───────────────────────────────────────────────────

    #[test]
    fn test_grad_trend_oscillating() {
        let cfg = GradientFlowConfig {
            stable_cv: 0.01, // tight threshold
            ..test_config()
        };
        // Alternating values → high cv, slope exactly 0 (symmetric) → Oscillating.
        let v = [1.0_f32, 10.0, 1.0, 10.0, 1.0];
        let trend = compute_grad_trend(&v, &cfg);
        assert_eq!(trend, GradTrend::Oscillating, "trend={trend:?}");
    }

    #[test]
    fn test_grad_trend_oscillating_with_nonzero_slope() {
        // Regression test: Oscillating must not require a bit-exact 0.0
        // slope. This sequence is nearly (but not exactly) symmetric, so
        // `flow_linear_regression` returns a small NONZERO slope — with
        // real float data, an exactly-zero slope essentially never
        // happens, so the old `slope == 0.0` check made this variant
        // unreachable in practice.
        let cfg = GradientFlowConfig {
            stable_cv: 0.01, // tight threshold: high cv still triggers here
            ..test_config()
        };
        let v = [1.0_f32, 10.0, 1.0, 10.0, 1.01];
        let slope = flow_linear_regression(&v);
        assert_ne!(slope, 0.0, "sanity: slope should be nonzero for this input");
        let trend = compute_grad_trend(&v, &cfg);
        assert_eq!(trend, GradTrend::Oscillating, "trend={trend:?}");
    }

    #[test]
    fn test_grad_trend_zero_mean_stable() {
        let cfg = test_config();
        // All zero norms → zero mean → cv treated as 0 → Stable.
        let v = [0.0_f32; 5];
        assert_eq!(compute_grad_trend(&v, &cfg), GradTrend::Stable);
    }

    // ─── Multi-step window analysis ───────────────────────────────────────────

    #[test]
    fn test_analyze_group_multi_step_window() {
        let cfg = GradientFlowConfig {
            history_capacity: 100,
            trend_window: 5,
            ..GradientFlowConfig::default()
        };
        let mut tracker = GradientFlowTracker::new(cfg);
        // Record 10 steps with varying norms.
        for step in 0..10usize {
            let norm_val = (step as f32 + 1.0) * 0.5;
            let grads = vec![norm_val];
            tracker
                .record(step, vec![("positions".to_string(), grads.as_slice())])
                .unwrap();
        }
        // Analyze last 5 steps.
        let report = tracker.analyze_group("positions", 5).unwrap();
        assert!(report.peak_norm > report.min_norm);
        assert!(report.mean_norm > 0.0);
    }

    #[test]
    fn test_analyze_all_multi_group_multi_step() {
        let mut tracker = GradientFlowTracker::new(GradientFlowConfig {
            history_capacity: 20,
            ..GradientFlowConfig::default()
        });
        for step in 0..5usize {
            let pos = vec![(step as f32 + 1.0) * 2.0];
            let ops = vec![(step as f32 + 1.0) * 0.001];
            tracker
                .record(
                    step,
                    vec![
                        ("positions".to_string(), pos.as_slice()),
                        ("opacities".to_string(), ops.as_slice()),
                    ],
                )
                .unwrap();
        }
        let report = tracker.analyze_all(3).unwrap();
        assert_eq!(report.groups.len(), 2);
        // positions should dominate because its norms are much larger.
        assert_eq!(report.dominant_group, "positions");
    }

    // ─── trend_window wiring (analyze_*_default_window) ─────────────────────

    #[test]
    fn test_analyze_group_default_window_uses_config_trend_window() {
        let cfg = GradientFlowConfig {
            history_capacity: 100,
            trend_window: 3,
            ..GradientFlowConfig::default()
        };
        let mut tracker = GradientFlowTracker::new(cfg);
        for step in 0..10usize {
            let grads = vec![(step as f32 + 1.0) * 0.5];
            tracker
                .record(step, vec![("positions".to_string(), grads.as_slice())])
                .unwrap();
        }
        let default_report = tracker
            .analyze_group_default_window("positions")
            .expect("trend_window=3 should be usable directly");
        let explicit_report = tracker.analyze_group("positions", 3).unwrap();
        assert!((default_report.mean_norm - explicit_report.mean_norm).abs() < 1e-6);
    }

    #[test]
    fn test_analyze_group_default_window_clamps_to_history_len() {
        // trend_window (50, the default) exceeds the 2 recorded steps —
        // the default-window helper must clamp instead of erroring.
        let mut tracker = GradientFlowTracker::new(GradientFlowConfig::default());
        let grads = vec![1.0_f32];
        tracker
            .record(0, vec![("g".to_string(), grads.as_slice())])
            .unwrap();
        tracker
            .record(1, vec![("g".to_string(), grads.as_slice())])
            .unwrap();
        let report = tracker
            .analyze_group_default_window("g")
            .expect("should clamp window to history_len instead of erroring");
        assert!((report.mean_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_analyze_all_default_window_matches_explicit_window() {
        let cfg = GradientFlowConfig {
            history_capacity: 100,
            trend_window: 4,
            ..GradientFlowConfig::default()
        };
        let mut tracker = GradientFlowTracker::new(cfg);
        for step in 0..10usize {
            let pos = vec![(step as f32 + 1.0) * 2.0];
            tracker
                .record(step, vec![("positions".to_string(), pos.as_slice())])
                .unwrap();
        }
        let default_report = tracker.analyze_all_default_window().unwrap();
        let explicit_report = tracker.analyze_all(4).unwrap();
        assert_eq!(default_report.step, explicit_report.step);
        assert!(
            (default_report.groups[0].mean_norm - explicit_report.groups[0].mean_norm).abs() < 1e-6
        );
    }
}
