//! Statistical analysis and histogram tools for 3D Gaussian Splatting models.
//!
//! This module provides:
//! - `ScalarStats`: summary statistics for a slice of f32 values
//! - `Histogram`: uniform-bin histogram with ASCII rendering
//! - `GaussianStats`: full per-field statistics for a `GaussianModel`
//! - `GaussianHistograms`: histogram set for a `GaussianModel`
//! - Free functions: `format_gaussian_report`, `detect_anomalies`

use crate::{GaussianModel, RenderError};

// ─── ScalarStats ────────────────────────────────────────────────────────────

/// Summary statistics for a slice of f32 values.
///
/// NaN and Inf values are counted but excluded from numeric aggregates.
#[derive(Debug, Clone)]
pub struct ScalarStats {
    /// Total number of input values.
    pub count: usize,
    /// Minimum of finite values.
    pub min: f32,
    /// Maximum of finite values.
    pub max: f32,
    /// Arithmetic mean of finite values.
    pub mean: f32,
    /// Variance of finite values.
    pub variance: f32,
    /// Standard deviation of finite values.
    pub std_dev: f32,
    /// Approximate median (sorted copy of finite values, middle element).
    pub median: f32,
    /// 5th percentile of finite values.
    pub p5: f32,
    /// 95th percentile of finite values.
    pub p95: f32,
    /// Number of NaN values in the input.
    pub nan_count: usize,
    /// Number of infinite values in the input.
    pub inf_count: usize,
}

impl Default for ScalarStats {
    fn default() -> Self {
        ScalarStats {
            count: 0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            variance: 0.0,
            std_dev: 0.0,
            median: 0.0,
            p5: 0.0,
            p95: 0.0,
            nan_count: 0,
            inf_count: 0,
        }
    }
}

/// Compute the linearly-interpolated percentile `p` (0..=100) of `slice`
/// using `select_nth_unstable_by` for the two bracketing order statistics
/// instead of a full sort — O(len) rather than O(len log len).
///
/// `slice` is partially reordered by this call (its element *set* is
/// unchanged). Panics only if `slice` is empty (callers must guard that).
fn percentile_via_select(slice: &mut [f32], p: f64) -> f32 {
    let n = slice.len();
    if n == 1 {
        return slice[0];
    }
    let idx = p / 100.0 * (n - 1) as f64;
    let lo = (idx.floor() as usize).min(n - 1);
    let frac = (idx - lo as f64) as f32;

    // `select_nth_unstable_by(lo)` partitions `slice` so that `slice[lo]`
    // holds the value that would be at sorted position `lo`, everything
    // before it is <=, and everything in `right` is >=. The *minimum* of
    // `right` is therefore exactly the value at sorted position `lo + 1`
    // (or equal to `v_lo` when `lo` is already the last index).
    let (_, &mut v_lo, right) = slice.select_nth_unstable_by(lo, f32::total_cmp);
    if frac <= 0.0 || right.is_empty() {
        return v_lo;
    }
    let v_hi = right.iter().copied().fold(f32::INFINITY, f32::min);
    v_lo + frac * (v_hi - v_lo)
}

/// Compute summary statistics for a slice of f32 values.
///
/// Empty input or all-NaN/Inf input returns `ScalarStats::default()` with
/// appropriate nan_count/inf_count set.
pub fn compute_scalar_stats(values: &[f32]) -> ScalarStats {
    let count = values.len();
    if count == 0 {
        return ScalarStats::default();
    }

    let mut nan_count = 0usize;
    let mut inf_count = 0usize;

    // Separate finite values and count special values
    let mut finite: Vec<f32> = Vec::with_capacity(count);
    for &v in values {
        if v.is_nan() {
            nan_count += 1;
        } else if v.is_infinite() {
            inf_count += 1;
        } else {
            finite.push(v);
        }
    }

    if finite.is_empty() {
        return ScalarStats {
            count,
            nan_count,
            inf_count,
            ..ScalarStats::default()
        };
    }

    // min / max
    let mut min = finite[0];
    let mut max = finite[0];
    let mut sum = 0.0_f64;
    for &v in &finite {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        sum += v as f64;
    }

    let n = finite.len();
    let mean = (sum / n as f64) as f32;

    // Variance (two-pass for numerical stability)
    let mut var_sum = 0.0_f64;
    for &v in &finite {
        let diff = (v as f64) - (mean as f64);
        var_sum += diff * diff;
    }
    let variance = (var_sum / n as f64) as f32;
    let std_dev = variance.sqrt();

    // Order statistics via `select_nth_unstable_by` (average O(len) per
    // call) instead of a full O(n log n) sort — `GaussianStats::compute`
    // calls this for nine fields, so for a multi-million-Gaussian model
    // avoiding the sort (and its dedicated allocation-free quickselect
    // partitioning) meaningfully cuts wall time. `select_nth_unstable_by`
    // is safe to call repeatedly on the same slice: each call produces a
    // correct result regardless of any partial ordering left by a
    // previous call.
    let median = percentile_via_select(&mut finite, 50.0);
    let p5 = percentile_via_select(&mut finite, 5.0);
    let p95 = percentile_via_select(&mut finite, 95.0);

    ScalarStats {
        count,
        min,
        max,
        mean,
        variance,
        std_dev,
        median,
        p5,
        p95,
        nan_count,
        inf_count,
    }
}

// ─── Histogram ──────────────────────────────────────────────────────────────

/// A uniform-bin histogram over a closed range [min_val, max_val].
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Number of bins.
    pub num_bins: usize,
    /// Inclusive lower bound.
    pub min_val: f32,
    /// Inclusive upper bound.
    pub max_val: f32,
    /// Per-bin counts.
    pub counts: Vec<u64>,
    /// Width of each bin.
    pub bin_width: f32,
}

impl Histogram {
    /// Create a new empty histogram with a fixed range.
    ///
    /// # Errors
    /// Returns `RenderError::Rasterize` if `num_bins == 0` or `min_val >= max_val`.
    pub fn new(num_bins: usize, min_val: f32, max_val: f32) -> Result<Self, RenderError> {
        if num_bins == 0 {
            return Err(RenderError::Rasterize(
                "Histogram: num_bins must be > 0".to_string(),
            ));
        }
        if min_val >= max_val {
            return Err(RenderError::Rasterize(format!(
                "Histogram: min_val ({min_val}) must be < max_val ({max_val})"
            )));
        }
        let bin_width = (max_val - min_val) / num_bins as f32;
        Ok(Histogram {
            num_bins,
            min_val,
            max_val,
            counts: vec![0u64; num_bins],
            bin_width,
        })
    }

    /// Build a histogram from a slice of values, auto-ranging on finite min/max.
    ///
    /// NaN and Inf values are ignored. If all values are NaN/Inf, or the input
    /// is empty, returns a zero-count histogram over [0.0, 1.0].
    pub fn compute(values: &[f32], num_bins: usize) -> Self {
        let num_bins = if num_bins == 0 { 1 } else { num_bins };

        // Find finite min/max
        let mut fmin = f32::INFINITY;
        let mut fmax = f32::NEG_INFINITY;
        for &v in values {
            if v.is_finite() {
                if v < fmin {
                    fmin = v;
                }
                if v > fmax {
                    fmax = v;
                }
            }
        }

        // Fallback range if no finite values
        if !fmin.is_finite() || !fmax.is_finite() || fmin >= fmax {
            // Degenerate case: single value or no finite values
            let min_val = if fmin.is_finite() { fmin } else { 0.0 };
            let max_val = min_val + 1.0;
            let bin_width = (max_val - min_val) / num_bins as f32;
            let mut hist = Histogram {
                num_bins,
                min_val,
                max_val,
                counts: vec![0u64; num_bins],
                bin_width,
            };
            // Insert the single value if there is one
            for &v in values {
                if v.is_finite() {
                    hist.insert(v);
                }
            }
            return hist;
        }

        let bin_width = (fmax - fmin) / num_bins as f32;
        let mut hist = Histogram {
            num_bins,
            min_val: fmin,
            max_val: fmax,
            counts: vec![0u64; num_bins],
            bin_width,
        };

        for &v in values {
            hist.insert(v);
        }
        hist
    }

    /// Return the bin index for a value, or `None` if out of range or non-finite.
    pub fn bin_index(&self, value: f32) -> Option<usize> {
        if !value.is_finite() {
            return None;
        }
        if value < self.min_val || value > self.max_val {
            return None;
        }
        // Clamp exactly-max into last bin
        if value >= self.max_val {
            return Some(self.num_bins - 1);
        }
        let idx = ((value - self.min_val) / self.bin_width) as usize;
        // Guard against floating-point overshoot
        Some(idx.min(self.num_bins - 1))
    }

    /// Insert a single value into the histogram.
    ///
    /// Values outside [min_val, max_val] or non-finite are silently ignored.
    pub fn insert(&mut self, value: f32) {
        if let Some(idx) = self.bin_index(value) {
            self.counts[idx] = self.counts[idx].saturating_add(1);
        }
    }

    /// Return the index of the bin with the highest count.
    ///
    /// Returns 0 if all counts are zero or the histogram is empty.
    pub fn peak_bin(&self) -> usize {
        self.counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Render the histogram as an ASCII bar chart.
    ///
    /// Each line has the form:
    /// `[min_val, max_val): ████████ (count)`
    ///
    /// `width` controls the maximum bar width in characters.
    pub fn format_ascii(&self, width: usize) -> String {
        let width = if width == 0 { 1 } else { width };
        let max_count = self.counts.iter().copied().max().unwrap_or(0);

        let mut out = String::new();
        for i in 0..self.num_bins {
            let lo = self.min_val + i as f32 * self.bin_width;
            let hi = lo + self.bin_width;
            let count = self.counts[i];

            let bar_len = if max_count > 0 {
                (count as f64 / max_count as f64 * width as f64).round() as usize
            } else {
                0
            };

            // Build bar using block characters
            let bar: String = "█".repeat(bar_len);
            out.push_str(&format!("[{lo:.4}, {hi:.4}): {bar} ({count})\n"));
        }
        out
    }
}

// ─── GaussianStats ──────────────────────────────────────────────────────────

/// Full statistics for a `GaussianModel`, one `ScalarStats` per parameter.
#[derive(Debug, Clone)]
pub struct GaussianStats {
    /// Total number of Gaussians.
    pub num_gaussians: usize,
    /// X positions.
    pub position_x: ScalarStats,
    /// Y positions.
    pub position_y: ScalarStats,
    /// Z positions.
    pub position_z: ScalarStats,
    /// Actual (exp) scale averaged across x/y/z dimensions.
    pub scale_mean: ScalarStats,
    /// Actual (exp) scale along X.
    pub scale_x: ScalarStats,
    /// Actual (exp) scale along Y.
    pub scale_y: ScalarStats,
    /// Actual (exp) scale along Z.
    pub scale_z: ScalarStats,
    /// Actual (sigmoid) opacity.
    pub opacity: ScalarStats,
    /// SH DC energy: `sqrt(dc_r² + dc_g² + dc_b²)` per Gaussian.
    pub sh_energy: ScalarStats,
    /// Total memory occupied by all model arrays in bytes.
    pub memory_bytes: usize,
    /// Number of rigid Gaussians.
    pub rigid_count: usize,
    /// Number of flexible Gaussians.
    pub flexible_count: usize,
    /// Exact count of Gaussians whose actual (sigmoid) opacity is > 0.99.
    /// Backs the "High opacity" check in [`detect_anomalies`] with a real
    /// count instead of a p95-based guess.
    pub opacity_above_099_count: usize,
    /// Exact count of Gaussians with actual scale < 0.001 on *at least one*
    /// axis (not just the mean of the three axes, which can hide a
    /// single-axis degenerate Gaussian). Backs the "Degenerate scale"
    /// check in [`detect_anomalies`].
    pub degenerate_scale_count: usize,
}

/// Per-Gaussian scalar fields extracted from a `GaussianModel` in a single
/// pass, shared by [`GaussianStats::compute`] and
/// [`GaussianHistograms::compute`] (and [`compute_stats_and_histograms`])
/// so a caller who needs both never re-derives the same fields twice.
struct ExtractedFields {
    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    pos_z: Vec<f32>,
    scale_x: Vec<f32>,
    scale_y: Vec<f32>,
    scale_z: Vec<f32>,
    scale_mean: Vec<f32>,
    opacity: Vec<f32>,
    sh_energy: Vec<f32>,
    opacity_above_099_count: usize,
    degenerate_scale_count: usize,
}

/// Extract [`ExtractedFields`] from `model` in one pass over
/// `model.gaussians` (plus the SH DC lookup, which shares the same index).
fn extract_fields(model: &GaussianModel) -> ExtractedFields {
    let n = model.gaussians.len();

    let mut pos_x = Vec::with_capacity(n);
    let mut pos_y = Vec::with_capacity(n);
    let mut pos_z = Vec::with_capacity(n);
    let mut scale_x = Vec::with_capacity(n);
    let mut scale_y = Vec::with_capacity(n);
    let mut scale_z = Vec::with_capacity(n);
    let mut scale_mean = Vec::with_capacity(n);
    let mut opacity = Vec::with_capacity(n);
    let mut sh_energy = Vec::with_capacity(n);
    let mut opacity_above_099_count = 0usize;
    let mut degenerate_scale_count = 0usize;

    let sh_total = ((model.sh_degree + 1) * (model.sh_degree + 1) * 3) as usize;
    let sh_valid = sh_total >= 3 && model.sh_coeffs.len() == n * sh_total;

    for (i, g) in model.gaussians.iter().enumerate() {
        pos_x.push(g.position[0]);
        pos_y.push(g.position[1]);
        pos_z.push(g.position[2]);

        // exp(stored) → actual scale
        let esx = g.scale[0].exp();
        let esy = g.scale[1].exp();
        let esz = g.scale[2].exp();
        scale_x.push(esx);
        scale_y.push(esy);
        scale_z.push(esz);
        scale_mean.push((esx + esy + esz) / 3.0);
        if esx.min(esy).min(esz) < 0.001 {
            degenerate_scale_count += 1;
        }

        // sigmoid(stored) → actual opacity
        let op = sigmoid(g.opacity);
        opacity.push(op);
        if op > 0.99 {
            opacity_above_099_count += 1;
        }

        // SH DC energy: sqrt(dc_r² + dc_g² + dc_b²)
        if sh_valid {
            let base = i * sh_total;
            let r = model.sh_coeffs[base];
            let gc = model.sh_coeffs[base + 1];
            let b = model.sh_coeffs[base + 2];
            sh_energy.push((r * r + gc * gc + b * b).sqrt());
        } else {
            sh_energy.push(0.0_f32);
        }
    }

    ExtractedFields {
        pos_x,
        pos_y,
        pos_z,
        scale_x,
        scale_y,
        scale_z,
        scale_mean,
        opacity,
        sh_energy,
        opacity_above_099_count,
        degenerate_scale_count,
    }
}

impl GaussianStats {
    /// Compute statistics from a `GaussianModel`.
    pub fn compute(model: &GaussianModel) -> Self {
        let fields = extract_fields(model);
        Self::from_fields(model, &fields)
    }

    fn from_fields(model: &GaussianModel, fields: &ExtractedFields) -> Self {
        let n = model.gaussians.len();
        let rigid_count = model.is_rigid.iter().filter(|&&r| r).count();
        let flexible_count = n - rigid_count;

        // Memory calculation
        let memory_bytes = std::mem::size_of::<crate::GaussianAttributes>() * n  // 48 bytes each
            + std::mem::size_of::<f32>() * model.sh_coeffs.len()
            + std::mem::size_of::<u32>() * model.face_indices.len()
            + std::mem::size_of::<[f32; 3]>() * model.barycentric.len()
            + std::mem::size_of::<[f32; 3]>() * model.local_offsets.len()
            + std::mem::size_of::<bool>() * model.is_rigid.len();

        GaussianStats {
            num_gaussians: n,
            position_x: compute_scalar_stats(&fields.pos_x),
            position_y: compute_scalar_stats(&fields.pos_y),
            position_z: compute_scalar_stats(&fields.pos_z),
            scale_mean: compute_scalar_stats(&fields.scale_mean),
            scale_x: compute_scalar_stats(&fields.scale_x),
            scale_y: compute_scalar_stats(&fields.scale_y),
            scale_z: compute_scalar_stats(&fields.scale_z),
            opacity: compute_scalar_stats(&fields.opacity),
            sh_energy: compute_scalar_stats(&fields.sh_energy),
            memory_bytes,
            rigid_count,
            flexible_count,
            opacity_above_099_count: fields.opacity_above_099_count,
            degenerate_scale_count: fields.degenerate_scale_count,
        }
    }
}

/// Logistic sigmoid function.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ─── GaussianHistograms ─────────────────────────────────────────────────────

/// A set of histograms, one per key parameter of a `GaussianModel`.
#[derive(Debug, Clone)]
pub struct GaussianHistograms {
    /// X position histogram.
    pub position_x: Histogram,
    /// Y position histogram.
    pub position_y: Histogram,
    /// Z position histogram.
    pub position_z: Histogram,
    /// Mean (across x/y/z) actual scale histogram.
    pub scale_mean: Histogram,
    /// Actual opacity histogram.
    pub opacity: Histogram,
    /// SH DC energy histogram.
    pub sh_energy: Histogram,
}

impl GaussianHistograms {
    /// Compute histograms from a `GaussianModel`.
    ///
    /// `num_bins` controls the number of bins; use 50 as a default.
    pub fn compute(model: &GaussianModel, num_bins: usize) -> Self {
        let fields = extract_fields(model);
        Self::from_fields(&fields, num_bins)
    }

    fn from_fields(fields: &ExtractedFields, num_bins: usize) -> Self {
        let num_bins = if num_bins == 0 { 50 } else { num_bins };

        GaussianHistograms {
            position_x: Histogram::compute(&fields.pos_x, num_bins),
            position_y: Histogram::compute(&fields.pos_y, num_bins),
            position_z: Histogram::compute(&fields.pos_z, num_bins),
            scale_mean: Histogram::compute(&fields.scale_mean, num_bins),
            opacity: Histogram::compute(&fields.opacity, num_bins),
            sh_energy: Histogram::compute(&fields.sh_energy, num_bins),
        }
    }

    /// Format all histograms as a text report.
    ///
    /// Each section has a title followed by a 40-character-wide ASCII bar chart.
    pub fn format_report(&self) -> String {
        let width = 40;
        let mut out = String::new();

        let sections: &[(&str, &Histogram)] = &[
            ("Position X", &self.position_x),
            ("Position Y", &self.position_y),
            ("Position Z", &self.position_z),
            ("Scale Mean (actual)", &self.scale_mean),
            ("Opacity (actual)", &self.opacity),
            ("SH DC Energy", &self.sh_energy),
        ];

        for (title, hist) in sections {
            out.push_str(&format!("\n=== {title} ===\n"));
            out.push_str(&hist.format_ascii(width));
        }
        out
    }
}

/// Compute both [`GaussianStats`] and [`GaussianHistograms`] from a single
/// extraction pass over `model`.
///
/// Calling [`GaussianStats::compute`] and [`GaussianHistograms::compute`]
/// separately each re-derive the same per-Gaussian scalar fields from
/// scratch; when both are needed for the same model (e.g. a full report),
/// this halves that field-derivation work.
pub fn compute_stats_and_histograms(
    model: &GaussianModel,
    num_bins: usize,
) -> (GaussianStats, GaussianHistograms) {
    let fields = extract_fields(model);
    let stats = GaussianStats::from_fields(model, &fields);
    let histograms = GaussianHistograms::from_fields(&fields, num_bins);
    (stats, histograms)
}

// ─── Report formatting ───────────────────────────────────────────────────────

/// Format a `GaussianStats` as a human-readable table.
///
/// Columns: Parameter | Count | Min | Max | Mean | Std | P5 | P95
pub fn format_gaussian_report(stats: &GaussianStats) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "Gaussian Model Report — {} Gaussians\n",
        stats.num_gaussians
    ));
    out.push_str(&format!(
        "Memory: {:.2} MB  Rigid: {}  Flexible: {}\n\n",
        stats.memory_bytes as f64 / (1024.0 * 1024.0),
        stats.rigid_count,
        stats.flexible_count
    ));

    // Header
    out.push_str(&format!(
        "{:<22} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}\n",
        "Parameter", "Count", "Min", "Max", "Mean", "Std", "P5", "P95"
    ));
    out.push_str(&"-".repeat(92));
    out.push('\n');

    let rows: &[(&str, &ScalarStats)] = &[
        ("Position X", &stats.position_x),
        ("Position Y", &stats.position_y),
        ("Position Z", &stats.position_z),
        ("Scale Mean (actual)", &stats.scale_mean),
        ("Scale X (actual)", &stats.scale_x),
        ("Scale Y (actual)", &stats.scale_y),
        ("Scale Z (actual)", &stats.scale_z),
        ("Opacity (actual)", &stats.opacity),
        ("SH DC Energy", &stats.sh_energy),
    ];

    for (name, s) in rows {
        out.push_str(&format!(
            "{:<22} {:>8} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}\n",
            name, s.count, s.min, s.max, s.mean, s.std_dev, s.p5, s.p95,
        ));
        if s.nan_count > 0 || s.inf_count > 0 {
            out.push_str(&format!(
                "{:<22}   NaN: {}  Inf: {}\n",
                "", s.nan_count, s.inf_count
            ));
        }
    }

    out
}

/// Detect anomalies in a `GaussianModel` and return warning strings.
///
/// An empty return value indicates no anomalies were found.
///
/// Checks (all exact counts from `stats`, not heuristics):
/// - High opacity: >10% of Gaussians have actual opacity > 0.99
/// - Exploding scale: max actual scale > 5.0
/// - Degenerate scale: >0% of Gaussians have actual scale < 0.001 on at
///   least one axis
/// - NaN in positions
pub fn detect_anomalies(stats: &GaussianStats) -> Vec<String> {
    let mut warnings = Vec::new();
    let n = stats.num_gaussians;

    if n == 0 {
        return warnings;
    }

    // NaN in positions
    let total_pos_nan =
        stats.position_x.nan_count + stats.position_y.nan_count + stats.position_z.nan_count;
    if total_pos_nan > 0 {
        warnings.push(format!("NaN in positions: {total_pos_nan} NaN values"));
    }

    // Exploding scale: max actual scale > 5.0
    let max_scale = stats
        .scale_x
        .max
        .max(stats.scale_y.max)
        .max(stats.scale_z.max);
    if max_scale > 5.0 {
        warnings.push(format!("Exploding scale: max scale {max_scale:.4} > 5.0"));
    }

    // High opacity: >10% of Gaussians have opacity > 0.99. `stats` now
    // carries the exact count (computed alongside the rest of the fields
    // in `GaussianStats::compute`), so this is a real measurement rather
    // than a p95-based guess.
    let opacity_pct = stats.opacity_above_099_count as f64 / n as f64 * 100.0;
    if opacity_pct > 10.0 {
        warnings.push(format!(
            "High opacity: {opacity_pct:.1}% of Gaussians have opacity > 0.99 ({} of {n})",
            stats.opacity_above_099_count
        ));
    }

    // Degenerate scale: >0% of Gaussians have actual scale < 0.001 on at
    // least one axis. Checked per-axis (not on the mean of the three
    // axes), so a Gaussian degenerate along only one axis is still caught.
    if stats.degenerate_scale_count > 0 {
        let degenerate_pct = stats.degenerate_scale_count as f64 / n as f64 * 100.0;
        warnings.push(format!(
            "Degenerate scale: {degenerate_pct:.1}% of Gaussians have scale < 0.001 on at least one axis ({} of {n})",
            stats.degenerate_scale_count
        ));
    }

    warnings
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::GaussianAttributes;

    // ── Helper ──────────────────────────────────────────────────────────────

    fn make_model(n: usize, sh_degree: u32) -> GaussianModel {
        let sh_total = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
        let gaussians: Vec<GaussianAttributes> = (0..n)
            .map(|i| {
                let fi = i as f32;
                GaussianAttributes {
                    position: [fi * 0.1, fi * 0.2, fi * 0.3],
                    _pad0: 0.0,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [fi * 0.1 - 2.0, fi * 0.1 - 1.5, fi * 0.1 - 1.0],
                    opacity: fi * 0.2 - 1.0, // logit space
                }
            })
            .collect();
        let sh_coeffs: Vec<f32> = (0..n * sh_total).map(|i| (i as f32) * 0.01).collect();
        let face_indices = vec![0u32; n];
        let barycentric = vec![[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]; n];
        let local_offsets = vec![[0.0_f32; 3]; n];
        let is_rigid = (0..n).map(|i| i % 2 == 0).collect();
        GaussianModel {
            gaussians,
            sh_coeffs,
            sh_degree,
            face_indices,
            barycentric,
            local_offsets,
            is_rigid,
        }
    }

    // ── ScalarStats tests ────────────────────────────────────────────────────

    #[test]
    fn test_scalar_stats_empty() {
        let s = compute_scalar_stats(&[]);
        assert_eq!(s.count, 0);
        assert_eq!(s.nan_count, 0);
        assert_eq!(s.inf_count, 0);
        assert_eq!(s.mean, 0.0);
    }

    #[test]
    fn test_scalar_stats_single_value() {
        let s = compute_scalar_stats(&[42.0_f32]);
        assert_eq!(s.count, 1);
        assert_eq!(s.min, 42.0);
        assert_eq!(s.max, 42.0);
        assert_eq!(s.mean, 42.0);
        assert_eq!(s.median, 42.0);
        assert_eq!(s.p5, 42.0);
        assert_eq!(s.p95, 42.0);
        assert_eq!(s.variance, 0.0);
        assert_eq!(s.std_dev, 0.0);
    }

    #[test]
    fn test_scalar_stats_basic() {
        let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let s = compute_scalar_stats(&values);
        assert_eq!(s.count, 5);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 5.0);
        assert!((s.mean - 3.0).abs() < 1e-5, "mean={}", s.mean);
        assert!((s.median - 3.0).abs() < 1e-5, "median={}", s.median);
        assert!(s.variance > 0.0);
        assert!(s.std_dev > 0.0);
        assert_eq!(s.nan_count, 0);
        assert_eq!(s.inf_count, 0);
    }

    #[test]
    fn test_scalar_stats_nan_counting() {
        let values = [1.0_f32, f32::NAN, 3.0, f32::NAN, 5.0];
        let s = compute_scalar_stats(&values);
        assert_eq!(s.count, 5);
        assert_eq!(s.nan_count, 2);
        assert_eq!(s.inf_count, 0);
        // Only finite values in stats
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 5.0);
        assert!((s.mean - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_scalar_stats_inf_counting() {
        let values = [1.0_f32, f32::INFINITY, 3.0, f32::NEG_INFINITY, 5.0];
        let s = compute_scalar_stats(&values);
        assert_eq!(s.count, 5);
        assert_eq!(s.inf_count, 2);
        assert_eq!(s.nan_count, 0);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 5.0);
    }

    #[test]
    fn test_scalar_stats_percentiles() {
        // 100 evenly spaced values 1..=100
        let values: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let s = compute_scalar_stats(&values);
        // p5 should be around 5.95, p95 around 95.05
        assert!(s.p5 > 1.0 && s.p5 < 10.0, "p5={}", s.p5);
        assert!(s.p95 > 90.0 && s.p95 <= 100.0, "p95={}", s.p95);
        assert!((s.median - 50.5).abs() < 1.0, "median={}", s.median);
    }

    // ── Histogram tests ──────────────────────────────────────────────────────

    #[test]
    fn test_histogram_new() {
        let h = Histogram::new(10, 0.0, 1.0).expect("valid histogram");
        assert_eq!(h.num_bins, 10);
        assert_eq!(h.counts.len(), 10);
        assert!((h.bin_width - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_histogram_new_invalid() {
        assert!(
            Histogram::new(0, 0.0, 1.0).is_err(),
            "num_bins=0 should fail"
        );
        assert!(
            Histogram::new(10, 1.0, 0.0).is_err(),
            "min >= max should fail"
        );
        assert!(
            Histogram::new(10, 1.0, 1.0).is_err(),
            "min == max should fail"
        );
    }

    #[test]
    fn test_histogram_compute_uniform() {
        // 100 values uniformly in [0, 1)
        let values: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let h = Histogram::compute(&values, 10);
        assert_eq!(h.num_bins, 10);
        // Each bin should have ~10 counts
        for &c in &h.counts {
            assert!((8..=12).contains(&c), "count={c}");
        }
    }

    #[test]
    fn test_histogram_insert() {
        let mut h = Histogram::new(5, 0.0, 5.0).expect("valid");
        h.insert(0.5);
        h.insert(1.5);
        h.insert(2.5);
        h.insert(3.5);
        h.insert(4.5);
        for &c in &h.counts {
            assert_eq!(c, 1);
        }
        // Out of range — should be ignored
        h.insert(-1.0);
        h.insert(6.0);
        h.insert(f32::NAN);
        h.insert(f32::INFINITY);
        let total: u64 = h.counts.iter().sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn test_histogram_peak_bin() {
        let mut h = Histogram::new(5, 0.0, 5.0).expect("valid");
        // Insert many into bin 2 ([2,3))
        for _ in 0..10 {
            h.insert(2.5);
        }
        h.insert(0.5);
        assert_eq!(h.peak_bin(), 2);
    }

    #[test]
    fn test_histogram_format_ascii() {
        let mut h = Histogram::new(3, 0.0, 3.0).expect("valid");
        h.insert(0.5);
        h.insert(0.5);
        h.insert(1.5);
        let out = h.format_ascii(20);
        assert!(out.contains("(2)"), "expected count 2, got: {out}");
        assert!(out.contains("(1)"), "expected count 1, got: {out}");
        assert!(out.contains("(0)"), "expected count 0, got: {out}");
        // Check line structure
        assert!(out.contains("[0.0000,"), "missing opening bracket: {out}");
    }

    #[test]
    fn test_histogram_bin_index() {
        let h = Histogram::new(4, 0.0, 4.0).expect("valid");
        assert_eq!(h.bin_index(0.0), Some(0));
        assert_eq!(h.bin_index(1.0), Some(1));
        assert_eq!(h.bin_index(3.9), Some(3));
        assert_eq!(h.bin_index(4.0), Some(3)); // max goes to last bin
        assert_eq!(h.bin_index(-0.1), None); // below range
        assert_eq!(h.bin_index(4.1), None); // above range
        assert_eq!(h.bin_index(f32::NAN), None);
        assert_eq!(h.bin_index(f32::INFINITY), None);
    }

    // ── GaussianStats tests ──────────────────────────────────────────────────

    #[test]
    fn test_gaussian_stats_compute() {
        let model = make_model(10, 1);
        let stats = GaussianStats::compute(&model);
        assert_eq!(stats.num_gaussians, 10);
        // Positions should be in [0, 0.9] for x, [0, 1.8] for y, etc.
        assert!(
            stats.position_x.min >= 0.0,
            "x min: {}",
            stats.position_x.min
        );
        assert!(
            stats.position_x.max <= 1.0,
            "x max: {}",
            stats.position_x.max
        );
        // Actual scale = exp(stored log-scale), should be positive
        assert!(stats.scale_x.min > 0.0, "scale_x min should be positive");
        // Actual opacity = sigmoid(...) in (0,1)
        assert!(stats.opacity.min > 0.0 && stats.opacity.max < 1.0);
        // SH energy should be non-negative
        assert!(stats.sh_energy.min >= 0.0);
    }

    #[test]
    fn test_gaussian_stats_memory_bytes() {
        let n = 5;
        let sh_degree = 1u32;
        let model = make_model(n, sh_degree);
        let stats = GaussianStats::compute(&model);
        // GaussianAttributes = 48 bytes (3 pos + pad + 4 rot + 3 scale + opacity = 11 f32 + 1 pad = 48)
        let sh_total = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
        let expected = 48 * n
            + 4 * n * sh_total    // sh_coeffs
            + 4 * n               // face_indices u32
            + 12 * n              // barycentric [f32;3]
            + 12 * n              // local_offsets [f32;3]
            + n; // is_rigid bool
        assert_eq!(stats.memory_bytes, expected, "memory_bytes mismatch");
    }

    #[test]
    fn test_gaussian_histograms_compute() {
        let model = make_model(20, 1);
        let histograms = GaussianHistograms::compute(&model, 10);
        assert_eq!(histograms.position_x.num_bins, 10);
        assert_eq!(histograms.opacity.num_bins, 10);
        // Total counts in opacity histogram should equal n
        let total: u64 = histograms.opacity.counts.iter().sum();
        assert_eq!(total, 20, "total opacity counts={total}");
    }

    #[test]
    fn test_format_gaussian_report() {
        let model = make_model(10, 1);
        let stats = GaussianStats::compute(&model);
        let report = format_gaussian_report(&stats);
        assert!(report.contains("Gaussian Model Report"), "report: {report}");
        assert!(report.contains("Position X"), "report: {report}");
        assert!(report.contains("Opacity"), "report: {report}");
        assert!(report.contains("Memory"), "report: {report}");
    }

    #[test]
    fn test_detect_anomalies_healthy() {
        // Model with normal parameters → no anomalies
        let n = 10;
        let sh_degree = 1u32;
        let sh_total = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
        let gaussians: Vec<GaussianAttributes> = (0..n)
            .map(|i| GaussianAttributes {
                position: [i as f32 * 0.01, 0.0, 0.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                // log-scale around -1 → exp(-1) ≈ 0.37, well under 5.0
                scale: [-1.0, -1.0, -1.0],
                // logit(-2) → sigmoid(-2) ≈ 0.12, well under 0.99
                opacity: -2.0,
            })
            .collect();
        let model = GaussianModel {
            gaussians,
            sh_coeffs: vec![0.5_f32; n * sh_total],
            sh_degree,
            face_indices: vec![0u32; n],
            barycentric: vec![[1.0 / 3.0; 3]; n],
            local_offsets: vec![[0.0; 3]; n],
            is_rigid: vec![false; n],
        };
        let stats = GaussianStats::compute(&model);
        let anomalies = detect_anomalies(&stats);
        assert!(
            anomalies.is_empty(),
            "Expected no anomalies, got: {anomalies:?}"
        );
    }

    #[test]
    fn test_detect_anomalies_high_opacity() {
        let n = 10;
        let sh_degree = 0u32;
        let sh_total = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
        // High logit → sigmoid ≈ 1.0 for all Gaussians
        let gaussians: Vec<GaussianAttributes> = (0..n)
            .map(|_| GaussianAttributes {
                position: [0.0, 0.0, 0.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [-1.0, -1.0, -1.0],
                opacity: 10.0, // sigmoid(10) ≈ 0.9999
            })
            .collect();
        let model = GaussianModel {
            gaussians,
            sh_coeffs: vec![0.0_f32; n * sh_total],
            sh_degree,
            face_indices: vec![0u32; n],
            barycentric: vec![[1.0 / 3.0; 3]; n],
            local_offsets: vec![[0.0; 3]; n],
            is_rigid: vec![false; n],
        };
        let stats = GaussianStats::compute(&model);
        let anomalies = detect_anomalies(&stats);
        assert!(
            anomalies.iter().any(|w| w.contains("opacity")),
            "Expected high opacity warning, got: {anomalies:?}"
        );
    }

    #[test]
    fn test_detect_anomalies_exploding_scale() {
        let n = 5;
        let sh_degree = 0u32;
        let sh_total = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
        // log-scale = 3.0 → exp(3.0) ≈ 20.09 >> 5.0
        let gaussians: Vec<GaussianAttributes> = (0..n)
            .map(|_| GaussianAttributes {
                position: [0.0, 0.0, 0.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [3.0, 3.0, 3.0],
                opacity: 0.0,
            })
            .collect();
        let model = GaussianModel {
            gaussians,
            sh_coeffs: vec![0.0_f32; n * sh_total],
            sh_degree,
            face_indices: vec![0u32; n],
            barycentric: vec![[1.0 / 3.0; 3]; n],
            local_offsets: vec![[0.0; 3]; n],
            is_rigid: vec![false; n],
        };
        let stats = GaussianStats::compute(&model);
        let anomalies = detect_anomalies(&stats);
        assert!(
            anomalies.iter().any(|w| w.contains("scale")),
            "Expected exploding scale warning, got: {anomalies:?}"
        );
    }

    // ── detect_anomalies: exact-count regression tests ───────────────────────

    #[test]
    fn test_detect_anomalies_high_opacity_exact_count_above_threshold() {
        // Exactly 3 of 20 (15%) Gaussians have opacity > 0.99 → must trigger
        // (>10%). The old p95-based heuristic could not distinguish this
        // from many other fractions; `opacity_above_099_count` now gives
        // the exact figure.
        let n = 20;
        let sh_degree = 0u32;
        let sh_total = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
        let gaussians: Vec<GaussianAttributes> = (0..n)
            .map(|i| GaussianAttributes {
                position: [0.0, 0.0, 0.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [-1.0, -1.0, -1.0],
                opacity: if i < 3 { 10.0 } else { -2.0 },
            })
            .collect();
        let model = GaussianModel {
            gaussians,
            sh_coeffs: vec![0.0_f32; n * sh_total],
            sh_degree,
            face_indices: vec![0u32; n],
            barycentric: vec![[1.0 / 3.0; 3]; n],
            local_offsets: vec![[0.0; 3]; n],
            is_rigid: vec![false; n],
        };
        let stats = GaussianStats::compute(&model);
        assert_eq!(stats.opacity_above_099_count, 3);
        let anomalies = detect_anomalies(&stats);
        assert!(
            anomalies
                .iter()
                .any(|w| w.contains("opacity") && w.contains("15.0%")),
            "Expected a 15% high-opacity warning, got: {anomalies:?}"
        );
    }

    #[test]
    fn test_detect_anomalies_high_opacity_at_threshold_no_warning() {
        // Exactly 2 of 20 (10%) Gaussians have opacity > 0.99 → must NOT
        // trigger, since the documented check is a strict ">10%".
        let n = 20;
        let sh_degree = 0u32;
        let sh_total = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
        let gaussians: Vec<GaussianAttributes> = (0..n)
            .map(|i| GaussianAttributes {
                position: [0.0, 0.0, 0.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [-1.0, -1.0, -1.0],
                opacity: if i < 2 { 10.0 } else { -2.0 },
            })
            .collect();
        let model = GaussianModel {
            gaussians,
            sh_coeffs: vec![0.0_f32; n * sh_total],
            sh_degree,
            face_indices: vec![0u32; n],
            barycentric: vec![[1.0 / 3.0; 3]; n],
            local_offsets: vec![[0.0; 3]; n],
            is_rigid: vec![false; n],
        };
        let stats = GaussianStats::compute(&model);
        assert_eq!(stats.opacity_above_099_count, 2);
        let anomalies = detect_anomalies(&stats);
        assert!(
            !anomalies.iter().any(|w| w.contains("opacity")),
            "Expected no high-opacity warning at exactly 10%, got: {anomalies:?}"
        );
    }

    #[test]
    fn test_detect_anomalies_degenerate_scale_single_axis_caught() {
        // Only the Z axis is degenerate (< 0.001); the *mean* of the three
        // axes is not, so the old mean-based check would have missed this
        // entirely.
        let n = 4;
        let sh_degree = 0u32;
        let sh_total = ((sh_degree + 1) * (sh_degree + 1) * 3) as usize;
        let gaussians: Vec<GaussianAttributes> = (0..n)
            .map(|_| GaussianAttributes {
                position: [0.0, 0.0, 0.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                // exp(-1) ≈ 0.368, exp(-1) ≈ 0.368, exp(-10) ≈ 0.0000454 (< 0.001)
                scale: [-1.0, -1.0, -10.0],
                opacity: 0.0,
            })
            .collect();
        let model = GaussianModel {
            gaussians,
            sh_coeffs: vec![0.0_f32; n * sh_total],
            sh_degree,
            face_indices: vec![0u32; n],
            barycentric: vec![[1.0 / 3.0; 3]; n],
            local_offsets: vec![[0.0; 3]; n],
            is_rigid: vec![false; n],
        };
        let stats = GaussianStats::compute(&model);
        // The mean-based check would NOT have flagged this: mean ≈ 0.245.
        assert!(stats.scale_mean.min > 0.001);
        // The per-axis check correctly flags all 4 Gaussians.
        assert_eq!(stats.degenerate_scale_count, 4);
        let anomalies = detect_anomalies(&stats);
        assert!(
            anomalies.iter().any(|w| w.contains("Degenerate")),
            "Expected a degenerate-scale warning, got: {anomalies:?}"
        );
    }

    // ── percentile_via_select: parity with the old full-sort computation ─────

    #[test]
    fn test_percentile_via_select_matches_full_sort() {
        // 137 pseudo-random-ish values (not evenly spaced, with
        // duplicates) — enough to exercise `select_nth_unstable_by` across
        // several distinct ranks and confirm it agrees with a plain sort.
        // `wrapping_mul` avoids an overflow panic in debug builds (the
        // product legitimately exceeds u32::MAX for i > ~1.6).
        let values: Vec<f32> = (0..137u32)
            .map(|i| ((i.wrapping_mul(2654435761)) % 10007) as f32 * 0.01)
            .collect();

        let mut sorted = values.clone();
        sorted.sort_by(f32::total_cmp);
        let expected_percentile = |p: f64| -> f32 {
            let n = sorted.len();
            let idx = p / 100.0 * (n - 1) as f64;
            let lo = idx.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            let frac = (idx - lo as f64) as f32;
            sorted[lo] + frac * (sorted[hi] - sorted[lo])
        };

        let stats = compute_scalar_stats(&values);
        for (name, p, expected) in [
            ("median", 50.0, expected_percentile(50.0)),
            ("p5", 5.0, expected_percentile(5.0)),
            ("p95", 95.0, expected_percentile(95.0)),
        ] {
            let actual = match name {
                "median" => stats.median,
                "p5" => stats.p5,
                _ => stats.p95,
            };
            assert!(
                (actual - expected).abs() < 1e-3,
                "{name} (p={p}) mismatch: select-based={actual}, full-sort={expected}"
            );
        }
    }

    // ── compute_stats_and_histograms: single-pass combined computation ───────

    #[test]
    fn test_compute_stats_and_histograms_matches_separate_calls() {
        let model = make_model(30, 1);
        let (stats, histograms) = compute_stats_and_histograms(&model, 10);

        let stats_alone = GaussianStats::compute(&model);
        let histograms_alone = GaussianHistograms::compute(&model, 10);

        assert_eq!(stats.num_gaussians, stats_alone.num_gaussians);
        assert_eq!(
            stats.opacity_above_099_count,
            stats_alone.opacity_above_099_count
        );
        assert_eq!(
            stats.degenerate_scale_count,
            stats_alone.degenerate_scale_count
        );
        assert!((stats.position_x.mean - stats_alone.position_x.mean).abs() < 1e-6);

        assert_eq!(
            histograms.position_x.num_bins,
            histograms_alone.position_x.num_bins
        );
        assert_eq!(
            histograms.opacity.counts, histograms_alone.opacity.counts,
            "histogram bin counts should match the separately-computed version"
        );
    }
}
