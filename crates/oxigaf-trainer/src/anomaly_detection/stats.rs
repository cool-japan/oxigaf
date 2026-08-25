//! Small statistics helpers shared by the anomaly checks.

use super::AnomalyDetectionError;

// ─────────────────────────────────────────────────────────────────────────────
// Statistics utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the mean and standard deviation of a slice.
/// Returns `(0.0, 0.0)` for an empty slice.
pub fn anom_mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f32;
    let mean = values.iter().copied().sum::<f32>() / n;
    let variance = values
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    (mean, variance.sqrt())
}

/// Count NaN and Inf values in a slice.
/// Returns `(n_nan, n_inf)`.
pub fn anom_count_nonfinite(values: &[f32]) -> (usize, usize) {
    let mut n_nan = 0usize;
    let mut n_inf = 0usize;
    for &v in values {
        if v.is_nan() {
            n_nan += 1;
        } else if v.is_infinite() {
            n_inf += 1;
        }
    }
    (n_nan, n_inf)
}

/// Check whether the last `n` values in a slice form a strictly monotone increasing sequence.
/// Returns `false` if `n == 0` or `n > values.len()`.
pub fn anom_is_monotone_increasing(values: &[f32], n: usize) -> bool {
    if n == 0 || n > values.len() {
        return false;
    }
    let tail = &values[values.len() - n..];
    tail.windows(2).all(|w| w[1] > w[0])
}

/// Fraction of the last `n` values' step-to-step intervals that are increases.
///
/// A window of `n` values has `n - 1` intervals, so the result is
/// `#{i : tail[i+1] > tail[i]} / (n - 1)` — `1.0` for a strictly increasing
/// tail, `0.0` for a non-increasing one. Unlike
/// [`anom_is_monotone_increasing`] a single dip only lowers the fraction
/// instead of rejecting the whole window, which is what makes it usable on a
/// noisy stochastic loss curve.
///
/// Returns `0.0` when `n < 2`, when `n > values.len()`, or when the tail
/// contains a non-finite value (NaN comparisons are never `>`, and NaN/Inf
/// have their own dedicated checks).
pub fn anom_increase_fraction(values: &[f32], n: usize) -> f32 {
    if n < 2 || n > values.len() {
        return 0.0;
    }
    let tail = &values[values.len() - n..];
    if tail.iter().any(|v| !v.is_finite()) {
        return 0.0;
    }
    let increases = tail.windows(2).filter(|w| w[1] > w[0]).count();
    increases as f32 / (n - 1) as f32
}

/// Least-squares slope of the last `n` values against their index, normalised
/// by the window's mean magnitude.
///
/// The result reads as *relative growth per step*: `0.01` means the values
/// climb by about 1 % of their own current magnitude every step, which is
/// scale-free and therefore comparable across loss terms of wildly different
/// size. Positive means rising, negative means falling.
///
/// Returns `0.0` when `n < 2`, when `n > values.len()`, when the tail contains
/// a non-finite value, or when the window's mean magnitude is effectively zero
/// (no scale to be relative to).
pub fn anom_relative_trend(values: &[f32], n: usize) -> f32 {
    if n < 2 || n > values.len() {
        return 0.0;
    }
    let tail = &values[values.len() - n..];
    if tail.iter().any(|v| !v.is_finite()) {
        return 0.0;
    }
    let n_f = n as f32;
    let mean_x = (n_f - 1.0) / 2.0;
    let mean_y = tail.iter().sum::<f32>() / n_f;
    let mut num = 0.0_f32;
    let mut den = 0.0_f32;
    for (i, &y) in tail.iter().enumerate() {
        let dx = i as f32 - mean_x;
        num += dx * (y - mean_y);
        den += dx * dx;
    }
    if den <= 0.0 {
        return 0.0;
    }
    let scale = tail.iter().map(|v| v.abs()).sum::<f32>() / n_f;
    if scale <= 1e-12 {
        return 0.0;
    }
    (num / den) / scale
}

/// Compute the L2 norm of a slice. Returns 0.0 for empty input.
pub fn anom_l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Compute the maximum absolute value in a slice. Returns 0.0 for empty input.
pub fn anom_max_abs(values: &[f32]) -> f32 {
    values.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()))
}

/// Compute the maximum per-element L2 distance between two slices of length N×3.
/// Returns an error if the lengths differ.
pub fn anom_max_pairwise_dist(a: &[f32], b: &[f32]) -> Result<f32, AnomalyDetectionError> {
    if a.len() != b.len() {
        return Err(AnomalyDetectionError::InvalidConfig(format!(
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Ok(0.0);
    }
    // Treat as N×3 vectors; compute per-point distance.
    // If not divisible by 3, fall back to element-wise comparison.
    let stride = if a.len().is_multiple_of(3) { 3 } else { 1 };
    let n_points = a.len() / stride;
    let mut max_dist = 0.0f32;
    for i in 0..n_points {
        let mut dist_sq = 0.0f32;
        for j in 0..stride {
            let d = a[i * stride + j] - b[i * stride + j];
            dist_sq += d * d;
        }
        let dist = dist_sq.sqrt();
        if dist > max_dist {
            max_dist = dist;
        }
    }
    Ok(max_dist)
}
