//! Maximum Mean Discrepancy: Gaussian kernel, biased/unbiased estimators,
//! the multi-scale sum and the median-bandwidth heuristic.

use super::batch::DomainBatch;
use super::common::{da_check_matrix, DomainAdaptationError};

// ---------------------------------------------------------------------------
// MMD configuration
// ---------------------------------------------------------------------------

/// Configuration for Maximum Mean Discrepancy computation.
pub struct MmdConfig {
    /// Kernel bandwidths for multi-scale MMD, e.g. `[0.1, 1.0, 10.0]`.
    pub kernel_bandwidths: Vec<f32>,
    /// If `true`, use biased estimator (includes i=j diagonal terms).
    pub biased: bool,
    /// Smallest kernel bandwidth this configuration admits: below it the
    /// Gaussian kernel's `exp(-‖x-y‖² / 2σ²)` degenerates (`0/0` for
    /// coincident points, underflow for every other pair), so
    /// [`da_mmd_multiscale`] rejects such a bandwidth up front and
    /// [`MmdConfig::from_median_heuristic`] floors its output at this value.
    pub eps: f32,
}

impl Default for MmdConfig {
    fn default() -> Self {
        Self {
            kernel_bandwidths: vec![0.1, 1.0, 10.0],
            biased: false,
            eps: 1e-8,
        }
    }
}

/// Multipliers applied to the median-heuristic bandwidth by
/// [`MmdConfig::from_median_heuristic`].
const MEDIAN_HEURISTIC_SCALES: [f32; 5] = [0.25, 0.5, 1.0, 2.0, 4.0];

impl MmdConfig {
    /// Check that the configuration can produce a well-defined kernel.
    ///
    /// # Errors
    /// - [`DomainAdaptationError::InvalidConfig`] when `kernel_bandwidths` is
    ///   empty or `eps` is not a positive finite number.
    /// - [`DomainAdaptationError::InvalidBandwidth`] when any bandwidth is
    ///   non-finite or smaller than [`eps`](Self::eps).
    pub fn validate(&self) -> Result<(), DomainAdaptationError> {
        if self.kernel_bandwidths.is_empty() {
            return Err(DomainAdaptationError::InvalidConfig {
                reason: "kernel_bandwidths must not be empty".to_owned(),
            });
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(DomainAdaptationError::InvalidConfig {
                reason: format!("eps must be a positive finite value, got {}", self.eps),
            });
        }
        for &bw in &self.kernel_bandwidths {
            if !bw.is_finite() || bw < self.eps {
                return Err(DomainAdaptationError::InvalidBandwidth { bw });
            }
        }
        Ok(())
    }

    /// Build a multi-scale configuration around the median pairwise distance
    /// of `features` (an `n × d` row-major matrix).
    ///
    /// The median heuristic ([`da_median_bandwidth`]) sets the middle scale;
    /// the surrounding scales are that value times
    /// `[0.25, 0.5, 1, 2, 4]`, each floored at [`eps`](Self::eps) so a
    /// degenerate feature set (all rows identical, median ≈ 0) still yields a
    /// usable kernel instead of a division that collapses.
    pub fn from_median_heuristic(features: &[f32], n: usize, d: usize) -> Self {
        let defaults = Self::default();
        let median = da_median_bandwidth(features, n, d);
        let kernel_bandwidths = MEDIAN_HEURISTIC_SCALES
            .iter()
            .map(|&s| (median * s).max(defaults.eps))
            .collect();
        Self {
            kernel_bandwidths,
            ..defaults
        }
    }
}

// ---------------------------------------------------------------------------
// MMD: Gaussian kernel
// ---------------------------------------------------------------------------

/// Compute the Gaussian (RBF) kernel between two feature vectors of length `d`.
///
/// `k(x, y) = exp(-||x - y||² / (2σ²))`
pub fn da_gaussian_kernel(x: &[f32], y: &[f32], d: usize, bandwidth: f32) -> f32 {
    debug_assert_eq!(x.len(), d);
    debug_assert_eq!(y.len(), d);
    let sq_dist: f32 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| {
            let diff = xi - yi;
            diff * diff
        })
        .sum();
    let denom = 2.0 * bandwidth * bandwidth;
    (-sq_dist / denom).exp()
}

// ---------------------------------------------------------------------------
// MMD: biased estimator
// ---------------------------------------------------------------------------

/// Compute biased MMD² between source and target distributions.
///
/// `MMD²(P,Q) = E[k(x,x')] - 2·E[k(x,y)] + E[k(y,y')]`
///
/// All n_s² (resp. n_t²) pairs are included (diagonal i=i is included).
pub fn da_mmd_biased(
    source: &[f32],
    target: &[f32],
    n_s: usize,
    n_t: usize,
    d: usize,
    bandwidth: f32,
) -> Result<f32, DomainAdaptationError> {
    if bandwidth <= 0.0 || !bandwidth.is_finite() {
        return Err(DomainAdaptationError::InvalidBandwidth { bw: bandwidth });
    }
    if n_s == 0 || n_t == 0 {
        return Err(DomainAdaptationError::EmptyFeatures);
    }
    da_check_matrix(source, n_s, d)?;
    da_check_matrix(target, n_t, d)?;

    // E[k(x,x')]
    let mut kss = 0.0f32;
    for i in 0..n_s {
        for j in 0..n_s {
            kss += da_gaussian_kernel(
                &source[i * d..(i + 1) * d],
                &source[j * d..(j + 1) * d],
                d,
                bandwidth,
            );
        }
    }
    kss /= (n_s * n_s) as f32;

    // E[k(y,y')]
    let mut ktt = 0.0f32;
    for i in 0..n_t {
        for j in 0..n_t {
            ktt += da_gaussian_kernel(
                &target[i * d..(i + 1) * d],
                &target[j * d..(j + 1) * d],
                d,
                bandwidth,
            );
        }
    }
    ktt /= (n_t * n_t) as f32;

    // 2·E[k(x,y)]
    let mut kst = 0.0f32;
    for i in 0..n_s {
        for j in 0..n_t {
            kst += da_gaussian_kernel(
                &source[i * d..(i + 1) * d],
                &target[j * d..(j + 1) * d],
                d,
                bandwidth,
            );
        }
    }
    kst /= (n_s * n_t) as f32;

    Ok((kss - 2.0 * kst + ktt).max(0.0))
}

// ---------------------------------------------------------------------------
// MMD: unbiased estimator
// ---------------------------------------------------------------------------

/// Compute unbiased MMD² between source and target distributions.
///
/// Diagonal terms (i=j) are excluded for source–source and target–target sums,
/// so each estimator is unbiased.  When `n_s < 2` or `n_t < 2`, the
/// within-distribution terms are defined as 0 (no pairs to average over).
pub fn da_mmd_unbiased(
    source: &[f32],
    target: &[f32],
    n_s: usize,
    n_t: usize,
    d: usize,
    bandwidth: f32,
) -> Result<f32, DomainAdaptationError> {
    if bandwidth <= 0.0 || !bandwidth.is_finite() {
        return Err(DomainAdaptationError::InvalidBandwidth { bw: bandwidth });
    }
    if n_s == 0 || n_t == 0 {
        return Err(DomainAdaptationError::EmptyFeatures);
    }
    da_check_matrix(source, n_s, d)?;
    da_check_matrix(target, n_t, d)?;

    // E_unbiased[k(x,x')] — skip diagonal i==j
    let kss = if n_s < 2 {
        0.0
    } else {
        let mut acc = 0.0f32;
        for i in 0..n_s {
            for j in 0..n_s {
                if i != j {
                    acc += da_gaussian_kernel(
                        &source[i * d..(i + 1) * d],
                        &source[j * d..(j + 1) * d],
                        d,
                        bandwidth,
                    );
                }
            }
        }
        acc / (n_s * (n_s - 1)) as f32
    };

    // E_unbiased[k(y,y')]
    let ktt = if n_t < 2 {
        0.0
    } else {
        let mut acc = 0.0f32;
        for i in 0..n_t {
            for j in 0..n_t {
                if i != j {
                    acc += da_gaussian_kernel(
                        &target[i * d..(i + 1) * d],
                        &target[j * d..(j + 1) * d],
                        d,
                        bandwidth,
                    );
                }
            }
        }
        acc / (n_t * (n_t - 1)) as f32
    };

    // Cross term (no diagonal issue since source ≠ target)
    let mut kst = 0.0f32;
    for i in 0..n_s {
        for j in 0..n_t {
            kst += da_gaussian_kernel(
                &source[i * d..(i + 1) * d],
                &target[j * d..(j + 1) * d],
                d,
                bandwidth,
            );
        }
    }
    kst /= (n_s * n_t) as f32;

    Ok((kss - 2.0 * kst + ktt).max(0.0))
}

// ---------------------------------------------------------------------------
// MMD: multi-scale
// ---------------------------------------------------------------------------

/// Compute multi-scale MMD by summing over all configured bandwidths.
///
/// The configuration is checked with [`MmdConfig::validate`] first, so a
/// bandwidth that would degenerate the Gaussian kernel (below
/// [`MmdConfig::eps`]) is reported rather than silently producing `NaN` or a
/// uniformly-zero kernel matrix.
pub fn da_mmd_multiscale(
    batch: &DomainBatch,
    config: &MmdConfig,
) -> Result<f32, DomainAdaptationError> {
    config.validate()?;
    let mut total = 0.0f32;
    for &bw in &config.kernel_bandwidths {
        let mmd = if config.biased {
            da_mmd_biased(
                &batch.source_features,
                &batch.target_features,
                batch.n_source,
                batch.n_target,
                batch.d,
                bw,
            )?
        } else {
            da_mmd_unbiased(
                &batch.source_features,
                &batch.target_features,
                batch.n_source,
                batch.n_target,
                batch.d,
                bw,
            )?
        };
        total += mmd;
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// MMD: median bandwidth heuristic
// ---------------------------------------------------------------------------

/// Estimate a good kernel bandwidth as the median pairwise distance divided
/// by `sqrt(2)`, matching [`da_gaussian_kernel`]'s `exp(-d² / (2σ²))`
/// parameterization.
///
/// This is the standard "median heuristic" used in practice: σ is set so
/// that the *median* pairwise squared distance equals `2σ²`, i.e.
/// `σ = median / sqrt(2)`. It has no dependence on the sample count `n`.
///
/// If `n * d` does not match `features.len()`, `n` is clamped down to the
/// number of complete rows actually available rather than indexing out of
/// bounds.
pub fn da_median_bandwidth(features: &[f32], n: usize, d: usize) -> f32 {
    if d == 0 {
        return 1.0;
    }
    let n = n.min(features.len() / d);
    if n < 2 {
        return 1.0;
    }
    // collect all pairwise squared distances
    let n_pairs = n * (n - 1) / 2;
    let mut dists: Vec<f32> = Vec::with_capacity(n_pairs);
    for i in 0..n {
        for j in (i + 1)..n {
            let sq: f32 = features[i * d..(i + 1) * d]
                .iter()
                .zip(features[j * d..(j + 1) * d].iter())
                .map(|(&a, &b)| {
                    let diff = a - b;
                    diff * diff
                })
                .sum();
            dists.push(sq.sqrt());
        }
    }
    // median by partial-sort
    let mid = dists.len() / 2;
    dists.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let median = dists[mid];
    (median / std::f32::consts::SQRT_2).max(1e-6)
}
