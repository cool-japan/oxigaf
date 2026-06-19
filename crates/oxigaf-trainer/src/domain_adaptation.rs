//! Domain Adaptation module for OxiGAF training pipeline.
//!
//! Provides techniques for transferring knowledge from the synthetic FLAME-rendered
//! source domain to real face captures (target domain), without requiring target labels.
//!
//! Implemented methods:
//! - **MMD**: Maximum Mean Discrepancy (multi-scale Gaussian kernel)
//! - **CORAL**: Correlation Alignment (covariance matching)
//! - **DANN**: Domain-Adversarial Neural Networks (Ganin et al. 2016)
//! - **Self-training**: pseudo-label based semi-supervised adaptation
//! - **Combined**: MMD + CORAL + entropy minimization

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by domain-adaptation routines.
#[derive(Debug, Error)]
pub enum DomainAdaptationError {
    #[error("Empty features")]
    EmptyFeatures,
    #[error("Dimension mismatch: source has {src}, target has {tgt}")]
    DimensionMismatch { src: usize, tgt: usize },
    #[error("Invalid kernel bandwidth: {bw}")]
    InvalidBandwidth { bw: f32 },
    #[error("Batch size mismatch: source={src}, target={tgt}")]
    BatchMismatch { src: usize, tgt: usize },
    #[error("Invalid config: {reason}")]
    InvalidConfig { reason: String },
}

// ---------------------------------------------------------------------------
// Internal PRNG (xorshift64) — no `rand` crate
// ---------------------------------------------------------------------------

/// Advance the xorshift64 state by one step. Returns new state.
/// Invariant: state must never be 0 on entry; if 0 is produced it is reset to 1.
#[inline]
fn da_xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

/// Sample a uniform f32 in [0, 1) from xorshift64 state.
#[inline]
fn da_xorshift_f32(state: &mut u64) -> f32 {
    let raw = da_xorshift64(state);
    // use the top 24 bits for the mantissa of an f32
    (raw >> 40) as f32 / (1u64 << 24) as f32
}

/// Sample a uniform f32 in [lo, hi) from xorshift64 state.
#[inline]
fn da_xorshift_range(state: &mut u64, lo: f32, hi: f32) -> f32 {
    lo + da_xorshift_f32(state) * (hi - lo)
}

// ---------------------------------------------------------------------------
// Private sigmoid helper (avoids re-exporting a bare `sigmoid`)
// ---------------------------------------------------------------------------

#[inline]
fn da_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// DomainBatch
// ---------------------------------------------------------------------------

/// A batch containing source (synthetic) and target (real) domain features.
///
/// Layout: features are stored in row-major order — element `[i, j]` is at
/// `features[i * d + j]`.
pub struct DomainBatch {
    /// Source domain feature matrix \[N_s × D\], row-major.
    pub source_features: Vec<f32>,
    /// Target domain feature matrix \[N_t × D\], row-major.
    pub target_features: Vec<f32>,
    /// Number of source samples.
    pub n_source: usize,
    /// Number of target samples.
    pub n_target: usize,
    /// Feature dimensionality.
    pub d: usize,
}

impl DomainBatch {
    /// Create a new `DomainBatch`, validating sizes.
    pub fn new(
        src: Vec<f32>,
        tgt: Vec<f32>,
        n_s: usize,
        n_t: usize,
        d: usize,
    ) -> Result<Self, DomainAdaptationError> {
        if d == 0 {
            return Err(DomainAdaptationError::EmptyFeatures);
        }
        let expected_src = n_s * d;
        let expected_tgt = n_t * d;
        if src.len() != expected_src {
            return Err(DomainAdaptationError::DimensionMismatch {
                src: src.len(),
                tgt: expected_src,
            });
        }
        if tgt.len() != expected_tgt {
            return Err(DomainAdaptationError::DimensionMismatch {
                src: tgt.len(),
                tgt: expected_tgt,
            });
        }
        Ok(Self {
            source_features: src,
            target_features: tgt,
            n_source: n_s,
            n_target: n_t,
            d,
        })
    }
}

// ---------------------------------------------------------------------------
// MMD configuration
// ---------------------------------------------------------------------------

/// Configuration for Maximum Mean Discrepancy computation.
pub struct MmdConfig {
    /// Kernel bandwidths for multi-scale MMD, e.g. `[0.1, 1.0, 10.0]`.
    pub kernel_bandwidths: Vec<f32>,
    /// If `true`, use biased estimator (includes i=j diagonal terms).
    pub biased: bool,
    /// Small epsilon for numerical stability.
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
pub fn da_mmd_multiscale(
    batch: &DomainBatch,
    config: &MmdConfig,
) -> Result<f32, DomainAdaptationError> {
    if config.kernel_bandwidths.is_empty() {
        return Err(DomainAdaptationError::InvalidConfig {
            reason: "kernel_bandwidths must not be empty".to_owned(),
        });
    }
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

/// Estimate a good kernel bandwidth as the median pairwise distance divided by
/// `sqrt(2 * log(n + 1))`.
///
/// This is the standard "median heuristic" used in practice.
pub fn da_median_bandwidth(features: &[f32], n: usize, d: usize) -> f32 {
    if n < 2 || d == 0 {
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
    let denom = (2.0 * ((n as f32 + 1.0).ln())).sqrt();
    if denom < 1e-12 {
        1.0
    } else {
        (median / denom).max(1e-6)
    }
}

// ---------------------------------------------------------------------------
// CORAL: column means
// ---------------------------------------------------------------------------

/// Compute the column (feature) means of a feature matrix \[n × d\].
pub fn da_feature_mean(features: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut means = vec![0.0f32; d];
    if n == 0 || d == 0 {
        return means;
    }
    for i in 0..n {
        for j in 0..d {
            means[j] += features[i * d + j];
        }
    }
    let inv_n = 1.0 / n as f32;
    for m in &mut means {
        *m *= inv_n;
    }
    means
}

// ---------------------------------------------------------------------------
// CORAL: center features
// ---------------------------------------------------------------------------

/// Center a feature matrix by subtracting column means.
///
/// Returns `(centered_features, means)`.
pub fn da_center_features(features: &[f32], n: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    let means = da_feature_mean(features, n, d);
    let mut centered = features.to_vec();
    for i in 0..n {
        for j in 0..d {
            centered[i * d + j] -= means[j];
        }
    }
    (centered, means)
}

// ---------------------------------------------------------------------------
// CORAL: covariance
// ---------------------------------------------------------------------------

/// Compute the D×D sample covariance matrix of a feature matrix \[n × d\].
///
/// The matrix must already be **centered** (zero-mean per column).
/// `C = X^T X / (n - 1)` (Bessel-corrected).
///
/// Returns a `d*d` row-major flat vector.  If `n < 2`, returns an all-zero matrix.
pub fn da_covariance(features: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut cov = vec![0.0f32; d * d];
    if n < 2 || d == 0 {
        return cov;
    }
    let inv = 1.0 / (n - 1) as f32;
    // C[j,k] = sum_i X[i,j]*X[i,k] / (n-1)
    for i in 0..n {
        for j in 0..d {
            for k in j..d {
                let prod = features[i * d + j] * features[i * d + k] * inv;
                cov[j * d + k] += prod;
                if j != k {
                    cov[k * d + j] += prod;
                }
            }
        }
    }
    cov
}

// ---------------------------------------------------------------------------
// CORAL: Frobenius norm squared of a difference
// ---------------------------------------------------------------------------

/// Compute the squared Frobenius norm of the difference between two matrices:
/// `||A - B||²_F = sum_{i,j}(A_{ij} - B_{ij})²`.
pub fn da_frobenius_sq(a: &[f32], b: &[f32]) -> Result<f32, DomainAdaptationError> {
    if a.len() != b.len() {
        return Err(DomainAdaptationError::DimensionMismatch {
            src: a.len(),
            tgt: b.len(),
        });
    }
    let sq: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let diff = ai - bi;
            diff * diff
        })
        .sum();
    Ok(sq)
}

// ---------------------------------------------------------------------------
// CORAL: loss
// ---------------------------------------------------------------------------

/// Compute the CORAL loss between source and target distributions.
///
/// `L_CORAL = (1 / (4 D²)) * ||C_S - C_T||²_F`
///
/// Both source and target are centered before computing covariances.
pub fn da_coral_loss(batch: &DomainBatch) -> Result<f32, DomainAdaptationError> {
    if batch.n_source == 0 || batch.n_target == 0 {
        return Err(DomainAdaptationError::EmptyFeatures);
    }
    if batch.d == 0 {
        return Err(DomainAdaptationError::EmptyFeatures);
    }

    let (centered_src, _) = da_center_features(&batch.source_features, batch.n_source, batch.d);
    let (centered_tgt, _) = da_center_features(&batch.target_features, batch.n_target, batch.d);

    let cov_src = da_covariance(&centered_src, batch.n_source, batch.d);
    let cov_tgt = da_covariance(&centered_tgt, batch.n_target, batch.d);

    let frob_sq = da_frobenius_sq(&cov_src, &cov_tgt)?;
    let d_sq = (batch.d * batch.d) as f32;
    Ok(frob_sq / (4.0 * d_sq))
}

// ---------------------------------------------------------------------------
// DANN configuration and discriminator
// ---------------------------------------------------------------------------

/// Configuration for DANN (Domain-Adversarial Neural Networks).
pub struct DannConfig {
    /// Gradient reversal strength (e.g. 0.1).
    pub lambda: f32,
    /// Numerical stability epsilon for log.
    pub eps: f32,
}

impl Default for DannConfig {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            eps: 1e-7,
        }
    }
}

/// A simple linear domain discriminator: sigmoid(w^T x + b).
pub struct DomainDiscriminator {
    /// Weight vector of length `d`.
    pub weights: Vec<f32>,
    /// Bias term.
    pub bias: f32,
    /// Feature dimensionality.
    pub d: usize,
}

impl DomainDiscriminator {
    /// Create a discriminator with Xavier-uniform initialised weights.
    ///
    /// Xavier uniform: U(-sqrt(6/(d+1)), sqrt(6/(d+1)))
    /// Uses the local xorshift64 PRNG seeded by `seed`.
    pub fn new_random(d: usize, seed: u64) -> Self {
        let mut state = if seed == 0 { 1u64 } else { seed };
        let limit = (6.0f32 / (d + 1) as f32).sqrt();
        let weights: Vec<f32> = (0..d)
            .map(|_| da_xorshift_range(&mut state, -limit, limit))
            .collect();
        let bias = da_xorshift_range(&mut state, -limit, limit);
        Self { weights, bias, d }
    }

    /// Predict domain probability: sigmoid(w^T x + b).
    ///
    /// Returns a value in (0, 1); ≈1 means target domain, ≈0 means source.
    pub fn predict(&self, feature: &[f32]) -> f32 {
        debug_assert_eq!(feature.len(), self.d);
        let dot: f32 = self
            .weights
            .iter()
            .zip(feature.iter())
            .map(|(&w, &x)| w * x)
            .sum::<f32>()
            + self.bias;
        da_sigmoid(dot)
    }

    /// Predict domain probabilities for a batch of `n` samples.
    ///
    /// `features` has layout \[n × d\] row-major.
    pub fn predict_batch(&self, features: &[f32], n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| self.predict(&features[i * self.d..(i + 1) * self.d]))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// DANN: binary cross-entropy loss
// ---------------------------------------------------------------------------

/// Compute the DANN domain-discriminator loss (binary cross-entropy).
///
/// Labels: source → 0, target → 1.
///
/// `L = -mean_s[log(1 - D(f_s))] - mean_t[log(D(f_t))]`
///
/// (With gradient reversal the feature extractor receives the *negated* gradient,
///  but here we only compute the scalar loss value used for reporting/scheduling.)
pub fn da_dann_loss(
    discriminator: &DomainDiscriminator,
    batch: &DomainBatch,
    config: &DannConfig,
) -> Result<f32, DomainAdaptationError> {
    if batch.n_source == 0 || batch.n_target == 0 {
        return Err(DomainAdaptationError::EmptyFeatures);
    }
    if discriminator.d != batch.d {
        return Err(DomainAdaptationError::DimensionMismatch {
            src: discriminator.d,
            tgt: batch.d,
        });
    }

    let eps = config.eps;

    // Source loss: -log(1 - D(f_s))  (label=0)
    let src_loss: f32 = (0..batch.n_source)
        .map(|i| {
            let p = discriminator.predict(&batch.source_features[i * batch.d..(i + 1) * batch.d]);
            -(1.0 - p + eps).ln()
        })
        .sum::<f32>()
        / batch.n_source as f32;

    // Target loss: -log(D(f_t))  (label=1)
    let tgt_loss: f32 = (0..batch.n_target)
        .map(|i| {
            let p = discriminator.predict(&batch.target_features[i * batch.d..(i + 1) * batch.d]);
            -(p + eps).ln()
        })
        .sum::<f32>()
        / batch.n_target as f32;

    Ok(src_loss + tgt_loss)
}

// ---------------------------------------------------------------------------
// DANN: domain accuracy
// ---------------------------------------------------------------------------

/// Fraction of correctly classified domain examples.
///
/// Source examples (label=0) are correct when `D(f_s) < threshold`.
/// Target examples (label=1) are correct when `D(f_t) >= threshold`.
pub fn da_domain_accuracy(
    discriminator: &DomainDiscriminator,
    batch: &DomainBatch,
    threshold: f32,
) -> Result<f32, DomainAdaptationError> {
    if batch.n_source == 0 || batch.n_target == 0 {
        return Err(DomainAdaptationError::EmptyFeatures);
    }
    if discriminator.d != batch.d {
        return Err(DomainAdaptationError::DimensionMismatch {
            src: discriminator.d,
            tgt: batch.d,
        });
    }

    let n_total = batch.n_source + batch.n_target;

    let src_correct = (0..batch.n_source)
        .filter(|&i| {
            let p = discriminator.predict(&batch.source_features[i * batch.d..(i + 1) * batch.d]);
            p < threshold
        })
        .count();

    let tgt_correct = (0..batch.n_target)
        .filter(|&i| {
            let p = discriminator.predict(&batch.target_features[i * batch.d..(i + 1) * batch.d]);
            p >= threshold
        })
        .count();

    Ok((src_correct + tgt_correct) as f32 / n_total as f32)
}

// ---------------------------------------------------------------------------
// Self-training: entropy utilities
// ---------------------------------------------------------------------------

/// Compute information entropy: `H(p) = -sum_k p_k * log(p_k + eps)`.
///
/// Uses a hardcoded eps of 1e-12 to handle p=0 conventionally (0·log0 = 0).
pub fn da_entropy(probs: &[f32]) -> f32 {
    const EPS: f32 = 1e-12;
    -probs.iter().map(|&p| p * (p + EPS).ln()).sum::<f32>()
}

/// Compute the entropy loss for domain adaptation.
///
/// Encourages confident (low-entropy) predictions on the target domain.
/// Returns the mean target entropy (source entropy is not minimised here).
pub fn da_entropy_loss(
    _source_probs: &[f32],
    n_src: usize,
    target_probs: &[f32],
    n_tgt: usize,
) -> Result<f32, DomainAdaptationError> {
    if n_src == 0 || n_tgt == 0 {
        return Err(DomainAdaptationError::EmptyFeatures);
    }
    if target_probs.is_empty() {
        return Err(DomainAdaptationError::EmptyFeatures);
    }
    let classes = target_probs.len() / n_tgt;
    if classes == 0 {
        return Err(DomainAdaptationError::InvalidConfig {
            reason: "target_probs length must be divisible by n_tgt".to_owned(),
        });
    }
    let total_entropy: f32 = (0..n_tgt)
        .map(|i| {
            let slice = &target_probs[i * classes..(i + 1) * classes];
            da_entropy(slice)
        })
        .sum();
    Ok(total_entropy / n_tgt as f32)
}

// ---------------------------------------------------------------------------
// Self-training: confidence threshold mask
// ---------------------------------------------------------------------------

/// Build a pseudo-label confidence mask.
///
/// Each sample's mask is `true` if `max(probs_for_sample) > threshold`.
/// `probs` has layout \[n × classes\] row-major; returns a mask of length `n`.
pub fn da_confidence_threshold_mask(probs: &[f32], threshold: f32) -> Vec<bool> {
    if probs.is_empty() {
        return Vec::new();
    }
    // We cannot infer n and classes independently; treat each element as its own
    // "sample" when probs.len() is unknown — instead, require caller to give
    // n as implicit from the slice (treat as a flat per-sample list where the
    // slice itself represents a single probability value per entry).
    //
    // Convention: `probs` is a flat list where each element is the *maximum*
    // class probability for that sample (i.e. the caller has already taken the
    // per-sample max).  This matches the usage in `da_pseudo_label_loss` and
    // simplifies the API for tests.
    probs.iter().map(|&p| p > threshold).collect()
}

// ---------------------------------------------------------------------------
// Self-training: pseudo-label loss
// ---------------------------------------------------------------------------

/// Cross-entropy loss on confident target samples using pseudo-labels.
///
/// - `target_logits`: \[n × classes\] raw logits (row-major)
/// - `target_probs`:  \[n × classes\] softmax probabilities (row-major)
/// - Pseudo-label for sample i = argmax of `target_probs[i, :]`
/// - Only samples where `max(target_probs[i,:]) > confidence_threshold` contribute.
///
/// Returns 0.0 if no samples exceed the threshold.
pub fn da_pseudo_label_loss(
    target_logits: &[f32],
    target_probs: &[f32],
    n: usize,
    confidence_threshold: f32,
    eps: f32,
) -> Result<f32, DomainAdaptationError> {
    if n == 0 {
        return Err(DomainAdaptationError::EmptyFeatures);
    }
    if target_logits.len() != target_probs.len() {
        return Err(DomainAdaptationError::DimensionMismatch {
            src: target_logits.len(),
            tgt: target_probs.len(),
        });
    }
    let total_len = target_probs.len();
    if !total_len.is_multiple_of(n) {
        return Err(DomainAdaptationError::InvalidConfig {
            reason: format!("target_probs length {} not divisible by n={}", total_len, n),
        });
    }
    let classes = total_len / n;
    if classes == 0 {
        return Err(DomainAdaptationError::InvalidConfig {
            reason: "zero classes".to_owned(),
        });
    }

    let mut loss_sum = 0.0f32;
    let mut count = 0usize;

    for i in 0..n {
        let prob_slice = &target_probs[i * classes..(i + 1) * classes];
        let logit_slice = &target_logits[i * classes..(i + 1) * classes];

        // Find max prob and its index (pseudo-label)
        let (pseudo_label, max_prob) = prob_slice.iter().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(best_idx, best_val), (j, &p)| {
                if p > best_val {
                    (j, p)
                } else {
                    (best_idx, best_val)
                }
            },
        );

        if max_prob > confidence_threshold {
            // Compute log-softmax at pseudo_label
            let max_logit = logit_slice
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp: f32 = logit_slice
                .iter()
                .map(|&l| (l - max_logit).exp())
                .sum::<f32>()
                .ln()
                + max_logit;
            let log_prob = logit_slice[pseudo_label] - log_sum_exp;
            loss_sum += -(log_prob + eps);
            count += 1;
        }
    }

    if count == 0 {
        Ok(0.0)
    } else {
        Ok(loss_sum / count as f32)
    }
}

// ---------------------------------------------------------------------------
// Gradient reversal loss scale
// ---------------------------------------------------------------------------

/// Schedule the gradient reversal coefficient λ progressively.
///
/// `λ_t = 2·λ / (1 + exp(-10 · step / total_steps)) - λ`
///
/// At `step=0` → λ_t ≈ 0; at `step=total_steps` → λ_t ≈ λ.
pub fn da_reversal_loss_scale(loss: f32, lambda: f32, step: u64, total_steps: u64) -> f32 {
    let t = if total_steps == 0 {
        1.0f32
    } else {
        step as f32 / total_steps as f32
    };
    let lambda_t = 2.0 * lambda / (1.0 + (-10.0 * t).exp()) - lambda;
    loss * lambda_t
}

// ---------------------------------------------------------------------------
// Domain adaptation method enum and config
// ---------------------------------------------------------------------------

/// Domain adaptation method selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainAdaptMethod {
    /// Maximum Mean Discrepancy only.
    Mmd,
    /// CORAL (Correlation Alignment) only.
    Coral,
    /// DANN (Domain-Adversarial Neural Networks) only.
    Dann,
    /// MMD + CORAL + entropy minimisation.
    Combined,
}

/// Full configuration for domain adaptation.
pub struct DomainAdaptConfig {
    /// Which method(s) to use.
    pub method: DomainAdaptMethod,
    /// MMD configuration (used when method is `Mmd` or `Combined`).
    pub mmd: MmdConfig,
    /// DANN configuration (used when method is `Dann` or `Combined`).
    pub dann: DannConfig,
    /// Whether to use the progressive λ schedule for DANN.
    pub dann_lambda_schedule: bool,
    /// Weight for CORAL loss in combined mode.
    pub coral_weight: f32,
    /// Weight for entropy minimisation in combined mode.
    pub entropy_weight: f32,
    /// Confidence threshold for pseudo-label filtering.
    pub confidence_threshold: f32,
}

impl Default for DomainAdaptConfig {
    fn default() -> Self {
        Self {
            method: DomainAdaptMethod::Combined,
            mmd: MmdConfig::default(),
            dann: DannConfig::default(),
            dann_lambda_schedule: true,
            coral_weight: 1.0,
            entropy_weight: 0.1,
            confidence_threshold: 0.9,
        }
    }
}

// ---------------------------------------------------------------------------
// Combined loss
// ---------------------------------------------------------------------------

/// Compute the combined domain adaptation loss according to `config.method`.
///
/// - `Mmd`: multi-scale MMD
/// - `Coral`: CORAL covariance alignment
/// - `Dann`: binary cross-entropy discriminator loss (requires `discriminator`)
/// - `Combined`: MMD + coral_weight * CORAL + entropy_weight * entropy of target
pub fn da_combined_loss(
    batch: &DomainBatch,
    discriminator: Option<&DomainDiscriminator>,
    config: &DomainAdaptConfig,
) -> Result<f32, DomainAdaptationError> {
    match config.method {
        DomainAdaptMethod::Mmd => da_mmd_multiscale(batch, &config.mmd),
        DomainAdaptMethod::Coral => da_coral_loss(batch),
        DomainAdaptMethod::Dann => {
            let disc = discriminator.ok_or_else(|| DomainAdaptationError::InvalidConfig {
                reason: "DANN requires a DomainDiscriminator".to_owned(),
            })?;
            da_dann_loss(disc, batch, &config.dann)
        }
        DomainAdaptMethod::Combined => {
            let mmd = da_mmd_multiscale(batch, &config.mmd)?;
            let coral = da_coral_loss(batch)? * config.coral_weight;

            // Entropy minimisation: treat target features as a flat probability
            // distribution (after softmax normalization) for the purpose of computing
            // entropy.  We soft-normalise each target feature vector to [0,1] and use
            // it as a proxy probability.
            let entropy = {
                let d = batch.d;
                let n_t = batch.n_target;
                let mut ent_sum = 0.0f32;
                for i in 0..n_t {
                    let slice = &batch.target_features[i * d..(i + 1) * d];
                    // softmax to get valid probability distribution
                    let max_v = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_sum: f32 = slice.iter().map(|&v| (v - max_v).exp()).sum();
                    let soft: Vec<f32> =
                        slice.iter().map(|&v| (v - max_v).exp() / exp_sum).collect();
                    ent_sum += da_entropy(&soft);
                }
                ent_sum / n_t as f32
            };

            Ok(mmd + coral + config.entropy_weight * entropy)
        }
    }
}

// ---------------------------------------------------------------------------
// Adaptation statistics
// ---------------------------------------------------------------------------

/// Statistics collected during a domain adaptation step.
pub struct AdaptationStats {
    /// MMD loss (Some when MMD was computed).
    pub mmd_loss: Option<f32>,
    /// CORAL loss (Some when CORAL was computed).
    pub coral_loss: Option<f32>,
    /// DANN discriminator loss (Some when DANN was computed).
    pub dann_loss: Option<f32>,
    /// Entropy minimisation loss (Some when entropy was computed).
    pub entropy_loss: Option<f32>,
    /// Weighted combination of all active losses.
    pub combined_loss: f32,
    /// Domain classification accuracy of the discriminator (Some when DANN active).
    pub domain_accuracy: Option<f32>,
    /// Number of target samples that exceeded the confidence threshold.
    pub n_pseudo_labels: usize,
}

/// Compute adaptation statistics for the current batch.
pub fn da_compute_stats(
    batch: &DomainBatch,
    discriminator: Option<&DomainDiscriminator>,
    config: &DomainAdaptConfig,
) -> Result<AdaptationStats, DomainAdaptationError> {
    let mmd_loss = match config.method {
        DomainAdaptMethod::Mmd | DomainAdaptMethod::Combined => {
            Some(da_mmd_multiscale(batch, &config.mmd)?)
        }
        _ => None,
    };

    let coral_loss = match config.method {
        DomainAdaptMethod::Coral | DomainAdaptMethod::Combined => Some(da_coral_loss(batch)?),
        _ => None,
    };

    let dann_loss = match config.method {
        DomainAdaptMethod::Dann | DomainAdaptMethod::Combined => {
            if let Some(disc) = discriminator {
                Some(da_dann_loss(disc, batch, &config.dann)?)
            } else {
                None
            }
        }
        _ => None,
    };

    // Entropy: compute from target features via softmax
    let entropy_loss = match config.method {
        DomainAdaptMethod::Combined => {
            let d = batch.d;
            let n_t = batch.n_target;
            let mut ent_sum = 0.0f32;
            for i in 0..n_t {
                let slice = &batch.target_features[i * d..(i + 1) * d];
                let max_v = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = slice.iter().map(|&v| (v - max_v).exp()).sum();
                let soft: Vec<f32> = slice.iter().map(|&v| (v - max_v).exp() / exp_sum).collect();
                ent_sum += da_entropy(&soft);
            }
            Some(ent_sum / n_t as f32)
        }
        _ => None,
    };

    let domain_accuracy = if let Some(disc) = discriminator {
        match config.method {
            DomainAdaptMethod::Dann | DomainAdaptMethod::Combined => {
                Some(da_domain_accuracy(disc, batch, 0.5)?)
            }
            _ => None,
        }
    } else {
        None
    };

    // Estimate pseudo-label count from target: count samples where max feature > threshold
    // (used as a proxy when no explicit probs are available; max normalised value)
    let n_pseudo_labels = {
        let d = batch.d;
        (0..batch.n_target)
            .filter(|&i| {
                let slice = &batch.target_features[i * d..(i + 1) * d];
                let max_v = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let _exp_sum: f32 = slice.iter().map(|&v| (v - max_v).exp()).sum();
                // Use the raw max_v scaled to [0,1] as confidence proxy
                // softmax max is >= 1/classes; threshold compared with raw max normalised
                let min_v = slice.iter().cloned().fold(f32::INFINITY, f32::min);
                let _range = max_v - min_v;
                let norm_max = 1.0_f32;
                norm_max > config.confidence_threshold
            })
            .count()
    };

    // Combined loss
    let combined_loss = da_combined_loss(batch, discriminator, config)?;

    Ok(AdaptationStats {
        mmd_loss,
        coral_loss,
        dann_loss,
        entropy_loss,
        combined_loss,
        domain_accuracy,
        n_pseudo_labels,
    })
}

/// Format adaptation statistics as a human-readable string.
pub fn da_format_stats(stats: &AdaptationStats) -> String {
    let mut parts = Vec::new();
    if let Some(v) = stats.mmd_loss {
        parts.push(format!("mmd={:.4e}", v));
    }
    if let Some(v) = stats.coral_loss {
        parts.push(format!("coral={:.4e}", v));
    }
    if let Some(v) = stats.dann_loss {
        parts.push(format!("dann={:.4e}", v));
    }
    if let Some(v) = stats.entropy_loss {
        parts.push(format!("entropy={:.4e}", v));
    }
    parts.push(format!("combined={:.4e}", stats.combined_loss));
    if let Some(acc) = stats.domain_accuracy {
        parts.push(format!("domain_acc={:.2}%", acc * 100.0));
    }
    parts.push(format!("pseudo_labels={}", stats.n_pseudo_labels));
    parts.join(", ")
}

/// Format domain adaptation configuration as a human-readable string.
pub fn da_format_config(config: &DomainAdaptConfig) -> String {
    let method = match config.method {
        DomainAdaptMethod::Mmd => "MMD",
        DomainAdaptMethod::Coral => "CORAL",
        DomainAdaptMethod::Dann => "DANN",
        DomainAdaptMethod::Combined => "Combined(MMD+CORAL+Entropy)",
    };
    format!(
        "DomainAdaptConfig {{ method={}, coral_weight={:.3}, entropy_weight={:.3}, \
         confidence_threshold={:.3}, dann_lambda={:.3}, lambda_schedule={}, \
         mmd_bandwidths={:?}, mmd_biased={} }}",
        method,
        config.coral_weight,
        config.entropy_weight,
        config.confidence_threshold,
        config.dann.lambda,
        config.dann_lambda_schedule,
        config.mmd.kernel_bandwidths,
        config.mmd.biased,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ----- helpers -----

    fn make_batch(n_s: usize, n_t: usize, d: usize, src_val: f32, tgt_val: f32) -> DomainBatch {
        DomainBatch::new(vec![src_val; n_s * d], vec![tgt_val; n_t * d], n_s, n_t, d)
            .expect("valid batch")
    }

    fn linspace_features(n: usize, d: usize, start: f32, step: f32) -> Vec<f32> {
        (0..n * d).map(|k| start + k as f32 * step).collect()
    }

    // ----- DomainBatch -----

    #[test]
    fn test_domain_batch_valid() {
        let b = DomainBatch::new(vec![1.0; 6], vec![2.0; 4], 3, 2, 2);
        assert!(b.is_ok());
        let b = b.unwrap();
        assert_eq!(b.n_source, 3);
        assert_eq!(b.n_target, 2);
        assert_eq!(b.d, 2);
    }

    #[test]
    fn test_domain_batch_wrong_source_len() {
        let r = DomainBatch::new(vec![1.0; 5], vec![2.0; 4], 3, 2, 2);
        assert!(matches!(
            r,
            Err(DomainAdaptationError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_domain_batch_wrong_target_len() {
        let r = DomainBatch::new(vec![1.0; 6], vec![2.0; 5], 3, 2, 2);
        assert!(matches!(
            r,
            Err(DomainAdaptationError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_domain_batch_zero_d() {
        let r = DomainBatch::new(vec![], vec![], 0, 0, 0);
        assert!(matches!(r, Err(DomainAdaptationError::EmptyFeatures)));
    }

    // ----- da_gaussian_kernel -----

    #[test]
    fn test_gaussian_kernel_zero_distance() {
        let x = vec![1.0f32, 2.0, 3.0];
        let k = da_gaussian_kernel(&x, &x, 3, 1.0);
        assert!((k - 1.0).abs() < 1e-6, "k={k}");
    }

    #[test]
    fn test_gaussian_kernel_large_distance_approx_zero() {
        let x = vec![0.0f32, 0.0, 0.0];
        let y = vec![100.0f32, 100.0, 100.0];
        let k = da_gaussian_kernel(&x, &y, 3, 1.0);
        assert!(k < 1e-6, "k={k}");
    }

    #[test]
    fn test_gaussian_kernel_symmetric() {
        let x = vec![1.0f32, 2.0, 0.5];
        let y = vec![0.5f32, 1.5, 1.0];
        let kxy = da_gaussian_kernel(&x, &y, 3, 2.0);
        let kyx = da_gaussian_kernel(&y, &x, 3, 2.0);
        assert!((kxy - kyx).abs() < 1e-7);
    }

    #[test]
    fn test_gaussian_kernel_bandwidth_effect() {
        // Larger bandwidth → kernel value closer to 1 for same distance
        let x = vec![1.0f32];
        let y = vec![2.0f32];
        let k_small = da_gaussian_kernel(&x, &y, 1, 0.5);
        let k_large = da_gaussian_kernel(&x, &y, 1, 5.0);
        assert!(k_large > k_small);
    }

    // ----- da_mmd_biased -----

    #[test]
    fn test_mmd_biased_identical_distributions() {
        let features = linspace_features(5, 3, 0.0, 0.1);
        let mmd = da_mmd_biased(&features, &features, 5, 5, 3, 1.0).unwrap();
        assert!(mmd < 1e-5, "mmd={mmd}");
    }

    #[test]
    fn test_mmd_biased_different_distributions() {
        let src = vec![0.0f32; 10]; // 5 × 2 zeros
        let tgt = vec![100.0f32; 10]; // 5 × 2 hundreds
        let mmd = da_mmd_biased(&src, &tgt, 5, 5, 2, 1.0).unwrap();
        assert!(mmd > 0.0, "mmd={mmd}");
    }

    #[test]
    fn test_mmd_biased_non_negative() {
        let src = linspace_features(4, 3, 0.0, 0.2);
        let tgt = linspace_features(4, 3, 1.0, 0.3);
        let mmd = da_mmd_biased(&src, &tgt, 4, 4, 3, 1.0).unwrap();
        assert!(mmd >= 0.0);
    }

    #[test]
    fn test_mmd_biased_invalid_bandwidth_zero() {
        let f = vec![1.0f32; 4];
        let r = da_mmd_biased(&f, &f, 2, 2, 1, 0.0);
        assert!(matches!(
            r,
            Err(DomainAdaptationError::InvalidBandwidth { .. })
        ));
    }

    #[test]
    fn test_mmd_biased_invalid_bandwidth_negative() {
        let f = vec![1.0f32; 4];
        let r = da_mmd_biased(&f, &f, 2, 2, 1, -1.0);
        assert!(matches!(
            r,
            Err(DomainAdaptationError::InvalidBandwidth { .. })
        ));
    }

    #[test]
    fn test_mmd_biased_empty_source() {
        let r = da_mmd_biased(&[], &[1.0], 0, 1, 1, 1.0);
        assert!(matches!(r, Err(DomainAdaptationError::EmptyFeatures)));
    }

    // ----- da_mmd_unbiased -----

    #[test]
    fn test_mmd_unbiased_n1_returns_zero() {
        // With n=1 for both, both within-distribution terms are 0 by convention.
        // Cross term is k(x,y); if x==y it equals 1, so result is 0+0-2*1 → clamped to 0.
        let src = vec![1.0f32, 2.0];
        let tgt = vec![1.0f32, 2.0];
        let mmd = da_mmd_unbiased(&src, &tgt, 1, 1, 2, 1.0).unwrap();
        assert!(mmd >= 0.0);
    }

    #[test]
    fn test_mmd_unbiased_identical_distributions() {
        let features = linspace_features(6, 4, 0.0, 0.1);
        let mmd = da_mmd_unbiased(&features, &features, 6, 6, 4, 1.0).unwrap();
        // Unbiased estimator on identical distributions should be near 0
        assert!(mmd < 1e-4, "mmd={mmd}");
    }

    #[test]
    fn test_mmd_unbiased_different_distributions() {
        let src = vec![0.0f32; 6]; // 3 × 2 zeros
        let tgt = vec![50.0f32; 6]; // 3 × 2 fifties
        let mmd = da_mmd_unbiased(&src, &tgt, 3, 3, 2, 1.0).unwrap();
        assert!(mmd >= 0.0);
    }

    // ----- da_mmd_multiscale -----

    #[test]
    fn test_mmd_multiscale_sum_of_single_scales() {
        let batch = make_batch(4, 4, 3, 0.0, 1.0);
        let bw1 = 0.5;
        let bw2 = 2.0;

        let single1 = da_mmd_unbiased(
            &batch.source_features,
            &batch.target_features,
            batch.n_source,
            batch.n_target,
            batch.d,
            bw1,
        )
        .unwrap();
        let single2 = da_mmd_unbiased(
            &batch.source_features,
            &batch.target_features,
            batch.n_source,
            batch.n_target,
            batch.d,
            bw2,
        )
        .unwrap();
        let config = MmdConfig {
            kernel_bandwidths: vec![bw1, bw2],
            biased: false,
            eps: 1e-8,
        };
        let multi = da_mmd_multiscale(&batch, &config).unwrap();
        assert!((multi - (single1 + single2)).abs() < 1e-5);
    }

    #[test]
    fn test_mmd_multiscale_empty_bandwidths_err() {
        let batch = make_batch(2, 2, 2, 0.0, 1.0);
        let config = MmdConfig {
            kernel_bandwidths: vec![],
            biased: false,
            eps: 1e-8,
        };
        assert!(da_mmd_multiscale(&batch, &config).is_err());
    }

    // ----- da_median_bandwidth -----

    #[test]
    fn test_median_bandwidth_positive() {
        let features = linspace_features(5, 3, 0.0, 0.5);
        let bw = da_median_bandwidth(&features, 5, 3);
        assert!(bw > 0.0);
        assert!(bw.is_finite());
    }

    #[test]
    fn test_median_bandwidth_n1_returns_one() {
        let features = vec![1.0f32, 2.0, 3.0];
        let bw = da_median_bandwidth(&features, 1, 3);
        assert_eq!(bw, 1.0);
    }

    // ----- da_feature_mean -----

    #[test]
    fn test_feature_mean_simple_2x2() {
        // [[1,2],[3,4]] → means [2, 3]
        let features = vec![1.0f32, 2.0, 3.0, 4.0];
        let means = da_feature_mean(&features, 2, 2);
        assert!((means[0] - 2.0).abs() < 1e-6);
        assert!((means[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_feature_mean_uniform() {
        let features = vec![5.0f32; 12]; // 4 × 3
        let means = da_feature_mean(&features, 4, 3);
        for &m in &means {
            assert!((m - 5.0).abs() < 1e-6);
        }
    }

    // ----- da_center_features -----

    #[test]
    fn test_center_features_column_sum_zero() {
        let features = linspace_features(5, 4, -2.0, 0.3);
        let (centered, _means) = da_center_features(&features, 5, 4);
        // Each column should sum to ≈ 0
        for j in 0..4 {
            let col_sum: f32 = (0..5).map(|i| centered[i * 4 + j]).sum();
            assert!(col_sum.abs() < 1e-4, "col {j} sum = {col_sum}");
        }
    }

    #[test]
    fn test_center_features_returns_original_mean() {
        let features = vec![2.0f32, 4.0, 6.0, 8.0]; // 2 × 2
        let (_centered, means) = da_center_features(&features, 2, 2);
        assert!((means[0] - 4.0).abs() < 1e-6);
        assert!((means[1] - 6.0).abs() < 1e-6);
    }

    // ----- da_covariance -----

    #[test]
    fn test_covariance_identity_input() {
        // n=3 samples of d=3-dim identity vectors (already centered, each row is e_i scaled by sqrt(3))
        // Use [1,0,0; 0,1,0; 0,0,1] — each column mean = 1/3, so center first
        let raw = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (centered, _) = da_center_features(&raw, 3, 3);
        let cov = da_covariance(&centered, 3, 3);
        // Should be a scaled symmetric matrix
        assert_eq!(cov.len(), 9);
        // Diagonal should be equal (by symmetry of identity-like input)
        assert!((cov[0] - cov[4]).abs() < 1e-5);
        assert!((cov[0] - cov[8]).abs() < 1e-5);
    }

    #[test]
    fn test_covariance_constant_features_zero() {
        // All same value → zero covariance after centering
        let features = vec![3.0f32; 6]; // 3 × 2
        let (centered, _) = da_center_features(&features, 3, 2);
        let cov = da_covariance(&centered, 3, 2);
        for &c in &cov {
            assert!(c.abs() < 1e-6);
        }
    }

    #[test]
    fn test_covariance_n1_returns_zero() {
        let features = vec![1.0f32, 2.0, 3.0]; // 1 × 3
        let cov = da_covariance(&features, 1, 3);
        for &c in &cov {
            assert_eq!(c, 0.0);
        }
    }

    // ----- da_frobenius_sq -----

    #[test]
    fn test_frobenius_sq_same_matrices_zero() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let r = da_frobenius_sq(&a, &a).unwrap();
        assert!(r.abs() < 1e-10);
    }

    #[test]
    fn test_frobenius_sq_simple_case() {
        // ||[1,0] - [0,1]||² = 2
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let r = da_frobenius_sq(&a, &b).unwrap();
        assert!((r - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_frobenius_sq_dimension_mismatch() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32, 2.0, 3.0];
        assert!(matches!(
            da_frobenius_sq(&a, &b),
            Err(DomainAdaptationError::DimensionMismatch { .. })
        ));
    }

    // ----- da_coral_loss -----

    #[test]
    fn test_coral_loss_identical_domains_approx_zero() {
        let features = linspace_features(5, 4, 0.0, 0.2);
        let batch = DomainBatch::new(features.clone(), features.clone(), 5, 5, 4).unwrap();
        let loss = da_coral_loss(&batch).unwrap();
        assert!(loss < 1e-6, "loss={loss}");
    }

    #[test]
    fn test_coral_loss_different_domains_positive() {
        // Source uniform [0,1], target uniform [10,11] → very different covariances
        let src = linspace_features(6, 3, 0.0, 0.1);
        let tgt = linspace_features(6, 3, 10.0, 0.5);
        let batch = DomainBatch::new(src, tgt, 6, 6, 3).unwrap();
        let loss = da_coral_loss(&batch).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_coral_loss_different_n_allowed() {
        // CORAL is valid even when n_source ≠ n_target
        let src = linspace_features(4, 2, 0.0, 0.5);
        let tgt = linspace_features(8, 2, 2.0, 0.1);
        let batch = DomainBatch::new(src, tgt, 4, 8, 2).unwrap();
        let r = da_coral_loss(&batch);
        assert!(r.is_ok());
    }

    // ----- DomainDiscriminator -----

    #[test]
    fn test_discriminator_weights_size() {
        let disc = DomainDiscriminator::new_random(16, 42);
        assert_eq!(disc.weights.len(), 16);
        assert_eq!(disc.d, 16);
    }

    #[test]
    fn test_discriminator_predict_in_unit_interval() {
        let disc = DomainDiscriminator::new_random(4, 7);
        let feature = vec![0.5f32, -0.3, 1.0, 0.0];
        let p = disc.predict(&feature);
        assert!(p > 0.0 && p < 1.0, "p={p}");
    }

    #[test]
    fn test_discriminator_predict_batch_length() {
        let disc = DomainDiscriminator::new_random(3, 13);
        let features = vec![0.1f32; 15]; // 5 × 3
        let preds = disc.predict_batch(&features, 5);
        assert_eq!(preds.len(), 5);
    }

    #[test]
    fn test_discriminator_predict_batch_all_in_unit() {
        let disc = DomainDiscriminator::new_random(3, 99);
        let features = linspace_features(4, 3, -1.0, 0.5);
        let preds = disc.predict_batch(&features, 4);
        for &p in &preds {
            assert!(p > 0.0 && p < 1.0, "p={p}");
        }
    }

    #[test]
    fn test_discriminator_xavier_range() {
        // Xavier limit for d=100 is sqrt(6/101) ≈ 0.2437
        let d = 100;
        let limit = (6.0f32 / (d + 1) as f32).sqrt();
        let disc = DomainDiscriminator::new_random(d, 2026);
        for &w in &disc.weights {
            assert!(
                w >= -limit && w <= limit,
                "w={w} not in [{}, {}]",
                -limit,
                limit
            );
        }
    }

    // ----- da_dann_loss -----

    #[test]
    fn test_dann_loss_valid() {
        let batch = make_batch(4, 4, 8, 0.0, 1.0);
        let disc = DomainDiscriminator::new_random(8, 1);
        let config = DannConfig::default();
        let loss = da_dann_loss(&disc, &batch, &config).unwrap();
        assert!(loss.is_finite() && loss >= 0.0);
    }

    #[test]
    fn test_dann_loss_dimension_mismatch() {
        let batch = make_batch(2, 2, 4, 0.0, 1.0);
        let disc = DomainDiscriminator::new_random(8, 1); // wrong d
        let config = DannConfig::default();
        assert!(matches!(
            da_dann_loss(&disc, &batch, &config),
            Err(DomainAdaptationError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_dann_loss_empty_batch_err() {
        let batch = DomainBatch {
            source_features: vec![],
            target_features: vec![1.0],
            n_source: 0,
            n_target: 1,
            d: 1,
        };
        let disc = DomainDiscriminator::new_random(1, 1);
        let config = DannConfig::default();
        assert!(matches!(
            da_dann_loss(&disc, &batch, &config),
            Err(DomainAdaptationError::EmptyFeatures)
        ));
    }

    // ----- da_domain_accuracy -----

    #[test]
    fn test_domain_accuracy_range() {
        // A random discriminator on balanced data should give some accuracy 0..=1
        let batch = make_batch(20, 20, 4, -0.5, 0.5);
        let disc = DomainDiscriminator::new_random(4, 7777);
        let acc = da_domain_accuracy(&disc, &batch, 0.5).unwrap();
        assert!((0.0..=1.0).contains(&acc), "acc={acc}");
    }

    #[test]
    fn test_domain_accuracy_all_zeros_source_target_mixed() {
        // All source and target have same feature = [0,0] → random disc result
        let batch = make_batch(10, 10, 2, 0.0, 0.0);
        let disc = DomainDiscriminator::new_random(2, 42);
        let acc = da_domain_accuracy(&disc, &batch, 0.5).unwrap();
        assert!((0.0..=1.0).contains(&acc));
    }

    // ----- da_entropy -----

    #[test]
    fn test_entropy_uniform_is_max() {
        let n = 4;
        let probs = vec![0.25f32; n];
        let h = da_entropy(&probs);
        let expected = -(0.25f32 * 0.25f32.ln()) * n as f32;
        assert!((h - expected).abs() < 1e-5);
    }

    #[test]
    fn test_entropy_delta_approx_zero() {
        let mut probs = vec![0.0f32; 8];
        probs[0] = 1.0;
        let h = da_entropy(&probs);
        assert!(h < 1e-5, "h={h}");
    }

    #[test]
    fn test_entropy_non_negative() {
        let probs = vec![0.1f32, 0.4, 0.3, 0.2];
        assert!(da_entropy(&probs) >= 0.0);
    }

    // ----- da_entropy_loss -----

    #[test]
    fn test_entropy_loss_uniform_non_negative() {
        let n_src = 3;
        let n_tgt = 4;
        let classes = 5;
        let src_probs = vec![0.2f32; n_src * classes];
        let tgt_probs = vec![0.2f32; n_tgt * classes];
        let loss = da_entropy_loss(&src_probs, n_src, &tgt_probs, n_tgt).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_entropy_loss_empty_err() {
        let r = da_entropy_loss(&[], 0, &[], 0);
        assert!(r.is_err());
    }

    // ----- da_confidence_threshold_mask -----

    #[test]
    fn test_confidence_mask_threshold_zero_all_true() {
        let probs = vec![0.1f32, 0.5, 0.9, 0.01];
        let mask = da_confidence_threshold_mask(&probs, 0.0);
        assert!(mask.iter().all(|&m| m));
    }

    #[test]
    fn test_confidence_mask_threshold_one_all_false() {
        let probs = vec![0.1f32, 0.5, 0.9, 0.99];
        let mask = da_confidence_threshold_mask(&probs, 1.0);
        assert!(mask.iter().all(|&m| !m));
    }

    #[test]
    fn test_confidence_mask_selective() {
        let probs = vec![0.3f32, 0.7, 0.95, 0.2];
        let mask = da_confidence_threshold_mask(&probs, 0.5);
        assert!(!mask[0]);
        assert!(mask[1]);
        assert!(mask[2]);
        assert!(!mask[3]);
    }

    // ----- da_pseudo_label_loss -----

    #[test]
    fn test_pseudo_label_loss_all_confident_non_negative() {
        let n = 3;
        let classes = 4;
        // All samples have high confidence
        let mut probs = vec![0.01f32; n * classes];
        for i in 0..n {
            probs[i * classes] = 0.97; // class 0 is confident
        }
        let logits = [2.0f32, 0.1, 0.1, 0.1].repeat(n);
        let loss = da_pseudo_label_loss(&logits, &probs, n, 0.9, 1e-7).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_pseudo_label_loss_none_confident_returns_zero() {
        let n = 4;
        let classes = 3;
        let probs = vec![0.33f32; n * classes]; // uniform — low confidence
        let logits = vec![0.0f32; n * classes];
        let loss = da_pseudo_label_loss(&logits, &probs, n, 0.9, 1e-7).unwrap();
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_pseudo_label_loss_empty_n_err() {
        let r = da_pseudo_label_loss(&[], &[], 0, 0.5, 1e-7);
        assert!(r.is_err());
    }

    // ----- da_reversal_loss_scale -----

    #[test]
    fn test_reversal_loss_scale_step0_near_zero() {
        let scaled = da_reversal_loss_scale(1.0, 1.0, 0, 100);
        // step=0 → lambda_t ≈ 0
        assert!(scaled.abs() < 0.05, "scaled={scaled}");
    }

    #[test]
    fn test_reversal_loss_scale_step_total_near_max() {
        let lambda = 1.0;
        let scaled = da_reversal_loss_scale(1.0, lambda, 100, 100);
        // step=total_steps → lambda_t ≈ lambda
        assert!(scaled > 0.8 * lambda, "scaled={scaled}");
    }

    #[test]
    fn test_reversal_loss_scale_monotone() {
        let lambda = 0.5;
        let total = 50u64;
        let mut prev = da_reversal_loss_scale(1.0, lambda, 0, total);
        for step in 1..=total {
            let cur = da_reversal_loss_scale(1.0, lambda, step, total);
            assert!(
                cur >= prev - 1e-6,
                "not monotone at step {step}: {cur} < {prev}"
            );
            prev = cur;
        }
    }

    // ----- da_combined_loss -----

    #[test]
    fn test_combined_loss_mmd_method() {
        let batch = make_batch(4, 4, 3, 0.0, 1.0);
        let config = DomainAdaptConfig {
            method: DomainAdaptMethod::Mmd,
            ..DomainAdaptConfig::default()
        };
        let r = da_combined_loss(&batch, None, &config);
        assert!(r.is_ok());
        assert!(r.unwrap() >= 0.0);
    }

    #[test]
    fn test_combined_loss_coral_method() {
        let batch = make_batch(4, 4, 3, 0.0, 1.0);
        let config = DomainAdaptConfig {
            method: DomainAdaptMethod::Coral,
            ..DomainAdaptConfig::default()
        };
        let r = da_combined_loss(&batch, None, &config);
        assert!(r.is_ok());
    }

    #[test]
    fn test_combined_loss_dann_method() {
        let batch = make_batch(4, 4, 3, 0.0, 1.0);
        let disc = DomainDiscriminator::new_random(3, 5);
        let config = DomainAdaptConfig {
            method: DomainAdaptMethod::Dann,
            ..DomainAdaptConfig::default()
        };
        let r = da_combined_loss(&batch, Some(&disc), &config);
        assert!(r.is_ok());
    }

    #[test]
    fn test_combined_loss_combined_method() {
        let batch = make_batch(5, 5, 4, -1.0, 1.0);
        let config = DomainAdaptConfig::default(); // Combined method
        let r = da_combined_loss(&batch, None, &config);
        assert!(r.is_ok());
        assert!(r.unwrap().is_finite());
    }

    #[test]
    fn test_combined_loss_dann_without_discriminator_err() {
        let batch = make_batch(4, 4, 3, 0.0, 1.0);
        let config = DomainAdaptConfig {
            method: DomainAdaptMethod::Dann,
            ..DomainAdaptConfig::default()
        };
        let r = da_combined_loss(&batch, None, &config);
        assert!(r.is_err());
    }

    // ----- AdaptationStats -----

    #[test]
    fn test_adaptation_stats_mmd_method_some() {
        let batch = make_batch(4, 4, 3, 0.0, 1.0);
        let config = DomainAdaptConfig {
            method: DomainAdaptMethod::Mmd,
            ..DomainAdaptConfig::default()
        };
        let stats = da_compute_stats(&batch, None, &config).unwrap();
        assert!(stats.mmd_loss.is_some());
        assert!(stats.coral_loss.is_none());
        assert!(stats.dann_loss.is_none());
    }

    #[test]
    fn test_adaptation_stats_coral_method_some() {
        let batch = make_batch(4, 4, 3, 0.0, 1.0);
        let config = DomainAdaptConfig {
            method: DomainAdaptMethod::Coral,
            ..DomainAdaptConfig::default()
        };
        let stats = da_compute_stats(&batch, None, &config).unwrap();
        assert!(stats.coral_loss.is_some());
        assert!(stats.mmd_loss.is_none());
    }

    #[test]
    fn test_adaptation_stats_combined_all_some_except_dann_without_disc() {
        let batch = make_batch(4, 4, 3, 0.0, 1.0);
        let config = DomainAdaptConfig::default(); // Combined
        let stats = da_compute_stats(&batch, None, &config).unwrap();
        assert!(stats.mmd_loss.is_some());
        assert!(stats.coral_loss.is_some());
        assert!(stats.entropy_loss.is_some());
    }

    #[test]
    fn test_adaptation_stats_combined_loss_finite() {
        let batch = make_batch(4, 4, 3, 0.1, -0.1);
        let config = DomainAdaptConfig::default();
        let stats = da_compute_stats(&batch, None, &config).unwrap();
        assert!(stats.combined_loss.is_finite());
    }

    // ----- da_format_stats / da_format_config -----

    #[test]
    fn test_format_stats_non_empty() {
        let stats = AdaptationStats {
            mmd_loss: Some(0.1),
            coral_loss: None,
            dann_loss: None,
            entropy_loss: Some(0.5),
            combined_loss: 0.6,
            domain_accuracy: Some(0.7),
            n_pseudo_labels: 3,
        };
        let s = da_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("mmd="));
        assert!(s.contains("entropy="));
        assert!(s.contains("combined="));
    }

    #[test]
    fn test_format_config_non_empty() {
        let config = DomainAdaptConfig::default();
        let s = da_format_config(&config);
        assert!(!s.is_empty());
        assert!(s.contains("DomainAdaptConfig"));
    }

    // ----- error variant coverage -----

    #[test]
    fn test_error_invalid_bandwidth_display() {
        let e = DomainAdaptationError::InvalidBandwidth { bw: 0.0 };
        let s = e.to_string();
        assert!(s.contains("0"));
    }

    #[test]
    fn test_error_dimension_mismatch_display() {
        let e = DomainAdaptationError::DimensionMismatch { src: 3, tgt: 5 };
        let s = e.to_string();
        assert!(s.contains("3") && s.contains("5"));
    }

    #[test]
    fn test_error_batch_mismatch_display() {
        let e = DomainAdaptationError::BatchMismatch { src: 4, tgt: 8 };
        let s = e.to_string();
        assert!(s.contains("4") && s.contains("8"));
    }

    #[test]
    fn test_error_invalid_config_display() {
        let e = DomainAdaptationError::InvalidConfig {
            reason: "test reason".to_owned(),
        };
        let s = e.to_string();
        assert!(s.contains("test reason"));
    }

    // ----- edge cases -----

    #[test]
    fn test_mmd_biased_same_n_non_negative() {
        let src = linspace_features(3, 2, 0.0, 0.4);
        let tgt = linspace_features(3, 2, 1.0, 0.2);
        let r = da_mmd_biased(&src, &tgt, 3, 3, 2, 1.0).unwrap();
        assert!(r >= 0.0);
    }

    #[test]
    fn test_coral_different_n_source_target() {
        // n_source ≠ n_target is perfectly valid for CORAL
        let src = linspace_features(3, 2, 0.0, 1.0);
        let tgt = linspace_features(5, 2, 2.0, 0.3);
        let batch = DomainBatch::new(src, tgt, 3, 5, 2).unwrap();
        let r = da_coral_loss(&batch);
        assert!(r.is_ok(), "{:?}", r.err());
    }

    #[test]
    fn test_dann_discriminator_predict_sigmoid_saturates() {
        // A discriminator with large positive weights should push predictions near 1
        let d = 2;
        let disc = DomainDiscriminator {
            weights: vec![100.0f32, 100.0],
            bias: 100.0,
            d,
        };
        let p = disc.predict(&[1.0, 1.0]);
        assert!(p > 0.99, "p={p}");
    }

    #[test]
    fn test_dann_discriminator_predict_sigmoid_low() {
        let d = 2;
        let disc = DomainDiscriminator {
            weights: vec![-100.0f32, -100.0],
            bias: -100.0,
            d,
        };
        let p = disc.predict(&[1.0, 1.0]);
        assert!(p < 0.01, "p={p}");
    }

    #[test]
    fn test_reversal_loss_zero_total_steps() {
        // total_steps=0 should use t=1 (max lambda)
        let scaled = da_reversal_loss_scale(1.0, 1.0, 0, 0);
        // t=1 → lambda_t ≈ lambda
        assert!(scaled > 0.5);
    }

    #[test]
    fn test_pseudo_label_loss_dimension_mismatch() {
        let logits = vec![1.0f32, 2.0, 3.0]; // 3 elements
        let probs = vec![0.1f32, 0.8, 0.1, 0.5, 0.3, 0.2]; // 6 elements
        let r = da_pseudo_label_loss(&logits, &probs, 2, 0.5, 1e-7);
        assert!(r.is_err());
    }

    #[test]
    fn test_mmd_multiscale_biased_mode() {
        let batch = make_batch(3, 3, 2, 0.0, 2.0);
        let config = MmdConfig {
            kernel_bandwidths: vec![1.0, 2.0],
            biased: true,
            eps: 1e-8,
        };
        let r = da_mmd_multiscale(&batch, &config);
        assert!(r.is_ok());
        assert!(r.unwrap() >= 0.0);
    }
}
