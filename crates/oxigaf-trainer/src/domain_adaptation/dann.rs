//! DANN (domain-adversarial neural networks): the linear discriminator, its
//! binary cross-entropy loss, domain accuracy and the gradient-reversal
//! coefficient schedule.

use super::batch::DomainBatch;
use super::common::{da_check_matrix, da_sigmoid, da_xorshift_range, DomainAdaptationError};

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
    ///
    /// # Errors
    ///
    /// Returns [`DomainAdaptationError::DimensionMismatch`] if
    /// `features.len() != n * self.d`.
    pub fn predict_batch(
        &self,
        features: &[f32],
        n: usize,
    ) -> Result<Vec<f32>, DomainAdaptationError> {
        da_check_matrix(features, n, self.d)?;
        Ok((0..n)
            .map(|i| self.predict(&features[i * self.d..(i + 1) * self.d]))
            .collect())
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
    da_check_matrix(&batch.source_features, batch.n_source, batch.d)?;
    da_check_matrix(&batch.target_features, batch.n_target, batch.d)?;

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
    da_check_matrix(&batch.source_features, batch.n_source, batch.d)?;
    da_check_matrix(&batch.target_features, batch.n_target, batch.d)?;

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
