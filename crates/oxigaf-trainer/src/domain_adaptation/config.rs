//! Method selector, the full [`DomainAdaptConfig`] and the combined loss that
//! dispatches on it.

use super::batch::DomainBatch;
use super::common::DomainAdaptationError;
use super::coral::da_coral_loss;
use super::dann::{da_dann_loss, da_reversal_loss_scale, DannConfig, DomainDiscriminator};
use super::mmd::{da_mmd_multiscale, MmdConfig};
use super::self_training::da_entropy;

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
    /// Whether the gradient-reversal coefficient λ ramps up over training
    /// (Ganin et al.'s progressive schedule) instead of staying pinned at
    /// [`DannConfig::lambda`].
    ///
    /// Read by [`DomainAdaptConfig::effective_lambda`], and through it by
    /// [`da_scaled_dann_loss`] and [`da_combined_loss_at_step`].
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

impl DomainAdaptConfig {
    /// The gradient-reversal coefficient λ in force at `step` of
    /// `total_steps`.
    ///
    /// With [`dann_lambda_schedule`](Self::dann_lambda_schedule) enabled this
    /// is Ganin et al.'s progressive ramp
    /// `λ_t = 2λ / (1 + exp(-10·step/total_steps)) − λ`, which starts at ≈ 0
    /// so the discriminator is not fought before it has learned anything, and
    /// approaches `dann.lambda` by the end of training. With the schedule
    /// disabled, λ is the constant [`DannConfig::lambda`] at every step.
    pub fn effective_lambda(&self, step: u64, total_steps: u64) -> f32 {
        if self.dann_lambda_schedule {
            // `da_reversal_loss_scale(1.0, λ, …)` is exactly λ_t.
            da_reversal_loss_scale(1.0, self.dann.lambda, step, total_steps)
        } else {
            self.dann.lambda
        }
    }
}

/// DANN discriminator loss scaled by the gradient-reversal coefficient in
/// force at `step` (see [`DomainAdaptConfig::effective_lambda`]).
///
/// This is the value a training loop should actually add to its objective:
/// the raw [`da_dann_loss`] carries no λ at all, so using it directly ignores
/// the configured reversal strength *and* its schedule.
pub fn da_scaled_dann_loss(
    discriminator: &DomainDiscriminator,
    batch: &DomainBatch,
    config: &DomainAdaptConfig,
    step: u64,
    total_steps: u64,
) -> Result<f32, DomainAdaptationError> {
    let loss = da_dann_loss(discriminator, batch, &config.dann)?;
    Ok(loss * config.effective_lambda(step, total_steps))
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
///
/// The `Dann` term here is *unscaled* — it carries no gradient-reversal
/// coefficient, because this function has no notion of where training is. Use
/// [`da_combined_loss_at_step`] to apply the configured λ (and its schedule,
/// when [`DomainAdaptConfig::dann_lambda_schedule`] is set).
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

/// [`da_combined_loss`] with the DANN term scaled by the gradient-reversal
/// coefficient in force at `step` of `total_steps`.
///
/// Only [`DomainAdaptMethod::Dann`] carries a λ, so every other method returns
/// exactly what [`da_combined_loss`] does. This is the entry point a training
/// loop should call once per step: it is what actually gives
/// [`DomainAdaptConfig::dann_lambda_schedule`] an effect on the objective,
/// ramping the adversarial pressure up instead of applying it at full strength
/// from step 0.
pub fn da_combined_loss_at_step(
    batch: &DomainBatch,
    discriminator: Option<&DomainDiscriminator>,
    config: &DomainAdaptConfig,
    step: u64,
    total_steps: u64,
) -> Result<f32, DomainAdaptationError> {
    match config.method {
        DomainAdaptMethod::Dann => {
            let disc = discriminator.ok_or_else(|| DomainAdaptationError::InvalidConfig {
                reason: "DANN requires a DomainDiscriminator".to_owned(),
            })?;
            da_scaled_dann_loss(disc, batch, config, step, total_steps)
        }
        _ => da_combined_loss(batch, discriminator, config),
    }
}
