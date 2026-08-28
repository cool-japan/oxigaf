//! Per-step [`AdaptationStats`] collection and human-readable formatting.

use super::batch::DomainBatch;
use super::common::DomainAdaptationError;
use super::config::{da_combined_loss, DomainAdaptConfig, DomainAdaptMethod};
use super::coral::da_coral_loss;
use super::dann::{da_dann_loss, da_domain_accuracy, DomainDiscriminator};
use super::mmd::da_mmd_multiscale;
use super::self_training::da_entropy;

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

    // Estimate pseudo-label count from target: treat each sample's feature
    // vector as unnormalized logits (matching the softmax convention used
    // for `entropy_loss` above) and count samples whose maximum softmax
    // probability exceeds `confidence_threshold`.
    let n_pseudo_labels = {
        let d = batch.d;
        (0..batch.n_target)
            .filter(|&i| {
                let slice = &batch.target_features[i * d..(i + 1) * d];
                let max_v = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = slice.iter().map(|&v| (v - max_v).exp()).sum();
                // The max-logit term contributes exp(0) = 1 to `exp_sum`, so
                // the softmax probability of the argmax class is exactly
                // 1/exp_sum.
                let soft_max = 1.0 / exp_sum;
                soft_max > config.confidence_threshold
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
