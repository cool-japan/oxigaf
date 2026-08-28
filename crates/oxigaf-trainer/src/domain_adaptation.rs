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

mod batch;
mod common;
mod config;
mod coral;
mod dann;
mod mmd;
mod self_training;
mod stats;

#[cfg(test)]
mod tests;

pub use batch::DomainBatch;
pub use common::DomainAdaptationError;
pub use config::{
    da_combined_loss, da_combined_loss_at_step, da_scaled_dann_loss, DomainAdaptConfig,
    DomainAdaptMethod,
};
pub use coral::{
    da_center_features, da_coral_loss, da_covariance, da_feature_mean, da_frobenius_sq,
};
pub use dann::{
    da_dann_loss, da_domain_accuracy, da_reversal_loss_scale, DannConfig, DomainDiscriminator,
};
pub use mmd::{
    da_gaussian_kernel, da_median_bandwidth, da_mmd_biased, da_mmd_multiscale, da_mmd_unbiased,
    MmdConfig,
};
pub use self_training::{
    da_confidence_threshold_mask, da_entropy, da_entropy_loss, da_pseudo_label_loss,
};
pub use stats::{da_compute_stats, da_format_config, da_format_stats, AdaptationStats};
