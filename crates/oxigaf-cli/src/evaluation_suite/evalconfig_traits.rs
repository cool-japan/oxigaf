//! # `EvalConfig` - Trait Implementations
//!
//! This module contains trait implementations for `EvalConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{EvalConfig, EvalMetricKind};

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            metrics: vec![
                EvalMetricKind::Psnr,
                EvalMetricKind::Ssim,
                EvalMetricKind::LpipsApprox,
                EvalMetricKind::Mae,
                EvalMetricKind::Rmse,
                EvalMetricKind::SsimMs,
            ],
            save_per_view_results: false,
            n_worst_views: 5,
            n_best_views: 5,
        }
    }
}
