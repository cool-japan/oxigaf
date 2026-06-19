//! # `RetargetConfig` - Trait Implementations
//!
//! This module contains trait implementations for `RetargetConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::RetargetConfig;

impl Default for RetargetConfig {
    fn default() -> Self {
        Self {
            expr_dim: 50,
            regularization: 1e-4_f32,
            scale_by_variance: true,
            include_jaw: true,
            smoothing_sigma: 2.0_f32,
        }
    }
}
