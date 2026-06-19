//! # `VolumetricRenderConfig` - Trait Implementations
//!
//! This module contains trait implementations for `VolumetricRenderConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{VolumetricIntegration, VolumetricRenderConfig};

impl Default for VolumetricRenderConfig {
    fn default() -> Self {
        Self {
            step_size: 0.01,
            max_steps: 1000,
            early_termination_alpha: 0.99,
            integration: VolumetricIntegration::FrontToBack,
            jitter: false,
            jitter_seed: 0x_dead_beef_cafe_babe,
        }
    }
}
