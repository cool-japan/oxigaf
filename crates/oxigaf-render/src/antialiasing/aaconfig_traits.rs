//! # `AaConfig` - Trait Implementations
//!
//! This module contains trait implementations for `AaConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{AaConfig, AaMethod};

impl Default for AaConfig {
    fn default() -> Self {
        Self {
            method: AaMethod::Fxaa,
            edge_threshold: 0.0833,
            edge_threshold_min: 0.0312,
            subpixel_quality: 0.75,
            search_steps: 12,
        }
    }
}
