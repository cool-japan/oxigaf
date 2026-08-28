//! # `LodConfig` - Trait Implementations
//!
//! This module contains trait implementations for `LodConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{LodConfig, LodStrategy};

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            n_levels: 4,
            reduction_ratios: vec![1.0, 0.5, 0.25, 0.1],
            strategy: LodStrategy::TopOpacity,
            sort_by_opacity: true,
        }
    }
}
