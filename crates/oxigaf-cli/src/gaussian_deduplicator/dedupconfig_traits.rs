//! # `DedupConfig` - Trait Implementations
//!
//! This module contains trait implementations for `DedupConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{DedupConfig, DedupKeepPolicy};

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            position_threshold: 0.001,
            opacity_threshold: 0.05,
            scale_threshold: 0.1,
            color_threshold: 0.1,
            keep_policy: DedupKeepPolicy::KeepHighestOpacity,
            use_spatial_hash: true,
            cell_size: 0.002,
        }
    }
}
