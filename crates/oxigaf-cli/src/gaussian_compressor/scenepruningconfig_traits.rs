//! # `ScenePruningConfig` - Trait Implementations
//!
//! This module contains trait implementations for `ScenePruningConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ScenePruningConfig;

impl Default for ScenePruningConfig {
    fn default() -> Self {
        Self {
            opacity_threshold: 0.01,
            max_log_scale: 6.0,
            min_log_scale: -10.0,
            target_n_gaussians: None,
            preserve_top_fraction: 1.0,
        }
    }
}
