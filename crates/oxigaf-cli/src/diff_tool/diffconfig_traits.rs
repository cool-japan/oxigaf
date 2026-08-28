//! # `DiffConfig` - Trait Implementations
//!
//! This module contains trait implementations for `DiffConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::DiffConfig;

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-6,
            normalize: false,
            include_inactive: true,
            match_radius: 0.5,
        }
    }
}
