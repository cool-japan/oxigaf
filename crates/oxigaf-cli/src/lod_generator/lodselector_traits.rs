//! # `LodSelector` - Trait Implementations
//!
//! This module contains trait implementations for `LodSelector`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::LodSelector;

impl Default for LodSelector {
    fn default() -> Self {
        Self {
            thresholds: vec![0.5, 2.0, 5.0],
        }
    }
}
