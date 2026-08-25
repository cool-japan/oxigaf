//! # `SplitConfig` - Trait Implementations
//!
//! This module contains trait implementations for `SplitConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{DatasetSplitStrategy, SplitConfig};

impl Default for SplitConfig {
    fn default() -> Self {
        SplitConfig {
            train_ratio: 0.8,
            val_ratio: 0.1,
            test_ratio: 0.1,
            seed: 42,
            shuffle: true,
            strategy: DatasetSplitStrategy::Random,
        }
    }
}
