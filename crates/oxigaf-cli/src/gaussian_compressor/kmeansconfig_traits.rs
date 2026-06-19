//! # `KMeansConfig` - Trait Implementations
//!
//! This module contains trait implementations for `KMeansConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::KMeansConfig;

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 256,
            n_iterations: 50,
            tolerance: 1e-4,
        }
    }
}
