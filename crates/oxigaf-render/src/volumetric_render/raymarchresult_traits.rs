//! # `RayMarchResult` - Trait Implementations
//!
//! This module contains trait implementations for `RayMarchResult`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::RayMarchResult;

impl Default for RayMarchResult {
    fn default() -> Self {
        Self {
            color: [0.0; 3],
            alpha: 0.0,
            n_steps: 0,
            t_entry: 0.0,
            t_exit: 0.0,
        }
    }
}
