//! # `DatasetScanner` - Trait Implementations
//!
//! This module contains trait implementations for `DatasetScanner`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::DatasetScanner;

impl Default for DatasetScanner {
    fn default() -> Self {
        DatasetScanner {
            extensions: vec![],
            recursive: true,
            min_size_bytes: 0,
            max_size_bytes: 0,
        }
    }
}
