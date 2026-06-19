//! # `EvalMetricKind` - Trait Implementations
//!
//! This module contains trait implementations for `EvalMetricKind`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::fmt;

use super::types::EvalMetricKind;

impl fmt::Display for EvalMetricKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
