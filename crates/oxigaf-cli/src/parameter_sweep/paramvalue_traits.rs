//! # `ParamValue` - Trait Implementations
//!
//! This module contains trait implementations for `ParamValue`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ParamValue;

impl std::fmt::Display for ParamValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamValue::Float(v) => write!(f, "{:.6}", v),
            ParamValue::Int(v) => write!(f, "{}", v),
            ParamValue::Choice(s) => write!(f, "{}", s),
        }
    }
}
