//! Expression retargeting between FLAME identities (linear mapper).
//!
//! Transfers facial expressions from a source identity's FLAME parameter space
//! to a target identity's FLAME parameter space. Since different people have
//! different face shapes, the same expression parameters may look visually
//! different on different identities. This module learns (or assumes) a linear
//! mapping M from source expression space to target expression space and applies
//! it frame-by-frame.
//!
//! # Overview
//!
//! - [`LinearExpressionRetargeter`]: learns an (`expr_dim` × `expr_dim`) affine map
//!   M from source→target expression pairs.
//! - [`retar_compute_variance`], [`retar_standardize`], [`retar_unstandardize`]:
//!   expression space statistics and normalization helpers.
//! - Trajectory analysis: velocity, acceleration, smoothing, resampling.
//! - Blending utilities: weighted blend, component-wise SLERP.
//! - [`retar_compute_stats`], [`retar_format_stats`], [`retar_format_config`].
//!
//! # Example
//!
//! ```rust
//! use oxigaf_flame::expression_retargeting::{
//!     ExpressionState, RetargetPair, RetargetConfig, LinearExpressionRetargeter,
//! };
//!
//! let config = RetargetConfig { expr_dim: 4, ..Default::default() };
//! let pairs = vec![
//!     RetargetPair {
//!         source: ExpressionState::neutral(4),
//!         target: ExpressionState::neutral(4),
//!     },
//!     RetargetPair {
//!         source: ExpressionState::from_params(vec![1.0, 0.0, 0.0, 0.0]),
//!         target: ExpressionState::from_params(vec![1.0, 0.0, 0.0, 0.0]),
//!     },
//! ];
//! let retargeter = LinearExpressionRetargeter::fit(&pairs, config).unwrap();
//! let result = retargeter.retarget(&ExpressionState::from_params(vec![0.5, 0.0, 0.0, 0.0])).unwrap();
//! ```

pub mod functions;
pub mod retargetconfig_traits;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
