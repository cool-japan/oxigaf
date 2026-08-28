//! Diff tool for comparing two Gaussian avatar model snapshots.
//!
//! This module compares two versions of a Gaussian avatar model
//! (snapshots/checkpoints) to analyze training progress, detect regressions,
//! and understand what changed between training steps. It operates on flat
//! float arrays of Gaussian parameters.
//!
//! # Example
//! ```rust,no_run
//! use oxigaf_cli::diff_tool::{
//!     ModelSnapshot, DiffConfig, diff_models, format_model_diff,
//! };
//!
//! let a = ModelSnapshot::new(
//!     "step_0", 0,
//!     vec![0.0f32; 300],   // 100 Gaussians * 3
//!     vec![0.0f32; 100],
//!     vec![0.0f32; 300],
//!     vec![0.5f32; 300],
//! ).expect("valid snapshot");
//! let config = DiffConfig::default();
//! let diff = diff_models(&a, &a, &config).expect("diff ok");
//! println!("{}", format_model_diff(&diff));
//! ```

pub mod diffconfig_traits;
pub mod functions;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
