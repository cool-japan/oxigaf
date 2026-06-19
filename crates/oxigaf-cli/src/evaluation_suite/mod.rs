//! Comprehensive evaluation suite for trained OxiGAF avatar models.
//!
//! Computes PSNR, SSIM, LPIPS-approximation, MAE, RMSE, and multi-scale SSIM
//! across a test set of rendered views, then produces aggregate statistics and
//! a detailed evaluation report suitable for benchmarking model quality.
//!
//! # Design
//!
//! All images are passed as flat `&[f32]` slices in interleaved RGB order,
//! values in [0, 1] range. Length must equal `width * height * 3`.
//!
//! - No `unwrap()` or `expect()` anywhere — all fallible paths return `Result`
//! - No `rand` crate — xorshift64 is available locally if needed
//! - No `ndarray` — all math on raw `Vec<f32>` / slice arithmetic
//! - Errors typed via [`EvalError`] using `thiserror`
//!
//! # Quick Example
//!
//! ```rust
//! use oxigaf_cli::evaluation_suite::{EvalTestItem, EvalConfig, eval_suite};
//!
//! let pred = vec![0.5_f32; 64 * 64 * 3];
//! let gt   = vec![0.5_f32; 64 * 64 * 3];
//! let items = vec![EvalTestItem {
//!     view_id: "view_0".to_string(),
//!     pred,
//!     gt,
//!     width: 64,
//!     height: 64,
//! }];
//! let config = EvalConfig::default();
//! let result = eval_suite(&items, &config).expect("eval failed");
//! assert!(result.mean_psnr.is_infinite());
//! ```

pub mod evalconfig_traits;
pub mod evalmetrickind_traits;
pub mod functions;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
