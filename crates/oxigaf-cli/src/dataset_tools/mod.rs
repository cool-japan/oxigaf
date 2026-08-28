//! Dataset management utilities for OxiGAF training pipelines.
//!
//! This module provides tools for scanning training data directories,
//! creating reproducible train/val/test splits, computing dataset statistics,
//! and validating dataset structure.
//!
//! # Example
//! ```rust,no_run
//! use oxigaf_cli::dataset_tools::{
//!     DatasetScanner, SplitConfig, split_dataset, compute_dataset_stats,
//!     apply_split, validate_dataset,
//! };
//! use std::path::Path;
//!
//! let dir = Path::new("/data/avatars");
//! let stats = validate_dataset(dir, 10).expect("valid dataset");
//! println!("{}", stats.format_summary());
//! ```

pub mod constants;
pub mod datasetscanner_traits;
pub mod functions;
pub mod splitconfig_traits;
pub mod type_aliases;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
