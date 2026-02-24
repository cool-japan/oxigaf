//! OxiGAF CLI library interface.
//!
//! This module exposes internal functionality for integration testing.

pub mod assets;
pub mod cache;
pub mod config;
pub mod interactive;
pub mod json_output;
pub mod log_rotation;
pub mod metrics;
pub mod progress;
pub mod stages;
pub mod verbosity;

// Re-export for tests
pub use interactive::InteractiveController;
