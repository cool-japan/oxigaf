//! Hyperparameter sweep management for 3DGS training.
//!
//! Provides grid search, random search, and a simplified pseudo-Bayesian surrogate
//! search for tuning 3D Gaussian Splatting training hyperparameters.
//!
//! # Example
//! ```rust
//! use oxigaf_cli::parameter_sweep::{
//!     ParamSpec, SweepConfig, SweepStrategy, ParameterSweep,
//! };
//!
//! let config = SweepConfig {
//!     specs: vec![
//!         ParamSpec::Continuous { name: "lr".into(), low: 1e-4, high: 1e-2, log_scale: true },
//!         ParamSpec::Discrete { name: "n_sh".into(), values: vec![1.0, 4.0, 9.0] },
//!     ],
//!     strategy: SweepStrategy::Random,
//!     max_trials: 20,
//!     seed: 42,
//!     minimize: true,
//! };
//!
//! let mut sweep = ParameterSweep::new(config).expect("Failed to create sweep");
//! let trial = sweep.suggest().expect("Failed to suggest trial");
//! println!("Trial {}: {}", trial.id, oxigaf_cli::parameter_sweep::format_sweep_trial(&trial));
//! ```

pub mod functions;
pub mod paramvalue_traits;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
