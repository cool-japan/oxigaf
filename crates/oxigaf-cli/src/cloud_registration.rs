//! Point cloud registration: align two Gaussian clouds via ICP, with each step
//! solved in closed form by the Umeyama similarity estimator.
//!
//! Finds the optimal rigid transform (rotation + translation + optional uniform scale)
//! that aligns a source cloud to a target cloud.
//!
//! # Example
//! ```rust,no_run
//! use oxigaf_cli::cloud_registration::{register_point_clouds, RegistrationConfig};
//!
//! let source = vec![0.0f32, 0.0, 0.0,  1.0, 0.0, 0.0,  0.0, 1.0, 0.0];
//! let target = vec![1.0f32, 0.0, 0.0,  2.0, 0.0, 0.0,  1.0, 1.0, 0.0];
//! let cfg = RegistrationConfig::default();
//! match register_point_clouds(&source, &target, &cfg) {
//!     Ok(result) => println!("RMSE: {:.4e}", result.final_rmse),
//!     Err(e) => eprintln!("registration failed: {e}"),
//! }
//! ```
//!
//! # Layout
//!
//! The implementation is split across private submodules, all re-exported here
//! so this module stays the single import path for callers:
//!
//! - `types` — the public error, transform, config, result and stats carriers.
//! - `math` — fixed-size matrix/vector/quaternion helpers and the symmetric
//!   4×4 eigen-solver behind the closed-form rotation.
//! - `kdtree` — the target-cloud index that makes nearest-neighbour search
//!   logarithmic instead of linear.
//! - `correspondence` — centroids, nearest-neighbour matching, outlier
//!   rejection and the pre-registration RMSE.
//! - `icp` — the closed-form estimator, one ICP iteration, and the driver that
//!   runs iterations to convergence.

mod correspondence;
mod icp;
mod kdtree;
mod math;
mod types;

#[cfg(test)]
mod test_support;

pub use correspondence::{
    compute_centroid_3d, compute_initial_rmse, filter_correspondences, find_correspondences,
};
pub use icp::{
    apply_registration_transform, compute_registration_stats, estimate_transform_umeyama,
    format_registration_result, icp_step, register_point_clouds, subsample_positions,
};
pub use types::{
    Correspondence, RegistrationConfig, RegistrationError, RegistrationResult, RegistrationStats,
    RegistrationTransform,
};

/// Former name of [`estimate_transform_umeyama`], kept so the crate root's
/// re-export list keeps resolving.
///
/// The `_approx` suffix is a misnomer: the estimator is the exact closed-form
/// Umeyama solution, not an approximation of it. Switch to the unsuffixed name;
/// this alias exists only until the crate root's `pub use` list is updated, and
/// is then removed.
pub use icp::estimate_transform_umeyama as estimate_transform_umeyama_approx;
