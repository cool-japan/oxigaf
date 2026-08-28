//! # Rigid and Similarity Alignment
//!
//! Implements rigid (and similarity) alignment of 3D head meshes via:
//! - Closed-form Procrustes (Umeyama algorithm) using 3×3 Jacobi SVD
//! - Iterative Closest Point (ICP) alignment
//! - Landmark-based (optionally weighted) alignment
//! - Nearest-neighbour search: brute-force public API; `align_icp` uses a
//!   `kiddo` KD-tree internally
//!
//! All matrix operations are manual (no ndarray).
//!
//! ## References
//! - Umeyama, "Least-squares estimation of transformation parameters …", 1991.
//! - Besl & `McKay`, "A method for registration of 3-D shapes", 1992.

pub mod icp;
pub mod landmarks;
pub mod procrustes;
pub mod svd;
pub mod types;

pub use icp::*;
pub use landmarks::*;
pub use procrustes::*;
pub(crate) use svd::svd_3x3;
pub use types::*;

#[cfg(test)]
mod tests;
