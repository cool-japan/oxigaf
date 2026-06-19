//! Geometric analysis and transformation utilities for 3D Gaussian Splatting clouds.
//!
//! This module provides operations on Gaussian point clouds stored as flat
//! `Vec<f32>` arrays: computing bounding volumes, applying rigid transforms,
//! computing spatial statistics, and performing geometric filtering.
//!
//! # Conventions
//! - `positions`: flat `[x0, y0, z0, x1, y1, z1, …]` — length must be divisible by 3.
//! - `rotations`: flat `[qx0, qy0, qz0, qw0, …]` — length must be divisible by 4 (w-last).
//! - `scales`: flat log-scale `[ls0x, ls0y, ls0z, …]` — actual scale = `exp(ls)`.
//!
//! # Example
//! ```rust,no_run
//! use oxigaf_cli::geometry_tools::{compute_gaussian_bbox, compute_geometry_stats};
//!
//! let positions = vec![0.0_f32, 0.0, 0.0,  1.0, 1.0, 1.0,  -1.0, -1.0, -1.0];
//! let scales = vec![-2.0_f32; 9];
//! let bbox = compute_gaussian_bbox(&positions).expect("bbox failed");
//! println!("BBox center: {:?}", bbox.center());
//! let stats = compute_geometry_stats(&positions, &scales).expect("stats failed");
//! println!("{}", stats.format_summary());
//! ```

pub mod functions;
pub mod type_aliases;
pub mod types;

// Re-export all types
pub use functions::*;
pub use type_aliases::*;
pub use types::*;

#[cfg(test)]
mod tests;
