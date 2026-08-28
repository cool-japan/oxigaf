//! Per-vertex and per-face visibility culling for FLAME meshes.
//!
//! Computes which vertices and faces are visible from a given camera viewpoint:
//! front-facing, inside the camera frustum and — when
//! [`VisibilityCullerConfig::use_depth_test`] is enabled — not self-occluded.
//! Useful for the training pipeline to determine which parts of a head avatar
//! are observable from each training view.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxigaf_flame::{Mesh, normal_map::Camera};
//! use oxigaf_flame::visibility_culling::{
//!     VisibilityCullerConfig, compute_vertex_visibility, compute_visibility_stats,
//!     format_visibility_stats,
//! };
//!
//! # fn example(mesh: &Mesh, camera: &Camera) -> Result<(), oxigaf_flame::visibility_culling::VisibilityError> {
//! let config = VisibilityCullerConfig::default();
//! let vis = compute_vertex_visibility(mesh, camera, &config)?;
//! let stats = compute_visibility_stats(&vis);
//! println!("{}", format_visibility_stats(&stats));
//! # Ok(()) }
//! ```

pub mod functions;
pub mod raster;
pub mod types;

pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
