//! CPU ray-marching volumetric renderer for the OxiGAF render crate.
//!
//! Complements the GPU 3DGS rasterizer with a CPU-side volume renderer that
//! supports arbitrary transfer functions, multiple integration modes and
//! empty-space skipping.
//!
//! # Quick start
//! ```ignore
//! let mut grid = VolumeGrid::new(64, 64, 64, [-1.0;3], [1.0/32.0;3]);
//! // fill grid.data …
//! let tf = TransferFunction::grayscale(1.0);
//! let cam = VolumetricCamera::default_front(256, 256);
//! let cfg = VolumetricRenderConfig::default();
//! let rgba = vr_render_image(&grid, &tf, &cam, &cfg).unwrap();
//! ```

pub mod functions;
pub mod raymarchresult_traits;
pub mod types;
pub mod volumetricrenderconfig_traits;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
