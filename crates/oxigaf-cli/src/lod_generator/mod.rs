//! Level-of-Detail (LOD) generation for 3D Gaussian Splatting clouds.
//!
//! This module creates multiple resolution variants of a Gaussian scene
//! that can be selected based on viewing distance. Lower LOD levels have
//! fewer Gaussians but remain perceptually similar by retaining the most
//! opaque/visible Gaussians.
//!
//! # Example
//! ```rust
//! use oxigaf_cli::lod_generator::{LodConfig, LodStrategy, generate_lod_chain};
//!
//! let n = 100usize;
//! let positions: Vec<f32> = (0..n * 3).map(|i| i as f32 * 0.01).collect();
//! let rotations: Vec<f32> = (0..n * 4)
//!     .map(|i| if i % 4 == 3 { 1.0 } else { 0.0 })
//!     .collect();
//! let scales: Vec<f32> = vec![0.1f32; n * 3];
//! let opacities: Vec<f32> = vec![0.5f32; n];
//! let sh_coefficients: Vec<f32> = vec![0.0f32; n * 9];
//!
//! let config = LodConfig::default();
//! let chain = generate_lod_chain(
//!     &positions, &rotations, &scales, &opacities, &sh_coefficients, &config,
//! ).expect("LOD generation failed");
//! println!("Generated {} LOD levels", chain.levels.len());
//! ```

pub mod functions;
pub mod lodconfig_traits;
pub mod lodselector_traits;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
