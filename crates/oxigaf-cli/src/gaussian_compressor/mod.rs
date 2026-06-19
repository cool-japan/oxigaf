//! Gaussian scene compression: quantization, pruning, and clustering.
//!
//! This module compresses 3D Gaussian Splatting scenes to reduce model size
//! via scalar quantization, opacity-based pruning, and optional k-means
//! position clustering. Useful for deploying trained avatar models to
//! mobile/web platforms where memory is constrained.
//!
//! # Example
//! ```rust
//! use oxigaf_cli::gaussian_compressor::{
//!     CompressionConfig, QuantizationPrecision, GcSceneSlices, gc_compress, gc_decompress,
//! };
//!
//! let n = 10usize;
//! let positions: Vec<f32> = (0..n * 3).map(|i| i as f32 * 0.1).collect();
//! let rotations: Vec<f32> = (0..n * 4)
//!     .map(|i| if i % 4 == 3 { 1.0f32 } else { 0.0 })
//!     .collect();
//! let scales: Vec<f32> = vec![-1.0f32; n * 3];
//! let opacities: Vec<f32> = vec![2.0f32; n]; // logit space → sigmoid ≈ 0.88
//! let sh_dc: Vec<f32> = vec![0.0f32; n * 3];
//! let sh_rest: Vec<f32> = vec![];
//!
//! let config = CompressionConfig::default();
//! let scene = gc_compress(
//!     GcSceneSlices {
//!         positions: &positions, rotations: &rotations, scales: &scales,
//!         opacities: &opacities, sh_dc: &sh_dc, sh_rest: &sh_rest, n_rest_per_gaussian: 0,
//!     },
//!     &config,
//! ).expect("compression failed");
//! println!("Compression ratio: {:.2}x", scene.compression_ratio());
//! ```

pub mod compressionconfig_traits;
pub mod functions;
pub mod kmeansconfig_traits;
pub mod scenepruningconfig_traits;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
