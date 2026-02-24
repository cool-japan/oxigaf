//! # oxigaf-flame
//!
//! FLAME parametric head model implementation in Pure Rust.
//!
//! This crate implements the [FLAME (Faces Learned with an Articulated Model and Expressions)](https://flame.is.tue.mpg.de/)
//! parametric 3D head model in pure Rust, with no dependencies on Python or C/C++ libraries.
//!
//! ## Features
//!
//! - **FLAME model loading** from `.npy` files (converted from the original `.pkl`)
//! - **Linear Blend Skinning (LBS)** forward pass: parameters → posed mesh
//! - **CPU normal map rendering** for diffusion model conditioning
//! - **Mesh surface point sampling** for Gaussian initialization
//! - **Zero-cost abstractions** with extensive use of `#[inline]` for performance
//!
//! ### Cargo Features
//!
//! This crate supports the following feature flags:
//!
//! - **`simd`** (optional, requires nightly Rust with `portable_simd`):
//!   Enables SIMD-accelerated vector operations for:
//!   - Normal map rendering (3-4× faster)
//!   - Rodrigues rotation computation
//!   - Blend shape evaluation
//!
//! - **`parallel`** (optional):
//!   Enables parallel batch processing with `rayon`:
//!   - `forward_batch_par()` - parallel mesh generation
//!   - `compute_normals_batch_par()` - parallel normal computation
//!   - Near-linear speedup with CPU core count
//!
//! - **`full`** (convenience): Enables both `simd` and `parallel` for maximum performance
//!
//! Example usage:
//! ```toml
//! # In Cargo.toml
//! oxigaf-flame = { version = "0.1", features = ["parallel"] }
//! ```
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use oxigaf_flame::{FlameModel, FlameParams};
//!
//! // Load FLAME model from directory containing .npy files
//! let model = FlameModel::load("path/to/flame/model")?;
//!
//! // Create neutral parameters (zero shape, expression, pose)
//! let params = FlameParams::neutral();
//!
//! // Run forward pass to get posed mesh
//! let mesh = model.forward(&params);
//!
//! println!("Generated mesh with {} vertices", mesh.vertices.len());
//! # Ok::<(), oxigaf_flame::FlameError>(())
//! ```
//!
//! ## FLAME Parameters
//!
//! The FLAME model is controlled by several parameter types:
//!
//! - **Shape parameters** (β): Control identity-specific features (typically 100-300 coefficients)
//! - **Expression parameters** (ψ): Control facial expressions (typically 50-100 coefficients)
//! - **Pose parameters** (θ): Control joint rotations (5 joints × 3 = 15 values)
//!   - Root rotation (global head orientation)
//!   - Neck rotation
//!   - Jaw rotation
//!   - Left eye rotation
//!   - Right eye rotation
//! - **Translation**: Global 3D translation applied after posing
//!
//! ## Coordinate System
//!
//! FLAME uses a **right-handed coordinate system**:
//! - +X: Right (from the subject's perspective)
//! - +Y: Up
//! - +Z: Forward (out of the face)
//!
//! Rotations are specified as **axis-angle** vectors and converted to rotation
//! matrices using [Rodrigues' formula](https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula).
//!
//! ## Performance
//!
//! The LBS forward pass is optimized for real-time performance:
//! - ~1-2ms for standard FLAME mesh (5023 vertices) on modern CPUs
//! - Critical path functions are marked with `#[inline]`
//! - Uses `ndarray` for efficient BLAS-accelerated operations
//!
//! Run benchmarks with: `cargo bench -p oxigaf-flame`
//!
//! ## References
//!
//! - [FLAME Paper](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/400/paper.pdf)
//! - [FLAME Model](https://flame.is.tue.mpg.de/)

// Strict no-unwrap policy
#![deny(clippy::unwrap_used)]
// Allow expect in test code but deny in library code
#![cfg_attr(not(test), deny(clippy::expect_used))]
// Additional quality lints
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
// Nightly feature for portable SIMD (requires both 'simd' feature AND nightly Rust)
#![cfg_attr(all(feature = "simd", nightly), feature(portable_simd))]

pub mod conversion;
pub mod error;
pub mod io;
pub mod io_safetensors;
pub mod mesh;
pub mod model;
pub mod normal_map;
pub mod params;
mod params_builder;
pub mod sampler;
pub mod sequence;

// SIMD module (requires nightly + simd feature)
#[cfg(all(feature = "simd", nightly))]
pub mod simd;

pub use error::FlameError;
pub use mesh::Mesh;
pub use model::{
    compute_normals_batch, compute_normals_into, recompute_batch_normals, rodrigues,
    BatchBufferPool, BatchedFlameOutput, FlameModel,
};
#[cfg(feature = "parallel")]
pub use model::{compute_normals_batch_par, recompute_batch_normals_par};
pub use normal_map::{Camera, NormalMapRenderer};
pub use params::FlameParams;
pub use params_builder::FlameParamsBuilder;
pub use sampler::{sample_mesh_surface, SurfacePoint};
