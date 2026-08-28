//! Near-duplicate Gaussian detection and removal for 3D Gaussian Splatting scenes.
//!
//! Detects and removes near-duplicate Gaussians based on spatial proximity,
//! scale similarity, opacity similarity, and DC color similarity. Duplicate
//! Gaussians waste GPU memory and cause rendering artifacts from double-counted
//! opacity.
//!
//! Uses spatial hashing for O(N) average-case detection or O(N²) brute force
//! for small scenes or ground-truth verification.

pub mod dedupconfig_traits;
pub mod functions;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
