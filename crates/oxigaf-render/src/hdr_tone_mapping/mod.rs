//! HDR tone mapping for 3DGS rendered images.
//!
//! Converts high-dynamic-range float images (values may exceed \[0,1\]) to
//! low-dynamic-range displayable images with values in \[0,1\].
//!
//! This is essential when rendering 3DGS scenes with physically-based lighting
//! or when accumulating multi-view images.
//!
//! # Operators
//! - **Reinhard** / **ReinhardExtended**: simple photographic tone mapping
//! - **ACES Filmic**: cinematic S-curve approximation (Narkowicz 2015)
//! - **Hable / Uncharted 2**: game-proven filmic curve
//! - **Filmic**: John Hable's simpler film simulation
//! - **Linear**: passthrough with \[0,1\] clamp
//! - **Custom**: parameterised shadow/midtone/highlight split

pub mod analysis;
pub mod config;
pub mod curves;
pub mod error;
pub mod gamma;
pub mod operator;
pub mod pipeline;
pub mod presets;

// Re-export all types
pub use analysis::*;
pub use config::*;
pub use curves::*;
pub use error::*;
pub use gamma::*;
pub use operator::*;
pub use pipeline::*;
pub use presets::*;

#[cfg(test)]
mod tests;
