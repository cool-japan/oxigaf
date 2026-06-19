//! Mip-Splatting-style anti-aliasing for 3D Gaussian Splatting.
//!
//! When Gaussians are viewed from far away their projected 2-D footprint becomes
//! smaller than a pixel, producing aliasing artifacts.  This module implements
//! the key ideas from Mip-Splatting:
//!
//! 1. **Screen-space radius** — perspective projection of a 3-D Gaussian scale
//!    onto the image plane (pixels).
//! 2. **Scale compensation** — if the projected radius is smaller than a
//!    configurable minimum (`min_2d_radius_px`), scale the Gaussian up so its
//!    screen-space footprint equals that minimum.
//! 3. **Opacity ramp** — linearly fade Gaussians whose projected radius falls
//!    below `opacity_ramp_max_px`, completely suppressing those smaller than
//!    `opacity_ramp_min_px`.
//!
//! All arithmetic is CPU-side pure Rust with no GPU or unsafe code.
//!
//! ## Opacity storage convention
//!
//! Gaussian opacities are stored in **logit space**: `opacity_logit = logit(p)`
//! where `p = sigmoid(opacity_logit)`.  When a scale factor `s ∈ [0, 1]` is
//! applied the stored value becomes `logit(sigmoid(opacity_logit) * s)`.

pub mod aaconfig_traits;
pub mod functions;
pub mod mipsplatconfig_traits;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
