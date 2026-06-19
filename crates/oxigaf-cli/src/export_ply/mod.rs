//! PLY file I/O for 3D Gaussian Splatting scenes.
//!
//! PLY is the de facto standard format for 3DGS — used by the original
//! Kerbl et al. implementation and all major viewers (SuperSplat, Luma AI,
//! Polycam). This module implements both read and write paths with full
//! round-trip fidelity.
//!
//! # Property layout
//!
//! Every Gaussian vertex in the PLY file has these properties (in order):
//!
//! | Property      | Type  | Description                              |
//! |---------------|-------|------------------------------------------|
//! | x, y, z       | float | World-space position                     |
//! | nx, ny, nz    | float | Normal (always 0 — required by PLY spec) |
//! | f_dc_0/1/2    | float | SH DC coefficients (3 channels)          |
//! | f_rest_0…N    | float | SH higher-order rest coefficients        |
//! | opacity       | float | Logit-space opacity (sigmoid → \[0,1\])  |
//! | scale_0/1/2   | float | Log-space scales (exp → real scale)      |
//! | rot_0/1/2/3   | float | Quaternion wxyz                           |
//!
//! # SH rest coefficient count
//!
//! ```text
//! degree 0 →  0 rest coefficients
//! degree 1 →  9 rest coefficients  (= (1+1)^2 × 3 - 3)
//! degree 2 → 24 rest coefficients  (= (2+1)^2 × 3 - 3)
//! degree 3 → 45 rest coefficients  (= (3+1)^2 × 3 - 3)
//! ```

pub mod functions;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests;
