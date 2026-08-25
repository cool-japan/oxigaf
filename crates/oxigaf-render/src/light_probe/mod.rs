//! Spherical harmonics light probes and image-based lighting for 3DGS avatar rendering.
//!
//! This module provides:
//! - Real SH basis functions (L=0,1,2) via `lp_sh_basis_l0/l1/l2`
//! - `IrradianceSH`: 27-coefficient (9 basis × 3 channels) SH representation
//! - `CubemapProbe`: 6-face cubemap with bilinear sampling
//! - `LightProbe`: positional probe with influence radius
//! - `LightProbeBlend`: multi-probe weighted blending
//! - Diffuse IBL evaluation via `lp_evaluate_diffuse_ibl` / `lp_apply_ibl_to_gaussians`
//!
//! ## Coefficient layout (IrradianceSH)
//!
//! Interleaved RGB: `coefficients[basis_i * 3 + channel]`
//! for `basis_i` in `0..9` and `channel` in `{0=R, 1=G, 2=B}`.
//!
//! ## xorshift64 PRNG
//!
//! Monte Carlo sphere sampling uses xorshift64 to avoid an external `rand`
//! dependency. One step of the generator is:
//!
//! ```
//! let mut state: u64 = 0x2545_F491_4F6C_DD1D;
//! state ^= state << 13;
//! state ^= state >> 7;
//! state ^= state << 17;
//! if state == 0 {
//!     state = 1;
//! }
//! assert_ne!(state, 0, "the generator must never latch to the zero state");
//! ```
//!
//! Implementation split across private submodules (`error`, `sh_math`,
//! `irradiance`, `sampling`, `projection`, `cubemap`, `probe`, `blend`,
//! `ibl`, `config`); every item below is re-exported at this module's root
//! unchanged, so the split is purely internal and does not move any public
//! path.

mod blend;
mod config;
mod cubemap;
mod error;
mod ibl;
mod irradiance;
mod probe;
mod projection;
mod sampling;
mod sh_math;

pub use blend::{lp_blend_irradiance_sh, LightProbeBlend, ProbeBlendMode};
pub use config::{
    lp_compute_stats, lp_format_config, lp_format_stats, LightProbeConfig, LightProbeStats,
};
pub use cubemap::{
    lp_cubemap_to_sh, lp_cubemap_to_sh_with_config, lp_dir_to_cubemap_uv, CubemapFace, CubemapProbe,
};
pub use error::LightProbeError;
pub use ibl::{lp_apply_ibl_to_gaussians, lp_evaluate_diffuse_ibl};
pub use irradiance::IrradianceSH;
pub use probe::LightProbe;
pub use projection::{
    lp_generate_sphere_samples, lp_project_latitude_longitude,
    lp_project_latitude_longitude_with_config, lp_project_samples_to_sh,
};
pub use sh_math::{lp_normalize_dir, lp_sh_basis, lp_sh_basis_l0, lp_sh_basis_l1, lp_sh_basis_l2};

#[cfg(test)]
mod tests;
