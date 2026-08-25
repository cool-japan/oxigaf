//! 6-face HDR cubemap: face identification, direction-to-UV projection, and
//! SH projection.

use super::config::LightProbeConfig;
use super::error::LightProbeError;
use super::irradiance::IrradianceSH;
use super::projection::{lp_generate_sphere_samples, lp_project_samples_to_sh};
use super::sampling::bilinear_sample_rgb;
use super::sh_math::lp_normalize_dir;

// ---------------------------------------------------------------------------
// Cubemap
// ---------------------------------------------------------------------------

/// Identifies one of the 6 cubemap faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum CubemapFace {
    /// +X face
    PosX = 0,
    /// -X face
    NegX = 1,
    /// +Y face
    PosY = 2,
    /// -Y face
    NegY = 3,
    /// +Z face
    PosZ = 4,
    /// -Z face
    NegZ = 5,
}

/// Convert a direction vector to a cubemap face and (u, v) in [0, 1]².
///
/// Selects the face with the largest absolute coordinate component.
pub fn lp_dir_to_cubemap_uv(dir: [f32; 3]) -> (CubemapFace, f32, f32) {
    let [x, y, z] = dir;
    let ax = x.abs();
    let ay = y.abs();
    let az = z.abs();

    if ax >= ay && ax >= az {
        // ±X dominant
        if x >= 0.0 {
            let inv = 0.5 / ax;
            let u = (-z * inv + 0.5).clamp(0.0, 1.0);
            let v = (-y * inv + 0.5).clamp(0.0, 1.0);
            (CubemapFace::PosX, u, v)
        } else {
            let inv = 0.5 / ax;
            let u = (z * inv + 0.5).clamp(0.0, 1.0);
            let v = (-y * inv + 0.5).clamp(0.0, 1.0);
            (CubemapFace::NegX, u, v)
        }
    } else if ay >= ax && ay >= az {
        // ±Y dominant
        if y >= 0.0 {
            let inv = 0.5 / ay;
            let u = (x * inv + 0.5).clamp(0.0, 1.0);
            let v = (z * inv + 0.5).clamp(0.0, 1.0);
            (CubemapFace::PosY, u, v)
        } else {
            let inv = 0.5 / ay;
            let u = (x * inv + 0.5).clamp(0.0, 1.0);
            let v = (-z * inv + 0.5).clamp(0.0, 1.0);
            (CubemapFace::NegY, u, v)
        }
    } else {
        // ±Z dominant
        if z >= 0.0 {
            let inv = 0.5 / az;
            let u = (x * inv + 0.5).clamp(0.0, 1.0);
            let v = (-y * inv + 0.5).clamp(0.0, 1.0);
            (CubemapFace::PosZ, u, v)
        } else {
            let inv = 0.5 / az;
            let u = (-x * inv + 0.5).clamp(0.0, 1.0);
            let v = (-y * inv + 0.5).clamp(0.0, 1.0);
            (CubemapFace::NegZ, u, v)
        }
    }
}

/// 6-face HDR cubemap.
///
/// Each face is stored as a flat RGB f32 array of size `resolution × resolution × 3`.
#[derive(Debug, Clone)]
pub struct CubemapProbe {
    /// 6 faces: index by `CubemapFace as usize`.
    pub faces: Vec<Vec<f32>>,
    /// Side length (must be a power of 2 and >= 4).
    pub resolution: u32,
}

impl CubemapProbe {
    /// Create a black cubemap with the given resolution.
    ///
    /// # Errors
    /// - `LightProbeError::InvalidResolution` if `resolution` is not a power of 2 or < 4.
    pub fn new(resolution: u32) -> Result<Self, LightProbeError> {
        if resolution < 4 || !resolution.is_power_of_two() {
            return Err(LightProbeError::InvalidResolution { res: resolution });
        }
        let face_len = (resolution as usize) * (resolution as usize) * 3;
        let faces = vec![vec![0.0_f32; face_len]; 6];
        Ok(Self { faces, resolution })
    }

    /// Bilinear sample the cubemap at a direction.
    ///
    /// # Errors
    /// - `LightProbeError::ZeroDirection` if `dir` is near-zero.
    pub fn sample(&self, dir: [f32; 3]) -> Result<[f32; 3], LightProbeError> {
        let d = lp_normalize_dir(dir)?;
        let (face, u, v) = lp_dir_to_cubemap_uv(d);
        let face_data = &self.faces[face as usize];
        let rgb = bilinear_sample_rgb(face_data, self.resolution, self.resolution, u, v);
        Ok(rgb)
    }

    /// Nearest-neighbor sample the cubemap at a direction.
    ///
    /// # Errors
    /// - `LightProbeError::ZeroDirection` if `dir` is near-zero.
    pub fn sample_nearest(&self, dir: [f32; 3]) -> Result<[f32; 3], LightProbeError> {
        let d = lp_normalize_dir(dir)?;
        let (face, u, v) = lp_dir_to_cubemap_uv(d);
        let res = self.resolution as usize;
        let px = ((u * self.resolution as f32) as usize).min(res - 1);
        let py = ((v * self.resolution as f32) as usize).min(res - 1);
        let base = (py * res + px) * 3;
        let face_data = &self.faces[face as usize];
        Ok([face_data[base], face_data[base + 1], face_data[base + 2]])
    }
}

/// Project a cubemap probe to L=2 SH coefficients via Monte Carlo sampling.
///
/// # Errors
/// - Propagates `LightProbeError::ZeroDirection` (should not occur for uniform sphere samples).
pub fn lp_cubemap_to_sh(
    probe: &CubemapProbe,
    n_samples: usize,
    seed: u64,
) -> Result<IrradianceSH, LightProbeError> {
    let dirs = lp_generate_sphere_samples(n_samples, seed);
    let mut radiances = Vec::with_capacity(n_samples);
    for d in &dirs {
        let rgb = probe.sample(*d)?;
        radiances.push(rgb);
    }
    lp_project_samples_to_sh(&dirs, &radiances)
}

/// Like [`lp_cubemap_to_sh`], but takes the sample count from
/// `config.n_samples_projection` instead of a bare parameter.
///
/// # Errors
/// - Propagates `LightProbeError::ZeroDirection` (should not occur for uniform sphere samples).
pub fn lp_cubemap_to_sh_with_config(
    probe: &CubemapProbe,
    config: &LightProbeConfig,
    seed: u64,
) -> Result<IrradianceSH, LightProbeError> {
    lp_cubemap_to_sh(probe, config.n_samples_projection, seed)
}
