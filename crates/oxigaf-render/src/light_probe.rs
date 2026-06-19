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
//! Monte Carlo sphere sampling uses xorshift64 to avoid external `rand` dependency:
//! ```
//! state ^= state << 13;
//! state ^= state >> 7;
//! state ^= state << 17;
//! if state == 0 { state = 1; }
//! ```

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// SH constants (private copies; avoids collision with spherical_harmonics re-exports)
// ---------------------------------------------------------------------------

/// L=0 normalisation: 1 / (2√π)
const LP_SH_C0: f32 = 0.282_094_791_77_f32;

/// L=1 normalisation: √(3/(4π))
const LP_SH_C1: f32 = 0.488_602_511_90_f32;

/// L=2 normalisations (m = −2,−1,0,+1,+2)
const LP_SH_C2: [f32; 5] = [
    1.092_548_430_59_f32, // m=−2: √(15/(4π))
    1.092_548_430_59_f32, // m=−1: √(15/(4π))
    0.315_391_565_25_f32, // m= 0: √(5/(16π))
    1.092_548_430_59_f32, // m=+1: √(15/(4π))
    0.546_274_215_29_f32, // m=+2: (1/2)√(15/π) -- note: this equals (1/2)*√(15/4π)*√4 = √(15/4π)*1/2*(√4)
];

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors produced by light probe operations.
#[derive(Debug, thiserror::Error)]
pub enum LightProbeError {
    /// Invalid cubemap resolution (must be power of 2 and >= 4).
    #[error("Invalid cubemap resolution {res}: must be power of 2 and >= 4")]
    InvalidResolution { res: u32 },

    /// Buffer length mismatch.
    #[error("Buffer length mismatch: expected {expected}, got {got}")]
    BufferMismatch { expected: usize, got: usize },

    /// Empty probe list supplied.
    #[error("Empty probe list")]
    EmptyProbeList,

    /// SH order not in {1, 2, 3}.
    #[error("Invalid SH order {order}: must be 1, 2, or 3")]
    InvalidOrder { order: usize },

    /// Zero-length direction vector.
    #[error("Invalid direction vector (zero length)")]
    ZeroDirection,
}

// ---------------------------------------------------------------------------
// SH basis functions
// ---------------------------------------------------------------------------

/// Normalize a direction; error if norm < 1e-7.
#[inline]
pub fn lp_normalize_dir(dir: [f32; 3]) -> Result<[f32; 3], LightProbeError> {
    let [x, y, z] = dir;
    let norm = (x * x + y * y + z * z).sqrt();
    if norm < 1e-7 {
        return Err(LightProbeError::ZeroDirection);
    }
    Ok([x / norm, y / norm, z / norm])
}

/// Evaluate the single L=0 SH basis function: Y_0^0 = 1/(2√π).
///
/// `dir` does **not** need to be a unit vector for this function (result is constant).
#[inline]
pub fn lp_sh_basis_l0(_dir: [f32; 3]) -> [f32; 1] {
    [LP_SH_C0]
}

/// Evaluate L=1 SH basis functions (3 values, excluding L=0).
///
/// `dir` should be a unit vector.
/// Returns `[Y_1^{-1}, Y_1^0, Y_1^1]`.
#[inline]
pub fn lp_sh_basis_l1(dir: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = dir;
    [LP_SH_C1 * y, LP_SH_C1 * z, LP_SH_C1 * x]
}

/// Evaluate L=2 SH basis functions (5 values, excluding L=0 and L=1).
///
/// `dir` should be a unit vector.
/// Returns `[Y_2^{-2}, Y_2^{-1}, Y_2^0, Y_2^1, Y_2^2]`.
#[inline]
pub fn lp_sh_basis_l2(dir: [f32; 3]) -> [f32; 5] {
    let [x, y, z] = dir;
    [
        LP_SH_C2[0] * x * y,
        LP_SH_C2[1] * y * z,
        LP_SH_C2[2] * (2.0 * z * z - x * x - y * y),
        LP_SH_C2[3] * x * z,
        LP_SH_C2[4] * (x * x - y * y),
    ]
}

/// Evaluate all SH basis functions up to `order` (1=L0 only, 2=L0+L1, 3=L0+L1+L2).
///
/// Returns 1, 4, or 9 coefficients respectively.
/// The `dir` is normalized internally.
///
/// # Errors
/// - `LightProbeError::InvalidOrder` if order ∉ {1, 2, 3}
/// - `LightProbeError::ZeroDirection` if `dir` has near-zero length
pub fn lp_sh_basis(dir: [f32; 3], order: usize) -> Result<Vec<f32>, LightProbeError> {
    let d = lp_normalize_dir(dir)?;
    match order {
        1 => Ok(lp_sh_basis_l0(d).to_vec()),
        2 => {
            let mut v = Vec::with_capacity(4);
            v.extend_from_slice(&lp_sh_basis_l0(d));
            v.extend_from_slice(&lp_sh_basis_l1(d));
            Ok(v)
        }
        3 => {
            let mut v = Vec::with_capacity(9);
            v.extend_from_slice(&lp_sh_basis_l0(d));
            v.extend_from_slice(&lp_sh_basis_l1(d));
            v.extend_from_slice(&lp_sh_basis_l2(d));
            Ok(v)
        }
        _ => Err(LightProbeError::InvalidOrder { order }),
    }
}

// ---------------------------------------------------------------------------
// IrradianceSH
// ---------------------------------------------------------------------------

/// 27-coefficient (9 SH basis × 3 channels) irradiance representation.
///
/// Layout: `coefficients[basis_i * 3 + channel]` where `channel` ∈ {0=R, 1=G, 2=B}.
#[derive(Debug, Clone, PartialEq)]
pub struct IrradianceSH {
    /// Flat array of 27 coefficients (interleaved RGB).
    pub coefficients: [f32; 27],
    /// SH order (always 2 = L=2 for this struct).
    pub order: usize,
}

impl IrradianceSH {
    /// Construct a zero-initialized `IrradianceSH` of order 2.
    pub fn new() -> Self {
        Self {
            coefficients: [0.0_f32; 27],
            order: 2,
        }
    }

    /// Construct from a pre-built coefficient array.
    pub fn from_coefficients(coeffs: [f32; 27]) -> Self {
        Self { coefficients: coeffs, order: 2 }
    }

    /// Evaluate irradiance at unit direction `dir`.
    ///
    /// `E(n) = Σ_{i=0}^{8} c_i * Y_i(n)` per channel.
    ///
    /// # Errors
    /// - `LightProbeError::ZeroDirection` if `dir` is degenerate.
    pub fn evaluate(&self, dir: [f32; 3]) -> Result<[f32; 3], LightProbeError> {
        let d = lp_normalize_dir(dir)?;
        let basis = lp_sh_full_9(d);
        let mut rgb = [0.0_f32; 3];
        for i in 0..9 {
            for c in 0..3 {
                rgb[c] += basis[i] * self.coefficients[i * 3 + c];
            }
        }
        Ok(rgb)
    }

    /// Return a new `IrradianceSH` with all coefficients scaled by `factor`.
    pub fn scale(&self, factor: f32) -> Self {
        let mut out = self.clone();
        for v in out.coefficients.iter_mut() {
            *v *= factor;
        }
        out
    }

    /// Return a new `IrradianceSH` that is the element-wise sum of `self` and `other`.
    pub fn add(&self, other: &IrradianceSH) -> Self {
        let mut out = self.clone();
        for i in 0..27 {
            out.coefficients[i] += other.coefficients[i];
        }
        out
    }

    /// Return the ambient (constant, L=0) term as RGB.
    ///
    /// Coefficient 0 represents the L=0 (DC) term. The actual radiated
    /// value is `c0 * Y_0^0`, but for ambient purposes we return `c0 * LP_SH_C0`
    /// per channel so it represents the average irradiance.
    pub fn ambient(&self) -> [f32; 3] {
        [
            self.coefficients[0] * LP_SH_C0,
            self.coefficients[1] * LP_SH_C0,
            self.coefficients[2] * LP_SH_C0,
        ]
    }
}

impl Default for IrradianceSH {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute all 9 SH basis values for a unit direction (internal helper).
#[inline]
fn lp_sh_full_9(dir: [f32; 3]) -> [f32; 9] {
    let l0 = lp_sh_basis_l0(dir);
    let l1 = lp_sh_basis_l1(dir);
    let l2 = lp_sh_basis_l2(dir);
    [l0[0], l1[0], l1[1], l1[2], l2[0], l2[1], l2[2], l2[3], l2[4]]
}

// ---------------------------------------------------------------------------
// Monte Carlo SH projection
// ---------------------------------------------------------------------------

/// xorshift64 PRNG — advances `state` and returns the next pseudo-random u64.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

/// Convert a xorshift64 output to a f32 in `[0, 1)`.
#[inline]
fn xorshift64_f32(state: &mut u64) -> f32 {
    (xorshift64(state) as f32) / (u64::MAX as f32)
}

/// Generate `n` uniform unit sphere samples using xorshift64.
///
/// Uses spherical coordinates: θ = arccos(1 − 2u₁), φ = 2π u₂.
pub fn lp_generate_sphere_samples(n: usize, seed: u64) -> Vec<[f32; 3]> {
    let mut state = if seed == 0 { 1u64 } else { seed };
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let u1 = xorshift64_f32(&mut state);
        let u2 = xorshift64_f32(&mut state);
        let cos_theta = 1.0 - 2.0 * u1;
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = 2.0 * PI * u2;
        let x = sin_theta * phi.cos();
        let y = sin_theta * phi.sin();
        let z = cos_theta;
        samples.push([x, y, z]);
    }
    samples
}

/// Project a set of direction-radiance samples to L=2 SH coefficients via Monte Carlo.
///
/// `c_lm = (4π / N) * Σ_i L(ω_i) * Y_lm(ω_i)`
///
/// # Errors
/// - `LightProbeError::BufferMismatch` if `directions.len() != radiances.len()`
pub fn lp_project_samples_to_sh(
    directions: &[[f32; 3]],
    radiances: &[[f32; 3]],
) -> Result<IrradianceSH, LightProbeError> {
    if directions.len() != radiances.len() {
        return Err(LightProbeError::BufferMismatch {
            expected: directions.len(),
            got: radiances.len(),
        });
    }
    let n = directions.len();
    let mut coeffs = [0.0_f32; 27];
    for (dir, rad) in directions.iter().zip(radiances.iter()) {
        // Skip near-zero directions silently (degenerate sample)
        let norm_sq = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
        if norm_sq < 1e-12 {
            continue;
        }
        let inv = norm_sq.sqrt().recip();
        let d = [dir[0] * inv, dir[1] * inv, dir[2] * inv];
        let basis = lp_sh_full_9(d);
        for i in 0..9 {
            for c in 0..3 {
                coeffs[i * 3 + c] += rad[c] * basis[i];
            }
        }
    }
    // Scale by 4π / N
    let scale = if n > 0 { 4.0 * PI / n as f32 } else { 0.0 };
    for v in coeffs.iter_mut() {
        *v *= scale;
    }
    Ok(IrradianceSH::from_coefficients(coeffs))
}

/// Bilinear sample from an f32 RGB row-major image.
#[inline]
fn bilinear_sample_rgb(image: &[f32], width: u32, height: u32, u: f32, v: f32) -> [f32; 3] {
    let w = width as usize;
    let h = height as usize;
    let px = (u * width as f32 - 0.5).max(0.0);
    let py = (v * height as f32 - 0.5).max(0.0);
    let x0 = (px.floor() as usize).min(w.saturating_sub(2));
    let y0 = (py.floor() as usize).min(h.saturating_sub(2));
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let tx = px - x0 as f32;
    let ty = py - y0 as f32;

    let idx = |row: usize, col: usize| -> [f32; 3] {
        let base = (row * w + col) * 3;
        [image[base], image[base + 1], image[base + 2]]
    };

    let c00 = idx(y0, x0);
    let c10 = idx(y0, x1);
    let c01 = idx(y1, x0);
    let c11 = idx(y1, x1);

    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    [
        lerp(lerp(c00[0], c10[0], tx), lerp(c01[0], c11[0], tx), ty),
        lerp(lerp(c00[1], c10[1], tx), lerp(c01[1], c11[1], tx), ty),
        lerp(lerp(c00[2], c10[2], tx), lerp(c01[2], c11[2], tx), ty),
    ]
}

/// Project an equirectangular (lat-long) panorama image to L=2 SH coefficients.
///
/// `image`: RGB f32 row-major, length = `width * height * 3`.
/// Sampling via Monte Carlo with `n_samples` directions.
///
/// # Errors
/// - `LightProbeError::BufferMismatch` if image length != width * height * 3.
pub fn lp_project_latitude_longitude(
    image: &[f32],
    width: u32,
    height: u32,
    n_samples: usize,
    seed: u64,
) -> Result<IrradianceSH, LightProbeError> {
    let expected = (width as usize) * (height as usize) * 3;
    if image.len() != expected {
        return Err(LightProbeError::BufferMismatch { expected, got: image.len() });
    }

    let dirs = lp_generate_sphere_samples(n_samples, seed);
    let mut radiances = Vec::with_capacity(n_samples);

    for d in &dirs {
        // Convert unit sphere direction to equirectangular (u, v)
        // φ = atan2(z, x) mapped to [0, 1], θ = acos(y) mapped to [0, 1]
        let phi = d[2].atan2(d[0]);          // [-π, π]
        let theta = d[1].clamp(-1.0, 1.0).acos(); // [0, π]
        let u = (phi / (2.0 * PI) + 0.5).rem_euclid(1.0);
        let v = theta / PI;
        let rgb = bilinear_sample_rgb(image, width, height, u, v);
        radiances.push(rgb);
    }

    lp_project_samples_to_sh(&dirs, &radiances)
}

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

// ---------------------------------------------------------------------------
// LightProbe
// ---------------------------------------------------------------------------

/// Positional light probe with irradiance SH and influence radius.
#[derive(Debug, Clone)]
pub struct LightProbe {
    /// World-space position.
    pub position: [f32; 3],
    /// Pre-convolved irradiance as SH coefficients.
    pub irradiance: IrradianceSH,
    /// Influence radius (0 = global/infinite).
    pub radius: f32,
    /// Intensity multiplier.
    pub intensity: f32,
    /// Unique identifier.
    pub id: u64,
}

/// Monotonically increasing probe ID counter (never decremented).
static PROBE_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl LightProbe {
    /// Construct a new probe, assigning an auto-incremented ID.
    pub fn new(position: [f32; 3], irradiance: IrradianceSH, radius: f32) -> Self {
        let id = PROBE_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self { position, irradiance, radius, intensity: 1.0, id }
    }

    /// Compute the smooth influence weight for a world-space `point`.
    ///
    /// - `radius == 0` → `1.0` everywhere (global probe).
    /// - Otherwise: `weight = 1 − clamp(dist / radius, 0, 1)²`
    pub fn weight_for(&self, point: [f32; 3]) -> f32 {
        if self.radius <= 0.0 {
            return 1.0;
        }
        let dx = point[0] - self.position[0];
        let dy = point[1] - self.position[1];
        let dz = point[2] - self.position[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        let t = (dist / self.radius).clamp(0.0, 1.0);
        1.0 - t * t
    }

    /// Evaluate irradiance at `point` from this probe.
    ///
    /// Evaluates SH at the surface `normal`, then multiplies by probe weight and intensity.
    ///
    /// # Errors
    /// - `LightProbeError::ZeroDirection` if `normal` is near-zero.
    pub fn evaluate(&self, point: [f32; 3], normal: [f32; 3]) -> Result<[f32; 3], LightProbeError> {
        let weight = self.weight_for(point);
        let irr = self.irradiance.evaluate(normal)?;
        Ok([
            irr[0] * weight * self.intensity,
            irr[1] * weight * self.intensity,
            irr[2] * weight * self.intensity,
        ])
    }
}

// ---------------------------------------------------------------------------
// Multi-probe blending
// ---------------------------------------------------------------------------

/// Strategy for combining contributions from multiple light probes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeBlendMode {
    /// Use only the nearest probe (weight = 1.0 for closest, 0.0 for rest).
    Nearest,
    /// Blend by distance-based influence weights, then normalise.
    WeightedAverage,
    /// Weight by influence volume overlap (same as `WeightedAverage` in CPU path).
    VolumeWeighted,
}

/// Blend a weighted set of `LightProbe` SH coefficients into a single `IrradianceSH`.
///
/// `weights` and `probes` must have the same length.
///
/// # Errors
/// - `LightProbeError::BufferMismatch` if lengths differ.
/// - `LightProbeError::EmptyProbeList` if the slice is empty.
pub fn lp_blend_irradiance_sh(
    probes: &[LightProbe],
    weights: &[f32],
) -> Result<IrradianceSH, LightProbeError> {
    if probes.is_empty() {
        return Err(LightProbeError::EmptyProbeList);
    }
    if probes.len() != weights.len() {
        return Err(LightProbeError::BufferMismatch {
            expected: probes.len(),
            got: weights.len(),
        });
    }

    let weight_sum: f32 = weights.iter().sum();
    let mut out = [0.0_f32; 27];

    if weight_sum < 1e-12 {
        // All weights zero — fall back to equal blending
        let inv = 1.0 / probes.len() as f32;
        for probe in probes {
            for i in 0..27 {
                out[i] += probe.irradiance.coefficients[i] * inv;
            }
        }
    } else {
        let inv = 1.0 / weight_sum;
        for (probe, &w) in probes.iter().zip(weights.iter()) {
            let nw = w * inv;
            for i in 0..27 {
                out[i] += probe.irradiance.coefficients[i] * nw;
            }
        }
    }

    Ok(IrradianceSH::from_coefficients(out))
}

/// Collection of light probes with a blending strategy.
#[derive(Debug, Clone)]
pub struct LightProbeBlend {
    /// Ordered list of probes.
    pub probes: Vec<LightProbe>,
    /// Blending mode.
    pub blend_mode: ProbeBlendMode,
}

impl LightProbeBlend {
    /// Construct from a non-empty list of probes.
    ///
    /// # Errors
    /// - `LightProbeError::EmptyProbeList` if `probes` is empty.
    pub fn new(probes: Vec<LightProbe>, mode: ProbeBlendMode) -> Result<Self, LightProbeError> {
        if probes.is_empty() {
            return Err(LightProbeError::EmptyProbeList);
        }
        Ok(Self { probes, blend_mode: mode })
    }

    /// Evaluate blended irradiance at a world-space `point` with surface `normal`.
    ///
    /// # Errors
    /// - `LightProbeError::ZeroDirection` if `normal` is degenerate.
    pub fn evaluate(&self, point: [f32; 3], normal: [f32; 3]) -> Result<[f32; 3], LightProbeError> {
        match self.blend_mode {
            ProbeBlendMode::Nearest => {
                // Find probe with highest weight for this point
                let (best_idx, _) = self.probes.iter().enumerate().fold(
                    (0usize, f32::NEG_INFINITY),
                    |(bi, bw), (i, p)| {
                        let w = p.weight_for(point);
                        if w > bw { (i, w) } else { (bi, bw) }
                    },
                );
                self.probes[best_idx].evaluate(point, normal)
            }
            ProbeBlendMode::WeightedAverage | ProbeBlendMode::VolumeWeighted => {
                let weights: Vec<f32> = self.probes.iter().map(|p| p.weight_for(point)).collect();
                let blended_sh = lp_blend_irradiance_sh(&self.probes, &weights)?;
                // Evaluate SH at normal
                let irr = blended_sh.evaluate(normal)?;
                // Compute total intensity (weighted average)
                let weight_sum: f32 = weights.iter().sum();
                let intensity = if weight_sum < 1e-12 {
                    1.0
                } else {
                    self.probes.iter().zip(weights.iter())
                        .map(|(p, &w)| p.intensity * w)
                        .sum::<f32>() / weight_sum
                };
                Ok([irr[0] * intensity, irr[1] * intensity, irr[2] * intensity])
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Diffuse IBL
// ---------------------------------------------------------------------------

/// Apply diffuse image-based lighting to a single Gaussian.
///
/// `L_diffuse = E(n) * albedo / π`
///
/// # Errors
/// - `LightProbeError::ZeroDirection` if `normal` is degenerate.
pub fn lp_evaluate_diffuse_ibl(
    normal: [f32; 3],
    sh: &IrradianceSH,
    albedo: [f32; 3],
) -> Result<[f32; 3], LightProbeError> {
    let irr = sh.evaluate(normal)?;
    let inv_pi = 1.0 / PI;
    Ok([
        irr[0] * albedo[0] * inv_pi,
        irr[1] * albedo[1] * inv_pi,
        irr[2] * albedo[2] * inv_pi,
    ])
}

/// Apply diffuse IBL to `n_gaussians` Gaussians.
///
/// `normals`: flat `[N × 3]` f32 array (x, y, z per Gaussian).
/// `albedo`:  flat `[N × 3]` f32 array (R, G, B per Gaussian).
///
/// Returns flat `[N × 3]` f32 RGB output.
///
/// # Errors
/// - `LightProbeError::BufferMismatch` if `normals.len()` or `albedo.len()` != `n_gaussians * 3`.
/// - `LightProbeError::ZeroDirection` if any normal is degenerate.
pub fn lp_apply_ibl_to_gaussians(
    normals: &[f32],
    sh: &IrradianceSH,
    albedo: &[f32],
    n_gaussians: usize,
) -> Result<Vec<f32>, LightProbeError> {
    let expected = n_gaussians * 3;
    if normals.len() != expected {
        return Err(LightProbeError::BufferMismatch { expected, got: normals.len() });
    }
    if albedo.len() != expected {
        return Err(LightProbeError::BufferMismatch { expected, got: albedo.len() });
    }

    let mut out = Vec::with_capacity(expected);
    for i in 0..n_gaussians {
        let base = i * 3;
        let normal = [normals[base], normals[base + 1], normals[base + 2]];
        let alb = [albedo[base], albedo[base + 1], albedo[base + 2]];
        let rgb = lp_evaluate_diffuse_ibl(normal, sh, alb)?;
        out.push(rgb[0]);
        out.push(rgb[1]);
        out.push(rgb[2]);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Statistics and configuration
// ---------------------------------------------------------------------------

/// Configuration for light probe operations.
#[derive(Debug, Clone)]
pub struct LightProbeConfig {
    /// Number of Monte Carlo samples for SH projection (default 10000).
    pub n_samples_projection: usize,
    /// Maximum number of probes in a scene.
    pub max_probes: usize,
    /// Default blending mode.
    pub blend_mode: ProbeBlendMode,
}

impl Default for LightProbeConfig {
    fn default() -> Self {
        Self {
            n_samples_projection: 10_000,
            max_probes: 64,
            blend_mode: ProbeBlendMode::WeightedAverage,
        }
    }
}

/// Aggregated statistics for a collection of light probes.
#[derive(Debug, Clone)]
pub struct LightProbeStats {
    /// Number of probes.
    pub n_probes: usize,
    /// Mean per-probe intensity.
    pub mean_intensity: f32,
    /// Maximum absolute SH coefficient value across all probes.
    pub max_coefficient: f32,
    /// Average ambient RGB (L=0 term).
    pub ambient_rgb: [f32; 3],
    /// Sum of squared SH coefficients (energy measure).
    pub sh_energy: f32,
}

/// Compute aggregate statistics for a slice of probes.
///
/// # Errors
/// - `LightProbeError::EmptyProbeList` if `probes` is empty.
pub fn lp_compute_stats(probes: &[LightProbe]) -> Result<LightProbeStats, LightProbeError> {
    if probes.is_empty() {
        return Err(LightProbeError::EmptyProbeList);
    }

    let n = probes.len();
    let mean_intensity = probes.iter().map(|p| p.intensity).sum::<f32>() / n as f32;

    let mut max_coefficient = 0.0_f32;
    let mut sh_energy = 0.0_f32;
    let mut ambient_sum = [0.0_f32; 3];

    for probe in probes {
        for &c in &probe.irradiance.coefficients {
            let abs = c.abs();
            if abs > max_coefficient {
                max_coefficient = abs;
            }
            sh_energy += c * c;
        }
        let amb = probe.irradiance.ambient();
        ambient_sum[0] += amb[0];
        ambient_sum[1] += amb[1];
        ambient_sum[2] += amb[2];
    }

    let inv_n = 1.0 / n as f32;
    Ok(LightProbeStats {
        n_probes: n,
        mean_intensity,
        max_coefficient,
        ambient_rgb: [ambient_sum[0] * inv_n, ambient_sum[1] * inv_n, ambient_sum[2] * inv_n],
        sh_energy,
    })
}

/// Format `LightProbeStats` as a human-readable string.
pub fn lp_format_stats(stats: &LightProbeStats) -> String {
    format!(
        "LightProbeStats {{ n_probes: {}, mean_intensity: {:.4}, max_coefficient: {:.6}, \
         ambient_rgb: [{:.4}, {:.4}, {:.4}], sh_energy: {:.6} }}",
        stats.n_probes,
        stats.mean_intensity,
        stats.max_coefficient,
        stats.ambient_rgb[0],
        stats.ambient_rgb[1],
        stats.ambient_rgb[2],
        stats.sh_energy,
    )
}

/// Format `LightProbeConfig` as a human-readable string.
pub fn lp_format_config(config: &LightProbeConfig) -> String {
    let mode_str = match config.blend_mode {
        ProbeBlendMode::Nearest => "Nearest",
        ProbeBlendMode::WeightedAverage => "WeightedAverage",
        ProbeBlendMode::VolumeWeighted => "VolumeWeighted",
    };
    format!(
        "LightProbeConfig {{ n_samples_projection: {}, max_probes: {}, blend_mode: {} }}",
        config.n_samples_projection, config.max_probes, mode_str,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SQRT_PI: f32 = 1.772_453_850_9_f32;

    // -----------------------------------------------------------------------
    // lp_normalize_dir
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalize_unit_vector() {
        let d = lp_normalize_dir([1.0, 0.0, 0.0]).expect("should not error");
        assert!((d[0] - 1.0).abs() < 1e-6);
        assert!(d[1].abs() < 1e-6);
        assert!(d[2].abs() < 1e-6);
    }

    #[test]
    fn test_normalize_scaled_vector() {
        let d = lp_normalize_dir([3.0, 4.0, 0.0]).expect("should not error");
        let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_error() {
        let result = lp_normalize_dir([0.0, 0.0, 0.0]);
        assert!(matches!(result, Err(LightProbeError::ZeroDirection)));
    }

    #[test]
    fn test_normalize_near_zero_error() {
        let result = lp_normalize_dir([1e-8, 0.0, 0.0]);
        assert!(matches!(result, Err(LightProbeError::ZeroDirection)));
    }

    // -----------------------------------------------------------------------
    // lp_sh_basis_l0
    // -----------------------------------------------------------------------

    #[test]
    fn test_sh_basis_l0_value() {
        // Y_0^0 = 1/(2√π)
        let expected = 1.0 / (2.0 * SQRT_PI);
        let result = lp_sh_basis_l0([1.0, 0.0, 0.0]);
        assert!((result[0] - expected).abs() < 1e-5, "got {}", result[0]);
    }

    #[test]
    fn test_sh_basis_l0_constant() {
        // Same for any direction
        let a = lp_sh_basis_l0([1.0, 0.0, 0.0])[0];
        let b = lp_sh_basis_l0([0.0, 1.0, 0.0])[0];
        let c = lp_sh_basis_l0([0.6, 0.8, 0.0])[0];
        assert!((a - b).abs() < 1e-9);
        assert!((a - c).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // lp_sh_basis_l1
    // -----------------------------------------------------------------------

    #[test]
    fn test_sh_basis_l1_along_z() {
        // dir = [0, 0, 1]: Y_1^{-1}=0, Y_1^0=SH_C1, Y_1^1=0
        let result = lp_sh_basis_l1([0.0, 0.0, 1.0]);
        assert!(result[0].abs() < 1e-6); // Y_1^{-1}=C1*y=0
        assert!((result[1] - LP_SH_C1).abs() < 1e-6); // Y_1^0=C1*z=C1
        assert!(result[2].abs() < 1e-6); // Y_1^1=C1*x=0
    }

    #[test]
    fn test_sh_basis_l1_along_x() {
        // dir=[1,0,0]: Y_1^{-1}=0, Y_1^0=0, Y_1^1=C1
        let result = lp_sh_basis_l1([1.0, 0.0, 0.0]);
        assert!(result[0].abs() < 1e-6);
        assert!(result[1].abs() < 1e-6);
        assert!((result[2] - LP_SH_C1).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // lp_sh_basis
    // -----------------------------------------------------------------------

    #[test]
    fn test_sh_basis_order1_length() {
        let v = lp_sh_basis([1.0, 0.0, 0.0], 1).expect("order 1 ok");
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_sh_basis_order2_length() {
        let v = lp_sh_basis([1.0, 0.0, 0.0], 2).expect("order 2 ok");
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_sh_basis_order3_length() {
        let v = lp_sh_basis([1.0, 0.0, 0.0], 3).expect("order 3 ok");
        assert_eq!(v.len(), 9);
    }

    #[test]
    fn test_sh_basis_invalid_order() {
        let r = lp_sh_basis([1.0, 0.0, 0.0], 0);
        assert!(matches!(r, Err(LightProbeError::InvalidOrder { .. })));
        let r2 = lp_sh_basis([1.0, 0.0, 0.0], 4);
        assert!(matches!(r2, Err(LightProbeError::InvalidOrder { .. })));
    }

    #[test]
    fn test_sh_basis_orthogonality_different_dirs() {
        let v1 = lp_sh_basis([1.0, 0.0, 0.0], 3).expect("ok");
        let v2 = lp_sh_basis([0.0, 1.0, 0.0], 3).expect("ok");
        // The basis vectors for two different unit directions must differ
        let diff: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.01, "basis vectors should differ for different directions");
    }

    // -----------------------------------------------------------------------
    // IrradianceSH
    // -----------------------------------------------------------------------

    #[test]
    fn test_irradiance_sh_evaluate_zero() {
        let sh = IrradianceSH::new();
        let result = sh.evaluate([0.0, 0.0, 1.0]).expect("ok");
        assert!(result[0].abs() < 1e-9);
        assert!(result[1].abs() < 1e-9);
        assert!(result[2].abs() < 1e-9);
    }

    #[test]
    fn test_irradiance_sh_constant_probe() {
        // Only L=0 coefficient set — result should be same for all directions
        let mut coeffs = [0.0_f32; 27];
        let val = 2.0_f32;
        coeffs[0] = val; // R
        coeffs[1] = val; // G
        coeffs[2] = val; // B
        let sh = IrradianceSH::from_coefficients(coeffs);

        let r1 = sh.evaluate([1.0, 0.0, 0.0]).expect("ok");
        let r2 = sh.evaluate([0.0, 1.0, 0.0]).expect("ok");
        let r3 = sh.evaluate([-1.0, 0.0, 0.0]).expect("ok");
        // All should be val * LP_SH_C0
        let expected = val * LP_SH_C0;
        assert!((r1[0] - expected).abs() < 1e-5);
        assert!((r2[0] - expected).abs() < 1e-5);
        assert!((r3[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_irradiance_sh_scale() {
        let mut coeffs = [0.0_f32; 27];
        coeffs[0] = 1.0;
        coeffs[3] = 2.0;
        let sh = IrradianceSH::from_coefficients(coeffs);
        let scaled = sh.scale(2.0);
        assert!((scaled.coefficients[0] - 2.0).abs() < 1e-9);
        assert!((scaled.coefficients[3] - 4.0).abs() < 1e-9);
        // Original unchanged
        assert!((sh.coefficients[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_irradiance_sh_add() {
        let mut c1 = [0.0_f32; 27];
        c1[0] = 1.0;
        let mut c2 = [0.0_f32; 27];
        c2[0] = 3.0;
        c2[6] = -1.0;
        let sh1 = IrradianceSH::from_coefficients(c1);
        let sh2 = IrradianceSH::from_coefficients(c2);
        let sum = sh1.add(&sh2);
        assert!((sum.coefficients[0] - 4.0).abs() < 1e-9);
        assert!((sum.coefficients[6] + 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_irradiance_sh_ambient_nonzero() {
        let mut coeffs = [0.0_f32; 27];
        coeffs[0] = 1.0;
        coeffs[1] = 0.5;
        coeffs[2] = 0.2;
        let sh = IrradianceSH::from_coefficients(coeffs);
        let amb = sh.ambient();
        assert!(amb[0] > 0.0);
        assert!(amb[1] > 0.0);
        assert!(amb[2] > 0.0);
    }

    #[test]
    fn test_irradiance_sh_evaluate_zero_dir_error() {
        let sh = IrradianceSH::new();
        let r = sh.evaluate([0.0, 0.0, 0.0]);
        assert!(matches!(r, Err(LightProbeError::ZeroDirection)));
    }

    // -----------------------------------------------------------------------
    // lp_generate_sphere_samples
    // -----------------------------------------------------------------------

    #[test]
    fn test_sphere_samples_unit_norm() {
        let samples = lp_generate_sphere_samples(200, 42);
        for s in &samples {
            let norm = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "norm={}", norm);
        }
    }

    #[test]
    fn test_sphere_samples_octant_distribution() {
        let samples = lp_generate_sphere_samples(8_000, 123);
        // Count samples in each of the 8 octants
        let mut counts = [0usize; 8];
        for s in &samples {
            let xi = if s[0] >= 0.0 { 1 } else { 0 };
            let yi = if s[1] >= 0.0 { 2 } else { 0 };
            let zi = if s[2] >= 0.0 { 4 } else { 0 };
            counts[xi + yi + zi] += 1;
        }
        // Each octant should have roughly N/8 = 1000 samples; allow ±40%
        for &count in &counts {
            assert!(count > 600 && count < 1400, "octant count = {}", count);
        }
    }

    // -----------------------------------------------------------------------
    // lp_project_samples_to_sh
    // -----------------------------------------------------------------------

    #[test]
    fn test_project_constant_radiance() {
        // Constant white radiance → almost all energy in L=0
        let n = 5_000usize;
        let dirs = lp_generate_sphere_samples(n, 7);
        let rads: Vec<[f32; 3]> = dirs.iter().map(|_| [1.0, 1.0, 1.0]).collect();
        let sh = lp_project_samples_to_sh(&dirs, &rads).expect("ok");

        // The L=0 coefficient (index 0) should dominate
        let l0_r = sh.coefficients[0].abs();
        for i in 1..9 {
            let higher = sh.coefficients[i * 3].abs();
            assert!(l0_r > higher * 5.0, "L={} coeff {} > L=0/5 for constant input", i, higher);
        }
    }

    #[test]
    fn test_project_length_mismatch() {
        let dirs = lp_generate_sphere_samples(10, 1);
        let rads: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0]; 5];
        let r = lp_project_samples_to_sh(&dirs, &rads);
        assert!(matches!(r, Err(LightProbeError::BufferMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // lp_project_latitude_longitude
    // -----------------------------------------------------------------------

    #[test]
    fn test_project_latlong_uniform() {
        // All-white image → near-constant SH (L>0 terms small relative to L=0)
        let w = 64u32;
        let h = 32u32;
        let image = vec![1.0_f32; (w * h * 3) as usize];
        let sh = lp_project_latitude_longitude(&image, w, h, 2000, 99).expect("ok");
        let l0 = sh.coefficients[0].abs();
        for i in 1..9 {
            let hi = sh.coefficients[i * 3].abs();
            assert!(hi < l0 * 0.3, "L={} coefficient too large for uniform input", i);
        }
    }

    #[test]
    fn test_project_latlong_buffer_mismatch() {
        let r = lp_project_latitude_longitude(&[1.0; 10], 4, 4, 100, 1);
        assert!(matches!(r, Err(LightProbeError::BufferMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // lp_dir_to_cubemap_uv
    // -----------------------------------------------------------------------

    #[test]
    fn test_cubemap_uv_pos_x() {
        let (face, _u, _v) = lp_dir_to_cubemap_uv([1.0, 0.0, 0.0]);
        assert_eq!(face, CubemapFace::PosX);
    }

    #[test]
    fn test_cubemap_uv_neg_x() {
        let (face, _u, _v) = lp_dir_to_cubemap_uv([-1.0, 0.0, 0.0]);
        assert_eq!(face, CubemapFace::NegX);
    }

    #[test]
    fn test_cubemap_uv_pos_y() {
        let (face, _u, _v) = lp_dir_to_cubemap_uv([0.0, 1.0, 0.0]);
        assert_eq!(face, CubemapFace::PosY);
    }

    #[test]
    fn test_cubemap_uv_neg_y() {
        let (face, _u, _v) = lp_dir_to_cubemap_uv([0.0, -1.0, 0.0]);
        assert_eq!(face, CubemapFace::NegY);
    }

    #[test]
    fn test_cubemap_uv_pos_z() {
        let (face, _u, _v) = lp_dir_to_cubemap_uv([0.0, 0.0, 1.0]);
        assert_eq!(face, CubemapFace::PosZ);
    }

    #[test]
    fn test_cubemap_uv_neg_z() {
        let (face, _u, _v) = lp_dir_to_cubemap_uv([0.0, 0.0, -1.0]);
        assert_eq!(face, CubemapFace::NegZ);
    }

    #[test]
    fn test_cubemap_uv_in_range() {
        // u, v must lie in [0, 1]
        for dir in [
            [1.0_f32, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
            [0.7, 0.7, 0.0], [0.5, 0.5, 0.7],
        ] {
            let (_, u, v) = lp_dir_to_cubemap_uv(dir);
            assert!(u >= 0.0 && u <= 1.0, "u={} out of range for dir={:?}", u, dir);
            assert!(v >= 0.0 && v <= 1.0, "v={} out of range for dir={:?}", v, dir);
        }
    }

    // -----------------------------------------------------------------------
    // CubemapProbe
    // -----------------------------------------------------------------------

    #[test]
    fn test_cubemap_new_invalid_resolution_odd() {
        let r = CubemapProbe::new(5);
        assert!(matches!(r, Err(LightProbeError::InvalidResolution { .. })));
    }

    #[test]
    fn test_cubemap_new_invalid_resolution_small() {
        let r = CubemapProbe::new(2);
        assert!(matches!(r, Err(LightProbeError::InvalidResolution { .. })));
    }

    #[test]
    fn test_cubemap_new_valid() {
        let probe = CubemapProbe::new(8).expect("8 is valid");
        assert_eq!(probe.resolution, 8);
        assert_eq!(probe.faces.len(), 6);
        for face in &probe.faces {
            assert_eq!(face.len(), 8 * 8 * 3);
        }
    }

    #[test]
    fn test_cubemap_sample_different_faces() {
        let mut probe = CubemapProbe::new(4).expect("ok");
        // Color face 0 (+X) with red
        for v in probe.faces[0].iter_mut() { *v = 0.0; }
        for px in 0..(4 * 4) {
            probe.faces[0][px * 3] = 1.0; // R channel
        }
        // Color face 1 (-X) with blue
        for v in probe.faces[1].iter_mut() { *v = 0.0; }
        for px in 0..(4 * 4) {
            probe.faces[1][px * 3 + 2] = 1.0; // B channel
        }
        let pos_x = probe.sample([1.0, 0.0, 0.0]).expect("ok");
        let neg_x = probe.sample([-1.0, 0.0, 0.0]).expect("ok");
        // +X face: red dominant
        assert!(pos_x[0] > 0.5, "pos_x R should be high, got {:?}", pos_x);
        // -X face: blue dominant
        assert!(neg_x[2] > 0.5, "neg_x B should be high, got {:?}", neg_x);
    }

    #[test]
    fn test_cubemap_sample_zero_dir_error() {
        let probe = CubemapProbe::new(4).expect("ok");
        let r = probe.sample([0.0, 0.0, 0.0]);
        assert!(matches!(r, Err(LightProbeError::ZeroDirection)));
    }

    #[test]
    fn test_cubemap_to_sh_constant() {
        // Constant grey cubemap → mostly L=0 SH
        let mut probe = CubemapProbe::new(4).expect("ok");
        let grey = 0.5_f32;
        for face in probe.faces.iter_mut() {
            for (i, v) in face.iter_mut().enumerate() {
                *v = if i % 3 == 0 { grey } else { grey };
            }
        }
        let sh = lp_cubemap_to_sh(&probe, 3000, 42).expect("ok");
        let l0 = sh.coefficients[0].abs();
        // Higher-order terms should be much smaller than L=0
        for i in 4..9 {
            let hi = sh.coefficients[i * 3].abs();
            assert!(hi < l0 * 0.5, "L2 coefficient {} too large", hi);
        }
    }

    #[test]
    fn test_cubemap_bilinear_smooth() {
        // Fill +X face with a horizontal gradient; verify smooth interpolation
        let mut probe = CubemapProbe::new(8).expect("ok");
        let res = 8usize;
        for py in 0..res {
            for px in 0..res {
                let val = px as f32 / (res - 1) as f32;
                let base = (py * res + px) * 3;
                probe.faces[0][base] = val;
                probe.faces[0][base + 1] = val;
                probe.faces[0][base + 2] = val;
            }
        }
        // Sample at two points: left and right of +X face
        let left  = probe.sample([1.0,  0.0,  0.9]).expect("ok");
        let right = probe.sample([1.0,  0.0, -0.9]).expect("ok");
        // Right side (low-z) maps to low u, left side (high-z) to high u or vice versa
        // We just check that the two values are different (gradient is captured)
        let diff = (left[0] - right[0]).abs();
        assert!(diff > 0.1, "bilinear should capture gradient, diff={}", diff);
    }

    // -----------------------------------------------------------------------
    // LightProbe::weight_for
    // -----------------------------------------------------------------------

    #[test]
    fn test_weight_at_position() {
        let pos = [1.0, 2.0, 3.0];
        let probe = LightProbe::new(pos, IrradianceSH::new(), 5.0);
        let w = probe.weight_for(pos);
        assert!((w - 1.0).abs() < 1e-6, "weight at center should be 1.0, got {}", w);
    }

    #[test]
    fn test_weight_radius_zero_is_global() {
        let probe = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::new(), 0.0);
        assert!((probe.weight_for([100.0, 100.0, 100.0]) - 1.0).abs() < 1e-6);
        assert!((probe.weight_for([-50.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_weight_beyond_radius_small() {
        let probe = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::new(), 2.0);
        let w = probe.weight_for([4.0, 0.0, 0.0]); // dist=4, radius=2 → clamp → 0
        assert!(w < 0.01, "weight beyond radius should be near zero, got {}", w);
    }

    #[test]
    fn test_weight_at_boundary() {
        let probe = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::new(), 3.0);
        let w = probe.weight_for([3.0, 0.0, 0.0]);
        assert!(w.abs() < 1e-5, "weight at exact radius edge should be 0, got {}", w);
    }

    // -----------------------------------------------------------------------
    // LightProbe::evaluate
    // -----------------------------------------------------------------------

    #[test]
    fn test_probe_evaluate_zero_normal_error() {
        let probe = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::new(), 0.0);
        let r = probe.evaluate([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(matches!(r, Err(LightProbeError::ZeroDirection)));
    }

    // -----------------------------------------------------------------------
    // LightProbeBlend
    // -----------------------------------------------------------------------

    #[test]
    fn test_probe_blend_empty_error() {
        let r = LightProbeBlend::new(vec![], ProbeBlendMode::WeightedAverage);
        assert!(matches!(r, Err(LightProbeError::EmptyProbeList)));
    }

    #[test]
    fn test_probe_blend_single_matches_probe() {
        let mut coeffs = [0.0_f32; 27];
        coeffs[0] = 1.0; coeffs[1] = 0.5; coeffs[2] = 0.25;
        let probe = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::from_coefficients(coeffs), 0.0);
        let point = [0.5, 0.0, 0.0];
        let normal = [0.0, 0.0, 1.0];
        let expected = probe.evaluate(point, normal).expect("ok");

        let blend = LightProbeBlend::new(vec![probe], ProbeBlendMode::WeightedAverage).expect("ok");
        let result = blend.evaluate(point, normal).expect("ok");

        for c in 0..3 {
            assert!((result[c] - expected[c]).abs() < 1e-4,
                "channel {}: expected {}, got {}", c, expected[c], result[c]);
        }
    }

    // -----------------------------------------------------------------------
    // lp_blend_irradiance_sh
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_single_weight_one() {
        let mut coeffs = [0.0_f32; 27];
        for i in 0..27 { coeffs[i] = i as f32; }
        let probe = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::from_coefficients(coeffs), 0.0);
        let blended = lp_blend_irradiance_sh(&[probe], &[1.0]).expect("ok");
        for i in 0..27 {
            assert!((blended.coefficients[i] - i as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_blend_equal_weights_average() {
        let mut c1 = [0.0_f32; 27];
        let mut c2 = [0.0_f32; 27];
        c1[0] = 2.0;
        c2[0] = 4.0;
        let p1 = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::from_coefficients(c1), 0.0);
        let p2 = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::from_coefficients(c2), 0.0);
        let blended = lp_blend_irradiance_sh(&[p1, p2], &[1.0, 1.0]).expect("ok");
        assert!((blended.coefficients[0] - 3.0).abs() < 1e-5, "expected 3.0, got {}", blended.coefficients[0]);
    }

    #[test]
    fn test_blend_empty_error() {
        let r = lp_blend_irradiance_sh(&[], &[]);
        assert!(matches!(r, Err(LightProbeError::EmptyProbeList)));
    }

    // -----------------------------------------------------------------------
    // lp_evaluate_diffuse_ibl
    // -----------------------------------------------------------------------

    #[test]
    fn test_diffuse_ibl_zero_albedo() {
        let sh = IrradianceSH::new();
        let mut c = [0.0_f32; 27];
        c[0] = 5.0;
        let sh = IrradianceSH::from_coefficients(c);
        let result = lp_evaluate_diffuse_ibl([0.0, 0.0, 1.0], &sh, [0.0, 0.0, 0.0]).expect("ok");
        assert!(result[0].abs() < 1e-9);
        assert!(result[1].abs() < 1e-9);
        assert!(result[2].abs() < 1e-9);
    }

    #[test]
    fn test_diffuse_ibl_white_albedo_nonzero() {
        let mut c = [0.0_f32; 27];
        c[0] = 1.0; c[1] = 1.0; c[2] = 1.0;
        let sh = IrradianceSH::from_coefficients(c);
        let result = lp_evaluate_diffuse_ibl([0.0, 0.0, 1.0], &sh, [1.0, 1.0, 1.0]).expect("ok");
        for chan in result {
            assert!(chan > 0.0, "Expected nonzero result for white ambient+albedo");
        }
    }

    #[test]
    fn test_diffuse_ibl_zero_normal_error() {
        let sh = IrradianceSH::new();
        let r = lp_evaluate_diffuse_ibl([0.0, 0.0, 0.0], &sh, [1.0, 1.0, 1.0]);
        assert!(matches!(r, Err(LightProbeError::ZeroDirection)));
    }

    // -----------------------------------------------------------------------
    // lp_apply_ibl_to_gaussians
    // -----------------------------------------------------------------------

    #[test]
    fn test_ibl_gaussians_output_length() {
        let mut c = [0.0_f32; 27];
        c[0] = 1.0; c[1] = 1.0; c[2] = 1.0;
        let sh = IrradianceSH::from_coefficients(c);
        let n = 7usize;
        let normals = vec![0.0_f32, 0.0, 1.0].into_iter().cycle().take(n * 3).collect::<Vec<_>>();
        let albedo = vec![0.5_f32, 0.5, 0.5].into_iter().cycle().take(n * 3).collect::<Vec<_>>();
        let out = lp_apply_ibl_to_gaussians(&normals, &sh, &albedo, n).expect("ok");
        assert_eq!(out.len(), n * 3);
    }

    #[test]
    fn test_ibl_gaussians_zero_normal_error() {
        let sh = IrradianceSH::new();
        let normals = vec![0.0_f32; 3];
        let albedo = vec![1.0_f32; 3];
        let r = lp_apply_ibl_to_gaussians(&normals, &sh, &albedo, 1);
        assert!(matches!(r, Err(LightProbeError::ZeroDirection)));
    }

    #[test]
    fn test_ibl_gaussians_buffer_mismatch() {
        let sh = IrradianceSH::new();
        let normals = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 1.0]; // 2 gaussians
        let albedo = vec![0.5_f32; 3]; // 1 gaussian
        let r = lp_apply_ibl_to_gaussians(&normals, &sh, &albedo, 2);
        assert!(matches!(r, Err(LightProbeError::BufferMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // lp_compute_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_empty_error() {
        let r = lp_compute_stats(&[]);
        assert!(matches!(r, Err(LightProbeError::EmptyProbeList)));
    }

    #[test]
    fn test_stats_one_probe() {
        let mut c = [0.0_f32; 27];
        c[0] = 2.0;
        let probe = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::from_coefficients(c), 5.0);
        let stats = lp_compute_stats(&[probe]).expect("ok");
        assert_eq!(stats.n_probes, 1);
        assert!(stats.max_coefficient > 0.0);
        assert!(stats.sh_energy > 0.0);
    }

    #[test]
    fn test_stats_format_nonempty() {
        let probe = LightProbe::new([0.0, 0.0, 0.0], IrradianceSH::new(), 0.0);
        let stats = lp_compute_stats(&[probe]).expect("ok");
        let s = lp_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("n_probes"));
    }

    #[test]
    fn test_config_format_nonempty() {
        let config = LightProbeConfig::default();
        let s = lp_format_config(&config);
        assert!(!s.is_empty());
        assert!(s.contains("n_samples_projection"));
    }

    // -----------------------------------------------------------------------
    // SH energy / projection quality
    // -----------------------------------------------------------------------

    #[test]
    fn test_sh_projection_energy_white_env() {
        // White environment → L=0 coefficient energy must dominate L=1, L=2
        let n = 6_000usize;
        let dirs = lp_generate_sphere_samples(n, 777);
        let rads: Vec<[f32; 3]> = dirs.iter().map(|_| [1.0, 1.0, 1.0]).collect();
        let sh = lp_project_samples_to_sh(&dirs, &rads).expect("ok");

        // Energy in L=0 (index 0,1,2)
        let e_l0: f32 = (0..3).map(|c| sh.coefficients[c] * sh.coefficients[c]).sum();
        // Energy in L=1 (indices 3..12)
        let e_l1: f32 = (1..4).flat_map(|i| (0..3).map(move |c| i * 3 + c))
            .map(|idx| sh.coefficients[idx] * sh.coefficients[idx])
            .sum();
        // Energy in L=2 (indices 12..27)
        let e_l2: f32 = (4..9).flat_map(|i| (0..3).map(move |c| i * 3 + c))
            .map(|idx| sh.coefficients[idx] * sh.coefficients[idx])
            .sum();

        assert!(e_l0 > e_l1 * 10.0, "L=0 energy should dominate L=1 for white env");
        assert!(e_l0 > e_l2 * 10.0, "L=0 energy should dominate L=2 for white env");
    }
}
