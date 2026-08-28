//! Monte Carlo SH projection: sphere sampling, radiance→irradiance SH
//! projection, and equirectangular panorama projection.

use std::f32::consts::PI;

use super::config::LightProbeConfig;
use super::error::LightProbeError;
use super::irradiance::IrradianceSH;
use super::sampling::bilinear_sample_rgb;
use super::sh_math::lp_sh_full_9;

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

/// Ramamoorthi & Hanrahan cosine-lobe convolution coefficients, indexed by
/// SH band `l` (0, 1, 2). Multiplying a *radiance* SH projection's band-`l`
/// coefficients by `LP_COSINE_LOBE_A[l]` converts it into an *irradiance* SH
/// projection (the clamped-cosine transfer function has already been folded
/// in), per "An Efficient Representation for Irradiance Environment Maps"
/// (Ramamoorthi & Hanrahan, 2001): `A0 = π`, `A1 = 2π/3`, `A2 = π/4`.
pub(super) const LP_COSINE_LOBE_A: [f32; 3] = [PI, 2.0 * PI / 3.0, PI / 4.0];

/// Project a set of direction-radiance samples to L=2 **irradiance** SH
/// coefficients via Monte Carlo.
///
/// This first computes the plain radiance projection
/// `c_lm = (4π / N) * Σ_i L(ω_i) * Y_lm(ω_i)`, then applies the
/// Ramamoorthi & Hanrahan cosine-lobe convolution (see
/// `LP_COSINE_LOBE_A`) so the result is directly usable as
/// [`IrradianceSH`] — i.e. `IrradianceSH::evaluate` returns the actual
/// irradiance `E(n)`, and [`super::lp_evaluate_diffuse_ibl`]'s division by
/// `π` yields correctly-scaled outgoing radiance instead of over-weighting
/// the L1/L2 bands by ~1.5x/~4x.
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
    // Cosine-lobe convolution: band 0 (index 0..3) *= A0, band 1 (3..12) *= A1,
    // band 2 (12..27) *= A2 — interleaved RGB layout, 3 floats per basis index.
    for (basis_i, coeff_chunk) in coeffs.chunks_exact_mut(3).enumerate() {
        let band = match basis_i {
            0 => 0,
            1..=3 => 1,
            _ => 2,
        };
        let a = LP_COSINE_LOBE_A[band];
        for v in coeff_chunk.iter_mut() {
            *v *= a;
        }
    }
    Ok(IrradianceSH::from_coefficients(coeffs))
}

/// Project an equirectangular (lat-long) panorama image to L=2 SH coefficients.
///
/// `image`: RGB f32 row-major, length = `width * height * 3`.
/// Sampling via Monte Carlo with `n_samples` directions.
///
/// # Errors
/// - `LightProbeError::InvalidImageDimensions` if `width == 0 || height == 0`.
/// - `LightProbeError::BufferMismatch` if image length != width * height * 3.
pub fn lp_project_latitude_longitude(
    image: &[f32],
    width: u32,
    height: u32,
    n_samples: usize,
    seed: u64,
) -> Result<IrradianceSH, LightProbeError> {
    if width == 0 || height == 0 {
        return Err(LightProbeError::InvalidImageDimensions { width, height });
    }
    let expected = (width as usize) * (height as usize) * 3;
    if image.len() != expected {
        return Err(LightProbeError::BufferMismatch {
            expected,
            got: image.len(),
        });
    }

    let dirs = lp_generate_sphere_samples(n_samples, seed);
    let mut radiances = Vec::with_capacity(n_samples);

    for d in &dirs {
        // Convert unit sphere direction to equirectangular (u, v)
        // φ = atan2(z, x) mapped to [0, 1], θ = acos(y) mapped to [0, 1]
        let phi = d[2].atan2(d[0]); // [-π, π]
        let theta = d[1].clamp(-1.0, 1.0).acos(); // [0, π]
        let u = (phi / (2.0 * PI) + 0.5).rem_euclid(1.0);
        let v = theta / PI;
        let rgb = bilinear_sample_rgb(image, width, height, u, v);
        radiances.push(rgb);
    }

    lp_project_samples_to_sh(&dirs, &radiances)
}

/// Like [`lp_project_latitude_longitude`], but takes the sample count from
/// `config.n_samples_projection` instead of a bare parameter.
///
/// # Errors
/// - `LightProbeError::InvalidImageDimensions` if `width == 0 || height == 0`.
/// - `LightProbeError::BufferMismatch` if image length != width * height * 3.
pub fn lp_project_latitude_longitude_with_config(
    image: &[f32],
    width: u32,
    height: u32,
    config: &LightProbeConfig,
    seed: u64,
) -> Result<IrradianceSH, LightProbeError> {
    lp_project_latitude_longitude(image, width, height, config.n_samples_projection, seed)
}
