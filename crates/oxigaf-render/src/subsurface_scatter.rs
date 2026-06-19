//! Subsurface scattering (SSS) approximations for skin rendering.
//!
//! Implements screen-space SSS based on Burley 2015 / d'Eon & Irving 2011,
//! using Gaussian mixture diffusion profiles and separable convolution.
//!
//! # Overview
//!
//! Subsurface scattering simulates light transport inside translucent materials
//! (skin, wax, marble). For faces: light enters skin, scatters through
//! dermis/epidermis layers, exits at a different point. This module provides:
//!
//! - Diffusion profiles (Gaussian mixtures) describing light spread
//! - Separable Gaussian blur (depth-aware and standard)
//! - Screen-space SSS accumulation across profile Gaussians
//! - Transmittance for thin-slab areas (ears, nose)
//! - Monte Carlo integration helpers
//! - Statistics and formatting utilities

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by subsurface scattering operations.
#[derive(Debug, Error)]
pub enum SssError {
    /// Empty image dimensions.
    #[error("Empty image: width={w}, height={h}")]
    EmptyImage { w: u32, h: u32 },

    /// Buffer length mismatch.
    #[error("Buffer length mismatch: expected {expected}, got {got}")]
    BufferMismatch { expected: usize, got: usize },

    /// Kernel radius must be positive.
    #[error("Invalid kernel radius: {r} (must be > 0)")]
    InvalidRadius { r: f32 },

    /// Profile validation failed.
    #[error("Invalid profile: {reason}")]
    InvalidProfile { reason: String },

    /// Texture size must be a power of two.
    #[error("Texture size must be power-of-two, got {size}")]
    NonPowerOfTwo { size: u32 },
}

// ---------------------------------------------------------------------------
// DiffusionProfile
// ---------------------------------------------------------------------------

/// Gaussian mixture diffusion profile R(r).
///
/// R(r) = Σᵢ wᵢ / (2π σᵢ²) · exp(−r² / (2σᵢ²))
///
/// where weights wᵢ sum to 1.0 and σᵢ are in world units (e.g. mm).
#[derive(Debug, Clone)]
pub struct DiffusionProfile {
    /// Per-Gaussian weights (must sum to 1.0).
    pub weights: Vec<f32>,
    /// Per-Gaussian widths in world units (e.g. mm).
    pub sigmas: Vec<f32>,
    /// Number of Gaussians in the mixture.
    pub n_gaussians: usize,
}

impl DiffusionProfile {
    /// Create a new diffusion profile.
    ///
    /// Validates that weights sum to approximately 1.0 and all sigmas are > 0.
    pub fn new(weights: Vec<f32>, sigmas: Vec<f32>) -> Result<Self, SssError> {
        if weights.len() != sigmas.len() {
            return Err(SssError::InvalidProfile {
                reason: format!(
                    "weights length {} != sigmas length {}",
                    weights.len(),
                    sigmas.len()
                ),
            });
        }
        if weights.is_empty() {
            return Err(SssError::InvalidProfile {
                reason: "profile must have at least one Gaussian".to_string(),
            });
        }

        let weight_sum: f32 = weights.iter().sum();
        if (weight_sum - 1.0_f32).abs() > 1e-3 {
            return Err(SssError::InvalidProfile {
                reason: format!("weights sum to {weight_sum:.6}, expected ~1.0"),
            });
        }

        for (i, &s) in sigmas.iter().enumerate() {
            if s <= 0.0 {
                return Err(SssError::InvalidProfile {
                    reason: format!("sigma[{i}] = {s} must be > 0"),
                });
            }
        }

        let n = weights.len();
        Ok(Self {
            weights,
            sigmas,
            n_gaussians: n,
        })
    }

    /// Evaluate R(r) at distance r using the Gaussian mixture.
    pub fn evaluate(&self, r: f32) -> f32 {
        let r2 = r * r;
        self.weights
            .iter()
            .zip(self.sigmas.iter())
            .map(|(&w, &sigma)| {
                let s2 = sigma * sigma;
                let norm = w / (2.0 * std::f32::consts::PI * s2);
                norm * (-r2 / (2.0 * s2)).exp()
            })
            .sum()
    }

    /// Human skin profile (Jensen et al. 3-Gaussian fit).
    ///
    /// weights = [0.0064, 0.8620, 0.1316], sigmas = [0.2, 0.7, 1.5] mm
    pub fn standard_skin() -> Self {
        // Weights already sum to 1.0: 0.0064 + 0.8620 + 0.1316 = 1.0000
        Self {
            weights: vec![0.0064, 0.8620, 0.1316],
            sigmas: vec![0.2, 0.7, 1.5],
            n_gaussians: 3,
        }
    }

    /// Marble diffusion profile.
    ///
    /// weights = [0.3, 0.4, 0.3], sigmas = [0.5, 1.5, 3.0] mm
    pub fn marble() -> Self {
        Self {
            weights: vec![0.3, 0.4, 0.3],
            sigmas: vec![0.5, 1.5, 3.0],
            n_gaussians: 3,
        }
    }

    /// Maximum effective radius: 3 × max(sigmas) (captures 99.7% of energy).
    pub fn max_radius(&self) -> f32 {
        let max_sigma = self
            .sigmas
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        3.0 * max_sigma
    }
}

// ---------------------------------------------------------------------------
// SssMaterial
// ---------------------------------------------------------------------------

/// Material parameters for subsurface scattering.
#[derive(Debug, Clone)]
pub struct SssMaterial {
    /// Diffuse albedo RGB (each channel in [0, 1]).
    pub albedo: [f32; 3],
    /// Per-channel scattering strength [0, 1].
    pub scattering_rgb: [f32; 3],
    /// Overall transmission factor [0, 1].
    pub transmission: f32,
    /// Diffusion profile describing light spread.
    pub profile: DiffusionProfile,
}

impl SssMaterial {
    /// Caucasian skin preset.
    pub fn skin_caucasian() -> Self {
        Self {
            albedo: [0.82, 0.68, 0.56],
            scattering_rgb: [1.0, 0.55, 0.30],
            transmission: 0.15,
            profile: DiffusionProfile::standard_skin(),
        }
    }

    /// Melanin-rich skin preset (higher scattering).
    pub fn skin_melanin_rich() -> Self {
        Self {
            albedo: [0.42, 0.30, 0.20],
            scattering_rgb: [0.80, 0.55, 0.38],
            transmission: 0.08,
            profile: DiffusionProfile::standard_skin(),
        }
    }

    /// Custom material.
    pub fn custom(
        albedo: [f32; 3],
        scattering: [f32; 3],
        transmission: f32,
        profile: DiffusionProfile,
    ) -> Self {
        Self {
            albedo,
            scattering_rgb: scattering,
            transmission,
            profile,
        }
    }
}

// ---------------------------------------------------------------------------
// Separable Gaussian kernels
// ---------------------------------------------------------------------------

/// Build a normalized 1D Gaussian kernel of length `2*radius+1`.
///
/// w\[i\] = exp(−(i − radius)² / (2σ²)), then normalized to sum = 1.
///
/// Returns `Err(SssError::InvalidRadius)` if `sigma_pixels <= 0`.
pub fn sss_gaussian_kernel_1d(sigma_pixels: f32, radius: usize) -> Result<Vec<f32>, SssError> {
    if sigma_pixels <= 0.0 {
        return Err(SssError::InvalidRadius { r: sigma_pixels });
    }
    let len = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(len);
    let two_sigma2 = 2.0 * sigma_pixels * sigma_pixels;
    for i in 0..len {
        let offset = i as f32 - radius as f32;
        kernel.push((-offset * offset / two_sigma2).exp());
    }
    let sum: f32 = kernel.iter().sum();
    if sum > 0.0 {
        for v in &mut kernel {
            *v /= sum;
        }
    }
    Ok(kernel)
}

/// 1D horizontal convolution with boundary clamping.
///
/// `image` is row-major with `channels` interleaved (e.g. RGB).
/// `kernel` length must equal `2*R+1` for some R (derived as `(len-1)/2`).
pub fn sss_blur_horizontal(
    image: &[f32],
    width: u32,
    height: u32,
    channels: usize,
    kernel: &[f32],
) -> Result<Vec<f32>, SssError> {
    if width == 0 || height == 0 {
        return Err(SssError::EmptyImage {
            w: width,
            h: height,
        });
    }
    let w = width as usize;
    let h = height as usize;
    let expected = w * h * channels;
    if image.len() != expected {
        return Err(SssError::BufferMismatch {
            expected,
            got: image.len(),
        });
    }
    let radius = kernel.len() / 2;
    let mut out = vec![0.0f32; expected];

    for y in 0..h {
        for x in 0..w {
            for c in 0..channels {
                let mut acc = 0.0f32;
                for (k, &kw) in kernel.iter().enumerate() {
                    let sx = (x as isize + k as isize - radius as isize).clamp(0, w as isize - 1)
                        as usize;
                    acc += kw * image[(y * w + sx) * channels + c];
                }
                out[(y * w + x) * channels + c] = acc;
            }
        }
    }
    Ok(out)
}

/// 1D vertical convolution with boundary clamping.
pub fn sss_blur_vertical(
    image: &[f32],
    width: u32,
    height: u32,
    channels: usize,
    kernel: &[f32],
) -> Result<Vec<f32>, SssError> {
    if width == 0 || height == 0 {
        return Err(SssError::EmptyImage {
            w: width,
            h: height,
        });
    }
    let w = width as usize;
    let h = height as usize;
    let expected = w * h * channels;
    if image.len() != expected {
        return Err(SssError::BufferMismatch {
            expected,
            got: image.len(),
        });
    }
    let radius = kernel.len() / 2;
    let mut out = vec![0.0f32; expected];

    for y in 0..h {
        for x in 0..w {
            for c in 0..channels {
                let mut acc = 0.0f32;
                for (k, &kw) in kernel.iter().enumerate() {
                    let sy = (y as isize + k as isize - radius as isize).clamp(0, h as isize - 1)
                        as usize;
                    acc += kw * image[(sy * w + x) * channels + c];
                }
                out[(y * w + x) * channels + c] = acc;
            }
        }
    }
    Ok(out)
}

/// Separable 2D Gaussian blur: horizontal pass then vertical pass.
///
/// The kernel radius is chosen as `ceil(3 * sigma)` (clamped to at least 1).
pub fn sss_blur_2d(
    image: &[f32],
    width: u32,
    height: u32,
    channels: usize,
    sigma: f32,
) -> Result<Vec<f32>, SssError> {
    if sigma <= 0.0 {
        return Err(SssError::InvalidRadius { r: sigma });
    }
    let radius = ((3.0 * sigma).ceil() as usize).max(1);
    let kernel = sss_gaussian_kernel_1d(sigma, radius)?;
    let horiz = sss_blur_horizontal(image, width, height, channels, &kernel)?;
    sss_blur_vertical(&horiz, width, height, channels, &kernel)
}

// ---------------------------------------------------------------------------
// Depth-aware (bilateral-style) blur
// ---------------------------------------------------------------------------

/// Depth-aware separable blur.
///
/// Like `sss_blur_2d`, but each sample's Gaussian weight is multiplied by
/// `exp(−|depth_center − depth_sample| * depth_falloff)`, preventing light
/// from bleeding across depth discontinuities.
pub fn sss_blur_depth_aware(
    image: &[f32],
    depth: &[f32],
    width: u32,
    height: u32,
    channels: usize,
    sigma: f32,
    depth_falloff: f32,
) -> Result<Vec<f32>, SssError> {
    if sigma <= 0.0 {
        return Err(SssError::InvalidRadius { r: sigma });
    }
    if width == 0 || height == 0 {
        return Err(SssError::EmptyImage {
            w: width,
            h: height,
        });
    }
    let w = width as usize;
    let h = height as usize;
    let expected_img = w * h * channels;
    let expected_dep = w * h;
    if image.len() != expected_img {
        return Err(SssError::BufferMismatch {
            expected: expected_img,
            got: image.len(),
        });
    }
    if depth.len() != expected_dep {
        return Err(SssError::BufferMismatch {
            expected: expected_dep,
            got: depth.len(),
        });
    }

    let radius = ((3.0 * sigma).ceil() as usize).max(1);
    let gauss_kernel = sss_gaussian_kernel_1d(sigma, radius)?;

    // Horizontal pass (bilateral)
    let mut horiz = vec![0.0f32; expected_img];
    for y in 0..h {
        for x in 0..w {
            let d_center = depth[y * w + x];
            let mut acc = [0.0f32; 16]; // up to 16 channels (overkill, but bounded)
            let mut weight_sum = 0.0f32;
            for (k, &gw) in gauss_kernel.iter().enumerate() {
                let sx =
                    (x as isize + k as isize - radius as isize).clamp(0, w as isize - 1) as usize;
                let d_samp = depth[y * w + sx];
                let depth_w = (-(d_center - d_samp).abs() * depth_falloff).exp();
                let w_total = gw * depth_w;
                weight_sum += w_total;
                for c in 0..channels {
                    acc[c] += w_total * image[(y * w + sx) * channels + c];
                }
            }
            for c in 0..channels {
                horiz[(y * w + x) * channels + c] = if weight_sum > 0.0 {
                    acc[c] / weight_sum
                } else {
                    0.0
                };
            }
        }
    }

    // Vertical pass (bilateral)
    let mut out = vec![0.0f32; expected_img];
    for y in 0..h {
        for x in 0..w {
            let d_center = depth[y * w + x];
            let mut acc = [0.0f32; 16];
            let mut weight_sum = 0.0f32;
            for (k, &gw) in gauss_kernel.iter().enumerate() {
                let sy =
                    (y as isize + k as isize - radius as isize).clamp(0, h as isize - 1) as usize;
                let d_samp = depth[sy * w + x];
                let depth_w = (-(d_center - d_samp).abs() * depth_falloff).exp();
                let w_total = gw * depth_w;
                weight_sum += w_total;
                for c in 0..channels {
                    acc[c] += w_total * horiz[(sy * w + x) * channels + c];
                }
            }
            for c in 0..channels {
                out[(y * w + x) * channels + c] = if weight_sum > 0.0 {
                    acc[c] / weight_sum
                } else {
                    0.0
                };
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// SssConfig / SssResult
// ---------------------------------------------------------------------------

/// Configuration for screen-space SSS rendering.
#[derive(Debug, Clone)]
pub struct SssConfig {
    /// World-space size of one pixel in mm (for sigma conversion).
    pub pixel_size_mm: f32,
    /// Overall SSS strength [0, 1]: lerp weight between original and blurred.
    pub strength: f32,
    /// Whether to use depth-aware blur (bilateral).
    pub use_depth_aware: bool,
    /// Depth falloff factor for bilateral filtering (e.g. 10.0).
    pub depth_falloff: f32,
    /// Number of Gaussian passes (1..=profile.n_gaussians).
    pub n_passes: usize,
}

/// Output of `sss_apply_profile`.
#[derive(Debug, Clone)]
pub struct SssResult {
    /// RGB f32 image data [W × H × 3].
    pub image: Vec<f32>,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Number of Gaussian passes applied.
    pub n_gaussians_applied: usize,
}

// ---------------------------------------------------------------------------
// Screen-space SSS pipeline
// ---------------------------------------------------------------------------

/// Apply a diffusion profile to an irradiance image (screen-space SSS).
///
/// For each RGB channel, accumulates Gaussian blurs weighted by the profile,
/// then lerps with the original irradiance by `strength * scattering_rgb[c]`.
///
/// # Arguments
///
/// * `irradiance` – RGB f32 irradiance image [W × H × 3].
/// * `depth` – Depth per pixel [W × H] (used when `config.use_depth_aware`).
/// * `width`, `height` – Image dimensions.
/// * `material` – SSS material with profile and per-channel scattering.
/// * `config` – SSS configuration.
pub fn sss_apply_profile(
    irradiance: &[f32],
    depth: &[f32],
    width: u32,
    height: u32,
    material: &SssMaterial,
    config: &SssConfig,
) -> Result<SssResult, SssError> {
    if width == 0 || height == 0 {
        return Err(SssError::EmptyImage {
            w: width,
            h: height,
        });
    }
    let w = width as usize;
    let h = height as usize;
    let expected = w * h * 3;
    if irradiance.len() != expected {
        return Err(SssError::BufferMismatch {
            expected,
            got: irradiance.len(),
        });
    }
    let expected_dep = w * h;
    if depth.len() != expected_dep {
        return Err(SssError::BufferMismatch {
            expected: expected_dep,
            got: depth.len(),
        });
    }

    let profile = &material.profile;
    let n_passes = config.n_passes.min(profile.n_gaussians).max(1);
    let pixel_size = config.pixel_size_mm.max(1e-6);

    // Accumulate blurred irradiance for each Gaussian pass (per channel).
    // blur_sum[channel][pixel_idx]
    let mut blur_sum: [Vec<f32>; 3] = [
        vec![0.0f32; w * h],
        vec![0.0f32; w * h],
        vec![0.0f32; w * h],
    ];

    // Extract single-channel planes from irradiance for convenience.
    // We operate channel by channel to allow per-channel sigma if desired,
    // but here all channels share the same spatial blur per Gaussian.
    // The per-channel weighting is via scattering_rgb after blurring.

    for g in 0..n_passes {
        let sigma_mm = profile.sigmas[g];
        let sigma_px = sigma_mm / pixel_size;
        let sigma_px = sigma_px.max(0.5); // at least half a pixel

        let blurred = if config.use_depth_aware {
            sss_blur_depth_aware(
                irradiance,
                depth,
                width,
                height,
                3,
                sigma_px,
                config.depth_falloff,
            )?
        } else {
            sss_blur_2d(irradiance, width, height, 3, sigma_px)?
        };

        let w_g = profile.weights[g];
        for px in 0..(w * h) {
            blur_sum[0][px] += w_g * blurred[px * 3];
            blur_sum[1][px] += w_g * blurred[px * 3 + 1];
            blur_sum[2][px] += w_g * blurred[px * 3 + 2];
        }
    }

    // Normalize by sum of profile weights used.
    let weight_sum: f32 = profile.weights[..n_passes].iter().sum();
    let inv_w = if weight_sum > 0.0 {
        1.0 / weight_sum
    } else {
        1.0
    };

    // Final lerp: result[c] = lerp(original[c], blur_sum[c] * inv_w, strength * scattering[c])
    let mut out = vec![0.0f32; expected];
    for px in 0..(w * h) {
        for c in 0..3 {
            let original = irradiance[px * 3 + c];
            let blurred_c = blur_sum[c][px] * inv_w;
            let alpha = (config.strength * material.scattering_rgb[c]).clamp(0.0, 1.0);
            out[px * 3 + c] = original + alpha * (blurred_c - original);
        }
    }

    Ok(SssResult {
        image: out,
        width,
        height,
        n_gaussians_applied: n_passes,
    })
}

// ---------------------------------------------------------------------------
// Transmittance
// ---------------------------------------------------------------------------

/// Thin-slab transmittance approximation per channel.
///
/// `T_c = transmission * exp(−thickness_mm / σ_mean)`
///
/// where `σ_mean` is the profile's mean sigma (weighted average).
pub fn sss_transmittance(thickness_mm: f32, material: &SssMaterial) -> [f32; 3] {
    let profile = &material.profile;
    let sigma_mean: f32 = profile
        .weights
        .iter()
        .zip(profile.sigmas.iter())
        .map(|(&w, &s)| w * s)
        .sum();
    let sigma_mean = sigma_mean.max(1e-6);
    let base = material.transmission * (-thickness_mm / sigma_mean).exp();
    [base, base, base]
}

/// Apply per-pixel transmittance from a thickness map.
///
/// `out[px] = back_irradiance[px] * T(thickness_map[px], material)`
///
/// Both `back_irradiance` (RGB, length W×H×3) and `thickness_map` (length W×H)
/// must be valid.
pub fn sss_apply_transmittance(
    back_irradiance: &[f32],
    thickness_map: &[f32],
    width: u32,
    height: u32,
    material: &SssMaterial,
) -> Result<Vec<f32>, SssError> {
    if width == 0 || height == 0 {
        return Err(SssError::EmptyImage {
            w: width,
            h: height,
        });
    }
    let n = (width as usize) * (height as usize);
    let expected_rgb = n * 3;
    if back_irradiance.len() != expected_rgb {
        return Err(SssError::BufferMismatch {
            expected: expected_rgb,
            got: back_irradiance.len(),
        });
    }
    if thickness_map.len() != n {
        return Err(SssError::BufferMismatch {
            expected: n,
            got: thickness_map.len(),
        });
    }

    let mut out = vec![0.0f32; expected_rgb];
    for px in 0..n {
        let t = sss_transmittance(thickness_map[px], material);
        for c in 0..3 {
            out[px * 3 + c] = back_irradiance[px * 3 + c] * t[c];
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Irradiance integration
// ---------------------------------------------------------------------------

/// Integrate a single sample: R(r) * irradiance.
pub fn sss_integrate_irradiance(irradiance: f32, r: f32, profile: &DiffusionProfile) -> f32 {
    profile.evaluate(r) * irradiance
}

/// Trapezoidal integration over (r, irradiance) sample pairs.
///
/// Computes ∫ irradiance(r) · 2π · r dr via the trapezoidal rule.
pub fn sss_radial_irradiance(samples: &[(f32, f32)]) -> f32 {
    if samples.len() < 2 {
        return samples
            .first()
            .map(|&(r, v)| v * 2.0 * std::f32::consts::PI * r)
            .unwrap_or(0.0);
    }
    let mut integral = 0.0f32;
    for i in 0..samples.len() - 1 {
        let (r0, v0) = samples[i];
        let (r1, v1) = samples[i + 1];
        let f0 = v0 * 2.0 * std::f32::consts::PI * r0;
        let f1 = v1 * 2.0 * std::f32::consts::PI * r1;
        integral += 0.5 * (f0 + f1) * (r1 - r0);
    }
    integral
}

/// Xorshift64 PRNG — next state and [0, 1) sample.
///
/// The state must not be zero; if seed is 0, it is forced to 1.
fn xorshift64(state: &mut u64) -> f64 {
    if *state == 0 {
        *state = 1;
    }
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    (*state as f64) / (u64::MAX as f64)
}

/// Monte Carlo estimate of ∫₀^max_r R(r) · 2π · r dr.
///
/// Uses importance sampling with uniform sampling in the disk
/// (importance weight = 2π·r / (π·max_r²)).
/// For a properly normalized profile this should approach 1.0.
pub fn sss_monte_carlo_integral(
    profile: &DiffusionProfile,
    max_r: f32,
    n_samples: usize,
    seed: u64,
) -> f32 {
    if n_samples == 0 || max_r <= 0.0 {
        return 0.0;
    }
    let mut state: u64 = if seed == 0 { 1 } else { seed };
    let area = std::f32::consts::PI * max_r * max_r;
    let mut sum = 0.0f64;

    for _ in 0..n_samples {
        // Sample uniformly in disk [0, max_r) by rejection or sqrt mapping.
        // Use sqrt(u) * max_r for r to get uniform density in disk.
        let u1 = xorshift64(&mut state);
        let r = (u1.sqrt() as f32) * max_r;
        let val = profile.evaluate(r);
        // Integrand: R(r) * 2π * r
        // pdf for r from sqrt-mapping: 2 * r / max_r²
        // So contribution = val * 2π * r / (2 * r / max_r²) = val * π * max_r²
        sum += val as f64 * area as f64;
    }
    (sum / n_samples as f64) as f32
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Summary statistics for an SSS pass.
#[derive(Debug, Clone)]
pub struct SssStats {
    /// Mean scatter distance in pixels (weighted sigma).
    pub mean_scatter_distance_px: f32,
    /// Maximum scatter distance in pixels (3 × max sigma in px).
    pub max_scatter_distance_px: f32,
    /// Energy conservation: integral of diffusion profile (should be ≈ 1).
    pub energy_conservation: f32,
    /// Approximate number of pixels affected by scattering.
    pub n_pixels_scattered: usize,
    /// Mean transmission factor (scalar, averaged over channels).
    pub mean_transmission: f32,
}

/// Compute statistics from an SSS result.
pub fn sss_compute_stats(
    result: &SssResult,
    material: &SssMaterial,
    config: &SssConfig,
) -> SssStats {
    let profile = &material.profile;
    let pixel_size = config.pixel_size_mm.max(1e-6);

    // Mean scatter distance (weighted average sigma, in pixels).
    let mean_sigma_mm: f32 = profile
        .weights
        .iter()
        .zip(profile.sigmas.iter())
        .map(|(&w, &s)| w * s)
        .sum();
    let mean_scatter_distance_px = mean_sigma_mm / pixel_size;

    // Max scatter distance.
    let max_sigma_mm = profile
        .sigmas
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let max_scatter_distance_px = 3.0 * max_sigma_mm / pixel_size;

    // Energy conservation via MC integration.
    let max_r = profile.max_radius();
    let energy_conservation = sss_monte_carlo_integral(profile, max_r, 4096, 0xDEAD_BEEF);

    // Pixels scattered: approximate as total pixels (all are affected).
    let n_pixels_scattered = (result.width as usize) * (result.height as usize);

    // Mean transmission (all channels equal for this model).
    let t = sss_transmittance(0.0, material);
    let mean_transmission = (t[0] + t[1] + t[2]) / 3.0;

    SssStats {
        mean_scatter_distance_px,
        max_scatter_distance_px,
        energy_conservation,
        n_pixels_scattered,
        mean_transmission,
    }
}

/// Format `SssStats` as a human-readable string.
pub fn sss_format_stats(stats: &SssStats) -> String {
    format!(
        "SssStats {{ mean_scatter={:.2}px, max_scatter={:.2}px, energy={:.4}, \
         n_scattered={}, mean_transmission={:.4} }}",
        stats.mean_scatter_distance_px,
        stats.max_scatter_distance_px,
        stats.energy_conservation,
        stats.n_pixels_scattered,
        stats.mean_transmission,
    )
}

/// Format `SssConfig` as a human-readable string.
pub fn sss_format_config(config: &SssConfig) -> String {
    format!(
        "SssConfig {{ pixel_size_mm={:.4}, strength={:.3}, use_depth_aware={}, \
         depth_falloff={:.2}, n_passes={} }}",
        config.pixel_size_mm,
        config.strength,
        config.use_depth_aware,
        config.depth_falloff,
        config.n_passes,
    )
}

/// Format `SssMaterial` as a human-readable string.
pub fn sss_format_material(material: &SssMaterial) -> String {
    format!(
        "SssMaterial {{ albedo=[{:.3},{:.3},{:.3}], scattering=[{:.3},{:.3},{:.3}], \
         transmission={:.3}, n_gaussians={} }}",
        material.albedo[0],
        material.albedo[1],
        material.albedo[2],
        material.scattering_rgb[0],
        material.scattering_rgb[1],
        material.scattering_rgb[2],
        material.transmission,
        material.profile.n_gaussians,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    // --- DiffusionProfile::evaluate ---

    #[test]
    fn test_profile_evaluate_at_zero_is_max() {
        let profile = DiffusionProfile::standard_skin();
        let v0 = profile.evaluate(0.0);
        let v1 = profile.evaluate(0.1);
        assert!(v0 > v1, "R(0) should exceed R(0.1)");
    }

    #[test]
    fn test_profile_evaluate_monotone_decreasing() {
        let profile = DiffusionProfile::standard_skin();
        let mut prev = profile.evaluate(0.0);
        for i in 1..=10 {
            let r = i as f32 * 0.2;
            let curr = profile.evaluate(r);
            assert!(
                curr <= prev,
                "R(r) should decrease: R({r})={curr} > R({:.1})={prev}",
                r - 0.2
            );
            prev = curr;
        }
    }

    #[test]
    fn test_profile_evaluate_near_zero_beyond_max_radius() {
        let profile = DiffusionProfile::standard_skin();
        let max_r = profile.max_radius();
        let v = profile.evaluate(max_r * 3.0);
        assert!(v < 1e-5, "R(3*max_r) should be ≈ 0, got {v}");
    }

    #[test]
    fn test_profile_evaluate_positive() {
        let profile = DiffusionProfile::standard_skin();
        let v = profile.evaluate(0.0);
        assert!(v > 0.0, "R(0) should be positive");
    }

    // --- DiffusionProfile::new validation ---

    #[test]
    fn test_profile_new_invalid_weights_sum() {
        let result = DiffusionProfile::new(vec![0.5, 0.1], vec![1.0, 2.0]);
        assert!(result.is_err(), "weights summing to 0.6 should error");
    }

    #[test]
    fn test_profile_new_sigma_zero_is_error() {
        let result = DiffusionProfile::new(vec![0.5, 0.5], vec![0.0, 1.0]);
        assert!(result.is_err(), "sigma=0 should error");
    }

    #[test]
    fn test_profile_new_valid() {
        let result = DiffusionProfile::new(vec![0.3, 0.7], vec![0.5, 1.5]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_profile_new_mismatched_lengths() {
        let result = DiffusionProfile::new(vec![0.5, 0.5], vec![1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_profile_new_empty_is_error() {
        let result = DiffusionProfile::new(vec![], vec![]);
        assert!(result.is_err());
    }

    // --- DiffusionProfile::standard_skin ---

    #[test]
    fn test_standard_skin_evaluate_at_zero_positive() {
        let profile = DiffusionProfile::standard_skin();
        assert!(profile.evaluate(0.0) > 0.0);
    }

    #[test]
    fn test_standard_skin_max_radius_positive() {
        let profile = DiffusionProfile::standard_skin();
        assert!(profile.max_radius() > 0.0);
    }

    #[test]
    fn test_standard_skin_max_radius_gt_middle_sigma() {
        // max sigma is 1.5, so max_radius = 4.5 > 3 * 0.7 = 2.1
        let profile = DiffusionProfile::standard_skin();
        assert!(profile.max_radius() > 3.0 * 0.7);
    }

    #[test]
    fn test_standard_skin_weights_sum_to_one() {
        let profile = DiffusionProfile::standard_skin();
        let sum: f32 = profile.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "weights sum = {sum}");
    }

    // --- Marble profile ---

    #[test]
    fn test_marble_evaluate_at_zero_positive() {
        let profile = DiffusionProfile::marble();
        assert!(profile.evaluate(0.0) > 0.0);
    }

    // --- sss_gaussian_kernel_1d ---

    #[test]
    fn test_kernel_sums_to_one() {
        let kernel = sss_gaussian_kernel_1d(2.0, 6).expect("valid kernel");
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < EPS, "kernel sum = {sum}");
    }

    #[test]
    fn test_kernel_symmetric() {
        let kernel = sss_gaussian_kernel_1d(1.5, 5).expect("valid kernel");
        let n = kernel.len();
        for i in 0..n / 2 {
            assert!((kernel[i] - kernel[n - 1 - i]).abs() < EPS);
        }
    }

    #[test]
    fn test_kernel_peak_at_center() {
        let kernel = sss_gaussian_kernel_1d(2.0, 4).expect("valid kernel");
        let radius = 4;
        let center = kernel[radius];
        for (i, &v) in kernel.iter().enumerate() {
            assert!(v <= center + EPS, "kernel[{i}]={v} > center={center}");
        }
    }

    #[test]
    fn test_kernel_invalid_sigma_zero() {
        let result = sss_gaussian_kernel_1d(0.0, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_kernel_invalid_sigma_negative() {
        let result = sss_gaussian_kernel_1d(-1.0, 3);
        assert!(result.is_err());
    }

    // --- sss_blur_horizontal ---

    #[test]
    fn test_blur_horizontal_constant_unchanged() {
        let w = 4u32;
        let h = 4u32;
        let image = vec![0.5f32; (w * h * 3) as usize];
        let kernel = sss_gaussian_kernel_1d(1.0, 3).expect("ok");
        let out = sss_blur_horizontal(&image, w, h, 3, &kernel).expect("ok");
        for &v in &out {
            assert!((v - 0.5).abs() < EPS, "expected 0.5, got {v}");
        }
    }

    #[test]
    fn test_blur_horizontal_same_dimensions() {
        let w = 8u32;
        let h = 6u32;
        let image = vec![0.3f32; (w * h * 3) as usize];
        let kernel = sss_gaussian_kernel_1d(1.0, 2).expect("ok");
        let out = sss_blur_horizontal(&image, w, h, 3, &kernel).expect("ok");
        assert_eq!(out.len(), (w * h * 3) as usize);
    }

    #[test]
    fn test_blur_horizontal_buffer_mismatch() {
        let w = 4u32;
        let h = 4u32;
        let image = vec![0.0f32; 10]; // wrong length
        let kernel = sss_gaussian_kernel_1d(1.0, 2).expect("ok");
        assert!(sss_blur_horizontal(&image, w, h, 3, &kernel).is_err());
    }

    // --- sss_blur_vertical ---

    #[test]
    fn test_blur_vertical_constant_unchanged() {
        let w = 4u32;
        let h = 4u32;
        let image = vec![0.7f32; (w * h * 3) as usize];
        let kernel = sss_gaussian_kernel_1d(1.0, 3).expect("ok");
        let out = sss_blur_vertical(&image, w, h, 3, &kernel).expect("ok");
        for &v in &out {
            assert!((v - 0.7).abs() < EPS, "expected 0.7, got {v}");
        }
    }

    #[test]
    fn test_blur_vertical_same_dimensions() {
        let w = 8u32;
        let h = 6u32;
        let image = vec![0.4f32; (w * h * 3) as usize];
        let kernel = sss_gaussian_kernel_1d(1.0, 2).expect("ok");
        let out = sss_blur_vertical(&image, w, h, 3, &kernel).expect("ok");
        assert_eq!(out.len(), (w * h * 3) as usize);
    }

    // --- sss_blur_2d ---

    #[test]
    fn test_blur_2d_single_pixel_unchanged() {
        let image = vec![0.6f32, 0.4f32, 0.2f32]; // 1×1×3
        let out = sss_blur_2d(&image, 1, 1, 3, 1.0).expect("ok");
        assert_eq!(out.len(), 3);
        for (i, (&a, &b)) in image.iter().zip(out.iter()).enumerate() {
            assert!((a - b).abs() < EPS, "channel {i}: expected {a}, got {b}");
        }
    }

    #[test]
    fn test_blur_2d_preserves_mean() {
        let w = 8u32;
        let h = 8u32;
        let n = (w * h * 3) as usize;
        let mut image = vec![0.0f32; n];
        // Alternating values.
        for (i, val) in image.iter_mut().enumerate().take(n) {
            *val = if i % 2 == 0 { 1.0 } else { 0.0 };
        }
        let mean_in: f32 = image.iter().sum::<f32>() / n as f32;
        let out = sss_blur_2d(&image, w, h, 3, 2.0).expect("ok");
        let mean_out: f32 = out.iter().sum::<f32>() / n as f32;
        assert!(
            (mean_in - mean_out).abs() < 0.05,
            "mean changed: {mean_in} -> {mean_out}"
        );
    }

    #[test]
    fn test_blur_2d_reduces_variance() {
        let w = 8u32;
        let h = 8u32;
        let n = (w * h * 3) as usize;
        let mut image = vec![0.0f32; n];
        for (i, val) in image.iter_mut().enumerate().take(n) {
            *val = (i % 3) as f32 * 0.5;
        }
        let var_in = variance(&image);
        let out = sss_blur_2d(&image, w, h, 3, 2.0).expect("ok");
        let var_out = variance(&out);
        assert!(
            var_out < var_in,
            "variance should decrease: {var_in} -> {var_out}"
        );
    }

    #[test]
    fn test_blur_2d_large_sigma_uniform() {
        // For a central patch far from edges, large sigma → near-uniform.
        let w = 16u32;
        let h = 16u32;
        let n = (w * h * 3) as usize;
        let mut image = vec![0.0f32; n];
        // Fill center 8×8 with 1.0, rest 0.
        for y in 4..12usize {
            for x in 4..12usize {
                for c in 0..3 {
                    image[(y * 16 + x) * 3 + c] = 1.0;
                }
            }
        }
        let out = sss_blur_2d(&image, w, h, 3, 8.0).expect("ok");
        // Center pixel should still be non-zero (blurred inward too).
        let center = out[(8 * 16 + 8) * 3];
        assert!(
            center > 0.0,
            "center should be > 0 after large blur, got {center}"
        );
    }

    #[test]
    fn test_blur_2d_invalid_sigma() {
        let image = vec![0.5f32; 4 * 4 * 3];
        assert!(sss_blur_2d(&image, 4, 4, 3, 0.0).is_err());
    }

    // --- sss_blur_depth_aware ---

    #[test]
    fn test_depth_aware_uniform_depth_same_as_plain() {
        let w = 4u32;
        let h = 4u32;
        let n = (w * h) as usize;
        let mut image = vec![0.0f32; n * 3];
        // Random-ish values.
        for (i, val) in image.iter_mut().enumerate() {
            *val = (i as f32 * 0.1) % 1.0;
        }
        let depth = vec![0.5f32; n];
        let plain = sss_blur_2d(&image, w, h, 3, 1.5).expect("ok");
        let aware = sss_blur_depth_aware(&image, &depth, w, h, 3, 1.5, 0.001).expect("ok");
        // With near-zero depth_falloff and uniform depth, results should be close.
        for (a, b) in plain.iter().zip(aware.iter()) {
            assert!((a - b).abs() < 0.05, "plain={a}, depth_aware={b}");
        }
    }

    #[test]
    fn test_depth_aware_discontinuity_reduces_bleeding() {
        // Left half depth=0, right half depth=10. Strong falloff.
        let w = 8u32;
        let h = 4u32;
        let n = (w * h) as usize;
        let mut image = vec![0.0f32; n * 3];
        // Left half lit, right half dark.
        for y in 0..(h as usize) {
            for x in 0..(w as usize / 2) {
                for c in 0..3 {
                    image[(y * w as usize + x) * 3 + c] = 1.0;
                }
            }
        }
        let mut depth = vec![0.0f32; n];
        for y in 0..(h as usize) {
            for x in (w as usize / 2)..(w as usize) {
                depth[y * w as usize + x] = 10.0;
            }
        }
        // Strong depth falloff — bleed should be minimal across boundary.
        let aware = sss_blur_depth_aware(&image, &depth, w, h, 3, 2.0, 100.0).expect("ok");
        // Pixel just right of boundary should be nearly 0.
        let x_right = w as usize / 2;
        let y_mid = h as usize / 2;
        let bleed = aware[(y_mid * w as usize + x_right) * 3];
        // Without depth awareness, bleed would be significant; with strong falloff it's small.
        assert!(
            bleed < 0.5,
            "bleed across discontinuity should be small: {bleed}"
        );
    }

    #[test]
    fn test_depth_aware_buffer_mismatch_depth() {
        let w = 4u32;
        let h = 4u32;
        let image = vec![0.5f32; (w * h * 3) as usize];
        let depth = vec![0.5f32; 5]; // wrong size
        let result = sss_blur_depth_aware(&image, &depth, w, h, 3, 1.0, 10.0);
        assert!(result.is_err());
    }

    // --- SssMaterial ---

    #[test]
    fn test_skin_caucasian_scattering_in_range() {
        let mat = SssMaterial::skin_caucasian();
        for &v in &mat.scattering_rgb {
            assert!((0.0..=1.0).contains(&v), "scattering {v} out of [0,1]");
        }
    }

    #[test]
    fn test_skin_melanin_rich_scattering_in_range() {
        let mat = SssMaterial::skin_melanin_rich();
        for &v in &mat.scattering_rgb {
            assert!((0.0..=1.0).contains(&v), "scattering {v} out of [0,1]");
        }
    }

    // --- sss_transmittance ---

    #[test]
    fn test_transmittance_zero_thickness() {
        let mat = SssMaterial::skin_caucasian();
        let t = sss_transmittance(0.0, &mat);
        // exp(0) = 1, so t = transmission
        for &v in &t {
            assert!(
                (v - mat.transmission).abs() < EPS,
                "t={v} expected {}",
                mat.transmission
            );
        }
    }

    #[test]
    fn test_transmittance_large_thickness_near_zero() {
        let mat = SssMaterial::skin_caucasian();
        let t = sss_transmittance(1000.0, &mat);
        for &v in &t {
            assert!(
                v < 1e-6,
                "large thickness should give near-zero transmittance, got {v}"
            );
        }
    }

    #[test]
    fn test_transmittance_values_in_range() {
        let mat = SssMaterial::skin_caucasian();
        for thickness in [0.0f32, 0.5, 1.0, 5.0] {
            let t = sss_transmittance(thickness, &mat);
            for &v in &t {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "transmittance {v} out of [0,1] at thickness {thickness}"
                );
            }
        }
    }

    // --- sss_apply_transmittance ---

    #[test]
    fn test_apply_transmittance_zero_map() {
        let w = 2u32;
        let h = 2u32;
        let mat = SssMaterial::skin_caucasian();
        let irradiance = vec![1.0f32; (w * h * 3) as usize];
        let thickness = vec![0.0f32; (w * h) as usize];
        let out = sss_apply_transmittance(&irradiance, &thickness, w, h, &mat).expect("ok");
        // All irradiance * transmission (at thickness 0)
        for &v in &out {
            assert!(
                (v - mat.transmission).abs() < EPS,
                "expected {}, got {v}",
                mat.transmission
            );
        }
    }

    #[test]
    fn test_apply_transmittance_wrong_size() {
        let mat = SssMaterial::skin_caucasian();
        let irradiance = vec![0.5f32; 10];
        let thickness = vec![0.0f32; 4];
        assert!(sss_apply_transmittance(&irradiance, &thickness, 2, 2, &mat).is_err());
    }

    // --- sss_integrate_irradiance ---

    #[test]
    fn test_integrate_irradiance_r_zero() {
        let profile = DiffusionProfile::standard_skin();
        let r0_profile = profile.evaluate(0.0);
        let result = sss_integrate_irradiance(1.0, 0.0, &profile);
        assert!((result - r0_profile).abs() < EPS);
    }

    // --- sss_radial_irradiance ---

    #[test]
    fn test_radial_irradiance_two_points() {
        // With constant irradiance=1 and r in [0, 1]:
        // integral ≈ 2π * ∫₀¹ r dr = 2π * 0.5 = π
        let samples = [(0.0f32, 1.0f32), (1.0f32, 1.0f32)];
        let result = sss_radial_irradiance(&samples);
        let expected = std::f32::consts::PI;
        assert!(
            (result - expected).abs() < 0.1,
            "expected {expected}, got {result}"
        );
    }

    #[test]
    fn test_radial_irradiance_empty() {
        let result = sss_radial_irradiance(&[]);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_radial_irradiance_single_point() {
        let samples = [(0.5f32, 2.0f32)];
        let result = sss_radial_irradiance(&samples);
        let expected = 2.0 * 2.0 * std::f32::consts::PI * 0.5;
        assert!((result - expected).abs() < EPS);
    }

    // --- sss_monte_carlo_integral ---

    #[test]
    fn test_monte_carlo_integral_standard_skin_near_one() {
        let profile = DiffusionProfile::standard_skin();
        let max_r = profile.max_radius();
        // With 16384 samples, should be within 20% of 1.0
        let val = sss_monte_carlo_integral(&profile, max_r, 16384, 12345);
        assert!(
            (val - 1.0).abs() < 0.2,
            "MC integral for standard_skin should ≈ 1.0, got {val}"
        );
    }

    #[test]
    fn test_monte_carlo_integral_zero_samples() {
        let profile = DiffusionProfile::standard_skin();
        let val = sss_monte_carlo_integral(&profile, 5.0, 0, 42);
        assert_eq!(val, 0.0);
    }

    // --- sss_apply_profile ---

    #[test]
    fn test_apply_profile_same_dimensions() {
        let w = 8u32;
        let h = 8u32;
        let mat = SssMaterial::skin_caucasian();
        let config = SssConfig {
            pixel_size_mm: 0.1,
            strength: 0.5,
            use_depth_aware: false,
            depth_falloff: 10.0,
            n_passes: 2,
        };
        let irradiance = vec![0.5f32; (w * h * 3) as usize];
        let depth = vec![1.0f32; (w * h) as usize];
        let result = sss_apply_profile(&irradiance, &depth, w, h, &mat, &config).expect("ok");
        assert_eq!(result.width, w);
        assert_eq!(result.height, h);
        assert_eq!(result.image.len(), (w * h * 3) as usize);
    }

    #[test]
    fn test_apply_profile_strength_zero_preserves_original() {
        let w = 4u32;
        let h = 4u32;
        let mat = SssMaterial::skin_caucasian();
        let config = SssConfig {
            pixel_size_mm: 0.1,
            strength: 0.0,
            use_depth_aware: false,
            depth_falloff: 10.0,
            n_passes: 1,
        };
        let irradiance: Vec<f32> = (0..((w * h * 3) as usize))
            .map(|i| (i as f32 * 0.01) % 1.0)
            .collect();
        let depth = vec![1.0f32; (w * h) as usize];
        let result = sss_apply_profile(&irradiance, &depth, w, h, &mat, &config).expect("ok");
        for (a, b) in irradiance.iter().zip(result.image.iter()) {
            assert!(
                (a - b).abs() < EPS,
                "strength=0 should preserve original: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_apply_profile_strength_one_uniform_mean_preserved() {
        let w = 8u32;
        let h = 8u32;
        let mat = SssMaterial::skin_caucasian();
        let config = SssConfig {
            pixel_size_mm: 0.5,
            strength: 1.0,
            use_depth_aware: false,
            depth_falloff: 10.0,
            n_passes: 1,
        };
        let irradiance = vec![0.4f32; (w * h * 3) as usize];
        let depth = vec![1.0f32; (w * h) as usize];
        let result = sss_apply_profile(&irradiance, &depth, w, h, &mat, &config).expect("ok");
        let mean: f32 = result.image.iter().sum::<f32>() / result.image.len() as f32;
        // Uniform input → uniform output → mean ≈ 0.4
        assert!(
            (mean - 0.4).abs() < 0.05,
            "mean should be ≈ 0.4, got {mean}"
        );
    }

    #[test]
    fn test_apply_profile_empty_image_error() {
        let mat = SssMaterial::skin_caucasian();
        let config = SssConfig {
            pixel_size_mm: 0.1,
            strength: 0.5,
            use_depth_aware: false,
            depth_falloff: 10.0,
            n_passes: 1,
        };
        let result = sss_apply_profile(&[], &[], 0, 0, &mat, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_profile_buffer_mismatch_error() {
        let w = 4u32;
        let h = 4u32;
        let mat = SssMaterial::skin_caucasian();
        let config = SssConfig {
            pixel_size_mm: 0.1,
            strength: 0.5,
            use_depth_aware: false,
            depth_falloff: 10.0,
            n_passes: 1,
        };
        let bad_irradiance = vec![0.5f32; 10]; // wrong length
        let depth = vec![1.0f32; (w * h) as usize];
        let result = sss_apply_profile(&bad_irradiance, &depth, w, h, &mat, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_profile_depth_aware_runs() {
        let w = 4u32;
        let h = 4u32;
        let mat = SssMaterial::skin_caucasian();
        let config = SssConfig {
            pixel_size_mm: 0.5,
            strength: 0.5,
            use_depth_aware: true,
            depth_falloff: 5.0,
            n_passes: 1,
        };
        let irradiance = vec![0.5f32; (w * h * 3) as usize];
        let depth = vec![1.0f32; (w * h) as usize];
        let result = sss_apply_profile(&irradiance, &depth, w, h, &mat, &config);
        assert!(result.is_ok());
    }

    // --- sss_compute_stats ---

    #[test]
    fn test_compute_stats_energy_near_one() {
        let mat = SssMaterial::skin_caucasian();
        let config = SssConfig {
            pixel_size_mm: 0.1,
            strength: 0.5,
            use_depth_aware: false,
            depth_falloff: 10.0,
            n_passes: 3,
        };
        let result = SssResult {
            image: vec![0.5f32; 4 * 4 * 3],
            width: 4,
            height: 4,
            n_gaussians_applied: 3,
        };
        let stats = sss_compute_stats(&result, &mat, &config);
        assert!(
            (stats.energy_conservation - 1.0).abs() < 0.25,
            "energy_conservation = {}",
            stats.energy_conservation
        );
    }

    // --- sss_format_* ---

    #[test]
    fn test_format_stats_nonempty() {
        let stats = SssStats {
            mean_scatter_distance_px: 2.5,
            max_scatter_distance_px: 10.0,
            energy_conservation: 0.98,
            n_pixels_scattered: 1024,
            mean_transmission: 0.15,
        };
        let s = sss_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("mean_scatter"));
    }

    #[test]
    fn test_format_config_nonempty() {
        let config = SssConfig {
            pixel_size_mm: 0.1,
            strength: 0.8,
            use_depth_aware: true,
            depth_falloff: 10.0,
            n_passes: 3,
        };
        let s = sss_format_config(&config);
        assert!(!s.is_empty());
        assert!(s.contains("strength"));
    }

    #[test]
    fn test_format_material_nonempty() {
        let mat = SssMaterial::skin_caucasian();
        let s = sss_format_material(&mat);
        assert!(!s.is_empty());
        assert!(s.contains("albedo"));
    }

    // --- Edge cases ---

    #[test]
    fn test_blur_2d_1x1_single_pixel() {
        let image = vec![0.3f32, 0.5f32, 0.7f32];
        let out = sss_blur_2d(&image, 1, 1, 3, 2.0).expect("ok");
        assert_eq!(out.len(), 3);
        for (i, (&a, &b)) in image.iter().zip(out.iter()).enumerate() {
            assert!((a - b).abs() < EPS, "ch {i}: in={a} out={b}");
        }
    }

    #[test]
    fn test_profile_n_gaussians_correct() {
        let profile = DiffusionProfile::standard_skin();
        assert_eq!(profile.n_gaussians, 3);
    }

    #[test]
    fn test_apply_transmittance_output_length() {
        let w = 3u32;
        let h = 2u32;
        let mat = SssMaterial::skin_caucasian();
        let irradiance = vec![1.0f32; (w * h * 3) as usize];
        let thickness = vec![0.5f32; (w * h) as usize];
        let out = sss_apply_transmittance(&irradiance, &thickness, w, h, &mat).expect("ok");
        assert_eq!(out.len(), (w * h * 3) as usize);
    }

    #[test]
    fn test_depth_aware_same_output_length() {
        let w = 4u32;
        let h = 4u32;
        let image = vec![0.5f32; (w * h * 3) as usize];
        let depth = vec![1.0f32; (w * h) as usize];
        let out = sss_blur_depth_aware(&image, &depth, w, h, 3, 1.5, 10.0).expect("ok");
        assert_eq!(out.len(), (w * h * 3) as usize);
    }

    #[test]
    fn test_profile_max_radius_equals_3_max_sigma() {
        let profile = DiffusionProfile::standard_skin();
        let max_s = profile
            .sigmas
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((profile.max_radius() - 3.0 * max_s).abs() < EPS);
    }

    #[test]
    fn test_monte_carlo_seed_zero_handled() {
        let profile = DiffusionProfile::standard_skin();
        // seed=0 should not panic or return NaN.
        let val = sss_monte_carlo_integral(&profile, 5.0, 100, 0);
        assert!(
            val.is_finite(),
            "MC integral should be finite even for seed=0"
        );
    }

    #[test]
    fn test_apply_profile_n_gaussians_applied() {
        let w = 4u32;
        let h = 4u32;
        let mat = SssMaterial::skin_caucasian();
        let config = SssConfig {
            pixel_size_mm: 0.5,
            strength: 0.5,
            use_depth_aware: false,
            depth_falloff: 10.0,
            n_passes: 2,
        };
        let irradiance = vec![0.5f32; (w * h * 3) as usize];
        let depth = vec![1.0f32; (w * h) as usize];
        let result = sss_apply_profile(&irradiance, &depth, w, h, &mat, &config).expect("ok");
        assert_eq!(result.n_gaussians_applied, 2);
    }

    // Helper: compute variance of a f32 slice.
    fn variance(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / data.len() as f32
    }
}
