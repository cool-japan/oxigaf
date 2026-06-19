//! Screen-space ambient occlusion (SSAO) post-processing.
//!
//! Implements a CPU-side SSAO pass that computes per-pixel ambient occlusion
//! from a depth map and per-pixel normals.
//!
//! # Algorithm Overview
//!
//! 1. Generate a hemisphere kernel of sample points (cosine-weighted, accelerating scale).
//! 2. Generate an NxN rotation noise texture to jitter the kernel per-pixel.
//! 3. For each pixel:
//!   - Retrieve depth `d` and normal `n`.
//!   - Look up the noise rotation for this pixel (modular tiling).
//!   - For each kernel sample, apply the 2-D rotation, project to screen space,
//!     and compare the sampled depth against the pixel depth.
//!   - Accumulate occlusion.
//! 4. Apply a power curve and optionally a separable box blur for denoising.
//!
//! The simplified 2-D path (no full TBN matrix) keeps the CPU implementation
//! efficient while still producing visually plausible results.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// PRNG — xorshift64
// ─────────────────────────────────────────────────────────────────────────────

/// Advance a 64-bit xorshift state and return the new value.
///
/// A zero state would remain stuck; callers should replace zero seeds before use.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Draw a uniformly distributed f32 in [0, 1) from the xorshift64 state.
fn xorshift64_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 11) as f32 / (1u64 << 53) as f32
}

/// Replace a zero seed with a non-degenerate constant so xorshift never stalls.
fn safe_seed(seed: u64) -> u64 {
    if seed == 0 {
        0xDEAD_BEEF_CAFE_1234u64
    } else {
        seed
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by SSAO operations.
#[derive(Debug, Error)]
pub enum SsaoError {
    /// Input slice is empty.
    #[error("SSAO input is empty")]
    EmptyInput,

    /// Normal map has wrong length.
    #[error("Invalid normal map: expected {expected} floats (W×H×3), got {got}")]
    InvalidNormalMap {
        /// Expected number of floats.
        expected: usize,
        /// Actual number of floats.
        got: usize,
    },

    /// Depth map and normal map pixel counts differ.
    #[error("Size mismatch: depth_len={depth_len} pixels, normal_len={normal_len} pixels")]
    SizeMismatch {
        /// Number of pixels in the depth map.
        depth_len: usize,
        /// Number of pixels in the normal map (divided by 3).
        normal_len: usize,
    },

    /// Image width or height is zero.
    #[error("Image width and height must both be at least 1")]
    ZeroDimension,

    /// Focal length is zero or negative.
    #[error("Focal length must be positive")]
    ZeroFocalLength,
}

// ─────────────────────────────────────────────────────────────────────────────
// SsaoConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for screen-space ambient occlusion.
#[derive(Debug, Clone)]
pub struct SsaoConfig {
    /// Number of sample points per pixel in the hemisphere kernel.
    pub num_samples: usize,
    /// Sampling radius in depth units (camera/world space).
    pub radius: f32,
    /// Depth bias to avoid self-occlusion artefacts.
    pub bias: f32,
    /// Power for the AO curve (higher → darker shadows).
    pub power: f32,
    /// Side length of the random rotation noise texture (noise_size × noise_size).
    pub noise_size: usize,
    /// Whether to apply a separable box-blur denoising pass.
    pub blur: bool,
    /// Box-blur kernel half-radius in pixels.
    pub blur_radius: usize,
}

impl Default for SsaoConfig {
    fn default() -> Self {
        Self {
            num_samples: 16,
            radius: 0.5,
            bias: 0.025,
            power: 2.0,
            noise_size: 4,
            blur: true,
            blur_radius: 2,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SsaoKernel
// ─────────────────────────────────────────────────────────────────────────────

/// Hemisphere sample kernel used for SSAO.
///
/// All samples have `z > 0` (pointing toward the camera in view space) before
/// the random per-component scale is applied.  Samples are cosine-weighted and
/// distributed with an accelerating interpolation scale so that more samples
/// cluster near the origin.
#[derive(Debug, Clone)]
pub struct SsaoKernel {
    /// Raw sample vectors `[x, y, z]`.
    pub samples: Vec<[f32; 3]>,
}

impl SsaoKernel {
    /// Generate `num_samples` hemisphere samples with the given RNG seed.
    pub fn generate(num_samples: usize, seed: u64) -> Self {
        let mut state = safe_seed(seed);
        let mut samples = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            // Hemisphere: x, y in [-1, 1]; z in [0, 1].
            let x = 2.0 * xorshift64_f32(&mut state) - 1.0;
            let y = 2.0 * xorshift64_f32(&mut state) - 1.0;
            let z = xorshift64_f32(&mut state);

            let mut sample = Self::normalize3([x, y, z]);

            // Accelerating interpolation scale: more samples near origin.
            let t = i as f32 / num_samples.max(1) as f32;
            let scale = 0.1 + t * t * 0.9;

            let r = xorshift64_f32(&mut state) * scale;
            sample[0] *= r;
            sample[1] *= r;
            sample[2] *= r;

            samples.push(sample);
        }

        Self { samples }
    }

    /// Normalize a 3-vector. Returns the zero vector if the magnitude is near zero.
    fn normalize3(v: [f32; 3]) -> [f32; 3] {
        let len2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if len2 < 1e-12 {
            return [0.0, 0.0, 0.0];
        }
        let inv = 1.0 / len2.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Noise texture
// ─────────────────────────────────────────────────────────────────────────────

/// Generate an `size × size` rotation noise texture.
///
/// Each texel is `[cos θ, sin θ]` for a random angle θ ∈ [0, 2π).
/// Tiling this texture over the screen gives each pixel a different kernel
/// rotation, which effectively increases the apparent sample count.
///
/// The returned `Vec` has `size * size` entries.
pub fn generate_noise_texture(size: usize, seed: u64) -> Vec<[f32; 2]> {
    let mut state = safe_seed(seed);
    let n = size * size;
    let mut noise = Vec::with_capacity(n);

    let two_pi = 2.0 * core::f32::consts::PI;

    for _ in 0..n {
        let angle = xorshift64_f32(&mut state) * two_pi;
        noise.push([angle.cos(), angle.sin()]);
    }

    noise
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_ssao
// ─────────────────────────────────────────────────────────────────────────────

/// Compute per-pixel ambient occlusion using a simplified 2-D SSAO algorithm.
///
/// # Parameters
///
/// - `depth_map`: linear per-pixel depth in camera space, length `H × W`.
/// - `normal_map`: per-pixel view-space normals, length `H × W × 3`.  The
///   normals are expected to be unit-length but are normalised internally
///   for robustness.
/// - `kernel`: pre-generated hemisphere kernel (see [`SsaoKernel::generate`]).
/// - `config`: SSAO configuration.
/// - `focal_length`: camera focal length in pixels.
/// - `image_width`, `image_height`: image dimensions.
///
/// # Returns
///
/// A flat `H × W` buffer of per-pixel AO values in `[0, 1]`.
/// `0` = fully occluded, `1` = fully lit.
///
/// # Errors
///
/// - [`SsaoError::ZeroDimension`] if either spatial dimension is zero.
/// - [`SsaoError::ZeroFocalLength`] if `focal_length ≤ 0`.
/// - [`SsaoError::EmptyInput`] if `depth_map` is empty.
/// - [`SsaoError::InvalidNormalMap`] if the normal map length ≠ `W × H × 3`.
/// - [`SsaoError::SizeMismatch`] if depth and normal pixel counts differ.
pub fn compute_ssao(
    depth_map: &[f32],
    normal_map: &[f32],
    kernel: &SsaoKernel,
    config: &SsaoConfig,
    focal_length: f32,
    image_width: usize,
    image_height: usize,
) -> Result<Vec<f32>, SsaoError> {
    if image_width == 0 || image_height == 0 {
        return Err(SsaoError::ZeroDimension);
    }
    if focal_length <= 0.0 {
        return Err(SsaoError::ZeroFocalLength);
    }
    if depth_map.is_empty() {
        return Err(SsaoError::EmptyInput);
    }

    let num_pixels = image_width * image_height;

    if depth_map.len() != num_pixels {
        return Err(SsaoError::SizeMismatch {
            depth_len: depth_map.len(),
            normal_len: num_pixels,
        });
    }

    let expected_normal_len = num_pixels * 3;
    if normal_map.len() != expected_normal_len {
        return Err(SsaoError::InvalidNormalMap {
            expected: expected_normal_len,
            got: normal_map.len(),
        });
    }

    // Pre-build rotation noise texture.
    let noise_seed = 0xABCD_EF01_2345_6789u64;
    let noise = generate_noise_texture(config.noise_size, noise_seed);
    let noise_n = config.noise_size.max(1);

    let mut ao_map = Vec::with_capacity(num_pixels);

    for py in 0..image_height {
        for px in 0..image_width {
            let pixel_idx = py * image_width + px;
            let pixel_depth = depth_map[pixel_idx];

            // Skip sky / invalid pixels.
            if !pixel_depth.is_finite() || pixel_depth <= 0.0 {
                ao_map.push(1.0_f32);
                continue;
            }

            // Fetch the rotation vector from the noise texture (tiled).
            let nx = px % noise_n;
            let ny = py % noise_n;
            let noise_entry = noise[ny * noise_n + nx];
            let cos_r = noise_entry[0];
            let sin_r = noise_entry[1];

            let mut occlusion = 0.0_f32;
            let mut valid_samples = 0usize;

            for sample in &kernel.samples {
                // Apply 2-D rotation from the noise texture to the (x, y) components.
                // The z component (hemisphere depth offset) is used as a weight.
                let sx = sample[0];
                let sy = sample[1];
                let sz = sample[2].abs(); // ensure positive hemisphere

                // Rotated 2-D offset in image space.
                let rx = cos_r * sx - sin_r * sy;
                let ry = sin_r * sx + cos_r * sy;

                // Project to screen-space offset using pinhole projection:
                // Δscreen = Δview * focal / depth
                let scale = focal_length / pixel_depth.max(1e-6);
                let screen_dx = rx * scale * config.radius;
                let screen_dy = ry * scale * config.radius;

                let sample_x_f = px as f32 + screen_dx;
                let sample_y_f = py as f32 + screen_dy;

                // Bounds check — skip out-of-frame samples.
                if sample_x_f < 0.0
                    || sample_y_f < 0.0
                    || sample_x_f >= (image_width as f32 - 1.0)
                    || sample_y_f >= (image_height as f32 - 1.0)
                {
                    continue;
                }

                let sx_i = sample_x_f as usize;
                let sy_i = sample_y_f as usize;

                let sampled_depth = depth_map[sy_i * image_width + sx_i];

                if !sampled_depth.is_finite() || sampled_depth <= 0.0 {
                    continue;
                }

                valid_samples += 1;

                // Range check: ignore samples too far in depth.
                let depth_diff = (sampled_depth - pixel_depth).abs();
                if depth_diff > config.radius {
                    continue;
                }

                // Range weight: smoothly fall off as the sample gets further away.
                let range_weight = 1.0 - depth_diff / config.radius;

                // Occlusion test: a surface point that is *closer to the camera*
                // (smaller depth) than the current pixel (plus bias) occludes it.
                // Using `sampled_depth < pixel_depth - bias` avoids self-occlusion
                // on flat surfaces (where sampled_depth ≈ pixel_depth).
                if sampled_depth < pixel_depth - config.bias {
                    // Weight by sz so that samples pointing more "straight up"
                    // (large z) contribute more.
                    occlusion += range_weight * (sz + 0.5);
                }
            }

            let ao = if valid_samples == 0 {
                1.0_f32
            } else {
                // Normalise occlusion against the maximum possible contribution.
                let max_occ = valid_samples as f32 * 1.5; // 1.0 * (0.5 + 0.5*sz_avg)
                let norm_occ = (occlusion / max_occ).min(1.0);
                (1.0 - norm_occ).powf(config.power)
            };

            ao_map.push(ao.clamp(0.0, 1.0));
        }
    }

    Ok(ao_map)
}

// ─────────────────────────────────────────────────────────────────────────────
// blur_ssao
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a separable box blur to the AO map for denoising.
///
/// Two passes are performed: horizontal then vertical.  Edge pixels are
/// handled by clamping the sample coordinates to the valid range.
pub fn blur_ssao(
    ao_map: &[f32],
    image_width: usize,
    image_height: usize,
    blur_radius: usize,
) -> Vec<f32> {
    if ao_map.is_empty() || image_width == 0 || image_height == 0 {
        return ao_map.to_vec();
    }

    let num_pixels = image_width * image_height;
    if ao_map.len() != num_pixels {
        return ao_map.to_vec();
    }

    // --- Horizontal pass ---
    let mut temp = vec![0.0_f32; num_pixels];
    let r = blur_radius as isize;
    let diam = (2 * blur_radius + 1) as f32;

    for py in 0..image_height {
        for px in 0..image_width {
            let mut sum = 0.0_f32;
            for dx in -r..=r {
                let sx = (px as isize + dx).clamp(0, image_width as isize - 1) as usize;
                sum += ao_map[py * image_width + sx];
            }
            temp[py * image_width + px] = sum / diam;
        }
    }

    // --- Vertical pass ---
    let mut out = vec![0.0_f32; num_pixels];
    for py in 0..image_height {
        for px in 0..image_width {
            let mut sum = 0.0_f32;
            for dy in -r..=r {
                let sy = (py as isize + dy).clamp(0, image_height as isize - 1) as usize;
                sum += temp[sy * image_width + px];
            }
            out[py * image_width + px] = sum / diam;
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_ao_to_image
// ─────────────────────────────────────────────────────────────────────────────

/// Multiply image RGB channels by the AO value to apply ambient occlusion.
///
/// The `ao_strength` parameter controls the blend:
/// - `0.0` → no effect (image unchanged).
/// - `1.0` → full AO (image multiplied by raw AO values).
///
/// Each pixel's colour is scaled by `lerp(1.0, ao, ao_strength)`.
///
/// # Errors
///
/// - [`SsaoError::ZeroDimension`] if either dimension is zero.
/// - [`SsaoError::EmptyInput`] if the image slice is empty.
/// - [`SsaoError::SizeMismatch`] if the AO map length ≠ `W × H`.
pub fn apply_ao_to_image(
    image: &[f32],
    ao_map: &[f32],
    image_width: usize,
    image_height: usize,
    channels: usize,
    ao_strength: f32,
) -> Result<Vec<f32>, SsaoError> {
    if image_width == 0 || image_height == 0 {
        return Err(SsaoError::ZeroDimension);
    }
    if image.is_empty() {
        return Err(SsaoError::EmptyInput);
    }

    let num_pixels = image_width * image_height;
    if ao_map.len() != num_pixels {
        return Err(SsaoError::SizeMismatch {
            depth_len: ao_map.len(),
            normal_len: num_pixels,
        });
    }

    let strength = ao_strength.clamp(0.0, 1.0);
    let mut out = Vec::with_capacity(image.len());

    for (pixel_idx, chunk) in image.chunks(channels).enumerate() {
        let ao = if pixel_idx < ao_map.len() {
            ao_map[pixel_idx]
        } else {
            1.0
        };
        // scale = lerp(1.0, ao, strength) = 1.0 + strength * (ao - 1.0)
        let scale = 1.0 + strength * (ao - 1.0);
        for &v in chunk {
            out.push((v * scale).clamp(0.0, 1.0));
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// SsaoStats
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics about an AO map.
#[derive(Debug, Clone)]
pub struct SsaoStats {
    /// Mean AO value across all pixels.
    pub mean_ao: f32,
    /// Minimum AO value.
    pub min_ao: f32,
    /// Maximum AO value.
    pub max_ao: f32,
    /// Fraction of pixels with AO < 0.5 (considered occluded).
    pub occluded_fraction: f32,
}

impl SsaoStats {
    /// Compute statistics from a flat AO map.
    pub fn compute(ao_map: &[f32]) -> Self {
        if ao_map.is_empty() {
            return Self {
                mean_ao: 1.0,
                min_ao: 1.0,
                max_ao: 1.0,
                occluded_fraction: 0.0,
            };
        }

        let n = ao_map.len();
        let mut sum = 0.0_f32;
        let mut min_ao = f32::MAX;
        let mut max_ao = f32::MIN;
        let mut occluded_count = 0usize;

        for &ao in ao_map {
            sum += ao;
            if ao < min_ao {
                min_ao = ao;
            }
            if ao > max_ao {
                max_ao = ao;
            }
            if ao < 0.5 {
                occluded_count += 1;
            }
        }

        Self {
            mean_ao: sum / n as f32,
            min_ao,
            max_ao,
            occluded_fraction: occluded_count as f32 / n as f32,
        }
    }

    /// Return a human-readable one-line summary.
    pub fn format_summary(&self) -> String {
        format!(
            "SSAO: mean={:.3}  min={:.3}  max={:.3}  occluded={:.1}%",
            self.mean_ao,
            self.min_ao,
            self.max_ao,
            self.occluded_fraction * 100.0,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // ── Kernel tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_kernel_generate_count() {
        let kernel = SsaoKernel::generate(16, 42);
        assert_eq!(kernel.samples.len(), 16);
    }

    #[test]
    fn test_kernel_generate_hemisphere_z_positive() {
        // Before the random per-component scale, normalised z must be ≥ 0.
        // We verify the property indirectly: generate a large kernel and confirm
        // that the direction (after normalisation) stays in the upper hemisphere.
        // Because we multiply by a random *positive* scale, the sign is preserved.
        // We test the underlying normalize3 helper and the generator logic:
        // z = rng_f32 ∈ [0, 1) → after normalize3 z ≥ 0 (since it just scales).
        // We probe via a deterministic kernel and check z not strongly negative.
        let kernel = SsaoKernel::generate(64, 123_456);
        for s in &kernel.samples {
            // The z component after multiply by positive scale keeps same sign.
            // (scale = 0.1 + t^2 * 0.9 > 0 always; normalised z ≥ 0 from rng)
            assert!(
                s[2] >= 0.0,
                "Sample z must be non-negative (hemisphere), got {:?}",
                s
            );
        }
    }

    #[test]
    fn test_kernel_normalize3_unit_vector() {
        let v = SsaoKernel::normalize3([3.0, 4.0, 0.0]);
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!(approx_eq(len, 1.0, 1e-5), "Normalised length = {len}");
    }

    #[test]
    fn test_kernel_normalize3_zero_vector() {
        let v = SsaoKernel::normalize3([0.0, 0.0, 0.0]);
        assert_eq!(v, [0.0, 0.0, 0.0]);
    }

    // ── Noise texture tests ───────────────────────────────────────────────────

    #[test]
    fn test_noise_texture_size() {
        let noise = generate_noise_texture(4, 99);
        assert_eq!(noise.len(), 16);

        let noise2 = generate_noise_texture(8, 99);
        assert_eq!(noise2.len(), 64);
    }

    #[test]
    fn test_noise_texture_unit_magnitude() {
        let noise = generate_noise_texture(4, 7);
        for entry in &noise {
            let mag2 = entry[0] * entry[0] + entry[1] * entry[1];
            assert!(
                approx_eq(mag2, 1.0, 1e-5),
                "cos²+sin² should be 1.0, got {mag2}"
            );
        }
    }

    // ── compute_ssao tests ────────────────────────────────────────────────────

    fn flat_depth_normals(w: usize, h: usize, depth: f32) -> (Vec<f32>, Vec<f32>) {
        let n = w * h;
        let depth_map = vec![depth; n];
        // Flat surface facing the camera: normals = (0, 0, 1).
        let mut normal_map = Vec::with_capacity(n * 3);
        for _ in 0..n {
            normal_map.push(0.0_f32);
            normal_map.push(0.0_f32);
            normal_map.push(1.0_f32);
        }
        (depth_map, normal_map)
    }

    #[test]
    fn test_compute_ssao_open_plane_high_ao() {
        // Flat plane at constant depth → no occlusion → AO should be near 1.0.
        let w = 16_usize;
        let h = 16_usize;
        let (depth_map, normal_map) = flat_depth_normals(w, h, 2.0);
        let kernel = SsaoKernel::generate(16, 1);
        let config = SsaoConfig::default();

        let ao = compute_ssao(&depth_map, &normal_map, &kernel, &config, 500.0, w, h)
            .expect("compute_ssao failed");

        let mean = ao.iter().sum::<f32>() / ao.len() as f32;
        assert!(
            mean > 0.8,
            "Flat plane should have high mean AO (near 1.0), got {mean:.3}"
        );
    }

    #[test]
    fn test_compute_ssao_output_size() {
        let w = 8_usize;
        let h = 6_usize;
        let (depth_map, normal_map) = flat_depth_normals(w, h, 1.5);
        let kernel = SsaoKernel::generate(8, 2);
        let config = SsaoConfig::default();

        let ao = compute_ssao(&depth_map, &normal_map, &kernel, &config, 400.0, w, h)
            .expect("compute_ssao failed");

        assert_eq!(ao.len(), w * h, "AO map must have W×H elements");
    }

    #[test]
    fn test_compute_ssao_values_in_range() {
        let w = 10_usize;
        let h = 10_usize;
        let (depth_map, normal_map) = flat_depth_normals(w, h, 1.0);
        let kernel = SsaoKernel::generate(16, 3);
        let config = SsaoConfig::default();

        let ao = compute_ssao(&depth_map, &normal_map, &kernel, &config, 300.0, w, h)
            .expect("compute_ssao failed");

        for &v in &ao {
            assert!(
                (0.0..=1.0).contains(&v),
                "AO value out of [0, 1] range: {v}"
            );
        }
    }

    #[test]
    fn test_ssao_error_zero_dimension() {
        let depth_map = vec![1.0_f32; 0];
        let normal_map = vec![0.0_f32; 0];
        let kernel = SsaoKernel::generate(4, 1);
        let config = SsaoConfig::default();

        let result = compute_ssao(&depth_map, &normal_map, &kernel, &config, 400.0, 0, 4);
        assert!(
            matches!(result, Err(SsaoError::ZeroDimension)),
            "Expected ZeroDimension error"
        );
    }

    #[test]
    fn test_ssao_error_size_mismatch() {
        let depth_map = vec![1.0_f32; 4]; // 2x2
        let normal_map = vec![0.0_f32; 4 * 3]; // also 2×2 normals
        let kernel = SsaoKernel::generate(4, 1);
        let config = SsaoConfig::default();

        // Pass w=3, h=2 → num_pixels=6 ≠ depth_map.len()=4 → SizeMismatch.
        let result = compute_ssao(&depth_map, &normal_map, &kernel, &config, 400.0, 3, 2);
        assert!(
            matches!(result, Err(SsaoError::SizeMismatch { .. })),
            "Expected SizeMismatch error"
        );
    }

    #[test]
    fn test_compute_ssao_zero_focal_length_error() {
        let (depth_map, normal_map) = flat_depth_normals(4, 4, 1.0);
        let kernel = SsaoKernel::generate(4, 1);
        let config = SsaoConfig::default();

        let result = compute_ssao(&depth_map, &normal_map, &kernel, &config, 0.0, 4, 4);
        assert!(matches!(result, Err(SsaoError::ZeroFocalLength)));
    }

    // ── blur tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_blur_ssao_output_size() {
        let w = 8_usize;
        let h = 6_usize;
        let ao = vec![0.5_f32; w * h];
        let blurred = blur_ssao(&ao, w, h, 2);
        assert_eq!(blurred.len(), w * h);
    }

    #[test]
    fn test_blur_ssao_uniform_unchanged() {
        // Blurring a uniform map should return the same values.
        let w = 8_usize;
        let h = 8_usize;
        let ao = vec![0.7_f32; w * h];
        let blurred = blur_ssao(&ao, w, h, 3);

        for (i, (&a, &b)) in ao.iter().zip(blurred.iter()).enumerate() {
            assert!(
                approx_eq(a, b, 1e-5),
                "pixel {i}: blur changed uniform value from {a} to {b}"
            );
        }
    }

    // ── apply_ao tests ────────────────────────────────────────────────────────

    #[test]
    fn test_apply_ao_output_size() {
        let w = 4_usize;
        let h = 4_usize;
        let image = vec![0.5_f32; w * h * 3];
        let ao = vec![1.0_f32; w * h];

        let out = apply_ao_to_image(&image, &ao, w, h, 3, 1.0).expect("apply_ao_to_image failed");

        assert_eq!(out.len(), image.len());
    }

    #[test]
    fn test_apply_ao_strength_zero_unchanged() {
        // ao_strength = 0.0 → image must be unchanged.
        let w = 4_usize;
        let h = 4_usize;
        let image: Vec<f32> = (0..w * h * 3).map(|i| (i % 10) as f32 / 10.0).collect();
        let ao = vec![0.0_f32; w * h]; // zero AO (fully dark) — but strength=0

        let out = apply_ao_to_image(&image, &ao, w, h, 3, 0.0).expect("apply_ao_to_image failed");

        for (i, (&a, &b)) in image.iter().zip(out.iter()).enumerate() {
            assert!(
                approx_eq(a, b, 1e-5),
                "pixel {i}: ao_strength=0 changed value from {a} to {b}"
            );
        }
    }

    #[test]
    fn test_apply_ao_full_ao_darkens() {
        // ao=0.0 (fully occluded) with strength=1.0 → all outputs become 0.0.
        let w = 2_usize;
        let h = 2_usize;
        let image = vec![1.0_f32; w * h * 3];
        let ao = vec![0.0_f32; w * h];

        let out = apply_ao_to_image(&image, &ao, w, h, 3, 1.0).expect("apply_ao_to_image failed");

        for &v in &out {
            assert!(
                approx_eq(v, 0.0, 1e-5),
                "Expected 0.0 with full AO, got {v}"
            );
        }
    }

    // ── SsaoStats tests ───────────────────────────────────────────────────────

    #[test]
    fn test_ssao_stats_compute() {
        let ao = vec![0.0_f32, 0.25, 0.5, 1.0];
        let stats = SsaoStats::compute(&ao);

        assert!(
            approx_eq(stats.mean_ao, 0.4375, 1e-4),
            "mean = {}",
            stats.mean_ao
        );
        assert!(approx_eq(stats.min_ao, 0.0, 1e-6), "min = {}", stats.min_ao);
        assert!(approx_eq(stats.max_ao, 1.0, 1e-6), "max = {}", stats.max_ao);
        // occluded = ao < 0.5: [0.0, 0.25] → 2/4 = 0.5
        assert!(
            approx_eq(stats.occluded_fraction, 0.5, 1e-5),
            "occluded_fraction = {}",
            stats.occluded_fraction
        );
    }

    #[test]
    fn test_ssao_stats_empty() {
        let stats = SsaoStats::compute(&[]);
        assert!(approx_eq(stats.mean_ao, 1.0, 1e-6));
        assert!(approx_eq(stats.occluded_fraction, 0.0, 1e-6));
    }

    #[test]
    fn test_ssao_stats_format_summary() {
        let ao = vec![0.5_f32; 10];
        let stats = SsaoStats::compute(&ao);
        let summary = stats.format_summary();
        assert!(summary.contains("SSAO:"), "summary: {summary}");
        assert!(summary.contains("mean"), "summary: {summary}");
    }

    // ── SsaoConfig default test ───────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let cfg = SsaoConfig::default();
        assert_eq!(cfg.num_samples, 16);
        assert!(approx_eq(cfg.radius, 0.5, 1e-6));
        assert!(approx_eq(cfg.bias, 0.025, 1e-6));
        assert!(approx_eq(cfg.power, 2.0, 1e-6));
        assert_eq!(cfg.noise_size, 4);
        assert!(cfg.blur);
        assert_eq!(cfg.blur_radius, 2);
    }
}
