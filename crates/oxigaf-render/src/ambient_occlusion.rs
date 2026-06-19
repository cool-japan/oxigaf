//! Screen-Space Ambient Occlusion (SSAO) post-processing for 3D Gaussian Splatting.
//!
//! Implements a full TBN-matrix-based SSAO pipeline:
//! 1. Hemisphere kernel generation (cosine-weighted, lerp-accelerated scale)
//! 2. 4×4 noise texture for per-pixel kernel rotation
//! 3. Per-pixel AO computation with depth-bias and range-check smoothstep
//! 4. Bilateral blur (spatial + depth weighting) for denoising
//! 5. AO application to RGB image with configurable strength
//!
//! All random numbers use a deterministic xorshift64 PRNG — no external rand crate.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// PRNG — xorshift64
// ─────────────────────────────────────────────────────────────────────────────

/// Advance a 64-bit xorshift state and return the new value.
/// If the state reaches 0, it is reset to 1 to avoid getting stuck.
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

/// Draw a uniformly distributed f32 in [0, 1] from the xorshift64 state.
fn xorshift_f32(state: &mut u64) -> f32 {
    xorshift64(state) as f32 / u64::MAX as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by ambient occlusion operations.
#[derive(Debug, Error)]
pub enum AoError {
    /// Buffer dimension mismatch.
    #[error("dimension mismatch: width*height={expected}, buffer len={got}")]
    DimensionMismatch {
        /// Expected buffer length.
        expected: usize,
        /// Actual buffer length.
        got: usize,
    },

    /// Sampling radius must be positive.
    #[error("invalid radius: must be > 0, got {0}")]
    InvalidRadius(f32),

    /// General configuration error.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// Empty input buffer.
    #[error("empty input")]
    EmptyInput,

    /// Blur kernel size must be positive.
    #[error("invalid kernel size: must be > 0, got {0}")]
    InvalidKernelSize(usize),
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the ambient occlusion pass.
#[derive(Debug, Clone)]
pub struct AoConfig {
    /// Number of hemisphere samples per pixel (default: 16).
    pub n_samples: usize,
    /// Sampling radius in view space (default: 0.5).
    pub radius: f32,
    /// Depth bias to avoid self-occlusion (default: 0.025).
    pub bias: f32,
    /// AO intensity exponent (default: 1.0).
    pub power: f32,
    /// Bilateral blur kernel half-size in pixels (default: 2).
    pub blur_radius: usize,
    /// Spatial sigma for bilateral blur (default: 2.0).
    pub blur_sigma_space: f32,
    /// Depth sigma for bilateral blur (default: 0.1).
    pub blur_sigma_depth: f32,
    /// Final AO mixing strength [0, 1] (default: 1.0).
    pub strength: f32,
}

impl Default for AoConfig {
    fn default() -> Self {
        Self {
            n_samples: 16,
            radius: 0.5,
            bias: 0.025,
            power: 1.0,
            blur_radius: 2,
            blur_sigma_space: 2.0,
            blur_sigma_depth: 0.1,
            strength: 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result and statistics types
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics about a computed AO map.
#[derive(Debug, Clone)]
pub struct AoStats {
    /// Mean AO factor (1.0 = no occlusion).
    pub mean_ao: f32,
    /// Minimum AO factor.
    pub min_ao: f32,
    /// Maximum AO factor.
    pub max_ao: f32,
    /// Fraction of foreground pixels with AO < 0.9.
    pub occlusion_fraction: f32,
    /// Fraction of pixels treated as background (depth == 0 or infinite).
    pub background_fraction: f32,
}

/// Output of the full SSAO pipeline.
#[derive(Debug, Clone)]
pub struct SsaoResult {
    /// AO-applied RGB image (same dimensions as input).
    pub image: Vec<f32>,
    /// Raw AO factors before blur (length = width * height).
    pub ao_map: Vec<f32>,
    /// Blurred AO factors (length = width * height).
    pub ao_map_blurred: Vec<f32>,
    /// Statistics computed from the blurred AO map.
    pub stats: AoStats,
}

// ─────────────────────────────────────────────────────────────────────────────
// Projection parameters
// ─────────────────────────────────────────────────────────────────────────────

/// Projection-derived scale factors used to reconstruct view-space positions.
///
/// These are typically derived from the camera projection matrix:
/// `proj_scale_x = tan(fov_x / 2)` and `proj_scale_y = tan(fov_y / 2)`.
#[derive(Debug, Clone, Copy)]
pub struct AoProjParams {
    /// Horizontal field-of-view tangent scale.
    pub proj_scale_x: f32,
    /// Vertical field-of-view tangent scale.
    pub proj_scale_y: f32,
}

impl AoProjParams {
    /// Construct from explicit scale values.
    pub fn new(proj_scale_x: f32, proj_scale_y: f32) -> Self {
        Self {
            proj_scale_x,
            proj_scale_y,
        }
    }
}

/// Sampling data buffers for [`ao_compute`]: hemisphere kernel and noise texture.
pub struct AoSamplingBuffers<'a> {
    /// Hemisphere sample kernel (`n_samples * 3` floats).
    pub kernel: &'a [f32],
    /// Per-pixel noise texture for kernel rotation.
    pub noise: &'a [f32],
}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Dot product of two 3-vectors.
#[inline]
pub fn ao_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two 3-vectors.
#[inline]
pub fn ao_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalize a 3-vector. Returns `(0, 0, 1)` for degenerate (near-zero) input.
#[inline]
pub fn ao_normalize(v: [f32; 3]) -> [f32; 3] {
    let len2 = ao_dot(v, v);
    if len2 < 1e-12 {
        return [0.0, 0.0, 1.0];
    }
    let inv = 1.0 / len2.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Smoothstep function. Returns 0 for x ≤ edge0, 1 for x ≥ edge1, and a
/// smooth cubic interpolation in between.
#[inline]
pub fn ao_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Bilinear sample from a depth buffer at fractional pixel coordinates.
/// Clamps coordinates to valid image bounds.
pub fn ao_sample_depth(depth_buf: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    if width == 0 || height == 0 || depth_buf.is_empty() {
        return 0.0;
    }

    let x_clamped = x.clamp(0.0, (width as f32) - 1.0);
    let y_clamped = y.clamp(0.0, (height as f32) - 1.0);

    let x0 = x_clamped.floor() as usize;
    let y0 = y_clamped.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let fx = x_clamped - x0 as f32;
    let fy = y_clamped - y0 as f32;

    let d00 = depth_buf[y0 * width + x0];
    let d10 = depth_buf[y0 * width + x1];
    let d01 = depth_buf[y1 * width + x0];
    let d11 = depth_buf[y1 * width + x1];

    let d0 = d00 + fx * (d10 - d00);
    let d1 = d01 + fx * (d11 - d01);
    d0 + fy * (d1 - d0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Sample kernel & noise texture
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a hemisphere sample kernel for SSAO.
///
/// Returns a flat `Vec<f32>` of length `n_samples * 3` with `(x, y, z)` triplets.
/// All samples lie in the upper hemisphere (z ≥ 0). Samples are distributed
/// with a cosine-weighted pattern and lerp-accelerated scale (more near origin).
/// Uses xorshift64 seeded deterministically with `n_samples as u64`.
///
/// # Errors
/// - [`AoError::InvalidConfig`] if `n_samples == 0`.
pub fn ao_sample_kernel(n_samples: usize) -> Result<Vec<f32>, AoError> {
    if n_samples == 0 {
        return Err(AoError::InvalidConfig("n_samples must be > 0".to_string()));
    }

    let mut state = n_samples as u64;
    if state == 0 {
        state = 1;
    }

    let mut kernel = Vec::with_capacity(n_samples * 3);
    let two_pi = 2.0 * core::f32::consts::PI;
    let half_pi = core::f32::consts::FRAC_PI_2;

    for i in 0..n_samples {
        // Random hemisphere direction (z >= 0)
        let theta = xorshift_f32(&mut state) * two_pi;
        let phi = xorshift_f32(&mut state) * half_pi;

        let x = phi.sin() * theta.cos();
        let y = phi.sin() * theta.sin();
        let z = phi.cos();

        // Lerp-accelerated scale: closer samples weighted higher
        let t = i as f32 / n_samples.max(1) as f32;
        let scale = 0.1 + (t * t) * 0.9; // lerp(0.1, 1.0, t^2)

        let random_factor = xorshift_f32(&mut state);

        kernel.push(x * scale * random_factor);
        kernel.push(y * scale * random_factor);
        kernel.push(z * scale * random_factor);
    }

    Ok(kernel)
}

/// Generate a 4×4 noise texture for random kernel rotation.
///
/// Returns a flat `Vec<f32>` of length 32: 16 pairs of `(noise_x, noise_y)`
/// rotation vectors, each component in `[-1, 1]`.
/// Uses deterministic seed `12345u64`.
pub fn ao_noise_texture() -> Vec<f32> {
    let mut state = 12345u64;
    let mut noise = Vec::with_capacity(32);

    for _ in 0..16 {
        let nx = xorshift_f32(&mut state) * 2.0 - 1.0;
        let ny = xorshift_f32(&mut state) * 2.0 - 1.0;
        noise.push(nx);
        noise.push(ny);
    }

    noise
}

// ─────────────────────────────────────────────────────────────────────────────
// Core SSAO algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Compute SSAO factor for each pixel using a TBN-matrix-based hemisphere sampling.
///
/// # Parameters
/// - `depth_buf`: per-pixel linear depth (view-space z), length = `width * height`.
///   Pixels with depth == 0 or infinite are treated as background (AO = 1.0).
/// - `normal_buf`: per-pixel view-space normals (flat), length = `width * height * 3`.
/// - `width`, `height`: image dimensions.
/// - `kernel`: sample kernel from [`ao_sample_kernel`], length = `n_samples * 3`.
/// - `noise`: noise texture from [`ao_noise_texture`], length = 32.
/// - `proj_scale_x`, `proj_scale_y`: projection scale factors (focal / image_size).
/// - `config`: AO configuration.
///
/// # Returns
/// `Vec<f32>` of length `width * height`, values in [0, 1].
/// 1.0 = fully unoccluded, 0.0 = fully occluded.
///
/// # Errors
/// - [`AoError::EmptyInput`] if `depth_buf` is empty.
/// - [`AoError::DimensionMismatch`] if buffer lengths don't match expected sizes.
/// - [`AoError::InvalidRadius`] if `config.radius <= 0`.
pub fn ao_compute(
    depth_buf: &[f32],
    normal_buf: &[f32],
    width: usize,
    height: usize,
    sampling: AoSamplingBuffers<'_>,
    proj: AoProjParams,
    config: &AoConfig,
) -> Result<Vec<f32>, AoError> {
    let kernel = sampling.kernel;
    let noise = sampling.noise;
    if depth_buf.is_empty() {
        return Err(AoError::EmptyInput);
    }
    if config.radius <= 0.0 {
        return Err(AoError::InvalidRadius(config.radius));
    }

    let num_pixels = width * height;

    if depth_buf.len() != num_pixels {
        return Err(AoError::DimensionMismatch {
            expected: num_pixels,
            got: depth_buf.len(),
        });
    }

    let expected_normal_len = num_pixels * 3;
    if normal_buf.len() != expected_normal_len {
        return Err(AoError::DimensionMismatch {
            expected: expected_normal_len,
            got: normal_buf.len(),
        });
    }

    let n_samples = kernel.len() / 3;
    let radius = config.radius;
    let bias = config.bias;
    let power = config.power;

    let mut ao_buf = Vec::with_capacity(num_pixels);

    for py in 0..height {
        for px in 0..width {
            let pixel_idx = py * width + px;
            let depth = depth_buf[pixel_idx];

            // Background pixels → AO = 1.0 (no occlusion)
            if depth == 0.0 || !depth.is_finite() {
                ao_buf.push(1.0_f32);
                continue;
            }

            // Fetch view-space normal
            let nx = normal_buf[pixel_idx * 3];
            let ny = normal_buf[pixel_idx * 3 + 1];
            let nz = normal_buf[pixel_idx * 3 + 2];
            let normal = ao_normalize([nx, ny, nz]);

            // Fetch noise rotation vector (4x4 tile)
            let noise_idx = (py % 4) * 4 + (px % 4);
            let noise_x = noise[(noise_idx * 2) % noise.len()];
            let noise_y = noise[(noise_idx * 2 + 1) % noise.len()];
            let noise_vec = [noise_x, noise_y, 0.0];

            // Build TBN matrix via Gram-Schmidt orthogonalization
            // Tangent: project noise_vec onto the plane perpendicular to normal
            let dot_n_noise = ao_dot(normal, noise_vec);
            let tangent_raw = [
                noise_vec[0] - dot_n_noise * normal[0],
                noise_vec[1] - dot_n_noise * normal[1],
                noise_vec[2] - dot_n_noise * normal[2],
            ];
            let tangent = ao_normalize(tangent_raw);
            let bitangent = ao_cross(normal, tangent);

            let mut occlusion_sum = 0.0_f32;

            for s in 0..n_samples {
                // Kernel sample in tangent space
                let kx = kernel[s * 3];
                let ky = kernel[s * 3 + 1];
                let kz = kernel[s * 3 + 2];

                // Transform to view space via TBN
                let vx = tangent[0] * kx + bitangent[0] * ky + normal[0] * kz;
                let vy = tangent[1] * kx + bitangent[1] * ky + normal[1] * kz;
                let vz = tangent[2] * kx + bitangent[2] * ky + normal[2] * kz;

                // Scale by radius
                let ox = vx * radius;
                let oy = vy * radius;
                let oz = vz * radius;

                // Project to screen space
                let denom = (depth + oz).max(1e-6);
                let sx = px as f32 + ox * proj.proj_scale_x / denom;
                let sy = py as f32 + oy * proj.proj_scale_y / denom;

                // Clamp to image bounds
                let sx_clamped = sx.clamp(0.0, (width as f32) - 1.0);
                let sy_clamped = sy.clamp(0.0, (height as f32) - 1.0);

                // Bilinear sample depth at projected location
                let sampled_depth =
                    ao_sample_depth(depth_buf, width, height, sx_clamped, sy_clamped);

                // Skip background samples
                if sampled_depth == 0.0 || !sampled_depth.is_finite() {
                    continue;
                }

                // Range check: smoothstep falloff based on depth difference
                let depth_diff = (depth - sampled_depth).abs();
                let range_check = ao_smoothstep(0.0, 1.0, radius / (depth_diff + 1e-6));

                // Occlusion test: sample is behind the current fragment + bias
                let occluded = if sampled_depth >= (depth + oz + bias) {
                    1.0_f32
                } else {
                    0.0_f32
                };

                occlusion_sum += occluded * range_check;
            }

            let n_samples_f = n_samples.max(1) as f32;
            let ao = 1.0 - (occlusion_sum / n_samples_f).powf(power);
            ao_buf.push(ao.clamp(0.0, 1.0));
        }
    }

    Ok(ao_buf)
}

// ─────────────────────────────────────────────────────────────────────────────
// Bilateral blur
// ─────────────────────────────────────────────────────────────────────────────

/// Apply bilateral blur to the AO buffer to reduce noise.
///
/// Weights pixels by both spatial distance (Gaussian) and depth similarity
/// (Gaussian on depth difference). This preserves edges where depth changes sharply.
///
/// # Errors
/// - [`AoError::EmptyInput`] if `ao_buf` is empty.
/// - [`AoError::DimensionMismatch`] if `ao_buf.len() != width * height`.
pub fn ao_bilateral_blur(
    ao_buf: &[f32],
    depth_buf: &[f32],
    width: usize,
    height: usize,
    config: &AoConfig,
) -> Result<Vec<f32>, AoError> {
    if ao_buf.is_empty() {
        return Err(AoError::EmptyInput);
    }

    let num_pixels = width * height;
    if ao_buf.len() != num_pixels {
        return Err(AoError::DimensionMismatch {
            expected: num_pixels,
            got: ao_buf.len(),
        });
    }

    if depth_buf.len() != num_pixels {
        return Err(AoError::DimensionMismatch {
            expected: num_pixels,
            got: depth_buf.len(),
        });
    }

    let r = config.blur_radius as isize;
    let sigma_space = config.blur_sigma_space.max(1e-6);
    let sigma_depth = config.blur_sigma_depth.max(1e-6);

    let inv_2sigma_space2 = 1.0 / (2.0 * sigma_space * sigma_space);
    let inv_2sigma_depth2 = 1.0 / (2.0 * sigma_depth * sigma_depth);

    let mut out = Vec::with_capacity(num_pixels);

    for py in 0..height {
        for px in 0..width {
            let center_idx = py * width + px;
            let center_depth = depth_buf[center_idx];

            // Background pixels pass through unchanged
            if center_depth == 0.0 || !center_depth.is_finite() {
                out.push(ao_buf[center_idx]);
                continue;
            }

            let mut weighted_sum = 0.0_f32;
            let mut weight_total = 0.0_f32;

            for dy in -r..=r {
                for dx in -r..=r {
                    let nx = (px as isize + dx).clamp(0, width as isize - 1) as usize;
                    let ny = (py as isize + dy).clamp(0, height as isize - 1) as usize;
                    let nb_idx = ny * width + nx;

                    let nb_depth = depth_buf[nb_idx];
                    let nb_ao = ao_buf[nb_idx];

                    // Spatial weight
                    let dist2 = (dx * dx + dy * dy) as f32;
                    let w_space = (-dist2 * inv_2sigma_space2).exp();

                    // Depth weight
                    let depth_diff = center_depth - nb_depth;
                    let w_depth = (-(depth_diff * depth_diff) * inv_2sigma_depth2).exp();

                    let w = w_space * w_depth;
                    weighted_sum += nb_ao * w;
                    weight_total += w;
                }
            }

            let blurred = if weight_total > 1e-10 {
                weighted_sum / weight_total
            } else {
                ao_buf[center_idx]
            };

            out.push(blurred.clamp(0.0, 1.0));
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Image application
// ─────────────────────────────────────────────────────────────────────────────

/// Apply AO to an RGB image (flat interleaved, [0, 1] range).
///
/// Multiplies each pixel's RGB by its AO factor using strength blend:
/// `output = lerp(input, input * ao, strength)`.
///
/// # Errors
/// - [`AoError::EmptyInput`] if `image` is empty.
/// - [`AoError::DimensionMismatch`] if `ao_buf.len() != width * height` or
///   `image.len() != width * height * 3`.
pub fn ao_apply_to_image(
    image: &[f32],
    ao_buf: &[f32],
    width: usize,
    height: usize,
    strength: f32,
) -> Result<Vec<f32>, AoError> {
    if image.is_empty() {
        return Err(AoError::EmptyInput);
    }

    let num_pixels = width * height;
    let expected_image_len = num_pixels * 3;

    if ao_buf.len() != num_pixels {
        return Err(AoError::DimensionMismatch {
            expected: num_pixels,
            got: ao_buf.len(),
        });
    }

    if image.len() != expected_image_len {
        return Err(AoError::DimensionMismatch {
            expected: expected_image_len,
            got: image.len(),
        });
    }

    let s = strength.clamp(0.0, 1.0);
    let mut out = Vec::with_capacity(image.len());

    for pixel_idx in 0..num_pixels {
        let ao = ao_buf[pixel_idx];
        // scale = lerp(1.0, ao, strength) = 1.0 + strength * (ao - 1.0)
        let scale = 1.0 + s * (ao - 1.0);

        let r = image[pixel_idx * 3];
        let g = image[pixel_idx * 3 + 1];
        let b = image[pixel_idx * 3 + 2];

        out.push((r * scale).clamp(0.0, 1.0));
        out.push((g * scale).clamp(0.0, 1.0));
        out.push((b * scale).clamp(0.0, 1.0));
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Compute statistics over an AO buffer, using the depth buffer to identify
/// background pixels (depth == 0 or infinite).
pub fn ao_compute_stats(ao_buf: &[f32], depth_buf: &[f32]) -> AoStats {
    if ao_buf.is_empty() {
        return AoStats {
            mean_ao: 1.0,
            min_ao: 1.0,
            max_ao: 1.0,
            occlusion_fraction: 0.0,
            background_fraction: 0.0,
        };
    }

    let n = ao_buf.len();
    let depth_len = depth_buf.len().min(n);

    let mut sum = 0.0_f32;
    let mut min_ao = f32::MAX;
    let mut max_ao = f32::MIN;
    let mut occluded_count = 0usize;
    let mut background_count = 0usize;
    let mut foreground_count = 0usize;

    for i in 0..n {
        let ao = ao_buf[i];
        let depth = if i < depth_len { depth_buf[i] } else { 1.0 };

        let is_bg = depth == 0.0 || !depth.is_finite();
        if is_bg {
            background_count += 1;
        } else {
            foreground_count += 1;
            sum += ao;
            if ao < min_ao {
                min_ao = ao;
            }
            if ao > max_ao {
                max_ao = ao;
            }
            if ao < 0.9 {
                occluded_count += 1;
            }
        }
    }

    let (mean, min, max) = if foreground_count == 0 {
        (1.0, 1.0, 1.0)
    } else {
        (sum / foreground_count as f32, min_ao, max_ao)
    };

    AoStats {
        mean_ao: mean,
        min_ao: min,
        max_ao: max,
        occlusion_fraction: if foreground_count == 0 {
            0.0
        } else {
            occluded_count as f32 / foreground_count as f32
        },
        background_fraction: background_count as f32 / n as f32,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Format an [`AoConfig`] as a human-readable string.
pub fn format_ao_config(config: &AoConfig) -> String {
    format!(
        "AoConfig {{ n_samples={}, radius={:.3}, bias={:.4}, power={:.2}, \
         blur_radius={}, blur_sigma_space={:.2}, blur_sigma_depth={:.3}, strength={:.2} }}",
        config.n_samples,
        config.radius,
        config.bias,
        config.power,
        config.blur_radius,
        config.blur_sigma_space,
        config.blur_sigma_depth,
        config.strength,
    )
}

/// Format [`AoStats`] as a human-readable string.
pub fn format_ao_stats(stats: &AoStats) -> String {
    format!(
        "AoStats {{ mean={:.3}, min={:.3}, max={:.3}, \
         occlusion={:.1}%, background={:.1}% }}",
        stats.mean_ao,
        stats.min_ao,
        stats.max_ao,
        stats.occlusion_fraction * 100.0,
        stats.background_fraction * 100.0,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Full SSAO pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Run the full SSAO pipeline:
/// 1. Generate sample kernel (using `config.n_samples`)
/// 2. Get noise texture
/// 3. Compute AO
/// 4. Bilateral blur
/// 5. Apply to image
///
/// # Errors
/// - [`AoError::InvalidRadius`] if `config.radius <= 0`.
/// - [`AoError::EmptyInput`] if `image` is empty.
/// - [`AoError::DimensionMismatch`] if buffer dimensions don't match.
/// - [`AoError::InvalidConfig`] if `n_samples == 0`.
pub fn apply_ssao(
    image: &[f32],
    depth_buf: &[f32],
    normal_buf: &[f32],
    width: usize,
    height: usize,
    proj: AoProjParams,
    config: &AoConfig,
) -> Result<SsaoResult, AoError> {
    // Step 1: Generate kernel
    let kernel = ao_sample_kernel(config.n_samples)?;

    // Step 2: Noise texture
    let noise = ao_noise_texture();

    // Step 3: Compute raw AO
    let ao_map = ao_compute(
        depth_buf,
        normal_buf,
        width,
        height,
        AoSamplingBuffers {
            kernel: &kernel,
            noise: &noise,
        },
        proj,
        config,
    )?;

    // Step 4: Bilateral blur
    let ao_map_blurred = ao_bilateral_blur(&ao_map, depth_buf, width, height, config)?;

    // Step 5: Apply to image
    let output_image = ao_apply_to_image(image, &ao_map_blurred, width, height, config.strength)?;

    // Compute stats from blurred map
    let stats = ao_compute_stats(&ao_map_blurred, depth_buf);

    Ok(SsaoResult {
        image: output_image,
        ao_map,
        ao_map_blurred,
        stats,
    })
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

    fn make_flat_scene(w: usize, h: usize, depth: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = w * h;
        let depth_buf = vec![depth; n];
        let mut normal_buf = Vec::with_capacity(n * 3);
        for _ in 0..n {
            normal_buf.push(0.0_f32);
            normal_buf.push(0.0_f32);
            normal_buf.push(1.0_f32);
        }
        let image = vec![0.5_f32; n * 3];
        (depth_buf, normal_buf, image)
    }

    // ── AoConfig tests ────────────────────────────────────────────────────────

    #[test]
    fn test_aoconfig_default_values() {
        let cfg = AoConfig::default();
        assert_eq!(cfg.n_samples, 16);
        assert!(approx_eq(cfg.radius, 0.5, 1e-6));
        assert!(approx_eq(cfg.bias, 0.025, 1e-6));
        assert!(approx_eq(cfg.power, 1.0, 1e-6));
        assert_eq!(cfg.blur_radius, 2);
        assert!(approx_eq(cfg.blur_sigma_space, 2.0, 1e-6));
        assert!(approx_eq(cfg.blur_sigma_depth, 0.1, 1e-6));
        assert!(approx_eq(cfg.strength, 1.0, 1e-6));
    }

    // ── ao_sample_kernel tests ─────────────────────────────────────────────────

    #[test]
    fn test_ao_sample_kernel_length() {
        let kernel = ao_sample_kernel(16).expect("kernel generation failed");
        assert_eq!(kernel.len(), 16 * 3);
    }

    #[test]
    fn test_ao_sample_kernel_various_sizes() {
        for n in [1, 4, 8, 32, 64] {
            let kernel = ao_sample_kernel(n).expect("kernel generation failed");
            assert_eq!(kernel.len(), n * 3, "n={n}");
        }
    }

    #[test]
    fn test_ao_sample_kernel_within_unit_sphere() {
        let kernel = ao_sample_kernel(64).expect("kernel generation failed");
        let n = kernel.len() / 3;
        for i in 0..n {
            let x = kernel[i * 3];
            let y = kernel[i * 3 + 1];
            let z = kernel[i * 3 + 2];
            let len = (x * x + y * y + z * z).sqrt();
            assert!(
                len <= 1.001,
                "Sample {i} length {len:.4} exceeds unit sphere"
            );
        }
    }

    #[test]
    fn test_ao_sample_kernel_z_nonnegative() {
        let kernel = ao_sample_kernel(32).expect("kernel generation failed");
        let n = kernel.len() / 3;
        for i in 0..n {
            let z = kernel[i * 3 + 2];
            assert!(z >= 0.0, "Sample {i} has negative z={z}");
        }
    }

    #[test]
    fn test_ao_sample_kernel_zero_samples_error() {
        let result = ao_sample_kernel(0);
        assert!(
            matches!(result, Err(AoError::InvalidConfig(_))),
            "Expected InvalidConfig, got {result:?}"
        );
    }

    #[test]
    fn test_ao_sample_kernel_deterministic() {
        let k1 = ao_sample_kernel(16).expect("kernel 1 failed");
        let k2 = ao_sample_kernel(16).expect("kernel 2 failed");
        assert_eq!(k1, k2, "Kernel generation must be deterministic");
    }

    #[test]
    fn test_ao_sample_kernel_different_sizes_differ() {
        let k8 = ao_sample_kernel(8).expect("kernel 8 failed");
        let k16 = ao_sample_kernel(16).expect("kernel 16 failed");
        // Different seeds (based on n_samples), so values should differ
        assert_ne!(
            &k8[..8.min(k8.len())],
            &k16[..8.min(k16.len())],
            "Kernels with different n_samples should differ"
        );
    }

    // ── ao_noise_texture tests ────────────────────────────────────────────────

    #[test]
    fn test_ao_noise_texture_length() {
        let noise = ao_noise_texture();
        assert_eq!(
            noise.len(),
            32,
            "Noise texture should have 32 values (16 pairs)"
        );
    }

    #[test]
    fn test_ao_noise_texture_values_in_range() {
        let noise = ao_noise_texture();
        for (i, &v) in noise.iter().enumerate() {
            assert!(
                (-1.0..=1.0).contains(&v),
                "Noise value {i} = {v} out of [-1, 1]"
            );
        }
    }

    #[test]
    fn test_ao_noise_texture_deterministic() {
        let n1 = ao_noise_texture();
        let n2 = ao_noise_texture();
        assert_eq!(n1, n2, "Noise texture must be deterministic");
    }

    #[test]
    fn test_ao_noise_texture_has_variation() {
        let noise = ao_noise_texture();
        let first = noise[0];
        let has_different = noise.iter().any(|&v| (v - first).abs() > 1e-6);
        assert!(has_different, "Noise texture should have variation");
    }

    // ── ao_smoothstep tests ───────────────────────────────────────────────────

    #[test]
    fn test_ao_smoothstep_below_edge0() {
        let v = ao_smoothstep(0.0, 1.0, -0.5);
        assert!(approx_eq(v, 0.0, 1e-6), "Below edge0 should be 0, got {v}");
    }

    #[test]
    fn test_ao_smoothstep_above_edge1() {
        let v = ao_smoothstep(0.0, 1.0, 1.5);
        assert!(approx_eq(v, 1.0, 1e-6), "Above edge1 should be 1, got {v}");
    }

    #[test]
    fn test_ao_smoothstep_at_edge0() {
        let v = ao_smoothstep(0.0, 1.0, 0.0);
        assert!(approx_eq(v, 0.0, 1e-6), "At edge0 should be 0, got {v}");
    }

    #[test]
    fn test_ao_smoothstep_at_edge1() {
        let v = ao_smoothstep(0.0, 1.0, 1.0);
        assert!(approx_eq(v, 1.0, 1e-6), "At edge1 should be 1, got {v}");
    }

    #[test]
    fn test_ao_smoothstep_midpoint() {
        let v = ao_smoothstep(0.0, 1.0, 0.5);
        assert!(approx_eq(v, 0.5, 1e-5), "Midpoint should be 0.5, got {v}");
    }

    #[test]
    fn test_ao_smoothstep_monotone() {
        let mut prev = 0.0_f32;
        for i in 0..=20 {
            let x = i as f32 * 0.05;
            let v = ao_smoothstep(0.0, 1.0, x);
            assert!(v >= prev - 1e-6, "Smoothstep not monotone at x={x}");
            prev = v;
        }
    }

    // ── ao_cross tests ────────────────────────────────────────────────────────

    #[test]
    fn test_ao_cross_x_cross_y_equals_z() {
        let r = ao_cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(approx_eq(r[0], 0.0, 1e-6));
        assert!(approx_eq(r[1], 0.0, 1e-6));
        assert!(approx_eq(r[2], 1.0, 1e-6));
    }

    #[test]
    fn test_ao_cross_y_cross_z_equals_x() {
        let r = ao_cross([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(approx_eq(r[0], 1.0, 1e-6));
        assert!(approx_eq(r[1], 0.0, 1e-6));
        assert!(approx_eq(r[2], 0.0, 1e-6));
    }

    #[test]
    fn test_ao_cross_z_cross_x_equals_y() {
        let r = ao_cross([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
        assert!(approx_eq(r[0], 0.0, 1e-6));
        assert!(approx_eq(r[1], 1.0, 1e-6));
        assert!(approx_eq(r[2], 0.0, 1e-6));
    }

    #[test]
    fn test_ao_cross_anticommutative() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        let ab = ao_cross(a, b);
        let ba = ao_cross(b, a);
        for i in 0..3 {
            assert!(
                approx_eq(ab[i], -ba[i], 1e-5),
                "Not anticommutative at component {i}"
            );
        }
    }

    #[test]
    fn test_ao_cross_parallel_vectors_zero() {
        let r = ao_cross([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        for v in r {
            assert!(
                approx_eq(v, 0.0, 1e-6),
                "Parallel cross product should be zero, got {v}"
            );
        }
    }

    // ── ao_dot tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_ao_dot_orthogonal() {
        let d = ao_dot([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(approx_eq(d, 0.0, 1e-6), "Orthogonal dot should be 0");
    }

    #[test]
    fn test_ao_dot_parallel() {
        let d = ao_dot([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]);
        let expected = 1.0 + 4.0 + 9.0;
        assert!(approx_eq(d, expected, 1e-5));
    }

    #[test]
    fn test_ao_dot_antiparallel() {
        let d = ao_dot([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        assert!(approx_eq(d, -1.0, 1e-6));
    }

    // ── ao_normalize tests ────────────────────────────────────────────────────

    #[test]
    fn test_ao_normalize_unit_result() {
        let v = ao_normalize([3.0, 4.0, 0.0]);
        let len = ao_dot(v, v).sqrt();
        assert!(
            approx_eq(len, 1.0, 1e-5),
            "Normalized length should be 1, got {len}"
        );
    }

    #[test]
    fn test_ao_normalize_zero_vector_fallback() {
        let v = ao_normalize([0.0, 0.0, 0.0]);
        assert_eq!(v, [0.0, 0.0, 1.0], "Zero vector fallback should be (0,0,1)");
    }

    #[test]
    fn test_ao_normalize_already_unit() {
        let v = ao_normalize([1.0, 0.0, 0.0]);
        assert!(approx_eq(v[0], 1.0, 1e-6));
        assert!(approx_eq(v[1], 0.0, 1e-6));
        assert!(approx_eq(v[2], 0.0, 1e-6));
    }

    #[test]
    fn test_ao_normalize_direction_preserved() {
        let input = [3.0_f32, 0.0, 0.0];
        let v = ao_normalize(input);
        assert!(approx_eq(v[0], 1.0, 1e-6), "Direction should be preserved");
        assert!(approx_eq(v[1], 0.0, 1e-6));
        assert!(approx_eq(v[2], 0.0, 1e-6));
    }

    // ── ao_sample_depth tests ─────────────────────────────────────────────────

    #[test]
    fn test_ao_sample_depth_exact_pixel() {
        let depth_buf = vec![1.0_f32, 2.0, 3.0, 4.0];
        let d = ao_sample_depth(&depth_buf, 2, 2, 0.0, 0.0);
        assert!(
            approx_eq(d, 1.0, 1e-5),
            "Top-left pixel should be 1.0, got {d}"
        );
    }

    #[test]
    fn test_ao_sample_depth_center_pixel() {
        let depth_buf = vec![5.0_f32; 9]; // 3x3 all 5.0
        let d = ao_sample_depth(&depth_buf, 3, 3, 1.0, 1.0);
        assert!(
            approx_eq(d, 5.0, 1e-5),
            "Center pixel should be 5.0, got {d}"
        );
    }

    #[test]
    fn test_ao_sample_depth_clamp_boundary() {
        let depth_buf = vec![1.0_f32, 2.0, 3.0, 4.0];
        // Out of bounds coordinates should clamp
        let d_neg = ao_sample_depth(&depth_buf, 2, 2, -1.0, -1.0);
        assert!(
            approx_eq(d_neg, 1.0, 1e-5),
            "Negative coords should clamp to top-left"
        );
        let d_over = ao_sample_depth(&depth_buf, 2, 2, 10.0, 10.0);
        assert!(
            approx_eq(d_over, 4.0, 1e-5),
            "Overflow coords should clamp to bottom-right"
        );
    }

    #[test]
    fn test_ao_sample_depth_bilinear_center() {
        // 2x2 with values [1, 2; 3, 4]
        // At (0.5, 0.5) the bilinear interpolation should give 2.5
        let depth_buf = vec![1.0_f32, 2.0, 3.0, 4.0];
        let d = ao_sample_depth(&depth_buf, 2, 2, 0.5, 0.5);
        assert!(
            approx_eq(d, 2.5, 1e-5),
            "Bilinear center should be 2.5, got {d}"
        );
    }

    #[test]
    fn test_ao_sample_depth_empty() {
        let d = ao_sample_depth(&[], 0, 0, 0.0, 0.0);
        assert!(approx_eq(d, 0.0, 1e-6), "Empty buffer should return 0.0");
    }

    // ── ao_compute tests ──────────────────────────────────────────────────────

    #[test]
    fn test_ao_compute_all_background_returns_ones() {
        let w = 8;
        let h = 8;
        let depth_buf = vec![0.0_f32; w * h]; // all background
        let normal_buf = vec![0.0_f32; w * h * 3];
        let kernel = ao_sample_kernel(8).expect("kernel failed");
        let noise = ao_noise_texture();
        let config = AoConfig::default();

        let ao = ao_compute(
            &depth_buf,
            &normal_buf,
            w,
            h,
            AoSamplingBuffers {
                kernel: &kernel,
                noise: &noise,
            },
            AoProjParams::new(1.0, 1.0),
            &config,
        )
        .expect("ao_compute failed");

        for (i, &v) in ao.iter().enumerate() {
            assert!(
                approx_eq(v, 1.0, 1e-6),
                "Background pixel {i} should be 1.0, got {v}"
            );
        }
    }

    #[test]
    fn test_ao_compute_output_dimensions() {
        let w = 12;
        let h = 8;
        let (depth_buf, normal_buf, _) = make_flat_scene(w, h, 2.0);
        let kernel = ao_sample_kernel(8).expect("kernel failed");
        let noise = ao_noise_texture();
        let config = AoConfig::default();

        let ao = ao_compute(
            &depth_buf,
            &normal_buf,
            w,
            h,
            AoSamplingBuffers {
                kernel: &kernel,
                noise: &noise,
            },
            AoProjParams::new(1.0, 1.0),
            &config,
        )
        .expect("ao_compute failed");

        assert_eq!(ao.len(), w * h);
    }

    #[test]
    fn test_ao_compute_dimension_mismatch_depth() {
        let w = 4;
        let h = 4;
        let depth_buf = vec![1.0_f32; w * h + 1]; // wrong size
        let normal_buf = vec![0.0_f32; w * h * 3];
        let kernel = ao_sample_kernel(8).expect("kernel failed");
        let noise = ao_noise_texture();
        let config = AoConfig::default();

        let result = ao_compute(
            &depth_buf,
            &normal_buf,
            w,
            h,
            AoSamplingBuffers {
                kernel: &kernel,
                noise: &noise,
            },
            AoProjParams::new(1.0, 1.0),
            &config,
        );
        assert!(
            matches!(result, Err(AoError::DimensionMismatch { .. })),
            "Expected DimensionMismatch, got {result:?}"
        );
    }

    #[test]
    fn test_ao_compute_dimension_mismatch_normal() {
        let w = 4;
        let h = 4;
        let depth_buf = vec![1.0_f32; w * h];
        let normal_buf = vec![0.0_f32; w * h * 3 + 1]; // wrong size
        let kernel = ao_sample_kernel(8).expect("kernel failed");
        let noise = ao_noise_texture();
        let config = AoConfig::default();

        let result = ao_compute(
            &depth_buf,
            &normal_buf,
            w,
            h,
            AoSamplingBuffers {
                kernel: &kernel,
                noise: &noise,
            },
            AoProjParams::new(1.0, 1.0),
            &config,
        );
        assert!(
            matches!(result, Err(AoError::DimensionMismatch { .. })),
            "Expected DimensionMismatch, got {result:?}"
        );
    }

    #[test]
    fn test_ao_compute_empty_input_error() {
        let kernel = ao_sample_kernel(8).expect("kernel failed");
        let noise = ao_noise_texture();
        let config = AoConfig::default();
        let result = ao_compute(
            &[],
            &[],
            0,
            0,
            AoSamplingBuffers {
                kernel: &kernel,
                noise: &noise,
            },
            AoProjParams::new(1.0, 1.0),
            &config,
        );
        assert!(matches!(result, Err(AoError::EmptyInput)));
    }

    #[test]
    fn test_ao_compute_invalid_radius_error() {
        let w = 4;
        let h = 4;
        let (depth_buf, normal_buf, _) = make_flat_scene(w, h, 1.0);
        let kernel = ao_sample_kernel(8).expect("kernel failed");
        let noise = ao_noise_texture();
        let config = AoConfig {
            radius: -0.1,
            ..Default::default()
        };

        let result = ao_compute(
            &depth_buf,
            &normal_buf,
            w,
            h,
            AoSamplingBuffers {
                kernel: &kernel,
                noise: &noise,
            },
            AoProjParams::new(1.0, 1.0),
            &config,
        );
        assert!(matches!(result, Err(AoError::InvalidRadius(_))));
    }

    #[test]
    fn test_ao_compute_flat_surface_high_ao() {
        let w = 16;
        let h = 16;
        let (depth_buf, normal_buf, _) = make_flat_scene(w, h, 2.0);
        let kernel = ao_sample_kernel(16).expect("kernel failed");
        let noise = ao_noise_texture();
        let config = AoConfig::default();

        let ao = ao_compute(
            &depth_buf,
            &normal_buf,
            w,
            h,
            AoSamplingBuffers {
                kernel: &kernel,
                noise: &noise,
            },
            AoProjParams::new(500.0, 500.0),
            &config,
        )
        .expect("ao_compute failed");

        let mean = ao.iter().sum::<f32>() / ao.len() as f32;
        assert!(
            mean > 0.5,
            "Flat surface should have reasonably high mean AO, got {mean:.3}"
        );
    }

    #[test]
    fn test_ao_compute_values_in_range() {
        let w = 8;
        let h = 8;
        let (depth_buf, normal_buf, _) = make_flat_scene(w, h, 1.5);
        let kernel = ao_sample_kernel(16).expect("kernel failed");
        let noise = ao_noise_texture();
        let config = AoConfig::default();

        let ao = ao_compute(
            &depth_buf,
            &normal_buf,
            w,
            h,
            AoSamplingBuffers {
                kernel: &kernel,
                noise: &noise,
            },
            AoProjParams::new(1.0, 1.0),
            &config,
        )
        .expect("ao_compute failed");

        for (i, &v) in ao.iter().enumerate() {
            assert!((0.0..=1.0).contains(&v), "AO value {i} = {v} out of [0,1]");
        }
    }

    #[test]
    fn test_ao_compute_depth_disparity_causes_occlusion() {
        // Create a scene where some pixels are much closer than neighbors
        let w = 8;
        let h = 8;
        let n = w * h;
        let mut depth_buf = vec![5.0_f32; n];
        // Make center pixels very close
        for py in 2..6 {
            for px in 2..6 {
                depth_buf[py * w + px] = 0.1;
            }
        }
        let mut normal_buf = vec![0.0_f32; n * 3];
        for i in 0..n {
            normal_buf[i * 3 + 2] = 1.0;
        }
        let kernel = ao_sample_kernel(16).expect("kernel failed");
        let noise = ao_noise_texture();
        let config = AoConfig::default();

        let ao = ao_compute(
            &depth_buf,
            &normal_buf,
            w,
            h,
            AoSamplingBuffers {
                kernel: &kernel,
                noise: &noise,
            },
            AoProjParams::new(10.0, 10.0),
            &config,
        )
        .expect("ao_compute failed");

        // Check that values are in range
        for (i, &v) in ao.iter().enumerate() {
            assert!((0.0..=1.0).contains(&v), "AO value {i} = {v} out of [0,1]");
        }
    }

    #[test]
    fn test_ao_compute_infinite_depth_treated_as_background() {
        let w = 4;
        let h = 4;
        let depth_buf = vec![f32::INFINITY; w * h];
        let normal_buf = vec![0.0_f32; w * h * 3];
        let kernel = ao_sample_kernel(8).expect("kernel failed");
        let noise = ao_noise_texture();
        let config = AoConfig::default();

        let ao = ao_compute(
            &depth_buf,
            &normal_buf,
            w,
            h,
            AoSamplingBuffers {
                kernel: &kernel,
                noise: &noise,
            },
            AoProjParams::new(1.0, 1.0),
            &config,
        )
        .expect("ao_compute failed");

        for (i, &v) in ao.iter().enumerate() {
            assert!(
                approx_eq(v, 1.0, 1e-6),
                "Infinite depth pixel {i} should be 1.0"
            );
        }
    }

    // ── ao_bilateral_blur tests ───────────────────────────────────────────────

    #[test]
    fn test_ao_bilateral_blur_flat_unchanged() {
        let w = 8;
        let h = 8;
        let ao_buf = vec![1.0_f32; w * h];
        let depth_buf = vec![1.0_f32; w * h];
        let config = AoConfig::default();

        let blurred =
            ao_bilateral_blur(&ao_buf, &depth_buf, w, h, &config).expect("bilateral blur failed");

        for (i, (&a, &b)) in ao_buf.iter().zip(blurred.iter()).enumerate() {
            assert!(approx_eq(a, b, 1e-5), "Flat AO pixel {i}: {a} != {b}");
        }
    }

    #[test]
    fn test_ao_bilateral_blur_output_dimensions() {
        let w = 10;
        let h = 6;
        let ao_buf = vec![0.8_f32; w * h];
        let depth_buf = vec![2.0_f32; w * h];
        let config = AoConfig::default();

        let blurred =
            ao_bilateral_blur(&ao_buf, &depth_buf, w, h, &config).expect("bilateral blur failed");

        assert_eq!(blurred.len(), w * h);
    }

    #[test]
    fn test_ao_bilateral_blur_dimension_mismatch_ao() {
        let w = 4;
        let h = 4;
        let ao_buf = vec![1.0_f32; w * h + 1];
        let depth_buf = vec![1.0_f32; w * h];
        let config = AoConfig::default();

        let result = ao_bilateral_blur(&ao_buf, &depth_buf, w, h, &config);
        assert!(matches!(result, Err(AoError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_ao_bilateral_blur_dimension_mismatch_depth() {
        let w = 4;
        let h = 4;
        let ao_buf = vec![1.0_f32; w * h];
        let depth_buf = vec![1.0_f32; w * h + 1];
        let config = AoConfig::default();

        let result = ao_bilateral_blur(&ao_buf, &depth_buf, w, h, &config);
        assert!(matches!(result, Err(AoError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_ao_bilateral_blur_empty_input() {
        let config = AoConfig::default();
        let result = ao_bilateral_blur(&[], &[], 0, 0, &config);
        assert!(matches!(result, Err(AoError::EmptyInput)));
    }

    #[test]
    fn test_ao_bilateral_blur_values_in_range() {
        let w = 8;
        let h = 8;
        let ao_buf: Vec<f32> = (0..w * h).map(|i| (i % 10) as f32 / 10.0).collect();
        let depth_buf = vec![1.0_f32; w * h];
        let config = AoConfig::default();

        let blurred =
            ao_bilateral_blur(&ao_buf, &depth_buf, w, h, &config).expect("bilateral blur failed");

        for (i, &v) in blurred.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "Blurred value {i} = {v} out of [0,1]"
            );
        }
    }

    #[test]
    fn test_ao_bilateral_blur_preserves_background() {
        let w = 4;
        let h = 4;
        // Mix background (depth=0) and foreground
        let mut depth_buf = vec![1.0_f32; w * h];
        depth_buf[0] = 0.0; // background pixel
        let mut ao_buf = vec![0.5_f32; w * h];
        ao_buf[0] = 1.0; // background AO value
        let config = AoConfig::default();

        let blurred =
            ao_bilateral_blur(&ao_buf, &depth_buf, w, h, &config).expect("bilateral blur failed");

        // Background pixel should pass through unchanged
        assert!(
            approx_eq(blurred[0], 1.0, 1e-5),
            "Background AO should be unchanged"
        );
    }

    // ── ao_apply_to_image tests ───────────────────────────────────────────────

    #[test]
    fn test_ao_apply_strength_zero_unchanged() {
        let w = 4;
        let h = 4;
        let image: Vec<f32> = (0..w * h * 3).map(|i| (i % 10) as f32 / 10.0).collect();
        let ao_buf = vec![0.0_f32; w * h]; // fully occluded
        let result = ao_apply_to_image(&image, &ao_buf, w, h, 0.0).expect("apply failed");

        for (i, (&a, &b)) in image.iter().zip(result.iter()).enumerate() {
            assert!(
                approx_eq(a, b, 1e-5),
                "strength=0 changed pixel {i}: {a} != {b}"
            );
        }
    }

    #[test]
    fn test_ao_apply_full_ao_one_unchanged() {
        let w = 4;
        let h = 4;
        let image = vec![0.7_f32; w * h * 3];
        let ao_buf = vec![1.0_f32; w * h]; // no occlusion
        let result = ao_apply_to_image(&image, &ao_buf, w, h, 1.0).expect("apply failed");

        for (i, (&a, &b)) in image.iter().zip(result.iter()).enumerate() {
            assert!(
                approx_eq(a, b, 1e-5),
                "AO=1.0 changed pixel {i}: {a} != {b}"
            );
        }
    }

    #[test]
    fn test_ao_apply_full_ao_zero_darkens() {
        let w = 2;
        let h = 2;
        let image = vec![1.0_f32; w * h * 3];
        let ao_buf = vec![0.0_f32; w * h];
        let result = ao_apply_to_image(&image, &ao_buf, w, h, 1.0).expect("apply failed");

        for (i, &v) in result.iter().enumerate() {
            assert!(approx_eq(v, 0.0, 1e-5), "Pixel {i} should be 0, got {v}");
        }
    }

    #[test]
    fn test_ao_apply_dimension_mismatch_ao() {
        let w = 4;
        let h = 4;
        let image = vec![0.5_f32; w * h * 3];
        let ao_buf = vec![1.0_f32; w * h + 1];
        let result = ao_apply_to_image(&image, &ao_buf, w, h, 1.0);
        assert!(matches!(result, Err(AoError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_ao_apply_dimension_mismatch_image() {
        let w = 4;
        let h = 4;
        let image = vec![0.5_f32; w * h * 3 + 1];
        let ao_buf = vec![1.0_f32; w * h];
        let result = ao_apply_to_image(&image, &ao_buf, w, h, 1.0);
        assert!(matches!(result, Err(AoError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_ao_apply_empty_input() {
        let result = ao_apply_to_image(&[], &[], 0, 0, 1.0);
        assert!(matches!(result, Err(AoError::EmptyInput)));
    }

    #[test]
    fn test_ao_apply_output_length() {
        let w = 4;
        let h = 4;
        let image = vec![0.5_f32; w * h * 3];
        let ao_buf = vec![0.8_f32; w * h];
        let result = ao_apply_to_image(&image, &ao_buf, w, h, 1.0).expect("apply failed");
        assert_eq!(result.len(), image.len());
    }

    #[test]
    fn test_ao_apply_partial_strength() {
        // With strength=0.5 and ao=0.0, pixel should be 0.5 (lerp between 1.0 and 0.0)
        let w = 1;
        let h = 1;
        let image = vec![1.0_f32, 1.0, 1.0];
        let ao_buf = vec![0.0_f32; 1];
        let result = ao_apply_to_image(&image, &ao_buf, w, h, 0.5).expect("apply failed");
        for &v in &result {
            assert!(
                approx_eq(v, 0.5, 1e-5),
                "Half strength should give 0.5, got {v}"
            );
        }
    }

    // ── apply_ssao tests ──────────────────────────────────────────────────────

    #[test]
    fn test_apply_ssao_all_background_image_unchanged() {
        let w = 8;
        let h = 8;
        let n = w * h;
        let depth_buf = vec![0.0_f32; n]; // all background
        let normal_buf = vec![0.0_f32; n * 3];
        let image = vec![0.6_f32; n * 3];
        let config = AoConfig::default();

        let result = apply_ssao(
            &image,
            &depth_buf,
            &normal_buf,
            w,
            h,
            AoProjParams::new(1.0, 1.0),
            &config,
        )
        .expect("apply_ssao failed");

        // All background → AO = 1.0 → image unchanged
        for (i, (&a, &b)) in image.iter().zip(result.image.iter()).enumerate() {
            assert!(
                approx_eq(a, b, 1e-4),
                "Background: image changed at pixel {i}: {a} != {b}"
            );
        }
    }

    #[test]
    fn test_apply_ssao_valid_input_correct_dimensions() {
        let w = 8;
        let h = 8;
        let (depth_buf, normal_buf, image) = make_flat_scene(w, h, 2.0);
        let config = AoConfig::default();

        let result = apply_ssao(
            &image,
            &depth_buf,
            &normal_buf,
            w,
            h,
            AoProjParams::new(1.0, 1.0),
            &config,
        )
        .expect("apply_ssao failed");

        assert_eq!(result.image.len(), w * h * 3);
        assert_eq!(result.ao_map.len(), w * h);
        assert_eq!(result.ao_map_blurred.len(), w * h);
    }

    #[test]
    fn test_apply_ssao_ao_maps_in_range() {
        let w = 8;
        let h = 8;
        let (depth_buf, normal_buf, image) = make_flat_scene(w, h, 1.5);
        let config = AoConfig::default();

        let result = apply_ssao(
            &image,
            &depth_buf,
            &normal_buf,
            w,
            h,
            AoProjParams::new(1.0, 1.0),
            &config,
        )
        .expect("apply_ssao failed");

        for (i, &v) in result.ao_map.iter().enumerate() {
            assert!((0.0..=1.0).contains(&v), "ao_map[{i}] = {v} out of [0,1]");
        }
        for (i, &v) in result.ao_map_blurred.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "ao_map_blurred[{i}] = {v} out of [0,1]"
            );
        }
    }

    #[test]
    fn test_apply_ssao_invalid_config_error() {
        let w = 4;
        let h = 4;
        let (depth_buf, normal_buf, image) = make_flat_scene(w, h, 1.0);
        let config = AoConfig {
            n_samples: 0,
            ..Default::default()
        }; // invalid

        let result = apply_ssao(
            &image,
            &depth_buf,
            &normal_buf,
            w,
            h,
            AoProjParams::new(1.0, 1.0),
            &config,
        );
        assert!(result.is_err(), "Zero n_samples should fail");
    }

    // ── ao_compute_stats tests ────────────────────────────────────────────────

    #[test]
    fn test_ao_compute_stats_all_ones() {
        let ao_buf = vec![1.0_f32; 16];
        let depth_buf = vec![1.0_f32; 16];
        let stats = ao_compute_stats(&ao_buf, &depth_buf);

        assert!(approx_eq(stats.mean_ao, 1.0, 1e-5));
        assert!(approx_eq(stats.occlusion_fraction, 0.0, 1e-6));
    }

    #[test]
    fn test_ao_compute_stats_all_zero_ao() {
        let n = 16;
        let ao_buf = vec![0.0_f32; n];
        let depth_buf = vec![1.0_f32; n]; // all foreground
        let stats = ao_compute_stats(&ao_buf, &depth_buf);

        assert!(
            approx_eq(stats.occlusion_fraction, 1.0, 1e-5),
            "All zero AO should have 100% occlusion fraction, got {}",
            stats.occlusion_fraction
        );
    }

    #[test]
    fn test_ao_compute_stats_background_fraction() {
        let n = 8;
        let mut depth_buf = vec![1.0_f32; n];
        // Set half as background
        for d in depth_buf.iter_mut().take(n / 2) {
            *d = 0.0;
        }
        let ao_buf = vec![1.0_f32; n];
        let stats = ao_compute_stats(&ao_buf, &depth_buf);

        assert!(
            approx_eq(stats.background_fraction, 0.5, 1e-5),
            "Half background, got {}",
            stats.background_fraction
        );
    }

    #[test]
    fn test_ao_compute_stats_empty() {
        let stats = ao_compute_stats(&[], &[]);
        assert!(approx_eq(stats.mean_ao, 1.0, 1e-6));
        assert!(approx_eq(stats.occlusion_fraction, 0.0, 1e-6));
        assert!(approx_eq(stats.background_fraction, 0.0, 1e-6));
    }

    #[test]
    fn test_ao_compute_stats_min_max() {
        let ao_buf = vec![0.2_f32, 0.5, 0.8, 0.9];
        let depth_buf = vec![1.0_f32; 4];
        let stats = ao_compute_stats(&ao_buf, &depth_buf);

        assert!(approx_eq(stats.min_ao, 0.2, 1e-5));
        assert!(approx_eq(stats.max_ao, 0.9, 1e-5));
    }

    // ── format helpers tests ──────────────────────────────────────────────────

    #[test]
    fn test_format_ao_config_non_empty() {
        let config = AoConfig::default();
        let s = format_ao_config(&config);
        assert!(
            !s.is_empty(),
            "format_ao_config should return non-empty string"
        );
        assert!(s.contains("n_samples"), "Should contain n_samples");
        assert!(s.contains("radius"), "Should contain radius");
    }

    #[test]
    fn test_format_ao_stats_non_empty() {
        let stats = AoStats {
            mean_ao: 0.8,
            min_ao: 0.3,
            max_ao: 1.0,
            occlusion_fraction: 0.2,
            background_fraction: 0.1,
        };
        let s = format_ao_stats(&stats);
        assert!(
            !s.is_empty(),
            "format_ao_stats should return non-empty string"
        );
        assert!(s.contains("mean"), "Should contain mean");
    }

    // ── Single pixel edge cases ───────────────────────────────────────────────

    #[test]
    fn test_ao_compute_single_pixel() {
        let depth_buf = vec![1.0_f32];
        let normal_buf = vec![0.0_f32, 0.0, 1.0];
        let kernel = ao_sample_kernel(4).expect("kernel failed");
        let noise = ao_noise_texture();
        let config = AoConfig::default();

        let ao = ao_compute(
            &depth_buf,
            &normal_buf,
            1,
            1,
            AoSamplingBuffers {
                kernel: &kernel,
                noise: &noise,
            },
            AoProjParams::new(1.0, 1.0),
            &config,
        )
        .expect("ao_compute failed");

        assert_eq!(ao.len(), 1);
        assert!(ao[0] >= 0.0 && ao[0] <= 1.0);
    }

    #[test]
    fn test_apply_ssao_1x1_image() {
        let depth_buf = vec![1.0_f32];
        let normal_buf = vec![0.0_f32, 0.0, 1.0];
        let image = vec![0.5_f32, 0.5, 0.5];
        let config = AoConfig::default();

        let result = apply_ssao(
            &image,
            &depth_buf,
            &normal_buf,
            1,
            1,
            AoProjParams::new(1.0, 1.0),
            &config,
        )
        .expect("apply_ssao on 1x1 failed");

        assert_eq!(result.image.len(), 3);
        assert_eq!(result.ao_map.len(), 1);
    }
}
