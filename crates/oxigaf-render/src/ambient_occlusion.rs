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
//!
//! # Relationship to [`crate::ssao`]
//!
//! Two CPU ambient-occlusion modules coexist in this crate:
//!
//! | Module | Normal input | Projection | Denoise |
//! |---|---|---|---|
//! | [`crate::ambient_occlusion`] (this one) | flat `[f32]` triples | caller-supplied [`AoProjParams`] | depth-aware bilateral blur |
//! | [`crate::ssao`] | `Vec<[f32; 3]>` normal map | fixed pinhole unprojection | separable box blur |
//!
//! Prefer this module when you need a custom projection or an
//! edge-preserving denoise. [`ao_dot`] / [`ao_cross`] here are the crate's
//! single implementation of AO vector maths; [`crate::ssao`] aliases them.
//! The two modules' PRNGs are deliberately *not* shared — they differ in
//! zero-state handling and in the `u64 → f32` mapping, so their sample
//! patterns are not interchangeable.

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
/// These are the camera's **focal length expressed in pixels**, i.e.
/// `focal_px_x = width / (2 * tan(fov_x / 2))` and
/// `focal_px_y = height / (2 * tan(fov_y / 2))`. [`ao_compute`] uses them to
/// convert a view-space offset at a given depth into a pixel-space offset:
/// `pixel_offset = view_offset * focal_px / depth`. Passing `tan(fov/2)`
/// directly (as opposed to the focal length in pixels) under-scales the
/// sample offsets by roughly the image size and degenerates SSAO into
/// sampling the centre pixel.
#[derive(Debug, Clone, Copy)]
pub struct AoProjParams {
    /// Horizontal focal length in pixels.
    pub focal_px_x: f32,
    /// Vertical focal length in pixels.
    pub focal_px_y: f32,
}

impl AoProjParams {
    /// Construct from explicit focal-length-in-pixels values.
    pub fn new(focal_px_x: f32, focal_px_y: f32) -> Self {
        Self {
            focal_px_x,
            focal_px_y,
        }
    }

    /// Construct from horizontal/vertical field-of-view (radians) and image
    /// dimensions in pixels: `focal_px = dimension / (2 * tan(fov / 2))`.
    ///
    /// `width` and `height` are taken as `f32` so callers can pass either
    /// pixel counts or, if needed, sub-pixel-accurate render targets.
    pub fn from_fov(fov_x: f32, fov_y: f32, width: f32, height: f32) -> Self {
        let focal_px_x = width / (2.0 * (fov_x * 0.5).tan());
        let focal_px_y = height / (2.0 * (fov_y * 0.5).tan());
        Self {
            focal_px_x,
            focal_px_y,
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

    for i in 0..n_samples {
        // Cosine-weighted hemisphere direction (z >= 0): azimuth uniform in
        // [0, 2*PI), polar angle drawn so the *solid-angle* density is
        // proportional to cos(phi) rather than phi itself uniform. With
        // u ~ Uniform(0, 1), z = sqrt(1 - u) gives a cosine-weighted z and
        // sin(phi) = sqrt(u) is the matching planar radius — this
        // concentrates fewer samples at the pole (the normal direction) and
        // more toward grazing angles than a naive uniform-in-phi draw,
        // which is what SSAO needs since most occlusion happens at grazing
        // angles near the tangent plane.
        let theta = xorshift_f32(&mut state) * two_pi;
        let u = xorshift_f32(&mut state);
        let sin_phi = u.max(0.0).sqrt();
        let cos_phi = (1.0 - u).max(0.0).sqrt();

        let x = sin_phi * theta.cos();
        let y = sin_phi * theta.sin();
        let z = cos_phi;

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
/// # Depth and normal convention
/// Depth is **positive-linear and increases away from the camera** (the
/// camera sits at depth 0, background/no-hit is encoded as depth `0.0` or
/// non-finite). Normals are in the same view space, so a surface facing the
/// camera has a **negative** z component (it points back toward smaller
/// depth); `(0, 0, -1)` is "facing the camera" and `(0, 0, 1)` is "facing
/// away from the camera". The hemisphere kernel (z ≥ 0 in tangent space,
/// see [`ao_sample_kernel`]) is oriented along the normal, so
/// under this convention samples land at depths ≤ the surface depth, i.e.
/// toward the camera / free space, as intended for AO sampling.
///
/// # Parameters
/// - `depth_buf`: per-pixel linear depth (view-space z), length = `width * height`.
///   Pixels with depth == 0 or infinite are treated as background (AO = 1.0).
/// - `normal_buf`: per-pixel view-space normals (flat), length = `width * height * 3`.
/// - `width`, `height`: image dimensions.
/// - `kernel`: sample kernel from [`ao_sample_kernel`], length = `n_samples * 3`
///   (must be non-empty and a multiple of 3).
/// - `noise`: noise texture from [`ao_noise_texture`], length = 32 (must have
///   an even length ≥ 32).
/// - `proj`: [`AoProjParams`] — focal length in pixels, used to project
///   view-space sample offsets into screen space.
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
/// - [`AoError::InvalidConfig`] if `kernel` is empty / not a multiple of 3,
///   or `noise` has fewer than 32 values or an odd length.
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
    if kernel.is_empty() || !kernel.len().is_multiple_of(3) {
        return Err(AoError::InvalidConfig(format!(
            "kernel must be a non-empty multiple of 3 (x, y, z triplets), got {}",
            kernel.len()
        )));
    }
    if noise.len() < 32 || !noise.len().is_multiple_of(2) {
        return Err(AoError::InvalidConfig(format!(
            "noise texture must have an even length >= 32, got {}",
            noise.len()
        )));
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
                let sx = px as f32 + ox * proj.focal_px_x / denom;
                let sy = py as f32 + oy * proj.focal_px_y / denom;

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

                // Occlusion test: real geometry exists in front of (closer
                // to the camera than) the hemisphere sample point, i.e. the
                // sample point is embedded in solid matter.
                let occluded = if sampled_depth <= (depth + oz - bias) {
                    1.0_f32
                } else {
                    0.0_f32
                };

                occlusion_sum += occluded * range_check;
            }

            let n_samples_f = n_samples.max(1) as f32;
            let ao = (1.0 - occlusion_sum / n_samples_f).powf(power);
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
mod tests;
