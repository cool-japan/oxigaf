//! Albedo map utilities for FLAME head model.
//!
//! This module provides:
//! - [`AlbedoColor`]: An RGB color value in `[0, 1]` per channel.
//! - [`AlbedoTexture`]: A 2-D RGB texture with bilinear sampling.
//! - [`AlbedoConfig`]: Configuration for albedo evaluation (SH, ambient scale).
//! - [`AlbedoStats`]: Perceptual statistics across a collection of albedo colors.
//! - Utility functions for SH-to-RGB conversion, per-vertex baking, blending,
//!   normalization, ambient occlusion, and procedural texture generation.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when working with albedo maps.
#[derive(Debug, Error)]
pub enum AlbedoMapError {
    /// Configuration is invalid.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
    /// UV coordinate error.
    #[error("UV error: {0}")]
    UvError(String),
    /// Requested texture was not found.
    #[error("Texture not found: {0}")]
    TextureNotFound(String),
    /// Sampling failed.
    #[error("Sampling error: {0}")]
    SamplingError(String),
    /// Input slices have incompatible lengths.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
}

// ---------------------------------------------------------------------------
// AlbedoColor
// ---------------------------------------------------------------------------

/// A linear RGB color with channels in `[0, 1]` (un-clamped until explicitly
/// clamped with [`AlbedoColor::clamp`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlbedoColor {
    /// Red channel.
    pub r: f32,
    /// Green channel.
    pub g: f32,
    /// Blue channel.
    pub b: f32,
}

impl AlbedoColor {
    /// Create a new [`AlbedoColor`] from individual channel values.
    #[inline]
    #[must_use]
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Pure black: `(0, 0, 0)`.
    #[inline]
    #[must_use]
    pub fn black() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Pure white: `(1, 1, 1)`.
    #[inline]
    #[must_use]
    pub fn white() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    /// Uniform gray with all channels set to `v`.
    #[inline]
    #[must_use]
    pub fn gray(v: f32) -> Self {
        Self::new(v, v, v)
    }

    /// Linearly interpolate between `self` and `other`.
    ///
    /// `t = 0` returns `self`; `t = 1` returns `other`.
    #[inline]
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        Self::new(
            self.r + (other.r - self.r) * t,
            self.g + (other.g - self.g) * t,
            self.b + (other.b - self.b) * t,
        )
    }

    /// Clamp each channel to `[0, 1]`.
    #[inline]
    #[must_use]
    pub fn clamp(&self) -> Self {
        Self::new(
            self.r.clamp(0.0, 1.0),
            self.g.clamp(0.0, 1.0),
            self.b.clamp(0.0, 1.0),
        )
    }

    /// Convert to a 3-element array `[r, g, b]`.
    #[inline]
    #[must_use]
    pub fn to_array(&self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }

    /// Create from a 3-element array `[r, g, b]`.
    #[inline]
    #[must_use]
    pub fn from_array(arr: [f32; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }
}

// ---------------------------------------------------------------------------
// AlbedoTexture
// ---------------------------------------------------------------------------

/// A 2-D RGB texture stored as f32 values in `[0, 1]`.
///
/// Pixels are stored in row-major order, RGB interleaved:
/// `data[(y * width + x) * 3 + c]` is channel `c` of pixel `(x, y)`,
/// where `y=0` is the top row.
#[derive(Debug, Clone)]
pub struct AlbedoTexture {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// RGB interleaved pixel data; length is `width * height * 3`.
    pub data: Vec<f32>,
}

impl AlbedoTexture {
    /// Create a new texture of `width × height` pixels filled with `fill`.
    #[must_use]
    pub fn new(width: usize, height: usize, fill: AlbedoColor) -> Self {
        let mut data = Vec::with_capacity(width * height * 3);
        for _ in 0..(width * height) {
            data.push(fill.r);
            data.push(fill.g);
            data.push(fill.b);
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Create a texture from existing pixel data.
    ///
    /// # Errors
    ///
    /// Returns [`AlbedoMapError::DimensionMismatch`] if `data.len() != width * height * 3`.
    pub fn from_data(width: usize, height: usize, data: Vec<f32>) -> Result<Self, AlbedoMapError> {
        let expected = width * height * 3;
        if data.len() != expected {
            return Err(AlbedoMapError::DimensionMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Get the color of pixel `(x, y)`.
    ///
    /// Returns `None` if `x >= self.width` or `y >= self.height`.
    #[must_use]
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<AlbedoColor> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = (y * self.width + x) * 3;
        Some(AlbedoColor::new(
            self.data[idx],
            self.data[idx + 1],
            self.data[idx + 2],
        ))
    }

    /// Sample the texture using bilinear interpolation with clamp-to-edge wrapping.
    ///
    /// `u` and `v` are in `[0, 1]`; out-of-range values are clamped to the edge.
    #[must_use]
    pub fn sample_bilinear(&self, u: f32, v: f32) -> AlbedoColor {
        // Clamp UV to [0, 1]
        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);

        // Map to continuous pixel coordinates [0, width-1] × [0, height-1]
        let px = u * (self.width.saturating_sub(1)) as f32;
        let py = v * (self.height.saturating_sub(1)) as f32;

        let x0 = px.floor() as usize;
        let y0 = py.floor() as usize;

        // Clamp to valid pixel range
        let x0 = x0.min(self.width.saturating_sub(1));
        let y0 = y0.min(self.height.saturating_sub(1));
        let x1 = (x0 + 1).min(self.width.saturating_sub(1));
        let y1 = (y0 + 1).min(self.height.saturating_sub(1));

        let tx = px - px.floor();
        let ty = py - py.floor();

        // Fetch four corners
        let c00 = self.get_pixel_unchecked(x0, y0);
        let c10 = self.get_pixel_unchecked(x1, y0);
        let c01 = self.get_pixel_unchecked(x0, y1);
        let c11 = self.get_pixel_unchecked(x1, y1);

        // Bilinear blend: along x first, then y
        let top = c00.lerp(&c10, tx);
        let bot = c01.lerp(&c11, tx);
        top.lerp(&bot, ty)
    }

    /// Unchecked pixel fetch (caller guarantees bounds).
    #[inline]
    fn get_pixel_unchecked(&self, x: usize, y: usize) -> AlbedoColor {
        let idx = (y * self.width + x) * 3;
        AlbedoColor::new(self.data[idx], self.data[idx + 1], self.data[idx + 2])
    }

    /// Convert the texture to 8-bit RGB bytes (values scaled from `[0,1]` to `0..=255`).
    #[must_use]
    pub fn to_rgb8(&self) -> Vec<u8> {
        self.data
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// AlbedoConfig
// ---------------------------------------------------------------------------

/// Configuration for albedo evaluation.
#[derive(Debug, Clone)]
pub struct AlbedoConfig {
    /// Fallback albedo color when no texture or SH is available.
    pub default_albedo: AlbedoColor,
    /// Whether to use spherical harmonics approximation.
    pub use_sh_approximation: bool,
    /// Number of SH bands to evaluate (1, 2, or 3).
    pub sh_bands: usize,
    /// Overall brightness scale applied to evaluated colors.
    pub ambient_scale: f32,
}

impl Default for AlbedoConfig {
    fn default() -> Self {
        Self {
            default_albedo: AlbedoColor::gray(0.7),
            use_sh_approximation: false,
            sh_bands: 1,
            ambient_scale: 1.0,
        }
    }
}

impl AlbedoConfig {
    /// Validate this configuration.
    ///
    /// # Errors
    ///
    /// Returns [`AlbedoMapError::InvalidConfig`] if `sh_bands` is 0 or greater than 3.
    pub fn validate(&self) -> Result<(), AlbedoMapError> {
        if self.sh_bands == 0 || self.sh_bands > 3 {
            return Err(AlbedoMapError::InvalidConfig(format!(
                "sh_bands must be 1, 2, or 3; got {}",
                self.sh_bands
            )));
        }
        if !self.ambient_scale.is_finite() {
            return Err(AlbedoMapError::InvalidConfig(format!(
                "ambient_scale must be finite; got {}",
                self.ambient_scale
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AlbedoStats
// ---------------------------------------------------------------------------

/// Perceptual statistics over a collection of albedo colors.
#[derive(Debug, Clone)]
pub struct AlbedoStats {
    /// Mean perceptual luminance across all colors.
    pub mean_brightness: f32,
    /// Variance of perceptual luminance across all colors.
    pub variance_brightness: f32,
    /// Minimum perceptual luminance.
    pub min_brightness: f32,
    /// Maximum perceptual luminance.
    pub max_brightness: f32,
    /// Mean red channel value.
    pub mean_r: f32,
    /// Mean green channel value.
    pub mean_g: f32,
    /// Mean blue channel value.
    pub mean_b: f32,
}

// ---------------------------------------------------------------------------
// Spherical harmonics constants
// ---------------------------------------------------------------------------

// Band-0 (l=0)
const SH_C0: f32 = 0.282_094_79; // 1/(2√π)

// Band-1 (l=1)
const SH_C1: f32 = 0.488_602_51; // √(3/(4π))

// Band-2 (l=2)
const SH_C2_0: f32 = 0.315_391_56; // √(5/(16π))    for m=0: 3z²-1 term
const SH_C2_1: f32 = 1.092_548_5; // √(15/(4π))    for m=±1
const SH_C2_2: f32 = 0.546_274_22; // √(15/(16π))   for m=±2

// Band-3 (l=3)
const SH_C3_0: f32 = 0.373_176_33; // √(7/(16π))    for m=0
const SH_C3_1: f32 = 1.445_305_7; // √(21/(4π))/2  for m=±1
const SH_C3_2: f32 = 0.590_043_6; // √(105/(4π))/2 for m=±2  (corrected)
const SH_C3_3: f32 = 0.590_043_6; // √(35/(16π))·√3 for m=±3 (same constant)

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

/// Evaluate real spherical harmonics of up to `bands` bands at a unit direction.
///
/// The coefficient array is RGB-interleaved: element `i*3 + c` is the i-th SH
/// coefficient for channel `c` (0=R, 1=G, 2=B).
///
/// For `bands` bands, the number of SH basis functions is `(bands + 1)²`, so
/// the total number of floats in `sh_coeffs` must be `(bands + 1)² * 3`.
///
/// # Errors
///
/// Returns [`AlbedoMapError::InvalidConfig`] if `bands` is 0 or > 3, or if
/// `sh_coeffs` has an incorrect length.
pub fn sh_to_rgb(
    sh_coeffs: &[f32],
    direction: [f32; 3],
    bands: usize,
) -> Result<AlbedoColor, AlbedoMapError> {
    if bands == 0 || bands > 3 {
        return Err(AlbedoMapError::InvalidConfig(format!(
            "bands must be 1, 2, or 3; got {bands}"
        )));
    }
    let num_basis = (bands + 1) * (bands + 1);
    let expected = num_basis * 3;
    if sh_coeffs.len() != expected {
        return Err(AlbedoMapError::DimensionMismatch {
            expected,
            actual: sh_coeffs.len(),
        });
    }

    let [dir_x, dir_y, dir_z] = direction;

    // We accumulate red, green, blue channels separately.
    // Coefficient layout: sh_coeffs[basis_idx * 3 + channel]
    let coeff = |basis: usize, channel: usize| -> f32 { sh_coeffs[basis * 3 + channel] };

    // l=0, m=0 (index 0)
    let mut red = SH_C0 * coeff(0, 0);
    let mut green = SH_C0 * coeff(0, 1);
    let mut blue = SH_C0 * coeff(0, 2);

    if bands >= 1 {
        // l=1: indices 1 (m=-1, dir_y), 2 (m=0, dir_z), 3 (m=1, dir_x)
        red += SH_C1 * (coeff(1, 0) * dir_y + coeff(2, 0) * dir_z + coeff(3, 0) * dir_x);
        green += SH_C1 * (coeff(1, 1) * dir_y + coeff(2, 1) * dir_z + coeff(3, 1) * dir_x);
        blue += SH_C1 * (coeff(1, 2) * dir_y + coeff(2, 2) * dir_z + coeff(3, 2) * dir_x);
    }

    if bands >= 2 {
        // l=2: indices 4..8
        let xy = dir_x * dir_y;
        let yz = dir_y * dir_z;
        let xz = dir_x * dir_z;
        let x2 = dir_x * dir_x;
        let y2 = dir_y * dir_y;
        let z2 = dir_z * dir_z;
        let b2_0 = SH_C2_1 * xy;
        let b2_1 = SH_C2_1 * yz;
        let b2_2 = SH_C2_0 * (2.0 * z2 - x2 - y2);
        let b2_3 = SH_C2_1 * xz;
        let b2_4 = SH_C2_2 * (x2 - y2);

        red += coeff(4, 0) * b2_0
            + coeff(5, 0) * b2_1
            + coeff(6, 0) * b2_2
            + coeff(7, 0) * b2_3
            + coeff(8, 0) * b2_4;
        green += coeff(4, 1) * b2_0
            + coeff(5, 1) * b2_1
            + coeff(6, 1) * b2_2
            + coeff(7, 1) * b2_3
            + coeff(8, 1) * b2_4;
        blue += coeff(4, 2) * b2_0
            + coeff(5, 2) * b2_1
            + coeff(6, 2) * b2_2
            + coeff(7, 2) * b2_3
            + coeff(8, 2) * b2_4;
    }

    if bands >= 3 {
        // l=3: indices 9..15
        let x2 = dir_x * dir_x;
        let y2 = dir_y * dir_y;
        let z2 = dir_z * dir_z;
        let b3_0 = SH_C3_3 * dir_y * (3.0 * x2 - y2);
        let b3_1 = SH_C3_2 * dir_x * dir_y * dir_z;
        let b3_2 = SH_C3_1 * dir_y * (4.0 * z2 - x2 - y2);
        let b3_3 = SH_C3_0 * dir_z * (2.0 * z2 - 3.0 * x2 - 3.0 * y2);
        let b3_4 = SH_C3_1 * dir_x * (4.0 * z2 - x2 - y2);
        let b3_5 = SH_C3_2 * (x2 - y2) * dir_z;
        let b3_6 = SH_C3_3 * dir_x * (x2 - 3.0 * y2);

        red += coeff(9, 0) * b3_0
            + coeff(10, 0) * b3_1
            + coeff(11, 0) * b3_2
            + coeff(12, 0) * b3_3
            + coeff(13, 0) * b3_4
            + coeff(14, 0) * b3_5
            + coeff(15, 0) * b3_6;
        green += coeff(9, 1) * b3_0
            + coeff(10, 1) * b3_1
            + coeff(11, 1) * b3_2
            + coeff(12, 1) * b3_3
            + coeff(13, 1) * b3_4
            + coeff(14, 1) * b3_5
            + coeff(15, 1) * b3_6;
        blue += coeff(9, 2) * b3_0
            + coeff(10, 2) * b3_1
            + coeff(11, 2) * b3_2
            + coeff(12, 2) * b3_3
            + coeff(13, 2) * b3_4
            + coeff(14, 2) * b3_5
            + coeff(15, 2) * b3_6;
    }

    Ok(AlbedoColor::new(red, green, blue))
}

/// Sample a texture at each UV coordinate in `vertices_uv` using bilinear interpolation.
///
/// UV coordinates are expected to be in `[0, 1]²`; out-of-range values are clamped.
#[must_use]
pub fn per_vertex_albedo_from_texture(
    vertices_uv: &[(f32, f32)],
    texture: &AlbedoTexture,
) -> Vec<AlbedoColor> {
    vertices_uv
        .iter()
        .map(|&(u, v)| texture.sample_bilinear(u, v))
        .collect()
}

/// Bake per-vertex albedo colors by projecting a texture through face UV data.
///
/// For each face corner `(vertex_idx, (u, v))` in `face_uvs`, the corresponding
/// vertex accumulates the sampled texture color.  Vertices not covered by any
/// face corner receive `config.default_albedo`.
///
/// # Errors
///
/// Returns [`AlbedoMapError::UvError`] if `vertex_idx >= num_vertices`.
pub fn bake_vertex_albedo(
    num_vertices: usize,
    face_uvs: &[(usize, (f32, f32))],
    texture: &AlbedoTexture,
    config: &AlbedoConfig,
) -> Result<Vec<AlbedoColor>, AlbedoMapError> {
    let mut accum_r = vec![0.0f32; num_vertices];
    let mut accum_g = vec![0.0f32; num_vertices];
    let mut accum_b = vec![0.0f32; num_vertices];
    let mut counts = vec![0u32; num_vertices];

    for &(vertex_idx, (u, v)) in face_uvs {
        if vertex_idx >= num_vertices {
            return Err(AlbedoMapError::UvError(format!(
                "vertex_idx {vertex_idx} out of range for num_vertices {num_vertices}"
            )));
        }
        let color = texture.sample_bilinear(u, v);
        accum_r[vertex_idx] += color.r;
        accum_g[vertex_idx] += color.g;
        accum_b[vertex_idx] += color.b;
        counts[vertex_idx] += 1;
    }

    let result = (0..num_vertices)
        .map(|i| {
            if counts[i] == 0 {
                config.default_albedo
            } else {
                let n = counts[i] as f32;
                AlbedoColor::new(accum_r[i] / n, accum_g[i] / n, accum_b[i] / n)
            }
        })
        .collect();

    Ok(result)
}

/// Flatten an albedo color slice to a raw float array for GPU upload.
///
/// The layout is `[r0, g0, b0, r1, g1, b1, ...]`.
#[must_use]
pub fn albedo_to_vertex_colors(albedos: &[AlbedoColor]) -> Vec<f32> {
    let mut out = Vec::with_capacity(albedos.len() * 3);
    for c in albedos {
        out.push(c.r);
        out.push(c.g);
        out.push(c.b);
    }
    out
}

/// Element-wise linear interpolation between two albedo maps.
///
/// `t = 0` returns `a`; `t = 1` returns `b`.
///
/// # Errors
///
/// Returns [`AlbedoMapError::DimensionMismatch`] if `a.len() != b.len()`.
pub fn blend_albedos(
    a: &[AlbedoColor],
    b: &[AlbedoColor],
    t: f32,
) -> Result<Vec<AlbedoColor>, AlbedoMapError> {
    if a.len() != b.len() {
        return Err(AlbedoMapError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(ca, cb)| ca.lerp(cb, t))
        .collect())
}

/// Compute the perceptual luminance of a color using ITU-R BT.709 coefficients.
///
/// `luminance = 0.2126·R + 0.7152·G + 0.0722·B`
#[inline]
#[must_use]
pub fn albedo_brightness(color: &AlbedoColor) -> f32 {
    0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b
}

/// Rescale all albedo colors so that the maximum perceptual brightness is 1.0.
///
/// If all colors are black (max brightness = 0), the slice is left unchanged.
pub fn normalize_albedo(albedos: &mut [AlbedoColor]) {
    let max_brightness = albedos.iter().map(albedo_brightness).fold(0.0f32, f32::max);

    if max_brightness <= 0.0 {
        return;
    }

    let scale = 1.0 / max_brightness;
    for c in albedos.iter_mut() {
        c.r *= scale;
        c.g *= scale;
        c.b *= scale;
    }
}

/// Compute perceptual statistics over a slice of albedo colors.
///
/// Returns a zeroed [`AlbedoStats`] when the slice is empty.
#[must_use]
pub fn compute_albedo_stats(albedos: &[AlbedoColor]) -> AlbedoStats {
    if albedos.is_empty() {
        return AlbedoStats {
            mean_brightness: 0.0,
            variance_brightness: 0.0,
            min_brightness: 0.0,
            max_brightness: 0.0,
            mean_r: 0.0,
            mean_g: 0.0,
            mean_b: 0.0,
        };
    }

    let n = albedos.len() as f32;
    let mut sum_r = 0.0f32;
    let mut sum_g = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut sum_lum = 0.0f32;
    let mut min_lum = f32::MAX;
    let mut max_lum = f32::MIN;

    for c in albedos {
        sum_r += c.r;
        sum_g += c.g;
        sum_b += c.b;
        let lum = albedo_brightness(c);
        sum_lum += lum;
        if lum < min_lum {
            min_lum = lum;
        }
        if lum > max_lum {
            max_lum = lum;
        }
    }

    let mean_lum = sum_lum / n;

    // Compute variance (population, not sample)
    let variance = albedos
        .iter()
        .map(|c| {
            let d = albedo_brightness(c) - mean_lum;
            d * d
        })
        .sum::<f32>()
        / n;

    AlbedoStats {
        mean_brightness: mean_lum,
        variance_brightness: variance,
        min_brightness: min_lum,
        max_brightness: max_lum,
        mean_r: sum_r / n,
        mean_g: sum_g / n,
        mean_b: sum_b / n,
    }
}

/// Apply ambient occlusion by multiplying each albedo color by its per-vertex AO factor.
///
/// `ao_factors` values should be in `[0, 1]` where 0 means fully occluded (black)
/// and 1 means fully lit.
///
/// # Errors
///
/// Returns [`AlbedoMapError::DimensionMismatch`] if `albedos.len() != ao_factors.len()`.
pub fn apply_ambient_occlusion(
    albedos: &[AlbedoColor],
    ao_factors: &[f32],
) -> Result<Vec<AlbedoColor>, AlbedoMapError> {
    if albedos.len() != ao_factors.len() {
        return Err(AlbedoMapError::DimensionMismatch {
            expected: albedos.len(),
            actual: ao_factors.len(),
        });
    }
    Ok(albedos
        .iter()
        .zip(ao_factors.iter())
        .map(|(c, &ao)| AlbedoColor::new(c.r * ao, c.g * ao, c.b * ao))
        .collect())
}

/// Generate a procedural checker-pattern texture.
///
/// The pattern tiles at frequency `freq` (number of checker squares per axis).
/// Even tiles use `color_a`; odd tiles use `color_b`.
#[must_use]
pub fn checker_texture(
    width: usize,
    height: usize,
    freq: usize,
    color_a: AlbedoColor,
    color_b: AlbedoColor,
) -> AlbedoTexture {
    let safe_freq = freq.max(1);
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let tile_x = x * safe_freq / width.max(1);
            let tile_y = y * safe_freq / height.max(1);
            let color = if (tile_x + tile_y).is_multiple_of(2) {
                color_a
            } else {
                color_b
            };
            data.push(color.r);
            data.push(color.g);
            data.push(color.b);
        }
    }
    // SAFETY: data length is exactly width * height * 3 by construction.
    AlbedoTexture {
        width,
        height,
        data,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // AlbedoColor
    // -----------------------------------------------------------------------

    #[test]
    fn test_albedo_color_new() {
        let c = AlbedoColor::new(0.1, 0.2, 0.3);
        assert!((c.r - 0.1).abs() < 1e-6);
        assert!((c.g - 0.2).abs() < 1e-6);
        assert!((c.b - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_albedo_color_black_white_gray() {
        let black = AlbedoColor::black();
        assert!(black.r.abs() < 1e-9);
        assert!(black.g.abs() < 1e-9);
        assert!(black.b.abs() < 1e-9);

        let white = AlbedoColor::white();
        assert!((white.r - 1.0).abs() < 1e-9);
        assert!((white.g - 1.0).abs() < 1e-9);
        assert!((white.b - 1.0).abs() < 1e-9);

        let gray = AlbedoColor::gray(0.5);
        assert!((gray.r - 0.5).abs() < 1e-6);
        assert!((gray.g - 0.5).abs() < 1e-6);
        assert!((gray.b - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_albedo_color_lerp_t0() {
        let a = AlbedoColor::new(0.0, 0.0, 0.0);
        let b = AlbedoColor::new(1.0, 1.0, 1.0);
        let result = a.lerp(&b, 0.0);
        assert!((result.r - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_albedo_color_lerp_t1() {
        let a = AlbedoColor::new(0.0, 0.0, 0.0);
        let b = AlbedoColor::new(1.0, 1.0, 1.0);
        let result = a.lerp(&b, 1.0);
        assert!((result.r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_albedo_color_lerp_half() {
        let a = AlbedoColor::new(0.0, 0.2, 0.4);
        let b = AlbedoColor::new(1.0, 0.8, 0.6);
        let result = a.lerp(&b, 0.5);
        assert!((result.r - 0.5).abs() < 1e-6);
        assert!((result.g - 0.5).abs() < 1e-6);
        assert!((result.b - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_albedo_color_clamp() {
        let c = AlbedoColor::new(-0.5, 0.5, 1.5);
        let clamped = c.clamp();
        assert!(clamped.r.abs() < 1e-9);
        assert!((clamped.g - 0.5).abs() < 1e-6);
        assert!((clamped.b - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_albedo_color_to_from_array() {
        let arr = [0.25, 0.5, 0.75];
        let c = AlbedoColor::from_array(arr);
        let out = c.to_array();
        assert!((out[0] - 0.25).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        assert!((out[2] - 0.75).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // AlbedoTexture
    // -----------------------------------------------------------------------

    #[test]
    fn test_albedo_texture_new_fill() {
        let fill = AlbedoColor::new(0.1, 0.2, 0.3);
        let tex = AlbedoTexture::new(4, 3, fill);
        assert_eq!(tex.width, 4);
        assert_eq!(tex.height, 3);
        assert_eq!(tex.data.len(), 4 * 3 * 3);
        let px = tex.get_pixel(2, 1).expect("pixel exists");
        assert!((px.r - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_albedo_texture_from_data_valid() {
        let data = vec![0.5f32; 2 * 2 * 3];
        let tex = AlbedoTexture::from_data(2, 2, data).expect("valid data");
        assert_eq!(tex.width, 2);
    }

    #[test]
    fn test_albedo_texture_from_data_invalid() {
        let data = vec![0.5f32; 5]; // wrong length
        let err = AlbedoTexture::from_data(2, 2, data).expect_err("should fail");
        assert!(matches!(err, AlbedoMapError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_albedo_texture_get_pixel_bounds() {
        let tex = AlbedoTexture::new(4, 4, AlbedoColor::white());
        assert!(tex.get_pixel(4, 0).is_none()); // x out of bounds
        assert!(tex.get_pixel(0, 4).is_none()); // y out of bounds
        assert!(tex.get_pixel(3, 3).is_some()); // last valid pixel
    }

    #[test]
    fn test_albedo_texture_bilinear_center() {
        // Uniform texture: bilinear sampling should return the fill color.
        let fill = AlbedoColor::new(0.4, 0.5, 0.6);
        let tex = AlbedoTexture::new(8, 8, fill);
        let sampled = tex.sample_bilinear(0.5, 0.5);
        assert!((sampled.r - 0.4).abs() < 1e-5);
        assert!((sampled.g - 0.5).abs() < 1e-5);
        assert!((sampled.b - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_albedo_texture_bilinear_clamp_outside() {
        // Values outside [0,1] should clamp to edge.
        let tex = AlbedoTexture::new(4, 4, AlbedoColor::gray(0.3));
        let a = tex.sample_bilinear(-0.5, 0.5);
        let b = tex.sample_bilinear(1.5, 0.5);
        assert!((a.r - 0.3).abs() < 1e-5);
        assert!((b.r - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_albedo_texture_to_rgb8() {
        let tex = AlbedoTexture::new(1, 1, AlbedoColor::white());
        let bytes = tex.to_rgb8();
        assert_eq!(bytes, vec![255u8, 255, 255]);
    }

    #[test]
    fn test_albedo_texture_to_rgb8_black() {
        let tex = AlbedoTexture::new(1, 1, AlbedoColor::black());
        let bytes = tex.to_rgb8();
        assert_eq!(bytes, vec![0u8, 0, 0]);
    }

    #[test]
    fn test_albedo_texture_bilinear_two_pixel() {
        // 2-pixel wide texture: left=black, right=white. Sample at u=0.5 should be mid-gray.
        let data = vec![
            0.0, 0.0, 0.0, // pixel (0,0): black
            1.0, 1.0, 1.0, // pixel (1,0): white
        ];
        let tex = AlbedoTexture::from_data(2, 1, data).expect("valid");
        let mid = tex.sample_bilinear(0.5, 0.0);
        assert!((mid.r - 0.5).abs() < 1e-5, "mid.r = {}", mid.r);
    }

    // -----------------------------------------------------------------------
    // AlbedoConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_albedo_config_validate_valid() {
        let cfg = AlbedoConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_albedo_config_validate_invalid_bands_zero() {
        let cfg = AlbedoConfig {
            sh_bands: 0,
            ..Default::default()
        };
        let err = cfg.validate().expect_err("sh_bands=0 is invalid");
        assert!(matches!(err, AlbedoMapError::InvalidConfig(_)));
    }

    #[test]
    fn test_albedo_config_validate_invalid_bands_four() {
        let cfg = AlbedoConfig {
            sh_bands: 4,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // per_vertex_albedo_from_texture
    // -----------------------------------------------------------------------

    #[test]
    fn test_per_vertex_albedo_basic() {
        let tex = AlbedoTexture::new(4, 4, AlbedoColor::gray(0.7));
        let uvs = vec![(0.0f32, 0.0f32), (0.5, 0.5), (1.0, 1.0)];
        let result = per_vertex_albedo_from_texture(&uvs, &tex);
        assert_eq!(result.len(), 3);
        for c in &result {
            assert!((c.r - 0.7).abs() < 1e-5);
        }
    }

    #[test]
    fn test_per_vertex_albedo_clamping() {
        let tex = AlbedoTexture::new(4, 4, AlbedoColor::new(0.2, 0.4, 0.6));
        let uvs = vec![(-1.0f32, -1.0f32), (2.0, 2.0)];
        let result = per_vertex_albedo_from_texture(&uvs, &tex);
        assert_eq!(result.len(), 2);
        // All should clamp to the fill color
        for c in &result {
            assert!((c.r - 0.2).abs() < 1e-5);
        }
    }

    #[test]
    fn test_per_vertex_albedo_empty() {
        let tex = AlbedoTexture::new(4, 4, AlbedoColor::white());
        let result = per_vertex_albedo_from_texture(&[], &tex);
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // bake_vertex_albedo
    // -----------------------------------------------------------------------

    #[test]
    fn test_bake_vertex_albedo_no_faces() {
        let tex = AlbedoTexture::new(4, 4, AlbedoColor::white());
        let cfg = AlbedoConfig::default();
        let result = bake_vertex_albedo(3, &[], &tex, &cfg).expect("ok");
        assert_eq!(result.len(), 3);
        // All vertices get default
        for c in &result {
            assert!((c.r - cfg.default_albedo.r).abs() < 1e-5);
        }
    }

    #[test]
    fn test_bake_vertex_albedo_single_face() {
        let fill = AlbedoColor::new(0.3, 0.5, 0.7);
        let tex = AlbedoTexture::new(4, 4, fill);
        let cfg = AlbedoConfig::default();
        let face_uvs = vec![(0usize, (0.5f32, 0.5f32))];
        let result = bake_vertex_albedo(2, &face_uvs, &tex, &cfg).expect("ok");
        assert_eq!(result.len(), 2);
        assert!((result[0].r - 0.3).abs() < 1e-5);
        // Vertex 1 has no UV, gets default
        assert!((result[1].r - cfg.default_albedo.r).abs() < 1e-5);
    }

    #[test]
    fn test_bake_vertex_albedo_overlapping_uvs_averaged() {
        // Two UV references to vertex 0 from different UV coords — should average.
        let data = vec![
            0.0, 0.0, 0.0, // pixel (0,0): black
            1.0, 1.0, 1.0, // pixel (1,0): white
        ];
        let tex = AlbedoTexture::from_data(2, 1, data).expect("valid");
        let cfg = AlbedoConfig::default();
        // Vertex 0 at u=0 → black, at u=1 → white → average = 0.5
        let face_uvs = vec![(0usize, (0.0f32, 0.0f32)), (0usize, (1.0f32, 0.0f32))];
        let result = bake_vertex_albedo(1, &face_uvs, &tex, &cfg).expect("ok");
        assert!((result[0].r - 0.5).abs() < 1e-5, "r={}", result[0].r);
    }

    #[test]
    fn test_bake_vertex_albedo_out_of_range_vertex() {
        let tex = AlbedoTexture::new(4, 4, AlbedoColor::white());
        let cfg = AlbedoConfig::default();
        let face_uvs = vec![(5usize, (0.5f32, 0.5f32))]; // vertex 5 out of range for num_vertices=3
        let err = bake_vertex_albedo(3, &face_uvs, &tex, &cfg).expect_err("should fail");
        assert!(matches!(err, AlbedoMapError::UvError(_)));
    }

    // -----------------------------------------------------------------------
    // blend_albedos
    // -----------------------------------------------------------------------

    #[test]
    fn test_blend_albedos_t0() {
        let a = vec![AlbedoColor::new(0.2, 0.3, 0.4)];
        let b = vec![AlbedoColor::new(0.8, 0.7, 0.6)];
        let result = blend_albedos(&a, &b, 0.0).expect("ok");
        assert!((result[0].r - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_blend_albedos_t1() {
        let a = vec![AlbedoColor::new(0.2, 0.3, 0.4)];
        let b = vec![AlbedoColor::new(0.8, 0.7, 0.6)];
        let result = blend_albedos(&a, &b, 1.0).expect("ok");
        assert!((result[0].r - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_blend_albedos_half() {
        let a = vec![AlbedoColor::new(0.0, 0.0, 0.0)];
        let b = vec![AlbedoColor::new(1.0, 1.0, 1.0)];
        let result = blend_albedos(&a, &b, 0.5).expect("ok");
        assert!((result[0].r - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_blend_albedos_length_mismatch() {
        let a = vec![AlbedoColor::black()];
        let b = vec![AlbedoColor::black(), AlbedoColor::white()];
        let err = blend_albedos(&a, &b, 0.5).expect_err("should fail");
        assert!(matches!(err, AlbedoMapError::DimensionMismatch { .. }));
    }

    // -----------------------------------------------------------------------
    // normalize_albedo
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalize_albedo_empty() {
        let mut albedos: Vec<AlbedoColor> = vec![];
        normalize_albedo(&mut albedos); // should not panic
        assert!(albedos.is_empty());
    }

    #[test]
    fn test_normalize_albedo_uniform() {
        let mut albedos = vec![AlbedoColor::gray(0.5); 4];
        normalize_albedo(&mut albedos);
        // max brightness was 0.5, now all r=g=b=1.0
        for c in &albedos {
            assert!((c.r - 1.0).abs() < 1e-5, "r={}", c.r);
        }
    }

    #[test]
    fn test_normalize_albedo_all_black() {
        let mut albedos = vec![AlbedoColor::black(); 4];
        normalize_albedo(&mut albedos);
        // max brightness = 0, unchanged
        for c in &albedos {
            assert!(c.r.abs() < 1e-9);
        }
    }

    #[test]
    fn test_normalize_albedo_varied() {
        let mut albedos = vec![AlbedoColor::gray(0.2), AlbedoColor::gray(0.8)];
        normalize_albedo(&mut albedos);
        // brightest (0.8) should become exactly 1.0
        assert!((albedos[1].r - 1.0).abs() < 1e-5);
        // darker should be 0.2/0.8 = 0.25
        assert!((albedos[0].r - 0.25).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // compute_albedo_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_albedo_stats_empty() {
        let stats = compute_albedo_stats(&[]);
        assert!(stats.mean_brightness.abs() < 1e-9);
        assert!(stats.variance_brightness.abs() < 1e-9);
    }

    #[test]
    fn test_compute_albedo_stats_single() {
        let c = AlbedoColor::new(1.0, 0.0, 0.0); // pure red
        let stats = compute_albedo_stats(&[c]);
        let expected_lum = 0.2126;
        assert!((stats.mean_brightness - expected_lum).abs() < 1e-4);
        assert!((stats.variance_brightness).abs() < 1e-8);
        assert!((stats.mean_r - 1.0).abs() < 1e-6);
        assert!((stats.mean_g).abs() < 1e-6);
    }

    #[test]
    fn test_compute_albedo_stats_multiple() {
        let albedos = vec![AlbedoColor::black(), AlbedoColor::white()];
        let stats = compute_albedo_stats(&albedos);
        // mean brightness should be halfway between 0 and 1
        assert!((stats.mean_brightness - 0.5).abs() < 1e-5);
        assert!((stats.min_brightness).abs() < 1e-6);
        assert!((stats.max_brightness - 1.0).abs() < 1e-6);
        assert!((stats.mean_r - 0.5).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // apply_ambient_occlusion
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_ao_full() {
        let albedos = vec![AlbedoColor::white()];
        let ao = vec![1.0f32];
        let result = apply_ambient_occlusion(&albedos, &ao).expect("ok");
        assert!((result[0].r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_ao_zero() {
        let albedos = vec![AlbedoColor::white()];
        let ao = vec![0.0f32];
        let result = apply_ambient_occlusion(&albedos, &ao).expect("ok");
        assert!((result[0].r).abs() < 1e-6);
    }

    #[test]
    fn test_apply_ao_length_mismatch() {
        let albedos = vec![AlbedoColor::white(), AlbedoColor::black()];
        let ao = vec![1.0f32]; // wrong length
        let err = apply_ambient_occlusion(&albedos, &ao).expect_err("should fail");
        assert!(matches!(err, AlbedoMapError::DimensionMismatch { .. }));
    }

    // -----------------------------------------------------------------------
    // sh_to_rgb
    // -----------------------------------------------------------------------

    #[test]
    fn test_sh_to_rgb_band1_dc_only() {
        // All SH coeffs zero except the DC term (index 0, R channel).
        // Result should be SH_C0 * coeff(0, channel).
        let mut coeffs = vec![0.0f32; 4 * 3]; // bands=1 → 4 bases × 3 channels
        coeffs[0] = 1.0; // R: DC coeff = 1
        coeffs[1] = 0.5; // G: DC coeff = 0.5
        coeffs[2] = 0.25; // B: DC coeff = 0.25
        let dir = [0.0f32, 0.0, 1.0]; // arbitrary unit direction
        let color = sh_to_rgb(&coeffs, dir, 1).expect("ok");
        let expected_r = SH_C0 * 1.0;
        let expected_g = SH_C0 * 0.5;
        let expected_b = SH_C0 * 0.25;
        assert!((color.r - expected_r).abs() < 1e-6, "r={}", color.r);
        assert!((color.g - expected_g).abs() < 1e-6, "g={}", color.g);
        assert!((color.b - expected_b).abs() < 1e-6, "b={}", color.b);
    }

    #[test]
    fn test_sh_to_rgb_band2() {
        let coeffs = vec![0.0f32; 9 * 3]; // bands=2 → 9 bases × 3 channels
        let dir = [1.0f32, 0.0, 0.0];
        let color = sh_to_rgb(&coeffs, dir, 2).expect("ok");
        // All-zero coefficients → black
        assert!(color.r.abs() < 1e-6);
    }

    #[test]
    fn test_sh_to_rgb_band3() {
        let coeffs = vec![0.1f32; 16 * 3]; // bands=3 → 16 bases × 3 channels
        let dir = [0.0f32, 0.0, 1.0]; // +Z axis
        let color = sh_to_rgb(&coeffs, dir, 3).expect("ok");
        // Just check it doesn't error and returns a finite color
        assert!(color.r.is_finite());
        assert!(color.g.is_finite());
        assert!(color.b.is_finite());
    }

    #[test]
    fn test_sh_to_rgb_invalid_bands_zero() {
        let coeffs = vec![0.0f32; 4 * 3];
        let err = sh_to_rgb(&coeffs, [0.0, 0.0, 1.0], 0).expect_err("bands=0 invalid");
        assert!(matches!(err, AlbedoMapError::InvalidConfig(_)));
    }

    #[test]
    fn test_sh_to_rgb_invalid_bands_four() {
        let coeffs = vec![0.0f32; 25 * 3];
        let err = sh_to_rgb(&coeffs, [0.0, 0.0, 1.0], 4).expect_err("bands=4 invalid");
        assert!(matches!(err, AlbedoMapError::InvalidConfig(_)));
    }

    #[test]
    fn test_sh_to_rgb_wrong_length() {
        let coeffs = vec![0.0f32; 5]; // wrong
        let err = sh_to_rgb(&coeffs, [0.0, 0.0, 1.0], 1).expect_err("wrong length");
        assert!(matches!(err, AlbedoMapError::DimensionMismatch { .. }));
    }

    // -----------------------------------------------------------------------
    // albedo_brightness
    // -----------------------------------------------------------------------

    #[test]
    fn test_albedo_brightness_pure_red() {
        let c = AlbedoColor::new(1.0, 0.0, 0.0);
        assert!((albedo_brightness(&c) - 0.2126).abs() < 1e-4);
    }

    #[test]
    fn test_albedo_brightness_pure_green() {
        let c = AlbedoColor::new(0.0, 1.0, 0.0);
        assert!((albedo_brightness(&c) - 0.7152).abs() < 1e-4);
    }

    #[test]
    fn test_albedo_brightness_gray() {
        let c = AlbedoColor::gray(0.5);
        // (0.2126 + 0.7152 + 0.0722) * 0.5 = 0.5
        assert!((albedo_brightness(&c) - 0.5).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // checker_texture
    // -----------------------------------------------------------------------

    #[test]
    fn test_checker_texture_pixel_values() {
        let black = AlbedoColor::black();
        let white = AlbedoColor::white();
        let tex = checker_texture(4, 4, 2, black, white);
        // Tile (0,0) → color_a = black
        let px00 = tex.get_pixel(0, 0).expect("pixel");
        assert!((px00.r).abs() < 1e-6);
        // Tile (1,0) → color_b = white
        let px20 = tex.get_pixel(2, 0).expect("pixel");
        assert!((px20.r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_checker_texture_dimensions() {
        let tex = checker_texture(8, 6, 4, AlbedoColor::black(), AlbedoColor::white());
        assert_eq!(tex.width, 8);
        assert_eq!(tex.height, 6);
        assert_eq!(tex.data.len(), 8 * 6 * 3);
    }
}
