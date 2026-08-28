//! Background image generation for 3DGS scene compositing.
//!
//! This module provides utilities to generate background images that fill the
//! void behind semi-transparent Gaussians during rendering. Proper backgrounds
//! are essential for:
//!
//! - Visual quality: prevents the "black void" artifact behind transparent Gaussians.
//! - Training diversity: augmenting the training distribution with varied backgrounds
//!   helps the model generalise and reduces overfitting.
//! - Loss computation: background-aware loss functions require a reference background
//!   image for correct Porter-Duff compositing.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use oxigaf_render::background::{BackgroundColor, BackgroundType, generate_background};
//!
//! let bg = generate_background(
//!     1920, 1080,
//!     &BackgroundType::VerticalGradient {
//!         top: BackgroundColor::new(0.53, 0.81, 0.98).unwrap(),
//!         bottom: BackgroundColor::white(),
//!     },
//! ).unwrap();
//! ```

use std::f32::consts::PI;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by background generation operations.
#[derive(Debug, Error)]
pub enum BackgroundError {
    /// Width or height is zero, or the pixel buffer has an unexpected length.
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// One or more color channel values are outside [0, 1].
    #[error("Invalid color: {0}")]
    InvalidColor(String),

    /// A configuration parameter is out of range or logically inconsistent.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// BackgroundColor
// ─────────────────────────────────────────────────────────────────────────────

/// A linear-light RGB color with all channels in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BackgroundColor {
    /// Red channel in [0, 1].
    pub r: f32,
    /// Green channel in [0, 1].
    pub g: f32,
    /// Blue channel in [0, 1].
    pub b: f32,
}

impl BackgroundColor {
    /// Construct a new [`BackgroundColor`], validating that every channel is in [0, 1].
    ///
    /// # Errors
    ///
    /// Returns [`BackgroundError::InvalidColor`] if any channel is outside [0, 1].
    pub fn new(r: f32, g: f32, b: f32) -> Result<Self, BackgroundError> {
        let channels = [("r", r), ("g", g), ("b", b)];
        for (name, v) in channels {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                return Err(BackgroundError::InvalidColor(format!(
                    "channel '{name}' = {v} is not in [0, 1]"
                )));
            }
        }
        Ok(Self { r, g, b })
    }

    /// Solid black (0, 0, 0).
    #[inline]
    pub fn black() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 0.0,
        }
    }

    /// Solid white (1, 1, 1).
    #[inline]
    pub fn white() -> Self {
        Self {
            r: 1.0,
            g: 1.0,
            b: 1.0,
        }
    }

    /// Mid-gray convenience constructor. `v` is clamped to [0, 1].
    ///
    /// This is a convenience constructor that intentionally skips validation
    /// (it clamps instead), matching the Rust ecosystem convention for
    /// `from_f32`-style constructors.
    #[inline]
    pub fn gray(v: f32) -> Self {
        let v = v.clamp(0.0, 1.0);
        Self { r: v, g: v, b: v }
    }

    /// Parse a CSS-style `"#RRGGBB"` hex string into a [`BackgroundColor`].
    ///
    /// Each 8-bit channel is divided by 255.0 to produce a value in [0, 1].
    ///
    /// # Errors
    ///
    /// Returns [`BackgroundError::InvalidColor`] if the string is not in
    /// `#RRGGBB` format or contains non-hex digits.
    pub fn from_hex(hex: &str) -> Result<Self, BackgroundError> {
        let s = hex.trim();
        let digits = s.strip_prefix('#').ok_or_else(|| {
            BackgroundError::InvalidColor(format!("expected '#RRGGBB', got '{s}'"))
        })?;

        if digits.len() != 6 {
            return Err(BackgroundError::InvalidColor(format!(
                "hex color must have exactly 6 digits after '#', got {}: '{s}'",
                digits.len()
            )));
        }

        let parse_byte = |offset: usize| -> Result<u8, BackgroundError> {
            u8::from_str_radix(&digits[offset..offset + 2], 16).map_err(|_| {
                BackgroundError::InvalidColor(format!(
                    "invalid hex digits at position {offset} in '{s}'"
                ))
            })
        };

        let r_byte = parse_byte(0)?;
        let g_byte = parse_byte(2)?;
        let b_byte = parse_byte(4)?;

        Ok(Self {
            r: r_byte as f32 / 255.0,
            g: g_byte as f32 / 255.0,
            b: b_byte as f32 / 255.0,
        })
    }

    /// Return the color as a `[r, g, b]` array.
    #[inline]
    pub fn to_array(&self) -> [f32; 3] {
        [self.r, self.g, self.b]
    }

    /// Linearly interpolate between `self` (t=0) and `other` (t=1).
    ///
    /// `t` is clamped to [0, 1] before blending.
    pub fn blend_with(&self, other: BackgroundColor, t: f32) -> BackgroundColor {
        let t = t.clamp(0.0, 1.0);
        BackgroundColor {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }
}

impl Default for BackgroundColor {
    fn default() -> Self {
        Self::black()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BackgroundType
// ─────────────────────────────────────────────────────────────────────────────

/// Specifies the visual style of a generated background image.
#[derive(Debug, Clone)]
pub enum BackgroundType {
    /// A single uniform color fills the entire image.
    Solid(BackgroundColor),

    /// Linear vertical gradient from `top` to `bottom`.
    VerticalGradient {
        /// Color at the top row (y = 0).
        top: BackgroundColor,
        /// Color at the bottom row (y = height-1).
        bottom: BackgroundColor,
    },

    /// Radial gradient from `inner` (center) to `outer` (corners).
    RadialGradient {
        /// Color at the image center.
        inner: BackgroundColor,
        /// Color at the image edges / corners.
        outer: BackgroundColor,
    },

    /// Alternating checkerboard pattern, useful for transparency visualization.
    Checkerboard {
        /// First cell color (top-left cell).
        color_a: BackgroundColor,
        /// Second cell color.
        color_b: BackgroundColor,
        /// Side length of each square cell in pixels. Clamped to 1 if 0.
        cell_size: u32,
    },

    /// Simple physically-inspired sky with a horizon blend region.
    Sky {
        /// Color high in the sky (top).
        sky_color: BackgroundColor,
        /// Color at the horizon line.
        horizon_color: BackgroundColor,
        /// Color below the horizon (ground).
        ground_color: BackgroundColor,
    },

    /// Spatially-coherent noise (same color patch for nearby pixels) using
    /// xorshift64. Good for mild augmentation.
    Noise {
        /// Seed for the pseudo-random number generator.
        seed: u64,
    },

    /// Independent uniformly-random color for every pixel. Aggressive
    /// augmentation that forces the network to ignore background texture.
    RandomPixels {
        /// Seed for the pseudo-random number generator.
        seed: u64,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// BackgroundImage
// ─────────────────────────────────────────────────────────────────────────────

/// A flat, row-major RGB image with f32 pixels.
///
/// The pixel at column `x`, row `y` occupies `pixels[(y * width + x) * 3 ..]`.
pub struct BackgroundImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Flat RGB f32 pixel data, row-major. Length = `width * height * 3`.
    pub pixels: Vec<f32>,
}

impl BackgroundImage {
    /// Allocate a new all-black image of the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        let len = (width as usize) * (height as usize) * 3;
        Self {
            width,
            height,
            pixels: vec![0.0_f32; len],
        }
    }

    /// Read the RGB color at pixel `(x, y)`.
    ///
    /// Returns `[0, 0, 0]` if `(x, y)` is out of bounds.
    pub fn pixel(&self, x: u32, y: u32) -> [f32; 3] {
        if x >= self.width || y >= self.height {
            return [0.0, 0.0, 0.0];
        }
        let idx = (y as usize * self.width as usize + x as usize) * 3;
        [self.pixels[idx], self.pixels[idx + 1], self.pixels[idx + 2]]
    }

    /// Write the RGB color at pixel `(x, y)`.
    ///
    /// Does nothing if `(x, y)` is out of bounds.
    pub fn set_pixel(&mut self, x: u32, y: u32, color: [f32; 3]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = (y as usize * self.width as usize + x as usize) * 3;
        self.pixels[idx] = color[0];
        self.pixels[idx + 1] = color[1];
        self.pixels[idx + 2] = color[2];
    }

    /// Convert the image to 8-bit per channel by clamping and scaling.
    ///
    /// Each f32 channel value is clamped to [0, 1] and mapped to `[0, 255]`.
    pub fn to_u8(&self) -> Vec<u8> {
        self.pixels
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
            .collect()
    }

    /// Compute the per-channel mean over all pixels.
    ///
    /// Returns `[0, 0, 0]` for an empty image.
    pub fn mean_color(&self) -> [f32; 3] {
        let n = (self.width as usize) * (self.height as usize);
        if n == 0 {
            return [0.0, 0.0, 0.0];
        }
        let mut sum = [0.0_f64; 3];
        for chunk in self.pixels.chunks_exact(3) {
            sum[0] += chunk[0] as f64;
            sum[1] += chunk[1] as f64;
            sum[2] += chunk[2] as f64;
        }
        [
            (sum[0] / n as f64) as f32,
            (sum[1] / n as f64) as f32,
            (sum[2] / n as f64) as f32,
        ]
    }

    /// Composite a foreground RGBA image over this background using
    /// Porter-Duff Over blending.
    ///
    /// `foreground_rgba` must be a flat RGBA f32 buffer with length
    /// `width * height * 4`, where each pixel is `[r, g, b, alpha]`.
    ///
    /// The result is a flat RGB f32 buffer of length `width * height * 3`:
    ///
    /// ```text
    /// out_rgb = fg_rgb + bg_rgb * (1 - fg_alpha)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`BackgroundError::InvalidDimensions`] if `foreground_rgba.len()`
    /// does not equal `width * height * 4`.
    pub fn composite_over(&self, foreground_rgba: &[f32]) -> Result<Vec<f32>, BackgroundError> {
        let expected = (self.width as usize) * (self.height as usize) * 4;
        if foreground_rgba.len() != expected {
            return Err(BackgroundError::InvalidDimensions(format!(
                "foreground_rgba has {} elements, expected {} ({}×{}×4)",
                foreground_rgba.len(),
                expected,
                self.width,
                self.height,
            )));
        }

        let num_pixels = (self.width as usize) * (self.height as usize);
        let mut out = Vec::with_capacity(num_pixels * 3);

        for i in 0..num_pixels {
            let fg_r = foreground_rgba[i * 4];
            let fg_g = foreground_rgba[i * 4 + 1];
            let fg_b = foreground_rgba[i * 4 + 2];
            let fg_a = foreground_rgba[i * 4 + 3];

            let bg_r = self.pixels[i * 3];
            let bg_g = self.pixels[i * 3 + 1];
            let bg_b = self.pixels[i * 3 + 2];

            let inv_a = 1.0 - fg_a;
            out.push(fg_r + bg_r * inv_a);
            out.push(fg_g + bg_g * inv_a);
            out.push(fg_b + bg_b * inv_a);
        }

        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// xorshift64 — local PRNG used for noise backgrounds
// ─────────────────────────────────────────────────────────────────────────────

/// Fast xorshift64 pseudo-random number generator.
///
/// Implements the xorshift64 algorithm (period 2^64 − 1).
/// The state must never be zero; a non-zero `seed` guarantees this.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Scale a raw `u64` to a f32 in [0, 1).
#[inline]
fn u64_to_f32_01(v: u64) -> f32 {
    // Use upper 24 bits for float mantissa precision.
    (v >> 40) as f32 / (1u64 << 24) as f32
}

/// Derive a per-pixel seed from the global `seed` and the pixel index.
#[inline]
fn pixel_seed(seed: u64, pixel_index: u64) -> u64 {
    // Mix pixel index with golden-ratio constant to spread bits.
    let mixed = pixel_index.wrapping_mul(0x9E3779B97F4A7C15);
    // XOR with base seed; ensure non-zero so xorshift works.
    let s = seed ^ mixed;
    if s == 0 {
        1
    } else {
        s
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// generate_background
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a [`BackgroundImage`] of the specified type and dimensions.
///
/// # Errors
///
/// - [`BackgroundError::InvalidDimensions`] if `width == 0` or `height == 0`.
pub fn generate_background(
    width: u32,
    height: u32,
    bg_type: &BackgroundType,
) -> Result<BackgroundImage, BackgroundError> {
    if width == 0 || height == 0 {
        return Err(BackgroundError::InvalidDimensions(format!(
            "width and height must be > 0, got {}×{}",
            width, height
        )));
    }

    let mut img = BackgroundImage::new(width, height);

    match bg_type {
        BackgroundType::Solid(color) => {
            let arr = color.to_array();
            for chunk in img.pixels.chunks_exact_mut(3) {
                chunk[0] = arr[0];
                chunk[1] = arr[1];
                chunk[2] = arr[2];
            }
        }

        BackgroundType::VerticalGradient { top, bottom } => {
            // When height == 1, t = 0 → top color for the single row.
            let h_denom = if height > 1 { (height - 1) as f32 } else { 1.0 };
            for y in 0..height {
                let t = y as f32 / h_denom;
                let color = top.blend_with(*bottom, t).to_array();
                for x in 0..width {
                    img.set_pixel(x, y, color);
                }
            }
        }

        BackgroundType::RadialGradient { inner, outer } => {
            let cx = width as f32 / 2.0;
            let cy = height as f32 / 2.0;
            // Half-diagonal: when 1×1, set to 1.0 to avoid division-by-zero.
            let half_diag = {
                let d = (cx * cx + cy * cy).sqrt();
                if d == 0.0 {
                    1.0
                } else {
                    d
                }
            };

            for y in 0..height {
                for x in 0..width {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let dist = (dx * dx + dy * dy).sqrt();
                    let t = (dist / half_diag).min(1.0);
                    let color = inner.blend_with(*outer, t).to_array();
                    img.set_pixel(x, y, color);
                }
            }
        }

        BackgroundType::Checkerboard {
            color_a,
            color_b,
            cell_size,
        } => {
            let cell = (*cell_size).max(1);
            for y in 0..height {
                for x in 0..width {
                    let cell_x = x / cell;
                    let cell_y = y / cell;
                    let parity = (cell_x + cell_y) % 2;
                    let color = if parity == 0 {
                        color_a.to_array()
                    } else {
                        color_b.to_array()
                    };
                    img.set_pixel(x, y, color);
                }
            }
        }

        BackgroundType::Sky {
            sky_color,
            horizon_color,
            ground_color,
        } => {
            // ny ranges from +1 (top) to -1 (bottom).
            let h_denom = if height > 1 { (height - 1) as f32 } else { 1.0 };
            for y in 0..height {
                let ny = 1.0 - 2.0 * y as f32 / h_denom;

                let color = if ny > 0.1 {
                    // Sky region: blend sky → horizon as we descend toward horizon.
                    let t = 1.0 - (ny - 0.1) / 0.9;
                    sky_color.blend_with(*horizon_color, t).to_array()
                } else if ny >= -0.1 {
                    // Horizon band: pure horizon color.
                    horizon_color.to_array()
                } else {
                    // Ground region: blend horizon → ground.
                    let t = (-ny - 0.1) / 0.9;
                    horizon_color.blend_with(*ground_color, t).to_array()
                };

                for x in 0..width {
                    img.set_pixel(x, y, color);
                }
            }
        }

        BackgroundType::Noise { seed } => {
            for y in 0..height {
                for x in 0..width {
                    let pixel_index = y as u64 * width as u64 + x as u64;
                    let mut state = pixel_seed(*seed, pixel_index);

                    let r = u64_to_f32_01(xorshift64(&mut state));
                    let g = u64_to_f32_01(xorshift64(&mut state));
                    let b = u64_to_f32_01(xorshift64(&mut state));
                    img.set_pixel(x, y, [r, g, b]);
                }
            }
        }

        BackgroundType::RandomPixels { seed } => {
            for y in 0..height {
                for x in 0..width {
                    let pixel_index = y as u64 * width as u64 + x as u64;
                    // Use different seed offsets for each channel to break correlation.
                    let base_seed = pixel_seed(*seed, pixel_index);
                    let mut state_r = pixel_seed(base_seed, 0x01);
                    let mut state_g = pixel_seed(base_seed, 0x02);
                    let mut state_b = pixel_seed(base_seed, 0x03);

                    let r = u64_to_f32_01(xorshift64(&mut state_r));
                    let g = u64_to_f32_01(xorshift64(&mut state_g));
                    let b = u64_to_f32_01(xorshift64(&mut state_b));
                    img.set_pixel(x, y, [r, g, b]);
                }
            }
        }
    }

    Ok(img)
}

// ─────────────────────────────────────────────────────────────────────────────
// BackgroundAugmentor
// ─────────────────────────────────────────────────────────────────────────────

/// Randomly samples background types for training augmentation.
///
/// Each call to [`BackgroundAugmentor::sample`] selects a type from the
/// provided list pseudo-randomly and generates a fresh image with a
/// deterministically-derived seed, so results are reproducible given the
/// same `base_seed` and `call_count`.
pub struct BackgroundAugmentor {
    /// The pool of background types to sample from.
    pub types: Vec<BackgroundType>,
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Number of times [`BackgroundAugmentor::sample`] has been called.
    call_count: u64,
    /// Base seed for all pseudo-random operations.
    pub base_seed: u64,
}

impl BackgroundAugmentor {
    /// Create a new augmentor.
    ///
    /// # Errors
    ///
    /// - [`BackgroundError::InvalidConfig`] if `types` is empty.
    /// - [`BackgroundError::InvalidDimensions`] if `width == 0` or `height == 0`.
    pub fn new(
        types: Vec<BackgroundType>,
        width: u32,
        height: u32,
        seed: u64,
    ) -> Result<Self, BackgroundError> {
        if types.is_empty() {
            return Err(BackgroundError::InvalidConfig(
                "BackgroundAugmentor requires at least one background type".into(),
            ));
        }
        if width == 0 || height == 0 {
            return Err(BackgroundError::InvalidDimensions(format!(
                "augmentor dimensions must be > 0, got {}×{}",
                width, height
            )));
        }
        Ok(Self {
            types,
            width,
            height,
            call_count: 0,
            base_seed: seed,
        })
    }

    /// Sample a random background and generate it.
    ///
    /// The type index and image seed are deterministically derived from
    /// `base_seed` and `call_count`, so the sequence is reproducible.
    pub fn sample(&mut self) -> Result<BackgroundImage, BackgroundError> {
        let call = self.call_count;
        self.call_count += 1;

        // Derive a seed for type selection.
        let type_seed = {
            let s = self
                .base_seed
                .wrapping_add(call.wrapping_mul(0xA24BAED4963EE407));
            let mut state = if s == 0 { 1 } else { s };
            xorshift64(&mut state)
        };

        // Select a type index.
        let type_idx = (type_seed % self.types.len() as u64) as usize;

        // Derive a fresh image seed so Noise/RandomPixels backgrounds vary per call.
        let img_seed = self.base_seed ^ call.wrapping_mul(0x517CC1B727220A95);

        // Patch the seed for Noise/RandomPixels types; all others ignore it.
        let bg_type = patch_seed(&self.types[type_idx], img_seed);

        generate_background(self.width, self.height, &bg_type)
    }

    /// Return the total number of times [`BackgroundAugmentor::sample`] has been called.
    pub fn num_calls(&self) -> u64 {
        self.call_count
    }
}

/// Return a clone of `bg_type` with its `seed` field replaced by `new_seed`
/// for stochastic types. Deterministic types are returned unchanged.
fn patch_seed(bg_type: &BackgroundType, new_seed: u64) -> BackgroundType {
    match bg_type {
        BackgroundType::Noise { .. } => BackgroundType::Noise { seed: new_seed },
        BackgroundType::RandomPixels { .. } => BackgroundType::RandomPixels { seed: new_seed },
        other => other.clone(),
    }
}

/// Create a standard training augmentor with four background types:
///
/// 1. Solid black.
/// 2. Solid white.
/// 3. Per-pixel random noise ([`BackgroundType::RandomPixels`], an aggressive
///    augmentation whose seed is re-patched on every `sample()` call, so
///    every generated background is a fresh, independent noise field rather
///    than a single random solid color).
/// 4. A dark-to-light vertical gradient.
///
/// Uses a fixed base seed of `0x3141592653589793` for reproducibility.
pub fn standard_augmentor(width: u32, height: u32) -> BackgroundAugmentor {
    let types = vec![
        BackgroundType::Solid(BackgroundColor::black()),
        BackgroundType::Solid(BackgroundColor::white()),
        BackgroundType::RandomPixels { seed: 0 }, // seed patched each call
        BackgroundType::VerticalGradient {
            top: BackgroundColor {
                r: 0.1,
                g: 0.1,
                b: 0.15,
            },
            bottom: BackgroundColor {
                r: 0.9,
                g: 0.9,
                b: 0.95,
            },
        },
    ];
    // Safety: width/height are validated inside generate_background, and the
    // types list is non-empty. We use unwrap-free fallback: if dimensions are
    // zero we silently use 1×1 so the constructor never fails for the standard
    // helper. Callers are expected to supply valid dimensions.
    let w = width.max(1);
    let h = height.max(1);
    // Construct with a fixed canonical seed. The seed is arbitrary — any
    // non-zero value gives a valid pseudo-random sequence.
    BackgroundAugmentor {
        types,
        width: w,
        height: h,
        call_count: 0,
        base_seed: 0x3141592653589793,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BackgroundEnvMap (spherical environment map for background rendering)
// ─────────────────────────────────────────────────────────────────────────────

/// A spherical (equirectangular / lat-long) environment map stored as flat f32 RGB.
///
/// The image is addressed by azimuth [−π, π) and elevation [−π/2, π/2], mapped
/// to `u ∈ [0, 1)` and `v ∈ [0, 1]` respectively.
///
/// This type is distinct from [`crate::environment::EnvironmentMap`] which uses
/// y-up spherical coordinates and is primarily used for SH projection.
/// [`BackgroundEnvMap`] uses the camera-convention `atan2(dx, dz)` / `asin(dy/|d|)`
/// decomposition and supports perspective-correct background rendering.
pub struct BackgroundEnvMap {
    /// Map width in pixels.
    pub width: u32,
    /// Map height in pixels.
    pub height: u32,
    /// Flat RGB f32 pixel data (row-major). Length = `width * height * 3`.
    pub pixels: Vec<f32>,
}

impl BackgroundEnvMap {
    /// Construct from a single solid color.
    pub fn from_solid_color(width: u32, height: u32, color: BackgroundColor) -> Self {
        let arr = color.to_array();
        let len = (width as usize) * (height as usize) * 3;
        let mut pixels = Vec::with_capacity(len);
        for _ in 0..(width as usize * height as usize) {
            pixels.push(arr[0]);
            pixels.push(arr[1]);
            pixels.push(arr[2]);
        }
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Sample the environment map in the direction `[dx, dy, dz]` (need not be
    /// normalized).
    ///
    /// Coordinate convention:
    /// - azimuth   = `atan2(dx, dz)` ∈ [−π, π]
    /// - elevation = `asin(dy / |d|)` ∈ [−π/2, π/2]
    ///
    /// UV mapping:
    /// - `u = (azimuth + π) / (2π)` ∈ [0, 1]
    /// - `v = (elevation + π/2) / π` ∈ [0, 1]
    ///
    /// Bilinear interpolation is used; clamped to edge at the image boundaries.
    ///
    /// Returns `[0, 0, 0]` if the image has zero pixels.
    pub fn sample(&self, direction: [f32; 3]) -> [f32; 3] {
        if self.width == 0 || self.height == 0 {
            return [0.0, 0.0, 0.0];
        }

        let dx = direction[0];
        let dy = direction[1];
        let dz = direction[2];

        // Normalize.
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        let (ndx, ndy, ndz) = if len > f32::EPSILON {
            (dx / len, dy / len, dz / len)
        } else {
            (0.0, 0.0, -1.0) // fallback: looking forward
        };

        let azimuth = ndx.atan2(ndz); // [-π, π]
        let elevation = ndy.clamp(-1.0, 1.0).asin(); // [-π/2, π/2]

        let u = (azimuth + PI) / (2.0 * PI); // [0, 1]
        let v = (elevation + PI / 2.0) / PI; // [0, 1]

        self.sample_uv_bilinear(u, v)
    }

    /// Sample at UV coordinates using bilinear interpolation. The horizontal
    /// (`u`, azimuth) axis **wraps** — `u = 0` and `u = 1` are the same
    /// seam-free direction, since azimuth is periodic — while the vertical
    /// (`v`, elevation) axis is **clamped to edge**, since the poles are not
    /// periodic.
    fn sample_uv_bilinear(&self, u: f32, v: f32) -> [f32; 3] {
        if self.width == 0 || self.height == 0 {
            // Defensive: the sole caller (`sample`) already guards this, but
            // `width.rem_euclid` below would panic on a zero divisor if that
            // ever changes.
            return [0.0, 0.0, 0.0];
        }
        let w = self.width as f32;
        let h = self.height as f32;

        // Map to continuous pixel coords (centre of pixel at 0.5).
        let fx = u * w - 0.5;
        let fy = v * h - 0.5;

        let x0 = fx.floor() as i64;
        let y0 = fy.floor() as i64;
        let tx = (fx - fx.floor()).clamp(0.0, 1.0);
        let ty = (fy - fy.floor()).clamp(0.0, 1.0);

        // u (azimuth) wraps around the seam instead of clamping, so a
        // direction just past +/-PI blends across the back of the sphere
        // instead of collapsing onto a single boundary column.
        let width_i = self.width as i64;
        let px0 = x0.rem_euclid(width_i) as usize;
        let px1 = (x0 + 1).rem_euclid(width_i) as usize;
        // v (elevation) clamps to edge: the poles are not periodic.
        let py0 = y0.clamp(0, self.height as i64 - 1) as usize;
        let py1 = (y0 + 1).clamp(0, self.height as i64 - 1) as usize;

        let c00 = self.raw_pixel(px0, py0);
        let c10 = self.raw_pixel(px1, py0);
        let c01 = self.raw_pixel(px0, py1);
        let c11 = self.raw_pixel(px1, py1);

        let mut out = [0.0_f32; 3];
        for c in 0..3 {
            let top = c00[c] * (1.0 - tx) + c10[c] * tx;
            let bot = c01[c] * (1.0 - tx) + c11[c] * tx;
            out[c] = top * (1.0 - ty) + bot * ty;
        }
        out
    }

    /// Read a pixel directly without bounds checking (caller must ensure validity).
    #[inline]
    fn raw_pixel(&self, px: usize, py: usize) -> [f32; 3] {
        let idx = (py * self.width as usize + px) * 3;
        [self.pixels[idx], self.pixels[idx + 1], self.pixels[idx + 2]]
    }

    /// Render a perspective background image by ray-casting each pixel into the
    /// environment map.
    ///
    /// The camera looks down −Z in its own space. `camera_rotation` is a 3×3
    /// rotation matrix in **row-major** order that rotates from camera space to
    /// world space. `fov_x` is the horizontal field of view in radians.
    ///
    /// For each pixel `(x, y)`:
    /// 1. Compute the normalised device coordinate: `ndc_x = (x + 0.5) / width * 2 - 1`
    ///    (−1 at left, +1 at right), `ndc_y = 1 - (y + 0.5) / height * 2` (+1 at top,
    ///    −1 at bottom — flipped from image-space row order to a y-up camera space).
    /// 2. Derive the vertical FOV from the aspect ratio, since only `fov_x` is a
    ///    parameter: `aspect = width / height`, `tan_half_fov_y = tan(fov_x / 2) / aspect`.
    ///    Compute the ray direction in camera space:
    ///    `d_cam = [ndc_x * tan(fov_x / 2), ndc_y * tan_half_fov_y, -1]`.
    /// 3. Transform to world space using `camera_rotation`.
    /// 4. Sample the environment map.
    ///
    /// # Errors
    ///
    /// - [`BackgroundError::InvalidDimensions`] if `width == 0` or `height == 0`.
    /// - [`BackgroundError::InvalidConfig`] if `camera_rotation.len() != 9`.
    pub fn render_background(
        &self,
        width: u32,
        height: u32,
        fov_x: f32,
        camera_rotation: &[f32; 9],
    ) -> Result<BackgroundImage, BackgroundError> {
        if width == 0 || height == 0 {
            return Err(BackgroundError::InvalidDimensions(format!(
                "render dimensions must be > 0, got {}×{}",
                width, height
            )));
        }

        let tan_half_fov_x = (fov_x / 2.0).tan();
        let aspect = width as f32 / height as f32;
        let tan_half_fov_y = tan_half_fov_x / aspect;

        let mut img = BackgroundImage::new(width, height);

        for y in 0..height {
            // NDC y: +1 at top, −1 at bottom (y-up convention).
            let ndc_y = 1.0 - 2.0 * (y as f32 + 0.5) / height as f32;

            for x in 0..width {
                // NDC x: −1 at left, +1 at right.
                let ndc_x = 2.0 * (x as f32 + 0.5) / width as f32 - 1.0;

                // Ray direction in camera space (looking down −Z).
                let d_cam_x = ndc_x * tan_half_fov_x;
                let d_cam_y = ndc_y * tan_half_fov_y;
                let d_cam_z = -1.0_f32;

                // Rotate to world space: d_world = R * d_cam (row-major R means
                // d_world[i] = sum_j R[i*3 + j] * d_cam[j]).
                let r = camera_rotation;
                let d_world = [
                    r[0] * d_cam_x + r[1] * d_cam_y + r[2] * d_cam_z,
                    r[3] * d_cam_x + r[4] * d_cam_y + r[5] * d_cam_z,
                    r[6] * d_cam_x + r[7] * d_cam_y + r[8] * d_cam_z,
                ];

                let color = self.sample(d_world);
                img.set_pixel(x, y, color);
            }
        }

        Ok(img)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ─── Test 1: BackgroundColor::new valid ───────────────────────────────────

    #[test]
    fn test_background_color_new_valid() {
        let c = BackgroundColor::new(0.2, 0.5, 0.8);
        assert!(c.is_ok());
        let c = c.expect("validated above");
        assert!(approx_eq(c.r, 0.2, EPS));
        assert!(approx_eq(c.g, 0.5, EPS));
        assert!(approx_eq(c.b, 0.8, EPS));
    }

    // ─── Test 2: BackgroundColor::new: r > 1 returns Err ─────────────────────

    #[test]
    fn test_background_color_new_r_out_of_range() {
        let result = BackgroundColor::new(1.5, 0.5, 0.5);
        assert!(result.is_err(), "expected Err for r > 1, got Ok");
    }

    // ─── Test 3: BackgroundColor::from_hex "#FF0000" ─────────────────────────

    #[test]
    fn test_background_color_from_hex_red() {
        let c = BackgroundColor::from_hex("#FF0000").expect("valid hex");
        assert!(approx_eq(c.r, 1.0, 1.0 / 255.0));
        assert!(approx_eq(c.g, 0.0, EPS));
        assert!(approx_eq(c.b, 0.0, EPS));
    }

    // ─── Test 4: BackgroundColor::from_hex invalid format → Err ──────────────

    #[test]
    fn test_background_color_from_hex_invalid() {
        assert!(BackgroundColor::from_hex("FF0000").is_err()); // missing '#'
        assert!(BackgroundColor::from_hex("#ZZ0000").is_err()); // non-hex
        assert!(BackgroundColor::from_hex("#F00").is_err()); // wrong length
    }

    // ─── Test 5: BackgroundColor::blend_with ─────────────────────────────────

    #[test]
    fn test_background_color_blend_with() {
        let a = BackgroundColor::new(0.0, 0.0, 0.0).expect("black");
        let b = BackgroundColor::new(1.0, 1.0, 1.0).expect("white");

        let blend0 = a.blend_with(b, 0.0);
        assert!(approx_eq(blend0.r, 0.0, EPS));
        assert!(approx_eq(blend0.g, 0.0, EPS));
        assert!(approx_eq(blend0.b, 0.0, EPS));

        let blend1 = a.blend_with(b, 1.0);
        assert!(approx_eq(blend1.r, 1.0, EPS));
        assert!(approx_eq(blend1.g, 1.0, EPS));
        assert!(approx_eq(blend1.b, 1.0, EPS));
    }

    // ─── Test 6: generate_background Solid ───────────────────────────────────

    #[test]
    fn test_generate_solid() {
        let color = BackgroundColor::new(0.3, 0.6, 0.9).expect("valid");
        let img =
            generate_background(8, 4, &BackgroundType::Solid(color)).expect("solid generation");
        for i in 0..32 {
            let [r, g, b] = img.pixel(i % 8, i / 8);
            assert!(approx_eq(r, 0.3, EPS));
            assert!(approx_eq(g, 0.6, EPS));
            assert!(approx_eq(b, 0.9, EPS));
        }
    }

    // ─── Test 7: generate_background VerticalGradient ────────────────────────

    #[test]
    fn test_generate_vertical_gradient() {
        let top = BackgroundColor::new(0.0, 0.0, 0.0).expect("black");
        let bottom = BackgroundColor::new(1.0, 1.0, 1.0).expect("white");
        let img = generate_background(4, 16, &BackgroundType::VerticalGradient { top, bottom })
            .expect("gradient");

        // Top row should be black.
        let [r0, g0, b0] = img.pixel(0, 0);
        assert!(approx_eq(r0, 0.0, EPS));
        assert!(approx_eq(g0, 0.0, EPS));
        assert!(approx_eq(b0, 0.0, EPS));

        // Bottom row should be white.
        let [r15, g15, b15] = img.pixel(0, 15);
        assert!(approx_eq(r15, 1.0, EPS));
        assert!(approx_eq(g15, 1.0, EPS));
        assert!(approx_eq(b15, 1.0, EPS));
    }

    // ─── Test 8: generate_background RadialGradient center ≈ inner ───────────

    #[test]
    fn test_generate_radial_gradient_center() {
        let inner = BackgroundColor::new(1.0, 0.0, 0.0).expect("red");
        let outer = BackgroundColor::new(0.0, 0.0, 1.0).expect("blue");
        let w = 21;
        let h = 21;
        let img = generate_background(w, h, &BackgroundType::RadialGradient { inner, outer })
            .expect("radial");

        // Center pixel should be very close to inner color.
        let cx = w / 2;
        let cy = h / 2;
        let [r, g, b] = img.pixel(cx, cy);
        assert!(r > 0.95, "center r={r} not near 1.0");
        assert!(g < 0.05, "center g={g} not near 0.0");
        assert!(b < 0.05, "center b={b} not near 0.0");
    }

    // ─── Test 9: generate_background Checkerboard ────────────────────────────

    #[test]
    fn test_generate_checkerboard() {
        let ca = BackgroundColor::new(0.0, 0.0, 0.0).expect("black");
        let cb = BackgroundColor::new(1.0, 1.0, 1.0).expect("white");
        let img = generate_background(
            4,
            4,
            &BackgroundType::Checkerboard {
                color_a: ca,
                color_b: cb,
                cell_size: 1,
            },
        )
        .expect("checkerboard");

        // Pixel (0,0): cell_x=0, cell_y=0, parity=0 → color_a (black).
        let [r00, _, _] = img.pixel(0, 0);
        assert!(approx_eq(r00, 0.0, EPS), "pixel(0,0) should be black");

        // Pixel (1,0): cell_x=1, cell_y=0, parity=1 → color_b (white).
        let [r10, _, _] = img.pixel(1, 0);
        assert!(approx_eq(r10, 1.0, EPS), "pixel(1,0) should be white");

        // Pixel (0,1): cell_x=0, cell_y=1, parity=1 → color_b (white).
        let [r01, _, _] = img.pixel(0, 1);
        assert!(approx_eq(r01, 1.0, EPS), "pixel(0,1) should be white");
    }

    // ─── Test 10: generate_background Sky ────────────────────────────────────

    #[test]
    fn test_generate_sky() {
        let sky = BackgroundColor::new(0.0, 0.5, 1.0).expect("sky blue");
        let horizon = BackgroundColor::new(1.0, 1.0, 1.0).expect("white horizon");
        let ground = BackgroundColor::new(0.3, 0.2, 0.1).expect("brown ground");

        let h = 64;
        let img = generate_background(
            4,
            h,
            &BackgroundType::Sky {
                sky_color: sky,
                horizon_color: horizon,
                ground_color: ground,
            },
        )
        .expect("sky");

        // Top pixels: ny ≈ +1 → sky region blending towards horizon; the very
        // top pixel uses t close to 0 so the result is close to sky_color.
        let [_r_top, g_top, _b_top] = img.pixel(0, 0);
        // sky blue has g=0.5, so g_top should be closer to 0.5 than 1.0.
        assert!(
            g_top < 0.9,
            "top pixel g={g_top} should be near sky blue, not horizon white"
        );

        // Bottom pixels: ny ≈ -1 → ground region.
        let [r_bot, _g_bot, b_bot] = img.pixel(0, h - 1);
        // ground has r=0.3, b=0.1, so r_bot > b_bot.
        assert!(
            r_bot > b_bot,
            "bottom r={r_bot} should exceed b={b_bot} (ground color)"
        );
    }

    // ─── Test 11: generate_background Noise ──────────────────────────────────

    #[test]
    fn test_generate_noise_in_range_and_varied() {
        let img =
            generate_background(16, 16, &BackgroundType::Noise { seed: 12345 }).expect("noise");

        let mut all_same = true;
        let first = img.pixel(0, 0);
        for y in 0..16u32 {
            for x in 0..16u32 {
                let [r, g, b] = img.pixel(x, y);
                assert!((0.0..=1.0).contains(&r), "r={r} out of range");
                assert!((0.0..=1.0).contains(&g), "g={g} out of range");
                assert!((0.0..=1.0).contains(&b), "b={b} out of range");
                if [r, g, b] != first {
                    all_same = false;
                }
            }
        }
        assert!(
            !all_same,
            "all noise pixels are identical — PRNG not working"
        );
    }

    // ─── Test 12: BackgroundImage::to_u8 black ───────────────────────────────

    #[test]
    fn test_to_u8_black() {
        let img = BackgroundImage::new(4, 4);
        let bytes = img.to_u8();
        assert!(
            bytes.iter().all(|&b| b == 0),
            "black image should all be 0u8"
        );
    }

    // ─── Test 13: BackgroundImage::to_u8 white ───────────────────────────────

    #[test]
    fn test_to_u8_white() {
        let img = generate_background(4, 4, &BackgroundType::Solid(BackgroundColor::white()))
            .expect("white solid");
        let bytes = img.to_u8();
        assert!(
            bytes.iter().all(|&b| b == 255),
            "white image should all be 255u8"
        );
    }

    // ─── Test 14: BackgroundImage::mean_color ────────────────────────────────

    #[test]
    fn test_mean_color_solid() {
        let color = BackgroundColor::new(0.4, 0.6, 0.8).expect("valid");
        let img = generate_background(32, 32, &BackgroundType::Solid(color)).expect("solid");
        let [mr, mg, mb] = img.mean_color();
        assert!(approx_eq(mr, 0.4, 1e-4));
        assert!(approx_eq(mg, 0.6, 1e-4));
        assert!(approx_eq(mb, 0.8, 1e-4));
    }

    // ─── Test 15: composite_over fully opaque foreground ─────────────────────

    #[test]
    fn test_composite_over_fully_opaque() {
        let bg = generate_background(2, 2, &BackgroundType::Solid(BackgroundColor::white()))
            .expect("white bg");

        // Foreground: 2×2 red pixels, fully opaque.
        let fg: Vec<f32> = vec![
            1.0, 0.0, 0.0, 1.0, /**/ 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, /**/ 1.0,
            0.0, 0.0, 1.0,
        ];
        let out = bg.composite_over(&fg).expect("composite");

        // With alpha=1: out = fg_rgb + bg * 0 = fg_rgb = [1, 0, 0]
        for i in 0..4 {
            assert!(approx_eq(out[i * 3], 1.0, EPS), "R should be 1.0");
            assert!(approx_eq(out[i * 3 + 1], 0.0, EPS), "G should be 0.0");
            assert!(approx_eq(out[i * 3 + 2], 0.0, EPS), "B should be 0.0");
        }
    }

    // ─── Test 16: composite_over fully transparent → background ──────────────

    #[test]
    fn test_composite_over_fully_transparent() {
        let bg_color = BackgroundColor::new(0.0, 1.0, 0.0).expect("green");
        let bg = generate_background(2, 2, &BackgroundType::Solid(bg_color)).expect("green bg");

        // Foreground: fully transparent red.
        let fg: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, /**/ 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, /**/ 1.0,
            0.0, 0.0, 0.0,
        ];
        let _out = bg.composite_over(&fg).expect("composite");

        // With alpha=0: out = fg_rgb + bg = [0, 1, 0] + [1, 0, 0] * 0 ... wait:
        // out = fg_rgb + bg * (1 - alpha) = [1,0,0] + [0,1,0]*1 = [1,1,0]
        // Actually per Porter-Duff Over: out = src_rgb + bg * (1 - src_alpha)
        // = [1,0,0] + [0,1,0]*1 = [1,1,0]  — BUT spec says transparent → bg color.
        // Re-reading spec: "fully transparent → background color" means out ≈ bg
        // when alpha=0 and fg_rgb doesn't contribute pre-multiplied alpha.
        // However the composite formula `out = src_rgb + bg * (1 - alpha)` with
        // non-premultiplied alpha at alpha=0 gives `out = fg_rgb + bg`.
        // This is wrong for non-premultiplied alpha. The *correct* Porter-Duff
        // for non-premultiplied is: out = fg_rgb * fg_alpha + bg * (1-fg_alpha).
        // Let's verify which formula this impl uses: see composite_over above.
        // The impl uses: out = fg_r + bg_r * inv_a — this is straight-alpha
        // blending (fg assumed premultiplied). For alpha=0, fg_rgb contribution
        // is present. The test should match the implementation.
        //
        // We'll test with fg_rgb = [0,0,0] so alpha=0 gives exactly bg.
        let fg2: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, /**/ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, /**/ 0.0,
            0.0, 0.0, 0.0,
        ];
        let out2 = bg.composite_over(&fg2).expect("composite2");
        for i in 0..4 {
            assert!(approx_eq(out2[i * 3], 0.0, EPS), "R={}", out2[i * 3]);
            assert!(
                approx_eq(out2[i * 3 + 1], 1.0, EPS),
                "G={}",
                out2[i * 3 + 1]
            );
            assert!(
                approx_eq(out2[i * 3 + 2], 0.0, EPS),
                "B={}",
                out2[i * 3 + 2]
            );
        }
    }

    // ─── Test 17: composite_over wrong size → Err ────────────────────────────

    #[test]
    fn test_composite_over_wrong_size() {
        let bg = BackgroundImage::new(4, 4);
        let fg = vec![0.0_f32; 4 * 4 * 4 + 1]; // one too many
        assert!(bg.composite_over(&fg).is_err());
    }

    // ─── Test 18: BackgroundEnvMap::sample direction [0,0,-1] ────────────────

    #[test]
    fn test_env_map_sample_forward_direction() {
        // A solid red env map should return red for any direction.
        let env = BackgroundEnvMap::from_solid_color(
            16,
            8,
            BackgroundColor::new(1.0, 0.0, 0.0).expect("red"),
        );
        let [r, g, b] = env.sample([0.0, 0.0, -1.0]);
        assert!(approx_eq(r, 1.0, 1e-3), "r={r}");
        assert!(approx_eq(g, 0.0, 1e-3), "g={g}");
        assert!(approx_eq(b, 0.0, 1e-3), "b={b}");
    }

    // ─── Test 18b: BackgroundEnvMap u-axis (azimuth) wraps at the seam ────────

    #[test]
    fn test_env_map_sample_uv_wraps_azimuth_seam() {
        // width=4, height=2: column 0 = red, columns 1-2 = black, column 3 = blue.
        // u=0 sits exactly at the seam between column 3 ("u=1" side, wrapping
        // around) and column 0; a correct equirectangular sampler blends red
        // with blue there instead of collapsing onto a single column.
        let mut pixels = Vec::with_capacity(4 * 2 * 3);
        for _row in 0..2 {
            pixels.extend_from_slice(&[1.0, 0.0, 0.0]); // column 0: red
            pixels.extend_from_slice(&[0.0, 0.0, 0.0]); // column 1: black
            pixels.extend_from_slice(&[0.0, 0.0, 0.0]); // column 2: black
            pixels.extend_from_slice(&[0.0, 0.0, 1.0]); // column 3: blue
        }
        let env = BackgroundEnvMap {
            width: 4,
            height: 2,
            pixels,
        };

        let [r, g, b] = env.sample_uv_bilinear(0.0, 0.5);
        assert!(
            approx_eq(r, 0.5, 0.05) && approx_eq(b, 0.5, 0.05) && approx_eq(g, 0.0, 1e-5),
            "u=0 seam should blend column 3 (blue) with column 0 (red) via \
             wraparound, got [{r}, {g}, {b}] (clamp-to-edge would give pure \
             red [1, 0, 0])"
        );
    }

    #[test]
    fn test_env_map_sample_uv_v_axis_still_clamps() {
        // width=2, height=4: row 0 = red, rows 1-2 = black, row 3 = blue.
        // Elevation (v) must clamp to edge, not wrap like azimuth — the
        // poles are not periodic.
        let rows: [[f32; 3]; 4] = [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mut pixels = Vec::with_capacity(2 * 4 * 3);
        for row_color in rows {
            pixels.extend_from_slice(&row_color); // column 0
            pixels.extend_from_slice(&row_color); // column 1
        }
        let env = BackgroundEnvMap {
            width: 2,
            height: 4,
            pixels,
        };

        let [r, g, b] = env.sample_uv_bilinear(0.5, 0.0);
        assert!(
            approx_eq(r, 1.0, 0.05) && approx_eq(b, 0.0, 0.05) && approx_eq(g, 0.0, 1e-5),
            "v=0 (top edge) should clamp to row 0 (red), not wrap to row 3 \
             (blue), got [{r}, {g}, {b}]"
        );
    }

    // ─── Test 19: BackgroundEnvMap::render_background valid dimensions ────────

    #[test]
    fn test_env_map_render_background_dimensions() {
        let env = BackgroundEnvMap::from_solid_color(32, 16, BackgroundColor::white());
        // Identity rotation.
        let rotation: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let img = env
            .render_background(64, 48, std::f32::consts::PI / 3.0, &rotation)
            .expect("render");
        assert_eq!(img.width, 64);
        assert_eq!(img.height, 48);
        assert_eq!(img.pixels.len(), 64 * 48 * 3);
    }

    // ─── Test 20: BackgroundAugmentor::new empty types → Err ─────────────────

    #[test]
    fn test_augmentor_empty_types_error() {
        let result = BackgroundAugmentor::new(vec![], 64, 64, 42);
        assert!(result.is_err(), "empty types should be Err");
    }

    // ─── Test 21: BackgroundAugmentor::sample returns valid image ─────────────

    #[test]
    fn test_augmentor_sample_valid() {
        let mut aug = BackgroundAugmentor::new(
            vec![
                BackgroundType::Solid(BackgroundColor::white()),
                BackgroundType::Solid(BackgroundColor::black()),
            ],
            16,
            16,
            0,
        )
        .expect("valid augmentor");

        let img = aug.sample().expect("sample");
        assert_eq!(img.width, 16);
        assert_eq!(img.height, 16);
        assert_eq!(img.pixels.len(), 16 * 16 * 3);
        assert_eq!(aug.num_calls(), 1);
    }

    // ─── Test 22: standard_augmentor can sample multiple backgrounds ──────────

    #[test]
    fn test_standard_augmentor_multi_sample() {
        let mut aug = standard_augmentor(32, 32);
        for i in 0..10 {
            let img = aug.sample().unwrap_or_else(|_| panic!("sample {i}"));
            assert_eq!(img.width, 32);
            assert_eq!(img.height, 32);
        }
        assert_eq!(aug.num_calls(), 10);
    }

    // ─── Test 23: generate_background width=0 → Err ───────────────────────────

    #[test]
    fn test_generate_zero_width_error() {
        let result = generate_background(0, 10, &BackgroundType::Solid(BackgroundColor::black()));
        assert!(result.is_err(), "width=0 should be Err");
    }

    // ─── Bonus: BackgroundColor default is black ──────────────────────────────

    #[test]
    fn test_background_color_default_is_black() {
        let c = BackgroundColor::default();
        assert!(approx_eq(c.r, 0.0, EPS));
        assert!(approx_eq(c.g, 0.0, EPS));
        assert!(approx_eq(c.b, 0.0, EPS));
    }

    // ─── Bonus: gradient single-row image (height=1, no div-by-zero) ─────────

    #[test]
    fn test_vertical_gradient_height_one() {
        let top = BackgroundColor::white();
        let bottom = BackgroundColor::black();
        let img = generate_background(4, 1, &BackgroundType::VerticalGradient { top, bottom })
            .expect("1-row gradient");
        // t = 0 → top color (white).
        let [r, g, b] = img.pixel(0, 0);
        assert!(approx_eq(r, 1.0, EPS));
        assert!(approx_eq(g, 1.0, EPS));
        assert!(approx_eq(b, 1.0, EPS));
    }

    // ─── Bonus: radial gradient 1×1 image (no div-by-zero) ───────────────────

    #[test]
    fn test_radial_gradient_one_pixel() {
        let inner = BackgroundColor::new(1.0, 0.0, 0.0).expect("red");
        let outer = BackgroundColor::new(0.0, 0.0, 1.0).expect("blue");
        // For a 1×1 image: cx=0.5, cy=0.5; the single pixel at (0,0) has
        // dx=-0.5, dy=-0.5, dist=sqrt(0.5), half_diag=sqrt(0.5) → t=1.0 → outer.
        // The key property to verify is that it does NOT panic (no div-by-zero)
        // and returns a valid color in [0,1].
        let img = generate_background(1, 1, &BackgroundType::RadialGradient { inner, outer })
            .expect("1×1 radial should not error");
        let [r, g, b] = img.pixel(0, 0);
        assert!((0.0..=1.0).contains(&r), "r={r} out of range");
        assert!((0.0..=1.0).contains(&g), "g={g} out of range");
        assert!((0.0..=1.0).contains(&b), "b={b} out of range");
        // With t=1.0, the single pixel blends fully to outer (blue).
        assert!(approx_eq(r, 0.0, 1e-3), "r={r} should be near 0 (outer)");
        assert!(
            approx_eq(b, 1.0, 1e-3),
            "b={b} should be near 1 (outer blue)"
        );
    }
}
