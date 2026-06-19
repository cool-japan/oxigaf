//! Physically-motivated lens post-processing for u8 RGBA images.
//!
//! This module implements chromatic aberration, barrel/pincushion distortion,
//! and vignetting effects on flat `Vec<u8>` RGBA images (row-major,
//! 4 bytes per pixel).  These effects complement the f32-RGB API in
//! [`chromatic_aberration`][crate::chromatic_aberration].
//!
//! # Quick start
//!
//! ```rust
//! use oxigaf_render::lens_effects::{
//!     LensEffectsConfig, ChromAberrationConfig, apply_lens_effects,
//! };
//!
//! let width  = 4u32;
//! let height = 4u32;
//! let image  = vec![128u8; (width * height * 4) as usize];
//! let config = LensEffectsConfig {
//!     chromatic_aberration: Some(ChromAberrationConfig::subtle()),
//!     distortion: None,
//!     vignette: None,
//! };
//! let out = apply_lens_effects(&image, width, height, &config).expect("lens effects failed");
//! assert_eq!(out.len(), image.len());
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the RGBA lens-effects functions.
#[derive(Debug, Error, PartialEq)]
pub enum LensEffectError {
    /// Buffer supplied to the function was empty (or a dimension was zero).
    #[error("Empty image buffer")]
    EmptyImage,

    /// Buffer length does not match `width × height × 4`.
    #[error(
        "Image buffer {actual} bytes does not match {width}\u{d7}{height}\u{d7}4 = {expected}"
    )]
    InvalidDimensions {
        /// Actual buffer length in bytes.
        actual: usize,
        /// Expected buffer length in bytes.
        expected: usize,
        /// Declared image width.
        width: u32,
        /// Declared image height.
        height: u32,
    },

    /// `strength` was outside `[0, 1]`.
    #[error("Invalid aberration strength {strength}: must be in [0, 1]")]
    InvalidAberrationStrength {
        /// The offending value.
        strength: f32,
    },

    /// A radial distortion coefficient was outside `[-1, 1]`.
    #[error("Invalid distortion coefficient {k}: must be in [-1, 1]")]
    InvalidDistortionCoeff {
        /// The offending coefficient value.
        k: f32,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Config structures
// ─────────────────────────────────────────────────────────────────────────────

/// Per-channel radial-shift chromatic aberration (RGBA u8 API).
///
/// Each channel is sampled from a slightly different radial offset from the
/// image centre.  `shift > 0` pushes the sample outward; `shift < 0` pulls it
/// inward.
#[derive(Debug, Clone, PartialEq)]
pub struct ChromAberrationConfig {
    /// Radial shift factor for the red channel (default `0.005`).
    pub red_shift: f32,
    /// Radial shift factor for the green channel (default `0.0`).
    pub green_shift: f32,
    /// Radial shift factor for the blue channel (default `-0.005`).
    pub blue_shift: f32,
    /// Overall effect strength in `[0, 1]` (default `1.0`).
    pub strength: f32,
}

impl Default for ChromAberrationConfig {
    fn default() -> Self {
        Self {
            red_shift: 0.005,
            green_shift: 0.0,
            blue_shift: -0.005,
            strength: 1.0,
        }
    }
}

impl ChromAberrationConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`LensEffectError::InvalidAberrationStrength`] if `strength ∉ [0, 1]`.
    pub fn validate(&self) -> Result<(), LensEffectError> {
        if !(0.0..=1.0).contains(&self.strength) {
            return Err(LensEffectError::InvalidAberrationStrength {
                strength: self.strength,
            });
        }
        Ok(())
    }

    /// Subtle aberration — barely noticeable.
    pub fn subtle() -> Self {
        Self {
            red_shift: 0.002,
            green_shift: 0.0,
            blue_shift: -0.002,
            strength: 1.0,
        }
    }

    /// Moderate aberration — visible colour fringing at edges.
    pub fn moderate() -> Self {
        Self {
            red_shift: 0.01,
            green_shift: 0.0,
            blue_shift: -0.01,
            strength: 1.0,
        }
    }

    /// Strong aberration — pronounced chromatic fringing.
    pub fn strong() -> Self {
        Self {
            red_shift: 0.03,
            green_shift: 0.0,
            blue_shift: -0.03,
            strength: 1.0,
        }
    }
}

/// Barrel/pincushion radial lens distortion (RGBA u8 API).
#[derive(Debug, Clone, PartialEq)]
pub struct DistortionConfig {
    /// First radial distortion coefficient `k1`
    /// (negative → barrel, positive → pincushion).
    /// Must be in `[-1, 1]` (default `0.0`).
    pub k1: f32,
    /// Second radial distortion coefficient `k2`.
    /// Must be in `[-1, 1]` (default `0.0`).
    pub k2: f32,
    /// Overall effect strength in `[0, 1]` (default `1.0`).
    pub strength: f32,
}

impl Default for DistortionConfig {
    fn default() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            strength: 1.0,
        }
    }
}

impl DistortionConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// - [`LensEffectError::InvalidDistortionCoeff`] if `k1` or `k2 ∉ [-1, 1]`.
    /// - [`LensEffectError::InvalidAberrationStrength`] if `strength ∉ [0, 1]`.
    pub fn validate(&self) -> Result<(), LensEffectError> {
        if !(-1.0..=1.0).contains(&self.k1) {
            return Err(LensEffectError::InvalidDistortionCoeff { k: self.k1 });
        }
        if !(-1.0..=1.0).contains(&self.k2) {
            return Err(LensEffectError::InvalidDistortionCoeff { k: self.k2 });
        }
        if !(0.0..=1.0).contains(&self.strength) {
            return Err(LensEffectError::InvalidAberrationStrength {
                strength: self.strength,
            });
        }
        Ok(())
    }

    /// Slight barrel distortion — gently bowing outward.
    pub fn slight_barrel() -> Self {
        Self {
            k1: -0.1,
            k2: 0.0,
            strength: 1.0,
        }
    }

    /// Slight pincushion distortion — gently bowing inward.
    pub fn slight_pincushion() -> Self {
        Self {
            k1: 0.1,
            k2: 0.0,
            strength: 1.0,
        }
    }
}

/// Vignetting configuration for the RGBA u8 lens pipeline.
///
/// For a more expressive multi-model API see the `vignetting` module's
/// [`VignettingConfig`][crate::VignettingConfig].
#[derive(Debug, Clone, PartialEq)]
pub struct LensVignetteConfig {
    /// Vignette strength in `[0, 1]` (default `0.5`).
    pub strength: f32,
    /// Normalised radius where vignetting begins (default `0.7`).
    pub radius: f32,
    /// Falloff steepness exponent (default `2.0`).
    pub falloff: f32,
    /// Vignette colour as linear RGB in `[0, 1]` (default `[0, 0, 0]` — black).
    pub color: [f32; 3],
}

impl Default for LensVignetteConfig {
    fn default() -> Self {
        Self {
            strength: 0.5,
            radius: 0.7,
            falloff: 2.0,
            color: [0.0, 0.0, 0.0],
        }
    }
}

impl LensVignetteConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`LensEffectError::InvalidAberrationStrength`] if `strength ∉ [0, 1]`.
    pub fn validate(&self) -> Result<(), LensEffectError> {
        if !(0.0..=1.0).contains(&self.strength) {
            return Err(LensEffectError::InvalidAberrationStrength {
                strength: self.strength,
            });
        }
        Ok(())
    }

    /// Subtle vignette — barely noticeable darkening.
    pub fn subtle() -> Self {
        Self {
            strength: 0.3,
            ..Self::default()
        }
    }

    /// Cinematic vignette — strong darkening for a filmic look.
    pub fn cinema() -> Self {
        Self {
            strength: 0.7,
            radius: 0.5,
            ..Self::default()
        }
    }
}

/// Combined lens effects configuration (RGBA u8 API).
#[derive(Debug, Clone, Default)]
pub struct LensEffectsConfig {
    /// Optional chromatic aberration step.
    pub chromatic_aberration: Option<ChromAberrationConfig>,
    /// Optional barrel/pincushion distortion step.
    pub distortion: Option<DistortionConfig>,
    /// Optional vignetting step.
    pub vignette: Option<LensVignetteConfig>,
}

/// Summary statistics for the lens effects applied to an image.
#[derive(Debug, Clone, PartialEq)]
pub struct LensEffectStats {
    /// Mean pixel displacement (in pixels) caused by chromatic aberration.
    pub mean_shift_pixels: f32,
    /// Maximum pixel displacement caused by chromatic aberration.
    pub max_shift_pixels: f32,
    /// Vignette factor at the image centre (≈ 1.0 with default settings).
    pub vignette_factor_center: f32,
    /// Vignette factor at the image corner `(1, 1)` in normalised coords.
    pub vignette_factor_corner: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Bilinear-sample the RGB channels of a u8 RGBA image at fractional pixel
/// coordinates, clamping to the image boundary.
///
/// Returns `[r, g, b]` as `f32` in `[0, 255]`.
fn bilinear_sample_f32(img: &[u8], width: u32, height: u32, fx: f32, fy: f32) -> [f32; 3] {
    if img.is_empty() || width == 0 || height == 0 {
        return [0.0; 3];
    }

    let xc = fx.clamp(0.0, (width - 1) as f32);
    let yc = fy.clamp(0.0, (height - 1) as f32);

    let x0 = xc.floor() as usize;
    let y0 = yc.floor() as usize;
    let x1 = (x0 + 1).min((width - 1) as usize);
    let y1 = (y0 + 1).min((height - 1) as usize);

    let tx = xc - x0 as f32;
    let ty = yc - y0 as f32;
    let w = width as usize;

    let sample = |row: usize, col: usize| -> [f32; 3] {
        let base = (row * w + col) * 4;
        [
            img.get(base).copied().unwrap_or(0) as f32,
            img.get(base + 1).copied().unwrap_or(0) as f32,
            img.get(base + 2).copied().unwrap_or(0) as f32,
        ]
    };

    let p00 = sample(y0, x0);
    let p10 = sample(y0, x1);
    let p01 = sample(y1, x0);
    let p11 = sample(y1, x1);

    let mut out = [0.0_f32; 3];
    for i in 0..3 {
        let top = p00[i] * (1.0 - tx) + p10[i] * tx;
        let bot = p01[i] * (1.0 - tx) + p11[i] * tx;
        out[i] = top * (1.0 - ty) + bot * ty;
    }
    out
}

/// Convert a pixel coordinate to the `[-1, 1]` normalised space.
#[inline]
fn normalized_coords(px: u32, py: u32, width: u32, height: u32) -> (f32, f32) {
    let nx = (px as f32 + 0.5) / width as f32 * 2.0 - 1.0;
    let ny = (py as f32 + 0.5) / height as f32 * 2.0 - 1.0;
    (nx, ny)
}

/// Inverse of [`normalized_coords`]: convert from `[-1, 1]` back to pixel space.
#[inline]
pub(crate) fn normalized_to_pixel(nx: f32, ny: f32, width: u32, height: u32) -> (f32, f32) {
    let px = (nx + 1.0) / 2.0 * width as f32 - 0.5;
    let py = (ny + 1.0) / 2.0 * height as f32 - 0.5;
    (px, py)
}

/// Compute a single-channel radial-shift plane from a u8 RGBA image.
///
/// Returns a `width * height` `Vec<f32>` with values in `[0, 255]`.
fn radial_shift_channel(
    img: &[u8],
    width: u32,
    height: u32,
    shift_factor: f32,
    channel: usize,
) -> Vec<f32> {
    let n = (width * height) as usize;
    let mut out = vec![0.0f32; n];

    for py in 0..height {
        for px in 0..width {
            let (nx, ny) = normalized_coords(px, py, width, height);
            let r2 = nx * nx + ny * ny;
            let scale = 1.0 + shift_factor * r2;
            let src_nx = nx * scale;
            let src_ny = ny * scale;
            let (src_px, src_py) = normalized_to_pixel(src_nx, src_ny, width, height);

            let rgb = bilinear_sample_f32(img, width, height, src_px, src_py);
            let idx = (py * width + px) as usize;
            if let Some(slot) = out.get_mut(idx) {
                *slot = rgb.get(channel).copied().unwrap_or(0.0);
            }
        }
    }

    out
}

/// Compute distorted source coordinates from normalised output coordinates.
///
/// Applies the Brown–Conrady radial model:
/// `radial_factor = 1 + k1·r² + k2·r⁴`.
#[inline]
pub(crate) fn apply_distortion_coords(nx: f32, ny: f32, k1: f32, k2: f32) -> (f32, f32) {
    let r2 = nx * nx + ny * ny;
    let rf = 1.0 + k1 * r2 + k2 * r2 * r2;
    (nx * rf, ny * rf)
}

/// Compute the vignette factor `∈ [0, 1]` for a normalised coordinate.
#[inline]
pub(crate) fn compute_vignette_factor(nx: f32, ny: f32, config: &LensVignetteConfig) -> f32 {
    let r = (nx * nx + ny * ny).sqrt();
    if r <= config.radius {
        return 1.0;
    }
    let outer = (config.radius - 1.0).abs().max(1e-6);
    let t = (r - config.radius) / outer;
    (1.0 - config.strength * t.powf(config.falloff)).clamp(0.0, 1.0)
}

/// Validate that the RGBA buffer matches the declared dimensions.
fn validate_rgba_buffer(img: &[u8], width: u32, height: u32) -> Result<(), LensEffectError> {
    if img.is_empty() || width == 0 || height == 0 {
        return Err(LensEffectError::EmptyImage);
    }
    let expected = (width as usize) * (height as usize) * 4;
    if img.len() != expected {
        return Err(LensEffectError::InvalidDimensions {
            actual: img.len(),
            expected,
            width,
            height,
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Public functions
// ─────────────────────────────────────────────────────────────────────────────

/// Apply per-channel radial chromatic aberration to a u8 RGBA image.
///
/// # Errors
///
/// - [`LensEffectError::EmptyImage`] for zero-dimension images.
/// - [`LensEffectError::InvalidDimensions`] if the buffer length mismatches.
/// - [`LensEffectError::InvalidAberrationStrength`] if `config.strength ∉ [0, 1]`.
pub fn apply_chromatic_aberration_rgba(
    image_rgba: &[u8],
    width: u32,
    height: u32,
    config: &ChromAberrationConfig,
) -> Result<Vec<u8>, LensEffectError> {
    validate_rgba_buffer(image_rgba, width, height)?;
    config.validate()?;

    let eff_r = config.red_shift * config.strength;
    let eff_g = config.green_shift * config.strength;
    let eff_b = config.blue_shift * config.strength;

    let r_plane = radial_shift_channel(image_rgba, width, height, eff_r, 0);
    let g_plane = radial_shift_channel(image_rgba, width, height, eff_g, 1);
    let b_plane = radial_shift_channel(image_rgba, width, height, eff_b, 2);

    let n = (width * height) as usize;
    let mut out = vec![0u8; n * 4];

    for i in 0..n {
        let r = r_plane.get(i).copied().unwrap_or(0.0).clamp(0.0, 255.0) as u8;
        let g = g_plane.get(i).copied().unwrap_or(0.0).clamp(0.0, 255.0) as u8;
        let b = b_plane.get(i).copied().unwrap_or(0.0).clamp(0.0, 255.0) as u8;
        let a = image_rgba.get(i * 4 + 3).copied().unwrap_or(255);
        if let Some(dst) = out.get_mut(i * 4..i * 4 + 4) {
            dst[0] = r;
            dst[1] = g;
            dst[2] = b;
            dst[3] = a;
        }
    }

    Ok(out)
}

/// Apply barrel/pincushion distortion to a u8 RGBA image.
///
/// # Errors
///
/// - [`LensEffectError::EmptyImage`] for zero-dimension images.
/// - [`LensEffectError::InvalidDimensions`] if the buffer length mismatches.
/// - [`LensEffectError::InvalidDistortionCoeff`] /
///   [`LensEffectError::InvalidAberrationStrength`] for out-of-range config values.
pub fn apply_barrel_distortion(
    image_rgba: &[u8],
    width: u32,
    height: u32,
    config: &DistortionConfig,
) -> Result<Vec<u8>, LensEffectError> {
    validate_rgba_buffer(image_rgba, width, height)?;
    config.validate()?;

    let k1 = config.k1 * config.strength;
    let k2 = config.k2 * config.strength;

    let n = (width * height) as usize;
    let mut out = vec![0u8; n * 4];

    for py in 0..height {
        for px in 0..width {
            let (nx, ny) = normalized_coords(px, py, width, height);
            let (src_nx, src_ny) = apply_distortion_coords(nx, ny, k1, k2);
            let (src_px, src_py) = normalized_to_pixel(src_nx, src_ny, width, height);

            let rgb = bilinear_sample_f32(image_rgba, width, height, src_px, src_py);
            let a = {
                let ax = src_px.clamp(0.0, (width - 1) as f32) as usize;
                let ay = src_py.clamp(0.0, (height - 1) as f32) as usize;
                image_rgba
                    .get((ay * width as usize + ax) * 4 + 3)
                    .copied()
                    .unwrap_or(255)
            };

            let idx = (py * width + px) as usize;
            if let Some(dst) = out.get_mut(idx * 4..idx * 4 + 4) {
                dst[0] = rgb[0].clamp(0.0, 255.0) as u8;
                dst[1] = rgb[1].clamp(0.0, 255.0) as u8;
                dst[2] = rgb[2].clamp(0.0, 255.0) as u8;
                dst[3] = a;
            }
        }
    }

    Ok(out)
}

/// Apply vignetting to a u8 RGBA image.
///
/// Alpha is preserved unchanged.
///
/// # Errors
///
/// - [`LensEffectError::EmptyImage`] for zero-dimension images.
/// - [`LensEffectError::InvalidDimensions`] if the buffer length mismatches.
/// - [`LensEffectError::InvalidAberrationStrength`] if `strength ∉ [0, 1]`.
pub fn apply_vignette_effect(
    image_rgba: &[u8],
    width: u32,
    height: u32,
    config: &LensVignetteConfig,
) -> Result<Vec<u8>, LensEffectError> {
    validate_rgba_buffer(image_rgba, width, height)?;
    config.validate()?;

    let n = (width * height) as usize;
    let mut out = vec![0u8; n * 4];

    for py in 0..height {
        for px in 0..width {
            let (nx, ny) = normalized_coords(px, py, width, height);
            let factor = compute_vignette_factor(nx, ny, config);
            let inv = 1.0 - factor;

            let idx = (py * width + px) as usize;
            let base = idx * 4;

            for ch in 0..3 {
                let src = image_rgba.get(base + ch).copied().unwrap_or(0) as f32;
                let color_val = config.color.get(ch).copied().unwrap_or(0.0) * 255.0;
                let blended = (factor * src + inv * color_val).clamp(0.0, 255.0) as u8;
                if let Some(dst) = out.get_mut(base + ch) {
                    *dst = blended;
                }
            }
            if let Some(dst) = out.get_mut(base + 3) {
                *dst = image_rgba.get(base + 3).copied().unwrap_or(255);
            }
        }
    }

    Ok(out)
}

/// Apply all configured lens effects in order: aberration → distortion → vignette.
///
/// Each step is optional.  If a step's `Option` is `None` the image passes
/// through unchanged for that step.
///
/// # Errors
///
/// Propagates errors from any of the individual effect functions.
pub fn apply_lens_effects(
    image_rgba: &[u8],
    width: u32,
    height: u32,
    config: &LensEffectsConfig,
) -> Result<Vec<u8>, LensEffectError> {
    validate_rgba_buffer(image_rgba, width, height)?;

    let mut current: Vec<u8> = image_rgba.to_vec();

    if let Some(ref ca) = config.chromatic_aberration {
        current = apply_chromatic_aberration_rgba(&current, width, height, ca)?;
    }
    if let Some(ref dist) = config.distortion {
        current = apply_barrel_distortion(&current, width, height, dist)?;
    }
    if let Some(ref vig) = config.vignette {
        current = apply_vignette_effect(&current, width, height, vig)?;
    }

    Ok(current)
}

/// Compute summary statistics for the requested lens effects without modifying
/// an image.
///
/// Pass `None` for any effect whose statistics should be skipped; those fields
/// will be zero or one as appropriate.
pub fn compute_lens_effect_stats(
    width: u32,
    height: u32,
    ca_config: Option<&ChromAberrationConfig>,
    vignette_config: Option<&LensVignetteConfig>,
) -> LensEffectStats {
    let (mean_shift, max_shift) = if let Some(ca) = ca_config {
        let max_sf = ca.red_shift.abs().max(ca.blue_shift.abs()) * ca.strength;
        let half_diag = ((width * width + height * height) as f32).sqrt() * 0.5;
        let mut sum = 0.0f32;
        let mut max_s = 0.0f32;
        let n = width * height;

        for py in 0..height {
            for px in 0..width {
                let (nx, ny) = normalized_coords(px, py, width, height);
                let r2 = nx * nx + ny * ny;
                let shift_px = max_sf * r2 * half_diag;
                sum += shift_px;
                if shift_px > max_s {
                    max_s = shift_px;
                }
            }
        }

        let mean = if n > 0 { sum / n as f32 } else { 0.0 };
        (mean, max_s)
    } else {
        (0.0, 0.0)
    };

    let (vf_center, vf_corner) = if let Some(vc) = vignette_config {
        (
            compute_vignette_factor(0.0, 0.0, vc),
            compute_vignette_factor(1.0, 1.0, vc),
        )
    } else {
        (1.0, 1.0)
    };

    LensEffectStats {
        mean_shift_pixels: mean_shift,
        max_shift_pixels: max_shift,
        vignette_factor_center: vf_center,
        vignette_factor_corner: vf_corner,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let n = (w * h) as usize;
        let mut img = vec![0u8; n * 4];
        for i in 0..n {
            img[i * 4] = r;
            img[i * 4 + 1] = g;
            img[i * 4 + 2] = b;
            img[i * 4 + 3] = a;
        }
        img
    }

    // ── 1. ChromAberrationConfig default values ────────────────────────────────

    #[test]
    fn test_chrom_aberration_config_defaults() {
        let cfg = ChromAberrationConfig::default();
        assert!((cfg.red_shift - 0.005).abs() < 1e-6);
        assert!((cfg.green_shift - 0.0).abs() < 1e-6);
        assert!((cfg.blue_shift + 0.005).abs() < 1e-6);
        assert!((cfg.strength - 1.0).abs() < 1e-6);
    }

    // ── 2. ChromAberrationConfig::validate: strength out of range ─────────────

    #[test]
    fn test_chrom_aberration_config_validate_bad_strength() {
        let cfg = ChromAberrationConfig {
            strength: 1.1,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(LensEffectError::InvalidAberrationStrength { .. })
        ));
        let cfg2 = ChromAberrationConfig {
            strength: -0.1,
            ..Default::default()
        };
        assert!(cfg2.validate().is_err());
    }

    // ── 3. ChromAberrationConfig::validate: valid strength ────────────────────

    #[test]
    fn test_chrom_aberration_config_validate_ok() {
        assert!(ChromAberrationConfig::default().validate().is_ok());
    }

    // ── 4. DistortionConfig default values ────────────────────────────────────

    #[test]
    fn test_distortion_config_defaults() {
        let cfg = DistortionConfig::default();
        assert!((cfg.k1 - 0.0).abs() < 1e-6);
        assert!((cfg.k2 - 0.0).abs() < 1e-6);
        assert!((cfg.strength - 1.0).abs() < 1e-6);
    }

    // ── 5. DistortionConfig::validate: k1 out of range → error ───────────────

    #[test]
    fn test_distortion_config_validate_bad_k1() {
        let cfg = DistortionConfig {
            k1: -1.5,
            ..DistortionConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(LensEffectError::InvalidDistortionCoeff { .. })
        ));
    }

    // ── 6. DistortionConfig::validate: k2 out of range → error ───────────────

    #[test]
    fn test_distortion_config_validate_bad_k2() {
        let cfg = DistortionConfig {
            k2: 1.5,
            ..DistortionConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(LensEffectError::InvalidDistortionCoeff { .. })
        ));
    }

    // ── 7. DistortionConfig::validate: strength out of range → error ──────────

    #[test]
    fn test_distortion_config_validate_bad_strength() {
        let cfg = DistortionConfig {
            strength: 2.0,
            ..DistortionConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    // ── 8. LensVignetteConfig default values ──────────────────────────────────

    #[test]
    fn test_lens_vignette_config_defaults() {
        let cfg = LensVignetteConfig::default();
        assert!((cfg.strength - 0.5).abs() < 1e-6);
        assert!((cfg.radius - 0.7).abs() < 1e-6);
        assert!((cfg.falloff - 2.0).abs() < 1e-6);
        assert_eq!(cfg.color, [0.0, 0.0, 0.0]);
    }

    // ── 9. LensVignetteConfig::validate: invalid strength → error ─────────────

    #[test]
    fn test_lens_vignette_config_validate_bad_strength() {
        let cfg = LensVignetteConfig {
            strength: -0.1,
            ..LensVignetteConfig::default()
        };
        assert!(cfg.validate().is_err());
        let cfg2 = LensVignetteConfig {
            strength: 1.5,
            ..LensVignetteConfig::default()
        };
        assert!(cfg2.validate().is_err());
    }

    // ── 10. normalized_coords: centre of 256×256 → near (0, 0) ───────────────

    #[test]
    fn test_normalized_coords_center() {
        let (nx, ny) = normalized_coords(127, 127, 256, 256);
        assert!(nx.abs() < 0.01, "Expected nx ≈ 0, got {nx}");
        assert!(ny.abs() < 0.01, "Expected ny ≈ 0, got {ny}");
    }

    // ── 11. normalized_to_pixel: inverse of normalized_coords ─────────────────

    #[test]
    fn test_normalized_to_pixel_inverse() {
        for &(px, py, w, h) in &[(3u32, 5u32, 16u32, 16u32), (0, 0, 8, 8), (7, 3, 10, 10)] {
            let (nx, ny) = normalized_coords(px, py, w, h);
            let (rx, ry) = normalized_to_pixel(nx, ny, w, h);
            assert!(
                (rx - px as f32).abs() < 1e-4,
                "px round-trip: {px} → {nx} → {rx}"
            );
            assert!(
                (ry - py as f32).abs() < 1e-4,
                "py round-trip: {py} → {ny} → {ry}"
            );
        }
    }

    // ── 12. apply_distortion_coords: k1=0, k2=0 → identity ───────────────────

    #[test]
    fn test_apply_distortion_coords_identity() {
        let (ox, oy) = apply_distortion_coords(0.3, -0.4, 0.0, 0.0);
        assert!((ox - 0.3).abs() < 1e-6);
        assert!((oy + 0.4).abs() < 1e-6);
    }

    // ── 13. apply_distortion_coords: negative k1 → barrel ────────────────────

    #[test]
    fn test_apply_distortion_coords_barrel() {
        let (ox, _) = apply_distortion_coords(0.5, 0.0, -0.5, 0.0);
        assert!(
            ox < 0.5,
            "Barrel: distorted x should be less than original: {ox}"
        );
    }

    // ── 14. compute_vignette_factor: centre → 1.0 ─────────────────────────────

    #[test]
    fn test_compute_vignette_factor_center() {
        let cfg = LensVignetteConfig::default();
        let f = compute_vignette_factor(0.0, 0.0, &cfg);
        assert!((f - 1.0).abs() < 1e-6, "Expected 1.0 at centre, got {f}");
    }

    // ── 15. compute_vignette_factor: far corner → reduced ─────────────────────

    #[test]
    fn test_compute_vignette_factor_corner_reduced() {
        let cfg = LensVignetteConfig::default();
        let corner = compute_vignette_factor(1.0, 1.0, &cfg);
        assert!(
            corner < 1.0,
            "Vignette at corner should be < 1.0, got {corner}"
        );
        assert!(corner >= 0.0);
    }

    // ── 16. bilinear_sample_f32: in-bounds → correct value ────────────────────

    #[test]
    fn test_bilinear_sample_f32_in_bounds() {
        let mut img = vec![0u8; 4 * 4]; // 2×2 RGBA
        img[0] = 200;
        img[1] = 100;
        img[2] = 50;
        img[3] = 255;
        let rgb = bilinear_sample_f32(&img, 2, 2, 0.0, 0.0);
        assert!((rgb[0] - 200.0).abs() < 1.0);
        assert!((rgb[1] - 100.0).abs() < 1.0);
        assert!((rgb[2] - 50.0).abs() < 1.0);
    }

    // ── 17. bilinear_sample_f32: out-of-bounds clamped ────────────────────────

    #[test]
    fn test_bilinear_sample_f32_clamped() {
        let img = solid_rgba(4, 4, 128, 200, 60, 255);
        let rgb = bilinear_sample_f32(&img, 4, 4, 100.0, 100.0);
        assert!((rgb[0] - 128.0).abs() < 1.0);
    }

    // ── 18. radial_shift_channel: zero shift → near-identity ─────────────────

    #[test]
    fn test_radial_shift_channel_zero_shift() {
        let img = solid_rgba(8, 8, 100, 150, 200, 255);
        let plane = radial_shift_channel(&img, 8, 8, 0.0, 0);
        for &v in &plane {
            assert!(
                (v - 100.0).abs() < 1.0,
                "Zero shift should preserve R values: {v}"
            );
        }
    }

    // ── 19. apply_chromatic_aberration_rgba: output size correct ──────────────

    #[test]
    fn test_apply_chromatic_aberration_rgba_output_size() {
        let img = solid_rgba(8, 6, 100, 120, 80, 255);
        let cfg = ChromAberrationConfig::default();
        let out =
            apply_chromatic_aberration_rgba(&img, 8, 6, &cfg).expect("chromatic aberration failed");
        assert_eq!(out.len(), 8 * 6 * 4);
    }

    // ── 20. apply_chromatic_aberration_rgba: zero shifts → identical ──────────

    #[test]
    fn test_apply_chromatic_aberration_rgba_zero_shifts_identity() {
        let img = solid_rgba(8, 8, 120, 80, 200, 255);
        let cfg = ChromAberrationConfig {
            red_shift: 0.0,
            green_shift: 0.0,
            blue_shift: 0.0,
            strength: 1.0,
        };
        let out = apply_chromatic_aberration_rgba(&img, 8, 8, &cfg).expect("CA failed");
        for i in 0..(8 * 8) {
            assert_eq!(img[i * 4], out[i * 4]);
            assert_eq!(img[i * 4 + 1], out[i * 4 + 1]);
            assert_eq!(img[i * 4 + 2], out[i * 4 + 2]);
        }
    }

    // ── 21. apply_chromatic_aberration_rgba: empty image → error ─────────────

    #[test]
    fn test_apply_chromatic_aberration_rgba_empty_error() {
        let cfg = ChromAberrationConfig::default();
        let result = apply_chromatic_aberration_rgba(&[], 0, 0, &cfg);
        assert!(matches!(result, Err(LensEffectError::EmptyImage)));
    }

    // ── 22. apply_chromatic_aberration_rgba: alpha channel preserved ──────────

    #[test]
    fn test_chromatic_aberration_rgba_alpha_preserved() {
        let img = solid_rgba(8, 8, 100, 150, 200, 77);
        let cfg = ChromAberrationConfig::moderate();
        let out = apply_chromatic_aberration_rgba(&img, 8, 8, &cfg).expect("CA failed");
        for i in 0..(8 * 8) {
            assert_eq!(out[i * 4 + 3], 77, "Alpha must equal 77 at pixel {i}");
        }
    }

    // ── 23. apply_barrel_distortion: output size correct ─────────────────────

    #[test]
    fn test_apply_barrel_distortion_output_size() {
        let img = solid_rgba(6, 4, 100, 100, 100, 255);
        let cfg = DistortionConfig::default();
        let out = apply_barrel_distortion(&img, 6, 4, &cfg).expect("barrel distortion failed");
        assert_eq!(out.len(), 6 * 4 * 4);
    }

    // ── 24. apply_barrel_distortion: k1=0 → near-identity ────────────────────

    #[test]
    fn test_apply_barrel_distortion_identity() {
        let img = solid_rgba(8, 8, 100, 150, 200, 255);
        let cfg = DistortionConfig::default();
        let out = apply_barrel_distortion(&img, 8, 8, &cfg).expect("barrel distortion failed");
        for i in 0..(8 * 8) {
            assert!((img[i * 4] as i32 - out[i * 4] as i32).abs() <= 1);
        }
    }

    // ── 25. apply_barrel_distortion: invalid buffer → error ──────────────────

    #[test]
    fn test_apply_barrel_distortion_invalid_buffer() {
        let bad = vec![0u8; 5];
        let cfg = DistortionConfig::default();
        let result = apply_barrel_distortion(&bad, 2, 2, &cfg);
        assert!(matches!(
            result,
            Err(LensEffectError::InvalidDimensions { .. })
        ));
    }

    // ── 26. apply_vignette_effect: output size correct ────────────────────────

    #[test]
    fn test_apply_vignette_effect_output_size() {
        let img = solid_rgba(10, 8, 200, 200, 200, 255);
        let cfg = LensVignetteConfig::default();
        let out = apply_vignette_effect(&img, 10, 8, &cfg).expect("vignette failed");
        assert_eq!(out.len(), 10 * 8 * 4);
    }

    // ── 27. apply_vignette_effect: strength=0 → identical output ─────────────

    #[test]
    fn test_apply_vignette_effect_zero_strength_identity() {
        let img = solid_rgba(8, 8, 200, 100, 50, 255);
        let cfg = LensVignetteConfig {
            strength: 0.0,
            ..LensVignetteConfig::default()
        };
        let out = apply_vignette_effect(&img, 8, 8, &cfg).expect("vignette failed");
        assert_eq!(img, out);
    }

    // ── 28. apply_vignette_effect: centre brighter than corner ───────────────

    #[test]
    fn test_apply_vignette_effect_center_vs_corner() {
        let w = 16u32;
        let h = 16u32;
        let img = solid_rgba(w, h, 200, 200, 200, 255);
        let cfg = LensVignetteConfig {
            strength: 0.8,
            radius: 0.3,
            ..LensVignetteConfig::default()
        };
        let out = apply_vignette_effect(&img, w, h, &cfg).expect("vignette failed");
        let cx = (h / 2 * w + w / 2) as usize;
        let center_r = out[cx * 4] as f32;
        let corner_r = out[0] as f32;
        assert!(
            center_r >= corner_r,
            "Centre ({center_r}) should be >= corner ({corner_r})"
        );
    }

    // ── 29. apply_vignette_effect: alpha preserved ────────────────────────────

    #[test]
    fn test_apply_vignette_effect_alpha_preserved() {
        let img = solid_rgba(4, 4, 128, 128, 128, 200);
        let cfg = LensVignetteConfig::default();
        let out = apply_vignette_effect(&img, 4, 4, &cfg).expect("vignette failed");
        for i in 0..(4 * 4) {
            assert_eq!(out[i * 4 + 3], 200, "Alpha must be preserved at pixel {i}");
        }
    }

    // ── 30. apply_vignette_effect: invalid buffer → error ────────────────────

    #[test]
    fn test_apply_vignette_effect_invalid_buffer() {
        let bad = vec![0u8; 3];
        let cfg = LensVignetteConfig::default();
        let result = apply_vignette_effect(&bad, 1, 1, &cfg);
        assert!(matches!(
            result,
            Err(LensEffectError::InvalidDimensions { .. })
        ));
    }

    // ── 31. apply_lens_effects: all None → pass-through ──────────────────────

    #[test]
    fn test_apply_lens_effects_all_none_passthrough() {
        let img = solid_rgba(8, 8, 100, 150, 200, 255);
        let cfg = LensEffectsConfig::default();
        let out = apply_lens_effects(&img, 8, 8, &cfg).expect("lens effects failed");
        assert_eq!(img, out);
    }

    // ── 32. apply_lens_effects: all Some → applies effects ───────────────────

    #[test]
    fn test_apply_lens_effects_all_some_applies() {
        let img = solid_rgba(8, 8, 100, 150, 200, 255);
        let cfg = LensEffectsConfig {
            chromatic_aberration: Some(ChromAberrationConfig::default()),
            distortion: Some(DistortionConfig::default()),
            vignette: Some(LensVignetteConfig::default()),
        };
        let out = apply_lens_effects(&img, 8, 8, &cfg).expect("lens effects failed");
        assert_eq!(out.len(), 8 * 8 * 4);
    }

    // ── 33. apply_lens_effects: empty buffer → error ──────────────────────────

    #[test]
    fn test_apply_lens_effects_empty_buffer_error() {
        let cfg = LensEffectsConfig::default();
        let result = apply_lens_effects(&[], 0, 0, &cfg);
        assert!(matches!(result, Err(LensEffectError::EmptyImage)));
    }

    // ── 34. compute_lens_effect_stats: vignette centre = 1.0 ─────────────────

    #[test]
    fn test_compute_lens_effect_stats_vignette_center() {
        let vc = LensVignetteConfig::default();
        let stats = compute_lens_effect_stats(128, 128, None, Some(&vc));
        assert!(
            (stats.vignette_factor_center - 1.0).abs() < 1e-5,
            "Centre factor should be 1.0: {}",
            stats.vignette_factor_center
        );
    }

    // ── 35. compute_lens_effect_stats: vignette corner < centre ──────────────

    #[test]
    fn test_compute_lens_effect_stats_vignette_corner_less_than_center() {
        let vc = LensVignetteConfig {
            strength: 0.8,
            radius: 0.3,
            ..LensVignetteConfig::default()
        };
        let stats = compute_lens_effect_stats(64, 64, None, Some(&vc));
        assert!(
            stats.vignette_factor_corner < stats.vignette_factor_center,
            "Corner ({}) should be < centre ({})",
            stats.vignette_factor_corner,
            stats.vignette_factor_center
        );
    }

    // ── 36. compute_lens_effect_stats: no effects → zeros/ones ───────────────

    #[test]
    fn test_compute_lens_effect_stats_none_effects() {
        let stats = compute_lens_effect_stats(64, 64, None, None);
        assert!((stats.mean_shift_pixels - 0.0).abs() < 1e-6);
        assert!((stats.max_shift_pixels - 0.0).abs() < 1e-6);
        assert!((stats.vignette_factor_center - 1.0).abs() < 1e-6);
        assert!((stats.vignette_factor_corner - 1.0).abs() < 1e-6);
    }

    // ── 37. compute_lens_effect_stats: CA gives positive shift ────────────────

    #[test]
    fn test_compute_lens_effect_stats_ca_positive_shift() {
        let ca = ChromAberrationConfig::strong();
        let stats = compute_lens_effect_stats(128, 128, Some(&ca), None);
        assert!(
            stats.mean_shift_pixels > 0.0,
            "Mean shift should be positive: {}",
            stats.mean_shift_pixels
        );
        assert!(stats.max_shift_pixels >= stats.mean_shift_pixels);
    }

    // ── 38. ChromAberrationConfig presets: ordering ───────────────────────────

    #[test]
    fn test_chrom_aberration_presets_ordering() {
        let subtle = ChromAberrationConfig::subtle();
        let moderate = ChromAberrationConfig::moderate();
        let strong = ChromAberrationConfig::strong();
        assert!(subtle.red_shift < moderate.red_shift);
        assert!(moderate.red_shift < strong.red_shift);
        assert!(subtle.blue_shift > moderate.blue_shift);
        assert!(moderate.blue_shift > strong.blue_shift);
    }

    // ── 39. DistortionConfig presets: barrel vs pincushion signs ─────────────

    #[test]
    fn test_distortion_presets_signs() {
        let barrel = DistortionConfig::slight_barrel();
        let pincushion = DistortionConfig::slight_pincushion();
        assert!(barrel.k1 < 0.0, "Barrel k1 should be negative");
        assert!(pincushion.k1 > 0.0, "Pincushion k1 should be positive");
    }

    // ── 40. LensVignetteConfig presets: strength ordering ────────────────────

    #[test]
    fn test_lens_vignette_presets_strength_ordering() {
        let subtle = LensVignetteConfig::subtle();
        let cinema = LensVignetteConfig::cinema();
        assert!(subtle.strength < cinema.strength);
    }

    // ── 41. LensEffectError variants: display messages ────────────────────────

    #[test]
    fn test_lens_effect_error_display() {
        let e1 = LensEffectError::EmptyImage;
        assert!(!e1.to_string().is_empty());

        let e2 = LensEffectError::InvalidDimensions {
            actual: 10,
            expected: 16,
            width: 2,
            height: 2,
        };
        assert!(e2.to_string().contains("10"));

        let e3 = LensEffectError::InvalidAberrationStrength { strength: 1.5 };
        assert!(e3.to_string().contains("1.5"));

        let e4 = LensEffectError::InvalidDistortionCoeff { k: -2.0 };
        assert!(e4.to_string().contains("-2"));
    }

    // ── 42. LensEffectsConfig fields accessible ───────────────────────────────

    #[test]
    fn test_lens_effects_config_fields() {
        let cfg = LensEffectsConfig {
            chromatic_aberration: Some(ChromAberrationConfig::default()),
            distortion: None,
            vignette: None,
        };
        assert!(cfg.chromatic_aberration.is_some());
        assert!(cfg.distortion.is_none());
    }

    // ── 43. 4×4 chromatic aberration smoke test ───────────────────────────────

    #[test]
    fn test_4x4_chromatic_aberration_smoke() {
        let img = solid_rgba(4, 4, 200, 100, 50, 255);
        let cfg = ChromAberrationConfig {
            strength: 0.5,
            ..ChromAberrationConfig::default()
        };
        let out = apply_chromatic_aberration_rgba(&img, 4, 4, &cfg).expect("4×4 CA failed");
        assert_eq!(out.len(), 4 * 4 * 4);
        for i in 0..(4 * 4) {
            assert_eq!(out[i * 4 + 3], 255);
        }
    }

    // ── 44. 4×4 barrel distortion smoke test ─────────────────────────────────

    #[test]
    fn test_4x4_barrel_distortion_smoke() {
        let img = solid_rgba(4, 4, 128, 64, 32, 128);
        let cfg = DistortionConfig::slight_barrel();
        let out = apply_barrel_distortion(&img, 4, 4, &cfg).expect("4×4 barrel failed");
        assert_eq!(out.len(), 4 * 4 * 4);
    }

    // ── 45. 4×4 vignette smoke test ───────────────────────────────────────────

    #[test]
    fn test_4x4_vignette_smoke() {
        let img = solid_rgba(4, 4, 200, 200, 200, 255);
        let cfg = LensVignetteConfig::cinema();
        let out = apply_vignette_effect(&img, 4, 4, &cfg).expect("4×4 vignette failed");
        assert_eq!(out.len(), 4 * 4 * 4);
    }

    // ── 46. LensEffectStats fields accessible ─────────────────────────────────

    #[test]
    fn test_lens_effect_stats_fields() {
        let stats = LensEffectStats {
            mean_shift_pixels: 1.5,
            max_shift_pixels: 3.0,
            vignette_factor_center: 1.0,
            vignette_factor_corner: 0.6,
        };
        assert!((stats.mean_shift_pixels - 1.5).abs() < 1e-6);
        assert!((stats.max_shift_pixels - 3.0).abs() < 1e-6);
        assert!((stats.vignette_factor_center - 1.0).abs() < 1e-6);
        assert!((stats.vignette_factor_corner - 0.6).abs() < 1e-5);
    }

    // ── 47. LensVignetteConfig::validate: valid strength ─────────────────────

    #[test]
    fn test_lens_vignette_config_validate_ok() {
        assert!(LensVignetteConfig::default().validate().is_ok());
        assert!(LensVignetteConfig::subtle().validate().is_ok());
        assert!(LensVignetteConfig::cinema().validate().is_ok());
    }

    // ── 48. DistortionConfig::validate: all valid → ok ────────────────────────

    #[test]
    fn test_distortion_config_validate_ok() {
        assert!(DistortionConfig::default().validate().is_ok());
        assert!(DistortionConfig::slight_barrel().validate().is_ok());
        assert!(DistortionConfig::slight_pincushion().validate().is_ok());
    }

    // ── 49. ChromAberrationConfig: presets validate successfully ─────────────

    #[test]
    fn test_chrom_aberration_presets_validate_ok() {
        assert!(ChromAberrationConfig::subtle().validate().is_ok());
        assert!(ChromAberrationConfig::moderate().validate().is_ok());
        assert!(ChromAberrationConfig::strong().validate().is_ok());
    }

    // ── 50. apply_lens_effects: only vignette applied ────────────────────────

    #[test]
    fn test_apply_lens_effects_only_vignette() {
        let img = solid_rgba(8, 8, 200, 200, 200, 255);
        let cfg = LensEffectsConfig {
            chromatic_aberration: None,
            distortion: None,
            vignette: Some(LensVignetteConfig {
                strength: 1.0,
                radius: 0.0,
                ..LensVignetteConfig::default()
            }),
        };
        let out = apply_lens_effects(&img, 8, 8, &cfg).expect("lens effects failed");
        // Corner pixel should be fully darkened.
        assert!(
            out[0] < 200,
            "Corner should be darkened by vignette, got {}",
            out[0]
        );
    }
}
