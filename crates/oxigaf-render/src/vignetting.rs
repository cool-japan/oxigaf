//! Photographic vignetting effects for 3DGS rendered images.
//!
//! Vignetting is a lens effect where illumination falls off toward the corners
//! and edges of an image, creating the natural "darkening around the edges" look
//! common in photography. Adding vignetting to 3DGS renders gives a photographic
//! quality.
//!
//! # Models
//!
//! - [`VignettingModel::PowerLaw`]: Simple radial falloff using a power function.
//! - [`VignettingModel::Cosine4`]: Natural vignetting from cos⁴(angle) optics law.
//! - [`VignettingModel::Polynomial`]: Lens distortion polynomial (k1, k2, k3).
//! - [`VignettingModel::Gaussian`]: Gaussian radial falloff.
//!
//! # Usage
//!
//! ```rust,no_run
//! use oxigaf_render::vignetting::{VignettingConfig, apply_vignetting};
//!
//! let width = 640;
//! let height = 480;
//! let image = vec![1.0_f32; width * height * 3];
//! let config = VignettingConfig::cinematic();
//! let vignetted = apply_vignetting(&image, width, height, &config).unwrap();
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by vignetting operations.
#[derive(Debug, Error, PartialEq)]
pub enum VignettingError {
    /// A configuration parameter is invalid.
    #[error("Invalid vignetting config: {0}")]
    InvalidConfig(String),

    /// Image buffer length does not match declared dimensions.
    #[error("Invalid image: {0}")]
    InvalidImage(String),

    /// Image has zero pixels (width or height is zero).
    #[error("Image is empty (zero width or height)")]
    EmptyImage,
}

// ─────────────────────────────────────────────────────────────────────────────
// VignettingModel
// ─────────────────────────────────────────────────────────────────────────────

/// Radial falloff functions for vignetting.
///
/// Each variant computes a vignetting factor `v ∈ [0, 1]` for a normalised
/// radius `r ∈ [0, 1]`, where `r = 0` is the image centre and `r = 1` is the
/// corner of a square image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VignettingModel {
    /// Simple power-law: `v = (1 − r²)^power`.
    ///
    /// `power` must be positive. Larger values create harder edges.
    PowerLaw { power: f32 },

    /// Natural cosine⁴ vignetting law: `v = cos⁴(angle)` where
    /// `angle = r * π/4`.
    ///
    /// At `r = 1`, `angle = π/4` and `cos⁴(π/4) ≈ 0.25`, matching the
    /// photographic natural vignetting formula.
    Cosine4,

    /// Polynomial lens model: `v = 1 + k1·r² + k2·r⁴ + k3·r⁶`.
    ///
    /// All coefficients must be finite. Negative coefficients produce
    /// darkening; positive produce brightening (unusual but valid).
    Polynomial { k1: f32, k2: f32, k3: f32 },

    /// Gaussian radial falloff: `v = exp(−r² / (2σ²))`.
    ///
    /// `sigma` must be positive. Smaller sigma produces tighter falloff.
    Gaussian { sigma: f32 },
}

impl VignettingModel {
    /// Compute the vignetting factor for normalised radius `r ∈ [0, 1]`.
    ///
    /// Returns a value in `[0, 1]`: `1.0` = full brightness (centre),
    /// `0.0` = fully dark (maximum falloff).
    pub fn factor(&self, r: f32) -> f32 {
        match self {
            Self::PowerLaw { power } => {
                // v = (1 - r²)^power
                let r_clamped = r.clamp(0.0, 1.0);
                (1.0 - r_clamped.powi(2)).powf(*power).max(0.0)
            }
            Self::Cosine4 => {
                // angle = r * π/4  →  v = cos⁴(angle)
                let angle = r * std::f32::consts::FRAC_PI_4;
                angle.cos().powi(4)
            }
            Self::Polynomial { k1, k2, k3 } => {
                // v = 1 + k1*r² + k2*r⁴ + k3*r⁶
                let r2 = r * r;
                let r4 = r2 * r2;
                let r6 = r4 * r2;
                (1.0 + k1 * r2 + k2 * r4 + k3 * r6).clamp(0.0, 1.0)
            }
            Self::Gaussian { sigma } => {
                // v = exp(-r² / (2σ²))
                let s2 = sigma * sigma;
                (-(r * r) / (2.0 * s2)).exp()
            }
        }
    }

    /// Validate that the configuration is physically meaningful.
    ///
    /// # Errors
    ///
    /// - [`VignettingError::InvalidConfig`] if `power <= 0` for [`Self::PowerLaw`].
    /// - [`VignettingError::InvalidConfig`] if any coefficient is non-finite for
    ///   [`Self::Polynomial`].
    /// - [`VignettingError::InvalidConfig`] if `sigma <= 0` for [`Self::Gaussian`].
    pub fn validate(&self) -> Result<(), VignettingError> {
        match self {
            Self::PowerLaw { power } => {
                if *power <= 0.0 {
                    return Err(VignettingError::InvalidConfig(format!(
                        "PowerLaw: power must be > 0, got {}",
                        power
                    )));
                }
            }
            Self::Cosine4 => {}
            Self::Polynomial { k1, k2, k3 } => {
                if !k1.is_finite() || !k2.is_finite() || !k3.is_finite() {
                    return Err(VignettingError::InvalidConfig(format!(
                        "Polynomial: k1={k1}, k2={k2}, k3={k3} must all be finite"
                    )));
                }
            }
            Self::Gaussian { sigma } => {
                if *sigma <= 0.0 {
                    return Err(VignettingError::InvalidConfig(format!(
                        "Gaussian: sigma must be > 0, got {}",
                        sigma
                    )));
                }
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VignettingConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the vignetting effect.
#[derive(Debug, Clone)]
pub struct VignettingConfig {
    /// Radial falloff model.
    pub model: VignettingModel,

    /// Effect strength in `[0, 1]`: `0.0` = no vignetting, `1.0` = full model output.
    pub strength: f32,

    /// If `true`, use elliptical falloff to correct for non-square images.
    ///
    /// When enabled the horizontal axis is scaled by `width / height` before
    /// computing the radius, ensuring the vignette appears circular rather than
    /// elliptical on wide images.
    pub aspect_correction: bool,

    /// Horizontal center offset from image center in normalised coords `[−0.5, 0.5]`.
    pub center_x: f32,

    /// Vertical center offset from image center in normalised coords `[−0.5, 0.5]`.
    pub center_y: f32,

    /// Red channel tint value `[0, 1]`.  `0.0` = black vignette (default).
    pub tint_r: f32,

    /// Green channel tint value `[0, 1]`.  `0.0` = black vignette (default).
    pub tint_g: f32,

    /// Blue channel tint value `[0, 1]`.  `0.0` = black vignette (default).
    pub tint_b: f32,
}

impl Default for VignettingConfig {
    fn default() -> Self {
        Self {
            model: VignettingModel::PowerLaw { power: 2.0 },
            strength: 0.5,
            aspect_correction: true,
            center_x: 0.0,
            center_y: 0.0,
            tint_r: 0.0,
            tint_g: 0.0,
            tint_b: 0.0,
        }
    }
}

impl VignettingConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// - [`VignettingError::InvalidConfig`] if `strength ∉ [0, 1]`.
    /// - [`VignettingError::InvalidConfig`] if `center_x ∉ [−0.5, 0.5]`.
    /// - [`VignettingError::InvalidConfig`] if `center_y ∉ [−0.5, 0.5]`.
    /// - [`VignettingError::InvalidConfig`] propagated from [`VignettingModel::validate`].
    pub fn validate(&self) -> Result<(), VignettingError> {
        if !(0.0..=1.0).contains(&self.strength) {
            return Err(VignettingError::InvalidConfig(format!(
                "strength must be in [0, 1], got {}",
                self.strength
            )));
        }
        if !(-0.5..=0.5).contains(&self.center_x) {
            return Err(VignettingError::InvalidConfig(format!(
                "center_x must be in [-0.5, 0.5], got {}",
                self.center_x
            )));
        }
        if !(-0.5..=0.5).contains(&self.center_y) {
            return Err(VignettingError::InvalidConfig(format!(
                "center_y must be in [-0.5, 0.5], got {}",
                self.center_y
            )));
        }
        self.model.validate()
    }

    /// Cinematic preset: strong power-law vignette with a warm amber tint.
    ///
    /// Suitable for dramatic, movie-like renders.
    pub fn cinematic() -> Self {
        Self {
            model: VignettingModel::PowerLaw { power: 3.0 },
            strength: 0.7,
            aspect_correction: true,
            center_x: 0.0,
            center_y: 0.0,
            tint_r: 0.1,
            tint_g: 0.05,
            tint_b: 0.0,
        }
    }

    /// Subtle preset: gentle cosine-law vignette with no tint.
    ///
    /// Barely visible, mimics high-quality prime lens characteristics.
    pub fn subtle() -> Self {
        Self {
            model: VignettingModel::Cosine4,
            strength: 0.3,
            aspect_correction: true,
            center_x: 0.0,
            center_y: 0.0,
            tint_r: 0.0,
            tint_g: 0.0,
            tint_b: 0.0,
        }
    }

    /// Hard edge preset: tight Gaussian vignette.
    ///
    /// Produces a dramatic circular spotlight effect with fast edge rolloff.
    pub fn hard_edge() -> Self {
        Self {
            model: VignettingModel::Gaussian { sigma: 0.5 },
            strength: 0.8,
            aspect_correction: true,
            center_x: 0.0,
            center_y: 0.0,
            tint_r: 0.0,
            tint_g: 0.0,
            tint_b: 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// pixel_radius
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the normalised radius for a pixel position.
///
/// Returns `r ∈ [0, ∞)` where:
/// - `r = 0` at the (possibly offset) image centre
/// - `r ≈ 1` at the edge/corner of a square image with default settings
///
/// # Arguments
///
/// - `px`, `py`: zero-based pixel coordinates.
/// - `width`, `height`: image dimensions in pixels.
/// - `center_x`, `center_y`: normalised center offsets in `[−0.5, 0.5]`.
/// - `aspect_correction`: if `true`, the x axis is pre-scaled by
///   `width / height` so the vignette appears circular on wide images.
pub fn pixel_radius(
    px: usize,
    py: usize,
    width: usize,
    height: usize,
    center_x: f32,
    center_y: f32,
    aspect_correction: bool,
) -> f32 {
    // Normalise pixel centre into [−0.5, 0.5], then subtract the user offset.
    let mut nx = (px as f32 + 0.5) / width as f32 - 0.5 - center_x;
    let ny = (py as f32 + 0.5) / height as f32 - 0.5 - center_y;

    if aspect_correction {
        nx *= width as f32 / height as f32;
    }

    // Scale by 2 so r = 1 at the horizontal/vertical edge of a square image.
    (nx * nx + ny * ny).sqrt() * 2.0
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that an RGB image buffer has the expected length.
fn validate_rgb(image: &[f32], width: usize, height: usize) -> Result<(), VignettingError> {
    if width == 0 || height == 0 {
        return Err(VignettingError::EmptyImage);
    }
    let expected = width * height * 3;
    if image.len() != expected {
        return Err(VignettingError::InvalidImage(format!(
            "expected {expected} values for {width}×{height} RGB, got {}",
            image.len()
        )));
    }
    Ok(())
}

/// Validate that an RGBA image buffer has the expected length.
fn validate_rgba(image: &[f32], width: usize, height: usize) -> Result<(), VignettingError> {
    if width == 0 || height == 0 {
        return Err(VignettingError::EmptyImage);
    }
    let expected = width * height * 4;
    if image.len() != expected {
        return Err(VignettingError::InvalidImage(format!(
            "expected {expected} values for {width}×{height} RGBA, got {}",
            image.len()
        )));
    }
    Ok(())
}

/// Check if any tint channel is non-zero (triggers tinted blending path).
#[inline]
fn has_tint(config: &VignettingConfig) -> bool {
    config.tint_r > 0.0 || config.tint_g > 0.0 || config.tint_b > 0.0
}

/// Compute effective vignetting factor: blends full-brightness and model output
/// according to `strength`.
///
/// `strength = 0` → `effective_v = 1.0` (no change).
/// `strength = 1` → `effective_v = model.factor(r)`.
#[inline]
fn effective_factor(model: &VignettingModel, r: f32, strength: f32) -> f32 {
    let v = model.factor(r);
    1.0 - strength * (1.0 - v)
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_vignetting
// ─────────────────────────────────────────────────────────────────────────────

/// Apply vignetting to an RGB image.
///
/// The `image` slice must be row-major RGB `f32` with `len == width * height * 3`.
/// Values are expected in `[0, 1]` and the output is clamped to `[0, 1]`.
///
/// When `config.strength == 0.0` the output is identical to the input.
///
/// # Errors
///
/// - [`VignettingError::EmptyImage`] if `width` or `height` is zero.
/// - [`VignettingError::InvalidImage`] if `image.len() != width * height * 3`.
/// - [`VignettingError::InvalidConfig`] if `config.validate()` fails.
pub fn apply_vignetting(
    image: &[f32],
    width: usize,
    height: usize,
    config: &VignettingConfig,
) -> Result<Vec<f32>, VignettingError> {
    validate_rgb(image, width, height)?;
    config.validate()?;

    let use_tint = has_tint(config);
    let mut output = Vec::with_capacity(image.len());

    for py in 0..height {
        for px in 0..width {
            let r = pixel_radius(
                px,
                py,
                width,
                height,
                config.center_x,
                config.center_y,
                config.aspect_correction,
            );
            let ev = effective_factor(&config.model, r, config.strength);
            let dark = 1.0 - ev; // how much of the tint/black to add

            let base = (py * width + px) * 3;
            let (ir, ig, ib) = (image[base], image[base + 1], image[base + 2]);

            let (or_, og, ob) = if use_tint {
                (
                    (ir * ev + config.tint_r * dark).clamp(0.0, 1.0),
                    (ig * ev + config.tint_g * dark).clamp(0.0, 1.0),
                    (ib * ev + config.tint_b * dark).clamp(0.0, 1.0),
                )
            } else {
                (
                    (ir * ev).clamp(0.0, 1.0),
                    (ig * ev).clamp(0.0, 1.0),
                    (ib * ev).clamp(0.0, 1.0),
                )
            };

            output.push(or_);
            output.push(og);
            output.push(ob);
        }
    }

    Ok(output)
}

/// Apply vignetting to an RGBA image.
///
/// The alpha channel is preserved unchanged. RGB channels are processed
/// identically to [`apply_vignetting`].
///
/// # Errors
///
/// - [`VignettingError::EmptyImage`] if `width` or `height` is zero.
/// - [`VignettingError::InvalidImage`] if `image.len() != width * height * 4`.
/// - [`VignettingError::InvalidConfig`] if `config.validate()` fails.
pub fn apply_vignetting_rgba(
    image: &[f32],
    width: usize,
    height: usize,
    config: &VignettingConfig,
) -> Result<Vec<f32>, VignettingError> {
    validate_rgba(image, width, height)?;
    config.validate()?;

    let use_tint = has_tint(config);
    let mut output = Vec::with_capacity(image.len());

    for py in 0..height {
        for px in 0..width {
            let r = pixel_radius(
                px,
                py,
                width,
                height,
                config.center_x,
                config.center_y,
                config.aspect_correction,
            );
            let ev = effective_factor(&config.model, r, config.strength);
            let dark = 1.0 - ev;

            let base = (py * width + px) * 4;
            let (ir, ig, ib, ia) = (
                image[base],
                image[base + 1],
                image[base + 2],
                image[base + 3],
            );

            let (or_, og, ob) = if use_tint {
                (
                    (ir * ev + config.tint_r * dark).clamp(0.0, 1.0),
                    (ig * ev + config.tint_g * dark).clamp(0.0, 1.0),
                    (ib * ev + config.tint_b * dark).clamp(0.0, 1.0),
                )
            } else {
                (
                    (ir * ev).clamp(0.0, 1.0),
                    (ig * ev).clamp(0.0, 1.0),
                    (ib * ev).clamp(0.0, 1.0),
                )
            };

            output.push(or_);
            output.push(og);
            output.push(ob);
            output.push(ia); // alpha unchanged
        }
    }

    Ok(output)
}

/// Generate a single-channel vignetting mask image.
///
/// Returns a `Vec<f32>` of length `width * height` where each value is the
/// effective vignetting factor at that pixel (`1.0` = bright centre,
/// `0.0` = dark corner at maximum strength).  Useful for previewing the
/// vignetting pattern or compositing externally.
///
/// # Errors
///
/// - [`VignettingError::EmptyImage`] if `width` or `height` is zero.
/// - [`VignettingError::InvalidConfig`] if `config.validate()` fails.
pub fn generate_vignetting_mask(
    width: usize,
    height: usize,
    config: &VignettingConfig,
) -> Result<Vec<f32>, VignettingError> {
    if width == 0 || height == 0 {
        return Err(VignettingError::EmptyImage);
    }
    config.validate()?;

    let mut mask = Vec::with_capacity(width * height);

    for py in 0..height {
        for px in 0..width {
            let r = pixel_radius(
                px,
                py,
                width,
                height,
                config.center_x,
                config.center_y,
                config.aspect_correction,
            );
            let ev = effective_factor(&config.model, r, config.strength);
            mask.push(ev);
        }
    }

    Ok(mask)
}

// ─────────────────────────────────────────────────────────────────────────────
// Animated / temporal vignetting
// ─────────────────────────────────────────────────────────────────────────────

/// Compute an animated vignetting strength for a given frame in a sequence.
///
/// Uses a sinusoidal pulse: `strength = base_strength + pulse_amplitude * sin(2π * frame / total)`.
/// The result is clamped to `[0, 1]`.
///
/// # Arguments
///
/// - `frame_idx`: zero-based frame index.
/// - `total_frames`: total number of frames in the sequence.
/// - `base_strength`: mean strength around which the pulse oscillates.
/// - `pulse_amplitude`: half-amplitude of the sinusoidal pulse.
pub fn animated_vignetting_strength(
    frame_idx: usize,
    total_frames: usize,
    base_strength: f32,
    pulse_amplitude: f32,
) -> f32 {
    let phase = if total_frames == 0 {
        0.0
    } else {
        std::f32::consts::TAU * frame_idx as f32 / total_frames as f32
    };
    (base_strength + pulse_amplitude * phase.sin()).clamp(0.0, 1.0)
}

/// Apply vignetting to a sequence of frames with per-frame strength values.
///
/// Each frame `frames[i]` is vignetted using a copy of `config` with
/// `strength` overridden to `strength_per_frame[i]`.
///
/// # Errors
///
/// - [`VignettingError::InvalidConfig`] if `frames.len() != strength_per_frame.len()`.
/// - Any error propagated from [`apply_vignetting`] for individual frames.
pub fn apply_vignetting_sequence(
    frames: &[Vec<f32>],
    width: usize,
    height: usize,
    config: &VignettingConfig,
    strength_per_frame: &[f32],
) -> Result<Vec<Vec<f32>>, VignettingError> {
    if frames.len() != strength_per_frame.len() {
        return Err(VignettingError::InvalidConfig(format!(
            "frames.len() ({}) != strength_per_frame.len() ({})",
            frames.len(),
            strength_per_frame.len()
        )));
    }

    let mut results = Vec::with_capacity(frames.len());
    for (frame, &strength) in frames.iter().zip(strength_per_frame.iter()) {
        let mut frame_config = config.clone();
        frame_config.strength = strength.clamp(0.0, 1.0);
        results.push(apply_vignetting(frame, width, height, &frame_config)?);
    }

    Ok(results)
}

// ─────────────────────────────────────────────────────────────────────────────
// VignettingStats
// ─────────────────────────────────────────────────────────────────────────────

/// Diagnostic statistics describing a vignetting operation.
#[derive(Debug, Clone)]
pub struct VignettingStats {
    /// Effective vignetting factor at the image centre (`r = 0`).
    pub mean_factor_at_center: f32,

    /// Effective vignetting factor at the corner of the image (`r = 1.0`).
    pub mean_factor_at_corner: f32,

    /// Effective vignetting factor at the image edge midpoint (`r = 0.5`).
    pub mean_factor_at_edge: f32,

    /// Fraction of pixels whose effective factor is below 0.5.
    pub effective_dark_area: f32,

    /// Mean absolute relative change: `mean(|vignetted − original| / (|original| + ε))`.
    pub mean_image_change: f32,
}

/// Compute vignetting statistics comparing an original and a vignetted image.
///
/// # Arguments
///
/// - `original`: the unmodified RGB image slice.
/// - `vignetted`: the vignetted RGB image slice.
/// - `width`, `height`: image dimensions.
/// - `config`: the vignetting configuration that was applied.
///
/// # Errors
///
/// - [`VignettingError::EmptyImage`] if `width` or `height` is zero.
/// - [`VignettingError::InvalidImage`] if slice lengths do not match dimensions.
/// - [`VignettingError::InvalidConfig`] if `config.validate()` fails.
pub fn compute_vignetting_stats(
    original: &[f32],
    vignetted: &[f32],
    width: usize,
    height: usize,
    config: &VignettingConfig,
) -> Result<VignettingStats, VignettingError> {
    validate_rgb(original, width, height)?;
    validate_rgb(vignetted, width, height)?;
    config.validate()?;

    // Analytic factors at canonical radii
    let factor_at_center = effective_factor(&config.model, 0.0, config.strength);
    let factor_at_corner = effective_factor(&config.model, 1.0, config.strength);
    let factor_at_edge = effective_factor(&config.model, 0.5, config.strength);

    // Count pixels with effective factor < 0.5 (per pixel, not per channel)
    let total_pixels = width * height;
    let mut dark_count = 0usize;
    for py in 0..height {
        for px in 0..width {
            let r = pixel_radius(
                px,
                py,
                width,
                height,
                config.center_x,
                config.center_y,
                config.aspect_correction,
            );
            let ev = effective_factor(&config.model, r, config.strength);
            if ev < 0.5 {
                dark_count += 1;
            }
        }
    }

    // Mean absolute relative change across all channels
    let n_values = original.len();
    let mut sum_change = 0.0f64;
    for (o, v) in original.iter().zip(vignetted.iter()) {
        let change = (*v - *o).abs() as f64;
        let denom = o.abs() as f64 + 1e-8;
        sum_change += change / denom;
    }
    let mean_image_change = (sum_change / n_values as f64) as f32;

    Ok(VignettingStats {
        mean_factor_at_center: factor_at_center,
        mean_factor_at_corner: factor_at_corner,
        mean_factor_at_edge: factor_at_edge,
        effective_dark_area: dark_count as f32 / total_pixels as f32,
        mean_image_change,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── VignettingModel::factor ───────────────────────────────────────────────

    #[test]
    fn test_power_law_center() {
        let m = VignettingModel::PowerLaw { power: 2.0 };
        assert!(
            (m.factor(0.0) - 1.0).abs() < 1e-6,
            "PowerLaw center should be 1.0"
        );
    }

    #[test]
    fn test_power_law_edge() {
        let m = VignettingModel::PowerLaw { power: 2.0 };
        assert!(
            (m.factor(1.0) - 0.0).abs() < 1e-6,
            "PowerLaw at r=1 should be 0.0"
        );
    }

    #[test]
    fn test_cosine4_center() {
        let m = VignettingModel::Cosine4;
        assert!(
            (m.factor(0.0) - 1.0).abs() < 1e-6,
            "Cosine4 center should be 1.0"
        );
    }

    #[test]
    fn test_cosine4_corner() {
        let m = VignettingModel::Cosine4;
        // cos^4(π/4) = (1/√2)^4 = 0.25
        let v = m.factor(1.0);
        assert!(
            (v - 0.25).abs() < 1e-5,
            "Cosine4 at r=1 should be ~0.25, got {v}"
        );
    }

    #[test]
    fn test_gaussian_center() {
        let m = VignettingModel::Gaussian { sigma: 0.5 };
        assert!(
            (m.factor(0.0) - 1.0).abs() < 1e-6,
            "Gaussian center should be 1.0"
        );
    }

    #[test]
    fn test_gaussian_falloff() {
        let m = VignettingModel::Gaussian { sigma: 0.5 };
        // Should strictly decrease with radius
        assert!(m.factor(0.5) < m.factor(0.0));
        assert!(m.factor(1.0) < m.factor(0.5));
    }

    #[test]
    fn test_polynomial_center() {
        let m = VignettingModel::Polynomial {
            k1: -0.3,
            k2: -0.1,
            k3: 0.0,
        };
        // At r=0: 1 + 0 + 0 + 0 = 1.0
        assert!(
            (m.factor(0.0) - 1.0).abs() < 1e-6,
            "Polynomial center should be 1.0"
        );
    }

    #[test]
    fn test_polynomial_negative_darkens() {
        let m = VignettingModel::Polynomial {
            k1: -0.5,
            k2: -0.3,
            k3: 0.0,
        };
        assert!(
            m.factor(1.0) < m.factor(0.0),
            "Negative k1/k2 should darken edges"
        );
    }

    // ── VignettingModel::validate ─────────────────────────────────────────────

    #[test]
    fn test_power_law_zero_power_invalid() {
        let m = VignettingModel::PowerLaw { power: 0.0 };
        assert!(m.validate().is_err(), "power=0 should be invalid");
    }

    #[test]
    fn test_power_law_negative_power_invalid() {
        let m = VignettingModel::PowerLaw { power: -1.0 };
        assert!(m.validate().is_err(), "negative power should be invalid");
    }

    #[test]
    fn test_cosine4_always_valid() {
        assert!(VignettingModel::Cosine4.validate().is_ok());
    }

    #[test]
    fn test_gaussian_zero_sigma_invalid() {
        let m = VignettingModel::Gaussian { sigma: 0.0 };
        assert!(m.validate().is_err(), "sigma=0 should be invalid");
    }

    #[test]
    fn test_polynomial_non_finite_invalid() {
        let m = VignettingModel::Polynomial {
            k1: f32::INFINITY,
            k2: 0.0,
            k3: 0.0,
        };
        assert!(m.validate().is_err(), "infinite k1 should be invalid");
    }

    // ── pixel_radius ──────────────────────────────────────────────────────────

    #[test]
    fn test_pixel_radius_center() {
        // Center pixel of a 101x101 image — radius should be essentially 0
        let r = pixel_radius(50, 50, 101, 101, 0.0, 0.0, false);
        assert!(r < 0.02, "center pixel radius should be near 0, got {r}");
    }

    #[test]
    fn test_pixel_radius_corner_square() {
        // For a square image with aspect_correction=true, corner should be ~√2
        // With correction: nx = 0.5 * (w/h) = 0.5, ny = 0.5 → r = √(0.5²+0.5²)*2 = √2 ≈ 1.414
        // Without correction: r = √(0.5²+0.5²)*2 = √0.5 * 2 ≈ 1.414
        // For a large square both corrections give the same
        let r = pixel_radius(0, 0, 100, 100, 0.0, 0.0, true);
        // Top-left corner of 100x100: nx≈−0.495, ny≈−0.495 → r≈2*√(0.495²+0.495²)≈1.4
        assert!(
            r > 1.3 && r < 1.5,
            "corner radius of square should be ~√2, got {r}"
        );
    }

    #[test]
    fn test_pixel_radius_without_aspect_correction() {
        // Same pixel, different aspect correction should give different results for non-square
        let r_corrected = pixel_radius(0, 50, 200, 100, 0.0, 0.0, true);
        let r_uncorrected = pixel_radius(0, 50, 200, 100, 0.0, 0.0, false);
        // aspect_correction scales nx by width/height = 2, so should differ
        assert!(
            (r_corrected - r_uncorrected).abs() > 0.1,
            "aspect correction should change radius"
        );
    }

    // ── VignettingConfig ──────────────────────────────────────────────────────

    #[test]
    fn test_default_config_valid() {
        let config = VignettingConfig::default();
        assert!(config.validate().is_ok(), "default config should be valid");
    }

    #[test]
    fn test_config_strength_above_one_invalid() {
        let config = VignettingConfig {
            strength: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err(), "strength > 1 should be invalid");
    }

    #[test]
    fn test_config_center_out_of_range() {
        let config = VignettingConfig {
            center_x: 0.6,
            ..Default::default()
        };
        assert!(
            config.validate().is_err(),
            "center_x > 0.5 should be invalid"
        );
    }

    #[test]
    fn test_cinematic_stronger_than_subtle() {
        let cin = VignettingConfig::cinematic();
        let sub = VignettingConfig::subtle();
        assert!(
            cin.strength > sub.strength,
            "cinematic ({}) should have higher strength than subtle ({})",
            cin.strength,
            sub.strength
        );
    }

    #[test]
    fn test_all_presets_valid() {
        assert!(VignettingConfig::cinematic().validate().is_ok());
        assert!(VignettingConfig::subtle().validate().is_ok());
        assert!(VignettingConfig::hard_edge().validate().is_ok());
    }

    // ── apply_vignetting ─────────────────────────────────────────────────────

    #[test]
    fn test_apply_vignetting_empty_image_error() {
        let config = VignettingConfig::default();
        let result = apply_vignetting(&[], 0, 0, &config);
        assert!(matches!(result, Err(VignettingError::EmptyImage)));
    }

    #[test]
    fn test_apply_vignetting_wrong_size_error() {
        let config = VignettingConfig::default();
        // 5 pixels instead of 4*4*3 = 48
        let result = apply_vignetting(&[0.5; 5], 4, 4, &config);
        assert!(matches!(result, Err(VignettingError::InvalidImage(_))));
    }

    #[test]
    fn test_apply_vignetting_strength_zero_unchanged() {
        let config = VignettingConfig {
            strength: 0.0,
            ..Default::default()
        };
        let w = 8;
        let h = 8;
        let image: Vec<f32> = (0..w * h * 3)
            .map(|i| i as f32 / (w * h * 3) as f32)
            .collect();
        let out = apply_vignetting(&image, w, h, &config).expect("apply failed");
        for (orig, vig) in image.iter().zip(out.iter()) {
            assert!(
                (*orig - *vig).abs() < 1e-6,
                "strength=0 should leave image unchanged, diff = {}",
                (*orig - *vig).abs()
            );
        }
    }

    #[test]
    fn test_apply_vignetting_center_brighter_than_corner() {
        let config = VignettingConfig::default(); // strength = 0.5
        let w = 64;
        let h = 64;
        let image = vec![1.0_f32; w * h * 3];
        let out = apply_vignetting(&image, w, h, &config).expect("apply failed");

        // Center pixel
        let center_idx = (h / 2 * w + w / 2) * 3;
        let center_val = out[center_idx];

        // Top-left corner pixel
        let corner_val = out[0];

        assert!(
            center_val > corner_val,
            "center ({center_val}) should be brighter than corner ({corner_val})"
        );
    }

    #[test]
    fn test_apply_vignetting_center_is_max() {
        let config = VignettingConfig::default();
        let w = 32;
        let h = 32;
        let image = vec![1.0_f32; w * h * 3];
        let out = apply_vignetting(&image, w, h, &config).expect("apply failed");

        let center_idx = (h / 2 * w + w / 2) * 3;
        let center_val = out[center_idx];

        for i in (0..out.len()).step_by(3) {
            assert!(
                out[i] <= center_val + 1e-5,
                "center should be the brightest pixel, found pixel {i} with value {}",
                out[i]
            );
        }
    }

    #[test]
    fn test_apply_vignetting_output_clamped() {
        let config = VignettingConfig::default();
        let w = 4;
        let h = 4;
        let image = vec![1.0_f32; w * h * 3];
        let out = apply_vignetting(&image, w, h, &config).expect("apply failed");
        for v in &out {
            assert!(
                *v >= 0.0 && *v <= 1.0,
                "output should be clamped to [0,1], got {v}"
            );
        }
    }

    #[test]
    fn test_apply_vignetting_tint_blends() {
        let config = VignettingConfig::cinematic(); // has warm tint
        let w = 8;
        let h = 8;
        let image = vec![0.5_f32; w * h * 3];
        let out = apply_vignetting(&image, w, h, &config).expect("apply failed");

        // The corner pixel should not be pure black (there is a warm tint)
        let corner_r = out[0];
        assert!(
            corner_r > 0.0,
            "corner with warm tint should not be pure black, got {corner_r}"
        );

        // And the center should be close to original (strong vignette only at corners)
        let center_idx = (h / 2 * w + w / 2) * 3;
        let center_r = out[center_idx];
        assert!(
            center_r > 0.4,
            "center should remain relatively bright, got {center_r}"
        );
    }

    // ── apply_vignetting_rgba ─────────────────────────────────────────────────

    #[test]
    fn test_apply_vignetting_rgba_alpha_unchanged() {
        let config = VignettingConfig::default();
        let w = 4;
        let h = 4;
        // RGBA: set alpha to 0.75 everywhere
        let mut image = vec![0.8_f32; w * h * 4];
        for i in (3..image.len()).step_by(4) {
            image[i] = 0.75;
        }
        let out = apply_vignetting_rgba(&image, w, h, &config).expect("apply_rgba failed");
        for i in (3..out.len()).step_by(4) {
            assert!(
                (out[i] - 0.75).abs() < 1e-6,
                "alpha should be unchanged, got {}",
                out[i]
            );
        }
    }

    #[test]
    fn test_apply_vignetting_rgba_wrong_size_error() {
        let config = VignettingConfig::default();
        // 5 values instead of 4*4*4 = 64
        let result = apply_vignetting_rgba(&[0.5; 5], 4, 4, &config);
        assert!(matches!(result, Err(VignettingError::InvalidImage(_))));
    }

    // ── generate_vignetting_mask ──────────────────────────────────────────────

    #[test]
    fn test_generate_vignetting_mask_correct_size() {
        let config = VignettingConfig::default();
        let w = 16;
        let h = 12;
        let mask = generate_vignetting_mask(w, h, &config).expect("mask failed");
        assert_eq!(mask.len(), w * h, "mask should have w*h elements");
    }

    #[test]
    fn test_generate_vignetting_mask_center_near_one() {
        let config = VignettingConfig::default(); // strength=0.5
        let w = 65;
        let h = 65;
        let mask = generate_vignetting_mask(w, h, &config).expect("mask failed");
        // Center pixel
        let center_idx = h / 2 * w + w / 2;
        let center_val = mask[center_idx];
        // With strength=0.5 and model factor=1.0 at center: effective = 1 - 0.5*(1-1) = 1.0
        assert!(
            center_val > 0.95,
            "center of mask should be near 1.0, got {center_val}"
        );
    }

    #[test]
    fn test_generate_vignetting_mask_corner_darker_than_center() {
        let config = VignettingConfig::default();
        let w = 32;
        let h = 32;
        let mask = generate_vignetting_mask(w, h, &config).expect("mask failed");

        let center_idx = h / 2 * w + w / 2;
        let corner_val = mask[0]; // top-left corner
        let center_val = mask[center_idx];

        assert!(
            corner_val < center_val,
            "corner ({corner_val}) should be darker than center ({center_val})"
        );
    }

    #[test]
    fn test_generate_vignetting_mask_empty_error() {
        let config = VignettingConfig::default();
        let result = generate_vignetting_mask(0, 10, &config);
        assert!(matches!(result, Err(VignettingError::EmptyImage)));
    }

    // ── animated_vignetting_strength ──────────────────────────────────────────

    #[test]
    fn test_animated_strength_in_range() {
        for frame in 0..60 {
            let s = animated_vignetting_strength(frame, 60, 0.5, 0.4);
            assert!(
                (0.0..=1.0).contains(&s),
                "animated strength must be in [0,1], got {s} at frame {frame}"
            );
        }
    }

    #[test]
    fn test_animated_strength_zero_frames_no_panic() {
        // Should not panic even with total_frames=0
        let s = animated_vignetting_strength(0, 0, 0.5, 0.3);
        assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn test_animated_strength_oscillates() {
        // Over a full cycle the min should be below and max above base_strength
        let base = 0.5;
        let amp = 0.3;
        let total = 100;
        let strengths: Vec<f32> = (0..total)
            .map(|f| animated_vignetting_strength(f, total, base, amp))
            .collect();
        let min_s = strengths.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_s = strengths.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(min_s < base, "should have frames below base strength");
        assert!(max_s > base, "should have frames above base strength");
    }

    // ── apply_vignetting_sequence ─────────────────────────────────────────────

    #[test]
    fn test_apply_vignetting_sequence_length_mismatch_error() {
        let config = VignettingConfig::default();
        let w = 4;
        let h = 4;
        let frame = vec![0.5_f32; w * h * 3];
        let frames = vec![frame.clone(), frame.clone()];
        let strengths = vec![0.5_f32]; // mismatched length
        let result = apply_vignetting_sequence(&frames, w, h, &config, &strengths);
        assert!(matches!(result, Err(VignettingError::InvalidConfig(_))));
    }

    #[test]
    fn test_apply_vignetting_sequence_correct_output_count() {
        let config = VignettingConfig::default();
        let w = 4;
        let h = 4;
        let frame = vec![0.8_f32; w * h * 3];
        let frames = vec![frame.clone(), frame.clone(), frame.clone()];
        let strengths = vec![0.3_f32, 0.5, 0.7];
        let out =
            apply_vignetting_sequence(&frames, w, h, &config, &strengths).expect("sequence failed");
        assert_eq!(out.len(), 3, "should produce 3 output frames");
    }

    #[test]
    fn test_apply_vignetting_sequence_zero_strength_frame_unchanged() {
        let config = VignettingConfig::default();
        let w = 4;
        let h = 4;
        let image = vec![0.7_f32; w * h * 3];
        let frames = vec![image.clone()];
        let strengths = vec![0.0_f32];
        let out =
            apply_vignetting_sequence(&frames, w, h, &config, &strengths).expect("sequence failed");
        for (a, b) in image.iter().zip(out[0].iter()) {
            assert!(
                (*a - *b).abs() < 1e-6,
                "strength=0 frame should be unchanged"
            );
        }
    }

    // ── compute_vignetting_stats ──────────────────────────────────────────────

    #[test]
    fn test_compute_vignetting_stats_center_near_one() {
        let config = VignettingConfig::default(); // strength=0.5
        let w = 32;
        let h = 32;
        let image = vec![1.0_f32; w * h * 3];
        let vignetted = apply_vignetting(&image, w, h, &config).expect("apply failed");
        let stats =
            compute_vignetting_stats(&image, &vignetted, w, h, &config).expect("stats failed");
        // effective factor at center = 1.0 (PowerLaw factor=1 at r=0)
        assert!(
            stats.mean_factor_at_center > 0.95,
            "center factor should be near 1.0, got {}",
            stats.mean_factor_at_center
        );
    }

    #[test]
    fn test_compute_vignetting_stats_corner_darker() {
        let config = VignettingConfig::default();
        let w = 8;
        let h = 8;
        let image = vec![1.0_f32; w * h * 3];
        let vignetted = apply_vignetting(&image, w, h, &config).expect("apply failed");
        let stats =
            compute_vignetting_stats(&image, &vignetted, w, h, &config).expect("stats failed");
        assert!(
            stats.mean_factor_at_corner < stats.mean_factor_at_center,
            "corner factor should be darker than center"
        );
    }

    #[test]
    fn test_compute_vignetting_stats_no_change_when_strength_zero() {
        let config = VignettingConfig {
            strength: 0.0,
            ..Default::default()
        };
        let w = 8;
        let h = 8;
        let image = vec![0.6_f32; w * h * 3];
        let vignetted = apply_vignetting(&image, w, h, &config).expect("apply failed");
        let stats =
            compute_vignetting_stats(&image, &vignetted, w, h, &config).expect("stats failed");
        assert!(
            stats.mean_image_change < 1e-5,
            "no change expected with strength=0, got {}",
            stats.mean_image_change
        );
    }

    #[test]
    fn test_compute_vignetting_stats_size_mismatch_error() {
        let config = VignettingConfig::default();
        let result = compute_vignetting_stats(&[0.5; 12], &[0.5; 15], 2, 2, &config);
        assert!(matches!(result, Err(VignettingError::InvalidImage(_))));
    }
}
