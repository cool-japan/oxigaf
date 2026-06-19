//! Image-space post-processing motion blur.
//!
//! Implements motion blur as a post-processing effect applied to flat RGBA
//! images (`Vec<u8>`, row-major, `width × height × 4` bytes).  Motion vectors
//! are flat `Vec<f32>` fields of shape H×W×2 (dx, dy per pixel, pixels/frame).
//!
//! # Types
//!
//! - [`MotionBlurConfig`] — common configuration (samples, shutter angle, …)
//! - [`ImageMotionField`] — flat f32 per-pixel motion vector field
//! - [`MotionStats`] — statistics over a motion field
//! - [`BlurType`] — selects Linear / Radial / Rotational / PerPixel blur

use crate::motion_blur::MotionBlurError;

// ─────────────────────────────────────────────────────────────────────────────
// MotionBlurConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for image-space motion blur post-processing.
#[derive(Debug, Clone)]
pub struct MotionBlurConfig {
    /// Number of samples taken along the motion vector (default: 16).
    pub samples: usize,
    /// Shutter angle in degrees — controls how much of the motion vector is
    /// used.  Must be in `(0, 360]`.  Default: 180.0.
    pub shutter_angle: f32,
    /// Maximum blur radius in pixels.  Motion vectors longer than this are
    /// clamped before sampling.  Default: 32.0.
    pub max_blur_pixels: f32,
    /// When `true`, blur samples are weighted by the alpha channel.
    /// Default: `false`.
    pub use_alpha_weight: bool,
}

impl Default for MotionBlurConfig {
    fn default() -> Self {
        Self {
            samples: 16,
            shutter_angle: 180.0,
            max_blur_pixels: 32.0,
            use_alpha_weight: false,
        }
    }
}

impl MotionBlurConfig {
    /// Validate configuration parameters.
    ///
    /// # Errors
    ///
    /// - [`MotionBlurError::InvalidSampleCount`] when `samples == 0`.
    /// - [`MotionBlurError::InvalidShutterAngle`] when `shutter_angle` is not
    ///   in `(0, 360]`.
    pub fn validate(&self) -> Result<(), MotionBlurError> {
        if self.samples == 0 {
            return Err(MotionBlurError::InvalidSampleCount { samples: 0 });
        }
        if self.shutter_angle <= 0.0 || self.shutter_angle > 360.0 {
            return Err(MotionBlurError::InvalidShutterAngle {
                angle: self.shutter_angle,
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ImageMotionField
// ─────────────────────────────────────────────────────────────────────────────

/// Per-pixel 2-D motion vector field using flat `f32` storage.
///
/// Layout: `[dx_0, dy_0, dx_1, dy_1, …]` in row-major order
/// (`len = width * height * 2`).
///
/// This is distinct from [`crate::MotionVectorField`] (from `temporal.rs`)
/// which stores typed `MotionVector` structs with `usize` dimensions.  This
/// type uses a compact flat buffer suitable for image-space post-processing.
#[derive(Debug, Clone)]
pub struct ImageMotionField {
    /// Flat motion vector data: `[dx, dy]` per pixel in row-major order.
    pub data: Vec<f32>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl ImageMotionField {
    /// Create a new field, validating `data.len() == width * height * 2`.
    ///
    /// # Errors
    ///
    /// [`MotionBlurError::MotionVectorMismatch`] when the buffer length does
    /// not match the declared dimensions.
    pub fn new(data: Vec<f32>, width: u32, height: u32) -> Result<Self, MotionBlurError> {
        let px_len = (width as usize).saturating_mul(height as usize);
        let expected = px_len.saturating_mul(2);
        if data.len() != expected {
            return Err(MotionBlurError::MotionVectorMismatch {
                mv_len: data.len(),
                px_len,
            });
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Create a field where every pixel has the same `(dx, dy)` motion vector.
    pub fn uniform(width: u32, height: u32, dx: f32, dy: f32) -> Self {
        let n = (width as usize).saturating_mul(height as usize);
        let mut data = Vec::with_capacity(n * 2);
        for _ in 0..n {
            data.push(dx);
            data.push(dy);
        }
        Self {
            data,
            width,
            height,
        }
    }

    /// Create a zero-motion field (all vectors are `(0, 0)`).
    pub fn zero(width: u32, height: u32) -> Self {
        let n = (width as usize).saturating_mul(height as usize);
        Self {
            data: vec![0.0_f32; n * 2],
            width,
            height,
        }
    }

    /// Return the motion magnitude at pixel `(x, y)`.  Returns `0.0` for OOB.
    pub fn magnitude_at(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        let idx = ((y as usize) * (self.width as usize) + (x as usize)) * 2;
        let dx = self.data.get(idx).copied().unwrap_or(0.0);
        let dy = self.data.get(idx + 1).copied().unwrap_or(0.0);
        (dx * dx + dy * dy).sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MotionStats
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics about an [`ImageMotionField`].
#[derive(Debug, Clone)]
pub struct MotionStats {
    /// Mean magnitude across all pixels.
    pub mean_magnitude: f32,
    /// Maximum magnitude across all pixels.
    pub max_magnitude: f32,
    /// Minimum magnitude across all pixels.
    pub min_magnitude: f32,
    /// Standard deviation of magnitudes.
    pub std_magnitude: f32,
    /// Fraction of pixels whose magnitude exceeds `0.5` pixels.
    pub fraction_moving: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// BlurType
// ─────────────────────────────────────────────────────────────────────────────

/// Selects the type of image-space motion blur to apply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlurType {
    /// Blur in a single global direction `(dx, dy)`.
    Linear,
    /// Radial blur radiating from a centre point (zoom blur).
    Radial,
    /// Rotational blur around a centre point.
    Rotational,
    /// Per-pixel blur driven by an [`ImageMotionField`].
    PerPixel,
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

fn validate_rgba(image: &[u8], width: u32, height: u32) -> Result<(), MotionBlurError> {
    if image.is_empty() {
        return Err(MotionBlurError::EmptyImage);
    }
    let expected = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(4);
    if image.len() != expected {
        return Err(MotionBlurError::InvalidDimensions {
            actual: image.len(),
            expected,
            width,
            height,
        });
    }
    Ok(())
}

/// Bilinear sample of a `u8` RGBA image at fractional coordinates.
/// Coordinates are clamped to image bounds.  Returns `[r, g, b, a]` in `[0,1]`.
fn bilinear_sample_u8_rgba(image: &[u8], width: u32, height: u32, fx: f32, fy: f32) -> [f32; 4] {
    let w = width as usize;
    let mx = w.saturating_sub(1);
    let my = (height as usize).saturating_sub(1);
    let fx = fx.clamp(0.0, mx as f32);
    let fy = fy.clamp(0.0, my as f32);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(mx);
    let y1 = (y0 + 1).min(my);
    let fx_f = fx - x0 as f32;
    let fy_f = fy - y0 as f32;
    let w00 = (1.0 - fy_f) * (1.0 - fx_f);
    let w10 = (1.0 - fy_f) * fx_f;
    let w01 = fy_f * (1.0 - fx_f);
    let w11 = fy_f * fx_f;
    macro_rules! px {
        ($xi:expr, $yi:expr) => {{
            let b = ($yi * w + $xi) * 4;
            [
                image.get(b).copied().unwrap_or(0) as f32 / 255.0,
                image.get(b + 1).copied().unwrap_or(0) as f32 / 255.0,
                image.get(b + 2).copied().unwrap_or(0) as f32 / 255.0,
                image.get(b + 3).copied().unwrap_or(0) as f32 / 255.0,
            ]
        }};
    }
    let p00 = px!(x0, y0);
    let p10 = px!(x1, y0);
    let p01 = px!(x0, y1);
    let p11 = px!(x1, y1);
    let mut out = [0.0_f32; 4];
    for c in 0..4 {
        out[c] = w00 * p00[c] + w10 * p10[c] + w01 * p01[c] + w11 * p11[c];
    }
    out
}

/// Pixel coordinate with an associated motion vector used by [`sample_along_motion`].
struct MotionSample {
    /// Pixel X coordinate.
    px: u32,
    /// Pixel Y coordinate.
    py: u32,
    /// Motion vector X component.
    dx: f32,
    /// Motion vector Y component.
    dy: f32,
}

/// Sample `n_samples` equally-spaced points along the motion vector from
/// `(ms.px, ms.py)` to `(ms.px + ms.dx, ms.py + ms.dy)` and return the RGBA average.
fn sample_along_motion(
    image: &[u8],
    width: u32,
    height: u32,
    ms: MotionSample,
    n_samples: usize,
) -> [f32; 4] {
    let n = n_samples.max(1);
    let mut acc = [0.0_f32; 4];
    for i in 0..n {
        let t = i as f32 / (n.saturating_sub(1).max(1)) as f32;
        let s = bilinear_sample_u8_rgba(
            image,
            width,
            height,
            ms.px as f32 + t * ms.dx,
            ms.py as f32 + t * ms.dy,
        );
        for c in 0..4 {
            acc[c] += s[c];
        }
    }
    let inv = 1.0 / n as f32;
    [acc[0] * inv, acc[1] * inv, acc[2] * inv, acc[3] * inv]
}

#[inline]
fn to_u8(v: [f32; 4]) -> [u8; 4] {
    [
        (v[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (v[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (v[2].clamp(0.0, 1.0) * 255.0).round() as u8,
        (v[3].clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Public blur functions
// ─────────────────────────────────────────────────────────────────────────────

/// Apply linear motion blur with a global direction `(dx, dy)`.
///
/// Actual motion used: `(dx, dy) * (shutter_angle / 360)`.
///
/// # Errors
/// [`MotionBlurError::EmptyImage`], [`MotionBlurError::InvalidDimensions`],
/// [`MotionBlurError::InvalidSampleCount`], [`MotionBlurError::InvalidShutterAngle`].
pub fn apply_linear_motion_blur(
    image_rgba: &[u8],
    width: u32,
    height: u32,
    dx: f32,
    dy: f32,
    config: &MotionBlurConfig,
) -> Result<Vec<u8>, MotionBlurError> {
    validate_rgba(image_rgba, width, height)?;
    config.validate()?;
    let sf = config.shutter_angle / 360.0;
    let (sdx, sdy) = (dx * sf, dy * sf);
    let mut out = Vec::with_capacity((width as usize) * (height as usize) * 4);
    for py in 0..height {
        for px in 0..width {
            let rgba = sample_along_motion(
                image_rgba,
                width,
                height,
                MotionSample {
                    px,
                    py,
                    dx: sdx,
                    dy: sdy,
                },
                config.samples,
            );
            out.extend_from_slice(&to_u8(rgba));
        }
    }
    Ok(out)
}

/// Apply radial (zoom) motion blur radiating from `(center_x, center_y)`.
///
/// # Errors
/// Same as [`apply_linear_motion_blur`].
pub fn apply_radial_blur(
    image_rgba: &[u8],
    width: u32,
    height: u32,
    center_x: f32,
    center_y: f32,
    strength: f32,
    config: &MotionBlurConfig,
) -> Result<Vec<u8>, MotionBlurError> {
    validate_rgba(image_rgba, width, height)?;
    config.validate()?;
    let sf = config.shutter_angle / 360.0;
    let mut out = Vec::with_capacity((width as usize) * (height as usize) * 4);
    for py in 0..height {
        for px in 0..width {
            let dx = (px as f32 - center_x) * strength * sf;
            let dy = (py as f32 - center_y) * strength * sf;
            let rgba = sample_along_motion(
                image_rgba,
                width,
                height,
                MotionSample { px, py, dx, dy },
                config.samples,
            );
            out.extend_from_slice(&to_u8(rgba));
        }
    }
    Ok(out)
}

/// Apply rotational motion blur around `(center_x, center_y)`.
///
/// The blur is the tangential arc each pixel sweeps through `angle_deg * shutter_angle/360`.
///
/// # Errors
/// Same as [`apply_linear_motion_blur`].
pub fn apply_rotational_blur(
    image_rgba: &[u8],
    width: u32,
    height: u32,
    center_x: f32,
    center_y: f32,
    angle_deg: f32,
    config: &MotionBlurConfig,
) -> Result<Vec<u8>, MotionBlurError> {
    validate_rgba(image_rgba, width, height)?;
    config.validate()?;
    use std::f32::consts::PI;
    let sf = config.shutter_angle / 360.0;
    let angle_rad = angle_deg * PI / 180.0;
    let mut out = Vec::with_capacity((width as usize) * (height as usize) * 4);
    for py in 0..height {
        for px in 0..width {
            let rx = px as f32 - center_x;
            let ry = py as f32 - center_y;
            let r = (rx * rx + ry * ry).sqrt();
            let tmag = (rx * rx + ry * ry).sqrt(); // same as r
            let (ntx, nty) = if tmag > 1e-9 {
                (-ry / tmag, rx / tmag)
            } else {
                (0.0, 0.0)
            };
            let scale = r * angle_rad * sf;
            let rgba = sample_along_motion(
                image_rgba,
                width,
                height,
                MotionSample {
                    px,
                    py,
                    dx: ntx * scale,
                    dy: nty * scale,
                },
                config.samples,
            );
            out.extend_from_slice(&to_u8(rgba));
        }
    }
    Ok(out)
}

/// Apply per-pixel motion blur using an [`ImageMotionField`].
///
/// Zero-motion pixels are copied directly.  Vectors exceeding
/// `config.max_blur_pixels` are clamped.
///
/// # Errors
/// [`MotionBlurError::EmptyImage`], [`MotionBlurError::InvalidDimensions`],
/// [`MotionBlurError::MotionVectorMismatch`], config validation errors.
pub fn apply_per_pixel_motion_blur(
    image_rgba: &[u8],
    width: u32,
    height: u32,
    motion_field: &ImageMotionField,
    config: &MotionBlurConfig,
) -> Result<Vec<u8>, MotionBlurError> {
    validate_rgba(image_rgba, width, height)?;
    config.validate()?;
    if motion_field.width != width || motion_field.height != height {
        return Err(MotionBlurError::MotionVectorMismatch {
            mv_len: motion_field.data.len(),
            px_len: (width as usize) * (height as usize),
        });
    }
    let max_blur = config.max_blur_pixels;
    let mut out = Vec::with_capacity((width as usize) * (height as usize) * 4);
    for py in 0..height {
        for px in 0..width {
            let idx = ((py as usize) * (width as usize) + (px as usize)) * 2;
            let dx = motion_field.data.get(idx).copied().unwrap_or(0.0);
            let dy = motion_field.data.get(idx + 1).copied().unwrap_or(0.0);
            let mag = (dx * dx + dy * dy).sqrt();
            if mag < 1e-9 {
                let base = ((py as usize) * (width as usize) + (px as usize)) * 4;
                let r = image_rgba.get(base).copied().unwrap_or(0);
                let g = image_rgba.get(base + 1).copied().unwrap_or(0);
                let b = image_rgba.get(base + 2).copied().unwrap_or(0);
                let a = image_rgba.get(base + 3).copied().unwrap_or(255);
                out.extend_from_slice(&[r, g, b, a]);
            } else {
                let (fdx, fdy) = if mag > max_blur {
                    let s = max_blur / mag;
                    (dx * s, dy * s)
                } else {
                    (dx, dy)
                };
                let rgba = sample_along_motion(
                    image_rgba,
                    width,
                    height,
                    MotionSample {
                        px,
                        py,
                        dx: fdx,
                        dy: fdy,
                    },
                    config.samples,
                );
                out.extend_from_slice(&to_u8(rgba));
            }
        }
    }
    Ok(out)
}

/// High-level dispatcher: apply the requested [`BlurType`] to `image_rgba`.
///
/// | `blur_type`   | `params`                             |
/// |---------------|--------------------------------------|
/// | `Linear`      | `[dx, dy]`                           |
/// | `Radial`      | `[center_x, center_y, strength]`     |
/// | `Rotational`  | `[center_x, center_y, angle_deg]`    |
/// | `PerPixel`    | *(use [`apply_per_pixel_motion_blur`])* |
///
/// # Errors
/// [`MotionBlurError::InvalidConfig`] for `PerPixel` or wrong param count.
pub fn apply_motion_blur(
    image_rgba: &[u8],
    width: u32,
    height: u32,
    blur_type: BlurType,
    config: &MotionBlurConfig,
    params: &[f32],
) -> Result<Vec<u8>, MotionBlurError> {
    validate_rgba(image_rgba, width, height)?;
    config.validate()?;
    match blur_type {
        BlurType::Linear => {
            if params.len() < 2 {
                return Err(MotionBlurError::InvalidConfig(format!(
                    "Linear blur needs [dx, dy], got {} params",
                    params.len()
                )));
            }
            apply_linear_motion_blur(image_rgba, width, height, params[0], params[1], config)
        }
        BlurType::Radial => {
            if params.len() < 3 {
                return Err(MotionBlurError::InvalidConfig(format!(
                    "Radial blur needs [cx, cy, strength], got {} params",
                    params.len()
                )));
            }
            apply_radial_blur(
                image_rgba, width, height, params[0], params[1], params[2], config,
            )
        }
        BlurType::Rotational => {
            if params.len() < 3 {
                return Err(MotionBlurError::InvalidConfig(format!(
                    "Rotational blur needs [cx, cy, angle_deg], got {} params",
                    params.len()
                )));
            }
            apply_rotational_blur(
                image_rgba, width, height, params[0], params[1], params[2], config,
            )
        }
        BlurType::PerPixel => Err(MotionBlurError::InvalidConfig(
            "PerPixel blur needs a motion field; use apply_per_pixel_motion_blur".to_string(),
        )),
    }
}

/// Estimate a per-pixel motion field from two RGBA frames via luminance-diff proxy.
///
/// `dx = |lum_a - lum_b|`, `dy = 0.0` for all pixels.  Intentionally simple —
/// real optical flow requires iterative search.
///
/// # Errors
/// [`MotionBlurError::InvalidDimensions`] when either frame has wrong length.
pub fn compute_motion_from_frames(
    frame_a: &[u8],
    frame_b: &[u8],
    width: u32,
    height: u32,
) -> Result<ImageMotionField, MotionBlurError> {
    validate_rgba(frame_a, width, height)?;
    validate_rgba(frame_b, width, height)?;
    let n = (width as usize) * (height as usize);
    let mut data = Vec::with_capacity(n * 2);
    for i in 0..n {
        let b = i * 4;
        let lum = |f: &[u8]| {
            0.299 * f.get(b).copied().unwrap_or(0) as f32 / 255.0
                + 0.587 * f.get(b + 1).copied().unwrap_or(0) as f32 / 255.0
                + 0.114 * f.get(b + 2).copied().unwrap_or(0) as f32 / 255.0
        };
        data.push((lum(frame_a) - lum(frame_b)).abs());
        data.push(0.0);
    }
    Ok(ImageMotionField {
        data,
        width,
        height,
    })
}

/// Compute statistics from an [`ImageMotionField`].
pub fn compute_image_motion_stats(field: &ImageMotionField) -> MotionStats {
    let n = (field.width as usize) * (field.height as usize);
    if n == 0 {
        return MotionStats {
            mean_magnitude: 0.0,
            max_magnitude: 0.0,
            min_magnitude: 0.0,
            std_magnitude: 0.0,
            fraction_moving: 0.0,
        };
    }
    let mags: Vec<f32> = field
        .data
        .chunks_exact(2)
        .map(|c| (c[0] * c[0] + c[1] * c[1]).sqrt())
        .collect();
    let sum: f32 = mags.iter().sum();
    let mean = sum / n as f32;
    let max_mag = mags.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_mag = mags.iter().cloned().fold(f32::INFINITY, f32::min);
    let var = mags
        .iter()
        .map(|&m| {
            let d = m - mean;
            d * d
        })
        .sum::<f32>()
        / n as f32;
    let moving = mags.iter().filter(|&&m| m > 0.5).count();
    MotionStats {
        mean_magnitude: mean,
        max_magnitude: if max_mag == f32::NEG_INFINITY {
            0.0
        } else {
            max_mag
        },
        min_magnitude: if min_mag == f32::INFINITY {
            0.0
        } else {
            min_mag
        },
        std_magnitude: var.sqrt(),
        fraction_moving: moving as f32 / n as f32,
    }
}

/// Blend a sequence of RGBA frames by equal-weight temporal accumulation.
///
/// # Errors
/// [`MotionBlurError::EmptyFrames`], [`MotionBlurError::InvalidDimensions`].
pub fn accumulate_motion_blur(
    frames: &[Vec<u8>],
    width: u32,
    height: u32,
    _config: &MotionBlurConfig,
) -> Result<Vec<u8>, MotionBlurError> {
    if frames.is_empty() {
        return Err(MotionBlurError::EmptyFrames);
    }
    for f in frames {
        validate_rgba(f, width, height)?;
    }
    let n_bytes = (width as usize) * (height as usize) * 4;
    let nf = frames.len() as f32;
    let mut acc = vec![0.0_f32; n_bytes];
    for f in frames {
        for (a, &b) in acc.iter_mut().zip(f.iter()) {
            *a += b as f32;
        }
    }
    Ok(acc
        .iter()
        .map(|&v| (v / nf).round().clamp(0.0, 255.0) as u8)
        .collect())
}

/// Format a human-readable summary of [`MotionStats`].
///
/// Example: `"MotionStats: mean=1.23px, max=12.34px, moving=34.5%"`
pub fn format_motion_stats(stats: &MotionStats) -> String {
    format!(
        "MotionStats: mean={:.2}px, max={:.2}px, moving={:.1}%",
        stats.mean_magnitude,
        stats.max_magnitude,
        stats.fraction_moving * 100.0,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (45+)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;
    fn ap(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }
    fn img(w: u32, h: u32, fill: u8) -> Vec<u8> {
        vec![fill; (w as usize) * (h as usize) * 4]
    }

    // ── MotionBlurConfig ──────────────────────────────────────────────────────

    #[test]
    fn test_config_default() {
        let c = MotionBlurConfig::default();
        assert_eq!(c.samples, 16);
        assert!(ap(c.shutter_angle, 180.0));
        assert!(ap(c.max_blur_pixels, 32.0));
        assert!(!c.use_alpha_weight);
    }

    #[test]
    fn test_config_validate_valid() {
        assert!(MotionBlurConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_samples() {
        let c = MotionBlurConfig {
            samples: 0,
            ..Default::default()
        };
        assert!(matches!(
            c.validate(),
            Err(MotionBlurError::InvalidSampleCount { samples: 0 })
        ));
    }

    #[test]
    fn test_config_validate_shutter_zero() {
        let c = MotionBlurConfig {
            shutter_angle: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            c.validate(),
            Err(MotionBlurError::InvalidShutterAngle { .. })
        ));
    }

    #[test]
    fn test_config_validate_shutter_over_360() {
        let c = MotionBlurConfig {
            shutter_angle: 361.0,
            ..Default::default()
        };
        assert!(matches!(
            c.validate(),
            Err(MotionBlurError::InvalidShutterAngle { .. })
        ));
    }

    #[test]
    fn test_config_validate_shutter_360_ok() {
        let c = MotionBlurConfig {
            shutter_angle: 360.0,
            ..Default::default()
        };
        assert!(c.validate().is_ok());
    }

    // ── ImageMotionField ──────────────────────────────────────────────────────

    #[test]
    fn test_field_new_valid() {
        assert!(ImageMotionField::new(vec![0.0; 4 * 3 * 2], 4, 3).is_ok());
    }

    #[test]
    fn test_field_new_mismatch() {
        let r = ImageMotionField::new(vec![0.0; 5], 4, 3);
        assert!(matches!(
            r,
            Err(MotionBlurError::MotionVectorMismatch { .. })
        ));
    }

    #[test]
    fn test_field_magnitude_at_known() {
        let f = ImageMotionField::uniform(4, 4, 3.0, 4.0);
        assert!(ap(f.magnitude_at(1, 1), 5.0));
    }

    #[test]
    fn test_field_magnitude_at_zero() {
        let f = ImageMotionField::zero(4, 4);
        assert!(ap(f.magnitude_at(2, 2), 0.0));
    }

    #[test]
    fn test_field_magnitude_at_oob() {
        let f = ImageMotionField::uniform(4, 4, 1.0, 0.0);
        assert_eq!(f.magnitude_at(10, 10), 0.0);
    }

    #[test]
    fn test_field_uniform_data() {
        let f = ImageMotionField::uniform(2, 2, 1.5, -2.0);
        assert_eq!(f.data.len(), 8);
        for i in 0..4 {
            assert!(ap(f.data[i * 2], 1.5));
            assert!(ap(f.data[i * 2 + 1], -2.0));
        }
    }

    #[test]
    fn test_field_zero_all_zeros() {
        let f = ImageMotionField::zero(3, 3);
        assert!(f.data.iter().all(|&v| v == 0.0));
    }

    // ── bilinear_sample_u8_rgba ───────────────────────────────────────────────

    #[test]
    fn test_bilinear_corner() {
        let px = vec![
            255u8, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 128, 128, 128, 255,
        ];
        let s = bilinear_sample_u8_rgba(&px, 2, 2, 0.0, 0.0);
        assert!(ap(s[0], 1.0) && ap(s[1], 0.0));
    }

    #[test]
    fn test_bilinear_oob_clamped() {
        let px = vec![100u8, 100, 100, 255, 200, 200, 200, 255];
        let s = bilinear_sample_u8_rgba(&px, 2, 1, 5.0, 0.0);
        assert!(ap(s[0], 200.0 / 255.0));
    }

    #[test]
    fn test_bilinear_center_2x2() {
        // 2×2 image, all same value → centre sample == that value
        let px = vec![128u8; 16];
        let s = bilinear_sample_u8_rgba(&px, 2, 2, 0.5, 0.5);
        assert!(ap(s[0], 128.0 / 255.0));
    }

    // ── sample_along_motion ───────────────────────────────────────────────────

    #[test]
    fn test_sample_single_returns_pixel() {
        let i = img(4, 4, 128);
        let r = sample_along_motion(
            &i,
            4,
            4,
            MotionSample {
                px: 2,
                py: 2,
                dx: 10.0,
                dy: 10.0,
            },
            1,
        );
        assert!(ap(r[0], 128.0 / 255.0));
    }

    #[test]
    fn test_sample_uniform_image_unchanged() {
        // Uniform image → any direction → same value
        let i = img(4, 4, 200);
        let r = sample_along_motion(
            &i,
            4,
            4,
            MotionSample {
                px: 1,
                py: 1,
                dx: 2.0,
                dy: 1.0,
            },
            8,
        );
        assert!(ap(r[0], 200.0 / 255.0));
    }

    // ── apply_linear_motion_blur ──────────────────────────────────────────────

    #[test]
    fn test_linear_zero_motion_identity() {
        let i = img(4, 4, 200);
        let cfg = MotionBlurConfig::default();
        assert_eq!(
            apply_linear_motion_blur(&i, 4, 4, 0.0, 0.0, &cfg).unwrap(),
            i
        );
    }

    #[test]
    fn test_linear_output_size() {
        let i = img(8, 6, 100);
        let cfg = MotionBlurConfig::default();
        assert_eq!(
            apply_linear_motion_blur(&i, 8, 6, 2.0, 1.0, &cfg)
                .unwrap()
                .len(),
            i.len()
        );
    }

    #[test]
    fn test_linear_empty_image_error() {
        let cfg = MotionBlurConfig::default();
        assert!(matches!(
            apply_linear_motion_blur(&[], 4, 4, 0.0, 0.0, &cfg),
            Err(MotionBlurError::EmptyImage)
        ));
    }

    #[test]
    fn test_linear_wrong_size_error() {
        let cfg = MotionBlurConfig::default();
        assert!(matches!(
            apply_linear_motion_blur(&[0u8; 10], 4, 4, 0.0, 0.0, &cfg),
            Err(MotionBlurError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn test_linear_full_shutter() {
        let i: Vec<u8> = (0..8 * 8 * 4).map(|v| (v % 255) as u8).collect();
        let cfg = MotionBlurConfig {
            shutter_angle: 360.0,
            samples: 8,
            ..Default::default()
        };
        assert_eq!(
            apply_linear_motion_blur(&i, 8, 8, 3.0, 0.0, &cfg)
                .unwrap()
                .len(),
            i.len()
        );
    }

    // ── apply_radial_blur ─────────────────────────────────────────────────────

    #[test]
    fn test_radial_output_size() {
        let i = img(6, 6, 128);
        let cfg = MotionBlurConfig {
            samples: 4,
            ..Default::default()
        };
        assert_eq!(
            apply_radial_blur(&i, 6, 6, 3.0, 3.0, 0.1, &cfg)
                .unwrap()
                .len(),
            i.len()
        );
    }

    #[test]
    fn test_radial_smoke() {
        let i: Vec<u8> = (0..4 * 4 * 4).map(|v| (v % 256) as u8).collect();
        let cfg = MotionBlurConfig {
            samples: 4,
            ..Default::default()
        };
        assert_eq!(
            apply_radial_blur(&i, 4, 4, 2.0, 2.0, 0.5, &cfg)
                .unwrap()
                .len(),
            i.len()
        );
    }

    #[test]
    fn test_radial_empty_error() {
        let cfg = MotionBlurConfig::default();
        assert!(matches!(
            apply_radial_blur(&[], 4, 4, 0.0, 0.0, 1.0, &cfg),
            Err(MotionBlurError::EmptyImage)
        ));
    }

    // ── apply_rotational_blur ─────────────────────────────────────────────────

    #[test]
    fn test_rotational_output_size() {
        let i = img(6, 6, 64);
        let cfg = MotionBlurConfig {
            samples: 4,
            ..Default::default()
        };
        assert_eq!(
            apply_rotational_blur(&i, 6, 6, 3.0, 3.0, 10.0, &cfg)
                .unwrap()
                .len(),
            i.len()
        );
    }

    #[test]
    fn test_rotational_smoke() {
        let i: Vec<u8> = (0..4 * 4 * 4).map(|v| (v % 200) as u8).collect();
        let cfg = MotionBlurConfig {
            samples: 4,
            ..Default::default()
        };
        assert_eq!(
            apply_rotational_blur(&i, 4, 4, 2.0, 2.0, 5.0, &cfg)
                .unwrap()
                .len(),
            i.len()
        );
    }

    // ── apply_per_pixel_motion_blur ───────────────────────────────────────────

    #[test]
    fn test_per_pixel_zero_field_identity() {
        let i = img(4, 4, 150);
        let f = ImageMotionField::zero(4, 4);
        let cfg = MotionBlurConfig::default();
        assert_eq!(apply_per_pixel_motion_blur(&i, 4, 4, &f, &cfg).unwrap(), i);
    }

    #[test]
    fn test_per_pixel_output_size() {
        let i = img(5, 5, 100);
        let f = ImageMotionField::uniform(5, 5, 2.0, 0.0);
        let cfg = MotionBlurConfig {
            samples: 4,
            ..Default::default()
        };
        assert_eq!(
            apply_per_pixel_motion_blur(&i, 5, 5, &f, &cfg)
                .unwrap()
                .len(),
            i.len()
        );
    }

    #[test]
    fn test_per_pixel_field_dim_mismatch() {
        let i = img(4, 4, 0);
        let f = ImageMotionField::zero(5, 5);
        let cfg = MotionBlurConfig::default();
        assert!(matches!(
            apply_per_pixel_motion_blur(&i, 4, 4, &f, &cfg),
            Err(MotionBlurError::MotionVectorMismatch { .. })
        ));
    }

    #[test]
    fn test_per_pixel_nonzero_field() {
        let i: Vec<u8> = (0..4 * 4 * 4).map(|v| (v % 200) as u8).collect();
        let f = ImageMotionField::uniform(4, 4, 1.5, 0.5);
        let cfg = MotionBlurConfig {
            samples: 4,
            ..Default::default()
        };
        assert_eq!(
            apply_per_pixel_motion_blur(&i, 4, 4, &f, &cfg)
                .unwrap()
                .len(),
            i.len()
        );
    }

    // ── apply_motion_blur (dispatcher) ────────────────────────────────────────

    #[test]
    fn test_dispatch_linear() {
        let i = img(4, 4, 200);
        let cfg = MotionBlurConfig::default();
        let out = apply_motion_blur(&i, 4, 4, BlurType::Linear, &cfg, &[1.0, 0.0]).unwrap();
        assert_eq!(out.len(), i.len());
    }

    #[test]
    fn test_dispatch_radial() {
        let i = img(4, 4, 100);
        let cfg = MotionBlurConfig {
            samples: 4,
            ..Default::default()
        };
        let out = apply_motion_blur(&i, 4, 4, BlurType::Radial, &cfg, &[2.0, 2.0, 0.1]).unwrap();
        assert_eq!(out.len(), i.len());
    }

    #[test]
    fn test_dispatch_rotational() {
        let i = img(4, 4, 100);
        let cfg = MotionBlurConfig {
            samples: 4,
            ..Default::default()
        };
        let out =
            apply_motion_blur(&i, 4, 4, BlurType::Rotational, &cfg, &[2.0, 2.0, 5.0]).unwrap();
        assert_eq!(out.len(), i.len());
    }

    #[test]
    fn test_dispatch_per_pixel_error() {
        let i = img(4, 4, 0);
        let cfg = MotionBlurConfig::default();
        assert!(matches!(
            apply_motion_blur(&i, 4, 4, BlurType::PerPixel, &cfg, &[]),
            Err(MotionBlurError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_dispatch_linear_wrong_params() {
        let i = img(4, 4, 0);
        let cfg = MotionBlurConfig::default();
        assert!(matches!(
            apply_motion_blur(&i, 4, 4, BlurType::Linear, &cfg, &[1.0]),
            Err(MotionBlurError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_dispatch_radial_wrong_params() {
        let i = img(4, 4, 0);
        let cfg = MotionBlurConfig::default();
        assert!(matches!(
            apply_motion_blur(&i, 4, 4, BlurType::Radial, &cfg, &[1.0, 2.0]),
            Err(MotionBlurError::InvalidConfig(_))
        ));
    }

    // ── compute_motion_from_frames ────────────────────────────────────────────

    #[test]
    fn test_frames_same_zero_diff() {
        let f = img(4, 4, 128);
        let field = compute_motion_from_frames(&f, &f, 4, 4).unwrap();
        assert!(field.data.iter().all(|&v| ap(v, 0.0)));
    }

    #[test]
    fn test_frames_length_mismatch() {
        let fa = img(4, 4, 0);
        let bad = vec![0u8; 5];
        assert!(matches!(
            compute_motion_from_frames(&fa, &bad, 4, 4),
            Err(MotionBlurError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn test_frames_output_field_dimensions() {
        let fa = img(5, 3, 100);
        let fb = img(5, 3, 200);
        let field = compute_motion_from_frames(&fa, &fb, 5, 3).unwrap();
        assert_eq!(field.width, 5);
        assert_eq!(field.height, 3);
        assert_eq!(field.data.len(), 5 * 3 * 2);
    }

    // ── compute_image_motion_stats ────────────────────────────────────────────

    #[test]
    fn test_stats_zero_field() {
        let f = ImageMotionField::zero(4, 4);
        let s = compute_image_motion_stats(&f);
        assert!(ap(s.mean_magnitude, 0.0) && ap(s.max_magnitude, 0.0));
        assert!(ap(s.fraction_moving, 0.0));
    }

    #[test]
    fn test_stats_uniform_field() {
        let f = ImageMotionField::uniform(4, 4, 3.0, 4.0); // mag=5.0
        let s = compute_image_motion_stats(&f);
        assert!(ap(s.mean_magnitude, 5.0));
        assert!(ap(s.max_magnitude, 5.0));
        assert!(ap(s.min_magnitude, 5.0));
        assert!(ap(s.std_magnitude, 0.0));
        assert!(ap(s.fraction_moving, 1.0));
    }

    #[test]
    fn test_stats_empty_field() {
        let f = ImageMotionField::zero(0, 0);
        let s = compute_image_motion_stats(&f);
        assert!(ap(s.mean_magnitude, 0.0));
    }

    // ── accumulate_motion_blur ────────────────────────────────────────────────

    #[test]
    fn test_accumulate_single_identity() {
        let i = img(4, 4, 200);
        let cfg = MotionBlurConfig::default();
        assert_eq!(
            accumulate_motion_blur(std::slice::from_ref(&i), 4, 4, &cfg).unwrap(),
            i
        );
    }

    #[test]
    fn test_accumulate_two_identical() {
        let i = img(4, 4, 100);
        let cfg = MotionBlurConfig::default();
        assert_eq!(
            accumulate_motion_blur(&[i.clone(), i.clone()], 4, 4, &cfg).unwrap(),
            i
        );
    }

    #[test]
    fn test_accumulate_empty_error() {
        let cfg = MotionBlurConfig::default();
        assert!(matches!(
            accumulate_motion_blur(&[], 4, 4, &cfg),
            Err(MotionBlurError::EmptyFrames)
        ));
    }

    #[test]
    fn test_accumulate_average_two_diff() {
        // frame A = all 0, frame B = all 100 → avg = 50
        let fa = img(2, 2, 0);
        let fb = img(2, 2, 100);
        let cfg = MotionBlurConfig::default();
        let out = accumulate_motion_blur(&[fa, fb], 2, 2, &cfg).unwrap();
        assert!(out.iter().all(|&v| v == 50));
    }

    // ── format_motion_stats ───────────────────────────────────────────────────

    #[test]
    fn test_format_non_empty() {
        let f = ImageMotionField::uniform(4, 4, 3.0, 4.0);
        let s = compute_image_motion_stats(&f);
        let str = format_motion_stats(&s);
        assert!(str.contains("MotionStats"));
        assert!(str.contains("mean="));
        assert!(str.contains("max="));
        assert!(str.contains("moving="));
    }

    // ── MotionBlurError variants ──────────────────────────────────────────────

    #[test]
    fn test_error_empty_image() {
        assert!(!MotionBlurError::EmptyImage.to_string().is_empty());
    }

    #[test]
    fn test_error_invalid_dimensions() {
        let e = MotionBlurError::InvalidDimensions {
            actual: 10,
            expected: 64,
            width: 4,
            height: 4,
        };
        assert!(e.to_string().contains("10"));
    }

    #[test]
    fn test_error_motion_vector_mismatch() {
        let e = MotionBlurError::MotionVectorMismatch {
            mv_len: 5,
            px_len: 16,
        };
        assert!(e.to_string().contains("5"));
    }

    #[test]
    fn test_error_invalid_sample_count() {
        let e = MotionBlurError::InvalidSampleCount { samples: 0 };
        assert!(e.to_string().contains("0"));
    }

    #[test]
    fn test_error_invalid_shutter_angle() {
        let e = MotionBlurError::InvalidShutterAngle { angle: -10.0 };
        assert!(e.to_string().contains("-10"));
    }

    // ── BlurType variants ─────────────────────────────────────────────────────

    #[test]
    fn test_blur_type_eq() {
        assert_eq!(BlurType::Linear, BlurType::Linear);
        assert_ne!(BlurType::Linear, BlurType::Radial);
        assert_ne!(BlurType::Rotational, BlurType::PerPixel);
    }

    #[test]
    fn test_blur_type_copy() {
        let b = BlurType::Radial;
        let c = b; // Copy
        assert_eq!(b, c);
    }
}
