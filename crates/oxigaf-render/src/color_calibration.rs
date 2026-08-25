//! CPU-side color calibration tools for 3D Gaussian Splatting rendered images.
//!
//! Provides white balance, color correction matrices, gamma, saturation, and
//! gamut operations on flat RGBA `Vec<u8>` buffers (width × height × 4 bytes).
//! All corrections operate on linear f32 `[0, 1]` internally.

use thiserror::Error;

// ─── Error Type ──────────────────────────────────────────────────────────────

/// Errors that can occur during color calibration operations.
#[derive(Debug, Error)]
pub enum CalibrationError {
    /// Buffer is empty.
    #[error("Empty image")]
    EmptyImage,

    /// Buffer size does not match the declared dimensions.
    #[error("Image buffer {actual} bytes does not match {width}×{height}×4 = {expected}")]
    BufferSizeMismatch {
        /// Actual byte length of the buffer.
        actual: usize,
        /// Expected byte length (width × height × 4).
        expected: usize,
        /// Declared image width.
        width: u32,
        /// Declared image height.
        height: u32,
    },

    /// Color temperature outside the valid range.
    #[error("Invalid color temperature {kelvin}K: must be in [1000, 20000]")]
    InvalidColorTemperature {
        /// The invalid kelvin value.
        kelvin: f32,
    },

    /// Gamma value is not positive.
    #[error("Invalid gamma {gamma}: must be > 0")]
    InvalidGamma {
        /// The invalid gamma value.
        gamma: f32,
    },

    /// Matrix is singular and cannot be inverted.
    #[error("Singular matrix: cannot invert color correction matrix")]
    SingularMatrix,

    /// Reference image dimensions differ from source.
    #[error("Reference image has different dimensions from source")]
    DimensionMismatch,
}

// ─── ColorMatrix ─────────────────────────────────────────────────────────────

/// 3×3 color correction matrix stored in row-major order (`data[row][col]`).
#[derive(Debug, Clone, Copy)]
pub struct ColorMatrix {
    /// Row-major entries: `data[row][col]`.
    pub data: [[f32; 3]; 3],
}

impl ColorMatrix {
    /// Identity matrix.
    pub fn identity() -> Self {
        Self {
            data: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Construct from three explicit rows.
    pub fn from_rows(r0: [f32; 3], r1: [f32; 3], r2: [f32; 3]) -> Self {
        Self { data: [r0, r1, r2] }
    }

    /// Apply matrix to an RGB triplet: `out = M * in`.
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let m = &self.data;
        [
            m[0][0] * rgb[0] + m[0][1] * rgb[1] + m[0][2] * rgb[2],
            m[1][0] * rgb[0] + m[1][1] * rgb[1] + m[1][2] * rgb[2],
            m[2][0] * rgb[0] + m[2][1] * rgb[1] + m[2][2] * rgb[2],
        ]
    }

    /// Matrix multiplication: `self * other`.
    pub fn mul(&self, other: &ColorMatrix) -> ColorMatrix {
        let a = &self.data;
        let b = &other.data;
        let mut out = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        ColorMatrix { data: out }
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> ColorMatrix {
        let m = &self.data;
        ColorMatrix {
            data: [
                [m[0][0], m[1][0], m[2][0]],
                [m[0][1], m[1][1], m[2][1]],
                [m[0][2], m[1][2], m[2][2]],
            ],
        }
    }

    /// Compute the determinant via cofactor expansion.
    pub fn determinant(&self) -> f32 {
        let m = &self.data;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    /// Compute the inverse, returning `None` when |det| < 1e-8.
    pub fn inverse(&self) -> Option<ColorMatrix> {
        let det = self.determinant();
        if det.abs() < 1e-8 {
            return None;
        }
        let inv_det = 1.0 / det;
        let m = &self.data;

        let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
        let c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
        let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];

        let c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
        let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
        let c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);

        let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
        let c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
        let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];

        // Adjugate is the transpose of the cofactor matrix.
        Some(ColorMatrix {
            data: [
                [c00 * inv_det, c10 * inv_det, c20 * inv_det],
                [c01 * inv_det, c11 * inv_det, c21 * inv_det],
                [c02 * inv_det, c12 * inv_det, c22 * inv_det],
            ],
        })
    }

    /// Diagonal (scale) matrix: `diag(r, g, b)`.
    pub fn scale(r: f32, g: f32, b: f32) -> Self {
        Self {
            data: [[r, 0.0, 0.0], [0.0, g, 0.0], [0.0, 0.0, b]],
        }
    }
}

// ─── WhiteBalance ────────────────────────────────────────────────────────────

/// White balance settings.
#[derive(Debug, Clone, Copy)]
pub struct WhiteBalance {
    /// Color temperature in Kelvin (default: 6500).
    pub temperature: f32,
    /// Green/magenta tint in `[-1, +1]`; positive = more green (default: 0).
    pub tint: f32,
    /// Exposure compensation in stops (default: 0.0).
    pub exposure: f32,
}

impl Default for WhiteBalance {
    fn default() -> Self {
        WhiteBalance {
            temperature: 6500.0,
            tint: 0.0,
            exposure: 0.0,
        }
    }
}

// ─── ColorCalibrationConfig ───────────────────────────────────────────────────

/// Full color calibration configuration combining multiple correction stages.
///
/// Pipeline order: white balance → color matrix → saturation → gamma → brightness.
/// White balance, the optional color matrix, and saturation are all linear
/// operations (see [`saturation_matrix`]'s "operating in linear RGB space"),
/// so they are composed into a single matrix and applied together, in
/// linear light, before the nonlinear gamma encode.
#[derive(Debug, Clone)]
pub struct ColorCalibrationConfig {
    /// White balance adjustment.
    pub white_balance: WhiteBalance,
    /// Optional 3×3 color correction matrix applied after white balance.
    pub color_matrix: Option<ColorMatrix>,
    /// Output gamma (1.0 = linear, 2.2 ≈ sRGB; default: 1.0).
    pub gamma: f32,
    /// Saturation multiplier (1.0 = unchanged; default: 1.0).
    pub saturation: f32,
    /// Brightness multiplier applied to all channels (default: 1.0).
    pub brightness: f32,
}

impl Default for ColorCalibrationConfig {
    fn default() -> Self {
        ColorCalibrationConfig {
            white_balance: WhiteBalance::default(),
            color_matrix: None,
            gamma: 1.0,
            saturation: 1.0,
            brightness: 1.0,
        }
    }
}

// ─── ColorStats ──────────────────────────────────────────────────────────────

/// Statistical summary of an image's color distribution.
#[derive(Debug, Clone)]
pub struct ColorStats {
    /// Mean red channel value in [0, 1].
    pub mean_r: f32,
    /// Mean green channel value in [0, 1].
    pub mean_g: f32,
    /// Mean blue channel value in [0, 1].
    pub mean_b: f32,
    /// Standard deviation of the red channel.
    pub std_r: f32,
    /// Standard deviation of the green channel.
    pub std_g: f32,
    /// Standard deviation of the blue channel.
    pub std_b: f32,
    /// Estimated white point (maximum of each channel).
    pub white_point: [f32; 3],
    /// Gray world mean (mean of each channel).
    pub gray_point: [f32; 3],
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Validate that a pixel buffer matches the declared dimensions.
fn validate_buffer(pixels: &[u8], width: u32, height: u32) -> Result<(), CalibrationError> {
    let expected = (width as usize) * (height as usize) * 4;
    if pixels.is_empty() {
        return Err(CalibrationError::EmptyImage);
    }
    if pixels.len() != expected {
        return Err(CalibrationError::BufferSizeMismatch {
            actual: pixels.len(),
            expected,
            width,
            height,
        });
    }
    Ok(())
}

// ─── Free Functions ───────────────────────────────────────────────────────────

/// Convert color temperature (Kelvin) to RGB multipliers using Tanner Helland's
/// piecewise polynomial approximation.
///
/// Temperature must be in `[1000, 20000]`. Returns `[r, g, b]` multipliers
/// normalized so `max(r, g, b) = 1.0`.
pub fn kelvin_to_rgb_multipliers(kelvin: f32) -> Result<[f32; 3], CalibrationError> {
    if !(1000.0..=20000.0).contains(&kelvin) {
        return Err(CalibrationError::InvalidColorTemperature { kelvin });
    }

    // Algorithm operates in units of 100K.
    let k = kelvin / 100.0;

    // ── Red ──────────────────────────────────────────────────────────────────
    let r = if k <= 66.0 {
        255.0_f32
    } else {
        (329.698_73 * (k - 60.0).powf(-0.133_204_76)).clamp(0.0, 255.0)
    };

    // ── Green ─────────────────────────────────────────────────────────────────
    let g = if k <= 66.0 {
        (99.470_8 * k.ln() - 161.119_57).clamp(0.0, 255.0)
    } else {
        (288.122_16 * (k - 60.0).powf(-0.075_514_846)).clamp(0.0, 255.0)
    };

    // ── Blue ──────────────────────────────────────────────────────────────────
    let b = if k >= 66.0 {
        255.0_f32
    } else if k <= 19.0 {
        0.0_f32
    } else {
        (138.517_73 * (k - 10.0).ln() - 305.044_8).clamp(0.0, 255.0)
    };

    // Normalize to [0, 1] and then so max = 1.0.
    let r_n = r / 255.0;
    let g_n = g / 255.0;
    let b_n = b / 255.0;

    let max_val = r_n.max(g_n).max(b_n);
    if max_val < 1e-9 {
        // Fallback: equal weights.
        return Ok([1.0, 1.0, 1.0]);
    }
    Ok([r_n / max_val, g_n / max_val, b_n / max_val])
}

/// Build a white balance matrix from [`WhiteBalance`] settings.
///
/// Steps:
/// 1. Convert temperature to RGB multipliers via [`kelvin_to_rgb_multipliers`].
/// 2. Apply tint: `g *= 1 + tint`.
/// 3. Apply exposure: multiply all channels by `2^exposure`.
/// 4. Return a diagonal [`ColorMatrix`].
pub fn white_balance_matrix(wb: &WhiteBalance) -> Result<ColorMatrix, CalibrationError> {
    let [mut r, mut g, mut b] = kelvin_to_rgb_multipliers(wb.temperature)?;
    g *= 1.0 + wb.tint;
    let exposure_factor = (2.0_f32).powf(wb.exposure);
    r *= exposure_factor;
    g *= exposure_factor;
    b *= exposure_factor;
    Ok(ColorMatrix::scale(r, g, b))
}

/// Convert linear RGB to luminance using the ITU-R BT.709 coefficients.
#[inline]
pub fn rgb_to_luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Build a saturation adjustment matrix operating in linear RGB space.
///
/// At `saturation = 0` the matrix reduces every pixel to its luminance.
/// At `saturation = 1` it is the identity.
///
/// `M[i][j] = lum[j] * (1 - s) + if i == j { s } else { 0 }`
pub fn saturation_matrix(saturation: f32) -> ColorMatrix {
    let s = saturation;
    let lum = [0.2126_f32, 0.7152_f32, 0.0722_f32];
    let mut data = [[0.0f32; 3]; 3];
    for (i, row) in data.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let diag = if i == j { s } else { 0.0 };
            *cell = lum[j] * (1.0 - s) + diag;
        }
    }
    ColorMatrix { data }
}

/// Apply gamma correction to a single linear f32 value in `[0, 1]`.
///
/// `out = max(x, 0)^(1/gamma)`
#[inline]
pub fn apply_gamma_f32(x: f32, gamma: f32) -> f32 {
    x.max(0.0).powf(1.0 / gamma)
}

/// Compute color statistics from an RGBA image.
pub fn compute_color_stats(
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Result<ColorStats, CalibrationError> {
    validate_buffer(pixels, width, height)?;

    let n = (width as usize) * (height as usize);
    let mut sum_r = 0.0_f64;
    let mut sum_g = 0.0_f64;
    let mut sum_b = 0.0_f64;
    let mut max_r = 0.0_f32;
    let mut max_g = 0.0_f32;
    let mut max_b = 0.0_f32;

    for chunk in pixels.chunks_exact(4) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        sum_r += r as f64;
        sum_g += g as f64;
        sum_b += b as f64;
        if r > max_r {
            max_r = r;
        }
        if g > max_g {
            max_g = g;
        }
        if b > max_b {
            max_b = b;
        }
    }

    let mean_r = (sum_r / n as f64) as f32;
    let mean_g = (sum_g / n as f64) as f32;
    let mean_b = (sum_b / n as f64) as f32;

    let mut var_r = 0.0_f64;
    let mut var_g = 0.0_f64;
    let mut var_b = 0.0_f64;

    for chunk in pixels.chunks_exact(4) {
        let r = chunk[0] as f64 / 255.0;
        let g = chunk[1] as f64 / 255.0;
        let b = chunk[2] as f64 / 255.0;
        var_r += (r - mean_r as f64).powi(2);
        var_g += (g - mean_g as f64).powi(2);
        var_b += (b - mean_b as f64).powi(2);
    }

    let std_r = ((var_r / n as f64) as f32).sqrt();
    let std_g = ((var_g / n as f64) as f32).sqrt();
    let std_b = ((var_b / n as f64) as f32).sqrt();

    Ok(ColorStats {
        mean_r,
        mean_g,
        mean_b,
        std_r,
        std_g,
        std_b,
        white_point: [max_r, max_g, max_b],
        gray_point: [mean_r, mean_g, mean_b],
    })
}

/// Compute a gray-world white balance matrix for an RGBA image.
///
/// Derives per-channel scale factors so that the mean of each channel equals
/// the overall mean gray, then normalizes so the largest scale factor is 1.0.
pub fn gray_world_white_balance(
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Result<ColorMatrix, CalibrationError> {
    validate_buffer(pixels, width, height)?;

    let n = (width as usize) * (height as usize);
    let mut sum_r = 0.0_f64;
    let mut sum_g = 0.0_f64;
    let mut sum_b = 0.0_f64;

    for chunk in pixels.chunks_exact(4) {
        sum_r += chunk[0] as f64;
        sum_g += chunk[1] as f64;
        sum_b += chunk[2] as f64;
    }

    let mean_r = (sum_r / n as f64) as f32 / 255.0;
    let mean_g = (sum_g / n as f64) as f32 / 255.0;
    let mean_b = (sum_b / n as f64) as f32 / 255.0;

    let gray = (mean_r + mean_g + mean_b) / 3.0;
    if gray < 1e-9 {
        return Ok(ColorMatrix::identity());
    }

    let wr = gray / mean_r.max(1e-9);
    let wg = gray / mean_g.max(1e-9);
    let wb = gray / mean_b.max(1e-9);

    let max_w = wr.max(wg).max(wb);
    Ok(ColorMatrix::scale(wr / max_w, wg / max_w, wb / max_w))
}

/// Apply white balance to an RGBA image, returning the corrected buffer.
///
/// Alpha channel is preserved unchanged.
pub fn apply_white_balance_image(
    pixels: &[u8],
    width: u32,
    height: u32,
    wb: &WhiteBalance,
) -> Result<Vec<u8>, CalibrationError> {
    validate_buffer(pixels, width, height)?;
    let matrix = white_balance_matrix(wb)?;
    apply_color_correction_matrix(pixels, width, height, &matrix)
}

/// Apply a 3×3 color correction matrix to an RGBA image.
///
/// RGB channels are converted to f32, the matrix is applied, values are clamped
/// to `[0, 1]`, and then converted back to u8. Alpha is preserved.
pub fn apply_color_correction_matrix(
    pixels: &[u8],
    width: u32,
    height: u32,
    matrix: &ColorMatrix,
) -> Result<Vec<u8>, CalibrationError> {
    validate_buffer(pixels, width, height)?;

    let mut out = Vec::with_capacity(pixels.len());
    for chunk in pixels.chunks_exact(4) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        let a = chunk[3];

        let [nr, ng, nb] = matrix.apply([r, g, b]);
        out.push((nr.clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((ng.clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((nb.clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push(a);
    }
    Ok(out)
}

/// Apply gamma correction to an RGBA image (RGB channels only; alpha unchanged).
pub fn apply_gamma_correction_image(
    pixels: &[u8],
    width: u32,
    height: u32,
    gamma: f32,
) -> Result<Vec<u8>, CalibrationError> {
    if gamma <= 0.0 {
        return Err(CalibrationError::InvalidGamma { gamma });
    }
    validate_buffer(pixels, width, height)?;

    let mut out = Vec::with_capacity(pixels.len());
    for chunk in pixels.chunks_exact(4) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        let a = chunk[3];

        out.push((apply_gamma_f32(r, gamma) * 255.0).round() as u8);
        out.push((apply_gamma_f32(g, gamma) * 255.0).round() as u8);
        out.push((apply_gamma_f32(b, gamma) * 255.0).round() as u8);
        out.push(a);
    }
    Ok(out)
}

/// Apply saturation adjustment to an RGBA image.
pub fn apply_saturation_image(
    pixels: &[u8],
    width: u32,
    height: u32,
    saturation: f32,
) -> Result<Vec<u8>, CalibrationError> {
    validate_buffer(pixels, width, height)?;
    let matrix = saturation_matrix(saturation);
    apply_color_correction_matrix(pixels, width, height, &matrix)
}

/// Apply the full calibration pipeline to an RGBA image.
///
/// Order of operations: white balance → color matrix → saturation → gamma → brightness.
/// The first three stages are linear (matrix) operations and are composed
/// into one matrix applied in a single pass, still ahead of the nonlinear
/// gamma step — see [`ColorCalibrationConfig`]'s doc.
pub fn apply_calibration(
    pixels: &[u8],
    width: u32,
    height: u32,
    config: &ColorCalibrationConfig,
) -> Result<Vec<u8>, CalibrationError> {
    validate_buffer(pixels, width, height)?;

    if config.gamma <= 0.0 {
        return Err(CalibrationError::InvalidGamma {
            gamma: config.gamma,
        });
    }

    // Stage 1: white balance.
    let wb_matrix = white_balance_matrix(&config.white_balance)?;

    // Stage 2: optional additional color matrix (compose with WB matrix).
    let combined = match &config.color_matrix {
        Some(cm) => cm.mul(&wb_matrix),
        None => wb_matrix,
    };

    // Compose saturation matrix.
    let sat_matrix = saturation_matrix(config.saturation);
    let full_matrix = sat_matrix.mul(&combined);

    // Apply combined matrix, then gamma, then brightness.
    let mut out = Vec::with_capacity(pixels.len());
    let gamma = config.gamma;
    let brightness = config.brightness;

    for chunk in pixels.chunks_exact(4) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        let a = chunk[3];

        let [mut nr, mut ng, mut nb] = full_matrix.apply([r, g, b]);

        // Gamma.
        if (gamma - 1.0).abs() > 1e-6 {
            nr = apply_gamma_f32(nr.clamp(0.0, 1.0), gamma);
            ng = apply_gamma_f32(ng.clamp(0.0, 1.0), gamma);
            nb = apply_gamma_f32(nb.clamp(0.0, 1.0), gamma);
        }

        // Brightness.
        nr *= brightness;
        ng *= brightness;
        nb *= brightness;

        out.push((nr.clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((ng.clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push((nb.clamp(0.0, 1.0) * 255.0).round() as u8);
        out.push(a);
    }
    Ok(out)
}

/// Compute a per-channel scale correction matrix to match source colors to a reference.
///
/// Returns `ColorMatrix::scale(ref_r/src_r, ref_g/src_g, ref_b/src_b)`.
/// Channels where the source mean is below 0.01 use a scale of 1.0.
pub fn compute_correction_matrix(
    source: &[u8],
    reference: &[u8],
    width: u32,
    height: u32,
) -> Result<ColorMatrix, CalibrationError> {
    validate_buffer(source, width, height)?;
    validate_buffer(reference, width, height)?;

    if source.len() != reference.len() {
        return Err(CalibrationError::DimensionMismatch);
    }

    let n = (width as usize) * (height as usize);
    let mut src_r = 0.0_f64;
    let mut src_g = 0.0_f64;
    let mut src_b = 0.0_f64;
    let mut ref_r = 0.0_f64;
    let mut ref_g = 0.0_f64;
    let mut ref_b = 0.0_f64;

    for (s, r) in source.chunks_exact(4).zip(reference.chunks_exact(4)) {
        src_r += s[0] as f64;
        src_g += s[1] as f64;
        src_b += s[2] as f64;
        ref_r += r[0] as f64;
        ref_g += r[1] as f64;
        ref_b += r[2] as f64;
    }

    let mean_src_r = (src_r / n as f64) as f32 / 255.0;
    let mean_src_g = (src_g / n as f64) as f32 / 255.0;
    let mean_src_b = (src_b / n as f64) as f32 / 255.0;
    let mean_ref_r = (ref_r / n as f64) as f32 / 255.0;
    let mean_ref_g = (ref_g / n as f64) as f32 / 255.0;
    let mean_ref_b = (ref_b / n as f64) as f32 / 255.0;

    let scale_r = if mean_src_r < 0.01 {
        1.0
    } else {
        mean_ref_r / mean_src_r
    };
    let scale_g = if mean_src_g < 0.01 {
        1.0
    } else {
        mean_ref_g / mean_src_g
    };
    let scale_b = if mean_src_b < 0.01 {
        1.0
    } else {
        mean_ref_b / mean_src_b
    };

    Ok(ColorMatrix::scale(scale_r, scale_g, scale_b))
}

/// Encode a linear light value to sRGB.
///
/// `x ≤ 0.0031308 → 12.92 * x`; otherwise `1.055 * x^(1/2.4) - 0.055`.
#[inline]
pub fn linear_to_srgb_calib(x: f32) -> f32 {
    if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// Decode an sRGB value to linear light.
///
/// `x ≤ 0.04045 → x / 12.92`; otherwise `((x + 0.055) / 1.055)^2.4`.
#[inline]
pub fn srgb_to_linear_calib(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

/// Apply sRGB encoding to an RGBA image (linear input → sRGB output).
///
/// Alpha channel is preserved unchanged.
pub fn apply_srgb_encoding(
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, CalibrationError> {
    validate_buffer(pixels, width, height)?;

    let mut out = Vec::with_capacity(pixels.len());
    for chunk in pixels.chunks_exact(4) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        let a = chunk[3];

        out.push((linear_to_srgb_calib(r.clamp(0.0, 1.0)) * 255.0).round() as u8);
        out.push((linear_to_srgb_calib(g.clamp(0.0, 1.0)) * 255.0).round() as u8);
        out.push((linear_to_srgb_calib(b.clamp(0.0, 1.0)) * 255.0).round() as u8);
        out.push(a);
    }
    Ok(out)
}

/// Apply sRGB decoding to an RGBA image (sRGB input → linear output).
///
/// Alpha channel is preserved unchanged.
pub fn apply_srgb_decoding(
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, CalibrationError> {
    validate_buffer(pixels, width, height)?;

    let mut out = Vec::with_capacity(pixels.len());
    for chunk in pixels.chunks_exact(4) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        let a = chunk[3];

        out.push((srgb_to_linear_calib(r.clamp(0.0, 1.0)) * 255.0).round() as u8);
        out.push((srgb_to_linear_calib(g.clamp(0.0, 1.0)) * 255.0).round() as u8);
        out.push((srgb_to_linear_calib(b.clamp(0.0, 1.0)) * 255.0).round() as u8);
        out.push(a);
    }
    Ok(out)
}

/// Compute a per-channel histogram-stretch scale matrix.
///
/// For each channel, find `min` and `max`; the returned diagonal scale
/// factor `1 / (max - min)` stretches that channel to fill `[0, 1]`.
/// Channels with a flat range (max - min < 1e-6) use a scale of 1.0.
///
/// Note: only the scale is representable as a 3×3 matrix; the per-channel
/// minimum offset is not applied here.
pub fn histogram_stretch_matrix(
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Result<ColorMatrix, CalibrationError> {
    validate_buffer(pixels, width, height)?;

    let mut min_r = f32::MAX;
    let mut min_g = f32::MAX;
    let mut min_b = f32::MAX;
    let mut max_r = f32::MIN;
    let mut max_g = f32::MIN;
    let mut max_b = f32::MIN;

    for chunk in pixels.chunks_exact(4) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;

        if r < min_r {
            min_r = r;
        }
        if g < min_g {
            min_g = g;
        }
        if b < min_b {
            min_b = b;
        }
        if r > max_r {
            max_r = r;
        }
        if g > max_g {
            max_g = g;
        }
        if b > max_b {
            max_b = b;
        }
    }

    let range_r = max_r - min_r;
    let range_g = max_g - min_g;
    let range_b = max_b - min_b;

    let scale_r = if range_r < 1e-6 { 1.0 } else { 1.0 / range_r };
    let scale_g = if range_g < 1e-6 { 1.0 } else { 1.0 / range_g };
    let scale_b = if range_b < 1e-6 { 1.0 } else { 1.0 / range_b };

    Ok(ColorMatrix::scale(scale_r, scale_g, scale_b))
}

/// Build a D65 daylight white balance matrix (standard CIE illuminant, ~6504 K).
///
/// Returns approximately the identity matrix for sRGB-encoded images.
pub fn d65_white_balance() -> ColorMatrix {
    let wb = WhiteBalance {
        temperature: 6504.0,
        tint: 0.0,
        exposure: 0.0,
    };
    // 6504 K is within the valid range [1000, 20000]; the match arm covers the
    // unreachable error branch to satisfy clippy::unwrap_used.
    match white_balance_matrix(&wb) {
        Ok(m) => m,
        Err(_) => ColorMatrix::identity(),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// Create a uniform RGBA image with the given channel values.
    fn uniform_image(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let n = (w * h) as usize;
        let mut v = Vec::with_capacity(n * 4);
        for _ in 0..n {
            v.push(r);
            v.push(g);
            v.push(b);
            v.push(a);
        }
        v
    }

    // ── ColorMatrix tests ─────────────────────────────────────────────────────

    #[test]
    fn test_color_matrix_identity_apply() {
        let m = ColorMatrix::identity();
        let rgb = [0.3, 0.5, 0.8];
        let out = m.apply(rgb);
        assert!(approx(out[0], 0.3, 1e-6));
        assert!(approx(out[1], 0.5, 1e-6));
        assert!(approx(out[2], 0.8, 1e-6));
    }

    #[test]
    fn test_color_matrix_from_rows() {
        let m = ColorMatrix::from_rows([1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]);
        let out = m.apply([1.0, 1.0, 1.0]);
        assert!(approx(out[0], 1.0, 1e-6));
        assert!(approx(out[1], 2.0, 1e-6));
        assert!(approx(out[2], 3.0, 1e-6));
    }

    #[test]
    fn test_color_matrix_mul_identity() {
        let m = ColorMatrix::scale(2.0, 3.0, 4.0);
        let result = m.mul(&ColorMatrix::identity());
        let out = result.apply([1.0, 1.0, 1.0]);
        assert!(approx(out[0], 2.0, 1e-5));
        assert!(approx(out[1], 3.0, 1e-5));
        assert!(approx(out[2], 4.0, 1e-5));
    }

    #[test]
    fn test_color_matrix_transpose() {
        let m = ColorMatrix::from_rows([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]);
        let t = m.transpose();
        assert!(approx(t.data[0][1], 4.0, 1e-6));
        assert!(approx(t.data[1][0], 2.0, 1e-6));
        assert!(approx(t.data[2][0], 3.0, 1e-6));
    }

    #[test]
    fn test_color_matrix_scale() {
        let m = ColorMatrix::scale(0.5, 2.0, 1.0);
        let out = m.apply([1.0, 1.0, 1.0]);
        assert!(approx(out[0], 0.5, 1e-6));
        assert!(approx(out[1], 2.0, 1e-6));
        assert!(approx(out[2], 1.0, 1e-6));
    }

    #[test]
    fn test_color_matrix_determinant_identity() {
        let det = ColorMatrix::identity().determinant();
        assert!(approx(det, 1.0, 1e-6));
    }

    #[test]
    fn test_color_matrix_inverse_identity() {
        let inv = ColorMatrix::identity().inverse();
        assert!(inv.is_some());
        let out = inv.unwrap().apply([1.0, 0.5, 0.25]);
        assert!(approx(out[0], 1.0, 1e-5));
    }

    #[test]
    fn test_color_matrix_inverse_singular() {
        let m = ColorMatrix::from_rows([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]);
        assert!(m.inverse().is_none());
    }

    #[test]
    fn test_color_matrix_inverse_roundtrip() {
        let m = ColorMatrix::scale(2.0, 0.5, 4.0);
        let inv = m.inverse().unwrap();
        let composed = m.mul(&inv);
        let out = composed.apply([0.7, 0.4, 0.9]);
        assert!(approx(out[0], 0.7, 1e-5));
        assert!(approx(out[1], 0.4, 1e-5));
        assert!(approx(out[2], 0.9, 1e-5));
    }

    // ── kelvin_to_rgb_multipliers tests ──────────────────────────────────────

    #[test]
    fn test_kelvin_daylight_6500() {
        let rgb = kelvin_to_rgb_multipliers(6500.0).unwrap();
        // At 6500 K, green and blue dominate; max should be 1.0.
        let max = rgb[0].max(rgb[1]).max(rgb[2]);
        assert!(approx(max, 1.0, 1e-5));
        // All values in [0, 1].
        for &v in &rgb {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_kelvin_warm_2700() {
        let rgb = kelvin_to_rgb_multipliers(2700.0).unwrap();
        // Warm light: red should be stronger than blue.
        assert!(rgb[0] > rgb[2]);
        let max = rgb[0].max(rgb[1]).max(rgb[2]);
        assert!(approx(max, 1.0, 1e-5));
    }

    #[test]
    fn test_kelvin_cool_10000() {
        let rgb = kelvin_to_rgb_multipliers(10000.0).unwrap();
        // Cool light: blue stronger than red.
        assert!(rgb[2] > rgb[0]);
        let max = rgb[0].max(rgb[1]).max(rgb[2]);
        assert!(approx(max, 1.0, 1e-5));
    }

    #[test]
    fn test_kelvin_boundary_valid() {
        assert!(kelvin_to_rgb_multipliers(1000.0).is_ok());
        assert!(kelvin_to_rgb_multipliers(20000.0).is_ok());
    }

    #[test]
    fn test_kelvin_invalid() {
        assert!(kelvin_to_rgb_multipliers(500.0).is_err());
        assert!(kelvin_to_rgb_multipliers(25000.0).is_err());
    }

    // ── white_balance_matrix tests ────────────────────────────────────────────

    #[test]
    fn test_wb_matrix_neutral_near_identity() {
        // At 6500 K with no tint and no exposure, the matrix should be close to
        // a uniform scaling (max component = 1).
        let wb = WhiteBalance::default();
        let m = white_balance_matrix(&wb).unwrap();
        let out = m.apply([1.0, 1.0, 1.0]);
        // No channel should exceed 1.0 significantly (max multiplier is 1).
        let max_out = out[0].max(out[1]).max(out[2]);
        assert!(max_out <= 1.01);
    }

    #[test]
    fn test_wb_matrix_warm_temperature() {
        let wb = WhiteBalance {
            temperature: 3000.0,
            tint: 0.0,
            exposure: 0.0,
        };
        let m = white_balance_matrix(&wb).unwrap();
        let out = m.apply([1.0, 1.0, 1.0]);
        // Red multiplier >= blue multiplier for warm light.
        assert!(out[0] >= out[2]);
    }

    #[test]
    fn test_wb_matrix_exposure_plus_one() {
        let wb_base = WhiteBalance {
            temperature: 6500.0,
            tint: 0.0,
            exposure: 0.0,
        };
        let wb_exp = WhiteBalance {
            temperature: 6500.0,
            tint: 0.0,
            exposure: 1.0,
        };
        let m_base = white_balance_matrix(&wb_base).unwrap();
        let m_exp = white_balance_matrix(&wb_exp).unwrap();
        let out_base = m_base.apply([0.5, 0.5, 0.5]);
        let out_exp = m_exp.apply([0.5, 0.5, 0.5]);
        // +1 stop should double the output (before clamping).
        assert!(approx(out_exp[0], out_base[0] * 2.0, 1e-4));
    }

    #[test]
    fn test_wb_matrix_invalid_temperature() {
        let wb = WhiteBalance {
            temperature: 0.0,
            tint: 0.0,
            exposure: 0.0,
        };
        assert!(white_balance_matrix(&wb).is_err());
    }

    // ── rgb_to_luminance tests ────────────────────────────────────────────────

    #[test]
    fn test_luminance_white() {
        assert!(approx(rgb_to_luminance(1.0, 1.0, 1.0), 1.0, 1e-5));
    }

    #[test]
    fn test_luminance_black() {
        assert!(approx(rgb_to_luminance(0.0, 0.0, 0.0), 0.0, 1e-5));
    }

    #[test]
    fn test_luminance_pure_red() {
        assert!(approx(rgb_to_luminance(1.0, 0.0, 0.0), 0.2126, 1e-5));
    }

    // ── saturation_matrix tests ───────────────────────────────────────────────

    #[test]
    fn test_saturation_zero_is_grayscale() {
        let m = saturation_matrix(0.0);
        let rgb = [0.8, 0.3, 0.1];
        let out = m.apply(rgb);
        let lum = rgb_to_luminance(rgb[0], rgb[1], rgb[2]);
        assert!(approx(out[0], lum, 1e-5));
        assert!(approx(out[1], lum, 1e-5));
        assert!(approx(out[2], lum, 1e-5));
    }

    #[test]
    fn test_saturation_one_is_identity() {
        let m = saturation_matrix(1.0);
        let rgb = [0.4, 0.6, 0.9];
        let out = m.apply(rgb);
        assert!(approx(out[0], rgb[0], 1e-5));
        assert!(approx(out[1], rgb[1], 1e-5));
        assert!(approx(out[2], rgb[2], 1e-5));
    }

    #[test]
    fn test_saturation_half() {
        // Intermediate value: result should be between gray and original.
        let m = saturation_matrix(0.5);
        let rgb = [1.0, 0.0, 0.0];
        let out = m.apply(rgb);
        let lum = rgb_to_luminance(1.0, 0.0, 0.0);
        // red channel: between lum and 1.0.
        assert!(out[0] >= lum && out[0] <= 1.0);
    }

    // ── apply_gamma_f32 tests ─────────────────────────────────────────────────

    #[test]
    fn test_gamma_one_is_identity() {
        assert!(approx(apply_gamma_f32(0.5, 1.0), 0.5, 1e-6));
        assert!(approx(apply_gamma_f32(0.0, 1.0), 0.0, 1e-6));
        assert!(approx(apply_gamma_f32(1.0, 1.0), 1.0, 1e-6));
    }

    #[test]
    fn test_gamma_two_is_sqrt() {
        let val = 0.25_f32;
        assert!(approx(apply_gamma_f32(val, 2.0), val.sqrt(), 1e-6));
    }

    #[test]
    fn test_gamma_negative_clamped_to_zero() {
        // Negative input is clamped to 0 before pow.
        assert!(approx(apply_gamma_f32(-0.5, 2.0), 0.0, 1e-6));
    }

    // ── linear_to_srgb / srgb_to_linear tests ────────────────────────────────

    #[test]
    fn test_srgb_encode_decode_roundtrip() {
        for &v in &[0.0f32, 0.01, 0.1, 0.5, 0.9, 1.0] {
            let encoded = linear_to_srgb_calib(v);
            let decoded = srgb_to_linear_calib(encoded);
            assert!(approx(decoded, v, 1e-5), "roundtrip failed at {}", v);
        }
    }

    #[test]
    fn test_srgb_decode_encode_roundtrip() {
        for &v in &[0.0f32, 0.04, 0.2, 0.5, 0.8, 1.0] {
            let decoded = srgb_to_linear_calib(v);
            let encoded = linear_to_srgb_calib(decoded);
            assert!(approx(encoded, v, 1e-5), "roundtrip failed at {}", v);
        }
    }

    #[test]
    fn test_srgb_known_values() {
        // At x = 1.0 both should be 1.0; at x = 0.0 both should be 0.0.
        assert!(approx(linear_to_srgb_calib(1.0), 1.0, 1e-5));
        assert!(approx(srgb_to_linear_calib(0.0), 0.0, 1e-5));
    }

    #[test]
    fn test_srgb_midpoint_lt_linear() {
        // sRGB encoding raises values: encoded(0.5) > 0.5 in sRGB.
        assert!(linear_to_srgb_calib(0.5) > 0.5);
        // And decoding lowers: linear(0.5) < 0.5.
        assert!(srgb_to_linear_calib(0.5) < 0.5);
    }

    // ── compute_color_stats tests ─────────────────────────────────────────────

    #[test]
    fn test_color_stats_uniform() {
        let img = uniform_image(4, 4, 128, 64, 200, 255);
        let stats = compute_color_stats(&img, 4, 4).unwrap();
        assert!(approx(stats.mean_r, 128.0 / 255.0, 1e-4));
        assert!(approx(stats.mean_g, 64.0 / 255.0, 1e-4));
        assert!(approx(stats.mean_b, 200.0 / 255.0, 1e-4));
        assert!(approx(stats.std_r, 0.0, 1e-4));
    }

    #[test]
    fn test_color_stats_two_color() {
        // Half red, half blue.
        let mut img = uniform_image(2, 1, 255, 0, 0, 255);
        img.extend_from_slice(&[0, 0, 255, 255]);
        let stats = compute_color_stats(&img, 3, 1).unwrap();
        assert!(stats.mean_r > 0.0);
        assert!(stats.std_r > 0.0);
    }

    #[test]
    fn test_color_stats_empty_error() {
        assert!(compute_color_stats(&[], 2, 2).is_err());
    }

    // ── gray_world_white_balance tests ────────────────────────────────────────

    #[test]
    fn test_gray_world_gray_image_near_identity() {
        // A uniform gray image → gray world should return something close to identity scaling.
        let img = uniform_image(4, 4, 128, 128, 128, 255);
        let m = gray_world_white_balance(&img, 4, 4).unwrap();
        let out = m.apply([0.5, 0.5, 0.5]);
        // All channels should be equal (gray world preserves gray).
        assert!(approx(out[0], out[1], 1e-4));
        assert!(approx(out[1], out[2], 1e-4));
    }

    #[test]
    fn test_gray_world_red_tinted() {
        // Image with extra red → matrix should reduce red channel.
        let img = uniform_image(4, 4, 200, 128, 128, 255);
        let m = gray_world_white_balance(&img, 4, 4).unwrap();
        // Red multiplier should be less than green/blue multipliers.
        assert!(m.data[0][0] <= m.data[1][1]);
        assert!(m.data[0][0] <= m.data[2][2]);
    }

    #[test]
    fn test_gray_world_all_black_returns_identity() {
        let img = uniform_image(4, 4, 0, 0, 0, 255);
        let m = gray_world_white_balance(&img, 4, 4).unwrap();
        let out = m.apply([0.5, 0.5, 0.5]);
        // Identity scale.
        assert!(approx(out[0], 0.5, 1e-5));
    }

    // ── apply_white_balance_image tests ──────────────────────────────────────

    #[test]
    fn test_apply_white_balance_neutral_unchanged() {
        // Default white balance (6500 K, no tint, no exposure).
        // The WB matrix at 6500 K is not exactly identity because the kelvin
        // polynomial gives slightly unequal RGB weights; the max channel is
        // normalized to 1.0. We verify alpha is preserved and the output stays
        // close (within ±4/255) to the original.
        let img = uniform_image(2, 2, 128, 128, 128, 255);
        let wb = WhiteBalance::default();
        let out = apply_white_balance_image(&img, 2, 2, &wb).unwrap();
        // Alpha must be preserved.
        assert_eq!(out[3], 255);
        // All channel values should be within 4 u8 steps of the original.
        let diff_r = (out[0] as i32 - 128).abs();
        let diff_g = (out[1] as i32 - 128).abs();
        let diff_b = (out[2] as i32 - 128).abs();
        assert!(diff_r <= 4, "R diff too large: {}", diff_r);
        assert!(diff_g <= 4, "G diff too large: {}", diff_g);
        assert!(diff_b <= 4, "B diff too large: {}", diff_b);
    }

    #[test]
    fn test_apply_white_balance_warm_shift() {
        let img = uniform_image(2, 2, 200, 200, 200, 255);
        let wb = WhiteBalance {
            temperature: 3000.0,
            tint: 0.0,
            exposure: 0.0,
        };
        let out = apply_white_balance_image(&img, 2, 2, &wb).unwrap();
        // Warm shift: red >= blue.
        assert!(out[0] >= out[2]);
    }

    #[test]
    fn test_apply_white_balance_invalid_temp() {
        let img = uniform_image(1, 1, 128, 128, 128, 255);
        let wb = WhiteBalance {
            temperature: 50.0,
            tint: 0.0,
            exposure: 0.0,
        };
        assert!(apply_white_balance_image(&img, 1, 1, &wb).is_err());
    }

    // ── apply_color_correction_matrix tests ──────────────────────────────────

    #[test]
    fn test_apply_ccm_identity_unchanged() {
        let img = uniform_image(2, 2, 100, 150, 200, 255);
        let m = ColorMatrix::identity();
        let out = apply_color_correction_matrix(&img, 2, 2, &m).unwrap();
        assert_eq!(out[0], 100);
        assert_eq!(out[1], 150);
        assert_eq!(out[2], 200);
        assert_eq!(out[3], 255);
    }

    #[test]
    fn test_apply_ccm_scale_doubles_red() {
        let img = uniform_image(2, 2, 50, 100, 100, 255);
        let m = ColorMatrix::scale(2.0, 1.0, 1.0);
        let out = apply_color_correction_matrix(&img, 2, 2, &m).unwrap();
        // 50/255 * 2.0 * 255 = 100 (approx).
        assert!(out[0] >= 98 && out[0] <= 102);
        assert_eq!(out[1], 100);
    }

    #[test]
    fn test_apply_ccm_clamping() {
        let img = uniform_image(1, 1, 200, 200, 200, 255);
        let m = ColorMatrix::scale(2.0, 2.0, 2.0);
        let out = apply_color_correction_matrix(&img, 1, 1, &m).unwrap();
        // All should clamp to 255.
        assert_eq!(out[0], 255);
        assert_eq!(out[1], 255);
        assert_eq!(out[2], 255);
    }

    // ── apply_gamma_correction_image tests ───────────────────────────────────

    #[test]
    fn test_apply_gamma_one_unchanged() {
        let img = uniform_image(2, 2, 128, 64, 200, 255);
        let out = apply_gamma_correction_image(&img, 2, 2, 1.0).unwrap();
        // Gamma 1.0 = identity.
        assert_eq!(out[0], 128);
        assert_eq!(out[1], 64);
        assert_eq!(out[2], 200);
    }

    #[test]
    fn test_apply_gamma_invalid() {
        let img = uniform_image(1, 1, 128, 128, 128, 255);
        assert!(apply_gamma_correction_image(&img, 1, 1, 0.0).is_err());
        assert!(apply_gamma_correction_image(&img, 1, 1, -1.0).is_err());
    }

    // ── apply_saturation_image tests ──────────────────────────────────────────

    #[test]
    fn test_apply_saturation_one_unchanged() {
        let img = uniform_image(2, 2, 100, 150, 200, 255);
        let out = apply_saturation_image(&img, 2, 2, 1.0).unwrap();
        assert_eq!(out[0], 100);
        assert_eq!(out[1], 150);
        assert_eq!(out[2], 200);
    }

    #[test]
    fn test_apply_saturation_zero_grayscale() {
        let img = uniform_image(2, 2, 200, 100, 50, 255);
        let out = apply_saturation_image(&img, 2, 2, 0.0).unwrap();
        // All channels should be equal (grayscale).
        assert_eq!(out[0], out[1]);
        assert_eq!(out[1], out[2]);
    }

    #[test]
    fn test_apply_saturation_alpha_preserved() {
        let img = uniform_image(2, 2, 100, 100, 100, 200);
        let out = apply_saturation_image(&img, 2, 2, 0.5).unwrap();
        assert_eq!(out[3], 200);
    }

    // ── apply_calibration tests ───────────────────────────────────────────────

    #[test]
    fn test_apply_calibration_default_near_unchanged() {
        // Default config: 6500 K WB (not exact identity), gamma 1.0, sat 1.0, brightness 1.0.
        // The 6500 K polynomial gives slightly unequal RGB weights; check that
        // the output stays within ±4/255 of the input.
        let img = uniform_image(2, 2, 128, 128, 128, 255);
        let config = ColorCalibrationConfig::default();
        let out = apply_calibration(&img, 2, 2, &config).unwrap();
        let diff_r = (out[0] as i32 - 128).abs();
        let diff_g = (out[1] as i32 - 128).abs();
        let diff_b = (out[2] as i32 - 128).abs();
        assert!(diff_r <= 4, "R diff too large: {}", diff_r);
        assert!(diff_g <= 4, "G diff too large: {}", diff_g);
        assert!(diff_b <= 4, "B diff too large: {}", diff_b);
    }

    #[test]
    fn test_apply_calibration_with_gamma() {
        let img = uniform_image(2, 2, 128, 128, 128, 255);
        let config = ColorCalibrationConfig {
            gamma: 2.0,
            ..Default::default()
        };
        let out = apply_calibration(&img, 2, 2, &config).unwrap();
        // Gamma 2.0 brightens the image: output > input.
        assert!(out[0] > 128);
    }

    #[test]
    fn test_apply_calibration_brightness() {
        let img = uniform_image(2, 2, 128, 128, 128, 255);
        let config = ColorCalibrationConfig {
            brightness: 0.5,
            ..Default::default()
        };
        let out = apply_calibration(&img, 2, 2, &config).unwrap();
        // Half brightness.
        assert!(out[0] < 128);
    }

    #[test]
    fn test_apply_calibration_pipeline_order_saturation_then_gamma() {
        // Regression test pinning the documented pipeline order: white
        // balance -> color matrix -> saturation -> gamma -> brightness.
        // Saturation is folded into the same linear matrix as white
        // balance/color matrix and applied *before* the nonlinear gamma
        // step -- see `saturation_matrix`'s doc ("operating in linear RGB
        // space"). Uses a non-gray pixel with saturation != 1 and gamma !=
        // 1 so the two possible orderings (saturation-then-gamma vs.
        // gamma-then-saturation) provably diverge.
        let config = ColorCalibrationConfig {
            white_balance: WhiteBalance::default(),
            color_matrix: None,
            gamma: 2.2,
            saturation: 0.4,
            brightness: 1.0,
        };
        let img = vec![200u8, 80, 40, 255]; // single non-gray pixel
        let out = apply_calibration(&img, 1, 1, &config).unwrap();

        let wb_matrix = white_balance_matrix(&config.white_balance).unwrap();
        let sat_matrix = saturation_matrix(config.saturation);
        let r = img[0] as f32 / 255.0;
        let g = img[1] as f32 / 255.0;
        let b = img[2] as f32 / 255.0;

        // Reference computed in the documented order: matrix (WB * sat),
        // then gamma, then brightness.
        let full_matrix = sat_matrix.mul(&wb_matrix);
        let [mr, mg, mb] = full_matrix.apply([r, g, b]);
        let expected = [mr, mg, mb].map(|c| {
            (apply_gamma_f32(c.clamp(0.0, 1.0), config.gamma) * config.brightness).clamp(0.0, 1.0)
        });
        for (i, &e) in expected.iter().enumerate() {
            let got = out[i] as f32 / 255.0;
            assert!(
                (got - e).abs() < 1.0 / 255.0 + 1e-4,
                "channel {i}: expected {e}, got {got} (saturation-then-gamma order)"
            );
        }

        // The alternate order (gamma applied first, then saturation) must
        // give a *different* result for this input, or the two orderings
        // wouldn't be distinguishable and the assertion above would be
        // vacuous.
        let [wr, wg, wb_] = wb_matrix.apply([r, g, b]);
        let gamma_first = [wr, wg, wb_].map(|c| apply_gamma_f32(c.clamp(0.0, 1.0), config.gamma));
        let [ar, ag, ab] = sat_matrix.apply(gamma_first);
        let alt = [ar, ag, ab].map(|c| (c * config.brightness).clamp(0.0, 1.0));
        assert!(
            (alt[0] - expected[0]).abs() > 1e-3 || (alt[1] - expected[1]).abs() > 1e-3,
            "test input should distinguish saturation-then-gamma from gamma-then-saturation"
        );
    }

    // ── compute_correction_matrix tests ──────────────────────────────────────

    #[test]
    fn test_correction_matrix_identical_images_identity() {
        let img = uniform_image(4, 4, 100, 150, 200, 255);
        let m = compute_correction_matrix(&img, &img, 4, 4).unwrap();
        // Scale should be 1.0 for all channels.
        assert!(approx(m.data[0][0], 1.0, 1e-4));
        assert!(approx(m.data[1][1], 1.0, 1e-4));
        assert!(approx(m.data[2][2], 1.0, 1e-4));
    }

    #[test]
    fn test_correction_matrix_scaled_reference() {
        let src = uniform_image(4, 4, 100, 100, 100, 255);
        // Reference red channel is double the source.
        let ref_img = uniform_image(4, 4, 200, 100, 100, 255);
        // Keep ref_img as Vec<u8>.
        let _ = ref_img.len();
        let m = compute_correction_matrix(&src, &ref_img, 4, 4).unwrap();
        // Red scale should be ~2.0.
        assert!(approx(m.data[0][0], 2.0, 0.05));
        // Green and blue scales should be ~1.0.
        assert!(approx(m.data[1][1], 1.0, 0.05));
        assert!(approx(m.data[2][2], 1.0, 0.05));
    }

    #[test]
    fn test_correction_matrix_dimension_mismatch() {
        let src = uniform_image(2, 2, 100, 100, 100, 255);
        let reference = uniform_image(4, 4, 100, 100, 100, 255);
        assert!(compute_correction_matrix(&src, &reference, 2, 2).is_err());
    }

    // ── apply_srgb_encoding / decoding tests ─────────────────────────────────

    #[test]
    fn test_srgb_encoding_roundtrip_image() {
        let img = uniform_image(2, 2, 100, 150, 200, 255);
        let encoded = apply_srgb_encoding(&img, 2, 2).unwrap();
        let decoded = apply_srgb_decoding(&encoded, 2, 2).unwrap();
        // After encode then decode, should be close to original.
        let diff_r = (decoded[0] as i32 - img[0] as i32).abs();
        assert!(
            diff_r <= 2,
            "R channel roundtrip diff too large: {}",
            diff_r
        );
    }

    #[test]
    fn test_srgb_decoding_roundtrip_image() {
        let img = uniform_image(2, 2, 80, 120, 180, 200);
        let decoded = apply_srgb_decoding(&img, 2, 2).unwrap();
        let reencoded = apply_srgb_encoding(&decoded, 2, 2).unwrap();
        let diff_r = (reencoded[0] as i32 - img[0] as i32).abs();
        assert!(
            diff_r <= 2,
            "R channel roundtrip diff too large: {}",
            diff_r
        );
    }

    #[test]
    fn test_srgb_alpha_preserved() {
        let img = uniform_image(1, 1, 100, 100, 100, 128);
        let enc = apply_srgb_encoding(&img, 1, 1).unwrap();
        assert_eq!(enc[3], 128);
        let dec = apply_srgb_decoding(&img, 1, 1).unwrap();
        assert_eq!(dec[3], 128);
    }

    // ── histogram_stretch_matrix tests ────────────────────────────────────────

    #[test]
    fn test_histogram_stretch_full_range_scale_one() {
        // Image with full range [0, 255] per channel → scale = 1.0.
        let mut img = Vec::with_capacity(8);
        img.extend_from_slice(&[0, 0, 0, 255]);
        img.extend_from_slice(&[255, 255, 255, 255]);
        let m = histogram_stretch_matrix(&img, 1, 2).unwrap();
        assert!(approx(m.data[0][0], 1.0, 1e-4));
        assert!(approx(m.data[1][1], 1.0, 1e-4));
        assert!(approx(m.data[2][2], 1.0, 1e-4));
    }

    #[test]
    fn test_histogram_stretch_half_range() {
        // Image with range [0, 127] per channel → scale ≈ 2.0.
        let mut img = Vec::with_capacity(8);
        img.extend_from_slice(&[0, 0, 0, 255]);
        img.extend_from_slice(&[127, 127, 127, 255]);
        let m = histogram_stretch_matrix(&img, 1, 2).unwrap();
        // scale = 255 / 127 ≈ 2.0.
        assert!(m.data[0][0] > 1.9);
    }

    // ── d65_white_balance test ────────────────────────────────────────────────

    #[test]
    fn test_d65_white_balance_near_identity() {
        let m = d65_white_balance();
        // D65 WB applied to pure white should give a result close to white.
        let out = m.apply([1.0, 1.0, 1.0]);
        // Max component should not exceed 1.0 (multipliers are normalized).
        let max_out = out[0].max(out[1]).max(out[2]);
        assert!(max_out <= 1.01);
        // All channels should be positive.
        assert!(out[0] > 0.0 && out[1] > 0.0 && out[2] > 0.0);
    }

    // ── Buffer validation tests ───────────────────────────────────────────────

    #[test]
    fn test_buffer_size_mismatch_error() {
        let bad_buf = vec![0u8; 10];
        assert!(compute_color_stats(&bad_buf, 4, 4).is_err());
    }

    #[test]
    fn test_empty_buffer_error() {
        assert!(compute_color_stats(&[], 2, 2).is_err());
    }
}
