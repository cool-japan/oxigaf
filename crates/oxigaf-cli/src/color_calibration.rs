//! Color calibration utilities for OxiGAF.
//!
//! Implements color checker calibration, white balance computation, and color
//! correction for matching renders to reference images or physical cameras.

use thiserror::Error;

// ───────────────────────────────────────── Error type ──────────────────────

#[derive(Debug, Error)]
pub enum CalibrationError {
    #[error("not enough patches: need at least {needed}, got {got}")]
    NotEnoughPatches { needed: usize, got: usize },
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("singular matrix: cannot invert")]
    SingularMatrix,
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("empty input")]
    EmptyInput,
}

// ───────────────────────────────── Macbeth ColorChecker patches ────────────

/// Standard Macbeth ColorChecker patch reference values (linear sRGB).
#[derive(Debug, Clone)]
pub struct ColorPatch {
    pub name: &'static str,
    pub reference_rgb: [f32; 3],
}

/// Returns the 24 standard Macbeth ColorChecker patches in linear sRGB.
///
/// Values are D65-adapted linear sRGB approximations from the standard
/// ColorChecker chart. These are reference values for calibration purposes.
pub fn cal_macbeth_patches() -> [ColorPatch; 24] {
    [
        ColorPatch {
            name: "Dark Skin",
            reference_rgb: [0.1152, 0.0611, 0.0423],
        },
        ColorPatch {
            name: "Light Skin",
            reference_rgb: [0.3973, 0.2567, 0.1729],
        },
        ColorPatch {
            name: "Blue Sky",
            reference_rgb: [0.0912, 0.1247, 0.2588],
        },
        ColorPatch {
            name: "Foliage",
            reference_rgb: [0.0654, 0.1000, 0.0477],
        },
        ColorPatch {
            name: "Blue Flower",
            reference_rgb: [0.2243, 0.1990, 0.4129],
        },
        ColorPatch {
            name: "Bluish Green",
            reference_rgb: [0.0908, 0.3972, 0.3847],
        },
        ColorPatch {
            name: "Orange",
            reference_rgb: [0.6890, 0.2453, 0.0194],
        },
        ColorPatch {
            name: "Purplish Blue",
            reference_rgb: [0.0710, 0.0834, 0.3430],
        },
        ColorPatch {
            name: "Moderate Red",
            reference_rgb: [0.5296, 0.0938, 0.1012],
        },
        ColorPatch {
            name: "Purple",
            reference_rgb: [0.1024, 0.0507, 0.1357],
        },
        ColorPatch {
            name: "Yellow Green",
            reference_rgb: [0.3626, 0.4985, 0.0462],
        },
        ColorPatch {
            name: "Orange Yellow",
            reference_rgb: [0.7935, 0.4088, 0.0182],
        },
        ColorPatch {
            name: "Blue",
            reference_rgb: [0.0295, 0.0427, 0.3255],
        },
        ColorPatch {
            name: "Green",
            reference_rgb: [0.0794, 0.3122, 0.0716],
        },
        ColorPatch {
            name: "Red",
            reference_rgb: [0.5632, 0.0373, 0.0296],
        },
        ColorPatch {
            name: "Yellow",
            reference_rgb: [0.9139, 0.7596, 0.0240],
        },
        ColorPatch {
            name: "Magenta",
            reference_rgb: [0.5420, 0.0887, 0.3278],
        },
        ColorPatch {
            name: "Cyan",
            reference_rgb: [0.0275, 0.2593, 0.4339],
        },
        ColorPatch {
            name: "White",
            reference_rgb: [0.9000, 0.9000, 0.9000],
        },
        ColorPatch {
            name: "Neutral 8",
            reference_rgb: [0.5780, 0.5780, 0.5780],
        },
        ColorPatch {
            name: "Neutral 6.5",
            reference_rgb: [0.3515, 0.3515, 0.3515],
        },
        ColorPatch {
            name: "Neutral 5",
            reference_rgb: [0.1913, 0.1913, 0.1913],
        },
        ColorPatch {
            name: "Neutral 3.5",
            reference_rgb: [0.0865, 0.0865, 0.0865],
        },
        ColorPatch {
            name: "Black",
            reference_rgb: [0.0280, 0.0280, 0.0280],
        },
    ]
}

// ─────────────────────────── 3×3 matrix helper (private) ──────────────────

/// Invert a 3×3 matrix (row-major). Returns `Err` if singular (|det| < 1e-10).
fn invert_3x3(m: &[f32; 9]) -> Result<[f32; 9], CalibrationError> {
    // Cofactors
    let c00 = m[4] * m[8] - m[5] * m[7];
    let c01 = m[5] * m[6] - m[3] * m[8];
    let c02 = m[3] * m[7] - m[4] * m[6];
    let c10 = m[2] * m[7] - m[1] * m[8];
    let c11 = m[0] * m[8] - m[2] * m[6];
    let c12 = m[1] * m[6] - m[0] * m[7];
    let c20 = m[1] * m[5] - m[2] * m[4];
    let c21 = m[2] * m[3] - m[0] * m[5];
    let c22 = m[0] * m[4] - m[1] * m[3];

    let det = m[0] * c00 + m[1] * c01 + m[2] * c02;
    if det.abs() < 1e-10 {
        return Err(CalibrationError::SingularMatrix);
    }
    let inv_det = 1.0 / det;

    // Adjugate (transpose of cofactor matrix) divided by determinant
    Ok([
        c00 * inv_det,
        c10 * inv_det,
        c20 * inv_det,
        c01 * inv_det,
        c11 * inv_det,
        c21 * inv_det,
        c02 * inv_det,
        c12 * inv_det,
        c22 * inv_det,
    ])
}

/// Multiply two 3×3 matrices (row-major): result = a * b.
fn mul_3x3(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let mut out = [0.0f32; 9];
    for row in 0..3 {
        for col in 0..3 {
            out[row * 3 + col] =
                a[row * 3] * b[col] + a[row * 3 + 1] * b[3 + col] + a[row * 3 + 2] * b[6 + col];
        }
    }
    out
}

// ──────────────────────────── CalibrationMatrix ────────────────────────────

/// 3×3 color correction matrix (row-major).
#[derive(Debug, Clone)]
pub struct CalibrationMatrix {
    pub m: [f32; 9],
}

impl CalibrationMatrix {
    /// Identity CCM.
    pub fn identity() -> Self {
        Self {
            m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Apply the matrix to a colour column vector: out = M * rgb.
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let m = &self.m;
        [
            m[0] * rgb[0] + m[1] * rgb[1] + m[2] * rgb[2],
            m[3] * rgb[0] + m[4] * rgb[1] + m[5] * rgb[2],
            m[6] * rgb[0] + m[7] * rgb[1] + m[8] * rgb[2],
        ]
    }

    /// Compose two matrices: `self * other`.
    pub fn compose(&self, other: &CalibrationMatrix) -> CalibrationMatrix {
        CalibrationMatrix {
            m: mul_3x3(&self.m, &other.m),
        }
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> CalibrationMatrix {
        let m = &self.m;
        CalibrationMatrix {
            m: [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]],
        }
    }
}

// ──────────────────────────── CCM solving (least squares) ──────────────────

/// Solve for 3×3 CCM using least squares: minimise `‖Measured · M − Reference‖²`.
///
/// Uses normal equations: `M = (Measᵀ Meas)⁻¹ Measᵀ Ref`.
/// At least 3 non-degenerate patches are required.
pub fn cal_solve_ccm(
    measured: &[[f32; 3]],
    reference: &[[f32; 3]],
) -> Result<CalibrationMatrix, CalibrationError> {
    if measured.len() < 3 {
        return Err(CalibrationError::NotEnoughPatches {
            needed: 3,
            got: measured.len(),
        });
    }
    if measured.len() != reference.len() {
        return Err(CalibrationError::DimensionMismatch {
            expected: measured.len(),
            got: reference.len(),
        });
    }

    let n = measured.len();

    // Build Aᵀ A (3×3) where A is the n×3 measured matrix.
    let mut ata = [0.0f32; 9];
    for p in measured {
        for r in 0..3 {
            for c in 0..3 {
                ata[r * 3 + c] += p[r] * p[c];
            }
        }
    }

    let ata_inv = invert_3x3(&ata)?;

    // Build Aᵀ B (3×3) where B is the n×3 reference matrix.
    // Each element [r][c] = Σ_i measured[i][r] * reference[i][c].
    let mut atb = [0.0f32; 9];
    for i in 0..n {
        for r in 0..3 {
            for c in 0..3 {
                atb[r * 3 + c] += measured[i][r] * reference[i][c];
            }
        }
    }

    // M_solve = (Aᵀ A)⁻¹ Aᵀ B  solves A * M_solve ≈ B in row-vector convention.
    // CalibrationMatrix::apply uses column-vector convention: out = M * col.
    // So we need M = M_solve^T.
    let solved = CalibrationMatrix {
        m: mul_3x3(&ata_inv, &atb),
    };
    Ok(solved.transpose())
}

/// Apply a CCM to a flat interleaved RGB image (width × height × 3 floats).
pub fn cal_apply_ccm(
    image: &[f32],
    width: usize,
    height: usize,
    ccm: &CalibrationMatrix,
) -> Result<Vec<f32>, CalibrationError> {
    let expected = width * height * 3;
    if image.len() != expected {
        return Err(CalibrationError::DimensionMismatch {
            expected,
            got: image.len(),
        });
    }
    if expected == 0 {
        return Err(CalibrationError::EmptyInput);
    }

    let mut out = Vec::with_capacity(expected);
    for chunk in image.chunks_exact(3) {
        let rgb = [chunk[0], chunk[1], chunk[2]];
        let corrected = ccm.apply(rgb);
        out.push(corrected[0]);
        out.push(corrected[1]);
        out.push(corrected[2]);
    }
    Ok(out)
}

// ─────────────────────────────── White Balance ─────────────────────────────

/// Per-channel multiplicative white balance gains.
#[derive(Debug, Clone)]
pub struct WhiteBalance {
    pub gains: [f32; 3],
}

impl WhiteBalance {
    /// Compute white balance from a measured neutral/gray patch.
    ///
    /// Gains are set so that the measured patch is mapped to equal-energy (1,1,1).
    /// The green channel is used as reference, so its gain is 1.0.
    pub fn from_gray_patch(measured: [f32; 3]) -> Result<WhiteBalance, CalibrationError> {
        let [r, g, b] = measured;
        if r <= 0.0 || g <= 0.0 || b <= 0.0 {
            return Err(CalibrationError::InvalidConfig(
                "gray patch channels must be positive".into(),
            ));
        }
        // Normalise so green gain = 1
        let r_gain = g / r;
        let g_gain = 1.0f32;
        let b_gain = g / b;
        Ok(WhiteBalance {
            gains: [r_gain, g_gain, b_gain],
        })
    }

    /// Compute white balance from colour temperature in Kelvin (2000 K – 10 000 K).
    ///
    /// Uses a Planckian locus approximation to derive chromaticity and then gains.
    pub fn from_temperature(kelvin: f32) -> Result<WhiteBalance, CalibrationError> {
        if !(2000.0..=10_000.0).contains(&kelvin) {
            return Err(CalibrationError::InvalidConfig(format!(
                "colour temperature {kelvin} K out of range [2000, 10000]"
            )));
        }

        // Planckian locus approximation (Robertson 1968 / Kim 2002 piecewise)
        let t = kelvin;
        let x = planckian_x(t);
        let y = planckian_y(t, x);

        // Convert CIE xy to XYZ (Y = 1)
        let (xyz_r, xyz_g, xyz_b) = xy_to_rgb_gains(x, y);

        // Normalise so green = 1
        let r_gain = xyz_g / xyz_r.max(1e-6);
        let g_gain = 1.0f32;
        let b_gain = xyz_g / xyz_b.max(1e-6);

        Ok(WhiteBalance {
            gains: [r_gain, g_gain, b_gain],
        })
    }

    /// Daylight / neutral white balance — no-op gains.
    pub fn daylight() -> Self {
        WhiteBalance {
            gains: [1.0, 1.0, 1.0],
        }
    }

    /// Apply white balance gains, clamping result to [0, ∞).
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        [
            (rgb[0] * self.gains[0]).max(0.0),
            (rgb[1] * self.gains[1]).max(0.0),
            (rgb[2] * self.gains[2]).max(0.0),
        ]
    }
}

/// Planckian locus x chromaticity approximation.
fn planckian_x(t: f32) -> f32 {
    if t < 4000.0 {
        -0.2661239e9 / (t * t * t) - 0.2343589e6 / (t * t) + 0.8776956e3 / t + 0.179910
    } else {
        -3.0258469e9 / (t * t * t) + 2.107_038e6 / (t * t) + 0.2226347e3 / t + 0.240390
    }
}

/// Planckian locus y chromaticity approximation.
fn planckian_y(t: f32, x: f32) -> f32 {
    if t < 2222.0 {
        -1.1063814 * x * x * x - 1.348_110_2 * x * x + 2.185_558_3 * x - 0.20219683
    } else if t < 4000.0 {
        -0.9549476 * x * x * x - 1.374_185_9 * x * x + 2.091_37 * x - 0.16748867
    } else {
        3.081_758 * x * x * x - 5.873_387 * x * x + 3.751_129_9 * x - 0.37001483
    }
}

/// Convert CIE xy to approximate linear sRGB channel gains (D65 reference).
fn xy_to_rgb_gains(x: f32, y: f32) -> (f32, f32, f32) {
    // Y = 1, compute XYZ
    let y_n = 1.0f32;
    let x_xyz = if y > 1e-6 { x * y_n / y } else { 0.0 };
    let z_xyz = if y > 1e-6 {
        (1.0 - x - y) * y_n / y
    } else {
        0.0
    };

    // Apply sRGB / D65 matrix to get linear sRGB
    let r = 3.240479 * x_xyz - 1.537_15 * y_n - 0.498535 * z_xyz;
    let g = -0.969256 * x_xyz + 1.875992 * y_n + 0.041556 * z_xyz;
    let b = 0.055648 * x_xyz - 0.204043 * y_n + 1.057311 * z_xyz;
    (r.max(1e-6), g.max(1e-6), b.max(1e-6))
}

/// Apply white balance to a flat interleaved RGB image.
pub fn cal_apply_white_balance(
    image: &[f32],
    width: usize,
    height: usize,
    wb: &WhiteBalance,
) -> Result<Vec<f32>, CalibrationError> {
    let expected = width * height * 3;
    if image.len() != expected {
        return Err(CalibrationError::DimensionMismatch {
            expected,
            got: image.len(),
        });
    }
    if expected == 0 {
        return Err(CalibrationError::EmptyInput);
    }
    let mut out = Vec::with_capacity(expected);
    for chunk in image.chunks_exact(3) {
        let corrected = wb.apply([chunk[0], chunk[1], chunk[2]]);
        out.push(corrected[0]);
        out.push(corrected[1]);
        out.push(corrected[2]);
    }
    Ok(out)
}

// ──────────────────────────── Gamma / tone curves ──────────────────────────

/// Gamma encoding profile.
#[derive(Debug, Clone)]
pub enum GammaProfile {
    /// No encoding — passes through unchanged.
    Linear,
    /// Piecewise sRGB standard (IEC 61966-2-1).
    Srgb,
    /// Power-law with exponent γ (encode: x^(1/γ)).
    Custom(f32),
}

/// Apply gamma encoding to a single channel value in [0, 1].
pub fn cal_apply_gamma_channel(value: f32, profile: &GammaProfile) -> f32 {
    let v = value.max(0.0);
    match profile {
        GammaProfile::Linear => v,
        GammaProfile::Srgb => {
            if v <= 0.003_130_8 {
                12.92 * v
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            }
        }
        GammaProfile::Custom(gamma) => {
            if *gamma <= 0.0 {
                v
            } else {
                v.powf(1.0 / gamma)
            }
        }
    }
}

/// Apply gamma decoding (linearisation) to a single channel value in [0, 1].
pub fn cal_apply_gamma_inv_channel(value: f32, profile: &GammaProfile) -> f32 {
    let v = value.max(0.0);
    match profile {
        GammaProfile::Linear => v,
        GammaProfile::Srgb => {
            if v <= 0.040_45 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        }
        GammaProfile::Custom(gamma) => {
            if *gamma <= 0.0 {
                v
            } else {
                v.powf(*gamma)
            }
        }
    }
}

/// Apply gamma encoding to a flat interleaved RGB image.
pub fn cal_apply_gamma(
    image: &[f32],
    width: usize,
    height: usize,
    profile: &GammaProfile,
) -> Result<Vec<f32>, CalibrationError> {
    let expected = width * height * 3;
    if image.len() != expected {
        return Err(CalibrationError::DimensionMismatch {
            expected,
            got: image.len(),
        });
    }
    if expected == 0 {
        return Err(CalibrationError::EmptyInput);
    }
    Ok(image
        .iter()
        .map(|&v| cal_apply_gamma_channel(v, profile))
        .collect())
}

/// Apply gamma decoding (linearisation) to a flat interleaved RGB image.
pub fn cal_apply_gamma_inv(
    image: &[f32],
    width: usize,
    height: usize,
    profile: &GammaProfile,
) -> Result<Vec<f32>, CalibrationError> {
    let expected = width * height * 3;
    if image.len() != expected {
        return Err(CalibrationError::DimensionMismatch {
            expected,
            got: image.len(),
        });
    }
    if expected == 0 {
        return Err(CalibrationError::EmptyInput);
    }
    Ok(image
        .iter()
        .map(|&v| cal_apply_gamma_inv_channel(v, profile))
        .collect())
}

// ────────────────────────── Color space conversions ─────────────────────────

/// sRGB (linear) → CIE XYZ D65 conversion matrix (IEC 61966-2-1).
const SRGB_TO_XYZ: [f32; 9] = [
    0.412_456_4,
    0.357_576_1,
    0.180_437_5,
    0.212_672_9,
    0.715_152_2,
    0.072_175_0,
    0.019_333_9,
    0.119_192,
    0.950_304_1,
];

/// CIE XYZ D65 → linear sRGB.
const XYZ_TO_SRGB: [f32; 9] = [
    3.240_479, -1.537_15, -0.498_535, -0.969_256, 1.875_992, 0.041_556, 0.055_648, -0.204_043,
    1.057_311,
];

/// D65 white point in XYZ (normalised to Y = 1).
const D65_XYZ: [f32; 3] = [0.950_456, 1.000_000, 1.088_906];

/// Linear sRGB to CIE XYZ (D65).
pub fn cal_srgb_to_xyz(rgb: [f32; 3]) -> [f32; 3] {
    let m = &SRGB_TO_XYZ;
    [
        m[0] * rgb[0] + m[1] * rgb[1] + m[2] * rgb[2],
        m[3] * rgb[0] + m[4] * rgb[1] + m[5] * rgb[2],
        m[6] * rgb[0] + m[7] * rgb[1] + m[8] * rgb[2],
    ]
}

/// CIE XYZ (D65) to linear sRGB.
pub fn cal_xyz_to_srgb(xyz: [f32; 3]) -> [f32; 3] {
    let m = &XYZ_TO_SRGB;
    [
        m[0] * xyz[0] + m[1] * xyz[1] + m[2] * xyz[2],
        m[3] * xyz[0] + m[4] * xyz[1] + m[5] * xyz[2],
        m[6] * xyz[0] + m[7] * xyz[1] + m[8] * xyz[2],
    ]
}

/// f(t) function used in Lab conversion.
#[inline]
fn lab_f(t: f32) -> f32 {
    const DELTA: f32 = 6.0 / 29.0;
    const DELTA3: f32 = DELTA * DELTA * DELTA; // ~0.008856
    if t > DELTA3 {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

/// Inverse f(t) for Lab → XYZ.
#[inline]
fn lab_f_inv(t: f32) -> f32 {
    const DELTA: f32 = 6.0 / 29.0;
    if t > DELTA {
        t * t * t
    } else {
        3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
}

/// CIE XYZ to CIELAB (D65 illuminant).
pub fn cal_xyz_to_lab(xyz: [f32; 3]) -> [f32; 3] {
    let fx = lab_f(xyz[0] / D65_XYZ[0]);
    let fy = lab_f(xyz[1] / D65_XYZ[1]);
    let fz = lab_f(xyz[2] / D65_XYZ[2]);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// CIELAB to CIE XYZ (D65 illuminant).
pub fn cal_lab_to_xyz(lab: [f32; 3]) -> [f32; 3] {
    let [l, a, b] = lab;
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    [
        lab_f_inv(fx) * D65_XYZ[0],
        lab_f_inv(fy) * D65_XYZ[1],
        lab_f_inv(fz) * D65_XYZ[2],
    ]
}

/// Linear sRGB to CIELAB (D65): sRGB → XYZ → Lab.
pub fn cal_srgb_to_lab(rgb: [f32; 3]) -> [f32; 3] {
    cal_xyz_to_lab(cal_srgb_to_xyz(rgb))
}

// ───────────────────────────── Delta E metrics ──────────────────────────────

/// CIE Delta E 1976 — Euclidean distance in Lab colour space.
pub fn cal_delta_e_76(lab1: [f32; 3], lab2: [f32; 3]) -> f32 {
    let dl = lab1[0] - lab2[0];
    let da = lab1[1] - lab2[1];
    let db = lab1[2] - lab2[2];
    (dl * dl + da * da + db * db).sqrt()
}

/// CIE Delta E 2000 — full CIEDE2000 formula.
pub fn cal_delta_e_2000(lab1: [f32; 3], lab2: [f32; 3]) -> f32 {
    const K_L: f32 = 1.0;
    const K_C: f32 = 1.0;
    const K_H: f32 = 1.0;

    let [l1, a1, b1] = lab1;
    let [l2, a2, b2] = lab2;

    // Step 1: C* and h* in Lab
    let c_ab1 = (a1 * a1 + b1 * b1).sqrt();
    let c_ab2 = (a2 * a2 + b2 * b2).sqrt();
    let c_ab_bar = (c_ab1 + c_ab2) * 0.5;

    // G factor
    let c_ab_bar7 = c_ab_bar.powi(7);
    let g = 0.5 * (1.0 - (c_ab_bar7 / (c_ab_bar7 + 25.0f32.powi(7))).sqrt());

    // a' values
    let a1_prime = a1 * (1.0 + g);
    let a2_prime = a2 * (1.0 + g);

    // C' values
    let c1_prime = (a1_prime * a1_prime + b1 * b1).sqrt();
    let c2_prime = (a2_prime * a2_prime + b2 * b2).sqrt();

    // h' values in degrees [0, 360)
    let h1_prime = hue_angle(a1_prime, b1);
    let h2_prime = hue_angle(a2_prime, b2);

    // Step 2: delta L', delta C', delta h'
    let delta_l_prime = l2 - l1;
    let delta_c_prime = c2_prime - c1_prime;

    let delta_h_prime = {
        if c1_prime * c2_prime < 1e-10 {
            0.0
        } else {
            let d = h2_prime - h1_prime;
            if d.abs() <= 180.0 {
                d
            } else if d > 180.0 {
                d - 360.0
            } else {
                d + 360.0
            }
        }
    };

    let delta_cap_h_prime =
        2.0 * (c1_prime * c2_prime).sqrt() * (delta_h_prime.to_radians() * 0.5).sin();

    // Step 3: CIEDE2000 weighting functions
    let l_bar_prime = (l1 + l2) * 0.5;
    let c_bar_prime = (c1_prime + c2_prime) * 0.5;

    let h_bar_prime = if c1_prime * c2_prime < 1e-10 {
        h1_prime + h2_prime
    } else {
        let sum = h1_prime + h2_prime;
        if (h1_prime - h2_prime).abs() <= 180.0 {
            sum * 0.5
        } else if sum < 360.0 {
            (sum + 360.0) * 0.5
        } else {
            (sum - 360.0) * 0.5
        }
    };

    let t = 1.0 - 0.17 * (h_bar_prime - 30.0).to_radians().cos()
        + 0.24 * (2.0 * h_bar_prime).to_radians().cos()
        + 0.32 * (3.0 * h_bar_prime + 6.0).to_radians().cos()
        - 0.20 * (4.0 * h_bar_prime - 63.0).to_radians().cos();

    let s_l = {
        let x = l_bar_prime - 50.0;
        1.0 + 0.015 * x * x / (20.0 + x * x).sqrt()
    };
    let s_c = 1.0 + 0.045 * c_bar_prime;
    let s_h = 1.0 + 0.015 * c_bar_prime * t;

    let c_bar_prime7 = c_bar_prime.powi(7);
    let r_c = 2.0 * (c_bar_prime7 / (c_bar_prime7 + 25.0f32.powi(7))).sqrt();
    let d_theta = 30.0 * (-((h_bar_prime - 275.0) / 25.0).powi(2)).exp();
    let r_t = -r_c * (2.0 * d_theta.to_radians()).sin();

    let l_term = delta_l_prime / (K_L * s_l);
    let c_term = delta_c_prime / (K_C * s_c);
    let h_term = delta_cap_h_prime / (K_H * s_h);

    (l_term * l_term + c_term * c_term + h_term * h_term + r_t * c_term * h_term).sqrt()
}

/// Compute hue angle in degrees [0, 360) from a' and b.
#[inline]
fn hue_angle(a_prime: f32, b: f32) -> f32 {
    if a_prime.abs() < 1e-10 && b.abs() < 1e-10 {
        return 0.0;
    }
    let angle = b.atan2(a_prime).to_degrees();
    if angle < 0.0 {
        angle + 360.0
    } else {
        angle
    }
}

// ─────────────────────────── Calibration statistics ────────────────────────

/// Quality statistics for a calibration.
#[derive(Debug, Clone)]
pub struct CalibrationStats {
    pub mean_delta_e: f32,
    pub max_delta_e: f32,
    pub min_delta_e: f32,
    pub rmse_rgb: f32,
    pub per_patch_delta_e: Vec<f32>,
}

/// Evaluate calibration quality: apply CCM + WB to measured patches, compare to reference.
pub fn cal_evaluate(
    measured: &[[f32; 3]],
    reference: &[[f32; 3]],
    ccm: &CalibrationMatrix,
    wb: &WhiteBalance,
) -> Result<CalibrationStats, CalibrationError> {
    if measured.is_empty() {
        return Err(CalibrationError::EmptyInput);
    }
    if measured.len() != reference.len() {
        return Err(CalibrationError::DimensionMismatch {
            expected: measured.len(),
            got: reference.len(),
        });
    }

    let n = measured.len();
    let mut per_patch_delta_e = Vec::with_capacity(n);
    let mut sum_sq_rgb = 0.0f32;

    for (m, r) in measured.iter().zip(reference.iter()) {
        // Apply WB then CCM
        let wb_applied = wb.apply(*m);
        let corrected = ccm.apply(wb_applied);

        // Delta E 76 in Lab
        let lab_corrected = cal_srgb_to_lab(corrected);
        let lab_ref = cal_srgb_to_lab(*r);
        let de = cal_delta_e_76(lab_corrected, lab_ref);
        per_patch_delta_e.push(de);

        // RMSE RGB
        for c in 0..3 {
            let diff = corrected[c] - r[c];
            sum_sq_rgb += diff * diff;
        }
    }

    let mean_delta_e = per_patch_delta_e.iter().copied().sum::<f32>() / n as f32;
    let max_delta_e = per_patch_delta_e
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_delta_e = per_patch_delta_e
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let rmse_rgb = (sum_sq_rgb / (n * 3) as f32).sqrt();

    Ok(CalibrationStats {
        mean_delta_e,
        max_delta_e,
        min_delta_e,
        rmse_rgb,
        per_patch_delta_e,
    })
}

/// Format calibration statistics as a human-readable string.
pub fn cal_format_stats(stats: &CalibrationStats) -> String {
    format!(
        "Calibration Stats:\n  Mean ΔE:  {:.4}\n  Max ΔE:   {:.4}\n  Min ΔE:   {:.4}\n  RMSE RGB: {:.6}\n  Patches:  {}",
        stats.mean_delta_e,
        stats.max_delta_e,
        stats.min_delta_e,
        stats.rmse_rgb,
        stats.per_patch_delta_e.len()
    )
}

// ───────────────────────────────────── Tests ────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn approx_eq_arr3(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        a.iter().zip(b.iter()).all(|(&x, &y)| (x - y).abs() < tol)
    }

    fn approx_eq_arr9(a: [f32; 9], b: [f32; 9], tol: f32) -> bool {
        a.iter().zip(b.iter()).all(|(&x, &y)| (x - y).abs() < tol)
    }

    // ── invert_3x3 ───────────────────────────────────────────────────────────

    #[test]
    fn test_invert_3x3_identity() {
        let id = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let inv = invert_3x3(&id).expect("identity should invert");
        assert!(approx_eq_arr9(inv, id, 1e-6));
    }

    #[test]
    fn test_invert_3x3_known() {
        // [2,1,0 / 1,3,0 / 0,0,4] -> det = (2*3-1*1)*4 = 20
        let m = [2.0f32, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 4.0];
        let inv = invert_3x3(&m).expect("should invert");
        let prod = mul_3x3(&m, &inv);
        let eye = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!(
            approx_eq_arr9(prod, eye, 1e-5),
            "A * A_inv should be identity, got {prod:?}"
        );
    }

    #[test]
    fn test_invert_3x3_singular() {
        // All-ones matrix is singular
        let m = [1.0f32; 9];
        assert!(invert_3x3(&m).is_err());
    }

    #[test]
    fn test_invert_3x3_zero() {
        let m = [0.0f32; 9];
        assert!(invert_3x3(&m).is_err());
    }

    // ── CalibrationMatrix ─────────────────────────────────────────────────────

    #[test]
    fn test_identity_apply() {
        let id = CalibrationMatrix::identity();
        let rgb = [0.5f32, 0.3, 0.7];
        let result = id.apply(rgb);
        assert!(approx_eq_arr3(result, rgb, 1e-6));
    }

    #[test]
    fn test_identity_compose_identity() {
        let id = CalibrationMatrix::identity();
        let composed = id.compose(&id);
        let result = composed.apply([0.4, 0.2, 0.8]);
        assert!(approx_eq_arr3(result, [0.4, 0.2, 0.8], 1e-6));
    }

    #[test]
    fn test_transpose() {
        let m = CalibrationMatrix {
            m: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        let t = m.transpose();
        assert_eq!(t.m, [1.0f32, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_transpose_twice_identity() {
        let m = CalibrationMatrix {
            m: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        let tt = m.transpose().transpose();
        assert!(approx_eq_arr9(tt.m, m.m, 1e-6));
    }

    #[test]
    fn test_compose_with_inverse_approx_identity() {
        // Build a simple scale matrix and its inverse
        let scale = CalibrationMatrix {
            m: [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0],
        };
        let inv_m = invert_3x3(&scale.m).expect("scale matrix is invertible");
        let inv = CalibrationMatrix { m: inv_m };
        let product = scale.compose(&inv);
        let rgb = [0.5f32, 0.3, 0.7];
        let result = product.apply(rgb);
        assert!(approx_eq_arr3(result, rgb, 1e-5));
    }

    // ── cal_solve_ccm ─────────────────────────────────────────────────────────

    #[test]
    fn test_solve_ccm_identity_case() {
        // When measured == reference, CCM should be approximately identity
        let patches: Vec<[f32; 3]> = vec![
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3],
            [0.5, 0.3, 0.1],
            [0.2, 0.4, 0.6],
        ];
        let ccm = cal_solve_ccm(&patches, &patches).expect("should succeed");
        let rgb = [0.4f32, 0.3, 0.2];
        let result = ccm.apply(rgb);
        assert!(
            approx_eq_arr3(result, rgb, 1e-4),
            "CCM should be identity-like: {result:?}"
        );
    }

    #[test]
    fn test_solve_ccm_not_enough_patches() {
        let p: Vec<[f32; 3]> = vec![[0.5, 0.3, 0.1], [0.2, 0.4, 0.6]];
        let err = cal_solve_ccm(&p, &p).expect_err("should fail");
        assert!(matches!(err, CalibrationError::NotEnoughPatches { .. }));
    }

    #[test]
    fn test_solve_ccm_dimension_mismatch() {
        let meas: Vec<[f32; 3]> = vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]];
        let refr: Vec<[f32; 3]> = vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        assert!(cal_solve_ccm(&meas, &refr).is_err());
    }

    #[test]
    fn test_solve_ccm_cross_channel_permutation() {
        // Red→Green, Green→Blue, Blue→Red (cyclic permutation)
        // Exact 3-patch solution: M should be the permutation matrix
        let meas: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let refr: Vec<[f32; 3]> = vec![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
        let ccm = cal_solve_ccm(&meas, &refr).expect("should succeed");
        let result = ccm.apply([1.0, 0.0, 0.0]);
        assert!(
            approx_eq_arr3(result, [0.0, 1.0, 0.0], 1e-4),
            "red input should map to green: {result:?}"
        );
        let result2 = ccm.apply([0.0, 1.0, 0.0]);
        assert!(
            approx_eq_arr3(result2, [0.0, 0.0, 1.0], 1e-4),
            "green input should map to blue: {result2:?}"
        );
    }

    #[test]
    fn test_solve_ccm_singular_patches() {
        // All identical patches make Aᵀ A singular
        let p: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5]; 5];
        assert!(cal_solve_ccm(&p, &p).is_err());
    }

    #[test]
    fn test_solve_ccm_scale_correction() {
        // Reference is 2x measured in all channels
        let meas: Vec<[f32; 3]> = vec![
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3],
            [0.2, 0.1, 0.1],
            [0.1, 0.3, 0.2],
        ];
        let refr: Vec<[f32; 3]> = meas
            .iter()
            .map(|p| [p[0] * 2.0, p[1] * 2.0, p[2] * 2.0])
            .collect();
        let ccm = cal_solve_ccm(&meas, &refr).expect("should succeed");
        let test_rgb = [0.1f32, 0.2, 0.3];
        let result = ccm.apply(test_rgb);
        let expected = [0.2f32, 0.4, 0.6];
        assert!(approx_eq_arr3(result, expected, 1e-4));
    }

    // ── cal_apply_ccm ─────────────────────────────────────────────────────────

    #[test]
    fn test_apply_ccm_dimension_mismatch() {
        let img = vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5]; // 2 pixels
        let ccm = CalibrationMatrix::identity();
        // Claim 2×2 = 4 pixels (12 values) but only 6 provided
        assert!(cal_apply_ccm(&img, 2, 2, &ccm).is_err());
    }

    #[test]
    fn test_apply_ccm_identity() {
        let img = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let ccm = CalibrationMatrix::identity();
        let out = cal_apply_ccm(&img, 2, 1, &ccm).expect("should succeed");
        for (a, b) in img.iter().zip(out.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_apply_ccm_empty() {
        let img: Vec<f32> = vec![];
        let ccm = CalibrationMatrix::identity();
        assert!(cal_apply_ccm(&img, 0, 0, &ccm).is_err());
    }

    // ── WhiteBalance ──────────────────────────────────────────────────────────

    #[test]
    fn test_wb_from_gray_patch_neutral() {
        let wb = WhiteBalance::from_gray_patch([1.0, 1.0, 1.0]).expect("should succeed");
        assert!(approx_eq(wb.gains[0], 1.0, 1e-6));
        assert!(approx_eq(wb.gains[1], 1.0, 1e-6));
        assert!(approx_eq(wb.gains[2], 1.0, 1e-6));
    }

    #[test]
    fn test_wb_from_gray_patch_red_biased() {
        let wb = WhiteBalance::from_gray_patch([2.0, 1.0, 1.0]).expect("should succeed");
        // Red is high, so its gain should be < 1 (attenuate)
        assert!(
            wb.gains[0] < 1.0,
            "red gain should be less than 1 for red-biased patch"
        );
        assert!(approx_eq(wb.gains[1], 1.0, 1e-6));
        assert!(approx_eq(wb.gains[2], 1.0, 1e-6));
    }

    #[test]
    fn test_wb_from_gray_patch_blue_biased() {
        let wb = WhiteBalance::from_gray_patch([1.0, 1.0, 2.0]).expect("should succeed");
        assert!(
            wb.gains[2] < 1.0,
            "blue gain should be less than 1 for blue-biased"
        );
    }

    #[test]
    fn test_wb_from_gray_patch_invalid_zero_channel() {
        assert!(WhiteBalance::from_gray_patch([0.0, 1.0, 1.0]).is_err());
        assert!(WhiteBalance::from_gray_patch([1.0, 0.0, 1.0]).is_err());
        assert!(WhiteBalance::from_gray_patch([1.0, 1.0, 0.0]).is_err());
    }

    #[test]
    fn test_wb_from_temperature_d65_range() {
        // 6500 K ~ D65
        let wb = WhiteBalance::from_temperature(6500.0).expect("should succeed");
        assert!(
            wb.gains.iter().all(|&g| g > 0.0),
            "all gains should be positive"
        );
    }

    #[test]
    fn test_wb_from_temperature_warm() {
        let wb_warm = WhiteBalance::from_temperature(3000.0).expect("should succeed");
        let wb_cool = WhiteBalance::from_temperature(8000.0).expect("should succeed");
        // Warm light → more red, so red gain should be lower (less attenuation needed)
        // and blue gain higher to correct.
        assert!(
            wb_warm.gains[2] > wb_cool.gains[2] || wb_cool.gains[0] > wb_warm.gains[0],
            "warm and cool WB should differ"
        );
    }

    #[test]
    fn test_wb_from_temperature_out_of_range() {
        assert!(WhiteBalance::from_temperature(1000.0).is_err());
        assert!(WhiteBalance::from_temperature(12000.0).is_err());
    }

    #[test]
    fn test_wb_daylight_noop() {
        let wb = WhiteBalance::daylight();
        let rgb = [0.5f32, 0.3, 0.7];
        let result = wb.apply(rgb);
        assert!(approx_eq_arr3(result, rgb, 1e-6));
    }

    #[test]
    fn test_wb_apply_clamping() {
        let wb = WhiteBalance {
            gains: [1.0, 1.0, 1.0],
        };
        let result = wb.apply([-1.0, 0.5, 0.5]);
        assert!(result[0] >= 0.0, "clamped to zero");
    }

    #[test]
    fn test_apply_white_balance_dimension_mismatch() {
        let img = vec![0.5f32; 6]; // 2 pixels
        let wb = WhiteBalance::daylight();
        assert!(cal_apply_white_balance(&img, 2, 2, &wb).is_err());
    }

    #[test]
    fn test_apply_white_balance_daylight() {
        let img = vec![0.2f32, 0.4, 0.6, 0.1, 0.3, 0.5];
        let wb = WhiteBalance::daylight();
        let out = cal_apply_white_balance(&img, 2, 1, &wb).expect("should succeed");
        for (a, b) in img.iter().zip(out.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_apply_white_balance_empty() {
        let wb = WhiteBalance::daylight();
        assert!(cal_apply_white_balance(&[], 0, 0, &wb).is_err());
    }

    // ── Gamma ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_gamma_linear_is_identity() {
        let v = 0.5f32;
        assert_eq!(cal_apply_gamma_channel(v, &GammaProfile::Linear), v);
        assert_eq!(cal_apply_gamma_inv_channel(v, &GammaProfile::Linear), v);
    }

    #[test]
    fn test_gamma_srgb_roundtrip() {
        for &v in &[0.0f32, 0.001, 0.01, 0.1, 0.5, 0.9, 1.0] {
            let encoded = cal_apply_gamma_channel(v, &GammaProfile::Srgb);
            let decoded = cal_apply_gamma_inv_channel(encoded, &GammaProfile::Srgb);
            assert!(
                approx_eq(v, decoded, 1e-4),
                "sRGB round-trip failed for {v}: encoded={encoded}, decoded={decoded}"
            );
        }
    }

    #[test]
    fn test_gamma_srgb_midgray() {
        // Linear 0.18 should encode to approximately 0.461 (sRGB 18% gray)
        let enc = cal_apply_gamma_channel(0.18, &GammaProfile::Srgb);
        assert!(
            enc > 0.40 && enc < 0.52,
            "18% linear → sRGB should be ~0.46, got {enc}"
        );
    }

    #[test]
    fn test_gamma_custom_roundtrip() {
        let gamma = GammaProfile::Custom(2.2);
        for &v in &[0.0f32, 0.1, 0.3, 0.5, 0.8, 1.0] {
            let enc = cal_apply_gamma_channel(v, &gamma);
            let dec = cal_apply_gamma_inv_channel(enc, &gamma);
            assert!(
                approx_eq(v, dec, 1e-5),
                "Custom(2.2) round-trip for {v}: {dec}"
            );
        }
    }

    #[test]
    fn test_gamma_custom_power_law() {
        let gamma = GammaProfile::Custom(2.0);
        let enc = cal_apply_gamma_channel(0.25, &gamma);
        // 0.25^(1/2) = 0.5
        assert!(
            approx_eq(enc, 0.5, 1e-5),
            "0.25^(1/2) should be 0.5, got {enc}"
        );
    }

    #[test]
    fn test_apply_gamma_image_roundtrip() {
        let img = vec![0.1f32, 0.5, 0.9, 0.2, 0.4, 0.8];
        let enc = cal_apply_gamma(&img, 2, 1, &GammaProfile::Srgb).expect("encode");
        let dec = cal_apply_gamma_inv(&enc, 2, 1, &GammaProfile::Srgb).expect("decode");
        for (a, b) in img.iter().zip(dec.iter()) {
            assert!(
                approx_eq(*a, *b, 1e-4),
                "gamma image round-trip: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_apply_gamma_dimension_mismatch() {
        let img = vec![0.5f32; 6];
        assert!(cal_apply_gamma(&img, 2, 2, &GammaProfile::Srgb).is_err());
        assert!(cal_apply_gamma_inv(&img, 2, 2, &GammaProfile::Srgb).is_err());
    }

    #[test]
    fn test_apply_gamma_empty() {
        assert!(cal_apply_gamma(&[], 0, 0, &GammaProfile::Linear).is_err());
    }

    // ── Color space conversions ───────────────────────────────────────────────

    #[test]
    fn test_srgb_to_xyz_roundtrip() {
        let rgb = [0.3f32, 0.5, 0.7];
        let xyz = cal_srgb_to_xyz(rgb);
        let back = cal_xyz_to_srgb(xyz);
        assert!(
            approx_eq_arr3(back, rgb, 1e-4),
            "sRGB→XYZ→sRGB round-trip: {back:?}"
        );
    }

    #[test]
    fn test_srgb_to_xyz_black() {
        let xyz = cal_srgb_to_xyz([0.0, 0.0, 0.0]);
        assert!(approx_eq_arr3(xyz, [0.0, 0.0, 0.0], 1e-6));
    }

    #[test]
    fn test_srgb_to_xyz_white() {
        // Linear sRGB white → D65 XYZ
        let xyz = cal_srgb_to_xyz([1.0, 1.0, 1.0]);
        // Should be approximately [0.9505, 1.0000, 1.0890]
        assert!(
            approx_eq(xyz[1], 1.0, 1e-4),
            "Y of white should be 1.0, got {}",
            xyz[1]
        );
    }

    #[test]
    fn test_xyz_to_lab_roundtrip() {
        let xyz = [0.2f32, 0.3, 0.4];
        let lab = cal_xyz_to_lab(xyz);
        let back = cal_lab_to_xyz(lab);
        assert!(
            approx_eq_arr3(back, xyz, 1e-4),
            "XYZ→Lab→XYZ round-trip: {back:?}"
        );
    }

    #[test]
    fn test_xyz_to_lab_d65_white() {
        // D65 XYZ white → Lab (100, 0, 0)
        let lab = cal_xyz_to_lab(D65_XYZ);
        assert!(
            approx_eq(lab[0], 100.0, 1e-3),
            "L* of D65 white should be 100, got {}",
            lab[0]
        );
        assert!(
            approx_eq(lab[1], 0.0, 1e-3),
            "a* should be 0, got {}",
            lab[1]
        );
        assert!(
            approx_eq(lab[2], 0.0, 1e-3),
            "b* should be 0, got {}",
            lab[2]
        );
    }

    #[test]
    fn test_srgb_to_lab_white() {
        // Linear sRGB [1,1,1] → Lab approximately (100, 0, 0)
        let lab = cal_srgb_to_lab([1.0, 1.0, 1.0]);
        assert!(
            approx_eq(lab[0], 100.0, 1.0),
            "L* of linear white should be ~100, got {}",
            lab[0]
        );
        assert!(lab[1].abs() < 1.0, "a* should be near 0, got {}", lab[1]);
        assert!(lab[2].abs() < 1.0, "b* should be near 0, got {}", lab[2]);
    }

    #[test]
    fn test_srgb_to_lab_black() {
        let lab = cal_srgb_to_lab([0.0, 0.0, 0.0]);
        assert!(
            approx_eq(lab[0], 0.0, 1e-3),
            "L* of black should be 0, got {}",
            lab[0]
        );
    }

    // ── Delta E ───────────────────────────────────────────────────────────────

    #[test]
    fn test_delta_e_76_same_color() {
        let lab = [50.0f32, 10.0, -10.0];
        assert!(approx_eq(cal_delta_e_76(lab, lab), 0.0, 1e-6));
    }

    #[test]
    fn test_delta_e_76_known_value() {
        let lab1 = [50.0f32, 0.0, 0.0];
        let lab2 = [53.0f32, 4.0, 0.0];
        let de = cal_delta_e_76(lab1, lab2);
        assert!(approx_eq(de, 5.0, 1e-4), "ΔE76 should be 5.0, got {de}");
    }

    #[test]
    fn test_delta_e_76_positive() {
        let lab1 = [50.0f32, 20.0, 10.0];
        let lab2 = [30.0f32, -10.0, 5.0];
        assert!(cal_delta_e_76(lab1, lab2) > 0.0);
    }

    #[test]
    fn test_delta_e_2000_same_color() {
        let lab = [50.0f32, 10.0, -5.0];
        let de = cal_delta_e_2000(lab, lab);
        assert!(
            approx_eq(de, 0.0, 1e-5),
            "ΔE2000 of same color should be 0, got {de}"
        );
    }

    #[test]
    fn test_delta_e_2000_positive_different() {
        let lab1 = [50.0f32, 25.0, 10.0];
        let lab2 = [60.0f32, -15.0, -5.0];
        let de = cal_delta_e_2000(lab1, lab2);
        assert!(de > 0.0, "ΔE2000 should be positive for different colors");
    }

    #[test]
    fn test_delta_e_2000_small_shift() {
        // A small lightness shift should give a small but non-zero ΔE
        let lab1 = [50.0f32, 0.0, 0.0];
        let lab2 = [51.0f32, 0.0, 0.0];
        let de = cal_delta_e_2000(lab1, lab2);
        assert!(
            de > 0.0 && de < 5.0,
            "small shift ΔE2000 should be small: {de}"
        );
    }

    #[test]
    fn test_delta_e_2000_symmetry() {
        let lab1 = [50.0f32, 20.0, -10.0];
        let lab2 = [60.0f32, -5.0, 15.0];
        let de12 = cal_delta_e_2000(lab1, lab2);
        let de21 = cal_delta_e_2000(lab2, lab1);
        assert!(
            approx_eq(de12, de21, 1e-4),
            "ΔE2000 should be symmetric: {de12} vs {de21}"
        );
    }

    // ── CalibrationStats ──────────────────────────────────────────────────────

    #[test]
    fn test_evaluate_perfect_match() {
        let patches: Vec<[f32; 3]> = vec![[0.2, 0.3, 0.4], [0.5, 0.1, 0.6], [0.1, 0.8, 0.2]];
        let ccm = CalibrationMatrix::identity();
        let wb = WhiteBalance::daylight();
        let stats = cal_evaluate(&patches, &patches, &ccm, &wb).expect("should succeed");
        assert!(
            approx_eq(stats.mean_delta_e, 0.0, 1e-3),
            "perfect match should have mean ΔE ≈ 0, got {}",
            stats.mean_delta_e
        );
        assert!(approx_eq(stats.rmse_rgb, 0.0, 1e-5));
    }

    #[test]
    fn test_evaluate_empty_input() {
        let ccm = CalibrationMatrix::identity();
        let wb = WhiteBalance::daylight();
        assert!(cal_evaluate(&[], &[], &ccm, &wb).is_err());
    }

    #[test]
    fn test_evaluate_dimension_mismatch() {
        let meas: Vec<[f32; 3]> = vec![[0.5, 0.3, 0.1], [0.2, 0.4, 0.6]];
        let refr: Vec<[f32; 3]> = vec![[0.5, 0.3, 0.1]];
        let ccm = CalibrationMatrix::identity();
        let wb = WhiteBalance::daylight();
        assert!(cal_evaluate(&meas, &refr, &ccm, &wb).is_err());
    }

    #[test]
    fn test_evaluate_returns_correct_patch_count() {
        let patches: Vec<[f32; 3]> = vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.5]];
        let ccm = CalibrationMatrix::identity();
        let wb = WhiteBalance::daylight();
        let stats = cal_evaluate(&patches, &patches, &ccm, &wb).expect("ok");
        assert_eq!(stats.per_patch_delta_e.len(), 3);
    }

    #[test]
    fn test_evaluate_nonzero_error() {
        let meas: Vec<[f32; 3]> = vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.5]];
        let refr: Vec<[f32; 3]> = vec![[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]];
        let ccm = CalibrationMatrix::identity();
        let wb = WhiteBalance::daylight();
        let stats = cal_evaluate(&meas, &refr, &ccm, &wb).expect("ok");
        assert!(stats.mean_delta_e > 0.0);
    }

    // ── cal_format_stats ──────────────────────────────────────────────────────

    #[test]
    fn test_format_stats_non_empty() {
        let stats = CalibrationStats {
            mean_delta_e: 2.5,
            max_delta_e: 5.1,
            min_delta_e: 0.3,
            rmse_rgb: 0.012,
            per_patch_delta_e: vec![2.0, 3.0, 2.5],
        };
        let s = cal_format_stats(&stats);
        assert!(!s.is_empty(), "formatted stats should not be empty");
        assert!(s.contains("2.5"), "should contain mean ΔE");
        assert!(
            s.contains("5.1") || s.contains("5.10"),
            "should contain max ΔE"
        );
    }

    #[test]
    fn test_format_stats_zero() {
        let stats = CalibrationStats {
            mean_delta_e: 0.0,
            max_delta_e: 0.0,
            min_delta_e: 0.0,
            rmse_rgb: 0.0,
            per_patch_delta_e: vec![0.0],
        };
        let s = cal_format_stats(&stats);
        assert!(!s.is_empty());
    }

    // ── cal_macbeth_patches ───────────────────────────────────────────────────

    #[test]
    fn test_macbeth_patches_count() {
        let patches = cal_macbeth_patches();
        assert_eq!(patches.len(), 24);
    }

    #[test]
    fn test_macbeth_patches_channels_in_range() {
        for patch in cal_macbeth_patches().iter() {
            for &c in &patch.reference_rgb {
                assert!(
                    (0.0..=1.0).contains(&c),
                    "patch '{}' has out-of-range channel {c}",
                    patch.name
                );
            }
        }
    }

    #[test]
    fn test_macbeth_patches_names_nonempty() {
        for patch in cal_macbeth_patches().iter() {
            assert!(!patch.name.is_empty());
        }
    }

    #[test]
    fn test_macbeth_patches_grayscale_monotone() {
        // Neutral patches (indices 18–23) should be roughly neutral (R≈G≈B)
        let patches = cal_macbeth_patches();
        for patch in &patches[18..24] {
            let [r, g, b] = patch.reference_rgb;
            assert!(
                (r - g).abs() < 0.01 && (g - b).abs() < 0.01,
                "Neutral patch '{}' is not neutral: [{r},{g},{b}]",
                patch.name
            );
        }
    }

    #[test]
    fn test_macbeth_white_brighter_than_black() {
        let patches = cal_macbeth_patches();
        let white = patches[18].reference_rgb[0]; // White patch luminance
        let black = patches[23].reference_rgb[0]; // Black patch luminance
        assert!(
            white > black,
            "White patch should be brighter than Black: {white} vs {black}"
        );
    }

    // ── hue_angle ─────────────────────────────────────────────────────────────

    #[test]
    fn test_hue_angle_zero_zero() {
        assert_eq!(hue_angle(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_hue_angle_positive_a() {
        let h = hue_angle(1.0, 0.0);
        assert!(
            approx_eq(h, 0.0, 1e-4),
            "positive a', b=0 → hue=0°, got {h}"
        );
    }

    #[test]
    fn test_hue_angle_positive_b() {
        let h = hue_angle(0.0, 1.0);
        assert!(
            approx_eq(h, 90.0, 1e-4),
            "a'=0, positive b → hue=90°, got {h}"
        );
    }

    #[test]
    fn test_hue_angle_range() {
        for a in [-1.0f32, 0.0, 1.0] {
            for b in [-1.0f32, 0.0, 1.0] {
                let h = hue_angle(a, b);
                assert!(
                    (0.0..360.0).contains(&h),
                    "hue out of range for a={a}, b={b}: {h}"
                );
            }
        }
    }
}
