//! CPU-side spherical harmonics utilities for 3D Gaussian Splatting.
//!
//! Implements real SH basis evaluation, coefficient-to-color conversion,
//! Monte Carlo projection, degree conversion, DC operations, Z-axis rotation,
//! and band energy computation — all matching the 3DGS shader convention.
//!
//! ## Coefficient Layout
//!
//! Functions operating on multi-channel SH use **per-channel** layout:
//! `[r0..rN, g0..gN, b0..bN]` where `N = (degree+1)^2` coefficients per channel.
//! Total length = `3 * N`.

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// L=0 normalisation: 1 / (2√π).
pub const SH_C0: f32 = 0.282_094_8_f32;

/// L=1 normalisation: √(3/(4π)).
pub const SH_C1: f32 = 0.488_602_52_f32;

/// L=2 normalisations (one per m-index: m = −2,−1,0,+1,+2).
pub const SH_C2: [f32; 5] = [
    1.092_548_5_f32,  // m=−2: √(15/(4π))
    1.092_548_5_f32,  // m=−1: √(15/(4π))
    0.315_391_57_f32, // m= 0: √(5/(16π))
    1.092_548_5_f32,  // m=+1: √(15/(4π))
    0.546_274_24_f32, // m=+2: √(15/(16π))
];

/// L=3 normalisations (one per m-index: m = −3..=+3).
pub const SH_C3: [f32; 7] = [
    0.590_043_6_f32,  // m=−3: √(35/(32π)) * √2
    2.890_611_4_f32,  // m=−2: √(105/(4π))
    0.457_045_8_f32,  // m=−1: √(21/(32π)) * √2
    0.373_176_34_f32, // m= 0: √(7/(16π))
    0.457_045_8_f32,  // m=+1: √(21/(32π)) * √2
    1.445_305_7_f32,  // m=+2: √(105/(16π))
    0.590_043_6_f32,  // m=+3: √(35/(32π)) * √2
];

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by spherical harmonics operations.
#[derive(Debug, thiserror::Error)]
pub enum ShError {
    /// SH degree > 3 is not supported.
    #[error("Unsupported SH degree: {0}, max is 3")]
    UnsupportedDegree(usize),

    /// Coefficient slice length does not match the expected count.
    #[error("Coefficient count mismatch: expected {expected}, got {got}")]
    CoeffCountMismatch { expected: usize, got: usize },

    /// Supplied direction vector has near-zero length.
    #[error("Direction vector must be non-zero")]
    ZeroDirection,

    /// Rotation matrix supplied to a function is not 3×3 row-major.
    #[error("Rotation matrix must be 3x3 row-major")]
    InvalidRotationMatrix,
}

// ---------------------------------------------------------------------------
// ShBasis descriptor
// ---------------------------------------------------------------------------

/// Descriptor for a particular SH degree.
#[derive(Debug, Clone)]
pub struct ShBasis {
    /// Maximum SH degree (0–3).
    pub degree: usize,
    /// Total number of basis functions: `(degree+1)^2`.
    pub num_coeffs: usize,
}

impl ShBasis {
    /// Construct from degree, or error if > 3.
    pub fn new(degree: usize) -> Result<Self, ShError> {
        if degree > 3 {
            return Err(ShError::UnsupportedDegree(degree));
        }
        Ok(Self {
            degree,
            num_coeffs: sh_num_coeffs(degree),
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the number of real SH basis functions through the given degree: `(degree+1)^2`.
#[inline]
pub fn sh_num_coeffs(degree: usize) -> usize {
    (degree + 1) * (degree + 1)
}

/// Normalize a direction; error if norm < 1e-7.
#[inline]
fn normalize(direction: [f32; 3]) -> Result<[f32; 3], ShError> {
    let [x, y, z] = direction;
    let norm = (x * x + y * y + z * z).sqrt();
    if norm < 1e-7 {
        return Err(ShError::ZeroDirection);
    }
    Ok([x / norm, y / norm, z / norm])
}

// ---------------------------------------------------------------------------
// Per-degree basis evaluation
// ---------------------------------------------------------------------------

/// Evaluate the single L=0 SH basis value.
#[inline]
pub fn sh_eval_degree0(_x: f32, _y: f32, _z: f32) -> [f32; 1] {
    [SH_C0]
}

/// Evaluate L=0 and L=1 SH basis values (4 total).
///
/// Output order: `[Y_0^0, Y_1^{-1}, Y_1^0, Y_1^1]`
#[inline]
pub fn sh_eval_degree1(x: f32, y: f32, z: f32) -> [f32; 4] {
    [SH_C0, SH_C1 * y, SH_C1 * z, SH_C1 * x]
}

/// Evaluate L=0..=2 SH basis values (9 total).
///
/// Output appends L=2 to `sh_eval_degree1`:
/// `[..., Y_2^{-2}, Y_2^{-1}, Y_2^0, Y_2^1, Y_2^2]`
#[inline]
pub fn sh_eval_degree2(x: f32, y: f32, z: f32) -> [f32; 9] {
    let [b0, b1, b2, b3] = sh_eval_degree1(x, y, z);
    [
        b0,
        b1,
        b2,
        b3,
        SH_C2[0] * x * y,
        SH_C2[1] * y * z,
        SH_C2[2] * (2.0 * z * z - x * x - y * y),
        SH_C2[3] * x * z,
        SH_C2[4] * (x * x - y * y),
    ]
}

/// Evaluate L=0..=3 SH basis values (16 total).
///
/// Output appends L=3 to `sh_eval_degree2`:
/// `[..., Y_3^{-3}, Y_3^{-2}, Y_3^{-1}, Y_3^0, Y_3^1, Y_3^2, Y_3^3]`
#[inline]
pub fn sh_eval_degree3(x: f32, y: f32, z: f32) -> [f32; 16] {
    let [b0, b1, b2, b3, b4, b5, b6, b7, b8] = sh_eval_degree2(x, y, z);
    [
        b0,
        b1,
        b2,
        b3,
        b4,
        b5,
        b6,
        b7,
        b8,
        SH_C3[0] * y * (3.0 * x * x - y * y),
        SH_C3[1] * x * y * z,
        SH_C3[2] * y * (4.0 * z * z - x * x - y * y),
        SH_C3[3] * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y),
        SH_C3[4] * x * (4.0 * z * z - x * x - y * y),
        SH_C3[5] * z * (x * x - y * y),
        SH_C3[6] * x * (x * x - 3.0 * y * y),
    ]
}

/// Evaluate real SH basis up to `degree` for `direction` (normalises first).
///
/// Returns a `Vec<f32>` of length `(degree+1)^2`.
pub fn sh_eval(direction: [f32; 3], degree: usize) -> Result<Vec<f32>, ShError> {
    if degree > 3 {
        return Err(ShError::UnsupportedDegree(degree));
    }
    let [x, y, z] = normalize(direction)?;
    let out: Vec<f32> = match degree {
        0 => sh_eval_degree0(x, y, z).to_vec(),
        1 => sh_eval_degree1(x, y, z).to_vec(),
        2 => sh_eval_degree2(x, y, z).to_vec(),
        _ => sh_eval_degree3(x, y, z).to_vec(),
    };
    Ok(out)
}

// ---------------------------------------------------------------------------
// SH coefficients → RGB color
// ---------------------------------------------------------------------------

/// Convert per-channel SH coefficients to a view-dependent RGB colour.
///
/// `sh_coeffs` layout: `[r_0..r_N, g_0..g_N, b_0..b_N]` where `N = (degree+1)^2`.
/// Total required length: `3 * N`.
///
/// Returns clamped `[0, 1]` RGB: `color = 0.5 + Σ_k (c_k * Y_k)`.
pub fn sh_to_color(
    sh_coeffs: &[f32],
    direction: [f32; 3],
    degree: usize,
) -> Result<[f32; 3], ShError> {
    if degree > 3 {
        return Err(ShError::UnsupportedDegree(degree));
    }
    let n = sh_num_coeffs(degree);
    let expected = 3 * n;
    if sh_coeffs.len() != expected {
        return Err(ShError::CoeffCountMismatch {
            expected,
            got: sh_coeffs.len(),
        });
    }
    let basis = sh_eval(direction, degree)?;
    let mut rgb = [0.5_f32; 3];
    for (ch, rgb_val) in rgb.iter_mut().enumerate() {
        let offset = ch * n;
        for (k, &basis_val) in basis.iter().enumerate().take(n) {
            *rgb_val += sh_coeffs[offset + k] * basis_val;
        }
    }
    Ok([
        rgb[0].clamp(0.0, 1.0),
        rgb[1].clamp(0.0, 1.0),
        rgb[2].clamp(0.0, 1.0),
    ])
}

// ---------------------------------------------------------------------------
// DC helpers
// ---------------------------------------------------------------------------

/// Convert degree-0 SH coefficients to RGB (with 3DGS +0.5 offset), clamped \[0,1\].
#[inline]
pub fn sh_dc_to_color(dc_r: f32, dc_g: f32, dc_b: f32) -> [f32; 3] {
    [
        (0.5 + SH_C0 * dc_r).clamp(0.0, 1.0),
        (0.5 + SH_C0 * dc_g).clamp(0.0, 1.0),
        (0.5 + SH_C0 * dc_b).clamp(0.0, 1.0),
    ]
}

/// Inverse of `sh_dc_to_color`: convert linear RGB to degree-0 SH coefficients.
#[inline]
pub fn color_to_sh_dc(r: f32, g: f32, b: f32) -> [f32; 3] {
    [(r - 0.5) / SH_C0, (g - 0.5) / SH_C0, (b - 0.5) / SH_C0]
}

// ---------------------------------------------------------------------------
// Monte Carlo SH projection
// ---------------------------------------------------------------------------

/// Project a direction-indexed RGB function to SH coefficients via Monte Carlo integration.
///
/// `samples`: slice of `(unit_direction, rgb_color)` pairs (uniform sphere distribution).
///
/// Returns per-channel coefficients: `[r_0..r_N, g_0..g_N, b_0..b_N]`, length `3*N`.
/// Weight per sample: `4π / num_samples`.
pub fn sh_project_monte_carlo(
    samples: &[([f32; 3], [f32; 3])],
    degree: usize,
) -> Result<Vec<f32>, ShError> {
    if degree > 3 {
        return Err(ShError::UnsupportedDegree(degree));
    }
    let n = sh_num_coeffs(degree);
    let num = samples.len();
    let mut coeffs = vec![0.0_f32; 3 * n];
    if num == 0 {
        return Ok(coeffs);
    }
    let weight = 4.0 * std::f32::consts::PI / num as f32;
    for (dir, rgb) in samples {
        let basis = sh_eval(*dir, degree)?;
        for k in 0..n {
            for ch in 0..3 {
                coeffs[ch * n + k] += rgb[ch] * basis[k] * weight;
            }
        }
    }
    // Normalize by 1/N (weight already includes the 4π factor but not the 1/N division)
    // Actually weight = 4π/N already encodes the 1/N factor — accumulation is a sum,
    // so dividing by N again would double-count. The sum itself IS the integral estimate.
    Ok(coeffs)
}

// ---------------------------------------------------------------------------
// Degree conversion
// ---------------------------------------------------------------------------

/// Upsample SH coefficients from `from_degree` to `to_degree` by zero-padding.
pub fn sh_upsample(
    coeffs: &[f32],
    from_degree: usize,
    to_degree: usize,
) -> Result<Vec<f32>, ShError> {
    if from_degree > 3 {
        return Err(ShError::UnsupportedDegree(from_degree));
    }
    if to_degree > 3 {
        return Err(ShError::UnsupportedDegree(to_degree));
    }
    let n_from = sh_num_coeffs(from_degree);
    let expected = 3 * n_from;
    if coeffs.len() != expected {
        return Err(ShError::CoeffCountMismatch {
            expected,
            got: coeffs.len(),
        });
    }
    if to_degree <= from_degree {
        return Ok(coeffs.to_vec());
    }
    let n_to = sh_num_coeffs(to_degree);
    let mut out = vec![0.0_f32; 3 * n_to];
    for ch in 0..3 {
        let src_offset = ch * n_from;
        let dst_offset = ch * n_to;
        out[dst_offset..dst_offset + n_from]
            .copy_from_slice(&coeffs[src_offset..src_offset + n_from]);
    }
    Ok(out)
}

/// Downsample SH coefficients from `from_degree` to `to_degree` by truncation.
pub fn sh_downsample(
    coeffs: &[f32],
    from_degree: usize,
    to_degree: usize,
) -> Result<Vec<f32>, ShError> {
    if from_degree > 3 {
        return Err(ShError::UnsupportedDegree(from_degree));
    }
    if to_degree > 3 {
        return Err(ShError::UnsupportedDegree(to_degree));
    }
    let n_from = sh_num_coeffs(from_degree);
    let expected = 3 * n_from;
    if coeffs.len() != expected {
        return Err(ShError::CoeffCountMismatch {
            expected,
            got: coeffs.len(),
        });
    }
    if to_degree >= from_degree {
        return Ok(coeffs.to_vec());
    }
    let n_to = sh_num_coeffs(to_degree);
    let mut out = vec![0.0_f32; 3 * n_to];
    for ch in 0..3 {
        let src_offset = ch * n_from;
        let dst_offset = ch * n_to;
        out[dst_offset..dst_offset + n_to].copy_from_slice(&coeffs[src_offset..src_offset + n_to]);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Z-axis rotation
// ---------------------------------------------------------------------------

/// Rotate SH coefficients around the Z axis by `angle_rad`.
///
/// For each band `l` and signed order `m ≠ 0`:
/// - `new[l^{+|m|}] = old[l^{+|m|}] * cos(m*θ) − old[l^{-|m|}] * sin(m*θ)`
/// - `new[l^{-|m|}] = old[l^{-|m|}] * cos(m*θ) + old[l^{+|m|}] * sin(m*θ)`
///
/// `m=0` coefficients are unchanged. Applied independently per channel.
pub fn rotate_sh_coeffs_z(
    coeffs: &[f32],
    angle_rad: f32,
    degree: usize,
) -> Result<Vec<f32>, ShError> {
    if degree > 3 {
        return Err(ShError::UnsupportedDegree(degree));
    }
    let n = sh_num_coeffs(degree);
    let expected = 3 * n;
    if coeffs.len() != expected {
        return Err(ShError::CoeffCountMismatch {
            expected,
            got: coeffs.len(),
        });
    }
    let mut out = coeffs.to_vec();
    for l in 1..=degree {
        for m in 1..=l {
            let mf = m as f32;
            let c = (mf * angle_rad).cos();
            let s = (mf * angle_rad).sin();
            // Within a channel block, index of Y_l^{-m} is l*l + (l - m),
            // and Y_l^{+m} is l*l + (l + m).
            let idx_neg = l * l + (l - m); // Y_l^{-m}
            let idx_pos = l * l + (l + m); // Y_l^{+m}
            for ch in 0..3 {
                let base = ch * n;
                let old_pos = coeffs[base + idx_pos];
                let old_neg = coeffs[base + idx_neg];
                out[base + idx_pos] = old_pos * c - old_neg * s;
                out[base + idx_neg] = old_neg * c + old_pos * s;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Band energy
// ---------------------------------------------------------------------------

/// Compute RMS energy of one SH band across all channels.
///
/// `band`: SH band index (0, 1, 2, or 3).
/// `num_channels`: number of colour channels (normally 3).
pub fn sh_band_energy(coeffs: &[f32], band: usize, num_channels: usize) -> Result<f32, ShError> {
    if band > 3 {
        return Err(ShError::UnsupportedDegree(band));
    }
    let coeffs_per_channel = coeffs.len() / num_channels.max(1);
    let n_band = 2 * band + 1; // number of basis functions in this band
    let band_start = band * band; // l^2
    let band_end = band_start + n_band;
    if band_end > coeffs_per_channel {
        return Err(ShError::CoeffCountMismatch {
            expected: band_end * num_channels,
            got: coeffs.len(),
        });
    }
    let mut sum_sq = 0.0_f32;
    let mut count = 0_usize;
    for ch in 0..num_channels {
        let offset = ch * coeffs_per_channel;
        for k in band_start..band_end {
            let v = coeffs[offset + k];
            sum_sq += v * v;
            count += 1;
        }
    }
    if count == 0 {
        return Ok(0.0);
    }
    Ok((sum_sq / count as f32).sqrt())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // 1. sh_num_coeffs
    #[test]
    fn test_sh_num_coeffs() {
        assert_eq!(sh_num_coeffs(0), 1);
        assert_eq!(sh_num_coeffs(1), 4);
        assert_eq!(sh_num_coeffs(2), 9);
        assert_eq!(sh_num_coeffs(3), 16);
    }

    // 2. sh_eval_degree0 constant
    #[test]
    fn test_sh_eval_degree0_constant() {
        let dirs = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5_f32, 0.5, 0.707]];
        for [x, y, z] in dirs {
            let b = sh_eval_degree0(x, y, z);
            assert!(approx(b[0], SH_C0, EPS), "degree0 must equal SH_C0");
        }
    }

    // 3. sh_eval_degree1 for (1,0,0)
    #[test]
    fn test_sh_eval_degree1_x_axis() {
        let b = sh_eval_degree1(1.0, 0.0, 0.0);
        assert!(approx(b[0], SH_C0, EPS));
        assert!(approx(b[1], 0.0, EPS)); // SH_C1 * y = 0
        assert!(approx(b[2], 0.0, EPS)); // SH_C1 * z = 0
        assert!(approx(b[3], SH_C1, EPS)); // SH_C1 * x = SH_C1
    }

    // 4. sh_eval normalizes non-unit vector
    #[test]
    fn test_sh_eval_normalizes() {
        let unit = sh_eval([1.0, 0.0, 0.0], 2).expect("unit ok");
        let scaled = sh_eval([3.0, 0.0, 0.0], 2).expect("scaled ok");
        for (a, b) in unit.iter().zip(scaled.iter()) {
            assert!(
                approx(*a, *b, EPS),
                "normalized and un-normalized must match"
            );
        }
    }

    // 5. sh_eval ZeroDirection
    #[test]
    fn test_sh_eval_zero_direction_error() {
        let result = sh_eval([0.0, 0.0, 0.0], 1);
        assert!(matches!(result, Err(ShError::ZeroDirection)));
    }

    // 6. sh_eval degree 4 → UnsupportedDegree
    #[test]
    fn test_sh_eval_unsupported_degree() {
        let result = sh_eval([1.0, 0.0, 0.0], 4);
        assert!(matches!(result, Err(ShError::UnsupportedDegree(4))));
    }

    // 7. sh_to_color degree 0 → constant color
    #[test]
    fn test_sh_to_color_degree0_constant() {
        // With degree 0 only, color = clamp(0.5 + SH_C0 * c_r)
        let dc_r = 0.1_f32;
        let dc_g = -0.5_f32;
        let dc_b = 0.3_f32;
        let coeffs = [dc_r, dc_g, dc_b];
        let dir1 = sh_to_color(&coeffs, [1.0, 0.0, 0.0], 0).expect("ok");
        let dir2 = sh_to_color(&coeffs, [0.0, 1.0, 0.0], 0).expect("ok");
        for ch in 0..3 {
            assert!(
                approx(dir1[ch], dir2[ch], EPS),
                "degree-0 color must be view-independent"
            );
        }
        let expected_r = (0.5 + SH_C0 * dc_r).clamp(0.0, 1.0);
        assert!(approx(dir1[0], expected_r, EPS));
    }

    // 8. sh_to_color length mismatch → error
    #[test]
    fn test_sh_to_color_length_mismatch() {
        let bad = [0.0_f32; 5]; // wrong for any degree
        let result = sh_to_color(&bad, [1.0, 0.0, 0.0], 1); // expects 3*4=12
        assert!(matches!(result, Err(ShError::CoeffCountMismatch { .. })));
    }

    // 9. sh_dc_to_color roundtrip
    #[test]
    fn test_sh_dc_to_color_roundtrip() {
        // color_to_sh_dc then sh_dc_to_color should recover original (within [0,1])
        let r = 0.6_f32;
        let g = 0.4_f32;
        let b = 0.8_f32;
        let [dr, dg, db] = color_to_sh_dc(r, g, b);
        let recovered = sh_dc_to_color(dr, dg, db);
        assert!(approx(recovered[0], r, EPS));
        assert!(approx(recovered[1], g, EPS));
        assert!(approx(recovered[2], b, EPS));
    }

    // 10. sh_dc_to_color clamps
    #[test]
    fn test_sh_dc_to_color_clamps() {
        // Very large coefficient → clamped to 1.0
        let out = sh_dc_to_color(100.0, -100.0, 0.0);
        assert!(approx(out[0], 1.0, EPS));
        assert!(approx(out[1], 0.0, EPS));
    }

    // 11. sh_upsample 0→1 adds zeros
    #[test]
    fn test_sh_upsample_0_to_1() {
        let coeffs = [0.5_f32, 0.3, 0.2]; // 3 * 1 coefficients
        let up = sh_upsample(&coeffs, 0, 1).expect("ok");
        assert_eq!(up.len(), 3 * 4); // degree-1 → 3*4=12
                                     // First element of each channel preserved
        assert!(approx(up[0], 0.5, EPS));
        assert!(approx(up[4], 0.3, EPS));
        assert!(approx(up[8], 0.2, EPS));
        // Higher elements are zero
        for &v in &up[1..4] {
            assert!(approx(v, 0.0, EPS));
        }
    }

    // 12. sh_upsample 0→2 adds zeros
    #[test]
    fn test_sh_upsample_0_to_2() {
        let coeffs = [1.0_f32, 2.0, 3.0];
        let up = sh_upsample(&coeffs, 0, 2).expect("ok");
        assert_eq!(up.len(), 3 * 9);
        assert!(approx(up[0], 1.0, EPS));
        assert!(approx(up[9], 2.0, EPS));
        assert!(approx(up[18], 3.0, EPS));
        for (i, &val) in up.iter().enumerate().take(9).skip(1) {
            assert!(approx(val, 0.0, EPS), "up[{i}] should be 0");
        }
    }

    // 13. sh_downsample 2→0 keeps DC only
    #[test]
    fn test_sh_downsample_2_to_0() {
        let mut coeffs = vec![0.0_f32; 3 * 9];
        coeffs[0] = 0.7; // r DC
        coeffs[9] = 0.5; // g DC
        coeffs[18] = 0.3; // b DC
        coeffs[1] = 99.0; // should be discarded
        let down = sh_downsample(&coeffs, 2, 0).expect("ok");
        assert_eq!(down.len(), 3);
        assert!(approx(down[0], 0.7, EPS));
        assert!(approx(down[1], 0.5, EPS));
        assert!(approx(down[2], 0.3, EPS));
    }

    // 14. sh_downsample 2→1 keeps bands 0,1
    #[test]
    fn test_sh_downsample_2_to_1() {
        let mut coeffs = vec![0.0_f32; 3 * 9];
        for i in 0..4 {
            coeffs[i] = i as f32 + 1.0; // r band 0+1
            coeffs[9 + i] = i as f32 + 5.0; // g
            coeffs[18 + i] = i as f32 + 9.0; // b
        }
        coeffs[4] = 999.0; // L=2 → discarded
        let down = sh_downsample(&coeffs, 2, 1).expect("ok");
        assert_eq!(down.len(), 3 * 4);
        for i in 0..4 {
            assert!(approx(down[i], i as f32 + 1.0, EPS));
            assert!(approx(down[4 + i], i as f32 + 5.0, EPS));
            assert!(approx(down[8 + i], i as f32 + 9.0, EPS));
        }
    }

    // 15. sh_project_monte_carlo constant function → DC coefficient matches theory
    #[test]
    fn test_sh_project_monte_carlo_constant() {
        // For a constant function f over S², the SH projection integral is:
        //   c_0 = ∫_{S²} f * Y_0^0 dΩ = f * SH_C0 * 4π
        // The estimator with weight 4π/N gives c_0 ≈ f * SH_C0 * 4π.
        use std::f32::consts::PI;
        let target = [0.6_f32, 0.4_f32, 0.2_f32];
        let n = 2000_usize;
        let golden = (1.0_f32 + 5.0_f32.sqrt()) / 2.0;
        let samples: Vec<([f32; 3], [f32; 3])> = (0..n)
            .map(|i| {
                let fi = i as f32;
                let theta = (1.0 - 2.0 * (fi + 0.5) / n as f32).clamp(-1.0, 1.0).acos();
                let phi = 2.0 * PI * fi / golden;
                let dir = [
                    theta.sin() * phi.cos(),
                    theta.cos(),
                    theta.sin() * phi.sin(),
                ];
                (dir, target)
            })
            .collect();
        let coeffs = sh_project_monte_carlo(&samples, 0).expect("ok");
        // Expected DC coefficient per channel: f * SH_C0 * 4π
        for ch in 0..3 {
            let expected = target[ch] * SH_C0 * 4.0 * PI;
            assert!(
                approx(coeffs[ch], expected, 0.05),
                "ch {ch} DC coeff: got {}, expected {expected}",
                coeffs[ch]
            );
        }
    }

    // 16. sh_band_energy for known coefficients
    #[test]
    fn test_sh_band_energy_band0() {
        // 3-channel degree-0: coeffs = [r, g, b]
        // Band 0 has 1 coeff per channel: energy = sqrt((r^2 + g^2 + b^2) / 3)
        let coeffs = [2.0_f32, 0.0, 0.0]; // 1 coeff per channel
                                          // But wait — for degree-0, num_channels * 1 = 3 coeffs total, coeffs_per_channel = 1
        let energy = sh_band_energy(&coeffs, 0, 3).expect("ok");
        let expected = ((4.0 + 0.0 + 0.0) / 3.0_f32).sqrt();
        assert!(
            approx(energy, expected, EPS),
            "got {energy}, expected {expected}"
        );
    }

    // 17. rotate_sh_coeffs_z angle=0 → identity
    #[test]
    fn test_rotate_sh_coeffs_z_zero_angle() {
        let coeffs: Vec<f32> = (0..3 * 9).map(|i| i as f32 * 0.1).collect();
        let rotated = rotate_sh_coeffs_z(&coeffs, 0.0, 2).expect("ok");
        for (a, b) in coeffs.iter().zip(rotated.iter()) {
            assert!(approx(*a, *b, EPS));
        }
    }

    // 18. rotate_sh_coeffs_z angle=2π → identity
    #[test]
    fn test_rotate_sh_coeffs_z_full_revolution() {
        let coeffs: Vec<f32> = (0..3 * 16).map(|i| (i as f32 - 24.0) * 0.05).collect();
        let rotated = rotate_sh_coeffs_z(&coeffs, 2.0 * std::f32::consts::PI, 3).expect("ok");
        for (a, b) in coeffs.iter().zip(rotated.iter()) {
            assert!(approx(*a, *b, 1e-4), "original {a} vs rotated {b}");
        }
    }

    // 19. sh_to_color degree 1 with known direction
    #[test]
    fn test_sh_to_color_degree1_known() {
        // degree-1: 4 basis, 12 coefficients
        // Layout [r0,r1,r2,r3, g0,g1,g2,g3, b0,b1,b2,b3]
        let mut coeffs = [0.0_f32; 12];
        // Set only the DC term for R channel:
        coeffs[0] = 0.2; // r_0
        coeffs[4] = 0.3; // g_0
        coeffs[8] = 0.4; // b_0
        let dir = [0.0_f32, 1.0, 0.0];
        let color = sh_to_color(&coeffs, dir, 1).expect("ok");
        let expected_r = (0.5 + SH_C0 * 0.2).clamp(0.0, 1.0);
        let expected_g = (0.5 + SH_C0 * 0.3).clamp(0.0, 1.0);
        let expected_b = (0.5 + SH_C0 * 0.4).clamp(0.0, 1.0);
        assert!(approx(color[0], expected_r, 1e-4));
        assert!(approx(color[1], expected_g, 1e-4));
        assert!(approx(color[2], expected_b, 1e-4));
    }

    // 20. sh_eval_degree2 orthogonality check at 6 cardinal directions
    #[test]
    fn test_sh_eval_degree2_cardinal_sum() {
        // Sum of Y_2^0 over 6 cardinal directions should be zero (alternating sign).
        // Y_2^0 = SH_C2[2] * (2z^2 - x^2 - y^2)
        // +x: 2(0)-1-0 = -1, -x: same, +y: 0-0-1=-1, -y: same, +z: 2-0-0=2, -z: same
        // Sum = 4*(-1)*C + 2*(2)*C = -4C + 4C = 0 ✓ (orthogonality)
        let cards = [
            [1.0_f32, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let y20_sum: f32 = cards
            .iter()
            .map(|&[x, y, z]| {
                sh_eval_degree2(x, y, z)[6] // index 6 = Y_2^0
            })
            .sum();
        assert!(approx(y20_sum, 0.0, EPS), "Y_2^0 cardinal sum = {y20_sum}");
    }
}
