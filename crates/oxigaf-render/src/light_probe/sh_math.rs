//! Spherical harmonics basis constants and basis-function evaluation.
//!
//! Private copies of the SH normalisation constants (avoids collision with
//! `spherical_harmonics` re-exports elsewhere in the crate).

use super::error::LightProbeError;

// ---------------------------------------------------------------------------
// SH constants (private copies; avoids collision with spherical_harmonics re-exports)
// ---------------------------------------------------------------------------

/// L=0 normalisation: 1 / (2√π) ≈ 0.282_094_791_77 (rounded to the nearest f32).
pub(super) const LP_SH_C0: f32 = 0.282_094_8_f32;

/// L=1 normalisation: √(3/(4π)) ≈ 0.488_602_511_90 (rounded to the nearest f32).
pub(super) const LP_SH_C1: f32 = 0.488_602_52_f32;

/// L=2 normalisations (m = −2,−1,0,+1,+2), each rounded to the nearest f32.
const LP_SH_C2: [f32; 5] = [
    1.092_548_5_f32,  // m=−2: √(15/(4π)) ≈ 1.092_548_430_59
    1.092_548_5_f32,  // m=−1: √(15/(4π)) ≈ 1.092_548_430_59
    0.315_391_57_f32, // m= 0: √(5/(16π)) ≈ 0.315_391_565_25
    1.092_548_5_f32,  // m=+1: √(15/(4π)) ≈ 1.092_548_430_59
    0.546_274_24_f32, // m=+2: (1/2)√(15/π) ≈ 0.546_274_215_29, i.e. half of √(15/(4π))
];

// ---------------------------------------------------------------------------
// SH basis functions
// ---------------------------------------------------------------------------

/// Normalize a direction; error if norm < 1e-7.
#[inline]
pub fn lp_normalize_dir(dir: [f32; 3]) -> Result<[f32; 3], LightProbeError> {
    let [x, y, z] = dir;
    let norm = (x * x + y * y + z * z).sqrt();
    if norm < 1e-7 {
        return Err(LightProbeError::ZeroDirection);
    }
    Ok([x / norm, y / norm, z / norm])
}

/// Evaluate the single L=0 SH basis function: Y_0^0 = 1/(2√π).
///
/// `dir` does **not** need to be a unit vector for this function (result is constant).
#[inline]
pub fn lp_sh_basis_l0(_dir: [f32; 3]) -> [f32; 1] {
    [LP_SH_C0]
}

/// Evaluate L=1 SH basis functions (3 values, excluding L=0).
///
/// `dir` should be a unit vector.
/// Returns `[Y_1^{-1}, Y_1^0, Y_1^1]`.
#[inline]
pub fn lp_sh_basis_l1(dir: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = dir;
    [LP_SH_C1 * y, LP_SH_C1 * z, LP_SH_C1 * x]
}

/// Evaluate L=2 SH basis functions (5 values, excluding L=0 and L=1).
///
/// `dir` should be a unit vector.
/// Returns `[Y_2^{-2}, Y_2^{-1}, Y_2^0, Y_2^1, Y_2^2]`.
#[inline]
pub fn lp_sh_basis_l2(dir: [f32; 3]) -> [f32; 5] {
    let [x, y, z] = dir;
    [
        LP_SH_C2[0] * x * y,
        LP_SH_C2[1] * y * z,
        LP_SH_C2[2] * (2.0 * z * z - x * x - y * y),
        LP_SH_C2[3] * x * z,
        LP_SH_C2[4] * (x * x - y * y),
    ]
}

/// Evaluate all SH basis functions up to `order` (1=L0 only, 2=L0+L1, 3=L0+L1+L2).
///
/// Returns 1, 4, or 9 coefficients respectively.
/// The `dir` is normalized internally.
///
/// # Errors
/// - `LightProbeError::InvalidOrder` if order ∉ {1, 2, 3}
/// - `LightProbeError::ZeroDirection` if `dir` has near-zero length
pub fn lp_sh_basis(dir: [f32; 3], order: usize) -> Result<Vec<f32>, LightProbeError> {
    let d = lp_normalize_dir(dir)?;
    match order {
        1 => Ok(lp_sh_basis_l0(d).to_vec()),
        2 => {
            let mut v = Vec::with_capacity(4);
            v.extend_from_slice(&lp_sh_basis_l0(d));
            v.extend_from_slice(&lp_sh_basis_l1(d));
            Ok(v)
        }
        3 => {
            let mut v = Vec::with_capacity(9);
            v.extend_from_slice(&lp_sh_basis_l0(d));
            v.extend_from_slice(&lp_sh_basis_l1(d));
            v.extend_from_slice(&lp_sh_basis_l2(d));
            Ok(v)
        }
        _ => Err(LightProbeError::InvalidOrder { order }),
    }
}

/// Compute all 9 SH basis values for a unit direction (internal helper).
///
/// Shared by [`super::irradiance::IrradianceSH::evaluate`] and the Monte
/// Carlo SH projection in [`super::projection`].
#[inline]
pub(super) fn lp_sh_full_9(dir: [f32; 3]) -> [f32; 9] {
    let l0 = lp_sh_basis_l0(dir);
    let l1 = lp_sh_basis_l1(dir);
    let l2 = lp_sh_basis_l2(dir);
    [
        l0[0], l1[0], l1[1], l1[2], l2[0], l2[1], l2[2], l2[3], l2[4],
    ]
}
