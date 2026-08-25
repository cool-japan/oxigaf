//! Diffuse image-based lighting evaluation.

use std::f32::consts::PI;

use super::error::LightProbeError;
use super::irradiance::IrradianceSH;

// ---------------------------------------------------------------------------
// Diffuse IBL
// ---------------------------------------------------------------------------

/// Apply diffuse image-based lighting to a single Gaussian.
///
/// `L_diffuse = E(n) * albedo / π`
///
/// # Errors
/// - `LightProbeError::ZeroDirection` if `normal` is degenerate.
pub fn lp_evaluate_diffuse_ibl(
    normal: [f32; 3],
    sh: &IrradianceSH,
    albedo: [f32; 3],
) -> Result<[f32; 3], LightProbeError> {
    let irr = sh.evaluate(normal)?;
    let inv_pi = 1.0 / PI;
    Ok([
        irr[0] * albedo[0] * inv_pi,
        irr[1] * albedo[1] * inv_pi,
        irr[2] * albedo[2] * inv_pi,
    ])
}

/// Apply diffuse IBL to `n_gaussians` Gaussians.
///
/// `normals`: flat `[N × 3]` f32 array (x, y, z per Gaussian).
/// `albedo`:  flat `[N × 3]` f32 array (R, G, B per Gaussian).
///
/// Returns flat `[N × 3]` f32 RGB output.
///
/// # Errors
/// - `LightProbeError::BufferMismatch` if `normals.len()` or `albedo.len()` != `n_gaussians * 3`.
/// - `LightProbeError::ZeroDirection` if any normal is degenerate.
pub fn lp_apply_ibl_to_gaussians(
    normals: &[f32],
    sh: &IrradianceSH,
    albedo: &[f32],
    n_gaussians: usize,
) -> Result<Vec<f32>, LightProbeError> {
    let expected = n_gaussians * 3;
    if normals.len() != expected {
        return Err(LightProbeError::BufferMismatch {
            expected,
            got: normals.len(),
        });
    }
    if albedo.len() != expected {
        return Err(LightProbeError::BufferMismatch {
            expected,
            got: albedo.len(),
        });
    }

    let mut out = Vec::with_capacity(expected);
    for i in 0..n_gaussians {
        let base = i * 3;
        let normal = [normals[base], normals[base + 1], normals[base + 2]];
        let alb = [albedo[base], albedo[base + 1], albedo[base + 2]];
        let rgb = lp_evaluate_diffuse_ibl(normal, sh, alb)?;
        out.push(rgb[0]);
        out.push(rgb[1]);
        out.push(rgb[2]);
    }
    Ok(out)
}
