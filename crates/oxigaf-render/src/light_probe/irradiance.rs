//! 27-coefficient (9 SH basis × 3 channels) irradiance representation.

use super::error::LightProbeError;
use super::sh_math::{lp_normalize_dir, lp_sh_full_9, LP_SH_C0};

// ---------------------------------------------------------------------------
// IrradianceSH
// ---------------------------------------------------------------------------

/// 27-coefficient (9 SH basis × 3 channels) irradiance representation.
///
/// Layout: `coefficients[basis_i * 3 + channel]` where `channel` ∈ {0=R, 1=G, 2=B}.
#[derive(Debug, Clone, PartialEq)]
pub struct IrradianceSH {
    /// Flat array of 27 coefficients (interleaved RGB).
    pub coefficients: [f32; 27],
    /// SH order (always 2 = L=2 for this struct).
    pub order: usize,
}

impl IrradianceSH {
    /// Construct a zero-initialized `IrradianceSH` of order 2.
    pub fn new() -> Self {
        Self {
            coefficients: [0.0_f32; 27],
            order: 2,
        }
    }

    /// Construct from a pre-built coefficient array.
    pub fn from_coefficients(coeffs: [f32; 27]) -> Self {
        Self {
            coefficients: coeffs,
            order: 2,
        }
    }

    /// Evaluate irradiance at unit direction `dir`.
    ///
    /// `E(n) = Σ_{i=0}^{8} c_i * Y_i(n)` per channel.
    ///
    /// # Errors
    /// - `LightProbeError::ZeroDirection` if `dir` is degenerate.
    pub fn evaluate(&self, dir: [f32; 3]) -> Result<[f32; 3], LightProbeError> {
        let d = lp_normalize_dir(dir)?;
        let basis = lp_sh_full_9(d);
        let mut rgb = [0.0_f32; 3];
        for (i, &y) in basis.iter().enumerate() {
            for (c, channel) in rgb.iter_mut().enumerate() {
                *channel += y * self.coefficients[i * 3 + c];
            }
        }
        Ok(rgb)
    }

    /// Return a new `IrradianceSH` with all coefficients scaled by `factor`.
    pub fn scale(&self, factor: f32) -> Self {
        let mut out = self.clone();
        for v in out.coefficients.iter_mut() {
            *v *= factor;
        }
        out
    }

    /// Return a new `IrradianceSH` that is the element-wise sum of `self` and `other`.
    pub fn add(&self, other: &IrradianceSH) -> Self {
        let mut out = self.clone();
        for i in 0..27 {
            out.coefficients[i] += other.coefficients[i];
        }
        out
    }

    /// Return the ambient (constant, L=0) term as RGB.
    ///
    /// Coefficient 0 represents the L=0 (DC) term. The actual radiated
    /// value is `c0 * Y_0^0`, but for ambient purposes we return `c0 * LP_SH_C0`
    /// per channel so it represents the average irradiance.
    pub fn ambient(&self) -> [f32; 3] {
        [
            self.coefficients[0] * LP_SH_C0,
            self.coefficients[1] * LP_SH_C0,
            self.coefficients[2] * LP_SH_C0,
        ]
    }
}

impl Default for IrradianceSH {
    fn default() -> Self {
        Self::new()
    }
}
