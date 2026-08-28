//! Multi-probe blending strategies.

use std::f32::consts::PI;

use super::config::LightProbeConfig;
use super::error::LightProbeError;
use super::irradiance::IrradianceSH;
use super::probe::LightProbe;

// ---------------------------------------------------------------------------
// Multi-probe blending
// ---------------------------------------------------------------------------

/// Strategy for combining contributions from multiple light probes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeBlendMode {
    /// Use only the nearest probe (weight = 1.0 for closest, 0.0 for rest).
    Nearest,
    /// Blend by distance-based influence weights, then normalise.
    WeightedAverage,
    /// Blend weighted by inverse influence-sphere volume: smaller, more
    /// localised probes dominate over larger ones where their spheres
    /// overlap at the sample point (a "smallest probe wins" convention
    /// mirroring reflection-probe blending in real-time renderers),
    /// instead of the plain distance falloff used by `WeightedAverage`.
    VolumeWeighted,
}

/// Inverse of a probe's influence-sphere volume, used by
/// [`ProbeBlendMode::VolumeWeighted`] so that smaller, more localised
/// probes dominate over larger ones when their influence volumes overlap.
///
/// A `radius <= 0.0` ("global") probe is treated as an effectively huge
/// sphere so it never outweighs a bounded probe unless it is the only
/// contributor.
#[inline]
fn lp_inverse_volume(probe: &LightProbe) -> f32 {
    let r = if probe.radius > 0.0 {
        probe.radius
    } else {
        1.0e6_f32
    };
    let volume = (4.0 / 3.0) * PI * r * r * r;
    1.0 / volume.max(1e-12)
}

/// Blend a weighted set of `LightProbe` SH coefficients into a single `IrradianceSH`.
///
/// `weights` and `probes` must have the same length.
///
/// # Errors
/// - `LightProbeError::BufferMismatch` if lengths differ.
/// - `LightProbeError::EmptyProbeList` if the slice is empty.
pub fn lp_blend_irradiance_sh(
    probes: &[LightProbe],
    weights: &[f32],
) -> Result<IrradianceSH, LightProbeError> {
    if probes.is_empty() {
        return Err(LightProbeError::EmptyProbeList);
    }
    if probes.len() != weights.len() {
        return Err(LightProbeError::BufferMismatch {
            expected: probes.len(),
            got: weights.len(),
        });
    }

    let weight_sum: f32 = weights.iter().sum();
    let mut out = [0.0_f32; 27];

    if weight_sum < 1e-12 {
        // All weights zero — fall back to equal blending
        let inv = 1.0 / probes.len() as f32;
        for probe in probes {
            for (acc, &c) in out.iter_mut().zip(probe.irradiance.coefficients.iter()) {
                *acc += c * inv;
            }
        }
    } else {
        let inv = 1.0 / weight_sum;
        for (probe, &w) in probes.iter().zip(weights.iter()) {
            let nw = w * inv;
            for (acc, &c) in out.iter_mut().zip(probe.irradiance.coefficients.iter()) {
                *acc += c * nw;
            }
        }
    }

    Ok(IrradianceSH::from_coefficients(out))
}

/// Collection of light probes with a blending strategy.
#[derive(Debug, Clone)]
pub struct LightProbeBlend {
    /// Ordered list of probes.
    pub probes: Vec<LightProbe>,
    /// Blending mode.
    pub blend_mode: ProbeBlendMode,
}

impl LightProbeBlend {
    /// Construct from a non-empty list of probes.
    ///
    /// # Errors
    /// - `LightProbeError::EmptyProbeList` if `probes` is empty.
    pub fn new(probes: Vec<LightProbe>, mode: ProbeBlendMode) -> Result<Self, LightProbeError> {
        if probes.is_empty() {
            return Err(LightProbeError::EmptyProbeList);
        }
        Ok(Self {
            probes,
            blend_mode: mode,
        })
    }

    /// Construct from `config`, using `config.blend_mode` and enforcing
    /// `config.max_probes`.
    ///
    /// # Errors
    /// - `LightProbeError::EmptyProbeList` if `probes` is empty.
    /// - `LightProbeError::TooManyProbes` if `probes.len() > config.max_probes`.
    pub fn with_config(
        probes: Vec<LightProbe>,
        config: &LightProbeConfig,
    ) -> Result<Self, LightProbeError> {
        if probes.len() > config.max_probes {
            return Err(LightProbeError::TooManyProbes {
                count: probes.len(),
                max: config.max_probes,
            });
        }
        Self::new(probes, config.blend_mode)
    }

    /// Evaluate blended irradiance at a world-space `point` with surface `normal`.
    ///
    /// `Nearest`, `WeightedAverage` and `VolumeWeighted` all agree (within
    /// floating-point tolerance) when only a single probe is present, since
    /// all three reduce to `weight_for(point) * intensity * SH(normal)` in
    /// that case — they differ only in how *multiple* overlapping probes'
    /// SH data is combined.
    ///
    /// # Errors
    /// - `LightProbeError::ZeroDirection` if `normal` is degenerate.
    pub fn evaluate(&self, point: [f32; 3], normal: [f32; 3]) -> Result<[f32; 3], LightProbeError> {
        match self.blend_mode {
            ProbeBlendMode::Nearest => {
                // Find probe with highest weight for this point
                let (best_idx, _) = self.probes.iter().enumerate().fold(
                    (0usize, f32::NEG_INFINITY),
                    |(bi, bw), (i, p)| {
                        let w = p.weight_for(point);
                        if w > bw {
                            (i, w)
                        } else {
                            (bi, bw)
                        }
                    },
                );
                self.probes[best_idx].evaluate(point, normal)
            }
            ProbeBlendMode::WeightedAverage | ProbeBlendMode::VolumeWeighted => {
                // Distance-based weights drive the overall attenuation (so
                // this stays consistent with `Nearest` instead of the old
                // behaviour where normalisation cancelled the distance
                // falloff entirely). `VolumeWeighted` additionally scales
                // each probe's contribution to the *shape* of the blended
                // SH by its inverse influence volume so smaller probes
                // dominate — this cancels out of the normalisation for a
                // single probe (see doc above) but matters with several
                // overlapping probes.
                let distance_weights: Vec<f32> =
                    self.probes.iter().map(|p| p.weight_for(point)).collect();
                let blend_weights: Vec<f32> = if self.blend_mode == ProbeBlendMode::VolumeWeighted {
                    self.probes
                        .iter()
                        .zip(distance_weights.iter())
                        .map(|(p, &dw)| dw * lp_inverse_volume(p))
                        .collect()
                } else {
                    distance_weights.clone()
                };

                let blended_sh = lp_blend_irradiance_sh(&self.probes, &blend_weights)?;
                let irr = blended_sh.evaluate(normal)?;

                let blend_weight_sum: f32 = blend_weights.iter().sum();
                let intensity = if blend_weight_sum < 1e-12 {
                    1.0
                } else {
                    self.probes
                        .iter()
                        .zip(blend_weights.iter())
                        .map(|(p, &w)| p.intensity * w)
                        .sum::<f32>()
                        / blend_weight_sum
                };

                let attenuation = distance_weights.iter().sum::<f32>().clamp(0.0, 1.0);
                Ok([
                    irr[0] * intensity * attenuation,
                    irr[1] * intensity * attenuation,
                    irr[2] * intensity * attenuation,
                ])
            }
        }
    }
}
