//! Light probe configuration and aggregate statistics.

use super::blend::ProbeBlendMode;
use super::error::LightProbeError;
use super::probe::LightProbe;

// ---------------------------------------------------------------------------
// Statistics and configuration
// ---------------------------------------------------------------------------

/// Configuration for light probe operations.
#[derive(Debug, Clone)]
pub struct LightProbeConfig {
    /// Number of Monte Carlo samples for SH projection (default 10000).
    pub n_samples_projection: usize,
    /// Maximum number of probes in a scene.
    pub max_probes: usize,
    /// Default blending mode.
    pub blend_mode: ProbeBlendMode,
}

impl Default for LightProbeConfig {
    fn default() -> Self {
        Self {
            n_samples_projection: 10_000,
            max_probes: 64,
            blend_mode: ProbeBlendMode::WeightedAverage,
        }
    }
}

/// Aggregated statistics for a collection of light probes.
#[derive(Debug, Clone)]
pub struct LightProbeStats {
    /// Number of probes.
    pub n_probes: usize,
    /// Mean per-probe intensity.
    pub mean_intensity: f32,
    /// Maximum absolute SH coefficient value across all probes.
    pub max_coefficient: f32,
    /// Average ambient RGB (L=0 term).
    pub ambient_rgb: [f32; 3],
    /// Sum of squared SH coefficients (energy measure).
    pub sh_energy: f32,
}

/// Compute aggregate statistics for a slice of probes.
///
/// # Errors
/// - `LightProbeError::EmptyProbeList` if `probes` is empty.
pub fn lp_compute_stats(probes: &[LightProbe]) -> Result<LightProbeStats, LightProbeError> {
    if probes.is_empty() {
        return Err(LightProbeError::EmptyProbeList);
    }

    let n = probes.len();
    let mean_intensity = probes.iter().map(|p| p.intensity).sum::<f32>() / n as f32;

    let mut max_coefficient = 0.0_f32;
    let mut sh_energy = 0.0_f32;
    let mut ambient_sum = [0.0_f32; 3];

    for probe in probes {
        for &c in &probe.irradiance.coefficients {
            let abs = c.abs();
            if abs > max_coefficient {
                max_coefficient = abs;
            }
            sh_energy += c * c;
        }
        let amb = probe.irradiance.ambient();
        ambient_sum[0] += amb[0];
        ambient_sum[1] += amb[1];
        ambient_sum[2] += amb[2];
    }

    let inv_n = 1.0 / n as f32;
    Ok(LightProbeStats {
        n_probes: n,
        mean_intensity,
        max_coefficient,
        ambient_rgb: [
            ambient_sum[0] * inv_n,
            ambient_sum[1] * inv_n,
            ambient_sum[2] * inv_n,
        ],
        sh_energy,
    })
}

/// Format `LightProbeStats` as a human-readable string.
pub fn lp_format_stats(stats: &LightProbeStats) -> String {
    format!(
        "LightProbeStats {{ n_probes: {}, mean_intensity: {:.4}, max_coefficient: {:.6}, \
         ambient_rgb: [{:.4}, {:.4}, {:.4}], sh_energy: {:.6} }}",
        stats.n_probes,
        stats.mean_intensity,
        stats.max_coefficient,
        stats.ambient_rgb[0],
        stats.ambient_rgb[1],
        stats.ambient_rgb[2],
        stats.sh_energy,
    )
}

/// Format `LightProbeConfig` as a human-readable string.
pub fn lp_format_config(config: &LightProbeConfig) -> String {
    let mode_str = match config.blend_mode {
        ProbeBlendMode::Nearest => "Nearest",
        ProbeBlendMode::WeightedAverage => "WeightedAverage",
        ProbeBlendMode::VolumeWeighted => "VolumeWeighted",
    };
    format!(
        "LightProbeConfig {{ n_samples_projection: {}, max_probes: {}, blend_mode: {} }}",
        config.n_samples_projection, config.max_probes, mode_str,
    )
}
