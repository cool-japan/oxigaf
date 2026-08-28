//! Positional light probe with irradiance SH and influence radius.

use super::error::LightProbeError;
use super::irradiance::IrradianceSH;

// ---------------------------------------------------------------------------
// LightProbe
// ---------------------------------------------------------------------------

/// Positional light probe with irradiance SH and influence radius.
#[derive(Debug, Clone)]
pub struct LightProbe {
    /// World-space position.
    pub position: [f32; 3],
    /// Pre-convolved irradiance as SH coefficients.
    pub irradiance: IrradianceSH,
    /// Influence radius (0 = global/infinite).
    pub radius: f32,
    /// Intensity multiplier.
    pub intensity: f32,
    /// Unique identifier.
    pub id: u64,
}

/// Monotonically increasing probe ID counter (never decremented).
static PROBE_ID_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl LightProbe {
    /// Construct a new probe, assigning an auto-incremented ID.
    pub fn new(position: [f32; 3], irradiance: IrradianceSH, radius: f32) -> Self {
        let id = PROBE_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self {
            position,
            irradiance,
            radius,
            intensity: 1.0,
            id,
        }
    }

    /// Compute the smooth influence weight for a world-space `point`.
    ///
    /// - `radius == 0` → `1.0` everywhere (global probe).
    /// - Otherwise: `weight = 1 − clamp(dist / radius, 0, 1)²`
    pub fn weight_for(&self, point: [f32; 3]) -> f32 {
        if self.radius <= 0.0 {
            return 1.0;
        }
        let dx = point[0] - self.position[0];
        let dy = point[1] - self.position[1];
        let dz = point[2] - self.position[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        let t = (dist / self.radius).clamp(0.0, 1.0);
        1.0 - t * t
    }

    /// Evaluate irradiance at `point` from this probe.
    ///
    /// Evaluates SH at the surface `normal`, then multiplies by probe weight and intensity.
    ///
    /// # Errors
    /// - `LightProbeError::ZeroDirection` if `normal` is near-zero.
    pub fn evaluate(&self, point: [f32; 3], normal: [f32; 3]) -> Result<[f32; 3], LightProbeError> {
        let weight = self.weight_for(point);
        let irr = self.irradiance.evaluate(normal)?;
        Ok([
            irr[0] * weight * self.intensity,
            irr[1] * weight * self.intensity,
            irr[2] * weight * self.intensity,
        ])
    }
}
