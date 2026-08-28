//! Tone mapping operator dispatch: the legacy per-channel `ToneMappingOperator` and the newer luminance-preserving `ToneMapOperator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::curves::{
    aces_filmic, filmic, hable, lottes, reinhard, reinhard_extended, tone_luminance, LottesParams,
};

/// Simple HDR tone mapping operators for use with [`apply_operator`] and
/// [`crate::tone_map`] / [`crate::tone_map_inplace`].
#[derive(Debug, Clone)]
pub enum ToneMapOperator {
    /// Global Reinhard: `Y / (1 + Y)` per luminance, preserving hue.
    Reinhard,
    /// Extended Reinhard with a configurable maximum luminance.
    ReinhardExtended {
        /// Maximum luminance value (maps to 1.0).
        max_luminance: f32,
    },
    /// Hable / Uncharted-2 filmic curve (exposure bias = 2.0, white = 11.2).
    Filmic,
    /// ACES filmic approximation (Narkowicz 2015), per-channel.
    Aces,
    /// Timothy Lottes' tone mapper (GDC 2016), per-channel. See
    /// [`LottesParams`] / [`lottes`] for the actual 4-parameter curve.
    Lottes(LottesParams),
    /// Simple exposure adjustment: `channel * 2^stops`.
    Exposure {
        /// EV stops to apply.
        stops: f32,
    },
    /// Linear rescale from `[min, max]` to `[0, 1]`.
    Linear {
        /// Input minimum.
        min: f32,
        /// Input maximum.
        max: f32,
    },
}
/// Available HDR tone mapping operators.
#[derive(Debug, Clone)]
pub enum ToneMappingOperator {
    /// Simple Reinhard: `x / (1 + x)`.
    Reinhard,
    /// Extended Reinhard with configurable white point.
    ReinhardExtended {
        /// Luminance value that maps to 1.0.
        white: f32,
    },
    /// ACES filmic approximation (Narkowicz 2015); output clamped to \[0,1\].
    AcesFilmic,
    /// Hable "Uncharted 2" filmic curve with configurable pre-exposure.
    Hable {
        /// Pre-exposure multiplier (applied before the curve, not in EV stops).
        exposure: f32,
    },
    /// Simplified John Hable filmic S-curve.
    Filmic,
    /// Pass-through: only clamp to `[0, 1]`.
    Linear,
    /// Custom parameterised shadow/midtone/highlight operator.
    Custom {
        /// Power applied in the shadow region. Must be > 0
        /// (validated by [`crate::ToneMappingConfig::validate`]).
        shadow_gamma: f32,
        /// Linear scale applied in the midtone region. Must be > 0
        /// (validated by [`crate::ToneMappingConfig::validate`]).
        midtone_scale: f32,
        /// Roll-off strength in the highlight region. Must be > 0
        /// (validated by [`crate::ToneMappingConfig::validate`]) — larger values
        /// compress highlight input more strongly (the output grows more
        /// slowly above 1.0). See [`ToneMappingOperator::apply_channel`]
        /// for the exact curve.
        highlight_rolloff: f32,
    },
}
impl ToneMappingOperator {
    /// Apply this operator to a single HDR channel value.
    ///
    /// Returns a value nominally in `[0, 1]` — with one documented
    /// exception: the `Custom` operator's highlight region (see below) is
    /// intentionally not upper-clamped, matching [`crate::apply_gamma`] elsewhere
    /// in this module. Callers that need a hard `[0, 1]` result (e.g.
    /// [`crate::tone_map_image`]) already clamp the final pixel value themselves.
    pub fn apply_channel(&self, x: f32) -> f32 {
        match self {
            ToneMappingOperator::Reinhard => reinhard(x),
            ToneMappingOperator::ReinhardExtended { white } => reinhard_extended(x, *white),
            ToneMappingOperator::AcesFilmic => aces_filmic(x),
            ToneMappingOperator::Hable { exposure } => hable(x, *exposure),
            ToneMappingOperator::Filmic => filmic(x),
            ToneMappingOperator::Linear => x.clamp(0.0, 1.0),
            ToneMappingOperator::Custom {
                shadow_gamma,
                midtone_scale,
                highlight_rolloff,
            } => {
                let x_adj = x * midtone_scale;
                if x_adj <= 0.0 {
                    0.0
                } else if x_adj < 1.0 {
                    x_adj.powf(*shadow_gamma).clamp(0.0, 1.0)
                } else {
                    // Highlight roll-off for x_adj >= 1.
                    //
                    // The shadow branch above approaches exactly 1.0 in the
                    // limit x_adj -> 1 (x_adj.powf(g) -> 1 for *any* g), so
                    // continuity requires this branch to equal 1.0 exactly
                    // at x_adj = 1 too. Given that boundary value, a curve
                    // that is *also* monotonically non-decreasing and
                    // bounded above by 1.0 is forced to be the constant
                    // 1.0 (it can never rise once it starts at the
                    // maximum). The previous formula ignored this and
                    // produced a strictly *decreasing* curve instead —
                    // HDR highlights rendered darker than midtones and
                    // eventually clamped to black.
                    //
                    // To keep `highlight_rolloff` meaningful (rather than
                    // a validated-but-unused parameter) this branch grows
                    // *logarithmically* above 1.0 instead of hard-clamping:
                    // strictly monotonic, exactly continuous at x_adj = 1
                    // (ln(1) = 0), and grows extremely slowly, giving a
                    // genuine (if soft) roll-off shape rather than a flat
                    // clip. Larger `highlight_rolloff` compresses harder
                    // (slower growth).
                    let h = highlight_rolloff.max(1e-3);
                    (1.0 + x_adj.ln() / h).max(0.0)
                }
            }
        }
    }
    /// Human-readable name for this operator.
    pub fn name(&self) -> &str {
        match self {
            ToneMappingOperator::Reinhard => "reinhard",
            ToneMappingOperator::ReinhardExtended { .. } => "reinhard_extended",
            ToneMappingOperator::AcesFilmic => "aces_filmic",
            ToneMappingOperator::Hable { .. } => "hable",
            ToneMappingOperator::Filmic => "filmic",
            ToneMappingOperator::Linear => "linear",
            ToneMappingOperator::Custom { .. } => "custom",
        }
    }
}
/// Apply a [`ToneMapOperator`] to a single RGB triple `(r, g, b)`.
///
/// Returns the tone-mapped `(r, g, b)`.  Values may still exceed `[0, 1]` for
/// the `Exposure` variant — clip with [`crate::ToneMapConfig::clip`] if needed.
pub fn apply_operator(r: f32, g: f32, b: f32, op: &ToneMapOperator) -> (f32, f32, f32) {
    match op {
        ToneMapOperator::Reinhard => {
            let lum = tone_luminance(r, g, b);
            if lum < 1e-10 {
                return (0.0, 0.0, 0.0);
            }
            let lum_out = lum / (1.0 + lum);
            let scale = lum_out / lum;
            (r * scale, g * scale, b * scale)
        }
        ToneMapOperator::ReinhardExtended { max_luminance } => {
            let lum = tone_luminance(r, g, b);
            if lum < 1e-10 {
                return (0.0, 0.0, 0.0);
            }
            let w = max_luminance.max(1e-6);
            let lum_out = (lum * (1.0 + lum / (w * w))) / (1.0 + lum);
            let scale = (lum_out / lum).clamp(0.0, 10.0);
            (
                (r * scale).clamp(0.0, 1.0),
                (g * scale).clamp(0.0, 1.0),
                (b * scale).clamp(0.0, 1.0),
            )
        }
        ToneMapOperator::Filmic => {
            // Hable/Uncharted 2: apply per-channel with exposure bias = 2.0
            (hable(r, 2.0), hable(g, 2.0), hable(b, 2.0))
        }
        ToneMapOperator::Aces => (aces_filmic(r), aces_filmic(g), aces_filmic(b)),
        ToneMapOperator::Lottes(params) => {
            (lottes(r, params), lottes(g, params), lottes(b, params))
        }
        ToneMapOperator::Exposure { stops } => {
            let scale = (2.0_f32).powf(*stops);
            (r * scale, g * scale, b * scale)
        }
        ToneMapOperator::Linear { min, max } => {
            let range = (max - min).abs() + 1e-7;
            let map = |v: f32| (v - min) / range;
            (map(r), map(g), map(b))
        }
    }
}
