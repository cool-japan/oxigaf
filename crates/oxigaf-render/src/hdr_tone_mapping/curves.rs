//! Primitive per-channel tone-curve operators: Reinhard, ACES filmic, Hable, the simplified Hable filmic curve, generalised Reinhard, and the Lottes (GDC 2016) tone mapper.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Parameters for Timothy Lottes' HDR tone-mapping operator (GDC 2016,
/// "Advanced Techniques and Optimization of HDR Color Pipelines").
///
/// A 4-parameter generalised Reinhard curve with independently tunable
/// overall contrast (`contrast`) and highlight-shoulder contrast
/// (`shoulder`), anchored so that `mid_in` maps to `mid_out` and `hdr_max`
/// maps to `1.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LottesParams {
    /// Overall contrast exponent (typical range ~1.0–1.7).
    pub contrast: f32,
    /// Shoulder (highlight) contrast exponent (typical range ~0.9–1.5).
    pub shoulder: f32,
    /// Maximum HDR input value the curve is normalised against; maps to `1.0`.
    pub hdr_max: f32,
    /// Input luminance considered "middle grey".
    pub mid_in: f32,
    /// Output luminance that `mid_in` should map to.
    pub mid_out: f32,
}
impl Default for LottesParams {
    fn default() -> Self {
        // Values from Lottes' reference implementation / commonly-cited
        // presets (e.g. FidelityFX / Fubaxiusz's "Lottes" ReShade port).
        Self {
            contrast: 1.6,
            shoulder: 0.977,
            hdr_max: 8.0,
            mid_in: 0.18,
            mid_out: 0.267,
        }
    }
}
/// Simple Reinhard: `out = x / (1 + x)`.
///
/// Maps `[0, ∞)` to `[0, 1)`.
#[inline]
pub fn reinhard(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    x / (1.0 + x)
}

/// Extended Reinhard: `out = x * (1 + x/white²) / (1 + x)`.
///
/// `white` is the luminance that should map to exactly 1.  Values of `white`
/// ≤ 0 are treated as 1 to avoid division by zero.
#[inline]
pub fn reinhard_extended(x: f32, white: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let w = if white > 0.0 { white } else { 1.0 };
    let numer = x * (1.0 + x / (w * w));
    let denom = 1.0 + x;
    (numer / denom).clamp(0.0, 1.0)
}

/// ACES filmic curve approximation (Narkowicz 2015).
///
/// `f(x) = (x * (a*x + b)) / (x * (c*x + d) + e)`
/// with `a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14`.
///
/// Output is clamped to `[0, 1]`.
#[inline]
pub fn aces_filmic(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let a = 2.51_f32;
    let b = 0.03_f32;
    let c = 2.43_f32;
    let d = 0.59_f32;
    let e = 0.14_f32;
    let numer = x * (a * x + b);
    let denom = x * (c * x + d) + e;
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    (numer / denom).clamp(0.0, 1.0)
}

/// Internal Hable partial function `U(x)`.
///
/// `U(x) = ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F)) - E/F`
/// with `A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30`.
#[inline]
fn hable_partial(x: f32) -> f32 {
    let a = 0.15_f32;
    let b = 0.50_f32;
    let c = 0.10_f32;
    let d = 0.20_f32;
    let e = 0.02_f32;
    let f = 0.30_f32;
    let numer = x * (a * x + c * b) + d * e;
    let denom = x * (a * x + b) + d * f;
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    numer / denom - e / f
}

/// Hable "Uncharted 2" filmic tone mapping.
///
/// `result = U(exposure * x) / U(W)` where `W = 11.2` is the white point.
///
/// Output is clamped to `[0, 1]`.
#[inline]
pub fn hable(x: f32, exposure: f32) -> f32 {
    const WHITE: f32 = 11.2;
    let w_partial = hable_partial(WHITE);
    if w_partial.abs() < 1e-12 {
        return 0.0;
    }
    let mapped = hable_partial(exposure * x) / w_partial;
    mapped.clamp(0.0, 1.0)
}

/// Simplified filmic curve (John Hable).
///
/// Step 1: `x = max(0, x - 0.004)`
/// Step 2: `y = (x*(6.2*x + 0.5)) / (x*(6.2*x + 1.7) + 0.06)`
///
/// Negative inputs map to 0. No output clamping is necessary; the formula
/// is naturally bounded in `[0, 1]` for non-negative input.
#[inline]
pub fn filmic(x: f32) -> f32 {
    let x = (x - 0.004_f32).max(0.0);
    if x < 1e-12 {
        return 0.0;
    }
    let numer = x * (6.2 * x + 0.5);
    let denom = x * (6.2 * x + 1.7) + 0.06;
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    (numer / denom).clamp(0.0, 1.0)
}

/// Generalised Reinhard-style compressive curve: `x^1.6 / (x^1.6 + 1)`.
///
/// This is a simple, monotonically increasing compressive curve — it is
/// *not* Timothy Lottes' tone mapper (despite this function's previous
/// name); for the actual Lottes GDC-2016 4-parameter curve see [`lottes`] /
/// [`LottesParams`].
#[inline]
pub fn generalized_reinhard(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let a = 1.6_f32;
    let xa = x.powf(a);
    (xa / (xa + 1.0)).clamp(0.0, 1.0)
}

/// Evaluate Timothy Lottes' tone-mapping curve for a single channel value.
///
/// `f(x) = x^a / (x^(a*d) * b + c)`, where `a` = `params.contrast`, `d` =
/// `params.shoulder`, and `b`, `c` are derived from `params` so that
/// `f(mid_in) = mid_out` and `f(hdr_max) = 1.0`.
pub fn lottes(x: f32, params: &LottesParams) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let a = params.contrast.max(1e-4);
    let d = params.shoulder.max(1e-4);
    let hdr_max = params.hdr_max.max(1e-4);
    let mid_in = params.mid_in.max(1e-6);
    let mid_out = params.mid_out.max(1e-6);

    let hdr_max_a = hdr_max.powf(a);
    let hdr_max_ad = hdr_max.powf(a * d);
    let mid_in_a = mid_in.powf(a);
    let mid_in_ad = mid_in.powf(a * d);

    let denom = (hdr_max_ad - mid_in_ad) * mid_out;
    if denom.abs() < 1e-12 {
        // Degenerate parameters (e.g. hdr_max == mid_in): fall back to a
        // simple clamp rather than dividing by ~0.
        return x.clamp(0.0, 1.0);
    }

    let b = (-mid_in_a + hdr_max_a * mid_out) / denom;
    let c = (hdr_max_ad * mid_in_a - hdr_max_a * mid_in_ad * mid_out) / denom;

    let x_a = x.powf(a);
    let x_ad = x.powf(a * d);
    let curve_denom = x_ad * b + c;
    if curve_denom.abs() < 1e-12 {
        return 0.0;
    }
    (x_a / curve_denom).clamp(0.0, 1.0)
}

/// BT.709 luminance: `0.2126*r + 0.7152*g + 0.0722*b`.
///
/// Named `tone_luminance` to avoid conflict with the re-exported
/// `color_grading::luminance` in the crate root.
#[inline]
pub fn tone_luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}
