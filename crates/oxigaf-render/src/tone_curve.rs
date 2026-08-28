//! Parametric tone curves for post-processing rendered 3DGS images.
//!
//! Supports per-channel curves, S-curves, and Bezier-based tone mapping.
//! All curve evaluations use monotone cubic Hermite interpolation (Fritsch-Carlson),
//! which guarantees monotone output without overshooting.
//!
//! # Examples
//! ```
//! use oxigaf_render::tone_curve::{ToneCurve, BezierCurve, ChannelCurves};
//!
//! // Identity curve: output equals input
//! let identity = ToneCurve::identity();
//! assert!((identity.evaluate(0.5) - 0.5).abs() < 1e-4);
//!
//! // S-curve for enhanced contrast
//! let s = ToneCurve::s_curve(1.0);
//! assert!(s.evaluate(0.75) > 0.75); // highlights boosted
//!
//! // Bezier identity
//! let bez = BezierCurve::identity();
//! assert!((bez.evaluate(0.5) - 0.5).abs() < 0.05);
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by tone curve operations.
#[derive(Debug, Error)]
pub enum ToneCurveError {
    /// Curve definition is invalid.
    #[error("Invalid curve: {0}")]
    InvalidCurve(String),

    /// No control points were provided.
    #[error("Empty control points")]
    EmptyControlPoints,

    /// Not enough control points for the operation.
    #[error("Insufficient control points: need {need}, got {got}")]
    InsufficientControlPoints {
        /// Minimum required.
        need: usize,
        /// Actually provided.
        got: usize,
    },

    /// Control points are not strictly increasing in x.
    #[error("Control points not monotone")]
    NotMonotone,

    /// Image dimensions/length are inconsistent.
    #[error("Invalid image: {0}")]
    InvalidImage(String),

    /// Image slice is empty.
    #[error("Empty image")]
    EmptyImage,

    /// A value is outside the expected [0, 1] range.
    #[error("Value out of range: {val}, expected [0, 1]")]
    ValueOutOfRange {
        /// The out-of-range value.
        val: f32,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Monotone cubic Hermite helpers (Fritsch-Carlson)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute tangents for monotone cubic interpolation (Fritsch-Carlson method).
///
/// Given sorted (x, y) control points, returns a tangent (slope) vector of the
/// same length. The Fritsch-Carlson conditions are applied to guarantee that the
/// resulting cubic is monotone on every interval.
pub fn compute_tangents(points: &[(f32, f32)]) -> Vec<f32> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0.0];
    }

    // Step 1 – secant slopes between consecutive points.
    let mut delta: Vec<f32> = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let dx = points[i + 1].0 - points[i].0;
        let dy = points[i + 1].1 - points[i].1;
        delta.push(if dx.abs() < 1e-12 { 0.0 } else { dy / dx });
    }

    // Step 2 – initial tangent estimates (three-point average at interior nodes).
    let mut m: Vec<f32> = vec![0.0; n];
    // Endpoints: one-sided secant.
    m[0] = delta[0];
    m[n - 1] = delta[n - 2];
    // Interior: weighted average of neighboring secant slopes.
    // delta[i-1] is the slope to the left (interval [i-1, i]),
    // delta[i]   is the slope to the right (interval [i, i+1]).
    // Weight by the opposite interval width so that uniformly-spaced
    // points yield the simple arithmetic mean.
    for i in 1..n - 1 {
        let dx0 = points[i].0 - points[i - 1].0; // width of left interval
        let dx1 = points[i + 1].0 - points[i].0; // width of right interval
        let total = dx0 + dx1;
        if total.abs() < 1e-12 {
            m[i] = 0.0;
        } else {
            // Cardinal spline weighted mean:
            //   m = (dx1 * delta[i-1] + dx0 * delta[i]) / (dx0 + dx1)
            // For equal spacing (dx0==dx1) this collapses to the arithmetic mean.
            m[i] = (dx1 * delta[i - 1] + dx0 * delta[i]) / total;
        }
    }

    // Step 3 – Fritsch-Carlson monotonicity constraints.
    for i in 0..n - 1 {
        let d = delta[i];
        if d.abs() < 1e-12 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
            continue;
        }
        let alpha = m[i] / d;
        let beta = m[i + 1] / d;
        // Keep in the monotone region: alpha^2 + beta^2 <= 9.
        let tau = alpha * alpha + beta * beta;
        if tau > 9.0 {
            let t = 3.0 / tau.sqrt();
            m[i] = t * alpha * d;
            m[i + 1] = t * beta * d;
        }
    }

    m
}

/// Evaluate a monotone cubic Hermite spline at `x`.
///
/// `points` must be sorted by increasing x. `x` is clamped to the domain of
/// `points` before evaluation.
///
/// This recomputes the Fritsch-Carlson tangents (two heap allocations, see
/// [`compute_tangents`]) on every call. When evaluating the same `points`
/// repeatedly (e.g. building a LUT), compute the tangents once and use
/// [`monotone_cubic_interp_with_tangents`] instead -- this is exactly what
/// [`ToneCurve::evaluate`] does internally.
pub fn monotone_cubic_interp(points: &[(f32, f32)], x: f32) -> f32 {
    let n = points.len();
    if n == 0 {
        return x.clamp(0.0, 1.0);
    }
    if n == 1 {
        return points[0].1.clamp(0.0, 1.0);
    }

    let x0 = points[0].0;
    let x1 = points[n - 1].0;

    // Clamp to domain (avoids computing tangents at all for out-of-domain x).
    if x <= x0 {
        return points[0].1.clamp(0.0, 1.0);
    }
    if x >= x1 {
        return points[n - 1].1.clamp(0.0, 1.0);
    }

    let tangents = compute_tangents(points);
    monotone_cubic_interp_with_tangents(points, &tangents, x)
}

/// Evaluate a monotone cubic Hermite spline at `x`, using tangents already
/// computed by [`compute_tangents`] for these exact `points`.
///
/// `points` must be sorted by increasing x, and `tangents` must be
/// `compute_tangents(points)` (or equal to it); `x` is clamped to the
/// domain of `points` before evaluation. If `tangents.len() != points.len()`
/// this falls back to [`monotone_cubic_interp`], which recomputes them.
///
/// Prefer this over [`monotone_cubic_interp`] whenever the same `points`
/// are evaluated at many `x` values, to avoid recomputing the tangents (and
/// their two heap allocations) on every call.
pub fn monotone_cubic_interp_with_tangents(points: &[(f32, f32)], tangents: &[f32], x: f32) -> f32 {
    let n = points.len();
    if n == 0 {
        return x.clamp(0.0, 1.0);
    }
    if n == 1 {
        return points[0].1.clamp(0.0, 1.0);
    }

    let x0 = points[0].0;
    let x1 = points[n - 1].0;

    // Clamp to domain.
    if x <= x0 {
        return points[0].1.clamp(0.0, 1.0);
    }
    if x >= x1 {
        return points[n - 1].1.clamp(0.0, 1.0);
    }

    if tangents.len() != n {
        return monotone_cubic_interp(points, x);
    }

    // Binary search for the interval.
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if points[mid].0 <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let (x0, y0) = points[lo];
    let (x1, y1) = points[hi];
    let m0 = tangents[lo];
    let m1 = tangents[hi];
    let h = x1 - x0;
    if h.abs() < 1e-12 {
        return y0.clamp(0.0, 1.0);
    }

    // Hermite basis functions.
    let t = (x - x0) / h;
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    let y = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1;
    y.clamp(0.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// ToneCurve
// ─────────────────────────────────────────────────────────────────────────────

/// A 1-D tone curve defined by control points `(input, output)` in \[0,1\]².
///
/// Evaluation uses monotone cubic Hermite interpolation (Fritsch-Carlson),
/// guaranteeing no overshoots and a smooth, well-behaved remapping.
#[derive(Debug, Clone)]
pub struct ToneCurve {
    /// Control points sorted by input value.  All values in [0, 1].
    pub points: Vec<(f32, f32)>,
    /// Fritsch-Carlson tangents for `points`, cached at construction time
    /// so [`Self::evaluate`] never has to recompute them (and their two
    /// heap allocations) on every call -- see [`compute_tangents`]. Kept in
    /// sync with `points` by every constructor; [`Self::evaluate`] falls
    /// back to recomputing on the fly if the lengths ever disagree (which
    /// can only happen if `points`, a public field, is mutated directly
    /// after construction).
    tangents: Vec<f32>,
}

impl ToneCurve {
    /// Create a new `ToneCurve` from a list of `(input, output)` control points.
    ///
    /// # Errors
    /// - [`ToneCurveError::EmptyControlPoints`] if `points` is empty.
    /// - [`ToneCurveError::InsufficientControlPoints`] if fewer than 2 points.
    /// - [`ToneCurveError::ValueOutOfRange`] if any coordinate is outside [0, 1].
    /// - [`ToneCurveError::NotMonotone`] if input values are not strictly increasing.
    pub fn new(mut points: Vec<(f32, f32)>) -> Result<Self, ToneCurveError> {
        if points.is_empty() {
            return Err(ToneCurveError::EmptyControlPoints);
        }
        if points.len() < 2 {
            return Err(ToneCurveError::InsufficientControlPoints {
                need: 2,
                got: points.len(),
            });
        }

        // Sort by input so the caller needn't pre-sort.
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Validate range.
        for &(x, y) in &points {
            if !(0.0..=1.0).contains(&x) {
                return Err(ToneCurveError::ValueOutOfRange { val: x });
            }
            if !(0.0..=1.0).contains(&y) {
                return Err(ToneCurveError::ValueOutOfRange { val: y });
            }
        }

        // Validate strict monotonicity of inputs.
        for w in points.windows(2) {
            if w[1].0 <= w[0].0 {
                return Err(ToneCurveError::NotMonotone);
            }
        }

        Ok(Self::from_validated_points(points))
    }

    /// Build a `ToneCurve` from points already known to satisfy `new`'s
    /// invariants (non-empty, at least 2 points, all coordinates in
    /// `[0, 1]`, strictly increasing in x), computing and caching the
    /// Fritsch-Carlson tangents once.
    ///
    /// Only called with points that either passed `new`'s runtime checks
    /// already, or are fixed literals whose validity is obvious by
    /// construction (`identity`, `s_curve`, `lift_gamma_gain`). Debug
    /// builds assert the invariants so a future change to one of those
    /// constructors that breaks them is caught by tests instead of
    /// silently producing wrong interpolation.
    fn from_validated_points(points: Vec<(f32, f32)>) -> Self {
        debug_assert!(!points.is_empty(), "ToneCurve must have at least one point");
        debug_assert!(
            points.windows(2).all(|w| w[1].0 > w[0].0),
            "ToneCurve control points must be strictly increasing in x"
        );
        debug_assert!(
            points
                .iter()
                .all(|&(x, y)| (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y)),
            "ToneCurve control points must lie in [0, 1]^2"
        );
        let tangents = compute_tangents(&points);
        Self { points, tangents }
    }

    /// Identity curve: passes each value unchanged.
    pub fn identity() -> Self {
        Self::from_validated_points(vec![(0.0, 0.0), (1.0, 1.0)])
    }

    /// S-shaped tone curve for contrast enhancement.
    ///
    /// `contrast` is in [0, 1].  At 0, the curve is near-identity.
    /// At 1, a strong S is applied: shadows pulled down, highlights pushed up.
    ///
    /// Five control points: `[(0,0), (0.25, 0.25-c), (0.5, 0.5), (0.75, 0.75+c), (1,1)]`
    /// where `c = contrast * 0.15`.
    pub fn s_curve(contrast: f32) -> Self {
        let contrast = contrast.clamp(0.0, 1.0);
        let c = contrast * 0.15;
        let p1 = (0.25_f32, (0.25 - c).clamp(0.0, 1.0));
        let p2 = (0.75_f32, (0.75 + c).clamp(0.0, 1.0));
        Self::from_validated_points(vec![(0.0, 0.0), p1, (0.5, 0.5), p2, (1.0, 1.0)])
    }

    /// Lift-gamma-gain tone curve.
    ///
    /// - `lift`:  raises the black point (shifts shadows up). Typical range [0, 0.2].
    /// - `gamma`: midtone power.  > 1 darkens, < 1 lightens. Typical range [0.5, 2.0].
    /// - `gain`:  multiplies the white point (scales highlights). Typical range [0.8, 1.2].
    ///
    /// Returns a 5-point curve approximating the combination of these operations.
    pub fn lift_gamma_gain(lift: f32, gamma: f32, gain: f32) -> Self {
        let gamma = gamma.max(1e-4);
        let gain = gain.max(0.0);
        let lift = lift.clamp(0.0, 1.0);

        // Sample at 5 evenly-spaced input values and apply the mapping.
        let inputs = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
        let points: Vec<(f32, f32)> = inputs
            .iter()
            .map(|&x| {
                // Apply gain, then lift, then gamma.
                let scaled = (x * gain).clamp(0.0, 1.0);
                let lifted = (scaled + lift * (1.0 - scaled)).clamp(0.0, 1.0);
                let gamma_applied = lifted.powf(1.0 / gamma).clamp(0.0, 1.0);
                (x, gamma_applied)
            })
            .collect();

        Self::from_validated_points(points)
    }

    /// Evaluate the tone curve at `x`.
    ///
    /// `x` is clamped to [0, 1] before evaluation.  The output is also clamped
    /// to [0, 1].
    pub fn evaluate(&self, x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);
        if self.tangents.len() == self.points.len() {
            monotone_cubic_interp_with_tangents(&self.points, &self.tangents, x)
        } else {
            // Defensive fallback: `tangents` is kept in sync with `points`
            // by every constructor, but `points` is a public field, so
            // guard against a caller mutating it directly and leaving the
            // cache stale rather than risk an out-of-bounds index or a
            // wrong result below.
            monotone_cubic_interp(&self.points, x)
        }
    }

    /// Pre-compute a 256-entry lookup table for fast image processing.
    ///
    /// Entry `i` corresponds to input `i / 255.0`.
    pub fn lut_256(&self) -> [f32; 256] {
        let mut lut = [0.0_f32; 256];
        for (i, slot) in lut.iter_mut().enumerate() {
            let x = i as f32 / 255.0;
            *slot = self.evaluate(x);
        }
        lut
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ChannelCurves
// ─────────────────────────────────────────────────────────────────────────────

/// Per-channel tone curves (R, G, B) plus a master curve applied first.
#[derive(Debug, Clone)]
pub struct ChannelCurves {
    /// Applied to all channels before the per-channel curves.
    pub master: ToneCurve,
    /// Red channel curve (applied after master).
    pub red: ToneCurve,
    /// Green channel curve (applied after master).
    pub green: ToneCurve,
    /// Blue channel curve (applied after master).
    pub blue: ToneCurve,
}

impl ChannelCurves {
    /// All curves are identity.
    pub fn identity() -> Self {
        Self {
            master: ToneCurve::identity(),
            red: ToneCurve::identity(),
            green: ToneCurve::identity(),
            blue: ToneCurve::identity(),
        }
    }

    /// Replace the master curve (builder pattern).
    pub fn with_master(mut self, curve: ToneCurve) -> Self {
        self.master = curve;
        self
    }

    /// Replace the red channel curve (builder pattern).
    pub fn with_red(mut self, curve: ToneCurve) -> Self {
        self.red = curve;
        self
    }

    /// Replace the green channel curve (builder pattern).
    pub fn with_green(mut self, curve: ToneCurve) -> Self {
        self.green = curve;
        self
    }

    /// Replace the blue channel curve (builder pattern).
    pub fn with_blue(mut self, curve: ToneCurve) -> Self {
        self.blue = curve;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BezierCurve
// ─────────────────────────────────────────────────────────────────────────────

/// A cubic Bezier tone curve defined by four control points.
///
/// The curve is parameterised by `t ∈ [0,1]`, yielding `(Bx(t), By(t))`.
/// To evaluate at a given input `x`, the curve numerically inverts `Bx(t) = x`
/// via binary search, then returns `By(t)`.
#[derive(Debug, Clone)]
pub struct BezierCurve {
    /// Start point (typically `(0, 0)`).
    pub p0: (f32, f32),
    /// First handle.
    pub p1: (f32, f32),
    /// Second handle.
    pub p2: (f32, f32),
    /// End point (typically `(1, 1)`).
    pub p3: (f32, f32),
}

impl BezierCurve {
    /// Create a new `BezierCurve` from four control points.
    pub fn new(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), p3: (f32, f32)) -> Self {
        Self { p0, p1, p2, p3 }
    }

    /// Linear identity curve.
    ///
    /// `p0=(0,0), p1=(0.33,0.33), p2=(0.67,0.67), p3=(1,1)`
    pub fn identity() -> Self {
        Self {
            p0: (0.0, 0.0),
            p1: (0.33, 0.33),
            p2: (0.67, 0.67),
            p3: (1.0, 1.0),
        }
    }

    /// Standard S-curve.
    ///
    /// `p0=(0,0), p1=(0.25,0.15), p2=(0.75,0.85), p3=(1,1)`
    pub fn standard_s() -> Self {
        Self {
            p0: (0.0, 0.0),
            p1: (0.25, 0.15),
            p2: (0.75, 0.85),
            p3: (1.0, 1.0),
        }
    }

    /// Evaluate the Bezier x-coordinate at parameter `t`.
    #[inline]
    fn bx(&self, t: f32) -> f32 {
        let mt = 1.0 - t;
        mt * mt * mt * self.p0.0
            + 3.0 * mt * mt * t * self.p1.0
            + 3.0 * mt * t * t * self.p2.0
            + t * t * t * self.p3.0
    }

    /// Evaluate the Bezier y-coordinate at parameter `t`.
    #[inline]
    fn by(&self, t: f32) -> f32 {
        let mt = 1.0 - t;
        mt * mt * mt * self.p0.1
            + 3.0 * mt * mt * t * self.p1.1
            + 3.0 * mt * t * t * self.p2.1
            + t * t * t * self.p3.1
    }

    /// Evaluate the tone curve at input `x`.
    ///
    /// Uses a 50-step binary search on `t ∈ [0, 1]` to invert `Bx(t) = x`,
    /// then returns `By(t)`. Output is clamped to [0, 1].
    pub fn evaluate(&self, x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);
        let mut lo = 0.0_f32;
        let mut hi = 1.0_f32;
        for _ in 0..50 {
            let mid = (lo + hi) * 0.5;
            let bx_mid = self.bx(mid);
            if bx_mid < x {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = (lo + hi) * 0.5;
        self.by(t).clamp(0.0, 1.0)
    }

    /// Convert the Bezier curve to a `ToneCurve` by sampling `n_samples` points.
    ///
    /// # Errors
    /// Returns [`ToneCurveError::InsufficientControlPoints`] if `n_samples < 2`.
    pub fn to_tone_curve(&self, n_samples: usize) -> Result<ToneCurve, ToneCurveError> {
        if n_samples < 2 {
            return Err(ToneCurveError::InsufficientControlPoints {
                need: 2,
                got: n_samples,
            });
        }

        // Sample the Bezier at uniform parameter values to collect (x, y) pairs.
        let mut raw: Vec<(f32, f32)> = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let t = i as f32 / (n_samples - 1) as f32;
            raw.push((self.bx(t), self.by(t)));
        }

        // Sort and deduplicate by x to guarantee strict monotone inputs.
        raw.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut deduped: Vec<(f32, f32)> = Vec::with_capacity(raw.len());
        for pt in raw {
            if let Some(last) = deduped.last() {
                if (pt.0 - last.0).abs() < 1e-6 {
                    continue;
                }
            }
            deduped.push(pt);
        }

        if deduped.len() < 2 {
            return Err(ToneCurveError::InsufficientControlPoints {
                need: 2,
                got: deduped.len(),
            });
        }

        ToneCurve::new(deduped)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Image-level application functions
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that a pixel buffer has the expected HxWx3 byte layout.
fn validate_image(pixels: &[u8], width: usize, height: usize) -> Result<(), ToneCurveError> {
    if pixels.is_empty() {
        return Err(ToneCurveError::EmptyImage);
    }
    let expected = width
        .checked_mul(height)
        .and_then(|wh| wh.checked_mul(3))
        .ok_or_else(|| ToneCurveError::InvalidImage("Dimensions overflow".to_string()))?;
    if pixels.len() != expected {
        return Err(ToneCurveError::InvalidImage(format!(
            "Expected {}×{}×3 = {} bytes, got {}",
            width,
            height,
            expected,
            pixels.len()
        )));
    }
    Ok(())
}

/// Apply a tone curve to an RGB image using a precomputed 256-entry LUT.
///
/// The same curve is applied to all three channels.
///
/// # Errors
/// Returns [`ToneCurveError`] if the image dimensions are inconsistent.
pub fn apply_tone_curve(
    pixels: &[u8],
    width: usize,
    height: usize,
    curve: &ToneCurve,
) -> Result<Vec<u8>, ToneCurveError> {
    validate_image(pixels, width, height)?;
    let lut = curve.lut_256();
    apply_lut_256(pixels, &lut)
}

/// Apply per-channel tone curves to an RGB image.
///
/// Processing order: master curve → red curve / green curve / blue curve.
///
/// # Errors
/// Returns [`ToneCurveError`] if the image dimensions are inconsistent.
pub fn apply_channel_curves(
    pixels: &[u8],
    width: usize,
    height: usize,
    curves: &ChannelCurves,
) -> Result<Vec<u8>, ToneCurveError> {
    validate_image(pixels, width, height)?;

    let master_lut = curves.master.lut_256();
    let r_lut = curves.red.lut_256();
    let g_lut = curves.green.lut_256();
    let b_lut = curves.blue.lut_256();

    let mut out = vec![0u8; pixels.len()];
    for chunk in 0..width * height {
        let i = chunk * 3;
        // Apply master then per-channel.
        let r_master = master_lut[pixels[i] as usize];
        let g_master = master_lut[pixels[i + 1] as usize];
        let b_master = master_lut[pixels[i + 2] as usize];

        let r_idx = (r_master * 255.0).clamp(0.0, 255.0) as usize;
        let g_idx = (g_master * 255.0).clamp(0.0, 255.0) as usize;
        let b_idx = (b_master * 255.0).clamp(0.0, 255.0) as usize;

        out[i] = (r_lut[r_idx] * 255.0).clamp(0.0, 255.0) as u8;
        out[i + 1] = (g_lut[g_idx] * 255.0).clamp(0.0, 255.0) as u8;
        out[i + 2] = (b_lut[b_idx] * 255.0).clamp(0.0, 255.0) as u8;
    }
    Ok(out)
}

/// Apply a Bezier curve to an RGB image.
///
/// Converts the Bezier to a 64-point `ToneCurve` then applies it via LUT.
///
/// # Errors
/// Returns [`ToneCurveError`] if image dimensions are inconsistent or the curve
/// cannot be converted.
pub fn apply_bezier_curve(
    pixels: &[u8],
    width: usize,
    height: usize,
    bezier: &BezierCurve,
) -> Result<Vec<u8>, ToneCurveError> {
    validate_image(pixels, width, height)?;
    let curve = bezier.to_tone_curve(64)?;
    apply_tone_curve(pixels, width, height, &curve)
}

/// Apply a 256-entry float LUT to an RGB image.
///
/// Each channel byte is looked up in `lut`, multiplied by 255, and rounded.
///
/// # Errors
/// Returns [`ToneCurveError::EmptyImage`] if `pixels` is empty.
pub fn apply_lut_256(pixels: &[u8], lut: &[f32; 256]) -> Result<Vec<u8>, ToneCurveError> {
    if pixels.is_empty() {
        return Err(ToneCurveError::EmptyImage);
    }
    let out: Vec<u8> = pixels
        .iter()
        .map(|&p| (lut[p as usize] * 255.0).clamp(0.0, 255.0) as u8)
        .collect();
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility functions
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a tone curve from histogram equalization of an image.
///
/// Computes the cumulative distribution function of the luminance histogram and
/// returns it as a 256-point `ToneCurve`.
///
/// # Errors
/// Returns [`ToneCurveError`] if the image is empty or dimensions are wrong.
pub fn histogram_tone_curve(
    pixels: &[u8],
    width: usize,
    height: usize,
) -> Result<ToneCurve, ToneCurveError> {
    validate_image(pixels, width, height)?;

    // Build a simple luminance histogram (grey-world average of RGB).
    let n_pixels = width * height;
    let mut hist = [0u64; 256];
    for chunk in 0..n_pixels {
        let i = chunk * 3;
        let r = pixels[i] as f32;
        let g = pixels[i + 1] as f32;
        let b = pixels[i + 2] as f32;
        let lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 255.0) as usize;
        hist[lum] += 1;
    }

    // CDF → tone curve.
    let total = n_pixels as f64;
    let mut cdf = [0.0_f64; 256];
    let mut running = 0u64;
    for (i, &h) in hist.iter().enumerate() {
        running += h;
        cdf[i] = running as f64 / total;
    }

    // Build control points at a coarser resolution to keep the curve manageable.
    // Use 17 evenly-spaced samples (0, 16, 32, … 256) for the LUT.
    let mut pts: Vec<(f32, f32)> = Vec::with_capacity(17);
    for step in 0..=16_usize {
        let idx = (step * 16).min(255);
        let x = idx as f32 / 255.0;
        let y = cdf[idx] as f32;
        pts.push((x, y));
    }

    // Ensure the curve starts exactly at (0,0) and ends at (1,1).
    if let Some(first) = pts.first_mut() {
        *first = (0.0, 0.0);
    }
    if let Some(last) = pts.last_mut() {
        *last = (1.0, 1.0);
    }

    // Dedup any coincident x values introduced by the min()/clamp.
    pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6);

    ToneCurve::new(pts)
}

/// Estimate the effective contrast of a tone curve.
///
/// Computed as the approximate derivative at `x = 0.5`:
/// `(curve(0.6) - curve(0.4)) / 0.2`
///
/// An identity curve returns ≈ 1.0; an S-curve returns > 1.0.
pub fn curve_contrast(curve: &ToneCurve) -> f32 {
    let y_hi = curve.evaluate(0.6);
    let y_lo = curve.evaluate(0.4);
    (y_hi - y_lo) / 0.2
}

/// Invert a tone curve by swapping `(x, y)` → `(y, x)` on every control point.
///
/// # Errors
/// Returns [`ToneCurveError::NotMonotone`] if the original curve is not
/// strictly monotone in y (increasing or decreasing).
pub fn invert_curve(curve: &ToneCurve) -> Result<ToneCurve, ToneCurveError> {
    // `ToneCurve::new` sorts its input by x before checking monotonicity.
    // Naively swapping (x, y) -> (y, x) and handing the result straight to
    // `new` would let that sort silently re-order a source curve whose y
    // values are not monotone into something that passes `new`'s x-only
    // check without being a real inverse (e.g. [(0,0),(0.5,0.8),(1,0.4)]
    // swaps to [(0,0),(0.8,0.5),(0.4,1)], which sorts by x to
    // [(0,0),(0.4,1),(0.8,0.5)] -- strictly increasing in x, but not the
    // inverse of anything). So check strict monotonicity in y on the
    // original, already x-sorted, points first.
    let strictly_increasing = curve.points.windows(2).all(|w| w[1].1 > w[0].1);
    let strictly_decreasing = curve.points.windows(2).all(|w| w[1].1 < w[0].1);
    if !strictly_increasing && !strictly_decreasing {
        return Err(ToneCurveError::NotMonotone);
    }

    let inverted: Vec<(f32, f32)> = curve.points.iter().map(|&(x, y)| (y, x)).collect();
    ToneCurve::new(inverted)
}

/// Compose two tone curves: `outer(inner(x))`.
///
/// Samples `inner` at `n_points` uniformly spaced inputs, applies `outer`, and
/// returns the resulting `ToneCurve`.
///
/// # Errors
/// Returns [`ToneCurveError::InsufficientControlPoints`] if `n_points < 2`.
pub fn compose_curves(
    outer: &ToneCurve,
    inner: &ToneCurve,
    n_points: usize,
) -> Result<ToneCurve, ToneCurveError> {
    if n_points < 2 {
        return Err(ToneCurveError::InsufficientControlPoints {
            need: 2,
            got: n_points,
        });
    }
    let pts: Vec<(f32, f32)> = (0..n_points)
        .map(|i| {
            let x = i as f32 / (n_points - 1) as f32;
            let y = outer.evaluate(inner.evaluate(x));
            (x, y)
        })
        .collect();
    ToneCurve::new(pts)
}

/// Blend two tone curves: `lerp(a, b, t)` at each sample point.
///
/// `t = 0` returns a curve equivalent to `a`; `t = 1` returns one equivalent to `b`.
///
/// # Errors
/// Returns [`ToneCurveError::InsufficientControlPoints`] if `n_points < 2`.
pub fn blend_curves(
    a: &ToneCurve,
    b: &ToneCurve,
    t: f32,
    n_points: usize,
) -> Result<ToneCurve, ToneCurveError> {
    if n_points < 2 {
        return Err(ToneCurveError::InsufficientControlPoints {
            need: 2,
            got: n_points,
        });
    }
    let t = t.clamp(0.0, 1.0);
    let pts: Vec<(f32, f32)> = (0..n_points)
        .map(|i| {
            let x = i as f32 / (n_points - 1) as f32;
            let ya = a.evaluate(x);
            let yb = b.evaluate(x);
            let y = ya * (1.0 - t) + yb * t;
            (x, y)
        })
        .collect();
    ToneCurve::new(pts)
}

// ─────────────────────────────────────────────────────────────────────────────
// Curve statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics summarising a `ToneCurve`.
#[derive(Debug, Clone)]
pub struct CurveStats {
    /// Mean output over 100 uniform input samples.
    pub mean_output: f32,
    /// Approximate derivative at x = 0.5 (contrast indicator).
    pub contrast_score: f32,
    /// Output at x = 0.1 (shadow lift).
    pub shadow_lift: f32,
    /// `1 − output` at x = 0.9 (highlight compression).
    pub highlight_compression: f32,
    /// `true` if all sampled outputs are within 0.01 of the input.
    pub is_identity: bool,
}

/// Analyse a `ToneCurve` and return statistics.
pub fn analyze_curve(curve: &ToneCurve) -> CurveStats {
    const N: usize = 100;
    let mut sum = 0.0_f32;
    let mut is_identity = true;

    for i in 0..N {
        let x = i as f32 / (N - 1) as f32;
        let y = curve.evaluate(x);
        sum += y;
        if (y - x).abs() > 0.01 {
            is_identity = false;
        }
    }

    let mean_output = sum / N as f32;
    let contrast_score = curve_contrast(curve);
    let shadow_lift = curve.evaluate(0.1);
    let highlight_compression = 1.0 - curve.evaluate(0.9);

    CurveStats {
        mean_output,
        contrast_score,
        shadow_lift,
        highlight_compression,
        is_identity,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ToneCurve::new ──────────────────────────────────────────────────────

    #[test]
    fn tone_curve_new_valid() {
        let c = ToneCurve::new(vec![(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]);
        assert!(c.is_ok());
    }

    #[test]
    fn tone_curve_new_empty() {
        let c = ToneCurve::new(vec![]);
        assert!(matches!(c, Err(ToneCurveError::EmptyControlPoints)));
    }

    #[test]
    fn tone_curve_new_single_point() {
        let c = ToneCurve::new(vec![(0.5, 0.5)]);
        assert!(matches!(
            c,
            Err(ToneCurveError::InsufficientControlPoints { need: 2, got: 1 })
        ));
    }

    #[test]
    fn tone_curve_new_not_monotone() {
        // Duplicate x values are not strictly increasing.
        let c = ToneCurve::new(vec![(0.0, 0.0), (0.5, 0.5), (0.5, 0.8), (1.0, 1.0)]);
        assert!(matches!(c, Err(ToneCurveError::NotMonotone)));
    }

    #[test]
    fn tone_curve_new_out_of_range() {
        let c = ToneCurve::new(vec![(0.0, 0.0), (0.5, 1.5), (1.0, 1.0)]);
        assert!(matches!(c, Err(ToneCurveError::ValueOutOfRange { .. })));
    }

    #[test]
    fn tone_curve_new_sorts_points() {
        // Provide unsorted — should succeed after internal sort.
        let c = ToneCurve::new(vec![(1.0, 1.0), (0.0, 0.0), (0.5, 0.5)]);
        let Ok(curve) = c else {
            panic!("expected Ok from ToneCurve::new")
        };
        assert_eq!(curve.points[0].0, 0.0);
        assert_eq!(curve.points[2].0, 1.0);
    }

    #[test]
    fn tone_curve_tangents_len_matches_points() {
        // Invariant every constructor must uphold for the cache in
        // `evaluate` to actually be used (see `ToneCurve::tangents`).
        let curves: Vec<ToneCurve> = vec![
            ToneCurve::identity(),
            ToneCurve::s_curve(0.3),
            ToneCurve::lift_gamma_gain(0.0, 1.0, 1.0),
            ToneCurve::new(vec![(0.0, 0.0), (1.0, 1.0)]).expect("valid ToneCurve"),
        ];
        for curve in &curves {
            assert_eq!(curve.tangents.len(), curve.points.len());
        }
    }

    #[test]
    fn tone_curve_evaluate_matches_uncached_interp() {
        // Regression test for the tangent-caching optimization: `evaluate`
        // (which uses the cached tangents) must produce the same result as
        // the free function that recomputes tangents fresh on every call.
        let curves: Vec<ToneCurve> = vec![
            ToneCurve::identity(),
            ToneCurve::s_curve(0.8),
            ToneCurve::lift_gamma_gain(0.1, 1.4, 1.05),
            ToneCurve::new(vec![(0.0, 0.0), (0.3, 0.1), (0.6, 0.9), (1.0, 1.0)])
                .expect("valid ToneCurve"),
        ];

        for curve in &curves {
            for i in 0..=20 {
                let x = i as f32 / 20.0;
                let cached = curve.evaluate(x);
                let fresh = monotone_cubic_interp(&curve.points, x);
                assert!(
                    (cached - fresh).abs() < 1e-6,
                    "cached evaluate({x}) = {cached} but fresh interp = {fresh}"
                );
            }
        }
    }

    // ── ToneCurve::identity ─────────────────────────────────────────────────

    #[test]
    fn tone_curve_identity_midpoint() {
        let c = ToneCurve::identity();
        let y = c.evaluate(0.5);
        assert!((y - 0.5).abs() < 1e-4, "identity(0.5) = {y}");
    }

    #[test]
    fn tone_curve_identity_endpoints() {
        let c = ToneCurve::identity();
        assert!((c.evaluate(0.0) - 0.0).abs() < 1e-6);
        assert!((c.evaluate(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tone_curve_identity_linear() {
        let c = ToneCurve::identity();
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            let y = c.evaluate(x);
            assert!((y - x).abs() < 1e-4, "identity({x}) = {y}");
        }
    }

    // ── ToneCurve::s_curve ──────────────────────────────────────────────────

    #[test]
    fn s_curve_zero_contrast_near_identity() {
        let c = ToneCurve::s_curve(0.0);
        // 5-point cubic spline at zero contrast: control points are exactly on y=x,
        // but the Hermite polynomial may deviate slightly between sample nodes.
        // Allow up to 0.05 tolerance for sub-node inputs.
        for i in 1..=9 {
            let x = i as f32 / 10.0;
            let y = c.evaluate(x);
            assert!((y - x).abs() < 0.05, "s_curve(0)({}): {}", x, y);
        }
    }

    #[test]
    fn s_curve_high_contrast_s_shaped() {
        let c = ToneCurve::s_curve(1.0);
        // Shadows pulled down.
        assert!(c.evaluate(0.25) < 0.25 + 0.01, "shadow not pulled down");
        // Highlights pushed up.
        assert!(c.evaluate(0.75) > 0.75 - 0.01, "highlight not pushed up");
        // Midpoint preserved.
        assert!((c.evaluate(0.5) - 0.5).abs() < 0.01);
    }

    #[test]
    fn s_curve_has_five_points() {
        let c = ToneCurve::s_curve(0.8);
        assert_eq!(c.points.len(), 5);
    }

    // ── ToneCurve::lift_gamma_gain ──────────────────────────────────────────

    #[test]
    fn lift_gamma_gain_five_points() {
        let c = ToneCurve::lift_gamma_gain(0.05, 1.0, 1.0);
        assert_eq!(c.points.len(), 5);
    }

    #[test]
    fn lift_gamma_gain_outputs_in_range() {
        let c = ToneCurve::lift_gamma_gain(0.1, 1.2, 0.95);
        for i in 0..=20 {
            let x = i as f32 / 20.0;
            let y = c.evaluate(x);
            assert!((0.0..=1.0).contains(&y), "out of range at x={x}: y={y}");
        }
    }

    #[test]
    fn lift_gamma_gain_identity_passthrough() {
        // lift=0, gamma=1, gain=1 should be near-identity.
        // The 5-point cubic spline can deviate up to ~0.05 at sub-sample points.
        let c = ToneCurve::lift_gamma_gain(0.0, 1.0, 1.0);
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            let y = c.evaluate(x);
            assert!((y - x).abs() < 0.05, "lgg-identity({x}) = {y}");
        }
    }

    // ── ToneCurve::evaluate / clamping ─────────────────────────────────────

    #[test]
    fn evaluate_clamps_input_below_zero() {
        let c = ToneCurve::identity();
        assert!((c.evaluate(-0.5) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn evaluate_clamps_input_above_one() {
        let c = ToneCurve::identity();
        assert!((c.evaluate(2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn evaluate_output_always_in_range() {
        let c = ToneCurve::s_curve(1.0);
        for i in -5..=15 {
            let x = i as f32 / 10.0;
            let y = c.evaluate(x);
            assert!((0.0..=1.0).contains(&y), "out of range at x={x}: y={y}");
        }
    }

    // ── ToneCurve::lut_256 ──────────────────────────────────────────────────

    #[test]
    fn lut_256_has_256_entries() {
        let lut = ToneCurve::identity().lut_256();
        assert_eq!(lut.len(), 256);
    }

    #[test]
    fn lut_256_first_entry_zero() {
        let lut = ToneCurve::identity().lut_256();
        assert!((lut[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn lut_256_last_entry_one() {
        let lut = ToneCurve::identity().lut_256();
        assert!((lut[255] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn lut_256_all_entries_in_range() {
        let lut = ToneCurve::s_curve(1.0).lut_256();
        for (i, &v) in lut.iter().enumerate() {
            assert!((0.0..=1.0).contains(&v), "lut[{i}] = {v} out of range");
        }
    }

    // ── ChannelCurves ───────────────────────────────────────────────────────

    #[test]
    fn channel_curves_identity_passthrough() {
        let cc = ChannelCurves::identity();
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            assert!((cc.master.evaluate(x) - x).abs() < 1e-4);
            assert!((cc.red.evaluate(x) - x).abs() < 1e-4);
            assert!((cc.green.evaluate(x) - x).abs() < 1e-4);
            assert!((cc.blue.evaluate(x) - x).abs() < 1e-4);
        }
    }

    #[test]
    fn channel_curves_builder_with_master() {
        let s = ToneCurve::s_curve(1.0);
        let cc = ChannelCurves::identity().with_master(s.clone());
        assert_eq!(cc.master.points.len(), s.points.len());
    }

    #[test]
    fn channel_curves_builder_with_channels() {
        let s = ToneCurve::s_curve(0.5);
        let cc = ChannelCurves::identity()
            .with_red(s.clone())
            .with_green(s.clone())
            .with_blue(s.clone());
        assert_eq!(cc.red.points.len(), 5);
        assert_eq!(cc.green.points.len(), 5);
        assert_eq!(cc.blue.points.len(), 5);
    }

    // ── BezierCurve ─────────────────────────────────────────────────────────

    #[test]
    fn bezier_identity_midpoint() {
        let b = BezierCurve::identity();
        let y = b.evaluate(0.5);
        assert!((y - 0.5).abs() < 0.05, "identity bezier(0.5) = {y}");
    }

    #[test]
    fn bezier_identity_endpoints() {
        let b = BezierCurve::identity();
        assert!((b.evaluate(0.0) - 0.0).abs() < 0.05);
        assert!((b.evaluate(1.0) - 1.0).abs() < 0.05);
    }

    #[test]
    fn bezier_standard_s_midpoint() {
        let b = BezierCurve::standard_s();
        let y = b.evaluate(0.5);
        assert!((y - 0.5).abs() < 0.05, "standard_s(0.5) = {y}");
    }

    #[test]
    fn bezier_standard_s_shadow_pulled_down() {
        let b = BezierCurve::standard_s();
        // At x=0.25 the output should be < 0.25 (shadows pulled down in S-curve).
        let y = b.evaluate(0.25);
        assert!(y < 0.25, "standard_s(0.25) = {y}, expected < 0.25");
    }

    #[test]
    fn bezier_to_tone_curve_valid() {
        let b = BezierCurve::standard_s();
        let tc = b.to_tone_curve(16);
        assert!(tc.is_ok(), "to_tone_curve failed: {:?}", tc.err());
    }

    #[test]
    fn bezier_to_tone_curve_insufficient_samples() {
        let b = BezierCurve::identity();
        let tc = b.to_tone_curve(1);
        assert!(matches!(
            tc,
            Err(ToneCurveError::InsufficientControlPoints { need: 2, .. })
        ));
    }

    // ── apply_tone_curve ────────────────────────────────────────────────────

    #[test]
    fn apply_tone_curve_identity_unchanged() {
        let pixels: Vec<u8> = (0u8..=255).flat_map(|v| [v, v, v]).collect();
        let w = 256;
        let h = 1;
        let identity = ToneCurve::identity();
        let Ok(out) = apply_tone_curve(&pixels, w, h, &identity) else {
            panic!("apply_tone_curve failed")
        };
        for (orig, got) in pixels.iter().zip(out.iter()) {
            let diff = (*orig as i16 - *got as i16).abs();
            assert!(diff <= 1, "identity curve changed pixel: {orig} → {got}");
        }
    }

    #[test]
    fn apply_tone_curve_s_curve_differs() {
        let pixels: Vec<u8> = (0u8..=255).flat_map(|v| [v, v, v]).collect();
        let s = ToneCurve::s_curve(1.0);
        let Ok(out) = apply_tone_curve(&pixels, 256, 1, &s) else {
            panic!("apply_tone_curve s_curve failed")
        };
        let differs = pixels.iter().zip(out.iter()).any(|(&a, &b)| a != b);
        assert!(differs, "s-curve should change at least some pixels");
    }

    #[test]
    fn apply_tone_curve_invalid_size() {
        let pixels = vec![0u8; 5];
        let c = ToneCurve::identity();
        let r = apply_tone_curve(&pixels, 2, 1, &c);
        assert!(r.is_err());
    }

    #[test]
    fn apply_tone_curve_1x1_image() {
        let pixels = vec![128u8, 64, 200];
        let c = ToneCurve::identity();
        let Ok(out) = apply_tone_curve(&pixels, 1, 1, &c) else {
            panic!("apply_tone_curve 1x1 failed")
        };
        assert_eq!(out.len(), 3);
    }

    // ── apply_channel_curves ────────────────────────────────────────────────

    #[test]
    fn apply_channel_curves_identity_unchanged() {
        let pixels: Vec<u8> = (0u8..=255).flat_map(|v| [v, v, v]).collect();
        let cc = ChannelCurves::identity();
        let Ok(out) = apply_channel_curves(&pixels, 256, 1, &cc) else {
            panic!("apply_channel_curves failed")
        };
        for (o, g) in pixels.iter().zip(out.iter()) {
            // Two sequential LUT lookups (master + channel) each introduce up
            // to ±1 count of quantisation error, so the total tolerance is ±2.
            let diff = (*o as i16 - *g as i16).abs();
            assert!(
                diff <= 2,
                "identity channel curves changed pixel: {o} → {g}"
            );
        }
    }

    #[test]
    fn apply_channel_curves_s_curve_master_differs() {
        let pixels: Vec<u8> = (0u8..=255).flat_map(|v| [v, v, v]).collect();
        let cc = ChannelCurves::identity().with_master(ToneCurve::s_curve(1.0));
        let Ok(out) = apply_channel_curves(&pixels, 256, 1, &cc) else {
            panic!("apply_channel_curves s-curve failed")
        };
        let differs = pixels.iter().zip(out.iter()).any(|(&a, &b)| a != b);
        assert!(differs, "master s-curve should change pixels");
    }

    // ── apply_bezier_curve ──────────────────────────────────────────────────

    #[test]
    fn apply_bezier_curve_valid_output() {
        let pixels: Vec<u8> = vec![100u8, 150, 200, 50, 80, 120];
        let b = BezierCurve::identity();
        let Ok(out) = apply_bezier_curve(&pixels, 2, 1, &b) else {
            panic!("apply_bezier_curve failed")
        };
        assert_eq!(out.len(), 6);
    }

    #[test]
    fn apply_bezier_curve_1x1() {
        let pixels = vec![200u8, 100, 50];
        let b = BezierCurve::standard_s();
        let out = apply_bezier_curve(&pixels, 1, 1, &b);
        assert!(out.is_ok());
    }

    // ── monotone_cubic_interp ───────────────────────────────────────────────

    #[test]
    fn monotone_cubic_interp_linear_exact() {
        let pts = vec![(0.0_f32, 0.0), (0.5, 0.5), (1.0, 1.0)];
        for i in 0..=20 {
            let x = i as f32 / 20.0;
            let y = monotone_cubic_interp(&pts, x);
            assert!((y - x).abs() < 1e-5, "linear interp({x}) = {y}");
        }
    }

    #[test]
    fn monotone_cubic_interp_clamped_below() {
        let pts = vec![(0.0_f32, 0.0), (1.0, 1.0)];
        let y = monotone_cubic_interp(&pts, -1.0);
        assert!((y - 0.0).abs() < 1e-6);
    }

    #[test]
    fn monotone_cubic_interp_clamped_above() {
        let pts = vec![(0.0_f32, 0.0), (1.0, 1.0)];
        let y = monotone_cubic_interp(&pts, 2.0);
        assert!((y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn monotone_cubic_interp_single_point() {
        let pts = vec![(0.5_f32, 0.7)];
        let y = monotone_cubic_interp(&pts, 0.5);
        assert!((y - 0.7).abs() < 1e-6);
    }

    // ── histogram_tone_curve ────────────────────────────────────────────────

    #[test]
    fn histogram_tone_curve_valid() {
        let pixels: Vec<u8> = (0u8..=255).flat_map(|v| [v, v, v]).collect();
        let Ok(tc) = histogram_tone_curve(&pixels, 256, 1) else {
            panic!("histogram_tone_curve failed")
        };
        assert!(tc.points.len() >= 2);
        // Should span [0,1].
        let first_x = tc.points.first().map(|p| p.0).unwrap_or(f32::NAN);
        let last_x = tc.points.last().map(|p| p.0).unwrap_or(f32::NAN);
        assert!((first_x - 0.0).abs() < 1e-6, "first x = {first_x}");
        assert!((last_x - 1.0).abs() < 1e-6, "last x = {last_x}");
    }

    #[test]
    fn histogram_tone_curve_empty_image() {
        let r = histogram_tone_curve(&[], 0, 0);
        assert!(r.is_err());
    }

    #[test]
    fn histogram_tone_curve_1x1() {
        let pixels = vec![128u8, 128, 128];
        let tc = histogram_tone_curve(&pixels, 1, 1);
        assert!(tc.is_ok());
    }

    // ── curve_contrast ──────────────────────────────────────────────────────

    #[test]
    fn curve_contrast_identity_near_one() {
        let c = ToneCurve::identity();
        let contrast = curve_contrast(&c);
        assert!(
            (contrast - 1.0).abs() < 0.05,
            "identity contrast = {contrast}"
        );
    }

    #[test]
    fn curve_contrast_s_curve_greater_than_one() {
        let c = ToneCurve::s_curve(1.0);
        let contrast = curve_contrast(&c);
        assert!(contrast > 1.0, "s-curve contrast = {contrast}");
    }

    // ── invert_curve ────────────────────────────────────────────────────────

    #[test]
    fn invert_curve_identity_is_identity() {
        let c = ToneCurve::identity();
        let Ok(inv) = invert_curve(&c) else {
            panic!("invert_curve failed")
        };
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            let y = inv.evaluate(x);
            assert!((y - x).abs() < 0.01, "inv-identity({x}) = {y}");
        }
    }

    #[test]
    fn invert_curve_s_curve() {
        // The s_curve passes through (0,0), (0.5,0.5), (1,1) so it's invertible.
        let c = ToneCurve::s_curve(0.5);
        let inv = invert_curve(&c);
        assert!(inv.is_ok(), "{:?}", inv.err());
    }

    #[test]
    fn invert_curve_rejects_non_monotone_y() {
        // Regression test: a curve whose y goes up then down must be
        // rejected, not silently "inverted" into an unrelated curve via
        // `ToneCurve::new`'s sort-by-x-then-check (see the comment on
        // `invert_curve`). `ToneCurve::new` itself only requires x to be
        // monotone, so this is a valid curve to construct.
        let c = ToneCurve::new(vec![(0.0, 0.0), (0.5, 0.8), (1.0, 0.4)])
            .expect("valid ToneCurve: only x needs to be monotone");
        let result = invert_curve(&c);
        assert!(
            matches!(result, Err(ToneCurveError::NotMonotone)),
            "expected NotMonotone, got {result:?}"
        );
    }

    #[test]
    fn invert_curve_accepts_strictly_decreasing_y() {
        // A strictly decreasing curve is a valid function to invert, even
        // though `ToneCurve` itself only requires x-monotonicity.
        let c = ToneCurve::new(vec![(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]).expect("valid ToneCurve");
        let inv = invert_curve(&c);
        assert!(inv.is_ok(), "{:?}", inv.err());
    }

    // ── compose_curves ───────────────────────────────────────────────────────

    #[test]
    fn compose_curves_two_identities() {
        let id = ToneCurve::identity();
        let Ok(comp) = compose_curves(&id, &id, 32) else {
            panic!("compose_curves failed")
        };
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            let y = comp.evaluate(x);
            assert!((y - x).abs() < 0.01, "compose(id,id)({x}) = {y}");
        }
    }

    #[test]
    fn compose_curves_valid_output_range() {
        let s = ToneCurve::s_curve(0.5);
        let id = ToneCurve::identity();
        let Ok(comp) = compose_curves(&s, &id, 64) else {
            panic!("compose_curves s/id failed")
        };
        for i in 0..=20 {
            let x = i as f32 / 20.0;
            let y = comp.evaluate(x);
            assert!((0.0..=1.0).contains(&y), "compose out of range at {x}: {y}");
        }
    }

    #[test]
    fn compose_curves_insufficient_points() {
        let id = ToneCurve::identity();
        let r = compose_curves(&id, &id, 1);
        assert!(matches!(
            r,
            Err(ToneCurveError::InsufficientControlPoints { need: 2, .. })
        ));
    }

    // ── blend_curves ─────────────────────────────────────────────────────────

    #[test]
    fn blend_curves_t0_equals_a() {
        let id = ToneCurve::identity();
        let s = ToneCurve::s_curve(1.0);
        let Ok(blended) = blend_curves(&id, &s, 0.0, 32) else {
            panic!("blend_curves t=0 failed")
        };
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            let y = blended.evaluate(x);
            let ya = id.evaluate(x);
            assert!(
                (y - ya).abs() < 0.01,
                "blend(t=0)({x}) = {y}, expected {ya}"
            );
        }
    }

    #[test]
    fn blend_curves_t1_equals_b() {
        let id = ToneCurve::identity();
        let s = ToneCurve::s_curve(1.0);
        let Ok(blended) = blend_curves(&id, &s, 1.0, 32) else {
            panic!("blend_curves t=1 failed")
        };
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            let y = blended.evaluate(x);
            let yb = s.evaluate(x);
            assert!(
                (y - yb).abs() < 0.01,
                "blend(t=1)({x}) = {y}, expected {yb}"
            );
        }
    }

    #[test]
    fn blend_curves_midpoint() {
        let id = ToneCurve::identity();
        let s = ToneCurve::s_curve(1.0);
        let Ok(blended) = blend_curves(&id, &s, 0.5, 32) else {
            panic!("blend_curves t=0.5 failed")
        };
        // midpoint should be average.
        let ya = id.evaluate(0.5);
        let yb = s.evaluate(0.5);
        let expected = (ya + yb) * 0.5;
        let y = blended.evaluate(0.5);
        assert!(
            (y - expected).abs() < 0.02,
            "blend(0.5)(0.5) = {y}, expected {expected}"
        );
    }

    // ── analyze_curve ────────────────────────────────────────────────────────

    #[test]
    fn analyze_curve_identity_is_identity() {
        let c = ToneCurve::identity();
        let stats = analyze_curve(&c);
        assert!(stats.is_identity, "identity should report is_identity=true");
        assert!((stats.mean_output - 0.5).abs() < 0.05);
    }

    #[test]
    fn analyze_curve_s_curve_not_identity() {
        let c = ToneCurve::s_curve(1.0);
        let stats = analyze_curve(&c);
        assert!(!stats.is_identity, "s-curve should not be identity");
        assert!(
            stats.contrast_score > 1.0,
            "s-curve contrast_score should be > 1"
        );
    }

    #[test]
    fn analyze_curve_shadow_lift() {
        let c = ToneCurve::identity();
        let stats = analyze_curve(&c);
        assert!(
            (stats.shadow_lift - 0.1).abs() < 0.01,
            "identity shadow lift = {}",
            stats.shadow_lift
        );
    }

    #[test]
    fn analyze_curve_highlight_compression() {
        let c = ToneCurve::identity();
        let stats = analyze_curve(&c);
        assert!(
            (stats.highlight_compression - 0.1).abs() < 0.01,
            "identity highlight compression = {}",
            stats.highlight_compression
        );
    }

    // ── apply_lut_256 ───────────────────────────────────────────────────────

    #[test]
    fn apply_lut_256_identity_unchanged() {
        let pixels: Vec<u8> = (0u8..=255).flat_map(|v| [v, v, v]).collect();
        let lut = ToneCurve::identity().lut_256();
        let Ok(out) = apply_lut_256(&pixels, &lut) else {
            panic!("apply_lut_256 identity failed")
        };
        for (o, g) in pixels.iter().zip(out.iter()) {
            let diff = (*o as i16 - *g as i16).abs();
            assert!(diff <= 1, "identity lut changed: {o} → {g}");
        }
    }

    #[test]
    fn apply_lut_256_empty_error() {
        let lut = ToneCurve::identity().lut_256();
        let r = apply_lut_256(&[], &lut);
        assert!(matches!(r, Err(ToneCurveError::EmptyImage)));
    }

    #[test]
    fn apply_lut_256_output_length() {
        let pixels = vec![10u8, 20, 30, 40, 50, 60];
        let lut = ToneCurve::identity().lut_256();
        let Ok(out) = apply_lut_256(&pixels, &lut) else {
            panic!("apply_lut_256 length failed")
        };
        assert_eq!(out.len(), pixels.len());
    }

    // ── 1×1 edge cases ───────────────────────────────────────────────────────

    #[test]
    fn apply_channel_curves_1x1() {
        let pixels = vec![50u8, 100, 200];
        let cc = ChannelCurves::identity();
        let Ok(out) = apply_channel_curves(&pixels, 1, 1, &cc) else {
            panic!("apply_channel_curves 1x1 failed")
        };
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn compute_tangents_two_points() {
        let pts = vec![(0.0_f32, 0.0), (1.0, 1.0)];
        let t = compute_tangents(&pts);
        assert_eq!(t.len(), 2);
        // Both tangents should be 1.0 (slope of identity).
        assert!((t[0] - 1.0).abs() < 1e-5, "t[0] = {}", t[0]);
        assert!((t[1] - 1.0).abs() < 1e-5, "t[1] = {}", t[1]);
    }

    #[test]
    fn compute_tangents_empty() {
        let t = compute_tangents(&[]);
        assert!(t.is_empty());
    }
}
