//! Color grading post-processing for rendered 3DGS images.
//!
//! Provides common color grading operations applied to flat `Vec<f32>` images
//! (H×W×3, RGB in \[0,1\]):
//!
//! - **Exposure**: multiply by 2^stops.
//! - **Contrast**: stretch/compress around midpoint 0.5.
//! - **Saturation**: scale HSV saturation channel.
//! - **Tone Curves**: piecewise-linear remapping of per-channel luminance.
//! - **3D LUT**: trilinear lookup into a 3D color table.
//! - **Pipeline**: composable chain of the above, with an optional final LUT.
//! - **Histogram**: per-channel count analysis.

use std::collections::HashMap;

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by color grading operations.
#[derive(Debug, Error)]
pub enum ColorGradingError {
    /// Image has wrong dimensions or non-multiple-of-3 length.
    #[error("Invalid image: {0}")]
    InvalidImage(String),

    /// LUT size is not cubic or values are out of range.
    #[error("Invalid LUT: {0}")]
    InvalidLut(String),

    /// Configuration parameter is out of valid range.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Color space helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert sRGB to linear (gamma ≈ 2.2): `out = in^2.2`.
///
/// Input is clamped to [0, 1] before the power operation.
#[inline]
pub fn srgb_to_linear(v: f32) -> f32 {
    v.clamp(0.0, 1.0).powf(2.2)
}

/// Convert linear to sRGB: `out = in^(1/2.2)`.
///
/// Input is clamped to [0, 1] before the power operation.
#[inline]
pub fn linear_to_srgb(v: f32) -> f32 {
    v.clamp(0.0, 1.0).powf(1.0 / 2.2)
}

/// Luminance of an RGB color: `0.2126*r + 0.7152*g + 0.0722*b`.
#[inline]
pub fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Convert RGB to HSV.
///
/// All inputs are expected in [0, 1]. Returns `(h, s, v)` where `h` is in
/// `[0, 360)`, `s` and `v` are in `[0, 1]`.
pub fn rgb_to_hsv_grading(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;

    let s = if max > 1e-7 { delta / max } else { 0.0 };

    let h = if delta < 1e-7 {
        0.0
    } else if max == r {
        let h = 60.0 * ((g - b) / delta);
        if h < 0.0 {
            h + 360.0
        } else {
            h
        }
    } else if max == g {
        60.0 * ((b - r) / delta) + 120.0
    } else {
        60.0 * ((r - g) / delta) + 240.0
    };

    (h, s, v)
}

/// Convert HSV to RGB.
///
/// `h` is in `[0, 360)`, `s` and `v` are in `[0, 1]`. Returns `(r, g, b)` clamped
/// to `[0, 1]`.
pub fn hsv_to_rgb_grading(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    if s < 1e-7 {
        let c = v.clamp(0.0, 1.0);
        return (c, c, c);
    }

    let h = ((h % 360.0) + 360.0) % 360.0; // normalise into [0, 360)
    let sector = (h / 60.0) as i32;
    let f = h / 60.0 - sector as f32;

    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    let (r, g, b) = match sector {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };

    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

// ─────────────────────────────────────────────────────────────────────────────
// Image validation helper
// ─────────────────────────────────────────────────────────────────────────────

fn validate_rgb_image(image: &[f32]) -> Result<(), ColorGradingError> {
    if !image.len().is_multiple_of(3) {
        return Err(ColorGradingError::InvalidImage(format!(
            "image length {} is not a multiple of 3",
            image.len()
        )));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ExposureAdjust
// ─────────────────────────────────────────────────────────────────────────────

/// Adjust exposure: `out = clamp(in * 2^stops, 0, 1)`.
///
/// `stops = 1.0` doubles brightness; `stops = -1.0` halves it.
#[derive(Debug, Clone)]
pub struct ExposureAdjust {
    /// Exposure value stops. Positive values brighten, negative darken.
    pub stops: f32,
}

impl ExposureAdjust {
    /// Create a new [`ExposureAdjust`] with the given EV stops.
    pub fn new(stops: f32) -> Self {
        Self { stops }
    }

    /// Apply exposure adjustment to a single RGB pixel.
    ///
    /// Each channel is multiplied by 2^stops and clamped to [0, 1].
    #[inline]
    pub fn apply_pixel(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let scale = (self.stops * std::f32::consts::LN_2).exp();
        (
            (r * scale).clamp(0.0, 1.0),
            (g * scale).clamp(0.0, 1.0),
            (b * scale).clamp(0.0, 1.0),
        )
    }

    /// Apply exposure adjustment to a flat RGB image (`width * height * 3` elements).
    ///
    /// # Errors
    ///
    /// Returns [`ColorGradingError::InvalidImage`] if the image length is not a multiple of 3.
    pub fn apply_image(&self, image: &[f32]) -> Result<Vec<f32>, ColorGradingError> {
        validate_rgb_image(image)?;
        let scale = (self.stops * std::f32::consts::LN_2).exp();
        let out = image.iter().map(|&v| (v * scale).clamp(0.0, 1.0)).collect();
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ContrastAdjust
// ─────────────────────────────────────────────────────────────────────────────

/// Adjust contrast around midpoint 0.5: `out = (in - 0.5) * contrast + 0.5`.
///
/// Values outside [0, 1] after adjustment are clamped.
#[derive(Debug, Clone)]
pub struct ContrastAdjust {
    /// Contrast scale factor. 1.0 = no change; >1 = more contrast; 0..1 = less.
    pub contrast: f32,
}

impl ContrastAdjust {
    /// Create a new [`ContrastAdjust`].
    ///
    /// # Errors
    ///
    /// Returns [`ColorGradingError::InvalidConfig`] if `contrast <= 0`.
    pub fn new(contrast: f32) -> Result<Self, ColorGradingError> {
        if contrast <= 0.0 {
            return Err(ColorGradingError::InvalidConfig(format!(
                "contrast must be positive, got {}",
                contrast
            )));
        }
        Ok(Self { contrast })
    }

    /// Apply contrast to a single channel value.
    #[inline]
    pub fn apply_pixel(&self, v: f32) -> f32 {
        ((v - 0.5) * self.contrast + 0.5).clamp(0.0, 1.0)
    }

    /// Apply contrast to a flat RGB image.
    ///
    /// # Errors
    ///
    /// Returns [`ColorGradingError::InvalidImage`] if the image length is not a multiple of 3.
    pub fn apply_image(&self, image: &[f32]) -> Result<Vec<f32>, ColorGradingError> {
        validate_rgb_image(image)?;
        let out = image.iter().map(|&v| self.apply_pixel(v)).collect();
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SaturationAdjust
// ─────────────────────────────────────────────────────────────────────────────

/// Adjust color saturation via HSV conversion.
///
/// `saturation = 0` produces grayscale; `saturation = 1` leaves colors unchanged;
/// values >1 over-saturate.
#[derive(Debug, Clone)]
pub struct SaturationAdjust {
    /// Saturation scale factor (>= 0).
    pub saturation: f32,
}

impl SaturationAdjust {
    /// Create a new [`SaturationAdjust`].
    ///
    /// # Errors
    ///
    /// Returns [`ColorGradingError::InvalidConfig`] if `saturation < 0`.
    pub fn new(saturation: f32) -> Result<Self, ColorGradingError> {
        if saturation < 0.0 {
            return Err(ColorGradingError::InvalidConfig(format!(
                "saturation must be >= 0, got {}",
                saturation
            )));
        }
        Ok(Self { saturation })
    }

    /// Apply saturation adjustment to a single RGB pixel.
    pub fn apply_pixel(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let (h, s, v) = rgb_to_hsv_grading(r, g, b);
        let new_s = (s * self.saturation).clamp(0.0, 1.0);
        hsv_to_rgb_grading(h, new_s, v)
    }

    /// Apply saturation adjustment to a flat RGB image.
    ///
    /// # Errors
    ///
    /// Returns [`ColorGradingError::InvalidImage`] if the image length is not a multiple of 3.
    pub fn apply_image(&self, image: &[f32]) -> Result<Vec<f32>, ColorGradingError> {
        validate_rgb_image(image)?;
        let mut out = Vec::with_capacity(image.len());
        for chunk in image.chunks_exact(3) {
            let (r, g, b) = self.apply_pixel(chunk[0], chunk[1], chunk[2]);
            out.push(r);
            out.push(g);
            out.push(b);
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ToneCurve
// ─────────────────────────────────────────────────────────────────────────────

/// A tone curve defined by control points `(x, y)` in [0, 1] × [0, 1].
///
/// Evaluated via piecewise linear interpolation. Applied identically to all
/// three channels.
#[derive(Debug, Clone)]
pub struct ToneCurve {
    /// Control points sorted by x. Must have at least 2 points spanning x=0 and x=1.
    pub control_points: Vec<(f32, f32)>,
}

impl ToneCurve {
    /// Linear identity curve: `(0, 0) → (1, 1)`.
    pub fn identity() -> Self {
        Self {
            control_points: vec![(0.0, 0.0), (1.0, 1.0)],
        }
    }

    /// S-curve with enhanced shadow and highlight contrast.
    ///
    /// Control points: `(0,0), (0.25,0.18), (0.5,0.5), (0.75,0.82), (1,1)`.
    pub fn s_curve() -> Self {
        Self {
            control_points: vec![
                (0.0, 0.0),
                (0.25, 0.18),
                (0.5, 0.5),
                (0.75, 0.82),
                (1.0, 1.0),
            ],
        }
    }

    /// Film-like curve: slight lift in shadows, rolloff in highlights.
    ///
    /// Control points: `(0,0.02), (0.25,0.22), (0.5,0.5), (0.75,0.78), (1,0.97)`.
    pub fn film() -> Self {
        Self {
            control_points: vec![
                (0.0, 0.02),
                (0.25, 0.22),
                (0.5, 0.5),
                (0.75, 0.78),
                (1.0, 0.97),
            ],
        }
    }

    /// Evaluate the curve at `x` using piecewise linear interpolation.
    ///
    /// `x` is clamped to [0, 1]. If `x` lies before the first control point or
    /// after the last, the nearest endpoint value is returned (constant extrapolation).
    pub fn evaluate(&self, x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);
        let pts = &self.control_points;

        if pts.is_empty() {
            return x;
        }
        if pts.len() == 1 {
            return pts[0].1;
        }

        // Clamp to first/last
        if x <= pts[0].0 {
            return pts[0].1;
        }
        if x >= pts[pts.len() - 1].0 {
            return pts[pts.len() - 1].1;
        }

        // Binary search for the segment containing x
        let idx = pts.partition_point(|&(px, _)| px <= x);
        // idx is the first point with px > x, so segment is [idx-1, idx]
        let idx = idx.min(pts.len() - 1).max(1);
        let (x0, y0) = pts[idx - 1];
        let (x1, y1) = pts[idx];

        if (x1 - x0).abs() < 1e-9 {
            return y0;
        }

        let t = (x - x0) / (x1 - x0);
        y0 + t * (y1 - y0)
    }

    /// Apply the tone curve to a single RGB pixel.
    #[inline]
    pub fn apply_pixel(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        (self.evaluate(r), self.evaluate(g), self.evaluate(b))
    }

    /// Apply the tone curve to a flat RGB image.
    ///
    /// # Errors
    ///
    /// Returns [`ColorGradingError::InvalidImage`] if the image length is not a multiple of 3.
    pub fn apply_image(&self, image: &[f32]) -> Result<Vec<f32>, ColorGradingError> {
        validate_rgb_image(image)?;
        let out = image.iter().map(|&v| self.evaluate(v)).collect();
        Ok(out)
    }

    /// Insert a control point, maintaining sorted order by x.
    pub fn add_point(&mut self, x: f32, y: f32) {
        let idx = self.control_points.partition_point(|&(px, _)| px < x);
        self.control_points.insert(idx, (x, y));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lut3D
// ─────────────────────────────────────────────────────────────────────────────

/// A 3D color lookup table mapping `(R, G, B) → (R', G', B')`.
///
/// The lattice has `size × size × size` points. Lookup uses trilinear interpolation.
/// Storage is `(ri, gi, bi)` major with index:
///
/// ```text
/// base = (ri * size * size + gi * size + bi) * 3
/// table[base]   = R'
/// table[base+1] = G'
/// table[base+2] = B'
/// ```
#[derive(Debug, Clone)]
pub struct Lut3D {
    /// Lattice size per axis (e.g. 17, 33, 65). Must be >= 2.
    pub size: usize,
    /// Flat table: `size^3 * 3` floats.
    pub table: Vec<f32>,
}

impl Lut3D {
    /// Build an identity LUT where output equals input.
    ///
    /// # Errors
    ///
    /// Returns [`ColorGradingError::InvalidLut`] if `size < 2`.
    pub fn identity(size: usize) -> Result<Self, ColorGradingError> {
        if size < 2 {
            return Err(ColorGradingError::InvalidLut(format!(
                "LUT size must be >= 2, got {}",
                size
            )));
        }
        let n = size * size * size * 3;
        let mut table = vec![0.0f32; n];
        let scale = 1.0 / (size - 1) as f32;
        for ri in 0..size {
            for gi in 0..size {
                for bi in 0..size {
                    let base = (ri * size * size + gi * size + bi) * 3;
                    table[base] = ri as f32 * scale;
                    table[base + 1] = gi as f32 * scale;
                    table[base + 2] = bi as f32 * scale;
                }
            }
        }
        Ok(Self { size, table })
    }

    /// Validate this LUT's internal invariants.
    ///
    /// `size` and `table` are both public with no invariant enforcement
    /// outside the [`Lut3D::identity`]/[`Lut3D::from_tone_curve`]
    /// constructors, so a directly-constructed or hand-deserialized (e.g.
    /// from a truncated `.cube` file) `Lut3D` can violate them.
    ///
    /// # Errors
    ///
    /// Returns [`ColorGradingError::InvalidLut`] if `size < 2` or
    /// `table.len() != size^3 * 3`.
    pub fn validate(&self) -> Result<(), ColorGradingError> {
        if self.size < 2 {
            return Err(ColorGradingError::InvalidLut(format!(
                "LUT size must be >= 2, got {}",
                self.size
            )));
        }
        let expected = self.size * self.size * self.size * 3;
        if self.table.len() != expected {
            return Err(ColorGradingError::InvalidLut(format!(
                "LUT table has {} entries, expected size^3*3 = {} for size {}",
                self.table.len(),
                expected,
                self.size
            )));
        }
        Ok(())
    }

    /// Trilinear lookup for an RGB value in [0, 1].
    ///
    /// Degenerate LUTs (`size == 0`, or a `table` shorter than `size^3 * 3`
    /// — see [`Lut3D::validate`]) do not panic: `size == 0` returns
    /// `(0.0, 0.0, 0.0)` directly, and missing table entries read as `0.0`.
    pub fn lookup(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        if self.size == 0 {
            return (0.0, 0.0, 0.0);
        }
        let s = (self.size - 1) as f32;

        let lr = (r.clamp(0.0, 1.0) * s).min(s);
        let lg = (g.clamp(0.0, 1.0) * s).min(s);
        let lb = (b.clamp(0.0, 1.0) * s).min(s);

        let r0 = lr.floor() as usize;
        let g0 = lg.floor() as usize;
        let b0 = lb.floor() as usize;
        let r1 = (r0 + 1).min(self.size - 1);
        let g1 = (g0 + 1).min(self.size - 1);
        let b1 = (b0 + 1).min(self.size - 1);

        let fr = lr - r0 as f32;
        let fg = lg - g0 as f32;
        let fb = lb - b0 as f32;

        // Fetch a table value at lattice point (ri, gi, bi), channel ch.
        // `.get(..)` rather than direct indexing: a `table` shorter than
        // `size^3 * 3` (see `validate`) must not panic here.
        let fetch = |ri: usize, gi: usize, bi: usize, ch: usize| -> f32 {
            let base = (ri * self.size * self.size + gi * self.size + bi) * 3;
            self.table.get(base + ch).copied().unwrap_or(0.0)
        };

        // Trilinear interpolation per channel.
        let interp = |ch: usize| -> f32 {
            let c000 = fetch(r0, g0, b0, ch);
            let c001 = fetch(r0, g0, b1, ch);
            let c010 = fetch(r0, g1, b0, ch);
            let c011 = fetch(r0, g1, b1, ch);
            let c100 = fetch(r1, g0, b0, ch);
            let c101 = fetch(r1, g0, b1, ch);
            let c110 = fetch(r1, g1, b0, ch);
            let c111 = fetch(r1, g1, b1, ch);

            let c00 = c000 * (1.0 - fb) + c001 * fb;
            let c01 = c010 * (1.0 - fb) + c011 * fb;
            let c10 = c100 * (1.0 - fb) + c101 * fb;
            let c11 = c110 * (1.0 - fb) + c111 * fb;

            let c0 = c00 * (1.0 - fg) + c01 * fg;
            let c1 = c10 * (1.0 - fg) + c11 * fg;

            c0 * (1.0 - fr) + c1 * fr
        };

        (interp(0), interp(1), interp(2))
    }

    /// Apply the 3D LUT to a flat RGB image.
    ///
    /// # Errors
    ///
    /// - [`ColorGradingError::InvalidImage`] if the image length is not a multiple of 3.
    /// - [`ColorGradingError::InvalidLut`] if `self` fails [`Lut3D::validate`].
    pub fn apply_image(&self, image: &[f32]) -> Result<Vec<f32>, ColorGradingError> {
        validate_rgb_image(image)?;
        self.validate()?;
        let mut out = Vec::with_capacity(image.len());
        for chunk in image.chunks_exact(3) {
            let (r, g, b) = self.lookup(chunk[0], chunk[1], chunk[2]);
            out.push(r);
            out.push(g);
            out.push(b);
        }
        Ok(out)
    }

    /// Build a 3D LUT from a [`ToneCurve`] applied identically to all channels.
    ///
    /// # Errors
    ///
    /// Returns [`ColorGradingError::InvalidLut`] if `size < 2`.
    pub fn from_tone_curve(size: usize, curve: &ToneCurve) -> Result<Self, ColorGradingError> {
        if size < 2 {
            return Err(ColorGradingError::InvalidLut(format!(
                "LUT size must be >= 2, got {}",
                size
            )));
        }
        let n = size * size * size * 3;
        let mut table = vec![0.0f32; n];
        let scale = 1.0 / (size - 1) as f32;
        for ri in 0..size {
            for gi in 0..size {
                for bi in 0..size {
                    let base = (ri * size * size + gi * size + bi) * 3;
                    table[base] = curve.evaluate(ri as f32 * scale);
                    table[base + 1] = curve.evaluate(gi as f32 * scale);
                    table[base + 2] = curve.evaluate(bi as f32 * scale);
                }
            }
        }
        Ok(Self { size, table })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ColorGradingPipeline
// ─────────────────────────────────────────────────────────────────────────────

/// A single color grading step.
#[derive(Debug, Clone)]
pub enum GradingStep {
    /// Exposure adjustment in EV stops.
    Exposure(f32),
    /// Contrast scaling factor.
    Contrast(f32),
    /// Saturation scaling factor.
    Saturation(f32),
    /// Tone curve remapping.
    ToneCurve(ToneCurve),
    /// A named user-defined step. Dispatched by name to a handler
    /// registered via [`ColorGradingPipeline::with_custom_handler`];
    /// [`ColorGradingPipeline::apply`] returns
    /// [`ColorGradingError::InvalidConfig`] if no handler is registered
    /// for the name.
    Custom(String),
}

/// A user-supplied transform for a named [`GradingStep::Custom`] step.
type CustomStepHandler = Box<dyn Fn(&[f32]) -> Result<Vec<f32>, ColorGradingError> + Send + Sync>;

/// Composable color grading pipeline.
///
/// Steps are applied in insertion order, followed by an optional 3D LUT.
pub struct ColorGradingPipeline {
    steps: Vec<GradingStep>,
    /// Optional 3D LUT applied after all other steps.
    pub lut: Option<Lut3D>,
    /// Handlers for named [`GradingStep::Custom`] steps, registered via
    /// [`Self::with_custom_handler`].
    custom_handlers: HashMap<String, CustomStepHandler>,
}

impl ColorGradingPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            lut: None,
            custom_handlers: HashMap::new(),
        }
    }

    /// Append a grading step (builder pattern).
    pub fn push(mut self, step: GradingStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Attach a 3D LUT to be applied last (builder pattern).
    pub fn with_lut(mut self, lut: Lut3D) -> Self {
        self.lut = Some(lut);
        self
    }

    /// Register a handler for a named [`GradingStep::Custom`] step
    /// (builder pattern).
    ///
    /// When [`Self::apply`] encounters `GradingStep::Custom(name)`, it
    /// looks up `name` in this registry and invokes the handler with the
    /// current image buffer; the handler's output becomes the input to the
    /// next step. A `Custom` step whose name has no registered handler
    /// makes `apply` return [`ColorGradingError::InvalidConfig`].
    pub fn with_custom_handler<F>(mut self, name: impl Into<String>, handler: F) -> Self
    where
        F: Fn(&[f32]) -> Result<Vec<f32>, ColorGradingError> + Send + Sync + 'static,
    {
        self.custom_handlers.insert(name.into(), Box::new(handler));
        self
    }

    /// Number of steps in the pipeline (not counting the LUT).
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns `true` if the pipeline has no steps and no LUT.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty() && self.lut.is_none()
    }

    /// Borrow the slice of grading steps.
    pub fn steps(&self) -> &[GradingStep] {
        &self.steps
    }

    /// Apply the full pipeline to a flat RGB image.
    ///
    /// Steps are applied sequentially; a final LUT lookup is performed if one is set.
    ///
    /// # Errors
    ///
    /// Propagates any [`ColorGradingError`] from individual steps or the LUT.
    pub fn apply(&self, image: &[f32]) -> Result<Vec<f32>, ColorGradingError> {
        validate_rgb_image(image)?;

        let mut current: Vec<f32> = image.to_vec();

        for step in &self.steps {
            current = match step {
                GradingStep::Exposure(stops) => {
                    ExposureAdjust::new(*stops).apply_image(&current)?
                }
                GradingStep::Contrast(c) => {
                    // validate inline; negative contrast is an error
                    ContrastAdjust::new(*c)?.apply_image(&current)?
                }
                GradingStep::Saturation(s) => SaturationAdjust::new(*s)?.apply_image(&current)?,
                GradingStep::ToneCurve(curve) => curve.apply_image(&current)?,
                GradingStep::Custom(name) => match self.custom_handlers.get(name) {
                    Some(handler) => handler(&current)?,
                    None => {
                        return Err(ColorGradingError::InvalidConfig(format!(
                            "no handler registered for custom grading step {name:?} \
                             (register one with ColorGradingPipeline::with_custom_handler)"
                        )))
                    }
                },
            };
        }

        if let Some(lut) = &self.lut {
            current = lut.apply_image(&current)?;
        }

        Ok(current)
    }

    /// Standard preset: gentle S-curve, +10% saturation, -0.1 EV exposure.
    pub fn standard_preset() -> Self {
        Self::new()
            .push(GradingStep::ToneCurve(ToneCurve::s_curve()))
            .push(GradingStep::Saturation(1.1))
            .push(GradingStep::Exposure(-0.1))
    }

    /// Cinematic preset: film curve, -15% saturation, +15% contrast.
    pub fn cinematic_preset() -> Self {
        Self::new()
            .push(GradingStep::ToneCurve(ToneCurve::film()))
            .push(GradingStep::Saturation(0.85))
            .push(GradingStep::Contrast(1.15))
    }
}

impl Default for ColorGradingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RgbHistogram
// ─────────────────────────────────────────────────────────────────────────────

/// Per-channel histogram for an RGB image.
pub struct RgbHistogram {
    /// Number of bins per channel.
    pub bins: usize,
    /// Bin counts for the red channel.
    pub r_counts: Vec<u32>,
    /// Bin counts for the green channel.
    pub g_counts: Vec<u32>,
    /// Bin counts for the blue channel.
    pub b_counts: Vec<u32>,
    /// Total number of pixels processed.
    pub total_pixels: usize,
}

impl RgbHistogram {
    /// Compute a histogram over a flat RGB image.
    ///
    /// Each pixel value in [0, 1] is mapped to a bin: `bin = (v * bins).floor()`,
    /// clamped to `bins - 1`.
    ///
    /// # Errors
    ///
    /// - [`ColorGradingError::InvalidImage`] if the image length is not a multiple of 3.
    /// - [`ColorGradingError::InvalidConfig`] if `bins == 0`.
    pub fn compute(image: &[f32], bins: usize) -> Result<Self, ColorGradingError> {
        validate_rgb_image(image)?;
        if bins == 0 {
            return Err(ColorGradingError::InvalidConfig(
                "bins must be > 0".to_string(),
            ));
        }

        let mut r_counts = vec![0u32; bins];
        let mut g_counts = vec![0u32; bins];
        let mut b_counts = vec![0u32; bins];

        let to_bin = |v: f32| -> usize {
            let idx = (v.clamp(0.0, 1.0) * bins as f32) as usize;
            idx.min(bins - 1)
        };

        for chunk in image.chunks_exact(3) {
            r_counts[to_bin(chunk[0])] += 1;
            g_counts[to_bin(chunk[1])] += 1;
            b_counts[to_bin(chunk[2])] += 1;
        }

        let total_pixels = image.len() / 3;
        Ok(Self {
            bins,
            r_counts,
            g_counts,
            b_counts,
            total_pixels,
        })
    }

    fn weighted_mean(counts: &[u32], bins: usize, total: usize) -> f32 {
        if total == 0 {
            return 0.0;
        }
        let scale = 1.0 / bins as f32;
        counts
            .iter()
            .enumerate()
            .map(|(i, &c)| (i as f32 + 0.5) * scale * c as f32)
            .sum::<f32>()
            / total as f32
    }

    /// Weighted mean of the red channel.
    pub fn mean_r(&self) -> f32 {
        Self::weighted_mean(&self.r_counts, self.bins, self.total_pixels)
    }

    /// Weighted mean of the green channel.
    pub fn mean_g(&self) -> f32 {
        Self::weighted_mean(&self.g_counts, self.bins, self.total_pixels)
    }

    /// Weighted mean of the blue channel.
    pub fn mean_b(&self) -> f32 {
        Self::weighted_mean(&self.b_counts, self.bins, self.total_pixels)
    }

    /// Bin index with the highest count in the red channel.
    pub fn peak_bin_r(&self) -> usize {
        self.r_counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Render a simple ASCII bar chart for the red channel.
    ///
    /// `height` is the maximum bar height in character rows.
    pub fn format_ascii(&self, height: usize) -> String {
        if self.bins == 0 || height == 0 {
            return String::new();
        }

        let max_count = self.r_counts.iter().copied().max().unwrap_or(0);
        if max_count == 0 {
            return "(empty)\n".to_string();
        }

        let mut rows: Vec<String> = Vec::with_capacity(height + 2);
        for row in (0..height).rev() {
            let threshold = max_count as f32 * (row as f32 + 1.0) / height as f32;
            let line: String = self
                .r_counts
                .iter()
                .map(|&c| if c as f32 >= threshold { '#' } else { ' ' })
                .collect();
            rows.push(line);
        }
        // Axis
        rows.push("-".repeat(self.bins));
        rows.join("\n") + "\n"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    // 1. srgb_to_linear boundary values
    #[test]
    fn test_srgb_to_linear_boundaries() {
        assert!(approx_eq(srgb_to_linear(0.0), 0.0, EPSILON));
        assert!(approx_eq(srgb_to_linear(1.0), 1.0, EPSILON));
    }

    // 2. linear_to_srgb roundtrip
    #[test]
    fn test_linear_to_srgb_roundtrip() {
        for v in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let roundtripped = linear_to_srgb(srgb_to_linear(v));
            assert!(
                approx_eq(v, roundtripped, 1e-3),
                "roundtrip failed for v={}: got {}",
                v,
                roundtripped
            );
        }
    }

    // 3. rgb_to_hsv_grading: red
    #[test]
    fn test_rgb_to_hsv_red() {
        let (h, s, v) = rgb_to_hsv_grading(1.0, 0.0, 0.0);
        assert!(approx_eq(h, 0.0, EPSILON), "hue: {}", h);
        assert!(approx_eq(s, 1.0, EPSILON), "sat: {}", s);
        assert!(approx_eq(v, 1.0, EPSILON), "val: {}", v);
    }

    // 4. rgb_to_hsv_grading: white → s=0, v=1
    #[test]
    fn test_rgb_to_hsv_white() {
        let (_, s, v) = rgb_to_hsv_grading(1.0, 1.0, 1.0);
        assert!(approx_eq(s, 0.0, EPSILON), "sat should be 0: {}", s);
        assert!(approx_eq(v, 1.0, EPSILON), "val should be 1: {}", v);
    }

    // 5. hsv_to_rgb_grading: black
    #[test]
    fn test_hsv_to_rgb_black() {
        let (r, g, b) = hsv_to_rgb_grading(0.0, 0.0, 0.0);
        assert!(approx_eq(r, 0.0, EPSILON));
        assert!(approx_eq(g, 0.0, EPSILON));
        assert!(approx_eq(b, 0.0, EPSILON));
    }

    // 6. hsv_to_rgb_grading: roundtrip
    #[test]
    fn test_hsv_rgb_roundtrip() {
        let cases = [(0.8, 0.3, 0.6f32), (0.1, 0.9, 0.7), (0.5, 0.5, 0.5)];
        for (r, g, b) in cases {
            let (h, s, v) = rgb_to_hsv_grading(r, g, b);
            let (r2, g2, b2) = hsv_to_rgb_grading(h, s, v);
            assert!(approx_eq(r, r2, 1e-3), "r roundtrip: {} → {}", r, r2);
            assert!(approx_eq(g, g2, 1e-3), "g roundtrip: {} → {}", g, g2);
            assert!(approx_eq(b, b2, 1e-3), "b roundtrip: {} → {}", b, b2);
        }
    }

    // 7. luminance: red channel, weights sum
    #[test]
    fn test_luminance_red_and_weights() {
        assert!(approx_eq(luminance(1.0, 0.0, 0.0), 0.2126, EPSILON));
        let total = 0.2126_f32 + 0.7152 + 0.0722;
        assert!(approx_eq(total, 1.0, EPSILON));
    }

    // 8. ExposureAdjust: stops=0 → no change
    #[test]
    fn test_exposure_zero_stops() {
        let ea = ExposureAdjust::new(0.0);
        let (r, g, b) = ea.apply_pixel(0.5, 0.3, 0.8);
        assert!(approx_eq(r, 0.5, EPSILON));
        assert!(approx_eq(g, 0.3, EPSILON));
        assert!(approx_eq(b, 0.8, EPSILON));
    }

    // 9. ExposureAdjust: stops=1 → doubled (clamped to 1)
    #[test]
    fn test_exposure_one_stop() {
        let ea = ExposureAdjust::new(1.0);
        let (r, _g, _b) = ea.apply_pixel(0.3, 0.0, 0.0);
        assert!(approx_eq(r, (0.3 * 2.0f32).min(1.0), EPSILON));

        // Values > 0.5 get clamped to 1.0
        let (r2, _, _) = ea.apply_pixel(0.8, 0.0, 0.0);
        assert!(approx_eq(r2, 1.0, EPSILON));
    }

    // 10. ExposureAdjust: wrong image length → Err
    #[test]
    fn test_exposure_wrong_length() {
        let ea = ExposureAdjust::new(0.0);
        let bad = vec![1.0f32, 0.5];
        assert!(ea.apply_image(&bad).is_err());
    }

    // 11. ContrastAdjust: contrast=1.0 → midpoint unchanged
    #[test]
    fn test_contrast_one_no_change() {
        let ca = ContrastAdjust::new(1.0).expect("valid contrast");
        assert!(approx_eq(ca.apply_pixel(0.5), 0.5, EPSILON));
    }

    // 12. ContrastAdjust: contrast > 1 → values spread from midpoint
    #[test]
    fn test_contrast_spreads_values() {
        let ca = ContrastAdjust::new(2.0).expect("valid contrast");
        // Value above 0.5 should increase; value below 0.5 should decrease.
        assert!(ca.apply_pixel(0.7) > 0.7);
        assert!(ca.apply_pixel(0.3) < 0.3);
    }

    // 13. SaturationAdjust: saturation=0 → grayscale
    #[test]
    fn test_saturation_zero_grayscale() {
        let sa = SaturationAdjust::new(0.0).expect("valid");
        let (r, g, b) = sa.apply_pixel(0.8, 0.3, 0.1);
        // All channels should be equal (grayscale)
        assert!(approx_eq(r, g, 1e-3), "r={} g={}", r, g);
        assert!(approx_eq(g, b, 1e-3), "g={} b={}", g, b);
    }

    // 14. SaturationAdjust: saturation=1 → unchanged
    #[test]
    fn test_saturation_one_unchanged() {
        let sa = SaturationAdjust::new(1.0).expect("valid");
        let (r, g, b) = sa.apply_pixel(0.8, 0.3, 0.1);
        assert!(approx_eq(r, 0.8, 1e-3));
        assert!(approx_eq(g, 0.3, 1e-3));
        assert!(approx_eq(b, 0.1, 1e-3));
    }

    // 15. ToneCurve::identity: evaluate at endpoints
    #[test]
    fn test_tone_curve_identity_endpoints() {
        let tc = ToneCurve::identity();
        assert!(approx_eq(tc.evaluate(0.0), 0.0, EPSILON));
        assert!(approx_eq(tc.evaluate(1.0), 1.0, EPSILON));
    }

    // 16. ToneCurve: linear interpolation between control points
    #[test]
    fn test_tone_curve_interpolation() {
        let tc = ToneCurve {
            control_points: vec![(0.0, 0.0), (0.5, 0.25), (1.0, 1.0)],
        };
        // Midpoint of [0,0]→[0.5,0.25] at x=0.25 should be y=0.125
        let y = tc.evaluate(0.25);
        assert!(approx_eq(y, 0.125, EPSILON), "got {}", y);
    }

    // 17. ToneCurve::s_curve: midpoint is preserved
    #[test]
    fn test_s_curve_midpoint() {
        let tc = ToneCurve::s_curve();
        assert!(approx_eq(tc.evaluate(0.5), 0.5, EPSILON));
    }

    // 18. Lut3D::identity: midpoint lookup
    #[test]
    fn test_lut3d_identity_midpoint() {
        let lut = Lut3D::identity(5).expect("valid size");
        let (r, g, b) = lut.lookup(0.5, 0.5, 0.5);
        assert!(approx_eq(r, 0.5, 0.05), "r={}", r);
        assert!(approx_eq(g, 0.5, 0.05), "g={}", g);
        assert!(approx_eq(b, 0.5, 0.05), "b={}", b);
    }

    // 19. Lut3D::identity: corners
    #[test]
    fn test_lut3d_identity_corners() {
        let lut = Lut3D::identity(5).expect("valid size");
        let (r, g, b) = lut.lookup(0.0, 0.0, 0.0);
        assert!(approx_eq(r, 0.0, EPSILON));
        assert!(approx_eq(g, 0.0, EPSILON));
        assert!(approx_eq(b, 0.0, EPSILON));

        let (r, g, b) = lut.lookup(1.0, 1.0, 1.0);
        assert!(approx_eq(r, 1.0, EPSILON));
        assert!(approx_eq(g, 1.0, EPSILON));
        assert!(approx_eq(b, 1.0, EPSILON));
    }

    // 20. Lut3D::from_tone_curve with identity curve → identity lookup
    #[test]
    fn test_lut3d_from_identity_curve() {
        let curve = ToneCurve::identity();
        let lut = Lut3D::from_tone_curve(9, &curve).expect("valid");
        let (r, g, b) = lut.lookup(0.3, 0.6, 0.9);
        assert!(approx_eq(r, 0.3, 0.02), "r={}", r);
        assert!(approx_eq(g, 0.6, 0.02), "g={}", g);
        assert!(approx_eq(b, 0.9, 0.02), "b={}", b);
    }

    // 20b. Lut3D::lookup: size == 0 does not panic (usize underflow guard).
    #[test]
    fn test_lut3d_lookup_zero_size_no_panic() {
        let lut = Lut3D {
            size: 0,
            table: Vec::new(),
        };
        assert_eq!(lut.lookup(0.5, 0.5, 0.5), (0.0, 0.0, 0.0));
    }

    // 20c. Lut3D::lookup: truncated table does not panic (out-of-bounds guard).
    #[test]
    fn test_lut3d_lookup_truncated_table_no_panic() {
        // size=2 needs 2*2*2*3=24 floats; give it far fewer.
        let lut = Lut3D {
            size: 2,
            table: vec![0.5f32; 4],
        };
        let (r, g, b) = lut.lookup(1.0, 1.0, 1.0);
        assert!(
            r.is_finite() && g.is_finite() && b.is_finite(),
            "missing table entries should read as 0.0, not panic: ({r}, {g}, {b})"
        );
    }

    // 20d. Lut3D::validate: rejects size < 2 and a mismatched table length.
    #[test]
    fn test_lut3d_validate_rejects_invalid_luts() {
        assert!(Lut3D {
            size: 0,
            table: Vec::new(),
        }
        .validate()
        .is_err());
        assert!(Lut3D {
            size: 2,
            table: vec![0.5f32; 4], // needs 24, has 4
        }
        .validate()
        .is_err());
        assert!(Lut3D::identity(4).expect("valid").validate().is_ok());
    }

    // 20e. Lut3D::apply_image: propagates validate()'s error instead of
    // panicking on a malformed LUT.
    #[test]
    fn test_lut3d_apply_image_rejects_invalid_lut() {
        let lut = Lut3D {
            size: 0,
            table: Vec::new(),
        };
        let image = vec![0.5f32, 0.5, 0.5];
        assert!(matches!(
            lut.apply_image(&image),
            Err(ColorGradingError::InvalidLut(_))
        ));
    }

    // 21. ColorGradingPipeline: empty pipeline → image unchanged
    #[test]
    fn test_pipeline_empty_unchanged() {
        let pipeline = ColorGradingPipeline::new();
        let image = vec![0.2f32, 0.5, 0.8, 0.1, 0.9, 0.4];
        let out = pipeline.apply(&image).expect("should succeed");
        for (a, b) in image.iter().zip(out.iter()) {
            assert!(approx_eq(*a, *b, EPSILON));
        }
    }

    // 22. ColorGradingPipeline: exposure=0 + saturation=1 → unchanged
    #[test]
    fn test_pipeline_noop_steps() {
        let pipeline = ColorGradingPipeline::new()
            .push(GradingStep::Exposure(0.0))
            .push(GradingStep::Saturation(1.0));
        let image = vec![0.4f32, 0.6, 0.2, 0.9, 0.1, 0.5];
        let out = pipeline.apply(&image).expect("should succeed");
        for (a, b) in image.iter().zip(out.iter()) {
            assert!(approx_eq(*a, *b, 2e-3), "a={} b={}", a, b);
        }
    }

    // 22b. ColorGradingPipeline: unregistered Custom step is now an error,
    // not a silent no-op.
    #[test]
    fn test_pipeline_unregistered_custom_step_errors() {
        let pipeline = ColorGradingPipeline::new().push(GradingStep::Custom("my_lut".into()));
        let image = vec![0.4f32, 0.6, 0.2];
        let result = pipeline.apply(&image);
        assert!(matches!(result, Err(ColorGradingError::InvalidConfig(_))));
    }

    // 22c. ColorGradingPipeline: registered Custom handler is actually invoked.
    #[test]
    fn test_pipeline_registered_custom_handler_is_invoked() {
        let pipeline = ColorGradingPipeline::new()
            .with_custom_handler("invert", |img| Ok(img.iter().map(|&v| 1.0 - v).collect()))
            .push(GradingStep::Custom("invert".to_string()));
        let image = vec![0.2f32, 0.5, 0.9];
        let out = pipeline
            .apply(&image)
            .expect("registered handler should run");
        for (a, b) in image.iter().zip(out.iter()) {
            assert!(
                approx_eq(1.0 - *a, *b, EPSILON),
                "expected inverted value: a={a} b={b}"
            );
        }
    }

    // 22d. ColorGradingPipeline: unrelated Custom name still errors even
    // when a different handler is registered.
    #[test]
    fn test_pipeline_custom_handler_name_must_match() {
        let pipeline = ColorGradingPipeline::new()
            .with_custom_handler("invert", |img| Ok(img.to_vec()))
            .push(GradingStep::Custom("not_invert".to_string()));
        let image = vec![0.4f32, 0.6, 0.2];
        let result = pipeline.apply(&image);
        assert!(matches!(result, Err(ColorGradingError::InvalidConfig(_))));
    }

    // 23. RgbHistogram: uniform image → all bins roughly equal
    #[test]
    fn test_histogram_uniform() {
        const BINS: usize = 4;
        const PIXELS: usize = 100;
        // Build image with evenly spread values across [0,1]
        let mut image = Vec::with_capacity(PIXELS * 3);
        for i in 0..PIXELS {
            let v = i as f32 / PIXELS as f32;
            image.push(v);
            image.push(v);
            image.push(v);
        }
        let hist = RgbHistogram::compute(&image, BINS).expect("valid");
        // Each bin should have approximately PIXELS/BINS counts
        let expected = PIXELS as f32 / BINS as f32;
        for &c in &hist.r_counts {
            let diff = (c as f32 - expected).abs();
            assert!(
                diff <= expected * 0.5 + 2.0,
                "count {} far from {}",
                c,
                expected
            );
        }
    }

    // 24. RgbHistogram::mean_r: correct weighted mean
    #[test]
    fn test_histogram_mean_r() {
        // Single pixel at 0.75
        let image = vec![0.75f32, 0.0, 0.0];
        let hist = RgbHistogram::compute(&image, 10).expect("valid");
        let mean = hist.mean_r();
        // Mean should be near 0.75 (within half a bin width = 0.05)
        assert!(approx_eq(mean, 0.75, 0.1), "mean_r={}", mean);
    }
}
