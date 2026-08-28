//! Camera exposure and tone adjustment pipeline for rendered images.
//!
//! Provides a comprehensive set of exposure control tools operating on u8 RGB
//! images (stored as `&[u8]` slices with layout H×W×3):
//!
//! - **EV adjustment** — multiply channels by 2^ev
//! - **Gamma correction** — standard power-law gamma
//! - **Lift / Gain / Offset** — ASC-CDL style shadow/highlight control
//! - **Full pipeline** — combine the above via [`ExposureConfig`]
//! - **Histogram tools** — [`LuminanceHistogram`] with CDF, percentile, mean
//! - **Auto-exposure / metering** — bring mean luminance to a target value
//! - **Histogram equalization** — global, HSL-based
//! - **CLAHE** — contrast-limited adaptive histogram equalization
//! - **Scene key** — log-average luminance (Reinhard 2002)
//! - **Exposure bracketing** — generate N exposures around a base EV

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by exposure operations.
#[derive(Debug, Error)]
pub enum ExposureError {
    /// Configuration parameter is out of valid range.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// Image has wrong pixel count or channel layout.
    #[error("Invalid image: width={w}, height={h}, channels={c}")]
    InvalidImage { w: usize, h: usize, c: usize },

    /// Image slice is empty.
    #[error("Empty image")]
    EmptyImage,

    /// Histogram has no data.
    #[error("Empty histogram")]
    EmptyHistogram,

    /// Slice length does not match expected pixel count.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

// ─────────────────────────────────────────────────────────────────────────────
// ExposureConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Exposure control parameters for the full adjustment pipeline.
///
/// Applied in order: EV → lift/gain/offset → gamma.
#[derive(Debug, Clone)]
pub struct ExposureConfig {
    /// Exposure value in stops (0 = no change, +1 = 2× brighter, -1 = 0.5×).
    pub ev: f32,
    /// Gamma correction applied **after** EV (default 1.0 = no change).
    pub gamma: f32,
    /// Shadow lift in [0, 0.5] (default 0.0).
    pub lift: f32,
    /// Highlight gain in [0.5, 2.0] (default 1.0).
    pub gain: f32,
    /// Brightness offset in [-0.5, 0.5] (default 0.0).
    pub offset: f32,
}

impl Default for ExposureConfig {
    fn default() -> Self {
        Self {
            ev: 0.0,
            gamma: 1.0,
            lift: 0.0,
            gain: 1.0,
            offset: 0.0,
        }
    }
}

impl ExposureConfig {
    /// Validate that all parameters are in their valid ranges.
    pub fn validate(&self) -> Result<(), ExposureError> {
        if self.gamma <= 0.0 {
            return Err(ExposureError::InvalidConfig(format!(
                "gamma must be > 0, got {}",
                self.gamma
            )));
        }
        if self.lift < 0.0 || self.lift > 0.5 {
            return Err(ExposureError::InvalidConfig(format!(
                "lift must be in [0, 0.5], got {}",
                self.lift
            )));
        }
        if self.gain < 0.5 || self.gain > 2.0 {
            return Err(ExposureError::InvalidConfig(format!(
                "gain must be in [0.5, 2.0], got {}",
                self.gain
            )));
        }
        if self.offset < -0.5 || self.offset > 0.5 {
            return Err(ExposureError::InvalidConfig(format!(
                "offset must be in [-0.5, 0.5], got {}",
                self.offset
            )));
        }
        Ok(())
    }

    /// Returns an identity config that makes no change to the image.
    pub fn identity() -> Self {
        Self::default()
    }

    /// Create a config that only adjusts EV stops; all other parameters are identity.
    pub fn from_ev(ev: f32) -> Self {
        Self {
            ev,
            ..Self::default()
        }
    }

    /// The linear scale factor corresponding to this EV: 2^ev.
    pub fn ev_scale(&self) -> f32 {
        (2.0_f32).powf(self.ev)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExposureMetering
// ─────────────────────────────────────────────────────────────────────────────

/// Result of auto-exposure metering.
#[derive(Debug, Clone)]
pub struct ExposureMetering {
    /// Mean (average) luminance over all pixels.
    pub mean_luminance: f32,
    /// Median luminance (50th percentile).
    pub median_luminance: f32,
    /// 5th percentile luminance (shadow anchor).
    pub p5_luminance: f32,
    /// 95th percentile luminance (highlight anchor).
    pub p95_luminance: f32,
    /// EV stop adjustment needed to bring `mean_luminance` to `target_luminance`.
    pub suggested_ev: f32,
    /// Fraction of pixels with luminance > 0.95 (overexposed).
    pub overexposed_fraction: f32,
    /// Fraction of pixels with luminance < 0.05 (underexposed).
    pub underexposed_fraction: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// LuminanceHistogram
// ─────────────────────────────────────────────────────────────────────────────

/// 256-bin histogram of BT.709 luminance values mapped from [0, 1] to bins [0, 255].
#[derive(Debug, Clone)]
pub struct LuminanceHistogram {
    /// Bin counts; `bins[i]` counts pixels with luminance in `[i/255, (i+1)/255)`.
    pub bins: Vec<u32>,
    /// Total number of pixels counted.
    pub total: usize,
}

impl Default for LuminanceHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl LuminanceHistogram {
    /// Create a new histogram with 256 zero-initialised bins.
    pub fn new() -> Self {
        Self {
            bins: vec![0u32; 256],
            total: 0,
        }
    }

    /// Build a histogram from a slice of per-pixel luminance values in [0, 1].
    ///
    /// Values outside [0, 1] are clamped before binning.
    pub fn from_luminances(luminances: &[f32]) -> Self {
        let mut hist = Self::new();
        for &l in luminances {
            let idx = (l.clamp(0.0, 1.0) * 255.0).round() as usize;
            let idx = idx.min(255);
            hist.bins[idx] += 1;
        }
        hist.total = luminances.len();
        hist
    }

    /// Compute the cumulative distribution function (running sum of bins).
    ///
    /// Returns a vec of length 256 where `cdf[i] = Σ bins[0..=i]`.
    pub fn cumulative(&self) -> Vec<u32> {
        let mut cdf = Vec::with_capacity(256);
        let mut running = 0u32;
        for &b in &self.bins {
            running = running.saturating_add(b);
            cdf.push(running);
        }
        cdf
    }

    /// Return the luminance value at the given percentile `p` (0–100).
    ///
    /// Returns a value in [0, 1].  `p` is clamped to [0, 100].
    pub fn percentile_value(&self, p: f32) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        let p_clamped = p.clamp(0.0, 100.0);
        let target = (p_clamped / 100.0 * self.total as f32).ceil() as u32;
        let cdf = self.cumulative();
        for (i, &cum) in cdf.iter().enumerate() {
            if cum >= target {
                return i as f32 / 255.0;
            }
        }
        1.0
    }

    /// Return the index of the bin with the highest count.
    pub fn peak_bin(&self) -> usize {
        self.bins
            .iter()
            .enumerate()
            .max_by_key(|&(_, &v)| v)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Compute the mean luminance from the histogram.
    ///
    /// Returns a value in [0, 1].  Returns 0.0 if the histogram is empty.
    pub fn mean(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        let weighted_sum: f64 = self
            .bins
            .iter()
            .enumerate()
            .map(|(i, &c)| i as f64 / 255.0 * c as f64)
            .sum();
        (weighted_sum / self.total as f64) as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BT.709 luminance helper (private to this module)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute BT.709 luminance from normalised (0-1) RGB triplet.
#[inline]
fn bt709_luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Validate that a u8 pixel buffer matches the expected H×W×3 layout.
fn validate_rgb_u8(pixels: &[u8], width: usize, height: usize) -> Result<(), ExposureError> {
    if pixels.is_empty() {
        return Err(ExposureError::EmptyImage);
    }
    let expected = width
        .checked_mul(height)
        .and_then(|wh| wh.checked_mul(3))
        .ok_or(ExposureError::InvalidImage {
            w: width,
            h: height,
            c: 3,
        })?;
    if pixels.len() != expected {
        return Err(ExposureError::DimensionMismatch {
            expected,
            actual: pixels.len(),
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Core exposure functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the BT.709 luminance for every pixel in an RGB u8 image.
///
/// Returns a `Vec<f32>` of length `height * width` with values in [0, 1].
pub fn compute_luminance_map(
    pixels: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<f32>, ExposureError> {
    validate_rgb_u8(pixels, width, height)?;
    let n = width * height;
    let mut lum = Vec::with_capacity(n);
    for px in pixels.chunks_exact(3) {
        let r = px[0] as f32 / 255.0;
        let g = px[1] as f32 / 255.0;
        let b = px[2] as f32 / 255.0;
        lum.push(bt709_luminance(r, g, b));
    }
    Ok(lum)
}

/// Apply an exposure adjustment of `ev` stops to an RGB u8 image in place.
///
/// Each channel is multiplied by `2^ev` and clamped to [0, 255].
/// `ev = 0` leaves the image unchanged.
pub fn apply_ev(pixels: &mut [u8], ev: f32) -> Result<(), ExposureError> {
    if pixels.is_empty() {
        return Err(ExposureError::EmptyImage);
    }
    if !pixels.len().is_multiple_of(3) {
        return Err(ExposureError::InvalidImage {
            w: 0,
            h: 0,
            c: pixels.len() % 3,
        });
    }
    let scale = (2.0_f32).powf(ev);
    for v in pixels.iter_mut() {
        let out = (*v as f32 * scale).clamp(0.0, 255.0).round();
        *v = out as u8;
    }
    Ok(())
}

/// Apply gamma correction to an RGB u8 image in place.
///
/// `pixel_out = (pixel_in / 255)^(1 / gamma) * 255`.
///
/// - `gamma = 1.0` — no change.
/// - `gamma = 2.2` — approximate sRGB **encode** (linear → display; raising
///   to the `1/2.2` power brightens the image). To perform the sRGB
///   **decode** (display → linear, which *darkens* the image) instead,
///   pass `gamma = 1.0 / 2.2` so the exponent becomes `2.2`.
/// - `gamma < 1.0` — darkens.
pub fn apply_gamma_correction(pixels: &mut [u8], gamma: f32) -> Result<(), ExposureError> {
    if pixels.is_empty() {
        return Err(ExposureError::EmptyImage);
    }
    if gamma <= 0.0 {
        return Err(ExposureError::InvalidConfig(format!(
            "gamma must be > 0, got {}",
            gamma
        )));
    }
    let inv_gamma = 1.0 / gamma;
    for v in pixels.iter_mut() {
        let normalised = *v as f32 / 255.0;
        let corrected = normalised.powf(inv_gamma).clamp(0.0, 1.0);
        *v = (corrected * 255.0).round() as u8;
    }
    Ok(())
}

/// Apply lift / gain / offset colour grading to an RGB u8 image in place.
///
/// For each channel value `v_in ∈ [0, 1]`:
/// ```text
/// v_out = (v_in + offset) * gain + lift
/// ```
/// The result is clamped to [0, 255] after rescaling.
pub fn apply_lift_gain_offset(
    pixels: &mut [u8],
    lift: f32,
    gain: f32,
    offset: f32,
) -> Result<(), ExposureError> {
    if pixels.is_empty() {
        return Err(ExposureError::EmptyImage);
    }
    for v in pixels.iter_mut() {
        let normalised = *v as f32 / 255.0;
        let adjusted = (normalised + offset) * gain + lift;
        *v = (adjusted.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    Ok(())
}

/// Apply the full [`ExposureConfig`] pipeline to an RGB u8 image in place.
///
/// Processing order:
/// 1. EV adjustment (`apply_ev`)
/// 2. Lift / gain / offset (`apply_lift_gain_offset`)
/// 3. Gamma correction (`apply_gamma_correction`)
pub fn apply_exposure(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    config: &ExposureConfig,
) -> Result<(), ExposureError> {
    validate_rgb_u8(pixels, width, height)?;
    config.validate()?;

    apply_ev(pixels, config.ev)?;
    apply_lift_gain_offset(pixels, config.lift, config.gain, config.offset)?;
    apply_gamma_correction(pixels, config.gamma)?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// RGB ↔ HSL conversion
// ─────────────────────────────────────────────────────────────────────────────

/// Convert an 8-bit RGB colour to HSL.
///
/// Returns `(h, s, l)` where `h ∈ [0, 360)`, `s, l ∈ [0, 1]`.
pub fn rgb_to_hsl(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;

    let cmax = rf.max(gf).max(bf);
    let cmin = rf.min(gf).min(bf);
    let delta = cmax - cmin;

    let l = (cmax + cmin) / 2.0;

    let s = if delta < 1e-7 {
        0.0
    } else {
        delta / (1.0 - (2.0 * l - 1.0).abs())
    };

    let h = if delta < 1e-7 {
        0.0
    } else if cmax == rf {
        let segment = (gf - bf) / delta;
        let shift = if segment < 0.0 { 6.0 } else { 0.0 };
        60.0 * (segment + shift)
    } else if cmax == gf {
        60.0 * ((bf - rf) / delta + 2.0)
    } else {
        60.0 * ((rf - gf) / delta + 4.0)
    };

    (h, s.clamp(0.0, 1.0), l.clamp(0.0, 1.0))
}

/// Convert HSL to 8-bit RGB.
///
/// `h ∈ [0, 360)` (wraps), `s, l ∈ [0, 1]`.
pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let s = s.clamp(0.0, 1.0);
    let l = l.clamp(0.0, 1.0);

    if s < 1e-7 {
        let v = (l * 255.0).round().clamp(0.0, 255.0) as u8;
        return (v, v, v);
    }

    let h = ((h % 360.0) + 360.0) % 360.0;

    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r1, g1, b1) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0_f32),
        1 => (x, c, 0.0_f32),
        2 => (0.0_f32, c, x),
        3 => (0.0_f32, x, c),
        4 => (x, 0.0_f32, c),
        _ => (c, 0.0_f32, x),
    };

    let to_u8 = |v: f32| ((v + m).clamp(0.0, 1.0) * 255.0).round() as u8;
    (to_u8(r1), to_u8(g1), to_u8(b1))
}

// ─────────────────────────────────────────────────────────────────────────────
// Histogram equalization
// ─────────────────────────────────────────────────────────────────────────────

/// Perform global histogram equalization on the luminance (L) channel of an
/// RGB u8 image, preserving hue and saturation.
///
/// The image is converted to HSL, the L channel is equalized using a global
/// CDF-based mapping, then converted back to RGB.
pub fn histogram_equalize(
    pixels: &mut [u8],
    width: usize,
    height: usize,
) -> Result<(), ExposureError> {
    validate_rgb_u8(pixels, width, height)?;

    let n = width * height;

    // Build histogram of L values (256 bins)
    let mut bins = [0u32; 256];
    // Collect HSL values to avoid recomputing
    let mut hsl_vals: Vec<(f32, f32, f32)> = Vec::with_capacity(n);
    for px in pixels.chunks_exact(3) {
        let (h, s, l) = rgb_to_hsl(px[0], px[1], px[2]);
        hsl_vals.push((h, s, l));
        let idx = (l * 255.0).round().clamp(0.0, 255.0) as usize;
        bins[idx.min(255)] += 1;
    }

    // Build CDF
    let cdf: Vec<u32> = {
        let mut running = 0u32;
        let mut v = Vec::with_capacity(256);
        for &b in &bins {
            running = running.saturating_add(b);
            v.push(running);
        }
        v
    };
    let cdf_min = *cdf.iter().find(|&&x| x > 0).unwrap_or(&0);
    let total = n as u32;

    // Equalization map: bin → new L in [0, 1]
    let eq_map: Vec<f32> = cdf
        .iter()
        .map(|&c| {
            if total <= cdf_min {
                0.0
            } else {
                (c.saturating_sub(cdf_min)) as f32 / (total - cdf_min) as f32
            }
        })
        .collect();

    // Apply equalized L back to image
    for (i, (h, s, l)) in hsl_vals.iter().enumerate() {
        let l_idx = (*l * 255.0).round().clamp(0.0, 255.0) as usize;
        let l_new = eq_map[l_idx.min(255)];
        let (r, g, b) = hsl_to_rgb(*h, *s, l_new);
        pixels[i * 3] = r;
        pixels[i * 3 + 1] = g;
        pixels[i * 3 + 2] = b;
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Exposure metering
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the luminance histogram of an RGB u8 image.
pub fn compute_histogram(
    pixels: &[u8],
    width: usize,
    height: usize,
) -> Result<LuminanceHistogram, ExposureError> {
    validate_rgb_u8(pixels, width, height)?;
    let lum = compute_luminance_map(pixels, width, height)?;
    Ok(LuminanceHistogram::from_luminances(&lum))
}

/// Meter the exposure of an image relative to a desired `target_luminance`.
///
/// Computes statistical luminance descriptors and the EV adjustment needed
/// to bring the mean luminance to `target_luminance`.
pub fn meter_exposure(
    pixels: &[u8],
    width: usize,
    height: usize,
    target_luminance: f32,
) -> Result<ExposureMetering, ExposureError> {
    validate_rgb_u8(pixels, width, height)?;
    let lum_map = compute_luminance_map(pixels, width, height)?;
    let n = lum_map.len();

    if n == 0 {
        return Err(ExposureError::EmptyImage);
    }

    let hist = LuminanceHistogram::from_luminances(&lum_map);
    let mean_luminance = hist.mean();
    let median_luminance = hist.percentile_value(50.0);
    let p5_luminance = hist.percentile_value(5.0);
    let p95_luminance = hist.percentile_value(95.0);

    let overexposed_fraction = lum_map.iter().filter(|&&l| l > 0.95).count() as f32 / n as f32;
    let underexposed_fraction = lum_map.iter().filter(|&&l| l < 0.05).count() as f32 / n as f32;

    // suggested_ev = log2(target / mean) — positive if image is too dark
    let suggested_ev = if mean_luminance > 1e-7 {
        (target_luminance / mean_luminance).log2()
    } else {
        // Image is nearly black — clamp to a large positive EV
        (target_luminance / 1e-7_f32).log2()
    };

    Ok(ExposureMetering {
        mean_luminance,
        median_luminance,
        p5_luminance,
        p95_luminance,
        suggested_ev,
        overexposed_fraction,
        underexposed_fraction,
    })
}

/// Compute and apply the optimal EV to bring mean luminance to `target_luminance`.
///
/// Returns the metering result *before* the EV correction is applied (so
/// `suggested_ev` indicates how much was adjusted).
pub fn auto_expose(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    target_luminance: f32,
) -> Result<ExposureMetering, ExposureError> {
    validate_rgb_u8(pixels, width, height)?;
    let metering = meter_exposure(pixels, width, height, target_luminance)?;
    apply_ev(pixels, metering.suggested_ev)?;
    Ok(metering)
}

// ─────────────────────────────────────────────────────────────────────────────
// CLAHE
// ─────────────────────────────────────────────────────────────────────────────

/// Build an equalization look-up table for a single tile's luminance histogram.
///
/// `clip_limit` caps any bin before redistributing excess counts evenly.
fn build_tile_eq_map(tile_bins: &[u32; 256], clip_limit: u32) -> [u8; 256] {
    // Clip and redistribute
    let mut bins = *tile_bins;
    let total_before: u32 = bins.iter().sum();

    if clip_limit > 0 {
        let mut excess = 0u32;
        for b in bins.iter_mut() {
            if *b > clip_limit {
                excess = excess.saturating_add(*b - clip_limit);
                *b = clip_limit;
            }
        }
        // Redistribute excess evenly
        let per_bin = excess / 256;
        let remainder = (excess % 256) as usize;
        for b in bins.iter_mut() {
            *b = b.saturating_add(per_bin);
        }
        for b in bins.iter_mut().take(remainder) {
            *b = b.saturating_add(1);
        }
    }

    // Build CDF
    let mut cdf = [0u32; 256];
    let mut running = 0u32;
    for (i, &b) in bins.iter().enumerate() {
        running = running.saturating_add(b);
        cdf[i] = running;
    }

    let cdf_min = *cdf.iter().find(|&&x| x > 0).unwrap_or(&0);
    let total = total_before.max(1);

    let mut eq_map = [0u8; 256];
    for (i, &c) in cdf.iter().enumerate() {
        let mapped = if total <= cdf_min {
            0u8
        } else {
            let num = c.saturating_sub(cdf_min) as f32;
            let den = (total - cdf_min) as f32;
            ((num / den) * 255.0).round().clamp(0.0, 255.0) as u8
        };
        eq_map[i] = mapped;
    }
    eq_map
}

/// CLAHE — Contrast Limited Adaptive Histogram Equalization.
///
/// Divides the image into a `grid_size × grid_size` grid of tiles, computes
/// an equalization map for each tile (with bin clipping at `clip_limit`), then
/// bilinearly interpolates the four surrounding tile maps for each pixel.
///
/// `clip_limit = 0` disables clipping (standard AHE).
pub fn clahe(
    pixels: &mut [u8],
    width: usize,
    height: usize,
    grid_size: usize,
    clip_limit: u32,
) -> Result<(), ExposureError> {
    validate_rgb_u8(pixels, width, height)?;

    if grid_size == 0 {
        return Err(ExposureError::InvalidConfig(
            "grid_size must be >= 1".to_string(),
        ));
    }

    let cols = grid_size;
    let cols_f = cols as f32;
    let rows = grid_size;
    let rows_f = rows as f32;

    // Tile pixel dimensions (last tile may be smaller)
    let tile_w = width.div_ceil(cols);
    let tile_h = height.div_ceil(rows);

    // Collect HSL values for all pixels
    let n = width * height;
    let mut hsl_all: Vec<(f32, f32, f32)> = Vec::with_capacity(n);
    for px in pixels.chunks_exact(3) {
        hsl_all.push(rgb_to_hsl(px[0], px[1], px[2]));
    }

    // Build per-tile equalization maps
    // tile_maps[ty][tx] = [u8; 256]
    let mut tile_maps: Vec<Vec<[u8; 256]>> = Vec::with_capacity(rows);
    for ty in 0..rows {
        let mut row_maps: Vec<[u8; 256]> = Vec::with_capacity(cols);
        for tx in 0..cols {
            let mut bins = [0u32; 256];
            let y_start = ty * tile_h;
            let y_end = ((ty + 1) * tile_h).min(height);
            let x_start = tx * tile_w;
            let x_end = ((tx + 1) * tile_w).min(width);
            for y in y_start..y_end {
                for x in x_start..x_end {
                    let (_, _, l) = hsl_all[y * width + x];
                    let idx = (l * 255.0).round().clamp(0.0, 255.0) as usize;
                    bins[idx.min(255)] += 1;
                }
            }
            row_maps.push(build_tile_eq_map(&bins, clip_limit));
        }
        tile_maps.push(row_maps);
    }

    // Bilinear interpolation between tile maps
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let (h, s, l) = hsl_all[idx];
            let l_bin = (l * 255.0).round().clamp(0.0, 255.0) as usize;
            let l_bin = l_bin.min(255);

            // Tile coordinates (fractional)
            // Map pixel centre to tile-centre coordinates
            let fx = (x as f32 + 0.5) / width as f32 * cols_f - 0.5;
            let fy = (y as f32 + 0.5) / height as f32 * rows_f - 0.5;

            let tx0 = fx.floor() as i32;
            let ty0 = fy.floor() as i32;
            let tx1 = tx0 + 1;
            let ty1 = ty0 + 1;

            let wx1 = (fx - tx0 as f32).clamp(0.0, 1.0);
            let wy1 = (fy - ty0 as f32).clamp(0.0, 1.0);
            let wx0 = 1.0 - wx1;
            let wy0 = 1.0 - wy1;

            let clamp_tx = |t: i32| (t.clamp(0, (cols as i32) - 1)) as usize;
            let clamp_ty = |t: i32| (t.clamp(0, (rows as i32) - 1)) as usize;

            let v00 = tile_maps[clamp_ty(ty0)][clamp_tx(tx0)][l_bin] as f32;
            let v10 = tile_maps[clamp_ty(ty0)][clamp_tx(tx1)][l_bin] as f32;
            let v01 = tile_maps[clamp_ty(ty1)][clamp_tx(tx0)][l_bin] as f32;
            let v11 = tile_maps[clamp_ty(ty1)][clamp_tx(tx1)][l_bin] as f32;

            let l_new_u8 = (v00 * wx0 * wy0 + v10 * wx1 * wy0 + v01 * wx0 * wy1 + v11 * wx1 * wy1)
                .round()
                .clamp(0.0, 255.0) as u8;

            let l_new = l_new_u8 as f32 / 255.0;
            let (r, g, b) = hsl_to_rgb(h, s, l_new);
            pixels[idx * 3] = r;
            pixels[idx * 3 + 1] = g;
            pixels[idx * 3 + 2] = b;
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene key (log-average luminance)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the log-average luminance of the scene (Reinhard 2002 key value).
///
/// `key = exp(1/N * Σ log(δ + L_i))` where `δ = 1e-6` for numerical stability.
pub fn scene_key(pixels: &[u8], width: usize, height: usize) -> Result<f32, ExposureError> {
    validate_rgb_u8(pixels, width, height)?;
    let lum_map = compute_luminance_map(pixels, width, height)?;
    let n = lum_map.len();
    if n == 0 {
        return Err(ExposureError::EmptyImage);
    }
    let delta = 1e-6_f32;
    let log_sum: f32 = lum_map.iter().map(|&l| (delta + l).ln()).sum();
    Ok((log_sum / n as f32).exp())
}

// ─────────────────────────────────────────────────────────────────────────────
// Exposure bracketing
// ─────────────────────────────────────────────────────────────────────────────

/// Generate `n` bracketed exposures around `base_ev` with step `ev_step`.
///
/// If `n = 3` and `ev_step = 1.0`, the returned images have EVs
/// `base_ev - 1.0`, `base_ev`, `base_ev + 1.0` (indices 0, 1, 2).
///
/// Returned images share the same H×W×3 u8 layout as the input.
pub fn exposure_bracket(
    pixels: &[u8],
    width: usize,
    height: usize,
    base_ev: f32,
    ev_step: f32,
    n: usize,
) -> Result<Vec<Vec<u8>>, ExposureError> {
    validate_rgb_u8(pixels, width, height)?;
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut results = Vec::with_capacity(n);
    let start_idx = if n > 1 {
        -((n as f32 - 1.0) / 2.0)
    } else {
        0.0
    };

    for i in 0..n {
        let ev = base_ev + (start_idx + i as f32) * ev_step;
        let mut buf = pixels.to_vec();
        apply_ev(&mut buf, ev)?;
        results.push(buf);
    }
    Ok(results)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ExposureConfig ──────────────────────────────────────────────────────

    #[test]
    fn test_exposure_config_default() {
        let cfg = ExposureConfig::default();
        assert_eq!(cfg.ev, 0.0);
        assert_eq!(cfg.gamma, 1.0);
        assert_eq!(cfg.lift, 0.0);
        assert_eq!(cfg.gain, 1.0);
        assert_eq!(cfg.offset, 0.0);
    }

    #[test]
    fn test_exposure_config_identity_validates() {
        assert!(ExposureConfig::identity().validate().is_ok());
    }

    #[test]
    fn test_exposure_config_gamma_zero_invalid() {
        let cfg = ExposureConfig {
            gamma: 0.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_exposure_config_gamma_negative_invalid() {
        let cfg = ExposureConfig {
            gamma: -1.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_exposure_config_lift_out_of_range() {
        let cfg = ExposureConfig {
            lift: 0.6,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_exposure_config_gain_out_of_range() {
        let cfg = ExposureConfig {
            gain: 3.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_exposure_config_offset_out_of_range() {
        let cfg = ExposureConfig {
            offset: 0.6,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_exposure_config_from_ev() {
        let cfg = ExposureConfig::from_ev(2.0);
        assert_eq!(cfg.ev, 2.0);
        assert_eq!(cfg.gamma, 1.0);
        assert_eq!(cfg.lift, 0.0);
        assert_eq!(cfg.gain, 1.0);
    }

    #[test]
    fn test_exposure_config_ev_scale() {
        let cfg = ExposureConfig::from_ev(1.0);
        let scale = cfg.ev_scale();
        assert!((scale - 2.0).abs() < 1e-5, "expected 2.0 got {}", scale);

        let cfg0 = ExposureConfig::from_ev(0.0);
        assert!((cfg0.ev_scale() - 1.0).abs() < 1e-5);

        let cfg_neg = ExposureConfig::from_ev(-1.0);
        assert!((cfg_neg.ev_scale() - 0.5).abs() < 1e-5);
    }

    // ── LuminanceHistogram ──────────────────────────────────────────────────

    #[test]
    fn test_histogram_new_all_zero() {
        let h = LuminanceHistogram::new();
        assert_eq!(h.bins.len(), 256);
        assert!(h.bins.iter().all(|&b| b == 0));
        assert_eq!(h.total, 0);
    }

    #[test]
    fn test_histogram_from_luminances_black() {
        let lums = vec![0.0f32; 100];
        let h = LuminanceHistogram::from_luminances(&lums);
        assert_eq!(h.bins[0], 100);
        assert_eq!(h.total, 100);
    }

    #[test]
    fn test_histogram_from_luminances_white() {
        let lums = vec![1.0f32; 50];
        let h = LuminanceHistogram::from_luminances(&lums);
        assert_eq!(h.bins[255], 50);
        assert_eq!(h.total, 50);
    }

    #[test]
    fn test_histogram_cumulative() {
        let lums = vec![0.0_f32, 0.5, 1.0];
        let h = LuminanceHistogram::from_luminances(&lums);
        let cdf = h.cumulative();
        assert_eq!(cdf.len(), 256);
        assert_eq!(*cdf.last().unwrap_or(&0), 3);
    }

    #[test]
    fn test_histogram_percentile_empty() {
        let h = LuminanceHistogram::new();
        assert_eq!(h.percentile_value(50.0), 0.0);
    }

    #[test]
    fn test_histogram_percentile_all_black() {
        let lums = vec![0.0_f32; 10];
        let h = LuminanceHistogram::from_luminances(&lums);
        let p50 = h.percentile_value(50.0);
        assert!(p50 < 0.05, "expected near 0 got {}", p50);
    }

    #[test]
    fn test_histogram_percentile_all_white() {
        let lums = vec![1.0_f32; 10];
        let h = LuminanceHistogram::from_luminances(&lums);
        let p50 = h.percentile_value(50.0);
        assert!(p50 > 0.95, "expected near 1 got {}", p50);
    }

    #[test]
    fn test_histogram_mean_uniform() {
        // All at luminance 0.5 → mean ≈ 0.5
        let lums = vec![0.5_f32; 1000];
        let h = LuminanceHistogram::from_luminances(&lums);
        let m = h.mean();
        assert!((m - 0.5).abs() < 0.01, "mean={}", m);
    }

    #[test]
    fn test_histogram_mean_empty() {
        let h = LuminanceHistogram::new();
        assert_eq!(h.mean(), 0.0);
    }

    #[test]
    fn test_histogram_peak_bin_all_zeros() {
        // All at 0 → peak_bin should be 0
        let lums = vec![0.0_f32; 5];
        let h = LuminanceHistogram::from_luminances(&lums);
        assert_eq!(h.peak_bin(), 0);
    }

    #[test]
    fn test_histogram_peak_bin_all_white() {
        let lums = vec![1.0_f32; 5];
        let h = LuminanceHistogram::from_luminances(&lums);
        assert_eq!(h.peak_bin(), 255);
    }

    // ── compute_luminance_map ───────────────────────────────────────────────

    #[test]
    fn test_compute_luminance_map_white() {
        let pixels = vec![255u8, 255, 255];
        let lum = compute_luminance_map(&pixels, 1, 1).unwrap();
        assert!((lum[0] - 1.0).abs() < 0.001, "got {}", lum[0]);
    }

    #[test]
    fn test_compute_luminance_map_black() {
        let pixels = vec![0u8, 0, 0];
        let lum = compute_luminance_map(&pixels, 1, 1).unwrap();
        assert!((lum[0] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_luminance_map_gray() {
        // Pure gray: R=G=B so luminance = value
        let v = 128u8;
        let pixels = vec![v, v, v];
        let lum = compute_luminance_map(&pixels, 1, 1).unwrap();
        let expected = v as f32 / 255.0;
        assert!((lum[0] - expected).abs() < 0.01, "got {}", lum[0]);
    }

    #[test]
    fn test_compute_luminance_map_length() {
        let pixels = vec![100u8; 3 * 4 * 5]; // 4×5 image
        let lum = compute_luminance_map(&pixels, 4, 5).unwrap();
        assert_eq!(lum.len(), 20);
    }

    #[test]
    fn test_compute_luminance_map_wrong_size() {
        let pixels = vec![0u8; 5]; // not divisible by 3
        assert!(compute_luminance_map(&pixels, 1, 1).is_err());
    }

    // ── apply_ev ───────────────────────────────────────────────────────────

    #[test]
    fn test_apply_ev_zero_no_change() {
        let original = vec![100u8, 150, 200];
        let mut pixels = original.clone();
        apply_ev(&mut pixels, 0.0).unwrap();
        assert_eq!(pixels, original);
    }

    #[test]
    fn test_apply_ev_one_doubles_brightness() {
        let mut pixels = vec![64u8, 64, 64];
        apply_ev(&mut pixels, 1.0).unwrap();
        // 64 * 2 = 128
        assert_eq!(pixels[0], 128);
    }

    #[test]
    fn test_apply_ev_clamping() {
        let mut pixels = vec![200u8, 200, 200];
        apply_ev(&mut pixels, 2.0).unwrap(); // 200 * 4 = 800 → 255
        assert_eq!(pixels[0], 255);
    }

    #[test]
    fn test_apply_ev_negative() {
        let mut pixels = vec![128u8, 128, 128];
        apply_ev(&mut pixels, -1.0).unwrap(); // 128 * 0.5 = 64
        assert_eq!(pixels[0], 64);
    }

    // ── apply_gamma_correction ─────────────────────────────────────────────

    #[test]
    fn test_apply_gamma_identity() {
        let original = vec![50u8, 128, 200];
        let mut pixels = original.clone();
        apply_gamma_correction(&mut pixels, 1.0).unwrap();
        // gamma=1.0 means x^(1/1) = x — should be unchanged (within rounding)
        for (a, b) in pixels.iter().zip(original.iter()) {
            let diff = (*a as i32 - *b as i32).abs();
            assert!(diff <= 1, "diff {} at channel", diff);
        }
    }

    #[test]
    fn test_apply_gamma_darkens_with_gamma_less_than_1() {
        // gamma=0.5 → x^(1/0.5) = x^2 → darker
        let mut pixels = vec![128u8, 128, 128];
        apply_gamma_correction(&mut pixels, 0.5).unwrap();
        assert!(pixels[0] < 128, "expected darker, got {}", pixels[0]);
    }

    #[test]
    fn test_apply_gamma_brightens_with_gamma_greater_than_1() {
        // gamma=2.0 → x^(1/2) = sqrt(x) → brighter
        let mut pixels = vec![128u8, 128, 128];
        apply_gamma_correction(&mut pixels, 2.0).unwrap();
        assert!(pixels[0] > 128, "expected brighter, got {}", pixels[0]);
    }

    #[test]
    fn test_apply_gamma_zero_invalid() {
        let mut pixels = vec![128u8, 128, 128];
        assert!(apply_gamma_correction(&mut pixels, 0.0).is_err());
    }

    // ── apply_lift_gain_offset ─────────────────────────────────────────────

    #[test]
    fn test_apply_lift_gain_offset_identity() {
        let original = vec![50u8, 100, 200];
        let mut pixels = original.clone();
        apply_lift_gain_offset(&mut pixels, 0.0, 1.0, 0.0).unwrap();
        for (a, b) in pixels.iter().zip(original.iter()) {
            let diff = (*a as i32 - *b as i32).abs();
            assert!(diff <= 1, "identity failed: diff {}", diff);
        }
    }

    #[test]
    fn test_apply_lift_brightens_shadows() {
        let mut pixels = vec![0u8, 0, 0]; // pure black
        apply_lift_gain_offset(&mut pixels, 0.2, 1.0, 0.0).unwrap();
        // lift=0.2 → 0 + 0.2 = 0.2 → 51
        assert!(
            pixels[0] > 0,
            "lift should brighten shadows, got {}",
            pixels[0]
        );
    }

    #[test]
    fn test_apply_gain_doubles_midtones() {
        let mut pixels = vec![64u8, 64, 64];
        apply_lift_gain_offset(&mut pixels, 0.0, 2.0, 0.0).unwrap();
        // 64/255 * 2.0 ≈ 0.502 → 128
        assert!((pixels[0] as i32 - 128).abs() <= 1, "got {}", pixels[0]);
    }

    // ── apply_exposure ─────────────────────────────────────────────────────

    #[test]
    fn test_apply_exposure_pipeline() {
        // Identity config should leave image nearly unchanged
        let original = vec![100u8, 150, 200];
        let mut pixels = original.clone();
        let cfg = ExposureConfig::identity();
        apply_exposure(&mut pixels, 1, 1, &cfg).unwrap();
        for (a, b) in pixels.iter().zip(original.iter()) {
            let diff = (*a as i32 - *b as i32).abs();
            assert!(diff <= 1, "identity pipeline changed pixel by {}", diff);
        }
    }

    #[test]
    fn test_apply_exposure_ev_only() {
        let mut pixels = vec![64u8, 64, 64];
        let cfg = ExposureConfig::from_ev(1.0);
        apply_exposure(&mut pixels, 1, 1, &cfg).unwrap();
        // ev=1 → 64*2=128
        assert_eq!(pixels[0], 128);
    }

    #[test]
    fn test_apply_exposure_invalid_config() {
        let mut pixels = vec![128u8, 128, 128];
        let cfg = ExposureConfig {
            gamma: -1.0,
            ..Default::default()
        };
        assert!(apply_exposure(&mut pixels, 1, 1, &cfg).is_err());
    }

    // ── histogram_equalize ─────────────────────────────────────────────────

    #[test]
    fn test_histogram_equalize_same_size() {
        let w = 4;
        let h = 4;
        let mut pixels: Vec<u8> = (0..(w * h))
            .flat_map(|i| vec![(i * 16) as u8, (i * 8) as u8, (i * 4) as u8])
            .collect();
        histogram_equalize(&mut pixels, w, h).unwrap();
        assert_eq!(pixels.len(), w * h * 3);
    }

    #[test]
    fn test_histogram_equalize_uniform_image() {
        // A uniform image should remain uniform after equalization
        let mut pixels = vec![128u8; 3 * 9]; // 3×3, all 50% gray
        histogram_equalize(&mut pixels, 3, 3).unwrap();
        assert_eq!(pixels.len(), 3 * 3 * 3);
    }

    // ── meter_exposure ──────────────────────────────────────────────────────

    #[test]
    fn test_meter_exposure_white_image_suggests_negative_ev() {
        let pixels = vec![255u8; 3 * 4]; // 1×4 all white
        let m = meter_exposure(&pixels, 1, 4, 0.5).unwrap();
        // Mean lum ≈ 1.0, target = 0.5 → suggested_ev should be negative
        assert!(
            m.suggested_ev < 0.0,
            "expected neg EV, got {}",
            m.suggested_ev
        );
        assert!(m.mean_luminance > 0.9);
    }

    #[test]
    fn test_meter_exposure_black_image_suggests_positive_ev() {
        let pixels = vec![0u8; 3 * 4]; // all black
        let m = meter_exposure(&pixels, 1, 4, 0.5).unwrap();
        assert!(
            m.suggested_ev > 0.0,
            "expected pos EV, got {}",
            m.suggested_ev
        );
        assert!(m.underexposed_fraction > 0.5);
    }

    #[test]
    fn test_meter_exposure_overexposed_fraction() {
        let pixels = vec![255u8; 3 * 10]; // all white
        let m = meter_exposure(&pixels, 1, 10, 0.5).unwrap();
        assert!((m.overexposed_fraction - 1.0).abs() < 0.01);
        assert!((m.underexposed_fraction - 0.0).abs() < 0.01);
    }

    // ── auto_expose ─────────────────────────────────────────────────────────

    #[test]
    fn test_auto_expose_approaches_target() {
        // Start with a dark image, auto-expose to 0.5
        let mut pixels = vec![50u8; 3 * 16]; // 4×4
        let target = 0.5_f32;
        auto_expose(&mut pixels, 4, 4, target).unwrap();

        let lum_map = compute_luminance_map(&pixels, 4, 4).unwrap();
        let mean_after: f32 = lum_map.iter().sum::<f32>() / lum_map.len() as f32;
        // Mean should be closer to 0.5 than the original (~0.196)
        assert!(
            (mean_after - target).abs() < 0.15,
            "mean after auto_expose = {}",
            mean_after
        );
    }

    // ── compute_histogram ───────────────────────────────────────────────────

    #[test]
    fn test_compute_histogram_all_black() {
        let pixels = vec![0u8; 3 * 5];
        let h = compute_histogram(&pixels, 1, 5).unwrap();
        assert_eq!(h.bins[0], 5);
    }

    #[test]
    fn test_compute_histogram_all_white() {
        let pixels = vec![255u8; 3 * 5];
        let h = compute_histogram(&pixels, 1, 5).unwrap();
        assert_eq!(h.bins[255], 5);
    }

    // ── clahe ───────────────────────────────────────────────────────────────

    #[test]
    fn test_clahe_same_size() {
        let w = 8;
        let h = 8;
        let mut pixels: Vec<u8> = (0..(w * h))
            .flat_map(|i| vec![(i * 3) as u8, (i * 2) as u8, i as u8])
            .collect();
        clahe(&mut pixels, w, h, 4, 40).unwrap();
        assert_eq!(pixels.len(), w * h * 3);
    }

    #[test]
    fn test_clahe_1x1_no_crash() {
        let mut pixels = vec![128u8, 64, 200];
        clahe(&mut pixels, 1, 1, 1, 40).unwrap();
        assert_eq!(pixels.len(), 3);
    }

    #[test]
    fn test_clahe_zero_grid_error() {
        let mut pixels = vec![128u8, 64, 200];
        assert!(clahe(&mut pixels, 1, 1, 0, 40).is_err());
    }

    // ── rgb_to_hsl / hsl_to_rgb round-trips ────────────────────────────────

    fn round_trip(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
        let (h, s, l) = rgb_to_hsl(r, g, b);
        hsl_to_rgb(h, s, l)
    }

    #[test]
    fn test_rgb_hsl_roundtrip_red() {
        let (r, g, b) = round_trip(255, 0, 0);
        assert!(
            (r as i32 - 255).abs() <= 2 && g <= 2 && b <= 2,
            "got {} {} {}",
            r,
            g,
            b
        );
    }

    #[test]
    fn test_rgb_hsl_roundtrip_green() {
        let (r, g, b) = round_trip(0, 255, 0);
        assert!(
            r <= 2 && (g as i32 - 255).abs() <= 2 && b <= 2,
            "got {} {} {}",
            r,
            g,
            b
        );
    }

    #[test]
    fn test_rgb_hsl_roundtrip_blue() {
        let (r, g, b) = round_trip(0, 0, 255);
        assert!(
            r <= 2 && g <= 2 && (b as i32 - 255).abs() <= 2,
            "got {} {} {}",
            r,
            g,
            b
        );
    }

    #[test]
    fn test_rgb_hsl_roundtrip_white() {
        let (r, g, b) = round_trip(255, 255, 255);
        assert_eq!((r, g, b), (255, 255, 255));
    }

    #[test]
    fn test_rgb_hsl_roundtrip_black() {
        let (r, g, b) = round_trip(0, 0, 0);
        assert_eq!((r, g, b), (0, 0, 0));
    }

    #[test]
    fn test_rgb_hsl_roundtrip_gray() {
        let (r, g, b) = round_trip(128, 128, 128);
        assert!(
            (r as i32 - 128).abs() <= 2
                && (g as i32 - 128).abs() <= 2
                && (b as i32 - 128).abs() <= 2,
            "got {} {} {}",
            r,
            g,
            b
        );
    }

    #[test]
    fn test_hsl_hue_red() {
        let (h, s, _l) = rgb_to_hsl(255, 0, 0);
        assert!(
            !(5.0..=355.0).contains(&h),
            "red hue should be ~0, got {}",
            h
        );
        assert!(s > 0.9, "red saturation should be ~1, got {}", s);
    }

    #[test]
    fn test_hsl_hue_green() {
        let (h, _s, _l) = rgb_to_hsl(0, 255, 0);
        assert!(
            (h - 120.0).abs() < 5.0,
            "green hue should be ~120, got {}",
            h
        );
    }

    #[test]
    fn test_hsl_hue_blue() {
        let (h, _s, _l) = rgb_to_hsl(0, 0, 255);
        assert!(
            (h - 240.0).abs() < 5.0,
            "blue hue should be ~240, got {}",
            h
        );
    }

    // ── scene_key ───────────────────────────────────────────────────────────

    #[test]
    fn test_scene_key_all_white() {
        // All white: L=1.0 → log(1e-6+1) ≈ 0 → key ≈ exp(~0) ≈ 1.0
        let pixels = vec![255u8; 3 * 4];
        let key = scene_key(&pixels, 1, 4).unwrap();
        assert!(key > 0.5, "expected near 1.0 got {}", key);
    }

    #[test]
    fn test_scene_key_all_black() {
        // All black: L=0 → log(1e-6) ≈ -13.8 → key ≈ exp(-13.8) ≈ 1e-6
        let pixels = vec![0u8; 3 * 4];
        let key = scene_key(&pixels, 1, 4).unwrap();
        assert!(key < 0.01, "expected near 0 got {}", key);
    }

    #[test]
    fn test_scene_key_gray() {
        // Mid-gray (128,128,128) → L ≈ 0.502
        let pixels = vec![128u8; 3 * 100];
        let key = scene_key(&pixels, 1, 100).unwrap();
        // Expected: exp(log(1e-6 + 0.502)) ≈ 0.502
        assert!((key - 0.502).abs() < 0.01, "got {}", key);
    }

    // ── exposure_bracket ────────────────────────────────────────────────────

    #[test]
    fn test_exposure_bracket_count() {
        let pixels = vec![128u8; 3 * 4];
        let brackets = exposure_bracket(&pixels, 1, 4, 0.0, 1.0, 3).unwrap();
        assert_eq!(brackets.len(), 3);
    }

    #[test]
    fn test_exposure_bracket_zero_n() {
        let pixels = vec![128u8; 3 * 4];
        let brackets = exposure_bracket(&pixels, 1, 4, 0.0, 1.0, 0).unwrap();
        assert_eq!(brackets.len(), 0);
    }

    #[test]
    fn test_exposure_bracket_increasing_brightness() {
        // For 3 brackets around EV=0, step=1: EVs are -1, 0, +1
        // So brackets[0] < brackets[1] < brackets[2] (in average brightness)
        let pixels = vec![64u8; 3 * 10];
        let brackets = exposure_bracket(&pixels, 1, 10, 0.0, 1.0, 3).unwrap();
        let mean = |buf: &[u8]| buf.iter().map(|&v| v as f64).sum::<f64>() / buf.len() as f64;
        let m0 = mean(&brackets[0]);
        let m1 = mean(&brackets[1]);
        let m2 = mean(&brackets[2]);
        assert!(
            m0 < m1,
            "bracket[0] ({}) should be darker than bracket[1] ({})",
            m0,
            m1
        );
        assert!(
            m1 < m2,
            "bracket[1] ({}) should be darker than bracket[2] ({})",
            m1,
            m2
        );
    }

    #[test]
    fn test_exposure_bracket_output_size_matches_input() {
        let pixels = vec![100u8; 3 * 5];
        let brackets = exposure_bracket(&pixels, 1, 5, 0.0, 0.5, 5).unwrap();
        for (i, b) in brackets.iter().enumerate() {
            assert_eq!(b.len(), pixels.len(), "bracket {} size mismatch", i);
        }
    }
}
