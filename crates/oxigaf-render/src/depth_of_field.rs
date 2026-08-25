//! Physically-based depth-of-field (bokeh) post-processing for f32 RGB images.
//!
//! Implements a CPU-side gather-based depth-of-field effect using per-pixel
//! circle-of-confusion (CoC) radii computed from the thin-lens formula.
//!
//! # Algorithm overview
//!
//! 1. Compute per-pixel CoC radius via [`dof_compute_coc`].
//! 2. Generate stochastic bokeh kernel offsets via [`dof_make_kernel`].
//! 3. Gather-blur each pixel by sampling its neighbourhood scaled by CoC.
//! 4. Blend blurred image with original using `strength`.
//!
//! Images are flat `Vec<f32>` in row-major HWC layout (3 channels, RGB).
//! Depth buffers are `Vec<f32>` with one value per pixel.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the physical depth-of-field operations.
#[derive(Debug, Error)]
pub enum DofError {
    /// Buffer or image dimensions don't match expected values.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Focal length must be positive.
    #[error("invalid focal length: must be > 0, got {0}")]
    InvalidFocalLength(f32),

    /// Focus distance must be positive.
    #[error("invalid focus distance: must be > 0, got {0}")]
    InvalidFocusDistance(f32),

    /// Aperture (f-stop) must be positive.
    #[error("invalid aperture: must be > 0, got {0}")]
    InvalidAperture(f32),

    /// Configuration value is invalid.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// Input buffer is empty.
    #[error("empty input")]
    EmptyInput,
}

// ─────────────────────────────────────────────────────────────────────────────
// BokehShape
// ─────────────────────────────────────────────────────────────────────────────

/// Aperture shape that controls the bokeh kernel profile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BokehShape {
    /// Smooth circular disc (ideal thin lens).
    Circular,
    /// Six-sided polygon (vintage lens with 6 aperture blades).
    Hexagonal,
    /// Five-sided polygon (5-blade aperture).
    Pentagonal,
    /// Square aperture (sensor-like, hard-edged bokeh).
    Square,
}

// ─────────────────────────────────────────────────────────────────────────────
// DofConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Physical configuration for the depth-of-field post-processing pass.
///
/// Default values model a classic portrait lens (85 mm / f2.8) on a
/// full-frame sensor, focused at 1 m.
#[derive(Debug, Clone)]
pub struct DofConfig {
    /// Camera focal length in millimetres. Default: 85.0.
    pub focal_length: f32,
    /// Distance to the in-focus plane in world units (assumed to be metres
    /// for the purposes of the thin-lens CoC formula - see the
    /// `MM_PER_WORLD_UNIT` constant in this module). Default: 1.0.
    pub focus_distance: f32,
    /// Aperture f-stop number (f/2.8 → aperture=2.8). Default: 2.8.
    pub aperture: f32,
    /// Physical sensor width in millimetres. Default: 36.0 (full-frame).
    pub sensor_width_mm: f32,
    /// Maximum circle-of-confusion radius in pixels. Default: 20.0.
    pub max_coc_pixels: f32,
    /// Number of gather samples per pixel. Default: 32.
    pub n_samples: usize,
    /// Shape of the bokeh aperture. Default: Circular.
    pub bokeh_shape: BokehShape,
    /// Whether to blur pixels in front of the focal plane. Default: true.
    ///
    /// When `false`, every pixel with `depth < focus_distance` is forced to
    /// `CoC = 0` by [`dof_compute_coc`] (so it is never itself blurred) and
    /// [`dof_gather`] never lets it bleed into neighbouring pixels either.
    pub near_blur: bool,
    /// Overall DoF strength in [0, 1]. Default: 1.0.
    pub strength: f32,
}

impl Default for DofConfig {
    fn default() -> Self {
        Self {
            focal_length: 85.0,
            focus_distance: 1.0,
            aperture: 2.8,
            sensor_width_mm: 36.0,
            max_coc_pixels: 20.0,
            n_samples: 32,
            bokeh_shape: BokehShape::Circular,
            near_blur: true,
            strength: 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DofCocBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// Per-pixel circle-of-confusion data.
#[derive(Debug, Clone)]
pub struct DofCocBuffer {
    /// Absolute CoC radius in pixels for every pixel (row-major, length = W×H).
    pub coc: Vec<f32>,
    /// `true` when the pixel is in front of the focal plane.
    pub is_near: Vec<bool>,
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// DofStats / DofResult
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics from a depth-of-field pass.
#[derive(Debug, Clone)]
pub struct DofStats {
    /// Mean CoC radius across all pixels.
    pub mean_coc: f32,
    /// Maximum CoC radius.
    pub max_coc: f32,
    /// Fraction of pixels whose CoC is below 1 pixel (in-focus region).
    pub in_focus_fraction: f32,
    /// Fraction of near-plane (in front of focus) pixels.
    pub near_fraction: f32,
    /// Fraction of far-plane (behind focus) pixels.
    pub far_fraction: f32,
}

/// Output of the full depth-of-field pipeline.
#[derive(Debug, Clone)]
pub struct DofResult {
    /// DoF-blurred RGB image (flat f32, length = W×H×3).
    pub image: Vec<f32>,
    /// CoC radius per pixel for visualisation (length = W×H).
    pub coc_map: Vec<f32>,
    /// Statistics for this pass.
    pub stats: DofStats,
}

// ─────────────────────────────────────────────────────────────────────────────
// PRNG helpers (xorshift64 — no rand crate)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

#[inline]
fn xorshift_f32(state: &mut u64) -> f32 {
    xorshift64(state) as f32 / u64::MAX as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// CoC computation
// ─────────────────────────────────────────────────────────────────────────────

/// Millimetres per world unit.
///
/// `DofConfig::focal_length` and `DofConfig::sensor_width_mm` are
/// millimetres, but `DofConfig::focus_distance` and the incoming depth
/// buffer are in "world units" - the scene-space units the rest of the
/// crate (and a typical COLMAP/NeRF/3DGS scene) uses. This constant assumes
/// the standard 3-D graphics convention that one world unit is one metre,
/// and is used to convert world-unit distances into millimetres so the
/// thin-lens formula in [`dof_compute_coc`] is dimensionally consistent.
const MM_PER_WORLD_UNIT: f32 = 1000.0;

/// Compute per-pixel circle-of-confusion radii from a depth buffer.
///
/// Uses the thin-lens approximation, with all lengths converted to a common
/// millimetre scale (see the `MM_PER_WORLD_UNIT` constant in this module)
/// before combining:
/// ```text
/// S_mm  = focus_distance × MM_PER_WORLD_UNIT
/// CoC_mm ≈ (focal_length² / aperture)
///          × |depth − focus_distance| / (depth × (S_mm − focal_length))
/// CoC_pixels = CoC_mm × image_width / sensor_width_mm
/// ```
///
/// Background pixels (`depth == 0` or infinite) receive `max_coc_pixels`.
/// The result is clamped to `[0, max_coc_pixels]`.
///
/// # Errors
///
/// - [`DofError::InvalidFocalLength`] when `focal_length ≤ 0`.
/// - [`DofError::InvalidFocusDistance`] when `focus_distance ≤ 0`.
/// - [`DofError::InvalidAperture`] when `aperture ≤ 0`.
/// - [`DofError::DimensionMismatch`] when `depth_buf.len() ≠ width × height`.
/// - [`DofError::EmptyInput`] when `depth_buf` is empty.
pub fn dof_compute_coc(
    depth_buf: &[f32],
    width: usize,
    height: usize,
    config: &DofConfig,
) -> Result<DofCocBuffer, DofError> {
    if config.focal_length <= 0.0 || !config.focal_length.is_finite() {
        return Err(DofError::InvalidFocalLength(config.focal_length));
    }
    if config.focus_distance <= 0.0 || !config.focus_distance.is_finite() {
        return Err(DofError::InvalidFocusDistance(config.focus_distance));
    }
    if config.aperture <= 0.0 || !config.aperture.is_finite() {
        return Err(DofError::InvalidAperture(config.aperture));
    }
    if depth_buf.is_empty() {
        return Err(DofError::EmptyInput);
    }
    let expected = width * height;
    if depth_buf.len() != expected {
        return Err(DofError::DimensionMismatch {
            expected,
            got: depth_buf.len(),
        });
    }

    // Thin-lens formula, in a common millimetre scale (see
    // `MM_PER_WORLD_UNIT`):
    //   CoC_mm = (f_mm^2 / N) * |D - S| / (D * (S_mm - f_mm))
    // `focal_length` (f_mm) is already millimetres; `focus_distance` (S) and
    // every depth sample (D) are world units and only `S` needs converting
    // to millimetres (`fd_mm`) since `f_mm` and `S_mm` must share a scale
    // for `(S_mm - f_mm)` to be meaningful, while `D` and `S` may stay in
    // world units in the `|D - S| / D` factor (a dimensionless ratio once
    // paired with a `D` of the same unit). The previous `fl*fl/(ap*fd)`
    // scale mixed millimetres directly with a raw world-unit `fd` and
    // omitted the `(S - f)` term entirely, giving a scale off by orders of
    // magnitude for the module's own documented defaults (85mm/f2.8/1
    // world-unit focus → scale ≈ 2580 "mm" instead of a physically
    // sensible value), which clamped to `max_coc_pixels` almost everywhere.
    let fl = config.focal_length;
    let fd = config.focus_distance;
    let fd_mm = fd * MM_PER_WORLD_UNIT;
    let ap = config.aperture;
    let thin_lens_scale = (fl * fl) / ap;
    let s_minus_f_mm = fd_mm - fl;

    // mm → pixels conversion factor.
    let mm_to_px = if config.sensor_width_mm > 0.0 {
        width as f32 / config.sensor_width_mm
    } else {
        1.0
    };

    let max_r = config.max_coc_pixels;

    let mut coc = Vec::with_capacity(expected);
    let mut is_near = Vec::with_capacity(expected);

    for &d in depth_buf {
        if !d.is_finite() || d <= 0.0 {
            coc.push(max_r);
            is_near.push(false);
            continue;
        }

        let is_near_pixel = d < fd;
        if is_near_pixel && !config.near_blur {
            // Near-blur disabled: foreground pixels are always treated as
            // perfectly in focus so they are never themselves blurred, and
            // (via `is_near` + `coc == 0`) never satisfy `dof_gather`'s
            // acceptance test either, so they cannot bleed into neighbours.
            coc.push(0.0);
            is_near.push(true);
            continue;
        }

        let diff = (d - fd).abs();
        // `.abs()` guards the non-physical `focus_distance <= focal_length`
        // regime (a focus distance shorter than the lens' own focal length,
        // in millimetres) where `s_minus_f_mm` goes negative: without it, a
        // background pixel there would compute a negative CoC that then
        // clamps *down* to 0 (wrongly reporting "in focus") instead of
        // clamping up to `max_r` ("very blurred").
        let coc_mm = (thin_lens_scale * diff / (d * s_minus_f_mm)).abs();
        let coc_px = (coc_mm * mm_to_px).clamp(0.0, max_r);

        coc.push(coc_px);
        is_near.push(is_near_pixel);
    }

    Ok(DofCocBuffer {
        coc,
        is_near,
        width,
        height,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Bokeh kernel generators (stochastic, sample points in [-1,1] disk space)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate `n` sample points inside a unit circular disk.
///
/// Returns `n * 2` values (`x₀, y₀, x₁, y₁, …`) in `[-1, 1] × [-1, 1]`
/// with `x² + y² ≤ 1`. Uses rejection sampling.
///
/// `n == 0` returns an empty `Vec`.
pub fn dof_circular_kernel(n: usize, rng_state: &mut u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * 2);
    let mut accepted = 0usize;
    while accepted < n {
        let x = xorshift_f32(rng_state) * 2.0 - 1.0;
        let y = xorshift_f32(rng_state) * 2.0 - 1.0;
        if x * x + y * y <= 1.0 {
            out.push(x);
            out.push(y);
            accepted += 1;
        }
    }
    out
}

/// Generate `n` sample points inside a unit regular hexagon.
///
/// Returns `n * 2` values (`x₀, y₀, …`) in `[-1, 1] × [-1, 1]`.
/// Uses polar coordinates with per-angle hexagonal radial limit.
pub fn dof_hexagonal_kernel(n: usize, rng_state: &mut u64) -> Vec<f32> {
    const TWO_PI: f32 = 2.0 * std::f32::consts::PI;
    const SIDES: f32 = 6.0;
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let angle = xorshift_f32(rng_state) * TWO_PI;
        // Radial limit for a regular n-gon at angle θ:
        // r_max(θ) = cos(π/n) / cos(θ mod (2π/n) − π/n)
        let sector_angle = TWO_PI / SIDES;
        let half_sector = sector_angle / 2.0;
        let theta_mod = ((angle % sector_angle) + sector_angle) % sector_angle - half_sector;
        let r_max = half_sector.cos() / theta_mod.cos().max(1e-6);
        // Area-uniform radial sampling: a uniform `r` in `[0, r_max)` biases
        // samples toward the centre (density ∝ 1/r for a fixed angular
        // slice), washing out the hexagon's edges into a soft radial
        // falloff indistinguishable from a circle. `sqrt(u) * r_max` makes
        // the sample density uniform over the polygon's area instead.
        let r = r_max * xorshift_f32(rng_state).sqrt();
        out.push(r * angle.cos());
        out.push(r * angle.sin());
    }
    out
}

/// Generate `n` sample points inside a unit regular pentagon.
///
/// Returns `n * 2` values in `[-1, 1] × [-1, 1]`.
pub fn dof_pentagonal_kernel(n: usize, rng_state: &mut u64) -> Vec<f32> {
    const TWO_PI: f32 = 2.0 * std::f32::consts::PI;
    const SIDES: f32 = 5.0;
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let angle = xorshift_f32(rng_state) * TWO_PI;
        let sector_angle = TWO_PI / SIDES;
        let half_sector = sector_angle / 2.0;
        let theta_mod = ((angle % sector_angle) + sector_angle) % sector_angle - half_sector;
        let r_max = half_sector.cos() / theta_mod.cos().max(1e-6);
        // Area-uniform radial sampling - see `dof_hexagonal_kernel` above.
        let r = r_max * xorshift_f32(rng_state).sqrt();
        out.push(r * angle.cos());
        out.push(r * angle.sin());
    }
    out
}

/// Generate `n` sample points inside a unit square `[-1, 1] × [-1, 1]`.
///
/// Returns `n * 2` values.
pub fn dof_square_kernel(n: usize, rng_state: &mut u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let x = xorshift_f32(rng_state) * 2.0 - 1.0;
        let y = xorshift_f32(rng_state) * 2.0 - 1.0;
        out.push(x);
        out.push(y);
    }
    out
}

/// Dispatch to the kernel generator for a given [`BokehShape`].
pub fn dof_make_kernel(shape: &BokehShape, n: usize, rng_state: &mut u64) -> Vec<f32> {
    match shape {
        BokehShape::Circular => dof_circular_kernel(n, rng_state),
        BokehShape::Hexagonal => dof_hexagonal_kernel(n, rng_state),
        BokehShape::Pentagonal => dof_pentagonal_kernel(n, rng_state),
        BokehShape::Square => dof_square_kernel(n, rng_state),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bilinear sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Sample an RGB image at fractional pixel coordinates using bilinear interpolation.
///
/// Coordinates outside `[0, width-1] × [0, height-1]` are clamped to the border.
/// `image` must have length `width × height × 3` (row-major RGB).
pub fn dof_bilinear_sample(image: &[f32], width: usize, height: usize, x: f32, y: f32) -> [f32; 3] {
    if image.is_empty() || width == 0 || height == 0 {
        return [0.0; 3];
    }

    let x_clamped = x.clamp(0.0, (width as f32) - 1.0);
    let y_clamped = y.clamp(0.0, (height as f32) - 1.0);

    let x0 = x_clamped.floor() as usize;
    let y0 = y_clamped.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let tx = x_clamped - x0 as f32;
    let ty = y_clamped - y0 as f32;

    let idx = |row: usize, col: usize| (row * width + col) * 3;

    let i00 = idx(y0, x0);
    let i10 = idx(y1, x0);
    let i01 = idx(y0, x1);
    let i11 = idx(y1, x1);

    let mut result = [0.0_f32; 3];
    for c in 0..3 {
        let v00 = image[i00 + c];
        let v10 = image[i10 + c];
        let v01 = image[i01 + c];
        let v11 = image[i11 + c];
        let top = v00 + (v01 - v00) * tx;
        let bot = v10 + (v11 - v10) * tx;
        result[c] = top + (bot - top) * ty;
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Gather-based DoF blur
// ─────────────────────────────────────────────────────────────────────────────

/// Gather-based depth-of-field blur.
///
/// For each output pixel `(x, y)`:
/// - If `CoC < 1 px` → copy pixel unchanged (in focus).
/// - Otherwise: sample `n_samples` neighbours in the CoC disk using
///   precomputed kernel offsets. A sample from `(nx, ny)` contributes when
///   either:
///   - its own CoC reaches the current pixel (`neighbour_coc ≥ dist`), which
///     alone already covers a genuinely defocused near-plane neighbour
///     (its CoC grows with distance from the focal plane exactly like a far
///     neighbour's does); or
///   - `config.near_blur` is enabled, the neighbour is on the near side of
///     the focal plane, **and** it is itself already blurred
///     (`neighbour_coc ≥ 1px`) - letting a defocused foreground object
///     bleed over the background unconditionally within the gather disc,
///     the classic foreground-bokeh look. A near-plane neighbour that is
///     itself sharp never gets this relaxation, so in-focus foreground
///     content cannot smear into neighbouring blurred pixels, and disabling
///     `near_blur` removes the relaxation entirely.
///
///   Weight = neighbour CoC radius (larger blur → more weight).
///
/// # Errors
///
/// - [`DofError::DimensionMismatch`] when buffer sizes are inconsistent.
pub fn dof_gather(
    image: &[f32],
    coc_buf: &DofCocBuffer,
    kernel: &[f32],
    config: &DofConfig,
) -> Result<Vec<f32>, DofError> {
    let n_pixels = coc_buf.width * coc_buf.height;
    let expected_img = n_pixels * 3;
    if image.len() != expected_img {
        return Err(DofError::DimensionMismatch {
            expected: expected_img,
            got: image.len(),
        });
    }
    if coc_buf.coc.len() != n_pixels {
        return Err(DofError::DimensionMismatch {
            expected: n_pixels,
            got: coc_buf.coc.len(),
        });
    }

    let w = coc_buf.width;
    let h = coc_buf.height;
    let n_samples = if kernel.is_empty() {
        0
    } else {
        kernel.len() / 2
    };

    let mut output = vec![0.0_f32; expected_img];

    for py in 0..h {
        for px in 0..w {
            let pidx = py * w + px;
            let coc_r = coc_buf.coc[pidx];

            // Sharp pixel — copy as-is.
            if coc_r < 1.0 || n_samples == 0 {
                let base = pidx * 3;
                output[base] = image[base];
                output[base + 1] = image[base + 1];
                output[base + 2] = image[base + 2];
                continue;
            }

            let pxf = px as f32;
            let pyf = py as f32;

            let mut acc = [0.0_f32; 3];
            let mut weight_sum = 0.0_f32;

            for s in 0..n_samples {
                let kx = kernel[s * 2];
                let ky = kernel[s * 2 + 1];

                let sx = pxf + kx * coc_r;
                let sy = pyf + ky * coc_r;

                // Clamp sample coords.
                let sx_c = sx.clamp(0.0, (w as f32) - 1.0);
                let sy_c = sy.clamp(0.0, (h as f32) - 1.0);

                // Nearest integer for CoC lookup (bilinear for colour).
                let nx = sx_c.round() as usize;
                let ny = sy_c.round() as usize;
                let nx = nx.min(w - 1);
                let ny = ny.min(h - 1);
                let nidx = ny * w + nx;
                let n_coc = coc_buf.coc[nidx];
                let n_is_near = coc_buf.is_near[nidx];

                // Acceptance: neighbour's own CoC must reach the current
                // pixel (its CoC ≥ distance from centre) - this alone
                // already lets a genuinely defocused near-plane neighbour
                // bleed outward, since its CoC grows with distance from the
                // focal plane the same way a far neighbour's does. The only
                // extra relaxation is for a near-plane neighbour that is
                // *itself already blurred* (`n_coc >= 1px`, matching the
                // sharp/blurred threshold used elsewhere in this module),
                // when `near_blur` is enabled: it may bleed unconditionally
                // within the gather disc regardless of exact distance,
                // matching classic foreground bokeh. A near-plane neighbour
                // that is still sharp (`n_coc < 1px`, e.g. barely in front
                // of the focal plane) never gets this relaxation - admitting
                // it unconditionally would smear in-focus foreground detail
                // into neighbouring blurred pixels.
                let dist = ((sx - pxf).powi(2) + (sy - pyf).powi(2)).sqrt();
                let near_bleed = config.near_blur && n_is_near && n_coc >= 1.0;
                if n_coc >= dist || near_bleed {
                    let weight = n_coc.max(1.0);
                    let colour = dof_bilinear_sample(image, w, h, sx_c, sy_c);
                    acc[0] += colour[0] * weight;
                    acc[1] += colour[1] * weight;
                    acc[2] += colour[2] * weight;
                    weight_sum += weight;
                }
            }

            let base = pidx * 3;
            if weight_sum > 0.0 {
                let inv = 1.0 / weight_sum;
                output[base] = acc[0] * inv;
                output[base + 1] = acc[1] * inv;
                output[base + 2] = acc[2] * inv;
            } else {
                // No sample accepted — use centre pixel.
                output[base] = image[base];
                output[base + 1] = image[base + 1];
                output[base + 2] = image[base + 2];
            }
        }
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer separation and compositing
// ─────────────────────────────────────────────────────────────────────────────

/// Separate an image into near-field and far-field layers.
///
/// Returns `(near_image, far_image)`, each of length `width × height × 3`.
/// Near pixels are in `near_image`; far pixels are in `far_image`.
/// The opposite layer is zeroed.
///
/// # Errors
///
/// - [`DofError::DimensionMismatch`] when buffer sizes are inconsistent.
pub fn dof_separate_layers(
    image: &[f32],
    coc_buf: &DofCocBuffer,
) -> Result<(Vec<f32>, Vec<f32>), DofError> {
    let n_pixels = coc_buf.width * coc_buf.height;
    let expected = n_pixels * 3;
    if image.len() != expected {
        return Err(DofError::DimensionMismatch {
            expected,
            got: image.len(),
        });
    }

    let mut near = vec![0.0_f32; expected];
    let mut far = vec![0.0_f32; expected];

    for i in 0..n_pixels {
        let base = i * 3;
        if coc_buf.is_near[i] {
            near[base] = image[base];
            near[base + 1] = image[base + 1];
            near[base + 2] = image[base + 2];
        } else {
            far[base] = image[base];
            far[base + 1] = image[base + 1];
            far[base + 2] = image[base + 2];
        }
    }

    Ok((near, far))
}

/// Composite near-blurred, far-blurred, and in-focus layers.
///
/// For each pixel:
/// - `focus_weight = clamp(1 - coc[pixel], 0, 1)` - `1.0` for a perfectly
///   sharp pixel (`coc == 0`), fading linearly to `0.0` once its CoC reaches
///   the 1px sharp/blurred threshold used throughout this module.
/// - The defocused component is `near_blurred[pixel]` when the pixel is on
///   the near side of the focal plane, otherwise `far_blurred[pixel]`.
/// - `result = lerp(defocused_component, in_focus[pixel], focus_weight)`.
///
/// All input buffers must have length `width × height × 3`.
///
/// # Errors
///
/// - [`DofError::DimensionMismatch`] when any buffer size is inconsistent.
pub fn dof_composite_layers(
    near_blurred: &[f32],
    far_blurred: &[f32],
    in_focus: &[f32],
    coc_buf: &DofCocBuffer,
) -> Result<Vec<f32>, DofError> {
    let n_pixels = coc_buf.width * coc_buf.height;
    let expected = n_pixels * 3;

    for (name, buf) in [
        ("near_blurred", near_blurred),
        ("far_blurred", far_blurred),
        ("in_focus", in_focus),
    ] {
        if buf.len() != expected {
            return Err(DofError::DimensionMismatch {
                expected,
                got: buf.len(),
            });
        }
        let _ = name;
    }

    let mut result = vec![0.0_f32; expected];

    for i in 0..n_pixels {
        let base = i * 3;
        let focus_weight = (1.0 - coc_buf.coc[i]).clamp(0.0, 1.0);
        let defocused = if coc_buf.is_near[i] {
            near_blurred
        } else {
            far_blurred
        };

        // lerp(defocused, in_focus, focus_weight)
        for c in 0..3 {
            result[base + c] =
                defocused[base + c] + (in_focus[base + c] - defocused[base + c]) * focus_weight;
        }
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Compute depth-of-field statistics from a [`DofCocBuffer`].
pub fn dof_compute_stats(coc_buf: &DofCocBuffer) -> DofStats {
    let n = coc_buf.coc.len();
    if n == 0 {
        return DofStats {
            mean_coc: 0.0,
            max_coc: 0.0,
            in_focus_fraction: 1.0,
            near_fraction: 0.0,
            far_fraction: 0.0,
        };
    }

    let mut sum = 0.0_f32;
    let mut max_coc = 0.0_f32;
    let mut in_focus_count = 0usize;
    let mut near_count = 0usize;

    for (i, &c) in coc_buf.coc.iter().enumerate() {
        sum += c;
        if c > max_coc {
            max_coc = c;
        }
        if c < 1.0 {
            in_focus_count += 1;
        } else if coc_buf.is_near[i] {
            near_count += 1;
        }
    }

    let nf = n as f32;
    let far_blurred = n - in_focus_count - near_count;

    DofStats {
        mean_coc: sum / nf,
        max_coc,
        in_focus_fraction: in_focus_count as f32 / nf,
        near_fraction: near_count as f32 / nf,
        far_fraction: far_blurred as f32 / nf,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Format the DoF configuration into a human-readable string.
pub fn dof_format_config(config: &DofConfig) -> String {
    format!(
        "DofConfig {{ focal_length: {:.1}mm, focus_distance: {:.2}, aperture: f/{:.1}, \
         sensor: {:.0}mm, max_coc: {:.0}px, n_samples: {}, shape: {:?}, near_blur: {}, strength: {:.2} }}",
        config.focal_length,
        config.focus_distance,
        config.aperture,
        config.sensor_width_mm,
        config.max_coc_pixels,
        config.n_samples,
        config.bokeh_shape,
        config.near_blur,
        config.strength,
    )
}

/// Format depth-of-field statistics into a human-readable string.
pub fn dof_format_stats(stats: &DofStats) -> String {
    format!(
        "DofStats {{ mean_coc: {:.2}px, max_coc: {:.2}px, in_focus: {:.1}%, near: {:.1}%, far: {:.1}% }}",
        stats.mean_coc,
        stats.max_coc,
        stats.in_focus_fraction * 100.0,
        stats.near_fraction * 100.0,
        stats.far_fraction * 100.0,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Full DoF pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the full depth-of-field pipeline to an f32 RGB image.
///
/// Steps:
/// 1. Validate config and compute CoC buffer.
/// 2. Generate bokeh kernel (stochastic, seeded deterministically).
/// 3. Gather-blur the image.
/// 4. Blend with `strength` → `result = lerp(original, blurred, strength)`.
/// 5. Compute and return statistics.
///
/// `image` must have length `width × height × 3` (row-major RGB, f32).
/// `depth_buf` must have length `width × height`.
///
/// # Errors
///
/// - Any [`DofError`] variant for invalid config or mismatched dimensions.
pub fn apply_depth_of_field(
    image: &[f32],
    depth_buf: &[f32],
    width: usize,
    height: usize,
    config: &DofConfig,
) -> Result<DofResult, DofError> {
    // Validate config up-front.
    if config.focal_length <= 0.0 || !config.focal_length.is_finite() {
        return Err(DofError::InvalidFocalLength(config.focal_length));
    }
    if config.focus_distance <= 0.0 || !config.focus_distance.is_finite() {
        return Err(DofError::InvalidFocusDistance(config.focus_distance));
    }
    if config.aperture <= 0.0 || !config.aperture.is_finite() {
        return Err(DofError::InvalidAperture(config.aperture));
    }
    if config.strength < 0.0 || config.strength > 1.0 || !config.strength.is_finite() {
        return Err(DofError::InvalidConfig(format!(
            "strength must be in [0, 1], got {}",
            config.strength
        )));
    }

    let n_pixels = width * height;
    let expected_img = n_pixels * 3;

    if image.len() != expected_img {
        return Err(DofError::DimensionMismatch {
            expected: expected_img,
            got: image.len(),
        });
    }
    if depth_buf.len() != n_pixels {
        return Err(DofError::DimensionMismatch {
            expected: n_pixels,
            got: depth_buf.len(),
        });
    }

    // Step 1: Compute CoC.
    let coc_buf = dof_compute_coc(depth_buf, width, height, config)?;
    let coc_map = coc_buf.coc.clone();

    // Step 2: Generate kernel (seed = 0xdeadbeef for determinism).
    let mut rng = 0xdeadbeef_u64;
    let kernel = dof_make_kernel(&config.bokeh_shape, config.n_samples, &mut rng);

    // Step 3: Gather blur.
    let blurred = dof_gather(image, &coc_buf, &kernel, config)?;

    // Step 4: Blend with strength.
    let result_image = if (config.strength - 1.0).abs() < 1e-6 {
        blurred
    } else if config.strength < 1e-6 {
        image.to_vec()
    } else {
        let s = config.strength;
        let mut out = vec![0.0_f32; expected_img];
        for i in 0..expected_img {
            out[i] = image[i] + (blurred[i] - image[i]) * s;
        }
        out
    };

    // Step 5: Statistics.
    let stats = dof_compute_stats(&coc_buf);

    Ok(DofResult {
        image: result_image,
        coc_map,
        stats,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    /// Build a solid-colour RGB f32 image.
    fn solid_rgb(w: usize, h: usize, r: f32, g: f32, b: f32) -> Vec<f32> {
        let mut buf = Vec::with_capacity(w * h * 3);
        for _ in 0..w * h {
            buf.push(r);
            buf.push(g);
            buf.push(b);
        }
        buf
    }

    fn default_cfg() -> DofConfig {
        DofConfig::default()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DofConfig::default
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_dof_config_default_values() {
        let cfg = DofConfig::default();
        assert!(approx_eq(cfg.focal_length, 85.0, 1e-5));
        assert!(approx_eq(cfg.focus_distance, 1.0, 1e-5));
        assert!(approx_eq(cfg.aperture, 2.8, 1e-5));
        assert!(approx_eq(cfg.sensor_width_mm, 36.0, 1e-5));
        assert!(approx_eq(cfg.max_coc_pixels, 20.0, 1e-5));
        assert_eq!(cfg.n_samples, 32);
        assert_eq!(cfg.bokeh_shape, BokehShape::Circular);
        assert!(cfg.near_blur);
        assert!(approx_eq(cfg.strength, 1.0, 1e-5));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // dof_compute_coc
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_coc_background_gets_max() {
        let cfg = default_cfg();
        let depth = vec![0.0_f32; 4]; // background
        let buf = dof_compute_coc(&depth, 2, 2, &cfg).expect("coc ok");
        for &c in &buf.coc {
            assert!(approx_eq(c, cfg.max_coc_pixels, 1e-4), "got {c}");
        }
    }

    #[test]
    fn test_coc_at_focus_is_zero() {
        let cfg = DofConfig {
            focus_distance: 2.0,
            ..DofConfig::default()
        };
        let depth = vec![2.0_f32; 6];
        let buf = dof_compute_coc(&depth, 3, 2, &cfg).expect("coc ok");
        for &c in &buf.coc {
            assert!(approx_eq(c, 0.0, 1e-4), "CoC at focus should be 0, got {c}");
        }
    }

    #[test]
    fn test_coc_near_is_near() {
        let cfg = DofConfig {
            focus_distance: 2.0,
            ..DofConfig::default()
        };
        let depth = vec![0.5_f32; 4]; // in front of focus
        let buf = dof_compute_coc(&depth, 2, 2, &cfg).expect("coc ok");
        for &near in &buf.is_near {
            assert!(near, "depth < focus_distance must be near");
        }
    }

    #[test]
    fn test_coc_far_is_not_near() {
        let cfg = DofConfig {
            focus_distance: 1.0,
            ..DofConfig::default()
        };
        let depth = vec![5.0_f32; 4]; // behind focus
        let buf = dof_compute_coc(&depth, 2, 2, &cfg).expect("coc ok");
        for &near in &buf.is_near {
            assert!(!near, "depth > focus_distance must not be near");
        }
    }

    #[test]
    fn test_coc_invalid_focal_length() {
        let cfg = DofConfig {
            focal_length: 0.0,
            ..DofConfig::default()
        };
        let depth = vec![1.0_f32; 4];
        assert!(matches!(
            dof_compute_coc(&depth, 2, 2, &cfg),
            Err(DofError::InvalidFocalLength(_))
        ));
    }

    #[test]
    fn test_coc_negative_focal_length() {
        let cfg = DofConfig {
            focal_length: -10.0,
            ..DofConfig::default()
        };
        let depth = vec![1.0_f32; 4];
        assert!(matches!(
            dof_compute_coc(&depth, 2, 2, &cfg),
            Err(DofError::InvalidFocalLength(_))
        ));
    }

    #[test]
    fn test_coc_invalid_aperture() {
        let cfg = DofConfig {
            aperture: 0.0,
            ..DofConfig::default()
        };
        let depth = vec![1.0_f32; 4];
        assert!(matches!(
            dof_compute_coc(&depth, 2, 2, &cfg),
            Err(DofError::InvalidAperture(_))
        ));
    }

    #[test]
    fn test_coc_invalid_focus_distance() {
        let cfg = DofConfig {
            focus_distance: -1.0,
            ..DofConfig::default()
        };
        let depth = vec![1.0_f32; 4];
        assert!(matches!(
            dof_compute_coc(&depth, 2, 2, &cfg),
            Err(DofError::InvalidFocusDistance(_))
        ));
    }

    #[test]
    fn test_coc_dimension_mismatch() {
        let cfg = default_cfg();
        let depth = vec![1.0_f32; 5]; // wrong size for 2×2
        assert!(matches!(
            dof_compute_coc(&depth, 2, 2, &cfg),
            Err(DofError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_coc_empty_input() {
        let cfg = default_cfg();
        assert!(matches!(
            dof_compute_coc(&[], 2, 2, &cfg),
            Err(DofError::EmptyInput)
        ));
    }

    #[test]
    fn test_coc_increases_with_distance() {
        let cfg = DofConfig {
            focus_distance: 2.0,
            ..DofConfig::default()
        };
        let depths = [3.0_f32, 5.0, 10.0, 20.0];
        let mut prev_coc = 0.0_f32;
        for &d in &depths {
            let buf = dof_compute_coc(&[d], 1, 1, &cfg).expect("ok");
            assert!(
                buf.coc[0] >= prev_coc,
                "CoC should increase: depth={d}, coc={}",
                buf.coc[0]
            );
            prev_coc = buf.coc[0];
        }
    }

    #[test]
    fn test_coc_exact_zero_at_focus() {
        let cfg = DofConfig {
            focus_distance: 3.0,
            ..DofConfig::default()
        };
        let buf = dof_compute_coc(&[3.0_f32], 1, 1, &cfg).expect("ok");
        assert!(approx_eq(buf.coc[0], 0.0, 1e-4), "got {}", buf.coc[0]);
    }

    #[test]
    fn test_coc_infinite_depth_gets_max() {
        let cfg = default_cfg();
        let buf = dof_compute_coc(&[f32::INFINITY], 1, 1, &cfg).expect("ok");
        assert!(approx_eq(buf.coc[0], cfg.max_coc_pixels, 1e-4));
    }

    #[test]
    fn test_coc_nan_depth_gets_max() {
        let cfg = default_cfg();
        let buf = dof_compute_coc(&[f32::NAN], 1, 1, &cfg).expect("ok");
        assert!(approx_eq(buf.coc[0], cfg.max_coc_pixels, 1e-4));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Bokeh kernels
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_circular_kernel_length() {
        let mut rng = 42u64;
        let k = dof_circular_kernel(16, &mut rng);
        assert_eq!(k.len(), 32);
    }

    #[test]
    fn test_circular_kernel_inside_unit_circle() {
        let mut rng = 1u64;
        let k = dof_circular_kernel(100, &mut rng);
        for pair in k.chunks_exact(2) {
            let (x, y) = (pair[0], pair[1]);
            assert!(
                x * x + y * y <= 1.0 + 1e-5,
                "point ({x},{y}) outside unit circle"
            );
        }
    }

    #[test]
    fn test_circular_kernel_zero_samples() {
        let mut rng = 1u64;
        let k = dof_circular_kernel(0, &mut rng);
        assert!(k.is_empty());
    }

    #[test]
    fn test_hexagonal_kernel_length() {
        let mut rng = 7u64;
        let k = dof_hexagonal_kernel(20, &mut rng);
        assert_eq!(k.len(), 40);
    }

    #[test]
    fn test_hexagonal_kernel_zero_samples() {
        let mut rng = 1u64;
        let k = dof_hexagonal_kernel(0, &mut rng);
        assert!(k.is_empty());
    }

    #[test]
    fn test_pentagonal_kernel_length() {
        let mut rng = 99u64;
        let k = dof_pentagonal_kernel(12, &mut rng);
        assert_eq!(k.len(), 24);
    }

    #[test]
    fn test_pentagonal_kernel_zero_samples() {
        let mut rng = 1u64;
        let k = dof_pentagonal_kernel(0, &mut rng);
        assert!(k.is_empty());
    }

    #[test]
    fn test_square_kernel_length() {
        let mut rng = 3u64;
        let k = dof_square_kernel(8, &mut rng);
        assert_eq!(k.len(), 16);
    }

    #[test]
    fn test_square_kernel_bounds() {
        let mut rng = 5u64;
        let k = dof_square_kernel(200, &mut rng);
        for &v in &k {
            assert!((-1.0..=1.0).contains(&v), "square sample {v} out of [-1,1]");
        }
    }

    #[test]
    fn test_square_kernel_zero_samples() {
        let mut rng = 1u64;
        let k = dof_square_kernel(0, &mut rng);
        assert!(k.is_empty());
    }

    #[test]
    fn test_make_kernel_circular() {
        let mut rng = 1u64;
        let k = dof_make_kernel(&BokehShape::Circular, 10, &mut rng);
        assert_eq!(k.len(), 20);
    }

    #[test]
    fn test_make_kernel_hexagonal() {
        let mut rng = 2u64;
        let k = dof_make_kernel(&BokehShape::Hexagonal, 10, &mut rng);
        assert_eq!(k.len(), 20);
    }

    #[test]
    fn test_make_kernel_pentagonal() {
        let mut rng = 3u64;
        let k = dof_make_kernel(&BokehShape::Pentagonal, 10, &mut rng);
        assert_eq!(k.len(), 20);
    }

    #[test]
    fn test_make_kernel_square() {
        let mut rng = 4u64;
        let k = dof_make_kernel(&BokehShape::Square, 10, &mut rng);
        assert_eq!(k.len(), 20);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // dof_gather
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_gather_output_length() {
        let cfg = DofConfig {
            focus_distance: 5.0,
            ..DofConfig::default()
        };
        let w = 4;
        let h = 3;
        let image = solid_rgb(w, h, 0.5, 0.5, 0.5);
        let depth = vec![5.0_f32; w * h];
        let coc_buf = dof_compute_coc(&depth, w, h, &cfg).expect("coc");
        let mut rng = 1u64;
        let kernel = dof_circular_kernel(cfg.n_samples, &mut rng);
        let out = dof_gather(&image, &coc_buf, &kernel, &cfg).expect("gather ok");
        assert_eq!(out.len(), w * h * 3);
    }

    #[test]
    fn test_gather_all_in_focus_unchanged() {
        // All CoC < 1 → output == input
        let cfg = DofConfig {
            focus_distance: 2.0,
            focal_length: 10.0,
            aperture: 22.0,
            ..DofConfig::default()
        };
        let w = 4;
        let h = 4;
        let image: Vec<f32> = (0..w * h * 3).map(|i| i as f32 / 100.0).collect();
        let depth = vec![2.0_f32; w * h]; // exactly at focus → CoC = 0
        let coc_buf = dof_compute_coc(&depth, w, h, &cfg).expect("coc");
        // Verify all CoC are < 1
        for &c in &coc_buf.coc {
            assert!(c < 1.0, "expected CoC < 1, got {c}");
        }
        let mut rng = 1u64;
        let kernel = dof_circular_kernel(cfg.n_samples, &mut rng);
        let out = dof_gather(&image, &coc_buf, &kernel, &cfg).expect("gather ok");
        for (i, (&a, &b)) in image.iter().zip(out.iter()).enumerate() {
            assert!(approx_eq(a, b, 1e-5), "pixel {i}: expected {a}, got {b}");
        }
    }

    #[test]
    fn test_gather_dimension_mismatch() {
        let cfg = default_cfg();
        let w = 4;
        let h = 4;
        let image = vec![0.5_f32; w * h * 3 + 1]; // wrong
        let depth = vec![5.0_f32; w * h];
        let coc_buf = dof_compute_coc(&depth, w, h, &cfg).expect("coc");
        let mut rng = 1u64;
        let kernel = dof_circular_kernel(cfg.n_samples, &mut rng);
        assert!(matches!(
            dof_gather(&image, &coc_buf, &kernel, &cfg),
            Err(DofError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_gather_uniform_image_stable() {
        let cfg = default_cfg();
        let w = 8;
        let h = 8;
        let image = solid_rgb(w, h, 0.7, 0.3, 0.5);
        let depth = vec![10.0_f32; w * h]; // far, big CoC
        let coc_buf = dof_compute_coc(&depth, w, h, &cfg).expect("coc");
        let mut rng = 1u64;
        let kernel = dof_circular_kernel(cfg.n_samples, &mut rng);
        let out = dof_gather(&image, &coc_buf, &kernel, &cfg).expect("gather ok");
        for chunk in out.chunks_exact(3) {
            assert!(approx_eq(chunk[0], 0.7, 0.05), "R ≈ 0.7, got {}", chunk[0]);
            assert!(approx_eq(chunk[1], 0.3, 0.05), "G ≈ 0.3, got {}", chunk[1]);
            assert!(approx_eq(chunk[2], 0.5, 0.05), "B ≈ 0.5, got {}", chunk[2]);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // dof_separate_layers
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_separate_layers_near_far() {
        // Half near, half far
        let w = 2;
        let h = 2;
        let image = vec![
            1.0, 0.0, 0.0, // pixel 0
            0.0, 1.0, 0.0, // pixel 1
            0.0, 0.0, 1.0, // pixel 2
            1.0, 1.0, 0.0, // pixel 3
        ];
        let coc = DofCocBuffer {
            coc: vec![2.0, 2.0, 2.0, 2.0],
            is_near: vec![true, false, true, false],
            width: w,
            height: h,
        };
        let (near, far) = dof_separate_layers(&image, &coc).expect("separate ok");
        // Near pixels (0 and 2) should be in near layer
        assert!(approx_eq(near[0], 1.0, 1e-5));
        assert!(approx_eq(near[1], 0.0, 1e-5));
        assert!(approx_eq(near[2], 0.0, 1e-5));
        assert!(approx_eq(near[3], 0.0, 1e-5)); // pixel 1 not near
                                                // Far pixel 1 in far layer
        assert!(approx_eq(far[3], 0.0, 1e-5));
        assert!(approx_eq(far[4], 1.0, 1e-5));
        assert!(approx_eq(far[5], 0.0, 1e-5));
    }

    #[test]
    fn test_separate_layers_near_zeroed_in_far() {
        let w = 2;
        let h = 1;
        let image = vec![0.5, 0.5, 0.5, 0.8, 0.8, 0.8];
        let coc = DofCocBuffer {
            coc: vec![3.0, 3.0],
            is_near: vec![true, false],
            width: w,
            height: h,
        };
        let (near, far) = dof_separate_layers(&image, &coc).expect("ok");
        // Far layer: near pixel should be zeroed
        assert!(approx_eq(far[0], 0.0, 1e-5));
        assert!(approx_eq(far[1], 0.0, 1e-5));
        assert!(approx_eq(far[2], 0.0, 1e-5));
        // Near layer: far pixel should be zeroed
        assert!(approx_eq(near[3], 0.0, 1e-5));
    }

    #[test]
    fn test_separate_layers_dimension_mismatch() {
        let coc = DofCocBuffer {
            coc: vec![1.0; 4],
            is_near: vec![false; 4],
            width: 2,
            height: 2,
        };
        let bad_image = vec![0.5_f32; 10]; // wrong
        assert!(matches!(
            dof_separate_layers(&bad_image, &coc),
            Err(DofError::DimensionMismatch { .. })
        ));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // dof_composite_layers
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_composite_dimension_mismatch() {
        let coc = DofCocBuffer {
            coc: vec![1.0; 4],
            is_near: vec![false; 4],
            width: 2,
            height: 2,
        };
        let good = vec![0.5_f32; 12];
        let bad = vec![0.5_f32; 7];
        assert!(matches!(
            dof_composite_layers(&bad, &good, &good, &coc),
            Err(DofError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_composite_all_far_returns_far() {
        // No near pixels → result should match far_blurred.
        let w = 2;
        let h = 2;
        let near = vec![1.0_f32; 12];
        let far = solid_rgb(w, h, 0.3, 0.6, 0.9);
        let focus = solid_rgb(w, h, 0.5, 0.5, 0.5);
        let coc = DofCocBuffer {
            coc: vec![3.0; 4],
            is_near: vec![false, false, false, false],
            width: w,
            height: h,
        };
        let out = dof_composite_layers(&near, &far, &focus, &coc).expect("composite ok");
        for (i, chunk) in out.chunks_exact(3).enumerate() {
            assert!(approx_eq(chunk[0], 0.3, 1e-4), "pixel {i} R");
            assert!(approx_eq(chunk[1], 0.6, 1e-4), "pixel {i} G");
            assert!(approx_eq(chunk[2], 0.9, 1e-4), "pixel {i} B");
        }
    }

    #[test]
    fn test_composite_all_near_blends_near() {
        // All near with max CoC → alpha=1 → result = near_blurred.
        let w = 2;
        let h = 2;
        let near = solid_rgb(w, h, 1.0, 0.0, 0.0);
        let far = solid_rgb(w, h, 0.0, 1.0, 0.0);
        let focus = solid_rgb(w, h, 0.5, 0.5, 0.5);
        let max_coc = 5.0;
        let coc = DofCocBuffer {
            coc: vec![max_coc; 4],
            is_near: vec![true, true, true, true],
            width: w,
            height: h,
        };
        let out = dof_composite_layers(&near, &far, &focus, &coc).expect("composite ok");
        // coc = 5.0 >= the 1px blurred threshold → focus_weight = 0 → fully
        // defocused (near_blurred, since every pixel is on the near side).
        for chunk in out.chunks_exact(3) {
            assert!(approx_eq(chunk[0], 1.0, 1e-4), "R should be near=1.0");
            assert!(approx_eq(chunk[1], 0.0, 1e-4), "G should be near=0.0");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // apply_depth_of_field
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_apply_dof_output_length() {
        let cfg = default_cfg();
        let w = 5;
        let h = 4;
        let image = solid_rgb(w, h, 0.5, 0.5, 0.5);
        let depth = vec![1.0_f32; w * h];
        let result = apply_depth_of_field(&image, &depth, w, h, &cfg).expect("ok");
        assert_eq!(result.image.len(), w * h * 3);
    }

    #[test]
    fn test_apply_dof_all_in_focus_unchanged() {
        // Very small CoC → blurred ≈ original; strength=1 → result ≈ original.
        let cfg = DofConfig {
            focal_length: 10.0,
            focus_distance: 5.0,
            aperture: 22.0,      // small aperture → small CoC
            max_coc_pixels: 0.1, // force tiny CoC
            n_samples: 4,
            ..DofConfig::default()
        };
        let w = 4;
        let h = 4;
        let image: Vec<f32> = (0..w * h * 3).map(|i| i as f32 / 100.0).collect();
        let depth = vec![5.0_f32; w * h]; // at focus
        let result = apply_depth_of_field(&image, &depth, w, h, &cfg).expect("ok");
        for (i, (&a, &b)) in image.iter().zip(result.image.iter()).enumerate() {
            assert!(approx_eq(a, b, 0.01), "pixel {i}: expected {a}, got {b}");
        }
    }

    #[test]
    fn test_apply_dof_invalid_config_error() {
        let cfg = DofConfig {
            focal_length: 0.0,
            ..DofConfig::default()
        };
        let image = solid_rgb(2, 2, 0.5, 0.5, 0.5);
        let depth = vec![1.0_f32; 4];
        assert!(apply_depth_of_field(&image, &depth, 2, 2, &cfg).is_err());
    }

    #[test]
    fn test_apply_dof_invalid_aperture_error() {
        let cfg = DofConfig {
            aperture: -1.0,
            ..DofConfig::default()
        };
        let image = solid_rgb(2, 2, 0.5, 0.5, 0.5);
        let depth = vec![1.0_f32; 4];
        assert!(matches!(
            apply_depth_of_field(&image, &depth, 2, 2, &cfg),
            Err(DofError::InvalidAperture(_))
        ));
    }

    #[test]
    fn test_apply_dof_strength_zero_returns_input() {
        let cfg = DofConfig {
            strength: 0.0,
            ..DofConfig::default()
        };
        let w = 4;
        let h = 4;
        let image: Vec<f32> = (0..w * h * 3).map(|i| i as f32 / 100.0).collect();
        let depth = vec![10.0_f32; w * h]; // far → lots of blur
        let result = apply_depth_of_field(&image, &depth, w, h, &cfg).expect("ok");
        for (i, (&a, &b)) in image.iter().zip(result.image.iter()).enumerate() {
            assert!(
                approx_eq(a, b, 1e-5),
                "strength=0: pixel {i}: expected {a}, got {b}"
            );
        }
    }

    #[test]
    fn test_apply_dof_coc_map_length() {
        let cfg = default_cfg();
        let w = 3;
        let h = 3;
        let image = solid_rgb(w, h, 0.5, 0.5, 0.5);
        let depth = vec![2.0_f32; w * h];
        let result = apply_depth_of_field(&image, &depth, w, h, &cfg).expect("ok");
        assert_eq!(result.coc_map.len(), w * h);
    }

    #[test]
    fn test_apply_dof_dimension_mismatch() {
        let cfg = default_cfg();
        let image = vec![0.5_f32; 10]; // wrong
        let depth = vec![1.0_f32; 4];
        assert!(matches!(
            apply_depth_of_field(&image, &depth, 2, 2, &cfg),
            Err(DofError::DimensionMismatch { .. })
        ));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // dof_compute_stats
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_all_in_focus() {
        let coc_buf = DofCocBuffer {
            coc: vec![0.0; 9],
            is_near: vec![false; 9],
            width: 3,
            height: 3,
        };
        let stats = dof_compute_stats(&coc_buf);
        assert!(approx_eq(stats.in_focus_fraction, 1.0, 1e-5));
        assert!(approx_eq(stats.near_fraction, 0.0, 1e-5));
        assert!(approx_eq(stats.far_fraction, 0.0, 1e-5));
    }

    #[test]
    fn test_stats_fractions_sum_to_one() {
        let coc_buf = DofCocBuffer {
            coc: vec![0.0, 0.5, 2.0, 3.0],
            is_near: vec![false, false, true, false],
            width: 2,
            height: 2,
        };
        let stats = dof_compute_stats(&coc_buf);
        let total = stats.in_focus_fraction + stats.near_fraction + stats.far_fraction;
        assert!(approx_eq(total, 1.0, 1e-4), "fractions sum = {total}");
    }

    #[test]
    fn test_stats_empty_coc_buf() {
        let coc_buf = DofCocBuffer {
            coc: vec![],
            is_near: vec![],
            width: 0,
            height: 0,
        };
        let stats = dof_compute_stats(&coc_buf);
        assert!(approx_eq(stats.in_focus_fraction, 1.0, 1e-5));
        assert!(approx_eq(stats.mean_coc, 0.0, 1e-5));
    }

    #[test]
    fn test_stats_near_fraction() {
        let coc_buf = DofCocBuffer {
            coc: vec![2.0, 2.0, 2.0, 2.0],
            is_near: vec![true, true, false, false],
            width: 2,
            height: 2,
        };
        let stats = dof_compute_stats(&coc_buf);
        assert!(approx_eq(stats.near_fraction, 0.5, 1e-5));
        assert!(approx_eq(stats.far_fraction, 0.5, 1e-5));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // dof_bilinear_sample
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_bilinear_exact_pixel() {
        let image = vec![
            1.0, 0.0, 0.0, // (0,0)
            0.0, 1.0, 0.0, // (1,0)
            0.0, 0.0, 1.0, // (0,1)
            1.0, 1.0, 0.0, // (1,1)
        ];
        let c = dof_bilinear_sample(&image, 2, 2, 0.0, 0.0);
        assert!(approx_eq(c[0], 1.0, 1e-5));
        assert!(approx_eq(c[1], 0.0, 1e-5));
        assert!(approx_eq(c[2], 0.0, 1e-5));
    }

    #[test]
    fn test_bilinear_fractional_midpoint() {
        let image = vec![
            0.0, 0.0, 0.0, // (0,0)
            1.0, 1.0, 1.0, // (1,0)
            0.0, 0.0, 0.0, // (0,1)
            1.0, 1.0, 1.0, // (1,1)
        ];
        // Midpoint between x=0 and x=1 at y=0 → should be 0.5
        let c = dof_bilinear_sample(&image, 2, 2, 0.5, 0.0);
        assert!(approx_eq(c[0], 0.5, 1e-4));
    }

    #[test]
    fn test_bilinear_boundary_clamping() {
        let image = vec![0.8, 0.4, 0.2, 0.8, 0.4, 0.2, 0.8, 0.4, 0.2, 0.8, 0.4, 0.2];
        // Out-of-bounds → clamped to border
        let c = dof_bilinear_sample(&image, 2, 2, -10.0, -10.0);
        assert!(approx_eq(c[0], 0.8, 1e-5));

        let c2 = dof_bilinear_sample(&image, 2, 2, 100.0, 100.0);
        assert!(approx_eq(c2[0], 0.8, 1e-5));
    }

    #[test]
    fn test_bilinear_empty_image() {
        let c = dof_bilinear_sample(&[], 0, 0, 0.0, 0.0);
        assert!(approx_eq(c[0], 0.0, 1e-9));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Formatting
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_format_config_non_empty() {
        let cfg = DofConfig::default();
        let s = dof_format_config(&cfg);
        assert!(!s.is_empty());
        assert!(s.contains("DofConfig"));
    }

    #[test]
    fn test_format_stats_non_empty() {
        let stats = DofStats {
            mean_coc: 3.5,
            max_coc: 12.0,
            in_focus_fraction: 0.6,
            near_fraction: 0.1,
            far_fraction: 0.3,
        };
        let s = dof_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("DofStats"));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Near-plane blur behaviour
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_near_blur_enabled_produces_blur() {
        let cfg = DofConfig {
            near_blur: true,
            focus_distance: 5.0,
            strength: 1.0,
            n_samples: 32,
            ..DofConfig::default()
        };
        let w = 8;
        let h = 8;
        // Near pixels (depth < focus_distance) should get blurred.
        let depth = vec![0.5_f32; w * h];
        let buf = dof_compute_coc(&depth, w, h, &cfg).expect("coc ok");
        for &near in &buf.is_near {
            assert!(near, "depth=0.5 < focus=5.0 should be near");
        }
        // CoC should be > 0 for near pixels when near_blur enabled.
        for &c in &buf.coc {
            assert!(c > 0.0, "near pixel should have positive CoC");
        }
    }

    #[test]
    fn test_xorshift64_non_zero_state() {
        // Verify PRNG doesn't stay at zero.
        let mut state = 1u64;
        for _ in 0..100 {
            let v = xorshift64(&mut state);
            assert_ne!(v, 0, "xorshift64 should never produce 0 (state fixed)");
        }
    }

    #[test]
    fn test_xorshift_f32_range() {
        let mut state = 12345u64;
        for _ in 0..1000 {
            let v = xorshift_f32(&mut state);
            assert!((0.0..=1.0).contains(&v), "xorshift_f32 out of [0,1]: {v}");
        }
    }

    #[test]
    fn test_apply_dof_bokeh_shapes_all_work() {
        let w = 6;
        let h = 6;
        let image = solid_rgb(w, h, 0.5, 0.3, 0.7);
        let depth = vec![3.0_f32; w * h];
        for shape in [
            BokehShape::Circular,
            BokehShape::Hexagonal,
            BokehShape::Pentagonal,
            BokehShape::Square,
        ] {
            let cfg = DofConfig {
                bokeh_shape: shape,
                n_samples: 8,
                ..DofConfig::default()
            };
            let result = apply_depth_of_field(&image, &depth, w, h, &cfg).expect("smoke ok");
            assert_eq!(result.image.len(), w * h * 3, "shape {:?}", shape);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Regression tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_coc_ramp_is_smooth_not_a_step() {
        // Regression test for the mm/world-unit unit-mixing bug: with the
        // documented defaults (85mm f/2.8 lens focused at 1 world-unit) and
        // a realistic image width, a depth just 5% beyond the focal plane
        // must produce a small, smoothly-varying CoC - not an instant jump
        // to `max_coc_pixels` (the pre-fix scale clamped for depths as
        // little as ~0.44% away from focus).
        let cfg = DofConfig::default();
        // `dof_compute_coc` requires `depth_buf.len() == width * height`, so a
        // "one pixel at depth d, at a realistic image width" probe is a
        // single row of `width` identical samples (a 1-element buffer would
        // just return `DimensionMismatch`).
        let width = 512;
        let probe = |d: f32| -> f32 {
            let depth_row = vec![d; width];
            dof_compute_coc(&depth_row, width, 1, &cfg)
                .expect("coc ok")
                .coc[0]
        };

        let coc_just_past_focus = probe(1.05);
        assert!(
            coc_just_past_focus < cfg.max_coc_pixels * 0.5,
            "CoC 5% past focus should be far below the clamp ceiling, got {} (max {})",
            coc_just_past_focus,
            cfg.max_coc_pixels
        );
        assert!(
            coc_just_past_focus > 0.0,
            "CoC should be non-zero away from focus"
        );

        // The far side should still ramp up smoothly (not clamp everywhere)
        // as depth increases further from the focal plane.
        //
        // Hand-derived expectation for the defaults (f=85mm, N=2.8, S=1
        // world unit = 1000mm, sensor 36mm, width 512 -> 14.22 px/mm):
        //   CoC_mm = (85^2/2.8) * |d-1| / (d * (1000-85))
        //   d=1.05 -> 0.134mm -> 1.91px    d=1.2 -> 0.470mm ->  6.68px
        //   d=1.5  -> 0.940mm -> 13.37px   d=2.0 -> 1.410mm -> 20.05px (clamps)
        //   d=3.0  -> 1.880mm -> 26.74px (clamps)
        // i.e. a monotone ramp that only reaches the 20px ceiling at the far
        // end, which is exactly what this test asserts.
        let depths = [1.05_f32, 1.2, 1.5, 2.0, 3.0];
        let mut prev = 0.0_f32;
        let mut clamped = 0usize;
        for &d in &depths {
            let c = probe(d);
            assert!(
                c >= prev - 1e-5,
                "CoC should not decrease with depth: {c} < {prev}"
            );
            if c >= cfg.max_coc_pixels - 1e-4 {
                clamped += 1;
            }
            prev = c;
        }
        assert!(
            clamped < depths.len(),
            "expected a smooth ramp, but every sampled depth clamped to max_coc_pixels: {depths:?}"
        );
    }

    #[test]
    fn test_hexagonal_kernel_radius_is_area_uniform() {
        // Regression test: sampling `r` uniformly in `[0, r_max)` biases
        // samples toward the centre (density ∝ 1/r), so a hexagon/pentagon
        // kernel would look like a soft radial falloff indistinguishable
        // from a circle. Area-uniform sampling (`r = r_max * sqrt(u)`) makes
        // the *count* of samples falling in an annulus grow with its area,
        // i.e. roughly with its radius.
        let mut rng = 123u64;
        let n = 20_000;
        let k = dof_hexagonal_kernel(n, &mut rng);

        // Bin samples by polar radius into 5 equal-width bands over [0, 1)
        // (a regular hexagon inscribed in the unit circle has radius <= 1 in
        // every direction, so every sample's polar radius is <= 1).
        const BANDS: usize = 5;
        let mut counts = [0usize; BANDS];
        for pair in k.chunks_exact(2) {
            let r = (pair[0] * pair[0] + pair[1] * pair[1])
                .sqrt()
                .min(0.999_999);
            let band = ((r * BANDS as f32) as usize).min(BANDS - 1);
            counts[band] += 1;
        }

        // Area-uniform sampling puts markedly more mass in the outer band
        // than the inner one (annulus area grows with radius); uniform-in-r
        // sampling would instead give roughly equal counts per band.
        assert!(
            counts[BANDS - 1] > counts[0] * 2,
            "expected area-uniform radial sampling (outer band >> inner band), got {counts:?}"
        );
    }

    #[test]
    fn test_pentagonal_kernel_radius_is_area_uniform() {
        // Same check as `test_hexagonal_kernel_radius_is_area_uniform`, for
        // the pentagonal kernel.
        let mut rng = 456u64;
        let n = 20_000;
        let k = dof_pentagonal_kernel(n, &mut rng);

        const BANDS: usize = 5;
        let mut counts = [0usize; BANDS];
        for pair in k.chunks_exact(2) {
            let r = (pair[0] * pair[0] + pair[1] * pair[1])
                .sqrt()
                .min(0.999_999);
            let band = ((r * BANDS as f32) as usize).min(BANDS - 1);
            counts[band] += 1;
        }

        assert!(
            counts[BANDS - 1] > counts[0] * 2,
            "expected area-uniform radial sampling (outer band >> inner band), got {counts:?}"
        );
    }

    #[test]
    fn test_near_blur_disabled_forces_zero_coc_and_leaves_pixels_untouched() {
        // Regression test: `near_blur = false` must (a) make
        // `dof_compute_coc` report CoC = 0 for every near-plane pixel, and
        // (b) make the full pipeline leave those pixels completely
        // untouched - previously `near_blur` was written only in `Default`
        // and read nowhere, so it had no effect at all.
        let cfg = DofConfig {
            near_blur: false,
            focus_distance: 5.0,
            strength: 1.0,
            n_samples: 16,
            ..DofConfig::default()
        };
        // A realistic width (not the 1-pixel-wide buffers some other tests
        // use) so the far half's CoC actually clears the 1px blur
        // threshold - otherwise "still genuinely blurred" below wouldn't
        // hold and this test would only exercise the trivial "both halves
        // happen to be sharp" case.
        let w = 256;
        let h = 8;
        let image: Vec<f32> = (0..w * h * 3).map(|i| (i % 7) as f32 / 10.0).collect();
        // Top half is near (depth 0.5 < focus 5.0); bottom half is far
        // (depth 20.0 > focus 5.0) so it is still genuinely blurred.
        let mut depth = vec![20.0_f32; w * h];
        for row in 0..h / 2 {
            for col in 0..w {
                depth[row * w + col] = 0.5;
            }
        }

        let coc_buf = dof_compute_coc(&depth, w, h, &cfg).expect("coc ok");
        for row in 0..h / 2 {
            for col in 0..w {
                let idx = row * w + col;
                assert_eq!(
                    coc_buf.coc[idx], 0.0,
                    "near_blur=false must force near-pixel CoC to 0 at ({col},{row})"
                );
            }
        }

        let result = apply_depth_of_field(&image, &depth, w, h, &cfg).expect("dof ok");
        for row in 0..h / 2 {
            for col in 0..w {
                let base = (row * w + col) * 3;
                for c in 0..3 {
                    assert!(
                        approx_eq(image[base + c], result.image[base + c], 1e-5),
                        "near pixel ({col},{row}) channel {c} should be untouched when near_blur=false"
                    );
                }
            }
        }
    }

    #[test]
    fn test_gather_sharp_near_pixel_does_not_bleed() {
        // Regression test: a near-plane neighbour must only bleed into
        // another pixel's blur when it is itself already blurred
        // (`n_coc >= 1px`); a near-plane neighbour that is essentially in
        // focus (small CoC) must not be admitted just because `is_near` is
        // true - that was the foreground-bleed bug in the old
        // `n_coc >= dist || n_is_near` acceptance test.
        //
        // 3 pixels in a row: [red destination (coc=1, blurred), green sharp
        // near-plane pixel (coc=0.3, is_near), blue filler]. A single
        // deterministic kernel sample offset by +1px (scaled by the
        // destination's own coc_r=1.0) lands exactly on the green pixel.
        let w = 3;
        let h = 1;
        let image = vec![
            1.0, 0.0, 0.0, // pixel 0: red (destination)
            0.0, 1.0, 0.0, // pixel 1: green, sharp near pixel
            0.0, 0.0, 1.0, // pixel 2: blue filler
        ];
        let coc_buf = DofCocBuffer {
            coc: vec![1.0, 0.3, 1.0],
            is_near: vec![false, true, false],
            width: w,
            height: h,
        };
        let cfg = DofConfig {
            near_blur: true,
            ..DofConfig::default()
        };
        let kernel = vec![1.0_f32, 0.0_f32]; // single deterministic sample: +1px in x
        let out = dof_gather(&image, &coc_buf, &kernel, &cfg).expect("gather ok");

        // Destination pixel 0 must stay red: the sharp (coc=0.3 < 1px) near
        // pixel it sampled must not have been admitted "for free".
        assert!(
            (out[0] - 1.0).abs() < 1e-5 && out[1] < 1e-5 && out[2] < 1e-5,
            "sharp near pixel leaked into destination: got {:?}",
            &out[0..3]
        );
    }

    #[test]
    fn test_composite_partial_coc_blends_toward_in_focus() {
        // Regression test: `in_focus` must actually be used - a pixel whose
        // CoC is between 0 and the 1px threshold should be a blend of its
        // defocused layer and the in-focus layer, not just the defocused
        // layer verbatim (which is what happened when `in_focus` was
        // validated for size but never read).
        let w = 1;
        let h = 1;
        let near = vec![0.0_f32, 0.0, 0.0]; // black
        let far = vec![0.0_f32, 0.0, 0.0]; // black (unused: pixel is near)
        let focus = vec![1.0_f32, 1.0, 1.0]; // white
        let coc_buf = DofCocBuffer {
            coc: vec![0.5], // halfway to the 1px blurred threshold
            is_near: vec![true],
            width: w,
            height: h,
        };
        let out = dof_composite_layers(&near, &far, &focus, &coc_buf).expect("composite ok");
        // focus_weight = 1 - 0.5 = 0.5 → result = lerp(near=black, focus=white, 0.5) = 0.5
        for &v in &out {
            assert!(
                (v - 0.5).abs() < 1e-4,
                "expected a 50/50 blend of near_blurred and in_focus, got {v}"
            );
        }
    }
}
