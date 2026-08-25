//! # Cross-Frame Consistency
//!
//! Measures and enforces temporal consistency of rendered head avatar sequences,
//! ensuring smooth, stable appearance across frames without jitter or flicker.
//!
//! Used in the diffusion training loss and evaluation pipeline to penalize temporal
//! inconsistency in generated sequences. Provides Horn-Schunck optical flow, warping,
//! PSNR/SSIM metrics, and a differentiable consistency loss.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by cross-frame consistency operations.
#[derive(Debug, Error)]
pub enum ConsistencyError {
    /// Sequence has fewer than 2 frames; inter-frame metrics are undefined.
    #[error("empty sequence: need at least 2 frames")]
    EmptySequence,

    /// Two frames have incompatible pixel-buffer sizes.
    #[error("dimension mismatch: frame {a} has {na} pixels, frame {b} has {nb} pixels")]
    FrameDimensionMismatch {
        a: usize,
        b: usize,
        na: usize,
        nb: usize,
    },

    /// A configuration parameter is out of range or otherwise invalid.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// Sequence is too short for the requested operation.
    #[error("sequence too short: need at least {needed} frames, got {got}")]
    TooShort { needed: usize, got: usize },

    /// A weight value is invalid (NaN, negative, or infinite).
    #[error("invalid weight: {0}")]
    InvalidWeight(f32),
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame
// ─────────────────────────────────────────────────────────────────────────────

/// A single RGB frame with flat interleaved pixel storage.
///
/// Pixels are stored as `R0,G0,B0, R1,G1,B1, …` in row-major order.
/// Values are expected to be in `[0, 1]`.
///
/// The fields are private and the invariant `pixels.len() == width *
/// height * 3` is enforced at construction time (via [`Frame::new`] or
/// [`Frame::from_pixels`], the only ways to build one), so every function
/// that indexes `pixels` using `width`/`height` is panic-free by
/// construction rather than by ad hoc bounds checks.
pub struct Frame {
    /// Interleaved RGB values; length must equal `width * height * 3`.
    pixels: Vec<f32>,
    /// Image width in pixels.
    width: usize,
    /// Image height in pixels.
    height: usize,
}

impl Frame {
    /// Create a new all-zero frame with the given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        Frame {
            pixels: vec![0.0_f32; width * height * 3],
            width,
            height,
        }
    }

    /// Create a frame from an existing pixel buffer.
    ///
    /// Returns [`ConsistencyError::InvalidConfig`] if the buffer length does not
    /// equal `width * height * 3`.
    pub fn from_pixels(
        pixels: Vec<f32>,
        width: usize,
        height: usize,
    ) -> Result<Self, ConsistencyError> {
        let expected = width * height * 3;
        if pixels.len() != expected {
            return Err(ConsistencyError::InvalidConfig(format!(
                "pixel buffer length {} does not match {}×{}×3 = {}",
                pixels.len(),
                width,
                height,
                expected
            )));
        }
        Ok(Frame {
            pixels,
            width,
            height,
        })
    }

    /// Image width in pixels.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height in pixels.
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Read-only access to the interleaved RGB pixel buffer.
    #[inline]
    pub fn pixels(&self) -> &[f32] {
        &self.pixels
    }

    /// Number of pixels (not channels).
    #[inline]
    pub fn n_pixels(&self) -> usize {
        self.width * self.height
    }

    /// RGB triple for the pixel at column `x`, row `y`.
    ///
    /// Returns [`ConsistencyError::InvalidConfig`] when coordinates are out of bounds.
    pub fn pixel_at(&self, x: usize, y: usize) -> Result<[f32; 3], ConsistencyError> {
        if x >= self.width || y >= self.height {
            return Err(ConsistencyError::InvalidConfig(format!(
                "pixel ({x},{y}) out of bounds for {}×{} frame",
                self.width, self.height
            )));
        }
        let base = (y * self.width + x) * 3;
        Ok([
            self.pixels[base],
            self.pixels[base + 1],
            self.pixels[base + 2],
        ])
    }

    /// Mean luminance averaged over all pixels and channels.
    pub fn mean_brightness(&self) -> f32 {
        if self.pixels.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.pixels.iter().copied().sum();
        sum / self.pixels.len() as f32
    }

    /// Variance of pixel values across all channels.
    pub fn variance(&self) -> f32 {
        if self.pixels.len() < 2 {
            return 0.0;
        }
        let n = self.pixels.len() as f32;
        let mean = self.mean_brightness();
        let sq_sum: f32 = self.pixels.iter().map(|&v| (v - mean) * (v - mean)).sum();
        sq_sum / n
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Optical flow configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Horn-Schunck optical flow estimator.
#[derive(Debug, Clone)]
pub struct FlowConfig {
    /// Smoothness regularisation weight (λ in the Horn-Schunck energy).
    /// Higher values produce smoother but less accurate flow fields.
    /// Default: 100.0.
    pub alpha: f32,
    /// Number of fixed-point (Jacobi) iterations. Default: 20.
    pub n_iterations: usize,
    /// Downscale factor applied before flow estimation for speed.
    /// `1` = full resolution, `2` = half, etc. Default: 2.
    pub scale: usize,
}

impl Default for FlowConfig {
    fn default() -> Self {
        FlowConfig {
            alpha: 100.0,
            n_iterations: 20,
            scale: 2,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Optical flow utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Convert an RGB frame to a single-channel grayscale image.
///
/// Uses the standard ITU-R BT.709 luminance coefficients:
/// `Y = 0.2126·R + 0.7152·G + 0.0722·B`.
///
/// Returns a flat `Vec<f32>` of length `width × height`.
pub fn cfc_to_grayscale(frame: &Frame) -> Vec<f32> {
    let n = frame.n_pixels();
    let mut grey = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * 3;
        let lum = 0.2126 * frame.pixels[base]
            + 0.7152 * frame.pixels[base + 1]
            + 0.0722 * frame.pixels[base + 2];
        grey.push(lum);
    }
    grey
}

/// Bilinear sample of an RGB frame at sub-pixel coordinates `(x, y)`.
///
/// Out-of-range coordinates are clamped to the frame boundary. Returns
/// `[0.0; 3]` for a zero-width or zero-height frame, since there is no
/// pixel to sample.
pub fn cfc_bilinear_sample(frame: &Frame, x: f32, y: f32) -> [f32; 3] {
    if frame.width == 0 || frame.height == 0 {
        return [0.0; 3];
    }
    let w = frame.width as f32;
    let h = frame.height as f32;

    // Clamp to valid range
    let xc = x.clamp(0.0, w - 1.0);
    let yc = y.clamp(0.0, h - 1.0);

    let x0 = xc.floor() as usize;
    let y0 = yc.floor() as usize;
    let x1 = (x0 + 1).min(frame.width.saturating_sub(1));
    let y1 = (y0 + 1).min(frame.height.saturating_sub(1));

    let tx = xc - x0 as f32;
    let ty = yc - y0 as f32;

    let idx = |row: usize, col: usize| -> usize { (row * frame.width + col) * 3 };

    let i00 = idx(y0, x0);
    let i01 = idx(y0, x1);
    let i10 = idx(y1, x0);
    let i11 = idx(y1, x1);

    let mut out = [0.0_f32; 3];
    for (c, out_val) in out.iter_mut().enumerate() {
        let v00 = frame.pixels[i00 + c];
        let v01 = frame.pixels[i01 + c];
        let v10 = frame.pixels[i10 + c];
        let v11 = frame.pixels[i11 + c];
        *out_val = v00 * (1.0 - tx) * (1.0 - ty)
            + v01 * tx * (1.0 - ty)
            + v10 * (1.0 - tx) * ty
            + v11 * tx * ty;
    }
    out
}

/// Downscale a grayscale image by an integer factor using box averaging.
fn downscale_grey(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    factor: usize,
) -> (Vec<f32>, usize, usize) {
    if factor <= 1 {
        return (src.to_vec(), src_w, src_h);
    }
    let dw = src_w.div_ceil(factor);
    let dh = src_h.div_ceil(factor);
    let mut dst = vec![0.0_f32; dw * dh];
    let mut counts = vec![0u32; dw * dh];
    for row in 0..src_h {
        for col in 0..src_w {
            let dr = row / factor;
            let dc = col / factor;
            if dr < dh && dc < dw {
                dst[dr * dw + dc] += src[row * src_w + col];
                counts[dr * dw + dc] += 1;
            }
        }
    }
    for i in 0..dst.len() {
        if counts[i] > 0 {
            dst[i] /= counts[i] as f32;
        }
    }
    (dst, dw, dh)
}

/// Upscale a flow field from `(sw, sh)` back to `(tw, th)` by nearest-neighbour.
fn upscale_flow(src: &[f32], sw: usize, sh: usize, tw: usize, th: usize, factor: f32) -> Vec<f32> {
    let mut dst = vec![0.0_f32; tw * th];
    for row in 0..th {
        for col in 0..tw {
            let sr = ((row as f32 / factor) as usize).min(sh.saturating_sub(1));
            let sc = ((col as f32 / factor) as usize).min(sw.saturating_sub(1));
            dst[row * tw + col] = src[sr * sw + sc] * factor;
        }
    }
    dst
}

/// Compute the average of the 4-connected neighbours of pixel `(r, c)`.
fn laplacian_avg(field: &[f32], w: usize, h: usize, r: usize, c: usize) -> f32 {
    let mut sum = 0.0_f32;
    let mut n = 0u32;
    if r > 0 {
        sum += field[(r - 1) * w + c];
        n += 1;
    }
    if r + 1 < h {
        sum += field[(r + 1) * w + c];
        n += 1;
    }
    if c > 0 {
        sum += field[r * w + c - 1];
        n += 1;
    }
    if c + 1 < w {
        sum += field[r * w + c + 1];
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f32
    }
}

/// Estimate optical flow between two frames using a simplified Horn-Schunck
/// algorithm.
///
/// Returns `(flow_x, flow_y)` as flat `Vec<f32>` buffers of length
/// `frame_a.width * frame_a.height`, giving horizontal and vertical pixel
/// displacements respectively.
///
/// # Errors
/// - [`ConsistencyError::FrameDimensionMismatch`] when frames have different sizes.
/// - [`ConsistencyError::InvalidConfig`] when `scale == 0`.
pub fn cfc_compute_flow(
    frame_a: &Frame,
    frame_b: &Frame,
    config: &FlowConfig,
) -> Result<(Vec<f32>, Vec<f32>), ConsistencyError> {
    if frame_a.n_pixels() != frame_b.n_pixels()
        || frame_a.width != frame_b.width
        || frame_a.height != frame_b.height
    {
        return Err(ConsistencyError::FrameDimensionMismatch {
            a: 0,
            b: 1,
            na: frame_a.n_pixels(),
            nb: frame_b.n_pixels(),
        });
    }

    let grey_a = cfc_to_grayscale(frame_a);
    let grey_b = cfc_to_grayscale(frame_b);
    cfc_compute_flow_from_grayscale(&grey_a, &grey_b, frame_a.width, frame_a.height, config)
}

/// Core of [`cfc_compute_flow`], given precomputed grayscale planes.
///
/// Shared with the sequence-level fast path in [`cfc_sequence_consistency`],
/// which precomputes each frame's grayscale plane once and reuses it across
/// the (up to two) consecutive pairs it appears in, instead of recomputing
/// it on every [`cfc_compute_flow`]/[`cfc_ssim`] call.
fn cfc_compute_flow_from_grayscale(
    grey_a: &[f32],
    grey_b: &[f32],
    width: usize,
    height: usize,
    config: &FlowConfig,
) -> Result<(Vec<f32>, Vec<f32>), ConsistencyError> {
    if config.scale == 0 {
        return Err(ConsistencyError::InvalidConfig(
            "FlowConfig::scale must be >= 1".to_string(),
        ));
    }

    let (ds_a, dw, dh) = downscale_grey(grey_a, width, height, config.scale);
    let (ds_b, _, _) = downscale_grey(grey_b, width, height, config.scale);

    let n = dw * dh;
    let mut u = vec![0.0_f32; n]; // flow_x at downscaled res
    let mut v = vec![0.0_f32; n]; // flow_y at downscaled res
                                  // Reused scratch buffers for the previous iteration's values, swapped
                                  // in each iteration instead of cloning `u`/`v` (2 * n_iterations fewer
                                  // full-length allocations). Every element of `u`/`v` is overwritten
                                  // unconditionally by the inner loop below before being read again, so
                                  // the stale contents left behind by the swap are never observed.
    let mut u_prev = vec![0.0_f32; n];
    let mut v_prev = vec![0.0_f32; n];

    let alpha_sq = config.alpha * config.alpha;

    for _iter in 0..config.n_iterations {
        std::mem::swap(&mut u, &mut u_prev);
        std::mem::swap(&mut v, &mut v_prev);

        for row in 0..dh {
            for col in 0..dw {
                let i = row * dw + col;

                // Spatial gradient Ix: central difference (x direction)
                let ix = if col + 1 < dw && col > 0 {
                    (ds_a[row * dw + col + 1] - ds_a[row * dw + col - 1]) * 0.5
                } else if col + 1 < dw {
                    ds_a[row * dw + col + 1] - ds_a[row * dw + col]
                } else if col > 0 {
                    ds_a[row * dw + col] - ds_a[row * dw + col - 1]
                } else {
                    0.0
                };

                // Spatial gradient Iy: central difference (y direction)
                let iy = if row + 1 < dh && row > 0 {
                    (ds_a[(row + 1) * dw + col] - ds_a[(row - 1) * dw + col]) * 0.5
                } else if row + 1 < dh {
                    ds_a[(row + 1) * dw + col] - ds_a[row * dw + col]
                } else if row > 0 {
                    ds_a[row * dw + col] - ds_a[(row - 1) * dw + col]
                } else {
                    0.0
                };

                // Temporal gradient It: forward difference
                let it = ds_b[i] - ds_a[i];

                let u_avg = laplacian_avg(&u_prev, dw, dh, row, col);
                let v_avg = laplacian_avg(&v_prev, dw, dh, row, col);

                let denom = alpha_sq + ix * ix + iy * iy;
                let p = (ix * u_avg + iy * v_avg + it) / denom;

                u[i] = u_avg - ix * p;
                v[i] = v_avg - iy * p;
            }
        }
    }

    // Upscale back to original resolution
    let factor = config.scale as f32;
    let full_u = upscale_flow(&u, dw, dh, width, height, factor);
    let full_v = upscale_flow(&v, dw, dh, width, height, factor);

    Ok((full_u, full_v))
}

/// Warp `frame` backward by the given flow field using bilinear interpolation.
///
/// For each output pixel at `(x, y)`, samples `frame` at `(x + flow_x[i],
/// y + flow_y[i])` with border clamping.
///
/// # Errors
/// - [`ConsistencyError::InvalidConfig`] when flow buffer lengths do not match
///   the frame's pixel count.
pub fn cfc_warp_frame(
    frame: &Frame,
    flow_x: &[f32],
    flow_y: &[f32],
) -> Result<Frame, ConsistencyError> {
    let n = frame.n_pixels();
    if flow_x.len() != n || flow_y.len() != n {
        return Err(ConsistencyError::InvalidConfig(format!(
            "flow length mismatch: expected {n}, got flow_x={}, flow_y={}",
            flow_x.len(),
            flow_y.len()
        )));
    }

    let mut out = Frame::new(frame.width, frame.height);
    for row in 0..frame.height {
        for col in 0..frame.width {
            let i = row * frame.width + col;
            let sx = col as f32 + flow_x[i];
            let sy = row as f32 + flow_y[i];
            let rgb = cfc_bilinear_sample(frame, sx, sy);
            out.pixels[i * 3] = rgb[0];
            out.pixels[i * 3 + 1] = rgb[1];
            out.pixels[i * 3 + 2] = rgb[2];
        }
    }
    Ok(out)
}

/// Negate a flow field: `(fx, fy) -> (-fx, -fy)`.
///
/// [`cfc_compute_flow`] estimates the *forward* A→B flow (a point at `(x,
/// y)` in `A` appears at `(x+u, y+v)` in `B`). [`cfc_warp_frame`] performs
/// a *backward* warp, `warped(x) = frame(x + flow(x))`, which needs the
/// target→source flow to align `warped(frame_a)` with `frame_b`. Negating
/// the forward flow is the correct first-order approximation to that
/// inversion (exact for a spatially-uniform/translational flow field, and
/// the standard small-motion approximation otherwise) — call this before
/// warping `frame_a` toward `frame_b` with a flow computed from the same
/// pair, never `cfc_warp_frame` itself, which is correct as documented.
fn negate_flow(fx: &[f32], fy: &[f32]) -> (Vec<f32>, Vec<f32>) {
    (
        fx.iter().map(|&v| -v).collect(),
        fy.iter().map(|&v| -v).collect(),
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// PSNR / SSIM / MAE / RMSE utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Mean squared error between two frames (per channel-sample average).
fn frame_mse(a: &Frame, b: &Frame) -> f32 {
    debug_assert_eq!(a.pixels.len(), b.pixels.len());
    let n = a.pixels.len() as f32;
    let sum: f32 = a
        .pixels
        .iter()
        .zip(b.pixels.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum();
    sum / n
}

/// Check that two frames have identical dimensions.
fn check_dims(a: &Frame, b: &Frame) -> Result<(), ConsistencyError> {
    if a.n_pixels() != b.n_pixels() || a.width != b.width || a.height != b.height {
        return Err(ConsistencyError::FrameDimensionMismatch {
            a: 0,
            b: 1,
            na: a.n_pixels(),
            nb: b.n_pixels(),
        });
    }
    Ok(())
}

/// Peak signal-to-noise ratio (dB) between two frames.
///
/// Assumes pixel values are in `[0, 1]`, so `MAX = 1.0`.
/// Returns `f32::INFINITY` when the frames are identical (MSE = 0).
///
/// # Errors
/// - [`ConsistencyError::FrameDimensionMismatch`] on dimension mismatch.
pub fn cfc_psnr(frame_a: &Frame, frame_b: &Frame) -> Result<f32, ConsistencyError> {
    check_dims(frame_a, frame_b)?;
    let mse = frame_mse(frame_a, frame_b);
    if mse == 0.0 {
        return Ok(f32::INFINITY);
    }
    Ok(-10.0 * mse.log10())
}

/// Mean absolute error between two frames.
///
/// # Errors
/// - [`ConsistencyError::FrameDimensionMismatch`] on dimension mismatch.
pub fn cfc_mae(frame_a: &Frame, frame_b: &Frame) -> Result<f32, ConsistencyError> {
    check_dims(frame_a, frame_b)?;
    let n = frame_a.pixels.len() as f32;
    let sum: f32 = frame_a
        .pixels
        .iter()
        .zip(frame_b.pixels.iter())
        .map(|(&x, &y)| (x - y).abs())
        .sum();
    Ok(sum / n)
}

/// Root mean square error between two frames.
///
/// # Errors
/// - [`ConsistencyError::FrameDimensionMismatch`] on dimension mismatch.
pub fn cfc_rmse(frame_a: &Frame, frame_b: &Frame) -> Result<f32, ConsistencyError> {
    check_dims(frame_a, frame_b)?;
    Ok(frame_mse(frame_a, frame_b).sqrt())
}

/// Simplified SSIM between two frames, averaged over non-overlapping 8×8 windows.
///
/// Uses the standard stability constants `C1 = (0.01)² = 0.0001` and
/// `C2 = (0.03)² = 0.0009` (dynamic range L = 1).
///
/// # Errors
/// - [`ConsistencyError::FrameDimensionMismatch`] on dimension mismatch.
pub fn cfc_ssim(frame_a: &Frame, frame_b: &Frame) -> Result<f32, ConsistencyError> {
    check_dims(frame_a, frame_b)?;
    // Work on luminance channel only for SSIM
    let la = cfc_to_grayscale(frame_a);
    let lb = cfc_to_grayscale(frame_b);
    Ok(cfc_ssim_from_grayscale(
        &la,
        &lb,
        frame_a.width,
        frame_a.height,
    ))
}

/// Core of [`cfc_ssim`], given precomputed grayscale planes.
///
/// Shared with the sequence-level fast path in [`cfc_sequence_consistency`]
/// (see [`cfc_compute_flow_from_grayscale`] for why).
fn cfc_ssim_from_grayscale(la: &[f32], lb: &[f32], w: usize, h: usize) -> f32 {
    const WIN: usize = 8;
    const C1: f32 = 0.0001;
    const C2: f32 = 0.0009;

    let mut ssim_sum = 0.0_f32;
    let mut n_windows = 0u32;

    let mut wy = 0;
    while wy + WIN <= h {
        let mut wx = 0;
        while wx + WIN <= w {
            let count = (WIN * WIN) as f32;

            // First pass: means.
            let mut sum_a = 0.0_f32;
            let mut sum_b = 0.0_f32;
            for ry in 0..WIN {
                for rx in 0..WIN {
                    let idx = (wy + ry) * w + (wx + rx);
                    sum_a += la[idx];
                    sum_b += lb[idx];
                }
            }
            let mu_a = sum_a / count;
            let mu_b = sum_b / count;

            // Second pass: (co)variances from centered differences. This
            // avoids the catastrophic cancellation of the one-pass
            // `E[x^2] - E[x]^2` form, which can go slightly negative in f32
            // for near-saturated inputs and flip the SSIM denominator's
            // sign.
            let mut sig_aa = 0.0_f32;
            let mut sig_bb = 0.0_f32;
            let mut sig_ab = 0.0_f32;
            for ry in 0..WIN {
                for rx in 0..WIN {
                    let idx = (wy + ry) * w + (wx + rx);
                    let da = la[idx] - mu_a;
                    let db = lb[idx] - mu_b;
                    sig_aa += da * da;
                    sig_bb += db * db;
                    sig_ab += da * db;
                }
            }
            let sig_aa = (sig_aa / count).max(0.0);
            let sig_bb = (sig_bb / count).max(0.0);
            let sig_ab = sig_ab / count;

            let numerator = (2.0 * mu_a * mu_b + C1) * (2.0 * sig_ab + C2);
            let denominator = (mu_a * mu_a + mu_b * mu_b + C1) * (sig_aa + sig_bb + C2);

            ssim_sum += numerator / denominator;
            n_windows += 1;
            wx += WIN;
        }
        wy += WIN;
    }

    if n_windows == 0 {
        // Frame smaller than one window — fall back to per-pixel SSIM
        let mu_a = la.iter().copied().sum::<f32>() / la.len() as f32;
        let mu_b = lb.iter().copied().sum::<f32>() / lb.len() as f32;
        let sig_ab: f32 = la
            .iter()
            .zip(lb.iter())
            .map(|(&a, &b)| (a - mu_a) * (b - mu_b))
            .sum::<f32>()
            / la.len() as f32;
        let sig_aa: f32 =
            (la.iter().map(|&a| (a - mu_a) * (a - mu_a)).sum::<f32>() / la.len() as f32).max(0.0);
        let sig_bb: f32 =
            (lb.iter().map(|&b| (b - mu_b) * (b - mu_b)).sum::<f32>() / lb.len() as f32).max(0.0);
        let num = (2.0 * mu_a * mu_b + C1) * (2.0 * sig_ab + C2);
        let den = (mu_a * mu_a + mu_b * mu_b + C1) * (sig_aa + sig_bb + C2);
        return num / den;
    }

    ssim_sum / n_windows as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame-pair consistency
// ─────────────────────────────────────────────────────────────────────────────

/// Temporal consistency metrics between two consecutive frames.
pub struct FramePairConsistency {
    /// PSNR (dB) between the warped predecessor frame and the current frame.
    pub psnr: f32,
    /// SSIM between the warped predecessor and current frame.
    pub ssim: f32,
    /// Mean L1 error (per channel) after warping.
    pub mean_warp_error: f32,
    /// Average optical flow magnitude (pixels/frame).
    pub mean_flow_magnitude: f32,
    /// Fraction of pixels whose per-pixel L1 error exceeds 0.1.
    pub occlusion_ratio: f32,
}

/// Compute consistency between two frames using optical flow to warp `frame_a`
/// toward `frame_b`, then measuring the residual error.
///
/// # Errors
/// - Propagates [`ConsistencyError::FrameDimensionMismatch`] from flow computation.
pub fn cfc_frame_pair_consistency(
    frame_a: &Frame,
    frame_b: &Frame,
    flow_config: &FlowConfig,
) -> Result<FramePairConsistency, ConsistencyError> {
    let (fx, fy) = cfc_compute_flow(frame_a, frame_b, flow_config)?;
    // cfc_compute_flow returns the forward A→B flow; cfc_warp_frame needs
    // the negated flow to correctly backward-warp frame_a toward frame_b
    // (see `negate_flow`).
    let (neg_fx, neg_fy) = negate_flow(&fx, &fy);
    let warped = cfc_warp_frame(frame_a, &neg_fx, &neg_fy)?;

    let psnr = cfc_psnr(&warped, frame_b)?;
    let ssim = cfc_ssim(&warped, frame_b)?;
    let mean_warp_error = cfc_mae(&warped, frame_b)?;

    // Mean flow magnitude
    let n = fx.len() as f32;
    let mag_sum: f32 = fx
        .iter()
        .zip(fy.iter())
        .map(|(&u, &v)| (u * u + v * v).sqrt())
        .sum();
    let mean_flow_magnitude = mag_sum / n;

    // Occlusion ratio: fraction of pixels with per-pixel L1 > 0.1
    let n_pixels = frame_a.n_pixels();
    let mut occluded = 0u32;
    for i in 0..n_pixels {
        let base = i * 3;
        let err = (warped.pixels[base] - frame_b.pixels[base]).abs()
            + (warped.pixels[base + 1] - frame_b.pixels[base + 1]).abs()
            + (warped.pixels[base + 2] - frame_b.pixels[base + 2]).abs();
        if err / 3.0 > 0.1 {
            occluded += 1;
        }
    }
    let occlusion_ratio = occluded as f32 / n_pixels as f32;

    Ok(FramePairConsistency {
        psnr,
        ssim,
        mean_warp_error,
        mean_flow_magnitude,
        occlusion_ratio,
    })
}

/// Faster consistency measurement without optical flow.
///
/// Computes direct frame-difference metrics, setting `mean_flow_magnitude` to 0.
///
/// # Errors
/// - [`ConsistencyError::FrameDimensionMismatch`] on dimension mismatch.
pub fn cfc_frame_difference(
    frame_a: &Frame,
    frame_b: &Frame,
) -> Result<FramePairConsistency, ConsistencyError> {
    check_dims(frame_a, frame_b)?;

    let psnr = cfc_psnr(frame_a, frame_b)?;
    let ssim = cfc_ssim(frame_a, frame_b)?;
    let mean_warp_error = cfc_mae(frame_a, frame_b)?;

    let n_pixels = frame_a.n_pixels();
    let mut occluded = 0u32;
    for i in 0..n_pixels {
        let base = i * 3;
        let err = (frame_a.pixels[base] - frame_b.pixels[base]).abs()
            + (frame_a.pixels[base + 1] - frame_b.pixels[base + 1]).abs()
            + (frame_a.pixels[base + 2] - frame_b.pixels[base + 2]).abs();
        if err / 3.0 > 0.1 {
            occluded += 1;
        }
    }
    let occlusion_ratio = occluded as f32 / n_pixels as f32;

    Ok(FramePairConsistency {
        psnr,
        ssim,
        mean_warp_error,
        mean_flow_magnitude: 0.0,
        occlusion_ratio,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Sequence-level metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Overall temporal consistency report for a frame sequence.
pub struct SequenceConsistencyReport {
    /// Number of frames in the sequence.
    pub n_frames: usize,
    /// Mean PSNR (dB) across all consecutive frame pairs.
    pub mean_psnr: f32,
    /// Mean SSIM across all consecutive frame pairs.
    pub mean_ssim: f32,
    /// Mean warp error across all consecutive frame pairs.
    pub mean_warp_error: f32,
    /// Variance of per-pair PSNR values (measures temporal stability).
    pub temporal_variance: f32,
    /// Index (0-based, into the pairs array) of the most inconsistent pair.
    pub worst_frame_pair: usize,
    /// Index (0-based, into the pairs array) of the most consistent pair.
    pub best_frame_pair: usize,
    /// Linear regression slope on per-pair PSNR over time (positive = improving).
    pub psnr_trend: f32,
    /// Per-consecutive-pair PSNR values (length = `n_frames - 1`).
    pub per_pair_psnr: Vec<f32>,
    /// Per-consecutive-pair SSIM values (length = `n_frames - 1`).
    pub per_pair_ssim: Vec<f32>,
}

/// Linear regression slope for a sequence of values.
///
/// Returns 0 when fewer than 2 points are present.
fn linear_slope(values: &[f32]) -> f32 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f32;
    let sum_x: f32 = (0..n).map(|i| i as f32).sum();
    let sum_y: f32 = values.iter().copied().sum();
    let sum_xy: f32 = values.iter().enumerate().map(|(i, &y)| i as f32 * y).sum();
    let sum_x2: f32 = (0..n).map(|i| (i * i) as f32).sum();
    let denom = nf * sum_x2 - sum_x * sum_x;
    if denom == 0.0 {
        return 0.0;
    }
    (nf * sum_xy - sum_x * sum_y) / denom
}

/// Compute temporal consistency metrics for a sequence of frames.
///
/// # Arguments
/// - `frames`: slice of frames in temporal order (must have ≥ 2 frames).
/// - `use_optical_flow`: when `true`, uses Horn-Schunck to warp frames before
///   measuring error; when `false`, uses direct frame differences (faster).
/// - `flow_config`: parameters for optical flow estimation.
///
/// # Errors
/// - [`ConsistencyError::EmptySequence`] when the sequence has < 2 frames.
/// - Propagates per-pair metric errors.
pub fn cfc_sequence_consistency(
    frames: &[Frame],
    use_optical_flow: bool,
    flow_config: &FlowConfig,
) -> Result<SequenceConsistencyReport, ConsistencyError> {
    if frames.len() < 2 {
        if frames.is_empty() {
            return Err(ConsistencyError::EmptySequence);
        }
        return Err(ConsistencyError::TooShort {
            needed: 2,
            got: frames.len(),
        });
    }

    if use_optical_flow && flow_config.scale == 0 {
        return Err(ConsistencyError::InvalidConfig(
            "FlowConfig::scale must be >= 1".to_string(),
        ));
    }

    let n_pairs = frames.len() - 1;
    let mut per_pair_psnr = Vec::with_capacity(n_pairs);
    let mut per_pair_ssim = Vec::with_capacity(n_pairs);
    let mut per_pair_warp_error = Vec::with_capacity(n_pairs);

    // Grayscale planes are needed by both the optical-flow and SSIM paths;
    // precompute each frame's plane once and reuse it across the (up to
    // two) consecutive pairs it appears in, instead of recomputing it
    // several times per pair inside cfc_compute_flow/cfc_ssim (only the
    // pair-specific warped frame, which cannot be precomputed, still needs
    // a fresh conversion).
    let greys: Vec<Vec<f32>> = frames.iter().map(cfc_to_grayscale).collect();

    for i in 0..n_pairs {
        check_dims(&frames[i], &frames[i + 1])?;
        let (psnr, ssim, warp_error) = if use_optical_flow {
            let (fx, fy) = cfc_compute_flow_from_grayscale(
                &greys[i],
                &greys[i + 1],
                frames[i].width,
                frames[i].height,
                flow_config,
            )?;
            // Negate: cfc_compute_flow(_from_grayscale) returns the forward
            // flow, but cfc_warp_frame needs the negated flow to
            // backward-warp frames[i] toward frames[i + 1] (see
            // `negate_flow`).
            let (neg_fx, neg_fy) = negate_flow(&fx, &fy);
            let warped = cfc_warp_frame(&frames[i], &neg_fx, &neg_fy)?;
            let warped_grey = cfc_to_grayscale(&warped);
            let psnr = cfc_psnr(&warped, &frames[i + 1])?;
            let ssim = cfc_ssim_from_grayscale(
                &warped_grey,
                &greys[i + 1],
                frames[i].width,
                frames[i].height,
            );
            let warp_error = cfc_mae(&warped, &frames[i + 1])?;
            (psnr, ssim, warp_error)
        } else {
            let psnr = cfc_psnr(&frames[i], &frames[i + 1])?;
            let ssim = cfc_ssim_from_grayscale(
                &greys[i],
                &greys[i + 1],
                frames[i].width,
                frames[i].height,
            );
            let warp_error = cfc_mae(&frames[i], &frames[i + 1])?;
            (psnr, ssim, warp_error)
        };
        per_pair_psnr.push(psnr);
        per_pair_ssim.push(ssim);
        per_pair_warp_error.push(warp_error);
    }

    // Aggregate. `finite_psnr` (infinities filtered out, treating inf as a
    // perfect/unbounded match) is built once and shared by the mean and the
    // variance below instead of being collected twice.
    let finite_psnr: Vec<f32> = per_pair_psnr
        .iter()
        .filter(|v| v.is_finite())
        .cloned()
        .collect();
    let mean_psnr = if finite_psnr.is_empty() {
        f32::INFINITY
    } else {
        finite_psnr.iter().copied().sum::<f32>() / finite_psnr.len() as f32
    };
    let mean_ssim = per_pair_ssim.iter().copied().sum::<f32>() / n_pairs as f32;
    let mean_warp_error = per_pair_warp_error.iter().copied().sum::<f32>() / n_pairs as f32;

    // Variance of PSNR (use finite values); `mean_psnr` above is exactly
    // the mean of `finite_psnr` whenever it is non-empty, which is
    // guaranteed here since `finite_psnr.len() >= 2 > 0`.
    let temporal_variance = if finite_psnr.len() >= 2 {
        finite_psnr
            .iter()
            .map(|&v| (v - mean_psnr) * (v - mean_psnr))
            .sum::<f32>()
            / finite_psnr.len() as f32
    } else {
        0.0
    };

    // Worst / best pair (by PSNR; treat inf as very large)
    let to_sortable = |v: f32| if v.is_infinite() { f32::MAX } else { v };
    let worst_frame_pair = per_pair_psnr
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            to_sortable(**a)
                .partial_cmp(&to_sortable(**b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let best_frame_pair = per_pair_psnr
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            to_sortable(**a)
                .partial_cmp(&to_sortable(**b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let psnr_trend = linear_slope(&finite_psnr);

    Ok(SequenceConsistencyReport {
        n_frames: frames.len(),
        mean_psnr,
        mean_ssim,
        mean_warp_error,
        temporal_variance,
        worst_frame_pair,
        best_frame_pair,
        psnr_trend,
        per_pair_psnr,
        per_pair_ssim,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Consistency loss
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the temporal consistency training loss.
#[derive(Debug, Clone)]
pub struct ConsistencyLossConfig {
    /// Weight for the PSNR-derived penalty term (default 0.0). Implemented
    /// as `psnr_weight * mean_pair_mse`: minimizing MSE is equivalent to
    /// maximizing PSNR (`PSNR = -10·log10(MSE)`), without the `-∞` blow-up
    /// a raw `-PSNR` term would produce for identical frames.
    pub psnr_weight: f32,
    /// Weight for L1 frame-difference term (default 1.0).
    pub l1_weight: f32,
    /// Weight for flow-warped difference term (default 0.5).
    pub warp_weight: f32,
    /// Weight for second-order temporal smoothness term (default 0.1).
    pub temporal_smooth_weight: f32,
    /// When `true`, compute optical flow for the warp term (expensive).
    pub use_optical_flow: bool,
}

impl Default for ConsistencyLossConfig {
    fn default() -> Self {
        ConsistencyLossConfig {
            psnr_weight: 0.0,
            l1_weight: 1.0,
            warp_weight: 0.5,
            temporal_smooth_weight: 0.1,
            use_optical_flow: false,
        }
    }
}

/// Decomposed temporal consistency loss value.
pub struct ConsistencyLoss {
    /// Weighted sum of all enabled loss terms.
    pub total: f32,
    /// L1 temporal difference term.
    pub l1_term: f32,
    /// Flow-warped difference term.
    pub warp_term: f32,
    /// Second-order temporal smoothness term.
    pub smooth_term: f32,
    /// PSNR-derived penalty term (mean per-pair MSE; see
    /// [`ConsistencyLossConfig::psnr_weight`]).
    pub psnr_term: f32,
}

/// Mean L1 distance between the pixel buffers of two frames.
///
/// Assumes frames have the same size (unchecked in hot path).
fn mean_l1_frames(a: &Frame, b: &Frame) -> f32 {
    let n = a.pixels.len() as f32;
    a.pixels
        .iter()
        .zip(b.pixels.iter())
        .map(|(&x, &y)| (x - y).abs())
        .sum::<f32>()
        / n
}

/// Compute the temporal consistency loss for a sequence of frames.
///
/// # Loss formula
/// ```text
/// loss = l1_weight        * mean(|f_t - f_{t-1}|)
///      + warp_weight      * mean(|warp(f_{t-1}) - f_t|)     [0 unless use_optical_flow]
///      + smooth_weight    * mean(|f_t - 2·f_{t-1} + f_{t-2}|)
///      + psnr_weight      * mean(MSE(f_t, f_{t-1}))
/// ```
///
/// The `warp` term requires `config.use_optical_flow == true`; otherwise it
/// is exactly `0.0` regardless of `warp_weight`, since there is no
/// motion-compensated frame to compare without flow. The `psnr` term is the
/// raw per-pair MSE rather than `-10·log10(MSE)`, so minimizing the loss is
/// equivalent to maximizing PSNR without a `-∞` blow-up when frames are
/// identical.
///
/// # Errors
/// - [`ConsistencyError::TooShort`] when `frames.len() < 2` for L1/warp terms,
///   or `< 3` when the smooth term weight is nonzero.
pub fn cfc_consistency_loss(
    frames: &[Frame],
    config: &ConsistencyLossConfig,
    flow_config: &FlowConfig,
) -> Result<ConsistencyLoss, ConsistencyError> {
    if frames.len() < 2 {
        return Err(ConsistencyError::TooShort {
            needed: 2,
            got: frames.len(),
        });
    }

    // Validate weights
    for &w in &[
        config.psnr_weight,
        config.l1_weight,
        config.warp_weight,
        config.temporal_smooth_weight,
    ] {
        if !w.is_finite() || w < 0.0 {
            return Err(ConsistencyError::InvalidWeight(w));
        }
    }

    let n_pairs = frames.len() - 1;

    // ── L1 term ──────────────────────────────────────────────────────────────
    let mut l1_sum = 0.0_f32;
    for i in 0..n_pairs {
        check_dims(&frames[i], &frames[i + 1])?;
        l1_sum += mean_l1_frames(&frames[i], &frames[i + 1]);
    }
    let l1_term = l1_sum / n_pairs as f32;

    // ── Warp term ─────────────────────────────────────────────────────────────
    // Only meaningful when optical flow is actually enabled: without it,
    // there is no motion-compensated comparison to make, so the term is 0
    // regardless of `warp_weight` rather than silently degenerating into a
    // duplicate of `l1_term` (which used to make `warp_weight` secretly
    // re-scale the L1 term instead of doing nothing).
    let warp_term = if config.warp_weight > 0.0 && config.use_optical_flow {
        let mut warp_sum = 0.0_f32;
        for i in 0..n_pairs {
            let (fx, fy) = cfc_compute_flow(&frames[i], &frames[i + 1], flow_config)?;
            // Negate: cfc_compute_flow returns the forward flow, but
            // cfc_warp_frame needs the negated flow to backward-warp
            // frames[i] toward frames[i + 1] (see `negate_flow`).
            let (neg_fx, neg_fy) = negate_flow(&fx, &fy);
            let warped = cfc_warp_frame(&frames[i], &neg_fx, &neg_fy)?;
            warp_sum += mean_l1_frames(&warped, &frames[i + 1]);
        }
        warp_sum / n_pairs as f32
    } else {
        0.0
    };

    // ── Smooth term (second-order) ────────────────────────────────────────────
    let smooth_term = if config.temporal_smooth_weight > 0.0 {
        if frames.len() < 3 {
            return Err(ConsistencyError::TooShort {
                needed: 3,
                got: frames.len(),
            });
        }
        let n_triples = frames.len() - 2;
        let mut smooth_sum = 0.0_f32;
        for i in 0..n_triples {
            let a = &frames[i];
            let b = &frames[i + 1];
            let c = &frames[i + 2];
            check_dims(a, b)?;
            check_dims(b, c)?;
            let n = a.pixels.len() as f32;
            let triple_err: f32 = a
                .pixels
                .iter()
                .zip(b.pixels.iter().zip(c.pixels.iter()))
                .map(|(&fa, (&fb, &fc))| (fc - 2.0 * fb + fa).abs())
                .sum::<f32>()
                / n;
            smooth_sum += triple_err;
        }
        smooth_sum / n_triples as f32
    } else {
        0.0
    };

    // ── PSNR-derived term ───────────────────────────────────────────────────
    let psnr_term = if config.psnr_weight > 0.0 {
        let mut mse_sum = 0.0_f32;
        for i in 0..n_pairs {
            check_dims(&frames[i], &frames[i + 1])?;
            mse_sum += frame_mse(&frames[i], &frames[i + 1]);
        }
        mse_sum / n_pairs as f32
    } else {
        0.0
    };

    let total = config.l1_weight * l1_term
        + config.warp_weight * warp_term
        + config.temporal_smooth_weight * smooth_term
        + config.psnr_weight * psnr_term;

    Ok(ConsistencyLoss {
        total,
        l1_term,
        warp_term,
        smooth_term,
        psnr_term,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Format a [`FramePairConsistency`] as a human-readable string.
pub fn cfc_format_pair_consistency(c: &FramePairConsistency) -> String {
    format!(
        "FramePair {{ psnr: {:.2} dB, ssim: {:.4}, warp_err: {:.6}, \
         flow_mag: {:.3} px, occlusion: {:.2}% }}",
        c.psnr,
        c.ssim,
        c.mean_warp_error,
        c.mean_flow_magnitude,
        c.occlusion_ratio * 100.0
    )
}

/// Format a [`SequenceConsistencyReport`] as a human-readable summary.
pub fn cfc_format_report(report: &SequenceConsistencyReport) -> String {
    format!(
        "SequenceConsistency {{ frames: {}, mean PSNR: {:.2} dB, mean SSIM: {:.4}, \
         mean_warp_err: {:.6}, temporal_variance: {:.4}, \
         worst_pair: {}, best_pair: {}, PSNR trend: {:.4}/frame }}",
        report.n_frames,
        report.mean_psnr,
        report.mean_ssim,
        report.mean_warp_error,
        report.temporal_variance,
        report.worst_frame_pair,
        report.best_frame_pair,
        report.psnr_trend
    )
}

/// Format a [`ConsistencyLoss`] as a human-readable string.
pub fn cfc_format_loss(loss: &ConsistencyLoss) -> String {
    format!(
        "ConsistencyLoss {{ total: {:.6}, l1: {:.6}, warp: {:.6}, smooth: {:.6}, psnr: {:.6} }}",
        loss.total, loss.l1_term, loss.warp_term, loss.smooth_term, loss.psnr_term
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "cross_frame_consistency_tests.rs"]
mod tests;
