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
pub struct Frame {
    /// Interleaved RGB values; length must equal `width * height * 3`.
    pub pixels: Vec<f32>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
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
/// Out-of-range coordinates are clamped to the frame boundary.
pub fn cfc_bilinear_sample(frame: &Frame, x: f32, y: f32) -> [f32; 3] {
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

    if config.scale == 0 {
        return Err(ConsistencyError::InvalidConfig(
            "FlowConfig::scale must be >= 1".to_string(),
        ));
    }

    let grey_a = cfc_to_grayscale(frame_a);
    let grey_b = cfc_to_grayscale(frame_b);

    let (ds_a, dw, dh) = downscale_grey(&grey_a, frame_a.width, frame_a.height, config.scale);
    let (ds_b, _, _) = downscale_grey(&grey_b, frame_b.width, frame_b.height, config.scale);

    let n = dw * dh;
    let mut u = vec![0.0_f32; n]; // flow_x at downscaled res
    let mut v = vec![0.0_f32; n]; // flow_y at downscaled res

    let alpha_sq = config.alpha * config.alpha;

    for _iter in 0..config.n_iterations {
        let u_prev = u.clone();
        let v_prev = v.clone();

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
    let full_u = upscale_flow(&u, dw, dh, frame_a.width, frame_a.height, factor);
    let full_v = upscale_flow(&v, dw, dh, frame_a.width, frame_a.height, factor);

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

    const WIN: usize = 8;
    const C1: f32 = 0.0001;
    const C2: f32 = 0.0009;

    let w = frame_a.width;
    let h = frame_a.height;

    let mut ssim_sum = 0.0_f32;
    let mut n_windows = 0u32;

    // Work on luminance channel only for SSIM
    let la = cfc_to_grayscale(frame_a);
    let lb = cfc_to_grayscale(frame_b);

    let mut wy = 0;
    while wy + WIN <= h {
        let mut wx = 0;
        while wx + WIN <= w {
            let mut sum_a = 0.0_f32;
            let mut sum_b = 0.0_f32;
            let mut sum_aa = 0.0_f32;
            let mut sum_bb = 0.0_f32;
            let mut sum_ab = 0.0_f32;
            let count = (WIN * WIN) as f32;

            for ry in 0..WIN {
                for rx in 0..WIN {
                    let idx = (wy + ry) * w + (wx + rx);
                    let a = la[idx];
                    let b = lb[idx];
                    sum_a += a;
                    sum_b += b;
                    sum_aa += a * a;
                    sum_bb += b * b;
                    sum_ab += a * b;
                }
            }

            let mu_a = sum_a / count;
            let mu_b = sum_b / count;
            let sig_aa = (sum_aa / count) - mu_a * mu_a;
            let sig_bb = (sum_bb / count) - mu_b * mu_b;
            let sig_ab = (sum_ab / count) - mu_a * mu_b;

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
            la.iter().map(|&a| (a - mu_a) * (a - mu_a)).sum::<f32>() / la.len() as f32;
        let sig_bb: f32 =
            lb.iter().map(|&b| (b - mu_b) * (b - mu_b)).sum::<f32>() / lb.len() as f32;
        let num = (2.0 * mu_a * mu_b + C1) * (2.0 * sig_ab + C2);
        let den = (mu_a * mu_a + mu_b * mu_b + C1) * (sig_aa + sig_bb + C2);
        return Ok(num / den);
    }

    Ok(ssim_sum / n_windows as f32)
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
    let warped = cfc_warp_frame(frame_a, &fx, &fy)?;

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

    let n_pairs = frames.len() - 1;
    let mut per_pair_psnr = Vec::with_capacity(n_pairs);
    let mut per_pair_ssim = Vec::with_capacity(n_pairs);
    let mut per_pair_warp_error = Vec::with_capacity(n_pairs);

    for i in 0..n_pairs {
        let pair = if use_optical_flow {
            cfc_frame_pair_consistency(&frames[i], &frames[i + 1], flow_config)?
        } else {
            cfc_frame_difference(&frames[i], &frames[i + 1])?
        };
        per_pair_psnr.push(pair.psnr);
        per_pair_ssim.push(pair.ssim);
        per_pair_warp_error.push(pair.mean_warp_error);
    }

    // Aggregate
    let mean_psnr = {
        // Filter out infinities for mean (treat inf as max finite value)
        let finite: Vec<f32> = per_pair_psnr
            .iter()
            .filter(|v| v.is_finite())
            .cloned()
            .collect();
        if finite.is_empty() {
            f32::INFINITY
        } else {
            finite.iter().copied().sum::<f32>() / finite.len() as f32
        }
    };
    let mean_ssim = per_pair_ssim.iter().copied().sum::<f32>() / n_pairs as f32;
    let mean_warp_error = per_pair_warp_error.iter().copied().sum::<f32>() / n_pairs as f32;

    // Variance of PSNR (use finite values)
    let finite_psnr: Vec<f32> = per_pair_psnr
        .iter()
        .filter(|v| v.is_finite())
        .cloned()
        .collect();
    let temporal_variance = if finite_psnr.len() >= 2 {
        let m = finite_psnr.iter().copied().sum::<f32>() / finite_psnr.len() as f32;
        finite_psnr.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / finite_psnr.len() as f32
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
    /// Weight for PSNR-derived penalty (default 0.0; rarely used directly in loss).
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
///      + warp_weight      * mean(|warp(f_{t-1}) - f_t|)
///      + smooth_weight    * mean(|f_t - 2·f_{t-1} + f_{t-2}|)
/// ```
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
    let warp_term = if config.warp_weight > 0.0 {
        let mut warp_sum = 0.0_f32;
        for i in 0..n_pairs {
            let warp_err = if config.use_optical_flow {
                let (fx, fy) = cfc_compute_flow(&frames[i], &frames[i + 1], flow_config)?;
                let warped = cfc_warp_frame(&frames[i], &fx, &fy)?;
                mean_l1_frames(&warped, &frames[i + 1])
            } else {
                // Without flow, warp term degenerates to L1
                mean_l1_frames(&frames[i], &frames[i + 1])
            };
            warp_sum += warp_err;
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

    let total = config.l1_weight * l1_term
        + config.warp_weight * warp_term
        + config.temporal_smooth_weight * smooth_term;

    Ok(ConsistencyLoss {
        total,
        l1_term,
        warp_term,
        smooth_term,
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
        "ConsistencyLoss {{ total: {:.6}, l1: {:.6}, warp: {:.6}, smooth: {:.6} }}",
        loss.total, loss.l1_term, loss.warp_term, loss.smooth_term
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn uniform_frame(w: usize, h: usize, r: f32, g: f32, b: f32) -> Frame {
        let pixels: Vec<f32> = (0..w * h).flat_map(|_| [r, g, b]).collect();
        Frame {
            pixels,
            width: w,
            height: h,
        }
    }

    fn checkerboard_frame(w: usize, h: usize) -> Frame {
        let mut pixels = Vec::with_capacity(w * h * 3);
        for row in 0..h {
            for col in 0..w {
                let v = if (row + col) % 2 == 0 {
                    0.0_f32
                } else {
                    1.0_f32
                };
                pixels.push(v);
                pixels.push(v);
                pixels.push(v);
            }
        }
        Frame {
            pixels,
            width: w,
            height: h,
        }
    }

    fn gradient_frame(w: usize, h: usize) -> Frame {
        let mut pixels = Vec::with_capacity(w * h * 3);
        for row in 0..h {
            for col in 0..w {
                let v = col as f32 / w.max(1) as f32;
                let u = row as f32 / h.max(1) as f32;
                pixels.push(v);
                pixels.push(u);
                pixels.push(0.5_f32);
            }
        }
        Frame {
            pixels,
            width: w,
            height: h,
        }
    }

    // ── Frame construction ────────────────────────────────────────────────────

    #[test]
    fn test_frame_new_all_zeros() {
        let f = Frame::new(4, 4);
        assert!(f.pixels.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_frame_new_correct_dimensions() {
        let f = Frame::new(8, 6);
        assert_eq!(f.width, 8);
        assert_eq!(f.height, 6);
        assert_eq!(f.pixels.len(), 8 * 6 * 3);
    }

    #[test]
    fn test_frame_n_pixels() {
        let f = Frame::new(10, 5);
        assert_eq!(f.n_pixels(), 50);
    }

    #[test]
    fn test_frame_from_pixels_ok() -> Result<(), ConsistencyError> {
        let pixels = vec![0.5_f32; 4 * 4 * 3];
        let f = Frame::from_pixels(pixels, 4, 4)?;
        assert_eq!(f.width, 4);
        assert_eq!(f.height, 4);
        Ok(())
    }

    #[test]
    fn test_frame_from_pixels_wrong_size_error() {
        let pixels = vec![0.0_f32; 10]; // wrong length
        assert!(matches!(
            Frame::from_pixels(pixels, 4, 4),
            Err(ConsistencyError::InvalidConfig(_))
        ));
    }

    // ── pixel_at ─────────────────────────────────────────────────────────────

    #[test]
    fn test_pixel_at_correct_indexing() -> Result<(), ConsistencyError> {
        let f = uniform_frame(4, 4, 0.1, 0.2, 0.3);
        let px = f.pixel_at(2, 1)?;
        assert!((px[0] - 0.1).abs() < 1e-6);
        assert!((px[1] - 0.2).abs() < 1e-6);
        assert!((px[2] - 0.3).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_pixel_at_out_of_bounds_error() {
        let f = Frame::new(4, 4);
        assert!(matches!(
            f.pixel_at(10, 0),
            Err(ConsistencyError::InvalidConfig(_))
        ));
        assert!(matches!(
            f.pixel_at(0, 10),
            Err(ConsistencyError::InvalidConfig(_))
        ));
    }

    // ── mean_brightness ───────────────────────────────────────────────────────

    #[test]
    fn test_mean_brightness_zero_frame() {
        let f = Frame::new(8, 8);
        assert_eq!(f.mean_brightness(), 0.0);
    }

    #[test]
    fn test_mean_brightness_uniform() {
        let f = uniform_frame(4, 4, 0.5, 0.5, 0.5);
        assert!((f.mean_brightness() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_mean_brightness_half_white() {
        // Half pixels at 1.0, half at 0.0 (RGB all same)
        let mut pixels = vec![0.0_f32; 4 * 4 * 3];
        for p in pixels.iter_mut().take(8 * 3) {
            *p = 1.0;
        }
        let f = Frame {
            pixels,
            width: 4,
            height: 4,
        };
        let mb = f.mean_brightness();
        assert!((mb - 0.5).abs() < 1e-5, "got {mb}");
    }

    // ── variance ─────────────────────────────────────────────────────────────

    #[test]
    fn test_variance_constant_frame_zero() {
        let f = uniform_frame(8, 8, 0.5, 0.5, 0.5);
        assert!(f.variance().abs() < 1e-8);
    }

    #[test]
    fn test_variance_varying_frame_positive() {
        let f = checkerboard_frame(8, 8);
        assert!(f.variance() > 0.0);
    }

    // ── cfc_to_grayscale ──────────────────────────────────────────────────────

    #[test]
    fn test_to_grayscale_white() {
        let f = uniform_frame(4, 4, 1.0, 1.0, 1.0);
        let g = cfc_to_grayscale(&f);
        assert_eq!(g.len(), 16);
        for &v in &g {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_to_grayscale_black() {
        let f = Frame::new(4, 4);
        let g = cfc_to_grayscale(&f);
        for &v in &g {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_to_grayscale_length() {
        let f = Frame::new(6, 7);
        assert_eq!(cfc_to_grayscale(&f).len(), 6 * 7);
    }

    #[test]
    fn test_to_grayscale_known_value() {
        // Pure red pixel → 0.2126
        let f = uniform_frame(1, 1, 1.0, 0.0, 0.0);
        let g = cfc_to_grayscale(&f);
        assert!((g[0] - 0.2126).abs() < 1e-5, "got {}", g[0]);
    }

    // ── cfc_bilinear_sample ───────────────────────────────────────────────────

    #[test]
    fn test_bilinear_exact_pixel() {
        let f = uniform_frame(4, 4, 0.3, 0.6, 0.9);
        let rgb = cfc_bilinear_sample(&f, 2.0, 1.0);
        assert!((rgb[0] - 0.3).abs() < 1e-5);
        assert!((rgb[1] - 0.6).abs() < 1e-5);
        assert!((rgb[2] - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_bilinear_clamp_out_of_bounds() {
        let f = uniform_frame(4, 4, 0.7, 0.8, 0.9);
        // Coordinates way outside → clamped to border value
        let rgb = cfc_bilinear_sample(&f, -5.0, 100.0);
        assert!((rgb[0] - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_bilinear_midpoint_uniform() {
        // A uniform frame → any fractional coord returns the same color
        let f = uniform_frame(8, 8, 0.4, 0.5, 0.6);
        let rgb = cfc_bilinear_sample(&f, 3.7, 2.3);
        assert!((rgb[0] - 0.4).abs() < 1e-5);
        assert!((rgb[1] - 0.5).abs() < 1e-5);
        assert!((rgb[2] - 0.6).abs() < 1e-5);
    }

    // ── cfc_compute_flow ──────────────────────────────────────────────────────

    #[test]
    fn test_flow_identical_frames_near_zero() -> Result<(), ConsistencyError> {
        let f = checkerboard_frame(16, 16);
        let config = FlowConfig {
            n_iterations: 5,
            ..Default::default()
        };
        let (fx, fy) = cfc_compute_flow(&f, &f, &config)?;
        let max_u: f32 = fx.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_v: f32 = fy.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_u.abs() < 5.0,
            "large flow_x for identical frames: {max_u}"
        );
        assert!(
            max_v.abs() < 5.0,
            "large flow_y for identical frames: {max_v}"
        );
        Ok(())
    }

    #[test]
    fn test_flow_dimension_mismatch_error() {
        let a = Frame::new(8, 8);
        let b = Frame::new(4, 4);
        assert!(matches!(
            cfc_compute_flow(&a, &b, &FlowConfig::default()),
            Err(ConsistencyError::FrameDimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_flow_output_length() -> Result<(), ConsistencyError> {
        let f = Frame::new(8, 8);
        let config = FlowConfig {
            n_iterations: 2,
            ..Default::default()
        };
        let (fx, fy) = cfc_compute_flow(&f, &f, &config)?;
        assert_eq!(fx.len(), 64);
        assert_eq!(fy.len(), 64);
        Ok(())
    }

    // ── cfc_warp_frame ────────────────────────────────────────────────────────

    #[test]
    fn test_warp_zero_flow_same_frame() -> Result<(), ConsistencyError> {
        let f = gradient_frame(8, 8);
        let flow = vec![0.0_f32; 64];
        let warped = cfc_warp_frame(&f, &flow, &flow)?;
        for (a, b) in f.pixels.iter().zip(warped.pixels.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "zero-flow warp changed pixel: {a} vs {b}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_warp_wrong_flow_length_error() {
        let f = Frame::new(4, 4);
        let bad_flow = vec![0.0_f32; 5]; // wrong length
        assert!(matches!(
            cfc_warp_frame(&f, &bad_flow, &bad_flow),
            Err(ConsistencyError::InvalidConfig(_))
        ));
    }

    // ── cfc_psnr ─────────────────────────────────────────────────────────────

    #[test]
    fn test_psnr_identical_frames_infinity() -> Result<(), ConsistencyError> {
        let f = uniform_frame(8, 8, 0.5, 0.5, 0.5);
        let psnr = cfc_psnr(&f, &f)?;
        assert_eq!(psnr, f32::INFINITY);
        Ok(())
    }

    #[test]
    fn test_psnr_different_frames_finite() -> Result<(), ConsistencyError> {
        let a = Frame::new(8, 8);
        let b = uniform_frame(8, 8, 1.0, 1.0, 1.0);
        let psnr = cfc_psnr(&a, &b)?;
        assert!(psnr.is_finite());
        // MSE = 1.0, PSNR = 10*log10(1/1) = 0 dB
        assert!((psnr - 0.0).abs() < 1e-4, "got {psnr}");
        Ok(())
    }

    #[test]
    fn test_psnr_dimension_mismatch() {
        let a = Frame::new(4, 4);
        let b = Frame::new(8, 8);
        assert!(matches!(
            cfc_psnr(&a, &b),
            Err(ConsistencyError::FrameDimensionMismatch { .. })
        ));
    }

    // ── cfc_ssim ─────────────────────────────────────────────────────────────

    #[test]
    fn test_ssim_identical_near_one() -> Result<(), ConsistencyError> {
        let f = checkerboard_frame(16, 16);
        let s = cfc_ssim(&f, &f)?;
        assert!(
            s > 0.99,
            "SSIM of identical frames should be near 1.0, got {s}"
        );
        Ok(())
    }

    #[test]
    fn test_ssim_very_different_less_than_identical() -> Result<(), ConsistencyError> {
        let a = Frame::new(16, 16);
        let b = uniform_frame(16, 16, 1.0, 1.0, 1.0);
        let sa = cfc_ssim(&a, &a)?;
        let sab = cfc_ssim(&a, &b)?;
        assert!(
            sab < sa,
            "SSIM of different frames should be lower: {sab} vs {sa}"
        );
        Ok(())
    }

    #[test]
    fn test_ssim_dimension_mismatch() {
        let a = Frame::new(4, 4);
        let b = Frame::new(8, 8);
        assert!(matches!(
            cfc_ssim(&a, &b),
            Err(ConsistencyError::FrameDimensionMismatch { .. })
        ));
    }

    // ── cfc_mae ───────────────────────────────────────────────────────────────

    #[test]
    fn test_mae_identical_zero() -> Result<(), ConsistencyError> {
        let f = checkerboard_frame(8, 8);
        assert_eq!(cfc_mae(&f, &f)?, 0.0);
        Ok(())
    }

    #[test]
    fn test_mae_known_difference() -> Result<(), ConsistencyError> {
        let a = Frame::new(4, 4); // all 0
        let b = uniform_frame(4, 4, 0.5, 0.5, 0.5); // all 0.5
        let mae = cfc_mae(&a, &b)?;
        assert!((mae - 0.5).abs() < 1e-6, "got {mae}");
        Ok(())
    }

    #[test]
    fn test_mae_dimension_mismatch() {
        let a = Frame::new(4, 4);
        let b = Frame::new(8, 8);
        assert!(matches!(
            cfc_mae(&a, &b),
            Err(ConsistencyError::FrameDimensionMismatch { .. })
        ));
    }

    // ── cfc_rmse ──────────────────────────────────────────────────────────────

    #[test]
    fn test_rmse_identical_zero() -> Result<(), ConsistencyError> {
        let f = gradient_frame(8, 8);
        assert_eq!(cfc_rmse(&f, &f)?, 0.0);
        Ok(())
    }

    #[test]
    fn test_rmse_known_value() -> Result<(), ConsistencyError> {
        let a = Frame::new(4, 4); // all 0
        let b = uniform_frame(4, 4, 1.0, 1.0, 1.0); // all 1
        let rmse = cfc_rmse(&a, &b)?;
        assert!((rmse - 1.0).abs() < 1e-6, "got {rmse}");
        Ok(())
    }

    // ── cfc_frame_difference ──────────────────────────────────────────────────

    #[test]
    fn test_frame_difference_identical_zero_warp_error() -> Result<(), ConsistencyError> {
        let f = checkerboard_frame(8, 8);
        let c = cfc_frame_difference(&f, &f)?;
        assert_eq!(c.mean_warp_error, 0.0);
        assert_eq!(c.psnr, f32::INFINITY);
        Ok(())
    }

    #[test]
    fn test_frame_difference_dimension_mismatch() {
        let a = Frame::new(4, 4);
        let b = Frame::new(8, 8);
        assert!(matches!(
            cfc_frame_difference(&a, &b),
            Err(ConsistencyError::FrameDimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_frame_difference_ssim_range() -> Result<(), ConsistencyError> {
        let a = uniform_frame(16, 16, 0.2, 0.3, 0.4);
        let b = uniform_frame(16, 16, 0.5, 0.6, 0.7);
        let c = cfc_frame_difference(&a, &b)?;
        assert!(
            c.ssim >= -1.0 && c.ssim <= 1.0 + 1e-5,
            "ssim out of range: {}",
            c.ssim
        );
        Ok(())
    }

    // ── cfc_frame_pair_consistency ────────────────────────────────────────────

    #[test]
    fn test_frame_pair_consistency_identical_low_error() -> Result<(), ConsistencyError> {
        let f = gradient_frame(16, 16);
        let config = FlowConfig {
            n_iterations: 3,
            ..Default::default()
        };
        let c = cfc_frame_pair_consistency(&f, &f, &config)?;
        assert!(
            c.mean_warp_error < 0.05,
            "warp error too large for identical frames: {}",
            c.mean_warp_error
        );
        Ok(())
    }

    #[test]
    fn test_frame_pair_consistency_psnr_finite_or_inf() -> Result<(), ConsistencyError> {
        let a = uniform_frame(16, 16, 0.3, 0.4, 0.5);
        let b = uniform_frame(16, 16, 0.3, 0.4, 0.5);
        let config = FlowConfig {
            n_iterations: 2,
            ..Default::default()
        };
        let c = cfc_frame_pair_consistency(&a, &b, &config)?;
        assert!(c.psnr.is_infinite() || c.psnr > 20.0, "psnr={}", c.psnr);
        Ok(())
    }

    // ── cfc_sequence_consistency ──────────────────────────────────────────────

    #[test]
    fn test_sequence_two_frames_one_pair() -> Result<(), ConsistencyError> {
        let fa = uniform_frame(8, 8, 0.3, 0.3, 0.3);
        let fb = uniform_frame(8, 8, 0.4, 0.4, 0.4);
        let report = cfc_sequence_consistency(&[fa, fb], false, &FlowConfig::default())?;
        assert_eq!(report.per_pair_psnr.len(), 1);
        assert_eq!(report.per_pair_ssim.len(), 1);
        Ok(())
    }

    #[test]
    fn test_sequence_empty_error() {
        let result = cfc_sequence_consistency(&[], false, &FlowConfig::default());
        assert!(matches!(result, Err(ConsistencyError::EmptySequence)));
    }

    #[test]
    fn test_sequence_single_frame_error() {
        let f = Frame::new(8, 8);
        let result = cfc_sequence_consistency(&[f], false, &FlowConfig::default());
        assert!(matches!(result, Err(ConsistencyError::TooShort { .. })));
    }

    #[test]
    fn test_sequence_worst_best_valid_indices() -> Result<(), ConsistencyError> {
        let frames: Vec<Frame> = (0..5)
            .map(|i| uniform_frame(8, 8, i as f32 * 0.1, 0.0, 0.0))
            .collect();
        let report = cfc_sequence_consistency(&frames, false, &FlowConfig::default())?;
        assert!(report.worst_frame_pair < frames.len() - 1);
        assert!(report.best_frame_pair < frames.len() - 1);
        Ok(())
    }

    #[test]
    fn test_sequence_n_frames_matches() -> Result<(), ConsistencyError> {
        let frames: Vec<Frame> = (0..4).map(|_| Frame::new(8, 8)).collect();
        let report = cfc_sequence_consistency(&frames, false, &FlowConfig::default())?;
        assert_eq!(report.n_frames, 4);
        assert_eq!(report.per_pair_psnr.len(), 3);
        Ok(())
    }

    #[test]
    fn test_sequence_per_pair_psnr_length() -> Result<(), ConsistencyError> {
        let frames: Vec<Frame> = (0..6).map(|_| gradient_frame(8, 8)).collect();
        let report = cfc_sequence_consistency(&frames, false, &FlowConfig::default())?;
        assert_eq!(report.per_pair_psnr.len(), report.n_frames - 1);
        Ok(())
    }

    // ── psnr_trend ────────────────────────────────────────────────────────────

    #[test]
    fn test_psnr_trend_increasing_quality() -> Result<(), ConsistencyError> {
        // Build a sequence where consecutive-pair diffs shrink, so PSNR rises.
        // Frame values: 0.0, 0.4, 0.7, 0.85, 0.925, 1.0  (each step halves residual)
        // Differences: 0.4, 0.3, 0.15, 0.075, 0.075 — generally decreasing
        // → PSNR trend should be positive.
        let values = [0.0_f32, 0.4, 0.7, 0.85, 0.93, 1.0];
        let frames: Vec<Frame> = values
            .iter()
            .map(|&v| uniform_frame(8, 8, v, 0.0, 0.0))
            .collect();
        let report = cfc_sequence_consistency(&frames, false, &FlowConfig::default())?;
        // Consecutive diffs: 0.4, 0.3, 0.15, 0.08, 0.07 — strictly shrinking
        // PSNR increases monotonically, so trend slope must be positive.
        assert!(
            report.psnr_trend > 0.0,
            "expected positive psnr_trend, got {}",
            report.psnr_trend
        );
        Ok(())
    }

    // ── cfc_consistency_loss ──────────────────────────────────────────────────

    #[test]
    fn test_consistency_loss_constant_sequence_l1_zero() -> Result<(), ConsistencyError> {
        let frames: Vec<Frame> = (0..4).map(|_| uniform_frame(8, 8, 0.5, 0.5, 0.5)).collect();
        let cfg = ConsistencyLossConfig {
            use_optical_flow: false,
            ..Default::default()
        };
        let loss = cfc_consistency_loss(&frames, &cfg, &FlowConfig::default())?;
        assert!(
            loss.l1_term < 1e-6,
            "l1_term should be 0 for constant sequence: {}",
            loss.l1_term
        );
        Ok(())
    }

    #[test]
    fn test_consistency_loss_single_frame_error() {
        let f = Frame::new(8, 8);
        let cfg = ConsistencyLossConfig::default();
        let result = cfc_consistency_loss(&[f], &cfg, &FlowConfig::default());
        assert!(matches!(result, Err(ConsistencyError::TooShort { .. })));
    }

    #[test]
    fn test_consistency_loss_two_frames_smooth_term_error() {
        // With smooth term weight > 0 and only 2 frames → TooShort
        let fa = Frame::new(8, 8);
        let fb = Frame::new(8, 8);
        let cfg = ConsistencyLossConfig {
            temporal_smooth_weight: 0.1,
            ..Default::default()
        };
        let result = cfc_consistency_loss(&[fa, fb], &cfg, &FlowConfig::default());
        assert!(matches!(result, Err(ConsistencyError::TooShort { .. })));
    }

    #[test]
    fn test_consistency_loss_all_l1_weight() -> Result<(), ConsistencyError> {
        let fa = uniform_frame(8, 8, 0.0, 0.0, 0.0);
        let fb = uniform_frame(8, 8, 1.0, 1.0, 1.0);
        let fc = uniform_frame(8, 8, 0.5, 0.5, 0.5);
        let cfg = ConsistencyLossConfig {
            l1_weight: 1.0,
            warp_weight: 0.0,
            temporal_smooth_weight: 0.0,
            ..Default::default()
        };
        let loss = cfc_consistency_loss(&[fa, fb, fc], &cfg, &FlowConfig::default())?;
        assert!(loss.warp_term == 0.0, "warp_term should be 0 when weight=0");
        assert!(
            loss.smooth_term == 0.0,
            "smooth_term should be 0 when weight=0"
        );
        assert!(loss.total > 0.0);
        Ok(())
    }

    #[test]
    fn test_consistency_loss_warp_weight_applied() -> Result<(), ConsistencyError> {
        let fa = uniform_frame(8, 8, 0.0, 0.0, 0.0);
        let fb = uniform_frame(8, 8, 1.0, 1.0, 1.0);
        let fc = uniform_frame(8, 8, 0.5, 0.5, 0.5);
        let cfg_no_warp = ConsistencyLossConfig {
            warp_weight: 0.0,
            temporal_smooth_weight: 0.0,
            ..Default::default()
        };
        let cfg_with_warp = ConsistencyLossConfig {
            warp_weight: 1.0,
            temporal_smooth_weight: 0.0,
            ..Default::default()
        };
        let frames_no: Vec<Frame> = [fa.pixels.clone(), fb.pixels.clone(), fc.pixels.clone()]
            .into_iter()
            .zip([(8usize, 8usize), (8, 8), (8, 8)])
            .map(|(p, (w, h))| Frame {
                pixels: p,
                width: w,
                height: h,
            })
            .collect();
        let loss_no = cfc_consistency_loss(&frames_no, &cfg_no_warp, &FlowConfig::default())?;
        let loss_with =
            cfc_consistency_loss(&[fa, fb, fc], &cfg_with_warp, &FlowConfig::default())?;
        // Both should be positive; with-warp should differ from no-warp
        assert!(loss_no.total > 0.0);
        assert!(loss_with.total > 0.0);
        assert_ne!(loss_no.total, loss_with.total);
        Ok(())
    }

    // ── ConsistencyLossConfig default ─────────────────────────────────────────

    #[test]
    fn test_consistency_loss_config_default() {
        let cfg = ConsistencyLossConfig::default();
        assert_eq!(cfg.psnr_weight, 0.0);
        assert_eq!(cfg.l1_weight, 1.0);
        assert_eq!(cfg.warp_weight, 0.5);
        assert_eq!(cfg.temporal_smooth_weight, 0.1);
        assert!(!cfg.use_optical_flow);
    }

    // ── FlowConfig default ───────────────────────────────────────────────────

    #[test]
    fn test_flow_config_default() {
        let cfg = FlowConfig::default();
        assert_eq!(cfg.alpha, 100.0);
        assert_eq!(cfg.n_iterations, 20);
        assert_eq!(cfg.scale, 2);
    }

    // ── Formatting ────────────────────────────────────────────────────────────

    #[test]
    fn test_format_pair_consistency_nonempty() {
        let c = FramePairConsistency {
            psnr: 30.5,
            ssim: 0.95,
            mean_warp_error: 0.01,
            mean_flow_magnitude: 2.3,
            occlusion_ratio: 0.05,
        };
        let s = cfc_format_pair_consistency(&c);
        assert!(!s.is_empty());
        assert!(s.contains("psnr") || s.contains("PSNR") || s.contains("30.50"));
    }

    #[test]
    fn test_format_report_contains_frames_and_psnr() {
        let report = SequenceConsistencyReport {
            n_frames: 10,
            mean_psnr: 35.0,
            mean_ssim: 0.92,
            mean_warp_error: 0.005,
            temporal_variance: 1.2,
            worst_frame_pair: 2,
            best_frame_pair: 7,
            psnr_trend: 0.3,
            per_pair_psnr: vec![35.0; 9],
            per_pair_ssim: vec![0.92; 9],
        };
        let s = cfc_format_report(&report);
        assert!(s.contains("frames") || s.contains("10"));
        assert!(s.contains("PSNR") || s.contains("35.00") || s.contains("psnr"));
    }

    #[test]
    fn test_format_loss_contains_total() {
        let loss = ConsistencyLoss {
            total: 0.123,
            l1_term: 0.1,
            warp_term: 0.02,
            smooth_term: 0.003,
        };
        let s = cfc_format_loss(&loss);
        assert!(!s.is_empty());
        assert!(s.contains("total") || s.contains("0.123000"));
    }

    // ── Additional edge-case tests ────────────────────────────────────────────

    #[test]
    fn test_warp_uniform_frame_stays_uniform() -> Result<(), ConsistencyError> {
        let f = uniform_frame(8, 8, 0.4, 0.5, 0.6);
        let n = 64;
        let fx: Vec<f32> = (0..n).map(|_| 0.7_f32).collect();
        let fy: Vec<f32> = (0..n).map(|_| 0.3_f32).collect();
        let warped = cfc_warp_frame(&f, &fx, &fy)?;
        // Uniform frame: any shift should return the same color
        for chunk in warped.pixels.chunks(3) {
            assert!((chunk[0] - 0.4).abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn test_grayscale_weighted_correctly() {
        // Pure green → 0.7152
        let f = uniform_frame(1, 1, 0.0, 1.0, 0.0);
        let g = cfc_to_grayscale(&f);
        assert!((g[0] - 0.7152).abs() < 1e-5, "got {}", g[0]);
    }

    #[test]
    fn test_sequence_all_identical_psnr_infinite() -> Result<(), ConsistencyError> {
        let frames: Vec<Frame> = (0..3).map(|_| uniform_frame(8, 8, 0.5, 0.5, 0.5)).collect();
        let report = cfc_sequence_consistency(&frames, false, &FlowConfig::default())?;
        assert_eq!(report.mean_psnr, f32::INFINITY);
        Ok(())
    }

    #[test]
    fn test_frame_difference_zero_occlusion_identical() -> Result<(), ConsistencyError> {
        let f = uniform_frame(16, 16, 0.5, 0.5, 0.5);
        let c = cfc_frame_difference(&f, &f)?;
        assert_eq!(c.occlusion_ratio, 0.0);
        Ok(())
    }

    #[test]
    fn test_consistency_loss_smooth_term_nonzero() -> Result<(), ConsistencyError> {
        let fa = uniform_frame(8, 8, 0.0, 0.0, 0.0);
        let fb = uniform_frame(8, 8, 0.5, 0.5, 0.5);
        let fc = uniform_frame(8, 8, 0.0, 0.0, 0.0);
        // fc - 2*fb + fa = 0 - 1 + 0 = -1; |·| = 1 per channel
        let cfg = ConsistencyLossConfig {
            l1_weight: 0.0,
            warp_weight: 0.0,
            temporal_smooth_weight: 1.0,
            ..Default::default()
        };
        let loss = cfc_consistency_loss(&[fa, fb, fc], &cfg, &FlowConfig::default())?;
        assert!(
            loss.smooth_term > 0.0,
            "smooth_term should be nonzero: {}",
            loss.smooth_term
        );
        assert!((loss.total - loss.smooth_term).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_psnr_known_mse() -> Result<(), ConsistencyError> {
        // MSE = 0.01 → PSNR = 10*log10(1/0.01) = 10*2 = 20 dB
        // Build two frames where every channel differs by exactly sqrt(0.01)
        let diff = 0.1_f32; // diff^2 = 0.01
        let a = Frame::new(4, 4);
        let b = uniform_frame(4, 4, diff, diff, diff);
        let psnr = cfc_psnr(&a, &b)?;
        assert!((psnr - 20.0).abs() < 0.01, "expected 20 dB, got {psnr}");
        Ok(())
    }
}
