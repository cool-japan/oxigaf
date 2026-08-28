//! CPU-side edge detection on rendered RGBA images.
//!
//! Provides multiple edge detection algorithms for quality analysis,
//! silhouette enhancement, and diffusion model conditioning.
//!
//! Images are represented as flat `Vec<u8>` RGBA (width×height×4 bytes).
//!
//! # Algorithms
//!
//! - **Sobel**: gradient-based, 3×3 kernels
//! - **Prewitt**: alternative gradient operator
//! - **LoG (Laplacian of Gaussian)**: zero-crossing detection
//! - **Canny**: full pipeline (blur → Sobel → NMS → hysteresis)

use std::collections::VecDeque;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during edge detection operations.
#[derive(Debug, Error)]
pub enum EdgeDetectionError {
    /// Image contains zero pixels.
    #[error("Empty image (0 pixels)")]
    EmptyImage,

    /// Buffer length does not match `width * height * 4`.
    #[error("Image dimensions mismatch: expected {expected} bytes, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Gaussian sigma must be strictly positive.
    #[error("Invalid sigma {sigma}: must be > 0")]
    InvalidSigma { sigma: f32 },

    /// Hysteresis thresholds are inverted.
    #[error("Invalid threshold: low ({low}) must be <= high ({high})")]
    InvalidThreshold { low: f32, high: f32 },

    /// Image is smaller than 3×3, which several kernels require.
    #[error("Image too small: {width}×{height}, need at least 3×3")]
    TooSmall { width: u32, height: u32 },

    /// Kernel size must be a positive odd number.
    #[error("Invalid kernel size {size}: must be odd and >= 3")]
    InvalidKernelSize { size: u32 },
}

/// Largest kernel side accepted by any kernel-size-taking function in this
/// module (`build_edge_gaussian_kernel`, `edge_convolve`,
/// `edge_gaussian_blur`, `log_edges`, `CannyConfig::validate`).
///
/// Without a cap, a caller-supplied or `sigma`-derived kernel size can grow
/// unboundedly: a 1-D kernel of size `k` produces a `k × k` 2-D kernel
/// (`build_edge_gaussian_kernel`) and a per-pixel convolution cost of
/// `O(k²)` (`edge_convolve`) or `O(k)` per pass (`edge_gaussian_blur`), so
/// an unbounded `k` is effectively a hang or a multi-gigabyte allocation
/// from ordinary (non-malicious) input, e.g. a very large `sigma` passed to
/// `log_edges`. `129` is generous for any real blur radius while keeping
/// the worst case bounded.
const MAX_KERNEL_SIZE: u32 = 129;

// ─────────────────────────────────────────────────────────────────────────────
// EdgeMap
// ─────────────────────────────────────────────────────────────────────────────

/// A 2-D map of gradient magnitudes in `[0, 1]`.
#[derive(Debug, Clone)]
pub struct EdgeMap {
    /// Gradient magnitude values, row-major, one `f32` per pixel.
    pub data: Vec<f32>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl EdgeMap {
    /// Create an all-zero edge map of the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width as usize) * (height as usize);
        Self {
            data: vec![0.0_f32; n],
            width,
            height,
        }
    }

    /// Scale `[0, 1]` magnitudes to `[0, 255]` grayscale bytes.
    pub fn to_grayscale_u8(&self) -> Vec<u8> {
        self.data
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
    }

    /// Produce a flat RGBA buffer: edges appear as white (R=G=B=magnitude*255), A=255.
    pub fn to_rgba_u8(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data.len() * 4);
        for &v in &self.data {
            let byte = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            out.push(byte);
            out.push(byte);
            out.push(byte);
            out.push(255);
        }
        out
    }

    /// Maximum gradient magnitude across the map.
    pub fn max_magnitude(&self) -> f32 {
        self.data.iter().copied().fold(0.0_f32, f32::max)
    }

    /// Mean gradient magnitude across the map.
    pub fn mean_magnitude(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.data.iter().sum();
        sum / self.data.len() as f32
    }

    /// Binary edge mask: `true` where magnitude ≥ `t`.
    pub fn threshold(&self, t: f32) -> Vec<bool> {
        self.data.iter().map(|&v| v >= t).collect()
    }

    /// Clamped pixel read: coordinates outside the image return the border value.
    pub fn pixel(&self, x: u32, y: u32) -> f32 {
        let cx = x.min(self.width.saturating_sub(1));
        let cy = y.min(self.height.saturating_sub(1));
        let idx = cy as usize * self.width as usize + cx as usize;
        self.data.get(idx).copied().unwrap_or(0.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SobelResult
// ─────────────────────────────────────────────────────────────────────────────

/// Full Sobel decomposition: per-pixel gradients, magnitude, and direction.
#[derive(Debug, Clone)]
pub struct SobelResult {
    /// Horizontal gradient component (raw, not normalized).
    pub gx: Vec<f32>,
    /// Vertical gradient component (raw, not normalized).
    pub gy: Vec<f32>,
    /// Gradient magnitude normalized to `[0, 1]`.
    pub magnitude: EdgeMap,
    /// Gradient direction in radians `atan2(gy, gx)`.
    pub direction: Vec<f32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// CannyConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the full Canny edge detection pipeline.
#[derive(Debug, Clone)]
pub struct CannyConfig {
    /// Gaussian blur sigma (must be > 0).
    pub sigma: f32,
    /// Hysteresis low threshold (fraction of max, default 0.05).
    pub low_threshold: f32,
    /// Hysteresis high threshold (fraction of max, default 0.15).
    pub high_threshold: f32,
    /// Gaussian kernel size (must be odd, default 5).
    pub kernel_size: u32,
}

impl Default for CannyConfig {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            low_threshold: 0.05,
            high_threshold: 0.15,
            kernel_size: 5,
        }
    }
}

impl CannyConfig {
    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<(), EdgeDetectionError> {
        // An explicit `is_nan()` check (rather than plain `sigma <= 0.0`) is
        // required to reject NaN: every comparison with NaN is `false`, so
        // `NaN <= 0.0` is `false` and a NaN sigma would otherwise sail
        // through validation and propagate as an all-NaN kernel/image.
        if self.sigma.is_nan() || self.sigma <= 0.0 {
            return Err(EdgeDetectionError::InvalidSigma { sigma: self.sigma });
        }
        // Same NaN-rejection reasoning as above: `low_threshold >
        // high_threshold` alone would let a NaN threshold slip through.
        if self.low_threshold.is_nan()
            || self.high_threshold.is_nan()
            || self.low_threshold > self.high_threshold
        {
            return Err(EdgeDetectionError::InvalidThreshold {
                low: self.low_threshold,
                high: self.high_threshold,
            });
        }
        if self.kernel_size < 3
            || self.kernel_size.is_multiple_of(2)
            || self.kernel_size > MAX_KERNEL_SIZE
        {
            return Err(EdgeDetectionError::InvalidKernelSize {
                size: self.kernel_size,
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EdgeStats
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics for an `EdgeMap`.
#[derive(Debug, Clone)]
pub struct EdgeStats {
    /// Mean gradient magnitude.
    pub mean: f32,
    /// Maximum gradient magnitude.
    pub max: f32,
    /// Fraction of pixels with magnitude ≥ 0.1.
    pub edge_pixel_fraction: f32,
    /// Sum of squared magnitudes divided by pixel count.
    pub gradient_energy: f32,
}

/// Compute summary statistics for the given edge map.
pub fn compute_edge_stats(map: &EdgeMap) -> EdgeStats {
    let n = map.data.len();
    if n == 0 {
        return EdgeStats {
            mean: 0.0,
            max: 0.0,
            edge_pixel_fraction: 0.0,
            gradient_energy: 0.0,
        };
    }
    let mut sum = 0.0_f32;
    let mut max_val = 0.0_f32;
    let mut edge_count = 0usize;
    let mut sq_sum = 0.0_f32;
    for &v in &map.data {
        sum += v;
        if v > max_val {
            max_val = v;
        }
        if v >= 0.1 {
            edge_count += 1;
        }
        sq_sum += v * v;
    }
    EdgeStats {
        mean: sum / n as f32,
        max: max_val,
        edge_pixel_fraction: edge_count as f32 / n as f32,
        gradient_energy: sq_sum / n as f32,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation helpers
// ─────────────────────────────────────────────────────────────────────────────

fn validate_rgba(rgba: &[u8], width: u32, height: u32) -> Result<(), EdgeDetectionError> {
    let expected = (width as usize) * (height as usize) * 4;
    if expected == 0 {
        return Err(EdgeDetectionError::EmptyImage);
    }
    if rgba.len() != expected {
        return Err(EdgeDetectionError::DimensionMismatch {
            expected,
            actual: rgba.len(),
        });
    }
    Ok(())
}

fn validate_luminance(lum: &[f32], width: u32, height: u32) -> Result<(), EdgeDetectionError> {
    let expected = (width as usize) * (height as usize);
    if expected == 0 {
        return Err(EdgeDetectionError::EmptyImage);
    }
    if lum.len() != expected {
        return Err(EdgeDetectionError::DimensionMismatch {
            expected,
            actual: lum.len(),
        });
    }
    if width < 3 || height < 3 {
        return Err(EdgeDetectionError::TooSmall { width, height });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Luminance conversion
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a flat RGBA `u8` buffer to linear luminance values in `[0, 1]`.
///
/// Luminance formula: `L = 0.2126·R + 0.7152·G + 0.0722·B`.
pub fn rgba_to_luminance(
    rgba: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<f32>, EdgeDetectionError> {
    validate_rgba(rgba, width, height)?;
    let n = (width as usize) * (height as usize);
    let mut lum = Vec::with_capacity(n);
    for chunk in rgba.chunks_exact(4) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        lum.push(0.2126 * r + 0.7152 * g + 0.0722 * b);
    }
    Ok(lum)
}

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian kernel (2-D)
// ─────────────────────────────────────────────────────────────────────────────

/// Build a square 2-D Gaussian kernel of side `size` (must be odd) normalized
/// so all elements sum to 1.
pub fn build_edge_gaussian_kernel(size: u32, sigma: f32) -> Result<Vec<f32>, EdgeDetectionError> {
    // An explicit `is_nan()` check (a plain `sigma <= 0.0` does not reject
    // NaN, since every comparison with NaN is `false`) is required here:
    // otherwise a NaN sigma would produce an all-NaN kernel that then
    // silently passes the `sum > 0.0` normalisation guard below (NaN
    // comparisons are also always `false`).
    if sigma.is_nan() || sigma <= 0.0 {
        return Err(EdgeDetectionError::InvalidSigma { sigma });
    }
    if size < 1 || size.is_multiple_of(2) || size > MAX_KERNEL_SIZE {
        return Err(EdgeDetectionError::InvalidKernelSize { size });
    }
    let center = (size / 2) as f32;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut kernel = Vec::with_capacity((size * size) as usize);
    for ky in 0..size {
        for kx in 0..size {
            let dx = kx as f32 - center;
            let dy = ky as f32 - center;
            kernel.push((-(dx * dx + dy * dy) / two_sigma_sq).exp());
        }
    }
    let sum: f32 = kernel.iter().sum();
    if sum > 0.0 {
        for v in &mut kernel {
            *v /= sum;
        }
    }
    Ok(kernel)
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic 2-D convolution
// ─────────────────────────────────────────────────────────────────────────────

/// Convolve a single-channel f32 image with an arbitrary kernel.
///
/// - `kernel_size` must be odd.
/// - Edge handling: clamp-to-border (repeat nearest pixel).
pub fn edge_convolve(
    image: &[f32],
    width: u32,
    height: u32,
    kernel: &[f32],
    kernel_size: u32,
) -> Result<Vec<f32>, EdgeDetectionError> {
    let n = (width as usize) * (height as usize);
    if n == 0 {
        return Err(EdgeDetectionError::EmptyImage);
    }
    if image.len() != n {
        return Err(EdgeDetectionError::DimensionMismatch {
            expected: n,
            actual: image.len(),
        });
    }
    if kernel_size < 1 || kernel_size.is_multiple_of(2) || kernel_size > MAX_KERNEL_SIZE {
        return Err(EdgeDetectionError::InvalidKernelSize { size: kernel_size });
    }
    let expected_k = (kernel_size * kernel_size) as usize;
    if kernel.len() != expected_k {
        return Err(EdgeDetectionError::DimensionMismatch {
            expected: expected_k,
            actual: kernel.len(),
        });
    }

    let half = (kernel_size / 2) as i32;
    let w = width as i32;
    let h = height as i32;
    let mut out = vec![0.0_f32; n];

    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0_f32;
            let mut ki = 0usize;
            for ky in 0..kernel_size as i32 {
                for kx in 0..kernel_size as i32 {
                    let ix = (x + kx - half).clamp(0, w - 1) as usize;
                    let iy = (y + ky - half).clamp(0, h - 1) as usize;
                    acc += kernel[ki] * image[iy * width as usize + ix];
                    ki += 1;
                }
            }
            out[y as usize * width as usize + x as usize] = acc;
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Separable Gaussian blur
// ─────────────────────────────────────────────────────────────────────────────

/// Gaussian blur using separable 1-D horizontal + vertical passes.
///
/// Much faster than `edge_convolve` for large sigma values.
pub fn edge_gaussian_blur(
    image: &[f32],
    width: u32,
    height: u32,
    sigma: f32,
    kernel_size: u32,
) -> Result<Vec<f32>, EdgeDetectionError> {
    // `is_nan()` check also rejects NaN - see `build_edge_gaussian_kernel`.
    if sigma.is_nan() || sigma <= 0.0 {
        return Err(EdgeDetectionError::InvalidSigma { sigma });
    }
    if kernel_size < 1 || kernel_size.is_multiple_of(2) || kernel_size > MAX_KERNEL_SIZE {
        return Err(EdgeDetectionError::InvalidKernelSize { size: kernel_size });
    }
    let n = (width as usize) * (height as usize);
    if n == 0 {
        return Err(EdgeDetectionError::EmptyImage);
    }
    if image.len() != n {
        return Err(EdgeDetectionError::DimensionMismatch {
            expected: n,
            actual: image.len(),
        });
    }

    // Build 1-D kernel
    let half = (kernel_size / 2) as i32;
    let center = half as f32;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut k1d: Vec<f32> = (0..kernel_size)
        .map(|i| {
            let d = i as f32 - center;
            (-d * d / two_sigma_sq).exp()
        })
        .collect();
    let ksum: f32 = k1d.iter().sum();
    if ksum > 0.0 {
        for v in &mut k1d {
            *v /= ksum;
        }
    }

    let w = width as usize;
    let h = height as usize;
    let mut tmp = vec![0.0_f32; n];

    // Horizontal pass
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0_f32;
            for (ki, &kval) in k1d.iter().enumerate() {
                let sx = ((x as i32 + ki as i32 - half).clamp(0, w as i32 - 1)) as usize;
                acc += kval * image[y * w + sx];
            }
            tmp[y * w + x] = acc;
        }
    }

    // Vertical pass
    let mut out = vec![0.0_f32; n];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0_f32;
            for (ki, &kval) in k1d.iter().enumerate() {
                let sy = ((y as i32 + ki as i32 - half).clamp(0, h as i32 - 1)) as usize;
                acc += kval * tmp[sy * w + x];
            }
            out[y * w + x] = acc;
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Sobel edge detection
// ─────────────────────────────────────────────────────────────────────────────

/// Apply Sobel edge detection on a luminance image.
///
/// Kernels:
/// ```text
/// Kx = [[-1,0,1],[-2,0,2],[-1,0,1]]
/// Ky = [[-1,-2,-1],[0,0,0],[1,2,1]]
/// ```
///
/// The magnitude is normalized by its maximum to `[0, 1]`.
pub fn sobel_edges(
    luminance: &[f32],
    width: u32,
    height: u32,
) -> Result<SobelResult, EdgeDetectionError> {
    validate_luminance(luminance, width, height)?;
    let (gx, gy) = apply_3x3_gradients(luminance, width, height, &SOBEL_KX, &SOBEL_KY);

    let n = gx.len();
    let mut mag = vec![0.0_f32; n];
    for i in 0..n {
        mag[i] = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt();
    }

    let max_mag = mag.iter().copied().fold(0.0_f32, f32::max);
    // Use a noise floor: magnitudes below 1e-5 are treated as zero.
    // This prevents floating-point rounding in uniform images from
    // being normalised up to 1.0.
    let norm_factor = if max_mag > 1e-5 { 1.0 / max_mag } else { 0.0 };

    let mut direction = vec![0.0_f32; n];
    let mut norm_mag = vec![0.0_f32; n];
    for i in 0..n {
        norm_mag[i] = mag[i] * norm_factor;
        direction[i] = gy[i].atan2(gx[i]);
    }

    Ok(SobelResult {
        gx,
        gy,
        magnitude: EdgeMap {
            data: norm_mag,
            width,
            height,
        },
        direction,
    })
}

/// Sobel Kx coefficients (row-major, 3×3).
const SOBEL_KX: [f32; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
/// Sobel Ky coefficients (row-major, 3×3).
const SOBEL_KY: [f32; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

// ─────────────────────────────────────────────────────────────────────────────
// Prewitt edge detection
// ─────────────────────────────────────────────────────────────────────────────

/// Prewitt Kx coefficients (row-major, 3×3).
const PREWITT_KX: [f32; 9] = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0];
/// Prewitt Ky coefficients (row-major, 3×3).
const PREWITT_KY: [f32; 9] = [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

/// Apply Prewitt edge detection on a luminance image, returning a magnitude EdgeMap.
pub fn prewitt_edges(
    luminance: &[f32],
    width: u32,
    height: u32,
) -> Result<EdgeMap, EdgeDetectionError> {
    validate_luminance(luminance, width, height)?;
    let (gx, gy) = apply_3x3_gradients(luminance, width, height, &PREWITT_KX, &PREWITT_KY);

    let n = gx.len();
    let mut mag = vec![0.0_f32; n];
    for i in 0..n {
        mag[i] = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt();
    }
    let max_mag = mag.iter().copied().fold(0.0_f32, f32::max);
    // Apply the same noise-floor guard as sobel_edges to avoid normalising
    // f32 rounding residuals on uniform images up to 1.0.
    let norm_factor = if max_mag > 1e-5 { 1.0 / max_mag } else { 0.0 };
    for v in &mut mag {
        *v *= norm_factor;
    }
    Ok(EdgeMap {
        data: mag,
        width,
        height,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared 3×3 gradient helper
// ─────────────────────────────────────────────────────────────────────────────

fn apply_3x3_gradients(
    image: &[f32],
    width: u32,
    height: u32,
    kx: &[f32; 9],
    ky: &[f32; 9],
) -> (Vec<f32>, Vec<f32>) {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    let mut gx = vec![0.0_f32; n];
    let mut gy = vec![0.0_f32; n];

    let offsets: [(i32, i32); 9] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (0, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut ax = 0.0_f32;
            let mut ay = 0.0_f32;
            for (ki, &(dx, dy)) in offsets.iter().enumerate() {
                let sx = (x + dx).clamp(0, w as i32 - 1) as usize;
                let sy = (y + dy).clamp(0, h as i32 - 1) as usize;
                let v = image[sy * w + sx];
                ax += kx[ki] * v;
                ay += ky[ki] * v;
            }
            let idx = y as usize * w + x as usize;
            gx[idx] = ax;
            gy[idx] = ay;
        }
    }
    (gx, gy)
}

// ─────────────────────────────────────────────────────────────────────────────
// Laplacian of Gaussian (LoG) edge detection
// ─────────────────────────────────────────────────────────────────────────────

/// Laplacian kernel `[[0,-1,0],[-1,4,-1],[0,-1,0]]`.
const LAPLACIAN_K: [f32; 9] = [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0];

/// Apply LoG edge detection using zero-crossing detection after Gaussian blurring.
pub fn log_edges(
    luminance: &[f32],
    width: u32,
    height: u32,
    sigma: f32,
) -> Result<EdgeMap, EdgeDetectionError> {
    validate_luminance(luminance, width, height)?;
    // `is_nan()` check also rejects NaN - see `build_edge_gaussian_kernel`.
    if sigma.is_nan() || sigma <= 0.0 {
        return Err(EdgeDetectionError::InvalidSigma { sigma });
    }

    // Gaussian blur first. `kernel_size` grows unboundedly with `sigma` (a
    // 1-D kernel diameter of `6*sigma` is standard so the Gaussian's tails
    // are negligible), so it must be capped: an uncapped `sigma` of e.g.
    // 1e6 would demand a ~6-million-tap 1-D kernel (effectively a hang),
    // and a `sigma` large enough to overflow the `as u32` cast would
    // saturate to `u32::MAX` and attempt a multi-gigabyte allocation.
    let kernel_size = ((6.0 * sigma).ceil() as u32) | 1; // ensure odd
    let kernel_size = kernel_size.clamp(3, MAX_KERNEL_SIZE);
    let blurred = edge_gaussian_blur(luminance, width, height, sigma, kernel_size)?;

    // Apply Laplacian kernel
    let log_response = edge_convolve(&blurred, width, height, &LAPLACIAN_K, 3)?;

    // Zero-crossing detection with adaptive threshold
    let max_abs: f32 = log_response.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let threshold = 0.01 * max_abs;

    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    let mut edge_data = vec![0.0_f32; n];

    let neighbors_4 = [(0i32, -1i32), (0, 1), (-1, 0), (1, 0)];

    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let idx = y as usize * w + x as usize;
            let val = log_response[idx];
            if val.abs() <= threshold {
                continue;
            }
            // Check for sign change with any 4-neighbor
            let is_zero_crossing = neighbors_4.iter().any(|&(dx, dy)| {
                let nx = (x + dx).clamp(0, w as i32 - 1) as usize;
                let ny = (y + dy).clamp(0, h as i32 - 1) as usize;
                let nval = log_response[ny * w + nx];
                val * nval < 0.0
            });
            if is_zero_crossing {
                edge_data[idx] = val.abs() / max_abs.max(1e-10);
            }
        }
    }

    // Normalize LoG edge magnitudes to [0,1]
    let max_edge = edge_data.iter().copied().fold(0.0_f32, f32::max);
    if max_edge > 0.0 {
        for v in &mut edge_data {
            *v /= max_edge;
        }
    }

    Ok(EdgeMap {
        data: edge_data,
        width,
        height,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-maximum suppression
// ─────────────────────────────────────────────────────────────────────────────

/// Non-maximum suppression: thin edges to 1-pixel width along gradient direction.
///
/// Direction is quantized to one of four orientations (0°, 45°, 90°, 135°).
pub fn non_max_suppress(
    magnitude: &[f32],
    direction: &[f32],
    width: u32,
    height: u32,
) -> Result<Vec<f32>, EdgeDetectionError> {
    let n = (width as usize) * (height as usize);
    if n == 0 {
        return Err(EdgeDetectionError::EmptyImage);
    }
    if magnitude.len() != n || direction.len() != n {
        return Err(EdgeDetectionError::DimensionMismatch {
            expected: n,
            actual: magnitude.len().min(direction.len()),
        });
    }

    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0.0_f32; n];

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let mag = magnitude[idx];
            let dir = direction[idx];

            // Quantize direction to 4 orientations
            // Use absolute angle to determine axis
            let angle_deg = dir.to_degrees();
            // Map to [0, 180)
            let angle_norm = ((angle_deg % 180.0) + 180.0) % 180.0;

            // Determine neighbor offsets along gradient direction
            let (dx1, dy1, dx2, dy2) = if !(22.5..157.5).contains(&angle_norm) {
                // 0° (horizontal)
                (1i32, 0i32, -1i32, 0i32)
            } else if angle_norm < 67.5 {
                // 45° (diagonal)
                (1i32, -1i32, -1i32, 1i32)
            } else if angle_norm < 112.5 {
                // 90° (vertical)
                (0i32, -1i32, 0i32, 1i32)
            } else {
                // 135° (anti-diagonal)
                (1i32, 1i32, -1i32, -1i32)
            };

            let n1 = sample_clamped(magnitude, w, h, x as i32 + dx1, y as i32 + dy1);
            let n2 = sample_clamped(magnitude, w, h, x as i32 + dx2, y as i32 + dy2);

            if mag >= n1 && mag >= n2 {
                out[idx] = mag;
            }
            // else suppress (stays 0)
        }
    }
    Ok(out)
}

#[inline]
fn sample_clamped(data: &[f32], w: usize, h: usize, x: i32, y: i32) -> f32 {
    let cx = x.clamp(0, w as i32 - 1) as usize;
    let cy = y.clamp(0, h as i32 - 1) as usize;
    data[cy * w + cx]
}

// ─────────────────────────────────────────────────────────────────────────────
// Hysteresis thresholding
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum EdgeState {
    None,
    Weak,
    Strong,
    Edge,
}

/// Double threshold + hysteresis connectivity for Canny edge detection.
///
/// - Pixels above `high` → strong edges (always kept).
/// - Pixels in `[low, high]` → weak edges (kept if connected to strong).
/// - Pixels below `low` → suppressed.
pub fn hysteresis_threshold(
    suppressed: &[f32],
    width: u32,
    height: u32,
    low: f32,
    high: f32,
) -> Result<Vec<bool>, EdgeDetectionError> {
    let n = (width as usize) * (height as usize);
    if n == 0 {
        return Err(EdgeDetectionError::EmptyImage);
    }
    if suppressed.len() != n {
        return Err(EdgeDetectionError::DimensionMismatch {
            expected: n,
            actual: suppressed.len(),
        });
    }
    // `is_nan()` check also rejects NaN - see `CannyConfig::validate`.
    if low.is_nan() || high.is_nan() || low > high {
        return Err(EdgeDetectionError::InvalidThreshold { low, high });
    }

    let w = width as usize;
    let h = height as usize;

    let mut states: Vec<EdgeState> = suppressed
        .iter()
        .map(|&v| {
            if v >= high {
                EdgeState::Strong
            } else if v >= low {
                EdgeState::Weak
            } else {
                EdgeState::None
            }
        })
        .collect();

    // BFS from all strong edges
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (i, &s) in states.iter().enumerate() {
        if s == EdgeState::Strong {
            queue.push_back(i);
        }
    }

    let neighbor_offsets: [(i32, i32); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    while let Some(idx) = queue.pop_front() {
        if states[idx] == EdgeState::Edge {
            continue;
        }
        states[idx] = EdgeState::Edge;
        let y = (idx / w) as i32;
        let x = (idx % w) as i32;
        for &(dx, dy) in &neighbor_offsets {
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let nidx = ny as usize * w + nx as usize;
            if states[nidx] == EdgeState::Weak {
                states[nidx] = EdgeState::Strong; // promote to propagate
                queue.push_back(nidx);
            }
        }
    }

    Ok(states.iter().map(|&s| s == EdgeState::Edge).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Full Canny pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Full Canny edge detection pipeline on an RGBA image.
///
/// Steps: RGBA→luminance → Gaussian blur → Sobel → NMS → hysteresis → EdgeMap.
pub fn canny_edges(
    rgba: &[u8],
    width: u32,
    height: u32,
    config: &CannyConfig,
) -> Result<EdgeMap, EdgeDetectionError> {
    config.validate()?;

    let lum = rgba_to_luminance(rgba, width, height)?;
    let blurred = edge_gaussian_blur(&lum, width, height, config.sigma, config.kernel_size)?;
    let sobel = sobel_edges(&blurred, width, height)?;
    let suppressed = non_max_suppress(&sobel.magnitude.data, &sobel.direction, width, height)?;
    let edge_mask = hysteresis_threshold(
        &suppressed,
        width,
        height,
        config.low_threshold,
        config.high_threshold,
    )?;

    // Convert bool mask to f32 EdgeMap
    let data: Vec<f32> = edge_mask
        .iter()
        .map(|&e| if e { 1.0 } else { 0.0 })
        .collect();
    Ok(EdgeMap {
        data,
        width,
        height,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// Sobel edge detection on an RGBA image (convenience wrapper).
pub fn detect_edges_sobel(
    rgba: &[u8],
    width: u32,
    height: u32,
) -> Result<EdgeMap, EdgeDetectionError> {
    let lum = rgba_to_luminance(rgba, width, height)?;
    let result = sobel_edges(&lum, width, height)?;
    Ok(result.magnitude)
}

/// LoG edge detection on an RGBA image (convenience wrapper).
pub fn detect_edges_log(
    rgba: &[u8],
    width: u32,
    height: u32,
    sigma: f32,
) -> Result<EdgeMap, EdgeDetectionError> {
    let lum = rgba_to_luminance(rgba, width, height)?;
    log_edges(&lum, width, height, sigma)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Create a uniform RGBA image of given color.
    fn make_rgba(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let n = (width * height) as usize;
        let mut buf = Vec::with_capacity(n * 4);
        for _ in 0..n {
            buf.push(r);
            buf.push(g);
            buf.push(b);
            buf.push(a);
        }
        buf
    }

    /// Create a left-half white, right-half black RGBA image (vertical edge).
    fn make_vertical_edge_rgba(width: u32, height: u32) -> Vec<u8> {
        let n = (width * height) as usize;
        let half = width / 2;
        let mut buf = Vec::with_capacity(n * 4);
        for _y in 0..height {
            for x in 0..width {
                let v = if x < half { 255u8 } else { 0u8 };
                buf.push(v);
                buf.push(v);
                buf.push(v);
                buf.push(255);
            }
        }
        buf
    }

    /// Create a uniform f32 luminance image.
    fn make_lum(width: u32, height: u32, val: f32) -> Vec<f32> {
        vec![val; (width * height) as usize]
    }

    /// Create a vertical-edge luminance image (left half = high, right = low).
    fn make_vertical_edge_lum(width: u32, height: u32) -> Vec<f32> {
        let half = (width / 2) as usize;
        let w = width as usize;
        let h = height as usize;
        let mut buf = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 0..half {
                buf[y * w + x] = 1.0;
            }
        }
        buf
    }

    // ── rgba_to_luminance ────────────────────────────────────────────────────

    #[test]
    fn test_luminance_correct_weights() {
        // Pure red → lum ≈ 0.2126
        let rgba = vec![255u8, 0, 0, 255];
        let lum = rgba_to_luminance(&rgba, 1, 1).expect("luminance failed");
        let expected = 0.2126_f32;
        assert!((lum[0] - expected).abs() < 1e-4, "lum={}", lum[0]);
    }

    #[test]
    fn test_luminance_correct_size() {
        let rgba = make_rgba(4, 5, 128, 64, 32, 255);
        let lum = rgba_to_luminance(&rgba, 4, 5).expect("luminance failed");
        assert_eq!(lum.len(), 20);
    }

    #[test]
    fn test_luminance_all_white() {
        let rgba = make_rgba(3, 3, 255, 255, 255, 255);
        let lum = rgba_to_luminance(&rgba, 3, 3).expect("luminance failed");
        for v in lum {
            assert!((v - 1.0).abs() < 1e-4, "white lum={}", v);
        }
    }

    #[test]
    fn test_luminance_all_black() {
        let rgba = make_rgba(3, 3, 0, 0, 0, 255);
        let lum = rgba_to_luminance(&rgba, 3, 3).expect("luminance failed");
        for v in lum {
            assert!(v < 1e-6, "black lum={}", v);
        }
    }

    #[test]
    fn test_luminance_dimension_mismatch() {
        let rgba = vec![0u8; 10]; // wrong size for 3×3×4=36
        let result = rgba_to_luminance(&rgba, 3, 3);
        assert!(matches!(
            result,
            Err(EdgeDetectionError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_luminance_empty() {
        let result = rgba_to_luminance(&[], 0, 0);
        assert!(matches!(result, Err(EdgeDetectionError::EmptyImage)));
    }

    // ── build_edge_gaussian_kernel ───────────────────────────────────────────

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        let k = build_edge_gaussian_kernel(5, 1.0).expect("kernel failed");
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={}", sum);
    }

    #[test]
    fn test_gaussian_kernel_correct_size() {
        let k = build_edge_gaussian_kernel(7, 2.0).expect("kernel failed");
        assert_eq!(k.len(), 49);
    }

    #[test]
    fn test_gaussian_kernel_even_size_rejected() {
        let result = build_edge_gaussian_kernel(4, 1.0);
        assert!(matches!(
            result,
            Err(EdgeDetectionError::InvalidKernelSize { .. })
        ));
    }

    #[test]
    fn test_gaussian_kernel_zero_sigma_rejected() {
        let result = build_edge_gaussian_kernel(5, 0.0);
        assert!(matches!(
            result,
            Err(EdgeDetectionError::InvalidSigma { .. })
        ));
    }

    #[test]
    fn test_gaussian_kernel_center_is_max() {
        let k = build_edge_gaussian_kernel(5, 1.0).expect("kernel failed");
        let center = k[12]; // index 2*5+2 = 12
        for (i, &v) in k.iter().enumerate() {
            if i != 12 {
                assert!(v <= center + 1e-7, "corner ({}) > center ({})", v, center);
            }
        }
    }

    // ── edge_gaussian_blur ───────────────────────────────────────────────────

    #[test]
    fn test_blur_uniform_image_stays_uniform() {
        let lum = make_lum(8, 8, 0.5);
        let blurred = edge_gaussian_blur(&lum, 8, 8, 1.5, 5).expect("blur failed");
        for v in blurred {
            assert!((v - 0.5).abs() < 1e-4, "val={}", v);
        }
    }

    #[test]
    fn test_blur_reduces_variance() {
        // High-frequency noise: checkerboard
        let w = 8u32;
        let h = 8u32;
        let n = (w * h) as usize;
        let mut noisy: Vec<f32> = Vec::with_capacity(n);
        for i in 0..n {
            noisy.push(if i % 2 == 0 { 1.0 } else { 0.0 });
        }
        let blurred = edge_gaussian_blur(&noisy, w, h, 1.5, 5).expect("blur failed");
        let orig_var: f32 = noisy.iter().map(|&v| (v - 0.5).powi(2)).sum::<f32>() / n as f32;
        let blur_var: f32 = blurred.iter().map(|&v| (v - 0.5).powi(2)).sum::<f32>() / n as f32;
        assert!(
            blur_var < orig_var,
            "blur increased variance: {} >= {}",
            blur_var,
            orig_var
        );
    }

    #[test]
    fn test_blur_invalid_sigma() {
        let lum = make_lum(5, 5, 0.5);
        let result = edge_gaussian_blur(&lum, 5, 5, -1.0, 3);
        assert!(matches!(
            result,
            Err(EdgeDetectionError::InvalidSigma { .. })
        ));
    }

    // ── sobel_edges ──────────────────────────────────────────────────────────

    #[test]
    fn test_sobel_uniform_zero_gradient() {
        let lum = make_lum(5, 5, 0.7);
        let res = sobel_edges(&lum, 5, 5).expect("sobel failed");
        for v in &res.magnitude.data {
            assert!(
                *v < 1e-5,
                "uniform Sobel magnitude should be near-zero, got {}",
                v
            );
        }
    }

    #[test]
    fn test_sobel_vertical_edge_detected() {
        let lum = make_vertical_edge_lum(10, 10);
        let res = sobel_edges(&lum, 10, 10).expect("sobel failed");
        let max_mag = res.magnitude.max_magnitude();
        assert!(
            max_mag > 0.5,
            "expected large magnitude at edge, got {}",
            max_mag
        );
    }

    #[test]
    fn test_sobel_magnitude_normalized() {
        let lum = make_vertical_edge_lum(8, 8);
        let res = sobel_edges(&lum, 8, 8).expect("sobel failed");
        let max_v = res.magnitude.max_magnitude();
        assert!(max_v <= 1.0 + 1e-5, "magnitude not normalized: {}", max_v);
        assert!(max_v > 0.0, "magnitude should be > 0 for edge image");
    }

    #[test]
    fn test_sobel_direction_range() {
        let lum = make_vertical_edge_lum(6, 6);
        let res = sobel_edges(&lum, 6, 6).expect("sobel failed");
        for &d in &res.direction {
            assert!(
                (-std::f32::consts::PI - 1e-4..=std::f32::consts::PI + 1e-4).contains(&d),
                "direction out of range: {}",
                d
            );
        }
    }

    #[test]
    fn test_sobel_result_sizes() {
        let w = 7u32;
        let h = 6u32;
        let lum = make_lum(w, h, 0.3);
        let res = sobel_edges(&lum, w, h).expect("sobel failed");
        assert_eq!(res.gx.len(), (w * h) as usize);
        assert_eq!(res.gy.len(), (w * h) as usize);
        assert_eq!(res.magnitude.data.len(), (w * h) as usize);
        assert_eq!(res.direction.len(), (w * h) as usize);
    }

    // ── prewitt_edges ────────────────────────────────────────────────────────

    #[test]
    fn test_prewitt_uniform_zero_gradient() {
        let lum = make_lum(5, 5, 0.5);
        let map = prewitt_edges(&lum, 5, 5).expect("prewitt failed");
        for v in &map.data {
            assert!(*v < 1e-5, "uniform Prewitt magnitude non-zero: {}", v);
        }
    }

    #[test]
    fn test_prewitt_edge_detected() {
        let lum = make_vertical_edge_lum(10, 10);
        let map = prewitt_edges(&lum, 10, 10).expect("prewitt failed");
        assert!(
            map.max_magnitude() > 0.5,
            "Prewitt should detect vertical edge"
        );
    }

    #[test]
    fn test_prewitt_normalized() {
        let lum = make_vertical_edge_lum(8, 8);
        let map = prewitt_edges(&lum, 8, 8).expect("prewitt failed");
        assert!(map.max_magnitude() <= 1.0 + 1e-5);
    }

    // ── log_edges ────────────────────────────────────────────────────────────

    #[test]
    fn test_log_uniform_zero() {
        let lum = make_lum(7, 7, 0.5);
        let map = log_edges(&lum, 7, 7, 1.0).expect("log failed");
        for v in &map.data {
            assert!(*v < 1e-5, "uniform LoG should be zero, got {}", v);
        }
    }

    #[test]
    fn test_log_detects_edge() {
        let lum = make_vertical_edge_lum(10, 10);
        let map = log_edges(&lum, 10, 10, 1.0).expect("log failed");
        assert!(map.max_magnitude() > 0.0, "LoG should detect edge");
    }

    #[test]
    fn test_log_invalid_sigma() {
        let lum = make_lum(5, 5, 0.5);
        let result = log_edges(&lum, 5, 5, 0.0);
        assert!(matches!(
            result,
            Err(EdgeDetectionError::InvalidSigma { .. })
        ));
    }

    // ── non_max_suppress ─────────────────────────────────────────────────────

    #[test]
    fn test_nms_suppresses_weak_neighbors() {
        // A single strong pixel surrounded by weaker ones
        let w = 5u32;
        let h = 5u32;
        let n = (w * h) as usize;
        let mut mag = vec![0.5_f32; n];
        mag[12] = 1.0; // center
        let dir = vec![0.0_f32; n]; // horizontal gradient
        let out = non_max_suppress(&mag, &dir, w, h).expect("nms failed");
        // Center should survive, horizontal neighbors should be suppressed
        assert_eq!(out[12], 1.0, "center should survive NMS");
        assert_eq!(out[11], 0.0, "left of center should be suppressed");
        assert_eq!(out[13], 0.0, "right of center should be suppressed");
    }

    #[test]
    fn test_nms_uniform_stays() {
        // Uniform magnitude → every pixel is local max → all survive
        let w = 5u32;
        let h = 5u32;
        let mag = vec![0.8_f32; (w * h) as usize];
        let dir = vec![0.0_f32; (w * h) as usize];
        let out = non_max_suppress(&mag, &dir, w, h).expect("nms failed");
        for v in out {
            assert!((v - 0.8).abs() < 1e-5);
        }
    }

    #[test]
    fn test_nms_output_size() {
        let w = 6u32;
        let h = 4u32;
        let n = (w * h) as usize;
        let mag = vec![0.5_f32; n];
        let dir = vec![0.0_f32; n];
        let out = non_max_suppress(&mag, &dir, w, h).expect("nms failed");
        assert_eq!(out.len(), n);
    }

    #[test]
    fn test_nms_vertical_direction() {
        let w = 3u32;
        let h = 5u32;
        let n = (w * h) as usize;
        // Vertical strip with strongest in middle row
        let mut mag = vec![0.3_f32; n];
        // Make row y=2 (middle) strongest
        for x in 0..w {
            mag[2 * 3 + x as usize] = 1.0;
        }
        let dir = vec![std::f32::consts::PI / 2.0; n]; // 90° = vertical
        let out = non_max_suppress(&mag, &dir, w, h).expect("nms failed");
        // middle row pixels should survive
        assert!(out[7] > 0.0, "middle row should survive");
    }

    // ── hysteresis_threshold ─────────────────────────────────────────────────

    #[test]
    fn test_hysteresis_strong_always_edge() {
        let w = 5u32;
        let h = 5u32;
        let n = (w * h) as usize;
        let mut sup = vec![0.0_f32; n];
        sup[12] = 0.9; // strong
        let mask = hysteresis_threshold(&sup, w, h, 0.05, 0.5).expect("hysteresis failed");
        assert!(mask[12], "strong pixel should be edge");
    }

    #[test]
    fn test_hysteresis_isolated_weak_not_edge() {
        let w = 7u32;
        let h = 7u32;
        let n = (w * h) as usize;
        let mut sup = vec![0.0_f32; n];
        sup[24] = 0.1; // weak, center of 7×7
        let mask = hysteresis_threshold(&sup, w, h, 0.05, 0.5).expect("hysteresis failed");
        assert!(!mask[24], "isolated weak should NOT be edge");
    }

    #[test]
    fn test_hysteresis_weak_connected_to_strong_is_edge() {
        let w = 5u32;
        let h = 5u32;
        let n = (w * h) as usize;
        let mut sup = vec![0.0_f32; n];
        sup[12] = 0.9; // strong
        sup[13] = 0.1; // weak, adjacent to strong
        let mask = hysteresis_threshold(&sup, w, h, 0.05, 0.5).expect("hysteresis failed");
        assert!(mask[12], "strong should be edge");
        assert!(mask[13], "weak adjacent to strong should be edge");
    }

    #[test]
    fn test_hysteresis_invalid_threshold() {
        let sup = vec![0.5_f32; 25];
        let result = hysteresis_threshold(&sup, 5, 5, 0.5, 0.1);
        assert!(matches!(
            result,
            Err(EdgeDetectionError::InvalidThreshold { .. })
        ));
    }

    // ── canny_edges ──────────────────────────────────────────────────────────

    #[test]
    fn test_canny_end_to_end_uniform() {
        let rgba = make_rgba(8, 8, 128, 128, 128, 255);
        let config = CannyConfig::default();
        let map = canny_edges(&rgba, 8, 8, &config).expect("canny failed");
        let n_edges = map.data.iter().filter(|&&v| v > 0.0).count();
        assert_eq!(n_edges, 0, "uniform image should have no Canny edges");
    }

    #[test]
    fn test_canny_end_to_end_edge_image() {
        let rgba = make_vertical_edge_rgba(20, 20);
        let config = CannyConfig {
            sigma: 1.0,
            low_threshold: 0.02,
            high_threshold: 0.1,
            kernel_size: 5,
        };
        let map = canny_edges(&rgba, 20, 20, &config).expect("canny failed");
        assert!(
            map.max_magnitude() > 0.0,
            "Canny should detect the vertical edge"
        );
    }

    #[test]
    fn test_canny_invalid_config_rejected() {
        let rgba = make_rgba(5, 5, 0, 0, 0, 255);
        let config = CannyConfig {
            sigma: -1.0,
            ..Default::default()
        };
        let result = canny_edges(&rgba, 5, 5, &config);
        assert!(matches!(
            result,
            Err(EdgeDetectionError::InvalidSigma { .. })
        ));
    }

    // ── detect_edges_sobel / detect_edges_log ────────────────────────────────

    #[test]
    fn test_detect_edges_sobel_wrapper() {
        let rgba = make_vertical_edge_rgba(10, 10);
        let map = detect_edges_sobel(&rgba, 10, 10).expect("sobel wrapper failed");
        assert!(map.max_magnitude() > 0.0);
    }

    #[test]
    fn test_detect_edges_log_wrapper() {
        let rgba = make_vertical_edge_rgba(10, 10);
        let map = detect_edges_log(&rgba, 10, 10, 1.0).expect("log wrapper failed");
        assert_eq!(map.width, 10);
        assert_eq!(map.height, 10);
    }

    #[test]
    fn test_detect_edges_sobel_uniform() {
        let rgba = make_rgba(5, 5, 200, 200, 200, 255);
        let map = detect_edges_sobel(&rgba, 5, 5).expect("sobel wrapper failed");
        assert_eq!(map.data.iter().filter(|&&v| v > 1e-5).count(), 0);
    }

    // ── EdgeMap methods ──────────────────────────────────────────────────────

    #[test]
    fn test_edge_map_to_grayscale_u8() {
        let mut map = EdgeMap::new(3, 1);
        map.data[0] = 0.0;
        map.data[1] = 0.5;
        map.data[2] = 1.0;
        let bytes = map.to_grayscale_u8();
        assert_eq!(bytes[0], 0);
        assert_eq!(bytes[2], 255);
        assert!(
            (bytes[1] as f32 - 127.5).abs() < 2.0,
            "mid value: {}",
            bytes[1]
        );
    }

    #[test]
    fn test_edge_map_to_rgba_u8_alpha() {
        let mut map = EdgeMap::new(2, 1);
        map.data[0] = 0.0;
        map.data[1] = 1.0;
        let bytes = map.to_rgba_u8();
        assert_eq!(bytes.len(), 8);
        assert_eq!(bytes[3], 255); // alpha of first pixel
        assert_eq!(bytes[7], 255); // alpha of second pixel
        assert_eq!(bytes[4], 255); // R of white pixel
    }

    #[test]
    fn test_edge_map_threshold() {
        let mut map = EdgeMap::new(4, 1);
        map.data[0] = 0.0;
        map.data[1] = 0.1;
        map.data[2] = 0.5;
        map.data[3] = 1.0;
        let mask = map.threshold(0.3);
        assert!(!mask[0]);
        assert!(!mask[1]);
        assert!(mask[2]);
        assert!(mask[3]);
    }

    #[test]
    fn test_edge_map_pixel_clamped() {
        let mut map = EdgeMap::new(3, 3);
        map.data[0] = 0.42;
        let v = map.pixel(0, 0);
        assert!((v - 0.42).abs() < 1e-6);
        // Out-of-bounds access should clamp, not panic
        let v2 = map.pixel(100, 100);
        assert!((0.0..=1.0).contains(&v2));
    }

    #[test]
    fn test_edge_map_mean_max() {
        let mut map = EdgeMap::new(4, 1);
        map.data = vec![0.0, 0.25, 0.5, 1.0];
        assert!((map.max_magnitude() - 1.0).abs() < 1e-6);
        assert!(
            (map.mean_magnitude() - 0.4375).abs() < 1e-4,
            "mean={}",
            map.mean_magnitude()
        );
    }

    // ── EdgeStats / compute_edge_stats ───────────────────────────────────────

    #[test]
    fn test_compute_edge_stats_uniform_zero() {
        let map = EdgeMap::new(5, 5);
        let stats = compute_edge_stats(&map);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.edge_pixel_fraction, 0.0);
        assert_eq!(stats.gradient_energy, 0.0);
    }

    #[test]
    fn test_compute_edge_stats_all_ones() {
        let mut map = EdgeMap::new(4, 4);
        map.data = vec![1.0; 16];
        let stats = compute_edge_stats(&map);
        assert!((stats.mean - 1.0).abs() < 1e-6);
        assert!((stats.max - 1.0).abs() < 1e-6);
        assert!((stats.edge_pixel_fraction - 1.0).abs() < 1e-6);
        assert!((stats.gradient_energy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_edge_stats_partial() {
        let lum = make_vertical_edge_lum(10, 10);
        let res = sobel_edges(&lum, 10, 10).expect("sobel failed");
        let stats = compute_edge_stats(&res.magnitude);
        assert!(stats.max > 0.0, "vertical edge should have nonzero max");
        assert!(stats.mean > 0.0, "mean should be nonzero");
        assert!(stats.gradient_energy > 0.0);
        assert!(stats.edge_pixel_fraction >= 0.0 && stats.edge_pixel_fraction <= 1.0);
    }

    // ── CannyConfig validation ───────────────────────────────────────────────

    #[test]
    fn test_canny_config_default_valid() {
        let config = CannyConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_canny_config_inverted_thresholds() {
        let config = CannyConfig {
            low_threshold: 0.5,
            high_threshold: 0.1,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(EdgeDetectionError::InvalidThreshold { .. })
        ));
    }

    #[test]
    fn test_canny_config_even_kernel_rejected() {
        let config = CannyConfig {
            kernel_size: 4,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(EdgeDetectionError::InvalidKernelSize { .. })
        ));
    }

    // ── edge_convolve ────────────────────────────────────────────────────────

    #[test]
    fn test_edge_convolve_identity_kernel() {
        // Identity kernel: [[0,0,0],[0,1,0],[0,0,0]]
        let kernel = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let image: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let out = edge_convolve(&image, 4, 4, &kernel, 3).expect("convolve failed");
        for (a, b) in image.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-5, "identity: {} != {}", a, b);
        }
    }

    #[test]
    fn test_edge_convolve_output_size() {
        let image = vec![0.5_f32; 20];
        let kernel = vec![1.0 / 9.0; 9];
        let out = edge_convolve(&image, 4, 5, &kernel, 3).expect("convolve failed");
        assert_eq!(out.len(), 20);
    }

    // ── Regression: NaN must not bypass sigma/threshold validation ───────────
    //
    // `x <= 0.0` (and `low > high`) are `false` for NaN, since every
    // comparison involving NaN is `false` - so these guards must pair that
    // check with an explicit `x.is_nan()` (rather than relying solely on
    // `!(x > 0.0)` / `!(low <= high)`, which clippy's `neg_cmp_op_on_partial_ord`
    // rejects) to actually reject NaN.

    #[test]
    fn test_canny_config_validate_rejects_nan_sigma() {
        let cfg = CannyConfig {
            sigma: f32::NAN,
            ..CannyConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(EdgeDetectionError::InvalidSigma { .. })
        ));
    }

    #[test]
    fn test_canny_config_validate_rejects_nan_threshold() {
        let cfg = CannyConfig {
            low_threshold: f32::NAN,
            ..CannyConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(EdgeDetectionError::InvalidThreshold { .. })
        ));
    }

    #[test]
    fn test_build_edge_gaussian_kernel_rejects_nan_sigma() {
        let result = build_edge_gaussian_kernel(5, f32::NAN);
        assert!(matches!(
            result,
            Err(EdgeDetectionError::InvalidSigma { .. })
        ));
    }

    #[test]
    fn test_edge_gaussian_blur_rejects_nan_sigma() {
        let lum = vec![0.5_f32; 25];
        let result = edge_gaussian_blur(&lum, 5, 5, f32::NAN, 3);
        assert!(matches!(
            result,
            Err(EdgeDetectionError::InvalidSigma { .. })
        ));
    }

    #[test]
    fn test_log_edges_rejects_nan_sigma() {
        let lum = vec![0.5_f32; 49];
        let result = log_edges(&lum, 7, 7, f32::NAN);
        assert!(matches!(
            result,
            Err(EdgeDetectionError::InvalidSigma { .. })
        ));
    }

    #[test]
    fn test_hysteresis_threshold_rejects_nan_low() {
        let suppressed = vec![0.5_f32; 25];
        let result = hysteresis_threshold(&suppressed, 5, 5, f32::NAN, 0.15);
        assert!(matches!(
            result,
            Err(EdgeDetectionError::InvalidThreshold { .. })
        ));
    }

    // ── Regression: kernel size must be bounded ───────────────────────────────

    #[test]
    fn test_log_edges_clamps_huge_sigma_kernel_size() {
        // A pathologically large sigma (whose naive `6*sigma` kernel diameter
        // would be millions of taps) must not hang or attempt a huge
        // allocation - `log_edges` clamps the derived kernel size to
        // `MAX_KERNEL_SIZE` instead of using it unbounded.
        let lum = vec![0.5_f32; 25];
        let result = log_edges(&lum, 5, 5, 1.0e6);
        assert!(result.is_ok(), "expected Ok, got {result:?}");
    }

    #[test]
    fn test_edge_gaussian_blur_rejects_oversized_kernel() {
        let lum = vec![0.5_f32; 25];
        let result = edge_gaussian_blur(&lum, 5, 5, 1.0, MAX_KERNEL_SIZE + 2);
        assert!(matches!(
            result,
            Err(EdgeDetectionError::InvalidKernelSize { .. })
        ));
    }

    #[test]
    fn test_edge_convolve_rejects_oversized_kernel() {
        let image = vec![0.5_f32; 25];
        let size = MAX_KERNEL_SIZE + 2;
        let kernel = vec![0.0_f32; (size * size) as usize];
        let result = edge_convolve(&image, 5, 5, &kernel, size);
        assert!(matches!(
            result,
            Err(EdgeDetectionError::InvalidKernelSize { .. })
        ));
    }

    #[test]
    fn test_build_edge_gaussian_kernel_rejects_oversized_kernel() {
        let result = build_edge_gaussian_kernel(MAX_KERNEL_SIZE + 2, 1.0);
        assert!(matches!(
            result,
            Err(EdgeDetectionError::InvalidKernelSize { .. })
        ));
    }

    #[test]
    fn test_canny_config_validate_rejects_oversized_kernel() {
        let cfg = CannyConfig {
            kernel_size: MAX_KERNEL_SIZE + 2,
            ..CannyConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(EdgeDetectionError::InvalidKernelSize { .. })
        ));
    }
}
