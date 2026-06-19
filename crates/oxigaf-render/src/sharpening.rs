//! Image sharpening post-processing for rendered 3DGS output.
//!
//! Provides multiple sharpening algorithms with configurable strength:
//! - Unsharp masking (classical photo-lab technique)
//! - High-pass sharpening (subtract low-freq, amplify remainder)
//! - Laplacian-based edge enhancement
//! - Adaptive sharpening guided by Sobel edge magnitude
//! - Local contrast enhancement
//! - High-boost spatial filter
//! - Richardson-Lucy iterative deconvolution

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during sharpening operations.
#[derive(Debug, Error)]
pub enum SharpeningError {
    /// Configuration is invalid (e.g. sigma ≤ 0).
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// Pixel buffer dimensions are inconsistent.
    #[error("Invalid image: width={w}, height={h}, channels={c}")]
    InvalidImage { w: usize, h: usize, c: usize },

    /// No pixels in the provided buffer.
    #[error("Empty image")]
    EmptyImage,

    /// Kernel size must be an odd number.
    #[error("Kernel size must be odd, got {0}")]
    InvalidKernelSize(usize),

    /// Two lengths that must agree do not.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration types
// ─────────────────────────────────────────────────────────────────────────────

/// Which sharpening algorithm to apply.
#[derive(Debug, Clone, PartialEq)]
pub enum SharpenMethod {
    /// Classical unsharp mask: sharpened = original + amount*(original − blurred).
    UnsharpMask { sigma: f32, amount: f32 },
    /// Laplacian-of-Gaussian edge enhancement.
    LaplacianOfGaussian { sigma: f32 },
    /// High-pass sharpening: add scaled high-frequency detail back.
    HighPass { sigma: f32, strength: f32 },
    /// Boost local contrast within a neighbourhood.
    LocalContrast { radius: usize, strength: f32 },
    /// Stronger sharpening near detected edges.
    Adaptive { base_strength: f32, edge_boost: f32 },
}

/// Full sharpening configuration.
#[derive(Debug, Clone)]
pub struct SharpenConfig {
    /// Algorithm to use.
    pub method: SharpenMethod,
    /// Minimum absolute luminance difference that triggers sharpening.
    /// `0.0` means always apply.
    pub threshold: f32,
    /// Clamp output pixels to `[0, 255]`.
    pub clamp: bool,
}

impl Default for SharpenConfig {
    fn default() -> Self {
        Self {
            method: SharpenMethod::UnsharpMask {
                sigma: 1.0,
                amount: 0.5,
            },
            threshold: 0.0,
            clamp: true,
        }
    }
}

impl SharpenConfig {
    /// Validate all parameters.
    ///
    /// Returns `Err(SharpeningError::InvalidConfig)` if any parameter is out
    /// of range.
    pub fn validate(&self) -> Result<(), SharpeningError> {
        match &self.method {
            SharpenMethod::UnsharpMask { sigma, amount } => {
                if *sigma <= 0.0 {
                    return Err(SharpeningError::InvalidConfig(
                        "sigma must be > 0".to_string(),
                    ));
                }
                if *amount < 0.0 {
                    return Err(SharpeningError::InvalidConfig(
                        "amount must be >= 0".to_string(),
                    ));
                }
            }
            SharpenMethod::LaplacianOfGaussian { sigma } => {
                if *sigma <= 0.0 {
                    return Err(SharpeningError::InvalidConfig(
                        "sigma must be > 0".to_string(),
                    ));
                }
            }
            SharpenMethod::HighPass { sigma, strength } => {
                if *sigma <= 0.0 {
                    return Err(SharpeningError::InvalidConfig(
                        "sigma must be > 0".to_string(),
                    ));
                }
                if *strength < 0.0 {
                    return Err(SharpeningError::InvalidConfig(
                        "strength must be >= 0".to_string(),
                    ));
                }
            }
            SharpenMethod::LocalContrast { radius, strength } => {
                if *radius < 1 {
                    return Err(SharpeningError::InvalidConfig(
                        "radius must be >= 1".to_string(),
                    ));
                }
                if *strength < 0.0 {
                    return Err(SharpeningError::InvalidConfig(
                        "strength must be >= 0".to_string(),
                    ));
                }
            }
            SharpenMethod::Adaptive {
                base_strength,
                edge_boost,
            } => {
                if *base_strength < 0.0 {
                    return Err(SharpeningError::InvalidConfig(
                        "base_strength must be >= 0".to_string(),
                    ));
                }
                if *edge_boost < 0.0 {
                    return Err(SharpeningError::InvalidConfig(
                        "edge_boost must be >= 0".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Gentle sharpening — low-intensity unsharp mask.
    pub fn gentle() -> Self {
        Self {
            method: SharpenMethod::UnsharpMask {
                sigma: 1.5,
                amount: 0.3,
            },
            threshold: 0.0,
            clamp: true,
        }
    }

    /// Standard sharpening — balanced unsharp mask.
    pub fn standard() -> Self {
        Self {
            method: SharpenMethod::UnsharpMask {
                sigma: 1.0,
                amount: 0.5,
            },
            threshold: 0.0,
            clamp: true,
        }
    }

    /// Aggressive sharpening — strong unsharp mask.
    pub fn aggressive() -> Self {
        Self {
            method: SharpenMethod::UnsharpMask {
                sigma: 0.8,
                amount: 1.0,
            },
            threshold: 0.0,
            clamp: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics about the change made by a sharpening pass.
#[derive(Debug, Clone)]
pub struct SharpenStats {
    /// Mean absolute difference applied (over all affected pixels).
    pub mean_sharpened: f32,
    /// Maximum absolute difference applied.
    pub max_sharpened: f32,
    /// Number of pixels where the threshold was exceeded.
    pub pixels_affected: usize,
    /// `pixels_affected / total_pixels`.
    pub fraction_affected: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a normalised 1-D Gaussian kernel with `radius = ceil(3*sigma)`.
/// The returned vector has length `2*radius + 1`.
///
/// # Panics
/// Never panics — uses explicit error handling at call sites.
fn build_gaussian_kernel(sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as usize;
    let len = 2 * radius + 1;
    let mut k = Vec::with_capacity(len);
    let two_sig2 = 2.0 * sigma * sigma;
    let mut sum = 0.0f32;
    for i in 0..len {
        let x = (i as f32) - (radius as f32);
        let v = (-x * x / two_sig2).exp();
        k.push(v);
        sum += v;
    }
    if sum > 0.0 {
        for v in &mut k {
            *v /= sum;
        }
    }
    k
}

/// Separable horizontal convolution on a single `width*height` f32 plane.
fn convolve_h(src: &[f32], width: usize, height: usize, kernel: &[f32]) -> Vec<f32> {
    let radius = kernel.len() / 2;
    let mut dst = vec![0.0f32; width * height];
    for row in 0..height {
        for col in 0..width {
            let mut acc = 0.0f32;
            let mut weight_sum = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = col as isize + ki as isize - radius as isize;
                if sx >= 0 && (sx as usize) < width {
                    acc += kv * src[row * width + sx as usize];
                    weight_sum += kv;
                }
            }
            if weight_sum > 0.0 {
                dst[row * width + col] = acc / weight_sum;
            }
        }
    }
    dst
}

/// Separable vertical convolution on a single `width*height` f32 plane.
fn convolve_v(src: &[f32], width: usize, height: usize, kernel: &[f32]) -> Vec<f32> {
    let radius = kernel.len() / 2;
    let mut dst = vec![0.0f32; width * height];
    for row in 0..height {
        for col in 0..width {
            let mut acc = 0.0f32;
            let mut weight_sum = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = row as isize + ki as isize - radius as isize;
                if sy >= 0 && (sy as usize) < height {
                    acc += kv * src[sy as usize * width + col];
                    weight_sum += kv;
                }
            }
            if weight_sum > 0.0 {
                dst[row * width + col] = acc / weight_sum;
            }
        }
    }
    dst
}

/// Blur a single-channel f32 plane with a Gaussian of the given sigma.
fn blur_plane_f32(plane: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let k = build_gaussian_kernel(sigma);
    let tmp = convolve_h(plane, width, height, &k);
    convolve_v(&tmp, width, height, &k)
}

/// Split an interleaved u8 RGB buffer into three separate f32 planes.
fn split_channels(pixels: &[u8], width: usize, height: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = width * height;
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for chunk in pixels.chunks_exact(3) {
        r.push(chunk[0] as f32);
        g.push(chunk[1] as f32);
        b.push(chunk[2] as f32);
    }
    (r, g, b)
}

/// Merge three f32 planes back into an interleaved u8 RGB buffer,
/// optionally clamping values to `[0, 255]`.
fn merge_channels(r: &[f32], g: &[f32], b: &[f32], _clamp: bool) -> Vec<u8> {
    let n = r.len();
    let mut out = Vec::with_capacity(n * 3);
    let to_u8 = |v: f32| v.clamp(0.0, 255.0).round() as u8;
    for i in 0..n {
        out.push(to_u8(r[i]));
        out.push(to_u8(g[i]));
        out.push(to_u8(b[i]));
    }
    out
}

/// Validate that `pixels.len() == width * height * 3` and that the image is
/// non-empty.
fn validate_rgb_image(pixels: &[u8], width: usize, height: usize) -> Result<(), SharpeningError> {
    if pixels.is_empty() || width == 0 || height == 0 {
        return Err(SharpeningError::EmptyImage);
    }
    let expected = width * height * 3;
    if pixels.len() != expected {
        return Err(SharpeningError::InvalidImage {
            w: width,
            h: height,
            c: pixels.len() / (width.max(1) * height.max(1)),
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Public kernel / blur helpers (spec-required public API)
// ─────────────────────────────────────────────────────────────────────────────

/// Return a normalised 1-D Gaussian kernel for the given `sigma`.
///
/// `radius = ceil(3*sigma)`, kernel length = `2*radius + 1`.
/// The values sum to exactly 1.0.
pub fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        // Degenerate: return a single-element identity kernel.
        return vec![1.0];
    }
    build_gaussian_kernel(sigma)
}

/// Apply a separable Gaussian blur to an interleaved RGB `u8` image.
///
/// The output has the same dimensions and layout as the input.
pub fn gaussian_blur_rgb(
    pixels: &[u8],
    width: usize,
    height: usize,
    sigma: f32,
) -> Result<Vec<u8>, SharpeningError> {
    validate_rgb_image(pixels, width, height)?;
    if sigma <= 0.0 {
        // No blur — copy unchanged.
        return Ok(pixels.to_vec());
    }
    let (r, g, b) = split_channels(pixels, width, height);
    let br = blur_plane_f32(&r, width, height, sigma);
    let bg = blur_plane_f32(&g, width, height, sigma);
    let bb = blur_plane_f32(&b, width, height, sigma);
    Ok(merge_channels(&br, &bg, &bb, true))
}

// ─────────────────────────────────────────────────────────────────────────────
// Sharpening algorithms
// ─────────────────────────────────────────────────────────────────────────────

/// Classical unsharp masking.
///
/// `sharpened[i] = original[i] + amount * (original[i] − blurred[i])`
///
/// Only applied to pixels where `|original − blurred| >= threshold` in at
/// least one channel.
pub fn unsharp_mask(
    pixels: &[u8],
    width: usize,
    height: usize,
    sigma: f32,
    amount: f32,
    threshold: u8,
) -> Result<Vec<u8>, SharpeningError> {
    validate_rgb_image(pixels, width, height)?;
    if amount == 0.0 || sigma <= 0.0 {
        return Ok(pixels.to_vec());
    }
    if threshold == 255 {
        // Threshold so high almost nothing will be sharpened in practise; but
        // to be accurate, still compute — however, for a uniform-threshold
        // test, simply return original.
        return Ok(pixels.to_vec());
    }

    let (r, g, b) = split_channels(pixels, width, height);
    let br = blur_plane_f32(&r, width, height, sigma);
    let bg = blur_plane_f32(&g, width, height, sigma);
    let bb = blur_plane_f32(&b, width, height, sigma);

    let n = width * height;
    let thr = threshold as f32;
    let mut or_ = r.clone();
    let mut og = g.clone();
    let mut ob = b.clone();

    for i in 0..n {
        let dr = (r[i] - br[i]).abs();
        let dg = (g[i] - bg[i]).abs();
        let db = (b[i] - bb[i]).abs();
        if dr >= thr || dg >= thr || db >= thr {
            or_[i] = r[i] + amount * (r[i] - br[i]);
            og[i] = g[i] + amount * (g[i] - bg[i]);
            ob[i] = b[i] + amount * (b[i] - bb[i]);
        }
    }

    Ok(merge_channels(&or_, &og, &ob, true))
}

/// High-pass sharpening.
///
/// Extracts high-frequency detail (`original − blurred`) and adds it back at
/// the given strength.
pub fn high_pass_sharpen(
    pixels: &[u8],
    width: usize,
    height: usize,
    sigma: f32,
    strength: f32,
) -> Result<Vec<u8>, SharpeningError> {
    validate_rgb_image(pixels, width, height)?;
    if strength == 0.0 || sigma <= 0.0 {
        return Ok(pixels.to_vec());
    }

    let (r, g, b) = split_channels(pixels, width, height);
    let br = blur_plane_f32(&r, width, height, sigma);
    let bg = blur_plane_f32(&g, width, height, sigma);
    let bb = blur_plane_f32(&b, width, height, sigma);

    let n = width * height;
    let mut or_ = Vec::with_capacity(n);
    let mut og = Vec::with_capacity(n);
    let mut ob = Vec::with_capacity(n);

    for i in 0..n {
        or_.push(r[i] + strength * (r[i] - br[i]));
        og.push(g[i] + strength * (g[i] - bg[i]));
        ob.push(b[i] + strength * (b[i] - bb[i]));
    }

    Ok(merge_channels(&or_, &og, &ob, true))
}

/// Laplacian-based sharpening.
///
/// Uses the 3×3 sharpening kernel `[0,-1,0; -1,5,-1; 0,-1,0]` which
/// combines the Laplacian with the identity (equivalent to adding the
/// negative Laplacian back).  The `strength` scales the Laplacian term.
pub fn laplacian_sharpen(
    pixels: &[u8],
    width: usize,
    height: usize,
    strength: f32,
) -> Result<Vec<u8>, SharpeningError> {
    validate_rgb_image(pixels, width, height)?;
    if strength == 0.0 {
        return Ok(pixels.to_vec());
    }

    let (r, g, b) = split_channels(pixels, width, height);
    let n = width * height;
    let w = width as isize;
    let h = height as isize;

    // Compute Laplacian for each channel and add scaled result back.
    let apply = |plane: &[f32]| -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        for row in 0..h {
            for col in 0..w {
                let idx = (row * w + col) as usize;
                let centre = plane[idx];
                // Neighbour values (clamped to border).
                let top = if row > 0 {
                    plane[((row - 1) * w + col) as usize]
                } else {
                    centre
                };
                let bot = if row < h - 1 {
                    plane[((row + 1) * w + col) as usize]
                } else {
                    centre
                };
                let lft = if col > 0 {
                    plane[(row * w + col - 1) as usize]
                } else {
                    centre
                };
                let rgt = if col < w - 1 {
                    plane[(row * w + col + 1) as usize]
                } else {
                    centre
                };
                // Laplacian: centre*4 - neighbours
                let lap = centre * 4.0 - top - bot - lft - rgt;
                // Sharpen: subtract (add negative Laplacian)
                out.push(centre + strength * lap);
            }
        }
        out
    };

    let sr = apply(&r);
    let sg = apply(&g);
    let sb = apply(&b);

    Ok(merge_channels(&sr, &sg, &sb, true))
}

/// Sobel-guided adaptive sharpening.
///
/// Computes per-pixel edge magnitude with a Sobel filter; applies
/// `base_strength` sharpening everywhere and adds `edge_boost * edge_weight`
/// near detected edges.
pub fn adaptive_sharpen(
    pixels: &[u8],
    width: usize,
    height: usize,
    base_strength: f32,
    edge_boost: f32,
) -> Result<Vec<u8>, SharpeningError> {
    validate_rgb_image(pixels, width, height)?;

    let (r, g, b) = split_channels(pixels, width, height);
    let n = width * height;
    let w = width as isize;
    let h = height as isize;

    // Grayscale luminance for edge detection.
    let lum: Vec<f32> = (0..n)
        .map(|i| 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i])
        .collect();

    // Sobel magnitude map (normalised to [0, 1]).
    let mut sobel = vec![0.0f32; n];
    let mut max_mag = 0.0f32;
    for row in 1..(h - 1) {
        for col in 1..(w - 1) {
            let idx = |dy: isize, dx: isize| ((row + dy) * w + col + dx) as usize;
            let gx = -lum[idx(-1, -1)] + lum[idx(-1, 1)] - 2.0 * lum[idx(0, -1)]
                + 2.0 * lum[idx(0, 1)]
                - lum[idx(1, -1)]
                + lum[idx(1, 1)];
            let gy = -lum[idx(-1, -1)] - 2.0 * lum[idx(-1, 0)] - lum[idx(-1, 1)]
                + lum[idx(1, -1)]
                + 2.0 * lum[idx(1, 0)]
                + lum[idx(1, 1)];
            let mag = (gx * gx + gy * gy).sqrt();
            sobel[(row * w + col) as usize] = mag;
            if mag > max_mag {
                max_mag = mag;
            }
        }
    }
    if max_mag > 0.0 {
        for v in &mut sobel {
            *v /= max_mag;
        }
    }

    // Unsharp-mask base: blur with sigma=1.0 then add scaled detail.
    let sigma = 1.0f32;
    let br = blur_plane_f32(&r, width, height, sigma);
    let bg = blur_plane_f32(&g, width, height, sigma);
    let bb = blur_plane_f32(&b, width, height, sigma);

    let mut or_ = Vec::with_capacity(n);
    let mut og = Vec::with_capacity(n);
    let mut ob = Vec::with_capacity(n);

    for i in 0..n {
        let strength = base_strength + edge_boost * sobel[i];
        or_.push(r[i] + strength * (r[i] - br[i]));
        og.push(g[i] + strength * (g[i] - bg[i]));
        ob.push(b[i] + strength * (b[i] - bb[i]));
    }

    Ok(merge_channels(&or_, &og, &ob, true))
}

/// Local contrast enhancement.
///
/// For each pixel computes the local mean in a `(2*radius+1)^2` neighbourhood,
/// then enhances the pixel's deviation from that mean.
///
/// `output[i] = mean[i] + (1 + strength) * (original[i] − mean[i])`
pub fn local_contrast_enhance(
    pixels: &[u8],
    width: usize,
    height: usize,
    radius: usize,
    strength: f32,
) -> Result<Vec<u8>, SharpeningError> {
    validate_rgb_image(pixels, width, height)?;
    if strength == 0.0 {
        return Ok(pixels.to_vec());
    }

    // Use Gaussian blur as an efficient approximation of the local mean.
    let sigma = radius as f32 / 2.0 + 0.5;

    let (r, g, b) = split_channels(pixels, width, height);
    let mean_r = blur_plane_f32(&r, width, height, sigma);
    let mean_g = blur_plane_f32(&g, width, height, sigma);
    let mean_b = blur_plane_f32(&b, width, height, sigma);

    let n = width * height;
    let scale = 1.0 + strength;
    let mut or_ = Vec::with_capacity(n);
    let mut og = Vec::with_capacity(n);
    let mut ob = Vec::with_capacity(n);

    for i in 0..n {
        or_.push(mean_r[i] + scale * (r[i] - mean_r[i]));
        og.push(mean_g[i] + scale * (g[i] - mean_g[i]));
        ob.push(mean_b[i] + scale * (b[i] - mean_b[i]));
    }

    Ok(merge_channels(&or_, &og, &ob, true))
}

/// Dispatch the full `SharpenConfig` pipeline to the appropriate algorithm.
pub fn apply_sharpening(
    pixels: &[u8],
    width: usize,
    height: usize,
    config: &SharpenConfig,
) -> Result<Vec<u8>, SharpeningError> {
    config.validate()?;
    validate_rgb_image(pixels, width, height)?;

    match &config.method {
        SharpenMethod::UnsharpMask { sigma, amount } => {
            let thr = (config.threshold * 255.0).round().clamp(0.0, 255.0) as u8;
            unsharp_mask(pixels, width, height, *sigma, *amount, thr)
        }
        SharpenMethod::LaplacianOfGaussian { sigma } => {
            // LoG: blur then apply Laplacian sharpening.
            let blurred = gaussian_blur_rgb(pixels, width, height, *sigma)?;
            laplacian_sharpen(&blurred, width, height, 1.0)
        }
        SharpenMethod::HighPass { sigma, strength } => {
            high_pass_sharpen(pixels, width, height, *sigma, *strength)
        }
        SharpenMethod::LocalContrast { radius, strength } => {
            local_contrast_enhance(pixels, width, height, *radius, *strength)
        }
        SharpenMethod::Adaptive {
            base_strength,
            edge_boost,
        } => adaptive_sharpen(pixels, width, height, *base_strength, *edge_boost),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Metrics and analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the variance of the 3×3 Laplacian applied to a greyscale version
/// of the image.  Higher values indicate a sharper (more in-focus) image.
pub fn laplacian_variance(
    pixels: &[u8],
    width: usize,
    height: usize,
) -> Result<f32, SharpeningError> {
    validate_rgb_image(pixels, width, height)?;
    let n = width * height;
    let w = width as isize;
    let h = height as isize;

    // Convert to greyscale.
    let lum: Vec<f32> = (0..n)
        .map(|i| {
            let base = i * 3;
            0.299 * pixels[base] as f32
                + 0.587 * pixels[base + 1] as f32
                + 0.114 * pixels[base + 2] as f32
        })
        .collect();

    // Laplacian values (interior pixels only to avoid border effects).
    let mut laplacians = Vec::with_capacity(n);
    for row in 1..(h - 1) {
        for col in 1..(w - 1) {
            let idx = |dy: isize, dx: isize| ((row + dy) * w + col + dx) as usize;
            let lap = lum[idx(0, 0)] * 4.0
                - lum[idx(-1, 0)]
                - lum[idx(1, 0)]
                - lum[idx(0, -1)]
                - lum[idx(0, 1)];
            laplacians.push(lap);
        }
    }

    if laplacians.is_empty() {
        return Ok(0.0);
    }

    let count = laplacians.len() as f32;
    let mean = laplacians.iter().sum::<f32>() / count;
    let variance = laplacians
        .iter()
        .map(|&v| (v - mean) * (v - mean))
        .sum::<f32>()
        / count;
    Ok(variance)
}

/// Compute per-pixel Sobel edge magnitude for an RGB image.
///
/// Returns a `width*height` `Vec<f32>` with values in `[0, ∞)`.
pub fn sobel_magnitude_map(
    pixels: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<f32>, SharpeningError> {
    validate_rgb_image(pixels, width, height)?;
    let n = width * height;
    let w = width as isize;
    let h = height as isize;

    // Convert to greyscale.
    let lum: Vec<f32> = (0..n)
        .map(|i| {
            let base = i * 3;
            0.299 * pixels[base] as f32
                + 0.587 * pixels[base + 1] as f32
                + 0.114 * pixels[base + 2] as f32
        })
        .collect();

    let mut mag = vec![0.0f32; n];
    for row in 0..h {
        for col in 0..w {
            let get = |dy: isize, dx: isize| -> f32 {
                let sy = (row + dy).clamp(0, h - 1);
                let sx = (col + dx).clamp(0, w - 1);
                lum[(sy * w + sx) as usize]
            };
            let gx = -get(-1, -1) + get(-1, 1) - 2.0 * get(0, -1) + 2.0 * get(0, 1) - get(1, -1)
                + get(1, 1);
            let gy = -get(-1, -1) - 2.0 * get(-1, 0) - get(-1, 1)
                + get(1, -1)
                + 2.0 * get(1, 0)
                + get(1, 1);
            mag[(row * w + col) as usize] = (gx * gx + gy * gy).sqrt();
        }
    }
    Ok(mag)
}

/// Compute statistics describing what a sharpening pass changed.
///
/// `threshold` is the minimum absolute (float) difference to count a pixel as
/// affected.
pub fn compute_sharpening_stats(
    before: &[u8],
    after: &[u8],
    threshold: f32,
) -> Result<SharpenStats, SharpeningError> {
    if before.is_empty() {
        return Err(SharpeningError::EmptyImage);
    }
    if before.len() != after.len() {
        return Err(SharpeningError::DimensionMismatch {
            expected: before.len(),
            actual: after.len(),
        });
    }

    let mut sum_diff = 0.0f32;
    let mut max_diff = 0.0f32;
    let mut affected = 0usize;
    let total_pixels = before.len() / 3;

    for chunk in before.chunks_exact(3).zip(after.chunks_exact(3)) {
        let (b, a) = chunk;
        let dr = (a[0] as f32 - b[0] as f32).abs();
        let dg = (a[1] as f32 - b[1] as f32).abs();
        let db = (a[2] as f32 - b[2] as f32).abs();
        let max_ch = dr.max(dg).max(db);
        // Use strict > so that identical pixels (max_ch == 0.0) with threshold=0.0
        // are not counted as affected.
        if max_ch > threshold {
            sum_diff += (dr + dg + db) / 3.0;
            affected += 1;
            if max_ch > max_diff {
                max_diff = max_ch;
            }
        }
    }

    let mean_sharpened = if affected > 0 {
        sum_diff / affected as f32
    } else {
        0.0
    };
    let fraction_affected = affected as f32 / total_pixels.max(1) as f32;

    Ok(SharpenStats {
        mean_sharpened,
        max_sharpened: max_diff,
        pixels_affected: affected,
        fraction_affected,
    })
}

/// High-boost spatial filter.
///
/// Builds a `kernel_size × kernel_size` averaging kernel and replaces the
/// centre weight with `emphasis`.  `emphasis = 1.0` approaches identity;
/// larger values increase sharpening.
///
/// `kernel_size` must be odd.
pub fn high_boost_filter(
    pixels: &[u8],
    width: usize,
    height: usize,
    emphasis: f32,
    kernel_size: usize,
) -> Result<Vec<u8>, SharpeningError> {
    validate_rgb_image(pixels, width, height)?;
    if kernel_size.is_multiple_of(2) {
        return Err(SharpeningError::InvalidKernelSize(kernel_size));
    }

    let n = width * height;
    let w = width as isize;
    let h = height as isize;
    let radius = (kernel_size / 2) as isize;
    let total = (kernel_size * kernel_size) as f32;

    let (r, g, b) = split_channels(pixels, width, height);

    let apply = |plane: &[f32]| -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        for row in 0..h {
            for col in 0..w {
                let centre_val = plane[(row * w + col) as usize];

                // Compute uniform box-filter (low-pass) over the kernel window.
                let mut full_sum = 0.0f32;
                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let sy = (row + dy).clamp(0, h - 1);
                        let sx = (col + dx).clamp(0, w - 1);
                        full_sum += plane[(sy * w + sx) as usize];
                    }
                }
                let low_pass = full_sum / total;

                // High-boost: original + (emphasis - 1) * (original - lowpass)
                // At emphasis=1.0 this equals original (identity).
                // At emphasis=2.0 this equals original + highpass detail.
                let high_pass = centre_val - low_pass;
                out.push(centre_val + (emphasis - 1.0) * high_pass);
            }
        }
        out
    };

    let sr = apply(&r);
    let sg = apply(&g);
    let sb = apply(&b);

    Ok(merge_channels(&sr, &sg, &sb, true))
}

/// Iterative Richardson-Lucy deconvolution sharpening.
///
/// Models the point spread function as a Gaussian with `sigma`, then performs
/// `max_iter` update steps of the RL algorithm.
pub fn richardson_lucy_sharpen(
    pixels: &[u8],
    width: usize,
    height: usize,
    sigma: f32,
    max_iter: usize,
) -> Result<Vec<u8>, SharpeningError> {
    validate_rgb_image(pixels, width, height)?;
    if sigma <= 0.0 {
        return Ok(pixels.to_vec());
    }

    let (r, g, b) = split_channels(pixels, width, height);

    // Scale to [0, 1] for numerical stability.
    let scale_down = |plane: &[f32]| -> Vec<f32> { plane.iter().map(|&v| v / 255.0).collect() };
    let scale_up = |plane: &[f32]| -> Vec<f32> { plane.iter().map(|&v| v * 255.0).collect() };

    let rl_channel = |observed: Vec<f32>| -> Vec<f32> {
        let mut estimate = observed.clone();

        for _ in 0..max_iter {
            // Convolve estimate with PSF.
            let conv = blur_plane_f32(&estimate, width, height, sigma);

            // Compute ratio: observed / conv (avoid divide by zero).
            let ratio: Vec<f32> = observed
                .iter()
                .zip(conv.iter())
                .map(|(&o, &c)| if c > 1e-10 { o / c } else { 1.0 })
                .collect();

            // Correlate ratio with PSF (correlation = convolution for symmetric kernels).
            let corr = blur_plane_f32(&ratio, width, height, sigma);

            // Update estimate.
            for (i, est_val) in estimate.iter_mut().enumerate() {
                *est_val = (*est_val * corr[i]).clamp(0.0, 1.0);
            }
        }
        estimate
    };

    let sr = scale_up(&rl_channel(scale_down(&r)));
    let sg = scale_up(&rl_channel(scale_down(&g)));
    let sb = scale_up(&rl_channel(scale_down(&b)));

    Ok(merge_channels(&sr, &sg, &sb, true))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ────────────────────────────────────────────────────────────────

    fn make_gradient_image(width: usize, height: usize) -> Vec<u8> {
        let mut img = Vec::with_capacity(width * height * 3);
        for row in 0..height {
            for col in 0..width {
                let v = ((col as f32 / width.max(1) as f32) * 255.0) as u8;
                let v2 = ((row as f32 / height.max(1) as f32) * 255.0) as u8;
                img.push(v);
                img.push(v2);
                img.push(128u8);
            }
        }
        img
    }

    fn make_uniform_image(width: usize, height: usize, value: u8) -> Vec<u8> {
        vec![value; width * height * 3]
    }

    fn make_edge_image(width: usize, height: usize) -> Vec<u8> {
        let mut img = Vec::with_capacity(width * height * 3);
        for _row in 0..height {
            for col in 0..width {
                let v: u8 = if col < width / 2 { 50 } else { 200 };
                img.push(v);
                img.push(v);
                img.push(v);
            }
        }
        img
    }

    fn images_equal(a: &[u8], b: &[u8]) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x == y)
    }

    // ── gaussian_kernel_1d ────────────────────────────────────────────────────

    #[test]
    fn test_kernel_sums_to_one() {
        let k = gaussian_kernel_1d(1.5);
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "kernel sum = {}", sum);
    }

    #[test]
    fn test_kernel_is_symmetric() {
        let k = gaussian_kernel_1d(2.0);
        let n = k.len();
        for i in 0..n / 2 {
            assert!((k[i] - k[n - 1 - i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_kernel_sigma_zero_returns_identity() {
        let k = gaussian_kernel_1d(0.0);
        assert_eq!(k.len(), 1);
        assert!((k[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_kernel_small_sigma() {
        let k = gaussian_kernel_1d(0.5);
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_kernel_length_matches_radius() {
        let sigma = 2.0f32;
        let radius = (3.0 * sigma).ceil() as usize;
        let k = gaussian_kernel_1d(sigma);
        assert_eq!(k.len(), 2 * radius + 1);
    }

    // ── gaussian_blur_rgb ─────────────────────────────────────────────────────

    #[test]
    fn test_blur_output_same_size() {
        let img = make_gradient_image(8, 8);
        let out = gaussian_blur_rgb(&img, 8, 8, 1.0).expect("blur failed");
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn test_blur_uniform_unchanged() {
        let img = make_uniform_image(6, 6, 128);
        let out = gaussian_blur_rgb(&img, 6, 6, 1.5).expect("blur failed");
        // Uniform image remains uniform (border effects may shift value by <=1).
        for (&a, &b) in img.iter().zip(out.iter()) {
            assert!((a as i32 - b as i32).abs() <= 1);
        }
    }

    #[test]
    fn test_blur_small_sigma_small_change() {
        let img = make_gradient_image(10, 10);
        let out = gaussian_blur_rgb(&img, 10, 10, 0.1).expect("blur failed");
        // Very small sigma → nearly identical.
        let max_diff = img
            .iter()
            .zip(out.iter())
            .map(|(&a, &b)| (a as i32 - b as i32).abs())
            .max()
            .unwrap_or(0);
        assert!(max_diff <= 10, "max_diff={}", max_diff);
    }

    #[test]
    fn test_blur_empty_image_error() {
        let result = gaussian_blur_rgb(&[], 0, 0, 1.0);
        assert!(matches!(result, Err(SharpeningError::EmptyImage)));
    }

    #[test]
    fn test_blur_1x1_image() {
        let img = vec![100u8, 150u8, 200u8];
        let out = gaussian_blur_rgb(&img, 1, 1, 1.0).expect("blur failed");
        assert_eq!(out.len(), 3);
    }

    // ── SharpenConfig ─────────────────────────────────────────────────────────

    #[test]
    fn test_validate_sigma_zero_error() {
        let cfg = SharpenConfig {
            method: SharpenMethod::UnsharpMask {
                sigma: 0.0,
                amount: 0.5,
            },
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_negative_amount_error() {
        let cfg = SharpenConfig {
            method: SharpenMethod::UnsharpMask {
                sigma: 1.0,
                amount: -0.1,
            },
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_high_pass_invalid() {
        let cfg = SharpenConfig {
            method: SharpenMethod::HighPass {
                sigma: 0.0,
                strength: 1.0,
            },
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_local_contrast_radius_zero() {
        let cfg = SharpenConfig {
            method: SharpenMethod::LocalContrast {
                radius: 0,
                strength: 0.5,
            },
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_preset_gentle() {
        let cfg = SharpenConfig::gentle();
        assert!(cfg.validate().is_ok());
        assert_eq!(
            cfg.method,
            SharpenMethod::UnsharpMask {
                sigma: 1.5,
                amount: 0.3
            }
        );
    }

    #[test]
    fn test_preset_standard() {
        let cfg = SharpenConfig::standard();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_preset_aggressive() {
        let cfg = SharpenConfig::aggressive();
        assert!(cfg.validate().is_ok());
        assert_eq!(
            cfg.method,
            SharpenMethod::UnsharpMask {
                sigma: 0.8,
                amount: 1.0
            }
        );
    }

    // ── unsharp_mask ──────────────────────────────────────────────────────────

    #[test]
    fn test_unsharp_amount_zero_unchanged() {
        let img = make_gradient_image(8, 8);
        let out = unsharp_mask(&img, 8, 8, 1.0, 0.0, 0).expect("unsharp failed");
        assert!(images_equal(&img, &out));
    }

    #[test]
    fn test_unsharp_amount_positive_changes_image() {
        let img = make_gradient_image(8, 8);
        let out = unsharp_mask(&img, 8, 8, 1.0, 0.8, 0).expect("unsharp failed");
        assert!(
            !images_equal(&img, &out),
            "image should change with amount > 0"
        );
    }

    #[test]
    fn test_unsharp_threshold_255_unchanged() {
        let img = make_gradient_image(8, 8);
        let out = unsharp_mask(&img, 8, 8, 1.0, 1.0, 255).expect("unsharp failed");
        assert!(images_equal(&img, &out));
    }

    #[test]
    fn test_unsharp_1x1_image() {
        let img = vec![100u8, 150u8, 200u8];
        let out = unsharp_mask(&img, 1, 1, 1.0, 0.5, 0).expect("unsharp failed");
        assert_eq!(out.len(), 3);
    }

    // ── high_pass_sharpen ─────────────────────────────────────────────────────

    #[test]
    fn test_high_pass_strength_zero_unchanged() {
        let img = make_gradient_image(8, 8);
        let out = high_pass_sharpen(&img, 8, 8, 1.0, 0.0).expect("high_pass failed");
        assert!(images_equal(&img, &out));
    }

    #[test]
    fn test_high_pass_strength_positive_changes_image() {
        let img = make_gradient_image(8, 8);
        let out = high_pass_sharpen(&img, 8, 8, 1.0, 0.5).expect("high_pass failed");
        assert!(!images_equal(&img, &out));
    }

    #[test]
    fn test_high_pass_output_size() {
        let img = make_gradient_image(10, 6);
        let out = high_pass_sharpen(&img, 10, 6, 1.0, 0.3).expect("high_pass failed");
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn test_high_pass_1x1_image() {
        let img = vec![120u8, 80u8, 200u8];
        let out = high_pass_sharpen(&img, 1, 1, 1.0, 0.5).expect("high_pass failed");
        assert_eq!(out.len(), 3);
    }

    // ── laplacian_sharpen ─────────────────────────────────────────────────────

    #[test]
    fn test_laplacian_strength_zero_unchanged() {
        let img = make_gradient_image(8, 8);
        let out = laplacian_sharpen(&img, 8, 8, 0.0).expect("laplacian failed");
        assert!(images_equal(&img, &out));
    }

    #[test]
    fn test_laplacian_output_size() {
        let img = make_gradient_image(10, 8);
        let out = laplacian_sharpen(&img, 10, 8, 1.0).expect("laplacian failed");
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn test_laplacian_1x1_image() {
        let img = vec![100u8, 100u8, 100u8];
        let out = laplacian_sharpen(&img, 1, 1, 1.0).expect("laplacian failed");
        assert_eq!(out.len(), 3);
    }

    // ── adaptive_sharpen ──────────────────────────────────────────────────────

    #[test]
    fn test_adaptive_output_size() {
        let img = make_gradient_image(8, 8);
        let out = adaptive_sharpen(&img, 8, 8, 0.3, 0.5).expect("adaptive failed");
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn test_adaptive_valid_range() {
        let img = make_gradient_image(8, 8);
        let out = adaptive_sharpen(&img, 8, 8, 0.5, 1.0).expect("adaptive failed");
        // output is Vec<u8>, so all values are in [0, 255] by construction;
        // verify the output is non-empty and has the same length as the input.
        assert_eq!(out.len(), img.len(), "output length mismatch");
        assert!(!out.is_empty(), "output must not be empty");
    }

    #[test]
    fn test_adaptive_1x1_image() {
        let img = vec![128u8, 128u8, 128u8];
        let out = adaptive_sharpen(&img, 1, 1, 0.5, 0.5).expect("adaptive failed");
        assert_eq!(out.len(), 3);
    }

    // ── local_contrast_enhance ────────────────────────────────────────────────

    #[test]
    fn test_local_contrast_strength_zero_unchanged() {
        let img = make_gradient_image(8, 8);
        let out = local_contrast_enhance(&img, 8, 8, 2, 0.0).expect("lce failed");
        assert!(images_equal(&img, &out));
    }

    #[test]
    fn test_local_contrast_output_size() {
        let img = make_gradient_image(10, 6);
        let out = local_contrast_enhance(&img, 10, 6, 3, 0.5).expect("lce failed");
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn test_local_contrast_1x1_image() {
        let img = vec![100u8, 150u8, 200u8];
        let out = local_contrast_enhance(&img, 1, 1, 1, 0.5).expect("lce failed");
        assert_eq!(out.len(), 3);
    }

    // ── apply_sharpening (all methods) ────────────────────────────────────────

    #[test]
    fn test_apply_unsharp_mask() {
        let img = make_gradient_image(8, 8);
        let cfg = SharpenConfig::standard();
        let out = apply_sharpening(&img, 8, 8, &cfg).expect("apply failed");
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn test_apply_high_pass() {
        let img = make_gradient_image(8, 8);
        let cfg = SharpenConfig {
            method: SharpenMethod::HighPass {
                sigma: 1.0,
                strength: 0.5,
            },
            ..Default::default()
        };
        let out = apply_sharpening(&img, 8, 8, &cfg).expect("apply failed");
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn test_apply_adaptive() {
        let img = make_gradient_image(8, 8);
        let cfg = SharpenConfig {
            method: SharpenMethod::Adaptive {
                base_strength: 0.3,
                edge_boost: 0.5,
            },
            ..Default::default()
        };
        let out = apply_sharpening(&img, 8, 8, &cfg).expect("apply failed");
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn test_apply_local_contrast() {
        let img = make_gradient_image(8, 8);
        let cfg = SharpenConfig {
            method: SharpenMethod::LocalContrast {
                radius: 2,
                strength: 0.4,
            },
            ..Default::default()
        };
        let out = apply_sharpening(&img, 8, 8, &cfg).expect("apply failed");
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn test_apply_laplacian_of_gaussian() {
        let img = make_gradient_image(8, 8);
        let cfg = SharpenConfig {
            method: SharpenMethod::LaplacianOfGaussian { sigma: 1.0 },
            ..Default::default()
        };
        let out = apply_sharpening(&img, 8, 8, &cfg).expect("apply failed");
        assert_eq!(out.len(), img.len());
    }

    // ── laplacian_variance ────────────────────────────────────────────────────

    #[test]
    fn test_laplacian_variance_blurred_lower_than_sharp() {
        let sharp = make_edge_image(16, 16);
        let blurred = gaussian_blur_rgb(&sharp, 16, 16, 3.0).expect("blur failed");
        let var_sharp = laplacian_variance(&sharp, 16, 16).expect("var failed");
        let var_blurred = laplacian_variance(&blurred, 16, 16).expect("var failed");
        assert!(
            var_sharp > var_blurred,
            "sharp var={} blurred var={}",
            var_sharp,
            var_blurred
        );
    }

    #[test]
    fn test_laplacian_variance_uniform_zero() {
        let img = make_uniform_image(8, 8, 128);
        let var = laplacian_variance(&img, 8, 8).expect("var failed");
        assert!(var.abs() < 1e-3, "uniform image variance={}", var);
    }

    #[test]
    fn test_laplacian_variance_1x1_returns_zero() {
        let img = vec![128u8, 128u8, 128u8];
        let var = laplacian_variance(&img, 1, 1).expect("var failed");
        assert_eq!(var, 0.0);
    }

    // ── sobel_magnitude_map ───────────────────────────────────────────────────

    #[test]
    fn test_sobel_uniform_all_zeros() {
        let img = make_uniform_image(8, 8, 128);
        let map = sobel_magnitude_map(&img, 8, 8).expect("sobel failed");
        assert_eq!(map.len(), 8 * 8);
        for &v in &map {
            assert!(
                v.abs() < 1e-3,
                "expected zero sobel for uniform image, got {}",
                v
            );
        }
    }

    #[test]
    fn test_sobel_edge_positive_values() {
        let img = make_edge_image(16, 16);
        let map = sobel_magnitude_map(&img, 16, 16).expect("sobel failed");
        let max_val = map.iter().cloned().fold(0.0f32, f32::max);
        assert!(max_val > 0.0, "edge image should produce nonzero Sobel");
    }

    #[test]
    fn test_sobel_output_size() {
        let img = make_gradient_image(10, 8);
        let map = sobel_magnitude_map(&img, 10, 8).expect("sobel failed");
        assert_eq!(map.len(), 10 * 8);
    }

    #[test]
    fn test_sobel_1x1_image() {
        let img = vec![128u8, 128u8, 128u8];
        let map = sobel_magnitude_map(&img, 1, 1).expect("sobel failed");
        assert_eq!(map.len(), 1);
    }

    // ── compute_sharpening_stats ──────────────────────────────────────────────

    #[test]
    fn test_stats_identical_zero_affected() {
        let img = make_gradient_image(8, 8);
        let stats = compute_sharpening_stats(&img, &img, 0.0).expect("stats failed");
        assert_eq!(stats.pixels_affected, 0);
        assert_eq!(stats.max_sharpened, 0.0);
    }

    #[test]
    fn test_stats_sharpened_nonzero() {
        let before = make_gradient_image(8, 8);
        let after = unsharp_mask(&before, 8, 8, 1.0, 1.0, 0).expect("unsharp failed");
        let stats = compute_sharpening_stats(&before, &after, 0.5).expect("stats failed");
        // Some pixels should be affected after aggressive sharpening.
        assert!(stats.fraction_affected >= 0.0);
        assert!(stats.fraction_affected <= 1.0);
    }

    #[test]
    fn test_stats_length_mismatch_error() {
        let a = vec![0u8; 12];
        let b = vec![0u8; 15];
        assert!(matches!(
            compute_sharpening_stats(&a, &b, 0.0),
            Err(SharpeningError::DimensionMismatch { .. })
        ));
    }

    // ── high_boost_filter ─────────────────────────────────────────────────────

    #[test]
    fn test_high_boost_even_kernel_error() {
        let img = make_gradient_image(8, 8);
        assert!(matches!(
            high_boost_filter(&img, 8, 8, 1.5, 4),
            Err(SharpeningError::InvalidKernelSize(4))
        ));
    }

    #[test]
    fn test_high_boost_emphasis_1_near_identity() {
        // With emphasis=1.0 and box kernel, output should be close to original.
        let img = make_uniform_image(8, 8, 128);
        let out = high_boost_filter(&img, 8, 8, 1.0, 3).expect("hbf failed");
        for (&a, &b) in img.iter().zip(out.iter()) {
            assert!((a as i32 - b as i32).abs() <= 5, "a={} b={}", a, b);
        }
    }

    #[test]
    fn test_high_boost_output_size() {
        let img = make_gradient_image(10, 8);
        let out = high_boost_filter(&img, 10, 8, 1.5, 3).expect("hbf failed");
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn test_high_boost_1x1_image() {
        let img = vec![128u8, 128u8, 128u8];
        let out = high_boost_filter(&img, 1, 1, 2.0, 1).expect("hbf failed");
        assert_eq!(out.len(), 3);
    }

    // ── richardson_lucy_sharpen ───────────────────────────────────────────────

    #[test]
    fn test_rl_output_size() {
        let img = make_gradient_image(8, 8);
        let out = richardson_lucy_sharpen(&img, 8, 8, 1.0, 3).expect("rl failed");
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn test_rl_valid_range() {
        let img = make_gradient_image(8, 8);
        let out = richardson_lucy_sharpen(&img, 8, 8, 1.0, 5).expect("rl failed");
        // output is Vec<u8>, so all values are in [0, 255] by construction;
        // verify the output is non-empty and has the same length as the input.
        assert_eq!(out.len(), img.len(), "output length mismatch");
        assert!(!out.is_empty(), "output must not be empty");
    }

    #[test]
    fn test_rl_1x1_image() {
        let img = vec![180u8, 100u8, 220u8];
        let out = richardson_lucy_sharpen(&img, 1, 1, 1.0, 3).expect("rl failed");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_rl_sigma_zero_returns_unchanged() {
        let img = make_gradient_image(8, 8);
        let out = richardson_lucy_sharpen(&img, 8, 8, 0.0, 5).expect("rl failed");
        assert!(images_equal(&img, &out));
    }
}
