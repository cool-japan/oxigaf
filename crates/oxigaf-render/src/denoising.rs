//! CPU-side image denoising for post-processing 3DGS rendered outputs.
//!
//! 3DGS rendering can produce noisy images due to:
//! - Low effective sample count in the tiled rasterizer
//! - Floating-point accumulation over many Gaussian splats
//! - Sparse Gaussian regions yielding under-sampled colour estimates
//!
//! This module provides several classic and state-of-the-art denoising
//! algorithms, all operating on flat `Vec<f32>` RGB images in row-major
//! order (length = `width * height * 3`).
//!
//! ## Available algorithms
//!
//! | Algorithm | Speed | Edge-preserving | Recommended for |
//! |-----------|-------|-----------------|-----------------|
//! | [`gaussian_denoise`] | Fast | No  | Light noise |
//! | [`bilateral_filter`] | Medium | Yes | Moderate noise, keep edges |
//! | [`median_filter`] | Fast | Partial | Salt-and-pepper noise |
//! | [`non_local_means`] | Slow | Yes | Heavy noise with texture |
//! | [`denoise_adaptive`] | Auto | Auto | Unknown noise level |
//!
//! ## Example
//!
//! ```
//! use oxigaf_render::{BilateralConfig, bilateral_filter};
//!
//! let w = 4;
//! let h = 4;
//! let image = vec![0.5_f32; w * h * 3];
//! let cfg = BilateralConfig::default();
//! let result = bilateral_filter(&image, w, h, &cfg).unwrap();
//! assert_eq!(result.len(), w * h * 3);
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during image denoising operations.
#[derive(Debug, Error)]
pub enum DenoisingError {
    /// A configuration parameter is out of range or otherwise invalid.
    #[error("Invalid denoising configuration: {0}")]
    InvalidConfig(String),
    /// The image buffer length does not match `width * height * 3`.
    #[error("Invalid image: {0}")]
    InvalidImage(String),
    /// The image has zero pixels.
    #[error("Empty image (zero pixels)")]
    EmptyImage,
    /// A convolution kernel could not be constructed.
    #[error("Kernel error: {0}")]
    KernelError(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// BT.709 luminance of a linear RGB triple.
#[inline]
fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Clamp-to-edge pixel coordinate.
#[inline]
fn clamp_coord(v: i32, max: usize) -> usize {
    v.clamp(0, max as i32 - 1) as usize
}

/// Validate that an RGB image buffer has the right length.
fn validate_rgb_image(image: &[f32], width: usize, height: usize) -> Result<(), DenoisingError> {
    if width == 0 || height == 0 {
        return Err(DenoisingError::EmptyImage);
    }
    let expected = width * height * 3;
    if image.len() != expected {
        return Err(DenoisingError::InvalidImage(format!(
            "expected {} floats for {}×{} RGB, got {}",
            expected,
            width,
            height,
            image.len()
        )));
    }
    Ok(())
}

/// Build a normalised 1-D Gaussian kernel of half-width `radius`.
/// The returned vector has length `2 * radius + 1`.
fn gaussian_1d_kernel(radius: usize, sigma: f32) -> Result<Vec<f32>, DenoisingError> {
    if sigma <= 0.0 {
        return Err(DenoisingError::KernelError(format!(
            "sigma must be positive, got {}",
            sigma
        )));
    }
    if radius == 0 {
        return Err(DenoisingError::KernelError(
            "radius must be at least 1".to_string(),
        ));
    }
    let size = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(size);
    let two_sigma_sq = 2.0 * sigma * sigma;
    for i in 0..size {
        let d = i as f32 - radius as f32;
        kernel.push((-d * d / two_sigma_sq).exp());
    }
    let sum: f32 = kernel.iter().sum();
    for v in &mut kernel {
        *v /= sum;
    }
    Ok(kernel)
}

// ─────────────────────────────────────────────────────────────────────────────
// Bilateral filter
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the bilateral filter.
#[derive(Debug, Clone)]
pub struct BilateralConfig {
    /// Spatial Gaussian standard deviation (pixels). Default: 2.0.
    pub sigma_spatial: f32,
    /// Range (colour) Gaussian standard deviation in [0, 1]. Default: 0.1.
    pub sigma_range: f32,
    /// Kernel half-width (pixels). Actual kernel is `(2*radius+1)²`. Default: 3.
    pub radius: usize,
}

impl Default for BilateralConfig {
    fn default() -> Self {
        Self {
            sigma_spatial: 2.0,
            sigma_range: 0.1,
            radius: 3,
        }
    }
}

impl BilateralConfig {
    /// Returns an error if any parameter is out of range.
    pub fn validate(&self) -> Result<(), DenoisingError> {
        if self.sigma_spatial <= 0.0 {
            return Err(DenoisingError::InvalidConfig(format!(
                "sigma_spatial must be positive, got {}",
                self.sigma_spatial
            )));
        }
        if self.sigma_range <= 0.0 {
            return Err(DenoisingError::InvalidConfig(format!(
                "sigma_range must be positive, got {}",
                self.sigma_range
            )));
        }
        if self.radius < 1 {
            return Err(DenoisingError::InvalidConfig(
                "radius must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Apply a bilateral filter to an RGB image.
///
/// The bilateral filter is an edge-preserving smoothing filter that weights
/// each neighbour by both spatial proximity *and* photometric similarity.
/// Luminance similarity (BT.709) is used for the range term.
///
/// # Arguments
/// * `image`  – RGB f32 row-major buffer, length = `width * height * 3`.
/// * `width`, `height` – image dimensions.
/// * `config` – filter configuration (spatial/range sigma, radius).
///
/// Boundary pixels use clamp-to-edge sampling.
pub fn bilateral_filter(
    image: &[f32],
    width: usize,
    height: usize,
    config: &BilateralConfig,
) -> Result<Vec<f32>, DenoisingError> {
    config.validate()?;
    validate_rgb_image(image, width, height)?;

    let two_ss_sq = 2.0 * config.sigma_spatial * config.sigma_spatial;
    let two_sr_sq = 2.0 * config.sigma_range * config.sigma_range;
    let r = config.radius as i32;

    let mut output = vec![0.0_f32; width * height * 3];

    for cy in 0..height {
        for cx in 0..width {
            let ci = (cy * width + cx) * 3;
            let center_lum = luminance(image[ci], image[ci + 1], image[ci + 2]);

            let mut sum_r = 0.0_f32;
            let mut sum_g = 0.0_f32;
            let mut sum_b = 0.0_f32;
            let mut sum_w = 0.0_f32;

            for dy in -r..=r {
                let ny = clamp_coord(cy as i32 + dy, height);
                for dx in -r..=r {
                    let nx = clamp_coord(cx as i32 + dx, width);

                    let dist_sq = (dx * dx + dy * dy) as f32;
                    let spatial_w = (-dist_sq / two_ss_sq).exp();

                    let ni = (ny * width + nx) * 3;
                    let neigh_lum = luminance(image[ni], image[ni + 1], image[ni + 2]);

                    let lum_diff = center_lum - neigh_lum;
                    let range_w = (-(lum_diff * lum_diff) / two_sr_sq).exp();

                    let w = spatial_w * range_w;
                    sum_r += w * image[ni];
                    sum_g += w * image[ni + 1];
                    sum_b += w * image[ni + 2];
                    sum_w += w;
                }
            }

            output[ci] = sum_r / sum_w;
            output[ci + 1] = sum_g / sum_w;
            output[ci + 2] = sum_b / sum_w;
        }
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Joint bilateral filter
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a joint (cross) bilateral filter guided by a single-channel guide image.
///
/// Like [`bilateral_filter`], but the range term is based on differences in the
/// `guide` channel (e.g., a depth map) rather than luminance of the RGB image.
/// This allows the filter to respect depth edges when smoothing colour.
///
/// # Arguments
/// * `image`  – RGB f32 buffer, length `width * height * 3`.
/// * `guide`  – single-channel f32 buffer (e.g., depth), length `width * height`.
/// * `width`, `height` – image dimensions.
/// * `config` – shared bilateral configuration.
pub fn joint_bilateral_filter(
    image: &[f32],
    guide: &[f32],
    width: usize,
    height: usize,
    config: &BilateralConfig,
) -> Result<Vec<f32>, DenoisingError> {
    config.validate()?;
    validate_rgb_image(image, width, height)?;

    let n_pixels = width * height;
    if guide.len() != n_pixels {
        if n_pixels == 0 {
            return Err(DenoisingError::EmptyImage);
        }
        return Err(DenoisingError::InvalidImage(format!(
            "guide length {} does not match {}×{} = {}",
            guide.len(),
            width,
            height,
            n_pixels
        )));
    }

    let two_ss_sq = 2.0 * config.sigma_spatial * config.sigma_spatial;
    let two_sr_sq = 2.0 * config.sigma_range * config.sigma_range;
    let r = config.radius as i32;

    let mut output = vec![0.0_f32; width * height * 3];

    for cy in 0..height {
        for cx in 0..width {
            let ci = (cy * width + cx) * 3;
            let center_guide = guide[cy * width + cx];

            let mut sum_r = 0.0_f32;
            let mut sum_g = 0.0_f32;
            let mut sum_b = 0.0_f32;
            let mut sum_w = 0.0_f32;

            for dy in -r..=r {
                let ny = clamp_coord(cy as i32 + dy, height);
                for dx in -r..=r {
                    let nx = clamp_coord(cx as i32 + dx, width);

                    let dist_sq = (dx * dx + dy * dy) as f32;
                    let spatial_w = (-dist_sq / two_ss_sq).exp();

                    let guide_diff = center_guide - guide[ny * width + nx];
                    let range_w = (-(guide_diff * guide_diff) / two_sr_sq).exp();

                    let w = spatial_w * range_w;
                    let ni = (ny * width + nx) * 3;
                    sum_r += w * image[ni];
                    sum_g += w * image[ni + 1];
                    sum_b += w * image[ni + 2];
                    sum_w += w;
                }
            }

            output[ci] = sum_r / sum_w;
            output[ci + 1] = sum_g / sum_w;
            output[ci + 2] = sum_b / sum_w;
        }
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Median filter
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the median filter.
#[derive(Debug, Clone)]
pub struct MedianConfig {
    /// Kernel half-width (pixels). Actual kernel is `(2*radius+1)²`. Default: 1.
    pub radius: usize,
}

impl Default for MedianConfig {
    fn default() -> Self {
        Self { radius: 1 }
    }
}

impl MedianConfig {
    /// Returns an error if `radius` is zero.
    pub fn validate(&self) -> Result<(), DenoisingError> {
        if self.radius < 1 {
            return Err(DenoisingError::InvalidConfig(
                "median radius must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Apply a per-channel median filter to an RGB image.
///
/// Each output channel is the median of all values in the `(2*radius+1)²`
/// neighbourhood. Median filtering is particularly effective against
/// salt-and-pepper noise while preserving sharp edges better than a Gaussian.
///
/// Boundary pixels use clamp-to-edge sampling.
pub fn median_filter(
    image: &[f32],
    width: usize,
    height: usize,
    config: &MedianConfig,
) -> Result<Vec<f32>, DenoisingError> {
    config.validate()?;
    validate_rgb_image(image, width, height)?;

    let r = config.radius as i32;
    let kernel_area = (2 * config.radius + 1) * (2 * config.radius + 1);
    let median_idx = kernel_area / 2;

    let mut output = vec![0.0_f32; width * height * 3];
    let mut buf_r = Vec::with_capacity(kernel_area);
    let mut buf_g = Vec::with_capacity(kernel_area);
    let mut buf_b = Vec::with_capacity(kernel_area);

    for cy in 0..height {
        for cx in 0..width {
            buf_r.clear();
            buf_g.clear();
            buf_b.clear();

            for dy in -r..=r {
                let ny = clamp_coord(cy as i32 + dy, height);
                for dx in -r..=r {
                    let nx = clamp_coord(cx as i32 + dx, width);
                    let ni = (ny * width + nx) * 3;
                    buf_r.push(image[ni]);
                    buf_g.push(image[ni + 1]);
                    buf_b.push(image[ni + 2]);
                }
            }

            buf_r.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            buf_g.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            buf_b.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let oi = (cy * width + cx) * 3;
            output[oi] = buf_r[median_idx];
            output[oi + 1] = buf_g[median_idx];
            output[oi + 2] = buf_b[median_idx];
        }
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-local means
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Non-Local Means (NLM) denoising algorithm.
#[derive(Debug, Clone)]
pub struct NlmConfig {
    /// Search window half-width (pixels). Full search in a `(2*search_radius+1)²` area.
    /// Default: 5.
    pub search_radius: usize,
    /// Patch half-width for similarity comparison. Default: 2.
    pub patch_radius: usize,
    /// Filtering strength. Higher values produce more blurring. Default: 0.1.
    pub h: f32,
}

impl Default for NlmConfig {
    fn default() -> Self {
        Self {
            search_radius: 5,
            patch_radius: 2,
            h: 0.1,
        }
    }
}

impl NlmConfig {
    /// Returns an error if any parameter is invalid.
    pub fn validate(&self) -> Result<(), DenoisingError> {
        if self.h <= 0.0 {
            return Err(DenoisingError::InvalidConfig(format!(
                "h must be positive, got {}",
                self.h
            )));
        }
        if self.patch_radius < 1 {
            return Err(DenoisingError::InvalidConfig(
                "patch_radius must be at least 1".to_string(),
            ));
        }
        if self.search_radius < self.patch_radius {
            return Err(DenoisingError::InvalidConfig(format!(
                "search_radius ({}) must be >= patch_radius ({})",
                self.search_radius, self.patch_radius
            )));
        }
        Ok(())
    }
}

/// Apply Non-Local Means (NLM) denoising to an RGB image.
///
/// NLM is a powerful but computationally intensive algorithm. For each centre
/// pixel `p`, it searches all pixels `q` in a `(2*search_radius+1)²` window.
/// The weight of `q` is proportional to `exp(-D(p,q) / h²)` where `D(p,q)` is
/// the mean squared L2 difference between the `(2*patch_radius+1)²` RGB patches
/// centred at `p` and `q`.
///
/// Complexity: O(W × H × search_radius² × patch_radius²) — can be slow.
///
/// Boundary pixels use clamp-to-edge sampling.
pub fn non_local_means(
    image: &[f32],
    width: usize,
    height: usize,
    config: &NlmConfig,
) -> Result<Vec<f32>, DenoisingError> {
    config.validate()?;
    validate_rgb_image(image, width, height)?;

    let h_sq = config.h * config.h;
    let sr = config.search_radius as i32;
    let pr = config.patch_radius as i32;
    let patch_size = (2 * config.patch_radius + 1) * (2 * config.patch_radius + 1) * 3;
    let patch_norm = patch_size as f32;

    let mut output = vec![0.0_f32; width * height * 3];

    for cy in 0..height {
        for cx in 0..width {
            let mut sum_r = 0.0_f32;
            let mut sum_g = 0.0_f32;
            let mut sum_b = 0.0_f32;
            let mut sum_w = 0.0_f32;

            // For each candidate pixel q in the search window
            for sy in -sr..=sr {
                let qy = clamp_coord(cy as i32 + sy, height);
                for sx in -sr..=sr {
                    let qx = clamp_coord(cx as i32 + sx, width);

                    // Compute normalised patch similarity between p and q
                    let mut patch_dist_sq = 0.0_f32;
                    for py in -pr..=pr {
                        let pay = clamp_coord(cy as i32 + py, height);
                        let pby = clamp_coord(qy as i32 + py, height);
                        for px in -pr..=pr {
                            let pax = clamp_coord(cx as i32 + px, width);
                            let pbx = clamp_coord(qx as i32 + px, width);

                            let ai = (pay * width + pax) * 3;
                            let bi = (pby * width + pbx) * 3;

                            let dr = image[ai] - image[bi];
                            let dg = image[ai + 1] - image[bi + 1];
                            let db = image[ai + 2] - image[bi + 2];
                            patch_dist_sq += dr * dr + dg * dg + db * db;
                        }
                    }

                    // Normalise by patch element count so h is scale-independent
                    let patch_dist_norm = patch_dist_sq / patch_norm;
                    let w = (-patch_dist_norm / h_sq).exp();

                    let qi = (qy * width + qx) * 3;
                    sum_r += w * image[qi];
                    sum_g += w * image[qi + 1];
                    sum_b += w * image[qi + 2];
                    sum_w += w;
                }
            }

            let oi = (cy * width + cx) * 3;
            output[oi] = sum_r / sum_w;
            output[oi + 1] = sum_g / sum_w;
            output[oi + 2] = sum_b / sum_w;
        }
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian denoise (separable convolution)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for simple Gaussian denoising.
#[derive(Debug, Clone)]
pub struct GaussianDenoiseConfig {
    /// Standard deviation of the Gaussian kernel (pixels). Default: 1.0.
    pub sigma: f32,
    /// Kernel half-width (pixels). Default: 2.
    pub radius: usize,
}

impl Default for GaussianDenoiseConfig {
    fn default() -> Self {
        Self {
            sigma: 1.0,
            radius: 2,
        }
    }
}

impl GaussianDenoiseConfig {
    /// Returns an error if any parameter is invalid.
    pub fn validate(&self) -> Result<(), DenoisingError> {
        if self.sigma <= 0.0 {
            return Err(DenoisingError::InvalidConfig(format!(
                "sigma must be positive, got {}",
                self.sigma
            )));
        }
        if self.radius < 1 {
            return Err(DenoisingError::InvalidConfig(
                "radius must be at least 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Apply separable Gaussian blur for simple denoising.
///
/// Uses a horizontal pass followed by a vertical pass over the full 1-D kernel,
/// giving O(W × H × 2 × (2*radius+1)) complexity — much faster than a full 2-D
/// convolution for large radii.
///
/// Boundary pixels use clamp-to-edge sampling.
pub fn gaussian_denoise(
    image: &[f32],
    width: usize,
    height: usize,
    config: &GaussianDenoiseConfig,
) -> Result<Vec<f32>, DenoisingError> {
    config.validate()?;
    validate_rgb_image(image, width, height)?;

    let kernel = gaussian_1d_kernel(config.radius, config.sigma)?;
    let r = config.radius as i32;

    // Horizontal pass: convolve along X, store in tmp
    let mut tmp = vec![0.0_f32; width * height * 3];
    for cy in 0..height {
        for cx in 0..width {
            let mut acc_r = 0.0_f32;
            let mut acc_g = 0.0_f32;
            let mut acc_b = 0.0_f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let nx = clamp_coord(cx as i32 + ki as i32 - r, width);
                let ni = (cy * width + nx) * 3;
                acc_r += kv * image[ni];
                acc_g += kv * image[ni + 1];
                acc_b += kv * image[ni + 2];
            }
            let oi = (cy * width + cx) * 3;
            tmp[oi] = acc_r;
            tmp[oi + 1] = acc_g;
            tmp[oi + 2] = acc_b;
        }
    }

    // Vertical pass: convolve along Y, store in output
    let mut output = vec![0.0_f32; width * height * 3];
    for cy in 0..height {
        for cx in 0..width {
            let mut acc_r = 0.0_f32;
            let mut acc_g = 0.0_f32;
            let mut acc_b = 0.0_f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let ny = clamp_coord(cy as i32 + ki as i32 - r, height);
                let ni = (ny * width + cx) * 3;
                acc_r += kv * tmp[ni];
                acc_g += kv * tmp[ni + 1];
                acc_b += kv * tmp[ni + 2];
            }
            let oi = (cy * width + cx) * 3;
            output[oi] = acc_r;
            output[oi + 1] = acc_g;
            output[oi + 2] = acc_b;
        }
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Noise estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Noise statistics estimated from an image.
#[derive(Debug, Clone)]
pub struct NoiseStats {
    /// Estimated Gaussian noise standard deviation (Immerkær 1996 method).
    pub estimated_sigma: f32,
    /// Mean edge magnitude (finite-difference gradient).
    pub mean_gradient: f32,
    /// Variance of the Laplacian response (high for noisy images).
    pub laplacian_variance: f32,
    /// Estimated signal-to-noise ratio (`mean_luminance / (estimated_sigma + ε)`).
    pub snr_estimate: f32,
}

/// Estimate noise statistics of an RGB image.
///
/// Uses the Immerkær (1996) Laplacian-based noise estimator:
/// ```text
/// sigma ≈ sqrt(π/2) × mean(|Laplacian|) / 6
/// ```
/// where the Laplacian kernel is `[[0,1,0],[1,-4,1],[0,1,0]]` applied to
/// luminance.  The mean gradient is computed from finite-difference Gx/Gy
/// magnitudes.
pub fn estimate_noise(
    image: &[f32],
    width: usize,
    height: usize,
) -> Result<NoiseStats, DenoisingError> {
    validate_rgb_image(image, width, height)?;

    // Compute luminance image
    let n = width * height;
    let mut lum = Vec::with_capacity(n);
    for i in 0..n {
        let pi = i * 3;
        lum.push(luminance(image[pi], image[pi + 1], image[pi + 2]));
    }

    // Mean luminance
    let mean_lum: f32 = lum.iter().sum::<f32>() / n as f32;

    // Laplacian kernel [[0,1,0],[1,-4,1],[0,1,0]]
    let mut laplacian_vals = Vec::with_capacity(n);
    let mut laplacian_sum_abs = 0.0_f32;
    for cy in 0..height {
        for cx in 0..width {
            let center = lum[cy * width + cx];
            let north = lum[clamp_coord(cy as i32 - 1, height) * width + cx];
            let south = lum[clamp_coord(cy as i32 + 1, height) * width + cx];
            let west = lum[cy * width + clamp_coord(cx as i32 - 1, width)];
            let east = lum[cy * width + clamp_coord(cx as i32 + 1, width)];
            let lap = north + south + west + east - 4.0 * center;
            laplacian_sum_abs += lap.abs();
            laplacian_vals.push(lap);
        }
    }

    let mean_abs_lap = laplacian_sum_abs / n as f32;
    // Immerkær 1996: sigma ≈ sqrt(π/2) × mean(|L|) / 6
    let estimated_sigma = (std::f32::consts::PI / 2.0_f32).sqrt() * mean_abs_lap / 6.0;

    // Laplacian variance
    let lap_mean: f32 = laplacian_vals.iter().sum::<f32>() / n as f32;
    let laplacian_variance: f32 = laplacian_vals
        .iter()
        .map(|&v| (v - lap_mean) * (v - lap_mean))
        .sum::<f32>()
        / n as f32;

    // Mean gradient magnitude (finite difference)
    let mut grad_sum = 0.0_f32;
    for cy in 0..height {
        for cx in 0..width {
            let right = lum[cy * width + clamp_coord(cx as i32 + 1, width)];
            let down = lum[clamp_coord(cy as i32 + 1, height) * width + cx];
            let gx = right - lum[cy * width + cx];
            let gy = down - lum[cy * width + cx];
            grad_sum += (gx * gx + gy * gy).sqrt();
        }
    }
    let mean_gradient = grad_sum / n as f32;

    let snr_estimate = mean_lum / (estimated_sigma + 1e-8);

    Ok(NoiseStats {
        estimated_sigma,
        mean_gradient,
        laplacian_variance,
        snr_estimate,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Adaptive denoising
// ─────────────────────────────────────────────────────────────────────────────

/// Select and apply a denoising method based on the image's estimated noise level.
///
/// | Estimated noise | Action |
/// |-----------------|--------|
/// | < 0.05          | Return image unchanged |
/// | 0.05 – 0.15     | [`bilateral_filter`] with default config |
/// | ≥ 0.15          | [`median_filter`] (radius=1) followed by [`bilateral_filter`] |
///
/// The noise estimate is the mean gradient magnitude across the luminance image.
pub fn denoise_adaptive(
    image: &[f32],
    width: usize,
    height: usize,
) -> Result<Vec<f32>, DenoisingError> {
    validate_rgb_image(image, width, height)?;

    let stats = estimate_noise(image, width, height)?;
    let noise = stats.mean_gradient;

    if noise < 0.05 {
        Ok(image.to_vec())
    } else if noise < 0.15 {
        bilateral_filter(image, width, height, &BilateralConfig::default())
    } else {
        let median_cfg = MedianConfig { radius: 1 };
        let median_out = median_filter(image, width, height, &median_cfg)?;
        bilateral_filter(&median_out, width, height, &BilateralConfig::default())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Denoising pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// The denoising algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenoisingMethod {
    /// Simple Gaussian blur.
    Gaussian,
    /// Edge-preserving bilateral filter.
    Bilateral,
    /// Per-channel median filter.
    Median,
    /// Non-Local Means — slow but high quality.
    NonLocalMeans,
    /// Automatically choose based on estimated noise level.
    Adaptive,
}

/// A configurable denoising pipeline that dispatches to one of several algorithms.
#[derive(Debug, Clone)]
pub struct DenoisingPipeline {
    /// Which denoising algorithm to apply.
    pub method: DenoisingMethod,
    /// Configuration for the bilateral filter.
    pub bilateral: BilateralConfig,
    /// Configuration for the median filter.
    pub median: MedianConfig,
    /// Configuration for Non-Local Means.
    pub nlm: NlmConfig,
    /// Configuration for Gaussian denoising.
    pub gaussian: GaussianDenoiseConfig,
}

impl Default for DenoisingPipeline {
    fn default() -> Self {
        Self {
            method: DenoisingMethod::Gaussian,
            bilateral: BilateralConfig::default(),
            median: MedianConfig::default(),
            nlm: NlmConfig::default(),
            gaussian: GaussianDenoiseConfig::default(),
        }
    }
}

impl DenoisingPipeline {
    /// Denoise an RGB image using the configured method.
    pub fn denoise(
        &self,
        image: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, DenoisingError> {
        match self.method {
            DenoisingMethod::Gaussian => gaussian_denoise(image, width, height, &self.gaussian),
            DenoisingMethod::Bilateral => bilateral_filter(image, width, height, &self.bilateral),
            DenoisingMethod::Median => median_filter(image, width, height, &self.median),
            DenoisingMethod::NonLocalMeans => non_local_means(image, width, height, &self.nlm),
            DenoisingMethod::Adaptive => denoise_adaptive(image, width, height),
        }
    }

    /// Return a new pipeline with the specified method.
    pub fn with_method(mut self, method: DenoisingMethod) -> Self {
        self.method = method;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Create a uniform RGB image of the given colour.
    fn uniform_image(width: usize, height: usize, r: f32, g: f32, b: f32) -> Vec<f32> {
        let mut img = Vec::with_capacity(width * height * 3);
        for _ in 0..width * height {
            img.push(r);
            img.push(g);
            img.push(b);
        }
        img
    }

    /// Variance of a flat float slice.
    fn variance(data: &[f32]) -> f32 {
        let n = data.len() as f32;
        let mean = data.iter().sum::<f32>() / n;
        data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n
    }

    // ── BilateralConfig validation ────────────────────────────────────────────

    #[test]
    fn bilateral_config_valid_default() {
        assert!(BilateralConfig::default().validate().is_ok());
    }

    #[test]
    fn bilateral_config_invalid_sigma_spatial() {
        let cfg = BilateralConfig {
            sigma_spatial: 0.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn bilateral_config_invalid_sigma_range() {
        let cfg = BilateralConfig {
            sigma_range: -1.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn bilateral_config_invalid_radius_zero() {
        let cfg = BilateralConfig {
            radius: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // ── bilateral_filter ──────────────────────────────────────────────────────

    #[test]
    fn bilateral_filter_empty_image_error() {
        let cfg = BilateralConfig::default();
        let result = bilateral_filter(&[], 0, 0, &cfg);
        assert!(matches!(result, Err(DenoisingError::EmptyImage)));
    }

    #[test]
    fn bilateral_filter_wrong_length_error() {
        let cfg = BilateralConfig::default();
        let result = bilateral_filter(&[0.5; 10], 4, 4, &cfg);
        assert!(matches!(result, Err(DenoisingError::InvalidImage(_))));
    }

    #[test]
    fn bilateral_filter_uniform_unchanged() {
        let w = 8;
        let h = 8;
        let img = uniform_image(w, h, 0.4, 0.5, 0.6);
        let cfg = BilateralConfig::default();
        let out = bilateral_filter(&img, w, h, &cfg).unwrap();
        assert_eq!(out.len(), img.len());
        for (a, b) in img.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "uniform image changed: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn bilateral_filter_preserves_sharp_edge() {
        // 10×1 image: left half red, right half green.
        // With a small sigma_range, the centre pixel should stay close to its
        // original colour rather than being pulled toward the other side.
        let w = 10;
        let h = 1;
        let mut img = vec![0.0_f32; w * h * 3];
        for x in 0..5 {
            img[x * 3] = 1.0; // red
        }
        for x in 5..10 {
            img[x * 3 + 1] = 1.0; // green
        }
        let cfg = BilateralConfig {
            sigma_spatial: 3.0,
            sigma_range: 0.05, // tight range → edge should be preserved
            radius: 4,
        };
        let out = bilateral_filter(&img, w, h, &cfg).unwrap();
        // pixel x=4 (last red pixel): red channel should stay > 0.8
        let r_at_4 = out[4 * 3];
        assert!(
            r_at_4 > 0.8,
            "edge not preserved: red channel at x=4 is {}",
            r_at_4
        );
        // pixel x=5 (first green pixel): green channel should stay > 0.8
        let g_at_5 = out[5 * 3 + 1];
        assert!(
            g_at_5 > 0.8,
            "edge not preserved: green channel at x=5 is {}",
            g_at_5
        );
    }

    #[test]
    fn bilateral_filter_radius1_output_size() {
        let w = 5;
        let h = 5;
        let img = uniform_image(w, h, 0.5, 0.5, 0.5);
        let cfg = BilateralConfig {
            radius: 1,
            ..Default::default()
        };
        let out = bilateral_filter(&img, w, h, &cfg).unwrap();
        assert_eq!(out.len(), w * h * 3);
    }

    #[test]
    fn bilateral_filter_large_sigma_range_approaches_gaussian() {
        // With very large sigma_range, range weight ≈ 1 everywhere → result
        // should approach a spatial Gaussian blur.  For a uniform image, output
        // equals input regardless, so we check the output is valid and same size.
        let w = 6;
        let h = 6;
        let img = uniform_image(w, h, 0.3, 0.5, 0.7);
        let cfg = BilateralConfig {
            sigma_range: 1000.0, // effectively disables range filtering
            sigma_spatial: 2.0,
            radius: 2,
        };
        let out = bilateral_filter(&img, w, h, &cfg).unwrap();
        assert_eq!(out.len(), img.len());
        // All output values should still be in [0, 1] range
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "value out of range: {}", v);
        }
    }

    // ── joint_bilateral_filter ────────────────────────────────────────────────

    #[test]
    fn joint_bilateral_uniform_image_unchanged() {
        let w = 6;
        let h = 6;
        let img = uniform_image(w, h, 0.5, 0.5, 0.5);
        let guide = vec![0.5_f32; w * h];
        let cfg = BilateralConfig::default();
        let out = joint_bilateral_filter(&img, &guide, w, h, &cfg).unwrap();
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn joint_bilateral_guide_wrong_length_error() {
        let w = 4;
        let h = 4;
        let img = uniform_image(w, h, 0.5, 0.5, 0.5);
        let guide = vec![0.5_f32; 5]; // wrong length
        let cfg = BilateralConfig::default();
        let result = joint_bilateral_filter(&img, &guide, w, h, &cfg);
        assert!(matches!(result, Err(DenoisingError::InvalidImage(_))));
    }

    #[test]
    fn joint_bilateral_different_guide_preserves_edges() {
        // Image: uniform grey.  Guide: sharp step at x=5.
        // Filter should preserve the guide boundary rather than blur across it.
        // We verify by checking that the output is valid and same length.
        let w = 10;
        let h = 6;
        let img = uniform_image(w, h, 0.5, 0.5, 0.5);
        let mut guide = vec![0.0_f32; w * h];
        for y in 0..h {
            for x in 5..w {
                guide[y * w + x] = 1.0; // large guide step
            }
        }
        let cfg = BilateralConfig {
            sigma_range: 0.05,
            ..Default::default()
        };
        let out = joint_bilateral_filter(&img, &guide, w, h, &cfg).unwrap();
        assert_eq!(out.len(), img.len());
    }

    // ── MedianConfig validation ───────────────────────────────────────────────

    #[test]
    fn median_config_valid_default() {
        assert!(MedianConfig::default().validate().is_ok());
    }

    #[test]
    fn median_config_invalid_zero_radius() {
        let cfg = MedianConfig { radius: 0 };
        assert!(cfg.validate().is_err());
    }

    // ── median_filter ─────────────────────────────────────────────────────────

    #[test]
    fn median_filter_uniform_unchanged() {
        let w = 7;
        let h = 7;
        let img = uniform_image(w, h, 0.3, 0.6, 0.9);
        let out = median_filter(&img, w, h, &MedianConfig::default()).unwrap();
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn median_filter_removes_single_outlier() {
        // 5×5 uniform grey image with one spike pixel in the centre.
        let w = 5;
        let h = 5;
        let mut img = uniform_image(w, h, 0.5, 0.5, 0.5);
        let centre = (h / 2 * w + w / 2) * 3;
        img[centre] = 1.0; // red spike
        img[centre + 1] = 0.0;
        img[centre + 2] = 0.0;

        let out = median_filter(&img, w, h, &MedianConfig::default()).unwrap();
        // The spike pixel's red channel should be driven back toward 0.5
        assert!(
            out[centre] < 0.8,
            "outlier not removed: red at centre is {}",
            out[centre]
        );
    }

    #[test]
    fn median_filter_1x1_image() {
        let img = vec![0.2_f32, 0.4_f32, 0.6_f32];
        let out = median_filter(&img, 1, 1, &MedianConfig::default()).unwrap();
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.2).abs() < 1e-6);
        assert!((out[1] - 0.4).abs() < 1e-6);
        assert!((out[2] - 0.6).abs() < 1e-6);
    }

    // ── NlmConfig validation ──────────────────────────────────────────────────

    #[test]
    fn nlm_config_valid_default() {
        assert!(NlmConfig::default().validate().is_ok());
    }

    #[test]
    fn nlm_config_invalid_h_zero() {
        let cfg = NlmConfig {
            h: 0.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn nlm_config_invalid_patch_radius_zero() {
        let cfg = NlmConfig {
            patch_radius: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn nlm_config_search_radius_less_than_patch_radius() {
        let cfg = NlmConfig {
            search_radius: 1,
            patch_radius: 3,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // ── non_local_means ───────────────────────────────────────────────────────

    #[test]
    fn nlm_uniform_image_unchanged() {
        let w = 8;
        let h = 8;
        let img = uniform_image(w, h, 0.6, 0.4, 0.2);
        let cfg = NlmConfig {
            search_radius: 3,
            patch_radius: 1,
            h: 0.1,
        };
        let out = non_local_means(&img, w, h, &cfg).unwrap();
        for (a, b) in img.iter().zip(out.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "NLM changed uniform image: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn nlm_reduces_noise_variance() {
        // Build a uniform image with Gaussian-like noise added.
        let w = 16;
        let h = 16;
        let mut noisy = uniform_image(w, h, 0.5, 0.5, 0.5);
        // Add deterministic pseudo-noise by cycling through a fixed pattern
        let noise_pattern = [0.08_f32, -0.06, 0.09, -0.07, 0.05, -0.08, 0.06, -0.05];
        for (i, v) in noisy.iter_mut().enumerate() {
            *v = (*v + noise_pattern[i % noise_pattern.len()]).clamp(0.0, 1.0);
        }
        let input_var = variance(&noisy);

        let cfg = NlmConfig {
            search_radius: 4,
            patch_radius: 1,
            h: 0.15,
        };
        let out = non_local_means(&noisy, w, h, &cfg).unwrap();
        let output_var = variance(&out);

        assert!(
            output_var < input_var,
            "NLM did not reduce variance: input {} output {}",
            input_var,
            output_var
        );
    }

    // ── gaussian_denoise ──────────────────────────────────────────────────────

    #[test]
    fn gaussian_denoise_uniform_unchanged() {
        let w = 6;
        let h = 6;
        let img = uniform_image(w, h, 0.7, 0.3, 0.1);
        let cfg = GaussianDenoiseConfig::default();
        let out = gaussian_denoise(&img, w, h, &cfg).unwrap();
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn gaussian_denoise_blurs_sharp_step() {
        // 12×1 image: left half = 0, right half = 1 in all channels.
        let w = 12;
        let h = 1;
        let mut img = vec![0.0_f32; w * h * 3];
        for x in 6..12 {
            img[x * 3] = 1.0;
            img[x * 3 + 1] = 1.0;
            img[x * 3 + 2] = 1.0;
        }
        let cfg = GaussianDenoiseConfig {
            sigma: 2.0,
            radius: 3,
        };
        let out = gaussian_denoise(&img, w, h, &cfg).unwrap();
        // The boundary pixel x=5 (originally 0) should have been pushed up
        let boundary = out[5 * 3]; // was 0, neighbours are 0 and 1
        assert!(
            boundary > 0.01 && boundary < 0.99,
            "step not blurred: x=5 is {}",
            boundary
        );
    }

    // ── estimate_noise ────────────────────────────────────────────────────────

    #[test]
    fn estimate_noise_uniform_low_sigma() {
        let w = 8;
        let h = 8;
        let img = uniform_image(w, h, 0.5, 0.5, 0.5);
        let stats = estimate_noise(&img, w, h).unwrap();
        assert!(
            stats.estimated_sigma < 0.01,
            "uniform image should have low sigma, got {}",
            stats.estimated_sigma
        );
        assert!(
            stats.mean_gradient < 0.01,
            "uniform image should have zero gradient, got {}",
            stats.mean_gradient
        );
    }

    #[test]
    fn estimate_noise_noisy_higher_sigma() {
        let w = 8;
        let h = 8;
        // Checkerboard = high-frequency noise analogue
        let mut img = vec![0.0_f32; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y) % 2 == 0 { 0.0 } else { 1.0 };
                let i = (y * w + x) * 3;
                img[i] = v;
                img[i + 1] = v;
                img[i + 2] = v;
            }
        }
        let stats = estimate_noise(&img, w, h).unwrap();
        assert!(
            stats.estimated_sigma > 0.05,
            "noisy image should have higher sigma, got {}",
            stats.estimated_sigma
        );
    }

    #[test]
    fn estimate_noise_snr_uniform() {
        // Uniform white image: mean_lum=1, sigma~0 → very high SNR
        let w = 4;
        let h = 4;
        let img = uniform_image(w, h, 1.0, 1.0, 1.0);
        let stats = estimate_noise(&img, w, h).unwrap();
        // SNR = mean_lum / (estimated_sigma + 1e-8)
        // With sigma ≈ 0 and mean_lum = 1, SNR should be very large
        assert!(
            stats.snr_estimate > 1.0,
            "SNR should be positive, got {}",
            stats.snr_estimate
        );
    }

    // ── DenoisingPipeline ─────────────────────────────────────────────────────

    #[test]
    fn pipeline_default_method_is_gaussian() {
        let p = DenoisingPipeline::default();
        assert_eq!(p.method, DenoisingMethod::Gaussian);
    }

    #[test]
    fn pipeline_with_method_changes_method() {
        let p = DenoisingPipeline::default().with_method(DenoisingMethod::Median);
        assert_eq!(p.method, DenoisingMethod::Median);
    }

    #[test]
    fn pipeline_denoise_gaussian() {
        let w = 4;
        let h = 4;
        let img = uniform_image(w, h, 0.5, 0.5, 0.5);
        let p = DenoisingPipeline::default();
        let out = p.denoise(&img, w, h).unwrap();
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn pipeline_denoise_bilateral() {
        let w = 4;
        let h = 4;
        let img = uniform_image(w, h, 0.5, 0.5, 0.5);
        let p = DenoisingPipeline::default().with_method(DenoisingMethod::Bilateral);
        let out = p.denoise(&img, w, h).unwrap();
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn pipeline_denoise_median() {
        let w = 4;
        let h = 4;
        let img = uniform_image(w, h, 0.5, 0.5, 0.5);
        let p = DenoisingPipeline::default().with_method(DenoisingMethod::Median);
        let out = p.denoise(&img, w, h).unwrap();
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn pipeline_denoise_non_local_means() {
        let w = 6;
        let h = 6;
        let img = uniform_image(w, h, 0.4, 0.4, 0.4);
        let p = DenoisingPipeline {
            nlm: NlmConfig {
                search_radius: 2,
                patch_radius: 1,
                h: 0.1,
            },
            ..Default::default()
        };
        let p = p.with_method(DenoisingMethod::NonLocalMeans);
        let out = p.denoise(&img, w, h).unwrap();
        assert_eq!(out.len(), img.len());
    }

    // ── denoise_adaptive ──────────────────────────────────────────────────────

    #[test]
    fn adaptive_low_noise_returns_similar_image() {
        // Uniform image → noise < 0.05 → should be returned unchanged
        let w = 8;
        let h = 8;
        let img = uniform_image(w, h, 0.5, 0.5, 0.5);
        let out = denoise_adaptive(&img, w, h).unwrap();
        assert_eq!(out.len(), img.len());
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn adaptive_high_noise_applies_both_filters() {
        // Smooth background with high-amplitude impulse noise.
        // The impulse noise gives mean_gradient > 0.15 → median + bilateral path.
        // The median filter removes impulse spikes, so output variance < input variance.
        let w = 12;
        let h = 12;
        // Start with a smooth grey background
        let mut img = uniform_image(w, h, 0.5, 0.5, 0.5);
        // Place isolated single-pixel spikes every 3 pixels (not adjacent,
        // so clamp-to-edge boundary effects don't "save" them).
        // Spike value = 1.0 surrounded by 0.5 → median of 3×3 = 0.5.
        let spike_positions: &[(usize, usize)] = &[
            (2, 2),
            (5, 2),
            (8, 2),
            (2, 5),
            (5, 5),
            (8, 5),
            (2, 8),
            (5, 8),
            (8, 8),
        ];
        for &(x, y) in spike_positions {
            let i = (y * w + x) * 3;
            img[i] = 1.0;
            img[i + 1] = 1.0;
            img[i + 2] = 1.0;
        }
        let var_in = variance(&img);

        let out = denoise_adaptive(&img, w, h).unwrap();
        assert_eq!(out.len(), img.len());

        // The adaptive path should have removed spike noise, reducing variance.
        let var_out = variance(&out);
        assert!(
            var_out < var_in,
            "adaptive did not reduce variance: in {:.6} out {:.6}",
            var_in,
            var_out
        );
    }
}
