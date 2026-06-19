//! Multi-resolution image loss computation for 3DGS training.
//!
//! Computes losses at multiple image scales (Gaussian pyramids) for better
//! training signal and stability. Supports L1, L2, SSIM, Laplacian pyramid
//! loss, and gradient L1 loss types.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error Type
// ---------------------------------------------------------------------------

/// Errors produced by multi-resolution loss computation.
#[derive(Debug, Error)]
pub enum MultiResLossError {
    #[error("Image size {got} != {width}×{height}×{channels}")]
    SizeMismatch {
        got: usize,
        width: usize,
        height: usize,
        channels: usize,
    },
    #[error("Width {0} and height must both be >= 2 for downsampling")]
    ImageTooSmall(usize),
    #[error("No pyramid levels configured")]
    NoLevels,
    #[error("Predicted and ground truth have different sizes")]
    ShapeMismatch,
    #[error("Invalid weight {0}: must be positive")]
    InvalidWeight(f32),
}

// ---------------------------------------------------------------------------
// Image Pyramid
// ---------------------------------------------------------------------------

/// A single level of a Gaussian image pyramid.
pub struct PyramidLevel {
    /// Flat f32 image data in H×W×channels order.
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
    /// Scale relative to original (1.0, 0.5, 0.25, ...).
    pub scale: f32,
}

/// An image represented as a Gaussian pyramid.
pub struct ImagePyramid {
    pub levels: Vec<PyramidLevel>,
    pub channels: usize,
}

impl ImagePyramid {
    /// Build a Gaussian pyramid from an image.
    ///
    /// Level 0 is the original. Each subsequent level is a 2× downsampled
    /// version. Building stops early if either dimension drops below 2.
    pub fn build(
        img: &[f32],
        width: usize,
        height: usize,
        channels: usize,
        n_levels: usize,
    ) -> Result<Self, MultiResLossError> {
        let expected = width * height * channels;
        if img.len() != expected {
            return Err(MultiResLossError::SizeMismatch {
                got: img.len(),
                width,
                height,
                channels,
            });
        }
        if n_levels == 0 {
            return Err(MultiResLossError::NoLevels);
        }

        let mut levels = Vec::with_capacity(n_levels);

        // Level 0: original image
        levels.push(PyramidLevel {
            data: img.to_vec(),
            width,
            height,
            scale: 1.0,
        });

        let mut cur_w = width;
        let mut cur_h = height;
        let mut cur_data = img.to_vec();

        for k in 1..n_levels {
            if cur_w < 2 || cur_h < 2 {
                break;
            }
            let (down_data, new_w, new_h) = mr_downsample(&cur_data, cur_w, cur_h, channels)?;
            let scale = 0.5_f32.powi(k as i32);
            levels.push(PyramidLevel {
                data: down_data.clone(),
                width: new_w,
                height: new_h,
                scale,
            });
            cur_data = down_data;
            cur_w = new_w;
            cur_h = new_h;
        }

        Ok(Self { levels, channels })
    }

    /// Number of pyramid levels.
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Access a specific pyramid level by index, or `None` if out of range.
    pub fn level(&self, idx: usize) -> Option<&PyramidLevel> {
        self.levels.get(idx)
    }

    /// Access the original (full-resolution) level.
    pub fn original(&self) -> &PyramidLevel {
        // SAFETY: build() always inserts at least level 0
        &self.levels[0]
    }
}

// ---------------------------------------------------------------------------
// Loss Configuration
// ---------------------------------------------------------------------------

/// The type of per-level loss to compute.
#[derive(Debug, Clone, PartialEq)]
pub enum MultiResLossType {
    L1,
    L2,
    Ssim,
    /// Laplacian pyramid loss (high-frequency detail).
    Laplacian,
    /// L1 on image gradients (edge-preserving).
    GradientL1,
}

/// Configuration for multi-resolution loss computation.
pub struct MultiResLossConfig {
    /// Number of pyramid levels.
    pub n_levels: usize,
    /// Weight for each pyramid level.
    pub level_weights: Vec<f32>,
    /// Loss types to include.
    pub loss_types: Vec<MultiResLossType>,
    /// Whether to normalize weights to sum to 1.
    pub normalize_weights: bool,
}

impl Default for MultiResLossConfig {
    fn default() -> Self {
        Self {
            n_levels: 4,
            level_weights: vec![1.0, 0.5, 0.25, 0.125],
            loss_types: vec![MultiResLossType::L1, MultiResLossType::Ssim],
            normalize_weights: true,
        }
    }
}

impl MultiResLossConfig {
    /// Validate that all weights are positive.
    pub fn validate(&self) -> Result<(), MultiResLossError> {
        for &w in &self.level_weights {
            if w <= 0.0 || !w.is_finite() {
                return Err(MultiResLossError::InvalidWeight(w));
            }
        }
        if self.n_levels == 0 || self.loss_types.is_empty() {
            return Err(MultiResLossError::NoLevels);
        }
        Ok(())
    }

    /// Compute effective (possibly normalized) weights, clamped to min(pyramid_levels, weight_count).
    fn effective_weights(&self, actual_levels: usize) -> Vec<f32> {
        let n = actual_levels.min(self.level_weights.len());
        let mut weights: Vec<f32> = self.level_weights[..n].to_vec();
        if self.normalize_weights {
            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                for w in &mut weights {
                    *w /= sum;
                }
            }
        }
        weights
    }
}

// ---------------------------------------------------------------------------
// Result Types
// ---------------------------------------------------------------------------

/// Result of multi-resolution loss computation.
pub struct MultiResLossResult {
    pub total_loss: f32,
    /// Raw loss at each pyramid level (averaged across loss types).
    pub per_level_losses: Vec<f32>,
    /// Loss for each loss type (averaged across levels and weights).
    pub per_type_losses: Vec<f32>,
    /// per_level_losses * weights.
    pub weighted_level_losses: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Main Loss Functions
// ---------------------------------------------------------------------------

/// Compute multi-resolution loss between predicted and ground truth images.
pub fn mr_compute_loss(
    predicted: &[f32],
    ground_truth: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    config: &MultiResLossConfig,
) -> Result<MultiResLossResult, MultiResLossError> {
    config.validate()?;

    let expected = width * height * channels;
    if predicted.len() != expected {
        return Err(MultiResLossError::SizeMismatch {
            got: predicted.len(),
            width,
            height,
            channels,
        });
    }
    if ground_truth.len() != expected {
        return Err(MultiResLossError::SizeMismatch {
            got: ground_truth.len(),
            width,
            height,
            channels,
        });
    }

    let pred_pyramid = ImagePyramid::build(predicted, width, height, channels, config.n_levels)?;
    let gt_pyramid = ImagePyramid::build(ground_truth, width, height, channels, config.n_levels)?;

    let actual_levels = pred_pyramid.n_levels().min(gt_pyramid.n_levels());
    let weights = config.effective_weights(actual_levels);
    let n_used = weights.len();

    let n_types = config.loss_types.len();

    // per_level_losses[k] = mean of all loss types at level k
    let mut per_level_losses = vec![0.0_f32; n_used];
    // per_type_losses[t] = weighted mean of loss_type t across levels
    let mut per_type_losses = vec![0.0_f32; n_types];
    let mut weighted_level_losses = vec![0.0_f32; n_used];

    for (k, &w) in weights.iter().enumerate() {
        let pred_lvl = pred_pyramid
            .level(k)
            .ok_or(MultiResLossError::ShapeMismatch)?;
        let gt_lvl = gt_pyramid
            .level(k)
            .ok_or(MultiResLossError::ShapeMismatch)?;

        let mut level_sum = 0.0_f32;
        for (t, loss_type) in config.loss_types.iter().enumerate() {
            let type_loss = mr_level_loss(pred_lvl, gt_lvl, loss_type)?;
            level_sum += type_loss;
            per_type_losses[t] += w * type_loss;
        }

        let level_mean = if n_types > 0 {
            level_sum / n_types as f32
        } else {
            0.0
        };
        per_level_losses[k] = level_mean;
        weighted_level_losses[k] = w * level_mean;
    }

    let total_loss: f32 = weighted_level_losses.iter().sum();

    Ok(MultiResLossResult {
        total_loss,
        per_level_losses,
        per_type_losses,
        weighted_level_losses,
    })
}

/// Compute loss at a single pyramid level for a given loss type.
pub fn mr_level_loss(
    predicted: &PyramidLevel,
    ground_truth: &PyramidLevel,
    loss_type: &MultiResLossType,
) -> Result<f32, MultiResLossError> {
    if predicted.data.len() != ground_truth.data.len() {
        return Err(MultiResLossError::ShapeMismatch);
    }
    match loss_type {
        MultiResLossType::L1 => mr_l1_loss(&predicted.data, &ground_truth.data),
        MultiResLossType::L2 => mr_l2_loss(&predicted.data, &ground_truth.data),
        MultiResLossType::Ssim => mr_ssim_loss(
            &predicted.data,
            &ground_truth.data,
            predicted.width,
            predicted.height,
        ),
        MultiResLossType::Laplacian => mr_laplacian_loss(
            &predicted.data,
            &ground_truth.data,
            predicted.width,
            predicted.height,
        ),
        MultiResLossType::GradientL1 => mr_gradient_l1_loss(
            &predicted.data,
            &ground_truth.data,
            predicted.width,
            predicted.height,
        ),
    }
}

// ---------------------------------------------------------------------------
// Individual Loss Functions
// ---------------------------------------------------------------------------

/// L1 loss: mean(|pred - gt|).
pub fn mr_l1_loss(pred: &[f32], gt: &[f32]) -> Result<f32, MultiResLossError> {
    if pred.len() != gt.len() {
        return Err(MultiResLossError::ShapeMismatch);
    }
    if pred.is_empty() {
        return Ok(0.0);
    }
    let sum: f32 = pred.iter().zip(gt.iter()).map(|(p, g)| (p - g).abs()).sum();
    Ok(sum / pred.len() as f32)
}

/// L2 (MSE) loss: mean((pred - gt)^2).
pub fn mr_l2_loss(pred: &[f32], gt: &[f32]) -> Result<f32, MultiResLossError> {
    if pred.len() != gt.len() {
        return Err(MultiResLossError::ShapeMismatch);
    }
    if pred.is_empty() {
        return Ok(0.0);
    }
    let sum: f32 = pred
        .iter()
        .zip(gt.iter())
        .map(|(p, g)| {
            let d = p - g;
            d * d
        })
        .sum();
    Ok(sum / pred.len() as f32)
}

/// SSIM loss: 1 - ssim(pred, gt).
///
/// Uses a simplified 3×3 window SSIM for efficiency.
/// Channels are inferred from `pred.len() / (width * height)`.
pub fn mr_ssim_loss(
    pred: &[f32],
    gt: &[f32],
    width: usize,
    height: usize,
) -> Result<f32, MultiResLossError> {
    if pred.len() != gt.len() {
        return Err(MultiResLossError::ShapeMismatch);
    }
    let n_pixels = width * height;
    if n_pixels == 0 {
        return Ok(0.0);
    }
    if !pred.len().is_multiple_of(n_pixels) {
        return Err(MultiResLossError::SizeMismatch {
            got: pred.len(),
            width,
            height,
            channels: pred.len() / n_pixels.max(1),
        });
    }
    let channels = pred.len() / n_pixels;

    // Constants following Wang et al. (2004), assuming data in [0,1]
    const C1: f32 = 0.0001;
    const C2: f32 = 0.0009;

    // Need at least a 3×3 neighborhood; if image is 1×1 fall back to perfect
    if width < 3 || height < 3 {
        // Can't form 3×3 windows — return 0 (identical → 0 loss)
        let l1 = mr_l1_loss(pred, gt)?;
        // If images are identical, return 0; otherwise approximate
        return Ok(if l1 < 1e-7 { 0.0 } else { l1.min(1.0) });
    }

    let mut ssim_sum = 0.0_f32;
    let mut count = 0u32;

    for c in 0..channels {
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                // Collect 3×3 window samples
                let mut mu_x = 0.0_f32;
                let mut mu_y = 0.0_f32;
                let mut samples_x = [0.0_f32; 9];
                let mut samples_y = [0.0_f32; 9];
                let mut idx = 0;

                for dy in 0usize..3 {
                    for dx in 0usize..3 {
                        let px = (x + dx).wrapping_sub(1).min(width - 1);
                        let py = (y + dy).wrapping_sub(1).min(height - 1);
                        let flat = (py * width + px) * channels + c;
                        let vx = pred[flat];
                        let vy = gt[flat];
                        samples_x[idx] = vx;
                        samples_y[idx] = vy;
                        mu_x += vx;
                        mu_y += vy;
                        idx += 1;
                    }
                }
                mu_x /= 9.0;
                mu_y /= 9.0;

                let mut sigma_x2 = 0.0_f32;
                let mut sigma_y2 = 0.0_f32;
                let mut sigma_xy = 0.0_f32;
                for i in 0..9 {
                    let dx_i = samples_x[i] - mu_x;
                    let dy_i = samples_y[i] - mu_y;
                    sigma_x2 += dx_i * dx_i;
                    sigma_y2 += dy_i * dy_i;
                    sigma_xy += dx_i * dy_i;
                }
                sigma_x2 /= 9.0;
                sigma_y2 /= 9.0;
                sigma_xy /= 9.0;

                let num = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2);
                let den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2);

                let ssim_val = if den.abs() < 1e-12 { 1.0 } else { num / den };
                ssim_sum += ssim_val.clamp(-1.0, 1.0);
                count += 1;
            }
        }
    }

    if count == 0 {
        return Ok(0.0);
    }
    let mean_ssim = ssim_sum / count as f32;
    Ok((1.0 - mean_ssim).max(0.0))
}

/// Laplacian pyramid loss: L1 between Laplacian pyramid coefficients.
///
/// Channels are inferred from `pred.len() / (width * height)`.
pub fn mr_laplacian_loss(
    pred: &[f32],
    gt: &[f32],
    width: usize,
    height: usize,
) -> Result<f32, MultiResLossError> {
    if pred.len() != gt.len() {
        return Err(MultiResLossError::ShapeMismatch);
    }
    let n_pixels = width * height;
    if n_pixels == 0 {
        return Ok(0.0);
    }
    if !pred.len().is_multiple_of(n_pixels) {
        return Err(MultiResLossError::SizeMismatch {
            got: pred.len(),
            width,
            height,
            channels: pred.len() / n_pixels.max(1),
        });
    }
    let channels = pred.len() / n_pixels;

    let n_levels = 3_usize; // use 3 Laplacian levels
    let pred_lap = mr_laplacian_pyramid(pred, width, height, channels, n_levels)?;
    let gt_lap = mr_laplacian_pyramid(gt, width, height, channels, n_levels)?;

    let n = pred_lap.len().min(gt_lap.len());
    if n == 0 {
        return Ok(0.0);
    }

    let mut total = 0.0_f32;
    for k in 0..n {
        let level_loss = mr_l1_loss(&pred_lap[k], &gt_lap[k])?;
        total += level_loss;
    }
    Ok(total / n as f32)
}

/// Gradient L1 loss: L1 between Sobel gradient magnitudes.
///
/// Channels are inferred from `pred.len() / (width * height)`.
pub fn mr_gradient_l1_loss(
    pred: &[f32],
    gt: &[f32],
    width: usize,
    height: usize,
) -> Result<f32, MultiResLossError> {
    if pred.len() != gt.len() {
        return Err(MultiResLossError::ShapeMismatch);
    }
    let n_pixels = width * height;
    if n_pixels == 0 {
        return Ok(0.0);
    }
    if !pred.len().is_multiple_of(n_pixels) {
        return Err(MultiResLossError::SizeMismatch {
            got: pred.len(),
            width,
            height,
            channels: pred.len() / n_pixels.max(1),
        });
    }
    let channels = pred.len() / n_pixels;

    let pred_mag = mr_sobel_magnitude(pred, width, height, channels);
    let gt_mag = mr_sobel_magnitude(gt, width, height, channels);
    mr_l1_loss(&pred_mag, &gt_mag)
}

// ---------------------------------------------------------------------------
// Image Processing Helpers
// ---------------------------------------------------------------------------

/// Downsample image by factor 2 using box filter (average 2×2 neighborhood).
///
/// Output size: `((width+1)/2, (height+1)/2)` (ceiling division).
pub fn mr_downsample(
    img: &[f32],
    width: usize,
    height: usize,
    channels: usize,
) -> Result<(Vec<f32>, usize, usize), MultiResLossError> {
    if width < 2 || height < 2 {
        return Err(MultiResLossError::ImageTooSmall(width.min(height)));
    }
    let expected = width * height * channels;
    if img.len() != expected {
        return Err(MultiResLossError::SizeMismatch {
            got: img.len(),
            width,
            height,
            channels,
        });
    }

    let new_w = width.div_ceil(2);
    let new_h = height.div_ceil(2);
    let mut out = vec![0.0_f32; new_w * new_h * channels];

    for oy in 0..new_h {
        for ox in 0..new_w {
            for c in 0..channels {
                let mut sum = 0.0_f32;
                let mut count = 0u32;
                // Sample up to 2×2 input pixels, clamping at boundaries
                for dy in 0..2_usize {
                    let iy = (2 * oy + dy).min(height - 1);
                    for dx in 0..2_usize {
                        let ix = (2 * ox + dx).min(width - 1);
                        sum += img[(iy * width + ix) * channels + c];
                        count += 1;
                    }
                }
                out[(oy * new_w + ox) * channels + c] = sum / count as f32;
            }
        }
    }

    Ok((out, new_w, new_h))
}

/// Upsample image by factor 2 using bilinear interpolation to an explicit target size.
pub fn mr_upsample(
    img: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    target_w: usize,
    target_h: usize,
) -> Vec<f32> {
    if width == 0 || height == 0 || target_w == 0 || target_h == 0 {
        return vec![0.0_f32; target_w * target_h * channels];
    }

    let mut out = vec![0.0_f32; target_w * target_h * channels];

    for ty in 0..target_h {
        for tx in 0..target_w {
            // Map target pixel to source coordinates
            // Scale: source x = tx * (width-1) / (target_w-1)
            let sx_f = if target_w == 1 {
                0.0_f32
            } else {
                tx as f32 * (width as f32 - 1.0) / (target_w as f32 - 1.0)
            };
            let sy_f = if target_h == 1 {
                0.0_f32
            } else {
                ty as f32 * (height as f32 - 1.0) / (target_h as f32 - 1.0)
            };

            let sx0 = sx_f.floor() as usize;
            let sy0 = sy_f.floor() as usize;
            let sx1 = (sx0 + 1).min(width - 1);
            let sy1 = (sy0 + 1).min(height - 1);

            let wx1 = sx_f - sx0 as f32;
            let wy1 = sy_f - sy0 as f32;
            let wx0 = 1.0 - wx1;
            let wy0 = 1.0 - wy1;

            for c in 0..channels {
                let v00 = img[(sy0 * width + sx0) * channels + c];
                let v01 = img[(sy0 * width + sx1) * channels + c];
                let v10 = img[(sy1 * width + sx0) * channels + c];
                let v11 = img[(sy1 * width + sx1) * channels + c];

                out[(ty * target_w + tx) * channels + c] =
                    wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
            }
        }
    }

    out
}

/// Apply 3×3 Gaussian blur with kernel \[1,2,1\]/4 (separable).
///
/// Border pixels are handled by clamping coordinates.
pub fn mr_gaussian_blur_3x3(img: &[f32], width: usize, height: usize, channels: usize) -> Vec<f32> {
    if img.is_empty() || width == 0 || height == 0 {
        return img.to_vec();
    }

    // Horizontal pass
    let mut tmp = vec![0.0_f32; width * height * channels];
    let kernel = [0.25_f32, 0.5, 0.25];
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let x0 = x.saturating_sub(1);
                let x2 = (x + 1).min(width - 1);
                let v = kernel[0] * img[(y * width + x0) * channels + c]
                    + kernel[1] * img[(y * width + x) * channels + c]
                    + kernel[2] * img[(y * width + x2) * channels + c];
                tmp[(y * width + x) * channels + c] = v;
            }
        }
    }

    // Vertical pass
    let mut out = vec![0.0_f32; width * height * channels];
    for y in 0..height {
        for x in 0..width {
            for c in 0..channels {
                let y0 = y.saturating_sub(1);
                let y2 = (y + 1).min(height - 1);
                let v = kernel[0] * tmp[(y0 * width + x) * channels + c]
                    + kernel[1] * tmp[(y * width + x) * channels + c]
                    + kernel[2] * tmp[(y2 * width + x) * channels + c];
                out[(y * width + x) * channels + c] = v;
            }
        }
    }

    out
}

/// Build a Laplacian pyramid from an image.
///
/// Returns `n_levels` coefficient maps (or fewer if image becomes too small).
/// Level 0 captures the finest detail; higher levels capture coarser detail.
///
/// Algorithm:
/// ```text
/// gauss[0] = gaussian_blur(img)
/// lap[0]   = img - upsample(downsample(gauss[0]))
/// gauss[k] = downsample(gauss[k-1])
/// lap[k]   = gauss[k-1] - upsample(gauss[k])  for k >= 1
/// ```
pub fn mr_laplacian_pyramid(
    img: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    n_levels: usize,
) -> Result<Vec<Vec<f32>>, MultiResLossError> {
    if n_levels == 0 {
        return Err(MultiResLossError::NoLevels);
    }
    let expected = width * height * channels;
    if img.len() != expected {
        return Err(MultiResLossError::SizeMismatch {
            got: img.len(),
            width,
            height,
            channels,
        });
    }

    let mut lap_levels = Vec::with_capacity(n_levels);

    // Gauss level 0 = blurred original
    let gauss0 = mr_gaussian_blur_3x3(img, width, height, channels);

    if width >= 2 && height >= 2 {
        // lap[0] = img - upsample(downsample(gauss[0]))
        let (down0, dw0, dh0) = mr_downsample(&gauss0, width, height, channels)?;
        let up0 = mr_upsample(&down0, dw0, dh0, channels, width, height);
        let lap0: Vec<f32> = img.iter().zip(up0.iter()).map(|(a, b)| a - b).collect();
        lap_levels.push(lap0);
    } else {
        // Can't downsample; the only level is the blurred image itself
        lap_levels.push(gauss0.clone());
        return Ok(lap_levels);
    }

    // Iteratively build remaining levels
    let mut cur_gauss = gauss0;
    let mut cur_w = width;
    let mut cur_h = height;

    for _ in 1..n_levels {
        if cur_w < 2 || cur_h < 2 {
            break;
        }
        let (next_gauss, next_w, next_h) = mr_downsample(&cur_gauss, cur_w, cur_h, channels)?;

        // lap[k] = gauss[k-1] - upsample(gauss[k], size of gauss[k-1])
        let up = mr_upsample(&next_gauss, next_w, next_h, channels, cur_w, cur_h);
        let lap: Vec<f32> = cur_gauss
            .iter()
            .zip(up.iter())
            .map(|(a, b)| a - b)
            .collect();
        lap_levels.push(lap);

        cur_gauss = next_gauss;
        cur_w = next_w;
        cur_h = next_h;
    }

    Ok(lap_levels)
}

/// Compute Sobel gradient magnitude, averaged across channels.
///
/// Border pixels are set to 0. Output has one value per pixel (`width * height` elements).
pub fn mr_sobel_magnitude(img: &[f32], width: usize, height: usize, channels: usize) -> Vec<f32> {
    let n_pixels = width * height;
    if n_pixels == 0 || channels == 0 {
        return Vec::new();
    }

    let mut out = vec![0.0_f32; n_pixels];

    // Sobel kernels
    //   Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
    //   Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]

    for y in 1..(height.saturating_sub(1)) {
        for x in 1..(width.saturating_sub(1)) {
            let mut mag_sum = 0.0_f32;
            for c in 0..channels {
                let p = |dy: isize, dx: isize| -> f32 {
                    let py = (y as isize + dy) as usize;
                    let px = (x as isize + dx) as usize;
                    img[(py * width + px) * channels + c]
                };

                let gx =
                    -p(-1, -1) + p(-1, 1) - 2.0 * p(0, -1) + 2.0 * p(0, 1) - p(1, -1) + p(1, 1);

                let gy =
                    -p(-1, -1) - 2.0 * p(-1, 0) - p(-1, 1) + p(1, -1) + 2.0 * p(1, 0) + p(1, 1);

                mag_sum += (gx * gx + gy * gy).sqrt();
            }
            out[y * width + x] = mag_sum / channels as f32;
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Statistical summary of a multi-resolution loss result.
pub struct MultiResStats {
    pub loss_by_level: Vec<f32>,
    /// Ratio of finest level loss to coarsest level loss.
    /// < 1.0 means fine detail is easier; > 1.0 means fine detail is harder.
    pub loss_improvement_ratio: f32,
    /// Index of the level with the highest weighted loss.
    pub dominant_scale: usize,
    /// Quality score: 1 / (1 + total_loss).
    pub quality_score: f32,
}

/// Compute summary statistics from a `MultiResLossResult`.
pub fn mr_compute_stats(result: &MultiResLossResult) -> MultiResStats {
    let loss_by_level = result.per_level_losses.clone();

    let dominant_scale = result
        .weighted_level_losses
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let loss_improvement_ratio = if result.per_level_losses.len() >= 2 {
        let finest = *result.per_level_losses.first().unwrap_or(&0.0);
        let coarsest = *result.per_level_losses.last().unwrap_or(&1.0);
        if coarsest.abs() < 1e-10 {
            1.0
        } else {
            finest / coarsest
        }
    } else {
        1.0
    };

    let quality_score = 1.0 / (1.0 + result.total_loss);

    MultiResStats {
        loss_by_level,
        loss_improvement_ratio,
        dominant_scale,
        quality_score,
    }
}

/// Format a `MultiResLossResult` as a human-readable string.
pub fn format_mr_result(result: &MultiResLossResult) -> String {
    let mut s = format!("MultiResLoss(total={:.6}", result.total_loss);
    s.push_str(", levels=[");
    for (k, &l) in result.per_level_losses.iter().enumerate() {
        if k > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("{:.4}", l));
    }
    s.push_str("], weighted=[");
    for (k, &w) in result.weighted_level_losses.iter().enumerate() {
        if k > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("{:.4}", w));
    }
    s.push_str("])");
    s
}

/// Format `MultiResStats` as a human-readable string.
pub fn format_mr_stats(stats: &MultiResStats) -> String {
    format!(
        "MultiResStats(quality={:.4}, dominant_scale={}, ratio={:.4})",
        stats.quality_score, stats.dominant_scale, stats.loss_improvement_ratio
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a uniform flat image
    fn uniform_image(w: usize, h: usize, c: usize, val: f32) -> Vec<f32> {
        vec![val; w * h * c]
    }

    // Helper: create a checkerboard pattern (0.0 and 1.0)
    fn checkerboard(w: usize, h: usize, c: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; w * h * c];
        for y in 0..h {
            for x in 0..w {
                let val = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
                for ch in 0..c {
                    v[(y * w + x) * c + ch] = val;
                }
            }
        }
        v
    }

    // Helper: create a horizontal edge image (top half = 0, bottom half = 1)
    fn edge_image(w: usize, h: usize, c: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; w * h * c];
        for y in (h / 2)..h {
            for x in 0..w {
                for ch in 0..c {
                    v[(y * w + x) * c + ch] = 1.0;
                }
            }
        }
        v
    }

    // ---------------------------------------------------------------------------
    // mr_downsample
    // ---------------------------------------------------------------------------

    #[test]
    fn test_downsample_4x4_to_2x2() {
        let img: Vec<f32> = (0..16).map(|i| i as f32).collect();
        // 4×4 single channel → 2×2
        let (out, ow, oh) = mr_downsample(&img, 4, 4, 1).expect("downsample failed");
        assert_eq!(ow, 2);
        assert_eq!(oh, 2);
        assert_eq!(out.len(), 4);
        // Top-left 2×2 block: [0,1,4,5] → mean = 2.5
        assert!((out[0] - 2.5).abs() < 1e-5, "TL={}", out[0]);
        // Top-right: [2,3,6,7] → 4.5
        assert!((out[1] - 4.5).abs() < 1e-5, "TR={}", out[1]);
    }

    #[test]
    fn test_downsample_uniform_stays_uniform() {
        let img = uniform_image(8, 8, 3, 0.7);
        let (out, ow, oh) = mr_downsample(&img, 8, 8, 3).expect("downsample");
        assert_eq!(ow, 4);
        assert_eq!(oh, 4);
        for &v in &out {
            assert!((v - 0.7).abs() < 1e-5);
        }
    }

    #[test]
    fn test_downsample_odd_size() {
        let img = uniform_image(5, 5, 1, 1.0);
        let (out, ow, oh) = mr_downsample(&img, 5, 5, 1).expect("downsample");
        assert_eq!(ow, 3);
        assert_eq!(oh, 3);
        assert_eq!(out.len(), 9);
    }

    #[test]
    fn test_downsample_too_small_error() {
        let img = vec![1.0_f32];
        let res = mr_downsample(&img, 1, 1, 1);
        assert!(matches!(res, Err(MultiResLossError::ImageTooSmall(_))));
    }

    #[test]
    fn test_downsample_multi_channel() {
        let img = uniform_image(4, 4, 3, 0.5);
        let (out, ow, oh) = mr_downsample(&img, 4, 4, 3).expect("downsample");
        assert_eq!(ow, 2);
        assert_eq!(oh, 2);
        assert_eq!(out.len(), 2 * 2 * 3);
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-5);
        }
    }

    // ---------------------------------------------------------------------------
    // mr_upsample
    // ---------------------------------------------------------------------------

    #[test]
    fn test_upsample_2x2_to_4x4() {
        // 2×2 uniform → 4×4
        let img = uniform_image(2, 2, 1, 0.6);
        let out = mr_upsample(&img, 2, 2, 1, 4, 4);
        assert_eq!(out.len(), 16);
        for &v in &out {
            assert!((v - 0.6).abs() < 1e-5);
        }
    }

    #[test]
    fn test_upsample_uniform_stays_uniform() {
        let img = uniform_image(3, 3, 2, 0.3);
        let out = mr_upsample(&img, 3, 3, 2, 5, 5);
        assert_eq!(out.len(), 5 * 5 * 2);
        for &v in &out {
            assert!((v - 0.3).abs() < 1e-5);
        }
    }

    #[test]
    fn test_upsample_identity_target() {
        // Upsample to same size should return identical (or very close)
        let img = checkerboard(4, 4, 1);
        let out = mr_upsample(&img, 4, 4, 1, 4, 4);
        assert_eq!(out.len(), img.len());
        for (a, b) in img.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_upsample_roundtrip_size() {
        // Downsample 5×5 → 3×3 → upsample back to 5×5
        let img = uniform_image(5, 5, 1, 1.0);
        let (down, dw, dh) = mr_downsample(&img, 5, 5, 1).expect("down");
        let up = mr_upsample(&down, dw, dh, 1, 5, 5);
        assert_eq!(up.len(), 5 * 5);
    }

    // ---------------------------------------------------------------------------
    // mr_gaussian_blur_3x3
    // ---------------------------------------------------------------------------

    #[test]
    fn test_gaussian_blur_uniform_unchanged() {
        let img = uniform_image(6, 6, 1, 0.5);
        let out = mr_gaussian_blur_3x3(&img, 6, 6, 1);
        assert_eq!(out.len(), img.len());
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_gaussian_blur_edge_pixels_handled() {
        let img = checkerboard(8, 8, 1);
        let out = mr_gaussian_blur_3x3(&img, 8, 8, 1);
        // Just verify no panic and output size is correct
        assert_eq!(out.len(), img.len());
        // All values should be in [0, 1]
        for &v in &out {
            assert!((0.0..=1.0 + 1e-5).contains(&v), "v={}", v);
        }
    }

    #[test]
    fn test_gaussian_blur_multi_channel() {
        let img = uniform_image(4, 4, 3, 0.8);
        let out = mr_gaussian_blur_3x3(&img, 4, 4, 3);
        assert_eq!(out.len(), 4 * 4 * 3);
        for &v in &out {
            assert!((v - 0.8).abs() < 1e-5);
        }
    }

    // ---------------------------------------------------------------------------
    // mr_l1_loss
    // ---------------------------------------------------------------------------

    #[test]
    fn test_l1_identical_is_zero() {
        let img = checkerboard(8, 8, 3);
        let loss = mr_l1_loss(&img, &img).expect("l1");
        assert!(loss.abs() < 1e-7);
    }

    #[test]
    fn test_l1_known_difference() {
        let pred = vec![0.0_f32, 1.0, 0.0, 1.0];
        let gt = vec![1.0_f32, 0.0, 1.0, 0.0];
        let loss = mr_l1_loss(&pred, &gt).expect("l1");
        assert!((loss - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l1_partial_difference() {
        let pred = vec![0.0_f32, 0.0];
        let gt = vec![0.5_f32, 0.5];
        let loss = mr_l1_loss(&pred, &gt).expect("l1");
        assert!((loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_l1_shape_mismatch_error() {
        let res = mr_l1_loss(&[1.0_f32, 2.0], &[1.0_f32]);
        assert!(matches!(res, Err(MultiResLossError::ShapeMismatch)));
    }

    // ---------------------------------------------------------------------------
    // mr_l2_loss
    // ---------------------------------------------------------------------------

    #[test]
    fn test_l2_identical_is_zero() {
        let img = uniform_image(4, 4, 1, 0.5);
        let loss = mr_l2_loss(&img, &img).expect("l2");
        assert!(loss.abs() < 1e-7);
    }

    #[test]
    fn test_l2_known_difference() {
        // Mean of (1-0)^2 = 1.0
        let pred = vec![0.0_f32; 4];
        let gt = vec![1.0_f32; 4];
        let loss = mr_l2_loss(&pred, &gt).expect("l2");
        assert!((loss - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_shape_mismatch_error() {
        let res = mr_l2_loss(&[1.0_f32], &[1.0_f32, 2.0]);
        assert!(matches!(res, Err(MultiResLossError::ShapeMismatch)));
    }

    // ---------------------------------------------------------------------------
    // mr_ssim_loss
    // ---------------------------------------------------------------------------

    #[test]
    fn test_ssim_identical_is_zero() {
        let img = checkerboard(8, 8, 1);
        let loss = mr_ssim_loss(&img, &img, 8, 8).expect("ssim");
        assert!(loss < 1e-5, "loss={}", loss);
    }

    #[test]
    fn test_ssim_different_is_positive() {
        let pred = uniform_image(8, 8, 1, 0.0);
        let gt = uniform_image(8, 8, 1, 1.0);
        let loss = mr_ssim_loss(&pred, &gt, 8, 8).expect("ssim");
        assert!(loss > 0.0, "Expected positive SSIM loss");
    }

    #[test]
    fn test_ssim_small_image_no_panic() {
        // 2×2 images — fall back to L1 approximation
        let pred = vec![0.0_f32, 1.0, 0.0, 1.0];
        let gt = vec![1.0_f32, 0.0, 1.0, 0.0];
        let loss = mr_ssim_loss(&pred, &gt, 2, 2).expect("ssim small");
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_ssim_shape_mismatch() {
        let res = mr_ssim_loss(&[1.0_f32, 2.0], &[1.0_f32], 1, 1);
        assert!(matches!(res, Err(MultiResLossError::ShapeMismatch)));
    }

    // ---------------------------------------------------------------------------
    // mr_gradient_l1_loss
    // ---------------------------------------------------------------------------

    #[test]
    fn test_gradient_l1_identical_is_zero() {
        let img = edge_image(8, 8, 1);
        let loss = mr_gradient_l1_loss(&img, &img, 8, 8).expect("grad");
        assert!(loss.abs() < 1e-7);
    }

    #[test]
    fn test_gradient_l1_edge_vs_flat_positive() {
        let pred = edge_image(8, 8, 1);
        let gt = uniform_image(8, 8, 1, 0.5);
        let loss = mr_gradient_l1_loss(&pred, &gt, 8, 8).expect("grad");
        assert!(loss > 0.0);
    }

    #[test]
    fn test_gradient_l1_shape_mismatch() {
        let res = mr_gradient_l1_loss(&[1.0_f32], &[1.0_f32, 2.0], 1, 1);
        assert!(matches!(res, Err(MultiResLossError::ShapeMismatch)));
    }

    // ---------------------------------------------------------------------------
    // mr_sobel_magnitude
    // ---------------------------------------------------------------------------

    #[test]
    fn test_sobel_flat_image_is_zero() {
        let img = uniform_image(6, 6, 1, 0.5);
        let mag = mr_sobel_magnitude(&img, 6, 6, 1);
        assert_eq!(mag.len(), 36);
        // Interior pixels should all be 0 for a uniform image
        for y in 1..5 {
            for x in 1..5 {
                assert!(
                    mag[y * 6 + x].abs() < 1e-5,
                    "mag[{},{}]={}",
                    y,
                    x,
                    mag[y * 6 + x]
                );
            }
        }
    }

    #[test]
    fn test_sobel_edge_image_nonzero() {
        let img = edge_image(8, 8, 1);
        let mag = mr_sobel_magnitude(&img, 8, 8, 1);
        assert_eq!(mag.len(), 64);
        // Should have nonzero gradient at the edge row
        let max_mag = mag.iter().cloned().fold(0.0_f32, f32::max);
        assert!(max_mag > 0.0, "Expected nonzero gradient");
    }

    #[test]
    fn test_sobel_border_pixels_are_zero() {
        let img = checkerboard(6, 6, 1);
        let mag = mr_sobel_magnitude(&img, 6, 6, 1);
        // Border pixels should be 0
        for x in 0..6 {
            assert_eq!(mag[x], 0.0, "top border x={}", x);
            assert_eq!(mag[5 * 6 + x], 0.0, "bottom border x={}", x);
        }
        for y in 0..6 {
            assert_eq!(mag[y * 6], 0.0, "left border y={}", y);
            assert_eq!(mag[y * 6 + 5], 0.0, "right border y={}", y);
        }
    }

    #[test]
    fn test_sobel_multi_channel() {
        let img = edge_image(8, 8, 3);
        let mag = mr_sobel_magnitude(&img, 8, 8, 3);
        // Output has one value per pixel
        assert_eq!(mag.len(), 64);
    }

    // ---------------------------------------------------------------------------
    // mr_laplacian_pyramid
    // ---------------------------------------------------------------------------

    #[test]
    fn test_laplacian_pyramid_length() {
        let img = checkerboard(16, 16, 1);
        let lap = mr_laplacian_pyramid(&img, 16, 16, 1, 4).expect("lap");
        assert_eq!(lap.len(), 4);
    }

    #[test]
    fn test_laplacian_pyramid_level_sizes() {
        let img = checkerboard(8, 8, 1);
        let lap = mr_laplacian_pyramid(&img, 8, 8, 1, 3).expect("lap");
        // lap[0] should be same size as original (8×8×1 = 64)
        assert_eq!(lap[0].len(), 64);
    }

    #[test]
    fn test_laplacian_pyramid_no_levels_error() {
        let img = vec![1.0_f32];
        let res = mr_laplacian_pyramid(&img, 1, 1, 1, 0);
        assert!(matches!(res, Err(MultiResLossError::NoLevels)));
    }

    #[test]
    fn test_laplacian_pyramid_single_level() {
        let img = uniform_image(4, 4, 1, 0.5);
        let lap = mr_laplacian_pyramid(&img, 4, 4, 1, 1).expect("lap");
        assert_eq!(lap.len(), 1);
    }

    // ---------------------------------------------------------------------------
    // mr_laplacian_loss
    // ---------------------------------------------------------------------------

    #[test]
    fn test_laplacian_loss_identical_near_zero() {
        let img = checkerboard(16, 16, 1);
        let loss = mr_laplacian_loss(&img, &img, 16, 16).expect("lap_loss");
        assert!(loss < 1e-5, "loss={}", loss);
    }

    #[test]
    fn test_laplacian_loss_different_positive() {
        let pred = uniform_image(8, 8, 1, 0.0);
        let gt = checkerboard(8, 8, 1);
        let loss = mr_laplacian_loss(&pred, &gt, 8, 8).expect("lap_loss");
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_laplacian_loss_shape_mismatch() {
        let res = mr_laplacian_loss(&[1.0_f32, 2.0], &[1.0_f32], 1, 1);
        assert!(matches!(res, Err(MultiResLossError::ShapeMismatch)));
    }

    // ---------------------------------------------------------------------------
    // ImagePyramid
    // ---------------------------------------------------------------------------

    #[test]
    fn test_pyramid_build_correct_levels() {
        let img = checkerboard(16, 16, 1);
        let pyr = ImagePyramid::build(&img, 16, 16, 1, 4).expect("build");
        assert_eq!(pyr.n_levels(), 4);
    }

    #[test]
    fn test_pyramid_scale_halves() {
        let img = checkerboard(16, 16, 1);
        let pyr = ImagePyramid::build(&img, 16, 16, 1, 4).expect("build");
        assert!((pyr.levels[0].scale - 1.0).abs() < 1e-6);
        assert!((pyr.levels[1].scale - 0.5).abs() < 1e-6);
        assert!((pyr.levels[2].scale - 0.25).abs() < 1e-6);
        assert!((pyr.levels[3].scale - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_pyramid_original_is_level_0() {
        let img = checkerboard(8, 8, 3);
        let pyr = ImagePyramid::build(&img, 8, 8, 3, 3).expect("build");
        assert_eq!(pyr.original().width, 8);
        assert_eq!(pyr.original().height, 8);
        assert_eq!(pyr.original().data.len(), 8 * 8 * 3);
    }

    #[test]
    fn test_pyramid_level_out_of_range_returns_none() {
        let img = checkerboard(8, 8, 1);
        let pyr = ImagePyramid::build(&img, 8, 8, 1, 3).expect("build");
        assert!(pyr.level(100).is_none());
        assert!(pyr.level(3).is_none());
    }

    #[test]
    fn test_pyramid_level_in_range() {
        let img = checkerboard(8, 8, 1);
        let pyr = ImagePyramid::build(&img, 8, 8, 1, 3).expect("build");
        assert!(pyr.level(0).is_some());
        assert!(pyr.level(2).is_some());
    }

    #[test]
    fn test_pyramid_size_mismatch_error() {
        let img = vec![1.0_f32; 10]; // wrong size
        let res = ImagePyramid::build(&img, 4, 4, 1, 2);
        assert!(matches!(res, Err(MultiResLossError::SizeMismatch { .. })));
    }

    #[test]
    fn test_pyramid_no_levels_error() {
        let img = uniform_image(4, 4, 1, 0.5);
        let res = ImagePyramid::build(&img, 4, 4, 1, 0);
        assert!(matches!(res, Err(MultiResLossError::NoLevels)));
    }

    #[test]
    fn test_pyramid_stops_when_too_small() {
        // A 2×2 image can only have 1 meaningful level + 1 downsampled
        let img = uniform_image(2, 2, 1, 0.5);
        let pyr = ImagePyramid::build(&img, 2, 2, 1, 10).expect("build");
        // Should not have 10 levels — image becomes too small
        assert!(pyr.n_levels() < 10);
    }

    // ---------------------------------------------------------------------------
    // mr_compute_loss
    // ---------------------------------------------------------------------------

    #[test]
    fn test_compute_loss_uniform_near_zero() {
        let pred = uniform_image(8, 8, 3, 0.5);
        let gt = uniform_image(8, 8, 3, 0.5);
        let config = MultiResLossConfig::default();
        let result = mr_compute_loss(&pred, &gt, 8, 8, 3, &config).expect("compute");
        assert!(result.total_loss < 1e-5, "total={}", result.total_loss);
    }

    #[test]
    fn test_compute_loss_different_images_positive() {
        let pred = uniform_image(8, 8, 3, 0.0);
        let gt = uniform_image(8, 8, 3, 1.0);
        let config = MultiResLossConfig::default();
        let result = mr_compute_loss(&pred, &gt, 8, 8, 3, &config).expect("compute");
        assert!(result.total_loss > 0.0, "Expected positive loss");
    }

    #[test]
    fn test_compute_loss_non_negative() {
        let pred = checkerboard(8, 8, 3);
        let gt = edge_image(8, 8, 3);
        let config = MultiResLossConfig::default();
        let result = mr_compute_loss(&pred, &gt, 8, 8, 3, &config).expect("compute");
        assert!(result.total_loss >= 0.0);
        for &l in &result.per_level_losses {
            assert!(l >= 0.0);
        }
        for &w in &result.weighted_level_losses {
            assert!(w >= 0.0);
        }
    }

    #[test]
    fn test_compute_loss_size_mismatch_error() {
        let pred = uniform_image(8, 8, 3, 0.5);
        let gt = uniform_image(8, 8, 3, 0.5);
        let config = MultiResLossConfig::default();
        // Pass wrong size
        let res = mr_compute_loss(&pred[..10], &gt, 8, 8, 3, &config);
        assert!(matches!(res, Err(MultiResLossError::SizeMismatch { .. })));
    }

    #[test]
    fn test_compute_loss_with_all_loss_types() {
        let pred = edge_image(16, 16, 1);
        let gt = checkerboard(16, 16, 1);
        let config = MultiResLossConfig {
            n_levels: 3,
            level_weights: vec![1.0, 0.5, 0.25],
            loss_types: vec![
                MultiResLossType::L1,
                MultiResLossType::L2,
                MultiResLossType::Ssim,
                MultiResLossType::Laplacian,
                MultiResLossType::GradientL1,
            ],
            normalize_weights: true,
        };
        let result = mr_compute_loss(&pred, &gt, 16, 16, 1, &config).expect("compute all types");
        assert!(result.total_loss >= 0.0);
        assert_eq!(result.per_type_losses.len(), 5);
    }

    #[test]
    fn test_compute_loss_per_level_count() {
        let pred = checkerboard(16, 16, 1);
        let gt = edge_image(16, 16, 1);
        let config = MultiResLossConfig {
            n_levels: 3,
            level_weights: vec![1.0, 0.5, 0.25],
            loss_types: vec![MultiResLossType::L1],
            normalize_weights: false,
        };
        let result = mr_compute_loss(&pred, &gt, 16, 16, 1, &config).expect("compute");
        assert_eq!(result.per_level_losses.len(), 3);
        assert_eq!(result.weighted_level_losses.len(), 3);
    }

    // ---------------------------------------------------------------------------
    // mr_level_loss
    // ---------------------------------------------------------------------------

    #[test]
    fn test_level_loss_l1() {
        let lvl_pred = PyramidLevel {
            data: vec![0.0_f32; 16],
            width: 4,
            height: 4,
            scale: 1.0,
        };
        let lvl_gt = PyramidLevel {
            data: vec![1.0_f32; 16],
            width: 4,
            height: 4,
            scale: 1.0,
        };
        let loss = mr_level_loss(&lvl_pred, &lvl_gt, &MultiResLossType::L1).expect("ll");
        assert!((loss - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_level_loss_l2() {
        let lvl_pred = PyramidLevel {
            data: vec![0.0_f32; 16],
            width: 4,
            height: 4,
            scale: 1.0,
        };
        let lvl_gt = PyramidLevel {
            data: vec![1.0_f32; 16],
            width: 4,
            height: 4,
            scale: 1.0,
        };
        let loss = mr_level_loss(&lvl_pred, &lvl_gt, &MultiResLossType::L2).expect("ll");
        assert!((loss - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_level_loss_ssim() {
        let pred_data = checkerboard(8, 8, 1);
        let lvl_pred = PyramidLevel {
            data: pred_data.clone(),
            width: 8,
            height: 8,
            scale: 1.0,
        };
        let lvl_gt = PyramidLevel {
            data: pred_data.clone(),
            width: 8,
            height: 8,
            scale: 1.0,
        };
        let loss = mr_level_loss(&lvl_pred, &lvl_gt, &MultiResLossType::Ssim).expect("ll ssim");
        assert!(loss < 1e-5, "loss={}", loss);
    }

    #[test]
    fn test_level_loss_laplacian() {
        let data = checkerboard(8, 8, 1);
        let lvl_pred = PyramidLevel {
            data: data.clone(),
            width: 8,
            height: 8,
            scale: 1.0,
        };
        let lvl_gt = PyramidLevel {
            data: data.clone(),
            width: 8,
            height: 8,
            scale: 1.0,
        };
        let loss = mr_level_loss(&lvl_pred, &lvl_gt, &MultiResLossType::Laplacian).expect("ll lap");
        assert!(loss < 1e-5, "loss={}", loss);
    }

    #[test]
    fn test_level_loss_gradient_l1() {
        let data = edge_image(8, 8, 1);
        let lvl_pred = PyramidLevel {
            data: data.clone(),
            width: 8,
            height: 8,
            scale: 1.0,
        };
        let lvl_gt = PyramidLevel {
            data: data.clone(),
            width: 8,
            height: 8,
            scale: 1.0,
        };
        let loss =
            mr_level_loss(&lvl_pred, &lvl_gt, &MultiResLossType::GradientL1).expect("ll grad");
        assert!(loss.abs() < 1e-7);
    }

    #[test]
    fn test_level_loss_shape_mismatch() {
        let lvl_pred = PyramidLevel {
            data: vec![0.0_f32; 4],
            width: 2,
            height: 2,
            scale: 1.0,
        };
        let lvl_gt = PyramidLevel {
            data: vec![0.0_f32; 9],
            width: 3,
            height: 3,
            scale: 1.0,
        };
        let res = mr_level_loss(&lvl_pred, &lvl_gt, &MultiResLossType::L1);
        assert!(matches!(res, Err(MultiResLossError::ShapeMismatch)));
    }

    // ---------------------------------------------------------------------------
    // mr_compute_stats
    // ---------------------------------------------------------------------------

    #[test]
    fn test_stats_quality_score() {
        let result = MultiResLossResult {
            total_loss: 0.0,
            per_level_losses: vec![0.0],
            per_type_losses: vec![0.0],
            weighted_level_losses: vec![0.0],
        };
        let stats = mr_compute_stats(&result);
        assert!((stats.quality_score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_stats_quality_decreases_with_loss() {
        let result_low = MultiResLossResult {
            total_loss: 0.1,
            per_level_losses: vec![0.1],
            per_type_losses: vec![0.1],
            weighted_level_losses: vec![0.1],
        };
        let result_high = MultiResLossResult {
            total_loss: 0.9,
            per_level_losses: vec![0.9],
            per_type_losses: vec![0.9],
            weighted_level_losses: vec![0.9],
        };
        let stats_low = mr_compute_stats(&result_low);
        let stats_high = mr_compute_stats(&result_high);
        assert!(stats_low.quality_score > stats_high.quality_score);
    }

    #[test]
    fn test_stats_dominant_scale() {
        let result = MultiResLossResult {
            total_loss: 1.0,
            per_level_losses: vec![0.1, 0.5, 0.3],
            per_type_losses: vec![0.5],
            weighted_level_losses: vec![0.1, 0.5, 0.3],
        };
        let stats = mr_compute_stats(&result);
        assert_eq!(
            stats.dominant_scale, 1,
            "Highest weighted loss is at index 1"
        );
    }

    #[test]
    fn test_stats_improvement_ratio() {
        let result = MultiResLossResult {
            total_loss: 0.5,
            per_level_losses: vec![0.2, 0.4],
            per_type_losses: vec![0.3],
            weighted_level_losses: vec![0.2, 0.2],
        };
        let stats = mr_compute_stats(&result);
        // fine(0.2) / coarse(0.4) = 0.5
        assert!((stats.loss_improvement_ratio - 0.5).abs() < 1e-5);
    }

    // ---------------------------------------------------------------------------
    // MultiResLossConfig
    // ---------------------------------------------------------------------------

    #[test]
    fn test_config_default_n_levels() {
        let cfg = MultiResLossConfig::default();
        assert_eq!(cfg.n_levels, 4);
    }

    #[test]
    fn test_config_default_weights() {
        let cfg = MultiResLossConfig::default();
        assert_eq!(cfg.level_weights.len(), 4);
        assert!((cfg.level_weights[0] - 1.0).abs() < 1e-6);
        assert!((cfg.level_weights[1] - 0.5).abs() < 1e-6);
        assert!((cfg.level_weights[2] - 0.25).abs() < 1e-6);
        assert!((cfg.level_weights[3] - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_config_default_loss_types() {
        let cfg = MultiResLossConfig::default();
        assert_eq!(cfg.loss_types.len(), 2);
        assert!(cfg.loss_types.contains(&MultiResLossType::L1));
        assert!(cfg.loss_types.contains(&MultiResLossType::Ssim));
    }

    #[test]
    fn test_config_invalid_weight_error() {
        let cfg = MultiResLossConfig {
            n_levels: 2,
            level_weights: vec![1.0, -0.5],
            loss_types: vec![MultiResLossType::L1],
            normalize_weights: false,
        };
        assert!(matches!(
            cfg.validate(),
            Err(MultiResLossError::InvalidWeight(_))
        ));
    }

    #[test]
    fn test_config_zero_weight_error() {
        let cfg = MultiResLossConfig {
            n_levels: 2,
            level_weights: vec![0.0, 1.0],
            loss_types: vec![MultiResLossType::L1],
            normalize_weights: false,
        };
        assert!(matches!(
            cfg.validate(),
            Err(MultiResLossError::InvalidWeight(_))
        ));
    }

    #[test]
    fn test_config_normalize_weights() {
        let cfg = MultiResLossConfig {
            n_levels: 2,
            level_weights: vec![2.0, 2.0],
            loss_types: vec![MultiResLossType::L1],
            normalize_weights: true,
        };
        let weights = cfg.effective_weights(2);
        assert!((weights[0] - 0.5).abs() < 1e-6);
        assert!((weights[1] - 0.5).abs() < 1e-6);
    }

    // ---------------------------------------------------------------------------
    // format functions
    // ---------------------------------------------------------------------------

    #[test]
    fn test_format_mr_result_non_empty() {
        let result = MultiResLossResult {
            total_loss: 0.42,
            per_level_losses: vec![0.1, 0.2],
            per_type_losses: vec![0.15],
            weighted_level_losses: vec![0.1, 0.1],
        };
        let s = format_mr_result(&result);
        assert!(!s.is_empty());
        assert!(s.contains("0.42") || s.contains("MultiRes"));
    }

    #[test]
    fn test_format_mr_stats_non_empty() {
        let result = MultiResLossResult {
            total_loss: 0.5,
            per_level_losses: vec![0.3, 0.7],
            per_type_losses: vec![0.5],
            weighted_level_losses: vec![0.3, 0.7],
        };
        let stats = mr_compute_stats(&result);
        let s = format_mr_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("quality") || s.contains("MultiRes"));
    }

    // ---------------------------------------------------------------------------
    // Error cases
    // ---------------------------------------------------------------------------

    #[test]
    fn test_error_image_too_small() {
        let img = vec![1.0_f32];
        let res = mr_downsample(&img, 1, 1, 1);
        assert!(matches!(res, Err(MultiResLossError::ImageTooSmall(_))));
    }

    #[test]
    fn test_error_no_levels() {
        let img = uniform_image(4, 4, 1, 0.5);
        let res = ImagePyramid::build(&img, 4, 4, 1, 0);
        assert!(matches!(res, Err(MultiResLossError::NoLevels)));
    }

    #[test]
    fn test_error_shape_mismatch_gt() {
        let pred = uniform_image(8, 8, 3, 0.5);
        let gt = uniform_image(4, 4, 3, 0.5); // different size
        let config = MultiResLossConfig::default();
        let res = mr_compute_loss(&pred, &gt, 8, 8, 3, &config);
        assert!(matches!(res, Err(MultiResLossError::SizeMismatch { .. })));
    }

    #[test]
    fn test_error_invalid_weight() {
        let cfg = MultiResLossConfig {
            n_levels: 1,
            level_weights: vec![f32::NAN],
            loss_types: vec![MultiResLossType::L1],
            normalize_weights: false,
        };
        assert!(matches!(
            cfg.validate(),
            Err(MultiResLossError::InvalidWeight(_))
        ));
    }
}
