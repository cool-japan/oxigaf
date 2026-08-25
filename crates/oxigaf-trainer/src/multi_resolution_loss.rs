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

    /// Access the original (full-resolution) level, or `None` if this
    /// pyramid has no levels.
    ///
    /// `build()` always inserts at least level 0, so this only returns
    /// `None` for a pyramid that was hand-constructed with an empty
    /// `levels` vec (the field is public) rather than via `build()`.
    pub fn original(&self) -> Option<&PyramidLevel> {
        self.levels.first()
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

/// Structural dissimilarity `1 - SSIM(pred, gt)`, clamped at `0.0`.
///
/// Uses the **same** SSIM definition as [`crate::loss::ssim_loss`] and
/// [`crate::view_synthesis_eval::eval_ssim`]: a separable Gaussian window
/// (11 taps, σ = 1.5) with replicate-boundary padding, `C1 = (0.01)²`,
/// `C2 = (0.03)²`, averaged over every pixel and channel. The kernel itself
/// comes from the shared [`crate::loss::gaussian_kernel_1d`] helper, so this
/// pyramid loss and the primary training objective cannot drift apart.
///
/// The one concession to the pyramid is window *size*: an 11-tap window is
/// meaningless on a deep pyramid level only a few pixels across, so the window
/// shrinks to the largest odd size that fits (`min(11, min(width, height))`,
/// rounded down to odd), with σ scaled by the same factor to preserve its
/// shape. Consequently this agrees numerically with
/// [`crate::loss::ssim_loss`] only when `min(width, height) >= 11`, where both
/// use the full 11-tap window; below that it is the same estimator evaluated
/// on a smaller support, not a different one.
///
/// Channels are inferred from `pred.len() / (width * height)` (any count, not
/// just RGB — pyramid levels are frequently single-channel).
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
    if channels == 0 {
        return Ok(0.0);
    }

    let taps = mr_ssim_window_size(width, height);
    // σ scaled with the window so a shrunken kernel keeps the reference
    // window's shape instead of becoming a near-box filter.
    let sigma = MR_SSIM_SIGMA * taps as f32 / MR_SSIM_TAPS as f32;
    let kernel = crate::loss::gaussian_kernel_1d(taps, sigma);

    let mut plane_pred = vec![0.0_f32; n_pixels];
    let mut plane_gt = vec![0.0_f32; n_pixels];
    let mut ssim_sum = 0.0_f32;
    for c in 0..channels {
        for (i, (dst_pred, dst_gt)) in plane_pred.iter_mut().zip(plane_gt.iter_mut()).enumerate() {
            *dst_pred = pred[i * channels + c];
            *dst_gt = gt[i * channels + c];
        }
        ssim_sum += mr_ssim_channel(&plane_pred, &plane_gt, width, height, &kernel);
    }

    let mean_ssim = ssim_sum / channels as f32;
    Ok((1.0 - mean_ssim).max(0.0))
}

/// Taps in the reference SSIM window, matching [`crate::loss::ssim_loss`].
const MR_SSIM_TAPS: usize = 11;

/// Standard deviation of the reference SSIM window, in pixels.
const MR_SSIM_SIGMA: f32 = 1.5;

/// Largest odd window (≥ 1) that fits inside a `width × height` image, capped
/// at [`MR_SSIM_TAPS`].
fn mr_ssim_window_size(width: usize, height: usize) -> usize {
    let fit = width.min(height).clamp(1, MR_SSIM_TAPS);
    if fit.is_multiple_of(2) {
        fit - 1
    } else {
        fit
    }
}

/// Mean SSIM over a single-channel plane, using the same statistics as
/// `crate::loss`'s `ssim_channel`.
fn mr_ssim_channel(pred: &[f32], gt: &[f32], width: usize, height: usize, kernel: &[f32]) -> f32 {
    // (K₁ L)² and (K₂ L)² for L = 1, identical to `crate::loss::ssim_channel`.
    const C1: f32 = 0.01 * 0.01;
    const C2: f32 = 0.03 * 0.03;

    let n = width * height;
    if n == 0 {
        return 0.0;
    }

    let pred_sq: Vec<f32> = pred.iter().map(|&v| v * v).collect();
    let gt_sq: Vec<f32> = gt.iter().map(|&v| v * v).collect();
    let pred_gt: Vec<f32> = pred.iter().zip(gt.iter()).map(|(&a, &b)| a * b).collect();

    let mu_x = mr_convolve_separable(pred, width, height, kernel);
    let mu_y = mr_convolve_separable(gt, width, height, kernel);
    let ex_sq = mr_convolve_separable(&pred_sq, width, height, kernel);
    let ey_sq = mr_convolve_separable(&gt_sq, width, height, kernel);
    let exy = mr_convolve_separable(&pred_gt, width, height, kernel);

    let mut ssim_sum = 0.0_f32;
    for i in 0..n {
        let mu_x_sq = mu_x[i] * mu_x[i];
        let mu_y_sq = mu_y[i] * mu_y[i];
        let mu_xy = mu_x[i] * mu_y[i];
        let sigma_x_sq = ex_sq[i] - mu_x_sq;
        let sigma_y_sq = ey_sq[i] - mu_y_sq;
        let sigma_xy = exy[i] - mu_xy;

        let num = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2);
        let den = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2);
        ssim_sum += num / den;
    }
    ssim_sum / n as f32
}

/// Separable 2-D convolution of a single-channel plane with replicate-boundary
/// padding — the same boundary rule `crate::loss`'s `convolve_separable` uses,
/// so the two SSIM implementations agree pixel for pixel.
fn mr_convolve_separable(image: &[f32], width: usize, height: usize, kernel: &[f32]) -> Vec<f32> {
    let k = kernel.len();
    let half = (k / 2) as isize;
    let w = width as isize;
    let h = height as isize;

    let mut temp = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0_f32;
            for (i, &tap) in kernel.iter().enumerate() {
                let ix = (x as isize + i as isize - half).clamp(0, w - 1) as usize;
                sum += image[y * width + ix] * tap;
            }
            temp[y * width + x] = sum;
        }
    }

    let mut out = vec![0.0_f32; width * height];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0_f32;
            for (i, &tap) in kernel.iter().enumerate() {
                let iy = (y as isize + i as isize - half).clamp(0, h - 1) as usize;
                sum += temp[iy * width + x] * tap;
            }
            out[y * width + x] = sum;
        }
    }
    out
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

    let pred_mag = mr_sobel_magnitude(pred, width, height, channels)?;
    let gt_mag = mr_sobel_magnitude(gt, width, height, channels)?;
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
///
/// # Errors
/// [`MultiResLossError::SizeMismatch`] if `img.len() != width * height * channels`.
pub fn mr_upsample(
    img: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    target_w: usize,
    target_h: usize,
) -> Result<Vec<f32>, MultiResLossError> {
    if width == 0 || height == 0 || target_w == 0 || target_h == 0 {
        return Ok(vec![0.0_f32; target_w * target_h * channels]);
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

    Ok(out)
}

/// Apply 3×3 Gaussian blur with kernel \[1,2,1\]/4 (separable).
///
/// Border pixels are handled by clamping coordinates.
///
/// # Errors
/// [`MultiResLossError::SizeMismatch`] if `img.len() != width * height * channels`
/// (for non-empty `img` with non-zero `width`/`height`).
pub fn mr_gaussian_blur_3x3(
    img: &[f32],
    width: usize,
    height: usize,
    channels: usize,
) -> Result<Vec<f32>, MultiResLossError> {
    if img.is_empty() || width == 0 || height == 0 {
        return Ok(img.to_vec());
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

    Ok(out)
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
    let gauss0 = mr_gaussian_blur_3x3(img, width, height, channels)?;

    if width >= 2 && height >= 2 {
        // lap[0] = img - upsample(downsample(gauss[0]))
        let (down0, dw0, dh0) = mr_downsample(&gauss0, width, height, channels)?;
        let up0 = mr_upsample(&down0, dw0, dh0, channels, width, height)?;
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
        let up = mr_upsample(&next_gauss, next_w, next_h, channels, cur_w, cur_h)?;
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
///
/// # Errors
/// [`MultiResLossError::SizeMismatch`] if `img.len() != width * height * channels`
/// (checked whenever `width * height != 0 && channels != 0`; a genuinely
/// empty request returns an empty `Vec` rather than an error).
pub fn mr_sobel_magnitude(
    img: &[f32],
    width: usize,
    height: usize,
    channels: usize,
) -> Result<Vec<f32>, MultiResLossError> {
    let n_pixels = width * height;
    if n_pixels == 0 || channels == 0 {
        return Ok(Vec::new());
    }
    let expected = n_pixels * channels;
    if img.len() != expected {
        return Err(MultiResLossError::SizeMismatch {
            got: img.len(),
            width,
            height,
            channels,
        });
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

    Ok(out)
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
mod tests;
