//! Loss functions for Gaussian avatar optimisation.
//!
//! * **L1 photometric** — mean absolute pixel error.
//! * **SSIM** — 11×11 Gaussian-windowed structural similarity (1 − SSIM).
//! * **MS-SSIM** — multi-scale structural similarity (5 scales).
//! * **LPIPS** — learned perceptual image patch similarity (VGG features).
//! * **Position regularisation** — penalise large local-offset magnitudes.
//! * **Scale regularisation** — penalise extreme Gaussian scales.
//! * **Opacity regularisation** — encourage binary (0 / 1) opacity values.
//! * **Normal consistency** — align Gaussian orientation with mesh surface
//!   normals, weighted by view angle.
//! * **Gradient penalty** — penalise large gradients for training stability.

use ndarray::Array2;
use oxigaf_flame::Mesh;
use oxigaf_render::gaussian::GaussianModel;
use std::path::Path;

use crate::config::LossConfig;
use crate::lpips::{lpips_loss, LpipsDistance};
use crate::TrainerError;

// ---------------------------------------------------------------------------
// LossOutput
// ---------------------------------------------------------------------------

/// Itemised output of a single loss evaluation.
#[derive(Debug, Clone)]
pub struct LossOutput {
    pub l1: f32,
    pub ssim: f32,
    pub ms_ssim: f32,
    pub lpips: f32,
    /// Score Distillation Sampling loss (from diffusion model).
    /// This is computed separately by the trainer and set here for logging.
    pub sds: f32,
    pub position_reg: f32,
    pub scale_reg: f32,
    pub opacity_reg: f32,
    pub normal: f32,
    pub gradient_penalty: f32,
    /// Weighted sum of all photometric and regularization terms.
    /// Note: SDS loss is tracked separately in StepOutput.sds_loss.
    pub total: f32,
}

// ---------------------------------------------------------------------------
// LpipsLossComputer
// ---------------------------------------------------------------------------

/// LPIPS perceptual loss computer with lazy loading.
///
/// This wrapper loads VGG weights only when `init()` is called, enabling
/// lazy initialization when LPIPS weight is > 0.
pub struct LpipsLossComputer {
    lpips: Option<LpipsDistance>,
}

impl LpipsLossComputer {
    /// Create a new LPIPS computer (not yet initialized).
    pub const fn new() -> Self {
        Self { lpips: None }
    }

    /// Initialize with VGG weights from safetensors.
    ///
    /// # Arguments
    /// * `vgg_weights_path` - Path to VGG16 weights in safetensors format.
    /// * `lpips_weights_path` - Path to LPIPS linear weights in safetensors format.
    ///
    /// For uniform (non-learned) weights, use `init_uniform` instead.
    pub fn init(
        &mut self,
        vgg_weights_path: &Path,
        lpips_weights_path: &Path,
    ) -> Result<(), TrainerError> {
        let device = candle_core::Device::Cpu;
        let lpips = LpipsDistance::new(vgg_weights_path, lpips_weights_path, &device)?;
        self.lpips = Some(lpips);
        Ok(())
    }

    /// Initialize with VGG weights and uniform LPIPS weights.
    ///
    /// Uses equal weighting for each VGG layer instead of learned weights.
    pub fn init_uniform(&mut self, vgg_weights_path: &Path) -> Result<(), TrainerError> {
        let device = candle_core::Device::Cpu;
        let lpips = LpipsDistance::with_uniform_weights(vgg_weights_path, &device)?;
        self.lpips = Some(lpips);
        Ok(())
    }

    /// Check if LPIPS is initialized.
    pub fn is_initialized(&self) -> bool {
        self.lpips.is_some()
    }

    /// Compute LPIPS loss for a pair of images.
    ///
    /// Returns `Ok(0.0)` if LPIPS is not initialized.
    pub fn compute(
        &self,
        pred: &[f32],
        target: &[f32],
        width: usize,
        height: usize,
    ) -> Result<f32, TrainerError> {
        match &self.lpips {
            Some(lpips) => lpips_loss(pred, target, width, height, lpips),
            None => Ok(0.0),
        }
    }

    /// Compute LPIPS loss averaged over multiple image pairs.
    pub fn compute_multi(
        &self,
        rendered: &[Vec<f32>],
        targets: &[Vec<f32>],
        width: usize,
        height: usize,
    ) -> Result<f32, TrainerError> {
        if !self.is_initialized() {
            return Ok(0.0);
        }

        let num_pairs = rendered.len().min(targets.len());
        if num_pairs == 0 {
            return Ok(0.0);
        }

        let mut sum = 0.0_f32;
        for (r, t) in rendered.iter().zip(targets.iter()) {
            sum += self.compute(r, t, width, height)?;
        }

        Ok(sum / num_pairs as f32)
    }
}

impl Default for LpipsLossComputer {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LpipsLossComputer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LpipsLossComputer")
            .field("initialized", &self.lpips.is_some())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// LossComputer
// ---------------------------------------------------------------------------

/// Aggregates individual loss terms using configured weights.
#[derive(Debug, Clone)]
pub struct LossComputer {
    config: LossConfig,
    /// Pre-computed 1-D Gaussian kernel (11 taps, σ = 1.5).
    ssim_kernel: Vec<f32>,
    /// MS-SSIM scale weights (5 scales).
    ms_ssim_weights: [f32; 5],
}

impl LossComputer {
    /// Default MS-SSIM weights from the original paper.
    pub const DEFAULT_MS_SSIM_WEIGHTS: [f32; 5] = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];

    pub fn new(config: LossConfig) -> Self {
        let ssim_kernel = gaussian_kernel_1d(11, 1.5);
        Self {
            config,
            ssim_kernel,
            ms_ssim_weights: Self::DEFAULT_MS_SSIM_WEIGHTS,
        }
    }

    /// Create with custom MS-SSIM weights.
    pub fn with_ms_ssim_weights(config: LossConfig, ms_ssim_weights: [f32; 5]) -> Self {
        let ssim_kernel = gaussian_kernel_1d(11, 1.5);
        Self {
            config,
            ssim_kernel,
            ms_ssim_weights,
        }
    }

    /// Evaluate all loss terms.
    ///
    /// * `rendered` / `targets` — lists of images (one per view), each a flat
    ///   `Vec<f32>` in **HWC** layout with values in `[0, 1]`.
    /// * `model` — current Gaussian model (for regularisation losses).
    /// * `mesh` — optional FLAME mesh (needed for normal consistency).
    /// * `view_directions` — optional camera view directions for view-weighted normal loss.
    /// * `gradients` — optional gradient buffer for gradient penalty computation.
    pub fn compute(
        &self,
        rendered: &[Vec<f32>],
        targets: &[Vec<f32>],
        width: usize,
        height: usize,
        model: &GaussianModel,
        mesh: Option<&Mesh>,
    ) -> LossOutput {
        self.compute_with_options(rendered, targets, width, height, model, mesh, None, None)
    }

    /// Full loss computation with all optional parameters.
    ///
    /// # Arguments
    /// * `rendered` / `targets` — image pairs per view.
    /// * `model` — Gaussian model.
    /// * `mesh` — optional FLAME mesh.
    /// * `view_directions` — camera view directions for view-weighted normal loss.
    /// * `gradients` — gradient buffer for gradient penalty computation.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_with_options(
        &self,
        rendered: &[Vec<f32>],
        targets: &[Vec<f32>],
        width: usize,
        height: usize,
        model: &GaussianModel,
        mesh: Option<&Mesh>,
        view_directions: Option<&[[f32; 3]]>,
        gradients: Option<&[f32]>,
    ) -> LossOutput {
        let cfg = &self.config;

        // ---- photometric losses (averaged over views) ----------------------
        let num_views = rendered.len().min(targets.len()).max(1);
        let mut l1_sum = 0.0_f32;
        let mut ssim_sum = 0.0_f32;
        let mut ms_ssim_sum = 0.0_f32;

        for (r, t) in rendered.iter().zip(targets.iter()) {
            l1_sum += l1_loss(r, t);
            ssim_sum += ssim_loss(r, t, width, height, &self.ssim_kernel);

            // MS-SSIM if weight is non-zero
            if cfg.w_ms_ssim > 0.0 {
                ms_ssim_sum += ms_ssim_loss(r, t, width, height, &self.ms_ssim_weights);
            }
        }

        let l1 = l1_sum / num_views as f32;
        let ssim = ssim_sum / num_views as f32;
        let ms_ssim = if cfg.w_ms_ssim > 0.0 {
            ms_ssim_sum / num_views as f32
        } else {
            0.0
        };

        // LPIPS is computed externally (requires neural network)
        let lpips = 0.0; // Will be set by caller if using LPIPS

        // ---- regularisation losses -----------------------------------------
        let pos_reg = position_reg(model);
        let sc_reg = scale_reg(model);
        let op_reg = opacity_reg(model);

        // Normal consistency with optional view weighting
        let nrm = match (mesh, view_directions) {
            (Some(m), Some(views)) => normal_consistency_view_weighted(model, m, views),
            (Some(m), None) => normal_consistency(model, m),
            _ => 0.0,
        };

        // Gradient penalty
        let grad_pen =
            gradients.map_or(0.0, |g| gradient_penalty(g, cfg.gradient_penalty_threshold));

        let total = cfg.w_l1 * l1
            + cfg.w_ssim * ssim
            + cfg.w_ms_ssim * ms_ssim
            + cfg.w_lpips * lpips
            + cfg.w_position_reg * pos_reg
            + cfg.w_scale_reg * sc_reg
            + cfg.w_opacity_reg * op_reg
            + cfg.w_normal * nrm
            + cfg.w_gradient_penalty * grad_pen;

        LossOutput {
            l1,
            ssim,
            ms_ssim,
            lpips,
            sds: 0.0, // SDS loss is computed by trainer, not here
            position_reg: pos_reg,
            scale_reg: sc_reg,
            opacity_reg: op_reg,
            normal: nrm,
            gradient_penalty: grad_pen,
            total,
        }
    }

    /// Compute loss with externally computed LPIPS value.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_with_lpips(
        &self,
        rendered: &[Vec<f32>],
        targets: &[Vec<f32>],
        width: usize,
        height: usize,
        model: &GaussianModel,
        mesh: Option<&Mesh>,
        lpips_value: f32,
    ) -> LossOutput {
        let mut output = self.compute(rendered, targets, width, height, model, mesh);
        output.lpips = lpips_value;
        output.total += self.config.w_lpips * lpips_value;
        output
    }
}

// ===========================================================================
// Individual loss functions
// ===========================================================================

/// Mean absolute error between two flat f32 images.
pub fn l1_loss(pred: &[f32], target: &[f32]) -> f32 {
    if pred.is_empty() {
        return 0.0;
    }
    let sum: f32 = pred
        .iter()
        .zip(target.iter())
        .map(|(p, t)| (p - t).abs())
        .sum();
    sum / pred.len() as f32
}

/// Structural dissimilarity (1 − SSIM) using an 11×11 Gaussian window.
///
/// `kernel` is the pre-computed 1-D Gaussian kernel (separable).
pub fn ssim_loss(pred: &[f32], target: &[f32], width: usize, height: usize, kernel: &[f32]) -> f32 {
    if pred.len() < width * height * 3 || target.len() < width * height * 3 {
        return 0.0;
    }

    let mut ssim_total = 0.0_f32;

    for c in 0..3 {
        let pred_ch =
            Array2::from_shape_fn((height, width), |(y, x)| pred[(y * width + x) * 3 + c]);
        let tgt_ch =
            Array2::from_shape_fn((height, width), |(y, x)| target[(y * width + x) * 3 + c]);
        ssim_total += ssim_channel(&pred_ch, &tgt_ch, kernel);
    }

    // Return *dissimilarity*: 1 − mean SSIM.
    1.0 - ssim_total / 3.0
}

// ---------------------------------------------------------------------------
// SSIM internals
// ---------------------------------------------------------------------------

/// SSIM for a single greyscale channel (returns mean SSIM ∈ [−1, 1]).
fn ssim_channel(pred: &Array2<f32>, target: &Array2<f32>, kernel: &[f32]) -> f32 {
    let c1: f32 = (0.01_f32).powi(2); // (K₁ L)²   L = 1
    let c2: f32 = (0.03_f32).powi(2); // (K₂ L)²

    let mu_x = convolve_separable(pred, kernel);
    let mu_y = convolve_separable(target, kernel);

    let mu_x_sq = &mu_x * &mu_x;
    let mu_y_sq = &mu_y * &mu_y;
    let mu_xy = &mu_x * &mu_y;

    let pred_sq = pred * pred;
    let tgt_sq = target * target;
    let pred_tgt = pred * target;

    let sigma_x_sq = convolve_separable(&pred_sq, kernel) - &mu_x_sq;
    let sigma_y_sq = convolve_separable(&tgt_sq, kernel) - &mu_y_sq;
    let sigma_xy = convolve_separable(&pred_tgt, kernel) - &mu_xy;

    let (h, w) = pred.dim();
    let n = (h * w) as f32;
    if n < 1.0 {
        return 0.0;
    }

    let mut ssim_sum = 0.0_f32;
    for y in 0..h {
        for x in 0..w {
            let num = (2.0 * mu_xy[[y, x]] + c1) * (2.0 * sigma_xy[[y, x]] + c2);
            let den = (mu_x_sq[[y, x]] + mu_y_sq[[y, x]] + c1)
                * (sigma_x_sq[[y, x]] + sigma_y_sq[[y, x]] + c2);
            ssim_sum += num / den;
        }
    }

    ssim_sum / n
}

/// 1-D Gaussian kernel (normalised to sum = 1).
pub fn gaussian_kernel_1d(size: usize, sigma: f32) -> Vec<f32> {
    let centre = (size as f32 - 1.0) / 2.0;
    let mut kernel: Vec<f32> = (0..size)
        .map(|i| {
            let x = i as f32 - centre;
            (-x * x / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let sum: f32 = kernel.iter().sum();
    for v in &mut kernel {
        *v /= sum;
    }
    kernel
}

/// Separable 2-D convolution with replicate-boundary padding.
fn convolve_separable(image: &Array2<f32>, kernel: &[f32]) -> Array2<f32> {
    let (h, w) = image.dim();
    let k = kernel.len();
    let half = k / 2;

    // Horizontal pass.
    let mut temp = Array2::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0_f32;
            #[allow(clippy::needless_range_loop)]
            for i in 0..k {
                let ix = (x as isize + i as isize - half as isize)
                    .max(0)
                    .min(w as isize - 1) as usize;
                sum += image[[y, ix]] * kernel[i];
            }
            temp[[y, x]] = sum;
        }
    }

    // Vertical pass.
    let mut out = Array2::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0_f32;
            #[allow(clippy::needless_range_loop)]
            for i in 0..k {
                let iy = (y as isize + i as isize - half as isize)
                    .max(0)
                    .min(h as isize - 1) as usize;
                sum += temp[[iy, x]] * kernel[i];
            }
            out[[y, x]] = sum;
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Multi-Scale SSIM (MS-SSIM)
// ---------------------------------------------------------------------------

/// Multi-Scale Structural Similarity (MS-SSIM) loss.
///
/// Computes SSIM at multiple scales (up to 5) and combines them
/// using the provided weights. Returns 1 - MS-SSIM (dissimilarity).
///
/// The MS-SSIM formula is:
/// `MS-SSIM = l_M^(w_M) * prod_{j=1}^{M} cs_j^(w_j)`
/// where l_M is luminance at the coarsest scale, cs_j is contrast*structure
/// at scale j, and w_j are the weights.
///
/// # Arguments
/// * `pred` - Predicted image, flat f32 in HWC layout.
/// * `target` - Target image, flat f32 in HWC layout.
/// * `width` - Image width.
/// * `height` - Image height.
/// * `weights` - Weights for each scale (5 values, should sum to 1).
///
/// # Returns
/// MS-SSIM dissimilarity (1 - MS-SSIM), in range [0, 2].
pub fn ms_ssim_loss(
    pred: &[f32],
    target: &[f32],
    width: usize,
    height: usize,
    weights: &[f32; 5],
) -> f32 {
    if width < 16 || height < 16 {
        // Image too small for multi-scale
        return 0.0;
    }

    if pred.len() < width * height * 3 || target.len() < width * height * 3 {
        return 0.0;
    }

    // Use a smaller kernel (7 taps) to support more scales with smaller images
    let kernel = gaussian_kernel_1d(7, 1.0);
    let min_dim = 7; // Minimum dimension for convolution

    // First pass: determine how many scales we can compute
    let mut dims = Vec::with_capacity(5);
    let mut w = width;
    let mut h = height;
    for _ in 0..5 {
        if w < min_dim || h < min_dim {
            break;
        }
        dims.push((w, h));
        w /= 2;
        h /= 2;
    }

    let num_scales = dims.len();
    if num_scales == 0 {
        return 0.0;
    }

    // Collect luminance and CS at each scale
    let mut luminances = Vec::with_capacity(num_scales);
    let mut contrast_structures = Vec::with_capacity(num_scales);

    let mut current_pred = pred.to_vec();
    let mut current_target = target.to_vec();
    let mut current_w = width;
    let mut current_h = height;

    for scale_idx in 0..num_scales {
        // Compute SSIM components at this scale
        let (luminance, contrast_structure) = ssim_components(
            &current_pred,
            &current_target,
            current_w,
            current_h,
            &kernel,
        );

        luminances.push(luminance);
        contrast_structures.push(contrast_structure);

        // Downsample by 2x for next scale (if not last)
        if scale_idx < num_scales - 1 {
            let (new_pred, new_w, new_h) = downsample_2x(&current_pred, current_w, current_h);
            let (new_target, _, _) = downsample_2x(&current_target, current_w, current_h);
            current_pred = new_pred;
            current_target = new_target;
            current_w = new_w;
            current_h = new_h;
        }
    }

    // Compute MS-SSIM according to the paper formula:
    // MS-SSIM = l_M^(w_M) * prod_{j=1}^{M} cs_j^(w_j)
    // where M is the coarsest scale (last one we computed)
    let mut ms_ssim_product = 1.0_f32;

    for scale_idx in 0..num_scales {
        let weight = weights[scale_idx];
        let cs = contrast_structures[scale_idx];

        // Apply CS at all scales
        let cs_term = cs.max(0.0).powf(weight);
        ms_ssim_product *= cs_term;
    }

    // Apply luminance at the coarsest (last) scale we computed
    let last_scale = num_scales - 1;
    let lum_weight = weights[last_scale];
    let luminance = luminances[last_scale];
    let lum_term = luminance.max(0.0).powf(lum_weight);
    ms_ssim_product *= lum_term;

    // Return dissimilarity, handling NaN
    let result = 1.0 - ms_ssim_product.clamp(0.0, 1.0);
    if result.is_nan() {
        1.0 // Maximum dissimilarity for invalid cases
    } else {
        result
    }
}

/// Compute SSIM luminance and contrast-structure components separately.
fn ssim_components(
    pred: &[f32],
    target: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32],
) -> (f32, f32) {
    let c1: f32 = (0.01_f32).powi(2);
    let c2: f32 = (0.03_f32).powi(2);

    let mut luminance_sum = 0.0_f32;
    let mut cs_sum = 0.0_f32;
    let mut count = 0_u32;

    for c in 0..3 {
        let pred_ch = Array2::from_shape_fn((height, width), |(y, x)| {
            let idx = (y * width + x) * 3 + c;
            if idx < pred.len() {
                pred[idx]
            } else {
                0.0
            }
        });
        let tgt_ch = Array2::from_shape_fn((height, width), |(y, x)| {
            let idx = (y * width + x) * 3 + c;
            if idx < target.len() {
                target[idx]
            } else {
                0.0
            }
        });

        let mu_x = convolve_separable(&pred_ch, kernel);
        let mu_y = convolve_separable(&tgt_ch, kernel);

        let mu_x_sq = &mu_x * &mu_x;
        let mu_y_sq = &mu_y * &mu_y;
        let mu_xy = &mu_x * &mu_y;

        let pred_sq = &pred_ch * &pred_ch;
        let tgt_sq = &tgt_ch * &tgt_ch;
        let pred_tgt = &pred_ch * &tgt_ch;

        let sigma_x_sq = convolve_separable(&pred_sq, kernel) - &mu_x_sq;
        let sigma_y_sq = convolve_separable(&tgt_sq, kernel) - &mu_y_sq;
        let sigma_xy = convolve_separable(&pred_tgt, kernel) - &mu_xy;

        let (h, w) = pred_ch.dim();
        for y in 0..h {
            for x in 0..w {
                // Luminance component: (2*mu_x*mu_y + C1) / (mu_x^2 + mu_y^2 + C1)
                let l = (2.0 * mu_xy[[y, x]] + c1) / (mu_x_sq[[y, x]] + mu_y_sq[[y, x]] + c1);

                // Contrast-Structure component: (2*sigma_xy + C2) / (sigma_x^2 + sigma_y^2 + C2)
                let cs =
                    (2.0 * sigma_xy[[y, x]] + c2) / (sigma_x_sq[[y, x]] + sigma_y_sq[[y, x]] + c2);

                luminance_sum += l;
                cs_sum += cs;
                count += 1;
            }
        }
    }

    let n = count.max(1) as f32;
    (luminance_sum / n, cs_sum / n)
}

/// Downsample image by 2x using averaging.
fn downsample_2x(image: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
    let new_w = width / 2;
    let new_h = height / 2;
    let channels = 3;

    if new_w == 0 || new_h == 0 {
        return (Vec::new(), 0, 0);
    }

    let mut result = vec![0.0_f32; new_w * new_h * channels];

    for y in 0..new_h {
        for x in 0..new_w {
            for c in 0..channels {
                // Average 2x2 block
                let mut sum = 0.0_f32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let src_y = y * 2 + dy;
                        let src_x = x * 2 + dx;
                        let idx = (src_y * width + src_x) * channels + c;
                        if idx < image.len() {
                            sum += image[idx];
                        }
                    }
                }
                let dst_idx = (y * new_w + x) * channels + c;
                result[dst_idx] = sum / 4.0;
            }
        }
    }

    (result, new_w, new_h)
}

// ---------------------------------------------------------------------------
// Enhanced Normal Consistency (View-Weighted)
// ---------------------------------------------------------------------------

/// View-weighted normal consistency loss.
///
/// Similar to `normal_consistency`, but weights the contribution by the
/// angle between the mesh normal and view direction. Normals facing away
/// from the camera are weighted less.
///
/// # Arguments
/// * `model` - Gaussian model.
/// * `mesh` - FLAME mesh.
/// * `view_directions` - Camera view directions (one per Gaussian, or broadcast).
pub fn normal_consistency_view_weighted(
    model: &GaussianModel,
    mesh: &Mesh,
    view_directions: &[[f32; 3]],
) -> f32 {
    if model.is_empty() || view_directions.is_empty() {
        return 0.0;
    }

    let mut weighted_sum = 0.0_f32;
    let mut weight_total = 0.0_f32;

    for (i, g) in model.gaussians.iter().enumerate() {
        let fi = model.face_indices.get(i).copied().unwrap_or(0) as usize;
        if fi >= mesh.faces.len() {
            continue;
        }

        // Get barycentric coords
        let bary = model.barycentric.get(i).copied().unwrap_or([1.0, 0.0, 0.0]);

        // Interpolated mesh normal
        let face = &mesh.faces[fi];
        let n0_idx = face[0] as usize;
        let n1_idx = face[1] as usize;
        let n2_idx = face[2] as usize;

        if n0_idx >= mesh.normals.len()
            || n1_idx >= mesh.normals.len()
            || n2_idx >= mesh.normals.len()
        {
            continue;
        }

        let n0 = &mesh.normals[n0_idx];
        let n1 = &mesh.normals[n1_idx];
        let n2 = &mesh.normals[n2_idx];
        let mesh_normal = (n0 * bary[0] + n1 * bary[1] + n2 * bary[2]).normalize();

        // Gaussian z-axis from quaternion
        let [qx, qy, qz, qw] = g.rotation;
        let gz = nalgebra::Vector3::new(
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        );

        // Get view direction (broadcast if only one provided)
        let view_idx = i.min(view_directions.len().saturating_sub(1));
        let view_dir = view_directions
            .get(view_idx)
            .copied()
            .unwrap_or([0.0, 0.0, 1.0]);
        let view_vec = nalgebra::Vector3::new(view_dir[0], view_dir[1], view_dir[2]).normalize();

        // Weight by how much the normal faces the camera
        // view_weight = max(0, dot(normal, -view_dir))
        let view_weight = mesh_normal.dot(&(-view_vec)).max(0.0);

        // Normal consistency: 1 - |dot(gz, mesh_normal)|
        let dot = gz.dot(&mesh_normal).abs().min(1.0);
        let consistency_loss = 1.0 - dot;

        weighted_sum += view_weight * consistency_loss;
        weight_total += view_weight;
    }

    if weight_total < 1e-8 {
        0.0
    } else {
        weighted_sum / weight_total
    }
}

// ---------------------------------------------------------------------------
// Gradient Penalty
// ---------------------------------------------------------------------------

/// Gradient penalty for training stability.
///
/// Computes the mean squared gradient norm and returns a penalty if it
/// exceeds the threshold. This helps detect and handle gradient explosions.
///
/// # Arguments
/// * `gradients` - Flat buffer of gradient values.
/// * `threshold` - Norm threshold above which penalty is applied.
///
/// # Returns
/// Gradient penalty value (0 if below threshold).
pub fn gradient_penalty(gradients: &[f32], threshold: f32) -> f32 {
    if gradients.is_empty() {
        return 0.0;
    }

    // Compute L2 norm
    let sum_sq: f32 = gradients.iter().map(|g| g * g).sum();
    let norm = sum_sq.sqrt();

    // Soft penalty: (max(0, norm - threshold))^2
    let excess = (norm - threshold).max(0.0);
    excess * excess
}

/// Compute gradient norm for monitoring.
///
/// # Returns
/// (total_norm, max_abs_gradient)
pub fn gradient_statistics(gradients: &[f32]) -> (f32, f32) {
    if gradients.is_empty() {
        return (0.0, 0.0);
    }

    let sum_sq: f32 = gradients.iter().map(|g| g * g).sum();
    let norm = sum_sq.sqrt();

    let max_abs = gradients
        .iter()
        .map(|g| g.abs())
        .fold(0.0_f32, |a, b| a.max(b));

    (norm, max_abs)
}

/// Clip gradients to a maximum norm (in-place).
///
/// Returns the original norm before clipping.
pub fn clip_gradients_by_norm(gradients: &mut [f32], max_norm: f32) -> f32 {
    if gradients.is_empty() || max_norm <= 0.0 {
        return 0.0;
    }

    let sum_sq: f32 = gradients.iter().map(|g| g * g).sum();
    let norm = sum_sq.sqrt();

    if norm > max_norm {
        let scale = max_norm / norm;
        for g in gradients.iter_mut() {
            *g *= scale;
        }
    }

    norm
}

/// Clip gradients to a maximum value per element (in-place).
///
/// Returns the number of clipped elements.
pub fn clip_gradients_by_value(gradients: &mut [f32], max_value: f32) -> usize {
    if max_value <= 0.0 {
        return 0;
    }

    let mut clipped = 0_usize;
    for g in gradients.iter_mut() {
        if *g > max_value {
            *g = max_value;
            clipped += 1;
        } else if *g < -max_value {
            *g = -max_value;
            clipped += 1;
        }
    }

    clipped
}

/// Detect and replace NaN/Inf gradients with zeros.
///
/// Returns the count of replaced values.
pub fn sanitize_gradients(gradients: &mut [f32]) -> usize {
    let mut replaced = 0_usize;
    for g in gradients.iter_mut() {
        if !g.is_finite() {
            *g = 0.0;
            replaced += 1;
        }
    }
    replaced
}

// ---------------------------------------------------------------------------
// View Consistency Loss
// ---------------------------------------------------------------------------

/// View consistency loss using depth-based warping between neighboring views.
///
/// This loss encourages 3D consistency across multiple viewpoints by:
/// 1. Warping rendered view A to the perspective of view B using depth.
/// 2. Computing photometric error between warped A and actual B.
/// 3. Masking out invalid warped regions (occlusions, out-of-bounds).
///
/// # Arguments
/// * `view_images` - Rendered images from multiple views `[N][H*W*3]`.
/// * `view_depths` - Depth maps for each view `[N][H*W]`.
/// * `view_poses` - Camera poses (4x4 matrices, flattened) for each view.
/// * `intrinsics` - Camera intrinsics (fx, fy, cx, cy).
/// * `width` - Image width.
/// * `height` - Image height.
///
/// # Returns
/// Mean photometric error across all valid warped pixels.
pub fn view_consistency_loss(
    view_images: &[Vec<f32>],
    view_depths: &[Vec<f32>],
    view_poses: &[[f32; 16]],
    intrinsics: &[f32; 4],
    width: usize,
    height: usize,
) -> f32 {
    if view_images.len() < 2 || view_depths.len() < 2 || view_poses.len() < 2 {
        return 0.0;
    }

    let num_views = view_images
        .len()
        .min(view_depths.len())
        .min(view_poses.len());
    let [fx, fy, cx, cy] = *intrinsics;

    let mut total_loss = 0.0_f32;
    let mut num_pairs = 0_usize;

    // Compare each view with its neighbors
    for i in 0..num_views {
        for j in 0..num_views {
            if i == j {
                continue;
            }

            // Warp view i to view j
            let warped_loss = warp_and_compare(
                &view_images[i],
                &view_images[j],
                &view_depths[i],
                &view_poses[i],
                &view_poses[j],
                fx,
                fy,
                cx,
                cy,
                width,
                height,
            );

            if warped_loss >= 0.0 {
                total_loss += warped_loss;
                num_pairs += 1;
            }
        }
    }

    if num_pairs == 0 {
        0.0
    } else {
        total_loss / num_pairs as f32
    }
}

/// Warp source image to target view and compute photometric error.
#[allow(clippy::too_many_arguments)]
fn warp_and_compare(
    source_img: &[f32],
    target_img: &[f32],
    source_depth: &[f32],
    source_pose: &[f32; 16],
    target_pose: &[f32; 16],
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    width: usize,
    height: usize,
) -> f32 {
    if source_img.len() < width * height * 3
        || target_img.len() < width * height * 3
        || source_depth.len() < width * height
    {
        return -1.0; // Invalid
    }

    // Compute relative transformation: T_target^-1 * T_source
    let source_mat = Mat4::from_array(source_pose);
    let target_mat = Mat4::from_array(target_pose);
    let rel_transform = target_mat.try_inverse().unwrap_or(Mat4::identity()) * source_mat;

    let mut error_sum = 0.0_f32;
    let mut valid_count = 0_usize;

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let depth = source_depth[idx];

            // Skip invalid depths
            if depth <= 0.0 || !depth.is_finite() {
                continue;
            }

            // Back-project to 3D in source camera space
            let x_norm = (x as f32 - cx) / fx;
            let y_norm = (y as f32 - cy) / fy;
            let p3d_source = [x_norm * depth, y_norm * depth, depth, 1.0];

            // Transform to target camera space
            let p3d_target = rel_transform.transform_point(&p3d_source);

            // Skip points behind target camera
            if p3d_target[2] <= 0.0 {
                continue;
            }

            // Project to target image
            let u_target = (p3d_target[0] / p3d_target[2]) * fx + cx;
            let v_target = (p3d_target[1] / p3d_target[2]) * fy + cy;

            // Check if in bounds
            let u_int = u_target as isize;
            let v_int = v_target as isize;

            if u_int < 0 || u_int >= width as isize || v_int < 0 || v_int >= height as isize {
                continue;
            }

            let target_idx = (v_int as usize * width + u_int as usize) * 3;

            // Bilinear interpolation would be better, but for simplicity use nearest
            if target_idx + 2 < target_img.len() {
                let source_color_idx = idx * 3;
                if source_color_idx + 2 < source_img.len() {
                    let dr = source_img[source_color_idx] - target_img[target_idx];
                    let dg = source_img[source_color_idx + 1] - target_img[target_idx + 1];
                    let db = source_img[source_color_idx + 2] - target_img[target_idx + 2];

                    error_sum += (dr * dr + dg * dg + db * db).sqrt();
                    valid_count += 1;
                }
            }
        }
    }

    if valid_count == 0 {
        -1.0
    } else {
        error_sum / valid_count as f32
    }
}

/// Simple 4x4 matrix for view warping.
struct Mat4 {
    data: [f32; 16],
}

impl Mat4 {
    fn from_array(arr: &[f32; 16]) -> Self {
        Self { data: *arr }
    }

    fn identity() -> Self {
        Self {
            data: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    /// Simple 4x4 matrix inverse (assumes valid camera matrix).
    fn try_inverse(&self) -> Option<Self> {
        // For camera matrices, we can use the fact that they're typically
        // [R | t] form where R is 3x3 rotation and t is translation.
        // Inverse is [R^T | -R^T * t]

        let m = &self.data;

        // Extract rotation (upper-left 3x3)
        let r00 = m[0];
        let r01 = m[1];
        let r02 = m[2];
        let r10 = m[4];
        let r11 = m[5];
        let r12 = m[6];
        let r20 = m[8];
        let r21 = m[9];
        let r22 = m[10];

        // Extract translation
        let tx = m[3];
        let ty = m[7];
        let tz = m[11];

        // R^T
        let rt00 = r00;
        let rt01 = r10;
        let rt02 = r20;
        let rt10 = r01;
        let rt11 = r11;
        let rt12 = r21;
        let rt20 = r02;
        let rt21 = r12;
        let rt22 = r22;

        // -R^T * t
        let new_tx = -(rt00 * tx + rt01 * ty + rt02 * tz);
        let new_ty = -(rt10 * tx + rt11 * ty + rt12 * tz);
        let new_tz = -(rt20 * tx + rt21 * ty + rt22 * tz);

        Some(Self {
            data: [
                rt00, rt01, rt02, new_tx, rt10, rt11, rt12, new_ty, rt20, rt21, rt22, new_tz, 0.0,
                0.0, 0.0, 1.0,
            ],
        })
    }

    /// Transform a point (homogeneous coordinates).
    fn transform_point(&self, p: &[f32; 4]) -> [f32; 4] {
        let m = &self.data;
        [
            m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3] * p[3],
            m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7] * p[3],
            m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11] * p[3],
            m[12] * p[0] + m[13] * p[1] + m[14] * p[2] + m[15] * p[3],
        ]
    }
}

impl std::ops::Mul for Mat4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let a = &self.data;
        let b = &rhs.data;
        let mut result = [0.0_f32; 16];

        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
                }
            }
        }

        Self { data: result }
    }
}

// ---------------------------------------------------------------------------
// Regularisation losses
// ---------------------------------------------------------------------------

/// Mean squared local-offset magnitude (encourages Gaussians to stay near the
/// mesh surface).
pub fn position_reg(model: &GaussianModel) -> f32 {
    if model.is_empty() {
        return 0.0;
    }
    let sum: f32 = model
        .local_offsets
        .iter()
        .map(|o| o[0] * o[0] + o[1] * o[1] + o[2] * o[2])
        .sum();
    sum / model.len() as f32
}

/// Mean squared scale (in log-space) — penalises extreme scales.
pub fn scale_reg(model: &GaussianModel) -> f32 {
    if model.is_empty() {
        return 0.0;
    }
    let sum: f32 = model
        .gaussians
        .iter()
        .map(|g| g.scale[0] * g.scale[0] + g.scale[1] * g.scale[1] + g.scale[2] * g.scale[2])
        .sum();
    sum / (model.len() as f32 * 3.0)
}

/// Opacity regularisation: penalises intermediate opacities to encourage
/// binary values (sigmoid → 0 or 1).
///
/// Uses `−(σ log σ + (1−σ) log (1−σ))` (binary entropy) averaged over all
/// Gaussians.
pub fn opacity_reg(model: &GaussianModel) -> f32 {
    if model.is_empty() {
        return 0.0;
    }
    let sum: f32 = model
        .gaussians
        .iter()
        .map(|g| {
            let s = sigmoid(g.opacity);
            let s = s.clamp(1e-6, 1.0 - 1e-6);
            -(s * s.ln() + (1.0 - s) * (1.0 - s).ln())
        })
        .sum();
    sum / model.len() as f32
}

/// Normal consistency: mean (1 − dot(Gaussian z-axis, mesh normal)).
///
/// The Gaussian's local z-axis is derived from its rotation quaternion.
pub fn normal_consistency(model: &GaussianModel, mesh: &Mesh) -> f32 {
    if model.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0_f32;
    let mut count = 0_u32;

    for (i, g) in model.gaussians.iter().enumerate() {
        let fi = model.face_indices[i] as usize;
        if fi >= mesh.faces.len() {
            continue;
        }

        // Interpolated mesh normal at binding point.
        let face = &mesh.faces[fi];
        let bary = &model.barycentric[i];
        let n0 = &mesh.normals[face[0] as usize];
        let n1 = &mesh.normals[face[1] as usize];
        let n2 = &mesh.normals[face[2] as usize];
        let mesh_normal = (n0 * bary[0] + n1 * bary[1] + n2 * bary[2]).normalize();

        // Gaussian z-axis from quaternion (x,y,z,w).
        let [qx, qy, qz, qw] = g.rotation;
        let gz = nalgebra::Vector3::new(
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        );

        let dot = gz.dot(&mesh_normal).abs().min(1.0);
        sum += 1.0 - dot;
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l1_identical_is_zero() {
        let a = vec![0.5; 30];
        assert!((l1_loss(&a, &a)).abs() < 1e-7);
    }

    #[test]
    fn ssim_identical_is_one() {
        let k = gaussian_kernel_1d(11, 1.5);
        let img = vec![0.5_f32; 32 * 32 * 3];
        let val = ssim_loss(&img, &img, 32, 32, &k);
        // dissimilarity should be ~0 for identical images
        assert!(val < 0.01, "expected ~0, got {val}");
    }

    #[test]
    fn gaussian_kernel_sums_to_one() {
        let k = gaussian_kernel_1d(11, 1.5);
        let sum: f32 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
