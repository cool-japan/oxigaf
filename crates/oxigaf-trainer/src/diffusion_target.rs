//! Diffusion target generation for iterative denoising distillation.
//!
//! This module provides the core infrastructure for generating pseudo ground-truth
//! images from diffusion models during Gaussian avatar training. The main approach
//! is Score Distillation Sampling (SDS), which uses a pre-trained diffusion model
//! to guide the 3D Gaussian optimization.
//!
//! Key components:
//! - [`DiffusionTargetGenerator`] — orchestrates pseudo-GT generation
//! - [`SdsLoss`] — Score Distillation Sampling loss computation
//! - [`ViewConsistencyLoss`] — ensures consistency across generated views

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use nalgebra as na;

use oxigaf_diffusion::{DiffusionConfig, DiffusionError, MultiViewDiffusionPipeline};
use oxigaf_flame::Camera;

use crate::TrainerError;

// ---------------------------------------------------------------------------
// DiffusionTargetConfig
// ---------------------------------------------------------------------------

/// Configuration for the diffusion target generator.
#[derive(Debug, Clone)]
pub struct DiffusionTargetConfig {
    /// Number of inference steps for diffusion denoising.
    pub num_inference_steps: usize,
    /// Classifier-free guidance scale.
    pub guidance_scale: f32,
    /// Weight for view consistency loss.
    pub view_consistency_weight: f32,
    /// Number of warmup iterations without diffusion.
    pub warmup_iterations: u32,
    /// Initial timestep for noise (annealed down during training).
    pub timestep_start: u32,
    /// Final timestep for noise.
    pub timestep_end: u32,
    /// Annealing steps for timestep.
    pub timestep_anneal_steps: u32,
    /// Weight for SDS loss vs photometric loss.
    pub sds_weight: f32,
    /// Enable view warping for consistency.
    pub enable_view_warping: bool,
}

impl Default for DiffusionTargetConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 50,
            guidance_scale: 3.0,
            view_consistency_weight: 0.1,
            warmup_iterations: 1000,
            timestep_start: 1000,
            timestep_end: 50,
            timestep_anneal_steps: 10_000,
            sds_weight: 0.5,
            enable_view_warping: true,
        }
    }
}

impl DiffusionTargetConfig {
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), TrainerError> {
        if self.num_inference_steps == 0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "num_inference_steps".into(),
                value: "0".into(),
                expected: "> 0".into(),
            });
        }
        if !self.guidance_scale.is_finite() || self.guidance_scale <= 0.0 {
            return Err(TrainerError::ParameterOutOfRange {
                param: "guidance_scale".into(),
                value: format!("{}", self.guidance_scale),
                expected: "> 0 and finite".into(),
            });
        }
        if self.timestep_start <= self.timestep_end {
            return Err(TrainerError::InvalidConfig(format!(
                "timestep_start ({}) must be > timestep_end ({})",
                self.timestep_start, self.timestep_end
            )));
        }
        Ok(())
    }

    /// Get the current timestep based on training iteration.
    pub fn current_timestep(&self, iteration: u32) -> u32 {
        if iteration < self.warmup_iterations {
            return self.timestep_start;
        }
        let adjusted_iter = iteration - self.warmup_iterations;
        let t = (adjusted_iter as f32) / (self.timestep_anneal_steps as f32).max(1.0);
        let t = t.min(1.0);

        let start = self.timestep_start as f32;
        let end = self.timestep_end as f32;
        ((1.0 - t) * start + t * end).round() as u32
    }
}

// ---------------------------------------------------------------------------
// DiffusionTargetGenerator
// ---------------------------------------------------------------------------

/// Generates pseudo ground-truth images using diffusion models.
///
/// During training, this generator:
/// 1. Takes current rendered views from the Gaussian model
/// 2. Adds noise at the current timestep
/// 3. Runs diffusion denoising to produce refined targets
/// 4. Returns these as pseudo-GT for loss computation
pub struct DiffusionTargetGenerator {
    /// Optional diffusion pipeline (loaded lazily).
    pipeline: Option<MultiViewDiffusionPipeline>,
    /// Diffusion configuration.
    diff_config: DiffusionConfig,
    /// Target generation configuration.
    target_config: DiffusionTargetConfig,
    /// Candle device for tensor operations.
    device: Device,
    /// Whether the pipeline is fully loaded.
    is_loaded: bool,
}

impl DiffusionTargetGenerator {
    /// Create a new generator with default CPU device.
    pub fn new(target_config: DiffusionTargetConfig) -> Self {
        Self {
            pipeline: None,
            diff_config: DiffusionConfig::default(),
            target_config,
            device: Device::Cpu,
            is_loaded: false,
        }
    }

    /// Create a generator with a specific device.
    pub fn with_device(target_config: DiffusionTargetConfig, device: Device) -> Self {
        Self {
            pipeline: None,
            diff_config: DiffusionConfig::default(),
            target_config,
            device,
            is_loaded: false,
        }
    }

    /// Load the diffusion pipeline from weights directory.
    pub fn load_pipeline(&mut self, weights_dir: &Path) -> Result<(), TrainerError> {
        let pipeline =
            MultiViewDiffusionPipeline::load(self.diff_config.clone(), weights_dir, &self.device)?;
        self.pipeline = Some(pipeline);
        self.is_loaded = true;
        tracing::info!("Diffusion pipeline loaded from {:?}", weights_dir);
        Ok(())
    }

    /// Check if the diffusion pipeline is loaded.
    pub fn is_loaded(&self) -> bool {
        self.is_loaded
    }

    /// Check if we're in the warmup period (no diffusion yet).
    pub fn is_warmup(&self, iteration: u32) -> bool {
        iteration < self.target_config.warmup_iterations
    }

    /// Get the SDS weight based on iteration.
    ///
    /// During warmup, returns 0.0 (no SDS).
    /// After warmup, ramps up to the configured weight.
    pub fn sds_weight(&self, iteration: u32) -> f32 {
        if iteration < self.target_config.warmup_iterations {
            return 0.0;
        }

        // Ramp up SDS weight over 500 iterations after warmup
        let ramp_steps = 500.0_f32;
        let adjusted = (iteration - self.target_config.warmup_iterations) as f32;
        let factor = (adjusted / ramp_steps).min(1.0);

        self.target_config.sds_weight * factor
    }

    /// Generate multi-view pseudo ground-truth targets.
    ///
    /// If the pipeline is not loaded or we're in warmup, returns the rendered
    /// images as-is (self-supervised mode).
    ///
    /// Otherwise:
    /// 1. Encodes rendered images to latent space
    /// 2. Adds noise at the current timestep
    /// 3. Runs diffusion denoising
    /// 4. Returns denoised images as pseudo-GT
    pub fn generate_targets(
        &mut self,
        rendered: &[Vec<f32>],
        cameras: &[Camera],
        iteration: u32,
        width: u32,
        height: u32,
    ) -> Result<Vec<Vec<f32>>, TrainerError> {
        // During warmup or if pipeline not loaded, return rendered as targets
        if self.is_warmup(iteration) || !self.is_loaded {
            tracing::trace!(
                "generate_targets: iteration {} (warmup={}, loaded={}), returning rendered",
                iteration,
                self.is_warmup(iteration),
                self.is_loaded
            );
            return Ok(rendered.to_vec());
        }

        let pipeline = self
            .pipeline
            .as_mut()
            .ok_or(TrainerError::DiffusionNotLoaded)?;

        if cameras.is_empty() || rendered.is_empty() {
            return Ok(Vec::new());
        }

        let num_views = rendered.len().min(cameras.len());
        let timestep = self.target_config.current_timestep(iteration);

        // Convert rendered images to tensor format
        // Each image is [H*W*3] HWC format, convert to [V, 3, H, W] NCHW
        let rendered_tensor = images_to_tensor(rendered, width, height, &self.device)?;

        // Normalize to [-1, 1] for diffusion
        let rendered_norm = ((&rendered_tensor * 2.0)
            .map_err(|e| DiffusionError::Inference(format!("scale: {e}")))?
            - 1.0)
            .map_err(|e| DiffusionError::Inference(format!("shift: {e}")))?;

        // Create camera pose tensor
        let camera_poses = cameras_to_tensor(cameras, &self.device)?;

        // Generate reference image for CLIP (use first camera)
        // Resize to 224x224 for CLIP
        let ref_image = prepare_reference_image(&rendered_norm, &self.device)?;

        // Create normal map placeholder (zeros for now - could be improved)
        let normal_latents = Tensor::zeros(
            (
                num_views,
                self.diff_config.latent_channels,
                self.diff_config.latent_size,
                self.diff_config.latent_size,
            ),
            DType::F32,
            &self.device,
        )
        .map_err(|e| DiffusionError::Inference(format!("normal latents: {e}")))?;

        // Run diffusion
        let output =
            pipeline.generate(&ref_image, &normal_latents, &camera_poses, iteration as u64)?;

        // Convert output back to Vec<Vec<f32>> in HWC format
        let mut targets = Vec::with_capacity(num_views);
        for img_tensor in &output.images {
            let hwc = tensor_to_hwc_image(img_tensor, output.width, output.height)?;
            targets.push(hwc);
        }

        tracing::trace!(
            "generate_targets: iteration {}, timestep {}, generated {} views",
            iteration,
            timestep,
            targets.len()
        );

        Ok(targets)
    }

    /// Compute the SDS (Score Distillation Sampling) gradient.
    ///
    /// SDS gradient: w(t) * (epsilon_pred - epsilon)
    ///
    /// This is used to compute the loss gradient that pushes the rendered image
    /// toward the diffusion model's prior.
    pub fn compute_sds_gradient(
        &self,
        rendered: &[f32],
        target: &[f32],
        iteration: u32,
    ) -> Vec<f32> {
        let timestep = self.target_config.current_timestep(iteration);
        let weight = sds_timestep_weight(timestep, 1000);
        let sds_w = self.sds_weight(iteration);

        // SDS gradient = w(t) * sds_weight * (rendered - target)
        rendered
            .iter()
            .zip(target.iter())
            .map(|(r, t)| weight * sds_w * (r - t))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// SDS Loss
// ---------------------------------------------------------------------------

/// Score Distillation Sampling loss computation.
///
/// SDS uses the diffusion model's score (gradient of log-density) to guide
/// 3D optimization. The loss is computed as:
///
/// L_SDS = w(t) * ||epsilon_pred - epsilon||^2
///
/// where:
/// - w(t) is a timestep-dependent weighting
/// - epsilon_pred is the noise predicted by the diffusion model
/// - epsilon is the actual noise added
#[derive(Debug, Clone)]
pub struct SdsLoss {
    /// Weighting function type.
    pub weighting: SdsWeighting,
    /// Maximum timestep for normalization.
    pub max_timestep: u32,
}

/// SDS weighting function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdsWeighting {
    /// Uniform weighting across all timesteps.
    Uniform,
    /// Linear decrease: w(t) = t / T
    Linear,
    /// Quadratic decrease: w(t) = (t / T)^2
    Quadratic,
    /// Sigma-based: w(t) = sigma(t)^2
    SigmaBased,
}

impl Default for SdsLoss {
    fn default() -> Self {
        Self {
            weighting: SdsWeighting::SigmaBased,
            max_timestep: 1000,
        }
    }
}

impl SdsLoss {
    /// Compute the SDS loss for a batch of images.
    ///
    /// - `rendered`: rendered images [N views, H*W*3]
    /// - `noise_pred`: predicted noise from diffusion model [N views, H*W*3]
    /// - `noise`: actual added noise [N views, H*W*3]
    /// - `timestep`: current diffusion timestep
    pub fn compute(&self, rendered: &[Vec<f32>], targets: &[Vec<f32>], timestep: u32) -> f32 {
        if rendered.is_empty() || targets.is_empty() {
            return 0.0;
        }

        let weight = self.weight(timestep);
        let num_views = rendered.len().min(targets.len());

        let mut loss_sum = 0.0_f32;
        for v in 0..num_views {
            let view_loss: f32 = rendered[v]
                .iter()
                .zip(targets[v].iter())
                .map(|(r, t)| {
                    let diff = r - t;
                    diff * diff
                })
                .sum();
            loss_sum += view_loss / rendered[v].len() as f32;
        }

        weight * loss_sum / num_views as f32
    }

    /// Get the weighting factor for a timestep.
    fn weight(&self, timestep: u32) -> f32 {
        let t_norm = (timestep as f32) / (self.max_timestep as f32).max(1.0);

        match self.weighting {
            SdsWeighting::Uniform => 1.0,
            SdsWeighting::Linear => t_norm,
            SdsWeighting::Quadratic => t_norm * t_norm,
            SdsWeighting::SigmaBased => {
                // Approximate sigma^2 weighting based on DDPM schedule
                let alpha_t = ddpm_alpha_cumprod(timestep, self.max_timestep);
                1.0 - alpha_t
            }
        }
    }
}

// ---------------------------------------------------------------------------
// View Consistency Loss
// ---------------------------------------------------------------------------

/// View consistency loss ensures multi-view coherence.
///
/// This loss penalizes differences between reprojected views, ensuring that
/// the generated targets are geometrically consistent.
#[derive(Debug, Clone)]
pub struct ViewConsistencyLoss {
    /// Weight for the consistency loss.
    pub weight: f32,
}

impl Default for ViewConsistencyLoss {
    fn default() -> Self {
        Self { weight: 0.1 }
    }
}

impl ViewConsistencyLoss {
    /// Compute view consistency loss across multiple views.
    ///
    /// For each pair of views, we:
    /// 1. Warp one view to the other using depth (if available)
    /// 2. Compute the photometric difference in overlapping regions
    pub fn compute(
        &self,
        views: &[Vec<f32>],
        cameras: &[Camera],
        depth_maps: Option<&[Vec<f32>]>,
        width: usize,
        height: usize,
    ) -> f32 {
        if views.len() < 2 || cameras.len() < 2 {
            return 0.0;
        }

        let num_views = views.len().min(cameras.len());
        let mut total_loss = 0.0_f32;
        let mut pair_count = 0_u32;

        // Compute pairwise consistency
        for i in 0..num_views {
            for j in (i + 1)..num_views {
                let loss = if let Some(depths) = depth_maps {
                    // Use depth-based warping if available
                    self.warped_consistency(
                        &views[i],
                        &cameras[i],
                        &views[j],
                        &cameras[j],
                        &depths[i],
                        width,
                        height,
                    )
                } else {
                    // Fall back to simple appearance consistency
                    self.appearance_consistency(&views[i], &views[j])
                };
                total_loss += loss;
                pair_count += 1;
            }
        }

        if pair_count == 0 {
            0.0
        } else {
            self.weight * total_loss / pair_count as f32
        }
    }

    /// Simple appearance consistency (no depth).
    fn appearance_consistency(&self, view1: &[f32], view2: &[f32]) -> f32 {
        // Compute normalized cross-correlation or simple L1
        if view1.len() != view2.len() || view1.is_empty() {
            return 0.0;
        }

        // Simple L1 as a baseline
        let sum: f32 = view1
            .iter()
            .zip(view2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        sum / view1.len() as f32
    }

    /// Depth-based warping consistency.
    #[allow(clippy::too_many_arguments)]
    fn warped_consistency(
        &self,
        src_view: &[f32],
        src_cam: &Camera,
        tgt_view: &[f32],
        tgt_cam: &Camera,
        src_depth: &[f32],
        width: usize,
        height: usize,
    ) -> f32 {
        if src_view.len() < width * height * 3 || tgt_view.len() < width * height * 3 {
            return 0.0;
        }

        // Warp source view to target view using depth
        let warped = warp_view(src_view, src_cam, tgt_cam, src_depth, width, height);

        // Compute loss only for valid (visible) pixels
        let mut loss_sum = 0.0_f32;
        let mut valid_count = 0_u32;

        for i in 0..(width * height) {
            // Check if pixel is valid (not at boundary/invalid)
            let warped_idx = i * 3;
            if warped_idx + 2 < warped.len() {
                let diff_r = (warped[warped_idx] - tgt_view[warped_idx]).abs();
                let diff_g = (warped[warped_idx + 1] - tgt_view[warped_idx + 1]).abs();
                let diff_b = (warped[warped_idx + 2] - tgt_view[warped_idx + 2]).abs();

                if diff_r.is_finite() && diff_g.is_finite() && diff_b.is_finite() {
                    loss_sum += diff_r + diff_g + diff_b;
                    valid_count += 1;
                }
            }
        }

        if valid_count == 0 {
            0.0
        } else {
            loss_sum / (valid_count * 3) as f32
        }
    }
}

// ---------------------------------------------------------------------------
// Temporal Consistency (for video/animation)
// ---------------------------------------------------------------------------

/// Temporal consistency for animated avatars.
///
/// Ensures smooth transitions between frames.
#[derive(Debug, Clone)]
pub struct TemporalConsistency {
    /// Weight for temporal loss.
    pub weight: f32,
    /// Buffer size for temporal history.
    pub buffer_size: usize,
}

impl Default for TemporalConsistency {
    fn default() -> Self {
        Self {
            weight: 0.05,
            buffer_size: 3,
        }
    }
}

impl TemporalConsistency {
    /// Compute temporal consistency loss.
    ///
    /// - `current`: current frame
    /// - `previous`: previous frame(s)
    /// - `optical_flow`: estimated optical flow between frames (optional)
    pub fn compute(
        &self,
        current: &[f32],
        previous: &[&[f32]],
        _optical_flow: Option<&[f32]>,
    ) -> f32 {
        if previous.is_empty() || current.is_empty() {
            return 0.0;
        }

        // Simple temporal smoothness: penalize differences from previous frame
        let prev = previous.last().copied().unwrap_or(&[]);
        if prev.len() != current.len() {
            return 0.0;
        }

        let diff: f32 = current
            .iter()
            .zip(prev.iter())
            .map(|(c, p)| (c - p).powi(2))
            .sum();

        self.weight * diff / current.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Convert HWC images to a batched tensor [N, C, H, W].
fn images_to_tensor(
    images: &[Vec<f32>],
    width: u32,
    height: u32,
    device: &Device,
) -> Result<Tensor, DiffusionError> {
    let n = images.len();
    let h = height as usize;
    let w = width as usize;

    // Create NCHW tensor
    let mut data = vec![0.0_f32; n * 3 * h * w];

    for (idx, img) in images.iter().enumerate() {
        for y in 0..h {
            for x in 0..w {
                let hwc_idx = (y * w + x) * 3;
                let r = img.get(hwc_idx).copied().unwrap_or(0.0);
                let g = img.get(hwc_idx + 1).copied().unwrap_or(0.0);
                let b = img.get(hwc_idx + 2).copied().unwrap_or(0.0);

                let base = idx * 3 * h * w;
                let channel_stride = h * w;
                data[base + y * w + x] = r;
                data[base + channel_stride + y * w + x] = g;
                data[base + 2 * channel_stride + y * w + x] = b;
            }
        }
    }

    Tensor::from_vec(data, (n, 3, h, w), device)
        .map_err(|e| DiffusionError::Inference(format!("images_to_tensor: {e}")))
}

/// Convert a tensor [C, H, W] to HWC Vec<f32>.
fn tensor_to_hwc_image(
    tensor: &Tensor,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, DiffusionError> {
    let h = height as usize;
    let w = width as usize;

    // Flatten and convert to Vec<f32>
    let data: Vec<f32> = tensor
        .flatten_all()
        .and_then(|t| t.to_vec1())
        .map_err(|e| DiffusionError::Inference(format!("tensor_to_hwc: {e}")))?;

    if data.len() < 3 * h * w {
        return Err(DiffusionError::Inference(format!(
            "tensor_to_hwc: data length {} < expected {}",
            data.len(),
            3 * h * w
        )));
    }

    // CHW to HWC
    let mut hwc = vec![0.0_f32; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let hwc_idx = (y * w + x) * 3;
            let channel_stride = h * w;
            let pixel_offset = y * w + x;
            hwc[hwc_idx] = data.get(pixel_offset).copied().unwrap_or(0.0);
            hwc[hwc_idx + 1] = data
                .get(channel_stride + pixel_offset)
                .copied()
                .unwrap_or(0.0);
            hwc[hwc_idx + 2] = data
                .get(2 * channel_stride + pixel_offset)
                .copied()
                .unwrap_or(0.0);
        }
    }

    Ok(hwc)
}

/// Convert cameras to a batched pose tensor [N, 12] (flattened 4x3 extrinsics).
fn cameras_to_tensor(cameras: &[Camera], device: &Device) -> Result<Tensor, DiffusionError> {
    let n = cameras.len();
    let mut data = vec![0.0_f32; n * 12];

    for (i, cam) in cameras.iter().enumerate() {
        // Flatten rotation (3x3) and translation (3)
        // Row-major: r00, r01, r02, r10, r11, r12, r20, r21, r22, tx, ty, tz
        for r in 0..3 {
            for c in 0..3 {
                data[i * 12 + r * 3 + c] = cam.rotation[(r, c)];
            }
        }
        data[i * 12 + 9] = cam.translation.x;
        data[i * 12 + 10] = cam.translation.y;
        data[i * 12 + 11] = cam.translation.z;
    }

    Tensor::from_vec(data, (n, 12), device)
        .map_err(|e| DiffusionError::Inference(format!("cameras_to_tensor: {e}")))
}

/// Prepare reference image for CLIP (resize to 224x224, first view).
fn prepare_reference_image(images: &Tensor, _device: &Device) -> Result<Tensor, DiffusionError> {
    // Take first image and resize to 224x224
    let first = images
        .narrow(0, 0, 1)
        .map_err(|e| DiffusionError::Inference(format!("narrow: {e}")))?;

    // Simple bilinear resize to 224x224
    // For now, use a simplified approach - actual implementation should use proper resize
    let (_b, c, h, w) = first
        .dims4()
        .map_err(|e| DiffusionError::Inference(format!("dims4: {e}")))?;

    if h == 224 && w == 224 {
        return Ok(first);
    }

    // Use upsample/downsample for resizing
    // This is a simplified version - production code should use proper interpolation
    let target_h = 224;
    let target_w = 224;

    // For now, just average pool or upsample
    if h > target_h && w > target_w {
        // Downsample via adaptive average pool
        // Candle doesn't have direct adaptive pool, so we'll use a workaround
        let scale_h = h / target_h;
        let scale_w = w / target_w;

        // Reshape and average
        let data: Vec<f32> = first
            .flatten_all()
            .and_then(|t| t.to_vec1())
            .map_err(|e| DiffusionError::Inference(format!("flatten: {e}")))?;

        let mut resized = vec![0.0_f32; c * target_h * target_w];
        for ch in 0..c {
            for y in 0..target_h {
                for x in 0..target_w {
                    let mut sum = 0.0_f32;
                    let mut count = 0;
                    for dy in 0..scale_h {
                        for dx in 0..scale_w {
                            let sy = y * scale_h + dy;
                            let sx = x * scale_w + dx;
                            if sy < h && sx < w {
                                if let Some(&val) = data.get(ch * h * w + sy * w + sx) {
                                    sum += val;
                                    count += 1;
                                }
                            }
                        }
                    }
                    resized[ch * target_h * target_w + y * target_w + x] =
                        if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
        }

        Tensor::from_vec(resized, (1, c, target_h, target_w), first.device())
            .map_err(|e| DiffusionError::Inference(format!("from_vec resize: {e}")))
    } else {
        // Upsample
        first
            .upsample_nearest2d(target_h, target_w)
            .map_err(|e| DiffusionError::Inference(format!("upsample: {e}")))
    }
}

/// Warp a source view to a target view using depth.
fn warp_view(
    src_view: &[f32],
    src_cam: &Camera,
    tgt_cam: &Camera,
    src_depth: &[f32],
    width: usize,
    height: usize,
) -> Vec<f32> {
    let mut warped = vec![0.0_f32; width * height * 3];

    // For each pixel in source view
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let depth = src_depth.get(idx).copied().unwrap_or(0.0);

            if depth <= 0.0 || !depth.is_finite() {
                continue;
            }

            // Unproject to 3D
            let px = (x as f32 - src_cam.cx) / src_cam.focal_x;
            let py = (y as f32 - src_cam.cy) / src_cam.focal_y;
            let point_cam = na::Vector3::new(px * depth, py * depth, depth);

            // Transform to world space
            let r_inv = src_cam.rotation.transpose();
            let point_world = r_inv * (point_cam - src_cam.translation);

            // Project to target camera
            let point_tgt_cam = tgt_cam.rotation * point_world + tgt_cam.translation;

            if point_tgt_cam.z <= 0.0 {
                continue;
            }

            let tx = (point_tgt_cam.x / point_tgt_cam.z) * tgt_cam.focal_x + tgt_cam.cx;
            let ty = (point_tgt_cam.y / point_tgt_cam.z) * tgt_cam.focal_y + tgt_cam.cy;

            let tx_i = tx.round() as i32;
            let ty_i = ty.round() as i32;

            if tx_i >= 0 && tx_i < width as i32 && ty_i >= 0 && ty_i < height as i32 {
                let tgt_idx = (ty_i as usize) * width + (tx_i as usize);
                let src_hwc = idx * 3;
                let tgt_hwc = tgt_idx * 3;

                warped[tgt_hwc] = src_view.get(src_hwc).copied().unwrap_or(0.0);
                warped[tgt_hwc + 1] = src_view.get(src_hwc + 1).copied().unwrap_or(0.0);
                warped[tgt_hwc + 2] = src_view.get(src_hwc + 2).copied().unwrap_or(0.0);
            }
        }
    }

    warped
}

/// SDS timestep weighting factor.
///
/// Higher timesteps (more noise) get higher weights.
fn sds_timestep_weight(timestep: u32, max_timestep: u32) -> f32 {
    let alpha = ddpm_alpha_cumprod(timestep, max_timestep);
    let sigma_sq = 1.0 - alpha;

    // w(t) = sigma(t)^2 for variance-preserving weighting
    sigma_sq.max(0.001)
}

/// Approximate DDPM alpha_cumprod for a given timestep.
fn ddpm_alpha_cumprod(timestep: u32, max_timestep: u32) -> f32 {
    // Scaled linear beta schedule (SD 2.1 style)
    let beta_start = 0.00085_f32.sqrt();
    let beta_end = 0.012_f32.sqrt();

    let mut alpha_cumprod = 1.0_f32;
    for t in 0..=timestep {
        let beta = beta_start + (beta_end - beta_start) * (t as f32) / (max_timestep as f32 - 1.0);
        let beta = beta * beta;
        let alpha = 1.0 - beta;
        alpha_cumprod *= alpha;
    }

    alpha_cumprod
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diffusion_config_default() {
        let config = DiffusionTargetConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.warmup_iterations, 1000);
        assert_eq!(config.timestep_start, 1000);
        assert_eq!(config.timestep_end, 50);
    }

    #[test]
    fn test_timestep_annealing() {
        let config = DiffusionTargetConfig {
            warmup_iterations: 100,
            timestep_start: 1000,
            timestep_end: 100,
            timestep_anneal_steps: 1000,
            ..Default::default()
        };

        // During warmup, timestep should be max
        assert_eq!(config.current_timestep(0), 1000);
        assert_eq!(config.current_timestep(50), 1000);
        assert_eq!(config.current_timestep(99), 1000);

        // After warmup, should anneal
        assert_eq!(config.current_timestep(100), 1000); // Just after warmup
        let mid = config.current_timestep(600); // ~500 steps after warmup
        assert!(mid < 1000 && mid > 100, "mid timestep = {}", mid);

        // At end of annealing
        let end = config.current_timestep(1100);
        assert_eq!(end, 100);
    }

    #[test]
    fn test_sds_weight_ramp() {
        let gen = DiffusionTargetGenerator::new(DiffusionTargetConfig {
            warmup_iterations: 100,
            sds_weight: 1.0,
            ..Default::default()
        });

        // During warmup
        assert_eq!(gen.sds_weight(0), 0.0);
        assert_eq!(gen.sds_weight(50), 0.0);
        assert_eq!(gen.sds_weight(99), 0.0);

        // Exactly at warmup boundary, ramp starts at 0
        assert_eq!(gen.sds_weight(100), 0.0);

        // After warmup, ramps up (iteration 101 has factor = 1/500)
        assert!(gen.sds_weight(101) > 0.0);
        assert!(gen.sds_weight(200) > gen.sds_weight(101));

        // Full weight after 500 steps post-warmup
        assert!((gen.sds_weight(600) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sds_loss_identical() {
        let loss = SdsLoss::default();
        let view = vec![0.5_f32; 100];

        let l = loss.compute(
            std::slice::from_ref(&view),
            std::slice::from_ref(&view),
            500,
        );
        assert!(
            l.abs() < 1e-6,
            "identical views should have ~0 loss, got {}",
            l
        );
    }

    #[test]
    fn test_sds_loss_different() {
        let loss = SdsLoss::default();
        let view1 = vec![0.0_f32; 100];
        let view2 = vec![1.0_f32; 100];

        let l = loss.compute(&[view1], &[view2], 500);
        assert!(l > 0.0, "different views should have positive loss");
    }

    #[test]
    fn test_ddpm_alpha_cumprod() {
        // At t=0, alpha_cumprod should be close to 1
        let alpha_0 = ddpm_alpha_cumprod(0, 1000);
        assert!(alpha_0 > 0.99, "alpha at t=0 = {}", alpha_0);

        // At t=999, alpha_cumprod should be small
        let alpha_999 = ddpm_alpha_cumprod(999, 1000);
        assert!(alpha_999 < 0.1, "alpha at t=999 = {}", alpha_999);

        // Should be monotonically decreasing
        assert!(ddpm_alpha_cumprod(100, 1000) > ddpm_alpha_cumprod(500, 1000));
    }

    #[test]
    fn test_view_consistency_empty() {
        let loss = ViewConsistencyLoss::default();
        let l = loss.compute(&[], &[], None, 64, 64);
        assert_eq!(l, 0.0);
    }

    #[test]
    fn test_temporal_consistency_empty() {
        let tc = TemporalConsistency::default();
        let l = tc.compute(&[], &[], None);
        assert_eq!(l, 0.0);
    }
}
