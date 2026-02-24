//! Full multi-view diffusion pipeline.
//!
//! Orchestrates the CLIP encoder, U-Net, VAE, and DDIM scheduler to
//! generate multi-view images from a single reference photo and camera poses.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn as nn;

use crate::clip::{build_clip_encoder, ClipImageEncoder};
use crate::config::DiffusionConfig;
use crate::scheduler::{DdimScheduler, PredictionType};
use crate::unet::MultiViewUNet;
use crate::upsampler::LatentUpsampler;
use crate::vae::Vae;
use crate::DiffusionError;

/// Output of the multi-view diffusion pipeline.
#[derive(Debug)]
pub struct MultiViewOutput {
    /// Generated images, one per view, as `(3, H, W)` tensors in `[0, 1]`.
    pub images: Vec<Tensor>,
    /// Width of each generated image.
    pub width: u32,
    /// Height of each generated image.
    pub height: u32,
}

/// The full multi-view diffusion pipeline.
pub struct MultiViewDiffusionPipeline {
    unet: MultiViewUNet,
    vae: Vae,
    clip_encoder: ClipImageEncoder,
    scheduler: DdimScheduler,
    upsampler: Option<LatentUpsampler>,
    config: DiffusionConfig,
    device: Device,
}

impl MultiViewDiffusionPipeline {
    /// Load a pipeline from a directory of safetensors files.
    ///
    /// Expected files:
    /// - `unet/diffusion_pytorch_model.safetensors`
    /// - `vae/diffusion_pytorch_model.safetensors`
    /// - `image_encoder/model.safetensors`
    /// - `upsampler/diffusion_pytorch_model.safetensors` (optional, for SdX2 mode)
    pub fn load(
        config: DiffusionConfig,
        weights_dir: &Path,
        device: &Device,
    ) -> std::result::Result<Self, DiffusionError> {
        let dtype = DType::F32;

        // Load U-Net weights
        let unet_path = weights_dir.join("unet/diffusion_pytorch_model.safetensors");
        let unet_data = std::fs::read(&unet_path)
            .map_err(|e| DiffusionError::ModelLoad(format!("Failed to read U-Net weights: {e}")))?;
        let unet_vb = nn::VarBuilder::from_buffered_safetensors(unet_data, dtype, device)
            .map_err(|e| DiffusionError::ModelLoad(format!("U-Net VarBuilder: {e}")))?;
        let unet = MultiViewUNet::new(unet_vb, &config)
            .map_err(|e| DiffusionError::ModelLoad(format!("U-Net build: {e}")))?;

        // Load VAE weights
        let vae_path = weights_dir.join("vae/diffusion_pytorch_model.safetensors");
        let vae_data = std::fs::read(&vae_path)
            .map_err(|e| DiffusionError::ModelLoad(format!("Failed to read VAE weights: {e}")))?;
        let vae_vb = nn::VarBuilder::from_buffered_safetensors(vae_data, dtype, device)
            .map_err(|e| DiffusionError::ModelLoad(format!("VAE VarBuilder: {e}")))?;
        let vae = Vae::new(vae_vb, config.latent_channels, config.vae_scale_factor)
            .map_err(|e| DiffusionError::ModelLoad(format!("VAE build: {e}")))?;

        // Load CLIP image encoder weights
        let clip_path = weights_dir.join("image_encoder/model.safetensors");
        let clip_data = std::fs::read(&clip_path)
            .map_err(|e| DiffusionError::ModelLoad(format!("Failed to read CLIP weights: {e}")))?;
        let clip_vb = nn::VarBuilder::from_buffered_safetensors(clip_data, dtype, device)
            .map_err(|e| DiffusionError::ModelLoad(format!("CLIP VarBuilder: {e}")))?;
        let clip_encoder = build_clip_encoder(clip_vb, &config)
            .map_err(|e| DiffusionError::ModelLoad(format!("CLIP build: {e}")))?;

        let scheduler = DdimScheduler::new(1000, PredictionType::VPrediction);

        // Load upsampler if configured
        let upsampler = if let Some(mode) = config.upsampler_mode {
            let upsampler_path = weights_dir.join("upsampler");
            Some(LatentUpsampler::load(mode, &upsampler_path, device)?)
        } else {
            None
        };

        Ok(Self {
            unet,
            vae,
            clip_encoder,
            scheduler,
            upsampler,
            config,
            device: device.clone(),
        })
    }

    /// Generate multi-view images from a reference image and camera poses.
    ///
    /// - `reference_image`: `(1, 3, 224, 224)` normalised image for CLIP.
    /// - `normal_map_latents`: `(num_views, latent_channels, h, w)` encoded normal maps.
    /// - `camera_poses`: `(num_views, pose_dim)` flattened extrinsics per view.
    /// - `seed`: RNG seed for reproducibility.
    ///
    /// # Classifier-Free Guidance (CFG)
    ///
    /// This pipeline implements CFG for IP-Adapter conditioning:
    /// - **Conditional pass**: Uses IP tokens from CLIP-encoded reference image
    /// - **Unconditional pass**: Skips IP tokens (no reference conditioning)
    /// - **Formula**: `pred = uncond + guidance_scale * (cond - uncond)`
    ///
    /// The `guidance_scale` parameter (from config) controls the strength of
    /// conditioning. Typical values:
    /// - `1.0` = no guidance (unconditional generation)
    /// - `3.0-7.5` = balanced (default: 3.0 for GAF)
    /// - `>10.0` = strong conditioning (may oversaturate)
    ///
    /// # Errors
    ///
    /// Returns `DiffusionError::Inference` if guidance_scale < 1.0 or if any
    /// tensor operation fails during generation.
    pub fn generate(
        &mut self,
        reference_image: &Tensor,
        normal_map_latents: &Tensor,
        camera_poses: &Tensor,
        _seed: u64,
    ) -> std::result::Result<MultiViewOutput, DiffusionError> {
        let num_views = self.config.num_views;
        let latent_size = self.config.latent_size;
        let latent_ch = self.config.latent_channels;

        // Validate guidance_scale
        if self.config.guidance_scale < 1.0 {
            return Err(DiffusionError::Inference(format!(
                "guidance_scale must be >= 1.0, got {}",
                self.config.guidance_scale
            )));
        }

        // 1. Encode reference image with CLIP for IP-Adapter conditioning
        let ip_tokens = self
            .clip_encoder
            .forward(reference_image)
            .map_err(|e| DiffusionError::Inference(format!("CLIP encode: {e}")))?;
        // Expand to all views: (1, seq, dim) -> (V, seq, dim)
        let ip_tokens = ip_tokens
            .repeat(&[num_views, 1, 1])
            .map_err(|e| DiffusionError::Inference(format!("IP token expand: {e}")))?;

        // 2. Prepare null text embedding (GAF doesn't use text conditioning)
        let null_context = Tensor::zeros(
            (num_views, 77, self.config.cross_attention_dim),
            DType::F32,
            &self.device,
        )
        .map_err(|e| DiffusionError::Inference(format!("null context: {e}")))?;

        // 3. Prepare initial noise
        let latent_shape = (num_views, latent_ch, latent_size, latent_size);
        let mut latents = Tensor::randn(0f32, 1f32, latent_shape, &self.device)
            .map_err(|e| DiffusionError::Inference(format!("noise init: {e}")))?;

        // 4. Set scheduler timesteps
        self.scheduler
            .set_timesteps(self.config.num_inference_steps);
        let timesteps = self.scheduler.timesteps().to_vec();

        // 5. Denoising loop with Classifier-Free Guidance (CFG)
        // We use separate forward passes for conditional and unconditional to
        // simplify implementation and avoid tensor concatenation issues with
        // IP-Adapter attention (which needs different shapes for cond/uncond).
        for &t in &timesteps {
            // Concatenate noise latents with normal-map latents
            let model_input = Tensor::cat(&[&latents, normal_map_latents], 1)
                .map_err(|e| DiffusionError::Inference(format!("concat: {e}")))?;

            // Forward pass 1: Conditional (with IP-Adapter tokens)
            // This provides identity-preserving conditioning from the reference image
            let noise_pred_cond = self.unet.forward(
                &model_input,
                t,
                Some(&null_context),
                Some(camera_poses),
                Some(&ip_tokens),
            )?;

            // Forward pass 2: Unconditional (without IP-Adapter tokens)
            // This provides the baseline without reference conditioning
            let noise_pred_uncond = self.unet.forward(
                &model_input,
                t,
                Some(&null_context),
                Some(camera_poses),
                None, // Skip IP tokens for unconditional
            )?;

            // Apply CFG formula: pred = uncond + scale * (cond - uncond)
            // This interpolates between unconditional and conditional predictions
            let diff = (&noise_pred_cond - &noise_pred_uncond)
                .map_err(|e| DiffusionError::Inference(format!("CFG diff: {e}")))?;
            let noise_pred = (&noise_pred_uncond + (diff * self.config.guidance_scale))
                .map_err(|e| DiffusionError::Inference(format!("CFG combine: {e}")))?;

            // Scheduler step
            latents = self
                .scheduler
                .step(&noise_pred, t, &latents)
                .map_err(|e| DiffusionError::Inference(format!("scheduler step: {e}")))?;
        }

        // 6. Upsample latents if configured (32×32 → 64×64)
        if let Some(ref mut upsampler) = self.upsampler {
            latents = upsampler
                .upsample(&latents, self.config.upsampler_steps)
                .map_err(|e| DiffusionError::Inference(format!("Upsampler: {e}")))?;
        }

        // 8. Decode latents with VAE
        let decoded = self
            .vae
            .decode(&latents)
            .map_err(|e| DiffusionError::Inference(format!("VAE decode: {e}")))?;

        // 9. Post-process: clamp to [0, 1]
        let images = ((decoded + 1.0)
            .map_err(|e| DiffusionError::Inference(format!("post +1: {e}")))?
            * 0.5)
            .map_err(|e| DiffusionError::Inference(format!("post *0.5: {e}")))?
            .clamp(0.0, 1.0)
            .map_err(|e| DiffusionError::Inference(format!("clamp: {e}")))?;

        // Split into per-view tensors
        let mut view_images = Vec::with_capacity(num_views);
        for i in 0..num_views {
            let img = images
                .narrow(0, i, 1)
                .and_then(|t| t.squeeze(0))
                .map_err(|e| DiffusionError::Inference(format!("split view {i}: {e}")))?;
            view_images.push(img);
        }

        // Calculate output size based on whether upsampling was used
        let size = if self.upsampler.is_some() {
            self.config.image_size as u32 * 2 // 512×512 with upsampling
        } else {
            self.config.image_size as u32 // 256×256 without upsampling
        };
        Ok(MultiViewOutput {
            images: view_images,
            width: size,
            height: size,
        })
    }
}
