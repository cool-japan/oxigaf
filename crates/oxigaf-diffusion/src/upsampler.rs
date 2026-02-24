//! Latent upsampler for 32×32 → 64×64 latent upsampling.
//!
//! This module implements the sd-x2-latent-upscaler pipeline for upsampling
//! latents from 32×32 to 64×64, enabling 512×512 output resolution (vs 256×256).
//!
//! The upsampler uses a separate U-Net model with 10-step DDIM denoising in
//! latent space. A fallback `BilinearVae` mode is also provided for cases
//! where the upsampler weights are not available.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn as nn;
use candle_nn::Module;

use crate::scheduler::{DdimScheduler, PredictionType};
use crate::DiffusionError;

// ---------------------------------------------------------------------------
// Upsampler mode
// ---------------------------------------------------------------------------

/// Upsampler mode selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpsamplerMode {
    /// Use sd-x2-latent-upscaler U-Net with 10-step DDIM denoising.
    SdX2,
    /// Fallback: bilinear upsampling without denoising U-Net.
    BilinearVae,
}

// ---------------------------------------------------------------------------
// Building blocks for upsampler U-Net
// ---------------------------------------------------------------------------

/// ResNet block with time-step conditioning for upsampler U-Net.
#[derive(Debug)]
struct UpsamplerResBlock {
    norm1: nn::GroupNorm,
    conv1: nn::Conv2d,
    time_emb_proj: nn::Linear,
    norm2: nn::GroupNorm,
    conv2: nn::Conv2d,
    residual_conv: Option<nn::Conv2d>,
}

impl UpsamplerResBlock {
    fn new(vs: nn::VarBuilder, in_ch: usize, out_ch: usize, time_dim: usize) -> Result<Self> {
        let norm1 = nn::group_norm(32, in_ch, 1e-5, vs.pp("norm1"))?;
        let conv1 = nn::conv2d(
            in_ch,
            out_ch,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("conv1"),
        )?;
        let time_emb_proj = nn::linear(time_dim, out_ch, vs.pp("time_emb_proj"))?;
        let norm2 = nn::group_norm(32, out_ch, 1e-5, vs.pp("norm2"))?;
        let conv2 = nn::conv2d(
            out_ch,
            out_ch,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("conv2"),
        )?;
        let residual_conv = if in_ch != out_ch {
            Some(nn::conv2d(
                in_ch,
                out_ch,
                1,
                Default::default(),
                vs.pp("conv_shortcut"),
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1,
            conv1,
            time_emb_proj,
            norm2,
            conv2,
            residual_conv,
        })
    }

    fn forward(&self, xs: &Tensor, time_emb: &Tensor) -> Result<Tensor> {
        let residual = if let Some(ref conv) = self.residual_conv {
            conv.forward(xs)?
        } else {
            xs.clone()
        };
        let h = self.norm1.forward(xs)?.silu()?;
        let h = self.conv1.forward(&h)?;

        // Add time embedding
        let t = self.time_emb_proj.forward(&time_emb.silu()?)?;
        let t = t.unsqueeze(2)?.unsqueeze(3)?;
        let h = (h.clone() + t.broadcast_as(h.shape())?)?;

        let h = self.norm2.forward(&h)?.silu()?;
        let h = self.conv2.forward(&h)?;
        h + residual
    }
}

/// Downsample with strided convolution for upsampler U-Net.
#[derive(Debug)]
struct UpsamplerDownsample {
    conv: nn::Conv2d,
}

impl UpsamplerDownsample {
    fn new(vs: nn::VarBuilder, channels: usize) -> Result<Self> {
        let conv = nn::conv2d(
            channels,
            channels,
            3,
            nn::Conv2dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
            vs.pp("conv"),
        )?;
        Ok(Self { conv })
    }
}

impl Module for UpsamplerDownsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.conv.forward(xs)
    }
}

/// Upsample with nearest-neighbor interpolation + conv for upsampler U-Net.
#[derive(Debug)]
struct UpsamplerUpsample {
    conv: nn::Conv2d,
}

impl UpsamplerUpsample {
    fn new(vs: nn::VarBuilder, channels: usize) -> Result<Self> {
        let conv = nn::conv2d(
            channels,
            channels,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("conv"),
        )?;
        Ok(Self { conv })
    }
}

impl Module for UpsamplerUpsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;
        let xs = xs.upsample_nearest2d(h * 2, w * 2)?;
        self.conv.forward(&xs)
    }
}

/// Self-attention block for upsampler U-Net.
#[derive(Debug)]
struct UpsamplerAttention {
    group_norm: nn::GroupNorm,
    to_qkv: nn::Conv2d,
    to_out: nn::Conv2d,
    channels: usize,
}

impl UpsamplerAttention {
    fn new(vs: nn::VarBuilder, channels: usize) -> Result<Self> {
        let group_norm = nn::group_norm(32, channels, 1e-5, vs.pp("group_norm"))?;
        let to_qkv = nn::conv2d(
            channels,
            channels * 3,
            1,
            Default::default(),
            vs.pp("to_qkv"),
        )?;
        let to_out = nn::conv2d(channels, channels, 1, Default::default(), vs.pp("to_out"))?;
        Ok(Self {
            group_norm,
            to_qkv,
            to_out,
            channels,
        })
    }
}

impl Module for UpsamplerAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let (b, _c, h, w) = xs.dims4()?;
        let xs = self.group_norm.forward(xs)?;
        let qkv = self.to_qkv.forward(&xs)?;
        let qkv = qkv.reshape((b, 3, self.channels, h * w))?;
        let q = qkv.narrow(1, 0, 1)?.squeeze(1)?;
        let k = qkv.narrow(1, 1, 1)?.squeeze(1)?;
        let v = qkv.narrow(1, 2, 1)?.squeeze(1)?;

        let scale = (self.channels as f64).powf(-0.5);
        let attn = (q.transpose(1, 2)?.matmul(&k)? * scale)?;
        let attn = nn::ops::softmax_last_dim(&attn)?;
        let out = v.matmul(&attn.transpose(1, 2)?)?;
        let out = out.reshape((b, self.channels, h, w))?;
        let out = self.to_out.forward(&out)?;
        out + residual
    }
}

// ---------------------------------------------------------------------------
// Time embedding
// ---------------------------------------------------------------------------

/// Sinusoidal timestep embedding.
fn timestep_embedding(timesteps: &Tensor, dim: usize) -> Result<Tensor> {
    let half = dim / 2;
    let freqs = Tensor::arange(0f32, half as f32, timesteps.device())?;
    let freqs = (freqs * (-10000f64.ln() / half as f64))?;
    let freqs = freqs.exp()?;
    let args = timesteps
        .unsqueeze(1)?
        .broadcast_mul(&freqs.unsqueeze(0)?)?;
    let sin = args.sin()?;
    let cos = args.cos()?;
    Tensor::cat(&[&cos, &sin], 1)
}

/// MLP for time embedding projection.
#[derive(Debug)]
struct TimeEmbedding {
    linear1: nn::Linear,
    linear2: nn::Linear,
}

impl TimeEmbedding {
    fn new(vs: nn::VarBuilder, in_dim: usize, out_dim: usize) -> Result<Self> {
        let linear1 = nn::linear(in_dim, out_dim, vs.pp("linear_1"))?;
        let linear2 = nn::linear(out_dim, out_dim, vs.pp("linear_2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.linear1.forward(x)?.silu()?;
        self.linear2.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Upsampler U-Net
// ---------------------------------------------------------------------------

/// Simplified U-Net for latent upsampling (32×32 → 64×64).
///
/// This is a lighter architecture than the main multi-view U-Net,
/// designed specifically for the upsampling task.
#[derive(Debug)]
struct UpsamplerUNet {
    conv_in: nn::Conv2d,
    time_embedding: TimeEmbedding,
    // Down blocks
    down_res_1: UpsamplerResBlock,
    down_attn_1: Option<UpsamplerAttention>,
    down_res_2: UpsamplerResBlock,
    downsample: UpsamplerDownsample,
    // Mid blocks
    mid_res_1: UpsamplerResBlock,
    mid_attn: UpsamplerAttention,
    mid_res_2: UpsamplerResBlock,
    // Up blocks
    upsample: UpsamplerUpsample,
    up_res_1: UpsamplerResBlock,
    up_attn_1: Option<UpsamplerAttention>,
    up_res_2: UpsamplerResBlock,
    // Out
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2d,
}

impl UpsamplerUNet {
    /// Build the upsampler U-Net from weights.
    ///
    /// Architecture:
    /// - Input: (B, 4, 32, 32)
    /// - Output: (B, 4, 64, 64)
    /// - Base channels: 128
    /// - Time embedding dimension: 512
    fn new(vs: nn::VarBuilder) -> Result<Self> {
        let in_channels = 4;
        let out_channels = 4;
        let base_channels = 128;
        let time_dim = 512;

        let conv_in = nn::conv2d(
            in_channels,
            base_channels,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("conv_in"),
        )?;

        let time_embedding = TimeEmbedding::new(vs.pp("time_embedding"), base_channels, time_dim)?;

        // Down blocks
        let down_res_1 = UpsamplerResBlock::new(
            vs.pp("down_blocks.0.resnets.0"),
            base_channels,
            base_channels,
            time_dim,
        )?;
        let down_attn_1 = Some(UpsamplerAttention::new(
            vs.pp("down_blocks.0.attentions.0"),
            base_channels,
        )?);
        let down_res_2 = UpsamplerResBlock::new(
            vs.pp("down_blocks.0.resnets.1"),
            base_channels,
            base_channels,
            time_dim,
        )?;
        let downsample =
            UpsamplerDownsample::new(vs.pp("down_blocks.0.downsamplers.0"), base_channels)?;

        // Mid blocks
        let mid_res_1 = UpsamplerResBlock::new(
            vs.pp("mid_block.resnets.0"),
            base_channels,
            base_channels,
            time_dim,
        )?;
        let mid_attn = UpsamplerAttention::new(vs.pp("mid_block.attentions.0"), base_channels)?;
        let mid_res_2 = UpsamplerResBlock::new(
            vs.pp("mid_block.resnets.1"),
            base_channels,
            base_channels,
            time_dim,
        )?;

        // Up blocks
        let upsample = UpsamplerUpsample::new(vs.pp("up_blocks.0.upsamplers.0"), base_channels)?;
        let up_res_1 = UpsamplerResBlock::new(
            vs.pp("up_blocks.0.resnets.0"),
            base_channels * 2, // concat with skip
            base_channels,
            time_dim,
        )?;
        let up_attn_1 = Some(UpsamplerAttention::new(
            vs.pp("up_blocks.0.attentions.0"),
            base_channels,
        )?);
        let up_res_2 = UpsamplerResBlock::new(
            vs.pp("up_blocks.0.resnets.1"),
            base_channels * 2, // concat with skip
            base_channels,
            time_dim,
        )?;

        // Output
        let conv_norm_out = nn::group_norm(32, base_channels, 1e-5, vs.pp("conv_norm_out"))?;
        let conv_out = nn::conv2d(
            base_channels,
            out_channels,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            time_embedding,
            down_res_1,
            down_attn_1,
            down_res_2,
            downsample,
            mid_res_1,
            mid_attn,
            mid_res_2,
            upsample,
            up_res_1,
            up_attn_1,
            up_res_2,
            conv_norm_out,
            conv_out,
        })
    }

    /// Forward pass of the upsampler U-Net.
    fn forward(&self, sample: &Tensor, timestep: usize) -> Result<Tensor> {
        let batch_size = sample.dim(0)?;
        let device = sample.device();

        // Time embedding
        let t_emb =
            timestep_embedding(&Tensor::full(timestep as f32, (batch_size,), device)?, 128)?;
        let emb = self.time_embedding.forward(&t_emb)?;

        // Input conv
        let mut h = self.conv_in.forward(sample)?;

        // Down blocks with skip connections
        let skip1 = h.clone();
        h = self.down_res_1.forward(&h, &emb)?;
        if let Some(ref attn) = self.down_attn_1 {
            h = attn.forward(&h)?;
        }
        let skip2 = h.clone();
        h = self.down_res_2.forward(&h, &emb)?;
        h = self.downsample.forward(&h)?;

        // Mid blocks
        h = self.mid_res_1.forward(&h, &emb)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_res_2.forward(&h, &emb)?;

        // Up blocks with skip connections
        h = self.upsample.forward(&h)?;
        h = Tensor::cat(&[h, skip2], 1)?;
        h = self.up_res_1.forward(&h, &emb)?;
        if let Some(ref attn) = self.up_attn_1 {
            h = attn.forward(&h)?;
        }
        h = Tensor::cat(&[h, skip1], 1)?;
        h = self.up_res_2.forward(&h, &emb)?;

        // Output
        h = self.conv_norm_out.forward(&h)?.silu()?;
        self.conv_out.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Latent Upsampler public API
// ---------------------------------------------------------------------------

/// Latent upsampler for 32×32 → 64×64 latent upsampling.
///
/// This enables 512×512 output resolution (vs 256×256) by upsampling the
/// latent representation before VAE decoding. The upsampler uses a separate
/// U-Net model with 10-step DDIM denoising in latent space.
///
/// # Modes
///
/// - **SdX2**: Uses the sd-x2-latent-upscaler U-Net with DDIM denoising.
/// - **BilinearVae**: Fallback mode that uses simple bilinear upsampling.
///
/// # Example
///
/// ```rust,ignore
/// use oxigaf_diffusion::{LatentUpsampler, UpsamplerMode};
/// use candle_core::Device;
///
/// let device = Device::Cpu;
/// let upsampler = LatentUpsampler::load(
///     UpsamplerMode::SdX2,
///     "weights/upsampler",
///     &device,
/// )?;
///
/// // Upsample latents from 32×32 to 64×64
/// let upsampled = upsampler.upsample(&latents, 10)?;
/// ```
#[derive(Debug)]
pub struct LatentUpsampler {
    mode: UpsamplerMode,
    unet: Option<UpsamplerUNet>,
    scheduler: DdimScheduler,
    device: Device,
}

impl LatentUpsampler {
    /// Load a latent upsampler from weights.
    ///
    /// # Arguments
    ///
    /// * `mode` - Upsampler mode (SdX2 or BilinearVae)
    /// * `weights_path` - Path to upsampler weights directory (for SdX2 mode)
    /// * `device` - Device to load the model on
    ///
    /// # Errors
    ///
    /// Returns `DiffusionError::ModelLoad` if weight loading fails.
    pub fn load(
        mode: UpsamplerMode,
        weights_path: &std::path::Path,
        device: &Device,
    ) -> std::result::Result<Self, DiffusionError> {
        let unet = match mode {
            UpsamplerMode::SdX2 => {
                let dtype = DType::F32;
                let safetensors_path = weights_path.join("diffusion_pytorch_model.safetensors");
                let data = std::fs::read(&safetensors_path).map_err(|e| {
                    DiffusionError::ModelLoad(format!("Failed to read upsampler weights: {e}"))
                })?;
                let vb = nn::VarBuilder::from_buffered_safetensors(data, dtype, device)
                    .map_err(|e| DiffusionError::ModelLoad(format!("Upsampler VarBuilder: {e}")))?;
                Some(UpsamplerUNet::new(vb).map_err(|e| {
                    DiffusionError::ModelLoad(format!("Upsampler U-Net build: {e}"))
                })?)
            }
            UpsamplerMode::BilinearVae => None,
        };

        let scheduler = DdimScheduler::new(1000, PredictionType::Epsilon);

        Ok(Self {
            mode,
            unet,
            scheduler,
            device: device.clone(),
        })
    }

    /// Upsample latents from 32×32 to 64×64.
    ///
    /// # Arguments
    ///
    /// * `latents` - Input latents `(B, 4, 32, 32)`
    /// * `num_steps` - Number of DDIM denoising steps (typically 10)
    ///
    /// # Returns
    ///
    /// Upsampled latents `(B, 4, 64, 64)`
    ///
    /// # Errors
    ///
    /// Returns `DiffusionError::Inference` if upsampling fails.
    pub fn upsample(
        &mut self,
        latents: &Tensor,
        num_steps: usize,
    ) -> std::result::Result<Tensor, DiffusionError> {
        match self.mode {
            UpsamplerMode::SdX2 => self.upsample_sdx2(latents, num_steps),
            UpsamplerMode::BilinearVae => self.upsample_bilinear(latents),
        }
    }

    /// Upsample using sd-x2-latent-upscaler with DDIM denoising.
    fn upsample_sdx2(
        &mut self,
        latents: &Tensor,
        num_steps: usize,
    ) -> std::result::Result<Tensor, DiffusionError> {
        let unet = self.unet.as_ref().ok_or_else(|| {
            DiffusionError::Inference("SdX2 mode requires U-Net, but it's not loaded".to_string())
        })?;

        // Validate input shape
        let (batch, ch, h, w) = latents
            .dims4()
            .map_err(|e| DiffusionError::Inference(format!("Invalid latent shape: {e}")))?;
        if ch != 4 || h != 32 || w != 32 {
            return Err(DiffusionError::InvalidLatentShape {
                expected: vec![batch, 4, 32, 32],
                got: vec![batch, ch, h, w],
            });
        }

        // Initialize noise at 64×64
        let mut current = Tensor::randn(0f32, 1f32, (batch, 4, 64, 64), &self.device)
            .map_err(|e| DiffusionError::Inference(format!("Noise init: {e}")))?;

        // Set scheduler timesteps
        self.scheduler.set_timesteps(num_steps);
        let timesteps = self.scheduler.timesteps().to_vec();

        // DDIM denoising loop
        for &t in &timesteps {
            // Downsample current to 32×32 for conditioning
            let current_downsampled = current
                .upsample_nearest2d(32, 32)
                .map_err(|e| DiffusionError::Inference(format!("Downsample for condition: {e}")))?;

            // Concatenate with input latents for conditioning
            let model_input = Tensor::cat(&[&current_downsampled, latents], 1)
                .map_err(|e| DiffusionError::Inference(format!("Concat condition: {e}")))?;

            // U-Net prediction
            let noise_pred =
                unet.forward(&model_input, t)
                    .map_err(|e| DiffusionError::UnetForwardFailed {
                        timestep: t,
                        reason: format!("{e}"),
                    })?;

            // Scheduler step
            current = self
                .scheduler
                .step(&noise_pred, t, &current)
                .map_err(|e| DiffusionError::Inference(format!("Scheduler step: {e}")))?;
        }

        Ok(current)
    }

    /// Upsample using simple bilinear interpolation (fallback mode).
    fn upsample_bilinear(&self, latents: &Tensor) -> std::result::Result<Tensor, DiffusionError> {
        latents
            .upsample_nearest2d(64, 64)
            .map_err(|e| DiffusionError::Inference(format!("Bilinear upsample: {e}")))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilinear_upsampling() -> Result<()> {
        let device = Device::Cpu;
        let mut upsampler = LatentUpsampler {
            mode: UpsamplerMode::BilinearVae,
            unet: None,
            scheduler: DdimScheduler::new(1000, PredictionType::Epsilon),
            device: device.clone(),
        };

        let latents = Tensor::randn(0f32, 1f32, (2, 4, 32, 32), &device)?;
        let upsampled = upsampler
            .upsample(&latents, 10)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;

        assert_eq!(upsampled.dims(), &[2, 4, 64, 64]);
        Ok(())
    }

    #[test]
    fn test_upsampler_mode_equality() {
        assert_eq!(UpsamplerMode::SdX2, UpsamplerMode::SdX2);
        assert_eq!(UpsamplerMode::BilinearVae, UpsamplerMode::BilinearVae);
        assert_ne!(UpsamplerMode::SdX2, UpsamplerMode::BilinearVae);
    }

    #[test]
    fn test_timestep_embedding_shape() -> Result<()> {
        let device = Device::Cpu;
        let timesteps = Tensor::full(500f32, (4,), &device)?;
        let emb = timestep_embedding(&timesteps, 128)?;
        assert_eq!(emb.dims(), &[4, 128]);
        Ok(())
    }

    #[test]
    fn test_bilinear_with_different_batch_sizes() -> Result<()> {
        let device = Device::Cpu;
        let mut upsampler = LatentUpsampler {
            mode: UpsamplerMode::BilinearVae,
            unet: None,
            scheduler: DdimScheduler::new(1000, PredictionType::Epsilon),
            device: device.clone(),
        };

        for batch_size in [1, 2, 4, 8] {
            let latents = Tensor::randn(0f32, 1f32, (batch_size, 4, 32, 32), &device)?;
            let upsampled = upsampler
                .upsample(&latents, 10)
                .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
            assert_eq!(upsampled.dims(), &[batch_size, 4, 64, 64]);
        }
        Ok(())
    }

    #[test]
    fn test_invalid_latent_shape_sdx2() -> Result<()> {
        let device = Device::Cpu;
        // Create upsampler without U-Net (will fail on validation)
        let mut upsampler = LatentUpsampler {
            mode: UpsamplerMode::SdX2,
            unet: None,
            scheduler: DdimScheduler::new(1000, PredictionType::Epsilon),
            device: device.clone(),
        };

        let latents = Tensor::randn(0f32, 1f32, (2, 3, 32, 32), &device)?; // Wrong channels
        let result = upsampler.upsample(&latents, 10);
        assert!(result.is_err());
        Ok(())
    }
}
