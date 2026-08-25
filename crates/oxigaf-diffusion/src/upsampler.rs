//! Latent upsampler that doubles latent resolution (e.g. 32×32 → 64×64).
//!
//! This module implements the sd-x2-latent-upscaler pipeline, enabling 512×512
//! output resolution (vs 256×256).
//!
//! The upsampler uses a separate U-Net model with 10-step DDIM denoising in
//! latent space. Following diffusers' `StableDiffusionLatentUpscalePipeline`,
//! the denoising runs **at the target resolution**: the conditioning latents
//! are nearest-upsampled 2× and concatenated channel-wise onto the noisy
//! target-resolution latents, so the U-Net sees `2 × latent_channels` inputs.
//!
//! A fallback `BilinearVae` mode is also provided for cases where the
//! upsampler weights are not available.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn as nn;
use candle_nn::Module;

use crate::pipeline::seeded_normal_tensor;
use crate::scheduler::{DdimScheduler, PredictionType};
use crate::DiffusionError;

/// Latent channel count produced by the SD VAE.
const UPSAMPLER_LATENT_CHANNELS: usize = 4;

/// U-Net input channels: noisy target-resolution latents concatenated with the
/// nearest-upsampled conditioning latents.
const UPSAMPLER_UNET_IN_CHANNELS: usize = UPSAMPLER_LATENT_CHANNELS * 2;

/// Training timestep count of the upsampler's DDIM scheduler.
///
/// Named rather than inlined so [`LatentUpsampler::load`] and
/// [`LatentUpsampler::sdx2_from_var_builder`] cannot drift apart.
const UPSAMPLER_TRAIN_TIMESTEPS: usize = 1000;

/// Seed [`LatentUpsampler::upsample`] uses when the caller supplies none.
///
/// Any fixed value makes the plain `upsample` entry point reproducible; use
/// [`LatentUpsampler::upsample_with_seed`] to vary it.
const DEFAULT_UPSAMPLER_SEED: u64 = 0;

/// Mixed into the caller's seed before drawing the upsampler's init noise.
///
/// The pipeline keys both its initial latents and (through
/// [`LatentUpsampler::upsample_with_seed`]) the upsampler's noise off the same
/// run seed. Without a salt the two would be prefixes of one identical sample
/// stream; XOR-ing this constant gives the upsampler an independent sub-stream
/// while keeping the whole run a pure function of the run seed.
const UPSAMPLER_NOISE_SALT: u64 = 0x5DEE_CE66_D5B2_1B0F;

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
// Bilinear resampling
// ---------------------------------------------------------------------------

/// Gather indices and interpolation weights for one output axis.
struct AxisTaps {
    /// Lower source index per output position.
    lo: Vec<u32>,
    /// Upper source index per output position (clamped at the last row/column).
    hi: Vec<u32>,
    /// Fractional distance from `lo` towards `hi`, in `[0, 1)`.
    frac: Vec<f32>,
}

/// Build `align_corners = false` bilinear taps for one axis.
///
/// `src = (out_idx + 0.5) * in_len / out_len - 0.5`, clamped to
/// `[0, in_len - 1]` — the same mapping `resolution::resize_image_bilinear`
/// uses, and what PIL, OpenCV and `torch.nn.functional.interpolate` produce.
fn axis_taps(in_len: usize, out_len: usize) -> AxisTaps {
    let scale = in_len as f32 / out_len as f32;
    let max_idx = in_len - 1;
    let mut lo = Vec::with_capacity(out_len);
    let mut hi = Vec::with_capacity(out_len);
    let mut frac = Vec::with_capacity(out_len);

    for out_idx in 0..out_len {
        let src = ((out_idx as f32 + 0.5) * scale - 0.5).clamp(0.0, max_idx as f32);
        let floor = src.floor();
        let lo_idx = floor as usize;
        lo.push(lo_idx as u32);
        hi.push((lo_idx + 1).min(max_idx) as u32);
        frac.push(src - floor);
    }

    AxisTaps { lo, hi, frac }
}

/// True bilinear resampling of an `(N, C, H, W)` tensor to `(N, C, out_h, out_w)`.
///
/// candle only ships nearest-neighbour `upsample_nearest2d`, so the four corner
/// taps are gathered with `index_select` and blended with precomputed weights.
/// Everything stays on the tensor's own device.
fn bilinear_upsample2d(xs: &Tensor, out_h: usize, out_w: usize) -> Result<Tensor> {
    let (_, _, in_h, in_w) = xs.dims4()?;
    if in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 {
        return Err(candle_core::Error::Msg(format!(
            "bilinear_upsample2d: zero-sized resample {in_h}x{in_w} -> {out_h}x{out_w}"
        )));
    }

    let device = xs.device();
    let dtype = xs.dtype();
    // `index_select` needs contiguous storage.
    let src = xs.contiguous()?;

    let rows = axis_taps(in_h, out_h);
    let cols = axis_taps(in_w, out_w);

    let y_lo = Tensor::from_vec(rows.lo, out_h, device)?;
    let y_hi = Tensor::from_vec(rows.hi, out_h, device)?;
    let x_lo = Tensor::from_vec(cols.lo, out_w, device)?;
    let x_hi = Tensor::from_vec(cols.hi, out_w, device)?;

    // Gather the two source rows once, then the two source columns from each.
    let top = src.index_select(&y_lo, 2)?;
    let bottom = src.index_select(&y_hi, 2)?;
    let p00 = top.index_select(&x_lo, 3)?;
    let p10 = top.index_select(&x_hi, 3)?;
    let p01 = bottom.index_select(&x_lo, 3)?;
    let p11 = bottom.index_select(&x_hi, 3)?;

    // Blend weights, shaped (1, 1, out_h, out_w) so they broadcast over batch
    // and channels.
    let count = out_h * out_w;
    let mut w00 = Vec::with_capacity(count);
    let mut w10 = Vec::with_capacity(count);
    let mut w01 = Vec::with_capacity(count);
    let mut w11 = Vec::with_capacity(count);
    for &ty in &rows.frac {
        let ty_inv = 1.0 - ty;
        for &tx in &cols.frac {
            let tx_inv = 1.0 - tx;
            w00.push(tx_inv * ty_inv);
            w10.push(tx * ty_inv);
            w01.push(tx_inv * ty);
            w11.push(tx * ty);
        }
    }

    let shape = (1usize, 1usize, out_h, out_w);
    let w00 = Tensor::from_vec(w00, shape, device)?.to_dtype(dtype)?;
    let w10 = Tensor::from_vec(w10, shape, device)?.to_dtype(dtype)?;
    let w01 = Tensor::from_vec(w01, shape, device)?.to_dtype(dtype)?;
    let w11 = Tensor::from_vec(w11, shape, device)?.to_dtype(dtype)?;

    let out = p00.broadcast_mul(&w00)?;
    let out = (out + p10.broadcast_mul(&w10)?)?;
    let out = (out + p01.broadcast_mul(&w01)?)?;
    out + p11.broadcast_mul(&w11)?
}

// ---------------------------------------------------------------------------
// Upsampler U-Net
// ---------------------------------------------------------------------------

/// Simplified U-Net for latent upsampling.
///
/// This is a lighter architecture than the main multi-view U-Net,
/// designed specifically for the upsampling task.
///
/// The network is spatially size-preserving: the 2× upscale comes from running
/// it on target-resolution latents, not from the network itself.
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
    /// - Input: `(B, in_channels, H, W)` — noisy latents concatenated with the
    ///   nearest-upsampled conditioning latents, so `in_channels` is
    ///   `UPSAMPLER_UNET_IN_CHANNELS` (8), not 4.
    /// - Output: `(B, 4, H, W)`
    /// - Base channels: 128
    /// - Time embedding dimension: 512
    fn new(vs: nn::VarBuilder, in_channels: usize) -> Result<Self> {
        let out_channels = UPSAMPLER_LATENT_CHANNELS;
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

/// Latent upsampler that doubles latent resolution (32×32 → 64×64 by default).
///
/// This enables 512×512 output resolution (vs 256×256) by upsampling the
/// latent representation before VAE decoding. The upsampler uses a separate
/// U-Net model with 10-step DDIM denoising in latent space.
///
/// # Modes
///
/// - **SdX2**: Uses the sd-x2-latent-upscaler U-Net with DDIM denoising, run
///   at the target resolution with the source latents concatenated on as
///   conditioning.
/// - **BilinearVae**: Fallback mode using true bilinear interpolation
///   (`align_corners = false`), no U-Net required.
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
        match mode {
            UpsamplerMode::SdX2 => {
                let safetensors_path = weights_path.join("diffusion_pytorch_model.safetensors");
                let data = std::fs::read(&safetensors_path).map_err(|e| {
                    DiffusionError::ModelLoad(format!("Failed to read upsampler weights: {e}"))
                })?;
                let vb = nn::VarBuilder::from_buffered_safetensors(data, DType::F32, device)
                    .map_err(|e| DiffusionError::ModelLoad(format!("Upsampler VarBuilder: {e}")))?;
                Self::sdx2_from_var_builder(vb, device)
            }
            // Pure interpolation: no parameters, nothing to read.
            UpsamplerMode::BilinearVae => Ok(Self {
                mode,
                unet: None,
                scheduler: DdimScheduler::new(UPSAMPLER_TRAIN_TIMESTEPS, PredictionType::Epsilon),
                device: device.clone(),
            }),
        }
    }

    /// Build an [`UpsamplerMode::SdX2`] upsampler from an already-open
    /// [`nn::VarBuilder`].
    ///
    /// [`Self::load`] is this plus a `std::fs::read`. Keeping the two apart
    /// lets an in-process caller build the upsampler from weights that never
    /// touch the filesystem — which is how [`crate::pipeline`]'s determinism
    /// test assembles a whole pipeline without a weights directory.
    ///
    /// # Errors
    ///
    /// Returns [`DiffusionError::ModelLoad`] when `vb` cannot supply a tensor
    /// the U-Net needs, or supplies one of the wrong shape.
    pub(crate) fn sdx2_from_var_builder(
        vb: nn::VarBuilder<'_>,
        device: &Device,
    ) -> std::result::Result<Self, DiffusionError> {
        let unet = UpsamplerUNet::new(vb, UPSAMPLER_UNET_IN_CHANNELS)
            .map_err(|e| DiffusionError::ModelLoad(format!("Upsampler U-Net build: {e}")))?;
        Ok(Self {
            mode: UpsamplerMode::SdX2,
            unet: Some(unet),
            scheduler: DdimScheduler::new(UPSAMPLER_TRAIN_TIMESTEPS, PredictionType::Epsilon),
            device: device.clone(),
        })
    }

    /// Upsample latents by 2× (e.g. 32×32 → 64×64).
    ///
    /// # Arguments
    ///
    /// * `latents` - Input latents `(B, 4, H, W)`
    /// * `num_steps` - Number of DDIM denoising steps (typically 10); ignored
    ///   in `BilinearVae` mode
    ///
    /// # Returns
    ///
    /// Upsampled latents `(B, 4, 2H, 2W)`
    ///
    /// # Errors
    ///
    /// Returns `DiffusionError::InvalidLatentShape` when `latents` is not a
    /// 4-channel 4-D tensor, and `DiffusionError::Inference` if upsampling
    /// fails.
    pub fn upsample(
        &mut self,
        latents: &Tensor,
        num_steps: usize,
    ) -> std::result::Result<Tensor, DiffusionError> {
        self.upsample_with_seed(latents, num_steps, DEFAULT_UPSAMPLER_SEED)
    }

    /// Upsample latents by 2× from an explicit noise seed.
    ///
    /// `SdX2` mode starts its denoising loop from fresh Gaussian noise. That
    /// noise used to come from candle's process-global RNG, which cannot be
    /// seeded on CPU — so a pipeline run with an upsampler configured was not
    /// reproducible even though every other stage was. It now comes from the
    /// same dependency-free, device-independent xorshift + Box-Muller stream
    /// [`crate::pipeline::MultiViewDiffusionPipeline::generate`] uses for its
    /// initial latents, keyed by `seed`.
    ///
    /// `BilinearVae` mode is pure interpolation and ignores `seed` entirely.
    ///
    /// # Errors
    ///
    /// Same as [`Self::upsample`].
    pub fn upsample_with_seed(
        &mut self,
        latents: &Tensor,
        num_steps: usize,
        seed: u64,
    ) -> std::result::Result<Tensor, DiffusionError> {
        match self.mode {
            UpsamplerMode::SdX2 => self.upsample_sdx2(latents, num_steps, seed),
            UpsamplerMode::BilinearVae => self.upsample_bilinear(latents),
        }
    }

    /// Validate `(B, 4, H, W)` and return the doubled target size `(2H, 2W)`.
    fn target_size(latents: &Tensor) -> std::result::Result<(usize, usize), DiffusionError> {
        let (batch, ch, h, w) = latents
            .dims4()
            .map_err(|e| DiffusionError::Inference(format!("Invalid latent shape: {e}")))?;
        if h == 0 || w == 0 {
            return Err(DiffusionError::InvalidLatentShape {
                expected: vec![batch, UPSAMPLER_LATENT_CHANNELS, 1, 1],
                got: vec![batch, ch, h, w],
            });
        }
        if ch != UPSAMPLER_LATENT_CHANNELS {
            return Err(DiffusionError::InvalidLatentShape {
                expected: vec![batch, UPSAMPLER_LATENT_CHANNELS, h, w],
                got: vec![batch, ch, h, w],
            });
        }
        Ok((h * 2, w * 2))
    }

    /// Upsample using sd-x2-latent-upscaler with DDIM denoising.
    ///
    /// Mirrors diffusers' `StableDiffusionLatentUpscalePipeline`: denoising
    /// happens at the **target** resolution, and the low-resolution latents are
    /// nearest-upsampled once and concatenated onto every model input as
    /// conditioning. The U-Net therefore receives
    /// `UPSAMPLER_UNET_IN_CHANNELS` channels at `2H × 2W`, and its prediction
    /// is shape-compatible with the sample handed to the scheduler.
    fn upsample_sdx2(
        &mut self,
        latents: &Tensor,
        num_steps: usize,
        seed: u64,
    ) -> std::result::Result<Tensor, DiffusionError> {
        let (out_h, out_w) = Self::target_size(latents)?;
        let batch = latents
            .dim(0)
            .map_err(|e| DiffusionError::Inference(format!("Invalid latent shape: {e}")))?;

        if num_steps == 0 {
            return Err(DiffusionError::Inference(
                "Latent upsampling requires at least one denoising step".to_string(),
            ));
        }

        let unet = self.unet.as_ref().ok_or_else(|| {
            DiffusionError::Inference("SdX2 mode requires U-Net, but it's not loaded".to_string())
        })?;

        // Conditioning: the low-resolution latents lifted to the target grid.
        // Computed once — it does not change between denoising steps.
        let condition = latents
            .upsample_nearest2d(out_h, out_w)
            .map_err(|e| DiffusionError::Inference(format!("Condition upsample: {e}")))?;

        // Initialize noise at the target resolution.
        //
        // Drawn from the crate's seeded xorshift + Box-Muller stream rather
        // than `Tensor::randn` (candle's process-global RNG, unseedable on
        // CPU), so an upsampled run reproduces bit-for-bit like every other
        // stage of the pipeline.
        let mut current = seeded_normal_tensor(
            (batch, UPSAMPLER_LATENT_CHANNELS, out_h, out_w),
            seed ^ UPSAMPLER_NOISE_SALT,
            &self.device,
        )?;

        // Set scheduler timesteps. `set_timesteps` rejects `0` (guarded above)
        // and any count above the scheduler's 1000 training timesteps, which
        // would otherwise collapse the schedule to repeated zero timesteps.
        self.scheduler.set_timesteps(num_steps)?;
        let timesteps = self.scheduler.timesteps().to_vec();

        // DDIM denoising loop
        for &t in &timesteps {
            // (B, 4, 2H, 2W) ⧺ (B, 4, 2H, 2W) -> (B, 8, 2H, 2W)
            let model_input = Tensor::cat(&[&current, &condition], 1)
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

    /// Upsample using true bilinear interpolation (fallback mode).
    ///
    /// Doubles the spatial resolution with the same `align_corners = false`
    /// coordinate mapping PIL/OpenCV use, so decoded output does not show the
    /// blocky 8×8-pixel artefacts that nearest-neighbour latent upscaling
    /// produces after the VAE.
    fn upsample_bilinear(&self, latents: &Tensor) -> std::result::Result<Tensor, DiffusionError> {
        let (out_h, out_w) = Self::target_size(latents)?;
        bilinear_upsample2d(latents, out_h, out_w)
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

    // ------------------------------------------------------------------
    // Bilinear fallback: must really interpolate, at a derived output size.
    // ------------------------------------------------------------------

    fn bilinear_upsampler(device: &Device) -> LatentUpsampler {
        LatentUpsampler {
            mode: UpsamplerMode::BilinearVae,
            unet: None,
            scheduler: DdimScheduler::new(1000, PredictionType::Epsilon),
            device: device.clone(),
        }
    }

    #[test]
    fn test_bilinear_upsample2d_interpolates_between_rows() -> Result<()> {
        let device = Device::Cpu;
        // Row 0 = 0, row 1 = 1. align_corners=false taps for 2 -> 4 give
        // fractional offsets 0, 0.25, 0.75, 0 (the last clamped to row 1).
        let xs = Tensor::from_vec(vec![0f32, 0.0, 1.0, 1.0], (1, 1, 2, 2), &device)?;
        let out = bilinear_upsample2d(&xs, 4, 4)?;
        assert_eq!(out.dims(), &[1, 1, 4, 4]);

        let values = out.flatten_all()?.to_vec1::<f32>()?;
        let expected = [0.0f32, 0.25, 0.75, 1.0];
        for (row, &want) in expected.iter().enumerate() {
            for col in 0..4 {
                let got = values[row * 4 + col];
                assert!(
                    (got - want).abs() < 1e-5,
                    "row {row} col {col}: {got} vs {want}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_bilinear_is_not_nearest_neighbour() -> Result<()> {
        // Regression: `upsample_bilinear` used to call `upsample_nearest2d`.
        let device = Device::Cpu;
        let mut upsampler = bilinear_upsampler(&device);

        let latents = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &device)?;
        let bilinear = upsampler
            .upsample(&latents, 0)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        let nearest = latents.upsample_nearest2d(16, 16)?;

        assert_eq!(bilinear.dims(), &[1, 4, 16, 16]);
        let diff = (bilinear - nearest)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(
            diff > 1e-3,
            "bilinear upsampling must not degenerate to nearest (total diff {diff})"
        );
        Ok(())
    }

    #[test]
    fn test_bilinear_output_size_derived_from_input() -> Result<()> {
        // Regression: the target size used to be hardcoded to 64×64.
        let device = Device::Cpu;
        let mut upsampler = bilinear_upsampler(&device);

        for (h, w) in [(8usize, 8usize), (16, 16), (12, 20)] {
            let latents = Tensor::randn(0f32, 1f32, (1, 4, h, w), &device)?;
            let out = upsampler
                .upsample(&latents, 0)
                .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
            assert_eq!(out.dims(), &[1, 4, h * 2, w * 2]);
        }
        Ok(())
    }

    #[test]
    fn test_bilinear_rejects_wrong_channel_count() -> Result<()> {
        // Regression: the fallback path never validated its input.
        let device = Device::Cpu;
        let mut upsampler = bilinear_upsampler(&device);
        let latents = Tensor::randn(0f32, 1f32, (1, 3, 8, 8), &device)?;
        assert!(upsampler.upsample(&latents, 0).is_err());
        Ok(())
    }

    // ------------------------------------------------------------------
    // SdX2: the denoising loop must actually run.
    // ------------------------------------------------------------------

    #[test]
    fn test_sdx2_denoising_step_runs_with_synthetic_weights() -> Result<()> {
        // Regression: the U-Net was built for 4 input channels while the loop
        // fed it 8, and the model input was built at half the resolution of
        // the sample handed to the scheduler. Both made this path unusable.
        let device = Device::Cpu;
        let varmap = nn::VarMap::new();
        let vb = nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let unet = UpsamplerUNet::new(vb, UPSAMPLER_UNET_IN_CHANNELS)?;

        let mut upsampler = LatentUpsampler {
            mode: UpsamplerMode::SdX2,
            unet: Some(unet),
            scheduler: DdimScheduler::new(1000, PredictionType::Epsilon),
            device: device.clone(),
        };

        let latents = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &device)?;
        let out = upsampler
            .upsample(&latents, 1)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;

        assert_eq!(out.dims(), &[1, 4, 16, 16]);
        Ok(())
    }

    /// Regression: the SdX2 loop initialised its latents with
    /// `Tensor::randn`, i.e. candle's process-global RNG, which cannot be
    /// seeded on CPU — so an upsampled pipeline run was irreproducible even
    /// though every other stage was deterministic.
    #[test]
    fn test_sdx2_is_reproducible_for_a_fixed_seed() -> Result<()> {
        let device = Device::Cpu;
        let varmap = nn::VarMap::new();
        let vb = nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        // One U-Net, reused: only the init noise may vary between runs.
        let mut upsampler = LatentUpsampler {
            mode: UpsamplerMode::SdX2,
            unet: Some(UpsamplerUNet::new(vb, UPSAMPLER_UNET_IN_CHANNELS)?),
            scheduler: DdimScheduler::new(1000, PredictionType::Epsilon),
            device: device.clone(),
        };

        let latents = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &device)?;
        let run = |up: &mut LatentUpsampler, seed: u64| -> Result<Vec<f32>> {
            up.upsample_with_seed(&latents, 1, seed)
                .map_err(|e| candle_core::Error::Msg(format!("{e}")))?
                .flatten_all()?
                .to_vec1::<f32>()
        };

        let first = run(&mut upsampler, 7)?;
        let again = run(&mut upsampler, 7)?;
        assert_eq!(first, again, "the same seed must reproduce bit-identically");

        let other = run(&mut upsampler, 8)?;
        assert_ne!(first, other, "a different seed must change the output");

        // `upsample` itself must also be reproducible (fixed default seed).
        let default_a = upsampler
            .upsample(&latents, 1)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let default_b = upsampler
            .upsample(&latents, 1)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(default_a, default_b);
        Ok(())
    }

    /// The upsampler's noise must not be a prefix-copy of the initial latents
    /// the pipeline draws from the same run seed.
    #[test]
    fn test_upsampler_noise_salt_decorrelates_the_streams() -> Result<()> {
        let device = Device::Cpu;
        let seed = 4242u64;
        let read = |s: u64| -> Result<Vec<f32>> {
            seeded_normal_tensor((1, 4, 4, 4), s, &device)
                .map_err(|e| candle_core::Error::Msg(format!("{e}")))?
                .flatten_all()?
                .to_vec1::<f32>()
        };
        // The pipeline keys its initial latents off the bare run seed; the
        // upsampler salts it, so the two must not share a sample stream.
        assert_ne!(read(seed)?, read(seed ^ UPSAMPLER_NOISE_SALT)?);
        assert_ne!(UPSAMPLER_NOISE_SALT, 0);
        Ok(())
    }

    #[test]
    fn test_upsampler_unet_declares_eight_input_channels() {
        // The conditioning latents are concatenated onto the noisy latents.
        assert_eq!(UPSAMPLER_UNET_IN_CHANNELS, 2 * UPSAMPLER_LATENT_CHANNELS);
        assert_eq!(UPSAMPLER_UNET_IN_CHANNELS, 8);
    }

    /// `sdx2_from_var_builder` must produce exactly what the file-backed
    /// [`LatentUpsampler::load`] would: `SdX2` mode, a U-Net, and the same
    /// 1000-timestep epsilon schedule.
    #[test]
    fn test_sdx2_from_var_builder_matches_the_loaded_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = nn::VarMap::new();
        let vb = nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut upsampler = LatentUpsampler::sdx2_from_var_builder(vb, &device)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;

        assert_eq!(upsampler.mode, UpsamplerMode::SdX2);
        assert!(upsampler.unet.is_some());
        // `set_timesteps` rejects a count above the training timesteps, which
        // pins the schedule at `UPSAMPLER_TRAIN_TIMESTEPS` without needing an
        // accessor the scheduler does not expose.
        assert!(upsampler
            .scheduler
            .set_timesteps(UPSAMPLER_TRAIN_TIMESTEPS)
            .is_ok());
        assert!(upsampler
            .scheduler
            .set_timesteps(UPSAMPLER_TRAIN_TIMESTEPS + 1)
            .is_err());

        let latents = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &device)?;
        let out = upsampler
            .upsample_with_seed(&latents, 1, 5)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        assert_eq!(out.dims(), &[1, 4, 16, 16]);
        Ok(())
    }

    /// `BilinearVae` has no parameters, so `load` must not go near the
    /// filesystem for it — the weights directory need not even exist.
    #[test]
    fn test_load_bilinear_needs_no_weights_on_disk() -> Result<()> {
        let device = Device::Cpu;
        let missing = std::env::temp_dir().join("oxigaf_upsampler_weights_that_do_not_exist");
        let upsampler = LatentUpsampler::load(UpsamplerMode::BilinearVae, &missing, &device)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        assert_eq!(upsampler.mode, UpsamplerMode::BilinearVae);
        assert!(upsampler.unet.is_none());

        // `SdX2` from the same directory must still fail: it does need weights.
        assert!(LatentUpsampler::load(UpsamplerMode::SdX2, &missing, &device).is_err());
        Ok(())
    }

    #[test]
    fn test_sdx2_zero_steps_errors() -> Result<()> {
        let device = Device::Cpu;
        let mut upsampler = LatentUpsampler {
            mode: UpsamplerMode::SdX2,
            unet: None,
            scheduler: DdimScheduler::new(1000, PredictionType::Epsilon),
            device: device.clone(),
        };
        let latents = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &device)?;
        assert!(upsampler.upsample(&latents, 0).is_err());
        Ok(())
    }

    #[test]
    fn test_sdx2_without_unet_errors() -> Result<()> {
        let device = Device::Cpu;
        let mut upsampler = LatentUpsampler {
            mode: UpsamplerMode::SdX2,
            unet: None,
            scheduler: DdimScheduler::new(1000, PredictionType::Epsilon),
            device: device.clone(),
        };
        let latents = Tensor::randn(0f32, 1f32, (1, 4, 8, 8), &device)?;
        assert!(upsampler.upsample(&latents, 10).is_err());
        Ok(())
    }

    #[test]
    fn test_axis_taps_clamp_at_edges() {
        let taps = axis_taps(2, 4);
        assert_eq!(taps.lo, vec![0, 0, 0, 1]);
        assert_eq!(taps.hi, vec![1, 1, 1, 1]);
        assert!((taps.frac[0] - 0.0).abs() < 1e-6);
        assert!((taps.frac[1] - 0.25).abs() < 1e-6);
        assert!((taps.frac[2] - 0.75).abs() < 1e-6);
        assert!((taps.frac[3] - 0.0).abs() < 1e-6);
    }
}
