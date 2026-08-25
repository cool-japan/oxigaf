//! Variational Autoencoder (SD 2.1 compatible).
//!
//! Encodes images to latent space and decodes latents back to pixel space.
//! This is a simplified but functional VAE that mirrors the architecture
//! used in Stable Diffusion 2.1.
//!
//! ## Weight naming
//!
//! Module paths follow the diffusers `AutoencoderKL` convention throughout
//! (`encoder.down_blocks.{i}.resnets.{j}`, `mid_block.attentions.0`, etc.),
//! matching [`crate::upsampler`]'s U-Net, which uses the same convention.
//! Within that: `ResnetBlock`'s shortcut conv is named `conv_shortcut`
//! (not the CompVis `nin_shortcut` name an earlier revision used — mixing
//! the two conventions inside one diffusers-shaped tree made every
//! shortcut-conv tensor name wrong for a real checkpoint); `Downsample`
//! reproduces diffusers' asymmetric `(0,1,0,1)` padding before an unpadded
//! stride-2 conv, rather than symmetric `padding=1` (the two select
//! different input pixels, so they are not numerically interchangeable even
//! when they happen to produce the same output size); and
//! `AttentionBlock` uses separate `to_q`/`to_k`/`to_v`/`to_out` 1×1
//! convolutions rather than one fused `to_qkv` (no released SD VAE
//! checkpoint — CompVis or diffusers — stores a fused QKV projection).
//!
//! This module still parameterises attention with 1×1 convolutions (as the
//! original CompVis LDM does) rather than diffusers' `Linear` layers over a
//! `(B, HW, C)` sequence view; the two are mathematically equivalent but
//! store weights in a different tensor rank, so loading real diffusers
//! attention weights into this module still requires a reshape step the
//! loader does not perform. Every other block (ResNet, GroupNorm,
//! Downsample, Upsample) is shape- and name-compatible with a real
//! diffusers `AutoencoderKL` checkpoint.

use candle_core::{Result, Tensor};
use candle_nn as nn;
use candle_nn::Module;

// ---------------------------------------------------------------------------
// Constants and pure helpers
// ---------------------------------------------------------------------------

/// SD 2.1 latent scaling factor (applied to the encoded mean before storing).
pub const SCALING_FACTOR: f32 = 0.18215;

/// Number of latent channels used by SD 2.1 (and most SD variants).
pub const LATENT_CHANNELS: usize = 4;

/// Spatial compression factor: the VAE down-samples each spatial dimension
/// by this factor during encoding (3 strided convolutions → 2^3 = 8).
pub const SPATIAL_COMPRESSION: usize = 8;

/// Convert an image side-length to the corresponding latent side-length.
///
/// This is a pure, allocation-free helper used in tests and memory estimation.
#[inline]
pub fn encode_latent_size(image_size: usize) -> usize {
    image_size / SPATIAL_COMPRESSION
}

// ---------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------

/// A ResNet block used in both encoder and decoder.
#[derive(Debug)]
struct ResnetBlock {
    norm1: nn::GroupNorm,
    conv1: nn::Conv2d,
    norm2: nn::GroupNorm,
    conv2: nn::Conv2d,
    residual_conv: Option<nn::Conv2d>,
}

impl ResnetBlock {
    fn new(vs: nn::VarBuilder, in_channels: usize, out_channels: usize) -> Result<Self> {
        let norm1 = nn::group_norm(32, in_channels, 1e-6, vs.pp("norm1"))?;
        let conv1 = nn::conv2d(
            in_channels,
            out_channels,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("conv1"),
        )?;
        let norm2 = nn::group_norm(32, out_channels, 1e-6, vs.pp("norm2"))?;
        let conv2 = nn::conv2d(
            out_channels,
            out_channels,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("conv2"),
        )?;
        let residual_conv = if in_channels != out_channels {
            Some(nn::conv2d(
                in_channels,
                out_channels,
                1,
                Default::default(),
                // diffusers name (not CompVis's "nin_shortcut") — matches
                // the diffusers-style block paths this module already uses
                // (encoder.down_blocks.{i}.resnets.{j}, ...) and
                // crate::upsampler's U-Net, which names its own shortcut
                // conv the same way.
                vs.pp("conv_shortcut"),
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            residual_conv,
        })
    }
}

impl Module for ResnetBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = if let Some(ref conv) = self.residual_conv {
            conv.forward(xs)?
        } else {
            xs.clone()
        };
        let h = self.norm1.forward(xs)?.silu()?;
        let h = self.conv1.forward(&h)?;
        let h = self.norm2.forward(&h)?.silu()?;
        let h = self.conv2.forward(&h)?;
        h + residual
    }
}

/// Self-attention block for the VAE mid-block.
///
/// Uses separate `to_q`/`to_k`/`to_v` 1×1 convolutions rather than one fused
/// `to_qkv` — no released SD VAE checkpoint (CompVis or diffusers) stores a
/// fused QKV projection, so a fused conv here could never load real weights
/// regardless of what it was named. `to_out` is named `to_out.0` to match
/// diffusers' `Attention` module, whose output projection is
/// `to_out = nn.ModuleList([Linear, Dropout])` (weights live under index 0).
#[derive(Debug)]
struct AttentionBlock {
    group_norm: nn::GroupNorm,
    to_q: nn::Conv2d,
    to_k: nn::Conv2d,
    to_v: nn::Conv2d,
    to_out: nn::Conv2d,
    channels: usize,
}

impl AttentionBlock {
    fn new(vs: nn::VarBuilder, channels: usize) -> Result<Self> {
        let group_norm = nn::group_norm(32, channels, 1e-6, vs.pp("group_norm"))?;
        let to_q = nn::conv2d(channels, channels, 1, Default::default(), vs.pp("to_q"))?;
        let to_k = nn::conv2d(channels, channels, 1, Default::default(), vs.pp("to_k"))?;
        let to_v = nn::conv2d(channels, channels, 1, Default::default(), vs.pp("to_v"))?;
        let to_out = nn::conv2d(channels, channels, 1, Default::default(), vs.pp("to_out.0"))?;
        Ok(Self {
            group_norm,
            to_q,
            to_k,
            to_v,
            to_out,
            channels,
        })
    }
}

impl Module for AttentionBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let (b, _c, h, w) = xs.dims4()?;
        let xs = self.group_norm.forward(xs)?;
        let q = self.to_q.forward(&xs)?.reshape((b, self.channels, h * w))?;
        let k = self.to_k.forward(&xs)?.reshape((b, self.channels, h * w))?;
        let v = self.to_v.forward(&xs)?.reshape((b, self.channels, h * w))?;

        let scale = (self.channels as f64).powf(-0.5);
        let attn = (q.transpose(1, 2)?.matmul(&k)? * scale)?;
        let attn = nn::ops::softmax_last_dim(&attn)?;
        let out = v.matmul(&attn.transpose(1, 2)?)?;
        let out = out.reshape((b, self.channels, h, w))?;
        let out = self.to_out.forward(&out)?;
        out + residual
    }
}

/// Downsample block (strided convolution).
///
/// diffusers' `Downsample2D` zero-pads asymmetrically — `F.pad(x, (0,1,0,1))`
/// (right column and bottom row only) — before an *unpadded* stride-2 3×3
/// conv, rather than a symmetric `padding=1`. The two select different
/// input pixels for each output position (a one-pixel spatial shift), so
/// they are not numerically interchangeable even on inputs where they
/// happen to produce the same output size (e.g. any even `H`/`W`, which
/// covers every latent size this module is actually used at).
#[derive(Debug)]
struct Downsample {
    conv: nn::Conv2d,
}

impl Downsample {
    fn new(vs: nn::VarBuilder, channels: usize) -> Result<Self> {
        let conv = nn::conv2d(
            channels,
            channels,
            3,
            nn::Conv2dConfig {
                stride: 2,
                padding: 0,
                ..Default::default()
            },
            vs.pp("conv"),
        )?;
        Ok(Self { conv })
    }
}

impl Module for Downsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Asymmetric (0,1,0,1) zero-pad: bottom row (dim 2, height) then
        // right column (dim 3, width), matching diffusers' Downsample2D —
        // not the symmetric padding=1 an unpadded stride-2 conv would need
        // to produce the same output size from a different set of pixels.
        let xs = xs.pad_with_zeros(2, 0, 1)?;
        let xs = xs.pad_with_zeros(3, 0, 1)?;
        self.conv.forward(&xs)
    }
}

/// Upsample block (nearest-neighbour interpolation + conv).
#[derive(Debug)]
struct Upsample {
    conv: nn::Conv2d,
}

impl Upsample {
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

impl Module for Upsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;
        let xs = xs.upsample_nearest2d(h * 2, w * 2)?;
        self.conv.forward(&xs)
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// VAE encoder: image → latent distribution parameters.
#[derive(Debug)]
struct Encoder {
    conv_in: nn::Conv2d,
    down_blocks: Vec<Vec<ResnetBlock>>,
    downsamplers: Vec<Option<Downsample>>,
    mid_block_1: ResnetBlock,
    mid_attn: AttentionBlock,
    mid_block_2: ResnetBlock,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2d,
}

impl Encoder {
    fn new(vs: nn::VarBuilder, in_channels: usize, latent_channels: usize) -> Result<Self> {
        let block_channels = [128, 256, 512, 512];
        let base_ch = block_channels[0];

        let conv_in = nn::conv2d(
            in_channels,
            base_ch,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("conv_in"),
        )?;

        let mut down_blocks = Vec::new();
        let mut downsamplers = Vec::new();
        let mut ch = base_ch;
        let vs_down = vs.pp("down_blocks");
        for (i, &out_ch) in block_channels.iter().enumerate() {
            let vs_block = vs_down.pp(i.to_string());
            let mut resnets = Vec::new();
            for j in 0..2 {
                let in_ch = if j == 0 { ch } else { out_ch };
                resnets.push(ResnetBlock::new(
                    vs_block.pp("resnets").pp(j.to_string()),
                    in_ch,
                    out_ch,
                )?);
            }
            ch = out_ch;
            down_blocks.push(resnets);
            if i < block_channels.len() - 1 {
                downsamplers.push(Some(Downsample::new(vs_block.pp("downsamplers.0"), ch)?));
            } else {
                downsamplers.push(None);
            }
        }

        let vs_mid = vs.pp("mid_block");
        let mid_block_1 = ResnetBlock::new(vs_mid.pp("resnets.0"), ch, ch)?;
        let mid_attn = AttentionBlock::new(vs_mid.pp("attentions.0"), ch)?;
        let mid_block_2 = ResnetBlock::new(vs_mid.pp("resnets.1"), ch, ch)?;

        let conv_norm_out = nn::group_norm(32, ch, 1e-6, vs.pp("conv_norm_out"))?;
        // Output 2× latent channels for mean + log_var
        let conv_out = nn::conv2d(
            ch,
            latent_channels * 2,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            down_blocks,
            downsamplers,
            mid_block_1,
            mid_attn,
            mid_block_2,
            conv_norm_out,
            conv_out,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(xs)?;

        for (resnets, ds) in self.down_blocks.iter().zip(self.downsamplers.iter()) {
            for resnet in resnets {
                h = resnet.forward(&h)?;
            }
            if let Some(ref downsample) = ds {
                h = downsample.forward(&h)?;
            }
        }

        h = self.mid_block_1.forward(&h)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_block_2.forward(&h)?;

        h = self.conv_norm_out.forward(&h)?.silu()?;
        self.conv_out.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// VAE decoder: latent → image.
#[derive(Debug)]
struct Decoder {
    conv_in: nn::Conv2d,
    mid_block_1: ResnetBlock,
    mid_attn: AttentionBlock,
    mid_block_2: ResnetBlock,
    up_blocks: Vec<Vec<ResnetBlock>>,
    upsamplers: Vec<Option<Upsample>>,
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2d,
}

impl Decoder {
    fn new(vs: nn::VarBuilder, latent_channels: usize, out_channels: usize) -> Result<Self> {
        let block_channels = [512, 512, 256, 128];
        let first_ch = block_channels[0];

        let conv_in = nn::conv2d(
            latent_channels,
            first_ch,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("conv_in"),
        )?;

        let vs_mid = vs.pp("mid_block");
        let mid_block_1 = ResnetBlock::new(vs_mid.pp("resnets.0"), first_ch, first_ch)?;
        let mid_attn = AttentionBlock::new(vs_mid.pp("attentions.0"), first_ch)?;
        let mid_block_2 = ResnetBlock::new(vs_mid.pp("resnets.1"), first_ch, first_ch)?;

        let mut up_blocks = Vec::new();
        let mut upsamplers = Vec::new();
        let mut ch = first_ch;
        let vs_up = vs.pp("up_blocks");
        for (i, &out_ch) in block_channels.iter().enumerate() {
            let vs_block = vs_up.pp(i.to_string());
            let mut resnets = Vec::new();
            for j in 0..3 {
                let in_ch = if j == 0 { ch } else { out_ch };
                resnets.push(ResnetBlock::new(
                    vs_block.pp("resnets").pp(j.to_string()),
                    in_ch,
                    out_ch,
                )?);
            }
            ch = out_ch;
            up_blocks.push(resnets);
            if i < block_channels.len() - 1 {
                upsamplers.push(Some(Upsample::new(vs_block.pp("upsamplers.0"), ch)?));
            } else {
                upsamplers.push(None);
            }
        }

        let conv_norm_out = nn::group_norm(32, ch, 1e-6, vs.pp("conv_norm_out"))?;
        let conv_out = nn::conv2d(
            ch,
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
            mid_block_1,
            mid_attn,
            mid_block_2,
            up_blocks,
            upsamplers,
            conv_norm_out,
            conv_out,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(xs)?;

        h = self.mid_block_1.forward(&h)?;
        h = self.mid_attn.forward(&h)?;
        h = self.mid_block_2.forward(&h)?;

        for (resnets, us) in self.up_blocks.iter().zip(self.upsamplers.iter()) {
            for resnet in resnets {
                h = resnet.forward(&h)?;
            }
            if let Some(ref upsample) = us {
                h = upsample.forward(&h)?;
            }
        }

        h = self.conv_norm_out.forward(&h)?.silu()?;
        self.conv_out.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// VAE public API
// ---------------------------------------------------------------------------

/// Variational Autoencoder for encoding/decoding between pixel and latent space.
#[derive(Debug)]
pub struct Vae {
    encoder: Encoder,
    decoder: Decoder,
    /// Learned post-quant convolution (1×1).
    quant_conv: nn::Conv2d,
    /// Learned pre-decode convolution (1×1).
    post_quant_conv: nn::Conv2d,
    /// Scaling factor for the latent space.
    scaling_factor: f64,
}

impl Vae {
    /// Load a VAE from a VarBuilder.
    pub fn new(vs: nn::VarBuilder, latent_channels: usize, scaling_factor: f64) -> Result<Self> {
        let encoder = Encoder::new(vs.pp("encoder"), 3, latent_channels)?;
        let decoder = Decoder::new(vs.pp("decoder"), latent_channels, 3)?;
        let quant_conv = nn::conv2d(
            latent_channels * 2,
            latent_channels * 2,
            1,
            Default::default(),
            vs.pp("quant_conv"),
        )?;
        let post_quant_conv = nn::conv2d(
            latent_channels,
            latent_channels,
            1,
            Default::default(),
            vs.pp("post_quant_conv"),
        )?;
        Ok(Self {
            encoder,
            decoder,
            quant_conv,
            post_quant_conv,
            scaling_factor,
        })
    }

    /// Encode an image to latent space (returns the mean of the posterior).
    ///
    /// - `image`: `(B, 3, H, W)` tensor in `[-1, 1]` range.
    ///
    /// Returns `(B, latent_channels, H/8, W/8)`.
    pub fn encode(&self, image: &Tensor) -> Result<Tensor> {
        let h = self.encoder.forward(image)?;
        let moments = self.quant_conv.forward(&h)?;
        let channels = moments.dim(1)? / 2;
        // Take the mean (first half of channels)
        let mean = moments.narrow(1, 0, channels)?;
        // Scale
        mean * self.scaling_factor
    }

    /// Decode a latent tensor back to pixel space.
    ///
    /// - `latents`: `(B, latent_channels, h, w)` scaled latent tensor.
    ///
    /// Returns `(B, 3, H, W)` in `[-1, 1]` range.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let z = (latents * (1.0 / self.scaling_factor))?;
        let z = self.post_quant_conv.forward(&z)?;
        self.decoder.forward(&z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    // 1. SCALING_FACTOR is the SD 2.1 standard value and is < 1.0
    #[test]
    fn scaling_factor_is_valid() {
        assert!((SCALING_FACTOR - 0.18215_f32).abs() < 1e-5);
        assert!(SCALING_FACTOR.abs() < 1.0);
    }

    // 2. LATENT_CHANNELS for SD 2.1 = 4
    #[test]
    fn latent_channels_is_four() {
        assert_eq!(LATENT_CHANNELS, 4);
    }

    // 3. Spatial compression factor 8: 512×512 image → 64×64 latents
    #[test]
    fn spatial_compression_512() {
        let latent = 512 / SPATIAL_COMPRESSION;
        assert_eq!(latent, 64);
    }

    // 4a. encode_latent_size(512) == 64
    #[test]
    fn encode_latent_size_512() {
        assert_eq!(encode_latent_size(512), 64);
    }

    // 4b. encode_latent_size(256) == 32
    #[test]
    fn encode_latent_size_256() {
        assert_eq!(encode_latent_size(256), 32);
    }

    // 4c. encode_latent_size(128) == 16
    #[test]
    fn encode_latent_size_128() {
        assert_eq!(encode_latent_size(128), 16);
    }

    // 5. scaling_factor * latent ∈ [−3,3] stays within [−0.55, 0.55] approximately
    #[test]
    fn scaled_latent_range() {
        let latent_min = -3.0_f32;
        let latent_max = 3.0_f32;
        let scaled_min = SCALING_FACTOR * latent_min;
        let scaled_max = SCALING_FACTOR * latent_max;
        assert!(
            scaled_min > -0.60,
            "scaled min should be > -0.60, got {scaled_min}"
        );
        assert!(
            scaled_max < 0.60,
            "scaled max should be < 0.60, got {scaled_max}"
        );
    }

    // 6. Decode scale ≈ 5.48 (1 / 0.18215 ≈ 5.4901)
    #[test]
    fn decode_scale_approx() {
        let decode_scale = 1.0_f32 / SCALING_FACTOR;
        assert!(
            (decode_scale - 5.48_f32).abs() < 0.02,
            "expected ~5.48, got {decode_scale}"
        );
    }

    // Additional: encode_latent_size is monotonic
    #[test]
    fn encode_latent_size_monotonic() {
        assert!(encode_latent_size(512) > encode_latent_size(256));
        assert!(encode_latent_size(256) > encode_latent_size(128));
    }

    // Additional: SPATIAL_COMPRESSION constant is 8
    #[test]
    fn spatial_compression_constant_is_eight() {
        assert_eq!(SPATIAL_COMPRESSION, 8);
    }

    /// Regression test for the weight-naming/attention/downsample fixes:
    /// run a full encode+decode pass so a real `VarBuilder` actually
    /// resolves every tensor this module constructs (`to_q`/`to_k`/`to_v`/
    /// `to_out.0`, `conv_shortcut`, the unpadded stride-2 `Downsample` conv)
    /// without shape errors, then check the registered tensor names
    /// directly to confirm the diffusers-style names are the ones that
    /// exist — not the fused `to_qkv` or CompVis `nin_shortcut` names an
    /// earlier revision used.
    #[test]
    fn vae_forward_pass_runs_and_registers_diffusers_style_names() -> Result<()> {
        let device = Device::Cpu;
        let varmap = nn::VarMap::new();
        let vb = nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let vae = Vae::new(vb, LATENT_CHANNELS, SCALING_FACTOR as f64)?;

        let image = Tensor::zeros((1, 3, 8, 8), DType::F32, &device)?;
        let latents = vae.encode(&image)?;
        assert_eq!(latents.dims4()?, (1, LATENT_CHANNELS, 1, 1));

        let decoded = vae.decode(&latents)?;
        assert_eq!(decoded.dims4()?, (1, 3, 8, 8));

        let names: Vec<String> = varmap.data().lock().unwrap().keys().cloned().collect();
        let has_suffix = |suffix: &str| names.iter().any(|n| n.ends_with(suffix));

        assert!(has_suffix("to_q.weight"), "no to_q tensor registered");
        assert!(has_suffix("to_k.weight"), "no to_k tensor registered");
        assert!(has_suffix("to_v.weight"), "no to_v tensor registered");
        assert!(
            has_suffix("to_out.0.weight"),
            "no to_out.0 tensor registered"
        );
        assert!(
            !names.iter().any(|n| n.contains("to_qkv")),
            "a fused to_qkv tensor should not exist, got names: {names:?}"
        );
        assert!(
            has_suffix("conv_shortcut.weight"),
            "no conv_shortcut tensor registered (expected on the first \
             channel-changing ResnetBlock)"
        );
        assert!(
            !names.iter().any(|n| n.contains("nin_shortcut")),
            "the CompVis nin_shortcut name should not exist, got names: {names:?}"
        );
        Ok(())
    }
}
