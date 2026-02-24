//! Multi-view U-Net with camera-conditioned cross-view attention.
//!
//! The U-Net follows the SD 2.1 architecture but replaces every spatial
//! transformer block with a `MultiViewSpatialTransformer` that adds:
//!
//! 1. **Cross-view attention**: Allows spatial positions to attend across all views
//! 2. **IP-Adapter conditioning**: Dedicated cross-attention layer (`attn_ip`) that
//!    conditions on CLIP image embeddings from the reference photo
//! 3. **Camera-pose conditioning**: Camera extrinsics added to timestep embedding
//!
//! ## IP-Adapter Integration
//!
//! Each transformer block contains four attention layers:
//! - `attn1`: Self-attention (within view)
//! - `attn_cv`: Cross-view attention (across views)
//! - `attn2`: Text cross-attention (unused in GAF, always zero)
//! - `attn_ip`: IP-Adapter cross-attention (reference image conditioning)
//!
//! When `ip_tokens` is `None` (unconditional pass), the `attn_ip` layer is
//! skipped entirely, producing the unconditional prediction for CFG.
//!
//! ## Architecture Details
//!
//! The U-Net structure:
//! - **Encoder**: 4 downsampling stages (320 → 640 → 1280 → 1280 channels)
//! - **Bottleneck**: ResBlock + Attention + ResBlock at 1280 channels
//! - **Decoder**: 4 upsampling stages with skip connections
//! - **Output**: GroupNorm + Conv → 4-channel latent prediction
//!
//! Each stage contains 2 ResBlocks + 1 MultiViewSpatialTransformer (if attention enabled).

use candle_core::{Result, Tensor};
use candle_nn as nn;
use candle_nn::Module;

use crate::attention::MultiViewSpatialTransformer;
use crate::camera::{timestep_embedding, CameraEmbedding, TimestepEmbedding};
use crate::config::DiffusionConfig;
use crate::DiffusionError;

// ---------------------------------------------------------------------------
// Building blocks
// ---------------------------------------------------------------------------

/// ResNet block with time-step conditioning.
#[derive(Debug)]
struct ResBlock {
    norm1: nn::GroupNorm,
    conv1: nn::Conv2d,
    time_emb_proj: nn::Linear,
    norm2: nn::GroupNorm,
    conv2: nn::Conv2d,
    residual_conv: Option<nn::Conv2d>,
}

impl ResBlock {
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

        // Add time embedding: project then unsqueeze spatial dims
        let t = self.time_emb_proj.forward(&time_emb.silu()?)?;
        let t = t.unsqueeze(2)?.unsqueeze(3)?;
        let h = (h.clone() + t.broadcast_as(h.shape())?)?;

        let h = self.norm2.forward(&h)?.silu()?;
        let h = self.conv2.forward(&h)?;
        h + residual
    }
}

/// Downsample with strided convolution.
#[derive(Debug)]
struct Downsample2d {
    conv: nn::Conv2d,
}

impl Downsample2d {
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

impl Module for Downsample2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.conv.forward(xs)
    }
}

/// Upsample with nearest-neighbor interpolation + conv.
#[derive(Debug)]
struct Upsample2d {
    conv: nn::Conv2d,
}

impl Upsample2d {
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

impl Module for Upsample2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;
        let xs = xs.upsample_nearest2d(h * 2, w * 2)?;
        self.conv.forward(&xs)
    }
}

// ---------------------------------------------------------------------------
// Down / Mid / Up blocks
// ---------------------------------------------------------------------------

/// A single downsampling stage with ResBlocks + optional spatial transformer.
#[derive(Debug)]
struct DownBlock {
    resnets: Vec<ResBlock>,
    attentions: Vec<MultiViewSpatialTransformer>,
    downsample: Option<Downsample2d>,
}

impl DownBlock {
    #[allow(clippy::too_many_arguments)]
    fn new(
        vs: nn::VarBuilder,
        in_ch: usize,
        out_ch: usize,
        time_dim: usize,
        num_layers: usize,
        has_attn: bool,
        n_heads: usize,
        d_head: usize,
        depth: usize,
        context_dim: usize,
        ip_dim: usize,
        num_views: usize,
        num_groups: usize,
        use_linear: bool,
        has_downsample: bool,
    ) -> Result<Self> {
        let vs_res = vs.pp("resnets");
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let ich = if i == 0 { in_ch } else { out_ch };
            resnets.push(ResBlock::new(
                vs_res.pp(i.to_string()),
                ich,
                out_ch,
                time_dim,
            )?);
        }

        let mut attentions = Vec::new();
        if has_attn {
            let vs_attn = vs.pp("attentions");
            for i in 0..num_layers {
                attentions.push(MultiViewSpatialTransformer::new(
                    vs_attn.pp(i.to_string()),
                    out_ch,
                    n_heads,
                    d_head,
                    depth,
                    context_dim,
                    ip_dim,
                    num_views,
                    num_groups,
                    use_linear,
                )?);
            }
        }

        let downsample = if has_downsample {
            Some(Downsample2d::new(vs.pp("downsamplers.0"), out_ch)?)
        } else {
            None
        };

        Ok(Self {
            resnets,
            attentions,
            downsample,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        time_emb: &Tensor,
        context: Option<&Tensor>,
        ip_tokens: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut h = xs.clone();
        let mut skip_connections = Vec::new();

        for (i, resnet) in self.resnets.iter().enumerate() {
            h = resnet.forward(&h, time_emb)?;
            if !self.attentions.is_empty() {
                h = self.attentions[i].forward(&h, context, ip_tokens)?;
            }
            skip_connections.push(h.clone());
        }

        if let Some(ref ds) = self.downsample {
            h = ds.forward(&h)?;
            skip_connections.push(h.clone());
        }

        Ok((h, skip_connections))
    }
}

/// Mid-block: ResBlock + attention + ResBlock.
#[derive(Debug)]
struct MidBlock {
    resnet1: ResBlock,
    attention: MultiViewSpatialTransformer,
    resnet2: ResBlock,
}

impl MidBlock {
    #[allow(clippy::too_many_arguments)]
    fn new(
        vs: nn::VarBuilder,
        channels: usize,
        time_dim: usize,
        n_heads: usize,
        d_head: usize,
        depth: usize,
        context_dim: usize,
        ip_dim: usize,
        num_views: usize,
        num_groups: usize,
        use_linear: bool,
    ) -> Result<Self> {
        let resnet1 = ResBlock::new(vs.pp("resnets.0"), channels, channels, time_dim)?;
        let attention = MultiViewSpatialTransformer::new(
            vs.pp("attentions.0"),
            channels,
            n_heads,
            d_head,
            depth,
            context_dim,
            ip_dim,
            num_views,
            num_groups,
            use_linear,
        )?;
        let resnet2 = ResBlock::new(vs.pp("resnets.1"), channels, channels, time_dim)?;
        Ok(Self {
            resnet1,
            attention,
            resnet2,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        time_emb: &Tensor,
        context: Option<&Tensor>,
        ip_tokens: Option<&Tensor>,
    ) -> Result<Tensor> {
        let h = self.resnet1.forward(xs, time_emb)?;
        let h = self.attention.forward(&h, context, ip_tokens)?;
        self.resnet2.forward(&h, time_emb)
    }
}

/// A single upsampling stage with ResBlocks + optional spatial transformer.
#[derive(Debug)]
struct UpBlock {
    resnets: Vec<ResBlock>,
    attentions: Vec<MultiViewSpatialTransformer>,
    upsample: Option<Upsample2d>,
}

impl UpBlock {
    #[allow(clippy::too_many_arguments)]
    fn new(
        vs: nn::VarBuilder,
        in_ch: usize,
        out_ch: usize,
        skip_ch: usize,
        time_dim: usize,
        num_layers: usize,
        has_attn: bool,
        n_heads: usize,
        d_head: usize,
        depth: usize,
        context_dim: usize,
        ip_dim: usize,
        num_views: usize,
        num_groups: usize,
        use_linear: bool,
        has_upsample: bool,
    ) -> Result<Self> {
        let vs_res = vs.pp("resnets");
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let ich = if i == 0 {
                in_ch + skip_ch
            } else {
                out_ch + skip_ch
            };
            resnets.push(ResBlock::new(
                vs_res.pp(i.to_string()),
                ich,
                out_ch,
                time_dim,
            )?);
        }

        let mut attentions = Vec::new();
        if has_attn {
            let vs_attn = vs.pp("attentions");
            for i in 0..num_layers {
                attentions.push(MultiViewSpatialTransformer::new(
                    vs_attn.pp(i.to_string()),
                    out_ch,
                    n_heads,
                    d_head,
                    depth,
                    context_dim,
                    ip_dim,
                    num_views,
                    num_groups,
                    use_linear,
                )?);
            }
        }

        let upsample = if has_upsample {
            Some(Upsample2d::new(vs.pp("upsamplers.0"), out_ch)?)
        } else {
            None
        };

        Ok(Self {
            resnets,
            attentions,
            upsample,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        time_emb: &Tensor,
        skip_connections: &mut Vec<Tensor>,
        context: Option<&Tensor>,
        ip_tokens: Option<&Tensor>,
    ) -> std::result::Result<Tensor, DiffusionError> {
        let mut h = xs.clone();

        for (i, resnet) in self.resnets.iter().enumerate() {
            let skip =
                skip_connections
                    .pop()
                    .ok_or_else(|| DiffusionError::SkipConnectionUnderflow {
                        expected: self.resnets.len(),
                        available: i,
                    })?;
            h = Tensor::cat(&[h, skip], 1)?;
            h = resnet.forward(&h, time_emb)?;
            if !self.attentions.is_empty() {
                h = self.attentions[i].forward(&h, context, ip_tokens)?;
            }
        }

        if let Some(ref us) = self.upsample {
            h = us.forward(&h)?;
        }

        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// Multi-view U-Net
// ---------------------------------------------------------------------------

/// The multi-view U-Net for diffusion-based avatar generation.
///
/// Architecture matches SD 2.1 but with multi-view cross-attention in every
/// spatial transformer block, camera-pose conditioning added to the timestep
/// embedding, and IP-adapter cross-attention for reference-image conditioning.
#[derive(Debug)]
pub struct MultiViewUNet {
    /// Input convolution: in_channels → base_channels.
    conv_in: nn::Conv2d,
    /// Sinusoidal → MLP time embedding.
    time_embedding: TimestepEmbedding,
    /// Camera-pose → time-embedding-dim MLP.
    camera_embedding: CameraEmbedding,
    /// Downsampling stages.
    down_blocks: Vec<DownBlock>,
    /// Bottleneck.
    mid_block: MidBlock,
    /// Upsampling stages.
    up_blocks: Vec<UpBlock>,
    /// Output: GroupNorm + conv → out_channels.
    conv_norm_out: nn::GroupNorm,
    conv_out: nn::Conv2d,
    /// Model config.
    config: DiffusionConfig,
}

impl MultiViewUNet {
    /// Build the U-Net from a DiffusionConfig and VarBuilder.
    pub fn new(vs: nn::VarBuilder, config: &DiffusionConfig) -> Result<Self> {
        let base = config.base_channels;
        let time_embed_dim = config.time_embed_dim;

        // Input conv
        let conv_in = nn::conv2d(
            config.unet_in_channels,
            base,
            3,
            nn::Conv2dConfig {
                padding: 1,
                ..Default::default()
            },
            vs.pp("conv_in"),
        )?;

        // Time embedding
        let time_embedding = TimestepEmbedding::new(vs.pp("time_embedding"), base, time_embed_dim)?;

        // Camera embedding
        let camera_embedding = CameraEmbedding::new(
            vs.pp("camera_embedding"),
            config.camera_pose_dim,
            time_embed_dim,
        )?;

        // Down blocks
        let mut down_blocks = Vec::new();
        let num_stages = config.num_stages();
        let vs_down = vs.pp("down_blocks");
        let mut input_ch = base;
        for i in 0..num_stages {
            let output_ch = config.stage_channels(i);
            let n_heads = output_ch / config.attention_head_dim[i];
            let d_head = config.attention_head_dim[i];
            let depth = config.transformer_layers_per_block[i];
            let has_ds = i < num_stages - 1;

            down_blocks.push(DownBlock::new(
                vs_down.pp(i.to_string()),
                input_ch,
                output_ch,
                time_embed_dim,
                config.layers_per_block,
                true, // all stages have attention in SD 2.1 768-v
                n_heads,
                d_head,
                depth,
                config.cross_attention_dim,
                config.clip_embed_dim,
                config.num_views,
                config.norm_num_groups,
                config.use_linear_projection,
                has_ds,
            )?);
            input_ch = output_ch;
        }

        // Mid block
        let last_ch = config.stage_channels(num_stages - 1);
        let mid_n_heads = last_ch / config.attention_head_dim[num_stages - 1];
        let mid_d_head = config.attention_head_dim[num_stages - 1];
        let mid_depth = config.transformer_layers_per_block[num_stages - 1];
        let mid_block = MidBlock::new(
            vs.pp("mid_block"),
            last_ch,
            time_embed_dim,
            mid_n_heads,
            mid_d_head,
            mid_depth,
            config.cross_attention_dim,
            config.clip_embed_dim,
            config.num_views,
            config.norm_num_groups,
            config.use_linear_projection,
        )?;

        // Up blocks (reverse order)
        let mut up_blocks = Vec::new();
        let vs_up = vs.pp("up_blocks");
        let reversed_channels: Vec<usize> = (0..num_stages)
            .rev()
            .map(|i| config.stage_channels(i))
            .collect();
        let mut prev_ch = last_ch;
        for i in 0..num_stages {
            let output_ch = reversed_channels[i];
            let skip_ch = if i == 0 {
                last_ch
            } else {
                reversed_channels[i - 1]
            };
            let stage_idx = num_stages - 1 - i;
            let n_heads = output_ch / config.attention_head_dim[stage_idx];
            let d_head = config.attention_head_dim[stage_idx];
            let depth = config.transformer_layers_per_block[stage_idx];
            let has_us = i < num_stages - 1;

            up_blocks.push(UpBlock::new(
                vs_up.pp(i.to_string()),
                prev_ch,
                output_ch,
                skip_ch,
                time_embed_dim,
                config.layers_per_block + 1, // +1 for the skip connection layer
                true,
                n_heads,
                d_head,
                depth,
                config.cross_attention_dim,
                config.clip_embed_dim,
                config.num_views,
                config.norm_num_groups,
                config.use_linear_projection,
                has_us,
            )?);
            prev_ch = output_ch;
        }

        // Output
        let conv_norm_out = nn::group_norm(
            config.norm_num_groups,
            base,
            config.norm_eps,
            vs.pp("conv_norm_out"),
        )?;
        let conv_out = nn::conv2d(
            base,
            config.unet_out_channels,
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
            camera_embedding,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
            config: config.clone(),
        })
    }

    /// Forward pass.
    ///
    /// - `sample`: `(B*V, in_channels, H, W)` noisy latent input.
    /// - `timestep`: scalar timestep (will be broadcast to batch).
    /// - `context`: `(B*V, seq_len, cross_attn_dim)` text/null embedding.
    /// - `camera_poses`: `(B*V, pose_dim)` flattened extrinsics.
    /// - `ip_tokens`: `(B*V, ip_len, ip_dim)` CLIP image tokens.
    ///
    /// # Errors
    ///
    /// Returns `DiffusionError::SkipConnectionUnderflow` if skip connections are
    /// exhausted before all up blocks have consumed them.
    pub fn forward(
        &self,
        sample: &Tensor,
        timestep: usize,
        context: Option<&Tensor>,
        camera_poses: Option<&Tensor>,
        ip_tokens: Option<&Tensor>,
    ) -> std::result::Result<Tensor, DiffusionError> {
        let batch_size = sample.dim(0)?;
        let device = sample.device();

        // 1. Time embedding
        let t_emb = timestep_embedding(
            &Tensor::full(timestep as f32, (batch_size,), device)?,
            self.config.base_channels,
        )?;
        let mut emb = self.time_embedding.forward(&t_emb)?;

        // 2. Add camera embedding if provided
        if let Some(cam) = camera_poses {
            let cam_emb = self.camera_embedding.forward(cam)?;
            emb = (emb + cam_emb)?;
        }

        // 3. Input conv
        let mut h = self.conv_in.forward(sample)?;

        // 4. Down blocks — collect skip connections
        let mut all_skips: Vec<Tensor> = Vec::new();
        for down in &self.down_blocks {
            let (out, skips) = down.forward(&h, &emb, context, ip_tokens)?;
            h = out;
            all_skips.extend(skips);
        }

        // 5. Mid block
        h = self.mid_block.forward(&h, &emb, context, ip_tokens)?;

        // 6. Up blocks — consume skip connections
        for up in &self.up_blocks {
            h = up.forward(&h, &emb, &mut all_skips, context, ip_tokens)?;
        }

        // 7. Output
        h = self.conv_norm_out.forward(&h)?.silu()?;
        Ok(self.conv_out.forward(&h)?)
    }
}
