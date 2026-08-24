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
    /// Build one decoder stage.
    ///
    /// `skip_channels` carries the width of the skip connection consumed by
    /// each resnet, in the order they are consumed. Diffusers'
    /// `CrossAttnUpBlock2D` does **not** use a single width for the whole
    /// stage: the first `num_layers - 1` resnets receive a skip of
    /// `out_ch` channels while the last receives the block's own
    /// `in_channels` (the next stage's width). `skip_channels.len()` must
    /// equal `num_layers`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        vs: nn::VarBuilder,
        in_ch: usize,
        out_ch: usize,
        skip_channels: &[usize],
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
        if skip_channels.len() != num_layers {
            return Err(candle_core::Error::Msg(format!(
                "UpBlock expects one skip width per layer: got {} widths for {num_layers} layers",
                skip_channels.len()
            )));
        }

        let vs_res = vs.pp("resnets");
        let mut resnets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            // Resnet input = previous stage output (first layer only) or this
            // stage's own output, concatenated with that layer's skip.
            let resnet_in = if i == 0 { in_ch } else { out_ch };
            let ich = resnet_in + skip_channels[i];
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
// Config-derived geometry helpers
// ---------------------------------------------------------------------------

/// Resolve `(num_heads, head_dim)` for one attention stage.
///
/// [`DiffusionConfig::attention_head_dim`] follows the diffusers SD 2.1 UNet
/// config, where the vector `[5, 10, 20, 20]` holds the **number of attention
/// heads** per stage and the per-head dimension is derived as
/// `stage_channels / num_heads` (64 for every SD 2.1 stage). Reading the vector
/// as the head *dimension* instead yields 64 heads of dim 5, which keeps
/// `inner_dim` — and therefore the projection weight shapes — unchanged while
/// computing attention over the wrong head partition and with a `1/sqrt(5)`
/// scale instead of `1/sqrt(64)`.
///
/// `DiffusionConfig` is user-constructible and never validated, so the vector
/// length, a zero head count and a non-divisible channel count are all
/// reported as errors rather than panicking.
fn resolve_attention_heads(
    config: &DiffusionConfig,
    stage: usize,
    stage_channels: usize,
) -> Result<(usize, usize)> {
    let n_heads = config
        .attention_head_dim
        .get(stage)
        .copied()
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "attention_head_dim has {} entries but stage {stage} was requested",
                config.attention_head_dim.len()
            ))
        })?;
    if n_heads == 0 {
        return Err(candle_core::Error::Msg(format!(
            "attention_head_dim[{stage}] must be non-zero"
        )));
    }
    if stage_channels % n_heads != 0 {
        return Err(candle_core::Error::Msg(format!(
            "stage {stage} has {stage_channels} channels, which is not divisible by \
             attention_head_dim[{stage}] = {n_heads}"
        )));
    }
    Ok((n_heads, stage_channels / n_heads))
}

/// Number of transformer blocks for one attention stage.
///
/// Guards the same user-supplied-vector indexing panic as
/// [`resolve_attention_heads`].
fn resolve_transformer_depth(config: &DiffusionConfig, stage: usize) -> Result<usize> {
    config
        .transformer_layers_per_block
        .get(stage)
        .copied()
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "transformer_layers_per_block has {} entries but stage {stage} was requested",
                config.transformer_layers_per_block.len()
            ))
        })
}

/// Skip-connection widths consumed by one decoder stage, in consumption order.
///
/// Mirrors diffusers' `CrossAttnUpBlock2D`:
/// `res_skip = out_channels` for the first `num_layers - 1` resnets and
/// `res_skip = block_in_channels` for the last, where `block_in_channels` is
/// the *next* (lower-resolution-index) stage's channel count.
fn up_block_skip_channels(out_ch: usize, block_in_ch: usize, num_layers: usize) -> Vec<usize> {
    (0..num_layers)
        .map(|layer| {
            if layer + 1 == num_layers {
                block_in_ch
            } else {
                out_ch
            }
        })
        .collect()
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
    ///
    /// # Errors
    ///
    /// `DiffusionConfig` is user-constructible and never validated, so this
    /// reports a mis-shaped config (no stages, a stage vector shorter than
    /// `channel_mult`, a zero or non-dividing head count) as an error rather
    /// than panicking.
    pub fn new(vs: nn::VarBuilder, config: &DiffusionConfig) -> Result<Self> {
        let base = config.base_channels;
        let time_embed_dim = config.time_embed_dim;

        // `num_stages - 1` indexes the deepest stage throughout; an empty
        // `channel_mult` would underflow it.
        if config.num_stages() == 0 {
            return Err(candle_core::Error::Msg(
                "channel_mult must declare at least one U-Net stage".to_string(),
            ));
        }

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
            let (n_heads, d_head) = resolve_attention_heads(config, i, output_ch)?;
            let depth = resolve_transformer_depth(config, i)?;
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
        let (mid_n_heads, mid_d_head) = resolve_attention_heads(config, num_stages - 1, last_ch)?;
        let mid_depth = resolve_transformer_depth(config, num_stages - 1)?;
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
        // Each decoder stage consumes `layers_per_block + 1` skips: one per
        // encoder resnet of the mirrored stage plus the stage's downsample
        // output (the shallowest stage instead re-uses the `conv_in` output).
        let up_layers = config.layers_per_block + 1;
        for i in 0..num_stages {
            let output_ch = reversed_channels[i];
            // Diffusers' `input_channel` for this up block: the channel count
            // of the next (shallower) stage, clamped at the last entry.
            let block_in_ch = reversed_channels[(i + 1).min(num_stages - 1)];
            let skip_channels = up_block_skip_channels(output_ch, block_in_ch, up_layers);
            let stage_idx = num_stages - 1 - i;
            let (n_heads, d_head) = resolve_attention_heads(config, stage_idx, output_ch)?;
            let depth = resolve_transformer_depth(config, stage_idx)?;
            let has_us = i < num_stages - 1;

            up_blocks.push(UpBlock::new(
                vs_up.pp(i.to_string()),
                prev_ch,
                output_ch,
                &skip_channels,
                time_embed_dim,
                up_layers, // +1 for the conv_in / downsample skip
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
    /// Returns `DiffusionError::SkipConnectionUnderflow` if the encoder produced
    /// fewer skip connections than the decoder consumes, and
    /// `DiffusionError::Inference` if it produced more. Both indicate a
    /// mis-wired configuration rather than bad input.
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

        // 4. Down blocks — collect skip connections.
        //
        // The `conv_in` output is itself a skip: diffusers seeds
        // `down_block_res_samples = (sample,)` before the first down block.
        // Without it the encoder yields exactly one skip fewer than the decoder
        // pops, and the last resnet of the shallowest up block underflows.
        let mut all_skips: Vec<Tensor> = vec![h.clone()];
        for down in &self.down_blocks {
            let (out, skips) = down.forward(&h, &emb, context, ip_tokens)?;
            h = out;
            all_skips.extend(skips);
        }

        // Encoder and decoder must agree on the skip budget: one per resnet of
        // every up block.
        let expected_skips: usize = self.up_blocks.iter().map(|up| up.resnets.len()).sum();
        if all_skips.len() < expected_skips {
            return Err(DiffusionError::SkipConnectionUnderflow {
                expected: expected_skips,
                available: all_skips.len(),
            });
        }
        if all_skips.len() > expected_skips {
            return Err(DiffusionError::Inference(format!(
                "skip connection surplus: encoder produced {} skips but the decoder \
                 consumes only {expected_skips}",
                all_skips.len()
            )));
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// A miniature but structurally faithful U-Net config.
    ///
    /// `ResBlock` group-normalises with 32 groups, so every channel width the
    /// decoder concatenates has to stay a multiple of 32; base 32 with
    /// `channel_mult = [1, 2]` is the smallest configuration that satisfies it.
    fn tiny_config() -> DiffusionConfig {
        DiffusionConfig {
            num_views: 1,
            base_channels: 32,
            channel_mult: vec![1, 2],
            layers_per_block: 1,
            attention_head_dim: vec![2, 4],
            transformer_layers_per_block: vec![1, 1],
            time_embed_dim: 64,
            cross_attention_dim: 32,
            clip_embed_dim: 16,
            unet_in_channels: 4,
            unet_out_channels: 4,
            image_size: 64,
            latent_size: 8,
            ..DiffusionConfig::default()
        }
    }

    // ------------------------------------------------------------------
    // Attention head partition
    // ------------------------------------------------------------------

    #[test]
    fn test_attention_head_dim_is_read_as_head_count() -> Result<()> {
        // SD 2.1 lists [5, 10, 20, 20] as the *head count*; every stage then
        // has a 64-wide head, not 5.
        let config = DiffusionConfig::default();
        let expected = [(5usize, 64usize), (10, 64), (20, 64), (20, 64)];
        for (stage, &(want_heads, want_dim)) in expected.iter().enumerate() {
            let channels = config.stage_channels(stage);
            let (heads, dim) = resolve_attention_heads(&config, stage, channels)?;
            assert_eq!(heads, want_heads, "stage {stage} head count");
            assert_eq!(dim, want_dim, "stage {stage} head dim");
            assert_eq!(heads * dim, channels, "inner_dim must equal stage channels");
        }
        Ok(())
    }

    #[test]
    fn test_resolve_attention_heads_rejects_zero_head_count() {
        let config = DiffusionConfig {
            attention_head_dim: vec![0, 10, 20, 20],
            ..DiffusionConfig::default()
        };
        assert!(resolve_attention_heads(&config, 0, 320).is_err());
    }

    #[test]
    fn test_resolve_attention_heads_rejects_short_vector() {
        let config = DiffusionConfig {
            attention_head_dim: vec![5],
            ..DiffusionConfig::default()
        };
        assert!(resolve_attention_heads(&config, 2, 1280).is_err());
    }

    #[test]
    fn test_resolve_attention_heads_rejects_indivisible_channels() {
        let config = DiffusionConfig {
            attention_head_dim: vec![7, 10, 20, 20],
            ..DiffusionConfig::default()
        };
        assert!(resolve_attention_heads(&config, 0, 320).is_err());
    }

    #[test]
    fn test_new_rejects_empty_channel_mult() {
        let varmap = nn::VarMap::new();
        let vs = nn::VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let config = DiffusionConfig {
            channel_mult: Vec::new(),
            ..tiny_config()
        };
        // `num_stages - 1` would underflow rather than report the bad config.
        assert!(MultiViewUNet::new(vs, &config).is_err());
    }

    #[test]
    fn test_resolve_transformer_depth_rejects_short_vector() {
        let config = DiffusionConfig {
            transformer_layers_per_block: vec![1, 1],
            ..DiffusionConfig::default()
        };
        assert!(resolve_transformer_depth(&config, 3).is_err());
        assert_eq!(resolve_transformer_depth(&config, 1).unwrap_or(0), 1);
    }

    // ------------------------------------------------------------------
    // Skip-connection wiring
    // ------------------------------------------------------------------

    #[test]
    fn test_up_block_skip_channels_are_per_layer() {
        // Diffusers: `out_channels` for the first num_layers-1 resnets, then
        // the block's own `in_channels`.
        assert_eq!(up_block_skip_channels(1280, 640, 3), vec![1280, 1280, 640]);
        assert_eq!(up_block_skip_channels(640, 320, 3), vec![640, 640, 320]);
        assert_eq!(up_block_skip_channels(320, 320, 3), vec![320, 320, 320]);
        assert_eq!(up_block_skip_channels(64, 32, 2), vec![64, 32]);
    }

    #[test]
    fn test_default_config_skip_budget_balances() {
        // Encoder must emit exactly as many skips as the decoder pops; the
        // conv_in output is what used to be missing.
        let config = DiffusionConfig::default();
        let num_stages = config.num_stages();

        let mut produced = 1; // conv_in
        for i in 0..num_stages {
            produced += config.layers_per_block;
            if i < num_stages - 1 {
                produced += 1; // downsample output
            }
        }
        let consumed = num_stages * (config.layers_per_block + 1);

        assert_eq!(produced, 12);
        assert_eq!(consumed, 12);
    }

    #[test]
    fn test_default_config_skip_widths_line_up() {
        let config = DiffusionConfig::default();
        let num_stages = config.num_stages();

        // Encoder skip stack, bottom → top.
        let mut stack: Vec<usize> = vec![config.base_channels]; // conv_in output
        for i in 0..num_stages {
            let out_ch = config.stage_channels(i);
            for _ in 0..config.layers_per_block {
                stack.push(out_ch);
            }
            if i < num_stages - 1 {
                stack.push(out_ch); // downsample output
            }
        }
        assert_eq!(
            stack,
            vec![320, 320, 320, 320, 640, 640, 640, 1280, 1280, 1280, 1280, 1280]
        );

        // The decoder pops LIFO and must find exactly the width each resnet
        // was declared for.
        let reversed: Vec<usize> = (0..num_stages)
            .rev()
            .map(|i| config.stage_channels(i))
            .collect();
        let up_layers = config.layers_per_block + 1;
        for i in 0..num_stages {
            let out_ch = reversed[i];
            let block_in_ch = reversed[(i + 1).min(num_stages - 1)];
            for want in up_block_skip_channels(out_ch, block_in_ch, up_layers) {
                let got = stack.pop().expect("skip stack must not underflow");
                assert_eq!(got, want, "up block {i} skip width mismatch");
            }
        }
        assert!(stack.is_empty(), "every encoder skip must be consumed");
    }

    #[test]
    fn test_up_block_rejects_mismatched_skip_widths() {
        let varmap = nn::VarMap::new();
        let vs = nn::VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        // 2 skip widths supplied for 3 layers.
        let result = UpBlock::new(
            vs.pp("up"),
            64,
            64,
            &[64, 32],
            64,
            3,
            false,
            2,
            32,
            1,
            32,
            16,
            1,
            32,
            true,
            false,
        );
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // End-to-end forward pass
    // ------------------------------------------------------------------

    #[test]
    fn test_tiny_unet_forward_runs_end_to_end() -> Result<()> {
        let device = Device::Cpu;
        let varmap = nn::VarMap::new();
        let vs = nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = tiny_config();
        let unet = MultiViewUNet::new(vs, &config)?;

        // The decoder consumes one skip per up-block resnet.
        let consumed: usize = unet.up_blocks.iter().map(|up| up.resnets.len()).sum();
        assert_eq!(
            consumed,
            config.num_stages() * (config.layers_per_block + 1)
        );

        let bv = config.num_views; // batch 1 × num_views
        let sample = Tensor::randn(0f32, 1f32, (bv, config.unet_in_channels, 8, 8), &device)?;
        let context = Tensor::randn(0f32, 1f32, (bv, 2, config.cross_attention_dim), &device)?;
        let poses = Tensor::randn(0f32, 1f32, (bv, config.camera_pose_dim), &device)?;
        let ip_tokens = Tensor::randn(0f32, 1f32, (bv, 3, config.clip_embed_dim), &device)?;

        let out = unet
            .forward(&sample, 10, Some(&context), Some(&poses), Some(&ip_tokens))
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;

        assert_eq!(out.dims(), &[bv, config.unet_out_channels, 8, 8]);
        Ok(())
    }

    #[test]
    fn test_tiny_unet_forward_without_ip_tokens() -> Result<()> {
        // The CFG unconditional pass skips the IP-adapter layer entirely.
        let device = Device::Cpu;
        let varmap = nn::VarMap::new();
        let vs = nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = tiny_config();
        let unet = MultiViewUNet::new(vs, &config)?;

        let bv = config.num_views;
        let sample = Tensor::randn(0f32, 1f32, (bv, config.unet_in_channels, 8, 8), &device)?;
        let context = Tensor::randn(0f32, 1f32, (bv, 2, config.cross_attention_dim), &device)?;

        let out = unet
            .forward(&sample, 0, Some(&context), None, None)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;

        assert_eq!(out.dims(), &[bv, config.unet_out_channels, 8, 8]);
        Ok(())
    }
}
