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
//! `attn_ip`'s context width is
//! [`DiffusionConfig::ip_adapter_context_dim`] — the width
//! [`crate::clip::ClipImageEncoder`] projects its output to, *not* the CLIP
//! tower's own `clip_embed_dim`. IP-Adapter puts a projection between the two,
//! so `clip_embed_dim` is that projection's input and this its output.
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

use std::sync::Arc;

use candle_core::{DType, Result, Tensor};
use candle_nn as nn;
use candle_nn::Module;

use crate::attention::{AttentionSpec, MultiViewSpatialTransformer, SpatialTransformerSpec};
use crate::camera::{timestep_embedding, CameraEmbedding, TimestepEmbedding};
use crate::config::DiffusionConfig;
use crate::controlnet::ControlNetProcessor;
use crate::kv_cache::KVCache;
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
    /// Build one encoder stage.
    ///
    /// `attention` is `Some` when this stage has spatial-transformer layers;
    /// it carries the layer geometry *and* the kernel selection derived from
    /// [`DiffusionConfig`], so the configured attention backend actually
    /// reaches the attention layers.
    fn new(
        vs: nn::VarBuilder,
        in_ch: usize,
        out_ch: usize,
        time_dim: usize,
        num_layers: usize,
        attention: Option<SpatialTransformerSpec>,
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
        if let Some(spec) = attention {
            let vs_attn = vs.pp("attentions");
            for i in 0..num_layers {
                attentions.push(MultiViewSpatialTransformer::with_spec(
                    vs_attn.pp(i.to_string()),
                    &spec,
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
    /// Build the bottleneck.
    ///
    /// `attention` carries the same config-derived kernel selection the
    /// encoder and decoder stages use.
    fn new(
        vs: nn::VarBuilder,
        channels: usize,
        time_dim: usize,
        attention: &SpatialTransformerSpec,
    ) -> Result<Self> {
        let resnet1 = ResBlock::new(vs.pp("resnets.0"), channels, channels, time_dim)?;
        let attention = MultiViewSpatialTransformer::with_spec(vs.pp("attentions.0"), attention)?;
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

/// Everything one [`UpBlock`] needs to be built.
///
/// Groups the decoder stage's geometry the way [`SpatialTransformerSpec`]
/// groups an attention stage's, so [`MultiViewUNet::new`] passes a single
/// value instead of eight positional arguments (which is also what
/// `clippy::too_many_arguments` flags once the crate stops blanket-allowing
/// it).
#[derive(Debug, Clone, Copy)]
struct UpBlockSpec<'a> {
    /// Channel count entering the stage — the previous decoder stage's output
    /// (or the bottleneck's, for the first stage).
    in_channels: usize,
    /// Channel count every resnet in this stage emits.
    out_channels: usize,
    /// Width of the skip connection consumed by each resnet, in the order they
    /// are consumed. Diffusers' `CrossAttnUpBlock2D` does **not** use a single
    /// width for the whole stage: the first `num_layers - 1` resnets receive a
    /// skip of `out_channels` channels while the last receives the block's own
    /// `in_channels` (the next stage's width). Its length must equal
    /// `num_layers`; [`UpBlock::new`] reports a mismatch rather than building a
    /// stage whose resnets disagree with the skip stack.
    skip_channels: &'a [usize],
    /// Width of the timestep (plus camera) embedding each resnet projects.
    time_dim: usize,
    /// Number of resnet layers — and, when `attention` is `Some`, of attention
    /// layers — in this stage.
    num_layers: usize,
    /// Spatial-transformer geometry for this stage, or `None` for a stage
    /// without attention.
    attention: Option<SpatialTransformerSpec>,
    /// Whether the stage ends with a 2× nearest-neighbour upsample.
    has_upsample: bool,
}

impl UpBlock {
    /// Build one decoder stage from its [`UpBlockSpec`].
    ///
    /// # Errors
    ///
    /// Returns an error when `spec.skip_channels.len() != spec.num_layers`, and
    /// propagates weight-loading failures.
    fn new(vs: nn::VarBuilder, spec: UpBlockSpec<'_>) -> Result<Self> {
        let UpBlockSpec {
            in_channels: in_ch,
            out_channels: out_ch,
            skip_channels,
            time_dim,
            num_layers,
            attention,
            has_upsample,
        } = spec;

        if skip_channels.len() != num_layers {
            return Err(candle_core::Error::Msg(format!(
                "UpBlock expects one skip width per layer: got {} widths for {num_layers} layers",
                skip_channels.len()
            )));
        }

        let vs_res = vs.pp("resnets");
        let mut resnets = Vec::with_capacity(num_layers);
        // `skip_channels.len() == num_layers` was checked above.
        for (i, &skip_ch) in skip_channels.iter().enumerate() {
            // Resnet input = previous stage output (first layer only) or this
            // stage's own output, concatenated with that layer's skip.
            let resnet_in = if i == 0 { in_ch } else { out_ch };
            let ich = resnet_in + skip_ch;
            resnets.push(ResBlock::new(
                vs_res.pp(i.to_string()),
                ich,
                out_ch,
                time_dim,
            )?);
        }

        let mut attentions = Vec::new();
        if let Some(spec) = attention {
            let vs_attn = vs.pp("attentions");
            for i in 0..num_layers {
                attentions.push(MultiViewSpatialTransformer::with_spec(
                    vs_attn.pp(i.to_string()),
                    &spec,
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
    if !stage_channels.is_multiple_of(n_heads) {
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

/// Build the spatial-transformer spec for one U-Net stage.
///
/// Threads [`DiffusionConfig::resolved_attention_backend`] (and therefore both
/// the legacy `use_flash_attention` flag and the newer `attention_backend`
/// selector) into every attention layer. Before this existed the U-Net always
/// built its transformers through a positional constructor that hardcoded the
/// standard kernel (since removed in favour of
/// [`MultiViewSpatialTransformer::with_spec`]), which made both config fields
/// inert for a real pipeline run.
///
/// `stage` indexes into the per-stage head-count and depth vectors; `channels`
/// is that stage's channel width.
///
/// The IP-Adapter context width comes from
/// [`DiffusionConfig::ip_adapter_context_dim`], the same accessor
/// [`crate::clip::build_clip_encoder`] projects its output to. It used to read
/// `config.clip_embed_dim` (the CLIP tower's *own* hidden width, 1280) while
/// the encoder projected to `cross_attention_dim` (1024), so `attn_ip` was
/// built for a width no CLIP encoder in this crate ever emits and the default
/// configuration shape-errored on its first `step_session` U-Net pass.
fn stage_transformer_spec(
    config: &DiffusionConfig,
    stage: usize,
    channels: usize,
) -> Result<SpatialTransformerSpec> {
    let (n_heads, d_head) = resolve_attention_heads(config, stage, channels)?;
    Ok(SpatialTransformerSpec {
        in_channels: channels,
        depth: resolve_transformer_depth(config, stage)?,
        context_dim: config.cross_attention_dim,
        ip_dim: config.ip_adapter_context_dim(),
        num_views: config.num_views,
        num_groups: config.norm_num_groups,
        use_linear_projection: config.use_linear_projection,
        attention: AttentionSpec::from_config(config, n_heads, d_head),
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
// ControlNet injection
// ---------------------------------------------------------------------------

/// Whether `control` would change anything at `stage_idx`.
///
/// [`ControlNetProcessor::apply_to_features`] already no-ops in these cases,
/// but reaching it costs a full device→host→device round trip of the feature
/// tensor, so the cheap checks happen before the readback.
fn control_affects_stage(control: &ControlNetProcessor, stage_idx: usize) -> bool {
    control.config.enabled
        && !control.conditions().is_empty()
        && control.config.injects_at(stage_idx)
}

/// Inject ControlNet conditioning into one stage's `(B, C, H, W)` features.
///
/// [`ControlNetProcessor::apply_to_features`] operates on a single sample's
/// `[channels][h * w]` row-major buffer, so the batch is walked one sample at a
/// time; applying it to the whole `B * C` buffer at once would make the
/// implied channel count `B * C` and stop any registered [`ZeroConv`] (sized
/// for `C` output channels) from matching.
///
/// The buffer is converted to `f32` for the projection and back to the input
/// dtype afterwards, so a mixed-precision run keeps its dtype.
fn inject_control(
    xs: &Tensor,
    control: &ControlNetProcessor,
    stage_idx: usize,
) -> std::result::Result<Tensor, DiffusionError> {
    let (batch, channels, height, width) = xs.dims4()?;
    let dtype = xs.dtype();

    let mut data = xs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let per_sample = channels.saturating_mul(height).saturating_mul(width);
    if per_sample == 0 {
        return Ok(xs.clone());
    }

    // Samples are independent: `apply_to_features` writes only into its own
    // slice and reads only the processor's immutable condition cache, so the
    // batch dimension parallelises without any coordination. The two branches
    // are arithmetic-identical.
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(per_sample).for_each(|sample| {
            control.apply_to_features(sample, stage_idx, height, width);
        });
    }
    #[cfg(not(feature = "parallel"))]
    {
        for sample in data.chunks_mut(per_sample) {
            control.apply_to_features(sample, stage_idx, height, width);
        }
    }

    let injected = Tensor::from_vec(data, (batch, channels, height, width), xs.device())?;
    Ok(injected.to_dtype(dtype)?)
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
            let has_ds = i < num_stages - 1;

            down_blocks.push(DownBlock::new(
                vs_down.pp(i.to_string()),
                input_ch,
                output_ch,
                time_embed_dim,
                config.layers_per_block,
                // All stages have attention in SD 2.1 768-v.
                Some(stage_transformer_spec(config, i, output_ch)?),
                has_ds,
            )?);
            input_ch = output_ch;
        }

        // Mid block
        let last_ch = config.stage_channels(num_stages - 1);
        let mid_block = MidBlock::new(
            vs.pp("mid_block"),
            last_ch,
            time_embed_dim,
            &stage_transformer_spec(config, num_stages - 1, last_ch)?,
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
            let has_us = i < num_stages - 1;

            up_blocks.push(UpBlock::new(
                vs_up.pp(i.to_string()),
                UpBlockSpec {
                    in_channels: prev_ch,
                    out_channels: output_ch,
                    skip_channels: &skip_channels,
                    time_dim: time_embed_dim,
                    // +1 for the conv_in / downsample skip.
                    num_layers: up_layers,
                    attention: Some(stage_transformer_spec(config, stage_idx, output_ch)?),
                    has_upsample: has_us,
                },
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
    /// - `ip_tokens`: `(B*V, ip_len, ip_dim)` CLIP image tokens, where `ip_dim`
    ///   is [`DiffusionConfig::ip_adapter_context_dim`] — exactly what
    ///   [`crate::clip::ClipImageEncoder::forward`] returns for the same config.
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
        self.forward_with_control(sample, timestep, context, camera_poses, ip_tokens, None)
    }

    /// Attach (or, with `None`, detach) a shared cross-attention KV cache.
    ///
    /// The IP-Adapter CLIP tokens are constant for a whole denoising run, so
    /// their `to_k`/`to_v` projections are identical at every timestep. With a
    /// cache attached, each `attn_ip` layer computes them once and replays them
    /// for the rest of the run.
    ///
    /// `conditioning_tag` must change whenever the IP tokens do — it is mixed
    /// into every cache key, so a stale entry from another reference image can
    /// never be served. [`crate::pipeline::MultiViewDiffusionPipeline`] derives
    /// it by hashing the tokens.
    ///
    /// Layer indices are assigned here in a fixed traversal order (encoder
    /// stages, then the bottleneck, then decoder stages), so the same model
    /// always produces the same keys.
    pub fn set_kv_cache(&mut self, cache: Option<Arc<KVCache>>, conditioning_tag: u64) {
        let mut layer_idx = 0usize;
        for down in self.down_blocks.iter_mut() {
            for attention in down.attentions.iter_mut() {
                attention.set_kv_cache(cache.clone(), layer_idx, conditioning_tag);
                layer_idx += 1;
            }
        }
        self.mid_block
            .attention
            .set_kv_cache(cache.clone(), layer_idx, conditioning_tag);
        layer_idx += 1;
        for up in self.up_blocks.iter_mut() {
            for attention in up.attentions.iter_mut() {
                attention.set_kv_cache(cache.clone(), layer_idx, conditioning_tag);
                layer_idx += 1;
            }
        }
    }

    /// Number of attention layers [`Self::set_kv_cache`] assigns keys to.
    pub fn num_attention_layers(&self) -> usize {
        self.down_blocks
            .iter()
            .map(|d| d.attentions.len())
            .chain(std::iter::once(1))
            .chain(self.up_blocks.iter().map(|u| u.attentions.len()))
            .sum()
    }

    /// Forward pass with optional ControlNet conditioning.
    ///
    /// Identical to [`Self::forward`] except that, when `control` is `Some`,
    /// [`ControlNetProcessor::apply_to_features`] is applied to the trunk
    /// activation leaving each encoder stage, with the stage's own index as
    /// the ControlNet layer index. [`ControlNetConfig::injects_at`] therefore
    /// selects which encoder stages receive conditioning, and
    /// [`ControlNetConfig::late_injection`] narrows that to the coarser half.
    ///
    /// [`ControlNetConfig::injects_at`]: crate::controlnet::ControlNetConfig::injects_at
    /// [`ControlNetConfig::late_injection`]: crate::controlnet::ControlNetConfig::late_injection
    ///
    /// Only the propagating trunk is conditioned: the skip connections a stage
    /// pushed for the decoder are captured *inside* the stage and are left as
    /// the encoder produced them, matching the "inject between U-Net stages"
    /// contract documented on [`ControlNetProcessor::apply_to_features`].
    ///
    /// With no [`ZeroConv`][crate::controlnet::ZeroConv] registered for a
    /// stage, that stage takes the processor's documented broadcast fallback (a
    /// channel-constant spatial bias) rather than learned control features —
    /// see the [`crate::controlnet`] module docs.
    ///
    /// # Errors
    ///
    /// As [`Self::forward`], plus `DiffusionError::Candle` if a feature buffer
    /// cannot be read back for injection.
    pub fn forward_with_control(
        &self,
        sample: &Tensor,
        timestep: usize,
        context: Option<&Tensor>,
        camera_poses: Option<&Tensor>,
        ip_tokens: Option<&Tensor>,
        control: Option<&ControlNetProcessor>,
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
        for (stage_idx, down) in self.down_blocks.iter().enumerate() {
            let (out, skips) = down.forward(&h, &emb, context, ip_tokens)?;
            h = out;
            all_skips.extend(skips);

            // ControlNet conditioning for this encoder stage.
            if let Some(processor) = control {
                if control_affects_stage(processor, stage_idx) {
                    h = inject_control(&h, processor, stage_idx)?;
                }
            }
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
    use crate::config::AttentionBackend;
    use crate::controlnet::{ControlNetCondition, ControlNetConfig, ZeroConv};
    use candle_core::Device;

    /// A miniature but structurally faithful U-Net config.
    ///
    /// `ResBlock` group-normalises with 32 groups, so every channel width the
    /// decoder concatenates has to stay a multiple of 32; base 32 with
    /// `channel_mult = [1, 2]` is the smallest configuration that satisfies it.
    ///
    /// `clip_embed_dim` (80) is deliberately *different* from
    /// `cross_attention_dim` (32): the CLIP tower's hidden width and the
    /// IP-Adapter context width are independent knobs, and every test that
    /// feeds IP tokens has to use the latter — via
    /// [`DiffusionConfig::ip_adapter_context_dim`] — or it stops exercising the
    /// shape the pipeline actually produces. 80 is the smallest width
    /// [`DiffusionConfig::validate`] accepts (ViT-H/14's head width), so this
    /// config stays valid as well as divergent.
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
            clip_embed_dim: 80,
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
    // IP-Adapter context width
    //
    // Regression: `stage_transformer_spec` built `attn_ip` for
    // `config.clip_embed_dim` (the CLIP tower's own hidden width, 1280 by
    // default) while `clip::build_clip_encoder` projected the encoder's output
    // to `cross_attention_dim` (1024). `step_session` therefore fed 1024-wide
    // IP tokens into a 1280-wide projection and shape-errored on the DEFAULT
    // configuration. Both sides now read
    // `DiffusionConfig::ip_adapter_context_dim`.
    // ------------------------------------------------------------------

    #[test]
    fn test_attn_ip_context_width_is_the_ip_adapter_context_dim() -> Result<()> {
        let config = DiffusionConfig::default();
        // The default config is the one that used to break: its two widths
        // genuinely differ, so reading the wrong one is not a no-op.
        assert_ne!(config.clip_embed_dim, config.ip_adapter_context_dim());

        for stage in 0..config.num_stages() {
            let channels = config.stage_channels(stage);
            let spec = stage_transformer_spec(&config, stage, channels)?;
            assert_eq!(
                spec.ip_dim,
                config.ip_adapter_context_dim(),
                "stage {stage} attn_ip context width"
            );
            // Text context is unaffected; it was never the divergent one.
            assert_eq!(spec.context_dim, config.cross_attention_dim);
        }
        Ok(())
    }

    /// A `step_session`-shaped run: encode a reference image with a CLIP
    /// encoder built the way [`crate::clip::build_clip_encoder`] builds one,
    /// expand the tokens to every view, and hand them to the U-Net.
    ///
    /// The config's `clip_embed_dim` (80) differs from its
    /// `cross_attention_dim` (32), reproducing the default configuration's
    /// divergence at a size a unit test can build. Before the fix the U-Net's
    /// `attn_ip` was built for the tower width while the encoder emitted the
    /// cross-attention width, and this forward pass failed with a matmul shape
    /// error.
    #[test]
    fn test_clip_tokens_flow_into_the_unet_when_the_two_widths_differ() -> Result<()> {
        use crate::clip::{ClipImageEncoder, ClipVisionConfig};

        let device = Device::Cpu;
        let config = tiny_config();
        assert_ne!(
            config.clip_embed_dim,
            config.ip_adapter_context_dim(),
            "this test is only meaningful while the two widths differ"
        );

        let unet_varmap = nn::VarMap::new();
        let unet = MultiViewUNet::new(
            nn::VarBuilder::from_varmap(&unet_varmap, DType::F32, &device),
            &config,
        )?;

        // A CLIP tower as wide as `clip_embed_dim`, projecting to the U-Net's
        // IP context width — exactly what `build_clip_encoder` does, minus
        // ViT-H/14's 32 layers of 1280 units.
        let vision = ClipVisionConfig {
            embed_dim: config.clip_embed_dim,
            num_heads: 2,
            num_layers: 1,
            intermediate_size: config.clip_embed_dim * 2,
            image_size: 8,
            patch_size: 4,
        };
        let clip_varmap = nn::VarMap::new();
        let clip = ClipImageEncoder::new(
            nn::VarBuilder::from_varmap(&clip_varmap, DType::F32, &device),
            &vision,
            Some(config.ip_adapter_context_dim()),
        )?;

        // 1. Encode the reference image (batch of one, as `step_session` does).
        let reference = Tensor::randn(
            0f32,
            1f32,
            (1usize, 3usize, vision.image_size, vision.image_size),
            &device,
        )?;
        let tokens = clip.forward(&reference)?;
        assert_eq!(
            tokens.dims(),
            &[1, vision.num_patches() + 1, config.ip_adapter_context_dim()]
        );

        // 2. Expand to every view, then denoise one step.
        let bv = config.num_views;
        let ip_tokens = tokens.repeat(&[bv, 1, 1])?;
        let sample = Tensor::randn(0f32, 1f32, (bv, config.unet_in_channels, 8, 8), &device)?;
        let context = Tensor::randn(0f32, 1f32, (bv, 2, config.cross_attention_dim), &device)?;
        let poses = Tensor::randn(0f32, 1f32, (bv, config.camera_pose_dim), &device)?;

        let out = unet
            .forward(&sample, 10, Some(&context), Some(&poses), Some(&ip_tokens))
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;

        assert_eq!(out.dims(), &[bv, config.unet_out_channels, 8, 8]);
        Ok(())
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
            UpBlockSpec {
                in_channels: 64,
                out_channels: 64,
                skip_channels: &[64, 32],
                time_dim: 64,
                num_layers: 3,
                attention: None,
                has_upsample: false,
            },
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
        let ip_tokens = Tensor::randn(
            0f32,
            1f32,
            (bv, 3, config.ip_adapter_context_dim()),
            &device,
        )?;

        let out = unet
            .forward(&sample, 10, Some(&context), Some(&poses), Some(&ip_tokens))
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;

        assert_eq!(out.dims(), &[bv, config.unet_out_channels, 8, 8]);
        Ok(())
    }

    // ------------------------------------------------------------------
    // ControlNet injection
    //
    // Regression: `MultiViewUNet::forward` never called
    // `ControlNetProcessor::apply_to_features`, so a configured processor had
    // no effect on inference at all.
    // ------------------------------------------------------------------

    fn edge_processor(scale: f32, injection_layers: Vec<usize>) -> ControlNetProcessor {
        let mut processor = ControlNetProcessor::new(ControlNetConfig {
            enabled: true,
            default_scale: scale,
            injection_layers,
            late_injection: false,
            ..ControlNetConfig::default_config()
        });
        // A 4×4 edge map with a constant, clearly non-zero signal.
        let condition = ControlNetCondition::edge_map(vec![1.0f32; 16], 4, 4)
            .expect("edge map construction failed");
        processor
            .add_condition(condition)
            .expect("add_condition failed");
        processor
    }

    #[test]
    fn test_inject_control_changes_features_at_an_injection_layer() -> Result<()> {
        let device = Device::Cpu;
        let processor = edge_processor(1.0, vec![0, 1, 2, 3]);
        let features = Tensor::zeros((2, 4, 4, 4), DType::F32, &device)?;

        assert!(control_affects_stage(&processor, 0));
        let injected = inject_control(&features, &processor, 0)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        assert_eq!(injected.dims(), features.dims());

        let sum = injected.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(
            sum > 1e-6,
            "ControlNet injection must change an all-zero feature buffer, got sum {sum}"
        );
        Ok(())
    }

    #[test]
    fn test_inject_control_is_a_no_op_outside_the_injection_layers() {
        let processor = edge_processor(1.0, vec![0]);
        assert!(control_affects_stage(&processor, 0));
        assert!(!control_affects_stage(&processor, 1));
        assert!(!control_affects_stage(&processor, 3));
    }

    #[test]
    fn test_control_affects_stage_requires_conditions_and_enabled() {
        // No conditions registered.
        let empty = ControlNetProcessor::new(ControlNetConfig {
            enabled: true,
            injection_layers: vec![0],
            ..ControlNetConfig::default_config()
        });
        assert!(!control_affects_stage(&empty, 0));

        // Disabled entirely.
        let disabled = ControlNetProcessor::new(ControlNetConfig::disabled());
        assert!(!control_affects_stage(&disabled, 0));
    }

    /// The `parallel` feature only reorders *when* each sample is injected,
    /// never what it computes, so a multi-sample batch must agree with the
    /// same samples injected one at a time.
    #[test]
    fn test_inject_control_batch_matches_per_sample_injection() -> Result<()> {
        let device = Device::Cpu;
        let processor = edge_processor(0.75, vec![0, 1]);
        let batched = Tensor::randn(0f32, 1f32, (3, 4, 4, 4), &device)?;

        let batched_out = inject_control(&batched, &processor, 1)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        let batched_values = batched_out.flatten_all()?.to_vec1::<f32>()?;

        let per_sample = 4 * 4 * 4;
        for sample in 0..3 {
            let single = batched.narrow(0, sample, 1)?;
            let single_out = inject_control(&single, &processor, 1)
                .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
            let single_values = single_out.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(single_values.len(), per_sample);
            for (i, want) in single_values.iter().enumerate() {
                let got = batched_values[sample * per_sample + i];
                assert!(
                    (got - want).abs() < 1e-6,
                    "sample {sample} element {i}: {got} != {want}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_inject_control_treats_each_batch_sample_separately() -> Result<()> {
        // A registered ZeroConv is only used when its out_channels match the
        // *per-sample* channel count; flattening the whole (B, C, H, W) buffer
        // would make the implied count B*C and silently drop to the fallback.
        let device = Device::Cpu;
        let mut processor = edge_processor(1.0, vec![0]);
        let channels = 4usize;
        let in_channels = processor.condition_channels();
        let conv = ZeroConv::from_weights(
            in_channels,
            channels,
            vec![0.0; in_channels * channels],
            // Bias only: every output channel gets a distinct constant.
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .expect("zero conv construction failed");
        processor.set_zero_conv(0, conv);

        let features = Tensor::zeros((2, channels, 4, 4), DType::F32, &device)?;
        let injected = inject_control(&features, &processor, 0)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        let values = injected.flatten_all()?.to_vec1::<f32>()?;

        let spatial = 16usize;
        for sample in 0..2 {
            for (channel, want) in [1.0f32, 2.0, 3.0, 4.0].iter().enumerate() {
                let base = sample * channels * spatial + channel * spatial;
                for offset in 0..spatial {
                    let got = values[base + offset];
                    assert!(
                        (got - want).abs() < 1e-5,
                        "sample {sample} channel {channel} offset {offset}: {got} != {want}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_tiny_unet_forward_with_control_changes_the_prediction() -> Result<()> {
        let device = Device::Cpu;
        let varmap = nn::VarMap::new();
        let vs = nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = tiny_config();
        let unet = MultiViewUNet::new(vs, &config)?;

        let bv = config.num_views;
        let sample = Tensor::randn(0f32, 1f32, (bv, config.unet_in_channels, 8, 8), &device)?;
        let context = Tensor::randn(0f32, 1f32, (bv, 2, config.cross_attention_dim), &device)?;

        let plain = unet
            .forward(&sample, 5, Some(&context), None, None)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;

        let processor = edge_processor(1.0, vec![0, 1]);
        let controlled = unet
            .forward_with_control(&sample, 5, Some(&context), None, None, Some(&processor))
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;

        assert_eq!(controlled.dims(), plain.dims());
        let diff = (&controlled - &plain)?
            .abs()?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff > 1e-5,
            "an enabled ControlNet processor must change the U-Net output, got diff {diff}"
        );

        // A disabled processor must leave the prediction untouched.
        let disabled = ControlNetProcessor::new(ControlNetConfig::disabled());
        let unchanged = unet
            .forward_with_control(&sample, 5, Some(&context), None, None, Some(&disabled))
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        let no_diff = (&unchanged - &plain)?
            .abs()?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(
            no_diff < 1e-6,
            "a disabled ControlNet processor must be a no-op, got diff {no_diff}"
        );
        Ok(())
    }

    // ------------------------------------------------------------------
    // Cross-attention KV cache fan-out
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // Attention backend selection
    //
    // Regression: `MultiViewUNet::new` always built its transformers through
    // the positional constructor that hardcoded the standard kernel (since
    // removed in favour of `MultiViewSpatialTransformer::with_spec`), so
    // `DiffusionConfig::use_flash_attention` was inert for a real run and
    // `SlicedAttention` was unreachable from a pipeline.
    // ------------------------------------------------------------------

    fn built_backends(config: &DiffusionConfig) -> Result<Vec<AttentionBackend>> {
        let varmap = nn::VarMap::new();
        let vs = nn::VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let unet = MultiViewUNet::new(vs, config)?;
        let mut backends = Vec::new();
        for down in &unet.down_blocks {
            backends.extend(down.attentions.iter().filter_map(|a| a.attention_backend()));
        }
        backends.extend(unet.mid_block.attention.attention_backend());
        for up in &unet.up_blocks {
            backends.extend(up.attentions.iter().filter_map(|a| a.attention_backend()));
        }
        Ok(backends)
    }

    #[test]
    fn test_config_attention_backend_reaches_every_layer() -> Result<()> {
        let config = DiffusionConfig {
            attention_backend: AttentionBackend::Sliced,
            attention_slice_size: Some(8),
            ..tiny_config()
        };
        let backends = built_backends(&config)?;
        assert!(!backends.is_empty());
        assert!(
            backends.iter().all(|b| *b == AttentionBackend::Sliced),
            "every layer must follow the configured backend: {backends:?}"
        );
        Ok(())
    }

    #[test]
    fn test_default_config_builds_standard_attention_layers() -> Result<()> {
        let config = tiny_config();
        let expected = config.resolved_attention_backend();
        let backends = built_backends(&config)?;
        assert!(backends.iter().all(|b| *b == expected), "{backends:?}");
        Ok(())
    }

    #[test]
    fn test_legacy_flash_flag_reaches_the_layers() -> Result<()> {
        let config = DiffusionConfig {
            use_flash_attention: true,
            ..tiny_config()
        };
        let backends = built_backends(&config)?;
        assert!(
            backends.iter().all(|b| *b == AttentionBackend::Flash),
            "the legacy flag must still select flash: {backends:?}"
        );
        Ok(())
    }

    #[test]
    fn test_sliced_backend_produces_the_same_prediction() -> Result<()> {
        // Same weights via one shared VarMap, two kernels: the U-Net's output
        // must not depend on which one runs.
        let device = Device::Cpu;
        let varmap = nn::VarMap::new();
        let base = tiny_config();
        let sliced_config = DiffusionConfig {
            attention_backend: AttentionBackend::Sliced,
            attention_slice_size: Some(5),
            ..base.clone()
        };

        let standard = MultiViewUNet::new(
            nn::VarBuilder::from_varmap(&varmap, DType::F32, &device),
            &base,
        )?;
        let sliced = MultiViewUNet::new(
            nn::VarBuilder::from_varmap(&varmap, DType::F32, &device),
            &sliced_config,
        )?;

        let bv = base.num_views;
        let sample = Tensor::randn(0f32, 1f32, (bv, base.unet_in_channels, 8, 8), &device)?;
        let context = Tensor::randn(0f32, 1f32, (bv, 2, base.cross_attention_dim), &device)?;

        let a = standard
            .forward(&sample, 3, Some(&context), None, None)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        let b = sliced
            .forward(&sample, 3, Some(&context), None, None)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;

        assert_eq!(a.dims(), b.dims());
        let diff = (&a - &b)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-3, "sliced U-Net diverged from standard: {diff}");
        Ok(())
    }

    #[test]
    fn test_set_kv_cache_reaches_every_attention_layer() -> Result<()> {
        let device = Device::Cpu;
        let varmap = nn::VarMap::new();
        let vs = nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = tiny_config();
        let mut unet = MultiViewUNet::new(vs, &config)?;

        // 2 stages × 1 attention each (down) + 1 mid + 2 stages × 2 (up).
        let expected_layers = unet.num_attention_layers();
        assert_eq!(expected_layers, 2 + 1 + 4);

        let cache = Arc::new(KVCache::new(crate::kv_cache::KVCacheConfig::default()));
        unet.set_kv_cache(Some(Arc::clone(&cache)), 0xfeed);

        let mut keys: Vec<String> = Vec::new();
        for down in &unet.down_blocks {
            for attention in &down.attentions {
                keys.extend(attention.ip_cache_keys().map(str::to_string));
            }
        }
        keys.extend(unet.mid_block.attention.ip_cache_keys().map(str::to_string));
        for up in &unet.up_blocks {
            for attention in &up.attentions {
                keys.extend(attention.ip_cache_keys().map(str::to_string));
            }
        }

        assert_eq!(keys.len(), expected_layers, "one key per attention layer");
        let unique: std::collections::HashSet<&String> = keys.iter().collect();
        assert_eq!(unique.len(), keys.len(), "keys must be unique: {keys:?}");
        assert!(keys.iter().all(|k| k.contains("cond=65261")), "{keys:?}");

        // Detaching must reach every layer too.
        unet.set_kv_cache(None, 0xfeed);
        assert!(unet
            .down_blocks
            .iter()
            .flat_map(|d| d.attentions.iter())
            .all(|a| a.ip_cache_keys().next().is_none()));
        Ok(())
    }

    #[test]
    fn test_cached_forward_matches_uncached_across_steps() -> Result<()> {
        let device = Device::Cpu;
        let varmap = nn::VarMap::new();
        let vs = nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = tiny_config();
        let mut unet = MultiViewUNet::new(vs, &config)?;

        let bv = config.num_views;
        let sample = Tensor::randn(0f32, 1f32, (bv, config.unet_in_channels, 8, 8), &device)?;
        let context = Tensor::randn(0f32, 1f32, (bv, 2, config.cross_attention_dim), &device)?;
        let ip_tokens = Tensor::randn(
            0f32,
            1f32,
            (bv, 3, config.ip_adapter_context_dim()),
            &device,
        )?;

        let baseline = unet
            .forward(&sample, 10, Some(&context), None, Some(&ip_tokens))
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;

        let cache = Arc::new(KVCache::new(crate::kv_cache::KVCacheConfig::default()));
        unet.set_kv_cache(Some(Arc::clone(&cache)), 1);

        // Two "denoising steps": the first populates the cache, the second
        // serves from it. Both must reproduce the uncached prediction.
        for step in 0..2 {
            let out = unet
                .forward(&sample, 10, Some(&context), None, Some(&ip_tokens))
                .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
            let diff = (&out - &baseline)?.abs()?.sum_all()?.to_scalar::<f32>()?;
            assert!(
                diff < 1e-4,
                "step {step}: cached U-Net output drifted: {diff}"
            );
        }

        let stats = cache.stats();
        let layers = unet.num_attention_layers() as u64;
        assert_eq!(stats.misses, layers, "the first pass must miss every layer");
        assert_eq!(stats.hits, layers, "the second pass must hit every layer");
        Ok(())
    }

    #[test]
    fn test_forward_matches_forward_with_control_none() -> Result<()> {
        let device = Device::Cpu;
        let varmap = nn::VarMap::new();
        let vs = nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = tiny_config();
        let unet = MultiViewUNet::new(vs, &config)?;

        let bv = config.num_views;
        let sample = Tensor::randn(0f32, 1f32, (bv, config.unet_in_channels, 8, 8), &device)?;
        let context = Tensor::randn(0f32, 1f32, (bv, 2, config.cross_attention_dim), &device)?;

        let via_forward = unet
            .forward(&sample, 1, Some(&context), None, None)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        let via_control = unet
            .forward_with_control(&sample, 1, Some(&context), None, None, None)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        let diff = (&via_forward - &via_control)?
            .abs()?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-9, "forward must delegate unchanged, diff {diff}");
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
