//! CLIP image encoder for reference-image conditioning.
//!
//! Implements the ViT-H/14 CLIP image encoder used to extract per-image
//! embeddings that condition the multi-view diffusion model via IP-adapter
//! cross-attention.

use candle_core::{Result, Tensor, D};
use candle_nn as nn;
use candle_nn::Module;

use crate::config::DiffusionConfig;

// ---------------------------------------------------------------------------
// Vision Transformer components
// ---------------------------------------------------------------------------

/// Multi-head self-attention for the CLIP ViT.
#[derive(Debug)]
struct ClipAttention {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl ClipAttention {
    fn new(vs: nn::VarBuilder, embed_dim: usize, num_heads: usize) -> Result<Self> {
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let q_proj = nn::linear(embed_dim, embed_dim, vs.pp("q_proj"))?;
        let k_proj = nn::linear(embed_dim, embed_dim, vs.pp("k_proj"))?;
        let v_proj = nn::linear(embed_dim, embed_dim, vs.pp("v_proj"))?;
        let out_proj = nn::linear(embed_dim, embed_dim, vs.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let reshape = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, seq_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)
        };

        let q = reshape(q)?;
        let k = reshape(k)?;
        let v = reshape(v)?;

        let attn = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;
        let attn = nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        let out = out.transpose(1, 2)?.reshape((b, seq_len, ()))?;
        self.out_proj.forward(&out)
    }
}

/// A single CLIP ViT encoder layer (pre-norm style).
#[derive(Debug)]
struct ClipEncoderLayer {
    layer_norm1: nn::LayerNorm,
    self_attn: ClipAttention,
    layer_norm2: nn::LayerNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl ClipEncoderLayer {
    fn new(
        vs: nn::VarBuilder,
        embed_dim: usize,
        num_heads: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        let layer_norm1 = nn::layer_norm(embed_dim, 1e-5, vs.pp("layer_norm1"))?;
        let self_attn = ClipAttention::new(vs.pp("self_attn"), embed_dim, num_heads)?;
        let layer_norm2 = nn::layer_norm(embed_dim, 1e-5, vs.pp("layer_norm2"))?;
        let fc1 = nn::linear(embed_dim, intermediate_size, vs.pp("mlp.fc1"))?;
        let fc2 = nn::linear(intermediate_size, embed_dim, vs.pp("mlp.fc2"))?;
        Ok(Self {
            layer_norm1,
            self_attn,
            layer_norm2,
            fc1,
            fc2,
        })
    }
}

impl Module for ClipEncoderLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.layer_norm1.forward(xs)?;
        let xs = self.self_attn.forward(&xs)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let h = self.layer_norm2.forward(&xs)?;
        // HuggingFace `laion/CLIP-ViT-H-14` sets `hidden_act = "gelu"`, which is
        // the exact erf-based GELU. candle's `.gelu()` is the tanh approximation;
        // `.gelu_erf()` matches the reference implementation (the ~1e-3 relative
        // per-activation gap compounds across 32 encoder layers otherwise).
        let h = self.fc1.forward(&h)?.gelu_erf()?;
        let h = self.fc2.forward(&h)?;
        h + residual
    }
}

// ---------------------------------------------------------------------------
// CLIP Vision Model
// ---------------------------------------------------------------------------

/// CLIP vision model configuration (ViT-H/14 defaults).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipVisionConfig {
    /// Hidden width of every encoder layer (ViT-H/14 = 1280).
    pub embed_dim: usize,
    /// Self-attention head count; [`Self::embed_dim`] must divide by it.
    pub num_heads: usize,
    /// Number of stacked encoder layers (ViT-H/14 = 32).
    pub num_layers: usize,
    /// Feed-forward hidden width (ViT-H/14 = 4 × `embed_dim` = 5120).
    pub intermediate_size: usize,
    /// Square input resolution the position embedding is sized for.
    pub image_size: usize,
    /// Square patch edge; [`Self::image_size`] must divide by it.
    pub patch_size: usize,
}

/// Head width of every ViT-H/14 attention head: `1280 / 16`.
///
/// [`ClipVisionConfig::vit_h14_with_embed_dim`] holds this constant while
/// scaling the tower's width, so `embed_dim` has to be a multiple of it.
pub const VIT_H14_HEAD_DIM: usize = 80;

/// Feed-forward expansion factor of a ViT-H/14 encoder layer: `5120 / 1280`.
pub const VIT_H14_MLP_RATIO: usize = 4;

impl Default for ClipVisionConfig {
    fn default() -> Self {
        Self {
            embed_dim: 1280,
            num_heads: 16,
            num_layers: 32,
            intermediate_size: 5120,
            image_size: 224,
            patch_size: 14,
        }
    }
}

impl ClipVisionConfig {
    /// Number of image patches the tower tokenises its input into.
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    /// ViT-H/14's geometry re-scaled to an arbitrary hidden width.
    ///
    /// Depth (32 layers), input resolution (224²) and patch size (14) are
    /// ViT-H/14's; the two width-derived quantities are held to ViT-H/14's own
    /// ratios rather than to its absolute numbers:
    ///
    /// - `num_heads = embed_dim / 80` — ViT-H/14's heads are 80 wide
    ///   (`1280 / 16`), and the encoder's attention partitions `embed_dim` by
    ///   the head *count*, so scaling the count is what keeps each head 80 wide.
    /// - `intermediate_size = 4 * embed_dim` — ViT-H/14's MLP ratio
    ///   (`5120 / 1280`).
    ///
    /// `vit_h14_with_embed_dim(1280)` therefore reproduces [`Self::default`]
    /// exactly; the tests below pin that.
    ///
    /// # Errors
    ///
    /// Returns [`crate::DiffusionError::InvalidConfig`] unless `embed_dim` is a
    /// **positive** multiple of [`VIT_H14_HEAD_DIM`]. Zero is rejected
    /// separately from the divisibility rule (`0` *is* a multiple of 80, but a
    /// zero-width tower has no weights `nn::linear` can build), and a width
    /// that is not a multiple of 80 would silently change the head geometry the
    /// checkpoint was trained with.
    pub fn vit_h14_with_embed_dim(embed_dim: usize) -> crate::DiffusionResult<Self> {
        if embed_dim == 0 {
            return Err(crate::DiffusionError::InvalidConfig(
                "CLIP vision embed_dim must be > 0".to_string(),
            ));
        }
        if !embed_dim.is_multiple_of(VIT_H14_HEAD_DIM) {
            return Err(crate::DiffusionError::InvalidConfig(format!(
                "CLIP vision embed_dim must be a multiple of {VIT_H14_HEAD_DIM} \
                 (ViT-H/14's head width), got {embed_dim}"
            )));
        }
        Ok(Self {
            embed_dim,
            num_heads: embed_dim / VIT_H14_HEAD_DIM,
            num_layers: 32,
            intermediate_size: VIT_H14_MLP_RATIO * embed_dim,
            image_size: 224,
            patch_size: 14,
        })
    }
}

/// CLIP ViT image encoder.
///
/// Produces per-patch embeddings suitable for IP-adapter cross-attention.
#[derive(Debug)]
pub struct ClipImageEncoder {
    patch_embedding: nn::Conv2d,
    position_embedding: nn::Embedding,
    class_embedding: Tensor,
    pre_layernorm: nn::LayerNorm,
    encoder_layers: Vec<ClipEncoderLayer>,
    post_layernorm: nn::LayerNorm,
    /// Optional projection to map to cross-attention dimension.
    ip_projection: Option<nn::Linear>,
    /// Width of the tokens `forward` returns: the projection target when
    /// `ip_projection` is present, the tower's own `embed_dim` otherwise.
    output_dim: usize,
    config: ClipVisionConfig,
}

impl ClipImageEncoder {
    /// Build a CLIP image encoder from a VarBuilder.
    pub fn new(
        vs: nn::VarBuilder,
        clip_config: &ClipVisionConfig,
        project_to: Option<usize>,
    ) -> Result<Self> {
        let embed_dim = clip_config.embed_dim;

        let patch_embedding = nn::conv2d(
            3,
            embed_dim,
            clip_config.patch_size,
            nn::Conv2dConfig {
                stride: clip_config.patch_size,
                ..Default::default()
            },
            vs.pp("embeddings.patch_embedding"),
        )?;

        let num_positions = clip_config.num_patches() + 1; // +1 for CLS token
        let position_embedding = nn::embedding(
            num_positions,
            embed_dim,
            vs.pp("embeddings.position_embedding"),
        )?;

        // HuggingFace `CLIPVisionModel` stores `embeddings.class_embedding` as a
        // rank-1 parameter of shape `(embed_dim,)`. `VarBuilder::get` requires an
        // exact shape match (it does not reshape), so requesting `(1, 1, embed_dim)`
        // directly would fail to load a real checkpoint. Fetch the rank-1 tensor and
        // reshape it here instead, so `forward`'s `broadcast_as` call is unaffected.
        let class_embedding = vs
            .get((embed_dim,), "embeddings.class_embedding")?
            .reshape((1, 1, embed_dim))?;

        let pre_layernorm = nn::layer_norm(embed_dim, 1e-5, vs.pp("pre_layrnorm"))?;

        let vs_layers = vs.pp("encoder.layers");
        let mut encoder_layers = Vec::with_capacity(clip_config.num_layers);
        for i in 0..clip_config.num_layers {
            encoder_layers.push(ClipEncoderLayer::new(
                vs_layers.pp(i.to_string()),
                embed_dim,
                clip_config.num_heads,
                clip_config.intermediate_size,
            )?);
        }

        let post_layernorm = nn::layer_norm(embed_dim, 1e-5, vs.pp("post_layernorm"))?;

        let ip_projection = if let Some(target_dim) = project_to {
            Some(nn::linear(embed_dim, target_dim, vs.pp("ip_projection"))?)
        } else {
            None
        };

        Ok(Self {
            patch_embedding,
            position_embedding,
            class_embedding,
            pre_layernorm,
            encoder_layers,
            post_layernorm,
            ip_projection,
            output_dim: project_to.unwrap_or(embed_dim),
            config: clip_config.clone(),
        })
    }

    /// Width of the tokens [`Self::forward`] returns.
    ///
    /// This is the IP-Adapter projection's output width when one was requested
    /// at construction (`project_to`), and the vision tower's own
    /// [`ClipVisionConfig::embed_dim`] otherwise. It is the width the U-Net's
    /// `attn_ip` cross-attention has to be built for — see
    /// [`DiffusionConfig::ip_adapter_context_dim`], which is where
    /// [`build_clip_encoder`] and [`crate::unet::MultiViewUNet::new`] both read
    /// it from so the two cannot disagree.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// The vision tower geometry this encoder was built with.
    pub fn vision_config(&self) -> &ClipVisionConfig {
        &self.config
    }

    /// Encode an image batch into patch-level embeddings.
    ///
    /// - `pixel_values`: `(B, 3, H, W)` normalised image tensor.
    ///
    /// Returns `(B, num_patches + 1, D)`, where `D` is [`Self::output_dim`] —
    /// the IP projection's target width when one was requested, the vision
    /// tower's own `embed_dim` otherwise.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let batch_size = pixel_values.dim(0)?;
        let device = pixel_values.device();
        let dtype = pixel_values.dtype();

        // Patch embedding: (B, 3, H, W) -> (B, embed_dim, H/P, W/P)
        let patches = self.patch_embedding.forward(pixel_values)?;
        let (_, _, h, w) = patches.dims4()?;
        let num_patches = h * w;

        // Flatten spatial: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        let patches = patches.flatten(2, 3)?.transpose(1, 2)?;

        // Prepend CLS token
        let cls = self
            .class_embedding
            .broadcast_as((batch_size, 1, self.config.embed_dim))?;
        let embeddings = Tensor::cat(&[cls.to_dtype(dtype)?, patches], 1)?;

        // Add position embeddings
        let position_ids = Tensor::arange(0u32, (num_patches + 1) as u32, device)?;
        let pos_embeds = self.position_embedding.forward(&position_ids)?;
        let embeddings = (embeddings + pos_embeds.unsqueeze(0)?)?;

        // Pre-layernorm
        let mut hidden = self.pre_layernorm.forward(&embeddings)?;

        // Encoder layers
        for layer in &self.encoder_layers {
            hidden = layer.forward(&hidden)?;
        }

        // Post-layernorm
        hidden = self.post_layernorm.forward(&hidden)?;

        // Optional IP projection
        if let Some(ref proj) = self.ip_projection {
            hidden = proj.forward(&hidden)?;
        }

        Ok(hidden)
    }
}

/// Build a CLIP encoder from a `DiffusionConfig`, on ViT-H/14's geometry.
///
/// The tower is [`ClipVisionConfig::vit_h14_with_embed_dim`] at
/// [`DiffusionConfig::clip_embed_dim`], so that field actually selects the
/// vision tower's width. It used to hardcode [`ClipVisionConfig::default`],
/// which pinned the tower at 1280 no matter what the configuration said —
/// [`crate::model_variants`] presets and any hand-built config could set
/// `clip_embed_dim` and see no effect at all, and a checkpoint whose vision
/// tower was not 1280 wide could not be loaded.
///
/// The IP projection targets [`DiffusionConfig::ip_adapter_context_dim`], which
/// is the same width [`crate::unet::MultiViewUNet::new`] builds its `attn_ip`
/// cross-attention for. Reading both from one accessor is what keeps the
/// encoder's output and the U-Net's IP context from diverging: they used to be
/// two independent fields (`cross_attention_dim` here, `clip_embed_dim` there),
/// so a run on [`DiffusionConfig::default`] fed 1024-wide tokens into a
/// 1280-wide projection and shape-errored on the first denoising step.
///
/// # Errors
///
/// Reports a `clip_embed_dim` that is not a positive multiple of
/// [`VIT_H14_HEAD_DIM`] as [`candle_core::Error::Msg`] — the same condition
/// [`DiffusionConfig::validate`] rejects up front, restated here because this
/// function is reachable without a prior `validate` call. Weight-loading
/// failures propagate unchanged.
pub fn build_clip_encoder(
    vs: nn::VarBuilder,
    config: &DiffusionConfig,
) -> Result<ClipImageEncoder> {
    let clip_config = ClipVisionConfig::vit_h14_with_embed_dim(config.clip_embed_dim)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    ClipImageEncoder::new(vs, &clip_config, Some(config.ip_adapter_context_dim()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Default config has valid image_size / patch_size (no panics, integer ratio)
    #[test]
    fn default_config_valid_ratio() {
        let cfg = ClipVisionConfig::default();
        assert!(cfg.image_size > 0);
        assert!(cfg.patch_size > 0);
        assert_eq!(
            cfg.image_size % cfg.patch_size,
            0,
            "image_size must be divisible by patch_size"
        );
    }

    // 2. num_patches for 224×224 with patch_size=14 → 256
    #[test]
    fn num_patches_224_14() {
        let cfg = ClipVisionConfig {
            image_size: 224,
            patch_size: 14,
            ..ClipVisionConfig::default()
        };
        assert_eq!(cfg.num_patches(), 256);
    }

    // 3. num_patches for 336×336 with patch_size=14 → 576
    #[test]
    fn num_patches_336_14() {
        let cfg = ClipVisionConfig {
            image_size: 336,
            patch_size: 14,
            ..ClipVisionConfig::default()
        };
        assert_eq!(cfg.num_patches(), 576);
    }

    // 4. embed_dim % num_heads == 0 for the default config
    #[test]
    fn default_embed_dim_divisible_by_num_heads() {
        let cfg = ClipVisionConfig::default();
        assert_eq!(
            cfg.embed_dim % cfg.num_heads,
            0,
            "embed_dim must be divisible by num_heads"
        );
    }

    // 5. Constructing via struct-update syntax does not panic
    #[test]
    fn struct_update_no_panic() {
        let cfg = ClipVisionConfig {
            image_size: 224,
            patch_size: 14,
            ..ClipVisionConfig::default()
        };
        // Simply confirming the values are as expected
        assert_eq!(cfg.image_size, 224);
        assert_eq!(cfg.patch_size, 14);
    }

    // 6. Sequence length = num_patches + 1 (CLS token) for the default config
    #[test]
    fn seq_len_default() {
        let cfg = ClipVisionConfig::default();
        // Default is 224/14 = 16, 16^2 = 256 patches, +1 CLS = 257
        let seq_len = cfg.num_patches() + 1;
        assert_eq!(seq_len, 257);
    }

    // 7. ViT-B/32: 224×224 with patch_size=32 → 49 patches
    #[test]
    fn num_patches_vit_b32() {
        let cfg = ClipVisionConfig {
            image_size: 224,
            patch_size: 32,
            embed_dim: 768,
            num_heads: 12,
            num_layers: 12,
            intermediate_size: 3072,
        };
        // (224/32)^2 = 7^2 = 49
        assert_eq!(cfg.num_patches(), 49);
    }

    // 8. Large image 504×504 with patch_size=14 → 36^2 = 1296
    #[test]
    fn num_patches_large_504() {
        let cfg = ClipVisionConfig {
            image_size: 504,
            patch_size: 14,
            ..ClipVisionConfig::default()
        };
        // 504/14 = 36, 36^2 = 1296
        assert_eq!(cfg.num_patches(), 1296);
    }

    // Additional: head_dim is integer for default config
    #[test]
    fn head_dim_is_integer_default() {
        let cfg = ClipVisionConfig::default();
        let head_dim = cfg.embed_dim / cfg.num_heads;
        // ViT-H/14: 1280/16 = 80
        assert_eq!(head_dim, 80);
    }

    // Additional: default num_patches is 256
    #[test]
    fn default_num_patches_is_256() {
        let cfg = ClipVisionConfig::default();
        assert_eq!(cfg.num_patches(), 256);
    }

    // ------------------------------------------------------------------
    // IP-Adapter width contract
    //
    // Regression: `build_clip_encoder` projected to `cross_attention_dim`
    // (1024) while `unet::stage_transformer_spec` built `attn_ip` for
    // `clip_embed_dim` (1280), so a run on the DEFAULT config fed 1024-wide
    // tokens into a 1280-wide projection and shape-errored on the first
    // denoising step. Both sides now read
    // `DiffusionConfig::ip_adapter_context_dim`.
    // ------------------------------------------------------------------

    /// A CLIP tower small enough to build in a unit test.
    ///
    /// ViT-H/14 — what [`build_clip_encoder`] builds — is 32 layers of 1280
    /// hidden units (~630 M parameters), which `nn::VarMap` cannot initialise
    /// in test time. The width contract does not depend on depth, so it is
    /// pinned on a 1-layer tower plus a weight-free assertion on
    /// [`DiffusionConfig::default`] below.
    fn tiny_vision_config(embed_dim: usize) -> ClipVisionConfig {
        ClipVisionConfig {
            embed_dim,
            num_heads: 2,
            num_layers: 1,
            intermediate_size: embed_dim * 2,
            image_size: 8,
            patch_size: 4,
        }
    }

    #[test]
    fn ip_projection_output_dim_is_the_ip_adapter_context_dim() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let varmap = nn::VarMap::new();
        let vs = nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        // Deliberately divergent widths: the tower is 80 wide, the U-Net's
        // IP context 24. The projection is what bridges them. (80 is the
        // narrowest tower `DiffusionConfig::validate` accepts.)
        let config = DiffusionConfig {
            cross_attention_dim: 24,
            clip_embed_dim: 80,
            ..DiffusionConfig::default()
        };
        let vision = tiny_vision_config(config.clip_embed_dim);
        let encoder = ClipImageEncoder::new(vs, &vision, Some(config.ip_adapter_context_dim()))?;

        assert_eq!(encoder.output_dim(), config.ip_adapter_context_dim());
        assert_eq!(encoder.vision_config().embed_dim, config.clip_embed_dim);

        let pixels = Tensor::randn(
            0f32,
            1f32,
            (1usize, 3usize, vision.image_size, vision.image_size),
            &device,
        )?;
        let tokens = encoder.forward(&pixels)?;
        assert_eq!(
            tokens.dims(),
            &[1, vision.num_patches() + 1, config.ip_adapter_context_dim()],
            "encoded tokens must be as wide as the U-Net's attn_ip context"
        );
        Ok(())
    }

    #[test]
    fn unprojected_encoder_reports_the_tower_width() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let varmap = nn::VarMap::new();
        let vs = nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        let vision = tiny_vision_config(16);
        let encoder = ClipImageEncoder::new(vs, &vision, None)?;
        assert_eq!(encoder.output_dim(), vision.embed_dim);

        let pixels = Tensor::randn(
            0f32,
            1f32,
            (1usize, 3usize, vision.image_size, vision.image_size),
            &device,
        )?;
        let tokens = encoder.forward(&pixels)?;
        assert_eq!(tokens.dims(), &[1, vision.num_patches() + 1, 16]);
        Ok(())
    }

    // ------------------------------------------------------------------
    // `clip_embed_dim` actually sizes the tower
    //
    // Regression: `build_clip_encoder` hardcoded `ClipVisionConfig::default()`,
    // so `DiffusionConfig::clip_embed_dim` was inert — every configuration got
    // a 1280-wide ViT-H/14 tower regardless of what it asked for.
    // ------------------------------------------------------------------

    #[test]
    fn vit_h14_with_1280_reproduces_default_exactly() {
        let scaled =
            ClipVisionConfig::vit_h14_with_embed_dim(1280).expect("1280 is ViT-H/14's own width");
        assert_eq!(
            scaled,
            ClipVisionConfig::default(),
            "the width-parameterised constructor must be a superset of Default"
        );
    }

    #[test]
    fn vit_h14_scaling_holds_the_head_width_and_mlp_ratio() {
        for embed_dim in [80usize, 160, 640, 1280, 2560] {
            let cfg = ClipVisionConfig::vit_h14_with_embed_dim(embed_dim)
                .expect("multiple of 80 must be accepted");
            assert_eq!(cfg.embed_dim, embed_dim);
            assert_eq!(
                cfg.embed_dim / cfg.num_heads,
                VIT_H14_HEAD_DIM,
                "every head must stay {VIT_H14_HEAD_DIM} wide at embed_dim={embed_dim}"
            );
            assert_eq!(cfg.intermediate_size, VIT_H14_MLP_RATIO * embed_dim);
            // Depth and patchification are ViT-H/14's, not scaled.
            assert_eq!(cfg.num_layers, 32);
            assert_eq!(cfg.image_size, 224);
            assert_eq!(cfg.patch_size, 14);
        }
    }

    #[test]
    fn vit_h14_rejects_zero_and_non_multiples_of_the_head_width() {
        // Zero is a multiple of 80, so it needs its own guard.
        assert!(ClipVisionConfig::vit_h14_with_embed_dim(0).is_err());
        for embed_dim in [1usize, 79, 81, 768, 1024] {
            assert!(
                ClipVisionConfig::vit_h14_with_embed_dim(embed_dim).is_err(),
                "embed_dim={embed_dim} is not a multiple of {VIT_H14_HEAD_DIM}"
            );
        }
    }

    #[test]
    fn build_clip_encoder_sizes_the_tower_from_clip_embed_dim() -> Result<()> {
        // A 1-layer check is impossible here (ViT-H/14 depth is fixed at 32),
        // so assert on the *geometry* `build_clip_encoder` derives rather than
        // on an instantiated tower.
        let config = DiffusionConfig {
            clip_embed_dim: 640,
            ..DiffusionConfig::default()
        };
        let vision = ClipVisionConfig::vit_h14_with_embed_dim(config.clip_embed_dim)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        assert_eq!(vision.embed_dim, 640);
        assert_ne!(
            vision,
            ClipVisionConfig::default(),
            "a non-default clip_embed_dim must not fall back to ViT-H/14's 1280"
        );
        Ok(())
    }

    #[test]
    fn build_clip_encoder_reports_an_unbuildable_clip_embed_dim() {
        let device = candle_core::Device::Cpu;
        let varmap = nn::VarMap::new();
        let vs = nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        // 1024 is ViT-L/14's width, not a multiple of ViT-H/14's 80-wide head.
        let config = DiffusionConfig {
            clip_embed_dim: 1024,
            ..DiffusionConfig::default()
        };
        let err = build_clip_encoder(vs, &config)
            .expect_err("1024 is not a multiple of the ViT-H/14 head width");
        assert!(
            err.to_string().contains("multiple of 80"),
            "error should name the rule, got: {err}"
        );
    }

    /// The default configuration's widths, asserted without building the
    /// 630 M-parameter tower [`build_clip_encoder`] would.
    #[test]
    fn default_config_projects_vit_h14_down_to_the_cross_attention_width() {
        let config = DiffusionConfig::default();
        assert_eq!(config.ip_adapter_context_dim(), config.cross_attention_dim);
        assert_eq!(config.ip_adapter_context_dim(), 1024);
        // The tower's own width is the projection's *input*, and it is not the
        // same number — which is exactly why reading it as the `attn_ip`
        // context width used to break every default-config run.
        assert_eq!(ClipVisionConfig::default().embed_dim, config.clip_embed_dim);
        assert_ne!(config.clip_embed_dim, config.ip_adapter_context_dim());
    }
}
