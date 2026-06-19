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
        let h = self.fc1.forward(&h)?.gelu()?;
        let h = self.fc2.forward(&h)?;
        h + residual
    }
}

// ---------------------------------------------------------------------------
// CLIP Vision Model
// ---------------------------------------------------------------------------

/// CLIP vision model configuration (ViT-H/14 defaults).
#[derive(Debug, Clone)]
pub struct ClipVisionConfig {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub image_size: usize,
    pub patch_size: usize,
}

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
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
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

        let class_embedding = vs.get((1, 1, embed_dim), "embeddings.class_embedding")?;

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
            config: clip_config.clone(),
        })
    }

    /// Encode an image batch into patch-level embeddings.
    ///
    /// - `pixel_values`: `(B, 3, H, W)` normalised image tensor.
    ///
    /// Returns `(B, num_patches + 1, embed_dim)` or projected dimension.
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

/// Build a CLIP encoder from a DiffusionConfig with default ViT-H/14 settings.
pub fn build_clip_encoder(
    vs: nn::VarBuilder,
    config: &DiffusionConfig,
) -> Result<ClipImageEncoder> {
    let clip_config = ClipVisionConfig::default();
    ClipImageEncoder::new(vs, &clip_config, Some(config.cross_attention_dim))
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
}
