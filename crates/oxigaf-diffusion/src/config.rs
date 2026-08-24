//! Configuration for the multi-view diffusion pipeline.
//!
//! # Classifier-Free Guidance (CFG)
//!
//! The pipeline uses CFG to control the strength of IP-Adapter conditioning.
//! CFG interpolates between conditional and unconditional predictions:
//!
//! ```text
//! prediction = unconditional + guidance_scale * (conditional - unconditional)
//! ```
//!
//! ## How CFG Works in GAF
//!
//! 1. **Conditional Pass**: U-Net forward pass WITH IP-Adapter tokens from
//!    the reference image (CLIP embeddings)
//! 2. **Unconditional Pass**: U-Net forward pass WITHOUT IP-Adapter tokens
//!    (skips reference conditioning)
//! 3. **Interpolation**: Combine predictions based on `guidance_scale`
//!
//! ## Guidance Scale Selection
//!
//! - **1.0**: Pure conditional (no guidance, equivalent to single forward pass)
//! - **3.0-7.5**: Balanced (recommended for GAF, default: 3.0)
//! - **>10.0**: Strong conditioning (may oversaturate or reduce diversity)
//!
//! # IP-Adapter Architecture
//!
//! IP-Adapter provides pixel-level identity preservation by conditioning on
//! CLIP image embeddings. The architecture includes:
//!
//! - **CLIP Encoder**: ViT-H/14 encodes reference image to 257×1280 embeddings
//! - **Projection**: Linear projection from 1280 → 1024 (cross_attention_dim)
//! - **IP Cross-Attention**: Dedicated `attn_ip` layer in each transformer block
//! - **Integration**: Each spatial position attends to image tokens
//!
//! This differs from text conditioning by providing direct visual features
//! rather than semantic embeddings.

use crate::upsampler::UpsamplerMode;

/// Full configuration for the multi-view diffusion model.
///
/// Contains all hyperparameters for the diffusion pipeline, including U-Net
/// architecture, attention settings, CFG parameters, and optional upsampling.
///
/// # Examples
///
/// ```rust
/// use oxigaf_diffusion::DiffusionConfig;
///
/// // Use default configuration (256×256, guidance_scale=3.0)
/// let config = DiffusionConfig::default();
///
/// // Customize guidance scale for stronger conditioning
/// let mut config = DiffusionConfig::default();
/// config.guidance_scale = 7.5;
///
/// // Enable upsampling for 512×512 output
/// use oxigaf_diffusion::UpsamplerMode;
/// config.upsampler_mode = Some(UpsamplerMode::SdX2);
/// ```
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    /// Number of views to generate simultaneously (default: 4).
    pub num_views: usize,
    /// Classifier-free guidance scale for IP-Adapter conditioning (default: 3.0).
    ///
    /// Controls the strength of reference image conditioning. Must be >= 1.0.
    /// Higher values increase identity preservation but may reduce diversity.
    ///
    /// - **1.0**: No guidance (pure conditional)
    /// - **3.0-7.5**: Balanced (recommended)
    /// - **>10.0**: Strong conditioning (may oversaturate)
    pub guidance_scale: f64,
    /// Number of DDIM denoising steps (default: 50).
    pub num_inference_steps: usize,
    /// Number of latent upsampler denoising steps (default: 10).
    pub upsampler_steps: usize,
    /// Input/output image resolution before upscaling (default: 256).
    pub image_size: usize,
    /// Latent spatial size (image_size / 8).
    pub latent_size: usize,
    /// Number of latent channels produced by the VAE (default: 4).
    pub latent_channels: usize,
    /// U-Net input channels: latent_channels + normal-map latent channels (default: 8).
    pub unet_in_channels: usize,
    /// U-Net output channels (default: 4).
    pub unet_out_channels: usize,
    /// Cross-attention dimension (SD 2.1 = 1024).
    pub cross_attention_dim: usize,
    /// CLIP image embedding dimension (ViT-H/14 = 1280).
    pub clip_embed_dim: usize,
    /// Time embedding dimension (default: 1280).
    pub time_embed_dim: usize,
    /// Base channels for the U-Net (default: 320).
    pub base_channels: usize,
    /// Channel multipliers per U-Net stage.
    pub channel_mult: Vec<usize>,
    /// Layers per block in the U-Net.
    pub layers_per_block: usize,
    /// Number of attention heads per head-dim for each stage.
    pub attention_head_dim: Vec<usize>,
    /// Number of transformer blocks per attention stage.
    pub transformer_layers_per_block: Vec<usize>,
    /// Group-norm number of groups (default: 32).
    pub norm_num_groups: usize,
    /// Group-norm epsilon.
    pub norm_eps: f64,
    /// Camera pose input dimension (4×3 flattened = 12).
    pub camera_pose_dim: usize,
    /// Whether to use linear projection in spatial transformer.
    pub use_linear_projection: bool,
    /// VAE scaling factor for latent space.
    pub vae_scale_factor: f64,
    /// Whether to use flash attention for memory-efficient O(N) attention.
    /// When enabled, uses block-wise computation with online softmax.
    /// Falls back to standard O(N^2) attention when disabled.
    /// Default: true (when feature is enabled).
    pub use_flash_attention: bool,
    /// Block size for flash attention tiled computation. Larger blocks use more
    /// memory but may be faster due to better cache utilization. Default: 64.
    pub flash_attention_block_size: usize,
    /// Upsampler mode for latent upsampling (32×32 → 64×64).
    /// - None: No upsampling, output is 256×256
    /// - Some(SdX2): Use sd-x2-latent-upscaler, output is 512×512
    /// - Some(BilinearVae): Use bilinear upsampling, output is 512×512
    ///
    /// Default: None (256×256 output).
    pub upsampler_mode: Option<UpsamplerMode>,
    /// Whether to process VAE encode/decode sequentially (one chunk at a time)
    /// to reduce peak GPU memory. When false, all views are batched together.
    ///
    /// Default: false.
    pub sequential_vae: bool,
    /// Number of views per chunk when `sequential_vae` is true.
    ///
    /// Default: 1.
    pub vae_chunk_size: usize,
    /// Weight offloading strategy for low-VRAM inference.
    ///
    /// Default: `OffloadStrategy::AllInMemory`.
    pub offload_strategy: crate::weight_offload::OffloadStrategy,
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            num_views: 4,
            guidance_scale: 3.0,
            num_inference_steps: 50,
            upsampler_steps: 10,
            image_size: 256,
            latent_size: 32,
            latent_channels: 4,
            unet_in_channels: 8,
            unet_out_channels: 4,
            cross_attention_dim: 1024,
            clip_embed_dim: 1280,
            time_embed_dim: 1280,
            base_channels: 320,
            channel_mult: vec![1, 2, 4, 4],
            layers_per_block: 2,
            attention_head_dim: vec![5, 10, 20, 20],
            transformer_layers_per_block: vec![1, 1, 1, 1],
            norm_num_groups: 32,
            norm_eps: 1e-5,
            camera_pose_dim: 12,
            use_linear_projection: true,
            vae_scale_factor: 0.18215,
            // Flash attention is enabled by default when the feature is available
            #[cfg(feature = "flash_attention")]
            use_flash_attention: true,
            #[cfg(not(feature = "flash_attention"))]
            use_flash_attention: false,
            flash_attention_block_size: 64,
            upsampler_mode: None,
            sequential_vae: false,
            vae_chunk_size: 1,
            offload_strategy: crate::weight_offload::OffloadStrategy::AllInMemory,
        }
    }
}

impl DiffusionConfig {
    /// Channel count for a given U-Net stage index.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= self.num_stages()` (i.e. `stage >= self.channel_mult.len()`).
    /// Callers that cannot guarantee `stage` is in range should use
    /// [`Self::try_stage_channels`] instead, or call [`Self::validate`] once at
    /// construction time to catch a malformed config before this is ever reached.
    pub fn stage_channels(&self, stage: usize) -> usize {
        debug_assert!(
            stage < self.channel_mult.len(),
            "stage_channels: stage index {stage} out of range (channel_mult has {} entries)",
            self.channel_mult.len()
        );
        self.base_channels * self.channel_mult[stage]
    }

    /// Checked variant of [`Self::stage_channels`] that returns `None` instead
    /// of panicking when `stage` is out of range.
    pub fn try_stage_channels(&self, stage: usize) -> Option<usize> {
        self.channel_mult
            .get(stage)
            .map(|&mult| self.base_channels * mult)
    }

    /// Total number of U-Net stages.
    pub fn num_stages(&self) -> usize {
        self.channel_mult.len()
    }

    /// Validates internal consistency of the configuration.
    ///
    /// Checks that:
    /// - `num_views > 0`.
    /// - `guidance_scale >= 1.0`.
    /// - `image_size` is a positive multiple of 8, and `latent_size == image_size / 8`
    ///   (the VAE's implicit 8x downsampling factor).
    /// - `norm_num_groups > 0`, and every stage's channel count
    ///   (`stage_channels(i)` for `i` in `0..num_stages()`) is evenly divisible by it,
    ///   as required by `nn::group_norm`.
    /// - `attention_head_dim` and `transformer_layers_per_block` each have at least
    ///   `num_stages()` entries.
    ///
    /// # Errors
    ///
    /// Returns [`crate::DiffusionError::InvalidConfig`] describing the first check
    /// that fails.
    pub fn validate(&self) -> crate::DiffusionResult<()> {
        if self.num_views == 0 {
            return Err(crate::DiffusionError::InvalidConfig(
                "num_views must be > 0".to_string(),
            ));
        }
        if self.guidance_scale < 1.0 {
            return Err(crate::DiffusionError::InvalidConfig(format!(
                "guidance_scale must be >= 1.0, got {}",
                self.guidance_scale
            )));
        }
        if self.image_size == 0 || self.image_size % 8 != 0 {
            return Err(crate::DiffusionError::InvalidConfig(format!(
                "image_size must be a positive multiple of 8, got {}",
                self.image_size
            )));
        }
        let expected_latent_size = self.image_size / 8;
        if self.latent_size != expected_latent_size {
            return Err(crate::DiffusionError::InvalidConfig(format!(
                "latent_size ({}) must equal image_size / 8 ({expected_latent_size})",
                self.latent_size
            )));
        }
        if self.norm_num_groups == 0 {
            return Err(crate::DiffusionError::InvalidConfig(
                "norm_num_groups must be > 0".to_string(),
            ));
        }
        let num_stages = self.num_stages();
        if self.attention_head_dim.len() < num_stages {
            return Err(crate::DiffusionError::InvalidConfig(format!(
                "attention_head_dim has {} entries, need >= {num_stages} (num_stages)",
                self.attention_head_dim.len()
            )));
        }
        if self.transformer_layers_per_block.len() < num_stages {
            return Err(crate::DiffusionError::InvalidConfig(format!(
                "transformer_layers_per_block has {} entries, need >= {num_stages} (num_stages)",
                self.transformer_layers_per_block.len()
            )));
        }
        for stage in 0..num_stages {
            // Safe: stage is bounded by num_stages() == channel_mult.len().
            let ch = self.stage_channels(stage);
            if ch % self.norm_num_groups != 0 {
                return Err(crate::DiffusionError::InvalidConfig(format!(
                    "stage {stage} channel count {ch} is not divisible by norm_num_groups {}",
                    self.norm_num_groups
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validates_ok() {
        assert!(DiffusionConfig::default().validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_zero_num_views() {
        let config = DiffusionConfig {
            num_views: 0,
            ..DiffusionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_guidance_scale_below_one() {
        let config = DiffusionConfig {
            guidance_scale: 0.5,
            ..DiffusionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_latent_size_mismatch() {
        let config = DiffusionConfig {
            image_size: 512,
            // Should be 512 / 8 = 64.
            latent_size: 32,
            ..DiffusionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_accepts_consistent_latent_size() {
        let config = DiffusionConfig {
            image_size: 512,
            latent_size: 64,
            ..DiffusionConfig::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_norm_num_groups_zero() {
        let config = DiffusionConfig {
            norm_num_groups: 0,
            ..DiffusionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_channel_not_divisible_by_norm_groups() {
        let config = DiffusionConfig {
            base_channels: 33,
            norm_num_groups: 32,
            ..DiffusionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_short_attention_head_dim() {
        let config = DiffusionConfig {
            attention_head_dim: vec![5, 10],
            ..DiffusionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_try_stage_channels_valid_returns_some() {
        let config = DiffusionConfig::default();
        assert_eq!(config.try_stage_channels(0), Some(320));
    }

    #[test]
    fn test_try_stage_channels_out_of_range_returns_none() {
        let config = DiffusionConfig::default();
        assert_eq!(config.try_stage_channels(config.num_stages()), None);
    }

    #[test]
    #[should_panic]
    fn test_stage_channels_out_of_range_panics() {
        let config = DiffusionConfig::default();
        let _ = config.stage_channels(config.num_stages());
    }
}
