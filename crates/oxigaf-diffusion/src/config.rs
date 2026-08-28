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

/// Which attention kernel the U-Net's cross-attention layers run.
///
/// Before this existed, [`crate::attention::MultiViewSpatialTransformer`] was
/// always built through its non-flash constructor from
/// [`crate::unet::MultiViewUNet`], so [`DiffusionConfig::use_flash_attention`]
/// had no effect on a real pipeline run and
/// [`crate::sliced_attention::SlicedAttention`] was reachable only from its own
/// module's tests.
///
/// All variants compute the same mathematical function; they trade compute
/// against peak memory differently. Pick one with
/// [`DiffusionConfig::attention_backend`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionBackend {
    /// Materialise the full `(batch, heads, seq, ctx)` score matrix.
    ///
    /// Fastest for the short sequences of a 32×32 latent; memory grows as
    /// `seq × ctx`.
    #[default]
    Standard,

    /// Tiled attention with an online softmax, `O(N)` in memory.
    ///
    /// Requires the `flash_attention` feature; without it this behaves exactly
    /// like [`AttentionBackend::Standard`], because the kernel is not compiled
    /// in. Tile width comes from
    /// [`DiffusionConfig::flash_attention_block_size`].
    ///
    // The kernel's module is `#[cfg]`-gated, so an unconditional intra-doc
    // link to it is unresolvable in a build without the feature.
    #[cfg_attr(
        feature = "flash_attention",
        doc = "The kernel lives in [`mod@crate::flash_attention`]."
    )]
    #[cfg_attr(
        not(feature = "flash_attention"),
        doc = "The kernel lives in the `flash_attention` module, which this build does not compile."
    )]
    Flash,

    /// Chunked query attention ([`crate::sliced_attention`]): the score matrix
    /// is built for `slice_size` queries at a time.
    ///
    /// Bounds peak score-matrix memory at the cost of a host round trip per
    /// call — the kernel operates on flat `f32` buffers. Slice width comes from
    /// [`DiffusionConfig::attention_slice_size`].
    Sliced,
}

impl AttentionBackend {
    /// `true` when this backend needs the `flash_attention` feature to differ
    /// from [`AttentionBackend::Standard`].
    pub fn needs_flash_feature(self) -> bool {
        matches!(self, AttentionBackend::Flash)
    }

    /// Human-readable name, for logs and reports.
    pub fn display_name(self) -> &'static str {
        match self {
            AttentionBackend::Standard => "standard",
            AttentionBackend::Flash => "flash",
            AttentionBackend::Sliced => "sliced",
        }
    }
}

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
    /// Cross-attention **context** width (SD 2.1 = 1024).
    ///
    /// This is the width of *every* context the U-Net's cross-attention layers
    /// consume: the (null) text embedding fed to `attn2` **and**, after the
    /// CLIP encoder's IP projection, the image tokens fed to `attn_ip`. Read it
    /// through [`Self::ip_adapter_context_dim`] when what you mean is the
    /// latter — see that method for why the two are deliberately the same knob.
    pub cross_attention_dim: usize,
    /// Hidden width of the CLIP vision tower itself (ViT-H/14 = 1280).
    ///
    /// This is the **input** side of the IP-Adapter projection, i.e. the width
    /// of the encoder's own hidden states before
    /// [`crate::clip::ClipImageEncoder`] projects them down to
    /// [`Self::ip_adapter_context_dim`]. It is *not* the width the U-Net's
    /// `attn_ip` layer sees; reading it as such is what used to make the
    /// default configuration shape-error on its first denoising step.
    ///
    /// [`crate::clip::build_clip_encoder`] sizes the tower from this field via
    /// [`crate::clip::ClipVisionConfig::vit_h14_with_embed_dim`], so it must be
    /// a positive multiple of [`crate::clip::VIT_H14_HEAD_DIM`] (80) —
    /// [`Self::validate`] checks that.
    pub clip_embed_dim: usize,
    /// Time embedding dimension (default: 1280).
    pub time_embed_dim: usize,
    /// Base channels for the U-Net (default: 320).
    pub base_channels: usize,
    /// Channel multipliers per U-Net stage.
    pub channel_mult: Vec<usize>,
    /// Layers per block in the U-Net.
    pub layers_per_block: usize,
    /// **Number of attention heads** for each U-Net stage.
    ///
    /// The name comes from the diffusers SD 2.1 UNet config key of the same
    /// spelling, and is a misnomer there as well: `[5, 10, 20, 20]` are head
    /// *counts*, and the per-head dimension is derived as
    /// `stage_channels / num_heads` (64 for every SD 2.1 stage). Use
    /// [`Self::num_attention_heads`] and [`Self::attention_head_size`] rather
    /// than reading this field directly — reading it as a head *dimension*
    /// yields 64 heads of width 5, which leaves `inner_dim` (and therefore
    /// every projection weight shape) unchanged while computing attention over
    /// the wrong head partition with a `1/sqrt(5)` scale.
    ///
    /// Must have at least [`Self::num_stages`] entries; [`Self::validate`]
    /// checks that.
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
    ///
    /// This is the legacy two-state switch. [`Self::attention_backend`] is the
    /// general selector; [`Self::resolved_attention_backend`] combines the two
    /// and is what [`crate::unet::MultiViewUNet`] actually builds from.
    pub use_flash_attention: bool,
    /// Block size for flash attention tiled computation. Larger blocks use more
    /// memory but may be faster due to better cache utilization. Default: 64.
    pub flash_attention_block_size: usize,
    /// Which attention kernel the U-Net's cross-attention layers run.
    ///
    /// Defaults to [`AttentionBackend::Standard`], in which case
    /// [`Self::use_flash_attention`] still selects
    /// [`AttentionBackend::Flash`] — see
    /// [`Self::resolved_attention_backend`]. Set this to anything other than
    /// `Standard` and it wins outright.
    pub attention_backend: AttentionBackend,
    /// Number of queries per slice for [`AttentionBackend::Sliced`].
    ///
    /// `None` means "one slice", i.e. no slicing at all (identical to
    /// [`AttentionBackend::Standard`] but through the sliced kernel). Smaller
    /// values bound peak score-matrix memory more tightly at the cost of more
    /// passes. Default: `Some(64)`.
    pub attention_slice_size: Option<usize>,
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
            // `Standard` defers to `use_flash_attention`, preserving the
            // historical two-state behaviour for callers that never touch the
            // new field.
            attention_backend: AttentionBackend::Standard,
            attention_slice_size: Some(64),
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

    /// The attention backend the U-Net will actually build.
    ///
    /// [`Self::attention_backend`] wins whenever it is set to anything but
    /// [`AttentionBackend::Standard`]. When it is `Standard`, the legacy
    /// [`Self::use_flash_attention`] flag still selects
    /// [`AttentionBackend::Flash`], so existing configurations keep behaving
    /// the way their field names promise.
    ///
    /// Note that [`AttentionBackend::Flash`] only differs from `Standard` when
    /// the `flash_attention` feature is compiled in; without it the kernel does
    /// not exist and the standard path runs.
    pub fn resolved_attention_backend(&self) -> AttentionBackend {
        match self.attention_backend {
            AttentionBackend::Standard if self.use_flash_attention => AttentionBackend::Flash,
            other => other,
        }
    }

    /// Width of the IP-Adapter context: the token width
    /// [`crate::clip::ClipImageEncoder`] emits *and* the context width
    /// [`crate::unet::MultiViewUNet`] builds its `attn_ip` cross-attention for.
    ///
    /// These are one knob on purpose. IP-Adapter inserts a projection between
    /// the CLIP vision tower and the U-Net (`ImageProjModel` in the reference
    /// implementation): the tower emits [`Self::clip_embed_dim`]-wide hidden
    /// states, the projection maps them to the U-Net's cross-attention width,
    /// and `attn_ip`'s `to_k`/`to_v` consume *that*. So the only width the two
    /// sides have to agree on is the projection's output, and both
    /// [`crate::clip::build_clip_encoder`] and
    /// [`crate::unet::MultiViewUNet::new`] read it from here.
    ///
    /// Before this existed the two sides read different fields — the encoder
    /// projected to `cross_attention_dim` (1024) while the U-Net built
    /// `attn_ip` for `clip_embed_dim` (1280) — so a run on
    /// [`DiffusionConfig::default`] shape-errored inside the first
    /// `step_session` U-Net pass, and every
    /// [`crate::model_variants`] preset whose `cross_attention_dim` is not 1280
    /// (SD 1.5's 768, SDXL's 2048) was unusable for the same reason.
    pub fn ip_adapter_context_dim(&self) -> usize {
        self.cross_attention_dim
    }

    /// Number of attention heads for `stage`, or `None` when out of range.
    ///
    /// Correctly-named accessor for [`Self::attention_head_dim`], whose
    /// diffusers-inherited name says "dim" but whose entries are head
    /// *counts*. Prefer this over indexing the field.
    pub fn num_attention_heads(&self, stage: usize) -> Option<usize> {
        self.attention_head_dim.get(stage).copied()
    }

    /// Per-head attention width for `stage`: `stage_channels / num_heads`.
    ///
    /// Returns `None` when `stage` is out of range for either
    /// `channel_mult` or [`Self::attention_head_dim`], when the head count is
    /// `0`, or when the stage's channel count is not divisible by it — the
    /// same three conditions `unet::resolve_attention_heads` reports as
    /// errors when building the model.
    pub fn attention_head_size(&self, stage: usize) -> Option<usize> {
        let heads = self.num_attention_heads(stage)?;
        let channels = self.try_stage_channels(stage)?;
        if heads == 0 || !channels.is_multiple_of(heads) {
            return None;
        }
        Some(channels / heads)
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
    /// - `cross_attention_dim > 0` (it is also
    ///   [`Self::ip_adapter_context_dim`], so a zero width would leave both the
    ///   text and the IP-Adapter cross-attention with an empty context).
    /// - `clip_embed_dim` is a positive multiple of
    ///   [`crate::clip::VIT_H14_HEAD_DIM`] (80), the rule
    ///   [`crate::clip::ClipVisionConfig::vit_h14_with_embed_dim`] enforces when
    ///   [`crate::clip::build_clip_encoder`] sizes the vision tower.
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
        if self.image_size == 0 || !self.image_size.is_multiple_of(8) {
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
        // `cross_attention_dim` is also the IP-Adapter context width (see
        // `ip_adapter_context_dim`), so zero would give `attn2` *and* `attn_ip`
        // a zero-width context.
        if self.cross_attention_dim == 0 {
            return Err(crate::DiffusionError::InvalidConfig(
                "cross_attention_dim must be > 0 (it is also the IP-Adapter context width)"
                    .to_string(),
            ));
        }
        // `clip_embed_dim` is the CLIP vision tower's own hidden width, which
        // `clip::build_clip_encoder` feeds to
        // `ClipVisionConfig::vit_h14_with_embed_dim`. Checking the same rule
        // here turns a failure deep inside weight loading into an up-front,
        // actionable configuration error; the message deliberately matches.
        if self.clip_embed_dim == 0 {
            return Err(crate::DiffusionError::InvalidConfig(
                "clip_embed_dim must be > 0".to_string(),
            ));
        }
        if !self
            .clip_embed_dim
            .is_multiple_of(crate::clip::VIT_H14_HEAD_DIM)
        {
            return Err(crate::DiffusionError::InvalidConfig(format!(
                "clip_embed_dim must be a multiple of {} (ViT-H/14's head width), got {}",
                crate::clip::VIT_H14_HEAD_DIM,
                self.clip_embed_dim
            )));
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
            if !ch.is_multiple_of(self.norm_num_groups) {
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

    // IP-Adapter width contract.
    //
    // Regression: `clip::build_clip_encoder` projected its output to
    // `cross_attention_dim` while `unet::stage_transformer_spec` built the
    // matching `attn_ip` layer for `clip_embed_dim`, so the DEFAULT config
    // shape-errored on the first denoising step. There is now one accessor
    // both sides read.

    #[test]
    fn test_ip_adapter_context_dim_is_the_cross_attention_width() {
        let config = DiffusionConfig::default();
        assert_eq!(config.ip_adapter_context_dim(), config.cross_attention_dim);
        assert_eq!(config.ip_adapter_context_dim(), 1024);
        // The CLIP tower's own width is the projection's input, and it differs
        // — which is what made reading the wrong field fatal rather than
        // merely confusing.
        assert_eq!(config.clip_embed_dim, 1280);
        assert_ne!(config.clip_embed_dim, config.ip_adapter_context_dim());
    }

    #[test]
    fn test_ip_adapter_context_dim_tracks_a_custom_cross_attention_width() {
        // SD 1.5 / Zero123 presets carry a 768-wide context while the CLIP
        // tower stays ViT-H/14; the IP context must follow the U-Net, not the
        // tower.
        let config = DiffusionConfig {
            cross_attention_dim: 768,
            ..DiffusionConfig::default()
        };
        assert_eq!(config.ip_adapter_context_dim(), 768);
    }

    #[test]
    fn test_validate_rejects_zero_cross_attention_dim() {
        let config = DiffusionConfig {
            cross_attention_dim: 0,
            ..DiffusionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_zero_clip_embed_dim() {
        let config = DiffusionConfig {
            clip_embed_dim: 0,
            ..DiffusionConfig::default()
        };
        // `0.is_multiple_of(80)` is true, so the divisibility rule alone would
        // let a zero-width tower through; the explicit guard is what stops it.
        let err = config
            .validate()
            .expect_err("a zero-width CLIP tower has no weights to build");
        assert!(
            err.to_string().contains("clip_embed_dim must be > 0"),
            "zero must be reported as zero, not as a divisibility failure: {err}"
        );
    }

    // Regression: `clip::build_clip_encoder` hardcoded
    // `ClipVisionConfig::default()`, so `clip_embed_dim` never reached the
    // vision tower. Now that it does, a width the ViT-H/14 geometry cannot be
    // scaled to has to be rejected here rather than deep inside weight loading.

    #[test]
    fn test_validate_rejects_clip_embed_dim_off_the_head_width() {
        // 1024 (ViT-L/14) and 768 (ViT-B) are real CLIP widths, but neither is
        // a multiple of ViT-H/14's 80-wide head.
        for clip_embed_dim in [768usize, 1024, 81] {
            let config = DiffusionConfig {
                clip_embed_dim,
                ..DiffusionConfig::default()
            };
            let err = config.validate().expect_err(
                "a clip_embed_dim off ViT-H/14's head width must not reach build_clip_encoder",
            );
            assert!(
                err.to_string().contains("multiple of 80"),
                "error should name the rule for {clip_embed_dim}, got: {err}"
            );
        }
    }

    #[test]
    fn test_validate_accepts_clip_embed_dim_multiples_of_the_head_width() {
        for clip_embed_dim in [80usize, 640, 1280, 2560] {
            let config = DiffusionConfig {
                clip_embed_dim,
                ..DiffusionConfig::default()
            };
            assert!(
                config.validate().is_ok(),
                "clip_embed_dim={clip_embed_dim} is a multiple of {}",
                crate::clip::VIT_H14_HEAD_DIM
            );
        }
    }

    #[test]
    fn test_validate_rejects_short_attention_head_dim() {
        let config = DiffusionConfig {
            attention_head_dim: vec![5, 10],
            ..DiffusionConfig::default()
        };
        assert!(config.validate().is_err());
    }

    // Attention backend selection.
    //
    // Regression: `MultiViewUNet` always built its transformers through the
    // non-flash constructor, so `use_flash_attention` had no effect on a real
    // run and `SlicedAttention` was unreachable outside its own tests.

    #[test]
    fn test_default_config_resolves_to_a_standard_backend() {
        let config = DiffusionConfig::default();
        assert_eq!(config.attention_backend, AttentionBackend::Standard);
        // Without the flash_attention feature `use_flash_attention` defaults
        // to false, so the resolved backend follows the default field.
        let expected = if config.use_flash_attention {
            AttentionBackend::Flash
        } else {
            AttentionBackend::Standard
        };
        assert_eq!(config.resolved_attention_backend(), expected);
    }

    #[test]
    fn test_legacy_flash_flag_still_selects_flash() {
        let config = DiffusionConfig {
            use_flash_attention: true,
            attention_backend: AttentionBackend::Standard,
            ..DiffusionConfig::default()
        };
        assert_eq!(config.resolved_attention_backend(), AttentionBackend::Flash);
    }

    #[test]
    fn test_explicit_backend_overrides_the_legacy_flag() {
        // Sliced was chosen deliberately; the legacy flag must not win.
        let config = DiffusionConfig {
            use_flash_attention: true,
            attention_backend: AttentionBackend::Sliced,
            ..DiffusionConfig::default()
        };
        assert_eq!(
            config.resolved_attention_backend(),
            AttentionBackend::Sliced
        );

        let flash = DiffusionConfig {
            use_flash_attention: false,
            attention_backend: AttentionBackend::Flash,
            ..DiffusionConfig::default()
        };
        assert_eq!(flash.resolved_attention_backend(), AttentionBackend::Flash);
    }

    #[test]
    fn test_attention_backend_metadata() {
        assert!(AttentionBackend::Flash.needs_flash_feature());
        assert!(!AttentionBackend::Standard.needs_flash_feature());
        assert!(!AttentionBackend::Sliced.needs_flash_feature());
        assert_eq!(AttentionBackend::default(), AttentionBackend::Standard);
        assert_eq!(AttentionBackend::Sliced.display_name(), "sliced");
    }

    // `attention_head_dim` holds head COUNTS despite its diffusers-inherited
    // name; these accessors exist so callers never have to know that.

    #[test]
    fn test_num_attention_heads_matches_sd21_head_counts() {
        let config = DiffusionConfig::default();
        assert_eq!(config.num_attention_heads(0), Some(5));
        assert_eq!(config.num_attention_heads(3), Some(20));
        assert_eq!(config.num_attention_heads(config.num_stages()), None);
    }

    #[test]
    fn test_attention_head_size_is_sixty_four_for_every_sd21_stage() {
        let config = DiffusionConfig::default();
        for stage in 0..config.num_stages() {
            assert_eq!(
                config.attention_head_size(stage),
                Some(64),
                "stage {stage} head width"
            );
            let heads = config
                .num_attention_heads(stage)
                .expect("head count in range");
            let size = config
                .attention_head_size(stage)
                .expect("head size in range");
            assert_eq!(
                heads * size,
                config.stage_channels(stage),
                "inner_dim must equal stage channels"
            );
        }
    }

    #[test]
    fn test_attention_head_size_rejects_zero_and_indivisible_counts() {
        let zero = DiffusionConfig {
            attention_head_dim: vec![0, 10, 20, 20],
            ..DiffusionConfig::default()
        };
        assert_eq!(zero.attention_head_size(0), None);

        let indivisible = DiffusionConfig {
            attention_head_dim: vec![7, 10, 20, 20],
            ..DiffusionConfig::default()
        };
        assert_eq!(indivisible.attention_head_size(0), None);
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
