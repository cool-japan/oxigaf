//! Model variant configurations for different diffusion architectures.
//!
//! This module defines configuration variants for Stable Diffusion 1.x, 2.x, XL,
//! Zero123, and custom architectures. It provides tooling to select, validate,
//! and adapt these configurations for the OxiGAF multi-view diffusion pipeline.
//!
//! ## Supported Architectures
//!
//! | Family  | Native Res | Cross-Attn Dim | Encoder    |
//! |---------|-----------|----------------|------------|
//! | SD 1.x  | 512px     | 768            | Single     |
//! | SD 2.x  | 768px     | 1024           | Single     |
//! | SDXL    | 1024px    | 2048           | Dual       |
//! | Zero123 | 256px     | 768            | Single     |
//!
//! ## Example
//!
//! ```rust
//! use oxigaf_diffusion::model_variants::{ModelVariantRegistry, ModelFamily, sd15_config};
//!
//! let registry = ModelVariantRegistry::new();
//! let config = registry.get("sd15").expect("sd15 config not found");
//! assert_eq!(config.cross_attention_dim, 768);
//!
//! let family = ModelFamily::from_str("sdxl").expect("invalid family");
//! assert_eq!(family.native_resolution(), 1024);
//! ```

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when working with model variant configurations.
#[derive(Debug, Error, PartialEq)]
pub enum ModelVariantError {
    /// The requested model variant name is not recognized.
    #[error("Unknown model variant: '{0}'")]
    UnknownVariant(String),

    /// Two configurations have incompatible parameters that prevent interoperability.
    #[error("Incompatible configuration: {0}")]
    IncompatibleConfig(String),

    /// A configuration parameter has an invalid value.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// The requested operation is not supported for this variant.
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

// ---------------------------------------------------------------------------
// ModelFamily
// ---------------------------------------------------------------------------

/// The high-level family of diffusion model architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFamily {
    /// Stable Diffusion 1.x (512px native, 4-channel latent).
    SD1,
    /// Stable Diffusion 2.x (768px native, 4-channel latent).
    SD2,
    /// Stable Diffusion XL (1024px native, 4-channel latent, dual encoder).
    SDXL,
    /// Zero123 / Zero-1-to-3 (single-image to novel view, 256px native).
    Zero123,
    /// Custom / user-defined architecture.
    Custom,
}

impl ModelFamily {
    /// Parse a family name from a string identifier.
    ///
    /// Recognized identifiers: `"sd1"`, `"sd1.5"`, `"sd2"`, `"sd2.1"`,
    /// `"sdxl"`, `"zero123"`, `"custom"`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelVariantError::UnknownVariant`] if the string is not
    /// a recognized identifier.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self, ModelVariantError> {
        match s.to_ascii_lowercase().as_str() {
            "sd1" | "sd1.5" | "sd1x" => Ok(Self::SD1),
            "sd2" | "sd2.1" | "sd2x" => Ok(Self::SD2),
            "sdxl" | "stable-diffusion-xl" => Ok(Self::SDXL),
            "zero123" | "zero-1-to-3" => Ok(Self::Zero123),
            "custom" => Ok(Self::Custom),
            other => Err(ModelVariantError::UnknownVariant(other.to_string())),
        }
    }

    /// Return a canonical string representation of this family.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SD1 => "sd1",
            Self::SD2 => "sd2",
            Self::SDXL => "sdxl",
            Self::Zero123 => "zero123",
            Self::Custom => "custom",
        }
    }

    /// The native (trained) image resolution for this family, in pixels.
    ///
    /// - SD1: 512
    /// - SD2: 768
    /// - SDXL: 1024
    /// - Zero123: 256
    /// - Custom: 512 (default)
    pub fn native_resolution(&self) -> u32 {
        match self {
            Self::SD1 => 512,
            Self::SD2 => 768,
            Self::SDXL => 1024,
            Self::Zero123 => 256,
            Self::Custom => 512,
        }
    }

    /// Number of latent channels produced by the VAE for this family.
    ///
    /// All current variants use 4 latent channels.
    pub fn latent_channels(&self) -> usize {
        4
    }

    /// The VAE scale factor: latent_size = image_size / vae_scale_factor.
    ///
    /// All current variants use a scale factor of 8.
    pub fn vae_scale_factor(&self) -> u32 {
        8
    }
}

// ---------------------------------------------------------------------------
// UNetVariantConfig
// ---------------------------------------------------------------------------

/// Core U-Net architecture configuration for a particular model variant.
///
/// This struct captures all the architectural hyperparameters that distinguish
/// different Stable Diffusion variants from one another.
///
/// # Examples
///
/// ```rust
/// use oxigaf_diffusion::model_variants::sd15_config;
///
/// let cfg = sd15_config();
/// assert_eq!(cfg.model_channels, 320);
/// assert_eq!(cfg.channels_at_level(1), 640);
/// ```
#[derive(Debug, Clone)]
pub struct UNetVariantConfig {
    /// Which model family this configuration belongs to.
    pub family: ModelFamily,

    /// Base channel count (SD1: 320, SD2: 320, SDXL: 320).
    pub model_channels: usize,

    /// Channel multiplier per resolution level.
    ///
    /// The actual channel count at level `i` is `model_channels * channel_mult[i]`.
    /// - SD1/2: `[1, 2, 4, 4]`
    /// - SDXL: `[1, 2, 4]`
    pub channel_mult: Vec<usize>,

    /// Number of ResNet blocks per resolution level (SD1/2/XL: 2).
    pub num_res_blocks: usize,

    /// Resolution levels that include attention layers.
    ///
    /// Values are downsampling factors: 4 means 1/4 of input resolution.
    /// - SD1/2: `[4, 2, 1]`
    /// - SDXL: `[4, 2]`
    pub attention_resolutions: Vec<usize>,

    /// Channels per attention head, used uniformly across all levels when set.
    ///
    /// `Some` for SD1.5/Zero123 (uniform 64-dim head, head *count* varies per
    /// level as `channels_at_level(i) / 64`). `None` for SD2.1/SDXL, which
    /// instead specify an explicit head count per level via
    /// [`Self::num_heads_per_level`].
    pub head_channels: Option<usize>,

    /// Number of transformer layers per resolution level.
    ///
    /// - SD1/2: `[1, 1, 1, 1]`
    /// - SDXL: `[1, 2, 10]` (one entry per down-level, matching `channel_mult`)
    pub transformer_depth: Vec<usize>,

    /// Cross-attention embedding dimension for CLIP conditioning.
    ///
    /// - SD1: 768  (ViT-L/14)
    /// - SD2: 1024 (ViT-H/14)
    /// - SDXL: 2048 (dual-encoder concatenation)
    pub cross_attention_dim: usize,

    /// Number of input channels (4 for normal latent, 8 for inpainting).
    pub in_channels: usize,

    /// Number of output channels (4 for noise prediction).
    pub out_channels: usize,

    /// Whether to use linear projection in spatial transformer blocks (SD2+).
    pub use_linear_projection: bool,

    /// Whether to use flash attention for memory-efficient O(N) attention.
    pub use_flash_attention: bool,

    /// Explicit number of attention heads per resolution level.
    ///
    /// Used when [`Self::head_channels`] is `None`: SD2.1 uses `[5, 10, 20, 20]`,
    /// SDXL uses `[5, 10, 20]` (both give a uniform 64-dim head, since
    /// `channels_at_level(i) / num_heads_per_level[i] == 64` at every level).
    /// Empty when the config instead uses a uniform [`Self::head_channels`]
    /// (SD1.5, Zero123).
    pub num_heads_per_level: Vec<usize>,
}

impl UNetVariantConfig {
    /// Compute the number of channels at the given resolution level.
    ///
    /// Returns `model_channels * channel_mult[level]`, falling back to
    /// `model_channels` if `level` is out of range.
    pub fn channels_at_level(&self, level: usize) -> usize {
        self.model_channels * self.channel_mult.get(level).copied().unwrap_or(1)
    }

    /// Number of resolution levels in the U-Net.
    pub fn num_levels(&self) -> usize {
        self.channel_mult.len()
    }

    /// A coarse **relative sizing index**, not a real parameter count.
    ///
    /// Uses the heuristic:
    /// `model_channels² × Σᵢ(channel_mult[i] × (num_res_blocks×2 + transformer_depth[i])) × 4`
    ///
    /// This omits attention QKV/output projections, cross-attention layers,
    /// GroupNorm affine parameters, timestep/label embeddings, and conv bias
    /// terms, so its absolute value is **not** the model's real parameter
    /// count — it is roughly 40-90x smaller, and the gap is not a constant
    /// multiplier (it grows with transformer depth, so SDXL is further off
    /// than SD1.5). For reference, the real published U-Net parameter counts
    /// are approximately:
    /// - SD1.5 ≈ 860M (this heuristic: ≈ 22.5M)
    /// - SD2.1 ≈ 865M (this heuristic: ≈ 22.5M)
    /// - SDXL ≈ 2.6B (this heuristic: ≈ 29.9M)
    ///
    /// Use this value only to *compare* configurations against each other
    /// (e.g. [`are_configs_compatible`]-adjacent sizing checks, or confirming
    /// that one variant is architecturally larger than another) — never as a
    /// stand-in for a real parameter count or a memory-sizing budget on its
    /// own (see [`Self::estimated_vram_mb`] for a memory estimate that does
    /// not depend on this being accurate in absolute terms).
    pub fn estimated_params(&self) -> u64 {
        let mc = self.model_channels as u64;
        let level_sum: u64 = self
            .channel_mult
            .iter()
            .enumerate()
            .map(|(i, &mult)| {
                let td = self.transformer_depth.get(i).copied().unwrap_or(1) as u64;
                mult as u64 * (self.num_res_blocks as u64 * 2 + td)
            })
            .sum();
        mc * mc * level_sum * 4
    }

    /// Estimated VRAM usage for inference (in MB) at a given image size and batch.
    ///
    /// Sums three explicit, per-level components instead of a global fudge
    /// factor:
    /// - **Latent bytes**: `(image_size/8)² × latent_channels × batch × 4`, the
    ///   input/output latent tensor itself.
    /// - **Activation bytes**: for each resolution level `i` (spatial side
    ///   halved once per level relative to the latent, per the standard U-Net
    ///   downsampling schedule), `side² × channels_at_level(i) × batch × 4`,
    ///   counted twice per ResNet block (`num_res_blocks`) to cover both the
    ///   block's hidden activation and its output/skip-connection buffer.
    /// - **Attention bytes**: for levels present in `attention_resolutions`
    ///   (tracked via the running downsample factor `ds`, matching the
    ///   original latent-diffusion `ds ∈ attention_resolutions` convention),
    ///   the `seq_len² × batch × 4` score matrix per transformer layer at
    ///   that level, where `seq_len = side²`.
    /// - **Weight bytes**: `estimated_params() × 2` (fp16 weights). Since
    ///   [`Self::estimated_params`] is a relative index rather than a true
    ///   parameter count, this term is a lower bound, not an exact figure.
    pub fn estimated_vram_mb(&self, image_size: u32, batch_size: usize) -> u64 {
        let latent_side = (image_size / 8) as u64;
        let batch = batch_size as u64;

        let latent_bytes =
            latent_side * latent_side * self.family.latent_channels() as u64 * batch * 4;

        let mut activation_bytes: u64 = 0;
        let mut ds: usize = 1;
        for level in 0..self.num_levels() {
            let side = (latent_side >> level).max(1);
            let channels = self.channels_at_level(level) as u64;
            let per_block = side * side * channels * batch * 4;
            activation_bytes += per_block * 2 * self.num_res_blocks as u64;

            if self.attention_resolutions.contains(&ds) {
                let seq_len = side * side;
                let td = self.transformer_depth.get(level).copied().unwrap_or(1) as u64;
                activation_bytes += seq_len * seq_len * batch * 4 * td.max(1);
            }
            ds = ds.saturating_mul(2);
        }

        let weight_bytes = self.estimated_params() * 2;

        (latent_bytes + activation_bytes + weight_bytes) / (1024 * 1024)
    }

    /// Returns `true` if this configuration uses dual CLIP encoders (SDXL only).
    pub fn uses_dual_encoder(&self) -> bool {
        matches!(self.family, ModelFamily::SDXL)
    }

    /// Returns a human-readable summary of the configuration.
    pub fn format_summary(&self) -> String {
        format!(
            "UNetVariantConfig {{ family: {:?}, channels: {}, levels: {}, \
             cross_attn_dim: {}, in/out: {}/{}, dual_encoder: {}, \
             size_index: {}M, linear_proj: {} }}",
            self.family,
            self.model_channels,
            self.num_levels(),
            self.cross_attention_dim,
            self.in_channels,
            self.out_channels,
            self.uses_dual_encoder(),
            self.estimated_params() / 1_000_000,
            self.use_linear_projection,
        )
    }

    /// Resolve the number of attention heads at each resolution level.
    ///
    /// Returns [`Self::num_heads_per_level`] directly when it is non-empty
    /// (SD2.1, SDXL). Otherwise derives a per-level head count from
    /// [`Self::head_channels`] (SD1.5, Zero123: `channels_at_level(i) /
    /// head_channels`), falling back to a 64-dim head when neither is set.
    /// The result always has exactly [`Self::num_levels`] entries.
    fn head_counts_per_level(&self) -> Vec<usize> {
        if !self.num_heads_per_level.is_empty() {
            return (0..self.num_levels())
                .map(|i| self.num_heads_per_level.get(i).copied().unwrap_or(1).max(1))
                .collect();
        }
        let hc = self.head_channels.unwrap_or(64).max(1);
        (0..self.num_levels())
            .map(|level| (self.channels_at_level(level) / hc).max(1))
            .collect()
    }

    /// Convert this U-Net variant configuration into a
    /// [`crate::config::DiffusionConfig`] that [`crate::unet::MultiViewUNet`]
    /// and [`crate::pipeline::MultiViewDiffusionPipeline`] can actually
    /// consume.
    ///
    /// Maps `model_channels` → `base_channels`, `channel_mult`,
    /// `transformer_depth` → `transformer_layers_per_block`,
    /// `cross_attention_dim`, `in_channels`/`out_channels`, and
    /// `use_linear_projection`/`use_flash_attention` directly. Per-level
    /// attention head counts come from [`Self::head_counts_per_level`].
    /// `image_size`/`latent_size` are derived from `self.family`. Fields not
    /// captured by this configuration at all (guidance scale, inference step
    /// count, VAE normalization scale, offload strategy, ...) are taken from
    /// [`crate::config::DiffusionConfig::default`].
    ///
    /// # Errors
    ///
    /// Returns [`ModelVariantError::InvalidConfig`] if the resulting
    /// configuration fails [`crate::config::DiffusionConfig::validate`] (for
    /// example, a hand-built custom variant whose `model_channels` is not
    /// divisible by the default `norm_num_groups`).
    pub fn to_diffusion_config(
        &self,
        num_views: usize,
    ) -> Result<crate::config::DiffusionConfig, ModelVariantError> {
        let num_levels = self.num_levels();
        let mut transformer_layers_per_block: Vec<usize> = self
            .transformer_depth
            .iter()
            .copied()
            .take(num_levels)
            .collect();
        while transformer_layers_per_block.len() < num_levels {
            transformer_layers_per_block.push(1);
        }

        let native_res = self.family.native_resolution();
        let vae_scale = self.family.vae_scale_factor();

        let config = crate::config::DiffusionConfig {
            num_views,
            image_size: native_res as usize,
            latent_size: (native_res / vae_scale) as usize,
            latent_channels: self.family.latent_channels(),
            unet_in_channels: self.in_channels,
            unet_out_channels: self.out_channels,
            cross_attention_dim: self.cross_attention_dim,
            base_channels: self.model_channels,
            channel_mult: self.channel_mult.clone(),
            layers_per_block: self.num_res_blocks,
            attention_head_dim: self.head_counts_per_level(),
            transformer_layers_per_block,
            use_linear_projection: self.use_linear_projection,
            use_flash_attention: self.use_flash_attention,
            ..crate::config::DiffusionConfig::default()
        };
        config
            .validate()
            .map_err(|e| ModelVariantError::InvalidConfig(e.to_string()))?;
        Ok(config)
    }
}

// ---------------------------------------------------------------------------
// Standard variant constructors
// ---------------------------------------------------------------------------

/// Configuration for Stable Diffusion 1.5.
///
/// - 320 base channels, 4 levels `[1, 2, 4, 4]`
/// - Cross-attention dim: 768 (ViT-L/14 CLIP)
/// - 64 channels per attention head
/// - No linear projection
pub fn sd15_config() -> UNetVariantConfig {
    UNetVariantConfig {
        family: ModelFamily::SD1,
        model_channels: 320,
        channel_mult: vec![1, 2, 4, 4],
        num_res_blocks: 2,
        attention_resolutions: vec![4, 2, 1],
        head_channels: Some(64),
        transformer_depth: vec![1, 1, 1, 1],
        cross_attention_dim: 768,
        in_channels: 4,
        out_channels: 4,
        use_linear_projection: false,
        use_flash_attention: false,
        num_heads_per_level: Vec::new(),
    }
}

/// Configuration for Stable Diffusion 2.1.
///
/// - 320 base channels, 4 levels `[1, 2, 4, 4]`
/// - Cross-attention dim: 1024 (ViT-H/14 CLIP)
/// - Per-level attention heads `[5, 10, 20, 20]` (uniform 64-dim heads),
///   linear projection enabled
pub fn sd21_config() -> UNetVariantConfig {
    UNetVariantConfig {
        family: ModelFamily::SD2,
        model_channels: 320,
        channel_mult: vec![1, 2, 4, 4],
        num_res_blocks: 2,
        attention_resolutions: vec![4, 2, 1],
        head_channels: None,
        transformer_depth: vec![1, 1, 1, 1],
        cross_attention_dim: 1024,
        in_channels: 4,
        out_channels: 4,
        use_linear_projection: true,
        use_flash_attention: false,
        num_heads_per_level: vec![5, 10, 20, 20],
    }
}

/// Configuration for Stable Diffusion XL base U-Net (not the refiner).
///
/// - 320 base channels, 3 levels `[1, 2, 4]`
/// - Cross-attention dim: 2048 (dual CLIP encoder concatenation)
/// - Per-level attention heads `[5, 10, 20]` (uniform 64-dim heads), linear
///   projection enabled
/// - Deeper transformer blocks at inner levels: `[1, 2, 10]` (one entry per
///   down-level, matching `channel_mult`; level 0 has no attention at all,
///   per `attention_resolutions`)
pub fn sdxl_config() -> UNetVariantConfig {
    UNetVariantConfig {
        family: ModelFamily::SDXL,
        model_channels: 320,
        channel_mult: vec![1, 2, 4],
        num_res_blocks: 2,
        attention_resolutions: vec![4, 2],
        head_channels: None,
        transformer_depth: vec![1, 2, 10],
        cross_attention_dim: 2048,
        in_channels: 4,
        out_channels: 4,
        use_linear_projection: true,
        use_flash_attention: false,
        num_heads_per_level: vec![5, 10, 20],
    }
}

/// Configuration for Zero123 (Zero-1-to-3) novel-view synthesis.
///
/// Based on SD 1.5 with reduced channel count (256) optimized for
/// single-image-to-novel-view generation.
pub fn zero123_config() -> UNetVariantConfig {
    UNetVariantConfig {
        family: ModelFamily::Zero123,
        model_channels: 256,
        channel_mult: vec![1, 2, 4, 4],
        num_res_blocks: 2,
        attention_resolutions: vec![4, 2, 1],
        head_channels: Some(64),
        transformer_depth: vec![1, 1, 1, 1],
        cross_attention_dim: 768,
        in_channels: 4,
        out_channels: 4,
        use_linear_projection: false,
        use_flash_attention: false,
        num_heads_per_level: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// MultiViewAdapterConfig
// ---------------------------------------------------------------------------

/// Configuration for adapting a single-view U-Net to multi-view generation.
///
/// Adds cross-view attention layers between views, optionally conditioned on
/// camera poses (azimuth, elevation, focal length encoded as Fourier features).
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::model_variants::{MultiViewAdapterConfig, sd15_config};
///
/// let base = sd15_config();
/// let adapter = MultiViewAdapterConfig::from_base(base, 4);
/// adapter.validate().expect("validation failed");
/// assert_eq!(adapter.num_views, 4);
/// ```
#[derive(Debug, Clone)]
pub struct MultiViewAdapterConfig {
    /// The underlying single-view U-Net configuration.
    pub base: UNetVariantConfig,

    /// Number of views to process simultaneously.
    pub num_views: usize,

    /// Resolution levels that receive cross-view attention layers.
    pub cross_view_attention_levels: Vec<usize>,

    /// Embedding dimension for cross-view attention (usually == `cross_attention_dim`).
    pub cross_view_dim: usize,

    /// Whether to condition cross-view attention on camera pose embeddings.
    pub use_camera_conditioning: bool,

    /// Dimension of camera pose embeddings.
    ///
    /// Encodes azimuth, elevation, and focal length as Fourier features.
    /// Default: 16.
    pub camera_embed_dim: usize,
}

impl MultiViewAdapterConfig {
    /// Construct a default multi-view adapter from a base U-Net configuration.
    ///
    /// Cross-view attention is inserted at the same resolution levels as
    /// standard attention. Camera conditioning is enabled with 16-dim Fourier
    /// embeddings for azimuth, elevation, and focal length.
    pub fn from_base(base: UNetVariantConfig, num_views: usize) -> Self {
        let cross_view_attention_levels = base.attention_resolutions.clone();
        let cross_view_dim = base.cross_attention_dim;
        Self {
            base,
            num_views,
            cross_view_attention_levels,
            cross_view_dim,
            use_camera_conditioning: true,
            camera_embed_dim: 16,
        }
    }

    /// Rough estimate of the additional parameter count introduced by
    /// cross-view attention layers.
    ///
    /// Heuristic:
    /// `cross_attention_dim² × 4 × len(levels) × num_views`
    pub fn estimated_extra_params(&self) -> u64 {
        let dim = self.cross_view_dim as u64;
        let levels = self.cross_view_attention_levels.len() as u64;
        let views = self.num_views as u64;
        dim * dim * 4 * levels * views
    }

    /// Validate this multi-view adapter configuration.
    ///
    /// # Errors
    ///
    /// - [`ModelVariantError::InvalidConfig`] if `num_views < 2` (single-view
    ///   does not need a multi-view adapter).
    /// - [`ModelVariantError::InvalidConfig`] if `cross_view_attention_levels`
    ///   is empty.
    pub fn validate(&self) -> Result<(), ModelVariantError> {
        if self.num_views < 2 {
            return Err(ModelVariantError::InvalidConfig(format!(
                "num_views must be >= 2, got {}",
                self.num_views
            )));
        }
        if self.cross_view_attention_levels.is_empty() {
            return Err(ModelVariantError::InvalidConfig(
                "cross_view_attention_levels must not be empty".to_string(),
            ));
        }
        Ok(())
    }

    /// Convert [`Self::base`] into a [`crate::config::DiffusionConfig`], using
    /// [`Self::num_views`] for the resulting config's view count.
    ///
    /// This wires the adapter's base U-Net configuration into the crate's
    /// real pipeline configuration type (see
    /// [`UNetVariantConfig::to_diffusion_config`]). Cross-view attention
    /// placement (`cross_view_attention_levels`, `cross_view_dim`) and camera
    /// pose conditioning (`use_camera_conditioning`, `camera_embed_dim`) are
    /// not yet modeled by [`crate::config::DiffusionConfig`] /
    /// [`crate::unet::MultiViewUNet`], so they are not applied by this
    /// conversion; wiring those into the U-Net's actual layer construction is
    /// tracked as a follow-up.
    ///
    /// # Errors
    ///
    /// Returns [`ModelVariantError::InvalidConfig`] under the same conditions
    /// as [`UNetVariantConfig::to_diffusion_config`].
    pub fn to_diffusion_config(&self) -> Result<crate::config::DiffusionConfig, ModelVariantError> {
        self.base.to_diffusion_config(self.num_views)
    }
}

// ---------------------------------------------------------------------------
// ModelVariantRegistry
// ---------------------------------------------------------------------------

/// A registry of named U-Net variant configurations.
///
/// Pre-populated with the four standard variants (`"sd15"`, `"sd21"`,
/// `"sdxl"`, `"zero123"`) and supports runtime registration of custom variants.
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::model_variants::{ModelVariantRegistry, ModelFamily};
///
/// let registry = ModelVariantRegistry::new();
/// assert!(registry.get("sd15").is_some());
/// let sdxl = ModelVariantRegistry::default_for_family(ModelFamily::SDXL);
/// assert_eq!(sdxl.cross_attention_dim, 2048);
/// ```
pub struct ModelVariantRegistry {
    variants: HashMap<String, UNetVariantConfig>,
}

impl ModelVariantRegistry {
    /// Create a new registry pre-populated with all standard variants.
    pub fn new() -> Self {
        let mut variants = HashMap::new();
        variants.insert("sd15".to_string(), sd15_config());
        variants.insert("sd21".to_string(), sd21_config());
        variants.insert("sdxl".to_string(), sdxl_config());
        variants.insert("zero123".to_string(), zero123_config());
        Self { variants }
    }

    /// Register a new variant under the given name.
    ///
    /// # Errors
    ///
    /// Returns [`ModelVariantError::InvalidConfig`] if `name` is empty.
    pub fn register(
        &mut self,
        name: String,
        config: UNetVariantConfig,
    ) -> Result<(), ModelVariantError> {
        if name.is_empty() {
            return Err(ModelVariantError::InvalidConfig(
                "Variant name must not be empty".to_string(),
            ));
        }
        self.variants.insert(name, config);
        Ok(())
    }

    /// Look up a variant configuration by name.
    pub fn get(&self, name: &str) -> Option<&UNetVariantConfig> {
        self.variants.get(name)
    }

    /// Return all registered variant names, sorted alphabetically.
    pub fn list_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.variants.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// Return the default configuration for a given model family.
    ///
    /// - [`ModelFamily::SD1`] → [`sd15_config`]
    /// - [`ModelFamily::SD2`] → [`sd21_config`]
    /// - [`ModelFamily::SDXL`] → [`sdxl_config`]
    /// - [`ModelFamily::Zero123`] → [`zero123_config`]
    /// - [`ModelFamily::Custom`] → [`sd15_config`] (fallback)
    pub fn default_for_family(family: ModelFamily) -> UNetVariantConfig {
        match family {
            ModelFamily::SD1 => sd15_config(),
            ModelFamily::SD2 => sd21_config(),
            ModelFamily::SDXL => sdxl_config(),
            ModelFamily::Zero123 => zero123_config(),
            ModelFamily::Custom => sd15_config(),
        }
    }
}

impl Default for ModelVariantRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Compatibility checking
// ---------------------------------------------------------------------------

/// Check whether two U-Net configurations are structurally compatible.
///
/// Two configs are considered compatible when they share the same:
/// - `model_channels` (base channel count)
/// - `num_levels()` (number of resolution levels)
/// - `in_channels` and `out_channels`
///
/// Compatibility does **not** require identical cross-attention dimensions or
/// transformer depths.
pub fn are_configs_compatible(a: &UNetVariantConfig, b: &UNetVariantConfig) -> bool {
    a.model_channels == b.model_channels
        && a.num_levels() == b.num_levels()
        && a.in_channels == b.in_channels
        && a.out_channels == b.out_channels
}

/// Describe the differences between two U-Net configurations.
///
/// Returns a list of human-readable strings, one per differing field.
/// Returns an empty `Vec` when the two configurations are identical in all
/// compared fields.
pub fn config_diff(a: &UNetVariantConfig, b: &UNetVariantConfig) -> Vec<String> {
    let mut diffs = Vec::new();

    if a.family != b.family {
        diffs.push(format!("family: {:?} vs {:?}", a.family, b.family));
    }
    if a.model_channels != b.model_channels {
        diffs.push(format!(
            "model_channels: {} vs {}",
            a.model_channels, b.model_channels
        ));
    }
    if a.channel_mult != b.channel_mult {
        diffs.push(format!(
            "channel_mult: {:?} vs {:?}",
            a.channel_mult, b.channel_mult
        ));
    }
    if a.num_res_blocks != b.num_res_blocks {
        diffs.push(format!(
            "num_res_blocks: {} vs {}",
            a.num_res_blocks, b.num_res_blocks
        ));
    }
    if a.attention_resolutions != b.attention_resolutions {
        diffs.push(format!(
            "attention_resolutions: {:?} vs {:?}",
            a.attention_resolutions, b.attention_resolutions
        ));
    }
    if a.head_channels != b.head_channels {
        diffs.push(format!(
            "head_channels: {:?} vs {:?}",
            a.head_channels, b.head_channels
        ));
    }
    if a.transformer_depth != b.transformer_depth {
        diffs.push(format!(
            "transformer_depth: {:?} vs {:?}",
            a.transformer_depth, b.transformer_depth
        ));
    }
    if a.cross_attention_dim != b.cross_attention_dim {
        diffs.push(format!(
            "cross_attention_dim: {} vs {}",
            a.cross_attention_dim, b.cross_attention_dim
        ));
    }
    if a.in_channels != b.in_channels {
        diffs.push(format!(
            "in_channels: {} vs {}",
            a.in_channels, b.in_channels
        ));
    }
    if a.out_channels != b.out_channels {
        diffs.push(format!(
            "out_channels: {} vs {}",
            a.out_channels, b.out_channels
        ));
    }
    if a.use_linear_projection != b.use_linear_projection {
        diffs.push(format!(
            "use_linear_projection: {} vs {}",
            a.use_linear_projection, b.use_linear_projection
        ));
    }
    if a.use_flash_attention != b.use_flash_attention {
        diffs.push(format!(
            "use_flash_attention: {} vs {}",
            a.use_flash_attention, b.use_flash_attention
        ));
    }
    if a.num_heads_per_level != b.num_heads_per_level {
        diffs.push(format!(
            "num_heads_per_level: {:?} vs {:?}",
            a.num_heads_per_level, b.num_heads_per_level
        ));
    }

    diffs
}

/// Estimate whether a configuration fits within available VRAM.
///
/// Uses [`UNetVariantConfig::estimated_vram_mb`] and compares against
/// `available_vram_mb`. Returns `true` when the estimate is within budget.
pub fn fits_in_vram(
    config: &UNetVariantConfig,
    image_size: u32,
    batch_size: usize,
    available_vram_mb: u64,
) -> bool {
    config.estimated_vram_mb(image_size, batch_size) <= available_vram_mb
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 1. ModelFamily::from_str: "sd1.5" → SD1
    #[test]
    fn test_model_family_from_str_sd15() {
        let family = ModelFamily::from_str("sd1.5").expect("should parse sd1.5");
        assert_eq!(family, ModelFamily::SD1);
    }

    // 2. ModelFamily::from_str: "sdxl" → SDXL
    #[test]
    fn test_model_family_from_str_sdxl() {
        let family = ModelFamily::from_str("sdxl").expect("should parse sdxl");
        assert_eq!(family, ModelFamily::SDXL);
    }

    // 3. ModelFamily::from_str: "unknown" → Err
    #[test]
    fn test_model_family_from_str_unknown() {
        let result = ModelFamily::from_str("unknown");
        assert!(result.is_err());
        assert!(matches!(result, Err(ModelVariantError::UnknownVariant(_))));
    }

    // 4. ModelFamily::native_resolution: SD1→512, SDXL→1024
    #[test]
    fn test_model_family_native_resolution() {
        assert_eq!(ModelFamily::SD1.native_resolution(), 512);
        assert_eq!(ModelFamily::SDXL.native_resolution(), 1024);
    }

    // 5. sd15_config: channels_at_level(0) == 320
    #[test]
    fn test_sd15_channels_at_level_0() {
        let cfg = sd15_config();
        assert_eq!(cfg.channels_at_level(0), 320);
    }

    // 6. sd15_config: channels_at_level(1) == 640
    #[test]
    fn test_sd15_channels_at_level_1() {
        let cfg = sd15_config();
        assert_eq!(cfg.channels_at_level(1), 640); // 320 * 2
    }

    // 7. sd21_config: cross_attention_dim == 1024
    #[test]
    fn test_sd21_cross_attention_dim() {
        let cfg = sd21_config();
        assert_eq!(cfg.cross_attention_dim, 1024);
    }

    // 8. sdxl_config: num_levels() == 3
    #[test]
    fn test_sdxl_num_levels() {
        let cfg = sdxl_config();
        assert_eq!(cfg.num_levels(), 3);
    }

    // 9. sdxl_config: uses_dual_encoder() == true
    #[test]
    fn test_sdxl_uses_dual_encoder() {
        let cfg = sdxl_config();
        assert!(cfg.uses_dual_encoder());
    }

    // 10. sd15_config: uses_dual_encoder() == false
    #[test]
    fn test_sd15_no_dual_encoder() {
        let cfg = sd15_config();
        assert!(!cfg.uses_dual_encoder());
    }

    // 11. sd15_config estimated_params > 0
    #[test]
    fn test_sd15_estimated_params_positive() {
        let cfg = sd15_config();
        assert!(cfg.estimated_params() > 0);
    }

    // 12. sdxl_config estimated_params > sd15_config estimated_params
    #[test]
    fn test_sdxl_params_greater_than_sd15() {
        let sd15 = sd15_config();
        let sdxl = sdxl_config();
        assert!(
            sdxl.estimated_params() > sd15.estimated_params(),
            "SDXL params ({}) should exceed SD1.5 params ({})",
            sdxl.estimated_params(),
            sd15.estimated_params()
        );
    }

    // 13. estimated_vram_mb: larger image → more VRAM
    #[test]
    fn test_estimated_vram_scales_with_image_size() {
        let cfg = sd15_config();
        let small = cfg.estimated_vram_mb(256, 1);
        let large = cfg.estimated_vram_mb(512, 1);
        assert!(
            large > small,
            "512px should need more VRAM ({large} MB) than 256px ({small} MB)"
        );
    }

    // 14. MultiViewAdapterConfig::from_base: num_views stored correctly
    #[test]
    fn test_multi_view_adapter_num_views() {
        let base = sd15_config();
        let adapter = MultiViewAdapterConfig::from_base(base, 4);
        assert_eq!(adapter.num_views, 4);
    }

    // 15. MultiViewAdapterConfig::validate: num_views=1 → Err
    #[test]
    fn test_multi_view_adapter_validate_single_view_fails() {
        let base = sd15_config();
        let adapter = MultiViewAdapterConfig::from_base(base, 1);
        let result = adapter.validate();
        assert!(result.is_err());
        assert!(matches!(result, Err(ModelVariantError::InvalidConfig(_))));
    }

    // 16. are_configs_compatible: sd15 vs sd15 → true
    #[test]
    fn test_are_configs_compatible_same() {
        let a = sd15_config();
        let b = sd15_config();
        assert!(are_configs_compatible(&a, &b));
    }

    // 17. are_configs_compatible: sd15 vs sdxl → false (different levels)
    #[test]
    fn test_are_configs_compatible_different() {
        let sd15 = sd15_config();
        let sdxl = sdxl_config();
        assert!(!are_configs_compatible(&sd15, &sdxl));
    }

    // 18. config_diff: same config → empty vec
    #[test]
    fn test_config_diff_same() {
        let a = sd15_config();
        let b = sd15_config();
        let diffs = config_diff(&a, &b);
        assert!(diffs.is_empty(), "Expected no diffs, got: {:?}", diffs);
    }

    // 19. config_diff: sd15 vs sd21 → contains "cross_attention_dim"
    #[test]
    fn test_config_diff_sd15_vs_sd21() {
        let sd15 = sd15_config();
        let sd21 = sd21_config();
        let diffs = config_diff(&sd15, &sd21);
        let has_cross_attn = diffs.iter().any(|d| d.contains("cross_attention_dim"));
        assert!(
            has_cross_attn,
            "Expected 'cross_attention_dim' in diffs, got: {:?}",
            diffs
        );
    }

    // 20. ModelVariantRegistry::new: contains "sd15", "sd21", "sdxl", "zero123"
    #[test]
    fn test_registry_contains_standard_variants() {
        let registry = ModelVariantRegistry::new();
        let names = registry.list_names();
        assert!(names.contains(&"sd15"), "Missing 'sd15'");
        assert!(names.contains(&"sd21"), "Missing 'sd21'");
        assert!(names.contains(&"sdxl"), "Missing 'sdxl'");
        assert!(names.contains(&"zero123"), "Missing 'zero123'");
    }

    // 21. registry.get("sd15") → Some(config)
    #[test]
    fn test_registry_get_sd15() {
        let registry = ModelVariantRegistry::new();
        let cfg = registry.get("sd15");
        assert!(cfg.is_some());
        let cfg = cfg.expect("sd15 config must be present");
        assert_eq!(cfg.model_channels, 320);
    }

    // 22. registry.default_for_family(SDXL): sdxl_config
    #[test]
    fn test_registry_default_for_family_sdxl() {
        let cfg = ModelVariantRegistry::default_for_family(ModelFamily::SDXL);
        assert_eq!(cfg.family, ModelFamily::SDXL);
        assert_eq!(cfg.cross_attention_dim, 2048);
        assert_eq!(cfg.num_levels(), 3);
    }

    // Additional tests for completeness

    // 23. ModelFamily::as_str round-trips correctly
    #[test]
    fn test_model_family_as_str() {
        assert_eq!(ModelFamily::SD1.as_str(), "sd1");
        assert_eq!(ModelFamily::SD2.as_str(), "sd2");
        assert_eq!(ModelFamily::SDXL.as_str(), "sdxl");
        assert_eq!(ModelFamily::Zero123.as_str(), "zero123");
        assert_eq!(ModelFamily::Custom.as_str(), "custom");
    }

    // 24. ModelFamily::vae_scale_factor is 8 for all families
    #[test]
    fn test_vae_scale_factor() {
        assert_eq!(ModelFamily::SD1.vae_scale_factor(), 8);
        assert_eq!(ModelFamily::SD2.vae_scale_factor(), 8);
        assert_eq!(ModelFamily::SDXL.vae_scale_factor(), 8);
        assert_eq!(ModelFamily::Zero123.vae_scale_factor(), 8);
    }

    // 25. zero123_config has reduced model_channels
    #[test]
    fn test_zero123_model_channels() {
        let cfg = zero123_config();
        assert_eq!(cfg.family, ModelFamily::Zero123);
        assert_eq!(cfg.model_channels, 256);
        assert_eq!(cfg.family.native_resolution(), 256);
    }

    // 26. MultiViewAdapterConfig camera_embed_dim default is 16
    #[test]
    fn test_multi_view_adapter_camera_embed_dim() {
        let base = sd15_config();
        let adapter = MultiViewAdapterConfig::from_base(base, 2);
        assert_eq!(adapter.camera_embed_dim, 16);
        assert!(adapter.use_camera_conditioning);
    }

    // 27. UNetVariantConfig::estimated_vram_mb scales with batch_size
    #[test]
    fn test_estimated_vram_scales_with_batch() {
        let cfg = sd15_config();
        let single = cfg.estimated_vram_mb(512, 1);
        let batch = cfg.estimated_vram_mb(512, 4);
        assert!(batch > single);
    }

    // 28. fits_in_vram returns false when budget is 0
    #[test]
    fn test_fits_in_vram_zero_budget() {
        let cfg = sd15_config();
        // Any config at any image size should exceed 0 MB budget
        assert!(!fits_in_vram(&cfg, 512, 1, 0));
    }

    // 29. fits_in_vram returns true for very large budget
    #[test]
    fn test_fits_in_vram_large_budget() {
        let cfg = sd15_config();
        assert!(fits_in_vram(&cfg, 512, 1, 100_000));
    }

    // 30. Registry default for Custom family falls back to sd15
    #[test]
    fn test_registry_default_custom_falls_back_to_sd15() {
        let cfg = ModelVariantRegistry::default_for_family(ModelFamily::Custom);
        assert_eq!(cfg.family, ModelFamily::SD1);
    }

    // 31. registry.register then get returns the registered config
    #[test]
    fn test_registry_register_and_get() {
        let mut registry = ModelVariantRegistry::new();
        let mut custom = sd15_config();
        custom.model_channels = 128;
        registry
            .register("tiny-sd".to_string(), custom)
            .expect("register should succeed");
        let retrieved = registry.get("tiny-sd").expect("should find tiny-sd");
        assert_eq!(retrieved.model_channels, 128);
    }

    // 32. registry.register with empty name returns Err
    #[test]
    fn test_registry_register_empty_name_fails() {
        let mut registry = ModelVariantRegistry::new();
        let result = registry.register(String::new(), sd15_config());
        assert!(result.is_err());
    }

    // 33. MultiViewAdapterConfig estimated_extra_params > 0
    #[test]
    fn test_multi_view_adapter_extra_params_positive() {
        let base = sd15_config();
        let adapter = MultiViewAdapterConfig::from_base(base, 4);
        assert!(adapter.estimated_extra_params() > 0);
    }

    // 34. channels_at_level for out-of-range level falls back gracefully
    #[test]
    fn test_channels_at_level_out_of_range() {
        let cfg = sd15_config();
        // level 99 is out of range, should fall back to model_channels * 1
        assert_eq!(cfg.channels_at_level(99), cfg.model_channels);
    }

    // 35. format_summary contains key fields
    #[test]
    fn test_format_summary_contains_key_info() {
        let cfg = sd15_config();
        let summary = cfg.format_summary();
        assert!(summary.contains("320"), "Should contain model_channels");
        assert!(
            summary.contains("768"),
            "Should contain cross_attention_dim"
        );
    }

    // 36. sd21_config / sdxl_config report per-level heads matching a uniform
    // 64-dim head, not a single flat num_heads=8 (regression for the
    // architecture-mismatch bug).
    #[test]
    fn test_sd21_sdxl_num_heads_per_level_matches_64_dim_head() {
        let sd21 = sd21_config();
        assert_eq!(sd21.num_heads_per_level, vec![5, 10, 20, 20]);
        for level in 0..sd21.num_levels() {
            assert_eq!(
                sd21.channels_at_level(level) / sd21.num_heads_per_level[level],
                64,
                "sd21 level {level} should have 64-dim heads"
            );
        }

        let sdxl = sdxl_config();
        assert_eq!(sdxl.num_heads_per_level, vec![5, 10, 20]);
        for level in 0..sdxl.num_levels() {
            assert_eq!(
                sdxl.channels_at_level(level) / sdxl.num_heads_per_level[level],
                64,
                "sdxl level {level} should have 64-dim heads"
            );
        }
    }

    // 37. sdxl_config's transformer_depth has one entry per down-level,
    // matching channel_mult (regression for the 6-vs-3 length mismatch).
    #[test]
    fn test_sdxl_transformer_depth_matches_num_levels() {
        let cfg = sdxl_config();
        assert_eq!(cfg.transformer_depth.len(), cfg.num_levels());
        assert_eq!(cfg.transformer_depth, vec![1, 2, 10]);
    }

    // 38. UNetVariantConfig::to_diffusion_config wires a variant into a real,
    // validated DiffusionConfig (regression for the "unreachable module" bug).
    #[test]
    fn test_sd15_to_diffusion_config() {
        let cfg = sd15_config();
        let diffusion_config = cfg
            .to_diffusion_config(4)
            .expect("sd15 should convert to a valid DiffusionConfig");
        assert!(diffusion_config.validate().is_ok());
        assert_eq!(diffusion_config.num_views, 4);
        assert_eq!(diffusion_config.base_channels, 320);
        assert_eq!(diffusion_config.channel_mult, vec![1, 2, 4, 4]);
        assert_eq!(diffusion_config.cross_attention_dim, 768);
        assert_eq!(diffusion_config.unet_in_channels, 4);
        assert_eq!(diffusion_config.unet_out_channels, 4);
        assert_eq!(diffusion_config.layers_per_block, 2);
        // SD1.5 uses uniform 64-dim heads: channels_at_level(i) / 64.
        assert_eq!(diffusion_config.attention_head_dim, vec![5, 10, 20, 20]);
    }

    // 39. sdxl_config converts with its per-level head counts and trimmed
    // transformer depth threaded through correctly.
    #[test]
    fn test_sdxl_to_diffusion_config() {
        let cfg = sdxl_config();
        let diffusion_config = cfg
            .to_diffusion_config(2)
            .expect("sdxl should convert to a valid DiffusionConfig");
        assert!(diffusion_config.validate().is_ok());
        assert_eq!(diffusion_config.attention_head_dim, vec![5, 10, 20]);
        assert_eq!(
            diffusion_config.transformer_layers_per_block,
            vec![1, 2, 10]
        );
        assert_eq!(diffusion_config.cross_attention_dim, 2048);
        assert_eq!(diffusion_config.image_size, 1024);
        assert_eq!(diffusion_config.latent_size, 128);
    }

    // 40. MultiViewAdapterConfig::to_diffusion_config threads num_views from
    // the adapter into the resulting DiffusionConfig.
    #[test]
    fn test_multi_view_adapter_to_diffusion_config() {
        let base = sd15_config();
        let adapter = MultiViewAdapterConfig::from_base(base, 6);
        let diffusion_config = adapter
            .to_diffusion_config()
            .expect("adapter should convert to a valid DiffusionConfig");
        assert_eq!(diffusion_config.num_views, 6);
    }

    // 41. estimated_vram_mb no longer depends on an unexplained magic
    // multiplier: a config with more transformer depth at attended levels
    // should report more VRAM than an otherwise-identical config with less
    // (regression for the 1000x fudge-factor bug).
    #[test]
    fn test_estimated_vram_reflects_transformer_depth() {
        let shallow = sd15_config();
        let mut deep = sd15_config();
        deep.transformer_depth = vec![4, 4, 4, 4];
        assert!(
            deep.estimated_vram_mb(512, 1) > shallow.estimated_vram_mb(512, 1),
            "deeper transformer blocks at attended levels should need more VRAM"
        );
    }

    // 42. estimated_vram_mb is strictly positive for every standard variant
    // (sanity check on the new per-level activation model).
    #[test]
    fn test_estimated_vram_positive_for_all_variants() {
        for cfg in [
            sd15_config(),
            sd21_config(),
            sdxl_config(),
            zero123_config(),
        ] {
            assert!(cfg.estimated_vram_mb(512, 1) > 0);
        }
    }
}
