//! Dynamic view count support for the multi-view diffusion pipeline.
//!
//! This module provides configuration, attention masking, position embeddings,
//! and memory estimation for N-view inference where N ∈ {1, 2, 3, 4, 8}.
//!
//! ## Supported View Counts
//!
//! The model supports a fixed set of view configurations:
//!
//! | Views | Cross-view attention | CFG batch |
//! |-------|---------------------|-----------|
//! | 1     | No                  | 1         |
//! | 2     | Yes                 | 2         |
//! | 3     | Yes                 | 2         |
//! | 4     | Yes                 | 2         |
//! | 8     | Yes                 | 2         |
//!
//! ## Usage
//!
//! ```rust
//! use oxigaf_diffusion::dynamic_views::{
//!     SupportedViewCount, ViewCountConfig, ViewAttentionMask,
//!     MultiViewPositionEmbedding,
//! };
//!
//! let config = ViewCountConfig::new(SupportedViewCount::Four)
//!     .with_resolution(512, 512);
//! config.validate().unwrap();
//!
//! let mask = ViewAttentionMask::ring(4);
//! let embed = MultiViewPositionEmbedding::sinusoidal(4, 64);
//! ```

use crate::DiffusionError;

// ---------------------------------------------------------------------------
// SupportedViewCount
// ---------------------------------------------------------------------------

/// Supported view count configurations for the multi-view diffusion pipeline.
///
/// Only these exact values are valid. Any other value passed to
/// [`SupportedViewCount::from_usize`] returns an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupportedViewCount {
    /// Single-view mode. Cross-view attention is disabled.
    One = 1,
    /// Two-view mode.
    Two = 2,
    /// Three-view mode.
    Three = 3,
    /// Four-view mode.
    Four = 4,
    /// Eight-view mode.
    Eight = 8,
}

impl SupportedViewCount {
    /// Parse a `usize` into a [`SupportedViewCount`].
    ///
    /// Returns [`DiffusionError::InvalidViewCount`] for any value not in
    /// {1, 2, 3, 4, 8}.
    pub fn from_usize(n: usize) -> Result<Self, DiffusionError> {
        match n {
            1 => Ok(Self::One),
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            4 => Ok(Self::Four),
            8 => Ok(Self::Eight),
            _ => Err(DiffusionError::InvalidViewCount {
                expected: 0, // sentinel: not a fixed expected count
                got: n,
            }),
        }
    }

    /// Return the numeric value of the view count.
    pub fn as_usize(&self) -> usize {
        *self as usize
    }

    /// Return a static slice of all supported view count variants in ascending
    /// order: [One, Two, Three, Four, Eight].
    pub fn all() -> &'static [SupportedViewCount] {
        &[
            SupportedViewCount::One,
            SupportedViewCount::Two,
            SupportedViewCount::Three,
            SupportedViewCount::Four,
            SupportedViewCount::Eight,
        ]
    }

    /// Returns `true` when cross-view attention is supported (i.e., num_views > 1).
    ///
    /// Single-view inference uses only self-attention; no cross-view keys/values
    /// are produced.
    pub fn supports_cross_attention(&self) -> bool {
        !matches!(self, Self::One)
    }
}

// ---------------------------------------------------------------------------
// ViewCountConfig
// ---------------------------------------------------------------------------

/// Configuration for a specific view count, including image resolution and
/// channel count.
///
/// Build with [`ViewCountConfig::new`] and optionally customise resolution via
/// [`ViewCountConfig::with_resolution`].
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::dynamic_views::{SupportedViewCount, ViewCountConfig};
///
/// let cfg = ViewCountConfig::new(SupportedViewCount::Four)
///     .with_resolution(512, 512);
/// assert_eq!(cfg.latent_height(), 64);
/// assert_eq!(cfg.latent_width(), 64);
/// ```
pub struct ViewCountConfig {
    /// Number of views.
    pub num_views: SupportedViewCount,
    /// Image height in pixels (default: 256).
    pub image_height: usize,
    /// Image width in pixels (default: 256).
    pub image_width: usize,
    /// Number of latent channels (default: 4).
    pub latent_channels: usize,
    /// Whether cross-view attention is enabled.
    ///
    /// Automatically set to `false` when `num_views` is
    /// [`SupportedViewCount::One`].
    pub use_cross_view_attention: bool,
}

impl ViewCountConfig {
    /// Create a new [`ViewCountConfig`] with default image resolution (256×256)
    /// and four latent channels.
    pub fn new(num_views: SupportedViewCount) -> Self {
        let use_cross_view_attention = num_views.supports_cross_attention();
        Self {
            num_views,
            image_height: 256,
            image_width: 256,
            latent_channels: 4,
            use_cross_view_attention,
        }
    }

    /// Override the image resolution. Returns `self` for builder chaining.
    pub fn with_resolution(mut self, h: usize, w: usize) -> Self {
        self.image_height = h;
        self.image_width = w;
        self
    }

    /// Validate this configuration.
    ///
    /// Checks that image dimensions and channel count are all > 0.
    ///
    /// # Errors
    ///
    /// Returns [`DiffusionError::InvalidConfig`] if any dimension is zero.
    pub fn validate(&self) -> Result<(), DiffusionError> {
        if self.image_height == 0 {
            return Err(DiffusionError::InvalidConfig(
                "image_height must be > 0".to_string(),
            ));
        }
        if self.image_width == 0 {
            return Err(DiffusionError::InvalidConfig(
                "image_width must be > 0".to_string(),
            ));
        }
        if self.latent_channels == 0 {
            return Err(DiffusionError::InvalidConfig(
                "latent_channels must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Height of the latent spatial grid (image_height / 8).
    pub fn latent_height(&self) -> usize {
        self.image_height / 8
    }

    /// Width of the latent spatial grid (image_width / 8).
    pub fn latent_width(&self) -> usize {
        self.image_width / 8
    }

    /// Total number of elements across all view latent tensors.
    ///
    /// Equals `num_views × latent_channels × latent_height × latent_width`.
    pub fn total_latent_elements(&self) -> usize {
        self.num_views.as_usize()
            * self.latent_channels
            * self.latent_height()
            * self.latent_width()
    }

    /// Effective batch size when classifier-free guidance (CFG) is active.
    ///
    /// Returns 2 (conditional + unconditional) when cross-view attention is
    /// enabled, or 1 for single-view where CFG is not required.
    pub fn batch_size_for_cfg(&self) -> usize {
        if self.use_cross_view_attention {
            2
        } else {
            1
        }
    }
}

// ---------------------------------------------------------------------------
// ViewAttentionMask
// ---------------------------------------------------------------------------

/// View attention mask for cross-view attention.
///
/// In cross-view attention the query comes from one view and the keys/values
/// come from all views. The mask controls which views a given view is allowed
/// to attend to.
///
/// `mask[i][j] == true` means view *i* can attend to view *j*.
/// The shape is `[num_views, num_views]`.
pub struct ViewAttentionMask {
    /// Number of views in the mask.
    pub num_views: usize,
    /// Boolean attention mask of shape `[num_views, num_views]`.
    ///
    /// `mask[i][j] = true` iff view *i* can attend to view *j*.
    pub mask: Vec<Vec<bool>>,
}

impl ViewAttentionMask {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Create a fully-connected attention mask (all views attend to all views).
    ///
    /// Every entry is `true`, including the diagonal.
    pub fn full(num_views: usize) -> Self {
        let mask = vec![vec![true; num_views]; num_views];
        Self { num_views, mask }
    }

    /// Create an identity (self-attention only) mask.
    ///
    /// Only the diagonal entries are `true`; each view attends only to itself.
    pub fn identity(num_views: usize) -> Self {
        let mut mask = vec![vec![false; num_views]; num_views];
        for (i, row) in mask.iter_mut().enumerate() {
            row[i] = true;
        }
        Self { num_views, mask }
    }

    /// Create a ring attention mask.
    ///
    /// Each view attends to itself and its immediate left/right neighbours in a
    /// circular (toroidal) arrangement. For N=1 this degenerates to identity.
    pub fn ring(num_views: usize) -> Self {
        let mut mask = vec![vec![false; num_views]; num_views];
        for (i, row) in mask.iter_mut().enumerate() {
            // Self
            row[i] = true;
            if num_views > 1 {
                // Left neighbour (circular)
                let left = (i + num_views - 1) % num_views;
                row[left] = true;
                // Right neighbour (circular)
                let right = (i + 1) % num_views;
                row[right] = true;
            }
        }
        Self { num_views, mask }
    }

    /// Build a mask from an explicit undirected edge list.
    ///
    /// Each `(a, b)` edge in `edges` enables both `mask[a][b]` and
    /// `mask[b][a]` (symmetric). The diagonal (self-attention) is always `true`.
    ///
    /// # Errors
    ///
    /// Returns [`DiffusionError::InvalidConfig`] if any edge index is ≥
    /// `num_views`.
    pub fn from_adjacency(
        num_views: usize,
        edges: &[(usize, usize)],
    ) -> Result<Self, DiffusionError> {
        // Validate all indices first
        for &(a, b) in edges {
            if a >= num_views || b >= num_views {
                return Err(DiffusionError::InvalidConfig(format!(
                    "Edge ({a}, {b}) contains an index >= num_views ({num_views})"
                )));
            }
        }
        // Start with the identity (self-attention always enabled)
        let mut mask = vec![vec![false; num_views]; num_views];
        for (i, row) in mask.iter_mut().enumerate() {
            row[i] = true;
        }
        // Add undirected edges symmetrically
        for &(a, b) in edges {
            mask[a][b] = true;
            mask[b][a] = true;
        }
        Ok(Self { num_views, mask })
    }

    // ------------------------------------------------------------------
    // Conversion helpers
    // ------------------------------------------------------------------

    /// Flatten the mask to a `Vec<f32>` of 0.0 / 1.0 values.
    ///
    /// Row-major order: index `i * num_views + j` corresponds to
    /// `mask[i][j]`. Can be multiplied with attention scores to zero out
    /// masked positions before softmax.
    pub fn to_flat_mask(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.num_views * self.num_views);
        for row in &self.mask {
            for &allowed in row {
                out.push(if allowed { 1.0 } else { 0.0 });
            }
        }
        out
    }

    /// Flatten the mask to an additive bias suitable for attention logits.
    ///
    /// - `true`  → `0.0` (unmasked, no effect on logits)
    /// - `false` → `-1e9` (effectively −∞, zeroed out after softmax)
    ///
    /// Row-major order: index `i * num_views + j` corresponds to
    /// `mask[i][j]`.
    pub fn to_additive_bias(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.num_views * self.num_views);
        for row in &self.mask {
            for &allowed in row {
                out.push(if allowed { 0.0 } else { -1e9 });
            }
        }
        out
    }

    /// Count the number of active (true) edges, **excluding** the diagonal.
    ///
    /// This counts only inter-view attention connections.
    pub fn num_active_edges(&self) -> usize {
        let mut count = 0;
        for i in 0..self.num_views {
            for j in 0..self.num_views {
                if i != j && self.mask[i][j] {
                    count += 1;
                }
            }
        }
        count
    }

    /// Returns `true` if every entry in the mask is `true` (fully connected).
    pub fn is_fully_connected(&self) -> bool {
        for row in &self.mask {
            for &allowed in row {
                if !allowed {
                    return false;
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// MultiViewPositionEmbedding
// ---------------------------------------------------------------------------

/// View-specific position embeddings for cross-view attention.
///
/// Each view gets a distinct embedding vector of length `embed_dim`. These
/// embeddings can be added to view features before cross-view attention to
/// inform the model which view is which.
///
/// Two initialisation strategies are available:
/// - [`MultiViewPositionEmbedding::sinusoidal`]: deterministic, no training
///   required.
/// - [`MultiViewPositionEmbedding::learned_init`]: random initialisation for
///   embeddings that will be fine-tuned.
pub struct MultiViewPositionEmbedding {
    /// Number of views.
    pub num_views: usize,
    /// Dimensionality of each embedding vector.
    pub embed_dim: usize,
    /// Per-view embedding vectors.
    ///
    /// `embeddings[view_index]` is a `Vec<f32>` of length `embed_dim`.
    pub embeddings: Vec<Vec<f32>>,
}

/// Simple xorshift64 PRNG.
///
/// Produces a `f64` value in `[0, 1)`. The `state` must be non-zero; if it is
/// zero on entry it is advanced to 1 before the shift sequence.
fn xorshift64(state: &mut u64) -> f64 {
    if *state == 0 {
        *state = 1;
    }
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

impl MultiViewPositionEmbedding {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Create sinusoidal position embeddings.
    ///
    /// Uses the standard Transformer positional encoding with the view index
    /// as the "position":
    ///
    /// - `embedding[2k]   = sin(i / 10000^(2k/D))`
    /// - `embedding[2k+1] = cos(i / 10000^(2k/D))`
    ///
    /// where `i` is the view index and `D` is `embed_dim`.
    pub fn sinusoidal(num_views: usize, embed_dim: usize) -> Self {
        let embeddings: Vec<Vec<f32>> = (0..num_views)
            .map(|i| {
                let mut vec = vec![0.0_f32; embed_dim];
                for k in 0..(embed_dim / 2) {
                    let exponent = (2 * k) as f64 / embed_dim as f64;
                    let denom = 10000_f64.powf(exponent);
                    let angle = i as f64 / denom;
                    vec[2 * k] = angle.sin() as f32;
                    if 2 * k + 1 < embed_dim {
                        vec[2 * k + 1] = angle.cos() as f32;
                    }
                }
                // Handle odd embed_dim: the last element gets only the sin term
                if embed_dim % 2 == 1 {
                    let k = embed_dim / 2;
                    let exponent = (2 * k) as f64 / embed_dim as f64;
                    let denom = 10000_f64.powf(exponent);
                    let angle = i as f64 / denom;
                    vec[2 * k] = angle.sin() as f32;
                }
                vec
            })
            .collect();

        Self {
            num_views,
            embed_dim,
            embeddings,
        }
    }

    /// Create learned-style embeddings initialised with small Gaussian noise.
    ///
    /// Uses a deterministic xorshift64 PRNG seeded with `seed`. Each value is
    /// drawn uniformly from `[-0.02, 0.02]`:
    ///
    /// ```text
    /// value = 0.02 * (2.0 * rand - 1.0)
    /// ```
    ///
    /// These should be registered as model parameters and fine-tuned.
    pub fn learned_init(num_views: usize, embed_dim: usize, seed: u64) -> Self {
        let scale = 0.02_f64;
        let mut state = seed;
        let embeddings: Vec<Vec<f32>> = (0..num_views)
            .map(|_| {
                (0..embed_dim)
                    .map(|_| {
                        let rand = xorshift64(&mut state);
                        (scale * (2.0 * rand - 1.0)) as f32
                    })
                    .collect()
            })
            .collect();

        Self {
            num_views,
            embed_dim,
            embeddings,
        }
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Return the embedding slice for the given view index, or `None` if
    /// `view_index >= num_views`.
    pub fn for_view(&self, view_index: usize) -> Option<&[f32]> {
        self.embeddings.get(view_index).map(|v| v.as_slice())
    }

    /// Add this view's embedding to a feature slice in-place.
    ///
    /// Only the first `min(embed_dim, feature_dim)` elements of `features` are
    /// modified. If `view_index >= num_views` the function is a no-op.
    ///
    /// # Arguments
    ///
    /// * `features`    – mutable slice of at least `feature_dim` elements.
    /// * `view_index`  – which view's embedding to add.
    /// * `feature_dim` – length of the logical feature vector inside `features`.
    pub fn add_to_features(&self, features: &mut [f32], view_index: usize, feature_dim: usize) {
        if let Some(embed) = self.for_view(view_index) {
            let copy_len = self.embed_dim.min(feature_dim).min(features.len());
            for i in 0..copy_len {
                features[i] += embed[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ViewCountMemoryEstimate
// ---------------------------------------------------------------------------

/// Memory usage estimate for a given view configuration.
///
/// All byte counts are approximate; they assume f32 (4 bytes per element) for
/// both the latent tensors and the QKV attention buffers.
pub struct ViewCountMemoryEstimate {
    /// Number of views in this estimate.
    pub num_views: usize,
    /// Bytes required for the latent tensors (CFG doubled).
    pub latent_bytes: usize,
    /// Bytes required for the Q, K, V attention buffers.
    pub attention_bytes: usize,
    /// `latent_bytes + attention_bytes`.
    pub total_bytes: usize,
    /// `total_bytes` expressed in mebibytes.
    pub total_mb: f32,
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Estimate memory requirements for a given view configuration.
///
/// Formulae:
/// - `latent_bytes  = 2 × num_views × channels × (H/8) × (W/8) × 4`
/// - `attention_bytes = num_views × seq_len × num_heads × head_dim × 3 × 4`
///
/// where the leading `2` in `latent_bytes` accounts for the classifier-free
/// guidance (conditional + unconditional) batch.
///
/// # Arguments
///
/// * `config`    – view configuration (resolution and channel count).
/// * `seq_len`   – spatial sequence length for attention (tokens per view).
/// * `num_heads` – number of attention heads.
/// * `head_dim`  – per-head dimension.
pub fn estimate_memory(
    config: &ViewCountConfig,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> ViewCountMemoryEstimate {
    let n = config.num_views.as_usize();
    let latent_bytes =
        2 * n * config.latent_channels * config.latent_height() * config.latent_width() * 4;
    let attention_bytes = n * seq_len * num_heads * head_dim * 3 * 4;
    let total_bytes = latent_bytes + attention_bytes;
    let total_mb = total_bytes as f32 / (1024.0 * 1024.0);

    ViewCountMemoryEstimate {
        num_views: n,
        latent_bytes,
        attention_bytes,
        total_bytes,
        total_mb,
    }
}

/// Recommend the highest view count that fits within `available_memory_bytes`.
///
/// Uses conservative assumptions:
/// - `num_heads = 8`
/// - `head_dim  = 64`
/// - `seq_len   = latent_height × latent_width`
///
/// Iterates from Eight down to One and returns the first view count whose
/// memory estimate fits within the budget. Always returns
/// [`SupportedViewCount::One`] as a safe fallback.
pub fn recommend_view_count(
    available_memory_bytes: usize,
    image_height: usize,
    image_width: usize,
) -> SupportedViewCount {
    let num_heads = 8;
    let head_dim = 64;

    for &vc in SupportedViewCount::all().iter().rev() {
        let config = ViewCountConfig::new(vc).with_resolution(image_height, image_width);
        let seq_len = config.latent_height() * config.latent_width();
        let estimate = estimate_memory(&config, seq_len, num_heads, head_dim);
        if estimate.total_bytes <= available_memory_bytes {
            return vc;
        }
    }
    SupportedViewCount::One
}

/// Validate that a [`ViewCountConfig`] is compatible with the model.
///
/// Checks that:
/// 1. Image dimensions are non-zero (via [`ViewCountConfig::validate`]).
/// 2. `use_cross_view_attention` is `false` when `num_views` is
///    [`SupportedViewCount::One`] (cross-view attention has nothing to
///    attend across in single-view mode).
///
/// # Errors
///
/// Returns [`DiffusionError::InvalidConfig`] if any constraint is violated.
pub fn validate_view_config(config: &ViewCountConfig) -> Result<(), DiffusionError> {
    // Dimension checks
    config.validate()?;

    // The view count is validated by construction, but double-check that the
    // `use_cross_view_attention` flag is consistent with the view count.
    if !config.num_views.supports_cross_attention() && config.use_cross_view_attention {
        return Err(DiffusionError::InvalidConfig(
            "use_cross_view_attention must be false for single-view mode".to_string(),
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // SupportedViewCount
    // ------------------------------------------------------------------

    #[test]
    fn test_supported_view_count_from_usize_valid() {
        assert_eq!(
            SupportedViewCount::from_usize(1).unwrap(),
            SupportedViewCount::One
        );
        assert_eq!(
            SupportedViewCount::from_usize(2).unwrap(),
            SupportedViewCount::Two
        );
        assert_eq!(
            SupportedViewCount::from_usize(3).unwrap(),
            SupportedViewCount::Three
        );
        assert_eq!(
            SupportedViewCount::from_usize(4).unwrap(),
            SupportedViewCount::Four
        );
        assert_eq!(
            SupportedViewCount::from_usize(8).unwrap(),
            SupportedViewCount::Eight
        );
    }

    #[test]
    fn test_supported_view_count_from_usize_invalid() {
        assert!(SupportedViewCount::from_usize(0).is_err());
        assert!(SupportedViewCount::from_usize(5).is_err());
        assert!(SupportedViewCount::from_usize(7).is_err());
        assert!(SupportedViewCount::from_usize(16).is_err());
        assert!(SupportedViewCount::from_usize(100).is_err());
    }

    // ------------------------------------------------------------------
    // ViewCountConfig
    // ------------------------------------------------------------------

    #[test]
    fn test_view_count_config_defaults() {
        let cfg = ViewCountConfig::new(SupportedViewCount::Four);
        assert_eq!(cfg.num_views, SupportedViewCount::Four);
        assert_eq!(cfg.image_height, 256);
        assert_eq!(cfg.image_width, 256);
        assert_eq!(cfg.latent_channels, 4);
        assert!(cfg.use_cross_view_attention);

        // Single-view disables cross attention
        let cfg_one = ViewCountConfig::new(SupportedViewCount::One);
        assert!(!cfg_one.use_cross_view_attention);
    }

    #[test]
    fn test_view_count_config_validate() {
        let valid = ViewCountConfig::new(SupportedViewCount::Two);
        assert!(valid.validate().is_ok());

        let mut bad_h = ViewCountConfig::new(SupportedViewCount::Two);
        bad_h.image_height = 0;
        assert!(bad_h.validate().is_err());

        let mut bad_w = ViewCountConfig::new(SupportedViewCount::Two);
        bad_w.image_width = 0;
        assert!(bad_w.validate().is_err());

        let mut bad_c = ViewCountConfig::new(SupportedViewCount::Two);
        bad_c.latent_channels = 0;
        assert!(bad_c.validate().is_err());
    }

    #[test]
    fn test_latent_dimensions() {
        let cfg = ViewCountConfig::new(SupportedViewCount::Four).with_resolution(512, 256);
        assert_eq!(cfg.latent_height(), 64);
        assert_eq!(cfg.latent_width(), 32);
        // total_latent_elements = 4 * 4 * 64 * 32 = 32768
        assert_eq!(cfg.total_latent_elements(), 4 * 4 * 64 * 32);
    }

    #[test]
    fn test_batch_size_for_cfg() {
        let one = ViewCountConfig::new(SupportedViewCount::One);
        assert_eq!(one.batch_size_for_cfg(), 1);

        let four = ViewCountConfig::new(SupportedViewCount::Four);
        assert_eq!(four.batch_size_for_cfg(), 2);

        let eight = ViewCountConfig::new(SupportedViewCount::Eight);
        assert_eq!(eight.batch_size_for_cfg(), 2);
    }

    // ------------------------------------------------------------------
    // ViewAttentionMask
    // ------------------------------------------------------------------

    #[test]
    fn test_full_attention_mask() {
        let mask = ViewAttentionMask::full(3);
        assert_eq!(mask.num_views, 3);
        assert!(mask.is_fully_connected());
        for i in 0..3 {
            for j in 0..3 {
                assert!(mask.mask[i][j], "full mask should have [{i}][{j}] = true");
            }
        }
    }

    #[test]
    fn test_identity_attention_mask() {
        let mask = ViewAttentionMask::identity(4);
        assert!(!mask.is_fully_connected());
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(mask.mask[i][j], i == j, "identity mask [{i}][{j}] mismatch");
            }
        }
        assert_eq!(mask.num_active_edges(), 0);
    }

    #[test]
    fn test_ring_attention_mask_4_views() {
        let mask = ViewAttentionMask::ring(4);
        // Each view should see itself + left + right
        // View 0: [0,1,3] (right=1, left=3)
        assert!(mask.mask[0][0]);
        assert!(mask.mask[0][1]);
        assert!(!mask.mask[0][2]);
        assert!(mask.mask[0][3]);

        // View 1: [0,1,2]
        assert!(mask.mask[1][0]);
        assert!(mask.mask[1][1]);
        assert!(mask.mask[1][2]);
        assert!(!mask.mask[1][3]);

        // Ring is not fully connected for N >= 4
        assert!(!mask.is_fully_connected());

        // Active edges (excluding diagonal): 4 views × 2 directions = 8
        assert_eq!(mask.num_active_edges(), 8);
    }

    #[test]
    fn test_ring_attention_mask_single_view() {
        // N=1: ring degenerates to identity
        let mask = ViewAttentionMask::ring(1);
        assert!(mask.mask[0][0]);
        assert_eq!(mask.num_active_edges(), 0);
    }

    #[test]
    fn test_ring_attention_mask_two_views() {
        // N=2: both views should see each other (ring wraps to the other view)
        let mask = ViewAttentionMask::ring(2);
        assert!(mask.mask[0][0]);
        assert!(mask.mask[0][1]);
        assert!(mask.mask[1][0]);
        assert!(mask.mask[1][1]);
        assert!(mask.is_fully_connected());
    }

    #[test]
    fn test_adjacency_mask_from_edges() {
        // 4 views, connect 0-1 and 2-3
        let mask =
            ViewAttentionMask::from_adjacency(4, &[(0, 1), (2, 3)]).expect("valid adjacency mask");

        // Diagonal always true
        for i in 0..4 {
            assert!(mask.mask[i][i]);
        }
        // Edges (symmetric)
        assert!(mask.mask[0][1]);
        assert!(mask.mask[1][0]);
        assert!(mask.mask[2][3]);
        assert!(mask.mask[3][2]);
        // Non-edges
        assert!(!mask.mask[0][2]);
        assert!(!mask.mask[0][3]);
        assert!(!mask.mask[1][2]);
        assert!(!mask.mask[1][3]);
    }

    #[test]
    fn test_adjacency_mask_invalid_index() {
        let result = ViewAttentionMask::from_adjacency(3, &[(0, 3)]);
        assert!(result.is_err(), "should fail: index 3 >= num_views=3");

        let result2 = ViewAttentionMask::from_adjacency(3, &[(5, 0)]);
        assert!(result2.is_err(), "should fail: index 5 >= num_views=3");
    }

    #[test]
    fn test_to_additive_bias() {
        // 2-view identity mask: off-diagonal should be -1e9
        let mask = ViewAttentionMask::identity(2);
        let bias = mask.to_additive_bias();
        assert_eq!(bias.len(), 4);
        // [true, false, false, true] → [0.0, -1e9, -1e9, 0.0]
        assert!((bias[0] - 0.0_f32).abs() < 1e-6);
        assert!((bias[1] - (-1e9_f32)).abs() < 1.0); // large value, coarse check
        assert!((bias[2] - (-1e9_f32)).abs() < 1.0);
        assert!((bias[3] - 0.0_f32).abs() < 1e-6);

        let flat = mask.to_flat_mask();
        assert_eq!(flat, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_is_fully_connected() {
        assert!(ViewAttentionMask::full(4).is_fully_connected());
        assert!(!ViewAttentionMask::identity(4).is_fully_connected());
        assert!(!ViewAttentionMask::ring(4).is_fully_connected());
        // 2-view ring is fully connected
        assert!(ViewAttentionMask::ring(2).is_fully_connected());
    }

    // ------------------------------------------------------------------
    // MultiViewPositionEmbedding
    // ------------------------------------------------------------------

    #[test]
    fn test_sinusoidal_embedding() {
        let embed = MultiViewPositionEmbedding::sinusoidal(4, 8);
        assert_eq!(embed.num_views, 4);
        assert_eq!(embed.embed_dim, 8);
        assert_eq!(embed.embeddings.len(), 4);

        // View 0: sin(0 / ...) = 0 for even indices
        for k in 0..4 {
            assert!(
                (embed.embeddings[0][2 * k] - 0.0_f32).abs() < 1e-6,
                "view 0 even index should be sin(0)=0"
            );
        }

        // Each embedding vector has correct length
        for v in &embed.embeddings {
            assert_eq!(v.len(), 8);
        }

        // View 1, first element: sin(1 / 10000^0) = sin(1.0)
        let expected = (1.0_f64).sin() as f32;
        assert!(
            (embed.embeddings[1][0] - expected).abs() < 1e-6,
            "view 1 embed[0] = sin(1.0) = {expected}"
        );
        // View 1, second element: cos(1 / 10000^0) = cos(1.0)
        let expected_cos = (1.0_f64).cos() as f32;
        assert!(
            (embed.embeddings[1][1] - expected_cos).abs() < 1e-6,
            "view 1 embed[1] = cos(1.0) = {expected_cos}"
        );
    }

    #[test]
    fn test_learned_init_embedding() {
        let embed = MultiViewPositionEmbedding::learned_init(4, 16, 42);
        assert_eq!(embed.num_views, 4);
        assert_eq!(embed.embed_dim, 16);

        // All values should be in [-0.02, 0.02]
        for (vi, view) in embed.embeddings.iter().enumerate() {
            assert_eq!(view.len(), 16);
            for (di, &val) in view.iter().enumerate() {
                assert!(
                    (-0.02..=0.02).contains(&val),
                    "embedding[{vi}][{di}] = {val} out of range [-0.02, 0.02]"
                );
            }
        }

        // Different seed should give different results
        let embed2 = MultiViewPositionEmbedding::learned_init(4, 16, 123);
        let same = embed
            .embeddings
            .iter()
            .zip(embed2.embeddings.iter())
            .all(|(a, b)| a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-9));
        assert!(!same, "different seeds should yield different embeddings");

        // Seed=0 must not produce all zeros (xorshift guard)
        let embed_zero = MultiViewPositionEmbedding::learned_init(2, 4, 0);
        let all_zero = embed_zero
            .embeddings
            .iter()
            .all(|v| v.iter().all(|&x| x == 0.0));
        assert!(!all_zero, "seed=0 guard should prevent all-zero embeddings");
    }

    #[test]
    fn test_for_view_out_of_range() {
        let embed = MultiViewPositionEmbedding::sinusoidal(4, 8);
        assert!(embed.for_view(0).is_some());
        assert!(embed.for_view(3).is_some());
        assert!(embed.for_view(4).is_none());
        assert!(embed.for_view(100).is_none());
    }

    #[test]
    fn test_add_to_features() {
        let embed = MultiViewPositionEmbedding::sinusoidal(2, 4);

        // View 1 embedding
        let view1 = embed.for_view(1).expect("view 1 should exist").to_vec();

        let mut features = vec![0.0_f32; 4];
        embed.add_to_features(&mut features, 1, 4);

        for i in 0..4 {
            assert!(
                (features[i] - view1[i]).abs() < 1e-7,
                "features[{i}] should equal embedding: {} vs {}",
                features[i],
                view1[i]
            );
        }

        // Out-of-range view index: no-op
        let mut features2 = vec![1.0_f32; 4];
        embed.add_to_features(&mut features2, 99, 4);
        assert_eq!(features2, vec![1.0_f32; 4]);
    }

    // ------------------------------------------------------------------
    // Memory estimation
    // ------------------------------------------------------------------

    #[test]
    fn test_memory_estimate() {
        let config = ViewCountConfig::new(SupportedViewCount::Four).with_resolution(256, 256);
        // latent_h = 32, latent_w = 32
        // latent_bytes = 2 * 4 * 4 * 32 * 32 * 4 = 131072
        let seq_len = 32 * 32;
        let estimate = estimate_memory(&config, seq_len, 8, 64);

        assert_eq!(estimate.num_views, 4);

        let expected_latent = 2 * 4 * 4 * 32 * 32 * 4_usize;
        assert_eq!(estimate.latent_bytes, expected_latent);

        // attention_bytes = 4 * 1024 * 8 * 64 * 3 * 4 = 25_165_824
        let expected_attn = 4 * seq_len * 8 * 64 * 3 * 4_usize;
        assert_eq!(estimate.attention_bytes, expected_attn);

        assert_eq!(estimate.total_bytes, expected_latent + expected_attn);
        assert!(estimate.total_mb > 0.0);
    }

    #[test]
    fn test_recommend_view_count() {
        // Tiny budget → should fall back to One
        let tiny = recommend_view_count(1024, 256, 256);
        assert_eq!(tiny, SupportedViewCount::One);

        // Very large budget → should allow Eight
        let huge = recommend_view_count(usize::MAX, 256, 256);
        assert_eq!(huge, SupportedViewCount::Eight);

        // The recommended view count should always be valid
        let mid = recommend_view_count(64 * 1024 * 1024, 256, 256);
        assert!(SupportedViewCount::from_usize(mid.as_usize()).is_ok());
    }

    #[test]
    fn test_recommend_view_count_uses_num_heads_8_not_128() {
        // Regression test: the rustdoc promises num_heads=8. Pick a budget
        // that fits Eight views under that assumption but not under the
        // old (buggy) num_heads=128 assumption, and confirm the function
        // now recommends Eight for it.
        let image_height = 512;
        let image_width = 512;
        let config = ViewCountConfig::new(SupportedViewCount::Eight)
            .with_resolution(image_height, image_width);
        let seq_len = config.latent_height() * config.latent_width();

        let estimate_correct = estimate_memory(&config, seq_len, 8, 64);
        let estimate_buggy = estimate_memory(&config, seq_len, 128, 64);
        // Sanity check: the two estimates must actually differ, or this
        // test would not be able to distinguish the fixed behaviour from
        // the old one.
        assert!(estimate_buggy.total_bytes > estimate_correct.total_bytes);

        let budget = (estimate_correct.total_bytes + estimate_buggy.total_bytes) / 2;
        assert!(budget >= estimate_correct.total_bytes);
        assert!(budget < estimate_buggy.total_bytes);

        let recommended = recommend_view_count(budget, image_height, image_width);
        assert_eq!(
            recommended,
            SupportedViewCount::Eight,
            "recommend_view_count should use num_heads=8 per its documented \
             assumption and recommend Eight for this budget"
        );
    }

    #[test]
    fn test_validate_view_config_ok() {
        let config = ViewCountConfig::new(SupportedViewCount::Four);
        assert!(validate_view_config(&config).is_ok());
    }

    #[test]
    fn test_validate_view_config_bad_cross_attention_flag() {
        // Manually set inconsistent flag
        let mut config = ViewCountConfig::new(SupportedViewCount::One);
        config.use_cross_view_attention = true; // inconsistent
        assert!(validate_view_config(&config).is_err());
    }
}
