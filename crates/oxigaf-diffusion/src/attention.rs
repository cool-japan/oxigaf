//! Attention-based building blocks for multi-view diffusion.
//!
//! Implements the multi-view transformer block that replaces the standard
//! SD 2.1 `BasicTransformerBlock` with additional layers:
//!
//! ## Multi-View Transformer Architecture
//!
//! Each `MultiViewTransformerBlock` contains five sequential operations:
//!
//! 1. **Self-Attention** (`attn1`): Attention within each view's spatial tokens
//! 2. **Cross-View Attention** (`attn_cv`): Attention across all N views at each
//!    spatial position, enabling 3D consistency
//! 3. **Text Cross-Attention** (`attn2`): Conditions on text embeddings
//!    (always zero in GAF since we don't use text prompts)
//! 4. **IP-Adapter Cross-Attention** (`attn_ip`): Conditions on CLIP image
//!    embeddings from the reference photo, providing identity preservation
//! 5. **Feed-Forward** (`ff`): GeGLU-activated MLP for feature processing
//!
//! ## IP-Adapter Mechanism
//!
//! The IP-Adapter layer enables pixel-level identity conditioning:
//!
//! - **Input**: CLIP ViT-H/14 encodes reference image → 257×1280 embeddings
//! - **Projection**: Linear layer projects to cross_attention_dim (1024)
//! - **Attention**: Each spatial position (h×w) attends to 257 image tokens
//! - **Output**: Spatially-varying conditioning based on reference features
//!
//! When `ip_tokens=None` (CFG unconditional pass), the IP-Adapter layer is
//! skipped entirely via early return, producing unconditional predictions.
//!
//! ## Flash Attention Support
//!
//! When the `flash_attention` feature is enabled, attention modules can use
//! memory-efficient flash attention with O(N) memory complexity instead of
//! O(N²). This is controlled via the `use_flash_attention` field in
//! `DiffusionConfig`.
//!
//! Flash attention provides 2-4× memory reduction for large images without
//! sacrificing accuracy (< 1e-3 numerical difference from standard attention).

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn as nn;
use candle_nn::Module;

use crate::attention_masking::AttentionMask;

#[cfg(feature = "flash_attention")]
use crate::flash_attention::{FlashAttention, FlashAttentionConfig};

// ---------------------------------------------------------------------------
// Attention-mask bridge: attention_masking::AttentionMask -> Tensor
// ---------------------------------------------------------------------------

/// Converts an [`AttentionMask`] into an additive `f32` bias tensor of shape
/// `(1, 1, seq_len, seq_len)`, ready to pass as `attn_mask` to
/// [`CrossAttention::forward_masked`] (broadcasts against scores of shape
/// `(batch, heads, seq_len, seq_len)`).
///
/// See [`CrossAttention::forward_masked`] and
/// [`MultiViewTransformerBlock::forward_with_mask`] for the shape each
/// attention site actually expects — in particular, a mask built by
/// [`crate::attention_masking::build_layer_mask`] is sized for
/// `num_views * tokens_per_view` and does **not** fit the cross-view
/// attention site (which needs a `num_views`-sized mask, i.e. one built with
/// `tokens_per_view = 1`).
pub fn mask_to_bias_tensor(mask: &AttentionMask, device: &Device) -> Result<Tensor> {
    let bias = mask.to_bias();
    Tensor::from_vec(bias, (1, 1, mask.seq_len, mask.seq_len), device)
}

// ---------------------------------------------------------------------------
// GeGLU activation
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct GeGlu {
    proj: nn::Linear,
}

impl GeGlu {
    fn new(vs: nn::VarBuilder, dim_in: usize, dim_out: usize) -> Result<Self> {
        let proj = nn::linear(dim_in, dim_out * 2, vs.pp("proj"))?;
        Ok(Self { proj })
    }
}

impl Module for GeGlu {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden_and_gate = self.proj.forward(xs)?.chunk(2, D::Minus1)?;
        &hidden_and_gate[0] * hidden_and_gate[1].gelu()?
    }
}

// ---------------------------------------------------------------------------
// Feed-forward network
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct FeedForward {
    project_in: GeGlu,
    linear_out: nn::Linear,
}

impl FeedForward {
    fn new(vs: nn::VarBuilder, dim: usize, mult: usize) -> Result<Self> {
        let inner_dim = dim * mult;
        let vs = vs.pp("net");
        let project_in = GeGlu::new(vs.pp("0"), dim, inner_dim)?;
        let linear_out = nn::linear(inner_dim, dim, vs.pp("2"))?;
        Ok(Self {
            project_in,
            linear_out,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.project_in.forward(xs)?;
        self.linear_out.forward(&xs)
    }
}

// ---------------------------------------------------------------------------
// Cross-attention (used for self-attn, text cross-attn, cross-view, IP)
// ---------------------------------------------------------------------------

/// Cross-attention module with optional flash attention support.
///
/// When flash attention is enabled (via feature flag and configuration),
/// uses memory-efficient O(N) block-wise attention computation instead
/// of the standard O(N^2) attention matrix.
#[derive(Debug)]
pub struct CrossAttention {
    to_q: nn::Linear,
    to_k: nn::Linear,
    to_v: nn::Linear,
    to_out: nn::Linear,
    heads: usize,
    dim_head: usize,
    scale: f64,
    /// Flash attention module (when feature is enabled)
    #[cfg(feature = "flash_attention")]
    flash_attention: Option<FlashAttention>,
    /// Whether to use flash attention for this module
    use_flash_attention: bool,
}

impl CrossAttention {
    /// Create a new cross-attention module with standard attention.
    pub fn new(
        vs: nn::VarBuilder,
        query_dim: usize,
        context_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
    ) -> Result<Self> {
        Self::new_with_flash(vs, query_dim, context_dim, heads, dim_head, false, 64)
    }

    /// Create a new cross-attention module with optional flash attention.
    ///
    /// # Arguments
    ///
    /// * `vs` - Variable builder for weight initialization
    /// * `query_dim` - Query input dimension
    /// * `context_dim` - Context dimension (None for self-attention)
    /// * `heads` - Number of attention heads
    /// * `dim_head` - Dimension per head
    /// * `use_flash_attention` - Whether to use flash attention
    /// * `flash_block_size` - Block size for flash attention tiling
    #[allow(unused_variables)]
    pub fn new_with_flash(
        vs: nn::VarBuilder,
        query_dim: usize,
        context_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
        use_flash_attention: bool,
        flash_block_size: usize,
    ) -> Result<Self> {
        let inner_dim = dim_head * heads;
        let context_dim = context_dim.unwrap_or(query_dim);
        let scale = 1.0 / (dim_head as f64).sqrt();
        let to_q = nn::linear_no_bias(query_dim, inner_dim, vs.pp("to_q"))?;
        let to_k = nn::linear_no_bias(context_dim, inner_dim, vs.pp("to_k"))?;
        let to_v = nn::linear_no_bias(context_dim, inner_dim, vs.pp("to_v"))?;
        let to_out = nn::linear(inner_dim, query_dim, vs.pp("to_out.0"))?;

        // Initialize flash attention if feature is enabled and requested
        #[cfg(feature = "flash_attention")]
        let flash_attention = if use_flash_attention {
            let config = FlashAttentionConfig::with_block_size(flash_block_size);
            Some(FlashAttention::new(dim_head, config))
        } else {
            None
        };

        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            heads,
            dim_head,
            scale,
            #[cfg(feature = "flash_attention")]
            flash_attention,
            use_flash_attention,
        })
    }

    /// Scaled-dot-product attention (standard or flash based on configuration).
    ///
    /// Automatically dispatches to flash attention when enabled and the feature
    /// is available, otherwise uses standard O(N^2) attention.
    pub fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        self.forward_masked(xs, context, None)
    }

    /// Scaled-dot-product attention with an optional additive bias mask.
    ///
    /// `attn_mask`, when present, must broadcast against the pre-softmax score
    /// tensor of shape `(batch, heads, seq_len, ctx_len)` — typically shape
    /// `(1, 1, seq_len, ctx_len)`. Build one from a
    /// [`crate::attention_masking::AttentionMask`] via [`mask_to_bias_tensor`].
    ///
    /// A mask forces the standard (non-flash) attention path even when flash
    /// attention is enabled, since flash attention here has no additive-bias
    /// support.
    pub fn forward_masked(
        &self,
        xs: &Tensor,
        context: Option<&Tensor>,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let context = context.unwrap_or(xs);
        let (b, seq_len, _) = xs.dims3()?;
        let q = self.to_q.forward(xs)?;
        let k = self.to_k.forward(context)?;
        let v = self.to_v.forward(context)?;

        // Reshape to (B, heads, seq, dim_head) and make contiguous for matmul
        let q = q
            .reshape((b, seq_len, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()?;
        let ctx_len = k.dim(1)?;
        let k = k
            .reshape((b, ctx_len, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, ctx_len, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()?;

        // Dispatch to flash attention or standard attention. A mask always
        // routes through standard_attention, since flash attention has no
        // additive-bias support here.
        #[cfg(feature = "flash_attention")]
        let out = if attn_mask.is_none() {
            if let Some(flash) = &self.flash_attention {
                flash.forward(&q, &k, &v)?
            } else {
                self.standard_attention(&q, &k, &v, attn_mask)?
            }
        } else {
            self.standard_attention(&q, &k, &v, attn_mask)?
        };

        #[cfg(not(feature = "flash_attention"))]
        let out = self.standard_attention(&q, &k, &v, attn_mask)?;

        // Reshape back to (B, seq, inner_dim)
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq_len, ()))?;
        self.to_out.forward(&out)
    }

    /// Standard O(N^2) scaled-dot-product attention.
    ///
    /// Computes the full attention matrix. Used as fallback when flash
    /// attention is disabled, unavailable, or an `attn_mask` is present.
    fn standard_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Compute attention in f32 for numerical stability
        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn = (q.matmul(&k_t)? * self.scale)?;
        let attn = match attn_mask {
            Some(mask) => attn.broadcast_add(&mask.to_dtype(DType::F32)?)?,
            None => attn,
        };
        let attn = nn::ops::softmax_last_dim(&attn)?;
        attn.matmul(&v)?.to_dtype(in_dtype)
    }

    /// Check if flash attention is enabled for this module.
    ///
    /// Returns `true` only if flash attention was requested during construction
    /// AND the `flash_attention` feature is enabled.
    pub fn is_flash_attention_enabled(&self) -> bool {
        #[cfg(feature = "flash_attention")]
        {
            self.use_flash_attention && self.flash_attention.is_some()
        }
        #[cfg(not(feature = "flash_attention"))]
        {
            // Even if requested, flash attention is not available without the feature
            let _ = self.use_flash_attention; // Suppress unused warning
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-view transformer block
// ---------------------------------------------------------------------------

/// A transformer block with multi-view cross-attention support.
///
/// Each block contains:
/// 1. Self-attention (within each view)
/// 2. Cross-view attention (across all N views)
/// 3. Text/prompt cross-attention
/// 4. IP cross-attention (reference image CLIP embedding)
/// 5. Feed-forward network
#[derive(Debug)]
pub struct MultiViewTransformerBlock {
    /// LayerNorm before self-attention
    norm1: nn::LayerNorm,
    /// Self-attention
    attn1: CrossAttention,
    /// LayerNorm before cross-view attention
    norm_cv: nn::LayerNorm,
    /// Cross-view attention
    attn_cv: CrossAttention,
    /// LayerNorm before text cross-attention
    norm2: nn::LayerNorm,
    /// Text cross-attention
    attn2: CrossAttention,
    /// LayerNorm before IP cross-attention
    norm_ip: nn::LayerNorm,
    /// IP-adapter cross-attention
    attn_ip: CrossAttention,
    /// LayerNorm before FFN
    norm3: nn::LayerNorm,
    /// Feed-forward network
    ff: FeedForward,
    /// Number of views
    num_views: usize,
}

impl MultiViewTransformerBlock {
    /// Create a new multi-view transformer block with standard attention.
    pub fn new(
        vs: nn::VarBuilder,
        dim: usize,
        n_heads: usize,
        d_head: usize,
        context_dim: usize,
        ip_dim: usize,
        num_views: usize,
    ) -> Result<Self> {
        Self::new_with_flash(
            vs,
            dim,
            n_heads,
            d_head,
            context_dim,
            ip_dim,
            num_views,
            false,
            64,
        )
    }

    /// Create a new multi-view transformer block with optional flash attention.
    ///
    /// # Arguments
    ///
    /// * `vs` - Variable builder for weight initialization
    /// * `dim` - Hidden dimension
    /// * `n_heads` - Number of attention heads
    /// * `d_head` - Dimension per head
    /// * `context_dim` - Text cross-attention context dimension
    /// * `ip_dim` - IP-adapter context dimension
    /// * `num_views` - Number of views for cross-view attention
    /// * `use_flash_attention` - Whether to use flash attention
    /// * `flash_block_size` - Block size for flash attention tiling
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_flash(
        vs: nn::VarBuilder,
        dim: usize,
        n_heads: usize,
        d_head: usize,
        context_dim: usize,
        ip_dim: usize,
        num_views: usize,
        use_flash_attention: bool,
        flash_block_size: usize,
    ) -> Result<Self> {
        let norm1 = nn::layer_norm(dim, 1e-5, vs.pp("norm1"))?;
        let attn1 = CrossAttention::new_with_flash(
            vs.pp("attn1"),
            dim,
            None,
            n_heads,
            d_head,
            use_flash_attention,
            flash_block_size,
        )?;

        let norm_cv = nn::layer_norm(dim, 1e-5, vs.pp("norm_cv"))?;
        // Cross-view attention typically has small sequence length (num_views),
        // so flash attention may not be beneficial here
        let attn_cv = CrossAttention::new(vs.pp("attn_cv"), dim, None, n_heads, d_head)?;

        let norm2 = nn::layer_norm(dim, 1e-5, vs.pp("norm2"))?;
        let attn2 = CrossAttention::new_with_flash(
            vs.pp("attn2"),
            dim,
            Some(context_dim),
            n_heads,
            d_head,
            use_flash_attention,
            flash_block_size,
        )?;

        let norm_ip = nn::layer_norm(dim, 1e-5, vs.pp("norm_ip"))?;
        let attn_ip = CrossAttention::new_with_flash(
            vs.pp("attn_ip"),
            dim,
            Some(ip_dim),
            n_heads,
            d_head,
            use_flash_attention,
            flash_block_size,
        )?;

        let norm3 = nn::layer_norm(dim, 1e-5, vs.pp("norm3"))?;
        let ff = FeedForward::new(vs.pp("ff"), dim, 4)?;

        Ok(Self {
            norm1,
            attn1,
            norm_cv,
            attn_cv,
            norm2,
            attn2,
            norm_ip,
            attn_ip,
            norm3,
            ff,
            num_views,
        })
    }

    /// Forward pass.
    ///
    /// - `xs`: `(B*num_views, seq_len, dim)` — spatial tokens for all views (batched)
    /// - `context`: `(B*num_views, ctx_len, context_dim)` — text encoder hidden states
    /// - `ip_tokens`: `(B*num_views, ip_len, ip_dim)` — CLIP image embedding tokens
    pub fn forward(
        &self,
        xs: &Tensor,
        context: Option<&Tensor>,
        ip_tokens: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_with_mask(xs, context, ip_tokens, None)
    }

    /// Forward pass with an optional cross-view attention mask.
    ///
    /// Identical to [`Self::forward`], except `cross_view_mask` — when
    /// present — additively biases the `attn_cv` (cross-view) attention
    /// scores, restricting which views may attend to which other views.
    ///
    /// `attn_cv` operates on tokens reshaped to `(B*seq_len, num_views, dim)`,
    /// so its pre-softmax scores have shape
    /// `(B*seq_len, heads, num_views, num_views)` — **not**
    /// `(num_views * tokens_per_view)²`. `cross_view_mask` must broadcast
    /// against that, i.e. it must be built with `tokens_per_view = 1`
    /// (e.g. `attention_masking::ring_view_mask(num_views, 1)` or
    /// `attention_masking::angular_proximity_mask(num_views, 1, positions,
    /// angle)`, converted via [`mask_to_bias_tensor`]). A mask built by
    /// `attention_masking::build_layer_mask` is sized for
    /// `num_views * tokens_per_view` and will not broadcast correctly here.
    pub fn forward_with_mask(
        &self,
        xs: &Tensor,
        context: Option<&Tensor>,
        ip_tokens: Option<&Tensor>,
        cross_view_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (bv, seq_len, dim) = xs.dims3()?;
        let b = bv / self.num_views;

        // 1. Self-attention (per-view)
        let residual = xs;
        let xs = (self.attn1.forward(&self.norm1.forward(xs)?, None)? + residual)?;

        // 2. Cross-view attention
        // Reshape so each position can attend across all views
        let residual = &xs;
        let normed = self.norm_cv.forward(&xs)?;
        // (B*V, S, D) -> (B, V, S, D) -> (B, S, V, D) -> (B*S, V, D)
        let cv_input = normed
            .reshape((b, self.num_views, seq_len, dim))?
            .transpose(1, 2)?
            .reshape((b * seq_len, self.num_views, dim))?;
        let cv_out = self
            .attn_cv
            .forward_masked(&cv_input, None, cross_view_mask)?;
        // (B*S, V, D) -> (B, S, V, D) -> (B, V, S, D) -> (B*V, S, D)
        let cv_out = cv_out
            .reshape((b, seq_len, self.num_views, dim))?
            .transpose(1, 2)?
            .reshape((bv, seq_len, dim))?;
        let xs = (cv_out + residual)?;

        // 3. Text cross-attention
        let residual = &xs;
        let xs = (self.attn2.forward(&self.norm2.forward(&xs)?, context)? + residual)?;

        // 4. IP cross-attention (reference image conditioning)
        let xs = if let Some(ip) = ip_tokens {
            let residual = &xs;
            (self
                .attn_ip
                .forward(&self.norm_ip.forward(&xs)?, Some(ip))?
                + residual)?
        } else {
            xs
        };

        // 5. Feed-forward
        let residual = &xs;
        self.ff.forward(&self.norm3.forward(&xs)?)? + residual
    }
}

// ---------------------------------------------------------------------------
// Multi-view spatial transformer (wraps projection + transformer blocks)
// ---------------------------------------------------------------------------

/// Spatial-to-token projection, matching either SD 2.x's linear projection or
/// SD 1.5 / Zero123's 1×1-conv projection, selected at construction time via
/// `use_linear_projection` so the loaded checkpoint's weight layout matches:
/// `(out, in)` for `Linear`, `(out, in, 1, 1)` for `Conv`.
#[derive(Debug)]
enum Projection {
    Linear(nn::Linear),
    Conv(nn::Conv2d),
}

/// A spatial transformer that includes multi-view attention in every block.
/// Replaces the standard `SpatialTransformer` from SD 2.1.
#[derive(Debug)]
pub struct MultiViewSpatialTransformer {
    norm: nn::GroupNorm,
    proj_in: Projection,
    transformer_blocks: Vec<MultiViewTransformerBlock>,
    proj_out: Projection,
    /// Transformer hidden dimension (`n_heads * d_head`).
    inner_dim: usize,
    /// Input/output feature-map channel count.
    in_channels: usize,
}

impl MultiViewSpatialTransformer {
    /// Create a new multi-view spatial transformer with standard attention.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        n_heads: usize,
        d_head: usize,
        depth: usize,
        context_dim: usize,
        ip_dim: usize,
        num_views: usize,
        num_groups: usize,
        use_linear_projection: bool,
    ) -> Result<Self> {
        Self::new_with_flash(
            vs,
            in_channels,
            n_heads,
            d_head,
            depth,
            context_dim,
            ip_dim,
            num_views,
            num_groups,
            use_linear_projection,
            false,
            64,
        )
    }

    /// Create a new multi-view spatial transformer with optional flash attention.
    ///
    /// # Arguments
    ///
    /// * `vs` - Variable builder for weight initialization
    /// * `in_channels` - Number of input channels
    /// * `n_heads` - Number of attention heads
    /// * `d_head` - Dimension per head
    /// * `depth` - Number of transformer blocks
    /// * `context_dim` - Text cross-attention context dimension
    /// * `ip_dim` - IP-adapter context dimension
    /// * `num_views` - Number of views for cross-view attention
    /// * `num_groups` - Number of groups for group normalization
    /// * `use_linear_projection` - Whether to use linear projection
    /// * `use_flash_attention` - Whether to use flash attention
    /// * `flash_block_size` - Block size for flash attention tiling
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_flash(
        vs: nn::VarBuilder,
        in_channels: usize,
        n_heads: usize,
        d_head: usize,
        depth: usize,
        context_dim: usize,
        ip_dim: usize,
        num_views: usize,
        num_groups: usize,
        use_linear_projection: bool,
        use_flash_attention: bool,
        flash_block_size: usize,
    ) -> Result<Self> {
        let inner_dim = n_heads * d_head;
        let norm = nn::group_norm(num_groups, in_channels, 1e-6, vs.pp("norm"))?;

        // SD 2.x checkpoints store proj_in/proj_out as (out, in) Linear
        // weights; SD 1.5 / Zero123 checkpoints store the *same* keys as
        // (out, in, 1, 1) 1x1-Conv2d weights. Building the layer type that
        // matches `use_linear_projection` (rather than always building
        // Linear) is what makes loading either checkpoint family possible.
        let (proj_in, proj_out) = if use_linear_projection {
            (
                Projection::Linear(nn::linear(in_channels, inner_dim, vs.pp("proj_in"))?),
                Projection::Linear(nn::linear(inner_dim, in_channels, vs.pp("proj_out"))?),
            )
        } else {
            (
                Projection::Conv(nn::conv2d(
                    in_channels,
                    inner_dim,
                    1,
                    Default::default(),
                    vs.pp("proj_in"),
                )?),
                Projection::Conv(nn::conv2d(
                    inner_dim,
                    in_channels,
                    1,
                    Default::default(),
                    vs.pp("proj_out"),
                )?),
            )
        };

        let vs_tb = vs.pp("transformer_blocks");
        let mut transformer_blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            transformer_blocks.push(MultiViewTransformerBlock::new_with_flash(
                vs_tb.pp(i.to_string()),
                inner_dim,
                n_heads,
                d_head,
                context_dim,
                ip_dim,
                num_views,
                use_flash_attention,
                flash_block_size,
            )?);
        }

        Ok(Self {
            norm,
            proj_in,
            transformer_blocks,
            proj_out,
            inner_dim,
            in_channels,
        })
    }

    /// Forward pass.
    ///
    /// - `xs`: `(B*V, C, H, W)` feature map
    /// - `context`: optional text cross-attention context
    /// - `ip_tokens`: optional IP-adapter tokens
    pub fn forward(
        &self,
        xs: &Tensor,
        context: Option<&Tensor>,
        ip_tokens: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_with_mask(xs, context, ip_tokens, None)
    }

    /// Forward pass with an optional cross-view attention mask, threaded
    /// through to every block's `attn_cv`. See
    /// [`MultiViewTransformerBlock::forward_with_mask`] for the required
    /// mask shape (built with `tokens_per_view = 1`).
    pub fn forward_with_mask(
        &self,
        xs: &Tensor,
        context: Option<&Tensor>,
        ip_tokens: Option<&Tensor>,
        cross_view_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, _channel, height, width) = xs.dims4()?;
        let residual = xs;

        let xs = self.norm.forward(xs)?;

        // Project into token space. Conv-style (SD 1.5 / Zero123) applies a
        // 1x1 convolution on the (B,C,H,W) map *before* flattening, matching
        // how those checkpoints store proj_in as a (out,in,1,1) conv weight;
        // linear-style (SD 2.x) flattens first, matching a (out,in) linear
        // weight. Both paths produce the same (B, H*W, inner_dim) token
        // layout for the transformer blocks.
        let tokens = match &self.proj_in {
            Projection::Linear(l) => {
                let flat = xs.transpose(1, 2)?.transpose(2, 3)?.reshape((
                    batch,
                    height * width,
                    self.in_channels,
                ))?;
                l.forward(&flat)?
            }
            Projection::Conv(c) => {
                let projected = c.forward(&xs)?; // (B, inner_dim, H, W)
                projected.transpose(1, 2)?.transpose(2, 3)?.reshape((
                    batch,
                    height * width,
                    self.inner_dim,
                ))?
            }
        };

        let mut h = tokens;
        for block in &self.transformer_blocks {
            h = block.forward_with_mask(&h, context, ip_tokens, cross_view_mask)?;
        }

        let result = match &self.proj_out {
            Projection::Linear(l) => {
                let h = l.forward(&h)?; // (B, HW, in_channels)
                h.reshape((batch, height, width, self.in_channels))?
                    .transpose(2, 3)?
                    .transpose(1, 2)?
            }
            Projection::Conv(c) => {
                let h_nchw = h
                    .reshape((batch, height, width, self.inner_dim))?
                    .transpose(2, 3)?
                    .transpose(1, 2)?
                    .contiguous()?;
                c.forward(&h_nchw)? // (B, in_channels, H, W)
            }
        };
        result + residual
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// This file previously had no inline test module (existing coverage lives in
// tests/attention_tests.rs). These tests are scoped to the fixes made here:
// the Conv2d projection branch and attention-mask threading. Note: a
// VarMap-backed VarBuilder *creates* each requested variable fresh at
// whatever shape is asked for, rather than loading real checkpoint data, so
// these tests can verify internal shape consistency and that the Conv2d code
// path is genuinely exercised and distinct from the Linear path — they
// cannot verify compatibility with a real SD 1.5 / Zero123 safetensors file.
#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn test_varbuilder() -> nn::VarBuilder<'static> {
        let varmap = VarMap::new();
        nn::VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu)
    }

    #[test]
    fn test_multi_view_spatial_transformer_conv_projection_shape() -> Result<()> {
        let vs = test_varbuilder();
        let in_channels = 8;
        let transformer = MultiViewSpatialTransformer::new(
            vs.pp("t"),
            in_channels,
            2, // n_heads
            4, // d_head
            1, // depth
            16,
            16,    // context_dim, ip_dim
            2,     // num_views
            4,     // num_groups
            false, // use_linear_projection -> exercises the Conv2d branch
        )?;
        let batch_views = 2; // B=1 * V=2
        let (h, w) = (4, 4);
        let xs = Tensor::randn(0f32, 1f32, (batch_views, in_channels, h, w), &Device::Cpu)?;
        let out = transformer.forward(&xs, None, None)?;
        assert_eq!(out.dims4()?, (batch_views, in_channels, h, w));
        Ok(())
    }

    #[test]
    fn test_multi_view_spatial_transformer_linear_projection_shape() -> Result<()> {
        let vs = test_varbuilder();
        let in_channels = 8;
        let transformer = MultiViewSpatialTransformer::new(
            vs.pp("t"),
            in_channels,
            2,
            4,
            1,
            16,
            16,
            2,
            4,
            true, // use_linear_projection
        )?;
        let batch_views = 2;
        let (h, w) = (4, 4);
        let xs = Tensor::randn(0f32, 1f32, (batch_views, in_channels, h, w), &Device::Cpu)?;
        let out = transformer.forward(&xs, None, None)?;
        assert_eq!(out.dims4()?, (batch_views, in_channels, h, w));
        Ok(())
    }

    #[test]
    fn test_mask_to_bias_tensor_shape() -> Result<()> {
        let mut mask = AttentionMask::new(4, true);
        mask.set(0, 1, false);
        let bias = mask_to_bias_tensor(&mask, &Device::Cpu)?;
        assert_eq!(bias.dims4()?, (1, 1, 4, 4));
        Ok(())
    }

    #[test]
    fn test_cross_attention_forward_masked_changes_output() -> Result<()> {
        let vs = test_varbuilder();
        let query_dim = 8;
        let attn = CrossAttention::new(vs.pp("attn"), query_dim, None, 2, 4)?;

        let seq_len = 4;
        let xs = Tensor::randn(0f32, 1f32, (1, seq_len, query_dim), &Device::Cpu)?;
        let unmasked = attn.forward_masked(&xs, None, None)?;

        // Block position 0 from attending to anything but itself.
        let mut mask = AttentionMask::new(seq_len, true);
        for k in 1..seq_len {
            mask.set(0, k, false);
        }
        let bias = mask_to_bias_tensor(&mask, &Device::Cpu)?;
        let masked = attn.forward_masked(&xs, None, Some(&bias))?;

        let diff = unmasked
            .sub(&masked)?
            .abs()?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff > 1e-4,
            "masking should change the attention output, diff={diff}"
        );
        Ok(())
    }

    #[test]
    fn test_multi_view_transformer_block_cross_view_mask_changes_output() -> Result<()> {
        let vs = test_varbuilder();
        let dim = 8;
        let num_views = 3;
        let block = MultiViewTransformerBlock::new(vs.pp("block"), dim, 2, 4, 8, 8, num_views)?;

        let seq_len = 2; // spatial tokens per view
        let bv = num_views; // B=1, V=num_views
        let xs = Tensor::randn(0f32, 1f32, (bv, seq_len, dim), &Device::Cpu)?;

        let unmasked = block.forward(&xs, None, None)?;

        // cross_view_mask is sized (num_views, num_views) -- built with
        // tokens_per_view = 1 -- not (num_views * tokens_per_view)^2.
        let mut mask = AttentionMask::new(num_views, true);
        for k in 1..num_views {
            mask.set(0, k, false);
        }
        let bias = mask_to_bias_tensor(&mask, &Device::Cpu)?;
        let masked = block.forward_with_mask(&xs, None, None, Some(&bias))?;

        let diff = unmasked
            .sub(&masked)?
            .abs()?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff > 1e-4,
            "cross_view_mask should change the block output, diff={diff}"
        );
        Ok(())
    }
}
