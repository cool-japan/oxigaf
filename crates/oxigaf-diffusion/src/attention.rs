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

use candle_core::{DType, Result, Tensor, D};
use candle_nn as nn;
use candle_nn::Module;

#[cfg(feature = "flash_attention")]
use crate::flash_attention::{FlashAttention, FlashAttentionConfig};

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

        // Dispatch to flash attention or standard attention
        #[cfg(feature = "flash_attention")]
        let out = if let Some(flash) = &self.flash_attention {
            flash.forward(&q, &k, &v)?
        } else {
            self.standard_attention(&q, &k, &v)?
        };

        #[cfg(not(feature = "flash_attention"))]
        let out = self.standard_attention(&q, &k, &v)?;

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
    /// attention is disabled or unavailable.
    fn standard_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        // Compute attention in f32 for numerical stability
        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn = (q.matmul(&k_t)? * self.scale)?;
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
        let cv_out = self.attn_cv.forward(&cv_input, None)?;
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

/// A spatial transformer that includes multi-view attention in every block.
/// Replaces the standard `SpatialTransformer` from SD 2.1.
#[derive(Debug)]
pub struct MultiViewSpatialTransformer {
    norm: nn::GroupNorm,
    proj_in: nn::Linear,
    transformer_blocks: Vec<MultiViewTransformerBlock>,
    proj_out: nn::Linear,
    use_linear_projection: bool,
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
        let proj_in = nn::linear(in_channels, inner_dim, vs.pp("proj_in"))?;
        let proj_out = nn::linear(inner_dim, in_channels, vs.pp("proj_out"))?;

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
            use_linear_projection,
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
        let (batch, _channel, height, width) = xs.dims4()?;
        let residual = xs;

        let xs = self.norm.forward(xs)?;
        // Flatten spatial dims and optionally project
        let inner_dim = if self.use_linear_projection {
            let inner_dim = xs.dim(1)?;
            let xs_flat =
                xs.transpose(1, 2)?
                    .transpose(2, 3)?
                    .reshape((batch, height * width, inner_dim))?;
            let xs_proj = self.proj_in.forward(&xs_flat)?;
            // Process through transformer blocks
            let mut h = xs_proj;
            for block in &self.transformer_blocks {
                h = block.forward(&h, context, ip_tokens)?;
            }
            let h = self.proj_out.forward(&h)?;
            let result = h
                .reshape((batch, height, width, inner_dim))?
                .transpose(2, 3)?
                .transpose(1, 2)?;
            return result + residual;
        } else {
            xs.dim(1)?
        };

        // Conv-style projection path (for completeness, though SD 2.1 uses linear)
        let xs_flat =
            xs.transpose(1, 2)?
                .transpose(2, 3)?
                .reshape((batch, height * width, inner_dim))?;
        let xs_proj = self.proj_in.forward(&xs_flat)?;
        let mut h = xs_proj;
        for block in &self.transformer_blocks {
            h = block.forward(&h, context, ip_tokens)?;
        }
        let h = self.proj_out.forward(&h)?;
        let result = h
            .reshape((batch, height, width, inner_dim))?
            .transpose(2, 3)?
            .transpose(1, 2)?;
        result + residual
    }
}
