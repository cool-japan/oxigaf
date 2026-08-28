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

use std::sync::Arc;

use crate::attention_masking::AttentionMask;
use crate::config::{AttentionBackend, DiffusionConfig};
use crate::kv_cache::{CacheKeyBuilder, KVCache};
use crate::sliced_attention::{SlicedAttention, SlicedAttentionConfig};

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
    /// Kernel this layer dispatches to; see [`AttentionBackend`].
    backend: AttentionBackend,
    /// Chunked-query kernel, built when `backend == AttentionBackend::Sliced`.
    sliced: SlicedHandle,
}

/// An optional [`SlicedAttention`] that can live inside a `#[derive(Debug)]`
/// model type.
///
/// [`SlicedAttention`] does not implement `Debug`, so storing it directly would
/// force a hand-written `Debug` on [`CrossAttention`] and every model type that
/// contains one. This newtype absorbs that, printing only whether the kernel is
/// selected.
#[derive(Default)]
struct SlicedHandle(Option<SlicedAttention>);

impl SlicedHandle {
    /// The kernel, when this layer selected it.
    fn get(&self) -> Option<&SlicedAttention> {
        self.0.as_ref()
    }
}

impl std::fmt::Debug for SlicedHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(_) => f.write_str("SlicedHandle(enabled)"),
            None => f.write_str("SlicedHandle(none)"),
        }
    }
}

/// How one attention layer should be built.
///
/// Bundles the head geometry with the kernel selection so the constructors
/// that thread it from [`crate::config::DiffusionConfig`] down to
/// [`CrossAttention`] stay within a readable argument count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionSpec {
    /// Number of attention heads.
    pub heads: usize,
    /// Width of each head.
    pub dim_head: usize,
    /// Kernel to run.
    pub backend: AttentionBackend,
    /// Tile width for [`AttentionBackend::Flash`].
    pub flash_block_size: usize,
    /// Queries per slice for [`AttentionBackend::Sliced`]; `None` = one slice.
    pub slice_size: Option<usize>,
}

impl AttentionSpec {
    /// A spec for `heads × dim_head` running the standard kernel.
    pub fn standard(heads: usize, dim_head: usize) -> Self {
        Self {
            heads,
            dim_head,
            backend: AttentionBackend::Standard,
            flash_block_size: 64,
            slice_size: Some(64),
        }
    }

    /// Derive the head geometry and kernel selection from a model config.
    ///
    /// Uses [`crate::config::DiffusionConfig::resolved_attention_backend`], so
    /// both the legacy `use_flash_attention` flag and the newer
    /// `attention_backend` selector are honoured.
    pub fn from_config(config: &DiffusionConfig, heads: usize, dim_head: usize) -> Self {
        Self {
            heads,
            dim_head,
            backend: config.resolved_attention_backend(),
            flash_block_size: config.flash_attention_block_size,
            slice_size: config.attention_slice_size,
        }
    }

    /// The same spec with a different kernel.
    pub fn with_backend(mut self, backend: AttentionBackend) -> Self {
        self.backend = backend;
        self
    }
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
    /// Equivalent to [`Self::with_spec`] with a spec whose backend is
    /// [`AttentionBackend::Flash`] or [`AttentionBackend::Standard`].
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
    pub fn new_with_flash(
        vs: nn::VarBuilder,
        query_dim: usize,
        context_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
        use_flash_attention: bool,
        flash_block_size: usize,
    ) -> Result<Self> {
        let backend = if use_flash_attention {
            AttentionBackend::Flash
        } else {
            AttentionBackend::Standard
        };
        Self::with_spec(
            vs,
            query_dim,
            context_dim,
            &AttentionSpec {
                heads,
                dim_head,
                backend,
                flash_block_size,
                slice_size: None,
            },
        )
    }

    /// Create a cross-attention module for an explicit [`AttentionSpec`].
    ///
    /// This is the constructor the U-Net uses, and the only one that can select
    /// [`AttentionBackend::Sliced`].
    ///
    /// # Errors
    ///
    /// Propagates weight-loading failures, and reports a
    /// [`SlicedAttentionConfig`] that [`SlicedAttention::new`] rejects (a zero
    /// head count, head width or slice size) as
    /// [`candle_core::Error::Msg`].
    pub fn with_spec(
        vs: nn::VarBuilder,
        query_dim: usize,
        context_dim: Option<usize>,
        spec: &AttentionSpec,
    ) -> Result<Self> {
        let heads = spec.heads;
        let dim_head = spec.dim_head;
        let use_flash_attention = spec.backend == AttentionBackend::Flash;
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
            let config = FlashAttentionConfig::with_block_size(spec.flash_block_size);
            Some(FlashAttention::new(dim_head, config))
        } else {
            None
        };

        let sliced = if spec.backend == AttentionBackend::Sliced {
            let config = SlicedAttentionConfig::new(spec.slice_size, heads, dim_head);
            SlicedHandle(Some(SlicedAttention::new(config).map_err(|e| {
                candle_core::Error::Msg(format!("sliced attention: {e}"))
            })?))
        } else {
            SlicedHandle::default()
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
            backend: spec.backend,
            sliced,
        })
    }

    /// The kernel this layer dispatches to.
    pub fn backend(&self) -> AttentionBackend {
        self.backend
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

        self.attend(&q, &k, &v, attn_mask, b, seq_len)
    }

    /// Run the attention kernel over already-projected `(B, heads, seq, dim)`
    /// tensors and apply the output projection.
    ///
    /// Shared by [`Self::forward_masked`] and [`Self::forward_cached`] so both
    /// take the identical flash/standard dispatch.
    fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        b: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        // A mask always routes through standard_attention: neither the flash
        // nor the sliced kernel supports an additive bias.
        let out = match self.sliced.get() {
            Some(sliced) if attn_mask.is_none() => self.sliced_attention(sliced, q, k, v)?,
            _ => {
                #[cfg(feature = "flash_attention")]
                {
                    if attn_mask.is_none() {
                        if let Some(flash) = &self.flash_attention {
                            flash.forward(q, k, v)?
                        } else {
                            self.standard_attention(q, k, v, attn_mask)?
                        }
                    } else {
                        self.standard_attention(q, k, v, attn_mask)?
                    }
                }
                #[cfg(not(feature = "flash_attention"))]
                {
                    self.standard_attention(q, k, v, attn_mask)?
                }
            }
        };

        // Reshape back to (B, seq, inner_dim)
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq_len, ()))?;
        self.to_out.forward(&out)
    }

    /// Chunked-query attention via [`crate::sliced_attention`].
    ///
    /// That kernel works on flat `f32` buffers, so `q`/`k`/`v` are read back to
    /// the host and the result is rebuilt as a tensor. The round trip is the
    /// price of bounding peak score-matrix memory at
    /// `slice_size × ctx_len` instead of `seq_len × ctx_len`; a non-`f32` input
    /// takes the standard path instead of being silently downcast.
    fn sliced_attention(
        &self,
        sliced: &SlicedAttention,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<Tensor> {
        if q.dtype() != DType::F32 {
            return self.standard_attention(q, k, v, None);
        }
        let (batch, _, seq_len_q, _) = q.dims4()?;
        let seq_len_k = k.dim(2)?;

        let read = |t: &Tensor| -> Result<Vec<f32>> { t.flatten_all()?.to_vec1::<f32>() };
        let out = sliced
            .forward(&read(q)?, &read(k)?, &read(v)?, batch, seq_len_q, seq_len_k)
            .map_err(|e| candle_core::Error::Msg(format!("sliced attention: {e}")))?;

        Tensor::from_vec(
            out,
            (batch, self.heads, seq_len_q, self.dim_head),
            q.device(),
        )
    }

    /// Cross-attention whose key/value projections of `context` are served
    /// from `cache`.
    ///
    /// In a diffusion loop the cross-attention context (IP-Adapter CLIP tokens
    /// here) is *constant* across every denoising timestep, so `to_k(context)`
    /// and `to_v(context)` recompute the same two matmuls at every step. This
    /// method computes them once per `key` and replays the stored tensors on
    /// subsequent steps; the query projection still runs every call because the
    /// latents change.
    ///
    /// `key` must identify both the layer and the conditioning — see
    /// [`crate::kv_cache::CacheKeyBuilder`]. Reusing a key across different
    /// conditioning would replay the wrong K/V.
    ///
    /// # Falls back to the uncached path
    ///
    /// - When `xs` or `context` is not `f32`:
    ///   [`KVEntry`][crate::kv_cache::KVEntry] stores `f32`, so a cached round
    ///   trip would silently change the dtype of a mixed-precision run.
    /// - When a cached entry's declared shape does not match this call's
    ///   `(batch, heads, ctx_len, dim_head)` — a stale entry is ignored rather
    ///   than reinterpreted.
    ///
    /// Both fallbacks produce exactly the same result as
    /// [`Self::forward`]; only the cache hit-rate changes.
    ///
    /// # Errors
    ///
    /// Propagates tensor-operation failures, and surfaces a cache failure as
    /// [`candle_core::Error::Msg`].
    pub fn forward_cached(
        &self,
        xs: &Tensor,
        context: &Tensor,
        cache: &KVCache,
        key: &str,
    ) -> Result<Tensor> {
        if xs.dtype() != DType::F32 || context.dtype() != DType::F32 {
            return self.forward(xs, Some(context));
        }

        let (b, seq_len, _) = xs.dims3()?;
        let ctx_len = context.dim(1)?;
        let q = self
            .to_q
            .forward(xs)?
            .reshape((b, seq_len, self.heads, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()?;

        let project = |linear: &nn::Linear| -> Result<Tensor> {
            linear
                .forward(context)?
                .reshape((b, ctx_len, self.heads, self.dim_head))?
                .transpose(1, 2)?
                .contiguous()
        };

        let entry = cache
            .get_or_compute(key.to_string(), || {
                let k = project(&self.to_k)?;
                let v = project(&self.to_v)?;
                Ok((
                    k.flatten_all()?.to_vec1::<f32>()?,
                    v.flatten_all()?.to_vec1::<f32>()?,
                    b,
                    self.heads,
                    ctx_len,
                    self.dim_head,
                ))
            })
            .map_err(|e| candle_core::Error::Msg(format!("KV cache: {e}")))?;

        let shape = (b, self.heads, ctx_len, self.dim_head);
        if (entry.batch, entry.num_heads, entry.seq_k, entry.head_dim) != shape {
            // A stale entry for this key: recompute without touching it.
            let k = project(&self.to_k)?;
            let v = project(&self.to_v)?;
            return self.attend(&q, &k, &v, None, b, seq_len);
        }

        let device = xs.device();
        let k = Tensor::from_slice(&entry.keys, shape, device)?;
        let v = Tensor::from_slice(&entry.values, shape, device)?;
        self.attend(&q, &k, &v, None, b, seq_len)
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
// KV-cache handle
// ---------------------------------------------------------------------------

/// An optional shared [`KVCache`] that can live inside a `#[derive(Debug)]`
/// model type.
///
/// [`KVCache`] holds a `Mutex<HashMap<..>>` and does not implement `Debug`, so
/// storing it directly would force a hand-written `Debug` on every enclosing
/// model struct. This newtype absorbs that: it prints whether a cache is
/// attached and nothing about its contents (which are large, and behind a lock
/// that a `Debug` impl must not block on).
#[derive(Default, Clone)]
struct KvCacheHandle(Option<Arc<KVCache>>);

impl KvCacheHandle {
    /// The cache, when one is attached.
    fn get(&self) -> Option<&KVCache> {
        self.0.as_deref()
    }

    /// `true` when a cache is attached.
    fn is_attached(&self) -> bool {
        self.0.is_some()
    }
}

impl std::fmt::Debug for KvCacheHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(_) => f.write_str("KvCacheHandle(attached)"),
            None => f.write_str("KvCacheHandle(none)"),
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
    /// Shared cross-attention KV cache, when one has been attached.
    ///
    /// Only `attn_ip` consults it: the IP-Adapter CLIP tokens are the one
    /// context that is genuinely constant across every denoising step of a
    /// run. `attn1`/`attn_cv` derive their K/V from the latents (which change
    /// every step), and `attn2`'s null context is cheap and stage-shaped.
    kv_cache: KvCacheHandle,
    /// Cache key for this block's `attn_ip` K/V projection; identifies both the
    /// layer and the conditioning it was computed from.
    ip_cache_key: String,
}

impl MultiViewTransformerBlock {
    /// Create a new multi-view transformer block with standard attention.
    ///
    /// Convenience wrapper over [`Self::with_spec`] for the standard kernel;
    /// use `with_spec` directly to select [`AttentionBackend::Flash`] or
    /// [`AttentionBackend::Sliced`].
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
    pub fn new(
        vs: nn::VarBuilder,
        dim: usize,
        n_heads: usize,
        d_head: usize,
        context_dim: usize,
        ip_dim: usize,
        num_views: usize,
    ) -> Result<Self> {
        Self::with_spec(
            vs,
            dim,
            &AttentionSpec::standard(n_heads, d_head),
            context_dim,
            ip_dim,
            num_views,
        )
    }

    /// Create a block whose attention layers follow an explicit
    /// [`AttentionSpec`].
    ///
    /// The cross-view layer (`attn_cv`) always uses the standard kernel: its
    /// sequence length is `num_views` (typically 4), far below the point where
    /// a tiled or chunked kernel pays for its overhead.
    ///
    /// # Errors
    ///
    /// Propagates weight-loading failures and an invalid sliced-attention
    /// configuration; see [`CrossAttention::with_spec`].
    pub fn with_spec(
        vs: nn::VarBuilder,
        dim: usize,
        spec: &AttentionSpec,
        context_dim: usize,
        ip_dim: usize,
        num_views: usize,
    ) -> Result<Self> {
        let norm1 = nn::layer_norm(dim, 1e-5, vs.pp("norm1"))?;
        let attn1 = CrossAttention::with_spec(vs.pp("attn1"), dim, None, spec)?;

        let norm_cv = nn::layer_norm(dim, 1e-5, vs.pp("norm_cv"))?;
        // Cross-view attention has a sequence length of `num_views`, so a
        // memory-optimising kernel would only add overhead here.
        let cv_spec = spec.with_backend(AttentionBackend::Standard);
        let attn_cv = CrossAttention::with_spec(vs.pp("attn_cv"), dim, None, &cv_spec)?;

        let norm2 = nn::layer_norm(dim, 1e-5, vs.pp("norm2"))?;
        let attn2 = CrossAttention::with_spec(vs.pp("attn2"), dim, Some(context_dim), spec)?;

        let norm_ip = nn::layer_norm(dim, 1e-5, vs.pp("norm_ip"))?;
        let attn_ip = CrossAttention::with_spec(vs.pp("attn_ip"), dim, Some(ip_dim), spec)?;

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
            kv_cache: KvCacheHandle::default(),
            ip_cache_key: String::new(),
        })
    }

    /// Attach (or, with `None`, detach) the shared cross-attention KV cache.
    ///
    /// `layer_idx` and `block_idx` identify this block within the model and
    /// `conditioning_tag` identifies the IP-Adapter tokens the cached
    /// projections were computed from — changing the reference image must
    /// change the tag, or the block would replay another image's K/V.
    ///
    /// Detaching leaves any entries in the cache; they simply stop being
    /// consulted.
    pub fn set_kv_cache(
        &mut self,
        cache: Option<Arc<KVCache>>,
        layer_idx: usize,
        block_idx: usize,
        conditioning_tag: u64,
    ) {
        self.ip_cache_key = CacheKeyBuilder::new()
            .layer(layer_idx)
            .head_group(block_idx)
            .conditioning_hash(conditioning_tag)
            .build();
        self.kv_cache = KvCacheHandle(cache);
    }

    /// The cache key this block's `attn_ip` layer uses, when a cache is
    /// attached.
    pub fn ip_cache_key(&self) -> Option<&str> {
        self.kv_cache
            .is_attached()
            .then_some(self.ip_cache_key.as_str())
    }

    /// Forward pass.
    ///
    /// - `xs`: `(B*num_views, seq_len, dim)` — spatial tokens for all views (batched)
    /// - `context`: `(B*num_views, ctx_len, context_dim)` — text encoder hidden
    ///   states. `None` skips the text cross-attention layer entirely (the
    ///   block then contributes nothing from `attn2`), exactly as `None`
    ///   `ip_tokens` skips `attn_ip` for the CFG unconditional pass.
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

        // 3. Text cross-attention.
        //
        // Skipped entirely when no context is supplied, mirroring the IP
        // branch below. `CrossAttention::forward` substitutes `xs` for a
        // missing context, and `attn2`'s `to_k`/`to_v` are built for
        // `context_dim` — so feeding it `xs` (width `dim`) was a hard matmul
        // shape error whenever `dim != context_dim`, which is every real SD
        // 2.1 stage (320/640/1280 channels vs a 1024-wide context).
        let xs = if let Some(ctx) = context {
            let residual = &xs;
            (self.attn2.forward(&self.norm2.forward(&xs)?, Some(ctx))? + residual)?
        } else {
            xs
        };

        // 4. IP cross-attention (reference image conditioning).
        //
        // The IP tokens are fixed for a whole denoising run, so `to_k`/`to_v`
        // over them are served from the shared KV cache when one is attached.
        let xs = if let Some(ip) = ip_tokens {
            let residual = &xs;
            let normed = self.norm_ip.forward(&xs)?;
            let attended = match self.kv_cache.get() {
                Some(cache) => {
                    self.attn_ip
                        .forward_cached(&normed, ip, cache, &self.ip_cache_key)?
                }
                None => self.attn_ip.forward(&normed, Some(ip))?,
            };
            (attended + residual)?
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

/// Everything one [`MultiViewSpatialTransformer`] stage needs to be built.
///
/// Groups the layer geometry with its [`AttentionSpec`] so
/// [`crate::unet::MultiViewUNet`] can pass a single value down through the
/// encoder/decoder block constructors instead of eight positional arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpatialTransformerSpec {
    /// Input/output feature-map channel count.
    pub in_channels: usize,
    /// Number of transformer blocks in this stage.
    pub depth: usize,
    /// Text cross-attention context dimension.
    pub context_dim: usize,
    /// IP-adapter context dimension.
    pub ip_dim: usize,
    /// Number of views attended across.
    pub num_views: usize,
    /// Group count for the input group-norm.
    pub num_groups: usize,
    /// `true` for the SD 2.x `Linear` projection, `false` for the SD 1.5 /
    /// Zero123 1×1-`Conv2d` projection.
    pub use_linear_projection: bool,
    /// How each attention layer in this stage is built.
    pub attention: AttentionSpec,
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
    /// Create a spatial transformer from a full [`SpatialTransformerSpec`].
    ///
    /// This is the **only** constructor, and the one
    /// [`crate::unet::MultiViewUNet`] uses; it is the path through which a
    /// [`crate::config::DiffusionConfig`]'s attention backend reaches the
    /// attention layers. Two positional constructors (`new`, taking ten
    /// arguments, and `new_with_flash`, taking twelve) used to shadow it; both
    /// did nothing but assemble a `SpatialTransformerSpec` from a run of
    /// same-typed `usize`s in which `context_dim`, `ip_dim`, `num_views` and
    /// `num_groups` were adjacent and silently interchangeable at every call
    /// site. Build the spec with named fields instead.
    ///
    /// # Errors
    ///
    /// Propagates weight-loading failures and an invalid sliced-attention
    /// configuration; see [`CrossAttention::with_spec`].
    pub fn with_spec(vs: nn::VarBuilder, spec: &SpatialTransformerSpec) -> Result<Self> {
        let SpatialTransformerSpec {
            in_channels,
            depth,
            context_dim,
            ip_dim,
            num_views,
            num_groups,
            use_linear_projection,
            attention,
        } = *spec;
        let n_heads = attention.heads;
        let d_head = attention.dim_head;
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
            transformer_blocks.push(MultiViewTransformerBlock::with_spec(
                vs_tb.pp(i.to_string()),
                inner_dim,
                &attention,
                context_dim,
                ip_dim,
                num_views,
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

    /// Attach (or, with `None`, detach) the shared cross-attention KV cache on
    /// every transformer block of this layer.
    ///
    /// `layer_idx` must be unique across the model — see
    /// [`crate::unet::MultiViewUNet::set_kv_cache`], which assigns the indices.
    /// Each block is keyed by its own position within this layer, so a
    /// `depth > 1` transformer keeps its blocks' projections apart.
    pub fn set_kv_cache(
        &mut self,
        cache: Option<Arc<KVCache>>,
        layer_idx: usize,
        conditioning_tag: u64,
    ) {
        for (block_idx, block) in self.transformer_blocks.iter_mut().enumerate() {
            block.set_kv_cache(cache.clone(), layer_idx, block_idx, conditioning_tag);
        }
    }

    /// Number of transformer blocks in this layer.
    pub fn depth(&self) -> usize {
        self.transformer_blocks.len()
    }

    /// The attention backend this layer's blocks were built with.
    ///
    /// Returns `None` for a zero-depth layer. Exposed so callers can confirm
    /// that a [`crate::config::DiffusionConfig`]'s selection reached the
    /// attention layers.
    pub fn attention_backend(&self) -> Option<AttentionBackend> {
        self.transformer_blocks
            .first()
            .map(|block| block.attn1.backend())
    }

    /// The `attn_ip` cache keys of this layer's blocks, in block order.
    ///
    /// Empty when no cache is attached. Exposed so callers (and tests) can
    /// verify that [`crate::unet::MultiViewUNet::set_kv_cache`] assigned a
    /// distinct key to every attention site.
    pub fn ip_cache_keys(&self) -> impl Iterator<Item = &str> {
        self.transformer_blocks
            .iter()
            .filter_map(|block| block.ip_cache_key())
    }

    /// Forward pass.
    ///
    /// - `xs`: `(B*V, C, H, W)` feature map
    /// - `context`: optional text cross-attention context (`None` skips the
    ///   text cross-attention layer entirely)
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

    /// A standard-kernel [`SpatialTransformerSpec`] over the shared test
    /// geometry: 2 heads of width 4 across 2 views, group-normed in 4 groups.
    fn test_spec(
        in_channels: usize,
        depth: usize,
        context_dim: usize,
        ip_dim: usize,
        use_linear_projection: bool,
    ) -> SpatialTransformerSpec {
        SpatialTransformerSpec {
            in_channels,
            depth,
            context_dim,
            ip_dim,
            num_views: 2,
            num_groups: 4,
            use_linear_projection,
            attention: AttentionSpec::standard(2, 4),
        }
    }

    #[test]
    fn test_multi_view_spatial_transformer_conv_projection_shape() -> Result<()> {
        let vs = test_varbuilder();
        let in_channels = 8;
        // `use_linear_projection: false` exercises the Conv2d branch.
        let transformer = MultiViewSpatialTransformer::with_spec(
            vs.pp("t"),
            &test_spec(in_channels, 1, 16, 16, false),
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
        let transformer = MultiViewSpatialTransformer::with_spec(
            vs.pp("t"),
            &test_spec(in_channels, 1, 16, 16, true),
        )?;
        let batch_views = 2;
        let (h, w) = (4, 4);
        let xs = Tensor::randn(0f32, 1f32, (batch_views, in_channels, h, w), &Device::Cpu)?;
        let out = transformer.forward(&xs, None, None)?;
        assert_eq!(out.dims4()?, (batch_views, in_channels, h, w));
        Ok(())
    }

    /// Regression: `forward(.., context: None, ..)` fed `xs` to `attn2`, whose
    /// `to_k`/`to_v` are built for `context_dim`. Whenever the two widths
    /// differ — which is every real SD 2.1 stage — that was a hard
    /// "shape mismatch in matmul" error, so `MultiViewUNet::forward` could
    /// never actually pass the `None` its signature advertises.
    #[test]
    fn test_transformer_block_without_context_skips_text_attention() -> Result<()> {
        let vs = test_varbuilder();
        let dim = 8;
        let context_dim = 16; // deliberately != dim
        let block = MultiViewTransformerBlock::new(vs.pp("b"), dim, 2, 4, context_dim, 16, 2)?;

        let xs = Tensor::randn(0f32, 1f32, (2usize, 4usize, dim), &Device::Cpu)?;
        let without = block.forward(&xs, None, None)?;
        assert_eq!(without.dims3()?, (2, 4, dim));

        // With a context of the declared width the layer runs and changes the
        // result, proving the `None` path really skipped a live layer rather
        // than one that happens to be a no-op.
        let context = Tensor::randn(0f32, 1f32, (2usize, 3usize, context_dim), &Device::Cpu)?;
        let with = block.forward(&xs, Some(&context), None)?;
        let diff = (&with - &without)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(
            diff > 1e-5,
            "supplying a context must change the block output, got diff {diff}"
        );
        Ok(())
    }

    #[test]
    fn test_spatial_transformer_without_context_when_widths_differ() -> Result<()> {
        // The same regression seen through the wrapper both failing inline
        // tests exercised.
        for use_linear in [false, true] {
            let vs = test_varbuilder();
            let in_channels = 8;
            // context_dim (16) != inner_dim (2 heads × 4 = 8).
            let transformer = MultiViewSpatialTransformer::with_spec(
                vs.pp("t"),
                &test_spec(in_channels, 1, 16, 16, use_linear),
            )?;
            let xs = Tensor::randn(
                0f32,
                1f32,
                (2usize, in_channels, 4usize, 4usize),
                &Device::Cpu,
            )?;
            let out = transformer.forward(&xs, None, None)?;
            assert_eq!(out.dims4()?, (2, in_channels, 4, 4));
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Cross-attention KV cache
    //
    // Regression: `KVCache` was never handed to any attention layer, so
    // `BatchStats::{cache_hits, cache_misses}` and
    // `GenerationResult::num_cached_kv` could only ever be 0.
    // ------------------------------------------------------------------

    fn test_cache() -> KVCache {
        KVCache::new(crate::kv_cache::KVCacheConfig::default())
    }

    #[test]
    fn test_forward_cached_matches_uncached_output() -> Result<()> {
        let vs = test_varbuilder();
        let (query_dim, context_dim) = (8usize, 16usize);
        let attn = CrossAttention::new(vs.pp("attn"), query_dim, Some(context_dim), 2, 4)?;

        let xs = Tensor::randn(0f32, 1f32, (2usize, 4usize, query_dim), &Device::Cpu)?;
        let context = Tensor::randn(0f32, 1f32, (2usize, 3usize, context_dim), &Device::Cpu)?;

        let uncached = attn.forward(&xs, Some(&context))?;
        let cache = test_cache();

        // First call misses and populates; the second must hit and still agree.
        for round in 0..2 {
            let cached = attn.forward_cached(&xs, &context, &cache, "layer=0:cond=1")?;
            let diff = (&cached - &uncached)?
                .abs()?
                .sum_all()?
                .to_scalar::<f32>()?;
            assert!(
                diff < 1e-5,
                "round {round}: cached output must match uncached, diff {diff}"
            );
        }

        let stats = cache.stats();
        assert_eq!(stats.misses, 1, "the first lookup must miss");
        assert!(stats.hits >= 1, "the second lookup must hit");
        Ok(())
    }

    #[test]
    fn test_forward_cached_changes_with_a_different_key() -> Result<()> {
        // Two different contexts under distinct keys must not cross-serve.
        let vs = test_varbuilder();
        let (query_dim, context_dim) = (8usize, 16usize);
        let attn = CrossAttention::new(vs.pp("attn"), query_dim, Some(context_dim), 2, 4)?;
        let cache = test_cache();

        let xs = Tensor::randn(0f32, 1f32, (1usize, 4usize, query_dim), &Device::Cpu)?;
        let ctx_a = Tensor::ones((1usize, 3usize, context_dim), DType::F32, &Device::Cpu)?;
        let ctx_b = Tensor::full(-2f32, (1usize, 3usize, context_dim), &Device::Cpu)?;

        let out_a = attn.forward_cached(&xs, &ctx_a, &cache, "layer=0:cond=aaa")?;
        let out_b = attn.forward_cached(&xs, &ctx_b, &cache, "layer=0:cond=bbb")?;
        let plain_b = attn.forward(&xs, Some(&ctx_b))?;

        let diff = (&out_b - &plain_b)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "distinct keys must not cross-serve: {diff}");
        let separation = (&out_a - &out_b)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(separation > 1e-5, "the two contexts must differ");
        Ok(())
    }

    #[test]
    fn test_forward_cached_ignores_a_shape_stale_entry() -> Result<()> {
        // A cached entry recorded at batch 1 must not be replayed for batch 2.
        let vs = test_varbuilder();
        let (query_dim, context_dim) = (8usize, 16usize);
        let attn = CrossAttention::new(vs.pp("attn"), query_dim, Some(context_dim), 2, 4)?;
        let cache = test_cache();
        let key = "layer=0:cond=stale";

        let xs1 = Tensor::randn(0f32, 1f32, (1usize, 4usize, query_dim), &Device::Cpu)?;
        let ctx1 = Tensor::randn(0f32, 1f32, (1usize, 3usize, context_dim), &Device::Cpu)?;
        let _ = attn.forward_cached(&xs1, &ctx1, &cache, key)?;

        let xs2 = Tensor::randn(0f32, 1f32, (2usize, 4usize, query_dim), &Device::Cpu)?;
        let ctx2 = Tensor::randn(0f32, 1f32, (2usize, 3usize, context_dim), &Device::Cpu)?;
        let cached = attn.forward_cached(&xs2, &ctx2, &cache, key)?;
        let plain = attn.forward(&xs2, Some(&ctx2))?;
        let diff = (&cached - &plain)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "a shape-stale entry must be ignored: {diff}");
        Ok(())
    }

    #[test]
    fn test_transformer_block_cache_matches_uncached_and_records_hits() -> Result<()> {
        let vs = test_varbuilder();
        let dim = 8;
        let (context_dim, ip_dim) = (16usize, 12usize);
        let mut block =
            MultiViewTransformerBlock::new(vs.pp("b"), dim, 2, 4, context_dim, ip_dim, 2)?;

        let xs = Tensor::randn(0f32, 1f32, (2usize, 4usize, dim), &Device::Cpu)?;
        let context = Tensor::randn(0f32, 1f32, (2usize, 3usize, context_dim), &Device::Cpu)?;
        let ip = Tensor::randn(0f32, 1f32, (2usize, 5usize, ip_dim), &Device::Cpu)?;

        assert!(block.ip_cache_key().is_none(), "no cache attached yet");
        let uncached = block.forward(&xs, Some(&context), Some(&ip))?;

        let cache = Arc::new(test_cache());
        block.set_kv_cache(Some(Arc::clone(&cache)), 3, 0, 0xabc);
        assert_eq!(block.ip_cache_key(), Some("layer=3:hg=0:cond=2748"));

        for step in 0..3 {
            let cached = block.forward(&xs, Some(&context), Some(&ip))?;
            let diff = (&cached - &uncached)?
                .abs()?
                .sum_all()?
                .to_scalar::<f32>()?;
            assert!(
                diff < 1e-5,
                "step {step}: cached block output drifted: {diff}"
            );
        }
        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 2, "steps 2 and 3 must hit");

        // Detaching restores the uncached path.
        block.set_kv_cache(None, 3, 0, 0xabc);
        assert!(block.ip_cache_key().is_none());
        Ok(())
    }

    #[test]
    fn test_transformer_block_cache_is_untouched_without_ip_tokens() -> Result<()> {
        // The CFG unconditional pass skips attn_ip entirely, so it must not
        // populate or consult the cache.
        let vs = test_varbuilder();
        let dim = 8;
        let mut block = MultiViewTransformerBlock::new(vs.pp("b"), dim, 2, 4, 16, 12, 2)?;
        let cache = Arc::new(test_cache());
        block.set_kv_cache(Some(Arc::clone(&cache)), 0, 0, 1);

        let xs = Tensor::randn(0f32, 1f32, (2usize, 4usize, dim), &Device::Cpu)?;
        let context = Tensor::randn(0f32, 1f32, (2usize, 3usize, 16usize), &Device::Cpu)?;
        let _ = block.forward(&xs, Some(&context), None)?;

        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert!(cache.is_empty());
        Ok(())
    }

    #[test]
    fn test_spatial_transformer_keys_each_block_separately() -> Result<()> {
        let vs = test_varbuilder();
        let mut transformer =
            MultiViewSpatialTransformer::with_spec(vs.pp("t"), &test_spec(8, 3, 16, 12, true))?;
        assert_eq!(transformer.depth(), 3);

        let cache = Arc::new(test_cache());
        transformer.set_kv_cache(Some(Arc::clone(&cache)), 7, 99);
        let keys: Vec<String> = transformer
            .transformer_blocks
            .iter()
            .filter_map(|b| b.ip_cache_key().map(str::to_string))
            .collect();
        assert_eq!(
            keys,
            vec![
                "layer=7:hg=0:cond=99".to_string(),
                "layer=7:hg=1:cond=99".to_string(),
                "layer=7:hg=2:cond=99".to_string(),
            ]
        );
        Ok(())
    }

    // ------------------------------------------------------------------
    // Attention backend selection
    //
    // Regression: SlicedAttention was reachable only from its own module's
    // tests, because nothing built a CrossAttention that used it.
    // ------------------------------------------------------------------

    #[test]
    fn test_sliced_backend_matches_standard_numerically() -> Result<()> {
        // Same weights, two kernels: the results must agree.
        let varmap = VarMap::new();
        let (query_dim, context_dim) = (8usize, 16usize);
        let spec = AttentionSpec {
            heads: 2,
            dim_head: 4,
            backend: AttentionBackend::Standard,
            flash_block_size: 64,
            slice_size: Some(3),
        };

        let build = |backend: AttentionBackend| -> Result<CrossAttention> {
            let vs = nn::VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
            CrossAttention::with_spec(
                vs.pp("attn"),
                query_dim,
                Some(context_dim),
                &spec.with_backend(backend),
            )
        };

        let standard = build(AttentionBackend::Standard)?;
        let sliced = build(AttentionBackend::Sliced)?;
        assert_eq!(standard.backend(), AttentionBackend::Standard);
        assert_eq!(sliced.backend(), AttentionBackend::Sliced);

        // seq_len 8 with slice_size 3 exercises a ragged final slice.
        let xs = Tensor::randn(0f32, 1f32, (2usize, 8usize, query_dim), &Device::Cpu)?;
        let context = Tensor::randn(0f32, 1f32, (2usize, 5usize, context_dim), &Device::Cpu)?;

        let a = standard.forward(&xs, Some(&context))?;
        let b = sliced.forward(&xs, Some(&context))?;
        assert_eq!(a.dims3()?, b.dims3()?);
        let diff = (&a - &b)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-4, "sliced kernel diverged from standard: {diff}");
        Ok(())
    }

    #[test]
    fn test_sliced_backend_falls_back_for_a_masked_call() -> Result<()> {
        // The sliced kernel has no additive-bias support, so a masked call must
        // route through standard_attention and still honour the mask.
        let vs = test_varbuilder();
        let query_dim = 8usize;
        let attn = CrossAttention::with_spec(
            vs.pp("attn"),
            query_dim,
            None,
            &AttentionSpec {
                heads: 2,
                dim_head: 4,
                backend: AttentionBackend::Sliced,
                flash_block_size: 64,
                slice_size: Some(2),
            },
        )?;

        let seq_len = 4;
        let xs = Tensor::randn(0f32, 1f32, (1usize, seq_len, query_dim), &Device::Cpu)?;
        let unmasked = attn.forward_masked(&xs, None, None)?;

        let mut mask = AttentionMask::new(seq_len, true);
        for k in 1..seq_len {
            mask.set(0, k, false);
        }
        let bias = mask_to_bias_tensor(&mask, &Device::Cpu)?;
        let masked = attn.forward_masked(&xs, None, Some(&bias))?;

        let diff = (&masked - &unmasked)?
            .abs()?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(diff > 1e-5, "the mask must still take effect: {diff}");
        Ok(())
    }

    #[test]
    fn test_spec_from_config_carries_the_resolved_backend() {
        let sliced = DiffusionConfig {
            attention_backend: AttentionBackend::Sliced,
            attention_slice_size: Some(16),
            ..DiffusionConfig::default()
        };
        let spec = AttentionSpec::from_config(&sliced, 5, 64);
        assert_eq!(spec.backend, AttentionBackend::Sliced);
        assert_eq!(spec.slice_size, Some(16));
        assert_eq!(spec.heads, 5);
        assert_eq!(spec.dim_head, 64);

        // The legacy flag still routes to Flash through resolved_*().
        let legacy = DiffusionConfig {
            use_flash_attention: true,
            ..DiffusionConfig::default()
        };
        assert_eq!(
            AttentionSpec::from_config(&legacy, 5, 64).backend,
            AttentionBackend::Flash
        );
    }

    #[test]
    fn test_block_keeps_cross_view_attention_standard() -> Result<()> {
        // attn_cv's sequence length is num_views; a memory-optimising kernel
        // there is pure overhead, so the spec must be downgraded for it.
        let vs = test_varbuilder();
        let block = MultiViewTransformerBlock::with_spec(
            vs.pp("b"),
            8,
            &AttentionSpec {
                heads: 2,
                dim_head: 4,
                backend: AttentionBackend::Sliced,
                flash_block_size: 64,
                slice_size: Some(2),
            },
            16,
            12,
            2,
        )?;
        assert_eq!(block.attn1.backend(), AttentionBackend::Sliced);
        assert_eq!(block.attn2.backend(), AttentionBackend::Sliced);
        assert_eq!(block.attn_ip.backend(), AttentionBackend::Sliced);
        assert_eq!(block.attn_cv.backend(), AttentionBackend::Standard);
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
