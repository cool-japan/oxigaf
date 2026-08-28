//! Flash Attention: Memory-efficient attention with block-wise computation.
//!
//! This module implements the Flash Attention algorithm, which reduces memory
//! complexity from O(N^2) to O(N) by using tiled computation with online softmax.
//!
//! # Algorithm Overview
//!
//! Instead of computing the full N x N attention matrix, Flash Attention:
//! 1. Splits Q, K, V into blocks of size `block_size`
//! 2. For each query block, iterates over all key/value blocks
//! 3. Computes local attention scores for each block pair
//! 4. Uses online softmax with running max/sum for numerical stability
//! 5. Accumulates weighted values with proper rescaling
//!
//! # Dispatch policy
//!
//! The tiled loop is expressed with individual `candle` tensor operations, each
//! of which materialises its own intermediate: it therefore only pays off when
//! the fully materialised `(batch, heads, seq_q, seq_k)` score matrix would not
//! comfortably fit in memory. [`FlashAttention`] consequently runs the plain
//! (single score matrix) kernel whenever
//!
//! * the whole sequence fits in one block, or
//! * the score matrix stays within the configured budget
//!   ([`DEFAULT_SCORE_MATRIX_BUDGET`], overridable with
//!   [`FlashAttention::with_score_matrix_budget`]).
//!
//! Both paths implement the same mathematics, causal masking included. They are
//! not bit-identical: besides the different accumulation order, the tiled path
//! normalises by `l + softmax_eps`, a systematic relative bias of about
//! `softmax_eps` (1e-6 by default).
//!
//! # References
//!
//! - Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
//!   with IO-Awareness", NeurIPS 2022

use candle_core::{DType, Device, Result, Tensor, D};
use std::collections::HashMap;

/// Configuration for Flash Attention computation.
#[derive(Debug, Clone, Copy)]
pub struct FlashAttentionConfig {
    /// Block size for tiled computation. Larger blocks use more memory but
    /// may be faster due to better cache utilization. Default: 64.
    pub block_size: usize,
    /// Whether to use causal masking (for autoregressive models). Default: false.
    pub causal: bool,
    /// Epsilon for numerical stability in softmax. Default: 1e-6.
    pub softmax_eps: f64,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            causal: false,
            softmax_eps: 1e-6,
        }
    }
}

impl FlashAttentionConfig {
    /// Create a new Flash Attention config with specified block size.
    pub fn with_block_size(block_size: usize) -> Self {
        Self {
            block_size,
            ..Default::default()
        }
    }

    /// Enable causal masking for autoregressive attention.
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }
}

/// Default budget (64 MiB) for the fully materialised attention score matrix.
///
/// Below this size the plain kernel is both faster and no less memory-friendly
/// than the tiled loop, so [`FlashAttention::forward`] uses it.
pub const DEFAULT_SCORE_MATRIX_BUDGET: usize = 64 * 1024 * 1024;

/// Flash Attention: Memory-efficient scaled dot-product attention.
///
/// This struct provides the flash attention computation without maintaining
/// learned parameters. It is designed to be used within attention modules
/// that handle Q, K, V projections separately.
#[derive(Debug, Clone)]
pub struct FlashAttention {
    config: FlashAttentionConfig,
    scale: f64,
    /// Largest score matrix (in bytes) that may be materialised in one piece.
    score_matrix_budget: usize,
}

impl FlashAttention {
    /// Create a new Flash Attention module.
    ///
    /// # Arguments
    ///
    /// * `dim_head` - Dimension of each attention head (for scaling)
    /// * `config` - Flash Attention configuration
    pub fn new(dim_head: usize, config: FlashAttentionConfig) -> Self {
        // `dim_head == 0` would make the scale non-finite; fall back to 1.0.
        let scale = if dim_head == 0 {
            1.0
        } else {
            1.0 / (dim_head as f64).sqrt()
        };
        Self {
            config,
            scale,
            score_matrix_budget: DEFAULT_SCORE_MATRIX_BUDGET,
        }
    }

    /// Create with default configuration.
    pub fn with_dim_head(dim_head: usize) -> Self {
        Self::new(dim_head, FlashAttentionConfig::default())
    }

    /// Override the score-matrix memory budget (see [`DEFAULT_SCORE_MATRIX_BUDGET`]).
    ///
    /// Sequences whose `(batch, heads, seq_q, seq_k)` f32 score matrix exceeds
    /// `bytes` are processed with the tiled kernel; everything else uses the
    /// plain one. Passing `0` therefore forces the tiled path (useful for
    /// testing and for very tight memory budgets).
    pub fn with_score_matrix_budget(mut self, bytes: usize) -> Self {
        self.score_matrix_budget = bytes;
        self
    }

    /// Current score-matrix memory budget in bytes.
    pub fn score_matrix_budget(&self) -> usize {
        self.score_matrix_budget
    }

    /// Configuration this module was built with.
    pub fn config(&self) -> &FlashAttentionConfig {
        &self.config
    }

    /// Whether the plain (non-tiled) kernel is used for the given shapes.
    fn use_standard_path(&self, batch: usize, heads: usize, seq_q: usize, seq_k: usize) -> bool {
        let block_size = self.config.block_size.max(1);
        if seq_q <= block_size && seq_k <= block_size {
            return true;
        }
        let elems = batch
            .saturating_mul(heads)
            .saturating_mul(seq_q)
            .saturating_mul(seq_k);
        let bytes = elems.saturating_mul(std::mem::size_of::<f32>());
        bytes <= self.score_matrix_budget
    }

    /// Compute flash attention.
    ///
    /// # Arguments
    ///
    /// * `q` - Query tensor of shape `(batch, heads, seq_q, dim_head)`
    /// * `k` - Key tensor of shape `(batch, heads, seq_k, dim_head)`
    /// * `v` - Value tensor of shape `(batch, heads, seq_k, dim_head)`
    ///
    /// # Returns
    ///
    /// Output tensor of shape `(batch, heads, seq_q, dim_head)`
    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (batch, heads, seq_q, dim_head) = q.dims4()?;
        let (_, _, seq_k, _) = k.dims4()?;

        // Materialising the score matrix is cheaper than the tiled loop as long
        // as it fits in the configured budget (see the module-level docs).
        if self.use_standard_path(batch, heads, seq_q, seq_k) {
            return self.standard_attention(q, k, v);
        }

        // Compute in f32 for numerical stability
        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        // Block-wise computation
        let block_size = self.config.block_size.max(1);
        let num_q_blocks = seq_q.div_ceil(block_size);
        let num_k_blocks = seq_k.div_ceil(block_size);

        let device = q.device().clone();

        // Causal masks depend only on (q_start - k_start, q_len, k_len), and the
        // block grid produces a handful of distinct combinations at most, so the
        // host-side build plus device upload is done once per distinct mask.
        let mut mask_cache: HashMap<(i64, usize, usize), Tensor> = HashMap::new();

        // Process each query block
        let mut output_blocks: Vec<Tensor> = Vec::with_capacity(num_q_blocks);

        for q_block_idx in 0..num_q_blocks {
            let q_start = q_block_idx * block_size;
            let q_end = (q_start + block_size).min(seq_q);
            let q_len = q_end - q_start;

            // Extract query block: (batch, heads, q_len, dim_head)
            let q_block = q.narrow(2, q_start, q_len)?;

            // Online softmax accumulators for this query block, created from the
            // first contributing key block:
            // m: running max, shape (batch, heads, q_len)
            // l: running sum, shape (batch, heads, q_len)
            // o: running output, shape (batch, heads, q_len, dim_head)
            let mut acc: Option<(Tensor, Tensor, Tensor)> = None;

            // Iterate over key/value blocks
            for k_block_idx in 0..num_k_blocks {
                let k_start = k_block_idx * block_size;
                let k_end = (k_start + block_size).min(seq_k);
                let k_len = k_end - k_start;

                // Check causal mask: skip if this KV block is entirely in the future
                if self.config.causal && k_start >= q_end {
                    continue;
                }

                // Extract key and value blocks
                let k_block = k.narrow(2, k_start, k_len)?;
                let v_block = v.narrow(2, k_start, k_len)?;

                // Compute attention scores for this block: (batch, heads, q_len, k_len)
                let k_t = k_block.transpose(D::Minus2, D::Minus1)?;
                let scores = (q_block.matmul(&k_t)? * self.scale)?;

                // Apply the causal mask only to the block that straddles the
                // diagonal; blocks entirely in the past are fully visible.
                let needs_mask = self.config.causal && k_start + k_len > q_start;
                let scores = if needs_mask {
                    let key = (q_start as i64 - k_start as i64, q_len, k_len);
                    let mask = match mask_cache.get(&key) {
                        Some(cached) => cached.clone(),
                        None => {
                            let built =
                                Self::causal_mask_tensor(q_start, k_start, q_len, k_len, &device)?;
                            mask_cache.insert(key, built.clone());
                            built
                        }
                    };
                    scores.broadcast_add(&mask)?
                } else {
                    scores
                };

                // Online softmax update
                acc = Some(match acc {
                    Some((m, l, o)) => self.online_softmax_update(&m, &l, &o, &scores, &v_block)?,
                    None => Self::init_softmax_state(&scores, &v_block)?,
                });
            }

            // Finalize output for this block: o = o / l
            let block_output = match acc {
                Some((_m, l, o)) => {
                    let l_expanded = l.unsqueeze(D::Minus1)?;
                    let l_safe = (l_expanded + self.config.softmax_eps)?;
                    o.broadcast_div(&l_safe)?
                }
                // No key block contributed (only reachable for degenerate
                // shapes); an all-zero row is the softmax-free limit of o / l.
                None => Tensor::zeros((batch, heads, q_len, dim_head), DType::F32, &device)?,
            };

            output_blocks.push(block_output);
        }

        // Concatenate all output blocks along sequence dimension
        let output = Tensor::cat(&output_blocks, 2)?;
        output.to_dtype(in_dtype)
    }

    /// Standard attention for small sequences (fallback path).
    ///
    /// Honours [`FlashAttentionConfig::causal`] exactly like the tiled path:
    /// the two kernels must never disagree on semantics.
    fn standard_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn = (q.matmul(&k_t)? * self.scale)?;
        let attn = if self.config.causal {
            self.apply_causal_mask(&attn, 0, 0)?
        } else {
            attn
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        out.to_dtype(in_dtype)
    }

    /// Seed the online-softmax accumulators from the first contributing block.
    ///
    /// Equivalent to [`Self::online_softmax_update`] against `m = -inf`,
    /// `l = 0`, `o = 0`, but without the three tensors that identity would
    /// allocate (and without the `-inf - -inf` NaN hazard).
    fn init_softmax_state(scores: &Tensor, v_block: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let m = scores.max(D::Minus1)?;
        let p = scores.broadcast_sub(&m.unsqueeze(D::Minus1)?)?.exp()?;
        let l = p.sum(D::Minus1)?;
        let o = p.matmul(v_block)?;
        Ok((m, l, o))
    }

    /// Online softmax update step.
    ///
    /// Given current statistics (m, l, o) and new block scores, compute updated
    /// statistics using the online softmax algorithm.
    ///
    /// # Arguments
    ///
    /// * `m` - Current running max, shape (batch, heads, q_len)
    /// * `l` - Current running sum, shape (batch, heads, q_len)
    /// * `o` - Current running output, shape (batch, heads, q_len, dim_head)
    /// * `scores` - New block scores, shape (batch, heads, q_len, k_len)
    /// * `v_block` - Value block, shape (batch, heads, k_len, dim_head)
    fn online_softmax_update(
        &self,
        m: &Tensor,
        l: &Tensor,
        o: &Tensor,
        scores: &Tensor,
        v_block: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // Compute block max: max over k dimension
        // scores: (batch, heads, q_len, k_len) -> m_block: (batch, heads, q_len)
        let m_block = scores.max(D::Minus1)?;

        // Compute new global max
        let m_new = m.maximum(&m_block)?;

        // Rescale old statistics: exp(m - m_new) for the values accumulated so
        // far. The new block needs no separate factor - its rescaling is folded
        // into p_block = exp(scores - m_new) below.
        let rescale_old = m.sub(&m_new)?.exp()?;

        // Compute softmax for current block (unnormalized)
        // p_block = exp(scores - m_new)
        let m_new_expanded = m_new.unsqueeze(D::Minus1)?;
        let p_block = scores.broadcast_sub(&m_new_expanded)?.exp()?;

        // Sum over k dimension for new block contribution to l
        let l_block = p_block.sum(D::Minus1)?;

        // Update l: l_new = l * rescale_old + l_block
        let l_new = (l.mul(&rescale_old)? + l_block)?;

        // Update o: o_new = o * rescale_old + p_block @ v_block
        let rescale_old_expanded = rescale_old.unsqueeze(D::Minus1)?;
        let o_rescaled = o.broadcast_mul(&rescale_old_expanded)?;
        let pv = p_block.matmul(v_block)?;
        let o_new = (o_rescaled + pv)?;

        Ok((m_new, l_new, o_new))
    }

    /// Build the additive causal mask for one `(q_start, k_start)` block.
    ///
    /// The result has shape `(1, 1, q_len, k_len)` and holds `-inf` wherever the
    /// key position exceeds the query position, `0` elsewhere. It is broadcast
    /// over batch and heads by the caller.
    fn causal_mask_tensor(
        q_start: usize,
        k_start: usize,
        q_len: usize,
        k_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let mut mask_data = vec![0.0f32; q_len * k_len];
        for i in 0..q_len {
            let q_pos = q_start + i;
            // Number of leading keys of this block that are visible to row `i`.
            let visible = (q_pos + 1).saturating_sub(k_start).min(k_len);
            let row = i * k_len;
            mask_data[row + visible..row + k_len].fill(f32::NEG_INFINITY);
        }

        Tensor::from_vec(mask_data, (1, 1, q_len, k_len), device)
    }

    /// Apply causal mask to attention scores.
    ///
    /// Sets scores to -inf where key position > query position.
    fn apply_causal_mask(&self, scores: &Tensor, q_start: usize, k_start: usize) -> Result<Tensor> {
        let (_batch, _heads, q_len, k_len) = scores.dims4()?;
        let mask = Self::causal_mask_tensor(q_start, k_start, q_len, k_len, scores.device())?;
        scores.broadcast_add(&mask)
    }
}

/// Compute flash attention with default settings.
///
/// This is a convenience function for one-off attention computations.
///
/// # Arguments
///
/// * `q` - Query tensor of shape `(batch, heads, seq_q, dim_head)`
/// * `k` - Key tensor of shape `(batch, heads, seq_k, dim_head)`
/// * `v` - Value tensor of shape `(batch, heads, seq_k, dim_head)`
/// * `dim_head` - Dimension of each attention head
///
/// # Returns
///
/// Output tensor of shape `(batch, heads, seq_q, dim_head)`
pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor, dim_head: usize) -> Result<Tensor> {
    let flash = FlashAttention::with_dim_head(dim_head);
    flash.forward(q, k, v)
}

/// Compute flash attention with custom configuration.
pub fn flash_attention_with_config(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    dim_head: usize,
    config: FlashAttentionConfig,
) -> Result<Tensor> {
    let flash = FlashAttention::new(dim_head, config);
    flash.forward(q, k, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use candle_core::Device;

    fn create_test_tensors(
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_k: usize,
        dim_head: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // Create deterministic test data
        let q_size = batch * heads * seq_q * dim_head;
        let k_size = batch * heads * seq_k * dim_head;

        let q_data: Vec<f32> = (0..q_size).map(|i| (i as f32 * 0.01).sin()).collect();
        let k_data: Vec<f32> = (0..k_size).map(|i| (i as f32 * 0.02).cos()).collect();
        let v_data: Vec<f32> = (0..k_size).map(|i| (i as f32 * 0.03).sin()).collect();

        let q = Tensor::from_vec(q_data, (batch, heads, seq_q, dim_head), device)?;
        let k = Tensor::from_vec(k_data, (batch, heads, seq_k, dim_head), device)?;
        let v = Tensor::from_vec(v_data, (batch, heads, seq_k, dim_head), device)?;

        Ok((q, k, v))
    }

    fn standard_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let k_t = k.transpose(D::Minus2, D::Minus1)?;
        let attn = (q.matmul(&k_t)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        attn.matmul(&v)
    }

    /// Reference causal attention: full score matrix with an explicit mask.
    fn causal_reference(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
        let (_, _, seq_q, _) = q.dims4()?;
        let (_, _, seq_k, _) = k.dims4()?;

        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn = (q.matmul(&k_t)? * scale)?;

        let mut mask_data = vec![0.0f32; seq_q * seq_k];
        for i in 0..seq_q {
            for (j, cell) in mask_data[i * seq_k..(i + 1) * seq_k].iter_mut().enumerate() {
                if j > i {
                    *cell = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(mask_data, (1, 1, seq_q, seq_k), q.device())?;

        let attn = attn.broadcast_add(&mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        attn.matmul(&v)
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        let a: Vec<f32> = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let b: Vec<f32> = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        assert_eq!(a.len(), b.len());
        Ok(a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max))
    }

    #[test]
    fn test_flash_attention_small_sequence() -> Result<()> {
        let device = Device::Cpu;
        let batch = 2;
        let heads = 4;
        let seq_len = 32;
        let dim_head = 64;

        let (q, k, v) = create_test_tensors(batch, heads, seq_len, seq_len, dim_head, &device)?;

        let flash = FlashAttention::with_dim_head(dim_head);
        let flash_out = flash.forward(&q, &k, &v)?;

        let scale = 1.0 / (dim_head as f64).sqrt();
        let std_out = standard_attention(&q, &k, &v, scale)?;

        // Compare outputs
        let flash_vec: Vec<f32> = flash_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let std_vec: Vec<f32> = std_out.flatten_all()?.to_vec1()?;

        assert_eq!(flash_vec.len(), std_vec.len());
        for (f, s) in flash_vec.iter().zip(std_vec.iter()) {
            assert_relative_eq!(f, s, epsilon = 1e-4);
        }

        Ok(())
    }

    #[test]
    fn test_flash_attention_large_sequence() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let heads = 2;
        let seq_len = 128; // Larger than default block size (64)
        let dim_head = 32;

        let (q, k, v) = create_test_tensors(batch, heads, seq_len, seq_len, dim_head, &device)?;

        // Budget 0 forces the tiled kernel for this small shape.
        let flash = FlashAttention::with_dim_head(dim_head).with_score_matrix_budget(0);
        let flash_out = flash.forward(&q, &k, &v)?;

        let scale = 1.0 / (dim_head as f64).sqrt();
        let std_out = standard_attention(&q, &k, &v, scale)?;

        // Compare outputs
        let flash_vec: Vec<f32> = flash_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let std_vec: Vec<f32> = std_out.flatten_all()?.to_vec1()?;

        assert_eq!(flash_vec.len(), std_vec.len());
        for (f, s) in flash_vec.iter().zip(std_vec.iter()) {
            assert_relative_eq!(f, s, epsilon = 1e-3);
        }

        Ok(())
    }

    #[test]
    fn test_flash_attention_asymmetric_sequences() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let heads = 2;
        let seq_q = 100;
        let seq_k = 150;
        let dim_head = 32;

        let (q, k, v) = create_test_tensors(batch, heads, seq_q, seq_k, dim_head, &device)?;

        // Budget 0 forces the tiled kernel, ragged trailing blocks included.
        let flash = FlashAttention::with_dim_head(dim_head).with_score_matrix_budget(0);
        let flash_out = flash.forward(&q, &k, &v)?;

        let scale = 1.0 / (dim_head as f64).sqrt();
        let std_out = standard_attention(&q, &k, &v, scale)?;

        // Compare outputs
        let flash_vec: Vec<f32> = flash_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let std_vec: Vec<f32> = std_out.flatten_all()?.to_vec1()?;

        assert_eq!(flash_vec.len(), std_vec.len());
        for (f, s) in flash_vec.iter().zip(std_vec.iter()) {
            assert_relative_eq!(f, s, epsilon = 1e-3);
        }

        Ok(())
    }

    #[test]
    fn test_flash_attention_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let batch = 2;
        let heads = 4;
        let seq_q = 96;
        let seq_k = 128;
        let dim_head = 64;

        let (q, k, v) = create_test_tensors(batch, heads, seq_q, seq_k, dim_head, &device)?;

        let flash = FlashAttention::with_dim_head(dim_head).with_score_matrix_budget(0);
        let out = flash.forward(&q, &k, &v)?;

        assert_eq!(out.dims(), &[batch, heads, seq_q, dim_head]);

        Ok(())
    }

    #[test]
    fn test_flash_attention_single_element() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let heads = 1;
        let seq_len = 1;
        let dim_head = 16;

        let (q, k, v) = create_test_tensors(batch, heads, seq_len, seq_len, dim_head, &device)?;

        let flash = FlashAttention::with_dim_head(dim_head);
        let flash_out = flash.forward(&q, &k, &v)?;

        // For single element, output should equal value (softmax of single element is 1)
        let flash_vec: Vec<f32> = flash_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let v_vec: Vec<f32> = v.flatten_all()?.to_vec1()?;

        for (f, vv) in flash_vec.iter().zip(v_vec.iter()) {
            assert_relative_eq!(f, vv, epsilon = 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_flash_attention_config_block_size() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let heads = 2;
        let seq_len = 200;
        let dim_head = 32;

        let (q, k, v) = create_test_tensors(batch, heads, seq_len, seq_len, dim_head, &device)?;

        // Test with different block sizes (tiled path forced for all of them)
        for block_size in [32, 64, 128] {
            let config = FlashAttentionConfig::with_block_size(block_size);
            let flash = FlashAttention::new(dim_head, config).with_score_matrix_budget(0);
            let flash_out = flash.forward(&q, &k, &v)?;

            let scale = 1.0 / (dim_head as f64).sqrt();
            let std_out = standard_attention(&q, &k, &v, scale)?;

            let flash_vec: Vec<f32> = flash_out.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let std_vec: Vec<f32> = std_out.flatten_all()?.to_vec1()?;

            for (f, s) in flash_vec.iter().zip(std_vec.iter()) {
                assert_relative_eq!(f, s, epsilon = 1e-3);
            }
        }

        Ok(())
    }

    /// Regression: the standard (non-tiled) kernel used to ignore
    /// `config.causal`, silently leaking future keys for short sequences.
    #[test]
    fn test_causal_standard_path_applies_mask() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let heads = 2;
        let seq_len = 16; // <= default block_size (64) => standard path
        let dim_head = 8;

        let (q, k, v) = create_test_tensors(batch, heads, seq_len, seq_len, dim_head, &device)?;

        let config = FlashAttentionConfig::default().with_causal(true);
        let causal_out = FlashAttention::new(dim_head, config).forward(&q, &k, &v)?;

        let scale = 1.0 / (dim_head as f64).sqrt();
        let reference = causal_reference(&q, &k, &v, scale)?;
        assert!(max_abs_diff(&causal_out, &reference)? < 1e-5);

        // ... and it must not coincide with unmasked attention.
        let non_causal = FlashAttention::with_dim_head(dim_head).forward(&q, &k, &v)?;
        let leak = max_abs_diff(&causal_out, &non_causal)?;
        assert!(
            leak > 1e-3,
            "causal config produced non-causal attention (max diff {leak})"
        );

        // The first query may only attend to the first key, i.e. output == v[0].
        let out_row: Vec<f32> = causal_out
            .narrow(2, 0, 1)?
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1()?;
        let v_row: Vec<f32> = v.narrow(2, 0, 1)?.flatten_all()?.to_vec1()?;
        for (o, expected) in out_row.iter().zip(v_row.iter()) {
            assert_relative_eq!(o, expected, epsilon = 1e-5);
        }

        Ok(())
    }

    /// The tiled kernel must agree with the masked reference, including the
    /// ragged trailing blocks and the cached diagonal masks.
    #[test]
    fn test_causal_tiled_path_matches_reference() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let heads = 2;
        let seq_q = 40;
        let seq_k = 56;
        let dim_head = 8;

        let (q, k, v) = create_test_tensors(batch, heads, seq_q, seq_k, dim_head, &device)?;

        let config = FlashAttentionConfig::with_block_size(16).with_causal(true);
        let tiled = FlashAttention::new(dim_head, config).with_score_matrix_budget(0);
        let tiled_out = tiled.forward(&q, &k, &v)?;

        let scale = 1.0 / (dim_head as f64).sqrt();
        let reference = causal_reference(&q, &k, &v, scale)?;

        assert_eq!(tiled_out.dims(), &[batch, heads, seq_q, dim_head]);
        assert!(max_abs_diff(&tiled_out, &reference)? < 1e-4);

        Ok(())
    }

    /// Both dispatch branches implement the same mathematics.
    #[test]
    fn test_score_matrix_budget_dispatch_agrees() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let heads = 2;
        let seq_len = 96;
        let dim_head = 16;

        let (q, k, v) = create_test_tensors(batch, heads, seq_len, seq_len, dim_head, &device)?;

        let config = FlashAttentionConfig::with_block_size(32);
        let standard = FlashAttention::new(dim_head, config);
        let tiled = FlashAttention::new(dim_head, config).with_score_matrix_budget(0);

        // A 96x96 score matrix is far below the default budget.
        assert_eq!(standard.score_matrix_budget(), DEFAULT_SCORE_MATRIX_BUDGET);
        assert_eq!(tiled.score_matrix_budget(), 0);
        assert!(standard.use_standard_path(batch, heads, seq_len, seq_len));
        assert!(!tiled.use_standard_path(batch, heads, seq_len, seq_len));
        assert_eq!(standard.config().block_size, 32);

        let standard_out = standard.forward(&q, &k, &v)?;
        let tiled_out = tiled.forward(&q, &k, &v)?;
        assert!(max_abs_diff(&standard_out, &tiled_out)? < 1e-4);

        Ok(())
    }

    /// A degenerate `block_size` of 0 must not panic (it is clamped to 1).
    #[test]
    fn test_zero_block_size_is_clamped() -> Result<()> {
        let device = Device::Cpu;
        let (q, k, v) = create_test_tensors(1, 1, 4, 4, 4, &device)?;

        let config = FlashAttentionConfig {
            block_size: 0,
            causal: false,
            softmax_eps: 1e-6,
        };
        let flash = FlashAttention::new(4, config).with_score_matrix_budget(0);
        let out = flash.forward(&q, &k, &v)?;
        assert_eq!(out.dims(), &[1, 1, 4, 4]);

        let scale = 1.0 / 4.0_f64.sqrt();
        let reference = standard_attention(&q, &k, &v, scale)?;
        assert!(max_abs_diff(&out, &reference)? < 1e-5);

        Ok(())
    }

    #[test]
    fn test_flash_attention_convenience_function() -> Result<()> {
        let device = Device::Cpu;
        let batch = 1;
        let heads = 2;
        let seq_len = 64;
        let dim_head = 32;

        let (q, k, v) = create_test_tensors(batch, heads, seq_len, seq_len, dim_head, &device)?;

        let out = flash_attention(&q, &k, &v, dim_head)?;
        assert_eq!(out.dims(), &[batch, heads, seq_len, dim_head]);

        Ok(())
    }
}
