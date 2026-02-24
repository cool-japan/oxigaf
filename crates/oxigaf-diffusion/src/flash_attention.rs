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
//! # References
//!
//! - Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
//!   with IO-Awareness", NeurIPS 2022

use candle_core::{DType, Result, Tensor, D};

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
    #[allow(dead_code)]
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }
}

/// Flash Attention: Memory-efficient scaled dot-product attention.
///
/// This struct provides the flash attention computation without maintaining
/// learned parameters. It is designed to be used within attention modules
/// that handle Q, K, V projections separately.
#[derive(Debug, Clone)]
pub struct FlashAttention {
    config: FlashAttentionConfig,
    scale: f64,
}

impl FlashAttention {
    /// Create a new Flash Attention module.
    ///
    /// # Arguments
    ///
    /// * `dim_head` - Dimension of each attention head (for scaling)
    /// * `config` - Flash Attention configuration
    pub fn new(dim_head: usize, config: FlashAttentionConfig) -> Self {
        let scale = 1.0 / (dim_head as f64).sqrt();
        Self { config, scale }
    }

    /// Create with default configuration.
    pub fn with_dim_head(dim_head: usize) -> Self {
        Self::new(dim_head, FlashAttentionConfig::default())
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

        // For small sequences, use standard attention (simpler and often faster)
        let use_standard = seq_q <= self.config.block_size && seq_k <= self.config.block_size;
        if use_standard {
            return self.standard_attention(q, k, v);
        }

        // Compute in f32 for numerical stability
        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        // Block-wise computation
        let block_size = self.config.block_size;
        let num_q_blocks = seq_q.div_ceil(block_size);
        let num_k_blocks = seq_k.div_ceil(block_size);

        // Initialize output accumulator and softmax statistics
        let device = q.device();
        let neg_inf = f32::NEG_INFINITY;

        // Process each query block
        let mut output_blocks: Vec<Tensor> = Vec::with_capacity(num_q_blocks);

        for q_block_idx in 0..num_q_blocks {
            let q_start = q_block_idx * block_size;
            let q_end = (q_start + block_size).min(seq_q);
            let q_len = q_end - q_start;

            // Extract query block: (batch, heads, q_len, dim_head)
            let q_block = q.narrow(2, q_start, q_len)?;

            // Initialize accumulators for this query block
            // m: running max, shape (batch, heads, q_len)
            // l: running sum, shape (batch, heads, q_len)
            // o: running output, shape (batch, heads, q_len, dim_head)
            let mut m = Tensor::full(neg_inf, (batch, heads, q_len), device)?;
            let mut l = Tensor::zeros((batch, heads, q_len), DType::F32, device)?;
            let mut o = Tensor::zeros((batch, heads, q_len, dim_head), DType::F32, device)?;

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

                // Apply causal mask if needed
                let scores = if self.config.causal {
                    self.apply_causal_mask(&scores, q_start, k_start)?
                } else {
                    scores
                };

                // Online softmax update
                let (m_new, l_new, o_new) =
                    self.online_softmax_update(&m, &l, &o, &scores, &v_block)?;

                m = m_new;
                l = l_new;
                o = o_new;
            }

            // Finalize output for this block: o = o / l
            let l_expanded = l.unsqueeze(D::Minus1)?;
            let l_safe = (l_expanded + self.config.softmax_eps)?;
            let block_output = o.broadcast_div(&l_safe)?;

            output_blocks.push(block_output);
        }

        // Concatenate all output blocks along sequence dimension
        let output = Tensor::cat(&output_blocks, 2)?;
        output.to_dtype(in_dtype)
    }

    /// Standard attention for small sequences (fallback path).
    fn standard_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn = (q.matmul(&k_t)? * self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        out.to_dtype(in_dtype)
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

        // Rescale old statistics
        // exp(m - m_new) for old values
        let m_diff_old = m.broadcast_sub(&m_new)?;
        let rescale_old = m_diff_old.exp()?;

        // exp(m_block - m_new) for new block (computed but kept for debugging/clarity)
        // The actual rescaling is done implicitly when computing p_block below
        let m_diff_new = m_block.broadcast_sub(&m_new)?;
        let _rescale_new = m_diff_new.exp()?;

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

    /// Apply causal mask to attention scores.
    ///
    /// Sets scores to -inf where key position > query position.
    fn apply_causal_mask(&self, scores: &Tensor, q_start: usize, k_start: usize) -> Result<Tensor> {
        let (batch, heads, q_len, k_len) = scores.dims4()?;
        let device = scores.device();

        // Create causal mask: mask[i,j] = true if k_start + j > q_start + i
        let mut mask_data = vec![0.0f32; q_len * k_len];
        let neg_inf = f32::NEG_INFINITY;

        for i in 0..q_len {
            let q_pos = q_start + i;
            for j in 0..k_len {
                let k_pos = k_start + j;
                if k_pos > q_pos {
                    mask_data[i * k_len + j] = neg_inf;
                }
            }
        }

        let mask = Tensor::from_vec(mask_data, (1, 1, q_len, k_len), device)?;
        let mask = mask.broadcast_as((batch, heads, q_len, k_len))?;
        scores.add(&mask)
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

        let flash = FlashAttention::with_dim_head(dim_head);
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

        let flash = FlashAttention::with_dim_head(dim_head);
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

        let flash = FlashAttention::with_dim_head(dim_head);
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

        // Test with different block sizes
        for block_size in [32, 64, 128] {
            let config = FlashAttentionConfig::with_block_size(block_size);
            let flash = FlashAttention::new(dim_head, config);
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
