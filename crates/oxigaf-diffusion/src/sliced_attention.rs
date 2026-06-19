//! Sliced (chunked) attention for memory-efficient multi-head attention on limited VRAM.
//!
//! This module implements pure-CPU multi-head attention with configurable query chunking.
//! By processing queries in chunks (slices), peak memory usage is reduced from
//! `O(batch * heads * seq_q * seq_k)` to `O(batch * heads * slice_size * seq_k)`.
//!
//! ## Algorithm
//!
//! Standard attention: `Softmax(QK^T / sqrt(d)) * V`
//!
//! Chunked version processes Q in chunks of `slice_size` tokens:
//! ```text
//! For each chunk of queries Q_i (rows i..i+slice_size of Q):
//!     S_i  = Q_i @ K^T / sqrt(d_k)        # [slice_size, seq_len_k]
//!     m_i  = max(S_i, axis=-1)             # running max for numerical stability
//!     e_i  = exp(S_i - m_i)               # [slice_size, seq_len_k]
//!     s_i  = e_i.sum(axis=-1)             # [slice_size]
//!     O_i  = (e_i @ V) / s_i              # [slice_size, d_v]
//! ```
//!
//! ## Flat layout
//!
//! Input tensors follow the layout `[batch, num_heads, seq_len, head_dim]`:
//! index `[b, h, s, d]` → `b*num_heads*seq_len*head_dim + h*seq_len*head_dim + s*head_dim + d`

use crate::numerics::AttentionPrecision;
use crate::DiffusionError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for attention slicing (chunked attention).
///
/// Sliced attention reduces peak memory by processing attention in chunks of
/// `slice_size` query tokens rather than the full sequence at once.
///
/// ## Precision Control
///
/// The `attention_precision` field selects the upcasting strategy for
/// softmax and attention weights. The default is
/// [`AttentionPrecision::UpcastedSoftmax`], which promotes only the softmax
/// step to FP32 (using the log-sum-exp stable kernel in [`crate::numerics`]).
/// Since `SlicedAttention` operates exclusively on `f32` slices, this field
/// is currently used for documentation and future mixed-precision integration
/// — the internal `softmax_and_weighted_v` helper is already numerically
/// stable.
#[derive(Debug, Clone)]
pub struct SlicedAttentionConfig {
    /// Number of query tokens to process per chunk.
    ///
    /// - `None` — no slicing; full attention matrix is computed.
    /// - `Some(s)` — process queries s tokens at a time. Smaller values use less
    ///   memory but require more sequential computation passes.
    pub slice_size: Option<usize>,

    /// Number of attention heads.
    pub num_heads: usize,

    /// Dimension per head (must equal total_dim / num_heads).
    pub head_dim: usize,

    /// Precision mode for the softmax step.
    ///
    /// Defaults to [`AttentionPrecision::UpcastedSoftmax`]. When mixed-precision
    /// inference (FP16/BF16) is enabled in a future release, this field controls
    /// whether Q, K, V are promoted before the attention kernel, or only the
    /// softmax sub-step is executed in FP32.
    pub attention_precision: AttentionPrecision,
}

impl Default for SlicedAttentionConfig {
    fn default() -> Self {
        Self {
            slice_size: None,
            num_heads: 8,
            head_dim: 64,
            attention_precision: AttentionPrecision::default(),
        }
    }
}

impl SlicedAttentionConfig {
    /// Create a new config with explicit parameters.
    ///
    /// The `attention_precision` field defaults to
    /// [`AttentionPrecision::UpcastedSoftmax`]. Use
    /// [`SlicedAttentionConfig::with_attention_precision`] to change it.
    pub fn new(slice_size: Option<usize>, num_heads: usize, head_dim: usize) -> Self {
        Self {
            slice_size,
            num_heads,
            head_dim,
            attention_precision: AttentionPrecision::default(),
        }
    }

    /// Set the attention precision mode, consuming and returning `self` (builder pattern).
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxigaf_diffusion::{SlicedAttentionConfig, AttentionPrecision};
    ///
    /// let cfg = SlicedAttentionConfig::new(Some(4), 8, 64)
    ///     .with_attention_precision(AttentionPrecision::FullUpcast);
    /// assert_eq!(cfg.attention_precision, AttentionPrecision::FullUpcast);
    /// ```
    pub fn with_attention_precision(mut self, precision: AttentionPrecision) -> Self {
        self.attention_precision = precision;
        self
    }

    /// Validate the configuration, returning an error on illegal values.
    pub fn validate(&self) -> Result<(), DiffusionError> {
        if self.num_heads == 0 {
            return Err(DiffusionError::InvalidConfig(
                "num_heads must be > 0".into(),
            ));
        }
        if self.head_dim == 0 {
            return Err(DiffusionError::InvalidConfig("head_dim must be > 0".into()));
        }
        if let Some(s) = self.slice_size {
            if s == 0 {
                return Err(DiffusionError::InvalidConfig(
                    "slice_size must be > 0 when Some".into(),
                ));
            }
        }
        Ok(())
    }

    /// Estimate memory (bytes) for the standard (non-sliced) attention matrix.
    ///
    /// This is the peak memory for storing the full `QK^T` matrix across all
    /// heads: `seq_len_q * seq_len_k * num_heads * 4` bytes (f32).
    pub fn memory_bytes_standard(&self, seq_len_q: usize, seq_len_k: usize) -> usize {
        seq_len_q * seq_len_k * self.num_heads * 4
    }

    /// Estimate memory (bytes) for the sliced attention matrix.
    ///
    /// Only `slice_size` (or 1 if `None`) rows of the attention matrix are
    /// materialised at a time: `chunk * seq_len_k * num_heads * 4` bytes.
    pub fn memory_bytes_sliced(&self, seq_len_k: usize) -> usize {
        let chunk = self.slice_size.unwrap_or(1);
        chunk * seq_len_k * self.num_heads * 4
    }
}

// ---------------------------------------------------------------------------
// SlicedAttention
// ---------------------------------------------------------------------------

/// Pure-CPU multi-head attention with optional query-slicing.
///
/// This struct wraps `SlicedAttentionConfig` and provides a single `forward`
/// method that operates on flat f32 slices in `[batch, heads, seq, dim]` layout.
pub struct SlicedAttention {
    /// Configuration controlling slicing behaviour.
    pub config: SlicedAttentionConfig,
}

impl SlicedAttention {
    /// Create a new `SlicedAttention` module.
    ///
    /// Returns an error if the configuration is invalid (e.g., zero heads).
    pub fn new(config: SlicedAttentionConfig) -> Result<Self, DiffusionError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Compute multi-head attention, optionally in query chunks.
    ///
    /// # Tensor layout
    ///
    /// All tensors are flat f32 slices with layout `[batch, num_heads, seq_len, head_dim]`.
    ///
    /// # Arguments
    ///
    /// * `q` — query:  `batch * num_heads * seq_len_q * head_dim` elements
    /// * `k` — key:    `batch * num_heads * seq_len_k * head_dim` elements
    /// * `v` — value:  `batch * num_heads * seq_len_k * head_dim` elements
    /// * `batch`    — number of samples in the batch
    /// * `seq_len_q` — query sequence length
    /// * `seq_len_k` — key / value sequence length
    ///
    /// # Returns
    ///
    /// Output flat f32 vector with layout `[batch, num_heads, seq_len_q, head_dim]`.
    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        seq_len_q: usize,
        seq_len_k: usize,
    ) -> Result<Vec<f32>, DiffusionError> {
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // Validate input sizes.
        let q_expected = batch * num_heads * seq_len_q * head_dim;
        let k_expected = batch * num_heads * seq_len_k * head_dim;

        if q.len() != q_expected {
            return Err(DiffusionError::ShapeMismatch {
                op: "SlicedAttention::forward q".into(),
                expected: vec![batch, num_heads, seq_len_q, head_dim],
                got: vec![q.len()],
            });
        }
        if k.len() != k_expected {
            return Err(DiffusionError::ShapeMismatch {
                op: "SlicedAttention::forward k".into(),
                expected: vec![batch, num_heads, seq_len_k, head_dim],
                got: vec![k.len()],
            });
        }
        if v.len() != k_expected {
            return Err(DiffusionError::ShapeMismatch {
                op: "SlicedAttention::forward v".into(),
                expected: vec![batch, num_heads, seq_len_k, head_dim],
                got: vec![v.len()],
            });
        }

        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let out_len = batch * num_heads * seq_len_q * head_dim;
        let mut output = vec![0.0_f32; out_len];

        // Stride constants for [b, h, s, d] layout.
        let stride_b_q = num_heads * seq_len_q * head_dim;
        let stride_h_q = seq_len_q * head_dim;
        let stride_b_kv = num_heads * seq_len_k * head_dim;
        let stride_h_kv = seq_len_k * head_dim;

        // Effective chunk size: if slice_size >= seq_len_q (or None), process all
        // at once (same as no slicing). We always proceed with the chunked loop;
        // the chunk naturally equals seq_len_q when slice_size is large enough.
        let chunk_size = match self.config.slice_size {
            None => seq_len_q,
            Some(s) => s.min(seq_len_q).max(1),
        };

        for b in 0..batch {
            for h in 0..num_heads {
                // Offsets for this (batch, head) tile.
                let q_base = b * stride_b_q + h * stride_h_q;
                let kv_base = b * stride_b_kv + h * stride_h_kv;
                let out_base = b * stride_b_q + h * stride_h_q;

                // Extract K tile: [seq_len_k, head_dim]
                let k_tile = &k[kv_base..kv_base + stride_h_kv];
                // Extract V tile: [seq_len_k, head_dim]
                let v_tile = &v[kv_base..kv_base + stride_h_kv];

                // Process Q in chunks.
                let mut qi = 0usize;
                while qi < seq_len_q {
                    let chunk_end = (qi + chunk_size).min(seq_len_q);
                    let chunk_len = chunk_end - qi;

                    // Q chunk: [chunk_len, head_dim]
                    let q_chunk = &q[q_base + qi * head_dim..q_base + chunk_end * head_dim];

                    // Compute scores = Q_chunk @ K^T / sqrt(d): [chunk_len, seq_len_k]
                    let scores =
                        compute_qkt(q_chunk, k_tile, chunk_len, seq_len_k, head_dim, scale);

                    // Numerically stable softmax weights and apply to V.
                    let out_chunk =
                        softmax_and_weighted_v(&scores, v_tile, chunk_len, seq_len_k, head_dim);

                    // Write output chunk.
                    let out_slice =
                        &mut output[out_base + qi * head_dim..out_base + chunk_end * head_dim];
                    out_slice.copy_from_slice(&out_chunk);

                    qi += chunk_size;
                }
            }
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute `Q_chunk @ K^T * scale`.
///
/// `q` has shape `[chunk_len, head_dim]` (row-major),
/// `k` has shape `[seq_len_k, head_dim]` (row-major).
/// Returns `scores` with shape `[chunk_len, seq_len_k]`.
fn compute_qkt(
    q: &[f32],
    k: &[f32],
    chunk_len: usize,
    seq_len_k: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut scores = vec![0.0_f32; chunk_len * seq_len_k];
    for qi in 0..chunk_len {
        let q_row = &q[qi * head_dim..(qi + 1) * head_dim];
        for ki in 0..seq_len_k {
            let k_row = &k[ki * head_dim..(ki + 1) * head_dim];
            // Dot product Q[qi] · K[ki]
            let dot: f32 = q_row.iter().zip(k_row.iter()).map(|(a, b)| a * b).sum();
            scores[qi * seq_len_k + ki] = dot * scale;
        }
    }
    scores
}

/// Apply row-wise numerically stable softmax to `scores` and compute weighted
/// sum of `V`.
///
/// `scores` shape: `[chunk_len, seq_len_k]`
/// `v_tile` shape: `[seq_len_k, head_dim]`
///
/// Returns output with shape `[chunk_len, head_dim]`.
fn softmax_and_weighted_v(
    scores: &[f32],
    v_tile: &[f32],
    chunk_len: usize,
    seq_len_k: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; chunk_len * head_dim];

    for qi in 0..chunk_len {
        let score_row = &scores[qi * seq_len_k..(qi + 1) * seq_len_k];

        // Row max for numerical stability.
        let max_s = score_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(score - max) and their sum.
        let mut exp_row = vec![0.0_f32; seq_len_k];
        let mut sum_exp = 0.0_f32;
        for (ki, s) in score_row.iter().enumerate() {
            let e = (s - max_s).exp();
            exp_row[ki] = e;
            sum_exp += e;
        }

        // Weighted sum of V rows: out[qi] = sum_k (exp_row[k] / sum_exp) * V[k]
        let out_row = &mut out[qi * head_dim..(qi + 1) * head_dim];
        for ki in 0..seq_len_k {
            let weight = exp_row[ki] / sum_exp;
            let v_row = &v_tile[ki * head_dim..(ki + 1) * head_dim];
            for d in 0..head_dim {
                out_row[d] += weight * v_row[d];
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// Build deterministic Q/K/V tensors (flat, layout [1,1,seq,dim]).
    fn make_tensors(
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_k: usize,
        head_dim: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_len = batch * heads * seq_q * head_dim;
        let k_len = batch * heads * seq_k * head_dim;
        let q: Vec<f32> = (0..q_len).map(|i| (i as f32 * 0.01).sin()).collect();
        let k: Vec<f32> = (0..k_len).map(|i| (i as f32 * 0.02).cos()).collect();
        let v: Vec<f32> = (0..k_len).map(|i| (i as f32 * 0.03).sin()).collect();
        (q, k, v)
    }

    /// Run forward with given slice_size and return output.
    #[allow(clippy::too_many_arguments)]
    fn run_forward(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: usize,
        num_heads: usize,
        seq_q: usize,
        seq_k: usize,
        head_dim: usize,
        slice_size: Option<usize>,
    ) -> Result<Vec<f32>, DiffusionError> {
        let cfg = SlicedAttentionConfig::new(slice_size, num_heads, head_dim);
        let attn = SlicedAttention::new(cfg)?;
        attn.forward(q, k, v, batch, seq_q, seq_k)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    // ------------------------------------------------------------------
    // Configuration tests
    // ------------------------------------------------------------------

    #[test]
    fn test_validate_rejects_zero_heads() {
        let cfg = SlicedAttentionConfig::new(None, 0, 64);
        assert!(cfg.validate().is_err(), "zero num_heads should fail");
    }

    #[test]
    fn test_validate_rejects_zero_head_dim() {
        let cfg = SlicedAttentionConfig::new(None, 8, 0);
        assert!(cfg.validate().is_err(), "zero head_dim should fail");
    }

    #[test]
    fn test_validate_rejects_zero_slice_size() {
        let cfg = SlicedAttentionConfig::new(Some(0), 8, 64);
        assert!(cfg.validate().is_err(), "slice_size=Some(0) should fail");
    }

    #[test]
    fn test_validate_accepts_valid_config() -> Result<(), DiffusionError> {
        let cfg = SlicedAttentionConfig::default();
        cfg.validate()
    }

    #[test]
    fn test_memory_bytes_standard() {
        let cfg = SlicedAttentionConfig::new(None, 4, 64);
        let mem = cfg.memory_bytes_standard(128, 256);
        assert_eq!(mem, 128 * 256 * 4 * 4); // seq_q * seq_k * heads * 4 bytes
    }

    #[test]
    fn test_memory_bytes_sliced_none() {
        // When slice_size is None, uses chunk=1
        let cfg = SlicedAttentionConfig::new(None, 4, 64);
        let mem = cfg.memory_bytes_sliced(256);
        assert_eq!(mem, 256 * 4 * 4);
    }

    #[test]
    fn test_memory_bytes_sliced_less_than_standard() {
        let cfg = SlicedAttentionConfig::new(Some(8), 4, 64);
        let seq_q = 1024;
        let seq_k = 1024;
        let std_mem = cfg.memory_bytes_standard(seq_q, seq_k);
        let sliced_mem = cfg.memory_bytes_sliced(seq_k);
        assert!(
            sliced_mem < std_mem,
            "sliced ({sliced_mem}) should be less than standard ({std_mem})"
        );
    }

    // ------------------------------------------------------------------
    // Output shape tests
    // ------------------------------------------------------------------

    #[test]
    fn test_output_shape_single_token_q() -> Result<(), DiffusionError> {
        let (batch, heads, seq_q, seq_k, head_dim) = (1, 2, 1, 8, 16);
        let (q, k, v) = make_tensors(batch, heads, seq_q, seq_k, head_dim);
        let out = run_forward(&q, &k, &v, batch, heads, seq_q, seq_k, head_dim, None)?;
        assert_eq!(out.len(), batch * heads * seq_q * head_dim);
        Ok(())
    }

    #[test]
    fn test_output_shape_batch2() -> Result<(), DiffusionError> {
        let (batch, heads, seq_q, seq_k, head_dim) = (2, 4, 6, 8, 16);
        let (q, k, v) = make_tensors(batch, heads, seq_q, seq_k, head_dim);
        let out = run_forward(&q, &k, &v, batch, heads, seq_q, seq_k, head_dim, Some(2))?;
        assert_eq!(out.len(), batch * heads * seq_q * head_dim);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Numerical equivalence tests
    // ------------------------------------------------------------------

    #[test]
    fn test_no_slicing_equals_full_seq_slice() -> Result<(), DiffusionError> {
        let (batch, heads, seq_q, seq_k, head_dim) = (1, 2, 4, 4, 8);
        let (q, k, v) = make_tensors(batch, heads, seq_q, seq_k, head_dim);

        let out_none = run_forward(&q, &k, &v, batch, heads, seq_q, seq_k, head_dim, None)?;
        let out_full = run_forward(
            &q,
            &k,
            &v,
            batch,
            heads,
            seq_q,
            seq_k,
            head_dim,
            Some(seq_q),
        )?;
        let diff = max_abs_diff(&out_none, &out_full);
        assert!(diff < 1e-5, "None vs Some(seq_q) diff={diff}");
        Ok(())
    }

    #[test]
    fn test_slice_size_1_equals_no_slicing() -> Result<(), DiffusionError> {
        let (batch, heads, seq_q, seq_k, head_dim) = (1, 2, 6, 8, 16);
        let (q, k, v) = make_tensors(batch, heads, seq_q, seq_k, head_dim);

        let out_none = run_forward(&q, &k, &v, batch, heads, seq_q, seq_k, head_dim, None)?;
        let out_s1 = run_forward(&q, &k, &v, batch, heads, seq_q, seq_k, head_dim, Some(1))?;
        let diff = max_abs_diff(&out_none, &out_s1);
        assert!(diff < 1e-5, "None vs slice_size=1 diff={diff}");
        Ok(())
    }

    #[test]
    fn test_slice_size_2_equals_no_slicing_on_4_tokens() -> Result<(), DiffusionError> {
        let (batch, heads, seq_q, seq_k, head_dim) = (1, 2, 4, 4, 8);
        let (q, k, v) = make_tensors(batch, heads, seq_q, seq_k, head_dim);

        let out_none = run_forward(&q, &k, &v, batch, heads, seq_q, seq_k, head_dim, None)?;
        let out_s2 = run_forward(&q, &k, &v, batch, heads, seq_q, seq_k, head_dim, Some(2))?;
        let diff = max_abs_diff(&out_none, &out_s2);
        assert!(diff < 1e-5, "None vs slice_size=2 diff={diff}");
        Ok(())
    }

    #[test]
    fn test_slice_size_larger_than_seq_q_is_graceful() -> Result<(), DiffusionError> {
        // slice_size > seq_len_q should work correctly (clamped internally)
        let (batch, heads, seq_q, seq_k, head_dim) = (1, 1, 3, 5, 8);
        let (q, k, v) = make_tensors(batch, heads, seq_q, seq_k, head_dim);

        let out_none = run_forward(&q, &k, &v, batch, heads, seq_q, seq_k, head_dim, None)?;
        let out_large = run_forward(&q, &k, &v, batch, heads, seq_q, seq_k, head_dim, Some(1000))?;
        let diff = max_abs_diff(&out_none, &out_large);
        assert!(diff < 1e-5, "None vs large slice_size diff={diff}");
        Ok(())
    }

    // ------------------------------------------------------------------
    // Numerical property tests
    // ------------------------------------------------------------------

    #[test]
    fn test_output_is_finite() -> Result<(), DiffusionError> {
        let (batch, heads, seq_q, seq_k, head_dim) = (1, 1, 8, 8, 4);
        let (q, k, v) = make_tensors(batch, heads, seq_q, seq_k, head_dim);
        let out = run_forward(&q, &k, &v, batch, heads, seq_q, seq_k, head_dim, Some(3))?;
        for (i, &x) in out.iter().enumerate() {
            assert!(x.is_finite(), "output[{i}] = {x} is not finite");
        }
        Ok(())
    }

    #[test]
    fn test_uniform_kv_equals_v_mean_pooling() -> Result<(), DiffusionError> {
        // When Q is uniform (all rows identical) and K is uniform, every query
        // attends uniformly to all keys, so output = mean of V rows.
        let (batch, heads, seq_q, seq_k, head_dim) = (1, 1, 4, 4, 4);
        // Build uniform Q (all same value), K (all same), V (non-trivial)
        let q_val = 0.1_f32;
        let k_val = 0.2_f32;
        let q = vec![q_val; batch * heads * seq_q * head_dim];
        let k = vec![k_val; batch * heads * seq_k * head_dim];
        // V: each key position has a distinct row
        let v: Vec<f32> = (0..seq_k)
            .flat_map(|ki| (0..head_dim).map(move |d| ki as f32 + d as f32 * 0.1))
            .collect();

        let out = run_forward(&q, &k, &v, batch, heads, seq_q, seq_k, head_dim, None)?;

        // Expected: mean of V rows
        let v_mean: Vec<f32> = (0..head_dim)
            .map(|d| (0..seq_k).map(|ki| v[ki * head_dim + d]).sum::<f32>() / seq_k as f32)
            .collect();

        // Every query output should equal v_mean (within floating-point tolerance)
        for qi in 0..seq_q {
            for d in 0..head_dim {
                let got = out[qi * head_dim + d];
                let exp = v_mean[d];
                assert!(
                    (got - exp).abs() < 1e-4,
                    "qi={qi} d={d}: got={got} expected={exp}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_identity_kv_output_proportional_to_q() -> Result<(), DiffusionError> {
        // K = V = I_{seq_k × head_dim} — in the non-square case K and V are identity-padded.
        // With seq_k = head_dim (square), softmax(Q·I^T/s)·I = softmax(Q/s)·I.
        // The output is the softmax of each Q row applied as weights over identity rows.
        // We verify the output is finite and shape-correct; exact value verification
        // is done through the equivalence tests above.
        let (batch, heads, seq_q, seq_k, head_dim) = (1, 1, 4, 4, 4);
        let q: Vec<f32> = (0..batch * heads * seq_q * head_dim)
            .map(|i| i as f32 * 0.05)
            .collect();
        // K = identity (seq_k × head_dim, seq_k == head_dim here)
        let mut k = vec![0.0_f32; batch * heads * seq_k * head_dim];
        for i in 0..seq_k {
            k[i * head_dim + i] = 1.0;
        }
        // V = identity same as K
        let v = k.clone();

        let out = run_forward(&q, &k, &v, batch, heads, seq_q, seq_k, head_dim, None)?;
        assert_eq!(out.len(), batch * heads * seq_q * head_dim);
        for (i, &x) in out.iter().enumerate() {
            assert!(x.is_finite(), "output[{i}] = {x} is not finite");
        }
        Ok(())
    }

    #[test]
    fn test_standard_attention_shape_b1h1s8k8d4() -> Result<(), DiffusionError> {
        let (batch, heads, seq_q, seq_k, head_dim) = (1, 1, 8, 8, 4);
        let (q, k, v) = make_tensors(batch, heads, seq_q, seq_k, head_dim);
        let out = run_forward(&q, &k, &v, batch, heads, seq_q, seq_k, head_dim, None)?;
        assert_eq!(out.len(), batch * heads * seq_q * head_dim);
        Ok(())
    }

    #[test]
    fn test_sliced_attention_shape_b1h1s8k8d4() -> Result<(), DiffusionError> {
        let (batch, heads, seq_q, seq_k, head_dim) = (1, 1, 8, 8, 4);
        let (q, k, v) = make_tensors(batch, heads, seq_q, seq_k, head_dim);
        let out = run_forward(&q, &k, &v, batch, heads, seq_q, seq_k, head_dim, Some(3))?;
        assert_eq!(out.len(), batch * heads * seq_q * head_dim);
        Ok(())
    }

    #[test]
    fn test_softmax_boundedness() -> Result<(), DiffusionError> {
        // The weighted sum of V (which has bounded values) should itself be bounded.
        let (batch, heads, seq_q, seq_k, head_dim) = (1, 2, 8, 16, 8);
        let (q, k, v) = make_tensors(batch, heads, seq_q, seq_k, head_dim);
        // All V values are in [-1, 1], so output must also be in [-1, 1].
        let max_v = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_v = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let out = run_forward(&q, &k, &v, batch, heads, seq_q, seq_k, head_dim, Some(4))?;
        for (i, &x) in out.iter().enumerate() {
            assert!(
                x >= min_v - 1e-4 && x <= max_v + 1e-4,
                "out[{i}]={x} outside V range [{min_v}, {max_v}]"
            );
        }
        Ok(())
    }

    #[test]
    fn test_memory_estimate_sliced_smaller_for_large_sequences() {
        let cfg = SlicedAttentionConfig::new(Some(4), 8, 64);
        let seq_q = 2048;
        let seq_k = 2048;
        let std_mem = cfg.memory_bytes_standard(seq_q, seq_k);
        let sliced_mem = cfg.memory_bytes_sliced(seq_k);
        // slice_size=4 vs seq_q=2048 → 512x reduction
        assert!(
            sliced_mem * 100 < std_mem,
            "sliced mem ({sliced_mem}) should be much less than standard ({std_mem})"
        );
    }

    #[test]
    fn test_slice_size_3_on_seq_q_8_equals_no_slicing() -> Result<(), DiffusionError> {
        // seq_q=8 doesn't divide evenly by 3, so tests boundary handling.
        let (batch, heads, seq_q, seq_k, head_dim) = (1, 2, 8, 6, 16);
        let (q, k, v) = make_tensors(batch, heads, seq_q, seq_k, head_dim);

        let out_none = run_forward(&q, &k, &v, batch, heads, seq_q, seq_k, head_dim, None)?;
        let out_s3 = run_forward(&q, &k, &v, batch, heads, seq_q, seq_k, head_dim, Some(3))?;
        let diff = max_abs_diff(&out_none, &out_s3);
        assert!(
            diff < 1e-5,
            "None vs slice_size=3 (non-divisible) diff={diff}"
        );
        Ok(())
    }

    #[test]
    fn test_new_returns_error_on_invalid_config() {
        let cfg = SlicedAttentionConfig::new(None, 0, 64);
        assert!(
            SlicedAttention::new(cfg).is_err(),
            "SlicedAttention::new should fail on invalid config"
        );
    }

    #[test]
    fn test_sliced_attention_config_attention_precision_default() {
        let cfg = SlicedAttentionConfig::default();
        assert_eq!(cfg.attention_precision, AttentionPrecision::UpcastedSoftmax);
    }

    #[test]
    fn test_sliced_attention_config_with_precision_builder() {
        let cfg = SlicedAttentionConfig::new(Some(4), 8, 64)
            .with_attention_precision(AttentionPrecision::FullUpcast);
        assert_eq!(cfg.attention_precision, AttentionPrecision::FullUpcast);
    }

    #[test]
    fn test_sliced_attention_config_standard_precision() {
        let cfg = SlicedAttentionConfig::new(None, 4, 32)
            .with_attention_precision(AttentionPrecision::Standard);
        assert_eq!(cfg.attention_precision, AttentionPrecision::Standard);
    }

    #[test]
    fn test_forward_shape_mismatch_q() {
        // Provide wrong-length Q.
        let cfg = SlicedAttentionConfig::new(None, 1, 4);
        let attn = SlicedAttention::new(cfg).expect("valid config");
        let q = vec![0.0_f32; 10]; // wrong size
        let k = vec![0.0_f32; 4 * 4];
        let v = k.clone();
        let result = attn.forward(&q, &k, &v, 1, 4, 4);
        assert!(result.is_err(), "mismatched Q length should return Err");
    }
}
