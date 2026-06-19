//! Fused QKV attention for memory-efficient transformer attention.
//!
//! In transformer attention, Q, K, V projections are usually separate matmuls.
//! Fusing them into a single larger matmul is more memory-efficient. This module
//! implements fused QKV projection and scaled dot-product attention.

use crate::DiffusionError;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for fused QKV attention.
#[derive(Debug, Clone)]
pub struct FusedAttentionConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    /// Dropout probability (default 0.0 — no dropout in inference).
    pub dropout_prob: f32,
    /// Attention scale factor (default: `1/sqrt(head_dim)`).
    pub scale: f32,
    /// Causal mask (default: false — not autoregressive).
    pub causal: bool,
}

impl FusedAttentionConfig {
    /// Create a new config with sensible defaults.
    pub fn new(num_heads: usize, head_dim: usize, seq_len: usize) -> Self {
        let scale = if head_dim > 0 {
            1.0 / (head_dim as f32).sqrt()
        } else {
            1.0
        };
        Self {
            num_heads,
            head_dim,
            seq_len,
            dropout_prob: 0.0,
            scale,
            causal: false,
        }
    }

    /// Config for standard Stable Diffusion attention (head_dim = 64).
    pub fn for_sd_attention(num_heads: usize, seq_len: usize) -> Self {
        Self::new(num_heads, 64, seq_len)
    }

    /// Embedding dimension: `num_heads * head_dim`.
    pub fn embed_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), DiffusionError> {
        if self.num_heads == 0 {
            return Err(DiffusionError::InvalidConfig(
                "num_heads must be > 0".to_string(),
            ));
        }
        if self.head_dim == 0 {
            return Err(DiffusionError::InvalidConfig(
                "head_dim must be > 0".to_string(),
            ));
        }
        if self.seq_len == 0 {
            return Err(DiffusionError::InvalidConfig(
                "seq_len must be > 0".to_string(),
            ));
        }
        if self.dropout_prob < 0.0 || self.dropout_prob > 1.0 {
            return Err(DiffusionError::InvalidConfig(format!(
                "dropout_prob must be in [0, 1], got {}",
                self.dropout_prob
            )));
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(DiffusionError::InvalidConfig(format!(
                "scale must be a positive finite value, got {}",
                self.scale
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Fused QKV projection result
// ---------------------------------------------------------------------------

/// Fused QKV projection result.
///
/// After splitting the packed projection, Q, K and V each have shape:
/// `[batch * num_heads * seq_len * head_dim]` stored in row-major order.
pub struct FusedQKV {
    /// Query tensor `[batch * num_heads * seq_len * head_dim]`.
    pub q: Vec<f32>,
    /// Key tensor `[batch * num_heads * seq_len * head_dim]`.
    pub k: Vec<f32>,
    /// Value tensor `[batch * num_heads * seq_len * head_dim]`.
    pub v: Vec<f32>,
    pub config: FusedAttentionConfig,
    pub batch_size: usize,
}

impl FusedQKV {
    /// Construct from a packed QKV tensor.
    ///
    /// `packed_qkv` has shape `[batch * seq_len * (3 * embed_dim)]`.
    /// The function splits it into Q, K, V each of shape
    /// `[batch * seq_len * embed_dim]`, then transposes each to
    /// `[batch * num_heads * seq_len * head_dim]`.
    pub fn from_packed(
        packed_qkv: &[f32],
        config: FusedAttentionConfig,
        batch_size: usize,
    ) -> Result<Self, DiffusionError> {
        config.validate()?;

        let embed_dim = config.embed_dim();
        let expected_len = batch_size * config.seq_len * 3 * embed_dim;
        if packed_qkv.len() != expected_len {
            return Err(DiffusionError::ShapeMismatch {
                op: "FusedQKV::from_packed".to_string(),
                expected: vec![batch_size, config.seq_len, 3 * embed_dim],
                got: vec![packed_qkv.len()],
            });
        }

        let seq_len = config.seq_len;
        let num_heads = config.num_heads;
        let head_dim = config.head_dim;

        // Each QKV component: [batch * num_heads * seq_len * head_dim]
        let component_len = batch_size * num_heads * seq_len * head_dim;
        let mut q = vec![0.0f32; component_len];
        let mut k = vec![0.0f32; component_len];
        let mut v = vec![0.0f32; component_len];

        // packed_qkv layout: [b, s, 3 * embed_dim]
        //   embed_dim = num_heads * head_dim
        // Q slice: [b, s, 0..embed_dim]
        // K slice: [b, s, embed_dim..2*embed_dim]
        // V slice: [b, s, 2*embed_dim..3*embed_dim]
        //
        // Target layout for each component: [b, h, s, d]
        //   index = b * num_heads * seq_len * head_dim
        //         + h * seq_len * head_dim
        //         + s * head_dim
        //         + d

        for b in 0..batch_size {
            for s in 0..seq_len {
                let src_base = (b * seq_len + s) * 3 * embed_dim;
                for h in 0..num_heads {
                    let head_offset = h * head_dim;
                    let dst_base =
                        b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim;
                    for d in 0..head_dim {
                        let q_src = src_base + head_offset + d;
                        let k_src = src_base + embed_dim + head_offset + d;
                        let v_src = src_base + 2 * embed_dim + head_offset + d;
                        let dst = dst_base + d;
                        q[dst] = packed_qkv[q_src];
                        k[dst] = packed_qkv[k_src];
                        v[dst] = packed_qkv[v_src];
                    }
                }
            }
        }

        Ok(Self {
            q,
            k,
            v,
            config,
            batch_size,
        })
    }

    /// Apply scaled dot-product attention and return output.
    ///
    /// Concatenates all heads → output shape `[batch * seq_len * embed_dim]`.
    pub fn apply_scaled_dot_product(&self) -> Result<Vec<f32>, DiffusionError> {
        let batch_size = self.batch_size;
        let num_heads = self.config.num_heads;
        let seq_len = self.config.seq_len;
        let head_dim = self.config.head_dim;
        let embed_dim = self.config.embed_dim();

        let output_len = batch_size * seq_len * embed_dim;
        let mut output = vec![0.0f32; output_len];

        for b in 0..batch_size {
            for h in 0..num_heads {
                let head_base = b * num_heads * seq_len * head_dim + h * seq_len * head_dim;
                let q_slice = &self.q[head_base..head_base + seq_len * head_dim];
                let k_slice = &self.k[head_base..head_base + seq_len * head_dim];
                let v_slice = &self.v[head_base..head_base + seq_len * head_dim];

                let head_out = scaled_dot_product_attention(
                    q_slice, k_slice, v_slice, seq_len, seq_len, head_dim,
                )?;

                // Scatter head output back to [b, s, h*head_dim..(h+1)*head_dim]
                for s in 0..seq_len {
                    let dst_base = (b * seq_len + s) * embed_dim + h * head_dim;
                    let src_base = s * head_dim;
                    output[dst_base..dst_base + head_dim]
                        .copy_from_slice(&head_out[src_base..src_base + head_dim]);
                }
            }
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Fused QKV matmul.
///
/// Splits the packed weight matrix into three equal slices (Q, K, V), then
/// multiplies each by the input to produce the packed output.
///
/// * `input`: `[batch * seq_len * in_dim]`
/// * `weight_qkv`: `[3 * embed_dim * in_dim]` — Q, K, V weights stacked row-wise
/// * Returns: `[batch * seq_len * 3 * embed_dim]`
pub fn fused_qkv_projection(
    input: &[f32],
    weight_qkv: &[f32],
    batch_size: usize,
    seq_len: usize,
    in_dim: usize,
    embed_dim: usize,
) -> Result<Vec<f32>, DiffusionError> {
    let expected_input = batch_size * seq_len * in_dim;
    if input.len() != expected_input {
        return Err(DiffusionError::ShapeMismatch {
            op: "fused_qkv_projection: input".to_string(),
            expected: vec![batch_size, seq_len, in_dim],
            got: vec![input.len()],
        });
    }

    let expected_weight = 3 * embed_dim * in_dim;
    if weight_qkv.len() != expected_weight {
        return Err(DiffusionError::ShapeMismatch {
            op: "fused_qkv_projection: weight_qkv".to_string(),
            expected: vec![3 * embed_dim, in_dim],
            got: vec![weight_qkv.len()],
        });
    }

    let token_count = batch_size * seq_len;
    let out_len = token_count * 3 * embed_dim;
    let mut output = vec![0.0f32; out_len];

    // Weight slices: each [embed_dim * in_dim]
    let w_q = &weight_qkv[0..embed_dim * in_dim];
    let w_k = &weight_qkv[embed_dim * in_dim..2 * embed_dim * in_dim];
    let w_v = &weight_qkv[2 * embed_dim * in_dim..3 * embed_dim * in_dim];

    for t in 0..token_count {
        let inp = &input[t * in_dim..(t + 1) * in_dim];
        let out_base = t * 3 * embed_dim;

        // Q
        matmul_vec(
            inp,
            w_q,
            &mut output[out_base..out_base + embed_dim],
            embed_dim,
            in_dim,
        );
        // K
        let k_base = out_base + embed_dim;
        matmul_vec(
            inp,
            w_k,
            &mut output[k_base..k_base + embed_dim],
            embed_dim,
            in_dim,
        );
        // V
        let v_base = out_base + 2 * embed_dim;
        matmul_vec(
            inp,
            w_v,
            &mut output[v_base..v_base + embed_dim],
            embed_dim,
            in_dim,
        );
    }

    Ok(output)
}

/// Multiply a weight matrix by a vector: `out[i] = sum_j weight[i*in_dim + j] * inp[j]`.
///
/// `weight` shape: `[out_dim * in_dim]`, row-major.
fn matmul_vec(inp: &[f32], weight: &[f32], out: &mut [f32], out_dim: usize, in_dim: usize) {
    for i in 0..out_dim {
        let row = &weight[i * in_dim..(i + 1) * in_dim];
        let mut acc = 0.0f32;
        for j in 0..in_dim {
            acc += row[j] * inp[j];
        }
        out[i] = acc;
    }
}

/// Numerically stable softmax in-place over the last dimension.
///
/// `data.len()` must be a multiple of `dim_size`.  Each chunk of `dim_size`
/// elements is treated as a separate probability vector.
pub fn softmax_over_dim(data: &mut [f32], dim_size: usize) {
    if dim_size == 0 {
        return;
    }
    for chunk in data.chunks_mut(dim_size) {
        // find max for numerical stability
        let max_val = chunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for x in chunk.iter_mut() {
            *x = (*x - max_val).exp();
            sum += *x;
        }
        if sum > 0.0 {
            for x in chunk.iter_mut() {
                *x /= sum;
            }
        }
    }
}

/// Scaled dot-product attention for a single head.
///
/// * `q`: `[seq_q * head_dim]`
/// * `k`: `[seq_k * head_dim]`
/// * `v`: `[seq_k * head_dim]`
/// * Returns: `[seq_q * head_dim]`
pub fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
) -> Result<Vec<f32>, DiffusionError> {
    if q.len() != seq_q * head_dim {
        return Err(DiffusionError::ShapeMismatch {
            op: "scaled_dot_product_attention: q".to_string(),
            expected: vec![seq_q, head_dim],
            got: vec![q.len()],
        });
    }
    if k.len() != seq_k * head_dim {
        return Err(DiffusionError::ShapeMismatch {
            op: "scaled_dot_product_attention: k".to_string(),
            expected: vec![seq_k, head_dim],
            got: vec![k.len()],
        });
    }
    if v.len() != seq_k * head_dim {
        return Err(DiffusionError::ShapeMismatch {
            op: "scaled_dot_product_attention: v".to_string(),
            expected: vec![seq_k, head_dim],
            got: vec![v.len()],
        });
    }

    let scale = if head_dim > 0 {
        1.0 / (head_dim as f32).sqrt()
    } else {
        1.0
    };

    // Compute attention scores: [seq_q * seq_k]
    let mut scores = vec![0.0f32; seq_q * seq_k];
    for qi in 0..seq_q {
        for ki in 0..seq_k {
            let q_row = &q[qi * head_dim..(qi + 1) * head_dim];
            let k_row = &k[ki * head_dim..(ki + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_row[d] * k_row[d];
            }
            scores[qi * seq_k + ki] = dot * scale;
        }
    }

    // Apply softmax over key dimension (each row of [seq_q, seq_k])
    softmax_over_dim(&mut scores, seq_k);

    // Multiply by V: [seq_q * head_dim]
    let mut output = vec![0.0f32; seq_q * head_dim];
    for qi in 0..seq_q {
        let attn_row = &scores[qi * seq_k..(qi + 1) * seq_k];
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for ki in 0..seq_k {
                acc += attn_row[ki] * v[ki * head_dim + d];
            }
            output[qi * head_dim + d] = acc;
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // --- Config tests -------------------------------------------------------

    #[test]
    fn test_config_new() {
        let cfg = FusedAttentionConfig::new(8, 64, 16);
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.seq_len, 16);
        assert_eq!(cfg.dropout_prob, 0.0);
        assert!(!cfg.causal);
        let expected_scale = 1.0f32 / 64.0f32.sqrt();
        assert_abs_diff_eq!(cfg.scale, expected_scale, epsilon = 1e-6);
    }

    #[test]
    fn test_config_validate() {
        assert!(FusedAttentionConfig::new(8, 64, 16).validate().is_ok());

        let mut cfg = FusedAttentionConfig::new(0, 64, 16);
        assert!(cfg.validate().is_err());

        cfg = FusedAttentionConfig::new(8, 0, 16);
        assert!(cfg.validate().is_err());

        cfg = FusedAttentionConfig::new(8, 64, 0);
        assert!(cfg.validate().is_err());

        let mut cfg2 = FusedAttentionConfig::new(8, 64, 16);
        cfg2.dropout_prob = 1.5;
        assert!(cfg2.validate().is_err());

        let mut cfg3 = FusedAttentionConfig::new(8, 64, 16);
        cfg3.scale = -1.0;
        assert!(cfg3.validate().is_err());
    }

    #[test]
    fn test_config_embed_dim() {
        let cfg = FusedAttentionConfig::new(8, 64, 16);
        assert_eq!(cfg.embed_dim(), 512);

        let cfg2 = FusedAttentionConfig::new(4, 32, 8);
        assert_eq!(cfg2.embed_dim(), 128);
    }

    #[test]
    fn test_fused_qkv_config_for_sd() {
        let cfg = FusedAttentionConfig::for_sd_attention(8, 77);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.seq_len, 77);
        assert_eq!(cfg.embed_dim(), 512);
    }

    // --- fused_qkv_projection tests -----------------------------------------

    #[test]
    fn test_fused_qkv_projection_shape() {
        let batch = 2;
        let seq_len = 4;
        let in_dim = 8;
        let embed_dim = 6;

        let input = vec![1.0f32; batch * seq_len * in_dim];
        let weight = vec![0.1f32; 3 * embed_dim * in_dim];

        let result = fused_qkv_projection(&input, &weight, batch, seq_len, in_dim, embed_dim);
        assert!(result.is_ok());
        let out = result.unwrap();
        assert_eq!(out.len(), batch * seq_len * 3 * embed_dim);
    }

    #[test]
    fn test_fused_qkv_projection_identity() {
        // Use in_dim == embed_dim and identity-like weights so output ≈ input (×3)
        let batch = 1;
        let seq_len = 2;
        let in_dim = 3;
        let embed_dim = 3;

        let input = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];

        // Build weight [3*embed_dim, in_dim] = [9, 3] with identity blocks
        // W_q = W_k = W_v = I_3
        let mut weight = vec![0.0f32; 3 * embed_dim * in_dim];
        for i in 0..embed_dim {
            // Q block
            weight[i * in_dim + i] = 1.0;
            // K block
            weight[(embed_dim + i) * in_dim + i] = 1.0;
            // V block
            weight[(2 * embed_dim + i) * in_dim + i] = 1.0;
        }

        let out = fused_qkv_projection(&input, &weight, batch, seq_len, in_dim, embed_dim)
            .expect("projection failed");
        // Each token should produce [token | token | token]
        // Token 0: [1,2,3]  → Q=[1,2,3], K=[1,2,3], V=[1,2,3]
        assert_abs_diff_eq!(out[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(out[1], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(out[2], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(out[3], 1.0, epsilon = 1e-6); // K copy
        assert_abs_diff_eq!(out[4], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(out[5], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(out[6], 1.0, epsilon = 1e-6); // V copy
    }

    #[test]
    fn test_fused_qkv_projection_invalid_input() {
        let batch = 1;
        let seq_len = 4;
        let in_dim = 8;
        let embed_dim = 4;

        // Wrong input length
        let input = vec![1.0f32; batch * seq_len * in_dim - 1];
        let weight = vec![0.0f32; 3 * embed_dim * in_dim];
        assert!(fused_qkv_projection(&input, &weight, batch, seq_len, in_dim, embed_dim).is_err());

        // Wrong weight length
        let input2 = vec![1.0f32; batch * seq_len * in_dim];
        let weight2 = vec![0.0f32; 3 * embed_dim * in_dim - 1];
        assert!(
            fused_qkv_projection(&input2, &weight2, batch, seq_len, in_dim, embed_dim).is_err()
        );
    }

    // --- FusedQKV::from_packed tests ----------------------------------------

    #[test]
    fn test_from_packed_shape_valid() {
        let batch = 2;
        let num_heads = 4;
        let head_dim = 8;
        let seq_len = 6;
        let embed_dim = num_heads * head_dim;

        let packed = vec![0.5f32; batch * seq_len * 3 * embed_dim];
        let cfg = FusedAttentionConfig::new(num_heads, head_dim, seq_len);
        let fqkv = FusedQKV::from_packed(&packed, cfg, batch).expect("from_packed failed");

        let expected_component_len = batch * num_heads * seq_len * head_dim;
        assert_eq!(fqkv.q.len(), expected_component_len);
        assert_eq!(fqkv.k.len(), expected_component_len);
        assert_eq!(fqkv.v.len(), expected_component_len);
    }

    #[test]
    fn test_from_packed_invalid_length() {
        let batch = 1;
        let cfg = FusedAttentionConfig::new(2, 4, 3);
        let embed_dim = cfg.embed_dim();

        // One element short
        let packed = vec![0.0f32; batch * cfg.seq_len * 3 * embed_dim - 1];
        assert!(FusedQKV::from_packed(&packed, cfg, batch).is_err());
    }

    // --- scaled_dot_product_attention tests ---------------------------------

    #[test]
    fn test_scaled_dot_product_single_query() {
        // 1 query, 3 keys/values, head_dim=2
        let head_dim = 2;
        let seq_q = 1;
        let seq_k = 3;

        let q = vec![1.0f32, 0.0f32];
        let k = vec![1.0f32, 0.0f32, 0.0f32, 1.0f32, -1.0f32, 0.0f32];
        let v = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];

        let out =
            scaled_dot_product_attention(&q, &k, &v, seq_q, seq_k, head_dim).expect("sdpa failed");
        assert_eq!(out.len(), seq_q * head_dim);
        // Output should be a weighted sum of V rows; all values finite
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_scaled_dot_product_attention_output_shape() {
        let seq_q = 5;
        let seq_k = 5;
        let head_dim = 16;

        let q = vec![0.1f32; seq_q * head_dim];
        let k = vec![0.2f32; seq_k * head_dim];
        let v = vec![0.3f32; seq_k * head_dim];

        let out =
            scaled_dot_product_attention(&q, &k, &v, seq_q, seq_k, head_dim).expect("sdpa failed");
        assert_eq!(out.len(), seq_q * head_dim);
    }

    #[test]
    fn test_scaled_dot_product_attention_invalid_lengths() {
        let q = vec![1.0f32; 6]; // seq_q=3, head_dim=2
        let k = vec![1.0f32; 4]; // seq_k=2, head_dim=2
        let v = vec![1.0f32; 3]; // wrong: should be seq_k * head_dim = 4

        assert!(scaled_dot_product_attention(&q, &k, &v, 3, 2, 2).is_err());
    }

    // --- softmax tests ------------------------------------------------------

    #[test]
    fn test_softmax_sums_to_one() {
        let mut data = vec![1.0f32, 2.0f32, 3.0f32, 0.5f32, 1.5f32, 2.5f32];
        softmax_over_dim(&mut data, 3);
        let sum0: f32 = data[0..3].iter().sum();
        let sum1: f32 = data[3..6].iter().sum();
        assert_abs_diff_eq!(sum0, 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sum1, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_softmax_max_stays_in_range() {
        // Even with extreme values, softmax output must be in [0, 1]
        let mut data = vec![1000.0f32, -1000.0f32, 500.0f32];
        softmax_over_dim(&mut data, 3);
        for v in &data {
            assert!(*v >= 0.0 && *v <= 1.0);
        }
        let sum: f32 = data.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }

    // --- apply_scaled_dot_product tests -------------------------------------

    #[test]
    fn test_apply_scaled_dot_product() {
        let batch = 1;
        let num_heads = 2;
        let head_dim = 4;
        let seq_len = 3;
        let embed_dim = num_heads * head_dim;

        let packed = vec![0.1f32; batch * seq_len * 3 * embed_dim];
        let cfg = FusedAttentionConfig::new(num_heads, head_dim, seq_len);
        let fqkv = FusedQKV::from_packed(&packed, cfg, batch).expect("from_packed");
        let out = fqkv.apply_scaled_dot_product().expect("attention failed");

        assert_eq!(out.len(), batch * seq_len * embed_dim);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_attention_with_identity_v() {
        // When V = I (one-hot rows), the output is a weighted combination of rows.
        // All output values should be in [0, 1] since attention weights are in [0,1].
        let seq_len = 4;
        let head_dim = 4;
        // V: identity matrix flattened [seq_len * head_dim]
        let mut v = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            v[i * head_dim + i] = 1.0;
        }
        let q = vec![0.5f32; seq_len * head_dim];
        let k = vec![0.5f32; seq_len * head_dim];

        let out = scaled_dot_product_attention(&q, &k, &v, seq_len, seq_len, head_dim)
            .expect("sdpa failed");

        // All values must be in [0, 1]
        for val in &out {
            assert!(
                *val >= 0.0 && *val <= 1.0 + 1e-6,
                "value {} out of range",
                val
            );
        }
    }

    #[test]
    fn test_zero_attention_output_shape() {
        // All-zero Q, K, V should produce all-zero output with correct shape.
        let batch = 3;
        let num_heads = 4;
        let head_dim = 8;
        let seq_len = 5;
        let embed_dim = num_heads * head_dim;

        let packed = vec![0.0f32; batch * seq_len * 3 * embed_dim];
        let cfg = FusedAttentionConfig::new(num_heads, head_dim, seq_len);
        let fqkv = FusedQKV::from_packed(&packed, cfg, batch).expect("from_packed");
        let out = fqkv.apply_scaled_dot_product().expect("attention");

        assert_eq!(out.len(), batch * seq_len * embed_dim);
        // Softmax of equal values produces 1/seq_k weights, V=0 → output=0
        for val in &out {
            assert_abs_diff_eq!(*val, 0.0, epsilon = 1e-6);
        }
    }
}
