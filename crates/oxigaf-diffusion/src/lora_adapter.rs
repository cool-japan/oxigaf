//! LoRA (Low-Rank Adaptation) for fine-tuning diffusion model attention weights.
//!
//! LoRA decomposes weight updates as ΔW = B·A where A ∈ ℝ^{r×k} and B ∈ ℝ^{d×r}
//! with r << min(d,k). This dramatically reduces the number of trainable parameters
//! while still allowing meaningful model adaptation.
//!
//! Reference: Hu et al. 2021 — "LoRA: Low-Rank Adaptation of Large Language Models"

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during LoRA operations.
#[derive(Debug, Error)]
pub enum LoraError {
    /// Rank must be at least 1.
    #[error("Invalid rank: {rank}, must be >= 1")]
    InvalidRank { rank: usize },

    /// Configuration is invalid.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// Matrix dimension does not match expectation.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Named layer was not found in the adapter.
    #[error("Layer not found: {name}")]
    LayerNotFound { name: String },

    /// Numerical issue (e.g., non-finite value).
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Adapter has no layers.
    #[error("Empty adapter")]
    EmptyAdapter,

    /// Serialization or deserialization error.
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

// ---------------------------------------------------------------------------
// PRNG helpers (xorshift64 + Box-Muller)
// ---------------------------------------------------------------------------

/// Advance xorshift64 state by one step, returning the new state and a u64 output.
/// Guarantees state != 0 by clamping before the shift sequence.
#[inline]
fn xorshift64(state: u64) -> (u64, u64) {
    let mut s = state.max(1);
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    (s, s)
}

/// Map a u64 to a uniform f32 in [0, 1).
#[inline]
fn u64_to_uniform(v: u64) -> f32 {
    // Use upper 23 bits for mantissa precision.
    let mantissa = (v >> 41) as f32;
    mantissa / (1u64 << 23) as f32
}

/// Draw one Gaussian sample N(0, stddev) using Box-Muller.
/// Consumes two xorshift64 steps; returns (sample, new_state).
fn gaussian_sample(state: u64, stddev: f32) -> (f32, u64) {
    let (s1, u1_raw) = xorshift64(state);
    let (s2, u2_raw) = xorshift64(s1);
    let u1 = u64_to_uniform(u1_raw).max(f32::EPSILON); // avoid log(0)
    let u2 = u64_to_uniform(u2_raw);
    let r = (-2.0 * u1.ln()).sqrt() * stddev;
    let theta = 2.0 * std::f32::consts::PI * u2;
    let sample = r * theta.cos();
    (sample, s2)
}

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------

/// Matrix multiply: C = A @ B
///
/// A is [m × k], B is [k × n], C is [m × n] — all row-major.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Result<Vec<f32>, LoraError> {
    if a.len() != m * k {
        return Err(LoraError::DimensionMismatch {
            expected: m * k,
            actual: a.len(),
        });
    }
    if b.len() != k * n {
        return Err(LoraError::DimensionMismatch {
            expected: k * n,
            actual: b.len(),
        });
    }
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for l in 0..k {
            let a_val = a[i * k + l];
            if a_val == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += a_val * b[l * n + j];
            }
        }
    }
    Ok(c)
}

/// Transpose a [rows × cols] row-major matrix to [cols × rows].
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// LoRA initialisation helpers
// ---------------------------------------------------------------------------

/// Gaussian-initialize a LoRA A matrix [rank × in_dim] with xorshift64 + Box-Muller.
/// Standard deviation = 1/sqrt(rank) (Kaiming-style approximation).
pub fn init_lora_a(rank: usize, in_dim: usize, seed: u64) -> Vec<f32> {
    let stddev = 1.0 / (rank as f32).sqrt();
    let mut state = seed.max(1);
    let mut out = Vec::with_capacity(rank * in_dim);
    for _ in 0..(rank * in_dim) {
        let (sample, new_state) = gaussian_sample(state, stddev);
        state = new_state;
        out.push(sample);
    }
    out
}

/// Zero-initialize LoRA B matrix [out_dim × rank] (standard LoRA).
pub fn init_lora_b(out_dim: usize, rank: usize) -> Vec<f32> {
    vec![0.0f32; out_dim * rank]
}

// ---------------------------------------------------------------------------
// LoraLayer
// ---------------------------------------------------------------------------

/// A single LoRA layer: ΔW = alpha/rank * B @ A
#[derive(Debug, Clone)]
pub struct LoraLayer {
    /// Layer identifier (e.g. `"unet.attn1.to_q"`).
    pub name: String,
    /// Input dimension k.
    pub in_dim: usize,
    /// Output dimension d.
    pub out_dim: usize,
    /// LoRA rank r.
    pub rank: usize,
    /// Scaling factor (default = rank).
    pub alpha: f32,
    /// A matrix [rank × in_dim] row-major.
    pub a_matrix: Vec<f32>,
    /// B matrix [out_dim × rank] row-major.
    pub b_matrix: Vec<f32>,
}

impl LoraLayer {
    /// Create a new LoRA layer.
    ///
    /// A ~ N(0, 1/sqrt(rank)), B = 0 (standard LoRA init).
    pub fn new(
        name: impl Into<String>,
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f32,
        seed: u64,
    ) -> Result<Self, LoraError> {
        if rank == 0 {
            return Err(LoraError::InvalidRank { rank });
        }
        let a_matrix = init_lora_a(rank, in_dim, seed);
        let b_matrix = init_lora_b(out_dim, rank);
        Ok(Self {
            name: name.into(),
            in_dim,
            out_dim,
            rank,
            alpha,
            a_matrix,
            b_matrix,
        })
    }

    /// Scaling factor applied to the LoRA delta: alpha / rank.
    #[inline]
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Apply LoRA delta to input: output = scaling * B @ A @ input
    ///
    /// `input`  layout: [in_dim  × batch_size] row-major (in_dim rows, batch cols)
    /// `output` layout: [out_dim × batch_size] row-major
    pub fn apply(&self, input: &[f32], batch_size: usize) -> Result<Vec<f32>, LoraError> {
        let expected_len = self.in_dim * batch_size;
        if input.len() != expected_len {
            return Err(LoraError::DimensionMismatch {
                expected: expected_len,
                actual: input.len(),
            });
        }
        // Step 1: intermediate = A @ input  [rank × batch_size]
        let intermediate = matmul(&self.a_matrix, input, self.rank, self.in_dim, batch_size)?;
        // Step 2: out = B @ intermediate    [out_dim × batch_size]
        let raw = matmul(
            &self.b_matrix,
            &intermediate,
            self.out_dim,
            self.rank,
            batch_size,
        )?;
        let scale = self.scaling();
        Ok(raw.into_iter().map(|v| v * scale).collect())
    }

    /// Compute the full weight delta ΔW = scaling * B @ A.
    ///
    /// Returns an [out_dim × in_dim] matrix (row-major).
    pub fn weight_delta(&self) -> Vec<f32> {
        // B @ A: B is [out_dim × rank], A is [rank × in_dim] → [out_dim × in_dim]
        let raw = matmul(
            &self.b_matrix,
            &self.a_matrix,
            self.out_dim,
            self.rank,
            self.in_dim,
        )
        .unwrap_or_else(|_| vec![0.0f32; self.out_dim * self.in_dim]);
        let scale = self.scaling();
        raw.into_iter().map(|v| v * scale).collect()
    }

    /// Merge the LoRA delta into a base weight matrix in-place.
    ///
    /// `base_weight` must be [out_dim × in_dim] row-major.
    pub fn merge_into_weight(&self, base_weight: &mut [f32]) -> Result<(), LoraError> {
        let expected = self.out_dim * self.in_dim;
        if base_weight.len() != expected {
            return Err(LoraError::DimensionMismatch {
                expected,
                actual: base_weight.len(),
            });
        }
        let delta = self.weight_delta();
        for (w, d) in base_weight.iter_mut().zip(delta.iter()) {
            *w += d;
        }
        Ok(())
    }

    /// Total trainable parameter count: rank * (in_dim + out_dim).
    #[inline]
    pub fn param_count(&self) -> usize {
        self.rank * (self.in_dim + self.out_dim)
    }

    /// Compression ratio: trainable params / full weight params.
    #[inline]
    pub fn compression_ratio(&self) -> f32 {
        let full = self.in_dim * self.out_dim;
        if full == 0 {
            return 0.0;
        }
        self.param_count() as f32 / full as f32
    }
}

// ---------------------------------------------------------------------------
// LoraConfig
// ---------------------------------------------------------------------------

/// LoRA configuration for model adaptation.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Default rank for all layers.
    pub rank: usize,
    /// Default alpha scaling factor.
    pub alpha: f32,
    /// Layer name patterns to apply LoRA to.
    pub target_modules: Vec<String>,
    /// LoRA dropout rate.
    pub dropout: f32,
    /// Bias handling: `"none"` | `"all"` | `"lora_only"`.
    pub bias: String,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 4,
            alpha: 4.0,
            target_modules: vec![
                "to_q".to_string(),
                "to_k".to_string(),
                "to_v".to_string(),
                "to_out".to_string(),
            ],
            dropout: 0.0,
            bias: "none".to_string(),
        }
    }
}

impl LoraConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), LoraError> {
        if self.rank == 0 {
            return Err(LoraError::InvalidRank { rank: self.rank });
        }
        if self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err(LoraError::InvalidConfig(format!(
                "dropout must be in [0, 1), got {}",
                self.dropout
            )));
        }
        let valid_bias = ["none", "all", "lora_only"];
        if !valid_bias.contains(&self.bias.as_str()) {
            return Err(LoraError::InvalidConfig(format!(
                "bias must be one of {:?}, got '{}'",
                valid_bias, self.bias
            )));
        }
        Ok(())
    }

    /// Create a config with a specific rank (alpha defaults to rank).
    pub fn with_rank(rank: usize) -> Self {
        Self {
            rank,
            alpha: rank as f32,
            ..Default::default()
        }
    }

    /// Preset suitable for avatar/face fine-tuning (rank=8, alpha=16).
    pub fn avatar_preset() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            target_modules: vec![
                "to_q".to_string(),
                "to_k".to_string(),
                "to_v".to_string(),
                "to_out".to_string(),
            ],
            dropout: 0.0,
            bias: "none".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// LoraAdapter
// ---------------------------------------------------------------------------

/// A collection of LoRA layers representing a full model adaptation.
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// All LoRA layers in this adapter.
    pub layers: Vec<LoraLayer>,
    /// Configuration used to create this adapter.
    pub config: LoraConfig,
    /// Current training step.
    pub step: usize,
}

impl LoraAdapter {
    /// Create a new empty adapter with the given configuration.
    pub fn new(config: LoraConfig) -> Result<Self, LoraError> {
        config.validate()?;
        Ok(Self {
            layers: Vec::new(),
            config,
            step: 0,
        })
    }

    /// Append a LoRA layer to the adapter.
    pub fn add_layer(&mut self, layer: LoraLayer) {
        self.layers.push(layer);
    }

    /// Look up a layer by name (immutable).
    pub fn get_layer(&self, name: &str) -> Option<&LoraLayer> {
        self.layers.iter().find(|l| l.name == name)
    }

    /// Look up a layer by name (mutable).
    pub fn get_layer_mut(&mut self, name: &str) -> Option<&mut LoraLayer> {
        self.layers.iter_mut().find(|l| l.name == name)
    }

    /// Number of layers in this adapter.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Sum of trainable parameters across all layers.
    pub fn total_params(&self) -> usize {
        self.layers.iter().map(|l| l.param_count()).sum()
    }

    /// Names of all layers.
    pub fn layer_names(&self) -> Vec<&str> {
        self.layers.iter().map(|l| l.name.as_str()).collect()
    }

    /// Scale all LoRA outputs by `scale` (modifies alpha of every layer).
    pub fn set_scale(&mut self, scale: f32) {
        for layer in self.layers.iter_mut() {
            layer.alpha = layer.rank as f32 * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// Dropout helper
// ---------------------------------------------------------------------------

/// Apply inverted dropout in-place to `activations`.
///
/// Each element is zeroed with probability `dropout_rate`, and survivors are
/// scaled by `1 / (1 - dropout_rate)`.
pub fn apply_lora_dropout(
    activations: &mut [f32],
    dropout_rate: f32,
    seed: u64,
) -> Result<(), LoraError> {
    if !(0.0..1.0).contains(&dropout_rate) {
        return Err(LoraError::InvalidConfig(format!(
            "dropout_rate must be in [0, 1), got {}",
            dropout_rate
        )));
    }
    if dropout_rate == 0.0 {
        return Ok(());
    }
    let keep_prob = 1.0 - dropout_rate;
    let scale = 1.0 / keep_prob;
    let mut state = seed.max(1);
    for v in activations.iter_mut() {
        let (new_state, raw) = xorshift64(state);
        state = new_state;
        let uniform = u64_to_uniform(raw);
        if uniform < dropout_rate {
            *v = 0.0;
        } else {
            *v *= scale;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Norm helper
// ---------------------------------------------------------------------------

/// Compute the L2 norm of the LoRA weight delta ΔW.
pub fn lora_weight_norm(layer: &LoraLayer) -> f32 {
    let delta = layer.weight_delta();
    let sum_sq: f32 = delta.iter().map(|&v| v * v).sum();
    sum_sq.sqrt()
}

// ---------------------------------------------------------------------------
// Merge helper
// ---------------------------------------------------------------------------

/// Merge all matching LoRA layers into a weight dictionary.
///
/// For each `(name, matrix)` pair in `weights` whose name matches a LoRA layer,
/// the delta is added in-place.  Returns the number of layers merged.
pub fn merge_lora_weights(
    weights: &mut [(String, Vec<f32>)],
    adapter: &LoraAdapter,
) -> Result<usize, LoraError> {
    let mut merged = 0usize;
    for (name, matrix) in weights.iter_mut() {
        if let Some(lora_layer) = adapter.get_layer(name.as_str()) {
            lora_layer.merge_into_weight(matrix)?;
            merged += 1;
        }
    }
    Ok(merged)
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Summary statistics for a LoRA adapter.
#[derive(Debug, Clone)]
pub struct LoraStats {
    /// Number of LoRA layers.
    pub n_layers: usize,
    /// Total trainable parameters across all layers.
    pub total_params: usize,
    /// Mean L2 norm of weight deltas.
    pub mean_weight_norm: f32,
    /// Maximum L2 norm.
    pub max_weight_norm: f32,
    /// Minimum L2 norm.
    pub min_weight_norm: f32,
    /// Approximate compression ratio over full weights.
    pub compression_ratio: f32,
}

/// Compute adapter statistics.
pub fn compute_lora_stats(adapter: &LoraAdapter) -> Result<LoraStats, LoraError> {
    if adapter.layers.is_empty() {
        return Err(LoraError::EmptyAdapter);
    }
    let n_layers = adapter.layers.len();
    let total_params = adapter.total_params();

    let norms: Vec<f32> = adapter.layers.iter().map(lora_weight_norm).collect();
    let sum: f32 = norms.iter().sum();
    let mean_weight_norm = sum / n_layers as f32;
    let max_weight_norm = norms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_weight_norm = norms.iter().cloned().fold(f32::INFINITY, f32::min);

    let total_base: usize = adapter.layers.iter().map(|l| l.in_dim * l.out_dim).sum();
    let compression_ratio = if total_base == 0 {
        0.0
    } else {
        total_params as f32 / total_base as f32
    };

    Ok(LoraStats {
        n_layers,
        total_params,
        mean_weight_norm,
        max_weight_norm,
        min_weight_norm,
        compression_ratio,
    })
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

/// Serialize an adapter to a flat `Vec<f32>` for checkpointing.
///
/// Format:
/// ```text
/// [n_layers as f32,
///  [in_dim, out_dim, rank, alpha, a_data..., b_data...] per layer...]
/// ```
pub fn serialize_adapter(adapter: &LoraAdapter) -> Vec<f32> {
    let n = adapter.layers.len();
    // Estimate capacity
    let cap = 1 + adapter
        .layers
        .iter()
        .fold(0, |acc, l| acc + 4 + l.a_matrix.len() + l.b_matrix.len());
    let mut out = Vec::with_capacity(cap);
    out.push(n as f32);
    for layer in &adapter.layers {
        out.push(layer.in_dim as f32);
        out.push(layer.out_dim as f32);
        out.push(layer.rank as f32);
        out.push(layer.alpha);
        out.extend_from_slice(&layer.a_matrix);
        out.extend_from_slice(&layer.b_matrix);
    }
    out
}

/// Deserialize an adapter from a flat `Vec<f32>`.
pub fn deserialize_adapter(data: &[f32], config: LoraConfig) -> Result<LoraAdapter, LoraError> {
    if data.is_empty() {
        return Err(LoraError::SerializationError("data is empty".to_string()));
    }
    let n_layers = *data
        .first()
        .ok_or_else(|| LoraError::SerializationError("missing n_layers".to_string()))?
        as usize;

    let mut adapter = LoraAdapter::new(config)?;
    let mut cursor = 1usize;

    for layer_idx in 0..n_layers {
        // Read header: [in_dim, out_dim, rank, alpha]
        if cursor + 4 > data.len() {
            return Err(LoraError::SerializationError(format!(
                "truncated header at layer {}",
                layer_idx
            )));
        }
        let in_dim = data[cursor] as usize;
        let out_dim = data[cursor + 1] as usize;
        let rank = data[cursor + 2] as usize;
        let alpha = data[cursor + 3];
        cursor += 4;

        if rank == 0 {
            return Err(LoraError::InvalidRank { rank });
        }

        let a_len = rank * in_dim;
        let b_len = out_dim * rank;

        if cursor + a_len + b_len > data.len() {
            return Err(LoraError::SerializationError(format!(
                "truncated matrix data at layer {}",
                layer_idx
            )));
        }

        let a_matrix = data[cursor..cursor + a_len].to_vec();
        cursor += a_len;
        let b_matrix = data[cursor..cursor + b_len].to_vec();
        cursor += b_len;

        let layer = LoraLayer {
            name: format!("layer_{}", layer_idx),
            in_dim,
            out_dim,
            rank,
            alpha,
            a_matrix,
            b_matrix,
        };
        adapter.add_layer(layer);
    }

    Ok(adapter)
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

/// Interpolate between two adapters for style mixing.
///
/// `out_layer = lerp(a_layer, b_layer, t)` — t=0 returns adapter_a, t=1 returns adapter_b.
/// Both adapters must have the same layers (same names and dimensions) in the same order.
pub fn interpolate_adapters(
    adapter_a: &LoraAdapter,
    adapter_b: &LoraAdapter,
    t: f32,
) -> Result<LoraAdapter, LoraError> {
    if adapter_a.layers.len() != adapter_b.layers.len() {
        return Err(LoraError::DimensionMismatch {
            expected: adapter_a.layers.len(),
            actual: adapter_b.layers.len(),
        });
    }
    let mut out = LoraAdapter::new(adapter_a.config.clone())?;
    let one_minus_t = 1.0 - t;
    for (la, lb) in adapter_a.layers.iter().zip(adapter_b.layers.iter()) {
        if la.name != lb.name
            || la.in_dim != lb.in_dim
            || la.out_dim != lb.out_dim
            || la.rank != lb.rank
        {
            return Err(LoraError::InvalidConfig(format!(
                "Layer mismatch: '{}' vs '{}'",
                la.name, lb.name
            )));
        }
        let a_matrix: Vec<f32> = la
            .a_matrix
            .iter()
            .zip(lb.a_matrix.iter())
            .map(|(&a, &b)| one_minus_t * a + t * b)
            .collect();
        let b_matrix: Vec<f32> = la
            .b_matrix
            .iter()
            .zip(lb.b_matrix.iter())
            .map(|(&a, &b)| one_minus_t * a + t * b)
            .collect();
        let alpha = one_minus_t * la.alpha + t * lb.alpha;
        out.add_layer(LoraLayer {
            name: la.name.clone(),
            in_dim: la.in_dim,
            out_dim: la.out_dim,
            rank: la.rank,
            alpha,
            a_matrix,
            b_matrix,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Backward pass
// ---------------------------------------------------------------------------

/// Compute the gradients of the LoRA loss w.r.t. A and B.
///
/// Forward:  Y = scaling * B @ A @ X
///
/// Gradients:
/// - `dL/dA = scaling * B^T @ grad_output @ X^T` → shape `[rank × in_dim]`
/// - `dL/dB = scaling * grad_output @ (A @ X)^T` → shape `[out_dim × rank]`
///
/// `input`       : [in_dim  × batch_size]
/// `grad_output` : [out_dim × batch_size]
pub fn lora_backward(
    layer: &LoraLayer,
    input: &[f32],
    grad_output: &[f32],
    batch_size: usize,
) -> Result<(Vec<f32>, Vec<f32>), LoraError> {
    let expected_input = layer.in_dim * batch_size;
    if input.len() != expected_input {
        return Err(LoraError::DimensionMismatch {
            expected: expected_input,
            actual: input.len(),
        });
    }
    let expected_grad = layer.out_dim * batch_size;
    if grad_output.len() != expected_grad {
        return Err(LoraError::DimensionMismatch {
            expected: expected_grad,
            actual: grad_output.len(),
        });
    }

    let scale = layer.scaling();

    // AX = A @ X   [rank × batch_size]
    let ax = matmul(&layer.a_matrix, input, layer.rank, layer.in_dim, batch_size)?;

    // dL/dB = scale * grad_output @ AX^T
    // grad_output: [out_dim × batch_size], AX^T: [batch_size × rank]
    let ax_t = transpose(&ax, layer.rank, batch_size);
    let grad_b_raw = matmul(grad_output, &ax_t, layer.out_dim, batch_size, layer.rank)?;
    let grad_b: Vec<f32> = grad_b_raw.iter().map(|&v| v * scale).collect();

    // dL/dA = scale * B^T @ grad_output @ X^T
    // B^T: [rank × out_dim], grad_output: [out_dim × batch_size] → BT_go: [rank × batch_size]
    let b_t = transpose(&layer.b_matrix, layer.out_dim, layer.rank);
    let bt_go = matmul(&b_t, grad_output, layer.rank, layer.out_dim, batch_size)?;

    // bt_go @ X^T: [rank × batch_size] @ [batch_size × in_dim] → [rank × in_dim]
    let x_t = transpose(input, layer.in_dim, batch_size);
    let grad_a_raw = matmul(&bt_go, &x_t, layer.rank, batch_size, layer.in_dim)?;
    let grad_a: Vec<f32> = grad_a_raw.iter().map(|&v| v * scale).collect();

    Ok((grad_a, grad_b))
}

// ---------------------------------------------------------------------------
// SGD update
// ---------------------------------------------------------------------------

/// Apply an SGD step to a LoRA layer's A and B matrices.
///
/// `param -= lr * grad`
pub fn lora_sgd_step(
    layer: &mut LoraLayer,
    grad_a: &[f32],
    grad_b: &[f32],
    lr: f32,
) -> Result<(), LoraError> {
    if grad_a.len() != layer.a_matrix.len() {
        return Err(LoraError::DimensionMismatch {
            expected: layer.a_matrix.len(),
            actual: grad_a.len(),
        });
    }
    if grad_b.len() != layer.b_matrix.len() {
        return Err(LoraError::DimensionMismatch {
            expected: layer.b_matrix.len(),
            actual: grad_b.len(),
        });
    }
    for (p, &g) in layer.a_matrix.iter_mut().zip(grad_a.iter()) {
        *p -= lr * g;
    }
    for (p, &g) in layer.b_matrix.iter_mut().zip(grad_b.iter()) {
        *p -= lr * g;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 42;

    // --- LoraLayer::new ---

    #[test]
    fn test_lora_layer_new_valid() {
        let layer = LoraLayer::new("test", 64, 128, 4, 4.0, SEED);
        assert!(layer.is_ok());
        let l = layer.unwrap();
        assert_eq!(l.in_dim, 64);
        assert_eq!(l.out_dim, 128);
        assert_eq!(l.rank, 4);
        assert_eq!(l.a_matrix.len(), 4 * 64);
        assert_eq!(l.b_matrix.len(), 128 * 4);
    }

    #[test]
    fn test_lora_layer_new_rank_zero_error() {
        let result = LoraLayer::new("test", 64, 128, 0, 1.0, SEED);
        assert!(matches!(result, Err(LoraError::InvalidRank { rank: 0 })));
    }

    // --- LoraLayer::scaling ---

    #[test]
    fn test_lora_layer_scaling_alpha_equals_rank() {
        let layer = LoraLayer::new("test", 8, 8, 4, 4.0, SEED).unwrap();
        let s = layer.scaling();
        assert!(
            (s - 1.0).abs() < 1e-6,
            "scaling should be 1.0 when alpha==rank, got {}",
            s
        );
    }

    #[test]
    fn test_lora_layer_scaling_custom() {
        let layer = LoraLayer::new("test", 8, 8, 4, 8.0, SEED).unwrap();
        assert!((layer.scaling() - 2.0).abs() < 1e-6);
    }

    // --- LoraLayer::apply ---

    #[test]
    fn test_lora_layer_apply_zero_b_gives_zero_output() {
        // B is initialised to zero, so output should be exactly zero.
        let layer = LoraLayer::new("test", 4, 8, 2, 2.0, SEED).unwrap();
        let input = vec![1.0f32; 4 * 3]; // batch_size=3
        let output = layer.apply(&input, 3).unwrap();
        assert_eq!(output.len(), 8 * 3);
        for &v in &output {
            assert_eq!(v, 0.0, "zero B should produce zero output");
        }
    }

    #[test]
    fn test_lora_layer_apply_dimension_mismatch() {
        let layer = LoraLayer::new("test", 4, 8, 2, 2.0, SEED).unwrap();
        let bad_input = vec![1.0f32; 3]; // wrong size
        let result = layer.apply(&bad_input, 3);
        assert!(matches!(result, Err(LoraError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_lora_layer_apply_known_result() {
        // Manually set A and B to known values for a small case.
        let mut layer = LoraLayer::new("test", 2, 2, 1, 1.0, SEED).unwrap();
        // A = [1, 0]  (1×2)
        // B = [0; 1]  (2×1)
        // ΔW = B @ A = [[0,0],[1,0]], scaling=1
        layer.a_matrix = vec![1.0, 0.0];
        layer.b_matrix = vec![0.0, 1.0];
        // input = [x, y] batch=1 → col vector [x; y]
        let input = vec![3.0f32, 5.0];
        let output = layer.apply(&input, 1).unwrap();
        // A @ input = [1*3 + 0*5] = [3]
        // B @ [3]   = [0*3; 1*3] = [0, 3]
        assert_eq!(output.len(), 2);
        assert!((output[0]).abs() < 1e-6, "expected 0, got {}", output[0]);
        assert!(
            (output[1] - 3.0).abs() < 1e-6,
            "expected 3, got {}",
            output[1]
        );
    }

    // --- LoraLayer::weight_delta ---

    #[test]
    fn test_weight_delta_shape() {
        let layer = LoraLayer::new("test", 8, 16, 4, 4.0, SEED).unwrap();
        let delta = layer.weight_delta();
        assert_eq!(
            delta.len(),
            16 * 8,
            "delta shape must be [out_dim × in_dim]"
        );
    }

    #[test]
    fn test_weight_delta_zero_b_is_zero() {
        let layer = LoraLayer::new("test", 8, 16, 4, 4.0, SEED).unwrap();
        let delta = layer.weight_delta();
        for &v in &delta {
            assert_eq!(v, 0.0);
        }
    }

    // --- LoraLayer::merge_into_weight ---

    #[test]
    fn test_merge_into_weight_correct() {
        let mut layer = LoraLayer::new("test", 2, 2, 1, 1.0, SEED).unwrap();
        layer.a_matrix = vec![1.0, 0.0];
        layer.b_matrix = vec![0.0, 1.0];
        let mut base = vec![1.0f32, 2.0, 3.0, 4.0]; // [2×2]
        layer.merge_into_weight(&mut base).unwrap();
        // delta = [[0,0],[1,0]]
        // base += delta → [[1,2],[4,4]]
        assert!((base[0] - 1.0).abs() < 1e-6);
        assert!((base[1] - 2.0).abs() < 1e-6);
        assert!((base[2] - 4.0).abs() < 1e-6);
        assert!((base[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_merge_into_weight_size_mismatch() {
        let layer = LoraLayer::new("test", 4, 8, 2, 2.0, SEED).unwrap();
        let mut base = vec![0.0f32; 10]; // wrong size
        let result = layer.merge_into_weight(&mut base);
        assert!(matches!(result, Err(LoraError::DimensionMismatch { .. })));
    }

    // --- LoraLayer::param_count and compression_ratio ---

    #[test]
    fn test_param_count() {
        let layer = LoraLayer::new("test", 64, 128, 4, 4.0, SEED).unwrap();
        assert_eq!(layer.param_count(), 4 * (64 + 128));
    }

    #[test]
    fn test_compression_ratio() {
        let layer = LoraLayer::new("test", 64, 128, 4, 4.0, SEED).unwrap();
        let expected = 4.0 * (64.0 + 128.0) / (64.0 * 128.0);
        assert!((layer.compression_ratio() - expected).abs() < 1e-6);
    }

    // --- LoraConfig ---

    #[test]
    fn test_config_validate_rank_zero() {
        let cfg = LoraConfig {
            rank: 0,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(LoraError::InvalidRank { .. })));
    }

    #[test]
    fn test_config_validate_negative_dropout() {
        let cfg = LoraConfig {
            dropout: -0.1,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(LoraError::InvalidConfig(_))));
    }

    #[test]
    fn test_config_validate_dropout_one() {
        let cfg = LoraConfig {
            dropout: 1.0,
            ..Default::default()
        };
        assert!(matches!(cfg.validate(), Err(LoraError::InvalidConfig(_))));
    }

    #[test]
    fn test_config_validate_valid() {
        assert!(LoraConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_with_rank() {
        let cfg = LoraConfig::with_rank(8);
        assert_eq!(cfg.rank, 8);
        assert!((cfg.alpha - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_avatar_preset() {
        let cfg = LoraConfig::avatar_preset();
        assert_eq!(cfg.rank, 8);
        assert!((cfg.alpha - 16.0).abs() < 1e-6);
        assert!(cfg.target_modules.contains(&"to_q".to_string()));
    }

    // --- LoraAdapter ---

    #[test]
    fn test_adapter_new_and_add_layer() {
        let mut adapter = LoraAdapter::new(LoraConfig::default()).unwrap();
        assert_eq!(adapter.num_layers(), 0);
        let layer = LoraLayer::new("unet.attn1.to_q", 64, 64, 4, 4.0, SEED).unwrap();
        adapter.add_layer(layer);
        assert_eq!(adapter.num_layers(), 1);
    }

    #[test]
    fn test_adapter_get_layer_found() {
        let mut adapter = LoraAdapter::new(LoraConfig::default()).unwrap();
        adapter.add_layer(LoraLayer::new("unet.to_q", 32, 32, 2, 2.0, SEED).unwrap());
        assert!(adapter.get_layer("unet.to_q").is_some());
    }

    #[test]
    fn test_adapter_get_layer_not_found() {
        let adapter = LoraAdapter::new(LoraConfig::default()).unwrap();
        assert!(adapter.get_layer("nonexistent").is_none());
    }

    #[test]
    fn test_adapter_total_params() {
        let mut adapter = LoraAdapter::new(LoraConfig::default()).unwrap();
        adapter.add_layer(LoraLayer::new("a", 64, 64, 4, 4.0, SEED).unwrap());
        adapter.add_layer(LoraLayer::new("b", 32, 32, 2, 2.0, SEED).unwrap());
        let expected = 4 * (64 + 64) + 2 * (32 + 32);
        assert_eq!(adapter.total_params(), expected);
    }

    // --- matmul ---

    #[test]
    fn test_matmul_identity() {
        // I @ v = v  (m=2, k=2, n=1)
        let a = vec![1.0f32, 0.0, 0.0, 1.0]; // 2×2 identity
        let b = vec![3.0f32, 7.0]; // 2×1
        let c = matmul(&a, &b, 2, 2, 1).unwrap();
        assert!((c[0] - 3.0).abs() < 1e-6);
        assert!((c[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_zero_matrix() {
        // a: 2×2 zero, b: 2×1 ones → c: 2×1 zeros
        let a = vec![0.0f32; 4];
        let b = vec![1.0f32; 2]; // k×n = 2×1
        let c = matmul(&a, &b, 2, 2, 1).unwrap();
        assert_eq!(c.len(), 2);
        for &v in &c {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_matmul_known_result() {
        // [[1,2],[3,4]] @ [[5],[6]] = [[17],[39]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0];
        let c = matmul(&a, &b, 2, 2, 1).unwrap();
        assert!((c[0] - 17.0).abs() < 1e-5);
        assert!((c[1] - 39.0).abs() < 1e-5);
    }

    #[test]
    fn test_matmul_dimension_error() {
        let a = vec![1.0f32; 3];
        let b = vec![1.0f32; 4];
        let result = matmul(&a, &b, 2, 2, 1); // a should be 4 elements
        assert!(matches!(result, Err(LoraError::DimensionMismatch { .. })));
    }

    // --- init_lora_a / init_lora_b ---

    #[test]
    fn test_init_lora_a_shape() {
        let a = init_lora_a(4, 64, SEED);
        assert_eq!(a.len(), 4 * 64);
    }

    #[test]
    fn test_init_lora_a_nonzero() {
        let a = init_lora_a(4, 64, SEED);
        let any_nonzero = a.iter().any(|&v| v != 0.0);
        assert!(any_nonzero, "A matrix should have non-zero values");
    }

    #[test]
    fn test_init_lora_b_all_zeros() {
        let b = init_lora_b(64, 4);
        assert_eq!(b.len(), 64 * 4);
        for &v in &b {
            assert_eq!(v, 0.0);
        }
    }

    // --- apply_lora_dropout ---

    #[test]
    fn test_dropout_rate_zero_unchanged() {
        let orig = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut v = orig.clone();
        apply_lora_dropout(&mut v, 0.0, SEED).unwrap();
        assert_eq!(v, orig);
    }

    #[test]
    fn test_dropout_rate_invalid() {
        let mut v = vec![1.0f32; 4];
        assert!(apply_lora_dropout(&mut v, 1.0, SEED).is_err());
        assert!(apply_lora_dropout(&mut v, -0.1, SEED).is_err());
    }

    #[test]
    fn test_dropout_large_rate_mostly_zero() {
        // With very high dropout (0.99) most values should be zeroed.
        let mut v = vec![1.0f32; 10000];
        apply_lora_dropout(&mut v, 0.99, SEED).unwrap();
        let zeros = v.iter().filter(|&&x| x == 0.0).count();
        // Expect roughly 9900 zeros; allow ±300 for randomness.
        assert!(
            zeros > 9000,
            "expected mostly zeros at 0.99 dropout, got {} zeros",
            zeros
        );
    }

    // --- lora_weight_norm ---

    #[test]
    fn test_lora_weight_norm_zero_b() {
        let layer = LoraLayer::new("test", 8, 16, 4, 4.0, SEED).unwrap();
        // B is initialised to zero so norm should be 0.
        let norm = lora_weight_norm(&layer);
        assert!(norm.abs() < 1e-8);
    }

    // --- merge_lora_weights ---

    #[test]
    fn test_merge_lora_weights_matched_and_unmatched() {
        let mut adapter = LoraAdapter::new(LoraConfig::default()).unwrap();
        let mut layer = LoraLayer::new("layer_a", 2, 2, 1, 1.0, SEED).unwrap();
        // Set non-trivial B so delta is non-zero.
        layer.a_matrix = vec![1.0, 0.0];
        layer.b_matrix = vec![1.0, 0.0];
        adapter.add_layer(layer);

        let mut weights = vec![
            ("layer_a".to_string(), vec![0.0f32; 4]),
            ("layer_b".to_string(), vec![5.0f32; 4]),
        ];
        let merged = merge_lora_weights(&mut weights, &adapter).unwrap();
        assert_eq!(merged, 1);
        // layer_b should be unchanged (no LoRA for it)
        for &v in &weights[1].1 {
            assert!((v - 5.0).abs() < 1e-6);
        }
    }

    // --- compute_lora_stats ---

    #[test]
    fn test_compute_lora_stats_n_layers() {
        let mut adapter = LoraAdapter::new(LoraConfig::default()).unwrap();
        adapter.add_layer(LoraLayer::new("a", 8, 8, 2, 2.0, SEED).unwrap());
        adapter.add_layer(LoraLayer::new("b", 16, 16, 4, 4.0, SEED).unwrap());
        let stats = compute_lora_stats(&adapter).unwrap();
        assert_eq!(stats.n_layers, 2);
    }

    #[test]
    fn test_compute_lora_stats_empty_adapter() {
        let adapter = LoraAdapter::new(LoraConfig::default()).unwrap();
        assert!(matches!(
            compute_lora_stats(&adapter),
            Err(LoraError::EmptyAdapter)
        ));
    }

    #[test]
    fn test_compute_lora_stats_norms_valid() {
        let mut adapter = LoraAdapter::new(LoraConfig::default()).unwrap();
        adapter.add_layer(LoraLayer::new("a", 8, 8, 2, 2.0, SEED).unwrap());
        let stats = compute_lora_stats(&adapter).unwrap();
        assert!(stats.mean_weight_norm >= 0.0);
        assert!(stats.max_weight_norm >= stats.min_weight_norm);
    }

    // --- serialize / deserialize round-trip ---

    #[test]
    fn test_serialize_deserialize_round_trip() {
        let mut adapter = LoraAdapter::new(LoraConfig::default()).unwrap();
        let layer_a = LoraLayer::new("a", 8, 16, 2, 2.0, SEED).unwrap();
        let layer_b = LoraLayer::new("b", 4, 8, 1, 1.0, SEED + 1).unwrap();
        adapter.add_layer(layer_a.clone());
        adapter.add_layer(layer_b.clone());

        let serialized = serialize_adapter(&adapter);
        let restored = deserialize_adapter(&serialized, LoraConfig::default()).unwrap();

        assert_eq!(restored.num_layers(), 2);
        let ra = restored.layers.first().unwrap();
        assert_eq!(ra.in_dim, layer_a.in_dim);
        assert_eq!(ra.out_dim, layer_a.out_dim);
        assert_eq!(ra.rank, layer_a.rank);
        // Check first A element is preserved.
        assert!((ra.a_matrix[0] - layer_a.a_matrix[0]).abs() < 1e-6);
    }

    #[test]
    fn test_deserialize_empty_error() {
        let result = deserialize_adapter(&[], LoraConfig::default());
        assert!(result.is_err());
    }

    // --- interpolate_adapters ---

    #[test]
    fn test_interpolate_t0_returns_adapter_a() {
        let mut a = LoraAdapter::new(LoraConfig::default()).unwrap();
        let mut layer_a = LoraLayer::new("l", 4, 4, 1, 1.0, SEED).unwrap();
        layer_a.a_matrix = vec![1.0, 2.0, 3.0, 4.0];
        a.add_layer(layer_a.clone());

        let mut b = LoraAdapter::new(LoraConfig::default()).unwrap();
        let mut layer_b = LoraLayer::new("l", 4, 4, 1, 1.0, SEED).unwrap();
        layer_b.a_matrix = vec![5.0, 6.0, 7.0, 8.0];
        b.add_layer(layer_b);

        let out = interpolate_adapters(&a, &b, 0.0).unwrap();
        let la = out.layers.first().unwrap();
        for (got, expected) in la.a_matrix.iter().zip(layer_a.a_matrix.iter()) {
            assert!((got - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_interpolate_t1_returns_adapter_b() {
        let mut a = LoraAdapter::new(LoraConfig::default()).unwrap();
        let mut layer_a = LoraLayer::new("l", 4, 4, 1, 1.0, SEED).unwrap();
        layer_a.a_matrix = vec![1.0, 2.0, 3.0, 4.0];
        a.add_layer(layer_a);

        let mut b = LoraAdapter::new(LoraConfig::default()).unwrap();
        let mut layer_b = LoraLayer::new("l", 4, 4, 1, 1.0, SEED).unwrap();
        layer_b.a_matrix = vec![5.0, 6.0, 7.0, 8.0];
        b.add_layer(layer_b.clone());

        let out = interpolate_adapters(&a, &b, 1.0).unwrap();
        let la = out.layers.first().unwrap();
        for (got, expected) in la.a_matrix.iter().zip(layer_b.a_matrix.iter()) {
            assert!((got - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_interpolate_mismatched_layers_error() {
        let mut a = LoraAdapter::new(LoraConfig::default()).unwrap();
        a.add_layer(LoraLayer::new("l1", 4, 4, 1, 1.0, SEED).unwrap());

        let mut b = LoraAdapter::new(LoraConfig::default()).unwrap();
        b.add_layer(LoraLayer::new("l1", 4, 4, 1, 1.0, SEED).unwrap());
        b.add_layer(LoraLayer::new("l2", 4, 4, 1, 1.0, SEED).unwrap());

        let result = interpolate_adapters(&a, &b, 0.5);
        assert!(matches!(result, Err(LoraError::DimensionMismatch { .. })));
    }

    // --- lora_backward ---

    #[test]
    fn test_lora_backward_output_shapes() {
        let in_dim = 8;
        let out_dim = 16;
        let rank = 2;
        let batch = 4;
        let layer = LoraLayer::new("test", in_dim, out_dim, rank, 1.0, SEED).unwrap();

        let input = vec![0.5f32; in_dim * batch];
        let grad_output = vec![0.1f32; out_dim * batch];
        let (grad_a, grad_b) = lora_backward(&layer, &input, &grad_output, batch).unwrap();

        assert_eq!(grad_a.len(), rank * in_dim, "grad_a shape mismatch");
        assert_eq!(grad_b.len(), out_dim * rank, "grad_b shape mismatch");
    }

    #[test]
    fn test_lora_backward_dimension_error() {
        let layer = LoraLayer::new("test", 8, 16, 2, 1.0, SEED).unwrap();
        let bad_input = vec![0.0f32; 3]; // wrong
        let grad = vec![0.0f32; 16 * 4];
        let result = lora_backward(&layer, &bad_input, &grad, 4);
        assert!(matches!(result, Err(LoraError::DimensionMismatch { .. })));
    }

    // --- lora_sgd_step ---

    #[test]
    fn test_lora_sgd_step_params_change() {
        let mut layer = LoraLayer::new("test", 4, 4, 1, 1.0, SEED).unwrap();
        let a_before = layer.a_matrix.clone();
        let b_before = layer.b_matrix.clone();

        let grad_a = vec![1.0f32; layer.a_matrix.len()];
        let grad_b = vec![0.5f32; layer.b_matrix.len()];
        lora_sgd_step(&mut layer, &grad_a, &grad_b, 0.01).unwrap();

        for (before, after) in a_before.iter().zip(layer.a_matrix.iter()) {
            assert!((after - (before - 0.01)).abs() < 1e-7);
        }
        for (before, after) in b_before.iter().zip(layer.b_matrix.iter()) {
            assert!((after - (before - 0.005)).abs() < 1e-7);
        }
    }

    #[test]
    fn test_lora_sgd_step_size_mismatch() {
        let mut layer = LoraLayer::new("test", 4, 4, 1, 1.0, SEED).unwrap();
        let grad_a = vec![1.0f32; 999]; // wrong
        let grad_b = vec![0.0f32; layer.b_matrix.len()];
        let result = lora_sgd_step(&mut layer, &grad_a, &grad_b, 0.01);
        assert!(matches!(result, Err(LoraError::DimensionMismatch { .. })));
    }

    // --- Layer names ---

    #[test]
    fn test_adapter_layer_names() {
        let mut adapter = LoraAdapter::new(LoraConfig::default()).unwrap();
        adapter.add_layer(LoraLayer::new("q", 8, 8, 2, 2.0, SEED).unwrap());
        adapter.add_layer(LoraLayer::new("k", 8, 8, 2, 2.0, SEED).unwrap());
        let names = adapter.layer_names();
        assert!(names.contains(&"q"));
        assert!(names.contains(&"k"));
    }

    // --- set_scale ---

    #[test]
    fn test_adapter_set_scale() {
        let mut adapter = LoraAdapter::new(LoraConfig::default()).unwrap();
        adapter.add_layer(LoraLayer::new("l", 8, 8, 2, 2.0, SEED).unwrap());
        adapter.set_scale(0.5);
        let layer = adapter.layers.first().unwrap();
        // alpha = rank * scale = 2 * 0.5 = 1.0
        assert!((layer.alpha - 1.0).abs() < 1e-6);
    }

    // --- Compression ratio bounds ---

    #[test]
    fn test_compression_ratio_less_than_one_for_low_rank() {
        // For rank=1, in=64, out=64: params=128, full=4096 → ratio≈0.03
        let layer = LoraLayer::new("test", 64, 64, 1, 1.0, SEED).unwrap();
        assert!(layer.compression_ratio() < 1.0);
    }
}
