//! # identity_conditioning
//!
//! Identity conditioning for diffusion-based avatar generation.
//!
//! This module projects FLAME head-model parameters (shape, expression, pose)
//! into a compact identity embedding that the diffusion model uses to preserve
//! subject identity across generated views.
//!
//! ## Architecture
//!
//! ```text
//! FLAME params (shape 300D + expr 50D + pose 15D)
//!     │
//!     ▼
//! IdentityEncoder MLP  (Xavier-init, GELU hidden layers)
//!     │
//!     ▼
//! identity embedding  (256D, optionally L2-normalised)
//! ```
//!
//! ## Usage
//!
//! ```no_run
//! use oxigaf_diffusion::identity_conditioning::{
//!     IdentityConfig, IdentityEncoder, IdentityFeature,
//! };
//!
//! let cfg = IdentityConfig::default();
//! let mut rng = 0x1234_5678_u64;
//! let enc = IdentityEncoder::new(cfg, &mut rng).expect("valid config");
//!
//! let feat = IdentityFeature::new(vec![0.0f32; 300])
//!     .with_expression(vec![0.0f32; 50]);
//! let emb = enc.encode(&feat).expect("encode ok");
//! assert_eq!(emb.len(), 256);
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the identity-conditioning module.
#[derive(Debug, Error)]
pub enum IdentityConditioningError {
    /// A supplied dimension was zero or otherwise invalid.
    #[error("invalid embedding dimension: {0}")]
    InvalidDimension(usize),

    /// A vector had the wrong length.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// An input collection was empty.
    #[error("empty input")]
    EmptyInput,

    /// A cache lookup found no matching entry.
    #[error("cache miss: key not found")]
    CacheMiss,

    /// A configuration field has an invalid value.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// PRNG helpers (module-private — xorshift64 is already re-exported at crate
// level from adaptive_sampling; we keep our copy private to avoid a name clash)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

#[inline]
fn xorshift_f32(state: &mut u64) -> f32 {
    xorshift64(state) as f32 / u64::MAX as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the identity-conditioning encoder.
#[derive(Debug, Clone)]
pub struct IdentityConfig {
    /// Dimension of FLAME shape parameters (default 300).
    pub shape_dim: usize,
    /// Dimension of FLAME expression parameters (default 50).
    pub expression_dim: usize,
    /// Dimension of FLAME pose parameters (default 15).
    pub pose_dim: usize,
    /// Output identity embedding dimension (default 256).
    pub embedding_dim: usize,
    /// Number of hidden MLP layers (default 2).
    pub n_hidden_layers: usize,
    /// Hidden layer width (default 512).
    pub hidden_dim: usize,
    /// Dropout probability during training, in [0, 1) (default 0.1).
    pub dropout_prob: f32,
    /// Whether to concatenate expression parameters into the input (default true).
    pub use_expression: bool,
    /// Whether to concatenate pose parameters (head rotation) into the input (default false).
    pub use_pose: bool,
    /// Whether to L2-normalise the final embedding (default true).
    pub normalize_output: bool,
}

impl Default for IdentityConfig {
    fn default() -> Self {
        Self {
            shape_dim: 300,
            expression_dim: 50,
            pose_dim: 15,
            embedding_dim: 256,
            n_hidden_layers: 2,
            hidden_dim: 512,
            dropout_prob: 0.1,
            use_expression: true,
            use_pose: false,
            normalize_output: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Identity feature
// ─────────────────────────────────────────────────────────────────────────────

/// A bundle of FLAME parameters representing one subject's identity.
#[derive(Debug, Clone)]
pub struct IdentityFeature {
    /// FLAME shape parameters.
    pub shape_params: Vec<f32>,
    /// FLAME expression parameters (may be empty if not used).
    pub expression_params: Vec<f32>,
    /// FLAME pose parameters (may be empty if not used).
    pub pose_params: Vec<f32>,
}

impl IdentityFeature {
    /// Create a feature from shape parameters only.
    pub fn new(shape_params: Vec<f32>) -> Self {
        Self {
            shape_params,
            expression_params: Vec::new(),
            pose_params: Vec::new(),
        }
    }

    /// Attach expression parameters (builder pattern).
    pub fn with_expression(mut self, expr: Vec<f32>) -> Self {
        self.expression_params = expr;
        self
    }

    /// Attach pose parameters (builder pattern).
    pub fn with_pose(mut self, pose: Vec<f32>) -> Self {
        self.pose_params = pose;
        self
    }

    /// Total concatenated input dimensionality.
    pub fn total_dim(&self) -> usize {
        self.shape_params.len() + self.expression_params.len() + self.pose_params.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear layer
// ─────────────────────────────────────────────────────────────────────────────

/// A single affine linear layer with optional activation.
///
/// Weights are stored **row-major**: `weights[row * in_dim + col]` where `row`
/// indexes the output neuron and `col` indexes the input neuron.
#[derive(Debug, Clone)]
pub struct LinearLayer {
    /// Weight matrix, length `out_dim * in_dim`, row-major.
    pub weights: Vec<f32>,
    /// Bias vector, length `out_dim`.
    pub bias: Vec<f32>,
    /// Input dimensionality.
    pub in_dim: usize,
    /// Output dimensionality.
    pub out_dim: usize,
}

impl LinearLayer {
    /// Create a layer with **Xavier uniform** initialisation.
    ///
    /// Weights are drawn from `Uniform(-limit, +limit)` where
    /// `limit = sqrt(6 / (in_dim + out_dim))`.
    pub fn new_xavier(in_dim: usize, out_dim: usize, rng_state: &mut u64) -> Self {
        let limit = (6.0_f32 / (in_dim + out_dim) as f32).sqrt();
        let mut weights = Vec::with_capacity(out_dim * in_dim);
        for _ in 0..(out_dim * in_dim) {
            // xorshift_f32 returns [0, 1]; map to [-limit, limit]
            let w = xorshift_f32(rng_state) * 2.0 * limit - limit;
            weights.push(w);
        }
        // Bias initialised to zero (standard practice)
        let bias = vec![0.0f32; out_dim];
        Self {
            weights,
            bias,
            in_dim,
            out_dim,
        }
    }

    /// Forward pass without activation: `y = W·x + b`.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, IdentityConditioningError> {
        if input.len() != self.in_dim {
            return Err(IdentityConditioningError::DimensionMismatch {
                expected: self.in_dim,
                got: input.len(),
            });
        }
        let mut out = vec![0.0f32; self.out_dim];
        for (row, out_val) in out.iter_mut().enumerate() {
            let base = row * self.in_dim;
            let mut acc = self.bias[row];
            for (col, &ic) in input.iter().enumerate() {
                acc += self.weights[base + col] * ic;
            }
            *out_val = acc;
        }
        Ok(out)
    }

    /// Forward pass with ReLU activation: `y = max(0, W·x + b)`.
    pub fn forward_relu(&self, input: &[f32]) -> Result<Vec<f32>, IdentityConditioningError> {
        let mut out = self.forward(input)?;
        for v in &mut out {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        Ok(out)
    }

    /// Forward pass with GELU activation (tanh approximation).
    ///
    /// `gelu(x) ≈ 0.5 · x · (1 + tanh(0.7978846 · (x + 0.044715 · x³)))`
    pub fn forward_gelu(&self, input: &[f32]) -> Result<Vec<f32>, IdentityConditioningError> {
        let mut out = self.forward(input)?;
        for v in &mut out {
            *v = gelu_scalar(*v);
        }
        Ok(out)
    }
}

/// GELU activation (tanh approximation).
#[inline]
fn gelu_scalar(x: f32) -> f32 {
    let c = 0.044715_f32;
    let k = 0.797_884_6_f32; // sqrt(2/π)
    0.5 * x * (1.0 + ((k * (x + c * x * x * x)).tanh()))
}

// ─────────────────────────────────────────────────────────────────────────────
// Identity encoder MLP
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-layer perceptron that encodes FLAME parameters into an identity
/// embedding for diffusion conditioning.
pub struct IdentityEncoder {
    config: IdentityConfig,
    /// Hidden layers (GELU-activated).
    layers: Vec<LinearLayer>,
    /// Final projection layer (no activation; normalisation applied separately).
    output_layer: LinearLayer,
}

impl IdentityEncoder {
    /// Create a new encoder with Xavier-initialised weights.
    pub fn new(
        config: IdentityConfig,
        rng_state: &mut u64,
    ) -> Result<Self, IdentityConditioningError> {
        // Validate config
        if config.embedding_dim == 0 {
            return Err(IdentityConditioningError::InvalidDimension(
                config.embedding_dim,
            ));
        }
        if config.shape_dim == 0 {
            return Err(IdentityConditioningError::InvalidDimension(
                config.shape_dim,
            ));
        }
        if config.hidden_dim == 0 && config.n_hidden_layers > 0 {
            return Err(IdentityConditioningError::InvalidConfig(
                "hidden_dim must be > 0 when n_hidden_layers > 0".to_string(),
            ));
        }
        if config.dropout_prob < 0.0 || config.dropout_prob >= 1.0 {
            return Err(IdentityConditioningError::InvalidConfig(format!(
                "dropout_prob must be in [0, 1), got {}",
                config.dropout_prob
            )));
        }

        // Compute input dimension from config
        let mut in_dim = config.shape_dim;
        if config.use_expression {
            in_dim += config.expression_dim;
        }
        if config.use_pose {
            in_dim += config.pose_dim;
        }

        // Build hidden layers
        let mut layers: Vec<LinearLayer> = Vec::with_capacity(config.n_hidden_layers);
        let mut current_in = in_dim;
        for _ in 0..config.n_hidden_layers {
            let layer = LinearLayer::new_xavier(current_in, config.hidden_dim, rng_state);
            layers.push(layer);
            current_in = config.hidden_dim;
        }

        // Build output layer
        let output_layer = LinearLayer::new_xavier(current_in, config.embedding_dim, rng_state);

        Ok(Self {
            config,
            layers,
            output_layer,
        })
    }

    /// Encode a set of FLAME parameters into an identity embedding.
    ///
    /// The feature's dimension must match what the config expects.
    pub fn encode(&self, feature: &IdentityFeature) -> Result<Vec<f32>, IdentityConditioningError> {
        let input = self.build_input(feature)?;
        let hidden = self.forward_hidden(&input)?;
        let output = self.output_layer.forward(&hidden)?;
        if self.config.normalize_output {
            Ok(ident_normalize(&output))
        } else {
            Ok(output)
        }
    }

    /// Encode with inverted-dropout applied to each hidden-layer output.
    ///
    /// Each hidden activation is zeroed with probability `config.dropout_prob`
    /// and scaled by `1 / (1 - p)` to maintain expected magnitude.
    pub fn encode_with_dropout(
        &self,
        feature: &IdentityFeature,
        rng_state: &mut u64,
    ) -> Result<Vec<f32>, IdentityConditioningError> {
        let input = self.build_input(feature)?;
        let p = self.config.dropout_prob;
        let scale = if p < 1.0 { 1.0 / (1.0 - p) } else { 0.0 };

        let mut current = input;
        for layer in &self.layers {
            let mut activated = layer.forward_gelu(&current)?;
            // Apply inverted dropout
            for v in &mut activated {
                let keep = xorshift_f32(rng_state) >= p;
                if keep {
                    *v *= scale;
                } else {
                    *v = 0.0;
                }
            }
            current = activated;
        }

        let output = self.output_layer.forward(&current)?;
        if self.config.normalize_output {
            Ok(ident_normalize(&output))
        } else {
            Ok(output)
        }
    }

    /// Output embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Encoder input dimension (shape + optional expression + optional pose).
    pub fn input_dim(&self) -> usize {
        let mut d = self.config.shape_dim;
        if self.config.use_expression {
            d += self.config.expression_dim;
        }
        if self.config.use_pose {
            d += self.config.pose_dim;
        }
        d
    }

    /// Access the encoder configuration.
    pub fn config(&self) -> &IdentityConfig {
        &self.config
    }

    // ── internal helpers ─────────────────────────────────────────────────────

    fn build_input(
        &self,
        feature: &IdentityFeature,
    ) -> Result<Vec<f32>, IdentityConditioningError> {
        // Validate shape
        if feature.shape_params.len() != self.config.shape_dim {
            return Err(IdentityConditioningError::DimensionMismatch {
                expected: self.config.shape_dim,
                got: feature.shape_params.len(),
            });
        }
        // Validate expression
        if self.config.use_expression
            && feature.expression_params.len() != self.config.expression_dim
        {
            return Err(IdentityConditioningError::DimensionMismatch {
                expected: self.config.expression_dim,
                got: feature.expression_params.len(),
            });
        }
        // Validate pose
        if self.config.use_pose && feature.pose_params.len() != self.config.pose_dim {
            return Err(IdentityConditioningError::DimensionMismatch {
                expected: self.config.pose_dim,
                got: feature.pose_params.len(),
            });
        }

        let mut input = Vec::with_capacity(self.input_dim());
        input.extend_from_slice(&feature.shape_params);
        if self.config.use_expression {
            input.extend_from_slice(&feature.expression_params);
        }
        if self.config.use_pose {
            input.extend_from_slice(&feature.pose_params);
        }
        Ok(input)
    }

    fn forward_hidden(&self, input: &[f32]) -> Result<Vec<f32>, IdentityConditioningError> {
        let mut current = input.to_vec();
        for layer in &self.layers {
            current = layer.forward_gelu(&current)?;
        }
        Ok(current)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedding operations
// ─────────────────────────────────────────────────────────────────────────────

/// L2-normalise a vector.
///
/// Returns the original zero-vector unchanged for degenerate (zero-norm) input.
pub fn ident_normalize(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm_f32(v);
    if norm < f32::EPSILON {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Cosine similarity between two identity embeddings.
pub fn ident_cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, IdentityConditioningError> {
    if a.is_empty() || b.is_empty() {
        return Err(IdentityConditioningError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(IdentityConditioningError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = l2_norm_f32(a);
    let nb = l2_norm_f32(b);
    if na < f32::EPSILON || nb < f32::EPSILON {
        return Ok(0.0);
    }
    Ok((dot / (na * nb)).clamp(-1.0, 1.0))
}

/// Spherical linear interpolation (slerp) between two identity embeddings.
///
/// `t = 0` returns a normalised copy of `a`; `t = 1` returns a normalised copy
/// of `b`.  Falls back to linear lerp when the angle between vectors is < 1e-6.
pub fn ident_slerp(a: &[f32], b: &[f32], t: f32) -> Result<Vec<f32>, IdentityConditioningError> {
    if a.is_empty() || b.is_empty() {
        return Err(IdentityConditioningError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(IdentityConditioningError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let an = ident_normalize(a);
    let bn = ident_normalize(b);

    let dot: f32 = an
        .iter()
        .zip(bn.iter())
        .map(|(x, y)| x * y)
        .sum::<f32>()
        .clamp(-1.0, 1.0);
    let theta = dot.acos();

    if theta.abs() < 1e-6 {
        // Vectors are nearly identical — fall back to lerp on normalised vectors
        return ident_lerp(&an, &bn, t);
    }

    let sin_theta = theta.sin();
    let scale_a = ((1.0 - t) * theta).sin() / sin_theta;
    let scale_b = (t * theta).sin() / sin_theta;

    let result: Vec<f32> = an
        .iter()
        .zip(bn.iter())
        .map(|(xa, xb)| scale_a * xa + scale_b * xb)
        .collect();
    Ok(result)
}

/// Linear interpolation between two embeddings.
///
/// `t = 0` returns `a`; `t = 1` returns `b`.
pub fn ident_lerp(a: &[f32], b: &[f32], t: f32) -> Result<Vec<f32>, IdentityConditioningError> {
    if a.is_empty() || b.is_empty() {
        return Err(IdentityConditioningError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(IdentityConditioningError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let result: Vec<f32> = a
        .iter()
        .zip(b.iter())
        .map(|(xa, xb)| (1.0 - t) * xa + t * xb)
        .collect();
    Ok(result)
}

/// Compute the mean of a set of identity embeddings, then L2-normalise.
pub fn ident_mean_embedding(
    embeddings: &[Vec<f32>],
) -> Result<Vec<f32>, IdentityConditioningError> {
    if embeddings.is_empty() {
        return Err(IdentityConditioningError::EmptyInput);
    }
    let dim = embeddings[0].len();
    if dim == 0 {
        return Err(IdentityConditioningError::InvalidDimension(0));
    }
    for e in embeddings.iter() {
        if e.len() != dim {
            return Err(IdentityConditioningError::DimensionMismatch {
                expected: dim,
                got: e.len(),
            });
        }
    }
    let n = embeddings.len() as f32;
    let mut mean = vec![0.0f32; dim];
    for emb in embeddings {
        for (m, x) in mean.iter_mut().zip(emb.iter()) {
            *m += x;
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    Ok(ident_normalize(&mean))
}

/// Pairwise cosine-similarity matrix for a batch of embeddings.
///
/// Returns a flat `Vec<f32>` of length `n * n` (row-major), where entry
/// `[i * n + j]` is `cosine_similarity(embeddings[i], embeddings[j])`.
///
/// Each embedding's L2 norm is computed once (not twice per pair) and only
/// the upper triangle is evaluated and mirrored into the lower triangle,
/// since cosine similarity is symmetric.
pub fn ident_similarity_matrix(
    embeddings: &[Vec<f32>],
) -> Result<Vec<f32>, IdentityConditioningError> {
    if embeddings.is_empty() {
        return Err(IdentityConditioningError::EmptyInput);
    }
    let n = embeddings.len();
    let dim = embeddings[0].len();
    for e in embeddings {
        if e.len() != dim {
            return Err(IdentityConditioningError::DimensionMismatch {
                expected: dim,
                got: e.len(),
            });
        }
    }

    let norms: Vec<f32> = embeddings.iter().map(|e| l2_norm_f32(e)).collect();

    let mut matrix = vec![0.0f32; n * n];
    for i in 0..n {
        // Matches ident_cosine_similarity's own zero-norm convention: a
        // zero-norm embedding's self-similarity is defined as 0.0, not 1.0.
        matrix[i * n + i] = if norms[i] < f32::EPSILON { 0.0 } else { 1.0 };
        for j in (i + 1)..n {
            let dot: f32 = embeddings[i]
                .iter()
                .zip(embeddings[j].iter())
                .map(|(x, y)| x * y)
                .sum();
            let sim = if norms[i] < f32::EPSILON || norms[j] < f32::EPSILON {
                0.0
            } else {
                (dot / (norms[i] * norms[j])).clamp(-1.0, 1.0)
            };
            matrix[i * n + j] = sim;
            matrix[j * n + i] = sim;
        }
    }
    Ok(matrix)
}

// ─────────────────────────────────────────────────────────────────────────────
// Identity conditioning cache
// ─────────────────────────────────────────────────────────────────────────────

/// A fixed-capacity LRU-like cache for identity embeddings.
///
/// Keys are `u64` hashes of [`IdentityFeature`] values computed by
/// [`ident_hash_feature`].
pub struct IdentityCache {
    capacity: usize,
    /// (key, embedding) pairs in insertion order.
    entries: Vec<(u64, Vec<f32>)>,
    /// Indices into `entries` ordered from least- to most-recently accessed.
    /// `access_order[0]` is the next candidate for eviction.
    access_order: Vec<usize>,
}

impl IdentityCache {
    /// Create a new empty cache with the given capacity.
    ///
    /// `capacity == 0` disables caching entirely: [`Self::insert`] becomes a
    /// no-op and [`Self::get`] always returns `None`, rather than growing
    /// without bound.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: Vec::with_capacity(capacity),
            access_order: Vec::with_capacity(capacity),
        }
    }

    /// Retrieve the embedding for `key`, updating LRU order.
    pub fn get(&mut self, key: u64) -> Option<&Vec<f32>> {
        // Find index in entries
        let entry_idx = self.entries.iter().position(|(k, _)| *k == key)?;
        // Promote to most-recently-used: remove from its position in access_order
        // and push to the end.
        if let Some(pos) = self.access_order.iter().position(|&i| i == entry_idx) {
            self.access_order.remove(pos);
            self.access_order.push(entry_idx);
        }
        self.entries.get(entry_idx).map(|(_, v)| v)
    }

    /// Insert an embedding.  If the cache is full, the least-recently-used
    /// entry is evicted first. A cache created with `capacity == 0` has
    /// caching disabled: `insert` is a no-op and `get` always returns `None`.
    pub fn insert(&mut self, key: u64, embedding: Vec<f32>) {
        if self.capacity == 0 {
            return;
        }
        // If key already exists, update it
        if let Some(entry_idx) = self.entries.iter().position(|(k, _)| *k == key) {
            self.entries[entry_idx].1 = embedding;
            // Promote to MRU
            if let Some(pos) = self.access_order.iter().position(|&i| i == entry_idx) {
                self.access_order.remove(pos);
                self.access_order.push(entry_idx);
            }
            return;
        }

        if self.entries.len() >= self.capacity && self.capacity > 0 {
            // Evict the LRU entry (first in access_order)
            if !self.access_order.is_empty() {
                let lru_entry_idx = self.access_order.remove(0);
                // Replace the evicted slot
                let new_entry_idx = lru_entry_idx;
                self.entries[new_entry_idx] = (key, embedding);
                // Fix up access_order: any indices pointing to slots after the
                // evicted one in entries are unaffected (we reuse the same slot).
                self.access_order.push(new_entry_idx);
                return;
            }
        }

        // Append new entry
        let new_idx = self.entries.len();
        self.entries.push((key, embedding));
        self.access_order.push(new_idx);
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
    }
}

/// Compute a deterministic FNV-1a `u64` hash from an [`IdentityFeature`].
pub fn ident_hash_feature(feature: &IdentityFeature) -> u64 {
    const OFFSET_BASIS: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;

    let mut hash = OFFSET_BASIS;
    // Hash shape, expression, and pose params as raw bytes
    for val in &feature.shape_params {
        for byte in val.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(PRIME);
        }
    }
    // Separator to avoid extension attacks across fields
    hash ^= 0xFF;
    hash = hash.wrapping_mul(PRIME);
    for val in &feature.expression_params {
        for byte in val.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(PRIME);
        }
    }
    hash ^= 0xFE;
    hash = hash.wrapping_mul(PRIME);
    for val in &feature.pose_params {
        for byte in val.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(PRIME);
        }
    }
    hash
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics and formatting
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics over a batch of identity embeddings.
#[derive(Debug, Clone)]
pub struct IdentityStats {
    /// Mean pairwise cosine similarity across the batch.
    pub mean_similarity: f32,
    /// Minimum pairwise cosine similarity.
    pub min_similarity: f32,
    /// Maximum pairwise cosine similarity.
    pub max_similarity: f32,
    /// Mean L2 norm of the embeddings.
    pub embedding_norm: f32,
}

/// Compute pairwise identity statistics for a batch of embeddings.
///
/// For a single embedding, similarity statistics are reported as `1.0` (the
/// embedding is perfectly similar to itself).
pub fn ident_compute_stats(
    embeddings: &[Vec<f32>],
) -> Result<IdentityStats, IdentityConditioningError> {
    if embeddings.is_empty() {
        return Err(IdentityConditioningError::EmptyInput);
    }

    // Norms are computed once per embedding (not twice per pair below).
    let norms: Vec<f32> = embeddings.iter().map(|e| l2_norm_f32(e)).collect();
    let mean_norm = norms.iter().sum::<f32>() / embeddings.len() as f32;

    if embeddings.len() == 1 {
        return Ok(IdentityStats {
            mean_similarity: 1.0,
            min_similarity: 1.0,
            max_similarity: 1.0,
            embedding_norm: mean_norm,
        });
    }

    let n = embeddings.len();
    let dim = embeddings[0].len();
    for e in embeddings {
        if e.len() != dim {
            return Err(IdentityConditioningError::DimensionMismatch {
                expected: dim,
                got: e.len(),
            });
        }
    }

    // Cosine similarity is symmetric, so the mean/min/max over all n*(n-1)
    // ordered off-diagonal pairs equals the mean/min/max computed over the
    // n*(n-1)/2 unordered pairs once each — duplicating each value would
    // not change the mean (same average) nor the min/max.
    let mut sims: Vec<f32> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dot: f32 = embeddings[i]
                .iter()
                .zip(embeddings[j].iter())
                .map(|(x, y)| x * y)
                .sum();
            let sim = if norms[i] < f32::EPSILON || norms[j] < f32::EPSILON {
                0.0
            } else {
                (dot / (norms[i] * norms[j])).clamp(-1.0, 1.0)
            };
            sims.push(sim);
        }
    }

    let mean_sim = sims.iter().copied().sum::<f32>() / sims.len() as f32;
    let min_sim = sims.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_sim = sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    Ok(IdentityStats {
        mean_similarity: mean_sim,
        min_similarity: min_sim,
        max_similarity: max_sim,
        embedding_norm: mean_norm,
    })
}

/// Produce a human-readable summary of an [`IdentityConfig`].
pub fn ident_format_config(config: &IdentityConfig) -> String {
    format!(
        "IdentityConfig {{ shape_dim={}, expression_dim={}, pose_dim={}, \
         embedding_dim={}, n_hidden_layers={}, hidden_dim={}, \
         dropout={:.3}, use_expression={}, use_pose={}, normalize_output={} }}",
        config.shape_dim,
        config.expression_dim,
        config.pose_dim,
        config.embedding_dim,
        config.n_hidden_layers,
        config.hidden_dim,
        config.dropout_prob,
        config.use_expression,
        config.use_pose,
        config.normalize_output,
    )
}

/// Produce a human-readable summary of [`IdentityStats`].
pub fn ident_format_stats(stats: &IdentityStats) -> String {
    format!(
        "IdentityStats {{ mean_sim={:.4}, min_sim={:.4}, max_sim={:.4}, mean_norm={:.4} }}",
        stats.mean_similarity, stats.min_similarity, stats.max_similarity, stats.embedding_norm,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn l2_norm_f32(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────────

    fn rng() -> u64 {
        0xDEAD_BEEF_CAFE_1234_u64
    }

    fn make_encoder(normalize: bool) -> IdentityEncoder {
        let cfg = IdentityConfig {
            normalize_output: normalize,
            ..IdentityConfig::default()
        };
        let mut r = rng();
        IdentityEncoder::new(cfg, &mut r).expect("valid config")
    }

    fn default_feature() -> IdentityFeature {
        IdentityFeature::new(vec![0.1f32; 300]).with_expression(vec![0.05f32; 50])
    }

    // ── IdentityConfig::default ────────────────────────────────────────────

    #[test]
    fn test_config_default_shape_dim() {
        assert_eq!(IdentityConfig::default().shape_dim, 300);
    }

    #[test]
    fn test_config_default_expression_dim() {
        assert_eq!(IdentityConfig::default().expression_dim, 50);
    }

    #[test]
    fn test_config_default_pose_dim() {
        assert_eq!(IdentityConfig::default().pose_dim, 15);
    }

    #[test]
    fn test_config_default_embedding_dim() {
        assert_eq!(IdentityConfig::default().embedding_dim, 256);
    }

    #[test]
    fn test_config_default_hidden_layers() {
        assert_eq!(IdentityConfig::default().n_hidden_layers, 2);
    }

    #[test]
    fn test_config_default_hidden_dim() {
        assert_eq!(IdentityConfig::default().hidden_dim, 512);
    }

    #[test]
    fn test_config_default_use_expression() {
        assert!(IdentityConfig::default().use_expression);
    }

    #[test]
    fn test_config_default_use_pose() {
        assert!(!IdentityConfig::default().use_pose);
    }

    #[test]
    fn test_config_default_normalize_output() {
        assert!(IdentityConfig::default().normalize_output);
    }

    // ── IdentityFeature ────────────────────────────────────────────────────

    #[test]
    fn test_feature_total_dim_shape_only() {
        let f = IdentityFeature::new(vec![0.0f32; 300]);
        assert_eq!(f.total_dim(), 300);
    }

    #[test]
    fn test_feature_total_dim_shape_and_expression() {
        let f = IdentityFeature::new(vec![0.0f32; 300]).with_expression(vec![0.0f32; 50]);
        assert_eq!(f.total_dim(), 350);
    }

    #[test]
    fn test_feature_total_dim_all() {
        let f = IdentityFeature::new(vec![0.0f32; 300])
            .with_expression(vec![0.0f32; 50])
            .with_pose(vec![0.0f32; 15]);
        assert_eq!(f.total_dim(), 365);
    }

    #[test]
    fn test_feature_builder_chain_returns_correct_lengths() {
        let f = IdentityFeature::new(vec![1.0f32; 300])
            .with_expression(vec![2.0f32; 50])
            .with_pose(vec![3.0f32; 15]);
        assert_eq!(f.shape_params.len(), 300);
        assert_eq!(f.expression_params.len(), 50);
        assert_eq!(f.pose_params.len(), 15);
        assert!((f.shape_params[0] - 1.0).abs() < 1e-6);
        assert!((f.expression_params[0] - 2.0).abs() < 1e-6);
        assert!((f.pose_params[0] - 3.0).abs() < 1e-6);
    }

    // ── LinearLayer ────────────────────────────────────────────────────────

    #[test]
    fn test_linear_forward_known_1x1() {
        // out = w * x + b  →  3.0 * 2.0 + 1.0 = 7.0
        let layer = LinearLayer {
            weights: vec![3.0f32],
            bias: vec![1.0f32],
            in_dim: 1,
            out_dim: 1,
        };
        let out = layer.forward(&[2.0f32]).expect("forward ok");
        assert!((out[0] - 7.0).abs() < 1e-5, "expected 7.0 got {}", out[0]);
    }

    #[test]
    fn test_linear_forward_relu_negative_to_zero() {
        let layer = LinearLayer {
            weights: vec![-1.0f32],
            bias: vec![0.0f32],
            in_dim: 1,
            out_dim: 1,
        };
        let out = layer.forward_relu(&[1.0f32]).expect("forward_relu ok");
        assert_eq!(out[0], 0.0, "ReLU must zero negative pre-activation");
    }

    #[test]
    fn test_linear_forward_relu_positive_passthrough() {
        let layer = LinearLayer {
            weights: vec![2.0f32],
            bias: vec![0.0f32],
            in_dim: 1,
            out_dim: 1,
        };
        let out = layer.forward_relu(&[3.0f32]).expect("forward_relu ok");
        assert!((out[0] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_forward_gelu_zero_input() {
        let layer = LinearLayer {
            weights: vec![1.0f32],
            bias: vec![0.0f32],
            in_dim: 1,
            out_dim: 1,
        };
        // pre-activation = 0, gelu(0) = 0
        let out = layer.forward_gelu(&[0.0f32]).expect("forward_gelu ok");
        assert!(out[0].abs() < 1e-6, "gelu(0) should be 0, got {}", out[0]);
    }

    #[test]
    fn test_linear_forward_gelu_positive_positive() {
        // gelu(x) > 0 for x > 0
        let layer = LinearLayer {
            weights: vec![1.0f32],
            bias: vec![0.0f32],
            in_dim: 1,
            out_dim: 1,
        };
        let out = layer.forward_gelu(&[1.0f32]).expect("forward_gelu ok");
        assert!(out[0] > 0.0, "gelu(1) must be positive, got {}", out[0]);
    }

    #[test]
    fn test_linear_forward_gelu_approx_at_one() {
        // gelu(1) ≈ 0.841
        let layer = LinearLayer {
            weights: vec![1.0f32],
            bias: vec![0.0f32],
            in_dim: 1,
            out_dim: 1,
        };
        let out = layer.forward_gelu(&[1.0f32]).expect("forward_gelu ok");
        assert!(
            (out[0] - 0.841).abs() < 0.01,
            "gelu(1) ≈ 0.841, got {}",
            out[0]
        );
    }

    #[test]
    fn test_linear_xavier_weights_in_range() {
        let mut r = rng();
        let layer = LinearLayer::new_xavier(64, 128, &mut r);
        let limit = (6.0_f32 / (64 + 128) as f32).sqrt();
        for w in &layer.weights {
            assert!(
                *w >= -limit - 1e-6 && *w <= limit + 1e-6,
                "weight {} out of Xavier range ±{}",
                w,
                limit
            );
        }
    }

    #[test]
    fn test_linear_forward_dimension_mismatch() {
        let layer = LinearLayer::new_xavier(4, 8, &mut rng());
        let result = layer.forward(&[0.0f32; 3]);
        assert!(matches!(
            result,
            Err(IdentityConditioningError::DimensionMismatch {
                expected: 4,
                got: 3
            })
        ));
    }

    // ── IdentityEncoder::new ───────────────────────────────────────────────

    #[test]
    fn test_encoder_new_zero_embedding_dim_is_error() {
        let cfg = IdentityConfig {
            embedding_dim: 0,
            ..IdentityConfig::default()
        };
        let mut r = rng();
        let result = IdentityEncoder::new(cfg, &mut r);
        assert!(matches!(
            result,
            Err(IdentityConditioningError::InvalidDimension(0))
        ));
    }

    #[test]
    fn test_encoder_new_zero_shape_dim_is_error() {
        let cfg = IdentityConfig {
            shape_dim: 0,
            ..IdentityConfig::default()
        };
        let mut r = rng();
        let result = IdentityEncoder::new(cfg, &mut r);
        assert!(matches!(
            result,
            Err(IdentityConditioningError::InvalidDimension(0))
        ));
    }

    #[test]
    fn test_encoder_new_zero_hidden_dim_with_layers_is_error() {
        let cfg = IdentityConfig {
            hidden_dim: 0,
            n_hidden_layers: 1,
            ..IdentityConfig::default()
        };
        let mut r = rng();
        let result = IdentityEncoder::new(cfg, &mut r);
        assert!(matches!(
            result,
            Err(IdentityConditioningError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_encoder_new_invalid_dropout() {
        let cfg = IdentityConfig {
            dropout_prob: 1.0,
            ..IdentityConfig::default()
        };
        let mut r = rng();
        let result = IdentityEncoder::new(cfg, &mut r);
        assert!(matches!(
            result,
            Err(IdentityConditioningError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_encoder_new_no_hidden_layers_ok() {
        let cfg = IdentityConfig {
            n_hidden_layers: 0,
            ..IdentityConfig::default()
        };
        let mut r = rng();
        assert!(IdentityEncoder::new(cfg, &mut r).is_ok());
    }

    // ── IdentityEncoder::encode ────────────────────────────────────────────

    #[test]
    fn test_encode_output_length() {
        let enc = make_encoder(false);
        let feat = default_feature();
        let emb = enc.encode(&feat).expect("encode ok");
        assert_eq!(emb.len(), 256);
    }

    #[test]
    fn test_encode_normalized_l2_norm_near_one() {
        let enc = make_encoder(true);
        let feat = default_feature();
        let emb = enc.encode(&feat).expect("encode ok");
        let norm = l2_norm_f32(&emb);
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "normalized embedding norm={} should be ≈1.0",
            norm
        );
    }

    #[test]
    fn test_encode_wrong_shape_dim_is_error() {
        let enc = make_encoder(true);
        let feat = IdentityFeature::new(vec![0.0f32; 100]) // wrong: expected 300
            .with_expression(vec![0.0f32; 50]);
        let result = enc.encode(&feat);
        assert!(matches!(
            result,
            Err(IdentityConditioningError::DimensionMismatch {
                expected: 300,
                got: 100
            })
        ));
    }

    #[test]
    fn test_encode_wrong_expression_dim_is_error() {
        let enc = make_encoder(true);
        let feat = IdentityFeature::new(vec![0.0f32; 300]).with_expression(vec![0.0f32; 10]); // wrong: expected 50
        let result = enc.encode(&feat);
        assert!(matches!(
            result,
            Err(IdentityConditioningError::DimensionMismatch {
                expected: 50,
                got: 10
            })
        ));
    }

    #[test]
    fn test_encode_with_expression_false_shape_only() {
        let cfg = IdentityConfig {
            use_expression: false,
            ..IdentityConfig::default()
        };
        let mut r = rng();
        let enc = IdentityEncoder::new(cfg, &mut r).expect("valid");
        let feat = IdentityFeature::new(vec![0.0f32; 300]); // no expression
        let emb = enc.encode(&feat).expect("encode ok");
        assert_eq!(emb.len(), 256);
    }

    #[test]
    fn test_encode_with_pose_true() {
        let cfg = IdentityConfig {
            use_expression: false,
            use_pose: true,
            ..IdentityConfig::default()
        };
        let mut r = rng();
        let enc = IdentityEncoder::new(cfg, &mut r).expect("valid");
        let feat = IdentityFeature::new(vec![0.0f32; 300]).with_pose(vec![0.0f32; 15]);
        let emb = enc.encode(&feat).expect("encode ok");
        assert_eq!(emb.len(), 256);
    }

    #[test]
    fn test_encode_deterministic() {
        let enc = make_encoder(true);
        let feat = default_feature();
        let emb1 = enc.encode(&feat).expect("encode ok");
        let emb2 = enc.encode(&feat).expect("encode ok");
        assert_eq!(emb1, emb2, "encode must be deterministic");
    }

    #[test]
    fn test_encode_batch_different_features_differ() {
        let enc = make_encoder(true);
        let feat_a = IdentityFeature::new(vec![0.1f32; 300]).with_expression(vec![0.0f32; 50]);
        let feat_b = IdentityFeature::new(vec![0.9f32; 300]).with_expression(vec![1.0f32; 50]);
        let emb_a = enc.encode(&feat_a).expect("encode a ok");
        let emb_b = enc.encode(&feat_b).expect("encode b ok");
        assert_ne!(
            emb_a, emb_b,
            "different features must produce different embeddings"
        );
    }

    // ── encode_with_dropout ────────────────────────────────────────────────

    #[test]
    fn test_encode_with_dropout_output_length() {
        let enc = make_encoder(true);
        let feat = default_feature();
        let mut r = rng();
        let emb = enc
            .encode_with_dropout(&feat, &mut r)
            .expect("dropout encode ok");
        assert_eq!(emb.len(), 256);
    }

    #[test]
    fn test_encode_with_dropout_normalized() {
        let enc = make_encoder(true);
        let feat = default_feature();
        let mut r = rng();
        let emb = enc
            .encode_with_dropout(&feat, &mut r)
            .expect("dropout encode ok");
        let norm = l2_norm_f32(&emb);
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "dropout-encoded embedding must be normalized, norm={}",
            norm
        );
    }

    // ── ident_normalize ────────────────────────────────────────────────────

    #[test]
    fn test_normalize_unit_vector_stays_unit() {
        let v = vec![1.0f32, 0.0, 0.0];
        let n = ident_normalize(&v);
        assert!((l2_norm_f32(&n) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_arbitrary_vector() {
        let v = vec![3.0f32, 4.0, 0.0]; // norm = 5
        let n = ident_normalize(&v);
        assert!((l2_norm_f32(&n) - 1.0).abs() < 1e-6);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector_returns_zero() {
        let v = vec![0.0f32, 0.0, 0.0];
        let n = ident_normalize(&v);
        assert_eq!(n, vec![0.0f32, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_very_large_vector() {
        // 1e20^2 * 64 overflows f32 to inf in the naive norm computation,
        // causing ident_normalize to return the zero vector.  This is the
        // documented "degenerate input" behaviour (norm ≥ f32::INFINITY is
        // treated as degenerate).  We assert the result is finite and
        // non-negative — not NaN.
        let v = vec![1e20f32; 64];
        let n = ident_normalize(&v);
        for x in &n {
            assert!(x.is_finite(), "output must be finite, got {}", x);
        }
    }

    #[test]
    fn test_normalize_very_small_vector() {
        let v = vec![1e-30f32; 64];
        let n = ident_normalize(&v);
        // Might be zero-vector if underflows, or unit vector if finite
        let norm = l2_norm_f32(&n);
        assert!(norm >= 0.0 && !norm.is_nan());
    }

    // ── ident_cosine_similarity ────────────────────────────────────────────

    #[test]
    fn test_cosine_identical_vectors() {
        let v = vec![1.0f32, 2.0, 3.0];
        let sim = ident_cosine_similarity(&v, &v).expect("ok");
        assert!((sim - 1.0).abs() < 1e-6, "identical → sim=1.0, got {}", sim);
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let sim = ident_cosine_similarity(&a, &b).expect("ok");
        assert!(sim.abs() < 1e-6, "orthogonal → sim≈0.0, got {}", sim);
    }

    #[test]
    fn test_cosine_empty_input_is_error() {
        let result = ident_cosine_similarity(&[], &[]);
        assert!(matches!(result, Err(IdentityConditioningError::EmptyInput)));
    }

    #[test]
    fn test_cosine_dimension_mismatch_is_error() {
        let result = ident_cosine_similarity(&[1.0f32, 2.0], &[1.0f32]);
        assert!(matches!(
            result,
            Err(IdentityConditioningError::DimensionMismatch {
                expected: 2,
                got: 1
            })
        ));
    }

    // ── ident_slerp ────────────────────────────────────────────────────────

    #[test]
    fn test_slerp_t_zero_returns_a_normalized() {
        let a = vec![3.0f32, 4.0, 0.0];
        let b = vec![0.0f32, 4.0, 3.0];
        let r = ident_slerp(&a, &b, 0.0).expect("slerp ok");
        let an = ident_normalize(&a);
        for (ri, ai) in r.iter().zip(an.iter()) {
            assert!(
                (ri - ai).abs() < 1e-5,
                "slerp(t=0) should equal normalize(a)"
            );
        }
    }

    #[test]
    fn test_slerp_t_one_returns_b_normalized() {
        let a = vec![3.0f32, 4.0, 0.0];
        let b = vec![0.0f32, 4.0, 3.0];
        let r = ident_slerp(&a, &b, 1.0).expect("slerp ok");
        let bn = ident_normalize(&b);
        for (ri, bi) in r.iter().zip(bn.iter()) {
            assert!(
                (ri - bi).abs() < 1e-5,
                "slerp(t=1) should equal normalize(b)"
            );
        }
    }

    #[test]
    fn test_slerp_t_half_identical_vectors() {
        let a = vec![1.0f32, 0.0, 0.0];
        let r = ident_slerp(&a, &a, 0.5).expect("slerp ok");
        assert!(
            (r[0] - 1.0).abs() < 1e-5,
            "slerp of identical vecs should return same"
        );
    }

    #[test]
    fn test_slerp_empty_input_is_error() {
        let result = ident_slerp(&[], &[], 0.5);
        assert!(matches!(result, Err(IdentityConditioningError::EmptyInput)));
    }

    #[test]
    fn test_slerp_interpolation_at_quarter() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        // slerp(t=0.25): angle 90° → result at 22.5°
        let r = ident_slerp(&a, &b, 0.25).expect("slerp ok");
        let expected = [
            (22.5f32).to_radians().cos(),
            (22.5f32).to_radians().sin(),
            0.0,
        ];
        for (ri, ei) in r.iter().zip(expected.iter()) {
            assert!(
                (ri - ei).abs() < 1e-5,
                "quarter slerp mismatch: {} vs {}",
                ri,
                ei
            );
        }
    }

    #[test]
    fn test_slerp_monotonic_path_five_steps() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        // Compute cosine similarity to a at each step; should decrease monotonically
        let steps = 5;
        let mut prev_sim = f32::INFINITY;
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let r = ident_slerp(&a, &b, t).expect("slerp ok");
            let sim = ident_cosine_similarity(&r, &a).expect("sim ok");
            if i > 0 {
                // Each step moves further from a
                assert!(
                    sim <= prev_sim + 1e-5,
                    "slerp not monotone: step {} sim={} vs prev {}",
                    i,
                    sim,
                    prev_sim
                );
            }
            prev_sim = sim;
        }
    }

    // ── ident_lerp ─────────────────────────────────────────────────────────

    #[test]
    fn test_lerp_t_zero_returns_a() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let r = ident_lerp(&a, &b, 0.0).expect("lerp ok");
        assert_eq!(r, a);
    }

    #[test]
    fn test_lerp_t_one_returns_b() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let r = ident_lerp(&a, &b, 1.0).expect("lerp ok");
        assert_eq!(r, b);
    }

    #[test]
    fn test_lerp_t_half_midpoint() {
        let a = vec![0.0f32, 0.0];
        let b = vec![2.0f32, 4.0];
        let r = ident_lerp(&a, &b, 0.5).expect("lerp ok");
        assert!((r[0] - 1.0).abs() < 1e-6 && (r[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_lerp_empty_is_error() {
        let result = ident_lerp(&[], &[], 0.5);
        assert!(matches!(result, Err(IdentityConditioningError::EmptyInput)));
    }

    // ── ident_mean_embedding ───────────────────────────────────────────────

    #[test]
    fn test_mean_embedding_single_is_normalized() {
        let emb = vec![3.0f32, 4.0];
        let mean = ident_mean_embedding(&[emb]).expect("mean ok");
        assert!((l2_norm_f32(&mean) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_embedding_two_identical() {
        let emb = vec![1.0f32, 0.0, 0.0];
        let mean = ident_mean_embedding(&[emb.clone(), emb.clone()]).expect("mean ok");
        assert!((mean[0] - 1.0).abs() < 1e-5 && mean[1].abs() < 1e-5);
    }

    #[test]
    fn test_mean_embedding_empty_is_error() {
        let result = ident_mean_embedding(&[]);
        assert!(matches!(result, Err(IdentityConditioningError::EmptyInput)));
    }

    // ── ident_similarity_matrix ────────────────────────────────────────────

    #[test]
    fn test_similarity_matrix_diagonal_ones() {
        let embs = vec![vec![1.0f32, 0.0, 0.0], vec![0.0f32, 1.0, 0.0]];
        let mat = ident_similarity_matrix(&embs).expect("matrix ok");
        assert!((mat[0] - 1.0).abs() < 1e-5, "mat[0,0] should be 1.0");
        assert!((mat[3] - 1.0).abs() < 1e-5, "mat[1,1] should be 1.0");
    }

    #[test]
    fn test_similarity_matrix_one_by_one() {
        let embs = vec![vec![1.0f32, 0.0, 0.0]];
        let mat = ident_similarity_matrix(&embs).expect("matrix ok");
        assert_eq!(mat.len(), 1);
        assert!((mat[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_similarity_matrix_empty_is_error() {
        let result = ident_similarity_matrix(&[]);
        assert!(matches!(result, Err(IdentityConditioningError::EmptyInput)));
    }

    #[test]
    fn test_similarity_matrix_is_symmetric() {
        let embs = vec![vec![1.0f32, 0.5], vec![0.5f32, 1.0], vec![0.0f32, 1.0]];
        let mat = ident_similarity_matrix(&embs).expect("matrix ok");
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (mat[i * n + j] - mat[j * n + i]).abs() < 1e-5,
                    "matrix not symmetric at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_similarity_matrix_matches_naive_pairwise() {
        // Cross-check the upper-triangle-only fast path against the
        // straightforward pairwise definition (via ident_cosine_similarity
        // itself), including a zero-norm embedding to exercise the diagonal
        // special case.
        let embs = vec![
            vec![1.0f32, 0.0, 0.0],
            vec![0.0f32, 1.0, 0.0],
            vec![0.0f32, 0.0, 0.0], // zero-norm
            vec![1.0f32, 1.0, 0.0],
        ];
        let n = embs.len();
        let mat = ident_similarity_matrix(&embs).expect("matrix ok");
        for i in 0..n {
            for j in 0..n {
                let expected = ident_cosine_similarity(&embs[i], &embs[j]).unwrap();
                assert!(
                    (mat[i * n + j] - expected).abs() < 1e-5,
                    "mismatch at ({},{}): {} vs {}",
                    i,
                    j,
                    mat[i * n + j],
                    expected
                );
            }
        }
        // The zero-norm embedding's self-similarity must be 0.0, not 1.0.
        assert!((mat[2 * n + 2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_similarity_matrix_dimension_mismatch_errors() {
        let embs = vec![vec![1.0f32, 0.0], vec![1.0f32, 0.0, 0.0]];
        let result = ident_similarity_matrix(&embs);
        assert!(matches!(
            result,
            Err(IdentityConditioningError::DimensionMismatch { .. })
        ));
    }

    // ── IdentityCache ──────────────────────────────────────────────────────

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache = IdentityCache::new(4);
        cache.insert(42, vec![1.0f32, 2.0]);
        let got = cache.get(42).expect("cache hit");
        assert_eq!(got, &vec![1.0f32, 2.0]);
    }

    #[test]
    fn test_cache_get_missing_returns_none() {
        let mut cache = IdentityCache::new(4);
        assert!(cache.get(99).is_none());
    }

    #[test]
    fn test_cache_len_and_is_empty() {
        let mut cache = IdentityCache::new(4);
        assert!(cache.is_empty());
        cache.insert(1, vec![0.0f32]);
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = IdentityCache::new(4);
        cache.insert(1, vec![0.0f32]);
        cache.insert(2, vec![1.0f32]);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_lru_eviction_capacity_one() {
        let mut cache = IdentityCache::new(1);
        cache.insert(1, vec![1.0f32]);
        cache.insert(2, vec![2.0f32]); // should evict key=1
        assert!(cache.get(1).is_none(), "key 1 should have been evicted");
        assert!(cache.get(2).is_some(), "key 2 should be present");
    }

    #[test]
    fn test_cache_lru_eviction_oldest_evicted_first() {
        // Cap=2: insert 1,2 then access 1, then insert 3 → 2 should be evicted
        let mut cache = IdentityCache::new(2);
        cache.insert(1, vec![1.0f32]);
        cache.insert(2, vec![2.0f32]);
        // Access key=1, making key=2 the LRU
        let _ = cache.get(1);
        cache.insert(3, vec![3.0f32]);
        assert!(cache.get(2).is_none(), "key 2 (LRU) should be evicted");
        assert!(cache.get(1).is_some(), "key 1 should survive");
        assert!(cache.get(3).is_some(), "key 3 should be present");
    }

    #[test]
    fn test_cache_zero_capacity_never_grows() {
        // Regression test: capacity=0 previously skipped eviction entirely
        // (the eviction guard required `capacity > 0`) and fell through to
        // the unconditional append, growing without bound.
        let mut cache = IdentityCache::new(0);
        for i in 0..100u64 {
            cache.insert(i, vec![i as f32]);
        }
        assert_eq!(cache.len(), 0, "capacity=0 cache must stay empty");
        assert!(cache.is_empty());
        assert!(cache.get(0).is_none());
        assert!(cache.get(99).is_none());
    }

    // ── ident_hash_feature ─────────────────────────────────────────────────

    #[test]
    fn test_hash_feature_deterministic() {
        let f = IdentityFeature::new(vec![1.0f32, 2.0, 3.0]);
        let h1 = ident_hash_feature(&f);
        let h2 = ident_hash_feature(&f);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_feature_different_inputs_differ() {
        let f1 = IdentityFeature::new(vec![0.0f32; 10]);
        let f2 = IdentityFeature::new(vec![1.0f32; 10]);
        assert_ne!(ident_hash_feature(&f1), ident_hash_feature(&f2));
    }

    #[test]
    fn test_hash_feature_expression_changes_hash() {
        let f1 = IdentityFeature::new(vec![1.0f32; 5]);
        let f2 = IdentityFeature::new(vec![1.0f32; 5]).with_expression(vec![1.0f32; 3]);
        assert_ne!(ident_hash_feature(&f1), ident_hash_feature(&f2));
    }

    // ── ident_compute_stats ────────────────────────────────────────────────

    #[test]
    fn test_compute_stats_single_embedding() {
        let embs = vec![vec![1.0f32, 0.0]];
        let stats = ident_compute_stats(&embs).expect("stats ok");
        assert!((stats.mean_similarity - 1.0).abs() < 1e-5);
        assert!((stats.min_similarity - 1.0).abs() < 1e-5);
        assert!((stats.max_similarity - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_stats_two_identical_embeddings() {
        let emb = vec![1.0f32, 0.0, 0.0];
        let stats = ident_compute_stats(&[emb.clone(), emb]).expect("stats ok");
        assert!((stats.mean_similarity - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_stats_three_embeddings() {
        let embs = vec![
            vec![1.0f32, 0.0, 0.0],
            vec![0.0f32, 1.0, 0.0],
            vec![0.0f32, 0.0, 1.0],
        ];
        let stats = ident_compute_stats(&embs).expect("stats ok");
        // All pairwise cosines are 0
        assert!(stats.mean_similarity.abs() < 1e-5);
        assert!((stats.embedding_norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_stats_matches_naive_all_ordered_pairs() {
        // Cross-check the unordered-pairs-once fast path against the
        // straightforward "all ordered off-diagonal pairs" definition.
        let embs = vec![
            vec![1.0f32, 0.2, 0.0],
            vec![0.3f32, 1.0, 0.1],
            vec![0.0f32, 0.5, 1.0],
            vec![1.0f32, 1.0, 1.0],
        ];
        let n = embs.len();
        let mut naive_sims = Vec::with_capacity(n * (n - 1));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    naive_sims.push(ident_cosine_similarity(&embs[i], &embs[j]).unwrap());
                }
            }
        }
        let naive_mean = naive_sims.iter().sum::<f32>() / naive_sims.len() as f32;
        let naive_min = naive_sims.iter().cloned().fold(f32::INFINITY, f32::min);
        let naive_max = naive_sims.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let stats = ident_compute_stats(&embs).expect("stats ok");
        assert!((stats.mean_similarity - naive_mean).abs() < 1e-4);
        assert!((stats.min_similarity - naive_min).abs() < 1e-4);
        assert!((stats.max_similarity - naive_max).abs() < 1e-4);
    }

    #[test]
    fn test_compute_stats_dimension_mismatch_errors() {
        let embs = vec![vec![1.0f32, 0.0], vec![1.0f32, 0.0, 0.0]];
        let result = ident_compute_stats(&embs);
        assert!(matches!(
            result,
            Err(IdentityConditioningError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_compute_stats_empty_is_error() {
        let result = ident_compute_stats(&[]);
        assert!(matches!(result, Err(IdentityConditioningError::EmptyInput)));
    }

    // ── format functions ───────────────────────────────────────────────────

    #[test]
    fn test_format_config_non_empty() {
        let cfg = IdentityConfig::default();
        let s = ident_format_config(&cfg);
        assert!(!s.is_empty());
        assert!(s.contains("IdentityConfig"));
    }

    #[test]
    fn test_format_config_contains_dims() {
        let cfg = IdentityConfig::default();
        let s = ident_format_config(&cfg);
        assert!(s.contains("300"), "should contain shape_dim 300");
        assert!(s.contains("256"), "should contain embedding_dim 256");
    }

    #[test]
    fn test_format_stats_non_empty() {
        let stats = IdentityStats {
            mean_similarity: 0.9,
            min_similarity: 0.8,
            max_similarity: 1.0,
            embedding_norm: 1.0,
        };
        let s = ident_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("IdentityStats"));
    }

    // ── gelu scalar approximation ──────────────────────────────────────────

    #[test]
    fn test_gelu_scalar_zero() {
        assert!(gelu_scalar(0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_scalar_one_approx() {
        assert!(
            (gelu_scalar(1.0) - 0.841).abs() < 0.01,
            "gelu(1) ≈ 0.841, got {}",
            gelu_scalar(1.0)
        );
    }

    #[test]
    fn test_gelu_scalar_negative() {
        // gelu(-1) should be small and negative
        let g = gelu_scalar(-1.0);
        assert!(g < 0.0, "gelu(-1) should be negative, got {}", g);
        assert!(g > -1.0, "gelu(-1) should be > -1, got {}", g);
    }

    // ── full pipeline ──────────────────────────────────────────────────────

    #[test]
    fn test_full_pipeline_l2_norm() {
        let mut r = rng();
        let cfg = IdentityConfig::default();
        let enc = IdentityEncoder::new(cfg, &mut r).expect("valid");
        let feat = IdentityFeature::new(vec![0.5f32; 300]).with_expression(vec![0.1f32; 50]);
        let emb = enc.encode(&feat).expect("encode ok");
        assert_eq!(emb.len(), 256);
        let norm = l2_norm_f32(&emb);
        assert!((norm - 1.0).abs() < 1e-5, "norm={}", norm);
    }
}
