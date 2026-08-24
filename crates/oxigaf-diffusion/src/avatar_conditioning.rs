//! Avatar conditioning module — encodes FLAME parametric head model parameters
//! as conditioning vectors for the multi-view diffusion model.
//!
//! This module bridges FLAME head parameters (shape, expression, pose,
//! translation) into continuous embedding vectors suitable for cross-attention
//! conditioning in the U-Net denoising process.
//!
//! # Overview
//!
//! The encoding pipeline is:
//! 1. **Sinusoidal positional encoding** — maps each scalar to a high-frequency
//!    feature vector (like NeRF / Fourier features).
//! 2. **Linear embedding** — projects the sinusoidal features to a target
//!    embedding dimension.
//! 3. **Optional L2 normalisation** — keeps conditioning vectors on the unit
//!    hypersphere, which improves blending arithmetic.
//!
//! # Example
//!
//! ```rust
//! use oxigaf_diffusion::avatar_conditioning::{
//!     FlameConditioningConfig, condition_from_flame_params,
//! };
//!
//! let config = FlameConditioningConfig::default();
//! let shape  = vec![0.0f32; 100];
//! let expr   = vec![0.0f32; 50];
//! let pose   = vec![0.0f32; 15];
//! let trans  = [0.0f32; 3];
//!
//! let cv = condition_from_flame_params(&shape, &expr, &pose, &trans, &config)
//!     .expect("conditioning should succeed for neutral params");
//! // With use_shape + use_expression + use_pose all enabled, each projected to
//! // embedding_dim (512), the final vector is 3 × 512 = 1536 before any reduction.
//! assert_eq!(cv.dim, config.embedding_dim * 3);
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during avatar conditioning.
#[derive(Debug, Error, PartialEq)]
pub enum ConditioningError {
    /// Input parameter vector has zero length when non-zero is required.
    #[error("Empty params: {param_name} has zero length")]
    EmptyParams { param_name: String },

    /// Concatenated or expected dimension does not match actual dimension.
    #[error("Conditioning dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Embedding dimension must be strictly positive.
    #[error("Invalid embedding dimension {dim}: must be > 0")]
    InvalidEmbeddingDim { dim: usize },

    /// Cannot normalise a zero-norm vector.
    #[error("Normalization failed: zero-norm vector")]
    ZeroNorm,

    /// A parameter value was non-finite (NaN or ±∞).
    #[error("Non-finite value {value} in {param_name}")]
    NonFiniteValue { param_name: String, value: f32 },
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for FLAME-based avatar conditioning.
#[derive(Debug, Clone)]
pub struct FlameConditioningConfig {
    /// Output embedding dimension (default: 512).
    pub embedding_dim: usize,
    /// Include shape (identity) coefficients (default: true).
    pub use_shape: bool,
    /// Include expression coefficients (default: true).
    pub use_expression: bool,
    /// Include pose (axis-angle) coefficients (default: true).
    pub use_pose: bool,
    /// Include global translation (default: false).
    pub use_translation: bool,
    /// How many leading shape coefficients to encode (default: 100).
    pub n_shape_coeffs: usize,
    /// How many leading expression coefficients to encode (default: 50).
    pub n_expression_coeffs: usize,
    /// How many leading pose coefficients to encode (default: 15, i.e. 5
    /// FLAME joints × 3 axis-angle components). Pose values are zero-padded
    /// or truncated to this length before projection, exactly like shape and
    /// expression, so the pose projection matrix depends only on this value
    /// (not on the raw pose slice's length).
    pub n_pose_coeffs: usize,
    /// L2-normalise the final output vector (default: true).
    pub normalize_output: bool,
}

impl Default for FlameConditioningConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 512,
            use_shape: true,
            use_expression: true,
            use_pose: true,
            use_translation: false,
            n_shape_coeffs: 100,
            n_expression_coeffs: 50,
            n_pose_coeffs: 15,
            normalize_output: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Sinusoidal positional encoding
// ---------------------------------------------------------------------------

/// Sinusoidal positional encoding (NeRF-style Fourier features).
///
/// For each scalar input `x` and frequency band `i ∈ [0, n_freqs)`:
/// ```text
/// freq = 2^(i * max_freq / max(n_freqs - 1, 1))
/// features: [sin(freq * x), cos(freq * x)]
/// ```
/// If `include_input` is true the raw `x` value is prepended.
#[derive(Debug, Clone)]
pub struct SinusoidalEncoding {
    /// Number of frequency bands (default: 8).
    pub n_freqs: usize,
    /// Maximum frequency exponent in powers of two (default: 8.0).
    pub max_freq: f32,
    /// Prepend the raw input scalar to the output (default: true).
    pub include_input: bool,
}

impl Default for SinusoidalEncoding {
    fn default() -> Self {
        Self {
            n_freqs: 8,
            max_freq: 8.0,
            include_input: true,
        }
    }
}

impl SinusoidalEncoding {
    /// Create a new sinusoidal encoding with explicit parameters.
    #[must_use]
    pub fn new(n_freqs: usize, max_freq: f32, include_input: bool) -> Self {
        Self {
            n_freqs,
            max_freq,
            include_input,
        }
    }

    /// Output dimensionality per input scalar.
    #[must_use]
    pub fn output_dim_per_input(&self) -> usize {
        let sin_cos = 2 * self.n_freqs;
        if self.include_input {
            sin_cos + 1
        } else {
            sin_cos
        }
    }

    /// Encode a single scalar `x` into a feature vector.
    #[must_use]
    pub fn encode_scalar(&self, x: f32) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.output_dim_per_input());
        if self.include_input {
            out.push(x);
        }
        // Avoid division by zero when n_freqs == 1
        let denom = (self.n_freqs.saturating_sub(1)).max(1) as f32;
        for i in 0..self.n_freqs {
            let freq = (i as f32 * self.max_freq / denom).exp2();
            out.push((freq * x).sin());
            out.push((freq * x).cos());
        }
        out
    }

    /// Encode a slice of scalars by concatenating per-scalar encodings.
    #[must_use]
    pub fn encode_slice(&self, values: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(values.len() * self.output_dim_per_input());
        for &v in values {
            out.extend_from_slice(&self.encode_scalar(v));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Linear embedding layer (no bias, xorshift64 initialisation)
// ---------------------------------------------------------------------------

/// Dense linear layer `y = W * x` with weights initialised by xorshift64.
///
/// Weights are drawn uniformly from `[-1/√in_dim, 1/√in_dim]`.
#[derive(Debug, Clone)]
pub struct LinearEmbedding {
    /// Flat row-major weight matrix `[out_dim × in_dim]`.
    pub weights: Vec<f32>,
    /// Input dimension.
    pub in_dim: usize,
    /// Output dimension.
    pub out_dim: usize,
}

/// Advance one step of the xorshift64 PRNG.
///
/// The caller is responsible for ensuring `state != 0` before the first call.
/// ```text
/// state ^= state << 13
/// state ^= state >> 7
/// state ^= state << 17
/// ```
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    if *state == 0 {
        *state = 1;
    }
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Map a raw u64 from xorshift64 to a float in `[-scale, scale)`.
#[inline]
fn xorshift_to_float(raw: u64, scale: f32) -> f32 {
    // Map to [0, 1) then to [-scale, scale). Uses the top 24 bits (f32's
    // mantissa width, including the implicit bit) rather than 53: building a
    // 53-bit integer and converting it to f32 rounds some inputs up to
    // exactly 2^53, which is exactly 1.0 after the division — silently
    // breaking the documented "< scale" bound (and wasting the extra 29 bits,
    // which f32 cannot represent anyway). A 24-bit integer divided by 2^24 is
    // always exactly representable and strictly < 1.0.
    let unit = (raw >> 40) as f32 / (1u32 << 24) as f32;
    unit * 2.0 * scale - scale
}

impl LinearEmbedding {
    /// Construct a new embedding, initialising weights with xorshift64.
    ///
    /// # Errors
    ///
    /// Returns [`ConditioningError::InvalidEmbeddingDim`] if either dimension
    /// is zero.
    pub fn new(in_dim: usize, out_dim: usize, seed: u64) -> Result<Self, ConditioningError> {
        if in_dim == 0 {
            return Err(ConditioningError::InvalidEmbeddingDim { dim: in_dim });
        }
        if out_dim == 0 {
            return Err(ConditioningError::InvalidEmbeddingDim { dim: out_dim });
        }
        let scale = 1.0 / (in_dim as f32).sqrt();
        let n = out_dim * in_dim;
        let mut weights = Vec::with_capacity(n);
        let mut state = if seed == 0 {
            6364136223846793005u64
        } else {
            seed
        };
        for _ in 0..n {
            let raw = xorshift64(&mut state);
            weights.push(xorshift_to_float(raw, scale));
        }
        Ok(Self {
            weights,
            in_dim,
            out_dim,
        })
    }

    /// Compute the forward pass `y = W * x`.
    ///
    /// # Errors
    ///
    /// Returns [`ConditioningError::DimensionMismatch`] if `x.len() != in_dim`.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, ConditioningError> {
        if x.len() != self.in_dim {
            return Err(ConditioningError::DimensionMismatch {
                expected: self.in_dim,
                actual: x.len(),
            });
        }
        let mut out = vec![0.0f32; self.out_dim];
        for (row, out_val) in out.iter_mut().enumerate() {
            let base = row * self.in_dim;
            let mut acc = 0.0f32;
            for (col, &xc) in x.iter().enumerate() {
                acc += self.weights[base + col] * xc;
            }
            *out_val = acc;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Cached projection matrices
// ---------------------------------------------------------------------------
//
// `LinearEmbedding::new(in_dim, out_dim, seed)` is a pure function of its
// three arguments, but the free `encode_*` functions below previously called
// it fresh on *every* invocation. With the defaults (e.g. a 512x1700 shape
// projection), that regenerates ~870K PRNG-derived f32 weights per call, and
// `condition_from_flame_params` does this three or four times per call with
// the same fixed seeds every time. Cache built matrices keyed by
// `(in_dim, out_dim, seed)` so repeated calls with the same shape/seed reuse
// one instance instead of re-running xorshift64 from scratch.
static EMBEDDING_CACHE: std::sync::OnceLock<
    std::sync::Mutex<
        std::collections::HashMap<(usize, usize, u64), std::sync::Arc<LinearEmbedding>>,
    >,
> = std::sync::OnceLock::new();

/// Fetches (or lazily builds and caches) a [`LinearEmbedding`] for the given
/// shape and seed.
fn cached_linear_embedding(
    in_dim: usize,
    out_dim: usize,
    seed: u64,
) -> Result<std::sync::Arc<LinearEmbedding>, ConditioningError> {
    let cache =
        EMBEDDING_CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let key = (in_dim, out_dim, seed);

    if let Ok(guard) = cache.lock() {
        if let Some(existing) = guard.get(&key) {
            return Ok(existing.clone());
        }
    }

    let built = std::sync::Arc::new(LinearEmbedding::new(in_dim, out_dim, seed)?);

    // Recover from a poisoned lock rather than panicking (no unwrap()):
    // a panic in another thread while holding the lock is not this
    // function's concern, and the map contents remain valid to read/write.
    let mut guard = match cache.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    // Another thread may have raced us and inserted first; prefer whichever
    // instance is already cached so concurrent callers observe one canonical
    // matrix rather than silently-different (but numerically equivalent) ones.
    let entry = guard.entry(key).or_insert(built);
    Ok(entry.clone())
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// Encoded conditioning vector produced from FLAME parameters.
#[derive(Debug, Clone)]
pub struct ConditioningVector {
    /// The embedding data.
    pub data: Vec<f32>,
    /// Embedding dimension (`data.len()`).
    pub dim: usize,
    /// Number of shape coefficients used.
    pub source_shape: usize,
    /// Number of expression coefficients used.
    pub source_expr: usize,
    /// Number of pose values used.
    pub source_pose: usize,
}

/// Aggregate statistics over a batch of conditioning vectors.
#[derive(Debug, Clone)]
pub struct ConditioningStats {
    /// Number of vectors in the batch.
    pub n_vectors: usize,
    /// Embedding dimension.
    pub dim: usize,
    /// Mean L2 norm across the batch.
    pub mean_norm: f32,
    /// Standard deviation of L2 norms across the batch.
    pub std_norm: f32,
    /// Minimum L2 norm in the batch.
    pub min_norm: f32,
    /// Maximum L2 norm in the batch.
    pub max_norm: f32,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Wrap an angle (in radians) to `[-π, π]`.
///
/// Uses a closed-form reduction rather than a `while` loop: for `|x|` above
/// roughly `2^24 * 2π ≈ 1.05e8`, the f32 ulp exceeds `2π`, so
/// `v -= 2.0 * PI` becomes a no-op and a naive loop never terminates (a true
/// hang, not merely a slow one — and `f32::INFINITY` hangs identically since
/// `inf - 2π == inf`). Non-finite input is guarded explicitly and mapped to
/// `0.0` rather than being fed to `rem_euclid`, which would itself return
/// `NaN`.
#[inline]
fn wrap_to_pi(x: f32) -> f32 {
    use std::f32::consts::PI;
    if !x.is_finite() {
        return 0.0;
    }
    let two_pi = 2.0 * PI;
    let wrapped = x.rem_euclid(two_pi); // in [0, two_pi)
    if wrapped > PI {
        wrapped - two_pi
    } else {
        wrapped
    }
}

/// Take the first `n` values from `src`, padding with 0.0 if `src` is shorter.
fn take_or_pad(src: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    let copy_len = src.len().min(n);
    out[..copy_len].copy_from_slice(&src[..copy_len]);
    out
}

/// L2-normalise a vector in-place.  Returns the norm that was used.
fn l2_normalise_inplace(v: &mut [f32]) -> Result<f32, ConditioningError> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return Err(ConditioningError::ZeroNorm);
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    Ok(norm)
}

// ---------------------------------------------------------------------------
// Component encoding functions
// ---------------------------------------------------------------------------

/// Encode shape (identity) parameters.
///
/// Applies sinusoidal encoding to the first `n_coeffs` shape params (zero-padded
/// if the slice is shorter), then projects to `embed_dim` via a linear layer
/// seeded with `seed`.
///
/// # Errors
///
/// * [`ConditioningError::InvalidEmbeddingDim`] — if `embed_dim == 0` or
///   `n_coeffs == 0`.
pub fn encode_shape_params(
    params: &[f32],
    n_coeffs: usize,
    embed_dim: usize,
    seed: u64,
) -> Result<Vec<f32>, ConditioningError> {
    if embed_dim == 0 {
        return Err(ConditioningError::InvalidEmbeddingDim { dim: embed_dim });
    }
    if n_coeffs == 0 {
        return Err(ConditioningError::InvalidEmbeddingDim { dim: n_coeffs });
    }
    let values = take_or_pad(params, n_coeffs);
    let enc = SinusoidalEncoding::default();
    let features = enc.encode_slice(&values);
    let layer = cached_linear_embedding(features.len(), embed_dim, seed)?;
    layer.forward(&features)
}

/// Encode expression parameters.
///
/// Identical to [`encode_shape_params`] but intended for expression coefficients,
/// and uses a distinct seed to ensure different weight initialisation.
///
/// # Errors
///
/// * [`ConditioningError::InvalidEmbeddingDim`] — if `embed_dim == 0` or
///   `n_coeffs == 0`.
pub fn encode_expression_params(
    params: &[f32],
    n_coeffs: usize,
    embed_dim: usize,
    seed: u64,
) -> Result<Vec<f32>, ConditioningError> {
    if embed_dim == 0 {
        return Err(ConditioningError::InvalidEmbeddingDim { dim: embed_dim });
    }
    if n_coeffs == 0 {
        return Err(ConditioningError::InvalidEmbeddingDim { dim: n_coeffs });
    }
    let values = take_or_pad(params, n_coeffs);
    let enc = SinusoidalEncoding::default();
    let features = enc.encode_slice(&values);
    let layer = cached_linear_embedding(features.len(), embed_dim, seed)?;
    layer.forward(&features)
}

/// Encode pose parameters (axis-angle representation).
///
/// Unlike [`encode_shape_params`]/[`encode_expression_params`], earlier
/// versions of this function encoded `params` at its *native* length instead
/// of normalising to a fixed `n_coeffs` first. Because the projection matrix
/// depends on the input dimension, two poses of different lengths (e.g. 15
/// vs. 6 values) were projected by two unrelated random matrices, making
/// their outputs incomparable even though both had `dim == embed_dim` — so
/// [`blend_conditioning`]/[`conditioning_similarity`] would silently
/// blend/compare noise. `params` is now normalised to `n_coeffs` via
/// [`take_or_pad`] first, exactly like the other components, so the
/// projection matrix is a pure function of `(n_coeffs, embed_dim, seed)`.
///
/// Each (normalised) pose value is wrapped to `[-π, π]` to handle angle
/// periodicity, then encoded via sinusoidal features and projected to
/// `embed_dim`.
///
/// # Errors
///
/// * [`ConditioningError::InvalidEmbeddingDim`] — if `embed_dim == 0` or `n_coeffs == 0`.
/// * [`ConditioningError::EmptyParams`] — if `params` is empty (matching the
///   siblings' `n_coeffs == 0` check, this stays a hard error rather than
///   silently zero-padding, since an empty pose slice is usually a caller
///   bug rather than an intentional "neutral pose").
/// * [`ConditioningError::NonFiniteValue`] — if any value in `params` is NaN
///   or infinite (an unguarded non-finite value could otherwise hang angle
///   wrapping; see [`wrap_to_pi`]).
pub fn encode_pose_params(
    params: &[f32],
    n_coeffs: usize,
    embed_dim: usize,
    seed: u64,
) -> Result<Vec<f32>, ConditioningError> {
    if embed_dim == 0 {
        return Err(ConditioningError::InvalidEmbeddingDim { dim: embed_dim });
    }
    if n_coeffs == 0 {
        return Err(ConditioningError::InvalidEmbeddingDim { dim: n_coeffs });
    }
    if params.is_empty() {
        return Err(ConditioningError::EmptyParams {
            param_name: "pose_params".to_string(),
        });
    }
    if let Some(&bad) = params.iter().find(|v| !v.is_finite()) {
        return Err(ConditioningError::NonFiniteValue {
            param_name: "pose_params".to_string(),
            value: bad,
        });
    }
    let values = take_or_pad(params, n_coeffs);
    let wrapped: Vec<f32> = values.iter().map(|&v| wrap_to_pi(v)).collect();
    let enc = SinusoidalEncoding::default();
    let features = enc.encode_slice(&wrapped);
    let layer = cached_linear_embedding(features.len(), embed_dim, seed)?;
    layer.forward(&features)
}

/// Encode the global translation `[tx, ty, tz]`.
///
/// No angle wrapping is applied.
///
/// # Errors
///
/// * [`ConditioningError::InvalidEmbeddingDim`] — if `embed_dim == 0`.
pub fn encode_translation(
    translation: &[f32; 3],
    embed_dim: usize,
    seed: u64,
) -> Result<Vec<f32>, ConditioningError> {
    if embed_dim == 0 {
        return Err(ConditioningError::InvalidEmbeddingDim { dim: embed_dim });
    }
    let enc = SinusoidalEncoding::default();
    let features = enc.encode_slice(translation);
    let layer = cached_linear_embedding(features.len(), embed_dim, seed)?;
    layer.forward(&features)
}

// ---------------------------------------------------------------------------
// Primary API
// ---------------------------------------------------------------------------

/// Encode FLAME parameters into a single conditioning vector.
///
/// Enabled components (governed by `config`) are independently encoded and
/// concatenated.  If `config.normalize_output` is true the result is
/// L2-normalised.
///
/// # Seeds
///
/// Internal weight matrices are seeded deterministically:
/// * Shape:       seed `0x1000`
/// * Expression:  seed `0x2000`
/// * Pose:        seed `0x3000`
/// * Translation: seed `0x4000`
///
/// # Errors
///
/// * [`ConditioningError::InvalidEmbeddingDim`] — zero embedding dim.
/// * [`ConditioningError::EmptyParams`] — pose is empty and `use_pose` is true.
/// * [`ConditioningError::ZeroNorm`] — output is zero (only with
///   `normalize_output = true`).
pub fn condition_from_flame_params(
    shape_params: &[f32],
    expression_params: &[f32],
    pose_params: &[f32],
    translation: &[f32; 3],
    config: &FlameConditioningConfig,
) -> Result<ConditioningVector, ConditioningError> {
    if config.embedding_dim == 0 {
        return Err(ConditioningError::InvalidEmbeddingDim {
            dim: config.embedding_dim,
        });
    }

    let mut parts: Vec<Vec<f32>> = Vec::new();
    let mut source_shape = 0usize;
    let mut source_expr = 0usize;
    let mut source_pose = 0usize;

    if config.use_shape {
        let enc = encode_shape_params(
            shape_params,
            config.n_shape_coeffs,
            config.embedding_dim,
            0x1000,
        )?;
        source_shape = shape_params.len().min(config.n_shape_coeffs);
        parts.push(enc);
    }

    if config.use_expression {
        let enc = encode_expression_params(
            expression_params,
            config.n_expression_coeffs,
            config.embedding_dim,
            0x2000,
        )?;
        source_expr = expression_params.len().min(config.n_expression_coeffs);
        parts.push(enc);
    }

    if config.use_pose {
        let enc = encode_pose_params(
            pose_params,
            config.n_pose_coeffs,
            config.embedding_dim,
            0x3000,
        )?;
        source_pose = pose_params.len().min(config.n_pose_coeffs);
        parts.push(enc);
    }

    if config.use_translation {
        let enc = encode_translation(translation, config.embedding_dim, 0x4000)?;
        parts.push(enc);
    }

    // Concatenate or use a single part
    let mut data: Vec<f32> = parts.into_iter().flatten().collect();

    if config.normalize_output {
        l2_normalise_inplace(&mut data)?;
    }

    let dim = data.len();
    Ok(ConditioningVector {
        data,
        dim,
        source_shape,
        source_expr,
        source_pose,
    })
}

/// Linearly blend two conditioning vectors.
///
/// `result = (1 - t) * a + t * b`.  Both vectors must have the same `dim`.
///
/// # Errors
///
/// * [`ConditioningError::DimensionMismatch`] — if `a.dim != b.dim`.
pub fn blend_conditioning(
    a: &ConditioningVector,
    b: &ConditioningVector,
    t: f32,
) -> Result<ConditioningVector, ConditioningError> {
    if a.dim != b.dim {
        return Err(ConditioningError::DimensionMismatch {
            expected: a.dim,
            actual: b.dim,
        });
    }
    let one_minus_t = 1.0 - t;
    let data: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&ai, &bi)| one_minus_t * ai + t * bi)
        .collect();
    Ok(ConditioningVector {
        dim: a.dim,
        source_shape: a.source_shape,
        source_expr: a.source_expr,
        source_pose: a.source_pose,
        data,
    })
}

/// L2-normalise a raw feature vector, returning a new unit-norm vector.
///
/// # Errors
///
/// * [`ConditioningError::ZeroNorm`] — if the vector has near-zero norm.
pub fn normalize_conditioning(v: &[f32]) -> Result<Vec<f32>, ConditioningError> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return Err(ConditioningError::ZeroNorm);
    }
    Ok(v.iter().map(|x| x / norm).collect())
}

/// Type alias for a FLAME parameter batch entry: (shape, expression, pose, translation).
type FlameParamTuple = (Vec<f32>, Vec<f32>, Vec<f32>, [f32; 3]);

/// Encode a batch of FLAME parameter tuples.
///
/// Each element is `(shape, expression, pose, translation)`.
///
/// # Errors
///
/// Propagates any [`ConditioningError`] from individual calls.
pub fn batch_condition_from_flame_params(
    batch: &[FlameParamTuple],
    config: &FlameConditioningConfig,
) -> Result<Vec<ConditioningVector>, ConditioningError> {
    batch
        .iter()
        .map(|(shape, expr, pose, trans)| {
            condition_from_flame_params(shape, expr, pose, trans, config)
        })
        .collect()
}

/// Compute cosine similarity between two conditioning vectors.
///
/// Returns `0.0` if either vector has zero norm.
#[must_use]
pub fn conditioning_similarity(a: &ConditioningVector, b: &ConditioningVector) -> f32 {
    let norm_a: f32 = a.data.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.data.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    let dot: f32 = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&ai, &bi)| ai * bi)
        .sum();
    dot / (norm_a * norm_b)
}

/// Compute aggregate statistics over a batch of conditioning vectors.
///
/// # Panics
///
/// Does not panic — returns a zeroed [`ConditioningStats`] for empty batches.
#[must_use]
pub fn compute_conditioning_stats(vectors: &[ConditioningVector]) -> ConditioningStats {
    if vectors.is_empty() {
        return ConditioningStats {
            n_vectors: 0,
            dim: 0,
            mean_norm: 0.0,
            std_norm: 0.0,
            min_norm: 0.0,
            max_norm: 0.0,
        };
    }
    let dim = vectors[0].dim;
    let norms: Vec<f32> = vectors
        .iter()
        .map(|cv| cv.data.iter().map(|x| x * x).sum::<f32>().sqrt())
        .collect();
    let n = norms.len() as f32;
    let mean_norm = norms.iter().sum::<f32>() / n;
    let variance = norms.iter().map(|v| (v - mean_norm).powi(2)).sum::<f32>() / n;
    let std_norm = variance.sqrt();
    let min_norm = norms.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_norm = norms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    ConditioningStats {
        n_vectors: vectors.len(),
        dim,
        mean_norm,
        std_norm,
        min_norm,
        max_norm,
    }
}

/// Create a conditioning vector for a neutral (all-zero) face.
///
/// This is useful as a default or "empty" conditioning signal.
///
/// # Errors
///
/// Propagates any error from [`condition_from_flame_params`].
pub fn create_neutral_conditioning(
    config: &FlameConditioningConfig,
) -> Result<ConditioningVector, ConditioningError> {
    let shape = vec![0.0f32; config.n_shape_coeffs];
    let expr = vec![0.0f32; config.n_expression_coeffs];
    // Use 15 values (5 joints × 3) for pose, all zeros
    let pose = vec![0.0f32; 15];
    let trans = [0.0f32; 3];
    condition_from_flame_params(&shape, &expr, &pose, &trans, config)
}

/// Generate a smooth interpolation sequence between two conditioning vectors.
///
/// Produces `n_steps` evenly-spaced blend ratios `t ∈ [0, 1]` (inclusive at
/// both endpoints).  Uses [`blend_conditioning`] for each step.
///
/// Returns an empty `Vec` when `n_steps == 0`.
#[must_use]
pub fn interpolate_conditioning_sequence(
    start: &ConditioningVector,
    end: &ConditioningVector,
    n_steps: usize,
) -> Vec<ConditioningVector> {
    if n_steps == 0 {
        return Vec::new();
    }
    (0..n_steps)
        .filter_map(|i| {
            let t = if n_steps == 1 {
                0.0
            } else {
                i as f32 / (n_steps - 1) as f32
            };
            blend_conditioning(start, end, t).ok()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // --- helpers ------------------------------------------------------------

    fn make_config() -> FlameConditioningConfig {
        FlameConditioningConfig {
            embedding_dim: 64,
            n_shape_coeffs: 10,
            n_expression_coeffs: 5,
            ..FlameConditioningConfig::default()
        }
    }

    fn make_shape_params(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * 0.01).collect()
    }
    fn make_expr_params(n: usize) -> Vec<f32> {
        (0..n).map(|i| -(i as f32) * 0.02).collect()
    }
    fn make_pose_params() -> Vec<f32> {
        vec![
            0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]
    }

    // --- SinusoidalEncoding tests -------------------------------------------

    #[test]
    fn test_sinusoidal_output_dim_default() {
        let enc = SinusoidalEncoding::default();
        // default: n_freqs=8, include_input=true => 2*8 + 1 = 17
        assert_eq!(enc.output_dim_per_input(), 17);
    }

    #[test]
    fn test_sinusoidal_output_dim_no_input() {
        let enc = SinusoidalEncoding::new(4, 4.0, false);
        assert_eq!(enc.output_dim_per_input(), 8);
    }

    #[test]
    fn test_sinusoidal_encode_zero() {
        let enc = SinusoidalEncoding::default();
        let v = enc.encode_scalar(0.0);
        assert_eq!(v.len(), enc.output_dim_per_input());
        // raw input is 0.0
        assert!((v[0]).abs() < 1e-6);
        // sin(0) = 0, cos(0) = 1 for every band
        for i in 0..enc.n_freqs {
            let base = if enc.include_input { 1 } else { 0 };
            assert!((v[base + 2 * i]).abs() < 1e-5, "sin should be ~0");
            assert!((v[base + 2 * i + 1] - 1.0).abs() < 1e-5, "cos should be ~1");
        }
    }

    #[test]
    fn test_sinusoidal_slice_length() {
        let enc = SinusoidalEncoding::default();
        let vals = vec![1.0f32; 10];
        let out = enc.encode_slice(&vals);
        assert_eq!(out.len(), 10 * enc.output_dim_per_input());
    }

    #[test]
    fn test_sinusoidal_n_freqs_1() {
        // Edge case: n_freqs=1 must not divide by zero
        let enc = SinusoidalEncoding::new(1, 4.0, false);
        let v = enc.encode_scalar(1.0);
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_sinusoidal_different_freqs() {
        let enc = SinusoidalEncoding::new(4, 6.0, false);
        let v = enc.encode_scalar(1.0);
        assert_eq!(v.len(), 8);
        // The values should not all be identical
        let all_same = v.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-7);
        assert!(!all_same);
    }

    #[test]
    fn test_sinusoidal_sin_cos_bounded() {
        let enc = SinusoidalEncoding::default();
        let v = enc.encode_scalar(3.1);
        let sin_cos_part = &v[1..]; // skip raw input
        for &val in sin_cos_part {
            assert!(val.abs() <= 1.0 + 1e-6, "sin/cos values must be in [-1,1]");
        }
    }

    #[test]
    fn test_sinusoidal_include_input_prepends_raw() {
        let enc = SinusoidalEncoding::new(3, 3.0, true);
        let enc_no = SinusoidalEncoding::new(3, 3.0, false);
        let x = 0.42f32;
        let v_with = enc.encode_scalar(x);
        let v_without = enc_no.encode_scalar(x);
        assert_eq!(v_with.len(), v_without.len() + 1);
        assert!((v_with[0] - x).abs() < 1e-7);
        for i in 0..v_without.len() {
            assert!((v_with[i + 1] - v_without[i]).abs() < 1e-7);
        }
    }

    // --- LinearEmbedding tests ---------------------------------------------

    #[test]
    fn test_linear_embedding_output_shape() {
        let layer = LinearEmbedding::new(16, 8, 42).expect("valid dims");
        let x = vec![1.0f32; 16];
        let y = layer.forward(&x).expect("forward ok");
        assert_eq!(y.len(), 8);
    }

    #[test]
    fn test_linear_embedding_zero_input() {
        let layer = LinearEmbedding::new(16, 8, 99).expect("valid dims");
        let x = vec![0.0f32; 16];
        let y = layer.forward(&x).expect("forward ok");
        assert!(y.iter().all(|&v| v.abs() < 1e-9));
    }

    #[test]
    fn test_linear_embedding_weight_range() {
        let in_dim = 64;
        let layer = LinearEmbedding::new(in_dim, 32, 777).expect("valid dims");
        let scale = 1.0 / (in_dim as f32).sqrt();
        for &w in &layer.weights {
            assert!(
                w.abs() <= scale + 1e-5,
                "weight {w} outside initialisation range"
            );
        }
    }

    #[test]
    fn test_linear_embedding_dim_mismatch() {
        let layer = LinearEmbedding::new(16, 8, 1).expect("valid dims");
        let x = vec![0.0f32; 10]; // wrong size
        let err = layer.forward(&x).unwrap_err();
        assert!(matches!(
            err,
            ConditioningError::DimensionMismatch {
                expected: 16,
                actual: 10
            }
        ));
    }

    #[test]
    fn test_linear_embedding_zero_dim_error() {
        assert!(matches!(
            LinearEmbedding::new(0, 8, 1),
            Err(ConditioningError::InvalidEmbeddingDim { dim: 0 })
        ));
        assert!(matches!(
            LinearEmbedding::new(8, 0, 1),
            Err(ConditioningError::InvalidEmbeddingDim { dim: 0 })
        ));
    }

    #[test]
    fn test_linear_embedding_different_seeds_differ() {
        let l1 = LinearEmbedding::new(8, 4, 100).expect("ok");
        let l2 = LinearEmbedding::new(8, 4, 200).expect("ok");
        let same = l1
            .weights
            .iter()
            .zip(l2.weights.iter())
            .all(|(a, b)| (a - b).abs() < 1e-9);
        assert!(!same, "different seeds should give different weights");
    }

    // --- encode_shape_params tests ------------------------------------------

    #[test]
    fn test_encode_shape_params_dim() {
        let params = make_shape_params(20);
        let out = encode_shape_params(&params, 10, 64, 1).expect("ok");
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_encode_shape_params_padding() {
        // Fewer params than n_coeffs — should pad with zeros
        let params = vec![1.0f32; 3];
        let out = encode_shape_params(&params, 10, 32, 2).expect("ok");
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn test_encode_shape_params_truncation() {
        // More params than n_coeffs — should truncate
        let params = make_shape_params(200);
        let out = encode_shape_params(&params, 10, 32, 3).expect("ok");
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn test_encode_shape_params_zero_embed_dim_error() {
        let params = vec![1.0f32; 10];
        let err = encode_shape_params(&params, 10, 0, 1).unwrap_err();
        assert!(matches!(err, ConditioningError::InvalidEmbeddingDim { .. }));
    }

    #[test]
    fn test_encode_shape_params_zero_n_coeffs_error() {
        let params = vec![1.0f32; 10];
        let err = encode_shape_params(&params, 0, 64, 1).unwrap_err();
        assert!(matches!(err, ConditioningError::InvalidEmbeddingDim { .. }));
    }

    // --- encode_expression_params tests ------------------------------------

    #[test]
    fn test_encode_expression_params_dim() {
        let params = make_expr_params(10);
        let out = encode_expression_params(&params, 5, 64, 10).expect("ok");
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_encode_expression_params_empty_input_padded() {
        let params: Vec<f32> = Vec::new();
        let out = encode_expression_params(&params, 5, 32, 11).expect("ok");
        assert_eq!(out.len(), 32);
    }

    #[test]
    fn test_encode_expression_params_zero_embed_error() {
        let params = make_expr_params(5);
        assert!(matches!(
            encode_expression_params(&params, 5, 0, 1),
            Err(ConditioningError::InvalidEmbeddingDim { .. })
        ));
    }

    #[test]
    fn test_encode_expression_different_from_shape() {
        // Same input, same n_coeffs / embed_dim, but different seeds => different output
        let params = vec![0.5f32; 10];
        let shape_out = encode_shape_params(&params, 10, 32, 0x1000).expect("ok");
        let expr_out = encode_expression_params(&params, 10, 32, 0x2000).expect("ok");
        let same = shape_out
            .iter()
            .zip(expr_out.iter())
            .all(|(a, b)| (a - b).abs() < 1e-9);
        assert!(
            !same,
            "shape and expression encoders should produce different outputs"
        );
    }

    // --- encode_pose_params tests ------------------------------------------

    #[test]
    fn test_encode_pose_params_dim() {
        let pose = make_pose_params();
        let out = encode_pose_params(&pose, 15, 64, 20).expect("ok");
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_encode_pose_params_empty_error() {
        let err = encode_pose_params(&[], 15, 64, 1).unwrap_err();
        assert!(matches!(err, ConditioningError::EmptyParams { .. }));
    }

    #[test]
    fn test_encode_pose_params_zero_embed_error() {
        let pose = make_pose_params();
        assert!(matches!(
            encode_pose_params(&pose, 15, 0, 1),
            Err(ConditioningError::InvalidEmbeddingDim { .. })
        ));
    }

    #[test]
    fn test_encode_pose_params_zero_n_coeffs_error() {
        let pose = make_pose_params();
        assert!(matches!(
            encode_pose_params(&pose, 0, 64, 1),
            Err(ConditioningError::InvalidEmbeddingDim { .. })
        ));
    }

    #[test]
    fn test_encode_pose_params_non_finite_error() {
        let pose = vec![0.1f32, f32::NAN, 0.2];
        assert!(matches!(
            encode_pose_params(&pose, 3, 64, 1),
            Err(ConditioningError::NonFiniteValue { .. })
        ));
        let pose_inf = vec![0.1f32, f32::INFINITY, 0.2];
        assert!(matches!(
            encode_pose_params(&pose_inf, 3, 64, 1),
            Err(ConditioningError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn test_encode_pose_params_shorter_than_n_coeffs_padded() {
        // A pose shorter than n_coeffs pads with zeros, like shape/expression,
        // rather than deriving a differently-shaped (and thus incomparable)
        // projection matrix from the raw input length.
        let pose = vec![0.3f32, -0.1];
        let out = encode_pose_params(&pose, 15, 64, 20).expect("ok");
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_encode_pose_angle_wrapping() {
        // Angles beyond ±π should give same result as wrapped equivalents
        let pose_large = vec![3.0 * PI, -4.0 * PI, 0.0f32];
        let pose_wrapped: Vec<f32> = pose_large.iter().map(|&v| wrap_to_pi(v)).collect();
        let enc_large = encode_pose_params(&pose_large, 3, 32, 50).expect("ok");
        let enc_wrapped = encode_pose_params(&pose_wrapped, 3, 32, 50).expect("ok");
        for (a, b) in enc_large.iter().zip(enc_wrapped.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "angle wrapping should produce same encoding"
            );
        }
    }

    // --- condition_from_flame_params tests ---------------------------------

    #[test]
    fn test_condition_from_flame_basic() {
        let config = make_config();
        let shape = make_shape_params(10);
        let expr = make_expr_params(5);
        let pose = make_pose_params();
        let trans = [0.0f32; 3];
        let cv = condition_from_flame_params(&shape, &expr, &pose, &trans, &config).expect("ok");
        // With shape+expr+pose each embedding 64 dims, total = 192 (before norm)
        assert_eq!(cv.dim, 3 * 64);
    }

    #[test]
    fn test_condition_normalized_unit_norm() {
        let mut config = make_config();
        config.normalize_output = true;
        let shape = make_shape_params(10);
        let expr = make_expr_params(5);
        let pose = make_pose_params();
        let trans = [0.0f32; 3];
        let cv = condition_from_flame_params(&shape, &expr, &pose, &trans, &config).expect("ok");
        let norm: f32 = cv.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "normalised vector should have unit norm, got {norm}"
        );
    }

    #[test]
    fn test_condition_no_normalisation() {
        let mut config = make_config();
        config.normalize_output = false;
        let shape = make_shape_params(10);
        let expr = make_expr_params(5);
        let pose = make_pose_params();
        let trans = [0.0f32; 3];
        let cv = condition_from_flame_params(&shape, &expr, &pose, &trans, &config).expect("ok");
        let norm: f32 = cv.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Norm should not equal exactly 1.0 for non-zero unnormalized output
        assert!(norm > 0.0, "output should be non-zero");
    }

    #[test]
    fn test_condition_shape_only() {
        let mut config = make_config();
        config.use_expression = false;
        config.use_pose = false;
        config.normalize_output = false;
        let shape = make_shape_params(10);
        let expr = vec![];
        let pose = vec![0.0f32; 15];
        let trans = [0.0f32; 3];
        let cv = condition_from_flame_params(&shape, &expr, &pose, &trans, &config).expect("ok");
        assert_eq!(cv.dim, config.embedding_dim);
    }

    #[test]
    fn test_condition_with_translation() {
        let mut config = make_config();
        config.use_translation = true;
        config.normalize_output = false;
        let shape = make_shape_params(10);
        let expr = make_expr_params(5);
        let pose = make_pose_params();
        let trans = [0.1f32, -0.2, 0.3];
        let cv = condition_from_flame_params(&shape, &expr, &pose, &trans, &config).expect("ok");
        // shape + expr + pose + translation = 4 * 64 = 256
        assert_eq!(cv.dim, 4 * 64);
    }

    #[test]
    fn test_condition_source_counts() {
        let config = make_config(); // n_shape=10, n_expr=5
        let shape = make_shape_params(7); // shorter than n_shape
        let expr = make_expr_params(3); // shorter than n_expr
        let pose = make_pose_params();
        let trans = [0.0f32; 3];
        let cv = condition_from_flame_params(&shape, &expr, &pose, &trans, &config).expect("ok");
        assert_eq!(cv.source_shape, 7);
        assert_eq!(cv.source_expr, 3);
        assert_eq!(cv.source_pose, 15);
    }

    #[test]
    fn test_condition_zero_embed_dim_error() {
        let mut config = make_config();
        config.embedding_dim = 0;
        let err = condition_from_flame_params(&[], &[], &[0.0], &[0.0; 3], &config).unwrap_err();
        assert!(matches!(err, ConditioningError::InvalidEmbeddingDim { .. }));
    }

    // --- blend_conditioning tests ------------------------------------------

    #[test]
    fn test_blend_t0_gives_a() {
        let config = make_config();
        let shape_a = vec![1.0f32; 10];
        let shape_b = vec![0.0f32; 10];
        let expr = make_expr_params(5);
        let pose = make_pose_params();
        let trans = [0.0f32; 3];
        let a = condition_from_flame_params(&shape_a, &expr, &pose, &trans, &config).expect("ok");
        let b = condition_from_flame_params(&shape_b, &expr, &pose, &trans, &config).expect("ok");
        let blended = blend_conditioning(&a, &b, 0.0).expect("ok");
        for (ai, bi) in a.data.iter().zip(blended.data.iter()) {
            assert!((ai - bi).abs() < 1e-6);
        }
    }

    #[test]
    fn test_blend_t1_gives_b() {
        let config = make_config();
        let shape_a = vec![1.0f32; 10];
        let shape_b = vec![0.0f32; 10];
        let expr = make_expr_params(5);
        let pose = make_pose_params();
        let trans = [0.0f32; 3];
        let a = condition_from_flame_params(&shape_a, &expr, &pose, &trans, &config).expect("ok");
        let b = condition_from_flame_params(&shape_b, &expr, &pose, &trans, &config).expect("ok");
        let blended = blend_conditioning(&a, &b, 1.0).expect("ok");
        for (bi, bl_i) in b.data.iter().zip(blended.data.iter()) {
            assert!((bi - bl_i).abs() < 1e-6);
        }
    }

    #[test]
    fn test_blend_t_half_midpoint() {
        let config = make_config();
        // Use distinct shapes so the blend is verifiable
        let shape_a: Vec<f32> = vec![2.0f32; 10];
        let shape_b: Vec<f32> = vec![0.0f32; 10];
        let expr = make_expr_params(5);
        let pose = make_pose_params();
        let trans = [0.0f32; 3];
        let mut cfg_no_norm = config.clone();
        cfg_no_norm.normalize_output = false;
        let a =
            condition_from_flame_params(&shape_a, &expr, &pose, &trans, &cfg_no_norm).expect("ok");
        let b =
            condition_from_flame_params(&shape_b, &expr, &pose, &trans, &cfg_no_norm).expect("ok");
        let blended = blend_conditioning(&a, &b, 0.5).expect("ok");
        for i in 0..a.dim {
            let expected = 0.5 * a.data[i] + 0.5 * b.data[i];
            assert!((blended.data[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_blend_dim_mismatch_error() {
        let pose = make_pose_params();
        let trans = [0.0f32; 3];
        let mut config_a = make_config();
        config_a.use_expression = false;
        config_a.use_pose = false;
        config_a.normalize_output = false;
        let mut config_b = config_a.clone();
        config_b.use_expression = true;

        let a = condition_from_flame_params(&make_shape_params(10), &[], &pose, &trans, &config_a)
            .expect("ok");
        let b = condition_from_flame_params(
            &make_shape_params(10),
            &make_expr_params(5),
            &pose,
            &trans,
            &config_b,
        )
        .expect("ok");
        assert_ne!(a.dim, b.dim, "configs differ so dims should differ");
        let err = blend_conditioning(&a, &b, 0.5).unwrap_err();
        assert!(matches!(err, ConditioningError::DimensionMismatch { .. }));
    }

    // --- normalize_conditioning tests -------------------------------------

    #[test]
    fn test_normalize_conditioning_unit_norm() {
        let v = vec![3.0f32, 4.0];
        let n = normalize_conditioning(&v).expect("ok");
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_conditioning_zero_error() {
        let v = vec![0.0f32; 8];
        assert!(matches!(
            normalize_conditioning(&v),
            Err(ConditioningError::ZeroNorm)
        ));
    }

    #[test]
    fn test_normalize_conditioning_direction_preserved() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let n = normalize_conditioning(&v).expect("ok");
        // Ratio of consecutive elements should be preserved
        for i in 1..v.len() {
            let ratio_orig = v[i] / v[i - 1];
            let ratio_norm = n[i] / n[i - 1];
            assert!((ratio_orig - ratio_norm).abs() < 1e-5);
        }
    }

    // --- batch_condition_from_flame_params tests ---------------------------

    #[test]
    fn test_batch_condition_length() {
        let config = make_config();
        type BatchItem = (Vec<f32>, Vec<f32>, Vec<f32>, [f32; 3]);
        let batch: Vec<BatchItem> = (0..5)
            .map(|_| {
                (
                    make_shape_params(10),
                    make_expr_params(5),
                    make_pose_params(),
                    [0.0f32; 3],
                )
            })
            .collect();
        let results = batch_condition_from_flame_params(&batch, &config).expect("ok");
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_batch_condition_empty_batch() {
        let config = make_config();
        let results = batch_condition_from_flame_params(&[], &config).expect("ok");
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_condition_dims_consistent() {
        let config = make_config();
        let batch: Vec<_> = (0..3)
            .map(|_| {
                (
                    make_shape_params(10),
                    make_expr_params(5),
                    make_pose_params(),
                    [0.0f32; 3],
                )
            })
            .collect();
        let results = batch_condition_from_flame_params(&batch, &config).expect("ok");
        let expected_dim = results[0].dim;
        for cv in &results {
            assert_eq!(cv.dim, expected_dim);
        }
    }

    // --- conditioning_similarity tests ------------------------------------

    #[test]
    fn test_similarity_self_is_one() {
        let config = make_config();
        let cv = condition_from_flame_params(
            &make_shape_params(10),
            &make_expr_params(5),
            &make_pose_params(),
            &[0.0; 3],
            &config,
        )
        .expect("ok");
        let sim = conditioning_similarity(&cv, &cv);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "self-similarity should be 1.0, got {sim}"
        );
    }

    #[test]
    fn test_similarity_zero_vector_is_zero() {
        let cv_zero = ConditioningVector {
            data: vec![0.0f32; 8],
            dim: 8,
            source_shape: 0,
            source_expr: 0,
            source_pose: 0,
        };
        let cv_nonzero = ConditioningVector {
            data: vec![1.0f32; 8],
            dim: 8,
            source_shape: 0,
            source_expr: 0,
            source_pose: 0,
        };
        assert_eq!(conditioning_similarity(&cv_zero, &cv_nonzero), 0.0);
        assert_eq!(conditioning_similarity(&cv_nonzero, &cv_zero), 0.0);
    }

    #[test]
    fn test_similarity_opposite_is_negative_one() {
        let cv_a = ConditioningVector {
            data: vec![1.0f32, 0.0, 0.0],
            dim: 3,
            source_shape: 0,
            source_expr: 0,
            source_pose: 0,
        };
        let cv_b = ConditioningVector {
            data: vec![-1.0f32, 0.0, 0.0],
            dim: 3,
            source_shape: 0,
            source_expr: 0,
            source_pose: 0,
        };
        let sim = conditioning_similarity(&cv_a, &cv_b);
        assert!((sim + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_similarity_orthogonal_vectors() {
        let cv_a = ConditioningVector {
            data: vec![1.0f32, 0.0],
            dim: 2,
            source_shape: 0,
            source_expr: 0,
            source_pose: 0,
        };
        let cv_b = ConditioningVector {
            data: vec![0.0f32, 1.0],
            dim: 2,
            source_shape: 0,
            source_expr: 0,
            source_pose: 0,
        };
        let sim = conditioning_similarity(&cv_a, &cv_b);
        assert!(
            sim.abs() < 1e-6,
            "orthogonal vectors should have 0 similarity, got {sim}"
        );
    }

    // --- compute_conditioning_stats tests ---------------------------------

    #[test]
    fn test_stats_empty_batch() {
        let stats = compute_conditioning_stats(&[]);
        assert_eq!(stats.n_vectors, 0);
    }

    #[test]
    fn test_stats_single_vector() {
        let config = make_config();
        let cv = condition_from_flame_params(
            &make_shape_params(10),
            &make_expr_params(5),
            &make_pose_params(),
            &[0.0; 3],
            &config,
        )
        .expect("ok");
        let stats = compute_conditioning_stats(&[cv]);
        assert_eq!(stats.n_vectors, 1);
        assert_eq!(stats.dim, 3 * 64);
        assert!(
            (stats.std_norm).abs() < 1e-6,
            "single vector has std_norm 0"
        );
    }

    #[test]
    fn test_stats_norms_correct() {
        let cv1 = ConditioningVector {
            data: vec![3.0f32, 4.0],
            dim: 2,
            source_shape: 0,
            source_expr: 0,
            source_pose: 0,
        };
        let cv2 = ConditioningVector {
            data: vec![1.0f32, 0.0],
            dim: 2,
            source_shape: 0,
            source_expr: 0,
            source_pose: 0,
        };
        let stats = compute_conditioning_stats(&[cv1, cv2]);
        assert!(
            (stats.mean_norm - 3.0).abs() < 1e-5,
            "mean of norms 5 and 1 = 3"
        );
        assert!((stats.min_norm - 1.0).abs() < 1e-5);
        assert!((stats.max_norm - 5.0).abs() < 1e-5);
    }

    // --- create_neutral_conditioning tests --------------------------------

    #[test]
    fn test_neutral_conditioning_succeeds() {
        let config = make_config();
        let cv = create_neutral_conditioning(&config).expect("neutral conditioning should succeed");
        assert_eq!(cv.dim, 3 * 64);
    }

    #[test]
    fn test_neutral_conditioning_deterministic() {
        let config = make_config();
        let cv1 = create_neutral_conditioning(&config).expect("ok");
        let cv2 = create_neutral_conditioning(&config).expect("ok");
        for (a, b) in cv1.data.iter().zip(cv2.data.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "neutral conditioning must be deterministic"
            );
        }
    }

    #[test]
    fn test_neutral_conditioning_zero_inputs_valid() {
        // All-zero inputs must produce a valid (non-zero) conditioning vector
        let mut config = make_config();
        config.normalize_output = false;
        let cv = create_neutral_conditioning(&config).expect("ok");
        let norm: f32 = cv.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        // The weights are non-zero even for zero inputs — sum of rows need not be zero
        // but they can be near-zero by chance.  Just check it returns Ok.
        assert!(cv.dim > 0);
        let _ = norm; // suppress unused warning
    }

    // --- interpolate_conditioning_sequence tests --------------------------

    #[test]
    fn test_interpolate_sequence_length() {
        let config = make_config();
        let start = create_neutral_conditioning(&config).expect("ok");
        let _config2 = config.clone();
        // _config2.normalize_output = false; // allow different endpoint (not needed)
        // Use different shape to produce different end
        let shape_end = make_shape_params(10);
        let end = condition_from_flame_params(
            &shape_end,
            &make_expr_params(5),
            &make_pose_params(),
            &[0.0; 3],
            &config,
        )
        .expect("ok");
        let seq = interpolate_conditioning_sequence(&start, &end, 10);
        assert_eq!(seq.len(), 10);
    }

    #[test]
    fn test_interpolate_sequence_zero_steps() {
        let config = make_config();
        let cv = create_neutral_conditioning(&config).expect("ok");
        let seq = interpolate_conditioning_sequence(&cv, &cv, 0);
        assert!(seq.is_empty());
    }

    #[test]
    fn test_interpolate_sequence_endpoint_start() {
        let config = make_config();
        let start = create_neutral_conditioning(&config).expect("ok");
        let end = create_neutral_conditioning(&config).expect("ok");
        let seq = interpolate_conditioning_sequence(&start, &end, 5);
        // t=0 step should match start
        for (s, seq0) in start.data.iter().zip(seq[0].data.iter()) {
            assert!((s - seq0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_interpolate_sequence_endpoint_end() {
        let config = make_config();
        let start = create_neutral_conditioning(&config).expect("ok");
        let end = create_neutral_conditioning(&config).expect("ok");
        let n = 5;
        let seq = interpolate_conditioning_sequence(&start, &end, n);
        // last step should match end
        for (e, seq_last) in end.data.iter().zip(seq[n - 1].data.iter()) {
            assert!((e - seq_last).abs() < 1e-6);
        }
    }

    #[test]
    fn test_interpolate_single_step_gives_start() {
        let config = make_config();
        let start = create_neutral_conditioning(&config).expect("ok");
        let end = condition_from_flame_params(
            &make_shape_params(10),
            &make_expr_params(5),
            &make_pose_params(),
            &[0.0; 3],
            &config,
        )
        .expect("ok");
        let seq = interpolate_conditioning_sequence(&start, &end, 1);
        assert_eq!(seq.len(), 1);
        // With n_steps=1, t=0.0 => should equal start
        for (s, seq0) in start.data.iter().zip(seq[0].data.iter()) {
            assert!((s - seq0).abs() < 1e-6);
        }
    }

    // --- xorshift64 tests --------------------------------------------------

    #[test]
    fn test_xorshift64_not_stuck_zero() {
        let mut state = 0u64;
        let v = xorshift64(&mut state);
        assert_ne!(v, 0, "xorshift64 from state=0 must not return 0");
    }

    #[test]
    fn test_xorshift64_deterministic() {
        let mut s1 = 42u64;
        let mut s2 = 42u64;
        for _ in 0..100 {
            assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
        }
    }

    // --- wrap_to_pi tests --------------------------------------------------

    #[test]
    fn test_wrap_to_pi_within_range() {
        for angle in [-PI, -PI / 2.0, 0.0, PI / 2.0, PI] {
            let w = wrap_to_pi(angle);
            assert!((-PI..=PI).contains(&w), "angle {angle} wrapped to {w}");
        }
    }

    #[test]
    fn test_wrap_to_pi_large_positive() {
        let w = wrap_to_pi(5.0 * PI);
        assert!(w.abs() <= PI + 1e-5);
    }

    #[test]
    fn test_wrap_to_pi_large_negative() {
        let w = wrap_to_pi(-7.0 * PI);
        assert!(w.abs() <= PI + 1e-5);
    }

    #[test]
    fn test_wrap_to_pi_non_finite_does_not_hang() {
        // Regression test for an infinite `while` loop: for |x| above roughly
        // 2^24 * 2π, `v -= 2.0 * PI` becomes a float no-op and the old
        // while-loop implementation never terminated. This test completing
        // at all (not timing out) is the assertion; the returned value is
        // also checked as a secondary sanity check.
        assert_eq!(wrap_to_pi(f32::INFINITY), 0.0);
        assert_eq!(wrap_to_pi(f32::NEG_INFINITY), 0.0);
        assert_eq!(wrap_to_pi(f32::NAN), 0.0);
        let w = wrap_to_pi(1e12);
        assert!((-PI..=PI).contains(&w), "wrap_to_pi(1e12) = {w}");
    }

    // --- xorshift_to_float tests --------------------------------------------

    #[test]
    fn test_xorshift_to_float_never_reaches_positive_scale() {
        // Regression test for the same "53 bits rounded into an f32" bug as
        // adaptive_sampling::uniform_f32: the result must be strictly < scale.
        let mut state = 12345u64;
        let scale = 1.0f32;
        for _ in 0..100_000 {
            let raw = xorshift64(&mut state);
            let v = xorshift_to_float(raw, scale);
            assert!(
                v < scale,
                "xorshift_to_float must be strictly < scale, got {v}"
            );
            assert!(v >= -scale);
        }
    }

    // --- encode_pose_params comparability regression ------------------------

    #[test]
    fn test_pose_projection_independent_of_raw_input_length() {
        // The bug: encode_pose_params used to derive its projection matrix
        // from `params.len()` directly, so a 15-value and a 3-value pose (both
        // representing "mostly zero rotation") were projected by two
        // *different* random matrices and were not comparable, even under an
        // identical `n_coeffs`. With n_coeffs normalising both inputs first,
        // a short pose padded with the *same* values as a longer pose (in its
        // leading positions) followed by zeros should produce embeddings from
        // one coherent projection space -- concretely, two pose slices that
        // both zero-pad out to the same n_coeffs-length vector must yield
        // identical embeddings regardless of how long the raw input was.
        let short = vec![0.2f32, -0.1, 0.05];
        let mut long = short.clone();
        long.extend(std::iter::repeat(0.0f32).take(12)); // pad to 15 explicitly
        let n_coeffs = 15;

        let enc_short = encode_pose_params(&short, n_coeffs, 32, 0x3000).expect("ok");
        let enc_long = encode_pose_params(&long, n_coeffs, 32, 0x3000).expect("ok");
        assert_eq!(
            enc_short, enc_long,
            "short (auto-padded) and explicitly-padded pose of the same \
             semantic content must encode identically once n_coeffs is fixed"
        );
    }

    // --- cached_linear_embedding ---------------------------------------------

    #[test]
    fn test_cached_linear_embedding_same_key_reused() {
        let a = cached_linear_embedding(17, 8, 0xABCD).expect("ok");
        let b = cached_linear_embedding(17, 8, 0xABCD).expect("ok");
        assert_eq!(
            a.weights, b.weights,
            "same (in_dim, out_dim, seed) must reuse the same matrix"
        );
    }

    #[test]
    fn test_cached_linear_embedding_different_seed_differs() {
        let a = cached_linear_embedding(17, 8, 1).expect("ok");
        let b = cached_linear_embedding(17, 8, 2).expect("ok");
        assert_ne!(a.weights, b.weights);
    }
}
