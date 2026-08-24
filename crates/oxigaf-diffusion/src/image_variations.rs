//! # image_variations
//!
//! Image variation generation for the diffusion pipeline.
//!
//! Provides tools for generating diverse variations of a latent code via noise
//! injection, spherical interpolation, and guided perturbation. All operations
//! are pure-Rust and use an inline xorshift64 PRNG with Box-Muller transform
//! for Gaussian noise — no external `rand` crate required.
//!
//! ## Layout convention
//!
//! [`VarLatentVector`] stores data in channels-first, row-major order:
//! `index = c * (H * W) + h * W + w`
//!
//! ## Example
//! ```rust
//! use oxigaf_diffusion::image_variations::{
//!     VarLatentVector, ImageVariationConfig, VariationMode, generate_variations,
//! };
//!
//! let base = VarLatentVector::new(4, 8, 8);
//! let config = ImageVariationConfig {
//!     num_variations: 3,
//!     noise_scale: 0.1,
//!     seed: 42,
//!     mode: VariationMode::Latent,
//!     temperature: 1.0,
//! };
//! let variations = generate_variations(&base, &config).unwrap();
//! assert_eq!(variations.len(), 3);
//! ```

use thiserror::Error;

use crate::classifier_guidance::ScoreFunction;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by image-variation operations.
#[derive(Debug, Error)]
pub enum ImageVariationError {
    /// Invalid configuration parameter.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// Invalid image data or shape.
    #[error("Invalid image: {0}")]
    InvalidImage(String),

    /// Encoding failed.
    #[error("Encoding error: {0}")]
    EncodingError(String),

    /// Decoding failed.
    #[error("Decoding error: {0}")]
    DecodingError(String),

    /// Numerical computation failed.
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Dimension mismatch between two operands.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Empty input where at least one element is required.
    #[error("Empty input")]
    EmptyInput,
}

// ---------------------------------------------------------------------------
// VariationMode
// ---------------------------------------------------------------------------

/// Strategy used to generate each image variation via [`generate_variations`].
#[derive(Debug, Clone, PartialEq)]
pub enum VariationMode {
    /// Add Gaussian noise directly to the latent code.
    Latent,
    /// Add Gaussian noise scaled by `noise_scale * temperature` — i.e.
    /// `Latent` with an extra temperature knob, *not* score-guided: this
    /// path has no score/gradient signal available ([`ImageVariationConfig`]
    /// carries none). For genuine score-weighted perturbation, call
    /// [`generate_guided_variations`] directly with a `ScoreFunction` (see
    /// `crate::classifier_guidance`).
    Guided,
    /// Linear interpolation between the base latent and a random latent.
    Interpolated,
}

// ---------------------------------------------------------------------------
// ImageVariationConfig
// ---------------------------------------------------------------------------

/// Configuration for variation generation.
#[derive(Debug, Clone)]
pub struct ImageVariationConfig {
    /// Number of variations to produce.
    pub num_variations: usize,
    /// Standard deviation of the additive Gaussian noise (default `0.1`).
    pub noise_scale: f32,
    /// Base seed for the PRNG.
    pub seed: u64,
    /// Variation strategy.
    pub mode: VariationMode,
    /// Temperature for `Guided` mode (default `1.0`).
    pub temperature: f32,
}

impl Default for ImageVariationConfig {
    fn default() -> Self {
        Self {
            num_variations: 4,
            noise_scale: 0.1,
            seed: 42,
            mode: VariationMode::Latent,
            temperature: 1.0,
        }
    }
}

impl ImageVariationConfig {
    /// Validate the configuration, returning an error describing the problem.
    pub fn validate(&self) -> Result<(), ImageVariationError> {
        if self.num_variations < 1 {
            return Err(ImageVariationError::InvalidConfig(
                "num_variations must be >= 1".to_string(),
            ));
        }
        if self.noise_scale <= 0.0 {
            return Err(ImageVariationError::InvalidConfig(
                "noise_scale must be > 0".to_string(),
            ));
        }
        if self.temperature <= 0.0 {
            return Err(ImageVariationError::InvalidConfig(
                "temperature must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VarLatentVector
// ---------------------------------------------------------------------------

/// A three-dimensional latent tensor stored in channels-first row-major layout.
///
/// Element at `(c, h, w)` lives at `data[c * (H * W) + h * W + w]`.
#[derive(Debug, Clone)]
pub struct VarLatentVector {
    /// Number of channels (C).
    pub channels: usize,
    /// Height (H).
    pub height: usize,
    /// Width (W).
    pub width: usize,
    /// Raw data in `[C, H, W]` layout.
    pub data: Vec<f32>,
}

impl VarLatentVector {
    /// Create a zero-initialised latent with the given spatial dimensions.
    pub fn new(channels: usize, height: usize, width: usize) -> Self {
        let n = channels * height * width;
        Self {
            channels,
            height,
            width,
            data: vec![0.0f32; n],
        }
    }

    /// Construct from an existing data buffer, verifying that its length
    /// matches `channels * height * width`.
    pub fn from_data(
        channels: usize,
        height: usize,
        width: usize,
        data: Vec<f32>,
    ) -> Result<Self, ImageVariationError> {
        let expected = channels * height * width;
        if data.len() != expected {
            return Err(ImageVariationError::DimensionMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self {
            channels,
            height,
            width,
            data,
        })
    }

    /// Read element at `(c, h, w)`, returning `None` when out of bounds.
    #[inline]
    pub fn get(&self, c: usize, h: usize, w: usize) -> Option<f32> {
        if c < self.channels && h < self.height && w < self.width {
            Some(self.data[c * (self.height * self.width) + h * self.width + w])
        } else {
            None
        }
    }

    /// Write `val` to `(c, h, w)`, returning an error when out of bounds.
    #[inline]
    pub fn set(
        &mut self,
        c: usize,
        h: usize,
        w: usize,
        val: f32,
    ) -> Result<(), ImageVariationError> {
        if c < self.channels && h < self.height && w < self.width {
            self.data[c * (self.height * self.width) + h * self.width + w] = val;
            Ok(())
        } else {
            Err(ImageVariationError::InvalidImage(format!(
                "index ({c}, {h}, {w}) out of bounds for shape ({}, {}, {})",
                self.channels, self.height, self.width
            )))
        }
    }

    /// Total number of elements: `channels * height * width`.
    #[inline]
    pub fn numel(&self) -> usize {
        self.channels * self.height * self.width
    }

    /// Shape as `(channels, height, width)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.channels, self.height, self.width)
    }

    /// L2 norm (Euclidean length) of the data vector.
    pub fn l2_norm(&self) -> f32 {
        self.data.iter().fold(0.0f32, |acc, &x| acc + x * x).sqrt()
    }

    /// Multiply every element by `factor` in-place.
    pub fn scale(&mut self, factor: f32) {
        for x in &mut self.data {
            *x *= factor;
        }
    }

    /// Element-wise addition in-place. Returns an error if the shapes differ.
    pub fn add_assign(&mut self, other: &VarLatentVector) -> Result<(), ImageVariationError> {
        if self.data.len() != other.data.len() {
            return Err(ImageVariationError::DimensionMismatch {
                expected: self.data.len(),
                actual: other.data.len(),
            });
        }
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ImageVariationStats
// ---------------------------------------------------------------------------

/// Summary statistics comparing a set of variations to a base latent.
#[derive(Debug, Clone)]
pub struct ImageVariationStats {
    /// Mean L2 distance from the base latent.
    pub mean_distance: f32,
    /// Maximum L2 distance from the base latent.
    pub max_distance: f32,
    /// Minimum L2 distance from the base latent.
    pub min_distance: f32,
    /// Standard deviation of L2 distances.
    pub std_distance: f32,
    /// Mean cosine similarity between base and each variation.
    pub mean_cosine_similarity: f32,
}

// ---------------------------------------------------------------------------
// Inline xorshift64 PRNG + Box-Muller
// ---------------------------------------------------------------------------

/// Advance one xorshift64 step, enforcing the non-zero invariant.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    // Ensure state is never zero (required by xorshift64 correctness).
    if *state == 0 {
        *state = 1;
    }
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Convert a `u64` to a uniform `f64` in `(0, 1]`.
/// The result is strictly positive so `ln` is always finite.
#[inline]
fn u64_to_f64_01(bits: u64) -> f64 {
    // Map to (0, 1]: (bits + 1) / (2^64)
    // Using f64 arithmetic; the +1 guarantees we never hit 0.
    let numer = (bits as u128 + 1) as f64;
    let denom = (u64::MAX as u128 + 1) as f64; // 2^64
    numer / denom
}

/// Generate a pair of independent standard-normal samples via Box-Muller.
///
/// Neither output will be NaN or ±∞ because `u1 ∈ (0, 1]` and `u2 ∈ (0, 1]`.
#[inline]
fn box_muller(state: &mut u64) -> (f32, f32) {
    let u1 = u64_to_f64_01(xorshift64(state));
    let u2 = u64_to_f64_01(xorshift64(state));
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    ((r * theta.cos()) as f32, (r * theta.sin()) as f32)
}

/// Decorrelate a `(base_seed, index)` pair into a well-mixed xorshift64 seed.
///
/// `xorshift64` has weak avalanche from seeds that are numerically close —
/// states differing only in the low bits (as `base_seed.wrapping_add(i)`
/// does for consecutive small `i`) stay close for the first several
/// outputs, so naively deriving each variation's seed as `base_seed + i`
/// produces visibly correlated low-frequency structure between adjacent
/// variations. Passing the pair through a SplitMix64-style finalizer first
/// breaks that correlation while staying fully deterministic.
#[inline]
fn mix_seed(base_seed: u64, index: u64) -> u64 {
    let mut z = base_seed.wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // xorshift64 requires non-zero state.
    z.max(1)
}

/// Generate a complete Gaussian-noise latent with the same shape as `template`.
fn gaussian_noise_latent(template: &VarLatentVector, scale: f32, seed: u64) -> VarLatentVector {
    let mut state: u64 = seed.max(1);
    let n = template.numel();
    let mut data = Vec::with_capacity(n);
    let pairs = n / 2;
    for _ in 0..pairs {
        let (z0, z1) = box_muller(&mut state);
        data.push(z0 * scale);
        data.push(z1 * scale);
    }
    if n % 2 == 1 {
        let (z0, _) = box_muller(&mut state);
        data.push(z0 * scale);
    }
    VarLatentVector {
        channels: template.channels,
        height: template.height,
        width: template.width,
        data,
    }
}

// ---------------------------------------------------------------------------
// Core free functions
// ---------------------------------------------------------------------------

/// Add Gaussian noise scaled by `scale` to all elements of `latent`.
///
/// Uses an inline xorshift64 + Box-Muller transform; no `rand` crate is used.
/// The returned latent has the same shape as the input.
pub fn add_latent_noise(latent: &VarLatentVector, scale: f32, seed: u64) -> VarLatentVector {
    let noise = gaussian_noise_latent(latent, scale, seed);
    let mut result = latent.clone();
    for (dst, &n) in result.data.iter_mut().zip(noise.data.iter()) {
        *dst += n;
    }
    result
}

/// Linear interpolation between two latents: `(1-t)*a + t*b`.
///
/// Returns an error if the shapes differ.
pub fn interpolate_latents(
    a: &VarLatentVector,
    b: &VarLatentVector,
    t: f32,
) -> Result<VarLatentVector, ImageVariationError> {
    if a.data.len() != b.data.len() {
        return Err(ImageVariationError::DimensionMismatch {
            expected: a.data.len(),
            actual: b.data.len(),
        });
    }
    let data: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&ai, &bi)| (1.0 - t) * ai + t * bi)
        .collect();
    Ok(VarLatentVector {
        channels: a.channels,
        height: a.height,
        width: a.width,
        data,
    })
}

/// Spherical linear interpolation (slerp) between two latents.
///
/// Falls back to ordinary linear interpolation when the angle between the
/// vectors is smaller than `1e-6` radians, or when either vector is zero.
pub fn spherical_interpolate_latents(
    a: &VarLatentVector,
    b: &VarLatentVector,
    t: f32,
) -> Result<VarLatentVector, ImageVariationError> {
    if a.data.len() != b.data.len() {
        return Err(ImageVariationError::DimensionMismatch {
            expected: a.data.len(),
            actual: b.data.len(),
        });
    }
    let norm_a = a.l2_norm();
    let norm_b = b.l2_norm();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        // At least one vector is zero — fall back to lerp.
        return interpolate_latents(a, b, t);
    }
    let dot: f32 = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&ai, &bi)| ai * bi)
        .sum::<f32>()
        / (norm_a * norm_b);
    // Clamp to [-1, 1] to guard against floating-point overshoot.
    let dot = dot.clamp(-1.0, 1.0);
    let angle = dot.acos(); // ∈ [0, π]
    if angle < 1e-6 {
        return interpolate_latents(a, b, t);
    }
    let sin_angle = angle.sin();
    let coeff_a = ((1.0 - t) * angle).sin() / sin_angle;
    let coeff_b = (t * angle).sin() / sin_angle;
    let data: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&ai, &bi)| coeff_a * ai + coeff_b * bi)
        .collect();
    Ok(VarLatentVector {
        channels: a.channels,
        height: a.height,
        width: a.width,
        data,
    })
}

/// Generate `config.num_variations` variations of `base` according to the
/// chosen [`VariationMode`].
///
/// | Mode          | Description |
/// |---------------|-------------|
/// | `Latent`      | Add Gaussian noise seeded differently per variation. |
/// | `Guided`      | Same as `Latent` but noise is additionally scaled by `temperature`. |
/// | `Interpolated`| Lerp between `base` and a fresh random latent. |
pub fn generate_variations(
    base: &VarLatentVector,
    config: &ImageVariationConfig,
) -> Result<Vec<VarLatentVector>, ImageVariationError> {
    config.validate()?;
    let mut variations = Vec::with_capacity(config.num_variations);
    for i in 0..config.num_variations {
        // Derive a unique, decorrelated seed for each variation.
        let variation_seed = mix_seed(config.seed, i as u64);
        let var = match config.mode {
            VariationMode::Latent => add_latent_noise(base, config.noise_scale, variation_seed),
            VariationMode::Guided => {
                let effective_scale = config.noise_scale * config.temperature;
                add_latent_noise(base, effective_scale, variation_seed)
            }
            VariationMode::Interpolated => {
                // Build a random latent and lerp.
                let random_latent = gaussian_noise_latent(base, 1.0, variation_seed);
                // Use a deterministic interpolation factor per variation.
                let t = (i as f32 + 1.0) / (config.num_variations as f32 + 1.0);
                interpolate_latents(base, &random_latent, t * config.noise_scale)?
            }
        };
        variations.push(var);
    }
    Ok(variations)
}

/// Generate `config.num_variations` *genuinely* score-guided variations of
/// `base`.
///
/// Each variation combines two components:
/// 1. Isotropic Gaussian noise (same as [`VariationMode::Latent`]) for
///    diversity, scaled by `config.noise_scale`.
/// 2. A bias along the estimated gradient of `score_fn` (steepest local
///    ascent, via central finite differences — no autograd is available in
///    pure Rust), scaled so that at `temperature == 1.0` its magnitude
///    matches the *typical* norm of the noise component
///    (`noise_scale * sqrt(numel)`); `temperature` scales this bias
///    linearly, and `temperature == 0.0` degenerates to plain
///    [`VariationMode::Latent`] noise (rejected by [`ImageVariationConfig::validate`],
///    which requires `temperature > 0.0` — call [`add_latent_noise`] directly
///    if you want an unguided baseline).
///
/// This is what [`VariationMode::Guided`] is documented as, but — routed
/// through [`generate_variations`] — cannot actually perform, because
/// [`ImageVariationConfig`] has nowhere to carry a score function. Call this
/// function directly whenever a concrete guidance signal is available; see
/// `crate::classifier_guidance` for ready-made [`ScoreFunction`]s (e.g.
/// `MeanMaximizer`, `TargetProximity`).
///
/// # Errors
///
/// - Propagates [`ImageVariationConfig::validate`] errors.
/// - [`ImageVariationError::NumericalError`] if `score_fn` fails to evaluate.
///
/// # Performance
///
/// The gradient estimate costs `2 * base.numel()` calls to `score_fn` per
/// variation (central differences, one dimension at a time) — for a large
/// latent this can dominate runtime. `crate::classifier_guidance` offers a
/// stochastic (SPSA) gradient estimator for that regime; this function
/// always uses the exact per-dimension estimate for simplicity.
pub fn generate_guided_variations<S: ScoreFunction>(
    base: &VarLatentVector,
    config: &ImageVariationConfig,
    score_fn: &S,
) -> Result<Vec<VarLatentVector>, ImageVariationError> {
    config.validate()?;
    let eps = 1e-3_f32;
    let mut variations = Vec::with_capacity(config.num_variations);
    for i in 0..config.num_variations {
        let variation_seed = mix_seed(config.seed, i as u64);
        // Isotropic noise component (diversity), same as `Latent` mode.
        let noisy = add_latent_noise(base, config.noise_scale, variation_seed);
        // Gradient bias component (guidance): unit-norm direction of steepest
        // local score ascent at the noisy point, scaled so `temperature == 1`
        // roughly matches the noise component's own typical magnitude.
        let direction = score_gradient_direction(&noisy.data, score_fn, eps)?;
        let grad_scale = config.temperature * config.noise_scale * (noisy.data.len() as f32).sqrt();
        let data: Vec<f32> = noisy
            .data
            .iter()
            .zip(direction.iter())
            .map(|(&x, &g)| x + grad_scale * g)
            .collect();
        variations.push(VarLatentVector {
            channels: base.channels,
            height: base.height,
            width: base.width,
            data,
        });
    }
    Ok(variations)
}

/// Central-difference gradient of `score_fn` at `latent`, normalised to unit
/// L2 norm (a pure *direction*; its magnitude carries no information about
/// the score function's own scale, by design — callers apply their own
/// scale). Returns an all-zero vector when the gradient's norm is ~0 (a
/// local optimum, or a constant score function).
fn score_gradient_direction<S: ScoreFunction>(
    latent: &[f32],
    score_fn: &S,
    eps: f32,
) -> Result<Vec<f32>, ImageVariationError> {
    let mut grad = vec![0.0f32; latent.len()];
    let mut probe = latent.to_vec();
    for i in 0..latent.len() {
        probe[i] = latent[i] + eps;
        let s_plus = score_fn
            .score(&probe)
            .map_err(|e| ImageVariationError::NumericalError(e.to_string()))?;
        probe[i] = latent[i] - eps;
        let s_minus = score_fn
            .score(&probe)
            .map_err(|e| ImageVariationError::NumericalError(e.to_string()))?;
        probe[i] = latent[i];
        grad[i] = (s_plus - s_minus) / (2.0 * eps);
    }
    let norm = grad.iter().fold(0.0f32, |acc, &x| acc + x * x).sqrt();
    if norm > f32::EPSILON {
        for g in grad.iter_mut() {
            *g /= norm;
        }
    }
    Ok(grad)
}

/// L2 distance between two latent vectors.
pub fn latent_distance(
    a: &VarLatentVector,
    b: &VarLatentVector,
) -> Result<f32, ImageVariationError> {
    if a.data.len() != b.data.len() {
        return Err(ImageVariationError::DimensionMismatch {
            expected: a.data.len(),
            actual: b.data.len(),
        });
    }
    let dist_sq: f32 = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&ai, &bi)| {
            let diff = ai - bi;
            diff * diff
        })
        .sum();
    Ok(dist_sq.sqrt())
}

/// Cosine similarity between two latent vectors.
///
/// Returns `0.0` if either vector has zero norm to avoid division by zero.
pub fn latent_cosine_similarity(
    a: &VarLatentVector,
    b: &VarLatentVector,
) -> Result<f32, ImageVariationError> {
    if a.data.len() != b.data.len() {
        return Err(ImageVariationError::DimensionMismatch {
            expected: a.data.len(),
            actual: b.data.len(),
        });
    }
    let norm_a = a.l2_norm();
    let norm_b = b.l2_norm();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return Ok(0.0);
    }
    let dot: f32 = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&ai, &bi)| ai * bi)
        .sum();
    Ok((dot / (norm_a * norm_b)).clamp(-1.0, 1.0))
}

/// Rank variations by average pairwise L2 distance (most diverse first).
///
/// Returns a `Vec<usize>` of indices into `variations`, sorted in descending
/// order of average pairwise distance to all other variations.  Ties are
/// broken by index (ascending).
///
/// Returns an empty `Vec` when `variations` is empty.
pub fn rank_by_diversity(variations: &[VarLatentVector]) -> Vec<usize> {
    let n = variations.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }
    // Compute average pairwise distance for each variation.
    let mut scores: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let avg = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    // Silently use 0.0 on dimension mismatch — callers are
                    // expected to pass homogeneous variation slices.
                    latent_distance(&variations[i], &variations[j]).unwrap_or(0.0)
                })
                .sum::<f32>()
                / (n - 1) as f32;
            (i, avg)
        })
        .collect();
    // Sort descending by score, then ascending by index for stability.
    scores.sort_by(|&(ia, fa), &(ib, fb)| {
        fb.partial_cmp(&fa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(ia.cmp(&ib))
    });
    scores.into_iter().map(|(i, _)| i).collect()
}

/// Compute summary statistics of a set of variations relative to `base`.
///
/// Returns an error when `variations` is empty.
pub fn compute_variation_stats(
    base: &VarLatentVector,
    variations: &[VarLatentVector],
) -> Result<ImageVariationStats, ImageVariationError> {
    if variations.is_empty() {
        return Err(ImageVariationError::EmptyInput);
    }
    let distances: Vec<f32> = variations
        .iter()
        .map(|v| latent_distance(base, v))
        .collect::<Result<Vec<_>, _>>()?;
    let cosines: Vec<f32> = variations
        .iter()
        .map(|v| latent_cosine_similarity(base, v))
        .collect::<Result<Vec<_>, _>>()?;
    let n = distances.len() as f32;
    let mean_distance = distances.iter().sum::<f32>() / n;
    let max_distance = distances.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_distance = distances.iter().cloned().fold(f32::INFINITY, f32::min);
    let variance = distances
        .iter()
        .map(|&d| {
            let diff = d - mean_distance;
            diff * diff
        })
        .sum::<f32>()
        / n;
    let std_distance = variance.sqrt();
    let mean_cosine_similarity = cosines.iter().sum::<f32>() / n;
    Ok(ImageVariationStats {
        mean_distance,
        max_distance,
        min_distance,
        std_distance,
        mean_cosine_similarity,
    })
}

/// Project a latent onto the unit sphere by L2-normalising it.
///
/// Returns an error if `latent` is the zero vector.
pub fn project_to_sphere(latent: &VarLatentVector) -> Result<VarLatentVector, ImageVariationError> {
    let norm = latent.l2_norm();
    if norm < f32::EPSILON {
        return Err(ImageVariationError::NumericalError(
            "cannot project a zero vector onto the unit sphere".to_string(),
        ));
    }
    let inv = 1.0 / norm;
    let data: Vec<f32> = latent.data.iter().map(|&x| x * inv).collect();
    Ok(VarLatentVector {
        channels: latent.channels,
        height: latent.height,
        width: latent.width,
        data,
    })
}

/// Compute per-channel mean and standard deviation across all spatial positions.
///
/// Returns `(means, stds)` where each `Vec<f32>` has length `latent.channels`.
/// An empty spatial extent (`H * W == 0`) returns `(vec![0.0; C], vec![0.0; C])`.
///
/// # Errors
///
/// [`ImageVariationError::DimensionMismatch`] if `latent.data.len() !=
/// latent.numel()`. `VarLatentVector`'s fields are all `pub`, so a
/// struct-literal-constructed value (or one whose `data` was truncated in
/// place) can desynchronize `data` from `channels`/`height`/`width`; without
/// this check, slicing per channel below would index out of bounds instead
/// of reporting the mismatch.
pub fn var_channel_statistics(
    latent: &VarLatentVector,
) -> Result<(Vec<f32>, Vec<f32>), ImageVariationError> {
    let expected = latent.numel();
    if latent.data.len() != expected {
        return Err(ImageVariationError::DimensionMismatch {
            expected,
            actual: latent.data.len(),
        });
    }
    let c = latent.channels;
    let spatial = latent.height * latent.width;
    let mut means = vec![0.0f32; c];
    let mut stds = vec![0.0f32; c];
    if spatial == 0 {
        return Ok((means, stds));
    }
    for ch in 0..c {
        let offset = ch * spatial;
        let slice = &latent.data[offset..offset + spatial];
        let mean = slice.iter().sum::<f32>() / spatial as f32;
        let var = slice
            .iter()
            .map(|&x| {
                let d = x - mean;
                d * d
            })
            .sum::<f32>()
            / spatial as f32;
        means[ch] = mean;
        stds[ch] = var.sqrt();
    }
    Ok((means, stds))
}

/// Clamp all values in `latent` to `[min_val, max_val]`.
///
/// # Errors
///
/// [`ImageVariationError::InvalidConfig`] if either bound is non-finite or
/// `min_val > max_val` — `f32::clamp` panics on exactly this input
/// (`"min > max, or either was NaN"`), and since this function previously
/// returned a plain `VarLatentVector`, there was no way to report the
/// problem instead of panicking.
pub fn clamp_latent(
    latent: &VarLatentVector,
    min_val: f32,
    max_val: f32,
) -> Result<VarLatentVector, ImageVariationError> {
    if !min_val.is_finite() || !max_val.is_finite() || min_val > max_val {
        return Err(ImageVariationError::InvalidConfig(format!(
            "clamp_latent bounds must be finite with min_val <= max_val, got min_val={min_val}, max_val={max_val}"
        )));
    }
    let data: Vec<f32> = latent
        .data
        .iter()
        .map(|&x| x.clamp(min_val, max_val))
        .collect();
    Ok(VarLatentVector {
        channels: latent.channels,
        height: latent.height,
        width: latent.width,
        data,
    })
}

// ---------------------------------------------------------------------------
// VariationExplorer
// ---------------------------------------------------------------------------

/// An interactive explorer that applies successive noisy perturbations to a
/// base latent, maintaining a history that can be reverted one step at a time.
#[derive(Debug, Clone)]
pub struct VariationExplorer {
    base: VarLatentVector,
    current: VarLatentVector,
    history: Vec<VarLatentVector>,
    config: ImageVariationConfig,
    step_count: usize,
}

impl VariationExplorer {
    /// Create a new explorer, validating the config up front.
    pub fn new(
        base: VarLatentVector,
        config: ImageVariationConfig,
    ) -> Result<Self, ImageVariationError> {
        config.validate()?;
        let current = base.clone();
        Ok(Self {
            base,
            current,
            history: Vec::new(),
            config,
            step_count: 0,
        })
    }

    /// Apply one noisy perturbation step and return a reference to the new
    /// current latent.  The previous state is pushed onto the history stack.
    pub fn step(&mut self) -> Result<&VarLatentVector, ImageVariationError> {
        // Derive a step-specific, decorrelated seed so each step is
        // deterministic but unique.
        let step_seed = mix_seed(self.config.seed, self.step_count as u64);
        let next = add_latent_noise(&self.current, self.config.noise_scale, step_seed);
        self.history.push(self.current.clone());
        self.current = next;
        self.step_count += 1;
        Ok(&self.current)
    }

    /// Pop one step from the history, restoring the previous state.
    ///
    /// Returns an error if the history is empty (i.e., we are at the base).
    pub fn revert(&mut self) -> Result<(), ImageVariationError> {
        match self.history.pop() {
            Some(prev) => {
                self.current = prev;
                if self.step_count > 0 {
                    self.step_count -= 1;
                }
                Ok(())
            }
            None => Err(ImageVariationError::InvalidConfig(
                "cannot revert: history is empty".to_string(),
            )),
        }
    }

    /// Reference to the current latent.
    pub fn current(&self) -> &VarLatentVector {
        &self.current
    }

    /// Total number of steps taken since construction (or last reset).
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Reset to the base latent, clearing history and step counter.
    pub fn reset(&mut self) {
        self.current = self.base.clone();
        self.history.clear();
        self.step_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers -----------------------------------------------------------

    fn make_latent(c: usize, h: usize, w: usize, fill: f32) -> VarLatentVector {
        VarLatentVector {
            channels: c,
            height: h,
            width: w,
            data: vec![fill; c * h * w],
        }
    }

    fn make_latent_ramp(c: usize, h: usize, w: usize) -> VarLatentVector {
        let n = c * h * w;
        VarLatentVector {
            channels: c,
            height: h,
            width: w,
            data: (0..n).map(|i| i as f32).collect(),
        }
    }

    fn default_config() -> ImageVariationConfig {
        ImageVariationConfig::default()
    }

    // ---- VarLatentVector ---------------------------------------------------

    #[test]
    fn test_latent_new_is_zero() {
        let l = VarLatentVector::new(2, 3, 4);
        assert_eq!(l.numel(), 24);
        assert!(l.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_latent_shape() {
        let l = VarLatentVector::new(4, 8, 16);
        assert_eq!(l.shape(), (4, 8, 16));
    }

    #[test]
    fn test_latent_from_data_ok() {
        let data = vec![1.0f32; 12];
        let l = VarLatentVector::from_data(2, 2, 3, data).unwrap();
        assert_eq!(l.numel(), 12);
    }

    #[test]
    fn test_latent_from_data_mismatch() {
        let data = vec![1.0f32; 10];
        let result = VarLatentVector::from_data(2, 2, 3, data);
        assert!(matches!(
            result,
            Err(ImageVariationError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_latent_get_set() {
        let mut l = VarLatentVector::new(2, 3, 4);
        l.set(1, 2, 3, 42.0).unwrap();
        assert_eq!(l.get(1, 2, 3), Some(42.0));
    }

    #[test]
    fn test_latent_get_out_of_bounds() {
        let l = VarLatentVector::new(2, 3, 4);
        assert_eq!(l.get(5, 0, 0), None);
    }

    #[test]
    fn test_latent_set_out_of_bounds() {
        let mut l = VarLatentVector::new(2, 3, 4);
        let result = l.set(5, 0, 0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_latent_l2_norm_zero() {
        let l = VarLatentVector::new(3, 3, 3);
        assert!((l.l2_norm() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_latent_l2_norm_known() {
        // Vector [3, 4] has norm 5.
        let l = VarLatentVector::from_data(1, 1, 2, vec![3.0, 4.0]).unwrap();
        assert!((l.l2_norm() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_latent_scale() {
        let mut l = make_latent(1, 1, 4, 2.0);
        l.scale(3.0);
        assert!(l.data.iter().all(|&x| (x - 6.0).abs() < 1e-6));
    }

    #[test]
    fn test_latent_add_assign_ok() {
        let mut a = make_latent(1, 1, 3, 1.0);
        let b = make_latent(1, 1, 3, 2.0);
        a.add_assign(&b).unwrap();
        assert!(a.data.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_latent_add_assign_mismatch() {
        let mut a = make_latent(1, 1, 3, 1.0);
        let b = make_latent(1, 1, 4, 2.0);
        assert!(a.add_assign(&b).is_err());
    }

    // ---- add_latent_noise --------------------------------------------------

    #[test]
    fn test_noise_is_applied() {
        let base = make_latent(1, 4, 4, 0.0);
        let noisy = add_latent_noise(&base, 1.0, 99);
        // With large scale and zero base, noisy should not be all zeros.
        let all_zero = noisy.data.iter().all(|&x| x == 0.0);
        assert!(!all_zero);
    }

    #[test]
    fn test_noise_scale_zero_approximates_identity() {
        // scale = 1e-9 should produce values very close to the base.
        let base = make_latent(1, 4, 4, 1.0);
        let noisy = add_latent_noise(&base, 1e-9, 1);
        let max_diff = noisy
            .data
            .iter()
            .zip(base.data.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-5, "max_diff = {max_diff}");
    }

    #[test]
    fn test_noise_different_seeds_give_different_results() {
        let base = make_latent(1, 4, 4, 0.0);
        let n1 = add_latent_noise(&base, 1.0, 1);
        let n2 = add_latent_noise(&base, 1.0, 2);
        let equal = n1.data.iter().zip(n2.data.iter()).all(|(&a, &b)| a == b);
        assert!(!equal);
    }

    #[test]
    fn test_noise_same_seed_is_deterministic() {
        let base = make_latent(1, 4, 4, 0.5);
        let n1 = add_latent_noise(&base, 0.5, 777);
        let n2 = add_latent_noise(&base, 0.5, 777);
        assert_eq!(n1.data, n2.data);
    }

    #[test]
    fn test_noise_preserves_shape() {
        let base = VarLatentVector::new(3, 8, 8);
        let noisy = add_latent_noise(&base, 0.1, 1);
        assert_eq!(noisy.shape(), base.shape());
    }

    // ---- mix_seed ------------------------------------------------------------

    #[test]
    fn test_mix_seed_avalanche() {
        // Consecutive indices must decorrelate: adjacent variation seeds
        // should differ in many bits, not just the low ones.
        let s0 = mix_seed(42, 0);
        let s1 = mix_seed(42, 1);
        let hamming = (s0 ^ s1).count_ones();
        assert!(
            hamming >= 20,
            "consecutive-index seeds should differ in many bits (avalanche), got {hamming} of 64"
        );
    }

    #[test]
    fn test_mix_seed_nonzero() {
        // xorshift64 requires non-zero state.
        for i in 0..1000u64 {
            assert_ne!(mix_seed(0, i), 0);
        }
    }

    #[test]
    fn test_mix_seed_deterministic() {
        assert_eq!(mix_seed(7, 3), mix_seed(7, 3));
    }

    // ---- interpolate_latents -----------------------------------------------

    #[test]
    fn test_lerp_t0_gives_a() {
        let a = make_latent_ramp(1, 2, 3);
        let b = make_latent(1, 2, 3, 100.0);
        let r = interpolate_latents(&a, &b, 0.0).unwrap();
        for (&ra, &aa) in r.data.iter().zip(a.data.iter()) {
            assert!((ra - aa).abs() < 1e-6);
        }
    }

    #[test]
    fn test_lerp_t1_gives_b() {
        let a = make_latent(1, 2, 3, 0.0);
        let b = make_latent(1, 2, 3, 5.0);
        let r = interpolate_latents(&a, &b, 1.0).unwrap();
        for (&rb, &bb) in r.data.iter().zip(b.data.iter()) {
            assert!((rb - bb).abs() < 1e-6);
        }
    }

    #[test]
    fn test_lerp_t05_is_midpoint() {
        let a = make_latent(1, 1, 4, 0.0);
        let b = make_latent(1, 1, 4, 4.0);
        let r = interpolate_latents(&a, &b, 0.5).unwrap();
        for &v in &r.data {
            assert!((v - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_lerp_dimension_mismatch() {
        let a = make_latent(1, 1, 3, 0.0);
        let b = make_latent(1, 1, 4, 1.0);
        let result = interpolate_latents(&a, &b, 0.5);
        assert!(matches!(
            result,
            Err(ImageVariationError::DimensionMismatch { .. })
        ));
    }

    // ---- spherical_interpolate_latents ------------------------------------

    #[test]
    fn test_slerp_t0_recovers_a() {
        let a = VarLatentVector::from_data(1, 1, 3, vec![1.0, 0.0, 0.0]).unwrap();
        let b = VarLatentVector::from_data(1, 1, 3, vec![0.0, 1.0, 0.0]).unwrap();
        let r = spherical_interpolate_latents(&a, &b, 0.0).unwrap();
        let diff: f32 = r
            .data
            .iter()
            .zip(a.data.iter())
            .map(|(&x, &y)| (x - y).abs())
            .sum();
        assert!(diff < 1e-5, "diff={diff}");
    }

    #[test]
    fn test_slerp_t1_recovers_b() {
        let a = VarLatentVector::from_data(1, 1, 3, vec![1.0, 0.0, 0.0]).unwrap();
        let b = VarLatentVector::from_data(1, 1, 3, vec![0.0, 1.0, 0.0]).unwrap();
        let r = spherical_interpolate_latents(&a, &b, 1.0).unwrap();
        let diff: f32 = r
            .data
            .iter()
            .zip(b.data.iter())
            .map(|(&x, &y)| (x - y).abs())
            .sum();
        assert!(diff < 1e-5, "diff={diff}");
    }

    #[test]
    fn test_slerp_opposite_vectors_fallback() {
        // a and b are antiparallel — slerp angle = π, sin(π) ≈ 0; falls back to lerp.
        let a = VarLatentVector::from_data(1, 1, 3, vec![1.0, 0.0, 0.0]).unwrap();
        let b = VarLatentVector::from_data(1, 1, 3, vec![-1.0, 0.0, 0.0]).unwrap();
        // Should not panic or NaN.
        let r = spherical_interpolate_latents(&a, &b, 0.5).unwrap();
        assert!(r.data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_slerp_zero_vector_fallback_to_lerp() {
        let a = VarLatentVector::from_data(1, 1, 3, vec![0.0, 0.0, 0.0]).unwrap();
        let b = VarLatentVector::from_data(1, 1, 3, vec![1.0, 0.0, 0.0]).unwrap();
        let r = spherical_interpolate_latents(&a, &b, 0.5).unwrap();
        assert!(r.data.iter().all(|x| x.is_finite()));
    }

    // ---- generate_variations -----------------------------------------------

    #[test]
    fn test_generate_latent_mode_count() {
        let base = VarLatentVector::new(4, 8, 8);
        let cfg = ImageVariationConfig {
            num_variations: 5,
            ..default_config()
        };
        let vars = generate_variations(&base, &cfg).unwrap();
        assert_eq!(vars.len(), 5);
    }

    #[test]
    fn test_generate_guided_mode() {
        let base = VarLatentVector::new(4, 8, 8);
        let cfg = ImageVariationConfig {
            mode: VariationMode::Guided,
            temperature: 2.0,
            num_variations: 3,
            ..default_config()
        };
        let vars = generate_variations(&base, &cfg).unwrap();
        assert_eq!(vars.len(), 3);
        // Each variation must differ from base (temperature > 0).
        for v in &vars {
            let dist = latent_distance(&base, v).unwrap();
            assert!(dist > 0.0);
        }
    }

    #[test]
    fn test_generate_interpolated_mode() {
        let base = make_latent(2, 4, 4, 1.0);
        let cfg = ImageVariationConfig {
            mode: VariationMode::Interpolated,
            num_variations: 4,
            ..default_config()
        };
        let vars = generate_variations(&base, &cfg).unwrap();
        assert_eq!(vars.len(), 4);
        for v in &vars {
            assert_eq!(v.shape(), base.shape());
        }
    }

    #[test]
    fn test_generate_num_zero_returns_error() {
        let base = VarLatentVector::new(4, 8, 8);
        let cfg = ImageVariationConfig {
            num_variations: 0,
            ..default_config()
        };
        assert!(generate_variations(&base, &cfg).is_err());
    }

    // ---- generate_guided_variations -----------------------------------------

    /// Toy score function: rewards latents whose mean is large (mirrors the
    /// `MeanMaximizer` example in `classifier_guidance`), so guided
    /// variations should trend toward a higher mean than the base latent.
    struct SumScore;

    impl ScoreFunction for SumScore {
        fn score(
            &self,
            latent: &[f32],
        ) -> Result<f32, crate::classifier_guidance::ClassifierGuidanceError> {
            Ok(latent.iter().sum())
        }

        fn name(&self) -> &str {
            "sum_score_test"
        }
    }

    #[test]
    fn test_generate_guided_variations_count_and_shape() {
        let base = VarLatentVector::new(2, 4, 4);
        let cfg = ImageVariationConfig {
            mode: VariationMode::Guided,
            num_variations: 3,
            temperature: 1.0,
            ..default_config()
        };
        let vars = generate_guided_variations(&base, &cfg, &SumScore).unwrap();
        assert_eq!(vars.len(), 3);
        for v in &vars {
            assert_eq!(v.shape(), base.shape());
        }
    }

    #[test]
    fn test_generate_guided_variations_moves_toward_higher_score() {
        let base = VarLatentVector::new(1, 4, 4);
        let cfg = ImageVariationConfig {
            mode: VariationMode::Guided,
            num_variations: 8,
            noise_scale: 0.05,
            temperature: 3.0,
            ..default_config()
        };
        let vars = generate_guided_variations(&base, &cfg, &SumScore).unwrap();
        // With a strong temperature relative to noise, the gradient bias
        // (pointing toward larger sum, i.e. all-ones direction) should
        // dominate: every variation's sum should end up positive.
        let base_score = SumScore.score(&base.data).unwrap();
        for v in &vars {
            let score = SumScore.score(&v.data).unwrap();
            assert!(
                score > base_score,
                "guided variation should score higher than base: {score} vs {base_score}"
            );
        }
    }

    #[test]
    fn test_generate_guided_variations_num_zero_returns_error() {
        let base = VarLatentVector::new(2, 4, 4);
        let cfg = ImageVariationConfig {
            num_variations: 0,
            ..default_config()
        };
        assert!(generate_guided_variations(&base, &cfg, &SumScore).is_err());
    }

    // ---- latent_distance ---------------------------------------------------

    #[test]
    fn test_distance_self_is_zero() {
        let a = make_latent_ramp(2, 3, 3);
        let d = latent_distance(&a, &a).unwrap();
        assert!(d.abs() < 1e-5);
    }

    #[test]
    fn test_distance_known_value() {
        let a = VarLatentVector::from_data(1, 1, 2, vec![0.0, 0.0]).unwrap();
        let b = VarLatentVector::from_data(1, 1, 2, vec![3.0, 4.0]).unwrap();
        let d = latent_distance(&a, &b).unwrap();
        assert!((d - 5.0).abs() < 1e-5, "d={d}");
    }

    #[test]
    fn test_distance_dimension_mismatch() {
        let a = make_latent(1, 1, 3, 0.0);
        let b = make_latent(1, 1, 4, 0.0);
        assert!(latent_distance(&a, &b).is_err());
    }

    // ---- latent_cosine_similarity ------------------------------------------

    #[test]
    fn test_cosine_parallel() {
        let a = VarLatentVector::from_data(1, 1, 3, vec![1.0, 0.0, 0.0]).unwrap();
        let b = VarLatentVector::from_data(1, 1, 3, vec![2.0, 0.0, 0.0]).unwrap();
        let sim = latent_cosine_similarity(&a, &b).unwrap();
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = VarLatentVector::from_data(1, 1, 3, vec![1.0, 0.0, 0.0]).unwrap();
        let b = VarLatentVector::from_data(1, 1, 3, vec![0.0, 1.0, 0.0]).unwrap();
        let sim = latent_cosine_similarity(&a, &b).unwrap();
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_zero_vector_returns_zero() {
        let a = make_latent(1, 1, 3, 0.0);
        let b = VarLatentVector::from_data(1, 1, 3, vec![1.0, 2.0, 3.0]).unwrap();
        let sim = latent_cosine_similarity(&a, &b).unwrap();
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_dimension_mismatch() {
        let a = make_latent(1, 1, 3, 1.0);
        let b = make_latent(1, 1, 4, 1.0);
        assert!(latent_cosine_similarity(&a, &b).is_err());
    }

    // ---- rank_by_diversity -------------------------------------------------

    #[test]
    fn test_rank_empty() {
        let result = rank_by_diversity(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rank_single() {
        let v = make_latent(1, 1, 3, 1.0);
        let result = rank_by_diversity(&[v]);
        assert_eq!(result, vec![0]);
    }

    #[test]
    fn test_rank_multiple_returns_all_indices() {
        let base = VarLatentVector::new(1, 1, 4);
        let cfg = default_config();
        let vars = generate_variations(&base, &cfg).unwrap();
        let ranking = rank_by_diversity(&vars);
        assert_eq!(ranking.len(), vars.len());
        let mut sorted = ranking.clone();
        sorted.sort();
        assert_eq!(sorted, (0..vars.len()).collect::<Vec<_>>());
    }

    // ---- compute_variation_stats -------------------------------------------

    #[test]
    fn test_stats_basic() {
        let base = make_latent(1, 1, 4, 0.0);
        let vars: Vec<VarLatentVector> = (1..=3).map(|i| make_latent(1, 1, 4, i as f32)).collect();
        let stats = compute_variation_stats(&base, &vars).unwrap();
        assert!(stats.mean_distance > 0.0);
        assert!(stats.max_distance >= stats.min_distance);
        assert!(stats.std_distance >= 0.0);
    }

    #[test]
    fn test_stats_empty_variations_returns_error() {
        let base = make_latent(1, 1, 4, 0.0);
        let result = compute_variation_stats(&base, &[]);
        assert!(matches!(result, Err(ImageVariationError::EmptyInput)));
    }

    // ---- project_to_sphere -------------------------------------------------

    #[test]
    fn test_project_unit_vector_unchanged() {
        let v = VarLatentVector::from_data(1, 1, 3, vec![1.0, 0.0, 0.0]).unwrap();
        let projected = project_to_sphere(&v).unwrap();
        let norm = projected.l2_norm();
        assert!((norm - 1.0).abs() < 1e-5);
        assert!((projected.data[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_project_arbitrary_vector_has_unit_norm() {
        let v = VarLatentVector::from_data(1, 1, 4, vec![3.0, 4.0, 0.0, 0.0]).unwrap();
        let projected = project_to_sphere(&v).unwrap();
        assert!((projected.l2_norm() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_project_zero_vector_error() {
        let v = make_latent(1, 2, 3, 0.0);
        let result = project_to_sphere(&v);
        assert!(matches!(
            result,
            Err(ImageVariationError::NumericalError(_))
        ));
    }

    // ---- var_channel_statistics --------------------------------------------

    #[test]
    fn test_channel_stats_single_channel_constant() {
        // All values are 3.0 → mean = 3, std = 0.
        let l = make_latent(1, 4, 4, 3.0);
        let (means, stds) = var_channel_statistics(&l).unwrap();
        assert_eq!(means.len(), 1);
        assert!((means[0] - 3.0).abs() < 1e-5);
        assert!(stds[0].abs() < 1e-5);
    }

    #[test]
    fn test_channel_stats_multi_channel() {
        // Channel 0: all 1.0, channel 1: all 2.0
        let data = {
            let mut d = vec![1.0f32; 8]; // channel 0: 2x4
            d.extend(vec![2.0f32; 8]); // channel 1: 2x4
            d
        };
        let l = VarLatentVector::from_data(2, 2, 4, data).unwrap();
        let (means, stds) = var_channel_statistics(&l).unwrap();
        assert_eq!(means.len(), 2);
        assert!((means[0] - 1.0).abs() < 1e-5);
        assert!((means[1] - 2.0).abs() < 1e-5);
        assert!(stds[0].abs() < 1e-5);
        assert!(stds[1].abs() < 1e-5);
    }

    // ---- clamp_latent ------------------------------------------------------

    #[test]
    fn test_clamp_latent_values_outside_range() {
        let data = vec![-5.0, -1.0, 0.5, 2.0, 10.0];
        let l = VarLatentVector::from_data(1, 1, 5, data).unwrap();
        let clamped = clamp_latent(&l, -1.0, 2.0).unwrap();
        assert_eq!(clamped.data, vec![-1.0, -1.0, 0.5, 2.0, 2.0]);
    }

    #[test]
    fn test_clamp_latent_preserves_shape() {
        let l = make_latent(3, 4, 4, 5.0);
        let c = clamp_latent(&l, 0.0, 3.0).unwrap();
        assert_eq!(c.shape(), l.shape());
        assert!(c.data.iter().all(|&x| (0.0..=3.0).contains(&x)));
    }

    #[test]
    fn test_clamp_latent_rejects_reversed_bounds() {
        let l = make_latent(1, 1, 4, 1.0);
        let result = clamp_latent(&l, 2.0, -2.0);
        assert!(matches!(result, Err(ImageVariationError::InvalidConfig(_))));
    }

    #[test]
    fn test_clamp_latent_rejects_nan_bound() {
        let l = make_latent(1, 1, 4, 1.0);
        let result = clamp_latent(&l, f32::NAN, 1.0);
        assert!(matches!(result, Err(ImageVariationError::InvalidConfig(_))));
    }

    #[test]
    fn test_var_channel_statistics_rejects_length_mismatch() {
        // `data` has 2 elements but the declared shape wants 12 — this must
        // report a `DimensionMismatch`, not panic while slicing per channel.
        let l = VarLatentVector {
            channels: 3,
            height: 2,
            width: 2,
            data: vec![1.0, 2.0],
        };
        let result = var_channel_statistics(&l);
        assert!(matches!(
            result,
            Err(ImageVariationError::DimensionMismatch { .. })
        ));
    }

    // ---- VariationExplorer -------------------------------------------------

    #[test]
    fn test_explorer_step_changes_current() {
        let base = make_latent(2, 4, 4, 0.0);
        let mut explorer = VariationExplorer::new(base.clone(), default_config()).unwrap();
        explorer.step().unwrap();
        let dist = latent_distance(&base, explorer.current()).unwrap();
        assert!(dist > 0.0);
    }

    #[test]
    fn test_explorer_step_count() {
        let base = VarLatentVector::new(2, 4, 4);
        let mut explorer = VariationExplorer::new(base, default_config()).unwrap();
        assert_eq!(explorer.step_count(), 0);
        explorer.step().unwrap();
        explorer.step().unwrap();
        assert_eq!(explorer.step_count(), 2);
    }

    #[test]
    fn test_explorer_revert() {
        let base = make_latent(2, 4, 4, 1.0);
        let mut explorer = VariationExplorer::new(base.clone(), default_config()).unwrap();
        let after_step = {
            explorer.step().unwrap();
            explorer.current().clone()
        };
        explorer.step().unwrap();
        explorer.revert().unwrap();
        // After one revert we should be back to `after_step`.
        assert_eq!(explorer.current().data, after_step.data);
    }

    #[test]
    fn test_explorer_revert_at_base_returns_error() {
        let base = VarLatentVector::new(2, 4, 4);
        let mut explorer = VariationExplorer::new(base, default_config()).unwrap();
        assert!(explorer.revert().is_err());
    }

    #[test]
    fn test_explorer_reset() {
        let base = make_latent(2, 4, 4, 7.0);
        let mut explorer = VariationExplorer::new(base.clone(), default_config()).unwrap();
        explorer.step().unwrap();
        explorer.step().unwrap();
        explorer.reset();
        assert_eq!(explorer.step_count(), 0);
        assert_eq!(explorer.current().data, base.data);
    }

    #[test]
    fn test_explorer_invalid_config_rejected() {
        let base = VarLatentVector::new(1, 1, 1);
        let bad_cfg = ImageVariationConfig {
            num_variations: 0,
            ..default_config()
        };
        assert!(VariationExplorer::new(base, bad_cfg).is_err());
    }

    // ---- config validation -------------------------------------------------

    #[test]
    fn test_config_valid_default() {
        assert!(default_config().validate().is_ok());
    }

    #[test]
    fn test_config_zero_variations_invalid() {
        let c = ImageVariationConfig {
            num_variations: 0,
            ..default_config()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_zero_noise_scale_invalid() {
        let c = ImageVariationConfig {
            noise_scale: 0.0,
            ..default_config()
        };
        assert!(c.validate().is_err());
    }

    #[test]
    fn test_config_negative_temperature_invalid() {
        let c = ImageVariationConfig {
            temperature: -1.0,
            ..default_config()
        };
        assert!(c.validate().is_err());
    }
}
