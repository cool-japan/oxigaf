//! Prompt embedding scheduler for diffusion model denoising trajectories.
//!
//! Manages how prompt embeddings evolve across denoising timesteps, enabling:
//! - Progressive prompt strengthening (start neutral, strengthen at later steps)
//! - Prompt interpolation between multiple reference embeddings
//! - Timestep-conditional conditioning schedules
//! - Weighted mixing of multiple prompt embeddings

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the prompt scheduler module.
#[derive(Debug, Error)]
pub enum PromptSchedulerError {
    /// Embedding dimension mismatch between two embeddings.
    #[error("Embedding dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    /// No embeddings are registered in the scheduler.
    #[error("No embeddings registered")]
    NoEmbeddings,
    /// Requested timestep is out of the valid range.
    #[error("Timestep {t} out of range [0, {max}]")]
    TimestepOutOfRange { t: usize, max: usize },
    /// Keyframe time is outside [0.0, 1.0].
    #[error("Keyframe time {time} out of range [0.0, 1.0]")]
    KeyframeOutOfRange { time: f32 },
    /// Embedding data vector is empty.
    #[error("Empty embedding vector")]
    EmptyEmbedding,
    /// A weight value is not positive.
    #[error("Weight {weight} must be positive")]
    InvalidWeight { weight: f32 },
}

// ---------------------------------------------------------------------------
// PromptEmbedding
// ---------------------------------------------------------------------------

/// A flat f32 embedding vector (CLIP image, text, or combined conditioning).
#[derive(Debug, Clone)]
pub struct PromptEmbedding {
    /// Raw embedding data.
    pub data: Vec<f32>,
    /// Human-readable label for debugging.
    pub label: String,
}

impl PromptEmbedding {
    /// Create a new embedding, verifying that `data` is non-empty.
    pub fn new(data: Vec<f32>, label: impl Into<String>) -> Result<Self, PromptSchedulerError> {
        if data.is_empty() {
            return Err(PromptSchedulerError::EmptyEmbedding);
        }
        Ok(Self {
            data,
            label: label.into(),
        })
    }

    /// Create an all-zeros embedding of the given dimension.
    ///
    /// Panics (in debug) if `dim == 0`; callers should guarantee `dim > 0`.
    pub fn zeros(dim: usize, label: impl Into<String>) -> Self {
        Self {
            data: vec![0.0_f32; dim.max(1)],
            label: label.into(),
        }
    }

    /// Dimensionality of this embedding.
    #[inline]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// L2 norm of the embedding vector.
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize the embedding to lie on the unit sphere.
    ///
    /// No-op when the norm is near zero (avoids division by zero).
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n > 1e-10 {
            let inv = 1.0 / n;
            for v in &mut self.data {
                *v *= inv;
            }
        }
    }

    /// Dot product with another embedding.
    ///
    /// Returns [`PromptSchedulerError::DimMismatch`] if dimensions differ.
    pub fn dot(&self, other: &PromptEmbedding) -> Result<f32, PromptSchedulerError> {
        if self.dim() != other.dim() {
            return Err(PromptSchedulerError::DimMismatch {
                expected: self.dim(),
                got: other.dim(),
            });
        }
        let sum = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum();
        Ok(sum)
    }

    /// Cosine similarity with another embedding.
    ///
    /// Returns `0.0` if either embedding has near-zero norm.
    /// Returns [`PromptSchedulerError::DimMismatch`] if dimensions differ.
    pub fn cosine_similarity(&self, other: &PromptEmbedding) -> Result<f32, PromptSchedulerError> {
        let d = self.dot(other)?;
        let n_self = self.norm();
        let n_other = other.norm();
        if n_self < 1e-10 || n_other < 1e-10 {
            return Ok(0.0);
        }
        Ok((d / (n_self * n_other)).clamp(-1.0, 1.0))
    }
}

// ---------------------------------------------------------------------------
// InterpolationMode and interpolation helper
// ---------------------------------------------------------------------------

/// Strategy for interpolating between two prompt embeddings.
#[derive(Debug, Clone, PartialEq)]
pub enum InterpolationMode {
    /// Linear interpolation in embedding space.
    Linear,
    /// Spherical linear interpolation (SLERP) on the unit sphere.
    Slerp,
    /// Step function: use `start` until the midpoint, then `end`.
    Step,
    /// Ease-in: slow start, fast end (cubic easing).
    EaseIn,
    /// Ease-out: fast start, slow end (cubic easing).
    EaseOut,
}

/// Interpolate between two embeddings at parameter `t ∈ [0, 1]`.
///
/// `t = 0` returns the start embedding (possibly normalized for Slerp),
/// `t = 1` returns the end embedding.
pub fn interpolate_embeddings(
    start: &PromptEmbedding,
    end: &PromptEmbedding,
    t: f32,
    mode: InterpolationMode,
) -> Result<PromptEmbedding, PromptSchedulerError> {
    if start.dim() != end.dim() {
        return Err(PromptSchedulerError::DimMismatch {
            expected: start.dim(),
            got: end.dim(),
        });
    }
    let t = t.clamp(0.0, 1.0);
    let dim = start.dim();

    let data = match mode {
        InterpolationMode::Linear => linear_interp(&start.data, &end.data, t),
        InterpolationMode::Slerp => slerp_interp(&start.data, &end.data, t),
        InterpolationMode::Step => {
            if t < 0.5 {
                start.data.clone()
            } else {
                end.data.clone()
            }
        }
        InterpolationMode::EaseIn => {
            // t³  (slow start → fast end)
            let t_eased = t * t * t;
            linear_interp(&start.data, &end.data, t_eased)
        }
        InterpolationMode::EaseOut => {
            // 1 - (1-t)³  (fast start → slow end)
            let inv = 1.0 - t;
            let t_eased = 1.0 - inv * inv * inv;
            linear_interp(&start.data, &end.data, t_eased)
        }
    };

    debug_assert_eq!(data.len(), dim, "interpolation must preserve dimension");
    Ok(PromptEmbedding {
        data,
        label: "interpolated".to_string(),
    })
}

// ---------------------------------------------------------------------------
// Internal interpolation helpers
// ---------------------------------------------------------------------------

fn linear_interp(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let one_minus_t = 1.0 - t;
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| one_minus_t * x + t * y)
        .collect()
}

/// Spherical linear interpolation between two raw vectors.
///
/// Both vectors are normalized before computing the arc. Falls back to linear
/// interpolation when the vectors are (anti-)parallel or either has zero norm.
fn slerp_interp(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    // Compute norms.
    let norm_a = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return linear_interp(a, b, t);
    }

    let inv_a = 1.0 / norm_a;
    let inv_b = 1.0 / norm_b;

    // Normalized copies.
    let a_norm: Vec<f32> = a.iter().map(|x| x * inv_a).collect();
    let b_norm: Vec<f32> = b.iter().map(|x| x * inv_b).collect();

    // Cosine of the angle between them.
    let raw_dot: f32 = a_norm.iter().zip(b_norm.iter()).map(|(x, y)| x * y).sum();
    let cos_theta = raw_dot.clamp(-1.0, 1.0);

    // Fallback to linear when (anti-)parallel.
    let eps = 1e-6;
    if cos_theta > 1.0 - eps || cos_theta < -1.0 + eps {
        return linear_interp(a, b, t);
    }

    let theta = cos_theta.acos();
    let sin_theta = theta.sin();
    let w_a = ((1.0 - t) * theta).sin() / sin_theta;
    let w_b = (t * theta).sin() / sin_theta;

    // Slerp on normalized vectors.
    a_norm
        .iter()
        .zip(b_norm.iter())
        .map(|(x, y)| w_a * x + w_b * y)
        .collect()
}

// ---------------------------------------------------------------------------
// PromptKeyframe
// ---------------------------------------------------------------------------

/// A keyframe in the denoising timeline.
#[derive(Debug, Clone)]
pub struct PromptKeyframe {
    /// Position in the timeline `[0.0, 1.0]`.
    /// 0.0 ≡ highest noise level, 1.0 ≡ cleanest (fully denoised).
    pub time: f32,
    /// Embedding to use at this keyframe.
    pub embedding: PromptEmbedding,
    /// Interpolation mode to the *next* keyframe.
    pub interpolation: InterpolationMode,
}

// ---------------------------------------------------------------------------
// PromptScheduler
// ---------------------------------------------------------------------------

/// Manages prompt embeddings across denoising timesteps via keyframes.
pub struct PromptScheduler {
    /// Keyframes sorted ascending by `time`.
    keyframes: Vec<PromptKeyframe>,
    /// Total number of denoising timesteps.
    total_timesteps: usize,
    /// Embedding dimensionality.
    dim: usize,
}

impl PromptScheduler {
    /// Create an empty scheduler for `total_timesteps` steps of `dim`-dimensional embeddings.
    pub fn new(total_timesteps: usize, dim: usize) -> Self {
        Self {
            keyframes: Vec::new(),
            total_timesteps,
            dim,
        }
    }

    /// Add a keyframe, maintaining ascending `time` order.
    ///
    /// Returns [`PromptSchedulerError::KeyframeOutOfRange`] when
    /// `keyframe.time ∉ [0.0, 1.0]`, and
    /// [`PromptSchedulerError::DimMismatch`] when the embedding dimension
    /// does not match the scheduler's configured `dim`.
    pub fn add_keyframe(&mut self, keyframe: PromptKeyframe) -> Result<(), PromptSchedulerError> {
        let t = keyframe.time;
        if !(0.0..=1.0).contains(&t) {
            return Err(PromptSchedulerError::KeyframeOutOfRange { time: t });
        }
        if keyframe.embedding.dim() != self.dim {
            return Err(PromptSchedulerError::DimMismatch {
                expected: self.dim,
                got: keyframe.embedding.dim(),
            });
        }
        // Sorted insertion by time (ascending).
        let pos = self
            .keyframes
            .partition_point(|kf| kf.time <= keyframe.time);
        self.keyframes.insert(pos, keyframe);
        Ok(())
    }

    /// Get the interpolated embedding at denoising timestep `t`.
    ///
    /// Position mapping: `position = t / (total_timesteps - 1)` so that
    /// `t = 0` → `position = 0.0` and `t = total_timesteps - 1` → `position = 1.0`.
    ///
    /// When there is only one keyframe it is returned unchanged.  When
    /// `position` falls before the first keyframe the first keyframe's
    /// embedding is returned.  When `position` falls after the last keyframe
    /// the last keyframe's embedding is returned.
    pub fn embedding_at(&self, t: usize) -> Result<PromptEmbedding, PromptSchedulerError> {
        if self.keyframes.is_empty() {
            return Err(PromptSchedulerError::NoEmbeddings);
        }
        let max = self.total_timesteps.saturating_sub(1);
        if t > max {
            return Err(PromptSchedulerError::TimestepOutOfRange { t, max });
        }

        // Compute normalized position.
        let position = if self.total_timesteps <= 1 {
            0.0_f32
        } else {
            t as f32 / (self.total_timesteps - 1) as f32
        };

        // Single keyframe — always return it.
        if self.keyframes.len() == 1 {
            return Ok(self.keyframes[0].embedding.clone());
        }

        // Before first keyframe.
        if position <= self.keyframes[0].time {
            return Ok(self.keyframes[0].embedding.clone());
        }

        // After last keyframe.
        let last = &self.keyframes[self.keyframes.len() - 1];
        if position >= last.time {
            return Ok(last.embedding.clone());
        }

        // Find the surrounding pair [i, i+1].
        let i = self
            .keyframes
            .partition_point(|kf| kf.time <= position)
            .saturating_sub(1);
        let j = i + 1;

        let kf_lo = &self.keyframes[i];
        let kf_hi = &self.keyframes[j];

        let span = kf_hi.time - kf_lo.time;
        let local_t = if span.abs() < 1e-10 {
            0.0
        } else {
            (position - kf_lo.time) / span
        };

        interpolate_embeddings(
            &kf_lo.embedding,
            &kf_hi.embedding,
            local_t,
            kf_lo.interpolation.clone(),
        )
    }

    /// Sample the interpolated embedding for every timestep in `0..total_timesteps`.
    pub fn all_embeddings(&self) -> Result<Vec<PromptEmbedding>, PromptSchedulerError> {
        (0..self.total_timesteps)
            .map(|t| self.embedding_at(t))
            .collect()
    }

    /// Number of keyframes currently registered.
    pub fn num_keyframes(&self) -> usize {
        self.keyframes.len()
    }

    /// Total number of denoising timesteps.
    pub fn total_timesteps(&self) -> usize {
        self.total_timesteps
    }

    /// Embedding dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// WeightedPrompt and mix_prompts
// ---------------------------------------------------------------------------

/// A prompt embedding together with its mixture weight.
#[derive(Debug, Clone)]
pub struct WeightedPrompt {
    /// The embedding to mix.
    pub embedding: PromptEmbedding,
    /// Positive weight for this prompt.
    pub weight: f32,
}

/// Mix multiple prompt embeddings into a single one using normalized weights.
///
/// `output = Σ (w_i / Σw_i) * e_i`
///
/// Returns [`PromptSchedulerError::NoEmbeddings`] for empty input,
/// [`PromptSchedulerError::InvalidWeight`] for any non-positive weight, and
/// [`PromptSchedulerError::DimMismatch`] for mismatched dimensions.
pub fn mix_prompts(prompts: &[WeightedPrompt]) -> Result<PromptEmbedding, PromptSchedulerError> {
    if prompts.is_empty() {
        return Err(PromptSchedulerError::NoEmbeddings);
    }

    // Validate weights and collect total weight.
    let mut total_weight = 0.0_f32;
    for p in prompts {
        if p.weight <= 0.0 {
            return Err(PromptSchedulerError::InvalidWeight { weight: p.weight });
        }
        total_weight += p.weight;
    }

    let dim = prompts[0].embedding.dim();
    // Check all dimensions match.
    for p in &prompts[1..] {
        if p.embedding.dim() != dim {
            return Err(PromptSchedulerError::DimMismatch {
                expected: dim,
                got: p.embedding.dim(),
            });
        }
    }

    let inv_total = 1.0 / total_weight;
    let mut out = vec![0.0_f32; dim];
    for p in prompts {
        let w = p.weight * inv_total;
        for (o, e) in out.iter_mut().zip(p.embedding.data.iter()) {
            *o += w * e;
        }
    }

    Ok(PromptEmbedding {
        data: out,
        label: "mixed".to_string(),
    })
}

// ---------------------------------------------------------------------------
// Augmentation helpers
// ---------------------------------------------------------------------------

/// xorshift64 pseudo-random number generator (minimal state advance).
fn xorshift64(s: &mut u64) -> u64 {
    let mut x = *s;
    if x == 0 {
        x = 1;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    x
}

/// Sample a uniform float in [0, 1) from the xorshift64 state.
fn xorshift_f32(s: &mut u64) -> f32 {
    (xorshift64(s) >> 11) as f32 / (1u64 << 53) as f32
}

/// Augment an embedding by adding Gaussian noise scaled by `strength`.
///
/// Uses xorshift64 PRNG seeded with `seed` and Box-Muller transform for
/// Gaussian samples.  The result has the same dimension as the input.
pub fn augment_embedding(embedding: &PromptEmbedding, strength: f32, seed: u64) -> PromptEmbedding {
    let mut state = seed;
    let mut out = embedding.data.clone();
    let n = out.len();
    let mut i = 0;
    while i < n {
        let u1 = xorshift_f32(&mut state);
        let u2 = xorshift_f32(&mut state);
        // Box-Muller: ensure u1 is not (near) zero.
        let u1_safe = u1.max(1e-10_f32);
        let mag = (-2.0 * u1_safe.ln()).sqrt();
        let angle = 2.0 * std::f32::consts::PI * u2;
        let z0 = mag * angle.cos();
        let z1 = mag * angle.sin();

        out[i] += z0 * strength;
        if i + 1 < n {
            out[i + 1] += z1 * strength;
        }
        i += 2;
    }
    PromptEmbedding {
        data: out,
        label: format!("{}_augmented", embedding.label),
    }
}

/// Mean pairwise cosine distance between a collection of embeddings.
///
/// Returns `0.0` for 0 or 1 embeddings (no pairs).
/// Returns [`PromptSchedulerError::NoEmbeddings`] if the slice is empty.
pub fn embedding_diversity(embeddings: &[PromptEmbedding]) -> Result<f32, PromptSchedulerError> {
    if embeddings.is_empty() {
        return Err(PromptSchedulerError::NoEmbeddings);
    }
    if embeddings.len() == 1 {
        return Ok(0.0);
    }
    let n = embeddings.len();
    let num_pairs = n * (n - 1) / 2;
    let mut total_distance = 0.0_f32;
    for i in 0..n {
        for j in (i + 1)..n {
            let cos_sim = embeddings[i]
                .cosine_similarity(&embeddings[j])
                .unwrap_or(0.0);
            // Cosine distance = 1 - cosine_similarity.
            total_distance += 1.0 - cos_sim;
        }
    }
    Ok(total_distance / num_pairs as f32)
}

// ---------------------------------------------------------------------------
// Scheduling patterns
// ---------------------------------------------------------------------------

/// Build a schedule that starts with `neutral` and gradually strengthens to `target`.
///
/// Creates two keyframes:
/// - `time = start_fraction` holding `neutral` (Linear interpolation to next)
/// - `time = 1.0` holding `target`
///
/// For positions before `start_fraction` the scheduler returns `neutral`.
pub fn make_strengthening_schedule(
    neutral: PromptEmbedding,
    target: PromptEmbedding,
    total_timesteps: usize,
    start_fraction: f32,
) -> Result<PromptScheduler, PromptSchedulerError> {
    if !(0.0..=1.0).contains(&start_fraction) {
        return Err(PromptSchedulerError::KeyframeOutOfRange {
            time: start_fraction,
        });
    }
    if neutral.dim() != target.dim() {
        return Err(PromptSchedulerError::DimMismatch {
            expected: neutral.dim(),
            got: target.dim(),
        });
    }
    let dim = neutral.dim();
    let mut scheduler = PromptScheduler::new(total_timesteps, dim);
    scheduler.add_keyframe(PromptKeyframe {
        time: start_fraction.clamp(0.0, 1.0),
        embedding: neutral,
        interpolation: InterpolationMode::Linear,
    })?;
    scheduler.add_keyframe(PromptKeyframe {
        time: 1.0,
        embedding: target,
        interpolation: InterpolationMode::Linear,
    })?;
    Ok(scheduler)
}

/// Build a cyclic schedule that oscillates between two embeddings.
///
/// Creates `2 * num_cycles + 1` keyframes alternating between `embed_a` and
/// `embed_b` at evenly spaced times `0, 1/(2n), 2/(2n), …, 1`.
pub fn make_cyclic_schedule(
    embed_a: PromptEmbedding,
    embed_b: PromptEmbedding,
    total_timesteps: usize,
    num_cycles: usize,
) -> Result<PromptScheduler, PromptSchedulerError> {
    if embed_a.dim() != embed_b.dim() {
        return Err(PromptSchedulerError::DimMismatch {
            expected: embed_a.dim(),
            got: embed_b.dim(),
        });
    }
    let dim = embed_a.dim();
    let mut scheduler = PromptScheduler::new(total_timesteps, dim);

    let total_keyframes = 2 * num_cycles + 1;
    let num_half = 2 * num_cycles;

    for k in 0..total_keyframes {
        let time = if num_half == 0 {
            0.0_f32
        } else {
            k as f32 / num_half as f32
        };
        let embedding = if k % 2 == 0 {
            embed_a.clone()
        } else {
            embed_b.clone()
        };
        scheduler.add_keyframe(PromptKeyframe {
            time: time.clamp(0.0, 1.0),
            embedding,
            interpolation: InterpolationMode::Linear,
        })?;
    }
    Ok(scheduler)
}

// ---------------------------------------------------------------------------
// ScheduleSummary and summarize_schedule
// ---------------------------------------------------------------------------

/// Summary statistics for a [`PromptScheduler`].
#[derive(Debug, Clone)]
pub struct ScheduleSummary {
    /// Total denoising timesteps.
    pub total_timesteps: usize,
    /// Number of keyframes.
    pub num_keyframes: usize,
    /// Embedding dimensionality.
    pub embedding_dim: usize,
    /// Mean cosine distance between consecutive timestep embeddings.
    pub mean_step_distance: f32,
    /// Maximum cosine distance between consecutive timestep embeddings.
    pub max_step_distance: f32,
}

/// Compute a [`ScheduleSummary`] by sampling all embeddings and measuring
/// consecutive cosine distances.
pub fn summarize_schedule(
    scheduler: &PromptScheduler,
) -> Result<ScheduleSummary, PromptSchedulerError> {
    let all = scheduler.all_embeddings()?;
    let total = scheduler.total_timesteps();

    if all.len() < 2 {
        return Ok(ScheduleSummary {
            total_timesteps: total,
            num_keyframes: scheduler.num_keyframes(),
            embedding_dim: scheduler.dim(),
            mean_step_distance: 0.0,
            max_step_distance: 0.0,
        });
    }

    let mut sum_dist = 0.0_f32;
    let mut max_dist = 0.0_f32;
    let n_pairs = all.len() - 1;

    for pair in all.windows(2) {
        let cos_sim = pair[0].cosine_similarity(&pair[1]).unwrap_or(0.0);
        let dist = 1.0 - cos_sim;
        sum_dist += dist;
        if dist > max_dist {
            max_dist = dist;
        }
    }

    Ok(ScheduleSummary {
        total_timesteps: total,
        num_keyframes: scheduler.num_keyframes(),
        embedding_dim: scheduler.dim(),
        mean_step_distance: sum_dist / n_pairs as f32,
        max_step_distance: max_dist,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn embed(data: Vec<f32>) -> PromptEmbedding {
        PromptEmbedding::new(data, "test").expect("valid embedding")
    }

    #[allow(dead_code)]
    fn zero_embed(dim: usize) -> PromptEmbedding {
        PromptEmbedding::zeros(dim, "zero")
    }

    // Test 1: new() with empty data → EmptyEmbedding error
    #[test]
    fn test_embedding_new_empty_error() {
        let result = PromptEmbedding::new(vec![], "test");
        assert!(matches!(result, Err(PromptSchedulerError::EmptyEmbedding)));
    }

    // Test 2: norm for [3, 4] → 5.0
    #[test]
    fn test_embedding_norm() {
        let e = embed(vec![3.0, 4.0]);
        let n = e.norm();
        assert!((n - 5.0).abs() < 1e-6, "expected 5.0, got {}", n);
    }

    // Test 3: normalize leaves unit vector unchanged
    #[test]
    fn test_embedding_normalize_unit_unchanged() {
        let mut e = embed(vec![1.0, 0.0]);
        e.normalize();
        assert!((e.data[0] - 1.0).abs() < 1e-6);
        assert!((e.data[1] - 0.0).abs() < 1e-6);
    }

    // Test 4: cosine_similarity of orthogonal vectors → 0.0
    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = embed(vec![1.0, 0.0]);
        let b = embed(vec![0.0, 1.0]);
        let sim = a.cosine_similarity(&b).expect("valid");
        assert!((sim - 0.0).abs() < 1e-6, "expected 0.0, got {}", sim);
    }

    // Test 5: cosine_similarity of parallel vectors → 1.0
    #[test]
    fn test_cosine_similarity_parallel() {
        let a = embed(vec![1.0, 2.0, 3.0]);
        let b = embed(vec![2.0, 4.0, 6.0]);
        let sim = a.cosine_similarity(&b).expect("valid");
        assert!((sim - 1.0).abs() < 1e-6, "expected 1.0, got {}", sim);
    }

    // Test 6: dot product with length mismatch → DimMismatch
    #[test]
    fn test_dot_dim_mismatch() {
        let a = embed(vec![1.0, 2.0]);
        let b = embed(vec![1.0, 2.0, 3.0]);
        let result = a.dot(&b);
        assert!(matches!(
            result,
            Err(PromptSchedulerError::DimMismatch { .. })
        ));
    }

    // Test 7: interpolate Linear t=0 → start
    #[test]
    fn test_interpolate_linear_t0() {
        let start = embed(vec![1.0, 2.0, 3.0]);
        let end = embed(vec![4.0, 5.0, 6.0]);
        let result =
            interpolate_embeddings(&start, &end, 0.0, InterpolationMode::Linear).expect("valid");
        for (r, s) in result.data.iter().zip(start.data.iter()) {
            assert!((r - s).abs() < 1e-6, "expected start at t=0");
        }
    }

    // Test 8: interpolate Linear t=1 → end
    #[test]
    fn test_interpolate_linear_t1() {
        let start = embed(vec![1.0, 2.0, 3.0]);
        let end = embed(vec![4.0, 5.0, 6.0]);
        let result =
            interpolate_embeddings(&start, &end, 1.0, InterpolationMode::Linear).expect("valid");
        for (r, e) in result.data.iter().zip(end.data.iter()) {
            assert!((r - e).abs() < 1e-6, "expected end at t=1");
        }
    }

    // Test 9: interpolate Linear t=0.5 → midpoint
    #[test]
    fn test_interpolate_linear_midpoint() {
        let start = embed(vec![0.0, 0.0]);
        let end = embed(vec![2.0, 4.0]);
        let result =
            interpolate_embeddings(&start, &end, 0.5, InterpolationMode::Linear).expect("valid");
        assert!((result.data[0] - 1.0).abs() < 1e-6);
        assert!((result.data[1] - 2.0).abs() < 1e-6);
    }

    // Test 10: interpolate Slerp t=0 → start (normalized)
    #[test]
    fn test_interpolate_slerp_t0() {
        let mut start = embed(vec![1.0, 0.0]);
        start.normalize();
        let mut end = embed(vec![0.0, 1.0]);
        end.normalize();
        let result =
            interpolate_embeddings(&start, &end, 0.0, InterpolationMode::Slerp).expect("valid");
        // At t=0, should equal the normalized start.
        assert!((result.data[0] - start.data[0]).abs() < 1e-5);
        assert!((result.data[1] - start.data[1]).abs() < 1e-5);
    }

    // Test 11: interpolate Step t<0.5 → start, t>=0.5 → end
    #[test]
    fn test_interpolate_step() {
        let start = embed(vec![1.0, 0.0]);
        let end = embed(vec![0.0, 1.0]);

        let lo = interpolate_embeddings(&start, &end, 0.3, InterpolationMode::Step).expect("valid");
        assert!((lo.data[0] - 1.0).abs() < 1e-6);
        assert!((lo.data[1] - 0.0).abs() < 1e-6);

        let hi = interpolate_embeddings(&start, &end, 0.5, InterpolationMode::Step).expect("valid");
        assert!((hi.data[0] - 0.0).abs() < 1e-6);
        assert!((hi.data[1] - 1.0).abs() < 1e-6);
    }

    // Test 12: PromptScheduler::new starts empty
    #[test]
    fn test_scheduler_new_empty() {
        let sched = PromptScheduler::new(100, 4);
        assert_eq!(sched.num_keyframes(), 0);
        assert_eq!(sched.total_timesteps(), 100);
        assert_eq!(sched.dim(), 4);
    }

    // Test 13: add_keyframe with out-of-range time → KeyframeOutOfRange
    #[test]
    fn test_add_keyframe_out_of_range() {
        let mut sched = PromptScheduler::new(10, 3);
        let kf = PromptKeyframe {
            time: 1.5,
            embedding: embed(vec![1.0, 0.0, 0.0]),
            interpolation: InterpolationMode::Linear,
        };
        let result = sched.add_keyframe(kf);
        assert!(matches!(
            result,
            Err(PromptSchedulerError::KeyframeOutOfRange { .. })
        ));
    }

    // Test 14: embedding_at with single keyframe → always that embedding
    #[test]
    fn test_embedding_at_single_keyframe() {
        let mut sched = PromptScheduler::new(10, 3);
        let data = vec![1.0, 2.0, 3.0];
        sched
            .add_keyframe(PromptKeyframe {
                time: 0.5,
                embedding: embed(data.clone()),
                interpolation: InterpolationMode::Linear,
            })
            .expect("valid");

        for t in 0..10 {
            let e = sched.embedding_at(t).expect("valid");
            for (got, expected) in e.data.iter().zip(data.iter()) {
                assert!((got - expected).abs() < 1e-6);
            }
        }
    }

    // Test 15: two keyframes: t=0 → first, t=T-1 → second
    #[test]
    fn test_embedding_at_two_keyframes_endpoints() {
        let total = 10_usize;
        let mut sched = PromptScheduler::new(total, 2);
        let a = embed(vec![1.0, 0.0]);
        let b = embed(vec![0.0, 1.0]);

        sched
            .add_keyframe(PromptKeyframe {
                time: 0.0,
                embedding: a.clone(),
                interpolation: InterpolationMode::Linear,
            })
            .expect("valid");
        sched
            .add_keyframe(PromptKeyframe {
                time: 1.0,
                embedding: b.clone(),
                interpolation: InterpolationMode::Linear,
            })
            .expect("valid");

        let e0 = sched.embedding_at(0).expect("valid");
        assert!(
            (e0.data[0] - 1.0).abs() < 1e-6,
            "t=0 should give first keyframe"
        );
        assert!((e0.data[1] - 0.0).abs() < 1e-6);

        let e_last = sched.embedding_at(total - 1).expect("valid");
        assert!(
            (e_last.data[0] - 0.0).abs() < 1e-6,
            "t=T-1 should give second keyframe"
        );
        assert!((e_last.data[1] - 1.0).abs() < 1e-6);
    }

    // Test 16: embedding_at out-of-range t → TimestepOutOfRange
    #[test]
    fn test_embedding_at_out_of_range() {
        let mut sched = PromptScheduler::new(10, 2);
        sched
            .add_keyframe(PromptKeyframe {
                time: 0.0,
                embedding: embed(vec![1.0, 0.0]),
                interpolation: InterpolationMode::Linear,
            })
            .expect("valid");
        let result = sched.embedding_at(10);
        assert!(matches!(
            result,
            Err(PromptSchedulerError::TimestepOutOfRange { .. })
        ));
    }

    // Test 17: all_embeddings returns T embeddings
    #[test]
    fn test_all_embeddings_count() {
        let total = 20_usize;
        let mut sched = PromptScheduler::new(total, 2);
        sched
            .add_keyframe(PromptKeyframe {
                time: 0.0,
                embedding: embed(vec![1.0, 0.0]),
                interpolation: InterpolationMode::Linear,
            })
            .expect("valid");
        sched
            .add_keyframe(PromptKeyframe {
                time: 1.0,
                embedding: embed(vec![0.0, 1.0]),
                interpolation: InterpolationMode::Linear,
            })
            .expect("valid");

        let all = sched.all_embeddings().expect("valid");
        assert_eq!(all.len(), total);
    }

    // Test 18: mix_prompts single prompt → same embedding
    #[test]
    fn test_mix_prompts_single() {
        let e = embed(vec![1.0, 2.0, 3.0]);
        let prompts = vec![WeightedPrompt {
            embedding: e.clone(),
            weight: 2.0,
        }];
        let result = mix_prompts(&prompts).expect("valid");
        for (r, s) in result.data.iter().zip(e.data.iter()) {
            assert!((r - s).abs() < 1e-6);
        }
    }

    // Test 19: mix_prompts two equal-weight prompts → average
    #[test]
    fn test_mix_prompts_two_equal_weights() {
        let a = embed(vec![0.0, 2.0]);
        let b = embed(vec![4.0, 0.0]);
        let prompts = vec![
            WeightedPrompt {
                embedding: a,
                weight: 1.0,
            },
            WeightedPrompt {
                embedding: b,
                weight: 1.0,
            },
        ];
        let result = mix_prompts(&prompts).expect("valid");
        assert!((result.data[0] - 2.0).abs() < 1e-6);
        assert!((result.data[1] - 1.0).abs() < 1e-6);
    }

    // Test 20: mix_prompts empty → NoEmbeddings
    #[test]
    fn test_mix_prompts_empty() {
        let result = mix_prompts(&[]);
        assert!(matches!(result, Err(PromptSchedulerError::NoEmbeddings)));
    }

    // Test 21: augment_embedding output same dimension
    #[test]
    fn test_augment_embedding_same_dim() {
        let e = embed(vec![1.0, 0.5, -0.3, 0.8]);
        let aug = augment_embedding(&e, 0.1, 42);
        assert_eq!(aug.dim(), e.dim());
    }

    // Test 22: embedding_diversity identical embeddings → 0.0
    #[test]
    fn test_embedding_diversity_identical() {
        let e = embed(vec![1.0, 0.0]);
        let embeddings = vec![e.clone(), e.clone(), e.clone()];
        let d = embedding_diversity(&embeddings).expect("valid");
        assert!(
            d.abs() < 1e-5,
            "identical embeddings should have 0 diversity, got {}",
            d
        );
    }

    // Test 23: make_strengthening_schedule has exactly 2 keyframes
    #[test]
    fn test_make_strengthening_schedule_two_keyframes() {
        let neutral = embed(vec![1.0, 0.0, 0.0]);
        let target = embed(vec![0.0, 1.0, 0.0]);
        let sched = make_strengthening_schedule(neutral, target, 50, 0.3).expect("valid schedule");
        assert_eq!(sched.num_keyframes(), 2);
    }

    // Test 24: summarize_schedule returns a valid summary
    #[test]
    fn test_summarize_schedule_valid() {
        let total = 20_usize;
        let mut sched = PromptScheduler::new(total, 2);
        sched
            .add_keyframe(PromptKeyframe {
                time: 0.0,
                embedding: embed(vec![1.0, 0.0]),
                interpolation: InterpolationMode::Linear,
            })
            .expect("valid");
        sched
            .add_keyframe(PromptKeyframe {
                time: 1.0,
                embedding: embed(vec![0.0, 1.0]),
                interpolation: InterpolationMode::Linear,
            })
            .expect("valid");

        let summary = summarize_schedule(&sched).expect("valid");
        assert_eq!(summary.total_timesteps, total);
        assert_eq!(summary.num_keyframes, 2);
        assert_eq!(summary.embedding_dim, 2);
        assert!(summary.mean_step_distance >= 0.0);
        assert!(summary.max_step_distance >= summary.mean_step_distance);
    }
}
