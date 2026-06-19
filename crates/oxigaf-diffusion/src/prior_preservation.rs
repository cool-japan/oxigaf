//! # Prior Preservation
//!
//! DreamBooth-style class-specific prior preservation loss for fine-tuning
//! generative models without catastrophic forgetting.
//!
//! ## Overview
//!
//! When fine-tuning a diffusion model on a small set of subject images, the model
//! can overfit and "forget" the general concept it was trained on (catastrophic
//! forgetting). Prior preservation counters this by simultaneously training on
//! class-representative latents drawn from a precomputed buffer, penalising
//! deviation from the original model's output distribution.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use oxigaf_diffusion::prior_preservation::{
//!     PriorPreservationConfig, PriorPreservationTracker,
//! };
//!
//! let config = PriorPreservationConfig::default();
//! let mut tracker = PriorPreservationTracker::new(config);
//!
//! // Push precomputed class latents into the buffer.
//! let latent = vec![0.0f32; 512];
//! tracker.add_class_latent(latent).unwrap();
//!
//! // During training, compute the prior loss.
//! let pred = vec![0.1f32; 512];
//! let mut rng = 42u64;
//! let loss = tracker.compute_loss(&pred, &mut rng).unwrap();
//! println!("prior loss: {loss}");
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// PRNG helpers (no `rand` crate allowed)
// ---------------------------------------------------------------------------

/// Xorshift64 PRNG — updates state in-place and returns a new pseudo-random u64.
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

/// Maps an xorshift64 output to a uniform `f32` in `[0, 1)`.
#[inline]
pub fn xorshift_f32(state: &mut u64) -> f32 {
    xorshift64(state) as f32 / u64::MAX as f32
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during prior preservation operations.
#[derive(Debug, Error)]
pub enum PriorError {
    /// The buffer or input slice is empty when a non-empty one is required.
    #[error("buffer is empty")]
    EmptyBuffer,

    /// The expected and actual dimensions do not match.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// A configuration parameter is invalid.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// A sample index is out of bounds for the current buffer.
    #[error("sample index out of bounds: {0}")]
    IndexOutOfBounds(usize),
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for prior preservation loss.
#[derive(Debug, Clone)]
pub struct PriorPreservationConfig {
    /// Scaling factor λ_prior applied to the loss.  Default `1.0`.
    pub weight: f32,
    /// Maximum number of class latents stored in the circular buffer.  Default `512`.
    pub buffer_capacity: usize,
    /// Number of training steps before the prior loss is enabled.  Default `1000`.
    pub warmup_steps: usize,
    /// Dimensionality of each latent vector stored in the buffer.
    pub latent_dim: usize,
    /// Temperature used to scale logits before softmax in the soft-KL loss.  Default `1.0`.
    pub temperature: f32,
    /// Exponential moving average decay for running loss statistics.  Default `0.999`.
    pub ema_decay: f32,
}

impl Default for PriorPreservationConfig {
    fn default() -> Self {
        Self {
            weight: 1.0,
            buffer_capacity: 512,
            warmup_steps: 1000,
            latent_dim: 512,
            temperature: 1.0,
            ema_decay: 0.999,
        }
    }
}

impl PriorPreservationConfig {
    /// Validate the configuration, returning an error if any field is out of range.
    pub fn validate(&self) -> Result<(), PriorError> {
        if self.weight < 0.0 {
            return Err(PriorError::InvalidConfig(
                "weight must be non-negative".into(),
            ));
        }
        if self.buffer_capacity == 0 {
            return Err(PriorError::InvalidConfig(
                "buffer_capacity must be > 0".into(),
            ));
        }
        if self.latent_dim == 0 {
            return Err(PriorError::InvalidConfig("latent_dim must be > 0".into()));
        }
        if self.temperature <= 0.0 {
            return Err(PriorError::InvalidConfig("temperature must be > 0".into()));
        }
        if !(0.0..1.0).contains(&self.ema_decay) {
            return Err(PriorError::InvalidConfig(
                "ema_decay must be in [0, 1)".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ClassPriorBuffer — circular buffer of class latents
// ---------------------------------------------------------------------------

/// Circular buffer that stores precomputed class latents for prior preservation.
///
/// Each entry is a flat `Vec<f32>` of length `latent_dim`.  When the buffer is
/// full, new entries overwrite the oldest ones (FIFO eviction).
#[derive(Debug, Clone)]
pub struct ClassPriorBuffer {
    capacity: usize,
    latent_dim: usize,
    entries: Vec<Vec<f32>>,
    write_pos: usize,
    total_added: usize,
}

impl ClassPriorBuffer {
    /// Create a new, empty buffer.
    ///
    /// # Panics
    ///
    /// Will not panic; errors are surfaced through the `push` / `sample` API.
    pub fn new(capacity: usize, latent_dim: usize) -> Self {
        Self {
            capacity,
            latent_dim,
            entries: Vec::with_capacity(capacity),
            write_pos: 0,
            total_added: 0,
        }
    }

    /// Push a latent vector into the buffer.
    ///
    /// If the buffer is already at capacity the oldest entry is replaced.
    ///
    /// # Errors
    ///
    /// Returns [`PriorError::DimensionMismatch`] when `latent.len() != latent_dim`.
    pub fn push(&mut self, latent: Vec<f32>) -> Result<(), PriorError> {
        if latent.len() != self.latent_dim {
            return Err(PriorError::DimensionMismatch {
                expected: self.latent_dim,
                got: latent.len(),
            });
        }
        if self.entries.len() < self.capacity {
            self.entries.push(latent);
        } else {
            self.entries[self.write_pos] = latent;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.total_added += 1;
        Ok(())
    }

    /// Sample `n` random latent vectors from the buffer (with replacement).
    ///
    /// # Errors
    ///
    /// - [`PriorError::EmptyBuffer`] if the buffer contains no entries.
    /// - [`PriorError::InvalidConfig`] if `n == 0`.
    pub fn sample(&self, n: usize, rng_state: &mut u64) -> Result<Vec<Vec<f32>>, PriorError> {
        if self.entries.is_empty() {
            return Err(PriorError::EmptyBuffer);
        }
        if n == 0 {
            return Err(PriorError::InvalidConfig("n must be > 0".into()));
        }
        let len = self.entries.len();
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = (xorshift64(rng_state) as usize) % len;
            out.push(self.entries[idx].clone());
        }
        Ok(out)
    }

    /// Number of latents currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the buffer contains no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether the buffer has reached capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    /// Maximum number of entries the buffer can hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Dimensionality of each stored latent vector.
    #[inline]
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Total number of latents that have been added (including overwritten ones).
    #[inline]
    pub fn total_added(&self) -> usize {
        self.total_added
    }
}

// ---------------------------------------------------------------------------
// PriorStats
// ---------------------------------------------------------------------------

/// Running statistics for the prior preservation loss.
#[derive(Debug, Clone)]
pub struct PriorStats {
    /// Arithmetic mean of losses seen so far.
    pub mean_loss: f32,
    /// Minimum loss seen so far.
    pub min_loss: f32,
    /// Maximum loss seen so far.
    pub max_loss: f32,
    /// Total number of loss samples accumulated.
    pub total_samples: usize,
    /// Fraction of the buffer that is currently filled (`len / capacity`).
    pub buffer_fill_ratio: f32,
    /// Exponential moving average of the loss.
    pub ema_loss: f32,
}

impl Default for PriorStats {
    fn default() -> Self {
        Self {
            mean_loss: 0.0,
            min_loss: f32::INFINITY,
            max_loss: f32::NEG_INFINITY,
            total_samples: 0,
            buffer_fill_ratio: 0.0,
            ema_loss: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Core free functions
// ---------------------------------------------------------------------------

/// MSE-based prior preservation loss between model predictions and class latents.
///
/// `loss = effective_weight * mean((pred - target)^2)`
///
/// If `step < warmup_steps` the effective weight is 0 and the function returns
/// `0.0` immediately (no computation needed).
///
/// # Errors
///
/// - [`PriorError::DimensionMismatch`] when `pred.len() != target.len()`.
/// - [`PriorError::EmptyBuffer`] when both slices are empty.
pub fn prior_preservation_loss(
    pred: &[f32],
    target: &[f32],
    config: &PriorPreservationConfig,
    step: usize,
) -> Result<f32, PriorError> {
    let eff_weight = prior_effective_weight(config, step);
    if eff_weight == 0.0 {
        return Ok(0.0);
    }
    if pred.is_empty() {
        return Err(PriorError::EmptyBuffer);
    }
    if pred.len() != target.len() {
        return Err(PriorError::DimensionMismatch {
            expected: pred.len(),
            got: target.len(),
        });
    }
    let mse: f32 = pred
        .iter()
        .zip(target.iter())
        .map(|(p, t)| (p - t) * (p - t))
        .sum::<f32>()
        / pred.len() as f32;
    Ok(eff_weight * mse)
}

/// Soft prior loss via temperature-scaled KL divergence.
///
/// Both `pred` and `target` are treated as unnormalized log-probabilities
/// (logits).  They are each converted to a probability distribution via
/// softmax with temperature scaling before computing KL(target_probs || pred_probs).
///
/// # Errors
///
/// - [`PriorError::DimensionMismatch`] when `pred.len() != target.len()`.
/// - [`PriorError::EmptyBuffer`] when both slices are empty.
pub fn prior_soft_loss(
    pred: &[f32],
    target: &[f32],
    config: &PriorPreservationConfig,
) -> Result<f32, PriorError> {
    if pred.is_empty() {
        return Err(PriorError::EmptyBuffer);
    }
    if pred.len() != target.len() {
        return Err(PriorError::DimensionMismatch {
            expected: pred.len(),
            got: target.len(),
        });
    }
    let t = config.temperature;
    // Scale logits by 1/temperature then softmax.
    let scaled_pred: Vec<f32> = pred.iter().map(|x| x / t).collect();
    let scaled_target: Vec<f32> = target.iter().map(|x| x / t).collect();
    let p = prior_softmax(&scaled_target); // "teacher" distribution
    let q = prior_softmax(&scaled_pred); // "student" distribution
    prior_kl_divergence(&p, &q)
}

/// Combined hard (MSE) + soft (KL) prior loss.
///
/// `loss = alpha * hard_mse + (1 - alpha) * soft_kl`
///
/// `alpha = 1.0` gives pure MSE; `alpha = 0.0` gives pure soft KL.
///
/// # Errors
///
/// Propagates errors from [`prior_preservation_loss`] and [`prior_soft_loss`].
pub fn prior_combined_loss(
    pred: &[f32],
    target: &[f32],
    alpha: f32,
    config: &PriorPreservationConfig,
    step: usize,
) -> Result<f32, PriorError> {
    let hard = prior_preservation_loss(pred, target, config, step)?;
    let soft = prior_soft_loss(pred, target, config)?;
    Ok(alpha * hard + (1.0 - alpha) * soft)
}

/// Compute per-sample MSE losses for a batch.
///
/// `pred` and `target` are flat arrays of shape `[batch_size * latent_dim]`.
/// Returns a `Vec<f32>` of length `batch_size`.
///
/// # Errors
///
/// - [`PriorError::EmptyBuffer`] when either slice is empty.
/// - [`PriorError::DimensionMismatch`] when lengths differ or do not divide
///   evenly by `latent_dim`.
pub fn prior_per_sample_losses(
    pred: &[f32],
    target: &[f32],
    latent_dim: usize,
) -> Result<Vec<f32>, PriorError> {
    if pred.is_empty() || target.is_empty() {
        return Err(PriorError::EmptyBuffer);
    }
    if pred.len() != target.len() {
        return Err(PriorError::DimensionMismatch {
            expected: pred.len(),
            got: target.len(),
        });
    }
    if latent_dim == 0 || !pred.len().is_multiple_of(latent_dim) {
        return Err(PriorError::DimensionMismatch {
            expected: 0, // sentinel — latent_dim doesn't divide evenly
            got: latent_dim,
        });
    }
    let batch = pred.len() / latent_dim;
    let mut losses = Vec::with_capacity(batch);
    for b in 0..batch {
        let start = b * latent_dim;
        let end = start + latent_dim;
        let mse = pred[start..end]
            .iter()
            .zip(target[start..end].iter())
            .map(|(p, t)| (p - t) * (p - t))
            .sum::<f32>()
            / latent_dim as f32;
        losses.push(mse);
    }
    Ok(losses)
}

/// Update running statistics in-place.
///
/// Uses an online (Welford-style) mean update and an EMA on the loss.
pub fn prior_update_stats(
    stats: &mut PriorStats,
    new_loss: f32,
    buffer: &ClassPriorBuffer,
    ema_decay: f32,
) {
    stats.total_samples += 1;
    // Online mean update: mean_n = mean_{n-1} + (x - mean_{n-1}) / n
    let delta = new_loss - stats.mean_loss;
    stats.mean_loss += delta / stats.total_samples as f32;
    if new_loss < stats.min_loss {
        stats.min_loss = new_loss;
    }
    if new_loss > stats.max_loss {
        stats.max_loss = new_loss;
    }
    let cap = buffer.capacity();
    stats.buffer_fill_ratio = if cap == 0 {
        0.0
    } else {
        buffer.len() as f32 / cap as f32
    };
    // EMA: ema = decay * ema + (1 - decay) * new
    stats.ema_loss = ema_decay * stats.ema_loss + (1.0 - ema_decay) * new_loss;
}

/// Compute the effective prior weight at `step` using a linear warmup schedule.
///
/// - For `step < warmup_steps`: returns `0.0`.
/// - For `step in [warmup_steps, 2 * warmup_steps]`: linearly ramps from `0` to
///   `config.weight`.
/// - For `step >= 2 * warmup_steps`: returns `config.weight`.
///
/// If `warmup_steps == 0` the weight is always `config.weight`.
pub fn prior_effective_weight(config: &PriorPreservationConfig, step: usize) -> f32 {
    let w = config.warmup_steps;
    if w == 0 {
        return config.weight;
    }
    if step < w {
        return 0.0;
    }
    if step >= 2 * w {
        return config.weight;
    }
    // step in [w, 2w) → fraction = (step - w) / w
    let fraction = (step - w) as f32 / w as f32;
    config.weight * fraction
}

/// Numerically stable softmax over a slice of logits.
///
/// Subtracts the maximum value before exponentiating to prevent overflow.
/// Returns a probability distribution summing to `1.0`.
///
/// If the input is empty returns an empty `Vec`.
pub fn prior_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        // All logits are -infinity; return uniform distribution.
        let n = logits.len() as f32;
        return vec![1.0 / n; logits.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

/// KL divergence KL(p ‖ q) = Σ p_i · ln(p_i / q_i).
///
/// Both `p` and `q` must already be probability distributions (non-negative,
/// summing to ~1).  Returns `f32::INFINITY` if `q_i == 0` for any `i` where
/// `p_i > 0`.
///
/// # Errors
///
/// - [`PriorError::DimensionMismatch`] when lengths differ.
/// - [`PriorError::EmptyBuffer`] when slices are empty.
pub fn prior_kl_divergence(p: &[f32], q: &[f32]) -> Result<f32, PriorError> {
    if p.is_empty() {
        return Err(PriorError::EmptyBuffer);
    }
    if p.len() != q.len() {
        return Err(PriorError::DimensionMismatch {
            expected: p.len(),
            got: q.len(),
        });
    }
    let mut kl = 0.0f32;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi == 0.0 {
            continue; // 0 · log(0/q) = 0 by convention
        }
        if qi <= 0.0 {
            return Ok(f32::INFINITY);
        }
        kl += pi * (pi / qi).ln();
    }
    Ok(kl)
}

/// Cosine similarity between two equal-length vectors.
///
/// Returns a value in `[-1, 1]`.  Returns `0.0` when either vector is
/// all-zeros (degenerate case).
///
/// # Errors
///
/// - [`PriorError::DimensionMismatch`] when lengths differ.
/// - [`PriorError::EmptyBuffer`] when slices are empty.
pub fn prior_cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, PriorError> {
    if a.is_empty() {
        return Err(PriorError::EmptyBuffer);
    }
    if a.len() != b.len() {
        return Err(PriorError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }
    Ok((dot / (norm_a * norm_b)).clamp(-1.0, 1.0))
}

/// Select `n` diverse samples from the buffer using greedy farthest-point
/// sampling in cosine-distance space.
///
/// The first sample is chosen uniformly at random; each subsequent sample
/// maximises the minimum cosine distance to all already-selected samples.
///
/// # Errors
///
/// - [`PriorError::EmptyBuffer`] if the buffer is empty.
/// - [`PriorError::InvalidConfig`] if `n == 0`.
/// - [`PriorError::IndexOutOfBounds`] if `n > buffer.len()`.
pub fn prior_diverse_sample(
    buffer: &ClassPriorBuffer,
    n: usize,
    rng_state: &mut u64,
) -> Result<Vec<Vec<f32>>, PriorError> {
    if buffer.is_empty() {
        return Err(PriorError::EmptyBuffer);
    }
    if n == 0 {
        return Err(PriorError::InvalidConfig("n must be > 0".into()));
    }
    if n > buffer.len() {
        return Err(PriorError::IndexOutOfBounds(n));
    }
    let entries = &buffer.entries;
    let m = entries.len();

    // For each candidate, track its minimum cosine distance to the selected set.
    // Cosine distance = 1 - cosine_similarity.
    let mut min_dist = vec![f32::INFINITY; m];
    let mut selected_indices: Vec<usize> = Vec::with_capacity(n);
    let mut selected_vecs: Vec<Vec<f32>> = Vec::with_capacity(n);

    // Pick a random first point.
    let first = (xorshift64(rng_state) as usize) % m;
    selected_indices.push(first);
    selected_vecs.push(entries[first].clone());

    // Update min-distance vector after adding the first point.
    for i in 0..m {
        let sim = prior_cosine_similarity(&entries[i], &entries[first])?;
        let dist = 1.0 - sim;
        if dist < min_dist[i] {
            min_dist[i] = dist;
        }
    }

    for _ in 1..n {
        // Choose the candidate that maximises min distance to the selected set.
        let mut best_idx = 0;
        let mut best_dist = f32::NEG_INFINITY;
        for (i, &d) in min_dist.iter().enumerate() {
            if selected_indices.contains(&i) {
                continue;
            }
            if d > best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        selected_indices.push(best_idx);
        selected_vecs.push(entries[best_idx].clone());

        // Update min-distances.
        for i in 0..m {
            let sim = prior_cosine_similarity(&entries[i], &entries[best_idx])?;
            let dist = 1.0 - sim;
            if dist < min_dist[i] {
                min_dist[i] = dist;
            }
        }
    }

    Ok(selected_vecs)
}

/// Format `PriorStats` as a human-readable multi-line string.
pub fn format_prior_stats(stats: &PriorStats) -> String {
    format!(
        "PriorStats {{ mean_loss: {:.6}, min_loss: {:.6}, max_loss: {:.6}, \
         total_samples: {}, buffer_fill_ratio: {:.3}, ema_loss: {:.6} }}",
        stats.mean_loss,
        stats.min_loss,
        stats.max_loss,
        stats.total_samples,
        stats.buffer_fill_ratio,
        stats.ema_loss,
    )
}

// ---------------------------------------------------------------------------
// PriorPreservationTracker
// ---------------------------------------------------------------------------

/// High-level tracker that couples the class prior buffer, configuration,
/// statistics, and loss history for use during a training loop.
#[derive(Debug, Clone)]
pub struct PriorPreservationTracker {
    config: PriorPreservationConfig,
    buffer: ClassPriorBuffer,
    stats: PriorStats,
    /// Per-step losses, capped at the last 1 000 entries.
    loss_history: Vec<f32>,
    current_step: usize,
}

impl PriorPreservationTracker {
    /// Create a new tracker from a configuration.
    pub fn new(config: PriorPreservationConfig) -> Self {
        let buffer = ClassPriorBuffer::new(config.buffer_capacity, config.latent_dim);
        Self {
            config,
            buffer,
            stats: PriorStats::default(),
            loss_history: Vec::new(),
            current_step: 0,
        }
    }

    /// Push a new class latent into the internal buffer.
    ///
    /// # Errors
    ///
    /// Propagates [`PriorError::DimensionMismatch`] from [`ClassPriorBuffer::push`].
    pub fn add_class_latent(&mut self, latent: Vec<f32>) -> Result<(), PriorError> {
        self.buffer.push(latent)
    }

    /// Compute the prior preservation loss for the current step.
    ///
    /// Samples one target latent from the buffer (with replacement) and
    /// computes the MSE loss against `pred`.  Stats and history are updated.
    ///
    /// # Errors
    ///
    /// - [`PriorError::EmptyBuffer`] if no class latents have been added.
    /// - [`PriorError::DimensionMismatch`] if `pred.len()` does not match
    ///   `config.latent_dim`.
    pub fn compute_loss(&mut self, pred: &[f32], rng_state: &mut u64) -> Result<f32, PriorError> {
        let targets = self.buffer.sample(1, rng_state)?;
        let target = &targets[0];
        let loss = prior_preservation_loss(pred, target, &self.config, self.current_step)?;
        let decay = self.config.ema_decay;
        prior_update_stats(&mut self.stats, loss, &self.buffer, decay);
        // Cap history at 1 000 entries.
        if self.loss_history.len() >= 1000 {
            self.loss_history.remove(0);
        }
        self.loss_history.push(loss);
        Ok(loss)
    }

    /// Current training step.
    #[inline]
    pub fn step(&self) -> usize {
        self.current_step
    }

    /// Advance the training step counter by one.
    #[inline]
    pub fn advance_step(&mut self) {
        self.current_step += 1;
    }

    /// Borrow the current running statistics.
    #[inline]
    pub fn stats(&self) -> &PriorStats {
        &self.stats
    }

    /// Borrow the loss history (up to the last 1 000 entries).
    #[inline]
    pub fn loss_history(&self) -> &[f32] {
        &self.loss_history
    }

    /// Whether the prior loss is currently active.
    ///
    /// Returns `true` when the current step is at or beyond `warmup_steps`
    /// **and** the buffer contains at least one entry.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.current_step >= self.config.warmup_steps && !self.buffer.is_empty()
    }

    /// Borrow the underlying class prior buffer.
    #[inline]
    pub fn buffer(&self) -> &ClassPriorBuffer {
        &self.buffer
    }

    /// Borrow the configuration.
    #[inline]
    pub fn config(&self) -> &PriorPreservationConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // PRNG helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_xorshift64_nonzero() -> Result<(), PriorError> {
        let mut state = 12345u64;
        for _ in 0..1000 {
            let v = xorshift64(&mut state);
            assert!(v != 0, "xorshift64 should never produce 0");
        }
        Ok(())
    }

    #[test]
    fn test_xorshift_f32_range() -> Result<(), PriorError> {
        let mut state = 99999u64;
        for _ in 0..1000 {
            let v = xorshift_f32(&mut state);
            assert!((0.0..=1.0).contains(&v), "f32 sample {v} out of [0,1]");
        }
        Ok(())
    }

    #[test]
    fn test_xorshift64_deterministic() -> Result<(), PriorError> {
        let mut s1 = 7u64;
        let mut s2 = 7u64;
        for _ in 0..100 {
            assert_eq!(xorshift64(&mut s1), xorshift64(&mut s2));
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // PriorPreservationConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default_valid() -> Result<(), PriorError> {
        PriorPreservationConfig::default().validate()
    }

    #[test]
    fn test_config_negative_weight_invalid() {
        let cfg = PriorPreservationConfig {
            weight: -1.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_capacity_invalid() {
        let cfg = PriorPreservationConfig {
            buffer_capacity: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_latent_dim_invalid() {
        let cfg = PriorPreservationConfig {
            latent_dim: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_temperature_invalid() {
        let cfg = PriorPreservationConfig {
            temperature: 0.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_ema_decay_out_of_range() {
        let cfg = PriorPreservationConfig {
            ema_decay: 1.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // ClassPriorBuffer
    // -----------------------------------------------------------------------

    #[test]
    fn test_buffer_push_and_len() -> Result<(), PriorError> {
        let mut buf = ClassPriorBuffer::new(4, 3);
        assert!(buf.is_empty());
        buf.push(vec![1.0, 2.0, 3.0])?;
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());
        Ok(())
    }

    #[test]
    fn test_buffer_push_wrong_dim() {
        let mut buf = ClassPriorBuffer::new(4, 3);
        let res = buf.push(vec![1.0, 2.0]); // dim 2, expected 3
        assert!(matches!(res, Err(PriorError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_buffer_circular_overflow() -> Result<(), PriorError> {
        let mut buf = ClassPriorBuffer::new(3, 2);
        for i in 0..6usize {
            buf.push(vec![i as f32, i as f32])?;
        }
        assert_eq!(
            buf.len(),
            3,
            "circular buffer should not grow beyond capacity"
        );
        assert!(buf.is_full());
        Ok(())
    }

    #[test]
    fn test_buffer_sample_empty() {
        let buf = ClassPriorBuffer::new(4, 3);
        let mut rng = 1u64;
        let res = buf.sample(2, &mut rng);
        assert!(matches!(res, Err(PriorError::EmptyBuffer)));
    }

    #[test]
    fn test_buffer_sample_zero_n() -> Result<(), PriorError> {
        let mut buf = ClassPriorBuffer::new(4, 3);
        buf.push(vec![1.0, 2.0, 3.0])?;
        let mut rng = 1u64;
        let res = buf.sample(0, &mut rng);
        assert!(matches!(res, Err(PriorError::InvalidConfig(_))));
        Ok(())
    }

    #[test]
    fn test_buffer_sample_returns_correct_count() -> Result<(), PriorError> {
        let mut buf = ClassPriorBuffer::new(10, 4);
        for i in 0..5usize {
            buf.push(vec![i as f32; 4])?;
        }
        let mut rng = 42u64;
        let samples = buf.sample(3, &mut rng)?;
        assert_eq!(samples.len(), 3);
        for s in &samples {
            assert_eq!(s.len(), 4);
        }
        Ok(())
    }

    #[test]
    fn test_buffer_capacity_accessor() {
        let buf = ClassPriorBuffer::new(8, 2);
        assert_eq!(buf.capacity(), 8);
        assert_eq!(buf.latent_dim(), 2);
    }

    #[test]
    fn test_buffer_total_added_tracks_overflow() -> Result<(), PriorError> {
        let mut buf = ClassPriorBuffer::new(2, 1);
        buf.push(vec![1.0])?;
        buf.push(vec![2.0])?;
        buf.push(vec![3.0])?; // overwrites first slot
        assert_eq!(buf.total_added(), 3);
        assert_eq!(buf.len(), 2); // still only 2 slots
        Ok(())
    }

    // -----------------------------------------------------------------------
    // prior_softmax
    // -----------------------------------------------------------------------

    #[test]
    fn test_softmax_sums_to_one() -> Result<(), PriorError> {
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let probs = prior_softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum {sum} != 1");
        Ok(())
    }

    #[test]
    fn test_softmax_single_element() -> Result<(), PriorError> {
        let probs = prior_softmax(&[42.0]);
        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_softmax_empty() -> Result<(), PriorError> {
        let probs = prior_softmax(&[]);
        assert!(probs.is_empty());
        Ok(())
    }

    #[test]
    fn test_softmax_handles_negative_logits() -> Result<(), PriorError> {
        let logits = vec![-100.0, -200.0, -50.0];
        let probs = prior_softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Largest logit should give the largest probability.
        assert!(probs[2] > probs[0] && probs[0] > probs[1]);
        Ok(())
    }

    #[test]
    fn test_softmax_uniform_logits() -> Result<(), PriorError> {
        let logits = vec![0.0; 5];
        let probs = prior_softmax(&logits);
        for p in &probs {
            assert!((p - 0.2).abs() < 1e-6, "expected 0.2, got {p}");
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // prior_kl_divergence
    // -----------------------------------------------------------------------

    #[test]
    fn test_kl_self_divergence_is_zero() -> Result<(), PriorError> {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let kl = prior_kl_divergence(&p, &p)?;
        assert!(kl.abs() < 1e-6, "KL(p||p) should be 0, got {kl}");
        Ok(())
    }

    #[test]
    fn test_kl_asymmetry() -> Result<(), PriorError> {
        let p = vec![0.9, 0.1];
        let q = vec![0.5, 0.5];
        let kl_pq = prior_kl_divergence(&p, &q)?;
        let kl_qp = prior_kl_divergence(&q, &p)?;
        assert!(
            (kl_pq - kl_qp).abs() > 1e-4,
            "KL divergence should be asymmetric"
        );
        Ok(())
    }

    #[test]
    fn test_kl_q_zero_returns_infinity() -> Result<(), PriorError> {
        let p = vec![0.5, 0.5];
        let q = vec![1.0, 0.0]; // q[1] = 0 but p[1] = 0.5
        let kl = prior_kl_divergence(&p, &q)?;
        assert!(
            kl.is_infinite(),
            "KL should be infinite when q_i=0 and p_i>0"
        );
        Ok(())
    }

    #[test]
    fn test_kl_empty_error() {
        let res = prior_kl_divergence(&[], &[]);
        assert!(matches!(res, Err(PriorError::EmptyBuffer)));
    }

    #[test]
    fn test_kl_dimension_mismatch() {
        let p = vec![0.5, 0.5];
        let q = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let res = prior_kl_divergence(&p, &q);
        assert!(matches!(res, Err(PriorError::DimensionMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // prior_cosine_similarity
    // -----------------------------------------------------------------------

    #[test]
    fn test_cosine_identical_vectors() -> Result<(), PriorError> {
        let a = vec![1.0, 2.0, 3.0];
        let sim = prior_cosine_similarity(&a, &a)?;
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "identical vectors should have cos=1"
        );
        Ok(())
    }

    #[test]
    fn test_cosine_orthogonal_vectors() -> Result<(), PriorError> {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = prior_cosine_similarity(&a, &b)?;
        assert!(sim.abs() < 1e-6, "orthogonal vectors should have cos=0");
        Ok(())
    }

    #[test]
    fn test_cosine_opposite_vectors() -> Result<(), PriorError> {
        let a = vec![1.0, 2.0];
        let b = vec![-1.0, -2.0];
        let sim = prior_cosine_similarity(&a, &b)?;
        assert!(
            (sim + 1.0).abs() < 1e-6,
            "opposite vectors should have cos=-1"
        );
        Ok(())
    }

    #[test]
    fn test_cosine_zero_vector() -> Result<(), PriorError> {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = prior_cosine_similarity(&a, &b)?;
        assert_eq!(sim, 0.0, "zero vector should return 0");
        Ok(())
    }

    #[test]
    fn test_cosine_empty_error() {
        let res = prior_cosine_similarity(&[], &[]);
        assert!(matches!(res, Err(PriorError::EmptyBuffer)));
    }

    #[test]
    fn test_cosine_dimension_mismatch() {
        let res = prior_cosine_similarity(&[1.0, 2.0], &[1.0]);
        assert!(matches!(res, Err(PriorError::DimensionMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // prior_effective_weight
    // -----------------------------------------------------------------------

    #[test]
    fn test_effective_weight_before_warmup() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig {
            weight: 2.0,
            warmup_steps: 100,
            ..Default::default()
        };
        assert_eq!(prior_effective_weight(&cfg, 0), 0.0);
        assert_eq!(prior_effective_weight(&cfg, 99), 0.0);
        Ok(())
    }

    #[test]
    fn test_effective_weight_at_warmup_start() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig {
            weight: 2.0,
            warmup_steps: 100,
            ..Default::default()
        };
        // At step == warmup_steps the ramp is at 0%.
        let w = prior_effective_weight(&cfg, 100);
        assert!(w.abs() < 1e-6, "expected ~0 at step=warmup_steps, got {w}");
        Ok(())
    }

    #[test]
    fn test_effective_weight_midway_through_ramp() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig {
            weight: 2.0,
            warmup_steps: 100,
            ..Default::default()
        };
        // At step 150 (halfway through the ramp [100, 200]) weight should be 1.0.
        let w = prior_effective_weight(&cfg, 150);
        assert!((w - 1.0).abs() < 1e-5, "expected 1.0 at step=150, got {w}");
        Ok(())
    }

    #[test]
    fn test_effective_weight_after_full_ramp() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig {
            weight: 2.0,
            warmup_steps: 100,
            ..Default::default()
        };
        let w = prior_effective_weight(&cfg, 200);
        assert!((w - 2.0).abs() < 1e-6, "expected 2.0 at step=200, got {w}");
        let w2 = prior_effective_weight(&cfg, 9999);
        assert!((w2 - 2.0).abs() < 1e-6, "expected 2.0 after ramp, got {w2}");
        Ok(())
    }

    #[test]
    fn test_effective_weight_zero_warmup() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig {
            weight: 3.0,
            warmup_steps: 0,
            ..Default::default()
        };
        // No warmup → always return config.weight.
        assert_eq!(prior_effective_weight(&cfg, 0), 3.0);
        assert_eq!(prior_effective_weight(&cfg, 1000), 3.0);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // prior_preservation_loss
    // -----------------------------------------------------------------------

    #[test]
    fn test_mse_zero_when_pred_equals_target() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig {
            warmup_steps: 0,
            ..Default::default()
        };
        let v = vec![1.0, 2.0, 3.0];
        let loss = prior_preservation_loss(&v, &v, &cfg, 0)?;
        assert!(
            loss.abs() < 1e-7,
            "loss should be 0 when pred==target, got {loss}"
        );
        Ok(())
    }

    #[test]
    fn test_mse_scales_with_weight() -> Result<(), PriorError> {
        let pred = vec![1.0, 0.0];
        let target = vec![0.0, 0.0];
        let cfg1 = PriorPreservationConfig {
            weight: 1.0,
            warmup_steps: 0,
            ..Default::default()
        };
        let cfg2 = PriorPreservationConfig {
            weight: 3.0,
            warmup_steps: 0,
            ..Default::default()
        };
        let l1 = prior_preservation_loss(&pred, &target, &cfg1, 0)?;
        let l2 = prior_preservation_loss(&pred, &target, &cfg2, 0)?;
        assert!(
            (l2 - 3.0 * l1).abs() < 1e-6,
            "loss should scale with weight"
        );
        Ok(())
    }

    #[test]
    fn test_mse_zero_during_warmup() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig {
            weight: 5.0,
            warmup_steps: 1000,
            ..Default::default()
        };
        let pred = vec![1.0f32; 4];
        let target = vec![0.0f32; 4];
        let loss = prior_preservation_loss(&pred, &target, &cfg, 500)?;
        assert_eq!(loss, 0.0, "loss should be 0 during warmup");
        Ok(())
    }

    #[test]
    fn test_mse_dimension_mismatch() {
        let cfg = PriorPreservationConfig {
            warmup_steps: 0,
            ..Default::default()
        };
        let res = prior_preservation_loss(&[1.0, 2.0], &[1.0], &cfg, 0);
        assert!(matches!(res, Err(PriorError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_mse_empty_input() {
        let cfg = PriorPreservationConfig {
            warmup_steps: 0,
            ..Default::default()
        };
        let res = prior_preservation_loss(&[], &[], &cfg, 0);
        assert!(matches!(res, Err(PriorError::EmptyBuffer)));
    }

    // -----------------------------------------------------------------------
    // prior_soft_loss
    // -----------------------------------------------------------------------

    #[test]
    fn test_soft_loss_identical_inputs_near_zero() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig::default();
        let v = vec![1.0, 2.0, 3.0];
        let loss = prior_soft_loss(&v, &v, &cfg)?;
        assert!(
            loss.abs() < 1e-5,
            "soft loss with identical inputs should be ~0, got {loss}"
        );
        Ok(())
    }

    #[test]
    fn test_soft_loss_temperature_effect() -> Result<(), PriorError> {
        let pred = vec![1.0, 0.0, -1.0];
        let target = vec![0.5, 0.5, 0.0];
        let cfg_low_t = PriorPreservationConfig {
            temperature: 0.5,
            ..Default::default()
        };
        let cfg_high_t = PriorPreservationConfig {
            temperature: 5.0,
            ..Default::default()
        };
        let loss_low = prior_soft_loss(&pred, &target, &cfg_low_t)?;
        let loss_high = prior_soft_loss(&pred, &target, &cfg_high_t)?;
        // Lower temperature sharpens distributions → larger KL divergence.
        assert!(
            loss_low > loss_high,
            "lower temperature should give higher soft loss ({loss_low} vs {loss_high})"
        );
        Ok(())
    }

    #[test]
    fn test_soft_loss_empty_error() {
        let cfg = PriorPreservationConfig::default();
        let res = prior_soft_loss(&[], &[], &cfg);
        assert!(matches!(res, Err(PriorError::EmptyBuffer)));
    }

    #[test]
    fn test_soft_loss_dimension_mismatch() {
        let cfg = PriorPreservationConfig::default();
        let res = prior_soft_loss(&[1.0, 2.0], &[1.0], &cfg);
        assert!(matches!(res, Err(PriorError::DimensionMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // prior_combined_loss
    // -----------------------------------------------------------------------

    #[test]
    fn test_combined_alpha_one_equals_mse() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig {
            warmup_steps: 0,
            ..Default::default()
        };
        let pred = vec![1.0, 0.5, -0.5];
        let target = vec![0.0, 0.0, 0.0];
        let combined = prior_combined_loss(&pred, &target, 1.0, &cfg, 0)?;
        let mse = prior_preservation_loss(&pred, &target, &cfg, 0)?;
        assert!(
            (combined - mse).abs() < 1e-6,
            "alpha=1 combined should equal MSE: {combined} vs {mse}"
        );
        Ok(())
    }

    #[test]
    fn test_combined_alpha_zero_equals_soft() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig {
            warmup_steps: 0,
            ..Default::default()
        };
        let pred = vec![1.0, 0.5, -0.5];
        let target = vec![0.0, 0.0, 0.0];
        let combined = prior_combined_loss(&pred, &target, 0.0, &cfg, 0)?;
        let soft = prior_soft_loss(&pred, &target, &cfg)?;
        assert!(
            (combined - soft).abs() < 1e-6,
            "alpha=0 combined should equal soft: {combined} vs {soft}"
        );
        Ok(())
    }

    #[test]
    fn test_combined_alpha_half_interpolates() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig {
            warmup_steps: 0,
            ..Default::default()
        };
        let pred = vec![1.0, -1.0];
        let target = vec![0.0, 0.0];
        let mse = prior_preservation_loss(&pred, &target, &cfg, 0)?;
        let soft = prior_soft_loss(&pred, &target, &cfg)?;
        let expected = 0.5 * mse + 0.5 * soft;
        let combined = prior_combined_loss(&pred, &target, 0.5, &cfg, 0)?;
        assert!(
            (combined - expected).abs() < 1e-6,
            "combined at alpha=0.5: expected {expected}, got {combined}"
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // prior_per_sample_losses
    // -----------------------------------------------------------------------

    #[test]
    fn test_per_sample_losses_basic() -> Result<(), PriorError> {
        // 2 samples, latent_dim=3.
        let pred = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let target = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let losses = prior_per_sample_losses(&pred, &target, 3)?;
        assert_eq!(losses.len(), 2);
        // First sample: MSE = (1^2 + 0 + 0) / 3 = 1/3
        assert!((losses[0] - 1.0 / 3.0).abs() < 1e-6);
        // Second sample: MSE = (0 + 1^2 + 0) / 3 = 1/3
        assert!((losses[1] - 1.0 / 3.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_per_sample_losses_zero_when_identical() -> Result<(), PriorError> {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let losses = prior_per_sample_losses(&v, &v, 2)?;
        for l in losses {
            assert!(l.abs() < 1e-7);
        }
        Ok(())
    }

    #[test]
    fn test_per_sample_losses_empty_error() {
        let res = prior_per_sample_losses(&[], &[], 4);
        assert!(matches!(res, Err(PriorError::EmptyBuffer)));
    }

    #[test]
    fn test_per_sample_losses_dim_mismatch_lengths() {
        let res = prior_per_sample_losses(&[1.0, 2.0], &[1.0], 1);
        assert!(matches!(res, Err(PriorError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_per_sample_losses_invalid_latent_dim() {
        let pred = vec![1.0, 2.0, 3.0];
        let target = vec![0.0; 3];
        // latent_dim=2 does not divide 3 evenly.
        let res = prior_per_sample_losses(&pred, &target, 2);
        assert!(matches!(res, Err(PriorError::DimensionMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // prior_update_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_stats_first_sample() -> Result<(), PriorError> {
        let buf = ClassPriorBuffer::new(10, 4);
        let mut stats = PriorStats::default();
        prior_update_stats(&mut stats, 2.0, &buf, 0.9);
        assert_eq!(stats.total_samples, 1);
        assert!((stats.mean_loss - 2.0).abs() < 1e-6);
        assert!((stats.min_loss - 2.0).abs() < 1e-6);
        assert!((stats.max_loss - 2.0).abs() < 1e-6);
        // EMA: 0.9*0 + 0.1*2.0 = 0.2
        assert!((stats.ema_loss - 0.2).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_update_stats_min_max_tracking() -> Result<(), PriorError> {
        let buf = ClassPriorBuffer::new(10, 4);
        let mut stats = PriorStats::default();
        for loss in [5.0f32, 1.0, 3.0] {
            prior_update_stats(&mut stats, loss, &buf, 0.9);
        }
        assert!((stats.min_loss - 1.0).abs() < 1e-6);
        assert!((stats.max_loss - 5.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_update_stats_buffer_fill_ratio() -> Result<(), PriorError> {
        let mut buf = ClassPriorBuffer::new(4, 1);
        buf.push(vec![0.0])?;
        buf.push(vec![1.0])?;
        let mut stats = PriorStats::default();
        prior_update_stats(&mut stats, 1.0, &buf, 0.9);
        assert!((stats.buffer_fill_ratio - 0.5).abs() < 1e-6);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // prior_diverse_sample
    // -----------------------------------------------------------------------

    #[test]
    fn test_diverse_sample_returns_n_items() -> Result<(), PriorError> {
        let mut buf = ClassPriorBuffer::new(10, 3);
        for i in 0..8usize {
            buf.push(vec![i as f32, 0.0, 0.0])?;
        }
        let mut rng = 7u64;
        let samples = prior_diverse_sample(&buf, 4, &mut rng)?;
        assert_eq!(samples.len(), 4);
        Ok(())
    }

    #[test]
    fn test_diverse_sample_distinct_indices() -> Result<(), PriorError> {
        // Five orthogonal unit vectors in R^5 — they should all be selected when n=5.
        let mut buf = ClassPriorBuffer::new(10, 5);
        for i in 0..5usize {
            let mut v = vec![0.0f32; 5];
            v[i] = 1.0;
            buf.push(v)?;
        }
        let mut rng = 123u64;
        let samples = prior_diverse_sample(&buf, 5, &mut rng)?;
        assert_eq!(samples.len(), 5);
        // All selected samples should be distinct (they have exactly one 1.0).
        let mut seen: Vec<usize> = Vec::new();
        for s in &samples {
            let idx = s.iter().position(|&x| x == 1.0);
            let idx = idx.ok_or(PriorError::EmptyBuffer)?;
            assert!(!seen.contains(&idx), "duplicate sample selected");
            seen.push(idx);
        }
        Ok(())
    }

    #[test]
    fn test_diverse_sample_empty_buffer_error() {
        let buf = ClassPriorBuffer::new(4, 2);
        let mut rng = 1u64;
        let res = prior_diverse_sample(&buf, 1, &mut rng);
        assert!(matches!(res, Err(PriorError::EmptyBuffer)));
    }

    #[test]
    fn test_diverse_sample_n_too_large() -> Result<(), PriorError> {
        let mut buf = ClassPriorBuffer::new(4, 2);
        buf.push(vec![1.0, 0.0])?;
        let mut rng = 1u64;
        let res = prior_diverse_sample(&buf, 5, &mut rng);
        assert!(matches!(res, Err(PriorError::IndexOutOfBounds(_))));
        Ok(())
    }

    #[test]
    fn test_diverse_sample_n_zero_error() -> Result<(), PriorError> {
        let mut buf = ClassPriorBuffer::new(4, 2);
        buf.push(vec![1.0, 0.0])?;
        let mut rng = 1u64;
        let res = prior_diverse_sample(&buf, 0, &mut rng);
        assert!(matches!(res, Err(PriorError::InvalidConfig(_))));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // format_prior_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_prior_stats_non_empty() -> Result<(), PriorError> {
        let stats = PriorStats {
            mean_loss: 0.5,
            min_loss: 0.1,
            max_loss: 0.9,
            total_samples: 42,
            buffer_fill_ratio: 0.75,
            ema_loss: 0.45,
        };
        let s = format_prior_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("0.500000") || s.contains("mean_loss"));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // PriorPreservationTracker — full lifecycle
    // -----------------------------------------------------------------------

    #[test]
    fn test_tracker_initial_state() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig::default();
        let tracker = PriorPreservationTracker::new(cfg);
        assert_eq!(tracker.step(), 0);
        assert!(tracker.loss_history().is_empty());
        assert!(!tracker.is_active()); // buffer is empty
        Ok(())
    }

    #[test]
    fn test_tracker_add_latent() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig {
            warmup_steps: 0,
            ..Default::default()
        };
        let mut tracker = PriorPreservationTracker::new(cfg);
        tracker.add_class_latent(vec![0.0; 512])?;
        assert!(!tracker.buffer().is_empty());
        assert!(tracker.is_active());
        Ok(())
    }

    #[test]
    fn test_tracker_is_not_active_during_warmup() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig {
            warmup_steps: 100,
            ..Default::default()
        };
        let mut tracker = PriorPreservationTracker::new(cfg);
        tracker.add_class_latent(vec![0.0; 512])?;
        // Step is 0 < warmup_steps=100 → not active.
        assert!(!tracker.is_active());
        Ok(())
    }

    #[test]
    fn test_tracker_becomes_active_after_warmup() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig {
            warmup_steps: 5,
            ..Default::default()
        };
        let mut tracker = PriorPreservationTracker::new(cfg);
        tracker.add_class_latent(vec![0.0; 512])?;
        for _ in 0..5 {
            tracker.advance_step();
        }
        assert!(tracker.is_active());
        Ok(())
    }

    #[test]
    fn test_tracker_compute_loss() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig {
            warmup_steps: 0,
            weight: 1.0,
            latent_dim: 4,
            buffer_capacity: 8,
            ..Default::default()
        };
        let mut tracker = PriorPreservationTracker::new(cfg);
        tracker.add_class_latent(vec![0.0; 4])?;
        let pred = vec![1.0, 0.0, 0.0, 0.0];
        let mut rng = 42u64;
        let loss = tracker.compute_loss(&pred, &mut rng)?;
        // MSE = (1^2 + 0 + 0 + 0) / 4 = 0.25
        assert!((loss - 0.25).abs() < 1e-6, "expected 0.25, got {loss}");
        assert_eq!(tracker.loss_history().len(), 1);
        assert_eq!(tracker.stats().total_samples, 1);
        Ok(())
    }

    #[test]
    fn test_tracker_advance_step() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig::default();
        let mut tracker = PriorPreservationTracker::new(cfg);
        tracker.advance_step();
        tracker.advance_step();
        assert_eq!(tracker.step(), 2);
        Ok(())
    }

    #[test]
    fn test_tracker_loss_history_capped_at_1000() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig {
            warmup_steps: 0,
            latent_dim: 2,
            buffer_capacity: 4,
            ..Default::default()
        };
        let mut tracker = PriorPreservationTracker::new(cfg);
        tracker.add_class_latent(vec![0.0; 2])?;
        let pred = vec![0.0f32; 2];
        let mut rng = 1u64;
        for _ in 0..1050 {
            tracker.compute_loss(&pred, &mut rng)?;
        }
        assert!(
            tracker.loss_history().len() <= 1000,
            "history should be capped at 1000, got {}",
            tracker.loss_history().len()
        );
        Ok(())
    }

    #[test]
    fn test_tracker_compute_loss_empty_buffer_error() {
        let cfg = PriorPreservationConfig {
            warmup_steps: 0,
            ..Default::default()
        };
        let mut tracker = PriorPreservationTracker::new(cfg);
        let pred = vec![0.0f32; 512];
        let mut rng = 1u64;
        let res = tracker.compute_loss(&pred, &mut rng);
        assert!(matches!(res, Err(PriorError::EmptyBuffer)));
    }

    #[test]
    fn test_tracker_compute_loss_returns_zero_during_warmup() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig {
            warmup_steps: 1000,
            weight: 5.0,
            latent_dim: 4,
            buffer_capacity: 8,
            ..Default::default()
        };
        let mut tracker = PriorPreservationTracker::new(cfg);
        tracker.add_class_latent(vec![0.0; 4])?;
        let pred = vec![9.9f32; 4];
        let mut rng = 42u64;
        // step=0 < warmup_steps=1000
        let loss = tracker.compute_loss(&pred, &mut rng)?;
        assert_eq!(loss, 0.0);
        Ok(())
    }

    #[test]
    fn test_tracker_stats_update_mean() -> Result<(), PriorError> {
        let cfg = PriorPreservationConfig {
            warmup_steps: 0,
            weight: 1.0,
            latent_dim: 2,
            buffer_capacity: 4,
            ema_decay: 0.0, // instant EMA for easy testing
            ..Default::default()
        };
        let mut tracker = PriorPreservationTracker::new(cfg);
        tracker.add_class_latent(vec![0.0; 2])?;
        // pred = [1,0], target = [0,0] → MSE = 0.5
        let pred = vec![1.0f32, 0.0];
        let mut rng = 42u64;
        tracker.compute_loss(&pred, &mut rng)?;
        assert!((tracker.stats().mean_loss - 0.5).abs() < 1e-5);
        Ok(())
    }
}
