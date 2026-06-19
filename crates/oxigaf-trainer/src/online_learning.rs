//! Online / streaming learning utilities for live-capture avatar training.
//!
//! Provides:
//! - [`StreamingBuffer`] — reservoir-sampled fixed-capacity buffer
//! - [`OnlineLossHistory`] — EMA + windowed loss statistics
//! - [`OnlineGradientStats`] — per-parameter Welford online gradient statistics
//! - [`AdaptiveLrState`] — loss-driven learning rate adaptation
//! - Various free functions for reservoir sampling, EMA, linear regression, etc.

use thiserror::Error;

// ─── Error ──────────────────────────────────────────────────────────────────

/// Errors produced by the online-learning subsystem.
#[derive(Debug, Error, PartialEq)]
pub enum OnlineLearningError {
    #[error("Buffer capacity must be > 0")]
    ZeroCapacity,

    #[error("Sample size {requested} exceeds buffer size {available}")]
    SampleTooLarge { requested: usize, available: usize },

    #[error("Empty buffer: cannot sample from empty buffer")]
    EmptyBuffer,

    #[error("Invalid decay rate {rate}: must be in (0, 1)")]
    InvalidDecayRate { rate: f32 },

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

// ─── xorshift64 ─────────────────────────────────────────────────────────────

/// Inline xorshift64 PRNG (no `rand` dependency).
fn xorshift64(state: &mut u64) -> u64 {
    if *state == 0 {
        *state = 1;
    }
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

// ─── StreamingBuffer ─────────────────────────────────────────────────────────

/// A fixed-capacity reservoir buffer that maintains a uniform random sample of
/// all items seen so far.
///
/// Uses reservoir sampling (Algorithm R) so that after N pushes, every item
/// has equal probability `capacity / N` of being in the buffer.
pub struct StreamingBuffer<T: Clone> {
    data: Vec<T>,
    capacity: usize,
    total_seen: usize,
    rng_state: u64,
}

impl<T: Clone> StreamingBuffer<T> {
    /// Create a new buffer with the given capacity and RNG seed.
    pub fn new(capacity: usize, seed: u64) -> Result<Self, OnlineLearningError> {
        if capacity == 0 {
            return Err(OnlineLearningError::ZeroCapacity);
        }
        Ok(Self {
            data: Vec::with_capacity(capacity),
            capacity,
            total_seen: 0,
            rng_state: seed.max(1),
        })
    }

    /// Add item to the buffer using reservoir sampling.
    ///
    /// If the buffer is not yet full, the item is appended unconditionally.
    /// Once full, the item replaces a random position with probability
    /// `capacity / total_seen`.
    pub fn push(&mut self, item: T) {
        self.total_seen += 1;
        if self.data.len() < self.capacity {
            self.data.push(item);
        } else {
            // Reservoir replacement: j ∈ [0, total_seen)
            let j = (xorshift64(&mut self.rng_state) % self.total_seen as u64) as usize;
            if j < self.capacity {
                self.data[j] = item;
            }
        }
    }

    /// Number of items currently stored in the buffer.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the buffer contains no items.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Maximum number of items the buffer can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Total number of items ever pushed (including those evicted).
    pub fn total_seen(&self) -> usize {
        self.total_seen
    }

    /// Returns `true` when `len() == capacity`.
    pub fn is_full(&self) -> bool {
        self.data.len() == self.capacity
    }

    /// Get an item by index without bounds panics.
    pub fn get(&self, idx: usize) -> Option<&T> {
        self.data.get(idx)
    }

    /// Sample `n` distinct items from the buffer without replacement.
    ///
    /// Uses partial Fisher-Yates on the index set `[0, len)`.
    pub fn sample(&mut self, n: usize) -> Result<Vec<T>, OnlineLearningError> {
        let len = self.data.len();
        if len == 0 {
            return Err(OnlineLearningError::EmptyBuffer);
        }
        if n > len {
            return Err(OnlineLearningError::SampleTooLarge {
                requested: n,
                available: len,
            });
        }
        // Build an index vector, then do partial Fisher-Yates for the first n slots.
        let mut indices: Vec<usize> = (0..len).collect();
        for i in 0..n {
            let remaining = len - i;
            let j = i + (xorshift64(&mut self.rng_state) % remaining as u64) as usize;
            indices.swap(i, j);
        }
        let result = indices[..n]
            .iter()
            .map(|&idx| self.data[idx].clone())
            .collect();
        Ok(result)
    }

    /// Clear all items and reset the seen counter.
    pub fn clear(&mut self) {
        self.data.clear();
        self.total_seen = 0;
    }

    /// Fraction of the buffer that is filled: `len / capacity`.
    pub fn fill_fraction(&self) -> f32 {
        self.data.len() as f32 / self.capacity as f32
    }
}

// ─── Free functions ──────────────────────────────────────────────────────────

/// Perform one exponential moving average update.
///
/// `result = decay * current_ema + (1 - decay) * new_value`
pub fn ema_update(current_ema: f32, new_value: f32, decay: f32) -> f32 {
    decay * current_ema + (1.0 - decay) * new_value
}

/// Compute the linear regression slope and intercept over a slice of `f32`
/// values (x = 0, 1, 2, …, n-1).
///
/// Returns `(slope, intercept)`.  Degenerate cases (n ≤ 1) return slope = 0.0.
pub fn linear_regression(values: &[f32]) -> (f32, f32) {
    let n = values.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    if n == 1 {
        return (0.0, values[0]);
    }
    let nf = n as f32;
    let sum_x: f32 = (0..n).map(|i| i as f32).sum();
    let sum_y: f32 = values.iter().sum();
    let sum_xy: f32 = values.iter().enumerate().map(|(i, &y)| i as f32 * y).sum();
    let sum_x2: f32 = (0..n).map(|i| (i as f32) * (i as f32)).sum();

    let denom = nf * sum_x2 - sum_x * sum_x;
    if denom.abs() < f32::EPSILON {
        // All x values identical (only possible for n=1, handled above, but guard anyway).
        let intercept = sum_y / nf;
        return (0.0, intercept);
    }
    let slope = (nf * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / nf;
    (slope, intercept)
}

/// One step of Welford's online algorithm for computing running mean and M2.
///
/// Returns `(new_mean, new_m2)`.
///
/// `count` must already be incremented before calling (i.e., it is the new count
/// including the current sample).
pub fn welford_update(count: usize, mean: f32, m2: f32, new_value: f32) -> (f32, f32) {
    let delta = new_value - mean;
    let new_mean = mean + delta / count as f32;
    let delta2 = new_value - new_mean;
    let new_m2 = m2 + delta * delta2;
    (new_mean, new_m2)
}

/// Detect a training stall: returns `true` when the last `n_steps` values are
/// all within `threshold` of their mean.
///
/// Returns `false` if there are fewer than `n_steps` values.
pub fn detect_stall(history: &[f32], n_steps: usize, threshold: f32) -> bool {
    if n_steps == 0 || history.len() < n_steps {
        return false;
    }
    let window = &history[history.len() - n_steps..];
    let mean = window.iter().sum::<f32>() / n_steps as f32;
    window.iter().all(|&v| (v - mean).abs() <= threshold)
}

/// Reservoir-sample `n` distinct items from `items` using xorshift64.
///
/// Returns `Err` if `n > items.len()` or the slice is empty and `n > 0`.
pub fn reservoir_sample<T: Clone>(
    items: &[T],
    n: usize,
    seed: u64,
) -> Result<Vec<T>, OnlineLearningError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let len = items.len();
    if len == 0 {
        return Err(OnlineLearningError::EmptyBuffer);
    }
    if n > len {
        return Err(OnlineLearningError::SampleTooLarge {
            requested: n,
            available: len,
        });
    }
    // Algorithm R: fill reservoir with first n, then probabilistically replace.
    let mut reservoir: Vec<T> = items[..n].to_vec();
    let mut state = seed.max(1);
    for (i, item) in (n..len).zip(items[n..].iter()) {
        // j ∈ [0, i+1)
        let j = (xorshift64(&mut state) % (i + 1) as u64) as usize;
        if j < n {
            reservoir[j] = item.clone();
        }
    }
    Ok(reservoir)
}

/// Sample `n` indices from `[0, weights.len())` proportional to `weights`.
///
/// Uses CDF construction + binary search.  All-zero weights fall back to
/// uniform sampling.  Returns an error if `n > weights.len()` or the slice
/// is empty.
pub fn weighted_sample_indices(
    weights: &[f32],
    n: usize,
    seed: u64,
) -> Result<Vec<usize>, OnlineLearningError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let len = weights.len();
    if len == 0 {
        return Err(OnlineLearningError::EmptyBuffer);
    }
    // Weighted sampling is with-replacement, so n > len is allowed.

    let total: f32 = weights.iter().map(|w| w.max(0.0)).sum();
    let mut state = seed.max(1);

    // Build normalized CDF.  If all weights are zero, use uniform distribution.
    let cdf: Vec<f32> = if total <= 0.0 {
        // Uniform.
        (1..=len).map(|i| i as f32 / len as f32).collect()
    } else {
        let mut acc = 0.0_f32;
        weights
            .iter()
            .map(|&w| {
                acc += w.max(0.0) / total;
                acc
            })
            .collect()
    };

    // Sample with replacement using CDF (simple approach for weighted).
    // We do NOT enforce "without replacement" here because priority-weighted
    // sampling is inherently with-replacement; the spec does not say otherwise.
    let mut result = Vec::with_capacity(n);
    for _ in 0..n {
        // Draw a uniform random in [0, 1).
        let u = (xorshift64(&mut state) as f64 / u64::MAX as f64) as f32;
        // Binary search for the first CDF entry >= u.
        let idx = cdf
            .binary_search_by(|&c| {
                if c < u {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            })
            .unwrap_or_else(|i| i)
            .min(len - 1);
        result.push(idx);
    }
    Ok(result)
}

/// Compute importance weights for prioritized experience replay.
///
/// `weight_i ∝ (1 / P(i))^beta`, where `P(i) = loss_i / sum(losses)`.
/// Weights are normalized so that `max(weight) = 1.0`.
///
/// `beta = 0` → uniform weights.  `beta = 1` → fully importance-corrected.
///
/// Losses of zero receive weight 0 (no correction needed for unseen samples).
pub fn compute_importance_weights(losses: &[f32], beta: f32) -> Vec<f32> {
    if losses.is_empty() {
        return Vec::new();
    }
    let total: f32 = losses.iter().map(|&l| l.max(0.0)).sum::<f32>();
    if total <= 0.0 || beta == 0.0 {
        // All equal weights.
        return vec![1.0; losses.len()];
    }
    let n = losses.len() as f32;
    // w_i = (n * P(i))^(-beta), but we normalise to max = 1.
    // P(i) = loss_i / total; add eps to avoid division by zero.
    let eps = 1e-8_f32;
    let raw: Vec<f32> = losses
        .iter()
        .map(|&l| {
            let p = l.max(0.0) / total;
            if p <= 0.0 {
                0.0
            } else {
                // weight ∝ (1/P)^beta = (total / l)^beta
                ((1.0 / (p * n + eps)).max(eps)).powf(beta)
            }
        })
        .collect();
    let max_w = raw.iter().cloned().fold(0.0_f32, f32::max);
    if max_w <= 0.0 {
        return vec![1.0; losses.len()];
    }
    raw.iter().map(|&w| w / max_w).collect()
}

// ─── OnlineLossHistory ────────────────────────────────────────────────────────

/// Rolling-window loss tracker with exponential moving average.
pub struct OnlineLossHistory {
    /// Raw loss values in the rolling window.
    pub values: Vec<f32>,
    /// Current exponential moving average.
    pub ema: f32,
    /// EMA decay factor (closer to 1 = slower adaptation).
    pub ema_decay: f32,
    /// Maximum number of values retained (rolling window).
    capacity: usize,
}

impl OnlineLossHistory {
    /// Create a new history tracker.
    ///
    /// `ema_decay` must be in `(0, 1)`.  `capacity` must be > 0.
    pub fn new(ema_decay: f32, capacity: usize) -> Result<Self, OnlineLearningError> {
        if ema_decay <= 0.0 || ema_decay >= 1.0 {
            return Err(OnlineLearningError::InvalidDecayRate { rate: ema_decay });
        }
        if capacity == 0 {
            return Err(OnlineLearningError::ZeroCapacity);
        }
        Ok(Self {
            values: Vec::with_capacity(capacity),
            ema: 0.0,
            ema_decay,
            capacity,
        })
    }

    /// Push a new loss value, updating the rolling window and EMA.
    pub fn push(&mut self, loss: f32) {
        if self.values.is_empty() {
            // Initialise EMA to the first value.
            self.ema = loss;
        } else {
            self.ema = ema_update(self.ema, loss, self.ema_decay);
        }
        if self.values.len() == self.capacity {
            self.values.remove(0);
        }
        self.values.push(loss);
    }

    /// Mean of values in the rolling window.
    pub fn mean(&self) -> f32 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f32>() / self.values.len() as f32
    }

    /// Population standard deviation of the rolling window.
    pub fn std(&self) -> f32 {
        let n = self.values.len();
        if n < 2 {
            return 0.0;
        }
        let m = self.mean();
        let variance = self.values.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / n as f32;
        variance.sqrt()
    }

    /// Minimum value in the rolling window.
    pub fn min(&self) -> f32 {
        self.values.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    /// Maximum value in the rolling window.
    pub fn max(&self) -> f32 {
        self.values
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Most recently pushed value.
    pub fn latest(&self) -> Option<f32> {
        self.values.last().copied()
    }

    /// Current EMA value.
    pub fn ema(&self) -> f32 {
        self.ema
    }

    /// Linear regression slope over the rolling window.
    ///
    /// Positive slope = increasing loss (bad).  Negative = decreasing (good).
    pub fn trend(&self) -> f32 {
        linear_regression(&self.values).0
    }

    /// Returns `true` if the last `n_steps` values have not changed more than
    /// `threshold` from their mean.
    pub fn is_stalled(&self, n_steps: usize, threshold: f32) -> bool {
        detect_stall(&self.values, n_steps, threshold)
    }

    /// Number of values currently in the rolling window.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` when no values have been pushed yet.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// ─── OnlineGradientStats ─────────────────────────────────────────────────────

/// Per-parameter gradient statistics maintained with Welford's online algorithm.
pub struct OnlineGradientStats {
    /// Number of parameters tracked.
    pub n_params: usize,
    /// Number of gradient vectors seen so far.
    pub count: usize,
    /// Online running mean per parameter.
    running_mean: Vec<f32>,
    /// Welford M2 (sum of squared deviations) per parameter.
    running_m2: Vec<f32>,
}

impl OnlineGradientStats {
    /// Create a tracker for `n_params` parameters.
    pub fn new(n_params: usize) -> Self {
        Self {
            n_params,
            count: 0,
            running_mean: vec![0.0; n_params],
            running_m2: vec![0.0; n_params],
        }
    }

    /// Update statistics with a new gradient vector using Welford's algorithm.
    pub fn update(&mut self, gradients: &[f32]) -> Result<(), OnlineLearningError> {
        if gradients.len() != self.n_params {
            return Err(OnlineLearningError::DimensionMismatch {
                expected: self.n_params,
                actual: gradients.len(),
            });
        }
        self.count += 1;
        for (i, &g) in gradients.iter().enumerate() {
            let (new_mean, new_m2) =
                welford_update(self.count, self.running_mean[i], self.running_m2[i], g);
            self.running_mean[i] = new_mean;
            self.running_m2[i] = new_m2;
        }
        Ok(())
    }

    /// Per-parameter running mean.
    pub fn mean(&self) -> &[f32] {
        &self.running_mean
    }

    /// Per-parameter population variance (`M2 / count`).
    ///
    /// Returns zeros when `count == 0`.
    pub fn variance(&self) -> Vec<f32> {
        if self.count == 0 {
            return vec![0.0; self.n_params];
        }
        self.running_m2
            .iter()
            .map(|&m2| m2 / self.count as f32)
            .collect()
    }

    /// Per-parameter standard deviation.
    pub fn std_dev(&self) -> Vec<f32> {
        self.variance().iter().map(|&v| v.sqrt()).collect()
    }

    /// Signal-to-noise ratio: `|mean| / (std_dev + eps)`.
    pub fn snr(&self) -> Vec<f32> {
        let std = self.std_dev();
        self.running_mean
            .iter()
            .zip(std.iter())
            .map(|(&m, &s)| m.abs() / (s + 1e-8))
            .collect()
    }

    /// L2 norm of the mean gradient vector.
    pub fn mean_norm(&self) -> f32 {
        self.running_mean.iter().map(|&m| m * m).sum::<f32>().sqrt()
    }

    /// Effective gradient: `mean / (std_dev + 1e-8)` — normalised direction.
    pub fn effective_gradient(&self) -> Vec<f32> {
        let std = self.std_dev();
        self.running_mean
            .iter()
            .zip(std.iter())
            .map(|(&m, &s)| m / (s + 1e-8))
            .collect()
    }

    /// Reset all statistics (mean, M2, count).
    pub fn reset(&mut self) {
        self.count = 0;
        self.running_mean.iter_mut().for_each(|v| *v = 0.0);
        self.running_m2.iter_mut().for_each(|v| *v = 0.0);
    }
}

// ─── AdaptiveLrState ─────────────────────────────────────────────────────────

/// Online learning rate state that reduces the LR when the loss plateaus.
pub struct AdaptiveLrState {
    /// Original base learning rate.
    pub base_lr: f32,
    /// Current effective learning rate.
    pub current_lr: f32,
    /// Number of non-improving steps tolerated before a reduction.
    pub patience: usize,
    /// Factor by which the LR is multiplied on plateau.
    pub reduction_factor: f32,
    /// Hard floor for the learning rate.
    pub min_lr: f32,
    /// Hard ceiling for the learning rate.
    pub max_lr: f32,
    /// Steps since the last improvement.
    stall_count: usize,
    /// Best loss observed so far.
    best_loss: f32,
}

impl AdaptiveLrState {
    /// Create a new adaptive LR state.
    ///
    /// `reduction_factor` must be in `(0, 1)`.
    pub fn new(
        base_lr: f32,
        patience: usize,
        reduction_factor: f32,
    ) -> Result<Self, OnlineLearningError> {
        if reduction_factor <= 0.0 || reduction_factor >= 1.0 {
            return Err(OnlineLearningError::InvalidDecayRate {
                rate: reduction_factor,
            });
        }
        Ok(Self {
            base_lr,
            current_lr: base_lr,
            patience,
            reduction_factor,
            min_lr: 1e-6,
            max_lr: 1e-2,
            stall_count: 0,
            best_loss: f32::INFINITY,
        })
    }

    /// Update with the current loss value.
    ///
    /// Reduces the learning rate if no improvement has been seen for
    /// `patience` consecutive calls.  Returns the (possibly updated) LR.
    pub fn step(&mut self, current_loss: f32) -> f32 {
        // Consider it an improvement if:
        //   - best_loss is still infinity (first step), OR
        //   - loss decreased by at least 0.01% relative to best_loss.
        let threshold = if self.best_loss.is_finite() {
            self.best_loss - 1e-4 * self.best_loss.abs()
        } else {
            f32::INFINITY
        };
        if current_loss < threshold {
            self.best_loss = current_loss;
            self.stall_count = 0;
        } else {
            self.stall_count += 1;
        }

        if self.stall_count >= self.patience {
            let new_lr = (self.current_lr * self.reduction_factor).max(self.min_lr);
            self.current_lr = new_lr;
            self.stall_count = 0;
        }

        self.current_lr
    }

    /// Reset stall counter and LR back to `base_lr`.
    pub fn reset(&mut self) {
        self.current_lr = self.base_lr;
        self.stall_count = 0;
        self.best_loss = f32::INFINITY;
    }

    /// Current learning rate.
    pub fn lr(&self) -> f32 {
        self.current_lr
    }

    /// Returns `true` when the current LR equals or is below `min_lr`.
    pub fn is_at_minimum(&self) -> bool {
        self.current_lr <= self.min_lr
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── StreamingBuffer ──────────────────────────────────────────────────────

    #[test]
    fn streaming_buffer_new_valid() {
        let buf: StreamingBuffer<i32> = StreamingBuffer::new(10, 42).expect("valid capacity");
        assert_eq!(buf.capacity(), 10);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn streaming_buffer_zero_capacity_error() {
        let result: Result<StreamingBuffer<i32>, _> = StreamingBuffer::new(0, 42);
        assert!(matches!(result, Err(OnlineLearningError::ZeroCapacity)));
    }

    #[test]
    fn streaming_buffer_push_below_capacity() {
        let mut buf: StreamingBuffer<i32> = StreamingBuffer::new(5, 1).expect("ok");
        buf.push(1);
        buf.push(2);
        buf.push(3);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.total_seen(), 3);
        assert!(!buf.is_full());
        assert_eq!(buf.get(0), Some(&1));
        assert_eq!(buf.get(2), Some(&3));
        assert_eq!(buf.get(5), None);
    }

    #[test]
    fn streaming_buffer_push_fills_to_capacity() {
        let mut buf: StreamingBuffer<i32> = StreamingBuffer::new(3, 99).expect("ok");
        for i in 0..3 {
            buf.push(i);
        }
        assert!(buf.is_full());
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.total_seen(), 3);
    }

    #[test]
    fn streaming_buffer_reservoir_maintains_size_after_overflow() {
        let mut buf: StreamingBuffer<i32> = StreamingBuffer::new(10, 7).expect("ok");
        for i in 0..100 {
            buf.push(i);
        }
        // Buffer must remain exactly at capacity.
        assert_eq!(buf.len(), 10);
        assert_eq!(buf.total_seen(), 100);
        assert!(buf.is_full());
    }

    #[test]
    fn streaming_buffer_sample_valid() {
        let mut buf: StreamingBuffer<i32> = StreamingBuffer::new(20, 13).expect("ok");
        for i in 0..20 {
            buf.push(i);
        }
        let sample = buf.sample(5).expect("should succeed");
        assert_eq!(sample.len(), 5);
        // All sampled items must be in [0, 20).
        for v in &sample {
            assert!(*v >= 0 && *v < 20, "unexpected value {v}");
        }
    }

    #[test]
    fn streaming_buffer_sample_empty_error() {
        let mut buf: StreamingBuffer<i32> = StreamingBuffer::new(5, 1).expect("ok");
        let result = buf.sample(1);
        assert!(matches!(result, Err(OnlineLearningError::EmptyBuffer)));
    }

    #[test]
    fn streaming_buffer_sample_too_large_error() {
        let mut buf: StreamingBuffer<i32> = StreamingBuffer::new(5, 1).expect("ok");
        buf.push(10);
        buf.push(20);
        let result = buf.sample(5);
        assert!(matches!(
            result,
            Err(OnlineLearningError::SampleTooLarge {
                requested: 5,
                available: 2
            })
        ));
    }

    #[test]
    fn streaming_buffer_clear_resets_state() {
        let mut buf: StreamingBuffer<i32> = StreamingBuffer::new(5, 2).expect("ok");
        for i in 0..5 {
            buf.push(i);
        }
        buf.clear();
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.total_seen(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn streaming_buffer_fill_fraction() {
        let mut buf: StreamingBuffer<i32> = StreamingBuffer::new(4, 5).expect("ok");
        assert!((buf.fill_fraction() - 0.0).abs() < f32::EPSILON);
        buf.push(1);
        buf.push(2);
        assert!((buf.fill_fraction() - 0.5).abs() < 1e-5);
        buf.push(3);
        buf.push(4);
        assert!((buf.fill_fraction() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn streaming_buffer_sample_no_duplicates_indices() {
        // With a small buffer and sampling all items, each must appear at most once.
        let mut buf: StreamingBuffer<i32> = StreamingBuffer::new(6, 77).expect("ok");
        for i in 0..6 {
            buf.push(i);
        }
        let sample = buf.sample(6).expect("ok");
        let mut sorted = sample.clone();
        sorted.sort();
        // Check all 6 distinct values present.
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
    }

    // ── OnlineLossHistory ─────────────────────────────────────────────────────

    #[test]
    fn loss_history_push_and_latest() {
        let mut hist = OnlineLossHistory::new(0.9, 100).expect("ok");
        assert!(hist.is_empty());
        hist.push(1.0);
        assert_eq!(hist.latest(), Some(1.0));
        assert_eq!(hist.len(), 1);
        hist.push(2.0);
        assert_eq!(hist.latest(), Some(2.0));
    }

    #[test]
    fn loss_history_mean() {
        let mut hist = OnlineLossHistory::new(0.9, 100).expect("ok");
        for &v in &[1.0_f32, 2.0, 3.0, 4.0, 5.0] {
            hist.push(v);
        }
        let m = hist.mean();
        assert!((m - 3.0).abs() < 1e-5, "mean = {m}");
    }

    #[test]
    fn loss_history_std() {
        let mut hist = OnlineLossHistory::new(0.9, 100).expect("ok");
        for &v in &[2.0_f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            hist.push(v);
        }
        // population std ≈ 2.0
        let s = hist.std();
        assert!(s > 1.5 && s < 2.5, "std = {s}");
    }

    #[test]
    fn loss_history_min_max() {
        let mut hist = OnlineLossHistory::new(0.9, 100).expect("ok");
        for &v in &[3.0_f32, 1.0, 5.0, 2.0, 4.0] {
            hist.push(v);
        }
        assert!((hist.min() - 1.0).abs() < 1e-6);
        assert!((hist.max() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn loss_history_ema_decay() {
        let mut hist = OnlineLossHistory::new(0.9, 100).expect("ok");
        hist.push(1.0);
        // First push: ema = 1.0
        assert!((hist.ema() - 1.0).abs() < 1e-6);
        hist.push(0.0);
        // ema = 0.9 * 1.0 + 0.1 * 0.0 = 0.9
        assert!((hist.ema() - 0.9).abs() < 1e-5, "ema = {}", hist.ema());
    }

    #[test]
    fn loss_history_rolling_window_eviction() {
        let mut hist = OnlineLossHistory::new(0.9, 3).expect("ok");
        hist.push(10.0);
        hist.push(20.0);
        hist.push(30.0);
        // Window: [10, 20, 30]
        hist.push(40.0);
        // Window: [20, 30, 40]
        assert_eq!(hist.len(), 3);
        assert!((hist.min() - 20.0).abs() < 1e-6);
    }

    #[test]
    fn loss_history_trend_positive() {
        let mut hist = OnlineLossHistory::new(0.9, 100).expect("ok");
        for i in 0..10 {
            hist.push(i as f32);
        }
        // Values: 0, 1, 2, ..., 9 — slope should be 1.0.
        let t = hist.trend();
        assert!(t > 0.5, "slope should be ~1.0, got {t}");
    }

    #[test]
    fn loss_history_trend_negative() {
        let mut hist = OnlineLossHistory::new(0.9, 100).expect("ok");
        for i in (0..10).rev() {
            hist.push(i as f32);
        }
        // Values: 9, 8, 7, ..., 0 — slope should be -1.0.
        let t = hist.trend();
        assert!(t < -0.5, "slope should be ~-1.0, got {t}");
    }

    #[test]
    fn loss_history_is_stalled() {
        let mut hist = OnlineLossHistory::new(0.9, 100).expect("ok");
        for _ in 0..10 {
            hist.push(1.0);
        }
        // All values are 1.0: well within any positive threshold.
        assert!(hist.is_stalled(5, 0.01));
        // Even threshold=0 detects stall because |1.0 - 1.0| = 0.0 ≤ 0.0.
        assert!(hist.is_stalled(5, 0.0));
        // Not enough history for 20 steps.
        assert!(!hist.is_stalled(20, 0.01));
    }

    #[test]
    fn loss_history_is_stalled_not_when_varied() {
        let mut hist = OnlineLossHistory::new(0.9, 100).expect("ok");
        for i in 0..10 {
            hist.push(i as f32);
        }
        // Values spread [0..9]: definitely not stalled with threshold 0.1.
        assert!(!hist.is_stalled(5, 0.1));
    }

    // ── OnlineGradientStats ───────────────────────────────────────────────────

    #[test]
    fn grad_stats_update_welford() {
        let mut stats = OnlineGradientStats::new(2);
        stats.update(&[1.0, 2.0]).expect("ok");
        stats.update(&[3.0, 4.0]).expect("ok");
        // Mean should be [2.0, 3.0].
        assert!((stats.mean()[0] - 2.0).abs() < 1e-5);
        assert!((stats.mean()[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn grad_stats_variance() {
        let mut stats = OnlineGradientStats::new(1);
        // Values: 2, 4, 4, 4, 5, 5, 7, 9 → population variance = 4.0
        for &v in &[2.0_f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stats.update(&[v]).expect("ok");
        }
        let var = stats.variance()[0];
        assert!((var - 4.0).abs() < 0.1, "variance = {var}");
    }

    #[test]
    fn grad_stats_snr() {
        let mut stats = OnlineGradientStats::new(1);
        for &v in &[1.0_f32, 1.0, 1.0, 1.0] {
            stats.update(&[v]).expect("ok");
        }
        // mean=1, std=0 → snr = 1/(0+1e-8) which is large.
        let snr = stats.snr()[0];
        assert!(snr > 1e6, "snr = {snr}");
    }

    #[test]
    fn grad_stats_effective_gradient() {
        let mut stats = OnlineGradientStats::new(2);
        for &v in &[2.0_f32, 2.0, 2.0] {
            stats.update(&[v, -v]).expect("ok");
        }
        let eg = stats.effective_gradient();
        // mean=[2,−2], std≈0 → eff_grad large positive/negative
        assert!(eg[0] > 0.0);
        assert!(eg[1] < 0.0);
    }

    #[test]
    fn grad_stats_reset() {
        let mut stats = OnlineGradientStats::new(2);
        stats.update(&[5.0, 10.0]).expect("ok");
        stats.reset();
        assert_eq!(stats.count, 0);
        assert!(stats.mean().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn grad_stats_dimension_mismatch() {
        let mut stats = OnlineGradientStats::new(3);
        let result = stats.update(&[1.0, 2.0]);
        assert!(matches!(
            result,
            Err(OnlineLearningError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }

    #[test]
    fn grad_stats_mean_norm() {
        let mut stats = OnlineGradientStats::new(3);
        stats.update(&[3.0, 4.0, 0.0]).expect("ok");
        // mean=[3,4,0], norm=5
        let norm = stats.mean_norm();
        assert!((norm - 5.0).abs() < 1e-4, "norm = {norm}");
    }

    // ── AdaptiveLrState ───────────────────────────────────────────────────────

    #[test]
    fn adaptive_lr_new_valid() {
        let state = AdaptiveLrState::new(1e-3, 10, 0.5).expect("ok");
        assert!((state.lr() - 1e-3).abs() < 1e-10);
    }

    #[test]
    fn adaptive_lr_invalid_reduction_factor() {
        let result = AdaptiveLrState::new(1e-3, 10, 1.5);
        assert!(matches!(
            result,
            Err(OnlineLearningError::InvalidDecayRate { rate: _ })
        ));
    }

    #[test]
    fn adaptive_lr_improvement_resets_stall() {
        let mut state = AdaptiveLrState::new(1e-3, 3, 0.5).expect("ok");
        state.step(1.0);
        state.step(1.0); // stall_count = 1
        state.step(0.5); // improvement → stall_count = 0
        let lr = state.step(0.4); // improvement again
        assert!(
            (lr - 1e-3).abs() < 1e-10,
            "lr should not have changed: {lr}"
        );
    }

    #[test]
    fn adaptive_lr_stall_triggers_reduction() {
        let mut state = AdaptiveLrState::new(1e-3, 3, 0.5).expect("ok");
        // 3 non-improving steps trigger a reduction.
        state.step(1.0); // sets best_loss = 1.0, stall_count = 0
        let _lr1 = state.step(1.0); // stall_count = 1
        let _lr2 = state.step(1.0); // stall_count = 2
        let lr3 = state.step(1.0); // stall_count = 3 → reduction
        assert!(lr3 < 1e-3, "lr should have been reduced, got {lr3}");
        assert!((lr3 - 5e-4).abs() < 1e-10, "expected 5e-4, got {lr3}");
    }

    #[test]
    fn adaptive_lr_at_minimum() {
        let mut state = AdaptiveLrState::new(2e-6, 1, 0.5).expect("ok");
        state.step(1.0);
        state.step(1.0); // reduce: 2e-6 * 0.5 = 1e-6 = min_lr
        assert!(state.is_at_minimum(), "should be at minimum");
    }

    #[test]
    fn adaptive_lr_reset() {
        let mut state = AdaptiveLrState::new(1e-3, 2, 0.5).expect("ok");
        state.step(1.0);
        state.step(1.0);
        state.step(1.0); // triggers reduction
        state.reset();
        assert!((state.lr() - 1e-3).abs() < 1e-10);
    }

    // ── reservoir_sample ─────────────────────────────────────────────────────

    #[test]
    fn reservoir_sample_n_zero() {
        let items = vec![1, 2, 3, 4, 5];
        let sample = reservoir_sample(&items, 0, 42).expect("ok");
        assert!(sample.is_empty());
    }

    #[test]
    fn reservoir_sample_n_all() {
        let items: Vec<i32> = (0..10).collect();
        let mut sample = reservoir_sample(&items, 10, 42).expect("ok");
        sample.sort();
        assert_eq!(sample, items);
    }

    #[test]
    fn reservoir_sample_n_subset() {
        let items: Vec<i32> = (0..20).collect();
        let sample = reservoir_sample(&items, 5, 7).expect("ok");
        assert_eq!(sample.len(), 5);
        for v in &sample {
            assert!(items.contains(v));
        }
    }

    #[test]
    fn reservoir_sample_deterministic_same_seed() {
        let items: Vec<i32> = (0..50).collect();
        let s1 = reservoir_sample(&items, 10, 123).expect("ok");
        let s2 = reservoir_sample(&items, 10, 123).expect("ok");
        assert_eq!(s1, s2);
    }

    #[test]
    fn reservoir_sample_too_large() {
        let items = vec![1, 2, 3];
        let result = reservoir_sample(&items, 5, 1);
        assert!(matches!(
            result,
            Err(OnlineLearningError::SampleTooLarge {
                requested: 5,
                available: 3
            })
        ));
    }

    // ── ema_update ───────────────────────────────────────────────────────────

    #[test]
    fn ema_update_decay_zero() {
        // decay=0 → result = new_value entirely.
        // Note: decay=0 is technically out of (0,1) for OnlineLossHistory,
        // but the free function has no restriction.
        let result = ema_update(100.0, 5.0, 0.0);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn ema_update_decay_one() {
        // decay=1 → result = old EMA entirely.
        let result = ema_update(100.0, 5.0, 1.0);
        assert!((result - 100.0).abs() < 1e-6);
    }

    #[test]
    fn ema_update_typical() {
        let result = ema_update(10.0, 0.0, 0.9);
        assert!((result - 9.0).abs() < 1e-5, "result = {result}");
    }

    // ── linear_regression ────────────────────────────────────────────────────

    #[test]
    fn linear_regression_known_slope() {
        // y = 2x + 1 for x = 0..5
        let values: Vec<f32> = (0..5).map(|i| 2.0 * i as f32 + 1.0).collect();
        let (slope, intercept) = linear_regression(&values);
        assert!((slope - 2.0).abs() < 1e-4, "slope = {slope}");
        assert!((intercept - 1.0).abs() < 1e-4, "intercept = {intercept}");
    }

    #[test]
    fn linear_regression_horizontal() {
        let values = vec![3.0_f32; 10];
        let (slope, _intercept) = linear_regression(&values);
        assert!(slope.abs() < 1e-5, "slope should be ~0, got {slope}");
    }

    #[test]
    fn linear_regression_single_point() {
        let values = vec![7.0_f32];
        let (slope, intercept) = linear_regression(&values);
        assert!((slope - 0.0).abs() < 1e-6);
        assert!((intercept - 7.0).abs() < 1e-6);
    }

    // ── detect_stall ─────────────────────────────────────────────────────────

    #[test]
    fn detect_stall_is_stalled() {
        let history = vec![1.0_f32; 20];
        assert!(detect_stall(&history, 10, 0.01));
    }

    #[test]
    fn detect_stall_not_stalled() {
        let history: Vec<f32> = (0..20).map(|i| i as f32).collect();
        assert!(!detect_stall(&history, 10, 0.5));
    }

    // ── welford_update ───────────────────────────────────────────────────────

    #[test]
    fn welford_update_incremental_matches_batch() {
        let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let mut mean = 0.0_f32;
        let mut m2 = 0.0_f32;
        for (i, &v) in values.iter().enumerate() {
            let (nm, nm2) = welford_update(i + 1, mean, m2, v);
            mean = nm;
            m2 = nm2;
        }
        // Batch mean = 3.0
        assert!((mean - 3.0).abs() < 1e-5, "mean = {mean}");
        // Population variance = M2/n = 2.0
        let var = m2 / values.len() as f32;
        assert!((var - 2.0).abs() < 1e-5, "variance = {var}");
    }

    #[test]
    fn welford_update_single_step() {
        // count=1, old mean=0, m2=0, new_value=5 → mean=5, m2=0
        let (new_mean, new_m2) = welford_update(1, 0.0, 0.0, 5.0);
        assert!((new_mean - 5.0).abs() < 1e-6);
        assert!(new_m2.abs() < 1e-6);
    }

    // ── weighted_sample_indices ───────────────────────────────────────────────

    #[test]
    fn weighted_sample_uniform_when_equal() {
        let weights = vec![1.0_f32; 5];
        let indices = weighted_sample_indices(&weights, 5, 42).expect("ok");
        assert_eq!(indices.len(), 5);
        // All should be in [0, 5).
        for &i in &indices {
            assert!(i < 5, "index out of range: {i}");
        }
    }

    #[test]
    fn weighted_sample_skewed() {
        // Weight 0: 0, weight 4: very large → most samples should be index 4.
        let weights = vec![0.0_f32, 0.0, 0.0, 0.0, 1000.0];
        let indices = weighted_sample_indices(&weights, 100, 7).expect("ok");
        let count_4 = indices.iter().filter(|&&i| i == 4).count();
        assert!(count_4 > 90, "expected mostly index 4, got {count_4}/100");
    }

    // ── compute_importance_weights ────────────────────────────────────────────

    #[test]
    fn importance_weights_beta_zero_uniform() {
        let losses = vec![1.0_f32, 2.0, 3.0, 0.5];
        let weights = compute_importance_weights(&losses, 0.0);
        // All should be 1.0 for beta=0.
        assert!(weights.iter().all(|&w| (w - 1.0).abs() < 1e-6));
    }

    #[test]
    fn importance_weights_beta_one_max_normalized() {
        let losses = vec![0.1_f32, 1.0, 10.0];
        let weights = compute_importance_weights(&losses, 1.0);
        // Max weight should be 1.0.
        let max_w = weights.iter().cloned().fold(0.0_f32, f32::max);
        assert!((max_w - 1.0).abs() < 1e-5, "max weight = {max_w}");
        // Larger loss → smaller weight (lower priority correction needed).
        assert!(weights[0] >= weights[1], "higher loss = lower weight");
        assert!(weights[1] >= weights[2], "higher loss = lower weight");
    }
}
