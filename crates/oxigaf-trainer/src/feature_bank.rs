//! Momentum feature bank for contrastive self-supervised learning.
//!
//! A momentum feature bank stores a running collection of feature vectors
//! (embeddings) from recent training iterations. Used in self-supervised
//! learning methods like MoCo (Momentum Contrast) to provide a large set of
//! negative samples for contrastive learning without requiring large batch
//! sizes.
//!
//! # Architecture
//!
//! The bank is a circular FIFO queue of [`FeatureEntry`] items. When the bank
//! is full and `fifo_eviction` is enabled, the oldest entry is silently
//! replaced by the newest insertion. If `fifo_eviction` is `false`, insertion
//! into a full bank returns [`FeatureBankError::BankFull`].
//!
//! A companion [`MomentumEncoder`] maintains an Exponential Moving Average
//! (EMA) of encoder weights, mirroring the MoCo design.
//!
//! # Example
//!
//! ```rust
//! use oxigaf_trainer::feature_bank::{BankConfig, FeatureBank, FeatureBankError};
//!
//! let config = BankConfig { capacity: 256, feature_dim: 4, ..Default::default() };
//! let mut bank = FeatureBank::new(config).unwrap();
//! bank.push(vec![1.0, 0.0, 0.0, 0.0], 1, 0).unwrap();
//! let sims = bank.query_similarity(&[1.0, 0.0, 0.0, 0.0]).unwrap();
//! assert_eq!(sims.len(), 1);
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by feature bank operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum FeatureBankError {
    /// The bank contains no entries and at least one is required.
    #[error("Empty bank: at least one entry is required")]
    EmptyBank,

    /// Two vectors that should have the same dimension do not.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// A configuration parameter is out of range or otherwise invalid.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// The bank is full and FIFO eviction is disabled.
    #[error("Bank is full (capacity {0}) and FIFO eviction is disabled")]
    BankFull(usize),

    /// A numerical error was encountered (e.g., NaN, insufficient negatives).
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// FeatureEntry
// ─────────────────────────────────────────────────────────────────────────────

/// A single entry stored in the feature bank.
#[derive(Debug, Clone)]
pub struct FeatureEntry {
    /// Feature vector (embedding).
    pub features: Vec<f32>,
    /// Class label (optional, 0 = unknown).
    pub label: u32,
    /// Step at which this entry was inserted.
    pub step: usize,
    /// Insertion index (monotonically increasing, set by the bank on push).
    pub insertion_idx: u64,
}

impl FeatureEntry {
    /// Create a new feature entry.
    ///
    /// `insertion_idx` is initialised to 0; the [`FeatureBank`] overwrites it
    /// with the correct monotonic value on insertion.
    pub fn new(features: Vec<f32>, label: u32, step: usize) -> Self {
        Self {
            features,
            label,
            step,
            insertion_idx: 0,
        }
    }

    /// Feature dimension (length of the feature vector).
    pub fn dim(&self) -> usize {
        self.features.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MomentumEncoder
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks momentum-updated encoder weights as flat `f32` vectors.
///
/// The momentum encoder maintains a "shadow" copy of the online encoder weights
/// that is updated more slowly using an Exponential Moving Average:
///
/// ```text
/// m_weights = momentum * m_weights + (1 - momentum) * online_weights
/// ```
///
/// High momentum values (e.g., 0.999) yield slow, stable updates.
pub struct MomentumEncoder {
    /// Online encoder weights.
    pub online_weights: Vec<f32>,
    /// Momentum encoder weights (shadow copy, updated more slowly).
    pub momentum_weights: Vec<f32>,
    /// Momentum coefficient. Must be in `[0, 1)`. High = slow updates.
    pub momentum: f32,
    /// Number of update steps applied.
    pub step: usize,
}

impl MomentumEncoder {
    /// Create a new momentum encoder from the given initial weights.
    ///
    /// `momentum` must be in `[0, 1)`. Typical value: `0.999`.
    ///
    /// # Errors
    ///
    /// Returns [`FeatureBankError::InvalidConfig`] if `momentum` is not in
    /// `[0, 1)`.
    pub fn new(initial_weights: Vec<f32>, momentum: f32) -> Result<Self, FeatureBankError> {
        if !(0.0..1.0).contains(&momentum) {
            return Err(FeatureBankError::InvalidConfig(format!(
                "momentum must be in [0, 1), got {momentum}"
            )));
        }
        let momentum_weights = initial_weights.clone();
        Ok(Self {
            online_weights: initial_weights,
            momentum_weights,
            momentum,
            step: 0,
        })
    }

    /// Update the momentum encoder via element-wise EMA from `online_weights`.
    ///
    /// ```text
    /// m_weights[i] = momentum * m_weights[i] + (1 - momentum) * online_weights[i]
    /// ```
    pub fn update(&mut self) {
        let one_minus_m = 1.0 - self.momentum;
        for (m, &o) in self
            .momentum_weights
            .iter_mut()
            .zip(self.online_weights.iter())
        {
            *m = self.momentum * *m + one_minus_m * o;
        }
        self.step += 1;
    }

    /// Reset momentum encoder to match online encoder state exactly.
    ///
    /// Sets `momentum_weights = online_weights.clone()`.
    pub fn reset_momentum(&mut self) {
        self.momentum_weights = self.online_weights.clone();
    }

    /// L2 distance between online and momentum weights (diagnostic).
    ///
    /// Returns `0.0` if both weight vectors are empty.
    pub fn weight_distance(&self) -> f32 {
        self.online_weights
            .iter()
            .zip(self.momentum_weights.iter())
            .map(|(&o, &m)| (o - m) * (o - m))
            .sum::<f32>()
            .sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BankConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a [`FeatureBank`].
#[derive(Debug, Clone)]
pub struct BankConfig {
    /// Maximum number of feature entries stored.
    pub capacity: usize,
    /// Feature vector dimension. All inserted features must match this.
    pub feature_dim: usize,
    /// If `true`, when the bank is full the oldest entry is evicted (FIFO).
    /// If `false`, insertion into a full bank returns [`FeatureBankError::BankFull`].
    pub fifo_eviction: bool,
    /// Temperature used when computing cosine similarities.
    pub temperature: f32,
}

impl Default for BankConfig {
    fn default() -> Self {
        Self {
            capacity: 4096,
            feature_dim: 128,
            fifo_eviction: true,
            temperature: 0.07,
        }
    }
}

impl BankConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`FeatureBankError::InvalidConfig`] if any parameter is invalid.
    pub fn validate(&self) -> Result<(), FeatureBankError> {
        if self.capacity < 1 {
            return Err(FeatureBankError::InvalidConfig(
                "capacity must be >= 1".to_string(),
            ));
        }
        if self.feature_dim < 1 {
            return Err(FeatureBankError::InvalidConfig(
                "feature_dim must be >= 1".to_string(),
            ));
        }
        if self.temperature <= 0.0 {
            return Err(FeatureBankError::InvalidConfig(format!(
                "temperature must be > 0, got {}",
                self.temperature
            )));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FeatureBank
// ─────────────────────────────────────────────────────────────────────────────

/// Circular momentum feature bank for contrastive learning.
///
/// Maintains a fixed-capacity FIFO queue of [`FeatureEntry`] items. When the
/// bank is full and `fifo_eviction` is enabled the oldest entry is replaced;
/// otherwise [`FeatureBankError::BankFull`] is returned.
pub struct FeatureBank {
    config: BankConfig,
    /// Stored entries in circular buffer order.
    entries: Vec<FeatureEntry>,
    /// Write head: index of the next slot to be written.
    write_head: usize,
    /// Total number of insertions ever performed.
    total_insertions: u64,
    /// Whether the bank has wrapped around (reached capacity at least once).
    is_full: bool,
}

impl FeatureBank {
    /// Create a new, empty feature bank from the given config.
    ///
    /// # Errors
    ///
    /// Returns [`FeatureBankError::InvalidConfig`] if the config fails
    /// validation.
    pub fn new(config: BankConfig) -> Result<Self, FeatureBankError> {
        config.validate()?;
        Ok(Self {
            entries: Vec::with_capacity(config.capacity),
            write_head: 0,
            total_insertions: 0,
            is_full: false,
            config,
        })
    }

    /// Insert a feature vector into the bank.
    ///
    /// If the bank is full:
    /// - When `fifo_eviction` is enabled the oldest entry (at `write_head`) is
    ///   overwritten.
    /// - When `fifo_eviction` is disabled [`FeatureBankError::BankFull`] is
    ///   returned.
    ///
    /// # Errors
    ///
    /// - [`FeatureBankError::DimensionMismatch`] if `features.len()` differs
    ///   from `config.feature_dim`.
    /// - [`FeatureBankError::BankFull`] if the bank is full and eviction is
    ///   disabled.
    pub fn push(
        &mut self,
        features: Vec<f32>,
        label: u32,
        step: usize,
    ) -> Result<(), FeatureBankError> {
        if features.len() != self.config.feature_dim {
            return Err(FeatureBankError::DimensionMismatch {
                expected: self.config.feature_dim,
                got: features.len(),
            });
        }

        if self.is_full && !self.config.fifo_eviction {
            return Err(FeatureBankError::BankFull(self.config.capacity));
        }

        let insertion_idx = self.total_insertions;
        self.total_insertions += 1;

        let entry = FeatureEntry {
            features,
            label,
            step,
            insertion_idx,
        };

        if self.is_full {
            // Overwrite the slot at write_head (the oldest entry).
            self.entries[self.write_head] = entry;
        } else {
            // Bank not yet full: append.
            self.entries.push(entry);
            if self.entries.len() == self.config.capacity {
                self.is_full = true;
            }
        }

        // Advance write head.
        self.write_head = (self.write_head + 1) % self.config.capacity;

        Ok(())
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the bank contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Maximum number of entries that can be stored.
    pub fn capacity(&self) -> usize {
        self.config.capacity
    }

    /// All stored entries in insertion order (oldest first).
    ///
    /// When the bank has wrapped around, the oldest entry is at `write_head`
    /// (which is the next slot to be overwritten).
    pub fn all_features(&self) -> Vec<&FeatureEntry> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        let n = self.entries.len();
        let mut result = Vec::with_capacity(n);

        if self.is_full {
            // After wrap-around: oldest is at write_head.
            for i in 0..n {
                let idx = (self.write_head + i) % self.config.capacity;
                result.push(&self.entries[idx]);
            }
        } else {
            // Not full yet: entries[0..n] in insertion order.
            for entry in &self.entries {
                result.push(entry);
            }
        }

        result
    }

    /// Get all entries with the specified label.
    ///
    /// If `label` is `0` all entries are returned (label 0 = unknown/wildcard).
    pub fn features_by_label(&self, label: u32) -> Vec<&FeatureEntry> {
        if label == 0 {
            return self.all_features();
        }
        self.all_features()
            .into_iter()
            .filter(|e| e.label == label)
            .collect()
    }

    /// Compute cosine similarities between `query` and all bank entries.
    ///
    /// Returns a `Vec<(similarity, entry_index)>` sorted by similarity
    /// **descending**. Similarity is divided by the configured temperature.
    ///
    /// # Errors
    ///
    /// - [`FeatureBankError::EmptyBank`] if the bank is empty.
    /// - [`FeatureBankError::DimensionMismatch`] if `query.len()` does not
    ///   match `feature_dim`.
    pub fn query_similarity(&self, query: &[f32]) -> Result<Vec<(f32, usize)>, FeatureBankError> {
        if self.is_empty() {
            return Err(FeatureBankError::EmptyBank);
        }
        if query.len() != self.config.feature_dim {
            return Err(FeatureBankError::DimensionMismatch {
                expected: self.config.feature_dim,
                got: query.len(),
            });
        }

        let q_norm = l2_norm(query);
        let q_normalized: Vec<f32> = if q_norm < 1e-8 {
            vec![0.0; query.len()]
        } else {
            query.iter().map(|x| x / q_norm).collect()
        };

        let mut results = Vec::with_capacity(self.entries.len());
        for (idx, entry) in self.entries.iter().enumerate() {
            let e_norm = l2_norm(&entry.features);
            let dot = if e_norm < 1e-8 {
                0.0_f32
            } else {
                q_normalized
                    .iter()
                    .zip(entry.features.iter())
                    .map(|(&q, &e)| q * (e / e_norm))
                    .sum::<f32>()
            };
            let sim = dot / self.config.temperature;
            results.push((sim, idx));
        }

        // Sort descending by similarity.
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Return the top-`k` most similar entries to `query`.
    ///
    /// If `k` exceeds the number of stored entries, all entries are returned.
    ///
    /// # Errors
    ///
    /// Same as [`query_similarity`](Self::query_similarity).
    pub fn top_k_similar(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(f32, usize)>, FeatureBankError> {
        let mut sims = self.query_similarity(query)?;
        sims.truncate(k.min(sims.len()));
        Ok(sims)
    }

    /// Compute the InfoNCE loss of `query` against all bank entries.
    ///
    /// `positive_idx` is the bank-storage index (into `self.entries`) of the
    /// positive sample. All other entries serve as negatives.
    ///
    /// ```text
    /// loss = -log( exp(sim[positive] / τ) / Σ_i exp(sim[i] / τ) )
    ///      = -log_softmax[positive]
    /// ```
    ///
    /// where `sim[i]` is the cosine similarity between `query` and entry `i`,
    /// and `τ` is the configured temperature.
    ///
    /// # Errors
    ///
    /// - All errors from [`query_similarity`](Self::query_similarity).
    /// - [`FeatureBankError::NumericalError`] if the bank has only one entry
    ///   (no negatives).
    /// - [`FeatureBankError::DimensionMismatch`] if `positive_idx` is out of
    ///   range.
    pub fn infonce_loss(
        &self,
        query: &[f32],
        positive_idx: usize,
    ) -> Result<f32, FeatureBankError> {
        if self.len() < 2 {
            return Err(FeatureBankError::NumericalError(
                "InfoNCE requires at least 2 entries (1 positive + 1 negative)".to_string(),
            ));
        }
        if positive_idx >= self.entries.len() {
            return Err(FeatureBankError::DimensionMismatch {
                expected: self.entries.len() - 1,
                got: positive_idx,
            });
        }

        if query.len() != self.config.feature_dim {
            return Err(FeatureBankError::DimensionMismatch {
                expected: self.config.feature_dim,
                got: query.len(),
            });
        }

        // Compute raw cosine similarities (scaled by temperature) in bank order.
        let q_norm = l2_norm(query);
        let q_normalized: Vec<f32> = if q_norm < 1e-8 {
            vec![0.0; query.len()]
        } else {
            query.iter().map(|x| x / q_norm).collect()
        };

        let mut logits: Vec<f32> = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            let e_norm = l2_norm(&entry.features);
            let dot = if e_norm < 1e-8 {
                0.0_f32
            } else {
                q_normalized
                    .iter()
                    .zip(entry.features.iter())
                    .map(|(&q, &e)| q * (e / e_norm))
                    .sum::<f32>()
            };
            logits.push(dot / self.config.temperature);
        }

        // Numerically stable log-sum-exp.
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp = logits
            .iter()
            .map(|&l| (l - max_logit).exp())
            .sum::<f32>()
            .ln()
            + max_logit;

        let loss = -(logits[positive_idx] - log_sum_exp);
        Ok(loss)
    }

    /// Clear all entries from the bank.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.write_head = 0;
        self.is_full = false;
    }

    /// Compute age statistics for all stored entries.
    ///
    /// Returns `(mean_age, max_age)` where age = `current_step - entry.step`.
    /// Returns `(0.0, 0)` if the bank is empty.
    pub fn age_stats(&self, current_step: usize) -> (f32, usize) {
        if self.entries.is_empty() {
            return (0.0, 0);
        }
        let mut total: usize = 0;
        let mut max_age: usize = 0;
        for entry in &self.entries {
            let age = current_step.saturating_sub(entry.step);
            total += age;
            if age > max_age {
                max_age = age;
            }
        }
        let mean = total as f32 / self.entries.len() as f32;
        (mean, max_age)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BankStatistics
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics over a [`FeatureBank`] snapshot.
#[derive(Debug, Clone)]
pub struct BankStatistics {
    /// Number of entries currently stored.
    pub num_entries: usize,
    /// Maximum capacity of the bank.
    pub capacity: usize,
    /// Fraction of capacity used (`num_entries / capacity`).
    pub fill_fraction: f32,
    /// Number of distinct labels among all entries (excluding label 0).
    pub num_labels: usize,
    /// Mean L2 norm of stored feature vectors.
    pub mean_feature_norm: f32,
    /// Smallest step value among all entries.
    pub oldest_step: usize,
    /// Largest step value among all entries.
    pub newest_step: usize,
}

/// Compute aggregate statistics from a [`FeatureBank`].
pub fn compute_bank_stats(bank: &FeatureBank) -> BankStatistics {
    let num_entries = bank.len();
    let capacity = bank.capacity();

    let fill_fraction = if capacity == 0 {
        0.0
    } else {
        num_entries as f32 / capacity as f32
    };

    if num_entries == 0 {
        return BankStatistics {
            num_entries: 0,
            capacity,
            fill_fraction,
            num_labels: 0,
            mean_feature_norm: 0.0,
            oldest_step: 0,
            newest_step: 0,
        };
    }

    let mut labels = std::collections::HashSet::new();
    let mut total_norm = 0.0_f32;
    let mut oldest_step = usize::MAX;
    let mut newest_step = 0_usize;

    for entry in &bank.entries {
        if entry.label != 0 {
            labels.insert(entry.label);
        }
        total_norm += l2_norm(&entry.features);
        if entry.step < oldest_step {
            oldest_step = entry.step;
        }
        if entry.step > newest_step {
            newest_step = entry.step;
        }
    }

    BankStatistics {
        num_entries,
        capacity,
        fill_fraction,
        num_labels: labels.len(),
        mean_feature_norm: total_norm / num_entries as f32,
        oldest_step: if oldest_step == usize::MAX {
            0
        } else {
            oldest_step
        },
        newest_step,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Negative / positive sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Inline xorshift64 PRNG. Advances `state` in place and returns a pseudo-random `u64`.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Sample up to `n` negative indices for a contrastive query.
///
/// Negatives are entries whose label differs from `label`. If `label` is `0`
/// all entries are eligible (since label 0 means "unknown"). Sampling is done
/// with a Fisher-Yates partial shuffle seeded by `seed`.
///
/// If `n >= available`, all available indices are returned (no duplication).
///
/// # Errors
///
/// Returns [`FeatureBankError::EmptyBank`] if there are no eligible negatives.
pub fn sample_negatives(
    bank: &FeatureBank,
    label: u32,
    n: usize,
    seed: u64,
) -> Result<Vec<usize>, FeatureBankError> {
    // Collect all indices whose label differs from query label.
    let candidates: Vec<usize> = bank
        .entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.label != label)
        .map(|(i, _)| i)
        .collect();

    if candidates.is_empty() {
        return Err(FeatureBankError::EmptyBank);
    }

    if n == 0 {
        return Ok(Vec::new());
    }

    let take = n.min(candidates.len());
    Ok(reservoir_sample(&candidates, take, seed))
}

/// Sample a single positive entry (same label) from the bank.
///
/// Returns the bank-storage index of the sampled positive entry.
///
/// # Errors
///
/// Returns [`FeatureBankError::EmptyBank`] if no entry with the specified
/// label exists.
pub fn sample_positive(
    bank: &FeatureBank,
    label: u32,
    seed: u64,
) -> Result<usize, FeatureBankError> {
    let candidates: Vec<usize> = bank
        .entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.label == label)
        .map(|(i, _)| i)
        .collect();

    if candidates.is_empty() {
        return Err(FeatureBankError::EmptyBank);
    }

    let sampled = reservoir_sample(&candidates, 1, seed);
    // reservoir_sample always returns exactly 1 element when take=1 and candidates non-empty.
    sampled
        .into_iter()
        .next()
        .ok_or(FeatureBankError::EmptyBank)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the L2 norm of a slice.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Reservoir-sample `take` items from `pool` using xorshift64 with the given seed.
///
/// Uses a partial Fisher-Yates shuffle on a local copy so that no allocation
/// beyond the result is required.
fn reservoir_sample(pool: &[usize], take: usize, seed: u64) -> Vec<usize> {
    if pool.is_empty() || take == 0 {
        return Vec::new();
    }

    let take = take.min(pool.len());
    let mut buf: Vec<usize> = pool.to_vec();
    let mut state = if seed == 0 { 0xdeadbeef_cafebabe } else { seed };

    for i in 0..take {
        let rand = xorshift64(&mut state);
        let j = i + (rand as usize % (buf.len() - i));
        buf.swap(i, j);
    }

    buf[..take].to_vec()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn unit_vec(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        if hot < dim {
            v[hot] = 1.0;
        }
        v
    }

    fn make_bank(capacity: usize, dim: usize) -> FeatureBank {
        FeatureBank::new(BankConfig {
            capacity,
            feature_dim: dim,
            fifo_eviction: true,
            temperature: 1.0, // temperature=1 keeps raw cosine values
        })
        .unwrap_or_else(|e| panic!("make_bank failed: {e}"))
    }

    // ── FeatureEntry ─────────────────────────────────────────────────────────

    #[test]
    fn feature_entry_new_fields() {
        let f = vec![1.0_f32, 2.0, 3.0];
        let entry = FeatureEntry::new(f.clone(), 7, 42);
        assert_eq!(entry.features, f);
        assert_eq!(entry.label, 7);
        assert_eq!(entry.step, 42);
        assert_eq!(entry.insertion_idx, 0); // set by bank on push
    }

    #[test]
    fn feature_entry_dim() {
        let entry = FeatureEntry::new(vec![0.0; 128], 0, 0);
        assert_eq!(entry.dim(), 128);
    }

    // ── MomentumEncoder ──────────────────────────────────────────────────────

    #[test]
    fn momentum_encoder_invalid_momentum_ge_1() {
        let err = MomentumEncoder::new(vec![1.0, 2.0], 1.0);
        assert!(matches!(err, Err(FeatureBankError::InvalidConfig(_))));
    }

    #[test]
    fn momentum_encoder_invalid_momentum_negative() {
        let err = MomentumEncoder::new(vec![1.0], -0.1);
        assert!(matches!(err, Err(FeatureBankError::InvalidConfig(_))));
    }

    #[test]
    fn momentum_encoder_valid_momentum_zero() {
        // momentum=0 is valid (immediate copy semantics)
        let enc = MomentumEncoder::new(vec![1.0, 2.0], 0.0);
        assert!(enc.is_ok());
    }

    #[test]
    fn momentum_encoder_weight_distance_zero_initially() {
        let enc = MomentumEncoder::new(vec![1.0, 2.0, 3.0], 0.9).unwrap_or_else(|e| panic!("{e}"));
        assert!((enc.weight_distance() - 0.0).abs() < 1e-7);
    }

    #[test]
    fn momentum_encoder_update_moves_toward_online() {
        let mut enc = MomentumEncoder::new(vec![0.0_f32; 4], 0.9).unwrap_or_else(|e| panic!("{e}"));
        // Change online weights to 1.0 everywhere.
        for w in enc.online_weights.iter_mut() {
            *w = 1.0;
        }
        enc.update();
        // After one update with momentum=0.9: m = 0.9*0 + 0.1*1 = 0.1
        for &m in &enc.momentum_weights {
            assert!((m - 0.1).abs() < 1e-6, "expected 0.1 got {m}");
        }
    }

    #[test]
    fn momentum_encoder_update_momentum_zero_immediate_copy() {
        let mut enc = MomentumEncoder::new(vec![0.0_f32; 3], 0.0).unwrap_or_else(|e| panic!("{e}"));
        enc.online_weights = vec![5.0, 6.0, 7.0];
        enc.update();
        // momentum=0 → m = 0*old + 1*online = online
        assert_eq!(enc.momentum_weights, vec![5.0, 6.0, 7.0]);
    }

    #[test]
    fn momentum_encoder_reset_copies_online_to_momentum() {
        let mut enc =
            MomentumEncoder::new(vec![1.0_f32; 4], 0.99).unwrap_or_else(|e| panic!("{e}"));
        enc.online_weights = vec![9.0; 4];
        enc.reset_momentum();
        assert_eq!(enc.momentum_weights, vec![9.0; 4]);
        assert!((enc.weight_distance() - 0.0).abs() < 1e-7);
    }

    // ── BankConfig ───────────────────────────────────────────────────────────

    #[test]
    fn bank_config_validate_valid() {
        let cfg = BankConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn bank_config_validate_zero_capacity() {
        let cfg = BankConfig {
            capacity: 0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(FeatureBankError::InvalidConfig(_))
        ));
    }

    #[test]
    fn bank_config_validate_zero_feature_dim() {
        let cfg = BankConfig {
            feature_dim: 0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(FeatureBankError::InvalidConfig(_))
        ));
    }

    #[test]
    fn bank_config_validate_zero_temperature() {
        let cfg = BankConfig {
            temperature: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(FeatureBankError::InvalidConfig(_))
        ));
    }

    #[test]
    fn bank_config_validate_negative_temperature() {
        let cfg = BankConfig {
            temperature: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(FeatureBankError::InvalidConfig(_))
        ));
    }

    // ── FeatureBank basic ─────────────────────────────────────────────────────

    #[test]
    fn feature_bank_new_empty_len_zero() {
        let bank = make_bank(64, 4);
        assert_eq!(bank.len(), 0);
    }

    #[test]
    fn feature_bank_is_empty_true_initially() {
        let bank = make_bank(8, 3);
        assert!(bank.is_empty());
    }

    #[test]
    fn feature_bank_push_correct_len() {
        let mut bank = make_bank(8, 3);
        bank.push(vec![1.0, 0.0, 0.0], 1, 0)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(bank.len(), 1);
        bank.push(vec![0.0, 1.0, 0.0], 2, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(bank.len(), 2);
    }

    #[test]
    fn feature_bank_push_dimension_mismatch() {
        let mut bank = make_bank(8, 4);
        let err = bank.push(vec![1.0, 0.0], 1, 0);
        assert!(matches!(
            err,
            Err(FeatureBankError::DimensionMismatch {
                expected: 4,
                got: 2
            })
        ));
    }

    #[test]
    fn feature_bank_push_fills_to_capacity_no_error() {
        let mut bank = make_bank(4, 2);
        for i in 0..4 {
            bank.push(vec![i as f32, 0.0], 0, i)
                .unwrap_or_else(|e| panic!("{e}"));
        }
        assert_eq!(bank.len(), 4);
    }

    #[test]
    fn feature_bank_push_beyond_capacity_fifo_len_stays() {
        let mut bank = make_bank(4, 2);
        for i in 0..8 {
            bank.push(vec![i as f32, 0.0], 0, i)
                .unwrap_or_else(|e| panic!("{e}"));
        }
        assert_eq!(bank.len(), 4);
    }

    #[test]
    fn feature_bank_push_no_fifo_returns_bank_full() {
        let mut bank = FeatureBank::new(BankConfig {
            capacity: 2,
            feature_dim: 2,
            fifo_eviction: false,
            temperature: 0.07,
        })
        .unwrap_or_else(|e| panic!("{e}"));
        bank.push(vec![1.0, 0.0], 1, 0)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(vec![0.0, 1.0], 2, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        let err = bank.push(vec![0.5, 0.5], 3, 2);
        assert!(matches!(err, Err(FeatureBankError::BankFull(_))));
    }

    #[test]
    fn feature_bank_all_features_insertion_order() {
        // Verify oldest-first after wrap-around.
        let mut bank = make_bank(3, 1);
        for i in 0..5_usize {
            bank.push(vec![i as f32], 0, i)
                .unwrap_or_else(|e| panic!("{e}"));
        }
        // After 5 pushes into capacity-3 FIFO, entries should be [2, 3, 4].
        let all = bank.all_features();
        assert_eq!(all.len(), 3);
        let steps: Vec<usize> = all.iter().map(|e| e.step).collect();
        assert_eq!(steps, vec![2, 3, 4], "expected steps [2,3,4] got {steps:?}");
    }

    #[test]
    fn feature_bank_features_by_label_filters_correctly() {
        let mut bank = make_bank(8, 2);
        bank.push(vec![1.0, 0.0], 1, 0)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(vec![0.0, 1.0], 2, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(vec![1.0, 1.0], 1, 2)
            .unwrap_or_else(|e| panic!("{e}"));

        let label1 = bank.features_by_label(1);
        assert_eq!(label1.len(), 2);
        for e in &label1 {
            assert_eq!(e.label, 1);
        }

        let label2 = bank.features_by_label(2);
        assert_eq!(label2.len(), 1);
    }

    // ── query_similarity ─────────────────────────────────────────────────────

    #[test]
    fn query_similarity_empty_bank_error() {
        let bank = make_bank(8, 4);
        let err = bank.query_similarity(&[1.0, 0.0, 0.0, 0.0]);
        assert!(matches!(err, Err(FeatureBankError::EmptyBank)));
    }

    #[test]
    fn query_similarity_dimension_mismatch() {
        let mut bank = make_bank(8, 4);
        bank.push(unit_vec(4, 0), 1, 0)
            .unwrap_or_else(|e| panic!("{e}"));
        let err = bank.query_similarity(&[1.0, 0.0]);
        assert!(matches!(
            err,
            Err(FeatureBankError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn query_similarity_returns_sorted_descending() {
        let mut bank = make_bank(8, 3);
        bank.push(unit_vec(3, 0), 1, 0)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(unit_vec(3, 1), 2, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(unit_vec(3, 2), 3, 2)
            .unwrap_or_else(|e| panic!("{e}"));

        // Query close to e_1 direction.
        let query = vec![0.9_f32, 0.1, 0.0];
        let sims = bank
            .query_similarity(&query)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(sims.len(), 3);
        for w in sims.windows(2) {
            assert!(w[0].0 >= w[1].0, "not sorted descending: {:?}", sims);
        }
    }

    #[test]
    fn query_similarity_self_query_is_highest() {
        let mut bank = make_bank(8, 4);
        let target = unit_vec(4, 2);
        bank.push(unit_vec(4, 0), 1, 0)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(unit_vec(4, 1), 2, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(target.clone(), 3, 2)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(unit_vec(4, 3), 4, 3)
            .unwrap_or_else(|e| panic!("{e}"));

        let sims = bank
            .query_similarity(&target)
            .unwrap_or_else(|e| panic!("{e}"));
        // The highest similarity should correspond to the target entry (bank index 2).
        let (_, top_idx) = sims[0];
        assert_eq!(
            top_idx, 2,
            "self-query should be most similar, got idx {top_idx}"
        );
    }

    // ── top_k_similar ────────────────────────────────────────────────────────

    #[test]
    fn top_k_similar_returns_at_most_k() {
        let mut bank = make_bank(16, 2);
        for i in 0..10_usize {
            bank.push(vec![i as f32, 0.0], 0, i)
                .unwrap_or_else(|e| panic!("{e}"));
        }
        let top3 = bank
            .top_k_similar(&[1.0, 0.0], 3)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(top3.len(), 3);
    }

    #[test]
    fn top_k_similar_k_exceeds_len_returns_all() {
        let mut bank = make_bank(16, 2);
        bank.push(vec![1.0, 0.0], 1, 0)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(vec![0.0, 1.0], 2, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        let top10 = bank
            .top_k_similar(&[1.0, 0.0], 10)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(top10.len(), 2);
    }

    // ── infonce_loss ─────────────────────────────────────────────────────────

    #[test]
    fn infonce_loss_single_entry_numerical_error() {
        let mut bank = make_bank(8, 3);
        bank.push(unit_vec(3, 0), 1, 0)
            .unwrap_or_else(|e| panic!("{e}"));
        let err = bank.infonce_loss(&unit_vec(3, 0), 0);
        assert!(matches!(err, Err(FeatureBankError::NumericalError(_))));
    }

    #[test]
    fn infonce_loss_perfect_match_low_loss() {
        // With temperature=1 and a perfect-match positive far from other entries,
        // the loss should be small (close to log(1/n) ≈ log(1/n) but positive_sim ≈ 1).
        let mut bank = FeatureBank::new(BankConfig {
            capacity: 16,
            feature_dim: 4,
            fifo_eviction: true,
            temperature: 0.1, // sharp temperature → positive dominates
        })
        .unwrap_or_else(|e| panic!("{e}"));

        // Positive: e0 direction. Negatives: orthogonal.
        bank.push(unit_vec(4, 0), 1, 0)
            .unwrap_or_else(|e| panic!("{e}")); // idx 0
        bank.push(unit_vec(4, 1), 2, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(unit_vec(4, 2), 3, 2)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(unit_vec(4, 3), 4, 3)
            .unwrap_or_else(|e| panic!("{e}"));

        let query = unit_vec(4, 0);
        let loss = bank
            .infonce_loss(&query, 0)
            .unwrap_or_else(|e| panic!("{e}"));
        // With temperature=0.1, positive logit = 1/0.1 = 10, negatives = 0/0.1 = 0.
        // softmax denominator ≈ exp(10) + 3*exp(0); loss ≈ -log(exp(10)/(exp(10)+3))
        // which is very small (≈ 0).
        assert!(
            loss < 0.1,
            "expected low loss for perfect match, got {loss}"
        );
    }

    // ── clear ────────────────────────────────────────────────────────────────

    #[test]
    fn feature_bank_clear_is_empty() {
        let mut bank = make_bank(8, 2);
        bank.push(vec![1.0, 0.0], 1, 0)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(vec![0.0, 1.0], 2, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.clear();
        assert!(bank.is_empty());
        assert_eq!(bank.len(), 0);
    }

    // ── age_stats ────────────────────────────────────────────────────────────

    #[test]
    fn feature_bank_age_stats_single_entry() {
        let mut bank = make_bank(8, 2);
        bank.push(vec![1.0, 0.0], 1, 10)
            .unwrap_or_else(|e| panic!("{e}"));
        let (mean, max) = bank.age_stats(15);
        assert!(
            (mean - 5.0).abs() < 1e-6,
            "mean age should be 5, got {mean}"
        );
        assert_eq!(max, 5);
    }

    #[test]
    fn feature_bank_age_stats_empty_bank() {
        let bank = make_bank(8, 2);
        let (mean, max) = bank.age_stats(100);
        assert_eq!(mean, 0.0);
        assert_eq!(max, 0);
    }

    // ── compute_bank_stats ───────────────────────────────────────────────────

    #[test]
    fn compute_bank_stats_fill_fraction() {
        let mut bank = make_bank(8, 2);
        bank.push(vec![1.0, 0.0], 1, 0)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(vec![0.0, 1.0], 2, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        let stats = compute_bank_stats(&bank);
        assert_eq!(stats.num_entries, 2);
        assert_eq!(stats.capacity, 8);
        assert!((stats.fill_fraction - 0.25).abs() < 1e-6);
    }

    #[test]
    fn compute_bank_stats_mean_feature_norm() {
        let mut bank = make_bank(8, 3);
        // All unit vectors → norms = 1.0.
        bank.push(unit_vec(3, 0), 1, 0)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(unit_vec(3, 1), 2, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        let stats = compute_bank_stats(&bank);
        assert!((stats.mean_feature_norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn compute_bank_stats_num_labels_excludes_zero() {
        let mut bank = make_bank(8, 2);
        bank.push(vec![1.0, 0.0], 0, 0)
            .unwrap_or_else(|e| panic!("{e}")); // unknown
        bank.push(vec![0.0, 1.0], 1, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(vec![1.0, 1.0], 1, 2)
            .unwrap_or_else(|e| panic!("{e}"));
        let stats = compute_bank_stats(&bank);
        assert_eq!(stats.num_labels, 1); // only label 1 is non-zero
    }

    // ── sample_negatives ─────────────────────────────────────────────────────

    #[test]
    fn sample_negatives_no_negatives_returns_empty_bank_error() {
        let mut bank = make_bank(8, 2);
        // All entries have the same label.
        bank.push(vec![1.0, 0.0], 5, 0)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(vec![0.0, 1.0], 5, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        let err = sample_negatives(&bank, 5, 4, 42);
        assert!(matches!(err, Err(FeatureBankError::EmptyBank)));
    }

    #[test]
    fn sample_negatives_picks_indices_with_different_label() {
        let mut bank = make_bank(16, 2);
        // Label 1 entries at indices 0, 1.
        bank.push(vec![1.0, 0.0], 1, 0)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(vec![0.9, 0.1], 1, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        // Label 2 entries at indices 2, 3.
        bank.push(vec![0.0, 1.0], 2, 2)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(vec![0.1, 0.9], 2, 3)
            .unwrap_or_else(|e| panic!("{e}"));

        let negs = sample_negatives(&bank, 1, 10, 99).unwrap_or_else(|e| panic!("{e}"));
        // All sampled indices must have label != 1.
        for &idx in &negs {
            assert_ne!(
                bank.entries[idx].label, 1,
                "negative has same label as query"
            );
        }
        // Should return exactly 2 (all available negatives).
        assert_eq!(negs.len(), 2);
    }

    // ── sample_positive ──────────────────────────────────────────────────────

    #[test]
    fn sample_positive_no_matching_label_returns_empty_bank_error() {
        let mut bank = make_bank(8, 2);
        bank.push(vec![1.0, 0.0], 1, 0)
            .unwrap_or_else(|e| panic!("{e}"));
        let err = sample_positive(&bank, 99, 42);
        assert!(matches!(err, Err(FeatureBankError::EmptyBank)));
    }

    #[test]
    fn sample_positive_returns_index_with_matching_label() {
        let mut bank = make_bank(16, 2);
        bank.push(vec![1.0, 0.0], 3, 0)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(vec![0.0, 1.0], 7, 1)
            .unwrap_or_else(|e| panic!("{e}"));
        bank.push(vec![0.5, 0.5], 3, 2)
            .unwrap_or_else(|e| panic!("{e}"));

        let idx = sample_positive(&bank, 3, 12345).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(bank.entries[idx].label, 3);
    }
}
