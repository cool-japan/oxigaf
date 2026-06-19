//! Contrastive and triplet losses for identity-preserving embedding training.
//!
//! In the GAF context these losses enforce that rendered views of the same
//! person are close in embedding space while different identities are far
//! apart.  Three complementary formulations are provided:
//!
//! - **Siamese contrastive loss** — pairwise positive/negative label.
//! - **Triplet loss** — anchor / positive / negative triples with optional
//!   hard-negative mining.
//! - **InfoNCE / NT-Xent** — SimCLR-style self-supervised contrastive loss.
//!
//! # Quick start
//! ```rust
//! use oxigaf_trainer::contrastive_loss::{
//!     ContrastiveConfig, contrastive_loss_pair,
//!     TripletConfig, triplet_loss,
//!     InfoNceConfig, infonce_loss,
//! };
//!
//! // Siamese pair (positive)
//! let a = vec![1.0_f32, 0.0];
//! let b = vec![0.9_f32, 0.1];
//! let loss = contrastive_loss_pair(&a, &b, 1.0, &ContrastiveConfig::default()).unwrap();
//!
//! // Triplet
//! let anchor   = vec![1.0_f32, 0.0];
//! let positive = vec![0.9_f32, 0.1];
//! let negative = vec![-1.0_f32, 0.0];
//! let tloss = triplet_loss(&anchor, &positive, &negative, &TripletConfig::default()).unwrap();
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by contrastive / triplet loss operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ContrastiveLossError {
    /// Two vectors that should have the same length do not.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// A configuration parameter is out of range or otherwise invalid.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// The batch provided is empty and at least one element is required.
    #[error("Empty batch: at least one element is required")]
    EmptyBatch,

    /// A non-finite value was encountered during computation.
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedding utilities
// ─────────────────────────────────────────────────────────────────────────────

/// L2 norm of a vector.
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Normalize a vector to unit length.
///
/// If the norm is below `1e-8` the zero vector is returned to avoid division
/// by zero.
pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let n = l2_norm(v);
    if n < 1e-8 {
        vec![0.0; v.len()]
    } else {
        v.iter().map(|x| x / n).collect()
    }
}

/// Pairwise squared L2 distance between two equal-length vectors.
///
/// # Errors
/// Returns [`ContrastiveLossError::DimensionMismatch`] when lengths differ.
pub fn squared_l2_distance(a: &[f32], b: &[f32]) -> Result<f32, ContrastiveLossError> {
    if a.len() != b.len() {
        return Err(ContrastiveLossError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum())
}

/// Cosine similarity between two equal-length vectors in `[-1, 1]`.
///
/// Uses `dot(a, b) / (norm_a * norm_b + 1e-8)` to avoid division by zero.
///
/// # Errors
/// Returns [`ContrastiveLossError::DimensionMismatch`] when lengths differ.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, ContrastiveLossError> {
    if a.len() != b.len() {
        return Err(ContrastiveLossError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = l2_norm(a);
    let nb = l2_norm(b);
    Ok(dot / (na * nb + 1e-8))
}

/// Pairwise squared-L2 distance matrix for a batch of embeddings.
///
/// Returns an n×n matrix in row-major order where `dist[i*n + j]` equals
/// `squared_l2_distance(embeddings[i], embeddings[j])`.
///
/// # Errors
/// - [`ContrastiveLossError::EmptyBatch`] if the slice is empty.
/// - [`ContrastiveLossError::DimensionMismatch`] if vectors have different lengths.
pub fn pairwise_distance_matrix(embeddings: &[Vec<f32>]) -> Result<Vec<f32>, ContrastiveLossError> {
    if embeddings.is_empty() {
        return Err(ContrastiveLossError::EmptyBatch);
    }
    let d = embeddings[0].len();
    let n = embeddings.len();
    let mut matrix = vec![0.0f32; n * n];
    for i in 0..n {
        if embeddings[i].len() != d {
            return Err(ContrastiveLossError::DimensionMismatch {
                expected: d,
                got: embeddings[i].len(),
            });
        }
        for j in 0..n {
            if i == j {
                continue; // remains 0.0
            }
            matrix[i * n + j] = squared_l2_distance(&embeddings[i], &embeddings[j])?;
        }
    }
    Ok(matrix)
}

// ─────────────────────────────────────────────────────────────────────────────
// Contrastive (Siamese) loss
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Siamese contrastive loss.
#[derive(Debug, Clone, PartialEq)]
pub struct ContrastiveConfig {
    /// Margin applied to negative pairs.  Must be > 0.  Default: `1.0`.
    pub margin: f32,
    /// Use cosine distance (`1 - cosine_similarity`) instead of L2 distance.
    /// Default: `false`.
    pub use_cosine: bool,
}

impl Default for ContrastiveConfig {
    fn default() -> Self {
        Self {
            margin: 1.0,
            use_cosine: false,
        }
    }
}

impl ContrastiveConfig {
    /// Validate configuration parameters.
    ///
    /// # Errors
    /// Returns [`ContrastiveLossError::InvalidConfig`] if `margin <= 0`.
    pub fn validate(&self) -> Result<(), ContrastiveLossError> {
        if self.margin <= 0.0 {
            return Err(ContrastiveLossError::InvalidConfig(format!(
                "margin must be > 0, got {}",
                self.margin
            )));
        }
        Ok(())
    }
}

/// Contrastive loss for a single embedding pair.
///
/// - `label = 1.0` — positive pair (same class).  Loss = `dist²`.
/// - `label = 0.0` — negative pair (different class).  Loss = `max(0, margin - dist)²`.
///
/// The distance used is either L2 or cosine depending on
/// [`ContrastiveConfig::use_cosine`].
///
/// # Errors
/// - Propagates [`ContrastiveLossError::DimensionMismatch`] when `a` and `b`
///   have different lengths.
/// - [`ContrastiveLossError::InvalidConfig`] if `config` fails validation.
pub fn contrastive_loss_pair(
    a: &[f32],
    b: &[f32],
    label: f32,
    config: &ContrastiveConfig,
) -> Result<f32, ContrastiveLossError> {
    config.validate()?;
    let dist = if config.use_cosine {
        1.0 - cosine_similarity(a, b)?
    } else {
        squared_l2_distance(a, b)?.sqrt()
    };
    let positive_loss = label * dist * dist;
    let hinge = (config.margin - dist).max(0.0);
    let negative_loss = (1.0 - label) * hinge * hinge;
    Ok(positive_loss + negative_loss)
}

/// Contrastive loss averaged over a batch of `(embedding_a, embedding_b, label)` pairs.
///
/// # Errors
/// - [`ContrastiveLossError::EmptyBatch`] if `pairs` is empty.
/// - Propagates errors from individual pair computations.
pub fn contrastive_loss_batch(
    pairs: &[(Vec<f32>, Vec<f32>, f32)],
    config: &ContrastiveConfig,
) -> Result<f32, ContrastiveLossError> {
    if pairs.is_empty() {
        return Err(ContrastiveLossError::EmptyBatch);
    }
    config.validate()?;
    let total: f32 = pairs
        .iter()
        .map(|(a, b, label)| contrastive_loss_pair(a, b, *label, config))
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .sum();
    Ok(total / pairs.len() as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Triplet loss
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the triplet loss.
#[derive(Debug, Clone, PartialEq)]
pub struct TripletConfig {
    /// Distance gap between positive and negative pairs.  Must be > 0.
    /// Default: `0.3`.
    pub margin: f32,
    /// Use soft margin `log(1 + exp(d_pos - d_neg + margin))` instead of
    /// `max(0, d_pos - d_neg + margin)`.  Default: `false`.
    pub soft_margin: bool,
    /// Use squared distances instead of L2 distances.  Default: `false`.
    pub squared: bool,
}

impl Default for TripletConfig {
    fn default() -> Self {
        Self {
            margin: 0.3,
            soft_margin: false,
            squared: false,
        }
    }
}

impl TripletConfig {
    /// Validate configuration parameters.
    ///
    /// # Errors
    /// Returns [`ContrastiveLossError::InvalidConfig`] if `margin <= 0`.
    pub fn validate(&self) -> Result<(), ContrastiveLossError> {
        if self.margin <= 0.0 {
            return Err(ContrastiveLossError::InvalidConfig(format!(
                "triplet margin must be > 0, got {}",
                self.margin
            )));
        }
        Ok(())
    }

    /// Compute the distance metric between two embedding vectors according to
    /// this configuration (squared or L2).
    fn distance(&self, a: &[f32], b: &[f32]) -> Result<f32, ContrastiveLossError> {
        let sq = squared_l2_distance(a, b)?;
        if self.squared {
            Ok(sq)
        } else {
            Ok(sq.sqrt())
        }
    }
}

/// Triplet loss for a single (anchor, positive, negative) triple.
///
/// - Standard: `max(0, d(anchor, positive) - d(anchor, negative) + margin)`
/// - Soft-margin: `log(1 + exp(d_pos - d_neg + margin))`
///
/// # Errors
/// Propagates dimension or config errors.
pub fn triplet_loss(
    anchor: &[f32],
    positive: &[f32],
    negative: &[f32],
    config: &TripletConfig,
) -> Result<f32, ContrastiveLossError> {
    config.validate()?;
    let d_pos = config.distance(anchor, positive)?;
    let d_neg = config.distance(anchor, negative)?;
    let diff = d_pos - d_neg + config.margin;
    if config.soft_margin {
        // numerically stable: log(1 + exp(x)) = x + log(1 + exp(-x)) for x > 0
        let loss = if diff > 0.0 {
            diff + (1.0 + (-diff).exp()).ln()
        } else {
            (1.0 + diff.exp()).ln()
        };
        Ok(loss)
    } else {
        Ok(diff.max(0.0))
    }
}

/// Triplet loss averaged over a batch of `(anchor, positive, negative)` triples.
///
/// # Errors
/// - [`ContrastiveLossError::EmptyBatch`] if `triplets` is empty.
/// - Propagates errors from individual triple computations.
pub fn triplet_loss_batch(
    triplets: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    config: &TripletConfig,
) -> Result<f32, ContrastiveLossError> {
    if triplets.is_empty() {
        return Err(ContrastiveLossError::EmptyBatch);
    }
    config.validate()?;
    let total: f32 = triplets
        .iter()
        .map(|(a, p, n)| triplet_loss(a, p, n, config))
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .sum();
    Ok(total / triplets.len() as f32)
}

/// Fraction of triplets that violate the margin constraint (i.e., have loss > 0).
///
/// A triplet violates the constraint when `d_pos >= d_neg - margin`, which is
/// equivalent to `d_pos - d_neg + margin >= 0` (the standard triplet loss term
/// is non-zero).
///
/// # Errors
/// - [`ContrastiveLossError::EmptyBatch`] if `triplets` is empty.
/// - Propagates dimension errors.
pub fn triplet_violation_rate(
    triplets: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    config: &TripletConfig,
) -> Result<f32, ContrastiveLossError> {
    if triplets.is_empty() {
        return Err(ContrastiveLossError::EmptyBatch);
    }
    config.validate()?;
    let mut violations = 0u32;
    for (a, p, n) in triplets {
        let d_pos = config.distance(a, p)?;
        let d_neg = config.distance(a, n)?;
        if d_pos - d_neg + config.margin > 0.0 {
            violations += 1;
        }
    }
    Ok(violations as f32 / triplets.len() as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Hard negative mining
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for mining hard triplets from a labelled embedding batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiningStrategy {
    /// All valid triplets enumerated exhaustively.
    AllTriplets,
    /// For each (anchor, positive) pair, choose the negative with the smallest
    /// distance to the anchor that is still greater than `d(anchor, positive)`.
    /// Falls back to the globally closest negative if no strictly harder one
    /// exists.
    HardNegative,
    /// For each (anchor, positive) pair, choose negatives where
    /// `d(anchor, positive) < d(anchor, negative) < d(anchor, positive) + margin`.
    /// Pairs with no qualifying negative are skipped.
    SemiHardNegative,
    /// For each anchor, choose the positive with the largest distance (hardest
    /// positive), then pair it with every valid negative.
    HardPositive,
}

/// Index triple referring to embeddings within a batch by position.
#[derive(Debug, Clone)]
pub struct TripletIndex {
    /// Index of the anchor embedding.
    pub anchor: usize,
    /// Index of the positive embedding (same class as anchor).
    pub positive: usize,
    /// Index of the negative embedding (different class from anchor).
    pub negative: usize,
}

/// Mine triplet indices from a labelled embedding batch.
///
/// # Arguments
/// - `embeddings` — batch of embedding vectors.
/// - `labels` — class label per embedding; same label means same identity.
/// - `strategy` — mining strategy to apply.
/// - `config` — triplet loss config (provides `margin` for `SemiHardNegative`).
///
/// # Errors
/// - [`ContrastiveLossError::EmptyBatch`] if the batch is empty.
/// - [`ContrastiveLossError::DimensionMismatch`] if `embeddings.len() !=
///   labels.len()`.
pub fn mine_triplets(
    embeddings: &[Vec<f32>],
    labels: &[u32],
    strategy: MiningStrategy,
    config: &TripletConfig,
) -> Result<Vec<TripletIndex>, ContrastiveLossError> {
    if embeddings.is_empty() {
        return Err(ContrastiveLossError::EmptyBatch);
    }
    if embeddings.len() != labels.len() {
        return Err(ContrastiveLossError::DimensionMismatch {
            expected: embeddings.len(),
            got: labels.len(),
        });
    }
    config.validate()?;

    let n = embeddings.len();

    // Pre-compute full pairwise distance matrix to avoid redundant computation.
    let dist = pairwise_distance_matrix(embeddings)?;
    // Convert squared distances to L2 or keep squared per config.
    let dist_fn = |i: usize, j: usize| -> f32 {
        let sq = dist[i * n + j];
        if config.squared {
            sq
        } else {
            sq.sqrt()
        }
    };

    let mut triplets = Vec::new();

    match strategy {
        MiningStrategy::AllTriplets => {
            for i in 0..n {
                for j in 0..n {
                    if j == i || labels[j] != labels[i] {
                        continue;
                    }
                    for k in 0..n {
                        if labels[k] == labels[i] {
                            continue;
                        }
                        triplets.push(TripletIndex {
                            anchor: i,
                            positive: j,
                            negative: k,
                        });
                    }
                }
            }
        }

        MiningStrategy::HardNegative => {
            for i in 0..n {
                for j in 0..n {
                    if j == i || labels[j] != labels[i] {
                        continue;
                    }
                    let d_pos = dist_fn(i, j);

                    // Find a negative with d(i,k) > d_pos and as small as possible.
                    let mut best_k: Option<usize> = None;
                    let mut best_d = f32::MAX;
                    let mut global_best_k: Option<usize> = None;
                    let mut global_best_d = f32::MAX;

                    for k in 0..n {
                        if labels[k] == labels[i] {
                            continue;
                        }
                        let d_neg = dist_fn(i, k);
                        if d_neg > d_pos && d_neg < best_d {
                            best_d = d_neg;
                            best_k = Some(k);
                        }
                        if d_neg < global_best_d {
                            global_best_d = d_neg;
                            global_best_k = Some(k);
                        }
                    }
                    let chosen = best_k.or(global_best_k);
                    if let Some(k) = chosen {
                        triplets.push(TripletIndex {
                            anchor: i,
                            positive: j,
                            negative: k,
                        });
                    }
                }
            }
        }

        MiningStrategy::SemiHardNegative => {
            for i in 0..n {
                for j in 0..n {
                    if j == i || labels[j] != labels[i] {
                        continue;
                    }
                    let d_pos = dist_fn(i, j);
                    let upper = d_pos + config.margin;

                    for k in 0..n {
                        if labels[k] == labels[i] {
                            continue;
                        }
                        let d_neg = dist_fn(i, k);
                        if d_neg > d_pos && d_neg < upper {
                            triplets.push(TripletIndex {
                                anchor: i,
                                positive: j,
                                negative: k,
                            });
                        }
                    }
                }
            }
        }

        MiningStrategy::HardPositive => {
            for i in 0..n {
                // Find the hardest positive: same class, largest distance.
                let mut hard_pos: Option<usize> = None;
                let mut max_d = -1.0f32;
                for j in 0..n {
                    if j == i || labels[j] != labels[i] {
                        continue;
                    }
                    let d = dist_fn(i, j);
                    if d > max_d {
                        max_d = d;
                        hard_pos = Some(j);
                    }
                }
                if let Some(j) = hard_pos {
                    for k in 0..n {
                        if labels[k] == labels[i] {
                            continue;
                        }
                        triplets.push(TripletIndex {
                            anchor: i,
                            positive: j,
                            negative: k,
                        });
                    }
                }
            }
        }
    }

    Ok(triplets)
}

/// Apply mined triplet loss over a labelled embedding batch.
///
/// Returns `(loss, num_triplets_used)`.
///
/// # Errors
/// - [`ContrastiveLossError::EmptyBatch`] when no valid triplets can be found.
/// - Propagates other errors from mining or loss computation.
pub fn mined_triplet_loss(
    embeddings: &[Vec<f32>],
    labels: &[u32],
    strategy: MiningStrategy,
    config: &TripletConfig,
) -> Result<(f32, usize), ContrastiveLossError> {
    let triplet_indices = mine_triplets(embeddings, labels, strategy, config)?;
    if triplet_indices.is_empty() {
        return Err(ContrastiveLossError::EmptyBatch);
    }
    let mut total_loss = 0.0f32;
    for idx in &triplet_indices {
        let loss = triplet_loss(
            &embeddings[idx.anchor],
            &embeddings[idx.positive],
            &embeddings[idx.negative],
            config,
        )?;
        total_loss += loss;
    }
    let n = triplet_indices.len();
    Ok((total_loss / n as f32, n))
}

// ─────────────────────────────────────────────────────────────────────────────
// InfoNCE / NT-Xent loss (SimCLR-style)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the InfoNCE / NT-Xent contrastive loss.
#[derive(Debug, Clone, PartialEq)]
pub struct InfoNceConfig {
    /// Temperature parameter; lower values produce sharper distributions.
    /// Must be > 0.  Default: `0.07`.
    pub temperature: f32,
}

impl Default for InfoNceConfig {
    fn default() -> Self {
        Self { temperature: 0.07 }
    }
}

impl InfoNceConfig {
    /// Validate configuration parameters.
    ///
    /// # Errors
    /// Returns [`ContrastiveLossError::InvalidConfig`] if `temperature <= 0`.
    pub fn validate(&self) -> Result<(), ContrastiveLossError> {
        if self.temperature <= 0.0 {
            return Err(ContrastiveLossError::InvalidConfig(format!(
                "temperature must be > 0, got {}",
                self.temperature
            )));
        }
        Ok(())
    }
}

/// InfoNCE / NT-Xent loss for `N` positive pairs from a batch of `2N` embeddings.
///
/// Given two augmented views `embeddings_a` and `embeddings_b` (each of length
/// `N`), pair `(a_i, b_i)` is the positive pair for index `i`; all other
/// combinations are treated as negatives.
///
/// The combined 2N-element array is ordered as `[a_0..a_{N-1}, b_0..b_{N-1}]`.
/// For `a_i` (row `i`) the positive is `b_i` at column `N + i`.
/// For `b_i` (row `N + i`) the positive is `a_i` at column `i`.
///
/// # Algorithm
/// 1. L2-normalize all 2N embeddings.
/// 2. Build the 2N×2N cosine-similarity matrix, scaled by `1 / temperature`.
/// 3. Set the diagonal to `-1e9` to exclude self-similarity from the softmax.
/// 4. Compute cross-entropy loss for each row against its positive target.
/// 5. Return the mean over all 2N rows.
///
/// # Errors
/// - [`ContrastiveLossError::EmptyBatch`] if either view list is empty.
/// - [`ContrastiveLossError::DimensionMismatch`] if the two view lists have
///   different lengths or any embedding has a different dimension.
/// - [`ContrastiveLossError::NumericalError`] if a NaN/Inf is produced.
pub fn infonce_loss(
    embeddings_a: &[Vec<f32>],
    embeddings_b: &[Vec<f32>],
    config: &InfoNceConfig,
) -> Result<f32, ContrastiveLossError> {
    config.validate()?;
    if embeddings_a.is_empty() {
        return Err(ContrastiveLossError::EmptyBatch);
    }
    if embeddings_a.len() != embeddings_b.len() {
        return Err(ContrastiveLossError::DimensionMismatch {
            expected: embeddings_a.len(),
            got: embeddings_b.len(),
        });
    }

    let n = embeddings_a.len();
    let dim = embeddings_a[0].len();

    // Validate all embedding dimensions and normalize.
    let mut all: Vec<Vec<f32>> = Vec::with_capacity(2 * n);
    for e in embeddings_a.iter().chain(embeddings_b.iter()) {
        if e.len() != dim {
            return Err(ContrastiveLossError::DimensionMismatch {
                expected: dim,
                got: e.len(),
            });
        }
        all.push(l2_normalize(e));
    }

    let total = 2 * n;
    let inv_temp = 1.0 / config.temperature;

    // Build similarity matrix: sim[i][j] = dot(all[i], all[j]) / temperature.
    // All embeddings are already unit-length so dot == cosine similarity.
    let mut sim = vec![0.0f32; total * total];
    for i in 0..total {
        for j in 0..total {
            if i == j {
                // Mask self-similarity.
                sim[i * total + j] = -1e9;
            } else {
                let dot: f32 = all[i].iter().zip(all[j].iter()).map(|(x, y)| x * y).sum();
                sim[i * total + j] = dot * inv_temp;
            }
        }
    }

    // For row i, the positive index is:
    //   if i < n → positive at N + i
    //   else     → positive at i - N
    let positive_idx = |i: usize| -> usize {
        if i < n {
            n + i
        } else {
            i - n
        }
    };

    let mut total_loss = 0.0f32;
    for i in 0..total {
        let logits = &sim[i * total..(i + 1) * total];
        let pos = positive_idx(i);

        // Numerically stable log-softmax: subtract max before exp.
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        if !exp_sum.is_finite() || exp_sum == 0.0 {
            return Err(ContrastiveLossError::NumericalError(format!(
                "exp_sum is not finite at row {i}: {exp_sum}"
            )));
        }
        let log_prob = (logits[pos] - max_logit) - exp_sum.ln();
        if !log_prob.is_finite() {
            return Err(ContrastiveLossError::NumericalError(format!(
                "log_prob is not finite at row {i}: {log_prob}"
            )));
        }
        // Cross-entropy: negative log-probability of the positive.
        total_loss -= log_prob;
    }

    Ok(total_loss / total as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── l2_norm ───────────────────────────────────────────────────────────────

    #[test]
    fn test_l2_norm_zero_vector() {
        let v = vec![0.0f32; 4];
        assert_eq!(l2_norm(&v), 0.0);
    }

    #[test]
    fn test_l2_norm_unit_vector() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!((l2_norm(&v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm_known_value() {
        let v = vec![3.0f32, 4.0]; // 3-4-5 triangle
        assert!((l2_norm(&v) - 5.0).abs() < 1e-6);
    }

    // ── l2_normalize ──────────────────────────────────────────────────────────

    #[test]
    fn test_l2_normalize_unit_vector() {
        let v = vec![1.0f32, 0.0, 0.0];
        let n = l2_normalize(&v);
        assert!((n[0] - 1.0).abs() < 1e-6);
        assert!(n[1].abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let v = vec![0.0f32; 4];
        let n = l2_normalize(&v);
        assert!(n.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_l2_normalize_produces_unit_norm() {
        let v = vec![3.0f32, 4.0];
        let n = l2_normalize(&v);
        assert!((l2_norm(&n) - 1.0).abs() < 1e-6);
    }

    // ── squared_l2_distance ───────────────────────────────────────────────────

    #[test]
    fn test_squared_l2_distance_same_vector() {
        let v = vec![1.0f32, 2.0, 3.0];
        let d = squared_l2_distance(&v, &v).expect("same-length vectors");
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn test_squared_l2_distance_known() {
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        let d = squared_l2_distance(&a, &b).expect("same-length");
        assert!((d - 25.0).abs() < 1e-5);
    }

    #[test]
    fn test_squared_l2_distance_mismatch() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32];
        let err = squared_l2_distance(&a, &b).unwrap_err();
        assert!(matches!(
            err,
            ContrastiveLossError::DimensionMismatch {
                expected: 2,
                got: 1
            }
        ));
    }

    // ── cosine_similarity ─────────────────────────────────────────────────────

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0f32, 2.0, 3.0];
        let s = cosine_similarity(&v, &v).expect("same-length");
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let s = cosine_similarity(&a, &b).expect("same-length");
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        let s = cosine_similarity(&a, &b).expect("same-length");
        assert!((s + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_dimension_mismatch() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0];
        let err = cosine_similarity(&a, &b).unwrap_err();
        assert!(matches!(
            err,
            ContrastiveLossError::DimensionMismatch { .. }
        ));
    }

    // ── pairwise_distance_matrix ──────────────────────────────────────────────

    #[test]
    fn test_pairwise_distance_matrix_symmetric_and_zero_diagonal() {
        let embs = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let mat = pairwise_distance_matrix(&embs).expect("valid batch");
        let n = 2;
        // Diagonal should be 0.
        assert!((mat[0]).abs() < 1e-6);
        assert!((mat[n + 1]).abs() < 1e-6);
        // Symmetric.
        assert!((mat[1] - mat[n]).abs() < 1e-6);
        // Known value: squared distance between (1,0) and (0,1) = 1+1 = 2.
        assert!((mat[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_pairwise_distance_matrix_empty() {
        let embs: Vec<Vec<f32>> = vec![];
        let err = pairwise_distance_matrix(&embs).unwrap_err();
        assert!(matches!(err, ContrastiveLossError::EmptyBatch));
    }

    // ── ContrastiveConfig::validate ───────────────────────────────────────────

    #[test]
    fn test_contrastive_config_valid() {
        let cfg = ContrastiveConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_contrastive_config_invalid_margin() {
        let cfg = ContrastiveConfig {
            margin: -0.5,
            use_cosine: false,
        };
        assert!(cfg.validate().is_err());
    }

    // ── contrastive_loss_pair ─────────────────────────────────────────────────

    #[test]
    fn test_contrastive_loss_pair_positive_zero_distance() {
        // Positive pair, same point → distance=0, loss=0.
        let v = vec![1.0f32, 0.0];
        let cfg = ContrastiveConfig::default();
        let loss = contrastive_loss_pair(&v, &v, 1.0, &cfg).expect("valid");
        assert!(loss.abs() < 1e-6);
    }

    #[test]
    fn test_contrastive_loss_pair_negative_zero_distance() {
        // Negative pair, distance=0 → hinge = margin, loss = margin².
        let v = vec![1.0f32, 0.0];
        let cfg = ContrastiveConfig::default(); // margin=1.0
        let loss = contrastive_loss_pair(&v, &v, 0.0, &cfg).expect("valid");
        assert!((loss - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_contrastive_loss_pair_negative_at_margin() {
        // Negative pair, distance == margin → hinge=0, loss=0.
        let a = vec![0.0f32];
        let b = vec![1.0f32]; // dist=1 = margin
        let cfg = ContrastiveConfig::default(); // margin=1.0
        let loss = contrastive_loss_pair(&a, &b, 0.0, &cfg).expect("valid");
        assert!(loss.abs() < 1e-5);
    }

    #[test]
    fn test_contrastive_loss_pair_cosine_mode() {
        // Identical unit vectors → cosine dist = 0 → positive pair loss = 0.
        let v = l2_normalize(&[1.0f32, 2.0, 3.0]);
        let cfg = ContrastiveConfig {
            margin: 0.5,
            use_cosine: true,
        };
        let loss = contrastive_loss_pair(&v, &v, 1.0, &cfg).expect("valid");
        assert!(loss.abs() < 1e-5);
    }

    // ── contrastive_loss_batch ────────────────────────────────────────────────

    #[test]
    fn test_contrastive_loss_batch_empty() {
        let pairs: Vec<(Vec<f32>, Vec<f32>, f32)> = vec![];
        let err = contrastive_loss_batch(&pairs, &ContrastiveConfig::default()).unwrap_err();
        assert!(matches!(err, ContrastiveLossError::EmptyBatch));
    }

    #[test]
    fn test_contrastive_loss_batch_single_pair() {
        let a = vec![0.0f32];
        let b = vec![0.0f32];
        let pairs = vec![(a, b, 1.0f32)];
        let loss = contrastive_loss_batch(&pairs, &ContrastiveConfig::default()).expect("valid");
        // Positive pair, distance=0 → loss=0.
        assert!(loss.abs() < 1e-6);
    }

    // ── TripletConfig::validate ───────────────────────────────────────────────

    #[test]
    fn test_triplet_config_valid() {
        let cfg = TripletConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_triplet_config_invalid_margin() {
        let cfg = TripletConfig {
            margin: 0.0,
            soft_margin: false,
            squared: false,
        };
        assert!(cfg.validate().is_err());
    }

    // ── triplet_loss ──────────────────────────────────────────────────────────

    #[test]
    fn test_triplet_loss_zero_distances_equals_margin() {
        // d_pos=0, d_neg=0 → loss = max(0, 0-0+margin) = margin.
        let v = vec![1.0f32, 0.0];
        let cfg = TripletConfig::default(); // margin=0.3
        let loss = triplet_loss(&v, &v, &v, &cfg).expect("valid");
        assert!((loss - cfg.margin).abs() < 1e-5);
    }

    #[test]
    fn test_triplet_loss_satisfied() {
        // d_pos < d_neg - margin → loss=0.
        let anchor = vec![0.0f32];
        let positive = vec![0.1f32]; // d_pos=0.1
        let negative = vec![2.0f32]; // d_neg=2.0; 0.1 < 2.0-0.3=1.7 → satisfied
        let cfg = TripletConfig::default();
        let loss = triplet_loss(&anchor, &positive, &negative, &cfg).expect("valid");
        assert!(loss.abs() < 1e-6);
    }

    #[test]
    fn test_triplet_loss_violated() {
        // d_pos > d_neg → strong violation.
        let anchor = vec![0.0f32];
        let positive = vec![3.0f32]; // d_pos=3.0
        let negative = vec![0.5f32]; // d_neg=0.5; 3.0-0.5+0.3=2.8 > 0
        let cfg = TripletConfig::default();
        let loss = triplet_loss(&anchor, &positive, &negative, &cfg).expect("valid");
        assert!(loss > 0.0);
        assert!((loss - 2.8f32).abs() < 1e-5);
    }

    #[test]
    fn test_triplet_loss_soft_margin() {
        let anchor = vec![0.0f32];
        let positive = vec![3.0f32];
        let negative = vec![0.5f32];
        let cfg = TripletConfig {
            margin: 0.3,
            soft_margin: true,
            squared: false,
        };
        let loss = triplet_loss(&anchor, &positive, &negative, &cfg).expect("valid");
        // diff = 3.0 - 0.5 + 0.3 = 2.8 → soft = log(1+exp(2.8)) > 2.8
        assert!(loss > 2.8);
    }

    // ── triplet_loss_batch ────────────────────────────────────────────────────

    #[test]
    fn test_triplet_loss_batch_empty() {
        let triplets: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = vec![];
        let err = triplet_loss_batch(&triplets, &TripletConfig::default()).unwrap_err();
        assert!(matches!(err, ContrastiveLossError::EmptyBatch));
    }

    // ── triplet_violation_rate ────────────────────────────────────────────────

    #[test]
    fn test_triplet_violation_rate_all_satisfied() {
        // All well-separated triplets → violation rate = 0.
        let anchor = vec![0.0f32];
        let positive = vec![0.1f32];
        let negative = vec![5.0f32];
        let triplets = vec![(anchor, positive, negative)];
        let rate = triplet_violation_rate(&triplets, &TripletConfig::default()).expect("valid");
        assert!(rate.abs() < 1e-6);
    }

    #[test]
    fn test_triplet_violation_rate_all_violated() {
        let anchor = vec![0.0f32];
        let positive = vec![5.0f32];
        let negative = vec![0.1f32];
        let triplets = vec![(anchor, positive, negative)];
        let rate = triplet_violation_rate(&triplets, &TripletConfig::default()).expect("valid");
        assert!((rate - 1.0).abs() < 1e-6);
    }

    // ── mine_triplets ─────────────────────────────────────────────────────────

    #[test]
    fn test_mine_triplets_all_single_class_no_triplets() {
        // Only one class → no negatives → no triplets.
        let embs = vec![vec![1.0f32, 0.0], vec![0.9f32, 0.1]];
        let labels = vec![0u32, 0u32];
        let cfg = TripletConfig::default();
        let triplets =
            mine_triplets(&embs, &labels, MiningStrategy::AllTriplets, &cfg).expect("valid call");
        assert!(triplets.is_empty());
    }

    #[test]
    fn test_mine_triplets_all_two_classes() {
        // 2 from class 0, 2 from class 1.
        // Each class-0 anchor has 1 positive in class-0 and 2 negatives in class-1.
        // Expected triplets: 2 anchors × 1 positive × 2 negatives = 4 per class
        // (mirrored), total 8.
        let embs = vec![
            vec![1.0f32, 0.0],  // label 0
            vec![0.9f32, 0.1],  // label 0
            vec![-1.0f32, 0.0], // label 1
            vec![-0.9f32, 0.1], // label 1
        ];
        let labels = vec![0u32, 0u32, 1u32, 1u32];
        let cfg = TripletConfig::default();
        let triplets =
            mine_triplets(&embs, &labels, MiningStrategy::AllTriplets, &cfg).expect("valid");
        assert_eq!(triplets.len(), 8);
    }

    #[test]
    fn test_mine_triplets_hard_negative() {
        let embs = vec![
            vec![0.0f32],  // anchor, label 0
            vec![0.2f32],  // positive, label 0
            vec![0.5f32],  // neg-far, label 1
            vec![0.25f32], // neg-close, label 1 (hardest: slightly > d_pos=0.2)
        ];
        let labels = vec![0u32, 0u32, 1u32, 1u32];
        let cfg = TripletConfig::default();
        let triplets =
            mine_triplets(&embs, &labels, MiningStrategy::HardNegative, &cfg).expect("valid");
        assert!(!triplets.is_empty());
    }

    #[test]
    fn test_mine_triplets_semi_hard_negative() {
        // anchor=0, positive=0.2 (d_pos=0.2), margin=0.3
        // semi-hard window: (0.2, 0.5)
        // neg at 0.4 qualifies; neg at 0.6 does not; neg at 0.1 does not.
        let embs = vec![
            vec![0.0f32], // label 0
            vec![0.2f32], // label 0
            vec![0.4f32], // label 1 (semi-hard)
            vec![0.6f32], // label 1 (too far)
            vec![0.1f32], // label 1 (too close)
        ];
        let labels = vec![0u32, 0u32, 1u32, 1u32, 1u32];
        let cfg = TripletConfig::default(); // margin=0.3
        let triplets =
            mine_triplets(&embs, &labels, MiningStrategy::SemiHardNegative, &cfg).expect("valid");
        // Only the neg at index 2 (d=0.4) should qualify for the (anchor=0,pos=1) pair.
        assert!(triplets
            .iter()
            .any(|t| t.anchor == 0 && t.positive == 1 && t.negative == 2));
        // Neg at index 3 (d=0.6) must NOT qualify.
        assert!(!triplets
            .iter()
            .any(|t| t.anchor == 0 && t.positive == 1 && t.negative == 3));
    }

    // ── InfoNceConfig::validate ───────────────────────────────────────────────

    #[test]
    fn test_infonce_config_valid() {
        assert!(InfoNceConfig::default().validate().is_ok());
    }

    #[test]
    fn test_infonce_config_invalid_temperature() {
        let cfg = InfoNceConfig { temperature: 0.0 };
        assert!(cfg.validate().is_err());
    }

    // ── infonce_loss ──────────────────────────────────────────────────────────

    #[test]
    fn test_infonce_loss_identical_views_low_loss() {
        // When a_i == b_i (identical views), the positive logit dominates.
        // With temperature=0.07 the loss should be near 0 for 2 pairs.
        let embs_a = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let embs_b = embs_a.clone();
        let cfg = InfoNceConfig { temperature: 0.07 };
        let loss = infonce_loss(&embs_a, &embs_b, &cfg).expect("valid");
        // With 4 total embeddings (2N=4) and perfect positives, loss should be < 0.5.
        assert!(
            loss < 0.5,
            "loss should be low for identical views, got {loss}"
        );
    }

    #[test]
    fn test_infonce_loss_random_views_higher_than_identical() {
        // Randomly paired views (no correlation) should have higher loss than identical.
        let embs_a = vec![
            l2_normalize(&[1.0f32, 0.0, 0.0]),
            l2_normalize(&[0.0f32, 1.0, 0.0]),
            l2_normalize(&[0.0f32, 0.0, 1.0]),
        ];
        let embs_b = vec![
            l2_normalize(&[-1.0f32, 0.0, 0.0]),
            l2_normalize(&[0.0f32, -1.0, 0.0]),
            l2_normalize(&[0.0f32, 0.0, -1.0]),
        ];
        let cfg = InfoNceConfig { temperature: 0.07 };
        let loss_identical = infonce_loss(&embs_a, &embs_a, &cfg).expect("valid");
        let loss_random = infonce_loss(&embs_a, &embs_b, &cfg).expect("valid");
        assert!(
            loss_random > loss_identical,
            "random views (loss={loss_random}) should have higher loss than identical (loss={loss_identical})"
        );
    }

    #[test]
    fn test_infonce_loss_empty_batch() {
        let embs_a: Vec<Vec<f32>> = vec![];
        let embs_b: Vec<Vec<f32>> = vec![];
        let err = infonce_loss(&embs_a, &embs_b, &InfoNceConfig::default()).unwrap_err();
        assert!(matches!(err, ContrastiveLossError::EmptyBatch));
    }

    #[test]
    fn test_infonce_loss_length_mismatch() {
        let embs_a = vec![vec![1.0f32, 0.0]];
        let embs_b = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let err = infonce_loss(&embs_a, &embs_b, &InfoNceConfig::default()).unwrap_err();
        assert!(matches!(
            err,
            ContrastiveLossError::DimensionMismatch { .. }
        ));
    }

    // ── mined_triplet_loss ────────────────────────────────────────────────────

    #[test]
    fn test_mined_triplet_loss_empty_batch_when_no_valid_triplets() {
        // Semi-hard with no semi-hard negatives → EmptyBatch.
        let embs = vec![
            vec![0.0f32], // label 0
            vec![0.1f32], // label 0
            // Negative is very far, outside semi-hard window.
            vec![10.0f32], // label 1
        ];
        let labels = vec![0u32, 0u32, 1u32];
        // margin=0.3; d_pos=0.1; upper=0.4; d_neg=10.0 > 0.4 → no semi-hard neg.
        let cfg = TripletConfig::default();
        let err =
            mined_triplet_loss(&embs, &labels, MiningStrategy::SemiHardNegative, &cfg).unwrap_err();
        assert!(matches!(err, ContrastiveLossError::EmptyBatch));
    }

    #[test]
    fn test_mined_triplet_loss_all_returns_positive() {
        let embs = vec![
            vec![0.0f32, 0.0], // label 0
            vec![0.1f32, 0.0], // label 0
            vec![5.0f32, 0.0], // label 1
        ];
        let labels = vec![0u32, 0u32, 1u32];
        let cfg = TripletConfig::default();
        let (loss, n) = mined_triplet_loss(&embs, &labels, MiningStrategy::AllTriplets, &cfg)
            .expect("valid triplets");
        assert!(n > 0);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_mined_triplet_loss_single_class_empty_batch() {
        // Single class → no negatives → no triplets → EmptyBatch.
        let embs = vec![vec![1.0f32], vec![2.0f32]];
        let labels = vec![0u32, 0u32];
        let cfg = TripletConfig::default();
        let err =
            mined_triplet_loss(&embs, &labels, MiningStrategy::AllTriplets, &cfg).unwrap_err();
        assert!(matches!(err, ContrastiveLossError::EmptyBatch));
    }

    #[test]
    fn test_triplet_loss_batch_mean() {
        // Two identical triplets with known loss → mean equals that loss.
        let anchor = vec![0.0f32];
        let positive = vec![3.0f32];
        let negative = vec![0.5f32];
        let cfg = TripletConfig::default();
        let single = triplet_loss(&anchor, &positive, &negative, &cfg).expect("valid");
        let triplets = vec![
            (anchor.clone(), positive.clone(), negative.clone()),
            (anchor.clone(), positive.clone(), negative.clone()),
        ];
        let batch = triplet_loss_batch(&triplets, &cfg).expect("valid");
        assert!((batch - single).abs() < 1e-5);
    }
}
