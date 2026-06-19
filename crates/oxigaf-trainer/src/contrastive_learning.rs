//! Contrastive learning for identity-discriminative facial embeddings.
//!
//! Implements NT-Xent (SimCLR), InfoNCE, triplet, and supervised contrastive
//! losses together with a MoCo-style memory queue, hard/semi-hard negative
//! mining, and alignment/uniformity metrics.
//!
//! All functions use the `cl_` prefix to avoid collisions with names already
//! exported from [`contrastive_loss`](super::contrastive_loss).

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by contrastive-learning operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ContrastiveError {
    #[error("batch too small: need at least 2 samples, got {0}")]
    BatchTooSmall(usize),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("invalid temperature: must be > 0, got {0}")]
    InvalidTemperature(f32),
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("queue empty")]
    QueueEmpty,
    #[error("mismatched labels and embeddings: {n_labels} labels, {n_embeds} embeddings")]
    LabelEmbedMismatch { n_labels: usize, n_embeds: usize },
}

// ─────────────────────────────────────────────────────────────────────────────
// PRNG (no rand crate)
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

#[allow(dead_code)]
#[inline]
fn xorshift_f32(state: &mut u64) -> f32 {
    xorshift64(state) as f32 / u64::MAX as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for contrastive-learning losses.
#[derive(Debug, Clone, PartialEq)]
pub struct ContrastiveLearningConfig {
    /// Temperature for similarity scaling. Must be > 0. Default: 0.07.
    pub temperature: f32,
    /// Embedding dimension. Default: 256.
    pub embedding_dim: usize,
    /// Memory-bank capacity (MoCo-style). Default: 4096.
    pub queue_size: usize,
    /// Momentum for key-encoder update. Default: 0.999.
    pub momentum: f32,
    /// L2-normalise embeddings before similarity computation. Default: true.
    pub normalize_embeddings: bool,
    /// Use only the hardest negative per anchor. Default: false.
    pub hard_negative_mining: bool,
    /// Margin for triplet loss. Default: 0.5.
    pub margin: f32,
    /// Negatives per positive (NT-Xent). Default: uses full batch.
    pub n_negatives: usize,
}

impl Default for ContrastiveLearningConfig {
    fn default() -> Self {
        Self {
            temperature: 0.07,
            embedding_dim: 256,
            queue_size: 4096,
            momentum: 0.999,
            normalize_embeddings: true,
            hard_negative_mining: false,
            margin: 0.5,
            n_negatives: 0, // 0 = full batch (2N-2)
        }
    }
}

impl ContrastiveLearningConfig {
    /// Validate key parameters. Returns `Err` if temperature ≤ 0.
    pub fn validate(&self) -> Result<(), ContrastiveError> {
        if self.temperature <= 0.0 {
            return Err(ContrastiveError::InvalidTemperature(self.temperature));
        }
        if self.embedding_dim == 0 {
            return Err(ContrastiveError::InvalidConfig(
                "embedding_dim must be > 0".into(),
            ));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Primitive vector operations
// ─────────────────────────────────────────────────────────────────────────────

/// L2 norm of a slice.
#[inline]
fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// L2 normalise a vector. Returns zero vector for near-zero inputs.
pub fn cl_normalize(v: &[f32]) -> Vec<f32> {
    let n = vec_norm(v);
    if n < 1e-8 {
        vec![0.0f32; v.len()]
    } else {
        v.iter().map(|x| x / n).collect()
    }
}

/// Dot product of two equal-length vectors.
///
/// # Errors
/// Returns [`ContrastiveError::DimensionMismatch`] when lengths differ.
pub fn cl_dot(a: &[f32], b: &[f32]) -> Result<f32, ContrastiveError> {
    if a.len() != b.len() {
        return Err(ContrastiveError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
}

/// Cosine similarity between two vectors, in `[-1, 1]`.
///
/// # Errors
/// Returns [`ContrastiveError::DimensionMismatch`] for empty or mismatched inputs.
pub fn cl_cosine_sim(a: &[f32], b: &[f32]) -> Result<f32, ContrastiveError> {
    if a.is_empty() || b.is_empty() {
        return Err(ContrastiveError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if a.len() != b.len() {
        return Err(ContrastiveError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = vec_norm(a);
    let nb = vec_norm(b);
    Ok(dot / (na * nb + 1e-8))
}

/// L2 distance between two equal-length vectors.
///
/// # Errors
/// Returns [`ContrastiveError::DimensionMismatch`] when lengths differ.
pub fn cl_l2_distance(a: &[f32], b: &[f32]) -> Result<f32, ContrastiveError> {
    if a.len() != b.len() {
        return Err(ContrastiveError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt())
}

/// Cosine similarity matrix for a batch of embeddings (row-major, n×n).
///
/// Optionally L2-normalises each embedding first.
///
/// # Errors
/// - [`ContrastiveError::BatchTooSmall`] if `embeddings` is empty.
/// - [`ContrastiveError::DimensionMismatch`] if rows have inconsistent length.
pub fn cl_similarity_matrix(
    embeddings: &[Vec<f32>],
    normalize: bool,
) -> Result<Vec<f32>, ContrastiveError> {
    if embeddings.is_empty() {
        return Err(ContrastiveError::BatchTooSmall(0));
    }
    let dim = embeddings[0].len();
    let n = embeddings.len();

    // Optionally normalise
    let vecs: Vec<Vec<f32>> = embeddings
        .iter()
        .map(|v| {
            if v.len() != dim {
                return Err(ContrastiveError::DimensionMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
            Ok(if normalize {
                cl_normalize(v)
            } else {
                v.clone()
            })
        })
        .collect::<Result<_, _>>()?;

    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let dot: f32 = vecs[i].iter().zip(vecs[j].iter()).map(|(x, y)| x * y).sum();
            out[i * n + j] = dot;
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Numerically stable softmax / log-sum-exp helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute log(sum(exp(v[i]))) stably.
fn log_sum_exp(v: &[f32]) -> f32 {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if max.is_infinite() {
        return f32::NEG_INFINITY;
    }
    max + v.iter().map(|x| (x - max).exp()).sum::<f32>().ln()
}

// ─────────────────────────────────────────────────────────────────────────────
// NT-Xent loss  (SimCLR)
// ─────────────────────────────────────────────────────────────────────────────

/// NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
///
/// Expects `embeddings.len() == 2N` where `(embeddings[2k], embeddings[2k+1])`
/// form positive pairs. All other samples in the batch act as negatives.
///
/// loss = -mean_{i in 2N} log( exp(sim(z_i, z_pos_i)/T) /
///                              sum_{k≠i} exp(sim(z_i, z_k)/T) )
///
/// # Errors
/// - [`ContrastiveError::BatchTooSmall`] if fewer than 2 embeddings.
/// - [`ContrastiveError::InvalidConfig`] if `embeddings.len()` is odd.
/// - [`ContrastiveError::InvalidTemperature`] if `temperature ≤ 0`.
/// - [`ContrastiveError::DimensionMismatch`] on shape inconsistency.
pub fn cl_nt_xent_loss(
    embeddings: &[Vec<f32>],
    config: &ContrastiveLearningConfig,
) -> Result<f32, ContrastiveError> {
    config.validate()?;
    let total = embeddings.len();
    if total < 2 {
        return Err(ContrastiveError::BatchTooSmall(total));
    }
    if !total.is_multiple_of(2) {
        return Err(ContrastiveError::InvalidConfig(format!(
            "NT-Xent requires an even number of embeddings (2N), got {}",
            total
        )));
    }
    let dim = embeddings[0].len();
    for e in embeddings.iter() {
        if e.len() != dim {
            return Err(ContrastiveError::DimensionMismatch {
                expected: dim,
                got: e.len(),
            });
        }
    }

    // Normalise all embeddings
    let vecs: Vec<Vec<f32>> = embeddings.iter().map(|v| cl_normalize(v)).collect();
    let t = config.temperature;

    // Compute full similarity matrix (temperature-scaled)
    let mut sim = vec![0.0f32; total * total];
    for i in 0..total {
        for j in 0..total {
            let dot: f32 = vecs[i].iter().zip(vecs[j].iter()).map(|(x, y)| x * y).sum();
            sim[i * total + j] = dot / t;
        }
    }

    let mut total_loss = 0.0f32;

    // For every sample i, its positive partner is:
    //   i_pos = i ^ 1  (0↔1, 2↔3, ...)
    for i in 0..total {
        let i_pos = i ^ 1;

        // Numerator: sim(z_i, z_pos)
        let num_logit = sim[i * total + i_pos];

        // Denominator: all k ≠ i
        let denom_logits: Vec<f32> = (0..total)
            .filter(|&k| k != i)
            .map(|k| sim[i * total + k])
            .collect();

        let lse = log_sum_exp(&denom_logits);
        total_loss += -(num_logit - lse);
    }

    Ok(total_loss / total as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// InfoNCE loss
// ─────────────────────────────────────────────────────────────────────────────

/// InfoNCE loss: each anchor has one positive and M shared negatives.
///
/// loss = -mean_i log( exp(sim(a_i, p_i)/T) /
///                     (exp(sim(a_i, p_i)/T) + sum_k exp(sim(a_i, n_k)/T)) )
///
/// # Errors
/// - [`ContrastiveError::BatchTooSmall`] for empty anchors or negatives.
/// - [`ContrastiveError::DimensionMismatch`] on shape mismatch.
/// - [`ContrastiveError::LabelEmbedMismatch`] if anchors and positives differ in count.
/// - [`ContrastiveError::InvalidTemperature`] if `temperature ≤ 0`.
pub fn cl_info_nce_loss(
    anchors: &[Vec<f32>],
    positives: &[Vec<f32>],
    negatives: &[Vec<f32>],
    config: &ContrastiveLearningConfig,
) -> Result<f32, ContrastiveError> {
    config.validate()?;
    if anchors.is_empty() {
        return Err(ContrastiveError::BatchTooSmall(0));
    }
    if negatives.is_empty() {
        return Err(ContrastiveError::BatchTooSmall(0));
    }
    if anchors.len() != positives.len() {
        return Err(ContrastiveError::LabelEmbedMismatch {
            n_labels: anchors.len(),
            n_embeds: positives.len(),
        });
    }

    let dim = anchors[0].len();
    // Dimension checks
    for v in anchors
        .iter()
        .chain(positives.iter())
        .chain(negatives.iter())
    {
        if v.len() != dim {
            return Err(ContrastiveError::DimensionMismatch {
                expected: dim,
                got: v.len(),
            });
        }
    }

    let t = config.temperature;
    let n = anchors.len();
    let mut total_loss = 0.0f32;

    for i in 0..n {
        let a = if config.normalize_embeddings {
            cl_normalize(&anchors[i])
        } else {
            anchors[i].clone()
        };
        let p = if config.normalize_embeddings {
            cl_normalize(&positives[i])
        } else {
            positives[i].clone()
        };

        let pos_dot: f32 = a.iter().zip(p.iter()).map(|(x, y)| x * y).sum();
        let pos_logit = pos_dot / t;

        let neg_logits: Vec<f32> = negatives
            .iter()
            .map(|nv| {
                let nv_n = if config.normalize_embeddings {
                    cl_normalize(nv)
                } else {
                    nv.clone()
                };
                let dot: f32 = a.iter().zip(nv_n.iter()).map(|(x, y)| x * y).sum();
                dot / t
            })
            .collect();

        // log(exp(pos) / (exp(pos) + sum_k exp(neg_k)))
        let all_logits: Vec<f32> = std::iter::once(pos_logit)
            .chain(neg_logits.iter().cloned())
            .collect();
        let lse = log_sum_exp(&all_logits);
        total_loss += -(pos_logit - lse);
    }

    Ok(total_loss / n as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Triplet loss
// ─────────────────────────────────────────────────────────────────────────────

/// Triplet margin loss: mean(max(0, d(a,p) - d(a,n) + margin)).
///
/// Uses L2 distance. All three slices must have the same length and dimension.
///
/// # Errors
/// - [`ContrastiveError::BatchTooSmall`] if slices are empty.
/// - [`ContrastiveError::DimensionMismatch`] on shape mismatch.
pub fn cl_triplet_loss(
    anchors: &[Vec<f32>],
    positives: &[Vec<f32>],
    negatives: &[Vec<f32>],
    margin: f32,
) -> Result<f32, ContrastiveError> {
    if anchors.is_empty() {
        return Err(ContrastiveError::BatchTooSmall(0));
    }
    if anchors.len() != positives.len() || anchors.len() != negatives.len() {
        return Err(ContrastiveError::DimensionMismatch {
            expected: anchors.len(),
            got: if anchors.len() != positives.len() {
                positives.len()
            } else {
                negatives.len()
            },
        });
    }

    let dim = anchors[0].len();
    let n = anchors.len();
    let mut total = 0.0f32;

    for i in 0..n {
        if positives[i].len() != dim || negatives[i].len() != dim || anchors[i].len() != dim {
            return Err(ContrastiveError::DimensionMismatch {
                expected: dim,
                got: if positives[i].len() != dim {
                    positives[i].len()
                } else if negatives[i].len() != dim {
                    negatives[i].len()
                } else {
                    anchors[i].len()
                },
            });
        }
        let d_pos = cl_l2_distance(&anchors[i], &positives[i])?;
        let d_neg = cl_l2_distance(&anchors[i], &negatives[i])?;
        let loss = (d_pos - d_neg + margin).max(0.0);
        total += loss;
    }

    Ok(total / n as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Supervised contrastive loss  (SupCon, Khosla et al. 2020)
// ─────────────────────────────────────────────────────────────────────────────

/// Supervised contrastive loss.
///
/// For each sample i, positives are all other samples with the same label.
/// Negatives are all samples with different labels. Samples with no positives
/// (unique label, or only one sample of that label) contribute 0 to the loss.
///
/// # Errors
/// - [`ContrastiveError::BatchTooSmall`] if fewer than 2 embeddings.
/// - [`ContrastiveError::LabelEmbedMismatch`] if `labels.len() != embeddings.len()`.
/// - [`ContrastiveError::InvalidTemperature`] if `temperature ≤ 0`.
/// - [`ContrastiveError::DimensionMismatch`] on shape mismatch.
pub fn cl_supcon_loss(
    embeddings: &[Vec<f32>],
    labels: &[usize],
    config: &ContrastiveLearningConfig,
) -> Result<f32, ContrastiveError> {
    config.validate()?;
    let n = embeddings.len();
    if n < 2 {
        return Err(ContrastiveError::BatchTooSmall(n));
    }
    if labels.len() != n {
        return Err(ContrastiveError::LabelEmbedMismatch {
            n_labels: labels.len(),
            n_embeds: n,
        });
    }

    let dim = embeddings[0].len();
    for e in embeddings.iter() {
        if e.len() != dim {
            return Err(ContrastiveError::DimensionMismatch {
                expected: dim,
                got: e.len(),
            });
        }
    }

    let t = config.temperature;
    // Optionally normalise
    let vecs: Vec<Vec<f32>> = embeddings
        .iter()
        .map(|v| {
            if config.normalize_embeddings {
                cl_normalize(v)
            } else {
                v.clone()
            }
        })
        .collect();

    // Precompute similarity matrix
    let mut sim = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let dot: f32 = vecs[i].iter().zip(vecs[j].iter()).map(|(x, y)| x * y).sum();
            sim[i * n + j] = dot / t;
        }
    }

    let mut total_loss = 0.0f32;
    let mut active = 0usize;

    for i in 0..n {
        // Collect positive indices (same label, not self)
        let positives: Vec<usize> = (0..n)
            .filter(|&j| j != i && labels[j] == labels[i])
            .collect();

        if positives.is_empty() {
            continue; // no positives — skip this anchor
        }
        active += 1;

        // Denominator: all k ≠ i
        let denom_logits: Vec<f32> = (0..n).filter(|&k| k != i).map(|k| sim[i * n + k]).collect();
        let lse = log_sum_exp(&denom_logits);

        // loss_i = -1/|P(i)| * sum_{p in P(i)} (sim(i,p) - log_sum_exp)
        let sum_pos: f32 = positives.iter().map(|&p| sim[i * n + p] - lse).sum::<f32>();
        total_loss += -sum_pos / positives.len() as f32;
    }

    if active == 0 {
        return Ok(0.0);
    }
    Ok(total_loss / active as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory queue  (MoCo-style)
// ─────────────────────────────────────────────────────────────────────────────

/// Fixed-size FIFO ring-buffer of embeddings for hard-negative mining.
#[derive(Debug, Clone)]
pub struct EmbeddingQueue {
    capacity: usize,
    dim: usize,
    entries: Vec<Vec<f32>>,
    labels: Vec<usize>,
    write_pos: usize,
    total_enqueued: usize,
}

impl EmbeddingQueue {
    /// Create a new empty queue with given capacity and embedding dimension.
    pub fn new(capacity: usize, dim: usize) -> Self {
        Self {
            capacity,
            dim,
            entries: Vec::with_capacity(capacity.min(4096)),
            labels: Vec::with_capacity(capacity.min(4096)),
            write_pos: 0,
            total_enqueued: 0,
        }
    }

    /// Enqueue a single embedding. Wraps around (circular overwrite) when full.
    ///
    /// # Errors
    /// Returns [`ContrastiveError::DimensionMismatch`] if embedding length ≠ dim.
    pub fn enqueue(&mut self, embedding: Vec<f32>, label: usize) -> Result<(), ContrastiveError> {
        if embedding.len() != self.dim {
            return Err(ContrastiveError::DimensionMismatch {
                expected: self.dim,
                got: embedding.len(),
            });
        }
        if self.entries.len() < self.capacity {
            self.entries.push(embedding);
            self.labels.push(label);
        } else {
            self.entries[self.write_pos] = embedding;
            self.labels[self.write_pos] = label;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.total_enqueued += 1;
        Ok(())
    }

    /// Enqueue a batch of embeddings.
    ///
    /// # Errors
    /// - [`ContrastiveError::LabelEmbedMismatch`] if lengths differ.
    /// - [`ContrastiveError::DimensionMismatch`] if any embedding has wrong dim.
    pub fn enqueue_batch(
        &mut self,
        embeddings: &[Vec<f32>],
        labels: &[usize],
    ) -> Result<(), ContrastiveError> {
        if embeddings.len() != labels.len() {
            return Err(ContrastiveError::LabelEmbedMismatch {
                n_labels: labels.len(),
                n_embeds: embeddings.len(),
            });
        }
        for (emb, &lbl) in embeddings.iter().zip(labels.iter()) {
            self.enqueue(emb.clone(), lbl)?;
        }
        Ok(())
    }

    /// Slice of all currently stored embeddings (up to `capacity`).
    pub fn as_slice(&self) -> &[Vec<f32>] {
        &self.entries
    }

    /// Slice of all currently stored labels.
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Current number of stored entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the queue holds no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// True when the queue is at capacity.
    pub fn is_full(&self) -> bool {
        self.entries.len() == self.capacity
    }

    /// Maximum number of entries the queue can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// References to embeddings whose label ≠ `exclude_label`.
    pub fn get_negatives(&self, exclude_label: usize) -> Vec<&Vec<f32>> {
        self.entries
            .iter()
            .zip(self.labels.iter())
            .filter(|(_, &lbl)| lbl != exclude_label)
            .map(|(emb, _)| emb)
            .collect()
    }

    /// Total number of embeddings ever enqueued (including overwritten ones).
    pub fn total_enqueued(&self) -> usize {
        self.total_enqueued
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hard negative mining
// ─────────────────────────────────────────────────────────────────────────────

/// Mine the `n_hard` hardest negatives (highest cosine similarity, different
/// label) for each anchor.
///
/// Returns a `Vec<Vec<usize>>` of length `anchors.len()`. Each inner vec
/// contains indices into `negatives`. If fewer than `n_hard` valid negatives
/// exist, all valid ones are returned.
///
/// # Errors
/// - [`ContrastiveError::BatchTooSmall`] if anchors or negatives are empty.
/// - [`ContrastiveError::LabelEmbedMismatch`] on anchor/label length mismatch.
/// - [`ContrastiveError::DimensionMismatch`] on shape mismatch.
pub fn cl_mine_hard_negatives(
    anchors: &[Vec<f32>],
    anchor_labels: &[usize],
    negatives: &[Vec<f32>],
    negative_labels: &[usize],
    n_hard: usize,
) -> Result<Vec<Vec<usize>>, ContrastiveError> {
    if anchors.is_empty() {
        return Err(ContrastiveError::BatchTooSmall(0));
    }
    if negatives.is_empty() {
        return Err(ContrastiveError::BatchTooSmall(0));
    }
    if anchors.len() != anchor_labels.len() {
        return Err(ContrastiveError::LabelEmbedMismatch {
            n_labels: anchor_labels.len(),
            n_embeds: anchors.len(),
        });
    }
    if negatives.len() != negative_labels.len() {
        return Err(ContrastiveError::LabelEmbedMismatch {
            n_labels: negative_labels.len(),
            n_embeds: negatives.len(),
        });
    }

    let dim = anchors[0].len();
    for v in anchors.iter().chain(negatives.iter()) {
        if v.len() != dim {
            return Err(ContrastiveError::DimensionMismatch {
                expected: dim,
                got: v.len(),
            });
        }
    }

    let n_hard_clamped = n_hard.max(1);
    let mut result = Vec::with_capacity(anchors.len());

    for (a, &a_lbl) in anchors.iter().zip(anchor_labels.iter()) {
        let a_norm = cl_normalize(a);
        // Collect (similarity, index) for all negatives with different label
        let mut sims: Vec<(f32, usize)> = negatives
            .iter()
            .zip(negative_labels.iter())
            .enumerate()
            .filter(|(_, (_, &n_lbl))| n_lbl != a_lbl)
            .map(|(idx, (nv, _))| {
                let nv_norm = cl_normalize(nv);
                let dot: f32 = a_norm.iter().zip(nv_norm.iter()).map(|(x, y)| x * y).sum();
                (dot, idx)
            })
            .collect();

        // Sort descending by similarity (hardest = most similar)
        sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        sims.truncate(n_hard_clamped);
        result.push(sims.into_iter().map(|(_, idx)| idx).collect());
    }

    Ok(result)
}

/// Mine semi-hard negatives for each anchor.
///
/// Semi-hard: d(a, n) > d(a, p) but d(a, n) < d(a, p) + margin.
/// Returns indices into `negatives`; empty inner vec when none qualify.
///
/// # Errors
/// - [`ContrastiveError::BatchTooSmall`] if any slice is empty.
/// - [`ContrastiveError::LabelEmbedMismatch`] if anchor/positive lengths differ.
/// - [`ContrastiveError::DimensionMismatch`] on shape mismatch.
pub fn cl_mine_semi_hard_negatives(
    anchors: &[Vec<f32>],
    positives: &[Vec<f32>],
    negatives: &[Vec<f32>],
    margin: f32,
) -> Result<Vec<Vec<usize>>, ContrastiveError> {
    if anchors.is_empty() || positives.is_empty() || negatives.is_empty() {
        return Err(ContrastiveError::BatchTooSmall(0));
    }
    if anchors.len() != positives.len() {
        return Err(ContrastiveError::LabelEmbedMismatch {
            n_labels: anchors.len(),
            n_embeds: positives.len(),
        });
    }

    let dim = anchors[0].len();
    for v in anchors
        .iter()
        .chain(positives.iter())
        .chain(negatives.iter())
    {
        if v.len() != dim {
            return Err(ContrastiveError::DimensionMismatch {
                expected: dim,
                got: v.len(),
            });
        }
    }

    let n = anchors.len();
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let d_pos = cl_l2_distance(&anchors[i], &positives[i])?;
        let semi_hard: Vec<usize> = negatives
            .iter()
            .enumerate()
            .filter_map(|(j, nv)| {
                let d_neg = cl_l2_distance(&anchors[i], nv).ok()?;
                // semi-hard: harder than positive but within margin
                if d_neg > d_pos && d_neg < d_pos + margin {
                    Some(j)
                } else {
                    None
                }
            })
            .collect();
        result.push(semi_hard);
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics and tracking
// ─────────────────────────────────────────────────────────────────────────────

/// Tracked statistics for a contrastive training run.
#[derive(Debug, Clone, PartialEq)]
pub struct ContrastiveStats {
    /// Arithmetic mean of loss values seen so far.
    pub mean_loss: f32,
    /// Mean cosine similarity among positive pairs.
    pub mean_positive_similarity: f32,
    /// Mean cosine similarity among negative pairs.
    pub mean_negative_similarity: f32,
    /// Alignment: mean positive-pair cosine similarity (higher = better).
    pub alignment: f32,
    /// Uniformity: -log(mean exp(-2 ‖z_i − z_j‖²)) (lower = more uniform).
    pub uniformity: f32,
    /// Number of pairs contributing to the current statistics.
    pub n_pairs: usize,
    /// Exponential moving average of loss (decay = 0.99).
    pub ema_loss: f32,
}

impl Default for ContrastiveStats {
    fn default() -> Self {
        Self {
            mean_loss: 0.0,
            mean_positive_similarity: 0.0,
            mean_negative_similarity: 0.0,
            alignment: 0.0,
            uniformity: 0.0,
            n_pairs: 0,
            ema_loss: 0.0,
        }
    }
}

/// Update a `ContrastiveStats` with one new observation.
///
/// Uses an EMA with decay = 0.99. `mean_loss`, `mean_positive_similarity`, and
/// `mean_negative_similarity` are updated as running means.
pub fn cl_update_stats(stats: &mut ContrastiveStats, loss: f32, pos_sim: f32, neg_sim: f32) {
    const DECAY: f32 = 0.99;
    stats.n_pairs += 1;
    let n = stats.n_pairs as f32;
    // Running mean (Welford-style incremental update)
    stats.mean_loss += (loss - stats.mean_loss) / n;
    stats.mean_positive_similarity += (pos_sim - stats.mean_positive_similarity) / n;
    stats.mean_negative_similarity += (neg_sim - stats.mean_negative_similarity) / n;
    // EMA
    stats.ema_loss = DECAY * stats.ema_loss + (1.0 - DECAY) * loss;
}

/// Alignment metric: mean cosine similarity between positive pairs.
///
/// Higher is better (perfect alignment = 1.0).
///
/// # Errors
/// - [`ContrastiveError::BatchTooSmall`] if slices are empty.
/// - [`ContrastiveError::LabelEmbedMismatch`] if lengths differ.
/// - [`ContrastiveError::DimensionMismatch`] on shape mismatch.
pub fn cl_alignment(
    positives_a: &[Vec<f32>],
    positives_b: &[Vec<f32>],
) -> Result<f32, ContrastiveError> {
    if positives_a.is_empty() {
        return Err(ContrastiveError::BatchTooSmall(0));
    }
    if positives_a.len() != positives_b.len() {
        return Err(ContrastiveError::LabelEmbedMismatch {
            n_labels: positives_a.len(),
            n_embeds: positives_b.len(),
        });
    }

    let n = positives_a.len();
    let mut total = 0.0f32;
    for i in 0..n {
        total += cl_cosine_sim(&positives_a[i], &positives_b[i])?;
    }
    Ok(total / n as f32)
}

/// Uniformity metric: -log(mean exp(-2 ‖z_i - z_j‖²)) over all i < j pairs.
///
/// Lower (more negative) = more uniformly distributed on the hypersphere.
/// For a single embedding the result is 0.0 (only one pair: the diagonal).
///
/// # Errors
/// - [`ContrastiveError::BatchTooSmall`] if `embeddings` is empty.
/// - [`ContrastiveError::DimensionMismatch`] on shape mismatch.
pub fn cl_uniformity(embeddings: &[Vec<f32>]) -> Result<f32, ContrastiveError> {
    if embeddings.is_empty() {
        return Err(ContrastiveError::BatchTooSmall(0));
    }
    let n = embeddings.len();
    let dim = embeddings[0].len();
    for e in embeddings.iter() {
        if e.len() != dim {
            return Err(ContrastiveError::DimensionMismatch {
                expected: dim,
                got: e.len(),
            });
        }
    }

    if n == 1 {
        // -log(exp(0)) = 0
        return Ok(0.0);
    }

    // Collect all i < j squared distances
    let mut sum_exp = 0.0f64;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let sq: f32 = embeddings[i]
                .iter()
                .zip(embeddings[j].iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum();
            sum_exp += (-2.0 * sq as f64).exp();
            count += 1;
        }
    }

    let mean_exp = sum_exp / count as f64;
    Ok(-(mean_exp.ln() as f32))
}

/// Format a `ContrastiveStats` as a human-readable string.
pub fn cl_format_stats(stats: &ContrastiveStats) -> String {
    format!(
        "ContrastiveStats {{ loss={:.4} ema={:.4} pos_sim={:.4} neg_sim={:.4} \
         align={:.4} uniform={:.4} n_pairs={} }}",
        stats.mean_loss,
        stats.ema_loss,
        stats.mean_positive_similarity,
        stats.mean_negative_similarity,
        stats.alignment,
        stats.uniformity,
        stats.n_pairs,
    )
}

/// Format a `ContrastiveLearningConfig` as a human-readable string.
pub fn cl_format_config(config: &ContrastiveLearningConfig) -> String {
    format!(
        "ContrastiveLearningConfig {{ temp={} dim={} queue={} mom={} norm={} \
         hard_neg={} margin={} n_neg={} }}",
        config.temperature,
        config.embedding_dim,
        config.queue_size,
        config.momentum,
        config.normalize_embeddings,
        config.hard_negative_mining,
        config.margin,
        config.n_negatives,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: roughly equal within tolerance
    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // Build a unit vector along axis `axis` of `dim` dimensions.
    fn axis_vec(dim: usize, axis: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[axis] = 1.0;
        v
    }

    // Build orthogonal embeddings
    fn ortho_pair(dim: usize) -> (Vec<f32>, Vec<f32>) {
        (axis_vec(dim, 0), axis_vec(dim, 1))
    }

    // ── ContrastiveLearningConfig ────────────────────────────────────────────

    #[test]
    fn test_config_default_values() {
        let cfg = ContrastiveLearningConfig::default();
        assert!((cfg.temperature - 0.07).abs() < 1e-6);
        assert_eq!(cfg.embedding_dim, 256);
        assert_eq!(cfg.queue_size, 4096);
        assert!((cfg.momentum - 0.999).abs() < 1e-6);
        assert!(cfg.normalize_embeddings);
        assert!(!cfg.hard_negative_mining);
        assert!((cfg.margin - 0.5).abs() < 1e-6);
        assert_eq!(cfg.n_negatives, 0);
    }

    #[test]
    fn test_config_validate_valid() {
        assert!(ContrastiveLearningConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_temperature() {
        let cfg = ContrastiveLearningConfig {
            temperature: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ContrastiveError::InvalidTemperature(_))
        ));
    }

    #[test]
    fn test_config_validate_negative_temperature() {
        let cfg = ContrastiveLearningConfig {
            temperature: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ContrastiveError::InvalidTemperature(_))
        ));
    }

    #[test]
    fn test_config_validate_zero_dim() {
        let cfg = ContrastiveLearningConfig {
            embedding_dim: 0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ContrastiveError::InvalidConfig(_))
        ));
    }

    // ── cl_normalize ─────────────────────────────────────────────────────────

    #[test]
    fn test_normalize_unit_vector_unchanged() {
        let v = vec![1.0f32, 0.0, 0.0];
        let n = cl_normalize(&v);
        assert!(approx(n[0], 1.0, 1e-6));
        assert!(approx(n[1], 0.0, 1e-6));
    }

    #[test]
    fn test_normalize_zero_vector_returns_zeros() {
        let v = vec![0.0f32; 4];
        let n = cl_normalize(&v);
        assert!(n.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_normalize_arbitrary_vector() {
        let v = vec![3.0f32, 4.0];
        let n = cl_normalize(&v);
        let norm: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(approx(norm, 1.0, 1e-6));
    }

    // ── cl_cosine_sim ─────────────────────────────────────────────────────────

    #[test]
    fn test_cosine_sim_identical() {
        let v = vec![1.0f32, 2.0, 3.0];
        let s = cl_cosine_sim(&v, &v).unwrap();
        assert!(approx(s, 1.0, 1e-5));
    }

    #[test]
    fn test_cosine_sim_opposite() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        let s = cl_cosine_sim(&a, &b).unwrap();
        assert!(approx(s, -1.0, 1e-5));
    }

    #[test]
    fn test_cosine_sim_orthogonal() {
        let (a, b) = ortho_pair(4);
        let s = cl_cosine_sim(&a, &b).unwrap();
        assert!(approx(s, 0.0, 1e-5));
    }

    #[test]
    fn test_cosine_sim_empty_error() {
        let res = cl_cosine_sim(&[], &[]);
        assert!(matches!(
            res,
            Err(ContrastiveError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_cosine_sim_mismatch_error() {
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32];
        assert!(matches!(
            cl_cosine_sim(&a, &b),
            Err(ContrastiveError::DimensionMismatch { .. })
        ));
    }

    // ── cl_dot ───────────────────────────────────────────────────────────────

    #[test]
    fn test_dot_correct() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let d = cl_dot(&a, &b).unwrap();
        assert!(approx(d, 32.0, 1e-5));
    }

    #[test]
    fn test_dot_mismatch_error() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32];
        assert!(matches!(
            cl_dot(&a, &b),
            Err(ContrastiveError::DimensionMismatch { .. })
        ));
    }

    // ── cl_l2_distance ───────────────────────────────────────────────────────

    #[test]
    fn test_l2_distance_identical() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert!(approx(cl_l2_distance(&v, &v).unwrap(), 0.0, 1e-6));
    }

    #[test]
    fn test_l2_distance_known_pair() {
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        assert!(approx(cl_l2_distance(&a, &b).unwrap(), 5.0, 1e-5));
    }

    #[test]
    fn test_l2_distance_mismatch_error() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32];
        assert!(matches!(
            cl_l2_distance(&a, &b),
            Err(ContrastiveError::DimensionMismatch { .. })
        ));
    }

    // ── cl_similarity_matrix ─────────────────────────────────────────────────

    #[test]
    fn test_similarity_matrix_1x1() {
        let v = vec![vec![1.0f32, 0.0, 0.0]];
        let m = cl_similarity_matrix(&v, true).unwrap();
        assert_eq!(m.len(), 1);
        assert!(approx(m[0], 1.0, 1e-5));
    }

    #[test]
    fn test_similarity_matrix_2x2_diagonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let vecs = vec![a, b];
        let m = cl_similarity_matrix(&vecs, true).unwrap();
        assert!(approx(m[0], 1.0, 1e-5)); // (0,0)
        assert!(approx(m[3], 1.0, 1e-5)); // (1,1)
        assert!(approx(m[1], 0.0, 1e-5)); // (0,1) orthogonal
    }

    #[test]
    fn test_similarity_matrix_empty_error() {
        assert!(matches!(
            cl_similarity_matrix(&[], true),
            Err(ContrastiveError::BatchTooSmall(_))
        ));
    }

    #[test]
    fn test_similarity_matrix_dim_mismatch_error() {
        let vecs = vec![vec![1.0f32, 0.0], vec![1.0f32]];
        assert!(matches!(
            cl_similarity_matrix(&vecs, false),
            Err(ContrastiveError::DimensionMismatch { .. })
        ));
    }

    // ── cl_nt_xent_loss ──────────────────────────────────────────────────────

    #[test]
    fn test_nt_xent_odd_batch_error() {
        let cfg = ContrastiveLearningConfig::default();
        let embs = vec![vec![1.0f32, 0.0]; 3];
        assert!(matches!(
            cl_nt_xent_loss(&embs, &cfg),
            Err(ContrastiveError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_nt_xent_too_small_error() {
        let cfg = ContrastiveLearningConfig::default();
        let embs = vec![vec![1.0f32, 0.0]];
        assert!(matches!(
            cl_nt_xent_loss(&embs, &cfg),
            Err(ContrastiveError::BatchTooSmall(_))
        ));
    }

    #[test]
    fn test_nt_xent_identical_pair_low_loss() {
        // When both embeddings in the only pair are identical (collinear),
        // the positive sim = 1, which is the maximum, so loss is minimised.
        let cfg = ContrastiveLearningConfig::default();
        let a = vec![1.0f32, 0.0];
        let embs = vec![a.clone(), a];
        let loss = cl_nt_xent_loss(&embs, &cfg).unwrap();
        // With only one negative (the pair partner), the denominator = just the pos itself,
        // so loss = -log(1) = 0. With 2 samples (one pair) there are no extra negatives.
        // The only non-self sample is the partner => loss → 0.
        assert!(loss.abs() < 1e-3, "loss={loss}");
    }

    #[test]
    fn test_nt_xent_orthogonal_pair_higher_loss() {
        let cfg = ContrastiveLearningConfig::default();
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];
        // orthogonal positive pair — positive sim = 0
        let embs = vec![a, b];
        let loss_orth = cl_nt_xent_loss(&embs, &cfg).unwrap();
        // identical pair has loss ≈ 0; orthogonal should be larger
        let a2 = vec![1.0f32, 0.0, 0.0, 0.0];
        let embs_ident = vec![a2.clone(), a2];
        let loss_ident = cl_nt_xent_loss(&embs_ident, &cfg).unwrap();
        assert!(
            loss_orth >= loss_ident,
            "orthogonal loss {loss_orth} should be >= identical loss {loss_ident}"
        );
    }

    #[test]
    fn test_nt_xent_larger_batch_positive() {
        let cfg = ContrastiveLearningConfig::default();
        // 4 embeddings = 2 positive pairs
        let embs = vec![
            vec![1.0f32, 0.0, 0.0],
            vec![0.9f32, 0.1, 0.0],
            vec![0.0f32, 1.0, 0.0],
            vec![0.0f32, 0.9, 0.1],
        ];
        let loss = cl_nt_xent_loss(&embs, &cfg).unwrap();
        assert!(loss > 0.0, "loss={loss}");
    }

    #[test]
    fn test_nt_xent_temperature_effect() {
        // Higher temperature → softer distribution → lower raw loss magnitude
        let embs = vec![
            vec![1.0f32, 0.0],
            vec![0.0f32, 1.0], // orthogonal — deliberately hard pair
        ];
        let cfg_low_t = ContrastiveLearningConfig {
            temperature: 0.01,
            ..Default::default()
        };
        let cfg_high_t = ContrastiveLearningConfig {
            temperature: 1.0,
            ..Default::default()
        };
        let loss_low = cl_nt_xent_loss(&embs, &cfg_low_t).unwrap();
        let loss_high = cl_nt_xent_loss(&embs, &cfg_high_t).unwrap();
        // Higher T → lower loss magnitude
        assert!(
            loss_high <= loss_low + 1e-3,
            "high-T loss {loss_high} should be <= low-T loss {loss_low}"
        );
    }

    // ── cl_info_nce_loss ─────────────────────────────────────────────────────

    #[test]
    fn test_info_nce_empty_anchors_error() {
        let cfg = ContrastiveLearningConfig::default();
        let res = cl_info_nce_loss(&[], &[], &[vec![1.0f32, 0.0]], &cfg);
        assert!(matches!(res, Err(ContrastiveError::BatchTooSmall(_))));
    }

    #[test]
    fn test_info_nce_empty_negatives_error() {
        let cfg = ContrastiveLearningConfig::default();
        let a = vec![vec![1.0f32, 0.0]];
        let p = vec![vec![1.0f32, 0.0]];
        let res = cl_info_nce_loss(&a, &p, &[], &cfg);
        assert!(matches!(res, Err(ContrastiveError::BatchTooSmall(_))));
    }

    #[test]
    fn test_info_nce_anchor_positive_mismatch_error() {
        let cfg = ContrastiveLearningConfig::default();
        let a = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let p = vec![vec![1.0f32, 0.0]];
        let n = vec![vec![-1.0f32, 0.0]];
        let res = cl_info_nce_loss(&a, &p, &n, &cfg);
        assert!(matches!(
            res,
            Err(ContrastiveError::LabelEmbedMismatch { .. })
        ));
    }

    #[test]
    fn test_info_nce_perfect_separation_low_loss() {
        // pos sim ~ 1, neg sim ~ -1 → near-zero loss
        let cfg = ContrastiveLearningConfig {
            temperature: 0.5,
            normalize_embeddings: true,
            ..Default::default()
        };
        let a = vec![vec![1.0f32, 0.0]];
        let p = vec![vec![1.0f32, 0.0]]; // identical
        let n = vec![vec![-1.0f32, 0.0]]; // opposite
        let loss = cl_info_nce_loss(&a, &p, &n, &cfg).unwrap();
        assert!(loss < 0.1, "loss={loss}");
    }

    #[test]
    fn test_info_nce_worst_case_high_loss() {
        // pos sim == neg sim → high loss
        let cfg = ContrastiveLearningConfig {
            temperature: 0.5,
            normalize_embeddings: true,
            ..Default::default()
        };
        let a = vec![vec![1.0f32, 0.0]];
        let p = vec![vec![0.0f32, 1.0]]; // orthogonal to anchor
        let n = vec![vec![0.0f32, 1.0]]; // same as positive
        let loss_hard = cl_info_nce_loss(&a, &p, &n, &cfg).unwrap();

        let p2 = vec![vec![1.0f32, 0.0]]; // identical to anchor
        let loss_easy = cl_info_nce_loss(&a, &p2, &n, &cfg).unwrap();
        assert!(loss_hard >= loss_easy, "hard={loss_hard} easy={loss_easy}");
    }

    // ── cl_triplet_loss ───────────────────────────────────────────────────────

    #[test]
    fn test_triplet_zero_when_margin_satisfied() {
        // d(a,p) << d(a,n) - margin
        let a = vec![vec![0.0f32, 0.0]];
        let p = vec![vec![0.1f32, 0.0]];
        let n = vec![vec![10.0f32, 0.0]];
        let loss = cl_triplet_loss(&a, &p, &n, 0.5).unwrap();
        assert!(approx(loss, 0.0, 1e-5));
    }

    #[test]
    fn test_triplet_positive_when_violated() {
        // d(a,p) > d(a,n) - margin
        let a = vec![vec![0.0f32]];
        let p = vec![vec![5.0f32]];
        let n = vec![vec![1.0f32]];
        let loss = cl_triplet_loss(&a, &p, &n, 0.5).unwrap();
        assert!(loss > 0.0, "loss={loss}");
    }

    #[test]
    fn test_triplet_dimension_mismatch_error() {
        let a = vec![vec![1.0f32, 0.0]];
        let p = vec![vec![1.0f32, 0.0]];
        let n = vec![vec![1.0f32]]; // wrong dim
        assert!(matches!(
            cl_triplet_loss(&a, &p, &n, 0.5),
            Err(ContrastiveError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_triplet_empty_error() {
        assert!(matches!(
            cl_triplet_loss(&[], &[], &[], 0.5),
            Err(ContrastiveError::BatchTooSmall(_))
        ));
    }

    #[test]
    fn test_triplet_batch_mean() {
        // Two triplets — verify mean
        let a = vec![vec![0.0f32], vec![0.0f32]];
        let p = vec![vec![0.0f32], vec![5.0f32]]; // 2nd violated
        let n = vec![vec![10.0f32], vec![1.0f32]];
        let loss = cl_triplet_loss(&a, &p, &n, 0.5).unwrap();
        // first triplet: d_pos=0, d_neg=10, 0-10+0.5 < 0 → 0
        // second triplet: d_pos=5, d_neg=1, 5-1+0.5=4.5 > 0
        assert!(approx(loss, 2.25, 0.01), "loss={loss}");
    }

    // ── cl_supcon_loss ────────────────────────────────────────────────────────

    #[test]
    fn test_supcon_all_same_label_no_positives_after_self() {
        // With only 1 sample per label-group (n=1, all different labels), loss = 0
        let cfg = ContrastiveLearningConfig::default();
        let embs = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let labels = vec![0usize, 1];
        // each has 0 positives → loss = 0
        let loss = cl_supcon_loss(&embs, &labels, &cfg).unwrap();
        assert!(approx(loss, 0.0, 1e-5));
    }

    #[test]
    fn test_supcon_unique_labels_identity() {
        // unique labels → 0 positives → graceful 0.0
        let cfg = ContrastiveLearningConfig::default();
        let embs = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0], vec![1.0f32, 1.0]];
        let labels = vec![0usize, 1, 2];
        let loss = cl_supcon_loss(&embs, &labels, &cfg).unwrap();
        assert!(approx(loss, 0.0, 1e-5));
    }

    #[test]
    fn test_supcon_two_labels_two_samples_each() {
        let cfg = ContrastiveLearningConfig {
            temperature: 0.5,
            normalize_embeddings: true,
            ..Default::default()
        };
        // 4 embeddings: 2 per label
        let embs = vec![
            vec![1.0f32, 0.0],
            vec![0.9f32, 0.1],
            vec![0.0f32, 1.0],
            vec![0.1f32, 0.9],
        ];
        let labels = vec![0usize, 0, 1, 1];
        let loss = cl_supcon_loss(&embs, &labels, &cfg).unwrap();
        assert!(loss >= 0.0, "loss={loss}");
    }

    #[test]
    fn test_supcon_label_embed_mismatch_error() {
        let cfg = ContrastiveLearningConfig::default();
        let embs = vec![vec![1.0f32, 0.0]; 3];
        let labels = vec![0usize, 1];
        assert!(matches!(
            cl_supcon_loss(&embs, &labels, &cfg),
            Err(ContrastiveError::LabelEmbedMismatch { .. })
        ));
    }

    #[test]
    fn test_supcon_batch_too_small_error() {
        let cfg = ContrastiveLearningConfig::default();
        let embs = vec![vec![1.0f32, 0.0]];
        let labels = vec![0usize];
        assert!(matches!(
            cl_supcon_loss(&embs, &labels, &cfg),
            Err(ContrastiveError::BatchTooSmall(_))
        ));
    }

    // ── EmbeddingQueue ────────────────────────────────────────────────────────

    #[test]
    fn test_queue_enqueue_and_wrap() {
        let mut q = EmbeddingQueue::new(2, 2);
        q.enqueue(vec![1.0f32, 0.0], 0).unwrap();
        q.enqueue(vec![0.0f32, 1.0], 1).unwrap();
        assert!(q.is_full());
        // Wrap: overwrite slot 0
        q.enqueue(vec![0.5f32, 0.5], 2).unwrap();
        assert_eq!(q.len(), 2); // still 2
        assert_eq!(q.total_enqueued(), 3);
    }

    #[test]
    fn test_queue_dim_mismatch_error() {
        let mut q = EmbeddingQueue::new(4, 3);
        let res = q.enqueue(vec![1.0f32, 0.0], 0);
        assert!(matches!(
            res,
            Err(ContrastiveError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_queue_get_negatives_excludes_same_label() {
        let mut q = EmbeddingQueue::new(10, 2);
        q.enqueue(vec![1.0f32, 0.0], 0).unwrap();
        q.enqueue(vec![0.0f32, 1.0], 1).unwrap();
        q.enqueue(vec![0.5f32, 0.5], 0).unwrap();

        let negs = q.get_negatives(0);
        assert_eq!(negs.len(), 1); // only label=1
    }

    #[test]
    fn test_queue_enqueue_batch_mismatch_error() {
        let mut q = EmbeddingQueue::new(10, 2);
        let embs = vec![vec![1.0f32, 0.0]; 3];
        let labels = vec![0usize, 1]; // len mismatch
        assert!(matches!(
            q.enqueue_batch(&embs, &labels),
            Err(ContrastiveError::LabelEmbedMismatch { .. })
        ));
    }

    #[test]
    fn test_queue_is_empty_initially() {
        let q = EmbeddingQueue::new(4, 2);
        assert!(q.is_empty());
        assert!(!q.is_full());
    }

    #[test]
    fn test_queue_enqueue_batch_ok() {
        let mut q = EmbeddingQueue::new(10, 2);
        let embs = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let labels = vec![0usize, 1];
        q.enqueue_batch(&embs, &labels).unwrap();
        assert_eq!(q.len(), 2);
    }

    // ── cl_mine_hard_negatives ────────────────────────────────────────────────

    #[test]
    fn test_mine_hard_negatives_count() {
        let anchors = vec![vec![1.0f32, 0.0]];
        let a_labels = vec![0usize];
        let negatives = vec![
            vec![0.9f32, 0.1],  // closest
            vec![0.0f32, 1.0],  // orthogonal
            vec![-1.0f32, 0.0], // farthest
        ];
        let n_labels = vec![1usize, 1, 1];
        let result = cl_mine_hard_negatives(&anchors, &a_labels, &negatives, &n_labels, 2).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2);
    }

    #[test]
    fn test_mine_hard_negatives_label_filtering() {
        let anchors = vec![vec![1.0f32, 0.0]];
        let a_labels = vec![0usize];
        let negatives = vec![
            vec![0.9f32, 0.1],  // label 0 — same as anchor → excluded
            vec![-1.0f32, 0.0], // label 1 — valid negative
        ];
        let n_labels = vec![0usize, 1];
        let result = cl_mine_hard_negatives(&anchors, &a_labels, &negatives, &n_labels, 5).unwrap();
        // only one valid negative (label 1)
        assert_eq!(result[0].len(), 1);
        assert_eq!(result[0][0], 1); // index of the label-1 entry
    }

    #[test]
    fn test_mine_hard_negatives_empty_anchors_error() {
        let res = cl_mine_hard_negatives(&[], &[], &[vec![1.0f32]], &[0usize], 1);
        assert!(matches!(res, Err(ContrastiveError::BatchTooSmall(_))));
    }

    #[test]
    fn test_mine_hard_negatives_hardest_first() {
        // Anchor = [1, 0]. Negatives: [0.9,0.1] sim≈0.99, [-1,0] sim=-1
        // With n_hard=1 → should pick index 0
        let anchors = vec![vec![1.0f32, 0.0]];
        let a_labels = vec![0usize];
        let negatives = vec![vec![-1.0f32, 0.0], vec![0.9f32, 0.1]];
        let n_labels = vec![1usize, 1];
        let result = cl_mine_hard_negatives(&anchors, &a_labels, &negatives, &n_labels, 1).unwrap();
        assert_eq!(result[0].len(), 1);
        assert_eq!(result[0][0], 1); // index 1 has the higher cosine similarity
    }

    // ── cl_mine_semi_hard_negatives ───────────────────────────────────────────

    #[test]
    fn test_mine_semi_hard_negatives_basic() {
        // anchor=[0], positive=[1] (d=1), negatives: [2] (d=2, semi-hard if margin>1), [5] (d=5, too far)
        let a = vec![vec![0.0f32]];
        let p = vec![vec![1.0f32]];
        let n = vec![vec![2.0f32], vec![5.0f32]];
        let margin = 2.0; // d_pos + margin = 3
        let result = cl_mine_semi_hard_negatives(&a, &p, &n, margin).unwrap();
        assert_eq!(result.len(), 1);
        // n[0]: d=2 > d_pos=1, 2 < 1+2=3 → semi-hard
        // n[1]: d=5, 5 >= 3 → not semi-hard
        assert!(result[0].contains(&0));
        assert!(!result[0].contains(&1));
    }

    #[test]
    fn test_mine_semi_hard_negatives_empty_error() {
        let res = cl_mine_semi_hard_negatives(&[], &[], &[vec![1.0f32]], 0.5);
        assert!(matches!(res, Err(ContrastiveError::BatchTooSmall(_))));
    }

    #[test]
    fn test_mine_semi_hard_negatives_none_qualify() {
        // All negatives too far
        let a = vec![vec![0.0f32]];
        let p = vec![vec![0.1f32]]; // d_pos = 0.1
        let n = vec![vec![100.0f32]]; // d = 100, far beyond d_pos + margin=0.6
        let result = cl_mine_semi_hard_negatives(&a, &p, &n, 0.5).unwrap();
        assert!(result[0].is_empty());
    }

    // ── cl_alignment ─────────────────────────────────────────────────────────

    #[test]
    fn test_alignment_identical_pairs() {
        let v = vec![1.0f32, 0.0, 0.0];
        let a = vec![v.clone(), v.clone()];
        let b = vec![v.clone(), v.clone()];
        let align = cl_alignment(&a, &b).unwrap();
        assert!(approx(align, 1.0, 1e-5));
    }

    #[test]
    fn test_alignment_orthogonal_pairs() {
        let e1 = vec![1.0f32, 0.0];
        let e2 = vec![0.0f32, 1.0];
        let a = vec![e1.clone()];
        let b = vec![e2.clone()];
        let align = cl_alignment(&a, &b).unwrap();
        assert!(approx(align, 0.0, 1e-5));
    }

    #[test]
    fn test_alignment_empty_error() {
        let res = cl_alignment(&[], &[]);
        assert!(matches!(res, Err(ContrastiveError::BatchTooSmall(_))));
    }

    #[test]
    fn test_alignment_mismatch_error() {
        let a = vec![vec![1.0f32, 0.0], vec![0.0f32, 1.0]];
        let b = vec![vec![1.0f32, 0.0]];
        assert!(matches!(
            cl_alignment(&a, &b),
            Err(ContrastiveError::LabelEmbedMismatch { .. })
        ));
    }

    // ── cl_uniformity ─────────────────────────────────────────────────────────

    #[test]
    fn test_uniformity_single_embedding_zero() {
        let embs = vec![vec![1.0f32, 0.0]];
        let u = cl_uniformity(&embs).unwrap();
        assert!(approx(u, 0.0, 1e-5));
    }

    #[test]
    fn test_uniformity_many_embeddings_negative() {
        // Spread embeddings on the unit circle.
        // uniformity = -log(mean exp(-2||z_i - z_j||^2)).
        // For spread points, mean_exp < 1, so -log(mean_exp) > 0.
        // The metric is called "negative" because lower = more uniform.
        let embs = vec![
            vec![1.0f32, 0.0],
            vec![-1.0f32, 0.0],
            vec![0.0f32, 1.0],
            vec![0.0f32, -1.0],
        ];
        let u = cl_uniformity(&embs).unwrap();
        // Spread embeddings → large positive value (far from 0)
        assert!(u > 0.0, "uniformity={u}");
    }

    #[test]
    fn test_uniformity_empty_error() {
        assert!(matches!(
            cl_uniformity(&[]),
            Err(ContrastiveError::BatchTooSmall(_))
        ));
    }

    // ── cl_update_stats ───────────────────────────────────────────────────────

    #[test]
    fn test_update_stats_ema_decay() {
        let mut stats = ContrastiveStats::default();
        cl_update_stats(&mut stats, 1.0, 0.8, 0.2);
        // After first update, ema = 0.99*0 + 0.01*1.0 = 0.01
        assert!(approx(stats.ema_loss, 0.01, 1e-5));
        cl_update_stats(&mut stats, 1.0, 0.8, 0.2);
        // ema = 0.99*0.01 + 0.01*1.0 = 0.0099 + 0.01 = 0.0199
        assert!(approx(stats.ema_loss, 0.0199, 1e-4));
    }

    #[test]
    fn test_update_stats_mean_loss() {
        let mut stats = ContrastiveStats::default();
        cl_update_stats(&mut stats, 2.0, 0.5, 0.1);
        cl_update_stats(&mut stats, 4.0, 0.5, 0.1);
        assert!(approx(stats.mean_loss, 3.0, 1e-5));
    }

    #[test]
    fn test_update_stats_n_pairs_increment() {
        let mut stats = ContrastiveStats::default();
        for _ in 0..5 {
            cl_update_stats(&mut stats, 1.0, 0.5, 0.3);
        }
        assert_eq!(stats.n_pairs, 5);
    }

    // ── formatting ────────────────────────────────────────────────────────────

    #[test]
    fn test_format_stats_non_empty() {
        let stats = ContrastiveStats::default();
        let s = cl_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("ContrastiveStats"));
    }

    #[test]
    fn test_format_config_non_empty() {
        let cfg = ContrastiveLearningConfig::default();
        let s = cl_format_config(&cfg);
        assert!(!s.is_empty());
        assert!(s.contains("ContrastiveLearningConfig"));
    }

    // ── PRNG smoke test ───────────────────────────────────────────────────────

    #[test]
    fn test_xorshift_non_zero() {
        let mut state = 12345u64;
        for _ in 0..100 {
            assert_ne!(xorshift64(&mut state), 0);
        }
    }

    #[test]
    fn test_xorshift_f32_range() {
        let mut state = 42u64;
        for _ in 0..100 {
            let v = xorshift_f32(&mut state);
            assert!((0.0..=1.0).contains(&v));
        }
    }

    // ── Integration: queue → info_nce ─────────────────────────────────────────

    #[test]
    fn test_queue_then_info_nce() {
        let mut q = EmbeddingQueue::new(16, 2);
        let neg_embs = vec![vec![-1.0f32, 0.0], vec![0.0f32, -1.0]];
        q.enqueue_batch(&neg_embs, &[1usize, 1]).unwrap();

        let anchors = vec![vec![1.0f32, 0.0]];
        let positives = vec![vec![1.0f32, 0.0]];
        let negatives: Vec<Vec<f32>> = q.as_slice().to_vec();

        let cfg = ContrastiveLearningConfig {
            temperature: 0.5,
            normalize_embeddings: true,
            ..Default::default()
        };
        let loss = cl_info_nce_loss(&anchors, &positives, &negatives, &cfg).unwrap();
        assert!(loss >= 0.0, "loss={loss}");
    }

    #[test]
    fn test_triplet_with_hard_negatives() {
        let anchors = vec![vec![1.0f32, 0.0]];
        let a_labels = vec![0usize];
        let neg_pool = vec![
            vec![0.9f32, 0.1],  // hardest
            vec![-1.0f32, 0.0], // easiest
        ];
        let n_labels = vec![1usize, 1];
        let hard_idxs =
            cl_mine_hard_negatives(&anchors, &a_labels, &neg_pool, &n_labels, 1).unwrap();
        let hard_neg = neg_pool[hard_idxs[0][0]].clone();

        let positives = vec![vec![1.0f32, 0.0]];
        let negatives = vec![hard_neg];
        // d(a,p)=0, d(a,n)≈0.14, margin=0.5 → loss > 0
        let loss = cl_triplet_loss(&anchors, &positives, &negatives, 0.5).unwrap();
        assert!(loss > 0.0, "loss={loss}");
    }
}
