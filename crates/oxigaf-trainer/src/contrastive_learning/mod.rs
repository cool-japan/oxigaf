//! Contrastive learning for identity-discriminative facial embeddings.
//!
//! Implements NT-Xent (SimCLR), InfoNCE, triplet, and supervised contrastive
//! losses together with a MoCo-style memory queue, hard/semi-hard negative
//! mining, and alignment/uniformity metrics.
//!
//! All functions use the `cl_` prefix to avoid collisions with names already
//! exported from [`contrastive_loss`](super::contrastive_loss).

use thiserror::Error;

use crate::feature_bank::MomentumEncoder;

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

/// Seeded xorshift64 step, the only randomness in this module.
///
/// [`cl_uniformity_sampled`] is the sole caller: it needs *index* draws to
/// pick `max_pairs` embedding pairs reproducibly from a caller-supplied seed.
/// A companion `xorshift_f32` used to sit here too, but nothing in this module
/// draws a float — every other estimator walks all pairs deterministically —
/// so it was removed rather than kept alive with `#[allow(dead_code)]`.
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
        // `queue_size` feeds `EmbeddingQueue::from_config`, whose own
        // constructor rejects `capacity == 0` (it would panic on first
        // enqueue otherwise — see `EmbeddingQueue::new`). Rejecting it here
        // too lets a caller catch a bad config before ever touching a
        // queue.
        if self.queue_size == 0 {
            return Err(ContrastiveError::InvalidConfig(
                "queue_size must be > 0".into(),
            ));
        }
        // `momentum` feeds `Self::new_momentum_encoder`, whose underlying
        // `MomentumEncoder::new` rejects values outside `[0, 1)`. Checking
        // it here too surfaces a bad config before a key-encoder is ever
        // constructed from it.
        if !(0.0..1.0).contains(&self.momentum) {
            return Err(ContrastiveError::InvalidConfig(format!(
                "momentum must be in [0, 1), got {}",
                self.momentum
            )));
        }
        Ok(())
    }

    /// Construct the MoCo-style momentum key-encoder the module doc
    /// advertises ("together with a MoCo-style memory queue"), seeded with
    /// `initial_weights` and using `self.momentum` as the EMA coefficient.
    ///
    /// `momentum` was previously declared on this config and documented
    /// ("Momentum for key-encoder update") but read by nothing in this
    /// module — no momentum encoder was ever constructed from it. This
    /// method wires it to the general-purpose [`MomentumEncoder`] the crate
    /// already provides (see [`crate::feature_bank::MomentumEncoder`]),
    /// rather than duplicating that implementation here.
    ///
    /// # Errors
    /// Returns [`ContrastiveError::InvalidConfig`] if `self.momentum` is not
    /// in `[0, 1)` (propagated from [`MomentumEncoder::new`]; also checked
    /// by [`Self::validate`]).
    pub fn new_momentum_encoder(
        &self,
        initial_weights: Vec<f32>,
    ) -> Result<MomentumEncoder, ContrastiveError> {
        MomentumEncoder::new(initial_weights, self.momentum)
            .map_err(|e| ContrastiveError::InvalidConfig(format!("momentum encoder config: {e}")))
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
    // `is_infinite()` is true for BOTH +inf and -inf, so a bare
    // `if max.is_infinite() { return f32::NEG_INFINITY }` (the previous
    // code) silently mapped a genuine +inf logit (reachable with a small
    // `temperature` on unnormalized embeddings) to -inf instead of the
    // mathematically correct +inf, corrupting the loss's sign rather than
    // surfacing the overflow. Handle the two cases explicitly instead:
    // - All-`-inf` input (empty softmax support / every logit underflowed):
    //   `log(sum(exp(-inf))) = log(0) = -inf`.
    // - Any `+inf` logit present: `log(sum(exp(...))) = +inf` regardless of
    //   the other terms, and computing `(x - max)` for the max element
    //   itself would be `inf - inf = NaN`, so this must be special-cased
    //   rather than falling through to the general formula below.
    if max == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if max == f32::INFINITY {
        return f32::INFINITY;
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

    // Optionally normalise all embeddings. Previously this unconditionally
    // normalized regardless of `config.normalize_embeddings`, unlike its
    // siblings `cl_info_nce_loss` and `cl_supcon_loss`, which both branch on
    // it — a caller setting `normalize_embeddings: false` to work in an
    // unnormalized space got it silently overridden for NT-Xent only, so
    // the three losses computed on different geometries from one shared
    // config.
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

        // Denominator: the positive itself, plus this anchor's negatives
        // (every other sample: k != i, k != i_pos). `config.n_negatives`
        // caps how many of those negatives are used, matching its doc
        // ("Negatives per positive (NT-Xent). Default: uses full batch").
        // `0` (the default) keeps the full batch. The cap keeps the first
        // `n_negatives` negatives in batch order rather than drawing a
        // random subset: `cl_nt_xent_loss` takes no seed parameter, and a
        // per-training-step-shuffled batch (the standard NT-Xent setup)
        // already makes a fixed-order truncation behave like a random
        // subsample across steps, without adding RNG plumbing to a
        // function whose public signature this bucket's fix must keep
        // stable.
        let mut negative_candidates: Vec<usize> =
            (0..total).filter(|&k| k != i && k != i_pos).collect();
        if config.n_negatives > 0 && config.n_negatives < negative_candidates.len() {
            negative_candidates.truncate(config.n_negatives);
        }

        let denom_logits: Vec<f32> = std::iter::once(i_pos)
            .chain(negative_candidates)
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

    // Hoisted out of the per-anchor loop below: previously every negative
    // was re-normalized once per anchor (O(A·N·d) multiply-adds and A·N
    // temporary `Vec<f32>` allocations for A anchors / N negatives), even
    // though normalization does not depend on the anchor at all. Normalize
    // (or clone, per config) each negative exactly once up front instead.
    let negs_n: Vec<Vec<f32>> = negatives
        .iter()
        .map(|nv| {
            if config.normalize_embeddings {
                cl_normalize(nv)
            } else {
                nv.clone()
            }
        })
        .collect();

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

        let mut neg_logits: Vec<f32> = negs_n
            .iter()
            .map(|nv_n| {
                let dot: f32 = a.iter().zip(nv_n.iter()).map(|(x, y)| x * y).sum();
                dot / t
            })
            .collect();

        // `hard_negative_mining`: use only the single hardest negative
        // (highest similarity to this anchor) per the field's own doc
        // ("Use only the hardest negative per anchor"). Previously this
        // flag was declared on the config but read by nothing — every
        // caller got the full negative set regardless of its value.
        if config.hard_negative_mining && neg_logits.len() > 1 {
            let hardest = neg_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            neg_logits = vec![hardest];
        }

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

/// [`cl_triplet_loss`] using `config.margin` instead of an explicit margin
/// argument.
///
/// `ContrastiveLearningConfig::margin` was previously declared and
/// documented ("Margin for triplet loss") but shadowed by
/// [`cl_triplet_loss`]'s own explicit `margin` parameter — a caller driving
/// triplet loss purely from a shared `ContrastiveLearningConfig` had no way
/// to have the margin actually come from it. This wrapper keeps
/// [`cl_triplet_loss`]'s signature (and its explicit-margin use cases)
/// unchanged, while giving config-driven callers the wiring the field
/// promises.
///
/// # Errors
/// Propagates [`cl_triplet_loss`]'s errors.
pub fn cl_triplet_loss_from_config(
    anchors: &[Vec<f32>],
    positives: &[Vec<f32>],
    negatives: &[Vec<f32>],
    config: &ContrastiveLearningConfig,
) -> Result<f32, ContrastiveError> {
    cl_triplet_loss(anchors, positives, negatives, config.margin)
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
    ///
    /// # Errors
    /// Returns [`ContrastiveError::InvalidConfig`] if `capacity == 0` or
    /// `dim == 0`. `capacity == 0` is not merely a degenerate "queue that
    /// never stores anything": the first [`Self::enqueue`] would index
    /// `self.entries[self.write_pos]` on an empty `Vec` (out-of-bounds
    /// panic) and then compute `(write_pos + 1) % capacity` (divide-by-zero
    /// panic) — both reachable from public API, e.g. by constructing the
    /// queue from an unvalidated `ContrastiveLearningConfig::queue_size`.
    pub fn new(capacity: usize, dim: usize) -> Result<Self, ContrastiveError> {
        if capacity == 0 {
            return Err(ContrastiveError::InvalidConfig(
                "EmbeddingQueue capacity must be > 0".into(),
            ));
        }
        if dim == 0 {
            return Err(ContrastiveError::InvalidConfig(
                "EmbeddingQueue dim must be > 0".into(),
            ));
        }
        Ok(Self {
            capacity,
            dim,
            entries: Vec::with_capacity(capacity.min(4096)),
            labels: Vec::with_capacity(capacity.min(4096)),
            write_pos: 0,
            total_enqueued: 0,
        })
    }

    /// Create a queue sized from a [`ContrastiveLearningConfig`]: capacity
    /// from `config.queue_size`, dimension from `config.embedding_dim`.
    ///
    /// This is the wiring the module doc promises ("MoCo-style memory
    /// queue") but that previously did not exist — `queue_size` was
    /// declared on the config and echoed by [`cl_format_config`], but no
    /// function ever read it to actually size a queue.
    ///
    /// # Errors
    /// Returns [`ContrastiveError::InvalidConfig`] under the same
    /// conditions as [`Self::new`] (propagated from `config.queue_size` /
    /// `config.embedding_dim`).
    pub fn from_config(config: &ContrastiveLearningConfig) -> Result<Self, ContrastiveError> {
        Self::new(config.queue_size, config.embedding_dim)
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
        // `self.capacity` is guaranteed > 0 by `new`'s validation, so this
        // modulo can never divide by zero.
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

    // Hoisted out of the per-anchor loop below: previously every negative
    // was re-normalized once per anchor (O(A·N·d) multiply-adds and A·N
    // temporary `Vec<f32>` allocations for A anchors / N negatives), even
    // though normalization does not depend on the anchor. Normalize each
    // negative exactly once up front instead — the identical pattern (and
    // fix) as `cl_info_nce_loss`'s `neg_logits` closure.
    let negatives_norm: Vec<Vec<f32>> = negatives.iter().map(|nv| cl_normalize(nv)).collect();

    for (a, &a_lbl) in anchors.iter().zip(anchor_labels.iter()) {
        let a_norm = cl_normalize(a);
        // Collect (similarity, index) for all negatives with different label
        let mut sims: Vec<(f32, usize)> = negatives_norm
            .iter()
            .zip(negative_labels.iter())
            .enumerate()
            .filter(|(_, (_, &n_lbl))| n_lbl != a_lbl)
            .map(|(idx, (nv_norm, _))| {
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

/// Update `stats.alignment` and `stats.uniformity` from a batch of
/// embeddings, using the same running-mean convention as
/// [`cl_update_stats`] (`stats.n_pairs` must already have been advanced by
/// a paired [`cl_update_stats`] call — this function does not touch
/// `n_pairs` itself, so call it alongside, not instead of,
/// [`cl_update_stats`]).
///
/// `cl_alignment` and `cl_uniformity` existed as free functions but were
/// never connected to [`ContrastiveStats`]: [`cl_update_stats`] — the
/// module's only mutator for it — left `alignment`/`uniformity` at their
/// `Default` value of `0.0` forever, so [`cl_format_stats`] printed
/// `align=0.0000 uniform=0.0000` regardless of the actual training
/// geometry for any caller following the (previously) only available
/// update path.
///
/// # Errors
/// Propagates [`cl_alignment`]'s and [`cl_uniformity`]'s errors (empty
/// input, dimension/length mismatch).
pub fn cl_update_geometry_stats(
    stats: &mut ContrastiveStats,
    positives_a: &[Vec<f32>],
    positives_b: &[Vec<f32>],
    all_embeddings: &[Vec<f32>],
) -> Result<(), ContrastiveError> {
    const DECAY: f32 = 0.99;
    let alignment = cl_alignment(positives_a, positives_b)?;
    let uniformity = cl_uniformity(all_embeddings)?;
    if stats.n_pairs <= 1 {
        // First observation (or called before any `cl_update_stats`):
        // seed directly rather than blending with the `Default` 0.0, to
        // avoid the exact same cold-start bias `cl_update_stats`'s own EMA
        // fields already avoid elsewhere in this struct.
        stats.alignment = alignment;
        stats.uniformity = uniformity;
    } else {
        stats.alignment = DECAY * stats.alignment + (1.0 - DECAY) * alignment;
        stats.uniformity = DECAY * stats.uniformity + (1.0 - DECAY) * uniformity;
    }
    Ok(())
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

/// Uniformity metric (Wang & Isola 2020):
/// -log(mean exp(-2 ‖z_i - z_j‖²)) over all i < j pairs.
///
/// Lower (more negative) = more uniformly distributed on the hypersphere.
/// For a single embedding the result is 0.0 (only one pair: the diagonal).
///
/// This metric is defined for **unit-norm** embeddings — the pairwise
/// squared distance ‖z_i - z_j‖² only has the "spread on the hypersphere"
/// interpretation the doc above (and the Wang & Isola paper) describes when
/// every `z` has norm 1. Every input is therefore L2-normalized internally
/// before the distance computation; there is no well-defined "unnormalized
/// uniformity" variant of this specific metric to opt into (unlike e.g.
/// NT-Xent, where both a normalized and an unnormalized loss are
/// legitimate, config-selectable choices).
///
/// This computes the EXACT metric over all `O(n²)` pairs — for very large
/// `n` (a full multi-thousand-entry [`EmbeddingQueue`], say), prefer
/// [`cl_uniformity_sampled`] instead, which bounds the pair count.
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

    let normalized: Vec<Vec<f32>> = embeddings.iter().map(|v| cl_normalize(v)).collect();

    // Collect all i < j squared distances
    let mut sum_exp = 0.0f64;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let sq: f32 = normalized[i]
                .iter()
                .zip(normalized[j].iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum();
            sum_exp += (-2.0 * sq as f64).exp();
            count += 1;
        }
    }

    let mean_exp = sum_exp / count as f64;
    Ok(-(mean_exp.ln() as f32))
}

/// Sampled variant of [`cl_uniformity`] for large batches.
///
/// The exact metric requires all `O(n²)` pairs — for a full 4096-entry
/// [`EmbeddingQueue`] at `d = 256` that is ~8.4M pairs × 256 dims ≈ 2.1G
/// multiply-adds per call, with no cap. This estimates the same quantity
/// from `max_pairs` uniformly-sampled index pairs (with replacement)
/// instead, seeded by `seed` for reproducibility. When
/// `n·(n-1)/2 <= max_pairs`, all pairs are covered anyway and this returns
/// exactly [`cl_uniformity`]'s result (modulo a possible handful of
/// duplicate draws, since sampling is with replacement even in that case).
///
/// # Errors
/// Same as [`cl_uniformity`].
pub fn cl_uniformity_sampled(
    embeddings: &[Vec<f32>],
    max_pairs: usize,
    seed: u64,
) -> Result<f32, ContrastiveError> {
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

    if n == 1 || max_pairs == 0 {
        return Ok(0.0);
    }

    let normalized: Vec<Vec<f32>> = embeddings.iter().map(|v| cl_normalize(v)).collect();

    let mut state = seed.max(1);
    let mut sum_exp = 0.0f64;
    for _ in 0..max_pairs {
        // Draw two distinct indices in [0, n), uniformly over all n*(n-1)
        // ordered pairs with i != j. Naively drawing `j` uniformly in
        // [0, n) and bumping it by one on collision (`if j == i { j += 1 }`)
        // is BIASED: it makes `j = i + 1` roughly twice as likely as any
        // other index (both "drew (i+1) directly" and "drew i, bumped to
        // i+1" land on it), which measurably skews the distance estimate
        // reported to a caller. Instead draw `i` uniformly, then an offset
        // `d` uniformly in `[1, n)` and set `j = (i + d) % n` — every
        // `j != i` is reachable via exactly one `d`, so this is exactly
        // uniform over `j != i` for any `n >= 2`.
        let i = (xorshift64(&mut state) as usize) % n;
        let d = 1 + (xorshift64(&mut state) as usize) % (n - 1);
        let j = (i + d) % n;
        let sq: f32 = normalized[i]
            .iter()
            .zip(normalized[j].iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum();
        sum_exp += (-2.0 * sq as f64).exp();
    }

    let mean_exp = sum_exp / max_pairs as f64;
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

#[cfg(test)]
mod tests;
