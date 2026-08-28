//! # Few-Shot Adaptation for 3DGS Avatar Creation
//!
//! Implements few-shot and meta-learning adaptation to create new identities
//! from minimal data (5-10 images).
//!
//! ## Key components
//! - **Episodic learning**: N-way K-shot episode construction
//! - **Prototypical Networks** (Snell et al. 2017): nearest-prototype classification
//! - **MAML-style inner loop**: fast gradient adaptation of latent avatar codes
//! - **LoRA-style adapters**: rank-decomposed Gaussian parameter updates
//! - **Statistics**: accuracy aggregation with confidence intervals

use thiserror::Error;

use crate::data_augmentation::{xorshift64, xorshift_f32};

// ── Error type ───────────────────────────────────────────────────────────────

/// Errors produced by the few-shot adaptation subsystem.
#[derive(Debug, Error)]
pub enum FewShotError {
    #[error("Empty support set")]
    EmptySupportSet,
    #[error("Empty query set")]
    EmptyQuerySet,
    #[error("Dimension mismatch: embedding={emb}, expected {expected}")]
    DimensionMismatch { emb: usize, expected: usize },
    #[error("Invalid N-way={n_way}: must be >= 2")]
    InvalidNWay { n_way: usize },
    #[error("Invalid K-shot={k_shot}: must be >= 1")]
    InvalidKShot { k_shot: usize },
    #[error("Not enough samples for {n_way}-way {k_shot}-shot: have {have}")]
    InsufficientSamples {
        n_way: usize,
        k_shot: usize,
        have: usize,
    },
    #[error("Invalid inner learning rate: {lr}")]
    InvalidLR { lr: f32 },
    #[error("No episodes to compute statistics")]
    NoEpisodes,
    #[error("Invalid support label {label}: must be in 0..{n_way}")]
    InvalidLabel { label: usize, n_way: usize },
}

// ── Support / Query / Episode structures ─────────────────────────────────────

/// Support set for an N-way K-shot episode.
#[derive(Debug, Clone)]
pub struct SupportSet {
    /// Stacked support embeddings `[N_way * K_shot * D]`.
    pub embeddings: Vec<f32>,
    /// Class labels in `0..N_way`, length `N_way * K_shot`.
    pub labels: Vec<usize>,
    pub n_way: usize,
    pub k_shot: usize,
    pub d: usize,
}

impl SupportSet {
    /// Create a new support set, validating shapes.
    pub fn new(
        embeddings: Vec<f32>,
        labels: Vec<usize>,
        n_way: usize,
        k_shot: usize,
        d: usize,
    ) -> Result<Self, FewShotError> {
        if n_way < 2 {
            return Err(FewShotError::InvalidNWay { n_way });
        }
        if k_shot < 1 {
            return Err(FewShotError::InvalidKShot { k_shot });
        }
        let n_samples = n_way * k_shot;
        if labels.len() != n_samples {
            return Err(FewShotError::DimensionMismatch {
                emb: labels.len(),
                expected: n_samples,
            });
        }
        if embeddings.len() != n_samples * d {
            return Err(FewShotError::DimensionMismatch {
                emb: embeddings.len(),
                expected: n_samples * d,
            });
        }
        if embeddings.is_empty() {
            return Err(FewShotError::EmptySupportSet);
        }
        if let Some(&bad) = labels.iter().find(|&&l| l >= n_way) {
            return Err(FewShotError::InvalidLabel { label: bad, n_way });
        }
        Ok(Self {
            embeddings,
            labels,
            n_way,
            k_shot,
            d,
        })
    }
}

/// Query set for classification evaluation.
#[derive(Debug, Clone)]
pub struct QuerySet {
    /// Query embeddings `[N_query * D]`.
    pub embeddings: Vec<f32>,
    /// True class labels in `0..N_way`, length `N_query`.
    pub labels: Vec<usize>,
    pub n_query: usize,
    pub d: usize,
}

impl QuerySet {
    /// Create a new query set, validating shapes.
    pub fn new(
        embeddings: Vec<f32>,
        labels: Vec<usize>,
        n_query: usize,
        d: usize,
    ) -> Result<Self, FewShotError> {
        if labels.len() != n_query {
            return Err(FewShotError::DimensionMismatch {
                emb: labels.len(),
                expected: n_query,
            });
        }
        if embeddings.len() != n_query * d {
            return Err(FewShotError::DimensionMismatch {
                emb: embeddings.len(),
                expected: n_query * d,
            });
        }
        if embeddings.is_empty() {
            return Err(FewShotError::EmptyQuerySet);
        }
        Ok(Self {
            embeddings,
            labels,
            n_query,
            d,
        })
    }
}

/// A full N-way K-shot episode with support and query sets.
#[derive(Debug, Clone)]
pub struct Episode {
    pub support: SupportSet,
    pub query: QuerySet,
    pub episode_id: u64,
}

// ── Episode sampling ──────────────────────────────────────────────────────────

/// Group sample indices by their class label.
///
/// Returns a `Vec<Vec<usize>>` indexed by class label.
/// Empty inner vectors mean the class had no samples.
pub fn fsa_class_indices(labels: &[usize], n_total: usize) -> Vec<Vec<usize>> {
    // Find max label to size the output
    let n_classes = labels.iter().copied().fold(0usize, |acc, l| acc.max(l + 1));
    let mut result: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (i, &lbl) in labels.iter().enumerate() {
        if i < n_total {
            result[lbl].push(i);
        }
    }
    result
}

/// Fisher-Yates shuffle of `indices[0..count]` using xorshift64 PRNG.
fn fsa_shuffle(indices: &mut [usize], count: usize, state: &mut u64) {
    let n = count.min(indices.len());
    for i in (1..n).rev() {
        let j = (xorshift64(state) as usize) % (i + 1);
        indices.swap(i, j);
    }
}

/// Sample a random N-way K-shot episode from a flat embedding array.
///
/// # Arguments
/// - `all_embeddings`: flat `[n_total * d]` array
/// - `all_labels`: per-sample class labels
/// - `n_total`: number of samples
/// - `d`: embedding dimension
/// - `n_way`: number of classes per episode
/// - `k_shot`: support samples per class
/// - `n_query`: query samples per class
/// - `seed`: xorshift64 seed (used as `episode_id`)
#[allow(clippy::too_many_arguments)]
pub fn fsa_sample_episode(
    all_embeddings: &[f32],
    all_labels: &[usize],
    n_total: usize,
    d: usize,
    n_way: usize,
    k_shot: usize,
    n_query: usize,
    seed: u64,
) -> Result<Episode, FewShotError> {
    if n_way < 2 {
        return Err(FewShotError::InvalidNWay { n_way });
    }
    if k_shot < 1 {
        return Err(FewShotError::InvalidKShot { k_shot });
    }

    let class_idx = fsa_class_indices(all_labels, n_total);
    let n_classes = class_idx.len();

    // Validate enough classes exist
    if n_classes < n_way {
        return Err(FewShotError::InsufficientSamples {
            n_way,
            k_shot,
            have: n_classes,
        });
    }

    let mut rng = if seed == 0 { 1u64 } else { seed };

    // Sample n_way distinct class indices
    let mut available_classes: Vec<usize> = (0..n_classes)
        .filter(|&c| class_idx[c].len() >= k_shot + n_query)
        .collect();

    if available_classes.len() < n_way {
        return Err(FewShotError::InsufficientSamples {
            n_way,
            k_shot,
            have: available_classes.len(),
        });
    }

    let avail_len = available_classes.len();
    fsa_shuffle(&mut available_classes, avail_len, &mut rng);
    let chosen_classes = &available_classes[..n_way];

    // Build support and query sets
    let mut support_emb: Vec<f32> = Vec::with_capacity(n_way * k_shot * d);
    let mut support_lbl: Vec<usize> = Vec::with_capacity(n_way * k_shot);
    let mut query_emb: Vec<f32> = Vec::with_capacity(n_way * n_query * d);
    let mut query_lbl: Vec<usize> = Vec::with_capacity(n_way * n_query);

    for (local_class, &global_class) in chosen_classes.iter().enumerate() {
        let mut sample_indices = class_idx[global_class].clone();
        let si_len = sample_indices.len();
        fsa_shuffle(&mut sample_indices, si_len, &mut rng);

        // First k_shot → support
        for &idx in &sample_indices[..k_shot] {
            let start = idx * d;
            support_emb.extend_from_slice(&all_embeddings[start..start + d]);
            support_lbl.push(local_class);
        }
        // Next n_query → query
        for &idx in &sample_indices[k_shot..k_shot + n_query] {
            let start = idx * d;
            query_emb.extend_from_slice(&all_embeddings[start..start + d]);
            query_lbl.push(local_class);
        }
    }

    let support = SupportSet::new(support_emb, support_lbl, n_way, k_shot, d)?;
    let query = QuerySet::new(query_emb, query_lbl, n_way * n_query, d)?;

    Ok(Episode {
        support,
        query,
        episode_id: seed,
    })
}

// ── Prototypical Networks ─────────────────────────────────────────────────────

/// Prototypical Network — one prototype (class mean) per class.
#[derive(Debug, Clone)]
pub struct PrototypicalNet {
    /// Class prototypes, shape `[N_way × D]`.
    pub prototypes: Vec<f32>,
    pub n_way: usize,
    pub d: usize,
}

impl PrototypicalNet {
    /// Compute class prototypes as the mean of support embeddings.
    pub fn from_support(support: &SupportSet) -> Self {
        let n_way = support.n_way;
        let d = support.d;
        let mut prototypes = vec![0.0f32; n_way * d];
        let mut counts = vec![0usize; n_way];

        for (i, &lbl) in support.labels.iter().enumerate() {
            let src = &support.embeddings[i * d..(i + 1) * d];
            let dst = &mut prototypes[lbl * d..(lbl + 1) * d];
            for (a, b) in dst.iter_mut().zip(src.iter()) {
                *a += b;
            }
            counts[lbl] += 1;
        }

        for c in 0..n_way {
            let cnt = counts[c].max(1) as f32;
            let proto = &mut prototypes[c * d..(c + 1) * d];
            for v in proto.iter_mut() {
                *v /= cnt;
            }
        }

        Self {
            prototypes,
            n_way,
            d,
        }
    }

    /// Squared Euclidean distance from `query_emb` to each prototype.
    ///
    /// Prototypical Networks (Snell et al. 2017, §3.2) define the class
    /// posterior as `softmax(-||f(x) - c_k||^2)` using the **squared**
    /// Euclidean distance -- this is what makes the classifier equivalent to
    /// a linear model under a Bregman-divergence parameterization. Used by
    /// [`Self::softmax_probs`] (and therefore [`fsa_proto_loss`]); see
    /// [`Self::query_distances`] for a true (non-squared) metric distance.
    pub fn query_sq_distances(&self, query_emb: &[f32]) -> Vec<f32> {
        (0..self.n_way)
            .map(|c| {
                let proto = &self.prototypes[c * self.d..(c + 1) * self.d];
                proto
                    .iter()
                    .zip(query_emb.iter())
                    .map(|(p, q)| (p - q) * (p - q))
                    .sum::<f32>()
            })
            .collect()
    }

    /// True (non-squared) Euclidean distance from `query_emb` to each
    /// prototype. For classification decisions and the prototypical-network
    /// softmax posterior, prefer [`Self::query_sq_distances`] -- the argmin
    /// is the same either way, but the posterior shape is not.
    pub fn query_distances(&self, query_emb: &[f32]) -> Vec<f32> {
        self.query_sq_distances(query_emb)
            .into_iter()
            .map(f32::sqrt)
            .collect()
    }

    /// Classify a single query embedding: returns the index of the nearest prototype.
    pub fn classify(&self, query_emb: &[f32]) -> usize {
        let dists = self.query_sq_distances(query_emb);
        dists
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Classify all query embeddings in batch.
    ///
    /// If `query_embs` is shorter than `n_query * self.d` (a caller-supplied
    /// `n_query` that does not match the buffer), `n_query` is silently
    /// clamped down to the number of complete embeddings actually available
    /// rather than indexing out of bounds.
    pub fn classify_batch(&self, query_embs: &[f32], n_query: usize) -> Vec<usize> {
        let d = self.d.max(1);
        let available = query_embs.len() / d;
        let n_query = n_query.min(available);
        (0..n_query)
            .map(|i| {
                let emb = &query_embs[i * d..(i + 1) * d];
                self.classify(emb)
            })
            .collect()
    }

    /// Softmax probabilities over classes: `exp(-dist²/T) / sum(exp(-dist²/T))`.
    ///
    /// Numerically stable — subtracts max before exponentiation.
    pub fn softmax_probs(&self, query_emb: &[f32], temperature: f32) -> Vec<f32> {
        let dists = self.query_sq_distances(query_emb);
        let t = temperature.max(1e-9);
        let neg_scaled: Vec<f32> = dists.iter().map(|&d| -d / t).collect();
        let max_val = neg_scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = neg_scaled.iter().map(|&v| (v - max_val).exp()).collect();
        let sum = exp_vals.iter().sum::<f32>().max(1e-30);
        exp_vals.into_iter().map(|e| e / sum).collect()
    }
}

/// Fraction of query samples correctly classified.
pub fn fsa_proto_accuracy(net: &PrototypicalNet, query: &QuerySet) -> f32 {
    if query.n_query == 0 {
        return 0.0;
    }
    let preds = net.classify_batch(&query.embeddings, query.n_query);
    let correct = preds
        .iter()
        .zip(query.labels.iter())
        .filter(|(&p, &l)| p == l)
        .count();
    correct as f32 / query.n_query as f32
}

/// Cross-entropy loss for prototypical classification.
///
/// `loss = mean over queries of -log(p_correct_class)`
pub fn fsa_proto_loss(net: &PrototypicalNet, query: &QuerySet, temperature: f32) -> f32 {
    if query.n_query == 0 {
        return 0.0;
    }
    let mut total = 0.0f32;
    for i in 0..query.n_query {
        let emb = &query.embeddings[i * net.d..(i + 1) * net.d];
        let probs = net.softmax_probs(emb, temperature);
        let true_lbl = query.labels[i];
        let p = probs.get(true_lbl).copied().unwrap_or(1e-30).max(1e-30);
        total -= p.ln();
    }
    total / query.n_query as f32
}

/// Build a `PrototypicalNet` from the episode's support set and evaluate on its query set.
pub fn fsa_episode_accuracy(episode: &Episode) -> f32 {
    let net = PrototypicalNet::from_support(&episode.support);
    fsa_proto_accuracy(&net, &episode.query)
}

// ── MAML-style gradient computation ──────────────────────────────────────────

/// MAML adaptation state: initial and adapted parameters.
#[derive(Debug, Clone)]
pub struct MamlState {
    /// Meta-initialization θ₀.
    pub init_params: Vec<f32>,
    /// After inner loop θᵢ.
    pub adapted_params: Vec<f32>,
    pub inner_lr: f32,
    pub n_inner_steps: usize,
    pub d: usize,
}

impl MamlState {
    /// Create a new MAML state.
    pub fn new(
        init_params: Vec<f32>,
        inner_lr: f32,
        n_inner_steps: usize,
    ) -> Result<Self, FewShotError> {
        if inner_lr <= 0.0 || !inner_lr.is_finite() {
            return Err(FewShotError::InvalidLR { lr: inner_lr });
        }
        let d = init_params.len();
        let adapted_params = init_params.clone();
        Ok(Self {
            init_params,
            adapted_params,
            inner_lr,
            n_inner_steps,
            d,
        })
    }

    /// Single gradient descent step: θ ← θ - lr · g.
    pub fn inner_update(&mut self, gradient: &[f32]) {
        for (p, &g) in self.adapted_params.iter_mut().zip(gradient.iter()) {
            *p -= self.inner_lr * g;
        }
    }

    /// Reset adapted params back to init.
    pub fn reset(&mut self) {
        self.adapted_params.clone_from(&self.init_params);
    }
}

/// MSE between `params` and `target`: `sum((pᵢ - tᵢ)²) / D`.
pub fn fsa_inner_loss(params: &[f32], target: &[f32]) -> f32 {
    let d = params.len().max(1);
    params
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| (p - t) * (p - t))
        .sum::<f32>()
        / d as f32
}

/// Gradient of MSE w.r.t. `params`: `2 * (params - target) / D`.
pub fn fsa_inner_gradient(params: &[f32], target: &[f32]) -> Vec<f32> {
    let d = params.len().max(1) as f32;
    params
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| 2.0 * (p - t) / d)
        .collect()
}

/// Run MAML inner-loop adaptation on the mean of support embeddings.
///
/// Performs `n_inner_steps` of gradient descent against the support mean.
/// Returns the final inner loss.
pub fn fsa_maml_adapt(state: &mut MamlState, support_embeddings: &[f32], n_support: usize) -> f32 {
    let d = state.d;
    if d == 0 {
        return 0.0;
    }
    // Defensively clamp down to the number of complete embeddings actually
    // present, so a caller-supplied `n_support` that overstates the buffer
    // cannot index out of bounds below.
    let n_support = n_support.min(support_embeddings.len() / d);
    if n_support == 0 {
        return 0.0;
    }

    // Compute mean of support embeddings
    let mut mean = vec![0.0f32; d];
    for i in 0..n_support {
        let emb = &support_embeddings[i * d..(i + 1) * d];
        for (m, &v) in mean.iter_mut().zip(emb.iter()) {
            *m += v;
        }
    }
    let n = n_support as f32;
    for m in mean.iter_mut() {
        *m /= n;
    }

    // Inner loop gradient descent
    let mut final_loss = 0.0f32;
    for _ in 0..state.n_inner_steps {
        let grad = fsa_inner_gradient(&state.adapted_params, &mean);
        state.inner_update(&grad);
        final_loss = fsa_inner_loss(&state.adapted_params, &mean);
    }
    final_loss
}

/// Average MSE between adapted params and each query embedding.
pub fn fsa_maml_query_loss(state: &MamlState, query_embeddings: &[f32], n_query: usize) -> f32 {
    if n_query == 0 || state.d == 0 {
        return 0.0;
    }
    let d = state.d;
    let total: f32 = (0..n_query)
        .map(|i| {
            let emb = &query_embeddings[i * d..(i + 1) * d];
            fsa_inner_loss(&state.adapted_params, emb)
        })
        .sum();
    total / n_query as f32
}

// ── LoRA-style adapter ────────────────────────────────────────────────────────

/// LoRA-inspired rank-r decomposition adapter for Gaussian parameter updates.
///
/// ΔW = B · A, where A is `[r × D_in]` and B is `[D_out × r]`.
/// Initialized so that ΔW = 0 (B = 0, A = random small noise).
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Down-projection matrix `[r × D_in]`, row-major.
    pub a_matrix: Vec<f32>,
    /// Up-projection matrix `[D_out × r]`, row-major.
    pub b_matrix: Vec<f32>,
    pub d_in: usize,
    pub d_out: usize,
    pub rank: usize,
    /// Scaling factor `alpha / rank`.
    pub scale: f32,
}

impl LoraAdapter {
    /// Initialize with A ~ N(0, 0.01), B = 0.
    ///
    /// Uses Box-Muller transform + xorshift64. No C/Fortran dependencies.
    pub fn new(d_in: usize, d_out: usize, rank: usize, alpha: f32, seed: u64) -> Self {
        let mut rng = if seed == 0 { 1u64 } else { seed };
        let scale = alpha / rank.max(1) as f32;

        // A matrix [rank × d_in]: small random N(0, 0.01)
        let a_size = rank * d_in;
        let mut a_matrix = Vec::with_capacity(a_size);

        let std_dev = 0.01f32;
        let mut i = 0usize;
        while i < a_size {
            // Box-Muller transform
            let u1 = (xorshift_f32(&mut rng) as f64).max(1e-12);
            let u2 = xorshift_f32(&mut rng) as f64;
            let mag = (-2.0 * u1.ln()).sqrt();
            let z0 = mag * (2.0 * std::f64::consts::PI * u2).cos();
            let z1 = mag * (2.0 * std::f64::consts::PI * u2).sin();
            a_matrix.push((z0 as f32) * std_dev);
            i += 1;
            if i < a_size {
                a_matrix.push((z1 as f32) * std_dev);
                i += 1;
            }
        }

        // B matrix [d_out × rank]: zeros so ΔW = 0 initially
        let b_matrix = vec![0.0f32; d_out * rank];

        Self {
            a_matrix,
            b_matrix,
            d_in,
            d_out,
            rank,
            scale,
        }
    }

    /// Forward pass: y = (B · (A · x)) * scale.
    ///
    /// - x: `[D_in]`
    /// - A: `[rank × D_in]` → intermediate z: `[rank]`
    /// - B: `[D_out × rank]` → output y: `[D_out]`
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let r = self.rank;
        let d_in = self.d_in;
        let d_out = self.d_out;

        // z = A · x  [rank]
        let mut z = vec![0.0f32; r];
        for (i, z_elem) in z.iter_mut().enumerate() {
            let a_row = &self.a_matrix[i * d_in..(i + 1) * d_in];
            *z_elem = a_row.iter().zip(x.iter()).map(|(&a, &xi)| a * xi).sum();
        }

        // y = B · z  [d_out]
        let mut y = vec![0.0f32; d_out];
        for (i, y_elem) in y.iter_mut().enumerate() {
            let b_row = &self.b_matrix[i * r..(i + 1) * r];
            *y_elem = b_row.iter().zip(z.iter()).map(|(&b, &zi)| b * zi).sum();
        }

        // Scale
        for yi in y.iter_mut() {
            *yi *= self.scale;
        }
        y
    }

    /// Total number of trainable parameters (A + B matrices).
    pub fn n_params(&self) -> usize {
        self.rank * self.d_in + self.d_out * self.rank
    }
}

/// Apply LoRA adapter on top of a base output: result = base + adapter.forward(input).
pub fn fsa_lora_apply(base_output: &[f32], adapter: &LoraAdapter, input: &[f32]) -> Vec<f32> {
    let delta = adapter.forward(input);
    base_output
        .iter()
        .zip(delta.iter())
        .map(|(&b, &d)| b + d)
        .collect()
}

// ── Adaptation tracking ───────────────────────────────────────────────────────

/// Tracks the accuracy improvement from a few-shot adaptation step.
#[derive(Debug, Clone)]
pub struct AdaptationResult {
    pub n_support: usize,
    pub n_query: usize,
    /// Accuracy before adaptation.
    pub pre_accuracy: f32,
    /// Accuracy after adaptation.
    pub post_accuracy: f32,
    pub n_steps: usize,
    pub final_loss: f32,
    /// `post_accuracy - pre_accuracy`.
    pub improvement: f32,
}

// ── Statistics ────────────────────────────────────────────────────────────────

/// Aggregate statistics across multiple few-shot episodes.
#[derive(Debug, Clone)]
pub struct FewShotStats {
    pub mean_accuracy: f32,
    pub std_accuracy: f32,
    pub min_accuracy: f32,
    pub max_accuracy: f32,
    pub n_episodes: usize,
    pub mean_n_way: f32,
    pub mean_k_shot: f32,
    /// 95% confidence interval: `1.96 * std / sqrt(n)`.
    pub ci_95: f32,
}

/// Compute aggregate statistics over a collection of episode accuracies.
pub fn fsa_compute_stats(
    accuracies: &[f32],
    n_ways: &[usize],
    k_shots: &[usize],
) -> Result<FewShotStats, FewShotError> {
    if accuracies.is_empty() {
        return Err(FewShotError::NoEpisodes);
    }
    let n = accuracies.len();
    let mean_accuracy = accuracies.iter().sum::<f32>() / n as f32;
    let variance = accuracies
        .iter()
        .map(|&a| (a - mean_accuracy) * (a - mean_accuracy))
        .sum::<f32>()
        / n as f32;
    let std_accuracy = variance.sqrt();
    let min_accuracy = accuracies.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_accuracy = accuracies.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_n_way = if n_ways.is_empty() {
        0.0
    } else {
        n_ways.iter().sum::<usize>() as f32 / n_ways.len() as f32
    };
    let mean_k_shot = if k_shots.is_empty() {
        0.0
    } else {
        k_shots.iter().sum::<usize>() as f32 / k_shots.len() as f32
    };
    let ci_95 = 1.96 * std_accuracy / (n as f32).sqrt();
    Ok(FewShotStats {
        mean_accuracy,
        std_accuracy,
        min_accuracy,
        max_accuracy,
        n_episodes: n,
        mean_n_way,
        mean_k_shot,
        ci_95,
    })
}

/// Format `FewShotStats` for display.
pub fn fsa_format_stats(stats: &FewShotStats) -> String {
    format!(
        "FewShotStats {{ n_episodes={}, mean_acc={:.4}, std={:.4}, \
         min={:.4}, max={:.4}, ci_95={:.4}, mean_n_way={:.1}, mean_k_shot={:.1} }}",
        stats.n_episodes,
        stats.mean_accuracy,
        stats.std_accuracy,
        stats.min_accuracy,
        stats.max_accuracy,
        stats.ci_95,
        stats.mean_n_way,
        stats.mean_k_shot,
    )
}

/// Format `FewShotConfig` for display.
pub fn fsa_format_config(config: &FewShotConfig) -> String {
    format!(
        "FewShotConfig {{ {}-way {}-shot, n_query={}, n_episodes={}, \
         inner_lr={}, n_inner_steps={}, proto_temperature={}, \
         lora_rank={}, lora_alpha={}, seed={} }}",
        config.n_way,
        config.k_shot,
        config.n_query,
        config.n_episodes,
        config.inner_lr,
        config.n_inner_steps,
        config.proto_temperature,
        config.lora_rank,
        config.lora_alpha,
        config.seed,
    )
}

/// Format `AdaptationResult` for display.
pub fn fsa_format_result(result: &AdaptationResult) -> String {
    format!(
        "AdaptationResult {{ n_support={}, n_query={}, pre_acc={:.4}, \
         post_acc={:.4}, improvement={:+.4}, steps={}, final_loss={:.6} }}",
        result.n_support,
        result.n_query,
        result.pre_accuracy,
        result.post_accuracy,
        result.improvement,
        result.n_steps,
        result.final_loss,
    )
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for few-shot adaptation training.
#[derive(Debug, Clone)]
pub struct FewShotConfig {
    pub n_way: usize,
    pub k_shot: usize,
    pub n_query: usize,
    pub n_episodes: usize,
    pub inner_lr: f32,
    pub n_inner_steps: usize,
    /// Softmax temperature for prototypical loss.
    pub proto_temperature: f32,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub seed: u64,
}

impl FewShotConfig {
    /// Standard 5-way 5-shot benchmark configuration.
    pub fn default_5way_5shot() -> Self {
        Self {
            n_way: 5,
            k_shot: 5,
            n_query: 15,
            n_episodes: 600,
            inner_lr: 0.01,
            n_inner_steps: 5,
            proto_temperature: 1.0,
            lora_rank: 4,
            lora_alpha: 4.0,
            seed: 42,
        }
    }

    /// Standard 5-way 1-shot benchmark configuration.
    pub fn default_5way_1shot() -> Self {
        Self {
            n_way: 5,
            k_shot: 1,
            n_query: 15,
            n_episodes: 600,
            inner_lr: 0.01,
            n_inner_steps: 5,
            proto_temperature: 1.0,
            lora_rank: 4,
            lora_alpha: 4.0,
            seed: 42,
        }
    }
}

// ── Episode runner ───────────────────────────────────────────────────────────

/// Run `config.n_episodes` full few-shot adaptation episodes end-to-end.
///
/// For each episode this samples an N-way K-shot task via
/// [`fsa_sample_episode`] (seeded by `config.seed + episode_index`), then
/// runs a genuine MAML pre/post adaptation comparison:
///
/// - **pre_accuracy**: query-set accuracy of a [`PrototypicalNet`] built from
///   the *un-adapted* (zero meta-init) prototypes -- the model has not yet
///   seen the episode's support set.
/// - **post_accuracy**: query-set accuracy of the *same* prototypes after
///   independently adapting each class's prototype via [`MamlState`] /
///   [`fsa_maml_adapt`] for `config.n_inner_steps` steps at `config.inner_lr`,
///   targeting that class's own support-set mean.
///
/// `config.proto_temperature` weights the post-adaptation query
/// cross-entropy ([`fsa_proto_loss`]) reported as each episode's
/// `final_loss`. Per-episode [`AdaptationResult`]s are logged via
/// [`fsa_format_result`] at `debug` level, and the post-adaptation
/// accuracies are aggregated via [`fsa_compute_stats`].
///
/// # Errors
///
/// Propagates any [`FewShotError`] from episode sampling or `MamlState`
/// construction (e.g. `config.inner_lr <= 0.0`), and
/// [`FewShotError::NoEpisodes`] if `config.n_episodes == 0`.
pub fn fsa_run_episodes(
    all_embeddings: &[f32],
    all_labels: &[usize],
    n_total: usize,
    d: usize,
    config: &FewShotConfig,
) -> Result<FewShotStats, FewShotError> {
    let mut accuracies: Vec<f32> = Vec::with_capacity(config.n_episodes);
    let mut n_ways: Vec<usize> = Vec::with_capacity(config.n_episodes);
    let mut k_shots: Vec<usize> = Vec::with_capacity(config.n_episodes);

    for i in 0..config.n_episodes {
        let seed_i = config.seed.wrapping_add(i as u64);
        let episode = fsa_sample_episode(
            all_embeddings,
            all_labels,
            n_total,
            d,
            config.n_way,
            config.k_shot,
            config.n_query,
            seed_i,
        )?;
        let sup = &episode.support;

        // "Meta-initial" prototypes: zero for every class, representing a
        // model that has not yet adapted to this episode's support set.
        let pre_net = PrototypicalNet {
            prototypes: vec![0.0f32; sup.n_way * sup.d],
            n_way: sup.n_way,
            d: sup.d,
        };
        let pre_accuracy = fsa_proto_accuracy(&pre_net, &episode.query);

        // MAML inner loop: independently adapt each class's prototype from
        // the same zero meta-init, via `n_inner_steps` of gradient descent
        // (at `inner_lr`) toward that class's own support mean.
        let mut adapted_prototypes = vec![0.0f32; sup.n_way * sup.d];
        for c in 0..sup.n_way {
            let start = c * sup.k_shot * sup.d;
            let end = start + sup.k_shot * sup.d;
            let class_support = &sup.embeddings[start..end];

            let mut maml_state =
                MamlState::new(vec![0.0f32; sup.d], config.inner_lr, config.n_inner_steps)?;
            fsa_maml_adapt(&mut maml_state, class_support, sup.k_shot);
            adapted_prototypes[c * sup.d..(c + 1) * sup.d]
                .copy_from_slice(&maml_state.adapted_params);
        }

        let post_net = PrototypicalNet {
            prototypes: adapted_prototypes,
            n_way: sup.n_way,
            d: sup.d,
        };
        let post_accuracy = fsa_proto_accuracy(&post_net, &episode.query);
        let final_loss = fsa_proto_loss(&post_net, &episode.query, config.proto_temperature);

        let result = AdaptationResult {
            n_support: sup.labels.len(),
            n_query: episode.query.labels.len(),
            pre_accuracy,
            post_accuracy,
            n_steps: config.n_inner_steps,
            final_loss,
            improvement: post_accuracy - pre_accuracy,
        };
        tracing::debug!("few-shot episode {i}: {}", fsa_format_result(&result));

        accuracies.push(post_accuracy);
        n_ways.push(sup.n_way);
        k_shots.push(sup.k_shot);
    }

    fsa_compute_stats(&accuracies, &n_ways, &k_shots)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper builders ──────────────────────────────────────────────────────

    fn make_support(n_way: usize, k_shot: usize, d: usize) -> SupportSet {
        let n = n_way * k_shot;
        let embeddings: Vec<f32> = (0..n * d)
            .map(|i| {
                let class = (i / d) / k_shot;
                class as f32 + (i % d) as f32 * 0.001
            })
            .collect();
        let labels: Vec<usize> = (0..n).map(|i| i / k_shot).collect();
        SupportSet::new(embeddings, labels, n_way, k_shot, d).expect("valid support")
    }

    fn make_query_for(net: &PrototypicalNet, n_query: usize) -> QuerySet {
        // Place each query exactly at the prototype (perfect classification)
        let n_way = net.n_way;
        let d = net.d;
        let mut embs = Vec::with_capacity(n_query * d);
        let mut lbls = Vec::with_capacity(n_query);
        for i in 0..n_query {
            let c = i % n_way;
            embs.extend_from_slice(&net.prototypes[c * d..(c + 1) * d]);
            lbls.push(c);
        }
        QuerySet::new(embs, lbls, n_query, d).expect("valid query")
    }

    // ── SupportSet tests ─────────────────────────────────────────────────────

    #[test]
    fn test_support_set_valid() {
        let ss = make_support(3, 2, 8);
        assert_eq!(ss.n_way, 3);
        assert_eq!(ss.k_shot, 2);
        assert_eq!(ss.d, 8);
        assert_eq!(ss.embeddings.len(), 3 * 2 * 8);
        assert_eq!(ss.labels.len(), 3 * 2);
    }

    #[test]
    fn test_support_set_wrong_embedding_length() {
        let emb = vec![0.0f32; 5]; // wrong
        let lbl = vec![0usize, 0, 1, 1];
        let result = SupportSet::new(emb, lbl, 2, 2, 4);
        assert!(matches!(
            result,
            Err(FewShotError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_support_set_wrong_label_length() {
        let emb = vec![0.0f32; 2 * 2 * 4]; // 2-way 2-shot d=4
        let lbl = vec![0usize, 0, 1]; // wrong — should be 4
        let result = SupportSet::new(emb, lbl, 2, 2, 4);
        assert!(matches!(
            result,
            Err(FewShotError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_support_set_label_out_of_range_errors() {
        let emb = vec![0.0f32; 2 * 2 * 4]; // n_way=2, k_shot=2, d=4
        let lbl = vec![0usize, 0, 1, 5]; // label 5 is out of range for n_way=2
        let result = SupportSet::new(emb, lbl, 2, 2, 4);
        assert!(matches!(
            result,
            Err(FewShotError::InvalidLabel { label: 5, n_way: 2 })
        ));
    }

    #[test]
    fn test_support_set_invalid_n_way() {
        let emb = vec![0.0f32; 2 * 4];
        let lbl = vec![0usize, 0];
        let result = SupportSet::new(emb, lbl, 1, 2, 4);
        assert!(matches!(result, Err(FewShotError::InvalidNWay { .. })));
    }

    #[test]
    fn test_support_set_invalid_k_shot() {
        let emb = vec![0.0f32; 0];
        let lbl: Vec<usize> = vec![];
        let result = SupportSet::new(emb, lbl, 2, 0, 4);
        assert!(matches!(result, Err(FewShotError::InvalidKShot { .. })));
    }

    // ── QuerySet tests ───────────────────────────────────────────────────────

    #[test]
    fn test_query_set_basic_creation() {
        let emb = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 × d=3
        let lbl = vec![0usize, 1];
        let qs = QuerySet::new(emb, lbl, 2, 3).expect("valid query");
        assert_eq!(qs.n_query, 2);
        assert_eq!(qs.d, 3);
    }

    #[test]
    fn test_query_set_wrong_embedding_length() {
        let emb = vec![1.0f32, 2.0]; // wrong
        let lbl = vec![0usize, 1];
        let result = QuerySet::new(emb, lbl, 2, 3);
        assert!(matches!(
            result,
            Err(FewShotError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_query_set_wrong_label_count() {
        let emb = vec![0.0f32; 2 * 4];
        let lbl = vec![0usize]; // should be 2
        let result = QuerySet::new(emb, lbl, 2, 4);
        assert!(matches!(
            result,
            Err(FewShotError::DimensionMismatch { .. })
        ));
    }

    // ── fsa_class_indices tests ──────────────────────────────────────────────

    #[test]
    fn test_class_indices_groups_correctly() {
        let labels = [0usize, 1, 0, 2, 1, 0];
        let idx = fsa_class_indices(&labels, 6);
        assert_eq!(idx[0], vec![0, 2, 5]);
        assert_eq!(idx[1], vec![1, 4]);
        assert_eq!(idx[2], vec![3]);
    }

    #[test]
    fn test_class_indices_all_classes_represented() {
        let labels = [0usize, 1, 2, 3, 0, 1, 2, 3];
        let idx = fsa_class_indices(&labels, 8);
        assert_eq!(idx.len(), 4);
        for (c, class_idx) in idx.iter().enumerate() {
            assert!(!class_idx.is_empty(), "class {c} should have samples");
        }
    }

    #[test]
    fn test_class_indices_single_class() {
        let labels = [0usize, 0, 0];
        let idx = fsa_class_indices(&labels, 3);
        assert_eq!(idx.len(), 1);
        assert_eq!(idx[0].len(), 3);
    }

    // ── fsa_sample_episode tests ─────────────────────────────────────────────

    fn build_synthetic_dataset(
        n_classes: usize,
        n_per_class: usize,
        d: usize,
    ) -> (Vec<f32>, Vec<usize>) {
        let n_total = n_classes * n_per_class;
        let embeddings: Vec<f32> = (0..n_total * d)
            .map(|i| {
                let sample = i / d;
                let class = sample / n_per_class;
                class as f32 * 10.0 + (i % d) as f32 * 0.01
            })
            .collect();
        let labels: Vec<usize> = (0..n_total).map(|i| i / n_per_class).collect();
        (embeddings, labels)
    }

    #[test]
    fn test_sample_episode_correct_sizes() {
        let (embs, lbls) = build_synthetic_dataset(5, 20, 16);
        let ep =
            fsa_sample_episode(&embs, &lbls, 100, 16, 3, 5, 10, 12345).expect("should succeed");
        assert_eq!(ep.support.n_way, 3);
        assert_eq!(ep.support.k_shot, 5);
        assert_eq!(ep.support.embeddings.len(), 3 * 5 * 16);
        assert_eq!(ep.query.n_query, 3 * 10);
    }

    #[test]
    fn test_sample_episode_support_labels_in_range() {
        let (embs, lbls) = build_synthetic_dataset(5, 20, 8);
        let ep = fsa_sample_episode(&embs, &lbls, 100, 8, 3, 4, 5, 99).expect("should succeed");
        for &l in &ep.support.labels {
            assert!(l < ep.support.n_way, "support label {l} out of range");
        }
    }

    #[test]
    fn test_sample_episode_query_labels_in_range() {
        let (embs, lbls) = build_synthetic_dataset(5, 20, 8);
        let ep = fsa_sample_episode(&embs, &lbls, 100, 8, 3, 4, 5, 77).expect("should succeed");
        for &l in &ep.query.labels {
            assert!(l < ep.support.n_way, "query label {l} out of range");
        }
    }

    #[test]
    fn test_sample_episode_insufficient_samples() {
        // Only 2 samples per class, but we need k_shot=3+n_query=2=5
        let (embs, lbls) = build_synthetic_dataset(5, 2, 4);
        let result = fsa_sample_episode(&embs, &lbls, 10, 4, 3, 3, 2, 1);
        assert!(matches!(
            result,
            Err(FewShotError::InsufficientSamples { .. })
        ));
    }

    #[test]
    fn test_sample_episode_insufficient_classes() {
        let (embs, lbls) = build_synthetic_dataset(2, 10, 4); // only 2 classes
        let result = fsa_sample_episode(&embs, &lbls, 20, 4, 5, 2, 3, 1); // needs 5-way
        assert!(matches!(
            result,
            Err(FewShotError::InsufficientSamples { .. })
        ));
    }

    #[test]
    fn test_sample_episode_invalid_n_way() {
        let (embs, lbls) = build_synthetic_dataset(5, 10, 4);
        let result = fsa_sample_episode(&embs, &lbls, 50, 4, 1, 2, 3, 1);
        assert!(matches!(result, Err(FewShotError::InvalidNWay { .. })));
    }

    #[test]
    fn test_sample_episode_invalid_k_shot() {
        let (embs, lbls) = build_synthetic_dataset(5, 10, 4);
        let result = fsa_sample_episode(&embs, &lbls, 50, 4, 3, 0, 3, 1);
        assert!(matches!(result, Err(FewShotError::InvalidKShot { .. })));
    }

    // ── PrototypicalNet tests ────────────────────────────────────────────────

    #[test]
    fn test_proto_from_support_shape() {
        let ss = make_support(4, 3, 16);
        let net = PrototypicalNet::from_support(&ss);
        assert_eq!(net.prototypes.len(), 4 * 16);
        assert_eq!(net.n_way, 4);
        assert_eq!(net.d, 16);
    }

    #[test]
    fn test_proto_from_support_single_class_equals_embedding() {
        // 2-way 1-shot with a known embedding for class 0
        let d = 4;
        let emb0 = vec![1.0f32, 2.0, 3.0, 4.0];
        let emb1 = vec![9.0f32, 9.0, 9.0, 9.0];
        let mut embs = emb0.clone();
        embs.extend_from_slice(&emb1);
        let labels = vec![0usize, 1];
        let ss = SupportSet::new(embs, labels, 2, 1, d).expect("valid");
        let net = PrototypicalNet::from_support(&ss);
        let proto0 = &net.prototypes[..d];
        assert!((proto0[0] - 1.0).abs() < 1e-6);
        assert!((proto0[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_proto_classify_closest_class() {
        // 2-way, place class 0 at [0,0] and class 1 at [10,10]
        let d = 2;
        let embs = vec![0.0f32, 0.0, 10.0, 10.0];
        let labels = vec![0usize, 1];
        let ss = SupportSet::new(embs, labels, 2, 1, d).expect("valid");
        let net = PrototypicalNet::from_support(&ss);
        assert_eq!(net.classify(&[0.1, 0.1]), 0);
        assert_eq!(net.classify(&[9.9, 9.9]), 1);
    }

    #[test]
    fn test_proto_classify_identical_to_prototype() {
        let ss = make_support(3, 2, 8);
        let net = PrototypicalNet::from_support(&ss);
        let proto1 = net.prototypes[8..2 * 8].to_vec();
        assert_eq!(net.classify(&proto1), 1);
    }

    #[test]
    fn test_proto_classify_batch_length() {
        let ss = make_support(3, 2, 8);
        let net = PrototypicalNet::from_support(&ss);
        let qembs: Vec<f32> = vec![0.0f32; 7 * 8];
        let preds = net.classify_batch(&qembs, 7);
        assert_eq!(preds.len(), 7);
    }

    #[test]
    fn test_proto_classify_batch_short_buffer_does_not_panic() {
        let ss = make_support(2, 2, 4);
        let net = PrototypicalNet::from_support(&ss);
        // Claim n_query=10 but only provide embeddings for 1 query.
        let short_query = vec![0.0f32; 4];
        let preds = net.classify_batch(&short_query, 10);
        assert_eq!(
            preds.len(),
            1,
            "should clamp to the embeddings actually available"
        );
    }

    #[test]
    fn test_proto_query_distances_non_negative() {
        let ss = make_support(3, 2, 8);
        let net = PrototypicalNet::from_support(&ss);
        let emb = vec![1.0f32; 8];
        for d in net.query_distances(&emb) {
            assert!(d >= 0.0, "distance must be non-negative");
        }
    }

    #[test]
    fn test_proto_softmax_uses_squared_distance() {
        // Two classes: prototype 0 at [0,0], prototype 1 at [3,4] (distance
        // 5 from the origin). Query at the origin: distance to class 0 = 0,
        // distance to class 1 = 5.
        //
        // With SQUARED distance (Snell et al., the correct formula): logits
        // are [0, -25/T]. With unsquared distance (the bug): logits would be
        // [0, -5/T]. At T=1 these produce very different probability ratios.
        let d = 2;
        let embs = vec![0.0f32, 0.0, 3.0, 4.0];
        let labels = vec![0usize, 1];
        let ss = SupportSet::new(embs, labels, 2, 1, d).expect("valid");
        let net = PrototypicalNet::from_support(&ss);
        let query = vec![0.0f32, 0.0];
        let probs = net.softmax_probs(&query, 1.0);

        // Expected under the squared-distance (correct) formula:
        // logit0 = -0^2 = 0, logit1 = -5^2 = -25.
        let expected_p1 = (-25.0f32).exp() / (1.0 + (-25.0f32).exp());
        assert!(
            (probs[1] - expected_p1).abs() < 1e-6,
            "expected squared-distance softmax p1={expected_p1}, got {}",
            probs[1]
        );
        // Sanity: under the old (buggy) unsquared-distance formula, p1 would
        // be exp(-5)/(1+exp(-5)) ≈ 0.0067 -- orders of magnitude larger than
        // the correct ~1.4e-11. Guard against regressing to that behaviour.
        let buggy_unsquared_p1 = (-5.0f32).exp() / (1.0 + (-5.0f32).exp());
        assert!(
            probs[1] < buggy_unsquared_p1 / 100.0,
            "softmax should use squared distance, not unsquared"
        );
    }

    #[test]
    fn test_proto_softmax_probs_sum_to_one() {
        let ss = make_support(3, 2, 8);
        let net = PrototypicalNet::from_support(&ss);
        let emb = vec![1.0f32; 8];
        let probs = net.softmax_probs(&emb, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probs should sum to 1, got {sum}");
    }

    #[test]
    fn test_proto_softmax_probs_nearest_highest() {
        let d = 2;
        let embs = vec![0.0f32, 0.0, 10.0, 10.0, 20.0, 20.0];
        let labels = vec![0usize, 1, 2];
        let ss = SupportSet::new(embs, labels, 3, 1, d).expect("valid");
        let net = PrototypicalNet::from_support(&ss);
        let query = vec![0.1f32, 0.1]; // closest to class 0
        let probs = net.softmax_probs(&query, 1.0);
        assert!(
            probs[0] > probs[1] && probs[0] > probs[2],
            "class 0 should have highest prob"
        );
    }

    // ── fsa_proto_accuracy tests ─────────────────────────────────────────────

    #[test]
    fn test_proto_accuracy_perfect() {
        let ss = make_support(3, 2, 8);
        let net = PrototypicalNet::from_support(&ss);
        let query = make_query_for(&net, 9);
        let acc = fsa_proto_accuracy(&net, &query);
        assert!(
            (acc - 1.0).abs() < 1e-6,
            "perfect accuracy expected, got {acc}"
        );
    }

    #[test]
    fn test_proto_accuracy_all_wrong() {
        // 2-way: class 0 at [0,0], class 1 at [100,100]
        // Query: label=0 but embedding near [100,100] → classified as 1
        let d = 2;
        let embs = vec![0.0f32, 0.0, 100.0, 100.0];
        let labels_s = vec![0usize, 1];
        let ss = SupportSet::new(embs, labels_s, 2, 1, d).expect("valid");
        let net = PrototypicalNet::from_support(&ss);
        // Query near class 1's prototype but labeled 0
        let q_embs = vec![99.0f32, 99.0, 99.0, 99.0];
        let q_labels = vec![0usize, 0]; // wrong label
        let qs = QuerySet::new(q_embs, q_labels, 2, d).expect("valid");
        let acc = fsa_proto_accuracy(&net, &qs);
        assert!(acc < 0.5, "accuracy should be 0, got {acc}");
    }

    #[test]
    fn test_proto_loss_perfect_prediction_low() {
        let ss = make_support(3, 2, 8);
        let net = PrototypicalNet::from_support(&ss);
        let query = make_query_for(&net, 9);
        let loss = fsa_proto_loss(&net, &query, 0.01); // low temperature → confident
        assert!(
            loss < 0.1,
            "loss should be near 0 for perfect prediction, got {loss}"
        );
    }

    #[test]
    fn test_episode_accuracy_perfect() {
        let (embs, lbls) = build_synthetic_dataset(5, 20, 16);
        let ep = fsa_sample_episode(&embs, &lbls, 100, 16, 3, 5, 5, 42).expect("valid episode");
        // The embeddings are class-offset by 10.0 so prototypes are well-separated
        let acc = fsa_episode_accuracy(&ep);
        assert!(
            acc > 0.9,
            "well-separated classes should yield high accuracy, got {acc}"
        );
    }

    // ── MamlState tests ──────────────────────────────────────────────────────

    #[test]
    fn test_maml_new_invalid_lr_zero() {
        let result = MamlState::new(vec![0.0f32; 4], 0.0, 5);
        assert!(matches!(result, Err(FewShotError::InvalidLR { .. })));
    }

    #[test]
    fn test_maml_new_invalid_lr_negative() {
        let result = MamlState::new(vec![0.0f32; 4], -0.01, 5);
        assert!(matches!(result, Err(FewShotError::InvalidLR { .. })));
    }

    #[test]
    fn test_maml_new_invalid_lr_nan() {
        let result = MamlState::new(vec![0.0f32; 4], f32::NAN, 5);
        assert!(matches!(result, Err(FewShotError::InvalidLR { .. })));
    }

    #[test]
    fn test_maml_inner_update_direction() {
        let mut state = MamlState::new(vec![2.0f32; 4], 0.1, 1).expect("valid");
        let grad = vec![1.0f32; 4]; // positive gradient → params decrease
        state.inner_update(&grad);
        for &p in &state.adapted_params {
            assert!((p - 1.9).abs() < 1e-6, "expected 1.9, got {p}");
        }
    }

    #[test]
    fn test_maml_inner_update_updates_adapted_not_init() {
        let mut state = MamlState::new(vec![5.0f32; 4], 0.5, 1).expect("valid");
        let grad = vec![1.0f32; 4];
        state.inner_update(&grad);
        // init_params unchanged
        assert_eq!(state.init_params, vec![5.0f32; 4]);
        // adapted_params updated
        assert!((state.adapted_params[0] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn test_maml_reset_reverts() {
        let mut state = MamlState::new(vec![3.0f32; 4], 0.1, 1).expect("valid");
        let grad = vec![1.0f32; 4];
        state.inner_update(&grad);
        state.reset();
        assert_eq!(state.adapted_params, state.init_params);
    }

    // ── fsa_inner_loss / gradient tests ─────────────────────────────────────

    #[test]
    fn test_inner_loss_identical_is_zero() {
        let p = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = vec![1.0f32, 2.0, 3.0, 4.0];
        assert!(fsa_inner_loss(&p, &t).abs() < 1e-9);
    }

    #[test]
    fn test_inner_loss_different_positive() {
        let p = vec![1.0f32; 4];
        let t = vec![0.0f32; 4];
        assert!(fsa_inner_loss(&p, &t) > 0.0);
    }

    #[test]
    fn test_inner_loss_known_value() {
        // MSE = sum((p-t)^2) / D = ((2-0)^2 + (2-0)^2) / 2 = 8/2 = 4.0  (d=2)
        let p = vec![2.0f32, 2.0];
        let t = vec![0.0f32, 0.0];
        let loss = fsa_inner_loss(&p, &t);
        assert!((loss - 4.0).abs() < 1e-6, "expected 4.0 got {loss}");
    }

    #[test]
    fn test_inner_gradient_zero_when_equal() {
        let p = vec![1.0f32, 2.0, 3.0];
        let t = vec![1.0f32, 2.0, 3.0];
        let g = fsa_inner_gradient(&p, &t);
        for &gi in &g {
            assert!(gi.abs() < 1e-9, "expected zero gradient");
        }
    }

    #[test]
    fn test_inner_gradient_direction() {
        // params > target → gradient positive → params should decrease
        let p = vec![3.0f32; 4];
        let t = vec![1.0f32; 4];
        let g = fsa_inner_gradient(&p, &t);
        for &gi in &g {
            assert!(gi > 0.0, "gradient should be positive when params > target");
        }
    }

    #[test]
    fn test_inner_gradient_known_value() {
        // d=2, params=2, target=0: g = 2*(2-0)/2 = 2.0
        let p = vec![2.0f32, 2.0];
        let t = vec![0.0f32, 0.0];
        let g = fsa_inner_gradient(&p, &t);
        assert!((g[0] - 2.0).abs() < 1e-6, "expected 2.0 got {}", g[0]);
    }

    // ── fsa_maml_adapt tests ─────────────────────────────────────────────────

    #[test]
    fn test_maml_adapt_loss_decreases() {
        let d = 8;
        let target = vec![5.0f32; d];
        let mut state = MamlState::new(vec![0.0f32; d], 0.1, 50).expect("valid");
        let init_loss = fsa_inner_loss(&state.adapted_params, &target);
        let final_loss = fsa_maml_adapt(&mut state, &target, 1);
        assert!(
            final_loss < init_loss,
            "loss should decrease: init={init_loss}, final={final_loss}"
        );
    }

    #[test]
    fn test_maml_adapt_returns_final_loss() {
        let d = 4;
        let target = vec![1.0f32; d];
        let mut state = MamlState::new(vec![0.0f32; d], 0.5, 10).expect("valid");
        let returned = fsa_maml_adapt(&mut state, &target, 1);
        let computed = fsa_inner_loss(&state.adapted_params, &target);
        assert!((returned - computed).abs() < 1e-6);
    }

    #[test]
    fn test_maml_adapt_converges_to_target() {
        let d = 4;
        let target = vec![3.0f32; d];
        let mut state = MamlState::new(vec![0.0f32; d], 0.5, 200).expect("valid");
        let loss = fsa_maml_adapt(&mut state, &target, 1);
        assert!(loss < 0.01, "should converge near target, loss={loss}");
    }

    // ── fsa_maml_query_loss tests ────────────────────────────────────────────

    #[test]
    fn test_maml_query_loss_zero_when_perfect() {
        let d = 4;
        let target = vec![2.0f32; d];
        let state = MamlState::new(target.clone(), 0.01, 1).expect("valid");
        // query embeddings identical to adapted_params
        let q_embs = target.clone();
        let loss = fsa_maml_query_loss(&state, &q_embs, 1);
        assert!(loss.abs() < 1e-9);
    }

    #[test]
    fn test_maml_query_loss_multiple_queries() {
        let d = 4;
        let adapted = vec![1.0f32; d];
        let state = MamlState::new(adapted, 0.01, 1).expect("valid");
        // 3 queries: 2 identical, 1 different
        let mut q_embs = vec![1.0f32; d]; // match
        q_embs.extend_from_slice(&vec![1.0f32; d]); // match
        q_embs.extend_from_slice(&vec![3.0f32; d]); // mismatch
        let loss = fsa_maml_query_loss(&state, &q_embs, 3);
        assert!(loss > 0.0, "non-zero loss expected for mismatched query");
    }

    // ── LoraAdapter tests ─────────────────────────────────────────────────────

    #[test]
    fn test_lora_correct_matrix_sizes() {
        let adapter = LoraAdapter::new(16, 32, 4, 4.0, 42);
        assert_eq!(adapter.a_matrix.len(), 4 * 16);
        assert_eq!(adapter.b_matrix.len(), 32 * 4);
    }

    #[test]
    fn test_lora_n_params() {
        let adapter = LoraAdapter::new(16, 32, 4, 4.0, 42);
        assert_eq!(adapter.n_params(), 4 * 16 + 32 * 4);
    }

    #[test]
    fn test_lora_forward_output_length() {
        let adapter = LoraAdapter::new(8, 16, 4, 1.0, 1);
        let x = vec![1.0f32; 8];
        let y = adapter.forward(&x);
        assert_eq!(y.len(), 16);
    }

    #[test]
    fn test_lora_forward_b_zero_init_output_zero() {
        // B is initialized to 0, so ΔW = B·A·x = 0 regardless of A and x
        let adapter = LoraAdapter::new(8, 16, 4, 1.0, 99);
        assert!(adapter.b_matrix.iter().all(|&v| v == 0.0));
        let x = vec![1.0f32; 8];
        let y = adapter.forward(&x);
        for &yi in &y {
            assert!(yi.abs() < 1e-9, "output should be zero when B=0, got {yi}");
        }
    }

    #[test]
    fn test_lora_apply_base_plus_zero_adapter() {
        let adapter = LoraAdapter::new(4, 4, 2, 1.0, 7);
        let base = vec![3.0f32, 1.0, 4.0, 1.0];
        let x = vec![1.0f32; 4];
        let result = fsa_lora_apply(&base, &adapter, &x);
        // B=0 so adapter output is 0, result should equal base
        for (&r, &b) in result.iter().zip(base.iter()) {
            assert!((r - b).abs() < 1e-9, "expected {b}, got {r}");
        }
    }

    #[test]
    fn test_lora_n_params_rank1() {
        let adapter = LoraAdapter::new(10, 20, 1, 1.0, 5);
        assert_eq!(adapter.n_params(), 10 + 20);
    }

    // ── fsa_compute_stats tests ───────────────────────────────────────────────

    #[test]
    fn test_compute_stats_mean_min_max() {
        let accs = vec![0.5f32, 0.7, 0.9];
        let n_ways = vec![5usize, 5, 5];
        let k_shots = vec![5usize, 5, 5];
        let s = fsa_compute_stats(&accs, &n_ways, &k_shots).expect("valid");
        assert!((s.mean_accuracy - (0.5 + 0.7 + 0.9) / 3.0).abs() < 1e-5);
        assert!((s.min_accuracy - 0.5).abs() < 1e-5);
        assert!((s.max_accuracy - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_compute_stats_ci_95() {
        let accs = vec![0.5f32, 0.7, 0.9, 0.6, 0.8];
        let n_ways = vec![5usize; 5];
        let k_shots = vec![5usize; 5];
        let s = fsa_compute_stats(&accs, &n_ways, &k_shots).expect("valid");
        let expected_ci = 1.96 * s.std_accuracy / (5.0f32).sqrt();
        assert!((s.ci_95 - expected_ci).abs() < 1e-5);
    }

    #[test]
    fn test_compute_stats_empty_error() {
        let result = fsa_compute_stats(&[], &[], &[]);
        assert!(matches!(result, Err(FewShotError::NoEpisodes)));
    }

    #[test]
    fn test_compute_stats_n_episodes() {
        let accs = vec![0.8f32; 10];
        let n_ways = vec![3usize; 10];
        let k_shots = vec![2usize; 10];
        let s = fsa_compute_stats(&accs, &n_ways, &k_shots).expect("valid");
        assert_eq!(s.n_episodes, 10);
    }

    #[test]
    fn test_compute_stats_single_episode() {
        let accs = vec![0.75f32];
        let s = fsa_compute_stats(&accs, &[3], &[2]).expect("valid");
        assert!((s.mean_accuracy - 0.75).abs() < 1e-5);
        assert!(s.std_accuracy.abs() < 1e-5);
        assert!(s.ci_95.abs() < 1e-5); // std=0 → ci=0
    }

    // ── Format function tests ─────────────────────────────────────────────────

    #[test]
    fn test_format_stats_non_empty() {
        let accs = vec![0.8f32, 0.9];
        let s = fsa_compute_stats(&accs, &[5, 5], &[5, 5]).expect("valid");
        let formatted = fsa_format_stats(&s);
        assert!(!formatted.is_empty());
        assert!(formatted.contains("FewShotStats"));
    }

    #[test]
    fn test_format_config_non_empty() {
        let cfg = FewShotConfig::default_5way_5shot();
        let formatted = fsa_format_config(&cfg);
        assert!(!formatted.is_empty());
        assert!(formatted.contains("FewShotConfig"));
    }

    #[test]
    fn test_format_result_non_empty() {
        let r = AdaptationResult {
            n_support: 25,
            n_query: 75,
            pre_accuracy: 0.3,
            post_accuracy: 0.85,
            n_steps: 10,
            final_loss: 0.015,
            improvement: 0.55,
        };
        let formatted = fsa_format_result(&r);
        assert!(!formatted.is_empty());
        assert!(formatted.contains("AdaptationResult"));
    }

    // ── FewShotConfig tests ───────────────────────────────────────────────────

    #[test]
    fn test_config_5way_5shot() {
        let cfg = FewShotConfig::default_5way_5shot();
        assert_eq!(cfg.n_way, 5);
        assert_eq!(cfg.k_shot, 5);
    }

    #[test]
    fn test_config_5way_1shot_k_shot() {
        let cfg = FewShotConfig::default_5way_1shot();
        assert_eq!(cfg.k_shot, 1);
        assert_eq!(cfg.n_way, 5);
    }

    // ── Error variant tests ───────────────────────────────────────────────────

    #[test]
    fn test_invalid_n_way_error_display() {
        let err = FewShotError::InvalidNWay { n_way: 1 };
        let msg = err.to_string();
        assert!(msg.contains("1"));
    }

    #[test]
    fn test_invalid_k_shot_error_display() {
        let err = FewShotError::InvalidKShot { k_shot: 0 };
        let msg = err.to_string();
        assert!(msg.contains("0"));
    }

    #[test]
    fn test_insufficient_samples_error() {
        let err = FewShotError::InsufficientSamples {
            n_way: 5,
            k_shot: 5,
            have: 3,
        };
        let msg = err.to_string();
        assert!(msg.contains("5"));
    }

    // ── 2-way 1-shot on 4-class data ─────────────────────────────────────────

    #[test]
    fn test_2way_1shot_on_4class_data() {
        let (embs, lbls) = build_synthetic_dataset(4, 10, 8);
        let ep = fsa_sample_episode(&embs, &lbls, 40, 8, 2, 1, 5, 2024)
            .expect("should succeed with 4-class data and 2-way");
        assert_eq!(ep.support.n_way, 2);
        assert_eq!(ep.support.k_shot, 1);
        assert_eq!(ep.support.labels.len(), 2);
        assert_eq!(ep.query.n_query, 10);
        let acc = fsa_episode_accuracy(&ep);
        // Well-separated classes (offset by 10), expect high accuracy
        assert!(
            acc > 0.7,
            "expected decent accuracy on well-separated data, got {acc}"
        );
    }

    // ── Proto loss bounds ─────────────────────────────────────────────────────

    #[test]
    fn test_proto_loss_is_finite() {
        let ss = make_support(3, 2, 8);
        let net = PrototypicalNet::from_support(&ss);
        let query = make_query_for(&net, 6);
        let loss = fsa_proto_loss(&net, &query, 1.0);
        assert!(loss.is_finite(), "proto loss should be finite, got {loss}");
    }

    #[test]
    fn test_proto_loss_non_negative() {
        let ss = make_support(3, 2, 8);
        let net = PrototypicalNet::from_support(&ss);
        let query = make_query_for(&net, 6);
        let loss = fsa_proto_loss(&net, &query, 1.0);
        assert!(loss >= 0.0, "proto loss should be non-negative");
    }

    // ── LoRA non-zero output after B update ──────────────────────────────────

    #[test]
    fn test_lora_non_zero_after_b_update() {
        let mut adapter = LoraAdapter::new(4, 4, 2, 1.0, 13);
        // Manually set B to identity-ish values
        for v in adapter.b_matrix.iter_mut() {
            *v = 1.0;
        }
        let x = vec![1.0f32; 4];
        let y = adapter.forward(&x);
        let any_non_zero = y.iter().any(|&v| v.abs() > 1e-9);
        assert!(any_non_zero, "after setting B, output should be non-zero");
    }

    // ── Additional robustness tests ──────────────────────────────────────────

    #[test]
    fn test_maml_adapt_multiple_support_samples() {
        let d = 4;
        // Two support samples: [1,1,1,1] and [3,3,3,3], mean=[2,2,2,2]
        let support_embs = vec![1.0f32, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0];
        let mut state = MamlState::new(vec![0.0f32; d], 0.5, 100).expect("valid");
        let loss = fsa_maml_adapt(&mut state, &support_embs, 2);
        // Should converge toward [2,2,2,2]
        for &p in &state.adapted_params {
            assert!((p - 2.0).abs() < 0.1, "expected ~2.0, got {p}");
        }
        assert!(loss < 0.01);
    }

    #[test]
    fn test_fsa_maml_adapt_short_buffer_does_not_panic() {
        let mut state = MamlState::new(vec![0.0f32; 4], 0.1, 3).expect("valid maml state");
        // Claim n_support=10 but only provide embeddings for 1 support
        // sample.
        let short_support = vec![1.0f32; 4];
        let loss = fsa_maml_adapt(&mut state, &short_support, 10);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_fsa_episode_id_equals_seed() {
        let (embs, lbls) = build_synthetic_dataset(5, 20, 8);
        let seed = 9999u64;
        let ep = fsa_sample_episode(&embs, &lbls, 100, 8, 3, 3, 4, seed).expect("valid");
        assert_eq!(ep.episode_id, seed);
    }

    #[test]
    fn test_proto_softmax_temperature_effect() {
        // Lower temperature → more confident (higher max prob)
        let d = 2;
        let embs = vec![0.0f32, 0.0, 5.0, 5.0, 10.0, 10.0];
        let labels = vec![0usize, 1, 2];
        let ss = SupportSet::new(embs, labels, 3, 1, d).expect("valid");
        let net = PrototypicalNet::from_support(&ss);
        let query = vec![0.1f32, 0.1];
        let probs_low_t = net.softmax_probs(&query, 0.01);
        let probs_high_t = net.softmax_probs(&query, 100.0);
        let max_low = probs_low_t
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let max_high = probs_high_t
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_low > max_high,
            "lower temperature should give higher max probability"
        );
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let err = FewShotError::DimensionMismatch {
            emb: 10,
            expected: 16,
        };
        let msg = err.to_string();
        assert!(msg.contains("10") && msg.contains("16"));
    }

    #[test]
    fn test_lora_scale_computed_correctly() {
        let adapter = LoraAdapter::new(8, 8, 4, 8.0, 1);
        // scale = alpha / rank = 8.0 / 4 = 2.0
        assert!((adapter.scale - 2.0).abs() < 1e-6);
    }

    // ── fsa_run_episodes ──────────────────────────────────────────────────────

    fn default_run_config(n_way: usize, k_shot: usize, n_query: usize) -> FewShotConfig {
        FewShotConfig {
            n_way,
            k_shot,
            n_query,
            n_episodes: 5,
            inner_lr: 0.5,
            n_inner_steps: 50,
            proto_temperature: 1.0,
            lora_rank: 4,
            lora_alpha: 4.0,
            seed: 123,
        }
    }

    #[test]
    fn test_fsa_run_episodes_pre_accuracy_is_uninformative_baseline() {
        // With zero meta-init prototypes shared by every class, pre-
        // adaptation classification carries no information about the
        // episode: `classify()` always picks class 0 (ties broken toward
        // the first element), so accuracy is exactly 1/n_way on a balanced
        // query set.
        let (embs, lbls) = build_synthetic_dataset(5, 20, 8);
        let config = default_run_config(3, 2, 4);
        let episode = fsa_sample_episode(
            &embs,
            &lbls,
            100,
            8,
            config.n_way,
            config.k_shot,
            config.n_query,
            config.seed,
        )
        .expect("episode should sample");
        let sup = &episode.support;
        let pre_net = PrototypicalNet {
            prototypes: vec![0.0f32; sup.n_way * sup.d],
            n_way: sup.n_way,
            d: sup.d,
        };
        let pre_accuracy = fsa_proto_accuracy(&pre_net, &episode.query);
        assert!(
            (pre_accuracy - 1.0 / config.n_way as f32).abs() < 1e-5,
            "with identical (zero) prototypes, classify() always picks class 0, \
             giving exactly 1/n_way accuracy; got {pre_accuracy}"
        );
    }

    #[test]
    fn test_fsa_run_episodes_post_accuracy_converges_to_closed_form() {
        let (embs, lbls) = build_synthetic_dataset(5, 30, 4);
        let config = FewShotConfig {
            n_way: 3,
            k_shot: 3,
            n_query: 5,
            n_episodes: 1,
            inner_lr: 0.9,
            n_inner_steps: 200, // enough for near-exact convergence
            proto_temperature: 1.0,
            lora_rank: 2,
            lora_alpha: 2.0,
            seed: 7,
        };
        let episode = fsa_sample_episode(
            &embs,
            &lbls,
            150,
            4,
            config.n_way,
            config.k_shot,
            config.n_query,
            config.seed,
        )
        .expect("episode should sample");
        let sup = &episode.support;

        // Ground truth: the closed-form per-class mean.
        let closed_form_net = PrototypicalNet::from_support(sup);
        let closed_form_accuracy = fsa_proto_accuracy(&closed_form_net, &episode.query);

        // MAML-adapted prototypes, starting from zero, targeting the same
        // per-class support mean that the closed form computes directly.
        let mut adapted = vec![0.0f32; sup.n_way * sup.d];
        for c in 0..sup.n_way {
            let start = c * sup.k_shot * sup.d;
            let end = start + sup.k_shot * sup.d;
            let class_support = &sup.embeddings[start..end];
            let mut state =
                MamlState::new(vec![0.0f32; sup.d], config.inner_lr, config.n_inner_steps)
                    .expect("valid maml state");
            fsa_maml_adapt(&mut state, class_support, sup.k_shot);
            adapted[c * sup.d..(c + 1) * sup.d].copy_from_slice(&state.adapted_params);
        }
        let post_net = PrototypicalNet {
            prototypes: adapted,
            n_way: sup.n_way,
            d: sup.d,
        };
        let post_accuracy = fsa_proto_accuracy(&post_net, &episode.query);

        assert!(
            (post_accuracy - closed_form_accuracy).abs() < 1e-4,
            "with enough inner steps, MAML-adapted prototypes should converge to \
             the closed-form mean: closed_form={closed_form_accuracy}, post={post_accuracy}"
        );
    }

    #[test]
    fn test_fsa_run_episodes_produces_stats_and_positive_improvement() {
        let (embs, lbls) = build_synthetic_dataset(5, 20, 8);
        let config = default_run_config(3, 2, 4);
        let stats = fsa_run_episodes(&embs, &lbls, 100, 8, &config).expect("episodes should run");
        assert_eq!(stats.n_episodes, 5);
        assert!(stats.mean_accuracy >= 0.0 && stats.mean_accuracy <= 1.0);
        // Post-adaptation accuracy on this well-separated synthetic dataset
        // should comfortably beat the 1/n_way chance baseline the
        // (uninformative) pre-adaptation prototypes are limited to.
        assert!(
            stats.mean_accuracy > 1.0 / config.n_way as f32,
            "post-adaptation accuracy should exceed the pre-adaptation chance \
             baseline, got {}",
            stats.mean_accuracy
        );
    }

    #[test]
    fn test_fsa_run_episodes_invalid_lr_errors() {
        let (embs, lbls) = build_synthetic_dataset(5, 20, 8);
        let mut config = default_run_config(3, 2, 4);
        config.inner_lr = -1.0;
        let result = fsa_run_episodes(&embs, &lbls, 100, 8, &config);
        assert!(matches!(result, Err(FewShotError::InvalidLR { .. })));
    }
}
