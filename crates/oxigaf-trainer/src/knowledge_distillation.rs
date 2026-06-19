//! Knowledge distillation utilities for compressing teacher diffusion models
//! into lightweight student models in the OxiGAF training pipeline.
//!
//! Provides soft-label KL divergence, feature matching, attention transfer
//! (Zagoruyko & Komodakis 2017), and relational KD (Park et al. 2019).
//!
//! # Quick start
//! ```rust
//! use oxigaf_trainer::knowledge_distillation::{
//!     DistillationConfig, kd_combined_loss,
//! };
//!
//! let config = DistillationConfig::default();
//! let student_logits = vec![1.8_f32, 1.1, 0.6];
//! let teacher_logits = vec![2.0_f32, 1.0, 0.5];
//! let target        = vec![0.0_f32, 1.0, 0.0];
//! let loss = kd_combined_loss(&student_logits, &teacher_logits, &target, &config)
//!     .expect("combined loss");
//! println!("total loss: {loss}");
//! ```

use thiserror::Error;

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
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the knowledge-distillation subsystem.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum DistillationError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("invalid temperature: must be > 0, got {0}")]
    InvalidTemperature(f32),

    #[error("invalid config: {0}")]
    InvalidConfig(String),

    #[error("empty input")]
    EmptyInput,

    #[error("mismatched layers: teacher {teacher}, student {student}")]
    LayerCountMismatch { teacher: usize, student: usize },
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for knowledge distillation training.
///
/// Final loss:
/// `alpha * hard_loss + (1 - alpha) * T² * soft_loss
///  + feature_weight * feature_loss
///  + attention_weight * attention_loss
///  + relational_weight * relational_loss`
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// Softmax temperature `T` (default 4.0).
    pub temperature: f32,
    /// Blend: `alpha*hard + (1-alpha)*soft` (default 0.5).
    pub alpha: f32,
    /// Weight for feature matching loss term (default 0.01).
    pub feature_weight: f32,
    /// Weight for attention transfer loss term (default 0.001).
    pub attention_weight: f32,
    /// Weight for relational KD loss term (default 0.0001).
    pub relational_weight: f32,
    /// Pseudo-Huber `c` for relational loss (default 1.0).
    pub beta: f32,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.5,
            feature_weight: 0.01,
            attention_weight: 0.001,
            relational_weight: 0.0001,
            beta: 1.0,
        }
    }
}

impl DistillationConfig {
    /// Validate that all fields are within acceptable ranges.
    pub fn validate(&self) -> Result<(), DistillationError> {
        if self.temperature <= 0.0 {
            return Err(DistillationError::InvalidTemperature(self.temperature));
        }
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err(DistillationError::InvalidConfig(format!(
                "alpha must be in [0,1], got {}",
                self.alpha
            )));
        }
        if self.feature_weight < 0.0 {
            return Err(DistillationError::InvalidConfig(format!(
                "feature_weight must be >= 0, got {}",
                self.feature_weight
            )));
        }
        if self.attention_weight < 0.0 {
            return Err(DistillationError::InvalidConfig(format!(
                "attention_weight must be >= 0, got {}",
                self.attention_weight
            )));
        }
        if self.relational_weight < 0.0 {
            return Err(DistillationError::InvalidConfig(format!(
                "relational_weight must be >= 0, got {}",
                self.relational_weight
            )));
        }
        if self.beta <= 0.0 {
            return Err(DistillationError::InvalidConfig(format!(
                "beta must be > 0, got {}",
                self.beta
            )));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss breakdown
// ─────────────────────────────────────────────────────────────────────────────

/// Component-wise distillation loss values.
#[derive(Debug, Clone)]
pub struct DistillationLoss {
    /// Weighted sum of all components.
    pub total: f32,
    /// Hard MSE term.
    pub hard: f32,
    /// Soft KL divergence term (pre T² scale applied internally).
    pub soft: f32,
    /// Feature matching term.
    pub feature: f32,
    /// Attention transfer term.
    pub attention: f32,
    /// Relational KD term.
    pub relational: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Temperature-scaled softmax (numerically stable via max subtraction).
///
/// Returns a probability vector that sums to 1.
pub fn kd_softmax_with_temperature(
    logits: &[f32],
    temperature: f32,
) -> Result<Vec<f32>, DistillationError> {
    if logits.is_empty() {
        return Err(DistillationError::EmptyInput);
    }
    if temperature <= 0.0 {
        return Err(DistillationError::InvalidTemperature(temperature));
    }
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 || !sum.is_finite() {
        // Fallback: uniform distribution
        let n = logits.len();
        return Ok(vec![1.0 / n as f32; n]);
    }
    Ok(exps.iter().map(|&e| e / sum).collect())
}

/// KL divergence `KL(p ‖ q) = Σ p_i · log(p_i / q_i)`.
///
/// Both `p` and `q` must be valid probability distributions of the same length.
/// Log arguments are clamped to `1e-8` for numerical stability.
pub fn kd_kl_divergence(p: &[f32], q: &[f32]) -> Result<f32, DistillationError> {
    if p.is_empty() {
        return Err(DistillationError::EmptyInput);
    }
    if p.len() != q.len() {
        return Err(DistillationError::DimensionMismatch {
            expected: p.len(),
            got: q.len(),
        });
    }
    let kl: f32 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            if pi < 1e-15 {
                0.0
            } else {
                let qi_c = qi.max(1e-8_f32);
                let pi_c = pi.max(1e-8_f32);
                pi * (pi_c / qi_c).ln()
            }
        })
        .sum();
    Ok(kl.max(0.0))
}

/// Cosine similarity between two embeddings: `a·b / (|a| |b|)`.
pub fn kd_cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, DistillationError> {
    if a.is_empty() {
        return Err(DistillationError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(DistillationError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return Ok(0.0);
    }
    Ok((dot / (norm_a * norm_b)).clamp(-1.0, 1.0))
}

/// Compute the attention map from a feature tensor `[H*W*C]`.
///
/// `A_ij = Σ_c F²_ijc`, then the resulting `[H*W]` vector is L2-normalised.
pub fn kd_attention_map(
    features: &[f32],
    height: usize,
    width: usize,
    channels: usize,
) -> Result<Vec<f32>, DistillationError> {
    let expected = height * width * channels;
    if expected == 0 {
        return Err(DistillationError::EmptyInput);
    }
    if features.len() != expected {
        return Err(DistillationError::DimensionMismatch {
            expected,
            got: features.len(),
        });
    }
    let hw = height * width;
    let mut map = vec![0.0_f32; hw];
    for (idx, &v) in features.iter().enumerate() {
        let spatial = idx / channels;
        map[spatial] += v * v;
    }
    let norm: f32 = map.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for v in map.iter_mut() {
            *v /= norm;
        }
    }
    Ok(map)
}

/// Pairwise L2 distances (upper-triangular, row-major) for `batch * embed_dim` embeddings.
///
/// Returns a flat Vec of length `batch*(batch-1)/2`.
pub fn kd_pairwise_distances(
    embeddings: &[f32],
    embed_dim: usize,
) -> Result<Vec<f32>, DistillationError> {
    if embeddings.is_empty() || embed_dim == 0 {
        return Err(DistillationError::EmptyInput);
    }
    if !embeddings.len().is_multiple_of(embed_dim) {
        return Err(DistillationError::DimensionMismatch {
            expected: embed_dim,
            got: embeddings.len() % embed_dim,
        });
    }
    let batch = embeddings.len() / embed_dim;
    if batch < 2 {
        return Ok(Vec::new());
    }
    let n_pairs = batch * (batch - 1) / 2;
    let mut dists = Vec::with_capacity(n_pairs);
    for i in 0..batch {
        for j in (i + 1)..batch {
            let row_i = &embeddings[i * embed_dim..(i + 1) * embed_dim];
            let row_j = &embeddings[j * embed_dim..(j + 1) * embed_dim];
            let sq: f32 = row_i
                .iter()
                .zip(row_j.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            dists.push(sq.sqrt());
        }
    }
    Ok(dists)
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss functions
// ─────────────────────────────────────────────────────────────────────────────

/// Soft-label KL divergence: `KL(student_soft ‖ teacher_soft)` with temperature `T`.
///
/// Applies softmax with temperature to both, then computes KL divergence.
/// Returns a scalar loss.
pub fn kd_soft_loss(
    student_logits: &[f32],
    teacher_logits: &[f32],
    temperature: f32,
) -> Result<f32, DistillationError> {
    if student_logits.is_empty() {
        return Err(DistillationError::EmptyInput);
    }
    if student_logits.len() != teacher_logits.len() {
        return Err(DistillationError::DimensionMismatch {
            expected: student_logits.len(),
            got: teacher_logits.len(),
        });
    }
    if temperature <= 0.0 {
        return Err(DistillationError::InvalidTemperature(temperature));
    }
    let student_soft = kd_softmax_with_temperature(student_logits, temperature)?;
    let teacher_soft = kd_softmax_with_temperature(teacher_logits, temperature)?;
    kd_kl_divergence(&student_soft, &teacher_soft)
}

/// Hard MSE loss between student output and ground-truth target.
///
/// `loss = mean((student - target)²)`
pub fn kd_hard_loss(student_output: &[f32], target: &[f32]) -> Result<f32, DistillationError> {
    if student_output.is_empty() {
        return Err(DistillationError::EmptyInput);
    }
    if student_output.len() != target.len() {
        return Err(DistillationError::DimensionMismatch {
            expected: student_output.len(),
            got: target.len(),
        });
    }
    let mse: f32 = student_output
        .iter()
        .zip(target.iter())
        .map(|(&s, &t)| (s - t) * (s - t))
        .sum::<f32>()
        / student_output.len() as f32;
    Ok(mse)
}

/// Combined hard + soft loss.
///
/// `loss = alpha * hard_loss + (1 - alpha) * T² * soft_loss`
///
/// The `T²` factor compensates for reduced gradient magnitude from temperature scaling.
pub fn kd_combined_loss(
    student_logits: &[f32],
    teacher_logits: &[f32],
    target: &[f32],
    config: &DistillationConfig,
) -> Result<f32, DistillationError> {
    config.validate()?;
    let hard = kd_hard_loss(student_logits, target)?;
    let soft = kd_soft_loss(student_logits, teacher_logits, config.temperature)?;
    let t2 = config.temperature * config.temperature;
    Ok(config.alpha * hard + (1.0 - config.alpha) * t2 * soft)
}

/// Feature matching loss (MSE between intermediate feature maps).
///
/// `student_features` and `teacher_features`: each is `Vec<Vec<f32>>` (one per layer).
pub fn kd_feature_loss(
    student_features: &[Vec<f32>],
    teacher_features: &[Vec<f32>],
) -> Result<f32, DistillationError> {
    if student_features.is_empty() || teacher_features.is_empty() {
        return Err(DistillationError::EmptyInput);
    }
    if student_features.len() != teacher_features.len() {
        return Err(DistillationError::LayerCountMismatch {
            teacher: teacher_features.len(),
            student: student_features.len(),
        });
    }
    let mut total_mse = 0.0_f32;
    let mut total_elements: usize = 0;
    for (sf, tf) in student_features.iter().zip(teacher_features.iter()) {
        if sf.len() != tf.len() {
            return Err(DistillationError::DimensionMismatch {
                expected: tf.len(),
                got: sf.len(),
            });
        }
        for (&s, &t) in sf.iter().zip(tf.iter()) {
            total_mse += (s - t) * (s - t);
        }
        total_elements += sf.len();
    }
    if total_elements == 0 {
        return Ok(0.0);
    }
    Ok(total_mse / total_elements as f32)
}

/// Attention transfer loss (Zagoruyko & Komodakis 2017).
///
/// `AT(A)_ij = Σ_c F²_ijc`, normalised by its L2 norm.
/// Loss = `‖AT(student) - AT(teacher)‖²_F`
///
/// `student_map` / `teacher_map`: flat `[H*W*C]`, spatial dims `height × width × channels`.
pub fn kd_attention_transfer_loss(
    student_map: &[f32],
    teacher_map: &[f32],
    height: usize,
    width: usize,
    channels: usize,
) -> Result<f32, DistillationError> {
    let expected = height * width * channels;
    if expected == 0 {
        return Err(DistillationError::EmptyInput);
    }
    if student_map.len() != expected {
        return Err(DistillationError::DimensionMismatch {
            expected,
            got: student_map.len(),
        });
    }
    if teacher_map.len() != expected {
        return Err(DistillationError::DimensionMismatch {
            expected,
            got: teacher_map.len(),
        });
    }
    let at_s = kd_attention_map(student_map, height, width, channels)?;
    let at_t = kd_attention_map(teacher_map, height, width, channels)?;
    let loss: f32 = at_s
        .iter()
        .zip(at_t.iter())
        .map(|(&s, &t)| (s - t) * (s - t))
        .sum();
    Ok(loss)
}

/// Relational knowledge distillation (Park et al. 2019).
///
/// Computes pairwise distance matrix for a batch of embeddings, then penalises
/// discrepancy between teacher and student distances using pseudo-Huber distance:
/// `sqrt((d_teacher - d_student)² + beta²) - beta`
///
/// `embeddings`: flat `[batch_size * embed_dim]`.
pub fn kd_relational_loss(
    student_embeddings: &[f32],
    teacher_embeddings: &[f32],
    embed_dim: usize,
    beta: f32,
) -> Result<f32, DistillationError> {
    if student_embeddings.is_empty() || teacher_embeddings.is_empty() {
        return Err(DistillationError::EmptyInput);
    }
    if embed_dim == 0 {
        return Err(DistillationError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if student_embeddings.len() != teacher_embeddings.len() {
        return Err(DistillationError::DimensionMismatch {
            expected: teacher_embeddings.len(),
            got: student_embeddings.len(),
        });
    }
    if !student_embeddings.len().is_multiple_of(embed_dim) {
        return Err(DistillationError::DimensionMismatch {
            expected: embed_dim,
            got: student_embeddings.len() % embed_dim,
        });
    }
    let batch = student_embeddings.len() / embed_dim;
    if batch < 2 {
        // Single sample → no pairwise distances possible
        return Ok(0.0);
    }
    let dists_s = kd_pairwise_distances(student_embeddings, embed_dim)?;
    let dists_t = kd_pairwise_distances(teacher_embeddings, embed_dim)?;
    let n_pairs = dists_s.len();
    let loss: f32 = dists_s
        .iter()
        .zip(dists_t.iter())
        .map(|(&ds, &dt)| {
            let diff = dt - ds;
            (diff * diff + beta * beta).sqrt() - beta
        })
        .sum::<f32>()
        / n_pairs as f32;
    Ok(loss)
}

/// Total distillation loss combining all terms.
///
/// Attention and relational terms are set to 0.0 (caller must compute them
/// separately if needed via `kd_attention_transfer_loss` / `kd_relational_loss`).
pub fn kd_total_loss(
    student_logits: &[f32],
    teacher_logits: &[f32],
    target: &[f32],
    student_features: &[Vec<f32>],
    teacher_features: &[Vec<f32>],
    config: &DistillationConfig,
) -> Result<DistillationLoss, DistillationError> {
    config.validate()?;
    let hard = kd_hard_loss(student_logits, target)?;
    let soft = kd_soft_loss(student_logits, teacher_logits, config.temperature)?;
    let t2 = config.temperature * config.temperature;
    let feature = if !student_features.is_empty() && !teacher_features.is_empty() {
        kd_feature_loss(student_features, teacher_features)?
    } else {
        0.0
    };
    let total =
        config.alpha * hard + (1.0 - config.alpha) * t2 * soft + config.feature_weight * feature;
    Ok(DistillationLoss {
        total,
        hard,
        soft,
        feature,
        attention: 0.0,
        relational: 0.0,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Format a `DistillationLoss` for logging.
pub fn kd_format_loss(loss: &DistillationLoss) -> String {
    format!(
        "total={:.6} hard={:.6} soft={:.6} feature={:.6} attention={:.6} relational={:.6}",
        loss.total, loss.hard, loss.soft, loss.feature, loss.attention, loss.relational
    )
}

/// Format `DistillationStats` for logging.
pub fn kd_format_stats(stats: &DistillationStats) -> String {
    format!(
        "steps={} mean_total={:.6} mean_hard={:.6} mean_soft={:.6} \
         mean_feature={:.6} ema_total={:.6}",
        stats.steps,
        stats.mean_total_loss,
        stats.mean_hard_loss,
        stats.mean_soft_loss,
        stats.mean_feature_loss,
        stats.ema_total_loss,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tracker
// ─────────────────────────────────────────────────────────────────────────────

const EMA_DECAY: f32 = 0.99;
const HISTORY_CAP: usize = 2000;

/// Running statistics aggregated over distillation steps.
#[derive(Debug, Clone)]
pub struct DistillationStats {
    pub mean_total_loss: f32,
    pub mean_hard_loss: f32,
    pub mean_soft_loss: f32,
    pub mean_feature_loss: f32,
    pub steps: usize,
    /// EMA of total loss with decay 0.99.
    pub ema_total_loss: f32,
}

impl Default for DistillationStats {
    fn default() -> Self {
        Self {
            mean_total_loss: 0.0,
            mean_hard_loss: 0.0,
            mean_soft_loss: 0.0,
            mean_feature_loss: 0.0,
            steps: 0,
            ema_total_loss: 0.0,
        }
    }
}

/// History tracker for distillation training.
pub struct DistillationHistory {
    config: DistillationConfig,
    stats: DistillationStats,
    /// Capped at [`HISTORY_CAP`] entries.
    loss_history: Vec<f32>,
    /// Running sums for incremental mean computation.
    sum_total: f32,
    sum_hard: f32,
    sum_soft: f32,
    sum_feature: f32,
}

impl DistillationHistory {
    /// Create a new tracker with the given configuration.
    pub fn new(config: DistillationConfig) -> Self {
        Self {
            config,
            stats: DistillationStats::default(),
            loss_history: Vec::new(),
            sum_total: 0.0,
            sum_hard: 0.0,
            sum_soft: 0.0,
            sum_feature: 0.0,
        }
    }

    /// Record a completed distillation loss.
    pub fn record(&mut self, loss: &DistillationLoss) {
        let n = self.stats.steps + 1;
        self.sum_total += loss.total;
        self.sum_hard += loss.hard;
        self.sum_soft += loss.soft;
        self.sum_feature += loss.feature;
        self.stats.steps = n;
        self.stats.mean_total_loss = self.sum_total / n as f32;
        self.stats.mean_hard_loss = self.sum_hard / n as f32;
        self.stats.mean_soft_loss = self.sum_soft / n as f32;
        self.stats.mean_feature_loss = self.sum_feature / n as f32;
        if n == 1 {
            self.stats.ema_total_loss = loss.total;
        } else {
            self.stats.ema_total_loss =
                EMA_DECAY * self.stats.ema_total_loss + (1.0 - EMA_DECAY) * loss.total;
        }
        if self.loss_history.len() >= HISTORY_CAP {
            self.loss_history.remove(0);
        }
        self.loss_history.push(loss.total);
    }

    /// Immutable access to current statistics.
    pub fn stats(&self) -> &DistillationStats {
        &self.stats
    }

    /// Immutable access to loss history (capped at 2000 entries).
    pub fn loss_history(&self) -> &[f32] {
        &self.loss_history
    }

    /// Return true if the last `window` steps have non-increasing mean loss.
    ///
    /// Computes the mean of the first half vs. the second half of the window.
    /// "Non-increasing" means second-half mean ≤ first-half mean.
    /// Returns `false` if fewer than `window` steps have been recorded or `window < 2`.
    pub fn is_converging(&self, window: usize) -> bool {
        if window < 2 || self.loss_history.len() < window {
            return false;
        }
        let slice = &self.loss_history[self.loss_history.len() - window..];
        let half = window / 2;
        let first_mean: f32 = slice[..half].iter().sum::<f32>() / half as f32;
        let second_mean: f32 = slice[half..].iter().sum::<f32>() / (window - half) as f32;
        second_mean <= first_mean
    }

    /// Immutable borrow of the config used at construction time.
    pub fn config(&self) -> &DistillationConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy compatibility aliases (kept for lib.rs re-exports)
// ─────────────────────────────────────────────────────────────────────────────

/// Outputs collected from a model for one sample.
/// Kept for lib.rs compatibility.
#[derive(Debug, Clone)]
pub struct TeacherResponse {
    pub logits: Vec<f32>,
    pub features: Vec<Vec<f32>>,
    pub embedding: Vec<f32>,
}

impl TeacherResponse {
    pub fn new(logits: Vec<f32>) -> Self {
        Self {
            logits,
            features: Vec::new(),
            embedding: Vec::new(),
        }
    }

    pub fn with_features(mut self, features: Vec<Vec<f32>>) -> Self {
        self.features = features;
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = embedding;
        self
    }
}

/// Alias — teacher and student share the same shape.
pub type StudentResponse = TeacherResponse;

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    // ── DistillationConfig::default ──────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let cfg = DistillationConfig::default();
        assert!(approx(cfg.temperature, 4.0, 1e-6));
        assert!(approx(cfg.alpha, 0.5, 1e-6));
        assert!(approx(cfg.feature_weight, 0.01, 1e-6));
        assert!(approx(cfg.attention_weight, 0.001, 1e-6));
        assert!(approx(cfg.relational_weight, 0.0001, 1e-6));
        assert!(approx(cfg.beta, 1.0, 1e-6));
    }

    #[test]
    fn test_config_validate_ok() {
        assert!(DistillationConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_validate_zero_temperature() {
        let c = DistillationConfig {
            temperature: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            c.validate(),
            Err(DistillationError::InvalidTemperature(_))
        ));
    }

    #[test]
    fn test_config_validate_negative_temperature() {
        let c = DistillationConfig {
            temperature: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            c.validate(),
            Err(DistillationError::InvalidTemperature(_))
        ));
    }

    #[test]
    fn test_config_validate_alpha_above_one() {
        let c = DistillationConfig {
            alpha: 1.5,
            ..Default::default()
        };
        assert!(matches!(
            c.validate(),
            Err(DistillationError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_config_validate_negative_alpha() {
        let c = DistillationConfig {
            alpha: -0.1,
            ..Default::default()
        };
        assert!(matches!(
            c.validate(),
            Err(DistillationError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_config_validate_negative_feature_weight() {
        let c = DistillationConfig {
            feature_weight: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            c.validate(),
            Err(DistillationError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_config_validate_zero_beta() {
        let c = DistillationConfig {
            beta: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            c.validate(),
            Err(DistillationError::InvalidConfig(_))
        ));
    }

    // ── kd_softmax_with_temperature ──────────────────────────────────────────

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let probs = kd_softmax_with_temperature(&logits, 1.0).unwrap();
        let sum: f32 = probs.iter().sum();
        assert!(approx(sum, 1.0, 1e-5), "sum={sum}");
    }

    #[test]
    fn test_softmax_t1_matches_standard_softmax() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let probs = kd_softmax_with_temperature(&logits, 1.0).unwrap();
        assert!(probs[0] < probs[1] && probs[1] < probs[2]);
        assert!(approx(probs.iter().sum(), 1.0, 1e-5));
    }

    #[test]
    fn test_softmax_high_temperature_approaches_uniform() {
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let n = logits.len();
        let probs = kd_softmax_with_temperature(&logits, 1000.0).unwrap();
        let expected = 1.0 / n as f32;
        for &p in &probs {
            assert!(approx(p, expected, 1e-3), "p={p}");
        }
    }

    #[test]
    fn test_softmax_invalid_temperature() {
        let r = kd_softmax_with_temperature(&[1.0], 0.0);
        assert!(matches!(r, Err(DistillationError::InvalidTemperature(_))));
    }

    #[test]
    fn test_softmax_empty_logits() {
        let r = kd_softmax_with_temperature(&[], 1.0);
        assert!(matches!(r, Err(DistillationError::EmptyInput)));
    }

    // ── kd_kl_divergence ─────────────────────────────────────────────────────

    #[test]
    fn test_kl_identical_is_zero() {
        let p = vec![0.25_f32; 4];
        let kl = kd_kl_divergence(&p, &p).unwrap();
        assert!(approx(kl, 0.0, 1e-6), "kl={kl}");
    }

    #[test]
    fn test_kl_different_is_positive() {
        let p = vec![0.5_f32, 0.5];
        let q = vec![0.25_f32, 0.75];
        let kl = kd_kl_divergence(&p, &q).unwrap();
        assert!(kl > 0.0, "kl={kl}");
    }

    #[test]
    fn test_kl_asymmetric() {
        let p = vec![0.5_f32, 0.5];
        let q = vec![0.1_f32, 0.9];
        let kl_pq = kd_kl_divergence(&p, &q).unwrap();
        let kl_qp = kd_kl_divergence(&q, &p).unwrap();
        // KL is not symmetric in general
        assert!((kl_pq - kl_qp).abs() > 1e-4, "expected asymmetry");
    }

    #[test]
    fn test_kl_empty() {
        assert!(matches!(
            kd_kl_divergence(&[], &[]),
            Err(DistillationError::EmptyInput)
        ));
    }

    #[test]
    fn test_kl_dimension_mismatch() {
        assert!(matches!(
            kd_kl_divergence(&[0.5, 0.5], &[0.3, 0.3, 0.4]),
            Err(DistillationError::DimensionMismatch { .. })
        ));
    }

    // ── kd_soft_loss ─────────────────────────────────────────────────────────

    #[test]
    fn test_soft_loss_same_logits_near_zero() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let loss = kd_soft_loss(&logits, &logits, 4.0).unwrap();
        assert!(approx(loss, 0.0, 1e-5), "loss={loss}");
    }

    #[test]
    fn test_soft_loss_different_is_positive() {
        let s = vec![1.0_f32, 0.0, 0.0];
        let t = vec![0.0_f32, 1.0, 0.0];
        let loss = kd_soft_loss(&s, &t, 4.0).unwrap();
        assert!(loss > 0.0, "loss={loss}");
    }

    #[test]
    fn test_soft_loss_invalid_temperature() {
        let r = kd_soft_loss(&[1.0], &[1.0], 0.0);
        assert!(matches!(r, Err(DistillationError::InvalidTemperature(_))));
    }

    #[test]
    fn test_soft_loss_empty() {
        let r = kd_soft_loss(&[], &[], 1.0);
        assert!(matches!(r, Err(DistillationError::EmptyInput)));
    }

    #[test]
    fn test_soft_loss_negative_temperature() {
        let r = kd_soft_loss(&[1.0, 2.0], &[1.0, 2.0], -1.0);
        assert!(matches!(r, Err(DistillationError::InvalidTemperature(_))));
    }

    // ── kd_hard_loss ─────────────────────────────────────────────────────────

    #[test]
    fn test_hard_loss_identical_is_zero() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let loss = kd_hard_loss(&v, &v).unwrap();
        assert!(approx(loss, 0.0, 1e-7));
    }

    #[test]
    fn test_hard_loss_different_is_positive() {
        let loss = kd_hard_loss(&[1.0_f32], &[0.0]).unwrap();
        assert!(loss > 0.0);
    }

    #[test]
    fn test_hard_loss_known_value() {
        // MSE([1,0], [0,0]) = (1+0)/2 = 0.5
        let loss = kd_hard_loss(&[1.0_f32, 0.0], &[0.0_f32, 0.0]).unwrap();
        assert!(approx(loss, 0.5, 1e-6));
    }

    #[test]
    fn test_hard_loss_empty() {
        assert!(matches!(
            kd_hard_loss(&[], &[]),
            Err(DistillationError::EmptyInput)
        ));
    }

    #[test]
    fn test_hard_loss_dimension_mismatch() {
        assert!(matches!(
            kd_hard_loss(&[1.0, 2.0], &[1.0]),
            Err(DistillationError::DimensionMismatch { .. })
        ));
    }

    // ── kd_combined_loss ─────────────────────────────────────────────────────

    #[test]
    fn test_combined_alpha1_equals_pure_hard() {
        let s = vec![1.0_f32, 0.5];
        let t = vec![0.8_f32, 0.6];
        let target = vec![1.0_f32, 0.0];
        let cfg = DistillationConfig {
            alpha: 1.0,
            ..Default::default()
        };
        let combined = kd_combined_loss(&s, &t, &target, &cfg).unwrap();
        let hard = kd_hard_loss(&s, &target).unwrap();
        assert!(approx(combined, hard, 1e-5));
    }

    #[test]
    fn test_combined_alpha0_only_soft_scaled() {
        let s = vec![1.0_f32, 0.5];
        let t = vec![0.8_f32, 0.6];
        let target = vec![1.0_f32, 0.0];
        let cfg = DistillationConfig {
            alpha: 0.0,
            temperature: 2.0,
            ..Default::default()
        };
        let combined = kd_combined_loss(&s, &t, &target, &cfg).unwrap();
        let soft = kd_soft_loss(&s, &t, cfg.temperature).unwrap();
        let expected = (1.0 - 0.0) * 4.0 * soft; // T²=4
        assert!(approx(combined, expected, 1e-5));
    }

    #[test]
    fn test_combined_non_negative() {
        let s = vec![2.0_f32, 1.0, 0.5];
        let t = vec![1.8_f32, 1.1, 0.6];
        let target = vec![0.0_f32, 1.0, 0.0];
        let combined = kd_combined_loss(&s, &t, &target, &DistillationConfig::default()).unwrap();
        assert!(combined >= 0.0, "combined={combined}");
    }

    // ── kd_feature_loss ───────────────────────────────────────────────────────

    #[test]
    fn test_feature_loss_identical_near_zero() {
        let feats = vec![vec![1.0_f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let loss = kd_feature_loss(&feats, &feats).unwrap();
        assert!(approx(loss, 0.0, 1e-7));
    }

    #[test]
    fn test_feature_loss_different_positive() {
        let s = vec![vec![0.0_f32, 0.0]];
        let t = vec![vec![1.0_f32, 1.0]];
        let loss = kd_feature_loss(&s, &t).unwrap();
        assert!(loss > 0.0, "loss={loss}");
    }

    #[test]
    fn test_feature_loss_known_value() {
        // MSE([1,0] vs [0,0]) = 0.5
        let s = vec![vec![1.0_f32, 0.0]];
        let t = vec![vec![0.0_f32, 0.0]];
        let loss = kd_feature_loss(&s, &t).unwrap();
        assert!(approx(loss, 0.5, 1e-6));
    }

    #[test]
    fn test_feature_loss_layer_count_mismatch() {
        let s = vec![vec![1.0_f32]];
        let t = vec![vec![1.0_f32], vec![2.0]];
        assert!(matches!(
            kd_feature_loss(&s, &t),
            Err(DistillationError::LayerCountMismatch { .. })
        ));
    }

    #[test]
    fn test_feature_loss_empty() {
        let r = kd_feature_loss(&[], &[]);
        assert!(matches!(r, Err(DistillationError::EmptyInput)));
    }

    #[test]
    fn test_feature_loss_dim_mismatch_within_layer() {
        let s = vec![vec![1.0_f32, 2.0]];
        let t = vec![vec![1.0_f32, 2.0, 3.0]];
        assert!(matches!(
            kd_feature_loss(&s, &t),
            Err(DistillationError::DimensionMismatch { .. })
        ));
    }

    // ── kd_attention_map ──────────────────────────────────────────────────────

    #[test]
    fn test_attention_map_output_length() {
        let feat = vec![1.0_f32; 4 * 4 * 8];
        let map = kd_attention_map(&feat, 4, 4, 8).unwrap();
        assert_eq!(map.len(), 16);
    }

    #[test]
    fn test_attention_map_normalised() {
        let feat = vec![1.0_f32; 3 * 3 * 4];
        let map = kd_attention_map(&feat, 3, 3, 4).unwrap();
        let norm: f32 = map.iter().map(|&x| x * x).sum::<f32>().sqrt();
        // Norm should be ~1 (L2-normalised) unless all-zero
        assert!(approx(norm, 1.0, 1e-5), "norm={norm}");
    }

    #[test]
    fn test_attention_map_empty_dims() {
        let r = kd_attention_map(&[], 0, 0, 0);
        assert!(matches!(r, Err(DistillationError::EmptyInput)));
    }

    #[test]
    fn test_attention_map_wrong_size() {
        let r = kd_attention_map(&[1.0; 5], 2, 2, 2);
        assert!(matches!(
            r,
            Err(DistillationError::DimensionMismatch { .. })
        ));
    }

    // ── kd_attention_transfer_loss ────────────────────────────────────────────

    #[test]
    fn test_attention_transfer_identical_near_zero() {
        let map = vec![1.0_f32; 4 * 4 * 2];
        let loss = kd_attention_transfer_loss(&map, &map, 4, 4, 2).unwrap();
        assert!(approx(loss, 0.0, 1e-6));
    }

    #[test]
    fn test_attention_transfer_different_positive() {
        let s: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let t: Vec<f32> = (0..8).map(|i| (7 - i) as f32).collect();
        let loss = kd_attention_transfer_loss(&s, &t, 2, 2, 2).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_attention_transfer_dimension_mismatch() {
        let s = vec![1.0_f32; 8];
        let t = vec![1.0_f32; 12];
        let r = kd_attention_transfer_loss(&s, &t, 2, 2, 2);
        assert!(matches!(
            r,
            Err(DistillationError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_attention_transfer_zero_dims() {
        let r = kd_attention_transfer_loss(&[], &[], 0, 0, 0);
        assert!(matches!(r, Err(DistillationError::EmptyInput)));
    }

    // ── kd_pairwise_distances ─────────────────────────────────────────────────

    #[test]
    fn test_pairwise_distances_two_embeddings() {
        let emb = vec![0.0_f32, 0.0, 3.0, 4.0]; // 2 embeddings of dim 2
        let dists = kd_pairwise_distances(&emb, 2).unwrap();
        assert_eq!(dists.len(), 1); // 2*(2-1)/2 = 1
        assert!(approx(dists[0], 5.0, 1e-5)); // sqrt(9+16)=5
    }

    #[test]
    fn test_pairwise_distances_three_embeddings() {
        let emb = vec![1.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0]; // 3 × 2
        let dists = kd_pairwise_distances(&emb, 2).unwrap();
        assert_eq!(dists.len(), 3); // 3*(3-1)/2 = 3
        for &d in &dists {
            assert!(d >= 0.0);
        }
    }

    #[test]
    fn test_pairwise_distances_single_embedding() {
        // Single sample → no pairs → empty
        let emb = vec![1.0_f32, 2.0];
        let dists = kd_pairwise_distances(&emb, 2).unwrap();
        assert_eq!(dists.len(), 0);
    }

    #[test]
    fn test_pairwise_distances_all_non_negative() {
        let emb: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let dists = kd_pairwise_distances(&emb, 3).unwrap();
        for &d in &dists {
            assert!(d >= 0.0);
        }
    }

    // ── kd_cosine_similarity ──────────────────────────────────────────────────

    #[test]
    fn test_cosine_identical_is_one() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let sim = kd_cosine_similarity(&v, &v).unwrap();
        assert!(approx(sim, 1.0, 1e-6));
    }

    #[test]
    fn test_cosine_opposite_is_neg_one() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![-1.0_f32, 0.0];
        let sim = kd_cosine_similarity(&a, &b).unwrap();
        assert!(approx(sim, -1.0, 1e-6));
    }

    #[test]
    fn test_cosine_orthogonal_is_zero() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let sim = kd_cosine_similarity(&a, &b).unwrap();
        assert!(approx(sim, 0.0, 1e-6));
    }

    #[test]
    fn test_cosine_empty() {
        assert!(matches!(
            kd_cosine_similarity(&[], &[]),
            Err(DistillationError::EmptyInput)
        ));
    }

    // ── kd_relational_loss ────────────────────────────────────────────────────

    #[test]
    fn test_relational_loss_identical_near_zero() {
        let emb = vec![1.0_f32, 0.0, 0.0, 1.0]; // 2 × 2
        let loss = kd_relational_loss(&emb, &emb, 2, 1.0).unwrap();
        assert!(approx(loss, 0.0, 1e-6));
    }

    #[test]
    fn test_relational_loss_different_positive() {
        let s = vec![1.0_f32, 0.0, 0.0, 1.0];
        let t = vec![2.0_f32, 0.0, 0.0, 2.0];
        let loss = kd_relational_loss(&s, &t, 2, 1.0).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_relational_loss_single_sample_returns_zero() {
        let emb = vec![1.0_f32, 2.0];
        let loss = kd_relational_loss(&emb, &emb, 2, 1.0).unwrap();
        assert!(approx(loss, 0.0, 1e-7));
    }

    #[test]
    fn test_relational_loss_empty() {
        assert!(matches!(
            kd_relational_loss(&[], &[], 2, 1.0),
            Err(DistillationError::EmptyInput)
        ));
    }

    #[test]
    fn test_relational_loss_zero_embed_dim() {
        assert!(matches!(
            kd_relational_loss(&[1.0], &[1.0], 0, 1.0),
            Err(DistillationError::DimensionMismatch { .. })
        ));
    }

    // ── kd_total_loss ─────────────────────────────────────────────────────────

    #[test]
    fn test_total_loss_all_components() {
        let s_logits = vec![1.0_f32, 0.5, 0.2];
        let t_logits = vec![0.9_f32, 0.6, 0.3];
        let target = vec![0.0_f32, 1.0, 0.0];
        let s_feats = vec![vec![1.0_f32, 2.0]];
        let t_feats = vec![vec![1.1_f32, 1.9]];
        let cfg = DistillationConfig::default();
        let loss = kd_total_loss(&s_logits, &t_logits, &target, &s_feats, &t_feats, &cfg).unwrap();
        assert!(loss.total >= 0.0);
        assert!(loss.hard >= 0.0);
        assert!(loss.soft >= 0.0);
        assert!(loss.feature >= 0.0);
    }

    #[test]
    fn test_total_loss_no_features() {
        let s_logits = vec![1.0_f32, 0.5];
        let t_logits = vec![0.9_f32, 0.6];
        let target = vec![1.0_f32, 0.0];
        let loss = kd_total_loss(
            &s_logits,
            &t_logits,
            &target,
            &[],
            &[],
            &DistillationConfig::default(),
        )
        .unwrap();
        assert!(approx(loss.feature, 0.0, 1e-7));
    }

    #[test]
    fn test_total_loss_total_matches_formula() {
        let s = vec![1.0_f32, 0.5];
        let t = vec![0.8_f32, 0.6];
        let target = vec![1.0_f32, 0.0];
        let cfg = DistillationConfig {
            alpha: 0.5,
            temperature: 2.0,
            feature_weight: 0.01,
            attention_weight: 0.001,
            relational_weight: 0.0001,
            beta: 1.0,
        };
        let loss = kd_total_loss(&s, &t, &target, &[], &[], &cfg).unwrap();
        let hard_ref = kd_hard_loss(&s, &target).unwrap();
        let soft_ref = kd_soft_loss(&s, &t, cfg.temperature).unwrap();
        let expected = 0.5 * hard_ref + 0.5 * 4.0 * soft_ref; // T²=4
        assert!(approx(loss.total, expected, 1e-5));
    }

    // ── DistillationHistory ───────────────────────────────────────────────────

    #[test]
    fn test_history_record_step_count() {
        let mut h = DistillationHistory::new(DistillationConfig::default());
        for _ in 0..10 {
            h.record(&DistillationLoss {
                total: 1.0,
                hard: 0.5,
                soft: 0.3,
                feature: 0.1,
                attention: 0.0,
                relational: 0.0,
            });
        }
        assert_eq!(h.stats().steps, 10);
    }

    #[test]
    fn test_history_ema_updates() {
        let mut h = DistillationHistory::new(DistillationConfig::default());
        h.record(&DistillationLoss {
            total: 1.0,
            hard: 0.5,
            soft: 0.3,
            feature: 0.1,
            attention: 0.0,
            relational: 0.0,
        });
        let ema_after_1 = h.stats().ema_total_loss;
        assert!(approx(ema_after_1, 1.0, 1e-6));
        h.record(&DistillationLoss {
            total: 0.5,
            hard: 0.3,
            soft: 0.1,
            feature: 0.05,
            attention: 0.0,
            relational: 0.0,
        });
        let ema_after_2 = h.stats().ema_total_loss;
        // EMA should move toward 0.5
        assert!(ema_after_2 < ema_after_1);
    }

    #[test]
    fn test_history_mean_correct() {
        let mut h = DistillationHistory::new(DistillationConfig::default());
        h.record(&DistillationLoss {
            total: 2.0,
            hard: 1.0,
            soft: 0.5,
            feature: 0.2,
            attention: 0.0,
            relational: 0.0,
        });
        h.record(&DistillationLoss {
            total: 4.0,
            hard: 2.0,
            soft: 1.0,
            feature: 0.4,
            attention: 0.0,
            relational: 0.0,
        });
        let stats = h.stats();
        assert!(approx(stats.mean_total_loss, 3.0, 1e-5));
        assert!(approx(stats.mean_hard_loss, 1.5, 1e-5));
    }

    #[test]
    fn test_history_loss_history_length() {
        let mut h = DistillationHistory::new(DistillationConfig::default());
        for i in 0..10 {
            h.record(&DistillationLoss {
                total: i as f32,
                hard: 0.0,
                soft: 0.0,
                feature: 0.0,
                attention: 0.0,
                relational: 0.0,
            });
        }
        assert_eq!(h.loss_history().len(), 10);
    }

    #[test]
    fn test_history_loss_history_capped() {
        let mut h = DistillationHistory::new(DistillationConfig::default());
        // Insert 2001 items → should cap at 2000
        for i in 0..2001 {
            h.record(&DistillationLoss {
                total: i as f32,
                hard: 0.0,
                soft: 0.0,
                feature: 0.0,
                attention: 0.0,
                relational: 0.0,
            });
        }
        assert_eq!(h.loss_history().len(), 2000);
    }

    #[test]
    fn test_history_is_converging_constant_loss() {
        let mut h = DistillationHistory::new(DistillationConfig::default());
        for _ in 0..20 {
            h.record(&DistillationLoss {
                total: 1.0,
                hard: 0.5,
                soft: 0.3,
                feature: 0.1,
                attention: 0.0,
                relational: 0.0,
            });
        }
        assert!(h.is_converging(10));
    }

    #[test]
    fn test_history_is_converging_increasing_loss() {
        let mut h = DistillationHistory::new(DistillationConfig::default());
        for i in 0..20 {
            h.record(&DistillationLoss {
                total: i as f32,
                hard: 0.0,
                soft: 0.0,
                feature: 0.0,
                attention: 0.0,
                relational: 0.0,
            });
        }
        // Increasing loss → not converging
        assert!(!h.is_converging(10));
    }

    #[test]
    fn test_history_is_converging_insufficient_data() {
        let mut h = DistillationHistory::new(DistillationConfig::default());
        h.record(&DistillationLoss {
            total: 1.0,
            hard: 0.5,
            soft: 0.0,
            feature: 0.0,
            attention: 0.0,
            relational: 0.0,
        });
        // Only 1 step recorded, window=10 → false
        assert!(!h.is_converging(10));
    }

    #[test]
    fn test_history_is_converging_decreasing_loss() {
        let mut h = DistillationHistory::new(DistillationConfig::default());
        for i in 0..20 {
            h.record(&DistillationLoss {
                total: 20.0 - i as f32,
                hard: 0.0,
                soft: 0.0,
                feature: 0.0,
                attention: 0.0,
                relational: 0.0,
            });
        }
        // Decreasing → converging
        assert!(h.is_converging(10));
    }

    // ── kd_format_loss / kd_format_stats ─────────────────────────────────────

    #[test]
    fn test_format_loss_non_empty() {
        let loss = DistillationLoss {
            total: 0.5,
            hard: 0.2,
            soft: 0.1,
            feature: 0.05,
            attention: 0.0,
            relational: 0.0,
        };
        let s = kd_format_loss(&loss);
        assert!(!s.is_empty());
        assert!(s.contains("total"));
    }

    #[test]
    fn test_format_stats_non_empty() {
        let stats = DistillationStats {
            mean_total_loss: 1.0,
            mean_hard_loss: 0.5,
            mean_soft_loss: 0.3,
            mean_feature_loss: 0.1,
            steps: 100,
            ema_total_loss: 0.95,
        };
        let s = kd_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("steps"));
    }

    // ── xorshift64 ────────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift_sequence_non_zero() {
        let mut state: u64 = 42;
        for _ in 0..1000 {
            let v = xorshift64(&mut state);
            assert!(v != 0, "xorshift64 produced zero");
        }
    }

    #[test]
    fn test_xorshift_f32_in_unit_interval() {
        let mut state: u64 = 12345;
        for _ in 0..1000 {
            let v = xorshift_f32(&mut state);
            assert!((0.0..=1.0).contains(&v), "v={v}");
        }
    }
}
