//! Online Hard Example Mining (OHEM) for 3DGS avatar training.
//!
//! Identifies and focuses training on the hardest samples in each batch,
//! enabling faster convergence on challenging regions (fine hair, eyes, teeth,
//! profile views). Supports top-K, threshold, soft-weighted, and focal-weighted
//! mining strategies at both sample and pixel level.
//!
//! # Quick start
//! ```rust,no_run
//! use oxigaf_trainer::online_hard_example_mining::{
//!     OnlineMiningConfig, OnlineMiningStrategy, ohem_mine,
//! };
//!
//! let config = OnlineMiningConfig::default();
//! let losses = vec![0.1, 0.8, 0.2, 0.5, 0.9];
//! let result = ohem_mine(&losses, &config, 1000).unwrap();
//! println!("Effective loss: {:.4}", result.effective_loss);
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// PRNG (xorshift64, no rand crate)
// ─────────────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
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

/// Errors produced by the online hard example mining subsystem.
#[derive(Debug, Error, PartialEq)]
pub enum OnlineMiningError {
    #[error("empty batch: need at least 1 sample")]
    EmptyBatch,

    #[error("invalid ratio: must be in (0, 1], got {0}")]
    InvalidRatio(f32),

    #[error("invalid config: {0}")]
    InvalidConfig(String),

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("index out of bounds: {0}")]
    IndexOutOfBounds(usize),
}

// ─────────────────────────────────────────────────────────────────────────────
// Mining strategy
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy controlling how hard examples are selected.
#[derive(Debug, Clone, PartialEq)]
pub enum OnlineMiningStrategy {
    /// Select top-k highest-loss samples.
    TopK,
    /// Select samples with loss above a threshold.
    Threshold(f32),
    /// All samples, weighted by loss via temperature-scaled softmax.
    SoftWeighted,
    /// All samples, weighted by focal factor `(1 - exp(-loss))^gamma`.
    FocalWeighted,
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Online Hard Example Mining.
#[derive(Debug, Clone)]
pub struct OnlineMiningConfig {
    /// Fraction of hardest samples to select. Must be in `(0, 1]`. Default 0.5.
    pub hard_ratio: f32,
    /// Steps before OHEM activates (use all samples). Default 500.
    pub warmup_steps: usize,
    /// EMA decay for per-sample loss tracking. Default 0.9.
    pub ema_decay: f32,
    /// Focal loss gamma; 0 = no focal weighting. Default 2.0.
    pub focal_gamma: f32,
    /// Minimum number of hard samples selected. Default 4.
    pub min_hard_samples: usize,
    /// Temperature for softmax weighting. Default 1.0.
    pub temperature: f32,
    /// Mining strategy. Default `TopK`.
    pub mining_strategy: OnlineMiningStrategy,
}

impl Default for OnlineMiningConfig {
    fn default() -> Self {
        Self {
            hard_ratio: 0.5,
            warmup_steps: 500,
            ema_decay: 0.9,
            focal_gamma: 2.0,
            min_hard_samples: 4,
            temperature: 1.0,
            mining_strategy: OnlineMiningStrategy::TopK,
        }
    }
}

impl OnlineMiningConfig {
    /// Validate configuration, returning an error if any field is out of range.
    pub fn validate(&self) -> Result<(), OnlineMiningError> {
        if self.hard_ratio <= 0.0 || self.hard_ratio > 1.0 {
            return Err(OnlineMiningError::InvalidRatio(self.hard_ratio));
        }
        if self.ema_decay <= 0.0 || self.ema_decay >= 1.0 {
            return Err(OnlineMiningError::InvalidConfig(format!(
                "ema_decay must be in (0,1), got {}",
                self.ema_decay
            )));
        }
        if self.temperature <= 0.0 {
            return Err(OnlineMiningError::InvalidConfig(format!(
                "temperature must be positive, got {}",
                self.temperature
            )));
        }
        if self.focal_gamma < 0.0 {
            return Err(OnlineMiningError::InvalidConfig(format!(
                "focal_gamma must be >= 0, got {}",
                self.focal_gamma
            )));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-sample loss history
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks per-sample loss history using EMA for stable difficulty estimation.
#[derive(Debug, Clone)]
pub struct SampleLossHistory {
    capacity: usize,
    ema_losses: Vec<f32>,
    last_seen_step: Vec<usize>,
    total_updates: usize,
}

impl SampleLossHistory {
    /// Create a new history tracker for `capacity` samples.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            ema_losses: vec![0.0; capacity],
            last_seen_step: vec![0; capacity],
            total_updates: 0,
        }
    }

    /// Update EMA loss for a specific sample.
    ///
    /// On first update (step == 0 AND `ema_losses[idx]` == 0.0), sets EMA = loss
    /// directly to avoid a cold-start bias.
    pub fn update(
        &mut self,
        sample_idx: usize,
        loss: f32,
        step: usize,
        ema_decay: f32,
    ) -> Result<(), OnlineMiningError> {
        if sample_idx >= self.capacity {
            return Err(OnlineMiningError::IndexOutOfBounds(sample_idx));
        }
        let current = self.ema_losses[sample_idx];
        self.ema_losses[sample_idx] = if self.last_seen_step[sample_idx] == 0 && current == 0.0 {
            loss
        } else {
            ema_decay * current + (1.0 - ema_decay) * loss
        };
        self.last_seen_step[sample_idx] = step;
        self.total_updates += 1;
        Ok(())
    }

    /// Get the current difficulty (EMA loss) for a sample.
    pub fn get_difficulty(&self, sample_idx: usize) -> Result<f32, OnlineMiningError> {
        if sample_idx >= self.capacity {
            return Err(OnlineMiningError::IndexOutOfBounds(sample_idx));
        }
        Ok(self.ema_losses[sample_idx])
    }

    /// Return the `n` most difficult samples as `(idx, difficulty)` sorted descending.
    pub fn most_difficult(&self, n: usize) -> Vec<(usize, f32)> {
        let capped = n.min(self.capacity);
        let mut pairs: Vec<(usize, f32)> = self
            .ema_losses
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(capped);
        pairs
    }

    /// Return the `n` least difficult samples as `(idx, difficulty)` sorted ascending.
    pub fn least_difficult(&self, n: usize) -> Vec<(usize, f32)> {
        let capped = n.min(self.capacity);
        let mut pairs: Vec<(usize, f32)> = self
            .ema_losses
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(capped);
        pairs
    }

    /// Number of tracked samples.
    pub fn len(&self) -> usize {
        self.capacity
    }

    /// Returns `true` if capacity is zero.
    pub fn is_empty(&self) -> bool {
        self.capacity == 0
    }

    /// Total number of EMA updates performed.
    pub fn total_updates(&self) -> usize {
        self.total_updates
    }

    /// Reset a single sample's EMA loss and last-seen step to zero.
    pub fn reset_sample(&mut self, sample_idx: usize) -> Result<(), OnlineMiningError> {
        if sample_idx >= self.capacity {
            return Err(OnlineMiningError::IndexOutOfBounds(sample_idx));
        }
        self.ema_losses[sample_idx] = 0.0;
        self.last_seen_step[sample_idx] = 0;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core OHEM functions
// ─────────────────────────────────────────────────────────────────────────────

/// Select hard examples by top-K loss.
///
/// Returns indices of the `k` highest-loss samples, sorted descending by loss.
/// Returns [`OnlineMiningError::EmptyBatch`] if `losses` is empty or `k == 0`.
pub fn ohem_select_top_k(losses: &[f32], k: usize) -> Result<Vec<usize>, OnlineMiningError> {
    if losses.is_empty() || k == 0 {
        return Err(OnlineMiningError::EmptyBatch);
    }
    let capped_k = k.min(losses.len());
    let mut indexed: Vec<(usize, f32)> = losses.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(indexed.into_iter().take(capped_k).map(|(i, _)| i).collect())
}

/// Select hard examples by threshold.
///
/// Returns indices where `loss > threshold`.  If none exceed the threshold,
/// returns all indices (fallback to full batch).
pub fn ohem_select_by_threshold(
    losses: &[f32],
    threshold: f32,
) -> Result<Vec<usize>, OnlineMiningError> {
    if losses.is_empty() {
        return Err(OnlineMiningError::EmptyBatch);
    }
    let above: Vec<usize> = losses
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > threshold)
        .map(|(i, _)| i)
        .collect();
    if above.is_empty() {
        // Fallback: return all indices.
        Ok((0..losses.len()).collect())
    } else {
        Ok(above)
    }
}

/// Numerically-stable softmax over a slice of values.
fn stable_softmax(values: &[f32]) -> Vec<f32> {
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        vec![1.0 / values.len() as f32; values.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

/// Compute softmax-based sample weights from losses.
///
/// Higher loss → higher weight. Temperature-scaled: `w[i] = softmax(loss[i] / T)`.
pub fn ohem_soft_weights(losses: &[f32], temperature: f32) -> Result<Vec<f32>, OnlineMiningError> {
    if losses.is_empty() {
        return Err(OnlineMiningError::EmptyBatch);
    }
    if temperature <= 0.0 {
        return Err(OnlineMiningError::InvalidConfig(format!(
            "temperature must be positive, got {temperature}"
        )));
    }
    let scaled: Vec<f32> = losses.iter().map(|&v| v / temperature).collect();
    Ok(stable_softmax(&scaled))
}

/// Compute focal-loss-style weights: `w[i] = (1 - exp(-loss_i))^gamma`.
///
/// Higher loss → lower `p_i = exp(-loss_i)` → higher weight.
/// When `gamma = 0`, all weights are 1.0.
pub fn ohem_focal_weights(losses: &[f32], gamma: f32) -> Result<Vec<f32>, OnlineMiningError> {
    if losses.is_empty() {
        return Err(OnlineMiningError::EmptyBatch);
    }
    let weights: Vec<f32> = losses
        .iter()
        .map(|&l| {
            let p = (-l.max(0.0)).exp();
            let base = (1.0 - p).max(0.0);
            if gamma == 0.0 {
                1.0f32
            } else {
                base.powf(gamma)
            }
        })
        .collect();
    Ok(weights)
}

/// Weighted loss: `sum(weights * losses) / sum(weights)`.
pub fn ohem_weighted_loss(losses: &[f32], weights: &[f32]) -> Result<f32, OnlineMiningError> {
    if losses.is_empty() {
        return Err(OnlineMiningError::EmptyBatch);
    }
    if weights.len() != losses.len() {
        return Err(OnlineMiningError::DimensionMismatch {
            expected: losses.len(),
            got: weights.len(),
        });
    }
    let weighted_sum: f32 = losses
        .iter()
        .zip(weights.iter())
        .map(|(&l, &w)| l * w)
        .sum();
    let weight_sum: f32 = weights.iter().sum();
    if weight_sum <= 0.0 {
        // Fallback to simple mean when all weights are zero.
        let mean = losses.iter().sum::<f32>() / losses.len() as f32;
        return Ok(mean);
    }
    Ok(weighted_sum / weight_sum)
}

/// Hard example loss: mean loss over selected hard sample indices.
pub fn ohem_hard_loss(losses: &[f32], hard_indices: &[usize]) -> Result<f32, OnlineMiningError> {
    if hard_indices.is_empty() {
        return Err(OnlineMiningError::EmptyBatch);
    }
    let mut sum = 0.0f32;
    for &idx in hard_indices {
        if idx >= losses.len() {
            return Err(OnlineMiningError::IndexOutOfBounds(idx));
        }
        sum += losses[idx];
    }
    Ok(sum / hard_indices.len() as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// OhemResult
// ─────────────────────────────────────────────────────────────────────────────

/// Result from an OHEM mining pass.
#[derive(Debug, Clone)]
pub struct OhemResult {
    /// The effective loss value to back-propagate.
    pub effective_loss: f32,
    /// Indices selected as hard examples (empty for `SoftWeighted` / `FocalWeighted`).
    pub selected_indices: Vec<usize>,
    /// Per-sample weights (uniform 1/N for TopK/Threshold; softmax/focal for others).
    pub weights: Vec<f32>,
    /// Fraction of samples selected as hard.
    pub hard_fraction: f32,
    /// Mean loss of hard examples.
    pub mean_hard_loss: f32,
    /// Mean loss of non-hard examples (equal to mean_hard_loss when all are selected).
    pub mean_easy_loss: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Main mining function
// ─────────────────────────────────────────────────────────────────────────────

/// Mine hard examples from a batch of per-sample losses.
///
/// Returns an [`OhemResult`] with the effective loss and selection metadata.
/// During warmup (`step < config.warmup_steps`) all samples are used equally.
pub fn ohem_mine(
    losses: &[f32],
    config: &OnlineMiningConfig,
    step: usize,
) -> Result<OhemResult, OnlineMiningError> {
    if losses.is_empty() {
        return Err(OnlineMiningError::EmptyBatch);
    }
    if config.hard_ratio <= 0.0 || config.hard_ratio > 1.0 {
        return Err(OnlineMiningError::InvalidRatio(config.hard_ratio));
    }

    let n = losses.len();
    let mean_all = losses.iter().sum::<f32>() / n as f32;

    // During warmup: treat all samples as equally important.
    if step < config.warmup_steps {
        let uniform = 1.0 / n as f32;
        return Ok(OhemResult {
            effective_loss: mean_all,
            selected_indices: Vec::new(),
            weights: vec![uniform; n],
            hard_fraction: 1.0,
            mean_hard_loss: mean_all,
            mean_easy_loss: mean_all,
        });
    }

    match &config.mining_strategy {
        OnlineMiningStrategy::TopK => {
            let k_raw = (n as f32 * config.hard_ratio).ceil() as usize;
            let k = k_raw.max(config.min_hard_samples).min(n);

            let hard_indices = ohem_select_top_k(losses, k)?;
            let hard_loss = ohem_hard_loss(losses, &hard_indices)?;

            // Easy = all samples NOT in hard set.
            let hard_set: std::collections::HashSet<usize> = hard_indices.iter().copied().collect();
            let easy_losses: Vec<f32> = losses
                .iter()
                .enumerate()
                .filter(|(i, _)| !hard_set.contains(i))
                .map(|(_, &v)| v)
                .collect();
            let easy_loss = if easy_losses.is_empty() {
                hard_loss
            } else {
                easy_losses.iter().sum::<f32>() / easy_losses.len() as f32
            };

            let uniform_hard = 1.0 / k as f32;
            let mut weights = vec![0.0f32; n];
            for &idx in &hard_indices {
                weights[idx] = uniform_hard;
            }

            Ok(OhemResult {
                effective_loss: hard_loss,
                selected_indices: hard_indices,
                weights,
                hard_fraction: k as f32 / n as f32,
                mean_hard_loss: hard_loss,
                mean_easy_loss: easy_loss,
            })
        }

        OnlineMiningStrategy::Threshold(threshold) => {
            let hard_indices = ohem_select_by_threshold(losses, *threshold)?;
            let hard_loss = ohem_hard_loss(losses, &hard_indices)?;

            let hard_set: std::collections::HashSet<usize> = hard_indices.iter().copied().collect();
            let easy_losses: Vec<f32> = losses
                .iter()
                .enumerate()
                .filter(|(i, _)| !hard_set.contains(i))
                .map(|(_, &v)| v)
                .collect();
            let easy_loss = if easy_losses.is_empty() {
                hard_loss
            } else {
                easy_losses.iter().sum::<f32>() / easy_losses.len() as f32
            };

            let k = hard_indices.len();
            let uniform_hard = if k > 0 {
                1.0 / k as f32
            } else {
                1.0 / n as f32
            };
            let mut weights = vec![0.0f32; n];
            for &idx in &hard_indices {
                weights[idx] = uniform_hard;
            }

            Ok(OhemResult {
                effective_loss: hard_loss,
                selected_indices: hard_indices,
                weights,
                hard_fraction: k as f32 / n as f32,
                mean_hard_loss: hard_loss,
                mean_easy_loss: easy_loss,
            })
        }

        OnlineMiningStrategy::SoftWeighted => {
            let weights = ohem_soft_weights(losses, config.temperature)?;
            let effective = ohem_weighted_loss(losses, &weights)?;

            Ok(OhemResult {
                effective_loss: effective,
                selected_indices: Vec::new(),
                weights,
                hard_fraction: 1.0,
                mean_hard_loss: effective,
                mean_easy_loss: effective,
            })
        }

        OnlineMiningStrategy::FocalWeighted => {
            let weights = ohem_focal_weights(losses, config.focal_gamma)?;
            let effective = ohem_weighted_loss(losses, &weights)?;

            Ok(OhemResult {
                effective_loss: effective,
                selected_indices: Vec::new(),
                weights,
                hard_fraction: 1.0,
                mean_hard_loss: effective,
                mean_easy_loss: effective,
            })
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pixel-level OHEM
// ─────────────────────────────────────────────────────────────────────────────

/// Pixel-level OHEM for image reconstruction losses.
///
/// Selects the `hard_ratio` fraction of pixels with highest per-pixel loss.
/// Returns a masked loss image where non-selected pixels are zeroed.
pub fn ohem_pixel_mining(
    pixel_losses: &[f32],
    width: usize,
    height: usize,
    hard_ratio: f32,
) -> Result<Vec<f32>, OnlineMiningError> {
    if pixel_losses.is_empty() {
        return Err(OnlineMiningError::EmptyBatch);
    }
    if hard_ratio <= 0.0 || hard_ratio > 1.0 {
        return Err(OnlineMiningError::InvalidRatio(hard_ratio));
    }
    let expected = width * height;
    if pixel_losses.len() != expected {
        return Err(OnlineMiningError::DimensionMismatch {
            expected,
            got: pixel_losses.len(),
        });
    }

    let n = pixel_losses.len();
    let k = ((n as f32 * hard_ratio).ceil() as usize).max(1).min(n);

    // Sort indices by loss descending; keep top-k.
    let mut indexed: Vec<(usize, f32)> = pixel_losses
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);

    let mut masked = vec![0.0f32; n];
    for (i, v) in indexed {
        masked[i] = v;
    }
    Ok(masked)
}

/// Per-pixel MSE loss between prediction and target (flat RGB or single-channel images).
///
/// `pred` and `target` must have the same length.
pub fn ohem_pixel_mse(pred: &[f32], target: &[f32]) -> Result<Vec<f32>, OnlineMiningError> {
    if pred.is_empty() {
        return Err(OnlineMiningError::EmptyBatch);
    }
    if pred.len() != target.len() {
        return Err(OnlineMiningError::DimensionMismatch {
            expected: pred.len(),
            got: target.len(),
        });
    }
    Ok(pred
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| {
            let d = p - t;
            d * d
        })
        .collect())
}

/// Per-pixel L1 loss between prediction and target.
///
/// `pred` and `target` must have the same length.
pub fn ohem_pixel_l1(pred: &[f32], target: &[f32]) -> Result<Vec<f32>, OnlineMiningError> {
    if pred.is_empty() {
        return Err(OnlineMiningError::EmptyBatch);
    }
    if pred.len() != target.len() {
        return Err(OnlineMiningError::DimensionMismatch {
            expected: pred.len(),
            got: target.len(),
        });
    }
    Ok(pred
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| (p - t).abs())
        .collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics and tracking
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics snapshot from one OHEM step.
#[derive(Debug, Clone)]
pub struct OnlineMiningStats {
    /// Training step index.
    pub step: usize,
    /// Whether OHEM is active (past warmup).
    pub is_active: bool,
    /// Mean loss across the entire batch.
    pub mean_batch_loss: f32,
    /// Mean loss of selected hard examples.
    pub mean_hard_loss: f32,
    /// Mean loss of non-hard examples.
    pub mean_easy_loss: f32,
    /// Fraction of samples selected as hard.
    pub hard_fraction: f32,
    /// EMA of effective loss (decay = 0.99).
    pub ema_loss: f32,
}

/// Tracks OHEM statistics over training.
pub struct OnlineMiningTracker {
    config: OnlineMiningConfig,
    /// Capped at 1000 entries.
    history: Vec<OnlineMiningStats>,
    ema_loss: f32,
    step: usize,
    sample_history: Option<SampleLossHistory>,
}

impl OnlineMiningTracker {
    /// Create a tracker with the given configuration.
    pub fn new(config: OnlineMiningConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            ema_loss: 0.0,
            step: 0,
            sample_history: None,
        }
    }

    /// Create a tracker that also maintains per-sample EMA loss history.
    pub fn with_sample_history(config: OnlineMiningConfig, n_samples: usize) -> Self {
        Self {
            config,
            history: Vec::new(),
            ema_loss: 0.0,
            step: 0,
            sample_history: Some(SampleLossHistory::new(n_samples)),
        }
    }

    /// Update EMA loss for a specific sample (requires `with_sample_history`).
    pub fn update_sample(&mut self, idx: usize, loss: f32) -> Result<(), OnlineMiningError> {
        match &mut self.sample_history {
            Some(hist) => {
                let decay = self.config.ema_decay;
                let step = self.step;
                hist.update(idx, loss, step, decay)
            }
            None => Err(OnlineMiningError::InvalidConfig(
                "sample history not enabled; use with_sample_history".to_owned(),
            )),
        }
    }

    /// Run OHEM on a batch, record statistics, and return the result.
    pub fn mine_batch(&mut self, losses: &[f32]) -> Result<OhemResult, OnlineMiningError> {
        if losses.is_empty() {
            return Err(OnlineMiningError::EmptyBatch);
        }

        let result = ohem_mine(losses, &self.config, self.step)?;

        let mean_batch = losses.iter().sum::<f32>() / losses.len() as f32;
        let is_active = self.step >= self.config.warmup_steps;

        // Update EMA of effective loss (decay = 0.99).
        const EMA_DECAY: f32 = 0.99;
        if self.step == 0 && self.ema_loss == 0.0 {
            self.ema_loss = result.effective_loss;
        } else {
            self.ema_loss = EMA_DECAY * self.ema_loss + (1.0 - EMA_DECAY) * result.effective_loss;
        }

        let stats = OnlineMiningStats {
            step: self.step,
            is_active,
            mean_batch_loss: mean_batch,
            mean_hard_loss: result.mean_hard_loss,
            mean_easy_loss: result.mean_easy_loss,
            hard_fraction: result.hard_fraction,
            ema_loss: self.ema_loss,
        };

        if self.history.len() < 1000 {
            self.history.push(stats);
        } else {
            // Circular overwrite: drop oldest.
            let idx = self.step % 1000;
            self.history[idx] = stats;
        }

        Ok(result)
    }

    /// Advance the training step counter by one.
    pub fn advance_step(&mut self) {
        self.step += 1;
    }

    /// Current training step.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Most recent statistics snapshot, or `None` if no batch has been mined.
    pub fn stats(&self) -> Option<&OnlineMiningStats> {
        self.history.last()
    }

    /// Full statistics history (up to 1000 entries).
    pub fn history(&self) -> &[OnlineMiningStats] {
        &self.history
    }

    /// Whether OHEM is active (past warmup period).
    pub fn is_active(&self) -> bool {
        self.step >= self.config.warmup_steps
    }

    /// Reference to the current configuration.
    pub fn config(&self) -> &OnlineMiningConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Format an [`OhemResult`] as a human-readable string.
pub fn ohem_format_result(result: &OhemResult) -> String {
    format!(
        "OhemResult {{ loss={:.4}, hard_frac={:.2}, n_hard={}, mean_hard={:.4}, mean_easy={:.4} }}",
        result.effective_loss,
        result.hard_fraction,
        result.selected_indices.len(),
        result.mean_hard_loss,
        result.mean_easy_loss,
    )
}

/// Format an [`OnlineMiningStats`] as a human-readable string.
pub fn ohem_format_stats(stats: &OnlineMiningStats) -> String {
    format!(
        "OnlineMiningStats {{ step={}, active={}, batch_loss={:.4}, hard_loss={:.4}, easy_loss={:.4}, hard_frac={:.2}, ema={:.4} }}",
        stats.step,
        stats.is_active,
        stats.mean_batch_loss,
        stats.mean_hard_loss,
        stats.mean_easy_loss,
        stats.hard_fraction,
        stats.ema_loss,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn default_config() -> OnlineMiningConfig {
        OnlineMiningConfig::default()
    }

    fn active_config() -> OnlineMiningConfig {
        OnlineMiningConfig {
            warmup_steps: 0,
            ..OnlineMiningConfig::default()
        }
    }

    // ── 1. OnlineMiningConfig::default field values ───────────────────────────
    #[test]
    fn test_config_default_fields() {
        let c = default_config();
        assert!((c.hard_ratio - 0.5).abs() < 1e-6);
        assert_eq!(c.warmup_steps, 500);
        assert!((c.ema_decay - 0.9).abs() < 1e-6);
        assert!((c.focal_gamma - 2.0).abs() < 1e-6);
        assert_eq!(c.min_hard_samples, 4);
        assert!((c.temperature - 1.0).abs() < 1e-6);
        assert_eq!(c.mining_strategy, OnlineMiningStrategy::TopK);
    }

    // ── 2. ohem_select_top_k: k=1 returns index of max-loss sample ────────────
    #[test]
    fn test_top_k_k1_returns_max() {
        let losses = vec![0.1, 0.9, 0.3, 0.5, 0.2];
        let result = ohem_select_top_k(&losses, 1).unwrap();
        assert_eq!(result, vec![1]);
    }

    // ── 3. ohem_select_top_k: k=all returns all sorted descending ─────────────
    #[test]
    fn test_top_k_all_sorted_descending() {
        let losses = vec![0.3, 0.9, 0.1, 0.7, 0.5];
        let result = ohem_select_top_k(&losses, 5).unwrap();
        assert_eq!(result.len(), 5);
        // Verify descending order by loss.
        for w in result.windows(2) {
            assert!(losses[w[0]] >= losses[w[1]]);
        }
    }

    // ── 4. ohem_select_top_k: k=0 returns EmptyBatch ──────────────────────────
    #[test]
    fn test_top_k_zero_k_returns_empty_batch() {
        let losses = vec![0.5, 0.3];
        let err = ohem_select_top_k(&losses, 0).unwrap_err();
        assert_eq!(err, OnlineMiningError::EmptyBatch);
    }

    // ── 5. ohem_select_top_k: empty losses returns EmptyBatch ─────────────────
    #[test]
    fn test_top_k_empty_losses_returns_empty_batch() {
        let err = ohem_select_top_k(&[], 2).unwrap_err();
        assert_eq!(err, OnlineMiningError::EmptyBatch);
    }

    // ── 6. ohem_select_top_k: k larger than batch clips to batch size ──────────
    #[test]
    fn test_top_k_clips_to_batch_size() {
        let losses = vec![0.1, 0.2, 0.3];
        let result = ohem_select_top_k(&losses, 10).unwrap();
        assert_eq!(result.len(), 3);
    }

    // ── 7. ohem_select_by_threshold: all above → all returned ─────────────────
    #[test]
    fn test_threshold_all_above() {
        let losses = vec![0.8, 0.9, 0.95];
        let result = ohem_select_by_threshold(&losses, 0.5).unwrap();
        assert_eq!(result.len(), 3);
    }

    // ── 8. ohem_select_by_threshold: all below → fallback returns all ──────────
    #[test]
    fn test_threshold_all_below_fallback() {
        let losses = vec![0.1, 0.2, 0.3];
        let result = ohem_select_by_threshold(&losses, 0.9).unwrap();
        assert_eq!(result.len(), 3);
    }

    // ── 9. ohem_select_by_threshold: some above threshold ─────────────────────
    #[test]
    fn test_threshold_some_above() {
        let losses = vec![0.1, 0.8, 0.3, 0.9, 0.05];
        let result = ohem_select_by_threshold(&losses, 0.5).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.contains(&1));
        assert!(result.contains(&3));
    }

    // ── 10. ohem_soft_weights: sum ≈ 1.0 ──────────────────────────────────────
    #[test]
    fn test_soft_weights_sum_to_one() {
        let losses = vec![0.1, 0.5, 0.9, 0.3];
        let w = ohem_soft_weights(&losses, 1.0).unwrap();
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
    }

    // ── 11. ohem_soft_weights: higher loss → higher weight ────────────────────
    #[test]
    fn test_soft_weights_higher_loss_higher_weight() {
        let losses = vec![0.1, 0.9];
        let w = ohem_soft_weights(&losses, 1.0).unwrap();
        assert!(w[1] > w[0], "w[1]={} should be > w[0]={}", w[1], w[0]);
    }

    // ── 12. ohem_soft_weights: higher T → more uniform ────────────────────────
    #[test]
    fn test_soft_weights_temperature_uniformity() {
        let losses = vec![0.1, 0.5, 0.9, 0.3];
        let w_low = ohem_soft_weights(&losses, 0.1).unwrap();
        let w_high = ohem_soft_weights(&losses, 10.0).unwrap();
        let var_low: f32 = {
            let mean = w_low.iter().sum::<f32>() / w_low.len() as f32;
            w_low.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / w_low.len() as f32
        };
        let var_high: f32 = {
            let mean = w_high.iter().sum::<f32>() / w_high.len() as f32;
            w_high.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / w_high.len() as f32
        };
        assert!(var_high < var_low, "high T should be more uniform");
    }

    // ── 13. ohem_soft_weights: empty → EmptyBatch ─────────────────────────────
    #[test]
    fn test_soft_weights_empty_error() {
        let err = ohem_soft_weights(&[], 1.0).unwrap_err();
        assert_eq!(err, OnlineMiningError::EmptyBatch);
    }

    // ── 14. ohem_focal_weights: loss=0 → weight=0 (gamma>0) ──────────────────
    #[test]
    fn test_focal_weights_zero_loss_zero_weight() {
        let losses = vec![0.0, 1.0];
        let w = ohem_focal_weights(&losses, 2.0).unwrap();
        assert!(w[0].abs() < 1e-6, "w[0]={} should be ~0", w[0]);
    }

    // ── 15. ohem_focal_weights: large loss → weight approaches 1 ──────────────
    #[test]
    fn test_focal_weights_large_loss_near_one() {
        let losses = vec![20.0];
        let w = ohem_focal_weights(&losses, 2.0).unwrap();
        assert!(w[0] > 0.99, "w[0]={} should be close to 1", w[0]);
    }

    // ── 16. ohem_focal_weights: gamma=0 → all weights ≈ 1.0 ──────────────────
    #[test]
    fn test_focal_weights_gamma_zero_uniform() {
        let losses = vec![0.1, 0.5, 0.9, 0.0];
        let w = ohem_focal_weights(&losses, 0.0).unwrap();
        for (i, &v) in w.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-6, "w[{i}]={v} should be 1.0");
        }
    }

    // ── 17. ohem_weighted_loss: uniform weights → arithmetic mean ─────────────
    #[test]
    fn test_weighted_loss_uniform_is_mean() {
        let losses = vec![0.2, 0.4, 0.6, 0.8];
        let weights = vec![0.25; 4];
        let wl = ohem_weighted_loss(&losses, &weights).unwrap();
        let mean = losses.iter().sum::<f32>() / 4.0;
        assert!((wl - mean).abs() < 1e-5, "wl={wl} mean={mean}");
    }

    // ── 18. ohem_weighted_loss: mismatched lengths → error ────────────────────
    #[test]
    fn test_weighted_loss_mismatch_error() {
        let losses = vec![0.1, 0.2, 0.3];
        let weights = vec![0.5, 0.5];
        let err = ohem_weighted_loss(&losses, &weights).unwrap_err();
        assert!(matches!(err, OnlineMiningError::DimensionMismatch { .. }));
    }

    // ── 19. ohem_hard_loss: mean of selected indices only ─────────────────────
    #[test]
    fn test_hard_loss_selected_only() {
        let losses = vec![0.1, 0.8, 0.3, 0.9, 0.05];
        let hl = ohem_hard_loss(&losses, &[1, 3]).unwrap();
        assert!((hl - 0.85).abs() < 1e-5, "hl={hl}");
    }

    // ── 20. ohem_hard_loss: empty indices → EmptyBatch ────────────────────────
    #[test]
    fn test_hard_loss_empty_indices_error() {
        let losses = vec![0.1, 0.2];
        let err = ohem_hard_loss(&losses, &[]).unwrap_err();
        assert_eq!(err, OnlineMiningError::EmptyBatch);
    }

    // ── 21. ohem_mine: step < warmup → returns mean of all ────────────────────
    #[test]
    fn test_mine_warmup_returns_mean() {
        let losses = vec![0.1, 0.9, 0.5];
        let config = default_config(); // warmup_steps = 500
        let result = ohem_mine(&losses, &config, 0).unwrap();
        let expected_mean = losses.iter().sum::<f32>() / 3.0;
        assert!((result.effective_loss - expected_mean).abs() < 1e-5);
        assert_eq!(result.hard_fraction, 1.0);
    }

    // ── 22. ohem_mine: TopK selects hard_ratio fraction ───────────────────────
    #[test]
    fn test_mine_topk_selects_correct_fraction() {
        let losses = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let config = OnlineMiningConfig {
            warmup_steps: 0,
            hard_ratio: 0.3,
            min_hard_samples: 1,
            ..OnlineMiningConfig::default()
        };
        let result = ohem_mine(&losses, &config, 0).unwrap();
        // 30% of 10 = 3
        assert_eq!(result.selected_indices.len(), 3);
    }

    // ── 23. ohem_mine: TopK hard_fraction ≈ hard_ratio ────────────────────────
    #[test]
    fn test_mine_topk_hard_fraction() {
        let losses: Vec<f32> = (0..10).map(|i| i as f32 * 0.1 + 0.05).collect();
        let config = active_config();
        let result = ohem_mine(&losses, &config, 0).unwrap();
        assert!(
            (result.hard_fraction - 0.5).abs() < 0.15,
            "frac={}",
            result.hard_fraction
        );
    }

    // ── 24. ohem_mine: SoftWeighted → weights sum ≈ 1.0 ──────────────────────
    #[test]
    fn test_mine_softweighted_weights_sum() {
        let losses = vec![0.1, 0.5, 0.3, 0.8];
        let config = OnlineMiningConfig {
            warmup_steps: 0,
            mining_strategy: OnlineMiningStrategy::SoftWeighted,
            ..OnlineMiningConfig::default()
        };
        let result = ohem_mine(&losses, &config, 0).unwrap();
        let sum: f32 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum={sum}");
    }

    // ── 25. ohem_mine: FocalWeighted → all weights positive ───────────────────
    #[test]
    fn test_mine_focalweighted_weights_positive() {
        let losses = vec![0.2, 0.5, 0.8, 1.5];
        let config = OnlineMiningConfig {
            warmup_steps: 0,
            mining_strategy: OnlineMiningStrategy::FocalWeighted,
            focal_gamma: 2.0,
            ..OnlineMiningConfig::default()
        };
        let result = ohem_mine(&losses, &config, 0).unwrap();
        for &w in &result.weights {
            assert!(w >= 0.0, "w={w}");
        }
    }

    // ── 26. ohem_mine: empty losses → EmptyBatch error ────────────────────────
    #[test]
    fn test_mine_empty_losses_error() {
        let config = active_config();
        let err = ohem_mine(&[], &config, 0).unwrap_err();
        assert_eq!(err, OnlineMiningError::EmptyBatch);
    }

    // ── 27. ohem_pixel_mining: hard_ratio=1.0 → no change ────────────────────
    #[test]
    fn test_pixel_mining_full_ratio() {
        let losses = vec![0.1, 0.5, 0.3, 0.9];
        let masked = ohem_pixel_mining(&losses, 2, 2, 1.0).unwrap();
        assert_eq!(masked.len(), losses.len());
        let sum_orig: f32 = losses.iter().sum();
        let sum_mask: f32 = masked.iter().sum();
        assert!((sum_orig - sum_mask).abs() < 1e-5);
    }

    // ── 28. ohem_pixel_mining: hard_ratio=0.0 → InvalidRatio error ───────────
    #[test]
    fn test_pixel_mining_zero_ratio_error() {
        let losses = vec![0.1, 0.5, 0.3, 0.9];
        let err = ohem_pixel_mining(&losses, 2, 2, 0.0).unwrap_err();
        assert_eq!(err, OnlineMiningError::InvalidRatio(0.0));
    }

    // ── 29. ohem_pixel_mining: top pixels correctly identified ────────────────
    #[test]
    fn test_pixel_mining_top_pixels_identified() {
        let losses = vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.0, 0.5, 0.4, 0.6];
        // Select top 30% = 3 pixels: indices with losses 0.9, 0.8, 0.7 → idx 1, 3, 5
        let masked = ohem_pixel_mining(&losses, 10, 1, 0.3).unwrap();
        assert!(masked[1] > 0.0, "idx 1 (0.9) should be selected");
        assert!(masked[3] > 0.0, "idx 3 (0.8) should be selected");
        assert!(masked[5] > 0.0, "idx 5 (0.7) should be selected");
        assert_eq!(masked[0], 0.0, "idx 0 (0.1) should be zeroed");
    }

    // ── 30. ohem_pixel_mse: identical → all zeros ─────────────────────────────
    #[test]
    fn test_pixel_mse_identical_zero() {
        let x = vec![0.2, 0.5, 0.8];
        let r = ohem_pixel_mse(&x, &x).unwrap();
        for &v in &r {
            assert!(v.abs() < 1e-6);
        }
    }

    // ── 31. ohem_pixel_mse: known diff → correct MSE ─────────────────────────
    #[test]
    fn test_pixel_mse_known_values() {
        let pred = vec![0.0, 1.0, 2.0];
        let tgt = vec![1.0, 1.0, 0.0];
        let r = ohem_pixel_mse(&pred, &tgt).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!((r[1] - 0.0).abs() < 1e-6);
        assert!((r[2] - 4.0).abs() < 1e-6);
    }

    // ── 32. ohem_pixel_l1: identical → all zeros ──────────────────────────────
    #[test]
    fn test_pixel_l1_identical_zero() {
        let x = vec![0.3, 0.7, 1.2];
        let r = ohem_pixel_l1(&x, &x).unwrap();
        for &v in &r {
            assert!(v.abs() < 1e-6);
        }
    }

    // ── 33. ohem_pixel_l1: known diff → correct L1 ────────────────────────────
    #[test]
    fn test_pixel_l1_known_values() {
        let pred = vec![0.0, 1.0, -1.0];
        let tgt = vec![1.0, 1.0, 1.0];
        let r = ohem_pixel_l1(&pred, &tgt).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!((r[1] - 0.0).abs() < 1e-6);
        assert!((r[2] - 2.0).abs() < 1e-6);
    }

    // ── 34. SampleLossHistory::update EMA converges ───────────────────────────
    #[test]
    fn test_sample_history_ema_converges() {
        let mut hist = SampleLossHistory::new(1);
        // Feed 1.0 repeatedly; EMA should stay near 1.0.
        for s in 1..=10usize {
            hist.update(0, 1.0, s, 0.9).unwrap();
        }
        let v = hist.get_difficulty(0).unwrap();
        assert!((v - 1.0).abs() < 0.01, "v={v}");

        // Now feed 0.0; EMA should drift toward 0.
        for s in 11..=20usize {
            hist.update(0, 0.0, s, 0.9).unwrap();
        }
        let v2 = hist.get_difficulty(0).unwrap();
        assert!(v2 < v, "EMA should decrease after feeding 0");
    }

    // ── 35. SampleLossHistory::most_difficult: correct ordering ───────────────
    #[test]
    fn test_sample_history_most_difficult_order() {
        let mut hist = SampleLossHistory::new(5);
        let losses = [0.3, 0.9, 0.1, 0.7, 0.5];
        for (i, &l) in losses.iter().enumerate() {
            hist.update(i, l, i + 1, 0.9).unwrap();
        }
        let top3 = hist.most_difficult(3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0].0, 1); // highest loss
        assert_eq!(top3[1].0, 3);
        assert_eq!(top3[2].0, 4);
    }

    // ── 36. SampleLossHistory::least_difficult: ascending order ───────────────
    #[test]
    fn test_sample_history_least_difficult_order() {
        let mut hist = SampleLossHistory::new(4);
        let losses = [0.8, 0.2, 0.6, 0.4];
        for (i, &l) in losses.iter().enumerate() {
            hist.update(i, l, i + 1, 0.9).unwrap();
        }
        let bot2 = hist.least_difficult(2);
        assert_eq!(bot2.len(), 2);
        assert_eq!(bot2[0].0, 1); // lowest loss
    }

    // ── 37. SampleLossHistory::reset_sample resets to zero ────────────────────
    #[test]
    fn test_sample_history_reset_sample() {
        let mut hist = SampleLossHistory::new(3);
        hist.update(0, 0.9, 1, 0.9).unwrap();
        hist.reset_sample(0).unwrap();
        let v = hist.get_difficulty(0).unwrap();
        assert!(v.abs() < 1e-6, "v={v}");
    }

    // ── 38. SampleLossHistory::reset_sample OOB → error ──────────────────────
    #[test]
    fn test_sample_history_reset_oob_error() {
        let mut hist = SampleLossHistory::new(3);
        let err = hist.reset_sample(5).unwrap_err();
        assert!(matches!(err, OnlineMiningError::IndexOutOfBounds(5)));
    }

    // ── 39. SampleLossHistory::total_updates increments ──────────────────────
    #[test]
    fn test_sample_history_total_updates() {
        let mut hist = SampleLossHistory::new(2);
        hist.update(0, 0.5, 1, 0.9).unwrap();
        hist.update(1, 0.3, 2, 0.9).unwrap();
        hist.update(0, 0.4, 3, 0.9).unwrap();
        assert_eq!(hist.total_updates(), 3);
    }

    // ── 40. SampleLossHistory::len and is_empty ───────────────────────────────
    #[test]
    fn test_sample_history_len_is_empty() {
        let hist = SampleLossHistory::new(5);
        assert_eq!(hist.len(), 5);
        assert!(!hist.is_empty());
        let empty = SampleLossHistory::new(0);
        assert!(empty.is_empty());
    }

    // ── 41. OnlineMiningTracker::mine_batch result dimensions ─────────────────
    #[test]
    fn test_tracker_mine_batch_dimensions() {
        let config = active_config();
        let mut tracker = OnlineMiningTracker::new(config);
        let losses = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        let result = tracker.mine_batch(&losses).unwrap();
        assert_eq!(result.weights.len(), losses.len());
    }

    // ── 42. OnlineMiningTracker::advance_step increments ─────────────────────
    #[test]
    fn test_tracker_advance_step() {
        let mut tracker = OnlineMiningTracker::new(default_config());
        assert_eq!(tracker.step(), 0);
        tracker.advance_step();
        assert_eq!(tracker.step(), 1);
        tracker.advance_step();
        tracker.advance_step();
        assert_eq!(tracker.step(), 3);
    }

    // ── 43. OnlineMiningTracker::is_active false before warmup ───────────────
    #[test]
    fn test_tracker_not_active_before_warmup() {
        let config = OnlineMiningConfig {
            warmup_steps: 100,
            ..OnlineMiningConfig::default()
        };
        let tracker = OnlineMiningTracker::new(config);
        assert!(!tracker.is_active());
    }

    // ── 44. OnlineMiningTracker::is_active true after warmup ─────────────────
    #[test]
    fn test_tracker_active_after_warmup() {
        let config = OnlineMiningConfig {
            warmup_steps: 3,
            ..OnlineMiningConfig::default()
        };
        let mut tracker = OnlineMiningTracker::new(config);
        for _ in 0..3 {
            tracker.advance_step();
        }
        assert!(tracker.is_active());
    }

    // ── 45. ohem_format_result non-empty ──────────────────────────────────────
    #[test]
    fn test_format_result_non_empty() {
        let result = OhemResult {
            effective_loss: 0.42,
            selected_indices: vec![0, 2],
            weights: vec![0.5, 0.0, 0.5],
            hard_fraction: 0.67,
            mean_hard_loss: 0.6,
            mean_easy_loss: 0.2,
        };
        let s = ohem_format_result(&result);
        assert!(!s.is_empty());
        assert!(s.contains("0.4200"));
    }

    // ── 46. ohem_format_stats non-empty ───────────────────────────────────────
    #[test]
    fn test_format_stats_non_empty() {
        let stats = OnlineMiningStats {
            step: 100,
            is_active: true,
            mean_batch_loss: 0.3,
            mean_hard_loss: 0.5,
            mean_easy_loss: 0.1,
            hard_fraction: 0.5,
            ema_loss: 0.35,
        };
        let s = ohem_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("step=100"));
    }

    // ── 47. Threshold strategy via ohem_mine ──────────────────────────────────
    #[test]
    fn test_mine_threshold_strategy() {
        let losses = vec![0.1, 0.8, 0.2, 0.9, 0.05, 0.7];
        let config = OnlineMiningConfig {
            warmup_steps: 0,
            mining_strategy: OnlineMiningStrategy::Threshold(0.5),
            ..OnlineMiningConfig::default()
        };
        let result = ohem_mine(&losses, &config, 0).unwrap();
        // Should select indices 1 (0.8), 3 (0.9), 5 (0.7).
        assert_eq!(result.selected_indices.len(), 3);
        assert!(result.selected_indices.contains(&1));
        assert!(result.selected_indices.contains(&3));
        assert!(result.selected_indices.contains(&5));
    }

    // ── 48. OhemResult: hard_fraction in [0,1] ────────────────────────────────
    #[test]
    fn test_result_hard_fraction_in_range() {
        let config = active_config();
        let losses = vec![0.1, 0.5, 0.3, 0.8];
        let result = ohem_mine(&losses, &config, 0).unwrap();
        assert!(result.hard_fraction >= 0.0 && result.hard_fraction <= 1.0);
    }

    // ── 49. OhemResult: mean_hard_loss >= mean_easy_loss (for TopK) ───────────
    #[test]
    fn test_result_topk_hard_loss_gte_easy() {
        let losses: Vec<f32> = (1..=10).map(|i| i as f32 * 0.1).collect();
        let config = OnlineMiningConfig {
            warmup_steps: 0,
            hard_ratio: 0.3,
            min_hard_samples: 1,
            ..OnlineMiningConfig::default()
        };
        let result = ohem_mine(&losses, &config, 0).unwrap();
        assert!(
            result.mean_hard_loss >= result.mean_easy_loss - 1e-5,
            "hard={} easy={}",
            result.mean_hard_loss,
            result.mean_easy_loss
        );
    }

    // ── 50. OnlineMiningTracker: stats() returns None before any mine ─────────
    #[test]
    fn test_tracker_stats_none_before_mine() {
        let tracker = OnlineMiningTracker::new(default_config());
        assert!(tracker.stats().is_none());
    }

    // ── 51. OnlineMiningTracker: stats() returns Some after mine ──────────────
    #[test]
    fn test_tracker_stats_some_after_mine() {
        let mut tracker = OnlineMiningTracker::new(active_config());
        tracker.mine_batch(&[0.1, 0.5, 0.3]).unwrap();
        assert!(tracker.stats().is_some());
    }

    // ── 52. OnlineMiningTracker: history grows ────────────────────────────────
    #[test]
    fn test_tracker_history_grows() {
        let mut tracker = OnlineMiningTracker::new(active_config());
        for i in 0..5usize {
            tracker.mine_batch(&[i as f32 * 0.1; 4]).unwrap();
        }
        assert_eq!(tracker.history().len(), 5);
    }

    // ── 53. OnlineMiningTracker with_sample_history: update_sample works ──────
    #[test]
    fn test_tracker_with_sample_history_update() {
        let mut tracker = OnlineMiningTracker::with_sample_history(active_config(), 5);
        tracker.update_sample(2, 0.7).unwrap();
        // Should not error; basic smoke test.
    }

    // ── 54. OnlineMiningTracker without sample_history: update_sample errors ──
    #[test]
    fn test_tracker_without_sample_history_error() {
        let mut tracker = OnlineMiningTracker::new(active_config());
        let err = tracker.update_sample(0, 0.5).unwrap_err();
        assert!(matches!(err, OnlineMiningError::InvalidConfig(_)));
    }

    // ── 55. ohem_pixel_mining: dimension mismatch → error ─────────────────────
    #[test]
    fn test_pixel_mining_dimension_mismatch() {
        // Providing 5 values for a 3x3 = 9 expected.
        let losses = vec![0.1, 0.5, 0.3, 0.8, 0.2];
        let err = ohem_pixel_mining(&losses, 3, 3, 0.5).unwrap_err();
        assert!(matches!(err, OnlineMiningError::DimensionMismatch { .. }));
    }

    // ── 56. ohem_pixel_mse: empty → EmptyBatch ───────────────────────────────
    #[test]
    fn test_pixel_mse_empty_error() {
        let err = ohem_pixel_mse(&[], &[]).unwrap_err();
        assert_eq!(err, OnlineMiningError::EmptyBatch);
    }

    // ── 57. ohem_pixel_l1: length mismatch → DimensionMismatch ───────────────
    #[test]
    fn test_pixel_l1_mismatch_error() {
        let err = ohem_pixel_l1(&[0.1, 0.2], &[0.1]).unwrap_err();
        assert!(matches!(err, OnlineMiningError::DimensionMismatch { .. }));
    }

    // ── 58. xorshift64 PRNG: state advances and stays non-zero ───────────────
    #[test]
    fn test_xorshift64_non_zero() {
        let mut state = 42u64;
        for _ in 0..100 {
            let v = xorshift64(&mut state);
            assert_ne!(v, 0, "xorshift64 should never produce 0");
        }
    }

    // ── 59. xorshift_f32 produces values in [0,1) ────────────────────────────
    #[test]
    fn test_xorshift_f32_range() {
        let mut state = 12345u64;
        for _ in 0..1000 {
            let v = xorshift_f32(&mut state);
            assert!((0.0..=1.0).contains(&v), "v={v} out of range");
        }
    }

    // ── 60. OnlineMiningConfig::validate rejects invalid ratio ────────────────
    #[test]
    fn test_config_validate_invalid_ratio() {
        let config = OnlineMiningConfig {
            hard_ratio: 0.0,
            ..OnlineMiningConfig::default()
        };
        let err = config.validate().unwrap_err();
        assert!(matches!(err, OnlineMiningError::InvalidRatio(0.0)));
    }

    // ── 61. OnlineMiningConfig::validate rejects bad temperature ──────────────
    #[test]
    fn test_config_validate_bad_temperature() {
        let config = OnlineMiningConfig {
            temperature: 0.0,
            ..OnlineMiningConfig::default()
        };
        let err = config.validate().unwrap_err();
        assert!(matches!(err, OnlineMiningError::InvalidConfig(_)));
    }

    // ── 62. ohem_mine: min_hard_samples enforced ──────────────────────────────
    #[test]
    fn test_mine_min_hard_samples_enforced() {
        let losses = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let config = OnlineMiningConfig {
            warmup_steps: 0,
            hard_ratio: 0.1, // Would give 1, below min.
            min_hard_samples: 3,
            ..OnlineMiningConfig::default()
        };
        let result = ohem_mine(&losses, &config, 0).unwrap();
        assert!(result.selected_indices.len() >= 3);
    }

    // ── 63. ohem_mine: Threshold fallback when no samples above threshold ──────
    #[test]
    fn test_mine_threshold_fallback_all() {
        let losses = vec![0.1, 0.2, 0.3];
        let config = OnlineMiningConfig {
            warmup_steps: 0,
            mining_strategy: OnlineMiningStrategy::Threshold(0.9),
            ..OnlineMiningConfig::default()
        };
        let result = ohem_mine(&losses, &config, 0).unwrap();
        // Fallback: all samples selected.
        assert_eq!(result.selected_indices.len(), 3);
    }

    // ── 64. ohem_weighted_loss: all-zero weights → simple mean ────────────────
    #[test]
    fn test_weighted_loss_zero_weights_fallback() {
        let losses = vec![0.2, 0.6, 1.0];
        let weights = vec![0.0, 0.0, 0.0];
        let wl = ohem_weighted_loss(&losses, &weights).unwrap();
        let expected = (0.2 + 0.6 + 1.0) / 3.0;
        assert!((wl - expected).abs() < 1e-5);
    }

    // ── 65. SampleLossHistory: most_difficult with n > capacity clips ──────────
    #[test]
    fn test_most_difficult_clips_to_capacity() {
        let hist = SampleLossHistory::new(3);
        let top = hist.most_difficult(100);
        assert_eq!(top.len(), 3);
    }
}
