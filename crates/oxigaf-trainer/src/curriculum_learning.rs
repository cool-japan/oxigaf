//! Curriculum learning for 3D Gaussian Splatting avatar training.
//!
//! Implements structured sample ordering strategies that progress from easy to
//! hard examples, improving convergence and final model quality.  "Easy" samples
//! are frontal views at high resolution; "hard" samples are extreme profile
//! views, occluded regions, or fine-detail patches.
//!
//! # Key types
//! - [`CurrLearningError`] — errors produced by this module
//! - [`CurriculumStrategy`] — how samples are selected (EasyFirst, HardFirst, …)
//! - [`PacingFunction`] — how the curriculum window grows over time
//! - [`DifficultyEstimator`] — assigns initial difficulty scores to each sample
//! - [`CurriculumSampler`] — core scheduler that combines all of the above

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Loss value above which a sample is considered maximally difficult.
const MAX_REASONABLE_LOSS: f32 = 10.0;

// ─────────────────────────────────────────────────────────────────────────────
// PRNG (private — xorshift64/xorshift_f32 are already pub in data_augmentation)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    if *state == 0 {
        *state = 1;
    }
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the curriculum-learning subsystem.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum CurrLearningError {
    #[error("empty dataset: no samples to select from")]
    EmptyDataset,

    #[error("invalid pacing: must be in [0, 1], got {0}")]
    InvalidPacing(f32),

    #[error("invalid config: {0}")]
    InvalidConfig(String),

    #[error("index out of bounds: {0}")]
    IndexOutOfBounds(usize),

    #[error("not enough samples: need {needed}, have {available}")]
    NotEnoughSamples { needed: usize, available: usize },
}

// ─────────────────────────────────────────────────────────────────────────────
// CurriculumStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// How samples are ordered and selected during training.
#[derive(Debug, Clone, PartialEq)]
pub enum CurriculumStrategy {
    /// Present samples from easiest to hardest (linear progression).
    EasyFirst,
    /// Present hardest samples first (for fast exploration).
    HardFirst,
    /// Mix easy and hard samples; fraction of hard samples increases over time.
    MixedPace {
        /// Initial fraction of easy samples in a batch (e.g. 0.8 = 80% easy).
        easy_fraction: f32,
    },
    /// Self-paced: select samples where model loss is highest (model decides difficulty).
    SelfPaced {
        /// Select from samples whose current loss is in the top X percentile.
        threshold_percentile: f32,
    },
    /// Competence-based: add samples once model achieves a target PSNR.
    CompetenceBased {
        /// Target PSNR value (dB) above which harder samples are unlocked.
        target_psnr: f32,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// PacingFunction
// ─────────────────────────────────────────────────────────────────────────────

/// Controls how the fraction of the dataset that is "unlocked" grows over time.
#[derive(Debug, Clone, PartialEq)]
pub enum PacingFunction {
    /// Linear increase from 0 to 1 over `n_steps`.
    Linear,
    /// Root-based: `pacing = (step / n_steps)^(1/k)`.
    Root {
        /// Exponent denominator; `k=2` gives square-root growth.
        k: f32,
    },
    /// Exponential: `pacing = 1 - exp(-k * step / n_steps)`.
    Exponential {
        /// Controls how quickly the pacing saturates.
        k: f32,
    },
    /// Step function: jumps from 0 to 1 at the given step.
    Step {
        /// Training step at which pacing becomes 1.0.
        threshold_step: usize,
    },
}

impl PacingFunction {
    /// Compute pacing value in `[0, 1]` at the given step.
    ///
    /// Returns `1.0` when `n_steps == 0` to avoid division-by-zero.
    pub fn pacing_at(&self, step: usize, n_steps: usize) -> f32 {
        if n_steps == 0 {
            return 1.0;
        }
        let t = (step as f32) / (n_steps as f32);
        let raw = match self {
            PacingFunction::Linear => t,
            PacingFunction::Root { k } => {
                if *k <= 0.0 {
                    1.0
                } else {
                    t.powf(1.0 / k)
                }
            }
            PacingFunction::Exponential { k } => 1.0 - (-k * t).exp(),
            PacingFunction::Step { threshold_step } => {
                if step >= *threshold_step {
                    1.0
                } else {
                    0.0
                }
            }
        };
        raw.clamp(0.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SampleDifficulty
// ─────────────────────────────────────────────────────────────────────────────

/// Difficulty tracking state for a single training sample.
#[derive(Debug, Clone)]
pub struct SampleDifficulty {
    /// Index into the original dataset.
    pub sample_idx: usize,
    /// Static difficulty estimate in `[0, 1]`: 0 = easiest, 1 = hardest.
    pub difficulty: f32,
    /// Exponential moving average of training loss for this sample.
    pub loss_ema: f32,
    /// Number of times this sample has been trained on.
    pub n_seen: usize,
    /// Training step at which this sample was last used.
    pub last_step: usize,
}

impl SampleDifficulty {
    /// Create a new [`SampleDifficulty`] with no training history.
    pub fn new(sample_idx: usize, initial_difficulty: f32) -> Self {
        Self {
            sample_idx,
            difficulty: initial_difficulty.clamp(0.0, 1.0),
            loss_ema: 0.0,
            n_seen: 0,
            last_step: 0,
        }
    }

    /// Update the loss EMA and bump counters.
    ///
    /// `ema_decay` should be in `(0, 1)`, e.g. 0.9.
    pub fn update_loss(&mut self, loss: f32, step: usize, ema_decay: f32) {
        if self.n_seen == 0 {
            // First observation: initialise EMA directly.
            self.loss_ema = loss;
        } else {
            self.loss_ema = ema_decay * self.loss_ema + (1.0 - ema_decay) * loss;
        }
        self.n_seen += 1;
        self.last_step = step;
    }

    /// Combined difficulty: maximum of static difficulty and normalised loss EMA.
    pub fn estimated_difficulty(&self) -> f32 {
        let loss_norm = (self.loss_ema / MAX_REASONABLE_LOSS).clamp(0.0, 1.0);
        self.difficulty.max(loss_norm)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DifficultyEstimator
// ─────────────────────────────────────────────────────────────────────────────

/// Assigns initial per-sample difficulty scores to a dataset.
#[derive(Debug, Clone)]
pub struct DifficultyEstimator {
    /// Total number of samples in the dataset.
    pub n_samples: usize,
    /// Per-sample difficulty in `[0, 1]`: 0 = easiest, 1 = hardest.
    pub difficulties: Vec<f32>,
}

impl DifficultyEstimator {
    /// All samples receive equal difficulty 0.5.
    pub fn uniform(n_samples: usize) -> Self {
        Self {
            n_samples,
            difficulties: vec![0.5; n_samples],
        }
    }

    /// Build from pre-computed scores, normalised to `[0, 1]`.
    ///
    /// Returns [`CurrLearningError::EmptyDataset`] for an empty slice.
    pub fn from_scores(scores: &[f32]) -> Result<Self, CurrLearningError> {
        if scores.is_empty() {
            return Err(CurrLearningError::EmptyDataset);
        }
        let difficulties = curr_normalize(scores);
        Ok(Self {
            n_samples: scores.len(),
            difficulties,
        })
    }

    /// Sample `i` has difficulty `i / (n - 1)` (easy → hard).
    ///
    /// For a single sample the difficulty is 0.0.
    pub fn linear_sequence(n_samples: usize) -> Self {
        if n_samples <= 1 {
            return Self {
                n_samples,
                difficulties: vec![0.0; n_samples],
            };
        }
        let denom = (n_samples - 1) as f32;
        let difficulties = (0..n_samples).map(|i| i as f32 / denom).collect();
        Self {
            n_samples,
            difficulties,
        }
    }

    /// Assign difficulty based on camera elevation angle.
    ///
    /// Frontal views (elevation ≈ 0) are easiest; profile views (elevation ≈ π/2)
    /// are hardest.  The absolute angle is used so both tilted-up and tilted-down
    /// cameras are treated symmetrically.
    pub fn from_elevations(elevations: &[f32]) -> Self {
        let n = elevations.len();
        if n == 0 {
            return Self {
                n_samples: 0,
                difficulties: vec![],
            };
        }
        // Map elevation to difficulty: difficulty = |elevation| / (π/2), clamped [0,1].
        let half_pi = std::f32::consts::FRAC_PI_2;
        let raw: Vec<f32> = elevations
            .iter()
            .map(|e| (e.abs() / half_pi).clamp(0.0, 1.0))
            .collect();
        Self {
            n_samples: n,
            difficulties: raw,
        }
    }

    /// Returns the difficulty for sample `idx`.
    pub fn difficulty_of(&self, idx: usize) -> Result<f32, CurrLearningError> {
        self.difficulties
            .get(idx)
            .copied()
            .ok_or(CurrLearningError::IndexOutOfBounds(idx))
    }

    /// Returns all sample indices sorted by difficulty from easy to hard.
    pub fn sorted_by_difficulty(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.n_samples).collect();
        indices.sort_by(|&a, &b| {
            self.difficulties[a]
                .partial_cmp(&self.difficulties[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    /// Returns the sample index at percentile `p` in the sorted order.
    ///
    /// `p = 0.0` → easiest sample; `p = 1.0` → hardest sample.
    pub fn percentile_index(&self, p: f32) -> usize {
        curr_percentile_idx(self.n_samples, p)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CurrLearningConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the [`CurriculumSampler`].
#[derive(Debug, Clone)]
pub struct CurrLearningConfig {
    /// Selection strategy (EasyFirst, HardFirst, etc.).
    pub strategy: CurriculumStrategy,
    /// How the dataset window grows over time.
    pub pacing_function: PacingFunction,
    /// Total number of training steps used for pacing computation.
    pub n_total_steps: usize,
    /// Number of samples to return per batch.
    pub batch_size: usize,
    /// EMA decay coefficient for per-sample loss tracking (default 0.9).
    pub ema_decay: f32,
    /// Steps before the curriculum activates; full dataset is used during warmup.
    pub warmup_steps: usize,
    /// How often (in steps) to re-sort samples by estimated difficulty.
    pub rescore_interval: usize,
}

impl Default for CurrLearningConfig {
    fn default() -> Self {
        Self {
            strategy: CurriculumStrategy::EasyFirst,
            pacing_function: PacingFunction::Linear,
            n_total_steps: 10_000,
            batch_size: 4,
            ema_decay: 0.9,
            warmup_steps: 0,
            rescore_interval: 100,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CurriculumSampler
// ─────────────────────────────────────────────────────────────────────────────

/// Core curriculum scheduler.
///
/// Combines a [`CurrLearningConfig`], a [`DifficultyEstimator`], and per-sample
/// loss history to produce ordered training batches.
pub struct CurriculumSampler {
    config: CurrLearningConfig,
    estimator: DifficultyEstimator,
    sample_difficulties: Vec<SampleDifficulty>,
    current_step: usize,
    n_activated: usize,
}

impl CurriculumSampler {
    /// Construct a new [`CurriculumSampler`].
    ///
    /// Returns [`CurrLearningError::EmptyDataset`] when the estimator has no
    /// samples, or [`CurrLearningError::InvalidConfig`] for bad config values.
    pub fn new(
        config: CurrLearningConfig,
        estimator: DifficultyEstimator,
    ) -> Result<Self, CurrLearningError> {
        if estimator.n_samples == 0 {
            return Err(CurrLearningError::EmptyDataset);
        }
        if config.batch_size == 0 {
            return Err(CurrLearningError::InvalidConfig(
                "batch_size must be > 0".into(),
            ));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(CurrLearningError::InvalidConfig(
                "ema_decay must be in (0, 1)".into(),
            ));
        }

        let n = estimator.n_samples;
        let sample_difficulties: Vec<SampleDifficulty> = (0..n)
            .map(|i| {
                let d = estimator.difficulties.get(i).copied().unwrap_or(0.5);
                SampleDifficulty::new(i, d)
            })
            .collect();

        let n_activated = n.max(1);

        Ok(Self {
            config,
            estimator,
            sample_difficulties,
            current_step: 0,
            n_activated,
        })
    }

    /// Return a batch of sample indices for the current training step.
    pub fn sample_batch(&self, rng_state: &mut u64) -> Result<Vec<usize>, CurrLearningError> {
        let n = self.estimator.n_samples;
        if n == 0 {
            return Err(CurrLearningError::EmptyDataset);
        }

        // During warmup the full dataset is available.
        let sorted = if self.current_step < self.config.warmup_steps {
            let mut all: Vec<usize> = (0..n).collect();
            curr_shuffle(&mut all, rng_state);
            return Ok(all.into_iter().take(self.config.batch_size).collect());
        } else {
            self.estimator.sorted_by_difficulty()
        };

        let pacing = self.current_pacing();
        let losses: Vec<f32> = self
            .sample_difficulties
            .iter()
            .map(|sd| sd.loss_ema)
            .collect();

        curr_select_indices(
            &sorted,
            &self.estimator.difficulties,
            &losses,
            self.config.batch_size,
            &self.config.strategy,
            pacing,
            rng_state,
        )
    }

    /// Update loss EMA values for a set of samples after a training step.
    pub fn update_losses(
        &mut self,
        sample_indices: &[usize],
        losses: &[f32],
    ) -> Result<(), CurrLearningError> {
        if sample_indices.len() != losses.len() {
            return Err(CurrLearningError::InvalidConfig(format!(
                "sample_indices and losses length mismatch: {} vs {}",
                sample_indices.len(),
                losses.len()
            )));
        }
        let step = self.current_step;
        let decay = self.config.ema_decay;
        for (&idx, &loss) in sample_indices.iter().zip(losses.iter()) {
            let sd = self
                .sample_difficulties
                .get_mut(idx)
                .ok_or(CurrLearningError::IndexOutOfBounds(idx))?;
            sd.update_loss(loss, step, decay);
        }
        Ok(())
    }

    /// Advance to the next training step.
    pub fn advance(&mut self) {
        self.current_step += 1;
        // Update n_activated based on current pacing.
        let pacing = self.current_pacing_at(self.current_step);
        let n = self.estimator.n_samples;
        self.n_activated = ((pacing * n as f32).ceil() as usize).clamp(1, n);
    }

    /// Current pacing value (fraction of dataset unlocked), in `[0, 1]`.
    pub fn current_pacing(&self) -> f32 {
        self.current_pacing_at(self.current_step)
    }

    /// Pacing value at an arbitrary step.
    fn current_pacing_at(&self, step: usize) -> f32 {
        if step < self.config.warmup_steps {
            return 1.0;
        }
        let adjusted = step.saturating_sub(self.config.warmup_steps);
        let total = self
            .config
            .n_total_steps
            .saturating_sub(self.config.warmup_steps);
        self.config.pacing_function.pacing_at(adjusted, total)
    }

    /// Number of currently active (unlocked) samples.
    pub fn n_active_samples(&self) -> usize {
        let pacing = self.current_pacing();
        let n = self.estimator.n_samples;
        ((pacing * n as f32).ceil() as usize).clamp(1, n)
    }

    /// Current training step.
    pub fn step(&self) -> usize {
        self.current_step
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &CurrLearningConfig {
        &self.config
    }

    /// Per-sample difficulty state for index `idx`.
    pub fn sample_difficulty(&self, idx: usize) -> Result<&SampleDifficulty, CurrLearningError> {
        self.sample_difficulties
            .get(idx)
            .ok_or(CurrLearningError::IndexOutOfBounds(idx))
    }

    /// Immutable view of all per-sample difficulties (for stats).
    pub(crate) fn all_sample_difficulties(&self) -> &[SampleDifficulty] {
        &self.sample_difficulties
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the pacing value in `[0, 1]` for a given step.
pub fn curr_compute_pacing(pacing_fn: &PacingFunction, step: usize, n_steps: usize) -> f32 {
    pacing_fn.pacing_at(step, n_steps)
}

/// Select `n_select` sample indices from `sorted_indices` (easy→hard) according
/// to the curriculum strategy and current pacing.
///
/// Returns [`CurrLearningError::EmptyDataset`] when `sorted_indices` is empty.
pub fn curr_select_indices(
    sorted_indices: &[usize],
    difficulties: &[f32],
    losses: &[f32],
    n_select: usize,
    strategy: &CurriculumStrategy,
    pacing: f32,
    rng_state: &mut u64,
) -> Result<Vec<usize>, CurrLearningError> {
    if sorted_indices.is_empty() {
        return Err(CurrLearningError::EmptyDataset);
    }

    let n_total = sorted_indices.len();
    let pacing_clamped = pacing.clamp(0.0, 1.0);

    match strategy {
        CurriculumStrategy::EasyFirst => {
            // Unlock the easiest `pacing` fraction of samples.
            let n_unlocked = ((pacing_clamped * n_total as f32).ceil() as usize).clamp(1, n_total);
            let pool: Vec<usize> = sorted_indices[..n_unlocked].to_vec();
            sample_with_replacement(&pool, n_select, rng_state)
        }

        CurriculumStrategy::HardFirst => {
            // Unlock the hardest `pacing` fraction of samples.
            let n_unlocked = ((pacing_clamped * n_total as f32).ceil() as usize).clamp(1, n_total);
            let start = n_total.saturating_sub(n_unlocked);
            let pool: Vec<usize> = sorted_indices[start..].to_vec();
            sample_with_replacement(&pool, n_select, rng_state)
        }

        CurriculumStrategy::MixedPace { easy_fraction } => {
            // initial easy fraction decreases toward 0 as pacing goes to 1.
            let current_easy_fraction = (*easy_fraction * (1.0 - pacing_clamped)).clamp(0.0, 1.0);
            let n_easy =
                ((current_easy_fraction * n_select as f32).round() as usize).clamp(0, n_select);
            let n_hard = n_select - n_easy;

            // Easy pool: first quarter of sorted list.
            let easy_cutoff = (n_total / 4).max(1);
            let easy_pool = &sorted_indices[..easy_cutoff];
            // Hard pool: last quarter of sorted list.
            let hard_start = n_total.saturating_sub(n_total / 4).max(easy_cutoff);
            let hard_pool = &sorted_indices[hard_start..];

            let mut result = sample_with_replacement(easy_pool, n_easy, rng_state)?;
            let hard_samples = sample_with_replacement(
                if hard_pool.is_empty() {
                    sorted_indices
                } else {
                    hard_pool
                },
                n_hard,
                rng_state,
            )?;
            result.extend(hard_samples);
            curr_shuffle(&mut result, rng_state);
            Ok(result)
        }

        CurriculumStrategy::SelfPaced {
            threshold_percentile,
        } => {
            // Select samples with the highest current losses.
            let n_total_losses = losses.len();
            if n_total_losses == 0 {
                return sample_with_replacement(sorted_indices, n_select, rng_state);
            }

            // Find the loss threshold.
            let mut sorted_losses: Vec<(usize, f32)> = sorted_indices
                .iter()
                .map(|&idx| (idx, *losses.get(idx).unwrap_or(&0.0)))
                .collect();
            sorted_losses
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let threshold_idx = curr_percentile_idx(sorted_losses.len(), *threshold_percentile);
            // Select from the highest-loss samples (above threshold).
            let high_loss_pool: Vec<usize> = sorted_losses[threshold_idx..]
                .iter()
                .map(|(idx, _)| *idx)
                .collect();

            let pool = if high_loss_pool.is_empty() {
                sorted_indices.to_vec()
            } else {
                high_loss_pool
            };
            sample_with_replacement(&pool, n_select, rng_state)
        }

        CurriculumStrategy::CompetenceBased { .. } => {
            // Use pacing to unlock increasingly difficult samples.
            let n_unlocked = ((pacing_clamped * n_total as f32).ceil() as usize).clamp(1, n_total);
            let pool: Vec<usize> = sorted_indices[..n_unlocked].to_vec();
            let _ = difficulties; // difficulties used for sorting externally
            sample_with_replacement(&pool, n_select, rng_state)
        }
    }
}

/// Sample `n` indices from `pool` (with replacement) using xorshift64.
fn sample_with_replacement(
    pool: &[usize],
    n: usize,
    rng_state: &mut u64,
) -> Result<Vec<usize>, CurrLearningError> {
    if pool.is_empty() {
        return Err(CurrLearningError::EmptyDataset);
    }
    let m = pool.len();
    let result = (0..n)
        .map(|_| pool[xorshift64(rng_state) as usize % m])
        .collect();
    Ok(result)
}

/// Shuffle `indices` in-place using the Fisher-Yates algorithm and xorshift64.
pub fn curr_shuffle(indices: &mut [usize], rng_state: &mut u64) {
    let n = indices.len();
    if n < 2 {
        return;
    }
    for i in (1..n).rev() {
        let j = xorshift64(rng_state) as usize % (i + 1);
        indices.swap(i, j);
    }
}

/// Return the index at percentile `p` of `n` sorted elements.
///
/// `p = 0.0` → index 0; `p = 1.0` → index `n - 1`.
pub fn curr_percentile_idx(n: usize, p: f32) -> usize {
    if n == 0 {
        return 0;
    }
    let p_clamped = p.clamp(0.0, 1.0);
    let idx = (p_clamped * (n as f32 - 1.0)).round() as usize;
    idx.min(n - 1)
}

/// Normalise a slice of `f32` values to `[0, 1]`.
///
/// - Empty input → empty output.
/// - All-same values → all zeros.
pub fn curr_normalize(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return vec![];
    }
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let range = max - min;
    if range == 0.0 {
        return vec![0.0; values.len()];
    }
    values.iter().map(|v| (v - min) / range).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics and reporting
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot of curriculum learning statistics at the current step.
#[derive(Debug, Clone)]
pub struct CurriculumStats {
    /// Current training step.
    pub step: usize,
    /// Current pacing value.
    pub pacing: f32,
    /// Number of currently active samples.
    pub n_active: usize,
    /// Mean difficulty of all active samples.
    pub mean_difficulty_active: f32,
    /// Mean EMA loss across all samples.
    pub mean_loss_ema: f32,
    /// Number of samples seen at least once.
    pub n_fully_explored: usize,
}

/// Compute current statistics for a [`CurriculumSampler`].
pub fn curr_compute_stats(sampler: &CurriculumSampler) -> CurriculumStats {
    let step = sampler.step();
    let pacing = sampler.current_pacing();
    let n_active = sampler.n_active_samples();
    let all_sd = sampler.all_sample_difficulties();

    let sorted = sampler.estimator.sorted_by_difficulty();
    let active_sorted: Vec<usize> = sorted.iter().take(n_active).copied().collect();

    let mean_difficulty_active = if active_sorted.is_empty() {
        0.0
    } else {
        let sum: f32 = active_sorted
            .iter()
            .filter_map(|&i| sampler.estimator.difficulties.get(i).copied())
            .sum();
        sum / active_sorted.len() as f32
    };

    let mean_loss_ema = if all_sd.is_empty() {
        0.0
    } else {
        let sum: f32 = all_sd.iter().map(|sd| sd.loss_ema).sum();
        sum / all_sd.len() as f32
    };

    let n_fully_explored = all_sd.iter().filter(|sd| sd.n_seen > 0).count();

    CurriculumStats {
        step,
        pacing,
        n_active,
        mean_difficulty_active,
        mean_loss_ema,
        n_fully_explored,
    }
}

/// Format statistics as a human-readable string.
pub fn curr_format_stats(stats: &CurriculumStats) -> String {
    format!(
        "step={} pacing={:.3} active={} mean_diff={:.3} mean_loss_ema={:.4} explored={}",
        stats.step,
        stats.pacing,
        stats.n_active,
        stats.mean_difficulty_active,
        stats.mean_loss_ema,
        stats.n_fully_explored,
    )
}

/// Format configuration as a human-readable string.
pub fn curr_format_config(config: &CurrLearningConfig) -> String {
    format!(
        "strategy={:?} pacing={:?} total_steps={} batch={} ema={:.2} warmup={} rescore={}",
        config.strategy,
        config.pacing_function,
        config.n_total_steps,
        config.batch_size,
        config.ema_decay,
        config.warmup_steps,
        config.rescore_interval,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PacingFunction ────────────────────────────────────────────────────────

    #[test]
    fn test_pacing_linear_step0() {
        let p = PacingFunction::Linear.pacing_at(0, 100);
        assert!((p - 0.0).abs() < 1e-6, "expected 0 got {p}");
    }

    #[test]
    fn test_pacing_linear_full() {
        let p = PacingFunction::Linear.pacing_at(100, 100);
        assert!((p - 1.0).abs() < 1e-6, "expected 1.0 got {p}");
    }

    #[test]
    fn test_pacing_linear_half() {
        let p = PacingFunction::Linear.pacing_at(50, 100);
        assert!((p - 0.5).abs() < 1e-5, "expected 0.5 got {p}");
    }

    #[test]
    fn test_pacing_root_full() {
        let p = PacingFunction::Root { k: 2.0 }.pacing_at(100, 100);
        assert!((p - 1.0).abs() < 1e-5, "expected 1.0 got {p}");
    }

    #[test]
    fn test_pacing_root_half() {
        let p = PacingFunction::Root { k: 2.0 }.pacing_at(50, 100);
        let expected = 0.5_f32.sqrt();
        assert!(
            (p - expected).abs() < 1e-5,
            "expected sqrt(0.5)={expected} got {p}"
        );
    }

    #[test]
    fn test_pacing_root_zero() {
        let p = PacingFunction::Root { k: 2.0 }.pacing_at(0, 100);
        assert!((p - 0.0).abs() < 1e-6, "expected 0 got {p}");
    }

    #[test]
    fn test_pacing_exponential_step0() {
        let p = PacingFunction::Exponential { k: 3.0 }.pacing_at(0, 100);
        assert!((p - 0.0).abs() < 1e-6, "expected 0 got {p}");
    }

    #[test]
    fn test_pacing_exponential_monotone() {
        let pf = PacingFunction::Exponential { k: 2.0 };
        let mut prev = 0.0_f32;
        for step in 1..=20 {
            let cur = pf.pacing_at(step * 5, 100);
            assert!(cur >= prev - 1e-6, "not monotone at step={}", step * 5);
            prev = cur;
        }
    }

    #[test]
    fn test_pacing_exponential_near_one_at_end() {
        let p = PacingFunction::Exponential { k: 5.0 }.pacing_at(100, 100);
        assert!(p > 0.99, "expected near 1.0 got {p}");
    }

    #[test]
    fn test_pacing_step_before_threshold() {
        let p = PacingFunction::Step { threshold_step: 50 }.pacing_at(49, 100);
        assert!((p - 0.0).abs() < 1e-6, "expected 0 got {p}");
    }

    #[test]
    fn test_pacing_step_at_threshold() {
        let p = PacingFunction::Step { threshold_step: 50 }.pacing_at(50, 100);
        assert!((p - 1.0).abs() < 1e-6, "expected 1.0 got {p}");
    }

    #[test]
    fn test_pacing_step_after_threshold() {
        let p = PacingFunction::Step { threshold_step: 50 }.pacing_at(99, 100);
        assert!((p - 1.0).abs() < 1e-6, "expected 1.0 got {p}");
    }

    #[test]
    fn test_pacing_zero_steps_returns_one() {
        let p = PacingFunction::Linear.pacing_at(0, 0);
        assert!((p - 1.0).abs() < 1e-6, "expected 1.0 got {p}");
    }

    // ── SampleDifficulty ──────────────────────────────────────────────────────

    #[test]
    fn test_sample_difficulty_new() {
        let sd = SampleDifficulty::new(3, 0.7);
        assert_eq!(sd.sample_idx, 3);
        assert!((sd.difficulty - 0.7).abs() < 1e-6);
        assert_eq!(sd.n_seen, 0);
        assert_eq!(sd.last_step, 0);
        assert!((sd.loss_ema - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sample_difficulty_clamps_difficulty() {
        let sd = SampleDifficulty::new(0, 2.5);
        assert!((sd.difficulty - 1.0).abs() < 1e-6);
        let sd2 = SampleDifficulty::new(0, -1.0);
        assert!((sd2.difficulty - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sample_difficulty_update_first() {
        let mut sd = SampleDifficulty::new(0, 0.5);
        sd.update_loss(1.0, 10, 0.9);
        assert!(
            (sd.loss_ema - 1.0).abs() < 1e-6,
            "first update sets EMA directly"
        );
        assert_eq!(sd.n_seen, 1);
        assert_eq!(sd.last_step, 10);
    }

    #[test]
    fn test_sample_difficulty_ema_convergence() {
        let mut sd = SampleDifficulty::new(0, 0.0);
        // Feed constant loss 2.0 many times — EMA should converge to 2.0.
        for step in 0..200 {
            sd.update_loss(2.0, step, 0.9);
        }
        assert!(
            (sd.loss_ema - 2.0).abs() < 0.05,
            "ema should converge to 2.0, got {}",
            sd.loss_ema
        );
    }

    #[test]
    fn test_sample_difficulty_estimated() {
        let mut sd = SampleDifficulty::new(0, 0.3);
        sd.update_loss(MAX_REASONABLE_LOSS, 0, 0.9); // first update = EMA = MAX
        let est = sd.estimated_difficulty();
        assert!((est - 1.0).abs() < 1e-5, "expected 1.0 got {est}");
    }

    // ── DifficultyEstimator ───────────────────────────────────────────────────

    #[test]
    fn test_estimator_uniform() {
        let est = DifficultyEstimator::uniform(5);
        assert_eq!(est.n_samples, 5);
        for &d in &est.difficulties {
            assert!((d - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_estimator_from_scores_ok() {
        let scores = vec![1.0, 2.0, 3.0, 4.0];
        let est = DifficultyEstimator::from_scores(&scores).unwrap();
        assert_eq!(est.n_samples, 4);
        assert!((est.difficulties[0] - 0.0).abs() < 1e-6);
        assert!((est.difficulties[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_estimator_from_scores_empty_error() {
        let result = DifficultyEstimator::from_scores(&[]);
        assert!(matches!(result, Err(CurrLearningError::EmptyDataset)));
    }

    #[test]
    fn test_estimator_linear_sequence_first_last() {
        let est = DifficultyEstimator::linear_sequence(5);
        assert!(
            (est.difficulties[0] - 0.0).abs() < 1e-6,
            "first should be 0"
        );
        assert!((est.difficulties[4] - 1.0).abs() < 1e-6, "last should be 1");
    }

    #[test]
    fn test_estimator_linear_sequence_single() {
        let est = DifficultyEstimator::linear_sequence(1);
        assert!((est.difficulties[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_estimator_linear_sequence_zero() {
        let est = DifficultyEstimator::linear_sequence(0);
        assert_eq!(est.n_samples, 0);
        assert!(est.difficulties.is_empty());
    }

    #[test]
    fn test_estimator_from_elevations_frontal_easy() {
        // Elevation 0 → frontal → difficulty 0.
        let est = DifficultyEstimator::from_elevations(&[0.0, std::f32::consts::FRAC_PI_2]);
        assert!(
            (est.difficulties[0] - 0.0).abs() < 1e-5,
            "frontal should be 0"
        );
    }

    #[test]
    fn test_estimator_from_elevations_profile_hard() {
        // Elevation π/2 → profile → difficulty 1.
        let est = DifficultyEstimator::from_elevations(&[0.0, std::f32::consts::FRAC_PI_2]);
        assert!(
            (est.difficulties[1] - 1.0).abs() < 1e-5,
            "profile should be 1"
        );
    }

    #[test]
    fn test_estimator_from_elevations_negative_symmetric() {
        let est = DifficultyEstimator::from_elevations(&[-std::f32::consts::FRAC_PI_2]);
        assert!((est.difficulties[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_estimator_sorted_by_difficulty() {
        let scores = vec![0.9, 0.1, 0.5, 0.3];
        let est = DifficultyEstimator::from_scores(&scores).unwrap();
        let sorted = est.sorted_by_difficulty();
        // Sorted should give indices in ascending difficulty order.
        let diffs: Vec<f32> = sorted.iter().map(|&i| est.difficulties[i]).collect();
        for w in diffs.windows(2) {
            assert!(w[0] <= w[1] + 1e-6, "not sorted: {:?}", w);
        }
    }

    #[test]
    fn test_estimator_percentile_index_boundaries() {
        let est = DifficultyEstimator::uniform(10);
        assert_eq!(est.percentile_index(0.0), 0);
        assert_eq!(est.percentile_index(1.0), 9);
    }

    #[test]
    fn test_estimator_difficulty_of_ok() {
        let est = DifficultyEstimator::uniform(3);
        assert!(est.difficulty_of(0).is_ok());
        assert!(est.difficulty_of(2).is_ok());
    }

    #[test]
    fn test_estimator_difficulty_of_oob() {
        let est = DifficultyEstimator::uniform(3);
        assert!(matches!(
            est.difficulty_of(3),
            Err(CurrLearningError::IndexOutOfBounds(3))
        ));
    }

    // ── CurriculumSampler ─────────────────────────────────────────────────────

    fn make_sampler(n: usize, strategy: CurriculumStrategy) -> CurriculumSampler {
        let estimator = DifficultyEstimator::linear_sequence(n);
        let config = CurrLearningConfig {
            strategy,
            pacing_function: PacingFunction::Linear,
            n_total_steps: 100,
            batch_size: 4,
            ema_decay: 0.9,
            warmup_steps: 0,
            rescore_interval: 10,
        };
        CurriculumSampler::new(config, estimator).unwrap()
    }

    #[test]
    fn test_sampler_new_ok() {
        let sampler = make_sampler(20, CurriculumStrategy::EasyFirst);
        assert_eq!(sampler.step(), 0);
    }

    #[test]
    fn test_sampler_new_empty_error() {
        let estimator = DifficultyEstimator::uniform(0);
        let config = CurrLearningConfig::default();
        let result = CurriculumSampler::new(config, estimator);
        assert!(matches!(result, Err(CurrLearningError::EmptyDataset)));
    }

    #[test]
    fn test_sampler_sample_batch_size() {
        let mut rng = 12345u64;
        let sampler = make_sampler(20, CurriculumStrategy::EasyFirst);
        let batch = sampler.sample_batch(&mut rng).unwrap();
        assert_eq!(batch.len(), 4);
    }

    #[test]
    fn test_sampler_sample_batch_indices_valid() {
        let mut rng = 42u64;
        let sampler = make_sampler(20, CurriculumStrategy::EasyFirst);
        let batch = sampler.sample_batch(&mut rng).unwrap();
        for idx in batch {
            assert!(idx < 20, "index {idx} out of range");
        }
    }

    #[test]
    fn test_sampler_easy_first_bias() {
        // With EasyFirst at step=0 (pacing≈0), almost all samples should be low-difficulty.
        let estimator = DifficultyEstimator::linear_sequence(100);
        let config = CurrLearningConfig {
            strategy: CurriculumStrategy::EasyFirst,
            pacing_function: PacingFunction::Linear,
            n_total_steps: 10_000,
            batch_size: 20,
            ema_decay: 0.9,
            warmup_steps: 0,
            rescore_interval: 100,
        };
        let sampler = CurriculumSampler::new(config, estimator).unwrap();
        let mut rng = 99u64;
        let mut total: usize = 0;
        let runs = 10;
        for _ in 0..runs {
            let batch = sampler.sample_batch(&mut rng).unwrap();
            total += batch.iter().sum::<usize>();
        }
        // Average sample index should be much below 50 (midpoint).
        let avg = total as f32 / (runs * 20) as f32;
        assert!(avg < 30.0, "expected easy (low index) bias, avg_idx={avg}");
    }

    #[test]
    fn test_sampler_hard_first_bias() {
        let estimator = DifficultyEstimator::linear_sequence(100);
        let config = CurrLearningConfig {
            strategy: CurriculumStrategy::HardFirst,
            pacing_function: PacingFunction::Linear,
            n_total_steps: 10_000,
            batch_size: 20,
            ema_decay: 0.9,
            warmup_steps: 0,
            rescore_interval: 100,
        };
        let sampler = CurriculumSampler::new(config, estimator).unwrap();
        let mut rng = 77u64;
        let mut total: usize = 0;
        let runs = 10;
        for _ in 0..runs {
            let batch = sampler.sample_batch(&mut rng).unwrap();
            total += batch.iter().sum::<usize>();
        }
        let avg = total as f32 / (runs * 20) as f32;
        assert!(avg > 70.0, "expected hard (high index) bias, avg_idx={avg}");
    }

    #[test]
    fn test_sampler_update_losses() {
        let mut sampler = make_sampler(20, CurriculumStrategy::EasyFirst);
        let result = sampler.update_losses(&[0, 1, 2], &[0.5, 0.8, 1.2]);
        assert!(result.is_ok());
        let sd0 = sampler.sample_difficulty(0).unwrap();
        assert!(
            (sd0.loss_ema - 0.5).abs() < 1e-5,
            "first update = direct set"
        );
    }

    #[test]
    fn test_sampler_update_losses_ema_convergence() {
        let mut sampler = make_sampler(20, CurriculumStrategy::EasyFirst);
        for _ in 0..200 {
            sampler.update_losses(&[5], &[3.0]).unwrap();
        }
        let sd = sampler.sample_difficulty(5).unwrap();
        assert!(
            (sd.loss_ema - 3.0).abs() < 0.1,
            "EMA should converge to 3.0"
        );
    }

    #[test]
    fn test_sampler_advance_increments_step() {
        let mut sampler = make_sampler(20, CurriculumStrategy::EasyFirst);
        sampler.advance();
        assert_eq!(sampler.step(), 1);
        sampler.advance();
        assert_eq!(sampler.step(), 2);
    }

    #[test]
    fn test_sampler_pacing_monotone_with_advance() {
        let mut sampler = make_sampler(100, CurriculumStrategy::EasyFirst);
        let mut prev = sampler.current_pacing();
        for _ in 0..50 {
            sampler.advance();
            let cur = sampler.current_pacing();
            assert!(cur >= prev - 1e-6);
            prev = cur;
        }
    }

    #[test]
    fn test_sampler_n_active_grows() {
        let mut sampler = make_sampler(100, CurriculumStrategy::EasyFirst);
        let n0 = sampler.n_active_samples();
        for _ in 0..50 {
            sampler.advance();
        }
        let n50 = sampler.n_active_samples();
        assert!(n50 >= n0, "n_active should grow with pacing");
    }

    #[test]
    fn test_sampler_warmup_uses_full_dataset() {
        let estimator = DifficultyEstimator::linear_sequence(100);
        let config = CurrLearningConfig {
            strategy: CurriculumStrategy::EasyFirst,
            pacing_function: PacingFunction::Linear,
            n_total_steps: 1000,
            batch_size: 10,
            ema_decay: 0.9,
            warmup_steps: 100,
            rescore_interval: 10,
        };
        let sampler = CurriculumSampler::new(config, estimator).unwrap();
        // At step 0 (within warmup), should return a batch from the full dataset.
        let mut rng = 1u64;
        let batch = sampler.sample_batch(&mut rng).unwrap();
        assert_eq!(batch.len(), 10);
        // All indices should be valid.
        for &idx in &batch {
            assert!(idx < 100);
        }
    }

    // ── curr_compute_pacing ───────────────────────────────────────────────────

    #[test]
    fn test_curr_compute_pacing_linear() {
        assert!((curr_compute_pacing(&PacingFunction::Linear, 0, 100) - 0.0).abs() < 1e-6);
        assert!((curr_compute_pacing(&PacingFunction::Linear, 100, 100) - 1.0).abs() < 1e-6);
        assert!((curr_compute_pacing(&PacingFunction::Linear, 50, 100) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_curr_compute_pacing_root() {
        let pf = PacingFunction::Root { k: 2.0 };
        assert!((curr_compute_pacing(&pf, 100, 100) - 1.0).abs() < 1e-5);
        let expected = 0.5_f32.sqrt();
        assert!((curr_compute_pacing(&pf, 50, 100) - expected).abs() < 1e-5);
    }

    #[test]
    fn test_curr_compute_pacing_step() {
        let pf = PacingFunction::Step { threshold_step: 50 };
        assert!((curr_compute_pacing(&pf, 49, 100) - 0.0).abs() < 1e-6);
        assert!((curr_compute_pacing(&pf, 50, 100) - 1.0).abs() < 1e-6);
    }

    // ── curr_select_indices ───────────────────────────────────────────────────

    #[test]
    fn test_curr_select_indices_count() {
        let sorted: Vec<usize> = (0..10).collect();
        let difficulties = vec![0.0; 10];
        let losses = vec![0.0; 10];
        let mut rng = 42u64;
        let result = curr_select_indices(
            &sorted,
            &difficulties,
            &losses,
            5,
            &CurriculumStrategy::EasyFirst,
            1.0,
            &mut rng,
        )
        .unwrap();
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_curr_select_indices_empty_error() {
        let sorted: Vec<usize> = vec![];
        let difficulties = vec![];
        let losses = vec![];
        let mut rng = 1u64;
        let result = curr_select_indices(
            &sorted,
            &difficulties,
            &losses,
            3,
            &CurriculumStrategy::EasyFirst,
            1.0,
            &mut rng,
        );
        assert!(matches!(result, Err(CurrLearningError::EmptyDataset)));
    }

    #[test]
    fn test_curr_select_indices_hard_first_high_indices() {
        let sorted: Vec<usize> = (0..100).collect();
        let difficulties: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
        let losses = vec![0.0; 100];
        let mut rng = 5u64;
        let result = curr_select_indices(
            &sorted,
            &difficulties,
            &losses,
            20,
            &CurriculumStrategy::HardFirst,
            0.1,
            &mut rng,
        )
        .unwrap();
        let avg = result.iter().sum::<usize>() as f32 / result.len() as f32;
        assert!(
            avg > 80.0,
            "HardFirst should select high indices, avg={avg}"
        );
    }

    // ── curr_shuffle ──────────────────────────────────────────────────────────

    #[test]
    fn test_curr_shuffle_length_unchanged() {
        let mut indices: Vec<usize> = (0..20).collect();
        let mut rng = 7u64;
        curr_shuffle(&mut indices, &mut rng);
        assert_eq!(indices.len(), 20);
    }

    #[test]
    fn test_curr_shuffle_deterministic() {
        let mut a: Vec<usize> = (0..10).collect();
        let mut b: Vec<usize> = (0..10).collect();
        let mut rng_a = 1234u64;
        let mut rng_b = 1234u64;
        curr_shuffle(&mut a, &mut rng_a);
        curr_shuffle(&mut b, &mut rng_b);
        assert_eq!(a, b, "same seed must produce same shuffle");
    }

    #[test]
    fn test_curr_shuffle_changes_order() {
        let original: Vec<usize> = (0..20).collect();
        let mut shuffled = original.clone();
        let mut rng = 9999u64;
        curr_shuffle(&mut shuffled, &mut rng);
        assert_ne!(shuffled, original, "shuffled should differ from original");
    }

    // ── curr_percentile_idx ───────────────────────────────────────────────────

    #[test]
    fn test_percentile_idx_zero() {
        assert_eq!(curr_percentile_idx(10, 0.0), 0);
    }

    #[test]
    fn test_percentile_idx_one() {
        assert_eq!(curr_percentile_idx(10, 1.0), 9);
    }

    #[test]
    fn test_percentile_idx_half() {
        let idx = curr_percentile_idx(10, 0.5);
        // Should be 5 (rounded from 4.5 → 5 by round()).
        assert_eq!(idx, 5, "expected midpoint, got {idx}");
    }

    #[test]
    fn test_percentile_idx_zero_n() {
        assert_eq!(curr_percentile_idx(0, 0.5), 0);
    }

    // ── curr_normalize ────────────────────────────────────────────────────────

    #[test]
    fn test_normalize_empty() {
        let out = curr_normalize(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_normalize_all_same() {
        let out = curr_normalize(&[5.0, 5.0, 5.0]);
        for v in out {
            assert!((v - 0.0).abs() < 1e-6, "all-same should normalize to 0");
        }
    }

    #[test]
    fn test_normalize_range() {
        let out = curr_normalize(&[0.0, 1.0, 2.0, 4.0]);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6);
        for &v in &out {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    // ── Statistics & reporting ────────────────────────────────────────────────

    #[test]
    fn test_curr_compute_stats_fields() {
        let sampler = make_sampler(20, CurriculumStrategy::EasyFirst);
        let stats = curr_compute_stats(&sampler);
        assert_eq!(stats.step, 0);
        assert!(stats.pacing >= 0.0 && stats.pacing <= 1.0);
        assert!(stats.n_active >= 1);
        assert!(stats.n_fully_explored == 0, "nothing trained yet");
    }

    #[test]
    fn test_curr_format_stats_non_empty() {
        let sampler = make_sampler(10, CurriculumStrategy::EasyFirst);
        let stats = curr_compute_stats(&sampler);
        let s = curr_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("step="));
    }

    #[test]
    fn test_curr_format_config_non_empty() {
        let config = CurrLearningConfig::default();
        let s = curr_format_config(&config);
        assert!(!s.is_empty());
        assert!(s.contains("batch="));
    }

    // ── Self-paced strategy ───────────────────────────────────────────────────

    #[test]
    fn test_self_paced_selects_high_loss() {
        // Create samples where indices 5-9 have high losses.
        let sorted: Vec<usize> = (0..10).collect();
        let difficulties = vec![0.5; 10];
        let mut losses = vec![0.1; 10];
        losses[7] = 5.0;
        losses[8] = 4.5;
        losses[9] = 4.0;
        let mut rng = 42u64;
        let strategy = CurriculumStrategy::SelfPaced {
            threshold_percentile: 0.7,
        };
        let result =
            curr_select_indices(&sorted, &difficulties, &losses, 3, &strategy, 1.0, &mut rng)
                .unwrap();
        // Result should contain some of the high-loss indices.
        assert!(!result.is_empty());
        let has_high = result.iter().any(|&i| i >= 7);
        assert!(has_high, "should select from high-loss samples");
    }

    // ── MixedPace strategy ────────────────────────────────────────────────────

    #[test]
    fn test_mixed_pace_returns_batch() {
        let sorted: Vec<usize> = (0..100).collect();
        let difficulties: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
        let losses = vec![0.0; 100];
        let mut rng = 55u64;
        let strategy = CurriculumStrategy::MixedPace { easy_fraction: 0.8 };
        let result = curr_select_indices(
            &sorted,
            &difficulties,
            &losses,
            10,
            &strategy,
            0.5,
            &mut rng,
        )
        .unwrap();
        assert_eq!(result.len(), 10);
    }

    // ── CompetenceBased strategy ──────────────────────────────────────────────

    #[test]
    fn test_competence_based_returns_batch() {
        let sorted: Vec<usize> = (0..50).collect();
        let difficulties: Vec<f32> = (0..50).map(|i| i as f32 / 49.0).collect();
        let losses = vec![0.0; 50];
        let mut rng = 8u64;
        let strategy = CurriculumStrategy::CompetenceBased { target_psnr: 30.0 };
        let result =
            curr_select_indices(&sorted, &difficulties, &losses, 5, &strategy, 0.5, &mut rng)
                .unwrap();
        assert_eq!(result.len(), 5);
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_sampler_sample_difficulty_oob() {
        let sampler = make_sampler(10, CurriculumStrategy::EasyFirst);
        let result = sampler.sample_difficulty(100);
        assert!(matches!(
            result,
            Err(CurrLearningError::IndexOutOfBounds(100))
        ));
    }

    #[test]
    fn test_sampler_update_losses_length_mismatch() {
        let mut sampler = make_sampler(10, CurriculumStrategy::EasyFirst);
        let result = sampler.update_losses(&[0, 1], &[0.5]);
        assert!(matches!(result, Err(CurrLearningError::InvalidConfig(_))));
    }

    #[test]
    fn test_estimator_from_elevations_empty() {
        let est = DifficultyEstimator::from_elevations(&[]);
        assert_eq!(est.n_samples, 0);
        assert!(est.difficulties.is_empty());
    }
}
