//! Unconditional dropout for classifier-free guidance (CFG) training.
//!
//! During training, conditioning embeddings (identity, camera, expression) are
//! randomly replaced with a null "unconditional" embedding at a fixed rate. This
//! trains the model to generate without conditioning, enabling inference-time CFG
//! interpolation. Distinct from `guidance.rs` (inference-time) and
//! `sampler_apply_cfg` (sampler step).
//!
//! ## Overview
//!
//! - [`UncondSchedule`]: how the dropout rate changes over training.
//! - [`NullEmbedding`]: the null/unconditional embedding that replaces conditioning.
//! - [`DropoutMask`]: per-sample boolean mask for a batch.
//! - [`UncondDropout`]: stateful core dropout module.
//! - [`MultiModalDropout`]: joint or independent dropout across several modalities.
//! - [`UncondDropoutStats`]: EMA-based statistics for monitoring.
//!
//! ## Standalone utilities
//!
//! - [`uncond_cfg_combine`] / [`uncond_cfg_combine_batch`]: inference-time CFG
//!   interpolation `output = uncond + scale * (cond - uncond)`.
//! - [`uncond_update_stats`], [`uncond_format_stats`], [`uncond_format_config`]:
//!   statistics helpers.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by unconditional-dropout operations.
#[derive(Debug, Error, PartialEq)]
pub enum UncondDropoutError {
    /// Dropout rate not in `[0, 1]`.
    #[error("invalid dropout rate: must be in [0, 1], got {0}")]
    InvalidRate(f32),

    /// Embedding dimension mismatch between conditioning and null embedding.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Required dimension.
        expected: usize,
        /// Received dimension.
        got: usize,
    },

    /// Batch has zero elements where at least one is required.
    #[error("empty batch")]
    EmptyBatch,

    /// Generic invalid configuration.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

// ---------------------------------------------------------------------------
// PRNG — xorshift64  (spec-mandated implementation)
// ---------------------------------------------------------------------------

/// Advance a 64-bit xorshift PRNG and return the new state.
///
/// Guards against the fixed-point at 0 by resetting to 1 after the shift.
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

/// Draw a uniform f32 in [0, 1] from the xorshift64 PRNG.
#[inline]
fn xorshift_f32(state: &mut u64) -> f32 {
    xorshift64(state) as f32 / u64::MAX as f32
}

// ---------------------------------------------------------------------------
// UncondSchedule
// ---------------------------------------------------------------------------

/// How the unconditional dropout rate changes over training steps.
#[derive(Debug, Clone, PartialEq)]
pub enum UncondSchedule {
    /// Fixed rate throughout training.
    Constant(f32),
    /// Linear warmup from 0 to `target_rate` over `warmup_steps`, then constant.
    Warmup {
        /// The final (plateau) dropout rate.
        target_rate: f32,
        /// Number of steps over which the rate linearly increases.
        warmup_steps: usize,
    },
    /// Cosine annealing: rate oscillates between `min_rate` and `max_rate`.
    ///
    /// `rate(step) = min_rate + (max_rate - min_rate) * 0.5 * (1 + cos(2π * step / period))`
    CosineAnnealing {
        /// Minimum dropout rate (reached at step = period/2).
        min_rate: f32,
        /// Maximum dropout rate (reached at step = 0, period, …).
        max_rate: f32,
        /// Length of one cosine cycle in training steps.
        period_steps: usize,
    },
    /// Curriculum: linearly increase rate from `start_rate` to `end_rate` over `n_steps`.
    Curriculum {
        /// Dropout rate at step 0.
        start_rate: f32,
        /// Dropout rate at step `n_steps` (and beyond).
        end_rate: f32,
        /// Number of steps for the ramp.
        n_steps: usize,
    },
}

impl UncondSchedule {
    /// Compute the dropout rate at a given training step.
    pub fn rate_at_step(&self, step: usize) -> f32 {
        match self {
            Self::Constant(r) => *r,

            Self::Warmup {
                target_rate,
                warmup_steps,
            } => {
                if *warmup_steps == 0 || step >= *warmup_steps {
                    *target_rate
                } else {
                    *target_rate * (step as f32 / *warmup_steps as f32)
                }
            }

            Self::CosineAnnealing {
                min_rate,
                max_rate,
                period_steps,
            } => {
                if *period_steps == 0 {
                    return *max_rate;
                }
                let t = step as f32 / *period_steps as f32;
                let cos_val = (2.0 * std::f32::consts::PI * t).cos();
                min_rate + (max_rate - min_rate) * 0.5 * (1.0 + cos_val)
            }

            Self::Curriculum {
                start_rate,
                end_rate,
                n_steps,
            } => {
                if *n_steps == 0 || step >= *n_steps {
                    *end_rate
                } else {
                    let frac = step as f32 / *n_steps as f32;
                    start_rate + (end_rate - start_rate) * frac
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NullEmbedding
// ---------------------------------------------------------------------------

/// The null embedding that replaces conditioning when a sample is dropped.
///
/// At inference the model conditions on this for the "unconditional" branch.
#[derive(Debug, Clone, PartialEq)]
pub struct NullEmbedding {
    /// Number of dimensions.
    pub dim: usize,
    /// The actual null-embedding values.
    pub values: Vec<f32>,
}

impl NullEmbedding {
    /// Zero null embedding (most common choice).
    pub fn zeros(dim: usize) -> Self {
        Self {
            dim,
            values: vec![0.0f32; dim],
        }
    }

    /// Constant-value null embedding (every element set to `value`).
    pub fn constant(dim: usize, value: f32) -> Self {
        Self {
            dim,
            values: vec![value; dim],
        }
    }

    /// Small-noise null embedding, generated deterministically via xorshift64
    /// with seed 42.  Values are in `[-scale, scale]`.
    pub fn noise(dim: usize, scale: f32) -> Self {
        let mut state = 42u64;
        let values = (0..dim)
            .map(|_| {
                // Map [0, 1] → [-scale, scale]
                let u = xorshift_f32(&mut state);
                (u * 2.0 - 1.0) * scale
            })
            .collect();
        Self { dim, values }
    }
}

// ---------------------------------------------------------------------------
// NullEmbeddingType
// ---------------------------------------------------------------------------

/// How to construct the null embedding inside [`UncondDropoutConfig`].
#[derive(Debug, Clone, PartialEq)]
pub enum NullEmbeddingType {
    /// All zeros.
    Zeros,
    /// Constant value.
    Constant(f32),
    /// Small deterministic noise; the f32 is the scale.
    Noise(f32),
}

impl NullEmbeddingType {
    fn build(&self, dim: usize) -> NullEmbedding {
        match self {
            Self::Zeros => NullEmbedding::zeros(dim),
            Self::Constant(v) => NullEmbedding::constant(dim, *v),
            Self::Noise(s) => NullEmbedding::noise(dim, *s),
        }
    }
}

// ---------------------------------------------------------------------------
// DropoutMask
// ---------------------------------------------------------------------------

/// Records which samples in a batch use conditioning vs. the null embedding.
///
/// `true` = use conditioning, `false` = drop (use null embedding).
#[derive(Debug, Clone, PartialEq)]
pub struct DropoutMask {
    /// Per-sample flag: `true` means conditioning is kept.
    pub use_conditioning: Vec<bool>,
    /// Number of samples in the batch.
    pub batch_size: usize,
    /// Dropout rate used when generating this mask.
    pub dropout_rate: f32,
}

impl DropoutMask {
    /// Every sample keeps its conditioning.
    pub fn all_conditional(batch_size: usize) -> Self {
        Self {
            use_conditioning: vec![true; batch_size],
            batch_size,
            dropout_rate: 0.0,
        }
    }

    /// Every sample uses the null (unconditional) embedding.
    pub fn all_unconditional(batch_size: usize) -> Self {
        Self {
            use_conditioning: vec![false; batch_size],
            batch_size,
            dropout_rate: 1.0,
        }
    }

    /// Number of samples that keep conditioning.
    pub fn n_conditional(&self) -> usize {
        self.use_conditioning.iter().filter(|&&b| b).count()
    }

    /// Number of samples that use the null embedding.
    pub fn n_unconditional(&self) -> usize {
        self.use_conditioning.iter().filter(|&&b| !b).count()
    }

    /// `true` when both conditional and unconditional samples are present.
    pub fn is_mixed(&self) -> bool {
        let n_cond = self.n_conditional();
        n_cond > 0 && n_cond < self.batch_size
    }
}

// ---------------------------------------------------------------------------
// UncondDropoutConfig
// ---------------------------------------------------------------------------

/// Configuration for the core [`UncondDropout`] module.
#[derive(Debug, Clone, PartialEq)]
pub struct UncondDropoutConfig {
    /// Conditioning embedding dimension.
    pub conditioning_dim: usize,
    /// How the dropout rate changes over training.
    pub schedule: UncondSchedule,
    /// How the null embedding is constructed.
    pub null_type: NullEmbeddingType,
    /// Ensure at least this fraction of each batch keeps conditioning.  Default: 0.0.
    pub min_batch_cond_fraction: f32,
}

// Note: there is deliberately no `joint_dropout` field here. `UncondDropout`
// (built from this config) handles exactly one conditioning stream, so
// "joint vs. independent dropout" is meaningless for it — that choice only
// makes sense once more than one modality is involved, which is exactly
// what [`MultiModalDropout`] is for (it takes its own `joint: bool`
// constructor argument). An earlier revision carried a `joint_dropout` field
// here that `UncondDropout` never read; it has been removed rather than
// wired up, since wiring it would have meant duplicating state that
// `MultiModalDropout` already owns correctly.
impl Default for UncondDropoutConfig {
    fn default() -> Self {
        Self {
            conditioning_dim: 256,
            schedule: UncondSchedule::Constant(0.1),
            null_type: NullEmbeddingType::Zeros,
            min_batch_cond_fraction: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// UncondDropoutStats
// ---------------------------------------------------------------------------

/// Running statistics for monitoring dropout behaviour during training.
#[derive(Debug, Clone, PartialEq)]
pub struct UncondDropoutStats {
    /// Current training step.
    pub step: usize,
    /// Exponential moving average of the observed dropout rate (decay = 0.99).
    pub mean_dropout_rate: f32,
    /// Total number of conditional samples seen so far.
    pub n_conditional_total: usize,
    /// Total number of unconditional (dropped) samples seen so far.
    pub n_unconditional_total: usize,
    /// Theoretical dropout rate from the schedule at the current step.
    pub current_schedule_rate: f32,
}

impl Default for UncondDropoutStats {
    fn default() -> Self {
        Self {
            step: 0,
            mean_dropout_rate: 0.0,
            n_conditional_total: 0,
            n_unconditional_total: 0,
            current_schedule_rate: 0.0,
        }
    }
}

/// Update `stats` using an applied `mask` and the theoretical `schedule_rate`.
pub fn uncond_update_stats(stats: &mut UncondDropoutStats, mask: &DropoutMask, schedule_rate: f32) {
    let observed_rate = if mask.batch_size == 0 {
        0.0
    } else {
        mask.n_unconditional() as f32 / mask.batch_size as f32
    };
    // EMA with decay=0.99
    const DECAY: f32 = 0.99;
    if stats.step == 0 {
        stats.mean_dropout_rate = observed_rate;
    } else {
        stats.mean_dropout_rate = DECAY * stats.mean_dropout_rate + (1.0 - DECAY) * observed_rate;
    }
    stats.n_conditional_total += mask.n_conditional();
    stats.n_unconditional_total += mask.n_unconditional();
    stats.current_schedule_rate = schedule_rate;
    stats.step += 1;
}

// ---------------------------------------------------------------------------
// UncondDropout
// ---------------------------------------------------------------------------

/// Core stateful module for CFG unconditional dropout during training.
pub struct UncondDropout {
    config: UncondDropoutConfig,
    null_embedding: NullEmbedding,
    step: usize,
    stats: UncondDropoutStats,
}

impl UncondDropout {
    /// Construct a new dropout module, validating the config.
    pub fn new(config: UncondDropoutConfig) -> Result<Self, UncondDropoutError> {
        // Validate schedule rates
        validate_schedule_rates(&config.schedule)?;
        if !(0.0..=1.0).contains(&config.min_batch_cond_fraction) {
            return Err(UncondDropoutError::InvalidConfig(format!(
                "min_batch_cond_fraction must be in [0, 1], got {}",
                config.min_batch_cond_fraction
            )));
        }
        let null_embedding = config.null_type.build(config.conditioning_dim);
        Ok(Self {
            config,
            null_embedding,
            step: 0,
            stats: UncondDropoutStats::default(),
        })
    }

    /// Generate a dropout mask for a batch.
    ///
    /// Samples Bernoulli(dropout_rate) independently per sample.  When
    /// `min_batch_cond_fraction > 0`, ensures the required minimum of
    /// conditional samples by flipping excess drops back to conditional.
    pub fn sample_mask(
        &self,
        batch_size: usize,
        rng_state: &mut u64,
    ) -> Result<DropoutMask, UncondDropoutError> {
        if batch_size == 0 {
            return Err(UncondDropoutError::EmptyBatch);
        }
        let rate = self.current_rate();
        let mut use_conditioning: Vec<bool> = (0..batch_size)
            .map(|_| xorshift_f32(rng_state) >= rate)
            .collect();

        // Enforce minimum conditional fraction
        let min_cond = (self.config.min_batch_cond_fraction * batch_size as f32).ceil() as usize;
        let n_cond = use_conditioning.iter().filter(|&&b| b).count();
        if n_cond < min_cond {
            // Flip some uncond → cond (in order) until satisfied
            let mut needed = min_cond - n_cond;
            for flag in use_conditioning.iter_mut() {
                if needed == 0 {
                    break;
                }
                if !*flag {
                    *flag = true;
                    needed -= 1;
                }
            }
        }

        Ok(DropoutMask {
            use_conditioning,
            batch_size,
            dropout_rate: rate,
        })
    }

    /// Apply dropout to a batch of conditioning embeddings.
    ///
    /// `embeddings[i]` must have length `conditioning_dim`.
    /// Returns a new `Vec<Vec<f32>>` where dropped samples get the null embedding.
    pub fn apply_dropout(
        &self,
        embeddings: &[Vec<f32>],
        mask: &DropoutMask,
    ) -> Result<Vec<Vec<f32>>, UncondDropoutError> {
        if embeddings.len() != mask.batch_size {
            return Err(UncondDropoutError::DimensionMismatch {
                expected: mask.batch_size,
                got: embeddings.len(),
            });
        }
        embeddings
            .iter()
            .zip(mask.use_conditioning.iter())
            .map(|(emb, &keep)| {
                if emb.len() != self.config.conditioning_dim {
                    return Err(UncondDropoutError::DimensionMismatch {
                        expected: self.config.conditioning_dim,
                        got: emb.len(),
                    });
                }
                if keep {
                    Ok(emb.clone())
                } else {
                    Ok(self.null_embedding.values.clone())
                }
            })
            .collect()
    }

    /// Apply dropout to a flat batch embedding of shape `[batch_size * dim]`.
    pub fn apply_dropout_flat(
        &self,
        embeddings: &[f32],
        batch_size: usize,
        mask: &DropoutMask,
    ) -> Result<Vec<f32>, UncondDropoutError> {
        let dim = self.config.conditioning_dim;
        let expected_len = batch_size * dim;
        if embeddings.len() != expected_len {
            return Err(UncondDropoutError::DimensionMismatch {
                expected: expected_len,
                got: embeddings.len(),
            });
        }
        if mask.batch_size != batch_size {
            return Err(UncondDropoutError::DimensionMismatch {
                expected: batch_size,
                got: mask.batch_size,
            });
        }
        let mut out = Vec::with_capacity(expected_len);
        for (i, &keep) in mask.use_conditioning.iter().enumerate() {
            let slice = &embeddings[i * dim..(i + 1) * dim];
            if keep {
                out.extend_from_slice(slice);
            } else {
                out.extend_from_slice(&self.null_embedding.values);
            }
        }
        Ok(out)
    }

    /// Current dropout rate given the schedule and current step.
    pub fn current_rate(&self) -> f32 {
        self.config.schedule.rate_at_step(self.step)
    }

    /// Advance the internal step counter by one.
    pub fn advance_step(&mut self) {
        self.step += 1;
    }

    /// Current training step.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &UncondDropoutConfig {
        &self.config
    }

    /// Reference to the null embedding.
    pub fn null_embedding(&self) -> &NullEmbedding {
        &self.null_embedding
    }

    /// Reference to the running statistics.
    pub fn stats(&self) -> &UncondDropoutStats {
        &self.stats
    }

    /// Apply a mask, record statistics, and advance the step.
    ///
    /// Convenience method combining `apply_dropout`, `uncond_update_stats`,
    /// and `advance_step`.
    pub fn apply_and_advance(
        &mut self,
        embeddings: &[Vec<f32>],
        mask: &DropoutMask,
    ) -> Result<Vec<Vec<f32>>, UncondDropoutError> {
        let out = self.apply_dropout(embeddings, mask)?;
        let schedule_rate = self.current_rate();
        uncond_update_stats(&mut self.stats, mask, schedule_rate);
        self.advance_step();
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// MultiModalDropout
// ---------------------------------------------------------------------------

/// Per-modality dropout configuration for multi-modal conditioning.
pub struct ModalityDropout {
    /// Human-readable name (e.g. `"identity"`, `"camera"`, `"expression"`).
    pub name: String,
    /// Embedding dimension for this modality.
    pub dim: usize,
    /// Dropout rate for this modality (independent of other modalities).
    pub dropout_rate: f32,
    /// Null embedding for this modality.
    pub null_embedding: NullEmbedding,
}

impl ModalityDropout {
    /// Construct with validation.
    pub fn new(
        name: impl Into<String>,
        dim: usize,
        dropout_rate: f32,
        null_embedding: NullEmbedding,
    ) -> Result<Self, UncondDropoutError> {
        if !(0.0..=1.0).contains(&dropout_rate) {
            return Err(UncondDropoutError::InvalidRate(dropout_rate));
        }
        if null_embedding.dim != dim {
            return Err(UncondDropoutError::DimensionMismatch {
                expected: dim,
                got: null_embedding.dim,
            });
        }
        Ok(Self {
            name: name.into(),
            dim,
            dropout_rate,
            null_embedding,
        })
    }
}

/// Apply joint or independent dropout across multiple conditioning modalities.
pub struct MultiModalDropout {
    modalities: Vec<ModalityDropout>,
    joint: bool,
}

impl MultiModalDropout {
    /// Construct with validation (at least one modality required).
    pub fn new(modalities: Vec<ModalityDropout>, joint: bool) -> Result<Self, UncondDropoutError> {
        if modalities.is_empty() {
            return Err(UncondDropoutError::InvalidConfig(
                "at least one modality required".into(),
            ));
        }
        Ok(Self { modalities, joint })
    }

    /// Generate one [`DropoutMask`] per modality.
    ///
    /// When `joint=true`, all modalities share the **same** mask (generated
    /// using the first modality's dropout rate).  When `joint=false`, each
    /// modality gets an independent mask drawn with its own dropout rate.
    pub fn sample_masks(
        &self,
        batch_size: usize,
        rng_state: &mut u64,
    ) -> Result<Vec<DropoutMask>, UncondDropoutError> {
        if batch_size == 0 {
            return Err(UncondDropoutError::EmptyBatch);
        }
        if self.joint {
            // Use first modality's rate for the joint mask.
            let rate = self.modalities[0].dropout_rate;
            let use_conditioning: Vec<bool> = (0..batch_size)
                .map(|_| xorshift_f32(rng_state) >= rate)
                .collect();
            let shared = DropoutMask {
                use_conditioning: use_conditioning.clone(),
                batch_size,
                dropout_rate: rate,
            };
            Ok(self.modalities.iter().map(|_| shared.clone()).collect())
        } else {
            self.modalities
                .iter()
                .map(|m| {
                    let rate = m.dropout_rate;
                    let use_conditioning: Vec<bool> = (0..batch_size)
                        .map(|_| xorshift_f32(rng_state) >= rate)
                        .collect();
                    Ok(DropoutMask {
                        use_conditioning,
                        batch_size,
                        dropout_rate: rate,
                    })
                })
                .collect()
        }
    }

    /// Apply the masks to embeddings for each modality.
    ///
    /// `embeddings[m][i]` is the embedding for modality `m`, sample `i`.
    /// `masks[m]` is the dropout mask for modality `m`.
    pub fn apply_all(
        &self,
        embeddings: &[Vec<Vec<f32>>],
        masks: &[DropoutMask],
    ) -> Result<Vec<Vec<Vec<f32>>>, UncondDropoutError> {
        if embeddings.len() != self.modalities.len() {
            return Err(UncondDropoutError::DimensionMismatch {
                expected: self.modalities.len(),
                got: embeddings.len(),
            });
        }
        if masks.len() != self.modalities.len() {
            return Err(UncondDropoutError::DimensionMismatch {
                expected: self.modalities.len(),
                got: masks.len(),
            });
        }
        self.modalities
            .iter()
            .zip(embeddings.iter())
            .zip(masks.iter())
            .map(|((modality, emb_batch), mask)| {
                if emb_batch.len() != mask.batch_size {
                    return Err(UncondDropoutError::DimensionMismatch {
                        expected: mask.batch_size,
                        got: emb_batch.len(),
                    });
                }
                emb_batch
                    .iter()
                    .zip(mask.use_conditioning.iter())
                    .map(|(emb, &keep)| {
                        if emb.len() != modality.dim {
                            return Err(UncondDropoutError::DimensionMismatch {
                                expected: modality.dim,
                                got: emb.len(),
                            });
                        }
                        if keep {
                            Ok(emb.clone())
                        } else {
                            Ok(modality.null_embedding.values.clone())
                        }
                    })
                    .collect::<Result<Vec<Vec<f32>>, _>>()
            })
            .collect()
    }

    /// Number of modalities.
    pub fn n_modalities(&self) -> usize {
        self.modalities.len()
    }

    /// Names of all modalities.
    pub fn modality_names(&self) -> Vec<&str> {
        self.modalities.iter().map(|m| m.name.as_str()).collect()
    }
}

// ---------------------------------------------------------------------------
// Inference-time CFG interpolation
// ---------------------------------------------------------------------------

/// Compute CFG guidance interpolation at inference time.
///
/// `output = uncond_output + scale * (cond_output - uncond_output)`
///
/// - `scale = 0.0` → returns `uncond_output` unchanged.
/// - `scale = 1.0` → returns `cond_output`.
/// - `scale > 1.0` → extrapolation beyond the conditional prediction.
pub fn uncond_cfg_combine(
    cond_output: &[f32],
    uncond_output: &[f32],
    guidance_scale: f32,
) -> Result<Vec<f32>, UncondDropoutError> {
    if cond_output.len() != uncond_output.len() {
        return Err(UncondDropoutError::DimensionMismatch {
            expected: cond_output.len(),
            got: uncond_output.len(),
        });
    }
    let out = cond_output
        .iter()
        .zip(uncond_output.iter())
        .map(|(&c, &u)| u + guidance_scale * (c - u))
        .collect();
    Ok(out)
}

/// Batch version of [`uncond_cfg_combine`].
///
/// Applies CFG combination independently to each sample pair.
pub fn uncond_cfg_combine_batch(
    cond_outputs: &[Vec<f32>],
    uncond_outputs: &[Vec<f32>],
    guidance_scale: f32,
) -> Result<Vec<Vec<f32>>, UncondDropoutError> {
    if cond_outputs.len() != uncond_outputs.len() {
        return Err(UncondDropoutError::DimensionMismatch {
            expected: cond_outputs.len(),
            got: uncond_outputs.len(),
        });
    }
    cond_outputs
        .iter()
        .zip(uncond_outputs.iter())
        .map(|(c, u)| uncond_cfg_combine(c, u, guidance_scale))
        .collect()
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Format a human-readable summary of [`UncondDropoutStats`].
pub fn uncond_format_stats(stats: &UncondDropoutStats) -> String {
    let total = stats.n_conditional_total + stats.n_unconditional_total;
    let actual_rate = if total == 0 {
        0.0
    } else {
        stats.n_unconditional_total as f32 / total as f32
    };
    format!(
        "UncondDropoutStats(step={}, schedule_rate={:.4}, ema_dropout_rate={:.4}, \
         n_conditional={}, n_unconditional={}, actual_rate={:.4})",
        stats.step,
        stats.current_schedule_rate,
        stats.mean_dropout_rate,
        stats.n_conditional_total,
        stats.n_unconditional_total,
        actual_rate,
    )
}

/// Format a human-readable summary of [`UncondDropoutConfig`].
pub fn uncond_format_config(config: &UncondDropoutConfig) -> String {
    let schedule_name = match &config.schedule {
        UncondSchedule::Constant(r) => format!("Constant({r:.4})"),
        UncondSchedule::Warmup {
            target_rate,
            warmup_steps,
        } => format!("Warmup(target={target_rate:.4}, steps={warmup_steps})"),
        UncondSchedule::CosineAnnealing {
            min_rate,
            max_rate,
            period_steps,
        } => {
            format!("CosineAnnealing(min={min_rate:.4}, max={max_rate:.4}, period={period_steps})")
        }
        UncondSchedule::Curriculum {
            start_rate,
            end_rate,
            n_steps,
        } => format!("Curriculum(start={start_rate:.4}, end={end_rate:.4}, n={n_steps})"),
    };
    let null_name = match &config.null_type {
        NullEmbeddingType::Zeros => "Zeros".to_string(),
        NullEmbeddingType::Constant(v) => format!("Constant({v:.4})"),
        NullEmbeddingType::Noise(s) => format!("Noise(scale={s:.4})"),
    };
    format!(
        "UncondDropoutConfig(dim={}, schedule={}, null={}, min_cond_frac={:.3})",
        config.conditioning_dim, schedule_name, null_name, config.min_batch_cond_fraction,
    )
}

// ---------------------------------------------------------------------------
// Backward-compatibility helpers (kept to avoid breaking lib.rs re-exports)
// ---------------------------------------------------------------------------

/// Compute per-sample loss weights based on the dropout pattern.
///
/// | Conditioning          | Weight |
/// |-----------------------|--------|
/// | Conditional (kept)    | 1.0    |
/// | Unconditional (dropped)| 2.0   |
pub fn compute_dropout_weights(mask: &DropoutMask) -> Vec<f32> {
    mask.use_conditioning
        .iter()
        .map(|&keep| if keep { 1.0f32 } else { 2.0f32 })
        .collect()
}

/// Normalise `weights` in-place so that their mean equals 1.0.
///
/// No-op when the slice is empty or the mean is zero.
pub fn normalize_weights(weights: &mut [f32]) {
    if weights.is_empty() {
        return;
    }
    let mean = weights.iter().sum::<f32>() / weights.len() as f32;
    if mean == 0.0 {
        return;
    }
    for w in weights.iter_mut() {
        *w /= mean;
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn validate_schedule_rates(schedule: &UncondSchedule) -> Result<(), UncondDropoutError> {
    match schedule {
        UncondSchedule::Constant(r) => {
            if !(*r >= 0.0 && *r <= 1.0) {
                return Err(UncondDropoutError::InvalidRate(*r));
            }
        }
        UncondSchedule::Warmup { target_rate, .. } => {
            if !(*target_rate >= 0.0 && *target_rate <= 1.0) {
                return Err(UncondDropoutError::InvalidRate(*target_rate));
            }
        }
        UncondSchedule::CosineAnnealing {
            min_rate, max_rate, ..
        } => {
            if !(*min_rate >= 0.0 && *min_rate <= 1.0) {
                return Err(UncondDropoutError::InvalidRate(*min_rate));
            }
            if !(*max_rate >= 0.0 && *max_rate <= 1.0) {
                return Err(UncondDropoutError::InvalidRate(*max_rate));
            }
        }
        UncondSchedule::Curriculum {
            start_rate,
            end_rate,
            ..
        } => {
            if !(*start_rate >= 0.0 && *start_rate <= 1.0) {
                return Err(UncondDropoutError::InvalidRate(*start_rate));
            }
            if !(*end_rate >= 0.0 && *end_rate <= 1.0) {
                return Err(UncondDropoutError::InvalidRate(*end_rate));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // UncondSchedule::Constant
    // -----------------------------------------------------------------------

    #[test]
    fn test_schedule_constant_rate_always_same() {
        let s = UncondSchedule::Constant(0.15);
        for step in [0, 1, 100, 10_000] {
            assert!((s.rate_at_step(step) - 0.15).abs() < 1e-6);
        }
    }

    #[test]
    fn test_schedule_constant_zero() {
        let s = UncondSchedule::Constant(0.0);
        assert_eq!(s.rate_at_step(0), 0.0);
        assert_eq!(s.rate_at_step(999), 0.0);
    }

    #[test]
    fn test_schedule_constant_one() {
        let s = UncondSchedule::Constant(1.0);
        assert_eq!(s.rate_at_step(0), 1.0);
        assert_eq!(s.rate_at_step(5000), 1.0);
    }

    // -----------------------------------------------------------------------
    // UncondSchedule::Warmup
    // -----------------------------------------------------------------------

    #[test]
    fn test_schedule_warmup_step_zero_is_zero() {
        let s = UncondSchedule::Warmup {
            target_rate: 0.5,
            warmup_steps: 100,
        };
        assert!((s.rate_at_step(0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_schedule_warmup_at_warmup_steps_is_target() {
        let s = UncondSchedule::Warmup {
            target_rate: 0.3,
            warmup_steps: 50,
        };
        assert!((s.rate_at_step(50) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_schedule_warmup_beyond_warmup_is_target() {
        let s = UncondSchedule::Warmup {
            target_rate: 0.2,
            warmup_steps: 10,
        };
        assert!((s.rate_at_step(11) - 0.2).abs() < 1e-6);
        assert!((s.rate_at_step(1000) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_schedule_warmup_monotone() {
        let s = UncondSchedule::Warmup {
            target_rate: 0.4,
            warmup_steps: 20,
        };
        let mut prev = 0.0f32;
        for step in 0..=20 {
            let r = s.rate_at_step(step);
            assert!(
                r >= prev - 1e-6,
                "not monotone at step {step}: {r} < {prev}"
            );
            prev = r;
        }
    }

    // -----------------------------------------------------------------------
    // UncondSchedule::CosineAnnealing
    // -----------------------------------------------------------------------

    #[test]
    fn test_schedule_cosine_step0_is_max() {
        let s = UncondSchedule::CosineAnnealing {
            min_rate: 0.05,
            max_rate: 0.3,
            period_steps: 100,
        };
        let r = s.rate_at_step(0);
        assert!((r - 0.3).abs() < 1e-5, "expected 0.3, got {r}");
    }

    #[test]
    fn test_schedule_cosine_half_period_is_min() {
        let s = UncondSchedule::CosineAnnealing {
            min_rate: 0.05,
            max_rate: 0.3,
            period_steps: 100,
        };
        let r = s.rate_at_step(50);
        assert!((r - 0.05).abs() < 1e-5, "expected 0.05, got {r}");
    }

    #[test]
    fn test_schedule_cosine_period_returns_max() {
        let s = UncondSchedule::CosineAnnealing {
            min_rate: 0.02,
            max_rate: 0.25,
            period_steps: 40,
        };
        let r = s.rate_at_step(40);
        assert!((r - 0.25).abs() < 1e-5, "expected 0.25, got {r}");
    }

    #[test]
    fn test_schedule_cosine_oscillates() {
        let s = UncondSchedule::CosineAnnealing {
            min_rate: 0.0,
            max_rate: 1.0,
            period_steps: 20,
        };
        // First period rises back to max
        let at_period = s.rate_at_step(20);
        assert!((at_period - 1.0).abs() < 1e-5);
        // Midpoint of second period is min
        let at_mid2 = s.rate_at_step(30);
        assert!((at_mid2 - 0.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // UncondSchedule::Curriculum
    // -----------------------------------------------------------------------

    #[test]
    fn test_schedule_curriculum_start_at_step0() {
        let s = UncondSchedule::Curriculum {
            start_rate: 0.05,
            end_rate: 0.4,
            n_steps: 100,
        };
        assert!((s.rate_at_step(0) - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_schedule_curriculum_end_at_n_steps() {
        let s = UncondSchedule::Curriculum {
            start_rate: 0.0,
            end_rate: 0.5,
            n_steps: 50,
        };
        assert!((s.rate_at_step(50) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_schedule_curriculum_monotone_increasing() {
        let s = UncondSchedule::Curriculum {
            start_rate: 0.0,
            end_rate: 0.8,
            n_steps: 100,
        };
        let mut prev = s.rate_at_step(0);
        for step in 1..=100 {
            let r = s.rate_at_step(step);
            assert!(r >= prev - 1e-6, "not monotone at step {step}");
            prev = r;
        }
    }

    #[test]
    fn test_schedule_curriculum_beyond_n_steps_is_end() {
        let s = UncondSchedule::Curriculum {
            start_rate: 0.1,
            end_rate: 0.6,
            n_steps: 10,
        };
        assert!((s.rate_at_step(11) - 0.6).abs() < 1e-6);
        assert!((s.rate_at_step(999) - 0.6).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // NullEmbedding
    // -----------------------------------------------------------------------

    #[test]
    fn test_null_embedding_zeros_all_zero() {
        let emb = NullEmbedding::zeros(16);
        assert_eq!(emb.dim, 16);
        assert_eq!(emb.values.len(), 16);
        assert!(emb.values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_null_embedding_zeros_empty() {
        let emb = NullEmbedding::zeros(0);
        assert_eq!(emb.dim, 0);
        assert!(emb.values.is_empty());
    }

    #[test]
    fn test_null_embedding_constant_all_value() {
        let emb = NullEmbedding::constant(8, 3.1);
        assert_eq!(emb.dim, 8);
        assert!(emb.values.iter().all(|&v| (v - 3.1).abs() < 1e-6));
    }

    #[test]
    fn test_null_embedding_noise_length_correct() {
        let emb = NullEmbedding::noise(32, 0.01);
        assert_eq!(emb.dim, 32);
        assert_eq!(emb.values.len(), 32);
    }

    #[test]
    fn test_null_embedding_noise_values_small() {
        let scale = 0.01;
        let emb = NullEmbedding::noise(64, scale);
        for &v in &emb.values {
            assert!(v.abs() <= scale + 1e-7, "value {v} exceeded scale {scale}");
        }
    }

    #[test]
    fn test_null_embedding_noise_deterministic() {
        let a = NullEmbedding::noise(16, 0.05);
        let b = NullEmbedding::noise(16, 0.05);
        assert_eq!(a.values, b.values);
    }

    // -----------------------------------------------------------------------
    // DropoutMask
    // -----------------------------------------------------------------------

    #[test]
    fn test_dropout_mask_all_conditional() {
        let m = DropoutMask::all_conditional(5);
        assert_eq!(m.batch_size, 5);
        assert!(m.use_conditioning.iter().all(|&b| b));
        assert_eq!(m.n_conditional(), 5);
        assert_eq!(m.n_unconditional(), 0);
    }

    #[test]
    fn test_dropout_mask_all_unconditional() {
        let m = DropoutMask::all_unconditional(4);
        assert_eq!(m.batch_size, 4);
        assert!(m.use_conditioning.iter().all(|&b| !b));
        assert_eq!(m.n_conditional(), 0);
        assert_eq!(m.n_unconditional(), 4);
    }

    #[test]
    fn test_dropout_mask_n_cond_plus_n_uncond_equals_batch() {
        let m = DropoutMask {
            use_conditioning: vec![true, false, true, false, true],
            batch_size: 5,
            dropout_rate: 0.4,
        };
        assert_eq!(m.n_conditional() + m.n_unconditional(), 5);
    }

    #[test]
    fn test_dropout_mask_is_mixed_true() {
        let m = DropoutMask {
            use_conditioning: vec![true, false, true],
            batch_size: 3,
            dropout_rate: 0.33,
        };
        assert!(m.is_mixed());
    }

    #[test]
    fn test_dropout_mask_is_mixed_all_cond() {
        let m = DropoutMask::all_conditional(4);
        assert!(!m.is_mixed());
    }

    #[test]
    fn test_dropout_mask_is_mixed_all_uncond() {
        let m = DropoutMask::all_unconditional(4);
        assert!(!m.is_mixed());
    }

    // -----------------------------------------------------------------------
    // UncondDropout::new validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_uncond_dropout_new_invalid_rate() {
        let cfg = UncondDropoutConfig {
            schedule: UncondSchedule::Constant(1.5),
            ..Default::default()
        };
        assert!(matches!(
            UncondDropout::new(cfg),
            Err(UncondDropoutError::InvalidRate(_))
        ));
    }

    #[test]
    fn test_uncond_dropout_new_negative_rate() {
        let cfg = UncondDropoutConfig {
            schedule: UncondSchedule::Constant(-0.1),
            ..Default::default()
        };
        assert!(matches!(
            UncondDropout::new(cfg),
            Err(UncondDropoutError::InvalidRate(_))
        ));
    }

    #[test]
    fn test_uncond_dropout_new_valid() {
        let cfg = UncondDropoutConfig::default();
        assert!(UncondDropout::new(cfg).is_ok());
    }

    // -----------------------------------------------------------------------
    // UncondDropout::sample_mask
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_mask_rate_zero_all_conditional() {
        let cfg = UncondDropoutConfig {
            schedule: UncondSchedule::Constant(0.0),
            ..Default::default()
        };
        let dropout = UncondDropout::new(cfg).unwrap();
        let mut rng = 12345u64;
        let mask = dropout.sample_mask(100, &mut rng).unwrap();
        assert_eq!(mask.n_unconditional(), 0, "rate=0 → all conditional");
    }

    #[test]
    fn test_sample_mask_rate_one_all_unconditional() {
        let cfg = UncondDropoutConfig {
            schedule: UncondSchedule::Constant(1.0),
            ..Default::default()
        };
        let dropout = UncondDropout::new(cfg).unwrap();
        let mut rng = 99u64;
        let mask = dropout.sample_mask(50, &mut rng).unwrap();
        assert_eq!(mask.n_conditional(), 0, "rate=1 → all unconditional");
    }

    #[test]
    fn test_sample_mask_large_batch_approximate_rate() {
        let rate = 0.3;
        let cfg = UncondDropoutConfig {
            conditioning_dim: 4,
            schedule: UncondSchedule::Constant(rate),
            ..Default::default()
        };
        let dropout = UncondDropout::new(cfg).unwrap();
        let mut rng = 77u64;
        let mask = dropout.sample_mask(10_000, &mut rng).unwrap();
        let actual_rate = mask.n_unconditional() as f32 / 10_000.0;
        assert!(
            (actual_rate - rate).abs() < 0.02,
            "actual rate {actual_rate} too far from {rate}"
        );
    }

    #[test]
    fn test_sample_mask_empty_batch_error() {
        let cfg = UncondDropoutConfig::default();
        let dropout = UncondDropout::new(cfg).unwrap();
        let mut rng = 1u64;
        assert!(matches!(
            dropout.sample_mask(0, &mut rng),
            Err(UncondDropoutError::EmptyBatch)
        ));
    }

    // -----------------------------------------------------------------------
    // UncondDropout::apply_dropout
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_dropout_masked_get_null() {
        let dim = 4;
        let cfg = UncondDropoutConfig {
            conditioning_dim: dim,
            schedule: UncondSchedule::Constant(0.0),
            null_type: NullEmbeddingType::Zeros,
            ..Default::default()
        };
        let dropout = UncondDropout::new(cfg).unwrap();
        let embeddings = vec![vec![1.0f32; dim], vec![2.0f32; dim]];
        // Manually craft an all-uncond mask
        let mask = DropoutMask::all_unconditional(2);
        let out = dropout.apply_dropout(&embeddings, &mask).unwrap();
        for row in &out {
            assert!(
                row.iter().all(|&v| v == 0.0),
                "dropped sample should be null"
            );
        }
    }

    #[test]
    fn test_apply_dropout_non_masked_unchanged() {
        let dim = 3;
        let cfg = UncondDropoutConfig {
            conditioning_dim: dim,
            ..Default::default()
        };
        let dropout = UncondDropout::new(cfg).unwrap();
        let orig = vec![vec![7.0f32, 8.0, 9.0]];
        let mask = DropoutMask::all_conditional(1);
        let out = dropout.apply_dropout(&orig, &mask).unwrap();
        assert_eq!(out[0], orig[0]);
    }

    #[test]
    fn test_apply_dropout_dim_mismatch_error() {
        let cfg = UncondDropoutConfig {
            conditioning_dim: 4,
            ..Default::default()
        };
        let dropout = UncondDropout::new(cfg).unwrap();
        let embeddings = vec![vec![1.0f32; 3]]; // wrong dim
        let mask = DropoutMask::all_conditional(1);
        assert!(matches!(
            dropout.apply_dropout(&embeddings, &mask),
            Err(UncondDropoutError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_apply_dropout_batch_size_mismatch_error() {
        let cfg = UncondDropoutConfig {
            conditioning_dim: 4,
            ..Default::default()
        };
        let dropout = UncondDropout::new(cfg).unwrap();
        let embeddings = vec![vec![1.0f32; 4], vec![2.0f32; 4]];
        let mask = DropoutMask::all_conditional(3); // wrong batch size
        assert!(matches!(
            dropout.apply_dropout(&embeddings, &mask),
            Err(UncondDropoutError::DimensionMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // UncondDropout::apply_dropout_flat
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_dropout_flat_correct() {
        let dim = 2;
        let cfg = UncondDropoutConfig {
            conditioning_dim: dim,
            null_type: NullEmbeddingType::Zeros,
            ..Default::default()
        };
        let dropout = UncondDropout::new(cfg).unwrap();
        // Two samples: first kept, second dropped
        let flat = vec![1.0f32, 2.0, 3.0, 4.0];
        let mask = DropoutMask {
            use_conditioning: vec![true, false],
            batch_size: 2,
            dropout_rate: 0.5,
        };
        let out = dropout.apply_dropout_flat(&flat, 2, &mask).unwrap();
        assert_eq!(out[0], 1.0); // sample 0: kept
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 0.0); // sample 1: dropped → zeros
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn test_apply_dropout_flat_size_mismatch_error() {
        let cfg = UncondDropoutConfig {
            conditioning_dim: 4,
            ..Default::default()
        };
        let dropout = UncondDropout::new(cfg).unwrap();
        let flat = vec![1.0f32; 7]; // 7 ≠ 2*4
        let mask = DropoutMask::all_conditional(2);
        assert!(matches!(
            dropout.apply_dropout_flat(&flat, 2, &mask),
            Err(UncondDropoutError::DimensionMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // UncondDropout::current_rate and advance_step
    // -----------------------------------------------------------------------

    #[test]
    fn test_current_rate_matches_schedule() {
        let cfg = UncondDropoutConfig {
            schedule: UncondSchedule::Warmup {
                target_rate: 0.5,
                warmup_steps: 10,
            },
            ..Default::default()
        };
        let dropout = UncondDropout::new(cfg).unwrap();
        assert!((dropout.current_rate() - 0.0).abs() < 1e-6); // step=0
    }

    #[test]
    fn test_advance_step_increments() {
        let cfg = UncondDropoutConfig::default();
        let mut dropout = UncondDropout::new(cfg).unwrap();
        assert_eq!(dropout.step(), 0);
        dropout.advance_step();
        assert_eq!(dropout.step(), 1);
        dropout.advance_step();
        assert_eq!(dropout.step(), 2);
    }

    // -----------------------------------------------------------------------
    // MultiModalDropout joint=true → all masks identical
    // -----------------------------------------------------------------------

    #[test]
    fn test_multimodal_joint_masks_identical() {
        let m1 = ModalityDropout::new("id", 4, 0.5, NullEmbedding::zeros(4)).unwrap();
        let m2 = ModalityDropout::new("cam", 4, 0.5, NullEmbedding::zeros(4)).unwrap();
        let m3 = ModalityDropout::new("expr", 4, 0.5, NullEmbedding::zeros(4)).unwrap();
        let mmd = MultiModalDropout::new(vec![m1, m2, m3], true).unwrap();
        let mut rng = 42u64;
        let masks = mmd.sample_masks(8, &mut rng).unwrap();
        assert_eq!(masks.len(), 3);
        for mask in &masks[1..] {
            assert_eq!(
                mask.use_conditioning, masks[0].use_conditioning,
                "joint masks must be identical"
            );
        }
    }

    // -----------------------------------------------------------------------
    // MultiModalDropout joint=false → masks may differ
    // -----------------------------------------------------------------------

    #[test]
    fn test_multimodal_independent_can_differ() {
        // Use rate=0.5 and a large batch; statistically masks will differ.
        let m1 = ModalityDropout::new("a", 4, 0.5, NullEmbedding::zeros(4)).unwrap();
        let m2 = ModalityDropout::new("b", 4, 0.5, NullEmbedding::zeros(4)).unwrap();
        let mmd = MultiModalDropout::new(vec![m1, m2], false).unwrap();
        let mut rng = 12345u64;
        let masks = mmd.sample_masks(200, &mut rng).unwrap();
        // The two masks must not be identical for a 200-sample batch at 50% drop.
        assert_ne!(
            masks[0].use_conditioning, masks[1].use_conditioning,
            "independent masks should differ on a large batch"
        );
    }

    // -----------------------------------------------------------------------
    // MultiModalDropout::sample_masks returns one mask per modality
    // -----------------------------------------------------------------------

    #[test]
    fn test_multimodal_one_mask_per_modality() {
        let mods: Vec<ModalityDropout> = (0..4)
            .map(|i| {
                ModalityDropout::new(format!("m{i}"), 8, 0.1, NullEmbedding::zeros(8)).unwrap()
            })
            .collect();
        let mmd = MultiModalDropout::new(mods, false).unwrap();
        let mut rng = 1u64;
        let masks = mmd.sample_masks(10, &mut rng).unwrap();
        assert_eq!(masks.len(), 4);
    }

    // -----------------------------------------------------------------------
    // MultiModalDropout::apply_all
    // -----------------------------------------------------------------------

    #[test]
    fn test_multimodal_apply_all_correct_masking() {
        let dim = 3;
        let m1 = ModalityDropout::new("id", dim, 0.0, NullEmbedding::zeros(dim)).unwrap();
        let m2 = ModalityDropout::new("cam", dim, 0.0, NullEmbedding::constant(dim, -1.0)).unwrap();
        let mmd = MultiModalDropout::new(vec![m1, m2], false).unwrap();

        let emb1 = vec![vec![1.0f32; dim], vec![2.0f32; dim]];
        let emb2 = vec![vec![3.0f32; dim], vec![4.0f32; dim]];
        // First sample kept, second dropped for modality 0
        let mask0 = DropoutMask {
            use_conditioning: vec![true, false],
            batch_size: 2,
            dropout_rate: 0.5,
        };
        // Second sample kept for modality 1
        let mask1 = DropoutMask {
            use_conditioning: vec![false, true],
            batch_size: 2,
            dropout_rate: 0.5,
        };
        let out = mmd.apply_all(&[emb1, emb2], &[mask0, mask1]).unwrap();
        // modality 0, sample 0: kept → [1,1,1]
        assert_eq!(out[0][0], vec![1.0f32; dim]);
        // modality 0, sample 1: dropped → zeros
        assert_eq!(out[0][1], vec![0.0f32; dim]);
        // modality 1, sample 0: dropped → [-1,-1,-1]
        assert_eq!(out[1][0], vec![-1.0f32; dim]);
        // modality 1, sample 1: kept → [4,4,4]
        assert_eq!(out[1][1], vec![4.0f32; dim]);
    }

    // -----------------------------------------------------------------------
    // uncond_update_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_uncond_update_stats_accumulates() {
        let mut stats = UncondDropoutStats::default();
        let mask = DropoutMask {
            use_conditioning: vec![true, false, true, false],
            batch_size: 4,
            dropout_rate: 0.5,
        };
        uncond_update_stats(&mut stats, &mask, 0.5);
        assert_eq!(stats.n_conditional_total, 2);
        assert_eq!(stats.n_unconditional_total, 2);
        assert_eq!(stats.step, 1);

        uncond_update_stats(&mut stats, &mask, 0.5);
        assert_eq!(stats.n_conditional_total, 4);
        assert_eq!(stats.n_unconditional_total, 4);
        assert_eq!(stats.step, 2);
    }

    #[test]
    fn test_uncond_update_stats_first_step_initialises_ema() {
        let mut stats = UncondDropoutStats::default();
        let mask = DropoutMask::all_unconditional(10);
        uncond_update_stats(&mut stats, &mask, 1.0);
        assert!((stats.mean_dropout_rate - 1.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // uncond_cfg_combine
    // -----------------------------------------------------------------------

    #[test]
    fn test_cfg_combine_scale_zero_is_uncond() {
        let cond = vec![1.0f32, 2.0, 3.0];
        let uncond = vec![4.0f32, 5.0, 6.0];
        let out = uncond_cfg_combine(&cond, &uncond, 0.0).unwrap();
        assert_eq!(out, uncond);
    }

    #[test]
    fn test_cfg_combine_scale_one_is_cond() {
        let cond = vec![1.0f32, 2.0, 3.0];
        let uncond = vec![4.0f32, 5.0, 6.0];
        let out = uncond_cfg_combine(&cond, &uncond, 1.0).unwrap();
        for (o, c) in out.iter().zip(cond.iter()) {
            assert!((o - c).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cfg_combine_scale_two_extrapolation() {
        let cond = vec![3.0f32];
        let uncond = vec![1.0f32];
        // uncond + 2*(cond - uncond) = 1 + 2*(3-1) = 5
        let out = uncond_cfg_combine(&cond, &uncond, 2.0).unwrap();
        assert!((out[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_cfg_combine_length_mismatch_error() {
        let cond = vec![1.0f32, 2.0];
        let uncond = vec![1.0f32];
        assert!(matches!(
            uncond_cfg_combine(&cond, &uncond, 1.0),
            Err(UncondDropoutError::DimensionMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // uncond_cfg_combine_batch
    // -----------------------------------------------------------------------

    #[test]
    fn test_cfg_combine_batch_processes_each() {
        let conds = vec![vec![2.0f32], vec![4.0f32]];
        let unconds = vec![vec![0.0f32], vec![0.0f32]];
        let out = uncond_cfg_combine_batch(&conds, &unconds, 1.5).unwrap();
        // 0 + 1.5*(2-0)=3 ; 0+1.5*(4-0)=6
        assert!((out[0][0] - 3.0).abs() < 1e-6);
        assert!((out[1][0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_cfg_combine_batch_length_mismatch_error() {
        let conds = vec![vec![1.0f32]];
        let unconds = vec![vec![0.0f32], vec![0.0f32]];
        assert!(matches!(
            uncond_cfg_combine_batch(&conds, &unconds, 1.0),
            Err(UncondDropoutError::DimensionMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // uncond_format_stats / uncond_format_config
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_stats_non_empty() {
        let stats = UncondDropoutStats::default();
        let s = uncond_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("UncondDropoutStats"));
    }

    #[test]
    fn test_format_config_non_empty() {
        let cfg = UncondDropoutConfig::default();
        let s = uncond_format_config(&cfg);
        assert!(!s.is_empty());
        assert!(s.contains("UncondDropoutConfig"));
    }

    /// Regression test: `UncondDropoutConfig` previously carried a
    /// `joint_dropout` field that `UncondDropout` (single conditioning
    /// stream) never read — only `MultiModalDropout::new`'s own `joint`
    /// argument controls that behaviour. The field was removed rather than
    /// wired up; guard against it (or an equally decorative substitute)
    /// silently reappearing in the config summary.
    #[test]
    fn test_format_config_does_not_claim_joint_dropout() {
        let cfg = UncondDropoutConfig::default();
        let s = uncond_format_config(&cfg);
        assert!(
            !s.to_lowercase().contains("joint"),
            "UncondDropoutConfig has no way to affect joint-vs-independent \
             dropout (that lives on MultiModalDropout::new), so its summary \
             must not claim otherwise: {s}"
        );
    }

    #[test]
    fn test_format_config_all_schedule_variants() {
        for cfg in [
            UncondDropoutConfig {
                schedule: UncondSchedule::Constant(0.1),
                ..Default::default()
            },
            UncondDropoutConfig {
                schedule: UncondSchedule::Warmup {
                    target_rate: 0.2,
                    warmup_steps: 100,
                },
                ..Default::default()
            },
            UncondDropoutConfig {
                schedule: UncondSchedule::CosineAnnealing {
                    min_rate: 0.05,
                    max_rate: 0.3,
                    period_steps: 50,
                },
                ..Default::default()
            },
            UncondDropoutConfig {
                schedule: UncondSchedule::Curriculum {
                    start_rate: 0.0,
                    end_rate: 0.5,
                    n_steps: 200,
                },
                ..Default::default()
            },
        ] {
            let s = uncond_format_config(&cfg);
            assert!(!s.is_empty());
        }
    }

    // -----------------------------------------------------------------------
    // compute_dropout_weights / normalize_weights (backward compat)
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_dropout_weights_all_cond() {
        let mask = DropoutMask::all_conditional(4);
        let w = compute_dropout_weights(&mask);
        assert!(w.iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_compute_dropout_weights_all_uncond() {
        let mask = DropoutMask::all_unconditional(4);
        let w = compute_dropout_weights(&mask);
        assert!(w.iter().all(|&x| (x - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_normalize_weights_mean_one() {
        let mut weights = vec![1.0f32, 1.5, 2.0, 1.5, 1.0];
        normalize_weights(&mut weights);
        let mean = weights.iter().sum::<f32>() / weights.len() as f32;
        assert!((mean - 1.0).abs() < 1e-6, "mean was {mean}");
    }

    #[test]
    fn test_normalize_weights_empty_noop() {
        let mut weights: Vec<f32> = vec![];
        normalize_weights(&mut weights); // must not panic
    }

    // -----------------------------------------------------------------------
    // modality_names helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_modality_names() {
        let mods = vec![
            ModalityDropout::new("alpha", 4, 0.1, NullEmbedding::zeros(4)).unwrap(),
            ModalityDropout::new("beta", 4, 0.2, NullEmbedding::zeros(4)).unwrap(),
        ];
        let mmd = MultiModalDropout::new(mods, false).unwrap();
        assert_eq!(mmd.modality_names(), vec!["alpha", "beta"]);
        assert_eq!(mmd.n_modalities(), 2);
    }

    // -----------------------------------------------------------------------
    // MultiModalDropout empty modalities → error
    // -----------------------------------------------------------------------

    #[test]
    fn test_multimodal_empty_error() {
        let result = MultiModalDropout::new(vec![], false);
        assert!(matches!(result, Err(UncondDropoutError::InvalidConfig(_))));
    }

    // -----------------------------------------------------------------------
    // ModalityDropout invalid rate
    // -----------------------------------------------------------------------

    #[test]
    fn test_modality_dropout_invalid_rate() {
        let result = ModalityDropout::new("x", 4, 1.5, NullEmbedding::zeros(4));
        assert!(matches!(result, Err(UncondDropoutError::InvalidRate(_))));
    }

    // -----------------------------------------------------------------------
    // ModalityDropout dim mismatch
    // -----------------------------------------------------------------------

    #[test]
    fn test_modality_dropout_dim_mismatch() {
        let null = NullEmbedding::zeros(8); // dim 8 ≠ declared dim 4
        let result = ModalityDropout::new("x", 4, 0.1, null);
        assert!(matches!(
            result,
            Err(UncondDropoutError::DimensionMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // UncondDropout::apply_and_advance increments stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_and_advance_increments_stats_and_step() {
        let cfg = UncondDropoutConfig {
            conditioning_dim: 2,
            ..Default::default()
        };
        let mut dropout = UncondDropout::new(cfg).unwrap();
        let embs = vec![vec![1.0f32, 2.0]];
        let mask = DropoutMask::all_conditional(1);
        dropout.apply_and_advance(&embs, &mask).unwrap();
        assert_eq!(dropout.step(), 1);
        assert_eq!(dropout.stats().n_conditional_total, 1);
    }
}
