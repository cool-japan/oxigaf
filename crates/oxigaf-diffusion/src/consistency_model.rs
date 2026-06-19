//! # Consistency Model Utilities
//!
//! Implements Consistency Models (Song et al., 2023) for fast single-step and
//! few-step image generation and sampling.  Also includes Latent Consistency
//! Model (LCM) helpers (Luo et al., 2023).
//!
//! ## Key concepts
//!
//! - **EDM noise schedule** (`CmNoiseSchedule`): Karras et al. schedule for
//!   mapping discrete training steps to continuous σ values.
//! - **Consistency preconditioning** (`ConsistencyPreconditioning`): the
//!   c_skip / c_out / c_in / c_noise factors from the EDM paper that make the
//!   consistency function well-conditioned.
//! - **Pseudo-Huber loss** (`cm_pseudo_huber_loss`): the robust training
//!   objective used instead of plain MSE.
//! - **LCM** helpers: guided output, skipped-timestep sequences.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// xorshift64 PRNG helpers (no `rand` crate)
// ─────────────────────────────────────────────────────────────────────────────

fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

fn xorshift_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 11) as f32 / (1u64 << 53) as f32
}

fn cm_box_muller(u1: f32, u2: f32) -> (f32, f32) {
    let r = (-2.0_f32 * u1.max(1e-10).ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by consistency model operations.
///
/// Named `CmConsistencyError` to avoid collision with
/// `multi_view_consistency::ConsistencyError` which is already re-exported at
/// the crate root.
#[derive(Debug, Error)]
pub enum CmConsistencyError {
    /// Input/output slice lengths do not match.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// σ must be strictly positive and finite.
    #[error("Invalid sigma {0}: must be in (0, ∞)")]
    InvalidSigma(f32),

    /// σ_min must be strictly less than σ_max.
    #[error("Sigma_min {min} must be less than sigma_max {max}")]
    SigmaRangeError { min: f32, max: f32 },

    /// The number of inference/training steps must be at least 1.
    #[error("N steps {0} must be >= 1")]
    InvalidNSteps(usize),

    /// A configuration parameter is out of range or otherwise invalid.
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// CmNoiseSchedule
// ─────────────────────────────────────────────────────────────────────────────

/// EDM-based noise schedule for consistency models.
///
/// Implements the schedule from Karras et al. (2022) / Song et al. (2023):
///
/// ```text
/// σ_k = (σ_max^(1/ρ) + k/(K−1) · (σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ
/// ```
///
/// where `k` is the step index (0 = σ_max end, K−1 = σ_min end).
#[derive(Debug, Clone)]
pub struct CmNoiseSchedule {
    /// Minimum noise level (lower bound of the schedule).
    pub sigma_min: f32,
    /// Maximum noise level (upper bound of the schedule).
    pub sigma_max: f32,
    /// Schedule exponent ρ (default 7.0 for EDM).
    pub rho: f32,
    /// Total number of training steps K.
    pub n_training_steps: usize,
}

impl CmNoiseSchedule {
    /// Create a new schedule, validating all parameters.
    pub fn new(
        sigma_min: f32,
        sigma_max: f32,
        n_training_steps: usize,
    ) -> Result<Self, CmConsistencyError> {
        if sigma_min <= 0.0 || !sigma_min.is_finite() {
            return Err(CmConsistencyError::InvalidSigma(sigma_min));
        }
        if sigma_max <= 0.0 || !sigma_max.is_finite() {
            return Err(CmConsistencyError::InvalidSigma(sigma_max));
        }
        if sigma_min >= sigma_max {
            return Err(CmConsistencyError::SigmaRangeError {
                min: sigma_min,
                max: sigma_max,
            });
        }
        if n_training_steps < 1 {
            return Err(CmConsistencyError::InvalidNSteps(n_training_steps));
        }
        Ok(Self {
            sigma_min,
            sigma_max,
            rho: 7.0,
            n_training_steps,
        })
    }

    /// Default EDM schedule: σ_min=0.002, σ_max=80, ρ=7, K=150.
    pub fn default_edm() -> Self {
        Self {
            sigma_min: 0.002,
            sigma_max: 80.0,
            rho: 7.0,
            n_training_steps: 150,
        }
    }

    /// σ at discrete step `k` (0-indexed, 0 → σ_max, K-1 → σ_min).
    ///
    /// Uses the EDM formula:
    /// `σ_k = (σ_max^(1/ρ) + k/(K−1) · (σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ`
    pub fn sigma_at_step(&self, step: usize) -> f32 {
        let k = self.n_training_steps.saturating_sub(1).max(1);
        let inv_rho = 1.0 / self.rho;
        let sigma_max_rho = self.sigma_max.powf(inv_rho);
        let sigma_min_rho = self.sigma_min.powf(inv_rho);
        let t = step as f32 / k as f32;
        let interpolated = sigma_max_rho + t * (sigma_min_rho - sigma_max_rho);
        interpolated.powf(self.rho)
    }

    /// A sequence of `n+1` σ values decreasing from σ_max to σ_min.
    ///
    /// Suitable for driving multi-step inference loops.
    pub fn timestep_sigma_sequence(&self, n_steps: usize) -> Vec<f32> {
        if n_steps == 0 {
            return vec![self.sigma_max];
        }
        (0..=n_steps)
            .map(|i| {
                let inv_rho = 1.0 / self.rho;
                let sigma_max_rho = self.sigma_max.powf(inv_rho);
                let sigma_min_rho = self.sigma_min.powf(inv_rho);
                let t = i as f32 / n_steps as f32;
                let interpolated = sigma_max_rho + t * (sigma_min_rho - sigma_max_rho);
                interpolated.powf(self.rho)
            })
            .collect()
    }

    /// EMA decay μ at training step `k`.
    ///
    /// Starts near 0.95 at step 0 and asymptotically approaches 0.9999.
    ///
    /// `μ(k) = 0.9999 − (0.9999 − 0.95) · exp(−k / (K/10))`
    pub fn mu_at_step(&self, step: usize) -> f32 {
        let k_tenth = (self.n_training_steps as f32 / 10.0).max(1.0);
        let mu = 0.9999 - (0.9999 - 0.95) * (-(step as f32) / k_tenth).exp();
        mu.clamp(0.95, 0.9999)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConsistencySkip
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for spacing the inference timesteps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsistencySkip {
    /// Uniformly-spaced timesteps.
    Uniform,
    /// Exponentially-spaced timesteps (denser near σ_min).
    Exponential,
}

// ─────────────────────────────────────────────────────────────────────────────
// LcmLossWeight
// ─────────────────────────────────────────────────────────────────────────────

/// Loss weighting scheme for (Latent) Consistency Model training.
#[derive(Debug, Clone)]
pub enum LcmLossWeight {
    /// Equal weight for every timestep.
    Uniform,
    /// `w = SNR + 1` (from the LCM paper).
    SnrPlus1,
    /// `w = min(SNR, γ)` — Min-SNR weighting (Hang et al., 2023).
    MinSnr {
        /// Clipping threshold γ (typically 5.0).
        gamma: f32,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// CmConsistencyConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for consistency model training and inference.
///
/// Named `CmConsistencyConfig` to avoid collision with
/// `multi_view_consistency::ConsistencyConfig` which is already re-exported at
/// the crate root.
#[derive(Debug, Clone)]
pub struct CmConsistencyConfig {
    /// Noise schedule for the consistency model.
    pub schedule: CmNoiseSchedule,
    /// Number of inference steps (1 for single-step, more for multi-step).
    pub n_inference_steps: usize,
    /// Strategy for choosing which timesteps to sample at inference.
    pub skip_timesteps: ConsistencySkip,
    /// Loss weighting scheme.
    pub loss_weight: LcmLossWeight,
}

impl Default for CmConsistencyConfig {
    fn default() -> Self {
        Self {
            schedule: CmNoiseSchedule::default_edm(),
            n_inference_steps: 1,
            skip_timesteps: ConsistencySkip::Uniform,
            loss_weight: LcmLossWeight::SnrPlus1,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConsistencyPreconditioning
// ─────────────────────────────────────────────────────────────────────────────

/// EDM preconditioning factors for the consistency function.
///
/// Given a noisy sample `x_t` and noise level σ, the denoised estimate is:
///
/// `D(x_t, σ) = c_skip(σ) · x_t + c_out(σ) · F_θ(c_in(σ) · x_t, c_noise(σ))`
///
/// where `F_θ` is the raw neural network output.
#[derive(Debug, Clone)]
pub struct ConsistencyPreconditioning {
    /// Data standard deviation σ_data (default 0.5 for latent diffusion).
    pub sigma_data: f32,
}

impl ConsistencyPreconditioning {
    /// Create a new preconditioning with the given data standard deviation.
    pub fn new(sigma_data: f32) -> Self {
        Self { sigma_data }
    }

    /// Skip coefficient: `c_skip(σ) = σ_data² / (σ² + σ_data²)`.
    pub fn c_skip(&self, sigma: f32) -> f32 {
        let sd2 = self.sigma_data * self.sigma_data;
        sd2 / (sigma * sigma + sd2)
    }

    /// Output coefficient: `c_out(σ) = σ · σ_data / √(σ² + σ_data²)`.
    pub fn c_out(&self, sigma: f32) -> f32 {
        let sd2 = self.sigma_data * self.sigma_data;
        sigma * self.sigma_data / (sigma * sigma + sd2).sqrt()
    }

    /// Input scaling: `c_in(σ) = 1 / √(σ² + σ_data²)`.
    pub fn c_in(&self, sigma: f32) -> f32 {
        let sd2 = self.sigma_data * self.sigma_data;
        1.0 / (sigma * sigma + sd2).sqrt()
    }

    /// Noise conditioning: `c_noise(σ) = ln(σ) / 4`.
    pub fn c_noise(&self, sigma: f32) -> f32 {
        sigma.max(1e-10).ln() / 4.0
    }

    /// Apply EDM preconditioning to produce the denoised estimate.
    ///
    /// `result[i] = c_skip(σ) · noisy_input[i] + c_out(σ) · model_output[i]`
    pub fn apply_preconditioning(
        &self,
        model_output: &[f32],
        noisy_input: &[f32],
        sigma: f32,
    ) -> Result<Vec<f32>, CmConsistencyError> {
        if model_output.len() != noisy_input.len() {
            return Err(CmConsistencyError::DimensionMismatch {
                expected: noisy_input.len(),
                got: model_output.len(),
            });
        }
        let c_skip = self.c_skip(sigma);
        let c_out = self.c_out(sigma);
        let result = noisy_input
            .iter()
            .zip(model_output.iter())
            .map(|(&x, &f)| c_skip * x + c_out * f)
            .collect();
        Ok(result)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core free functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the consistency distillation loss weight λ for a pair (σ, σ_next).
///
/// `λ(σ, σ_next) = 1 / (σ - σ_next)` (unnormalised; clipped to avoid div/0).
pub fn cm_loss_weight(sigma: f32, sigma_next: f32, _schedule: &CmNoiseSchedule) -> f32 {
    let diff = (sigma - sigma_next).abs();
    if diff < 1e-8 {
        1.0
    } else {
        1.0 / diff
    }
}

/// Single-step consistency model inference.
///
/// Applies preconditioning to produce `x_0 ≈ f_θ(x_σmax, σ_max)`.
pub fn cm_single_step_inference(
    noisy: &[f32],
    model_output: &[f32],
    sigma_max: f32,
    precon: &ConsistencyPreconditioning,
) -> Result<Vec<f32>, CmConsistencyError> {
    if noisy.is_empty() {
        return Err(CmConsistencyError::InvalidParam(
            "noisy input must be non-empty".into(),
        ));
    }
    if model_output.len() != noisy.len() {
        return Err(CmConsistencyError::DimensionMismatch {
            expected: noisy.len(),
            got: model_output.len(),
        });
    }
    if sigma_max <= 0.0 || !sigma_max.is_finite() {
        return Err(CmConsistencyError::InvalidSigma(sigma_max));
    }
    precon.apply_preconditioning(model_output, noisy, sigma_max)
}

/// Multi-step consistency model inference.
///
/// Iterates through `sigma_sequence` (length N+1, decreasing), applying the
/// preconditioning and a forward Euler step between each pair of σ values.
///
/// `model_outputs` must have the same length as `sigma_sequence`.
pub fn cm_multi_step_inference(
    initial_sample: &[f32],
    sigma_sequence: &[f32],
    model_outputs: &[Vec<f32>],
    precon: &ConsistencyPreconditioning,
) -> Result<Vec<f32>, CmConsistencyError> {
    if sigma_sequence.is_empty() {
        return Err(CmConsistencyError::InvalidParam(
            "sigma_sequence must be non-empty".into(),
        ));
    }
    if model_outputs.len() != sigma_sequence.len() {
        return Err(CmConsistencyError::DimensionMismatch {
            expected: sigma_sequence.len(),
            got: model_outputs.len(),
        });
    }
    let n = initial_sample.len();
    // Verify all model outputs have the right length.
    for mo in model_outputs.iter() {
        if mo.len() != n {
            return Err(CmConsistencyError::DimensionMismatch {
                expected: n,
                got: mo.len(),
            });
        }
    }

    let mut x = initial_sample.to_vec();

    for (i, &sigma) in sigma_sequence.iter().enumerate() {
        let denoised = precon.apply_preconditioning(&model_outputs[i], &x, sigma)?;

        // Forward Euler step to the next sigma (if not the last step).
        if i + 1 < sigma_sequence.len() {
            let sigma_next = sigma_sequence[i + 1];
            // x_next = denoised + sigma_next * (x - denoised) / sigma
            // = denoised + (sigma_next / sigma) * (x - denoised)
            let ratio = if sigma.abs() < 1e-8 {
                0.0
            } else {
                sigma_next / sigma
            };
            x = denoised
                .iter()
                .zip(x.iter())
                .map(|(&d, &xi)| d + ratio * (xi - d))
                .collect();
        } else {
            x = denoised;
        }
    }

    Ok(x)
}

/// Add Gaussian noise at level σ: `x_σ = x_0 + σ · ε`.
pub fn cm_add_noise(x0: &[f32], noise: &[f32], sigma: f32) -> Result<Vec<f32>, CmConsistencyError> {
    if x0.len() != noise.len() {
        return Err(CmConsistencyError::DimensionMismatch {
            expected: x0.len(),
            got: noise.len(),
        });
    }
    if !sigma.is_finite() || sigma < 0.0 {
        return Err(CmConsistencyError::InvalidSigma(sigma));
    }
    Ok(x0
        .iter()
        .zip(noise.iter())
        .map(|(&x, &e)| x + sigma * e)
        .collect())
}

/// Sample standard-normal noise of length `n` using xorshift64 + Box-Muller.
pub fn cm_sample_noise(n: usize, state: &mut u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = xorshift_f32(state);
        let u2 = xorshift_f32(state);
        let (z0, z1) = cm_box_muller(u1, u2);
        out.push(z0);
        if out.len() < n {
            out.push(z1);
        }
    }
    out
}

/// Compute the consistency training target.
///
/// The target is the teacher model's denoised estimate at `sigma_next`:
///
/// `target[i] = c_skip(σ_next) · x_sigma_next[i] + c_out(σ_next) · model_output_next[i]`
pub fn cm_compute_target(
    x_sigma_next: &[f32],
    model_output_next: &[f32],
    _sigma: f32,
    sigma_next: f32,
    precon: &ConsistencyPreconditioning,
) -> Result<Vec<f32>, CmConsistencyError> {
    if x_sigma_next.len() != model_output_next.len() {
        return Err(CmConsistencyError::DimensionMismatch {
            expected: x_sigma_next.len(),
            got: model_output_next.len(),
        });
    }
    if sigma_next <= 0.0 || !sigma_next.is_finite() {
        return Err(CmConsistencyError::InvalidSigma(sigma_next));
    }
    precon.apply_preconditioning(model_output_next, x_sigma_next, sigma_next)
}

/// Pseudo-Huber loss (also called Charbonnier loss).
///
/// Per-element: `√((pred_i − target_i)² + c²) − c`
///
/// Mean-reduces over all elements.
pub fn cm_pseudo_huber_loss(
    predicted: &[f32],
    target: &[f32],
    c: f32,
) -> Result<f32, CmConsistencyError> {
    if predicted.len() != target.len() {
        return Err(CmConsistencyError::DimensionMismatch {
            expected: target.len(),
            got: predicted.len(),
        });
    }
    if predicted.is_empty() {
        return Ok(0.0);
    }
    if c <= 0.0 || !c.is_finite() {
        return Err(CmConsistencyError::InvalidParam(format!(
            "pseudo-Huber c must be > 0, got {c}"
        )));
    }
    let c2 = c * c;
    let sum: f32 = predicted
        .iter()
        .zip(target.iter())
        .map(|(&p, &t)| {
            let diff = p - t;
            (diff * diff + c2).sqrt() - c
        })
        .sum();
    Ok(sum / predicted.len() as f32)
}

/// Weighted consistency loss combining pseudo-Huber loss and the chosen
/// `LcmLossWeight` weighting scheme.
pub fn cm_weighted_loss(
    predicted: &[f32],
    target: &[f32],
    sigma: f32,
    sigma_next: f32,
    schedule: &CmNoiseSchedule,
    loss_weight: &LcmLossWeight,
) -> Result<f32, CmConsistencyError> {
    let base_loss = cm_pseudo_huber_loss(predicted, target, 0.1)?;
    let sd2 = 0.5_f32 * 0.5; // sigma_data = 0.5 (standard for latent diffusion)
    let snr = sd2 / (sigma * sigma).max(1e-10);

    let w = match loss_weight {
        LcmLossWeight::Uniform => 1.0_f32,
        LcmLossWeight::SnrPlus1 => snr + 1.0,
        LcmLossWeight::MinSnr { gamma } => snr.min(*gamma),
    };

    // Also scale by the schedule's cm_loss_weight for the step pair.
    let step_w = cm_loss_weight(sigma, sigma_next, schedule);

    Ok(base_loss * w * step_w)
}

// ─────────────────────────────────────────────────────────────────────────────
// LCM utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Latent Consistency Model (LCM) inference.
#[derive(Debug, Clone)]
pub struct LcmConfig {
    /// Underlying consistency model configuration.
    pub consistency_config: CmConsistencyConfig,
    /// Classifier-free guidance scale (default 7.5).
    pub guidance_scale: f32,
    /// Step-skipping factor (default 20): every 20th DDIM timestep is used.
    pub skipping_step: usize,
    /// Total DDIM steps in the teacher schedule (default 50).
    pub n_ddim_steps: usize,
}

impl Default for LcmConfig {
    fn default() -> Self {
        Self {
            consistency_config: CmConsistencyConfig::default(),
            guidance_scale: 7.5,
            skipping_step: 20,
            n_ddim_steps: 50,
        }
    }
}

/// LCM loss weight λ_t.
///
/// `λ(σ) = (σ² + σ_data²) / (σ · σ_data)²`
pub fn lcm_lambda(sigma: f32, sigma_data: f32) -> f32 {
    let sd2 = sigma_data * sigma_data;
    let s2 = sigma * sigma;
    (s2 + sd2) / ((sigma * sigma_data) * (sigma * sigma_data)).max(1e-20)
}

/// Augment the model output using classifier-free guidance.
///
/// `output[i] = uncond[i] + w · (cond[i] − uncond[i])`
///
/// - `w = 0` → unconditional output
/// - `w = 1` → conditional output
/// - `w > 1` → amplified guidance
pub fn lcm_guided_output(
    cond: &[f32],
    uncond: &[f32],
    w: f32,
) -> Result<Vec<f32>, CmConsistencyError> {
    if cond.len() != uncond.len() {
        return Err(CmConsistencyError::DimensionMismatch {
            expected: uncond.len(),
            got: cond.len(),
        });
    }
    Ok(cond
        .iter()
        .zip(uncond.iter())
        .map(|(&c, &u)| u + w * (c - u))
        .collect())
}

/// Compute the skipped timestep indices for LCM inference.
///
/// Takes `n_ddim_steps` uniformly-spaced steps and selects every
/// `skipping_step`-th index, keeping at most `n_lcm_steps` entries.
pub fn lcm_skipped_timesteps(
    n_ddim_steps: usize,
    n_lcm_steps: usize,
    skipping_step: usize,
) -> Vec<usize> {
    if n_ddim_steps == 0 || skipping_step == 0 {
        return Vec::new();
    }
    // Full DDIM schedule from T down to 0 (inclusive).
    let full: Vec<usize> = (0..=n_ddim_steps).rev().collect();
    // Select every skipping_step-th index (0-based position in `full`).
    full.into_iter()
        .enumerate()
        .filter(|(pos, _)| pos % skipping_step == 0)
        .map(|(_, t)| t)
        .take(n_lcm_steps)
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Human-readable summary of a `CmConsistencyConfig`.
pub fn format_cm_config(config: &CmConsistencyConfig) -> String {
    let skip = match config.skip_timesteps {
        ConsistencySkip::Uniform => "uniform",
        ConsistencySkip::Exponential => "exponential",
    };
    let weight = match &config.loss_weight {
        LcmLossWeight::Uniform => "uniform".to_string(),
        LcmLossWeight::SnrPlus1 => "snr+1".to_string(),
        LcmLossWeight::MinSnr { gamma } => format!("min_snr(γ={gamma:.2})"),
    };
    format!(
        "CmConsistencyConfig {{ sigma_min={:.4}, sigma_max={:.4}, rho={:.2}, \
         n_training={}, n_inference={}, skip={skip}, loss_weight={weight} }}",
        config.schedule.sigma_min,
        config.schedule.sigma_max,
        config.schedule.rho,
        config.schedule.n_training_steps,
        config.n_inference_steps,
    )
}

/// Human-readable one-line statistics for a consistency model run.
pub fn format_cm_stats(n_steps: usize, sigma_min: f32, sigma_max: f32, loss: f32) -> String {
    format!(
        "ConsistencyModel {{ steps={n_steps}, σ=[{sigma_min:.5}, {sigma_max:.3}], loss={loss:.6} }}"
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CmNoiseSchedule ────────────────────────────────────────────────────

    #[test]
    fn test_cm_noise_schedule_valid() {
        let s = CmNoiseSchedule::new(0.002, 80.0, 150).unwrap();
        assert_eq!(s.sigma_min, 0.002);
        assert_eq!(s.sigma_max, 80.0);
        assert_eq!(s.n_training_steps, 150);
    }

    #[test]
    fn test_cm_noise_schedule_sigma_min_ge_max_error() {
        assert!(matches!(
            CmNoiseSchedule::new(80.0, 0.002, 150),
            Err(CmConsistencyError::SigmaRangeError { .. })
        ));
    }

    #[test]
    fn test_cm_noise_schedule_equal_sigma_error() {
        assert!(matches!(
            CmNoiseSchedule::new(1.0, 1.0, 150),
            Err(CmConsistencyError::SigmaRangeError { .. })
        ));
    }

    #[test]
    fn test_cm_noise_schedule_zero_sigma_min_error() {
        assert!(matches!(
            CmNoiseSchedule::new(0.0, 80.0, 150),
            Err(CmConsistencyError::InvalidSigma(_))
        ));
    }

    #[test]
    fn test_cm_noise_schedule_zero_steps_error() {
        assert!(matches!(
            CmNoiseSchedule::new(0.002, 80.0, 0),
            Err(CmConsistencyError::InvalidNSteps(0))
        ));
    }

    #[test]
    fn test_sigma_at_step_zero_is_sigma_max() {
        let s = CmNoiseSchedule::default_edm();
        let sigma_0 = s.sigma_at_step(0);
        assert!(
            (sigma_0 - s.sigma_max).abs() < 1e-4,
            "step 0 should be sigma_max, got {sigma_0}"
        );
    }

    #[test]
    fn test_sigma_at_step_last_is_sigma_min() {
        let s = CmNoiseSchedule::default_edm();
        let k = s.n_training_steps - 1;
        let sigma_k = s.sigma_at_step(k);
        assert!(
            (sigma_k - s.sigma_min).abs() < 1e-4,
            "step K-1 should be sigma_min, got {sigma_k}"
        );
    }

    #[test]
    fn test_sigma_at_step_monotone_decreasing() {
        let s = CmNoiseSchedule::default_edm();
        let mut prev = s.sigma_at_step(0);
        for k in 1..s.n_training_steps {
            let cur = s.sigma_at_step(k);
            assert!(
                cur <= prev + 1e-6,
                "sigma not monotone at step {k}: {prev} -> {cur}"
            );
            prev = cur;
        }
    }

    #[test]
    fn test_mu_at_step_zero_near_0_95() {
        let s = CmNoiseSchedule::default_edm();
        let mu = s.mu_at_step(0);
        // At step 0, exp(-0) = 1, so mu = 0.9999 - 0.0499 = 0.95 exactly.
        assert!(
            (mu - 0.95).abs() < 1e-4,
            "mu at step 0 should be ~0.95, got {mu}"
        );
    }

    #[test]
    fn test_mu_at_step_large_near_0_9999() {
        let s = CmNoiseSchedule::default_edm();
        let mu = s.mu_at_step(s.n_training_steps * 100);
        assert!(
            mu >= 0.9998,
            "mu at large step should be near 0.9999, got {mu}"
        );
    }

    #[test]
    fn test_mu_at_step_monotone() {
        let s = CmNoiseSchedule::default_edm();
        let mut prev = s.mu_at_step(0);
        for k in 1..=s.n_training_steps {
            let cur = s.mu_at_step(k);
            assert!(cur >= prev - 1e-6, "mu not monotone at step {k}");
            prev = cur;
        }
    }

    #[test]
    fn test_timestep_sigma_sequence_length() {
        let s = CmNoiseSchedule::default_edm();
        let seq = s.timestep_sigma_sequence(10);
        assert_eq!(seq.len(), 11);
    }

    #[test]
    fn test_timestep_sigma_sequence_decreasing() {
        let s = CmNoiseSchedule::default_edm();
        let seq = s.timestep_sigma_sequence(20);
        for i in 1..seq.len() {
            assert!(
                seq[i] <= seq[i - 1] + 1e-5,
                "sequence not decreasing at index {i}"
            );
        }
    }

    #[test]
    fn test_timestep_sigma_sequence_zero_steps() {
        let s = CmNoiseSchedule::default_edm();
        let seq = s.timestep_sigma_sequence(0);
        assert_eq!(seq.len(), 1);
        assert!((seq[0] - s.sigma_max).abs() < 1e-4);
    }

    // ── ConsistencyPreconditioning ─────────────────────────────────────────

    #[test]
    fn test_c_skip_sigma_equals_sigma_data() {
        let p = ConsistencyPreconditioning::new(0.5);
        // c_skip(σ_data) = σ_data² / (σ_data² + σ_data²) = 0.5
        let v = p.c_skip(0.5);
        assert!(
            (v - 0.5).abs() < 1e-6,
            "c_skip at sigma_data should be 0.5, got {v}"
        );
    }

    #[test]
    fn test_c_out_sigma_zero_is_zero() {
        let p = ConsistencyPreconditioning::new(0.5);
        // c_out(0) = 0 * σ_data / sqrt(σ_data²) = 0
        let v = p.c_out(0.0);
        assert!(v.abs() < 1e-6, "c_out at sigma=0 should be 0, got {v}");
    }

    #[test]
    fn test_c_in_sigma_zero_is_one_over_sigma_data() {
        let p = ConsistencyPreconditioning::new(0.5);
        // c_in(0) = 1 / sqrt(σ_data²) = 1/σ_data = 2.0
        let v = p.c_in(0.0);
        let expected = 1.0 / 0.5;
        assert!(
            (v - expected).abs() < 1e-5,
            "c_in at sigma=0 should be {expected}, got {v}"
        );
    }

    #[test]
    fn test_c_noise() {
        let p = ConsistencyPreconditioning::new(0.5);
        // c_noise(e) = ln(e) / 4 = 1/4
        let v = p.c_noise(std::f32::consts::E);
        assert!(
            (v - 0.25).abs() < 1e-5,
            "c_noise(e) should be 0.25, got {v}"
        );
    }

    #[test]
    fn test_apply_preconditioning_zero_model_output() {
        let p = ConsistencyPreconditioning::new(0.5);
        let noisy = vec![1.0_f32, 2.0, 3.0];
        let model_out = vec![0.0_f32; 3];
        let sigma = 1.0;
        let result = p.apply_preconditioning(&model_out, &noisy, sigma).unwrap();
        let c_skip = p.c_skip(sigma);
        for (i, (&r, &x)) in result.iter().zip(noisy.iter()).enumerate() {
            let expected = c_skip * x;
            assert!(
                (r - expected).abs() < 1e-6,
                "element {i}: expected {expected}, got {r}"
            );
        }
    }

    #[test]
    fn test_apply_preconditioning_dimension_mismatch() {
        let p = ConsistencyPreconditioning::new(0.5);
        let result = p.apply_preconditioning(&[0.0; 3], &[0.0; 4], 1.0);
        assert!(matches!(
            result,
            Err(CmConsistencyError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_apply_preconditioning_nonzero_output() {
        let p = ConsistencyPreconditioning::new(0.5);
        let noisy = vec![0.0_f32; 4];
        let model_out = vec![1.0_f32; 4];
        let sigma = 0.5;
        let result = p.apply_preconditioning(&model_out, &noisy, sigma).unwrap();
        let c_out = p.c_out(sigma);
        for r in &result {
            assert!((r - c_out).abs() < 1e-6, "expected {c_out}, got {r}");
        }
    }

    // ── cm_add_noise ───────────────────────────────────────────────────────

    #[test]
    fn test_cm_add_noise_sigma_zero_unchanged() {
        let x0 = vec![1.0_f32, 2.0, 3.0];
        let noise = vec![5.0_f32, 5.0, 5.0];
        let result = cm_add_noise(&x0, &noise, 0.0).unwrap();
        for (r, x) in result.iter().zip(x0.iter()) {
            assert!((r - x).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cm_add_noise_large_sigma_dominated_by_noise() {
        let x0 = vec![0.0_f32; 4];
        let noise = vec![1.0_f32; 4];
        let sigma = 100.0;
        let result = cm_add_noise(&x0, &noise, sigma).unwrap();
        for r in &result {
            assert!((r - sigma).abs() < 1e-4);
        }
    }

    #[test]
    fn test_cm_add_noise_dimension_mismatch() {
        let result = cm_add_noise(&[0.0; 3], &[0.0; 4], 1.0);
        assert!(matches!(
            result,
            Err(CmConsistencyError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_cm_add_noise_negative_sigma_error() {
        let result = cm_add_noise(&[0.0; 3], &[0.0; 3], -1.0);
        assert!(matches!(result, Err(CmConsistencyError::InvalidSigma(_))));
    }

    // ── cm_sample_noise ────────────────────────────────────────────────────

    #[test]
    fn test_cm_sample_noise_length() {
        let mut state = 12345_u64;
        let noise = cm_sample_noise(100, &mut state);
        assert_eq!(noise.len(), 100);
    }

    #[test]
    fn test_cm_sample_noise_odd_length() {
        let mut state = 99999_u64;
        let noise = cm_sample_noise(7, &mut state);
        assert_eq!(noise.len(), 7);
    }

    #[test]
    fn test_cm_sample_noise_roughly_normal() {
        let mut state = 42_u64;
        let n = 1000;
        let noise = cm_sample_noise(n, &mut state);
        let mean = noise.iter().sum::<f32>() / n as f32;
        let var = noise.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        // Mean should be ~0, variance ~1.
        assert!(mean.abs() < 0.2, "mean too far from 0: {mean}");
        assert!((var - 1.0).abs() < 0.3, "variance too far from 1: {var}");
    }

    #[test]
    fn test_cm_sample_noise_zero_length() {
        let mut state = 1_u64;
        let noise = cm_sample_noise(0, &mut state);
        assert!(noise.is_empty());
    }

    // ── cm_single_step_inference ───────────────────────────────────────────

    #[test]
    fn test_cm_single_step_inference_valid() {
        let precon = ConsistencyPreconditioning::new(0.5);
        let noisy = vec![0.5_f32; 8];
        let model_out = vec![0.1_f32; 8];
        let result = cm_single_step_inference(&noisy, &model_out, 80.0, &precon).unwrap();
        assert_eq!(result.len(), 8);
        for r in &result {
            assert!(r.is_finite());
        }
    }

    #[test]
    fn test_cm_single_step_inference_dimension_mismatch() {
        let precon = ConsistencyPreconditioning::new(0.5);
        let result = cm_single_step_inference(&[0.0; 4], &[0.0; 3], 80.0, &precon);
        assert!(matches!(
            result,
            Err(CmConsistencyError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_cm_single_step_inference_invalid_sigma() {
        let precon = ConsistencyPreconditioning::new(0.5);
        let result = cm_single_step_inference(&[0.0; 4], &[0.0; 4], -1.0, &precon);
        assert!(matches!(result, Err(CmConsistencyError::InvalidSigma(_))));
    }

    #[test]
    fn test_cm_single_step_inference_empty_error() {
        let precon = ConsistencyPreconditioning::new(0.5);
        let result = cm_single_step_inference(&[], &[], 1.0, &precon);
        assert!(matches!(result, Err(CmConsistencyError::InvalidParam(_))));
    }

    // ── cm_multi_step_inference ────────────────────────────────────────────

    #[test]
    fn test_cm_multi_step_inference_single_step() {
        let precon = ConsistencyPreconditioning::new(0.5);
        let initial = vec![1.0_f32; 4];
        let sigmas = vec![80.0_f32];
        let model_outs = vec![vec![0.0_f32; 4]];
        let result = cm_multi_step_inference(&initial, &sigmas, &model_outs, &precon).unwrap();
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_cm_multi_step_inference_multiple_steps() {
        let precon = ConsistencyPreconditioning::new(0.5);
        let sched = CmNoiseSchedule::default_edm();
        let sigmas = sched.timestep_sigma_sequence(4);
        let n = 8;
        let model_outs: Vec<Vec<f32>> = sigmas.iter().map(|_| vec![0.1_f32; n]).collect();
        let initial = vec![0.5_f32; n];
        let result = cm_multi_step_inference(&initial, &sigmas, &model_outs, &precon).unwrap();
        assert_eq!(result.len(), n);
        for r in &result {
            assert!(r.is_finite());
        }
    }

    #[test]
    fn test_cm_multi_step_inference_empty_sigma_error() {
        let precon = ConsistencyPreconditioning::new(0.5);
        let result = cm_multi_step_inference(&[0.0; 4], &[], &[], &precon);
        assert!(matches!(result, Err(CmConsistencyError::InvalidParam(_))));
    }

    #[test]
    fn test_cm_multi_step_inference_length_mismatch() {
        let precon = ConsistencyPreconditioning::new(0.5);
        let sigmas = vec![80.0_f32, 40.0, 20.0];
        let model_outs = vec![vec![0.0_f32; 4]; 2]; // wrong length
        let result = cm_multi_step_inference(&[0.0; 4], &sigmas, &model_outs, &precon);
        assert!(matches!(
            result,
            Err(CmConsistencyError::DimensionMismatch { .. })
        ));
    }

    // ── cm_compute_target ─────────────────────────────────────────────────

    #[test]
    fn test_cm_compute_target_known_inputs() {
        let precon = ConsistencyPreconditioning::new(0.5);
        let x_next = vec![1.0_f32; 4];
        let model_out = vec![0.0_f32; 4]; // zero model output
                                          // target = c_skip(sigma_next)*x_next + c_out(sigma_next)*model_out
        let sigma = 2.0;
        let sigma_next = 1.0;
        let result = cm_compute_target(&x_next, &model_out, sigma, sigma_next, &precon).unwrap();
        let c_skip = precon.c_skip(sigma_next);
        let expected = c_skip; // x_next=1, model_out=0
        for r in &result {
            assert!((r - expected).abs() < 1e-5, "expected {expected}, got {r}");
        }
    }

    #[test]
    fn test_cm_compute_target_dimension_mismatch() {
        let precon = ConsistencyPreconditioning::new(0.5);
        let result = cm_compute_target(&[0.0; 4], &[0.0; 3], 2.0, 1.0, &precon);
        assert!(matches!(
            result,
            Err(CmConsistencyError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_cm_compute_target_invalid_sigma_next() {
        let precon = ConsistencyPreconditioning::new(0.5);
        let result = cm_compute_target(&[0.0; 4], &[0.0; 4], 2.0, 0.0, &precon);
        assert!(matches!(result, Err(CmConsistencyError::InvalidSigma(_))));
    }

    // ── cm_pseudo_huber_loss ───────────────────────────────────────────────

    #[test]
    fn test_pseudo_huber_loss_same_pred_target_is_zero() {
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let loss = cm_pseudo_huber_loss(&x, &x, 0.1).unwrap();
        assert!(
            loss.abs() < 1e-6,
            "same pred/target should give 0, got {loss}"
        );
    }

    #[test]
    fn test_pseudo_huber_loss_large_diff_positive() {
        let pred = vec![10.0_f32; 4];
        let target = vec![0.0_f32; 4];
        let loss = cm_pseudo_huber_loss(&pred, &target, 0.1).unwrap();
        assert!(loss > 0.0, "loss should be positive for large diff");
    }

    #[test]
    fn test_pseudo_huber_loss_dimension_mismatch() {
        let result = cm_pseudo_huber_loss(&[0.0; 3], &[0.0; 4], 0.1);
        assert!(matches!(
            result,
            Err(CmConsistencyError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_pseudo_huber_loss_empty_is_zero() {
        let loss = cm_pseudo_huber_loss(&[], &[], 0.1).unwrap();
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_pseudo_huber_loss_invalid_c() {
        let result = cm_pseudo_huber_loss(&[1.0; 4], &[0.0; 4], 0.0);
        assert!(matches!(result, Err(CmConsistencyError::InvalidParam(_))));
    }

    #[test]
    fn test_pseudo_huber_loss_known_value() {
        // √((2-0)² + 1²) - 1 = √5 - 1 ≈ 1.2361
        let pred = vec![2.0_f32];
        let target = vec![0.0_f32];
        let loss = cm_pseudo_huber_loss(&pred, &target, 1.0).unwrap();
        let expected = (5.0_f32).sqrt() - 1.0;
        assert!(
            (loss - expected).abs() < 1e-5,
            "expected {expected}, got {loss}"
        );
    }

    // ── cm_weighted_loss ───────────────────────────────────────────────────

    #[test]
    fn test_cm_weighted_loss_uniform() {
        let sched = CmNoiseSchedule::default_edm();
        let pred = vec![1.0_f32; 4];
        let target = vec![0.0_f32; 4];
        let loss =
            cm_weighted_loss(&pred, &target, 2.0, 1.0, &sched, &LcmLossWeight::Uniform).unwrap();
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_cm_weighted_loss_min_snr() {
        let sched = CmNoiseSchedule::default_edm();
        let pred = vec![1.0_f32; 4];
        let target = vec![0.0_f32; 4];
        let loss = cm_weighted_loss(
            &pred,
            &target,
            2.0,
            1.0,
            &sched,
            &LcmLossWeight::MinSnr { gamma: 5.0 },
        )
        .unwrap();
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_cm_weighted_loss_snr_plus1() {
        let sched = CmNoiseSchedule::default_edm();
        let pred = vec![1.0_f32; 4];
        let target = vec![0.0_f32; 4];
        let loss =
            cm_weighted_loss(&pred, &target, 2.0, 1.0, &sched, &LcmLossWeight::SnrPlus1).unwrap();
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    // ── cm_loss_weight ─────────────────────────────────────────────────────

    #[test]
    fn test_cm_loss_weight_positive() {
        let sched = CmNoiseSchedule::default_edm();
        let w = cm_loss_weight(2.0, 1.0, &sched);
        assert!(w > 0.0, "cm_loss_weight should be positive, got {w}");
    }

    #[test]
    fn test_cm_loss_weight_equal_sigmas_returns_one() {
        let sched = CmNoiseSchedule::default_edm();
        let w = cm_loss_weight(1.0, 1.0, &sched);
        assert_eq!(w, 1.0, "equal sigmas should return 1.0 sentinel");
    }

    // ── lcm_lambda ─────────────────────────────────────────────────────────

    #[test]
    fn test_lcm_lambda_sigma_equals_sigma_data() {
        // λ(σ_data, σ_data) = (σ_data² + σ_data²) / (σ_data²)² = 2/σ_data²
        let sd = 0.5_f32;
        let v = lcm_lambda(sd, sd);
        let expected = 2.0 / (sd * sd);
        assert!((v - expected).abs() < 1e-4, "expected {expected}, got {v}");
    }

    #[test]
    fn test_lcm_lambda_positive() {
        let v = lcm_lambda(1.0, 0.5);
        assert!(v > 0.0, "lcm_lambda should be positive, got {v}");
    }

    // ── lcm_guided_output ──────────────────────────────────────────────────

    #[test]
    fn test_lcm_guided_w0_is_uncond() {
        let cond = vec![1.0_f32; 4];
        let uncond = vec![2.0_f32; 4];
        let result = lcm_guided_output(&cond, &uncond, 0.0).unwrap();
        for (r, u) in result.iter().zip(uncond.iter()) {
            assert!((r - u).abs() < 1e-6, "w=0 should return uncond");
        }
    }

    #[test]
    fn test_lcm_guided_w1_is_cond() {
        let cond = vec![3.0_f32; 4];
        let uncond = vec![1.0_f32; 4];
        let result = lcm_guided_output(&cond, &uncond, 1.0).unwrap();
        for (r, c) in result.iter().zip(cond.iter()) {
            assert!((r - c).abs() < 1e-6, "w=1 should return cond");
        }
    }

    #[test]
    fn test_lcm_guided_midpoint() {
        let cond = vec![4.0_f32; 4];
        let uncond = vec![2.0_f32; 4];
        // w=0.5: uncond + 0.5*(cond - uncond) = 3.0
        let result = lcm_guided_output(&cond, &uncond, 0.5).unwrap();
        for r in &result {
            assert!(
                (r - 3.0).abs() < 1e-6,
                "w=0.5 should give midpoint 3.0, got {r}"
            );
        }
    }

    #[test]
    fn test_lcm_guided_dimension_mismatch() {
        let result = lcm_guided_output(&[0.0; 3], &[0.0; 4], 1.0);
        assert!(matches!(
            result,
            Err(CmConsistencyError::DimensionMismatch { .. })
        ));
    }

    // ── lcm_skipped_timesteps ──────────────────────────────────────────────

    #[test]
    fn test_lcm_skipped_timesteps_correct_count() {
        // With 50 DDIM steps, skipping_step=20, n_lcm_steps=4:
        // Positions 0, 20, 40, 60, 80, 100 in [50..=0] reversed
        // i.e. t = 50, 30, 10 → 3 values (positions 0,20,40 exist; pos 60+ don't)
        let ts = lcm_skipped_timesteps(50, 4, 20);
        assert!(
            ts.len() <= 4,
            "should have at most 4 steps, got {}",
            ts.len()
        );
    }

    #[test]
    fn test_lcm_skipped_timesteps_zero_ddim_steps() {
        let ts = lcm_skipped_timesteps(0, 4, 20);
        assert!(ts.is_empty());
    }

    #[test]
    fn test_lcm_skipped_timesteps_skip_1_all_steps() {
        // With skip=1, every step is included; limit to n_lcm_steps.
        let ts = lcm_skipped_timesteps(10, 5, 1);
        assert_eq!(ts.len(), 5);
    }

    #[test]
    fn test_lcm_skipped_timesteps_zero_skipping_step() {
        let ts = lcm_skipped_timesteps(50, 4, 0);
        assert!(ts.is_empty());
    }

    // ── format helpers ─────────────────────────────────────────────────────

    #[test]
    fn test_format_cm_config_non_empty() {
        let config = CmConsistencyConfig::default();
        let s = format_cm_config(&config);
        assert!(!s.is_empty());
        assert!(s.contains("sigma_min"));
    }

    #[test]
    fn test_format_cm_stats_non_empty() {
        let s = format_cm_stats(10, 0.002, 80.0, 0.0123);
        assert!(!s.is_empty());
        assert!(s.contains("steps=10"));
    }

    #[test]
    fn test_format_cm_config_min_snr_weight() {
        let config = CmConsistencyConfig {
            loss_weight: LcmLossWeight::MinSnr { gamma: 5.0 },
            ..Default::default()
        };
        let s = format_cm_config(&config);
        assert!(s.contains("min_snr"), "expected min_snr in '{s}'");
    }

    // ── edge cases ─────────────────────────────────────────────────────────

    #[test]
    fn test_cm_add_noise_empty_slices() {
        let result = cm_add_noise(&[], &[], 1.0).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_cm_compute_target_empty_slices() {
        let precon = ConsistencyPreconditioning::new(0.5);
        let result = cm_compute_target(&[], &[], 2.0, 1.0, &precon).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_default_edm_schedule_boundaries() {
        let s = CmNoiseSchedule::default_edm();
        assert_eq!(s.sigma_min, 0.002);
        assert_eq!(s.sigma_max, 80.0);
        assert_eq!(s.rho, 7.0);
        assert_eq!(s.n_training_steps, 150);
    }

    #[test]
    fn test_default_cm_consistency_config() {
        let c = CmConsistencyConfig::default();
        assert_eq!(c.n_inference_steps, 1);
        assert_eq!(c.skip_timesteps, ConsistencySkip::Uniform);
    }

    #[test]
    fn test_lcm_config_default() {
        let c = LcmConfig::default();
        assert!((c.guidance_scale - 7.5).abs() < 1e-6);
        assert_eq!(c.skipping_step, 20);
        assert_eq!(c.n_ddim_steps, 50);
    }
}
