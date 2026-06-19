//! Multi-step diffusion samplers: DDIM, PLMS, and DPM++ (2M).
//!
//! Implements efficient inference-time samplers that reduce the number of
//! neural function evaluations from 1000 (full DDPM training schedule) to
//! as few as 10–50 steps with minimal quality loss.
//!
//! ## Samplers
//!
//! - **DDIM** (Denoising Diffusion Implicit Models, Song et al. 2020):
//!   Deterministic sampler (η=0) or stochastic (η=1 ≈ DDPM). Fast, stable.
//! - **PLMS** (Pseudo Linear Multi-Step, Liu et al. 2022):
//!   Adams-Bashforth 4th-order multi-step method. Better quality per step.
//! - **DPM++ 2M** (Lu et al. 2022):
//!   2nd-order multistep in log-SNR space. State-of-the-art efficiency.
//!
//! ## Example
//!
//! ```
//! use oxigaf_diffusion::multi_step_sampler::{
//!     SamplingNoiseSchedule, MultiStepSamplerConfig, MultiStepSampler, SamplerKind,
//! };
//!
//! let schedule = SamplingNoiseSchedule::cosine(1000);
//! let config = MultiStepSamplerConfig {
//!     kind: SamplerKind::Ddim { eta: 0.0 },
//!     n_inference_steps: 20,
//!     schedule,
//!     guidance_scale: 7.5,
//! };
//! let mut sampler = MultiStepSampler::new(config).unwrap();
//! sampler.set_timesteps().unwrap();
//! assert_eq!(sampler.total_steps(), 20);
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur in multi-step diffusion sampling.
#[derive(Debug, Error)]
pub enum SamplerError {
    /// Input tensor dimension does not match expected size.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Sampler has not been initialized via [`MultiStepSampler::set_timesteps`].
    #[error("Sampler not initialized: call set_timesteps first")]
    NotInitialized,

    /// All inference steps have been consumed.
    #[error("No more steps: sampling complete")]
    SamplingComplete,

    /// A configuration parameter is outside its valid range.
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),

    /// Not enough previous noise predictions to use the requested order.
    #[error("History too short for multi-step (need {need}, have {have})")]
    InsufficientHistory { need: usize, have: usize },
}

// ---------------------------------------------------------------------------
// SamplingNoiseSchedule
// ---------------------------------------------------------------------------

/// Discretized noise schedule used by the multi-step samplers.
///
/// Distinct from `noise_schedule_analysis::NoiseSchedule` which carries
/// full analysis metadata.  This struct stores only what the samplers need:
/// `alpha_bar[t]` and `sigma[t]`.
#[derive(Debug, Clone)]
pub struct SamplingNoiseSchedule {
    /// Total number of training timesteps.
    pub n_timesteps: usize,
    /// Cumulative product of (1 − β_t), indexed 0..n_timesteps.
    pub alpha_bars: Vec<f32>,
    /// σ_t = √(1 − ᾱ_t), indexed 0..n_timesteps.
    pub sigmas: Vec<f32>,
}

impl SamplingNoiseSchedule {
    /// Build a cosine schedule (Nichol & Dhariwal 2021).
    ///
    /// ```text
    /// s = 0.008
    /// ᾱ_t = cos²((t/T + s)/(1+s) · π/2) / cos²(s/(1+s) · π/2)
    /// ```
    /// Values are clamped to `[0.0001, 0.9999]`.
    pub fn cosine(n_timesteps: usize) -> Self {
        let s = 0.008_f32;
        let normalizer = {
            let frac = s / (1.0 + s);
            (frac * std::f32::consts::FRAC_PI_2).cos().powi(2)
        };

        let alpha_bars: Vec<f32> = (0..n_timesteps)
            .map(|t| {
                let frac = (t as f32 / n_timesteps as f32 + s) / (1.0 + s);
                let raw = (frac * std::f32::consts::FRAC_PI_2).cos().powi(2) / normalizer;
                raw.clamp(0.0001, 0.9999)
            })
            .collect();

        let sigmas = alpha_bars.iter().map(|&ab| (1.0 - ab).sqrt()).collect();

        Self {
            n_timesteps,
            alpha_bars,
            sigmas,
        }
    }

    /// Build a linear beta schedule.
    ///
    /// β_t is linearly spaced from `beta_start` to `beta_end`.
    /// `ᾱ_t = ∏_{i≤t}(1 - β_i)`.
    pub fn linear(n_timesteps: usize, beta_start: f32, beta_end: f32) -> Self {
        let mut alpha_bars = Vec::with_capacity(n_timesteps);
        let mut cumprod = 1.0_f32;
        for i in 0..n_timesteps {
            let beta = beta_start
                + (beta_end - beta_start) * (i as f32) / (n_timesteps.max(1) as f32 - 1.0).max(1.0);
            let alpha = 1.0 - beta;
            cumprod *= alpha;
            alpha_bars.push(cumprod.clamp(0.0001, 0.9999));
        }
        let sigmas = alpha_bars.iter().map(|&ab| (1.0 - ab).sqrt()).collect();
        Self {
            n_timesteps,
            alpha_bars,
            sigmas,
        }
    }

    /// Return ᾱ_t.  Clamps `t` to the valid range.
    #[inline]
    pub fn alpha_bar_at(&self, t: usize) -> f32 {
        let idx = t.min(self.n_timesteps.saturating_sub(1));
        self.alpha_bars[idx]
    }

    /// Return σ_t = √(1 − ᾱ_t).  Clamps `t` to the valid range.
    #[inline]
    pub fn sigma_at(&self, t: usize) -> f32 {
        let idx = t.min(self.n_timesteps.saturating_sub(1));
        self.sigmas[idx]
    }

    /// Signal-to-noise ratio: SNR_t = ᾱ_t / (1 − ᾱ_t).
    #[inline]
    pub fn snr_at(&self, t: usize) -> f32 {
        let ab = self.alpha_bar_at(t);
        ab / (1.0 - ab).max(f32::EPSILON)
    }
}

// ---------------------------------------------------------------------------
// SamplerKind
// ---------------------------------------------------------------------------

/// Which multi-step sampler algorithm to use.
#[derive(Debug, Clone)]
pub enum SamplerKind {
    /// Denoising Diffusion Implicit Models (Song et al. 2020).
    ///
    /// `eta = 0.0` → fully deterministic.
    /// `eta = 1.0` → DDPM-equivalent stochasticity.
    Ddim { eta: f32 },

    /// Pseudo Linear Multi-Step (Liu et al. 2022).
    ///
    /// 4th-order Adams-Bashforth update; accumulates up to 3 previous
    /// noise predictions in `MultiStepSampler::history`.
    Plms,

    /// DPM++ 2nd-order multistep (Lu et al. 2022).
    ///
    /// Operates in log-SNR space; uses one previous noise prediction.
    DpmPlusPlus2M,
}

// ---------------------------------------------------------------------------
// MultiStepSamplerConfig
// ---------------------------------------------------------------------------

/// Full configuration for a multi-step sampler.
#[derive(Debug, Clone)]
pub struct MultiStepSamplerConfig {
    /// Which sampling algorithm to use.
    pub kind: SamplerKind,
    /// Number of denoising steps at inference time (≪ `schedule.n_timesteps`).
    pub n_inference_steps: usize,
    /// Noise schedule that defines ᾱ_t for all training timesteps.
    pub schedule: SamplingNoiseSchedule,
    /// Classifier-free guidance scale (w).
    pub guidance_scale: f32,
}

impl Default for MultiStepSamplerConfig {
    fn default() -> Self {
        Self {
            kind: SamplerKind::Ddim { eta: 0.0 },
            n_inference_steps: 50,
            schedule: SamplingNoiseSchedule::cosine(1000),
            guidance_scale: 7.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Timestep schedule helper
// ---------------------------------------------------------------------------

/// Compute the inference timestep schedule.
///
/// Returns `n_steps + 1` timesteps linearly spaced from `n_total − 1` down
/// to `0` (inclusive on both ends), deduplicated and sorted in descending
/// order.  Typically `n_total = 1000` (training steps) and `n_steps = 50`.
///
/// The schedule includes both the starting noise level (`n_total − 1`) and
/// the final clean step (`0`), giving `n_steps + 1` boundary values from
/// which `n_steps` update intervals are derived.
pub fn compute_timestep_schedule(n_total: usize, n_steps: usize) -> Vec<usize> {
    if n_total == 0 || n_steps == 0 {
        return vec![0];
    }

    let max_t = n_total - 1;
    let mut ts: Vec<usize> = (0..=n_steps)
        .map(|i| {
            let frac = 1.0 - (i as f64) / (n_steps as f64);
            let raw = (max_t as f64) * frac;
            raw.round() as usize
        })
        .collect();

    // Deduplicate while preserving order (descending)
    ts.sort_unstable_by(|a, b| b.cmp(a));
    ts.dedup();

    ts
}

// ---------------------------------------------------------------------------
// Core free functions
// ---------------------------------------------------------------------------

/// Predict the clean image x₀ from a noisy sample xₜ and noise prediction.
///
/// Formula:
/// ```text
/// x₀_pred = (xₜ − √(1 − ᾱ_t) · ε) / √ᾱ_t
/// ```
pub fn predict_x0(
    sample: &[f32],
    noise_pred: &[f32],
    t: usize,
    schedule: &SamplingNoiseSchedule,
) -> Result<Vec<f32>, SamplerError> {
    if sample.len() != noise_pred.len() {
        return Err(SamplerError::DimensionMismatch {
            expected: sample.len(),
            got: noise_pred.len(),
        });
    }
    let alpha_bar = schedule.alpha_bar_at(t);
    let sqrt_ab = alpha_bar.sqrt();
    let sqrt_one_minus_ab = (1.0 - alpha_bar).sqrt();

    let inv_sqrt_ab = if sqrt_ab.abs() < f32::EPSILON {
        0.0
    } else {
        1.0 / sqrt_ab
    };

    let x0: Vec<f32> = sample
        .iter()
        .zip(noise_pred.iter())
        .map(|(&x, &e)| (x - sqrt_one_minus_ab * e) * inv_sqrt_ab)
        .collect();
    Ok(x0)
}

/// Apply classifier-free guidance (CFG) to combine conditional and unconditional
/// noise predictions.
///
/// Formula:
/// ```text
/// output = uncond + scale · (cond − uncond)
/// ```
///
/// This is distinct from `cfg_guidance::apply_cfg`; use `sampler_apply_cfg`
/// when working within the multi-step sampler module to avoid ambiguity.
pub fn sampler_apply_cfg(
    noise_pred_cond: &[f32],
    noise_pred_uncond: &[f32],
    guidance_scale: f32,
) -> Result<Vec<f32>, SamplerError> {
    if noise_pred_cond.len() != noise_pred_uncond.len() {
        return Err(SamplerError::DimensionMismatch {
            expected: noise_pred_cond.len(),
            got: noise_pred_uncond.len(),
        });
    }
    if guidance_scale < 0.0 {
        return Err(SamplerError::InvalidParam(format!(
            "guidance_scale must be ≥ 0, got {guidance_scale}"
        )));
    }
    let out = noise_pred_uncond
        .iter()
        .zip(noise_pred_cond.iter())
        .map(|(&u, &c)| u + guidance_scale * (c - u))
        .collect();
    Ok(out)
}

// ---------------------------------------------------------------------------
// DDIM step
// ---------------------------------------------------------------------------

/// One DDIM denoising step.
///
/// Computes xₜ₋₁ given the noise prediction εθ(xₜ, t):
///
/// ```text
/// x₀_pred  = (xₜ − √(1−ᾱ_t)·ε) / √ᾱ_t
/// σ_DDIM   = η · √((1−ᾱ_{t-1})/(1−ᾱ_t)) · √(1 − ᾱ_t/ᾱ_{t-1})
/// dir_xt   = √(1−ᾱ_{t-1} − σ²) · ε
/// xₜ₋₁    = √ᾱ_{t-1}·x₀_pred + dir_xt  [+ σ·noise if η>0]
/// ```
pub fn ddim_step(
    sample: &[f32],
    noise_pred: &[f32],
    t: usize,
    t_prev: usize,
    schedule: &SamplingNoiseSchedule,
    eta: f32,
    noise: Option<&[f32]>,
) -> Result<Vec<f32>, SamplerError> {
    if sample.len() != noise_pred.len() {
        return Err(SamplerError::DimensionMismatch {
            expected: sample.len(),
            got: noise_pred.len(),
        });
    }
    if let Some(n) = noise {
        if n.len() != sample.len() {
            return Err(SamplerError::DimensionMismatch {
                expected: sample.len(),
                got: n.len(),
            });
        }
    }

    let alpha_bar_t = schedule.alpha_bar_at(t);
    // For the very first "previous" step, ᾱ = 1.0 (fully clean)
    let alpha_bar_t_prev = if t_prev > 0 {
        schedule.alpha_bar_at(t_prev)
    } else {
        1.0_f32
    };

    let sqrt_ab_t = alpha_bar_t.sqrt();
    let sqrt_one_minus_ab_t = (1.0 - alpha_bar_t).sqrt();

    // Predict x0
    let inv_sqrt_ab_t = if sqrt_ab_t.abs() < f32::EPSILON {
        0.0
    } else {
        1.0 / sqrt_ab_t
    };

    // σ_DDIM: stochastic sigma when η > 0
    let ratio = if alpha_bar_t_prev > f32::EPSILON {
        1.0 - alpha_bar_t / alpha_bar_t_prev
    } else {
        0.0
    };
    let sigma_t = if eta > 0.0 {
        eta * ((1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t).max(f32::EPSILON)).sqrt()
            * ratio.max(0.0).sqrt()
    } else {
        0.0
    };

    // √(1 - ᾱ_{t-1} - σ²) — coefficient for direction pointing at xₜ
    let dir_coeff = {
        let inner = (1.0 - alpha_bar_t_prev - sigma_t * sigma_t).max(0.0);
        inner.sqrt()
    };
    let sqrt_ab_t_prev = alpha_bar_t_prev.sqrt();

    let mut x_prev = Vec::with_capacity(sample.len());
    for i in 0..sample.len() {
        let x0_pred = (sample[i] - sqrt_one_minus_ab_t * noise_pred[i]) * inv_sqrt_ab_t;
        let dir_xt = dir_coeff * noise_pred[i];
        let val = sqrt_ab_t_prev * x0_pred + dir_xt;
        x_prev.push(val);
    }

    // Add stochastic noise if requested
    if eta > 0.0 && sigma_t > 0.0 {
        if let Some(n) = noise {
            for (v, &ni) in x_prev.iter_mut().zip(n.iter()) {
                *v += sigma_t * ni;
            }
        }
    }

    Ok(x_prev)
}

// ---------------------------------------------------------------------------
// PLMS step (Adams-Bashforth multi-step)
// ---------------------------------------------------------------------------

/// One PLMS denoising step.
///
/// Uses the pseudo linear multi-step (Adams-Bashforth) method.
/// The effective noise estimate is blended from the current and up to 3
/// previous predictions, yielding up to 4th-order accuracy.
///
/// | `history.len()` | Order | Coefficients (current, h[-1], h[-2], h[-3]) |
/// |-----------------|-------|---------------------------------------------|
/// | 0               | 1     | 1                                           |
/// | 1               | 2     | 3/2, −1/2                                   |
/// | 2               | 3     | 23/12, −16/12, 5/12                         |
/// | ≥3              | 4     | 55/24, −59/24, 37/24, −9/24                 |
///
/// After computing the blended ε, applies a DDIM-style step with η=0.
pub fn plms_step(
    sample: &[f32],
    noise_pred: &[f32],
    history: &[Vec<f32>],
    t: usize,
    t_prev: usize,
    schedule: &SamplingNoiseSchedule,
) -> Result<Vec<f32>, SamplerError> {
    if sample.len() != noise_pred.len() {
        return Err(SamplerError::DimensionMismatch {
            expected: sample.len(),
            got: noise_pred.len(),
        });
    }
    for (k, prev) in history.iter().enumerate() {
        if prev.len() != sample.len() {
            return Err(SamplerError::DimensionMismatch {
                expected: sample.len(),
                got: prev.len(),
            });
        }
        let _ = k; // suppress unused variable
    }

    let order = (history.len() + 1).min(4);
    let len = sample.len();

    // Compute blended ε using Adams-Bashforth coefficients
    let blended: Vec<f32> = match order {
        1 => noise_pred.to_vec(),
        2 => {
            let h0 = &history[history.len() - 1];
            (0..len)
                .map(|i| (3.0 * noise_pred[i] - h0[i]) / 2.0)
                .collect()
        }
        3 => {
            let h0 = &history[history.len() - 1]; // k-1
            let h1 = &history[history.len() - 2]; // k-2
            (0..len)
                .map(|i| (23.0 * noise_pred[i] - 16.0 * h0[i] + 5.0 * h1[i]) / 12.0)
                .collect()
        }
        _ => {
            // order == 4
            let h0 = &history[history.len() - 1];
            let h1 = &history[history.len() - 2];
            let h2 = &history[history.len() - 3];
            (0..len)
                .map(|i| (55.0 * noise_pred[i] - 59.0 * h0[i] + 37.0 * h1[i] - 9.0 * h2[i]) / 24.0)
                .collect()
        }
    };

    // Apply a deterministic DDIM step (η=0) with the blended ε
    ddim_step(sample, &blended, t, t_prev, schedule, 0.0, None)
}

// ---------------------------------------------------------------------------
// DPM++ 2M step
// ---------------------------------------------------------------------------

/// One DPM++ 2nd-order multistep denoising step.
///
/// Operates in log-SNR space:
/// `λ_t = log(ᾱ_t / σ_t)`
///
/// **First step** (no previous prediction): uses 1st-order exponential
/// integrator (equivalent to DDIM η=0).
///
/// **Subsequent steps**: blends current and previous noise predictions via
/// the 2M (2nd-order multistep) correction term.
pub fn dpm_plus_plus_2m_step(
    sample: &[f32],
    noise_pred: &[f32],
    prev_noise_pred: Option<&[f32]>,
    t: usize,
    t_prev: usize,
    schedule: &SamplingNoiseSchedule,
) -> Result<Vec<f32>, SamplerError> {
    if sample.len() != noise_pred.len() {
        return Err(SamplerError::DimensionMismatch {
            expected: sample.len(),
            got: noise_pred.len(),
        });
    }
    if let Some(p) = prev_noise_pred {
        if p.len() != sample.len() {
            return Err(SamplerError::DimensionMismatch {
                expected: sample.len(),
                got: p.len(),
            });
        }
    }

    let alpha_bar_t = schedule.alpha_bar_at(t);
    let sigma_t = schedule.sigma_at(t);

    let alpha_bar_t_prev = if t_prev > 0 {
        schedule.alpha_bar_at(t_prev)
    } else {
        1.0_f32
    };
    let sigma_t_prev = if t_prev > 0 {
        schedule.sigma_at(t_prev)
    } else {
        0.0_f32
    };

    // λ = log(α / σ) — log-SNR
    let lambda_t = {
        let ab = alpha_bar_t;
        let s = sigma_t.max(f32::EPSILON);
        (ab / s).ln()
    };
    let lambda_t_prev = {
        let ab = alpha_bar_t_prev;
        let s = sigma_t_prev.max(f32::EPSILON);
        // Guard: avoid log(inf) when sigma → 0
        if sigma_t_prev < f32::EPSILON {
            lambda_t + 10.0 // artificially large (prev is cleaner)
        } else {
            (ab / s).ln()
        }
    };

    // h = λ_{t-1} − λ_t  (positive when denoising because λ increases toward clean)
    let h = lambda_t_prev - lambda_t;

    // Coefficients for the exponential integrator
    // For 1st-order (Euler in λ-space):
    //   x_{t-1} = (σ_{t-1}/σ_t)·xₜ − α_{t-1}·(e^{-h} − 1)·(xₜ − σ_t·ε) / σ_t
    //
    // In practice we derive x₀ from ε and then apply:
    //   x_{t-1} = σ_{t-1}/σ_t · xₜ − α_{t-1}·(1 − e^{-h})·x₀_pred

    let alpha_t_prev = alpha_bar_t_prev.sqrt(); // √ᾱ_{t-1}
    let sigma_ratio = if sigma_t.abs() < f32::EPSILON {
        0.0
    } else {
        sigma_t_prev / sigma_t
    };
    let expm1_neg_h = (-h).exp() - 1.0; // e^{-h} - 1  (negative when denoising)

    // Predict x₀ from current noise prediction
    let x0_pred_t = predict_x0(sample, noise_pred, t, schedule)?;

    match prev_noise_pred {
        None => {
            // 1st-order (first step)
            let x_prev: Vec<f32> = sample
                .iter()
                .zip(x0_pred_t.iter())
                .map(|(&xti, &x0i)| sigma_ratio * xti - alpha_t_prev * expm1_neg_h * x0i)
                .collect();
            Ok(x_prev)
        }
        Some(prev_eps) => {
            // Compute x₀ prediction from the previous noise prediction
            // We re-derive prev x₀ using the previous step's timestep:
            // For simplicity we use the prev_eps at the same sample (standard DPM++ 2M approach).
            let x0_pred_prev: Vec<f32> = sample
                .iter()
                .zip(prev_eps.iter())
                .map(|(&xti, &pi)| {
                    // Use alpha_bar_t for both since we store the pred at step t
                    // and apply the blended x0 correction
                    let ab = alpha_bar_t;
                    let s = sigma_t.max(f32::EPSILON);
                    let inv_sqrt_ab = if ab.sqrt() < f32::EPSILON {
                        0.0
                    } else {
                        1.0 / ab.sqrt()
                    };
                    (xti - s * pi) * inv_sqrt_ab
                })
                .collect();

            // D1 correction: (x0_pred_t − x0_pred_prev) / (1/2 correction factor)
            // r = 1/2 for uniform steps; exact r would need h_prev
            let r = 0.5_f32;
            let d1_coeff = 1.0 / (2.0 * r);

            let x_prev: Vec<f32> = sample
                .iter()
                .zip(x0_pred_t.iter())
                .zip(x0_pred_prev.iter())
                .map(|((&xti, &x0i), &x0_prev_i)| {
                    let d0 = x0i;
                    let d1 = (x0i - x0_prev_i) * d1_coeff;
                    let x0_eff = d0 + d1 * (1.0 + r) * expm1_neg_h / 2.0;
                    sigma_ratio * xti - alpha_t_prev * expm1_neg_h * x0_eff
                })
                .collect();
            Ok(x_prev)
        }
    }
}

// ---------------------------------------------------------------------------
// MultiStepSampler
// ---------------------------------------------------------------------------

/// Multi-step diffusion sampler supporting DDIM, PLMS, and DPM++ 2M.
///
/// Usage:
/// 1. Construct with [`MultiStepSampler::new`].
/// 2. Call [`set_timesteps`][Self::set_timesteps] to compute the inference schedule.
/// 3. Loop: provide `(noise_pred, sample)` to [`step`][Self::step] until
///    [`is_done`][Self::is_done] returns `true`.
#[derive(Debug, Clone)]
pub struct MultiStepSampler {
    config: MultiStepSamplerConfig,
    /// Descending timestep schedule (length = n_inference_steps + 1).
    timesteps: Vec<usize>,
    /// Index into `timesteps`; incremented after each [`step`][Self::step] call.
    current_step: usize,
    /// Ring buffer of recent noise predictions (for PLMS and DPM++).
    history: Vec<Vec<f32>>,
    /// Previous denoised sample (unused internally but available for diagnostics).
    prev_sample: Option<Vec<f32>>,
}

impl MultiStepSampler {
    /// Create a new sampler.
    ///
    /// Validates configuration but does **not** compute timesteps;
    /// call [`set_timesteps`][Self::set_timesteps] before the first
    /// [`step`][Self::step].
    pub fn new(config: MultiStepSamplerConfig) -> Result<Self, SamplerError> {
        if config.n_inference_steps == 0 {
            return Err(SamplerError::InvalidParam(
                "n_inference_steps must be > 0".to_string(),
            ));
        }
        if config.n_inference_steps > config.schedule.n_timesteps {
            return Err(SamplerError::InvalidParam(format!(
                "n_inference_steps ({}) must be ≤ schedule.n_timesteps ({})",
                config.n_inference_steps, config.schedule.n_timesteps
            )));
        }
        if let SamplerKind::Ddim { eta } = config.kind {
            if !(0.0..=1.0).contains(&eta) {
                return Err(SamplerError::InvalidParam(format!(
                    "eta must be in [0, 1], got {eta}"
                )));
            }
        }
        Ok(Self {
            config,
            timesteps: Vec::new(),
            current_step: 0,
            history: Vec::new(),
            prev_sample: None,
        })
    }

    /// Compute and store the inference timestep schedule.
    ///
    /// Must be called before the first [`step`][Self::step].
    /// Resets `current_step` and `history`.
    pub fn set_timesteps(&mut self) -> Result<(), SamplerError> {
        self.timesteps = compute_timestep_schedule(
            self.config.schedule.n_timesteps,
            self.config.n_inference_steps,
        );
        self.current_step = 0;
        self.history.clear();
        self.prev_sample = None;
        Ok(())
    }

    /// Return the current training timestep index, or `None` if done.
    pub fn current_timestep(&self) -> Option<usize> {
        if self.timesteps.is_empty() || self.current_step >= self.timesteps.len() {
            None
        } else {
            Some(self.timesteps[self.current_step])
        }
    }

    /// Return `true` once all inference steps have been consumed.
    pub fn is_done(&self) -> bool {
        if self.timesteps.is_empty() {
            return true;
        }
        // We have n_steps intervals across n_steps+1 timesteps.
        // Done when current_step has reached the last interval.
        self.current_step + 1 >= self.timesteps.len()
    }

    /// Perform one denoising step.
    ///
    /// - `noise_pred`: the model's noise prediction εθ(xₜ, t).
    /// - `sample`: the current noisy latent xₜ.
    ///
    /// Returns xₜ₋₁.
    pub fn step(&mut self, noise_pred: &[f32], sample: &[f32]) -> Result<Vec<f32>, SamplerError> {
        if self.timesteps.is_empty() {
            return Err(SamplerError::NotInitialized);
        }
        if self.is_done() {
            return Err(SamplerError::SamplingComplete);
        }

        let t = self.timesteps[self.current_step];
        let t_prev = if self.current_step + 1 < self.timesteps.len() {
            self.timesteps[self.current_step + 1]
        } else {
            0
        };

        let x_prev = match &self.config.kind.clone() {
            SamplerKind::Ddim { eta } => ddim_step(
                sample,
                noise_pred,
                t,
                t_prev,
                &self.config.schedule,
                *eta,
                None,
            )?,
            SamplerKind::Plms => {
                let result = plms_step(
                    sample,
                    noise_pred,
                    &self.history,
                    t,
                    t_prev,
                    &self.config.schedule,
                )?;
                // Update history: keep last 3 predictions
                self.history.push(noise_pred.to_vec());
                if self.history.len() > 3 {
                    self.history.remove(0);
                }
                self.prev_sample = Some(result.clone());
                self.current_step += 1;
                return Ok(result);
            }
            SamplerKind::DpmPlusPlus2M => {
                let prev = self.history.last().map(|v| v.as_slice());
                let result = dpm_plus_plus_2m_step(
                    sample,
                    noise_pred,
                    prev,
                    t,
                    t_prev,
                    &self.config.schedule,
                )?;
                // Keep only the most recent prediction
                self.history.push(noise_pred.to_vec());
                if self.history.len() > 1 {
                    self.history.remove(0);
                }
                self.prev_sample = Some(result.clone());
                self.current_step += 1;
                return Ok(result);
            }
        };

        // For DDIM, update history and advance step
        if matches!(&self.config.kind, SamplerKind::Ddim { .. }) {
            self.history.push(noise_pred.to_vec());
            if self.history.len() > 3 {
                self.history.remove(0);
            }
        }

        self.prev_sample = Some(x_prev.clone());
        self.current_step += 1;
        Ok(x_prev)
    }

    /// Number of denoising steps remaining.
    pub fn steps_remaining(&self) -> usize {
        if self.timesteps.is_empty() {
            return 0;
        }
        let total_intervals = self.timesteps.len().saturating_sub(1);
        total_intervals.saturating_sub(self.current_step)
    }

    /// Total number of inference steps configured.
    pub fn total_steps(&self) -> usize {
        self.config.n_inference_steps
    }

    /// The full timestep schedule (descending).
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }
}

// ---------------------------------------------------------------------------
// Formatting helper
// ---------------------------------------------------------------------------

/// Return a human-readable summary of the sampler's current state.
pub fn format_sampler_stats(sampler: &MultiStepSampler) -> String {
    let kind_name = match &sampler.config.kind {
        SamplerKind::Ddim { eta } => format!("DDIM(η={eta:.3})"),
        SamplerKind::Plms => "PLMS".to_string(),
        SamplerKind::DpmPlusPlus2M => "DPM++2M".to_string(),
    };
    let current_t = sampler
        .current_timestep()
        .map(|t| t.to_string())
        .unwrap_or_else(|| "done".to_string());
    format!(
        "MultiStepSampler {{ kind={kind_name}, steps={}/{}, current_t={}, history_len={}, cfg={:.1} }}",
        sampler.current_step,
        sampler.config.n_inference_steps,
        current_t,
        sampler.history.len(),
        sampler.config.guidance_scale,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_cosine_schedule(n: usize) -> SamplingNoiseSchedule {
        SamplingNoiseSchedule::cosine(n)
    }

    fn make_linear_schedule(n: usize) -> SamplingNoiseSchedule {
        SamplingNoiseSchedule::linear(n, 0.0001, 0.02)
    }

    fn default_sampler(kind: SamplerKind, n_steps: usize) -> MultiStepSampler {
        let cfg = MultiStepSamplerConfig {
            kind,
            n_inference_steps: n_steps,
            schedule: make_cosine_schedule(100),
            guidance_scale: 7.5,
        };
        let mut s = MultiStepSampler::new(cfg).unwrap();
        s.set_timesteps().unwrap();
        s
    }

    fn zeros(len: usize) -> Vec<f32> {
        vec![0.0; len]
    }

    fn ones(len: usize) -> Vec<f32> {
        vec![1.0; len]
    }

    fn constant(val: f32, len: usize) -> Vec<f32> {
        vec![val; len]
    }

    // -----------------------------------------------------------------------
    // SamplingNoiseSchedule::cosine
    // -----------------------------------------------------------------------

    #[test]
    fn cosine_length() {
        let s = make_cosine_schedule(1000);
        assert_eq!(s.alpha_bars.len(), 1000);
        assert_eq!(s.sigmas.len(), 1000);
        assert_eq!(s.n_timesteps, 1000);
    }

    #[test]
    fn cosine_monotone_decreasing_alpha_bars() {
        let s = make_cosine_schedule(100);
        for i in 0..s.alpha_bars.len() - 1 {
            assert!(
                s.alpha_bars[i] >= s.alpha_bars[i + 1],
                "alpha_bar not monotone at {i}: {} < {}",
                s.alpha_bars[i],
                s.alpha_bars[i + 1]
            );
        }
    }

    #[test]
    fn cosine_boundary_values() {
        let s = make_cosine_schedule(100);
        // First alpha_bar should be close to 1
        assert!(s.alpha_bars[0] > 0.95, "alpha_bars[0]={}", s.alpha_bars[0]);
        // Last alpha_bar should be close to 0 but clamped above 0.0001
        assert!(s.alpha_bars[99] < 0.05);
        assert!(s.alpha_bars[99] >= 0.0001);
    }

    #[test]
    fn cosine_sigmas_are_sqrt_one_minus_ab() {
        let s = make_cosine_schedule(50);
        for i in 0..s.n_timesteps {
            let expected = (1.0 - s.alpha_bars[i]).sqrt();
            assert!((s.sigmas[i] - expected).abs() < 1e-6, "mismatch at {i}");
        }
    }

    #[test]
    fn cosine_clamped_range() {
        let s = make_cosine_schedule(1000);
        for &ab in &s.alpha_bars {
            assert!((0.0001..=0.9999).contains(&ab), "out of range: {ab}");
        }
    }

    // -----------------------------------------------------------------------
    // SamplingNoiseSchedule::linear
    // -----------------------------------------------------------------------

    #[test]
    fn linear_length() {
        let s = make_linear_schedule(200);
        assert_eq!(s.alpha_bars.len(), 200);
    }

    #[test]
    fn linear_monotone_decreasing() {
        let s = make_linear_schedule(100);
        for i in 0..s.alpha_bars.len() - 1 {
            assert!(s.alpha_bars[i] >= s.alpha_bars[i + 1]);
        }
    }

    #[test]
    fn linear_boundary_values() {
        let s = make_linear_schedule(100);
        // First alpha_bar should be high (low beta_start)
        assert!(s.alpha_bars[0] > 0.9, "first ab = {}", s.alpha_bars[0]);
        // All values clamped
        for &ab in &s.alpha_bars {
            assert!((0.0001..=0.9999).contains(&ab));
        }
    }

    #[test]
    fn linear_sigmas_consistent() {
        let s = make_linear_schedule(50);
        for i in 0..s.n_timesteps {
            let expected = (1.0 - s.alpha_bars[i]).sqrt();
            assert!((s.sigmas[i] - expected).abs() < 1e-6);
        }
    }

    // -----------------------------------------------------------------------
    // SamplingNoiseSchedule methods
    // -----------------------------------------------------------------------

    #[test]
    fn alpha_bar_at_clamps_out_of_bounds() {
        let s = make_cosine_schedule(100);
        // Should not panic and should return the last element
        let val = s.alpha_bar_at(9999);
        assert_eq!(val, s.alpha_bars[99]);
    }

    #[test]
    fn sigma_at_clamps_out_of_bounds() {
        let s = make_cosine_schedule(100);
        let val = s.sigma_at(9999);
        assert_eq!(val, s.sigmas[99]);
    }

    #[test]
    fn snr_at_positive() {
        let s = make_cosine_schedule(100);
        // SNR = alpha_bar / (1 - alpha_bar) — always positive for ab in (0,1)
        for t in 0..100 {
            assert!(s.snr_at(t) > 0.0);
        }
    }

    #[test]
    fn snr_at_decreasing() {
        let s = make_cosine_schedule(100);
        // SNR should decrease as noise increases (higher t)
        let snr_early = s.snr_at(0);
        let snr_late = s.snr_at(99);
        assert!(
            snr_early > snr_late,
            "snr_early={snr_early}, snr_late={snr_late}"
        );
    }

    // -----------------------------------------------------------------------
    // compute_timestep_schedule
    // -----------------------------------------------------------------------

    #[test]
    fn schedule_length() {
        let ts = compute_timestep_schedule(1000, 50);
        // At most n_steps+1, could be fewer after dedup
        assert!(ts.len() <= 51);
        assert!(!ts.is_empty());
    }

    #[test]
    fn schedule_first_is_max() {
        let ts = compute_timestep_schedule(1000, 50);
        assert_eq!(*ts.first().unwrap(), 999);
    }

    #[test]
    fn schedule_last_is_zero() {
        let ts = compute_timestep_schedule(1000, 50);
        assert_eq!(*ts.last().unwrap(), 0);
    }

    #[test]
    fn schedule_descending() {
        let ts = compute_timestep_schedule(1000, 50);
        for i in 0..ts.len() - 1 {
            assert!(
                ts[i] >= ts[i + 1],
                "not descending at {i}: {} < {}",
                ts[i],
                ts[i + 1]
            );
        }
    }

    #[test]
    fn schedule_empty_inputs() {
        let ts = compute_timestep_schedule(0, 0);
        assert!(!ts.is_empty());
    }

    #[test]
    fn schedule_single_step() {
        let ts = compute_timestep_schedule(100, 1);
        assert!(ts.len() >= 2);
        assert_eq!(*ts.first().unwrap(), 99);
        assert_eq!(*ts.last().unwrap(), 0);
    }

    // -----------------------------------------------------------------------
    // predict_x0
    // -----------------------------------------------------------------------

    #[test]
    fn predict_x0_identity_at_zero_noise() {
        // When sigma_t = 0 (alpha_bar ≈ 1), x0_pred ≈ sample
        // Use a near-zero alpha_bar via last timestep of cosine which is ~0
        // Instead: use a cosine schedule at t=0 where alpha_bar is near 1
        let sched = make_cosine_schedule(100);
        let sample = vec![0.5_f32, -0.3, 0.8];
        let noise_pred = zeros(3);
        let x0 = predict_x0(&sample, &noise_pred, 0, &sched).unwrap();
        // At t=0, sqrt_one_minus_ab is small, so x0 ≈ sample / sqrt_ab
        // sqrt_ab ≈ 1 at t=0 for cosine
        for (g, &s) in x0.iter().zip(sample.iter()) {
            assert!((g - s).abs() < 0.05, "x0={g}, sample={s}");
        }
    }

    #[test]
    fn predict_x0_dimension_mismatch() {
        let sched = make_cosine_schedule(100);
        let err = predict_x0(&[1.0, 2.0], &[1.0], 0, &sched);
        assert!(matches!(err, Err(SamplerError::DimensionMismatch { .. })));
    }

    #[test]
    fn predict_x0_known_value() {
        // Synthetic: alpha_bar = 0.64, sqrt_ab = 0.8, sqrt(1-ab) = 0.6
        // sample = 0.8 * x0 + 0.6 * eps  => x0_pred = (sample - 0.6*eps)/0.8
        let sched = SamplingNoiseSchedule {
            n_timesteps: 1,
            alpha_bars: vec![0.64],
            sigmas: vec![0.6],
        };
        let x0_true = 2.0_f32;
        let eps = 1.5_f32;
        let sample = vec![0.8 * x0_true + 0.6 * eps];
        let noise_pred = vec![eps];
        let x0 = predict_x0(&sample, &noise_pred, 0, &sched).unwrap();
        assert!((x0[0] - x0_true).abs() < 1e-5, "x0={}", x0[0]);
    }

    // -----------------------------------------------------------------------
    // sampler_apply_cfg
    // -----------------------------------------------------------------------

    #[test]
    fn cfg_scale_1_equals_cond() {
        let cond = vec![1.0_f32, 2.0, 3.0];
        let uncond = vec![0.5_f32, 1.0, 1.5];
        let out = sampler_apply_cfg(&cond, &uncond, 1.0).unwrap();
        for (o, &c) in out.iter().zip(cond.iter()) {
            assert!((o - c).abs() < 1e-6, "o={o}, c={c}");
        }
    }

    #[test]
    fn cfg_scale_0_equals_uncond() {
        let cond = vec![1.0_f32, 2.0, 3.0];
        let uncond = vec![0.5_f32, 1.0, 1.5];
        let out = sampler_apply_cfg(&cond, &uncond, 0.0).unwrap();
        for (o, &u) in out.iter().zip(uncond.iter()) {
            assert!((o - u).abs() < 1e-6);
        }
    }

    #[test]
    fn cfg_dimension_mismatch() {
        let err = sampler_apply_cfg(&[1.0, 2.0], &[1.0], 7.5);
        assert!(matches!(err, Err(SamplerError::DimensionMismatch { .. })));
    }

    #[test]
    fn cfg_negative_scale_error() {
        let err = sampler_apply_cfg(&[1.0], &[1.0], -1.0);
        assert!(matches!(err, Err(SamplerError::InvalidParam(_))));
    }

    #[test]
    fn cfg_high_scale_amplifies() {
        let cond = vec![2.0_f32];
        let uncond = vec![1.0_f32];
        let out = sampler_apply_cfg(&cond, &uncond, 10.0).unwrap();
        // out = 1 + 10*(2-1) = 11
        assert!((out[0] - 11.0).abs() < 1e-5, "out={}", out[0]);
    }

    // -----------------------------------------------------------------------
    // ddim_step
    // -----------------------------------------------------------------------

    #[test]
    fn ddim_deterministic_eta0() {
        let sched = make_cosine_schedule(100);
        let sample = vec![0.5_f32; 4];
        let noise = zeros(4);
        let r1 = ddim_step(&sample, &noise, 50, 40, &sched, 0.0, None).unwrap();
        let r2 = ddim_step(&sample, &noise, 50, 40, &sched, 0.0, None).unwrap();
        assert_eq!(r1, r2, "DDIM with eta=0 must be deterministic");
    }

    #[test]
    fn ddim_eta1_with_noise() {
        let sched = make_cosine_schedule(100);
        let sample = vec![0.0_f32; 4];
        let noise_pred = zeros(4);
        let noise = constant(0.1, 4);
        // Should not error
        ddim_step(&sample, &noise_pred, 50, 40, &sched, 1.0, Some(&noise)).unwrap();
    }

    #[test]
    fn ddim_dim_mismatch_sample_noise_pred() {
        let sched = make_cosine_schedule(100);
        let err = ddim_step(&[1.0_f32, 2.0], &[1.0_f32], 50, 40, &sched, 0.0, None);
        assert!(matches!(err, Err(SamplerError::DimensionMismatch { .. })));
    }

    #[test]
    fn ddim_dim_mismatch_noise() {
        let sched = make_cosine_schedule(100);
        let err = ddim_step(
            &[1.0_f32, 2.0],
            &[1.0_f32, 2.0],
            50,
            40,
            &sched,
            1.0,
            Some(&[1.0_f32]),
        );
        assert!(matches!(err, Err(SamplerError::DimensionMismatch { .. })));
    }

    #[test]
    fn ddim_output_shape_preserved() {
        let sched = make_cosine_schedule(100);
        let n = 64;
        let sample = zeros(n);
        let noise_pred = ones(n);
        let out = ddim_step(&sample, &noise_pred, 50, 30, &sched, 0.0, None).unwrap();
        assert_eq!(out.len(), n);
    }

    #[test]
    fn ddim_zero_noise_pred_converges_toward_x0() {
        // If noise_pred = 0, then x0_pred = sample / sqrt_ab
        // and x_prev = sqrt_ab_prev * x0_pred = sqrt_ab_prev/sqrt_ab * sample
        let sched = make_cosine_schedule(100);
        let sample = vec![1.0_f32; 4];
        let noise_pred = zeros(4);
        let out = ddim_step(&sample, &noise_pred, 50, 30, &sched, 0.0, None).unwrap();
        // Values should be finite and non-zero
        for v in &out {
            assert!(v.is_finite(), "non-finite output");
        }
    }

    // -----------------------------------------------------------------------
    // plms_step
    // -----------------------------------------------------------------------

    #[test]
    fn plms_1st_order_matches_ddim() {
        let sched = make_cosine_schedule(100);
        let sample = vec![0.5_f32; 4];
        let noise_pred = constant(0.2, 4);
        let history: Vec<Vec<f32>> = vec![];
        let plms_out = plms_step(&sample, &noise_pred, &history, 50, 40, &sched).unwrap();
        let ddim_out = ddim_step(&sample, &noise_pred, 50, 40, &sched, 0.0, None).unwrap();
        for (p, d) in plms_out.iter().zip(ddim_out.iter()) {
            assert!((p - d).abs() < 1e-5, "plms={p}, ddim={d}");
        }
    }

    #[test]
    fn plms_2nd_order_valid() {
        let sched = make_cosine_schedule(100);
        let sample = vec![0.5_f32; 4];
        let noise_pred = constant(0.2, 4);
        let history = vec![constant(0.18, 4)];
        let out = plms_step(&sample, &noise_pred, &history, 50, 40, &sched).unwrap();
        assert_eq!(out.len(), 4);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn plms_3rd_order_valid() {
        let sched = make_cosine_schedule(100);
        let sample = constant(0.5, 4);
        let noise_pred = constant(0.2, 4);
        let history = vec![constant(0.18, 4), constant(0.17, 4)];
        let out = plms_step(&sample, &noise_pred, &history, 50, 40, &sched).unwrap();
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn plms_4th_order_valid() {
        let sched = make_cosine_schedule(100);
        let sample = constant(0.5, 4);
        let noise_pred = constant(0.2, 4);
        let history = vec![constant(0.18, 4), constant(0.17, 4), constant(0.16, 4)];
        let out = plms_step(&sample, &noise_pred, &history, 50, 40, &sched).unwrap();
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn plms_dim_mismatch() {
        let sched = make_cosine_schedule(100);
        let err = plms_step(&[1.0_f32, 2.0], &[1.0_f32], &[], 50, 40, &sched);
        assert!(matches!(err, Err(SamplerError::DimensionMismatch { .. })));
    }

    #[test]
    fn plms_history_dim_mismatch() {
        let sched = make_cosine_schedule(100);
        let history = vec![vec![1.0_f32]]; // wrong len
        let err = plms_step(&[1.0_f32, 2.0], &[0.1_f32, 0.2], &history, 50, 40, &sched);
        assert!(matches!(err, Err(SamplerError::DimensionMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // dpm_plus_plus_2m_step
    // -----------------------------------------------------------------------

    #[test]
    fn dpmpp_first_step_no_history() {
        let sched = make_cosine_schedule(100);
        let sample = constant(0.5, 4);
        let noise_pred = constant(0.3, 4);
        let out = dpm_plus_plus_2m_step(&sample, &noise_pred, None, 50, 40, &sched).unwrap();
        assert_eq!(out.len(), 4);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn dpmpp_second_step_with_history() {
        let sched = make_cosine_schedule(100);
        let sample = constant(0.5, 4);
        let noise_pred = constant(0.3, 4);
        let prev = constant(0.28, 4);
        let out = dpm_plus_plus_2m_step(&sample, &noise_pred, Some(&prev), 50, 40, &sched).unwrap();
        assert_eq!(out.len(), 4);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn dpmpp_dim_mismatch_sample() {
        let sched = make_cosine_schedule(100);
        let err = dpm_plus_plus_2m_step(&[1.0_f32, 2.0], &[1.0_f32], None, 50, 40, &sched);
        assert!(matches!(err, Err(SamplerError::DimensionMismatch { .. })));
    }

    #[test]
    fn dpmpp_dim_mismatch_prev() {
        let sched = make_cosine_schedule(100);
        let err = dpm_plus_plus_2m_step(
            &[1.0_f32, 2.0],
            &[0.1_f32, 0.2],
            Some(&[0.1_f32]),
            50,
            40,
            &sched,
        );
        assert!(matches!(err, Err(SamplerError::DimensionMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // MultiStepSampler::new
    // -----------------------------------------------------------------------

    #[test]
    fn sampler_new_valid_config() {
        let cfg = MultiStepSamplerConfig::default();
        assert!(MultiStepSampler::new(cfg).is_ok());
    }

    #[test]
    fn sampler_new_zero_steps_error() {
        let cfg = MultiStepSamplerConfig {
            n_inference_steps: 0,
            ..Default::default()
        };
        assert!(matches!(
            MultiStepSampler::new(cfg),
            Err(SamplerError::InvalidParam(_))
        ));
    }

    #[test]
    fn sampler_new_too_many_steps_error() {
        let cfg = MultiStepSamplerConfig {
            n_inference_steps: 2000,
            ..Default::default()
        };
        assert!(matches!(
            MultiStepSampler::new(cfg),
            Err(SamplerError::InvalidParam(_))
        ));
    }

    #[test]
    fn sampler_new_invalid_eta() {
        let cfg = MultiStepSamplerConfig {
            kind: SamplerKind::Ddim { eta: 1.5 },
            ..Default::default()
        };
        assert!(matches!(
            MultiStepSampler::new(cfg),
            Err(SamplerError::InvalidParam(_))
        ));
    }

    // -----------------------------------------------------------------------
    // MultiStepSampler::set_timesteps
    // -----------------------------------------------------------------------

    #[test]
    fn set_timesteps_correct_count() {
        let mut s = default_sampler(SamplerKind::Ddim { eta: 0.0 }, 20);
        // timesteps.len() should be at most n_inference_steps + 1
        assert!(s.timesteps().len() <= 21);
        assert!(!s.timesteps().is_empty());
        // After set_timesteps current_step is 0
        assert_eq!(s.current_step, 0);
        // Resetting should not error
        s.set_timesteps().unwrap();
    }

    #[test]
    fn set_timesteps_resets_history() {
        let cfg = MultiStepSamplerConfig {
            kind: SamplerKind::Plms,
            n_inference_steps: 10,
            schedule: make_cosine_schedule(100),
            guidance_scale: 7.5,
        };
        let mut s = MultiStepSampler::new(cfg).unwrap();
        s.set_timesteps().unwrap();
        // Inject some fake history
        let _ = s.step(&constant(0.1, 8), &constant(0.5, 8));
        // Reset
        s.set_timesteps().unwrap();
        assert_eq!(s.history.len(), 0);
        assert_eq!(s.current_step, 0);
    }

    // -----------------------------------------------------------------------
    // MultiStepSampler::step — DDIM
    // -----------------------------------------------------------------------

    #[test]
    fn ddim_sampler_full_run() {
        let mut s = default_sampler(SamplerKind::Ddim { eta: 0.0 }, 10);
        let n = 16;
        let mut sample = constant(1.0, n);
        while !s.is_done() {
            let noise_pred = constant(0.1, n);
            sample = s.step(&noise_pred, &sample).unwrap();
        }
        assert_eq!(sample.len(), n);
        for v in &sample {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn ddim_step_after_done_errors() {
        let mut s = default_sampler(SamplerKind::Ddim { eta: 0.0 }, 5);
        let n = 4;
        let sample = constant(0.5, n);
        // Exhaust all steps
        while !s.is_done() {
            let _ = s.step(&constant(0.1, n), &sample);
        }
        let err = s.step(&constant(0.1, n), &sample);
        assert!(matches!(err, Err(SamplerError::SamplingComplete)));
    }

    #[test]
    fn ddim_step_not_initialized_errors() {
        let cfg = MultiStepSamplerConfig::default();
        let mut s = MultiStepSampler::new(cfg).unwrap();
        // set_timesteps not called
        let err = s.step(&[0.1], &[0.5]);
        assert!(matches!(err, Err(SamplerError::NotInitialized)));
    }

    #[test]
    fn ddim_steps_remaining_decreases() {
        let mut s = default_sampler(SamplerKind::Ddim { eta: 0.0 }, 10);
        let initial_remaining = s.steps_remaining();
        assert!(initial_remaining > 0);
        let _ = s.step(&constant(0.1, 4), &constant(0.5, 4));
        assert_eq!(s.steps_remaining(), initial_remaining - 1);
    }

    #[test]
    fn ddim_total_steps() {
        let s = default_sampler(SamplerKind::Ddim { eta: 0.0 }, 25);
        assert_eq!(s.total_steps(), 25);
    }

    #[test]
    fn ddim_timesteps_returns_schedule() {
        let s = default_sampler(SamplerKind::Ddim { eta: 0.0 }, 10);
        let ts = s.timesteps();
        assert!(!ts.is_empty());
        assert_eq!(*ts.first().unwrap(), 99); // n_timesteps=100, max_t=99
    }

    // -----------------------------------------------------------------------
    // MultiStepSampler::step — PLMS
    // -----------------------------------------------------------------------

    #[test]
    fn plms_sampler_full_run() {
        let mut s = default_sampler(SamplerKind::Plms, 10);
        let n = 8;
        let mut sample = constant(1.0, n);
        while !s.is_done() {
            let noise_pred = constant(0.1, n);
            sample = s.step(&noise_pred, &sample).unwrap();
        }
        assert_eq!(sample.len(), n);
        for v in &sample {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn plms_history_grows_to_max_3() {
        let cfg = MultiStepSamplerConfig {
            kind: SamplerKind::Plms,
            n_inference_steps: 10,
            schedule: make_cosine_schedule(100),
            guidance_scale: 7.5,
        };
        let mut s = MultiStepSampler::new(cfg).unwrap();
        s.set_timesteps().unwrap();
        let n = 4;
        for _ in 0..5 {
            if s.is_done() {
                break;
            }
            let _ = s.step(&constant(0.1, n), &constant(0.5, n));
        }
        assert!(s.history.len() <= 3);
    }

    // -----------------------------------------------------------------------
    // MultiStepSampler::step — DPM++2M
    // -----------------------------------------------------------------------

    #[test]
    fn dpmpp_sampler_full_run() {
        let mut s = default_sampler(SamplerKind::DpmPlusPlus2M, 10);
        let n = 8;
        let mut sample = constant(1.0, n);
        while !s.is_done() {
            let noise_pred = constant(0.1, n);
            sample = s.step(&noise_pred, &sample).unwrap();
        }
        assert_eq!(sample.len(), n);
        for v in &sample {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn dpmpp_history_max_1() {
        let cfg = MultiStepSamplerConfig {
            kind: SamplerKind::DpmPlusPlus2M,
            n_inference_steps: 10,
            schedule: make_cosine_schedule(100),
            guidance_scale: 7.5,
        };
        let mut s = MultiStepSampler::new(cfg).unwrap();
        s.set_timesteps().unwrap();
        let n = 4;
        for _ in 0..5 {
            if s.is_done() {
                break;
            }
            let _ = s.step(&constant(0.1, n), &constant(0.5, n));
        }
        assert!(s.history.len() <= 1);
    }

    // -----------------------------------------------------------------------
    // Determinism across runs
    // -----------------------------------------------------------------------

    #[test]
    fn ddim_two_identical_runs_produce_same_output() {
        let make_sampler = || default_sampler(SamplerKind::Ddim { eta: 0.0 }, 5);
        let n = 8;
        let noise_preds: Vec<Vec<f32>> = (0..5).map(|i| constant(i as f32 * 0.05, n)).collect();
        let initial = constant(1.0, n);

        let run = |mut s: MultiStepSampler| -> Vec<f32> {
            let mut sample = initial.clone();
            for pred in noise_preds.iter().take(5) {
                if s.is_done() {
                    break;
                }
                sample = s.step(pred, &sample).unwrap();
            }
            sample
        };

        let out1 = run(make_sampler());
        let out2 = run(make_sampler());
        assert_eq!(out1, out2);
    }

    // -----------------------------------------------------------------------
    // current_timestep
    // -----------------------------------------------------------------------

    #[test]
    fn current_timestep_before_done() {
        let s = default_sampler(SamplerKind::Ddim { eta: 0.0 }, 10);
        assert!(s.current_timestep().is_some());
    }

    #[test]
    fn current_timestep_after_done() {
        let mut s = default_sampler(SamplerKind::Ddim { eta: 0.0 }, 5);
        let n = 4;
        while !s.is_done() {
            let _ = s.step(&constant(0.1, n), &constant(0.5, n));
        }
        // After done, current_timestep should be None or point to last
        // (either is valid — just must not panic)
        let _ = s.current_timestep();
    }

    // -----------------------------------------------------------------------
    // format_sampler_stats
    // -----------------------------------------------------------------------

    #[test]
    fn format_stats_non_empty() {
        let s = default_sampler(SamplerKind::Ddim { eta: 0.0 }, 10);
        let stats = format_sampler_stats(&s);
        assert!(!stats.is_empty());
        assert!(stats.contains("DDIM"));
    }

    #[test]
    fn format_stats_plms() {
        let s = default_sampler(SamplerKind::Plms, 10);
        let stats = format_sampler_stats(&s);
        assert!(stats.contains("PLMS"));
    }

    #[test]
    fn format_stats_dpmpp() {
        let s = default_sampler(SamplerKind::DpmPlusPlus2M, 10);
        let stats = format_sampler_stats(&s);
        assert!(stats.contains("DPM++2M"));
    }

    // -----------------------------------------------------------------------
    // Edge cases and regression
    // -----------------------------------------------------------------------

    #[test]
    fn cosine_schedule_single_step_sampler() {
        // Edge: n_inference_steps = 1
        let cfg = MultiStepSamplerConfig {
            kind: SamplerKind::Ddim { eta: 0.0 },
            n_inference_steps: 1,
            schedule: make_cosine_schedule(100),
            guidance_scale: 1.0,
        };
        let mut s = MultiStepSampler::new(cfg).unwrap();
        s.set_timesteps().unwrap();
        assert!(!s.is_done());
        let out = s.step(&constant(0.1, 4), &constant(0.5, 4)).unwrap();
        assert_eq!(out.len(), 4);
        assert!(s.is_done());
    }

    #[test]
    fn snr_at_consistent_with_alpha_bar() {
        let s = make_cosine_schedule(100);
        for t in 0..100 {
            let ab = s.alpha_bar_at(t);
            let snr_expected = ab / (1.0 - ab);
            let snr_actual = s.snr_at(t);
            assert!((snr_actual - snr_expected).abs() < 1e-5 * snr_expected.abs().max(1.0));
        }
    }

    #[test]
    fn predict_x0_large_vector() {
        let sched = make_cosine_schedule(100);
        let n = 1024;
        let sample = constant(0.5, n);
        let noise_pred = constant(0.2, n);
        let out = predict_x0(&sample, &noise_pred, 30, &sched).unwrap();
        assert_eq!(out.len(), n);
    }

    #[test]
    fn ddim_step_t_eq_t_prev_at_boundary() {
        // t_prev = 0: alpha_bar_t_prev = 1.0
        let sched = make_cosine_schedule(100);
        let sample = constant(0.5, 4);
        let noise_pred = constant(0.1, 4);
        let out = ddim_step(&sample, &noise_pred, 5, 0, &sched, 0.0, None).unwrap();
        assert_eq!(out.len(), 4);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn sampler_is_done_after_all_steps() {
        let n_steps = 8;
        let mut s = default_sampler(SamplerKind::Ddim { eta: 0.0 }, n_steps);
        let mut count = 0;
        while !s.is_done() {
            let _ = s.step(&constant(0.1, 4), &constant(0.5, 4));
            count += 1;
            assert!(count <= n_steps + 2, "infinite loop guard");
        }
        assert!(count <= n_steps);
    }

    #[test]
    fn default_config_values() {
        let cfg = MultiStepSamplerConfig::default();
        assert_eq!(cfg.n_inference_steps, 50);
        assert_eq!(cfg.guidance_scale, 7.5);
        assert!(matches!(cfg.kind, SamplerKind::Ddim { eta } if (eta - 0.0).abs() < 1e-9));
    }
}
