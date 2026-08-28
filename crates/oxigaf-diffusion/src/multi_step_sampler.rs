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
// Private PRNG (module-local copies, following the score_matching.rs pattern)
// ---------------------------------------------------------------------------

/// Default seed for a sampler's internal noise source: any fixed non-zero
/// constant, so stochastic DDIM is reproducible out of the box while
/// [`MultiStepSampler::set_seed`] varies the trajectory.
const DEFAULT_SAMPLER_SEED: u64 = 0x2545_F491_4F6C_DD1D;

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

/// Uniform f32 in [0, 1) using 53 mantissa bits.
#[inline]
fn xorshift_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 11) as f32 / (1u64 << 53) as f32
}

/// Box-Muller transform: one standard normal sample N(0, 1) per call.
fn bm_normal(state: &mut u64) -> f32 {
    let u1 = (xorshift_f32(state) + 1e-10_f32).max(1e-10_f32);
    let u2 = xorshift_f32(state);
    (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos()
}

/// Draw `len` independent standard-normal samples.
fn standard_normal_vec(state: &mut u64, len: usize) -> Vec<f32> {
    (0..len).map(|_| bm_normal(state)).collect()
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
    ///
    /// `n_timesteps == 0` (and any other schedule whose vectors are shorter
    /// than `n_timesteps`, which the public fields make representable) has no
    /// coefficient to return; the noise-free limit `1.0` is reported instead of
    /// indexing out of bounds.
    #[inline]
    pub fn alpha_bar_at(&self, t: usize) -> f32 {
        let idx = t.min(self.n_timesteps.saturating_sub(1));
        self.alpha_bars.get(idx).copied().unwrap_or(1.0)
    }

    /// Return σ_t = √(1 − ᾱ_t).  Clamps `t` to the valid range.
    ///
    /// Reports the noise-free limit `0.0` for an empty schedule, mirroring
    /// [`alpha_bar_at`][Self::alpha_bar_at].
    #[inline]
    pub fn sigma_at(&self, t: usize) -> f32 {
        let idx = t.min(self.n_timesteps.saturating_sub(1));
        self.sigmas.get(idx).copied().unwrap_or(0.0)
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
// Step kernels
// ---------------------------------------------------------------------------

mod steps;

pub use steps::{
    ddim_step, dpm_plus_plus_2m_step, dpm_step_size, plms_step, predict_x0, sampler_apply_cfg,
};

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
    /// Ring buffer of recent noise predictions: the Adams-Bashforth history
    /// PLMS integrates over, also kept for DDIM purely as diagnostics.
    history: Vec<Vec<f32>>,
    /// Previous denoised sample (unused internally but available for diagnostics).
    prev_sample: Option<Vec<f32>>,
    /// Previous step's x₀ prediction — DPM++ 2M second-order state.
    prev_x0: Option<Vec<f32>>,
    /// Previous step's log-SNR step size `h` — DPM++ 2M second-order state.
    prev_h: Option<f32>,
    /// Seed the internal noise source is (re)initialized from.
    seed: u64,
    /// Live state of the internal noise source used by stochastic DDIM (η > 0).
    rng_state: u64,
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
            prev_x0: None,
            prev_h: None,
            seed: DEFAULT_SAMPLER_SEED,
            rng_state: DEFAULT_SAMPLER_SEED,
        })
    }

    /// Seed the internal noise source used by stochastic DDIM (η > 0).
    ///
    /// Two samplers seeded alike produce identical stochastic trajectories. The
    /// seed applies immediately and survives
    /// [`set_timesteps`][Self::set_timesteps]. A zero seed maps to the default,
    /// because the xorshift state must be non-zero.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = if seed == 0 {
            DEFAULT_SAMPLER_SEED
        } else {
            seed
        };
        self.rng_state = self.seed;
    }

    /// Compute and store the inference timestep schedule.
    ///
    /// Must be called before the first [`step`][Self::step].
    /// Resets `current_step`, `history`, the multistep state and the noise
    /// source, so re-running a sampler reproduces the previous trajectory.
    pub fn set_timesteps(&mut self) -> Result<(), SamplerError> {
        self.timesteps = compute_timestep_schedule(
            self.config.schedule.n_timesteps,
            self.config.n_inference_steps,
        );
        self.current_step = 0;
        self.history.clear();
        self.prev_sample = None;
        self.prev_x0 = None;
        self.prev_h = None;
        self.rng_state = self.seed;
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
    ///
    /// Stochastic DDIM (`SamplerKind::Ddim` with η > 0) draws its own Gaussian
    /// noise from the sampler's seeded PRNG, so the σ_t·z term the η > 0
    /// variance schedule budgets for is actually applied. Use
    /// [`step_with_noise`][Self::step_with_noise] to supply that draw yourself.
    pub fn step(&mut self, noise_pred: &[f32], sample: &[f32]) -> Result<Vec<f32>, SamplerError> {
        self.step_inner(noise_pred, sample, None)
    }

    /// Perform one denoising step with caller-supplied stochastic noise.
    ///
    /// `noise` must match `sample` in length and is consumed only by
    /// `SamplerKind::Ddim` with η > 0; the other samplers are deterministic and
    /// ignore it.
    pub fn step_with_noise(
        &mut self,
        noise_pred: &[f32],
        sample: &[f32],
        noise: &[f32],
    ) -> Result<Vec<f32>, SamplerError> {
        self.step_inner(noise_pred, sample, Some(noise))
    }

    fn step_inner(
        &mut self,
        noise_pred: &[f32],
        sample: &[f32],
        noise: Option<&[f32]>,
    ) -> Result<Vec<f32>, SamplerError> {
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

        let x_prev = match self.config.kind.clone() {
            SamplerKind::Ddim { eta } => {
                // η > 0 shrinks the deterministic direction term by σ_t², which
                // only stays variance-preserving if σ_t·z is added back.
                let generated = if eta > 0.0 && noise.is_none() {
                    Some(standard_normal_vec(&mut self.rng_state, sample.len()))
                } else {
                    None
                };
                let draw = if eta > 0.0 {
                    noise.or(generated.as_deref())
                } else {
                    None
                };
                // Keep the most recent predictions for diagnostics.
                self.history.push(noise_pred.to_vec());
                if self.history.len() > 3 {
                    self.history.remove(0);
                }
                let sched = &self.config.schedule;
                ddim_step(sample, noise_pred, t, t_prev, sched, eta, draw)?
            }
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
                result
            }
            SamplerKind::DpmPlusPlus2M => {
                let result = dpm_plus_plus_2m_step(
                    sample,
                    noise_pred,
                    self.prev_x0.as_deref(),
                    self.prev_h,
                    t,
                    t_prev,
                    &self.config.schedule,
                )?;
                // Carry this step's x₀ estimate and log-SNR step size forward:
                // the 2M correction is a finite difference in λ-space, so both
                // the previous x₀ *and* the previous h are required.
                let x0_now = predict_x0(sample, noise_pred, t, &self.config.schedule)?;
                let h_now = dpm_step_size(t, t_prev, &self.config.schedule);
                self.prev_x0 = Some(x0_now);
                self.prev_h = h_now;
                result
            }
        };

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
        let out = dpm_plus_plus_2m_step(&sample, &noise_pred, None, None, 50, 40, &sched).unwrap();
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
        let prev_x0 = constant(0.28, 4);
        let h_prev = dpm_step_size(60, 50, &sched);
        let out =
            dpm_plus_plus_2m_step(&sample, &noise_pred, Some(&prev_x0), h_prev, 50, 40, &sched)
                .unwrap();
        assert_eq!(out.len(), 4);
        for v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn dpmpp_dim_mismatch_sample() {
        let sched = make_cosine_schedule(100);
        let err = dpm_plus_plus_2m_step(&[1.0_f32, 2.0], &[1.0_f32], None, None, 50, 40, &sched);
        assert!(matches!(err, Err(SamplerError::DimensionMismatch { .. })));
    }

    #[test]
    fn dpmpp_dim_mismatch_prev() {
        let sched = make_cosine_schedule(100);
        let err = dpm_plus_plus_2m_step(
            &[1.0_f32, 2.0],
            &[0.1_f32, 0.2],
            Some(&[0.1_f32]),
            Some(0.5),
            50,
            40,
            &sched,
        );
        assert!(matches!(err, Err(SamplerError::DimensionMismatch { .. })));
    }

    // -----------------------------------------------------------------------
    // Regression: DPM++ 2M first order is algebraically DDIM at η=0.
    // This pins λ = log(√ᾱ/σ); the earlier λ = log(ᾱ/σ) fails it.
    // -----------------------------------------------------------------------
    #[test]
    fn dpmpp_first_order_matches_ddim_eta0() {
        let sched = make_cosine_schedule(100);
        let sample = vec![0.5_f32, -0.3, 1.2, 0.0];
        let noise_pred = vec![0.3_f32, 0.1, -0.4, 0.7];
        for (t, t_prev) in [(50_usize, 40_usize), (5, 0)] {
            let dpm =
                dpm_plus_plus_2m_step(&sample, &noise_pred, None, None, t, t_prev, &sched).unwrap();
            let ddim = ddim_step(&sample, &noise_pred, t, t_prev, &sched, 0.0, None).unwrap();
            for (a, b) in dpm.iter().zip(ddim.iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "t={t}, t_prev={t_prev}: dpm={a}, ddim={b}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Regression: the 2M correction uses r = h_prev/h and the reference D.
    // -----------------------------------------------------------------------
    #[test]
    fn dpmpp_second_order_matches_reference_formula() {
        let sched = make_cosine_schedule(100);
        let sample = vec![0.5_f32, -0.3, 1.2];
        let noise_pred = vec![0.3_f32, 0.1, -0.4];
        let prev_x0 = vec![0.2_f32, 0.4, -0.1];
        let (t, t_prev) = (50_usize, 40_usize);
        let h = dpm_step_size(t, t_prev, &sched).expect("finite h away from the boundary");
        let h_prev = dpm_step_size(60, t, &sched).expect("finite h_prev");

        let got = dpm_plus_plus_2m_step(
            &sample,
            &noise_pred,
            Some(&prev_x0),
            Some(h_prev),
            t,
            t_prev,
            &sched,
        )
        .unwrap();

        // Reference: D = (1 + 1/(2r))·x0_t − (1/(2r))·x0_prev,
        //            x = (σ_prev/σ_t)·x − α_prev·(e^{-h} − 1)·D
        let r = h_prev / h;
        let c = 1.0 / (2.0 * r);
        let x0_t = predict_x0(&sample, &noise_pred, t, &sched).unwrap();
        let sigma_ratio = sched.sigma_at(t_prev) / sched.sigma_at(t);
        let alpha_prev = sched.alpha_bar_at(t_prev).sqrt();
        let expm1_neg_h = (-h).exp() - 1.0;
        for i in 0..sample.len() {
            let d = (1.0 + c) * x0_t[i] - c * prev_x0[i];
            let expected = sigma_ratio * sample[i] - alpha_prev * expm1_neg_h * d;
            assert!(
                (got[i] - expected).abs() < 1e-5,
                "i={i}: got {}, expected {expected}",
                got[i]
            );
        }

        // Without h_prev there is no usable r, so the step stays first order.
        let first_order = dpm_plus_plus_2m_step(
            &sample,
            &noise_pred,
            Some(&prev_x0),
            None,
            t,
            t_prev,
            &sched,
        )
        .unwrap();
        let baseline =
            dpm_plus_plus_2m_step(&sample, &noise_pred, None, None, t, t_prev, &sched).unwrap();
        assert_eq!(first_order, baseline);
        assert_ne!(got, baseline, "2M correction must change the result");
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
    fn dpmpp_carries_prev_x0_and_h() {
        let cfg = MultiStepSamplerConfig {
            kind: SamplerKind::DpmPlusPlus2M,
            n_inference_steps: 10,
            schedule: make_cosine_schedule(100),
            guidance_scale: 7.5,
        };
        let mut s = MultiStepSampler::new(cfg).unwrap();
        s.set_timesteps().unwrap();
        let n = 4;
        // Before the first step there is no second-order state.
        assert!(s.prev_x0.is_none() && s.prev_h.is_none());
        for _ in 0..5 {
            if s.is_done() {
                break;
            }
            let _ = s.step(&constant(0.1, n), &constant(0.5, n));
        }
        // The 2M correction needs the previous x₀ *and* the previous step size.
        assert_eq!(s.prev_x0.as_ref().map(|v| v.len()), Some(n));
        assert!(
            s.prev_h.is_some_and(|h| h.is_finite() && h > 0.0),
            "h_prev must be carried forward, got {:?}",
            s.prev_h
        );
        // set_timesteps must clear it so a re-run reproduces the trajectory.
        s.set_timesteps().unwrap();
        assert!(s.prev_x0.is_none() && s.prev_h.is_none());
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

    // -----------------------------------------------------------------------
    // Regression: stochastic DDIM must inject the σ_t·z term it budgets for.
    // -----------------------------------------------------------------------
    #[test]
    fn ddim_eta_positive_injects_noise() {
        let (n, sched) = (8, make_cosine_schedule(100));
        let sample = constant(0.5, n);
        let pred = constant(0.1, n);

        let mut s = default_sampler(SamplerKind::Ddim { eta: 1.0 }, 10);
        let stochastic = s.step(&pred, &sample).unwrap();
        let (t, t_prev) = (s.timesteps()[0], s.timesteps()[1]);

        // Omitting noise is the variance-deficient trajectory: must differ.
        let deficient = ddim_step(&sample, &pred, t, t_prev, &sched, 1.0, None).unwrap();
        assert!(
            stochastic
                .iter()
                .zip(deficient.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "eta > 0 lost its stochastic term"
        );

        // Caller-supplied noise is honoured verbatim.
        let z = constant(0.25, n);
        let mut s2 = default_sampler(SamplerKind::Ddim { eta: 1.0 }, 10);
        assert_eq!(
            s2.step_with_noise(&pred, &sample, &z).unwrap(),
            ddim_step(&sample, &pred, t, t_prev, &sched, 1.0, Some(&z)).unwrap()
        );

        // Same seed reproduces; a different seed diverges.
        let mut s3 = default_sampler(SamplerKind::Ddim { eta: 1.0 }, 10);
        assert_eq!(s3.step(&pred, &sample).unwrap(), stochastic);
        let mut s4 = default_sampler(SamplerKind::Ddim { eta: 1.0 }, 10);
        s4.set_seed(0x1234_5678);
        assert_ne!(s4.step(&pred, &sample).unwrap(), stochastic);
    }

    // -----------------------------------------------------------------------
    // Regression: an empty schedule must not index out of bounds.
    // -----------------------------------------------------------------------
    #[test]
    fn empty_schedule_accessors_do_not_panic() {
        let empty = SamplingNoiseSchedule::cosine(0);
        assert!(empty.alpha_bars.is_empty());
        assert_eq!(empty.alpha_bar_at(0), 1.0);
        assert_eq!(empty.sigma_at(7), 0.0);
        assert!(empty.snr_at(3).is_finite());
        assert_eq!(
            SamplingNoiseSchedule::linear(0, 0.0001, 0.02).sigma_at(0),
            0.0
        );

        let (sample, pred) = (vec![0.5_f32, 0.25], vec![0.1_f32, 0.2]);
        assert!(predict_x0(&sample, &pred, 0, &empty).is_ok());
        assert!(ddim_step(&sample, &pred, 0, 0, &empty, 0.0, None).is_ok());
        assert!(dpm_plus_plus_2m_step(&sample, &pred, None, None, 0, 0, &empty).is_ok());
        assert!(dpm_step_size(0, 0, &empty).is_none());
    }
}
