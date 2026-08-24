//! DDPM (Ho et al. 2020) stochastic reverse-process sampler.
//!
//! Implements beta schedules (linear, cosine, sigmoid, scaled), the
//! full noise schedule precomputation, forward/posterior/reverse step
//! functions, xorshift64+Box-Muller noise generation, dynamic thresholding,
//! and an end-to-end sampling loop.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur in the DDPM sampler.
#[derive(Debug, Error)]
pub enum DdpmError {
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
    #[error("Timestep out of range: t={t}, max={max_t}")]
    TimestepOutOfRange { t: usize, max_t: usize },
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("Numerical error: {0}")]
    NumericalError(String),
    #[error("Empty schedule")]
    EmptySchedule,
    #[error("Invalid noise prediction: {0}")]
    InvalidNoisePrediction(String),
}

// ---------------------------------------------------------------------------
// Beta schedule
// ---------------------------------------------------------------------------

/// Beta schedule variant.
#[derive(Debug, Clone, PartialEq)]
pub enum BetaSchedule {
    /// Linear schedule: beta_t interpolated from `start` to `end`.
    Linear { start: f32, end: f32 },
    /// Cosine schedule (Nichol & Dhariwal 2021).
    Cosine { offset: f32 },
    /// Sigmoid-based schedule. `start`/`end` are the sigmoid *domain*
    /// bounds (e.g. -3.0..3.0), not beta values; `beta_start`/`beta_end`
    /// are the actual output beta range (e.g. 0.0001..0.02).
    Sigmoid {
        start: f32,
        end: f32,
        tau: f32,
        beta_start: f32,
        beta_end: f32,
    },
    /// Scaled linear (for resolution adaptation).
    Scaled { scale: f32 },
}

impl BetaSchedule {
    /// Validate the parameters of this variant, rejecting combinations
    /// that would otherwise produce NaN/Inf betas (and, downstream, NaN/Inf
    /// in every schedule coefficient derived from them).
    pub fn validate(&self) -> Result<(), DdpmError> {
        match self {
            BetaSchedule::Linear { start, end } => {
                if !start.is_finite() || !end.is_finite() {
                    return Err(DdpmError::InvalidConfig(
                        "Linear beta schedule: start/end must be finite".into(),
                    ));
                }
            }
            BetaSchedule::Cosine { offset } => {
                if !offset.is_finite() || *offset <= -1.0 {
                    return Err(DdpmError::InvalidConfig(format!(
                        "Cosine beta schedule: offset must be finite and > -1.0, got {offset}"
                    )));
                }
            }
            BetaSchedule::Sigmoid {
                start,
                end,
                tau,
                beta_start,
                beta_end,
            } => {
                if !tau.is_finite() || *tau == 0.0 {
                    return Err(DdpmError::InvalidConfig(format!(
                        "Sigmoid beta schedule: tau must be finite and non-zero, got {tau}"
                    )));
                }
                if !start.is_finite() || !end.is_finite() || (start - end).abs() < f32::EPSILON {
                    return Err(DdpmError::InvalidConfig(
                        "Sigmoid beta schedule: start must differ from end".into(),
                    ));
                }
                if !beta_start.is_finite() || !beta_end.is_finite() {
                    return Err(DdpmError::InvalidConfig(
                        "Sigmoid beta schedule: beta_start/beta_end must be finite".into(),
                    ));
                }
            }
            BetaSchedule::Scaled { scale } => {
                if !scale.is_finite() || *scale <= 0.0 {
                    return Err(DdpmError::InvalidConfig(format!(
                        "Scaled beta schedule: scale must be finite and > 0.0, got {scale}"
                    )));
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Schedule construction helpers
// ---------------------------------------------------------------------------

/// Linearly interpolated beta schedule from `beta_start` to `beta_end`.
pub fn linear_beta_schedule(num_timesteps: usize, beta_start: f32, beta_end: f32) -> Vec<f32> {
    if num_timesteps == 0 {
        return Vec::new();
    }
    if num_timesteps == 1 {
        return vec![beta_start];
    }
    (0..num_timesteps)
        .map(|i| beta_start + (beta_end - beta_start) * (i as f32) / ((num_timesteps - 1) as f32))
        .collect()
}

/// Cosine beta schedule (Nichol & Dhariwal 2021).
///
/// beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}, clamped to 0.999.
pub fn cosine_beta_schedule(num_timesteps: usize, offset: f32) -> Vec<f32> {
    if num_timesteps == 0 {
        return Vec::new();
    }
    let t_range = num_timesteps as f32;
    // Compute alpha_bar at each t (0..=T)
    let alpha_bar = |t: f32| {
        let frac = (t / t_range + offset) / (1.0 + offset);
        (frac * std::f32::consts::PI * 0.5).cos().powi(2)
    };

    let mut betas = Vec::with_capacity(num_timesteps);
    let alpha_bar_0 = alpha_bar(0.0);
    let mut prev_alpha = alpha_bar_0;
    for t in 1..=(num_timesteps) {
        let cur_alpha = alpha_bar(t as f32);
        let beta = 1.0 - cur_alpha / prev_alpha;
        betas.push(beta.min(0.999_f32));
        prev_alpha = cur_alpha;
    }
    betas
}

/// Sigmoid-based beta schedule.
///
/// `beta_t = beta_start + (sigmoid(t/tau) - sigmoid(start/tau)) /
/// (sigmoid(end/tau) - sigmoid(start/tau)) * (beta_end - beta_start)`,
/// where `t` ranges linearly over `[start, end]` across the
/// `num_timesteps` entries.
///
/// `start`/`end` are the sigmoid *domain* bounds (e.g. -3.0..3.0), not beta
/// values — `beta_start`/`beta_end` are the actual output beta range (e.g.
/// 0.0001..0.02) the result is rescaled into. Without this rescale the
/// output spans the sigmoid's full `[0, 1]` range regardless of
/// `beta_start`/`beta_end`, which is far outside a valid beta range and
/// produces NaN/Inf in every schedule coefficient derived from it.
///
/// Guards against `tau == 0` (falls back to `tau = 1`) and a degenerate,
/// zero-width sigmoid domain (`sigmoid(end/tau) == sigmoid(start/tau)`,
/// which would otherwise divide by ~0) by falling back to linear
/// interpolation over the step index in that case. The result is always
/// clamped to `[1e-6, 0.999]`, a safe beta range.
pub fn sigmoid_beta_schedule(
    num_timesteps: usize,
    start: f32,
    end: f32,
    tau: f32,
    beta_start: f32,
    beta_end: f32,
) -> Vec<f32> {
    if num_timesteps == 0 {
        return Vec::new();
    }
    if num_timesteps == 1 {
        return vec![beta_start.clamp(1e-6, 0.999)];
    }
    let sig = |x: f32| 1.0_f32 / (1.0 + (-x).exp());
    let safe_tau = if tau == 0.0 { 1.0 } else { tau };
    let v_start = sig(start / safe_tau);
    let v_end = sig(end / safe_tau);
    let span = v_end - v_start;

    (0..num_timesteps)
        .map(|i| {
            let t = start + (end - start) * (i as f32) / ((num_timesteps - 1) as f32);
            let frac = if span.abs() < 1e-12 {
                i as f32 / (num_timesteps - 1) as f32
            } else {
                (sig(t / safe_tau) - v_start) / span
            };
            (beta_start + frac * (beta_end - beta_start)).clamp(1e-6, 0.999)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// DdpmSchedule
// ---------------------------------------------------------------------------

/// Precomputed DDPM noise schedule coefficients.
#[derive(Debug, Clone)]
pub struct DdpmSchedule {
    pub num_timesteps: usize,
    pub betas: Vec<f32>,
    pub alphas: Vec<f32>,
    pub alphas_cumprod: Vec<f32>,
    /// ᾱ_{t-1}, with ᾱ_{-1} = 1.
    pub alphas_cumprod_prev: Vec<f32>,
    pub sqrt_alphas_cumprod: Vec<f32>,
    pub sqrt_one_minus_alphas_cumprod: Vec<f32>,
    pub log_one_minus_alphas_cumprod: Vec<f32>,
    /// 1/√ᾱ_t
    pub sqrt_recip_alphas_cumprod: Vec<f32>,
    /// √(1/ᾱ_t − 1)
    pub sqrt_recipm1_alphas_cumprod: Vec<f32>,
    /// β̃_t = β_t (1 − ᾱ_{t-1}) / (1 − ᾱ_t)
    pub posterior_variance: Vec<f32>,
    /// log max(β̃_t, 1e-20)
    pub posterior_log_variance_clipped: Vec<f32>,
    /// c1 in posterior mean: β_t √ᾱ_{t-1} / (1 − ᾱ_t)
    pub posterior_mean_coef1: Vec<f32>,
    /// c2 in posterior mean: √α_t (1 − ᾱ_{t-1}) / (1 − ᾱ_t)
    pub posterior_mean_coef2: Vec<f32>,
}

impl DdpmSchedule {
    /// Build the full schedule from a beta schedule specification.
    pub fn new(num_timesteps: usize, beta_schedule: BetaSchedule) -> Result<Self, DdpmError> {
        if num_timesteps == 0 {
            return Err(DdpmError::EmptySchedule);
        }
        beta_schedule.validate()?;

        let betas = match beta_schedule {
            BetaSchedule::Linear { start, end } => linear_beta_schedule(num_timesteps, start, end),
            BetaSchedule::Cosine { offset } => cosine_beta_schedule(num_timesteps, offset),
            BetaSchedule::Sigmoid {
                start,
                end,
                tau,
                beta_start,
                beta_end,
            } => sigmoid_beta_schedule(num_timesteps, start, end, tau, beta_start, beta_end),
            BetaSchedule::Scaled { scale } => {
                // Scaled linear: betas linearly spaced in sqrt-space then squared.
                let beta_start = (0.0001_f32 * scale).sqrt();
                let beta_end = (0.02_f32 * scale).sqrt();
                if num_timesteps == 1 {
                    vec![beta_start * beta_start]
                } else {
                    (0..num_timesteps)
                        .map(|i| {
                            let v = beta_start
                                + (beta_end - beta_start) * (i as f32)
                                    / ((num_timesteps - 1) as f32);
                            v * v
                        })
                        .collect()
                }
            }
        };

        if betas.len() != num_timesteps {
            return Err(DdpmError::NumericalError(format!(
                "beta length {} != num_timesteps {}",
                betas.len(),
                num_timesteps
            )));
        }

        // Defense in depth: every beta must be finite and in the open (0, 1)
        // interval, or every downstream coefficient (alphas_cumprod,
        // posterior_variance, sqrt_recip_alphas_cumprod, ...) risks NaN/Inf.
        for (i, &b) in betas.iter().enumerate() {
            if !b.is_finite() || b <= 0.0 || b >= 1.0 {
                return Err(DdpmError::NumericalError(format!(
                    "beta[{i}] = {b} is out of the valid (0, 1) range"
                )));
            }
        }

        let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();

        // ᾱ_t = Π_{s=0}^t α_s
        let mut alphas_cumprod = Vec::with_capacity(num_timesteps);
        let mut cp = 1.0_f32;
        for &a in &alphas {
            cp *= a;
            alphas_cumprod.push(cp);
        }

        // ᾱ_{t-1}: shift right, prepend 1.0
        let mut alphas_cumprod_prev = Vec::with_capacity(num_timesteps);
        alphas_cumprod_prev.push(1.0_f32);
        alphas_cumprod_prev.extend_from_slice(&alphas_cumprod[..num_timesteps - 1]);

        let sqrt_alphas_cumprod: Vec<f32> = alphas_cumprod.iter().map(|&a| a.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod: Vec<f32> =
            alphas_cumprod.iter().map(|&a| (1.0 - a).sqrt()).collect();
        let log_one_minus_alphas_cumprod: Vec<f32> =
            alphas_cumprod.iter().map(|&a| (1.0 - a).ln()).collect();
        let sqrt_recip_alphas_cumprod: Vec<f32> =
            alphas_cumprod.iter().map(|&a| (1.0 / a).sqrt()).collect();
        let sqrt_recipm1_alphas_cumprod: Vec<f32> = alphas_cumprod
            .iter()
            .map(|&a| (1.0 / a - 1.0).sqrt())
            .collect();

        // β̃_t = β_t * (1 − ᾱ_{t-1}) / (1 − ᾱ_t)
        let posterior_variance: Vec<f32> = (0..num_timesteps)
            .map(|t| betas[t] * (1.0 - alphas_cumprod_prev[t]) / (1.0 - alphas_cumprod[t]))
            .collect();

        let posterior_log_variance_clipped: Vec<f32> = posterior_variance
            .iter()
            .map(|&v| v.max(1e-20_f32).ln())
            .collect();

        // c1 = β_t * √ᾱ_{t-1} / (1 − ᾱ_t)
        let posterior_mean_coef1: Vec<f32> = (0..num_timesteps)
            .map(|t| betas[t] * alphas_cumprod_prev[t].sqrt() / (1.0 - alphas_cumprod[t]))
            .collect();

        // c2 = √α_t * (1 − ᾱ_{t-1}) / (1 − ᾱ_t)
        let posterior_mean_coef2: Vec<f32> = (0..num_timesteps)
            .map(|t| alphas[t].sqrt() * (1.0 - alphas_cumprod_prev[t]) / (1.0 - alphas_cumprod[t]))
            .collect();

        Ok(Self {
            num_timesteps,
            betas,
            alphas,
            alphas_cumprod,
            alphas_cumprod_prev,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            log_one_minus_alphas_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            posterior_variance,
            posterior_log_variance_clipped,
            posterior_mean_coef1,
            posterior_mean_coef2,
        })
    }

    /// Get β_t.
    pub fn get_beta(&self, t: usize) -> Result<f32, DdpmError> {
        self.betas
            .get(t)
            .copied()
            .ok_or(DdpmError::TimestepOutOfRange {
                t,
                max_t: self.num_timesteps.saturating_sub(1),
            })
    }

    /// Get ᾱ_t.
    pub fn get_alpha_cumprod(&self, t: usize) -> Result<f32, DdpmError> {
        self.alphas_cumprod
            .get(t)
            .copied()
            .ok_or(DdpmError::TimestepOutOfRange {
                t,
                max_t: self.num_timesteps.saturating_sub(1),
            })
    }

    /// Signal-to-noise ratio SNR(t) = ᾱ_t / (1 − ᾱ_t).
    pub fn signal_to_noise_ratio(&self, t: usize) -> Result<f32, DdpmError> {
        let ac = self.get_alpha_cumprod(t)?;
        let denom = 1.0 - ac;
        if denom < f32::EPSILON {
            return Err(DdpmError::NumericalError(
                "SNR denominator is zero (alphas_cumprod ≈ 1)".into(),
            ));
        }
        Ok(ac / denom)
    }
}

// ---------------------------------------------------------------------------
// Sampler configuration
// ---------------------------------------------------------------------------

/// DDPM sampler configuration.
#[derive(Debug, Clone)]
pub struct DdpmSamplerConfig {
    /// Number of inference steps (may be < num_timesteps for strided sampling).
    pub num_inference_steps: usize,
    /// Posterior noise scale factor. The Gaussian noise sampled at each
    /// reverse step (`t > 0`) is multiplied by `eta` before being passed to
    /// [`p_sample`], which always uses the DDPM posterior mean
    /// `c1·x̂₀ + c2·x_t`. `eta = 0.0` therefore yields the DDPM posterior
    /// mean with *no* injected noise (a deterministic trajectory through
    /// posterior means); `eta = 1.0` yields the standard stochastic DDPM
    /// posterior.
    ///
    /// This is **not** the DDIM update (`√ᾱ_{t-1}·x̂₀ + √(1−ᾱ_{t-1})·ε`,
    /// with its own `σ_t` formula) — this sampler implements only DDPM's
    /// stochastic reverse process, with `eta` as a noise-amplitude dial
    /// rather than a DDPM/DDIM interpolation factor.
    pub eta: f32,
    /// Clip x_0 prediction to [−1, 1] after recovery.
    pub clip_sample: bool,
    /// Dynamic thresholding (Saharia 2022) instead of static clipping.
    pub thresholding: bool,
    /// Percentile for dynamic thresholding (default 0.995).
    pub dynamic_threshold_ratio: f32,
    /// RNG seed for noise generation.
    pub seed: u64,
}

impl Default for DdpmSamplerConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 50,
            eta: 1.0,
            clip_sample: true,
            thresholding: false,
            dynamic_threshold_ratio: 0.995,
            seed: 42,
        }
    }
}

impl DdpmSamplerConfig {
    /// Validate configuration fields.
    pub fn validate(&self) -> Result<(), DdpmError> {
        if self.num_inference_steps == 0 {
            return Err(DdpmError::InvalidConfig(
                "num_inference_steps must be > 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.eta) {
            return Err(DdpmError::InvalidConfig(format!(
                "eta must be in [0, 1], got {}",
                self.eta
            )));
        }
        if !(0.0..=1.0).contains(&self.dynamic_threshold_ratio) {
            return Err(DdpmError::InvalidConfig(format!(
                "dynamic_threshold_ratio must be in [0, 1], got {}",
                self.dynamic_threshold_ratio
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Noise generation (xorshift64 + Box-Muller)
// ---------------------------------------------------------------------------

/// xorshift64 step — returns the next state value.
#[inline]
fn xorshift64(state: u64) -> u64 {
    let mut x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Fill `out` with standard-normal samples using xorshift64 + Box-Muller,
/// advancing `state` in place.
///
/// Unlike [`generate_noise`] (which always restarts from a fresh seed),
/// repeated calls sharing the same `state` draw from disjoint windows of
/// one continuous PRNG stream, so consecutive draws do not overlap. This is
/// what [`sample`] uses internally so every reverse step gets independent
/// noise instead of a near-copy of the previous step's draw.
fn fill_noise(out: &mut [f32], state: &mut u64) {
    if *state == 0 {
        *state = 1;
    }
    let n = out.len();
    let mut i = 0usize;
    while i < n {
        // Uniform u1 in (0, 1)
        *state = xorshift64(*state);
        let u1 = *state as f64 / u64::MAX as f64;
        *state = xorshift64(*state);
        let u2 = *state as f64 / u64::MAX as f64;

        // Guard against ln(0)
        let u1 = u1.max(f64::EPSILON);
        let u2 = u2.max(f64::EPSILON);

        let mag = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;

        out[i] = (mag * theta.cos()) as f32;
        i += 1;

        if i < n {
            out[i] = (mag * theta.sin()) as f32;
            i += 1;
        }
    }
}

/// Generate `n` standard-normal samples using xorshift64 + Box-Muller.
///
/// The seed guard `state = state.max(1)` is applied before first use.
pub fn generate_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.max(1);
    let mut out = vec![0.0_f32; n];
    fill_noise(&mut out, &mut state);
    out
}

// ---------------------------------------------------------------------------
// Forward process
// ---------------------------------------------------------------------------

/// Forward diffusion: x_t = √ᾱ_t * x_0 + √(1−ᾱ_t) * ε.
///
/// `noise` must be the same length as `x_0`.
pub fn q_sample(
    x_0: &[f32],
    t: usize,
    schedule: &DdpmSchedule,
    noise: &[f32],
) -> Result<Vec<f32>, DdpmError> {
    if noise.len() != x_0.len() {
        return Err(DdpmError::DimensionMismatch {
            expected: x_0.len(),
            actual: noise.len(),
        });
    }
    let sqrt_ac =
        schedule
            .sqrt_alphas_cumprod
            .get(t)
            .copied()
            .ok_or(DdpmError::TimestepOutOfRange {
                t,
                max_t: schedule.num_timesteps.saturating_sub(1),
            })?;
    let sqrt_one_minus = schedule
        .sqrt_one_minus_alphas_cumprod
        .get(t)
        .copied()
        .ok_or(DdpmError::TimestepOutOfRange {
            t,
            max_t: schedule.num_timesteps.saturating_sub(1),
        })?;

    Ok(x_0
        .iter()
        .zip(noise.iter())
        .map(|(&x, &n)| sqrt_ac * x + sqrt_one_minus * n)
        .collect())
}

// ---------------------------------------------------------------------------
// Posterior q(x_{t-1} | x_t, x_0)
// ---------------------------------------------------------------------------

/// Posterior mean and variance for q(x_{t-1} | x_t, x_0).
///
/// Returns `(mean, variance, log_variance)`.
pub fn q_posterior_mean_variance(
    x_0: &[f32],
    x_t: &[f32],
    t: usize,
    schedule: &DdpmSchedule,
) -> Result<(Vec<f32>, f32, f32), DdpmError> {
    if x_0.len() != x_t.len() {
        return Err(DdpmError::DimensionMismatch {
            expected: x_0.len(),
            actual: x_t.len(),
        });
    }
    let c1 =
        schedule
            .posterior_mean_coef1
            .get(t)
            .copied()
            .ok_or(DdpmError::TimestepOutOfRange {
                t,
                max_t: schedule.num_timesteps.saturating_sub(1),
            })?;
    let c2 =
        schedule
            .posterior_mean_coef2
            .get(t)
            .copied()
            .ok_or(DdpmError::TimestepOutOfRange {
                t,
                max_t: schedule.num_timesteps.saturating_sub(1),
            })?;
    let variance = schedule.posterior_variance[t];
    let log_variance = schedule.posterior_log_variance_clipped[t];

    let mean: Vec<f32> = x_0
        .iter()
        .zip(x_t.iter())
        .map(|(&x0, &xt)| c1 * x0 + c2 * xt)
        .collect();

    Ok((mean, variance, log_variance))
}

// ---------------------------------------------------------------------------
// Predict x_0 from noise prediction
// ---------------------------------------------------------------------------

/// Predict x_0 from model noise prediction ε_θ(x_t, t).
///
/// x_0 = (1/√ᾱ_t) * x_t − √(1/ᾱ_t − 1) * ε_θ
pub fn predict_x_0_from_noise(
    x_t: &[f32],
    noise_pred: &[f32],
    t: usize,
    schedule: &DdpmSchedule,
) -> Result<Vec<f32>, DdpmError> {
    if noise_pred.len() != x_t.len() {
        return Err(DdpmError::DimensionMismatch {
            expected: x_t.len(),
            actual: noise_pred.len(),
        });
    }
    let recip = schedule.sqrt_recip_alphas_cumprod.get(t).copied().ok_or(
        DdpmError::TimestepOutOfRange {
            t,
            max_t: schedule.num_timesteps.saturating_sub(1),
        },
    )?;
    let recipm1 = schedule.sqrt_recipm1_alphas_cumprod.get(t).copied().ok_or(
        DdpmError::TimestepOutOfRange {
            t,
            max_t: schedule.num_timesteps.saturating_sub(1),
        },
    )?;

    Ok(x_t
        .iter()
        .zip(noise_pred.iter())
        .map(|(&xt, &eps)| recip * xt - recipm1 * eps)
        .collect())
}

// ---------------------------------------------------------------------------
// Reverse step p(x_{t-1} | x_t)
// ---------------------------------------------------------------------------

/// One stochastic DDPM reverse step.
///
/// `noise` should be `None` when `t == 0` (no noise at the final step).
pub fn p_sample(
    x_t: &[f32],
    noise_pred: &[f32],
    t: usize,
    schedule: &DdpmSchedule,
    noise: Option<&[f32]>,
) -> Result<Vec<f32>, DdpmError> {
    if noise_pred.len() != x_t.len() {
        return Err(DdpmError::DimensionMismatch {
            expected: x_t.len(),
            actual: noise_pred.len(),
        });
    }
    if let Some(n) = noise {
        if n.len() != x_t.len() {
            return Err(DdpmError::DimensionMismatch {
                expected: x_t.len(),
                actual: n.len(),
            });
        }
    }

    // Recover x_0 from noise prediction
    let x_0_pred = predict_x_0_from_noise(x_t, noise_pred, t, schedule)?;

    // Posterior mean
    let (mean, variance, _log_var) = q_posterior_mean_variance(&x_0_pred, x_t, t, schedule)?;

    // Add stochastic noise (skip at t=0)
    if t == 0 {
        return Ok(mean);
    }
    let std_dev = variance.sqrt();
    match noise {
        Some(n) => Ok(mean
            .iter()
            .zip(n.iter())
            .map(|(&m, &ni)| m + std_dev * ni)
            .collect()),
        None => Ok(mean),
    }
}

// ---------------------------------------------------------------------------
// Thresholding
// ---------------------------------------------------------------------------

/// Dynamic thresholding (Saharia 2022): percentile absolute-value clipping then rescale.
pub fn dynamic_threshold(x: &[f32], ratio: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let mut abs_vals: Vec<f32> = x.iter().map(|&v| v.abs()).collect();
    // `f32::total_cmp` is a genuine total order (unlike `partial_cmp` with an
    // `Equal` fallback for incomparable values, which is not transitive and
    // panics on current Rust's total-order-checking sort implementation as
    // soon as a NaN appears — exactly the untrusted-input case dynamic
    // thresholding is meant to survive).
    abs_vals.sort_by(f32::total_cmp);

    let idx = ((x.len() as f32) * ratio) as usize;
    let idx = idx.min(x.len() - 1);
    let threshold = abs_vals[idx].max(1.0_f32);

    x.iter()
        .map(|&v| (v / threshold).clamp(-1.0, 1.0))
        .collect()
}

/// Static clip to [−1, 1].
pub fn clip_sample(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.clamp(-1.0, 1.0)).collect()
}

// ---------------------------------------------------------------------------
// Timestep schedule for inference
// ---------------------------------------------------------------------------

/// Compute a descending timestep sequence for inference.
///
/// Returns `num_inference_steps` values in descending order, including 0.
pub fn inference_timesteps(num_train_steps: usize, num_inference_steps: usize) -> Vec<usize> {
    if num_inference_steps == 0 || num_train_steps == 0 {
        return Vec::new();
    }
    let n = num_inference_steps.min(num_train_steps);
    let stride = num_train_steps / n;
    // Generate n values: stride*(n-1), stride*(n-2), ..., 0
    (0..n).rev().map(|i| i * stride).collect()
}

// ---------------------------------------------------------------------------
// Full sampling loop
// ---------------------------------------------------------------------------

/// Full DDPM sampling loop.
///
/// Starts from Gaussian noise and iterates the reverse process using
/// `noise_model(x_t, timestep) -> noise_prediction`.
pub fn sample<F>(
    shape: usize,
    schedule: &DdpmSchedule,
    config: &DdpmSamplerConfig,
    noise_model: F,
) -> Result<Vec<f32>, DdpmError>
where
    F: Fn(&[f32], usize) -> Vec<f32>,
{
    config.validate()?;

    let timesteps = inference_timesteps(schedule.num_timesteps, config.num_inference_steps);
    if timesteps.is_empty() {
        return Err(DdpmError::InvalidConfig(
            "inference_timesteps produced an empty list".into(),
        ));
    }

    // Initialise x_T ~ N(0, I). A single PRNG state is threaded through the
    // whole routine (see `fill_noise`): every draw below advances it in
    // place, so the initial noise and every reverse step's noise come from
    // disjoint windows of one continuous stream instead of each step
    // re-seeding from a value only one xorshift64 application away from
    // the previous step's seed (which used to make consecutive steps'
    // "noise" nearly identical).
    let mut state = config.seed.max(1);
    let mut x_t = vec![0.0_f32; shape];
    fill_noise(&mut x_t, &mut state);

    for &t in &timesteps {
        // Get model noise prediction
        let noise_pred = noise_model(&x_t, t);

        if noise_pred.len() != shape {
            return Err(DdpmError::DimensionMismatch {
                expected: shape,
                actual: noise_pred.len(),
            });
        }

        // Optionally predict and clip x_0
        let noise_pred = if config.clip_sample || config.thresholding {
            let x_0 = predict_x_0_from_noise(&x_t, &noise_pred, t, schedule)?;
            let x_0 = if config.thresholding {
                dynamic_threshold(&x_0, config.dynamic_threshold_ratio)
            } else {
                clip_sample(&x_0)
            };
            // Re-derive noise_pred from clipped x_0: ε = (x_t − √ᾱ_t * x_0) / √(1−ᾱ_t)
            let sqrt_ac = schedule.sqrt_alphas_cumprod[t];
            let sqrt_om = schedule.sqrt_one_minus_alphas_cumprod[t];
            if sqrt_om < f32::EPSILON {
                noise_pred
            } else {
                x_t.iter()
                    .zip(x_0.iter())
                    .map(|(&xt, &x0)| (xt - sqrt_ac * x0) / sqrt_om)
                    .collect()
            }
        } else {
            noise_pred
        };

        // Step noise for t > 0, drawn from the shared PRNG stream so it
        // never overlaps the initial x_T draw or any previous step's draw.
        let step_noise = if t > 0 && config.eta > 0.0 {
            let mut raw = vec![0.0_f32; shape];
            fill_noise(&mut raw, &mut state);
            Some(
                raw.into_iter()
                    .map(|v| v * config.eta)
                    .collect::<Vec<f32>>(),
            )
        } else {
            None
        };

        x_t = p_sample(&x_t, &noise_pred, t, schedule, step_noise.as_deref())?;
    }

    Ok(x_t)
}

// ---------------------------------------------------------------------------
// SNR-weighted loss
// ---------------------------------------------------------------------------

/// SNR-weighted (or simple L2) loss for DDPM training.
///
/// With `use_snr_weighting = false`: MSE(noise_pred, noise_target).
/// With `use_snr_weighting = true`: SNR(t) * MSE(noise_pred, noise_target).
pub fn snr_weighted_loss(
    noise_pred: &[f32],
    noise_target: &[f32],
    t: usize,
    schedule: &DdpmSchedule,
    use_snr_weighting: bool,
) -> Result<f32, DdpmError> {
    if noise_pred.len() != noise_target.len() {
        return Err(DdpmError::DimensionMismatch {
            expected: noise_target.len(),
            actual: noise_pred.len(),
        });
    }
    if noise_pred.is_empty() {
        return Err(DdpmError::DimensionMismatch {
            expected: 1,
            actual: 0,
        });
    }

    let mse: f32 = noise_pred
        .iter()
        .zip(noise_target.iter())
        .map(|(&p, &g)| (p - g).powi(2))
        .sum::<f32>()
        / noise_pred.len() as f32;

    if use_snr_weighting {
        let snr = schedule.signal_to_noise_ratio(t)?;
        Ok(snr * mse)
    } else {
        Ok(mse)
    }
}

// ---------------------------------------------------------------------------
// Schedule statistics
// ---------------------------------------------------------------------------

/// Summary statistics for a noise schedule.
#[derive(Debug, Clone)]
pub struct ScheduleStats {
    pub min_beta: f32,
    pub max_beta: f32,
    pub min_snr: f32,
    pub max_snr: f32,
    /// First timestep where SNR crosses below 1.0 (if any).
    pub zero_crossing_t: Option<usize>,
}

/// Compute summary statistics for a schedule.
pub fn compute_schedule_stats(schedule: &DdpmSchedule) -> ScheduleStats {
    let min_beta = schedule.betas.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_beta = schedule
        .betas
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let mut min_snr = f32::INFINITY;
    let mut max_snr = f32::NEG_INFINITY;
    let mut zero_crossing_t: Option<usize> = None;

    for t in 0..schedule.num_timesteps {
        let ac = schedule.alphas_cumprod[t];
        let denom = 1.0 - ac;
        if denom < f32::EPSILON {
            continue;
        }
        let snr = ac / denom;
        if snr < min_snr {
            min_snr = snr;
        }
        if snr > max_snr {
            max_snr = snr;
        }
        if zero_crossing_t.is_none() && snr < 1.0 {
            zero_crossing_t = Some(t);
        }
    }

    if min_snr == f32::INFINITY {
        min_snr = 0.0;
    }
    if max_snr == f32::NEG_INFINITY {
        max_snr = 0.0;
    }

    ScheduleStats {
        min_beta,
        max_beta,
        min_snr,
        max_snr,
        zero_crossing_t,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    // --- linear_beta_schedule ---

    #[test]
    fn test_linear_beta_length() {
        let betas = linear_beta_schedule(100, 0.0001, 0.02);
        assert_eq!(betas.len(), 100);
    }

    #[test]
    fn test_linear_beta_first_last() {
        let betas = linear_beta_schedule(1000, 0.0001, 0.02);
        assert!((betas[0] - 0.0001).abs() < EPSILON);
        assert!((betas[999] - 0.02).abs() < EPSILON);
    }

    #[test]
    fn test_linear_beta_monotone_increasing() {
        let betas = linear_beta_schedule(100, 0.0001, 0.02);
        for w in betas.windows(2) {
            assert!(w[1] >= w[0], "betas must be non-decreasing");
        }
    }

    #[test]
    fn test_linear_beta_empty() {
        let betas = linear_beta_schedule(0, 0.0001, 0.02);
        assert!(betas.is_empty());
    }

    // --- cosine_beta_schedule ---

    #[test]
    fn test_cosine_beta_length() {
        let betas = cosine_beta_schedule(1000, 0.008);
        assert_eq!(betas.len(), 1000);
    }

    #[test]
    fn test_cosine_beta_valid_range() {
        let betas = cosine_beta_schedule(1000, 0.008);
        for &b in &betas {
            assert!(b > 0.0 && b <= 0.999, "beta={} out of range", b);
        }
    }

    #[test]
    fn test_cosine_beta_offset_effect() {
        let b1 = cosine_beta_schedule(100, 0.008);
        let b2 = cosine_beta_schedule(100, 0.02);
        // Different offsets should produce different schedules
        assert!(b1.iter().zip(b2.iter()).any(|(a, b)| (a - b).abs() > 1e-6));
    }

    // --- sigmoid_beta_schedule ---

    #[test]
    fn test_sigmoid_beta_length() {
        let betas = sigmoid_beta_schedule(100, -3.0, 3.0, 1.0, 0.0001, 0.02);
        assert_eq!(betas.len(), 100);
    }

    #[test]
    fn test_sigmoid_beta_valid_range() {
        let betas = sigmoid_beta_schedule(100, -3.0, 3.0, 1.0, 0.0001, 0.02);
        for &b in &betas {
            assert!(
                b >= 0.0001 - 1e-6 && b <= 0.02 + 1e-6,
                "beta={b} out of expected [beta_start, beta_end] range"
            );
        }
    }

    #[test]
    fn test_sigmoid_beta_schedule_respects_beta_range() {
        // Regression test: the schedule must be rescaled into
        // [beta_start, beta_end], not span the full sigmoid [0, 1] range
        // (which made betas[0] == 0.0 and betas[last] == 1.0, producing
        // NaN/Inf schedule coefficients downstream).
        let betas = sigmoid_beta_schedule(1000, -6.0, 6.0, 1.0, 0.0001, 0.02);
        assert!(
            (betas[0] - 0.0001).abs() < 1e-3,
            "betas[0] should be near beta_start=0.0001, got {}",
            betas[0]
        );
        assert!(
            (betas[betas.len() - 1] - 0.02).abs() < 1e-3,
            "betas[last] should be near beta_end=0.02, got {}",
            betas[betas.len() - 1]
        );
        for &b in &betas {
            assert!(
                b.is_finite() && b > 0.0,
                "beta must be finite and positive, got {b}"
            );
        }
    }

    #[test]
    fn test_sigmoid_beta_schedule_zero_tau_no_nan() {
        // tau == 0 must not produce NaN/Inf (guarded internally, even
        // though DdpmSchedule::new additionally rejects it via validate()).
        let betas = sigmoid_beta_schedule(50, -3.0, 3.0, 0.0, 0.0001, 0.02);
        assert_eq!(betas.len(), 50);
        assert!(betas.iter().all(|b| b.is_finite()));
    }

    // --- DdpmSchedule::new ---

    #[test]
    fn test_schedule_linear_new() {
        let sched = DdpmSchedule::new(
            1000,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        assert_eq!(sched.num_timesteps, 1000);
        assert_eq!(sched.betas.len(), 1000);
        assert_eq!(sched.alphas_cumprod.len(), 1000);
    }

    #[test]
    fn test_schedule_cosine_new() {
        let sched = DdpmSchedule::new(1000, BetaSchedule::Cosine { offset: 0.008 })
            .expect("should succeed");
        assert_eq!(sched.num_timesteps, 1000);
        // alphas_cumprod should be strictly decreasing
        for w in sched.alphas_cumprod.windows(2) {
            assert!(w[1] < w[0], "alphas_cumprod must be decreasing");
        }
    }

    #[test]
    fn test_schedule_empty_error() {
        let result = DdpmSchedule::new(
            0,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_schedule_alphas_cumprod_prev() {
        let sched = DdpmSchedule::new(
            100,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        // First element must be 1.0 (ᾱ_{-1})
        assert!((sched.alphas_cumprod_prev[0] - 1.0).abs() < EPSILON);
        // Element [t] == alphas_cumprod[t-1] for t >= 1
        for t in 1..sched.num_timesteps {
            assert!(
                (sched.alphas_cumprod_prev[t] - sched.alphas_cumprod[t - 1]).abs() < EPSILON,
                "prev mismatch at t={t}"
            );
        }
    }

    #[test]
    fn test_schedule_posterior_variance_t0_is_zero() {
        let sched = DdpmSchedule::new(
            100,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        // At t=0, alphas_cumprod_prev = 1.0, so (1 - 1.0) = 0, posterior_variance = 0.
        assert!(sched.posterior_variance[0].abs() < EPSILON);
    }

    // --- BetaSchedule::validate / DdpmSchedule::new parameter guards ---

    #[test]
    fn test_schedule_cosine_offset_minus_one_rejected() {
        let result = DdpmSchedule::new(10, BetaSchedule::Cosine { offset: -1.0 });
        assert!(result.is_err());
    }

    #[test]
    fn test_schedule_sigmoid_tau_zero_rejected() {
        let result = DdpmSchedule::new(
            10,
            BetaSchedule::Sigmoid {
                start: -3.0,
                end: 3.0,
                tau: 0.0,
                beta_start: 0.0001,
                beta_end: 0.02,
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_schedule_sigmoid_start_equals_end_rejected() {
        let result = DdpmSchedule::new(
            10,
            BetaSchedule::Sigmoid {
                start: 1.0,
                end: 1.0,
                tau: 1.0,
                beta_start: 0.0001,
                beta_end: 0.02,
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_schedule_scaled_negative_scale_rejected() {
        let result = DdpmSchedule::new(10, BetaSchedule::Scaled { scale: -1.0 });
        assert!(result.is_err());
    }

    #[test]
    fn test_schedule_scaled_zero_scale_rejected() {
        let result = DdpmSchedule::new(10, BetaSchedule::Scaled { scale: 0.0 });
        assert!(result.is_err());
    }

    #[test]
    fn test_schedule_scaled_num_timesteps_one_no_nan() {
        let sched =
            DdpmSchedule::new(1, BetaSchedule::Scaled { scale: 1.0 }).expect("should succeed");
        assert_eq!(sched.betas.len(), 1);
        assert!(sched.betas[0].is_finite());
    }

    #[test]
    fn test_schedule_sigmoid_new_no_nan_or_inf() {
        // End-to-end regression test for the sigmoid rescale fix: the full
        // derived schedule must be free of NaN/Inf.
        let sched = DdpmSchedule::new(
            200,
            BetaSchedule::Sigmoid {
                start: -3.0,
                end: 3.0,
                tau: 1.0,
                beta_start: 0.0001,
                beta_end: 0.02,
            },
        )
        .expect("should succeed");
        for &v in &sched.alphas_cumprod {
            assert!(
                v.is_finite() && v > 0.0,
                "alphas_cumprod must be finite and positive, got {v}"
            );
        }
        for &v in &sched.posterior_variance {
            assert!(v.is_finite(), "posterior_variance must be finite, got {v}");
        }
        for &v in &sched.sqrt_recip_alphas_cumprod {
            assert!(
                v.is_finite(),
                "sqrt_recip_alphas_cumprod must be finite, got {v}"
            );
        }
    }

    // --- get_beta / get_alpha_cumprod ---

    #[test]
    fn test_get_beta_in_range() {
        let sched = DdpmSchedule::new(
            10,
            BetaSchedule::Linear {
                start: 0.001,
                end: 0.01,
            },
        )
        .expect("should succeed");
        let b = sched.get_beta(5).expect("should succeed");
        assert!(b > 0.0 && b < 1.0);
    }

    #[test]
    fn test_get_beta_out_of_range() {
        let sched = DdpmSchedule::new(
            10,
            BetaSchedule::Linear {
                start: 0.001,
                end: 0.01,
            },
        )
        .expect("should succeed");
        assert!(sched.get_beta(10).is_err());
    }

    #[test]
    fn test_get_alpha_cumprod_out_of_range() {
        let sched = DdpmSchedule::new(
            10,
            BetaSchedule::Linear {
                start: 0.001,
                end: 0.01,
            },
        )
        .expect("should succeed");
        assert!(sched.get_alpha_cumprod(10).is_err());
    }

    // --- signal_to_noise_ratio ---

    #[test]
    fn test_snr_t0_large() {
        let sched = DdpmSchedule::new(
            1000,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        // At t=0, ᾱ close to 1, SNR >> 1
        let snr = sched.signal_to_noise_ratio(0).expect("should succeed");
        assert!(snr > 10.0, "SNR at t=0 should be large, got {}", snr);
    }

    #[test]
    fn test_snr_decreasing() {
        let sched = DdpmSchedule::new(
            100,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        let snr_early = sched.signal_to_noise_ratio(0).expect("should succeed");
        let snr_late = sched.signal_to_noise_ratio(99).expect("should succeed");
        assert!(snr_early > snr_late);
    }

    // --- q_sample ---

    #[test]
    fn test_q_sample_zero_noise_t0() {
        let sched = DdpmSchedule::new(
            1000,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        let x_0 = vec![1.0f32, 2.0, 3.0];
        let noise = vec![0.0f32; 3];
        let x_t = q_sample(&x_0, 0, &sched, &noise).expect("should succeed");
        // At t=0, √ᾱ_0 ≈ 1, so x_t ≈ x_0
        let sqrt_ac = sched.sqrt_alphas_cumprod[0];
        for (xt, &x0) in x_t.iter().zip(x_0.iter()) {
            assert!(
                (xt - sqrt_ac * x0).abs() < EPSILON,
                "q_sample mismatch at t=0"
            );
        }
    }

    #[test]
    fn test_q_sample_shape_preserved() {
        let sched = DdpmSchedule::new(
            100,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        let x_0 = vec![0.5f32; 64];
        let noise = vec![0.1f32; 64];
        let x_t = q_sample(&x_0, 50, &sched, &noise).expect("should succeed");
        assert_eq!(x_t.len(), 64);
    }

    #[test]
    fn test_q_sample_dimension_mismatch() {
        let sched = DdpmSchedule::new(
            10,
            BetaSchedule::Linear {
                start: 0.001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        let x_0 = vec![1.0f32; 4];
        let noise = vec![0.0f32; 5];
        assert!(q_sample(&x_0, 0, &sched, &noise).is_err());
    }

    // --- q_posterior_mean_variance ---

    #[test]
    fn test_q_posterior_shape() {
        let sched = DdpmSchedule::new(
            100,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        let x_0 = vec![0.5f32; 8];
        let x_t = vec![0.3f32; 8];
        let (mean, var, log_var) =
            q_posterior_mean_variance(&x_0, &x_t, 50, &sched).expect("should succeed");
        assert_eq!(mean.len(), 8);
        assert!(var >= 0.0, "variance must be non-negative");
        assert!(log_var.is_finite());
    }

    #[test]
    fn test_q_posterior_formula() {
        let sched = DdpmSchedule::new(
            100,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        let t = 50usize;
        let x_0 = vec![1.0f32];
        let x_t = vec![0.5f32];
        let (mean, _, _) =
            q_posterior_mean_variance(&x_0, &x_t, t, &sched).expect("should succeed");
        let expected =
            sched.posterior_mean_coef1[t] * x_0[0] + sched.posterior_mean_coef2[t] * x_t[0];
        assert!((mean[0] - expected).abs() < EPSILON);
    }

    // --- predict_x_0_from_noise ---

    #[test]
    fn test_predict_x0_from_zero_noise() {
        let sched = DdpmSchedule::new(
            1000,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        // If noise prediction is 0, x_0 ≈ (1/√ᾱ) * x_t
        let t = 0usize;
        let x_t = vec![0.5f32];
        let noise_pred = vec![0.0f32];
        let x_0 = predict_x_0_from_noise(&x_t, &noise_pred, t, &sched).expect("should succeed");
        let expected = sched.sqrt_recip_alphas_cumprod[t] * x_t[0];
        assert!((x_0[0] - expected).abs() < EPSILON);
    }

    #[test]
    fn test_predict_x0_shape_preserved() {
        let sched = DdpmSchedule::new(
            100,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        let x_t = vec![0.1f32; 32];
        let noise_pred = vec![0.05f32; 32];
        let x_0 = predict_x_0_from_noise(&x_t, &noise_pred, 50, &sched).expect("should succeed");
        assert_eq!(x_0.len(), 32);
    }

    // --- p_sample ---

    #[test]
    fn test_p_sample_shape_preserved() {
        let sched = DdpmSchedule::new(
            100,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        let x_t = vec![0.3f32; 16];
        let noise_pred = vec![0.1f32; 16];
        let noise = vec![0.0f32; 16];
        let x_prev = p_sample(&x_t, &noise_pred, 50, &sched, Some(&noise)).expect("should succeed");
        assert_eq!(x_prev.len(), 16);
    }

    #[test]
    fn test_p_sample_t0_no_noise_added() {
        let sched = DdpmSchedule::new(
            100,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        let x_t = vec![0.5f32; 4];
        let noise_pred = vec![0.1f32; 4];
        // With noise provided but t=0, noise should not be added
        let large_noise = vec![1000.0f32; 4];
        let x_prev_with_noise =
            p_sample(&x_t, &noise_pred, 0, &sched, Some(&large_noise)).expect("should succeed");
        let x_prev_no_noise = p_sample(&x_t, &noise_pred, 0, &sched, None).expect("should succeed");
        for (a, b) in x_prev_with_noise.iter().zip(x_prev_no_noise.iter()) {
            assert!((a - b).abs() < EPSILON, "t=0 must not add noise");
        }
    }

    #[test]
    fn test_p_sample_deterministic_with_same_noise() {
        let sched = DdpmSchedule::new(
            100,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        let x_t = vec![0.3f32; 8];
        let noise_pred = vec![0.1f32; 8];
        let noise = vec![0.5f32; 8];
        let r1 = p_sample(&x_t, &noise_pred, 50, &sched, Some(&noise)).expect("should succeed");
        let r2 = p_sample(&x_t, &noise_pred, 50, &sched, Some(&noise)).expect("should succeed");
        assert_eq!(r1, r2);
    }

    // --- generate_noise ---

    #[test]
    fn test_generate_noise_length() {
        let n = generate_noise(100, 42);
        assert_eq!(n.len(), 100);
    }

    #[test]
    fn test_generate_noise_nonzero() {
        let n = generate_noise(64, 1234);
        let all_zero = n.iter().all(|&v| v == 0.0);
        assert!(!all_zero);
    }

    #[test]
    fn test_generate_noise_reproducible() {
        let n1 = generate_noise(128, 99);
        let n2 = generate_noise(128, 99);
        assert_eq!(n1, n2);
    }

    #[test]
    fn test_generate_noise_different_seeds() {
        let n1 = generate_noise(64, 1);
        let n2 = generate_noise(64, 2);
        assert_ne!(n1, n2);
    }

    #[test]
    fn test_generate_noise_zero_seed_guard() {
        // seed=0 should not panic (guarded to max(seed, 1))
        let n = generate_noise(16, 0);
        assert_eq!(n.len(), 16);
    }

    // --- fill_noise (shared-state threading used by `sample`) ---

    #[test]
    fn test_fill_noise_advances_shared_state() {
        let mut state = 42_u64;
        let mut out = vec![0.0_f32; 8];
        fill_noise(&mut out, &mut state);
        assert_ne!(state, 42_u64, "fill_noise must advance the threaded state");
        assert!(out.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_fill_noise_consecutive_draws_not_a_shifted_copy() {
        // Regression test for the DDPM `sample()` bug: re-seeding
        // `generate_noise` from a seed only one or two xorshift64
        // applications away from the previous seed made each reverse
        // step's "noise" vector nearly identical to the previous one,
        // shifted by ~2 elements. Two `fill_noise` calls sharing one
        // threaded `state` must NOT exhibit that pattern: the second draw
        // should not line up with the first one shifted left by 2.
        let mut state = 42_u64;
        let mut first = vec![0.0_f32; 64];
        fill_noise(&mut first, &mut state);
        let mut second = vec![0.0_f32; 64];
        fill_noise(&mut second, &mut state);

        let shifted_matches = first[2..]
            .iter()
            .zip(second.iter())
            .filter(|(&a, &b)| (a - b).abs() < 1e-4)
            .count();
        assert!(
            shifted_matches < first.len() / 4,
            "second draw should not be a left-shift-by-2 copy of the first \
             (this was the DDPM sample() re-seeding bug's signature): \
             {shifted_matches} matching elements"
        );
    }

    // --- dynamic_threshold ---

    #[test]
    fn test_dynamic_threshold_clamped() {
        let x: Vec<f32> = (0..100).map(|i| i as f32 - 50.0).collect();
        let result = dynamic_threshold(&x, 0.995);
        for &v in &result {
            assert!((-1.0..=1.0).contains(&v), "value {} out of range", v);
        }
    }

    #[test]
    fn test_dynamic_threshold_empty() {
        let result = dynamic_threshold(&[], 0.995);
        assert!(result.is_empty());
    }

    #[test]
    fn test_dynamic_threshold_nan_input_does_not_panic() {
        // dynamic_threshold is called on the model's x_0 prediction, which
        // may contain NaN after a diverging step. `partial_cmp` with an
        // `Equal` fallback for NaN is not a total order and panics on
        // current Rust's sort; `total_cmp` must not.
        let mut x: Vec<f32> = (0..20).map(|i| i as f32 - 10.0).collect();
        x[5] = f32::NAN;
        x[15] = f32::NAN;
        let result = dynamic_threshold(&x, 0.995);
        assert_eq!(result.len(), x.len());
    }

    // --- clip_sample ---

    #[test]
    fn test_clip_sample_bounds() {
        let x = vec![-5.0f32, -1.0, 0.0, 1.0, 5.0];
        let clipped = clip_sample(&x);
        for &v in &clipped {
            assert!((-1.0..=1.0).contains(&v));
        }
        assert!((clipped[0] - (-1.0)).abs() < EPSILON);
        assert!((clipped[4] - 1.0).abs() < EPSILON);
    }

    // --- inference_timesteps ---

    #[test]
    fn test_inference_timesteps_count() {
        let ts = inference_timesteps(1000, 50);
        assert_eq!(ts.len(), 50);
    }

    #[test]
    fn test_inference_timesteps_includes_zero() {
        let ts = inference_timesteps(1000, 50);
        assert_eq!(*ts.last().expect("non-empty"), 0);
    }

    #[test]
    fn test_inference_timesteps_monotone_decreasing() {
        let ts = inference_timesteps(1000, 50);
        for w in ts.windows(2) {
            assert!(w[0] > w[1], "timesteps must be strictly decreasing");
        }
    }

    #[test]
    fn test_inference_timesteps_zero_steps() {
        let ts = inference_timesteps(1000, 0);
        assert!(ts.is_empty());
    }

    // --- snr_weighted_loss ---

    #[test]
    fn test_snr_loss_identical_pred_zero() {
        let sched = DdpmSchedule::new(
            100,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        let pred = vec![0.5f32, -0.3, 0.1];
        let loss = snr_weighted_loss(&pred, &pred, 50, &sched, false).expect("should succeed");
        assert!(loss.abs() < EPSILON, "identical pred/target loss must be 0");
    }

    #[test]
    fn test_snr_loss_weighting_larger() {
        let sched = DdpmSchedule::new(
            100,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        let pred = vec![0.5f32; 4];
        let target = vec![0.0f32; 4];
        // At early timesteps SNR > 1, so SNR-weighted loss > simple loss
        let simple = snr_weighted_loss(&pred, &target, 0, &sched, false).expect("should succeed");
        let weighted = snr_weighted_loss(&pred, &target, 0, &sched, true).expect("should succeed");
        assert!(weighted > simple, "SNR-weighted loss must be larger at t=0");
    }

    #[test]
    fn test_snr_loss_dimension_mismatch() {
        let sched = DdpmSchedule::new(
            10,
            BetaSchedule::Linear {
                start: 0.001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        let pred = vec![0.5f32; 4];
        let target = vec![0.0f32; 5];
        assert!(snr_weighted_loss(&pred, &target, 0, &sched, false).is_err());
    }

    // --- compute_schedule_stats ---

    #[test]
    fn test_schedule_stats_min_max_beta() {
        let sched = DdpmSchedule::new(
            100,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        let stats = compute_schedule_stats(&sched);
        assert!(stats.min_beta < stats.max_beta);
    }

    #[test]
    fn test_schedule_stats_snr_range() {
        let sched = DdpmSchedule::new(
            100,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        let stats = compute_schedule_stats(&sched);
        assert!(stats.min_snr < stats.max_snr);
        assert!(stats.max_snr > 0.0);
    }

    // --- DdpmSamplerConfig::validate ---

    #[test]
    fn test_config_validate_zero_inference_steps() {
        let config = DdpmSamplerConfig {
            num_inference_steps: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_ok() {
        let config = DdpmSamplerConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_invalid_eta() {
        let config = DdpmSamplerConfig {
            eta: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    // --- sample (integration) ---

    #[test]
    fn test_sample_shape() {
        let sched = DdpmSchedule::new(
            20,
            BetaSchedule::Linear {
                start: 0.0001,
                end: 0.02,
            },
        )
        .expect("should succeed");
        let config = DdpmSamplerConfig {
            num_inference_steps: 5,
            ..Default::default()
        };
        let result =
            sample(16, &sched, &config, |_x, _t| vec![0.0f32; 16]).expect("should succeed");
        assert_eq!(result.len(), 16);
    }
}
