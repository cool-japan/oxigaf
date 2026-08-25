//! The per-step update kernels behind [`MultiStepSampler`][super::MultiStepSampler].
//!
//! Split out of `multi_step_sampler.rs`, which had grown to within a handful of
//! lines of the 2000-line limit. These are pure functions over flat `f32`
//! latents: given a noise prediction at timestep `t`, each returns the latent
//! at `t_prev` under its own integrator (DDIM, PLMS Adams-Bashforth, or
//! DPM++ 2M in log-SNR space). None of them own sampler state, so the driver
//! type in the parent module supplies the history buffers they need.

use super::{SamplerError, SamplingNoiseSchedule};

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
    let (alpha_bar_t_prev, _) = prev_coeffs(t_prev, schedule);

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

/// ᾱ and σ at `t_prev`, using the clean-boundary convention `t_prev == 0`
/// → `(1, 0)` shared with [`ddim_step`].
#[inline]
fn prev_coeffs(t_prev: usize, schedule: &SamplingNoiseSchedule) -> (f32, f32) {
    if t_prev > 0 {
        (schedule.alpha_bar_at(t_prev), schedule.sigma_at(t_prev))
    } else {
        (1.0_f32, 0.0_f32)
    }
}

/// Log-SNR `λ = log(α / σ)` for `α = √ᾱ` (the **square root** of ᾱ) and
/// `σ = √(1 − ᾱ)` — the definition that makes the first-order DPM++ update
/// coincide exactly with DDIM at η = 0.
///
/// Returns `None` on the clean boundary (σ = 0), where λ is unbounded.
#[inline]
fn log_snr(alpha_bar: f32, sigma: f32) -> Option<f32> {
    if sigma <= f32::EPSILON {
        return None;
    }
    Some((alpha_bar.sqrt() / sigma).ln())
}

/// Step size in log-SNR space for a DPM++ update from `t` down to `t_prev`:
/// `h = λ_{t_prev} − λ_t` (positive when denoising).
///
/// Returns `None` when an endpoint sits on the clean boundary (σ = 0) and `h`
/// is unbounded; the update then takes its exact limit
/// (`e^{-h} − 1 → −1`, `σ_{t_prev}/σ_t → 0`, i.e. `x ← x₀_pred`). Callers
/// driving [`dpm_plus_plus_2m_step`] themselves use this to carry the previous
/// step's `h` forward for the second-order correction.
pub fn dpm_step_size(t: usize, t_prev: usize, schedule: &SamplingNoiseSchedule) -> Option<f32> {
    let lambda_t = log_snr(schedule.alpha_bar_at(t), schedule.sigma_at(t))?;
    let (alpha_bar_prev, sigma_prev) = prev_coeffs(t_prev, schedule);
    let lambda_prev = log_snr(alpha_bar_prev, sigma_prev)?;
    Some(lambda_prev - lambda_t)
}

/// One DPM++ 2nd-order multistep (DPM-Solver++(2M)) denoising step.
///
/// Operates in log-SNR space with `λ_t = log(√ᾱ_t / σ_t)` and
/// `h = λ_{t_prev} − λ_t`, integrating the **data prediction** x₀:
///
/// ```text
/// r  = h_prev / h
/// D  = (1 + 1/(2r))·x₀(t) − (1/(2r))·x₀(t_prev_step)
/// x_{t-1} = (σ_{t-1}/σ_t)·xₜ − α_{t-1}·(e^{-h} − 1)·D      with α = √ᾱ
/// ```
///
/// **First order** — used when `prev_x0` or `h_prev` is absent, when `h` is
/// degenerate, or on the final step (σ_{t_prev} = 0, matching the reference
/// implementation) — replaces `D` with `x₀(t)`, which is algebraically
/// identical to [`ddim_step`] at η = 0.
///
/// `prev_x0` is the **x₀ prediction of the previous step**, evaluated at that
/// step's own sample and timestep; `h_prev` is that step's log-SNR step size.
/// [`crate::MultiStepSampler`] carries both automatically.
pub fn dpm_plus_plus_2m_step(
    sample: &[f32],
    noise_pred: &[f32],
    prev_x0: Option<&[f32]>,
    h_prev: Option<f32>,
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
    if let Some(p) = prev_x0 {
        if p.len() != sample.len() {
            return Err(SamplerError::DimensionMismatch {
                expected: sample.len(),
                got: p.len(),
            });
        }
    }

    let sigma_t = schedule.sigma_at(t);
    let (alpha_bar_t_prev, sigma_t_prev) = prev_coeffs(t_prev, schedule);
    let alpha_t_prev = alpha_bar_t_prev.sqrt(); // α_{t-1} = √ᾱ_{t-1}

    // Exponential-integrator coefficients. `None` is the clean boundary, where
    // σ_{t_prev}/σ_t → 0 and e^{-h} − 1 → −1, so the update returns x₀ exactly.
    let h = dpm_step_size(t, t_prev, schedule);
    let (sigma_ratio, expm1_neg_h) = match h {
        Some(h) => {
            let ratio = if sigma_t.abs() < f32::EPSILON {
                0.0
            } else {
                sigma_t_prev / sigma_t
            };
            (ratio, (-h).exp() - 1.0)
        }
        None => (0.0_f32, -1.0_f32),
    };

    // x₀ estimate at the current step — the quantity the solver integrates.
    let x0_pred_t = predict_x0(sample, noise_pred, t, schedule)?;

    // Second order needs a usable ratio r = h_prev / h from two finite,
    // strictly positive step sizes.
    let second_order = match (h, h_prev, prev_x0) {
        (Some(h), Some(h_prev), Some(x0_prev))
            if h.is_finite() && h > f32::EPSILON && h_prev.is_finite() && h_prev > f32::EPSILON =>
        {
            Some((h_prev / h, x0_prev))
        }
        _ => None,
    };

    let x_prev: Vec<f32> = match second_order {
        Some((r, x0_prev)) => {
            let c = 1.0 / (2.0 * r);
            sample
                .iter()
                .zip(x0_pred_t.iter())
                .zip(x0_prev.iter())
                .map(|((&xti, &x0i), &x0_prev_i)| {
                    let d = (1.0 + c) * x0i - c * x0_prev_i;
                    sigma_ratio * xti - alpha_t_prev * expm1_neg_h * d
                })
                .collect()
        }
        None => sample
            .iter()
            .zip(x0_pred_t.iter())
            .map(|(&xti, &x0i)| sigma_ratio * xti - alpha_t_prev * expm1_neg_h * x0i)
            .collect(),
    };
    Ok(x_prev)
}
