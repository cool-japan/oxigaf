//! # Stochastic Interpolants (Albergo & Vanden-Eijnden 2022, Ma et al. 2023)
//!
//! Stochastic Interpolants provide a general framework for building continuous
//! normalising flows and diffusion models by constructing a time-indexed
//! stochastic process that interpolates between two arbitrary distributions.
//!
//! ## The Interpolant
//!
//! ```text
//! x_t = α(t)·x₀ + β(t)·x₁ + γ(t)·ε,   ε ~ N(0, I)
//! ```
//!
//! Boundary conditions ensure the endpoints collapse to pure samples:
//!
//! ```text
//! t = 0:  α(0) = 1,  β(0) = 0,  γ(0) = 0   →  x_0 = x₀
//! t = 1:  α(1) = 0,  β(1) = 1,  γ(1) = 0   →  x_1 = x₁
//! ```
//!
//! The conditional velocity along a trajectory (given x₀, x₁, ε) is:
//!
//! ```text
//! ẋ_t = α'(t)·x₀ + β'(t)·x₁ + γ'(t)·ε
//! ```
//!
//! A network learns the marginal drift b(x_t, t) = E[ẋ_t | x_t].
//!
//! ## References
//!
//! - Albergo & Vanden-Eijnden (2022), "Building Normalizing Flows with Stochastic Interpolants"
//! - Ma et al. (2023), "The Unified Framework for Stochastic Interpolants and Diffusion Models"
//! - Liu et al. (2022), "Flow Straight and Fast: Rectified Flow"

use std::f32::consts::PI;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors from stochastic interpolant operations.
#[derive(Debug, Error, PartialEq)]
pub enum StochasticInterpolantError {
    /// Batch contains no samples.
    #[error("Empty batch")]
    EmptyBatch,

    /// Input array lengths are incompatible.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Timestep is outside [0.0, 1.0].
    #[error("Invalid timestep t={t}: must be in [0.0, 1.0]")]
    InvalidTimestep { t: f32 },

    /// Configuration parameter is semantically invalid.
    #[error("Invalid interpolant config: {reason}")]
    InvalidConfig { reason: String },
}

// ─────────────────────────────────────────────────────────────────────────────
// Module-local PRNG (xorshift64 + Box-Muller)
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

/// Uniform f32 in [0, 1) from 53 high-quality mantissa bits.
#[inline]
fn xorshift_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 11) as f32 / (1u64 << 53) as f32
}

/// Box-Muller transform: two uniforms → one standard normal N(0, 1).
#[inline]
fn si_bm_normal(state: &mut u64) -> f32 {
    let u1 = xorshift_f32(state).max(1e-10_f32);
    let u2 = xorshift_f32(state);
    (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * PI * u2).cos()
}

// ─────────────────────────────────────────────────────────────────────────────
// Interpolant kind
// ─────────────────────────────────────────────────────────────────────────────

/// Selects the coefficient schedule α(t), β(t), γ(t) for the interpolant.
#[derive(Debug, Clone, PartialEq)]
pub enum InterpolantKind {
    /// Linear (Rectified Flow special case, deterministic):
    /// α(t) = 1 − t,  β(t) = t,  γ(t) = 0
    LinearDeterministic,

    /// Linear stochastic:
    /// α(t) = 1 − t,  β(t) = t,  γ(t) = σ · √(t·(1−t))
    LinearStochastic { sigma: f32 },

    /// Trigonometric (Flow Matching cosine schedule, deterministic):
    /// α(t) = cos(πt/2),  β(t) = sin(πt/2),  γ(t) = 0
    Trigonometric,

    /// Trigonometric stochastic:
    /// α(t) = cos(πt/2),  β(t) = sin(πt/2),  γ(t) = σ · sin(πt)
    TrigonometricStochastic { sigma: f32 },

    /// Quadratic polynomial:
    /// α(t) = (1−t)²,  β(t) = t²,  γ(t) = 2·t·(1−t)
    Polynomial,

    /// VP-SDE (DDPM-family):
    /// β_schedule(s) = β_min + (β_max − β_min)·s
    /// ᾱ_t = exp(−∫₀ᵗ β(s) ds)
    /// α(t) = √(ᾱ_t),  β(t) = 0,  γ(t) = √(1 − ᾱ_t)
    VpSde { beta_min: f32, beta_max: f32 },

    /// Custom schedule via look-up tables sampled at t = 0, 0.1, …, 1.0.
    Custom {
        /// α values at 11 knots.
        alpha_table: [f32; 11],
        /// β values at 11 knots.
        beta_table: [f32; 11],
        /// γ values at 11 knots.
        gamma_table: [f32; 11],
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluated coefficients
// ─────────────────────────────────────────────────────────────────────────────

/// Coefficients α(t), β(t), γ(t) and their time-derivatives.
#[derive(Debug, Clone, PartialEq)]
pub struct SiCoefficients {
    /// α(t)
    pub alpha: f32,
    /// β(t)
    pub beta: f32,
    /// γ(t)
    pub gamma: f32,
    /// α'(t)
    pub alpha_dot: f32,
    /// β'(t)
    pub beta_dot: f32,
    /// γ'(t)
    pub gamma_dot: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// VP-SDE helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute ᾱ_t = exp(−∫₀ᵗ β(s) ds) for a linear β schedule.
///
/// ∫₀ᵗ (β_min + (β_max − β_min)·s) ds = β_min·t + (β_max − β_min)·t²/2
#[inline]
fn vp_alpha_bar(t: f32, beta_min: f32, beta_max: f32) -> f32 {
    let integral = beta_min * t + 0.5 * (beta_max - beta_min) * t * t;
    (-integral).exp()
}

/// VP-SDE α(t) = √(ᾱ_t).
#[inline]
fn vp_alpha(t: f32, beta_min: f32, beta_max: f32) -> f32 {
    vp_alpha_bar(t, beta_min, beta_max).sqrt()
}

/// VP-SDE γ(t) = √(1 − ᾱ_t).
#[inline]
fn vp_gamma(t: f32, beta_min: f32, beta_max: f32) -> f32 {
    (1.0 - vp_alpha_bar(t, beta_min, beta_max)).max(0.0).sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Core coefficient evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate α(t), β(t), γ(t), α'(t), β'(t), γ'(t) for the chosen schedule.
///
/// Returns [`StochasticInterpolantError::InvalidTimestep`] if `t ∉ [0, 1]`.
pub fn si_eval_coefficients(
    kind: &InterpolantKind,
    t: f32,
) -> Result<SiCoefficients, StochasticInterpolantError> {
    if !(0.0..=1.0).contains(&t) {
        return Err(StochasticInterpolantError::InvalidTimestep { t });
    }

    match kind {
        InterpolantKind::LinearDeterministic => Ok(SiCoefficients {
            alpha: 1.0 - t,
            beta: t,
            gamma: 0.0,
            alpha_dot: -1.0,
            beta_dot: 1.0,
            gamma_dot: 0.0,
        }),

        InterpolantKind::LinearStochastic { sigma } => {
            let sigma = *sigma;
            let t1mt = (t * (1.0 - t)).max(0.0);
            let gamma = sigma * t1mt.sqrt();
            // γ'(t) = σ · d/dt √(t(1−t)) = σ · (1 − 2t) / (2√(t(1−t)))
            let gamma_dot = if t1mt > 1e-10 {
                sigma * (1.0 - 2.0 * t) / (2.0 * t1mt.sqrt())
            } else {
                0.0
            };
            Ok(SiCoefficients {
                alpha: 1.0 - t,
                beta: t,
                gamma,
                alpha_dot: -1.0,
                beta_dot: 1.0,
                gamma_dot,
            })
        }

        InterpolantKind::Trigonometric => {
            let half_pi_t = PI * t / 2.0;
            Ok(SiCoefficients {
                alpha: half_pi_t.cos(),
                beta: half_pi_t.sin(),
                gamma: 0.0,
                alpha_dot: -PI / 2.0 * half_pi_t.sin(),
                beta_dot: PI / 2.0 * half_pi_t.cos(),
                gamma_dot: 0.0,
            })
        }

        InterpolantKind::TrigonometricStochastic { sigma } => {
            let sigma = *sigma;
            let half_pi_t = PI * t / 2.0;
            let pi_t = PI * t;
            Ok(SiCoefficients {
                alpha: half_pi_t.cos(),
                beta: half_pi_t.sin(),
                gamma: sigma * pi_t.sin(),
                alpha_dot: -PI / 2.0 * half_pi_t.sin(),
                beta_dot: PI / 2.0 * half_pi_t.cos(),
                gamma_dot: sigma * PI * pi_t.cos(),
            })
        }

        InterpolantKind::Polynomial => {
            // α = (1−t)², β = t², γ = 2t(1−t)
            let one_m_t = 1.0 - t;
            Ok(SiCoefficients {
                alpha: one_m_t * one_m_t,
                beta: t * t,
                gamma: 2.0 * t * one_m_t,
                alpha_dot: -2.0 * one_m_t,
                beta_dot: 2.0 * t,
                gamma_dot: 2.0 - 4.0 * t,
            })
        }

        InterpolantKind::VpSde { beta_min, beta_max } => {
            let (bmin, bmax) = (*beta_min, *beta_max);
            let eps = 1e-4_f32;

            let alpha = vp_alpha(t, bmin, bmax);
            let gamma = vp_gamma(t, bmin, bmax);

            // Finite-difference derivatives
            let alpha_dot = if t < eps {
                (vp_alpha(t + eps, bmin, bmax) - vp_alpha(t, bmin, bmax)) / eps
            } else if t > 1.0 - eps {
                (vp_alpha(t, bmin, bmax) - vp_alpha(t - eps, bmin, bmax)) / eps
            } else {
                (vp_alpha(t + eps, bmin, bmax) - vp_alpha(t - eps, bmin, bmax)) / (2.0 * eps)
            };

            let gamma_dot = if t < eps {
                (vp_gamma(t + eps, bmin, bmax) - vp_gamma(t, bmin, bmax)) / eps
            } else if t > 1.0 - eps {
                (vp_gamma(t, bmin, bmax) - vp_gamma(t - eps, bmin, bmax)) / eps
            } else {
                (vp_gamma(t + eps, bmin, bmax) - vp_gamma(t - eps, bmin, bmax)) / (2.0 * eps)
            };

            Ok(SiCoefficients {
                alpha,
                beta: 0.0,
                gamma,
                alpha_dot,
                beta_dot: 0.0,
                gamma_dot,
            })
        }

        InterpolantKind::Custom {
            alpha_table,
            beta_table,
            gamma_table,
        } => {
            // Linear interpolation on 11 knots: 0, 0.1, ..., 1.0
            let scaled = t * 10.0;
            let lo = scaled.floor() as usize;
            let hi = (lo + 1).min(10);
            let frac = scaled - lo as f32;

            let lerp = |table: &[f32; 11]| table[lo] + frac * (table[hi] - table[lo]);

            let alpha = lerp(alpha_table);
            let beta = lerp(beta_table);
            let gamma = lerp(gamma_table);

            // Finite-difference derivatives from the table values
            let eps = 1e-4_f32;
            let eval_alpha = |t2: f32| {
                let s2 = (t2 * 10.0).clamp(0.0, 10.0);
                let l2 = s2.floor() as usize;
                let h2 = (l2 + 1).min(10);
                let f2 = s2 - l2 as f32;
                alpha_table[l2] + f2 * (alpha_table[h2] - alpha_table[l2])
            };
            let eval_beta = |t2: f32| {
                let s2 = (t2 * 10.0).clamp(0.0, 10.0);
                let l2 = s2.floor() as usize;
                let h2 = (l2 + 1).min(10);
                let f2 = s2 - l2 as f32;
                beta_table[l2] + f2 * (beta_table[h2] - beta_table[l2])
            };
            let eval_gamma = |t2: f32| {
                let s2 = (t2 * 10.0).clamp(0.0, 10.0);
                let l2 = s2.floor() as usize;
                let h2 = (l2 + 1).min(10);
                let f2 = s2 - l2 as f32;
                gamma_table[l2] + f2 * (gamma_table[h2] - gamma_table[l2])
            };

            let alpha_dot = if t < eps {
                (eval_alpha(t + eps) - eval_alpha(t)) / eps
            } else if t > 1.0 - eps {
                (eval_alpha(t) - eval_alpha(t - eps)) / eps
            } else {
                (eval_alpha(t + eps) - eval_alpha(t - eps)) / (2.0 * eps)
            };

            let beta_dot = if t < eps {
                (eval_beta(t + eps) - eval_beta(t)) / eps
            } else if t > 1.0 - eps {
                (eval_beta(t) - eval_beta(t - eps)) / eps
            } else {
                (eval_beta(t + eps) - eval_beta(t - eps)) / (2.0 * eps)
            };

            let gamma_dot = if t < eps {
                (eval_gamma(t + eps) - eval_gamma(t)) / eps
            } else if t > 1.0 - eps {
                (eval_gamma(t) - eval_gamma(t - eps)) / eps
            } else {
                (eval_gamma(t + eps) - eval_gamma(t - eps)) / (2.0 * eps)
            };

            Ok(SiCoefficients {
                alpha,
                beta,
                gamma,
                alpha_dot,
                beta_dot,
                gamma_dot,
            })
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch data structure
// ─────────────────────────────────────────────────────────────────────────────

/// A mini-batch of stochastic interpolant samples.
///
/// All flat arrays have length `n × d` in row-major order (sample × dimension).
#[derive(Debug, Clone)]
pub struct SiBatch {
    /// Source samples x₀ ~ N(0, I) — shape [N × D].
    pub x0: Vec<f32>,
    /// Target samples x₁ ~ p_data — shape [N × D].
    pub x1: Vec<f32>,
    /// Independent noise ε ~ N(0, I) — shape [N × D].
    pub noise: Vec<f32>,
    /// Interpolated x_t = α·x₀ + β·x₁ + γ·ε — shape [N × D].
    pub x_t: Vec<f32>,
    /// Conditional velocity ẋ_t = α'·x₀ + β'·x₁ + γ'·ε — shape [N × D].
    pub dx_t: Vec<f32>,
    /// Per-sample timestep — length N.
    pub t: Vec<f32>,
    /// Per-sample evaluated coefficients — length N.
    pub coeffs: Vec<SiCoefficients>,
    /// Number of samples N.
    pub n: usize,
    /// Dimensionality D.
    pub d: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Interpolation primitives
// ─────────────────────────────────────────────────────────────────────────────

/// Compute x_t = α·x₀ + β·x₁ + γ·ε for a single sample.
///
/// # Errors
///
/// Returns [`StochasticInterpolantError::DimensionMismatch`] if the three slices
/// have different lengths, or [`StochasticInterpolantError::EmptyBatch`] if they
/// are empty.
pub fn si_interpolate(
    x0: &[f32],
    x1: &[f32],
    noise: &[f32],
    coeffs: &SiCoefficients,
) -> Result<Vec<f32>, StochasticInterpolantError> {
    let d = x0.len();
    if d == 0 {
        return Err(StochasticInterpolantError::EmptyBatch);
    }
    if x1.len() != d {
        return Err(StochasticInterpolantError::DimensionMismatch {
            expected: d,
            got: x1.len(),
        });
    }
    if noise.len() != d {
        return Err(StochasticInterpolantError::DimensionMismatch {
            expected: d,
            got: noise.len(),
        });
    }
    Ok((0..d)
        .map(|i| coeffs.alpha * x0[i] + coeffs.beta * x1[i] + coeffs.gamma * noise[i])
        .collect())
}

/// Compute the conditional velocity ẋ_t = α'·x₀ + β'·x₁ + γ'·ε.
///
/// # Errors
///
/// Returns [`StochasticInterpolantError::DimensionMismatch`] if slices have
/// different lengths, or [`StochasticInterpolantError::EmptyBatch`] if empty.
pub fn si_velocity(
    x0: &[f32],
    x1: &[f32],
    noise: &[f32],
    coeffs: &SiCoefficients,
) -> Result<Vec<f32>, StochasticInterpolantError> {
    let d = x0.len();
    if d == 0 {
        return Err(StochasticInterpolantError::EmptyBatch);
    }
    if x1.len() != d {
        return Err(StochasticInterpolantError::DimensionMismatch {
            expected: d,
            got: x1.len(),
        });
    }
    if noise.len() != d {
        return Err(StochasticInterpolantError::DimensionMismatch {
            expected: d,
            got: noise.len(),
        });
    }
    Ok((0..d)
        .map(|i| coeffs.alpha_dot * x0[i] + coeffs.beta_dot * x1[i] + coeffs.gamma_dot * noise[i])
        .collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch construction
// ─────────────────────────────────────────────────────────────────────────────

/// Construct a [`SiBatch`] by:
///
/// 1. Sampling x₀ ~ N(0, I) and ε ~ N(0, I) using xorshift64 + Box-Muller.
/// 2. Computing x_t and ẋ_t for each sample at the given timestep.
///
/// # Parameters
///
/// - `x1`       — flat target data [N × D] (row-major)
/// - `n`        — number of samples N
/// - `d`        — dimensionality D
/// - `kind`     — interpolant schedule
/// - `t_values` — per-sample timesteps (length N; values must be in [0, 1])
/// - `seed`     — initial xorshift64 state
///
/// # Errors
///
/// - [`StochasticInterpolantError::EmptyBatch`] if `n == 0`
/// - [`StochasticInterpolantError::DimensionMismatch`] if `x1.len() != n * d`
/// - [`StochasticInterpolantError::InvalidTimestep`] if any t_values element
///   is outside [0, 1]
pub fn si_make_batch(
    x1: &[f32],
    n: usize,
    d: usize,
    kind: &InterpolantKind,
    t_values: &[f32],
    seed: u64,
) -> Result<SiBatch, StochasticInterpolantError> {
    if n == 0 {
        return Err(StochasticInterpolantError::EmptyBatch);
    }
    if x1.len() != n * d {
        return Err(StochasticInterpolantError::DimensionMismatch {
            expected: n * d,
            got: x1.len(),
        });
    }
    if t_values.len() != n {
        return Err(StochasticInterpolantError::DimensionMismatch {
            expected: n,
            got: t_values.len(),
        });
    }

    // Validate all timesteps up front
    for &t in t_values {
        if !(0.0..=1.0).contains(&t) {
            return Err(StochasticInterpolantError::InvalidTimestep { t });
        }
    }

    let mut state = if seed == 0 { 1u64 } else { seed };
    let nd = n * d;

    // Sample x0 ~ N(0, I) and noise ε ~ N(0, I) independently
    let mut x0 = Vec::with_capacity(nd);
    let mut noise = Vec::with_capacity(nd);
    for _ in 0..nd {
        x0.push(si_bm_normal(&mut state));
    }
    for _ in 0..nd {
        noise.push(si_bm_normal(&mut state));
    }

    let mut x_t = vec![0.0f32; nd];
    let mut dx_t = vec![0.0f32; nd];
    let mut coeffs_vec = Vec::with_capacity(n);

    for (i, &t) in t_values.iter().enumerate().take(n) {
        let c = si_eval_coefficients(kind, t)?;
        let base = i * d;

        for j in 0..d {
            let idx = base + j;
            x_t[idx] = c.alpha * x0[idx] + c.beta * x1[idx] + c.gamma * noise[idx];
            dx_t[idx] = c.alpha_dot * x0[idx] + c.beta_dot * x1[idx] + c.gamma_dot * noise[idx];
        }

        coeffs_vec.push(c);
    }

    Ok(SiBatch {
        x0,
        x1: x1.to_vec(),
        noise,
        x_t,
        dx_t,
        t: t_values.to_vec(),
        coeffs: coeffs_vec,
        n,
        d,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Training losses
// ─────────────────────────────────────────────────────────────────────────────

/// Mean squared error loss: mean‖v_pred − ẋ_t‖².
///
/// # Errors
///
/// - [`StochasticInterpolantError::EmptyBatch`] if `batch.n == 0`
/// - [`StochasticInterpolantError::DimensionMismatch`] if `predicted_v.len() ≠ n * d`
pub fn si_velocity_loss(
    predicted_v: &[f32],
    batch: &SiBatch,
) -> Result<f32, StochasticInterpolantError> {
    if batch.n == 0 {
        return Err(StochasticInterpolantError::EmptyBatch);
    }
    let nd = batch.n * batch.d;
    if predicted_v.len() != nd {
        return Err(StochasticInterpolantError::DimensionMismatch {
            expected: nd,
            got: predicted_v.len(),
        });
    }
    let sum_sq: f32 = predicted_v
        .iter()
        .zip(batch.dx_t.iter())
        .map(|(p, t)| (p - t) * (p - t))
        .sum();
    Ok(sum_sq / nd as f32)
}

/// Per-sample MSE losses: each element is mean‖v_pred\[i\] − ẋ_t\[i\]‖² over D.
///
/// # Errors
///
/// - [`StochasticInterpolantError::EmptyBatch`] if `batch.n == 0`
/// - [`StochasticInterpolantError::DimensionMismatch`] if `predicted_v.len() ≠ n * d`
pub fn si_velocity_loss_per_sample(
    predicted_v: &[f32],
    batch: &SiBatch,
) -> Result<Vec<f32>, StochasticInterpolantError> {
    if batch.n == 0 {
        return Err(StochasticInterpolantError::EmptyBatch);
    }
    let nd = batch.n * batch.d;
    if predicted_v.len() != nd {
        return Err(StochasticInterpolantError::DimensionMismatch {
            expected: nd,
            got: predicted_v.len(),
        });
    }
    let d = batch.d;
    let mut losses = Vec::with_capacity(batch.n);
    for i in 0..batch.n {
        let base = i * d;
        let sq: f32 = (0..d)
            .map(|j| {
                let diff = predicted_v[base + j] - batch.dx_t[base + j];
                diff * diff
            })
            .sum();
        losses.push(sq / d as f32);
    }
    Ok(losses)
}

/// Score function loss.
///
/// The implicit score target is s*(x_t, t) = −ε / γ(t) when γ(t) > 0.
///
/// Samples with γ(t) ≈ 0 (below `config.eps`) are skipped; if all samples
/// are skipped the function returns 0.0.
///
/// # Errors
///
/// - [`StochasticInterpolantError::EmptyBatch`] if `batch.n == 0`
/// - [`StochasticInterpolantError::DimensionMismatch`] if `predicted_s.len() ≠ n * d`
pub fn si_score_loss(
    predicted_s: &[f32],
    batch: &SiBatch,
    config: &InterpolantConfig,
) -> Result<f32, StochasticInterpolantError> {
    if batch.n == 0 {
        return Err(StochasticInterpolantError::EmptyBatch);
    }
    let nd = batch.n * batch.d;
    if predicted_s.len() != nd {
        return Err(StochasticInterpolantError::DimensionMismatch {
            expected: nd,
            got: predicted_s.len(),
        });
    }

    let eps = config.eps;
    let d = batch.d;
    let mut total = 0.0f32;
    let mut count = 0usize;

    for i in 0..batch.n {
        let gamma = batch.coeffs[i].gamma;
        if gamma.abs() < eps {
            continue; // skip degenerate samples
        }
        let base = i * d;
        for j in 0..d {
            let idx = base + j;
            // target = -ε / γ
            let target = -batch.noise[idx] / gamma;
            let diff = predicted_s[idx] - target;
            total += diff * diff;
        }
        count += d;
    }

    if count == 0 {
        return Ok(0.0);
    }
    Ok(total / count as f32)
}

// ─────────────────────────────────────────────────────────────────────────────
// ODE / SDE integration
// ─────────────────────────────────────────────────────────────────────────────

/// Intermediate state for a single integration step.
#[derive(Debug, Clone)]
pub struct SiDrift {
    /// Current position.
    pub x: Vec<f32>,
    /// Current time.
    pub t: f32,
    /// Predicted velocity from the network at (x, t).
    pub predicted_v: Vec<f32>,
}

/// Configuration for the stochastic interpolant inference pipeline.
#[derive(Debug, Clone)]
pub struct InterpolantConfig {
    /// Coefficient schedule.
    pub kind: InterpolantKind,
    /// Number of ODE/SDE integration steps.
    pub n_steps: usize,
    /// Numerical stability floor (e.g. 1e-7).
    pub eps: f32,
    /// If `true`, use Euler–Maruyama (stochastic); otherwise pure ODE.
    pub use_stochastic: bool,
    /// Noise amplitude σ for SDE mode (e.g. 0.01).
    pub sigma_inf: f32,
}

/// Deterministic Euler step: x_next = x + dt · v.
pub fn si_euler_step_ode(x: &[f32], v: &[f32], dt: f32) -> Vec<f32> {
    x.iter()
        .zip(v.iter())
        .map(|(xi, vi)| xi + dt * vi)
        .collect()
}

/// Stochastic Euler–Maruyama step: x_next = x + dt · v + √(2·σ·dt) · ε.
///
/// Noise is generated internally with xorshift64 + Box-Muller.
pub fn si_euler_step_sde(x: &[f32], v: &[f32], dt: f32, sigma_t: f32, seed: u64) -> Vec<f32> {
    let mut state = if seed == 0 { 1u64 } else { seed };
    let scale = (2.0_f32 * sigma_t * dt.abs()).max(0.0).sqrt();
    x.iter()
        .zip(v.iter())
        .map(|(xi, vi)| {
            let eps = si_bm_normal(&mut state);
            xi + dt * vi + scale * eps
        })
        .collect()
}

/// Integrate from t = 0 to t = 1 using the Euler method.
///
/// `velocity_fn(x, t)` — closure returning the predicted velocity at (x, t).
///
/// Returns a trajectory of length `n_steps + 1` (includes the initial state).
///
/// # Errors
///
/// - [`StochasticInterpolantError::EmptyBatch`] if `x0` is empty
/// - [`StochasticInterpolantError::InvalidConfig`] if `n_steps == 0`
pub fn si_ode_integrate(
    x0: &[f32],
    velocity_fn: impl Fn(&[f32], f32) -> Vec<f32>,
    config: &InterpolantConfig,
) -> Result<Vec<Vec<f32>>, StochasticInterpolantError> {
    if x0.is_empty() {
        return Err(StochasticInterpolantError::EmptyBatch);
    }
    if config.n_steps == 0 {
        return Err(StochasticInterpolantError::InvalidConfig {
            reason: "n_steps must be > 0".to_string(),
        });
    }

    let n_steps = config.n_steps;
    let dt = 1.0 / n_steps as f32;
    let mut trajectory = Vec::with_capacity(n_steps + 1);
    trajectory.push(x0.to_vec());

    let mut x_cur = x0.to_vec();
    let mut seed_state: u64 = 0xDEAD_BEEF_CAFE_1234;

    for step in 0..n_steps {
        let t = step as f32 * dt;
        let v = velocity_fn(&x_cur, t);

        x_cur = if config.use_stochastic {
            seed_state = xorshift64(&mut seed_state.clone());
            si_euler_step_sde(&x_cur, &v, dt, config.sigma_inf, seed_state)
        } else {
            si_euler_step_ode(&x_cur, &v, dt)
        };

        trajectory.push(x_cur.clone());
    }

    Ok(trajectory)
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

/// Generate `n` evenly spaced values from `a` to `b` inclusive.
///
/// If `n == 1`, returns `[a]`.
pub fn si_linspace(a: f32, b: f32, n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f32;
    (0..n).map(|i| a + i as f32 * step).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Boundary condition verification
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that the schedule satisfies the stochastic interpolant boundary conditions:
///
/// ```text
/// α(0) ≈ 1, β(0) ≈ 0, γ(0) ≈ 0
/// α(1) ≈ 0, β(1) ≈ 1, γ(1) ≈ 0
/// ```
///
/// VP-SDE is treated separately because it uses γ(t) as its stochastic component
/// (β = 0 always); its boundary conditions are α(0) ≈ 1, γ(0) ≈ 0, α(1) ≈ 0,
/// γ(1) ≈ 1.
///
/// # Errors
///
/// Returns [`StochasticInterpolantError::InvalidConfig`] with a descriptive
/// message on any violated condition.
pub fn si_check_boundary_conditions(
    kind: &InterpolantKind,
    eps: f32,
) -> Result<(), StochasticInterpolantError> {
    // VP-SDE has a different boundary structure (β≡0, γ(1)≈1)
    if let InterpolantKind::VpSde { beta_min, beta_max } = kind {
        let c0 = si_eval_coefficients(kind, 0.0)?;
        let c1 = si_eval_coefficients(kind, 1.0)?;

        let check = |name: &str, got: f32, want: f32| -> Result<(), StochasticInterpolantError> {
            if (got - want).abs() > eps {
                Err(StochasticInterpolantError::InvalidConfig {
                    reason: format!(
                        "VP-SDE boundary violation: {name}={got:.6}, want {want:.6} (beta_min={beta_min}, beta_max={beta_max})"
                    ),
                })
            } else {
                Ok(())
            }
        };

        check("alpha(0)", c0.alpha, 1.0)?;
        check("gamma(0)", c0.gamma, 0.0)?;
        check("alpha(1)", c1.alpha, 0.0)?;
        // gamma(1) ≈ 1 for VP-SDE
        check("gamma(1)", c1.gamma, 1.0)?;
        return Ok(());
    }

    let c0 = si_eval_coefficients(kind, 0.0)?;
    let c1 = si_eval_coefficients(kind, 1.0)?;

    let check = |name: &str, got: f32, want: f32| -> Result<(), StochasticInterpolantError> {
        if (got - want).abs() > eps {
            Err(StochasticInterpolantError::InvalidConfig {
                reason: format!("Boundary violation: {name}={got:.6}, expected {want:.6}"),
            })
        } else {
            Ok(())
        }
    };

    check("alpha(0)", c0.alpha, 1.0)?;
    check("beta(0)", c0.beta, 0.0)?;
    check("gamma(0)", c0.gamma, 0.0)?;
    check("alpha(1)", c1.alpha, 0.0)?;
    check("beta(1)", c1.beta, 1.0)?;
    check("gamma(1)", c1.gamma, 0.0)?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics and formatting
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics for a stochastic interpolant batch and config.
#[derive(Debug, Clone)]
pub struct SiStats {
    /// Mean L2 norm of the conditional velocities ẋ_t.
    pub mean_velocity_norm: f32,
    /// Mean γ(t) over samples — reflects the stochastic component's strength.
    pub mean_noise_level: f32,
    /// Whether boundary conditions are satisfied (within default ε = 1e-3).
    pub boundary_check_passed: bool,
    /// Human-readable name of the interpolant kind.
    pub kind_name: String,
    /// Integration step count from the config.
    pub n_steps: usize,
}

/// Compute summary statistics from a batch.
pub fn si_compute_stats(batch: &SiBatch, config: &InterpolantConfig) -> SiStats {
    let n = batch.n;
    let d = batch.d;

    let mean_velocity_norm = if n == 0 || d == 0 {
        0.0
    } else {
        let sum: f32 = (0..n)
            .map(|i| {
                let base = i * d;
                let sq: f32 = (0..d).map(|j| batch.dx_t[base + j].powi(2)).sum();
                sq.sqrt()
            })
            .sum();
        sum / n as f32
    };

    let mean_noise_level = if n == 0 {
        0.0
    } else {
        batch.coeffs.iter().map(|c| c.gamma).sum::<f32>() / n as f32
    };

    let boundary_check_passed = si_check_boundary_conditions(&config.kind, 1e-3).is_ok();

    SiStats {
        mean_velocity_norm,
        mean_noise_level,
        boundary_check_passed,
        kind_name: si_kind_name(&config.kind),
        n_steps: config.n_steps,
    }
}

/// Format stats as a human-readable multi-line string.
pub fn si_format_stats(stats: &SiStats) -> String {
    format!(
        "SiStats {{ kind={}, n_steps={}, mean_velocity_norm={:.6}, mean_noise_level={:.6}, boundary_ok={} }}",
        stats.kind_name,
        stats.n_steps,
        stats.mean_velocity_norm,
        stats.mean_noise_level,
        stats.boundary_check_passed
    )
}

/// Format config as a human-readable string.
pub fn si_format_config(config: &InterpolantConfig) -> String {
    format!(
        "InterpolantConfig {{ kind={}, n_steps={}, eps={:.2e}, use_stochastic={}, sigma_inf={:.4} }}",
        si_kind_name(&config.kind),
        config.n_steps,
        config.eps,
        config.use_stochastic,
        config.sigma_inf,
    )
}

/// Return a short human-readable name for the interpolant kind.
pub fn si_kind_name(kind: &InterpolantKind) -> String {
    match kind {
        InterpolantKind::LinearDeterministic => "LinearDeterministic".to_string(),
        InterpolantKind::LinearStochastic { sigma } => {
            format!("LinearStochastic(σ={sigma:.4})")
        }
        InterpolantKind::Trigonometric => "Trigonometric".to_string(),
        InterpolantKind::TrigonometricStochastic { sigma } => {
            format!("TrigonometricStochastic(σ={sigma:.4})")
        }
        InterpolantKind::Polynomial => "Polynomial".to_string(),
        InterpolantKind::VpSde { beta_min, beta_max } => {
            format!("VpSde(β_min={beta_min:.4}, β_max={beta_max:.4})")
        }
        InterpolantKind::Custom { .. } => "Custom".to_string(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    // ── si_eval_coefficients: LinearDeterministic ─────────────────────────────

    #[test]
    fn test_linear_det_at_t0() {
        let c = si_eval_coefficients(&InterpolantKind::LinearDeterministic, 0.0).unwrap();
        assert!((c.alpha - 1.0).abs() < TOL, "alpha(0)={}", c.alpha);
        assert!(c.beta.abs() < TOL, "beta(0)={}", c.beta);
        assert!(c.gamma.abs() < TOL, "gamma(0)={}", c.gamma);
    }

    #[test]
    fn test_linear_det_at_t1() {
        let c = si_eval_coefficients(&InterpolantKind::LinearDeterministic, 1.0).unwrap();
        assert!(c.alpha.abs() < TOL, "alpha(1)={}", c.alpha);
        assert!((c.beta - 1.0).abs() < TOL, "beta(1)={}", c.beta);
        assert!(c.gamma.abs() < TOL, "gamma(1)={}", c.gamma);
    }

    #[test]
    fn test_linear_det_at_t05() {
        let c = si_eval_coefficients(&InterpolantKind::LinearDeterministic, 0.5).unwrap();
        assert!((c.alpha - 0.5).abs() < TOL);
        assert!((c.beta - 0.5).abs() < TOL);
        assert!(c.gamma.abs() < TOL);
    }

    #[test]
    fn test_linear_det_derivatives_constant() {
        // α'=-1, β'=1 everywhere
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let c = si_eval_coefficients(&InterpolantKind::LinearDeterministic, t).unwrap();
            assert!((c.alpha_dot + 1.0).abs() < TOL, "alpha_dot at t={t}");
            assert!((c.beta_dot - 1.0).abs() < TOL, "beta_dot at t={t}");
            assert!(c.gamma_dot.abs() < TOL, "gamma_dot at t={t}");
        }
    }

    // ── si_eval_coefficients: Trigonometric ───────────────────────────────────

    #[test]
    fn test_trig_at_t0() {
        let c = si_eval_coefficients(&InterpolantKind::Trigonometric, 0.0).unwrap();
        assert!((c.alpha - 1.0).abs() < TOL);
        assert!(c.beta.abs() < TOL);
    }

    #[test]
    fn test_trig_at_t1() {
        let c = si_eval_coefficients(&InterpolantKind::Trigonometric, 1.0).unwrap();
        assert!(c.alpha.abs() < TOL);
        assert!((c.beta - 1.0).abs() < TOL);
    }

    #[test]
    fn test_trig_unit_circle() {
        // α² + β² = 1 for all t (since α=cos, β=sin of same angle)
        for t in [0.0f32, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] {
            let c = si_eval_coefficients(&InterpolantKind::Trigonometric, t).unwrap();
            let norm_sq = c.alpha * c.alpha + c.beta * c.beta;
            assert!((norm_sq - 1.0).abs() < TOL, "α²+β²={norm_sq} at t={t}");
        }
    }

    // ── si_eval_coefficients: Polynomial ─────────────────────────────────────

    #[test]
    fn test_poly_at_t0() {
        let c = si_eval_coefficients(&InterpolantKind::Polynomial, 0.0).unwrap();
        assert!((c.alpha - 1.0).abs() < TOL, "alpha(0)={}", c.alpha);
        assert!(c.beta.abs() < TOL, "beta(0)={}", c.beta);
        assert!(c.gamma.abs() < TOL, "gamma(0)={}", c.gamma);
    }

    #[test]
    fn test_poly_at_t1() {
        let c = si_eval_coefficients(&InterpolantKind::Polynomial, 1.0).unwrap();
        assert!(c.alpha.abs() < TOL, "alpha(1)={}", c.alpha);
        assert!((c.beta - 1.0).abs() < TOL, "beta(1)={}", c.beta);
        assert!(c.gamma.abs() < TOL, "gamma(1)={}", c.gamma);
    }

    #[test]
    fn test_poly_nonneg() {
        for t_i in 0..=20 {
            let t = t_i as f32 / 20.0;
            let c = si_eval_coefficients(&InterpolantKind::Polynomial, t).unwrap();
            assert!(c.alpha >= -TOL, "alpha < 0 at t={t}");
            assert!(c.beta >= -TOL, "beta < 0 at t={t}");
            assert!(c.gamma >= -TOL, "gamma < 0 at t={t}");
        }
    }

    // ── si_eval_coefficients: LinearStochastic ────────────────────────────────

    #[test]
    fn test_linear_stochastic_gamma_positive() {
        let kind = InterpolantKind::LinearStochastic { sigma: 1.0 };
        for t_i in 1..=9 {
            let t = t_i as f32 / 10.0;
            let c = si_eval_coefficients(&kind, t).unwrap();
            assert!(c.gamma > 0.0, "gamma should be > 0 for t={t}");
        }
    }

    #[test]
    fn test_linear_stochastic_boundary_zero() {
        let kind = InterpolantKind::LinearStochastic { sigma: 1.0 };
        let c0 = si_eval_coefficients(&kind, 0.0).unwrap();
        let c1 = si_eval_coefficients(&kind, 1.0).unwrap();
        assert!(c0.gamma.abs() < TOL, "gamma(0) must be 0");
        assert!(c1.gamma.abs() < TOL, "gamma(1) must be 0");
    }

    // ── si_eval_coefficients: TrigonometricStochastic ─────────────────────────

    #[test]
    fn test_trig_stochastic_gamma_at_half() {
        let kind = InterpolantKind::TrigonometricStochastic { sigma: 1.0 };
        let c = si_eval_coefficients(&kind, 0.5).unwrap();
        // γ(0.5) = sin(π·0.5) = sin(π/2) = 1.0
        assert!(c.gamma > 0.0, "gamma(0.5) should be > 0");
        assert!((c.gamma - 1.0).abs() < TOL, "gamma(0.5)={}", c.gamma);
    }

    // ── si_eval_coefficients: Custom ──────────────────────────────────────────

    #[test]
    fn test_custom_interpolates_table() {
        // Build a custom table from LinearDeterministic
        let mut alpha_table = [0.0f32; 11];
        let mut beta_table = [0.0f32; 11];
        let gamma_table = [0.0f32; 11];
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            alpha_table[i] = 1.0 - t;
            beta_table[i] = t;
        }
        let kind = InterpolantKind::Custom {
            alpha_table,
            beta_table,
            gamma_table,
        };
        // Check at t=0.5 (midpoint between indices 5 and 6, exactly on knot)
        let c = si_eval_coefficients(&kind, 0.5).unwrap();
        assert!((c.alpha - 0.5).abs() < TOL, "alpha(0.5)={}", c.alpha);
        assert!((c.beta - 0.5).abs() < TOL, "beta(0.5)={}", c.beta);
    }

    #[test]
    fn test_custom_at_endpoints() {
        let alpha_table = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0];
        let beta_table = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let gamma_table = [0.0f32; 11];
        let kind = InterpolantKind::Custom {
            alpha_table,
            beta_table,
            gamma_table,
        };
        let c0 = si_eval_coefficients(&kind, 0.0).unwrap();
        let c1 = si_eval_coefficients(&kind, 1.0).unwrap();
        assert!((c0.alpha - 1.0).abs() < TOL);
        assert!((c1.beta - 1.0).abs() < TOL);
    }

    // ── si_eval_coefficients: error cases ────────────────────────────────────

    #[test]
    fn test_invalid_timestep_negative() {
        let result = si_eval_coefficients(&InterpolantKind::LinearDeterministic, -0.1);
        assert!(matches!(
            result,
            Err(StochasticInterpolantError::InvalidTimestep { .. })
        ));
    }

    #[test]
    fn test_invalid_timestep_above_one() {
        let result = si_eval_coefficients(&InterpolantKind::LinearDeterministic, 1.1);
        assert!(matches!(
            result,
            Err(StochasticInterpolantError::InvalidTimestep { .. })
        ));
    }

    // ── si_interpolate ────────────────────────────────────────────────────────

    #[test]
    fn test_interpolate_t0_linear_det() {
        // At t=0: x_t = 1·x0 + 0·x1 + 0·ε = x0
        let coeffs = si_eval_coefficients(&InterpolantKind::LinearDeterministic, 0.0).unwrap();
        let x0 = vec![1.0f32, 2.0, 3.0];
        let x1 = vec![4.0f32, 5.0, 6.0];
        let noise = vec![7.0f32, 8.0, 9.0];
        let xt = si_interpolate(&x0, &x1, &noise, &coeffs).unwrap();
        for (a, b) in xt.iter().zip(x0.iter()) {
            assert!((a - b).abs() < TOL, "x_t should equal x0 at t=0");
        }
    }

    #[test]
    fn test_interpolate_t1_linear_det() {
        // At t=1: x_t = 0·x0 + 1·x1 + 0·ε = x1
        let coeffs = si_eval_coefficients(&InterpolantKind::LinearDeterministic, 1.0).unwrap();
        let x0 = vec![1.0f32, 2.0, 3.0];
        let x1 = vec![4.0f32, 5.0, 6.0];
        let noise = vec![7.0f32, 8.0, 9.0];
        let xt = si_interpolate(&x0, &x1, &noise, &coeffs).unwrap();
        for (a, b) in xt.iter().zip(x1.iter()) {
            assert!((a - b).abs() < TOL, "x_t should equal x1 at t=1");
        }
    }

    // ── si_velocity ───────────────────────────────────────────────────────────

    #[test]
    fn test_velocity_linear_det_equals_x1_minus_x0() {
        // LinearDeterministic: ẋ_t = -x0 + x1 + 0·ε = x1 - x0 (constant)
        let coeffs = si_eval_coefficients(&InterpolantKind::LinearDeterministic, 0.5).unwrap();
        let x0 = vec![1.0f32, 2.0, 3.0];
        let x1 = vec![4.0f32, 6.0, 9.0];
        let noise = vec![0.0f32, 0.0, 0.0];
        let v = si_velocity(&x0, &x1, &noise, &coeffs).unwrap();
        let expected: Vec<f32> = x0.iter().zip(x1.iter()).map(|(a, b)| b - a).collect();
        for (a, b) in v.iter().zip(expected.iter()) {
            assert!((a - b).abs() < TOL, "velocity mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_velocity_dimension_mismatch() {
        let coeffs = si_eval_coefficients(&InterpolantKind::LinearDeterministic, 0.5).unwrap();
        let x0 = vec![1.0f32, 2.0, 3.0];
        let x1 = vec![4.0f32, 5.0]; // wrong length
        let noise = vec![0.0f32, 0.0, 0.0];
        let result = si_velocity(&x0, &x1, &noise, &coeffs);
        assert!(matches!(
            result,
            Err(StochasticInterpolantError::DimensionMismatch { .. })
        ));
    }

    // ── si_make_batch ─────────────────────────────────────────────────────────

    #[test]
    fn test_make_batch_shapes() {
        let n = 8;
        let d = 4;
        let x1: Vec<f32> = (0..n * d).map(|i| i as f32).collect();
        let t_values = si_linspace(0.0, 1.0, n);
        let batch = si_make_batch(
            &x1,
            n,
            d,
            &InterpolantKind::LinearDeterministic,
            &t_values,
            42,
        )
        .unwrap();
        assert_eq!(batch.x_t.len(), n * d);
        assert_eq!(batch.dx_t.len(), n * d);
        assert_eq!(batch.x0.len(), n * d);
        assert_eq!(batch.noise.len(), n * d);
        assert_eq!(batch.n, n);
        assert_eq!(batch.d, d);
    }

    #[test]
    fn test_make_batch_noise_distribution() {
        // Box-Muller noise: mean ≈ 0, std ≈ 1 for large N
        let n = 1000;
        let d = 1;
        let x1 = vec![0.0f32; n * d];
        let t_values = vec![0.5f32; n];
        let batch = si_make_batch(
            &x1,
            n,
            d,
            &InterpolantKind::LinearDeterministic,
            &t_values,
            12345,
        )
        .unwrap();
        let mean: f32 = batch.noise.iter().sum::<f32>() / batch.noise.len() as f32;
        let var: f32 =
            batch.noise.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / batch.noise.len() as f32;
        assert!(mean.abs() < 0.15, "noise mean too large: {mean}");
        assert!(
            (var.sqrt() - 1.0).abs() < 0.15,
            "noise std too far from 1: {}",
            var.sqrt()
        );
    }

    #[test]
    fn test_make_batch_empty_error() {
        let result = si_make_batch(&[], 0, 4, &InterpolantKind::LinearDeterministic, &[], 1);
        assert!(matches!(
            result,
            Err(StochasticInterpolantError::EmptyBatch)
        ));
    }

    // ── si_velocity_loss ──────────────────────────────────────────────────────

    #[test]
    fn test_velocity_loss_zero_for_perfect() {
        let n = 4;
        let d = 3;
        let x1: Vec<f32> = vec![1.0; n * d];
        let t_values = vec![0.5f32; n];
        let batch = si_make_batch(
            &x1,
            n,
            d,
            &InterpolantKind::LinearDeterministic,
            &t_values,
            7,
        )
        .unwrap();
        // Perfect prediction = the actual velocity
        let loss = si_velocity_loss(&batch.dx_t, &batch).unwrap();
        assert!(
            loss.abs() < 1e-6,
            "perfect prediction should give 0 loss, got {loss}"
        );
    }

    #[test]
    fn test_velocity_loss_positive_for_wrong() {
        let n = 4;
        let d = 3;
        let x1: Vec<f32> = vec![1.0; n * d];
        let t_values = vec![0.5f32; n];
        let batch = si_make_batch(
            &x1,
            n,
            d,
            &InterpolantKind::LinearDeterministic,
            &t_values,
            7,
        )
        .unwrap();
        let wrong_v = vec![999.0f32; n * d];
        let loss = si_velocity_loss(&wrong_v, &batch).unwrap();
        assert!(loss > 0.0, "wrong prediction should give positive loss");
    }

    // ── si_velocity_loss_per_sample ───────────────────────────────────────────

    #[test]
    fn test_velocity_loss_per_sample_length() {
        let n = 6;
        let d = 2;
        let x1 = vec![0.5f32; n * d];
        let t_values = vec![0.3f32; n];
        let batch =
            si_make_batch(&x1, n, d, &InterpolantKind::Trigonometric, &t_values, 99).unwrap();
        let per = si_velocity_loss_per_sample(&batch.dx_t, &batch).unwrap();
        assert_eq!(per.len(), n);
    }

    #[test]
    fn test_velocity_loss_per_sample_nonneg() {
        let n = 5;
        let d = 3;
        let x1 = vec![0.0f32; n * d];
        let t_values = vec![0.4f32; n];
        let batch = si_make_batch(
            &x1,
            n,
            d,
            &InterpolantKind::LinearDeterministic,
            &t_values,
            55,
        )
        .unwrap();
        let wrong_v = vec![1.0f32; n * d];
        let per = si_velocity_loss_per_sample(&wrong_v, &batch).unwrap();
        for (i, &l) in per.iter().enumerate() {
            assert!(l >= 0.0, "per-sample loss[{i}] is negative: {l}");
        }
    }

    // ── si_score_loss ─────────────────────────────────────────────────────────

    #[test]
    fn test_score_loss_zero_gamma_graceful() {
        // LinearDeterministic has γ=0 everywhere — score_loss should return 0.0
        let n = 4;
        let d = 2;
        let x1 = vec![1.0f32; n * d];
        let t_values = vec![0.5f32; n];
        let kind = InterpolantKind::LinearDeterministic;
        let batch = si_make_batch(&x1, n, d, &kind, &t_values, 1).unwrap();
        let config = InterpolantConfig {
            kind,
            n_steps: 10,
            eps: 1e-7,
            use_stochastic: false,
            sigma_inf: 0.0,
        };
        let s_pred = vec![0.0f32; n * d];
        let loss = si_score_loss(&s_pred, &batch, &config).unwrap();
        assert!(
            loss.abs() < 1e-6,
            "score_loss with γ=0 should return 0, got {loss}"
        );
    }

    #[test]
    fn test_score_loss_stochastic_kind() {
        let n = 4;
        let d = 2;
        let x1 = vec![0.5f32; n * d];
        let t_values = vec![0.5f32; n];
        let kind = InterpolantKind::LinearStochastic { sigma: 1.0 };
        let batch = si_make_batch(&x1, n, d, &kind, &t_values, 77).unwrap();
        let config = InterpolantConfig {
            kind: InterpolantKind::LinearStochastic { sigma: 1.0 },
            n_steps: 10,
            eps: 1e-7,
            use_stochastic: false,
            sigma_inf: 0.0,
        };
        let s_pred = vec![0.0f32; n * d];
        let loss = si_score_loss(&s_pred, &batch, &config).unwrap();
        assert!(loss >= 0.0, "score loss must be non-negative");
    }

    // ── si_euler_step_ode ─────────────────────────────────────────────────────

    #[test]
    fn test_euler_step_ode_correctness() {
        let x = vec![1.0f32, 2.0, 3.0];
        let v = vec![0.1f32, -0.5, 2.0];
        let dt = 0.1;
        let x_next = si_euler_step_ode(&x, &v, dt);
        for (i, (xi, vi)) in x.iter().zip(v.iter()).enumerate() {
            let expected = xi + dt * vi;
            assert!(
                (x_next[i] - expected).abs() < TOL,
                "ODE step mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_euler_step_ode_zero_velocity() {
        let x = vec![3.0f32, 1.0, -2.0];
        let v = vec![0.0f32, 0.0, 0.0];
        let x_next = si_euler_step_ode(&x, &v, 0.5);
        for (a, b) in x_next.iter().zip(x.iter()) {
            assert!((a - b).abs() < TOL);
        }
    }

    // ── si_euler_step_sde ─────────────────────────────────────────────────────

    #[test]
    fn test_euler_step_sde_differs_from_ode() {
        // With non-zero sigma, SDE step should (almost certainly) differ from ODE
        let x = vec![0.0f32; 100];
        let v = vec![0.0f32; 100];
        let dt = 0.01;
        let sigma_t = 1.0;
        let sde = si_euler_step_sde(&x, &v, dt, sigma_t, 42);
        let ode = si_euler_step_ode(&x, &v, dt);
        let diff: f32 = sde.iter().zip(ode.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0, "SDE should produce different result from ODE");
    }

    #[test]
    fn test_euler_step_sde_zero_sigma_matches_ode() {
        // With σ=0, SDE reduces to ODE
        let x = vec![1.0f32, 2.0, 3.0];
        let v = vec![0.5f32, -1.0, 0.2];
        let dt = 0.05;
        let sde = si_euler_step_sde(&x, &v, dt, 0.0, 1);
        let ode = si_euler_step_ode(&x, &v, dt);
        for (a, b) in sde.iter().zip(ode.iter()) {
            assert!((a - b).abs() < TOL, "SDE with σ=0 should match ODE");
        }
    }

    // ── si_ode_integrate ──────────────────────────────────────────────────────

    #[test]
    fn test_ode_integrate_returns_correct_length() {
        let x0 = vec![0.0f32, 1.0, 2.0];
        let config = InterpolantConfig {
            kind: InterpolantKind::LinearDeterministic,
            n_steps: 10,
            eps: 1e-7,
            use_stochastic: false,
            sigma_inf: 0.0,
        };
        let traj = si_ode_integrate(&x0, |x, _t| x.to_vec(), &config).unwrap();
        assert_eq!(traj.len(), config.n_steps + 1);
    }

    #[test]
    fn test_ode_integrate_first_state_equals_x0() {
        let x0 = vec![1.0f32, -2.0, 3.5];
        let config = InterpolantConfig {
            kind: InterpolantKind::LinearDeterministic,
            n_steps: 5,
            eps: 1e-7,
            use_stochastic: false,
            sigma_inf: 0.0,
        };
        let traj = si_ode_integrate(&x0, |x, _t| x.to_vec(), &config).unwrap();
        for (a, b) in traj[0].iter().zip(x0.iter()) {
            assert!((a - b).abs() < TOL);
        }
    }

    #[test]
    fn test_ode_integrate_zero_velocity_stays_at_x0() {
        let x0 = vec![5.0f32, -3.0, 0.5];
        let config = InterpolantConfig {
            kind: InterpolantKind::LinearDeterministic,
            n_steps: 20,
            eps: 1e-7,
            use_stochastic: false,
            sigma_inf: 0.0,
        };
        let traj = si_ode_integrate(&x0, |x, _t| vec![0.0; x.len()], &config).unwrap();
        for state in &traj {
            for (a, b) in state.iter().zip(x0.iter()) {
                assert!((a - b).abs() < TOL, "should stay at x0 with v=0");
            }
        }
    }

    // ── si_linspace ───────────────────────────────────────────────────────────

    #[test]
    fn test_linspace_first_last() {
        let ls = si_linspace(0.0, 1.0, 11);
        assert!((ls[0] - 0.0).abs() < TOL);
        assert!((ls[10] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_linspace_length() {
        let ls = si_linspace(2.0, 5.0, 7);
        assert_eq!(ls.len(), 7);
    }

    #[test]
    fn test_linspace_evenly_spaced() {
        let ls = si_linspace(0.0, 1.0, 6);
        let expected_step = 0.2f32;
        for i in 1..ls.len() {
            assert!((ls[i] - ls[i - 1] - expected_step).abs() < TOL);
        }
    }

    #[test]
    fn test_linspace_single() {
        let ls = si_linspace(3.0, 7.0, 1);
        assert_eq!(ls.len(), 1);
        assert!((ls[0] - 3.0).abs() < TOL);
    }

    // ── si_check_boundary_conditions ─────────────────────────────────────────

    #[test]
    fn test_boundary_check_linear_det_ok() {
        assert!(si_check_boundary_conditions(&InterpolantKind::LinearDeterministic, 1e-4).is_ok());
    }

    #[test]
    fn test_boundary_check_trigonometric_ok() {
        assert!(si_check_boundary_conditions(&InterpolantKind::Trigonometric, 1e-4).is_ok());
    }

    #[test]
    fn test_boundary_check_custom_correct_ok() {
        // Build custom from LinearDeterministic
        let mut alpha_table = [0.0f32; 11];
        let mut beta_table = [0.0f32; 11];
        let gamma_table = [0.0f32; 11];
        for i in 0..=10 {
            let t = i as f32 / 10.0;
            alpha_table[i] = 1.0 - t;
            beta_table[i] = t;
        }
        let kind = InterpolantKind::Custom {
            alpha_table,
            beta_table,
            gamma_table,
        };
        assert!(si_check_boundary_conditions(&kind, 1e-3).is_ok());
    }

    #[test]
    fn test_boundary_check_polynomial_ok() {
        assert!(si_check_boundary_conditions(&InterpolantKind::Polynomial, 1e-4).is_ok());
    }

    // ── VP-SDE ────────────────────────────────────────────────────────────────

    #[test]
    fn test_vpsde_at_t0() {
        let kind = InterpolantKind::VpSde {
            beta_min: 0.1,
            beta_max: 20.0,
        };
        let c = si_eval_coefficients(&kind, 0.0).unwrap();
        assert!(
            (c.alpha - 1.0).abs() < 0.01,
            "alpha(0) should be ~1: {}",
            c.alpha
        );
        assert!(c.gamma.abs() < 0.01, "gamma(0) should be ~0: {}", c.gamma);
    }

    #[test]
    fn test_vpsde_at_t1() {
        let kind = InterpolantKind::VpSde {
            beta_min: 0.1,
            beta_max: 20.0,
        };
        let c = si_eval_coefficients(&kind, 1.0).unwrap();
        assert!(c.alpha < 0.1, "alpha(1) should be ~0: {}", c.alpha);
        assert!(c.gamma > 0.9, "gamma(1) should be ~1: {}", c.gamma);
    }

    #[test]
    fn test_vpsde_alpha_dot_negative() {
        let kind = InterpolantKind::VpSde {
            beta_min: 0.1,
            beta_max: 20.0,
        };
        // alpha_dot should be negative (alpha decreases with t)
        for t in [0.1f32, 0.3, 0.5, 0.7] {
            let c = si_eval_coefficients(&kind, t).unwrap();
            assert!(
                c.alpha_dot < 0.0,
                "alpha_dot should be negative at t={t}: {}",
                c.alpha_dot
            );
        }
    }

    // ── si_compute_stats ──────────────────────────────────────────────────────

    #[test]
    fn test_compute_stats_boundary_passed_linear_det() {
        let n = 4;
        let d = 2;
        let x1 = vec![1.0f32; n * d];
        let t_values = vec![0.5f32; n];
        let kind = InterpolantKind::LinearDeterministic;
        let batch = si_make_batch(&x1, n, d, &kind, &t_values, 1).unwrap();
        let config = InterpolantConfig {
            kind: InterpolantKind::LinearDeterministic,
            n_steps: 10,
            eps: 1e-7,
            use_stochastic: false,
            sigma_inf: 0.0,
        };
        let stats = si_compute_stats(&batch, &config);
        assert!(stats.boundary_check_passed);
    }

    // ── si_format_stats ───────────────────────────────────────────────────────

    #[test]
    fn test_format_stats_nonempty() {
        let stats = SiStats {
            mean_velocity_norm: 1.5,
            mean_noise_level: 0.3,
            boundary_check_passed: true,
            kind_name: "LinearDeterministic".to_string(),
            n_steps: 20,
        };
        let s = si_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("LinearDeterministic"));
    }

    // ── si_format_config ──────────────────────────────────────────────────────

    #[test]
    fn test_format_config_nonempty() {
        let config = InterpolantConfig {
            kind: InterpolantKind::Trigonometric,
            n_steps: 50,
            eps: 1e-7,
            use_stochastic: true,
            sigma_inf: 0.01,
        };
        let s = si_format_config(&config);
        assert!(!s.is_empty());
        assert!(s.contains("Trigonometric"));
    }

    // ── si_kind_name ──────────────────────────────────────────────────────────

    #[test]
    fn test_kind_name_linear_det() {
        assert!(!si_kind_name(&InterpolantKind::LinearDeterministic).is_empty());
    }

    #[test]
    fn test_kind_name_linear_stochastic() {
        assert!(!si_kind_name(&InterpolantKind::LinearStochastic { sigma: 0.5 }).is_empty());
    }

    #[test]
    fn test_kind_name_trigonometric() {
        assert!(!si_kind_name(&InterpolantKind::Trigonometric).is_empty());
    }

    #[test]
    fn test_kind_name_trigonometric_stochastic() {
        assert!(!si_kind_name(&InterpolantKind::TrigonometricStochastic { sigma: 0.5 }).is_empty());
    }

    #[test]
    fn test_kind_name_polynomial() {
        assert!(!si_kind_name(&InterpolantKind::Polynomial).is_empty());
    }

    #[test]
    fn test_kind_name_vpsde() {
        assert!(!si_kind_name(&InterpolantKind::VpSde {
            beta_min: 0.1,
            beta_max: 20.0
        })
        .is_empty());
    }

    #[test]
    fn test_kind_name_custom() {
        let kind = InterpolantKind::Custom {
            alpha_table: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            beta_table: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            gamma_table: [0.0f32; 11],
        };
        assert!(!si_kind_name(&kind).is_empty());
    }

    // ── Additional coverage tests ─────────────────────────────────────────────

    #[test]
    fn test_poly_alpha_plus_beta_plus_gamma() {
        // α+β+γ = (1-t)² + t² + 2t(1-t) = 1 (they sum to 1 for all t)
        for t_i in 0..=10 {
            let t = t_i as f32 / 10.0;
            let c = si_eval_coefficients(&InterpolantKind::Polynomial, t).unwrap();
            let sum = c.alpha + c.beta + c.gamma;
            assert!((sum - 1.0).abs() < TOL, "sum={sum} at t={t}");
        }
    }

    #[test]
    fn test_interpolate_dimension_mismatch_noise() {
        let coeffs = si_eval_coefficients(&InterpolantKind::LinearDeterministic, 0.5).unwrap();
        let x0 = vec![1.0f32, 2.0, 3.0];
        let x1 = vec![4.0f32, 5.0, 6.0];
        let noise = vec![0.0f32, 0.0]; // wrong length
        let result = si_interpolate(&x0, &x1, &noise, &coeffs);
        assert!(matches!(
            result,
            Err(StochasticInterpolantError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_interpolate_empty_returns_empty_batch_error() {
        let coeffs = si_eval_coefficients(&InterpolantKind::LinearDeterministic, 0.5).unwrap();
        let result = si_interpolate(&[], &[], &[], &coeffs);
        assert!(matches!(
            result,
            Err(StochasticInterpolantError::EmptyBatch)
        ));
    }

    #[test]
    fn test_make_batch_t_values_length_mismatch() {
        let x1 = vec![1.0f32; 12];
        let t_values = vec![0.5f32; 5]; // n=4 but 5 t_values
        let result = si_make_batch(
            &x1,
            4,
            3,
            &InterpolantKind::LinearDeterministic,
            &t_values,
            1,
        );
        assert!(matches!(
            result,
            Err(StochasticInterpolantError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_make_batch_invalid_t_value() {
        let x1 = vec![1.0f32; 4];
        let t_values = vec![1.5f32]; // invalid
        let result = si_make_batch(
            &x1,
            1,
            4,
            &InterpolantKind::LinearDeterministic,
            &t_values,
            1,
        );
        assert!(matches!(
            result,
            Err(StochasticInterpolantError::InvalidTimestep { .. })
        ));
    }

    #[test]
    fn test_ode_integrate_empty_x0_error() {
        let config = InterpolantConfig {
            kind: InterpolantKind::LinearDeterministic,
            n_steps: 10,
            eps: 1e-7,
            use_stochastic: false,
            sigma_inf: 0.0,
        };
        let result = si_ode_integrate(&[], |x, _t| x.to_vec(), &config);
        assert!(matches!(
            result,
            Err(StochasticInterpolantError::EmptyBatch)
        ));
    }

    #[test]
    fn test_ode_integrate_zero_steps_error() {
        let config = InterpolantConfig {
            kind: InterpolantKind::LinearDeterministic,
            n_steps: 0,
            eps: 1e-7,
            use_stochastic: false,
            sigma_inf: 0.0,
        };
        let result = si_ode_integrate(&[1.0, 2.0], |x, _t| x.to_vec(), &config);
        assert!(matches!(
            result,
            Err(StochasticInterpolantError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn test_boundary_check_linear_stochastic_ok() {
        let kind = InterpolantKind::LinearStochastic { sigma: 0.5 };
        assert!(si_check_boundary_conditions(&kind, 1e-4).is_ok());
    }

    #[test]
    fn test_trig_stochastic_boundary_ok() {
        let kind = InterpolantKind::TrigonometricStochastic { sigma: 0.5 };
        assert!(si_check_boundary_conditions(&kind, 1e-4).is_ok());
    }

    #[test]
    fn test_vpsde_boundary_conditions() {
        let kind = InterpolantKind::VpSde {
            beta_min: 0.1,
            beta_max: 20.0,
        };
        // VP-SDE has its own boundary conditions (handled separately)
        let result = si_check_boundary_conditions(&kind, 0.05);
        // Should succeed with the VP-SDE specific checks
        assert!(result.is_ok(), "VP-SDE boundary check failed: {:?}", result);
    }

    #[test]
    fn test_compute_stats_fields() {
        let n = 4;
        let d = 2;
        let x1 = vec![0.0f32; n * d];
        let t_values = vec![0.3f32; n];
        let kind = InterpolantKind::Trigonometric;
        let batch = si_make_batch(&x1, n, d, &kind, &t_values, 42).unwrap();
        let config = InterpolantConfig {
            kind: InterpolantKind::Trigonometric,
            n_steps: 20,
            eps: 1e-7,
            use_stochastic: false,
            sigma_inf: 0.0,
        };
        let stats = si_compute_stats(&batch, &config);
        assert_eq!(stats.n_steps, 20);
        assert!(!stats.kind_name.is_empty());
        assert!(stats.mean_velocity_norm >= 0.0);
        assert!(stats.mean_noise_level >= 0.0);
    }

    #[test]
    fn test_linspace_empty() {
        let ls = si_linspace(0.0, 1.0, 0);
        assert!(ls.is_empty());
    }

    #[test]
    fn test_velocity_loss_empty_batch_error() {
        let batch = SiBatch {
            x0: vec![],
            x1: vec![],
            noise: vec![],
            x_t: vec![],
            dx_t: vec![],
            t: vec![],
            coeffs: vec![],
            n: 0,
            d: 4,
        };
        let result = si_velocity_loss(&[], &batch);
        assert!(matches!(
            result,
            Err(StochasticInterpolantError::EmptyBatch)
        ));
    }
}
