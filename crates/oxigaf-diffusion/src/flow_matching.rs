//! # Flow Matching (Conditional Flow Matching)
//!
//! Implements Conditional Flow Matching (CFM) as described in Lipman et al. (2022).
//! Flow matching trains models to predict a velocity field that transports samples
//! from a noise distribution to a data distribution along smooth, optimal transport
//! paths.
//!
//! ## Key advantages over DDPM
//!
//! - **Simpler training**: Direct regression on velocity field, no denoising score matching
//! - **Faster sampling**: Deterministic ODE integration (fewer steps)
//! - **Optimal transport**: Linear paths minimise transport cost
//! - **Stability**: No variance explosion at extreme noise levels
//!
//! ## Supported paths
//!
//! | Path                 | Interpolant                                    |
//! |----------------------|------------------------------------------------|
//! | `Linear`             | `x_t = (1-(1-σ_min)*t)*x_0 + t*x_1`           |
//! | `CosineAnnealing`    | `x_t = cos(t*π/2)*x_0 + sin(t*π/2)*x_1`      |
//! | `VarianceExploding`  | `x_t = x_0 + σ_max·t·x_1`                    |
//! | `VariancePreserving` | `x_t = sqrt(ᾱ)*x_0 + sqrt(1-ᾱ)*x_1`         |
//!
//! ## References
//!
//! - Lipman et al. (2022), "Flow Matching for Generative Modelling"
//! - Albergo & Vanden-Eijnden (2022), "Building Normalizing Flows with Stochastic Interpolants"
//! - Liu et al. (2022), "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"

use std::f32::consts::PI;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise from flow matching operations.
#[derive(Debug, Error, PartialEq)]
pub enum FlowMatchingError {
    /// Input arrays have incompatible lengths.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Time value is outside the valid range [0, 1].
    #[error("Time t={0} must be in [0, 1]")]
    InvalidTime(f32),

    /// A configuration parameter is invalid.
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),

    /// Operation called on empty input.
    #[error("Empty input")]
    EmptyInput,
}

// ─────────────────────────────────────────────────────────────────────────────
// Private PRNG utilities (local copies; public variants live in adaptive_sampling)
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

/// Uniform f32 in [0, 1) using 53 mantissa bits.
#[inline]
fn xorshift_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 11) as f32 / (1u64 << 53) as f32
}

/// Box-Muller transform: maps two uniform samples to a pair of standard normals.
fn fm_box_muller(u1: f32, u2: f32) -> (f32, f32) {
    let r = (-2.0_f32 * u1.max(1e-10).ln()).sqrt();
    let theta = 2.0 * PI * u2;
    (r * theta.cos(), r * theta.sin())
}

/// Sample a single standard-normal deviate (discards the Box-Muller pair partner).
#[inline]
fn fm_sample_standard_normal(state: &mut u64) -> f32 {
    let u1 = xorshift_f32(state);
    let u2 = xorshift_f32(state);
    fm_box_muller(u1, u2).0
}

/// Sample from `Gamma(shape, 1)` via the Marsaglia & Tsang (2000) method.
///
/// Valid for any `shape > 0`. For `shape < 1` it uses the standard boosting
/// identity `Gamma(shape) = Gamma(shape + 1) * U^(1/shape)` (`U ~ Uniform(0,1)`).
/// Unlike Johnk's algorithm, the acceptance rate here does not collapse for
/// large shape parameters.
fn fm_sample_gamma(shape: f32, state: &mut u64) -> f32 {
    if shape < 1.0 {
        let boosted = fm_sample_gamma(shape + 1.0, state);
        let u = xorshift_f32(state).max(1e-12);
        return boosted * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    for _ in 0..1000 {
        let x = fm_sample_standard_normal(state);
        let v_lin = 1.0 + c * x;
        if v_lin <= 0.0 {
            continue;
        }
        let v = v_lin * v_lin * v_lin;
        let u = xorshift_f32(state).max(1e-12);
        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
    // Astronomically unlikely fallback: return the distribution's mean.
    shape.max(1e-6)
}

// ─────────────────────────────────────────────────────────────────────────────
// Flow path types
// ─────────────────────────────────────────────────────────────────────────────

/// Which interpolation path to use between noise (`x_1`) and data (`x_0`).
///
/// Each path defines how `x_t` is constructed at time `t ∈ [0, 1]`,
/// where `t = 0` corresponds to clean data and `t = 1` corresponds to noise.
#[derive(Debug, Clone, PartialEq)]
pub enum FlowPath {
    /// Linear optimal-transport path with a `sigma_min` noise floor:
    /// `x_t = (1-(1-σ_min)·t)·x_0 + t·x_1`.
    ///
    /// Target velocity is constant: `u_t = x_1 - (1-σ_min)·x_0`.
    /// When `σ_min = 0` this reduces to the classic `x_t = (1-t)·x_0 + t·x_1`
    /// / `u_t = x_1 - x_0` optimal-transport path.
    Linear,

    /// Cosine-annealed path: `x_t = cos(t·π/2)·x_0 + sin(t·π/2)·x_1`.
    ///
    /// Smoother transitions near `t = 0` and `t = 1`.
    CosineAnnealing,

    /// Variance-Exploding path: `x_t = x_0 + σ_max·t·x_1`.
    ///
    /// The noise scale grows linearly with time; noise at `t = 1` has
    /// standard deviation `σ_max`.
    VarianceExploding {
        /// Maximum noise standard deviation (must be positive).
        sigma_max: f32,
    },

    /// Variance-Preserving path (DDPM-compatible cosine diffusion):
    /// `x_t = sqrt(ᾱ(t))·x_0 + sqrt(1 − ᾱ(t))·x_1`
    ///
    /// where `ᾱ(t) = cos²(t·π/2)`.
    VariancePreserving,
}

/// How to sample continuous time `t ∈ [0, 1]` during training.
#[derive(Debug, Clone, PartialEq)]
pub enum TimeSchedule {
    /// Uniform distribution over `[0, 1]`.
    Uniform,

    /// Logit-normal distribution: concentrate steps near `t = 0.5`.
    ///
    /// A normal `N(mu, sigma²)` is squashed through the sigmoid function.
    LogitNormal {
        /// Mean of the underlying normal (shift; 0.0 centres on t = 0.5).
        mu: f32,
        /// Std of the underlying normal (spread; larger ⇒ more uniform).
        sigma: f32,
    },

    /// Beta distribution weighting: `Beta(alpha, beta)`.
    ///
    /// Useful to focus training on early (`alpha < 1`) or late (`beta < 1`) steps.
    Beta {
        /// First shape parameter (must be > 0).
        alpha: f32,
        /// Second shape parameter (must be > 0).
        beta: f32,
    },

    /// Cosine-shifted time: maps uniform `u` to `t = acos(1 − 2u) / π`.
    Cosine,
}

/// Configuration for Conditional Flow Matching.
#[derive(Debug, Clone)]
pub struct FlowMatchingConfig {
    /// Interpolation path between noise and data.
    pub path: FlowPath,
    /// Minimum noise std added for numerical stability (default: 0.001).
    pub sigma_min: f32,
    /// How to sample `t` during training.
    pub t_schedule: TimeSchedule,
}

impl Default for FlowMatchingConfig {
    fn default() -> Self {
        Self {
            path: FlowPath::Linear,
            sigma_min: 0.001,
            t_schedule: TimeSchedule::Uniform,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss weighting
// ─────────────────────────────────────────────────────────────────────────────

/// Weighting strategy applied to the flow matching MSE loss.
#[derive(Debug, Clone, PartialEq)]
pub enum FlowLossWeight {
    /// Constant weight = 1.0 at all times.
    Uniform,

    /// Weight = 1 / t — focusses on early generation steps (small t).
    InverseT,

    /// Weight = t·(1 − t) — peaks at t = 0.5, zero at endpoints.
    TimeTDependant,

    /// SNR-like weighting: `σ_min² / (t² + σ_min²)`.
    SnrLike {
        /// Noise floor (same as `FlowMatchingConfig::sigma_min`).
        sigma_min: f32,
    },
}

/// Compute the scalar loss weight for a given time and weighting scheme.
///
/// All returned values are non-negative. Two variants can reach exactly
/// `0.0`: [`FlowLossWeight::TimeTDependant`] vanishes at `t = 0` and `t = 1`
/// (by construction — see its own docs), and [`FlowLossWeight::SnrLike`]
/// approaches `0.0` as `sigma_min -> 0` for any `t > 0`.
pub fn fm_loss_weight(t: f32, weight_fn: &FlowLossWeight) -> f32 {
    match weight_fn {
        FlowLossWeight::Uniform => 1.0,
        FlowLossWeight::InverseT => {
            // Guard against division by zero near t = 0.
            1.0 / t.max(1e-6)
        }
        FlowLossWeight::TimeTDependant => t * (1.0 - t),
        FlowLossWeight::SnrLike { sigma_min } => {
            let s = *sigma_min;
            let s2 = s * s;
            s2 / (t * t + s2).max(1e-12)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Time sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Sample a continuous time value `t ∈ [0, 1]` according to the given schedule.
///
/// Calls `xorshift_f32` / Box-Muller internally; no external PRNG dependency.
pub fn fm_sample_time(schedule: &TimeSchedule, state: &mut u64) -> f32 {
    match schedule {
        TimeSchedule::Uniform => xorshift_f32(state),

        TimeSchedule::LogitNormal { mu, sigma } => {
            let u1 = xorshift_f32(state);
            let u2 = xorshift_f32(state);
            let (z, _) = fm_box_muller(u1, u2);
            let x = mu + sigma * z;
            // sigmoid(x) maps ℝ → (0, 1)
            1.0 / (1.0 + (-x).exp())
        }

        TimeSchedule::Beta { alpha, beta } => {
            // Gamma-ratio method: X ~ Gamma(a,1), Y ~ Gamma(b,1) independent
            // => X / (X+Y) ~ Beta(a,b) exactly (for any a,b > 0). This is
            // exact for all shape parameters, unlike Johnk's algorithm
            // (X=U^(1/a), Y=V^(1/b), accept if X+Y<=1) whose acceptance rate
            // collapses as a or b grow. Clamp to (0.001, 0.999) for the same
            // numerical-stability reason `t` is clamped elsewhere in this module.
            let a = alpha.max(1e-3);
            let b = beta.max(1e-3);
            let x = fm_sample_gamma(a, state);
            let y = fm_sample_gamma(b, state);
            let denom = x + y;
            let sample = if denom > 0.0 { x / denom } else { 0.5 };
            sample.clamp(0.001, 0.999)
        }

        TimeSchedule::Cosine => {
            // Maps uniform u ∈ [0, 1] → t = acos(1 − 2u) / π.
            // At u = 0 → t = 0, at u = 1 → t = 1; density peaks near endpoints.
            let u = xorshift_f32(state);
            let arg = (1.0 - 2.0 * u).clamp(-1.0 + 1e-7, 1.0 - 1e-7);
            arg.acos() / PI
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core interpolation and velocity
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that `t ∈ [0, 1]` and that arrays have the same non-zero length.
fn check_dims_and_time(x_0: &[f32], x_1: &[f32], t: f32) -> Result<(), FlowMatchingError> {
    if !(0.0..=1.0).contains(&t) {
        return Err(FlowMatchingError::InvalidTime(t));
    }
    if x_0.is_empty() {
        return Err(FlowMatchingError::EmptyInput);
    }
    if x_0.len() != x_1.len() {
        return Err(FlowMatchingError::DimensionMismatch {
            expected: x_0.len(),
            got: x_1.len(),
        });
    }
    Ok(())
}

/// Cosine-schedule cumulative signal power: `ᾱ(t) = cos²(t·π/2)`.
#[inline]
fn alpha_bar_cosine(t: f32) -> f32 {
    let c = (t * PI / 2.0).cos();
    c * c
}

/// Compute the interpolated sample `x_t` at time `t` along the chosen flow path.
///
/// - `x_0`: clean data sample (target distribution)
/// - `x_1`: noise sample (source distribution, typically N(0, I))
/// - `t = 0` → returns `x_0` exactly.
/// - `t = 1` → returns `x_1` when `config.sigma_min == 0`; for the `Linear`
///   path with `sigma_min > 0` it instead returns `sigma_min * x_0 + x_1`,
///   the standard Conditional Flow Matching noise floor (Lipman et al.
///   2022) that keeps the marginal at `t = 1` non-degenerate.
pub fn fm_interpolate(
    x_0: &[f32],
    x_1: &[f32],
    t: f32,
    config: &FlowMatchingConfig,
) -> Result<Vec<f32>, FlowMatchingError> {
    check_dims_and_time(x_0, x_1, t)?;
    let n = x_0.len();
    let mut out = Vec::with_capacity(n);

    match &config.path {
        FlowPath::Linear => {
            // Conditional Flow Matching linear path (Lipman et al. 2022):
            // x_t = (1 - (1 - sigma_min)*t) * x_0 + t * x_1
            let coeff_0 = 1.0 - (1.0 - config.sigma_min) * t;
            for i in 0..n {
                out.push(coeff_0 * x_0[i] + t * x_1[i]);
            }
        }
        FlowPath::CosineAnnealing => {
            let alpha = (t * PI / 2.0).cos();
            let beta = (t * PI / 2.0).sin();
            for i in 0..n {
                out.push(alpha * x_0[i] + beta * x_1[i]);
            }
        }
        FlowPath::VarianceExploding { sigma_max } => {
            // x_t = x_0 + sigma_max * t * x_1
            let scale = sigma_max * t;
            for i in 0..n {
                out.push(x_0[i] + scale * x_1[i]);
            }
        }
        FlowPath::VariancePreserving => {
            // x_t = sqrt(ᾱ) * x_0 + sqrt(1 - ᾱ) * x_1  where ᾱ = cos²(t*π/2)
            let ab = alpha_bar_cosine(t);
            let sqrt_ab = ab.max(0.0).sqrt();
            let sqrt_1_ab = (1.0 - ab).max(0.0).sqrt();
            for i in 0..n {
                out.push(sqrt_ab * x_0[i] + sqrt_1_ab * x_1[i]);
            }
        }
    }
    Ok(out)
}

/// Compute the target velocity field `u_t(x_t)` that a model should predict.
///
/// The velocity is the time-derivative of the interpolant: `dx_t/dt`.
pub fn fm_target_velocity(
    x_0: &[f32],
    x_1: &[f32],
    t: f32,
    config: &FlowMatchingConfig,
) -> Result<Vec<f32>, FlowMatchingError> {
    check_dims_and_time(x_0, x_1, t)?;
    let n = x_0.len();
    let mut out = Vec::with_capacity(n);

    match &config.path {
        FlowPath::Linear => {
            // dx_t/dt = x_1 - (1 - sigma_min) * x_0  (constant for all t)
            let coeff_0 = 1.0 - config.sigma_min;
            for i in 0..n {
                out.push(x_1[i] - coeff_0 * x_0[i]);
            }
        }
        FlowPath::CosineAnnealing => {
            // d/dt [cos(t*π/2)*x_0 + sin(t*π/2)*x_1]
            //   = -sin(t*π/2)*(π/2)*x_0 + cos(t*π/2)*(π/2)*x_1
            let half_pi = PI / 2.0;
            let sin_t = (t * half_pi).sin();
            let cos_t = (t * half_pi).cos();
            for i in 0..n {
                out.push(-sin_t * half_pi * x_0[i] + cos_t * half_pi * x_1[i]);
            }
        }
        FlowPath::VarianceExploding { sigma_max } => {
            // d/dt [x_0 + sigma_max * t * x_1] = sigma_max * x_1
            let s = *sigma_max;
            for &xi in x_1.iter().take(n) {
                out.push(s * xi);
            }
        }
        FlowPath::VariancePreserving => {
            // x_t = sqrt(ᾱ(t)) * x_0 + sqrt(1-ᾱ(t)) * x_1
            // ᾱ(t) = cos²(t*π/2)
            // d(sqrt(ᾱ))/dt = -sin(t*π/2)*cos(t*π/2)*(π/2) / sqrt(ᾱ)
            //                = -sin(t*π/2) * (π/2)   [since sqrt(ᾱ) = cos(t*π/2)]
            // d(sqrt(1-ᾱ))/dt = sin(t*π/2)*cos(t*π/2)*(π/2) / sqrt(1-ᾱ)
            //                  = cos(t*π/2) * (π/2)   [since sqrt(1-ᾱ) = sin(t*π/2)]
            let half_pi = PI / 2.0;
            let sin_t = (t * half_pi).sin();
            let cos_t = (t * half_pi).cos();
            // v_t = -sin(t*π/2)*(π/2)*x_0 + cos(t*π/2)*(π/2)*x_1
            for i in 0..n {
                out.push(-sin_t * half_pi * x_0[i] + cos_t * half_pi * x_1[i]);
            }
        }
    }
    Ok(out)
}

/// Reconstruct the clean sample `x_0` from a noisy observation `x_t` and
/// the model's predicted velocity `v_t`.
pub fn fm_reconstruct_x0(
    x_t: &[f32],
    v_t: &[f32],
    t: f32,
    config: &FlowMatchingConfig,
) -> Result<Vec<f32>, FlowMatchingError> {
    if !(0.0..=1.0).contains(&t) {
        return Err(FlowMatchingError::InvalidTime(t));
    }
    if x_t.is_empty() {
        return Err(FlowMatchingError::EmptyInput);
    }
    if x_t.len() != v_t.len() {
        return Err(FlowMatchingError::DimensionMismatch {
            expected: x_t.len(),
            got: v_t.len(),
        });
    }
    let n = x_t.len();
    let mut out = Vec::with_capacity(n);

    match &config.path {
        FlowPath::Linear => {
            // x_t = (1-(1-sigma_min)t)*x_0 + t*x_1, v_t = x_1 - (1-sigma_min)*x_0
            // Substituting x_1 = v_t + (1-sigma_min)*x_0:
            //   x_t = (1-(1-sigma_min)t)*x_0 + t*(v_t + (1-sigma_min)*x_0)
            //       = x_0 + t*v_t
            // => x_0 = x_t - t*v_t   (independent of sigma_min)
            for i in 0..n {
                out.push(x_t[i] - t * v_t[i]);
            }
        }
        FlowPath::CosineAnnealing => {
            // x_t = cos(t*π/2)*x_0 + sin(t*π/2)*x_1
            // v_t = -sin(t*π/2)*(π/2)*x_0 + cos(t*π/2)*(π/2)*x_1
            // Solve for x_0:
            //   x_0 = alpha * x_t - (2/π) * beta * v_t
            // where alpha = cos(t*π/2), beta = sin(t*π/2)
            let alpha = (t * PI / 2.0).cos();
            let beta = (t * PI / 2.0).sin();
            let two_over_pi = 2.0 / PI;
            for i in 0..n {
                out.push(alpha * x_t[i] - two_over_pi * beta * v_t[i]);
            }
        }
        FlowPath::VarianceExploding { sigma_max: _ } => {
            // x_t = x_0 + sigma_max * t * x_1, v_t = sigma_max * x_1
            // => sigma_max * x_1 = v_t
            // => x_0 = x_t - t * v_t
            for i in 0..n {
                out.push(x_t[i] - t * v_t[i]);
            }
        }
        FlowPath::VariancePreserving => {
            // VP path has same analytic form as CosineAnnealing for reconstruction:
            // x_t = sqrt(ᾱ)*x_0 + sqrt(1-ᾱ)*x_1
            //   where sqrt(ᾱ) = cos(t*π/2), sqrt(1-ᾱ) = sin(t*π/2)
            // v_t = -sin(t*π/2)*(π/2)*x_0 + cos(t*π/2)*(π/2)*x_1
            // Same closed-form solution as CosineAnnealing:
            let alpha = (t * PI / 2.0).cos();
            let beta = (t * PI / 2.0).sin();
            let two_over_pi = 2.0 / PI;
            for i in 0..n {
                out.push(alpha * x_t[i] - two_over_pi * beta * v_t[i]);
            }
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration steps
// ─────────────────────────────────────────────────────────────────────────────

/// Euler integration step: `x_{t+dt} = x_t + dt · v_t`.
///
/// Use negative `dt` when integrating from noise (`t = 1`) to data (`t = 0`).
pub fn fm_euler_step(
    x_t: &[f32],
    v_t: &[f32],
    _t: f32,
    dt: f32,
) -> Result<Vec<f32>, FlowMatchingError> {
    if x_t.is_empty() {
        return Err(FlowMatchingError::EmptyInput);
    }
    if x_t.len() != v_t.len() {
        return Err(FlowMatchingError::DimensionMismatch {
            expected: x_t.len(),
            got: v_t.len(),
        });
    }
    Ok(x_t
        .iter()
        .zip(v_t.iter())
        .map(|(&x, &v)| x + dt * v)
        .collect())
}

/// Heun (trapezoidal / midpoint) integration step for higher accuracy.
///
/// Averages the velocity at `t` and at `t + dt`, giving second-order accuracy.
///
/// `x_{t+dt} = x_t + dt · (v_t + v_{t+dt}) / 2`
pub fn fm_heun_step(
    x_t: &[f32],
    v_t: &[f32],
    v_t_next: &[f32],
    dt: f32,
) -> Result<Vec<f32>, FlowMatchingError> {
    if x_t.is_empty() {
        return Err(FlowMatchingError::EmptyInput);
    }
    if x_t.len() != v_t.len() {
        return Err(FlowMatchingError::DimensionMismatch {
            expected: x_t.len(),
            got: v_t.len(),
        });
    }
    if x_t.len() != v_t_next.len() {
        return Err(FlowMatchingError::DimensionMismatch {
            expected: x_t.len(),
            got: v_t_next.len(),
        });
    }
    Ok(x_t
        .iter()
        .zip(v_t.iter())
        .zip(v_t_next.iter())
        .map(|((&x, &v0), &v1)| x + dt * 0.5 * (v0 + v1))
        .collect())
}

/// Generate an evenly-spaced sequence of timesteps for inference.
///
/// Returns `n_steps + 1` values from `1.0` (noise) down to `0.0` (data):
/// `[1.0, 1.0 - 1/n_steps, …, 0.0]`. For `n_steps == 0` there is no step
/// size to divide by, so the single-element trajectory `[1.0]` is returned
/// (still `n_steps + 1 = 1` values, per the contract above) instead of
/// dividing by zero.
pub fn fm_inference_timesteps(n_steps: usize) -> Vec<f32> {
    if n_steps == 0 {
        return vec![1.0];
    }
    let n = n_steps + 1;
    (0..n).map(|i| 1.0 - i as f32 / n_steps as f32).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss
// ─────────────────────────────────────────────────────────────────────────────

/// Conditional Flow Matching loss: weighted MSE between predicted and target velocity.
///
/// Returns `weight(t) * mean((predicted - target)²)`.
pub fn fm_loss(
    predicted_velocity: &[f32],
    target_velocity: &[f32],
    t: f32,
    weight_fn: &FlowLossWeight,
) -> Result<f32, FlowMatchingError> {
    if !(0.0..=1.0).contains(&t) {
        return Err(FlowMatchingError::InvalidTime(t));
    }
    if predicted_velocity.is_empty() {
        return Err(FlowMatchingError::EmptyInput);
    }
    if predicted_velocity.len() != target_velocity.len() {
        return Err(FlowMatchingError::DimensionMismatch {
            expected: predicted_velocity.len(),
            got: target_velocity.len(),
        });
    }

    let mse: f32 = predicted_velocity
        .iter()
        .zip(target_velocity.iter())
        .map(|(&p, &tgt)| {
            let diff = p - tgt;
            diff * diff
        })
        .sum::<f32>()
        / predicted_velocity.len() as f32;

    let w = fm_loss_weight(t, weight_fn);
    Ok(w * mse)
}

// ─────────────────────────────────────────────────────────────────────────────
// Stochastic interpolant
// ─────────────────────────────────────────────────────────────────────────────

/// Sample `n` independent standard normal values using xorshift64 + Box-Muller.
pub fn fm_sample_noise(n: usize, state: &mut u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let u1 = xorshift_f32(state);
        let u2 = xorshift_f32(state);
        let (z0, z1) = fm_box_muller(u1, u2);
        out.push(z0);
        if i + 1 < n {
            out.push(z1);
        }
        i += 2;
    }
    out
}

/// Stochastic interpolant: adds an additional noise term with strength `gamma`.
///
/// `x_t = alpha(t)*x_0 + beta(t)*x_1 + gamma * sqrt(alpha(t)*beta(t)) * noise`
///
/// When `gamma = 0` the result is identical to [`fm_interpolate`] on the
/// `Linear` path with `sigma_min = 0` (this function always uses the pure
/// `alpha = 1-t`, `beta = t` interpolant and has no `sigma_min` of its own).
pub fn fm_stochastic_interpolant(
    x_0: &[f32],
    x_1: &[f32],
    t: f32,
    noise: &[f32],
    gamma: f32,
) -> Result<Vec<f32>, FlowMatchingError> {
    if !(0.0..=1.0).contains(&t) {
        return Err(FlowMatchingError::InvalidTime(t));
    }
    if x_0.is_empty() {
        return Err(FlowMatchingError::EmptyInput);
    }
    if x_0.len() != x_1.len() {
        return Err(FlowMatchingError::DimensionMismatch {
            expected: x_0.len(),
            got: x_1.len(),
        });
    }
    if x_0.len() != noise.len() {
        return Err(FlowMatchingError::DimensionMismatch {
            expected: x_0.len(),
            got: noise.len(),
        });
    }
    if gamma < 0.0 {
        return Err(FlowMatchingError::InvalidParam(
            "gamma must be non-negative".to_string(),
        ));
    }

    // Use linear path for the base interpolant (simplest stochastic interpolant).
    let alpha = 1.0 - t;
    let beta = t;
    // Noise scale: gamma * sqrt(alpha * beta).
    let noise_scale = gamma * (alpha * beta).max(0.0).sqrt();

    let n = x_0.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(alpha * x_0[i] + beta * x_1[i] + noise_scale * noise[i]);
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Flow integrator
// ─────────────────────────────────────────────────────────────────────────────

/// ODE integrator for flow matching inference.
///
/// Integrates from `x_1` (noise, `t = 1`) to `x_0` (data, `t = 0`) using either
/// Euler's method or Heun's second-order corrector.
pub struct FlowIntegrator {
    /// Flow path and schedule configuration.
    pub config: FlowMatchingConfig,
    /// Number of integration steps.
    pub n_steps: usize,
    /// If true, use Heun's method (requires two velocity evaluations per step).
    pub use_heun: bool,
}

impl FlowIntegrator {
    /// Creates a new integrator with the given configuration.
    pub fn new(config: FlowMatchingConfig, n_steps: usize, use_heun: bool) -> Self {
        Self {
            config,
            n_steps,
            use_heun,
        }
    }

    /// Integrate from `x_1` (noise) to `x_0` (data) using the given velocity function.
    ///
    /// `velocity_fn(x_t, t)` should return the predicted velocity at `(x_t, t)`.
    ///
    /// Integration proceeds from `t = 1.0` to `t = 0.0` in `n_steps` equal steps.
    pub fn integrate<F>(&self, x_1: &[f32], velocity_fn: F) -> Result<Vec<f32>, FlowMatchingError>
    where
        F: Fn(&[f32], f32) -> Result<Vec<f32>, FlowMatchingError>,
    {
        if x_1.is_empty() {
            return Err(FlowMatchingError::EmptyInput);
        }
        if self.n_steps == 0 {
            return Err(FlowMatchingError::InvalidParam(
                "n_steps must be >= 1".to_string(),
            ));
        }

        let timesteps = fm_inference_timesteps(self.n_steps);
        let dt = -1.0 / self.n_steps as f32;

        let mut x = x_1.to_vec();

        for &t in timesteps.iter().take(self.n_steps) {
            let v = velocity_fn(&x, t)?;

            if self.use_heun {
                // Euler predictor step.
                let x_pred = fm_euler_step(&x, &v, t, dt)?;
                let t_next = (t + dt).max(0.0);
                // Velocity at predicted next position.
                let v_next = velocity_fn(&x_pred, t_next)?;
                // Heun corrector.
                x = fm_heun_step(&x, &v, &v_next, dt)?;
            } else {
                x = fm_euler_step(&x, &v, t, dt)?;
            }
        }

        Ok(x)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics and utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Statistical summary of a batch of velocity predictions.
#[derive(Debug, Clone)]
pub struct FlowStats {
    /// Mean Euclidean norm of velocity vectors.
    pub mean_velocity_norm: f32,
    /// Maximum Euclidean norm across the batch.
    pub max_velocity_norm: f32,
    /// Mean of `sum(v_i²)` over the batch (i.e. the mean squared L2 norm).
    ///
    /// This is *not* an estimate of the velocity field's divergence
    /// (`div v = sum_i dv_i/dx_i`, the trace of its Jacobian) — computing
    /// that would require evaluating the velocity function at perturbed
    /// inputs (see `score_matching::sm_hutchinson_trace_estimate` for a
    /// genuine Hutchinson trace estimator). This field is a magnitude
    /// diagnostic only.
    pub mean_squared_norm: f32,
    /// Smoothness indicator: `1 − std(norms) / (mean(norms) + ε)`.
    pub smoothness: f32,
}

/// Compute statistics over a batch of velocity predictions.
///
/// `velocities` is a slice of velocity vectors; each inner `Vec<f32>` is one sample.
pub fn compute_flow_stats(velocities: &[Vec<f32>]) -> Result<FlowStats, FlowMatchingError> {
    if velocities.is_empty() {
        return Err(FlowMatchingError::EmptyInput);
    }
    for (i, v) in velocities.iter().enumerate() {
        if v.is_empty() {
            return Err(FlowMatchingError::InvalidParam(format!(
                "velocity at index {} is empty",
                i
            )));
        }
    }

    let norms: Vec<f32> = velocities
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
        .collect();

    let n = norms.len() as f32;
    let mean_norm = norms.iter().sum::<f32>() / n;
    let max_norm = norms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // mean_squared_norm: mean of sum(v_i²)
    let mean_sq_norm: f32 = velocities
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum::<f32>())
        .sum::<f32>()
        / n;

    // Smoothness: 1 - std(norms) / (mean(norms) + 1e-6)
    let variance = norms.iter().map(|&x| (x - mean_norm).powi(2)).sum::<f32>() / n;
    let std_norm = variance.sqrt();
    let smoothness = 1.0 - std_norm / (mean_norm + 1e-6);

    Ok(FlowStats {
        mean_velocity_norm: mean_norm,
        max_velocity_norm: max_norm,
        mean_squared_norm: mean_sq_norm,
        smoothness,
    })
}

/// Sample a matched pair `(x_0, x_1)` of standard normal vectors of length `n`.
///
/// `x_0` represents a data sample; `x_1` represents a noise sample.
/// Both are independently sampled from N(0, I).
pub fn fm_sample_pair(n: usize, state: &mut u64) -> (Vec<f32>, Vec<f32>) {
    let x_0 = fm_sample_noise(n, state);
    let x_1 = fm_sample_noise(n, state);
    (x_0, x_1)
}

/// Format a [`FlowMatchingConfig`] as a human-readable string.
pub fn format_flow_config(config: &FlowMatchingConfig) -> String {
    let path_str = match &config.path {
        FlowPath::Linear => "Linear".to_string(),
        FlowPath::CosineAnnealing => "CosineAnnealing".to_string(),
        FlowPath::VarianceExploding { sigma_max } => {
            format!("VarianceExploding(sigma_max={:.4})", sigma_max)
        }
        FlowPath::VariancePreserving => "VariancePreserving".to_string(),
    };
    let sched_str = match &config.t_schedule {
        TimeSchedule::Uniform => "Uniform".to_string(),
        TimeSchedule::LogitNormal { mu, sigma } => {
            format!("LogitNormal(mu={:.3}, sigma={:.3})", mu, sigma)
        }
        TimeSchedule::Beta { alpha, beta } => {
            format!("Beta(alpha={:.3}, beta={:.3})", alpha, beta)
        }
        TimeSchedule::Cosine => "Cosine".to_string(),
    };
    format!(
        "FlowMatchingConfig {{ path: {}, sigma_min: {:.6}, t_schedule: {} }}",
        path_str, config.sigma_min, sched_str
    )
}

/// Format [`FlowStats`] as a human-readable string.
pub fn format_flow_stats(stats: &FlowStats) -> String {
    format!(
        "FlowStats {{ mean_norm: {:.6}, max_norm: {:.6}, mean_sq_norm: {:.6}, smoothness: {:.6} }}",
        stats.mean_velocity_norm,
        stats.max_velocity_norm,
        stats.mean_squared_norm,
        stats.smoothness,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PRNG helpers ──────────────────────────────────────────────────────────

    fn test_state() -> u64 {
        0xDEAD_BEEF_1234_5678u64
    }

    // ── fm_sample_time ────────────────────────────────────────────────────────

    #[test]
    fn test_sample_time_uniform_in_range() {
        let mut state = test_state();
        for _ in 0..1000 {
            let t = fm_sample_time(&TimeSchedule::Uniform, &mut state);
            assert!((0.0..=1.0).contains(&t), "uniform t={} out of [0,1]", t);
        }
    }

    #[test]
    fn test_sample_time_uniform_spread() {
        let mut state = test_state();
        let mut sum = 0.0f32;
        let n = 10_000;
        for _ in 0..n {
            sum += fm_sample_time(&TimeSchedule::Uniform, &mut state);
        }
        let mean = sum / n as f32;
        // Expect mean near 0.5 ± 0.05
        assert!(
            (mean - 0.5).abs() < 0.05,
            "uniform mean={} too far from 0.5",
            mean
        );
    }

    #[test]
    fn test_sample_time_logit_normal_in_range() {
        let mut state = test_state();
        let sched = TimeSchedule::LogitNormal {
            mu: 0.0,
            sigma: 1.0,
        };
        for _ in 0..1000 {
            let t = fm_sample_time(&sched, &mut state);
            assert!(t > 0.0 && t < 1.0, "logit-normal t={} out of (0,1)", t);
        }
    }

    #[test]
    fn test_sample_time_logit_normal_mean_near_half() {
        let mut state = test_state();
        let sched = TimeSchedule::LogitNormal {
            mu: 0.0,
            sigma: 1.0,
        };
        let mut sum = 0.0f32;
        let n = 5_000;
        for _ in 0..n {
            sum += fm_sample_time(&sched, &mut state);
        }
        let mean = sum / n as f32;
        assert!(
            (mean - 0.5).abs() < 0.1,
            "logit-normal mean={} too far from 0.5",
            mean
        );
    }

    #[test]
    fn test_sample_time_cosine_in_range() {
        let mut state = test_state();
        let sched = TimeSchedule::Cosine;
        for _ in 0..1000 {
            let t = fm_sample_time(&sched, &mut state);
            assert!((0.0..=1.0).contains(&t), "cosine t={} out of [0,1]", t);
        }
    }

    #[test]
    fn test_sample_time_beta_in_range() {
        let mut state = test_state();
        let sched = TimeSchedule::Beta {
            alpha: 2.0,
            beta: 2.0,
        };
        for _ in 0..1000 {
            let t = fm_sample_time(&sched, &mut state);
            assert!(t > 0.0 && t < 1.0, "beta t={} out of (0,1)", t);
        }
    }

    #[test]
    fn test_sample_time_beta_symmetric() {
        // Beta(2,2) is symmetric around 0.5
        let mut state = test_state();
        let sched = TimeSchedule::Beta {
            alpha: 2.0,
            beta: 2.0,
        };
        let mut sum = 0.0f32;
        let n = 5_000;
        for _ in 0..n {
            sum += fm_sample_time(&sched, &mut state);
        }
        let mean = sum / n as f32;
        assert!(
            (mean - 0.5).abs() < 0.15,
            "Beta(2,2) mean={} too far from 0.5",
            mean
        );
    }

    #[test]
    fn test_sample_time_beta_1_1_is_uniform() {
        // Beta(1,1) is exactly Uniform[0,1]. Johnk's algorithm *without* the
        // rejection test (the original bug) instead returns U/(U+V), a
        // distribution that is symmetric around 0.5 with the SAME mean
        // (0.5) but is bell-shaped rather than flat — so a mean-only check
        // cannot catch the bug. Check flatness too: P(T < 0.25) should be
        // ~0.25 for a true Uniform, but only ~0.167 for U/(U+V).
        let mut state = test_state();
        let sched = TimeSchedule::Beta {
            alpha: 1.0,
            beta: 1.0,
        };
        let n = 20_000;
        let mut sum = 0.0f32;
        let mut below_quarter = 0u32;
        for _ in 0..n {
            let t = fm_sample_time(&sched, &mut state);
            sum += t;
            if t < 0.25 {
                below_quarter += 1;
            }
        }
        let mean = sum / n as f32;
        assert!((mean - 0.5).abs() < 0.02, "Beta(1,1) mean={} != 0.5", mean);
        let frac_below_quarter = below_quarter as f32 / n as f32;
        assert!(
            (frac_below_quarter - 0.25).abs() < 0.03,
            "Beta(1,1) should be flat/uniform: P(t<0.25)={} (expected ~0.25)",
            frac_below_quarter
        );
    }

    #[test]
    fn test_sample_time_beta_shape_below_one_in_range() {
        // Exercises the shape < 1 boosting branch of the gamma sampler.
        let mut state = test_state();
        let sched = TimeSchedule::Beta {
            alpha: 0.5,
            beta: 0.5,
        };
        for _ in 0..2000 {
            let t = fm_sample_time(&sched, &mut state);
            assert!(
                (0.0..=1.0).contains(&t),
                "Beta(0.5,0.5) t={} out of [0,1]",
                t
            );
        }
    }

    // ── fm_interpolate ────────────────────────────────────────────────────────

    #[test]
    fn test_interpolate_linear_t0_returns_x0() {
        let cfg = FlowMatchingConfig::default();
        let x_0 = vec![1.0, 2.0, 3.0];
        let x_1 = vec![4.0, 5.0, 6.0];
        let out = fm_interpolate(&x_0, &x_1, 0.0, &cfg).unwrap();
        for (a, b) in out.iter().zip(x_0.iter()) {
            assert!((a - b).abs() < 1e-6, "t=0 interpolate != x_0");
        }
    }

    #[test]
    fn test_interpolate_linear_t1_returns_x1_when_sigma_min_zero() {
        // The exact "t=1 -> x_1" boundary only holds for sigma_min = 0.
        let cfg = FlowMatchingConfig {
            sigma_min: 0.0,
            ..Default::default()
        };
        let x_0 = vec![1.0, 2.0, 3.0];
        let x_1 = vec![4.0, 5.0, 6.0];
        let out = fm_interpolate(&x_0, &x_1, 1.0, &cfg).unwrap();
        for (a, b) in out.iter().zip(x_1.iter()) {
            assert!((a - b).abs() < 1e-6, "t=1 interpolate != x_1");
        }
    }

    #[test]
    fn test_interpolate_linear_sigma_min_leaves_floor_at_t1() {
        // Regression test: FlowMatchingConfig::sigma_min must actually be
        // consulted by fm_interpolate. At t=1 the Linear path should equal
        // sigma_min * x_0 + x_1 (Lipman et al. 2022 CFM noise floor), not
        // exactly x_1.
        let cfg = FlowMatchingConfig {
            sigma_min: 0.1,
            ..Default::default()
        };
        let x_0 = vec![1.0, 2.0, 3.0];
        let x_1 = vec![4.0, 5.0, 6.0];
        let out = fm_interpolate(&x_0, &x_1, 1.0, &cfg).unwrap();
        for i in 0..3 {
            let expected = 0.1 * x_0[i] + x_1[i];
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "sigma_min floor not applied: {} vs {}",
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_interpolate_linear_t_half_is_midpoint() {
        let cfg = FlowMatchingConfig::default();
        let x_0 = vec![0.0, 0.0];
        let x_1 = vec![2.0, 4.0];
        let out = fm_interpolate(&x_0, &x_1, 0.5, &cfg).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_cosine_t0_returns_x0() {
        let cfg = FlowMatchingConfig {
            path: FlowPath::CosineAnnealing,
            ..Default::default()
        };
        let x_0 = vec![1.0, -1.0];
        let x_1 = vec![0.0, 0.0];
        let out = fm_interpolate(&x_0, &x_1, 0.0, &cfg).unwrap();
        for (a, b) in out.iter().zip(x_0.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_interpolate_cosine_t1_returns_x1() {
        let cfg = FlowMatchingConfig {
            path: FlowPath::CosineAnnealing,
            ..Default::default()
        };
        let x_0 = vec![5.0];
        let x_1 = vec![3.0];
        let out = fm_interpolate(&x_0, &x_1, 1.0, &cfg).unwrap();
        assert!((out[0] - x_1[0]).abs() < 1e-5);
    }

    #[test]
    fn test_interpolate_ve_t0_returns_x0() {
        let cfg = FlowMatchingConfig {
            path: FlowPath::VarianceExploding { sigma_max: 2.0 },
            ..Default::default()
        };
        let x_0 = vec![1.0, 2.0];
        let x_1 = vec![1.0, 1.0];
        let out = fm_interpolate(&x_0, &x_1, 0.0, &cfg).unwrap();
        assert!((out[0] - x_0[0]).abs() < 1e-6);
        assert!((out[1] - x_0[1]).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_vp_t0_returns_x0() {
        let cfg = FlowMatchingConfig {
            path: FlowPath::VariancePreserving,
            ..Default::default()
        };
        let x_0 = vec![1.0, 2.0];
        let x_1 = vec![3.0, 4.0];
        let out = fm_interpolate(&x_0, &x_1, 0.0, &cfg).unwrap();
        assert!((out[0] - x_0[0]).abs() < 1e-6);
        assert!((out[1] - x_0[1]).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_dim_mismatch() {
        let cfg = FlowMatchingConfig::default();
        let err = fm_interpolate(&[1.0, 2.0], &[1.0], 0.5, &cfg).unwrap_err();
        assert!(matches!(err, FlowMatchingError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_interpolate_invalid_time_negative() {
        let cfg = FlowMatchingConfig::default();
        let err = fm_interpolate(&[1.0], &[0.0], -0.1, &cfg).unwrap_err();
        assert!(matches!(err, FlowMatchingError::InvalidTime(_)));
    }

    #[test]
    fn test_interpolate_invalid_time_above_one() {
        let cfg = FlowMatchingConfig::default();
        let err = fm_interpolate(&[1.0], &[0.0], 1.1, &cfg).unwrap_err();
        assert!(matches!(err, FlowMatchingError::InvalidTime(_)));
    }

    // ── fm_target_velocity ────────────────────────────────────────────────────

    #[test]
    fn test_target_velocity_linear_is_x1_minus_x0_when_sigma_min_zero() {
        // v_t = x_1 - (1 - sigma_min) * x_0 reduces to x_1 - x_0 only when
        // sigma_min = 0.
        let cfg = FlowMatchingConfig {
            sigma_min: 0.0,
            ..Default::default()
        };
        let x_0 = vec![1.0, 2.0, 3.0];
        let x_1 = vec![4.0, 5.0, 6.0];
        for t in [0.0, 0.3, 0.7, 1.0] {
            let v = fm_target_velocity(&x_0, &x_1, t, &cfg).unwrap();
            assert!((v[0] - 3.0).abs() < 1e-6);
            assert!((v[1] - 3.0).abs() < 1e-6);
            assert!((v[2] - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_target_velocity_linear_sigma_min_shifts_velocity() {
        // Regression test: FlowMatchingConfig::sigma_min must actually be
        // consulted by fm_target_velocity: v_t = x_1 - (1-sigma_min)*x_0.
        let cfg = FlowMatchingConfig {
            sigma_min: 0.1,
            ..Default::default()
        };
        let x_0 = vec![1.0, 2.0, 3.0];
        let x_1 = vec![4.0, 5.0, 6.0];
        let v = fm_target_velocity(&x_0, &x_1, 0.5, &cfg).unwrap();
        for i in 0..3 {
            let expected = x_1[i] - 0.9 * x_0[i];
            assert!(
                (v[i] - expected).abs() < 1e-5,
                "sigma_min not applied to target velocity: {} vs {}",
                v[i],
                expected
            );
        }
    }

    #[test]
    fn test_target_velocity_cosine_at_t0() {
        // At t=0: v = -sin(0)*(π/2)*x_0 + cos(0)*(π/2)*x_1 = (π/2)*x_1
        let cfg = FlowMatchingConfig {
            path: FlowPath::CosineAnnealing,
            ..Default::default()
        };
        let x_0 = vec![0.0, 0.0];
        let x_1 = vec![1.0, 2.0];
        let v = fm_target_velocity(&x_0, &x_1, 0.0, &cfg).unwrap();
        assert!((v[0] - PI / 2.0).abs() < 1e-5);
        assert!((v[1] - PI).abs() < 1e-5);
    }

    #[test]
    fn test_target_velocity_cosine_at_t1() {
        // At t=1: v = -sin(π/2)*(π/2)*x_0 + cos(π/2)*(π/2)*x_1 = -(π/2)*x_0
        let cfg = FlowMatchingConfig {
            path: FlowPath::CosineAnnealing,
            ..Default::default()
        };
        let x_0 = vec![1.0, 2.0];
        let x_1 = vec![0.0, 0.0];
        let v = fm_target_velocity(&x_0, &x_1, 1.0, &cfg).unwrap();
        assert!((v[0] + PI / 2.0).abs() < 1e-5, "v[0]={}", v[0]);
        assert!((v[1] + PI).abs() < 1e-5, "v[1]={}", v[1]);
    }

    #[test]
    fn test_target_velocity_ve() {
        // VE velocity = sigma_max * x_1 (independent of x_0 and t)
        let sigma_max = 3.0;
        let cfg = FlowMatchingConfig {
            path: FlowPath::VarianceExploding { sigma_max },
            ..Default::default()
        };
        let x_0 = vec![10.0, -5.0];
        let x_1 = vec![1.0, 2.0];
        let v = fm_target_velocity(&x_0, &x_1, 0.5, &cfg).unwrap();
        assert!((v[0] - sigma_max * x_1[0]).abs() < 1e-6);
        assert!((v[1] - sigma_max * x_1[1]).abs() < 1e-6);
    }

    // ── fm_reconstruct_x0 ─────────────────────────────────────────────────────

    #[test]
    fn test_reconstruct_x0_linear_round_trip() {
        let cfg = FlowMatchingConfig::default();
        let x_0 = vec![1.0, -2.0, 3.0];
        let x_1 = vec![4.0, 5.0, -1.0];
        let t = 0.4;
        let x_t = fm_interpolate(&x_0, &x_1, t, &cfg).unwrap();
        let v_t = fm_target_velocity(&x_0, &x_1, t, &cfg).unwrap();
        let x_0_hat = fm_reconstruct_x0(&x_t, &v_t, t, &cfg).unwrap();
        for (a, b) in x_0_hat.iter().zip(x_0.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "linear round-trip failed: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_reconstruct_x0_cosine_round_trip() {
        let cfg = FlowMatchingConfig {
            path: FlowPath::CosineAnnealing,
            ..Default::default()
        };
        let x_0 = vec![2.0, -1.0];
        let x_1 = vec![0.5, 3.0];
        let t = 0.3;
        let x_t = fm_interpolate(&x_0, &x_1, t, &cfg).unwrap();
        let v_t = fm_target_velocity(&x_0, &x_1, t, &cfg).unwrap();
        let x_0_hat = fm_reconstruct_x0(&x_t, &v_t, t, &cfg).unwrap();
        for (a, b) in x_0_hat.iter().zip(x_0.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "cosine round-trip failed: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_reconstruct_x0_ve_round_trip() {
        let cfg = FlowMatchingConfig {
            path: FlowPath::VarianceExploding { sigma_max: 2.0 },
            ..Default::default()
        };
        let x_0 = vec![1.0, 2.0];
        let x_1 = vec![3.0, -1.0];
        let t = 0.5;
        let x_t = fm_interpolate(&x_0, &x_1, t, &cfg).unwrap();
        let v_t = fm_target_velocity(&x_0, &x_1, t, &cfg).unwrap();
        let x_0_hat = fm_reconstruct_x0(&x_t, &v_t, t, &cfg).unwrap();
        for (a, b) in x_0_hat.iter().zip(x_0.iter()) {
            assert!((a - b).abs() < 1e-5, "VE round-trip: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_reconstruct_x0_dim_mismatch() {
        let cfg = FlowMatchingConfig::default();
        let err = fm_reconstruct_x0(&[1.0, 2.0], &[1.0], 0.5, &cfg).unwrap_err();
        assert!(matches!(err, FlowMatchingError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_reconstruct_x0_invalid_time() {
        let cfg = FlowMatchingConfig::default();
        let err = fm_reconstruct_x0(&[1.0], &[0.0], 1.5, &cfg).unwrap_err();
        assert!(matches!(err, FlowMatchingError::InvalidTime(_)));
    }

    // ── fm_euler_step ─────────────────────────────────────────────────────────

    #[test]
    fn test_euler_step_basic() {
        let x_t = vec![0.0, 1.0];
        let v_t = vec![1.0, -1.0];
        let out = fm_euler_step(&x_t, &v_t, 0.5, 0.1).unwrap();
        assert!((out[0] - 0.1).abs() < 1e-6);
        assert!((out[1] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_euler_step_dt_zero_unchanged() {
        let x_t = vec![1.0, 2.0, 3.0];
        let v_t = vec![100.0, -200.0, 50.0];
        let out = fm_euler_step(&x_t, &v_t, 0.5, 0.0).unwrap();
        for (a, b) in out.iter().zip(x_t.iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_euler_step_dim_mismatch() {
        let err = fm_euler_step(&[1.0, 2.0], &[1.0], 0.5, 0.1).unwrap_err();
        assert!(matches!(err, FlowMatchingError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_euler_step_empty() {
        let err = fm_euler_step(&[], &[], 0.5, 0.1).unwrap_err();
        assert!(matches!(err, FlowMatchingError::EmptyInput));
    }

    // ── fm_heun_step ──────────────────────────────────────────────────────────

    #[test]
    fn test_heun_step_averages_velocities() {
        let x_t = vec![0.0];
        let v0 = vec![1.0];
        let v1 = vec![3.0];
        let out = fm_heun_step(&x_t, &v0, &v1, 1.0).unwrap();
        // x + dt * 0.5 * (v0 + v1) = 0 + 1 * 0.5 * 4 = 2
        assert!((out[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_heun_step_equal_velocities_same_as_euler() {
        let x_t = vec![1.0, 2.0];
        let v = vec![0.5, -0.5];
        let euler = fm_euler_step(&x_t, &v, 0.5, 0.1).unwrap();
        let heun = fm_heun_step(&x_t, &v, &v, 0.1).unwrap();
        for (a, b) in euler.iter().zip(heun.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn test_heun_step_dim_mismatch() {
        let err = fm_heun_step(&[1.0], &[1.0, 2.0], &[1.0], 0.1).unwrap_err();
        assert!(matches!(err, FlowMatchingError::DimensionMismatch { .. }));
    }

    // ── fm_inference_timesteps ────────────────────────────────────────────────

    #[test]
    fn test_inference_timesteps_length() {
        let n = 10;
        let ts = fm_inference_timesteps(n);
        assert_eq!(ts.len(), n + 1);
    }

    #[test]
    fn test_inference_timesteps_starts_at_one() {
        let ts = fm_inference_timesteps(20);
        assert!((ts[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_inference_timesteps_ends_at_zero() {
        let ts = fm_inference_timesteps(20);
        assert!((ts[20] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_inference_timesteps_monotone_decreasing() {
        let ts = fm_inference_timesteps(10);
        for w in ts.windows(2) {
            assert!(w[0] > w[1], "not monotone: {} <= {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_inference_timesteps_zero_steps_no_nan() {
        // Regression test: n_steps=0 previously computed 0.0/0.0 = NaN.
        let ts = fm_inference_timesteps(0);
        assert_eq!(ts.len(), 1, "n_steps+1 = 1 value expected");
        assert!(ts.iter().all(|v| v.is_finite()), "got NaN: {:?}", ts);
        assert!((ts[0] - 1.0).abs() < 1e-9);
    }

    // ── fm_loss ───────────────────────────────────────────────────────────────

    #[test]
    fn test_loss_identical_prediction_is_zero() {
        let v = vec![1.0, -2.0, 3.0];
        let loss = fm_loss(&v, &v, 0.5, &FlowLossWeight::Uniform).unwrap();
        assert!(loss.abs() < 1e-9, "identical prediction should give 0 loss");
    }

    #[test]
    fn test_loss_uniform_weight_one() {
        let pred = vec![0.0];
        let tgt = vec![1.0];
        let loss = fm_loss(&pred, &tgt, 0.5, &FlowLossWeight::Uniform).unwrap();
        // MSE = 1.0, weight = 1.0
        assert!((loss - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_loss_inverse_t_increases_near_zero() {
        let pred = vec![0.0];
        let tgt = vec![1.0];
        let loss_small_t = fm_loss(&pred, &tgt, 0.01, &FlowLossWeight::InverseT).unwrap();
        let loss_large_t = fm_loss(&pred, &tgt, 0.9, &FlowLossWeight::InverseT).unwrap();
        assert!(loss_small_t > loss_large_t);
    }

    #[test]
    fn test_loss_ttime_peaks_at_half() {
        let pred = vec![0.0];
        let tgt = vec![1.0];
        let loss_half = fm_loss(&pred, &tgt, 0.5, &FlowLossWeight::TimeTDependant).unwrap();
        let loss_near_zero = fm_loss(&pred, &tgt, 0.05, &FlowLossWeight::TimeTDependant).unwrap();
        let loss_near_one = fm_loss(&pred, &tgt, 0.95, &FlowLossWeight::TimeTDependant).unwrap();
        assert!(loss_half > loss_near_zero);
        assert!(loss_half > loss_near_one);
    }

    #[test]
    fn test_loss_invalid_time() {
        let v = vec![1.0];
        let err = fm_loss(&v, &v, -0.1, &FlowLossWeight::Uniform).unwrap_err();
        assert!(matches!(err, FlowMatchingError::InvalidTime(_)));
    }

    #[test]
    fn test_loss_dim_mismatch() {
        let err = fm_loss(&[1.0, 2.0], &[1.0], 0.5, &FlowLossWeight::Uniform).unwrap_err();
        assert!(matches!(err, FlowMatchingError::DimensionMismatch { .. }));
    }

    // ── fm_loss_weight ────────────────────────────────────────────────────────

    #[test]
    fn test_loss_weight_all_positive() {
        for t in [0.01, 0.25, 0.5, 0.75, 0.99] {
            let variants = vec![
                FlowLossWeight::Uniform,
                FlowLossWeight::InverseT,
                FlowLossWeight::TimeTDependant,
                FlowLossWeight::SnrLike { sigma_min: 0.001 },
            ];
            for w in &variants {
                let val = fm_loss_weight(t, w);
                assert!(val >= 0.0, "weight for {:?} at t={} is negative", w, t);
            }
        }
    }

    #[test]
    fn test_loss_weight_snr_like_decreases_with_t() {
        let w = FlowLossWeight::SnrLike { sigma_min: 0.01 };
        let w_small = fm_loss_weight(0.01, &w);
        let w_large = fm_loss_weight(0.9, &w);
        assert!(w_small > w_large);
    }

    // ── fm_stochastic_interpolant ─────────────────────────────────────────────

    #[test]
    fn test_stochastic_interpolant_gamma0_equals_linear() {
        let x_0 = vec![1.0, 2.0, 3.0];
        let x_1 = vec![4.0, 5.0, 6.0];
        let noise = vec![100.0, -100.0, 50.0]; // large noise, should be zeroed out
        let t = 0.4;
        let si = fm_stochastic_interpolant(&x_0, &x_1, t, &noise, 0.0).unwrap();
        // Without gamma noise, should equal linear interpolation
        let linear_alpha = 1.0 - t;
        let linear_beta = t;
        for i in 0..3 {
            let expected = linear_alpha * x_0[i] + linear_beta * x_1[i];
            assert!((si[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_stochastic_interpolant_gamma_positive_differs() {
        let x_0 = vec![0.0; 4];
        let x_1 = vec![0.0; 4];
        let noise = vec![1.0, -1.0, 2.0, -2.0];
        let t = 0.5;
        let si = fm_stochastic_interpolant(&x_0, &x_1, t, &noise, 1.0).unwrap();
        // With x_0 = x_1 = 0, output = gamma * sqrt(alpha*beta) * noise
        // alpha = beta = 0.5, gamma = 1, sqrt(0.25) = 0.5
        let scale = (0.5 * 0.5f32).sqrt();
        for (s, n) in si.iter().zip(noise.iter()) {
            let expected = scale * n;
            assert!((s - expected).abs() < 1e-6, "{} vs {}", s, expected);
        }
    }

    #[test]
    fn test_stochastic_interpolant_invalid_time() {
        let err = fm_stochastic_interpolant(&[1.0], &[0.0], 2.0, &[0.0], 0.0).unwrap_err();
        assert!(matches!(err, FlowMatchingError::InvalidTime(_)));
    }

    #[test]
    fn test_stochastic_interpolant_dim_mismatch_noise() {
        let err =
            fm_stochastic_interpolant(&[1.0, 2.0], &[3.0, 4.0], 0.5, &[0.0], 0.0).unwrap_err();
        assert!(matches!(err, FlowMatchingError::DimensionMismatch { .. }));
    }

    // ── fm_sample_noise ───────────────────────────────────────────────────────

    #[test]
    fn test_sample_noise_correct_length() {
        let mut state = test_state();
        for n in [1, 2, 3, 10, 100] {
            let noise = fm_sample_noise(n, &mut state);
            assert_eq!(noise.len(), n);
        }
    }

    #[test]
    fn test_sample_noise_roughly_normal() {
        let mut state = test_state();
        let noise = fm_sample_noise(10_000, &mut state);
        let mean = noise.iter().sum::<f32>() / noise.len() as f32;
        let var = noise.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / noise.len() as f32;
        let std = var.sqrt();
        assert!(mean.abs() < 0.1, "noise mean={} too far from 0", mean);
        assert!((std - 1.0).abs() < 0.1, "noise std={} too far from 1", std);
    }

    // ── FlowIntegrator ────────────────────────────────────────────────────────

    #[test]
    fn test_integrator_new() {
        let cfg = FlowMatchingConfig::default();
        let integrator = FlowIntegrator::new(cfg, 10, false);
        assert_eq!(integrator.n_steps, 10);
        assert!(!integrator.use_heun);
    }

    #[test]
    fn test_integrator_euler_constant_velocity() {
        // With constant velocity v = x_1 - x_0 and linear path,
        // Euler integration from x_1 back to x_0 should recover x_0 exactly
        // (for a constant field this is exact).
        let cfg = FlowMatchingConfig::default();
        let integrator = FlowIntegrator::new(cfg, 100, false);
        let x_0 = [1.0f32, -1.0];
        let x_1 = [3.0f32, 1.0];
        let vel: Vec<f32> = x_1.iter().zip(x_0.iter()).map(|(b, a)| b - a).collect();
        let result = integrator.integrate(&x_1, |_, _| Ok(vel.clone())).unwrap();
        // With 100 steps Euler is exact for linear (constant) velocity
        for (a, b) in result.iter().zip(x_0.iter()) {
            assert!((a - b).abs() < 1e-4, "euler: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_integrator_heun_constant_velocity() {
        let cfg = FlowMatchingConfig::default();
        let integrator = FlowIntegrator::new(cfg, 50, true);
        let x_0 = [0.0f32, 2.0];
        let x_1 = [1.0f32, 0.0];
        let vel: Vec<f32> = x_1.iter().zip(x_0.iter()).map(|(b, a)| b - a).collect();
        let result = integrator.integrate(&x_1, |_, _| Ok(vel.clone())).unwrap();
        for (a, b) in result.iter().zip(x_0.iter()) {
            assert!((a - b).abs() < 1e-4, "heun: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_integrator_empty_input() {
        let cfg = FlowMatchingConfig::default();
        let integrator = FlowIntegrator::new(cfg, 10, false);
        let err = integrator.integrate(&[], |_, _| Ok(vec![])).unwrap_err();
        assert!(matches!(err, FlowMatchingError::EmptyInput));
    }

    // ── compute_flow_stats ────────────────────────────────────────────────────

    #[test]
    fn test_flow_stats_zero_velocities() {
        let velocities = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let stats = compute_flow_stats(&velocities).unwrap();
        assert!(stats.mean_velocity_norm.abs() < 1e-9);
        assert!(stats.max_velocity_norm.abs() < 1e-9);
        assert!(stats.mean_squared_norm.abs() < 1e-9);
    }

    #[test]
    fn test_flow_stats_single_velocity() {
        let velocities = vec![vec![3.0, 4.0]]; // norm = 5.0
        let stats = compute_flow_stats(&velocities).unwrap();
        assert!((stats.mean_velocity_norm - 5.0).abs() < 1e-5);
        assert!((stats.max_velocity_norm - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_flow_stats_empty_batch() {
        let err = compute_flow_stats(&[]).unwrap_err();
        assert!(matches!(err, FlowMatchingError::EmptyInput));
    }

    #[test]
    fn test_flow_stats_smoothness_range() {
        let velocities = vec![vec![1.0], vec![1.0], vec![1.0]];
        let stats = compute_flow_stats(&velocities).unwrap();
        // All identical → std = 0 → smoothness = 1
        assert!((stats.smoothness - 1.0).abs() < 1e-4);
    }

    // ── fm_sample_pair ────────────────────────────────────────────────────────

    #[test]
    fn test_sample_pair_lengths() {
        let mut state = test_state();
        let (x0, x1) = fm_sample_pair(10, &mut state);
        assert_eq!(x0.len(), 10);
        assert_eq!(x1.len(), 10);
    }

    #[test]
    fn test_sample_pair_independent() {
        let mut state = test_state();
        let (x0, x1) = fm_sample_pair(100, &mut state);
        // They should not be identical
        let same = x0.iter().zip(x1.iter()).all(|(a, b)| (a - b).abs() < 1e-9);
        assert!(!same, "x0 and x1 should be independent");
    }

    // ── format functions ──────────────────────────────────────────────────────

    #[test]
    fn test_format_flow_config_non_empty() {
        let cfg = FlowMatchingConfig::default();
        let s = format_flow_config(&cfg);
        assert!(!s.is_empty());
        assert!(s.contains("Linear"));
    }

    #[test]
    fn test_format_flow_stats_non_empty() {
        let velocities = vec![vec![1.0, 0.0]];
        let stats = compute_flow_stats(&velocities).unwrap();
        let s = format_flow_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("FlowStats"));
    }

    #[test]
    fn test_format_flow_config_all_paths() {
        for path in [
            FlowPath::Linear,
            FlowPath::CosineAnnealing,
            FlowPath::VarianceExploding { sigma_max: 10.0 },
            FlowPath::VariancePreserving,
        ] {
            let cfg = FlowMatchingConfig {
                path,
                ..Default::default()
            };
            let s = format_flow_config(&cfg);
            assert!(!s.is_empty());
        }
    }

    #[test]
    fn test_format_flow_config_all_schedules() {
        for sched in [
            TimeSchedule::Uniform,
            TimeSchedule::LogitNormal {
                mu: 0.0,
                sigma: 1.0,
            },
            TimeSchedule::Beta {
                alpha: 2.0,
                beta: 2.0,
            },
            TimeSchedule::Cosine,
        ] {
            let cfg = FlowMatchingConfig {
                t_schedule: sched,
                ..Default::default()
            };
            let s = format_flow_config(&cfg);
            assert!(!s.is_empty());
        }
    }

    // ── edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_interpolate_empty_input() {
        let cfg = FlowMatchingConfig::default();
        let err = fm_interpolate(&[], &[], 0.5, &cfg).unwrap_err();
        assert!(matches!(err, FlowMatchingError::EmptyInput));
    }

    #[test]
    fn test_target_velocity_empty_input() {
        let cfg = FlowMatchingConfig::default();
        let err = fm_target_velocity(&[], &[], 0.5, &cfg).unwrap_err();
        assert!(matches!(err, FlowMatchingError::EmptyInput));
    }

    #[test]
    fn test_reconstruct_empty_input() {
        let cfg = FlowMatchingConfig::default();
        let err = fm_reconstruct_x0(&[], &[], 0.5, &cfg).unwrap_err();
        assert!(matches!(err, FlowMatchingError::EmptyInput));
    }

    #[test]
    fn test_vp_interpolant_unit_norm_preserved() {
        // VP path preserves norm when x_0 and x_1 are unit vectors: ||x_t|| = 1
        // because cos²(θ) + sin²(θ) = 1
        let x_0 = vec![1.0, 0.0];
        let x_1 = vec![0.0, 1.0];
        let cfg = FlowMatchingConfig {
            path: FlowPath::VariancePreserving,
            ..Default::default()
        };
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let x_t = fm_interpolate(&x_0, &x_1, t, &cfg).unwrap();
            let norm_sq: f32 = x_t.iter().map(|x| x * x).sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-5,
                "VP norm={} at t={}",
                norm_sq,
                t
            );
        }
    }

    #[test]
    fn test_loss_weight_ttime_at_endpoints_near_zero() {
        // t*(1-t) should be 0 at t=0 and t=1
        assert!(fm_loss_weight(0.0, &FlowLossWeight::TimeTDependant).abs() < 1e-6);
        assert!(fm_loss_weight(1.0, &FlowLossWeight::TimeTDependant).abs() < 1e-6);
    }

    #[test]
    fn test_fm_loss_snr_like_weight() {
        let w = FlowLossWeight::SnrLike { sigma_min: 0.001 };
        // At t = 0: weight ≈ 1 (sigma_min² / (0 + sigma_min²) = 1)
        let w0 = fm_loss_weight(0.0, &w);
        assert!((w0 - 1.0).abs() < 1e-3, "SNR weight at t=0: {}", w0);
    }
}
