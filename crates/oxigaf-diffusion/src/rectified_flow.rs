//! # Rectified Flow (Liu et al. 2022)
//!
//! Implements Rectified Flow ("Flow Straight and Fast") as described in
//! Liu et al. (2022). Rectified Flow trains trajectories to be straight by
//! learning constant-velocity paths:
//!
//! - Data: clean samples x1 ~ p_data, noise x0 ~ N(0,I)
//! - Linear interpolation: x_t = (1-t)*x0 + t*x1, t in \[0,1\]
//! - Target velocity: v* = x1 - x0 (constant, t-independent)
//! - Loss: L = E[||v_θ(x_t, t) - (x1 - x0)||²]
//!
//! ReFlow uses a trained model to generate (x0, x1) pairs, re-training on
//! these straighter pairs for progressively straighter trajectories.
//!
//! ## References
//!
//! - Liu et al. (2022), "Flow Straight and Fast: Learning to Generate and
//!   Transfer Data with Rectified Flow", ICLR 2023.

use std::f32::consts::PI;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise from rectified flow operations.
#[derive(Debug, Error, PartialEq)]
pub enum RectifiedFlowError {
    /// Batch is empty.
    #[error("Empty batch")]
    EmptyBatch,

    /// Input arrays have incompatible lengths.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Timestep is outside [0.0, 1.0].
    #[error("Invalid timestep: t={t}, must be in [0.0, 1.0]")]
    InvalidTimestep { t: f32 },

    /// A configuration parameter is invalid.
    #[error("Invalid config: {reason}")]
    InvalidConfig { reason: String },

    /// Integration failed at a specific step.
    #[error("Integration failed at step {step}: {reason}")]
    IntegrationFailed { step: usize, reason: String },
}

// ─────────────────────────────────────────────────────────────────────────────
// Private PRNG utilities
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
#[inline]
fn box_muller(u1: f32, u2: f32) -> (f32, f32) {
    let r = (-2.0_f32 * u1.max(1e-10).ln()).sqrt();
    let theta = 2.0 * PI * u2;
    (r * theta.cos(), r * theta.sin())
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Selects the ODE solver used during inference integration.
#[derive(Debug, Clone, PartialEq)]
pub enum RfSolverKind {
    /// First-order Euler method.
    Euler,
    /// Second-order midpoint method (RK2).
    Midpoint,
    /// Second-order Heun method (trapezoidal corrector).
    Heun,
    /// Fourth-order Runge-Kutta.
    Rk4,
}

/// Configuration for a Rectified Flow model.
#[derive(Debug, Clone)]
pub struct RectifiedFlowConfig {
    /// Number of ODE integration steps (e.g. 100).
    pub n_steps: usize,
    /// Minimum noise std at t=0 side for stability (e.g. 1e-5).
    pub sigma_min: f32,
    /// Minimum t for training (e.g. 0.001).
    pub t_min: f32,
    /// Maximum t for training (e.g. 0.999).
    pub t_max: f32,
    /// Which ODE solver to use during inference.
    pub solver: RfSolverKind,
    /// Steps for reflow pair generation (e.g. 10).
    pub reflow_n_steps: usize,
}

impl Default for RectifiedFlowConfig {
    fn default() -> Self {
        Self {
            n_steps: 100,
            sigma_min: 1e-5,
            t_min: 0.001,
            t_max: 0.999,
            solver: RfSolverKind::Euler,
            reflow_n_steps: 10,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core math: linear interpolation and target velocity
// ─────────────────────────────────────────────────────────────────────────────

/// Compute x_t = (1-t)*x0 + t*x1 (linear interpolation).
///
/// Returns `InvalidTimestep` if t is outside [0, 1], and `DimensionMismatch`
/// if x0 and x1 have different lengths.
pub fn rf_interpolate(x0: &[f32], x1: &[f32], t: f32) -> Result<Vec<f32>, RectifiedFlowError> {
    if !(0.0..=1.0).contains(&t) {
        return Err(RectifiedFlowError::InvalidTimestep { t });
    }
    if x0.len() != x1.len() {
        return Err(RectifiedFlowError::DimensionMismatch {
            expected: x0.len(),
            got: x1.len(),
        });
    }
    let one_minus_t = 1.0 - t;
    Ok(x0
        .iter()
        .zip(x1.iter())
        .map(|(&a, &b)| one_minus_t * a + t * b)
        .collect())
}

/// Compute the constant target velocity v* = x1 - x0 (straight-path velocity).
///
/// The velocity is t-independent for the rectified flow formulation.
pub fn rf_target_velocity(x0: &[f32], x1: &[f32]) -> Result<Vec<f32>, RectifiedFlowError> {
    if x0.len() != x1.len() {
        return Err(RectifiedFlowError::DimensionMismatch {
            expected: x0.len(),
            got: x1.len(),
        });
    }
    Ok(x0.iter().zip(x1.iter()).map(|(&a, &b)| b - a).collect())
}

/// Generate n evenly spaced points from `start` to `end` (inclusive).
pub fn rf_linspace(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n as f32 - 1.0);
    (0..n).map(|i| start + i as f32 * step).collect()
}
// ─────────────────────────────────────────────────────────────────────────────
// ODE step functions
// ─────────────────────────────────────────────────────────────────────────────

/// First-order Euler step: x_next = x + dt * v.
pub fn rf_euler_step(x: &[f32], v: &[f32], dt: f32) -> Vec<f32> {
    x.iter()
        .zip(v.iter())
        .map(|(&xi, &vi)| xi + dt * vi)
        .collect()
}

/// Second-order midpoint step: x_next = x + dt * v_mid.
///
/// `v_start` is the velocity at x, `v_mid` is the velocity at x + dt/2 * v_start.
pub fn rf_midpoint_step(x: &[f32], v_start: &[f32], v_mid: &[f32], dt: f32) -> Vec<f32> {
    // x_mid = x + (dt/2) * v_start  (computed externally and used for v_mid)
    // x_next = x + dt * v_mid
    let _ = v_start; // v_start was used to compute x_mid externally
    x.iter()
        .zip(v_mid.iter())
        .map(|(&xi, &vm)| xi + dt * vm)
        .collect()
}

/// Second-order Heun step (trapezoidal): x_next = x + dt/2 * (v_start + v_end).
pub fn rf_heun_step(x: &[f32], v_start: &[f32], v_end: &[f32], dt: f32) -> Vec<f32> {
    x.iter()
        .zip(v_start.iter())
        .zip(v_end.iter())
        .map(|((&xi, &vs), &ve)| xi + dt * 0.5 * (vs + ve))
        .collect()
}

/// Fourth-order Runge-Kutta step: x_next = x + dt/6 * (k1 + 2k2 + 2k3 + k4).
///
/// Like [`rf_heun_step`] / [`rf_midpoint_step`] / [`rf_euler_step`], mismatched
/// input lengths never panic here: `zip` truncates to the shortest of
/// `x`/`k1`/`k2`/`k3`/`k4` rather than indexing out of bounds. Callers that
/// need a hard error on a length mismatch (e.g. a caller-supplied velocity
/// function returning the wrong size) should check lengths themselves before
/// calling — [`RfOdeSolver::integrate`] does exactly this for every `k`.
pub fn rf_rk4_step(x: &[f32], k1: &[f32], k2: &[f32], k3: &[f32], k4: &[f32], dt: f32) -> Vec<f32> {
    let inv6 = dt / 6.0;
    x.iter()
        .zip(k1.iter())
        .zip(k2.iter())
        .zip(k3.iter())
        .zip(k4.iter())
        .map(|((((&xi, &k1i), &k2i), &k3i), &k4i)| xi + inv6 * (k1i + 2.0 * k2i + 2.0 * k3i + k4i))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss functions
// ─────────────────────────────────────────────────────────────────────────────

/// Flow-matching MSE loss: mean ||v_pred - (x1 - x0)||².
pub fn rf_flow_matching_loss(
    predicted_v: &[f32],
    x0: &[f32],
    x1: &[f32],
) -> Result<f32, RectifiedFlowError> {
    if predicted_v.is_empty() {
        return Err(RectifiedFlowError::EmptyBatch);
    }
    if predicted_v.len() != x0.len() {
        return Err(RectifiedFlowError::DimensionMismatch {
            expected: predicted_v.len(),
            got: x0.len(),
        });
    }
    if x0.len() != x1.len() {
        return Err(RectifiedFlowError::DimensionMismatch {
            expected: x0.len(),
            got: x1.len(),
        });
    }
    let sum_sq: f32 = predicted_v
        .iter()
        .zip(x0.iter())
        .zip(x1.iter())
        .map(|((&vp, &a), &b)| {
            let diff = vp - (b - a);
            diff * diff
        })
        .sum();
    Ok(sum_sq / predicted_v.len() as f32)
}

/// Per-sample weighted MSE loss on velocity.
///
/// `weights` must have length `n` (one weight per sample) where the full
/// arrays have length `n * d`.
pub fn rf_weighted_loss(
    predicted_v: &[f32],
    x0: &[f32],
    x1: &[f32],
    weights: &[f32],
) -> Result<f32, RectifiedFlowError> {
    if predicted_v.is_empty() {
        return Err(RectifiedFlowError::EmptyBatch);
    }
    if predicted_v.len() != x0.len() || x0.len() != x1.len() {
        return Err(RectifiedFlowError::DimensionMismatch {
            expected: predicted_v.len(),
            got: x0.len(),
        });
    }
    // weights length must divide the data length evenly
    let total = predicted_v.len();
    if weights.is_empty() {
        return Err(RectifiedFlowError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    let n = weights.len();
    if !total.is_multiple_of(n) {
        return Err(RectifiedFlowError::DimensionMismatch {
            expected: n,
            got: total,
        });
    }
    let d = total / n;
    let mut total_loss = 0.0f32;
    let mut weight_sum = 0.0f32;
    for (i, &w) in weights.iter().enumerate().take(n) {
        let offset = i * d;
        let mut sample_sq = 0.0f32;
        for j in 0..d {
            let vp = predicted_v[offset + j];
            let target = x1[offset + j] - x0[offset + j];
            let diff = vp - target;
            sample_sq += diff * diff;
        }
        total_loss += w * (sample_sq / d as f32);
        weight_sum += w;
    }
    if weight_sum == 0.0 {
        Ok(0.0)
    } else {
        Ok(total_loss / weight_sum)
    }
}

/// Per-sample L2 loss (batch × d layout → batch output).
///
/// Input arrays have length `n * d`; output has length `n`.
pub fn rf_loss_per_sample(
    predicted_v: &[f32],
    x0: &[f32],
    x1: &[f32],
    d: usize,
) -> Result<Vec<f32>, RectifiedFlowError> {
    if d == 0 {
        return Err(RectifiedFlowError::InvalidConfig {
            reason: "d must be > 0".to_string(),
        });
    }
    if predicted_v.is_empty() {
        return Err(RectifiedFlowError::EmptyBatch);
    }
    let total = predicted_v.len();
    if !total.is_multiple_of(d) {
        return Err(RectifiedFlowError::DimensionMismatch {
            expected: d,
            got: total,
        });
    }
    if x0.len() != total || x1.len() != total {
        return Err(RectifiedFlowError::DimensionMismatch {
            expected: total,
            got: x0.len(),
        });
    }
    let n = total / d;
    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * d;
        let sq: f32 = (0..d)
            .map(|j| {
                let diff = predicted_v[offset + j] - (x1[offset + j] - x0[offset + j]);
                diff * diff
            })
            .sum();
        results.push(sq / d as f32);
    }
    Ok(results)
}

// ─────────────────────────────────────────────────────────────────────────────
// Timestep sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Sample uniform timesteps t ∈ [t_min, t_max] for each batch item.
///
/// Uses xorshift64 PRNG seeded with `seed`.
pub fn rf_sample_t(seed: u64, batch_size: usize, t_min: f32, t_max: f32) -> Vec<f32> {
    let mut state = if seed == 0 { 1 } else { seed };
    let range = t_max - t_min;
    (0..batch_size)
        .map(|_| {
            let u = xorshift_f32(&mut state);
            t_min + u * range
        })
        .collect()
}

/// Sample logit-normal timesteps concentrated near 0.5.
///
/// Uses Box-Muller: sample z ~ N(mean, std²), then t = sigmoid(z),
/// clamped to [t_min, t_max].
pub fn rf_logit_normal_t(
    seed: u64,
    batch_size: usize,
    mean: f32,
    std: f32,
    t_min: f32,
    t_max: f32,
) -> Vec<f32> {
    let mut state = if seed == 0 { 1 } else { seed };
    let mut results = Vec::with_capacity(batch_size);
    let mut i = 0;
    while i < batch_size {
        let u1 = xorshift_f32(&mut state);
        let u2 = xorshift_f32(&mut state);
        let (z1, z2) = box_muller(u1, u2);

        let t1 = {
            let z = mean + std * z1;
            let sigmoid = 1.0 / (1.0 + (-z).exp());
            sigmoid.clamp(t_min, t_max)
        };
        results.push(t1);
        i += 1;

        if i < batch_size {
            let t2 = {
                let z = mean + std * z2;
                let sigmoid = 1.0 / (1.0 + (-z).exp());
                sigmoid.clamp(t_min, t_max)
            };
            results.push(t2);
            i += 1;
        }
    }
    results.truncate(batch_size);
    results
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch generation
// ─────────────────────────────────────────────────────────────────────────────

/// A training batch for rectified flow.
#[derive(Debug, Clone)]
pub struct RfBatch {
    /// Noise samples x0 ~ N(0,I), shape \[n × d\].
    pub x0: Vec<f32>,
    /// Data samples x1 ~ p_data, shape \[n × d\].
    pub x1: Vec<f32>,
    /// Interpolated samples x_t = (1-t)*x0 + t*x1, shape \[n × d\].
    pub x_t: Vec<f32>,
    /// Timestep per sample, shape \[n\].
    pub t: Vec<f32>,
    /// Target velocity = x1 - x0, shape \[n × d\].
    pub target_v: Vec<f32>,
    /// Number of samples in the batch.
    pub n: usize,
    /// Dimensionality of each sample.
    pub d: usize,
}

/// Generate a training batch with x0 ~ N(0,I), x_t interpolated, and target velocity.
///
/// `x1` has shape [n × d], `t_values` has length n.
pub fn rf_make_batch(
    x1: &[f32],
    n: usize,
    d: usize,
    t_values: &[f32],
    seed: u64,
) -> Result<RfBatch, RectifiedFlowError> {
    if n == 0 || d == 0 {
        return Err(RectifiedFlowError::EmptyBatch);
    }
    if x1.len() != n * d {
        return Err(RectifiedFlowError::DimensionMismatch {
            expected: n * d,
            got: x1.len(),
        });
    }
    if t_values.len() != n {
        return Err(RectifiedFlowError::DimensionMismatch {
            expected: n,
            got: t_values.len(),
        });
    }
    for (idx, &t) in t_values.iter().enumerate() {
        if !(0.0..=1.0).contains(&t) {
            return Err(RectifiedFlowError::InvalidTimestep { t });
        }
        let _ = idx;
    }

    // Generate x0 ~ N(0,I) using Box-Muller
    let total = n * d;
    let mut state = if seed == 0 { 1 } else { seed };
    let mut x0 = Vec::with_capacity(total);
    let mut k = 0;
    while k < total {
        let u1 = xorshift_f32(&mut state);
        let u2 = xorshift_f32(&mut state);
        let (z1, z2) = box_muller(u1, u2);
        x0.push(z1);
        k += 1;
        if k < total {
            x0.push(z2);
            k += 1;
        }
    }
    x0.truncate(total);

    // Compute x_t and target_v
    let mut x_t = Vec::with_capacity(total);
    let mut target_v = Vec::with_capacity(total);
    for (i, &t) in t_values.iter().enumerate().take(n) {
        let one_minus_t = 1.0 - t;
        for j in 0..d {
            let idx = i * d + j;
            let a = x0[idx];
            let b = x1[idx];
            x_t.push(one_minus_t * a + t * b);
            target_v.push(b - a);
        }
    }

    Ok(RfBatch {
        x0,
        x1: x1.to_vec(),
        x_t,
        t: t_values.to_vec(),
        target_v,
        n,
        d,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Trajectory and ODE integration
// ─────────────────────────────────────────────────────────────────────────────

/// A recorded trajectory of the ODE integration.
#[derive(Debug, Clone)]
pub struct RfTrajectory {
    /// States at each integration step, shape \[n_steps+1\]\[d\].
    pub states: Vec<Vec<f32>>,
    /// Timestep at each state, length n_steps+1.
    pub times: Vec<f32>,
    /// Number of ODE steps taken.
    pub n_steps: usize,
    /// Dimensionality of each state.
    pub d: usize,
}
/// Integrate a trajectory using pre-computed velocities at each step (Euler).
///
/// `velocities` has length n_steps, `times` has length n_steps+1.
pub fn rf_euler_integrate(
    x0: &[f32],
    velocities: &[Vec<f32>],
    times: &[f32],
) -> Result<RfTrajectory, RectifiedFlowError> {
    let n_steps = velocities.len();
    if times.len() != n_steps + 1 {
        return Err(RectifiedFlowError::DimensionMismatch {
            expected: n_steps + 1,
            got: times.len(),
        });
    }
    let d = x0.len();
    for (step, v) in velocities.iter().enumerate() {
        if v.len() != d {
            return Err(RectifiedFlowError::IntegrationFailed {
                step,
                reason: format!("velocity dim {} != state dim {}", v.len(), d),
            });
        }
    }

    let mut states = Vec::with_capacity(n_steps + 1);
    states.push(x0.to_vec());

    for step in 0..n_steps {
        let dt = times[step + 1] - times[step];
        let current = &states[step];
        let v = &velocities[step];
        let next = rf_euler_step(current, v, dt);
        states.push(next);
    }

    Ok(RfTrajectory {
        states,
        times: times.to_vec(),
        n_steps,
        d,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// ODE Solver struct
// ─────────────────────────────────────────────────────────────────────────────

/// ODE solver for Rectified Flow inference.
pub struct RfOdeSolver {
    /// Configuration used for integration.
    pub config: RectifiedFlowConfig,
}

impl RfOdeSolver {
    /// Create a new solver, validating the config.
    pub fn new(config: RectifiedFlowConfig) -> Result<Self, RectifiedFlowError> {
        if config.n_steps == 0 {
            return Err(RectifiedFlowError::InvalidConfig {
                reason: "n_steps must be > 0".to_string(),
            });
        }
        if config.t_min < 0.0 || config.t_min >= config.t_max {
            return Err(RectifiedFlowError::InvalidConfig {
                reason: format!(
                    "t_min ({}) must be in [0, t_max) where t_max={}",
                    config.t_min, config.t_max
                ),
            });
        }
        if config.t_max > 1.0 {
            return Err(RectifiedFlowError::InvalidConfig {
                reason: format!("t_max ({}) must be <= 1.0", config.t_max),
            });
        }
        if config.sigma_min < 0.0 {
            return Err(RectifiedFlowError::InvalidConfig {
                reason: "sigma_min must be >= 0".to_string(),
            });
        }
        if config.reflow_n_steps == 0 {
            return Err(RectifiedFlowError::InvalidConfig {
                reason: "reflow_n_steps must be > 0".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Generate the timestep schedule: linspace(0, 1, n_steps+1).
    pub fn generate_t_schedule(&self) -> Vec<f32> {
        rf_linspace(0.0, 1.0, self.config.n_steps + 1)
    }

    /// Integrate the ODE from x0 using the given velocity function.
    ///
    /// `t_schedule` has length n_steps+1, the velocity_fn maps (x_t, t) → velocity.
    pub fn integrate<F>(
        &self,
        x0: &[f32],
        t_schedule: &[f32],
        velocity_fn: F,
    ) -> Result<RfTrajectory, RectifiedFlowError>
    where
        F: Fn(&[f32], f32) -> Vec<f32>,
    {
        let n_steps = t_schedule.len().saturating_sub(1);
        if n_steps == 0 {
            return Err(RectifiedFlowError::InvalidConfig {
                reason: "t_schedule must have at least 2 entries".to_string(),
            });
        }
        let d = x0.len();
        let mut states = Vec::with_capacity(n_steps + 1);
        states.push(x0.to_vec());

        for step in 0..n_steps {
            let t_cur = t_schedule[step];
            let t_next = t_schedule[step + 1];
            let dt = t_next - t_cur;
            let x_cur = states[step].clone();

            let next = match self.config.solver {
                RfSolverKind::Euler => {
                    let v = velocity_fn(&x_cur, t_cur);
                    if v.len() != d {
                        return Err(RectifiedFlowError::IntegrationFailed {
                            step,
                            reason: format!("velocity dim {} != state dim {}", v.len(), d),
                        });
                    }
                    rf_euler_step(&x_cur, &v, dt)
                }
                RfSolverKind::Midpoint => {
                    let v_start = velocity_fn(&x_cur, t_cur);
                    if v_start.len() != d {
                        return Err(RectifiedFlowError::IntegrationFailed {
                            step,
                            reason: format!("velocity dim {} != state dim {}", v_start.len(), d),
                        });
                    }
                    // x_mid = x + (dt/2) * v_start
                    let t_mid = t_cur + dt * 0.5;
                    let x_mid: Vec<f32> = x_cur
                        .iter()
                        .zip(v_start.iter())
                        .map(|(&xi, &vi)| xi + 0.5 * dt * vi)
                        .collect();
                    let v_mid = velocity_fn(&x_mid, t_mid);
                    if v_mid.len() != d {
                        return Err(RectifiedFlowError::IntegrationFailed {
                            step,
                            reason: format!(
                                "midpoint velocity dim {} != state dim {}",
                                v_mid.len(),
                                d
                            ),
                        });
                    }
                    rf_midpoint_step(&x_cur, &v_start, &v_mid, dt)
                }
                RfSolverKind::Heun => {
                    let v_start = velocity_fn(&x_cur, t_cur);
                    if v_start.len() != d {
                        return Err(RectifiedFlowError::IntegrationFailed {
                            step,
                            reason: format!("velocity dim {} != state dim {}", v_start.len(), d),
                        });
                    }
                    let x_euler: Vec<f32> = x_cur
                        .iter()
                        .zip(v_start.iter())
                        .map(|(&xi, &vi)| xi + dt * vi)
                        .collect();
                    let v_end = velocity_fn(&x_euler, t_next);
                    if v_end.len() != d {
                        return Err(RectifiedFlowError::IntegrationFailed {
                            step,
                            reason: format!(
                                "heun end velocity dim {} != state dim {}",
                                v_end.len(),
                                d
                            ),
                        });
                    }
                    rf_heun_step(&x_cur, &v_start, &v_end, dt)
                }
                RfSolverKind::Rk4 => {
                    let k1 = velocity_fn(&x_cur, t_cur);
                    if k1.len() != d {
                        return Err(RectifiedFlowError::IntegrationFailed {
                            step,
                            reason: format!("k1 dim {} != state dim {}", k1.len(), d),
                        });
                    }
                    let x2: Vec<f32> = x_cur
                        .iter()
                        .zip(k1.iter())
                        .map(|(&xi, &ki)| xi + 0.5 * dt * ki)
                        .collect();
                    let k2 = velocity_fn(&x2, t_cur + 0.5 * dt);
                    if k2.len() != d {
                        return Err(RectifiedFlowError::IntegrationFailed {
                            step,
                            reason: format!("k2 dim {} != state dim {}", k2.len(), d),
                        });
                    }
                    let x3: Vec<f32> = x_cur
                        .iter()
                        .zip(k2.iter())
                        .map(|(&xi, &ki)| xi + 0.5 * dt * ki)
                        .collect();
                    let k3 = velocity_fn(&x3, t_cur + 0.5 * dt);
                    if k3.len() != d {
                        return Err(RectifiedFlowError::IntegrationFailed {
                            step,
                            reason: format!("k3 dim {} != state dim {}", k3.len(), d),
                        });
                    }
                    let x4: Vec<f32> = x_cur
                        .iter()
                        .zip(k3.iter())
                        .map(|(&xi, &ki)| xi + dt * ki)
                        .collect();
                    let k4 = velocity_fn(&x4, t_next);
                    if k4.len() != d {
                        return Err(RectifiedFlowError::IntegrationFailed {
                            step,
                            reason: format!("k4 dim {} != state dim {}", k4.len(), d),
                        });
                    }
                    rf_rk4_step(&x_cur, &k1, &k2, &k3, &k4, dt)
                }
            };
            states.push(next);
        }

        Ok(RfTrajectory {
            states,
            times: t_schedule.to_vec(),
            n_steps,
            d,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Trajectory analysis (ReFlow utilities)
// ─────────────────────────────────────────────────────────────────────────────

/// Create a (x0, x1_approx) reflow pair from an integrated trajectory.
///
/// Returns `(x0, x_final)` — x0 is the trajectory starting state and
/// x_final is the last state, forming a new (noise, data) pair for re-training.
pub fn rf_reflow_pair(x0: &[f32], trajectory: &RfTrajectory) -> (Vec<f32>, Vec<f32>) {
    let x_final = trajectory
        .states
        .last()
        .cloned()
        .unwrap_or_else(|| x0.to_vec());
    (x0.to_vec(), x_final)
}

/// Measure trajectory straightness via total angle change between consecutive velocities.
///
/// For a perfectly straight trajectory all angles are 0, so the result is 0.0.
/// Skips angle computation when consecutive velocity vectors have zero magnitude.
pub fn rf_trajectory_curvature(trajectory: &RfTrajectory) -> f32 {
    if trajectory.states.len() < 3 {
        return 0.0;
    }
    let n = trajectory.states.len();
    let mut total_angle = 0.0f32;

    for i in 0..n - 2 {
        // velocity segment i → i+1
        let v1: Vec<f32> = trajectory.states[i + 1]
            .iter()
            .zip(trajectory.states[i].iter())
            .map(|(&b, &a)| b - a)
            .collect();
        // velocity segment i+1 → i+2
        let v2: Vec<f32> = trajectory.states[i + 2]
            .iter()
            .zip(trajectory.states[i + 1].iter())
            .map(|(&b, &a)| b - a)
            .collect();

        let dot: f32 = v1.iter().zip(v2.iter()).map(|(&a, &b)| a * b).sum();
        let norm1: f32 = v1.iter().map(|&a| a * a).sum::<f32>().sqrt();
        let norm2: f32 = v2.iter().map(|&a| a * a).sum::<f32>().sqrt();

        if norm1 > 1e-12 && norm2 > 1e-12 {
            let cos_angle = (dot / (norm1 * norm2)).clamp(-1.0, 1.0);
            total_angle += cos_angle.acos();
        }
    }
    total_angle
}

/// Compute total path length of the trajectory (sum of segment lengths).
pub fn rf_trajectory_length(trajectory: &RfTrajectory) -> f32 {
    if trajectory.states.len() < 2 {
        return 0.0;
    }
    trajectory
        .states
        .windows(2)
        .map(|w| {
            let seg_sq: f32 = w[0]
                .iter()
                .zip(w[1].iter())
                .map(|(&a, &b)| (b - a) * (b - a))
                .sum();
            seg_sq.sqrt()
        })
        .sum()
}

/// Compute straight-line (chord) length from start to end of trajectory.
pub fn rf_straight_path_length(trajectory: &RfTrajectory) -> f32 {
    if trajectory.states.len() < 2 {
        return 0.0;
    }
    let first = &trajectory.states[0];
    let last = match trajectory.states.last() {
        Some(l) => l,
        None => return 0.0,
    };
    let sq: f32 = first
        .iter()
        .zip(last.iter())
        .map(|(&a, &b)| (b - a) * (b - a))
        .sum();
    sq.sqrt()
}

/// Ratio of straight-line length to total path length (1.0 = perfectly straight).
///
/// Returns 1.0 if the trajectory has zero total length (degenerate case).
pub fn rf_straightness_ratio(trajectory: &RfTrajectory) -> f32 {
    let total = rf_trajectory_length(trajectory);
    if total < 1e-12 {
        return 1.0;
    }
    let straight = rf_straight_path_length(trajectory);
    (straight / total).min(1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Coupling / transport plan
// ─────────────────────────────────────────────────────────────────────────────

/// Mode of coupling between noise and data.
#[derive(Debug, Clone, PartialEq)]
pub enum CouplingMode {
    /// Sample x0 ~ N(0,I) independently of x1 (standard RF).
    Independent,
    /// Greedy mini-batch optimal transport coupling.
    MiniBatchOt,
}

/// Coupling strategy between x0 (noise) and x1 (data).
#[derive(Debug, Clone)]
pub struct RfCoupling {
    /// Which coupling mode to use.
    pub mode: CouplingMode,
}

impl RfCoupling {
    /// Create a new coupling with the given mode.
    pub fn new(mode: CouplingMode) -> Self {
        Self { mode }
    }

    /// Sample n noise vectors of dimension d.
    ///
    /// For `Independent`, samples x0 ~ N(0,I) ignoring x1 (so `x1`'s length
    /// is never validated in this mode). For `MiniBatchOt`, also samples
    /// Gaussian but matches pairs via greedy OT, which requires `x1.len()
    /// == n * d`.
    ///
    /// # Errors
    ///
    /// In `MiniBatchOt` mode, propagates [`rf_greedy_ot_match`]'s errors
    /// (`n == 0 || d == 0`, or `x1.len() != n * d`).
    pub fn sample_x0(
        &self,
        x1: &[f32],
        n: usize,
        d: usize,
        seed: u64,
    ) -> Result<Vec<f32>, RectifiedFlowError> {
        let total = n * d;
        let mut state = if seed == 0 { 1 } else { seed };
        let mut x0 = Vec::with_capacity(total);
        let mut k = 0;
        while k < total {
            let u1 = xorshift_f32(&mut state);
            let u2 = xorshift_f32(&mut state);
            let (z1, z2) = box_muller(u1, u2);
            x0.push(z1);
            k += 1;
            if k < total {
                x0.push(z2);
                k += 1;
            }
        }
        x0.truncate(total);

        match self.mode {
            CouplingMode::Independent => Ok(x0),
            CouplingMode::MiniBatchOt => {
                // Greedy OT: rearrange x0 rows to match x1 rows
                let assignment = rf_greedy_ot_match(&x0, x1, n, d)?;
                let mut matched = vec![0.0f32; total];
                for (i, &j) in assignment.iter().enumerate() {
                    let src = j * d;
                    let dst = i * d;
                    matched[dst..dst + d].copy_from_slice(&x0[src..src + d]);
                }
                Ok(matched)
            }
        }
    }

    /// Compute greedy mini-batch OT matching: for each x1\[i\], find the nearest x0\[j\].
    ///
    /// Returns assignment\[i\] = j (the x0 row index assigned to x1 row i).
    ///
    /// # Errors
    ///
    /// Propagates [`rf_greedy_ot_match`]'s errors.
    pub fn match_pairs(
        &self,
        x0: &[f32],
        x1: &[f32],
        n: usize,
        d: usize,
    ) -> Result<Vec<usize>, RectifiedFlowError> {
        rf_greedy_ot_match(x0, x1, n, d)
    }
}

/// Greedy mini-batch OT: for each x1\[i\], find the nearest available x0\[j\].
///
/// Greedy assignment with no reassignment (each x0 can only be used once).
/// Returns assignment\[i\] = j where j is the x0 row assigned to x1 row i.
/// If all x0 are taken, falls back to the globally nearest (without availability check).
///
/// # Errors
///
/// Returns [`RectifiedFlowError::InvalidConfig`] if `n == 0 || d == 0`, or
/// [`RectifiedFlowError::DimensionMismatch`] if `x0.len() != n * d` or
/// `x1.len() != n * d`.
pub fn rf_greedy_ot_match(
    x0: &[f32],
    x1: &[f32],
    n: usize,
    d: usize,
) -> Result<Vec<usize>, RectifiedFlowError> {
    if n == 0 || d == 0 {
        return Err(RectifiedFlowError::InvalidConfig {
            reason: format!("rf_greedy_ot_match: n and d must both be > 0, got n={n}, d={d}"),
        });
    }
    if x0.len() != n * d {
        return Err(RectifiedFlowError::DimensionMismatch {
            expected: n * d,
            got: x0.len(),
        });
    }
    if x1.len() != n * d {
        return Err(RectifiedFlowError::DimensionMismatch {
            expected: n * d,
            got: x1.len(),
        });
    }
    let mut used = vec![false; n];
    let mut assignment = vec![0usize; n];

    for i in 0..n {
        let x1_row = &x1[i * d..(i * d + d)];
        let mut best_j = n; // sentinel for "no available slot found"
        let mut best_dist = f32::MAX;

        // Find nearest unused x0
        for j in 0..n {
            if used[j] {
                continue;
            }
            let x0_row = &x0[j * d..(j * d + d)];
            let dist: f32 = x1_row
                .iter()
                .zip(x0_row.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best_j = j;
            }
        }

        if best_j < n {
            used[best_j] = true;
            assignment[i] = best_j;
        } else {
            // All used: fall back to globally nearest (ignore availability)
            let mut fallback_best = 0;
            let mut fallback_dist = f32::MAX;
            for j in 0..n {
                let x0_row = &x0[j * d..(j * d + d)];
                let dist: f32 = x1_row
                    .iter()
                    .zip(x0_row.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                if dist < fallback_dist {
                    fallback_dist = dist;
                    fallback_best = j;
                }
            }
            assignment[i] = fallback_best;
        }
    }
    Ok(assignment)
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregated statistics over a training run or evaluation.
#[derive(Debug, Clone)]
pub struct RfStats {
    /// Mean loss over batches.
    pub mean_loss: f32,
    /// Minimum loss observed.
    pub min_loss: f32,
    /// Maximum loss observed.
    pub max_loss: f32,
    /// Mean trajectory curvature.
    pub mean_curvature: f32,
    /// Mean straightness ratio.
    pub mean_straightness: f32,
    /// Number of batches aggregated.
    pub n_batches: usize,
}

/// Compute aggregate statistics from per-batch arrays.
pub fn rf_compute_stats(losses: &[f32], curvatures: &[f32], straightnesses: &[f32]) -> RfStats {
    let n_batches = losses.len();
    if n_batches == 0 {
        return RfStats {
            mean_loss: 0.0,
            min_loss: 0.0,
            max_loss: 0.0,
            mean_curvature: 0.0,
            mean_straightness: 0.0,
            n_batches: 0,
        };
    }
    let mean_loss = losses.iter().sum::<f32>() / n_batches as f32;
    let min_loss = losses.iter().cloned().fold(f32::MAX, f32::min);
    let max_loss = losses.iter().cloned().fold(f32::MIN, f32::max);
    let mean_curvature = if curvatures.is_empty() {
        0.0
    } else {
        curvatures.iter().sum::<f32>() / curvatures.len() as f32
    };
    let mean_straightness = if straightnesses.is_empty() {
        0.0
    } else {
        straightnesses.iter().sum::<f32>() / straightnesses.len() as f32
    };
    RfStats {
        mean_loss,
        min_loss,
        max_loss,
        mean_curvature,
        mean_straightness,
        n_batches,
    }
}

/// Format statistics as a human-readable string.
pub fn rf_format_stats(stats: &RfStats) -> String {
    format!(
        "RfStats {{ n_batches={}, loss=[{:.6}/{:.6}/{:.6}] (min/mean/max), \
         curvature={:.6}, straightness={:.6} }}",
        stats.n_batches,
        stats.min_loss,
        stats.mean_loss,
        stats.max_loss,
        stats.mean_curvature,
        stats.mean_straightness,
    )
}

/// Format a config as a human-readable string.
pub fn rf_format_config(config: &RectifiedFlowConfig) -> String {
    format!(
        "RectifiedFlowConfig {{ n_steps={}, sigma_min={}, t_min={}, t_max={}, \
         solver={:?}, reflow_n_steps={} }}",
        config.n_steps,
        config.sigma_min,
        config.t_min,
        config.t_max,
        config.solver,
        config.reflow_n_steps,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "rectified_flow_tests.rs"]
mod tests;
