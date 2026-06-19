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
pub fn rf_rk4_step(x: &[f32], k1: &[f32], k2: &[f32], k3: &[f32], k4: &[f32], dt: f32) -> Vec<f32> {
    let inv6 = dt / 6.0;
    x.iter()
        .enumerate()
        .map(|(i, &xi)| xi + inv6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
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
                    let x3: Vec<f32> = x_cur
                        .iter()
                        .zip(k2.iter())
                        .map(|(&xi, &ki)| xi + 0.5 * dt * ki)
                        .collect();
                    let k3 = velocity_fn(&x3, t_cur + 0.5 * dt);
                    let x4: Vec<f32> = x_cur
                        .iter()
                        .zip(k3.iter())
                        .map(|(&xi, &ki)| xi + dt * ki)
                        .collect();
                    let k4 = velocity_fn(&x4, t_next);
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
    /// For `Independent`, samples x0 ~ N(0,I) ignoring x1.
    /// For `MiniBatchOt`, also samples Gaussian but matches pairs via greedy OT.
    pub fn sample_x0(&self, x1: &[f32], n: usize, d: usize, seed: u64) -> Vec<f32> {
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
            CouplingMode::Independent => x0,
            CouplingMode::MiniBatchOt => {
                // Greedy OT: rearrange x0 rows to match x1 rows
                let assignment = rf_greedy_ot_match(&x0, x1, n, d);
                let mut matched = vec![0.0f32; total];
                for (i, &j) in assignment.iter().enumerate() {
                    let src = j * d;
                    let dst = i * d;
                    matched[dst..dst + d].copy_from_slice(&x0[src..src + d]);
                }
                matched
            }
        }
    }

    /// Compute greedy mini-batch OT matching: for each x1\[i\], find the nearest x0\[j\].
    ///
    /// Returns assignment\[i\] = j (the x0 row index assigned to x1 row i).
    pub fn match_pairs(&self, x0: &[f32], x1: &[f32], n: usize, d: usize) -> Vec<usize> {
        rf_greedy_ot_match(x0, x1, n, d)
    }
}

/// Greedy mini-batch OT: for each x1\[i\], find the nearest available x0\[j\].
///
/// Greedy assignment with no reassignment (each x0 can only be used once).
/// Returns assignment\[i\] = j where j is the x0 row assigned to x1 row i.
/// If all x0 are taken, falls back to the globally nearest (without availability check).
pub fn rf_greedy_ot_match(x0: &[f32], x1: &[f32], n: usize, d: usize) -> Vec<usize> {
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
    assignment
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
mod tests {
    use super::*;

    // ── rf_interpolate ────────────────────────────────────────────────────────

    #[test]
    fn test_interpolate_t0_gives_x0() {
        let x0 = vec![1.0, 2.0, 3.0];
        let x1 = vec![4.0, 5.0, 6.0];
        let result = rf_interpolate(&x0, &x1, 0.0).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_t1_gives_x1() {
        let x0 = vec![1.0, 2.0, 3.0];
        let x1 = vec![4.0, 5.0, 6.0];
        let result = rf_interpolate(&x0, &x1, 1.0).unwrap();
        assert!((result[0] - 4.0).abs() < 1e-6);
        assert!((result[1] - 5.0).abs() < 1e-6);
        assert!((result[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_t_half_gives_midpoint() {
        let x0 = vec![0.0, 0.0];
        let x1 = vec![2.0, 4.0];
        let result = rf_interpolate(&x0, &x1, 0.5).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_dimension_mismatch_returns_error() {
        let x0 = vec![1.0, 2.0];
        let x1 = vec![3.0, 4.0, 5.0];
        assert!(matches!(
            rf_interpolate(&x0, &x1, 0.5),
            Err(RectifiedFlowError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_interpolate_invalid_t_negative() {
        let x0 = vec![0.0];
        let x1 = vec![1.0];
        assert!(matches!(
            rf_interpolate(&x0, &x1, -0.1),
            Err(RectifiedFlowError::InvalidTimestep { .. })
        ));
    }

    #[test]
    fn test_interpolate_invalid_t_above_one() {
        let x0 = vec![0.0];
        let x1 = vec![1.0];
        assert!(matches!(
            rf_interpolate(&x0, &x1, 1.1),
            Err(RectifiedFlowError::InvalidTimestep { .. })
        ));
    }

    // ── rf_target_velocity ────────────────────────────────────────────────────

    #[test]
    fn test_target_velocity_correctness() {
        let x0 = vec![1.0, 2.0, 3.0];
        let x1 = vec![4.0, 6.0, 9.0];
        let v = rf_target_velocity(&x0, &x1).unwrap();
        assert!((v[0] - 3.0).abs() < 1e-6);
        assert!((v[1] - 4.0).abs() < 1e-6);
        assert!((v[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_target_velocity_zero_when_equal() {
        let x = vec![1.0, 2.0, 3.0];
        let v = rf_target_velocity(&x, &x).unwrap();
        for vi in &v {
            assert!(vi.abs() < 1e-6);
        }
    }

    #[test]
    fn test_target_velocity_dimension_mismatch() {
        let x0 = vec![1.0, 2.0];
        let x1 = vec![1.0];
        assert!(matches!(
            rf_target_velocity(&x0, &x1),
            Err(RectifiedFlowError::DimensionMismatch { .. })
        ));
    }

    // ── rf_flow_matching_loss ─────────────────────────────────────────────────

    #[test]
    fn test_loss_zero_when_perfect() {
        let x0 = vec![0.0, 0.0];
        let x1 = vec![1.0, 1.0];
        let v_pred = vec![1.0, 1.0]; // exact target
        let loss = rf_flow_matching_loss(&v_pred, &x0, &x1).unwrap();
        assert!(loss.abs() < 1e-6);
    }

    #[test]
    fn test_loss_positive_when_not_matching() {
        let x0 = vec![0.0, 0.0];
        let x1 = vec![1.0, 1.0];
        let v_pred = vec![0.0, 0.0]; // off by 1,1
        let loss = rf_flow_matching_loss(&v_pred, &x0, &x1).unwrap();
        assert!(loss > 0.0);
    }

    #[test]
    fn test_loss_empty_returns_error() {
        assert!(matches!(
            rf_flow_matching_loss(&[], &[], &[]),
            Err(RectifiedFlowError::EmptyBatch)
        ));
    }

    // ── rf_loss_per_sample ────────────────────────────────────────────────────

    #[test]
    fn test_loss_per_sample_length_equals_n() {
        let n = 4;
        let d = 3;
        let x0 = vec![0.0f32; n * d];
        let x1 = vec![1.0f32; n * d];
        let v_pred = vec![1.0f32; n * d];
        let losses = rf_loss_per_sample(&v_pred, &x0, &x1, d).unwrap();
        assert_eq!(losses.len(), n);
    }

    #[test]
    fn test_loss_per_sample_zero_when_perfect() {
        let n = 2;
        let d = 2;
        let x0 = vec![0.0f32; n * d];
        let x1 = vec![2.0f32; n * d];
        let v_pred = vec![2.0f32; n * d]; // exact target
        let losses = rf_loss_per_sample(&v_pred, &x0, &x1, d).unwrap();
        for l in losses {
            assert!(l.abs() < 1e-6);
        }
    }

    // ── rf_weighted_loss ──────────────────────────────────────────────────────

    #[test]
    fn test_weighted_loss_scales_with_weights() {
        let x0 = vec![0.0, 0.0];
        let x1 = vec![1.0, 1.0];
        let v_pred = vec![0.0, 0.0]; // off by 1
        let w1 = vec![1.0];
        let w2 = vec![2.0];
        let l1 = rf_weighted_loss(&v_pred, &x0, &x1, &w1).unwrap();
        let l2 = rf_weighted_loss(&v_pred, &x0, &x1, &w2).unwrap();
        // Both should give the same normalized loss (weight / weight_sum = 1)
        assert!((l1 - l2).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_loss_dimension_mismatch_wrong_weights() {
        let x0 = vec![0.0, 0.0, 0.0, 0.0]; // 2 samples of d=2
        let x1 = vec![1.0, 1.0, 1.0, 1.0];
        let v_pred = vec![0.0, 0.0, 0.0, 0.0];
        let weights = vec![1.0, 1.0, 1.0]; // wrong: 3 instead of 2
        assert!(matches!(
            rf_weighted_loss(&v_pred, &x0, &x1, &weights),
            Err(RectifiedFlowError::DimensionMismatch { .. })
        ));
    }

    // ── rf_linspace ───────────────────────────────────────────────────────────

    #[test]
    fn test_linspace_first_and_last() {
        let v = rf_linspace(0.0, 1.0, 11);
        assert_eq!(v.len(), 11);
        assert!((v[0] - 0.0).abs() < 1e-6);
        assert!((v[10] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_linspace_evenly_spaced() {
        let v = rf_linspace(0.0, 1.0, 5);
        let step = 0.25;
        for (i, &val) in v.iter().enumerate().take(5) {
            assert!((val - i as f32 * step).abs() < 1e-5);
        }
    }

    #[test]
    fn test_linspace_n1_gives_start() {
        let v = rf_linspace(3.0, 7.0, 1);
        assert_eq!(v.len(), 1);
        assert!((v[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_linspace_n2_gives_endpoints() {
        let v = rf_linspace(2.0, 5.0, 2);
        assert_eq!(v.len(), 2);
        assert!((v[0] - 2.0).abs() < 1e-6);
        assert!((v[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_linspace_n0_empty() {
        let v = rf_linspace(0.0, 1.0, 0);
        assert!(v.is_empty());
    }

    // ── rf_sample_t ───────────────────────────────────────────────────────────

    #[test]
    fn test_sample_t_in_range() {
        let ts = rf_sample_t(42, 100, 0.001, 0.999);
        for t in &ts {
            assert!(*t >= 0.001 && *t <= 0.999, "t={} out of range", t);
        }
    }

    #[test]
    fn test_sample_t_correct_length() {
        let ts = rf_sample_t(1, 32, 0.0, 1.0);
        assert_eq!(ts.len(), 32);
    }

    #[test]
    fn test_sample_t_deterministic() {
        let ts1 = rf_sample_t(123, 10, 0.0, 1.0);
        let ts2 = rf_sample_t(123, 10, 0.0, 1.0);
        assert_eq!(ts1, ts2);
    }

    #[test]
    fn test_sample_t_different_seeds_differ() {
        let ts1 = rf_sample_t(42, 10, 0.0, 1.0);
        let ts2 = rf_sample_t(43, 10, 0.0, 1.0);
        assert_ne!(ts1, ts2);
    }

    // ── rf_logit_normal_t ─────────────────────────────────────────────────────

    #[test]
    fn test_logit_normal_t_in_range() {
        let ts = rf_logit_normal_t(7, 200, 0.0, 1.0, 0.001, 0.999);
        for t in &ts {
            assert!(*t >= 0.001 && *t <= 0.999, "t={} out of range", t);
        }
    }

    #[test]
    fn test_logit_normal_t_correct_length() {
        let ts = rf_logit_normal_t(99, 50, 0.0, 1.0, 0.0, 1.0);
        assert_eq!(ts.len(), 50);
    }

    #[test]
    fn test_logit_normal_t_deterministic() {
        let ts1 = rf_logit_normal_t(55, 20, 0.0, 1.0, 0.0, 1.0);
        let ts2 = rf_logit_normal_t(55, 20, 0.0, 1.0, 0.0, 1.0);
        assert_eq!(ts1, ts2);
    }

    #[test]
    fn test_logit_normal_t_bell_shaped_near_half() {
        // For mean=0, std=1: sigmoid(N(0,1)) should concentrate near 0.5
        let ts = rf_logit_normal_t(42, 1000, 0.0, 1.0, 0.0, 1.0);
        let mean: f32 = ts.iter().sum::<f32>() / ts.len() as f32;
        assert!((mean - 0.5).abs() < 0.05, "mean={} not near 0.5", mean);
    }

    // ── rf_make_batch ─────────────────────────────────────────────────────────

    #[test]
    fn test_make_batch_correct_shapes() {
        let n = 4;
        let d = 3;
        let x1 = vec![1.0f32; n * d];
        let ts = vec![0.5f32; n];
        let batch = rf_make_batch(&x1, n, d, &ts, 42).unwrap();
        assert_eq!(batch.x0.len(), n * d);
        assert_eq!(batch.x_t.len(), n * d);
        assert_eq!(batch.target_v.len(), n * d);
        assert_eq!(batch.t.len(), n);
        assert_eq!(batch.n, n);
        assert_eq!(batch.d, d);
    }

    #[test]
    fn test_make_batch_target_v_is_x1_minus_x0() {
        let n = 2;
        let d = 2;
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let ts = vec![0.3, 0.7];
        let batch = rf_make_batch(&x1, n, d, &ts, 1).unwrap();
        for i in 0..n * d {
            let expected = batch.x1[i] - batch.x0[i];
            assert!((batch.target_v[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_make_batch_deterministic() {
        let n = 3;
        let d = 4;
        let x1 = vec![0.5f32; n * d];
        let ts = vec![0.2, 0.5, 0.8];
        let b1 = rf_make_batch(&x1, n, d, &ts, 77).unwrap();
        let b2 = rf_make_batch(&x1, n, d, &ts, 77).unwrap();
        assert_eq!(b1.x0, b2.x0);
    }

    #[test]
    fn test_make_batch_noise_approx_gaussian() {
        // With N=1000, d=1, x0 should be approximately N(0,1)
        let n = 1000;
        let d = 1;
        let x1 = vec![0.0f32; n * d];
        let ts = vec![0.5f32; n];
        let batch = rf_make_batch(&x1, n, d, &ts, 314).unwrap();
        let mean: f32 = batch.x0.iter().sum::<f32>() / n as f32;
        let var: f32 = batch
            .x0
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>()
            / n as f32;
        let std = var.sqrt();
        assert!(mean.abs() < 0.15, "noise mean={} not near 0", mean);
        assert!((std - 1.0).abs() < 0.15, "noise std={} not near 1", std);
    }

    // ── ODE step functions ────────────────────────────────────────────────────

    #[test]
    fn test_euler_step_correctness() {
        let x = vec![1.0, 2.0];
        let v = vec![0.5, -0.5];
        let dt = 0.1;
        let next = rf_euler_step(&x, &v, dt);
        assert!((next[0] - 1.05).abs() < 1e-6);
        assert!((next[1] - 1.95).abs() < 1e-6);
    }

    #[test]
    fn test_heun_step_with_equal_velocities_matches_euler() {
        let x = vec![1.0, 2.0];
        let v = vec![0.5, -0.3];
        let dt = 0.2;
        let euler = rf_euler_step(&x, &v, dt);
        let heun = rf_heun_step(&x, &v, &v, dt);
        for (e, h) in euler.iter().zip(heun.iter()) {
            assert!((e - h).abs() < 1e-6);
        }
    }

    #[test]
    fn test_midpoint_step_correctness() {
        // x_next = x + dt * v_mid
        let x = vec![0.0, 0.0];
        let v_start = vec![1.0, 1.0]; // used externally to get x_mid
        let v_mid = vec![2.0, 3.0];
        let dt = 0.1;
        let next = rf_midpoint_step(&x, &v_start, &v_mid, dt);
        assert!((next[0] - 0.2).abs() < 1e-6);
        assert!((next[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_rk4_step_equal_k_matches_euler() {
        // If k1=k2=k3=k4=v, rk4 = x + dt/6*(v + 2v + 2v + v) = x + dt*v (Euler)
        let x = vec![1.0, 2.0];
        let v = vec![0.3, -0.1];
        let dt = 0.5;
        let euler = rf_euler_step(&x, &v, dt);
        let rk4 = rf_rk4_step(&x, &v, &v, &v, &v, dt);
        for (e, r) in euler.iter().zip(rk4.iter()) {
            assert!((e - r).abs() < 1e-5);
        }
    }

    // ── rf_euler_integrate ────────────────────────────────────────────────────

    #[test]
    fn test_euler_integrate_state_count() {
        let x0 = vec![0.0, 0.0];
        let velocities = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let times = vec![0.0, 0.5, 1.0];
        let traj = rf_euler_integrate(&x0, &velocities, &times).unwrap();
        assert_eq!(traj.states.len(), 3); // n_steps+1 = 3
        assert_eq!(traj.n_steps, 2);
    }

    #[test]
    fn test_euler_integrate_last_state_closer_to_x1() {
        // Starting at x0=[0,0], constant velocity toward x1=[1,0]
        let x0 = vec![0.0f32, 0.0];
        let x1 = [1.0f32, 0.0];
        let n_steps = 10;
        let v: Vec<f32> = x1.iter().zip(x0.iter()).map(|(&b, &a)| b - a).collect();
        let velocities = vec![v; n_steps];
        let times = rf_linspace(0.0, 1.0, n_steps + 1);
        let traj = rf_euler_integrate(&x0, &velocities, &times).unwrap();
        let last = &traj.states[n_steps];
        let dist_to_x1: f32 = last
            .iter()
            .zip(x1.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        let dist_to_x0: f32 = last
            .iter()
            .zip(x0.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        assert!(dist_to_x1 < dist_to_x0);
    }

    // ── RfOdeSolver ───────────────────────────────────────────────────────────

    #[test]
    fn test_solver_new_valid_config() {
        let config = RectifiedFlowConfig::default();
        assert!(RfOdeSolver::new(config).is_ok());
    }

    #[test]
    fn test_solver_new_invalid_n_steps_zero() {
        let config = RectifiedFlowConfig {
            n_steps: 0,
            ..Default::default()
        };
        assert!(matches!(
            RfOdeSolver::new(config),
            Err(RectifiedFlowError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn test_solver_new_invalid_t_min_gte_t_max() {
        let config = RectifiedFlowConfig {
            t_min: 0.9,
            t_max: 0.5,
            ..Default::default()
        };
        assert!(matches!(
            RfOdeSolver::new(config),
            Err(RectifiedFlowError::InvalidConfig { .. })
        ));
    }

    #[test]
    fn test_solver_generate_t_schedule_length() {
        let config = RectifiedFlowConfig::default(); // n_steps=100
        let solver = RfOdeSolver::new(config).unwrap();
        let sched = solver.generate_t_schedule();
        assert_eq!(sched.len(), 101);
    }

    #[test]
    fn test_solver_generate_t_schedule_endpoints() {
        let config = RectifiedFlowConfig::default();
        let solver = RfOdeSolver::new(config).unwrap();
        let sched = solver.generate_t_schedule();
        assert!((sched[0] - 0.0).abs() < 1e-6);
        assert!((sched[100] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_solver_integrate_euler_correct_n_steps() {
        let config = RectifiedFlowConfig {
            n_steps: 5,
            solver: RfSolverKind::Euler,
            ..Default::default()
        };
        let solver = RfOdeSolver::new(config).unwrap();
        let sched = solver.generate_t_schedule();
        let x0 = vec![0.0f32; 4];
        let traj = solver
            .integrate(&x0, &sched, |_, _| vec![1.0, 0.0, 0.0, 0.0])
            .unwrap();
        assert_eq!(traj.n_steps, 5);
        assert_eq!(traj.states.len(), 6);
    }

    #[test]
    fn test_solver_integrate_midpoint() {
        let config = RectifiedFlowConfig {
            n_steps: 4,
            solver: RfSolverKind::Midpoint,
            ..Default::default()
        };
        let solver = RfOdeSolver::new(config).unwrap();
        let sched = solver.generate_t_schedule();
        let x0 = vec![0.0f32; 2];
        // Constant velocity: x1 - x0 = [1, 1]
        let traj = solver
            .integrate(&x0, &sched, |_, _| vec![1.0, 1.0])
            .unwrap();
        assert_eq!(traj.n_steps, 4);
    }

    #[test]
    fn test_solver_integrate_heun() {
        let config = RectifiedFlowConfig {
            n_steps: 4,
            solver: RfSolverKind::Heun,
            ..Default::default()
        };
        let solver = RfOdeSolver::new(config).unwrap();
        let sched = solver.generate_t_schedule();
        let x0 = vec![0.0f32; 2];
        let traj = solver
            .integrate(&x0, &sched, |_, _| vec![1.0, 1.0])
            .unwrap();
        assert_eq!(traj.n_steps, 4);
    }

    #[test]
    fn test_solver_integrate_rk4() {
        let config = RectifiedFlowConfig {
            n_steps: 4,
            solver: RfSolverKind::Rk4,
            ..Default::default()
        };
        let solver = RfOdeSolver::new(config).unwrap();
        let sched = solver.generate_t_schedule();
        let x0 = vec![0.0f32; 2];
        let traj = solver
            .integrate(&x0, &sched, |_, _| vec![1.0, 1.0])
            .unwrap();
        assert_eq!(traj.n_steps, 4);
    }

    // ── ReFlow utilities ──────────────────────────────────────────────────────

    #[test]
    fn test_reflow_pair_returns_x0_and_final() {
        let x0 = vec![0.0, 0.0];
        let states = vec![vec![0.0, 0.0], vec![0.5, 0.5], vec![1.0, 1.0]];
        let traj = RfTrajectory {
            states,
            times: vec![0.0, 0.5, 1.0],
            n_steps: 2,
            d: 2,
        };
        let (got_x0, x_final) = rf_reflow_pair(&x0, &traj);
        assert_eq!(got_x0, x0);
        assert_eq!(x_final, vec![1.0, 1.0]);
    }

    #[test]
    fn test_trajectory_curvature_straight_is_zero() {
        // Straight trajectory: all steps move in the same direction
        let states = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
        ];
        let traj = RfTrajectory {
            states,
            times: vec![0.0, 0.333, 0.667, 1.0],
            n_steps: 3,
            d: 2,
        };
        let curv = rf_trajectory_curvature(&traj);
        assert!(curv.abs() < 1e-5, "curvature={} expected ~0", curv);
    }

    #[test]
    fn test_trajectory_length_positive() {
        let states = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        let traj = RfTrajectory {
            states,
            times: vec![0.0, 0.5, 1.0],
            n_steps: 2,
            d: 2,
        };
        let len = rf_trajectory_length(&traj);
        assert!(len > 0.0);
        assert!((len - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_straight_path_length_le_total_length() {
        let states = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 2.0], // curved path
        ];
        let traj = RfTrajectory {
            states,
            times: vec![0.0, 0.5, 1.0],
            n_steps: 2,
            d: 2,
        };
        let total = rf_trajectory_length(&traj);
        let straight = rf_straight_path_length(&traj);
        assert!(straight <= total + 1e-6);
    }

    #[test]
    fn test_straightness_ratio_straight_gives_one() {
        let states = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let traj = RfTrajectory {
            states,
            times: vec![0.0, 0.333, 0.667, 1.0],
            n_steps: 3,
            d: 1,
        };
        let ratio = rf_straightness_ratio(&traj);
        assert!((ratio - 1.0).abs() < 1e-5, "ratio={}", ratio);
    }

    #[test]
    fn test_straightness_ratio_curved_less_than_one() {
        // Detour: go right then back-left, ending up less far than path traveled
        let states = vec![
            vec![0.0, 0.0],
            vec![2.0, 0.0],
            vec![1.0, 0.0], // doubled back
        ];
        let traj = RfTrajectory {
            states,
            times: vec![0.0, 0.5, 1.0],
            n_steps: 2,
            d: 2,
        };
        let ratio = rf_straightness_ratio(&traj);
        assert!(ratio < 1.0, "ratio={} should be <1 for curved path", ratio);
    }

    // ── RfCoupling ────────────────────────────────────────────────────────────

    #[test]
    fn test_coupling_independent_sample_x0_shape() {
        let coupling = RfCoupling::new(CouplingMode::Independent);
        let n = 5;
        let d = 4;
        let x1 = vec![0.0f32; n * d];
        let x0 = coupling.sample_x0(&x1, n, d, 42);
        assert_eq!(x0.len(), n * d);
    }

    #[test]
    fn test_coupling_independent_deterministic() {
        let coupling = RfCoupling::new(CouplingMode::Independent);
        let n = 3;
        let d = 2;
        let x1 = vec![0.0f32; n * d];
        let x0a = coupling.sample_x0(&x1, n, d, 99);
        let x0b = coupling.sample_x0(&x1, n, d, 99);
        assert_eq!(x0a, x0b);
    }

    // ── rf_greedy_ot_match ────────────────────────────────────────────────────

    #[test]
    fn test_greedy_ot_match_length_n() {
        let n = 4;
        let d = 2;
        let x0 = vec![0.0f32; n * d];
        let x1 = vec![1.0f32; n * d];
        let assignment = rf_greedy_ot_match(&x0, &x1, n, d);
        assert_eq!(assignment.len(), n);
    }

    #[test]
    fn test_greedy_ot_match_assigns_nearest() {
        // x0 has two rows: [0,0] and [10,10]
        // x1 has two rows: [9,9] and [1,1]
        // Expected: x1[0]=[9,9] nearest x0[1]=[10,10], x1[1]=[1,1] nearest x0[0]=[0,0]
        let n = 2;
        let d = 2;
        let x0 = vec![0.0, 0.0, 10.0, 10.0]; // row0=[0,0], row1=[10,10]
        let x1 = vec![9.0, 9.0, 1.0, 1.0]; // row0=[9,9], row1=[1,1]
        let assignment = rf_greedy_ot_match(&x0, &x1, n, d);
        // x1[0]=[9,9] should map to x0[1]=[10,10] (greedy: closest first)
        assert_eq!(assignment[0], 1);
        // x1[1]=[1,1] should map to x0[0]=[0,0] (only one left)
        assert_eq!(assignment[1], 0);
    }

    // ── rf_compute_stats ──────────────────────────────────────────────────────

    #[test]
    fn test_compute_stats_correct_mean_min_max() {
        let losses = vec![1.0, 2.0, 3.0];
        let curvatures = vec![0.1, 0.2];
        let straightnesses = vec![0.9, 0.8];
        let stats = rf_compute_stats(&losses, &curvatures, &straightnesses);
        assert!((stats.mean_loss - 2.0).abs() < 1e-6);
        assert!((stats.min_loss - 1.0).abs() < 1e-6);
        assert!((stats.max_loss - 3.0).abs() < 1e-6);
        assert_eq!(stats.n_batches, 3);
    }

    #[test]
    fn test_compute_stats_empty() {
        let stats = rf_compute_stats(&[], &[], &[]);
        assert_eq!(stats.n_batches, 0);
    }

    // ── rf_format_stats / rf_format_config ────────────────────────────────────

    #[test]
    fn test_format_stats_non_empty() {
        let stats = RfStats {
            mean_loss: 0.5,
            min_loss: 0.1,
            max_loss: 0.9,
            mean_curvature: 0.01,
            mean_straightness: 0.99,
            n_batches: 10,
        };
        let s = rf_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("n_batches"));
    }

    #[test]
    fn test_format_config_non_empty() {
        let config = RectifiedFlowConfig::default();
        let s = rf_format_config(&config);
        assert!(!s.is_empty());
        assert!(s.contains("n_steps"));
    }

    // ── Additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_interpolate_t_quarter() {
        let x0 = vec![0.0, 0.0];
        let x1 = vec![4.0, 8.0];
        let result = rf_interpolate(&x0, &x1, 0.25).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_loss_per_sample_empty_input_error() {
        assert!(matches!(
            rf_loss_per_sample(&[], &[], &[], 2),
            Err(RectifiedFlowError::EmptyBatch)
        ));
    }

    #[test]
    fn test_make_batch_wrong_x1_length_error() {
        let x1 = vec![1.0f32; 5]; // should be n*d = 6
        let ts = vec![0.5; 2];
        assert!(matches!(
            rf_make_batch(&x1, 2, 3, &ts, 1),
            Err(RectifiedFlowError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_make_batch_wrong_t_length_error() {
        let n = 3;
        let d = 2;
        let x1 = vec![1.0f32; n * d];
        let ts = vec![0.5f32; 2]; // should be n=3
        assert!(matches!(
            rf_make_batch(&x1, n, d, &ts, 1),
            Err(RectifiedFlowError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_make_batch_invalid_t_in_batch() {
        let n = 2;
        let d = 2;
        let x1 = vec![1.0f32; n * d];
        let ts = vec![0.5f32, 1.5f32]; // 1.5 is invalid
        assert!(matches!(
            rf_make_batch(&x1, n, d, &ts, 1),
            Err(RectifiedFlowError::InvalidTimestep { .. })
        ));
    }

    #[test]
    fn test_heun_step_correctness() {
        let x = vec![0.0, 0.0];
        let v_start = vec![1.0, 0.0];
        let v_end = vec![0.0, 1.0];
        let dt = 1.0;
        // x_next = x + 0.5*(v_start + v_end) = [0.5, 0.5]
        let next = rf_heun_step(&x, &v_start, &v_end, dt);
        assert!((next[0] - 0.5).abs() < 1e-6);
        assert!((next[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_rk4_step_correctness() {
        let x = vec![0.0f32];
        let k1 = vec![1.0f32];
        let k2 = vec![2.0f32];
        let k3 = vec![2.0f32];
        let k4 = vec![3.0f32];
        let dt = 1.0;
        // x + dt/6*(1 + 2*2 + 2*2 + 3) = dt/6*12 = 2.0
        let next = rf_rk4_step(&x, &k1, &k2, &k3, &k4, dt);
        assert!((next[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_curvature_two_states_is_zero() {
        // Less than 3 states → no angle to compute
        let states = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let traj = RfTrajectory {
            states,
            times: vec![0.0, 1.0],
            n_steps: 1,
            d: 2,
        };
        assert_eq!(rf_trajectory_curvature(&traj), 0.0);
    }

    #[test]
    fn test_straight_path_length_single_state_zero() {
        let states = vec![vec![1.0, 2.0]];
        let traj = RfTrajectory {
            states,
            times: vec![0.0],
            n_steps: 0,
            d: 2,
        };
        assert_eq!(rf_straight_path_length(&traj), 0.0);
    }

    #[test]
    fn test_straightness_ratio_degenerate_zero_length_gives_one() {
        let states = vec![vec![1.0, 1.0], vec![1.0, 1.0]]; // stationary
        let traj = RfTrajectory {
            states,
            times: vec![0.0, 1.0],
            n_steps: 1,
            d: 2,
        };
        let ratio = rf_straightness_ratio(&traj);
        assert!((ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_coupling_minibt_ot_shape() {
        let coupling = RfCoupling::new(CouplingMode::MiniBatchOt);
        let n = 4;
        let d = 3;
        let x1 = vec![0.0f32; n * d];
        let x0 = coupling.sample_x0(&x1, n, d, 7);
        assert_eq!(x0.len(), n * d);
    }

    #[test]
    fn test_sample_t_zero_seed_still_works() {
        // seed=0 should be replaced by 1 internally
        let ts = rf_sample_t(0, 5, 0.0, 1.0);
        assert_eq!(ts.len(), 5);
        for t in &ts {
            assert!(*t >= 0.0 && *t <= 1.0);
        }
    }

    #[test]
    fn test_solver_integrate_constant_velocity_reaches_x1() {
        // Constant velocity v = x1 - x0; Euler should reach x1 after n steps
        let config = RectifiedFlowConfig {
            n_steps: 50,
            solver: RfSolverKind::Euler,
            ..Default::default()
        };
        let solver = RfOdeSolver::new(config).unwrap();
        let sched = solver.generate_t_schedule();
        let x0 = vec![0.0f32, 0.0];
        let x1 = vec![1.0f32, 2.0];
        let x1_c = x1.clone();
        let traj = solver
            .integrate(&x0, &sched, move |_, _| {
                x1_c.iter()
                    .zip([0.0, 0.0].iter())
                    .map(|(&b, &a)| b - a)
                    .collect()
            })
            .unwrap();
        let last = traj.states.last().unwrap();
        for (l, x) in last.iter().zip(x1.iter()) {
            assert!((l - x).abs() < 1e-4, "last={} x1={}", l, x);
        }
    }

    #[test]
    fn test_format_stats_contains_mean_loss() {
        let stats = rf_compute_stats(&[0.1, 0.3], &[0.0], &[1.0]);
        let s = rf_format_stats(&stats);
        assert!(s.contains("loss"));
    }

    #[test]
    fn test_format_config_contains_solver() {
        let config = RectifiedFlowConfig {
            solver: RfSolverKind::Rk4,
            ..Default::default()
        };
        let s = rf_format_config(&config);
        assert!(s.contains("Rk4"));
    }
}
