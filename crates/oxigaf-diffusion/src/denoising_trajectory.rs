//! Denoising trajectory tracking and analysis for diffusion models.
//!
//! Tracks and analyzes the sequence of intermediate latent states as the
//! diffusion model iteratively denoises from pure noise to a clean sample.
//! Useful for understanding convergence, detecting anomalies, and visualizing
//! the denoising process.
//!
//! Latent tensors are represented as flat [`Vec<f32>`] in row-major H×W×C
//! order with values typically in `[-1, 1]`.
//!
//! ## Example
//!
//! ```
//! use oxigaf_diffusion::denoising_trajectory::{
//!     DenoisingTrajectory, TrajectoryStep,
//!     compute_trajectory_stats, detect_trajectory_anomalies,
//! };
//!
//! let mut traj = DenoisingTrajectory::new();
//! traj.push(TrajectoryStep {
//!     timestep: 1000,
//!     latent: vec![0.5, 0.5, 0.5, 0.5],
//!     noise_level: 1.0,
//!     predicted_x0: None,
//! }).unwrap();
//! traj.push(TrajectoryStep {
//!     timestep: 0,
//!     latent: vec![0.1, 0.1, 0.1, 0.1],
//!     noise_level: 0.0,
//!     predicted_x0: None,
//! }).unwrap();
//!
//! let stats = compute_trajectory_stats(&traj).unwrap();
//! println!("Path length: {}", stats.total_path_length);
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during trajectory operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum TrajectoryError {
    /// Trajectory has no steps recorded.
    #[error("Empty trajectory: no steps recorded")]
    EmptyTrajectory,

    /// Not enough steps for the requested operation.
    #[error("Trajectory has only {got} steps, need at least {need}")]
    TooFewSteps { got: usize, need: usize },

    /// Latent dimension differs from the established trajectory dimension.
    #[error("Latent dimension mismatch: step {step} has {got} elements, expected {expected}")]
    DimensionMismatch {
        step: usize,
        got: usize,
        expected: usize,
    },

    /// Step index is outside the trajectory bounds.
    #[error("Step index {idx} out of range (trajectory has {len} steps)")]
    StepOutOfRange { idx: usize, len: usize },

    /// Interpolation parameter t is not in `[0, 1]`.
    #[error("Invalid interpolation parameter t={t}: must be in [0, 1]")]
    InvalidInterpolation { t: f32 },
}

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// A single step in the denoising trajectory.
#[derive(Debug, Clone)]
pub struct TrajectoryStep {
    /// Diffusion timestep (typically decreasing from T to 0).
    pub timestep: usize,
    /// Latent state at this step.
    pub latent: Vec<f32>,
    /// Estimated noise level (sigma or sqrt(1 − ᾱ)).
    pub noise_level: f32,
    /// Optional x₀ prediction at this step.
    pub predicted_x0: Option<Vec<f32>>,
}

/// Full denoising trajectory from T steps to 0.
///
/// The fields are private and the invariant `steps[i].latent.len() ==
/// latent_dim` for every step is enforced at construction time (via
/// [`DenoisingTrajectory::push`], the only way to add a step), so every
/// function that indexes a latent using `latent_dim` (or another step's
/// latent length) is panic-free by construction.
#[derive(Debug, Clone)]
pub struct DenoisingTrajectory {
    /// All recorded steps.
    steps: Vec<TrajectoryStep>,
    /// Dimensionality of each latent vector (set from the first pushed step).
    latent_dim: usize,
}

/// Aggregate statistics about a denoising trajectory.
#[derive(Debug, Clone)]
pub struct TrajectoryStats {
    /// Total number of steps.
    pub n_steps: usize,
    /// Sum of L2 distances between consecutive steps.
    pub total_path_length: f32,
    /// Average L2 distance per step.
    pub mean_step_size: f32,
    /// Maximum single-step movement.
    pub max_step_size: f32,
    /// Minimum single-step movement.
    pub min_step_size: f32,
    /// Variance of step sizes.
    pub step_size_variance: f32,
    /// Noise level at the first step.
    pub initial_noise_level: f32,
    /// Noise level at the last step.
    pub final_noise_level: f32,
    /// How strongly steps decelerate toward the end (0 = flat, 1 = strongly decelerating).
    pub convergence_score: f32,
}

/// Anomaly classification for trajectory behaviour.
#[derive(Debug, Clone, PartialEq)]
pub enum TrajectoryAnomaly {
    /// Step size nearly zero (< 1e-5).
    StuckStep { step_idx: usize },
    /// Step size exceeds 10× the mean.
    Explosion { step_idx: usize, size: f32 },
    /// Moved back toward the previous position (dot product of consecutive step
    /// vectors is negative).
    Oscillation { step_idx: usize },
    /// NaN or Inf detected in the latent at this step.
    NanOrInf { step_idx: usize },
}

/// Comparison between two trajectories of equal length and latent dimension.
#[derive(Debug, Clone)]
pub struct TrajectoryComparison {
    /// Number of steps in both trajectories.
    pub n_steps: usize,
    /// Mean L2 distance between corresponding latents.
    pub mean_latent_distance: f32,
    /// Maximum L2 distance at any single step.
    pub max_latent_distance: f32,
    /// `total_path_length_a / total_path_length_b` (saturates to 0 when b is
    /// zero-length).
    pub path_length_ratio: f32,
    /// L2 distance between the final latents of the two trajectories.
    pub final_distance: f32,
}

// ---------------------------------------------------------------------------
// DenoisingTrajectory methods
// ---------------------------------------------------------------------------

impl DenoisingTrajectory {
    /// Create an empty trajectory.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            latent_dim: 0,
        }
    }

    /// Append a step, validating latent dimension consistency.
    ///
    /// On the first call the trajectory adopts the step's latent length as its
    /// canonical `latent_dim`.
    pub fn push(&mut self, step: TrajectoryStep) -> Result<(), TrajectoryError> {
        if self.steps.is_empty() {
            self.latent_dim = step.latent.len();
        } else if step.latent.len() != self.latent_dim {
            return Err(TrajectoryError::DimensionMismatch {
                step: self.steps.len(),
                got: step.latent.len(),
                expected: self.latent_dim,
            });
        }
        self.steps.push(step);
        Ok(())
    }

    /// All recorded steps, in order.
    pub fn steps(&self) -> &[TrajectoryStep] {
        &self.steps
    }

    /// Dimensionality of each latent vector (0 for an empty trajectory).
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Number of steps recorded.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Returns `true` when no steps have been recorded.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Return a reference to the step at `idx`.
    pub fn get_step(&self, idx: usize) -> Result<&TrajectoryStep, TrajectoryError> {
        if idx >= self.steps.len() {
            return Err(TrajectoryError::StepOutOfRange {
                idx,
                len: self.steps.len(),
            });
        }
        Ok(&self.steps[idx])
    }

    /// Return the latent of the first step.
    pub fn initial_latent(&self) -> Result<&Vec<f32>, TrajectoryError> {
        self.steps
            .first()
            .map(|s| &s.latent)
            .ok_or(TrajectoryError::EmptyTrajectory)
    }

    /// Return the latent of the last step.
    pub fn final_latent(&self) -> Result<&Vec<f32>, TrajectoryError> {
        self.steps
            .last()
            .map(|s| &s.latent)
            .ok_or(TrajectoryError::EmptyTrajectory)
    }
}

impl Default for DenoisingTrajectory {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Distance / similarity primitives
// ---------------------------------------------------------------------------

/// Compute the L2 distance between two latent vectors.
///
/// Returns [`TrajectoryError::DimensionMismatch`] (using `step = 0`) when the
/// lengths differ.
pub fn latent_l2_distance(a: &[f32], b: &[f32]) -> Result<f32, TrajectoryError> {
    if a.len() != b.len() {
        return Err(TrajectoryError::DimensionMismatch {
            step: 0,
            got: b.len(),
            expected: a.len(),
        });
    }
    let sum_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    Ok(sum_sq.sqrt())
}

/// Compute the L1 distance between two latent vectors.
///
/// Returns [`TrajectoryError::DimensionMismatch`] (using `step = 0`) when the
/// lengths differ.
pub fn latent_l1_distance(a: &[f32], b: &[f32]) -> Result<f32, TrajectoryError> {
    if a.len() != b.len() {
        return Err(TrajectoryError::DimensionMismatch {
            step: 0,
            got: b.len(),
            expected: a.len(),
        });
    }
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    Ok(sum)
}

/// Cosine similarity between two latent vectors.
///
/// Returns `0.0` when either vector has a norm close to zero, or when the
/// vectors have different lengths (degenerate case treated as orthogonal).
pub fn latent_cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = latent_norm(a);
    let norm_b = latent_norm(b);
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// L2 norm of a latent vector. Returns `0.0` for an empty vector.
pub fn latent_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ---------------------------------------------------------------------------
// Step-size and statistics helpers
// ---------------------------------------------------------------------------

/// Compute the L2 distances between consecutive latent vectors.
///
/// Returns a [`Vec`] of length `n_steps - 1`.  Requires at least 2 steps.
pub fn compute_step_sizes(trajectory: &DenoisingTrajectory) -> Result<Vec<f32>, TrajectoryError> {
    let n = trajectory.steps.len();
    if n < 2 {
        return Err(TrajectoryError::TooFewSteps { got: n, need: 2 });
    }
    let mut sizes = Vec::with_capacity(n - 1);
    for i in 1..n {
        let dist =
            latent_l2_distance(&trajectory.steps[i - 1].latent, &trajectory.steps[i].latent)?;
        sizes.push(dist);
    }
    Ok(sizes)
}

/// Compute aggregate statistics for a trajectory.
///
/// Requires at least 2 steps (to compute at least one step size).
pub fn compute_trajectory_stats(
    trajectory: &DenoisingTrajectory,
) -> Result<TrajectoryStats, TrajectoryError> {
    let n = trajectory.steps.len();
    if n < 2 {
        return Err(TrajectoryError::TooFewSteps { got: n, need: 2 });
    }

    let step_sizes = compute_step_sizes(trajectory)?;
    let m = step_sizes.len() as f32;

    let total_path_length: f32 = step_sizes.iter().sum();
    let mean_step_size = total_path_length / m;

    let max_step_size = step_sizes.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_step_size = step_sizes.iter().cloned().fold(f32::INFINITY, f32::min);

    let variance = step_sizes
        .iter()
        .map(|&s| (s - mean_step_size).powi(2))
        .sum::<f32>()
        / m;

    // Linear regression slope of step sizes vs. step index.
    // x_i = i (0-based), y_i = step_size[i]
    let n_ss = step_sizes.len() as f32;
    let sum_x: f32 = (0..step_sizes.len()).map(|i| i as f32).sum();
    let sum_y: f32 = step_sizes.iter().sum();
    let sum_xy: f32 = step_sizes
        .iter()
        .enumerate()
        .map(|(i, &y)| i as f32 * y)
        .sum();
    let sum_x2: f32 = (0..step_sizes.len()).map(|i| (i as f32).powi(2)).sum();
    let denom = n_ss * sum_x2 - sum_x * sum_x;
    let slope = if denom.abs() < 1e-12 {
        0.0_f32
    } else {
        (n_ss * sum_xy - sum_x * sum_y) / denom
    };

    // Convergence score: normalised negative slope, clamped to [0, 1].
    let convergence_score = if slope < 0.0 {
        (slope.abs() / (mean_step_size + 1e-8)).min(1.0)
    } else {
        0.0
    };

    Ok(TrajectoryStats {
        n_steps: n,
        total_path_length,
        mean_step_size,
        max_step_size,
        min_step_size,
        step_size_variance: variance,
        initial_noise_level: trajectory.steps[0].noise_level,
        final_noise_level: trajectory.steps[n - 1].noise_level,
        convergence_score,
    })
}

// ---------------------------------------------------------------------------
// Anomaly detection
// ---------------------------------------------------------------------------

/// Detect anomalous behaviour in a trajectory.
///
/// Requires at least 2 steps.
pub fn detect_trajectory_anomalies(
    trajectory: &DenoisingTrajectory,
) -> Result<Vec<TrajectoryAnomaly>, TrajectoryError> {
    let n = trajectory.steps.len();
    if n < 2 {
        return Err(TrajectoryError::TooFewSteps { got: n, need: 2 });
    }

    let step_sizes = compute_step_sizes(trajectory)?;
    let mean_size: f32 = step_sizes.iter().sum::<f32>() / step_sizes.len() as f32;

    let mut anomalies = Vec::new();

    for i in 1..n {
        let step_size = step_sizes[i - 1];
        let latent = &trajectory.steps[i].latent;

        // NaN / Inf check.
        if latent.iter().any(|v| v.is_nan() || v.is_infinite()) {
            anomalies.push(TrajectoryAnomaly::NanOrInf { step_idx: i });
            // Still continue to check other anomalies.
        }

        // Stuck step.
        if step_size < 1e-5 {
            anomalies.push(TrajectoryAnomaly::StuckStep { step_idx: i });
        }

        // Explosion (only meaningful when mean_size > 0).
        if step_size > 10.0 * mean_size && mean_size > 0.0 {
            anomalies.push(TrajectoryAnomaly::Explosion {
                step_idx: i,
                size: step_size,
            });
        }

        // Oscillation: requires two consecutive step vectors (i >= 2).
        if i >= 2 {
            let prev_latent = &trajectory.steps[i - 2].latent;
            let curr_latent_prev = &trajectory.steps[i - 1].latent;
            let curr_latent = &trajectory.steps[i].latent;

            let dim = prev_latent.len();
            // Step vector i-1 → i-1: delta_{i-1} = curr_latent_prev - prev_latent
            // Step vector i-1 → i:   delta_i     = curr_latent     - curr_latent_prev
            let dot: f32 = (0..dim)
                .map(|d| {
                    let delta_prev = curr_latent_prev[d] - prev_latent[d];
                    let delta_curr = curr_latent[d] - curr_latent_prev[d];
                    delta_prev * delta_curr
                })
                .sum();

            if dot < 0.0 {
                anomalies.push(TrajectoryAnomaly::Oscillation { step_idx: i });
            }
        }
    }

    Ok(anomalies)
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

/// Linearly interpolate a latent vector at fractional position `t ∈ [0, 1]`.
///
/// `t = 0` maps to the initial (first) step; `t = 1` maps to the final step.
pub fn interpolate_trajectory_step(
    trajectory: &DenoisingTrajectory,
    t: f32,
) -> Result<Vec<f32>, TrajectoryError> {
    if !(0.0..=1.0).contains(&t) {
        return Err(TrajectoryError::InvalidInterpolation { t });
    }

    let n = trajectory.steps.len();
    if n == 0 {
        return Err(TrajectoryError::EmptyTrajectory);
    }
    if n == 1 {
        return Ok(trajectory.steps[0].latent.clone());
    }

    let idx_f = t * (n - 1) as f32;
    let lo = idx_f.floor() as usize;
    let hi = idx_f.ceil() as usize;
    let frac = idx_f - lo as f32;

    let lo = lo.min(n - 1);
    let hi = hi.min(n - 1);

    let latent_lo = &trajectory.steps[lo].latent;
    let latent_hi = &trajectory.steps[hi].latent;

    let interp: Vec<f32> = latent_lo
        .iter()
        .zip(latent_hi.iter())
        .map(|(&a, &b)| a + frac * (b - a))
        .collect();

    Ok(interp)
}

// ---------------------------------------------------------------------------
// Latent statistics
// ---------------------------------------------------------------------------

/// Compute element-wise statistics of a latent vector.
///
/// Returns `(mean, std, min, max)`. Returns `(0.0, 0.0, 0.0, 0.0)` for an
/// empty vector.
pub fn compute_latent_statistics(latent: &[f32]) -> (f32, f32, f32, f32) {
    if latent.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let n = latent.len() as f32;
    let mean = latent.iter().sum::<f32>() / n;
    let variance = latent.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();
    let min = latent.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = latent.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    (mean, std, min, max)
}

// ---------------------------------------------------------------------------
// Trajectory comparison
// ---------------------------------------------------------------------------

/// Compare two trajectories step by step.
///
/// Both trajectories must have the same number of steps and the same
/// `latent_dim`.
pub fn compare_trajectories(
    a: &DenoisingTrajectory,
    b: &DenoisingTrajectory,
) -> Result<TrajectoryComparison, TrajectoryError> {
    let n = a.steps.len();
    if n == 0 {
        return Err(TrajectoryError::EmptyTrajectory);
    }
    if n != b.steps.len() {
        return Err(TrajectoryError::TooFewSteps {
            got: b.steps.len().min(n),
            need: n,
        });
    }
    if a.latent_dim != b.latent_dim {
        return Err(TrajectoryError::DimensionMismatch {
            step: 0,
            got: b.latent_dim,
            expected: a.latent_dim,
        });
    }

    let mut distances = Vec::with_capacity(n);
    for i in 0..n {
        let d = latent_l2_distance(&a.steps[i].latent, &b.steps[i].latent)?;
        distances.push(d);
    }

    let mean_latent_distance = distances.iter().sum::<f32>() / n as f32;
    let max_latent_distance = distances.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Path lengths.
    let path_a: f32 = if n >= 2 {
        compute_step_sizes(a)?.iter().sum()
    } else {
        0.0
    };
    let path_b: f32 = if n >= 2 {
        compute_step_sizes(b)?.iter().sum()
    } else {
        0.0
    };

    let path_length_ratio = if path_b.abs() < 1e-12 {
        0.0
    } else {
        path_a / path_b
    };

    let final_distance = *distances.last().unwrap_or(&0.0);

    Ok(TrajectoryComparison {
        n_steps: n,
        mean_latent_distance,
        max_latent_distance,
        path_length_ratio,
        final_distance,
    })
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/// Resample a trajectory to exactly `n_steps` evenly-spaced steps via linear
/// interpolation.
///
/// The resampled steps have:
/// - `latent`: linearly interpolated
/// - `timestep` / `noise_level`: linearly interpolated between initial and final
/// - `predicted_x0`: always `None`
pub fn resample_trajectory(
    trajectory: &DenoisingTrajectory,
    n_steps: usize,
) -> Result<DenoisingTrajectory, TrajectoryError> {
    let orig_n = trajectory.steps.len();
    if orig_n == 0 {
        return Err(TrajectoryError::EmptyTrajectory);
    }
    if n_steps == 0 {
        return Err(TrajectoryError::TooFewSteps { got: 0, need: 1 });
    }

    let ts_init = trajectory.steps[0].timestep as f32;
    let ts_final = trajectory.steps[orig_n - 1].timestep as f32;
    let nl_init = trajectory.steps[0].noise_level;
    let nl_final = trajectory.steps[orig_n - 1].noise_level;

    let mut out = DenoisingTrajectory::new();

    for i in 0..n_steps {
        let t = if n_steps == 1 {
            0.0_f32
        } else {
            i as f32 / (n_steps - 1) as f32
        };

        let latent = interpolate_trajectory_step(trajectory, t)?;
        let timestep = (ts_init + t * (ts_final - ts_init)).round() as usize;
        let noise_level = nl_init + t * (nl_final - nl_init);

        out.push(TrajectoryStep {
            timestep,
            latent,
            noise_level,
            predicted_x0: None,
        })?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Smoothing
// ---------------------------------------------------------------------------

/// Apply a moving-average smoothing to latent vectors with the given window size.
///
/// Each output `latent[i]` is the mean of the latents in the window
/// `[max(0, i − w/2) .. min(n, i + w/2 + 1)]`.  `timestep` and `noise_level`
/// are preserved from the original steps.  `predicted_x0` is preserved as-is.
///
/// `window` must be ≥ 1.
pub fn smooth_trajectory(
    trajectory: &DenoisingTrajectory,
    window: usize,
) -> Result<DenoisingTrajectory, TrajectoryError> {
    let n = trajectory.steps.len();
    if n == 0 {
        return Err(TrajectoryError::EmptyTrajectory);
    }
    if window == 0 {
        return Err(TrajectoryError::TooFewSteps { got: 0, need: 1 });
    }

    let half = window / 2;
    let dim = trajectory.latent_dim;

    let mut out = DenoisingTrajectory::new();

    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let count = (hi - lo) as f32;

        let mut smoothed = vec![0.0_f32; dim];
        for j in lo..hi {
            for (d, v) in trajectory.steps[j].latent.iter().enumerate() {
                smoothed[d] += v;
            }
        }
        for v in &mut smoothed {
            *v /= count;
        }

        out.push(TrajectoryStep {
            timestep: trajectory.steps[i].timestep,
            latent: smoothed,
            noise_level: trajectory.steps[i].noise_level,
            predicted_x0: trajectory.steps[i].predicted_x0.clone(),
        })?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Trajectory divergence
// ---------------------------------------------------------------------------

/// Compute the per-step L2 distances between two trajectories.
///
/// Returns a [`Vec<f32>`] of length `n_steps`.  Both trajectories must have
/// the same number of steps and the same `latent_dim`.
pub fn trajectory_divergence(
    a: &DenoisingTrajectory,
    b: &DenoisingTrajectory,
) -> Result<Vec<f32>, TrajectoryError> {
    let n = a.steps.len();
    if n == 0 {
        return Err(TrajectoryError::EmptyTrajectory);
    }
    if n != b.steps.len() {
        return Err(TrajectoryError::TooFewSteps {
            got: b.steps.len().min(n),
            need: n,
        });
    }
    if a.latent_dim != b.latent_dim {
        return Err(TrajectoryError::DimensionMismatch {
            step: 0,
            got: b.latent_dim,
            expected: a.latent_dim,
        });
    }

    let mut dists = Vec::with_capacity(n);
    for i in 0..n {
        let d = latent_l2_distance(&a.steps[i].latent, &b.steps[i].latent)?;
        dists.push(d);
    }

    Ok(dists)
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Format trajectory statistics into a compact human-readable string.
pub fn format_trajectory_stats(stats: &TrajectoryStats) -> String {
    format!(
        "Trajectory[{} steps]: path={:.4}, mean_step={:.4}, max_step={:.4}, convergence={:.2}",
        stats.n_steps,
        stats.total_path_length,
        stats.mean_step_size,
        stats.max_step_size,
        stats.convergence_score,
    )
}

/// Format a list of anomalies into a compact human-readable string.
pub fn format_anomalies(anomalies: &[TrajectoryAnomaly]) -> String {
    if anomalies.is_empty() {
        return "No anomalies detected".to_string();
    }

    let parts: Vec<String> = anomalies
        .iter()
        .map(|a| match a {
            TrajectoryAnomaly::StuckStep { step_idx } => format!("StuckStep@{}", step_idx),
            TrajectoryAnomaly::Explosion { step_idx, size } => {
                format!("Explosion@{}(size={:.4})", step_idx, size)
            }
            TrajectoryAnomaly::Oscillation { step_idx } => {
                format!("Oscillation@{}", step_idx)
            }
            TrajectoryAnomaly::NanOrInf { step_idx } => format!("NanOrInf@{}", step_idx),
        })
        .collect();

    format!("Anomalies: {}", parts.join(", "))
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

    fn make_step(timestep: usize, latent: Vec<f32>, noise_level: f32) -> TrajectoryStep {
        TrajectoryStep {
            timestep,
            latent,
            noise_level,
            predicted_x0: None,
        }
    }

    fn two_step_traj() -> DenoisingTrajectory {
        let mut t = DenoisingTrajectory::new();
        t.push(make_step(100, vec![0.0, 0.0, 0.0, 0.0], 1.0))
            .unwrap();
        t.push(make_step(0, vec![1.0, 0.0, 0.0, 0.0], 0.0)).unwrap();
        t
    }

    fn three_step_traj() -> DenoisingTrajectory {
        let mut t = DenoisingTrajectory::new();
        t.push(make_step(100, vec![0.0, 0.0], 1.0)).unwrap();
        t.push(make_step(50, vec![1.0, 0.0], 0.5)).unwrap();
        t.push(make_step(0, vec![2.0, 0.0], 0.0)).unwrap();
        t
    }

    // -----------------------------------------------------------------------
    // DenoisingTrajectory::new
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_is_empty() {
        let t = DenoisingTrajectory::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert_eq!(t.latent_dim, 0);
    }

    #[test]
    fn test_default_equals_new() {
        let t: DenoisingTrajectory = Default::default();
        assert!(t.is_empty());
        assert_eq!(t.latent_dim, 0);
    }

    // -----------------------------------------------------------------------
    // DenoisingTrajectory::push
    // -----------------------------------------------------------------------

    #[test]
    fn test_push_sets_latent_dim() {
        let mut t = DenoisingTrajectory::new();
        t.push(make_step(10, vec![1.0, 2.0, 3.0], 0.5)).unwrap();
        assert_eq!(t.latent_dim, 3);
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn test_push_dimension_mismatch_error() {
        let mut t = DenoisingTrajectory::new();
        t.push(make_step(10, vec![1.0, 2.0], 0.5)).unwrap();
        let err = t.push(make_step(5, vec![1.0], 0.2)).unwrap_err();
        assert!(matches!(err, TrajectoryError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_push_multiple_same_dim_ok() {
        let mut t = DenoisingTrajectory::new();
        for i in 0..5 {
            t.push(make_step(i, vec![i as f32, 0.0], 0.1)).unwrap();
        }
        assert_eq!(t.len(), 5);
        assert_eq!(t.latent_dim, 2);
    }

    // -----------------------------------------------------------------------
    // DenoisingTrajectory::get_step
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_step_valid() {
        let t = two_step_traj();
        let s = t.get_step(1).unwrap();
        assert_eq!(s.timestep, 0);
    }

    #[test]
    fn test_get_step_out_of_range() {
        let t = two_step_traj();
        let err = t.get_step(5).unwrap_err();
        assert!(matches!(
            err,
            TrajectoryError::StepOutOfRange { idx: 5, len: 2 }
        ));
    }

    #[test]
    fn test_get_step_empty_trajectory() {
        let t = DenoisingTrajectory::new();
        let err = t.get_step(0).unwrap_err();
        assert!(matches!(
            err,
            TrajectoryError::StepOutOfRange { idx: 0, len: 0 }
        ));
    }

    // -----------------------------------------------------------------------
    // initial_latent / final_latent
    // -----------------------------------------------------------------------

    #[test]
    fn test_initial_latent_empty_error() {
        let t = DenoisingTrajectory::new();
        assert!(matches!(
            t.initial_latent().unwrap_err(),
            TrajectoryError::EmptyTrajectory
        ));
    }

    #[test]
    fn test_final_latent_empty_error() {
        let t = DenoisingTrajectory::new();
        assert!(matches!(
            t.final_latent().unwrap_err(),
            TrajectoryError::EmptyTrajectory
        ));
    }

    #[test]
    fn test_initial_latent_correct() {
        let t = two_step_traj();
        assert_eq!(*t.initial_latent().unwrap(), vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_final_latent_correct() {
        let t = two_step_traj();
        assert_eq!(*t.final_latent().unwrap(), vec![1.0, 0.0, 0.0, 0.0]);
    }

    // -----------------------------------------------------------------------
    // latent_l2_distance
    // -----------------------------------------------------------------------

    #[test]
    fn test_l2_distance_known_value() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        let d = latent_l2_distance(&a, &b).unwrap();
        assert!((d - 5.0).abs() < 1e-6, "expected 5.0, got {}", d);
    }

    #[test]
    fn test_l2_distance_same_vector() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((latent_l2_distance(&v, &v).unwrap()).abs() < 1e-7);
    }

    #[test]
    fn test_l2_distance_dim_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert!(matches!(
            latent_l2_distance(&a, &b).unwrap_err(),
            TrajectoryError::DimensionMismatch { .. }
        ));
    }

    // -----------------------------------------------------------------------
    // latent_l1_distance
    // -----------------------------------------------------------------------

    #[test]
    fn test_l1_distance_known_value() {
        // |1-4| + |2-6| + |3-9| = 3 + 4 + 6 = 13
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 9.0];
        let d = latent_l1_distance(&a, &b).unwrap();
        assert!((d - 13.0).abs() < 1e-6, "expected 13.0, got {}", d);
    }

    #[test]
    fn test_l1_distance_dim_mismatch() {
        assert!(latent_l1_distance(&[1.0], &[1.0, 2.0]).is_err());
    }

    // -----------------------------------------------------------------------
    // latent_cosine_sim
    // -----------------------------------------------------------------------

    #[test]
    fn test_cosine_sim_same_vector() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = latent_cosine_sim(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_sim_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = latent_cosine_sim(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_sim_zero_norm() {
        let zero = vec![0.0, 0.0, 0.0];
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(latent_cosine_sim(&zero, &v), 0.0);
        assert_eq!(latent_cosine_sim(&v, &zero), 0.0);
    }

    #[test]
    fn test_cosine_sim_dim_mismatch_returns_zero() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert_eq!(latent_cosine_sim(&a, &b), 0.0);
    }

    // -----------------------------------------------------------------------
    // latent_norm
    // -----------------------------------------------------------------------

    #[test]
    fn test_latent_norm_known() {
        let v = vec![3.0, 4.0];
        assert!((latent_norm(&v) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_latent_norm_zero_vector() {
        assert_eq!(latent_norm(&[0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_latent_norm_empty() {
        assert_eq!(latent_norm(&[]), 0.0);
    }

    // -----------------------------------------------------------------------
    // compute_step_sizes
    // -----------------------------------------------------------------------

    #[test]
    fn test_step_sizes_two_steps_one_value() {
        let t = two_step_traj();
        let sizes = compute_step_sizes(&t).unwrap();
        assert_eq!(sizes.len(), 1);
        // distance from [0,0,0,0] to [1,0,0,0] = 1
        assert!((sizes[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_step_sizes_monotone_trajectory() {
        let t = three_step_traj();
        let sizes = compute_step_sizes(&t).unwrap();
        assert_eq!(sizes.len(), 2);
        // each step moves [1,0] → distance 1
        assert!((sizes[0] - 1.0).abs() < 1e-6);
        assert!((sizes[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_step_sizes_too_few_steps() {
        let mut t = DenoisingTrajectory::new();
        t.push(make_step(0, vec![1.0], 0.0)).unwrap();
        assert!(matches!(
            compute_step_sizes(&t).unwrap_err(),
            TrajectoryError::TooFewSteps { got: 1, need: 2 }
        ));
    }

    // -----------------------------------------------------------------------
    // compute_trajectory_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_single_step_error() {
        let mut t = DenoisingTrajectory::new();
        t.push(make_step(0, vec![0.0], 0.0)).unwrap();
        assert!(matches!(
            compute_trajectory_stats(&t).unwrap_err(),
            TrajectoryError::TooFewSteps { .. }
        ));
    }

    #[test]
    fn test_stats_two_steps_known_path() {
        let t = two_step_traj();
        let stats = compute_trajectory_stats(&t).unwrap();
        assert_eq!(stats.n_steps, 2);
        assert!((stats.total_path_length - 1.0).abs() < 1e-6);
        assert!((stats.mean_step_size - 1.0).abs() < 1e-6);
        assert!((stats.max_step_size - 1.0).abs() < 1e-6);
        assert!((stats.min_step_size - 1.0).abs() < 1e-6);
        assert_eq!(stats.initial_noise_level, 1.0);
        assert_eq!(stats.final_noise_level, 0.0);
    }

    #[test]
    fn test_stats_three_steps_uniform() {
        let t = three_step_traj();
        let stats = compute_trajectory_stats(&t).unwrap();
        assert!((stats.total_path_length - 2.0).abs() < 1e-6);
        assert!((stats.mean_step_size - 1.0).abs() < 1e-6);
        assert_eq!(stats.step_size_variance, 0.0);
    }

    #[test]
    fn test_stats_convergence_score_in_range() {
        let t = three_step_traj();
        let stats = compute_trajectory_stats(&t).unwrap();
        assert!(stats.convergence_score >= 0.0 && stats.convergence_score <= 1.0);
    }

    // -----------------------------------------------------------------------
    // detect_trajectory_anomalies
    // -----------------------------------------------------------------------

    #[test]
    fn test_anomaly_no_anomaly() {
        let t = three_step_traj();
        let anomalies = detect_trajectory_anomalies(&t).unwrap();
        assert!(
            anomalies.is_empty(),
            "unexpected anomalies: {:?}",
            anomalies
        );
    }

    #[test]
    fn test_anomaly_stuck_step() {
        let mut t = DenoisingTrajectory::new();
        t.push(make_step(10, vec![1.0, 2.0], 1.0)).unwrap();
        // Very tiny move → stuck
        t.push(make_step(5, vec![1.0 + 1e-9, 2.0], 0.5)).unwrap();
        t.push(make_step(0, vec![2.0, 2.0], 0.0)).unwrap();
        let anomalies = detect_trajectory_anomalies(&t).unwrap();
        assert!(
            anomalies
                .iter()
                .any(|a| matches!(a, TrajectoryAnomaly::StuckStep { .. })),
            "expected StuckStep, got {:?}",
            anomalies
        );
    }

    #[test]
    fn test_anomaly_explosion() {
        // Many small steps of size 0.01, then a huge jump of ~1000.
        // mean ≈ (0.01 * 9 + 1000) / 10 ≈ 100.09
        // The large step (1000) > 10 * 100.09 ≈ 1000.9? No — need a bigger ratio.
        // Use 9 steps of 0.01 and one step of 10000 so last step >> 10 * mean.
        // mean = (9 * 0.01 + 10000) / 10 ≈ 1000.009
        // last step 10000 > 10 * 1000.009 ≈ 10000.09? Barely not.
        // Better: use many more small steps. 99 steps of 0.01 + 1 step of 1000.
        // mean ≈ (0.99 + 1000) / 100 ≈ 10.0099
        // last step 1000 > 10 * 10.0099 ≈ 100.099 → YES.
        let mut t = DenoisingTrajectory::new();
        let n_small = 100_usize;
        for i in 0..=n_small {
            let x = i as f32 * 0.01;
            t.push(make_step(
                n_small - i,
                vec![x, 0.0],
                (n_small - i) as f32 / n_small as f32,
            ))
            .unwrap();
        }
        // One large extra step
        t.push(make_step(0, vec![1000.0, 0.0], 0.0)).unwrap();
        let anomalies = detect_trajectory_anomalies(&t).unwrap();
        assert!(
            anomalies
                .iter()
                .any(|a| matches!(a, TrajectoryAnomaly::Explosion { .. })),
            "expected Explosion, got {:?}",
            anomalies
        );
    }

    #[test]
    fn test_anomaly_oscillation() {
        // Step 0→1: move +1 along x
        // Step 1→2: move −1 along x (opposite direction → oscillation)
        let mut t = DenoisingTrajectory::new();
        t.push(make_step(10, vec![0.0], 1.0)).unwrap();
        t.push(make_step(5, vec![1.0], 0.5)).unwrap();
        t.push(make_step(0, vec![0.0], 0.0)).unwrap();
        let anomalies = detect_trajectory_anomalies(&t).unwrap();
        assert!(
            anomalies
                .iter()
                .any(|a| matches!(a, TrajectoryAnomaly::Oscillation { .. })),
            "expected Oscillation, got {:?}",
            anomalies
        );
    }

    #[test]
    fn test_anomaly_nan_or_inf() {
        let mut t = DenoisingTrajectory::new();
        t.push(make_step(10, vec![0.0, 1.0], 1.0)).unwrap();
        t.push(make_step(0, vec![f32::NAN, 0.0], 0.0)).unwrap();
        let anomalies = detect_trajectory_anomalies(&t).unwrap();
        assert!(
            anomalies
                .iter()
                .any(|a| matches!(a, TrajectoryAnomaly::NanOrInf { .. })),
            "expected NanOrInf, got {:?}",
            anomalies
        );
    }

    #[test]
    fn test_anomaly_too_few_steps() {
        let mut t = DenoisingTrajectory::new();
        t.push(make_step(0, vec![0.0], 0.0)).unwrap();
        assert!(matches!(
            detect_trajectory_anomalies(&t).unwrap_err(),
            TrajectoryError::TooFewSteps { .. }
        ));
    }

    // -----------------------------------------------------------------------
    // interpolate_trajectory_step
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpolate_t0_equals_first() {
        let t = three_step_traj();
        let v = interpolate_trajectory_step(&t, 0.0).unwrap();
        assert_eq!(v, t.steps[0].latent);
    }

    #[test]
    fn test_interpolate_t1_equals_last() {
        let t = three_step_traj();
        let v = interpolate_trajectory_step(&t, 1.0).unwrap();
        assert_eq!(v, t.steps[2].latent);
    }

    #[test]
    fn test_interpolate_t_half_midpoint() {
        let t = three_step_traj();
        // steps: [0,0], [1,0], [2,0] → midpoint = [1,0]
        let v = interpolate_trajectory_step(&t, 0.5).unwrap();
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_invalid_t_negative() {
        let t = two_step_traj();
        assert!(matches!(
            interpolate_trajectory_step(&t, -0.1).unwrap_err(),
            TrajectoryError::InvalidInterpolation { .. }
        ));
    }

    #[test]
    fn test_interpolate_invalid_t_greater_than_one() {
        let t = two_step_traj();
        assert!(matches!(
            interpolate_trajectory_step(&t, 1.1).unwrap_err(),
            TrajectoryError::InvalidInterpolation { .. }
        ));
    }

    #[test]
    fn test_interpolate_empty_trajectory() {
        let t = DenoisingTrajectory::new();
        assert!(matches!(
            interpolate_trajectory_step(&t, 0.5).unwrap_err(),
            TrajectoryError::EmptyTrajectory
        ));
    }

    #[test]
    fn test_interpolate_single_step() {
        let mut t = DenoisingTrajectory::new();
        t.push(make_step(0, vec![3.0, 4.0], 0.0)).unwrap();
        let v = interpolate_trajectory_step(&t, 0.0).unwrap();
        assert_eq!(v, vec![3.0, 4.0]);
        let v2 = interpolate_trajectory_step(&t, 1.0).unwrap();
        assert_eq!(v2, vec![3.0, 4.0]);
    }

    // -----------------------------------------------------------------------
    // compute_latent_statistics
    // -----------------------------------------------------------------------

    #[test]
    fn test_latent_stats_known_values() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let (mean, std, min, max) = compute_latent_statistics(&v);
        assert!((mean - 2.5).abs() < 1e-6);
        // Variance = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4
        //           = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25
        assert!((std - 1.25_f32.sqrt()).abs() < 1e-5);
        assert_eq!(min, 1.0);
        assert_eq!(max, 4.0);
    }

    #[test]
    fn test_latent_stats_single_element() {
        let (mean, std, min, max) = compute_latent_statistics(&[7.0]);
        assert_eq!(mean, 7.0);
        assert_eq!(std, 0.0);
        assert_eq!(min, 7.0);
        assert_eq!(max, 7.0);
    }

    #[test]
    fn test_latent_stats_empty() {
        let (mean, std, min, max) = compute_latent_statistics(&[]);
        assert_eq!((mean, std, min, max), (0.0, 0.0, 0.0, 0.0));
    }

    // -----------------------------------------------------------------------
    // compare_trajectories
    // -----------------------------------------------------------------------

    #[test]
    fn test_compare_same_trajectory_distance_zero() {
        let t = three_step_traj();
        let cmp = compare_trajectories(&t, &t).unwrap();
        assert_eq!(cmp.n_steps, 3);
        assert!(cmp.mean_latent_distance < 1e-7);
        assert!(cmp.max_latent_distance < 1e-7);
        assert!(cmp.final_distance < 1e-7);
    }

    #[test]
    fn test_compare_path_length_ratio_equals_one() {
        let t = three_step_traj();
        let cmp = compare_trajectories(&t, &t).unwrap();
        assert!((cmp.path_length_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compare_different_step_counts() {
        let t2 = two_step_traj();
        let t3 = three_step_traj();
        // Different latent_dim AND step counts → error
        assert!(compare_trajectories(&t2, &t3).is_err());
    }

    #[test]
    fn test_compare_dim_mismatch() {
        let mut a = DenoisingTrajectory::new();
        a.push(make_step(0, vec![1.0, 2.0], 0.0)).unwrap();
        let mut b = DenoisingTrajectory::new();
        b.push(make_step(0, vec![1.0, 2.0, 3.0], 0.0)).unwrap();
        assert!(matches!(
            compare_trajectories(&a, &b).unwrap_err(),
            TrajectoryError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_compare_shifted_trajectory() {
        let mut a = DenoisingTrajectory::new();
        let mut b = DenoisingTrajectory::new();
        for _ in 0..3 {
            a.push(make_step(0, vec![0.0, 0.0], 0.0)).unwrap();
            b.push(make_step(0, vec![3.0, 4.0], 0.0)).unwrap();
        }
        let cmp = compare_trajectories(&a, &b).unwrap();
        assert!((cmp.mean_latent_distance - 5.0).abs() < 1e-6);
        assert!((cmp.final_distance - 5.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // resample_trajectory
    // -----------------------------------------------------------------------

    #[test]
    fn test_resample_preserves_endpoints() {
        let t = three_step_traj();
        let r = resample_trajectory(&t, 5).unwrap();
        assert_eq!(r.len(), 5);
        // First latent should match original first
        for (a, b) in r.steps[0].latent.iter().zip(t.steps[0].latent.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
        // Last latent should match original last
        for (a, b) in r
            .steps
            .last()
            .unwrap()
            .latent
            .iter()
            .zip(t.steps.last().unwrap().latent.iter())
        {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_resample_to_one_step() {
        let t = three_step_traj();
        let r = resample_trajectory(&t, 1).unwrap();
        assert_eq!(r.len(), 1);
        // Single step should be the initial latent (t=0)
        for (a, b) in r.steps[0].latent.iter().zip(t.steps[0].latent.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_resample_zero_steps_error() {
        let t = three_step_traj();
        assert!(matches!(
            resample_trajectory(&t, 0).unwrap_err(),
            TrajectoryError::TooFewSteps { .. }
        ));
    }

    #[test]
    fn test_resample_empty_error() {
        let t = DenoisingTrajectory::new();
        assert!(matches!(
            resample_trajectory(&t, 3).unwrap_err(),
            TrajectoryError::EmptyTrajectory
        ));
    }

    #[test]
    fn test_resample_predicted_x0_is_none() {
        let mut t = DenoisingTrajectory::new();
        t.push(TrajectoryStep {
            timestep: 10,
            latent: vec![1.0, 2.0],
            noise_level: 0.5,
            predicted_x0: Some(vec![0.5, 0.5]),
        })
        .unwrap();
        t.push(make_step(0, vec![2.0, 3.0], 0.0)).unwrap();
        let r = resample_trajectory(&t, 3).unwrap();
        for s in &r.steps {
            assert!(s.predicted_x0.is_none());
        }
    }

    // -----------------------------------------------------------------------
    // smooth_trajectory
    // -----------------------------------------------------------------------

    #[test]
    fn test_smooth_window1_identity() {
        let t = three_step_traj();
        let s = smooth_trajectory(&t, 1).unwrap();
        assert_eq!(s.len(), 3);
        for (orig, smoothed) in t.steps.iter().zip(s.steps.iter()) {
            for (a, b) in orig.latent.iter().zip(smoothed.latent.iter()) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_smooth_large_window_nearly_constant() {
        let mut t = DenoisingTrajectory::new();
        t.push(make_step(3, vec![0.0, 0.0], 1.0)).unwrap();
        t.push(make_step(2, vec![1.0, 0.0], 0.5)).unwrap();
        t.push(make_step(1, vec![2.0, 0.0], 0.25)).unwrap();
        t.push(make_step(0, vec![3.0, 0.0], 0.0)).unwrap();
        // Window covers all steps → each output is the global mean ≈ [1.5, 0]
        // (edges will be averages of fewer elements, so tolerance is loose)
        let s = smooth_trajectory(&t, 10).unwrap();
        assert_eq!(s.len(), 4);
        // Middle steps should be close to the full average
        let mid = &s.steps[1].latent[0];
        let mid2 = &s.steps[2].latent[0];
        assert!(
            (mid - mid2).abs() < 1.0,
            "values diverged more than expected"
        );
    }

    #[test]
    fn test_smooth_empty_error() {
        let t = DenoisingTrajectory::new();
        assert!(matches!(
            smooth_trajectory(&t, 3).unwrap_err(),
            TrajectoryError::EmptyTrajectory
        ));
    }

    #[test]
    fn test_smooth_window_zero_error() {
        let t = three_step_traj();
        assert!(matches!(
            smooth_trajectory(&t, 0).unwrap_err(),
            TrajectoryError::TooFewSteps { .. }
        ));
    }

    // -----------------------------------------------------------------------
    // trajectory_divergence
    // -----------------------------------------------------------------------

    #[test]
    fn test_divergence_same_trajectory_zeros() {
        let t = three_step_traj();
        let divs = trajectory_divergence(&t, &t).unwrap();
        assert_eq!(divs.len(), 3);
        for d in divs {
            assert!(d.abs() < 1e-7);
        }
    }

    #[test]
    fn test_divergence_shifted_trajectory() {
        let mut a = DenoisingTrajectory::new();
        let mut b = DenoisingTrajectory::new();
        for _ in 0..2 {
            a.push(make_step(0, vec![0.0, 0.0], 0.0)).unwrap();
            b.push(make_step(0, vec![3.0, 4.0], 0.0)).unwrap();
        }
        let divs = trajectory_divergence(&a, &b).unwrap();
        for d in divs {
            assert!((d - 5.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_divergence_step_count_mismatch() {
        let t2 = two_step_traj();
        let t3 = three_step_traj();
        assert!(trajectory_divergence(&t2, &t3).is_err());
    }

    #[test]
    fn test_divergence_empty_trajectory() {
        let a = DenoisingTrajectory::new();
        let b = DenoisingTrajectory::new();
        assert!(matches!(
            trajectory_divergence(&a, &b).unwrap_err(),
            TrajectoryError::EmptyTrajectory
        ));
    }

    // -----------------------------------------------------------------------
    // format_trajectory_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_trajectory_stats_non_empty() {
        let t = two_step_traj();
        let stats = compute_trajectory_stats(&t).unwrap();
        let s = format_trajectory_stats(&stats);
        assert!(s.starts_with("Trajectory["), "got: {}", s);
        assert!(s.contains("path="), "got: {}", s);
        assert!(s.contains("convergence="), "got: {}", s);
    }

    // -----------------------------------------------------------------------
    // format_anomalies
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_anomalies_empty() {
        assert_eq!(format_anomalies(&[]), "No anomalies detected");
    }

    #[test]
    fn test_format_anomalies_non_empty() {
        let a = vec![
            TrajectoryAnomaly::StuckStep { step_idx: 2 },
            TrajectoryAnomaly::Explosion {
                step_idx: 5,
                size: 0.45,
            },
        ];
        let s = format_anomalies(&a);
        assert!(s.starts_with("Anomalies:"), "got: {}", s);
        assert!(s.contains("StuckStep@2"), "got: {}", s);
        assert!(s.contains("Explosion@5"), "got: {}", s);
    }

    // -----------------------------------------------------------------------
    // TrajectoryAnomaly variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_anomaly_variants_clone_and_eq() {
        let a = TrajectoryAnomaly::StuckStep { step_idx: 3 };
        let b = a.clone();
        assert_eq!(a, b);

        let c = TrajectoryAnomaly::Oscillation { step_idx: 7 };
        assert_ne!(a, c);

        let d = TrajectoryAnomaly::NanOrInf { step_idx: 1 };
        let e = d.clone();
        assert_eq!(d, e);
    }

    // -----------------------------------------------------------------------
    // TrajectoryComparison fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_comparison_fields_accessible() {
        let t = three_step_traj();
        let cmp = compare_trajectories(&t, &t).unwrap();
        let _ = cmp.n_steps;
        let _ = cmp.mean_latent_distance;
        let _ = cmp.max_latent_distance;
        let _ = cmp.path_length_ratio;
        let _ = cmp.final_distance;
    }

    // -----------------------------------------------------------------------
    // TrajectoryStats fields
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_fields_accessible() {
        let t = three_step_traj();
        let stats = compute_trajectory_stats(&t).unwrap();
        let _ = stats.n_steps;
        let _ = stats.total_path_length;
        let _ = stats.mean_step_size;
        let _ = stats.max_step_size;
        let _ = stats.min_step_size;
        let _ = stats.step_size_variance;
        let _ = stats.initial_noise_level;
        let _ = stats.final_noise_level;
        let _ = stats.convergence_score;
    }
}
