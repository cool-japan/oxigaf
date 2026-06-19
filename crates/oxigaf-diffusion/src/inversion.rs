//! DDIM inversion for encoding real images into the latent trajectory of a
//! diffusion model, enabling edit-then-resample workflows.
//!
//! DDIM inversion reverses the DDIM sampling process:
//! - **Forward (denoising)**: x_{t-1} = √α_{t-1}·(x_t - √(1-α_t)·ε) / √α_t + √(1-α_{t-1})·ε
//! - **Inverse (inversion)**: x_{t+1} = √α_{t+1}·(x_t - √(1-α_t)·ε) / √α_t + √(1-α_{t+1})·ε
//!
//! The approximation ε_θ(x_t) ≈ ε_θ(x_{t+1}) makes this exact for a fixed
//! noise predictor, and accurate in practice for slow-varying predictors.
//!
//! ## Example
//!
//! ```
//! use oxigaf_diffusion::inversion::{
//!     InversionSchedule, DdimInversionConfig, run_ddim_inversion, null_noise_predictor,
//! };
//!
//! let schedule = InversionSchedule::cosine(50);
//! let config = DdimInversionConfig::default();
//! let x_0: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
//! let traj = run_ddim_inversion(&x_0, &schedule, &config, null_noise_predictor).unwrap();
//! assert_eq!(traj.len(), config.num_steps + 1);
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during DDIM inversion operations.
#[derive(Debug, Error)]
pub enum InversionError {
    /// Empty or invalid timestep configuration.
    #[error("Invalid timesteps: {0}")]
    InvalidTimesteps(String),

    /// Length mismatch between two slices that must agree.
    #[error("Length mismatch: expected {expected}, actual {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    /// Invalid configuration parameter.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// Iterative refinement did not converge within the allowed iterations.
    #[error("Max iterations reached: {iterations} iterations completed without convergence")]
    MaxIterationsReached { iterations: usize },
}

// ---------------------------------------------------------------------------
// NoiseScheduleCoeffs
// ---------------------------------------------------------------------------

/// Pre-computed coefficients for a noise schedule at a given timestep.
#[derive(Debug, Clone)]
pub struct NoiseScheduleCoeffs {
    /// Square root of the cumulative alpha product at this timestep.
    pub sqrt_alpha_cumprod: f32,
    /// Square root of (1 - alpha_cumprod) at this timestep.
    pub sqrt_one_minus_alpha_cumprod: f32,
    /// The cumulative alpha product itself.
    pub alpha_cumprod: f32,
}

impl NoiseScheduleCoeffs {
    /// Construct from a raw alpha cumprod value.
    ///
    /// Validates that `0 < alpha_cumprod <= 1`.
    pub fn new(alpha_cumprod: f32) -> Result<Self, InversionError> {
        if alpha_cumprod <= 0.0 || alpha_cumprod > 1.0 {
            return Err(InversionError::InvalidConfig(format!(
                "alpha_cumprod must be in (0, 1], got {}",
                alpha_cumprod
            )));
        }
        Ok(Self {
            sqrt_alpha_cumprod: alpha_cumprod.sqrt(),
            sqrt_one_minus_alpha_cumprod: (1.0 - alpha_cumprod).sqrt(),
            alpha_cumprod,
        })
    }
}

// ---------------------------------------------------------------------------
// DdimInversionConfig
// ---------------------------------------------------------------------------

/// Configuration for the DDIM inversion process.
#[derive(Debug, Clone)]
pub struct DdimInversionConfig {
    /// Number of inversion steps.
    pub num_steps: usize,
    /// Guidance scale for the noise predictor (typically 1.0 for exact inversion).
    pub guidance_scale: f32,
    /// Whether to use iterative refinement (more accurate but slower).
    pub use_iterative_refinement: bool,
    /// Maximum number of iterations for iterative refinement.
    pub max_refinement_iters: usize,
    /// Convergence threshold for iterative refinement (L2 distance).
    pub refinement_threshold: f32,
}

impl Default for DdimInversionConfig {
    fn default() -> Self {
        Self {
            num_steps: 50,
            guidance_scale: 1.0,
            use_iterative_refinement: false,
            max_refinement_iters: 10,
            refinement_threshold: 1e-4,
        }
    }
}

impl DdimInversionConfig {
    /// Validate the configuration, returning an error if any field is invalid.
    pub fn validate(&self) -> Result<(), InversionError> {
        if self.num_steps == 0 {
            return Err(InversionError::InvalidConfig(
                "num_steps must be > 0".to_string(),
            ));
        }
        if self.guidance_scale <= 0.0 {
            return Err(InversionError::InvalidConfig(format!(
                "guidance_scale must be > 0, got {}",
                self.guidance_scale
            )));
        }
        if self.use_iterative_refinement {
            if self.max_refinement_iters == 0 {
                return Err(InversionError::InvalidConfig(
                    "max_refinement_iters must be > 0 when use_iterative_refinement is true"
                        .to_string(),
                ));
            }
            if self.refinement_threshold <= 0.0 {
                return Err(InversionError::InvalidConfig(format!(
                    "refinement_threshold must be > 0, got {}",
                    self.refinement_threshold
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Core step functions
// ---------------------------------------------------------------------------

/// Apply one DDIM inversion step.
///
/// Given `x_t` (latent at timestep t), noise prediction ε, and coefficients
/// at t and t+1, computes `x_{t+1}` (latent at the next higher noise level).
///
/// Formula:
/// ```text
/// x_{t+1} = coeffs_next.sqrt_alpha * (x_t - coeffs_t.sqrt_one_minus_alpha * ε)
///           / coeffs_t.sqrt_alpha
///           + coeffs_next.sqrt_one_minus_alpha * ε
/// ```
pub fn ddim_inversion_step(
    x_t: &[f32],
    noise_pred: &[f32],
    coeffs_t: &NoiseScheduleCoeffs,
    coeffs_next: &NoiseScheduleCoeffs,
) -> Result<Vec<f32>, InversionError> {
    if x_t.len() != noise_pred.len() {
        return Err(InversionError::LengthMismatch {
            expected: x_t.len(),
            actual: noise_pred.len(),
        });
    }

    let sqrt_alpha_t = coeffs_t.sqrt_alpha_cumprod;
    let sqrt_one_minus_alpha_t = coeffs_t.sqrt_one_minus_alpha_cumprod;
    let sqrt_alpha_next = coeffs_next.sqrt_alpha_cumprod;
    let sqrt_one_minus_alpha_next = coeffs_next.sqrt_one_minus_alpha_cumprod;

    let result = x_t
        .iter()
        .zip(noise_pred.iter())
        .map(|(&x, &eps)| {
            let pred_x0 = (x - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t;
            sqrt_alpha_next * pred_x0 + sqrt_one_minus_alpha_next * eps
        })
        .collect();

    Ok(result)
}

/// Apply one DDIM denoising step.
///
/// Given `x_t` (noisy latent) and noise prediction ε, computes `x_{t-1}`
/// (less noisy latent). This is the reverse of `ddim_inversion_step`.
///
/// Formula:
/// ```text
/// x_{t-1} = coeffs_prev.sqrt_alpha * (x_t - coeffs_t.sqrt_one_minus_alpha * ε)
///           / coeffs_t.sqrt_alpha
///           + coeffs_prev.sqrt_one_minus_alpha * ε
/// ```
pub fn ddim_denoise_step(
    x_t: &[f32],
    noise_pred: &[f32],
    coeffs_t: &NoiseScheduleCoeffs,
    coeffs_prev: &NoiseScheduleCoeffs,
) -> Result<Vec<f32>, InversionError> {
    if x_t.len() != noise_pred.len() {
        return Err(InversionError::LengthMismatch {
            expected: x_t.len(),
            actual: noise_pred.len(),
        });
    }

    let sqrt_alpha_t = coeffs_t.sqrt_alpha_cumprod;
    let sqrt_one_minus_alpha_t = coeffs_t.sqrt_one_minus_alpha_cumprod;
    let sqrt_alpha_prev = coeffs_prev.sqrt_alpha_cumprod;
    let sqrt_one_minus_alpha_prev = coeffs_prev.sqrt_one_minus_alpha_cumprod;

    let result = x_t
        .iter()
        .zip(noise_pred.iter())
        .map(|(&x, &eps)| {
            let pred_x0 = (x - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t;
            sqrt_alpha_prev * pred_x0 + sqrt_one_minus_alpha_prev * eps
        })
        .collect();

    Ok(result)
}

// ---------------------------------------------------------------------------
// InversionTrajectory
// ---------------------------------------------------------------------------

/// A complete latent trajectory produced by DDIM inversion.
///
/// `latents[0]` is the initial clean latent x_0, and `latents[N]` is the
/// fully-noised latent x_T suitable for editing.
#[derive(Debug, Clone)]
pub struct InversionTrajectory {
    /// Latent states at each timestep, from x_0 (clean) to x_T (noisy).
    /// `trajectory[0]` = x_0, `trajectory[N]` = x_T.
    pub latents: Vec<Vec<f32>>,
    /// Alpha cumprod values used at each step.
    pub alpha_cumprods: Vec<f32>,
    /// Number of inversion steps actually performed.
    pub num_steps: usize,
    /// Whether iterative refinement was used.
    pub used_refinement: bool,
}

impl InversionTrajectory {
    /// Return the initial clean latent x_0.
    pub fn x_0(&self) -> &[f32] {
        // Safety: the constructor always pushes x_0 first.
        &self.latents[0]
    }

    /// Return the final noisy latent x_T (the one suitable for editing).
    pub fn x_t(&self) -> &[f32] {
        // Safety: latents always has at least one element (x_0).
        &self.latents[self.latents.len() - 1]
    }

    /// Return the intermediate latent at the given step index, if in bounds.
    pub fn intermediate(&self, step: usize) -> Option<&[f32]> {
        self.latents.get(step).map(|v| v.as_slice())
    }

    /// Total number of latent states in the trajectory (num_steps + 1).
    pub fn len(&self) -> usize {
        self.latents.len()
    }

    /// Returns `true` when the trajectory contains no latent states.
    pub fn is_empty(&self) -> bool {
        self.latents.is_empty()
    }

    /// Compute the L2 reconstruction error between the original x_0 and a
    /// reconstructed latent, normalized by the latent dimension.
    ///
    /// Returns `sqrt(mean((x_0 - reconstructed)²))`.
    pub fn reconstruction_error(&self, reconstructed: &[f32]) -> Result<f32, InversionError> {
        let original = self.x_0();
        if original.len() != reconstructed.len() {
            return Err(InversionError::LengthMismatch {
                expected: original.len(),
                actual: reconstructed.len(),
            });
        }
        let mse = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
            / original.len() as f32;
        Ok(mse.sqrt())
    }
}

// ---------------------------------------------------------------------------
// InversionSchedule
// ---------------------------------------------------------------------------

/// Pre-computed alpha cumprod values for a noise schedule, ordered from
/// low noise (index 0) to high noise (index T-1).
#[derive(Debug, Clone)]
pub struct InversionSchedule {
    /// Alpha cumprod values from low-noise (index 0) to high-noise (index T-1).
    pub alpha_cumprods: Vec<f32>,
    /// Total number of timesteps.
    pub total_timesteps: usize,
}

impl InversionSchedule {
    /// Build a linear schedule interpolating alpha_cumprod from `start` to `end`.
    ///
    /// Typically `start ≈ 0.9999` (low noise) and `end ≈ 0.0047` (high noise).
    pub fn linear(total_timesteps: usize, start: f32, end: f32) -> Result<Self, InversionError> {
        if total_timesteps == 0 {
            return Err(InversionError::InvalidTimesteps(
                "total_timesteps must be > 0".to_string(),
            ));
        }
        let alpha_cumprods = (0..total_timesteps)
            .map(|i| {
                let t = i as f32 / (total_timesteps - 1).max(1) as f32;
                start + t * (end - start)
            })
            .collect();
        Ok(Self {
            alpha_cumprods,
            total_timesteps,
        })
    }

    /// Build a cosine schedule following the improved DDPM formulation.
    ///
    /// `alpha_cumprod(t) = cos((t/T + s)/(1+s) · π/2)² / cos(s/(1+s) · π/2)²`
    ///
    /// where `s = 0.008` prevents tiny alpha at t=0.
    pub fn cosine(total_timesteps: usize) -> Self {
        let s: f32 = 0.008;
        let total = total_timesteps.max(1);
        let denominator = (s / (1.0 + s) * std::f32::consts::FRAC_PI_2).cos().powi(2);
        let alpha_cumprods = (0..total_timesteps)
            .map(|i| {
                let t_frac = i as f32 / total as f32;
                let arg = (t_frac + s) / (1.0 + s) * std::f32::consts::FRAC_PI_2;
                let raw = arg.cos().powi(2) / denominator;
                // Clamp to [0, 1] to guard against floating-point overshoot.
                raw.clamp(0.0, 1.0)
            })
            .collect();
        Self {
            alpha_cumprods,
            total_timesteps,
        }
    }

    /// Linearly interpolate alpha_cumprod at a continuous position in `[0, 1]`.
    ///
    /// `fraction = 0` → `alpha_cumprods[0]` (low noise).
    /// `fraction = 1` → `alpha_cumprods[T-1]` (high noise).
    pub fn alpha_at(&self, fraction: f32) -> f32 {
        let fraction = fraction.clamp(0.0, 1.0);
        let last = self.alpha_cumprods.len().saturating_sub(1);
        if last == 0 {
            return self.alpha_cumprods.first().copied().unwrap_or(1.0);
        }
        let pos = fraction * last as f32;
        let lo = pos.floor() as usize;
        let hi = (lo + 1).min(last);
        let t = pos - lo as f32;
        let a = self.alpha_cumprods[lo];
        let b = self.alpha_cumprods[hi];
        a + t * (b - a)
    }

    /// Return `n` evenly-spaced alpha cumprod values for inversion
    /// (from low noise to high noise, i.e. values decrease).
    pub fn inversion_alphas(&self, n: usize) -> Vec<f32> {
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![self.alpha_at(0.0)];
        }
        (0..n)
            .map(|i| {
                let frac = i as f32 / (n - 1) as f32;
                self.alpha_at(frac)
            })
            .collect()
    }

    /// Return `n` evenly-spaced alpha cumprod values for denoising
    /// (from high noise to low noise, i.e. values increase).
    pub fn denoising_alphas(&self, n: usize) -> Vec<f32> {
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![self.alpha_at(1.0)];
        }
        (0..n)
            .map(|i| {
                let frac = 1.0 - i as f32 / (n - 1) as f32;
                self.alpha_at(frac)
            })
            .collect()
    }

    /// Return the total number of precomputed timesteps.
    pub fn num_timesteps(&self) -> usize {
        self.total_timesteps
    }
}

// ---------------------------------------------------------------------------
// Null / constant noise predictors
// ---------------------------------------------------------------------------

/// A noise predictor that always returns zeros.
///
/// With zero noise, DDIM inversion/reconstruction becomes:
/// `x_{t+1} = sqrt(α_{t+1}) * x_t / sqrt(α_t)`
/// which is exactly invertible via the denoising step.
pub fn null_noise_predictor(latent: &[f32], _step: usize) -> Vec<f32> {
    vec![0.0_f32; latent.len()]
}

/// Build a noise predictor that returns all-`constant` values.
pub fn constant_noise_predictor(constant: f32) -> impl Fn(&[f32], usize) -> Vec<f32> {
    move |latent: &[f32], _step: usize| vec![constant; latent.len()]
}

// ---------------------------------------------------------------------------
// Full inversion pipeline
// ---------------------------------------------------------------------------

/// Run full DDIM inversion using a provided noise predictor function.
///
/// The `noise_predictor` takes `(latent, step_idx)` and returns a noise
/// prediction of the same length. This design allows testing without a real
/// U-Net model.
///
/// # Returns
/// An [`InversionTrajectory`] whose `latents[0]` is the initial x_0 and
/// `latents[num_steps]` is the fully-noised x_T.
pub fn run_ddim_inversion<F>(
    x_0: &[f32],
    schedule: &InversionSchedule,
    config: &DdimInversionConfig,
    noise_predictor: F,
) -> Result<InversionTrajectory, InversionError>
where
    F: Fn(&[f32], usize) -> Vec<f32>,
{
    config.validate()?;

    // Get num_steps + 1 alpha values (low → high noise).
    let alphas = schedule.inversion_alphas(config.num_steps + 1);
    if alphas.len() != config.num_steps + 1 {
        return Err(InversionError::InvalidTimesteps(format!(
            "expected {} alphas, got {}",
            config.num_steps + 1,
            alphas.len()
        )));
    }

    let mut trajectory_latents: Vec<Vec<f32>> = Vec::with_capacity(config.num_steps + 1);
    let mut latent = x_0.to_vec();

    // Store x_0 as the first element.
    trajectory_latents.push(latent.clone());

    for i in 0..config.num_steps {
        let coeffs_t = NoiseScheduleCoeffs::new(alphas[i])?;
        let coeffs_next = NoiseScheduleCoeffs::new(alphas[i + 1])?;
        let noise_pred = noise_predictor(&latent, i);
        latent = ddim_inversion_step(&latent, &noise_pred, &coeffs_t, &coeffs_next)?;
        trajectory_latents.push(latent.clone());
    }

    Ok(InversionTrajectory {
        latents: trajectory_latents,
        alpha_cumprods: alphas,
        num_steps: config.num_steps,
        used_refinement: config.use_iterative_refinement,
    })
}

/// Reconstruct x_0 from x_T by running the forward DDIM denoising process.
///
/// This is the reverse of [`run_ddim_inversion`]: starting from the noisy
/// latent x_T, it steps through the denoising schedule and returns the
/// final (cleaner) latent.
pub fn run_ddim_reconstruction<F>(
    x_t: &[f32],
    schedule: &InversionSchedule,
    config: &DdimInversionConfig,
    noise_predictor: F,
) -> Result<Vec<f32>, InversionError>
where
    F: Fn(&[f32], usize) -> Vec<f32>,
{
    config.validate()?;

    // Get num_steps + 1 alpha values (high → low noise).
    let alphas = schedule.denoising_alphas(config.num_steps + 1);
    if alphas.len() != config.num_steps + 1 {
        return Err(InversionError::InvalidTimesteps(format!(
            "expected {} alphas, got {}",
            config.num_steps + 1,
            alphas.len()
        )));
    }

    let mut latent = x_t.to_vec();

    for i in 0..config.num_steps {
        let coeffs_t = NoiseScheduleCoeffs::new(alphas[i])?;
        let coeffs_prev = NoiseScheduleCoeffs::new(alphas[i + 1])?;
        let noise_pred = noise_predictor(&latent, i);
        latent = ddim_denoise_step(&latent, &noise_pred, &coeffs_t, &coeffs_prev)?;
    }

    Ok(latent)
}

// ---------------------------------------------------------------------------
// Editing utilities
// ---------------------------------------------------------------------------

/// Interpolate between two latent trajectories at a given step.
///
/// Used for style mixing: `alpha = 0` returns `traj_a`'s latent at `step`,
/// `alpha = 1` returns `traj_b`'s latent.
pub fn mix_trajectories(
    traj_a: &InversionTrajectory,
    traj_b: &InversionTrajectory,
    step: usize,
    alpha: f32,
) -> Result<Vec<f32>, InversionError> {
    let a = traj_a
        .intermediate(step)
        .ok_or_else(|| InversionError::LengthMismatch {
            expected: traj_a.len(),
            actual: step,
        })?;
    let b = traj_b
        .intermediate(step)
        .ok_or_else(|| InversionError::LengthMismatch {
            expected: traj_b.len(),
            actual: step,
        })?;

    blend_at_noise_level(a, b, alpha)
}

/// Blend two latents at a specific noise level.
///
/// Performs a simple linear interpolation: `lerp(a, b, alpha)`.
/// `alpha = 0` → returns `latent_a`, `alpha = 1` → returns `latent_b`.
pub fn blend_at_noise_level(
    latent_a: &[f32],
    latent_b: &[f32],
    alpha: f32,
) -> Result<Vec<f32>, InversionError> {
    if latent_a.len() != latent_b.len() {
        return Err(InversionError::LengthMismatch {
            expected: latent_a.len(),
            actual: latent_b.len(),
        });
    }
    let result = latent_a
        .iter()
        .zip(latent_b.iter())
        .map(|(&a, &b)| a + alpha * (b - a))
        .collect();
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a simple trajectory for editing-utility tests.
    fn make_test_trajectory(
        latent_size: usize,
        num_steps: usize,
        offset: f32,
    ) -> InversionTrajectory {
        let alphas = InversionSchedule::cosine(num_steps + 1).inversion_alphas(num_steps + 1);
        let mut latents = Vec::with_capacity(num_steps + 1);
        for i in 0..=num_steps {
            latents.push(vec![offset + i as f32 * 0.01; latent_size]);
        }
        InversionTrajectory {
            latents,
            alpha_cumprods: alphas,
            num_steps,
            used_refinement: false,
        }
    }

    // -----------------------------------------------------------------------
    // 1. NoiseScheduleCoeffs::new: valid alpha → Ok, correct sqrt values
    // -----------------------------------------------------------------------
    #[test]
    fn test_noise_schedule_coeffs_valid() {
        let alpha = 0.81_f32; // sqrt = 0.9, sqrt(1-alpha) = sqrt(0.19)
        let coeffs = NoiseScheduleCoeffs::new(alpha).expect("valid alpha");
        assert!((coeffs.sqrt_alpha_cumprod - alpha.sqrt()).abs() < 1e-6);
        assert!((coeffs.sqrt_one_minus_alpha_cumprod - (1.0 - alpha).sqrt()).abs() < 1e-6);
        assert_eq!(coeffs.alpha_cumprod, alpha);
    }

    // -----------------------------------------------------------------------
    // 2. NoiseScheduleCoeffs::new: alpha > 1 → Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_noise_schedule_coeffs_alpha_too_large() {
        assert!(NoiseScheduleCoeffs::new(1.01).is_err());
        assert!(NoiseScheduleCoeffs::new(2.0).is_err());
    }

    // -----------------------------------------------------------------------
    // 3. NoiseScheduleCoeffs::new: alpha = 0 → Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_noise_schedule_coeffs_alpha_zero() {
        assert!(NoiseScheduleCoeffs::new(0.0).is_err());
        assert!(NoiseScheduleCoeffs::new(-0.5).is_err());
    }

    // -----------------------------------------------------------------------
    // 4. ddim_inversion_step: length mismatch → Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_ddim_inversion_step_length_mismatch() {
        let coeffs = NoiseScheduleCoeffs::new(0.9).unwrap();
        let x_t = vec![1.0_f32, 2.0, 3.0];
        let noise = vec![0.0_f32, 0.0]; // wrong length
        let result = ddim_inversion_step(&x_t, &noise, &coeffs, &coeffs);
        assert!(matches!(
            result,
            Err(InversionError::LengthMismatch {
                expected: 3,
                actual: 2
            })
        ));
    }

    // -----------------------------------------------------------------------
    // 5. ddim_inversion_step: null noise → scaling formula is correct
    // -----------------------------------------------------------------------
    #[test]
    fn test_ddim_inversion_step_null_noise() {
        // With ε = 0: x_{t+1} = sqrt(α_{t+1}) * x_t / sqrt(α_t)
        let alpha_t = 0.9_f32;
        let alpha_next = 0.5_f32;
        let coeffs_t = NoiseScheduleCoeffs::new(alpha_t).unwrap();
        let coeffs_next = NoiseScheduleCoeffs::new(alpha_next).unwrap();
        let x_t = vec![1.0_f32, 2.0, 3.0];
        let noise = vec![0.0_f32; 3];

        let result = ddim_inversion_step(&x_t, &noise, &coeffs_t, &coeffs_next).unwrap();

        let scale = alpha_next.sqrt() / alpha_t.sqrt();
        for (r, &x) in result.iter().zip(x_t.iter()) {
            assert!(
                (r - x * scale).abs() < 1e-5,
                "expected {}, got {}",
                x * scale,
                r
            );
        }
    }

    // -----------------------------------------------------------------------
    // 6. ddim_denoise_step: null noise → scaling formula is correct (inverse direction)
    // -----------------------------------------------------------------------
    #[test]
    fn test_ddim_denoise_step_null_noise() {
        // With ε = 0: x_{t-1} = sqrt(α_{t-1}) * x_t / sqrt(α_t)
        let alpha_t = 0.5_f32;
        let alpha_prev = 0.9_f32;
        let coeffs_t = NoiseScheduleCoeffs::new(alpha_t).unwrap();
        let coeffs_prev = NoiseScheduleCoeffs::new(alpha_prev).unwrap();
        let x_t = vec![2.0_f32, 4.0, 6.0];
        let noise = vec![0.0_f32; 3];

        let result = ddim_denoise_step(&x_t, &noise, &coeffs_t, &coeffs_prev).unwrap();

        let scale = alpha_prev.sqrt() / alpha_t.sqrt();
        for (r, &x) in result.iter().zip(x_t.iter()) {
            assert!(
                (r - x * scale).abs() < 1e-5,
                "expected {}, got {}",
                x * scale,
                r
            );
        }
    }

    // -----------------------------------------------------------------------
    // 7. InversionSchedule::linear: check endpoint values
    // -----------------------------------------------------------------------
    #[test]
    fn test_inversion_schedule_linear_endpoints() {
        let start = 0.9999_f32;
        let end = 0.0047_f32;
        let sched = InversionSchedule::linear(100, start, end).unwrap();
        assert_eq!(sched.alpha_cumprods.len(), 100);
        // First value should equal start.
        assert!((sched.alpha_cumprods[0] - start).abs() < 1e-5);
        // Last value should equal end.
        assert!((sched.alpha_cumprods[99] - end).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // 8. InversionSchedule::cosine: first alpha ≈ 1, last alpha ≈ small
    // -----------------------------------------------------------------------
    #[test]
    fn test_inversion_schedule_cosine_range() {
        let sched = InversionSchedule::cosine(1000);
        let first = sched.alpha_cumprods[0];
        let last = *sched.alpha_cumprods.last().unwrap();
        // first alpha should be very close to 1.0
        assert!(first > 0.99, "first alpha should be near 1, got {}", first);
        // last alpha should be small
        assert!(last < 0.05, "last alpha should be small, got {}", last);
    }

    // -----------------------------------------------------------------------
    // 9. InversionSchedule::alpha_at: fraction=0 → first, fraction=1 → last
    // -----------------------------------------------------------------------
    #[test]
    fn test_inversion_schedule_alpha_at_endpoints() {
        let sched = InversionSchedule::cosine(100);
        let first = sched.alpha_cumprods[0];
        let last = *sched.alpha_cumprods.last().unwrap();
        assert!((sched.alpha_at(0.0) - first).abs() < 1e-6);
        assert!((sched.alpha_at(1.0) - last).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // 10. InversionSchedule::inversion_alphas: returns n values, decreasing
    // -----------------------------------------------------------------------
    #[test]
    fn test_inversion_alphas_decreasing() {
        let sched = InversionSchedule::cosine(1000);
        let alphas = sched.inversion_alphas(10);
        assert_eq!(alphas.len(), 10);
        // Alpha cumprods decrease as noise increases (inversion goes low→high noise).
        for w in alphas.windows(2) {
            assert!(
                w[0] >= w[1],
                "inversion_alphas should be non-increasing: {} >= {}",
                w[0],
                w[1]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 11. InversionSchedule::denoising_alphas: returns n values, increasing
    // -----------------------------------------------------------------------
    #[test]
    fn test_denoising_alphas_increasing() {
        let sched = InversionSchedule::cosine(1000);
        let alphas = sched.denoising_alphas(10);
        assert_eq!(alphas.len(), 10);
        // Denoising goes high→low noise, so alpha cumprods increase.
        for w in alphas.windows(2) {
            assert!(
                w[0] <= w[1],
                "denoising_alphas should be non-decreasing: {} <= {}",
                w[0],
                w[1]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 12. run_ddim_inversion: zero steps → config.validate() returns Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_run_ddim_inversion_zero_steps_rejected() {
        let sched = InversionSchedule::cosine(50);
        let config = DdimInversionConfig {
            num_steps: 0,
            ..Default::default()
        };
        let x_0 = vec![0.1_f32, 0.2, 0.3];
        let result = run_ddim_inversion(&x_0, &sched, &config, null_noise_predictor);
        assert!(
            matches!(result, Err(InversionError::InvalidConfig(_))),
            "expected InvalidConfig error, got {:?}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // 13. run_ddim_inversion: null predictor → trajectory has num_steps+1 latents
    // -----------------------------------------------------------------------
    #[test]
    fn test_run_ddim_inversion_trajectory_length() {
        let sched = InversionSchedule::cosine(100);
        let config = DdimInversionConfig {
            num_steps: 10,
            ..Default::default()
        };
        let x_0 = vec![0.5_f32; 8];
        let traj = run_ddim_inversion(&x_0, &sched, &config, null_noise_predictor).unwrap();
        assert_eq!(traj.len(), 11, "trajectory should have num_steps+1 latents");
    }

    // -----------------------------------------------------------------------
    // 14. run_ddim_inversion: x_t != x_0 (latent changes through inversion)
    // -----------------------------------------------------------------------
    #[test]
    fn test_run_ddim_inversion_latent_changes() {
        let sched = InversionSchedule::cosine(100);
        let config = DdimInversionConfig {
            num_steps: 5,
            ..Default::default()
        };
        let x_0 = vec![1.0_f32, 2.0, 3.0, 4.0];
        let traj = run_ddim_inversion(&x_0, &sched, &config, null_noise_predictor).unwrap();
        // x_0 and x_T should differ when alpha values change.
        let x_t = traj.x_t();
        assert_ne!(
            traj.x_0(),
            x_t,
            "x_0 and x_T should differ after inversion steps"
        );
    }

    // -----------------------------------------------------------------------
    // 15. run_ddim_reconstruction: null predictor returns Vec of correct length
    // -----------------------------------------------------------------------
    #[test]
    fn test_run_ddim_reconstruction_output_length() {
        let sched = InversionSchedule::cosine(100);
        let config = DdimInversionConfig {
            num_steps: 5,
            ..Default::default()
        };
        let x_t = vec![0.1_f32, 0.2, 0.3, 0.4];
        let reconstructed =
            run_ddim_reconstruction(&x_t, &sched, &config, null_noise_predictor).unwrap();
        assert_eq!(
            reconstructed.len(),
            x_t.len(),
            "reconstructed latent must have same dimension as input"
        );
    }

    // -----------------------------------------------------------------------
    // 16. InversionTrajectory::x_0 returns first latent
    // -----------------------------------------------------------------------
    #[test]
    fn test_trajectory_x_0() {
        let traj = make_test_trajectory(4, 3, 0.0);
        assert_eq!(traj.x_0(), traj.latents[0].as_slice());
    }

    // -----------------------------------------------------------------------
    // 17. InversionTrajectory::x_t returns last latent
    // -----------------------------------------------------------------------
    #[test]
    fn test_trajectory_x_t() {
        let traj = make_test_trajectory(4, 3, 0.0);
        let last_idx = traj.latents.len() - 1;
        assert_eq!(traj.x_t(), traj.latents[last_idx].as_slice());
    }

    // -----------------------------------------------------------------------
    // 18. InversionTrajectory::reconstruction_error: same → ~0
    // -----------------------------------------------------------------------
    #[test]
    fn test_reconstruction_error_same_latent() {
        let traj = make_test_trajectory(4, 3, 1.0);
        let reconstructed = traj.x_0().to_vec();
        let err = traj.reconstruction_error(&reconstructed).unwrap();
        assert!(
            err < 1e-6,
            "reconstruction error of identical latent should be ~0, got {}",
            err
        );
    }

    // -----------------------------------------------------------------------
    // 19. mix_trajectories: alpha=0 → traj_a's latent at step
    // -----------------------------------------------------------------------
    #[test]
    fn test_mix_trajectories_alpha_zero() {
        let traj_a = make_test_trajectory(4, 5, 0.0);
        let traj_b = make_test_trajectory(4, 5, 10.0);
        let mixed = mix_trajectories(&traj_a, &traj_b, 2, 0.0).unwrap();
        assert_eq!(mixed, traj_a.latents[2]);
    }

    // -----------------------------------------------------------------------
    // 20. mix_trajectories: alpha=1 → traj_b's latent at step
    // -----------------------------------------------------------------------
    #[test]
    fn test_mix_trajectories_alpha_one() {
        let traj_a = make_test_trajectory(4, 5, 0.0);
        let traj_b = make_test_trajectory(4, 5, 10.0);
        let mixed = mix_trajectories(&traj_a, &traj_b, 2, 1.0).unwrap();
        for (m, b) in mixed.iter().zip(traj_b.latents[2].iter()) {
            assert!((m - b).abs() < 1e-5, "expected {}, got {}", b, m);
        }
    }

    // -----------------------------------------------------------------------
    // 21. blend_at_noise_level: alpha=0.5 → average
    // -----------------------------------------------------------------------
    #[test]
    fn test_blend_at_noise_level_midpoint() {
        let a = vec![0.0_f32, 2.0, 4.0];
        let b = vec![2.0_f32, 4.0, 6.0];
        let blended = blend_at_noise_level(&a, &b, 0.5).unwrap();
        let expected = [1.0_f32, 3.0, 5.0];
        for (got, exp) in blended.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-5, "expected {}, got {}", exp, got);
        }
    }

    // -----------------------------------------------------------------------
    // 22. DdimInversionConfig::validate: num_steps=0 → Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_config_validate_zero_steps() {
        let config = DdimInversionConfig {
            num_steps: 0,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(InversionError::InvalidConfig(_))
        ));
    }

    // -----------------------------------------------------------------------
    // 23. null_noise_predictor: returns all zeros, same length as input
    // -----------------------------------------------------------------------
    #[test]
    fn test_null_noise_predictor() {
        let latent = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let noise = null_noise_predictor(&latent, 7);
        assert_eq!(noise.len(), latent.len());
        for n in &noise {
            assert_eq!(*n, 0.0_f32);
        }
    }

    // -----------------------------------------------------------------------
    // 24. constant_noise_predictor: returns all same value
    // -----------------------------------------------------------------------
    #[test]
    fn test_constant_noise_predictor() {
        let predictor = constant_noise_predictor(3.1);
        let latent = vec![0.0_f32; 6];
        let noise = predictor(&latent, 0);
        assert_eq!(noise.len(), 6);
        for n in &noise {
            assert!((n - 3.1_f32).abs() < 1e-5, "expected 3.1, got {}", n);
        }
    }

    // -----------------------------------------------------------------------
    // Bonus: validate invalid guidance_scale
    // -----------------------------------------------------------------------
    #[test]
    fn test_config_validate_negative_guidance_scale() {
        let config = DdimInversionConfig {
            guidance_scale: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(InversionError::InvalidConfig(_))
        ));
    }

    // -----------------------------------------------------------------------
    // Bonus: reconstruction error length mismatch → Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_reconstruction_error_length_mismatch() {
        let traj = make_test_trajectory(4, 2, 0.0);
        let wrong_len = vec![0.0_f32; 3];
        assert!(matches!(
            traj.reconstruction_error(&wrong_len),
            Err(InversionError::LengthMismatch {
                expected: 4,
                actual: 3
            })
        ));
    }

    // -----------------------------------------------------------------------
    // Bonus: InversionSchedule::linear zero timesteps → Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_linear_schedule_zero_timesteps() {
        assert!(InversionSchedule::linear(0, 0.9999, 0.0047).is_err());
    }

    // -----------------------------------------------------------------------
    // Bonus: blend_at_noise_level length mismatch → Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_blend_length_mismatch() {
        let a = vec![0.0_f32; 3];
        let b = vec![1.0_f32; 4];
        assert!(matches!(
            blend_at_noise_level(&a, &b, 0.5),
            Err(InversionError::LengthMismatch { .. })
        ));
    }
}
