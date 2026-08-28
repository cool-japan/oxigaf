//! DDIM scheduler with v-prediction parameterisation.
//!
//! Implements the deterministic DDIM sampling loop used by Stable Diffusion 2.1
//! and the GAF multi-view diffusion model.

use crate::{DiffusionError, DiffusionResult};
use candle_core::{DType, Device, Result, Tensor};

/// DDIM scheduler state.
#[derive(Debug)]
pub struct DdimScheduler {
    /// Cumulative product of (1 - beta_t).
    alphas_cumprod: Vec<f64>,
    /// Total training timesteps.
    num_train_timesteps: usize,
    /// Inference timesteps (reversed, evenly spaced).
    timesteps: Vec<usize>,
    /// Whether the model predicts v (velocity) rather than noise.
    prediction_type: PredictionType,
}

/// What the model predicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionType {
    /// Model predicts the noise ε.
    Epsilon,
    /// Model predicts v = α_t · ε − σ_t · x_0.
    VPrediction,
}

impl DdimScheduler {
    /// Create a new DDIM scheduler.
    ///
    /// Uses a "scaled linear" beta schedule matching SD 2.1 defaults:
    /// `beta_start=0.00085`, `beta_end=0.012`, 1000 training steps.
    pub fn new(num_train_timesteps: usize, prediction_type: PredictionType) -> Self {
        let beta_start: f64 = 0.00085_f64.sqrt();
        let beta_end: f64 = 0.012_f64.sqrt();

        let mut alphas_cumprod = Vec::with_capacity(num_train_timesteps);
        let mut cumprod = 1.0_f64;
        // `num_train_timesteps == 1` would otherwise divide by
        // `(1 - 1) as f64 == 0.0`, producing `0.0/0.0 = NaN` and poisoning
        // every alpha with NaN (the loop body never runs at all when
        // `num_train_timesteps == 0`, so that case needs no guard). A
        // single-step schedule's only meaningful beta is `beta_start`
        // itself, which this denominator choice produces exactly.
        let denom = num_train_timesteps.saturating_sub(1).max(1) as f64;
        for i in 0..num_train_timesteps {
            let beta = beta_start + (beta_end - beta_start) * (i as f64) / denom;
            let beta = beta * beta; // scaled-linear
            let alpha = 1.0 - beta;
            cumprod *= alpha;
            alphas_cumprod.push(cumprod);
        }

        Self {
            alphas_cumprod,
            num_train_timesteps,
            timesteps: Vec::new(),
            prediction_type,
        }
    }

    /// Configure evenly-spaced timesteps for a given number of inference steps.
    ///
    /// # Errors
    ///
    /// Returns [`DiffusionError::InvalidConfig`] if `num_inference_steps` is
    /// `0` (this used to panic with "attempt to divide by zero") or exceeds
    /// `num_train_timesteps` (this used to silently truncate the integer
    /// division to `step = 0`, emitting a schedule of `num_inference_steps`
    /// identical zero timesteps instead of an error).
    pub fn set_timesteps(&mut self, num_inference_steps: usize) -> DiffusionResult<()> {
        if num_inference_steps == 0 {
            return Err(DiffusionError::InvalidConfig(
                "num_inference_steps must be > 0".to_string(),
            ));
        }
        if num_inference_steps > self.num_train_timesteps {
            return Err(DiffusionError::InvalidConfig(format!(
                "num_inference_steps ({}) must not exceed num_train_timesteps ({})",
                num_inference_steps, self.num_train_timesteps
            )));
        }
        let step = self.num_train_timesteps / num_inference_steps;
        self.timesteps = (0..num_inference_steps).rev().map(|i| i * step).collect();
        Ok(())
    }

    /// Return the current list of timesteps (descending).
    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Perform one DDIM step (deterministic, η=0).
    ///
    /// - `model_output`: the raw network prediction at timestep `t`.
    /// - `t`: current timestep index.
    /// - `sample`: the current noisy latent x_t.
    ///
    /// Returns the denoised latent x_{t-1}.
    ///
    /// # Errors
    ///
    /// - [`DiffusionError::InvalidTimestep`] if `t >= num_train_timesteps`
    ///   (this used to index `alphas_cumprod[t]` unchecked and panic).
    /// - [`DiffusionError::SchedulerNotInitialized`] if [`Self::set_timesteps`]
    ///   was never called (this used to divide by
    ///   `self.timesteps.len() == 0` and panic).
    /// - [`DiffusionError::Candle`] if an underlying tensor operation fails.
    pub fn step(
        &self,
        model_output: &Tensor,
        t: usize,
        sample: &Tensor,
    ) -> DiffusionResult<Tensor> {
        if t >= self.num_train_timesteps {
            return Err(DiffusionError::InvalidTimestep {
                value: t,
                max: self.num_train_timesteps.saturating_sub(1),
            });
        }
        if self.timesteps.is_empty() {
            return Err(DiffusionError::SchedulerNotInitialized);
        }

        let alpha_prod_t = self.alphas_cumprod[t];
        let alpha_prod_t_prev = if t > 0 {
            // find previous timestep
            let step = self.num_train_timesteps / self.timesteps.len();
            if t >= step {
                self.alphas_cumprod[t - step]
            } else {
                1.0
            }
        } else {
            1.0
        };

        let sqrt_alpha_prod = alpha_prod_t.sqrt();
        let sqrt_one_minus_alpha_prod = (1.0 - alpha_prod_t).sqrt();

        // Recover x_0 prediction depending on parameterisation
        let pred_x0 = match self.prediction_type {
            PredictionType::Epsilon => {
                // x_0 = (x_t - sqrt(1-α) * ε) / sqrt(α)
                ((sample - (model_output * sqrt_one_minus_alpha_prod)?)? * (1.0 / sqrt_alpha_prod))?
            }
            PredictionType::VPrediction => {
                // x_0 = sqrt(α) * x_t - sqrt(1-α) * v
                ((sample * sqrt_alpha_prod)? - (model_output * sqrt_one_minus_alpha_prod)?)?
            }
        };

        // Predict noise direction
        let pred_epsilon = match self.prediction_type {
            PredictionType::Epsilon => model_output.clone(),
            PredictionType::VPrediction => {
                ((model_output * sqrt_alpha_prod)? + (sample * sqrt_one_minus_alpha_prod)?)?
            }
        };

        // DDIM deterministic step (η = 0)
        let sqrt_alpha_prod_prev = alpha_prod_t_prev.sqrt();
        let sqrt_one_minus_alpha_prod_prev = (1.0 - alpha_prod_t_prev).sqrt();

        // x_{t-1} = sqrt(α_{t-1}) · x_0 + sqrt(1-α_{t-1}) · ε
        let prev_sample = ((&pred_x0 * sqrt_alpha_prod_prev)?
            + (&pred_epsilon * sqrt_one_minus_alpha_prod_prev)?)?;
        Ok(prev_sample)
    }

    /// Add noise to latents for a given timestep (forward diffusion process).
    ///
    /// x_t = sqrt(α_t) · x_0 + sqrt(1-α_t) · noise
    ///
    /// # Errors
    ///
    /// - [`DiffusionError::InvalidTimestep`] if `timestep >= num_train_timesteps`
    ///   (this used to index `alphas_cumprod[timestep]` unchecked and panic).
    /// - [`DiffusionError::Candle`] if an underlying tensor operation fails.
    pub fn add_noise(
        &self,
        original: &Tensor,
        noise: &Tensor,
        timestep: usize,
    ) -> DiffusionResult<Tensor> {
        if timestep >= self.num_train_timesteps {
            return Err(DiffusionError::InvalidTimestep {
                value: timestep,
                max: self.num_train_timesteps.saturating_sub(1),
            });
        }
        let alpha = self.alphas_cumprod[timestep];
        let sqrt_alpha = alpha.sqrt();
        let sqrt_one_minus_alpha = (1.0 - alpha).sqrt();
        let noisy = ((original * sqrt_alpha)? + (noise * sqrt_one_minus_alpha)?)?;
        Ok(noisy)
    }

    /// Create a tensor of timestep values on the given device.
    pub fn timestep_tensor(&self, t: usize, batch_size: usize, device: &Device) -> Result<Tensor> {
        Tensor::full(t as f32, (batch_size,), device)?.to_dtype(DType::F32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alphas_cumprod_decreasing() {
        let sched = DdimScheduler::new(1000, PredictionType::VPrediction);
        assert!(sched.alphas_cumprod[0] > sched.alphas_cumprod[999]);
        // First alpha should be close to 1
        assert!(sched.alphas_cumprod[0] > 0.99);
        // Last alpha should be small
        assert!(sched.alphas_cumprod[999] < 0.01);
    }

    #[test]
    fn test_set_timesteps() {
        let mut sched = DdimScheduler::new(1000, PredictionType::Epsilon);
        sched.set_timesteps(50).unwrap();
        assert_eq!(sched.timesteps().len(), 50);
        // Should be descending
        assert!(sched.timesteps()[0] > sched.timesteps()[49]);
    }

    /// Regression test: `set_timesteps(0)` used to panic with "attempt to
    /// divide by zero" (`num_train_timesteps / num_inference_steps`).
    #[test]
    fn test_set_timesteps_zero_is_an_error_not_a_panic() {
        let mut sched = DdimScheduler::new(1000, PredictionType::Epsilon);
        let err = sched.set_timesteps(0).unwrap_err();
        assert!(matches!(err, DiffusionError::InvalidConfig(_)));
    }

    /// Regression test: `num_inference_steps > num_train_timesteps` used to
    /// silently truncate the integer division to `step = 0`, producing a
    /// schedule of N identical zero timesteps instead of an error.
    #[test]
    fn test_set_timesteps_exceeding_train_steps_is_an_error() {
        let mut sched = DdimScheduler::new(100, PredictionType::Epsilon);
        let err = sched.set_timesteps(500).unwrap_err();
        assert!(matches!(err, DiffusionError::InvalidConfig(_)));
    }

    /// Regression test: `DdimScheduler::new(1, ..)` used to produce NaN
    /// alphas (the beta interpolation divided by `(1-1) as f64 == 0.0`).
    #[test]
    fn test_new_single_train_timestep_does_not_produce_nan() {
        let sched = DdimScheduler::new(1, PredictionType::Epsilon);
        assert!(
            sched.alphas_cumprod[0].is_finite(),
            "alphas_cumprod[0] must be finite for a 1-step training schedule, got {}",
            sched.alphas_cumprod[0]
        );
    }

    /// Regression test: `step()` used to index `alphas_cumprod[t]`
    /// unchecked and panic for any `t >= num_train_timesteps`.
    #[test]
    fn test_step_out_of_range_timestep_is_an_error_not_a_panic() {
        let mut sched = DdimScheduler::new(100, PredictionType::Epsilon);
        sched.set_timesteps(10).unwrap();
        let device = Device::Cpu;
        let sample = Tensor::zeros((1, 4, 2, 2), DType::F32, &device).unwrap();
        let model_output = sample.clone();
        let err = sched.step(&model_output, 100, &sample).unwrap_err();
        assert!(matches!(
            err,
            DiffusionError::InvalidTimestep {
                value: 100,
                max: 99
            }
        ));
    }

    /// Regression test: `step()` used to divide by
    /// `self.timesteps.len() == 0` and panic when called before
    /// `set_timesteps`.
    #[test]
    fn test_step_before_set_timesteps_is_an_error_not_a_panic() {
        let sched = DdimScheduler::new(100, PredictionType::Epsilon);
        let device = Device::Cpu;
        let sample = Tensor::zeros((1, 4, 2, 2), DType::F32, &device).unwrap();
        let model_output = sample.clone();
        let err = sched.step(&model_output, 50, &sample).unwrap_err();
        assert!(matches!(err, DiffusionError::SchedulerNotInitialized));
    }

    /// Regression test: `add_noise()` used to index
    /// `alphas_cumprod[timestep]` unchecked and panic for any
    /// `timestep >= num_train_timesteps`.
    #[test]
    fn test_add_noise_out_of_range_timestep_is_an_error_not_a_panic() {
        let sched = DdimScheduler::new(100, PredictionType::Epsilon);
        let device = Device::Cpu;
        let original = Tensor::zeros((1, 4, 2, 2), DType::F32, &device).unwrap();
        let noise = original.clone();
        let err = sched.add_noise(&original, &noise, 100).unwrap_err();
        assert!(matches!(
            err,
            DiffusionError::InvalidTimestep {
                value: 100,
                max: 99
            }
        ));
    }
}
