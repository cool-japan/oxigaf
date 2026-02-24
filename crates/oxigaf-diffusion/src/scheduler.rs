//! DDIM scheduler with v-prediction parameterisation.
//!
//! Implements the deterministic DDIM sampling loop used by Stable Diffusion 2.1
//! and the GAF multi-view diffusion model.

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
        for i in 0..num_train_timesteps {
            let beta = beta_start
                + (beta_end - beta_start) * (i as f64) / ((num_train_timesteps - 1) as f64);
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
    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        let step = self.num_train_timesteps / num_inference_steps;
        self.timesteps = (0..num_inference_steps).rev().map(|i| i * step).collect();
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
    pub fn step(&self, model_output: &Tensor, t: usize, sample: &Tensor) -> Result<Tensor> {
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
        (&pred_x0 * sqrt_alpha_prod_prev)? + (&pred_epsilon * sqrt_one_minus_alpha_prod_prev)?
    }

    /// Add noise to latents for a given timestep (forward diffusion process).
    ///
    /// x_t = sqrt(α_t) · x_0 + sqrt(1-α_t) · noise
    pub fn add_noise(&self, original: &Tensor, noise: &Tensor, timestep: usize) -> Result<Tensor> {
        let alpha = self.alphas_cumprod[timestep];
        let sqrt_alpha = alpha.sqrt();
        let sqrt_one_minus_alpha = (1.0 - alpha).sqrt();
        (original * sqrt_alpha)? + (noise * sqrt_one_minus_alpha)?
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
        sched.set_timesteps(50);
        assert_eq!(sched.timesteps().len(), 50);
        // Should be descending
        assert!(sched.timesteps()[0] > sched.timesteps()[49]);
    }
}
