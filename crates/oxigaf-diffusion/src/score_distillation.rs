//! Score Distillation Sampling (SDS) and Variational Score Distillation (VSD)
//! for distilling 3D Gaussian fields from 2D diffusion model priors.
//!
//! ## References
//! - DreamFusion (Poole et al. 2022): <https://dreamfusion3d.github.io/>
//! - Fantasia3D (Chen et al. 2023): <https://fantasia3d.github.io/>
//! - VSD / ProlificDreamer (Wang et al. 2024): <https://ml.cs.tsinghua.edu.cn/prolificdreamer/>
//!
//! ## SDS Overview
//!
//! Given a differentiable renderer producing latent codes `x = g(θ)` from 3D
//! Gaussian parameters `θ`, SDS provides a gradient signal:
//!
//! ```text
//! ∇_θ L_SDS = E_{t,ε} [ w(t) (ε_θ(x_t, t, y) − ε) ∂x/∂θ ] / σ_t
//! ```
//!
//! where `ε_θ` is the diffusion model's noise prediction conditioned on text `y`,
//! `ε` is the actual noise used to create `x_t`, and `w(t)` is a weighting function.
//!
//! ## VSD Overview
//!
//! VSD replaces the naive target distribution `p(x)` with a variational approximation
//! `q_φ(x)` maintained by a LoRA-adapted score model, reducing gradient variance:
//!
//! ```text
//! ∇_θ L_VSD = E_{t,ε} [ w(t) (ε_θ_frozen(x_t, t, y) − ε_θ_lora(x_t, t)) ∂x/∂θ ] / σ_t
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during score distillation operations.
#[derive(Debug, Error)]
pub enum ScoreDistillationError {
    /// Configuration is semantically invalid.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// Array dimensions do not agree.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Timestep index falls outside the valid range.
    #[error("Invalid timestep: t={t}, max={max_t}")]
    InvalidTimestep { t: usize, max_t: usize },

    /// An operation was attempted on an empty latent vector.
    #[error("Empty latent")]
    EmptyLatent,

    /// A floating-point computation produced a non-finite value.
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// A noise schedule is required but has not been provided.
    #[error("No noise schedule")]
    NoNoiseSchedule,

    /// Propagated from the LoRA adapter machinery backing [`VsdState`].
    #[error("LoRA error: {0}")]
    Lora(#[from] crate::lora_adapter::LoraError),
}

// ---------------------------------------------------------------------------
// Weighting schemes
// ---------------------------------------------------------------------------

/// Weighting function `w(t)` applied to the SDS gradient.
///
/// The choice of weighting affects gradient magnitude and training stability.
#[derive(Debug, Clone, PartialEq)]
pub enum ScoreWeighting {
    /// Constant weight — `w(t) = w`.
    Fixed(f32),

    /// SNR-based weight — `w(t) = σ_t²`.
    SnrWeighted,

    /// Clamped SNR weight — `w(t) = min(SNR(t), 1.0)` where
    /// `SNR(t) = ᾱ_t / σ_t²`.
    MaxSnrWeighted,

    /// Timestep decay — `w(t) = t^{-decay}` (t is 1-based to avoid zero
    /// division).
    TimestepDecay { decay: f32 },
}

// ---------------------------------------------------------------------------
// SDS configuration
// ---------------------------------------------------------------------------

/// Configuration for Score Distillation Sampling.
#[derive(Debug, Clone)]
pub struct SdsConfig {
    /// Minimum timestep ratio in `[0, 1]` (default `0.02`).
    pub min_step_percent: f32,
    /// Maximum timestep ratio in `[0, 1]` (default `0.98`).
    pub max_step_percent: f32,
    /// Classifier-Free Guidance scale (default `7.5`).
    pub guidance_scale: f32,
    /// Gradient weighting scheme.
    pub weighting: ScoreWeighting,
    /// Total number of diffusion timesteps (default `1000`).
    pub num_timesteps: usize,
}

impl Default for SdsConfig {
    fn default() -> Self {
        Self {
            min_step_percent: 0.02,
            max_step_percent: 0.98,
            guidance_scale: 7.5,
            weighting: ScoreWeighting::MaxSnrWeighted,
            num_timesteps: 1000,
        }
    }
}

impl SdsConfig {
    /// Validate that configuration values are logically consistent.
    ///
    /// Returns [`ScoreDistillationError::InvalidConfig`] if any field is out of
    /// range.
    pub fn validate(&self) -> Result<(), ScoreDistillationError> {
        if self.min_step_percent >= self.max_step_percent {
            return Err(ScoreDistillationError::InvalidConfig(format!(
                "min_step_percent ({}) must be less than max_step_percent ({})",
                self.min_step_percent, self.max_step_percent
            )));
        }
        if !(0.0..=1.0).contains(&self.min_step_percent) {
            return Err(ScoreDistillationError::InvalidConfig(format!(
                "min_step_percent ({}) must be in [0, 1]",
                self.min_step_percent
            )));
        }
        if !(0.0..=1.0).contains(&self.max_step_percent) {
            return Err(ScoreDistillationError::InvalidConfig(format!(
                "max_step_percent ({}) must be in [0, 1]",
                self.max_step_percent
            )));
        }
        if self.guidance_scale <= 0.0 {
            return Err(ScoreDistillationError::InvalidConfig(format!(
                "guidance_scale ({}) must be positive",
                self.guidance_scale
            )));
        }
        if self.num_timesteps == 0 {
            return Err(ScoreDistillationError::InvalidConfig(
                "num_timesteps must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// DreamFusion paper defaults (Poole et al. 2022).
    pub fn dreamfusion() -> Self {
        Self {
            min_step_percent: 0.02,
            max_step_percent: 0.98,
            guidance_scale: 100.0,
            weighting: ScoreWeighting::Fixed(1.0),
            num_timesteps: 1000,
        }
    }

    /// Fantasia3D defaults — caps max timestep at 0.5 for geometry quality.
    pub fn fantasia3d() -> Self {
        Self {
            min_step_percent: 0.02,
            max_step_percent: 0.5,
            guidance_scale: 7.5,
            weighting: ScoreWeighting::MaxSnrWeighted,
            num_timesteps: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// VSD configuration
// ---------------------------------------------------------------------------

/// Configuration for Variational Score Distillation (ProlificDreamer).
///
/// `num_particles`, `lora_rank` and `vsd_lr` are consumed by [`VsdState`],
/// not by [`vsd_gradient`] itself: `vsd_gradient` is a stateless function
/// over an already-computed `lora_noise_pred`, so it has no particle
/// ensemble or optimiser state of its own to configure. [`VsdState`] is the
/// companion that actually owns one LoRA adapter per particle and applies
/// the `vsd_lr` SGD update — construct it from this config when you need
/// real VSD machinery rather than supplying `lora_noise_pred` from your own
/// external adaptation.
#[derive(Debug, Clone)]
pub struct VsdConfig {
    /// Underlying SDS configuration.
    pub sds_config: SdsConfig,
    /// Number of 3D field particles in the variational distribution (default
    /// `1`). Consumed by [`VsdState::new`], which allocates one LoRA
    /// adapter per particle.
    pub num_particles: usize,
    /// LoRA rank for the score model adaptation (default `4`). Consumed by
    /// [`VsdState::new`] when constructing each particle's adapter.
    pub lora_rank: usize,
    /// Learning rate for LoRA adaptation (default `1e-4`). Consumed by
    /// [`VsdState::sgd_update`] as the SGD step size.
    pub vsd_lr: f32,
}

impl Default for VsdConfig {
    fn default() -> Self {
        Self {
            sds_config: SdsConfig::default(),
            num_particles: 1,
            lora_rank: 4,
            vsd_lr: 1e-4,
        }
    }
}

impl VsdConfig {
    /// Validate the VSD configuration.
    pub fn validate(&self) -> Result<(), ScoreDistillationError> {
        self.sds_config.validate()?;
        if self.num_particles == 0 {
            return Err(ScoreDistillationError::InvalidConfig(
                "num_particles must be > 0".to_string(),
            ));
        }
        if self.lora_rank == 0 {
            return Err(ScoreDistillationError::InvalidConfig(
                "lora_rank must be > 0".to_string(),
            ));
        }
        if self.vsd_lr <= 0.0 {
            return Err(ScoreDistillationError::InvalidConfig(format!(
                "vsd_lr ({}) must be positive",
                self.vsd_lr
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VSD particle / LoRA state
// ---------------------------------------------------------------------------

/// Owns the per-particle LoRA adapters that back [`VsdConfig`]'s
/// `num_particles`, `lora_rank` and `vsd_lr` fields, making them affect real
/// behaviour rather than sitting validated-but-unread.
///
/// [`vsd_gradient`] is a stateless function over an already-computed
/// `lora_noise_pred` — the model forward pass, with a particle's LoRA delta
/// merged in, happens in the caller's U-Net, which this crate-internal
/// module does not own. `VsdState` is the companion piece: it holds one
/// [`LoraAdapter`](crate::lora_adapter::LoraAdapter) per particle (rank
/// `lora_rank`) and applies the SGD update at `vsd_lr` once the caller has
/// backpropagated the VSD gradient through the LoRA layer it applied.
///
/// # Typical usage
///
/// 1. [`VsdState::new`] — allocates `num_particles` empty adapters.
/// 2. For each particle, add the
///    [`LoraLayer`](crate::lora_adapter::LoraLayer)s it needs (matching
///    whichever U-Net projections the caller adapts) via [`Self::add_layer`].
/// 3. Per training step and per particle: apply that particle's adapter
///    (e.g. via [`LoraLayer::apply`](crate::lora_adapter::LoraLayer::apply)
///    or by merging it into the base weights), run the frozen and
///    LoRA-adapted forward passes, call [`vsd_gradient`], backprop the
///    result through the adapted layer with
///    [`lora_backward`](crate::lora_adapter::lora_backward), and apply the
///    resulting gradients with [`Self::sgd_update`].
#[derive(Debug, Clone)]
pub struct VsdState {
    /// One LoRA adapter per particle; `particles.len() == config.num_particles`.
    pub particles: Vec<crate::lora_adapter::LoraAdapter>,
    /// The configuration this state was built from. `lora_rank` shaped every
    /// particle's adapter at construction; `vsd_lr` is read by
    /// [`Self::sgd_update`].
    pub config: VsdConfig,
}

impl VsdState {
    /// Allocate `config.num_particles` empty LoRA adapters, each at rank
    /// `config.lora_rank`.
    ///
    /// # Errors
    ///
    /// Propagates [`VsdConfig::validate`].
    pub fn new(config: &VsdConfig) -> Result<Self, ScoreDistillationError> {
        config.validate()?;
        let lora_config = crate::lora_adapter::LoraConfig::with_rank(config.lora_rank);
        let particles = (0..config.num_particles)
            .map(|_| crate::lora_adapter::LoraAdapter::new(lora_config.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            particles,
            config: config.clone(),
        })
    }

    /// Number of particles (`== config.num_particles` at construction time).
    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }

    /// Add a named LoRA layer (rank `config.lora_rank`, scaling `alpha =
    /// lora_rank`) to one particle's adapter.
    ///
    /// # Errors
    ///
    /// [`ScoreDistillationError::InvalidConfig`] if `particle_idx` is out of
    /// range.
    pub fn add_layer(
        &mut self,
        particle_idx: usize,
        name: impl Into<String>,
        in_dim: usize,
        out_dim: usize,
        seed: u64,
    ) -> Result<(), ScoreDistillationError> {
        let rank = self.config.lora_rank;
        let particle = self.particles.get_mut(particle_idx).ok_or_else(|| {
            ScoreDistillationError::InvalidConfig(format!(
                "particle_idx {particle_idx} out of range (have {} particles)",
                self.config.num_particles
            ))
        })?;
        let layer =
            crate::lora_adapter::LoraLayer::new(name, in_dim, out_dim, rank, rank as f32, seed)?;
        particle.add_layer(layer);
        Ok(())
    }

    /// Apply one SGD update (`param -= config.vsd_lr * grad`) to a named
    /// layer within one particle, from gradients already computed via
    /// [`lora_backward`](crate::lora_adapter::lora_backward).
    ///
    /// # Errors
    ///
    /// [`ScoreDistillationError::InvalidConfig`] if `particle_idx` is out of
    /// range or the named layer does not exist on that particle;
    /// [`ScoreDistillationError::Lora`] on a gradient shape mismatch.
    pub fn sgd_update(
        &mut self,
        particle_idx: usize,
        layer_name: &str,
        grad_a: &[f32],
        grad_b: &[f32],
    ) -> Result<(), ScoreDistillationError> {
        let lr = self.config.vsd_lr;
        let num_particles = self.particles.len();
        let particle = self.particles.get_mut(particle_idx).ok_or_else(|| {
            ScoreDistillationError::InvalidConfig(format!(
                "particle_idx {particle_idx} out of range (have {num_particles} particles)"
            ))
        })?;
        let layer = particle.get_layer_mut(layer_name).ok_or_else(|| {
            ScoreDistillationError::InvalidConfig(format!(
                "layer '{layer_name}' not found on particle {particle_idx}"
            ))
        })?;
        crate::lora_adapter::lora_sgd_step(layer, grad_a, grad_b, lr)?;
        particle.step += 1;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Noise schedule
// ---------------------------------------------------------------------------

/// Pre-computed noise schedule coefficients for SDS.
///
/// The schedule maps every timestep `t ∈ [0, num_timesteps)` to:
/// - `ᾱ_t` (alpha_bar / `alphas_cumprod[t]`): signal retention factor
/// - `σ_t = √(1 − ᾱ_t)` (`sigmas[t]`): noise level
#[derive(Debug, Clone)]
pub struct SdsNoiseSchedule {
    /// Total number of timesteps.
    pub num_timesteps: usize,
    /// Cumulative product of `(1 − β_t)` for each timestep.
    pub alphas_cumprod: Vec<f32>,
    /// `√(1 − ᾱ_t)` for each timestep.
    pub sigmas: Vec<f32>,
}

impl SdsNoiseSchedule {
    /// Build a "scaled linear" beta schedule with SD-style endpoints.
    ///
    /// Betas are `linspace(sqrt(β_start), sqrt(β_end), T)^2` with
    /// `β_start = 0.00085` and `β_end = 0.012`, matching the Stable Diffusion
    /// 1.x/2.x convention (this is the same schedule [`crate::scheduler`]
    /// and [`crate::noise_schedule_analysis::NoiseSchedule::scaled_linear`]
    /// use; a *plain* linear ramp between the same endpoints is a materially
    /// different schedule — its mean beta differs by roughly 20%, shifting
    /// every SNR/weight derived from it).
    pub fn linear(num_timesteps: usize) -> Self {
        let beta_start: f64 = 0.00085;
        let beta_end: f64 = 0.012;
        let sqrt_beta_start = beta_start.sqrt();
        let sqrt_beta_end = beta_end.sqrt();
        let t = num_timesteps.max(1);

        let mut alphas_cumprod = Vec::with_capacity(t);
        let mut running: f64 = 1.0;
        for i in 0..t {
            let frac = i as f64 / (t - 1).max(1) as f64;
            let sqrt_beta = sqrt_beta_start + frac * (sqrt_beta_end - sqrt_beta_start);
            let beta = sqrt_beta * sqrt_beta;
            running *= 1.0 - beta;
            alphas_cumprod.push(running.clamp(0.0, 1.0) as f32);
        }

        let sigmas = alphas_cumprod
            .iter()
            .map(|&a| (1.0 - a).max(0.0).sqrt())
            .collect();

        Self {
            num_timesteps: t,
            alphas_cumprod,
            sigmas,
        }
    }

    /// Build a cosine beta schedule (Nichol & Dhariwal 2021).
    ///
    /// Uses the formulation `ᾱ_t = f(t) / f(0)` where
    /// `f(t) = cos²(((t/T + s) / (1 + s)) · π/2)`, `s = 0.008`.
    pub fn cosine(num_timesteps: usize) -> Self {
        let t = num_timesteps.max(1);
        let s: f64 = 0.008;
        let pi_half = std::f64::consts::PI / 2.0;

        let f = |step: f64| {
            let ratio = (step / t as f64 + s) / (1.0 + s);
            (ratio * pi_half).cos().powi(2)
        };
        let f0 = f(0.0);

        let mut alphas_cumprod = Vec::with_capacity(t);
        for i in 0..t {
            // Evaluate at the *right* endpoint of step i's interval
            // (i.e. f(i+1)), not its midpoint: this reaches ᾱ = 0 exactly at
            // i = T-1, which the left-endpoint form f(i) does not. Note
            // `noise_schedule_analysis::cosine` uses the f(i) convention
            // instead — the two are deliberately different conventions, not
            // a shared implementation.
            let raw = (f(i as f64 + 1.0) / f0).clamp(0.0, 1.0) as f32;
            alphas_cumprod.push(raw);
        }

        let sigmas = alphas_cumprod
            .iter()
            .map(|&a| (1.0 - a).max(0.0).sqrt())
            .collect();

        Self {
            num_timesteps: t,
            alphas_cumprod,
            sigmas,
        }
    }

    /// Return `σ_t = √(1 − ᾱ_t)` at timestep `t`.
    pub fn get_sigma(&self, t: usize) -> Result<f32, ScoreDistillationError> {
        if t >= self.num_timesteps {
            return Err(ScoreDistillationError::InvalidTimestep {
                t,
                max_t: self.num_timesteps - 1,
            });
        }
        Ok(self.sigmas[t])
    }

    /// Return `ᾱ_t` at timestep `t`.
    pub fn get_alpha(&self, t: usize) -> Result<f32, ScoreDistillationError> {
        if t >= self.num_timesteps {
            return Err(ScoreDistillationError::InvalidTimestep {
                t,
                max_t: self.num_timesteps - 1,
            });
        }
        Ok(self.alphas_cumprod[t])
    }
}

// ---------------------------------------------------------------------------
// xorshift64 PRNG (inline, no external dependency)
// ---------------------------------------------------------------------------

/// Advance an xorshift64 state by one step and return the new state.
///
/// The seed guard `state = state.max(1)` ensures the state is never zero, which
/// would make xorshift produce an infinite stream of zeros.
#[inline]
fn xorshift64(state: u64) -> u64 {
    let mut s = state.max(1);
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    s
}

// ---------------------------------------------------------------------------
// Core sampling functions
// ---------------------------------------------------------------------------

/// Sample a random timestep uniformly from the discrete range
/// `[t_min, t_max]` using a single xorshift64 step from `seed`.
///
/// # Arguments
/// - `min_percent`: lower bound as a fraction of `num_timesteps` in `[0, 1]`.
/// - `max_percent`: upper bound as a fraction of `num_timesteps` in `[0, 1]`.
/// - `num_timesteps`: total number of timesteps in the schedule.
/// - `seed`: xorshift64 seed (must be > 0; the seed guard is applied
///   internally).
///
/// # Returns
/// A timestep index in `[t_min, t_max]`.
pub fn sample_timestep(
    min_percent: f32,
    max_percent: f32,
    num_timesteps: usize,
    seed: u64,
) -> usize {
    let t_min =
        ((min_percent * num_timesteps as f32) as usize).min(num_timesteps.saturating_sub(1));
    let t_max = ((max_percent * num_timesteps as f32) as usize)
        .min(num_timesteps.saturating_sub(1))
        .max(t_min);

    if t_min == t_max {
        return t_min;
    }

    let state = xorshift64(seed);
    let range = (t_max - t_min + 1) as u64;
    let offset = (state % range) as usize;
    t_min + offset
}

/// Add noise to a latent at diffusion timestep `t`:
///
/// `x_t = √ᾱ_t · x₀ + σ_t · ε`
///
/// # Arguments
/// - `latent`: clean latent `x₀` with shape `[D]`.
/// - `t`: diffusion timestep index.
/// - `schedule`: pre-computed noise schedule.
/// - `noise`: Gaussian noise `ε` with the same shape as `latent`.
///
/// # Errors
/// - [`ScoreDistillationError::EmptyLatent`] if `latent` is empty.
/// - [`ScoreDistillationError::DimensionMismatch`] if `noise.len() !=
///   latent.len()`.
/// - [`ScoreDistillationError::InvalidTimestep`] if `t` is out of range.
pub fn add_sds_noise(
    latent: &[f32],
    t: usize,
    schedule: &SdsNoiseSchedule,
    noise: &[f32],
) -> Result<Vec<f32>, ScoreDistillationError> {
    if latent.is_empty() {
        return Err(ScoreDistillationError::EmptyLatent);
    }
    if noise.len() != latent.len() {
        return Err(ScoreDistillationError::DimensionMismatch {
            expected: latent.len(),
            actual: noise.len(),
        });
    }
    let alpha = schedule.get_alpha(t)?;
    let sigma = schedule.get_sigma(t)?;
    let sqrt_alpha = alpha.sqrt();

    let noisy: Vec<f32> = latent
        .iter()
        .zip(noise.iter())
        .map(|(&x, &e)| sqrt_alpha * x + sigma * e)
        .collect();

    Ok(noisy)
}

/// Compute the classifier-free guidance noise prediction:
///
/// `ε_cfg = ε_uncond + scale · (ε_cond − ε_uncond)`
///
/// When `guidance_scale = 0` the result equals `ε_uncond`; when
/// `guidance_scale = 1` the result equals `ε_cond`.
///
/// # Errors
/// - [`ScoreDistillationError::EmptyLatent`] if inputs are empty.
/// - [`ScoreDistillationError::DimensionMismatch`] on shape disagreement.
pub fn classifier_free_guidance(
    noise_pred_cond: &[f32],
    noise_pred_uncond: &[f32],
    guidance_scale: f32,
) -> Result<Vec<f32>, ScoreDistillationError> {
    if noise_pred_cond.is_empty() {
        return Err(ScoreDistillationError::EmptyLatent);
    }
    if noise_pred_uncond.len() != noise_pred_cond.len() {
        return Err(ScoreDistillationError::DimensionMismatch {
            expected: noise_pred_cond.len(),
            actual: noise_pred_uncond.len(),
        });
    }
    let out = noise_pred_uncond
        .iter()
        .zip(noise_pred_cond.iter())
        .map(|(&u, &c)| u + guidance_scale * (c - u))
        .collect();
    Ok(out)
}

/// Compute the SDS gradient in latent space:
///
/// `∇L_SDS = w(t) · (ε_cfg − ε) / σ_t`
///
/// The stop-gradient on `x_t` is implicit — this function computes the gradient
/// value only, not backpropagation through the renderer.  CFG is applied
/// internally using `config.guidance_scale`.
///
/// # Arguments
/// - `noise_pred_cond`: conditional noise prediction `ε_θ(x_t, t, y=text)`.
/// - `noise_pred_uncond`: unconditional noise prediction `ε_θ(x_t, t, y=∅)`.
/// - `noise`: actual noise `ε` that was added to create `x_t`.
/// - `t`: diffusion timestep index.
/// - `schedule`: noise schedule for `σ_t` and weighting.
/// - `config`: SDS configuration.
///
/// # Errors
/// Returns an error if inputs are mismatched, `t` is invalid, or the schedule
/// has a zero sigma at `t`.
pub fn sds_gradient(
    noise_pred_cond: &[f32],
    noise_pred_uncond: &[f32],
    noise: &[f32],
    t: usize,
    schedule: &SdsNoiseSchedule,
    config: &SdsConfig,
) -> Result<Vec<f32>, ScoreDistillationError> {
    if noise_pred_cond.is_empty() {
        return Err(ScoreDistillationError::EmptyLatent);
    }
    if noise_pred_uncond.len() != noise_pred_cond.len() {
        return Err(ScoreDistillationError::DimensionMismatch {
            expected: noise_pred_cond.len(),
            actual: noise_pred_uncond.len(),
        });
    }
    if noise.len() != noise_pred_cond.len() {
        return Err(ScoreDistillationError::DimensionMismatch {
            expected: noise_pred_cond.len(),
            actual: noise.len(),
        });
    }

    let sigma = schedule.get_sigma(t)?;
    if sigma == 0.0 {
        return Err(ScoreDistillationError::NumericalError(format!(
            "sigma at t={t} is zero — cannot divide"
        )));
    }

    let weight = compute_weight(t, schedule, &config.weighting)?;
    let scale = config.guidance_scale;

    // Apply CFG inline, then compute (eps_cfg - eps) / sigma, scaled by w(t).
    let grad = noise_pred_uncond
        .iter()
        .zip(noise_pred_cond.iter())
        .zip(noise.iter())
        .map(|((&u, &c), &e)| {
            let eps_cfg = u + scale * (c - u);
            weight * (eps_cfg - e) / sigma
        })
        .collect();

    Ok(grad)
}

/// Compute the scalar SDS loss for logging purposes:
///
/// `L_SDS = 0.5 · ‖ε_θ − ε‖²`
///
/// This is not backpropagated; it is intended for monitoring training progress.
///
/// # Errors
/// - [`ScoreDistillationError::EmptyLatent`] if inputs are empty.
/// - [`ScoreDistillationError::DimensionMismatch`] on shape disagreement.
pub fn sds_loss(noise_pred: &[f32], noise: &[f32]) -> Result<f32, ScoreDistillationError> {
    if noise_pred.is_empty() {
        return Err(ScoreDistillationError::EmptyLatent);
    }
    if noise.len() != noise_pred.len() {
        return Err(ScoreDistillationError::DimensionMismatch {
            expected: noise_pred.len(),
            actual: noise.len(),
        });
    }
    let sum_sq: f32 = noise_pred
        .iter()
        .zip(noise.iter())
        .map(|(&p, &e)| (p - e).powi(2))
        .sum();
    Ok(0.5 * sum_sq)
}

/// Compute the weighting factor `w(t)` for the SDS gradient.
///
/// # Errors
/// - [`ScoreDistillationError::InvalidTimestep`] if `t >= schedule.num_timesteps`.
/// - [`ScoreDistillationError::NumericalError`] if a zero-sigma is encountered.
pub fn compute_weight(
    t: usize,
    schedule: &SdsNoiseSchedule,
    weighting: &ScoreWeighting,
) -> Result<f32, ScoreDistillationError> {
    match weighting {
        ScoreWeighting::Fixed(w) => Ok(*w),
        ScoreWeighting::SnrWeighted => {
            let sigma = schedule.get_sigma(t)?;
            Ok(sigma * sigma)
        }
        ScoreWeighting::MaxSnrWeighted => {
            let alpha = schedule.get_alpha(t)?;
            let sigma = schedule.get_sigma(t)?;
            if sigma == 0.0 {
                return Err(ScoreDistillationError::NumericalError(
                    "sigma is zero — SNR undefined".to_string(),
                ));
            }
            // SNR = ᾱ / σ²; weight = min(SNR, 1.0)
            let snr = alpha / (sigma * sigma);
            Ok(snr.min(1.0))
        }
        ScoreWeighting::TimestepDecay { decay } => {
            // Use 1-based to avoid zero division.
            let t_f = (t + 1) as f32;
            Ok(t_f.powf(-decay))
        }
    }
}

/// Compute the VSD gradient using a LoRA-adapted reference model.
///
/// `∇L_VSD = w(t) · (ε_θ_frozen(x_t, t, y) − ε_θ_lora(x_t, t)) / σ_t`
///
/// This simplified version treats the LoRA noise prediction as the variational
/// score target, reducing variance compared to raw SDS.
///
/// This function only combines two already-computed noise predictions — it
/// does not run the LoRA-adapted model itself or update its weights. Use
/// [`VsdState`] to own the per-particle LoRA adapters (`config.num_particles`
/// of them, at rank `config.lora_rank`) that produce `lora_noise_pred`, and
/// [`VsdState::sgd_update`] (at `config.vsd_lr`) after backpropagating this
/// gradient through the adapted layer.
///
/// # Arguments
/// - `ref_noise_pred`: noise prediction from the frozen (text-conditioned) model.
/// - `lora_noise_pred`: noise prediction from the LoRA-adapted score model.
/// - `t`: diffusion timestep index.
/// - `schedule`: noise schedule.
/// - `config`: VSD configuration.
///
/// # Errors
/// Same categories as [`sds_gradient`].
pub fn vsd_gradient(
    ref_noise_pred: &[f32],
    lora_noise_pred: &[f32],
    t: usize,
    schedule: &SdsNoiseSchedule,
    config: &VsdConfig,
) -> Result<Vec<f32>, ScoreDistillationError> {
    if ref_noise_pred.is_empty() {
        return Err(ScoreDistillationError::EmptyLatent);
    }
    if lora_noise_pred.len() != ref_noise_pred.len() {
        return Err(ScoreDistillationError::DimensionMismatch {
            expected: ref_noise_pred.len(),
            actual: lora_noise_pred.len(),
        });
    }

    let sigma = schedule.get_sigma(t)?;
    if sigma == 0.0 {
        return Err(ScoreDistillationError::NumericalError(format!(
            "sigma at t={t} is zero — cannot divide"
        )));
    }

    let weight = compute_weight(t, schedule, &config.sds_config.weighting)?;

    let grad = ref_noise_pred
        .iter()
        .zip(lora_noise_pred.iter())
        .map(|(&r, &l)| weight * (r - l) / sigma)
        .collect();

    Ok(grad)
}

/// Back-propagate the SDS gradient from latent space into 3DGS parameter space.
///
/// Computes `Jᵀ · g` where `J` is the `latent_dim × param_dim` render Jacobian
/// (row-major) and `g` is the SDS gradient vector of length `latent_dim`.
///
/// The result is a vector of length `param_dim` — the gradient of the SDS loss
/// w.r.t. the 3D Gaussian parameters.
///
/// # Errors
/// - [`ScoreDistillationError::DimensionMismatch`] if `render_jacobian.len() !=
///   latent_dim * param_dim` or `sds_grad.len() != latent_dim`.
pub fn backprop_sds_to_gaussians(
    render_jacobian: &[f32],
    sds_grad: &[f32],
    latent_dim: usize,
    param_dim: usize,
) -> Result<Vec<f32>, ScoreDistillationError> {
    let expected_jac_len = latent_dim * param_dim;
    if render_jacobian.len() != expected_jac_len {
        return Err(ScoreDistillationError::DimensionMismatch {
            expected: expected_jac_len,
            actual: render_jacobian.len(),
        });
    }
    if sds_grad.len() != latent_dim {
        return Err(ScoreDistillationError::DimensionMismatch {
            expected: latent_dim,
            actual: sds_grad.len(),
        });
    }

    // J^T * g: result[j] = sum_{i=0}^{latent_dim-1} J[i, j] * g[i]
    //          J[i, j]   = render_jacobian[i * param_dim + j]
    let mut result = vec![0.0f32; param_dim];
    for (i, &g_i) in sds_grad.iter().enumerate().take(latent_dim) {
        if g_i == 0.0 {
            continue;
        }
        let row_offset = i * param_dim;
        for j in 0..param_dim {
            result[j] += render_jacobian[row_offset + j] * g_i;
        }
    }

    Ok(result)
}

/// Compute an annealed timestep range for coarse-to-fine SDS training.
///
/// Returns `(current_min_percent, current_max_percent)` where the maximum
/// decreases linearly from `initial_max_percent` to `final_max_percent` over
/// `total_steps` training iterations.
///
/// At `step = 0` the max equals `initial_max_percent`; at `step = total_steps`
/// it equals `final_max_percent`.  The minimum is held constant.
pub fn annealed_timestep_range(
    step: usize,
    total_steps: usize,
    initial_max_percent: f32,
    final_max_percent: f32,
    min_percent: f32,
) -> (f32, f32) {
    if total_steps == 0 {
        return (min_percent, initial_max_percent);
    }
    let frac = (step as f32 / total_steps as f32).clamp(0.0, 1.0);
    let current_max = initial_max_percent + frac * (final_max_percent - initial_max_percent);
    let current_max = current_max.max(min_percent);
    (min_percent, current_max)
}

// ---------------------------------------------------------------------------
// Gradient utilities
// ---------------------------------------------------------------------------

/// Compute the L2 norm of an SDS gradient vector.
pub fn sds_grad_norm(grad: &[f32]) -> f32 {
    grad.iter().map(|&g| g * g).sum::<f32>().sqrt()
}

/// Clip an SDS gradient by its L2 norm.
///
/// If `‖grad‖ > max_norm`, scales every element so that `‖grad‖ = max_norm`.
/// Otherwise the gradient is left unchanged.
pub fn clip_sds_gradient(grad: &mut [f32], max_norm: f32) {
    let norm = sds_grad_norm(grad);
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for g in grad.iter_mut() {
            *g *= scale;
        }
    }
}

// ---------------------------------------------------------------------------
// Step statistics
// ---------------------------------------------------------------------------

/// Diagnostics computed for a single SDS update step.
#[derive(Debug, Clone)]
pub struct SdsStepStats {
    /// Diffusion timestep used for this step.
    pub timestep: usize,
    /// Weighting factor `w(t)` applied to the gradient.
    pub weight: f32,
    /// Scalar SDS loss value `0.5 · ‖ε_θ − ε‖²` (for logging).
    pub sds_loss_value: f32,
    /// L2 norm of the SDS gradient.
    pub grad_norm: f32,
    /// Noise level `σ_t` at the sampled timestep.
    pub sigma: f32,
    /// Signal retention `ᾱ_t` at the sampled timestep.
    pub alpha: f32,
}

/// Collect diagnostics for a completed SDS step.
///
/// # Arguments
/// - `sds_grad`: the computed SDS gradient vector.
/// - `t`: diffusion timestep used.
/// - `schedule`: noise schedule.
/// - `weighting`: weighting scheme (used to retrieve `w(t)`).
/// - `loss`: pre-computed scalar loss value (from [`sds_loss`]).
///
/// # Errors
/// Propagates errors from [`compute_weight`], [`SdsNoiseSchedule::get_sigma`],
/// and [`SdsNoiseSchedule::get_alpha`].
pub fn compute_sds_step_stats(
    sds_grad: &[f32],
    t: usize,
    schedule: &SdsNoiseSchedule,
    weighting: &ScoreWeighting,
    loss: f32,
) -> Result<SdsStepStats, ScoreDistillationError> {
    let weight = compute_weight(t, schedule, weighting)?;
    let sigma = schedule.get_sigma(t)?;
    let alpha = schedule.get_alpha(t)?;
    let grad_norm = sds_grad_norm(sds_grad);

    Ok(SdsStepStats {
        timestep: t,
        weight,
        sds_loss_value: loss,
        grad_norm,
        sigma,
        alpha,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- SdsConfig::validate ------------------------------------------------

    #[test]
    fn test_sds_config_default_valid() {
        SdsConfig::default().validate().unwrap();
    }

    #[test]
    fn test_sds_config_min_ge_max_err() {
        let cfg = SdsConfig {
            min_step_percent: 0.5,
            max_step_percent: 0.3,
            ..SdsConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ScoreDistillationError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_sds_config_min_eq_max_err() {
        let cfg = SdsConfig {
            min_step_percent: 0.5,
            max_step_percent: 0.5,
            ..SdsConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ScoreDistillationError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_sds_config_negative_scale_err() {
        let cfg = SdsConfig {
            guidance_scale: -1.0,
            ..SdsConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ScoreDistillationError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_sds_config_zero_scale_err() {
        let cfg = SdsConfig {
            guidance_scale: 0.0,
            ..SdsConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ScoreDistillationError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_sds_config_dreamfusion_valid() {
        SdsConfig::dreamfusion().validate().unwrap();
    }

    #[test]
    fn test_sds_config_fantasia3d_valid() {
        SdsConfig::fantasia3d().validate().unwrap();
    }

    #[test]
    fn test_sds_config_min_out_of_range_err() {
        let cfg = SdsConfig {
            min_step_percent: -0.1,
            max_step_percent: 0.5,
            ..SdsConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ScoreDistillationError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_sds_config_max_out_of_range_err() {
        let cfg = SdsConfig {
            min_step_percent: 0.02,
            max_step_percent: 1.1,
            ..SdsConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ScoreDistillationError::InvalidConfig(_))
        ));
    }

    // ---- VsdConfig::validate ------------------------------------------------

    #[test]
    fn test_vsd_config_default_valid() {
        VsdConfig::default().validate().unwrap();
    }

    #[test]
    fn test_vsd_config_zero_particles_err() {
        let cfg = VsdConfig {
            num_particles: 0,
            ..VsdConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ScoreDistillationError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_vsd_config_zero_lora_rank_err() {
        let cfg = VsdConfig {
            lora_rank: 0,
            ..VsdConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ScoreDistillationError::InvalidConfig(_))
        ));
    }

    // ---- VsdState -------------------------------------------------------
    //
    // Regression coverage: `num_particles`, `lora_rank` and `vsd_lr` used to
    // be validated but never consumed anywhere. `VsdState` is the piece that
    // now makes them load-bearing.

    #[test]
    fn test_vsd_state_new_allocates_one_adapter_per_particle() {
        let cfg = VsdConfig {
            num_particles: 3,
            lora_rank: 4,
            ..VsdConfig::default()
        };
        let state = VsdState::new(&cfg).unwrap();
        assert_eq!(state.num_particles(), 3);
        assert_eq!(state.particles.len(), 3);
    }

    #[test]
    fn test_vsd_state_new_rejects_invalid_config() {
        let cfg = VsdConfig {
            num_particles: 0,
            ..VsdConfig::default()
        };
        assert!(VsdState::new(&cfg).is_err());
    }

    #[test]
    fn test_vsd_state_add_layer_uses_configured_rank() {
        let cfg = VsdConfig {
            num_particles: 1,
            lora_rank: 8,
            ..VsdConfig::default()
        };
        let mut state = VsdState::new(&cfg).unwrap();
        state.add_layer(0, "to_q", 16, 16, 42).unwrap();
        let layer = state.particles[0].get_layer("to_q").unwrap();
        assert_eq!(
            layer.rank, 8,
            "layer rank should come from config.lora_rank"
        );
    }

    #[test]
    fn test_vsd_state_add_layer_out_of_range_particle_err() {
        let cfg = VsdConfig {
            num_particles: 1,
            ..VsdConfig::default()
        };
        let mut state = VsdState::new(&cfg).unwrap();
        let err = state.add_layer(5, "to_q", 16, 16, 1).unwrap_err();
        assert!(matches!(err, ScoreDistillationError::InvalidConfig(_)));
    }

    #[test]
    fn test_vsd_state_sgd_update_moves_params_by_configured_lr() {
        let cfg = VsdConfig {
            num_particles: 1,
            lora_rank: 2,
            vsd_lr: 0.5,
            ..VsdConfig::default()
        };
        let mut state = VsdState::new(&cfg).unwrap();
        state.add_layer(0, "to_q", 4, 4, 7).unwrap();

        let a_before = state.particles[0]
            .get_layer("to_q")
            .unwrap()
            .a_matrix
            .clone();
        let b_before = state.particles[0]
            .get_layer("to_q")
            .unwrap()
            .b_matrix
            .clone();
        let grad_a = vec![1.0_f32; a_before.len()];
        let grad_b = vec![1.0_f32; b_before.len()];

        state.sgd_update(0, "to_q", &grad_a, &grad_b).unwrap();

        let layer_after = state.particles[0].get_layer("to_q").unwrap();
        // param -= lr * grad, with grad == 1.0 everywhere and lr == 0.5.
        for (before, after) in a_before.iter().zip(layer_after.a_matrix.iter()) {
            assert!((before - 0.5 - after).abs() < 1e-6);
        }
        for (before, after) in b_before.iter().zip(layer_after.b_matrix.iter()) {
            assert!((before - 0.5 - after).abs() < 1e-6);
        }
        assert_eq!(state.particles[0].step, 1);
    }

    #[test]
    fn test_vsd_state_sgd_update_missing_layer_err() {
        let cfg = VsdConfig {
            num_particles: 1,
            ..VsdConfig::default()
        };
        let mut state = VsdState::new(&cfg).unwrap();
        let err = state.sgd_update(0, "does_not_exist", &[], &[]).unwrap_err();
        assert!(matches!(err, ScoreDistillationError::InvalidConfig(_)));
    }

    // ---- SdsNoiseSchedule::linear -------------------------------------------

    /// Regression test: `linear()` documents (and its name implies) the
    /// Stable Diffusion "scaled linear" convention
    /// (betas = linspace(sqrt(beta_start), sqrt(beta_end), T)^2), not a
    /// plain linear ramp between the same endpoints. Reconstruct beta_500
    /// from consecutive alphas_cumprod entries and check it matches the
    /// scaled formula rather than the plain one (the two disagree by
    /// roughly 30% at the midpoint, so f32 rounding cannot hide the
    /// difference).
    #[test]
    fn test_linear_schedule_uses_scaled_not_plain_beta() {
        let schedule = SdsNoiseSchedule::linear(1000);

        let beta_start = 0.00085_f64;
        let beta_end = 0.012_f64;
        let frac = 500.0_f64 / 999.0;
        let sqrt_beta_start = beta_start.sqrt();
        let sqrt_beta_end = beta_end.sqrt();
        let scaled_beta_500 = (sqrt_beta_start + frac * (sqrt_beta_end - sqrt_beta_start)).powi(2);
        let plain_beta_500 = beta_start + frac * (beta_end - beta_start);
        // Sanity: the two candidate formulas actually disagree here.
        assert!(
            (scaled_beta_500 - plain_beta_500).abs() > 1e-3,
            "scaled={scaled_beta_500} plain={plain_beta_500} should differ materially"
        );

        // alphas_cumprod[i] = alphas_cumprod[i-1] * (1 - beta_i)
        // => beta_500 = 1 - alphas_cumprod[500] / alphas_cumprod[499]
        let alpha_499 = schedule.alphas_cumprod[499] as f64;
        let alpha_500 = schedule.alphas_cumprod[500] as f64;
        let observed_beta_500 = 1.0 - alpha_500 / alpha_499;

        assert!(
            (observed_beta_500 - scaled_beta_500).abs() < 1e-4,
            "observed beta_500={observed_beta_500}, expected scaled-linear \
             beta_500={scaled_beta_500} (plain-linear would give {plain_beta_500})"
        );
    }

    #[test]
    fn test_linear_schedule_alphas_decreasing() {
        let schedule = SdsNoiseSchedule::linear(100);
        for i in 1..schedule.num_timesteps {
            assert!(
                schedule.alphas_cumprod[i] <= schedule.alphas_cumprod[i - 1],
                "alpha not monotonically decreasing at i={i}"
            );
        }
    }

    #[test]
    fn test_linear_schedule_sigmas_increasing() {
        let schedule = SdsNoiseSchedule::linear(100);
        for i in 1..schedule.num_timesteps {
            assert!(
                schedule.sigmas[i] >= schedule.sigmas[i - 1],
                "sigma not monotonically increasing at i={i}"
            );
        }
    }

    #[test]
    fn test_linear_schedule_alphas_in_unit_range() {
        let schedule = SdsNoiseSchedule::linear(1000);
        for &a in &schedule.alphas_cumprod {
            assert!((0.0..=1.0).contains(&a));
        }
    }

    #[test]
    fn test_linear_schedule_sigmas_in_unit_range() {
        let schedule = SdsNoiseSchedule::linear(1000);
        for &s in &schedule.sigmas {
            assert!((0.0..=1.0).contains(&s));
        }
    }

    #[test]
    fn test_cosine_schedule_alphas_decreasing() {
        let schedule = SdsNoiseSchedule::cosine(100);
        for i in 1..schedule.num_timesteps {
            assert!(
                schedule.alphas_cumprod[i] <= schedule.alphas_cumprod[i - 1],
                "cosine alpha not monotonically decreasing at i={i}"
            );
        }
    }

    // ---- SdsNoiseSchedule::get_sigma / get_alpha ----------------------------

    #[test]
    fn test_get_sigma_valid() {
        let schedule = SdsNoiseSchedule::linear(1000);
        let sigma = schedule.get_sigma(500).unwrap();
        assert!(sigma > 0.0 && sigma < 1.0);
    }

    #[test]
    fn test_get_sigma_out_of_range_err() {
        let schedule = SdsNoiseSchedule::linear(1000);
        assert!(matches!(
            schedule.get_sigma(1000),
            Err(ScoreDistillationError::InvalidTimestep {
                t: 1000,
                max_t: 999
            })
        ));
    }

    #[test]
    fn test_get_alpha_valid() {
        let schedule = SdsNoiseSchedule::linear(1000);
        let alpha = schedule.get_alpha(0).unwrap();
        assert!(alpha > 0.0 && alpha <= 1.0);
    }

    #[test]
    fn test_get_alpha_out_of_range_err() {
        let schedule = SdsNoiseSchedule::linear(100);
        assert!(matches!(
            schedule.get_alpha(100),
            Err(ScoreDistillationError::InvalidTimestep { .. })
        ));
    }

    // ---- sample_timestep ----------------------------------------------------

    #[test]
    fn test_sample_timestep_in_range() {
        let num_t = 1000;
        for seed in [1u64, 42, 999, u64::MAX / 2] {
            let t = sample_timestep(0.02, 0.98, num_t, seed);
            assert!(
                (20..=980).contains(&t),
                "t={t} out of expected range with seed={seed}"
            );
        }
    }

    #[test]
    fn test_sample_timestep_deterministic() {
        let t1 = sample_timestep(0.02, 0.98, 1000, 12345);
        let t2 = sample_timestep(0.02, 0.98, 1000, 12345);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_sample_timestep_min_eq_max() {
        let t = sample_timestep(0.5, 0.5, 1000, 1);
        assert_eq!(t, 500);
    }

    // ---- add_sds_noise ------------------------------------------------------

    #[test]
    fn test_add_sds_noise_shape_preserved() {
        let schedule = SdsNoiseSchedule::linear(1000);
        let latent = vec![1.0f32; 32];
        let noise = vec![0.0f32; 32];
        let noisy = add_sds_noise(&latent, 100, &schedule, &noise).unwrap();
        assert_eq!(noisy.len(), 32);
    }

    #[test]
    fn test_add_sds_noise_zero_noise_near_original() {
        let schedule = SdsNoiseSchedule::linear(1000);
        // At t=0 sigma is very small, so the noisy version ≈ original * sqrt(alpha_0).
        // With zero noise the result is exactly sqrt(alpha) * x₀.
        let latent = vec![1.0f32; 4];
        let noise = vec![0.0f32; 4];
        let noisy = add_sds_noise(&latent, 0, &schedule, &noise).unwrap();
        let alpha = schedule.get_alpha(0).unwrap();
        let expected = alpha.sqrt();
        for &v in &noisy {
            assert!((v - expected).abs() < 1e-5, "v={v}, expected={expected}");
        }
    }

    #[test]
    fn test_add_sds_noise_empty_latent_err() {
        let schedule = SdsNoiseSchedule::linear(1000);
        let err = add_sds_noise(&[], 0, &schedule, &[]).unwrap_err();
        assert!(matches!(err, ScoreDistillationError::EmptyLatent));
    }

    #[test]
    fn test_add_sds_noise_dimension_mismatch_err() {
        let schedule = SdsNoiseSchedule::linear(1000);
        let err = add_sds_noise(&[1.0], 0, &schedule, &[1.0, 2.0]).unwrap_err();
        assert!(matches!(
            err,
            ScoreDistillationError::DimensionMismatch { .. }
        ));
    }

    // ---- classifier_free_guidance -------------------------------------------

    #[test]
    fn test_cfg_scale_zero_returns_uncond() {
        let cond = vec![1.0f32, 2.0, 3.0];
        let uncond = vec![0.1f32, 0.2, 0.3];
        let out = classifier_free_guidance(&cond, &uncond, 0.0).unwrap();
        for (o, u) in out.iter().zip(uncond.iter()) {
            assert!((o - u).abs() < 1e-6, "expected uncond but got {o} vs {u}");
        }
    }

    #[test]
    fn test_cfg_scale_one_returns_cond() {
        let cond = vec![1.0f32, 2.0, 3.0];
        let uncond = vec![0.1f32, 0.2, 0.3];
        let out = classifier_free_guidance(&cond, &uncond, 1.0).unwrap();
        for (o, c) in out.iter().zip(cond.iter()) {
            assert!((o - c).abs() < 1e-6, "expected cond but got {o} vs {c}");
        }
    }

    #[test]
    fn test_cfg_typical_scale() {
        let cond = vec![1.0f32];
        let uncond = vec![0.0f32];
        let out = classifier_free_guidance(&cond, &uncond, 7.5).unwrap();
        assert!((out[0] - 7.5).abs() < 1e-5);
    }

    #[test]
    fn test_cfg_empty_err() {
        let err = classifier_free_guidance(&[], &[], 1.0).unwrap_err();
        assert!(matches!(err, ScoreDistillationError::EmptyLatent));
    }

    #[test]
    fn test_cfg_dimension_mismatch_err() {
        let err = classifier_free_guidance(&[1.0, 2.0], &[1.0], 1.0).unwrap_err();
        assert!(matches!(
            err,
            ScoreDistillationError::DimensionMismatch { .. }
        ));
    }

    // ---- sds_gradient -------------------------------------------------------

    #[test]
    fn test_sds_gradient_zero_diff_near_zero() {
        let schedule = SdsNoiseSchedule::linear(1000);
        let cfg = SdsConfig {
            weighting: ScoreWeighting::Fixed(1.0),
            guidance_scale: 7.5,
            ..SdsConfig::default()
        };
        // When cond == uncond == noise, CFG output == noise → gradient ≈ 0.
        let pred = vec![0.5f32; 8];
        let grad = sds_gradient(&pred, &pred, &pred, 500, &schedule, &cfg).unwrap();
        for g in &grad {
            assert!(g.abs() < 1e-6, "expected ~0 gradient but got {g}");
        }
    }

    #[test]
    fn test_sds_gradient_shape_preserved() {
        let schedule = SdsNoiseSchedule::linear(1000);
        let cfg = SdsConfig::default();
        let pred = vec![1.0f32; 16];
        let noise = vec![0.0f32; 16];
        let grad = sds_gradient(&pred, &pred, &noise, 200, &schedule, &cfg).unwrap();
        assert_eq!(grad.len(), 16);
    }

    #[test]
    fn test_sds_gradient_empty_err() {
        let schedule = SdsNoiseSchedule::linear(1000);
        let cfg = SdsConfig::default();
        let err = sds_gradient(&[], &[], &[], 0, &schedule, &cfg).unwrap_err();
        assert!(matches!(err, ScoreDistillationError::EmptyLatent));
    }

    // ---- sds_loss -----------------------------------------------------------

    #[test]
    fn test_sds_loss_identical_pred_noise_zero() {
        let pred = vec![1.0f32, 2.0, 3.0];
        let loss = sds_loss(&pred, &pred).unwrap();
        assert!(loss.abs() < 1e-6, "expected 0 loss but got {loss}");
    }

    #[test]
    fn test_sds_loss_known_value() {
        // pred = [1], noise = [0] → loss = 0.5 * 1² = 0.5
        let loss = sds_loss(&[1.0], &[0.0]).unwrap();
        assert!((loss - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sds_loss_empty_err() {
        let err = sds_loss(&[], &[]).unwrap_err();
        assert!(matches!(err, ScoreDistillationError::EmptyLatent));
    }

    #[test]
    fn test_sds_loss_mismatch_err() {
        let err = sds_loss(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(matches!(
            err,
            ScoreDistillationError::DimensionMismatch { .. }
        ));
    }

    // ---- compute_weight -----------------------------------------------------

    #[test]
    fn test_compute_weight_fixed() {
        let schedule = SdsNoiseSchedule::linear(1000);
        let w = compute_weight(500, &schedule, &ScoreWeighting::Fixed(3.1)).unwrap();
        assert!((w - 3.1).abs() < 1e-5);
    }

    #[test]
    fn test_compute_weight_snr_weighted_schedule_dependent() {
        let schedule = SdsNoiseSchedule::linear(1000);
        let w0 = compute_weight(100, &schedule, &ScoreWeighting::SnrWeighted).unwrap();
        let w1 = compute_weight(800, &schedule, &ScoreWeighting::SnrWeighted).unwrap();
        // Later timesteps have larger sigma → larger SnrWeighted value.
        assert!(w1 > w0);
    }

    #[test]
    fn test_compute_weight_max_snr_capped_at_one() {
        let schedule = SdsNoiseSchedule::linear(1000);
        // At early timesteps SNR is very high; the capped weight should be ≤ 1.
        let w = compute_weight(10, &schedule, &ScoreWeighting::MaxSnrWeighted).unwrap();
        assert!(w <= 1.0 + 1e-5);
    }

    #[test]
    fn test_compute_weight_timestep_decay_decreasing() {
        let schedule = SdsNoiseSchedule::linear(1000);
        let decay = ScoreWeighting::TimestepDecay { decay: 1.0 };
        let w0 = compute_weight(0, &schedule, &decay).unwrap(); // t=1 → 1.0
        let w1 = compute_weight(99, &schedule, &decay).unwrap(); // t=100 → 0.01
        assert!(w0 > w1);
    }

    // ---- vsd_gradient -------------------------------------------------------

    #[test]
    fn test_vsd_gradient_identical_near_zero() {
        let schedule = SdsNoiseSchedule::linear(1000);
        let cfg = VsdConfig::default();
        let pred = vec![0.3f32; 8];
        let grad = vsd_gradient(&pred, &pred, 500, &schedule, &cfg).unwrap();
        for g in &grad {
            assert!(g.abs() < 1e-6, "expected ~0 but got {g}");
        }
    }

    #[test]
    fn test_vsd_gradient_shape_preserved() {
        let schedule = SdsNoiseSchedule::linear(1000);
        let cfg = VsdConfig::default();
        let ref_pred = vec![1.0f32; 12];
        let lora_pred = vec![0.0f32; 12];
        let grad = vsd_gradient(&ref_pred, &lora_pred, 300, &schedule, &cfg).unwrap();
        assert_eq!(grad.len(), 12);
    }

    #[test]
    fn test_vsd_gradient_empty_err() {
        let schedule = SdsNoiseSchedule::linear(1000);
        let cfg = VsdConfig::default();
        let err = vsd_gradient(&[], &[], 0, &schedule, &cfg).unwrap_err();
        assert!(matches!(err, ScoreDistillationError::EmptyLatent));
    }

    #[test]
    fn test_vsd_gradient_mismatch_err() {
        let schedule = SdsNoiseSchedule::linear(1000);
        let cfg = VsdConfig::default();
        let err = vsd_gradient(&[1.0, 2.0], &[1.0], 0, &schedule, &cfg).unwrap_err();
        assert!(matches!(
            err,
            ScoreDistillationError::DimensionMismatch { .. }
        ));
    }

    // ---- backprop_sds_to_gaussians ------------------------------------------

    #[test]
    fn test_backprop_zero_grad_zero_output() {
        // J^T * 0 = 0
        let jacobian = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2
        let sds_grad = vec![0.0f32; 2];
        let out = backprop_sds_to_gaussians(&jacobian, &sds_grad, 2, 2).unwrap();
        assert_eq!(out, vec![0.0; 2]);
    }

    #[test]
    fn test_backprop_identity_jacobian() {
        // J = identity 2x2, g = [1, 2] → J^T g = [1, 2]
        let jacobian = vec![1.0f32, 0.0, 0.0, 1.0];
        let sds_grad = vec![1.0f32, 2.0];
        let out = backprop_sds_to_gaussians(&jacobian, &sds_grad, 2, 2).unwrap();
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_backprop_known_value() {
        // J = [[1,2],[3,4]], g = [1,1]
        // J^T g = [1*1+3*1, 2*1+4*1] = [4, 6]
        let jacobian = vec![1.0f32, 2.0, 3.0, 4.0]; // row-major: J[0,0]=1, J[0,1]=2, J[1,0]=3, J[1,1]=4
        let sds_grad = vec![1.0f32, 1.0];
        let out = backprop_sds_to_gaussians(&jacobian, &sds_grad, 2, 2).unwrap();
        assert!((out[0] - 4.0).abs() < 1e-6, "out[0]={}", out[0]);
        assert!((out[1] - 6.0).abs() < 1e-6, "out[1]={}", out[1]);
    }

    #[test]
    fn test_backprop_dimension_error_jacobian() {
        let err = backprop_sds_to_gaussians(&[1.0, 2.0], &[1.0, 2.0], 2, 3).unwrap_err();
        assert!(matches!(
            err,
            ScoreDistillationError::DimensionMismatch {
                expected: 6,
                actual: 2
            }
        ));
    }

    #[test]
    fn test_backprop_dimension_error_grad() {
        let jacobian = vec![1.0f32; 4]; // 2x2
        let err = backprop_sds_to_gaussians(&jacobian, &[1.0, 2.0, 3.0], 2, 2).unwrap_err();
        assert!(matches!(
            err,
            ScoreDistillationError::DimensionMismatch {
                expected: 2,
                actual: 3
            }
        ));
    }

    // ---- annealed_timestep_range --------------------------------------------

    #[test]
    fn test_annealed_range_step_zero_returns_initial() {
        let (min_p, max_p) = annealed_timestep_range(0, 1000, 0.98, 0.5, 0.02);
        assert!((min_p - 0.02).abs() < 1e-6);
        assert!((max_p - 0.98).abs() < 1e-6);
    }

    #[test]
    fn test_annealed_range_step_total_returns_final() {
        let (min_p, max_p) = annealed_timestep_range(1000, 1000, 0.98, 0.5, 0.02);
        assert!((min_p - 0.02).abs() < 1e-6);
        assert!((max_p - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_annealed_range_monotone_decreasing_max() {
        let total = 1000;
        let mut prev_max = 0.98f32;
        for step in [0, 100, 250, 500, 750, 1000] {
            let (_, max_p) = annealed_timestep_range(step, total, 0.98, 0.5, 0.02);
            assert!(max_p <= prev_max + 1e-6);
            prev_max = max_p;
        }
    }

    // ---- compute_sds_step_stats ---------------------------------------------

    #[test]
    fn test_compute_sds_step_stats_valid() {
        let schedule = SdsNoiseSchedule::linear(1000);
        let grad = vec![0.1f32, -0.2, 0.3];
        let stats =
            compute_sds_step_stats(&grad, 500, &schedule, &ScoreWeighting::Fixed(1.0), 0.42)
                .unwrap();
        assert_eq!(stats.timestep, 500);
        assert!((stats.sds_loss_value - 0.42).abs() < 1e-6);
        assert!(stats.grad_norm > 0.0);
        assert!(stats.sigma > 0.0);
        assert!(stats.alpha > 0.0);
    }

    #[test]
    fn test_compute_sds_step_stats_loss_field() {
        let schedule = SdsNoiseSchedule::linear(1000);
        let stats =
            compute_sds_step_stats(&[0.0], 0, &schedule, &ScoreWeighting::Fixed(1.0), 3.1).unwrap();
        assert!((stats.sds_loss_value - 3.1).abs() < 1e-5);
    }

    // ---- sds_grad_norm ------------------------------------------------------

    #[test]
    fn test_sds_grad_norm_zero_vector() {
        let norm = sds_grad_norm(&[0.0, 0.0, 0.0]);
        assert!(norm.abs() < 1e-6);
    }

    #[test]
    fn test_sds_grad_norm_unit_vector() {
        let norm = sds_grad_norm(&[1.0, 0.0, 0.0]);
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sds_grad_norm_known_value() {
        // [3, 4] → 5
        let norm = sds_grad_norm(&[3.0, 4.0]);
        assert!((norm - 5.0).abs() < 1e-5);
    }

    // ---- clip_sds_gradient --------------------------------------------------

    #[test]
    fn test_clip_under_max_unchanged() {
        let mut grad = vec![0.3f32, 0.4f32]; // norm = 0.5
        clip_sds_gradient(&mut grad, 1.0);
        assert!((grad[0] - 0.3).abs() < 1e-6);
        assert!((grad[1] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_clip_over_max_reduces_norm() {
        let mut grad = vec![3.0f32, 4.0f32]; // norm = 5
        clip_sds_gradient(&mut grad, 1.0);
        let norm_after = sds_grad_norm(&grad);
        assert!((norm_after - 1.0).abs() < 1e-5, "norm_after={norm_after}");
    }

    #[test]
    fn test_clip_preserves_direction() {
        let mut grad = vec![3.0f32, 4.0f32]; // direction ratio 3:4
        clip_sds_gradient(&mut grad, 2.5);
        // After clipping: norm = 2.5, direction unchanged
        let norm = sds_grad_norm(&grad);
        assert!((norm - 2.5).abs() < 1e-5);
        // ratio should still be 3:4
        assert!((grad[0] / grad[1] - 3.0 / 4.0).abs() < 1e-5);
    }
}
