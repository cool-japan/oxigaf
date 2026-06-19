//! Noise injection techniques for 3DGS training robustness.
//!
//! Adding controlled noise to Gaussian Splatting parameters during training
//! acts as regularization and data augmentation, improving generalization.
//!
//! Provides:
//! - Additive and multiplicative Gaussian noise injection
//! - Rotation perturbation via small axis-angle rotations
//! - Curriculum noise scheduling (constant, linear, cosine, step-decay)
//! - Stateful `NoiseInjector` for per-step management
//! - `NoiseAnalysis` for quantifying injected noise statistics

use thiserror::Error;

// ──────────────────────────────────────────────────────────────────────────────
// Errors
// ──────────────────────────────────────────────────────────────────────────────

/// Errors produced by noise injection operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum NoiseInjectionError {
    /// Configuration value is invalid (e.g. negative std, bad probability).
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// Input slice lengths did not match the expected size.
    #[error("length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    /// Input slice was empty when at least one element was required.
    #[error("empty input")]
    EmptyInput,
}

// ──────────────────────────────────────────────────────────────────────────────
// Inline xorshift64 PRNG (spec version, not augmentation.rs version)
// ──────────────────────────────────────────────────────────────────────────────

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[inline]
fn xorshift_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 32) as f32 / u32::MAX as f32
}

/// Box-Muller transform for Gaussian noise.
///
/// Returns two independent standard-normal samples from two uniform draws.
fn box_muller(state: &mut u64) -> (f32, f32) {
    let u1 = xorshift_f32(state).max(1e-10);
    let u2 = xorshift_f32(state);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    (r * theta.cos(), r * theta.sin())
}

// ──────────────────────────────────────────────────────────────────────────────
// NoiseSchedule
// ──────────────────────────────────────────────────────────────────────────────

/// Schedule that governs how noise magnitude evolves over training.
///
/// Curriculum noise typically decreases over time so the model starts with
/// higher regularization and gradually relies less on noise for robustness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseSchedule {
    /// Constant noise level throughout training (scale = 1.0 always).
    Constant,

    /// Linearly decrease from `start_scale` to `end_scale` over `total_steps`.
    Linear {
        start_scale: f32,
        end_scale: f32,
        total_steps: usize,
    },

    /// Cosine annealing: `scale = end + 0.5*(start-end)*(1+cos(π*step/total))`.
    Cosine {
        start_scale: f32,
        end_scale: f32,
        total_steps: usize,
    },

    /// Step decay: scale halved every `step_size` steps.
    StepDecay {
        initial_scale: f32,
        step_size: usize,
        decay_factor: f32,
    },
}

impl NoiseSchedule {
    /// Returns the multiplicative scale factor at the given training step.
    ///
    /// The effective std is `base_std * scale_at(step)`.
    pub fn scale_at(&self, step: usize) -> f32 {
        match self {
            NoiseSchedule::Constant => 1.0,

            NoiseSchedule::Linear {
                start_scale,
                end_scale,
                total_steps,
            } => {
                if *total_steps == 0 {
                    return *end_scale;
                }
                let t = (step as f32 / *total_steps as f32).clamp(0.0, 1.0);
                (1.0 - t) * (start_scale - end_scale) + end_scale
            }

            NoiseSchedule::Cosine {
                start_scale,
                end_scale,
                total_steps,
            } => {
                if *total_steps == 0 {
                    return *end_scale;
                }
                let t = (step as f32 / *total_steps as f32).clamp(0.0, 1.0);
                let cos_factor = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
                end_scale + cos_factor * (start_scale - end_scale)
            }

            NoiseSchedule::StepDecay {
                initial_scale,
                step_size,
                decay_factor,
            } => {
                if *step_size == 0 {
                    return *initial_scale;
                }
                let n_decays = (step / step_size) as i32;
                initial_scale * decay_factor.powi(n_decays)
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NoiseTarget
// ──────────────────────────────────────────────────────────────────────────────

/// Specifies which 3DGS parameter buffer(s) receive noise injection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseTarget {
    /// xyz positions (3 floats per Gaussian).
    Positions,
    /// Log-space scales (3 floats per Gaussian).
    LogScales,
    /// Quaternion rotations (4 floats per Gaussian).
    Rotations,
    /// SH DC colour coefficients (3 floats per Gaussian).
    Colors,
    /// Logit-space opacities (1 float per Gaussian).
    Opacities,
    /// All of the above.
    All,
}

// ──────────────────────────────────────────────────────────────────────────────
// NoiseConfig
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for a single noise-injection pass.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseConfig {
    /// Which parameter buffer(s) to perturb.
    pub target: NoiseTarget,
    /// Base noise standard deviation (non-negative).
    pub std: f32,
    /// Probability of applying noise to each Gaussian element (0..=1).
    pub probability: f32,
    /// Noise schedule for curriculum training.
    pub schedule: NoiseSchedule,
    /// Clip noise magnitude to this many std devs (0 = no clip).
    pub clip_sigma: f32,
    /// Random seed base.
    pub seed: u64,
}

impl NoiseConfig {
    /// Create a new config with sensible defaults.
    ///
    /// Default: `probability=1.0`, `schedule=Constant`, `clip_sigma=3.0`,
    /// `seed=42`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if `std < 0`.
    pub fn new(target: NoiseTarget, std: f32) -> Result<Self, NoiseInjectionError> {
        if std < 0.0 {
            return Err(NoiseInjectionError::InvalidConfig(format!(
                "std must be >= 0, got {std}"
            )));
        }
        Ok(Self {
            target,
            std,
            probability: 1.0,
            schedule: NoiseSchedule::Constant,
            clip_sigma: 3.0,
            seed: 42,
        })
    }

    /// Set the per-element application probability.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` if `p` is not in `[0, 1]`.
    pub fn with_probability(mut self, p: f32) -> Result<Self, NoiseInjectionError> {
        if !(0.0..=1.0).contains(&p) {
            return Err(NoiseInjectionError::InvalidConfig(format!(
                "probability must be in [0, 1], got {p}"
            )));
        }
        self.probability = p;
        Ok(self)
    }

    /// Set the noise schedule.
    pub fn with_schedule(mut self, schedule: NoiseSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    /// Set the clip sigma threshold (0 = no clipping).
    pub fn with_clip_sigma(mut self, clip: f32) -> Self {
        self.clip_sigma = clip;
        self
    }

    /// Set the random seed base.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validate all fields.
    ///
    /// # Errors
    ///
    /// Returns `InvalidConfig` when `std < 0`, `probability ∉ [0,1]`, or
    /// `clip_sigma < 0`.
    pub fn validate(&self) -> Result<(), NoiseInjectionError> {
        if self.std < 0.0 {
            return Err(NoiseInjectionError::InvalidConfig(format!(
                "std must be >= 0, got {}",
                self.std
            )));
        }
        if !(0.0..=1.0).contains(&self.probability) {
            return Err(NoiseInjectionError::InvalidConfig(format!(
                "probability must be in [0, 1], got {}",
                self.probability
            )));
        }
        if self.clip_sigma < 0.0 {
            return Err(NoiseInjectionError::InvalidConfig(format!(
                "clip_sigma must be >= 0, got {}",
                self.clip_sigma
            )));
        }
        Ok(())
    }
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            target: NoiseTarget::Positions,
            std: 0.001,
            probability: 1.0,
            schedule: NoiseSchedule::Constant,
            clip_sigma: 3.0,
            seed: 42,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Core noise functions
// ──────────────────────────────────────────────────────────────────────────────

/// Sample Gaussian noise for a flat parameter buffer.
///
/// Returns a `Vec<f32>` of length `n`. Uses Box-Muller sampling.
/// If `clip_sigma > 0`, each sample is clamped to `[-clip_sigma*std, clip_sigma*std]`.
/// For `n == 0` returns an empty `Vec`.
pub fn sample_gaussian_noise(n: usize, std: f32, clip_sigma: f32, seed: u64) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }

    let mut state = seed.max(1);
    let mut noise = Vec::with_capacity(n);

    let mut i = 0;
    while i < n {
        let (a, b) = box_muller(&mut state);
        let na = a * std;
        let nb = b * std;

        let na = if clip_sigma > 0.0 {
            na.clamp(-clip_sigma * std, clip_sigma * std)
        } else {
            na
        };

        noise.push(na);
        i += 1;

        if i < n {
            let nb = if clip_sigma > 0.0 {
                nb.clamp(-clip_sigma * std, clip_sigma * std)
            } else {
                nb
            };
            noise.push(nb);
            i += 1;
        }
    }

    noise
}

/// Apply additive Gaussian noise to `params` in-place.
///
/// Only applies to elements where a Bernoulli draw (probability `probability`)
/// is true. Returns the number of elements modified.
///
/// # Errors
///
/// - `EmptyInput` when `params` is empty.
/// - `InvalidConfig` when `probability ∉ [0, 1]`.
pub fn inject_additive_noise(
    params: &mut [f32],
    std: f32,
    probability: f32,
    clip_sigma: f32,
    seed: u64,
) -> Result<usize, NoiseInjectionError> {
    if params.is_empty() {
        return Err(NoiseInjectionError::EmptyInput);
    }
    if !(0.0..=1.0).contains(&probability) {
        return Err(NoiseInjectionError::InvalidConfig(format!(
            "probability must be in [0, 1], got {probability}"
        )));
    }

    let n = params.len();
    let noise = sample_gaussian_noise(n, std, clip_sigma, seed);

    // Use a separate PRNG state (seeded differently) for the Bernoulli draws
    let mut state_b = seed.wrapping_add(0xDEADBEEF_CAFEBABE).max(1);

    let mut modified = 0usize;
    for (i, param) in params.iter_mut().enumerate() {
        let u = xorshift_f32(&mut state_b);
        if u < probability {
            *param += noise[i];
            modified += 1;
        }
    }

    Ok(modified)
}

/// Apply multiplicative noise: `params[i] *= (1 + noise[i])`.
///
/// Good for scale parameters where a relative perturbation is desired.
/// Only applies to elements where a Bernoulli draw (probability `probability`)
/// is true. Returns the number of elements modified.
///
/// # Errors
///
/// - `EmptyInput` when `params` is empty.
/// - `InvalidConfig` when `probability ∉ [0, 1]`.
pub fn inject_multiplicative_noise(
    params: &mut [f32],
    std: f32,
    probability: f32,
    clip_sigma: f32,
    seed: u64,
) -> Result<usize, NoiseInjectionError> {
    if params.is_empty() {
        return Err(NoiseInjectionError::EmptyInput);
    }
    if !(0.0..=1.0).contains(&probability) {
        return Err(NoiseInjectionError::InvalidConfig(format!(
            "probability must be in [0, 1], got {probability}"
        )));
    }

    let n = params.len();
    let noise = sample_gaussian_noise(n, std, clip_sigma, seed);

    let mut state_b = seed.wrapping_add(0xDEADBEEF_CAFEBABE).max(1);

    let mut modified = 0usize;
    for (i, param) in params.iter_mut().enumerate() {
        let u = xorshift_f32(&mut state_b);
        if u < probability {
            *param *= 1.0 + noise[i];
            modified += 1;
        }
    }

    Ok(modified)
}

/// Perturb quaternion rotations by small random axis-angle perturbations.
///
/// `quaternions` is a flat buffer in `[qx, qy, qz, qw, ...]` order.
/// Its length must be divisible by 4.
///
/// For each quaternion selected by the Bernoulli draw:
/// 1. Sample a random unit axis by normalising 3 Gaussian draws.
/// 2. Sample angle `θ ~ N(0, angle_std)`, clamped to `[-π/4, π/4]`.
/// 3. Compute perturbation quaternion `q_p = [u·sin(θ/2), v·sin(θ/2), w·sin(θ/2), cos(θ/2)]`.
/// 4. Compose `q_new = q_p ⊗ q_original` (Hamilton product).
/// 5. Renormalise `q_new`.
///
/// Returns the number of quaternions perturbed.
///
/// # Errors
///
/// - `LengthMismatch` when `quaternions.len()` is not divisible by 4.
/// - `EmptyInput` when `quaternions` is empty.
/// - `InvalidConfig` when `probability ∉ [0, 1]`.
pub fn perturb_rotations(
    quaternions: &mut [f32],
    angle_std: f32,
    probability: f32,
    seed: u64,
) -> Result<usize, NoiseInjectionError> {
    if quaternions.is_empty() {
        return Err(NoiseInjectionError::EmptyInput);
    }
    if !quaternions.len().is_multiple_of(4) {
        return Err(NoiseInjectionError::LengthMismatch {
            expected: (quaternions.len() / 4) * 4,
            actual: quaternions.len(),
        });
    }
    if !(0.0..=1.0).contains(&probability) {
        return Err(NoiseInjectionError::InvalidConfig(format!(
            "probability must be in [0, 1], got {probability}"
        )));
    }

    let n_quats = quaternions.len() / 4;
    let mut state = seed.max(1);
    // Separate state for axis and angle sampling vs. Bernoulli draws
    let mut state_b = seed.wrapping_add(0xCAFEBABE_DEADBEEF).max(1);

    let mut modified = 0usize;

    for q_idx in 0..n_quats {
        let u_draw = xorshift_f32(&mut state_b);
        if u_draw >= probability {
            continue;
        }

        // Sample random axis from 3 Gaussian draws
        let (gx, gy) = box_muller(&mut state);
        let (gz, _gw) = box_muller(&mut state);

        let axis_norm = (gx * gx + gy * gy + gz * gz).sqrt();
        // Guard against degenerate zero-norm axis
        if axis_norm < 1e-8 {
            continue;
        }
        let (ax, ay, az) = (gx / axis_norm, gy / axis_norm, gz / axis_norm);

        // Sample angle from N(0, angle_std), clamp to [-π/4, π/4]
        let (ga, _) = box_muller(&mut state);
        let theta =
            (ga * angle_std).clamp(-std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_4);

        let half_theta = theta * 0.5;
        let sin_ht = half_theta.sin();
        let cos_ht = half_theta.cos();

        // Perturbation quaternion [px, py, pz, pw]
        let px = ax * sin_ht;
        let py = ay * sin_ht;
        let pz = az * sin_ht;
        let pw = cos_ht;

        // Original quaternion — layout: [qx, qy, qz, qw]
        let base = q_idx * 4;
        let ox = quaternions[base];
        let oy = quaternions[base + 1];
        let oz = quaternions[base + 2];
        let ow = quaternions[base + 3];

        // Hamilton product: q_new = q_p ⊗ q_original
        // (p1=(px,py,pz,pw), p2=(ox,oy,oz,ow))
        let nx = pw * ox + px * ow + py * oz - pz * oy;
        let ny = pw * oy - px * oz + py * ow + pz * ox;
        let nz = pw * oz + px * oy - py * ox + pz * ow;
        let nw = pw * ow - px * ox - py * oy - pz * oz;

        // Renormalise
        let norm = (nx * nx + ny * ny + nz * nz + nw * nw).sqrt();
        if norm < 1e-8 {
            continue;
        }

        quaternions[base] = nx / norm;
        quaternions[base + 1] = ny / norm;
        quaternions[base + 2] = nz / norm;
        quaternions[base + 3] = nw / norm;

        modified += 1;
    }

    Ok(modified)
}

// ──────────────────────────────────────────────────────────────────────────────
// NoiseInjector (stateful)
// ──────────────────────────────────────────────────────────────────────────────

/// Stateful noise injector that tracks training steps and cumulative statistics.
///
/// Applies noise injection to 3DGS parameter buffers with schedule-based
/// std scaling and per-step seed variation.
pub struct NoiseInjector {
    /// Underlying noise configuration.
    pub config: NoiseConfig,
    /// Current training step index.
    pub current_step: usize,
    /// Total number of elements modified across all steps.
    pub total_modified: usize,
    /// Total number of `step()` calls executed.
    pub num_applications: usize,
}

impl NoiseInjector {
    /// Create a new injector, validating the configuration.
    ///
    /// # Errors
    ///
    /// Propagates `NoiseConfig::validate` errors.
    pub fn new(config: NoiseConfig) -> Result<Self, NoiseInjectionError> {
        config.validate()?;
        Ok(Self {
            config,
            current_step: 0,
            total_modified: 0,
            num_applications: 0,
        })
    }

    /// Effective standard deviation at the current step.
    ///
    /// `effective_std = base_std * schedule.scale_at(current_step)`
    pub fn effective_std(&self) -> f32 {
        self.config.std * self.config.schedule.scale_at(self.current_step)
    }

    /// Apply noise to the appropriate parameter buffer(s) for the current step.
    ///
    /// The seed is varied per step:
    /// `step_seed = config.seed ^ (current_step.wrapping_mul(0x9E3779B97F4A7C15))`
    ///
    /// Increments `current_step` after applying. Returns the number of elements
    /// modified in this step.
    ///
    /// # Errors
    ///
    /// Propagates errors from the underlying injection functions.
    pub fn step(
        &mut self,
        positions: &mut [f32],
        log_scales: &mut [f32],
        rotations: &mut [f32],
        colors: &mut [f32],
        opacities: &mut [f32],
    ) -> Result<usize, NoiseInjectionError> {
        let step_seed = self
            .config
            .seed
            .wrapping_add((self.current_step as u64).wrapping_mul(0x9E3779B97F4A7C15_u64));

        let eff_std = self.effective_std();
        let prob = self.config.probability;
        let clip = self.config.clip_sigma;

        let modified = match self.config.target {
            NoiseTarget::Positions => {
                inject_additive_noise(positions, eff_std, prob, clip, step_seed)?
            }
            NoiseTarget::LogScales => {
                inject_multiplicative_noise(log_scales, eff_std, prob, clip, step_seed)?
            }
            NoiseTarget::Rotations => perturb_rotations(rotations, eff_std, prob, step_seed)?,
            NoiseTarget::Colors => inject_additive_noise(colors, eff_std, prob, clip, step_seed)?,
            NoiseTarget::Opacities => {
                inject_additive_noise(opacities, eff_std, prob, clip, step_seed)?
            }
            NoiseTarget::All => {
                // Vary seed per target to avoid correlation across buffers
                let mut total = 0usize;

                if !positions.is_empty() {
                    total += inject_additive_noise(
                        positions,
                        eff_std,
                        prob,
                        clip,
                        step_seed ^ 0x1111_1111_1111_1111,
                    )?;
                }
                if !log_scales.is_empty() {
                    total += inject_multiplicative_noise(
                        log_scales,
                        eff_std,
                        prob,
                        clip,
                        step_seed ^ 0x2222_2222_2222_2222,
                    )?;
                }
                if !rotations.is_empty() && rotations.len().is_multiple_of(4) {
                    total += perturb_rotations(
                        rotations,
                        eff_std,
                        prob,
                        step_seed ^ 0x3333_3333_3333_3333,
                    )?;
                }
                if !colors.is_empty() {
                    total += inject_additive_noise(
                        colors,
                        eff_std,
                        prob,
                        clip,
                        step_seed ^ 0x4444_4444_4444_4444,
                    )?;
                }
                if !opacities.is_empty() {
                    total += inject_additive_noise(
                        opacities,
                        eff_std,
                        prob,
                        clip,
                        step_seed ^ 0x5555_5555_5555_5555,
                    )?;
                }

                total
            }
        };

        self.total_modified += modified;
        self.num_applications += 1;
        self.current_step += 1;

        Ok(modified)
    }

    /// Reset the step counter and cumulative statistics.
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.total_modified = 0;
        self.num_applications = 0;
    }

    /// Format a human-readable statistics summary.
    pub fn format_stats(&self) -> String {
        format!(
            "NoiseInjector[step={}, applications={}, total_modified={}, eff_std={:.6}]",
            self.current_step,
            self.num_applications,
            self.total_modified,
            self.effective_std(),
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Noise analysis
// ──────────────────────────────────────────────────────────────────────────────

/// Statistical summary of injected noise.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseAnalysis {
    /// Mean absolute noise value.
    pub mean_abs_noise: f32,
    /// Standard deviation of the noise values.
    pub std_noise: f32,
    /// Maximum absolute noise value.
    pub max_abs_noise: f32,
    /// Noise-to-signal ratio: `std_noise / (signal_rms + 1e-8)`.
    pub noise_to_signal_ratio: f32,
    /// Number of elements that received non-zero noise.
    pub num_injected: usize,
    /// Total number of elements.
    pub num_total: usize,
}

/// Analyse noise by comparing original and noisy parameter buffers.
///
/// # Errors
///
/// - `EmptyInput` when `original` is empty.
/// - `LengthMismatch` when `original.len() != noisy.len()`.
pub fn analyze_noise(
    original: &[f32],
    noisy: &[f32],
) -> Result<NoiseAnalysis, NoiseInjectionError> {
    if original.is_empty() {
        return Err(NoiseInjectionError::EmptyInput);
    }
    if original.len() != noisy.len() {
        return Err(NoiseInjectionError::LengthMismatch {
            expected: original.len(),
            actual: noisy.len(),
        });
    }

    let n = original.len();
    let noise: Vec<f32> = original
        .iter()
        .zip(noisy.iter())
        .map(|(&o, &n)| n - o)
        .collect();

    let mean_abs_noise = noise.iter().map(|v| v.abs()).sum::<f32>() / n as f32;

    let mean_noise = noise.iter().sum::<f32>() / n as f32;
    let variance = noise.iter().map(|v| (v - mean_noise).powi(2)).sum::<f32>() / n as f32;
    let std_noise = variance.sqrt();

    let max_abs_noise = noise.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    let signal_rms = original.iter().map(|v| v * v).sum::<f32>().sqrt() / n as f32;
    // Use std_noise as the noise power measure for NSR
    let noise_to_signal_ratio = std_noise / (signal_rms + 1e-8);

    let num_injected = noise.iter().filter(|&&v| v != 0.0).count();

    Ok(NoiseAnalysis {
        mean_abs_noise,
        std_noise,
        max_abs_noise,
        noise_to_signal_ratio,
        num_injected,
        num_total: n,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ────────────────────────────────────────────────────────────────

    fn make_unit_quaternions(n: usize) -> Vec<f32> {
        // All identity quaternions [0, 0, 0, 1]
        let mut v = vec![0.0f32; n * 4];
        for i in 0..n {
            v[i * 4 + 3] = 1.0;
        }
        v
    }

    // ── Test 1: sample_gaussian_noise length ──────────────────────────────────

    #[test]
    fn test_sample_noise_length() {
        let noise = sample_gaussian_noise(100, 0.01, 3.0, 42);
        assert_eq!(noise.len(), 100);
    }

    // ── Test 2: values within clip range ─────────────────────────────────────

    #[test]
    fn test_sample_noise_clip_range() {
        let std = 0.01f32;
        let clip_sigma = 3.0f32;
        let noise = sample_gaussian_noise(10_000, std, clip_sigma, 123);
        let bound = clip_sigma * std;
        for v in &noise {
            assert!(
                *v >= -bound - 1e-6 && *v <= bound + 1e-6,
                "noise value {} outside clip range [{}, {}]",
                v,
                -bound,
                bound
            );
        }
    }

    // ── Test 3: different seeds → different outputs ───────────────────────────

    #[test]
    fn test_sample_noise_different_seeds() {
        let a = sample_gaussian_noise(50, 0.01, 3.0, 1);
        let b = sample_gaussian_noise(50, 0.01, 3.0, 2);
        assert_ne!(a, b);
    }

    // ── Test 4: inject_additive_noise changes params ──────────────────────────

    #[test]
    fn test_inject_additive_changes_params() {
        let mut params = vec![1.0f32; 100];
        let original = params.clone();
        inject_additive_noise(&mut params, 0.01, 1.0, 3.0, 42).unwrap();
        assert_ne!(params, original);
    }

    // ── Test 5: probability=0 → no modification ───────────────────────────────

    #[test]
    fn test_inject_additive_prob_zero() {
        let mut params = vec![1.0f32; 100];
        let original = params.clone();
        let modified = inject_additive_noise(&mut params, 0.01, 0.0, 3.0, 42).unwrap();
        assert_eq!(modified, 0);
        assert_eq!(params, original);
    }

    // ── Test 6: probability=1 → all modified ─────────────────────────────────

    #[test]
    fn test_inject_additive_prob_one() {
        let mut params = vec![1.0f32; 100];
        let modified = inject_additive_noise(&mut params, 0.01, 1.0, 3.0, 42).unwrap();
        assert_eq!(modified, 100);
    }

    // ── Test 7: std=0 → values unchanged ────────────────────────────────────

    #[test]
    fn test_inject_additive_zero_std() {
        let mut params = vec![1.0f32; 50];
        let original = params.clone();
        inject_additive_noise(&mut params, 0.0, 1.0, 3.0, 42).unwrap();
        for (a, b) in params.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6, "expected no change with std=0");
        }
    }

    // ── Test 8: inject_multiplicative_noise has multiplicative effect ─────────

    #[test]
    fn test_inject_multiplicative_effect() {
        let original = vec![2.0f32; 100];
        let mut params = original.clone();
        inject_multiplicative_noise(&mut params, 0.1, 1.0, 3.0, 99).unwrap();
        // With std=0.1 the params should be ~ 2*(1 ± 0.1), not exactly 2
        let max_diff = params
            .iter()
            .zip(original.iter())
            .map(|(p, o)| (p - o).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff > 0.0, "multiplicative noise should change params");
        // Each modified value should still be close to the original * factor
        for (&p, &o) in params.iter().zip(original.iter()) {
            let factor = p / o;
            assert!(
                (factor - 1.0).abs() <= 0.35,
                "factor {factor} out of reasonable range"
            );
        }
    }

    // ── Test 9: perturb_rotations non-multiple-of-4 → Err ───────────────────

    #[test]
    fn test_perturb_rotations_bad_length() {
        let mut q = vec![0.0f32, 0.0, 0.0, 1.0, 0.0];
        let result = perturb_rotations(&mut q, 0.01, 1.0, 42);
        assert!(
            matches!(result, Err(NoiseInjectionError::LengthMismatch { .. })),
            "expected LengthMismatch, got {:?}",
            result
        );
    }

    // ── Test 10: perturb_rotations probability=0 → no change ─────────────────

    #[test]
    fn test_perturb_rotations_prob_zero() {
        let mut q = make_unit_quaternions(10);
        let original = q.clone();
        let modified = perturb_rotations(&mut q, 0.1, 0.0, 42).unwrap();
        assert_eq!(modified, 0);
        assert_eq!(q, original);
    }

    // ── Test 11: perturb_rotations → unit quaternions ────────────────────────

    #[test]
    fn test_perturb_rotations_unit_norm() {
        let mut q = make_unit_quaternions(50);
        perturb_rotations(&mut q, 0.1, 1.0, 77).unwrap();
        for i in 0..50 {
            let base = i * 4;
            let norm =
                (q[base].powi(2) + q[base + 1].powi(2) + q[base + 2].powi(2) + q[base + 3].powi(2))
                    .sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "quaternion {i} norm {norm} != 1");
        }
    }

    // ── Test 12: NoiseSchedule::Constant ─────────────────────────────────────

    #[test]
    fn test_schedule_constant() {
        let sched = NoiseSchedule::Constant;
        assert_eq!(sched.scale_at(0), 1.0);
        assert_eq!(sched.scale_at(1000), 1.0);
        assert_eq!(sched.scale_at(usize::MAX / 2), 1.0);
    }

    // ── Test 13: NoiseSchedule::Linear decreases ─────────────────────────────

    #[test]
    fn test_schedule_linear_decreases() {
        let sched = NoiseSchedule::Linear {
            start_scale: 1.0,
            end_scale: 0.0,
            total_steps: 100,
        };
        let s0 = sched.scale_at(0);
        let s50 = sched.scale_at(50);
        let s99 = sched.scale_at(99);
        assert!(s0 > s50, "scale should decrease: {s0} > {s50}");
        assert!(s50 > s99, "scale should decrease: {s50} > {s99}");
    }

    // ── Test 14: NoiseSchedule::Linear at total_steps → end_scale ────────────

    #[test]
    fn test_schedule_linear_end() {
        let end = 0.1f32;
        let sched = NoiseSchedule::Linear {
            start_scale: 1.0,
            end_scale: end,
            total_steps: 100,
        };
        let scale = sched.scale_at(100);
        assert!(
            (scale - end).abs() < 1e-5,
            "scale at total_steps should be end_scale, got {scale}"
        );
        let scale_over = sched.scale_at(200);
        assert!(
            (scale_over - end).abs() < 1e-5,
            "scale past total_steps should be end_scale, got {scale_over}"
        );
    }

    // ── Test 15: NoiseSchedule::Cosine endpoints ──────────────────────────────

    #[test]
    fn test_schedule_cosine_endpoints() {
        let start = 1.0f32;
        let end = 0.1f32;
        let total = 100usize;
        let sched = NoiseSchedule::Cosine {
            start_scale: start,
            end_scale: end,
            total_steps: total,
        };
        let s0 = sched.scale_at(0);
        let s_end = sched.scale_at(total);
        assert!(
            (s0 - start).abs() < 1e-5,
            "at step=0 should be start_scale, got {s0}"
        );
        assert!(
            (s_end - end).abs() < 1e-5,
            "at step=total should be end_scale, got {s_end}"
        );
    }

    // ── Test 16: NoiseSchedule::StepDecay ────────────────────────────────────

    #[test]
    fn test_schedule_step_decay() {
        let sched = NoiseSchedule::StepDecay {
            initial_scale: 1.0,
            step_size: 10,
            decay_factor: 0.5,
        };
        let s0 = sched.scale_at(0);
        let s10 = sched.scale_at(10);
        let s20 = sched.scale_at(20);
        assert!(
            (s0 - 1.0).abs() < 1e-5,
            "initial scale should be 1.0, got {s0}"
        );
        assert!((s10 - 0.5).abs() < 1e-5, "after 1 decay: 0.5, got {s10}");
        assert!((s20 - 0.25).abs() < 1e-5, "after 2 decays: 0.25, got {s20}");
    }

    // ── Test 17: NoiseConfig::new valid std → Ok ──────────────────────────────

    #[test]
    fn test_config_new_valid() {
        let cfg = NoiseConfig::new(NoiseTarget::Positions, 0.01);
        assert!(cfg.is_ok());
    }

    // ── Test 18: NoiseConfig::new std < 0 → Err ──────────────────────────────

    #[test]
    fn test_config_new_negative_std() {
        let cfg = NoiseConfig::new(NoiseTarget::Positions, -0.01);
        assert!(
            matches!(cfg, Err(NoiseInjectionError::InvalidConfig(_))),
            "expected InvalidConfig, got {:?}",
            cfg
        );
    }

    // ── Test 19: NoiseConfig::with_probability p > 1 → Err ───────────────────

    #[test]
    fn test_config_with_probability_invalid() {
        let cfg = NoiseConfig::new(NoiseTarget::Positions, 0.01)
            .unwrap()
            .with_probability(1.5);
        assert!(
            matches!(cfg, Err(NoiseInjectionError::InvalidConfig(_))),
            "expected InvalidConfig for p=1.5, got {:?}",
            cfg
        );
    }

    // ── Test 20: NoiseInjector effective_std = std * scale_at(step) ──────────

    #[test]
    fn test_injector_effective_std() {
        let cfg = NoiseConfig {
            target: NoiseTarget::Positions,
            std: 0.01,
            probability: 1.0,
            schedule: NoiseSchedule::Linear {
                start_scale: 1.0,
                end_scale: 0.0,
                total_steps: 100,
            },
            clip_sigma: 3.0,
            seed: 42,
        };
        let mut injector = NoiseInjector::new(cfg).unwrap();
        // At step 0: scale = 1.0, eff_std = 0.01
        assert!((injector.effective_std() - 0.01).abs() < 1e-6);
        // Simulate advancing to step 50
        injector.current_step = 50;
        let expected = 0.01 * 0.5;
        assert!(
            (injector.effective_std() - expected).abs() < 1e-5,
            "effective_std at step 50 should be {expected}, got {}",
            injector.effective_std()
        );
    }

    // ── Test 21: NoiseInjector::step increments current_step ─────────────────

    #[test]
    fn test_injector_step_increments() {
        let cfg = NoiseConfig::new(NoiseTarget::Positions, 0.001).unwrap();
        let mut injector = NoiseInjector::new(cfg).unwrap();
        let mut positions = vec![0.0f32; 30];
        let mut dummy_log_scales: Vec<f32> = Vec::new();
        let mut dummy_rotations: Vec<f32> = Vec::new();
        let mut dummy_colors: Vec<f32> = Vec::new();
        let mut dummy_opacities: Vec<f32> = Vec::new();

        injector
            .step(
                &mut positions,
                &mut dummy_log_scales,
                &mut dummy_rotations,
                &mut dummy_colors,
                &mut dummy_opacities,
            )
            .unwrap();

        assert_eq!(injector.current_step, 1);
        assert_eq!(injector.num_applications, 1);
    }

    // ── Test 22: NoiseInjector::step All modifies multiple buffers ────────────

    #[test]
    fn test_injector_step_all_targets() {
        let cfg = NoiseConfig {
            target: NoiseTarget::All,
            std: 0.01,
            probability: 1.0,
            schedule: NoiseSchedule::Constant,
            clip_sigma: 3.0,
            seed: 42,
        };
        let mut injector = NoiseInjector::new(cfg).unwrap();

        let mut positions = vec![1.0f32; 30];
        // log_scales must be non-zero so multiplicative noise is observable
        let mut log_scales = vec![1.0f32; 30];
        let mut rotations = make_unit_quaternions(10);
        let mut colors = vec![0.5f32; 30];
        let mut opacities = vec![0.0f32; 10];

        let positions_orig = positions.clone();
        let log_scales_orig = log_scales.clone();
        let rotations_orig = rotations.clone();
        let colors_orig = colors.clone();
        let opacities_orig = opacities.clone();

        let total_mod = injector
            .step(
                &mut positions,
                &mut log_scales,
                &mut rotations,
                &mut colors,
                &mut opacities,
            )
            .unwrap();

        assert!(total_mod > 0, "All target should modify elements");
        assert_ne!(positions, positions_orig, "positions should be modified");
        assert_ne!(log_scales, log_scales_orig, "log_scales should be modified");
        assert_ne!(colors, colors_orig, "colors should be modified");
        assert_ne!(opacities, opacities_orig, "opacities should be modified");
        // Rotations: values may differ because of perturbation
        let rot_changed = rotations
            .iter()
            .zip(rotations_orig.iter())
            .any(|(a, b)| (a - b).abs() > 1e-8);
        assert!(rot_changed, "rotations should be perturbed");
    }

    // ── Test 23: analyze_noise: noise_to_signal_ratio > 0 ────────────────────

    #[test]
    fn test_analyze_noise_nsr() {
        let original = vec![1.0f32; 100];
        let mut noisy = original.clone();
        inject_additive_noise(&mut noisy, 0.1, 1.0, 0.0, 42).unwrap();
        let analysis = analyze_noise(&original, &noisy).unwrap();
        assert!(
            analysis.noise_to_signal_ratio > 0.0,
            "NSR should be > 0 when noise was applied, got {}",
            analysis.noise_to_signal_ratio
        );
        assert!(analysis.num_injected > 0);
        assert_eq!(analysis.num_total, 100);
    }

    // ── Bonus: analyze_noise length mismatch ─────────────────────────────────

    #[test]
    fn test_analyze_noise_length_mismatch() {
        let original = vec![1.0f32; 10];
        let noisy = vec![1.0f32; 5];
        let result = analyze_noise(&original, &noisy);
        assert!(
            matches!(result, Err(NoiseInjectionError::LengthMismatch { .. })),
            "expected LengthMismatch, got {:?}",
            result
        );
    }

    // ── Bonus: analyze_noise empty input → Err ────────────────────────────────

    #[test]
    fn test_analyze_noise_empty() {
        let result = analyze_noise(&[], &[]);
        assert!(
            matches!(result, Err(NoiseInjectionError::EmptyInput)),
            "expected EmptyInput, got {:?}",
            result
        );
    }

    // ── Bonus: inject_additive_noise empty input → Err ───────────────────────

    #[test]
    fn test_inject_additive_empty() {
        let mut params: Vec<f32> = Vec::new();
        let result = inject_additive_noise(&mut params, 0.01, 1.0, 3.0, 42);
        assert!(
            matches!(result, Err(NoiseInjectionError::EmptyInput)),
            "expected EmptyInput, got {:?}",
            result
        );
    }
}
