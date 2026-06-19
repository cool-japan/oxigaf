//! # Adaptive Timestep Sampling for Diffusion Training
//!
//! Implements adaptive timestep sampling strategies for more efficient diffusion model
//! training. Instead of uniform timestep sampling, these strategies focus training on
//! timesteps where the model can learn most effectively.
//!
//! ## Strategies
//!
//! - **Log-normal sampling** (Karras et al. EDM): Sample timesteps from a log-normal
//!   distribution in sigma space to focus on perceptually important noise levels.
//! - **Min-SNR weighting** (Hang et al.): Weight loss by min(SNR, γ) / SNR to balance
//!   training across noisy and clean timesteps.
//! - **Continuous time sampling**: Sample t uniformly in [0, 1] for flow matching / DDPM.
//! - **Importance sampling**: Sample timesteps proportional to expected loss magnitude.
//!
//! ## References
//!
//! - Karras et al. (2022), "Elucidating the Design Space of Diffusion-Based Generative Models"
//! - Hang et al. (2023), "Efficient Diffusion Training via Min-SNR Weighting Strategy"

use std::f32::consts::PI;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise from adaptive sampling operations.
#[derive(Debug, Error, PartialEq)]
pub enum AdaptiveSamplingError {
    /// Log-normal mean is not a finite number.
    #[error("Invalid mean {mean}: log-normal mean must be finite")]
    InvalidMean { mean: f32 },

    /// Log-normal std is not positive.
    #[error("Invalid std {std}: log-normal std must be > 0")]
    InvalidStd { std: f32 },

    /// Timestep count must be at least 1.
    #[error("Invalid timestep count {n}: must be >= 1")]
    InvalidTimestepCount { n: usize },

    /// Min-SNR gamma must be positive.
    #[error("Invalid gamma {gamma} for Min-SNR: must be > 0")]
    InvalidGamma { gamma: f32 },

    /// Timestep index is out of range.
    #[error("Timestep {t} out of range [0, {max_t}]")]
    TimestepOutOfRange { t: usize, max_t: usize },

    /// All sampling weights are zero; cannot sample.
    #[error("Empty weight distribution: all weights are zero")]
    ZeroWeights,
}

// ─────────────────────────────────────────────────────────────────────────────
// Random number generation (xorshift64 + Box-Muller, no `rand` crate)
// ─────────────────────────────────────────────────────────────────────────────

/// Xorshift64 pseudo-random number generator.
///
/// Advances `state` in-place and returns the new value.  The state is never
/// allowed to be zero (a fixed-point of the algorithm).
#[inline]
pub fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    if *state == 0 {
        *state = 1;
    }
    *state
}

/// Generates a uniform float in `[0, 1)` using 53 mantissa bits.
#[inline]
pub fn uniform_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 11) as f32 / (1u64 << 53) as f32
}

/// Generates a standard normal sample via Box-Muller transform.
///
/// Returns a single `f32` sample from N(0, 1).
pub fn box_muller(state: &mut u64) -> f32 {
    let u1 = uniform_f32(state);
    let u2 = uniform_f32(state);

    (-2.0_f32 * (u1 + 1e-10_f32).ln()).sqrt() * (2.0_f32 * PI * u2).cos()
}

/// Samples from a log-normal distribution `LogNormal(mean, std)`.
///
/// The returned value is always positive.
pub fn sample_log_normal(mean: f32, std: f32, state: &mut u64) -> f32 {
    let z = box_muller(state) * std + mean;
    z.exp()
}

/// Samples from a logit-normal distribution.
///
/// The underlying normal sample is squashed through the sigmoid, yielding a
/// value in `(0, 1)`.  Used for continuous-time flow-matching schedules.
pub fn sample_logit_normal(mean: f32, std: f32, state: &mut u64) -> f32 {
    let z = box_muller(state) * std + mean;
    // sigmoid(z) = 1 / (1 + exp(-z))
    1.0_f32 / (1.0_f32 + (-z).exp())
}

/// Draws a discrete index from a categorical distribution defined by `weights`.
///
/// Uses the inverse-CDF method with a linear scan through the cumulative sum.
/// Returns `Err(AdaptiveSamplingError::ZeroWeights)` when all weights are zero.
pub fn sample_cdf(weights: &[f32], state: &mut u64) -> Result<usize, AdaptiveSamplingError> {
    let total: f32 = weights.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return Err(AdaptiveSamplingError::ZeroWeights);
    }

    let threshold = uniform_f32(state) * total;
    let mut cumulative = 0.0_f32;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += w;
        if cumulative > threshold {
            return Ok(i);
        }
    }
    // Numerical rounding: fall back to last non-zero weight index.
    let last = weights
        .iter()
        .enumerate()
        .rev()
        .find(|(_, &w)| w > 0.0)
        .map(|(i, _)| i)
        .ok_or(AdaptiveSamplingError::ZeroWeights)?;
    Ok(last)
}

// ─────────────────────────────────────────────────────────────────────────────
// LogNormalSampler — Karras et al. EDM
// ─────────────────────────────────────────────────────────────────────────────

/// Log-normal timestep sampler from the EDM paper (Karras et al. 2022).
///
/// Samples noise sigma values from a log-normal distribution and converts them
/// to discrete timestep indices.  The default hyper-parameters match the EDM
/// paper's recommended values.
#[derive(Debug, Clone)]
pub struct LogNormalSampler {
    /// Mean of the underlying normal in log-sigma space (default: −1.2).
    pub p_mean: f32,
    /// Std of the underlying normal in log-sigma space (default: 1.2).
    pub p_std: f32,
    /// Minimum noise sigma (default: 0.002).
    pub sigma_min: f32,
    /// Maximum noise sigma (default: 80.0).
    pub sigma_max: f32,
}

impl Default for LogNormalSampler {
    fn default() -> Self {
        Self {
            p_mean: -1.2,
            p_std: 1.2,
            sigma_min: 0.002,
            sigma_max: 80.0,
        }
    }
}

impl LogNormalSampler {
    /// Validates sampler parameters.
    pub fn validate(&self) -> Result<(), AdaptiveSamplingError> {
        if !self.p_mean.is_finite() {
            return Err(AdaptiveSamplingError::InvalidMean { mean: self.p_mean });
        }
        if self.p_std <= 0.0 || !self.p_std.is_finite() {
            return Err(AdaptiveSamplingError::InvalidStd { std: self.p_std });
        }
        Ok(())
    }

    /// Samples a noise sigma value, clamped to `[sigma_min, sigma_max]`.
    pub fn sample_sigma(&self, state: &mut u64) -> f32 {
        let sigma = sample_log_normal(self.p_mean, self.p_std, state);
        sigma.clamp(self.sigma_min, self.sigma_max)
    }

    /// Maps a sigma value to a discrete timestep index in `[0, max_timesteps − 1]`.
    pub fn sigma_to_timestep(&self, sigma: f32, max_timesteps: usize) -> usize {
        if max_timesteps == 0 {
            return 0;
        }
        let range = self.sigma_max - self.sigma_min;
        let frac = if range > 0.0 {
            (sigma - self.sigma_min) / range
        } else {
            0.0
        };
        let t = (frac * max_timesteps as f32).round() as isize;
        t.clamp(0, (max_timesteps as isize) - 1) as usize
    }

    /// Samples a discrete timestep via the log-normal sigma distribution.
    pub fn sample_timestep_lognormal(&self, max_timesteps: usize, state: &mut u64) -> usize {
        let sigma = self.sample_sigma(state);
        self.sigma_to_timestep(sigma, max_timesteps)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MinSnrWeighter — Hang et al. 2023
// ─────────────────────────────────────────────────────────────────────────────

/// Min-SNR loss weighting strategy from Hang et al. (2023).
///
/// Computes per-timestep weights `min(SNR_t, γ) / SNR_t` to balance training
/// across high-SNR (clean) and low-SNR (noisy) timesteps.
#[derive(Debug, Clone)]
pub struct MinSnrWeighter {
    /// SNR clamping value γ (default: 5.0).
    pub gamma: f32,
    /// Total number of diffusion timesteps (default: 1000).
    pub max_timesteps: usize,
    /// Noise schedule: "linear" or "cosine" (default: "cosine").
    pub schedule: String,
}

impl Default for MinSnrWeighter {
    fn default() -> Self {
        Self {
            gamma: 5.0,
            max_timesteps: 1000,
            schedule: "cosine".to_string(),
        }
    }
}

impl MinSnrWeighter {
    /// Validates weighter parameters.
    pub fn validate(&self) -> Result<(), AdaptiveSamplingError> {
        if self.gamma <= 0.0 || !self.gamma.is_finite() {
            return Err(AdaptiveSamplingError::InvalidGamma { gamma: self.gamma });
        }
        Ok(())
    }

    /// Computes ᾱ_t (cumulative product of (1 − β)) for the given timestep.
    fn alpha_bar(&self, t: usize) -> f32 {
        match self.schedule.as_str() {
            "linear" => {
                // Simple linear schedule: ᾱ_t decreases linearly from 1 to 0.
                let frac = t as f32 / self.max_timesteps.max(1) as f32;
                (1.0 - frac).max(0.0)
            }
            _ => {
                // Cosine schedule (Nichol & Dhariwal 2021).
                let s = 0.008_f32;
                let t_frac = t as f32 / self.max_timesteps.max(1) as f32;
                let arg = (t_frac + s) / (1.0 + s) * PI / 2.0;
                arg.cos().powi(2)
            }
        }
    }

    /// Computes the signal-to-noise ratio at timestep `t`.
    ///
    /// SNR_t = ᾱ_t / (1 − ᾱ_t).
    pub fn compute_snr(&self, t: usize) -> Result<f32, AdaptiveSamplingError> {
        let max_t = self.max_timesteps.saturating_sub(1);
        if t > max_t {
            return Err(AdaptiveSamplingError::TimestepOutOfRange { t, max_t });
        }
        let ab = self.alpha_bar(t);
        let snr = ab / (1.0 - ab).max(1e-8);
        Ok(snr)
    }

    /// Computes the Min-SNR loss weight at timestep `t`.
    ///
    /// weight = min(SNR_t, γ) / max(SNR_t, ε)
    pub fn compute_weight(&self, t: usize) -> Result<f32, AdaptiveSamplingError> {
        let snr = self.compute_snr(t)?;
        let weight = snr.min(self.gamma) / snr.max(1e-8);
        Ok(weight)
    }

    /// Computes Min-SNR weights for every timestep `0..max_timesteps`.
    pub fn compute_all_weights(&self) -> Result<Vec<f32>, AdaptiveSamplingError> {
        (0..self.max_timesteps)
            .map(|t| self.compute_weight(t))
            .collect()
    }

    /// Samples a timestep proportional to its Min-SNR weight.
    pub fn sample_timestep_minsnr(&self, state: &mut u64) -> Result<usize, AdaptiveSamplingError> {
        let weights = self.compute_all_weights()?;
        sample_cdf(&weights, state)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// UniformSampler
// ─────────────────────────────────────────────────────────────────────────────

/// Uniform timestep sampler with configurable range.
#[derive(Debug, Clone)]
pub struct UniformSampler {
    /// Total diffusion timesteps T (default: 1000).
    pub max_timesteps: usize,
    /// Minimum timestep index (default: 0).
    pub low: usize,
    /// Maximum timestep index, inclusive (default: 999).
    pub high: usize,
}

impl Default for UniformSampler {
    fn default() -> Self {
        Self {
            max_timesteps: 1000,
            low: 0,
            high: 999,
        }
    }
}

impl UniformSampler {
    /// Samples a timestep uniformly from `[low, high]`, clamped to `[0, max_timesteps − 1]`.
    pub fn sample_timestep_uniform(&self, state: &mut u64) -> usize {
        let range = self.high.saturating_sub(self.low) + 1;
        let t = xorshift64(state) as usize % range + self.low;
        t.min(self.max_timesteps.saturating_sub(1))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ContinuousSampler — flow matching / DDPM continuous time
// ─────────────────────────────────────────────────────────────────────────────

/// Continuous-time sampler for flow-matching and continuous DDPM.
///
/// Samples t ∈ [t_min, t_max], optionally from a logit-normal distribution.
#[derive(Debug, Clone)]
pub struct ContinuousSampler {
    /// Minimum time value (default: 0.0).
    pub t_min: f32,
    /// Maximum time value (default: 1.0).
    pub t_max: f32,
    /// Use logit-normal distribution instead of uniform (default: false).
    pub logit_normal: bool,
    /// Mean of the underlying normal for logit-normal (default: 0.0).
    pub lognormal_mean: f32,
    /// Std of the underlying normal for logit-normal (default: 1.0).
    pub lognormal_std: f32,
}

impl Default for ContinuousSampler {
    fn default() -> Self {
        Self {
            t_min: 0.0,
            t_max: 1.0,
            logit_normal: false,
            lognormal_mean: 0.0,
            lognormal_std: 1.0,
        }
    }
}

impl ContinuousSampler {
    /// Samples a continuous time value in `[t_min, t_max]`.
    ///
    /// When `logit_normal` is enabled the sample is drawn from a logit-normal
    /// distribution (sigmoid of a normal), then linearly rescaled into the
    /// requested interval.
    pub fn sample_t(&self, state: &mut u64) -> f32 {
        let unit = if self.logit_normal {
            sample_logit_normal(self.lognormal_mean, self.lognormal_std, state)
        } else {
            uniform_f32(state)
        };
        // Rescale from [0, 1] to [t_min, t_max].
        self.t_min + unit * (self.t_max - self.t_min)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ImportanceSampler
// ─────────────────────────────────────────────────────────────────────────────

/// Importance-weighted timestep sampler.
///
/// Samples timesteps proportional to per-timestep weights, e.g. based on
/// observed loss magnitudes so that harder timesteps are trained more often.
#[derive(Debug, Clone)]
pub struct ImportanceSampler {
    /// Per-timestep sampling weights (unnormalized).
    pub weights: Vec<f32>,
    /// Total number of timesteps.
    pub max_timesteps: usize,
}

impl ImportanceSampler {
    /// Creates a sampler with uniform weights over `max_timesteps`.
    pub fn from_uniform(max_timesteps: usize) -> Self {
        Self {
            weights: vec![1.0; max_timesteps],
            max_timesteps,
        }
    }

    /// Creates a sampler whose weights are proportional to the provided per-timestep losses.
    ///
    /// Returns `Err(ZeroWeights)` when all loss values are zero.
    pub fn from_loss_history(
        loss_history: &[f32],
    ) -> Result<ImportanceSampler, AdaptiveSamplingError> {
        let total: f32 = loss_history.iter().map(|l| l.max(0.0)).sum();
        if total <= 0.0 {
            return Err(AdaptiveSamplingError::ZeroWeights);
        }
        let weights: Vec<f32> = loss_history.iter().map(|l| l.max(0.0)).collect();
        let max_timesteps = weights.len();
        Ok(ImportanceSampler {
            weights,
            max_timesteps,
        })
    }

    /// Samples a timestep index proportional to its weight.
    pub fn sample(&self, state: &mut u64) -> Result<usize, AdaptiveSamplingError> {
        sample_cdf(&self.weights, state)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TimestepHistory — online tracking of per-timestep loss
// ─────────────────────────────────────────────────────────────────────────────

/// Online history of sampled timesteps and observed losses.
///
/// Keeps an EMA (exponential moving average) of the loss for each timestep, which
/// can drive an `ImportanceSampler` for curriculum-style training.
#[derive(Debug, Clone)]
pub struct TimestepHistory {
    /// Number of times each timestep has been sampled.
    pub counts: Vec<usize>,
    /// EMA loss estimate for each timestep.
    pub losses: Vec<f32>,
    /// Total number of timesteps.
    pub max_timesteps: usize,
    /// EMA decay factor (closer to 1 → slower update).
    pub ema_decay: f32,
}

impl TimestepHistory {
    /// Creates a new history for `max_timesteps` timesteps with the given EMA decay.
    pub fn new(max_timesteps: usize, ema_decay: f32) -> Self {
        Self {
            counts: vec![0; max_timesteps],
            losses: vec![0.0; max_timesteps],
            max_timesteps,
            ema_decay,
        }
    }

    /// Records an observation: increments the count and updates the EMA loss.
    ///
    /// Silently ignores `timestep >= max_timesteps` to avoid panics in edge cases.
    pub fn record(&mut self, timestep: usize, loss: f32) {
        if timestep >= self.max_timesteps {
            return;
        }
        self.counts[timestep] = self.counts[timestep].saturating_add(1);
        self.losses[timestep] =
            self.ema_decay * self.losses[timestep] + (1.0 - self.ema_decay) * loss;
    }

    /// Computes summary statistics over the recorded history.
    pub fn compute_stats(&self) -> SamplingStats {
        let total_count: usize = self.counts.iter().sum();
        let n = self.max_timesteps as f32;

        // Weighted mean timestep (weight = sampling frequency).
        let mean_timestep = if total_count == 0 {
            n / 2.0
        } else {
            self.counts
                .iter()
                .enumerate()
                .map(|(t, &c)| t as f32 * c as f32)
                .sum::<f32>()
                / total_count as f32
        };

        // Weighted std.
        let variance = if total_count == 0 {
            0.0
        } else {
            self.counts
                .iter()
                .enumerate()
                .map(|(t, &c)| {
                    let diff = t as f32 - mean_timestep;
                    diff * diff * c as f32
                })
                .sum::<f32>()
                / total_count as f32
        };
        let std_timestep = variance.sqrt();

        // Most / least sampled.
        let most_sampled = self
            .counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);

        let least_sampled = self
            .counts
            .iter()
            .enumerate()
            .min_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);

        let sampled_count = self.counts.iter().filter(|&&c| c > 0).count();
        let coverage = if self.max_timesteps == 0 {
            0.0
        } else {
            sampled_count as f32 / self.max_timesteps as f32
        };

        SamplingStats {
            mean_timestep,
            std_timestep,
            most_sampled,
            least_sampled,
            coverage,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SamplingStats
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics describing a timestep sampling distribution.
#[derive(Debug, Clone)]
pub struct SamplingStats {
    /// Weighted mean of sampled timesteps.
    pub mean_timestep: f32,
    /// Weighted standard deviation of sampled timesteps.
    pub std_timestep: f32,
    /// Timestep sampled most frequently.
    pub most_sampled: usize,
    /// Timestep sampled least frequently.
    pub least_sampled: usize,
    /// Fraction of timesteps sampled at least once (0.0 – 1.0).
    pub coverage: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Free functions
// ─────────────────────────────────────────────────────────────────────────────

/// Convenience wrapper: compute Min-SNR sampling weights for all timesteps.
///
/// Returns a `Vec<f32>` of length `max_timesteps` whose entries are
/// `min(SNR_t, gamma) / SNR_t`.
pub fn compute_snr_sampling_weights(
    max_timesteps: usize,
    gamma: f32,
    schedule: &str,
) -> Result<Vec<f32>, AdaptiveSamplingError> {
    let weighter = MinSnrWeighter {
        gamma,
        max_timesteps,
        schedule: schedule.to_string(),
    };
    weighter.validate()?;
    weighter.compute_all_weights()
}

/// Maps a discrete timestep to a noise sigma via linear interpolation.
///
/// `sigma = sigma_min + (sigma_max − sigma_min) * t / (max_timesteps − 1)`
pub fn timestep_to_sigma(t: usize, max_timesteps: usize, sigma_min: f32, sigma_max: f32) -> f32 {
    if max_timesteps <= 1 {
        return sigma_min;
    }
    sigma_min + (sigma_max - sigma_min) * t as f32 / (max_timesteps as f32 - 1.0)
}

/// Converts a noise sigma to cumulative ᾱ using the simplified EDM parameterisation.
///
/// `alpha_bar = 1 / (1 + sigma²)`
pub fn sigma_to_alpha_bar(sigma: f32) -> f32 {
    1.0 / (1.0 + sigma * sigma)
}

/// Empirically estimates the SNR range produced by a `LogNormalSampler`.
///
/// Draws `n_samples` sigma values and returns `(min_snr, max_snr)`.
pub fn compute_effective_snr_range(
    sampler: &LogNormalSampler,
    n_samples: usize,
    seed: u64,
) -> (f32, f32) {
    let mut state = if seed == 0 { 1 } else { seed };
    let mut min_snr = f32::MAX;
    let mut max_snr = f32::MIN;

    for _ in 0..n_samples {
        let sigma = sampler.sample_sigma(&mut state);
        let ab = sigma_to_alpha_bar(sigma);
        let snr = ab / (1.0 - ab).max(1e-8);
        if snr < min_snr {
            min_snr = snr;
        }
        if snr > max_snr {
            max_snr = snr;
        }
    }

    if n_samples == 0 {
        (0.0, 0.0)
    } else {
        (min_snr, max_snr)
    }
}

/// Formats `SamplingStats` into a compact human-readable string.
pub fn format_sampling_stats(stats: &SamplingStats) -> String {
    format!(
        "TimestepSampling: mean_t={:.1}, std={:.1}, coverage={:.1}%",
        stats.mean_timestep,
        stats.std_timestep,
        stats.coverage * 100.0,
    )
}

/// Aggregates per-timestep sample counts into `n_bins` histogram bins.
///
/// Returns a `Vec<usize>` of length `n_bins`.
pub fn compute_timestep_histogram(history: &TimestepHistory, n_bins: usize) -> Vec<usize> {
    if n_bins == 0 || history.max_timesteps == 0 {
        return Vec::new();
    }
    let mut hist = vec![0usize; n_bins];
    for (t, &count) in history.counts.iter().enumerate() {
        // Map timestep t to bin index.
        let bin = (t * n_bins / history.max_timesteps).min(n_bins - 1);
        hist[bin] = hist[bin].saturating_add(count);
    }
    hist
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── xorshift64 ───────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift64_state_changes() {
        let mut state = 12345u64;
        let v1 = xorshift64(&mut state);
        let v2 = xorshift64(&mut state);
        assert_ne!(v1, v2, "consecutive xorshift64 outputs must differ");
    }

    #[test]
    fn test_xorshift64_never_zero() {
        let mut state = 1u64;
        for _ in 0..10_000 {
            let v = xorshift64(&mut state);
            assert_ne!(v, 0, "xorshift64 must never produce zero");
        }
    }

    #[test]
    fn test_xorshift64_zero_seed_fixed() {
        // A zero seed should be corrected to 1 on first call.
        let mut state = 0u64;
        // Directly force the correction path: the guard runs *after* the shifts,
        // but if we start at 0 the first shift still yields 0 — the guard catches it.
        xorshift64(&mut state); // after this state must be non-zero
        assert_ne!(state, 0);
    }

    // ── uniform_f32 ──────────────────────────────────────────────────────────

    #[test]
    fn test_uniform_f32_range() {
        let mut state = 42u64;
        for _ in 0..10_000 {
            let u = uniform_f32(&mut state);
            assert!(
                (0.0..1.0).contains(&u),
                "uniform_f32 must be in [0, 1), got {u}"
            );
        }
    }

    #[test]
    fn test_uniform_f32_mean_near_half() {
        let mut state = 1u64;
        let n = 100_000;
        let sum: f32 = (0..n).map(|_| uniform_f32(&mut state)).sum();
        let mean = sum / n as f32;
        assert!(
            (mean - 0.5).abs() < 0.01,
            "mean should be near 0.5, got {mean}"
        );
    }

    // ── box_muller ───────────────────────────────────────────────────────────

    #[test]
    fn test_box_muller_mean_near_zero() {
        let mut state = 99u64;
        let n = 100_000;
        let sum: f32 = (0..n).map(|_| box_muller(&mut state)).sum();
        let mean = sum / n as f32;
        assert!(
            mean.abs() < 0.02,
            "Box-Muller mean should be near 0, got {mean}"
        );
    }

    #[test]
    fn test_box_muller_std_near_one() {
        let mut state = 7u64;
        let n = 100_000;
        let samples: Vec<f32> = (0..n).map(|_| box_muller(&mut state)).collect();
        let mean = samples.iter().sum::<f32>() / n as f32;
        let var = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let std = var.sqrt();
        assert!(
            (std - 1.0).abs() < 0.02,
            "Box-Muller std should be near 1, got {std}"
        );
    }

    // ── sample_log_normal ────────────────────────────────────────────────────

    #[test]
    fn test_sample_log_normal_positive() {
        let mut state = 314u64;
        for _ in 0..1_000 {
            let v = sample_log_normal(-1.2, 1.2, &mut state);
            assert!(v > 0.0, "log-normal must always be positive");
        }
    }

    #[test]
    fn test_sample_log_normal_bulk_within_expected_range() {
        // 3σ rule: ~99.7% should be within [exp(mean-3*std), exp(mean+3*std)].
        let mean = -1.2_f32;
        let std = 1.2_f32;
        let lo = (mean - 3.0 * std).exp();
        let hi = (mean + 3.0 * std).exp();

        let mut state = 271u64;
        let n = 10_000;
        let inside = (0..n)
            .filter(|_| {
                let v = sample_log_normal(mean, std, &mut state);
                v >= lo && v <= hi
            })
            .count();
        // Allow a little slack beyond the 3σ bound.
        assert!(
            inside as f32 / n as f32 > 0.99,
            "expected >99% within 3σ range, got {}%",
            inside as f32 / n as f32 * 100.0
        );
    }

    // ── sample_logit_normal ──────────────────────────────────────────────────

    #[test]
    fn test_sample_logit_normal_in_unit_interval() {
        let mut state = 1234u64;
        for _ in 0..1_000 {
            let v = sample_logit_normal(0.0, 1.0, &mut state);
            assert!(
                (0.0..=1.0).contains(&v),
                "logit-normal must be in [0, 1], got {v}"
            );
        }
    }

    // ── sample_cdf ───────────────────────────────────────────────────────────

    #[test]
    fn test_sample_cdf_uniform_weights_roughly_uniform() {
        let weights = vec![1.0_f32; 5];
        let mut state = 555u64;
        let n = 50_000;
        let mut counts = vec![0usize; 5];
        for _ in 0..n {
            let idx = sample_cdf(&weights, &mut state).expect("sample_cdf failed");
            counts[idx] += 1;
        }
        for &c in &counts {
            let frac = c as f32 / n as f32;
            assert!(
                (frac - 0.2).abs() < 0.02,
                "uniform weight should yield ~20% per bucket, got {:.2}%",
                frac * 100.0
            );
        }
    }

    #[test]
    fn test_sample_cdf_concentrated_weights() {
        // All weight on index 2.
        let weights = vec![0.0, 0.0, 1.0, 0.0];
        let mut state = 7u64;
        for _ in 0..100 {
            let idx = sample_cdf(&weights, &mut state).expect("should succeed");
            assert_eq!(idx, 2);
        }
    }

    #[test]
    fn test_sample_cdf_zero_weights_error() {
        let weights = vec![0.0_f32; 4];
        let mut state = 1u64;
        let result = sample_cdf(&weights, &mut state);
        assert_eq!(result, Err(AdaptiveSamplingError::ZeroWeights));
    }

    #[test]
    fn test_sample_cdf_known_probabilities() {
        // weights [1, 3]: P(0) = 0.25, P(1) = 0.75
        let weights = vec![1.0_f32, 3.0_f32];
        let mut state = 9999u64;
        let n = 100_000;
        let count_zero = (0..n)
            .filter(|_| sample_cdf(&weights, &mut state).unwrap_or(1) == 0)
            .count();
        let frac = count_zero as f32 / n as f32;
        assert!(
            (frac - 0.25).abs() < 0.02,
            "expected ~25% zeros, got {:.2}%",
            frac * 100.0
        );
    }

    // ── LogNormalSampler ─────────────────────────────────────────────────────

    #[test]
    fn test_lognormal_sampler_defaults() {
        let s = LogNormalSampler::default();
        assert_eq!(s.p_mean, -1.2);
        assert_eq!(s.p_std, 1.2);
        assert_eq!(s.sigma_min, 0.002);
        assert_eq!(s.sigma_max, 80.0);
    }

    #[test]
    fn test_lognormal_sampler_validate_ok() {
        let s = LogNormalSampler::default();
        assert!(s.validate().is_ok());
    }

    #[test]
    fn test_lognormal_sampler_validate_std_zero() {
        let s = LogNormalSampler {
            p_std: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            s.validate(),
            Err(AdaptiveSamplingError::InvalidStd { .. })
        ));
    }

    #[test]
    fn test_lognormal_sampler_validate_negative_std() {
        let s = LogNormalSampler {
            p_std: -0.5,
            ..Default::default()
        };
        assert!(matches!(
            s.validate(),
            Err(AdaptiveSamplingError::InvalidStd { .. })
        ));
    }

    #[test]
    fn test_lognormal_sampler_validate_infinite_mean() {
        let s = LogNormalSampler {
            p_mean: f32::INFINITY,
            ..Default::default()
        };
        assert!(matches!(
            s.validate(),
            Err(AdaptiveSamplingError::InvalidMean { .. })
        ));
    }

    #[test]
    fn test_lognormal_sampler_sample_sigma_in_range() {
        let s = LogNormalSampler::default();
        let mut state = 1_000_000u64;
        for _ in 0..1_000 {
            let sigma = s.sample_sigma(&mut state);
            assert!(
                sigma >= s.sigma_min && sigma <= s.sigma_max,
                "sigma {sigma} out of [{}, {}]",
                s.sigma_min,
                s.sigma_max
            );
        }
    }

    #[test]
    fn test_lognormal_sampler_sigma_to_timestep_min() {
        let s = LogNormalSampler::default();
        let t = s.sigma_to_timestep(s.sigma_min, 1000);
        assert_eq!(t, 0);
    }

    #[test]
    fn test_lognormal_sampler_sigma_to_timestep_max() {
        let s = LogNormalSampler::default();
        let t = s.sigma_to_timestep(s.sigma_max, 1000);
        assert_eq!(t, 999);
    }

    #[test]
    fn test_lognormal_sampler_timestep_valid_range() {
        let s = LogNormalSampler::default();
        let mut state = 77u64;
        for _ in 0..500 {
            let t = s.sample_timestep_lognormal(1000, &mut state);
            assert!(t < 1000, "timestep {t} must be < 1000");
        }
    }

    // ── MinSnrWeighter ───────────────────────────────────────────────────────

    #[test]
    fn test_minsnr_weighter_defaults() {
        let w = MinSnrWeighter::default();
        assert_eq!(w.gamma, 5.0);
        assert_eq!(w.max_timesteps, 1000);
        assert_eq!(w.schedule, "cosine");
    }

    #[test]
    fn test_minsnr_weighter_validate_gamma_zero_error() {
        let w = MinSnrWeighter {
            gamma: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            w.validate(),
            Err(AdaptiveSamplingError::InvalidGamma { .. })
        ));
    }

    #[test]
    fn test_minsnr_weighter_snr_t0_high() {
        let w = MinSnrWeighter::default();
        let snr = w.compute_snr(0).expect("SNR at t=0 should succeed");
        // At t=0 with cosine schedule ᾱ ≈ 1 → SNR should be very large.
        assert!(snr > 100.0, "SNR at t=0 should be high, got {snr}");
    }

    #[test]
    fn test_minsnr_weighter_snr_t_max_minus_1_low() {
        let w = MinSnrWeighter::default();
        let snr = w.compute_snr(999).expect("SNR at t=999 should succeed");
        assert!(snr < 1.0, "SNR at t=999 should be low, got {snr}");
    }

    #[test]
    fn test_minsnr_weighter_snr_out_of_range() {
        let w = MinSnrWeighter::default();
        let result = w.compute_snr(1000);
        assert!(matches!(
            result,
            Err(AdaptiveSamplingError::TimestepOutOfRange { .. })
        ));
    }

    #[test]
    fn test_minsnr_weighter_compute_weight_monotone() {
        // At t=0 SNR is very high → weight ≈ gamma/SNR ≪ 1.
        // At t=999 SNR < gamma → weight = 1.
        let w = MinSnrWeighter::default();
        let w_low_t = w.compute_weight(0).expect("weight at t=0");
        let w_high_t = w.compute_weight(999).expect("weight at t=999");
        assert!(
            w_high_t > w_low_t,
            "high-noise weight should exceed low-noise weight"
        );
    }

    #[test]
    fn test_minsnr_weighter_weight_out_of_range() {
        let w = MinSnrWeighter::default();
        let result = w.compute_weight(1000);
        assert!(matches!(
            result,
            Err(AdaptiveSamplingError::TimestepOutOfRange { .. })
        ));
    }

    #[test]
    fn test_minsnr_weighter_all_weights_correct_length() {
        let w = MinSnrWeighter::default();
        let weights = w.compute_all_weights().expect("should compute");
        assert_eq!(weights.len(), 1000);
    }

    #[test]
    fn test_minsnr_weighter_all_weights_positive() {
        let w = MinSnrWeighter::default();
        let weights = w.compute_all_weights().expect("should compute");
        for &wt in &weights {
            assert!(wt > 0.0, "all weights must be positive");
        }
    }

    #[test]
    fn test_minsnr_weighter_sample_valid_range() {
        let w = MinSnrWeighter::default();
        let mut state = 2025u64;
        for _ in 0..100 {
            let t = w.sample_timestep_minsnr(&mut state).expect("should sample");
            assert!(t < 1000, "sampled timestep {t} out of range");
        }
    }

    #[test]
    fn test_minsnr_weighter_linear_schedule() {
        let w = MinSnrWeighter {
            schedule: "linear".to_string(),
            ..Default::default()
        };
        let snr_t0 = w.compute_snr(0).expect("linear t=0");
        let snr_t999 = w.compute_snr(999).expect("linear t=999");
        assert!(
            snr_t0 > snr_t999,
            "SNR must decrease over time for linear schedule"
        );
    }

    // ── UniformSampler ───────────────────────────────────────────────────────

    #[test]
    fn test_uniform_sampler_defaults() {
        let s = UniformSampler::default();
        assert_eq!(s.max_timesteps, 1000);
        assert_eq!(s.low, 0);
        assert_eq!(s.high, 999);
    }

    #[test]
    fn test_uniform_sampler_range() {
        let s = UniformSampler::default();
        let mut state = 13u64;
        for _ in 0..1_000 {
            let t = s.sample_timestep_uniform(&mut state);
            assert!(t < 1000, "timestep {t} must be < 1000");
        }
    }

    #[test]
    fn test_uniform_sampler_respects_low_high() {
        let s = UniformSampler {
            max_timesteps: 1000,
            low: 200,
            high: 400,
        };
        let mut state = 55u64;
        for _ in 0..1_000 {
            let t = s.sample_timestep_uniform(&mut state);
            assert!(
                (200..=400).contains(&t),
                "timestep {t} must be in [200, 400]"
            );
        }
    }

    // ── ContinuousSampler ────────────────────────────────────────────────────

    #[test]
    fn test_continuous_sampler_defaults() {
        let s = ContinuousSampler::default();
        assert_eq!(s.t_min, 0.0);
        assert_eq!(s.t_max, 1.0);
        assert!(!s.logit_normal);
    }

    #[test]
    fn test_continuous_sampler_uniform_in_range() {
        let s = ContinuousSampler::default();
        let mut state = 888u64;
        for _ in 0..1_000 {
            let t = s.sample_t(&mut state);
            assert!(
                (s.t_min..=s.t_max).contains(&t),
                "t={t} must be in [{}, {}]",
                s.t_min,
                s.t_max
            );
        }
    }

    #[test]
    fn test_continuous_sampler_logit_normal_in_range() {
        let s = ContinuousSampler {
            logit_normal: true,
            lognormal_mean: 0.0,
            lognormal_std: 1.0,
            t_min: 0.0,
            t_max: 1.0,
        };
        let mut state = 4321u64;
        for _ in 0..1_000 {
            let t = s.sample_t(&mut state);
            assert!(
                (0.0..=1.0).contains(&t),
                "logit-normal t={t} must be in [0, 1]"
            );
        }
    }

    #[test]
    fn test_continuous_sampler_custom_range() {
        let s = ContinuousSampler {
            t_min: 0.1,
            t_max: 0.9,
            ..Default::default()
        };
        let mut state = 1111u64;
        for _ in 0..1_000 {
            let t = s.sample_t(&mut state);
            assert!((0.1..=0.9).contains(&t), "t={t} must be in [0.1, 0.9]");
        }
    }

    // ── ImportanceSampler ────────────────────────────────────────────────────

    #[test]
    fn test_importance_sampler_from_uniform() {
        let s = ImportanceSampler::from_uniform(10);
        assert_eq!(s.weights.len(), 10);
        assert!(s.weights.iter().all(|&w| (w - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_importance_sampler_from_loss_history() {
        let losses = vec![1.0, 2.0, 3.0, 4.0];
        let s = ImportanceSampler::from_loss_history(&losses).expect("should succeed");
        assert_eq!(s.weights.len(), 4);
        // Weights should be proportional to losses.
        assert!(s.weights[3] > s.weights[0]);
    }

    #[test]
    fn test_importance_sampler_from_zero_losses_error() {
        let losses = vec![0.0, 0.0, 0.0];
        let result = ImportanceSampler::from_loss_history(&losses);
        assert!(matches!(result, Err(AdaptiveSamplingError::ZeroWeights)));
    }

    #[test]
    fn test_importance_sampler_sample_valid_index() {
        let s = ImportanceSampler::from_uniform(50);
        let mut state = 64u64;
        for _ in 0..500 {
            let idx = s.sample(&mut state).expect("should succeed");
            assert!(idx < 50, "index {idx} must be < 50");
        }
    }

    #[test]
    fn test_importance_sampler_concentrated() {
        // Only index 3 has non-zero weight.
        let losses = [0.0_f32, 0.0, 0.0, 10.0, 0.0];
        let s = ImportanceSampler::from_loss_history(&losses).expect("ok");
        let mut state = 42u64;
        for _ in 0..100 {
            assert_eq!(s.sample(&mut state).expect("ok"), 3);
        }
    }

    // ── TimestepHistory ───────────────────────────────────────────────────────

    #[test]
    fn test_timestep_history_record_count() {
        let mut h = TimestepHistory::new(10, 0.9);
        h.record(3, 0.5);
        h.record(3, 0.3);
        assert_eq!(h.counts[3], 2);
    }

    #[test]
    fn test_timestep_history_ema_updates() {
        let mut h = TimestepHistory::new(10, 0.9);
        // First update: loss[5] = 0.0 * 0.9 + 1.0 * 0.1 = 0.1
        h.record(5, 1.0);
        assert!((h.losses[5] - 0.1).abs() < 1e-6);
        // Second: loss[5] = 0.1 * 0.9 + 0.5 * 0.1 = 0.09 + 0.05 = 0.14
        h.record(5, 0.5);
        assert!((h.losses[5] - 0.14).abs() < 1e-5);
    }

    #[test]
    fn test_timestep_history_out_of_range_no_panic() {
        let mut h = TimestepHistory::new(10, 0.9);
        // Should not panic.
        h.record(999, 1.0);
    }

    #[test]
    fn test_timestep_history_compute_stats_coverage() {
        let mut h = TimestepHistory::new(10, 0.9);
        h.record(0, 1.0);
        h.record(5, 1.0);
        let stats = h.compute_stats();
        // 2 out of 10 timesteps sampled → coverage = 0.2
        assert!((stats.coverage - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_timestep_history_compute_stats_most_sampled() {
        let mut h = TimestepHistory::new(5, 0.9);
        h.record(2, 1.0);
        h.record(2, 1.0);
        h.record(2, 1.0);
        h.record(4, 1.0);
        let stats = h.compute_stats();
        assert_eq!(stats.most_sampled, 2);
    }

    #[test]
    fn test_timestep_history_compute_stats_least_sampled() {
        let mut h = TimestepHistory::new(4, 0.9);
        h.record(0, 1.0);
        h.record(0, 1.0);
        h.record(1, 1.0);
        // Indices 2 and 3 have zero counts — min_by_key picks 2 (earlier).
        let stats = h.compute_stats();
        assert!(stats.least_sampled == 2 || stats.least_sampled == 3);
    }

    // ── compute_snr_sampling_weights ─────────────────────────────────────────

    #[test]
    fn test_compute_snr_sampling_weights_correct_length() {
        let weights = compute_snr_sampling_weights(1000, 5.0, "cosine").expect("ok");
        assert_eq!(weights.len(), 1000);
    }

    #[test]
    fn test_compute_snr_sampling_weights_all_positive() {
        let weights = compute_snr_sampling_weights(100, 5.0, "cosine").expect("ok");
        assert!(weights.iter().all(|&w| w > 0.0));
    }

    #[test]
    fn test_compute_snr_sampling_weights_invalid_gamma() {
        let result = compute_snr_sampling_weights(100, 0.0, "cosine");
        assert!(matches!(
            result,
            Err(AdaptiveSamplingError::InvalidGamma { .. })
        ));
    }

    // ── timestep_to_sigma ────────────────────────────────────────────────────

    #[test]
    fn test_timestep_to_sigma_min_endpoint() {
        let sigma = timestep_to_sigma(0, 1000, 0.002, 80.0);
        assert!((sigma - 0.002).abs() < 1e-5);
    }

    #[test]
    fn test_timestep_to_sigma_max_endpoint() {
        let sigma = timestep_to_sigma(999, 1000, 0.002, 80.0);
        assert!((sigma - 80.0).abs() < 0.01);
    }

    #[test]
    fn test_timestep_to_sigma_monotone() {
        for t in 0..999 {
            let s0 = timestep_to_sigma(t, 1000, 0.002, 80.0);
            let s1 = timestep_to_sigma(t + 1, 1000, 0.002, 80.0);
            assert!(s1 >= s0, "sigma must be monotonically non-decreasing");
        }
    }

    // ── sigma_to_alpha_bar ───────────────────────────────────────────────────

    #[test]
    fn test_sigma_to_alpha_bar_zero_sigma() {
        let ab = sigma_to_alpha_bar(0.0);
        assert!((ab - 1.0).abs() < 1e-6, "sigma=0 → alpha_bar=1, got {ab}");
    }

    #[test]
    fn test_sigma_to_alpha_bar_large_sigma() {
        let ab = sigma_to_alpha_bar(1000.0);
        assert!(ab < 0.001, "large sigma → alpha_bar near 0, got {ab}");
    }

    #[test]
    fn test_sigma_to_alpha_bar_unit_sigma() {
        let ab = sigma_to_alpha_bar(1.0);
        // 1 / (1 + 1) = 0.5
        assert!((ab - 0.5).abs() < 1e-6);
    }

    // ── compute_effective_snr_range ──────────────────────────────────────────

    #[test]
    fn test_compute_effective_snr_range_min_le_max() {
        let sampler = LogNormalSampler::default();
        let (min_snr, max_snr) = compute_effective_snr_range(&sampler, 1000, 42);
        assert!(
            min_snr <= max_snr,
            "min_snr {min_snr} must be <= max_snr {max_snr}"
        );
    }

    #[test]
    fn test_compute_effective_snr_range_zero_samples() {
        let sampler = LogNormalSampler::default();
        let (min_snr, max_snr) = compute_effective_snr_range(&sampler, 0, 42);
        assert_eq!(min_snr, 0.0);
        assert_eq!(max_snr, 0.0);
    }

    // ── format_sampling_stats ────────────────────────────────────────────────

    #[test]
    fn test_format_sampling_stats_non_empty() {
        let stats = SamplingStats {
            mean_timestep: 500.0,
            std_timestep: 100.0,
            most_sampled: 499,
            least_sampled: 0,
            coverage: 0.95,
        };
        let s = format_sampling_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("mean_t=500.0"));
        assert!(s.contains("95.0%"));
    }

    // ── compute_timestep_histogram ───────────────────────────────────────────

    #[test]
    fn test_compute_timestep_histogram_total_count() {
        let mut h = TimestepHistory::new(100, 0.9);
        for t in 0..100 {
            h.record(t, 1.0);
        }
        let hist = compute_timestep_histogram(&h, 10);
        let total: usize = hist.iter().sum();
        assert_eq!(total, 100, "histogram total should match sample count");
    }

    #[test]
    fn test_compute_timestep_histogram_correct_bins() {
        let h = TimestepHistory::new(100, 0.9);
        let hist = compute_timestep_histogram(&h, 10);
        assert_eq!(hist.len(), 10);
    }

    #[test]
    fn test_compute_timestep_histogram_empty() {
        let h = TimestepHistory::new(100, 0.9);
        let hist = compute_timestep_histogram(&h, 0);
        assert!(hist.is_empty());
    }
}
