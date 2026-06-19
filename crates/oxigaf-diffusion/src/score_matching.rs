//! # Score Matching Objectives for Diffusion Training
//!
//! Implements Denoising Score Matching (DSM), Implicit Score Matching (ISM),
//! Sliced Score Matching (SSM), and Karras et al. preconditioning (EDM style).
//!
//! ## Theory
//!
//! Score matching minimises the Fisher divergence between the model score and
//! the true data score:
//!
//! ```text
//! L_SM = E_x [ || s_θ(x) − ∇_x log p(x) ||² ]
//! ```
//!
//! Because `∇_x log p(x)` is intractable, Denoising Score Matching (Vincent 2011)
//! uses noisy samples:
//!
//! ```text
//! L_DSM(σ) = E_{x~p, ε~N(0,I)} [ w(σ) || s_θ(x + σ·ε) − (−ε/σ) ||² ]
//! ```
//!
//! where the target score of a Gaussian perturbation kernel is `−ε/σ²` in the
//! normalised form (with σ² absorbed into the expectation).
//!
//! ## References
//!
//! - Vincent (2011), "A Connection Between Score Matching and Denoising Autoencoders"
//! - Song & Ermon (2019), "Generative Modeling by Estimating Gradients of the Data Distribution"
//! - Karras et al. (2022), "Elucidating the Design Space of Diffusion-Based Generative Models"
//! - Song et al. (2020), "Sliced Score Matching: A Scalable Approach to Density and Score Estimation"

use std::f32::consts::PI;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Private PRNG (module-local copies, following flow_matching.rs pattern)
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

/// Box-Muller transform: returns a single standard normal sample N(0, 1).
///
/// Consumes two uniform draws from the xorshift state.
fn bm_normal(state: &mut u64) -> f32 {
    let u1 = (xorshift_f32(state) + 1e-10_f32).max(1e-10_f32);
    let u2 = xorshift_f32(state);
    (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * PI * u2).cos()
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise from score matching operations.
#[derive(Debug, Error, PartialEq)]
pub enum ScoreMatchingError {
    /// Configuration parameter is semantically invalid.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Array dimensions do not agree.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Operation on an empty batch or array.
    #[error("Empty input: {0}")]
    EmptyInput(String),

    /// Noise level sigma is non-positive.
    #[error("Invalid sigma {sigma}: must be > 0")]
    InvalidSigma { sigma: f32 },

    /// A floating-point result was not finite (NaN or Inf).
    #[error("Non-finite value in '{op}'")]
    NonFinite { op: String },
}

// ─────────────────────────────────────────────────────────────────────────────
// Weighting schemes
// ─────────────────────────────────────────────────────────────────────────────

/// Loss weighting strategy applied at each noise level σ.
#[derive(Debug, Clone, PartialEq)]
pub enum SmWeighting {
    /// Equal weight 1.0 for all σ.
    Uniform,

    /// EDM-style weight: w(σ) = σ².
    Sigma2,

    /// Standard DSM weight: w(σ) = 1 / σ².
    InvSigma2,

    /// Learnable (stored as log-weight, exponentiated before use).
    Learned(f32),
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for denoising score matching training.
#[derive(Debug, Clone)]
pub struct ScoreMatchingConfig {
    /// Minimum noise level σ_min (e.g. 0.002).
    pub sigma_min: f32,
    /// Maximum noise level σ_max (e.g. 80.0).
    pub sigma_max: f32,
    /// Number of geometric noise schedule steps.
    pub n_sigmas: usize,
    /// How to weight losses at different σ levels.
    pub loss_weighting: SmWeighting,
    /// Whether to apply Karras et al. preconditioning.
    pub use_preconditioning: bool,
    /// Small constant for numerical stability (e.g. 1e-8).
    pub eps: f32,
}

impl Default for ScoreMatchingConfig {
    fn default() -> Self {
        Self {
            sigma_min: 0.002,
            sigma_max: 80.0,
            n_sigmas: 50,
            loss_weighting: SmWeighting::Sigma2,
            use_preconditioning: true,
            eps: 1e-8,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Noise schedule
// ─────────────────────────────────────────────────────────────────────────────

/// Build a geometric noise schedule of `n` sigma values.
///
/// The i-th value is: `σ_i = σ_min · (σ_max / σ_min)^(i / (n−1))`
///
/// Returns `Err` when `n < 2`, `sigma_min <= 0`, or `sigma_max <= sigma_min`.
pub fn sm_geometric_sigmas(
    sigma_min: f32,
    sigma_max: f32,
    n: usize,
) -> Result<Vec<f32>, ScoreMatchingError> {
    if n < 2 {
        return Err(ScoreMatchingError::InvalidConfig(format!(
            "n_sigmas must be >= 2, got {n}"
        )));
    }
    if sigma_min <= 0.0 || !sigma_min.is_finite() {
        return Err(ScoreMatchingError::InvalidConfig(format!(
            "sigma_min must be > 0, got {sigma_min}"
        )));
    }
    if sigma_max <= sigma_min {
        return Err(ScoreMatchingError::InvalidConfig(format!(
            "sigma_max ({sigma_max}) must be > sigma_min ({sigma_min})"
        )));
    }

    let log_min = sigma_min.ln();
    let log_max = sigma_max.ln();
    let sigmas: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            (log_min + t * (log_max - log_min)).exp()
        })
        .collect();
    Ok(sigmas)
}

/// Sample a sigma level from `sigmas` uniformly using xorshift64.
///
/// Uses `step` mixed with `seed` to derive the PRNG state.
pub fn sm_sample_sigma(sigmas: &[f32], step: u64, seed: u64) -> Result<f32, ScoreMatchingError> {
    if sigmas.is_empty() {
        return Err(ScoreMatchingError::EmptyInput("sigmas".to_string()));
    }
    let mut state = seed.wrapping_add(step.wrapping_mul(6364136223846793005));
    if state == 0 {
        state = 1;
    }
    let _ = xorshift64(&mut state); // warm-up
    let u = xorshift_f32(&mut state);
    let idx = (u * sigmas.len() as f32) as usize;
    let idx = idx.min(sigmas.len() - 1);
    Ok(sigmas[idx])
}

/// Compute the loss weight for a given sigma under a weighting scheme.
pub fn sm_loss_weight(sigma: f32, weighting: &SmWeighting) -> Result<f32, ScoreMatchingError> {
    match weighting {
        SmWeighting::Uniform => Ok(1.0),
        SmWeighting::Sigma2 => {
            if !sigma.is_finite() {
                return Err(ScoreMatchingError::InvalidSigma { sigma });
            }
            Ok(sigma * sigma)
        }
        SmWeighting::InvSigma2 => {
            if sigma <= 0.0 || !sigma.is_finite() {
                return Err(ScoreMatchingError::InvalidSigma { sigma });
            }
            Ok(1.0 / (sigma * sigma))
        }
        SmWeighting::Learned(log_w) => {
            let w = log_w.exp();
            if !w.is_finite() {
                return Err(ScoreMatchingError::NonFinite {
                    op: "Learned weight".to_string(),
                });
            }
            Ok(w)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Noisy batch
// ─────────────────────────────────────────────────────────────────────────────

/// A batch of clean + noisy samples at a fixed sigma level.
#[derive(Debug, Clone)]
pub struct NoisyBatch {
    /// Clean (original) samples, shape [N × D].
    pub clean: Vec<f32>,
    /// Noisy samples x + σ·ε, shape [N × D].
    pub noisy: Vec<f32>,
    /// The noise ε ~ N(0, I) added, shape [N × D].
    pub noise: Vec<f32>,
    /// The noise level used.
    pub sigma: f32,
    /// Number of samples N.
    pub n: usize,
    /// Dimensionality D.
    pub d: usize,
}

/// Add Gaussian noise at level `sigma` to `clean`, returning a `NoisyBatch`.
///
/// Uses Box-Muller via xorshift64 — no external RNG crate.
pub fn sm_add_noise(
    clean: &[f32],
    sigma: f32,
    seed: u64,
    n: usize,
    d: usize,
) -> Result<NoisyBatch, ScoreMatchingError> {
    if clean.len() != n * d {
        return Err(ScoreMatchingError::DimensionMismatch {
            expected: n * d,
            got: clean.len(),
        });
    }
    if sigma < 0.0 || !sigma.is_finite() {
        return Err(ScoreMatchingError::InvalidSigma { sigma });
    }
    if n == 0 || d == 0 {
        return Err(ScoreMatchingError::EmptyInput(
            "n and d must both be > 0".to_string(),
        ));
    }

    let mut state: u64 = if seed == 0 { 1 } else { seed };
    // warm-up
    xorshift64(&mut state);

    let total = n * d;
    let mut noise = Vec::with_capacity(total);
    let mut noisy = Vec::with_capacity(total);

    for &x in clean.iter() {
        let eps = bm_normal(&mut state);
        noise.push(eps);
        noisy.push(x + sigma * eps);
    }

    Ok(NoisyBatch {
        clean: clean.to_vec(),
        noisy,
        noise,
        sigma,
        n,
        d,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// DSM loss
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the scalar Denoising Score Matching loss.
///
/// Target score = −ε / σ² (the analytical score of a Gaussian kernel).
/// Loss = w(σ) · (1/N) · Σ_i ||predicted_score_i − target_score_i||²
pub fn sm_dsm_loss(
    predicted_scores: &[f32],
    batch: &NoisyBatch,
    config: &ScoreMatchingConfig,
) -> Result<f32, ScoreMatchingError> {
    let total = batch.n * batch.d;
    if predicted_scores.len() != total {
        return Err(ScoreMatchingError::DimensionMismatch {
            expected: total,
            got: predicted_scores.len(),
        });
    }
    if batch.sigma <= 0.0 {
        return Err(ScoreMatchingError::InvalidSigma { sigma: batch.sigma });
    }
    if total == 0 {
        return Err(ScoreMatchingError::EmptyInput("batch".to_string()));
    }

    let sigma2 = batch.sigma * batch.sigma + config.eps;
    let weight = sm_loss_weight(batch.sigma, &config.loss_weighting)?;

    let mut sum_sq = 0.0_f32;
    for (&noise_val, &score_val) in batch.noise.iter().zip(predicted_scores.iter()) {
        let target = -noise_val / sigma2;
        let diff = score_val - target;
        sum_sq += diff * diff;
    }

    let loss = weight * sum_sq / total as f32;
    if !loss.is_finite() {
        return Err(ScoreMatchingError::NonFinite {
            op: "sm_dsm_loss".to_string(),
        });
    }
    Ok(loss)
}

/// Compute per-sample DSM losses (one scalar per sample).
///
/// Each sample's loss is: w(σ) · (1/D) · ||predicted_i − target_i||²
pub fn sm_dsm_loss_per_sample(
    predicted_scores: &[f32],
    batch: &NoisyBatch,
    config: &ScoreMatchingConfig,
) -> Result<Vec<f32>, ScoreMatchingError> {
    let total = batch.n * batch.d;
    if predicted_scores.len() != total {
        return Err(ScoreMatchingError::DimensionMismatch {
            expected: total,
            got: predicted_scores.len(),
        });
    }
    if batch.sigma <= 0.0 {
        return Err(ScoreMatchingError::InvalidSigma { sigma: batch.sigma });
    }
    if batch.n == 0 || batch.d == 0 {
        return Err(ScoreMatchingError::EmptyInput("batch".to_string()));
    }

    let sigma2 = batch.sigma * batch.sigma + config.eps;
    let weight = sm_loss_weight(batch.sigma, &config.loss_weighting)?;
    let mut per_sample = Vec::with_capacity(batch.n);

    for s in 0..batch.n {
        let offset = s * batch.d;
        let mut sq = 0.0_f32;
        for d in 0..batch.d {
            let target = -batch.noise[offset + d] / sigma2;
            let diff = predicted_scores[offset + d] - target;
            sq += diff * diff;
        }
        per_sample.push(weight * sq / batch.d as f32);
    }
    Ok(per_sample)
}

// ─────────────────────────────────────────────────────────────────────────────
// Karras et al. (EDM) preconditioning
// ─────────────────────────────────────────────────────────────────────────────

/// EDM c_skip: `σ_data² / (σ² + σ_data²)`.
///
/// Use `sm_c_skip` at the re-export level; `c_skip` is the local name.
pub fn sm_c_skip(sigma: f32, sigma_data: f32) -> f32 {
    let sd2 = sigma_data * sigma_data;
    sd2 / (sigma * sigma + sd2)
}

/// EDM c_out: `σ · σ_data / sqrt(σ² + σ_data²)`.
pub fn sm_c_out(sigma: f32, sigma_data: f32) -> f32 {
    let denom = (sigma * sigma + sigma_data * sigma_data).sqrt();
    sigma * sigma_data / denom.max(1e-10)
}

/// EDM c_in: `1 / sqrt(σ² + σ_data²)`.
pub fn sm_c_in(sigma: f32, sigma_data: f32) -> f32 {
    let denom = (sigma * sigma + sigma_data * sigma_data).sqrt();
    1.0 / denom.max(1e-10)
}

/// EDM c_noise: `(1/4) · ln(σ)`.
///
/// Used as log-noise conditioning signal fed to the network.
pub fn sm_c_noise(sigma: f32) -> f32 {
    0.25_f32 * sigma.max(1e-10).ln()
}

/// Scale a noisy input by `c_in` for unit-variance network input.
pub fn sm_precondition_input(noisy: &[f32], sigma: f32, sigma_data: f32) -> Vec<f32> {
    let scale = sm_c_in(sigma, sigma_data);
    noisy.iter().map(|&x| x * scale).collect()
}

/// Compute the preconditioned output:
/// `D(x) = c_skip · x + c_out · F(c_in · x, c_noise(σ))`
///
/// Here `raw_output` is the network output `F(·)` evaluated at the scaled input.
pub fn sm_precondition_output(
    raw_output: &[f32],
    noisy: &[f32],
    sigma: f32,
    sigma_data: f32,
) -> Result<Vec<f32>, ScoreMatchingError> {
    if raw_output.len() != noisy.len() {
        return Err(ScoreMatchingError::DimensionMismatch {
            expected: noisy.len(),
            got: raw_output.len(),
        });
    }
    let cs = sm_c_skip(sigma, sigma_data);
    let co = sm_c_out(sigma, sigma_data);
    let out: Vec<f32> = raw_output
        .iter()
        .zip(noisy.iter())
        .map(|(&f, &x)| cs * x + co * f)
        .collect();
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Implicit Score Matching (Hutchinson trace estimator)
// ─────────────────────────────────────────────────────────────────────────────

/// Finite-difference approximation of the Hutchinson trace estimator.
///
/// Approximates `tr(∇_x s_θ(x)) ≈ v^T · (s(x + ε·v) − s(x)) / ε`
/// for a probe vector `v ~ N(0, I)`.
///
/// # Arguments
/// * `scores` — s_θ(x), shape [N × D]
/// * `perturbed_scores` — s_θ(x + eps·probe), shape [N × D]
/// * `probe` — random probe vector v, shape [N × D]
/// * `eps` — finite-difference step size
/// * `n`, `d` — batch size and dimensionality
pub fn sm_hutchinson_trace_estimate(
    scores: &[f32],
    perturbed_scores: &[f32],
    probe: &[f32],
    eps: f32,
    n: usize,
    d: usize,
) -> Result<f32, ScoreMatchingError> {
    let total = n * d;
    if scores.len() != total {
        return Err(ScoreMatchingError::DimensionMismatch {
            expected: total,
            got: scores.len(),
        });
    }
    if perturbed_scores.len() != total {
        return Err(ScoreMatchingError::DimensionMismatch {
            expected: total,
            got: perturbed_scores.len(),
        });
    }
    if probe.len() != total {
        return Err(ScoreMatchingError::DimensionMismatch {
            expected: total,
            got: probe.len(),
        });
    }
    if eps <= 0.0 {
        return Err(ScoreMatchingError::InvalidConfig(format!(
            "eps must be > 0, got {eps}"
        )));
    }

    let inv_eps = 1.0 / eps;
    let mut trace_sum = 0.0_f32;
    for i in 0..total {
        let jv = (perturbed_scores[i] - scores[i]) * inv_eps;
        trace_sum += probe[i] * jv;
    }
    // Average over batch samples
    Ok(trace_sum / n as f32)
}

/// Implicit Score Matching loss (per-batch average).
///
/// ISM loss = (1/N) · Σ_i [ (1/2) · ||s_θ(x_i)||² + tr(∇ s_θ(x_i)) ]
///
/// `trace_estimate` should come from `sm_hutchinson_trace_estimate`.
pub fn sm_ism_loss(
    scores: &[f32],
    trace_estimate: f32,
    n: usize,
) -> Result<f32, ScoreMatchingError> {
    if scores.is_empty() {
        return Err(ScoreMatchingError::EmptyInput("scores".to_string()));
    }
    if n == 0 {
        return Err(ScoreMatchingError::EmptyInput("n must be > 0".to_string()));
    }

    let norm_sq: f32 = scores.iter().map(|&s| s * s).sum();
    let half_norm_sq = 0.5 * norm_sq / n as f32;
    let loss = half_norm_sq + trace_estimate;
    if !loss.is_finite() {
        return Err(ScoreMatchingError::NonFinite {
            op: "sm_ism_loss".to_string(),
        });
    }
    Ok(loss)
}

// ─────────────────────────────────────────────────────────────────────────────
// Sliced Score Matching
// ─────────────────────────────────────────────────────────────────────────────

/// Sliced Score Matching loss using random projections.
///
/// For each projection v_k:
/// `L_k = (v_k^T s)² + 2 · v_k^T · (s_perturbed − s) / eps`
///
/// The total loss is the mean over all N × n_proj terms.
///
/// # Arguments
/// * `scores` — s_θ(x), shape [N × D]
/// * `scores_perturbed` — s_θ(x + eps·v), shape [N × D] (one projection direction)
/// * `projections` — random projection vectors, shape [n_proj × D]
/// * `eps` — finite-difference step
/// * `n`, `d` — batch size and dimensionality
/// * `n_proj` — number of projection directions
pub fn sm_sliced_score_matching_loss(
    scores: &[f32],
    scores_perturbed: &[f32],
    projections: &[f32],
    eps: f32,
    n: usize,
    d: usize,
    n_proj: usize,
) -> Result<f32, ScoreMatchingError> {
    let total = n * d;
    if scores.len() != total {
        return Err(ScoreMatchingError::DimensionMismatch {
            expected: total,
            got: scores.len(),
        });
    }
    if scores_perturbed.len() != total {
        return Err(ScoreMatchingError::DimensionMismatch {
            expected: total,
            got: scores_perturbed.len(),
        });
    }
    if projections.len() != n_proj * d {
        return Err(ScoreMatchingError::DimensionMismatch {
            expected: n_proj * d,
            got: projections.len(),
        });
    }
    if eps <= 0.0 {
        return Err(ScoreMatchingError::InvalidConfig(format!(
            "eps must be > 0, got {eps}"
        )));
    }
    if n == 0 || d == 0 || n_proj == 0 {
        return Err(ScoreMatchingError::EmptyInput(
            "n, d, n_proj must all be > 0".to_string(),
        ));
    }

    let inv_eps = 1.0 / eps;
    let mut total_loss = 0.0_f32;

    for s in 0..n {
        let s_offset = s * d;
        for p in 0..n_proj {
            let p_offset = p * d;
            // vT s
            let mut vts = 0.0_f32;
            // vT (s_perturbed - s) / eps
            let mut vt_jv = 0.0_f32;
            for dd in 0..d {
                let v = projections[p_offset + dd];
                let sc = scores[s_offset + dd];
                let sc_p = scores_perturbed[s_offset + dd];
                vts += v * sc;
                vt_jv += v * (sc_p - sc) * inv_eps;
            }
            total_loss += vts * vts + 2.0 * vt_jv;
        }
    }

    let loss = total_loss / (n * n_proj) as f32;
    if !loss.is_finite() {
        return Err(ScoreMatchingError::NonFinite {
            op: "sm_sliced_score_matching_loss".to_string(),
        });
    }
    Ok(loss)
}

// ─────────────────────────────────────────────────────────────────────────────
// ScoreFunction wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Score function wrapper combining config, preconditioning, and loss utilities.
pub struct ScoreFunction {
    /// Score matching configuration.
    pub config: ScoreMatchingConfig,
    /// Expected data standard deviation (for Karras preconditioning).
    pub sigma_data: f32,
}

impl ScoreFunction {
    /// Create a new `ScoreFunction`.
    pub fn new(config: ScoreMatchingConfig, sigma_data: f32) -> Self {
        Self { config, sigma_data }
    }

    /// Compute the DSM loss (not weighted by σ schedule).
    pub fn loss(&self, predicted: &[f32], batch: &NoisyBatch) -> Result<f32, ScoreMatchingError> {
        sm_dsm_loss(predicted, batch, &self.config)
    }

    /// Compute the DSM loss weighted by the configured weighting scheme.
    ///
    /// Equivalent to `sm_dsm_loss` (weight is already baked in), kept as an
    /// explicit entry point to make the API intent clear.
    pub fn weighted_loss(
        &self,
        predicted: &[f32],
        batch: &NoisyBatch,
    ) -> Result<f32, ScoreMatchingError> {
        sm_dsm_loss(predicted, batch, &self.config)
    }

    /// Compute the analytical target score: `−noise / σ²` element-wise.
    pub fn target_score(&self, batch: &NoisyBatch) -> Vec<f32> {
        let sigma2 = batch.sigma * batch.sigma + self.config.eps;
        batch.noise.iter().map(|&e| -e / sigma2).collect()
    }

    /// Compute the Karras-preconditioned target.
    ///
    /// Under EDM preconditioning the network learns to denoise, and the
    /// effective target is: `(noisy − c_skip · noisy) / c_out = noisy · (1 − c_skip) / c_out`
    /// which simplifies to `noise / sigma_data` at leading order.
    /// Here we return `clean − c_skip · noisy` (the D-target) element-wise,
    /// scaled by 1/c_out.
    pub fn preconditioned_target(&self, batch: &NoisyBatch) -> Vec<f32> {
        let cs = sm_c_skip(batch.sigma, self.sigma_data);
        let co = sm_c_out(batch.sigma, self.sigma_data).max(1e-10);
        batch
            .clean
            .iter()
            .zip(batch.noisy.iter())
            .map(|(&x0, &xt)| (x0 - cs * xt) / co)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics for a batch of per-sample score matching losses.
#[derive(Debug, Clone)]
pub struct ScoreMatchingStats {
    /// Mean loss across samples.
    pub mean_loss: f32,
    /// Minimum loss across samples.
    pub min_loss: f32,
    /// Maximum loss across samples.
    pub max_loss: f32,
    /// Mean sigma across samples.
    pub mean_sigma: f32,
    /// Total number of samples.
    pub n_samples: usize,
    /// Effective weight (geometric mean of per-sample weights or 1.0 when unavailable).
    pub effective_weight: f32,
}

/// Compute statistics from per-sample losses and corresponding sigma values.
///
/// Returns sensible zeros/defaults when the input slices are empty.
pub fn sm_compute_stats(losses: &[f32], sigmas: &[f32]) -> ScoreMatchingStats {
    let n = losses.len();
    if n == 0 {
        return ScoreMatchingStats {
            mean_loss: 0.0,
            min_loss: 0.0,
            max_loss: 0.0,
            mean_sigma: 0.0,
            n_samples: 0,
            effective_weight: 1.0,
        };
    }

    let mean_loss = losses.iter().sum::<f32>() / n as f32;
    let min_loss = losses.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_loss = losses.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mean_sigma = if sigmas.is_empty() {
        0.0
    } else {
        sigmas.iter().sum::<f32>() / sigmas.len() as f32
    };

    // Effective weight: geometric mean of sigma-based weights (Sigma2 style)
    let effective_weight = if sigmas.is_empty() {
        1.0
    } else {
        let log_sum: f32 = sigmas.iter().map(|&s| 2.0 * s.max(1e-10).ln()).sum::<f32>();
        (log_sum / sigmas.len() as f32).exp()
    };

    ScoreMatchingStats {
        mean_loss,
        min_loss,
        max_loss,
        mean_sigma,
        n_samples: n,
        effective_weight,
    }
}

/// Format `ScoreMatchingStats` as a human-readable string.
pub fn sm_format_stats(stats: &ScoreMatchingStats) -> String {
    format!(
        "ScoreMatchingStats {{ n={}, loss=[{:.4e}, {:.4e}] mean={:.4e}, mean_sigma={:.4e}, eff_w={:.4e} }}",
        stats.n_samples,
        stats.min_loss,
        stats.max_loss,
        stats.mean_loss,
        stats.mean_sigma,
        stats.effective_weight,
    )
}

/// Format `ScoreMatchingConfig` as a human-readable string.
pub fn sm_format_config(config: &ScoreMatchingConfig) -> String {
    format!(
        "ScoreMatchingConfig {{ sigma=[{:.4e}, {:.4e}], n_sigmas={}, weighting={:?}, precond={}, eps={:.2e} }}",
        config.sigma_min,
        config.sigma_max,
        config.n_sigmas,
        config.loss_weighting,
        config.use_preconditioning,
        config.eps,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn default_config() -> ScoreMatchingConfig {
        ScoreMatchingConfig::default()
    }

    /// Make a simple clean batch of all-zeros for testing shape logic.
    fn zero_clean(n: usize, d: usize) -> Vec<f32> {
        vec![0.0_f32; n * d]
    }

    /// Make a NoisyBatch with explicit noise values (for exact loss checks).
    fn manual_batch(clean: Vec<f32>, noise: Vec<f32>, sigma: f32) -> NoisyBatch {
        let total = clean.len();
        let noisy: Vec<f32> = clean
            .iter()
            .zip(noise.iter())
            .map(|(&x, &e)| x + sigma * e)
            .collect();
        NoisyBatch {
            clean: clean.clone(),
            noisy,
            noise,
            sigma,
            n: total,
            d: 1,
        }
    }

    // ── sm_geometric_sigmas ───────────────────────────────────────────────

    #[test]
    fn test_geometric_sigmas_first_last() {
        let sigmas = sm_geometric_sigmas(0.002, 80.0, 50).unwrap();
        assert_eq!(sigmas.len(), 50);
        assert!(
            (sigmas[0] - 0.002).abs() < 1e-6,
            "first sigma should be sigma_min"
        );
        assert!(
            (sigmas[49] - 80.0).abs() < 1e-3,
            "last sigma should be sigma_max"
        );
    }

    #[test]
    fn test_geometric_sigmas_geometric_ratio() {
        let sigmas = sm_geometric_sigmas(1.0, 100.0, 10).unwrap();
        // Each consecutive ratio should be approximately equal
        let ratio0 = sigmas[1] / sigmas[0];
        for i in 1..9 {
            let ratio = sigmas[i + 1] / sigmas[i];
            assert!(
                (ratio - ratio0).abs() < 1e-5,
                "ratio at {i} ({ratio}) differs from ratio0 ({ratio0})"
            );
        }
    }

    #[test]
    fn test_geometric_sigmas_monotone_increasing() {
        let sigmas = sm_geometric_sigmas(0.01, 10.0, 20).unwrap();
        for i in 1..sigmas.len() {
            assert!(sigmas[i] > sigmas[i - 1], "not monotone at {i}");
        }
    }

    #[test]
    fn test_geometric_sigmas_error_n_less_2() {
        assert!(sm_geometric_sigmas(0.002, 80.0, 1).is_err());
    }

    #[test]
    fn test_geometric_sigmas_error_sigma_min_zero() {
        assert!(sm_geometric_sigmas(0.0, 80.0, 50).is_err());
    }

    #[test]
    fn test_geometric_sigmas_error_sigma_min_neg() {
        assert!(sm_geometric_sigmas(-1.0, 80.0, 50).is_err());
    }

    #[test]
    fn test_geometric_sigmas_error_max_le_min() {
        assert!(sm_geometric_sigmas(80.0, 0.002, 50).is_err());
        assert!(sm_geometric_sigmas(1.0, 1.0, 50).is_err());
    }

    #[test]
    fn test_geometric_sigmas_n2() {
        let sigmas = sm_geometric_sigmas(1.0, 10.0, 2).unwrap();
        assert_eq!(sigmas.len(), 2);
        assert!((sigmas[0] - 1.0).abs() < 1e-6);
        assert!((sigmas[1] - 10.0).abs() < 1e-4);
    }

    // ── sm_sample_sigma ───────────────────────────────────────────────────

    #[test]
    fn test_sample_sigma_in_range() {
        let sigmas = sm_geometric_sigmas(0.002, 80.0, 50).unwrap();
        // Use the actual min/max from the computed schedule (exp(ln(x)) has tiny f32 error)
        let smin = sigmas.iter().cloned().fold(f32::INFINITY, f32::min);
        let smax = sigmas.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for step in 0..100_u64 {
            let s = sm_sample_sigma(&sigmas, step, 42).unwrap();
            assert!(
                s >= smin && s <= smax,
                "sigma {s} out of range [{smin}, {smax}] at step {step}"
            );
        }
    }

    #[test]
    fn test_sample_sigma_deterministic() {
        let sigmas = sm_geometric_sigmas(0.002, 80.0, 50).unwrap();
        let s1 = sm_sample_sigma(&sigmas, 7, 99).unwrap();
        let s2 = sm_sample_sigma(&sigmas, 7, 99).unwrap();
        assert_eq!(s1, s2, "same (step, seed) must give same sigma");
    }

    #[test]
    fn test_sample_sigma_different_steps() {
        let sigmas = sm_geometric_sigmas(0.002, 80.0, 50).unwrap();
        let results: Vec<f32> = (0..20)
            .map(|s| sm_sample_sigma(&sigmas, s, 1).unwrap())
            .collect();
        // Not all the same (may collide occasionally — check variance > 0)
        let mean = results.iter().sum::<f32>() / results.len() as f32;
        let var = results
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>()
            / results.len() as f32;
        assert!(var > 0.0, "all sampled sigmas identical — PRNG not varying");
    }

    #[test]
    fn test_sample_sigma_empty_error() {
        assert!(sm_sample_sigma(&[], 0, 42).is_err());
    }

    // ── sm_loss_weight ────────────────────────────────────────────────────

    #[test]
    fn test_loss_weight_uniform() {
        let w = sm_loss_weight(5.0, &SmWeighting::Uniform).unwrap();
        assert_eq!(w, 1.0);
    }

    #[test]
    fn test_loss_weight_sigma2() {
        let w = sm_loss_weight(3.0, &SmWeighting::Sigma2).unwrap();
        assert!((w - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_loss_weight_inv_sigma2() {
        let w = sm_loss_weight(2.0, &SmWeighting::InvSigma2).unwrap();
        assert!((w - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_loss_weight_learned() {
        let log_w = 0.0_f32; // exp(0) = 1.0
        let w = sm_loss_weight(1.0, &SmWeighting::Learned(log_w)).unwrap();
        assert!((w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_loss_weight_learned_positive() {
        let log_w = 2.0_f32; // exp(2) ≈ 7.389
        let w = sm_loss_weight(1.0, &SmWeighting::Learned(log_w)).unwrap();
        assert!((w - 2.0_f32.exp()).abs() < 1e-4);
    }

    #[test]
    fn test_loss_weight_inv_sigma2_zero_error() {
        assert!(sm_loss_weight(0.0, &SmWeighting::InvSigma2).is_err());
    }

    // ── sm_add_noise ──────────────────────────────────────────────────────

    #[test]
    fn test_add_noise_shape_preservation() {
        let n = 10;
        let d = 8;
        let clean = zero_clean(n, d);
        let batch = sm_add_noise(&clean, 1.0, 42, n, d).unwrap();
        assert_eq!(batch.clean.len(), n * d);
        assert_eq!(batch.noisy.len(), n * d);
        assert_eq!(batch.noise.len(), n * d);
        assert_eq!(batch.n, n);
        assert_eq!(batch.d, d);
        assert_eq!(batch.sigma, 1.0);
    }

    #[test]
    fn test_add_noise_zero_sigma() {
        let clean = zero_clean(5, 4);
        let batch = sm_add_noise(&clean, 0.0, 42, 5, 4).unwrap();
        // With sigma=0, noisy = clean element-wise
        for (&x, &xn) in batch.clean.iter().zip(batch.noisy.iter()) {
            assert!((x - xn).abs() < 1e-10);
        }
    }

    #[test]
    fn test_add_noise_mean_approx_zero() {
        // N(0,1) noise: mean should be near 0 for large samples
        let n = 1000;
        let d = 1;
        let clean = zero_clean(n, d);
        let batch = sm_add_noise(&clean, 1.0, 7, n, d).unwrap();
        let mean: f32 = batch.noise.iter().sum::<f32>() / batch.noise.len() as f32;
        assert!(mean.abs() < 0.15, "noise mean {mean} too far from 0");
    }

    #[test]
    fn test_add_noise_std_approx_sigma() {
        let n = 1000;
        let d = 1;
        let sigma = 3.0;
        let clean = zero_clean(n, d);
        let batch = sm_add_noise(&clean, sigma, 13, n, d).unwrap();
        // noisy = 0 + sigma*eps  =>  std(noisy) ≈ sigma
        let mean: f32 = batch.noisy.iter().sum::<f32>() / batch.noisy.len() as f32;
        let var: f32 = batch
            .noisy
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>()
            / batch.noisy.len() as f32;
        let std = var.sqrt();
        assert!(
            (std - sigma).abs() / sigma < 0.10,
            "noise std {std} not within 10% of sigma {sigma}"
        );
    }

    #[test]
    fn test_add_noise_negative_sigma_error() {
        let clean = zero_clean(5, 4);
        assert!(sm_add_noise(&clean, -1.0, 42, 5, 4).is_err());
    }

    #[test]
    fn test_add_noise_dimension_mismatch_error() {
        let clean = vec![0.0_f32; 10];
        // claim n=3, d=4 but clean.len()=10
        assert!(sm_add_noise(&clean, 1.0, 42, 3, 4).is_err());
    }

    // ── Box-Muller distribution quality ──────────────────────────────────

    #[test]
    fn test_box_muller_mean_and_std() {
        // Generate 2000 standard-normal samples and check moments
        let mut state = 0xdeadbeef_u64;
        let samples: Vec<f32> = (0..2000).map(|_| bm_normal(&mut state)).collect();
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let var = samples
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>()
            / samples.len() as f32;
        let std = var.sqrt();
        assert!(mean.abs() < 0.1, "mean {mean} too far from 0");
        assert!((std - 1.0).abs() < 0.1, "std {std} too far from 1");
    }

    // ── sm_dsm_loss ───────────────────────────────────────────────────────

    #[test]
    fn test_dsm_loss_zero_when_perfect() {
        // predicted = target => loss = 0
        let sigma = 1.0_f32;
        let noise = vec![0.5_f32, -0.3, 0.1, 0.8];
        let config = ScoreMatchingConfig {
            loss_weighting: SmWeighting::Uniform,
            ..default_config()
        };
        let sigma2 = sigma * sigma + config.eps;
        let target: Vec<f32> = noise.iter().map(|&e| -e / sigma2).collect();

        let batch = NoisyBatch {
            clean: vec![0.0; 4],
            noisy: vec![0.5, -0.3, 0.1, 0.8],
            noise: noise.clone(),
            sigma,
            n: 4,
            d: 1,
        };
        let loss = sm_dsm_loss(&target, &batch, &config).unwrap();
        assert!(loss.abs() < 1e-6, "loss should be ~0, got {loss}");
    }

    #[test]
    fn test_dsm_loss_positive_for_error() {
        let sigma = 1.0_f32;
        let noise = vec![0.0_f32; 4];
        let predicted = vec![1.0_f32; 4]; // non-zero error
        let config = ScoreMatchingConfig {
            loss_weighting: SmWeighting::Uniform,
            ..default_config()
        };
        let batch = NoisyBatch {
            clean: vec![0.0; 4],
            noisy: vec![0.0; 4],
            noise,
            sigma,
            n: 4,
            d: 1,
        };
        let loss = sm_dsm_loss(&predicted, &batch, &config).unwrap();
        assert!(loss > 0.0, "loss should be positive");
    }

    #[test]
    fn test_dsm_loss_sigma2_weighting() {
        let sigma = 2.0_f32;
        let noise = vec![1.0_f32];
        let config = ScoreMatchingConfig {
            loss_weighting: SmWeighting::Sigma2,
            eps: 0.0,
            ..default_config()
        };
        // target = -1/4, predicted = 0 => diff = 1/4 => sq = 1/16
        // weight = 4, loss = 4 * 1/16 = 0.25
        let predicted = vec![0.0_f32];
        let batch = NoisyBatch {
            clean: vec![0.0],
            noisy: vec![0.0],
            noise,
            sigma,
            n: 1,
            d: 1,
        };
        let loss = sm_dsm_loss(&predicted, &batch, &config).unwrap();
        let expected = 4.0 * (1.0_f32 / 16.0);
        assert!(
            (loss - expected).abs() < 1e-5,
            "loss={loss} expected={expected}"
        );
    }

    #[test]
    fn test_dsm_loss_dimension_mismatch_error() {
        let batch = manual_batch(vec![0.0; 4], vec![0.0; 4], 1.0);
        let wrong_pred = vec![0.0_f32; 3];
        assert!(sm_dsm_loss(&wrong_pred, &batch, &default_config()).is_err());
    }

    #[test]
    fn test_dsm_loss_zero_sigma_error() {
        let batch = manual_batch(vec![0.0; 4], vec![0.0; 4], 0.0);
        let pred = vec![0.0_f32; 4];
        assert!(sm_dsm_loss(&pred, &batch, &default_config()).is_err());
    }

    // ── sm_dsm_loss_per_sample ────────────────────────────────────────────

    #[test]
    fn test_dsm_loss_per_sample_length() {
        let n = 7;
        let d = 5;
        let total = n * d;
        let clean = zero_clean(n, d);
        let batch = sm_add_noise(&clean, 1.0, 99, n, d).unwrap();
        let target = batch
            .noise
            .iter()
            .map(|&e| -e / (1.0 + 1e-8))
            .collect::<Vec<_>>();
        let per = sm_dsm_loss_per_sample(&target, &batch, &default_config()).unwrap();
        assert_eq!(per.len(), n, "per-sample losses should have length N");
        let _ = total; // silence unused warning
    }

    #[test]
    fn test_dsm_loss_per_sample_all_nonneg() {
        let n = 10;
        let d = 4;
        let clean = zero_clean(n, d);
        let batch = sm_add_noise(&clean, 1.5, 77, n, d).unwrap();
        let pred = vec![0.0_f32; n * d];
        let per = sm_dsm_loss_per_sample(&pred, &batch, &default_config()).unwrap();
        for (i, &l) in per.iter().enumerate() {
            assert!(l >= 0.0, "negative per-sample loss at index {i}: {l}");
        }
    }

    #[test]
    fn test_dsm_loss_per_sample_sum_close_to_scalar() {
        // mean of per-sample losses ~ scalar loss
        let n = 5;
        let d = 4;
        let clean = zero_clean(n, d);
        let batch = sm_add_noise(&clean, 1.0, 55, n, d).unwrap();
        let config = ScoreMatchingConfig {
            loss_weighting: SmWeighting::Uniform,
            eps: 1e-8,
            ..default_config()
        };
        let pred = vec![0.0_f32; n * d];
        let per = sm_dsm_loss_per_sample(&pred, &batch, &config).unwrap();
        let scalar = sm_dsm_loss(&pred, &batch, &config).unwrap();
        let per_mean = per.iter().sum::<f32>() / per.len() as f32;
        assert!(
            (per_mean - scalar).abs() < 1e-5,
            "per-sample mean {per_mean} != scalar {scalar}"
        );
    }

    // ── Karras preconditioning ────────────────────────────────────────────

    #[test]
    fn test_sm_c_skip_limit_sigma_zero() {
        // As sigma→0, c_skip → 1
        let c = sm_c_skip(1e-6, 1.0);
        assert!(
            (c - 1.0).abs() < 1e-4,
            "c_skip at sigma~0 should be ~1, got {c}"
        );
    }

    #[test]
    fn test_sm_c_skip_limit_sigma_large() {
        // As sigma→∞, c_skip → 0
        let c = sm_c_skip(1e6, 1.0);
        assert!(c < 1e-5, "c_skip at sigma large should be ~0, got {c}");
    }

    #[test]
    fn test_sm_c_out_limit_sigma_zero() {
        // As sigma→0, c_out → 0
        let c = sm_c_out(1e-6, 1.0);
        assert!(c < 1e-4, "c_out at sigma~0 should be ~0, got {c}");
    }

    #[test]
    fn test_sm_c_out_limit_sigma_large() {
        // As sigma→∞, c_out → sigma_data
        let sd = 1.0_f32;
        let c = sm_c_out(1e6, sd);
        assert!(
            (c - sd).abs() < 0.01,
            "c_out at sigma large should be ~sigma_data={sd}, got {c}"
        );
    }

    #[test]
    fn test_sm_c_in_decreases_with_sigma() {
        let c1 = sm_c_in(1.0, 1.0);
        let c2 = sm_c_in(10.0, 1.0);
        assert!(c2 < c1, "c_in should decrease as sigma increases");
    }

    #[test]
    fn test_sm_c_noise_monotone() {
        // ln is monotone => c_noise is monotone in sigma
        let c1 = sm_c_noise(1.0);
        let c2 = sm_c_noise(10.0);
        assert!(c2 > c1, "c_noise should increase with sigma");
    }

    #[test]
    fn test_sm_c_noise_value() {
        // c_noise(e) = 0.25 * 1 = 0.25
        let c = sm_c_noise(std::f32::consts::E);
        assert!((c - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_precondition_input_smaller_magnitude() {
        let noisy = vec![10.0_f32; 8];
        let precond = sm_precondition_input(&noisy, 1.0, 1.0);
        // c_in = 1/sqrt(2) ≈ 0.707
        for &v in &precond {
            assert!(
                v.abs() < 10.0,
                "preconditioned value {v} not smaller than original"
            );
        }
    }

    #[test]
    fn test_precondition_input_scale() {
        let noisy = vec![1.0_f32];
        let sigma = 3.0;
        let sd = 4.0;
        let out = sm_precondition_input(&noisy, sigma, sd);
        let expected = sm_c_in(sigma, sd);
        assert!((out[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_precondition_output_shape() {
        let raw = vec![0.5_f32; 16];
        let noisy = vec![1.0_f32; 16];
        let out = sm_precondition_output(&raw, &noisy, 1.0, 1.0).unwrap();
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_precondition_output_dimension_error() {
        let raw = vec![0.0_f32; 5];
        let noisy = vec![0.0_f32; 4];
        assert!(sm_precondition_output(&raw, &noisy, 1.0, 1.0).is_err());
    }

    // ── sm_hutchinson_trace_estimate ──────────────────────────────────────

    #[test]
    fn test_hutchinson_zero_scores() {
        let n = 4;
        let d = 3;
        let total = n * d;
        let zeros = vec![0.0_f32; total];
        let probe = vec![1.0_f32; total];
        // scores = perturbed_scores = 0 => trace = 0
        let trace = sm_hutchinson_trace_estimate(&zeros, &zeros, &probe, 0.01, n, d).unwrap();
        assert!(
            trace.abs() < 1e-10,
            "trace should be 0 for zero scores, got {trace}"
        );
    }

    #[test]
    fn test_hutchinson_linear_case() {
        // For s(x) = A·x with A = c·I, tr(J) = c·D per element, per sample
        // s(x) = c·x, s(x + eps·v) = c·(x + eps·v)
        // diff = c·eps·v, jv = c·v, v^T·jv = c·||v||²
        // For v = e_1 (unit), trace ≈ c (first component)
        // Use c=2, v=all-ones, then trace = c * d per sample = 2*d/n*n = 2*d
        let c = 2.0_f32;
        let n = 1;
        let d = 4;
        let total = n * d;
        let x = vec![1.0_f32; total];
        let v = vec![1.0_f32; total];
        let eps = 0.001_f32;
        let scores: Vec<f32> = x.iter().map(|&xi| c * xi).collect();
        let perturbed_x: Vec<f32> = x
            .iter()
            .zip(v.iter())
            .map(|(&xi, &vi)| xi + eps * vi)
            .collect();
        let perturbed_scores: Vec<f32> = perturbed_x.iter().map(|&xi| c * xi).collect();
        let trace =
            sm_hutchinson_trace_estimate(&scores, &perturbed_scores, &v, eps, n, d).unwrap();
        // trace ≈ c * d (each component contributes c, summed over d=4)
        let expected = c * d as f32;
        assert!(
            (trace - expected).abs() < 0.01,
            "trace={trace} expected={expected}"
        );
    }

    #[test]
    fn test_hutchinson_dimension_mismatch() {
        let n = 2;
        let d = 3;
        let total = n * d;
        let s = vec![0.0_f32; total];
        let sp = vec![0.0_f32; total];
        let bad_probe = vec![0.0_f32; total - 1];
        assert!(sm_hutchinson_trace_estimate(&s, &sp, &bad_probe, 0.01, n, d).is_err());
    }

    #[test]
    fn test_hutchinson_invalid_eps() {
        let n = 2;
        let d = 3;
        let total = n * d;
        let s = vec![0.0_f32; total];
        assert!(sm_hutchinson_trace_estimate(&s, &s, &s, 0.0, n, d).is_err());
        assert!(sm_hutchinson_trace_estimate(&s, &s, &s, -0.01, n, d).is_err());
    }

    // ── sm_ism_loss ───────────────────────────────────────────────────────

    #[test]
    fn test_ism_loss_zero_scores() {
        let scores = vec![0.0_f32; 8];
        // trace = 0 (for zero scores with zero Jacobian)
        let loss = sm_ism_loss(&scores, 0.0, 8).unwrap();
        assert!(
            loss.abs() < 1e-10,
            "ISM loss should be 0 for zero scores, got {loss}"
        );
    }

    #[test]
    fn test_ism_loss_positive_for_nonzero_scores() {
        let scores = vec![1.0_f32; 8];
        // half_norm_sq = 0.5 * 8 / 8 = 0.5, trace = 0 => loss = 0.5
        let loss = sm_ism_loss(&scores, 0.0, 8).unwrap();
        assert!((loss - 0.5).abs() < 1e-5, "loss={loss}");
    }

    #[test]
    fn test_ism_loss_trace_contribution() {
        let scores = vec![0.0_f32; 4];
        let trace = 3.0;
        let loss = sm_ism_loss(&scores, trace, 4).unwrap();
        assert!(
            (loss - trace).abs() < 1e-5,
            "loss={loss} expected trace={trace}"
        );
    }

    #[test]
    fn test_ism_loss_empty_error() {
        assert!(sm_ism_loss(&[], 0.0, 0).is_err());
    }

    // ── sm_sliced_score_matching_loss ─────────────────────────────────────

    #[test]
    fn test_sliced_sm_zero_scores() {
        let n = 4;
        let d = 3;
        let n_proj = 2;
        let zeros = vec![0.0_f32; n * d];
        let proj = vec![1.0_f32; n_proj * d];
        let loss =
            sm_sliced_score_matching_loss(&zeros, &zeros, &proj, 0.01, n, d, n_proj).unwrap();
        assert!(
            loss.abs() < 1e-10,
            "SSM loss should be 0 for zero scores, got {loss}"
        );
    }

    #[test]
    fn test_sliced_sm_dimension_error() {
        let n = 3;
        let d = 4;
        let n_proj = 2;
        let s = vec![0.0_f32; n * d];
        let bad_proj = vec![0.0_f32; n_proj * (d + 1)]; // wrong size
        assert!(sm_sliced_score_matching_loss(&s, &s, &bad_proj, 0.01, n, d, n_proj).is_err());
    }

    #[test]
    fn test_sliced_sm_invalid_eps() {
        let n = 2;
        let d = 3;
        let n_proj = 1;
        let s = vec![0.0_f32; n * d];
        let proj = vec![1.0_f32; n_proj * d];
        assert!(sm_sliced_score_matching_loss(&s, &s, &proj, 0.0, n, d, n_proj).is_err());
    }

    #[test]
    fn test_sliced_sm_matches_hutchinson_form() {
        // v^T s = 0, s_perturbed = s => second term = 0, first term = (v^T s)^2 = 0
        let n = 1;
        let d = 4;
        let n_proj = 1;
        let scores = vec![0.0_f32; n * d];
        let perturbed = vec![0.0_f32; n * d];
        let proj = vec![1.0_f32; n_proj * d];
        let loss =
            sm_sliced_score_matching_loss(&scores, &perturbed, &proj, 0.001, n, d, n_proj).unwrap();
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn test_sliced_sm_nonzero_for_nonzero_score() {
        // v^T s ≠ 0 => loss > 0
        let n = 1;
        let d = 2;
        let n_proj = 1;
        let scores = vec![1.0_f32, 1.0];
        let perturbed = vec![1.0_f32, 1.0]; // same => second term = 0
        let proj = vec![1.0_f32, 0.0];
        let loss =
            sm_sliced_score_matching_loss(&scores, &perturbed, &proj, 0.01, n, d, n_proj).unwrap();
        // (v^T s)^2 = 1^2 = 1
        assert!((loss - 1.0).abs() < 1e-5, "loss={loss}");
    }

    // ── ScoreFunction ─────────────────────────────────────────────────────

    #[test]
    fn test_score_function_target_score_formula() {
        let config = ScoreMatchingConfig {
            eps: 0.0,
            ..default_config()
        };
        let sf = ScoreFunction::new(config, 1.0);
        let sigma = 2.0;
        let noise = vec![1.0_f32, -0.5, 0.3];
        let batch = NoisyBatch {
            clean: vec![0.0; 3],
            noisy: vec![0.0; 3],
            noise: noise.clone(),
            sigma,
            n: 3,
            d: 1,
        };
        let target = sf.target_score(&batch);
        let sigma2 = sigma * sigma;
        for (i, (&t, &e)) in target.iter().zip(noise.iter()).enumerate() {
            let expected = -e / sigma2;
            assert!(
                (t - expected).abs() < 1e-6,
                "target[{i}]={t} expected={expected}"
            );
        }
    }

    #[test]
    fn test_score_function_loss_zero_when_perfect() {
        let config = ScoreMatchingConfig {
            loss_weighting: SmWeighting::Uniform,
            eps: 0.0,
            ..default_config()
        };
        let sf = ScoreFunction::new(config, 1.0);
        let sigma = 1.0;
        let noise = vec![0.5_f32, -0.3];
        let sigma2 = sigma * sigma;
        let target: Vec<f32> = noise.iter().map(|&e| -e / sigma2).collect();
        let batch = NoisyBatch {
            clean: vec![0.0; 2],
            noisy: vec![0.5, -0.3],
            noise,
            sigma,
            n: 2,
            d: 1,
        };
        let loss = sf.loss(&target, &batch).unwrap();
        assert!(loss.abs() < 1e-6, "loss={loss}");
    }

    #[test]
    fn test_score_function_weighted_loss_equals_loss() {
        let config = default_config();
        let sf = ScoreFunction::new(config, 1.0);
        let clean = zero_clean(4, 4);
        let batch = sm_add_noise(&clean, 1.0, 42, 4, 4).unwrap();
        let pred = vec![0.0_f32; 16];
        let l1 = sf.loss(&pred, &batch).unwrap();
        let l2 = sf.weighted_loss(&pred, &batch).unwrap();
        assert!((l1 - l2).abs() < 1e-10);
    }

    #[test]
    fn test_preconditioned_target_shape() {
        let config = default_config();
        let sf = ScoreFunction::new(config, 1.0);
        let n = 8;
        let d = 4;
        let clean = zero_clean(n, d);
        let batch = sm_add_noise(&clean, 1.0, 42, n, d).unwrap();
        let t = sf.preconditioned_target(&batch);
        assert_eq!(t.len(), n * d);
    }

    // ── ScoreMatchingStats ────────────────────────────────────────────────

    #[test]
    fn test_compute_stats_correct_mean() {
        let losses = vec![1.0_f32, 2.0, 3.0, 4.0];
        let sigmas = vec![1.0_f32; 4];
        let stats = sm_compute_stats(&losses, &sigmas);
        assert!((stats.mean_loss - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_compute_stats_correct_min_max() {
        let losses = vec![0.5_f32, 1.5, 3.0, 0.1];
        let sigmas = vec![1.0_f32; 4];
        let stats = sm_compute_stats(&losses, &sigmas);
        assert!((stats.min_loss - 0.1).abs() < 1e-6);
        assert!((stats.max_loss - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_stats_n_samples() {
        let losses = vec![0.0_f32; 10];
        let sigmas = vec![1.0_f32; 10];
        let stats = sm_compute_stats(&losses, &sigmas);
        assert_eq!(stats.n_samples, 10);
    }

    #[test]
    fn test_compute_stats_empty() {
        let stats = sm_compute_stats(&[], &[]);
        assert_eq!(stats.n_samples, 0);
        assert_eq!(stats.mean_loss, 0.0);
        assert_eq!(stats.effective_weight, 1.0);
    }

    #[test]
    fn test_compute_stats_mean_sigma() {
        let losses = vec![1.0_f32, 1.0];
        let sigmas = vec![2.0_f32, 4.0];
        let stats = sm_compute_stats(&losses, &sigmas);
        assert!((stats.mean_sigma - 3.0).abs() < 1e-6);
    }

    // ── sm_format_stats / sm_format_config ────────────────────────────────

    #[test]
    fn test_format_stats_non_empty() {
        let stats = ScoreMatchingStats {
            mean_loss: 0.5,
            min_loss: 0.1,
            max_loss: 0.9,
            mean_sigma: 1.0,
            n_samples: 10,
            effective_weight: 1.0,
        };
        let s = sm_format_stats(&stats);
        assert!(s.contains("n=10"), "formatted string missing n: {s}");
        assert!(s.contains("mean"), "formatted string missing 'mean': {s}");
    }

    #[test]
    fn test_format_config_non_empty() {
        let config = default_config();
        let s = sm_format_config(&config);
        assert!(s.contains("sigma"), "formatted string missing 'sigma': {s}");
        assert!(
            s.contains("n_sigmas"),
            "formatted string missing 'n_sigmas': {s}"
        );
    }

    // ── Edge cases ────────────────────────────────────────────────────────

    #[test]
    fn test_single_sample_batch() {
        let clean = vec![1.0_f32, 2.0, 3.0];
        let batch = sm_add_noise(&clean, 0.5, 1, 1, 3).unwrap();
        assert_eq!(batch.n, 1);
        assert_eq!(batch.d, 3);
        let pred = vec![0.0_f32; 3];
        let loss = sm_dsm_loss(&pred, &batch, &default_config()).unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn test_empty_clean_dimension_error() {
        let clean: Vec<f32> = vec![];
        assert!(sm_add_noise(&clean, 1.0, 42, 1, 1).is_err());
    }

    #[test]
    fn test_ism_loss_n_zero_error() {
        assert!(sm_ism_loss(&[1.0], 0.0, 0).is_err());
    }

    #[test]
    fn test_sliced_sm_zero_n_error() {
        let s = vec![1.0_f32; 4];
        let proj = vec![1.0_f32; 4];
        assert!(sm_sliced_score_matching_loss(&s, &s, &proj, 0.01, 0, 4, 1).is_err());
    }

    #[test]
    fn test_ln2_constant_sanity() {
        // Verify the natural log of 2 matches the standard library value
        assert!((std::f32::consts::LN_2 - 2.0_f32.ln()).abs() < 1e-6);
    }
}
