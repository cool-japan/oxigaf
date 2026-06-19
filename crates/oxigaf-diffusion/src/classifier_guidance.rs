//! Classifier guidance for diffusion sampling.
//!
//! Classifier guidance steers diffusion sampling toward a desired attribute by
//! following the gradient of a classifier (or surrogate score function) w.r.t.
//! the noisy latent. This is distinct from CFG (which uses conditional vs
//! unconditional score). Classifier guidance uses an external signal — e.g.,
//! "make this face look more like the target" — expressed as a differentiable
//! score.
//!
//! Since autograd is not available in pure Rust, this module implements:
//!
//! 1. Finite-difference gradient approximation (perturb latent, evaluate score,
//!    estimate gradient)
//! 2. Latent steering in gradient direction
//! 3. Score function interfaces (trait-based, with CPU-implementable mock scorers)
//! 4. Multi-objective guidance (combine multiple scoring signals)
//! 5. Stochastic gradient estimation via SPSA for large latents
//!
//! ## Example
//!
//! ```
//! use oxigaf_diffusion::classifier_guidance::{
//!     MeanMaximizer, GuidanceConfig, apply_guidance_step,
//! };
//!
//! let latent = vec![0.0f32; 16];
//! let config = GuidanceConfig::default();
//! let (updated, score) = apply_guidance_step(&latent, &MeanMaximizer, &config).unwrap();
//! // score improves since guidance pushes latent mean upward
//! assert!(score <= updated.iter().sum::<f32>() / updated.len() as f32);
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during classifier guidance operations.
#[derive(Debug, Error, PartialEq)]
pub enum ClassifierGuidanceError {
    /// Invalid configuration parameter.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// Dimension mismatch between latent and reference.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// No guidance components specified.
    #[error("Empty guidance: at least one score function is required")]
    EmptyGuidance,

    /// Score function evaluation failed.
    #[error("Score evaluation failed: {0}")]
    ScoreEvaluationFailed(String),
}

// ---------------------------------------------------------------------------
// Score function trait
// ---------------------------------------------------------------------------

/// A differentiable score function for classifier guidance.
///
/// Takes a latent vector and returns a scalar score (higher = better).
pub trait ScoreFunction: Send + Sync {
    /// Evaluate the score for a given latent vector.
    fn score(&self, latent: &[f32]) -> Result<f32, ClassifierGuidanceError>;

    /// Human-readable name for this score function.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Concrete score functions
// ---------------------------------------------------------------------------

/// Score that maximizes the mean of the latent (toy example).
///
/// Returns the arithmetic mean: `sum(latent) / len(latent)`.
pub struct MeanMaximizer;

impl ScoreFunction for MeanMaximizer {
    fn score(&self, latent: &[f32]) -> Result<f32, ClassifierGuidanceError> {
        if latent.is_empty() {
            return Err(ClassifierGuidanceError::ScoreEvaluationFailed(
                "MeanMaximizer received empty latent".to_string(),
            ));
        }
        let sum: f32 = latent.iter().sum();
        Ok(sum / latent.len() as f32)
    }

    fn name(&self) -> &str {
        "mean_maximizer"
    }
}

/// Score based on L2 proximity to a target latent.
///
/// Returns `-l2_distance(latent, target)` so that higher score means closer
/// to the target (negated because we maximise score).
pub struct TargetProximity {
    /// The reference latent that we want to approach.
    pub target: Vec<f32>,
}

impl TargetProximity {
    /// Create a new `TargetProximity` scorer with the given target latent.
    pub fn new(target: Vec<f32>) -> Self {
        Self { target }
    }
}

impl ScoreFunction for TargetProximity {
    fn score(&self, latent: &[f32]) -> Result<f32, ClassifierGuidanceError> {
        if latent.len() != self.target.len() {
            return Err(ClassifierGuidanceError::DimensionMismatch {
                expected: self.target.len(),
                got: latent.len(),
            });
        }
        let dist_sq: f32 = latent
            .iter()
            .zip(self.target.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        Ok(-dist_sq.sqrt())
    }

    fn name(&self) -> &str {
        "target_proximity"
    }
}

/// Score that penalizes large latent magnitude (L2 regularizer).
///
/// Returns `-lambda * sum(x^2)` so that higher score means smaller norm.
pub struct L2Regularizer {
    /// Regularization strength.
    pub lambda: f32,
}

impl ScoreFunction for L2Regularizer {
    fn score(&self, latent: &[f32]) -> Result<f32, ClassifierGuidanceError> {
        let sq_sum: f32 = latent.iter().map(|x| x * x).sum();
        Ok(-self.lambda * sq_sum)
    }

    fn name(&self) -> &str {
        "l2_regularizer"
    }
}

// ---------------------------------------------------------------------------
// Finite-difference gradient estimation
// ---------------------------------------------------------------------------

/// Estimate the gradient of a score function w.r.t. a latent using central differences.
///
/// For each dimension `i`:
/// ```text
/// grad[i] = (score(latent + eps*e_i) - score(latent - eps*e_i)) / (2 * eps)
/// ```
///
/// # Errors
///
/// - [`ClassifierGuidanceError::InvalidConfig`] if `eps <= 0`.
/// - Propagates errors from the score function.
pub fn fd_gradient<F: ScoreFunction>(
    latent: &[f32],
    score_fn: &F,
    eps: f32,
) -> Result<Vec<f32>, ClassifierGuidanceError> {
    if eps <= 0.0 {
        return Err(ClassifierGuidanceError::InvalidConfig(format!(
            "fd_gradient eps must be > 0, got {eps}"
        )));
    }

    let n = latent.len();
    let mut perturbed = latent.to_vec();
    let mut gradient = vec![0.0f32; n];

    for i in 0..n {
        // Forward perturbation
        perturbed[i] = latent[i] + eps;
        let s_plus = score_fn.score(&perturbed)?;

        // Backward perturbation
        perturbed[i] = latent[i] - eps;
        let s_minus = score_fn.score(&perturbed)?;

        // Restore
        perturbed[i] = latent[i];

        gradient[i] = (s_plus - s_minus) / (2.0 * eps);
    }

    Ok(gradient)
}

/// Estimate gradient using one-sided (forward) differences (faster, less accurate).
///
/// For each dimension `i`:
/// ```text
/// grad[i] = (score(latent + eps*e_i) - score(latent)) / eps
/// ```
///
/// # Errors
///
/// - [`ClassifierGuidanceError::InvalidConfig`] if `eps <= 0`.
/// - Propagates errors from the score function.
pub fn fd_gradient_forward<F: ScoreFunction>(
    latent: &[f32],
    score_fn: &F,
    eps: f32,
) -> Result<Vec<f32>, ClassifierGuidanceError> {
    if eps <= 0.0 {
        return Err(ClassifierGuidanceError::InvalidConfig(format!(
            "fd_gradient_forward eps must be > 0, got {eps}"
        )));
    }

    let n = latent.len();
    let s_base = score_fn.score(latent)?;
    let mut perturbed = latent.to_vec();
    let mut gradient = vec![0.0f32; n];

    for i in 0..n {
        perturbed[i] = latent[i] + eps;
        let s_plus = score_fn.score(&perturbed)?;
        perturbed[i] = latent[i]; // restore
        gradient[i] = (s_plus - s_base) / eps;
    }

    Ok(gradient)
}

// ---------------------------------------------------------------------------
// Gradient clipping
// ---------------------------------------------------------------------------

/// Clip a gradient vector to have at most `max_norm` L2 norm.
///
/// If the gradient's L2 norm exceeds `max_norm`, it is scaled down so that
/// the norm equals `max_norm`. If `max_norm <= 0`, no clipping is applied.
pub fn clip_gradient(gradient: &mut [f32], max_norm: f32) {
    if max_norm <= 0.0 {
        return;
    }
    let norm_sq: f32 = gradient.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in gradient.iter_mut() {
            *g *= scale;
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for one step of classifier guidance.
#[derive(Debug, Clone)]
pub struct GuidanceConfig {
    /// Step size for finite-difference gradient estimation.
    ///
    /// Default: `1e-3`.
    pub fd_eps: f32,

    /// Guidance scale (multiplier on the gradient direction).
    ///
    /// Default: `1.0`.
    pub guidance_scale: f32,

    /// Maximum gradient L2 norm to clip to. `0.0` disables clipping.
    ///
    /// Default: `0.0`.
    pub grad_clip: f32,

    /// If `true`, use central differences; if `false`, use forward differences.
    ///
    /// Central differences are more accurate (O(eps²) error) but require
    /// 2N score evaluations vs N+1 for forward differences.
    ///
    /// Default: `true`.
    pub use_central_diff: bool,

    /// Number of random directions for stochastic gradient estimation.
    /// `0` means full (per-dimension) finite differences.
    ///
    /// Default: `0`.
    pub num_random_directions: usize,
}

impl Default for GuidanceConfig {
    fn default() -> Self {
        Self {
            fd_eps: 1e-3,
            guidance_scale: 1.0,
            grad_clip: 0.0,
            use_central_diff: true,
            num_random_directions: 0,
        }
    }
}

impl GuidanceConfig {
    /// Validate that all config fields are within acceptable ranges.
    ///
    /// # Errors
    ///
    /// - [`ClassifierGuidanceError::InvalidConfig`] if `fd_eps <= 0`,
    ///   `guidance_scale < 0`, or `grad_clip < 0`.
    pub fn validate(&self) -> Result<(), ClassifierGuidanceError> {
        if self.fd_eps <= 0.0 {
            return Err(ClassifierGuidanceError::InvalidConfig(format!(
                "fd_eps must be > 0, got {}",
                self.fd_eps
            )));
        }
        if self.guidance_scale < 0.0 {
            return Err(ClassifierGuidanceError::InvalidConfig(format!(
                "guidance_scale must be >= 0, got {}",
                self.guidance_scale
            )));
        }
        if self.grad_clip < 0.0 {
            return Err(ClassifierGuidanceError::InvalidConfig(format!(
                "grad_clip must be >= 0, got {}",
                self.grad_clip
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Core guidance step
// ---------------------------------------------------------------------------

/// Apply one step of classifier guidance to a latent.
///
/// Updates the latent by adding `guidance_scale * gradient`. Returns the
/// updated latent and the score evaluated at the *input* latent (before update).
///
/// The gradient is estimated using the method configured in `config`:
/// - If `num_random_directions > 0`: SPSA stochastic gradient (seeded by 0).
/// - If `use_central_diff`: central finite differences.
/// - Otherwise: forward finite differences.
///
/// # Errors
///
/// - [`ClassifierGuidanceError::InvalidConfig`] if config is invalid.
/// - Propagates errors from score evaluation.
pub fn apply_guidance_step<F: ScoreFunction>(
    latent: &[f32],
    score_fn: &F,
    config: &GuidanceConfig,
) -> Result<(Vec<f32>, f32), ClassifierGuidanceError> {
    config.validate()?;

    let score_before = score_fn.score(latent)?;

    let mut gradient = if config.num_random_directions > 0 {
        stochastic_fd_gradient(
            latent,
            score_fn,
            config.fd_eps,
            config.num_random_directions,
            0,
        )
    } else if config.use_central_diff {
        fd_gradient(latent, score_fn, config.fd_eps)
    } else {
        fd_gradient_forward(latent, score_fn, config.fd_eps)
    }?;

    if config.grad_clip > 0.0 {
        clip_gradient(&mut gradient, config.grad_clip);
    }

    let updated: Vec<f32> = latent
        .iter()
        .zip(gradient.iter())
        .map(|(x, g)| x + config.guidance_scale * g)
        .collect();

    Ok((updated, score_before))
}

// ---------------------------------------------------------------------------
// Stochastic gradient estimation (SPSA)
// ---------------------------------------------------------------------------

/// Inline xorshift64 PRNG — deterministic, no external crate required.
///
/// Returns the next state and a pseudo-random `u64`.
#[inline]
fn xorshift64(state: u64) -> (u64, u64) {
    let mut s = state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    (s, s)
}

/// Estimate gradient using random orthogonal directions (SPSA-like).
///
/// Perturbs the latent in `num_directions` random Rademacher ±1 directions and
/// accumulates gradient estimates. This is the SPSA (Simultaneous Perturbation
/// Stochastic Approximation) estimator:
///
/// ```text
/// For k in 0..num_directions:
///   sample delta[i] ∈ {+1, -1}  (Rademacher)
///   grad += (score(latent + eps*delta) - score(latent - eps*delta)) / (2*eps) * delta
/// grad /= num_directions
/// ```
///
/// Uses xorshift64 seeded by `seed` for reproducibility.
///
/// # Errors
///
/// - [`ClassifierGuidanceError::InvalidConfig`] if `eps <= 0` or
///   `num_directions == 0`.
/// - Propagates errors from score evaluation.
pub fn stochastic_fd_gradient<F: ScoreFunction>(
    latent: &[f32],
    score_fn: &F,
    eps: f32,
    num_directions: usize,
    seed: u64,
) -> Result<Vec<f32>, ClassifierGuidanceError> {
    if eps <= 0.0 {
        return Err(ClassifierGuidanceError::InvalidConfig(format!(
            "stochastic_fd_gradient eps must be > 0, got {eps}"
        )));
    }
    if num_directions == 0 {
        return Err(ClassifierGuidanceError::InvalidConfig(
            "num_directions must be > 0 for stochastic gradient".to_string(),
        ));
    }

    let n = latent.len();
    let mut gradient = vec![0.0f32; n];
    // Use a non-zero seed (xorshift64 requires non-zero state)
    let mut rng_state = if seed == 0 {
        0xDEAD_BEEF_1234_5678u64
    } else {
        seed
    };

    for _ in 0..num_directions {
        // Sample Rademacher delta vector: +1 or -1 per dimension
        let mut delta = vec![0.0f32; n];
        for d in delta.iter_mut() {
            let (next_state, val) = xorshift64(rng_state);
            rng_state = next_state;
            *d = if val & 1 == 0 { 1.0 } else { -1.0 };
        }

        // latent + eps * delta
        let latent_plus: Vec<f32> = latent
            .iter()
            .zip(delta.iter())
            .map(|(x, d)| x + eps * d)
            .collect();

        // latent - eps * delta
        let latent_minus: Vec<f32> = latent
            .iter()
            .zip(delta.iter())
            .map(|(x, d)| x - eps * d)
            .collect();

        let s_plus = score_fn.score(&latent_plus)?;
        let s_minus = score_fn.score(&latent_minus)?;

        let coeff = (s_plus - s_minus) / (2.0 * eps);
        for (g, d) in gradient.iter_mut().zip(delta.iter()) {
            *g += coeff * d;
        }
    }

    // Average over directions
    let n_dirs = num_directions as f32;
    for g in gradient.iter_mut() {
        *g /= n_dirs;
    }

    Ok(gradient)
}

// ---------------------------------------------------------------------------
// Multi-objective guidance
// ---------------------------------------------------------------------------

/// A weighted combination of multiple score functions.
///
/// The composite score is: `sum_i (weight_i * score_i(latent))`.
pub struct CompositeScoreFunction {
    components: Vec<(Box<dyn ScoreFunction>, f32)>,
}

impl CompositeScoreFunction {
    /// Create an empty composite score function.
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
        }
    }

    /// Add a score function with a given weight (builder-style).
    pub fn add(mut self, scorer: Box<dyn ScoreFunction>, weight: f32) -> Self {
        self.components.push((scorer, weight));
        self
    }

    /// Number of component score functions.
    pub fn num_components(&self) -> usize {
        self.components.len()
    }

    /// Sum of absolute values of component weights.
    pub fn total_weight(&self) -> f32 {
        self.components.iter().map(|(_, w)| w.abs()).sum()
    }
}

impl Default for CompositeScoreFunction {
    fn default() -> Self {
        Self::new()
    }
}

impl ScoreFunction for CompositeScoreFunction {
    fn score(&self, latent: &[f32]) -> Result<f32, ClassifierGuidanceError> {
        // Empty composite returns 0.0 (no guidance)
        if self.components.is_empty() {
            return Ok(0.0);
        }
        let mut total = 0.0f32;
        for (scorer, weight) in &self.components {
            let s = scorer.score(latent)?;
            total += weight * s;
        }
        Ok(total)
    }

    fn name(&self) -> &str {
        "composite"
    }
}

// ---------------------------------------------------------------------------
// Guidance statistics
// ---------------------------------------------------------------------------

/// Statistics collected during a single guidance step.
#[derive(Debug, Clone)]
pub struct GuidanceStats {
    /// Which step index (0-based).
    pub step: usize,
    /// Score evaluated at the input latent (before update).
    pub score_before: f32,
    /// Score evaluated at the updated latent (after update).
    pub score_after: f32,
    /// `score_after - score_before`.
    pub score_delta: f32,
    /// L2 norm of the gradient used for the update.
    pub gradient_norm: f32,
    /// L2 norm of the change applied to the latent (`guidance_scale * gradient`).
    pub latent_update_norm: f32,
}

/// Run multiple guidance steps and collect per-step statistics.
///
/// Returns `(final_latent, stats_per_step)`.
///
/// # Errors
///
/// - Propagates errors from [`apply_guidance_step`] or score evaluation.
pub fn run_guidance_steps<F: ScoreFunction>(
    initial_latent: Vec<f32>,
    score_fn: &F,
    config: &GuidanceConfig,
    num_steps: usize,
) -> Result<(Vec<f32>, Vec<GuidanceStats>), ClassifierGuidanceError> {
    config.validate()?;

    let mut latent = initial_latent;
    let mut stats_vec = Vec::with_capacity(num_steps);

    for step in 0..num_steps {
        let (updated, score_before) = apply_guidance_step(&latent, score_fn, config)?;
        let score_after = score_fn.score(&updated)?;
        let score_delta = score_after - score_before;

        // Compute gradient norm: derive from update norm and guidance_scale
        // We need the actual gradient, so recompute gradient norm from update delta.
        // latent_update = updated - latent = guidance_scale * gradient
        let update_delta: Vec<f32> = updated
            .iter()
            .zip(latent.iter())
            .map(|(u, x)| u - x)
            .collect();
        let latent_update_norm: f32 = update_delta.iter().map(|d| d * d).sum::<f32>().sqrt();

        // gradient_norm = latent_update_norm / guidance_scale (guard against zero scale)
        let gradient_norm = if config.guidance_scale.abs() > f32::EPSILON {
            latent_update_norm / config.guidance_scale.abs()
        } else {
            0.0
        };

        stats_vec.push(GuidanceStats {
            step,
            score_before,
            score_after,
            score_delta,
            gradient_norm,
            latent_update_norm,
        });

        latent = updated;
    }

    Ok((latent, stats_vec))
}

// ---------------------------------------------------------------------------
// Annealing helper
// ---------------------------------------------------------------------------

/// Anneal guidance scale over steps using cosine decay from `scale_start` to `scale_end`.
///
/// ```text
/// t = min(step, total_steps) / max(total_steps, 1)
/// cos_factor = 0.5 * (1 + cos(π * t))
/// output = scale_start * cos_factor + scale_end * (1 - cos_factor)
/// ```
///
/// At `step == 0`:  output ≈ `scale_start`.
/// At `step == total_steps`: output ≈ `scale_end`.
pub fn annealed_guidance_scale(
    step: usize,
    total_steps: usize,
    scale_start: f32,
    scale_end: f32,
) -> f32 {
    let t = step.min(total_steps) as f32 / total_steps.max(1) as f32;
    let cos_factor = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
    scale_start * cos_factor + scale_end * (1.0 - cos_factor)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- MeanMaximizer ----

    #[test]
    fn test_mean_maximizer_basic() {
        let latent = vec![1.0f32, 2.0, 3.0, 4.0];
        let score = MeanMaximizer.score(&latent).unwrap();
        assert!(
            (score - 2.5).abs() < 1e-6,
            "mean should be 2.5, got {score}"
        );
    }

    #[test]
    fn test_mean_maximizer_empty_error() {
        let result = MeanMaximizer.score(&[]);
        assert!(result.is_err(), "empty latent should error");
    }

    #[test]
    fn test_mean_maximizer_name() {
        assert_eq!(MeanMaximizer.name(), "mean_maximizer");
    }

    // ---- TargetProximity ----

    #[test]
    fn test_target_proximity_matching_latent_is_zero() {
        let target = vec![1.0f32, 2.0, 3.0];
        let scorer = TargetProximity::new(target.clone());
        let score = scorer.score(&target).unwrap();
        assert!(
            score.abs() < 1e-6,
            "score should be 0.0 when latent == target, got {score}"
        );
    }

    #[test]
    fn test_target_proximity_different_latent_is_negative() {
        let target = vec![1.0f32, 2.0, 3.0];
        let scorer = TargetProximity::new(target);
        let latent = vec![0.0f32, 0.0, 0.0];
        let score = scorer.score(&latent).unwrap();
        assert!(
            score < 0.0,
            "score should be negative (negated distance), got {score}"
        );
    }

    #[test]
    fn test_target_proximity_dimension_mismatch_errors() {
        let scorer = TargetProximity::new(vec![1.0f32, 2.0]);
        let result = scorer.score(&[1.0f32, 2.0, 3.0]);
        assert!(
            matches!(
                result,
                Err(ClassifierGuidanceError::DimensionMismatch {
                    expected: 2,
                    got: 3
                })
            ),
            "expected DimensionMismatch, got {result:?}"
        );
    }

    #[test]
    fn test_target_proximity_name() {
        let scorer = TargetProximity::new(vec![0.0f32]);
        assert_eq!(scorer.name(), "target_proximity");
    }

    // ---- L2Regularizer ----

    #[test]
    fn test_l2_regularizer_lambda_effect() {
        let latent = vec![1.0f32, 0.0, 0.0];
        let s1 = L2Regularizer { lambda: 1.0 }.score(&latent).unwrap();
        let s2 = L2Regularizer { lambda: 2.0 }.score(&latent).unwrap();
        // s1 = -1.0, s2 = -2.0
        assert!((s1 - (-1.0)).abs() < 1e-6, "got {s1}");
        assert!((s2 - (-2.0)).abs() < 1e-6, "got {s2}");
    }

    #[test]
    fn test_l2_regularizer_zero_latent() {
        let latent = vec![0.0f32; 4];
        let score = L2Regularizer { lambda: 5.0 }.score(&latent).unwrap();
        assert!(
            score.abs() < 1e-6,
            "zero latent should give zero score, got {score}"
        );
    }

    #[test]
    fn test_l2_regularizer_name() {
        assert_eq!(L2Regularizer { lambda: 1.0 }.name(), "l2_regularizer");
    }

    // ---- fd_gradient (central differences) ----

    #[test]
    fn test_fd_gradient_mean_maximizer() {
        // MeanMaximizer gradient = 1/n for each dimension
        let latent = vec![0.0f32; 8];
        let grad = fd_gradient(&latent, &MeanMaximizer, 1e-3).unwrap();
        assert_eq!(grad.len(), 8);
        let expected = 1.0 / 8.0;
        for (i, g) in grad.iter().enumerate() {
            assert!(
                (g - expected).abs() < 1e-5,
                "grad[{i}] = {g}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_fd_gradient_target_proximity() {
        // TargetProximity: gradient points from latent toward target
        let target = vec![1.0f32, 0.0, 0.0];
        let scorer = TargetProximity::new(target.clone());
        let latent = vec![0.0f32, 0.0, 0.0];
        let grad = fd_gradient(&latent, &scorer, 1e-4).unwrap();
        assert_eq!(grad.len(), 3);
        // Gradient of -||x - t|| w.r.t. x[0] at x=0, t=[1,0,0]:
        // = -(x[0] - t[0]) / ||x - t|| = -(-1) / 1 = 1.0
        assert!(
            grad[0] > 0.0,
            "gradient in dim 0 should be positive (point toward target)"
        );
    }

    #[test]
    fn test_fd_gradient_invalid_eps() {
        let latent = vec![0.0f32; 4];
        let result = fd_gradient(&latent, &MeanMaximizer, 0.0);
        assert!(result.is_err());
        let result = fd_gradient(&latent, &MeanMaximizer, -1e-3);
        assert!(result.is_err());
    }

    // ---- fd_gradient_forward ----

    #[test]
    fn test_fd_gradient_forward_sign_correct() {
        // Forward difference gradient of MeanMaximizer should also be positive
        let latent = vec![0.0f32; 8];
        let grad = fd_gradient_forward(&latent, &MeanMaximizer, 1e-3).unwrap();
        assert_eq!(grad.len(), 8);
        for g in &grad {
            assert!(
                *g > 0.0,
                "forward gradient for MeanMaximizer should be positive, got {g}"
            );
        }
    }

    #[test]
    fn test_fd_gradient_forward_invalid_eps() {
        let latent = vec![0.0f32; 4];
        let result = fd_gradient_forward(&latent, &MeanMaximizer, 0.0);
        assert!(result.is_err());
    }

    // ---- GuidanceConfig ----

    #[test]
    fn test_guidance_config_validate_valid() {
        let config = GuidanceConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_guidance_config_validate_zero_eps() {
        let config = GuidanceConfig {
            fd_eps: 0.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_guidance_config_validate_negative_scale() {
        let config = GuidanceConfig {
            guidance_scale: -1.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_guidance_config_validate_negative_grad_clip() {
        let config = GuidanceConfig {
            grad_clip: -0.1,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    // ---- clip_gradient ----

    #[test]
    fn test_clip_gradient_zero_gradient_unchanged() {
        let mut grad = vec![0.0f32; 4];
        clip_gradient(&mut grad, 1.0);
        for g in &grad {
            assert!(g.abs() < 1e-9, "zero gradient should remain zero");
        }
    }

    #[test]
    fn test_clip_gradient_small_norm_unchanged() {
        // norm = sqrt(1+0+0) = 1.0, max_norm = 2.0 → no clipping
        let mut grad = vec![1.0f32, 0.0, 0.0];
        let orig = grad.clone();
        clip_gradient(&mut grad, 2.0);
        for (g, o) in grad.iter().zip(orig.iter()) {
            assert!((g - o).abs() < 1e-6, "small gradient should not be clipped");
        }
    }

    #[test]
    fn test_clip_gradient_large_norm_clipped() {
        // norm = sqrt(3*9) = 3*sqrt(3) ≈ 5.196, max_norm = 1.0
        let mut grad = vec![3.0f32, 3.0, 3.0];
        clip_gradient(&mut grad, 1.0);
        let norm: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "norm after clip should be 1.0, got {norm}"
        );
    }

    // ---- apply_guidance_step ----

    #[test]
    fn test_apply_guidance_step_mean_maximizer_score_improves() {
        let latent = vec![0.0f32; 8];
        let config = GuidanceConfig {
            guidance_scale: 0.1,
            ..Default::default()
        };
        let (updated, score_before) =
            apply_guidance_step(&latent, &MeanMaximizer, &config).unwrap();
        let score_after = MeanMaximizer.score(&updated).unwrap();
        assert!(
            score_after > score_before,
            "score should improve: before={score_before}, after={score_after}"
        );
    }

    #[test]
    fn test_apply_guidance_step_latent_moves_toward_target() {
        let target = vec![10.0f32; 4];
        let scorer = TargetProximity::new(target.clone());
        let latent = vec![0.0f32; 4];
        let config = GuidanceConfig {
            guidance_scale: 0.5,
            ..Default::default()
        };
        let (updated, _) = apply_guidance_step(&latent, &scorer, &config).unwrap();

        // Each updated[i] should be closer to target[i] = 10.0
        let dist_before: f32 = latent
            .iter()
            .zip(target.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let dist_after: f32 = updated
            .iter()
            .zip(target.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            dist_after < dist_before,
            "latent should move toward target: before={dist_before}, after={dist_after}"
        );
    }

    #[test]
    fn test_apply_guidance_step_with_grad_clip() {
        let latent = vec![0.0f32; 4];
        let config = GuidanceConfig {
            guidance_scale: 100.0, // large scale would make huge update without clip
            grad_clip: 0.01,
            ..Default::default()
        };
        let (updated, _) = apply_guidance_step(&latent, &MeanMaximizer, &config).unwrap();
        // Clipping should limit the update size
        let update_norm: f32 = updated
            .iter()
            .zip(latent.iter())
            .map(|(u, x)| (u - x) * (u - x))
            .sum::<f32>()
            .sqrt();
        // Max update = guidance_scale * grad_clip = 100.0 * 0.01 = 1.0
        assert!(
            update_norm <= 1.0 + 1e-5,
            "clipped update norm should be at most 1.0, got {update_norm}"
        );
    }

    // ---- stochastic_fd_gradient ----

    #[test]
    fn test_stochastic_fd_gradient_sign_correct_mean_maximizer() {
        // SPSA gradient for MeanMaximizer should be mostly positive
        let latent = vec![0.0f32; 16];
        let grad = stochastic_fd_gradient(&latent, &MeanMaximizer, 1e-3, 100, 42).unwrap();
        let mean_grad: f32 = grad.iter().sum::<f32>() / grad.len() as f32;
        assert!(
            mean_grad > 0.0,
            "mean SPSA gradient should be positive, got {mean_grad}"
        );
    }

    #[test]
    fn test_stochastic_fd_gradient_dimension_correct() {
        let latent = vec![1.0f32; 12];
        let grad = stochastic_fd_gradient(&latent, &MeanMaximizer, 1e-3, 50, 123).unwrap();
        assert_eq!(grad.len(), 12, "gradient must match latent dimension");
    }

    #[test]
    fn test_stochastic_fd_gradient_invalid_num_directions() {
        let latent = vec![0.0f32; 4];
        let result = stochastic_fd_gradient(&latent, &MeanMaximizer, 1e-3, 0, 0);
        assert!(result.is_err());
    }

    // ---- CompositeScoreFunction ----

    #[test]
    fn test_composite_empty_returns_zero() {
        let composite = CompositeScoreFunction::new();
        let score = composite.score(&[1.0f32, 2.0, 3.0]).unwrap();
        assert!(
            score.abs() < 1e-6,
            "empty composite should return 0.0, got {score}"
        );
    }

    #[test]
    fn test_composite_single_component_matches() {
        let composite = CompositeScoreFunction::new().add(Box::new(MeanMaximizer), 1.0);
        let latent = vec![1.0f32, 3.0];
        let expected = MeanMaximizer.score(&latent).unwrap();
        let got = composite.score(&latent).unwrap();
        assert!(
            (got - expected).abs() < 1e-6,
            "single component: got={got}, expected={expected}"
        );
    }

    #[test]
    fn test_composite_two_components_weighted_sum() {
        let latent = vec![1.0f32, 2.0, 3.0];
        let s_mean = MeanMaximizer.score(&latent).unwrap();
        let s_reg = L2Regularizer { lambda: 1.0 }.score(&latent).unwrap();
        let expected = 2.0 * s_mean + 0.5 * s_reg;

        let composite = CompositeScoreFunction::new()
            .add(Box::new(MeanMaximizer), 2.0)
            .add(Box::new(L2Regularizer { lambda: 1.0 }), 0.5);
        let got = composite.score(&latent).unwrap();
        assert!(
            (got - expected).abs() < 1e-5,
            "weighted sum: got={got}, expected={expected}"
        );
    }

    #[test]
    fn test_composite_num_components_and_total_weight() {
        let composite = CompositeScoreFunction::new()
            .add(Box::new(MeanMaximizer), 2.0)
            .add(Box::new(L2Regularizer { lambda: 1.0 }), -0.5);
        assert_eq!(composite.num_components(), 2);
        // total_weight = |2.0| + |-0.5| = 2.5
        assert!((composite.total_weight() - 2.5).abs() < 1e-6);
    }

    // ---- run_guidance_steps ----

    #[test]
    fn test_run_guidance_steps_zero_steps_empty_stats() {
        let latent = vec![0.0f32; 4];
        let config = GuidanceConfig::default();
        let (final_latent, stats) =
            run_guidance_steps(latent.clone(), &MeanMaximizer, &config, 0).unwrap();
        assert!(stats.is_empty(), "zero steps should produce no stats");
        assert_eq!(
            final_latent, latent,
            "zero steps should leave latent unchanged"
        );
    }

    #[test]
    fn test_run_guidance_steps_mean_maximizer_scores_increase() {
        let latent = vec![0.0f32; 8];
        let config = GuidanceConfig {
            guidance_scale: 0.1,
            ..Default::default()
        };
        let (_, stats) = run_guidance_steps(latent, &MeanMaximizer, &config, 5).unwrap();
        assert_eq!(stats.len(), 5);
        for (i, s) in stats.iter().enumerate() {
            assert!(
                s.score_after > s.score_before,
                "step {i}: score should increase, before={}, after={}",
                s.score_before,
                s.score_after
            );
        }
    }

    #[test]
    fn test_run_guidance_steps_target_proximity_converges() {
        let target = vec![5.0f32; 4];
        let scorer = TargetProximity::new(target.clone());
        let latent = vec![0.0f32; 4];
        let config = GuidanceConfig {
            guidance_scale: 0.5,
            ..Default::default()
        };
        let (final_latent, _) = run_guidance_steps(latent, &scorer, &config, 10).unwrap();

        // Final latent should be closer to target than initial
        let final_dist: f32 = final_latent
            .iter()
            .zip(target.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        let initial_dist = (4.0f32 * 25.0).sqrt(); // sqrt(4 * 5^2) = 10.0
        assert!(
            final_dist < initial_dist,
            "final distance {final_dist} should be less than initial {initial_dist}"
        );
    }

    // ---- annealed_guidance_scale ----

    #[test]
    fn test_annealed_guidance_scale_at_step_zero() {
        let scale = annealed_guidance_scale(0, 100, 5.0, 1.0);
        // t=0, cos_factor=1.0, result=5.0
        assert!(
            (scale - 5.0).abs() < 1e-5,
            "at step=0 should equal scale_start=5.0, got {scale}"
        );
    }

    #[test]
    fn test_annealed_guidance_scale_at_total_steps() {
        let scale = annealed_guidance_scale(100, 100, 5.0, 1.0);
        // t=1.0, cos_factor=0.0, result=1.0
        assert!(
            (scale - 1.0).abs() < 1e-5,
            "at step=total should equal scale_end=1.0, got {scale}"
        );
    }

    #[test]
    fn test_annealed_guidance_scale_monotone_decreasing() {
        let total = 20usize;
        let start = 10.0f32;
        let end = 1.0f32;
        let mut prev = annealed_guidance_scale(0, total, start, end);
        for step in 1..=total {
            let curr = annealed_guidance_scale(step, total, start, end);
            assert!(
                curr <= prev + 1e-6,
                "scale should be non-increasing at step {step}: prev={prev}, curr={curr}"
            );
            prev = curr;
        }
    }

    #[test]
    fn test_annealed_guidance_scale_zero_total_steps() {
        // Should not panic; .max(1) guards against division by zero
        let scale = annealed_guidance_scale(0, 0, 3.0, 1.0);
        // t = min(0,0)/max(0,1) = 0/1 = 0, cos_factor=1.0, result=3.0
        assert!((scale - 3.0).abs() < 1e-5, "got {scale}");
    }
}
