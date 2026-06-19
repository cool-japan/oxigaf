//! Few-step DDIM scheduler with configurable timestep selection strategies.
//!
//! Provides [`StepPattern`] variants for selecting a subset of inference
//! timesteps from the full training schedule, enabling fast inference at
//! 4, 10, 20, or any arbitrary number of steps.

use crate::DiffusionError;

// ---------------------------------------------------------------------------
// StepPattern
// ---------------------------------------------------------------------------

/// Strategy for selecting timestep indices when using fewer inference steps.
#[derive(Debug, Clone, PartialEq)]
pub enum StepPattern {
    /// Evenly spaced steps: step_size = total / num_steps,
    /// indices = [step_size-1, 2*step_size-1, …], reversed (high → low).
    Uniform,
    /// Quadratic spacing: more steps near t=0 for better final-quality
    /// denoising. `timestep[i] = total * (1 - (i/num_steps)^2)`.
    Quadratic,
    /// Exponential spacing: very dense early steps (high-noise region).
    /// `timestep[i] = total * exp(-3.0 * i/num_steps)`.
    Exponential,
    /// Leading steps: first `num_steps` of the original `total` timesteps.
    Leading,
    /// Trailing steps: last `num_steps` of the original `total` timesteps
    /// (the low-noise end near t = 0).
    Trailing,
    /// Custom user-provided timestep indices (must all be < total_training_steps).
    Custom(Vec<usize>),
}

impl StepPattern {
    /// Uniform: step_size = total / num_steps; indices = [step_size-1, …,
    /// num_steps*step_size-1] then reversed so they run high → low.
    pub fn uniform_indices(total: usize, num_steps: usize) -> Vec<usize> {
        if num_steps == 0 || total == 0 {
            return Vec::new();
        }
        let step_size = total / num_steps;
        let step_size = step_size.max(1);
        let mut indices: Vec<usize> = (1..=num_steps)
            .map(|i| {
                let idx = i * step_size - 1;
                idx.min(total - 1)
            })
            .collect();
        // sort descending (high timestep → low)
        indices.sort_unstable_by(|a, b| b.cmp(a));
        indices.dedup();
        indices
    }

    /// Quadratic spacing: `timestep[i] = total * (1 - (i/num_steps)^2)`.
    /// Results are clamped to [0, total-1], deduplicated, sorted descending.
    pub fn quadratic_indices(total: usize, num_steps: usize) -> Vec<usize> {
        if num_steps == 0 || total == 0 {
            return Vec::new();
        }
        let total_f = total as f32;
        let n_f = num_steps as f32;
        let mut indices: Vec<usize> = (0..num_steps)
            .map(|i| {
                let t = i as f32 / n_f;
                let raw = total_f * (1.0 - t * t);
                // clamp to [0, total-1]
                (raw as usize).min(total - 1)
            })
            .collect();
        indices.sort_unstable_by(|a, b| b.cmp(a));
        indices.dedup();
        indices
    }

    /// Exponential spacing: `timestep[i] = total * exp(-3.0 * i/num_steps)`.
    /// Dense at the beginning (high noise), sparse later.
    /// Results clamped to [0, total-1], deduplicated, sorted descending.
    pub fn exponential_indices(total: usize, num_steps: usize) -> Vec<usize> {
        if num_steps == 0 || total == 0 {
            return Vec::new();
        }
        let total_f = total as f32;
        let n_f = num_steps as f32;
        let lambda = 3.0_f32;
        let mut indices: Vec<usize> = (0..num_steps)
            .map(|i| {
                let t = i as f32 / n_f;
                let raw = total_f * (-lambda * t).exp();
                // clamp to [0, total-1]
                (raw as usize).min(total - 1)
            })
            .collect();
        indices.sort_unstable_by(|a, b| b.cmp(a));
        indices.dedup();
        indices
    }

    /// Leading: the first `num_steps` of `0..total`, sorted descending.
    fn leading_indices(total: usize, num_steps: usize) -> Vec<usize> {
        if num_steps == 0 || total == 0 {
            return Vec::new();
        }
        let count = num_steps.min(total);
        let mut indices: Vec<usize> = (0..count).collect();
        indices.sort_unstable_by(|a, b| b.cmp(a));
        indices
    }

    /// Trailing: the last `num_steps` of `0..total`, sorted descending.
    fn trailing_indices(total: usize, num_steps: usize) -> Vec<usize> {
        if num_steps == 0 || total == 0 {
            return Vec::new();
        }
        let count = num_steps.min(total);
        let start = total - count;
        let mut indices: Vec<usize> = (start..total).collect();
        indices.sort_unstable_by(|a, b| b.cmp(a));
        indices
    }

    /// Compute the timestep index list for this pattern.
    ///
    /// For [`StepPattern::Custom`], the provided indices are validated against
    /// `total` and returned as a descending-sorted, deduplicated list.
    pub fn indices(&self, total: usize, num_steps: usize) -> Vec<usize> {
        match self {
            StepPattern::Uniform => Self::uniform_indices(total, num_steps),
            StepPattern::Quadratic => Self::quadratic_indices(total, num_steps),
            StepPattern::Exponential => Self::exponential_indices(total, num_steps),
            StepPattern::Leading => Self::leading_indices(total, num_steps),
            StepPattern::Trailing => Self::trailing_indices(total, num_steps),
            StepPattern::Custom(v) => {
                let mut out = v.clone();
                // Clamp to valid range
                out.retain(|&x| x < total);
                out.sort_unstable_by(|a, b| b.cmp(a));
                out.dedup();
                out
            }
        }
    }
}

// ---------------------------------------------------------------------------
// StepScheduleConfig
// ---------------------------------------------------------------------------

/// Configuration for a few-step DDIM schedule.
#[derive(Debug, Clone)]
pub struct StepScheduleConfig {
    /// Total timesteps the model was trained with (default: 1000).
    pub total_training_steps: usize,
    /// Desired number of inference steps (e.g., 4, 10, 20, 50).
    pub inference_steps: usize,
    /// Timestep selection pattern.
    pub pattern: StepPattern,
    /// Whether to clamp denoised predictions to [-1, 1] (default: true).
    pub clip_denoised: bool,
    /// Whether to set alpha of the final step to 1.0 (default: false).
    pub set_alpha_to_one: bool,
}

impl StepScheduleConfig {
    /// Create a new config with the given inference steps and pattern.
    /// Defaults: `total_training_steps = 1000`, `clip_denoised = true`,
    /// `set_alpha_to_one = false`.
    pub fn new(inference_steps: usize, pattern: StepPattern) -> Self {
        Self {
            total_training_steps: 1000,
            inference_steps,
            pattern,
            clip_denoised: true,
            set_alpha_to_one: false,
        }
    }

    /// 4-step schedule (Trailing) — preview quality.
    pub fn fast_4step() -> Self {
        Self::new(4, StepPattern::Trailing)
    }

    /// 10-step schedule (Uniform) — fast iteration quality.
    pub fn fast_10step() -> Self {
        Self::new(10, StepPattern::Uniform)
    }

    /// 20-step schedule (Quadratic) — good quality.
    pub fn fast_20step() -> Self {
        Self::new(20, StepPattern::Quadratic)
    }

    /// Validate this configuration.
    ///
    /// Returns [`DiffusionError::InvalidConfig`] if inference_steps is 0 or
    /// exceeds total_training_steps.
    pub fn validate(&self) -> Result<(), DiffusionError> {
        if self.inference_steps == 0 {
            return Err(DiffusionError::InvalidConfig(
                "inference_steps must be > 0".into(),
            ));
        }
        if self.inference_steps > self.total_training_steps {
            return Err(DiffusionError::InvalidConfig(format!(
                "inference_steps ({}) exceeds total_training_steps ({})",
                self.inference_steps, self.total_training_steps
            )));
        }
        Ok(())
    }

    /// Compute the ordered list of timestep indices (high → low) for this
    /// configuration.
    pub fn timestep_indices(&self) -> Vec<usize> {
        self.pattern
            .indices(self.total_training_steps, self.inference_steps)
    }
}

// ---------------------------------------------------------------------------
// DdimStepScheduler
// ---------------------------------------------------------------------------

/// DDIM scheduler that uses a few-step schedule for fast inference.
pub struct DdimStepScheduler {
    /// Schedule configuration.
    pub config: StepScheduleConfig,
    /// Precomputed timesteps in descending order (high → low).
    pub timesteps: Vec<usize>,
    /// `alphas_cumprod` values at each selected timestep, in the same order
    /// as `timesteps` (i.e., the first entry corresponds to the highest
    /// timestep).
    pub alphas_cumprod: Vec<f32>,
    current_step: usize,
}

impl DdimStepScheduler {
    /// Create a new few-step DDIM scheduler.
    ///
    /// `alphas_cumprod_all` must have exactly `config.total_training_steps`
    /// entries (one per training timestep, in ascending order).
    pub fn new(
        config: StepScheduleConfig,
        alphas_cumprod_all: &[f32],
    ) -> Result<Self, DiffusionError> {
        config.validate()?;
        if alphas_cumprod_all.len() != config.total_training_steps {
            return Err(DiffusionError::InvalidConfig(format!(
                "alphas_cumprod_all length ({}) != total_training_steps ({})",
                alphas_cumprod_all.len(),
                config.total_training_steps
            )));
        }
        let timesteps = config.timestep_indices();
        let alphas_cumprod: Vec<f32> = timesteps.iter().map(|&t| alphas_cumprod_all[t]).collect();
        Ok(Self {
            config,
            timesteps,
            alphas_cumprod,
            current_step: 0,
        })
    }

    /// Return the current timestep, or `None` if all steps are done.
    pub fn current_timestep(&self) -> Option<usize> {
        self.timesteps.get(self.current_step).copied()
    }

    /// Perform one deterministic DDIM update step.
    ///
    /// Standard DDIM formula (η = 0):
    /// ```text
    /// alpha_t    = alphas_cumprod[current_step]
    /// alpha_prev = alphas_cumprod[current_step + 1]  (or 1.0 at the last step)
    /// x0_pred    = (sample - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)
    /// dir_xt     = sqrt(1 - alpha_prev) * noise_pred
    /// prev       = sqrt(alpha_prev) * x0_pred + dir_xt
    /// ```
    ///
    /// If `config.clip_denoised` is `true`, `x0_pred` is clamped to [-1, 1].
    ///
    /// Advances the internal step counter.
    pub fn step_with_noise_pred(
        &mut self,
        noise_pred: &[f32],
        sample: &[f32],
    ) -> Result<Vec<f32>, DiffusionError> {
        if self.is_done() {
            return Err(DiffusionError::Inference(
                "DdimStepScheduler: all steps exhausted".into(),
            ));
        }
        if noise_pred.len() != sample.len() {
            return Err(DiffusionError::Inference(format!(
                "noise_pred length ({}) != sample length ({})",
                noise_pred.len(),
                sample.len()
            )));
        }

        let alpha_t = self.alphas_cumprod[self.current_step];

        // alpha_prev: next entry in our schedule, or handle the final step.
        let alpha_prev = if self.current_step + 1 < self.alphas_cumprod.len() {
            self.alphas_cumprod[self.current_step + 1]
        } else if self.config.set_alpha_to_one {
            1.0_f32
        } else {
            // Use the actual alpha at the last selected timestep (same as
            // current since we've reached the end).
            self.alphas_cumprod[self.current_step]
        };

        let sqrt_alpha_t = alpha_t.sqrt();
        let sqrt_one_minus_alpha_t = (1.0 - alpha_t).sqrt();
        let sqrt_alpha_prev = alpha_prev.sqrt();
        let sqrt_one_minus_alpha_prev = (1.0 - alpha_prev).sqrt();

        let n = sample.len();
        let mut prev_sample = Vec::with_capacity(n);

        for i in 0..n {
            // Predict clean image x0
            let mut x0_pred = (sample[i] - sqrt_one_minus_alpha_t * noise_pred[i]) / sqrt_alpha_t;

            // Optionally clamp denoised prediction
            if self.config.clip_denoised {
                x0_pred = x0_pred.clamp(-1.0, 1.0);
            }

            // Direction pointing to x_t
            let dir_xt = sqrt_one_minus_alpha_prev * noise_pred[i];

            // Previous sample
            prev_sample.push(sqrt_alpha_prev * x0_pred + dir_xt);
        }

        self.current_step += 1;
        Ok(prev_sample)
    }

    /// Returns `true` when all inference steps have been executed.
    pub fn is_done(&self) -> bool {
        self.current_step >= self.timesteps.len()
    }

    /// Number of remaining inference steps.
    pub fn steps_remaining(&self) -> usize {
        self.timesteps.len().saturating_sub(self.current_step)
    }

    /// Reset the scheduler to the initial state so it can be run again.
    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ---------------------------------------------------------------------------
// Quality / comparison helpers
// ---------------------------------------------------------------------------

/// Summary of the speed/quality trade-off for a particular inference step
/// count.
#[derive(Debug, Clone)]
pub struct StepCountComparison {
    /// Name of the pattern used.
    pub pattern: String,
    /// Number of inference steps.
    pub inference_steps: usize,
    /// Theoretical speedup = total_steps / inference_steps.
    pub theoretical_speedup: f32,
    /// Fraction of memory needed (always 1.0 — same model, different loop
    /// count).
    pub memory_reduction: f32,
    /// Recommended use-case string: "preview", "fast_iteration", or
    /// "quality".
    pub recommended_for: String,
}

/// Generate a [`StepCountComparison`] for each entry in `step_options`.
///
/// Pattern label is chosen based on step count matching the preset factories:
/// 4 → "Trailing", 10 → "Uniform", 20 → "Quadratic", others → "Uniform".
pub fn compare_step_counts(total: usize, step_options: &[usize]) -> Vec<StepCountComparison> {
    step_options
        .iter()
        .map(|&steps| {
            let theoretical_speedup = total as f32 / steps as f32;
            let recommended_for = if steps <= 4 {
                "preview".to_string()
            } else if steps <= 10 {
                "fast_iteration".to_string()
            } else {
                "quality".to_string()
            };
            // Assign a descriptive pattern label for the default presets.
            let pattern = match steps {
                4 => "Trailing".to_string(),
                10 => "Uniform".to_string(),
                20 => "Quadratic".to_string(),
                _ => "Uniform".to_string(),
            };
            StepCountComparison {
                pattern,
                inference_steps: steps,
                theoretical_speedup,
                memory_reduction: 1.0,
                recommended_for,
            }
        })
        .collect()
}

/// Recommend an inference step count that fits within the given time budget.
///
/// Chooses the largest of [4, 10, 20, 50] that satisfies
/// `steps * ms_per_step <= max_time_budget_ms`.
///
/// Falls back to 4 if no candidate fits.
pub fn recommend_inference_steps(max_time_budget_ms: f32, ms_per_step: f32) -> usize {
    let max_steps = if ms_per_step > 0.0 {
        (max_time_budget_ms / ms_per_step) as usize
    } else {
        usize::MAX
    };
    // Candidates from largest to smallest so we pick the most steps that fit.
    let candidates = [50_usize, 20, 10, 4];
    for &c in &candidates {
        if c <= max_steps {
            return c;
        }
    }
    4
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- helper: monotonically decreasing check ---
    fn is_strictly_decreasing(v: &[usize]) -> bool {
        v.windows(2).all(|w| w[0] > w[1])
    }

    // -----------------------------------------------------------------------
    // StepPattern index generation
    // -----------------------------------------------------------------------

    #[test]
    fn test_uniform_indices_count() {
        let indices = StepPattern::uniform_indices(1000, 10);
        // After dedup the count may be <= 10; for a typical total/steps ratio
        // there should be no duplicates.
        assert_eq!(indices.len(), 10);
    }

    #[test]
    fn test_uniform_indices_decreasing() {
        let indices = StepPattern::uniform_indices(1000, 20);
        assert!(!indices.is_empty());
        assert!(is_strictly_decreasing(&indices));
    }

    #[test]
    fn test_quadratic_indices_count() {
        let indices = StepPattern::quadratic_indices(1000, 20);
        // Quadratic spread over 1000 steps should not produce many duplicates.
        assert!(
            indices.len() >= 10,
            "expected ≥ 10 unique indices, got {}",
            indices.len()
        );
        // All must be in range
        assert!(indices.iter().all(|&x| x < 1000));
    }

    #[test]
    fn test_exponential_indices_count() {
        let indices = StepPattern::exponential_indices(1000, 20);
        assert!(!indices.is_empty());
        // All must be in range
        assert!(indices.iter().all(|&x| x < 1000));
    }

    #[test]
    fn test_trailing_indices() {
        let indices = StepPattern::Trailing.indices(1000, 4);
        // Trailing should give the last 4 indices: 999, 998, 997, 996
        assert_eq!(indices.len(), 4);
        assert_eq!(indices[0], 999);
        assert_eq!(indices[3], 996);
        assert!(is_strictly_decreasing(&indices));
    }

    #[test]
    fn test_leading_indices() {
        let indices = StepPattern::Leading.indices(1000, 4);
        // Leading should give the first 4 indices descending: 3, 2, 1, 0
        assert_eq!(indices.len(), 4);
        assert_eq!(indices[0], 3);
        assert_eq!(indices[3], 0);
        assert!(is_strictly_decreasing(&indices));
    }

    #[test]
    fn test_custom_pattern() {
        let custom = StepPattern::Custom(vec![500, 200, 800, 100]);
        let indices = custom.indices(1000, 4); // num_steps ignored for Custom
                                               // Should be sorted descending and all < 1000
        assert!(is_strictly_decreasing(&indices));
        assert_eq!(indices[0], 800);
        assert_eq!(indices[3], 100);
    }

    // -----------------------------------------------------------------------
    // StepScheduleConfig presets
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_fast_4step() {
        let cfg = StepScheduleConfig::fast_4step();
        assert_eq!(cfg.inference_steps, 4);
        assert_eq!(cfg.pattern, StepPattern::Trailing);
        assert_eq!(cfg.total_training_steps, 1000);
    }

    #[test]
    fn test_config_fast_10step() {
        let cfg = StepScheduleConfig::fast_10step();
        assert_eq!(cfg.inference_steps, 10);
        assert_eq!(cfg.pattern, StepPattern::Uniform);
    }

    #[test]
    fn test_config_fast_20step() {
        let cfg = StepScheduleConfig::fast_20step();
        assert_eq!(cfg.inference_steps, 20);
        assert_eq!(cfg.pattern, StepPattern::Quadratic);
    }

    #[test]
    fn test_config_validate_zero_steps_error() {
        let cfg = StepScheduleConfig::new(0, StepPattern::Uniform);
        let result = cfg.validate();
        assert!(result.is_err(), "Expected error for zero inference steps");
    }

    #[test]
    fn test_config_validate_steps_exceeds_total() {
        let cfg = StepScheduleConfig {
            total_training_steps: 50,
            inference_steps: 100,
            pattern: StepPattern::Uniform,
            clip_denoised: true,
            set_alpha_to_one: false,
        };
        let result = cfg.validate();
        assert!(result.is_err(), "Expected error when steps > total");
    }

    #[test]
    fn test_timestep_indices_from_config() {
        let cfg = StepScheduleConfig::fast_4step();
        let indices = cfg.timestep_indices();
        assert_eq!(indices.len(), 4);
        // All < 1000
        assert!(indices.iter().all(|&x| x < 1000));
    }

    // -----------------------------------------------------------------------
    // DdimStepScheduler
    // -----------------------------------------------------------------------

    /// Build a simple linearly decreasing alphas_cumprod for testing.
    fn make_alphas(n: usize) -> Vec<f32> {
        // alpha[t] = 1.0 - t / n  (decreasing from ~1 to ~0)
        (0..n).map(|t| 1.0 - t as f32 / n as f32).collect()
    }

    #[test]
    fn test_scheduler_new() {
        let alphas = make_alphas(1000);
        let cfg = StepScheduleConfig::fast_10step();
        let sched = DdimStepScheduler::new(cfg, &alphas);
        assert!(sched.is_ok(), "Expected Ok, got {:?}", sched.err());
        let sched = sched.expect("already checked");
        assert_eq!(sched.timesteps.len(), 10);
        assert_eq!(sched.alphas_cumprod.len(), 10);
    }

    #[test]
    fn test_scheduler_step() {
        let alphas = make_alphas(1000);
        let cfg = StepScheduleConfig::fast_4step();
        let mut sched = DdimStepScheduler::new(cfg, &alphas).expect("valid config");

        let n = 16_usize;
        let sample: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let noise_pred: Vec<f32> = vec![0.05_f32; n];

        let result = sched.step_with_noise_pred(&noise_pred, &sample);
        assert!(result.is_ok(), "Expected Ok, got {:?}", result.err());
        let prev = result.expect("already checked");
        assert_eq!(prev.len(), n);
        // Step counter advanced
        assert_eq!(sched.steps_remaining(), 3);
    }

    #[test]
    fn test_scheduler_is_done_after_all_steps() {
        let alphas = make_alphas(1000);
        let cfg = StepScheduleConfig::fast_4step();
        let mut sched = DdimStepScheduler::new(cfg, &alphas).expect("valid config");

        let n = 4_usize;
        let sample = vec![0.0_f32; n];
        let noise = vec![0.0_f32; n];

        for _ in 0..4 {
            assert!(!sched.is_done());
            sched
                .step_with_noise_pred(&noise, &sample)
                .expect("step ok");
        }
        assert!(sched.is_done());
        // Additional step should error
        let extra = sched.step_with_noise_pred(&noise, &sample);
        assert!(extra.is_err());
    }

    #[test]
    fn test_scheduler_reset() {
        let alphas = make_alphas(1000);
        let cfg = StepScheduleConfig::fast_4step();
        let mut sched = DdimStepScheduler::new(cfg, &alphas).expect("valid config");

        let n = 4_usize;
        let sample = vec![0.0_f32; n];
        let noise = vec![0.0_f32; n];

        for _ in 0..4 {
            sched.step_with_noise_pred(&noise, &sample).expect("ok");
        }
        assert!(sched.is_done());

        sched.reset();
        assert!(!sched.is_done());
        assert_eq!(sched.steps_remaining(), 4);
        assert!(sched.current_timestep().is_some());
    }

    // -----------------------------------------------------------------------
    // Quality analysis helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_compare_step_counts() {
        let comparisons = compare_step_counts(1000, &[4, 10, 20, 50]);
        assert_eq!(comparisons.len(), 4);

        let c4 = &comparisons[0];
        assert_eq!(c4.inference_steps, 4);
        assert!((c4.theoretical_speedup - 250.0).abs() < 1e-3);
        assert_eq!(c4.memory_reduction, 1.0);
        assert_eq!(c4.recommended_for, "preview");

        let c10 = &comparisons[1];
        assert_eq!(c10.recommended_for, "fast_iteration");

        let c20 = &comparisons[2];
        assert_eq!(c20.recommended_for, "quality");

        let c50 = &comparisons[3];
        assert_eq!(c50.recommended_for, "quality");
        assert!((c50.theoretical_speedup - 20.0).abs() < 1e-3);
    }

    #[test]
    fn test_recommend_inference_steps() {
        // Enough time for 50 steps
        assert_eq!(recommend_inference_steps(5000.0, 50.0), 50);
        // Only enough for 20
        assert_eq!(recommend_inference_steps(1000.0, 50.0), 20);
        // Only enough for 10
        assert_eq!(recommend_inference_steps(500.0, 50.0), 10);
        // Only enough for 4
        assert_eq!(recommend_inference_steps(200.0, 50.0), 4);
        // Extremely tight budget — should fall back to 4
        assert_eq!(recommend_inference_steps(1.0, 50.0), 4);
    }
}
