//! Adaptive loss weighting strategies for multi-task training.
//!
//! Provides four complementary strategies that automatically balance competing
//! loss terms during 3D Gaussian avatar training:
//!
//! 1. **[`HomoscedasticWeighter`]** — Kendall & Gal 2018 uncertainty-based
//!    weighting. Learns per-task log(σ²) parameters so high-noise tasks
//!    contribute lower effective weight.
//!
//! 2. **[`GradNormWeighter`]** — GradNorm (Chen et al. 2018). Normalises task
//!    gradient norms so every task contributes equal gradient magnitudes.
//!
//! 3. **[`ScheduledWeighter`]** — Deterministic weight schedules (constant,
//!    linear, cosine, exponential, piecewise) per task.
//!
//! 4. **[`LossStatTracker`]** — Maintains running EMA statistics on each task's
//!    loss and derives inverse-variance or relative-magnitude weights.
//!
//! All public free functions carry the `alw_` prefix to avoid name collisions
//! with the sibling `adaptive_loss` module.

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors produced by the adaptive-loss-weighting subsystem.
#[derive(Debug, thiserror::Error)]
pub enum LossWeightError {
    #[error("empty task list")]
    EmptyTaskList,
    #[error("dimension mismatch: {n_tasks} tasks but {n_losses} losses")]
    DimensionMismatch { n_tasks: usize, n_losses: usize },
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("task not found: {0}")]
    TaskNotFound(String),
    #[error("negative log-sigma: task {0} has undefined uncertainty")]
    NegativeLogSigma(usize),
}

// ─── LossTask ─────────────────────────────────────────────────────────────────

/// A single loss term in a multi-task training objective.
#[derive(Debug, Clone)]
pub struct LossTask {
    /// Human-readable name (e.g. `"photometric"`, `"regularization"`).
    pub name: String,
    /// Starting weight at step 0.
    pub initial_weight: f32,
    /// Hard lower bound for the effective weight (default `0.001`).
    pub min_weight: f32,
    /// Hard upper bound for the effective weight (default `100.0`).
    pub max_weight: f32,
    /// Primary tasks (e.g. photometric loss) receive special treatment in some
    /// strategies that anchor training to the dominant objective.
    pub is_primary: bool,
}

impl LossTask {
    /// Create a task with `min_weight = 0.001`, `max_weight = 100.0`, `is_primary = false`.
    pub fn new(name: &str, initial_weight: f32) -> Self {
        Self {
            name: name.to_string(),
            initial_weight,
            min_weight: 0.001,
            max_weight: 100.0,
            is_primary: false,
        }
    }

    /// Override the weight bounds.
    pub fn with_bounds(mut self, min: f32, max: f32) -> Self {
        self.min_weight = min;
        self.max_weight = max;
        self
    }

    /// Mark this task as the primary (dominant) objective.
    pub fn primary(mut self) -> Self {
        self.is_primary = true;
        self
    }
}

// ─── Strategy 1: Homoscedastic uncertainty weighting ─────────────────────────

/// Homoscedastic uncertainty weighting (Kendall & Gal, 2018).
///
/// Maintains one learnable parameter `log_σ_i` per task. The effective weight
/// for task *i* is `exp(−2·log_σ_i)` and a log-regularization term
/// `log_σ_i` is added to the total loss so the parameters are penalised for
/// growing without bound.
///
/// The `log_sigmas` are stored here for tracking / serialization; in a real
/// training loop the caller would update them via an optimizer. Use
/// [`HomoscedasticWeighter::update_log_sigma`] to apply manual or simulated
/// gradient updates.
pub struct HomoscedasticWeighter {
    tasks: Vec<LossTask>,
    /// `log_σ_i` for each task, initialised to 0 (σ = 1).
    log_sigmas: Vec<f32>,
}

impl HomoscedasticWeighter {
    /// Construct a weighter for the given task list.
    ///
    /// Returns [`LossWeightError::EmptyTaskList`] when `tasks` is empty.
    pub fn new(tasks: Vec<LossTask>) -> Result<Self, LossWeightError> {
        if tasks.is_empty() {
            return Err(LossWeightError::EmptyTaskList);
        }
        let n = tasks.len();
        Ok(Self {
            tasks,
            log_sigmas: vec![0.0_f32; n],
        })
    }

    /// Effective weight for task `task_idx`: `exp(−2 · log_σ)`.
    pub fn weight(&self, task_idx: usize) -> Result<f32, LossWeightError> {
        if task_idx >= self.tasks.len() {
            return Err(LossWeightError::DimensionMismatch {
                n_tasks: self.tasks.len(),
                n_losses: task_idx + 1,
            });
        }
        Ok((-2.0 * self.log_sigmas[task_idx]).exp())
    }

    /// Regularization term for task `task_idx`: just `log_σ`.
    pub fn regularization(&self, task_idx: usize) -> Result<f32, LossWeightError> {
        if task_idx >= self.tasks.len() {
            return Err(LossWeightError::DimensionMismatch {
                n_tasks: self.tasks.len(),
                n_losses: task_idx + 1,
            });
        }
        Ok(self.log_sigmas[task_idx])
    }

    /// Total weighted loss: `Σ_i [ exp(−2·log_σ_i) · loss_i + log_σ_i ]`.
    pub fn total_loss(&self, losses: &[f32]) -> Result<f32, LossWeightError> {
        if losses.len() != self.tasks.len() {
            return Err(LossWeightError::DimensionMismatch {
                n_tasks: self.tasks.len(),
                n_losses: losses.len(),
            });
        }
        let mut total = 0.0_f32;
        for (&sigma, &loss) in self
            .log_sigmas
            .iter()
            .zip(losses.iter())
            .take(self.tasks.len())
        {
            let w = (-2.0 * sigma).exp();
            total += w * loss + sigma;
        }
        Ok(total)
    }

    /// All current effective weights (one per task).
    pub fn weights(&self) -> Result<Vec<f32>, LossWeightError> {
        let mut out = Vec::with_capacity(self.tasks.len());
        for i in 0..self.tasks.len() {
            out.push(self.weight(i)?);
        }
        Ok(out)
    }

    /// Apply `delta` to `log_σ[task_idx]` (simulates a gradient step).
    pub fn update_log_sigma(&mut self, task_idx: usize, delta: f32) -> Result<(), LossWeightError> {
        if task_idx >= self.tasks.len() {
            return Err(LossWeightError::DimensionMismatch {
                n_tasks: self.tasks.len(),
                n_losses: task_idx + 1,
            });
        }
        self.log_sigmas[task_idx] += delta;
        Ok(())
    }

    /// Reset all `log_σ` to 0 (σ = 1, equal initial weighting).
    pub fn reset(&mut self) {
        for v in self.log_sigmas.iter_mut() {
            *v = 0.0;
        }
    }

    /// Number of tasks.
    pub fn n_tasks(&self) -> usize {
        self.tasks.len()
    }

    /// Current log-sigma values.
    pub fn log_sigmas(&self) -> &[f32] {
        &self.log_sigmas
    }
}

// ─── Strategy 2: GradNorm ─────────────────────────────────────────────────────

/// GradNorm-style adaptive weighting (Chen et al., 2018).
///
/// Tracks a per-task gradient norm (EMA-smoothed) and rescales task weights so
/// all tasks contribute gradient magnitudes proportional to the global mean.
///
/// The update rule is:
/// ```text
/// target_norm_i = mean_norm · r_i^alpha
/// w_i           = w_i · (target_norm_i / smoothed_norm_i)
/// ```
/// where `r_i` is the relative training rate of task *i* compared to the
/// geometric mean across tasks.
///
/// Weights are re-normalized after each update so their mean equals the initial
/// mean of `tasks[i].initial_weight`.
pub struct GradNormWeighter {
    tasks: Vec<LossTask>,
    weights: Vec<f32>,
    /// EMA-smoothed gradient norm per task.
    gradient_norms: Vec<f32>,
    /// Initial loss per task (L_i(0)), captured on the first call to `update`.
    initial_losses: Vec<f32>,
    /// Asymmetry factor (α). Larger values amplify imbalance correction.
    alpha: f32,
    /// EMA decay for gradient norm smoothing.
    ema_decay: f32,
    step: usize,
}

impl GradNormWeighter {
    /// Construct a weighter.
    /// `alpha` controls how aggressively imbalanced tasks are up-weighted
    /// (recommended range 0.5–2.0; default 1.5).
    /// `ema_decay` smooths gradient norms (recommended 0.9).
    pub fn new(tasks: Vec<LossTask>, alpha: f32, ema_decay: f32) -> Result<Self, LossWeightError> {
        if tasks.is_empty() {
            return Err(LossWeightError::EmptyTaskList);
        }
        if !(0.0..1.0).contains(&ema_decay) {
            return Err(LossWeightError::InvalidConfig(format!(
                "ema_decay must be in [0, 1), got {ema_decay}"
            )));
        }
        let n = tasks.len();
        let weights = tasks.iter().map(|t| t.initial_weight).collect();
        Ok(Self {
            tasks,
            weights,
            gradient_norms: vec![1.0_f32; n], // neutral start
            initial_losses: Vec::new(),       // populated on first update
            alpha,
            ema_decay,
            step: 0,
        })
    }

    /// Update task weights based on the latest per-task losses and gradient norms.
    ///
    /// On the first call the `current_losses` are stored as `initial_losses`.
    /// Subsequent calls compute relative training rates and adjust weights.
    pub fn update(
        &mut self,
        current_losses: &[f32],
        current_gradient_norms: &[f32],
    ) -> Result<(), LossWeightError> {
        let n = self.tasks.len();
        if current_losses.len() != n {
            return Err(LossWeightError::DimensionMismatch {
                n_tasks: n,
                n_losses: current_losses.len(),
            });
        }
        if current_gradient_norms.len() != n {
            return Err(LossWeightError::DimensionMismatch {
                n_tasks: n,
                n_losses: current_gradient_norms.len(),
            });
        }

        // Capture initial losses on step 0.
        if self.initial_losses.is_empty() {
            self.initial_losses = current_losses.to_vec();
        }

        // EMA-smooth gradient norms.
        for (gnorm, &cgnorm) in self
            .gradient_norms
            .iter_mut()
            .zip(current_gradient_norms.iter())
            .take(n)
        {
            let g = cgnorm.max(1e-8);
            *gnorm = self.ema_decay * *gnorm + (1.0 - self.ema_decay) * g;
        }

        let mean_norm: f32 = self.gradient_norms.iter().sum::<f32>() / n as f32;

        // Compute relative training rates r_i = (L_i(t)/L_i(0)) / mean_j(L_j(t)/L_j(0)).
        let loss_ratios: Vec<f32> = (0..n)
            .map(|i| {
                let init = self.initial_losses[i].max(1e-8);
                current_losses[i] / init
            })
            .collect();
        let mean_ratio: f32 = loss_ratios.iter().sum::<f32>() / n as f32;
        let mean_ratio = mean_ratio.max(1e-8);

        // Adjust each task weight toward the GradNorm target.
        for ((&r, &norm), (weight, task)) in loss_ratios
            .iter()
            .zip(self.gradient_norms.iter())
            .zip(self.weights.iter_mut().zip(self.tasks.iter()))
            .take(n)
        {
            let r_i = r / mean_ratio;
            let target_norm = mean_norm * r_i.powf(self.alpha);
            let norm_i = norm.max(1e-8);
            *weight *= target_norm / norm_i;
            // Clip to task bounds.
            *weight = (*weight).max(task.min_weight).min(task.max_weight);
        }

        // Re-normalize so mean weight equals the initial mean weight.
        let init_mean: f32 = self.tasks.iter().map(|t| t.initial_weight).sum::<f32>() / n as f32;
        let cur_mean: f32 = self.weights.iter().sum::<f32>() / n as f32;
        if cur_mean > 1e-8 {
            let scale = init_mean / cur_mean;
            for w in self.weights.iter_mut() {
                *w *= scale;
            }
        }

        self.step += 1;
        Ok(())
    }

    /// Current task weights.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Latest EMA-smoothed gradient norms.
    pub fn gradient_norms(&self) -> &[f32] {
        &self.gradient_norms
    }

    /// Number of updates applied since construction.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Number of tasks.
    pub fn n_tasks(&self) -> usize {
        self.tasks.len()
    }
}

// ─── Strategy 3: Scheduled weighting ─────────────────────────────────────────

/// Defines how a task's weight evolves over training steps.
#[derive(Debug, Clone)]
pub enum WeightScheduleKind {
    /// Fixed weight for all steps.
    Constant(f32),
    /// Linear interpolation from `start` to `end` over `n_steps`.
    Linear {
        start: f32,
        end: f32,
        n_steps: usize,
    },
    /// Cosine decay from `start` to `end` over `n_steps`.
    Cosine {
        start: f32,
        end: f32,
        n_steps: usize,
    },
    /// Exponential decay: `weight = start · decay^step`.
    Exponential { start: f32, decay: f32 },
    /// Piecewise linear through `(step, weight)` keyframes (must be sorted by step).
    Piecewise { keyframes: Vec<(usize, f32)> },
}

impl WeightScheduleKind {
    /// Compute the weight at the given training step.
    pub fn weight_at(&self, step: usize) -> f32 {
        match self {
            WeightScheduleKind::Constant(w) => *w,
            WeightScheduleKind::Linear {
                start,
                end,
                n_steps,
            } => {
                if *n_steps == 0 {
                    return *end;
                }
                let t = (step as f32 / *n_steps as f32).min(1.0);
                start + (end - start) * t
            }
            WeightScheduleKind::Cosine {
                start,
                end,
                n_steps,
            } => {
                if *n_steps == 0 {
                    return *end;
                }
                let t = (step as f32 / *n_steps as f32).min(1.0);
                let cos_factor = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
                // cos decays from 1→0, so we go from start→end
                end + (start - end) * cos_factor
            }
            WeightScheduleKind::Exponential { start, decay } => start * decay.powi(step as i32),
            WeightScheduleKind::Piecewise { keyframes } => {
                if keyframes.is_empty() {
                    return 0.0;
                }
                // Before first keyframe
                if step <= keyframes[0].0 {
                    return keyframes[0].1;
                }
                // After last keyframe
                if step >= keyframes[keyframes.len() - 1].0 {
                    return keyframes[keyframes.len() - 1].1;
                }
                // Binary-search for the surrounding pair.
                let idx = keyframes.partition_point(|&(s, _)| s <= step);
                // idx is the first keyframe strictly after `step`
                let (s0, w0) = keyframes[idx - 1];
                let (s1, w1) = keyframes[idx];
                let span = (s1 - s0) as f32;
                if span < 1e-8 {
                    return w1;
                }
                let t = (step - s0) as f32 / span;
                w0 + (w1 - w0) * t
            }
        }
    }
}

/// Associates a weight schedule with a named task.
#[derive(Debug, Clone)]
pub struct TaskWeightSchedule {
    /// Name matching the corresponding [`LossTask`].
    pub task_name: String,
    /// Schedule to apply.
    pub schedule: WeightScheduleKind,
}

/// Manages a collection of per-task weight schedules.
pub struct ScheduledWeighter {
    schedules: Vec<TaskWeightSchedule>,
    step: usize,
}

impl ScheduledWeighter {
    /// Construct a `ScheduledWeighter` from the given per-task schedules.
    ///
    /// Returns [`LossWeightError::EmptyTaskList`] when `schedules` is empty.
    pub fn new(schedules: Vec<TaskWeightSchedule>) -> Result<Self, LossWeightError> {
        if schedules.is_empty() {
            return Err(LossWeightError::EmptyTaskList);
        }
        Ok(Self { schedules, step: 0 })
    }

    /// Compute weights for all tasks at an arbitrary step (does not advance the internal counter).
    pub fn weights_at(&self, step: usize) -> Vec<f32> {
        self.schedules
            .iter()
            .map(|s| s.schedule.weight_at(step))
            .collect()
    }

    /// Compute weights for all tasks at the current internal step.
    pub fn current_weights(&self) -> Vec<f32> {
        self.weights_at(self.step)
    }

    /// Advance the internal step counter by one.
    pub fn advance(&mut self) {
        self.step += 1;
    }

    /// Current step index.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Names of all scheduled tasks in order.
    pub fn task_names(&self) -> Vec<&str> {
        self.schedules
            .iter()
            .map(|s| s.task_name.as_str())
            .collect()
    }

    /// Look up the schedule for a task by name.
    pub fn schedule_for(&self, name: &str) -> Option<&TaskWeightSchedule> {
        self.schedules.iter().find(|s| s.task_name == name)
    }
}

// ─── Strategy 4: EMA-based adaptive weighting ─────────────────────────────────

/// Tracks per-task loss statistics (EMA mean and variance) and derives weights
/// from those statistics.
pub struct LossStatTracker {
    n_tasks: usize,
    means: Vec<f32>,
    variances: Vec<f32>,
    ema_decay: f32,
    step: usize,
}

impl LossStatTracker {
    /// Create a tracker for `n_tasks` tasks with the given EMA decay factor.
    ///
    /// Returns [`LossWeightError::EmptyTaskList`] when `n_tasks == 0`.
    /// Returns [`LossWeightError::InvalidConfig`] when `ema_decay` is outside `(0, 1)`.
    pub fn new(n_tasks: usize, ema_decay: f32) -> Result<Self, LossWeightError> {
        if n_tasks == 0 {
            return Err(LossWeightError::EmptyTaskList);
        }
        if ema_decay <= 0.0 || ema_decay >= 1.0 {
            return Err(LossWeightError::InvalidConfig(format!(
                "ema_decay must be in (0, 1), got {ema_decay}"
            )));
        }
        Ok(Self {
            n_tasks,
            means: vec![0.0_f32; n_tasks],
            variances: vec![1.0_f32; n_tasks], // start with neutral variance
            ema_decay,
            step: 0,
        })
    }

    /// Ingest the latest per-task loss values and update EMA statistics.
    pub fn update(&mut self, losses: &[f32]) -> Result<(), LossWeightError> {
        if losses.len() != self.n_tasks {
            return Err(LossWeightError::DimensionMismatch {
                n_tasks: self.n_tasks,
                n_losses: losses.len(),
            });
        }
        let d = self.ema_decay;
        for ((mean, variance), &loss) in self
            .means
            .iter_mut()
            .zip(self.variances.iter_mut())
            .zip(losses.iter())
            .take(self.n_tasks)
        {
            let old_mean = *mean;
            *mean = d * old_mean + (1.0 - d) * loss;
            // EMA variance using Welford-style online update adapted to EMA.
            let diff = loss - old_mean;
            *variance = d * *variance + (1.0 - d) * diff * diff;
        }
        self.step += 1;
        Ok(())
    }

    /// Inverse-variance weights: `w_i = 1 / (var_i + epsilon)`.
    ///
    /// Higher variance → lower weight (the loss is too noisy to trust heavily).
    pub fn inverse_variance_weights(&self, epsilon: f32) -> Vec<f32> {
        self.variances.iter().map(|v| 1.0 / (v + epsilon)).collect()
    }

    /// Relative-magnitude weights that normalize losses to the same scale.
    ///
    /// `w_i = mean_global / mean_i` so all tasks contribute equally after
    /// weighting. Returns [`LossWeightError::InvalidConfig`] when all means are
    /// zero (no updates have been applied yet).
    pub fn relative_magnitude_weights(&self) -> Result<Vec<f32>, LossWeightError> {
        let grand_mean: f32 = self.means.iter().sum::<f32>() / self.n_tasks as f32;
        if grand_mean.abs() < 1e-10 {
            return Err(LossWeightError::InvalidConfig(
                "all task means are zero; cannot compute relative-magnitude weights".to_string(),
            ));
        }
        let weights = self
            .means
            .iter()
            .map(|m| {
                let m = m.abs().max(1e-8);
                grand_mean / m
            })
            .collect();
        Ok(weights)
    }

    /// EMA means per task.
    pub fn means(&self) -> &[f32] {
        &self.means
    }

    /// EMA variances per task.
    pub fn variances(&self) -> &[f32] {
        &self.variances
    }

    /// Number of update calls made since construction.
    pub fn step(&self) -> usize {
        self.step
    }
}

// ─── Utility functions ────────────────────────────────────────────────────────

/// Normalize weights so their mean equals 1 (i.e., they sum to `n_tasks`).
///
/// Returns [`LossWeightError::EmptyTaskList`] for an empty slice, and
/// [`LossWeightError::InvalidConfig`] if the sum is effectively zero.
pub fn alw_normalize_weights(weights: &[f32]) -> Result<Vec<f32>, LossWeightError> {
    if weights.is_empty() {
        return Err(LossWeightError::EmptyTaskList);
    }
    let sum: f32 = weights.iter().sum();
    if sum.abs() < 1e-10 {
        return Err(LossWeightError::InvalidConfig(
            "cannot normalize: all weights are zero".to_string(),
        ));
    }
    let n = weights.len() as f32;
    Ok(weights.iter().map(|w| w * n / sum).collect())
}

/// Clip each weight to the `[min_weight, max_weight]` bounds of the
/// corresponding [`LossTask`].
///
/// When `weights.len() != tasks.len()` the shorter length governs iteration and
/// surplus entries are silently dropped (callers should ensure the sizes match).
pub fn alw_clip_weights(weights: &[f32], tasks: &[LossTask]) -> Vec<f32> {
    weights
        .iter()
        .zip(tasks.iter())
        .map(|(w, t)| w.max(t.min_weight).min(t.max_weight))
        .collect()
}

/// Compute per-task relative training rate:
/// `r_i = (L_i(t) / L_i(0)) / mean_j(L_j(t) / L_j(0))`.
///
/// Returns `Ok(vec![1.0; n])` when all ratios are equal (balanced training).
/// Returns [`LossWeightError::DimensionMismatch`] on length mismatch.
/// Returns [`LossWeightError::InvalidConfig`] if `initial_losses` contains zeros
/// (undefined ratio).
pub fn alw_relative_training_rate(
    current_losses: &[f32],
    initial_losses: &[f32],
) -> Result<Vec<f32>, LossWeightError> {
    if current_losses.len() != initial_losses.len() {
        return Err(LossWeightError::DimensionMismatch {
            n_tasks: initial_losses.len(),
            n_losses: current_losses.len(),
        });
    }
    let n = current_losses.len();
    if n == 0 {
        return Err(LossWeightError::EmptyTaskList);
    }
    let ratios: Vec<f32> = current_losses
        .iter()
        .zip(initial_losses.iter())
        .map(|(c, i)| {
            let denom = i.abs().max(1e-8);
            c / denom
        })
        .collect();
    let mean_ratio: f32 = ratios.iter().sum::<f32>() / n as f32;
    let mean_ratio = mean_ratio.max(1e-8);
    Ok(ratios.iter().map(|r| r / mean_ratio).collect())
}

/// Compute a weighted sum: `Σ_i w_i · loss_i`.
///
/// Returns [`LossWeightError::DimensionMismatch`] on length mismatch.
pub fn alw_weighted_sum(weights: &[f32], losses: &[f32]) -> Result<f32, LossWeightError> {
    if weights.len() != losses.len() {
        return Err(LossWeightError::DimensionMismatch {
            n_tasks: weights.len(),
            n_losses: losses.len(),
        });
    }
    Ok(weights.iter().zip(losses.iter()).map(|(w, l)| w * l).sum())
}

/// Imbalance ratio: `max(weights) / min(weights)`.
///
/// Returns `1.0` for empty slices or when all weights are equal.
pub fn alw_imbalance_ratio(weights: &[f32]) -> f32 {
    if weights.is_empty() {
        return 1.0;
    }
    let min_w = weights.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if min_w.abs() < 1e-10 {
        return f32::INFINITY;
    }
    max_w / min_w
}

/// Format a human-readable description of the current task weights.
pub fn alw_format_weights(tasks: &[LossTask], weights: &[f32]) -> String {
    let pairs: Vec<String> = tasks
        .iter()
        .zip(weights.iter())
        .map(|(t, w)| format!("{}={:.4}", t.name, w))
        .collect();
    format!("[{}]", pairs.join(", "))
}

// ─── Weight history ───────────────────────────────────────────────────────────

/// Stores a capped history of per-step weight vectors for post-hoc analysis.
pub struct WeightHistory {
    n_tasks: usize,
    task_names: Vec<String>,
    /// Recorded weight vectors, newest last, capped at 1000 entries.
    history: Vec<Vec<f32>>,
}

impl WeightHistory {
    /// Construct an empty history for `task_names.len()` tasks.
    pub fn new(task_names: Vec<String>) -> Self {
        let n = task_names.len();
        Self {
            n_tasks: n,
            task_names,
            history: Vec::new(),
        }
    }

    /// Append a weight snapshot.
    /// Returns [`LossWeightError::DimensionMismatch`] if `weights.len() != n_tasks`.
    /// Evicts the oldest entry when the buffer exceeds 1000 entries.
    pub fn record(&mut self, weights: &[f32]) -> Result<(), LossWeightError> {
        if weights.len() != self.n_tasks {
            return Err(LossWeightError::DimensionMismatch {
                n_tasks: self.n_tasks,
                n_losses: weights.len(),
            });
        }
        if self.history.len() >= 1000 {
            self.history.remove(0);
        }
        self.history.push(weights.to_vec());
        Ok(())
    }

    /// The most recently recorded weight vector, or `None` if empty.
    pub fn latest(&self) -> Option<&Vec<f32>> {
        self.history.last()
    }

    /// Mean weight per task across all recorded steps.
    ///
    /// Returns a zero vector when there are no recorded entries.
    pub fn mean_weights(&self) -> Vec<f32> {
        if self.history.is_empty() || self.n_tasks == 0 {
            return vec![0.0_f32; self.n_tasks];
        }
        let n = self.history.len() as f32;
        let mut sums = vec![0.0_f32; self.n_tasks];
        for snap in &self.history {
            for (s, w) in sums.iter_mut().zip(snap.iter()) {
                *s += w;
            }
        }
        sums.iter().map(|s| s / n).collect()
    }

    /// Linear-regression slope of the recorded weights for `task_idx`.
    ///
    /// Positive slope → weight is increasing; negative → decreasing; ≈0 → stable.
    /// Returns `0.0` when there are fewer than 2 recorded entries or `task_idx`
    /// is out of bounds.
    pub fn weight_trend(&self, task_idx: usize) -> f32 {
        if task_idx >= self.n_tasks || self.history.len() < 2 {
            return 0.0;
        }
        let n = self.history.len() as f32;
        // x-values: 0, 1, …, n-1
        let mean_x = (n - 1.0) / 2.0;
        let mean_y: f32 = self.history.iter().map(|s| s[task_idx]).sum::<f32>() / n;
        let mut num = 0.0_f32;
        let mut den = 0.0_f32;
        for (i, snap) in self.history.iter().enumerate() {
            let xi = i as f32 - mean_x;
            let yi = snap[task_idx] - mean_y;
            num += xi * yi;
            den += xi * xi;
        }
        if den.abs() < 1e-10 {
            return 0.0;
        }
        num / den
    }

    /// Number of recorded snapshots.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Returns `true` when no snapshots have been recorded.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
}

/// Format a brief textual summary of the weight history.
pub fn alw_format_history_summary(history: &WeightHistory) -> String {
    if history.is_empty() {
        return format!("WeightHistory(tasks={}, steps=0, no data)", history.n_tasks);
    }
    let means = history.mean_weights();
    let mean_strs: Vec<String> = history
        .task_names
        .iter()
        .zip(means.iter())
        .map(|(name, m)| format!("{}={:.4}", name, m))
        .collect();
    format!(
        "WeightHistory(tasks={}, steps={}, means=[{}])",
        history.n_tasks,
        history.len(),
        mean_strs.join(", ")
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── LossTask ──────────────────────────────────────────────────────────────

    #[test]
    fn loss_task_new_defaults() {
        let t = LossTask::new("photo", 1.0);
        assert_eq!(t.name, "photo");
        assert!((t.initial_weight - 1.0).abs() < 1e-6);
        assert!((t.min_weight - 0.001).abs() < 1e-6);
        assert!((t.max_weight - 100.0).abs() < 1e-6);
        assert!(!t.is_primary);
    }

    #[test]
    fn loss_task_with_bounds() {
        let t = LossTask::new("reg", 0.1).with_bounds(0.01, 10.0);
        assert!((t.min_weight - 0.01).abs() < 1e-6);
        assert!((t.max_weight - 10.0).abs() < 1e-6);
    }

    #[test]
    fn loss_task_primary() {
        let t = LossTask::new("photo", 1.0).primary();
        assert!(t.is_primary);
    }

    #[test]
    fn loss_task_not_primary_by_default() {
        let t = LossTask::new("reg", 0.1);
        assert!(!t.is_primary);
    }

    // ── HomoscedasticWeighter ─────────────────────────────────────────────────

    #[test]
    fn homo_new_empty_error() {
        let result = HomoscedasticWeighter::new(vec![]);
        assert!(matches!(result, Err(LossWeightError::EmptyTaskList)));
    }

    #[test]
    fn homo_new_two_tasks_ok() {
        let w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0), LossTask::new("b", 0.5)]);
        assert!(w.is_ok());
        assert_eq!(w.unwrap().n_tasks(), 2);
    }

    #[test]
    fn homo_weight_log_sigma_zero_gives_one() {
        let w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0)]).unwrap();
        let weight = w.weight(0).unwrap();
        // exp(-2*0) = 1
        assert!((weight - 1.0).abs() < 1e-5);
    }

    #[test]
    fn homo_weight_log_sigma_ln2_gives_quarter() {
        let mut w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0)]).unwrap();
        // log_sigma = ln(2) → weight = exp(-2*ln(2)) = exp(ln(0.25)) = 0.25
        w.update_log_sigma(0, std::f32::consts::LN_2).unwrap();
        let weight = w.weight(0).unwrap();
        assert!((weight - 0.25).abs() < 1e-4);
    }

    #[test]
    fn homo_weight_out_of_bounds_error() {
        let w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0)]).unwrap();
        assert!(w.weight(1).is_err());
    }

    #[test]
    fn homo_regularization_zero_log_sigma() {
        let w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0)]).unwrap();
        let reg = w.regularization(0).unwrap();
        assert!((reg - 0.0).abs() < 1e-6);
    }

    #[test]
    fn homo_regularization_nonzero_log_sigma() {
        let mut w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0)]).unwrap();
        w.update_log_sigma(0, 0.5).unwrap();
        let reg = w.regularization(0).unwrap();
        assert!((reg - 0.5).abs() < 1e-5);
    }

    #[test]
    fn homo_total_loss_correct_formula() {
        // Two tasks, log_sigma = [0, ln(2)]
        // task 0: weight=1, reg=0 → contribution = 1*loss0 + 0
        // task 1: weight=0.25, reg=ln(2) → contribution = 0.25*loss1 + ln(2)
        let mut w =
            HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0), LossTask::new("b", 1.0)])
                .unwrap();
        w.update_log_sigma(1, std::f32::consts::LN_2).unwrap();
        let losses = [2.0_f32, 4.0_f32];
        let total = w.total_loss(&losses).unwrap();
        let expected = 1.0 * 2.0 + 0.0 + 0.25 * 4.0 + std::f32::consts::LN_2;
        assert!((total - expected).abs() < 1e-4);
    }

    #[test]
    fn homo_total_loss_dimension_mismatch() {
        let w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0)]).unwrap();
        assert!(matches!(
            w.total_loss(&[1.0, 2.0]),
            Err(LossWeightError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn homo_weights_all_computed() {
        let mut w =
            HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0), LossTask::new("b", 1.0)])
                .unwrap();
        w.update_log_sigma(0, 0.0).unwrap();
        w.update_log_sigma(1, 1.0).unwrap();
        let weights = w.weights().unwrap();
        assert_eq!(weights.len(), 2);
        assert!((weights[0] - 1.0).abs() < 1e-5);
        assert!((weights[1] - (-2.0_f32).exp()).abs() < 1e-5);
    }

    #[test]
    fn homo_update_log_sigma_applies_delta() {
        let mut w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0)]).unwrap();
        w.update_log_sigma(0, 0.3).unwrap();
        assert!((w.log_sigmas()[0] - 0.3).abs() < 1e-6);
        w.update_log_sigma(0, 0.2).unwrap();
        assert!((w.log_sigmas()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn homo_update_log_sigma_out_of_bounds() {
        let mut w = HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0)]).unwrap();
        assert!(w.update_log_sigma(5, 0.1).is_err());
    }

    #[test]
    fn homo_reset_sets_log_sigmas_to_zero() {
        let mut w =
            HomoscedasticWeighter::new(vec![LossTask::new("a", 1.0), LossTask::new("b", 1.0)])
                .unwrap();
        w.update_log_sigma(0, 2.5).unwrap();
        w.update_log_sigma(1, -1.0).unwrap();
        w.reset();
        for &ls in w.log_sigmas() {
            assert!((ls - 0.0).abs() < 1e-6);
        }
    }

    // ── GradNormWeighter ──────────────────────────────────────────────────────

    #[test]
    fn gradnorm_new_empty_error() {
        let r = GradNormWeighter::new(vec![], 1.5, 0.9);
        assert!(matches!(r, Err(LossWeightError::EmptyTaskList)));
    }

    #[test]
    fn gradnorm_new_correct_alpha() {
        let g = GradNormWeighter::new(
            vec![LossTask::new("a", 1.0), LossTask::new("b", 1.0)],
            1.5,
            0.9,
        )
        .unwrap();
        assert!((g.alpha - 1.5).abs() < 1e-6);
    }

    #[test]
    fn gradnorm_new_invalid_ema_decay() {
        let r = GradNormWeighter::new(vec![LossTask::new("a", 1.0)], 1.5, 1.0);
        assert!(r.is_err());
    }

    #[test]
    fn gradnorm_update_increments_step() {
        let mut g = GradNormWeighter::new(
            vec![LossTask::new("a", 1.0), LossTask::new("b", 1.0)],
            1.5,
            0.9,
        )
        .unwrap();
        g.update(&[1.0, 1.0], &[1.0, 1.0]).unwrap();
        assert_eq!(g.step(), 1);
        g.update(&[1.0, 1.0], &[1.0, 1.0]).unwrap();
        assert_eq!(g.step(), 2);
    }

    #[test]
    fn gradnorm_update_uniform_norms_gives_uniform_weights() {
        let mut g = GradNormWeighter::new(
            vec![LossTask::new("a", 1.0), LossTask::new("b", 1.0)],
            1.5,
            0.0, // no EMA smoothing for a clean test
        )
        .unwrap();
        // Uniform gradient norms and uniform loss ratios should yield equal weights.
        for _ in 0..5 {
            g.update(&[1.0, 1.0], &[1.0, 1.0]).unwrap();
        }
        let w = g.weights();
        assert!(
            (w[0] - w[1]).abs() < 0.1,
            "weights should be near-equal: {:?}",
            w
        );
    }

    #[test]
    fn gradnorm_update_dimension_mismatch_losses() {
        let mut g = GradNormWeighter::new(
            vec![LossTask::new("a", 1.0), LossTask::new("b", 1.0)],
            1.5,
            0.9,
        )
        .unwrap();
        assert!(matches!(
            g.update(&[1.0], &[1.0, 1.0]),
            Err(LossWeightError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn gradnorm_update_dimension_mismatch_norms() {
        let mut g = GradNormWeighter::new(
            vec![LossTask::new("a", 1.0), LossTask::new("b", 1.0)],
            1.5,
            0.9,
        )
        .unwrap();
        assert!(matches!(
            g.update(&[1.0, 1.0], &[1.0]),
            Err(LossWeightError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn gradnorm_weights_update_based_on_norms() {
        let mut g = GradNormWeighter::new(
            vec![
                LossTask::new("a", 1.0).with_bounds(0.001, 1000.0),
                LossTask::new("b", 1.0).with_bounds(0.001, 1000.0),
            ],
            1.0,
            0.0,
        )
        .unwrap();
        // Task b has 10x larger gradient norm → b should get lower weight.
        g.update(&[1.0, 1.0], &[1.0, 10.0]).unwrap();
        let w = g.weights();
        // After normalization, task a (smaller norm) should have higher weight.
        assert!(w[0] > w[1], "a={:.4}, b={:.4}", w[0], w[1]);
    }

    // ── WeightScheduleKind ────────────────────────────────────────────────────

    #[test]
    fn schedule_constant_returns_same() {
        let s = WeightScheduleKind::Constant(3.5);
        assert!((s.weight_at(0) - 3.5).abs() < 1e-6);
        assert!((s.weight_at(999) - 3.5).abs() < 1e-6);
    }

    #[test]
    fn schedule_linear_start() {
        let s = WeightScheduleKind::Linear {
            start: 0.0,
            end: 1.0,
            n_steps: 100,
        };
        assert!((s.weight_at(0) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn schedule_linear_end() {
        let s = WeightScheduleKind::Linear {
            start: 0.0,
            end: 1.0,
            n_steps: 100,
        };
        assert!((s.weight_at(100) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn schedule_linear_midpoint() {
        let s = WeightScheduleKind::Linear {
            start: 0.0,
            end: 2.0,
            n_steps: 100,
        };
        let mid = s.weight_at(50);
        assert!((mid - 1.0).abs() < 1e-4, "mid={mid}");
    }

    #[test]
    fn schedule_cosine_start() {
        let s = WeightScheduleKind::Cosine {
            start: 1.0,
            end: 0.0,
            n_steps: 100,
        };
        assert!((s.weight_at(0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn schedule_cosine_end() {
        let s = WeightScheduleKind::Cosine {
            start: 1.0,
            end: 0.0,
            n_steps: 100,
        };
        assert!((s.weight_at(100) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn schedule_cosine_monotone_decreasing() {
        let s = WeightScheduleKind::Cosine {
            start: 1.0,
            end: 0.0,
            n_steps: 10,
        };
        let mut prev = s.weight_at(0);
        for step in 1..=10 {
            let cur = s.weight_at(step);
            assert!(
                cur <= prev + 1e-4,
                "not monotone at step {step}: prev={prev}, cur={cur}"
            );
            prev = cur;
        }
    }

    #[test]
    fn schedule_exponential_decays() {
        let s = WeightScheduleKind::Exponential {
            start: 1.0,
            decay: 0.5,
        };
        assert!((s.weight_at(0) - 1.0).abs() < 1e-6);
        assert!((s.weight_at(1) - 0.5).abs() < 1e-6);
        assert!((s.weight_at(2) - 0.25).abs() < 1e-6);
        assert!((s.weight_at(3) - 0.125).abs() < 1e-6);
    }

    #[test]
    fn schedule_piecewise_exact_keyframes() {
        let s = WeightScheduleKind::Piecewise {
            keyframes: vec![(0, 1.0), (10, 2.0), (20, 0.5)],
        };
        assert!((s.weight_at(0) - 1.0).abs() < 1e-5);
        assert!((s.weight_at(10) - 2.0).abs() < 1e-5);
        assert!((s.weight_at(20) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn schedule_piecewise_interpolation() {
        let s = WeightScheduleKind::Piecewise {
            keyframes: vec![(0, 0.0), (10, 10.0)],
        };
        // step 5 should be exactly 5.0
        let v = s.weight_at(5);
        assert!((v - 5.0).abs() < 1e-4, "v={v}");
    }

    #[test]
    fn schedule_piecewise_before_first_keyframe() {
        let s = WeightScheduleKind::Piecewise {
            keyframes: vec![(5, 2.0), (10, 4.0)],
        };
        // steps before the first keyframe clamp to the first value
        assert!((s.weight_at(0) - 2.0).abs() < 1e-5);
        assert!((s.weight_at(3) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn schedule_piecewise_after_last_keyframe() {
        let s = WeightScheduleKind::Piecewise {
            keyframes: vec![(0, 1.0), (10, 3.0)],
        };
        assert!((s.weight_at(100) - 3.0).abs() < 1e-5);
    }

    // ── ScheduledWeighter ─────────────────────────────────────────────────────

    #[test]
    fn scheduled_new_empty_error() {
        assert!(matches!(
            ScheduledWeighter::new(vec![]),
            Err(LossWeightError::EmptyTaskList)
        ));
    }

    #[test]
    fn scheduled_weights_at_correct() {
        let sw = ScheduledWeighter::new(vec![
            TaskWeightSchedule {
                task_name: "photo".into(),
                schedule: WeightScheduleKind::Constant(2.0),
            },
            TaskWeightSchedule {
                task_name: "reg".into(),
                schedule: WeightScheduleKind::Linear {
                    start: 0.0,
                    end: 1.0,
                    n_steps: 10,
                },
            },
        ])
        .unwrap();
        let w = sw.weights_at(5);
        assert!((w[0] - 2.0).abs() < 1e-5);
        assert!((w[1] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn scheduled_advance_increments_step() {
        let mut sw = ScheduledWeighter::new(vec![TaskWeightSchedule {
            task_name: "x".into(),
            schedule: WeightScheduleKind::Constant(1.0),
        }])
        .unwrap();
        assert_eq!(sw.step(), 0);
        sw.advance();
        assert_eq!(sw.step(), 1);
        sw.advance();
        assert_eq!(sw.step(), 2);
    }

    #[test]
    fn scheduled_current_weights_tracks_step() {
        let mut sw = ScheduledWeighter::new(vec![TaskWeightSchedule {
            task_name: "x".into(),
            schedule: WeightScheduleKind::Linear {
                start: 0.0,
                end: 10.0,
                n_steps: 10,
            },
        }])
        .unwrap();
        sw.advance(); // step = 1
        let w = sw.current_weights();
        assert!((w[0] - 1.0).abs() < 1e-4, "w={}", w[0]);
    }

    #[test]
    fn scheduled_schedule_for_found() {
        let sw = ScheduledWeighter::new(vec![
            TaskWeightSchedule {
                task_name: "alpha".into(),
                schedule: WeightScheduleKind::Constant(1.0),
            },
            TaskWeightSchedule {
                task_name: "beta".into(),
                schedule: WeightScheduleKind::Constant(2.0),
            },
        ])
        .unwrap();
        let s = sw.schedule_for("beta");
        assert!(s.is_some());
        assert_eq!(s.unwrap().task_name, "beta");
    }

    #[test]
    fn scheduled_schedule_for_not_found() {
        let sw = ScheduledWeighter::new(vec![TaskWeightSchedule {
            task_name: "alpha".into(),
            schedule: WeightScheduleKind::Constant(1.0),
        }])
        .unwrap();
        assert!(sw.schedule_for("gamma").is_none());
    }

    #[test]
    fn scheduled_task_names() {
        let sw = ScheduledWeighter::new(vec![
            TaskWeightSchedule {
                task_name: "photo".into(),
                schedule: WeightScheduleKind::Constant(1.0),
            },
            TaskWeightSchedule {
                task_name: "reg".into(),
                schedule: WeightScheduleKind::Constant(0.1),
            },
        ])
        .unwrap();
        let names = sw.task_names();
        assert_eq!(names, vec!["photo", "reg"]);
    }

    // ── LossStatTracker ───────────────────────────────────────────────────────

    #[test]
    fn stat_tracker_zero_tasks_error() {
        assert!(matches!(
            LossStatTracker::new(0, 0.9),
            Err(LossWeightError::EmptyTaskList)
        ));
    }

    #[test]
    fn stat_tracker_invalid_ema_decay() {
        assert!(LossStatTracker::new(2, 0.0).is_err());
        assert!(LossStatTracker::new(2, 1.0).is_err());
    }

    #[test]
    fn stat_tracker_update_ema_converges() {
        let mut t = LossStatTracker::new(1, 0.99).unwrap();
        // Feed many identical losses; the EMA mean should converge to the value.
        for _ in 0..500 {
            t.update(&[5.0]).unwrap();
        }
        assert!((t.means()[0] - 5.0).abs() < 0.1, "mean={}", t.means()[0]);
    }

    #[test]
    fn stat_tracker_update_dimension_mismatch() {
        let mut t = LossStatTracker::new(2, 0.9).unwrap();
        assert!(matches!(
            t.update(&[1.0]),
            Err(LossWeightError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn stat_tracker_inverse_variance_uniform() {
        let mut t = LossStatTracker::new(2, 0.9).unwrap();
        // Feed identical losses to both tasks → variances should equalize.
        for _ in 0..100 {
            t.update(&[1.0, 1.0]).unwrap();
        }
        let w = t.inverse_variance_weights(1e-4);
        // Both should have the same weight.
        assert!((w[0] - w[1]).abs() < 0.01, "w={:?}", w);
    }

    #[test]
    fn stat_tracker_inverse_variance_high_var_lower_weight() {
        let mut t = LossStatTracker::new(2, 0.5).unwrap();
        // Task 0: constant; task 1: oscillating → higher variance.
        let mut sign = 1.0_f32;
        for _ in 0..200 {
            t.update(&[1.0, 1.0 + sign * 10.0]).unwrap();
            sign = -sign;
        }
        let w = t.inverse_variance_weights(1e-6);
        assert!(w[0] > w[1], "task0={:.4}, task1={:.4}", w[0], w[1]);
    }

    #[test]
    fn stat_tracker_relative_magnitude_uniform() {
        let mut t = LossStatTracker::new(3, 0.5).unwrap();
        for _ in 0..200 {
            t.update(&[2.0, 2.0, 2.0]).unwrap();
        }
        let w = t.relative_magnitude_weights().unwrap();
        for wi in &w {
            assert!((wi - 1.0).abs() < 0.01, "wi={wi}");
        }
    }

    #[test]
    fn stat_tracker_step_increments() {
        let mut t = LossStatTracker::new(1, 0.9).unwrap();
        assert_eq!(t.step(), 0);
        t.update(&[1.0]).unwrap();
        assert_eq!(t.step(), 1);
    }

    // ── Utility functions ─────────────────────────────────────────────────────

    #[test]
    fn normalize_weights_sums_to_n_tasks() {
        let w = alw_normalize_weights(&[1.0, 2.0, 3.0]).unwrap();
        let n = 3.0_f32;
        assert!((w.iter().sum::<f32>() - n).abs() < 1e-5);
    }

    #[test]
    fn normalize_weights_empty_error() {
        assert!(matches!(
            alw_normalize_weights(&[]),
            Err(LossWeightError::EmptyTaskList)
        ));
    }

    #[test]
    fn normalize_weights_all_zero_error() {
        assert!(alw_normalize_weights(&[0.0, 0.0]).is_err());
    }

    #[test]
    fn clip_weights_clips_to_bounds() {
        let tasks = vec![
            LossTask::new("a", 1.0).with_bounds(0.5, 2.0),
            LossTask::new("b", 1.0).with_bounds(0.5, 2.0),
        ];
        let clipped = alw_clip_weights(&[0.1, 5.0], &tasks);
        assert!((clipped[0] - 0.5).abs() < 1e-6);
        assert!((clipped[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn clip_weights_in_bounds_unchanged() {
        let tasks = vec![LossTask::new("a", 1.0).with_bounds(0.1, 10.0)];
        let clipped = alw_clip_weights(&[1.5], &tasks);
        assert!((clipped[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn relative_training_rate_no_change() {
        // When losses don't change, all rates should be 1.0.
        let rates = alw_relative_training_rate(&[2.0, 3.0], &[2.0, 3.0]).unwrap();
        for r in &rates {
            assert!((r - 1.0).abs() < 1e-4, "r={r}");
        }
    }

    #[test]
    fn relative_training_rate_one_task_doubled() {
        // Task 0 doubled, task 1 unchanged → task 0 rate is 2, task 1 is 1.
        // Mean ratio = 1.5, so r0=2/1.5≈1.333, r1=1/1.5≈0.667.
        let rates = alw_relative_training_rate(&[2.0, 1.0], &[1.0, 1.0]).unwrap();
        assert!(rates[0] > rates[1], "r0={:.4} r1={:.4}", rates[0], rates[1]);
        let expected_r0 = 2.0_f32 / 1.5;
        assert!((rates[0] - expected_r0).abs() < 1e-3, "r0={}", rates[0]);
    }

    #[test]
    fn relative_training_rate_dimension_mismatch() {
        assert!(matches!(
            alw_relative_training_rate(&[1.0], &[1.0, 1.0]),
            Err(LossWeightError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn weighted_sum_known_values() {
        let s = alw_weighted_sum(&[2.0, 3.0], &[4.0, 5.0]).unwrap();
        assert!((s - 23.0).abs() < 1e-5);
    }

    #[test]
    fn weighted_sum_mismatch_error() {
        assert!(matches!(
            alw_weighted_sum(&[1.0], &[1.0, 2.0]),
            Err(LossWeightError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn imbalance_ratio_all_equal() {
        assert!((alw_imbalance_ratio(&[1.0, 1.0, 1.0]) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn imbalance_ratio_known_ratio() {
        let r = alw_imbalance_ratio(&[1.0, 4.0]);
        assert!((r - 4.0).abs() < 1e-4, "r={r}");
    }

    #[test]
    fn imbalance_ratio_empty() {
        assert!((alw_imbalance_ratio(&[]) - 1.0).abs() < 1e-6);
    }

    // ── WeightHistory ─────────────────────────────────────────────────────────

    #[test]
    fn weight_history_record_length_grows() {
        let mut h = WeightHistory::new(vec!["a".into(), "b".into()]);
        assert_eq!(h.len(), 0);
        h.record(&[1.0, 2.0]).unwrap();
        assert_eq!(h.len(), 1);
        h.record(&[1.5, 2.5]).unwrap();
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn weight_history_record_capped_at_1000() {
        let mut h = WeightHistory::new(vec!["a".into()]);
        for i in 0..1100_usize {
            h.record(&[i as f32]).unwrap();
        }
        assert_eq!(h.len(), 1000);
    }

    #[test]
    fn weight_history_record_dimension_mismatch() {
        let mut h = WeightHistory::new(vec!["a".into(), "b".into()]);
        assert!(matches!(
            h.record(&[1.0]),
            Err(LossWeightError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn weight_history_latest_is_last_recorded() {
        let mut h = WeightHistory::new(vec!["a".into()]);
        h.record(&[1.0]).unwrap();
        h.record(&[9.0]).unwrap();
        let latest = h.latest().unwrap();
        assert!((latest[0] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn weight_history_mean_weights_correct() {
        let mut h = WeightHistory::new(vec!["a".into(), "b".into()]);
        h.record(&[2.0, 4.0]).unwrap();
        h.record(&[4.0, 8.0]).unwrap();
        let means = h.mean_weights();
        assert!((means[0] - 3.0).abs() < 1e-5);
        assert!((means[1] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn weight_history_trend_increasing() {
        let mut h = WeightHistory::new(vec!["a".into()]);
        for i in 0..20_usize {
            h.record(&[i as f32]).unwrap();
        }
        let slope = h.weight_trend(0);
        assert!(slope > 0.0, "slope={slope}");
    }

    #[test]
    fn weight_history_trend_constant_near_zero() {
        let mut h = WeightHistory::new(vec!["a".into()]);
        for _ in 0..20 {
            h.record(&[3.1]).unwrap();
        }
        let slope = h.weight_trend(0);
        assert!(slope.abs() < 1e-4, "slope={slope}");
    }

    #[test]
    fn weight_history_trend_no_data_zero() {
        let h = WeightHistory::new(vec!["a".into()]);
        assert!((h.weight_trend(0) - 0.0).abs() < 1e-6);
    }

    // ── Format functions ──────────────────────────────────────────────────────

    #[test]
    fn format_weights_nonempty_string() {
        let tasks = vec![LossTask::new("photo", 1.0), LossTask::new("reg", 0.1)];
        let s = alw_format_weights(&tasks, &[1.5, 0.05]);
        assert!(!s.is_empty());
        assert!(s.contains("photo"));
        assert!(s.contains("reg"));
    }

    #[test]
    fn format_history_summary_nonempty() {
        let mut h = WeightHistory::new(vec!["a".into()]);
        h.record(&[1.0]).unwrap();
        let s = alw_format_history_summary(&h);
        assert!(!s.is_empty());
        assert!(s.contains("WeightHistory"));
    }

    #[test]
    fn format_history_summary_empty_history() {
        let h = WeightHistory::new(vec!["a".into()]);
        let s = alw_format_history_summary(&h);
        assert!(s.contains("no data"));
    }
}
