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
//! [`LossTask::min_weight`]/[`LossTask::max_weight`] are hard bounds enforced
//! automatically only by [`HomoscedasticWeighter`] and [`GradNormWeighter`],
//! which hold the task list directly. [`ScheduledWeighter`] and
//! [`LossStatTracker`] have no notion of per-task bounds at all — callers
//! that need hard limits on weights coming from those two strategies should
//! pass them through [`alw_clip_weights`].
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
    /// A by-name lookup found no task with the requested name.
    ///
    /// Constructed by [`HomoscedasticWeighter::weight_by_name`],
    /// [`GradNormWeighter::weight_by_name`],
    /// [`ScheduledWeighter::weight_for_task`] and
    /// [`WeightHistory::task_index`]; the payload is the name that was asked
    /// for.
    #[error("task not found: {0}")]
    TaskNotFound(String),
    /// A σ (uncertainty) supplied for a task is not a positive finite number,
    /// so `log σ` — the parameter [`HomoscedasticWeighter`] actually stores —
    /// is undefined for it.
    ///
    /// Constructed by [`HomoscedasticWeighter::with_sigmas`] and
    /// [`HomoscedasticWeighter::set_sigma`]; the payload is the offending
    /// task index.
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
#[derive(Debug)]
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

    /// Construct a weighter whose per-task uncertainties start at `sigmas`
    /// instead of the default σ = 1 (`log σ = 0`).
    ///
    /// Useful when prior per-task noise estimates are known — for example
    /// carried over from a previous training run — since the stored parameter
    /// is `log σ` and the caller normally thinks in σ.
    ///
    /// # Errors
    /// - [`LossWeightError::EmptyTaskList`] when `tasks` is empty.
    /// - [`LossWeightError::DimensionMismatch`] when `sigmas.len() !=
    ///   tasks.len()`.
    /// - [`LossWeightError::NegativeLogSigma`] when any σ is not a positive
    ///   finite number, because `log σ` is undefined there.
    pub fn with_sigmas(tasks: Vec<LossTask>, sigmas: &[f32]) -> Result<Self, LossWeightError> {
        if tasks.is_empty() {
            return Err(LossWeightError::EmptyTaskList);
        }
        if sigmas.len() != tasks.len() {
            return Err(LossWeightError::DimensionMismatch {
                n_tasks: tasks.len(),
                n_losses: sigmas.len(),
            });
        }
        let mut log_sigmas = Vec::with_capacity(sigmas.len());
        for (idx, &sigma) in sigmas.iter().enumerate() {
            if !(sigma.is_finite() && sigma > 0.0) {
                return Err(LossWeightError::NegativeLogSigma(idx));
            }
            log_sigmas.push(sigma.ln());
        }
        Ok(Self { tasks, log_sigmas })
    }

    /// Replace task `task_idx`'s uncertainty with `sigma`, storing `log σ`.
    ///
    /// # Errors
    /// - [`LossWeightError::DimensionMismatch`] when `task_idx` is out of range.
    /// - [`LossWeightError::NegativeLogSigma`] when `sigma` is not a positive
    ///   finite number.
    pub fn set_sigma(&mut self, task_idx: usize, sigma: f32) -> Result<(), LossWeightError> {
        let slot = self
            .log_sigmas
            .get_mut(task_idx)
            .ok_or(LossWeightError::DimensionMismatch {
                n_tasks: self.tasks.len(),
                n_losses: task_idx + 1,
            })?;
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(LossWeightError::NegativeLogSigma(task_idx));
        }
        *slot = sigma.ln();
        Ok(())
    }

    /// Current per-task uncertainties σ = `exp(log σ)`.
    pub fn sigmas(&self) -> Vec<f32> {
        self.log_sigmas.iter().map(|&s| s.exp()).collect()
    }

    /// Index of the task called `name`.
    ///
    /// Returns [`LossWeightError::TaskNotFound`] when no task has that name.
    pub fn task_index(&self, name: &str) -> Result<usize, LossWeightError> {
        self.tasks
            .iter()
            .position(|t| t.name == name)
            .ok_or_else(|| LossWeightError::TaskNotFound(name.to_string()))
    }

    /// Effective weight of the task called `name`.
    ///
    /// Returns [`LossWeightError::TaskNotFound`] when no task has that name.
    pub fn weight_by_name(&self, name: &str) -> Result<f32, LossWeightError> {
        self.weight(self.task_index(name)?)
    }

    /// Effective weight for task `task_idx`: `exp(−2 · log_σ)`, clipped to
    /// the task's configured `[min_weight, max_weight]` bounds.
    pub fn weight(&self, task_idx: usize) -> Result<f32, LossWeightError> {
        if task_idx >= self.tasks.len() {
            return Err(LossWeightError::DimensionMismatch {
                n_tasks: self.tasks.len(),
                n_losses: task_idx + 1,
            });
        }
        let raw = (-2.0 * self.log_sigmas[task_idx]).exp();
        let task = &self.tasks[task_idx];
        Ok(raw.max(task.min_weight).min(task.max_weight))
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

    /// Total weighted loss: `Σ_i [ w_i · loss_i + log_σ_i ]`, where `w_i` is
    /// the bounds-clipped [`HomoscedasticWeighter::weight`] for task `i`
    /// (the `log_σ_i` regularization term itself is never clipped).
    pub fn total_loss(&self, losses: &[f32]) -> Result<f32, LossWeightError> {
        if losses.len() != self.tasks.len() {
            return Err(LossWeightError::DimensionMismatch {
                n_tasks: self.tasks.len(),
                n_losses: losses.len(),
            });
        }
        let mut total = 0.0_f32;
        for ((&sigma, &loss), task) in self
            .log_sigmas
            .iter()
            .zip(losses.iter())
            .zip(self.tasks.iter())
            .take(self.tasks.len())
        {
            let raw_w = (-2.0 * sigma).exp();
            let w = raw_w.max(task.min_weight).min(task.max_weight);
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
        if !alpha.is_finite() || alpha < 0.0 {
            return Err(LossWeightError::InvalidConfig(format!(
                "alpha must be finite and non-negative, got {alpha}"
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

        // Adjust each task weight toward the GradNorm target (unclipped).
        // Primary tasks (`LossTask::is_primary`) anchor the training
        // objective: they are left untouched here and excluded from the
        // rescale below, so their weight never drifts from its initial
        // value while every other task is rebalanced around it.
        for ((&r, &norm), (weight, task)) in loss_ratios
            .iter()
            .zip(self.gradient_norms.iter())
            .zip(self.weights.iter_mut().zip(self.tasks.iter()))
            .take(n)
        {
            if task.is_primary {
                continue;
            }
            let r_i = r / mean_ratio;
            // Relative training rates can go negative for signed loss terms
            // (e.g. regularizers); `powf` of a negative base with a
            // non-integer exponent is NaN, so clamp to a non-negative rate
            // first.
            let target_norm = mean_norm * r_i.max(0.0).powf(self.alpha);
            let norm_i = norm.max(1e-8);
            let new_weight = *weight * (target_norm / norm_i);
            *weight = if new_weight.is_finite() {
                new_weight
            } else {
                task.initial_weight
            };
        }

        // Re-normalize so the mean weight (over all `n` tasks, including
        // primaries — whose weight is a constant equal to their
        // `initial_weight`, since it is never touched by this method) still
        // equals the initial mean, then clip to each task's bounds. Only
        // non-primary weights are actually rescaled; a primary's weight
        // stays pinned at its constant value and is excluded from the
        // rescale multiplication (though it still counts toward both
        // means). Iterated to a fixed point: clipping after rescaling can
        // pull a weight back out toward its bound, which the previous
        // single-pass clip-then-rescale order could silently violate.
        let init_mean: f32 = self.tasks.iter().map(|t| t.initial_weight).sum::<f32>() / n as f32;
        for _ in 0..3 {
            let cur_mean: f32 = self.weights.iter().sum::<f32>() / n as f32;
            if cur_mean > 1e-8 {
                let scale = init_mean / cur_mean;
                for (w, task) in self.weights.iter_mut().zip(self.tasks.iter()) {
                    if !task.is_primary {
                        *w *= scale;
                    }
                }
            }
            for (w, task) in self.weights.iter_mut().zip(self.tasks.iter()) {
                *w = w.max(task.min_weight).min(task.max_weight);
            }
        }
        debug_assert!(self
            .weights
            .iter()
            .zip(self.tasks.iter())
            .all(|(w, t)| *w >= t.min_weight - 1e-4 && *w <= t.max_weight + 1e-4));

        self.step += 1;
        Ok(())
    }

    /// Current task weights.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Current weight of the task called `name`.
    ///
    /// Returns [`LossWeightError::TaskNotFound`] when no task has that name.
    pub fn weight_by_name(&self, name: &str) -> Result<f32, LossWeightError> {
        let idx = self
            .tasks
            .iter()
            .position(|t| t.name == name)
            .ok_or_else(|| LossWeightError::TaskNotFound(name.to_string()))?;
        self.weights
            .get(idx)
            .copied()
            .ok_or_else(|| LossWeightError::TaskNotFound(name.to_string()))
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
                // Binary-search for the surrounding pair. This assumes
                // `keyframes` is sorted by step (enforced by
                // `ScheduledWeighter::new`); if it is not — this variant can
                // also be constructed directly — fall back to the nearer
                // keyframe's weight instead of underflowing the `usize`
                // subtraction below or indexing out of bounds.
                let idx = keyframes.partition_point(|&(s, _)| s <= step);
                let idx0 = idx.saturating_sub(1).min(keyframes.len() - 1);
                let idx1 = idx.min(keyframes.len() - 1);
                let (s0, w0) = keyframes[idx0];
                let (s1, w1) = keyframes[idx1];
                if s1 <= s0 || step < s0 {
                    return w1;
                }
                let span = (s1 - s0) as f32;
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
    /// Returns [`LossWeightError::InvalidConfig`] when any
    /// [`WeightScheduleKind::Piecewise`] schedule's keyframes are not sorted
    /// by step (as its variant doc requires).
    pub fn new(schedules: Vec<TaskWeightSchedule>) -> Result<Self, LossWeightError> {
        if schedules.is_empty() {
            return Err(LossWeightError::EmptyTaskList);
        }
        for s in &schedules {
            if let WeightScheduleKind::Piecewise { keyframes } = &s.schedule {
                if keyframes.windows(2).any(|w| w[1].0 < w[0].0) {
                    return Err(LossWeightError::InvalidConfig(format!(
                        "task '{}': Piecewise keyframes must be sorted by step",
                        s.task_name
                    )));
                }
            }
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

    /// Weight of the task called `name` at `step`.
    ///
    /// Returns [`LossWeightError::TaskNotFound`] when no schedule carries that
    /// task name — the fallible counterpart of
    /// [`schedule_for`](Self::schedule_for), for callers that look weights up
    /// by name (e.g. from a config file) and must not silently drop a term
    /// whose name was mistyped.
    pub fn weight_for_task(&self, name: &str, step: usize) -> Result<f32, LossWeightError> {
        self.schedule_for(name)
            .map(|s| s.schedule.weight_at(step))
            .ok_or_else(|| LossWeightError::TaskNotFound(name.to_string()))
    }

    /// Weight of the task called `name` at the current internal step.
    ///
    /// Returns [`LossWeightError::TaskNotFound`] when no schedule carries that
    /// task name.
    pub fn current_weight_for_task(&self, name: &str) -> Result<f32, LossWeightError> {
        self.weight_for_task(name, self.step)
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
    ///
    /// The first call seeds `means`/`variances` directly from the observed
    /// losses instead of EMA-blending against the artificial `mean = 0.0`,
    /// `variance = 1.0` construction-time defaults, which would otherwise
    /// dominate the reported statistics for the first several dozen steps
    /// (exactly the phase of training where per-task loss scales differ
    /// most, and where `relative_magnitude_weights`/`inverse_variance_weights`
    /// most need to be trustworthy).
    pub fn update(&mut self, losses: &[f32]) -> Result<(), LossWeightError> {
        if losses.len() != self.n_tasks {
            return Err(LossWeightError::DimensionMismatch {
                n_tasks: self.n_tasks,
                n_losses: losses.len(),
            });
        }
        let d = self.ema_decay;
        let first_update = self.step == 0;
        for ((mean, variance), &loss) in self
            .means
            .iter_mut()
            .zip(self.variances.iter_mut())
            .zip(losses.iter())
            .take(self.n_tasks)
        {
            if first_update {
                *mean = loss;
                *variance = 0.0;
                continue;
            }
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

    /// Index of the task called `name`.
    ///
    /// Returns [`LossWeightError::TaskNotFound`] when no task has that name.
    pub fn task_index(&self, name: &str) -> Result<usize, LossWeightError> {
        self.task_names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| LossWeightError::TaskNotFound(name.to_string()))
    }

    /// [`weight_trend`](Self::weight_trend) for the task called `name`.
    ///
    /// Returns [`LossWeightError::TaskNotFound`] when no task has that name,
    /// instead of the index-based version's silent `0.0` for an out-of-range
    /// index.
    pub fn weight_trend_by_name(&self, name: &str) -> Result<f32, LossWeightError> {
        Ok(self.weight_trend(self.task_index(name)?))
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
mod tests;
