//! Continual learning techniques for Gaussian avatar models.
//!
//! Prevents catastrophic forgetting when adapting a model to new subjects while
//! retaining knowledge of previous subjects. Implements:
//! - **EWC** (Elastic Weight Consolidation): Fisher-information-based penalty
//! - **PackNet**-style progressive parameter freezing via task masks
//! - **Experience replay** with a circular buffer

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by continual learning operations.
#[derive(Debug, Error)]
pub enum ContinualLearningError {
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    #[error("Empty parameters")]
    EmptyParameters,

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Task not registered: {task_id}")]
    TaskNotRegistered { task_id: usize },

    #[error("No Fisher information computed yet")]
    NoFisherInfo,

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("Replay buffer empty")]
    ReplayBufferEmpty,
}

// ---------------------------------------------------------------------------
// EWC configuration
// ---------------------------------------------------------------------------

/// Configuration for Elastic Weight Consolidation.
#[derive(Debug, Clone)]
pub struct EwcConfig {
    /// Regularization strength (λ). Default: 1000.0.
    pub lambda: f32,
    /// Number of gradient samples used for Fisher estimation. Default: 200.
    pub fisher_samples: usize,
    /// `true` = online EWC (moving-average Fisher), `false` = vanilla EWC.
    pub online: bool,
    /// Decay factor for online EWC Fisher update. Default: 0.95.
    pub gamma: f32,
}

impl Default for EwcConfig {
    fn default() -> Self {
        Self {
            lambda: 1000.0,
            fisher_samples: 200,
            online: false,
            gamma: 0.95,
        }
    }
}

impl EwcConfig {
    /// Validates the configuration.
    ///
    /// Returns an error if:
    /// - `lambda` ≤ 0
    /// - `fisher_samples` < 1
    /// - `gamma` is not in `(0, 1]`
    pub fn validate(&self) -> Result<(), ContinualLearningError> {
        if self.lambda <= 0.0 {
            return Err(ContinualLearningError::InvalidConfig(format!(
                "lambda must be > 0, got {}",
                self.lambda
            )));
        }
        if self.fisher_samples < 1 {
            return Err(ContinualLearningError::InvalidConfig(
                "fisher_samples must be >= 1".to_string(),
            ));
        }
        if self.gamma <= 0.0 || self.gamma > 1.0 {
            return Err(ContinualLearningError::InvalidConfig(format!(
                "gamma must be in (0, 1], got {}",
                self.gamma
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Fisher information
// ---------------------------------------------------------------------------

/// Diagonal Fisher information estimate anchored to a completed task.
#[derive(Debug, Clone)]
pub struct FisherInformation {
    /// Identifier of the task this Fisher was computed for.
    pub task_id: usize,
    /// θ* — optimal parameter values at the end of this task.
    pub param_values: Vec<f32>,
    /// Diagonal Fisher: element-wise mean of squared gradients.
    pub fisher_diag: Vec<f32>,
}

impl FisherInformation {
    /// Constructs a new `FisherInformation`, verifying that `param_values` and
    /// `fisher_diag` have the same length and are non-empty.
    pub fn new(
        task_id: usize,
        param_values: Vec<f32>,
        fisher_diag: Vec<f32>,
    ) -> Result<Self, ContinualLearningError> {
        if param_values.is_empty() {
            return Err(ContinualLearningError::EmptyParameters);
        }
        if fisher_diag.len() != param_values.len() {
            return Err(ContinualLearningError::DimensionMismatch {
                expected: param_values.len(),
                actual: fisher_diag.len(),
            });
        }
        Ok(Self {
            task_id,
            param_values,
            fisher_diag,
        })
    }

    /// Number of parameters tracked.
    pub fn n_params(&self) -> usize {
        self.param_values.len()
    }
}

// ---------------------------------------------------------------------------
// EWC Regularizer
// ---------------------------------------------------------------------------

/// Maintains Fisher information for all previously completed tasks and computes
/// the EWC regularization penalty.
pub struct EwcRegularizer {
    /// EWC configuration.
    pub config: EwcConfig,
    /// Stored Fisher anchors, one per completed task.
    pub task_fishers: Vec<FisherInformation>,
}

impl EwcRegularizer {
    /// Creates a new `EwcRegularizer` with the given configuration.
    pub fn new(config: EwcConfig) -> Result<Self, ContinualLearningError> {
        config.validate()?;
        Ok(Self {
            config,
            task_fishers: Vec::new(),
        })
    }

    /// Number of tasks whose Fisher information has been registered.
    pub fn n_tasks(&self) -> usize {
        self.task_fishers.len()
    }

    /// Register the current parameter state as the optimum for a completed task.
    ///
    /// For online EWC, the Fisher diagonal of a previously registered task with the
    /// same `task_id` is updated via the exponential moving average.
    pub fn register_task(
        &mut self,
        task_id: usize,
        params: Vec<f32>,
        fisher: Vec<f32>,
    ) -> Result<(), ContinualLearningError> {
        if params.is_empty() {
            return Err(ContinualLearningError::EmptyParameters);
        }
        if fisher.len() != params.len() {
            return Err(ContinualLearningError::DimensionMismatch {
                expected: params.len(),
                actual: fisher.len(),
            });
        }

        if self.config.online {
            // Find existing anchor for this task_id and apply EMA update.
            if let Some(existing) = self.task_fishers.iter_mut().find(|f| f.task_id == task_id) {
                let gamma = self.config.gamma;
                for (old, new_val) in existing.fisher_diag.iter_mut().zip(fisher.iter()) {
                    *old = gamma * (*old) + (1.0 - gamma) * new_val;
                }
                existing.param_values = params;
                return Ok(());
            }
        }

        let info = FisherInformation::new(task_id, params, fisher)?;
        self.task_fishers.push(info);
        Ok(())
    }

    /// Compute the total EWC penalty across all registered tasks:
    ///
    /// `L_EWC = (λ/2) * Σ_i Σ_j F_ij * (θ_j − θ*_ij)²`
    ///
    /// Returns `Ok(0.0)` when no tasks have been registered yet.
    pub fn penalty(&self, current_params: &[f32]) -> Result<f32, ContinualLearningError> {
        if self.task_fishers.is_empty() {
            return Ok(0.0);
        }

        let n = current_params.len();
        let mut total = 0.0f32;
        for fi in &self.task_fishers {
            if fi.n_params() != n {
                return Err(ContinualLearningError::DimensionMismatch {
                    expected: fi.n_params(),
                    actual: n,
                });
            }
            total += ewc_penalty_single(
                current_params,
                &fi.param_values,
                &fi.fisher_diag,
                self.config.lambda,
            )?;
        }
        Ok(total)
    }

    /// Compute the gradient of the total EWC penalty with respect to `current_params`.
    ///
    /// Returns a zero vector when no tasks have been registered yet.
    pub fn penalty_gradient(
        &self,
        current_params: &[f32],
    ) -> Result<Vec<f32>, ContinualLearningError> {
        let n = current_params.len();
        if self.task_fishers.is_empty() {
            return Ok(vec![0.0; n]);
        }

        let mut grad = vec![0.0f32; n];
        for fi in &self.task_fishers {
            if fi.n_params() != n {
                return Err(ContinualLearningError::DimensionMismatch {
                    expected: fi.n_params(),
                    actual: n,
                });
            }
            let task_grad = ewc_gradient_single(
                current_params,
                &fi.param_values,
                &fi.fisher_diag,
                self.config.lambda,
            )?;
            for (g, tg) in grad.iter_mut().zip(task_grad.iter()) {
                *g += tg;
            }
        }
        Ok(grad)
    }
}

// ---------------------------------------------------------------------------
// Replay buffer
// ---------------------------------------------------------------------------

/// A single (input, target) sample pair stored by the replay buffer.
type SamplePair = (Vec<f32>, Vec<f32>);

/// Circular experience replay buffer storing (input, target) pairs.
#[derive(Debug, Clone)]
pub struct ReplayBuffer {
    /// Maximum number of entries the buffer holds.
    pub capacity: usize,
    samples: Vec<(Vec<f32>, Vec<f32>)>,
    /// Index of the next write position (circular).
    head: usize,
    /// Current number of valid entries (≤ capacity).
    size: usize,
}

impl ReplayBuffer {
    /// Creates an empty replay buffer with the given capacity.
    ///
    /// Panics are intentionally avoided; capacity of 0 is accepted and will
    /// cause every `push` to be a no-op.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            samples: Vec::with_capacity(capacity),
            head: 0,
            size: 0,
        }
    }

    /// Push a new (input, target) pair into the buffer.
    ///
    /// When the buffer is full the oldest entry is overwritten (circular FIFO).
    pub fn push(&mut self, input: Vec<f32>, target: Vec<f32>) {
        if self.capacity == 0 {
            return;
        }
        if self.size < self.capacity {
            self.samples.push((input, target));
            self.size += 1;
        } else {
            self.samples[self.head] = (input, target);
        }
        self.head = (self.head + 1) % self.capacity;
    }

    /// Current number of entries in the buffer.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` when the buffer contains no entries.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Sample `n` random (input, target) pairs (with replacement when `n > size`).
    pub fn sample(&self, n: usize, seed: u64) -> Result<Vec<SamplePair>, ContinualLearningError> {
        if self.is_empty() {
            return Err(ContinualLearningError::ReplayBufferEmpty);
        }

        let mut state = seed.max(1);
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = xorshift64(&mut state) as usize % self.size;
            result.push(self.samples[idx].clone());
        }
        Ok(result)
    }

    /// Remove all entries from the buffer.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.head = 0;
        self.size = 0;
    }
}

// ---------------------------------------------------------------------------
// Task mask (PackNet-style)
// ---------------------------------------------------------------------------

/// Binary mask selecting which parameters are active (unfrozen) for a task.
#[derive(Debug, Clone)]
pub struct TaskMask {
    /// Owning task identifier.
    pub task_id: usize,
    /// `true` = active/unfrozen, `false` = frozen.
    pub mask: Vec<bool>,
    /// Total number of parameters.
    pub n_params: usize,
}

impl TaskMask {
    /// Create a mask where every parameter is active.
    pub fn new_all_active(task_id: usize, n_params: usize) -> Self {
        Self {
            task_id,
            mask: vec![true; n_params],
            n_params,
        }
    }

    /// Fraction of parameters that are active (in `[0.0, 1.0]`).
    pub fn active_fraction(&self) -> f32 {
        if self.n_params == 0 {
            return 0.0;
        }
        let active = self.mask.iter().filter(|&&b| b).count();
        active as f32 / self.n_params as f32
    }

    /// Zero out gradients for frozen parameters (mask entry = `false`).
    pub fn apply_mask(&self, gradients: &mut [f32]) -> Result<(), ContinualLearningError> {
        if gradients.len() != self.n_params {
            return Err(ContinualLearningError::DimensionMismatch {
                expected: self.n_params,
                actual: gradients.len(),
            });
        }
        for (g, &active) in gradients.iter_mut().zip(self.mask.iter()) {
            if !active {
                *g = 0.0;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Estimate the diagonal Fisher information from a collection of gradient snapshots.
///
/// Each snapshot is the gradient vector computed on one training sample.
/// Returns the element-wise mean of squared gradients:
/// `F_j = (1/N) * Σ_i g_{ij}²`
pub fn estimate_fisher_diagonal(
    gradients: &[Vec<f32>],
) -> Result<Vec<f32>, ContinualLearningError> {
    if gradients.is_empty() {
        return Err(ContinualLearningError::EmptyParameters);
    }

    let n_params = gradients[0].len();
    if n_params == 0 {
        return Err(ContinualLearningError::EmptyParameters);
    }

    let mut fisher = vec![0.0f32; n_params];
    for snapshot in gradients {
        if snapshot.len() != n_params {
            return Err(ContinualLearningError::DimensionMismatch {
                expected: n_params,
                actual: snapshot.len(),
            });
        }
        for (f, g) in fisher.iter_mut().zip(snapshot.iter()) {
            *f += g * g;
        }
    }

    let n = gradients.len() as f32;
    for f in fisher.iter_mut() {
        *f /= n;
    }
    Ok(fisher)
}

/// Online EWC Fisher update via exponential moving average.
///
/// `F_new = γ * F_old + (1 − γ) * g²`
///
/// The raw `new_gradients` are squared element-wise inside this function.
pub fn update_online_fisher(
    old_fisher: &[f32],
    new_gradients: &[f32],
    gamma: f32,
) -> Result<Vec<f32>, ContinualLearningError> {
    if old_fisher.len() != new_gradients.len() {
        return Err(ContinualLearningError::DimensionMismatch {
            expected: old_fisher.len(),
            actual: new_gradients.len(),
        });
    }
    if old_fisher.is_empty() {
        return Err(ContinualLearningError::EmptyParameters);
    }

    let one_minus_gamma = 1.0 - gamma;
    let result = old_fisher
        .iter()
        .zip(new_gradients.iter())
        .map(|(&f_old, &g)| gamma * f_old + one_minus_gamma * g * g)
        .collect();
    Ok(result)
}

/// Compute the EWC penalty for a single previous task.
///
/// `L_EWC = (λ/2) * Σ_j F_j * (θ_j − θ*_j)²`
pub fn ewc_penalty_single(
    current_params: &[f32],
    optimal_params: &[f32],
    fisher: &[f32],
    lambda: f32,
) -> Result<f32, ContinualLearningError> {
    let n = current_params.len();
    if n == 0 {
        return Err(ContinualLearningError::EmptyParameters);
    }
    if optimal_params.len() != n {
        return Err(ContinualLearningError::DimensionMismatch {
            expected: n,
            actual: optimal_params.len(),
        });
    }
    if fisher.len() != n {
        return Err(ContinualLearningError::DimensionMismatch {
            expected: n,
            actual: fisher.len(),
        });
    }

    let penalty: f32 = current_params
        .iter()
        .zip(optimal_params.iter())
        .zip(fisher.iter())
        .map(|((&theta, &theta_star), &f)| f * (theta - theta_star) * (theta - theta_star))
        .sum();

    Ok(0.5 * lambda * penalty)
}

/// Compute the gradient of the EWC penalty for a single task:
///
/// `∂L_EWC/∂θ_j = λ * F_j * (θ_j − θ*_j)`
///
/// Note: the factor of 2 from differentiating `x²` cancels the 1/2 in the penalty.
pub fn ewc_gradient_single(
    current_params: &[f32],
    optimal_params: &[f32],
    fisher: &[f32],
    lambda: f32,
) -> Result<Vec<f32>, ContinualLearningError> {
    let n = current_params.len();
    if n == 0 {
        return Err(ContinualLearningError::EmptyParameters);
    }
    if optimal_params.len() != n {
        return Err(ContinualLearningError::DimensionMismatch {
            expected: n,
            actual: optimal_params.len(),
        });
    }
    if fisher.len() != n {
        return Err(ContinualLearningError::DimensionMismatch {
            expected: n,
            actual: fisher.len(),
        });
    }

    let grad = current_params
        .iter()
        .zip(optimal_params.iter())
        .zip(fisher.iter())
        .map(|((&theta, &theta_star), &f)| lambda * f * (theta - theta_star))
        .collect();
    Ok(grad)
}

/// Compute per-parameter importance scores for PackNet pruning.
///
/// Importance is the element-wise mean of squared gradients across all snapshots
/// (identical to the diagonal Fisher estimator, re-exposed with a clearer name for
/// the masking use case).
pub fn compute_parameter_importance(
    gradients: &[Vec<f32>],
) -> Result<Vec<f32>, ContinualLearningError> {
    estimate_fisher_diagonal(gradients)
}

/// Create a PackNet-style task mask.
///
/// The `active_fraction` most important parameters (by importance score) are kept
/// active; the rest are frozen.  Parameters already frozen in `previous_mask` cannot
/// be reactivated regardless of their importance score.
///
/// - `active_fraction = 1.0` → all parameters active (subject to previous mask).
/// - `active_fraction = 0.0` → all parameters frozen.
pub fn create_task_mask(
    importance: &[f32],
    task_id: usize,
    active_fraction: f32,
    previous_mask: Option<&TaskMask>,
) -> Result<TaskMask, ContinualLearningError> {
    if importance.is_empty() {
        return Err(ContinualLearningError::EmptyParameters);
    }
    if !(0.0..=1.0).contains(&active_fraction) {
        return Err(ContinualLearningError::InvalidConfig(format!(
            "active_fraction must be in [0, 1], got {}",
            active_fraction
        )));
    }

    let n = importance.len();

    // Determine which indices are *eligible* to be active (not frozen previously).
    let eligible: Vec<usize> = match previous_mask {
        None => (0..n).collect(),
        Some(pm) => {
            if pm.n_params != n {
                return Err(ContinualLearningError::DimensionMismatch {
                    expected: n,
                    actual: pm.n_params,
                });
            }
            (0..n).filter(|&i| pm.mask[i]).collect()
        }
    };

    let n_to_activate = (eligible.len() as f32 * active_fraction).round() as usize;

    // Rank eligible indices by descending importance.
    let mut ranked = eligible.clone();
    ranked.sort_by(|&a, &b| {
        importance[b]
            .partial_cmp(&importance[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build the mask: start all frozen, then activate the top n_to_activate.
    let mut mask = vec![false; n];
    for &idx in ranked.iter().take(n_to_activate) {
        mask[idx] = true;
    }

    Ok(TaskMask {
        task_id,
        mask,
        n_params: n,
    })
}

/// Forgetting measure: amount by which performance on task i has degraded.
///
/// Returns a positive value when forgetting has occurred.
/// `forgetting = acc_at_task_i_end − acc_current`
pub fn forgetting_measure(acc_at_task_i_end: f32, acc_current: f32) -> f32 {
    acc_at_task_i_end - acc_current
}

/// Backward transfer: how training on task j affected task i (trained before j).
///
/// Positive = helpful backward transfer, negative = forgetting.
/// `BWT = acc_after_j − acc_before_j`
pub fn backward_transfer(acc_after_j: f32, acc_before_j: f32) -> f32 {
    acc_after_j - acc_before_j
}

/// Compute the mean accuracy across all tasks.
///
/// Returns an error when the slice is empty.
pub fn average_task_accuracy(accuracies: &[f32]) -> Result<f32, ContinualLearningError> {
    if accuracies.is_empty() {
        return Err(ContinualLearningError::EmptyParameters);
    }
    let sum: f32 = accuracies.iter().sum();
    Ok(sum / accuracies.len() as f32)
}

/// Generate synthetic gradient snapshots for testing and ablation studies.
///
/// Uses inline xorshift64 + Box-Muller to produce `n_samples` gradient vectors
/// each of length `n_params`, scaled by `task_difficulty`.
pub fn simulate_task_gradients(
    n_params: usize,
    n_samples: usize,
    task_difficulty: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    if n_params == 0 || n_samples == 0 {
        return Vec::new();
    }

    let mut state = seed.max(1);
    let mut gradients = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let mut snapshot = Vec::with_capacity(n_params);
        let mut i = 0;
        while i < n_params {
            // Box-Muller transform: two uniform samples → two normals
            let u1 = (xorshift64(&mut state) as f64 + 1.0) / (u64::MAX as f64 + 2.0);
            let u2 = (xorshift64(&mut state) as f64 + 1.0) / (u64::MAX as f64 + 2.0);
            let mag = (-2.0 * u1.ln()).sqrt();
            let angle = 2.0 * std::f64::consts::PI * u2;
            let z0 = (mag * angle.cos()) as f32 * task_difficulty;
            let z1 = (mag * angle.sin()) as f32 * task_difficulty;
            snapshot.push(z0);
            i += 1;
            if i < n_params {
                snapshot.push(z1);
                i += 1;
            }
        }
        gradients.push(snapshot);
    }

    gradients
}

/// Compute the replay-augmented loss.
///
/// Combines the current task loss with the mean replay loss via a convex combination:
///
/// `L = (1 − replay_weight) * current_loss + replay_weight * mean(replay_losses)`
///
/// When `replay_losses` is empty the current loss is returned unchanged.
pub fn replay_loss(current_loss: f32, replay_losses: &[f32], replay_weight: f32) -> f32 {
    if replay_losses.is_empty() {
        return current_loss;
    }
    let mean_replay: f32 = replay_losses.iter().sum::<f32>() / replay_losses.len() as f32;
    (1.0 - replay_weight) * current_loss + replay_weight * mean_replay
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Summary statistics for a continual learning training state.
#[derive(Debug, Clone)]
pub struct ContinualLearningStats {
    /// Number of registered tasks.
    pub n_tasks: usize,
    /// Current EWC penalty value.
    pub ewc_penalty: f32,
    /// Mean forgetting across all tasks (from `task_accuracies` pairs).
    pub mean_forgetting: f32,
    /// Mean backward transfer across all tasks.
    pub backward_transfer: f32,
    /// Current number of entries in the replay buffer.
    pub replay_buffer_size: usize,
    /// Fraction of active parameters in the current task mask.
    pub active_param_fraction: f32,
}

/// Compute a snapshot of continual learning statistics.
///
/// `task_accuracies`: pairs of `(acc_at_task_i_end, acc_current)`.
pub fn compute_cl_stats(
    regularizer: &EwcRegularizer,
    current_params: &[f32],
    task_accuracies: &[(f32, f32)],
    replay: &ReplayBuffer,
    mask: Option<&TaskMask>,
) -> Result<ContinualLearningStats, ContinualLearningError> {
    let ewc_penalty = regularizer.penalty(current_params)?;

    let (mean_forgetting, mean_bwt) = if task_accuracies.is_empty() {
        (0.0, 0.0)
    } else {
        let forgetting_sum: f32 = task_accuracies
            .iter()
            .map(|&(end, cur)| forgetting_measure(end, cur))
            .sum();
        let bwt_sum: f32 = task_accuracies
            .iter()
            .map(|&(cur, end)| backward_transfer(cur, end))
            .sum();
        let n = task_accuracies.len() as f32;
        (forgetting_sum / n, bwt_sum / n)
    };

    let active_param_fraction = mask.map(|m| m.active_fraction()).unwrap_or(1.0);

    Ok(ContinualLearningStats {
        n_tasks: regularizer.n_tasks(),
        ewc_penalty,
        mean_forgetting,
        backward_transfer: mean_bwt,
        replay_buffer_size: replay.len(),
        active_param_fraction,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Xorshift64 PRNG — seed must never be zero.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    // Guard: zero is a fixed point for xorshift64.
    *state = (*state).max(1);
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helper
    // ------------------------------------------------------------------
    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ------------------------------------------------------------------
    // EwcConfig::validate
    // ------------------------------------------------------------------
    #[test]
    fn test_ewc_config_default_is_valid() {
        assert!(EwcConfig::default().validate().is_ok());
    }

    #[test]
    fn test_ewc_config_lambda_zero_error() {
        let cfg = EwcConfig {
            lambda: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ContinualLearningError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_ewc_config_lambda_negative_error() {
        let cfg = EwcConfig {
            lambda: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ContinualLearningError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_ewc_config_gamma_above_one_error() {
        let cfg = EwcConfig {
            gamma: 1.1,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ContinualLearningError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_ewc_config_gamma_zero_error() {
        let cfg = EwcConfig {
            gamma: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ContinualLearningError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_ewc_config_fisher_samples_zero_error() {
        let cfg = EwcConfig {
            fisher_samples: 0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(ContinualLearningError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_ewc_config_gamma_one_valid() {
        let cfg = EwcConfig {
            gamma: 1.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    // ------------------------------------------------------------------
    // FisherInformation::new
    // ------------------------------------------------------------------
    #[test]
    fn test_fisher_info_new_valid() {
        let fi = FisherInformation::new(0, vec![1.0, 2.0], vec![0.5, 0.5]);
        assert!(fi.is_ok());
        assert_eq!(fi.expect("ok").n_params(), 2);
    }

    #[test]
    fn test_fisher_info_empty_params_error() {
        assert!(matches!(
            FisherInformation::new(0, vec![], vec![]),
            Err(ContinualLearningError::EmptyParameters)
        ));
    }

    #[test]
    fn test_fisher_info_dimension_mismatch() {
        assert!(matches!(
            FisherInformation::new(0, vec![1.0, 2.0], vec![0.5]),
            Err(ContinualLearningError::DimensionMismatch { .. })
        ));
    }

    // ------------------------------------------------------------------
    // EwcRegularizer
    // ------------------------------------------------------------------
    #[test]
    fn test_ewc_regularizer_new_ok() {
        let reg = EwcRegularizer::new(EwcConfig::default());
        assert!(reg.is_ok());
        assert_eq!(reg.expect("ok").n_tasks(), 0);
    }

    #[test]
    fn test_ewc_regularizer_new_bad_config_fails() {
        let cfg = EwcConfig {
            lambda: 0.0,
            ..Default::default()
        };
        assert!(EwcRegularizer::new(cfg).is_err());
    }

    #[test]
    fn test_ewc_register_task() {
        let mut reg = EwcRegularizer::new(EwcConfig::default()).expect("ok");
        reg.register_task(0, vec![1.0, 2.0], vec![0.1, 0.2])
            .expect("register");
        assert_eq!(reg.n_tasks(), 1);
    }

    #[test]
    fn test_ewc_penalty_zero_at_optimal() {
        let mut reg = EwcRegularizer::new(EwcConfig::default()).expect("ok");
        reg.register_task(0, vec![1.0, 2.0], vec![1.0, 1.0])
            .expect("register");
        // current == optimal → penalty must be 0
        let p = reg.penalty(&[1.0, 2.0]).expect("penalty");
        assert!(close(p, 0.0, 1e-6));
    }

    #[test]
    fn test_ewc_penalty_positive_when_displaced() {
        let mut reg = EwcRegularizer::new(EwcConfig::default()).expect("ok");
        reg.register_task(0, vec![0.0, 0.0], vec![1.0, 1.0])
            .expect("register");
        let p = reg.penalty(&[1.0, 0.0]).expect("penalty");
        assert!(p > 0.0);
    }

    #[test]
    fn test_ewc_penalty_gradient_zero_at_optimal() {
        let mut reg = EwcRegularizer::new(EwcConfig::default()).expect("ok");
        reg.register_task(0, vec![1.0, 2.0], vec![1.0, 1.0])
            .expect("register");
        let g = reg.penalty_gradient(&[1.0, 2.0]).expect("grad");
        for v in &g {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_ewc_penalty_gradient_direction() {
        let mut reg = EwcRegularizer::new(EwcConfig {
            lambda: 1.0,
            ..Default::default()
        })
        .expect("ok");
        // optimal = [0, 0], fisher = [1, 1]
        reg.register_task(0, vec![0.0, 0.0], vec![1.0, 1.0])
            .expect("register");
        // current = [1, -1] → gradient should be [+1, -1]
        let g = reg.penalty_gradient(&[1.0, -1.0]).expect("grad");
        assert!(g[0] > 0.0);
        assert!(g[1] < 0.0);
    }

    #[test]
    fn test_ewc_no_tasks_penalty_is_zero() {
        let reg = EwcRegularizer::new(EwcConfig::default()).expect("ok");
        let p = reg.penalty(&[1.0, 2.0]).expect("penalty");
        assert!(close(p, 0.0, 1e-10));
    }

    // ------------------------------------------------------------------
    // ReplayBuffer
    // ------------------------------------------------------------------
    #[test]
    fn test_replay_buffer_new() {
        let buf = ReplayBuffer::new(10);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        assert_eq!(buf.capacity, 10);
    }

    #[test]
    fn test_replay_buffer_push_and_len() {
        let mut buf = ReplayBuffer::new(3);
        buf.push(vec![1.0], vec![0.0]);
        assert_eq!(buf.len(), 1);
        buf.push(vec![2.0], vec![0.0]);
        buf.push(vec![3.0], vec![0.0]);
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn test_replay_buffer_circular_eviction() {
        let mut buf = ReplayBuffer::new(3);
        buf.push(vec![1.0], vec![10.0]);
        buf.push(vec![2.0], vec![20.0]);
        buf.push(vec![3.0], vec![30.0]);
        // Buffer is full; pushing one more evicts the oldest (index 0)
        buf.push(vec![4.0], vec![40.0]);
        assert_eq!(buf.len(), 3); // capacity stays the same
                                  // The buffer should now contain entries with inputs 2, 3, 4
        let stored_inputs: Vec<f32> = buf.samples.iter().map(|(inp, _)| inp[0]).collect();
        assert!(stored_inputs.contains(&2.0));
        assert!(stored_inputs.contains(&3.0));
        assert!(stored_inputs.contains(&4.0));
        assert!(!stored_inputs.contains(&1.0));
    }

    #[test]
    fn test_replay_buffer_sample_empty_error() {
        let buf = ReplayBuffer::new(10);
        assert!(matches!(
            buf.sample(5, 42),
            Err(ContinualLearningError::ReplayBufferEmpty)
        ));
    }

    #[test]
    fn test_replay_buffer_sample_with_replacement() {
        let mut buf = ReplayBuffer::new(3);
        buf.push(vec![1.0], vec![0.0]);
        buf.push(vec![2.0], vec![0.0]);
        // Sample more than available — should work via replacement
        let result = buf.sample(10, 99);
        assert!(result.is_ok());
        assert_eq!(result.expect("sample").len(), 10);
    }

    #[test]
    fn test_replay_buffer_clear() {
        let mut buf = ReplayBuffer::new(5);
        buf.push(vec![1.0], vec![0.0]);
        buf.push(vec![2.0], vec![0.0]);
        buf.clear();
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    // ------------------------------------------------------------------
    // TaskMask
    // ------------------------------------------------------------------
    #[test]
    fn test_task_mask_new_all_active() {
        let m = TaskMask::new_all_active(0, 5);
        assert_eq!(m.n_params, 5);
        assert!(m.mask.iter().all(|&b| b));
        assert!(close(m.active_fraction(), 1.0, 1e-6));
    }

    #[test]
    fn test_task_mask_active_fraction() {
        let m = TaskMask {
            task_id: 0,
            mask: vec![true, false, true, false],
            n_params: 4,
        };
        assert!(close(m.active_fraction(), 0.5, 1e-6));
    }

    #[test]
    fn test_task_mask_apply_mask_zeros_frozen() {
        let m = TaskMask {
            task_id: 0,
            mask: vec![true, false, true],
            n_params: 3,
        };
        let mut grads = vec![1.0, 2.0, 3.0];
        m.apply_mask(&mut grads).expect("apply");
        assert!(close(grads[0], 1.0, 1e-7));
        assert!(close(grads[1], 0.0, 1e-7)); // frozen → zeroed
        assert!(close(grads[2], 3.0, 1e-7));
    }

    #[test]
    fn test_task_mask_apply_mask_dimension_mismatch() {
        let m = TaskMask::new_all_active(0, 3);
        let mut grads = vec![1.0, 2.0]; // wrong length
        assert!(matches!(
            m.apply_mask(&mut grads),
            Err(ContinualLearningError::DimensionMismatch { .. })
        ));
    }

    // ------------------------------------------------------------------
    // estimate_fisher_diagonal
    // ------------------------------------------------------------------
    #[test]
    fn test_estimate_fisher_single_snapshot_is_squared() {
        let grads = vec![vec![2.0_f32, 3.0_f32]];
        let fisher = estimate_fisher_diagonal(&grads).expect("ok");
        assert!(close(fisher[0], 4.0, 1e-6));
        assert!(close(fisher[1], 9.0, 1e-6));
    }

    #[test]
    fn test_estimate_fisher_multiple_snapshots_mean_of_squares() {
        let grads = vec![vec![2.0_f32, 0.0_f32], vec![0.0_f32, 4.0_f32]];
        let fisher = estimate_fisher_diagonal(&grads).expect("ok");
        // mean([4, 0]) = 2,  mean([0, 16]) = 8
        assert!(close(fisher[0], 2.0, 1e-6));
        assert!(close(fisher[1], 8.0, 1e-6));
    }

    #[test]
    fn test_estimate_fisher_empty_error() {
        let grads: Vec<Vec<f32>> = vec![];
        assert!(matches!(
            estimate_fisher_diagonal(&grads),
            Err(ContinualLearningError::EmptyParameters)
        ));
    }

    // ------------------------------------------------------------------
    // update_online_fisher
    // ------------------------------------------------------------------
    #[test]
    fn test_update_online_fisher_gamma_one_old_unchanged() {
        let old = vec![5.0_f32, 3.0_f32];
        let new_g = vec![10.0_f32, 20.0_f32];
        let result = update_online_fisher(&old, &new_g, 1.0).expect("ok");
        // gamma=1 → 1.0*old + 0.0*g² = old
        assert!(close(result[0], 5.0, 1e-6));
        assert!(close(result[1], 3.0, 1e-6));
    }

    #[test]
    fn test_update_online_fisher_gamma_zero_new_squared() {
        let old = vec![5.0_f32, 3.0_f32];
        let new_g = vec![3.0_f32, 4.0_f32];
        let result = update_online_fisher(&old, &new_g, 0.0).expect("ok");
        // gamma=0 → 0*old + 1.0*g² = g²
        assert!(close(result[0], 9.0, 1e-6));
        assert!(close(result[1], 16.0, 1e-6));
    }

    // ------------------------------------------------------------------
    // ewc_penalty_single
    // ------------------------------------------------------------------
    #[test]
    fn test_ewc_penalty_single_at_optimal_is_zero() {
        let p = ewc_penalty_single(&[1.0, 2.0], &[1.0, 2.0], &[10.0, 10.0], 1000.0).expect("ok");
        assert!(close(p, 0.0, 1e-6));
    }

    #[test]
    fn test_ewc_penalty_single_displaced_positive() {
        // λ=2, F=[1,1], Δθ=[1,0] → 0.5*2*1*1 = 1.0
        let p = ewc_penalty_single(&[1.0, 0.0], &[0.0, 0.0], &[1.0, 1.0], 2.0).expect("ok");
        assert!(close(p, 1.0, 1e-6));
    }

    // ------------------------------------------------------------------
    // ewc_gradient_single
    // ------------------------------------------------------------------
    #[test]
    fn test_ewc_gradient_single_at_optimal_zero() {
        let g = ewc_gradient_single(&[1.0, 2.0], &[1.0, 2.0], &[10.0, 10.0], 1000.0).expect("ok");
        for v in &g {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_ewc_gradient_single_direction() {
        // λ=1, F=[1,1], θ=[2,0], θ*=[0,0] → grad=[2,0]
        let g = ewc_gradient_single(&[2.0, 0.0], &[0.0, 0.0], &[1.0, 1.0], 1.0).expect("ok");
        assert!(close(g[0], 2.0, 1e-6));
        assert!(close(g[1], 0.0, 1e-6));
    }

    // ------------------------------------------------------------------
    // compute_parameter_importance
    // ------------------------------------------------------------------
    #[test]
    fn test_compute_parameter_importance_larger_grad_higher() {
        let grads = vec![vec![1.0_f32, 5.0_f32], vec![1.0_f32, 5.0_f32]];
        let imp = compute_parameter_importance(&grads).expect("ok");
        // param 1 has larger gradient → higher importance
        assert!(imp[1] > imp[0]);
    }

    // ------------------------------------------------------------------
    // create_task_mask
    // ------------------------------------------------------------------
    #[test]
    fn test_create_task_mask_all_active() {
        let importance = vec![1.0, 2.0, 3.0, 4.0];
        let mask = create_task_mask(&importance, 0, 1.0, None).expect("ok");
        assert!(mask.mask.iter().all(|&b| b));
    }

    #[test]
    fn test_create_task_mask_none_active() {
        let importance = vec![1.0, 2.0, 3.0];
        let mask = create_task_mask(&importance, 0, 0.0, None).expect("ok");
        assert!(mask.mask.iter().all(|&b| !b));
    }

    #[test]
    fn test_create_task_mask_previous_frozen_stay_frozen() {
        // 4 params; previous mask freezes params 0 and 1
        let prev = TaskMask {
            task_id: 0,
            mask: vec![false, false, true, true],
            n_params: 4,
        };
        // All eligible (2 and 3) should be active when active_fraction=1.0
        let importance = vec![10.0, 10.0, 1.0, 2.0];
        let new_mask = create_task_mask(&importance, 1, 1.0, Some(&prev)).expect("ok");
        // params 0 and 1 must remain frozen
        assert!(!new_mask.mask[0]);
        assert!(!new_mask.mask[1]);
        // params 2 and 3 should be active
        assert!(new_mask.mask[2]);
        assert!(new_mask.mask[3]);
    }

    #[test]
    fn test_create_task_mask_selects_highest_importance() {
        // 4 params, pick top 50% (2 out of 4)
        let importance = vec![1.0_f32, 5.0, 3.0, 0.5];
        let mask = create_task_mask(&importance, 0, 0.5, None).expect("ok");
        let active: Vec<usize> = mask
            .mask
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i)
            .collect();
        // Highest: index 1 (5.0) and index 2 (3.0)
        assert!(active.contains(&1));
        assert!(active.contains(&2));
    }

    // ------------------------------------------------------------------
    // forgetting_measure
    // ------------------------------------------------------------------
    #[test]
    fn test_forgetting_measure_no_forgetting() {
        assert!(close(forgetting_measure(0.9, 0.9), 0.0, 1e-7));
    }

    #[test]
    fn test_forgetting_measure_positive_when_degraded() {
        assert!(forgetting_measure(0.9, 0.7) > 0.0);
    }

    // ------------------------------------------------------------------
    // backward_transfer
    // ------------------------------------------------------------------
    #[test]
    fn test_backward_transfer_positive() {
        assert!(backward_transfer(0.9, 0.7) > 0.0);
    }

    #[test]
    fn test_backward_transfer_negative() {
        assert!(backward_transfer(0.7, 0.9) < 0.0);
    }

    #[test]
    fn test_backward_transfer_zero() {
        assert!(close(backward_transfer(0.8, 0.8), 0.0, 1e-7));
    }

    // ------------------------------------------------------------------
    // average_task_accuracy
    // ------------------------------------------------------------------
    #[test]
    fn test_average_task_accuracy_empty_error() {
        assert!(matches!(
            average_task_accuracy(&[]),
            Err(ContinualLearningError::EmptyParameters)
        ));
    }

    #[test]
    fn test_average_task_accuracy_valid() {
        let acc = average_task_accuracy(&[0.8, 0.6, 0.7]).expect("ok");
        assert!(close(acc, 0.7, 1e-6));
    }

    // ------------------------------------------------------------------
    // simulate_task_gradients
    // ------------------------------------------------------------------
    #[test]
    fn test_simulate_task_gradients_correct_shape() {
        let grads = simulate_task_gradients(10, 5, 1.0, 42);
        assert_eq!(grads.len(), 5);
        for snapshot in &grads {
            assert_eq!(snapshot.len(), 10);
        }
    }

    #[test]
    fn test_simulate_task_gradients_non_zero() {
        let grads = simulate_task_gradients(4, 3, 1.0, 7);
        let all_zero = grads.iter().all(|s| s.iter().all(|&v| v.abs() < 1e-9));
        assert!(!all_zero);
    }

    #[test]
    fn test_simulate_task_gradients_empty_params_returns_empty() {
        let grads = simulate_task_gradients(0, 5, 1.0, 1);
        assert!(grads.is_empty());
    }

    // ------------------------------------------------------------------
    // replay_loss
    // ------------------------------------------------------------------
    #[test]
    fn test_replay_loss_weight_zero_only_current() {
        let l = replay_loss(2.0, &[10.0, 20.0], 0.0);
        assert!(close(l, 2.0, 1e-6));
    }

    #[test]
    fn test_replay_loss_weight_one_only_replay_mean() {
        let l = replay_loss(99.0, &[4.0, 6.0], 1.0);
        // mean replay = 5.0
        assert!(close(l, 5.0, 1e-6));
    }

    #[test]
    fn test_replay_loss_empty_replay_returns_current() {
        let l = replay_loss(3.0, &[], 0.5);
        assert!(close(l, 3.0, 1e-6));
    }

    #[test]
    fn test_replay_loss_mixed() {
        // weight=0.5: 0.5*4 + 0.5*mean([2,4]) = 2 + 1.5 = 3.5  wait:
        // mean([2,4]) = 3 → 0.5*4 + 0.5*3 = 2 + 1.5 = 3.5
        let l = replay_loss(4.0, &[2.0, 4.0], 0.5);
        assert!(close(l, 3.5, 1e-6));
    }

    // ------------------------------------------------------------------
    // compute_cl_stats
    // ------------------------------------------------------------------
    #[test]
    fn test_compute_cl_stats_valid() {
        let mut reg = EwcRegularizer::new(EwcConfig::default()).expect("ok");
        reg.register_task(0, vec![1.0, 1.0], vec![1.0, 1.0])
            .expect("register");

        let mut replay = ReplayBuffer::new(10);
        replay.push(vec![1.0], vec![0.0]);
        replay.push(vec![2.0], vec![1.0]);

        let mask = TaskMask::new_all_active(1, 2);

        let stats = compute_cl_stats(
            &reg,
            &[1.0, 1.0],
            &[(0.9_f32, 0.8_f32)],
            &replay,
            Some(&mask),
        )
        .expect("stats");

        assert_eq!(stats.n_tasks, 1);
        assert!(close(stats.ewc_penalty, 0.0, 1e-5)); // at optimal
        assert_eq!(stats.replay_buffer_size, 2);
        assert!(close(stats.active_param_fraction, 1.0, 1e-6));
        // mean forgetting = 0.9 - 0.8 = 0.1
        assert!(close(stats.mean_forgetting, 0.1, 1e-5));
    }

    #[test]
    fn test_compute_cl_stats_no_mask() {
        let reg = EwcRegularizer::new(EwcConfig::default()).expect("ok");
        let replay = ReplayBuffer::new(5);
        let stats = compute_cl_stats(&reg, &[1.0], &[], &replay, None).expect("stats");
        assert!(close(stats.active_param_fraction, 1.0, 1e-6));
    }
}
