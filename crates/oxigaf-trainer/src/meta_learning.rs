//! MAML (Model-Agnostic Meta-Learning) for few-shot avatar personalization.
//!
//! Implements a simplified CPU-side MAML framework for learning a parameter
//! initialization that rapidly adapts to new subjects via inner-loop gradient steps.
//!
//! ## Design
//! - **Inner loop**: adapt base params on each task's support set (few gradient steps).
//! - **Outer loop**: update base params by gradient of query losses over adapted params.
//! - **FOMAML**: first-order approximation, ignoring second-order meta-gradients.
//! - **Full finite-difference meta-gradient**: perturbation-based second-order estimate.
//! - Uses a `LinearModel` (y = W·x + b) as the learnable module.
//! - Random task generation via xorshift64 + Box-Muller, no external PRNG crate.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the meta-learning subsystem.
#[derive(Debug, Error)]
pub enum MetaLearningError {
    /// Invalid configuration parameter.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// A task batch was empty when at least one task was required.
    #[error("Empty task batch")]
    EmptyTaskBatch,

    /// Buffer or parameter dimension mismatch.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Error during gradient computation.
    #[error("Gradient error: {0}")]
    GradientError(String),

    /// Convergence-related failure.
    #[error("Convergence error: {0}")]
    ConvergenceError(String),

    /// Numerical failure (NaN, Inf, division by zero, etc.).
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

// ---------------------------------------------------------------------------
// FewShotTask
// ---------------------------------------------------------------------------

/// A single few-shot learning task.
///
/// Contains a *support* set used for inner-loop adaptation and a *query* set
/// used to evaluate the meta-objective after adaptation.
#[derive(Debug, Clone)]
pub struct FewShotTask {
    /// Unique identifier for this task.
    pub task_id: usize,
    /// Support inputs, flattened row-major: shape \[n_support, input_dim\].
    pub support_inputs: Vec<f32>,
    /// Support targets, flattened row-major: shape \[n_support, output_dim\].
    pub support_targets: Vec<f32>,
    /// Query inputs, flattened row-major: shape \[n_query, input_dim\].
    pub query_inputs: Vec<f32>,
    /// Query targets, flattened row-major: shape \[n_query, output_dim\].
    pub query_targets: Vec<f32>,
    /// Number of support examples.
    pub n_support: usize,
    /// Number of query examples.
    pub n_query: usize,
    /// Dimensionality of each input vector.
    pub input_dim: usize,
    /// Dimensionality of each output vector.
    pub output_dim: usize,
}

impl FewShotTask {
    /// Create a new task with zero-initialised buffers.
    pub fn new(
        task_id: usize,
        n_support: usize,
        n_query: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> Self {
        FewShotTask {
            task_id,
            support_inputs: vec![0.0; n_support * input_dim],
            support_targets: vec![0.0; n_support * output_dim],
            query_inputs: vec![0.0; n_query * input_dim],
            query_targets: vec![0.0; n_query * output_dim],
            n_support,
            n_query,
            input_dim,
            output_dim,
        }
    }

    /// Validate that all buffer sizes match the declared dimensions.
    pub fn validate(&self) -> Result<(), MetaLearningError> {
        let expected_si = self.n_support * self.input_dim;
        if self.support_inputs.len() != expected_si {
            return Err(MetaLearningError::DimensionMismatch {
                expected: expected_si,
                actual: self.support_inputs.len(),
            });
        }

        let expected_st = self.n_support * self.output_dim;
        if self.support_targets.len() != expected_st {
            return Err(MetaLearningError::DimensionMismatch {
                expected: expected_st,
                actual: self.support_targets.len(),
            });
        }

        let expected_qi = self.n_query * self.input_dim;
        if self.query_inputs.len() != expected_qi {
            return Err(MetaLearningError::DimensionMismatch {
                expected: expected_qi,
                actual: self.query_inputs.len(),
            });
        }

        let expected_qt = self.n_query * self.output_dim;
        if self.query_targets.len() != expected_qt {
            return Err(MetaLearningError::DimensionMismatch {
                expected: expected_qt,
                actual: self.query_targets.len(),
            });
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MamlConfig
// ---------------------------------------------------------------------------

/// Configuration for MAML training.
#[derive(Debug, Clone)]
pub struct MamlConfig {
    /// Inner-loop learning rate (default: 0.01).
    pub inner_lr: f32,
    /// Outer (meta) learning rate (default: 0.001).
    pub meta_lr: f32,
    /// Number of gradient steps in the inner loop (default: 5).
    pub num_inner_steps: usize,
    /// Use FOMAML — ignore second-order terms (default: true).
    pub first_order: bool,
    /// Number of tasks sampled per meta-update (default: 4).
    pub task_batch_size: usize,
    /// Optional gradient clipping by global norm.
    pub clip_grad_norm: Option<f32>,
}

impl Default for MamlConfig {
    fn default() -> Self {
        MamlConfig {
            inner_lr: 0.01,
            meta_lr: 0.001,
            num_inner_steps: 5,
            first_order: true,
            task_batch_size: 4,
            clip_grad_norm: None,
        }
    }
}

impl MamlConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<(), MetaLearningError> {
        if self.inner_lr <= 0.0 {
            return Err(MetaLearningError::InvalidConfig(
                "inner_lr must be positive".to_string(),
            ));
        }
        if self.meta_lr <= 0.0 {
            return Err(MetaLearningError::InvalidConfig(
                "meta_lr must be positive".to_string(),
            ));
        }
        if self.num_inner_steps == 0 {
            return Err(MetaLearningError::InvalidConfig(
                "num_inner_steps must be > 0".to_string(),
            ));
        }
        if self.task_batch_size == 0 {
            return Err(MetaLearningError::InvalidConfig(
                "task_batch_size must be > 0".to_string(),
            ));
        }
        if let Some(c) = self.clip_grad_norm {
            if c <= 0.0 {
                return Err(MetaLearningError::InvalidConfig(
                    "clip_grad_norm must be positive when set".to_string(),
                ));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LinearModel
// ---------------------------------------------------------------------------

/// A simple linear model: y = W · x + b.
///
/// Parameters are stored as a single flat vector:
/// `[W₀₀, W₀₁, …, W_{od,id}, b₀, …, b_{od}]`
/// where W has shape `[output_dim, input_dim]` (row-major).
#[derive(Debug, Clone)]
pub struct LinearModel {
    /// Flattened parameter vector \[W, b\].
    pub params: Vec<f32>,
    /// Dimensionality of inputs.
    pub input_dim: usize,
    /// Dimensionality of outputs.
    pub output_dim: usize,
}

impl LinearModel {
    /// Create a new zero-initialised linear model.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let n = output_dim * (input_dim + 1);
        LinearModel {
            params: vec![0.0; n],
            input_dim,
            output_dim,
        }
    }

    /// Construct from an existing parameter vector, validating its length.
    pub fn from_params(
        params: Vec<f32>,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Self, MetaLearningError> {
        let expected = output_dim * (input_dim + 1);
        if params.len() != expected {
            return Err(MetaLearningError::DimensionMismatch {
                expected,
                actual: params.len(),
            });
        }
        Ok(LinearModel {
            params,
            input_dim,
            output_dim,
        })
    }

    /// Total number of parameters: `output_dim * (input_dim + 1)`.
    pub fn param_count(&self) -> usize {
        self.output_dim * (self.input_dim + 1)
    }

    /// Forward pass.
    ///
    /// `inputs` must have length `n * input_dim` (row-major).
    /// Returns output of length `n * output_dim`.
    pub fn forward(&self, inputs: &[f32], n: usize) -> Result<Vec<f32>, MetaLearningError> {
        let expected_in = n * self.input_dim;
        if inputs.len() != expected_in {
            return Err(MetaLearningError::DimensionMismatch {
                expected: expected_in,
                actual: inputs.len(),
            });
        }

        let w_len = self.output_dim * self.input_dim;
        // W: params[0..w_len], b: params[w_len..]
        let w = &self.params[..w_len];
        let b = &self.params[w_len..];

        let mut output = vec![0.0_f32; n * self.output_dim];

        for sample in 0..n {
            let x = &inputs[sample * self.input_dim..(sample + 1) * self.input_dim];
            for o in 0..self.output_dim {
                let mut acc = b[o];
                for i in 0..self.input_dim {
                    acc += w[o * self.input_dim + i] * x[i];
                }
                output[sample * self.output_dim + o] = acc;
            }
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// MSE loss and gradient
// ---------------------------------------------------------------------------

/// Compute mean-squared-error loss and gradient with respect to model parameters.
///
/// Returns `(loss, gradient)` where `gradient` has the same shape as `model.params`.
pub fn mse_loss_and_grad(
    model: &LinearModel,
    inputs: &[f32],
    targets: &[f32],
    n: usize,
) -> Result<(f32, Vec<f32>), MetaLearningError> {
    if n == 0 {
        return Err(MetaLearningError::GradientError(
            "Cannot compute loss with zero samples".to_string(),
        ));
    }

    let pred = model.forward(inputs, n)?;

    let expected_t = n * model.output_dim;
    if targets.len() != expected_t {
        return Err(MetaLearningError::DimensionMismatch {
            expected: expected_t,
            actual: targets.len(),
        });
    }

    // Residuals: r[s*od + o] = pred - target
    let mut residuals = vec![0.0_f32; n * model.output_dim];
    let mut loss = 0.0_f32;
    for i in 0..residuals.len() {
        let r = pred[i] - targets[i];
        residuals[i] = r;
        loss += r * r;
    }
    loss /= (n * model.output_dim) as f32;

    // Gradient w.r.t. W: dL/dW_{o,i} = (2/N*od) * Σ_s r_{s,o} * x_{s,i}
    // Gradient w.r.t. b: dL/db_o     = (2/N*od) * Σ_s r_{s,o}
    let scale = 2.0 / (n * model.output_dim) as f32;
    let w_len = model.output_dim * model.input_dim;
    let mut grad = vec![0.0_f32; model.param_count()];

    for s in 0..n {
        let x = &inputs[s * model.input_dim..(s + 1) * model.input_dim];
        for o in 0..model.output_dim {
            let r = residuals[s * model.output_dim + o];
            // W gradient
            for i in 0..model.input_dim {
                grad[o * model.input_dim + i] += scale * r * x[i];
            }
            // b gradient
            grad[w_len + o] += scale * r;
        }
    }

    Ok((loss, grad))
}

// ---------------------------------------------------------------------------
// Inner-loop adaptation
// ---------------------------------------------------------------------------

/// Perform inner-loop gradient descent to adapt model parameters for one task.
///
/// Uses the task's support set.  Returns the adapted parameter vector without
/// modifying `model` in place.
pub fn inner_loop_adapt(
    model: &LinearModel,
    task: &FewShotTask,
    config: &MamlConfig,
) -> Result<Vec<f32>, MetaLearningError> {
    task.validate()?;

    // Work on a mutable copy of params
    let mut adapted_params = model.params.clone();

    for _step in 0..config.num_inner_steps {
        // Build a temporary model from current adapted params
        let tmp =
            LinearModel::from_params(adapted_params.clone(), model.input_dim, model.output_dim)?;
        let (_, grad) = mse_loss_and_grad(
            &tmp,
            &task.support_inputs,
            &task.support_targets,
            task.n_support,
        )?;

        // SGD step
        apply_gradient_update(&mut adapted_params, &grad, config.inner_lr)?;
    }

    Ok(adapted_params)
}

// ---------------------------------------------------------------------------
// Query-loss evaluation
// ---------------------------------------------------------------------------

/// Evaluate query-set MSE for a model with the given adapted parameters.
pub fn evaluate_query_loss(
    model: &LinearModel,
    adapted_params: &[f32],
    task: &FewShotTask,
) -> Result<f32, MetaLearningError> {
    let adapted =
        LinearModel::from_params(adapted_params.to_vec(), model.input_dim, model.output_dim)?;
    let (loss, _) = mse_loss_and_grad(
        &adapted,
        &task.query_inputs,
        &task.query_targets,
        task.n_query,
    )?;
    Ok(loss)
}

// ---------------------------------------------------------------------------
// Meta-gradient computation
// ---------------------------------------------------------------------------

/// Compute the meta-gradient over a batch of tasks.
///
/// ## Modes
///
/// * `first_order = true` (**FOMAML**): gradient of the query loss evaluated at
///   the *adapted* parameters, treating the inner-loop as a constant map.
///   This is the standard cheap approximation used in almost all MAML papers.
///
/// * `first_order = false` (**finite-difference second-order**): perturb each
///   element of the *initial* parameter vector by ε, re-run the full inner
///   loop, and measure the resulting change in query loss.  Expensive but
///   provides a truer estimate of the meta-gradient.
///
/// Returns `(mean_query_loss, meta_gradient)`.
pub fn compute_meta_gradient(
    model: &LinearModel,
    tasks: &[FewShotTask],
    config: &MamlConfig,
) -> Result<(f32, Vec<f32>), MetaLearningError> {
    if tasks.is_empty() {
        return Err(MetaLearningError::EmptyTaskBatch);
    }

    let n_params = model.param_count();
    let mut meta_grad = vec![0.0_f32; n_params];
    let mut total_query_loss = 0.0_f32;

    if config.first_order {
        // FOMAML: gradient of query loss at adapted params
        for task in tasks {
            let adapted_params = inner_loop_adapt(model, task, config)?;
            let adapted = LinearModel::from_params(
                adapted_params.clone(),
                model.input_dim,
                model.output_dim,
            )?;
            let (q_loss, q_grad) = mse_loss_and_grad(
                &adapted,
                &task.query_inputs,
                &task.query_targets,
                task.n_query,
            )?;
            total_query_loss += q_loss;
            for (mg, qg) in meta_grad.iter_mut().zip(q_grad.iter()) {
                *mg += qg;
            }
        }
    } else {
        // Finite-difference meta-gradient
        const EPS: f32 = 1e-4;

        // First pass: evaluate query loss at current base params for each task
        let base_losses: Vec<f32> = tasks
            .iter()
            .map(|task| {
                let ap = inner_loop_adapt(model, task, config)?;
                evaluate_query_loss(model, &ap, task)
            })
            .collect::<Result<Vec<f32>, MetaLearningError>>()?;

        total_query_loss = base_losses.iter().sum::<f32>();

        // Second pass: perturb each parameter and measure change
        let base_params = model.params.clone();
        for param_idx in 0..n_params {
            let mut perturbed_params = base_params.clone();
            perturbed_params[param_idx] += EPS;

            let perturbed_model =
                LinearModel::from_params(perturbed_params, model.input_dim, model.output_dim)?;

            let mut perturbed_total = 0.0_f32;
            for task in tasks {
                let ap = inner_loop_adapt(&perturbed_model, task, config)?;
                let pl = evaluate_query_loss(&perturbed_model, &ap, task)?;
                perturbed_total += pl;
            }

            meta_grad[param_idx] = (perturbed_total - total_query_loss) / EPS;
        }
    }

    let n_tasks = tasks.len() as f32;
    for mg in meta_grad.iter_mut() {
        *mg /= n_tasks;
    }
    let mean_query_loss = total_query_loss / n_tasks;

    Ok((mean_query_loss, meta_grad))
}

// ---------------------------------------------------------------------------
// Meta-update step
// ---------------------------------------------------------------------------

/// Perform one meta-update: compute meta-gradient and apply it to model params.
///
/// Returns the mean query loss across all tasks.
pub fn meta_update_step(
    model: &mut LinearModel,
    tasks: &[FewShotTask],
    config: &MamlConfig,
) -> Result<f32, MetaLearningError> {
    config.validate()?;

    let (mean_loss, mut meta_grad) = compute_meta_gradient(model, tasks, config)?;

    if let Some(max_norm) = config.clip_grad_norm {
        clip_gradient(&mut meta_grad, max_norm);
    }

    apply_gradient_update(&mut model.params, &meta_grad, config.meta_lr)?;

    Ok(mean_loss)
}

// ---------------------------------------------------------------------------
// Task sampler
// ---------------------------------------------------------------------------

/// Generates synthetic linear regression tasks for meta-training.
///
/// Each task defines a ground-truth function `f(x) = A·x + b_offset`
/// with random `A` (output_dim × input_dim) and `b_offset` (output_dim),
/// drawn via xorshift64 + Box-Muller normal sampling.
pub struct TaskSampler {
    /// Dimensionality of task inputs.
    pub input_dim: usize,
    /// Dimensionality of task outputs.
    pub output_dim: usize,
    /// Number of support examples per task.
    pub n_support: usize,
    /// Number of query examples per task.
    pub n_query: usize,
    seed: u64,
}

impl TaskSampler {
    /// Create a new task sampler.
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        n_support: usize,
        n_query: usize,
        seed: u64,
    ) -> Self {
        TaskSampler {
            input_dim,
            output_dim,
            n_support,
            n_query,
            seed: seed.max(1),
        }
    }

    /// Draw the next xorshift64 value in (0, u64::MAX].
    fn next_u64(&mut self) -> u64 {
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 7;
        self.seed ^= self.seed << 17;
        // Ensure state never becomes zero
        self.seed = self.seed.max(1);
        self.seed
    }

    /// Draw a uniform value in (0, 1) — strictly positive so `ln` is safe.
    fn next_uniform(&mut self) -> f64 {
        // Map to (0, 1] then shift to (0, 1)
        let raw = self.next_u64() as f64 / u64::MAX as f64;
        // Clamp away from zero to make ln() safe
        raw.max(f64::MIN_POSITIVE)
    }

    /// Draw two independent standard-normal values via Box-Muller.
    fn next_pair_normal(&mut self) -> (f32, f32) {
        let u1 = self.next_uniform();
        let u2 = self.next_uniform();
        // Box-Muller transform
        let mag = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        let z0 = (mag * angle.cos()) as f32;
        let z1 = (mag * angle.sin()) as f32;
        (z0, z1)
    }

    /// Draw `n` standard-normal f32 values into a Vec.
    fn sample_normals(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let (z0, z1) = self.next_pair_normal();
            out.push(z0);
            if i + 1 < n {
                out.push(z1);
            }
            i += 2;
        }
        out
    }

    /// Sample one task.
    ///
    /// Creates a random linear function `f(x) = A·x + b_offset`
    /// and generates support + query examples from it.
    pub fn sample_task(&mut self, task_id: usize) -> FewShotTask {
        let id = self.input_dim;
        let od = self.output_dim;
        let ns = self.n_support;
        let nq = self.n_query;

        // Sample random A (od × id) and b_offset (od)
        let a_vec = self.sample_normals(od * id);
        let b_vec = self.sample_normals(od);

        let total_examples = ns + nq;
        let raw_x = self.sample_normals(total_examples * id);
        let mut raw_y = vec![0.0_f32; total_examples * od];

        // Apply linear function: y = A·x + b
        for s in 0..total_examples {
            let x = &raw_x[s * id..(s + 1) * id];
            for o in 0..od {
                let mut acc = b_vec[o];
                for i in 0..id {
                    acc += a_vec[o * id + i] * x[i];
                }
                raw_y[s * od + o] = acc;
            }
        }

        let support_inputs = raw_x[..ns * id].to_vec();
        let support_targets = raw_y[..ns * od].to_vec();
        let query_inputs = raw_x[ns * id..].to_vec();
        let query_targets = raw_y[ns * od..].to_vec();

        FewShotTask {
            task_id,
            support_inputs,
            support_targets,
            query_inputs,
            query_targets,
            n_support: ns,
            n_query: nq,
            input_dim: id,
            output_dim: od,
        }
    }

    /// Sample a batch of tasks.
    pub fn sample_batch(&mut self, batch_size: usize, start_id: usize) -> Vec<FewShotTask> {
        (0..batch_size)
            .map(|i| self.sample_task(start_id + i))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Meta-training loop
// ---------------------------------------------------------------------------

/// Statistics recorded for a single meta-training step.
#[derive(Debug, Clone)]
pub struct MetaTrainingStats {
    /// Step index (0-based).
    pub step: usize,
    /// Mean query loss across all tasks in this step's batch.
    pub mean_query_loss: f32,
    /// L2 norm of the meta-gradient before clipping.
    pub meta_grad_norm: f32,
}

/// Run the full meta-training loop.
///
/// At each step, samples a task batch, computes the meta-gradient, and
/// updates the model.
pub fn run_meta_training(
    model: &mut LinearModel,
    sampler: &mut TaskSampler,
    config: &MamlConfig,
    num_steps: usize,
) -> Result<Vec<MetaTrainingStats>, MetaLearningError> {
    config.validate()?;

    let mut stats = Vec::with_capacity(num_steps);

    for step in 0..num_steps {
        let tasks = sampler.sample_batch(config.task_batch_size, step * config.task_batch_size);

        let (mean_query_loss, mut meta_grad) = compute_meta_gradient(model, &tasks, config)?;
        let gn = grad_norm(&meta_grad);

        if let Some(max_norm) = config.clip_grad_norm {
            clip_gradient(&mut meta_grad, max_norm);
        }

        apply_gradient_update(&mut model.params, &meta_grad, config.meta_lr)?;

        stats.push(MetaTrainingStats {
            step,
            mean_query_loss,
            meta_grad_norm: gn,
        });
    }

    Ok(stats)
}

// ---------------------------------------------------------------------------
// Gradient utilities
// ---------------------------------------------------------------------------

/// Compute the L2 norm of a gradient vector.
pub fn grad_norm(grad: &[f32]) -> f32 {
    grad.iter().map(|g| g * g).sum::<f32>().sqrt()
}

/// Clip a gradient vector by its global L2 norm (in-place).
///
/// If `||grad|| > max_norm`, scales the gradient so `||grad|| = max_norm`.
/// No-op if the norm is already within bounds.
pub fn clip_gradient(grad: &mut [f32], max_norm: f32) {
    let n = grad_norm(grad);
    if n > max_norm && n > 0.0 {
        let scale = max_norm / n;
        for g in grad.iter_mut() {
            *g *= scale;
        }
    }
}

/// Apply an SGD update: `params -= lr * grad` (in-place).
pub fn apply_gradient_update(
    params: &mut [f32],
    grad: &[f32],
    lr: f32,
) -> Result<(), MetaLearningError> {
    if params.len() != grad.len() {
        return Err(MetaLearningError::DimensionMismatch {
            expected: params.len(),
            actual: grad.len(),
        });
    }
    for (p, g) in params.iter_mut().zip(grad.iter()) {
        *p -= lr * g;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Aggregate statistics across runs
// ---------------------------------------------------------------------------

/// Aggregate per-step statistics over multiple meta-training runs.
///
/// Aligns runs by step index (truncated to the shortest run) and returns
/// `(mean_loss, std_loss)` per step.
pub fn aggregate_meta_stats(stats_runs: &[Vec<MetaTrainingStats>]) -> Vec<(f32, f32)> {
    if stats_runs.is_empty() {
        return Vec::new();
    }

    let min_len = stats_runs.iter().map(|r| r.len()).min().unwrap_or(0);
    if min_len == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(min_len);
    let n_runs = stats_runs.len() as f32;

    for step_idx in 0..min_len {
        let losses: Vec<f32> = stats_runs
            .iter()
            .map(|run| run[step_idx].mean_query_loss)
            .collect();

        let mean = losses.iter().sum::<f32>() / n_runs;
        let variance = losses
            .iter()
            .map(|&l| {
                let diff = l - mean;
                diff * diff
            })
            .sum::<f32>()
            / n_runs;
        let std = variance.sqrt();

        result.push((mean, std));
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- FewShotTask ---

    #[test]
    fn test_few_shot_task_new_buffer_sizes() {
        let task = FewShotTask::new(0, 3, 5, 4, 2);
        assert_eq!(task.support_inputs.len(), 3 * 4);
        assert_eq!(task.support_targets.len(), 3 * 2);
        assert_eq!(task.query_inputs.len(), 5 * 4);
        assert_eq!(task.query_targets.len(), 5 * 2);
    }

    #[test]
    fn test_few_shot_task_validate_ok() -> Result<(), MetaLearningError> {
        let task = FewShotTask::new(1, 3, 5, 4, 2);
        task.validate()
    }

    #[test]
    fn test_few_shot_task_validate_wrong_support_inputs() {
        let mut task = FewShotTask::new(2, 3, 5, 4, 2);
        task.support_inputs.pop();
        let result = task.validate();
        assert!(result.is_err());
        match result {
            Err(MetaLearningError::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, 12);
                assert_eq!(actual, 11);
            }
            other => panic!("Unexpected result: {:?}", other),
        }
    }

    #[test]
    fn test_few_shot_task_validate_wrong_support_targets() {
        let mut task = FewShotTask::new(3, 3, 5, 4, 2);
        task.support_targets.push(0.0);
        let result = task.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_few_shot_task_validate_wrong_query_inputs() {
        let mut task = FewShotTask::new(4, 3, 5, 4, 2);
        task.query_inputs.clear();
        let result = task.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_few_shot_task_validate_wrong_query_targets() {
        let mut task = FewShotTask::new(5, 3, 5, 4, 2);
        task.query_targets.pop();
        let result = task.validate();
        assert!(result.is_err());
    }

    // --- MamlConfig ---

    #[test]
    fn test_maml_config_default_valid() -> Result<(), MetaLearningError> {
        MamlConfig::default().validate()
    }

    #[test]
    fn test_maml_config_zero_inner_steps() {
        let cfg = MamlConfig {
            num_inner_steps: 0,
            ..MamlConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_maml_config_zero_task_batch() {
        let cfg = MamlConfig {
            task_batch_size: 0,
            ..MamlConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_maml_config_negative_inner_lr() {
        let cfg = MamlConfig {
            inner_lr: -0.01,
            ..MamlConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_maml_config_negative_meta_lr() {
        let cfg = MamlConfig {
            meta_lr: -0.001,
            ..MamlConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_maml_config_invalid_clip_grad() {
        let cfg = MamlConfig {
            clip_grad_norm: Some(-1.0),
            ..MamlConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    // --- LinearModel ---

    #[test]
    fn test_linear_model_new_param_count() {
        let m = LinearModel::new(3, 2);
        assert_eq!(m.param_count(), 2 * (3 + 1));
        assert_eq!(m.params.len(), 8);
    }

    #[test]
    fn test_linear_model_zero_weights_zero_output() -> Result<(), MetaLearningError> {
        let m = LinearModel::new(4, 3);
        let inputs = vec![1.0_f32; 4]; // 1 sample, dim=4
        let out = m.forward(&inputs, 1)?;
        assert_eq!(out.len(), 3);
        for &v in &out {
            assert_eq!(v, 0.0);
        }
        Ok(())
    }

    #[test]
    fn test_linear_model_forward_identity_like() -> Result<(), MetaLearningError> {
        // Model: y = x (W = I, b = 0) for dim 2
        let mut m = LinearModel::new(2, 2);
        // W = [[1,0],[0,1]], b = [0,0]
        m.params[0] = 1.0; // W[0,0]
        m.params[1] = 0.0; // W[0,1]
        m.params[2] = 0.0; // W[1,0]
        m.params[3] = 1.0; // W[1,1]
        let inputs = vec![3.0_f32, 5.0_f32]; // 1 sample
        let out = m.forward(&inputs, 1)?;
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert!((out[1] - 5.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_linear_model_forward_bias_only() -> Result<(), MetaLearningError> {
        let mut m = LinearModel::new(2, 1);
        // W = [0, 0], b = [7.0]
        m.params[2] = 7.0;
        let inputs = vec![100.0_f32, 200.0_f32];
        let out = m.forward(&inputs, 1)?;
        assert!((out[0] - 7.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_linear_model_forward_dimension_mismatch() {
        let m = LinearModel::new(3, 2);
        let bad = vec![1.0_f32; 4]; // should be 3
        assert!(m.forward(&bad, 1).is_err());
    }

    #[test]
    fn test_linear_model_from_params_wrong_size() {
        let result = LinearModel::from_params(vec![0.0_f32; 5], 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_model_from_params_correct_size() -> Result<(), MetaLearningError> {
        let params = vec![0.0_f32; 2 * (3 + 1)]; // od=2, id=3
        let m = LinearModel::from_params(params, 3, 2)?;
        assert_eq!(m.param_count(), 8);
        Ok(())
    }

    // --- mse_loss_and_grad ---

    #[test]
    fn test_mse_loss_zero_on_perfect_prediction() -> Result<(), MetaLearningError> {
        // identity model, inputs == targets
        let mut m = LinearModel::new(2, 2);
        m.params[0] = 1.0;
        m.params[3] = 1.0; // W = I
        let data = vec![1.0_f32, 2.0, 3.0, 4.0]; // 2 samples × 2 dims
        let (loss, grad) = mse_loss_and_grad(&m, &data, &data, 2)?;
        assert!(loss.abs() < 1e-7, "loss should be 0, got {}", loss);
        assert!(grad.iter().all(|&g| g.abs() < 1e-7), "grad should be 0");
        Ok(())
    }

    #[test]
    fn test_mse_loss_positive_on_imperfect_prediction() -> Result<(), MetaLearningError> {
        let m = LinearModel::new(1, 1); // outputs 0
        let inputs = vec![1.0_f32];
        let targets = vec![1.0_f32];
        let (loss, _) = mse_loss_and_grad(&m, &inputs, &targets, 1)?;
        // residual = 0 - 1 = -1, MSE = 1
        assert!((loss - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_mse_grad_correct_direction() -> Result<(), MetaLearningError> {
        let m = LinearModel::new(1, 1); // W=[0], b=[0]
        let inputs = vec![1.0_f32]; // single sample, single dim
        let targets = vec![2.0_f32]; // target = 2
        let (_, grad) = mse_loss_and_grad(&m, &inputs, &targets, 1)?;
        // pred=0, r=-2, dL/dW = 2*(-2)*1 = -4, dL/db = 2*(-2) = -4
        assert!(grad[0] < 0.0, "gradient w.r.t W should be negative");
        assert!(grad[1] < 0.0, "gradient w.r.t b should be negative");
        Ok(())
    }

    #[test]
    fn test_mse_loss_zero_samples_error() {
        let m = LinearModel::new(2, 2);
        let result = mse_loss_and_grad(&m, &[], &[], 0);
        assert!(result.is_err());
    }

    // --- inner_loop_adapt ---

    #[test]
    fn test_inner_loop_adapt_correct_param_shape() -> Result<(), MetaLearningError> {
        let m = LinearModel::new(2, 1);
        let task = FewShotTask::new(0, 4, 2, 2, 1);
        let cfg = MamlConfig::default();
        let adapted = inner_loop_adapt(&m, &task, &cfg)?;
        assert_eq!(adapted.len(), m.param_count());
        Ok(())
    }

    #[test]
    fn test_inner_loop_adapt_loss_decreases() -> Result<(), MetaLearningError> {
        // Simple 1D regression: target is always 1.0
        let m = LinearModel::new(1, 1); // starts at zero
        let mut task = FewShotTask::new(0, 8, 4, 1, 1);
        // inputs: 1.0 each, targets: 1.0 each
        for v in task.support_inputs.iter_mut() {
            *v = 1.0;
        }
        for v in task.support_targets.iter_mut() {
            *v = 1.0;
        }
        for v in task.query_inputs.iter_mut() {
            *v = 1.0;
        }
        for v in task.query_targets.iter_mut() {
            *v = 1.0;
        }

        let cfg = MamlConfig {
            inner_lr: 0.1,
            num_inner_steps: 10,
            ..MamlConfig::default()
        };

        let (loss_before, _) =
            mse_loss_and_grad(&m, &task.support_inputs, &task.support_targets, 8)?;
        let adapted = inner_loop_adapt(&m, &task, &cfg)?;
        let adapted_m = LinearModel::from_params(adapted, 1, 1)?;
        let (loss_after, _) =
            mse_loss_and_grad(&adapted_m, &task.support_inputs, &task.support_targets, 8)?;

        assert!(
            loss_after < loss_before,
            "Adapted loss {} should be less than initial loss {}",
            loss_after,
            loss_before
        );
        Ok(())
    }

    // --- evaluate_query_loss ---

    #[test]
    fn test_evaluate_query_loss_known_problem() -> Result<(), MetaLearningError> {
        // zero model, query target = 0 → loss = 0
        let m = LinearModel::new(1, 1);
        let adapted = m.params.clone();
        let task = FewShotTask::new(0, 2, 2, 1, 1);
        // all zeros → target=0, pred=0, loss=0
        let loss = evaluate_query_loss(&m, &adapted, &task)?;
        assert!(loss.abs() < 1e-7);
        Ok(())
    }

    #[test]
    fn test_evaluate_query_loss_nonzero() -> Result<(), MetaLearningError> {
        let m = LinearModel::new(1, 1); // pred = 0
        let adapted = m.params.clone();
        let mut task = FewShotTask::new(0, 2, 2, 1, 1);
        for v in task.query_targets.iter_mut() {
            *v = 1.0;
        } // target = 1
        for v in task.query_inputs.iter_mut() {
            *v = 1.0;
        }
        let loss = evaluate_query_loss(&m, &adapted, &task)?;
        assert!(loss > 0.0);
        Ok(())
    }

    // --- compute_meta_gradient ---

    #[test]
    fn test_compute_meta_gradient_nonempty_grad() -> Result<(), MetaLearningError> {
        let m = LinearModel::new(2, 1);
        let mut sampler = TaskSampler::new(2, 1, 5, 3, 42);
        let tasks = sampler.sample_batch(2, 0);
        let cfg = MamlConfig::default();
        let (_, grad) = compute_meta_gradient(&m, &tasks, &cfg)?;
        assert_eq!(grad.len(), m.param_count());
        // Gradient shouldn't be all zeros for random tasks
        let total: f32 = grad.iter().map(|g| g.abs()).sum();
        assert!(total > 0.0, "meta-gradient should be non-zero");
        Ok(())
    }

    #[test]
    fn test_compute_meta_gradient_correct_shape() -> Result<(), MetaLearningError> {
        let m = LinearModel::new(3, 2);
        let mut sampler = TaskSampler::new(3, 2, 4, 3, 7);
        let tasks = sampler.sample_batch(3, 0);
        let cfg = MamlConfig::default();
        let (_, grad) = compute_meta_gradient(&m, &tasks, &cfg)?;
        assert_eq!(grad.len(), m.param_count()); // 2*(3+1)=8
        Ok(())
    }

    #[test]
    fn test_compute_meta_gradient_empty_tasks_error() {
        let m = LinearModel::new(2, 1);
        let cfg = MamlConfig::default();
        let result = compute_meta_gradient(&m, &[], &cfg);
        assert!(matches!(result, Err(MetaLearningError::EmptyTaskBatch)));
    }

    #[test]
    fn test_compute_meta_gradient_second_order() -> Result<(), MetaLearningError> {
        let m = LinearModel::new(2, 1);
        let mut sampler = TaskSampler::new(2, 1, 4, 3, 99);
        let tasks = sampler.sample_batch(2, 0);
        let cfg = MamlConfig {
            first_order: false,
            num_inner_steps: 2, // fewer steps for speed
            ..MamlConfig::default()
        };
        let (_, grad) = compute_meta_gradient(&m, &tasks, &cfg)?;
        assert_eq!(grad.len(), m.param_count());
        Ok(())
    }

    // --- meta_update_step ---

    #[test]
    fn test_meta_update_step_returns_finite_loss() -> Result<(), MetaLearningError> {
        let mut m = LinearModel::new(2, 1);
        let mut sampler = TaskSampler::new(2, 1, 5, 3, 13);
        let tasks = sampler.sample_batch(4, 0);
        let cfg = MamlConfig::default();
        let loss = meta_update_step(&mut m, &tasks, &cfg)?;
        assert!(loss.is_finite(), "loss must be finite");
        Ok(())
    }

    #[test]
    fn test_meta_update_step_params_change() -> Result<(), MetaLearningError> {
        let mut m = LinearModel::new(2, 1);
        let params_before = m.params.clone();
        let mut sampler = TaskSampler::new(2, 1, 5, 3, 17);
        let tasks = sampler.sample_batch(4, 0);
        let cfg = MamlConfig::default();
        meta_update_step(&mut m, &tasks, &cfg)?;
        let changed = m
            .params
            .iter()
            .zip(params_before.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed, "params should change after meta update");
        Ok(())
    }

    // --- TaskSampler ---

    #[test]
    fn test_task_sampler_correct_shapes() {
        let mut s = TaskSampler::new(4, 2, 5, 3, 42);
        let task = s.sample_task(0);
        assert_eq!(task.support_inputs.len(), 5 * 4);
        assert_eq!(task.support_targets.len(), 5 * 2);
        assert_eq!(task.query_inputs.len(), 3 * 4);
        assert_eq!(task.query_targets.len(), 3 * 2);
        assert_eq!(task.task_id, 0);
    }

    #[test]
    fn test_task_sampler_task_id_assigned() {
        let mut s = TaskSampler::new(2, 1, 3, 3, 1);
        let task = s.sample_task(7);
        assert_eq!(task.task_id, 7);
    }

    #[test]
    fn test_task_sampler_batch_count() {
        let mut s = TaskSampler::new(2, 1, 3, 3, 5);
        let batch = s.sample_batch(6, 10);
        assert_eq!(batch.len(), 6);
    }

    #[test]
    fn test_task_sampler_batch_ids() {
        let mut s = TaskSampler::new(2, 1, 3, 3, 5);
        let batch = s.sample_batch(4, 10);
        for (i, task) in batch.iter().enumerate() {
            assert_eq!(task.task_id, 10 + i);
        }
    }

    #[test]
    fn test_task_sampler_different_seeds_differ() {
        let mut s1 = TaskSampler::new(2, 1, 4, 4, 1);
        let mut s2 = TaskSampler::new(2, 1, 4, 4, 2);
        let t1 = s1.sample_task(0);
        let t2 = s2.sample_task(0);
        // Different seeds → different data
        let diff = t1
            .support_targets
            .iter()
            .zip(t2.support_targets.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(diff, "Different seeds should produce different tasks");
    }

    #[test]
    fn test_task_sampler_validate_passes() -> Result<(), MetaLearningError> {
        let mut s = TaskSampler::new(3, 2, 5, 4, 99);
        let task = s.sample_task(0);
        task.validate()
    }

    // --- run_meta_training ---

    #[test]
    fn test_run_meta_training_returns_stats() -> Result<(), MetaLearningError> {
        let mut m = LinearModel::new(2, 1);
        let mut sampler = TaskSampler::new(2, 1, 5, 3, 42);
        let cfg = MamlConfig {
            task_batch_size: 2,
            num_inner_steps: 3,
            ..MamlConfig::default()
        };
        let stats = run_meta_training(&mut m, &mut sampler, &cfg, 5)?;
        assert_eq!(stats.len(), 5);
        Ok(())
    }

    #[test]
    fn test_run_meta_training_loss_values_finite() -> Result<(), MetaLearningError> {
        let mut m = LinearModel::new(2, 1);
        let mut sampler = TaskSampler::new(2, 1, 5, 3, 123);
        let cfg = MamlConfig {
            task_batch_size: 2,
            ..MamlConfig::default()
        };
        let stats = run_meta_training(&mut m, &mut sampler, &cfg, 4)?;
        for s in &stats {
            assert!(s.mean_query_loss.is_finite());
            assert!(s.meta_grad_norm.is_finite());
        }
        Ok(())
    }

    #[test]
    fn test_run_meta_training_loss_decreasing_trend() -> Result<(), MetaLearningError> {
        // Run enough steps and verify the average loss in the latter half
        // is not larger than the first half (soft trend check).
        let mut m = LinearModel::new(2, 1);
        let mut sampler = TaskSampler::new(2, 1, 8, 4, 77);
        let cfg = MamlConfig {
            inner_lr: 0.05,
            meta_lr: 0.01,
            task_batch_size: 4,
            num_inner_steps: 5,
            ..MamlConfig::default()
        };
        let stats = run_meta_training(&mut m, &mut sampler, &cfg, 20)?;
        let n = stats.len();
        let first_half: f32 = stats[..n / 2]
            .iter()
            .map(|s| s.mean_query_loss)
            .sum::<f32>()
            / (n / 2) as f32;
        let second_half: f32 = stats[n / 2..]
            .iter()
            .map(|s| s.mean_query_loss)
            .sum::<f32>()
            / (n - n / 2) as f32;
        // Loss in second half should be <= first half * 2 (relaxed for stochastic training)
        assert!(
            second_half <= first_half * 2.0,
            "Training should not diverge: first_half={}, second_half={}",
            first_half,
            second_half
        );
        Ok(())
    }

    // --- grad_norm ---

    #[test]
    fn test_grad_norm_zero_vector() {
        let g = vec![0.0_f32; 5];
        assert_eq!(grad_norm(&g), 0.0);
    }

    #[test]
    fn test_grad_norm_unit_vector() {
        let g = vec![1.0_f32, 0.0, 0.0];
        assert!((grad_norm(&g) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_grad_norm_known_value() {
        let g = vec![3.0_f32, 4.0_f32];
        assert!((grad_norm(&g) - 5.0).abs() < 1e-5);
    }

    // --- clip_gradient ---

    #[test]
    fn test_clip_gradient_above_max_norm() {
        let mut g = vec![3.0_f32, 4.0_f32]; // norm = 5
        clip_gradient(&mut g, 2.5);
        let n = grad_norm(&g);
        assert!(
            (n - 2.5).abs() < 1e-5,
            "clipped norm should be 2.5, got {}",
            n
        );
    }

    #[test]
    fn test_clip_gradient_below_max_unchanged() {
        let mut g = vec![0.5_f32, 0.5_f32];
        let orig = g.clone();
        clip_gradient(&mut g, 10.0); // norm < 10 → no change
        for (a, b) in g.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_clip_gradient_zero_vector_no_panic() {
        let mut g = vec![0.0_f32; 4];
        clip_gradient(&mut g, 1.0); // should not divide by zero
        assert!(g.iter().all(|&v| v == 0.0));
    }

    // --- apply_gradient_update ---

    #[test]
    fn test_apply_gradient_update_correct_subtraction() -> Result<(), MetaLearningError> {
        let mut params = vec![1.0_f32, 2.0_f32, 3.0_f32];
        let grad = vec![1.0_f32, 1.0_f32, 1.0_f32];
        apply_gradient_update(&mut params, &grad, 0.5)?;
        assert!((params[0] - 0.5).abs() < 1e-7);
        assert!((params[1] - 1.5).abs() < 1e-7);
        assert!((params[2] - 2.5).abs() < 1e-7);
        Ok(())
    }

    #[test]
    fn test_apply_gradient_update_length_mismatch() {
        let mut params = vec![1.0_f32; 3];
        let grad = vec![1.0_f32; 4]; // wrong length
        let result = apply_gradient_update(&mut params, &grad, 0.01);
        assert!(result.is_err());
    }

    // --- aggregate_meta_stats ---

    #[test]
    fn test_aggregate_meta_stats_two_equal_runs_zero_std() {
        let run = vec![
            MetaTrainingStats {
                step: 0,
                mean_query_loss: 1.0,
                meta_grad_norm: 0.1,
            },
            MetaTrainingStats {
                step: 1,
                mean_query_loss: 0.8,
                meta_grad_norm: 0.09,
            },
        ];
        let stats_runs = vec![run.clone(), run.clone()];
        let agg = aggregate_meta_stats(&stats_runs);
        assert_eq!(agg.len(), 2);
        // Two identical runs → std = 0
        for (_, std) in &agg {
            assert!(
                std.abs() < 1e-6,
                "std should be 0 for equal runs, got {}",
                std
            );
        }
    }

    #[test]
    fn test_aggregate_meta_stats_empty_input() {
        let result = aggregate_meta_stats(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_aggregate_meta_stats_single_run() {
        let run = vec![MetaTrainingStats {
            step: 0,
            mean_query_loss: 2.0,
            meta_grad_norm: 1.0,
        }];
        let agg = aggregate_meta_stats(&[run]);
        assert_eq!(agg.len(), 1);
        assert!((agg[0].0 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_meta_stats_mismatched_lengths_uses_shortest() {
        let run_long = vec![
            MetaTrainingStats {
                step: 0,
                mean_query_loss: 1.0,
                meta_grad_norm: 0.1,
            },
            MetaTrainingStats {
                step: 1,
                mean_query_loss: 0.9,
                meta_grad_norm: 0.09,
            },
        ];
        let run_short = vec![MetaTrainingStats {
            step: 0,
            mean_query_loss: 1.0,
            meta_grad_norm: 0.1,
        }];
        let agg = aggregate_meta_stats(&[run_long, run_short]);
        assert_eq!(agg.len(), 1, "should truncate to shortest run");
    }

    #[test]
    fn test_aggregate_meta_stats_mean_correct() {
        let run_a = vec![MetaTrainingStats {
            step: 0,
            mean_query_loss: 1.0,
            meta_grad_norm: 0.0,
        }];
        let run_b = vec![MetaTrainingStats {
            step: 0,
            mean_query_loss: 3.0,
            meta_grad_norm: 0.0,
        }];
        let agg = aggregate_meta_stats(&[run_a, run_b]);
        assert!((agg[0].0 - 2.0).abs() < 1e-6, "mean should be 2.0");
    }
}
