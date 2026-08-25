//! Gradient accumulation for simulating large effective batch sizes.
//!
//! Gradient accumulation allows training with large effective batch sizes even
//! when GPU memory limits the per-step physical batch size.  Instead of
//! applying an optimizer update on every forward/backward pass, gradients are
//! accumulated across N micro-batches and the optimizer is updated only once,
//! after N steps.
//!
//! # Quick start
//! ```rust
//! use oxigaf_trainer::gradient_accumulation::{
//!     AccumulationConfig, GradientAccumulator, GradNormalization,
//! };
//!
//! let config = AccumulationConfig {
//!     accumulation_steps: 4,
//!     normalization: GradNormalization::MeanOverSteps,
//!     auto_clear: true,
//! };
//! let mut acc = GradientAccumulator::new(config).unwrap();
//! acc.initialize(&[8, 4]).unwrap();
//!
//! for step in 0..4 {
//!     let grads = vec![vec![1.0_f32; 8], vec![0.5_f32; 4]];
//!     acc.accumulate(&grads, 2).unwrap();
//! }
//! assert!(acc.should_update());
//! let normalized = acc.apply().unwrap();
//! assert_eq!(normalized[0][0], 1.0); // mean over 4 steps of 1.0
//! ```

use std::collections::VecDeque;
use std::fmt::Write as FmtWrite;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// AccumulationError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the gradient accumulation subsystem.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum AccumulationError {
    /// Configuration is invalid (e.g. `accumulation_steps = 0`).
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// The number of parameter groups provided does not match the initialized
    /// count.
    #[error("Length mismatch: expected {expected} parameter groups, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    /// `apply()` or `get_gradients()` was called before enough micro-batches
    /// had been accumulated.
    #[error("Not ready: accumulation step not reached yet")]
    NotReady,

    /// An operation was attempted on an empty gradient set.
    #[error("No gradients available (accumulator not initialized or empty)")]
    EmptyGradients,
}

// ─────────────────────────────────────────────────────────────────────────────
// GradNormalization
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for normalizing accumulated gradients before returning them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GradNormalization {
    /// Divide by `accumulation_steps` — equivalent to computing the mean
    /// gradient over all micro-batches.
    MeanOverSteps,
    /// Return the raw sum of gradients without any normalization.
    SumOnly,
    /// Divide each element by the total number of samples seen across all
    /// micro-batches in the accumulation window.  Requires the caller to pass
    /// the correct `batch_size` to [`GradientAccumulator::accumulate`].
    TotalBatchSize,
}

// ─────────────────────────────────────────────────────────────────────────────
// AccumulationConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for gradient accumulation.
#[derive(Debug, Clone)]
pub struct AccumulationConfig {
    /// Number of micro-batches to accumulate before applying an optimizer
    /// update.  Must be ≥ 1.
    pub accumulation_steps: usize,
    /// How to normalize the accumulated gradients before returning them.
    pub normalization: GradNormalization,
    /// Whether to clear accumulated gradients automatically after
    /// [`GradientAccumulator::apply`].
    pub auto_clear: bool,
}

impl Default for AccumulationConfig {
    fn default() -> Self {
        Self {
            accumulation_steps: 4,
            normalization: GradNormalization::MeanOverSteps,
            auto_clear: true,
        }
    }
}

impl AccumulationConfig {
    /// Validate the configuration, returning an error if any field is invalid.
    pub fn validate(&self) -> Result<(), AccumulationError> {
        if self.accumulation_steps == 0 {
            return Err(AccumulationError::InvalidConfig(
                "accumulation_steps must be ≥ 1".to_string(),
            ));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GradientAccumulator
// ─────────────────────────────────────────────────────────────────────────────

/// Manages gradient accumulation across multiple micro-batches.
///
/// Call [`initialize`](GradientAccumulator::initialize) once to allocate
/// buffers, then call [`accumulate`](GradientAccumulator::accumulate) once per
/// micro-batch.  When [`should_update`](GradientAccumulator::should_update)
/// returns `true`, call [`apply`](GradientAccumulator::apply) to get
/// normalized gradients ready for the optimizer.
#[derive(Debug)]
pub struct GradientAccumulator {
    /// Configuration controlling accumulation behaviour.
    pub config: AccumulationConfig,
    /// Accumulated gradients per parameter group.
    accumulated: Vec<Vec<f32>>,
    /// Total samples seen in current accumulation window (for
    /// [`GradNormalization::TotalBatchSize`]).
    total_samples: usize,
    /// Steps accumulated so far in this window.
    pub steps_accumulated: usize,
    /// Total optimizer update steps performed (= apply() calls).
    pub total_updates: usize,
    /// Total micro-steps seen (= accumulate() calls).
    pub total_steps: usize,
    /// Per-micro-step batch sizes for the current accumulation window.
    batch_sizes: Vec<usize>,
}

impl GradientAccumulator {
    /// Create a new accumulator from the given configuration.
    ///
    /// Returns [`AccumulationError::InvalidConfig`] if the config is invalid.
    pub fn new(config: AccumulationConfig) -> Result<Self, AccumulationError> {
        config.validate()?;
        Ok(Self {
            config,
            accumulated: Vec::new(),
            total_samples: 0,
            steps_accumulated: 0,
            total_updates: 0,
            total_steps: 0,
            batch_sizes: Vec::new(),
        })
    }

    /// Initialize zero-filled gradient buffers for `param_sizes` parameter
    /// groups.
    ///
    /// Each element in `param_sizes` is the number of scalar gradient values
    /// in that parameter group.  Must be called before the first
    /// [`accumulate`](Self::accumulate) call.
    pub fn initialize(&mut self, param_sizes: &[usize]) -> Result<(), AccumulationError> {
        self.accumulated = param_sizes.iter().map(|&n| vec![0.0_f32; n]).collect();
        self.total_samples = 0;
        self.steps_accumulated = 0;
        self.batch_sizes.clear();
        Ok(())
    }

    /// Accumulate gradients from one micro-batch.
    ///
    /// `gradients` must contain one `Vec<f32>` per parameter group, each with
    /// the same length as passed to [`initialize`](Self::initialize).
    /// `batch_size` is the number of samples in this micro-batch; it is only
    /// used when the normalization mode is
    /// [`GradNormalization::TotalBatchSize`].
    ///
    /// Returns [`AccumulationError::EmptyGradients`] if the accumulator has
    /// not been initialized, and [`AccumulationError::LengthMismatch`] if the
    /// number of parameter groups or the size of any group does not match.
    pub fn accumulate(
        &mut self,
        gradients: &[Vec<f32>],
        batch_size: usize,
    ) -> Result<(), AccumulationError> {
        if self.accumulated.is_empty() {
            return Err(AccumulationError::EmptyGradients);
        }
        if gradients.len() != self.accumulated.len() {
            return Err(AccumulationError::LengthMismatch {
                expected: self.accumulated.len(),
                actual: gradients.len(),
            });
        }
        for (acc, grad) in self.accumulated.iter_mut().zip(gradients.iter()) {
            if acc.len() != grad.len() {
                return Err(AccumulationError::LengthMismatch {
                    expected: acc.len(),
                    actual: grad.len(),
                });
            }
            for (a, g) in acc.iter_mut().zip(grad.iter()) {
                *a += g;
            }
        }
        self.total_samples += batch_size;
        self.batch_sizes.push(batch_size);
        self.steps_accumulated += 1;
        self.total_steps += 1;
        Ok(())
    }

    /// Returns `true` if enough micro-batches have been accumulated and an
    /// optimizer update should be applied.
    pub fn should_update(&self) -> bool {
        self.steps_accumulated >= self.config.accumulation_steps
    }

    /// Update the number of accumulation steps used to decide when
    /// [`should_update`](Self::should_update) returns `true`.
    ///
    /// This lets a caller (e.g. [`AccumulationScheduler`], see
    /// [`AccumulationMonitor::with_scheduler`]) dynamically adjust the
    /// accumulation window between calls to [`accumulate`](Self::accumulate).
    /// Changing it mid-window does not retroactively affect
    /// `steps_accumulated`; it only changes the threshold checked by future
    /// calls to [`should_update`](Self::should_update).
    ///
    /// Returns [`AccumulationError::InvalidConfig`] if `n == 0`.
    pub fn set_accumulation_steps(&mut self, n: usize) -> Result<(), AccumulationError> {
        if n == 0 {
            return Err(AccumulationError::InvalidConfig(
                "accumulation_steps must be ≥ 1".to_string(),
            ));
        }
        self.config.accumulation_steps = n;
        Ok(())
    }

    /// Return the normalized accumulated gradients, ready to pass to the
    /// optimizer.
    ///
    /// Returns [`AccumulationError::NotReady`] if
    /// [`should_update`](Self::should_update) is `false`, and
    /// [`AccumulationError::EmptyGradients`] if the accumulator has not been
    /// initialized.
    pub fn get_gradients(&self) -> Result<Vec<Vec<f32>>, AccumulationError> {
        if self.accumulated.is_empty() {
            return Err(AccumulationError::EmptyGradients);
        }
        if !self.should_update() {
            return Err(AccumulationError::NotReady);
        }
        let scale = match self.config.normalization {
            GradNormalization::MeanOverSteps => {
                // steps_accumulated > 0 because should_update() returned true
                // and accumulation_steps >= 1 from validation.
                1.0_f32 / self.steps_accumulated as f32
            }
            GradNormalization::SumOnly => 1.0_f32,
            GradNormalization::TotalBatchSize => {
                if self.total_samples == 0 {
                    1.0_f32
                } else {
                    1.0_f32 / self.total_samples as f32
                }
            }
        };

        let normalized = self
            .accumulated
            .iter()
            .map(|group| group.iter().map(|&v| v * scale).collect())
            .collect();
        Ok(normalized)
    }

    /// Apply the accumulated update: return normalized gradients and
    /// optionally reset the accumulator.
    ///
    /// - Increments `total_updates`.
    /// - If `auto_clear` is `true` in the config, calls
    ///   [`clear`](Self::clear).
    ///
    /// Returns [`AccumulationError::NotReady`] if
    /// [`should_update`](Self::should_update) is `false`.
    pub fn apply(&mut self) -> Result<Vec<Vec<f32>>, AccumulationError> {
        let gradients = self.get_gradients()?;
        self.total_updates += 1;
        if self.config.auto_clear {
            self.clear();
        }
        Ok(gradients)
    }

    /// Clear accumulated gradients and reset the step counter for this window.
    ///
    /// The accumulated buffers are zeroed in-place (their sizes are preserved
    /// so that `accumulate` can continue without another `initialize` call).
    /// `total_updates` and `total_steps` are **not** reset.
    pub fn clear(&mut self) {
        for group in self.accumulated.iter_mut() {
            for v in group.iter_mut() {
                *v = 0.0;
            }
        }
        self.steps_accumulated = 0;
        self.total_samples = 0;
        self.batch_sizes.clear();
    }

    /// Per-micro-step batch sizes recorded so far in the *current*
    /// accumulation window, in the order they were accumulated.
    ///
    /// Cleared by [`clear`](Self::clear) (and therefore by
    /// [`apply`](Self::apply) when `auto_clear` is set), so this is bounded by
    /// `accumulation_steps` rather than growing with the training run.
    pub fn window_batch_sizes(&self) -> &[usize] {
        &self.batch_sizes
    }

    /// Total samples seen in the current accumulation window.
    ///
    /// This is the divisor used by [`GradNormalization::TotalBatchSize`].
    pub fn window_total_samples(&self) -> usize {
        self.total_samples
    }

    /// Mean micro-batch size in the current window, or `None` when no
    /// micro-batch has been accumulated yet.
    pub fn mean_batch_size(&self) -> Option<f32> {
        if self.batch_sizes.is_empty() {
            return None;
        }
        Some(self.total_samples as f32 / self.batch_sizes.len() as f32)
    }

    /// Ratio `max_batch_size / min_batch_size` across the current window, or
    /// `None` when the window is empty or its smallest micro-batch is zero.
    ///
    /// A ratio above `1.0` means the micro-batches are uneven, which biases
    /// [`GradNormalization::MeanOverSteps`]: that mode weights every
    /// micro-batch equally regardless of how many samples it held, so an
    /// imbalanced window silently over-weights the small micro-batches.
    /// [`GradNormalization::TotalBatchSize`] is the unbiased choice when this
    /// is far from `1.0`.
    pub fn batch_size_imbalance(&self) -> Option<f32> {
        let min = *self.batch_sizes.iter().min()?;
        let max = *self.batch_sizes.iter().max()?;
        if min == 0 {
            return None;
        }
        Some(max as f32 / min as f32)
    }

    /// Accumulation progress within the current window: `steps_accumulated /
    /// accumulation_steps`, clamped to `[0.0, 1.0]`.
    pub fn progress(&self) -> f64 {
        if self.config.accumulation_steps == 0 {
            return 1.0;
        }
        (self.steps_accumulated as f64 / self.config.accumulation_steps as f64).min(1.0)
    }

    /// Update efficiency: fraction of micro-steps that resulted in an
    /// optimizer update.
    ///
    /// When no steps have been taken yet, returns the theoretical efficiency
    /// `1.0 / accumulation_steps`.
    pub fn update_efficiency(&self) -> f64 {
        if self.total_steps == 0 {
            1.0 / self.config.accumulation_steps as f64
        } else {
            self.total_updates as f64 / self.total_steps as f64
        }
    }

    /// Format a one-line status summary.
    ///
    /// Example:
    /// `"acc=2/4 updates=3 steps=12 efficiency=25.00% samples=16 mean_batch=8.00 imbalance=1.00"`
    ///
    /// The trailing window statistics are omitted while the current window is
    /// still empty (nothing has been accumulated since the last `clear`).
    pub fn format_status(&self) -> String {
        let mut out = String::new();
        let _ = write!(
            out,
            "acc={}/{} updates={} steps={} efficiency={:.2}%",
            self.steps_accumulated,
            self.config.accumulation_steps,
            self.total_updates,
            self.total_steps,
            self.update_efficiency() * 100.0,
        );
        if let Some(mean_batch) = self.mean_batch_size() {
            let _ = write!(
                out,
                " samples={} mean_batch={mean_batch:.2}",
                self.total_samples
            );
            if let Some(imbalance) = self.batch_size_imbalance() {
                let _ = write!(out, " imbalance={imbalance:.2}");
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mixed precision helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Scale a scalar loss by `scale_factor` before gradient accumulation.
///
/// Used in FP16 training to prevent underflow during the backward pass.  The
/// accumulated gradients should later be unscaled with
/// [`unscale_gradients`].
#[inline]
pub fn scale_loss(loss: f32, scale_factor: f32) -> f32 {
    loss * scale_factor
}

/// Divide every element in every parameter group by `scale_factor`.
///
/// Call this after accumulation (and before the optimizer step) to recover
/// the true gradients when training with loss scaling.
pub fn unscale_gradients(gradients: &mut [Vec<f32>], scale_factor: f32) {
    if scale_factor == 0.0 {
        return;
    }
    let inv = 1.0 / scale_factor;
    for group in gradients.iter_mut() {
        for v in group.iter_mut() {
            *v *= inv;
        }
    }
}

/// Return `true` if any gradient value across all parameter groups is
/// infinite (positive or negative).
///
/// Indicates an overflow in FP16 training — the optimizer step should be
/// skipped and the loss scale should be reduced.
pub fn gradients_have_inf(gradients: &[Vec<f32>]) -> bool {
    gradients
        .iter()
        .any(|group| group.iter().any(|&v| v.is_infinite()))
}

/// Return `true` if any gradient value across all parameter groups is NaN.
pub fn gradients_have_nan(gradients: &[Vec<f32>]) -> bool {
    gradients
        .iter()
        .any(|group| group.iter().any(|&v| v.is_nan()))
}

// ─────────────────────────────────────────────────────────────────────────────
// AccumulationScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Dynamically adjusts the number of accumulation steps based on training
/// progress.
///
/// Before `switch_at_step` training steps the scheduler uses
/// `initial_steps`; after that it switches to `target_steps`.  This allows
/// a warm-up period with a different accumulation factor (e.g. smaller at
/// the start when gradients are noisy, larger once training is stable).
#[derive(Debug)]
pub struct AccumulationScheduler {
    /// Accumulation steps used at the start of training.
    pub initial_steps: usize,
    /// Accumulation steps used after the warm-up period.
    pub target_steps: usize,
    /// The training step index at which to switch from `initial_steps` to
    /// `target_steps`.
    pub switch_at_step: usize,
    /// Current training step (incremented by [`advance`](Self::advance)).
    current_training_step: usize,
}

impl AccumulationScheduler {
    /// Create a new scheduler.
    ///
    /// Returns [`AccumulationError::InvalidConfig`] if either `initial_steps`
    /// or `target_steps` is zero.
    pub fn new(
        initial_steps: usize,
        target_steps: usize,
        switch_at_step: usize,
    ) -> Result<Self, AccumulationError> {
        if initial_steps == 0 {
            return Err(AccumulationError::InvalidConfig(
                "initial_steps must be ≥ 1".to_string(),
            ));
        }
        if target_steps == 0 {
            return Err(AccumulationError::InvalidConfig(
                "target_steps must be ≥ 1".to_string(),
            ));
        }
        Ok(Self {
            initial_steps,
            target_steps,
            switch_at_step,
            current_training_step: 0,
        })
    }

    /// Return the accumulation step count applicable at the current training
    /// step.
    pub fn current_accumulation_steps(&self) -> usize {
        if self.current_training_step < self.switch_at_step {
            self.initial_steps
        } else {
            self.target_steps
        }
    }

    /// Advance the internal training step counter by one.
    pub fn advance(&mut self) {
        self.current_training_step += 1;
    }

    /// Whether the micro-batch at `micro_step` (zero-indexed within the
    /// accumulation window) is the last one before an optimizer update.
    ///
    /// Returns `true` when
    /// `micro_step % current_accumulation_steps() == current_accumulation_steps() - 1`.
    ///
    /// The validating constructor ([`Self::new`]) rejects `initial_steps` /
    /// `target_steps` of `0`, but both fields are `pub`, so a struct literal
    /// can still reach this method with `current_accumulation_steps() == 0`;
    /// guard that case explicitly to avoid a modulo-by-zero panic and a
    /// `usize` underflow on `n - 1`.
    pub fn should_update(&self, micro_step: usize) -> bool {
        let n = self.current_accumulation_steps();
        if n == 0 {
            return true;
        }
        micro_step % n == n - 1
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AccumulationStats / AccumulationMonitor
// ─────────────────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of accumulation statistics.
#[derive(Debug, Clone)]
pub struct AccumulationStats {
    /// Total micro-steps seen.
    pub total_steps: usize,
    /// Total optimizer updates applied.
    pub total_updates: usize,
    /// Mean L2 norm of gradients over all recorded micro-steps.
    ///
    /// `0.0` if no steps have been recorded.
    pub mean_grad_norm: f32,
    /// Fraction of micro-steps that produced an optimizer update.
    pub update_efficiency: f64,
    /// Number of micro-steps where gradients contained infinity.
    pub overflows: usize,
}

/// Default number of recent per-micro-step gradient norms
/// [`AccumulationMonitor`] retains.
///
/// Only the running mean is needed for [`AccumulationStats`], and that is kept
/// exactly (as a running sum plus a count) regardless of this window — the
/// window exists so recent norms can be *inspected*
/// ([`AccumulationMonitor::recent_grad_norms`]) without the buffer growing for
/// the whole training run.
pub const DEFAULT_GRAD_NORM_WINDOW: usize = 1024;

/// Wraps a [`GradientAccumulator`] with automatic norm tracking and overflow
/// detection.
#[derive(Debug)]
pub struct AccumulationMonitor {
    accumulator: GradientAccumulator,
    /// Sliding window of the most recent per-micro-step L2 norms, bounded by
    /// `grad_norm_window`.
    grad_norms: VecDeque<f32>,
    /// Maximum number of entries retained in `grad_norms`.
    grad_norm_window: usize,
    /// Sum of *every* recorded micro-step norm (not just the retained
    /// window), accumulated in `f64` so a long run does not lose precision.
    grad_norm_sum: f64,
    /// Number of micro-step norms folded into `grad_norm_sum`.
    grad_norm_count: usize,
    /// Count of micro-steps where [`gradients_have_inf`] was true.
    overflows: usize,
    /// Optional dynamic accumulation-step schedule. When present, its
    /// current value is pushed into the wrapped [`GradientAccumulator`] at
    /// the start of every [`step`](Self::step) call (via
    /// [`GradientAccumulator::set_accumulation_steps`]), and it is then
    /// advanced by one training step -- see [`Self::with_scheduler`].
    scheduler: Option<AccumulationScheduler>,
}

impl AccumulationMonitor {
    /// Create a new monitor wrapping a fresh accumulator built from `config`,
    /// retaining [`DEFAULT_GRAD_NORM_WINDOW`] recent gradient norms.
    pub fn new(config: AccumulationConfig) -> Result<Self, AccumulationError> {
        Self::with_grad_norm_window(config, DEFAULT_GRAD_NORM_WINDOW)
    }

    /// Create a new monitor that retains at most `grad_norm_window` recent
    /// per-micro-step gradient norms.
    ///
    /// The window is bounded so a long training run cannot grow the norm
    /// buffer without limit; [`AccumulationStats::mean_grad_norm`] stays the
    /// mean over *all* recorded micro-steps regardless, because it is
    /// maintained as a running sum rather than recomputed from the buffer.
    ///
    /// Returns [`AccumulationError::InvalidConfig`] if `grad_norm_window` is
    /// zero (or the accumulation config is invalid).
    pub fn with_grad_norm_window(
        config: AccumulationConfig,
        grad_norm_window: usize,
    ) -> Result<Self, AccumulationError> {
        if grad_norm_window == 0 {
            return Err(AccumulationError::InvalidConfig(
                "grad_norm_window must be ≥ 1".to_string(),
            ));
        }
        let accumulator = GradientAccumulator::new(config)?;
        Ok(Self {
            accumulator,
            grad_norms: VecDeque::with_capacity(grad_norm_window),
            grad_norm_window,
            grad_norm_sum: 0.0,
            grad_norm_count: 0,
            overflows: 0,
            scheduler: None,
        })
    }

    /// Attach a dynamic accumulation-step [`AccumulationScheduler`] to this
    /// monitor.
    ///
    /// Once attached, every [`step`](Self::step) call first reads
    /// [`AccumulationScheduler::current_accumulation_steps`], pushes it into
    /// the wrapped accumulator, and then advances the scheduler's internal
    /// training-step counter -- so the schedule configured on the scheduler
    /// actually changes when the accumulator fires optimizer updates,
    /// instead of the two types being constructed independently and never
    /// interacting.
    #[must_use]
    pub fn with_scheduler(mut self, scheduler: AccumulationScheduler) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Shared reference to the attached scheduler, if any (see
    /// [`Self::with_scheduler`]).
    pub fn scheduler(&self) -> Option<&AccumulationScheduler> {
        self.scheduler.as_ref()
    }

    /// Process one micro-batch.
    ///
    /// - If a scheduler is attached (see [`Self::with_scheduler`]), pushes
    ///   its current accumulation-step count into the wrapped accumulator
    ///   and advances it.
    /// - Lazily initializes the accumulator on the first call using the sizes
    ///   of `gradients`.
    /// - Accumulates `gradients`.
    /// - Records the L2 gradient norm and overflow flag.
    /// - Returns `Some(normalized_gradients)` when an update is ready,
    ///   otherwise `None`.
    pub fn step(
        &mut self,
        gradients: &[Vec<f32>],
        batch_size: usize,
    ) -> Result<Option<Vec<Vec<f32>>>, AccumulationError> {
        if let Some(scheduler) = &mut self.scheduler {
            let n = scheduler.current_accumulation_steps();
            self.accumulator.set_accumulation_steps(n)?;
            scheduler.advance();
        }

        // Lazy initialization: if the inner accumulator has no buffers yet,
        // derive the sizes from the provided gradients.
        if self.accumulator.accumulated.is_empty() {
            if gradients.is_empty() {
                return Err(AccumulationError::EmptyGradients);
            }
            let sizes: Vec<usize> = gradients.iter().map(|g| g.len()).collect();
            self.accumulator.initialize(&sizes)?;
        }

        // Compute L2 norm across all parameter groups.
        let sq_sum: f32 = gradients
            .iter()
            .flat_map(|g| g.iter())
            .map(|&v| v * v)
            .sum();
        let norm = sq_sum.sqrt();
        self.grad_norm_sum += f64::from(norm);
        self.grad_norm_count += 1;
        if self.grad_norms.len() == self.grad_norm_window {
            self.grad_norms.pop_front();
        }
        self.grad_norms.push_back(norm);

        // Track overflow.
        if gradients_have_inf(gradients) {
            self.overflows += 1;
        }

        self.accumulator.accumulate(gradients, batch_size)?;

        if self.accumulator.should_update() {
            let normalized = self.accumulator.apply()?;
            return Ok(Some(normalized));
        }
        Ok(None)
    }

    /// The most recent per-micro-step gradient norms, oldest first.
    ///
    /// Bounded by [`grad_norm_window`](Self::grad_norm_window); norms older
    /// than the window have been evicted (their contribution to
    /// [`AccumulationStats::mean_grad_norm`] is retained regardless).
    pub fn recent_grad_norms(&self) -> Vec<f32> {
        self.grad_norms.iter().copied().collect()
    }

    /// Maximum number of recent gradient norms retained by this monitor.
    pub fn grad_norm_window(&self) -> usize {
        self.grad_norm_window
    }

    /// Return a snapshot of the current accumulation statistics.
    pub fn stats(&self) -> AccumulationStats {
        let mean_grad_norm = if self.grad_norm_count == 0 {
            0.0
        } else {
            (self.grad_norm_sum / self.grad_norm_count as f64) as f32
        };
        AccumulationStats {
            total_steps: self.accumulator.total_steps,
            total_updates: self.accumulator.total_updates,
            mean_grad_norm,
            update_efficiency: self.accumulator.update_efficiency(),
            overflows: self.overflows,
        }
    }

    /// Shared reference to the inner [`GradientAccumulator`].
    pub fn inner(&self) -> &GradientAccumulator {
        &self.accumulator
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_config(steps: usize, norm: GradNormalization) -> AccumulationConfig {
        AccumulationConfig {
            accumulation_steps: steps,
            normalization: norm,
            auto_clear: true,
        }
    }

    fn make_acc(steps: usize, norm: GradNormalization) -> GradientAccumulator {
        GradientAccumulator::new(make_config(steps, norm)).expect("valid config must not fail")
    }

    // ── 1. AccumulationConfig::validate — steps=0 ────────────────────────────

    #[test]
    fn test_config_validate_steps_zero_is_err() {
        let cfg = AccumulationConfig {
            accumulation_steps: 0,
            normalization: GradNormalization::MeanOverSteps,
            auto_clear: true,
        };
        assert!(
            cfg.validate().is_err(),
            "accumulation_steps=0 must return Err"
        );
    }

    // ── 2. AccumulationConfig::validate — steps=1 ────────────────────────────

    #[test]
    fn test_config_validate_steps_one_is_ok() {
        let cfg = AccumulationConfig {
            accumulation_steps: 1,
            normalization: GradNormalization::MeanOverSteps,
            auto_clear: true,
        };
        assert!(cfg.validate().is_ok(), "accumulation_steps=1 must be Ok");
    }

    // ── 3. GradientAccumulator::new — valid config ────────────────────────────

    #[test]
    fn test_accumulator_new_valid_config() {
        let config = AccumulationConfig::default();
        let result = GradientAccumulator::new(config);
        assert!(result.is_ok(), "new with valid config must succeed");
        let acc = result.expect("valid");
        assert_eq!(acc.steps_accumulated, 0);
        assert_eq!(acc.total_updates, 0);
        assert_eq!(acc.total_steps, 0);
    }

    // ── 4. initialize: zero buffers of correct sizes ──────────────────────────

    #[test]
    fn test_initialize_creates_zero_buffers() {
        let mut acc = make_acc(4, GradNormalization::MeanOverSteps);
        acc.initialize(&[3, 5, 2]).expect("initialize must succeed");
        assert_eq!(acc.accumulated.len(), 3);
        assert_eq!(acc.accumulated[0].len(), 3);
        assert_eq!(acc.accumulated[1].len(), 5);
        assert_eq!(acc.accumulated[2].len(), 2);
        for group in &acc.accumulated {
            for &v in group {
                assert_eq!(v, 0.0, "buffer must be zero-initialized");
            }
        }
    }

    // ── 5. accumulate: element-wise addition ─────────────────────────────────

    #[test]
    fn test_accumulate_adds_element_wise() {
        let mut acc = make_acc(4, GradNormalization::SumOnly);
        acc.initialize(&[3]).expect("initialize");
        let g1 = vec![vec![1.0_f32, 2.0, 3.0]];
        let g2 = vec![vec![10.0_f32, 20.0, 30.0]];
        acc.accumulate(&g1, 1).expect("accumulate g1");
        acc.accumulate(&g2, 1).expect("accumulate g2");
        assert!((acc.accumulated[0][0] - 11.0).abs() < 1e-6);
        assert!((acc.accumulated[0][1] - 22.0).abs() < 1e-6);
        assert!((acc.accumulated[0][2] - 33.0).abs() < 1e-6);
    }

    // ── 6. should_update: false before N steps ───────────────────────────────

    #[test]
    fn test_should_update_false_before_n_steps() {
        let mut acc = make_acc(4, GradNormalization::MeanOverSteps);
        acc.initialize(&[2]).expect("initialize");
        let g = vec![vec![1.0_f32, 1.0]];
        for _ in 0..3 {
            acc.accumulate(&g, 1).expect("accumulate");
            assert!(
                !acc.should_update(),
                "must be false before reaching N steps"
            );
        }
    }

    // ── 7. should_update: true after N steps ─────────────────────────────────

    #[test]
    fn test_should_update_true_after_n_steps() {
        let mut acc = make_acc(4, GradNormalization::MeanOverSteps);
        acc.initialize(&[2]).expect("initialize");
        let g = vec![vec![1.0_f32, 1.0]];
        for _ in 0..4 {
            acc.accumulate(&g, 1).expect("accumulate");
        }
        assert!(acc.should_update(), "must be true after N steps");
    }

    // ── 8. get_gradients: NotReady before N steps ─────────────────────────────

    #[test]
    fn test_get_gradients_not_ready_before_n_steps() {
        let mut acc = make_acc(4, GradNormalization::MeanOverSteps);
        acc.initialize(&[2]).expect("initialize");
        let g = vec![vec![1.0_f32, 1.0]];
        acc.accumulate(&g, 1).expect("accumulate");
        let result = acc.get_gradients();
        assert!(
            matches!(result, Err(AccumulationError::NotReady)),
            "must return NotReady when steps < accumulation_steps"
        );
    }

    // ── 9. get_gradients: MeanOverSteps divides by N ─────────────────────────

    #[test]
    fn test_get_gradients_mean_over_steps() {
        let mut acc = make_acc(4, GradNormalization::MeanOverSteps);
        acc.initialize(&[2]).expect("initialize");
        let g = vec![vec![4.0_f32, 8.0]];
        for _ in 0..4 {
            acc.accumulate(&g, 1).expect("accumulate");
        }
        let normalized = acc.get_gradients().expect("should be ready");
        // Sum = 4*4 = 16.0 / 4 = 4.0  and  4*8 = 32.0 / 4 = 8.0
        assert!(
            (normalized[0][0] - 4.0).abs() < 1e-5,
            "mean over 4 steps of 4.0 must be 4.0"
        );
        assert!(
            (normalized[0][1] - 8.0).abs() < 1e-5,
            "mean over 4 steps of 8.0 must be 8.0"
        );
    }

    // ── 10. get_gradients: SumOnly returns raw sum ────────────────────────────

    #[test]
    fn test_get_gradients_sum_only() {
        let mut acc = make_acc(3, GradNormalization::SumOnly);
        acc.initialize(&[2]).expect("initialize");
        let g = vec![vec![2.0_f32, 3.0]];
        for _ in 0..3 {
            acc.accumulate(&g, 1).expect("accumulate");
        }
        let normalized = acc.get_gradients().expect("ready");
        assert!((normalized[0][0] - 6.0).abs() < 1e-5, "sum of 3×2.0 = 6.0");
        assert!((normalized[0][1] - 9.0).abs() < 1e-5, "sum of 3×3.0 = 9.0");
    }

    // ── 11. apply: returns gradients and resets ───────────────────────────────

    #[test]
    fn test_apply_returns_gradients_and_resets() {
        let mut acc = make_acc(2, GradNormalization::MeanOverSteps);
        acc.initialize(&[1]).expect("initialize");
        let g = vec![vec![6.0_f32]];
        acc.accumulate(&g, 1).expect("step 1");
        acc.accumulate(&g, 1).expect("step 2");
        let result = acc.apply().expect("apply must succeed");
        assert!(
            (result[0][0] - 6.0).abs() < 1e-5,
            "mean of 6.0 over 2 steps = 6.0"
        );
        // After apply with auto_clear, steps_accumulated must be reset.
        assert_eq!(acc.steps_accumulated, 0, "steps must reset after apply");
    }

    // ── 12. apply: increments total_updates ──────────────────────────────────

    #[test]
    fn test_apply_increments_total_updates() {
        let mut acc = make_acc(2, GradNormalization::MeanOverSteps);
        acc.initialize(&[1]).expect("initialize");
        let g = vec![vec![1.0_f32]];
        for _ in 0..2 {
            acc.accumulate(&g, 1).expect("acc");
        }
        acc.apply().expect("apply 1");
        assert_eq!(acc.total_updates, 1);
        for _ in 0..2 {
            acc.accumulate(&g, 1).expect("acc");
        }
        acc.apply().expect("apply 2");
        assert_eq!(acc.total_updates, 2);
    }

    // ── 13. clear: zeroes accumulated and resets step counter ─────────────────

    #[test]
    fn test_clear_zeroes_buffers_and_resets_counter() {
        let mut acc = make_acc(4, GradNormalization::MeanOverSteps);
        acc.initialize(&[3]).expect("initialize");
        let g = vec![vec![5.0_f32, 5.0, 5.0]];
        acc.accumulate(&g, 1).expect("acc");
        acc.accumulate(&g, 1).expect("acc");
        assert_eq!(acc.steps_accumulated, 2);
        acc.clear();
        assert_eq!(acc.steps_accumulated, 0);
        for v in &acc.accumulated[0] {
            assert_eq!(*v, 0.0, "buffer must be zeroed after clear");
        }
    }

    // ── 14. progress: 0 before, 1.0 when ready ───────────────────────────────

    #[test]
    fn test_progress_values() {
        let mut acc = make_acc(4, GradNormalization::MeanOverSteps);
        acc.initialize(&[1]).expect("initialize");
        let p0 = acc.progress();
        assert!((p0 - 0.0).abs() < 1e-9, "progress at start must be 0.0");
        let g = vec![vec![1.0_f32]];
        for _ in 0..4 {
            acc.accumulate(&g, 1).expect("acc");
        }
        let p_full = acc.progress();
        assert!(
            (p_full - 1.0).abs() < 1e-9,
            "progress when ready must be 1.0"
        );
    }

    // ── 15. accumulate: length mismatch ──────────────────────────────────────

    #[test]
    fn test_accumulate_length_mismatch_returns_err() {
        let mut acc = make_acc(4, GradNormalization::MeanOverSteps);
        acc.initialize(&[3, 3]).expect("initialize with 2 groups");
        // Pass 3 groups instead of 2.
        let g = vec![vec![1.0_f32; 3], vec![1.0_f32; 3], vec![1.0_f32; 3]];
        let result = acc.accumulate(&g, 1);
        assert!(
            matches!(result, Err(AccumulationError::LengthMismatch { .. })),
            "mismatched group count must return LengthMismatch"
        );
    }

    // ── 16. unscale_gradients ────────────────────────────────────────────────

    #[test]
    fn test_unscale_gradients_divides_by_scale_factor() {
        let mut grads = vec![vec![4.0_f32, 8.0], vec![16.0_f32]];
        unscale_gradients(&mut grads, 4.0);
        assert!((grads[0][0] - 1.0).abs() < 1e-6, "4 / 4 = 1");
        assert!((grads[0][1] - 2.0).abs() < 1e-6, "8 / 4 = 2");
        assert!((grads[1][0] - 4.0).abs() < 1e-6, "16 / 4 = 4");
    }

    // ── 17. gradients_have_inf ────────────────────────────────────────────────

    #[test]
    fn test_gradients_have_inf_detects_infinity() {
        let grads = vec![vec![1.0_f32, f32::INFINITY]];
        assert!(gradients_have_inf(&grads), "+Inf must be detected");
        let grads_neg = vec![vec![1.0_f32, f32::NEG_INFINITY]];
        assert!(gradients_have_inf(&grads_neg), "-Inf must be detected");
        let clean = vec![vec![1.0_f32, 2.0]];
        assert!(!gradients_have_inf(&clean), "clean grads must not trigger");
    }

    // ── 18. gradients_have_nan ────────────────────────────────────────────────

    #[test]
    fn test_gradients_have_nan_detects_nan() {
        let grads = vec![vec![1.0_f32, f32::NAN]];
        assert!(gradients_have_nan(&grads), "NaN must be detected");
        let clean = vec![vec![1.0_f32, 2.0]];
        assert!(!gradients_have_nan(&clean), "clean grads must not trigger");
    }

    // ── 19. scale_loss ────────────────────────────────────────────────────────

    #[test]
    fn test_scale_loss_multiplies_correctly() {
        let scaled = scale_loss(3.0, 1024.0);
        assert!((scaled - 3072.0).abs() < 1e-3, "3.0 * 1024 = 3072.0");
        let no_op = scale_loss(2.5, 1.0);
        assert!((no_op - 2.5).abs() < 1e-6, "scale by 1.0 must be identity");
    }

    // ── 20. AccumulationScheduler: before switch → initial_steps ─────────────

    #[test]
    fn test_scheduler_before_switch_returns_initial_steps() {
        let sched = AccumulationScheduler::new(2, 8, 100).expect("valid scheduler");
        assert_eq!(
            sched.current_accumulation_steps(),
            2,
            "before switch must return initial_steps"
        );
    }

    // ── 21. AccumulationScheduler: after switch → target_steps ───────────────

    #[test]
    fn test_scheduler_after_switch_returns_target_steps() {
        let mut sched = AccumulationScheduler::new(2, 8, 3).expect("valid scheduler");
        // Advance past switch_at_step (= 3).
        for _ in 0..3 {
            sched.advance();
        }
        assert_eq!(
            sched.current_accumulation_steps(),
            8,
            "after switch must return target_steps"
        );
    }

    // ── 22. AccumulationMonitor::step: None until N, Some on Nth ─────────────

    #[test]
    fn test_monitor_step_returns_none_until_n_then_some() {
        let config = AccumulationConfig {
            accumulation_steps: 3,
            normalization: GradNormalization::MeanOverSteps,
            auto_clear: true,
        };
        let mut monitor = AccumulationMonitor::new(config).expect("new");
        let g = vec![vec![3.0_f32, 6.0]];
        let r1 = monitor.step(&g, 1).expect("step 1");
        assert!(r1.is_none(), "step 1 of 3 must return None");
        let r2 = monitor.step(&g, 1).expect("step 2");
        assert!(r2.is_none(), "step 2 of 3 must return None");
        let r3 = monitor.step(&g, 1).expect("step 3");
        assert!(r3.is_some(), "step 3 of 3 must return Some");
        let normalized = r3.expect("some");
        // Mean of 3 steps of 3.0 = 3.0
        assert!((normalized[0][0] - 3.0).abs() < 1e-5, "mean must be 3.0");
        assert!((normalized[0][1] - 6.0).abs() < 1e-5, "mean must be 6.0");
    }

    // ── 23. AccumulationMonitor::stats: correct counts ────────────────────────

    #[test]
    fn test_monitor_stats_correct_counts() {
        let config = AccumulationConfig {
            accumulation_steps: 2,
            normalization: GradNormalization::MeanOverSteps,
            auto_clear: true,
        };
        let mut monitor = AccumulationMonitor::new(config).expect("new");
        let g = vec![vec![1.0_f32]];
        let inf_g = vec![vec![f32::INFINITY]];

        // Step 1: normal.
        monitor.step(&g, 1).expect("step 1");
        // Step 2: normal → triggers update.
        monitor.step(&g, 1).expect("step 2");
        // Step 3: inf → overflow.
        monitor.step(&inf_g, 1).expect("step 3");
        // Step 4: normal → triggers update.
        monitor.step(&g, 1).expect("step 4");

        let stats = monitor.stats();
        assert_eq!(stats.total_steps, 4, "4 micro-steps");
        assert_eq!(stats.total_updates, 2, "2 optimizer updates");
        assert_eq!(stats.overflows, 1, "1 overflow (step 3 had inf)");
        assert!(
            stats.mean_grad_norm >= 0.0,
            "mean norm must be non-negative"
        );
    }

    // ── Extra: TotalBatchSize normalization ───────────────────────────────────

    #[test]
    fn test_get_gradients_total_batch_size() {
        let config = AccumulationConfig {
            accumulation_steps: 3,
            normalization: GradNormalization::TotalBatchSize,
            auto_clear: false,
        };
        let mut acc = GradientAccumulator::new(config).expect("new");
        acc.initialize(&[2]).expect("init");
        // 3 steps with batch sizes 2, 3, 5 → total = 10
        acc.accumulate(&[vec![20.0_f32, 40.0]], 2).expect("s1");
        acc.accumulate(&[vec![30.0_f32, 60.0]], 3).expect("s2");
        acc.accumulate(&[vec![50.0_f32, 100.0]], 5).expect("s3");
        let g = acc.get_gradients().expect("ready");
        // sum = 100.0, total_samples = 10 → normalized = 10.0
        assert!((g[0][0] - 10.0).abs() < 1e-4, "100/10 = 10.0");
        assert!((g[0][1] - 20.0).abs() < 1e-4, "200/10 = 20.0");
    }

    // ── Extra: update_efficiency when total_steps = 0 ────────────────────────

    #[test]
    fn test_update_efficiency_no_steps() {
        let acc = make_acc(4, GradNormalization::MeanOverSteps);
        let eff = acc.update_efficiency();
        assert!(
            (eff - 0.25).abs() < 1e-9,
            "efficiency with no steps = 1/4 = 0.25"
        );
    }

    // ── Extra: format_status ─────────────────────────────────────────────────

    #[test]
    fn test_format_status_contains_expected_fields() {
        let acc = make_acc(4, GradNormalization::MeanOverSteps);
        let s = acc.format_status();
        assert!(s.contains("acc="), "must contain acc field");
        assert!(s.contains("updates="), "must contain updates field");
        assert!(s.contains("efficiency="), "must contain efficiency field");
    }

    // ── Extra: accumulator not initialized returns EmptyGradients ────────────

    #[test]
    fn test_accumulate_without_initialize_returns_empty_gradients() {
        let mut acc = make_acc(4, GradNormalization::MeanOverSteps);
        let g = vec![vec![1.0_f32]];
        let result = acc.accumulate(&g, 1);
        assert!(
            matches!(result, Err(AccumulationError::EmptyGradients)),
            "must return EmptyGradients before initialize"
        );
    }

    // ── Extra: apply NotReady ─────────────────────────────────────────────────

    #[test]
    fn test_apply_not_ready_returns_err() {
        let mut acc = make_acc(4, GradNormalization::MeanOverSteps);
        acc.initialize(&[1]).expect("init");
        let g = vec![vec![1.0_f32]];
        acc.accumulate(&g, 1).expect("acc");
        let result = acc.apply();
        assert!(
            matches!(result, Err(AccumulationError::NotReady)),
            "apply before ready must return NotReady"
        );
    }

    // ── Extra: scheduler invalid configs ─────────────────────────────────────

    #[test]
    fn test_scheduler_invalid_initial_steps() {
        let result = AccumulationScheduler::new(0, 4, 100);
        assert!(result.is_err(), "initial_steps=0 must fail");
    }

    #[test]
    fn test_scheduler_should_update_logic() {
        let sched = AccumulationScheduler::new(4, 4, 1000).expect("valid");
        // With accumulation_steps=4: update at micro_step 3, 7, 11 …
        assert!(!sched.should_update(0));
        assert!(!sched.should_update(1));
        assert!(!sched.should_update(2));
        assert!(sched.should_update(3));
        assert!(!sched.should_update(4));
        assert!(sched.should_update(7));
    }

    // ── set_accumulation_steps / should_update(0) panic sweep ────────────────

    #[test]
    fn test_scheduler_should_update_zero_steps_no_panic() {
        // Bypass the validating constructor via a direct struct literal (as
        // a misused external caller might, since the fields are `pub`).
        let sched = AccumulationScheduler {
            initial_steps: 0,
            target_steps: 4,
            switch_at_step: 100,
            current_training_step: 0,
        };
        // Must not panic (modulo/division by zero, or a `0 - 1` underflow).
        assert!(sched.should_update(0));
        assert!(sched.should_update(12345));
    }

    #[test]
    fn test_set_accumulation_steps_updates_config_and_validates() {
        let mut acc = make_acc(4, GradNormalization::MeanOverSteps);
        acc.set_accumulation_steps(10).expect("valid");
        assert_eq!(acc.config.accumulation_steps, 10);

        let err = acc.set_accumulation_steps(0);
        assert!(matches!(err, Err(AccumulationError::InvalidConfig(_))));
        // A rejected update must not have mutated the config.
        assert_eq!(acc.config.accumulation_steps, 10);
    }

    // ── AccumulationMonitor::with_scheduler ───────────────────────────────────

    #[test]
    fn test_monitor_with_scheduler_changes_effective_accumulation_steps() {
        // Base config uses accumulation_steps=99 -- irrelevant once the
        // scheduler is attached, since its value takes over on the very
        // first `step()` call.
        let config = AccumulationConfig {
            accumulation_steps: 99,
            normalization: GradNormalization::MeanOverSteps,
            auto_clear: true,
        };
        let scheduler = AccumulationScheduler::new(2, 5, 4).expect("valid scheduler");
        let mut monitor = AccumulationMonitor::new(config)
            .expect("new")
            .with_scheduler(scheduler);

        let g = vec![vec![1.0_f32]];
        // Before switch_at_step=4: scheduler reports initial_steps=2, so an
        // update should fire every 2 micro-steps.
        let r1 = monitor.step(&g, 1).expect("step 1");
        assert!(r1.is_none(), "1 of 2 accumulated");
        let r2 = monitor.step(&g, 1).expect("step 2");
        assert!(r2.is_some(), "2 of 2 accumulated -> update");

        let r3 = monitor.step(&g, 1).expect("step 3");
        assert!(r3.is_none(), "1 of 2 accumulated (still initial_steps)");
        let r4 = monitor.step(&g, 1).expect("step 4");
        assert!(r4.is_some(), "2 of 2 accumulated -> update");

        // By now the scheduler's internal training-step counter has
        // advanced to 4, at/after switch_at_step=4, so the accumulation
        // window should have grown to target_steps=5.
        assert_eq!(
            monitor
                .scheduler()
                .expect("scheduler attached")
                .current_accumulation_steps(),
            5,
            "scheduler should have switched to target_steps by now"
        );
        for i in 0..4 {
            let r = monitor.step(&g, 1).expect("mid-window step");
            assert!(r.is_none(), "step {i} of 5 should not update yet");
        }
        let r_final = monitor.step(&g, 1).expect("5th step");
        assert!(r_final.is_some(), "5 of 5 accumulated -> update");
    }

    // ── Regression (F284): the monitor's norm buffer is a bounded window ─────
    // `grad_norms` used to be an unbounded `Vec<f32>` pushed once per
    // micro-step, so it grew for the whole training run. The window is now
    // bounded, and `mean_grad_norm` must still be the mean over *every*
    // recorded micro-step, not just the retained tail.
    #[test]
    fn test_monitor_grad_norms_bounded_but_mean_covers_all_steps() {
        let config = make_config(1, GradNormalization::MeanOverSteps);
        let mut monitor =
            AccumulationMonitor::with_grad_norm_window(config, 4).expect("window 4 is valid");
        assert_eq!(monitor.grad_norm_window(), 4);

        // 100 micro-steps with norms 1.0 .. 100.0 (single-element gradients,
        // so the L2 norm is the element itself).
        for i in 1..=100 {
            monitor.step(&[vec![i as f32]], 1).expect("step");
        }

        let recent = monitor.recent_grad_norms();
        assert_eq!(recent.len(), 4, "norm buffer must stay bounded");
        for (j, &v) in recent.iter().enumerate() {
            let expected = (97 + j) as f32;
            assert!(
                (v - expected).abs() < 1e-3,
                "entry {j} was {v}, expected {expected}"
            );
        }

        // Mean over all 100 steps = (1 + ... + 100) / 100 = 50.5, NOT the
        // mean of the retained window (98.5).
        let stats = monitor.stats();
        assert!(
            (stats.mean_grad_norm - 50.5).abs() < 1e-2,
            "mean_grad_norm must cover every step, got {}",
            stats.mean_grad_norm
        );
        assert_eq!(stats.total_steps, 100);
    }

    #[test]
    fn test_monitor_zero_grad_norm_window_is_rejected() {
        let config = make_config(2, GradNormalization::MeanOverSteps);
        let err = AccumulationMonitor::with_grad_norm_window(config, 0)
            .expect_err("zero window must be rejected");
        assert!(matches!(err, AccumulationError::InvalidConfig(_)));
    }

    // ── Regression (F285): batch_sizes feeds real window statistics ──────────
    // The field was pushed and cleared but never read by anything.
    #[test]
    fn test_window_batch_size_statistics() {
        let config = AccumulationConfig {
            accumulation_steps: 3,
            normalization: GradNormalization::TotalBatchSize,
            auto_clear: false,
        };
        let mut acc = GradientAccumulator::new(config).expect("new");
        acc.initialize(&[1]).expect("init");
        assert!(acc.window_batch_sizes().is_empty());
        assert!(acc.mean_batch_size().is_none());
        assert!(acc.batch_size_imbalance().is_none());

        acc.accumulate(&[vec![1.0_f32]], 2).expect("s1");
        acc.accumulate(&[vec![1.0_f32]], 4).expect("s2");
        acc.accumulate(&[vec![1.0_f32]], 6).expect("s3");

        assert_eq!(acc.window_batch_sizes(), &[2, 4, 6]);
        assert_eq!(acc.window_total_samples(), 12);
        let mean = acc.mean_batch_size().expect("window is non-empty");
        assert!((mean - 4.0).abs() < 1e-6, "mean batch size was {mean}");
        let imbalance = acc.batch_size_imbalance().expect("window is non-empty");
        assert!((imbalance - 3.0).abs() < 1e-6, "imbalance was {imbalance}");

        let status = acc.format_status();
        assert!(status.contains("samples=12"), "status: {status}");
        assert!(status.contains("mean_batch=4.00"), "status: {status}");
        assert!(status.contains("imbalance=3.00"), "status: {status}");

        // Clearing the window resets the statistics with it.
        acc.clear();
        assert!(acc.window_batch_sizes().is_empty());
        assert!(acc.mean_batch_size().is_none());
        assert!(!acc.format_status().contains("mean_batch"));
    }

    #[test]
    fn test_batch_size_imbalance_none_on_zero_sized_micro_batch() {
        let mut acc = make_acc(2, GradNormalization::MeanOverSteps);
        acc.initialize(&[1]).expect("init");
        acc.accumulate(&[vec![1.0_f32]], 0).expect("s1");
        acc.accumulate(&[vec![1.0_f32]], 4).expect("s2");
        assert!(
            acc.batch_size_imbalance().is_none(),
            "a zero-sized micro-batch has no finite imbalance ratio"
        );
        // The mean is still well defined.
        let mean = acc.mean_batch_size().expect("window is non-empty");
        assert!((mean - 2.0).abs() < 1e-6, "mean batch size was {mean}");
    }

    #[test]
    fn test_monitor_without_scheduler_has_none() {
        let config = make_config(4, GradNormalization::MeanOverSteps);
        let monitor = AccumulationMonitor::new(config).expect("new");
        assert!(monitor.scheduler().is_none());
    }
}
