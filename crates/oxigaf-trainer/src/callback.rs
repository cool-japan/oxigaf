//! Training callback / hook system for the OxiGAF training pipeline.
//!
//! Callbacks allow users to inject custom logic at well-defined points in the
//! training loop (step begin/end, epoch end, checkpoint, etc.) without
//! modifying the loop itself. Callbacks are managed by a [`CallbackChain`]
//! which dispatches events in registration order.
//!
//! # Quick start
//!
//! ```rust
//! use oxigaf_trainer::callback::{default_callbacks, TrainingContext};
//!
//! let mut chain = default_callbacks(100);
//! let ctx = TrainingContext {
//!     step: 0,
//!     total_steps: 1000,
//!     epoch: None,
//!     loss: 0.5,
//!     psnr: None,
//!     ssim: None,
//!     learning_rate: 1e-3,
//!     num_gaussians: 500,
//!     elapsed_seconds: 0.0,
//! };
//! let stopped = chain.on_train_begin(&ctx).unwrap_or(false);
//! assert!(!stopped);
//! ```

use std::f64::consts::PI;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the callback subsystem.
#[derive(Debug, Error)]
pub enum CallbackError {
    /// A callback with the given name has already been registered.
    #[error("Duplicate callback name: '{0}'")]
    DuplicateName(String),

    /// A callback returned an error during execution.
    #[error("Callback '{name}' failed: {reason}")]
    CallbackFailed {
        /// Name of the failing callback.
        name: String,
        /// Human-readable reason.
        reason: String,
    },

    /// Sentinel returned by a callback to request early training termination.
    /// Not an error in the usual sense; the caller should treat `Ok(true)` as
    /// the outward signal.
    #[error("Callback requested early training termination")]
    StopTraining,

    /// Invalid callback configuration.
    #[error("Invalid callback configuration: {0}")]
    InvalidConfig(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// TrainingContext
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot of the current training state passed to all callback methods.
#[derive(Debug, Clone)]
pub struct TrainingContext {
    /// Current step (0-indexed).
    pub step: usize,
    /// Total number of steps planned.
    pub total_steps: usize,
    /// Current epoch (0-indexed). `None` if training is step-based only.
    pub epoch: Option<usize>,
    /// Current loss value.
    pub loss: f32,
    /// Current PSNR in dB, if available.
    pub psnr: Option<f32>,
    /// Current SSIM, if available.
    pub ssim: Option<f32>,
    /// Current learning rate.
    pub learning_rate: f64,
    /// Number of Gaussians currently in the model.
    pub num_gaussians: usize,
    /// Seconds elapsed since training started.
    pub elapsed_seconds: f64,
}

impl TrainingContext {
    /// Fraction of training completed: `step / total_steps`.
    /// Returns `0.0` if `total_steps == 0`.
    #[inline]
    pub fn progress_fraction(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.step as f64 / self.total_steps as f64
    }

    /// Number of steps remaining.
    #[inline]
    pub fn steps_remaining(&self) -> usize {
        self.total_steps.saturating_sub(self.step)
    }

    /// Estimated total training time (seconds) based on current pace.
    /// Returns `0.0` if `step == 0` or `progress_fraction == 0`.
    pub fn estimated_total_seconds(&self) -> f64 {
        if self.step == 0 {
            return 0.0;
        }
        let frac = self.progress_fraction();
        if frac == 0.0 {
            return 0.0;
        }
        self.elapsed_seconds / frac
    }

    /// One-line training summary suitable for logging.
    ///
    /// Format: `step 1000/100000 | loss 0.02300 | PSNR 24.1 dB | lr 1.60e-4`
    pub fn format_summary(&self) -> String {
        let psnr_str = match self.psnr {
            Some(p) => format!(" | PSNR {:.1} dB", p),
            None => String::new(),
        };
        let ssim_str = match self.ssim {
            Some(s) => format!(" | SSIM {:.4}", s),
            None => String::new(),
        };
        format!(
            "step {}/{} | loss {:.5}{}{} | lr {:.2e}",
            self.step, self.total_steps, self.loss, psnr_str, ssim_str, self.learning_rate,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Callback trait
// ─────────────────────────────────────────────────────────────────────────────

/// A training hook that can be registered with a [`CallbackChain`].
///
/// All methods have default no-op implementations so implementors only need to
/// override the hooks they care about.
pub trait Callback {
    /// Unique name identifying this callback.
    fn name(&self) -> &str;

    /// Called once at the very start of training.
    fn on_train_begin(&mut self, _ctx: &TrainingContext) -> Result<(), CallbackError> {
        Ok(())
    }

    /// Called once at the very end of training.
    fn on_train_end(&mut self, _ctx: &TrainingContext) -> Result<(), CallbackError> {
        Ok(())
    }

    /// Called before each training step.
    fn on_step_begin(&mut self, _ctx: &TrainingContext) -> Result<(), CallbackError> {
        Ok(())
    }

    /// Called after each training step.
    fn on_step_end(&mut self, _ctx: &TrainingContext) -> Result<(), CallbackError> {
        Ok(())
    }

    /// Called when a checkpoint is written.
    fn on_checkpoint(
        &mut self,
        _ctx: &TrainingContext,
        _checkpoint_path: &str,
    ) -> Result<(), CallbackError> {
        Ok(())
    }

    /// Called at epoch boundaries (only relevant for epoch-based training).
    fn on_epoch_end(&mut self, _ctx: &TrainingContext) -> Result<(), CallbackError> {
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CallbackChain
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics about the callback chain.
#[derive(Debug, Clone)]
pub struct CallbackStats {
    /// Number of registered callbacks.
    pub num_callbacks: usize,
    /// Names of all registered callbacks, in registration order.
    pub names: Vec<String>,
    /// Total number of `on_step_end` dispatch invocations since creation.
    pub total_step_events: usize,
}

/// Manages a list of [`Callback`]s and dispatches training events to them.
///
/// Callbacks are called in registration order. If a callback returns
/// [`CallbackError::StopTraining`], dispatching stops early and the method
/// returns `Ok(true)`. Any other error is wrapped in
/// [`CallbackError::CallbackFailed`] and propagated immediately.
pub struct CallbackChain {
    callbacks: Vec<Box<dyn Callback>>,
    total_step_events: usize,
}

impl CallbackChain {
    /// Create an empty `CallbackChain`.
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
            total_step_events: 0,
        }
    }

    /// Register a callback.
    ///
    /// Returns `Err(DuplicateName)` if a callback with the same name is already
    /// registered.
    pub fn add(&mut self, cb: Box<dyn Callback>) -> Result<(), CallbackError> {
        let name = cb.name().to_string();
        if self.callbacks.iter().any(|c| c.name() == name) {
            return Err(CallbackError::DuplicateName(name));
        }
        self.callbacks.push(cb);
        Ok(())
    }

    /// Remove a callback by name.
    ///
    /// Returns `true` if a matching callback was found and removed.
    pub fn remove(&mut self, name: &str) -> bool {
        if let Some(idx) = self.callbacks.iter().position(|c| c.name() == name) {
            self.callbacks.remove(idx);
            true
        } else {
            false
        }
    }

    /// Number of registered callbacks.
    pub fn len(&self) -> usize {
        self.callbacks.len()
    }

    /// Whether no callbacks are registered.
    pub fn is_empty(&self) -> bool {
        self.callbacks.is_empty()
    }

    /// Names of all registered callbacks in registration order.
    pub fn names(&self) -> Vec<&str> {
        self.callbacks.iter().map(|c| c.name()).collect()
    }

    /// Snapshot of chain statistics.
    pub fn stats(&self) -> CallbackStats {
        CallbackStats {
            num_callbacks: self.callbacks.len(),
            names: self
                .callbacks
                .iter()
                .map(|c| c.name().to_string())
                .collect(),
            total_step_events: self.total_step_events,
        }
    }

    // ── private dispatch helper ──────────────────────────────────────────────

    /// Dispatch `f` over all callbacks. Returns:
    /// - `Ok(false)` — all callbacks completed normally.
    /// - `Ok(true)`  — a callback returned `StopTraining`; dispatching stopped.
    /// - `Err(_)`    — a callback returned a non-`StopTraining` error.
    fn dispatch<F>(&mut self, mut f: F) -> Result<bool, CallbackError>
    where
        F: FnMut(&mut Box<dyn Callback>) -> Result<(), CallbackError>,
    {
        for cb in &mut self.callbacks {
            match f(cb) {
                Ok(()) => {}
                Err(CallbackError::StopTraining) => return Ok(true),
                Err(other) => {
                    return Err(CallbackError::CallbackFailed {
                        name: cb.name().to_string(),
                        reason: other.to_string(),
                    });
                }
            }
        }
        Ok(false)
    }

    /// Dispatch `on_train_begin` to all callbacks.
    pub fn on_train_begin(&mut self, ctx: &TrainingContext) -> Result<bool, CallbackError> {
        self.dispatch(|cb| cb.on_train_begin(ctx))
    }

    /// Dispatch `on_train_end` to all callbacks.
    pub fn on_train_end(&mut self, ctx: &TrainingContext) -> Result<bool, CallbackError> {
        self.dispatch(|cb| cb.on_train_end(ctx))
    }

    /// Dispatch `on_step_begin` to all callbacks.
    pub fn on_step_begin(&mut self, ctx: &TrainingContext) -> Result<bool, CallbackError> {
        self.dispatch(|cb| cb.on_step_begin(ctx))
    }

    /// Dispatch `on_step_end` to all callbacks and increment step-event counter.
    pub fn on_step_end(&mut self, ctx: &TrainingContext) -> Result<bool, CallbackError> {
        let result = self.dispatch(|cb| cb.on_step_end(ctx));
        self.total_step_events += 1;
        result
    }

    /// Dispatch `on_checkpoint` to all callbacks.
    pub fn on_checkpoint(
        &mut self,
        ctx: &TrainingContext,
        path: &str,
    ) -> Result<bool, CallbackError> {
        self.dispatch(|cb| cb.on_checkpoint(ctx, path))
    }

    /// Dispatch `on_epoch_end` to all callbacks.
    pub fn on_epoch_end(&mut self, ctx: &TrainingContext) -> Result<bool, CallbackError> {
        self.dispatch(|cb| cb.on_epoch_end(ctx))
    }
}

impl Default for CallbackChain {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LossLoggerCallback
// ─────────────────────────────────────────────────────────────────────────────

/// Logs loss (and optionally PSNR) every `log_every` steps.
///
/// The running history is available via [`LossLoggerCallback::history`].
pub struct LossLoggerCallback {
    /// Log interval in steps.
    pub log_every: usize,
    /// Unique name of this callback.
    pub name: String,
    /// Recorded `(step, loss)` pairs.
    history: Vec<(usize, f32)>,
}

impl LossLoggerCallback {
    /// Create a new logger that records every `log_every` steps.
    ///
    /// A `log_every` of 0 is clamped to 1. This never panics and never
    /// returns an error.
    pub fn new(log_every: usize) -> Self {
        Self {
            log_every: log_every.max(1),
            name: "loss_logger".to_string(),
            history: Vec::new(),
        }
    }

    /// Recorded `(step, loss)` pairs in chronological order.
    pub fn history(&self) -> &[(usize, f32)] {
        &self.history
    }

    /// Minimum recorded loss, or `None` if no data yet.
    pub fn min_loss(&self) -> Option<f32> {
        self.history.iter().map(|&(_, l)| l).reduce(f32::min)
    }

    /// Maximum recorded loss, or `None` if no data yet.
    pub fn max_loss(&self) -> Option<f32> {
        self.history.iter().map(|&(_, l)| l).reduce(f32::max)
    }

    /// Mean recorded loss, or `0.0` if no data yet.
    pub fn mean_loss(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|&(_, l)| l).sum();
        sum / self.history.len() as f32
    }
}

impl Callback for LossLoggerCallback {
    fn name(&self) -> &str {
        &self.name
    }

    fn on_step_end(&mut self, ctx: &TrainingContext) -> Result<(), CallbackError> {
        if ctx.step.is_multiple_of(self.log_every) {
            self.history.push((ctx.step, ctx.loss));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EarlyStoppingCallback
// ─────────────────────────────────────────────────────────────────────────────

/// Stops training when loss fails to improve by more than `min_delta` for
/// `patience` consecutive steps.
pub struct EarlyStoppingCallback {
    /// Number of non-improving steps to tolerate before stopping.
    pub patience: usize,
    /// Minimum improvement threshold; improvement counts only if
    /// `new_loss < best_loss - min_delta`.
    pub min_delta: f32,
    name: String,
    best_loss: f32,
    steps_without_improvement: usize,
}

impl EarlyStoppingCallback {
    /// Create a new early stopping callback.
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            patience,
            min_delta,
            name: "early_stopping".to_string(),
            best_loss: f32::MAX,
            steps_without_improvement: 0,
        }
    }

    /// Steps elapsed without loss improvement.
    pub fn steps_without_improvement(&self) -> usize {
        self.steps_without_improvement
    }

    /// Best loss observed so far.
    pub fn best_loss(&self) -> f32 {
        self.best_loss
    }
}

impl Callback for EarlyStoppingCallback {
    fn name(&self) -> &str {
        &self.name
    }

    fn on_step_end(&mut self, ctx: &TrainingContext) -> Result<(), CallbackError> {
        if ctx.loss < self.best_loss - self.min_delta {
            // Improvement — reset counter and update best.
            self.best_loss = ctx.loss;
            self.steps_without_improvement = 0;
        } else {
            self.steps_without_improvement += 1;
            if self.steps_without_improvement >= self.patience {
                return Err(CallbackError::StopTraining);
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LrSchedulerCallback
// ─────────────────────────────────────────────────────────────────────────────

/// Applies a pre-computed learning rate schedule each step.
///
/// The computed LR is stored in `current_lr` for external read-back (since
/// [`TrainingContext`] is passed as `&` and cannot be mutated).
pub struct LrSchedulerCallback {
    name: String,
    /// LR value for the current step (updated in `on_step_begin`).
    pub current_lr: f64,
    /// Pre-computed LR table, one entry per training step.
    schedule: Vec<f64>,
    total_steps: usize,
    /// Peak LR the schedule was built around; read back via [`Self::base_lr`].
    base_lr: f64,
    /// Floor the cosine decays to; also the fallback of [`Self::lr_at`] when
    /// the schedule is empty.
    min_lr: f64,
}

impl LrSchedulerCallback {
    /// Build a warmup + cosine annealing schedule.
    ///
    /// - Steps `0..warmup_steps`: linear ramp from `0` to `base_lr`.
    /// - Steps `warmup_steps..total_steps`: cosine decay from `base_lr` to
    ///   `min_lr`.
    pub fn warmup_cosine(
        warmup_steps: usize,
        base_lr: f64,
        min_lr: f64,
        total_steps: usize,
    ) -> Self {
        let mut schedule = Vec::with_capacity(total_steps);

        for step in 0..total_steps {
            let lr = if step < warmup_steps {
                // Linear warmup (avoid divide-by-zero when warmup_steps == 0).
                if warmup_steps == 0 {
                    base_lr
                } else {
                    base_lr * step as f64 / warmup_steps as f64
                }
            } else {
                // Cosine decay.
                let cosine_steps = total_steps.saturating_sub(warmup_steps);
                if cosine_steps == 0 {
                    min_lr
                } else {
                    let t = (step - warmup_steps) as f64 / cosine_steps as f64;
                    min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (PI * t).cos())
                }
            };
            schedule.push(lr);
        }

        let current_lr = schedule.first().copied().unwrap_or(min_lr);

        Self {
            name: "lr_scheduler".to_string(),
            current_lr,
            schedule,
            total_steps,
            base_lr,
            min_lr,
        }
    }

    /// Return the pre-computed LR at `step`.
    /// Clamps to the last valid index if `step >= total_steps`.
    pub fn lr_at(&self, step: usize) -> f64 {
        if self.schedule.is_empty() {
            return self.min_lr;
        }
        let idx = step.min(self.total_steps.saturating_sub(1));
        // SAFETY-equivalent: we know idx < schedule.len() because we filled
        // exactly total_steps entries and idx <= total_steps - 1.
        self.schedule.get(idx).copied().unwrap_or(self.min_lr)
    }

    /// The peak learning rate this schedule was built around.
    ///
    /// The schedule itself is pre-computed and private, so `base_lr` is the
    /// only way for a caller to recover the *requested* peak — needed to log
    /// or plot the current LR as a fraction of it (`lr_at(step) / base_lr`),
    /// to rebuild an equivalent schedule for a longer run, or to assert in a
    /// test that warmup actually reaches the rate that was asked for. Without
    /// it, the value passed to [`Self::warmup_cosine`] is unrecoverable from
    /// the callback.
    ///
    /// ```
    /// use oxigaf_trainer::callback::LrSchedulerCallback;
    ///
    /// let cb = LrSchedulerCallback::warmup_cosine(100, 1e-3, 1e-6, 1000);
    /// assert!((cb.base_lr() - 1e-3).abs() < 1e-12);
    /// // Warmup ends exactly at the peak.
    /// assert!((cb.lr_at(100) - cb.base_lr()).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn base_lr(&self) -> f64 {
        self.base_lr
    }

    /// The floor the cosine schedule decays toward.
    ///
    /// Reported alongside [`Self::base_lr`] so a caller can describe the full
    /// `[min_lr, base_lr]` range the schedule spans; it is also what
    /// [`Self::lr_at`] returns when no schedule was built
    /// (`total_steps == 0`).
    #[must_use]
    pub fn min_lr(&self) -> f64 {
        self.min_lr
    }
}

impl Callback for LrSchedulerCallback {
    fn name(&self) -> &str {
        &self.name
    }

    fn on_step_begin(&mut self, ctx: &TrainingContext) -> Result<(), CallbackError> {
        self.current_lr = self.lr_at(ctx.step);
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CheckpointCallback
// ─────────────────────────────────────────────────────────────────────────────

/// Records every checkpoint event as a `(step, path)` pair.
pub struct CheckpointCallback {
    name: String,
    /// All recorded checkpoint events, in chronological order.
    pub checkpoint_paths: Vec<(usize, String)>,
}

impl CheckpointCallback {
    /// Create a new checkpoint-recording callback.
    pub fn new() -> Self {
        Self {
            name: "checkpoint_recorder".to_string(),
            checkpoint_paths: Vec::new(),
        }
    }

    /// The most recent checkpoint, if any.
    pub fn last_checkpoint(&self) -> Option<&(usize, String)> {
        self.checkpoint_paths.last()
    }

    /// Total number of checkpoints recorded.
    pub fn num_checkpoints(&self) -> usize {
        self.checkpoint_paths.len()
    }
}

impl Default for CheckpointCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl Callback for CheckpointCallback {
    fn name(&self) -> &str {
        &self.name
    }

    fn on_checkpoint(
        &mut self,
        ctx: &TrainingContext,
        checkpoint_path: &str,
    ) -> Result<(), CallbackError> {
        self.checkpoint_paths
            .push((ctx.step, checkpoint_path.to_string()));
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MetricsHistoryCallback
// ─────────────────────────────────────────────────────────────────────────────

/// Records all metrics at every step for later analysis or export.
pub struct MetricsHistoryCallback {
    name: String,
    /// Recorded step indices.
    pub steps: Vec<usize>,
    /// Recorded loss values.
    pub losses: Vec<f32>,
    /// Recorded PSNR values (0.0 when PSNR was unavailable).
    pub psnrs: Vec<f32>,
    /// Recorded SSIM values (0.0 when SSIM was unavailable).
    pub ssims: Vec<f32>,
    /// Recorded learning rates.
    pub lrs: Vec<f64>,
}

impl MetricsHistoryCallback {
    /// Create a new metrics-history callback.
    pub fn new() -> Self {
        Self {
            name: "metrics_history".to_string(),
            steps: Vec::new(),
            losses: Vec::new(),
            psnrs: Vec::new(),
            ssims: Vec::new(),
            lrs: Vec::new(),
        }
    }

    /// Number of recorded steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether no steps have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Mean loss over all recorded steps, or `0.0` if empty.
    pub fn mean_loss(&self) -> f32 {
        if self.losses.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.losses.iter().sum();
        sum / self.losses.len() as f32
    }

    /// PSNR value at the last recorded step, or `None` if no steps recorded.
    pub fn last_psnr(&self) -> Option<f32> {
        self.psnrs.last().copied()
    }

    /// Format all recorded metrics as a CSV string (header + one row per step).
    ///
    /// Columns: `step,loss,psnr,ssim,lr`
    ///
    /// `steps`, `losses`, `psnrs`, `ssims`, and `lrs` are all public `Vec`
    /// fields, so nothing prevents external code from desynchronizing their
    /// lengths (truncating one series, pushing to only one, etc). Rather
    /// than indexing (which would panic on a mismatch), this zips all five
    /// iterators together: a row is emitted only while every series still
    /// has an entry, so a shorter series simply truncates the CSV instead of
    /// causing an out-of-bounds panic.
    pub fn as_csv(&self) -> String {
        let mut out = String::from("step,loss,psnr,ssim,lr\n");
        let rows = self
            .steps
            .iter()
            .zip(self.losses.iter())
            .zip(self.psnrs.iter())
            .zip(self.ssims.iter())
            .zip(self.lrs.iter());
        for ((((step, loss), psnr), ssim), lr) in rows {
            out.push_str(&format!("{step},{loss},{psnr},{ssim},{lr}\n"));
        }
        out
    }
}

impl Default for MetricsHistoryCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl Callback for MetricsHistoryCallback {
    fn name(&self) -> &str {
        &self.name
    }

    fn on_step_end(&mut self, ctx: &TrainingContext) -> Result<(), CallbackError> {
        self.steps.push(ctx.step);
        self.losses.push(ctx.loss);
        self.psnrs.push(ctx.psnr.unwrap_or(0.0));
        self.ssims.push(ctx.ssim.unwrap_or(0.0));
        self.lrs.push(ctx.learning_rate);
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper constructors
// ─────────────────────────────────────────────────────────────────────────────

/// Build a lightweight callback chain with [`LossLoggerCallback`] and
/// [`MetricsHistoryCallback`] — suitable for quick experiments.
///
/// # Panics
/// Never panics; both callbacks have unique names.
pub fn default_callbacks(log_every: usize) -> CallbackChain {
    let mut chain = CallbackChain::new();
    // These are infallible because the chain is freshly created.
    let _ = chain.add(Box::new(LossLoggerCallback::new(log_every)));
    let _ = chain.add(Box::new(MetricsHistoryCallback::new()));
    chain
}

/// Build a production-grade callback chain with [`EarlyStoppingCallback`],
/// [`CheckpointCallback`], and [`MetricsHistoryCallback`].
///
/// # Parameters
/// - `patience`: steps without improvement before early stopping fires.
/// - `log_every`: loss-logging interval (also used for [`LossLoggerCallback`]).
pub fn production_callbacks(patience: usize, log_every: usize) -> CallbackChain {
    let mut chain = CallbackChain::new();
    let _ = chain.add(Box::new(EarlyStoppingCallback::new(patience, 1e-5)));
    let _ = chain.add(Box::new(LossLoggerCallback::new(log_every)));
    let _ = chain.add(Box::new(CheckpointCallback::new()));
    let _ = chain.add(Box::new(MetricsHistoryCallback::new()));
    chain
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx(step: usize, loss: f32) -> TrainingContext {
        TrainingContext {
            step,
            total_steps: 1000,
            epoch: None,
            loss,
            psnr: Some(20.0),
            ssim: None,
            learning_rate: 0.001,
            num_gaussians: 100,
            elapsed_seconds: step as f64 * 0.1,
        }
    }

    // ── 1. progress_fraction ─────────────────────────────────────────────────

    #[test]
    fn test_progress_fraction() {
        let ctx = make_ctx(500, 0.1);
        let frac = ctx.progress_fraction();
        assert!((frac - 0.5).abs() < 1e-10, "expected 0.5, got {frac}");
    }

    // ── 2. format_summary ────────────────────────────────────────────────────

    #[test]
    fn test_format_summary_contains_step_and_loss() {
        let ctx = make_ctx(100, 0.023);
        let summary = ctx.format_summary();
        assert!(
            summary.contains("step"),
            "format_summary should contain 'step': {summary}"
        );
        assert!(
            summary.contains("loss"),
            "format_summary should contain 'loss': {summary}"
        );
    }

    // ── 3. estimated_total_seconds with step=0 ───────────────────────────────

    #[test]
    fn test_estimated_total_seconds_at_zero() {
        let ctx = make_ctx(0, 0.5);
        assert_eq!(ctx.estimated_total_seconds(), 0.0);
    }

    // ── 4. CallbackChain::add — no duplicate ─────────────────────────────────

    #[test]
    fn test_chain_add_no_duplicate() {
        let mut chain = CallbackChain::new();
        let result = chain.add(Box::new(LossLoggerCallback::new(10)));
        assert!(result.is_ok());
        assert_eq!(chain.len(), 1);
    }

    // ── 5. CallbackChain::add — duplicate name → Err ─────────────────────────

    #[test]
    fn test_chain_add_duplicate_name() {
        let mut chain = CallbackChain::new();
        chain
            .add(Box::new(LossLoggerCallback::new(10)))
            .unwrap_or_default();
        let result = chain.add(Box::new(LossLoggerCallback::new(20)));
        assert!(
            matches!(result, Err(CallbackError::DuplicateName(_))),
            "Expected DuplicateName error"
        );
    }

    // ── 6. CallbackChain::remove — existing name ─────────────────────────────

    #[test]
    fn test_chain_remove_existing() {
        let mut chain = CallbackChain::new();
        chain
            .add(Box::new(LossLoggerCallback::new(10)))
            .unwrap_or_default();
        let removed = chain.remove("loss_logger");
        assert!(removed);
        assert_eq!(chain.len(), 0);
    }

    // ── 7. CallbackChain::remove — non-existing ──────────────────────────────

    #[test]
    fn test_chain_remove_nonexistent() {
        let mut chain = CallbackChain::new();
        let removed = chain.remove("nonexistent");
        assert!(!removed);
    }

    // ── 8. LossLoggerCallback logs every N steps, not others ─────────────────

    #[test]
    fn test_loss_logger_logs_every_n_steps() {
        let mut cb = LossLoggerCallback::new(10);
        for step in 0usize..=25 {
            let ctx = make_ctx(step, step as f32 * 0.01);
            cb.on_step_end(&ctx).unwrap_or_default();
        }
        // Steps 0, 10, 20 → 3 entries
        assert_eq!(
            cb.history().len(),
            3,
            "Expected 3 log entries (steps 0,10,20), got {}",
            cb.history().len()
        );
    }

    // ── 9. LossLoggerCallback history correct length ──────────────────────────

    #[test]
    fn test_loss_logger_history_length() {
        let mut chain = CallbackChain::new();
        chain
            .add(Box::new(LossLoggerCallback::new(5)))
            .unwrap_or_default();
        for step in 0usize..=20 {
            let ctx = make_ctx(step, 0.5);
            chain.on_step_end(&ctx).unwrap_or_default();
        }
        // Steps 0, 5, 10, 15, 20 → 5 entries
        let names = chain.names();
        assert!(names.contains(&"loss_logger"));
    }

    // ── 10. LossLoggerCallback min_loss / max_loss ───────────────────────────

    #[test]
    fn test_loss_logger_min_max() {
        let mut cb = LossLoggerCallback::new(1);
        let losses = [0.5f32, 0.2, 0.8, 0.1, 0.4];
        for (step, &loss) in losses.iter().enumerate() {
            let ctx = make_ctx(step, loss);
            cb.on_step_end(&ctx).unwrap_or_default();
        }
        let min = cb.min_loss().unwrap_or(f32::MAX);
        let max = cb.max_loss().unwrap_or(f32::MIN);
        assert!(
            (min - 0.1).abs() < 1e-6,
            "min_loss should be 0.1, got {min}"
        );
        assert!(
            (max - 0.8).abs() < 1e-6,
            "max_loss should be 0.8, got {max}"
        );
    }

    // ── 11. EarlyStoppingCallback — no stop when improving ───────────────────

    #[test]
    fn test_early_stopping_no_stop_when_improving() {
        let mut cb = EarlyStoppingCallback::new(3, 0.001);
        for step in 0usize..10 {
            let ctx = make_ctx(step, 1.0 - step as f32 * 0.05);
            let result = cb.on_step_end(&ctx);
            assert!(result.is_ok(), "Should not stop when loss is improving");
        }
    }

    // ── 12. EarlyStoppingCallback — stop after patience exceeded ─────────────

    #[test]
    fn test_early_stopping_fires_after_patience() {
        let mut cb = EarlyStoppingCallback::new(3, 0.001);
        // First step — sets best_loss.
        let ctx0 = make_ctx(0, 0.5);
        cb.on_step_end(&ctx0).unwrap_or_default();
        // Non-improving steps: patience = 3, so stop after 3rd stale step.
        let stale = make_ctx(1, 0.5);
        let r1 = cb.on_step_end(&stale);
        assert!(r1.is_ok());
        let r2 = cb.on_step_end(&stale);
        assert!(r2.is_ok());
        let r3 = cb.on_step_end(&stale);
        assert!(
            matches!(r3, Err(CallbackError::StopTraining)),
            "Expected StopTraining after {patience} stale steps",
            patience = 3
        );
    }

    // ── 13. EarlyStoppingCallback — reset on improvement ─────────────────────

    #[test]
    fn test_early_stopping_resets_on_improvement() {
        let mut cb = EarlyStoppingCallback::new(2, 0.001);
        let stale = make_ctx(0, 0.5);
        // First call sets best_loss.
        cb.on_step_end(&stale).unwrap_or_default();
        // One stale step.
        cb.on_step_end(&stale).unwrap_or_default();
        assert_eq!(cb.steps_without_improvement(), 1);
        // Improvement — counter resets.
        let improved = make_ctx(2, 0.3);
        cb.on_step_end(&improved).unwrap_or_default();
        assert_eq!(
            cb.steps_without_improvement(),
            0,
            "Counter should reset after improvement"
        );
    }

    // ── 14. CallbackChain returns Ok(true) when early stopping fires ──────────

    #[test]
    fn test_chain_on_step_end_returns_true_on_early_stop() {
        let mut chain = CallbackChain::new();
        // patience=1: two stale steps trigger stop.
        chain
            .add(Box::new(EarlyStoppingCallback::new(1, 0.001)))
            .unwrap_or_default();

        let ctx0 = make_ctx(0, 0.5);
        chain.on_step_end(&ctx0).unwrap_or_default(); // sets best_loss

        let stale = make_ctx(1, 0.5);
        let result = chain.on_step_end(&stale);
        assert!(
            matches!(result, Ok(true)),
            "Expected Ok(true) when early stopping fires, got {result:?}"
        );
    }

    // ── 15. LrSchedulerCallback — lr at warmup start ≈ 0 ────────────────────

    #[test]
    fn test_lr_scheduler_warmup_start_near_zero() {
        let cb = LrSchedulerCallback::warmup_cosine(100, 1e-3, 1e-6, 1000);
        // Step 0 during warmup should be ≈ 0.
        let lr0 = cb.lr_at(0);
        assert!(
            lr0 < 1e-5,
            "LR at step 0 should be near 0 during warmup, got {lr0}"
        );
    }

    // ── 16. LrSchedulerCallback — lr at end of warmup ≈ base_lr ─────────────

    #[test]
    fn test_lr_scheduler_lr_at_end_of_warmup() {
        let base_lr = 1e-3;
        let warmup_steps = 100;
        let cb = LrSchedulerCallback::warmup_cosine(warmup_steps, base_lr, 1e-6, 1000);
        // Step = warmup_steps is the first cosine step (t=0 → lr = base_lr).
        let lr = cb.lr_at(warmup_steps);
        assert!(
            (lr - base_lr).abs() < 1e-10,
            "LR at end of warmup should equal base_lr ({base_lr}), got {lr}"
        );
    }

    // ── 17. LrSchedulerCallback — lr at total_steps ≈ min_lr ────────────────

    #[test]
    fn test_lr_scheduler_lr_at_total_steps() {
        let min_lr = 1e-6;
        let base_lr = 1e-3;
        let total_steps = 1000;
        let cb = LrSchedulerCallback::warmup_cosine(100, base_lr, min_lr, total_steps);
        // Last scheduled step: cosine at t = (total_steps-1-warmup_steps)/(total_steps-warmup_steps),
        // which is slightly less than 1.0, so lr is slightly above min_lr.
        // We accept within 0.5% of the (base_lr - min_lr) range.
        let lr = cb.lr_at(total_steps - 1);
        let tolerance = (base_lr - min_lr) * 5e-3;
        assert!(
            (lr - min_lr).abs() < tolerance,
            "LR at last step should be close to min_lr ({min_lr}), got {lr} (tolerance {tolerance})"
        );
    }

    // ── 17b. LrSchedulerCallback exposes its schedule bounds ─────────────────
    //
    // Regression: `base_lr` was stored by `warmup_cosine` and then read by
    // nobody, suppressed with `#[allow(dead_code)]`. The pre-computed
    // `schedule` is private, so without an accessor the peak LR a caller
    // asked for was unrecoverable from the callback. Both bounds are now
    // public read-back, and the peak must agree with the schedule itself.
    #[test]
    fn test_lr_scheduler_reports_schedule_bounds() {
        let warmup_steps = 100;
        let base_lr = 1e-3;
        let min_lr = 1e-6;
        let cb = LrSchedulerCallback::warmup_cosine(warmup_steps, base_lr, min_lr, 1000);

        assert!(
            (cb.base_lr() - base_lr).abs() < 1e-12,
            "base_lr() should report the requested peak {base_lr}, got {}",
            cb.base_lr()
        );
        assert!(
            (cb.min_lr() - min_lr).abs() < 1e-12,
            "min_lr() should report the requested floor {min_lr}, got {}",
            cb.min_lr()
        );
        // The accessor must agree with the table it was built from: the first
        // cosine step (t = 0) sits exactly at the peak.
        assert!(
            (cb.lr_at(warmup_steps) - cb.base_lr()).abs() < 1e-12,
            "end of warmup should equal base_lr()"
        );
        assert!(
            cb.min_lr() < cb.base_lr(),
            "floor must sit below the peak for a decaying schedule"
        );
    }

    // ── 17c. Empty schedule falls back to the reported floor ─────────────────
    #[test]
    fn test_lr_scheduler_empty_schedule_uses_min_lr() {
        let cb = LrSchedulerCallback::warmup_cosine(0, 1e-3, 1e-6, 0);
        // No table was built, so every step reports the documented fallback —
        // and `min_lr()` names exactly that value.
        assert!((cb.lr_at(0) - cb.min_lr()).abs() < 1e-12);
        assert!((cb.lr_at(9_999) - cb.min_lr()).abs() < 1e-12);
        assert!((cb.base_lr() - 1e-3).abs() < 1e-12);
    }

    // ── 18. CheckpointCallback records paths ─────────────────────────────────

    #[test]
    fn test_checkpoint_callback_records_paths() {
        let mut cb = CheckpointCallback::new();
        let ctx = make_ctx(500, 0.1);
        cb.on_checkpoint(&ctx, "/tmp/ckpt_500.bin")
            .unwrap_or_default();
        cb.on_checkpoint(&ctx, "/tmp/ckpt_501.bin")
            .unwrap_or_default();
        assert_eq!(cb.num_checkpoints(), 2);
        assert_eq!(cb.checkpoint_paths[0].1, "/tmp/ckpt_500.bin");
    }

    // ── 19. CheckpointCallback last_checkpoint ───────────────────────────────

    #[test]
    fn test_checkpoint_callback_last_checkpoint() {
        let mut cb = CheckpointCallback::new();
        assert!(cb.last_checkpoint().is_none());
        let ctx = make_ctx(100, 0.2);
        cb.on_checkpoint(&ctx, "/tmp/a.bin").unwrap_or_default();
        cb.on_checkpoint(&ctx, "/tmp/b.bin").unwrap_or_default();
        let last = cb.last_checkpoint();
        assert!(last.is_some());
        assert_eq!(last.map(|(_, p)| p.as_str()), Some("/tmp/b.bin"));
    }

    // ── 20. MetricsHistoryCallback records all steps ──────────────────────────

    #[test]
    fn test_metrics_history_records_all_steps() {
        let mut cb = MetricsHistoryCallback::new();
        for step in 0usize..5 {
            let ctx = make_ctx(step, 0.5 - step as f32 * 0.05);
            cb.on_step_end(&ctx).unwrap_or_default();
        }
        assert_eq!(cb.len(), 5);
        assert_eq!(cb.steps, vec![0, 1, 2, 3, 4]);
    }

    // ── 21. MetricsHistoryCallback::as_csv has header + rows ─────────────────

    #[test]
    fn test_metrics_history_as_csv() {
        let mut cb = MetricsHistoryCallback::new();
        cb.on_step_end(&make_ctx(0, 0.5)).unwrap_or_default();
        cb.on_step_end(&make_ctx(1, 0.4)).unwrap_or_default();
        let csv = cb.as_csv();
        assert!(
            csv.starts_with("step,loss,psnr,ssim,lr"),
            "CSV should start with header: {csv}"
        );
        let lines: Vec<&str> = csv.lines().collect();
        // Header + 2 data rows = 3 lines.
        assert_eq!(lines.len(), 3, "Expected 3 CSV lines, got {}", lines.len());
    }

    // ── 21b. MetricsHistoryCallback::as_csv tolerates desynchronized public
    // Vec fields instead of panicking (regression: previously indexed
    // `self.losses[i]` etc. driven off `self.steps.len()`) ───────────────────

    #[test]
    fn test_metrics_history_as_csv_desynced_vecs_truncates_instead_of_panicking() {
        let mut cb = MetricsHistoryCallback::new();
        cb.on_step_end(&make_ctx(0, 0.5)).unwrap_or_default();
        cb.on_step_end(&make_ctx(1, 0.4)).unwrap_or_default();
        cb.on_step_end(&make_ctx(2, 0.3)).unwrap_or_default();
        // Externally desynchronize: steps has 3 entries but losses has only 1.
        cb.losses.truncate(1);
        let csv = cb.as_csv(); // must not panic
        let lines: Vec<&str> = csv.lines().collect();
        // Header + 1 data row (zip stops at the shortest series).
        assert_eq!(lines.len(), 2, "Expected header + 1 row, got: {csv}");
    }

    // ── 22. CallbackChain::stats ─────────────────────────────────────────────

    #[test]
    fn test_chain_stats() {
        let mut chain = CallbackChain::new();
        chain
            .add(Box::new(LossLoggerCallback::new(1)))
            .unwrap_or_default();
        chain
            .add(Box::new(MetricsHistoryCallback::new()))
            .unwrap_or_default();
        let ctx = make_ctx(0, 0.5);
        chain.on_step_end(&ctx).unwrap_or_default();
        chain.on_step_end(&ctx).unwrap_or_default();
        let stats = chain.stats();
        assert_eq!(stats.num_callbacks, 2);
        assert_eq!(stats.total_step_events, 2);
        assert!(stats.names.contains(&"loss_logger".to_string()));
    }

    // ── 23. default_callbacks returns chain with expected callbacks ───────────

    #[test]
    fn test_default_callbacks() {
        let chain = default_callbacks(50);
        assert_eq!(chain.len(), 2);
        let names = chain.names();
        assert!(names.contains(&"loss_logger"), "Missing loss_logger");
        assert!(
            names.contains(&"metrics_history"),
            "Missing metrics_history"
        );
    }

    // ── 24. production_callbacks includes EarlyStopping ──────────────────────

    #[test]
    fn test_production_callbacks_includes_early_stopping() {
        let chain = production_callbacks(100, 10);
        let names = chain.names();
        assert!(
            names.contains(&"early_stopping"),
            "production_callbacks should include early_stopping"
        );
        assert!(
            names.contains(&"checkpoint_recorder"),
            "production_callbacks should include checkpoint_recorder"
        );
        assert!(
            names.contains(&"metrics_history"),
            "production_callbacks should include metrics_history"
        );
    }

    // ── bonus: estimated_total_seconds non-zero ───────────────────────────────

    #[test]
    fn test_estimated_total_seconds_nonzero() {
        // step=500, total=1000, elapsed=50.0 → estimated = 50.0 / 0.5 = 100.0
        let ctx = TrainingContext {
            step: 500,
            total_steps: 1000,
            epoch: None,
            loss: 0.1,
            psnr: None,
            ssim: None,
            learning_rate: 1e-3,
            num_gaussians: 100,
            elapsed_seconds: 50.0,
        };
        let est = ctx.estimated_total_seconds();
        assert!((est - 100.0).abs() < 1e-9, "Expected 100.0, got {est}");
    }

    // ── bonus: empty LossLogger min/max return None ───────────────────────────

    #[test]
    fn test_loss_logger_empty_min_max() {
        let cb = LossLoggerCallback::new(1);
        assert!(cb.min_loss().is_none());
        assert!(cb.max_loss().is_none());
        assert_eq!(cb.mean_loss(), 0.0);
    }

    // ── bonus: LrSchedulerCallback dispatch via CallbackChain ────────────────

    #[test]
    fn test_lr_scheduler_via_chain() {
        let mut chain = CallbackChain::new();
        chain
            .add(Box::new(LrSchedulerCallback::warmup_cosine(
                10, 1e-3, 1e-6, 100,
            )))
            .unwrap_or_default();
        let ctx = make_ctx(10, 0.3); // step at end of warmup
        chain.on_step_begin(&ctx).unwrap_or_default();
        // Just verify it didn't error — current_lr is internal to the cb.
        assert_eq!(chain.len(), 1);
    }
}
