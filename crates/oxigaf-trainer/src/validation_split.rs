//! Train/validation frame splitting and monitoring.
//!
//! Provides:
//! - [`SplitStrategy`]        — how to partition frames into train/val sets
//! - [`DataSplit`]            — the computed index partition
//! - [`ValidationTracker`]   — history of validation steps with trend analysis
//! - [`EarlyStopper`]        — patience-based early stopping
//! - [`SplitError`]          — error type for split operations

use std::fmt;
use thiserror::Error;

use oxigaf_render::gaussian::GaussianModel;

// ---------------------------------------------------------------------------
// xorshift64 PRNG (inline, matching session_recorder.rs pattern)
// ---------------------------------------------------------------------------

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn xorshift64_init(seed: u64) -> u64 {
    if seed == 0 {
        0xdeadbeef_cafebabe
    } else {
        seed
    }
}

// ---------------------------------------------------------------------------
// SplitError
// ---------------------------------------------------------------------------

/// Errors produced by [`DataSplit::from_strategy`] and related operations.
#[derive(Debug, Error)]
pub enum SplitError {
    #[error("Insufficient frames: have {total}, requested {requested}")]
    InsufficientFrames { total: usize, requested: usize },

    #[error("Invalid fraction: {fraction} (must be in (0.0, 1.0))")]
    InvalidFraction { fraction: f32 },

    #[error("EveryNth step cannot be zero")]
    ZeroStep,

    #[error("Manual index {idx} is out of range (total frames: {total})")]
    InvalidIndex { idx: usize, total: usize },

    #[error("Split would leave training set empty")]
    EmptyTrainSet,
}

// ---------------------------------------------------------------------------
// SplitStrategy
// ---------------------------------------------------------------------------

/// Strategy for partitioning frames into training and validation sets.
#[derive(Debug, Clone)]
pub enum SplitStrategy {
    /// Reserve the last N frames for validation.
    LastN(usize),
    /// Reserve a fraction of frames for validation (random selection with seed).
    RandomFraction { fraction: f32, seed: u64 },
    /// Take every Nth frame as validation.
    EveryNth(usize),
    /// Manually specify validation indices.
    Manual(Vec<usize>),
}

impl Default for SplitStrategy {
    fn default() -> Self {
        Self::LastN(5)
    }
}

// ---------------------------------------------------------------------------
// DataSplit
// ---------------------------------------------------------------------------

/// A computed partition of frame indices into training and validation sets.
#[derive(Debug, Clone)]
pub struct DataSplit {
    pub train_indices: Vec<usize>,
    pub val_indices: Vec<usize>,
    pub total_frames: usize,
}

impl DataSplit {
    /// Create a split where all frames are used for training (no validation).
    pub fn all_train(n: usize) -> Self {
        Self {
            train_indices: (0..n).collect(),
            val_indices: Vec::new(),
            total_frames: n,
        }
    }

    /// Create a split from a strategy and total frame count.
    pub fn from_strategy(
        strategy: &SplitStrategy,
        total_frames: usize,
    ) -> Result<Self, SplitError> {
        match strategy {
            SplitStrategy::LastN(n) => {
                if *n > total_frames {
                    return Err(SplitError::InsufficientFrames {
                        total: total_frames,
                        requested: *n,
                    });
                }
                let split_point = total_frames - n;
                let train_indices = (0..split_point).collect();
                let val_indices = (split_point..total_frames).collect();
                // Guard: if n == total_frames, train is empty but n == 0 was already allowed above
                // Actually n=0 is fine (no validation), n=total_frames makes train empty
                if *n == total_frames && total_frames > 0 {
                    return Err(SplitError::EmptyTrainSet);
                }
                Ok(Self {
                    train_indices,
                    val_indices,
                    total_frames,
                })
            }

            SplitStrategy::RandomFraction { fraction, seed } => {
                if !fraction.is_finite() || *fraction <= 0.0 || *fraction >= 1.0 {
                    return Err(SplitError::InvalidFraction {
                        fraction: *fraction,
                    });
                }

                let val_count = ((*fraction * total_frames as f32).floor() as usize).max(1);
                if val_count >= total_frames {
                    return Err(SplitError::EmptyTrainSet);
                }

                // Fisher-Yates shuffle on [0..total_frames]
                let mut indices: Vec<usize> = (0..total_frames).collect();
                let mut state = xorshift64_init(*seed);
                for i in (1..total_frames).rev() {
                    let j = (xorshift64(&mut state) % (i as u64 + 1)) as usize;
                    indices.swap(i, j);
                }

                // First val_count become validation, the rest become training
                let mut val_indices: Vec<usize> = indices[..val_count].to_vec();
                let mut train_indices: Vec<usize> = indices[val_count..].to_vec();
                val_indices.sort_unstable();
                train_indices.sort_unstable();

                Ok(Self {
                    train_indices,
                    val_indices,
                    total_frames,
                })
            }

            SplitStrategy::EveryNth(n) => {
                if *n == 0 {
                    return Err(SplitError::ZeroStep);
                }
                let val_indices: Vec<usize> = (0..total_frames).filter(|&i| i % n == 0).collect();
                let train_indices: Vec<usize> = (0..total_frames).filter(|&i| i % n != 0).collect();
                if train_indices.is_empty() && total_frames > 0 {
                    return Err(SplitError::EmptyTrainSet);
                }
                Ok(Self {
                    train_indices,
                    val_indices,
                    total_frames,
                })
            }

            SplitStrategy::Manual(indices) => {
                // Validate all indices are in range
                for &idx in indices {
                    if idx >= total_frames {
                        return Err(SplitError::InvalidIndex {
                            idx,
                            total: total_frames,
                        });
                    }
                }

                // Dedup and sort
                let mut val_indices = indices.clone();
                val_indices.sort_unstable();
                val_indices.dedup();

                // `val_indices` is sorted, so the training set (every index
                // in `0..total_frames` not in `val_indices`) is built with a
                // single O(total_frames) linear merge instead of an O(len)
                // `contains` scan per frame (previously O(total_frames *
                // val_indices.len()) — ~1e9 comparisons for a 100k-frame
                // sequence with a 10k-frame validation set).
                let mut train_indices =
                    Vec::with_capacity(total_frames.saturating_sub(val_indices.len()));
                let mut val_pos = 0usize;
                for i in 0..total_frames {
                    if val_pos < val_indices.len() && val_indices[val_pos] == i {
                        val_pos += 1;
                    } else {
                        train_indices.push(i);
                    }
                }

                if train_indices.is_empty() && total_frames > 0 {
                    return Err(SplitError::EmptyTrainSet);
                }

                Ok(Self {
                    train_indices,
                    val_indices,
                    total_frames,
                })
            }
        }
    }

    pub fn train_count(&self) -> usize {
        self.train_indices.len()
    }

    pub fn val_count(&self) -> usize {
        self.val_indices.len()
    }

    pub fn val_fraction(&self) -> f32 {
        if self.total_frames == 0 {
            return 0.0;
        }
        self.val_indices.len() as f32 / self.total_frames as f32
    }

    /// Whether `idx` is a validation frame.
    ///
    /// `O(log val_count)`: every [`SplitStrategy`] arm leaves `val_indices`
    /// sorted ascending (`LastN`/`EveryNth` build it in order;
    /// `RandomFraction`/`Manual` sort explicitly before returning), so a
    /// binary search is always valid here — this is the natural per-frame
    /// hot-path query during training, previously an `O(val_count)` linear
    /// `contains` scan per call.
    pub fn is_val_frame(&self, idx: usize) -> bool {
        self.val_indices.binary_search(&idx).is_ok()
    }

    pub fn format_summary(&self) -> String {
        format!(
            "DataSplit {{ total: {}, train: {}, val: {}, val_fraction: {:.3} }}",
            self.total_frames,
            self.train_count(),
            self.val_count(),
            self.val_fraction()
        )
    }
}

// ---------------------------------------------------------------------------
// ValidationStep
// ---------------------------------------------------------------------------

/// A single recorded validation checkpoint.
#[derive(Debug, Clone)]
pub struct ValidationStep {
    pub training_step: usize,
    pub val_psnr: f32,
    pub val_loss: f32,
    /// Training loss at this step (for overfitting comparison).
    pub train_loss: f32,
    pub timestamp_ms: u64,
}

// ---------------------------------------------------------------------------
// Linear regression helper
// ---------------------------------------------------------------------------

/// Compute the slope of a linear fit to (x_i, y_i) pairs.
/// Returns 0.0 if the denominator is effectively zero.
fn linear_slope(xs: &[f32], ys: &[f32]) -> f32 {
    debug_assert_eq!(xs.len(), ys.len());
    let n = xs.len() as f32;
    if n < 2.0 {
        return 0.0;
    }
    let sum_x: f32 = xs.iter().sum();
    let sum_y: f32 = ys.iter().sum();
    let sum_xy: f32 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();
    let sum_xx: f32 = xs.iter().map(|x| x * x).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return 0.0;
    }
    (n * sum_xy - sum_x * sum_y) / denom
}

// ---------------------------------------------------------------------------
// ValidationTracker
// ---------------------------------------------------------------------------

/// Tracks validation history and trends during training.
pub struct ValidationTracker {
    pub history: Vec<ValidationStep>,
    pub best_val_psnr: f32,
    pub best_step: usize,
    /// Run validation every N training steps.
    pub val_interval: usize,
}

impl fmt::Debug for ValidationTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValidationTracker")
            .field("history_len", &self.history.len())
            .field("best_val_psnr", &self.best_val_psnr)
            .field("best_step", &self.best_step)
            .field("val_interval", &self.val_interval)
            .finish()
    }
}

impl ValidationTracker {
    pub fn new(val_interval: usize) -> Self {
        Self {
            history: Vec::new(),
            best_val_psnr: f32::NEG_INFINITY,
            best_step: 0,
            val_interval,
        }
    }

    /// Record a validation step.
    pub fn record(&mut self, step: ValidationStep) {
        if step.val_psnr > self.best_val_psnr {
            self.best_val_psnr = step.val_psnr;
            self.best_step = step.training_step;
        }
        self.history.push(step);
    }

    /// Should we run validation at this training step?
    pub fn should_validate(&self, training_step: usize) -> bool {
        training_step.is_multiple_of(self.val_interval)
    }

    /// Detect potential overfitting: returns true if val_loss trend is increasing
    /// while train_loss trend is decreasing over the last `window` steps.
    pub fn detect_overfitting(&self, window: usize) -> bool {
        let n = self.history.len();
        if n < 2 {
            return false;
        }
        let start = n.saturating_sub(window);
        let slice = &self.history[start..];

        let xs: Vec<f32> = slice.iter().map(|s| s.training_step as f32).collect();
        let val_losses: Vec<f32> = slice.iter().map(|s| s.val_loss).collect();
        let train_losses: Vec<f32> = slice.iter().map(|s| s.train_loss).collect();

        let val_slope = linear_slope(&xs, &val_losses);
        let train_slope = linear_slope(&xs, &train_losses);

        val_slope > 0.0 && train_slope < 0.0
    }

    /// Compute the trend in val_psnr over the last `window` steps.
    /// Positive slope = improving, negative = degrading.
    pub fn val_psnr_trend(&self, window: usize) -> f32 {
        let n = self.history.len();
        if n < 2 {
            return 0.0;
        }
        let start = n.saturating_sub(window);
        let slice = &self.history[start..];

        let xs: Vec<f32> = (0..slice.len()).map(|i| i as f32).collect();
        let ys: Vec<f32> = slice.iter().map(|s| s.val_psnr).collect();

        linear_slope(&xs, &ys)
    }

    pub fn format_summary(&self) -> String {
        format!(
            "ValidationTracker {{ steps: {}, best_psnr: {:.2} dB @ step {}, interval: {} }}",
            self.history.len(),
            self.best_val_psnr,
            self.best_step,
            self.val_interval
        )
    }
}

// ---------------------------------------------------------------------------
// MonitorMetric
// ---------------------------------------------------------------------------

/// Which metric the early stopper should monitor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MonitorMetric {
    /// Higher is better (maximize).
    ValPsnr,
    /// Lower is better (minimize).
    ValLoss,
}

// ---------------------------------------------------------------------------
// EarlyStopperConfig
// ---------------------------------------------------------------------------

/// Configuration for the early stopper.
#[derive(Debug, Clone)]
pub struct EarlyStopperConfig {
    /// Stop training if no improvement for this many validation steps.
    pub patience: usize,
    /// Minimum improvement to count as progress.
    pub min_delta: f32,
    /// Whether [`EarlyStopper::record_val_with_model`] should snapshot the
    /// model whenever it sees a new best, so [`EarlyStopper::restore_best_into`]
    /// can later restore it. Has no effect on [`EarlyStopper::record_val`],
    /// which never sees a model to snapshot.
    pub restore_best: bool,
    /// Which metric to monitor.
    pub monitor_metric: MonitorMetric,
}

impl Default for EarlyStopperConfig {
    fn default() -> Self {
        Self {
            patience: 10,
            min_delta: 0.01,
            restore_best: true,
            monitor_metric: MonitorMetric::ValPsnr,
        }
    }
}

// ---------------------------------------------------------------------------
// EarlyStopper
// ---------------------------------------------------------------------------

/// Patience-based early stopping controller.
pub struct EarlyStopper {
    pub config: EarlyStopperConfig,
    best_value: f32,
    steps_without_improvement: usize,
    best_step: usize,
    pub stopped: bool,
    /// Snapshot of the model captured at `best_step`, when
    /// [`Self::record_val_with_model`] was used and `config.restore_best`
    /// was set at that improving step. `None` if no such snapshot has been
    /// captured yet (including: only plain [`Self::record_val`] has been
    /// called so far, or `restore_best` was `false`).
    best_weights: Option<GaussianModel>,
}

impl fmt::Debug for EarlyStopper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EarlyStopper")
            .field("config", &self.config)
            .field("best_value", &self.best_value)
            .field("steps_without_improvement", &self.steps_without_improvement)
            .field("best_step", &self.best_step)
            .field("stopped", &self.stopped)
            .field("has_best_weights", &self.best_weights.is_some())
            .finish()
    }
}

impl EarlyStopper {
    pub fn new(config: EarlyStopperConfig) -> Self {
        let best_value = match config.monitor_metric {
            MonitorMetric::ValPsnr => f32::NEG_INFINITY,
            MonitorMetric::ValLoss => f32::INFINITY,
        };
        Self {
            config,
            best_value,
            steps_without_improvement: 0,
            best_step: 0,
            stopped: false,
            best_weights: None,
        }
    }

    /// Core bookkeeping shared by [`Self::record_val`] and
    /// [`Self::record_val_with_model`]. Returns whether this call was a new
    /// best (i.e. whether a caller wanting to snapshot weights should do so
    /// now) — `self.stopped` carries the "should training stop" signal, as
    /// it did before this was split out.
    fn record_val_core(&mut self, step: usize, val_psnr: f32, val_loss: f32) -> bool {
        let current = match self.config.monitor_metric {
            MonitorMetric::ValPsnr => val_psnr,
            MonitorMetric::ValLoss => val_loss,
        };

        let is_better = match self.config.monitor_metric {
            MonitorMetric::ValPsnr => current > self.best_value + self.config.min_delta,
            MonitorMetric::ValLoss => current < self.best_value - self.config.min_delta,
        };

        if is_better {
            self.best_value = current;
            self.best_step = step;
            self.steps_without_improvement = 0;
        } else {
            self.steps_without_improvement += 1;
        }

        if self.steps_without_improvement >= self.config.patience {
            self.stopped = true;
        }

        is_better
    }

    /// Record a validation result. Returns true if training should stop.
    ///
    /// Does **not** capture a weight snapshot even when
    /// `config.restore_best` is set — [`Self::restore_best_into`] will have
    /// nothing to restore unless [`Self::record_val_with_model`] is used
    /// instead (this variant exists because `EarlyStopper` cannot see the
    /// model on its own; see [`Self::record_val_with_model`] for the
    /// implementation of what `restore_best` actually promises).
    pub fn record_val(&mut self, step: usize, val_psnr: f32, val_loss: f32) -> bool {
        self.record_val_core(step, val_psnr, val_loss);
        self.stopped
    }

    /// Record a validation result exactly like [`Self::record_val`], and —
    /// when `config.restore_best` is set and this step is a new best —
    /// additionally clone `model` into an internal snapshot that
    /// [`Self::restore_best_into`] can later restore.
    ///
    /// The snapshot is a full clone of `model`, so this is only as cheap as
    /// cloning a [`GaussianModel`]; call it at your validation cadence, not
    /// every training step.
    pub fn record_val_with_model(
        &mut self,
        step: usize,
        val_psnr: f32,
        val_loss: f32,
        model: &GaussianModel,
    ) -> bool {
        let improved = self.record_val_core(step, val_psnr, val_loss);
        if improved && self.config.restore_best {
            self.best_weights = Some(model.clone());
        }
        self.stopped
    }

    /// Whether a weight snapshot is available for [`Self::restore_best_into`]
    /// to restore.
    pub fn has_best_weights(&self) -> bool {
        self.best_weights.is_some()
    }

    /// Overwrite `model` with the snapshot captured at [`Self::best_step`],
    /// if one exists. Returns `true` if a snapshot was restored, `false`
    /// (leaving `model` untouched) if none is available — e.g. only
    /// [`Self::record_val`] has been called so far, or `config.restore_best`
    /// was `false` at every improving step.
    pub fn restore_best_into(&self, model: &mut GaussianModel) -> bool {
        match &self.best_weights {
            Some(snapshot) => {
                *model = snapshot.clone();
                true
            }
            None => false,
        }
    }

    pub fn should_stop(&self) -> bool {
        self.stopped
    }

    pub fn best_step(&self) -> usize {
        self.best_step
    }

    pub fn best_value(&self) -> f32 {
        self.best_value
    }

    pub fn steps_without_improvement(&self) -> usize {
        self.steps_without_improvement
    }

    pub fn format_status(&self) -> String {
        let metric = match self.config.monitor_metric {
            MonitorMetric::ValPsnr => "val_psnr",
            MonitorMetric::ValLoss => "val_loss",
        };
        format!(
            "EarlyStopper {{ metric: {}, best: {:.4} @ step {}, no_improve: {}/{}, stopped: {} }}",
            metric,
            self.best_value,
            self.best_step,
            self.steps_without_improvement,
            self.config.patience,
            self.stopped
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a ValidationStep with only the fields we care about.
    fn make_step(
        training_step: usize,
        val_psnr: f32,
        val_loss: f32,
        train_loss: f32,
    ) -> ValidationStep {
        ValidationStep {
            training_step,
            val_psnr,
            val_loss,
            train_loss,
            timestamp_ms: 0,
        }
    }

    // ---- DataSplit tests ----

    #[test]
    fn test_split_last_n() -> Result<(), SplitError> {
        let split = DataSplit::from_strategy(&SplitStrategy::LastN(3), 10)?;
        assert_eq!(split.train_count(), 7);
        assert_eq!(split.val_count(), 3);
        assert_eq!(split.val_indices, vec![7, 8, 9]);
        assert_eq!(split.train_indices, (0..7).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn test_split_last_n_zero() -> Result<(), SplitError> {
        let split = DataSplit::from_strategy(&SplitStrategy::LastN(0), 10)?;
        assert_eq!(split.val_count(), 0);
        assert_eq!(split.train_count(), 10);
        Ok(())
    }

    #[test]
    fn test_split_last_n_too_large_error() {
        let result = DataSplit::from_strategy(&SplitStrategy::LastN(11), 10);
        assert!(matches!(
            result,
            Err(SplitError::InsufficientFrames {
                total: 10,
                requested: 11
            })
        ));
    }

    #[test]
    fn test_split_last_n_all_val_error() {
        // LastN == total_frames means empty train set
        let result = DataSplit::from_strategy(&SplitStrategy::LastN(10), 10);
        assert!(matches!(result, Err(SplitError::EmptyTrainSet)));
    }

    #[test]
    fn test_split_random_fraction_count() -> Result<(), SplitError> {
        let n = 100;
        let fraction = 0.2;
        let split =
            DataSplit::from_strategy(&SplitStrategy::RandomFraction { fraction, seed: 42 }, n)?;
        // floor(0.2 * 100) = 20
        assert_eq!(split.val_count(), 20);
        assert_eq!(split.train_count(), 80);
        assert_eq!(split.total_frames, n);
        Ok(())
    }

    #[test]
    fn test_split_random_fraction_deterministic() -> Result<(), SplitError> {
        let strategy = SplitStrategy::RandomFraction {
            fraction: 0.3,
            seed: 1234,
        };
        let a = DataSplit::from_strategy(&strategy, 50)?;
        let b = DataSplit::from_strategy(&strategy, 50)?;
        assert_eq!(a.val_indices, b.val_indices);
        assert_eq!(a.train_indices, b.train_indices);
        Ok(())
    }

    #[test]
    fn test_split_random_fraction_different_seeds() -> Result<(), SplitError> {
        let a = DataSplit::from_strategy(
            &SplitStrategy::RandomFraction {
                fraction: 0.3,
                seed: 1,
            },
            50,
        )?;
        let b = DataSplit::from_strategy(
            &SplitStrategy::RandomFraction {
                fraction: 0.3,
                seed: 2,
            },
            50,
        )?;
        // Different seeds almost certainly produce different splits for n=50
        assert_ne!(a.val_indices, b.val_indices);
        Ok(())
    }

    #[test]
    fn test_split_random_fraction_invalid_fraction() {
        assert!(matches!(
            DataSplit::from_strategy(
                &SplitStrategy::RandomFraction {
                    fraction: 0.0,
                    seed: 1
                },
                10
            ),
            Err(SplitError::InvalidFraction { .. })
        ));
        assert!(matches!(
            DataSplit::from_strategy(
                &SplitStrategy::RandomFraction {
                    fraction: 1.0,
                    seed: 1
                },
                10
            ),
            Err(SplitError::InvalidFraction { .. })
        ));
        assert!(matches!(
            DataSplit::from_strategy(
                &SplitStrategy::RandomFraction {
                    fraction: -0.1,
                    seed: 1
                },
                10
            ),
            Err(SplitError::InvalidFraction { .. })
        ));
    }

    #[test]
    fn test_split_every_nth() -> Result<(), SplitError> {
        let split = DataSplit::from_strategy(&SplitStrategy::EveryNth(3), 9)?;
        // Indices 0, 3, 6 are val
        assert_eq!(split.val_indices, vec![0, 3, 6]);
        assert_eq!(split.train_indices, vec![1, 2, 4, 5, 7, 8]);
        Ok(())
    }

    #[test]
    fn test_split_every_nth_zero_error() {
        let result = DataSplit::from_strategy(&SplitStrategy::EveryNth(0), 10);
        assert!(matches!(result, Err(SplitError::ZeroStep)));
    }

    #[test]
    fn test_split_every_nth_step_one_empty_train_error() {
        // EveryNth(1) means every frame is validation → empty train
        let result = DataSplit::from_strategy(&SplitStrategy::EveryNth(1), 5);
        assert!(matches!(result, Err(SplitError::EmptyTrainSet)));
    }

    #[test]
    fn test_split_manual() -> Result<(), SplitError> {
        let split = DataSplit::from_strategy(
            &SplitStrategy::Manual(vec![2, 5, 5, 8]), // 5 is duplicated
            10,
        )?;
        assert_eq!(split.val_indices, vec![2, 5, 8]);
        assert_eq!(split.val_count(), 3);
        assert_eq!(split.train_count(), 7);
        Ok(())
    }

    #[test]
    fn test_split_manual_invalid_index_error() {
        let result = DataSplit::from_strategy(&SplitStrategy::Manual(vec![0, 10]), 10);
        assert!(matches!(
            result,
            Err(SplitError::InvalidIndex { idx: 10, total: 10 })
        ));
    }

    #[test]
    fn test_split_is_val_frame() -> Result<(), SplitError> {
        let split = DataSplit::from_strategy(&SplitStrategy::LastN(3), 10)?;
        assert!(!split.is_val_frame(6));
        assert!(split.is_val_frame(7));
        assert!(split.is_val_frame(9));
        Ok(())
    }

    #[test]
    fn test_is_val_frame_out_of_range_idx_does_not_panic() -> Result<(), SplitError> {
        let split = DataSplit::from_strategy(&SplitStrategy::LastN(3), 10)?;
        assert!(!split.is_val_frame(1_000));
        Ok(())
    }

    #[test]
    fn test_split_manual_empty_val_indices() -> Result<(), SplitError> {
        let split = DataSplit::from_strategy(&SplitStrategy::Manual(vec![]), 5)?;
        assert!(split.val_indices.is_empty());
        assert_eq!(split.train_indices, vec![0, 1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn test_split_manual_first_and_last_frame_val() -> Result<(), SplitError> {
        // Exercises the O(n) merge's boundary conditions: val index 0
        // (consumed on the very first loop iteration) and val index ==
        // total_frames - 1 (consumed on the very last).
        let split = DataSplit::from_strategy(&SplitStrategy::Manual(vec![0, 9]), 10)?;
        assert_eq!(split.val_indices, vec![0, 9]);
        assert_eq!(split.train_indices, (1..9).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn test_split_fraction() -> Result<(), SplitError> {
        let split = DataSplit::from_strategy(&SplitStrategy::LastN(2), 10)?;
        let frac = split.val_fraction();
        assert!((frac - 0.2).abs() < 1e-6, "expected 0.2, got {}", frac);
        Ok(())
    }

    #[test]
    fn test_split_all_train() {
        let split = DataSplit::all_train(5);
        assert_eq!(split.train_count(), 5);
        assert_eq!(split.val_count(), 0);
        assert_eq!(split.val_fraction(), 0.0);
    }

    // ---- ValidationTracker tests ----

    #[test]
    fn test_tracker_should_validate() {
        let tracker = ValidationTracker::new(10);
        assert!(tracker.should_validate(0));
        assert!(tracker.should_validate(10));
        assert!(tracker.should_validate(20));
        assert!(!tracker.should_validate(5));
        assert!(!tracker.should_validate(15));
    }

    #[test]
    fn test_tracker_record_best_psnr() {
        let mut tracker = ValidationTracker::new(10);
        tracker.record(make_step(10, 25.0, 0.5, 0.8));
        tracker.record(make_step(20, 30.0, 0.4, 0.6));
        tracker.record(make_step(30, 28.0, 0.45, 0.5));
        assert!((tracker.best_val_psnr - 30.0).abs() < 1e-6);
        assert_eq!(tracker.best_step, 20);
        assert_eq!(tracker.history.len(), 3);
    }

    #[test]
    fn test_tracker_detect_overfitting_true() {
        let mut tracker = ValidationTracker::new(1);
        // val_loss increasing, train_loss decreasing → overfitting
        for i in 0..10usize {
            tracker.record(make_step(
                i,
                25.0,
                0.1 + i as f32 * 0.05, // val_loss increasing
                1.0 - i as f32 * 0.05, // train_loss decreasing
            ));
        }
        assert!(tracker.detect_overfitting(10));
    }

    #[test]
    fn test_tracker_detect_overfitting_false() {
        let mut tracker = ValidationTracker::new(1);
        // Both val_loss and train_loss decreasing → healthy training
        for i in 0..10usize {
            tracker.record(make_step(
                i,
                25.0 + i as f32,
                1.0 - i as f32 * 0.05, // val_loss decreasing
                1.2 - i as f32 * 0.06, // train_loss decreasing
            ));
        }
        assert!(!tracker.detect_overfitting(10));
    }

    #[test]
    fn test_tracker_val_psnr_trend_positive() {
        let mut tracker = ValidationTracker::new(1);
        for i in 0..5usize {
            tracker.record(make_step(i, i as f32 * 2.0, 0.5, 0.5));
        }
        let trend = tracker.val_psnr_trend(5);
        assert!(trend > 0.0, "expected positive trend, got {}", trend);
    }

    #[test]
    fn test_tracker_val_psnr_trend_negative() {
        let mut tracker = ValidationTracker::new(1);
        for i in 0..5usize {
            tracker.record(make_step(i, 10.0 - i as f32 * 2.0, 0.5, 0.5));
        }
        let trend = tracker.val_psnr_trend(5);
        assert!(trend < 0.0, "expected negative trend, got {}", trend);
    }

    #[test]
    fn test_tracker_detect_overfitting_insufficient_data() {
        let mut tracker = ValidationTracker::new(1);
        // Only 1 step — can't compute slope
        tracker.record(make_step(0, 25.0, 0.5, 0.8));
        assert!(!tracker.detect_overfitting(10));
    }

    // ---- EarlyStopper tests ----

    #[test]
    fn test_early_stopper_stops_after_patience() {
        let config = EarlyStopperConfig {
            patience: 3,
            min_delta: 0.01,
            restore_best: false,
            monitor_metric: MonitorMetric::ValPsnr,
        };
        let mut stopper = EarlyStopper::new(config);
        // First call — improvement (sets baseline)
        assert!(!stopper.record_val(0, 25.0, 0.5));
        // Next 3 calls — no improvement
        assert!(!stopper.record_val(1, 25.0, 0.5)); // steps_without = 1
        assert!(!stopper.record_val(2, 25.0, 0.5)); // steps_without = 2
        assert!(stopper.record_val(3, 25.0, 0.5)); // steps_without = 3 → stop
        assert!(stopper.should_stop());
    }

    #[test]
    fn test_early_stopper_resets_on_improvement() {
        let config = EarlyStopperConfig {
            patience: 3,
            min_delta: 0.01,
            restore_best: false,
            monitor_metric: MonitorMetric::ValPsnr,
        };
        let mut stopper = EarlyStopper::new(config);
        stopper.record_val(0, 25.0, 0.5);
        stopper.record_val(1, 25.0, 0.5); // no improve → 1
        stopper.record_val(2, 25.0, 0.5); // no improve → 2
                                          // Improvement resets counter
        stopper.record_val(3, 26.0, 0.4);
        assert_eq!(stopper.steps_without_improvement(), 0);
        assert!(!stopper.should_stop());
    }

    #[test]
    fn test_early_stopper_val_loss_monitor() {
        let config = EarlyStopperConfig {
            patience: 2,
            min_delta: 0.0,
            restore_best: false,
            monitor_metric: MonitorMetric::ValLoss,
        };
        let mut stopper = EarlyStopper::new(config);
        // Lower val_loss is better
        assert!(!stopper.record_val(0, 25.0, 1.0)); // sets best=1.0
        assert!(!stopper.record_val(1, 25.0, 0.8)); // improvement → reset
        assert!(!stopper.record_val(2, 25.0, 0.9)); // no improve (0.9 > 0.8) → 1
        assert!(stopper.record_val(3, 25.0, 1.0)); // no improve → 2 → stop
        assert!(stopper.should_stop());
        assert!((stopper.best_value() - 0.8).abs() < 1e-6);
        assert_eq!(stopper.best_step(), 1);
    }

    #[test]
    fn test_early_stopper_best_step_tracked() {
        let config = EarlyStopperConfig::default(); // patience=10, ValPsnr
        let mut stopper = EarlyStopper::new(config);
        stopper.record_val(0, 20.0, 0.9);
        stopper.record_val(5, 30.0, 0.5);
        stopper.record_val(10, 28.0, 0.6);
        assert_eq!(stopper.best_step(), 5);
        assert!((stopper.best_value() - 30.0).abs() < 1e-6);
    }

    fn make_test_model(marker: f32) -> GaussianModel {
        GaussianModel {
            gaussians: vec![oxigaf_render::gaussian::GaussianAttributes {
                position: [marker, 0.0, 0.0],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [0.0; 3],
                opacity: 0.0,
            }],
            sh_coeffs: vec![0.0; 3],
            sh_degree: 0,
            face_indices: vec![0],
            barycentric: vec![[1.0 / 3.0; 3]],
            local_offsets: vec![[0.0; 3]],
            is_rigid: vec![true],
        }
    }

    #[test]
    fn test_early_stopper_restore_best_into_false_without_snapshot() {
        // Regression: `restore_best` was documented as "tracked only, no
        // actual weight copy" — a caller enabling it got the final (worse)
        // weights instead of the best ones. Plain `record_val` still takes
        // no model, so there is genuinely nothing to restore.
        let config = EarlyStopperConfig::default(); // restore_best: true
        let mut stopper = EarlyStopper::new(config);
        stopper.record_val(0, 25.0, 0.5);
        assert!(!stopper.has_best_weights());
        let mut model = make_test_model(0.0);
        assert!(!stopper.restore_best_into(&mut model));
        assert_eq!(
            model.gaussians[0].position[0], 0.0,
            "untouched when nothing to restore"
        );
    }

    #[test]
    fn test_early_stopper_record_val_with_model_captures_and_restores_best() {
        let config = EarlyStopperConfig {
            patience: 10,
            min_delta: 0.0,
            restore_best: true,
            monitor_metric: MonitorMetric::ValPsnr,
        };
        let mut stopper = EarlyStopper::new(config);

        // Step 0: first result, always an improvement over -inf.
        stopper.record_val_with_model(0, 20.0, 0.9, &make_test_model(1.0));
        assert!(stopper.has_best_weights());

        // Step 1: worse — must NOT overwrite the step-0 snapshot.
        stopper.record_val_with_model(1, 15.0, 1.0, &make_test_model(2.0));

        // Step 2: new best — snapshot should now be step 2's model.
        stopper.record_val_with_model(2, 30.0, 0.5, &make_test_model(3.0));
        assert_eq!(stopper.best_step(), 2);

        let mut model = make_test_model(0.0);
        assert!(stopper.restore_best_into(&mut model));
        assert_eq!(
            model.gaussians[0].position[0], 3.0,
            "restored snapshot must be the one taken at the actual best step"
        );
    }

    #[test]
    fn test_early_stopper_record_val_with_model_respects_restore_best_flag() {
        let config = EarlyStopperConfig {
            patience: 10,
            min_delta: 0.0,
            restore_best: false, // explicitly disabled
            monitor_metric: MonitorMetric::ValPsnr,
        };
        let mut stopper = EarlyStopper::new(config);
        stopper.record_val_with_model(0, 20.0, 0.9, &make_test_model(1.0));
        assert!(
            !stopper.has_best_weights(),
            "restore_best: false must never capture a snapshot"
        );
    }

    #[test]
    fn test_format_methods_compile() {
        // Smoke test: ensure format methods produce non-empty strings
        let split = DataSplit::all_train(10);
        assert!(!split.format_summary().is_empty());

        let tracker = ValidationTracker::new(5);
        assert!(!tracker.format_summary().is_empty());

        let stopper = EarlyStopper::new(EarlyStopperConfig::default());
        assert!(!stopper.format_status().is_empty());
    }
}
