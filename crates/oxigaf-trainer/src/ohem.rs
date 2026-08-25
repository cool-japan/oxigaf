//! Online Hard Example Mining (OHEM) for training sample selection.
//!
//! Selects training samples (views/frames) with the highest loss for additional
//! focus, improving training efficiency by concentrating compute on the hardest
//! examples.
//!
//! # Quick start
//! ```rust
//! use oxigaf_trainer::ohem::{OhemConfig, OhemTracker};
//!
//! let config = OhemConfig::default();
//! let mut tracker = OhemTracker::new(100, config).expect("100 examples");
//!
//! tracker.record_loss(0, 0.8).expect("valid index");
//! tracker.record_loss(1, 0.2).expect("valid index");
//! tracker.step();
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// PRNG (xorshift64, no rand crate)
// ─────────────────────────────────────────────────────────────────────────────

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = 0x123456789ABCDEF0;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn xorshift_f32(state: &mut u64) -> f32 {
    (xorshift64(state) >> 11) as f32 / (1u64 << 53) as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// OhemError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the OHEM subsystem.
#[derive(Debug, Error)]
pub enum OhemError {
    #[error("No examples registered")]
    NoExamples,

    #[error("Top-k {k} exceeds total examples {total}")]
    KExceedsTotal { k: usize, total: usize },

    #[error("Example index {0} out of bounds")]
    IndexOutOfBounds(usize),

    #[error("Invalid EMA alpha {0}: must be in (0, 1]")]
    InvalidAlpha(f32),

    #[error("Empty loss history for example {0}")]
    EmptyHistory(usize),

    #[error("Invalid {field} {value}: must be in [0, 1]")]
    InvalidFraction { field: &'static str, value: f32 },

    #[error("Invalid priority_exponent {0}: must be finite")]
    InvalidExponent(f32),
}

// ─────────────────────────────────────────────────────────────────────────────
// ExampleRecord
// ─────────────────────────────────────────────────────────────────────────────

/// Per-example loss history with EMA smoothing.
#[derive(Debug, Clone)]
pub struct ExampleRecord {
    /// View/frame index in the dataset.
    pub index: usize,
    /// Most recent raw loss.
    pub raw_loss: f32,
    /// EMA-smoothed loss.
    pub ema_loss: f32,
    /// How many times this example was evaluated.
    pub visit_count: u64,
    /// Training step of the most recent update.
    pub last_step: u64,
    /// How many times selected as a hard example.
    pub selection_count: u64,
}

impl ExampleRecord {
    fn new(index: usize) -> Self {
        Self {
            index,
            raw_loss: 0.0,
            ema_loss: 0.0,
            visit_count: 0,
            last_step: 0,
            selection_count: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OhemConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for OHEM.
#[derive(Debug, Clone)]
pub struct OhemConfig {
    /// EMA decay factor α for smoothing loss. Default 0.1 (fast response).
    pub ema_alpha: f32,
    /// Fraction of examples to select as "hard". Default 0.3.
    pub hard_fraction: f32,
    /// Minimum times an example must be visited before being eligible for hard selection.
    pub min_visits: u64,
    /// Prioritize by: raw loss or EMA loss.
    pub use_ema: bool,
    /// Add uniform random exploration fraction (so unvisited examples get sampled).
    /// In [0, 1]. Default 0.1.
    pub exploration_fraction: f32,
    /// Exponent for loss-based sampling weight: w_i = loss_i^priority_exponent.
    /// 1.0 = proportional, 2.0 = squared priority. Default 1.0.
    pub priority_exponent: f32,
}

impl Default for OhemConfig {
    fn default() -> Self {
        Self {
            ema_alpha: 0.1,
            hard_fraction: 0.3,
            min_visits: 3,
            use_ema: true,
            exploration_fraction: 0.1,
            priority_exponent: 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OhemStats
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics about mining effectiveness.
#[derive(Debug, Clone)]
pub struct OhemStats {
    /// Total number of examples.
    pub num_examples: usize,
    /// Examples with visit_count >= 1.
    pub num_visited: usize,
    /// Examples with visit_count >= min_visits.
    pub num_eligible: usize,
    /// Mean EMA loss (0 if none visited).
    pub mean_loss: f32,
    /// Max EMA loss (0 if none).
    pub max_loss: f32,
    /// Min EMA loss among visited (0 if none).
    pub min_loss: f32,
    /// Sum of all visit counts.
    pub total_visits: u64,
    /// num_visited / num_examples.
    pub coverage_fraction: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// OhemTracker
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks per-example losses and selects hard examples for OHEM.
#[derive(Debug)]
pub struct OhemTracker {
    config: OhemConfig,
    records: Vec<ExampleRecord>,
    num_examples: usize,
    current_step: u64,
}

impl OhemTracker {
    /// Create a tracker for the given number of examples (views/frames).
    ///
    /// Returns [`OhemError::NoExamples`] if `num_examples == 0`.
    /// Returns [`OhemError::InvalidAlpha`] if `config.ema_alpha` is not in `(0, 1]`.
    /// Returns [`OhemError::InvalidFraction`] if `config.exploration_fraction`
    /// or `config.hard_fraction` is outside `[0, 1]`.
    /// Returns [`OhemError::InvalidExponent`] if `config.priority_exponent`
    /// is not finite.
    pub fn new(num_examples: usize, config: OhemConfig) -> Result<Self, OhemError> {
        if num_examples == 0 {
            return Err(OhemError::NoExamples);
        }
        if config.ema_alpha <= 0.0 || config.ema_alpha > 1.0 {
            return Err(OhemError::InvalidAlpha(config.ema_alpha));
        }
        // `OhemConfig` has all-`pub` fields, so a struct literal can bypass
        // any single validating constructor; validate every field consulted
        // by arithmetic elsewhere in this module (`suggest_batch` divides
        // `batch_size` between exploration/hard counts derived from these
        // fractions, and `compute_priority_weights` raises losses to
        // `priority_exponent`).
        if !(0.0..=1.0).contains(&config.exploration_fraction) {
            return Err(OhemError::InvalidFraction {
                field: "exploration_fraction",
                value: config.exploration_fraction,
            });
        }
        if !(0.0..=1.0).contains(&config.hard_fraction) {
            return Err(OhemError::InvalidFraction {
                field: "hard_fraction",
                value: config.hard_fraction,
            });
        }
        if !config.priority_exponent.is_finite() {
            return Err(OhemError::InvalidExponent(config.priority_exponent));
        }
        let records = (0..num_examples).map(ExampleRecord::new).collect();
        Ok(Self {
            config,
            records,
            num_examples,
            current_step: 0,
        })
    }

    /// Record loss for a single example at the current step.
    ///
    /// Updates EMA: `ema = alpha * raw + (1 - alpha) * ema`.
    /// On first visit (`visit_count == 0`), `ema = raw`.
    pub fn record_loss(&mut self, index: usize, loss: f32) -> Result<(), OhemError> {
        if index >= self.num_examples {
            return Err(OhemError::IndexOutOfBounds(index));
        }
        let alpha = self.config.ema_alpha;
        let step = self.current_step;
        let record = &mut self.records[index];

        record.raw_loss = loss;
        if record.visit_count == 0 {
            record.ema_loss = loss;
        } else {
            record.ema_loss = alpha * loss + (1.0 - alpha) * record.ema_loss;
        }
        record.visit_count += 1;
        record.last_step = step;
        Ok(())
    }

    /// Advance the training step counter.
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Current training step.
    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    /// Select top-k hardest examples.
    ///
    /// Eligibility: `visit_count >= config.min_visits`.
    /// If fewer than k eligible, fills remaining slots with unvisited examples
    /// (sorted by index, round-robin).
    /// Returns indices sorted by loss (highest first).
    pub fn select_hard(&self, k: usize) -> Result<Vec<usize>, OhemError> {
        if k > self.num_examples {
            return Err(OhemError::KExceedsTotal {
                k,
                total: self.num_examples,
            });
        }

        let use_ema = self.config.use_ema;
        let min_visits = self.config.min_visits;

        // Collect eligible examples (visit_count >= min_visits).
        let mut eligible: Vec<usize> = self
            .records
            .iter()
            .filter(|r| r.visit_count >= min_visits)
            .map(|r| r.index)
            .collect();

        // Sort eligible by loss descending (highest loss first).
        eligible.sort_by(|&a, &b| {
            let loss_a = if use_ema {
                self.records[a].ema_loss
            } else {
                self.records[a].raw_loss
            };
            let loss_b = if use_ema {
                self.records[b].ema_loss
            } else {
                self.records[b].raw_loss
            };
            loss_b
                .partial_cmp(&loss_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take at most k from eligible.
        let selected_eligible: Vec<usize> = eligible.into_iter().take(k).collect();
        let remaining = k - selected_eligible.len();

        if remaining == 0 {
            return Ok(selected_eligible);
        }

        // Fill remaining from unvisited examples (visit_count < min_visits),
        // scanned in ascending index order.
        let selected_set: std::collections::HashSet<usize> =
            selected_eligible.iter().copied().collect();

        let mut fill: Vec<usize> = self
            .records
            .iter()
            .filter(|r| r.visit_count < min_visits && !selected_set.contains(&r.index))
            .map(|r| r.index)
            .take(remaining)
            .collect();

        // Sort fill by index ascending (round-robin order).
        fill.sort_unstable();

        let mut result = selected_eligible;
        result.extend(fill);
        Ok(result)
    }

    /// Select examples via priority sampling (stochastic, proportional to
    /// `loss^exponent`).
    ///
    /// Uses xorshift64 PRNG seeded by `seed XOR 0xDEAD_BEEF`.
    /// Returns `n` indices (may repeat if population is small).
    /// With probability `exploration_fraction`, picks uniformly at random
    /// instead.
    pub fn sample_weighted(&self, n: usize, seed: u64) -> Result<Vec<usize>, OhemError> {
        if self.num_examples == 0 {
            return Err(OhemError::NoExamples);
        }
        let weights = compute_priority_weights(self);
        let mut state = seed ^ 0xDEAD_BEEF;
        // Ensure non-zero state.
        if state == 0 {
            state = 0x123456789ABCDEF0;
        }

        let exploration = self.config.exploration_fraction;
        let mut result = Vec::with_capacity(n);

        // Build cumulative weights for weighted sampling.
        let cumulative: Vec<f32> = {
            let mut cum = Vec::with_capacity(weights.len());
            let mut sum = 0.0f32;
            for &w in &weights {
                sum += w;
                cum.push(sum);
            }
            cum
        };
        let total_weight = cumulative.last().copied().unwrap_or(1.0);

        for _ in 0..n {
            let u = xorshift_f32(&mut state);
            let idx = if u < exploration {
                // Uniform random exploration.
                let r = xorshift_f32(&mut state);
                (r * self.num_examples as f32) as usize % self.num_examples
            } else {
                // Weighted selection via binary search on cumulative weights.
                let target = xorshift_f32(&mut state) * total_weight;
                let pos = cumulative.partition_point(|&c| c < target);
                pos.min(self.num_examples - 1)
            };
            result.push(idx);
        }

        Ok(result)
    }

    /// Suggest next batch of `batch_size` examples.
    ///
    /// - `(1 - exploration_fraction) * batch_size` from hard examples
    /// - `exploration_fraction * batch_size` from random examples
    ///
    /// Returns indices (length = `batch_size`).
    pub fn suggest_batch(&self, batch_size: usize, seed: u64) -> Result<Vec<usize>, OhemError> {
        if self.num_examples == 0 {
            return Err(OhemError::NoExamples);
        }

        // `OhemTracker::new` validates `exploration_fraction` is in [0, 1],
        // but defend against a mismatched batch_size (or, prior to that
        // validation, a config that skipped `new` entirely — see its
        // comment) rounding `exploration_count` above `batch_size`, which
        // would otherwise underflow the `usize` subtraction below.
        let exploration_count = ((self.config.exploration_fraction * batch_size as f32).round()
            as usize)
            .min(batch_size);
        let hard_count = batch_size.saturating_sub(exploration_count);

        let mut state = seed ^ 0xDEAD_BEEF;
        if state == 0 {
            state = 0x123456789ABCDEF0;
        }

        let mut result = Vec::with_capacity(batch_size);

        // Hard examples: select from top-k via weighted sampling using current step seed.
        if hard_count > 0 {
            // How many top-hard candidates to draw from.
            let hard_pool_size = {
                let fraction =
                    (self.config.hard_fraction * self.num_examples as f32).ceil() as usize;
                fraction.max(1).min(self.num_examples)
            };

            // Get ordered hard candidates (may be padded with unvisited).
            let candidates = self.select_hard(hard_pool_size)?;

            // Sample hard_count from candidates with xorshift.
            for _ in 0..hard_count {
                let r = xorshift_f32(&mut state);
                let idx = (r * candidates.len() as f32) as usize % candidates.len();
                result.push(candidates[idx]);
            }
        }

        // Exploration examples: uniform random.
        for _ in 0..exploration_count {
            let r = xorshift_f32(&mut state);
            let idx = (r * self.num_examples as f32) as usize % self.num_examples;
            result.push(idx);
        }

        Ok(result)
    }

    /// Get the record for a specific example.
    pub fn record(&self, index: usize) -> Option<&ExampleRecord> {
        self.records.get(index)
    }

    /// All records, ordered by example index.
    pub fn records(&self) -> &[ExampleRecord] {
        &self.records
    }

    /// Mean EMA loss across all visited examples.
    ///
    /// Returns `None` if no examples have been visited.
    pub fn mean_loss(&self) -> Option<f32> {
        let visited: Vec<f32> = self
            .records
            .iter()
            .filter(|r| r.visit_count >= 1)
            .map(|r| r.ema_loss)
            .collect();
        if visited.is_empty() {
            return None;
        }
        Some(visited.iter().sum::<f32>() / visited.len() as f32)
    }

    /// Max EMA loss (the hardest example's loss).
    ///
    /// Returns `None` if no examples have been visited.
    pub fn max_loss(&self) -> Option<f32> {
        self.records
            .iter()
            .filter(|r| r.visit_count >= 1)
            .map(|r| r.ema_loss)
            .reduce(f32::max)
    }

    /// Statistics about mining effectiveness.
    pub fn stats(&self) -> OhemStats {
        let num_visited = self.records.iter().filter(|r| r.visit_count >= 1).count();
        let num_eligible = self
            .records
            .iter()
            .filter(|r| r.visit_count >= self.config.min_visits)
            .count();
        let total_visits: u64 = self.records.iter().map(|r| r.visit_count).sum();
        let coverage_fraction = if self.num_examples == 0 {
            0.0
        } else {
            num_visited as f32 / self.num_examples as f32
        };

        let visited_losses: Vec<f32> = self
            .records
            .iter()
            .filter(|r| r.visit_count >= 1)
            .map(|r| r.ema_loss)
            .collect();

        let mean_loss = if visited_losses.is_empty() {
            0.0
        } else {
            visited_losses.iter().sum::<f32>() / visited_losses.len() as f32
        };
        let max_loss = visited_losses
            .iter()
            .copied()
            .reduce(f32::max)
            .unwrap_or(0.0);
        let min_loss = visited_losses
            .iter()
            .copied()
            .reduce(f32::min)
            .unwrap_or(0.0);

        OhemStats {
            num_examples: self.num_examples,
            num_visited,
            num_eligible,
            mean_loss,
            max_loss,
            min_loss,
            total_visits,
            coverage_fraction,
        }
    }

    /// Reset all loss history (keeps `num_examples` and `config`).
    pub fn reset(&mut self) {
        for record in &mut self.records {
            *record = ExampleRecord::new(record.index);
        }
        self.current_step = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_priority_weights (free function)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute normalized priority weights for all examples.
///
/// `w_i = loss_i^exponent / sum(loss_j^exponent)`.
/// Unvisited examples get weight `= mean_visited_loss^exponent` (or `1.0` if
/// none visited).
pub fn compute_priority_weights(tracker: &OhemTracker) -> Vec<f32> {
    let use_ema = tracker.config.use_ema;
    let exponent = tracker.config.priority_exponent;

    // Compute the mean loss of visited examples (for filling unvisited slots).
    let visited_losses: Vec<f32> = tracker
        .records
        .iter()
        .filter(|r| r.visit_count >= 1)
        .map(|r| if use_ema { r.ema_loss } else { r.raw_loss })
        .collect();

    let mean_visited = if visited_losses.is_empty() {
        1.0f32
    } else {
        visited_losses.iter().sum::<f32>() / visited_losses.len() as f32
    };

    // Raw weights before normalization.
    let raw_weights: Vec<f32> = tracker
        .records
        .iter()
        .map(|r| {
            let loss = if r.visit_count >= 1 {
                if use_ema {
                    r.ema_loss
                } else {
                    r.raw_loss
                }
            } else {
                mean_visited
            };
            loss.max(0.0).powf(exponent)
        })
        .collect();

    let total: f32 = raw_weights.iter().sum();
    if total <= 0.0 {
        // Uniform weights when all losses are zero.
        let uniform = 1.0 / tracker.num_examples as f32;
        return vec![uniform; tracker.num_examples];
    }

    raw_weights.iter().map(|&w| w / total).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn default_tracker(n: usize) -> OhemTracker {
        OhemTracker::new(n, OhemConfig::default()).unwrap()
    }

    // 1. `new` creates tracker with n examples, all at zero.
    #[test]
    fn test_new_creates_n_examples() {
        let tracker = default_tracker(10);
        assert_eq!(tracker.num_examples, 10);
        assert_eq!(tracker.records().len(), 10);
        for (i, r) in tracker.records().iter().enumerate() {
            assert_eq!(r.index, i);
            assert_eq!(r.visit_count, 0);
            assert_eq!(r.selection_count, 0);
            assert!((r.ema_loss - 0.0).abs() < 1e-6);
            assert!((r.raw_loss - 0.0).abs() < 1e-6);
        }
    }

    // 2. `new` with 0 examples returns NoExamples error.
    #[test]
    fn test_new_zero_examples_returns_error() {
        let result = OhemTracker::new(0, OhemConfig::default());
        assert!(matches!(result, Err(OhemError::NoExamples)));
    }

    // 3. `record_loss` updates raw_loss and ema_loss.
    #[test]
    fn test_record_loss_updates_fields() {
        let mut tracker = default_tracker(5);
        tracker.record_loss(2, 0.75).unwrap();
        let r = tracker.record(2).unwrap();
        assert!((r.raw_loss - 0.75).abs() < 1e-4);
        // First visit: ema == raw.
        assert!((r.ema_loss - 0.75).abs() < 1e-4);
        assert_eq!(r.visit_count, 1);
    }

    // 4. `record_loss` first visit: ema == raw.
    #[test]
    fn test_record_loss_first_visit_ema_equals_raw() {
        let mut tracker = default_tracker(3);
        tracker.record_loss(0, 0.42).unwrap();
        let r = tracker.record(0).unwrap();
        assert!((r.ema_loss - r.raw_loss).abs() < 1e-6);
        assert!((r.ema_loss - 0.42).abs() < 1e-4);
    }

    // 5. `record_loss` EMA converges toward new values.
    #[test]
    fn test_record_loss_ema_converges() {
        let mut tracker = default_tracker(1);
        // Record high loss 5 times.
        for _ in 0..5 {
            tracker.record_loss(0, 1.0).unwrap();
        }
        let ema_after_high = tracker.record(0).unwrap().ema_loss;
        // Now record low loss 3 times; EMA should decrease.
        for _ in 0..3 {
            tracker.record_loss(0, 0.0).unwrap();
        }
        let ema_after_low = tracker.record(0).unwrap().ema_loss;
        // EMA should have decreased toward 0.
        assert!(ema_after_low < ema_after_high);
        // But still > 0 (EMA hasn't fully converged).
        assert!(ema_after_low > 0.0);
    }

    // 6. `record_loss` out of bounds returns IndexOutOfBounds.
    #[test]
    fn test_record_loss_out_of_bounds() {
        let mut tracker = default_tracker(5);
        let result = tracker.record_loss(5, 0.1);
        assert!(matches!(result, Err(OhemError::IndexOutOfBounds(5))));
    }

    // 7. `step` increments current_step.
    #[test]
    fn test_step_increments() {
        let mut tracker = default_tracker(5);
        assert_eq!(tracker.current_step(), 0);
        tracker.step();
        assert_eq!(tracker.current_step(), 1);
        tracker.step();
        tracker.step();
        assert_eq!(tracker.current_step(), 3);
    }

    // 8. `select_hard` with all eligible returns top-k by loss.
    #[test]
    fn test_select_hard_returns_top_k() {
        let config = OhemConfig {
            min_visits: 1,
            ..OhemConfig::default()
        };
        let mut tracker = OhemTracker::new(5, config).unwrap();
        // Record losses: indices 0..=4 get losses 0.1, 0.5, 0.3, 0.9, 0.7.
        tracker.record_loss(0, 0.1).unwrap();
        tracker.record_loss(1, 0.5).unwrap();
        tracker.record_loss(2, 0.3).unwrap();
        tracker.record_loss(3, 0.9).unwrap();
        tracker.record_loss(4, 0.7).unwrap();

        let hard = tracker.select_hard(3).unwrap();
        assert_eq!(hard.len(), 3);
        // Top-3 by loss descending: 3 (0.9), 4 (0.7), 1 (0.5).
        assert_eq!(hard[0], 3);
        assert_eq!(hard[1], 4);
        assert_eq!(hard[2], 1);
    }

    // 9. `select_hard` with none eligible falls back to unvisited.
    #[test]
    fn test_select_hard_falls_back_to_unvisited() {
        // Default min_visits = 3; record only 2 times per example.
        let mut tracker = default_tracker(5);
        for _ in 0..2 {
            tracker.record_loss(0, 0.5).unwrap();
            tracker.record_loss(1, 0.8).unwrap();
        }
        // No example has 3 visits, so none are eligible.
        let hard = tracker.select_hard(2).unwrap();
        assert_eq!(hard.len(), 2);
        // Filled from unvisited/under-visited in index order; indices 0..4 all count.
        // The unvisited (not at 0, 1) are indices 2, 3, 4.
        // But select_hard first checks eligible (none), then fills from unvisited.
        // "Unvisited" means visit_count < min_visits — that includes 0,1,2,3,4 all.
        for &idx in &hard {
            assert!(idx < 5);
        }
    }

    // 10. `select_hard` k > total returns error.
    #[test]
    fn test_select_hard_k_exceeds_total() {
        let tracker = default_tracker(5);
        let result = tracker.select_hard(6);
        assert!(matches!(
            result,
            Err(OhemError::KExceedsTotal { k: 6, total: 5 })
        ));
    }

    // 11. `select_hard` result is sorted highest-first.
    #[test]
    fn test_select_hard_sorted_highest_first() {
        let config = OhemConfig {
            min_visits: 1,
            ..OhemConfig::default()
        };
        let mut tracker = OhemTracker::new(10, config).unwrap();
        for i in 0..10usize {
            tracker.record_loss(i, i as f32 * 0.1).unwrap();
        }
        let hard = tracker.select_hard(5).unwrap();
        // Eligible set is all 10 examples (min_visits=1, all have 1 visit).
        // Top-5 by descending EMA loss: indices 9,8,7,6,5.
        for window in hard.windows(2) {
            let loss_a = tracker.record(window[0]).unwrap().ema_loss;
            let loss_b = tracker.record(window[1]).unwrap().ema_loss;
            assert!(
                loss_a >= loss_b,
                "expected descending order: {loss_a} >= {loss_b}"
            );
        }
    }

    // 12. `mean_loss` returns None when no examples visited.
    #[test]
    fn test_mean_loss_none_when_unvisited() {
        let tracker = default_tracker(5);
        assert!(tracker.mean_loss().is_none());
    }

    // 13. `mean_loss` correct mean after recording.
    #[test]
    fn test_mean_loss_correct() {
        let mut tracker = default_tracker(4);
        tracker.record_loss(0, 0.2).unwrap();
        tracker.record_loss(1, 0.4).unwrap();
        tracker.record_loss(2, 0.6).unwrap();
        tracker.record_loss(3, 0.8).unwrap();
        // All first-visit: EMA == raw; mean = (0.2+0.4+0.6+0.8)/4 = 0.5.
        let mean = tracker.mean_loss().unwrap();
        assert!((mean - 0.5).abs() < 1e-4, "expected mean ~0.5, got {mean}");
    }

    // 14. `max_loss` returns the highest EMA loss.
    #[test]
    fn test_max_loss_returns_highest() {
        let mut tracker = default_tracker(4);
        tracker.record_loss(0, 0.1).unwrap();
        tracker.record_loss(1, 0.9).unwrap();
        tracker.record_loss(2, 0.5).unwrap();
        let max = tracker.max_loss().unwrap();
        assert!((max - 0.9).abs() < 1e-4, "expected max ~0.9, got {max}");
    }

    // 15. `stats` coverage_fraction = num_visited / num_examples.
    #[test]
    fn test_stats_coverage_fraction() {
        let mut tracker = default_tracker(10);
        tracker.record_loss(0, 0.3).unwrap();
        tracker.record_loss(5, 0.7).unwrap();
        let stats = tracker.stats();
        assert_eq!(stats.num_visited, 2);
        assert!((stats.coverage_fraction - 0.2).abs() < 1e-4);
    }

    // 16. `stats` num_eligible accounts for min_visits threshold.
    #[test]
    fn test_stats_num_eligible() {
        // Default min_visits = 3.
        let mut tracker = default_tracker(5);
        // Index 0: 3 visits (eligible), index 1: 2 visits (not eligible).
        for _ in 0..3 {
            tracker.record_loss(0, 0.5).unwrap();
        }
        for _ in 0..2 {
            tracker.record_loss(1, 0.4).unwrap();
        }
        let stats = tracker.stats();
        assert_eq!(stats.num_eligible, 1);
        assert_eq!(stats.num_visited, 2);
    }

    // 17. `reset` clears all records.
    #[test]
    fn test_reset_clears_records() {
        let mut tracker = default_tracker(5);
        tracker.record_loss(0, 1.0).unwrap();
        tracker.step();
        tracker.record_loss(1, 0.5).unwrap();
        tracker.reset();
        assert_eq!(tracker.current_step(), 0);
        for r in tracker.records() {
            assert_eq!(r.visit_count, 0);
            assert!((r.raw_loss - 0.0).abs() < 1e-6);
            assert!((r.ema_loss - 0.0).abs() < 1e-6);
        }
    }

    // 18. `compute_priority_weights` sums to ~1.0.
    #[test]
    fn test_compute_priority_weights_sums_to_one() {
        let mut tracker = default_tracker(5);
        tracker.record_loss(0, 0.3).unwrap();
        tracker.record_loss(1, 0.7).unwrap();
        tracker.record_loss(2, 0.1).unwrap();
        let weights = compute_priority_weights(&tracker);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "weights sum {sum} != 1.0");
    }

    // 19. `compute_priority_weights` higher-loss gets higher weight.
    #[test]
    fn test_compute_priority_weights_proportional_to_loss() {
        let config = OhemConfig {
            priority_exponent: 1.0,
            ..OhemConfig::default()
        };
        let mut tracker = OhemTracker::new(3, config).unwrap();
        tracker.record_loss(0, 0.1).unwrap();
        tracker.record_loss(1, 0.5).unwrap();
        tracker.record_loss(2, 0.9).unwrap();
        let weights = compute_priority_weights(&tracker);
        // Higher loss index should have a higher weight.
        assert!(
            weights[2] > weights[1],
            "w[2] > w[1]: {} > {}",
            weights[2],
            weights[1]
        );
        assert!(
            weights[1] > weights[0],
            "w[1] > w[0]: {} > {}",
            weights[1],
            weights[0]
        );
    }

    // 20. `sample_weighted` returns n indices.
    #[test]
    fn test_sample_weighted_returns_n() {
        let mut tracker = default_tracker(10);
        for i in 0..10 {
            tracker.record_loss(i, (i as f32 + 1.0) * 0.1).unwrap();
        }
        let samples = tracker.sample_weighted(20, 42).unwrap();
        assert_eq!(samples.len(), 20);
        for &idx in &samples {
            assert!(idx < 10, "sampled index {idx} out of range");
        }
    }

    // 21. `suggest_batch` returns batch_size indices.
    #[test]
    fn test_suggest_batch_returns_batch_size() {
        let mut tracker = default_tracker(20);
        for i in 0..10 {
            for _ in 0..3 {
                tracker.record_loss(i, (i as f32 + 1.0) * 0.05).unwrap();
            }
        }
        let batch = tracker.suggest_batch(16, 99).unwrap();
        assert_eq!(
            batch.len(),
            16,
            "batch length should be 16, got {}",
            batch.len()
        );
        for &idx in &batch {
            assert!(idx < 20, "batch index {idx} out of range");
        }
    }

    // 22. `OhemTracker::new` rejects exploration_fraction outside [0, 1].
    #[test]
    fn test_new_rejects_exploration_fraction_above_one() {
        let config = OhemConfig {
            exploration_fraction: 1.5,
            ..OhemConfig::default()
        };
        let result = OhemTracker::new(2, config);
        assert!(
            matches!(
                result,
                Err(OhemError::InvalidFraction {
                    field: "exploration_fraction",
                    ..
                })
            ),
            "expected InvalidFraction, got {result:?}"
        );
    }

    // 23. `suggest_batch` never underflows batch_size even if a
    // hand-constructed config (bypassing `new`'s validation via a struct
    // literal on the private-but-same-module `OhemTracker`) somehow carried
    // an out-of-range fraction through to this point.
    #[test]
    fn test_suggest_batch_no_underflow_with_extreme_exploration_fraction() {
        let mut tracker = default_tracker(4);
        for i in 0..4 {
            tracker.record_loss(i, 0.5).unwrap();
        }
        // Directly mutate the (module-private, same-file-accessible) config
        // to the exact pathological value the original bug report used:
        // exploration_fraction=1.5 with batch_size=2 would previously
        // compute exploration_count=3 and then `2 - 3`, underflowing.
        tracker.config.exploration_fraction = 1.5;
        let batch = tracker.suggest_batch(2, 7);
        assert!(batch.is_ok(), "suggest_batch must not panic: {batch:?}");
        assert_eq!(batch.unwrap().len(), 2);
    }
}
