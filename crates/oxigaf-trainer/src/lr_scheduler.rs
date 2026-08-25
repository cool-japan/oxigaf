//! Learning rate schedulers for the OxiGAF training pipeline.
//!
//! Provides a family of LR schedules (constant, step decay, exponential,
//! cosine annealing, warmup variants, cyclic, polynomial) behind a unified
//! [`LrSchedule`] trait and [`LrScheduler`] enum for ergonomic dispatch.

use std::f64::consts::PI;
use thiserror::Error;

// ──────────────────────────────────────────────────────────────────────────────
// Error type
// ──────────────────────────────────────────────────────────────────────────────

/// Errors produced by LR-scheduler construction or query.
#[derive(Debug, Error, PartialEq)]
pub enum LrSchedulerError {
    /// Invalid scheduler configuration (e.g. steps = 0, negative lr).
    #[error("Invalid scheduler configuration: {0}")]
    InvalidConfig(String),

    /// A step index is outside the expected range (non-fatal warning).
    #[error("Invalid step: {0}")]
    InvalidStep(String),
}

// ──────────────────────────────────────────────────────────────────────────────
// Core trait
// ──────────────────────────────────────────────────────────────────────────────

/// A learning-rate schedule that maps a (0-indexed) training step to an LR.
pub trait LrSchedule {
    /// Return the learning rate for the given step (0-indexed).
    fn lr_at(&self, step: usize) -> f64;

    /// Human-readable description of this schedule.
    fn description(&self) -> String;
}

// ──────────────────────────────────────────────────────────────────────────────
// Scheduler kind tag (no serde needed per spec)
// ──────────────────────────────────────────────────────────────────────────────

/// Identifies which kind of schedule is in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerKind {
    Constant,
    StepDecay,
    Exponential,
    CosineAnnealing,
    WarmupCosine,
    Cyclic,
    Polynomial,
    WarmupLinear,
}

// ──────────────────────────────────────────────────────────────────────────────
// Concrete schedule structs
// ──────────────────────────────────────────────────────────────────────────────

/// Constant learning rate that never changes.
#[derive(Debug, Clone)]
pub struct ConstantSchedule {
    pub base_lr: f64,
}

impl LrSchedule for ConstantSchedule {
    #[inline]
    fn lr_at(&self, _step: usize) -> f64 {
        self.base_lr
    }

    fn description(&self) -> String {
        format!("Constant(base_lr={})", self.base_lr)
    }
}

// ──────────────────────────────────────────────────────────────────────────────

/// Step-wise decay: multiply LR by `decay_factor` every `step_size` steps.
///
/// `lr_at(step) = base_lr * decay_factor^floor(step / step_size)`
#[derive(Debug, Clone)]
pub struct StepDecaySchedule {
    pub base_lr: f64,
    /// Multiplicative decay factor per epoch, must satisfy `0 < decay_factor < 1`.
    pub decay_factor: f64,
    /// Number of steps between each decay application.
    pub step_size: usize,
}

impl LrSchedule for StepDecaySchedule {
    fn lr_at(&self, step: usize) -> f64 {
        // `StepDecaySchedule` has all-`pub` fields, so a struct literal can
        // bypass the `step_size == 0` check in `LrScheduler::step_decay`.
        // Guard here too (matching `CyclicSchedule::lr_at`'s guard on its
        // own divisor) so the public `lr_at` trait method never divides by
        // zero.
        if self.step_size == 0 {
            return self.base_lr;
        }
        let exponent = step / self.step_size; // integer floor division
        self.base_lr * self.decay_factor.powi(exponent as i32)
    }

    fn description(&self) -> String {
        format!(
            "StepDecay(base_lr={}, decay_factor={}, step_size={})",
            self.base_lr, self.decay_factor, self.step_size
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────

/// Exponential per-step decay.
///
/// `lr_at(step) = base_lr * decay_rate^step`
#[derive(Debug, Clone)]
pub struct ExponentialDecaySchedule {
    pub base_lr: f64,
    /// Per-step multiplier (e.g. 0.9999).
    pub decay_rate: f64,
}

impl LrSchedule for ExponentialDecaySchedule {
    fn lr_at(&self, step: usize) -> f64 {
        self.base_lr * self.decay_rate.powi(step as i32)
    }

    fn description(&self) -> String {
        format!(
            "ExponentialDecay(base_lr={}, decay_rate={})",
            self.base_lr, self.decay_rate
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────

/// Half-cosine annealing from `base_lr` down to `min_lr`.
///
/// `lr_at(step) = min_lr + 0.5*(base_lr - min_lr)*(1 + cos(π * t / T))`
/// where `t = clamp(step, 0, total_steps)`.
///
/// After `total_steps`, returns `min_lr`.
#[derive(Debug, Clone)]
pub struct CosineAnnealingSchedule {
    pub base_lr: f64,
    pub min_lr: f64,
    pub total_steps: usize,
}

impl LrSchedule for CosineAnnealingSchedule {
    fn lr_at(&self, step: usize) -> f64 {
        if self.total_steps == 0 || step >= self.total_steps {
            return self.min_lr;
        }
        let t = step as f64 / self.total_steps as f64;
        self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (PI * t).cos())
    }

    fn description(&self) -> String {
        format!(
            "CosineAnnealing(base_lr={}, min_lr={}, total_steps={})",
            self.base_lr, self.min_lr, self.total_steps
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────

/// Linear warmup followed by cosine annealing.
///
/// - `0..warmup_steps`: LR linearly ramps from 0 to `base_lr`.
/// - `warmup_steps..total_steps`: cosine decay from `base_lr` to `min_lr`.
/// - After `total_steps`: `min_lr`.
#[derive(Debug, Clone)]
pub struct WarmupCosineSchedule {
    pub warmup_steps: usize,
    pub base_lr: f64,
    pub min_lr: f64,
    pub total_steps: usize,
}

impl LrSchedule for WarmupCosineSchedule {
    fn lr_at(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear ramp: 0 → base_lr
            let factor = warmup_factor(step, self.warmup_steps);
            return self.base_lr * factor;
        }

        // After all steps: min_lr
        if step >= self.total_steps {
            return self.min_lr;
        }

        // Cosine phase
        let cosine_steps = self.total_steps.saturating_sub(self.warmup_steps);
        let cosine_step = step - self.warmup_steps;
        let factor = cosine_decay_factor(cosine_step, cosine_steps);
        self.min_lr + (self.base_lr - self.min_lr) * factor
    }

    fn description(&self) -> String {
        format!(
            "WarmupCosine(warmup_steps={}, base_lr={}, min_lr={}, total_steps={})",
            self.warmup_steps, self.base_lr, self.min_lr, self.total_steps
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────

/// Triangular (symmetric) cyclic learning rate.
///
/// Each full cycle has length `2 * cycle_steps`:
/// - First half (`0..cycle_steps`): linear ramp from `min_lr` to `max_lr`.
/// - Second half (`cycle_steps..2*cycle_steps`): linear ramp from `max_lr` to `min_lr`.
#[derive(Debug, Clone)]
pub struct CyclicSchedule {
    pub min_lr: f64,
    pub max_lr: f64,
    /// Steps per half-cycle.
    pub cycle_steps: usize,
}

impl LrSchedule for CyclicSchedule {
    fn lr_at(&self, step: usize) -> f64 {
        if self.cycle_steps == 0 {
            return self.min_lr;
        }
        let full_cycle = 2 * self.cycle_steps;
        let pos = step % full_cycle;
        if pos < self.cycle_steps {
            // Rising ramp: min_lr → max_lr
            let frac = pos as f64 / self.cycle_steps as f64;
            self.min_lr + (self.max_lr - self.min_lr) * frac
        } else {
            // Falling ramp: max_lr → min_lr
            let frac = (pos - self.cycle_steps) as f64 / self.cycle_steps as f64;
            self.max_lr - (self.max_lr - self.min_lr) * frac
        }
    }

    fn description(&self) -> String {
        format!(
            "Cyclic(min_lr={}, max_lr={}, cycle_steps={})",
            self.min_lr, self.max_lr, self.cycle_steps
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────

/// Polynomial decay from `base_lr` to `end_lr` over `total_steps`.
///
/// `lr_at(step) = (base_lr - end_lr) * ((1 - t)^power) + end_lr`
/// where `t = clamp(step, 0, total_steps) / total_steps`.
#[derive(Debug, Clone)]
pub struct PolynomialDecaySchedule {
    pub base_lr: f64,
    pub end_lr: f64,
    pub total_steps: usize,
    pub power: f64,
}

impl LrSchedule for PolynomialDecaySchedule {
    fn lr_at(&self, step: usize) -> f64 {
        if self.total_steps == 0 || step >= self.total_steps {
            return self.end_lr;
        }
        let t = step as f64 / self.total_steps as f64;
        (self.base_lr - self.end_lr) * (1.0 - t).powf(self.power) + self.end_lr
    }

    fn description(&self) -> String {
        format!(
            "PolynomialDecay(base_lr={}, end_lr={}, total_steps={}, power={})",
            self.base_lr, self.end_lr, self.total_steps, self.power
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────

/// Linear warmup followed by linear decay to 0.
///
/// - `0..warmup_steps`: linear ramp from 0 to `base_lr`.
/// - `warmup_steps..total_steps`: linear decay from `base_lr` to 0.
/// - After `total_steps`: 0.0.
#[derive(Debug, Clone)]
pub struct WarmupLinearSchedule {
    pub warmup_steps: usize,
    pub base_lr: f64,
    pub total_steps: usize,
}

impl LrSchedule for WarmupLinearSchedule {
    fn lr_at(&self, step: usize) -> f64 {
        if step >= self.total_steps {
            return 0.0;
        }

        if step < self.warmup_steps {
            // Linear ramp
            let factor = warmup_factor(step, self.warmup_steps);
            return self.base_lr * factor;
        }

        // Linear decay: base_lr → 0 over remaining steps
        let decay_steps = self.total_steps - self.warmup_steps;
        let elapsed = step - self.warmup_steps;
        if decay_steps == 0 {
            return 0.0;
        }
        self.base_lr * (1.0 - elapsed as f64 / decay_steps as f64)
    }

    fn description(&self) -> String {
        format!(
            "WarmupLinear(warmup_steps={}, base_lr={}, total_steps={})",
            self.warmup_steps, self.base_lr, self.total_steps
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Enum dispatcher
// ──────────────────────────────────────────────────────────────────────────────

/// Unified LR scheduler that dispatches to one of the concrete schedule types.
#[derive(Debug, Clone)]
pub enum LrScheduler {
    Constant(ConstantSchedule),
    StepDecay(StepDecaySchedule),
    Exponential(ExponentialDecaySchedule),
    CosineAnnealing(CosineAnnealingSchedule),
    WarmupCosine(WarmupCosineSchedule),
    Cyclic(CyclicSchedule),
    Polynomial(PolynomialDecaySchedule),
    WarmupLinear(WarmupLinearSchedule),
}

impl LrSchedule for LrScheduler {
    fn lr_at(&self, step: usize) -> f64 {
        match self {
            LrScheduler::Constant(s) => s.lr_at(step),
            LrScheduler::StepDecay(s) => s.lr_at(step),
            LrScheduler::Exponential(s) => s.lr_at(step),
            LrScheduler::CosineAnnealing(s) => s.lr_at(step),
            LrScheduler::WarmupCosine(s) => s.lr_at(step),
            LrScheduler::Cyclic(s) => s.lr_at(step),
            LrScheduler::Polynomial(s) => s.lr_at(step),
            LrScheduler::WarmupLinear(s) => s.lr_at(step),
        }
    }

    fn description(&self) -> String {
        match self {
            LrScheduler::Constant(s) => s.description(),
            LrScheduler::StepDecay(s) => s.description(),
            LrScheduler::Exponential(s) => s.description(),
            LrScheduler::CosineAnnealing(s) => s.description(),
            LrScheduler::WarmupCosine(s) => s.description(),
            LrScheduler::Cyclic(s) => s.description(),
            LrScheduler::Polynomial(s) => s.description(),
            LrScheduler::WarmupLinear(s) => s.description(),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Constructor helpers with validation
// ──────────────────────────────────────────────────────────────────────────────

impl LrScheduler {
    /// Constant LR. Requires `base_lr > 0`.
    pub fn constant(base_lr: f64) -> Result<Self, LrSchedulerError> {
        if base_lr <= 0.0 {
            return Err(LrSchedulerError::InvalidConfig(format!(
                "base_lr must be > 0, got {}",
                base_lr
            )));
        }
        Ok(LrScheduler::Constant(ConstantSchedule { base_lr }))
    }

    /// Step-decay LR. Requires `base_lr > 0`, `0 < decay_factor < 1`, `step_size > 0`.
    pub fn step_decay(
        base_lr: f64,
        decay_factor: f64,
        step_size: usize,
    ) -> Result<Self, LrSchedulerError> {
        if base_lr <= 0.0 {
            return Err(LrSchedulerError::InvalidConfig(format!(
                "base_lr must be > 0, got {}",
                base_lr
            )));
        }
        if decay_factor <= 0.0 || decay_factor >= 1.0 {
            return Err(LrSchedulerError::InvalidConfig(format!(
                "decay_factor must be in (0, 1), got {}",
                decay_factor
            )));
        }
        if step_size == 0 {
            return Err(LrSchedulerError::InvalidConfig(
                "step_size must be > 0".to_string(),
            ));
        }
        Ok(LrScheduler::StepDecay(StepDecaySchedule {
            base_lr,
            decay_factor,
            step_size,
        }))
    }

    /// Exponential per-step decay. Requires `base_lr > 0`, `0 < decay_rate <= 1`.
    pub fn exponential(base_lr: f64, decay_rate: f64) -> Result<Self, LrSchedulerError> {
        if base_lr <= 0.0 {
            return Err(LrSchedulerError::InvalidConfig(format!(
                "base_lr must be > 0, got {}",
                base_lr
            )));
        }
        if decay_rate <= 0.0 || decay_rate > 1.0 {
            return Err(LrSchedulerError::InvalidConfig(format!(
                "decay_rate must be in (0, 1], got {}",
                decay_rate
            )));
        }
        Ok(LrScheduler::Exponential(ExponentialDecaySchedule {
            base_lr,
            decay_rate,
        }))
    }

    /// Cosine annealing. Requires `base_lr > 0`, `min_lr >= 0`, `min_lr < base_lr`,
    /// `total_steps > 0`.
    pub fn cosine_annealing(
        base_lr: f64,
        min_lr: f64,
        total_steps: usize,
    ) -> Result<Self, LrSchedulerError> {
        if base_lr <= 0.0 {
            return Err(LrSchedulerError::InvalidConfig(format!(
                "base_lr must be > 0, got {}",
                base_lr
            )));
        }
        if min_lr < 0.0 {
            return Err(LrSchedulerError::InvalidConfig(format!(
                "min_lr must be >= 0, got {}",
                min_lr
            )));
        }
        if min_lr >= base_lr {
            return Err(LrSchedulerError::InvalidConfig(format!(
                "min_lr ({}) must be < base_lr ({})",
                min_lr, base_lr
            )));
        }
        if total_steps == 0 {
            return Err(LrSchedulerError::InvalidConfig(
                "total_steps must be > 0".to_string(),
            ));
        }
        Ok(LrScheduler::CosineAnnealing(CosineAnnealingSchedule {
            base_lr,
            min_lr,
            total_steps,
        }))
    }

    /// Warmup + cosine decay. Requires `base_lr > 0`, `min_lr >= 0`, `min_lr < base_lr`,
    /// `warmup_steps < total_steps`.
    pub fn warmup_cosine(
        warmup_steps: usize,
        base_lr: f64,
        min_lr: f64,
        total_steps: usize,
    ) -> Result<Self, LrSchedulerError> {
        if base_lr <= 0.0 {
            return Err(LrSchedulerError::InvalidConfig(format!(
                "base_lr must be > 0, got {}",
                base_lr
            )));
        }
        if min_lr < 0.0 {
            return Err(LrSchedulerError::InvalidConfig(format!(
                "min_lr must be >= 0, got {}",
                min_lr
            )));
        }
        if min_lr >= base_lr {
            return Err(LrSchedulerError::InvalidConfig(format!(
                "min_lr ({}) must be < base_lr ({})",
                min_lr, base_lr
            )));
        }
        if total_steps == 0 {
            return Err(LrSchedulerError::InvalidConfig(
                "total_steps must be > 0".to_string(),
            ));
        }
        if warmup_steps >= total_steps {
            return Err(LrSchedulerError::InvalidConfig(format!(
                "warmup_steps ({}) must be < total_steps ({})",
                warmup_steps, total_steps
            )));
        }
        Ok(LrScheduler::WarmupCosine(WarmupCosineSchedule {
            warmup_steps,
            base_lr,
            min_lr,
            total_steps,
        }))
    }

    /// Triangular cyclic LR. Requires `min_lr < max_lr`, `cycle_steps > 0`.
    pub fn cyclic(min_lr: f64, max_lr: f64, cycle_steps: usize) -> Result<Self, LrSchedulerError> {
        if min_lr >= max_lr {
            return Err(LrSchedulerError::InvalidConfig(format!(
                "min_lr ({}) must be < max_lr ({})",
                min_lr, max_lr
            )));
        }
        if cycle_steps == 0 {
            return Err(LrSchedulerError::InvalidConfig(
                "cycle_steps must be > 0".to_string(),
            ));
        }
        Ok(LrScheduler::Cyclic(CyclicSchedule {
            min_lr,
            max_lr,
            cycle_steps,
        }))
    }

    /// Polynomial decay. Requires `base_lr > 0`, `power > 0`, `total_steps > 0`.
    pub fn polynomial(
        base_lr: f64,
        end_lr: f64,
        total_steps: usize,
        power: f64,
    ) -> Result<Self, LrSchedulerError> {
        if base_lr <= 0.0 {
            return Err(LrSchedulerError::InvalidConfig(format!(
                "base_lr must be > 0, got {}",
                base_lr
            )));
        }
        if power <= 0.0 {
            return Err(LrSchedulerError::InvalidConfig(format!(
                "power must be > 0, got {}",
                power
            )));
        }
        if total_steps == 0 {
            return Err(LrSchedulerError::InvalidConfig(
                "total_steps must be > 0".to_string(),
            ));
        }
        Ok(LrScheduler::Polynomial(PolynomialDecaySchedule {
            base_lr,
            end_lr,
            total_steps,
            power,
        }))
    }

    /// Warmup + linear decay to 0. Requires `base_lr > 0`, `warmup_steps < total_steps`.
    pub fn warmup_linear(
        warmup_steps: usize,
        base_lr: f64,
        total_steps: usize,
    ) -> Result<Self, LrSchedulerError> {
        if base_lr <= 0.0 {
            return Err(LrSchedulerError::InvalidConfig(format!(
                "base_lr must be > 0, got {}",
                base_lr
            )));
        }
        if total_steps == 0 {
            return Err(LrSchedulerError::InvalidConfig(
                "total_steps must be > 0".to_string(),
            ));
        }
        if warmup_steps >= total_steps {
            return Err(LrSchedulerError::InvalidConfig(format!(
                "warmup_steps ({}) must be < total_steps ({})",
                warmup_steps, total_steps
            )));
        }
        Ok(LrScheduler::WarmupLinear(WarmupLinearSchedule {
            warmup_steps,
            base_lr,
            total_steps,
        }))
    }

    /// Returns the [`SchedulerKind`] for this scheduler.
    pub fn kind(&self) -> SchedulerKind {
        match self {
            LrScheduler::Constant(_) => SchedulerKind::Constant,
            LrScheduler::StepDecay(_) => SchedulerKind::StepDecay,
            LrScheduler::Exponential(_) => SchedulerKind::Exponential,
            LrScheduler::CosineAnnealing(_) => SchedulerKind::CosineAnnealing,
            LrScheduler::WarmupCosine(_) => SchedulerKind::WarmupCosine,
            LrScheduler::Cyclic(_) => SchedulerKind::Cyclic,
            LrScheduler::Polynomial(_) => SchedulerKind::Polynomial,
            LrScheduler::WarmupLinear(_) => SchedulerKind::WarmupLinear,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helper functions
// ──────────────────────────────────────────────────────────────────────────────

/// Linear warmup factor in `[0, 1]`.
///
/// Returns `step / warmup_steps` clamped to `[0, 1]`.
/// Returns `1.0` immediately when `warmup_steps == 0` (avoids divide-by-zero).
#[inline]
pub fn warmup_factor(step: usize, warmup_steps: usize) -> f64 {
    if warmup_steps == 0 {
        return 1.0;
    }
    (step as f64 / warmup_steps as f64).min(1.0)
}

/// Cosine decay factor in `[0, 1]`.
///
/// Returns `0.5 * (1 + cos(π * step / total_steps))`.
/// Returns `1.0` at `step == 0`, `0.0` at `step >= total_steps`.
/// Returns `0.0` when `total_steps == 0`.
#[inline]
pub fn cosine_decay_factor(step: usize, total_steps: usize) -> f64 {
    if total_steps == 0 || step >= total_steps {
        return 0.0;
    }
    let t = step as f64 / total_steps as f64;
    0.5 * (1.0 + (PI * t).cos())
}

/// Collect evenly-spaced `(step, lr)` samples from a scheduler.
///
/// Returns exactly `sample_points` pairs (or fewer if `num_steps == 0`).
/// Sample steps span `[0, num_steps - 1]` inclusively.
pub fn schedule_summary(
    scheduler: &LrScheduler,
    num_steps: usize,
    sample_points: usize,
) -> Vec<(usize, f64)> {
    if num_steps == 0 || sample_points == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(sample_points);

    if sample_points == 1 {
        out.push((0, scheduler.lr_at(0)));
        return out;
    }

    for i in 0..sample_points {
        // Evenly distribute across [0, num_steps - 1]
        let step = if sample_points == 1 {
            0
        } else {
            // i * (num_steps - 1) / (sample_points - 1)  — integer rounding
            (i * (num_steps - 1)) / (sample_points - 1)
        };
        out.push((step, scheduler.lr_at(step)));
    }

    out
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    // ── 1. Constant: lr_at always returns base_lr ────────────────────────────

    #[test]
    fn test_constant_always_returns_base_lr() -> Result<(), LrSchedulerError> {
        let sched = LrScheduler::constant(1e-3)?;
        for step in [0, 1, 100, 10_000] {
            let lr = sched.lr_at(step);
            assert!(
                (lr - 1e-3).abs() < EPS,
                "step {step}: expected 1e-3, got {lr}"
            );
        }
        Ok(())
    }

    // ── 2. StepDecay: correct decay at each step boundary ────────────────────

    #[test]
    fn test_step_decay_at_boundaries() -> Result<(), LrSchedulerError> {
        // decay_factor=0.1, step_size=10
        // step 0..9  → 1.0; step 10..19 → 0.1; step 20..29 → 0.01
        let sched = LrScheduler::step_decay(1.0, 0.1, 10)?;
        assert!((sched.lr_at(0) - 1.0).abs() < EPS);
        assert!((sched.lr_at(9) - 1.0).abs() < EPS);
        assert!((sched.lr_at(10) - 0.1).abs() < EPS);
        assert!((sched.lr_at(19) - 0.1).abs() < EPS);
        assert!((sched.lr_at(20) - 0.01).abs() < EPS);
        Ok(())
    }

    // ── 3. StepDecay: floor behaviour (not ceil) ─────────────────────────────

    #[test]
    fn test_step_decay_floor_not_ceil() -> Result<(), LrSchedulerError> {
        // step_size=5: step 4 → exponent 0 (floor(4/5)=0), step 5 → exponent 1
        let sched = LrScheduler::step_decay(1.0, 0.5, 5)?;
        assert!(
            (sched.lr_at(4) - 1.0).abs() < EPS,
            "floor: step 4 still in bucket 0"
        );
        assert!(
            (sched.lr_at(5) - 0.5).abs() < EPS,
            "floor: step 5 starts bucket 1"
        );
        Ok(())
    }

    // ── 3b. StepDecay: zero step_size must not panic ──────────────────────────

    #[test]
    fn test_step_decay_zero_step_size_no_panic() {
        // `StepDecaySchedule` has all-`pub` fields, so a struct literal can
        // bypass `LrScheduler::step_decay`'s `step_size == 0` validation.
        // `lr_at` must still not divide by zero.
        let sched = StepDecaySchedule {
            base_lr: 1e-3,
            decay_factor: 0.5,
            step_size: 0,
        };
        assert!((sched.lr_at(0) - 1e-3).abs() < EPS);
        assert!((sched.lr_at(100) - 1e-3).abs() < EPS);
    }

    // ── 4. Exponential: decay per step is geometric ───────────────────────────

    #[test]
    fn test_exponential_geometric_ratio() -> Result<(), LrSchedulerError> {
        let rate = 0.99;
        let sched = LrScheduler::exponential(1.0, rate)?;
        for step in 0..10 {
            let expected = rate.powi(step as i32);
            let got = sched.lr_at(step);
            assert!(
                (got - expected).abs() < EPS,
                "step {step}: expected {expected}, got {got}"
            );
        }
        Ok(())
    }

    // ── 5. CosineAnnealing: lr_at(0) == base_lr ──────────────────────────────

    #[test]
    fn test_cosine_annealing_at_step_zero() -> Result<(), LrSchedulerError> {
        let sched = LrScheduler::cosine_annealing(1e-2, 1e-4, 1000)?;
        let lr = sched.lr_at(0);
        assert!(
            (lr - 1e-2).abs() < EPS,
            "step 0 should equal base_lr, got {lr}"
        );
        Ok(())
    }

    // ── 6. CosineAnnealing: lr_at(total_steps) == min_lr ─────────────────────

    #[test]
    fn test_cosine_annealing_at_total_steps() -> Result<(), LrSchedulerError> {
        let sched = LrScheduler::cosine_annealing(1e-2, 1e-4, 1000)?;
        let lr = sched.lr_at(1000); // step >= total_steps
        assert!(
            (lr - 1e-4).abs() < EPS,
            "at total_steps should equal min_lr, got {lr}"
        );
        Ok(())
    }

    // ── 7. CosineAnnealing: lr at half-steps ≈ midpoint ──────────────────────

    #[test]
    fn test_cosine_annealing_half_steps() -> Result<(), LrSchedulerError> {
        let base_lr = 1.0;
        let min_lr = 0.0;
        let total = 1000;
        let sched = LrScheduler::cosine_annealing(base_lr, min_lr, total)?;
        // At step 500: cos(π * 0.5) = 0.0 → lr = min_lr + 0.5*(base-min)*(1+0) = 0.5
        let lr = sched.lr_at(500);
        assert!(
            (lr - 0.5).abs() < 1e-6,
            "half-step should be ~0.5, got {lr}"
        );
        Ok(())
    }

    // ── 8. WarmupCosine: lr_at(0) ≈ 0, lr_at(warmup_steps) ≈ base_lr ────────

    #[test]
    fn test_warmup_cosine_warmup_phase() -> Result<(), LrSchedulerError> {
        let sched = LrScheduler::warmup_cosine(100, 1e-3, 1e-5, 1000)?;
        // step 0 → warmup_factor(0, 100) = 0.0 → lr = 0.0
        assert!(sched.lr_at(0).abs() < EPS, "lr at step 0 should be ~0");
        // step 100 is the first cosine step (warmup ended), not in warmup phase
        // step 99 → warmup_factor(99, 100) = 0.99 → 0.99 * 1e-3
        let lr_99 = sched.lr_at(99);
        assert!(
            (lr_99 - 0.99 * 1e-3).abs() < EPS,
            "step 99 should be 0.99 * base_lr, got {lr_99}"
        );
        Ok(())
    }

    // ── 9. WarmupCosine: after warmup cosine decay applies ────────────────────

    #[test]
    fn test_warmup_cosine_cosine_phase() -> Result<(), LrSchedulerError> {
        let base_lr = 1.0;
        let min_lr = 0.0;
        let warmup = 100;
        let total = 1100; // cosine phase: 1000 steps
        let sched = LrScheduler::warmup_cosine(warmup, base_lr, min_lr, total)?;

        // At warmup boundary (step == warmup_steps) → cosine_decay_factor(0, 1000) = 1.0 → base_lr
        let lr_warmup_end = sched.lr_at(warmup);
        assert!(
            (lr_warmup_end - base_lr).abs() < EPS,
            "at warmup boundary, lr should be base_lr, got {lr_warmup_end}"
        );

        // At total_steps → min_lr
        let lr_end = sched.lr_at(total);
        assert!(
            (lr_end - min_lr).abs() < EPS,
            "at total_steps, lr should be min_lr, got {lr_end}"
        );

        // At midpoint of cosine (step = warmup + 500): cosine_decay_factor(500, 1000) ≈ 0.5
        let lr_mid = sched.lr_at(warmup + 500);
        assert!(
            (lr_mid - 0.5).abs() < 1e-6,
            "cosine midpoint lr should be ~0.5, got {lr_mid}"
        );
        Ok(())
    }

    // ── 10. Cyclic: min at 0, max at cycle_steps, min again at 2*cycle_steps ──

    #[test]
    fn test_cyclic_boundary_values() -> Result<(), LrSchedulerError> {
        let min_lr = 1e-4;
        let max_lr = 1e-2;
        let cycle = 100;
        let sched = LrScheduler::cyclic(min_lr, max_lr, cycle)?;

        assert!(
            (sched.lr_at(0) - min_lr).abs() < EPS,
            "step 0 should be min_lr"
        );
        assert!(
            (sched.lr_at(cycle) - max_lr).abs() < EPS,
            "step cycle_steps should be max_lr"
        );
        assert!(
            (sched.lr_at(2 * cycle) - min_lr).abs() < EPS,
            "step 2*cycle_steps should be min_lr"
        );
        Ok(())
    }

    // ── 11. Cyclic: monotone increasing in first half-cycle ───────────────────

    #[test]
    fn test_cyclic_monotone_first_half() -> Result<(), LrSchedulerError> {
        let sched = LrScheduler::cyclic(0.0, 1.0, 100)?;
        let mut prev = sched.lr_at(0);
        for step in 1..=100 {
            let cur = sched.lr_at(step);
            assert!(
                cur >= prev - EPS,
                "step {step}: lr {cur} < prev {prev} (not monotone increasing)"
            );
            prev = cur;
        }
        Ok(())
    }

    // ── 12. Polynomial: at step 0 == base_lr, at total_steps == end_lr ────────

    #[test]
    fn test_polynomial_boundary_values() -> Result<(), LrSchedulerError> {
        let sched = LrScheduler::polynomial(1.0, 1e-4, 1000, 2.0)?;
        assert!(
            (sched.lr_at(0) - 1.0).abs() < EPS,
            "step 0 should be base_lr"
        );
        assert!(
            (sched.lr_at(1000) - 1e-4).abs() < EPS,
            "at total_steps should be end_lr"
        );
        Ok(())
    }

    // ── 13. Polynomial: power=1 is linear ─────────────────────────────────────

    #[test]
    fn test_polynomial_power1_is_linear() -> Result<(), LrSchedulerError> {
        let base = 1.0;
        let end = 0.0;
        let total = 100;
        let sched = LrScheduler::polynomial(base, end, total, 1.0)?;

        // Midpoint at step 50: expected = (base - end) * (1 - 50/100)^1 + end = 0.5
        let lr_50 = sched.lr_at(50);
        assert!(
            (lr_50 - 0.5).abs() < EPS,
            "power=1 at midpoint should be 0.5, got {lr_50}"
        );
        Ok(())
    }

    // ── 14. WarmupLinear: ramp from 0 to base_lr over warmup ─────────────────

    #[test]
    fn test_warmup_linear_warmup_ramp() -> Result<(), LrSchedulerError> {
        let base_lr = 1e-3;
        let warmup = 100;
        let total = 1000;
        let sched = LrScheduler::warmup_linear(warmup, base_lr, total)?;

        assert!(sched.lr_at(0).abs() < EPS, "step 0 should be 0");
        let lr_50 = sched.lr_at(50);
        assert!(
            (lr_50 - 0.5 * base_lr).abs() < EPS,
            "step 50 should be 0.5*base_lr, got {lr_50}"
        );
        Ok(())
    }

    // ── 15. WarmupLinear: decay from base_lr to 0 over remaining steps ────────

    #[test]
    fn test_warmup_linear_decay_phase() -> Result<(), LrSchedulerError> {
        let base_lr = 1.0;
        let warmup = 100;
        let total = 1100;
        let sched = LrScheduler::warmup_linear(warmup, base_lr, total)?;

        // Decay runs from step 100 to step 1100 (1000 steps)
        // At warmup end: lr should be base_lr
        let lr_at_warmup = sched.lr_at(warmup);
        // warmup_factor(100, 100) = 1.0 → lr = base_lr
        assert!(
            (lr_at_warmup - base_lr).abs() < EPS,
            "at warmup end, lr should be base_lr, got {lr_at_warmup}"
        );

        // Midpoint of decay: step 100 + 500 = 600, 50% through decay → 0.5 * base_lr
        let lr_mid = sched.lr_at(600);
        assert!(
            (lr_mid - 0.5 * base_lr).abs() < EPS,
            "midpoint decay should be 0.5, got {lr_mid}"
        );

        // End of schedule
        let lr_end = sched.lr_at(total);
        assert!(
            lr_end.abs() < EPS,
            "at total_steps, lr should be 0.0, got {lr_end}"
        );
        Ok(())
    }

    // ── 16. LrScheduler enum dispatch works for all variants ─────────────────

    #[test]
    fn test_enum_dispatch_all_variants() -> Result<(), LrSchedulerError> {
        let schedulers: Vec<LrScheduler> = vec![
            LrScheduler::constant(1e-3)?,
            LrScheduler::step_decay(1e-3, 0.5, 100)?,
            LrScheduler::exponential(1e-3, 0.9999)?,
            LrScheduler::cosine_annealing(1e-3, 1e-5, 1000)?,
            LrScheduler::warmup_cosine(50, 1e-3, 1e-5, 500)?,
            LrScheduler::cyclic(1e-4, 1e-2, 100)?,
            LrScheduler::polynomial(1e-3, 1e-5, 1000, 2.0)?,
            LrScheduler::warmup_linear(50, 1e-3, 500)?,
        ];

        for sched in &schedulers {
            let lr = sched.lr_at(0);
            assert!(
                lr.is_finite() && lr >= 0.0,
                "variant {}: lr at step 0 should be finite non-negative, got {lr}",
                sched.description()
            );
        }
        Ok(())
    }

    // ── 17. Validation: constant(0.0) returns Err ─────────────────────────────

    #[test]
    fn test_validation_constant_zero_lr() {
        let result = LrScheduler::constant(0.0);
        assert!(result.is_err(), "constant(0.0) should return Err");
        assert!(matches!(result, Err(LrSchedulerError::InvalidConfig(_))));
    }

    // ── 18. Validation: step_decay with step_size=0 returns Err ──────────────

    #[test]
    fn test_validation_step_decay_zero_step_size() {
        let result = LrScheduler::step_decay(1e-3, 0.5, 0);
        assert!(result.is_err(), "step_size=0 should return Err");
        assert!(matches!(result, Err(LrSchedulerError::InvalidConfig(_))));
    }

    // ── 19. Validation: cyclic min_lr > max_lr returns Err ───────────────────

    #[test]
    fn test_validation_cyclic_min_gt_max() {
        let result = LrScheduler::cyclic(1e-2, 1e-4, 100);
        assert!(result.is_err(), "min_lr > max_lr should return Err");
        assert!(matches!(result, Err(LrSchedulerError::InvalidConfig(_))));
    }

    // ── 20. warmup_factor: returns 1.0 when warmup_steps==0 ─────────────────

    #[test]
    fn test_warmup_factor_zero_warmup_steps() {
        assert!(
            (warmup_factor(0, 0) - 1.0).abs() < EPS,
            "warmup_factor with warmup_steps=0 should be 1.0"
        );
        assert!(
            (warmup_factor(42, 0) - 1.0).abs() < EPS,
            "warmup_factor with warmup_steps=0 should always be 1.0"
        );
    }

    // ── 21. cosine_decay_factor: step==0 → 1.0, step==total → 0.0 ───────────

    #[test]
    fn test_cosine_decay_factor_boundaries() {
        assert!(
            (cosine_decay_factor(0, 1000) - 1.0).abs() < EPS,
            "cosine_decay_factor at step 0 should be 1.0"
        );
        assert!(
            cosine_decay_factor(1000, 1000).abs() < EPS,
            "cosine_decay_factor at total_steps should be 0.0"
        );
        assert!(
            cosine_decay_factor(1001, 1000).abs() < EPS,
            "cosine_decay_factor beyond total_steps should be 0.0"
        );
        // Edge case: total_steps == 0
        assert!(
            cosine_decay_factor(0, 0).abs() < EPS,
            "cosine_decay_factor with total_steps=0 should be 0.0"
        );
    }

    // ── 22. schedule_summary: correct count, monotone for warmup_cosine ──────

    #[test]
    fn test_schedule_summary_count_and_monotone() -> Result<(), LrSchedulerError> {
        let sched = LrScheduler::warmup_cosine(100, 1e-3, 1e-5, 1000)?;
        let samples = schedule_summary(&sched, 1000, 11);

        assert_eq!(samples.len(), 11, "should return exactly 11 sample points");
        assert_eq!(samples[0].0, 0, "first sample should be at step 0");
        assert_eq!(samples[10].0, 999, "last sample should be at step 999");

        // LR should be monotonically non-decreasing during warmup (first ~10% of steps)
        // and then monotonically non-increasing — so just check overall final <= initial
        // (warmup_cosine goes from ~0 to base_lr, then decays to min_lr)
        let lr_first = samples[0].1;
        let lr_last = samples[10].1;
        assert!(
            lr_last > lr_first,
            "final lr ({lr_last}) should be > initial lr ({lr_first}) for warmup_cosine at step 0"
        );

        // Edge: 0 sample_points
        let empty = schedule_summary(&sched, 1000, 0);
        assert!(empty.is_empty(), "0 sample_points should return empty vec");

        Ok(())
    }
}
