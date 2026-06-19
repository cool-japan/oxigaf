//! Training curriculum scheduler for progressively increasing task difficulty.
//!
//! A curriculum organises training into [`CurriculumStage`]s, each covering a
//! range of training steps and associating difficulty values to named
//! [`DifficultyDimension`]s.  A [`CurriculumController`] wraps a
//! [`CurriculumSchedule`] and produces a ready-to-use [`CurriculumState`] at
//! any training step, optionally supporting loss-threshold-based stage
//! advancement.

use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the curriculum subsystem.
#[derive(Debug, Error)]
pub enum CurriculumError {
    #[error("curriculum schedule has no stages")]
    EmptySchedule,

    #[error("stages overlap: '{stage_a}' and '{stage_b}'")]
    OverlappingStages { stage_a: String, stage_b: String },

    #[error("stage '{stage}' is unsorted: prev_end={prev_end}, start={start}")]
    UnsortedStages {
        stage: String,
        prev_end: usize,
        start: usize,
    },

    #[error("invalid schedule: {0}")]
    InvalidSchedule(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// DifficultyDimension
// ─────────────────────────────────────────────────────────────────────────────

/// Axes along which training difficulty is parameterised.
#[derive(Debug, Clone, PartialEq)]
pub enum DifficultyDimension {
    /// Render image resolution (width = height for square images).
    RenderResolution,
    /// Number of diffusion views generated per optimisation step.
    NumViews,
    /// Diffusion guidance scale (classifier-free guidance weight).
    GuidanceScale,
    /// Noise power / perturbation standard deviation.
    NoisePower,
    /// Spherical-harmonics degree (0 …= 3).
    ShDegree,
    /// Number of DDIM denoising steps.
    DdimSteps,
}

impl DifficultyDimension {
    /// Stable kebab-case name used as the key in `HashMap<String, f64>`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::RenderResolution => "render_resolution",
            Self::NumViews => "num_views",
            Self::GuidanceScale => "guidance_scale",
            Self::NoisePower => "noise_power",
            Self::ShDegree => "sh_degree",
            Self::DdimSteps => "ddim_steps",
        }
    }
}

impl fmt::Display for DifficultyDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CurriculumStage
// ─────────────────────────────────────────────────────────────────────────────

/// One phase of a training curriculum, covering a half-open step interval
/// `[step_start, step_end)`.  Use `step_end = usize::MAX` for an open-ended
/// final stage.
#[derive(Debug, Clone)]
pub struct CurriculumStage {
    /// Human-readable label (e.g. `"warmup"`, `"stage_1"`).
    pub name: String,
    /// First training step that belongs to this stage (inclusive).
    pub step_start: usize,
    /// First training step that belongs to the *next* stage (exclusive).
    /// Set to `usize::MAX` for an unbounded final stage.
    pub step_end: usize,
    /// Difficulty values keyed by [`DifficultyDimension::name`].
    pub difficulty_values: HashMap<String, f64>,
    /// If `Some(threshold)`, the curriculum controller may advance to the next
    /// stage early once the smoothed loss falls below this value.
    pub advance_loss_threshold: Option<f32>,
}

impl CurriculumStage {
    /// Create a new stage with the given name and step range.
    pub fn new(name: impl Into<String>, step_start: usize, step_end: usize) -> Self {
        Self {
            name: name.into(),
            step_start,
            step_end,
            difficulty_values: HashMap::new(),
            advance_loss_threshold: None,
        }
    }

    /// Builder: set the value for a difficulty dimension.
    pub fn with_value(mut self, dim: &DifficultyDimension, value: f64) -> Self {
        self.difficulty_values.insert(dim.name().to_owned(), value);
        self
    }

    /// Builder: set an optional loss threshold for early advancement.
    pub fn with_advance_threshold(mut self, loss: f32) -> Self {
        self.advance_loss_threshold = Some(loss);
        self
    }

    /// Look up the value of a dimension in this stage, if configured.
    pub fn get_value(&self, dim: &DifficultyDimension) -> Option<f64> {
        self.difficulty_values.get(dim.name()).copied()
    }

    /// Returns `true` if `step` falls within `[step_start, step_end)`.
    pub fn contains_step(&self, step: usize) -> bool {
        step >= self.step_start && step < self.step_end
    }

    /// Number of steps in this stage, or `usize::MAX` for the final open stage.
    pub fn duration(&self) -> usize {
        if self.step_end == usize::MAX {
            usize::MAX
        } else {
            self.step_end - self.step_start
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CurriculumSchedule
// ─────────────────────────────────────────────────────────────────────────────

/// An ordered collection of [`CurriculumStage`]s that together define how
/// training difficulty evolves over time.
#[derive(Debug, Clone)]
pub struct CurriculumSchedule {
    stages: Vec<CurriculumStage>,
    /// When `true`, difficulty values between adjacent stages are linearly
    /// interpolated based on intra-stage progress.  When `false`, the active
    /// stage's value is returned unchanged (step function).
    pub interpolate: bool,
}

impl CurriculumSchedule {
    /// Create an empty schedule with step-function (no interpolation) mode.
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            interpolate: false,
        }
    }

    /// Builder: enable linear interpolation between consecutive stage values.
    pub fn with_interpolation(mut self) -> Self {
        self.interpolate = true;
        self
    }

    /// Builder: append a stage to the schedule.
    pub fn add_stage(mut self, stage: CurriculumStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Number of stages currently in the schedule.
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Validate the schedule: stages must be non-overlapping and sorted by
    /// `step_start`.  Returns an error describing the first violation found.
    pub fn validate(&self) -> Result<(), CurriculumError> {
        if self.stages.is_empty() {
            return Err(CurriculumError::EmptySchedule);
        }
        let mut prev_end: usize = 0;
        for (i, stage) in self.stages.iter().enumerate() {
            if stage.step_start < prev_end && i > 0 {
                let prev = &self.stages[i - 1];
                // Decide whether it is an overlap or unsorted order.
                if stage.step_start < prev.step_start {
                    return Err(CurriculumError::UnsortedStages {
                        stage: stage.name.clone(),
                        prev_end,
                        start: stage.step_start,
                    });
                }
                return Err(CurriculumError::OverlappingStages {
                    stage_a: prev.name.clone(),
                    stage_b: stage.name.clone(),
                });
            }
            prev_end = stage.step_end;
        }
        Ok(())
    }

    /// Find the stage that is active at `step`, or `None` if `step` precedes
    /// the first stage.
    pub fn active_stage_at(&self, step: usize) -> Option<&CurriculumStage> {
        // Linear scan; schedules are short (typically < 10 stages).
        for stage in &self.stages {
            if stage.contains_step(step) {
                return Some(stage);
            }
        }
        // If step is past the last finite stage, return the last stage.
        self.stages.last().filter(|s| step >= s.step_start)
    }

    /// Get the index of the active stage at `step`, or `None`.
    fn active_stage_index_at(&self, step: usize) -> Option<usize> {
        for (i, stage) in self.stages.iter().enumerate() {
            if stage.contains_step(step) {
                return Some(i);
            }
        }
        // Past the last stage → return the last index.
        if self.stages.last().is_some_and(|s| step >= s.step_start) {
            Some(self.stages.len() - 1)
        } else {
            None
        }
    }

    /// Look up a difficulty value at the given training step.
    ///
    /// - When `interpolate = false`: returns the active stage's value.
    /// - When `interpolate = true`: linearly blends between the current and
    ///   next stage's values using intra-stage progress, falling back to the
    ///   current value when the dimension is missing from the next stage or when
    ///   the current stage is the last one.
    ///
    /// Returns `None` if the dimension is not configured in any reachable stage.
    pub fn value_at(&self, step: usize, dim: &DifficultyDimension) -> Option<f64> {
        let idx = self.active_stage_index_at(step)?;
        let current = &self.stages[idx];
        let current_val = current.get_value(dim)?;

        if !self.interpolate {
            return Some(current_val);
        }

        // Interpolation is only possible when:
        //   1. There is a next stage.
        //   2. The next stage also has this dimension.
        //   3. The current stage has a finite duration (step_end != usize::MAX).
        let next = self.stages.get(idx + 1);
        let (Some(next_stage), Some(next_val)) = (next, next.and_then(|s| s.get_value(dim))) else {
            return Some(current_val);
        };

        if current.step_end == usize::MAX {
            return Some(current_val);
        }

        // Intra-stage fractional progress.
        let duration = (current.step_end - current.step_start) as f64;
        let elapsed = (step.saturating_sub(current.step_start)) as f64;
        let t = if duration > 0.0 {
            (elapsed / duration).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Linear interpolation: current → next (transition happens at step_end).
        let _ = next_stage; // suppress unused warning
        Some(current_val + t * (next_val - current_val))
    }

    /// Fractional progress within the active stage: 0.0 = stage start,
    /// 1.0 = stage end (clamped, infinite stages return 0.0).
    pub fn stage_progress(&self, step: usize) -> f32 {
        let Some(stage) = self.active_stage_at(step) else {
            return 0.0;
        };
        if stage.step_end == usize::MAX {
            return 0.0;
        }
        let duration = stage.step_end - stage.step_start;
        if duration == 0 {
            return 1.0;
        }
        let elapsed = step.saturating_sub(stage.step_start);
        (elapsed as f32 / duration as f32).clamp(0.0, 1.0)
    }

    /// Total training duration: the `step_end` of the last stage, or
    /// `usize::MAX` if the last stage is open-ended.
    pub fn total_steps(&self) -> usize {
        self.stages.last().map(|s| s.step_end).unwrap_or(usize::MAX)
    }

    /// Default four-stage GAF curriculum.
    ///
    /// | Stage | Steps             | Res   | Views | Guidance | SH |
    /// |-------|-------------------|-------|-------|----------|----|
    /// | 0     | 0 …< 2 000        | 64    | 1     | 1.0      | 0  |
    /// | 1     | 2 000 …< 8 000    | 128   | 2     | 3.0      | 1  |
    /// | 2     | 8 000 …< 20 000   | 256   | 4     | 5.0      | 2  |
    /// | 3     | 20 000 …< MAX     | 512   | 4     | 7.5      | 3  |
    pub fn default_gaf() -> Self {
        let stage0 = CurriculumStage::new("stage_0", 0, 2_000)
            .with_value(&DifficultyDimension::RenderResolution, 64.0)
            .with_value(&DifficultyDimension::NumViews, 1.0)
            .with_value(&DifficultyDimension::GuidanceScale, 1.0)
            .with_value(&DifficultyDimension::ShDegree, 0.0);

        let stage1 = CurriculumStage::new("stage_1", 2_000, 8_000)
            .with_value(&DifficultyDimension::RenderResolution, 128.0)
            .with_value(&DifficultyDimension::NumViews, 2.0)
            .with_value(&DifficultyDimension::GuidanceScale, 3.0)
            .with_value(&DifficultyDimension::ShDegree, 1.0);

        let stage2 = CurriculumStage::new("stage_2", 8_000, 20_000)
            .with_value(&DifficultyDimension::RenderResolution, 256.0)
            .with_value(&DifficultyDimension::NumViews, 4.0)
            .with_value(&DifficultyDimension::GuidanceScale, 5.0)
            .with_value(&DifficultyDimension::ShDegree, 2.0);

        let stage3 = CurriculumStage::new("stage_3", 20_000, usize::MAX)
            .with_value(&DifficultyDimension::RenderResolution, 512.0)
            .with_value(&DifficultyDimension::NumViews, 4.0)
            .with_value(&DifficultyDimension::GuidanceScale, 7.5)
            .with_value(&DifficultyDimension::ShDegree, 3.0);

        Self::new()
            .add_stage(stage0)
            .add_stage(stage1)
            .add_stage(stage2)
            .add_stage(stage3)
    }
}

impl Default for CurriculumSchedule {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CurriculumConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Runtime configuration for the [`CurriculumController`].
#[derive(Debug, Clone)]
pub struct CurriculumConfig {
    /// Allow the controller to advance the stage when the smoothed loss falls
    /// below the current stage's `advance_loss_threshold` (if set).
    pub enable_loss_advance: bool,
    /// Once a stage has been advanced (by loss or manually), prevent regression
    /// to an earlier stage based on the current step alone.
    pub no_regression: bool,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            enable_loss_advance: true,
            no_regression: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CurriculumState
// ─────────────────────────────────────────────────────────────────────────────

/// A snapshot of the curriculum at a specific training step, ready for use by
/// the training loop.
#[derive(Debug, Clone)]
pub struct CurriculumState {
    /// Training step at which this state was sampled.
    pub step: usize,
    /// Human-readable name of the currently active stage.
    pub stage_name: String,
    /// Zero-based index of the currently active stage.
    pub stage_index: usize,
    /// Render image resolution (square; pixels per side).
    pub render_resolution: usize,
    /// Number of diffusion views to generate per step.
    pub num_views: usize,
    /// Diffusion guidance scale.
    pub guidance_scale: f64,
    /// Noise power / perturbation standard deviation.
    pub noise_power: f64,
    /// Spherical-harmonics degree (0 …= 3).
    pub sh_degree: u32,
    /// Number of DDIM denoising steps.
    pub ddim_steps: usize,
    /// Fractional progress within the current stage (0.0 …= 1.0).
    pub stage_progress: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// CurriculumController
// ─────────────────────────────────────────────────────────────────────────────

/// High-level curriculum controller that combines a [`CurriculumSchedule`]
/// with optional loss-based stage advancement.
pub struct CurriculumController {
    schedule: CurriculumSchedule,
    config: CurriculumConfig,
    current_stage_idx: usize,
    /// Rolling window of recently recorded loss values.
    recent_losses: Vec<f32>,
    /// Maximum number of losses kept for the rolling average.
    recent_loss_window: usize,
}

impl CurriculumController {
    /// Create a controller from a validated schedule and configuration.
    ///
    /// # Errors
    /// Returns [`CurriculumError`] if the schedule fails validation.
    pub fn new(
        schedule: CurriculumSchedule,
        config: CurriculumConfig,
    ) -> Result<Self, CurriculumError> {
        schedule.validate()?;
        Ok(Self {
            schedule,
            config,
            current_stage_idx: 0,
            recent_losses: Vec::new(),
            recent_loss_window: 100,
        })
    }

    /// Compute the effective stage index at `step`, honouring `no_regression`.
    fn effective_stage_index(&self, step: usize) -> usize {
        let step_based = self
            .schedule
            .active_stage_index_at(step)
            .unwrap_or(self.schedule.stages.len().saturating_sub(1));

        if self.config.no_regression {
            step_based.max(self.current_stage_idx)
        } else {
            step_based
        }
    }

    /// Produce a [`CurriculumState`] describing the curriculum at `step`.
    pub fn state_at(&self, step: usize) -> CurriculumState {
        let eff_idx = self.effective_stage_index(step);
        let stage_name = self
            .schedule
            .stages
            .get(eff_idx)
            .map(|s| s.name.clone())
            .unwrap_or_else(|| "unknown".to_owned());

        let get_val = |dim: &DifficultyDimension, default: f64| -> f64 {
            self.schedule.value_at(step, dim).unwrap_or(default)
        };

        let render_resolution =
            get_val(&DifficultyDimension::RenderResolution, 256.0).round() as usize;
        let num_views = get_val(&DifficultyDimension::NumViews, 1.0).round() as usize;
        let guidance_scale = get_val(&DifficultyDimension::GuidanceScale, 7.5);
        let noise_power = get_val(&DifficultyDimension::NoisePower, 0.0);
        let sh_degree = (get_val(&DifficultyDimension::ShDegree, 3.0).round() as u32).min(3);
        let ddim_steps = get_val(&DifficultyDimension::DdimSteps, 20.0).round() as usize;

        let stage_progress = self.schedule.stage_progress(step);

        CurriculumState {
            step,
            stage_name,
            stage_index: eff_idx,
            render_resolution,
            num_views,
            guidance_scale,
            noise_power,
            sh_degree,
            ddim_steps,
            stage_progress,
        }
    }

    /// Record a loss value.  If loss-based advancement is enabled and the
    /// rolling-window average falls below the current stage's threshold, the
    /// stage is advanced.
    ///
    /// Returns `true` if the stage was advanced as a result.
    pub fn record_loss(&mut self, step: usize, loss: f32) -> bool {
        self.recent_losses.push(loss);
        if self.recent_losses.len() > self.recent_loss_window {
            self.recent_losses.remove(0);
        }

        if !self.config.enable_loss_advance {
            return false;
        }

        let stage = match self.schedule.stages.get(self.current_stage_idx) {
            Some(s) => s,
            None => return false,
        };
        let threshold = match stage.advance_loss_threshold {
            Some(t) => t,
            None => return false,
        };

        // Require a full window before acting.
        if self.recent_losses.len() < self.recent_loss_window {
            return false;
        }

        let avg: f32 =
            self.recent_losses.iter().copied().sum::<f32>() / self.recent_losses.len() as f32;

        if avg < threshold {
            let advanced = self.advance_stage_inner(step);
            if advanced {
                self.recent_losses.clear();
            }
            return advanced;
        }

        false
    }

    /// Internal helper: advance one stage without clearing losses.
    fn advance_stage_inner(&mut self, _step: usize) -> bool {
        if self.current_stage_idx + 1 < self.schedule.stages.len() {
            self.current_stage_idx += 1;
            return true;
        }
        false
    }

    /// Manually advance to the next stage.  Returns `true` if successful
    /// (i.e. not already at the last stage).
    pub fn advance_stage(&mut self) -> bool {
        self.advance_stage_inner(0)
    }

    /// Zero-based index of the controller's current stage.
    pub fn current_stage_idx(&self) -> usize {
        self.current_stage_idx
    }

    /// Borrow the underlying [`CurriculumSchedule`].
    pub fn schedule(&self) -> &CurriculumSchedule {
        &self.schedule
    }

    /// Format the current curriculum state as a human-readable string.
    pub fn format_state(&self, step: usize) -> String {
        let s = self.state_at(step);
        format!(
            "step={step:>7} | stage[{idx}]={name} | res={res}² | \
             views={views} | guidance={guidance:.2} | \
             sh={sh} | ddim={ddim} | noise={noise:.4} | progress={prog:.1}%",
            step = s.step,
            idx = s.stage_index,
            name = s.stage_name,
            res = s.render_resolution,
            views = s.num_views,
            guidance = s.guidance_scale,
            sh = s.sh_degree,
            ddim = s.ddim_steps,
            noise = s.noise_power,
            prog = s.stage_progress * 100.0,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProgressTracker
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight bookkeeping for overall training progress across stages.
#[derive(Debug, Clone)]
pub struct ProgressTracker {
    /// Names of stages that have been marked as complete.
    pub completed_stages: Vec<String>,
    /// Expected total number of training steps (informational).
    pub total_steps: usize,
    /// Last training step recorded via [`Self::advance`].
    pub current_step: usize,
}

impl ProgressTracker {
    /// Create a fresh tracker.
    pub fn new() -> Self {
        Self {
            completed_stages: Vec::new(),
            total_steps: 0,
            current_step: 0,
        }
    }

    /// Update the current step counter.
    pub fn advance(&mut self, step: usize) {
        self.current_step = step;
    }

    /// Record a stage as having been completed.
    pub fn mark_stage_complete(&mut self, stage_name: String) {
        self.completed_stages.push(stage_name);
    }

    /// Fraction of `total_steps` completed (clamped to [0.0, 1.0]).
    /// Returns 0.0 if `total_steps` is zero or `usize::MAX`.
    pub fn completion_fraction(&self, total_steps: usize) -> f32 {
        if total_steps == 0 || total_steps == usize::MAX {
            return 0.0;
        }
        (self.current_step as f32 / total_steps as f32).clamp(0.0, 1.0)
    }

    /// Human-readable progress string.
    pub fn format_progress(&self) -> String {
        format!(
            "step {}/{} | stages complete: [{}]",
            self.current_step,
            if self.total_steps == usize::MAX {
                "∞".to_owned()
            } else {
                self.total_steps.to_string()
            },
            self.completed_stages.join(", ")
        )
    }
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DifficultyDimension ───────────────────────────────────────────────

    #[test]
    fn test_difficulty_dimension_name() {
        assert_eq!(
            DifficultyDimension::RenderResolution.name(),
            "render_resolution"
        );
        assert_eq!(DifficultyDimension::NumViews.name(), "num_views");
        assert_eq!(DifficultyDimension::GuidanceScale.name(), "guidance_scale");
        assert_eq!(DifficultyDimension::NoisePower.name(), "noise_power");
        assert_eq!(DifficultyDimension::ShDegree.name(), "sh_degree");
        assert_eq!(DifficultyDimension::DdimSteps.name(), "ddim_steps");
    }

    // ── CurriculumStage ───────────────────────────────────────────────────

    #[test]
    fn test_curriculum_stage_contains_step() {
        let stage = CurriculumStage::new("s0", 100, 200);
        assert!(!stage.contains_step(99));
        assert!(stage.contains_step(100));
        assert!(stage.contains_step(150));
        assert!(stage.contains_step(199));
        assert!(!stage.contains_step(200));
    }

    #[test]
    fn test_curriculum_stage_with_value() {
        let stage = CurriculumStage::new("s0", 0, 100)
            .with_value(&DifficultyDimension::RenderResolution, 64.0)
            .with_value(&DifficultyDimension::NumViews, 2.0);

        assert_eq!(
            stage.get_value(&DifficultyDimension::RenderResolution),
            Some(64.0)
        );
        assert_eq!(stage.get_value(&DifficultyDimension::NumViews), Some(2.0));
        assert_eq!(stage.get_value(&DifficultyDimension::GuidanceScale), None);
    }

    // ── CurriculumSchedule – add / validate ──────────────────────────────

    #[test]
    fn test_schedule_add_and_validate() -> Result<(), CurriculumError> {
        let schedule = CurriculumSchedule::new()
            .add_stage(CurriculumStage::new("a", 0, 100))
            .add_stage(CurriculumStage::new("b", 100, 200));
        schedule.validate()?;
        assert_eq!(schedule.num_stages(), 2);
        Ok(())
    }

    #[test]
    fn test_schedule_unsorted_error() {
        let schedule = CurriculumSchedule::new()
            .add_stage(CurriculumStage::new("b", 100, 200))
            .add_stage(CurriculumStage::new("a", 0, 100));
        let result = schedule.validate();
        assert!(
            matches!(
                result,
                Err(CurriculumError::UnsortedStages { .. })
                    | Err(CurriculumError::OverlappingStages { .. })
            ),
            "expected unsorted/overlapping error"
        );
    }

    #[test]
    fn test_schedule_overlapping_error() {
        let schedule = CurriculumSchedule::new()
            .add_stage(CurriculumStage::new("a", 0, 150))
            .add_stage(CurriculumStage::new("b", 100, 200));
        let result = schedule.validate();
        assert!(
            matches!(result, Err(CurriculumError::OverlappingStages { .. })),
            "expected overlapping error, got {:?}",
            result
        );
    }

    // ── CurriculumSchedule – active stage ────────────────────────────────

    #[test]
    fn test_schedule_active_stage_at() {
        let schedule = CurriculumSchedule::new()
            .add_stage(CurriculumStage::new("first", 0, 500))
            .add_stage(CurriculumStage::new("second", 500, usize::MAX));

        let s0 = schedule.active_stage_at(0);
        assert!(s0.map(|s| s.name.as_str()) == Some("first"));

        let s1 = schedule.active_stage_at(499);
        assert!(s1.map(|s| s.name.as_str()) == Some("first"));

        let s2 = schedule.active_stage_at(500);
        assert!(s2.map(|s| s.name.as_str()) == Some("second"));

        let s3 = schedule.active_stage_at(1_000_000);
        assert!(s3.map(|s| s.name.as_str()) == Some("second"));
    }

    // ── CurriculumSchedule – value_at (no interpolation) ─────────────────

    #[test]
    fn test_schedule_value_at_no_interpolation() -> Result<(), CurriculumError> {
        let schedule = CurriculumSchedule::new()
            .add_stage(
                CurriculumStage::new("lo", 0, 1000)
                    .with_value(&DifficultyDimension::RenderResolution, 64.0),
            )
            .add_stage(
                CurriculumStage::new("hi", 1000, usize::MAX)
                    .with_value(&DifficultyDimension::RenderResolution, 256.0),
            );
        schedule.validate()?;

        let val_lo = schedule.value_at(500, &DifficultyDimension::RenderResolution);
        assert_eq!(val_lo, Some(64.0));

        let val_hi = schedule.value_at(1500, &DifficultyDimension::RenderResolution);
        assert_eq!(val_hi, Some(256.0));

        Ok(())
    }

    // ── CurriculumSchedule – value_at (with interpolation) ───────────────

    #[test]
    fn test_schedule_value_at_with_interpolation() -> Result<(), CurriculumError> {
        let schedule = CurriculumSchedule::new()
            .with_interpolation()
            .add_stage(
                CurriculumStage::new("lo", 0, 1000)
                    .with_value(&DifficultyDimension::GuidanceScale, 1.0),
            )
            .add_stage(
                CurriculumStage::new("hi", 1000, usize::MAX)
                    .with_value(&DifficultyDimension::GuidanceScale, 5.0),
            );
        schedule.validate()?;

        // At step 0 progress = 0 → value should be 1.0.
        let v0 = schedule.value_at(0, &DifficultyDimension::GuidanceScale);
        assert_eq!(v0, Some(1.0));

        // At step 500 progress = 0.5 → value should be 1.0 + 0.5*(5.0−1.0) = 3.0
        let v500 = schedule.value_at(500, &DifficultyDimension::GuidanceScale);
        let v500 = v500.ok_or_else(|| CurriculumError::InvalidSchedule("missing".into()))?;
        assert!((v500 - 3.0).abs() < 1e-9, "expected 3.0 got {v500}");

        // At or after step_end, in the next stage → value = 5.0 (no further blending).
        let v1000 = schedule.value_at(1000, &DifficultyDimension::GuidanceScale);
        assert_eq!(v1000, Some(5.0));

        Ok(())
    }

    // ── CurriculumSchedule – stage_progress ──────────────────────────────

    #[test]
    fn test_schedule_stage_progress() -> Result<(), CurriculumError> {
        let schedule = CurriculumSchedule::new()
            .add_stage(CurriculumStage::new("a", 0, 1000))
            .add_stage(CurriculumStage::new("b", 1000, usize::MAX));
        schedule.validate()?;

        let p0 = schedule.stage_progress(0);
        assert!((p0 - 0.0).abs() < 1e-6, "p0={p0}");

        let p500 = schedule.stage_progress(500);
        assert!((p500 - 0.5).abs() < 1e-5, "p500={p500}");

        let p999 = schedule.stage_progress(999);
        assert!(p999 > 0.99, "p999={p999}");

        // Open-ended stage always returns 0.0.
        let p_inf = schedule.stage_progress(5_000_000);
        assert_eq!(p_inf, 0.0);

        Ok(())
    }

    // ── CurriculumSchedule – default_gaf ─────────────────────────────────

    #[test]
    fn test_schedule_default_gaf() -> Result<(), CurriculumError> {
        let schedule = CurriculumSchedule::default_gaf();
        schedule.validate()?;
        assert_eq!(schedule.num_stages(), 4);

        // Stage 0 at step 0.
        let s0 = schedule.active_stage_at(0);
        assert!(s0.map(|s| s.name.as_str()) == Some("stage_0"));

        // Stage 3 at step 20_000.
        let s3 = schedule.active_stage_at(20_000);
        assert!(
            s3.map(|s| s.name.as_str()) == Some("stage_3"),
            "{:?}",
            s3.map(|s| &s.name)
        );

        // Resolution check.
        let res0 = schedule.value_at(0, &DifficultyDimension::RenderResolution);
        assert_eq!(res0, Some(64.0));

        let res3 = schedule.value_at(50_000, &DifficultyDimension::RenderResolution);
        assert_eq!(res3, Some(512.0));

        // Total steps → usize::MAX (open-ended last stage).
        assert_eq!(schedule.total_steps(), usize::MAX);
        Ok(())
    }

    // ── CurriculumController – state_at ──────────────────────────────────

    #[test]
    fn test_controller_state_at() -> Result<(), CurriculumError> {
        let schedule = CurriculumSchedule::default_gaf();
        let config = CurriculumConfig::default();
        let ctrl = CurriculumController::new(schedule, config)?;

        let state = ctrl.state_at(0);
        assert_eq!(state.render_resolution, 64);
        assert_eq!(state.num_views, 1);
        assert_eq!(state.stage_name, "stage_0");
        assert_eq!(state.stage_index, 0);

        let state3 = ctrl.state_at(50_000);
        assert_eq!(state3.render_resolution, 512);
        assert_eq!(state3.num_views, 4);
        assert_eq!(state3.sh_degree, 3);

        Ok(())
    }

    // ── CurriculumController – state_at defaults ─────────────────────────

    #[test]
    fn test_controller_state_defaults() -> Result<(), CurriculumError> {
        // A schedule with only one dimension configured; defaults fill the rest.
        let schedule = CurriculumSchedule::new().add_stage(
            CurriculumStage::new("only_res", 0, usize::MAX)
                .with_value(&DifficultyDimension::RenderResolution, 128.0),
        );
        let config = CurriculumConfig::default();
        let ctrl = CurriculumController::new(schedule, config)?;

        let state = ctrl.state_at(0);
        assert_eq!(state.render_resolution, 128);
        // Defaults
        assert_eq!(state.num_views, 1);
        assert_eq!(state.ddim_steps, 20);
        assert_eq!(state.sh_degree, 3);
        assert!((state.guidance_scale - 7.5).abs() < 1e-9);
        assert_eq!(state.noise_power, 0.0);

        Ok(())
    }

    // ── CurriculumController – loss-based advance ─────────────────────────

    #[test]
    fn test_controller_record_loss_advances_stage() -> Result<(), CurriculumError> {
        let schedule = CurriculumSchedule::new()
            .add_stage(
                CurriculumStage::new("low_res", 0, 1000)
                    .with_value(&DifficultyDimension::RenderResolution, 64.0)
                    .with_advance_threshold(0.1),
            )
            .add_stage(
                CurriculumStage::new("high_res", 1000, usize::MAX)
                    .with_value(&DifficultyDimension::RenderResolution, 256.0),
            );
        let config = CurriculumConfig {
            enable_loss_advance: true,
            no_regression: true,
        };
        let mut ctrl = CurriculumController::new(schedule, config)?;

        // Force window fill with losses below threshold.
        let mut advanced = false;
        for i in 0..200_usize {
            advanced = ctrl.record_loss(i, 0.05);
        }
        // After enough losses below threshold, the stage should have advanced.
        assert!(
            advanced || ctrl.current_stage_idx() == 1,
            "stage should have advanced; idx={}",
            ctrl.current_stage_idx()
        );

        Ok(())
    }

    // ── CurriculumController – force advance ──────────────────────────────

    #[test]
    fn test_controller_force_advance() -> Result<(), CurriculumError> {
        let schedule = CurriculumSchedule::default_gaf();
        let config = CurriculumConfig::default();
        let mut ctrl = CurriculumController::new(schedule, config)?;

        assert_eq!(ctrl.current_stage_idx(), 0);
        let ok = ctrl.advance_stage();
        assert!(ok);
        assert_eq!(ctrl.current_stage_idx(), 1);

        ctrl.advance_stage();
        ctrl.advance_stage();
        assert_eq!(ctrl.current_stage_idx(), 3);

        // Cannot advance past the last stage.
        let ok = ctrl.advance_stage();
        assert!(!ok);
        assert_eq!(ctrl.current_stage_idx(), 3);

        Ok(())
    }

    // ── ProgressTracker ───────────────────────────────────────────────────

    #[test]
    fn test_progress_tracker_completion_fraction() {
        let mut tracker = ProgressTracker::new();
        tracker.advance(500);
        let frac = tracker.completion_fraction(1000);
        assert!((frac - 0.5).abs() < 1e-6, "frac={frac}");

        let frac_full = tracker.completion_fraction(500);
        assert!((frac_full - 1.0).abs() < 1e-6);

        // Zero total steps → 0.0.
        assert_eq!(tracker.completion_fraction(0), 0.0);
        // usize::MAX → 0.0.
        assert_eq!(tracker.completion_fraction(usize::MAX), 0.0);
    }

    // ── CurriculumController – format_state ──────────────────────────────

    #[test]
    fn test_controller_format_state() -> Result<(), CurriculumError> {
        let schedule = CurriculumSchedule::default_gaf();
        let config = CurriculumConfig::default();
        let ctrl = CurriculumController::new(schedule, config)?;

        let s = ctrl.format_state(1000);
        assert!(
            s.contains("stage_0") || s.contains("stage_1"),
            "unexpected output: {s}"
        );
        assert!(s.contains("step="));
        assert!(s.contains("res="));
        Ok(())
    }
}
