//! Layer freezing utilities for Gaussian avatar models.
//!
//! Provides parameter freezing/unfreezing utilities for Gaussian models,
//! enabling progressive fine-tuning strategies that prevent catastrophic
//! forgetting during identity or expression adaptation.
//!
//! # Parameter Groups
//! Gaussian models have six standard parameter groups:
//! - `Positions` — 3D Gaussian centers (XYZ)
//! - `Rotations` — unit quaternions for orientation
//! - `Scales` — per-axis log-scale factors
//! - `Opacities` — per-Gaussian opacity (logit-space)
//! - `ShDc` — DC term of spherical harmonics (color)
//! - `ShRest` — higher-order SH coefficients
//!
//! # Usage Example
//! ```rust,ignore
//! use oxigaf_trainer::layer_freezing::{FreezeConfig, ParameterFreezer, ProgressiveUnfreezeSchedule, GaussianParamGroup};
//!
//! // Freeze everything, unfreeze appearance first
//! let config = FreezeConfig::all_frozen();
//! let schedule = ProgressiveUnfreezeSchedule::uniform(
//!     GaussianParamGroup::all_standard(),
//!     10_000,
//!     500,
//! ).unwrap();
//! let mut freezer = ParameterFreezer::with_schedule(config, schedule);
//! for _step in 0..10_000 {
//!     freezer.step();
//!     // freezer.apply_to_gradients(&mut gradient_groups);
//! }
//! ```

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by layer freezing operations.
#[derive(Debug, Error)]
pub enum FreezingError {
    #[error("Gradient length {gradient_len} does not match frozen mask length {mask_len}")]
    LengthMismatch {
        gradient_len: usize,
        mask_len: usize,
    },

    #[error("Group index {idx} out of range (have {n_groups} groups)")]
    GroupIndexOutOfRange { idx: usize, n_groups: usize },

    #[error("Invalid unfreeze fraction {frac}: must be in [0, 1]")]
    InvalidFraction { frac: f32 },

    #[error("No groups to unfreeze")]
    NoGroupsToUnfreeze,

    #[error("Schedule step {step} out of range [0, {total_steps}]")]
    StepOutOfRange { step: usize, total_steps: usize },
}

// ---------------------------------------------------------------------------
// GaussianParamGroup
// ---------------------------------------------------------------------------

/// Names of the standard Gaussian parameter groups.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum GaussianParamGroup {
    /// 3D Gaussian center positions (XYZ).
    Positions,
    /// Unit quaternion orientations.
    Rotations,
    /// Per-axis log-scale factors.
    Scales,
    /// Per-Gaussian opacity (logit-space).
    Opacities,
    /// DC term of spherical harmonics (base color).
    ShDc,
    /// Higher-order SH coefficients (view-dependent color).
    ShRest,
    /// User-defined custom parameter group.
    Custom(String),
}

impl GaussianParamGroup {
    /// Returns all six standard parameter groups in canonical order.
    ///
    /// Order: `[Positions, Rotations, Scales, Opacities, ShDc, ShRest]`
    pub fn all_standard() -> Vec<Self> {
        vec![
            Self::Positions,
            Self::Rotations,
            Self::Scales,
            Self::Opacities,
            Self::ShDc,
            Self::ShRest,
        ]
    }

    /// Returns the appearance-related parameter groups.
    ///
    /// Order: `[ShDc, ShRest, Opacities]`
    pub fn appearance() -> Vec<Self> {
        vec![Self::ShDc, Self::ShRest, Self::Opacities]
    }

    /// Returns the geometry/shape-related parameter groups.
    ///
    /// Order: `[Positions, Rotations, Scales]`
    pub fn geometry() -> Vec<Self> {
        vec![Self::Positions, Self::Rotations, Self::Scales]
    }

    /// Returns a human-readable name for the parameter group.
    pub fn name(&self) -> &str {
        match self {
            Self::Positions => "positions",
            Self::Rotations => "rotations",
            Self::Scales => "scales",
            Self::Opacities => "opacities",
            Self::ShDc => "sh_dc",
            Self::ShRest => "sh_rest",
            Self::Custom(s) => s.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// FreezeConfig
// ---------------------------------------------------------------------------

/// Configuration specifying which parameter groups are currently frozen.
///
/// A group with `frozen = true` will have its gradients zeroed out during
/// the backward pass, preventing those parameters from being updated.
#[derive(Debug, Clone)]
pub struct FreezeConfig {
    /// Per-group freeze state. `true` = frozen (gradient zeroed).
    pub frozen: HashMap<GaussianParamGroup, bool>,
}

impl FreezeConfig {
    /// Create a config where all standard groups are frozen.
    pub fn all_frozen() -> Self {
        let mut frozen = HashMap::new();
        for group in GaussianParamGroup::all_standard() {
            frozen.insert(group, true);
        }
        Self { frozen }
    }

    /// Create a config where no groups are frozen (all trainable).
    pub fn none_frozen() -> Self {
        let mut frozen = HashMap::new();
        for group in GaussianParamGroup::all_standard() {
            frozen.insert(group, false);
        }
        Self { frozen }
    }

    /// Create a config where only appearance groups (ShDc, ShRest, Opacities) are frozen.
    pub fn appearance_frozen() -> Self {
        let mut frozen = HashMap::new();
        for group in GaussianParamGroup::all_standard() {
            let is_frozen = GaussianParamGroup::appearance().contains(&group);
            frozen.insert(group, is_frozen);
        }
        Self { frozen }
    }

    /// Create a config where only geometry groups (Positions, Rotations, Scales) are frozen.
    pub fn geometry_frozen() -> Self {
        let mut frozen = HashMap::new();
        for group in GaussianParamGroup::all_standard() {
            let is_frozen = GaussianParamGroup::geometry().contains(&group);
            frozen.insert(group, is_frozen);
        }
        Self { frozen }
    }

    /// Check if a specific group is currently frozen.
    ///
    /// Returns `false` for groups not in the config (unknown groups are trainable by default).
    pub fn is_frozen(&self, group: &GaussianParamGroup) -> bool {
        self.frozen.get(group).copied().unwrap_or(false)
    }

    /// Set the freeze state for a specific group.
    pub fn set_frozen(&mut self, group: GaussianParamGroup, frozen: bool) {
        self.frozen.insert(group, frozen);
    }

    /// Returns the number of currently frozen groups.
    pub fn frozen_count(&self) -> usize {
        self.frozen.values().filter(|&&f| f).count()
    }

    /// Returns the number of currently unfrozen (trainable) groups.
    pub fn unfrozen_count(&self) -> usize {
        self.frozen.values().filter(|&&f| !f).count()
    }

    /// Returns references to all currently frozen groups.
    pub fn frozen_groups(&self) -> Vec<&GaussianParamGroup> {
        self.frozen
            .iter()
            .filter_map(|(g, &f)| if f { Some(g) } else { None })
            .collect()
    }

    /// Returns references to all currently unfrozen (trainable) groups.
    pub fn unfrozen_groups(&self) -> Vec<&GaussianParamGroup> {
        self.frozen
            .iter()
            .filter_map(|(g, &f)| if !f { Some(g) } else { None })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// FrozenMask
// ---------------------------------------------------------------------------

/// Per-parameter boolean mask for gradient zeroing.
///
/// `mask[i] = true` means parameter `i` is frozen (its gradient will be
/// zeroed before the optimizer update step).
#[derive(Debug)]
pub struct FrozenMask {
    /// Per-element freeze state. `true` = frozen.
    pub mask: Vec<bool>,
}

impl FrozenMask {
    /// Create a mask of length `n` where all parameters are frozen.
    pub fn new_all_frozen(n: usize) -> Self {
        Self {
            mask: vec![true; n],
        }
    }

    /// Create a mask of length `n` where no parameters are frozen.
    pub fn new_none_frozen(n: usize) -> Self {
        Self {
            mask: vec![false; n],
        }
    }

    /// Returns the total number of parameters tracked by this mask.
    pub fn len(&self) -> usize {
        self.mask.len()
    }

    /// Returns `true` if the mask covers zero parameters.
    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    /// Returns the number of currently frozen parameters.
    pub fn frozen_count(&self) -> usize {
        self.mask.iter().filter(|&&f| f).count()
    }

    /// Returns the number of currently unfrozen parameters.
    pub fn unfrozen_count(&self) -> usize {
        self.mask.iter().filter(|&&f| !f).count()
    }

    /// Freeze parameters in the inclusive range `[start, end]`.
    ///
    /// Indices outside `[0, len())` are silently clamped/skipped.
    pub fn freeze_range(&mut self, start: usize, end: usize) {
        let end_clamped = end.min(self.mask.len().saturating_sub(1));
        if start > end_clamped {
            return;
        }
        for i in start..=end_clamped {
            self.mask[i] = true;
        }
    }

    /// Unfreeze parameters in the inclusive range `[start, end]`.
    ///
    /// Indices outside `[0, len())` are silently clamped/skipped.
    pub fn unfreeze_range(&mut self, start: usize, end: usize) {
        let end_clamped = end.min(self.mask.len().saturating_sub(1));
        if start > end_clamped {
            return;
        }
        for i in start..=end_clamped {
            self.mask[i] = false;
        }
    }

    /// Combine two masks: `result[i] = self[i] OR other[i]`.
    ///
    /// A parameter is frozen if it is frozen in *either* mask.
    /// Both masks must have the same length.
    pub fn union(&self, other: &FrozenMask) -> Result<FrozenMask, FreezingError> {
        if self.mask.len() != other.mask.len() {
            return Err(FreezingError::LengthMismatch {
                gradient_len: self.mask.len(),
                mask_len: other.mask.len(),
            });
        }
        let mask = self
            .mask
            .iter()
            .zip(other.mask.iter())
            .map(|(&a, &b)| a || b)
            .collect();
        Ok(FrozenMask { mask })
    }

    /// Combine two masks: `result[i] = self[i] AND other[i]`.
    ///
    /// A parameter is frozen only if it is frozen in *both* masks.
    /// Both masks must have the same length.
    pub fn intersection(&self, other: &FrozenMask) -> Result<FrozenMask, FreezingError> {
        if self.mask.len() != other.mask.len() {
            return Err(FreezingError::LengthMismatch {
                gradient_len: self.mask.len(),
                mask_len: other.mask.len(),
            });
        }
        let mask = self
            .mask
            .iter()
            .zip(other.mask.iter())
            .map(|(&a, &b)| a && b)
            .collect();
        Ok(FrozenMask { mask })
    }
}

// ---------------------------------------------------------------------------
// ProgressiveUnfreezeSchedule
// ---------------------------------------------------------------------------

/// Schedule for progressively unfreezing parameter groups during training.
///
/// Groups are unfrozen one at a time in a specified order, starting after
/// an optional warmup period. This prevents catastrophic forgetting by
/// keeping sensitive parameters frozen while the model adapts.
#[derive(Debug)]
pub struct ProgressiveUnfreezeSchedule {
    /// Order in which groups are unfrozen (index 0 = first to unfreeze).
    pub unfreeze_order: Vec<GaussianParamGroup>,
    /// Training step at which each group becomes unfrozen.
    pub unfreeze_steps: Vec<usize>,
    /// Total number of training steps in the schedule.
    pub total_steps: usize,
}

impl ProgressiveUnfreezeSchedule {
    /// Create a uniform schedule: unfreeze one group every `(total_steps - warmup_steps) / n_groups` steps.
    ///
    /// Group `i` unfreezes at step `warmup_steps + i * step_interval`.
    ///
    /// # Errors
    /// - `NoGroupsToUnfreeze` if `groups` is empty.
    pub fn uniform(
        groups: Vec<GaussianParamGroup>,
        total_steps: usize,
        warmup_steps: usize,
    ) -> Result<Self, FreezingError> {
        if groups.is_empty() {
            return Err(FreezingError::NoGroupsToUnfreeze);
        }

        let n_groups = groups.len();
        let available_steps = total_steps.saturating_sub(warmup_steps);
        let step_interval = available_steps.checked_div(n_groups).unwrap_or(0);

        let unfreeze_steps = (0..n_groups)
            .map(|i| warmup_steps + i * step_interval)
            .collect();

        Ok(Self {
            unfreeze_order: groups,
            unfreeze_steps,
            total_steps,
        })
    }

    /// Returns the groups that should be unfrozen (trainable) at the given step.
    ///
    /// A group is unfrozen if the current step is at or beyond its scheduled unfreeze step.
    pub fn unfrozen_at_step(&self, step: usize) -> Vec<&GaussianParamGroup> {
        self.unfreeze_order
            .iter()
            .zip(self.unfreeze_steps.iter())
            .filter_map(|(group, &unfreeze_step)| {
                if step >= unfreeze_step {
                    Some(group)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Compute the `FreezeConfig` that should apply at the given training step.
    ///
    /// All groups start as frozen; groups are unfrozen as `step` advances past
    /// their scheduled `unfreeze_steps` entries.
    pub fn freeze_config_at_step(&self, step: usize) -> FreezeConfig {
        // Start with all groups in the schedule frozen, plus any standard groups
        let mut frozen: HashMap<GaussianParamGroup, bool> = GaussianParamGroup::all_standard()
            .into_iter()
            .map(|g| (g, true))
            .collect();

        // Also add all groups from the schedule (may include Custom variants)
        for group in &self.unfreeze_order {
            frozen.entry(group.clone()).or_insert(true);
        }

        // Unfreeze groups whose step has been reached
        for (group, &unfreeze_step) in self.unfreeze_order.iter().zip(self.unfreeze_steps.iter()) {
            if step >= unfreeze_step {
                frozen.insert(group.clone(), false);
            }
        }

        FreezeConfig { frozen }
    }

    /// Returns the group that is newly unfrozen at this exact step, if any.
    ///
    /// Returns `Some(group)` if a group's scheduled unfreeze step equals `step`,
    /// otherwise `None`.
    pub fn newly_unfrozen_at_step(&self, step: usize) -> Option<&GaussianParamGroup> {
        self.unfreeze_order
            .iter()
            .zip(self.unfreeze_steps.iter())
            .find_map(|(group, &unfreeze_step)| {
                if unfreeze_step == step {
                    Some(group)
                } else {
                    None
                }
            })
    }

    /// Format the schedule as a human-readable table.
    ///
    /// Shows each group and the step at which it becomes trainable.
    pub fn format_schedule(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Progressive Unfreeze Schedule (total_steps={})",
            self.total_steps
        ));
        lines.push(format!("{:<20} {}", "Group", "Unfreeze Step"));
        lines.push("-".repeat(35));
        for (group, &step) in self.unfreeze_order.iter().zip(self.unfreeze_steps.iter()) {
            lines.push(format!("{:<20} {}", group.name(), step));
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// ParameterFreezer
// ---------------------------------------------------------------------------

/// Stateful manager for parameter freezing during training.
///
/// Tracks the current training step and applies a `ProgressiveUnfreezeSchedule`
/// if one is provided, automatically updating the freeze state as training progresses.
pub struct ParameterFreezer {
    /// Current freeze configuration.
    pub config: FreezeConfig,
    /// Optional progressive unfreeze schedule.
    pub schedule: Option<ProgressiveUnfreezeSchedule>,
    /// Current training step counter.
    pub step: usize,
}

impl ParameterFreezer {
    /// Create a new freezer with the given config and no schedule.
    pub fn new(config: FreezeConfig) -> Self {
        Self {
            config,
            schedule: None,
            step: 0,
        }
    }

    /// Create a new freezer with a config and a progressive unfreeze schedule.
    pub fn with_schedule(config: FreezeConfig, schedule: ProgressiveUnfreezeSchedule) -> Self {
        Self {
            config,
            schedule: Some(schedule),
            step: 0,
        }
    }

    /// Advance one training step.
    ///
    /// If a schedule is present, the freeze config is updated to reflect
    /// the state at the current step before incrementing the counter.
    pub fn step(&mut self) {
        if let Some(ref schedule) = self.schedule {
            self.config = schedule.freeze_config_at_step(self.step);
        }
        self.step += 1;
    }

    /// Check if the specified parameter group is currently frozen.
    pub fn is_frozen(&self, group: &GaussianParamGroup) -> bool {
        self.config.is_frozen(group)
    }

    /// Force-freeze a parameter group, overriding the schedule.
    pub fn freeze(&mut self, group: GaussianParamGroup) {
        self.config.set_frozen(group, true);
    }

    /// Force-unfreeze a parameter group, overriding the schedule.
    pub fn unfreeze(&mut self, group: GaussianParamGroup) {
        self.config.set_frozen(group, false);
    }

    /// Zero out gradients for all frozen groups.
    ///
    /// Returns the total number of gradient values that were zeroed.
    pub fn apply_to_gradients(
        &self,
        gradients_by_group: &mut [(GaussianParamGroup, Vec<f32>)],
    ) -> usize {
        let mut total_zeroed = 0usize;
        for (group, gradients) in gradients_by_group.iter_mut() {
            if self.config.is_frozen(group) {
                for g in gradients.iter_mut() {
                    *g = 0.0;
                }
                total_zeroed += gradients.len();
            }
        }
        total_zeroed
    }

    /// Format the current freeze status as a human-readable string.
    ///
    /// Example: `"Frozen: positions, rotations | Unfrozen: scales, opacities, sh_dc, sh_rest"`
    pub fn format_status(&self) -> String {
        let mut frozen_names: Vec<&str> = self
            .config
            .frozen_groups()
            .into_iter()
            .map(|g| g.name())
            .collect();
        frozen_names.sort_unstable();

        let mut unfrozen_names: Vec<&str> = self
            .config
            .unfrozen_groups()
            .into_iter()
            .map(|g| g.name())
            .collect();
        unfrozen_names.sort_unstable();

        format!(
            "Frozen: {} | Unfrozen: {}",
            if frozen_names.is_empty() {
                "(none)".to_string()
            } else {
                frozen_names.join(", ")
            },
            if unfrozen_names.is_empty() {
                "(none)".to_string()
            } else {
                unfrozen_names.join(", ")
            }
        )
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Zero out gradients at positions marked as frozen in the mask.
///
/// Returns the number of gradient elements that were zeroed.
///
/// # Errors
/// Returns `LengthMismatch` if `gradients.len() != mask.len()`.
pub fn apply_frozen_mask(gradients: &mut [f32], mask: &FrozenMask) -> Result<usize, FreezingError> {
    if gradients.len() != mask.len() {
        return Err(FreezingError::LengthMismatch {
            gradient_len: gradients.len(),
            mask_len: mask.len(),
        });
    }
    let mut zeroed = 0usize;
    for (g, &frozen) in gradients.iter_mut().zip(mask.mask.iter()) {
        if frozen {
            *g = 0.0;
            zeroed += 1;
        }
    }
    Ok(zeroed)
}

/// Create a mask where the first `n_frozen` elements are frozen and the rest are trainable.
pub fn front_frozen_mask(total: usize, n_frozen: usize) -> FrozenMask {
    let n_frozen = n_frozen.min(total);
    let mask = (0..total).map(|i| i < n_frozen).collect();
    FrozenMask { mask }
}

/// Create a mask where the last `n_frozen` elements are frozen and the rest are trainable.
pub fn back_frozen_mask(total: usize, n_frozen: usize) -> FrozenMask {
    let n_frozen = n_frozen.min(total);
    let freeze_start = total.saturating_sub(n_frozen);
    let mask = (0..total).map(|i| i >= freeze_start).collect();
    FrozenMask { mask }
}

/// Create a progressive mask based on the current training step.
///
/// At step 0, all parameters are frozen.
/// At step `total_steps`, all parameters are unfrozen.
/// Between those extremes, the frozen fraction decreases linearly from the front.
///
/// Frozen fraction: `1.0 - (step / total_steps)`, applied as `floor(total * fraction)`.
/// Front elements are frozen first (i.e., unfrozen last).
///
/// If `total_steps == 0`, returns an all-unfrozen mask.
pub fn progressive_mask(total: usize, step: usize, total_steps: usize) -> FrozenMask {
    if total_steps == 0 {
        return FrozenMask::new_none_frozen(total);
    }
    let step_clamped = step.min(total_steps);
    let frozen_fraction = 1.0 - (step_clamped as f32 / total_steps as f32);
    let n_frozen = (total as f32 * frozen_fraction).floor() as usize;
    front_frozen_mask(total, n_frozen)
}

/// Compute the fraction of parameters that should be unfrozen at a given step.
///
/// - Returns `0.0` before `warmup_steps`.
/// - Returns `1.0` at or after `total_steps`.
/// - Linearly interpolates between `warmup_steps` and `total_steps`.
///
/// If `total_steps == warmup_steps`, returns `1.0` (all unfrozen immediately after warmup).
pub fn unfreeze_fraction(step: usize, warmup_steps: usize, total_steps: usize) -> f32 {
    if step < warmup_steps {
        return 0.0;
    }
    let available = total_steps.saturating_sub(warmup_steps);
    if available == 0 {
        return 1.0;
    }
    let elapsed = step.saturating_sub(warmup_steps);
    (elapsed as f32 / available as f32).min(1.0)
}

/// Sanity-check that all frozen positions in the gradient have approximately zero value.
///
/// Returns `true` if all frozen positions have `|gradient| < 1e-8`, `false` otherwise.
///
/// # Errors
/// Returns `LengthMismatch` if `gradients.len() != mask.len()`.
pub fn check_frozen_gradients_zeroed(
    gradients: &[f32],
    mask: &FrozenMask,
) -> Result<bool, FreezingError> {
    if gradients.len() != mask.len() {
        return Err(FreezingError::LengthMismatch {
            gradient_len: gradients.len(),
            mask_len: mask.len(),
        });
    }
    let all_zero = gradients
        .iter()
        .zip(mask.mask.iter())
        .filter(|(_, &frozen)| frozen)
        .all(|(&g, _)| g.abs() < 1e-8);
    Ok(all_zero)
}

/// Estimate the fraction of gradient computation saved by freezing.
///
/// Returns `frozen_count / len`. Returns `0.0` for an empty mask.
pub fn frozen_compute_savings(mask: &FrozenMask) -> f32 {
    if mask.is_empty() {
        return 0.0;
    }
    mask.frozen_count() as f32 / mask.len() as f32
}

/// Split a flat Gaussian model into labeled parameter groups.
///
/// Returns `(GaussianParamGroup, size)` pairs for all non-empty arrays,
/// preserving canonical group ordering.
pub fn split_into_groups(
    positions: &[f32],
    rotations: &[f32],
    scales: &[f32],
    opacities: &[f32],
    sh_dc: &[f32],
    sh_rest: &[f32],
) -> Vec<(GaussianParamGroup, usize)> {
    let candidates = [
        (GaussianParamGroup::Positions, positions.len()),
        (GaussianParamGroup::Rotations, rotations.len()),
        (GaussianParamGroup::Scales, scales.len()),
        (GaussianParamGroup::Opacities, opacities.len()),
        (GaussianParamGroup::ShDc, sh_dc.len()),
        (GaussianParamGroup::ShRest, sh_rest.len()),
    ];
    candidates.into_iter().filter(|(_, len)| *len > 0).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // GaussianParamGroup tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_standard_has_six_groups() {
        let groups = GaussianParamGroup::all_standard();
        assert_eq!(groups.len(), 6);
        assert!(groups.contains(&GaussianParamGroup::Positions));
        assert!(groups.contains(&GaussianParamGroup::Rotations));
        assert!(groups.contains(&GaussianParamGroup::Scales));
        assert!(groups.contains(&GaussianParamGroup::Opacities));
        assert!(groups.contains(&GaussianParamGroup::ShDc));
        assert!(groups.contains(&GaussianParamGroup::ShRest));
    }

    #[test]
    fn test_appearance_groups() {
        let groups = GaussianParamGroup::appearance();
        assert_eq!(groups.len(), 3);
        assert!(groups.contains(&GaussianParamGroup::ShDc));
        assert!(groups.contains(&GaussianParamGroup::ShRest));
        assert!(groups.contains(&GaussianParamGroup::Opacities));
        assert!(!groups.contains(&GaussianParamGroup::Positions));
    }

    #[test]
    fn test_geometry_groups() {
        let groups = GaussianParamGroup::geometry();
        assert_eq!(groups.len(), 3);
        assert!(groups.contains(&GaussianParamGroup::Positions));
        assert!(groups.contains(&GaussianParamGroup::Rotations));
        assert!(groups.contains(&GaussianParamGroup::Scales));
        assert!(!groups.contains(&GaussianParamGroup::ShDc));
    }

    #[test]
    fn test_group_names() {
        assert_eq!(GaussianParamGroup::Positions.name(), "positions");
        assert_eq!(GaussianParamGroup::Rotations.name(), "rotations");
        assert_eq!(GaussianParamGroup::Scales.name(), "scales");
        assert_eq!(GaussianParamGroup::Opacities.name(), "opacities");
        assert_eq!(GaussianParamGroup::ShDc.name(), "sh_dc");
        assert_eq!(GaussianParamGroup::ShRest.name(), "sh_rest");
        assert_eq!(GaussianParamGroup::Custom("foo".to_string()).name(), "foo");
    }

    // -----------------------------------------------------------------------
    // FreezeConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_frozen_config() {
        let config = FreezeConfig::all_frozen();
        assert_eq!(config.frozen_count(), 6);
        assert_eq!(config.unfrozen_count(), 0);
        assert!(config.is_frozen(&GaussianParamGroup::Positions));
        assert!(config.is_frozen(&GaussianParamGroup::ShRest));
    }

    #[test]
    fn test_none_frozen_config() {
        let config = FreezeConfig::none_frozen();
        assert_eq!(config.frozen_count(), 0);
        assert_eq!(config.unfrozen_count(), 6);
        assert!(!config.is_frozen(&GaussianParamGroup::Positions));
    }

    #[test]
    fn test_appearance_frozen_config() {
        let config = FreezeConfig::appearance_frozen();
        assert!(config.is_frozen(&GaussianParamGroup::ShDc));
        assert!(config.is_frozen(&GaussianParamGroup::ShRest));
        assert!(config.is_frozen(&GaussianParamGroup::Opacities));
        assert!(!config.is_frozen(&GaussianParamGroup::Positions));
        assert!(!config.is_frozen(&GaussianParamGroup::Rotations));
        assert!(!config.is_frozen(&GaussianParamGroup::Scales));
        assert_eq!(config.frozen_count(), 3);
    }

    #[test]
    fn test_geometry_frozen_config() {
        let config = FreezeConfig::geometry_frozen();
        assert!(config.is_frozen(&GaussianParamGroup::Positions));
        assert!(config.is_frozen(&GaussianParamGroup::Rotations));
        assert!(config.is_frozen(&GaussianParamGroup::Scales));
        assert!(!config.is_frozen(&GaussianParamGroup::ShDc));
        assert!(!config.is_frozen(&GaussianParamGroup::ShRest));
        assert!(!config.is_frozen(&GaussianParamGroup::Opacities));
        assert_eq!(config.frozen_count(), 3);
    }

    #[test]
    fn test_set_frozen() {
        let mut config = FreezeConfig::none_frozen();
        assert!(!config.is_frozen(&GaussianParamGroup::Positions));
        config.set_frozen(GaussianParamGroup::Positions, true);
        assert!(config.is_frozen(&GaussianParamGroup::Positions));
        config.set_frozen(GaussianParamGroup::Positions, false);
        assert!(!config.is_frozen(&GaussianParamGroup::Positions));
    }

    #[test]
    fn test_frozen_groups_list() {
        let config = FreezeConfig::geometry_frozen();
        let frozen = config.frozen_groups();
        assert_eq!(frozen.len(), 3);
        let unfrozen = config.unfrozen_groups();
        assert_eq!(unfrozen.len(), 3);
    }

    #[test]
    fn test_is_frozen_unknown_group_returns_false() {
        let config = FreezeConfig::all_frozen();
        // Custom group not in map → defaults to false
        assert!(!config.is_frozen(&GaussianParamGroup::Custom("xyz".to_string())));
    }

    #[test]
    fn test_frozen_count_after_mutations() {
        let mut config = FreezeConfig::none_frozen();
        assert_eq!(config.frozen_count(), 0);
        config.set_frozen(GaussianParamGroup::Positions, true);
        config.set_frozen(GaussianParamGroup::Scales, true);
        assert_eq!(config.frozen_count(), 2);
        assert_eq!(config.unfrozen_count(), 4);
    }

    // -----------------------------------------------------------------------
    // FrozenMask tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mask_new_all_frozen() {
        let mask = FrozenMask::new_all_frozen(5);
        assert_eq!(mask.len(), 5);
        assert_eq!(mask.frozen_count(), 5);
        assert_eq!(mask.unfrozen_count(), 0);
        assert!(mask.mask.iter().all(|&f| f));
    }

    #[test]
    fn test_mask_new_none_frozen() {
        let mask = FrozenMask::new_none_frozen(5);
        assert_eq!(mask.len(), 5);
        assert_eq!(mask.frozen_count(), 0);
        assert_eq!(mask.unfrozen_count(), 5);
        assert!(mask.mask.iter().all(|&f| !f));
    }

    #[test]
    fn test_mask_freeze_range() {
        let mut mask = FrozenMask::new_none_frozen(10);
        mask.freeze_range(2, 5);
        for i in 0..10 {
            assert_eq!(mask.mask[i], (2..=5).contains(&i), "index {}", i);
        }
        assert_eq!(mask.frozen_count(), 4);
    }

    #[test]
    fn test_mask_unfreeze_range() {
        let mut mask = FrozenMask::new_all_frozen(10);
        mask.unfreeze_range(3, 7);
        for i in 0..10 {
            assert_eq!(mask.mask[i], !(3..=7).contains(&i), "index {}", i);
        }
        assert_eq!(mask.unfrozen_count(), 5);
    }

    #[test]
    fn test_mask_union() {
        let a = front_frozen_mask(6, 3); // [T,T,T,F,F,F]
        let b = back_frozen_mask(6, 3); // [F,F,F,T,T,T]
        let result = a.union(&b).expect("union should succeed");
        assert!(result.mask.iter().all(|&f| f)); // all frozen
    }

    #[test]
    fn test_mask_intersection() {
        let a = front_frozen_mask(6, 4); // [T,T,T,T,F,F]
        let b = front_frozen_mask(6, 2); // [T,T,F,F,F,F]
        let result = a.intersection(&b).expect("intersection should succeed");
        // Only first 2 should be frozen
        assert_eq!(result.frozen_count(), 2);
        assert!(result.mask[0]);
        assert!(result.mask[1]);
        assert!(!result.mask[2]);
    }

    #[test]
    fn test_mask_union_length_mismatch() {
        let a = FrozenMask::new_all_frozen(3);
        let b = FrozenMask::new_all_frozen(5);
        let result = a.union(&b);
        assert!(matches!(result, Err(FreezingError::LengthMismatch { .. })));
    }

    #[test]
    fn test_mask_intersection_length_mismatch() {
        let a = FrozenMask::new_all_frozen(3);
        let b = FrozenMask::new_all_frozen(4);
        let result = a.intersection(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_mask_is_empty() {
        let empty = FrozenMask::new_all_frozen(0);
        assert!(empty.is_empty());
        let nonempty = FrozenMask::new_all_frozen(1);
        assert!(!nonempty.is_empty());
    }

    // -----------------------------------------------------------------------
    // apply_frozen_mask tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_frozen_mask_zeroes_frozen() {
        let mask = front_frozen_mask(6, 3);
        let mut grads = vec![1.0f32; 6];
        let zeroed = apply_frozen_mask(&mut grads, &mask).expect("should succeed");
        assert_eq!(zeroed, 3);
        assert_eq!(&grads[..3], &[0.0, 0.0, 0.0]);
        assert_eq!(&grads[3..], &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_apply_frozen_mask_leaves_unfrozen_unchanged() {
        let mask = FrozenMask::new_none_frozen(4);
        let mut grads = vec![2.5f32; 4];
        let zeroed = apply_frozen_mask(&mut grads, &mask).expect("should succeed");
        assert_eq!(zeroed, 0);
        assert!(grads.iter().all(|&g| g == 2.5));
    }

    #[test]
    fn test_apply_frozen_mask_length_mismatch() {
        let mask = FrozenMask::new_all_frozen(3);
        let mut grads = vec![1.0f32; 5];
        let result = apply_frozen_mask(&mut grads, &mask);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_frozen_mask_all_frozen() {
        let mask = FrozenMask::new_all_frozen(4);
        let mut grads = vec![3.1f32; 4];
        let zeroed = apply_frozen_mask(&mut grads, &mask).expect("should succeed");
        assert_eq!(zeroed, 4);
        assert!(grads.iter().all(|&g| g == 0.0));
    }

    // -----------------------------------------------------------------------
    // front_frozen_mask / back_frozen_mask tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_front_frozen_mask_correct_split() {
        let mask = front_frozen_mask(10, 4);
        assert_eq!(mask.frozen_count(), 4);
        for i in 0..4 {
            assert!(mask.mask[i], "index {} should be frozen", i);
        }
        for i in 4..10 {
            assert!(!mask.mask[i], "index {} should be unfrozen", i);
        }
    }

    #[test]
    fn test_back_frozen_mask_correct_split() {
        let mask = back_frozen_mask(10, 4);
        assert_eq!(mask.frozen_count(), 4);
        for i in 0..6 {
            assert!(!mask.mask[i], "index {} should be unfrozen", i);
        }
        for i in 6..10 {
            assert!(mask.mask[i], "index {} should be frozen", i);
        }
    }

    #[test]
    fn test_front_frozen_mask_clamps_n_frozen() {
        // n_frozen > total should clamp to total
        let mask = front_frozen_mask(5, 100);
        assert_eq!(mask.frozen_count(), 5);
    }

    #[test]
    fn test_back_frozen_mask_clamps_n_frozen() {
        let mask = back_frozen_mask(5, 100);
        assert_eq!(mask.frozen_count(), 5);
    }

    // -----------------------------------------------------------------------
    // progressive_mask tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_progressive_mask_step_zero_fully_frozen() {
        let mask = progressive_mask(100, 0, 100);
        assert_eq!(mask.frozen_count(), 100);
    }

    #[test]
    fn test_progressive_mask_step_total_fully_unfrozen() {
        let mask = progressive_mask(100, 100, 100);
        assert_eq!(mask.frozen_count(), 0);
    }

    #[test]
    fn test_progressive_mask_middle() {
        // At step 50/100, frozen_fraction = 0.5 → 50 frozen
        let mask = progressive_mask(100, 50, 100);
        assert_eq!(mask.frozen_count(), 50);
        // Front elements frozen
        assert!(mask.mask[0]);
        assert!(!mask.mask[99]);
    }

    #[test]
    fn test_progressive_mask_zero_total_steps() {
        // Guard: total_steps=0 → all unfrozen
        let mask = progressive_mask(10, 0, 0);
        assert_eq!(mask.frozen_count(), 0);
    }

    // -----------------------------------------------------------------------
    // unfreeze_fraction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_unfreeze_fraction_before_warmup() {
        assert_eq!(unfreeze_fraction(0, 500, 10_000), 0.0);
        assert_eq!(unfreeze_fraction(499, 500, 10_000), 0.0);
    }

    #[test]
    fn test_unfreeze_fraction_after_warmup() {
        let frac = unfreeze_fraction(5_250, 500, 10_000);
        // (5250 - 500) / (10000 - 500) = 4750/9500 = 0.5
        let expected = 4750.0_f32 / 9500.0;
        assert!((frac - expected).abs() < 1e-5, "got {}", frac);
    }

    #[test]
    fn test_unfreeze_fraction_at_total() {
        assert_eq!(unfreeze_fraction(10_000, 500, 10_000), 1.0);
    }

    #[test]
    fn test_unfreeze_fraction_warmup_equals_total() {
        // Edge case: no room after warmup → immediately 1.0
        assert_eq!(unfreeze_fraction(1000, 1000, 1000), 1.0);
        assert_eq!(unfreeze_fraction(1001, 1000, 1000), 1.0);
    }

    // -----------------------------------------------------------------------
    // ProgressiveUnfreezeSchedule tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_uniform_schedule_correct_steps() {
        let groups = GaussianParamGroup::all_standard();
        let schedule = ProgressiveUnfreezeSchedule::uniform(groups, 1200, 200)
            .expect("should create schedule");
        // available = 1000, interval = 1000/6 = 166
        // steps: [200, 366, 532, 698, 864, 1030]
        assert_eq!(schedule.unfreeze_steps.len(), 6);
        assert_eq!(schedule.unfreeze_steps[0], 200);
        let interval = 1000 / 6;
        for i in 0..6 {
            assert_eq!(schedule.unfreeze_steps[i], 200 + i * interval);
        }
    }

    #[test]
    fn test_uniform_schedule_unfrozen_at_step() {
        let groups = vec![
            GaussianParamGroup::Positions,
            GaussianParamGroup::Rotations,
            GaussianParamGroup::Scales,
        ];
        let schedule =
            ProgressiveUnfreezeSchedule::uniform(groups, 1000, 0).expect("schedule creation");
        // interval = 1000/3 = 333
        // steps: [0, 333, 666]
        let unfrozen_early = schedule.unfrozen_at_step(0);
        assert_eq!(unfrozen_early.len(), 1); // Positions at step 0

        let unfrozen_mid = schedule.unfrozen_at_step(333);
        assert_eq!(unfrozen_mid.len(), 2); // Positions + Rotations

        let unfrozen_late = schedule.unfrozen_at_step(999);
        assert_eq!(unfrozen_late.len(), 3); // All
    }

    #[test]
    fn test_uniform_schedule_newly_unfrozen() {
        let groups = vec![GaussianParamGroup::Positions, GaussianParamGroup::Rotations];
        let schedule = ProgressiveUnfreezeSchedule::uniform(groups, 1000, 0).expect("schedule");
        // interval = 1000/2 = 500
        // Positions at 0, Rotations at 500
        assert!(schedule.newly_unfrozen_at_step(0).is_some());
        let newly = schedule.newly_unfrozen_at_step(500);
        assert_eq!(newly, Some(&GaussianParamGroup::Rotations));
        assert!(schedule.newly_unfrozen_at_step(1).is_none());
    }

    #[test]
    fn test_uniform_schedule_empty_groups_error() {
        let result = ProgressiveUnfreezeSchedule::uniform(vec![], 1000, 100);
        assert!(matches!(result, Err(FreezingError::NoGroupsToUnfreeze)));
    }

    #[test]
    fn test_freeze_config_at_step() {
        let groups = vec![GaussianParamGroup::Positions, GaussianParamGroup::Rotations];
        let schedule = ProgressiveUnfreezeSchedule::uniform(groups, 1000, 0).expect("schedule");
        // interval = 500; Positions unfrozen at 0, Rotations at 500
        let config_early = schedule.freeze_config_at_step(0);
        // Positions unfrozen (step 0 >= 0), Rotations still frozen
        assert!(!config_early.is_frozen(&GaussianParamGroup::Positions));
        assert!(config_early.is_frozen(&GaussianParamGroup::Rotations));

        let config_late = schedule.freeze_config_at_step(500);
        assert!(!config_late.is_frozen(&GaussianParamGroup::Positions));
        assert!(!config_late.is_frozen(&GaussianParamGroup::Rotations));
    }

    // -----------------------------------------------------------------------
    // ParameterFreezer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parameter_freezer_new() {
        let config = FreezeConfig::all_frozen();
        let freezer = ParameterFreezer::new(config);
        assert_eq!(freezer.step, 0);
        assert!(freezer.is_frozen(&GaussianParamGroup::Positions));
    }

    #[test]
    fn test_parameter_freezer_step_advances() {
        let config = FreezeConfig::all_frozen();
        let mut freezer = ParameterFreezer::new(config);
        freezer.step();
        assert_eq!(freezer.step, 1);
        freezer.step();
        assert_eq!(freezer.step, 2);
    }

    #[test]
    fn test_parameter_freezer_freeze_unfreeze() {
        let config = FreezeConfig::none_frozen();
        let mut freezer = ParameterFreezer::new(config);
        assert!(!freezer.is_frozen(&GaussianParamGroup::Scales));
        freezer.freeze(GaussianParamGroup::Scales);
        assert!(freezer.is_frozen(&GaussianParamGroup::Scales));
        freezer.unfreeze(GaussianParamGroup::Scales);
        assert!(!freezer.is_frozen(&GaussianParamGroup::Scales));
    }

    #[test]
    fn test_parameter_freezer_with_schedule_advances() {
        let groups = vec![GaussianParamGroup::Positions, GaussianParamGroup::Rotations];
        let schedule = ProgressiveUnfreezeSchedule::uniform(groups, 1000, 0).expect("schedule");
        let config = FreezeConfig::all_frozen();
        let mut freezer = ParameterFreezer::with_schedule(config, schedule);

        // At step 0, positions should be unfrozen (unfreeze_steps[0] = 0)
        // step() applies config for current step (0), then increments to 1
        freezer.step();
        assert!(!freezer.is_frozen(&GaussianParamGroup::Positions));
        assert!(freezer.is_frozen(&GaussianParamGroup::Rotations));
    }

    #[test]
    fn test_parameter_freezer_format_status() {
        let config = FreezeConfig::geometry_frozen();
        let freezer = ParameterFreezer::new(config);
        let status = freezer.format_status();
        assert!(status.contains("Frozen:"));
        assert!(status.contains("Unfrozen:"));
        assert!(status.contains("positions"));
    }

    // -----------------------------------------------------------------------
    // apply_to_gradients tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_apply_to_gradients_frozen_group_zeroed() {
        let config = FreezeConfig::geometry_frozen(); // positions/rotations/scales frozen
        let freezer = ParameterFreezer::new(config);
        let mut grad_groups = vec![
            (GaussianParamGroup::Positions, vec![1.0f32; 10]),
            (GaussianParamGroup::ShDc, vec![2.0f32; 5]),
        ];
        let zeroed = freezer.apply_to_gradients(&mut grad_groups);
        assert_eq!(zeroed, 10);
        assert!(grad_groups[0].1.iter().all(|&g| g == 0.0));
        assert!(grad_groups[1].1.iter().all(|&g| g == 2.0));
    }

    #[test]
    fn test_apply_to_gradients_unfrozen_unchanged() {
        let config = FreezeConfig::none_frozen();
        let freezer = ParameterFreezer::new(config);
        let mut grad_groups = vec![(GaussianParamGroup::Positions, vec![5.0f32; 8])];
        let zeroed = freezer.apply_to_gradients(&mut grad_groups);
        assert_eq!(zeroed, 0);
        assert!(grad_groups[0].1.iter().all(|&g| g == 5.0));
    }

    #[test]
    fn test_apply_to_gradients_all_frozen() {
        let config = FreezeConfig::all_frozen();
        let freezer = ParameterFreezer::new(config);
        let mut grad_groups = vec![
            (GaussianParamGroup::Positions, vec![1.0f32; 3]),
            (GaussianParamGroup::Rotations, vec![1.0f32; 4]),
            (GaussianParamGroup::ShDc, vec![1.0f32; 2]),
        ];
        let zeroed = freezer.apply_to_gradients(&mut grad_groups);
        assert_eq!(zeroed, 9); // 3 + 4 + 2
        for (_, grads) in &grad_groups {
            assert!(grads.iter().all(|&g| g == 0.0));
        }
    }

    // -----------------------------------------------------------------------
    // check_frozen_gradients_zeroed tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_check_frozen_gradients_zeroed_clean() {
        let mask = front_frozen_mask(6, 3);
        let grads = vec![0.0f32, 0.0, 0.0, 1.0, 2.0, 3.0];
        let ok = check_frozen_gradients_zeroed(&grads, &mask).expect("no error");
        assert!(ok);
    }

    #[test]
    fn test_check_frozen_gradients_zeroed_dirty() {
        let mask = front_frozen_mask(6, 3);
        let grads = vec![0.0f32, 1.0, 0.0, 1.0, 2.0, 3.0]; // grads[1] non-zero in frozen region
        let ok = check_frozen_gradients_zeroed(&grads, &mask).expect("no error");
        assert!(!ok);
    }

    // -----------------------------------------------------------------------
    // frozen_compute_savings tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_frozen_compute_savings_half() {
        let mask = front_frozen_mask(100, 50);
        let savings = frozen_compute_savings(&mask);
        assert!((savings - 0.5).abs() < 1e-6, "got {}", savings);
    }

    #[test]
    fn test_frozen_compute_savings_empty_mask() {
        let mask = FrozenMask::new_all_frozen(0);
        let savings = frozen_compute_savings(&mask);
        assert_eq!(savings, 0.0);
    }

    #[test]
    fn test_frozen_compute_savings_all_frozen() {
        let mask = FrozenMask::new_all_frozen(10);
        let savings = frozen_compute_savings(&mask);
        assert!((savings - 1.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // split_into_groups tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_split_into_groups_all_populated() {
        let positions = vec![0.0f32; 30]; // 10 gaussians * 3 coords
        let rotations = vec![0.0f32; 40]; // 10 * 4
        let scales = vec![0.0f32; 30]; // 10 * 3
        let opacities = vec![0.0f32; 10]; // 10 * 1
        let sh_dc = vec![0.0f32; 30]; // 10 * 3
        let sh_rest = vec![0.0f32; 450]; // 10 * 45 (degree-3 SH)

        let groups = split_into_groups(
            &positions, &rotations, &scales, &opacities, &sh_dc, &sh_rest,
        );
        assert_eq!(groups.len(), 6);
        assert_eq!(groups[0].0, GaussianParamGroup::Positions);
        assert_eq!(groups[0].1, 30);
        assert_eq!(groups[5].0, GaussianParamGroup::ShRest);
        assert_eq!(groups[5].1, 450);
    }

    #[test]
    fn test_split_into_groups_empty_excluded() {
        let positions = vec![0.0f32; 30];
        let sh_dc = vec![0.0f32; 30];
        // All other groups empty
        let groups = split_into_groups(&positions, &[], &[], &[], &sh_dc, &[]);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].0, GaussianParamGroup::Positions);
        assert_eq!(groups[1].0, GaussianParamGroup::ShDc);
    }

    // -----------------------------------------------------------------------
    // Edge case / boundary tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_freeze_range_out_of_bounds_clamped() {
        let mut mask = FrozenMask::new_none_frozen(5);
        // end > len-1 should be clamped
        mask.freeze_range(3, 100);
        assert_eq!(mask.frozen_count(), 2); // indices 3 and 4
    }

    #[test]
    fn test_unfreeze_range_out_of_bounds_clamped() {
        let mut mask = FrozenMask::new_all_frozen(5);
        mask.unfreeze_range(0, 100);
        assert_eq!(mask.frozen_count(), 0);
    }

    #[test]
    fn test_freeze_range_empty_range() {
        let mut mask = FrozenMask::new_none_frozen(5);
        // start > end is a no-op
        mask.freeze_range(4, 2);
        assert_eq!(mask.frozen_count(), 0);
    }

    #[test]
    fn test_progressive_mask_quarter_way() {
        // At step 25/100, frozen_fraction = 0.75 → 75 frozen
        let mask = progressive_mask(100, 25, 100);
        assert_eq!(mask.frozen_count(), 75);
    }

    #[test]
    fn test_unfreeze_fraction_clamped_at_one() {
        // step > total_steps → should clamp to 1.0
        let frac = unfreeze_fraction(20_000, 500, 10_000);
        assert_eq!(frac, 1.0);
    }

    #[test]
    fn test_format_schedule_contains_group_names() {
        let groups = GaussianParamGroup::all_standard();
        let schedule = ProgressiveUnfreezeSchedule::uniform(groups, 1000, 0).expect("schedule");
        let formatted = schedule.format_schedule();
        assert!(formatted.contains("positions"));
        assert!(formatted.contains("sh_rest"));
        assert!(formatted.contains("1000"));
    }
}
