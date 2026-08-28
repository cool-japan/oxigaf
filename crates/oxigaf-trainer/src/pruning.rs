//! Gaussian pruning utilities for 3D Gaussian Splatting training.
//!
//! Provides opacity-based, scale-based, radius-based, and random pruning strategies,
//! along with mask combinatorics, schedule computation, and compact model extraction.

use oxigaf_render::gaussian::GaussianModel;
use thiserror::Error;

// ────────────────────────────────────────────────────────────────────────────
// Error type
// ────────────────────────────────────────────────────────────────────────────

/// Errors produced by the pruning subsystem.
#[derive(Debug, Error)]
pub enum PruningError {
    #[error("Empty Gaussian model: no Gaussians to prune")]
    EmptyModel,

    #[error("Size mismatch: positions has {positions} elements (expected {expected}×3)")]
    SizeMismatch { positions: usize, expected: usize },

    #[error("Invalid threshold {value}: must be in [0, 1]")]
    InvalidThreshold { value: f32 },

    #[error("Invalid keep ratio {value}: must be in (0, 1]")]
    InvalidKeepRatio { value: f32 },

    #[error("Pruning schedule error: {0}")]
    ScheduleError(String),
}

// ────────────────────────────────────────────────────────────────────────────
// Inline PRNG (xorshift64) — no `rand` dependency
// ────────────────────────────────────────────────────────────────────────────

/// Xorshift64 pseudo-random number generator (period 2^64 − 1).
#[inline(always)]
fn xorshift64(state: &mut u64) -> u64 {
    *state = (*state).max(1);
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Convert a `xorshift64` output to a uniform float in [0, 1).
#[inline(always)]
fn xorshift64_f32(state: &mut u64) -> f32 {
    let raw = xorshift64(state);
    // Use f64 intermediate to avoid precision issues near u64::MAX.
    (raw as f64 / u64::MAX as f64) as f32
}

// ────────────────────────────────────────────────────────────────────────────
// Sigmoid helper
// ────────────────────────────────────────────────────────────────────────────

/// Sigmoid activation: 1 / (1 + exp(-x)).
///
/// Opacities in the Gaussian model are stored in logit space; `sigmoid` maps
/// them to the [0, 1] probability range used by pruning thresholds.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ────────────────────────────────────────────────────────────────────────────
// PruningMask
// ────────────────────────────────────────────────────────────────────────────

/// Boolean mask over Gaussians: `true` = keep, `false` = prune.
#[derive(Debug, Clone)]
pub struct PruningMask {
    /// Per-Gaussian keep flags.
    pub keep: Vec<bool>,
}

impl PruningMask {
    /// Create a mask with `n` entries all set to `default_keep`.
    pub fn new(n: usize, default_keep: bool) -> Self {
        Self {
            keep: vec![default_keep; n],
        }
    }

    /// Number of Gaussians that will be kept (true entries).
    pub fn keep_count(&self) -> usize {
        self.keep.iter().filter(|&&k| k).count()
    }

    /// Number of Gaussians that will be pruned (false entries).
    pub fn prune_count(&self) -> usize {
        self.keep.iter().filter(|&&k| !k).count()
    }

    /// True if the mask contains no entries.
    pub fn is_empty(&self) -> bool {
        self.keep.is_empty()
    }

    /// Total number of Gaussians covered by this mask.
    pub fn len(&self) -> usize {
        self.keep.len()
    }

    /// Logical AND: keep a Gaussian only if BOTH masks say keep.
    pub fn and(&self, other: &PruningMask) -> Self {
        let keep = self
            .keep
            .iter()
            .zip(other.keep.iter())
            .map(|(&a, &b)| a && b)
            .collect();
        Self { keep }
    }

    /// Logical OR: keep a Gaussian if EITHER mask says keep.
    pub fn or(&self, other: &PruningMask) -> Self {
        let keep = self
            .keep
            .iter()
            .zip(other.keep.iter())
            .map(|(&a, &b)| a || b)
            .collect();
        Self { keep }
    }

    /// Invert the mask: swap keep ↔ prune for every Gaussian.
    pub fn invert(&self) -> Self {
        let keep = self.keep.iter().map(|&k| !k).collect();
        Self { keep }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PruningStats
// ────────────────────────────────────────────────────────────────────────────

/// Statistics summarising the effect of one pruning step.
#[derive(Debug, Clone)]
pub struct PruningStats {
    /// Number of Gaussians before pruning.
    pub before_count: usize,
    /// Number of Gaussians remaining after pruning.
    pub after_count: usize,
    /// Number of Gaussians removed.
    pub removed_count: usize,
    /// Fraction removed: `removed_count / before_count`. Zero when `before_count == 0`.
    pub sparsity: f32,
}

impl PruningStats {
    /// Derive statistics given the pre-pruning Gaussian count and the applied mask.
    pub fn compute(before: usize, mask: &PruningMask) -> Self {
        let after = mask.keep_count();
        let removed = before.saturating_sub(after);
        let sparsity = if before == 0 {
            0.0
        } else {
            removed as f32 / before as f32
        };
        Self {
            before_count: before,
            after_count: after,
            removed_count: removed,
            sparsity,
        }
    }

    /// Human-readable one-liner, e.g. `"Pruned 1 000 → 800 (-200, sparsity 20.00%)"`.
    pub fn format_summary(&self) -> String {
        format!(
            "Pruned {} → {} (-{}, sparsity {:.2}%)",
            self.before_count,
            self.after_count,
            self.removed_count,
            self.sparsity * 100.0,
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PruningSchedule
// ────────────────────────────────────────────────────────────────────────────

/// Cubic sparsity schedule: sparsity ramps from 0 to `target_sparsity` over
/// `[pruning_start, pruning_end]` following a cubic ease-out curve.
#[derive(Debug, Clone)]
pub struct PruningSchedule {
    /// Sparsity at the very beginning (always 0.0).
    pub initial_sparsity: f32,
    /// Target sparsity fraction at `pruning_end` (e.g. 0.5 → remove 50 % of Gaussians).
    pub target_sparsity: f32,
    /// Number of warm-up steps before any pruning begins.
    pub warmup_steps: usize,
    /// Training step at which pruning begins.
    pub pruning_start: usize,
    /// Training step at which pruning reaches `target_sparsity`.
    pub pruning_end: usize,
}

impl PruningSchedule {
    /// Construct a pruning schedule.
    ///
    /// # Errors
    /// Returns [`PruningError::ScheduleError`] if `target_sparsity` is outside
    /// (0, 1] or `pruning_end <= pruning_start`.
    pub fn new(
        target_sparsity: f32,
        pruning_start: usize,
        pruning_end: usize,
    ) -> Result<Self, PruningError> {
        if !(0.0 < target_sparsity && target_sparsity <= 1.0) {
            return Err(PruningError::ScheduleError(format!(
                "target_sparsity {target_sparsity} must be in (0, 1]"
            )));
        }
        if pruning_end <= pruning_start {
            return Err(PruningError::ScheduleError(format!(
                "pruning_end ({pruning_end}) must be strictly greater than pruning_start ({pruning_start})"
            )));
        }
        Ok(Self {
            initial_sparsity: 0.0,
            target_sparsity,
            warmup_steps: pruning_start,
            pruning_start,
            pruning_end,
        })
    }

    /// Compute the desired sparsity at the given training `step`.
    ///
    /// - Before `pruning_start`: returns 0.0 (initial sparsity).
    /// - After `pruning_end`: returns `target_sparsity`.
    /// - In between: cubic ease-out interpolation.
    pub fn sparsity_at_step(&self, step: usize) -> f32 {
        cubic_sparsity(
            self.initial_sparsity,
            self.target_sparsity,
            step,
            self.pruning_start,
            self.pruning_end,
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PruningConfig & GaussianPruner
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for [`GaussianPruner`].
#[derive(Debug, Clone)]
pub struct PruningConfig {
    /// Opacity threshold (after sigmoid).  Gaussians with sigmoid(opacity) below
    /// this value are removed.  Default: 0.005.
    pub opacity_threshold: f32,
    /// Maximum allowed screen-space radius fraction.  Gaussians whose projected
    /// radius exceeds this value are removed.  Default: 0.1.
    pub max_screen_radius: f32,
    /// Minimum allowed log-scale value.  Gaussians where every scale dimension
    /// satisfies exp(scale) < exp(min_scale_log) are removed.  Default: -10.0.
    pub min_scale_log: f32,
    /// How often (in training steps) the opacity statistics are reset.  Default: 3000.
    pub opacity_reset_interval: u32,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            opacity_threshold: 0.005,
            max_screen_radius: 0.1,
            min_scale_log: -10.0,
            opacity_reset_interval: 3000,
        }
    }
}

/// High-level Gaussian pruner driven by a [`PruningConfig`].
#[derive(Debug, Clone)]
pub struct GaussianPruner {
    /// Configuration used by all pruning methods.
    pub config: PruningConfig,
}

impl GaussianPruner {
    /// Create a new pruner with the given configuration.
    pub fn new(config: PruningConfig) -> Self {
        Self { config }
    }

    /// Build an opacity-based pruning mask from logit-space opacities.
    ///
    /// Internally applies [`sigmoid`] and compares against `config.opacity_threshold`.
    pub fn prune_by_opacity(&self, opacities: &[f32]) -> Result<PruningMask, PruningError> {
        prune_by_opacity(opacities, self.config.opacity_threshold)
    }

    /// Build a screen-radius–based pruning mask.
    ///
    /// Gaussians with `radii[i] > config.max_screen_radius` are removed.
    pub fn prune_by_screen_radius(&self, radii: &[f32]) -> Result<PruningMask, PruningError> {
        prune_large_gaussians(radii, self.config.max_screen_radius)
    }

    /// Build a scale-based pruning mask using `config.min_scale_log`.
    ///
    /// Gaussians where every scale dimension satisfies
    /// `exp(scale) < exp(config.min_scale_log)` are removed — i.e. the
    /// linear-space threshold passed to [`prune_small_gaussians`] is
    /// `exp(config.min_scale_log)`, matching this field's documented
    /// formula.
    pub fn prune_by_min_scale(&self, scales: &[f32]) -> Result<PruningMask, PruningError> {
        prune_small_gaussians(scales, self.config.min_scale_log.exp())
    }

    /// Build a combined mask: prune if opacity is too low OR screen radius is too large.
    pub fn prune_combined(
        &self,
        opacities: &[f32],
        radii: &[f32],
    ) -> Result<PruningMask, PruningError> {
        if opacities.len() != radii.len() {
            return Err(PruningError::SizeMismatch {
                positions: opacities.len(),
                expected: radii.len(),
            });
        }
        let opacity_mask = self.prune_by_opacity(opacities)?;
        let radius_mask = self.prune_by_screen_radius(radii)?;
        // Keep a Gaussian only if BOTH masks say keep (neither condition triggered).
        Ok(opacity_mask.and(&radius_mask))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Free functions
// ────────────────────────────────────────────────────────────────────────────

/// Prune Gaussians whose sigmoid(opacity) < `threshold`.
///
/// `threshold` must lie in [0, 1].
///
/// Returns a [`PruningMask`] where `keep[i] = sigmoid(opacities[i]) >= threshold`.
pub fn prune_by_opacity(opacities: &[f32], threshold: f32) -> Result<PruningMask, PruningError> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(PruningError::InvalidThreshold { value: threshold });
    }
    let keep = opacities
        .iter()
        .map(|&logit| sigmoid(logit) >= threshold)
        .collect();
    Ok(PruningMask { keep })
}

/// Prune Gaussians that are too small on the model surface.
///
/// For each Gaussian *i* the effective scale is `max(exp(scales[i*3..i*3+3]))`.
/// Gaussians where this maximum is below `threshold` are pruned.
///
/// `threshold` must be non-negative.
///
/// # Errors
/// - [`PruningError::InvalidThreshold`] if `threshold` is outside [0, 1].
///   (We accept any finite non-negative value; values > 1.0 are allowed for
///   world-space scales, so we only reject negatives.)
pub fn prune_small_gaussians(scales: &[f32], threshold: f32) -> Result<PruningMask, PruningError> {
    if threshold < 0.0 || !threshold.is_finite() {
        return Err(PruningError::InvalidThreshold { value: threshold });
    }
    if scales.is_empty() {
        return Ok(PruningMask::new(0, true));
    }
    let n = scales.len() / 3;
    let keep = (0..n)
        .map(|i| {
            let base = i * 3;
            let max_scale = scales[base]
                .exp()
                .max(scales[base + 1].exp())
                .max(scales[base + 2].exp());
            max_scale >= threshold
        })
        .collect();
    Ok(PruningMask { keep })
}

/// Prune Gaussians that project too large on the screen.
///
/// Gaussians with `radii[i] > max_radius` are pruned.
///
/// `max_radius` must be finite and non-negative.
pub fn prune_large_gaussians(radii: &[f32], max_radius: f32) -> Result<PruningMask, PruningError> {
    if max_radius < 0.0 || !max_radius.is_finite() {
        return Err(PruningError::InvalidThreshold { value: max_radius });
    }
    let keep = radii.iter().map(|&r| r <= max_radius).collect();
    Ok(PruningMask { keep })
}

/// Randomly prune Gaussians to reach an approximate `keep_ratio`.
///
/// Uses an inline xorshift64 PRNG seeded with `seed`. Each of the `n` Gaussians
/// is kept independently with probability `keep_ratio`.
///
/// `keep_ratio` must be in (0, 1].
pub fn random_prune(n: usize, keep_ratio: f32, seed: u64) -> Result<PruningMask, PruningError> {
    if !(keep_ratio > 0.0 && keep_ratio <= 1.0) {
        return Err(PruningError::InvalidKeepRatio { value: keep_ratio });
    }
    let mut state = seed.max(1);
    let keep = (0..n)
        .map(|_| xorshift64_f32(&mut state) < keep_ratio)
        .collect();
    Ok(PruningMask { keep })
}

/// Prune the lowest-scoring fraction of Gaussians to reach `target_sparsity`.
///
/// `scores` is a per-Gaussian importance/quality score (higher = keep
/// preferentially — e.g. an EMA of opacity or contribution to rendered
/// loss). The `ceil(n * target_sparsity)` lowest-scoring Gaussians are
/// pruned. This is the mask-producing counterpart
/// [`PruningSchedule::sparsity_at_step`] / [`cubic_sparsity`] otherwise
/// lack: those compute a target sparsity *fraction*, but every other
/// mask-producing function in this module is threshold-based rather than
/// rank-based, so without this there was no way to act on a scheduled
/// sparsity target at all.
///
/// Runs in O(n) via [`slice::select_nth_unstable_by`] rather than a full
/// O(n log n) sort, since only the lowest-`k` partition (not a total order)
/// is needed.
///
/// # Errors
/// - [`PruningError::EmptyModel`] if `scores` is empty.
/// - [`PruningError::InvalidThreshold`] if `target_sparsity` is outside `[0, 1]`.
pub fn prune_to_sparsity(
    scores: &[f32],
    target_sparsity: f32,
) -> Result<PruningMask, PruningError> {
    if scores.is_empty() {
        return Err(PruningError::EmptyModel);
    }
    if !(0.0..=1.0).contains(&target_sparsity) {
        return Err(PruningError::InvalidThreshold {
            value: target_sparsity,
        });
    }
    let n = scores.len();
    let n_prune = (((n as f32) * target_sparsity).ceil() as usize).min(n);

    if n_prune == 0 {
        return Ok(PruningMask::new(n, true));
    }
    if n_prune == n {
        return Ok(PruningMask::new(n, false));
    }

    // Partition (not fully sort) so the lowest `n_prune` scores land in the
    // front partition, each paired with its original index.
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.select_nth_unstable_by(n_prune - 1, |a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = vec![true; n];
    for &(idx, _) in &indexed[..n_prune] {
        keep[idx] = false;
    }
    Ok(PruningMask { keep })
}

/// Result of [`apply_mask`]: `(positions, rotations, scales, opacities, sh_coefficients)`.
type MaskedGaussians = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// Apply a pruning mask: extract the kept Gaussians from all parameter arrays.
///
/// All arrays use flat layout:
/// - `positions`:       N × 3
/// - `rotations`:       N × 4  (quaternion [qx, qy, qz, qw])
/// - `scales`:          N × 3  (log-scale)
/// - `opacities`:       N
/// - `sh_coefficients`: N × C  (C = `sh_coefficients.len() / N`, may be 0)
///
/// Returns five compact arrays in the same layout but containing only the kept
/// Gaussians, in their original order.
///
/// # Errors
/// - [`PruningError::EmptyModel`] if `positions` is empty.
/// - [`PruningError::SizeMismatch`] if any array has the wrong length.
pub fn apply_mask(
    positions: &[f32],
    rotations: &[f32],
    scales: &[f32],
    opacities: &[f32],
    sh_coefficients: &[f32],
    mask: &PruningMask,
) -> Result<MaskedGaussians, PruningError> {
    if positions.is_empty() {
        return Err(PruningError::EmptyModel);
    }
    let n = positions.len() / 3;

    // Validate positions length divisibility.
    if !positions.len().is_multiple_of(3) {
        return Err(PruningError::SizeMismatch {
            positions: positions.len(),
            expected: n,
        });
    }
    // Validate all other arrays.
    if rotations.len() != n * 4 {
        return Err(PruningError::SizeMismatch {
            positions: rotations.len(),
            expected: n * 4,
        });
    }
    if scales.len() != n * 3 {
        return Err(PruningError::SizeMismatch {
            positions: scales.len(),
            expected: n * 3,
        });
    }
    if opacities.len() != n {
        return Err(PruningError::SizeMismatch {
            positions: opacities.len(),
            expected: n,
        });
    }
    if mask.len() != n {
        return Err(PruningError::SizeMismatch {
            positions: mask.len(),
            expected: n,
        });
    }
    // Compute per-Gaussian SH coefficient count.
    let sh_per_gaussian = if n > 0 && !sh_coefficients.is_empty() {
        if !sh_coefficients.len().is_multiple_of(n) {
            return Err(PruningError::SizeMismatch {
                positions: sh_coefficients.len(),
                expected: n,
            });
        }
        sh_coefficients.len() / n
    } else {
        0
    };

    let kept: usize = mask.keep_count();

    let mut out_positions = Vec::with_capacity(kept * 3);
    let mut out_rotations = Vec::with_capacity(kept * 4);
    let mut out_scales = Vec::with_capacity(kept * 3);
    let mut out_opacities = Vec::with_capacity(kept);
    let mut out_sh = Vec::with_capacity(kept * sh_per_gaussian);

    for i in 0..n {
        if !mask.keep[i] {
            continue;
        }
        // positions
        out_positions.extend_from_slice(&positions[i * 3..i * 3 + 3]);
        // rotations
        out_rotations.extend_from_slice(&rotations[i * 4..i * 4 + 4]);
        // scales
        out_scales.extend_from_slice(&scales[i * 3..i * 3 + 3]);
        // opacity (scalar)
        out_opacities.push(opacities[i]);
        // SH coefficients
        if sh_per_gaussian > 0 {
            out_sh.extend_from_slice(&sh_coefficients[i * sh_per_gaussian..][..sh_per_gaussian]);
        }
    }

    Ok((
        out_positions,
        out_rotations,
        out_scales,
        out_opacities,
        out_sh,
    ))
}

/// Apply a pruning mask directly to a [`GaussianModel`], compacting every
/// per-Gaussian buffer including the FLAME-binding side arrays
/// (`face_indices`, `barycentric`, `local_offsets`, `is_rigid`) that
/// [`apply_mask`] — which only knows about loose flat parameter arrays —
/// cannot touch, so it alone cannot prune a real `GaussianModel` without
/// desyncing those side arrays from the compacted Gaussians.
///
/// # Errors
/// - [`PruningError::EmptyModel`] if the model has no Gaussians.
/// - [`PruningError::SizeMismatch`] if `mask.len()`, `face_indices.len()`,
///   `barycentric.len()`, `local_offsets.len()`, `is_rigid.len()`, or
///   `sh_coeffs.len()` do not match the model's Gaussian count.
pub fn apply_mask_to_model(
    model: &GaussianModel,
    mask: &PruningMask,
) -> Result<GaussianModel, PruningError> {
    let n = model.gaussians.len();
    if n == 0 {
        return Err(PruningError::EmptyModel);
    }
    if mask.len() != n {
        return Err(PruningError::SizeMismatch {
            positions: mask.len(),
            expected: n,
        });
    }
    if model.face_indices.len() != n {
        return Err(PruningError::SizeMismatch {
            positions: model.face_indices.len(),
            expected: n,
        });
    }
    if model.barycentric.len() != n {
        return Err(PruningError::SizeMismatch {
            positions: model.barycentric.len(),
            expected: n,
        });
    }
    if model.local_offsets.len() != n {
        return Err(PruningError::SizeMismatch {
            positions: model.local_offsets.len(),
            expected: n,
        });
    }
    if model.is_rigid.len() != n {
        return Err(PruningError::SizeMismatch {
            positions: model.is_rigid.len(),
            expected: n,
        });
    }
    let sh_per_gaussian = if model.sh_coeffs.is_empty() {
        0
    } else {
        if !model.sh_coeffs.len().is_multiple_of(n) {
            return Err(PruningError::SizeMismatch {
                positions: model.sh_coeffs.len(),
                expected: n,
            });
        }
        model.sh_coeffs.len() / n
    };

    let kept = mask.keep_count();
    let mut gaussians = Vec::with_capacity(kept);
    let mut sh_coeffs = Vec::with_capacity(kept * sh_per_gaussian);
    let mut face_indices = Vec::with_capacity(kept);
    let mut barycentric = Vec::with_capacity(kept);
    let mut local_offsets = Vec::with_capacity(kept);
    let mut is_rigid = Vec::with_capacity(kept);

    for i in 0..n {
        if !mask.keep[i] {
            continue;
        }
        gaussians.push(model.gaussians[i]);
        if sh_per_gaussian > 0 {
            sh_coeffs.extend_from_slice(&model.sh_coeffs[i * sh_per_gaussian..][..sh_per_gaussian]);
        }
        face_indices.push(model.face_indices[i]);
        barycentric.push(model.barycentric[i]);
        local_offsets.push(model.local_offsets[i]);
        is_rigid.push(model.is_rigid[i]);
    }

    Ok(GaussianModel {
        gaussians,
        sh_coeffs,
        sh_degree: model.sh_degree,
        face_indices,
        barycentric,
        local_offsets,
        is_rigid,
    })
}

/// Compute cubic sparsity for iterative pruning schedules.
///
/// The sparsity follows a cubic ease-out curve:
/// ```text
/// s(t) = initial + (target - initial) * (1 - (1 - t)^3)
/// ```
/// where `t = clamp((step - start_step) / (end_step - start_step), 0, 1)`.
///
/// Returns `initial` before `start_step` and `target` after `end_step`.
pub fn cubic_sparsity(
    initial: f32,
    target: f32,
    step: usize,
    start_step: usize,
    end_step: usize,
) -> f32 {
    if step <= start_step {
        return initial;
    }
    if step >= end_step {
        return target;
    }
    let span = (end_step - start_step) as f32;
    let elapsed = (step - start_step) as f32;
    let t = (elapsed / span).clamp(0.0, 1.0);
    let one_minus_t = 1.0 - t;
    initial + (target - initial) * (1.0 - one_minus_t * one_minus_t * one_minus_t)
}

/// Count how many Gaussians would be pruned by opacity thresholding.
///
/// Returns the number of Gaussians where `sigmoid(opacities[i]) < threshold`.
pub fn count_prunable_by_opacity(opacities: &[f32], threshold: f32) -> usize {
    opacities
        .iter()
        .filter(|&&logit| sigmoid(logit) < threshold)
        .count()
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PruningMask ──────────────────────────────────────────────────────────

    #[test]
    fn test_mask_new_all_keep() {
        let m = PruningMask::new(5, true);
        assert_eq!(m.len(), 5);
        assert_eq!(m.keep_count(), 5);
        assert_eq!(m.prune_count(), 0);
    }

    #[test]
    fn test_mask_new_all_prune() {
        let m = PruningMask::new(4, false);
        assert_eq!(m.len(), 4);
        assert_eq!(m.keep_count(), 0);
        assert_eq!(m.prune_count(), 4);
    }

    #[test]
    fn test_mask_is_empty() {
        assert!(PruningMask::new(0, true).is_empty());
        assert!(!PruningMask::new(1, true).is_empty());
    }

    #[test]
    fn test_mask_and() {
        let a = PruningMask {
            keep: vec![true, true, false, false],
        };
        let b = PruningMask {
            keep: vec![true, false, true, false],
        };
        let c = a.and(&b);
        assert_eq!(c.keep, vec![true, false, false, false]);
    }

    #[test]
    fn test_mask_or() {
        let a = PruningMask {
            keep: vec![true, true, false, false],
        };
        let b = PruningMask {
            keep: vec![true, false, true, false],
        };
        let c = a.or(&b);
        assert_eq!(c.keep, vec![true, true, true, false]);
    }

    #[test]
    fn test_mask_invert() {
        let m = PruningMask {
            keep: vec![true, false, true],
        };
        let inv = m.invert();
        assert_eq!(inv.keep, vec![false, true, false]);
    }

    #[test]
    fn test_mask_keep_and_prune_counts_mixed() {
        let m = PruningMask {
            keep: vec![true, false, true, false, true],
        };
        assert_eq!(m.keep_count(), 3);
        assert_eq!(m.prune_count(), 2);
    }

    #[test]
    fn test_mask_zero_length() {
        let m = PruningMask::new(0, false);
        assert_eq!(m.keep_count(), 0);
        assert_eq!(m.prune_count(), 0);
    }

    // ── prune_by_opacity ────────────────────────────────────────────────────

    #[test]
    fn test_opacity_prune_below_threshold() {
        // sigmoid(-5.0) ≈ 0.0067, below 0.01 → prune
        let opacities = vec![-5.0_f32];
        let mask = prune_by_opacity(&opacities, 0.01).expect("should succeed");
        assert!(!mask.keep[0]);
    }

    #[test]
    fn test_opacity_prune_above_threshold() {
        // sigmoid(5.0) ≈ 0.993, above 0.01 → keep
        let opacities = vec![5.0_f32];
        let mask = prune_by_opacity(&opacities, 0.01).expect("should succeed");
        assert!(mask.keep[0]);
    }

    #[test]
    fn test_opacity_prune_mixed() {
        // sigmoid(-100) ≈ 0 → prune; sigmoid(100) ≈ 1 → keep
        let opacities = vec![-100.0_f32, 100.0_f32];
        let mask = prune_by_opacity(&opacities, 0.5).expect("should succeed");
        assert!(!mask.keep[0]);
        assert!(mask.keep[1]);
    }

    #[test]
    fn test_opacity_prune_invalid_threshold_negative() {
        let result = prune_by_opacity(&[0.0], -0.1);
        assert!(matches!(result, Err(PruningError::InvalidThreshold { .. })));
    }

    #[test]
    fn test_opacity_prune_invalid_threshold_above_one() {
        let result = prune_by_opacity(&[0.0], 1.1);
        assert!(matches!(result, Err(PruningError::InvalidThreshold { .. })));
    }

    #[test]
    fn test_opacity_prune_threshold_boundary() {
        // sigmoid(0.0) = 0.5; threshold = 0.5 → keep (>=)
        let mask = prune_by_opacity(&[0.0], 0.5).expect("should succeed");
        assert!(mask.keep[0]);
    }

    // ── prune_small_gaussians ───────────────────────────────────────────────

    #[test]
    fn test_small_gaussians_pruned() {
        // log-scale -20 → exp(-20) ≈ 2e-9, well below threshold 0.001
        let scales = vec![-20.0_f32, -20.0, -20.0];
        let mask = prune_small_gaussians(&scales, 0.001).expect("should succeed");
        assert!(!mask.keep[0]);
    }

    #[test]
    fn test_normal_size_gaussians_kept() {
        // log-scale 0.0 → exp(0) = 1.0, well above threshold 0.001
        let scales = vec![0.0_f32, 0.0, 0.0];
        let mask = prune_small_gaussians(&scales, 0.001).expect("should succeed");
        assert!(mask.keep[0]);
    }

    #[test]
    fn test_small_gaussians_invalid_threshold() {
        let result = prune_small_gaussians(&[0.0, 0.0, 0.0], -1.0);
        assert!(matches!(result, Err(PruningError::InvalidThreshold { .. })));
    }

    #[test]
    fn test_small_gaussians_two_gaussians_mixed() {
        // G0: max(exp(-20), exp(-20), exp(-20)) = exp(-20) ≈ 0 → prune
        // G1: max(exp(1), exp(1), exp(1)) = e ≈ 2.71 → keep
        let scales = vec![-20.0_f32, -20.0, -20.0, 1.0, 1.0, 1.0];
        let mask = prune_small_gaussians(&scales, 0.001).expect("should succeed");
        assert!(!mask.keep[0]);
        assert!(mask.keep[1]);
    }

    // ── prune_large_gaussians ───────────────────────────────────────────────

    #[test]
    fn test_large_radius_pruned() {
        let radii = vec![0.5_f32];
        let mask = prune_large_gaussians(&radii, 0.1).expect("should succeed");
        assert!(!mask.keep[0]);
    }

    #[test]
    fn test_normal_radius_kept() {
        let radii = vec![0.05_f32];
        let mask = prune_large_gaussians(&radii, 0.1).expect("should succeed");
        assert!(mask.keep[0]);
    }

    #[test]
    fn test_large_gaussians_mixed() {
        let radii = vec![0.05_f32, 0.2, 0.1];
        let mask = prune_large_gaussians(&radii, 0.1).expect("should succeed");
        assert!(mask.keep[0]); // 0.05 <= 0.1
        assert!(!mask.keep[1]); // 0.2 > 0.1
        assert!(mask.keep[2]); // 0.1 <= 0.1 (exactly equal → keep)
    }

    #[test]
    fn test_large_gaussians_invalid_max_radius() {
        let result = prune_large_gaussians(&[0.1], -0.01);
        assert!(matches!(result, Err(PruningError::InvalidThreshold { .. })));
    }

    // ── random_prune ────────────────────────────────────────────────────────

    #[test]
    fn test_random_prune_deterministic() {
        let m1 = random_prune(100, 0.5, 42).expect("should succeed");
        let m2 = random_prune(100, 0.5, 42).expect("should succeed");
        assert_eq!(m1.keep, m2.keep);
    }

    #[test]
    fn test_random_prune_different_seeds() {
        let m1 = random_prune(100, 0.5, 1).expect("should succeed");
        let m2 = random_prune(100, 0.5, 9999).expect("should succeed");
        // With high probability two different seeds produce different results
        assert_ne!(m1.keep, m2.keep);
    }

    #[test]
    fn test_random_prune_keep_ratio_one() {
        // keep_ratio = 1.0 → all Gaussians should be kept
        let mask = random_prune(50, 1.0, 7).expect("should succeed");
        assert_eq!(mask.keep_count(), 50);
    }

    #[test]
    fn test_random_prune_invalid_ratio_zero() {
        let result = random_prune(10, 0.0, 1);
        assert!(matches!(result, Err(PruningError::InvalidKeepRatio { .. })));
    }

    // ── prune_to_sparsity ────────────────────────────────────────────────────

    #[test]
    fn test_prune_to_sparsity_prunes_lowest_scoring() {
        let scores = vec![0.9_f32, 0.1, 0.5, 0.8, 0.2];
        // target 40% of 5 => ceil(2.0) = 2 lowest-scoring pruned:
        // index 1 (0.1) and index 4 (0.2).
        let mask = prune_to_sparsity(&scores, 0.4).expect("prune_to_sparsity");
        assert_eq!(mask.prune_count(), 2);
        assert!(!mask.keep[1], "lowest score (index 1) should be pruned");
        assert!(
            !mask.keep[4],
            "second-lowest score (index 4) should be pruned"
        );
        assert!(mask.keep[0] && mask.keep[2] && mask.keep[3]);
    }

    #[test]
    fn test_prune_to_sparsity_zero_prunes_nothing() {
        let scores = vec![0.1_f32, 0.2, 0.3];
        let mask = prune_to_sparsity(&scores, 0.0).expect("prune_to_sparsity");
        assert_eq!(mask.prune_count(), 0);
    }

    #[test]
    fn test_prune_to_sparsity_one_prunes_everything() {
        let scores = vec![0.1_f32, 0.2, 0.3];
        let mask = prune_to_sparsity(&scores, 1.0).expect("prune_to_sparsity");
        assert_eq!(mask.keep_count(), 0);
    }

    #[test]
    fn test_prune_to_sparsity_empty_scores_error() {
        let result = prune_to_sparsity(&[], 0.5);
        assert!(matches!(result, Err(PruningError::EmptyModel)));
    }

    #[test]
    fn test_prune_to_sparsity_invalid_target_error() {
        let result = prune_to_sparsity(&[0.1, 0.2], 1.5);
        assert!(matches!(result, Err(PruningError::InvalidThreshold { .. })));
    }

    // ── GaussianPruner::prune_by_min_scale ──────────────────────────────────

    #[test]
    fn test_prune_by_min_scale_uses_config_threshold() {
        let config = PruningConfig {
            min_scale_log: 0.0, // exp(0.0) = 1.0 linear threshold
            ..PruningConfig::default()
        };
        let pruner = GaussianPruner::new(config);
        // log-scales: exp(-1.0)=0.37 (below threshold), exp(1.0)=2.72 (above)
        let scales = vec![-1.0_f32, -1.0, -1.0, 1.0, 1.0, 1.0];
        let mask = pruner
            .prune_by_min_scale(&scales)
            .expect("prune_by_min_scale");
        assert_eq!(mask.keep, vec![false, true]);
    }

    // ── apply_mask ──────────────────────────────────────────────────────────

    type GaussianTuple = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    fn make_gaussian(n: usize, sh_dim: usize) -> GaussianTuple {
        let positions: Vec<f32> = (0..n * 3).map(|i| i as f32).collect();
        let rotations: Vec<f32> = (0..n * 4).map(|i| i as f32 * 0.1).collect();
        let scales: Vec<f32> = vec![0.0_f32; n * 3];
        let opacities: Vec<f32> = vec![0.0_f32; n];
        let sh: Vec<f32> = (0..n * sh_dim).map(|i| i as f32 * 0.01).collect();
        (positions, rotations, scales, opacities, sh)
    }

    #[test]
    fn test_apply_mask_basic_extraction() {
        let (pos, rot, sc, op, sh) = make_gaussian(3, 0);
        // keep only Gaussian 1 (middle)
        let mask = PruningMask {
            keep: vec![false, true, false],
        };
        let (p2, r2, s2, o2, sh2) =
            apply_mask(&pos, &rot, &sc, &op, &sh, &mask).expect("apply_mask should succeed");
        // Gaussian 1: positions [3, 4, 5]
        assert_eq!(p2, vec![3.0, 4.0, 5.0]);
        // Gaussian 1: rotations [0.4, 0.5, 0.6, 0.7]
        assert!((r2[0] - 0.4).abs() < 1e-5);
        assert_eq!(s2.len(), 3);
        assert_eq!(o2.len(), 1);
        assert!(sh2.is_empty());
    }

    #[test]
    fn test_apply_mask_all_keep() {
        let (pos, rot, sc, op, sh) = make_gaussian(4, 3);
        let mask = PruningMask::new(4, true);
        let (p2, r2, s2, o2, sh2) =
            apply_mask(&pos, &rot, &sc, &op, &sh, &mask).expect("apply_mask should succeed");
        assert_eq!(p2, pos);
        assert_eq!(r2, rot);
        assert_eq!(s2, sc);
        assert_eq!(o2, op);
        assert_eq!(sh2, sh);
    }

    #[test]
    fn test_apply_mask_all_prune() {
        let (pos, rot, sc, op, sh) = make_gaussian(4, 3);
        let mask = PruningMask::new(4, false);
        let (p2, r2, s2, o2, sh2) =
            apply_mask(&pos, &rot, &sc, &op, &sh, &mask).expect("apply_mask should succeed");
        assert!(p2.is_empty());
        assert!(r2.is_empty());
        assert!(s2.is_empty());
        assert!(o2.is_empty());
        assert!(sh2.is_empty());
    }

    #[test]
    fn test_apply_mask_sh_handling() {
        // 2 Gaussians, sh_dim = 9 (second-degree SH)
        let (pos, rot, sc, op, sh) = make_gaussian(2, 9);
        // Keep only Gaussian 0
        let mask = PruningMask {
            keep: vec![true, false],
        };
        let (_, _, _, _, sh2) =
            apply_mask(&pos, &rot, &sc, &op, &sh, &mask).expect("apply_mask should succeed");
        assert_eq!(sh2.len(), 9);
        // SH for Gaussian 0 = sh[0..9]
        assert_eq!(sh2, sh[0..9].to_vec());
    }

    #[test]
    fn test_apply_mask_empty_model_error() {
        let mask = PruningMask::new(0, true);
        let result = apply_mask(&[], &[], &[], &[], &[], &mask);
        assert!(matches!(result, Err(PruningError::EmptyModel)));
    }

    // ── apply_mask_to_model ──────────────────────────────────────────────────

    fn make_test_model(n: usize, sh_dim: usize) -> GaussianModel {
        use oxigaf_render::gaussian::GaussianAttributes;
        let gaussians: Vec<GaussianAttributes> = (0..n)
            .map(|i| GaussianAttributes {
                position: [i as f32, i as f32, i as f32],
                _pad0: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [0.0, 0.0, 0.0],
                opacity: 0.0,
            })
            .collect();
        GaussianModel {
            gaussians,
            sh_coeffs: (0..n * sh_dim).map(|i| i as f32 * 0.01).collect(),
            sh_degree: 0,
            face_indices: (0..n as u32).collect(),
            barycentric: vec![[1.0, 0.0, 0.0]; n],
            local_offsets: vec![[0.0; 3]; n],
            is_rigid: vec![false; n],
        }
    }

    #[test]
    fn test_apply_mask_to_model_compacts_flame_side_arrays() {
        let model = make_test_model(3, 0);
        // keep only Gaussian 1 (middle)
        let mask = PruningMask {
            keep: vec![false, true, false],
        };
        let pruned = apply_mask_to_model(&model, &mask).expect("apply_mask_to_model");
        assert_eq!(pruned.gaussians.len(), 1);
        assert_eq!(pruned.gaussians[0].position, [1.0, 1.0, 1.0]);
        assert_eq!(pruned.face_indices, vec![1]);
        assert_eq!(pruned.barycentric.len(), 1);
        assert_eq!(pruned.local_offsets.len(), 1);
        assert_eq!(pruned.is_rigid.len(), 1);
    }

    #[test]
    fn test_apply_mask_to_model_with_sh_coeffs() {
        let model = make_test_model(2, 9);
        let mask = PruningMask {
            keep: vec![true, false],
        };
        let pruned = apply_mask_to_model(&model, &mask).expect("apply_mask_to_model");
        assert_eq!(pruned.sh_coeffs.len(), 9);
        assert_eq!(pruned.sh_coeffs, model.sh_coeffs[0..9].to_vec());
    }

    #[test]
    fn test_apply_mask_to_model_empty_error() {
        let model = make_test_model(0, 0);
        let mask = PruningMask::new(0, true);
        let result = apply_mask_to_model(&model, &mask);
        assert!(matches!(result, Err(PruningError::EmptyModel)));
    }

    #[test]
    fn test_apply_mask_to_model_size_mismatch_error() {
        let model = make_test_model(3, 0);
        let mask = PruningMask::new(2, true); // wrong length vs. model's 3
        let result = apply_mask_to_model(&model, &mask);
        assert!(matches!(result, Err(PruningError::SizeMismatch { .. })));
    }

    // ── PruningSchedule ─────────────────────────────────────────────────────

    #[test]
    fn test_schedule_before_start() {
        let sched = PruningSchedule::new(0.5, 1000, 5000).expect("valid schedule");
        assert!((sched.sparsity_at_step(0) - 0.0).abs() < 1e-6);
        assert!((sched.sparsity_at_step(999) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_schedule_after_end() {
        let sched = PruningSchedule::new(0.5, 1000, 5000).expect("valid schedule");
        assert!((sched.sparsity_at_step(5000) - 0.5).abs() < 1e-6);
        assert!((sched.sparsity_at_step(10000) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_schedule_cubic_ramp_midpoint() {
        // At the midpoint t=0.5: cubic_ease(0.5) = 1 - (0.5)^3 = 0.875
        // so sparsity = 0.0 + 0.5 * 0.875 = 0.4375
        let sched = PruningSchedule::new(0.5, 0, 1000).expect("valid schedule");
        let mid = sched.sparsity_at_step(500);
        let expected = 0.5 * (1.0 - 0.5_f32.powi(3));
        assert!(
            (mid - expected).abs() < 1e-5,
            "mid={mid}, expected={expected}"
        );
    }

    #[test]
    fn test_schedule_invalid_end_le_start() {
        let result = PruningSchedule::new(0.5, 5000, 1000);
        assert!(matches!(result, Err(PruningError::ScheduleError(_))));
    }

    // ── PruningStats ────────────────────────────────────────────────────────

    #[test]
    fn test_pruning_stats_sparsity() {
        let mask = PruningMask {
            keep: vec![true, true, false, false, false],
        };
        let stats = PruningStats::compute(5, &mask);
        assert_eq!(stats.before_count, 5);
        assert_eq!(stats.after_count, 2);
        assert_eq!(stats.removed_count, 3);
        assert!((stats.sparsity - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_pruning_stats_format_summary_contains_numbers() {
        let mask = PruningMask::new(100, true);
        let stats = PruningStats::compute(100, &mask);
        let summary = stats.format_summary();
        // Zero removals
        assert!(summary.contains("100"));
        assert!(summary.contains("0.00%") || summary.contains("sparsity"));
    }

    #[test]
    fn test_pruning_stats_zero_before() {
        let mask = PruningMask::new(0, true);
        let stats = PruningStats::compute(0, &mask);
        assert_eq!(stats.sparsity, 0.0);
    }

    // ── GaussianPruner integration ──────────────────────────────────────────

    #[test]
    fn test_gaussian_pruner_opacity() {
        let config = PruningConfig {
            opacity_threshold: 0.5,
            ..PruningConfig::default()
        };
        let pruner = GaussianPruner::new(config);
        // sigmoid(0.0) = 0.5 >= 0.5 → keep; sigmoid(-10.0) ≈ 0 < 0.5 → prune
        let opacities = vec![0.0_f32, -10.0];
        let mask = pruner.prune_by_opacity(&opacities).expect("should succeed");
        assert!(mask.keep[0]);
        assert!(!mask.keep[1]);
    }

    #[test]
    fn test_gaussian_pruner_screen_radius() {
        let config = PruningConfig {
            max_screen_radius: 0.1,
            ..PruningConfig::default()
        };
        let pruner = GaussianPruner::new(config);
        let radii = vec![0.05_f32, 0.2];
        let mask = pruner
            .prune_by_screen_radius(&radii)
            .expect("should succeed");
        assert!(mask.keep[0]);
        assert!(!mask.keep[1]);
    }

    #[test]
    fn test_gaussian_pruner_combined() {
        let config = PruningConfig {
            opacity_threshold: 0.5,
            max_screen_radius: 0.1,
            ..PruningConfig::default()
        };
        let pruner = GaussianPruner::new(config);
        // G0: high opacity, small radius → keep
        // G1: low opacity, small radius → prune (opacity fails)
        // G2: high opacity, large radius → prune (radius fails)
        let opacities = vec![5.0_f32, -10.0, 5.0];
        let radii = vec![0.05_f32, 0.05, 0.5];
        let mask = pruner
            .prune_combined(&opacities, &radii)
            .expect("should succeed");
        assert!(mask.keep[0]);
        assert!(!mask.keep[1]);
        assert!(!mask.keep[2]);
    }

    // ── sigmoid ─────────────────────────────────────────────────────────────

    #[test]
    fn test_sigmoid_zero() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_large_positive() {
        assert!(sigmoid(100.0) > 0.999);
    }

    #[test]
    fn test_sigmoid_large_negative() {
        assert!(sigmoid(-100.0) < 0.001);
    }

    // ── cubic_sparsity ───────────────────────────────────────────────────────

    #[test]
    fn test_cubic_sparsity_before_start() {
        let s = cubic_sparsity(0.0, 0.8, 50, 100, 500);
        assert!((s - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cubic_sparsity_after_end() {
        let s = cubic_sparsity(0.0, 0.8, 600, 100, 500);
        assert!((s - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_cubic_sparsity_at_end_boundary() {
        let s = cubic_sparsity(0.0, 0.8, 500, 100, 500);
        assert!((s - 0.8).abs() < 1e-6);
    }

    // ── count_prunable_by_opacity ────────────────────────────────────────────

    #[test]
    fn test_count_prunable_by_opacity_all_below() {
        let opacities = vec![-100.0_f32, -100.0, -100.0];
        assert_eq!(count_prunable_by_opacity(&opacities, 0.5), 3);
    }

    #[test]
    fn test_count_prunable_by_opacity_none_below() {
        let opacities = vec![100.0_f32, 100.0, 100.0];
        assert_eq!(count_prunable_by_opacity(&opacities, 0.5), 0);
    }
}
