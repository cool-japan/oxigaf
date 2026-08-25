//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use thiserror::Error;

/// A snapshot of a Gaussian model suitable for diffing.
///
/// All arrays are flat with a fixed stride:
/// - `positions`: length = n_gaussians × 3
/// - `opacities`:  length = n_gaussians  (raw pre-sigmoid logits)
/// - `scales`:     length = n_gaussians × 3 (log-scale)
/// - `colors`:     length = n_gaussians × 3 (SH DC component)
#[derive(Debug, Clone)]
pub struct ModelSnapshot {
    /// Human-readable name for the snapshot (e.g. "checkpoint_500").
    pub name: String,
    /// Training step at which this snapshot was taken.
    pub step: usize,
    /// Number of Gaussians in this snapshot.
    pub n_gaussians: usize,
    /// Flat position array, length = n_gaussians × 3.
    pub positions: Vec<f32>,
    /// Flat opacity logit array, length = n_gaussians.
    pub opacities: Vec<f32>,
    /// Flat log-scale array, length = n_gaussians × 3.
    pub scales: Vec<f32>,
    /// Flat SH DC colour array, length = n_gaussians × 3.
    pub colors: Vec<f32>,
}
impl ModelSnapshot {
    /// Construct and validate a new snapshot.
    ///
    /// Returns [`DiffError::DimensionError`] if any array length is wrong.
    pub fn new(
        name: impl Into<String>,
        step: usize,
        positions: Vec<f32>,
        opacities: Vec<f32>,
        scales: Vec<f32>,
        colors: Vec<f32>,
    ) -> Result<Self, DiffError> {
        if !positions.len().is_multiple_of(3) {
            return Err(DiffError::DimensionError(format!(
                "positions length {} is not divisible by 3",
                positions.len()
            )));
        }
        let n = positions.len() / 3;
        if opacities.len() != n {
            return Err(DiffError::DimensionError(format!(
                "opacities length {} does not match n_gaussians {}",
                opacities.len(),
                n
            )));
        }
        if scales.len() != n * 3 {
            return Err(DiffError::DimensionError(format!(
                "scales length {} does not match n_gaussians*3 {}",
                scales.len(),
                n * 3
            )));
        }
        if colors.len() != n * 3 {
            return Err(DiffError::DimensionError(format!(
                "colors length {} does not match n_gaussians*3 {}",
                colors.len(),
                n * 3
            )));
        }
        Ok(Self {
            name: name.into(),
            step,
            n_gaussians: n,
            positions,
            opacities,
            scales,
            colors,
        })
    }
    /// Return the number of Gaussians stored in this snapshot.
    #[inline]
    pub fn n_gaussians(&self) -> usize {
        self.n_gaussians
    }
    /// Sigmoid-activated opacity for the i-th Gaussian.
    ///
    /// Computes `sigmoid(opacities[i]) = 1 / (1 + exp(-opacities[i]))`.
    #[inline]
    pub fn activated_opacity(&self, i: usize) -> f32 {
        let logit = self.opacities[i];
        1.0 / (1.0 + (-logit).exp())
    }
    /// Exponentiated (activated) scale for the i-th Gaussian on `axis` (0, 1, or 2).
    ///
    /// Computes `exp(scales[i * 3 + axis])`.
    #[inline]
    pub fn activated_scale(&self, i: usize, axis: usize) -> f32 {
        self.scales[i * 3 + axis].exp()
    }
}
/// Full diff between two model snapshots.
#[derive(Debug, Clone)]
pub struct ModelDiff {
    /// Name of model A.
    pub name_a: String,
    /// Name of model B.
    pub name_b: String,
    /// Training step of model A.
    pub step_a: usize,
    /// Training step of model B.
    pub step_b: usize,
    /// Number of Gaussians (both models must agree).
    pub n_gaussians: usize,
    /// Number of Gaussians whose fields actually fed the statistics below.
    ///
    /// Equal to `n_gaussians` unless [`DiffConfig::include_inactive`] is
    /// `false` (for [`diff_models`](crate::diff_tool::diff_models)) or some
    /// Gaussians in A/B went unmatched (for
    /// [`diff_models_variable`](crate::diff_tool::diff_models_variable),
    /// where it equals the matched-pair count). When this is `0`, every
    /// [`FieldDiff`] below reports the
    /// vacuous "nothing changed" statistics (`cosine_similarity: 1.0`, all
    /// zero magnitudes) for lack of anything to compare -- callers must
    /// check this field to distinguish that from the models genuinely being
    /// identical.
    pub n_compared: usize,
    /// Statistics for position differences.
    pub position_diff: FieldDiff,
    /// Statistics for opacity differences.
    pub opacity_diff: FieldDiff,
    /// Statistics for scale differences.
    pub scale_diff: FieldDiff,
    /// Statistics for colour differences.
    pub color_diff: FieldDiff,
    /// Gaussians present in B but not A (simplified: count difference when sizes differ).
    pub added_gaussians: usize,
    /// Gaussians present in A but not B.
    pub removed_gaussians: usize,
    /// Overall change magnitude, normalised to [0, 1].
    pub summary_score: f32,
}
/// Summary of training progress across a sequence of diffs.
#[derive(Debug, Clone)]
pub struct ProgressSummary {
    /// Number of diffs in the sequence.
    pub n_steps: usize,
    /// Total steps from the first snapshot to the last (step_b_last − step_a_first).
    pub total_steps: usize,
    /// Mean summary_score divided by mean step delta (change per step).
    pub mean_change_per_step: f32,
    /// True if summary scores are generally decreasing (converging).
    pub converging: bool,
    /// True if the last few diffs have near-zero change (< stall_threshold).
    pub stalled: bool,
    /// Number of diffs where any field regressed (using default thresholds).
    pub regression_count: usize,
}
/// Per-field difference statistics between two flat float arrays.
#[derive(Debug, Clone)]
pub struct FieldDiff {
    /// Name of the field (e.g. "position", "opacity").
    pub field_name: String,
    /// Mean of (B − A) per element.
    pub mean_change: f32,
    /// Standard deviation of (B − A).
    pub std_change: f32,
    /// Maximum of |B − A| across all elements.
    pub max_abs_change: f32,
    /// Root-mean-square of (B − A).
    pub rms_change: f32,
    /// Fraction of elements where |B − A| > epsilon.
    pub fraction_changed: f32,
    /// L2 norm of (B − A).
    pub l2_distance: f32,
    /// Cosine similarity between A and B.
    pub cosine_similarity: f32,
}
/// Configuration for diff computation.
#[derive(Debug, Clone)]
pub struct DiffConfig {
    /// Threshold used to decide whether an element has "changed" (default 1e-6).
    pub epsilon: f32,
    /// If true, normalise differences by the mean magnitude of A before statistics.
    pub normalize: bool,
    /// If true, include Gaussians with activated opacity < 0.1 in statistics.
    /// If false, those Gaussians are skipped when computing per-Gaussian metrics.
    pub include_inactive: bool,
    /// Spatial radius (in world units) within which two Gaussians are considered
    /// a match during nearest-neighbour spatial matching (default 0.5).
    pub match_radius: f32,
}
impl DiffConfig {
    /// Validate that the configuration values are sensible.
    pub fn validate(&self) -> Result<(), DiffError> {
        if self.epsilon < 0.0 {
            return Err(DiffError::InvalidConfig(format!(
                "epsilon must be >= 0, got {}",
                self.epsilon
            )));
        }
        if self.match_radius <= 0.0 {
            return Err(DiffError::InvalidConfig(format!(
                "match_radius must be > 0, got {}",
                self.match_radius
            )));
        }
        Ok(())
    }
}
/// Report of whether a diff indicates regression.
#[derive(Debug, Clone)]
pub struct RegressionReport {
    /// Mean opacity decreased (opacity regressed).
    pub opacity_regressed: bool,
    /// Mean max scale increased above threshold.
    pub scale_regressed: bool,
    /// RMS position change exceeded threshold.
    pub position_unstable: bool,
    /// Any field regressed.
    pub overall_regression: bool,
    /// Human-readable details for each detected regression.
    pub details: Vec<String>,
}
/// Errors that can occur during model diff operations.
#[derive(Debug, Error)]
pub enum DiffError {
    /// Model A contains no Gaussians.
    #[error("Empty model A")]
    EmptyModelA,
    /// Model B contains no Gaussians.
    #[error("Empty model B")]
    EmptyModelB,
    /// The two models have different numbers of Gaussians.
    #[error("Size mismatch: model A has {a} Gaussians, model B has {b}")]
    SizeMismatch { a: usize, b: usize },
    /// A field name is unrecognised.
    #[error("Invalid field: {0}")]
    InvalidField(String),
    /// A configuration parameter is invalid.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// A dimension error occurred (e.g. flat array length does not match stride).
    #[error("Dimension error: {0}")]
    DimensionError(String),
}
