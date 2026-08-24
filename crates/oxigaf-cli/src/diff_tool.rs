//! Diff tool for comparing two Gaussian avatar model snapshots.
//!
//! This module compares two versions of a Gaussian avatar model
//! (snapshots/checkpoints) to analyze training progress, detect regressions,
//! and understand what changed between training steps. It operates on flat
//! float arrays of Gaussian parameters.
//!
//! # Example
//! ```rust,no_run
//! use oxigaf_cli::diff_tool::{
//!     ModelSnapshot, DiffConfig, diff_models, format_model_diff,
//! };
//!
//! let a = ModelSnapshot::new(
//!     "step_0", 0,
//!     vec![0.0f32; 300],   // 100 Gaussians * 3
//!     vec![0.0f32; 100],
//!     vec![0.0f32; 300],
//!     vec![0.5f32; 300],
//! ).expect("valid snapshot");
//! let config = DiffConfig::default();
//! let diff = diff_models(&a, &a, &config).expect("diff ok");
//! println!("{}", format_model_diff(&diff));
//! ```

use std::borrow::Cow;
use std::collections::HashMap;

use thiserror::Error;

// ---------------------------------------------------------------------------
// DiffError
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// ModelSnapshot
// ---------------------------------------------------------------------------

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
        // Infer n_gaussians from the positions array.
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

// ---------------------------------------------------------------------------
// FieldDiff
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// ModelDiff
// ---------------------------------------------------------------------------

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
    /// `false` (for [`diff_models`]) or some Gaussians in A/B went unmatched
    /// (for [`diff_models_variable`], where it equals the matched-pair
    /// count). When this is `0`, every [`FieldDiff`] below reports the
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

// ---------------------------------------------------------------------------
// DiffConfig
// ---------------------------------------------------------------------------

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

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-6,
            normalize: false,
            include_inactive: true,
            match_radius: 0.5,
        }
    }
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

// ---------------------------------------------------------------------------
// Helper: basic statistics over a slice
// ---------------------------------------------------------------------------

/// Compute mean of a slice. Returns 0.0 for empty input.
fn mean_f32(data: &[f32]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let sum: f32 = data.iter().sum();
    sum / data.len() as f32
}

// The former `std_f32` two-pass helper (population std dev over a
// materialised slice) has no remaining callers: `compute_field_diff` now
// computes standard deviation in its single streaming pass via
// `E[d^2] - E[d]^2` instead of allocating a `diffs` array to feed this.

// ---------------------------------------------------------------------------
// compute_field_diff
// ---------------------------------------------------------------------------

/// Compute [`FieldDiff`] between two flat arrays of the same length.
///
/// # Errors
/// Returns [`DiffError::DimensionError`] if `field_a` and `field_b` differ in length.
pub fn compute_field_diff(
    field_a: &[f32],
    field_b: &[f32],
    field_name: impl Into<String>,
    epsilon: f32,
) -> Result<FieldDiff, DiffError> {
    let name = field_name.into();
    if field_a.len() != field_b.len() {
        return Err(DiffError::DimensionError(format!(
            "field '{}': length mismatch {} vs {}",
            name,
            field_a.len(),
            field_b.len()
        )));
    }

    let n = field_a.len();

    if n == 0 {
        return Ok(FieldDiff {
            field_name: name,
            mean_change: 0.0,
            std_change: 0.0,
            max_abs_change: 0.0,
            rms_change: 0.0,
            fraction_changed: 0.0,
            l2_distance: 0.0,
            cosine_similarity: 1.0,
        });
    }

    // Single streaming pass over the paired elements: accumulate everything
    // every statistic below needs without materialising an O(n)
    // intermediate `diffs` array (the diff/model-comparison path this feeds
    // routinely runs on multi-million-Gaussian snapshots).
    let mut sum_d = 0.0f32;
    let mut sum_d2 = 0.0f32;
    let mut max_abs_change = 0.0f32;
    let mut changed_count = 0usize;
    let mut dot_ab = 0.0f32;
    let mut norm_a_sq = 0.0f32;
    let mut norm_b_sq = 0.0f32;

    for (&a, &b) in field_a.iter().zip(field_b.iter()) {
        let d = b - a;
        sum_d += d;
        sum_d2 += d * d;
        let abs_d = d.abs();
        if abs_d > max_abs_change {
            max_abs_change = abs_d;
        }
        if abs_d > epsilon {
            changed_count += 1;
        }
        dot_ab += a * b;
        norm_a_sq += a * a;
        norm_b_sq += b * b;
    }

    let n_f = n as f32;
    let mean_change = sum_d / n_f;
    // Population variance via E[d^2] - E[d]^2 (algebraically identical to
    // the two-pass `mean(sum((d - mean)^2))` this replaces); clamped at 0 to
    // guard against a tiny negative value from floating-point cancellation.
    let variance = (sum_d2 / n_f - mean_change * mean_change).max(0.0);
    let std_change = variance.sqrt();
    let rms_change = (sum_d2 / n_f).sqrt();
    let fraction_changed = changed_count as f32 / n_f;
    let l2_distance = sum_d2.sqrt();

    // Cosine similarity.
    let norm_a = norm_a_sq.sqrt();
    let norm_b = norm_b_sq.sqrt();
    let cosine_similarity = if norm_a == 0.0 && norm_b == 0.0 {
        // Both zero vectors — consider them identical.
        1.0
    } else if norm_a == 0.0 || norm_b == 0.0 {
        // One is zero, the other is not.
        0.0
    } else {
        (dot_ab / (norm_a * norm_b)).clamp(-1.0, 1.0)
    };

    Ok(FieldDiff {
        field_name: name,
        mean_change,
        std_change,
        max_abs_change,
        rms_change,
        fraction_changed,
        l2_distance,
        cosine_similarity,
    })
}

// ---------------------------------------------------------------------------
// diff_models
// ---------------------------------------------------------------------------

/// Compute a full [`ModelDiff`] between two model snapshots.
///
/// # Errors
/// - [`DiffError::EmptyModelA`] / [`DiffError::EmptyModelB`] if either is empty.
/// - [`DiffError::SizeMismatch`] if Gaussian counts differ.
/// - Propagates [`DiffError::DimensionError`] from field diffs.
pub fn diff_models(
    a: &ModelSnapshot,
    b: &ModelSnapshot,
    config: &DiffConfig,
) -> Result<ModelDiff, DiffError> {
    config.validate()?;

    if a.n_gaussians == 0 {
        return Err(DiffError::EmptyModelA);
    }
    if b.n_gaussians == 0 {
        return Err(DiffError::EmptyModelB);
    }
    if a.n_gaussians != b.n_gaussians {
        return Err(DiffError::SizeMismatch {
            a: a.n_gaussians,
            b: b.n_gaussians,
        });
    }

    let n = a.n_gaussians;
    let eps = config.epsilon;

    // Optionally build index masks for active Gaussians.
    let active_indices: Vec<usize> = if config.include_inactive {
        (0..n).collect()
    } else {
        (0..n)
            .filter(|&i| a.activated_opacity(i) >= 0.1 || b.activated_opacity(i) >= 0.1)
            .collect()
    };
    let n_compared = active_indices.len();

    // Gather active fields. When `include_inactive` is true (the default)
    // -- or an explicit filter happens to keep every Gaussian -- this
    // borrows the snapshot's arrays directly rather than copying the whole
    // model: `gather_field` only allocates when `active_indices` is a
    // proper subset.
    let pos_a = gather_field(&a.positions, &active_indices, 3);
    let pos_b = gather_field(&b.positions, &active_indices, 3);
    let opa_a = gather_field(&a.opacities, &active_indices, 1);
    let opa_b = gather_field(&b.opacities, &active_indices, 1);
    let sca_a = gather_field(&a.scales, &active_indices, 3);
    let sca_b = gather_field(&b.scales, &active_indices, 3);
    let col_a = gather_field(&a.colors, &active_indices, 3);
    let col_b = gather_field(&b.colors, &active_indices, 3);

    let (pa, pb) = normalize_field_pair(&pos_a, &pos_b, config.normalize);
    let (oa, ob) = normalize_field_pair(&opa_a, &opa_b, config.normalize);
    let (sa, sb) = normalize_field_pair(&sca_a, &sca_b, config.normalize);
    let (ca, cb) = normalize_field_pair(&col_a, &col_b, config.normalize);

    let position_diff = compute_field_diff(&pa, &pb, "position", eps)?;
    let opacity_diff = compute_field_diff(&oa, &ob, "opacity", eps)?;
    let scale_diff = compute_field_diff(&sa, &sb, "scale", eps)?;
    let color_diff = compute_field_diff(&ca, &cb, "color", eps)?;

    // Summary score: map combined RMS to [0, 1] via 1 - exp(-mean_rms).
    let combined_rms = (position_diff.rms_change
        + opacity_diff.rms_change
        + scale_diff.rms_change
        + color_diff.rms_change)
        / 4.0;
    let summary_score = (1.0 - (-combined_rms).exp()).clamp(0.0, 1.0);

    Ok(ModelDiff {
        name_a: a.name.clone(),
        name_b: b.name.clone(),
        step_a: a.step,
        step_b: b.step,
        n_gaussians: n,
        n_compared,
        position_diff,
        opacity_diff,
        scale_diff,
        color_diff,
        // Simplified: same size models have 0 added/removed.
        added_gaussians: 0,
        removed_gaussians: 0,
        summary_score,
    })
}

/// Gather `data[i * stride .. i * stride + stride]` for each `i` in
/// `indices`, or borrow `data` outright when `indices` covers every
/// Gaussian in it (the common case: `DiffConfig::include_inactive` is
/// `true` by default, or an explicit filter happens to keep everything).
/// `indices` is assumed sorted and free of duplicates (true for every
/// caller in this module, which all build it via `(0..n).filter(..)`), so
/// `indices.len() * stride == data.len()` is a safe proxy for "this is
/// really the identity range and a copy is unnecessary".
fn gather_field<'a>(data: &'a [f32], indices: &[usize], stride: usize) -> Cow<'a, [f32]> {
    if indices.len().saturating_mul(stride) == data.len() {
        return Cow::Borrowed(data);
    }
    let mut out = Vec::with_capacity(indices.len() * stride);
    for &i in indices {
        for s in 0..stride {
            out.push(data[i * stride + s]);
        }
    }
    Cow::Owned(out)
}

/// Normalise a pair of fields by the mean magnitude of `fa`, or borrow both
/// unchanged when `normalize` is `false` (the default) rather than always
/// copying, even when there is nothing to scale.
fn normalize_field_pair<'a>(
    fa: &'a [f32],
    fb: &'a [f32],
    normalize: bool,
) -> (Cow<'a, [f32]>, Cow<'a, [f32]>) {
    if !normalize {
        return (Cow::Borrowed(fa), Cow::Borrowed(fb));
    }
    let mean_mag_a = fa.iter().map(|x| x.abs()).sum::<f32>() / fa.len().max(1) as f32;
    if mean_mag_a == 0.0 {
        return (Cow::Borrowed(fa), Cow::Borrowed(fb));
    }
    let scale = 1.0 / mean_mag_a;
    (
        Cow::Owned(fa.iter().map(|x| x * scale).collect()),
        Cow::Owned(fb.iter().map(|x| x * scale).collect()),
    )
}

// ---------------------------------------------------------------------------
// grid_key helper (spatial hashing)
// ---------------------------------------------------------------------------

/// Convert a world-space position to an integer grid cell key given a cell size.
///
/// Uses floor division so that negative coordinates are handled correctly.
#[inline]
fn grid_key(pos: [f32; 3], cell_size: f32) -> (i32, i32, i32) {
    (
        (pos[0] / cell_size).floor() as i32,
        (pos[1] / cell_size).floor() as i32,
        (pos[2] / cell_size).floor() as i32,
    )
}

// ---------------------------------------------------------------------------
// diff_models_variable
// ---------------------------------------------------------------------------

/// Compute a [`ModelDiff`] between two model snapshots that may have different
/// numbers of Gaussians using spatial grid-hash nearest-neighbour matching.
///
/// Unlike [`diff_models`] (which is index-based and requires equal sizes),
/// this function builds a spatial hash of model A's Gaussians and then, for
/// every B Gaussian, searches the 27-cell neighbourhood to find the closest
/// A Gaussian within `config.match_radius`.
///
/// - Matched pairs contribute to the field-diff statistics.
/// - B Gaussians without a nearby A match are counted as **added**.
/// - A Gaussians that were never matched by any B Gaussian are counted as
///   **removed**.
///
/// `n_gaussians` in the returned [`ModelDiff`] is set to the number of
/// successfully matched pairs.
///
/// # Errors
/// - [`DiffError::EmptyModelA`] if model A is empty.
/// - [`DiffError::EmptyModelB`] if model B is empty.
/// - [`DiffError::InvalidConfig`] if the config fails validation.
pub fn diff_models_variable(
    a: &ModelSnapshot,
    b: &ModelSnapshot,
    config: &DiffConfig,
) -> Result<ModelDiff, DiffError> {
    config.validate()?;

    if a.n_gaussians == 0 {
        return Err(DiffError::EmptyModelA);
    }
    if b.n_gaussians == 0 {
        return Err(DiffError::EmptyModelB);
    }

    let cell_size = config.match_radius;
    let radius_sq = config.match_radius * config.match_radius;

    // ------------------------------------------------------------------
    // Build spatial grid from model A.
    // ------------------------------------------------------------------
    // Key: (gx, gy, gz) integer cell coordinates.
    // Value: list of A-Gaussian indices whose position hashes to that cell.
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::with_capacity(a.n_gaussians);

    for idx in 0..a.n_gaussians {
        let pos = [
            a.positions[idx * 3],
            a.positions[idx * 3 + 1],
            a.positions[idx * 3 + 2],
        ];
        let key = grid_key(pos, cell_size);
        grid.entry(key).or_default().push(idx);
    }

    // ------------------------------------------------------------------
    // Match B Gaussians to A Gaussians.
    // ------------------------------------------------------------------
    // matched_a[a_idx] = true once that A Gaussian has been claimed.
    let mut matched_a = vec![false; a.n_gaussians];
    // For each B Gaussian, the index of its A match (or None).
    let mut b_to_a: Vec<Option<usize>> = vec![None; b.n_gaussians];

    for (b_idx, slot) in b_to_a.iter_mut().enumerate() {
        let bx = b.positions[b_idx * 3];
        let by = b.positions[b_idx * 3 + 1];
        let bz = b.positions[b_idx * 3 + 2];

        let (cx, cy, cz) = grid_key([bx, by, bz], cell_size);

        let mut best_dist_sq = f32::INFINITY;
        let mut best_a_idx: Option<usize> = None;

        // Search all 27 neighbouring cells (±1 in each axis).
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                for dz in -1i32..=1 {
                    let cell = (cx + dx, cy + dy, cz + dz);
                    if let Some(candidates) = grid.get(&cell) {
                        for &a_idx in candidates {
                            if matched_a[a_idx] {
                                continue; // already claimed
                            }
                            let ax = a.positions[a_idx * 3];
                            let ay = a.positions[a_idx * 3 + 1];
                            let az = a.positions[a_idx * 3 + 2];
                            let dist_sq = (bx - ax) * (bx - ax)
                                + (by - ay) * (by - ay)
                                + (bz - az) * (bz - az);
                            if dist_sq < radius_sq && dist_sq < best_dist_sq {
                                best_dist_sq = dist_sq;
                                best_a_idx = Some(a_idx);
                            }
                        }
                    }
                }
            }
        }

        if let Some(a_idx) = best_a_idx {
            matched_a[a_idx] = true;
            *slot = Some(a_idx);
        }
    }

    // ------------------------------------------------------------------
    // Count added / removed.
    // ------------------------------------------------------------------
    let added_gaussians: usize = b_to_a.iter().filter(|m| m.is_none()).count();
    let removed_gaussians: usize = matched_a.iter().filter(|&&m| !m).count();
    let matched_count: usize = b_to_a.iter().filter(|m| m.is_some()).count();

    // ------------------------------------------------------------------
    // Collect matched-pair field data for statistics.
    // ------------------------------------------------------------------
    // Collect flat arrays from the matched pairs only.
    let mut pos_a_flat: Vec<f32> = Vec::with_capacity(matched_count * 3);
    let mut pos_b_flat: Vec<f32> = Vec::with_capacity(matched_count * 3);
    let mut opa_a_flat: Vec<f32> = Vec::with_capacity(matched_count);
    let mut opa_b_flat: Vec<f32> = Vec::with_capacity(matched_count);
    let mut sca_a_flat: Vec<f32> = Vec::with_capacity(matched_count * 3);
    let mut sca_b_flat: Vec<f32> = Vec::with_capacity(matched_count * 3);
    let mut col_a_flat: Vec<f32> = Vec::with_capacity(matched_count * 3);
    let mut col_b_flat: Vec<f32> = Vec::with_capacity(matched_count * 3);

    for (b_idx, match_opt) in b_to_a.iter().enumerate() {
        if let Some(a_idx) = *match_opt {
            for s in 0..3 {
                pos_a_flat.push(a.positions[a_idx * 3 + s]);
                pos_b_flat.push(b.positions[b_idx * 3 + s]);
                sca_a_flat.push(a.scales[a_idx * 3 + s]);
                sca_b_flat.push(b.scales[b_idx * 3 + s]);
                col_a_flat.push(a.colors[a_idx * 3 + s]);
                col_b_flat.push(b.colors[b_idx * 3 + s]);
            }
            opa_a_flat.push(a.opacities[a_idx]);
            opa_b_flat.push(b.opacities[b_idx]);
        }
    }

    // ------------------------------------------------------------------
    // Optional normalisation.
    // ------------------------------------------------------------------
    let (pa, pb) = normalize_field_pair(&pos_a_flat, &pos_b_flat, config.normalize);
    let (oa, ob) = normalize_field_pair(&opa_a_flat, &opa_b_flat, config.normalize);
    let (sa, sb) = normalize_field_pair(&sca_a_flat, &sca_b_flat, config.normalize);
    let (ca, cb) = normalize_field_pair(&col_a_flat, &col_b_flat, config.normalize);

    let eps = config.epsilon;
    let position_diff = compute_field_diff(&pa, &pb, "position", eps)?;
    let opacity_diff = compute_field_diff(&oa, &ob, "opacity", eps)?;
    let scale_diff = compute_field_diff(&sa, &sb, "scale", eps)?;
    let color_diff = compute_field_diff(&ca, &cb, "color", eps)?;

    // ------------------------------------------------------------------
    // Summary score from matched-pair statistics.
    // ------------------------------------------------------------------
    let combined_rms = (position_diff.rms_change
        + opacity_diff.rms_change
        + scale_diff.rms_change
        + color_diff.rms_change)
        / 4.0;
    let summary_score = (1.0 - (-combined_rms).exp()).clamp(0.0, 1.0);

    Ok(ModelDiff {
        name_a: a.name.clone(),
        name_b: b.name.clone(),
        step_a: a.step,
        step_b: b.step,
        n_gaussians: matched_count,
        n_compared: matched_count,
        position_diff,
        opacity_diff,
        scale_diff,
        color_diff,
        added_gaussians,
        removed_gaussians,
        summary_score,
    })
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/// Format a [`ModelDiff`] as a human-readable multi-line text block.
pub fn format_model_diff(diff: &ModelDiff) -> String {
    let step_delta = diff.step_b.saturating_sub(diff.step_a);
    let mut s = String::new();
    s.push_str(&format!(
        "=== Model Diff: '{}' (step {}) → '{}' (step {}) | Δsteps={} ===\n",
        diff.name_a, diff.step_a, diff.name_b, diff.step_b, step_delta
    ));
    s.push_str(&format!(
        "  Gaussians : {} | Compared: {} | Added: {} | Removed: {}\n",
        diff.n_gaussians, diff.n_compared, diff.added_gaussians, diff.removed_gaussians
    ));
    if diff.n_compared == 0 {
        s.push_str(
            "  WARNING: 0 Gaussians were compared -- the statistics below are vacuous, \
             not evidence the models are identical.\n",
        );
    }
    s.push_str(&format!(
        "  Summary score : {:.6} (0=identical, 1=completely different)\n",
        diff.summary_score
    ));
    s.push('\n');
    s.push_str(&format!(
        "  {:<10} {}\n",
        "Field",
        format_field_diff_header()
    ));
    s.push_str(&format!(
        "  {:<10} {}\n",
        "position",
        format_field_diff(&diff.position_diff)
    ));
    s.push_str(&format!(
        "  {:<10} {}\n",
        "opacity",
        format_field_diff(&diff.opacity_diff)
    ));
    s.push_str(&format!(
        "  {:<10} {}\n",
        "scale",
        format_field_diff(&diff.scale_diff)
    ));
    s.push_str(&format!(
        "  {:<10} {}\n",
        "color",
        format_field_diff(&diff.color_diff)
    ));
    s
}

/// Return a header row matching the columns produced by [`format_field_diff`].
pub fn format_field_diff_header() -> String {
    format!(
        "{:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "mean_chg", "std_chg", "max_abs", "rms", "frac_chg", "cos_sim"
    )
}

/// Format a [`FieldDiff`] as a compact table row.
pub fn format_field_diff(diff: &FieldDiff) -> String {
    format!(
        "{:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
        diff.mean_change,
        diff.std_change,
        diff.max_abs_change,
        diff.rms_change,
        diff.fraction_changed,
        diff.cosine_similarity
    )
}

// ---------------------------------------------------------------------------
// largest_position_changes
// ---------------------------------------------------------------------------

/// Find the top-`k` Gaussians with the largest position change between A and B.
///
/// Returns a `Vec<(gaussian_idx, change_magnitude)>` sorted descending by magnitude.
/// If `k > n_gaussians`, all Gaussians are returned.
///
/// # Errors
/// - [`DiffError::EmptyModelA`] / [`DiffError::EmptyModelB`] if either is empty.
/// - [`DiffError::SizeMismatch`] if Gaussian counts differ.
pub fn largest_position_changes(
    a: &ModelSnapshot,
    b: &ModelSnapshot,
    k: usize,
) -> Result<Vec<(usize, f32)>, DiffError> {
    if a.n_gaussians == 0 {
        return Err(DiffError::EmptyModelA);
    }
    if b.n_gaussians == 0 {
        return Err(DiffError::EmptyModelB);
    }
    if a.n_gaussians != b.n_gaussians {
        return Err(DiffError::SizeMismatch {
            a: a.n_gaussians,
            b: b.n_gaussians,
        });
    }

    let n = a.n_gaussians;
    let mut magnitudes: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let dx = b.positions[i * 3] - a.positions[i * 3];
            let dy = b.positions[i * 3 + 1] - a.positions[i * 3 + 1];
            let dz = b.positions[i * 3 + 2] - a.positions[i * 3 + 2];
            let mag = (dx * dx + dy * dy + dz * dz).sqrt();
            (i, mag)
        })
        .collect();

    // Sort descending by magnitude.
    magnitudes.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
    magnitudes.truncate(k.min(n));
    Ok(magnitudes)
}

// ---------------------------------------------------------------------------
// opacity_changes
// ---------------------------------------------------------------------------

/// Find Gaussians that crossed the activation `threshold` between A and B.
///
/// Returns `(newly_active, newly_inactive)`:
/// - `newly_active`: indices where A's activated opacity < threshold and B's >= threshold.
/// - `newly_inactive`: indices where A's activated opacity >= threshold and B's < threshold.
///
/// # Errors
/// - [`DiffError::EmptyModelA`] / [`DiffError::EmptyModelB`] if either is empty.
/// - [`DiffError::SizeMismatch`] if Gaussian counts differ.
pub fn opacity_changes(
    a: &ModelSnapshot,
    b: &ModelSnapshot,
    threshold: f32,
) -> Result<(Vec<usize>, Vec<usize>), DiffError> {
    if a.n_gaussians == 0 {
        return Err(DiffError::EmptyModelA);
    }
    if b.n_gaussians == 0 {
        return Err(DiffError::EmptyModelB);
    }
    if a.n_gaussians != b.n_gaussians {
        return Err(DiffError::SizeMismatch {
            a: a.n_gaussians,
            b: b.n_gaussians,
        });
    }

    let mut newly_active = Vec::new();
    let mut newly_inactive = Vec::new();

    for i in 0..a.n_gaussians {
        let opa_a = a.activated_opacity(i);
        let opa_b = b.activated_opacity(i);
        if opa_a < threshold && opa_b >= threshold {
            newly_active.push(i);
        } else if opa_a >= threshold && opa_b < threshold {
            newly_inactive.push(i);
        }
    }

    Ok((newly_active, newly_inactive))
}

// ---------------------------------------------------------------------------
// per_gaussian_change_score
// ---------------------------------------------------------------------------

/// Compute a per-Gaussian change score as a weighted sum of positional,
/// opacity, and scale changes.
///
/// Weights: position (0.5) + opacity (0.25) + max_scale (0.25).
///
/// Returns a `Vec<f32>` of length `n_gaussians`.
///
/// # Errors
/// - [`DiffError::EmptyModelA`] / [`DiffError::EmptyModelB`] if either is empty.
/// - [`DiffError::SizeMismatch`] if Gaussian counts differ.
pub fn per_gaussian_change_score(
    a: &ModelSnapshot,
    b: &ModelSnapshot,
) -> Result<Vec<f32>, DiffError> {
    if a.n_gaussians == 0 {
        return Err(DiffError::EmptyModelA);
    }
    if b.n_gaussians == 0 {
        return Err(DiffError::EmptyModelB);
    }
    if a.n_gaussians != b.n_gaussians {
        return Err(DiffError::SizeMismatch {
            a: a.n_gaussians,
            b: b.n_gaussians,
        });
    }

    let n = a.n_gaussians;
    let mut scores = Vec::with_capacity(n);

    for i in 0..n {
        // Position change magnitude.
        let dx = b.positions[i * 3] - a.positions[i * 3];
        let dy = b.positions[i * 3 + 1] - a.positions[i * 3 + 1];
        let dz = b.positions[i * 3 + 2] - a.positions[i * 3 + 2];
        let pos_mag = (dx * dx + dy * dy + dz * dz).sqrt();

        // Opacity change magnitude (activated).
        let opa_delta = (b.activated_opacity(i) - a.activated_opacity(i)).abs();

        // Max scale change (activated).
        let max_scale_delta = (0..3usize)
            .map(|ax| (b.activated_scale(i, ax) - a.activated_scale(i, ax)).abs())
            .fold(0.0f32, f32::max);

        let score = 0.5 * pos_mag + 0.25 * opa_delta + 0.25 * max_scale_delta;
        scores.push(score);
    }

    Ok(scores)
}

// ---------------------------------------------------------------------------
// change_score_histogram
// ---------------------------------------------------------------------------

/// Compute a histogram of per-Gaussian change scores.
///
/// Returns `(bin_edges, counts)`:
/// - `bin_edges`: length `bins + 1`, evenly spaced from `min` to `max` of `scores`.
/// - `counts`: length `bins`, number of scores falling in each bin.
///
/// # Errors
/// - [`DiffError::InvalidConfig`] if `bins == 0` or `scores` is empty.
pub fn change_score_histogram(
    scores: &[f32],
    bins: usize,
) -> Result<(Vec<f32>, Vec<usize>), DiffError> {
    if bins == 0 {
        return Err(DiffError::InvalidConfig("bins must be > 0".into()));
    }
    if scores.is_empty() {
        return Err(DiffError::InvalidConfig("scores slice is empty".into()));
    }

    let min_val = scores.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Build evenly-spaced edges.
    let range = max_val - min_val;
    let mut bin_edges = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        bin_edges.push(min_val + range * i as f32 / bins as f32);
    }

    let mut counts = vec![0usize; bins];

    if range == 0.0 {
        // All values identical; every score falls in bin 0.
        counts[0] = scores.len();
        return Ok((bin_edges, counts));
    }

    for &s in scores {
        let idx = ((s - min_val) / range * bins as f32) as usize;
        // Clamp: the maximum value lands exactly on the last edge.
        let idx = idx.min(bins - 1);
        counts[idx] += 1;
    }

    Ok((bin_edges, counts))
}

// ---------------------------------------------------------------------------
// snapshots_approximately_equal
// ---------------------------------------------------------------------------

/// Check whether two snapshots are approximately equal using NumPy conventions:
/// `|a_i - b_i| <= atol + rtol * |b_i|` for all elements of all fields.
///
/// # Errors
/// - [`DiffError::EmptyModelA`] / [`DiffError::EmptyModelB`] if either is empty.
/// - [`DiffError::SizeMismatch`] if Gaussian counts differ.
pub fn snapshots_approximately_equal(
    a: &ModelSnapshot,
    b: &ModelSnapshot,
    rtol: f32,
    atol: f32,
) -> Result<bool, DiffError> {
    if a.n_gaussians == 0 {
        return Err(DiffError::EmptyModelA);
    }
    if b.n_gaussians == 0 {
        return Err(DiffError::EmptyModelB);
    }
    if a.n_gaussians != b.n_gaussians {
        return Err(DiffError::SizeMismatch {
            a: a.n_gaussians,
            b: b.n_gaussians,
        });
    }

    let all_close = |fa: &[f32], fb: &[f32]| -> bool {
        fa.iter()
            .zip(fb.iter())
            .all(|(ai, bi)| (ai - bi).abs() <= atol + rtol * bi.abs())
    };

    Ok(all_close(&a.positions, &b.positions)
        && all_close(&a.opacities, &b.opacities)
        && all_close(&a.scales, &b.scales)
        && all_close(&a.colors, &b.colors))
}

// ---------------------------------------------------------------------------
// diff_sequence
// ---------------------------------------------------------------------------

/// Compute sequential diffs across a slice of snapshots.
///
/// Returns a `Vec<ModelDiff>` of length `snapshots.len() - 1`.
///
/// # Errors
/// - [`DiffError::InvalidConfig`] if `snapshots` has fewer than 2 entries.
/// - Propagates errors from [`diff_models`].
pub fn diff_sequence(
    snapshots: &[ModelSnapshot],
    config: &DiffConfig,
) -> Result<Vec<ModelDiff>, DiffError> {
    if snapshots.len() < 2 {
        return Err(DiffError::InvalidConfig(format!(
            "diff_sequence requires at least 2 snapshots, got {}",
            snapshots.len()
        )));
    }

    let mut diffs = Vec::with_capacity(snapshots.len() - 1);
    for pair in snapshots.windows(2) {
        let d = diff_models(&pair[0], &pair[1], config)?;
        diffs.push(d);
    }
    Ok(diffs)
}

// ---------------------------------------------------------------------------
// RegressionReport + detect_regression
// ---------------------------------------------------------------------------

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

/// Detect regressions by comparing diff metrics against given thresholds.
///
/// - `opacity_threshold`: maximum allowed opacity *decrease* (default 0.05).
/// - `scale_threshold`: maximum allowed mean scale *increase* (default 0.1).
/// - `position_threshold`: maximum allowed RMS position change (default 0.01).
pub fn detect_regression(
    diff: &ModelDiff,
    opacity_threshold: f32,
    scale_threshold: f32,
    position_threshold: f32,
) -> RegressionReport {
    let mut details = Vec::new();

    // Opacity regression: mean_change is negative and its magnitude exceeds threshold.
    let opacity_regressed = diff.opacity_diff.mean_change < -opacity_threshold.abs();
    if opacity_regressed {
        details.push(format!(
            "Opacity regressed: mean change {:.6} < threshold -{:.6}",
            diff.opacity_diff.mean_change,
            opacity_threshold.abs()
        ));
    }

    // Scale regression: mean_change positive and exceeds threshold.
    let scale_regressed = diff.scale_diff.mean_change > scale_threshold.abs();
    if scale_regressed {
        details.push(format!(
            "Scale regressed: mean change {:.6} > threshold {:.6}",
            diff.scale_diff.mean_change,
            scale_threshold.abs()
        ));
    }

    // Position unstable: RMS exceeds threshold.
    let position_unstable = diff.position_diff.rms_change > position_threshold.abs();
    if position_unstable {
        details.push(format!(
            "Position unstable: RMS {:.6} > threshold {:.6}",
            diff.position_diff.rms_change,
            position_threshold.abs()
        ));
    }

    let overall_regression = opacity_regressed || scale_regressed || position_unstable;

    RegressionReport {
        opacity_regressed,
        scale_regressed,
        position_unstable,
        overall_regression,
        details,
    }
}

// ---------------------------------------------------------------------------
// ProgressSummary + summarize_progress
// ---------------------------------------------------------------------------

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

/// Analyse a sequence of diffs for training-progress trends.
///
/// # Errors
/// - [`DiffError::InvalidConfig`] if `diffs` is empty.
pub fn summarize_progress(
    diffs: &[ModelDiff],
    stall_threshold: f32,
) -> Result<ProgressSummary, DiffError> {
    if diffs.is_empty() {
        return Err(DiffError::InvalidConfig(
            "summarize_progress requires at least 1 diff".into(),
        ));
    }

    let n_steps = diffs.len();

    let total_steps = {
        let first = &diffs[0];
        let last = &diffs[diffs.len() - 1];
        last.step_b.saturating_sub(first.step_a)
    };

    // Mean change per training step.
    let scores: Vec<f32> = diffs.iter().map(|d| d.summary_score).collect();
    let mean_score = mean_f32(&scores);
    let mean_step_delta = if n_steps > 0 {
        let deltas: Vec<f32> = diffs
            .iter()
            .map(|d| {
                if d.step_b >= d.step_a {
                    (d.step_b - d.step_a) as f32
                } else {
                    1.0
                }
            })
            .collect();
        mean_f32(&deltas)
    } else {
        1.0
    };
    let mean_change_per_step = if mean_step_delta > 0.0 {
        mean_score / mean_step_delta
    } else {
        mean_score
    };

    // Converging: check that scores are on a downward trend (simple linear regression slope).
    let converging = if n_steps >= 2 {
        let xs: Vec<f32> = (0..n_steps).map(|i| i as f32).collect();
        let ys = &scores;
        let mean_x = mean_f32(&xs);
        let mean_y = mean_f32(ys);
        let numer: f32 = xs
            .iter()
            .zip(ys.iter())
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum();
        let denom: f32 = xs.iter().map(|x| (x - mean_x) * (x - mean_x)).sum();
        if denom > 0.0 {
            numer / denom < 0.0 // negative slope → converging
        } else {
            false
        }
    } else {
        // Single diff cannot determine trend.
        false
    };

    // Stalled: look at the last min(3, n_steps) diffs.
    let stall_window = 3.min(n_steps);
    let last_scores = &scores[n_steps - stall_window..];
    let stalled = last_scores.iter().all(|&s| s < stall_threshold);

    // Count regressions with default thresholds.
    let regression_count = diffs
        .iter()
        .filter(|d| detect_regression(d, 0.05, 0.1, 0.01).overall_regression)
        .count();

    Ok(ProgressSummary {
        n_steps,
        total_steps,
        mean_change_per_step,
        converging,
        stalled,
        regression_count,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn make_snapshot(name: &str, step: usize, n: usize, fill: f32) -> ModelSnapshot {
        ModelSnapshot::new(
            name,
            step,
            vec![fill; n * 3],
            vec![fill; n],
            vec![fill; n * 3],
            vec![fill; n * 3],
        )
        .expect("valid snapshot")
    }

    fn make_snapshot_vals(
        name: &str,
        step: usize,
        positions: Vec<f32>,
        opacities: Vec<f32>,
        scales: Vec<f32>,
        colors: Vec<f32>,
    ) -> ModelSnapshot {
        ModelSnapshot::new(name, step, positions, opacities, scales, colors)
            .expect("valid snapshot")
    }

    // ------------------------------------------------------------------
    // ModelSnapshot::new
    // ------------------------------------------------------------------

    #[test]
    fn test_snapshot_new_valid() {
        let s = ModelSnapshot::new(
            "a",
            0,
            vec![0.0; 6],
            vec![0.0; 2],
            vec![0.0; 6],
            vec![0.0; 6],
        );
        assert!(s.is_ok());
        let s = s.expect("ok");
        assert_eq!(s.n_gaussians, 2);
    }

    #[test]
    fn test_snapshot_new_positions_not_divisible_by_3() {
        let r = ModelSnapshot::new(
            "a",
            0,
            vec![0.0; 5],
            vec![0.0; 1],
            vec![0.0; 3],
            vec![0.0; 3],
        );
        assert!(matches!(r, Err(DiffError::DimensionError(_))));
    }

    #[test]
    fn test_snapshot_new_opacities_mismatch() {
        let r = ModelSnapshot::new(
            "a",
            0,
            vec![0.0; 6],
            vec![0.0; 3],
            vec![0.0; 6],
            vec![0.0; 6],
        );
        assert!(matches!(r, Err(DiffError::DimensionError(_))));
    }

    #[test]
    fn test_snapshot_new_scales_mismatch() {
        let r = ModelSnapshot::new(
            "a",
            0,
            vec![0.0; 6],
            vec![0.0; 2],
            vec![0.0; 5],
            vec![0.0; 6],
        );
        assert!(matches!(r, Err(DiffError::DimensionError(_))));
    }

    #[test]
    fn test_snapshot_new_colors_mismatch() {
        let r = ModelSnapshot::new(
            "a",
            0,
            vec![0.0; 6],
            vec![0.0; 2],
            vec![0.0; 6],
            vec![0.0; 5],
        );
        assert!(matches!(r, Err(DiffError::DimensionError(_))));
    }

    // ------------------------------------------------------------------
    // ModelSnapshot::activated_opacity
    // ------------------------------------------------------------------

    #[test]
    fn test_activated_opacity_zero_logit() {
        let s = make_snapshot("a", 0, 2, 0.0);
        // sigmoid(0) = 0.5
        let opa = s.activated_opacity(0);
        assert!((opa - 0.5).abs() < 1e-6, "expected 0.5, got {}", opa);
    }

    #[test]
    fn test_activated_opacity_large_positive() {
        let s = make_snapshot_vals(
            "a",
            0,
            vec![0.0; 3],
            vec![100.0],
            vec![0.0; 3],
            vec![0.0; 3],
        );
        // sigmoid(100) ≈ 1.0
        assert!(s.activated_opacity(0) > 0.999);
    }

    #[test]
    fn test_activated_opacity_large_negative() {
        let s = make_snapshot_vals(
            "a",
            0,
            vec![0.0; 3],
            vec![-100.0],
            vec![0.0; 3],
            vec![0.0; 3],
        );
        // sigmoid(-100) ≈ 0.0
        assert!(s.activated_opacity(0) < 1e-3);
    }

    // ------------------------------------------------------------------
    // ModelSnapshot::activated_scale
    // ------------------------------------------------------------------

    #[test]
    fn test_activated_scale_zero_log() {
        let s = make_snapshot("a", 0, 2, 0.0);
        // exp(0) = 1.0
        let sc = s.activated_scale(0, 0);
        assert!((sc - 1.0).abs() < 1e-6, "expected 1.0, got {}", sc);
    }

    #[test]
    fn test_activated_scale_log_one() {
        let s = make_snapshot_vals(
            "a",
            0,
            vec![0.0; 3],
            vec![0.0],
            vec![1.0, 2.0, 3.0],
            vec![0.0; 3],
        );
        assert!((s.activated_scale(0, 0) - 1.0f32.exp()).abs() < 1e-5);
        assert!((s.activated_scale(0, 1) - 2.0f32.exp()).abs() < 1e-5);
        assert!((s.activated_scale(0, 2) - 3.0f32.exp()).abs() < 1e-5);
    }

    // ------------------------------------------------------------------
    // compute_field_diff
    // ------------------------------------------------------------------

    #[test]
    fn test_field_diff_identical() {
        let a = vec![1.0f32, 2.0, 3.0];
        let d = compute_field_diff(&a, &a, "test", 1e-6).expect("ok");
        assert!((d.mean_change).abs() < 1e-7);
        assert!((d.l2_distance).abs() < 1e-7);
        assert!((d.cosine_similarity - 1.0).abs() < 1e-6);
        assert_eq!(d.fraction_changed, 0.0);
    }

    #[test]
    fn test_field_diff_known_difference() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 1.0, 1.0];
        let d = compute_field_diff(&a, &b, "test", 1e-6).expect("ok");
        assert!((d.mean_change - 1.0).abs() < 1e-6);
        assert!((d.rms_change - 1.0).abs() < 1e-6);
        assert!((d.l2_distance - 3.0f32.sqrt()).abs() < 1e-6);
        assert_eq!(d.fraction_changed, 1.0);
    }

    #[test]
    fn test_field_diff_both_zero_vectors_cosine() {
        let a = vec![0.0f32; 4];
        let d = compute_field_diff(&a, &a, "zeros", 1e-6).expect("ok");
        assert!((d.cosine_similarity - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_field_diff_one_zero_vector_cosine() {
        let a = vec![0.0f32; 4];
        let b = vec![1.0f32; 4];
        let d = compute_field_diff(&a, &b, "t", 1e-6).expect("ok");
        assert!((d.cosine_similarity).abs() < 1e-7);
    }

    #[test]
    fn test_field_diff_length_mismatch() {
        let a = vec![1.0f32; 3];
        let b = vec![1.0f32; 4];
        assert!(matches!(
            compute_field_diff(&a, &b, "x", 1e-6),
            Err(DiffError::DimensionError(_))
        ));
    }

    // ------------------------------------------------------------------
    // diff_models
    // ------------------------------------------------------------------

    #[test]
    fn test_diff_models_identical() {
        let a = make_snapshot("a", 0, 10, 0.5);
        let config = DiffConfig::default();
        let d = diff_models(&a, &a, &config).expect("ok");
        assert!((d.position_diff.rms_change).abs() < 1e-7);
        assert!((d.opacity_diff.rms_change).abs() < 1e-7);
        assert!((d.scale_diff.rms_change).abs() < 1e-7);
        assert!((d.color_diff.rms_change).abs() < 1e-7);
        assert!(d.summary_score < 1e-6);
    }

    #[test]
    fn test_diff_models_size_mismatch() {
        let a = make_snapshot("a", 0, 5, 0.0);
        let b = make_snapshot("b", 1, 7, 0.0);
        let config = DiffConfig::default();
        assert!(matches!(
            diff_models(&a, &b, &config),
            Err(DiffError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn test_diff_models_empty_a() {
        let a = ModelSnapshot {
            name: "a".into(),
            step: 0,
            n_gaussians: 0,
            positions: vec![],
            opacities: vec![],
            scales: vec![],
            colors: vec![],
        };
        let b = make_snapshot("b", 1, 3, 0.0);
        let config = DiffConfig::default();
        assert!(matches!(
            diff_models(&a, &b, &config),
            Err(DiffError::EmptyModelA)
        ));
    }

    #[test]
    fn test_diff_models_empty_b() {
        let a = make_snapshot("a", 0, 3, 0.0);
        let b = ModelSnapshot {
            name: "b".into(),
            step: 1,
            n_gaussians: 0,
            positions: vec![],
            opacities: vec![],
            scales: vec![],
            colors: vec![],
        };
        let config = DiffConfig::default();
        assert!(matches!(
            diff_models(&a, &b, &config),
            Err(DiffError::EmptyModelB)
        ));
    }

    #[test]
    fn test_diff_models_n_compared_matches_default_include_inactive() {
        // include_inactive: true (the default) -- every Gaussian counts as
        // compared, matching n_gaussians.
        let a = make_snapshot("a", 0, 10, 0.5);
        let config = DiffConfig::default();
        let d = diff_models(&a, &a, &config).expect("ok");
        assert_eq!(d.n_gaussians, 10);
        assert_eq!(d.n_compared, 10);
    }

    // Regression coverage for: two models where every Gaussian is inactive
    // (activated opacity < 0.1) used to report `cosine_similarity: 1.0` and
    // all-zero change statistics -- indistinguishable from "these models
    // are identical" -- with nothing indicating that zero Gaussians were
    // actually compared.
    #[test]
    fn test_diff_models_all_inactive_reports_zero_compared_not_identical() {
        let n = 5;
        // sigmoid(-10) ~= 4.5e-5, well under the 0.1 activation threshold:
        // every Gaussian in both snapshots is "inactive".
        let very_negative = vec![-10.0_f32; n];
        let a = make_snapshot_vals(
            "a",
            0,
            vec![0.0; n * 3],
            very_negative.clone(),
            vec![0.0; n * 3],
            vec![0.0; n * 3],
        );
        // Genuinely different positions from `a`, so an *honest* comparison
        // (if these Gaussians had been included) would not be "identical".
        let b = make_snapshot_vals(
            "b",
            1,
            vec![5.0; n * 3],
            very_negative,
            vec![0.0; n * 3],
            vec![0.0; n * 3],
        );
        let config = DiffConfig {
            include_inactive: false,
            ..DiffConfig::default()
        };
        let diff = diff_models(&a, &b, &config).expect("diff ok");

        assert_eq!(
            diff.n_gaussians, n,
            "total Gaussian count is still reported"
        );
        assert_eq!(
            diff.n_compared, 0,
            "no Gaussian passed the activation threshold, so none were compared"
        );

        let text = format_model_diff(&diff);
        assert!(
            text.contains("WARNING"),
            "output must warn that the comparison is vacuous: {text}"
        );
        assert!(
            text.contains("Compared: 0"),
            "output must surface n_compared: {text}"
        );
    }

    #[test]
    fn test_diff_models_n_compared_reflects_partial_activity() {
        let n = 4;
        // Gaussians 0,1 active (opacity logit 5.0 -> sigmoid ~0.993);
        // Gaussians 2,3 inactive (opacity logit -10.0 -> sigmoid ~4.5e-5).
        let opacities = vec![5.0, 5.0, -10.0, -10.0];
        let a = make_snapshot_vals(
            "a",
            0,
            vec![0.0; n * 3],
            opacities.clone(),
            vec![0.0; n * 3],
            vec![0.0; n * 3],
        );
        let b = make_snapshot_vals(
            "b",
            1,
            vec![1.0; n * 3],
            opacities,
            vec![0.0; n * 3],
            vec![0.0; n * 3],
        );
        let config = DiffConfig {
            include_inactive: false,
            ..DiffConfig::default()
        };
        let diff = diff_models(&a, &b, &config).expect("diff ok");
        assert_eq!(diff.n_gaussians, 4);
        assert_eq!(
            diff.n_compared, 2,
            "only the 2 active Gaussians should be compared"
        );
        // The 2 active Gaussians moved by 1.0 on every axis, so this must
        // not report "no change" the way an all-inactive comparison would.
        assert!(diff.position_diff.rms_change > 0.5);
    }

    // ------------------------------------------------------------------
    // format_model_diff
    // ------------------------------------------------------------------

    #[test]
    fn test_format_model_diff_contains_step_info() {
        let a = make_snapshot("snap_100", 100, 5, 0.0);
        let b = make_snapshot("snap_200", 200, 5, 1.0);
        let config = DiffConfig::default();
        let diff = diff_models(&a, &b, &config).expect("ok");
        let text = format_model_diff(&diff);
        assert!(text.contains("100"), "should contain step_a");
        assert!(text.contains("200"), "should contain step_b");
        assert!(text.contains("snap_100"));
        assert!(text.contains("snap_200"));
    }

    // ------------------------------------------------------------------
    // largest_position_changes
    // ------------------------------------------------------------------

    #[test]
    fn test_largest_position_changes_k1() {
        // Gaussian 1 moves 10 units, others stay.
        let mut pos_b = vec![0.0f32; 9]; // 3 Gaussians
        pos_b[3] = 10.0; // Gaussian 1 moves on x.
        let a = make_snapshot_vals(
            "a",
            0,
            vec![0.0; 9],
            vec![0.0; 3],
            vec![0.0; 9],
            vec![0.0; 9],
        );
        let b = make_snapshot_vals("b", 1, pos_b, vec![0.0; 3], vec![0.0; 9], vec![0.0; 9]);
        let result = largest_position_changes(&a, &b, 1).expect("ok");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 1); // Gaussian index 1
        assert!((result[0].1 - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_largest_position_changes_k_greater_than_n() {
        // k=100 with only 3 Gaussians → returns all 3.
        let a = make_snapshot("a", 0, 3, 0.0);
        let b = make_snapshot("b", 1, 3, 1.0);
        let result = largest_position_changes(&a, &b, 100).expect("ok");
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_largest_position_changes_size_mismatch() {
        let a = make_snapshot("a", 0, 3, 0.0);
        let b = make_snapshot("b", 1, 5, 0.0);
        assert!(matches!(
            largest_position_changes(&a, &b, 1),
            Err(DiffError::SizeMismatch { .. })
        ));
    }

    // ------------------------------------------------------------------
    // opacity_changes
    // ------------------------------------------------------------------

    #[test]
    fn test_opacity_changes_all_below_threshold() {
        // All logits very negative → all below 0.5 threshold.
        let a = make_snapshot_vals(
            "a",
            0,
            vec![0.0; 3],
            vec![-10.0],
            vec![0.0; 3],
            vec![0.0; 3],
        );
        let b = make_snapshot_vals(
            "b",
            1,
            vec![0.0; 3],
            vec![-10.0],
            vec![0.0; 3],
            vec![0.0; 3],
        );
        let (active, inactive) = opacity_changes(&a, &b, 0.5).expect("ok");
        assert!(active.is_empty());
        assert!(inactive.is_empty());
    }

    #[test]
    fn test_opacity_changes_crossing() {
        // Gaussian 0: A=inactive (logit -10), B=active (logit +10).
        // Gaussian 1: A=active (logit +10), B=inactive (logit -10).
        let a = make_snapshot_vals(
            "a",
            0,
            vec![0.0; 6],
            vec![-10.0, 10.0],
            vec![0.0; 6],
            vec![0.0; 6],
        );
        let b = make_snapshot_vals(
            "b",
            1,
            vec![0.0; 6],
            vec![10.0, -10.0],
            vec![0.0; 6],
            vec![0.0; 6],
        );
        let (active, inactive) = opacity_changes(&a, &b, 0.5).expect("ok");
        assert_eq!(active, vec![0]);
        assert_eq!(inactive, vec![1]);
    }

    // ------------------------------------------------------------------
    // per_gaussian_change_score
    // ------------------------------------------------------------------

    #[test]
    fn test_per_gaussian_change_score_identical() {
        let a = make_snapshot("a", 0, 5, 1.0);
        let scores = per_gaussian_change_score(&a, &a).expect("ok");
        assert_eq!(scores.len(), 5);
        for s in &scores {
            assert!(s.abs() < 1e-6, "expected ~0, got {}", s);
        }
    }

    #[test]
    fn test_per_gaussian_change_score_different() {
        let a = make_snapshot("a", 0, 4, 0.0);
        let b = make_snapshot("b", 1, 4, 1.0);
        let scores = per_gaussian_change_score(&a, &b).expect("ok");
        for s in &scores {
            assert!(*s > 0.0, "expected positive score");
        }
    }

    #[test]
    fn test_per_gaussian_change_score_size_mismatch() {
        let a = make_snapshot("a", 0, 3, 0.0);
        let b = make_snapshot("b", 1, 4, 0.0);
        assert!(matches!(
            per_gaussian_change_score(&a, &b),
            Err(DiffError::SizeMismatch { .. })
        ));
    }

    // ------------------------------------------------------------------
    // change_score_histogram
    // ------------------------------------------------------------------

    #[test]
    fn test_change_score_histogram_bin_count() {
        let scores = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
        let (edges, counts) = change_score_histogram(&scores, 4).expect("ok");
        assert_eq!(edges.len(), 5); // bins + 1
        assert_eq!(counts.len(), 4);
    }

    #[test]
    fn test_change_score_histogram_sum_equals_n() {
        let scores: Vec<f32> = (0..20).map(|i| i as f32 * 0.05).collect();
        let (_edges, counts) = change_score_histogram(&scores, 5).expect("ok");
        let total: usize = counts.iter().sum();
        assert_eq!(total, 20);
    }

    #[test]
    fn test_change_score_histogram_bins_zero_error() {
        let scores = vec![1.0f32; 5];
        assert!(matches!(
            change_score_histogram(&scores, 0),
            Err(DiffError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_change_score_histogram_empty_error() {
        assert!(matches!(
            change_score_histogram(&[], 5),
            Err(DiffError::InvalidConfig(_))
        ));
    }

    // ------------------------------------------------------------------
    // snapshots_approximately_equal
    // ------------------------------------------------------------------

    #[test]
    fn test_snapshots_approximately_equal_identical() {
        let a = make_snapshot("a", 0, 10, 0.5);
        let eq = snapshots_approximately_equal(&a, &a, 1e-5, 1e-8).expect("ok");
        assert!(eq);
    }

    #[test]
    fn test_snapshots_approximately_equal_large_diff() {
        let a = make_snapshot("a", 0, 5, 0.0);
        let b = make_snapshot("b", 1, 5, 100.0);
        let eq = snapshots_approximately_equal(&a, &b, 1e-5, 1e-8).expect("ok");
        assert!(!eq);
    }

    #[test]
    fn test_snapshots_approximately_equal_within_atol() {
        let a = make_snapshot("a", 0, 3, 1.0);
        let b = make_snapshot_vals(
            "b",
            1,
            vec![1.0 + 1e-7; 9],
            vec![1.0 + 1e-7; 3],
            vec![1.0 + 1e-7; 9],
            vec![1.0 + 1e-7; 9],
        );
        let eq = snapshots_approximately_equal(&a, &b, 1e-5, 1e-6).expect("ok");
        assert!(eq);
    }

    // ------------------------------------------------------------------
    // diff_sequence
    // ------------------------------------------------------------------

    #[test]
    fn test_diff_sequence_empty_error() {
        let config = DiffConfig::default();
        assert!(matches!(
            diff_sequence(&[], &config),
            Err(DiffError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_diff_sequence_single_error() {
        let a = make_snapshot("a", 0, 5, 0.0);
        let config = DiffConfig::default();
        assert!(matches!(
            diff_sequence(&[a], &config),
            Err(DiffError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_diff_sequence_two_snapshots() {
        let a = make_snapshot("a", 0, 5, 0.0);
        let b = make_snapshot("b", 10, 5, 1.0);
        let config = DiffConfig::default();
        let diffs = diff_sequence(&[a, b], &config).expect("ok");
        assert_eq!(diffs.len(), 1);
    }

    #[test]
    fn test_diff_sequence_three_snapshots() {
        let a = make_snapshot("a", 0, 5, 0.0);
        let b = make_snapshot("b", 10, 5, 0.5);
        let c = make_snapshot("c", 20, 5, 1.0);
        let config = DiffConfig::default();
        let diffs = diff_sequence(&[a, b, c], &config).expect("ok");
        assert_eq!(diffs.len(), 2);
    }

    // ------------------------------------------------------------------
    // detect_regression
    // ------------------------------------------------------------------

    #[test]
    fn test_detect_regression_identical_no_regression() {
        let a = make_snapshot("a", 0, 5, 0.5);
        let config = DiffConfig::default();
        let diff = diff_models(&a, &a, &config).expect("ok");
        let report = detect_regression(&diff, 0.05, 0.1, 0.01);
        assert!(!report.overall_regression);
        assert!(report.details.is_empty());
    }

    #[test]
    fn test_detect_regression_scale_increased() {
        // Model B has much larger scales (log-scale increased from 0 to 5).
        let a = make_snapshot("a", 0, 5, 0.0);
        let b = make_snapshot_vals(
            "b",
            10,
            vec![0.0; 15],
            vec![0.0; 5],
            vec![5.0; 15], // scales increased
            vec![0.0; 15],
        );
        let config = DiffConfig::default();
        let diff = diff_models(&a, &b, &config).expect("ok");
        // mean_change of scale should be 5.0 > scale_threshold 0.1.
        let report = detect_regression(&diff, 0.05, 0.1, 100.0); // high pos threshold
        assert!(report.scale_regressed);
        assert!(report.overall_regression);
    }

    #[test]
    fn test_detect_regression_position_unstable() {
        let a = make_snapshot("a", 0, 5, 0.0);
        let b = make_snapshot_vals(
            "b",
            1,
            vec![10.0; 15], // large position shift
            vec![0.0; 5],
            vec![0.0; 15],
            vec![0.0; 15],
        );
        let config = DiffConfig::default();
        let diff = diff_models(&a, &b, &config).expect("ok");
        let report = detect_regression(&diff, 100.0, 100.0, 0.01);
        assert!(report.position_unstable);
        assert!(report.overall_regression);
    }

    // ------------------------------------------------------------------
    // summarize_progress
    // ------------------------------------------------------------------

    #[test]
    fn test_summarize_progress_empty_error() {
        assert!(matches!(
            summarize_progress(&[], 1e-4),
            Err(DiffError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_summarize_progress_one_diff() {
        let a = make_snapshot("a", 0, 5, 0.0);
        let b = make_snapshot("b", 100, 5, 1.0);
        let config = DiffConfig::default();
        let diff = diff_models(&a, &b, &config).expect("ok");
        let summary = summarize_progress(&[diff], 1e-4).expect("ok");
        assert_eq!(summary.n_steps, 1);
        assert_eq!(summary.total_steps, 100);
    }

    #[test]
    fn test_summarize_progress_converging() {
        // Three diffs with decreasing summary_scores → converging.
        let make_diff = |score: f32, step_a: usize, step_b: usize| {
            // Build a ModelDiff with the desired summary_score.
            // We can construct a diff from identical + small perturbation.
            let a = make_snapshot("a", step_a, 5, 0.0);
            let b = make_snapshot_vals(
                "b",
                step_b,
                vec![score; 15], // positions differ by 'score'
                vec![0.0; 5],
                vec![0.0; 15],
                vec![0.0; 15],
            );
            let config = DiffConfig::default();
            diff_models(&a, &b, &config).expect("ok")
        };
        let d1 = make_diff(3.0, 0, 10);
        let d2 = make_diff(2.0, 10, 20);
        let d3 = make_diff(1.0, 20, 30);
        let summary = summarize_progress(&[d1, d2, d3], 1e-4).expect("ok");
        assert!(summary.converging);
        assert_eq!(summary.total_steps, 30);
    }

    #[test]
    fn test_summarize_progress_stalled() {
        // All diffs have near-zero change → stalled.
        let a = make_snapshot("a", 0, 5, 0.5);
        let config = DiffConfig::default();
        let d1 = diff_models(&a, &a, &config).expect("ok");
        let d2 = diff_models(&a, &a, &config).expect("ok");
        let d3 = diff_models(&a, &a, &config).expect("ok");
        let summary = summarize_progress(&[d1, d2, d3], 1e-3).expect("ok");
        assert!(summary.stalled);
    }

    // ------------------------------------------------------------------
    // DiffConfig::validate
    // ------------------------------------------------------------------

    #[test]
    fn test_diff_config_validate_negative_epsilon() {
        let config = DiffConfig {
            epsilon: -1.0,
            ..DiffConfig::default()
        };
        assert!(matches!(
            config.validate(),
            Err(DiffError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_diff_config_validate_default_ok() {
        assert!(DiffConfig::default().validate().is_ok());
    }

    // ------------------------------------------------------------------
    // diff_models_variable
    // ------------------------------------------------------------------

    /// Helper: build a snapshot from a list of (x,y,z) positions with
    /// uniform opacities/scales/colors.
    fn make_snapshot_positions(name: &str, step: usize, positions: &[[f32; 3]]) -> ModelSnapshot {
        let n = positions.len();
        let pos_flat: Vec<f32> = positions.iter().flat_map(|p| p.iter().copied()).collect();
        ModelSnapshot::new(
            name,
            step,
            pos_flat,
            vec![0.5_f32; n],
            vec![0.0_f32; n * 3],
            vec![0.3_f32; n * 3],
        )
        .expect("valid snapshot")
    }

    #[test]
    fn test_diff_variable_identical_models() {
        // Same model diffed against itself → 0 added, 0 removed, all matched.
        let positions: Vec<[f32; 3]> = (0..20).map(|i| [i as f32 * 2.0, 0.0, 0.0]).collect();
        let a = make_snapshot_positions("a", 0, &positions);
        let config = DiffConfig::default();
        let diff = diff_models_variable(&a, &a, &config).expect("ok");
        assert_eq!(diff.added_gaussians, 0, "no added gaussians");
        assert_eq!(diff.removed_gaussians, 0, "no removed gaussians");
        assert_eq!(diff.n_gaussians, 20, "all 20 matched");
    }

    #[test]
    fn test_diff_variable_b_larger() {
        // A has 10 Gaussians on a line, B has those same 10 plus 10 extra far away.
        let base: Vec<[f32; 3]> = (0..10).map(|i| [i as f32 * 2.0, 0.0, 0.0]).collect();
        let extra: Vec<[f32; 3]> = (0..10).map(|i| [i as f32 * 2.0, 1000.0, 0.0]).collect();
        let a_positions = base.clone();
        let mut b_positions = base.clone();
        b_positions.extend_from_slice(&extra);

        let a = make_snapshot_positions("a", 0, &a_positions);
        let b = make_snapshot_positions("b", 1, &b_positions);
        let config = DiffConfig::default();
        let diff = diff_models_variable(&a, &b, &config).expect("ok");
        assert_eq!(diff.added_gaussians, 10, "10 extra B gaussians are added");
        assert_eq!(diff.removed_gaussians, 0, "no A gaussians removed");
    }

    #[test]
    fn test_diff_variable_a_larger() {
        // A has 10 base + 10 extra far away, B has only the 10 base.
        let base: Vec<[f32; 3]> = (0..10).map(|i| [i as f32 * 2.0, 0.0, 0.0]).collect();
        let extra: Vec<[f32; 3]> = (0..10).map(|i| [i as f32 * 2.0, 1000.0, 0.0]).collect();
        let mut a_positions = base.clone();
        a_positions.extend_from_slice(&extra);
        let b_positions = base.clone();

        let a = make_snapshot_positions("a", 0, &a_positions);
        let b = make_snapshot_positions("b", 1, &b_positions);
        let config = DiffConfig::default();
        let diff = diff_models_variable(&a, &b, &config).expect("ok");
        assert_eq!(diff.added_gaussians, 0, "no B gaussians are added");
        assert_eq!(diff.removed_gaussians, 10, "10 A gaussians have no match");
    }

    #[test]
    fn test_diff_variable_all_moved_far() {
        // All B positions shifted by 100.0 on x — far beyond the default 0.5 match_radius.
        let a_positions: Vec<[f32; 3]> = (0..5).map(|i| [i as f32, 0.0, 0.0]).collect();
        let b_positions: Vec<[f32; 3]> = (0..5).map(|i| [i as f32 + 100.0, 0.0, 0.0]).collect();

        let a = make_snapshot_positions("a", 0, &a_positions);
        let b = make_snapshot_positions("b", 1, &b_positions);
        let config = DiffConfig::default();
        let diff = diff_models_variable(&a, &b, &config).expect("ok");
        assert_eq!(diff.added_gaussians, 5, "all B gaussians are new");
        assert_eq!(diff.removed_gaussians, 5, "all A gaussians are gone");
        assert_eq!(diff.n_gaussians, 0, "no matched pairs");
    }

    #[test]
    fn test_diff_variable_partial_match() {
        // A has 10 Gaussians; B has 8 close-matches + 4 far-away extras.
        // A[8] and A[9] have no B match → 2 removed.
        // B[8..12] are far away → 4 added.
        let a_positions: Vec<[f32; 3]> = (0..10).map(|i| [i as f32 * 3.0, 0.0, 0.0]).collect();
        // B[0..8]: very close to A[0..8] (offset 0.01 — within 0.5 radius)
        let mut b_positions: Vec<[f32; 3]> =
            (0..8).map(|i| [i as f32 * 3.0 + 0.01, 0.0, 0.0]).collect();
        // B[8..12]: far from any A position (y = 500)
        for j in 0..4 {
            b_positions.push([j as f32, 500.0, 0.0]);
        }

        let a = make_snapshot_positions("a", 0, &a_positions);
        let b = make_snapshot_positions("b", 1, &b_positions);
        let config = DiffConfig::default();
        let diff = diff_models_variable(&a, &b, &config).expect("ok");
        assert_eq!(diff.added_gaussians, 4, "4 far B gaussians are added");
        assert_eq!(diff.removed_gaussians, 2, "A[8] and A[9] have no match");
        assert_eq!(diff.n_gaussians, 8, "8 matched pairs");
    }

    #[test]
    fn test_diff_variable_empty_a() {
        let a = ModelSnapshot {
            name: "a".into(),
            step: 0,
            n_gaussians: 0,
            positions: vec![],
            opacities: vec![],
            scales: vec![],
            colors: vec![],
        };
        let b = make_snapshot_positions("b", 1, &[[0.0, 0.0, 0.0]]);
        let config = DiffConfig::default();
        assert!(matches!(
            diff_models_variable(&a, &b, &config),
            Err(DiffError::EmptyModelA)
        ));
    }

    #[test]
    fn test_diff_variable_empty_b() {
        let a = make_snapshot_positions("a", 0, &[[0.0, 0.0, 0.0]]);
        let b = ModelSnapshot {
            name: "b".into(),
            step: 1,
            n_gaussians: 0,
            positions: vec![],
            opacities: vec![],
            scales: vec![],
            colors: vec![],
        };
        let config = DiffConfig::default();
        assert!(matches!(
            diff_models_variable(&a, &b, &config),
            Err(DiffError::EmptyModelB)
        ));
    }

    #[test]
    fn test_diff_variable_format_output() {
        // Ensure format_model_diff produces output containing "Added:" text.
        let a_positions: Vec<[f32; 3]> = (0..5).map(|i| [i as f32 * 3.0, 0.0, 0.0]).collect();
        let mut b_positions: Vec<[f32; 3]> = (0..5).map(|i| [i as f32 * 3.0, 0.0, 0.0]).collect();
        // Add 3 extra Gaussians far away.
        for k in 0..3 {
            b_positions.push([k as f32, 999.0, 0.0]);
        }
        let a = make_snapshot_positions("snap_a", 0, &a_positions);
        let b = make_snapshot_positions("snap_b", 10, &b_positions);
        let config = DiffConfig::default();
        let diff = diff_models_variable(&a, &b, &config).expect("ok");
        let text = format_model_diff(&diff);
        assert!(
            text.contains("Added:"),
            "formatted diff should contain 'Added:' but got:\n{}",
            text
        );
        assert_eq!(diff.added_gaussians, 3);
        assert_eq!(diff.removed_gaussians, 0);
    }
}
