//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::borrow::Cow;
use std::collections::HashMap;

use super::types::{
    DiffConfig, DiffError, FieldDiff, ModelDiff, ModelSnapshot, ProgressSummary, RegressionReport,
};

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
