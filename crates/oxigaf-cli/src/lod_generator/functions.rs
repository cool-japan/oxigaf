//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    LodChain, LodConfig, LodError, LodInputSlices, LodLevel, LodStats, LodStrategy,
};

/// Advance xorshift64 state and return the next pseudo-random u64.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    if *state == 0 {
        *state = 1;
    }
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Apply sigmoid to convert logit opacities to probabilities in [0, 1].
///
/// `sigmoid(x) = 1 / (1 + exp(-x))`
pub fn compute_opacity_values(opacities: &[f32]) -> Vec<f32> {
    opacities
        .iter()
        .map(|&x| 1.0_f32 / (1.0_f32 + (-x).exp()))
        .collect()
}

/// Return the stable-sorted indices of the top-k Gaussians by sigmoid(opacity).
///
/// Indices are returned in ascending order for stability.
pub fn select_top_opacity_indices(n_gaussians: usize, opacities: &[f32], k: usize) -> Vec<usize> {
    let probs = compute_opacity_values(opacities);
    let mut pairs: Vec<(f32, usize)> = probs.into_iter().enumerate().map(|(i, p)| (p, i)).collect();
    // Sort descending by probability.
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let take = k.min(n_gaussians);
    let mut indices: Vec<usize> = pairs.iter().take(take).map(|&(_, i)| i).collect();
    // Restore ascending index order for stability.
    indices.sort_unstable();
    indices
}

/// Select k uniformly spaced indices from [0, n_gaussians).
///
/// Always includes index 0 and index n_gaussians-1 when k > 1.
pub fn select_uniform_indices(n_gaussians: usize, k: usize) -> Vec<usize> {
    if k == 0 || n_gaussians == 0 {
        return Vec::new();
    }
    if k >= n_gaussians {
        return (0..n_gaussians).collect();
    }
    let mut indices = Vec::with_capacity(k);
    for i in 0..k {
        indices.push(i * n_gaussians / k);
    }
    // Force last element to be the final index when k > 1.
    if k > 1 {
        if let Some(last) = indices.last_mut() {
            *last = n_gaussians - 1;
        }
    }
    indices.sort_unstable();
    indices.dedup();
    indices
}

/// Select exactly `min(k, n_gaussians)` Gaussians distributed across a 3D
/// spatial grid.
///
/// Divides the bounding box into ceil(k^(1/3)) cells per axis. For each
/// non-empty cell the first encountered Gaussian is selected. Real point
/// clouds rarely occupy every cell of the bounding grid, so this first pass
/// alone typically yields fewer than `k` picks; a second pass tops up the
/// remainder from the unpicked Gaussians (in storage order) so the caller
/// reliably gets the count it asked for instead of an unreported shortfall.
/// Returns indices in ascending order.
pub fn select_spatial_grid_indices(positions: &[f32], k: usize) -> Result<Vec<usize>, LodError> {
    if positions.is_empty() {
        return Err(LodError::EmptyCloud);
    }
    let n_gaussians = positions.len() / 3;
    if n_gaussians == 0 {
        return Err(LodError::EmptyCloud);
    }
    if k == 0 {
        return Ok(Vec::new());
    }
    if k >= n_gaussians {
        return Ok((0..n_gaussians).collect());
    }

    // Compute bounding box.
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for i in 0..n_gaussians {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];
        if x < min_x {
            min_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if z < min_z {
            min_z = z;
        }
        if x > max_x {
            max_x = x;
        }
        if y > max_y {
            max_y = y;
        }
        if z > max_z {
            max_z = z;
        }
    }

    // Cells per axis: ceil(k^(1/3)), at least 1.
    let cells_per_axis = ((k as f64).cbrt().ceil() as usize).max(1);
    let total_cells = cells_per_axis * cells_per_axis * cells_per_axis;
    let mut cell_occupied: Vec<bool> = vec![false; total_cells];
    let mut picked: Vec<bool> = vec![false; n_gaussians];
    let mut selected: Vec<usize> = Vec::with_capacity(k);

    let range_x = (max_x - min_x).max(f32::EPSILON);
    let range_y = (max_y - min_y).max(f32::EPSILON);
    let range_z = (max_z - min_z).max(f32::EPSILON);
    let c = cells_per_axis as f32;

    for i in 0..n_gaussians {
        if selected.len() >= k {
            break;
        }
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];

        let cx = (((x - min_x) / range_x) * c).min(c - 1.0) as usize;
        let cy = (((y - min_y) / range_y) * c).min(c - 1.0) as usize;
        let cz = (((z - min_z) / range_z) * c).min(c - 1.0) as usize;
        let cell_id = cx * cells_per_axis * cells_per_axis + cy * cells_per_axis + cz;

        if !cell_occupied[cell_id] {
            cell_occupied[cell_id] = true;
            picked[i] = true;
            selected.push(i);
        }
    }

    // Top up: the grid pass alone almost always undershoots `k` because
    // real clouds only occupy a fraction of the bounding grid's cells.
    // Without this, callers silently got far fewer Gaussians than
    // requested (e.g. k=50 on a 4×4×4=64-cell grid with ~25-30 occupied
    // cells), and the achieved ratio was reported as if it had been asked
    // for. `k < n_gaussians` is guaranteed by the early return above, so
    // there are always enough unpicked Gaussians to reach exactly `k`.
    if selected.len() < k {
        for (i, is_picked) in picked.iter_mut().enumerate() {
            if selected.len() >= k {
                break;
            }
            if !*is_picked {
                *is_picked = true;
                selected.push(i);
            }
        }
    }

    selected.sort_unstable();
    Ok(selected)
}

/// Fisher-Yates shuffle using xorshift64; return the first k indices in ascending order.
pub fn select_random_indices(n_gaussians: usize, k: usize, seed: u64) -> Vec<usize> {
    if k == 0 || n_gaussians == 0 {
        return Vec::new();
    }
    let take = k.min(n_gaussians);
    let mut indices: Vec<usize> = (0..n_gaussians).collect();
    let mut state = if seed == 0 { 1u64 } else { seed };

    for i in 0..(n_gaussians - 1) {
        let rand_val = xorshift64(&mut state);
        let j = i + (rand_val as usize % (n_gaussians - i));
        indices.swap(i, j);
    }
    let mut result = indices[..take].to_vec();
    result.sort_unstable();
    result
}

/// Permutation of `0..opacities.len()` ordered by descending sigmoid-opacity
/// (most opaque first).
///
/// Used to let rank-based selectors ([`select_uniform_indices`],
/// [`select_random_indices`]) operate on opacity rank instead of raw
/// storage order when [`LodConfig::sort_by_opacity`] is set.
pub(super) fn opacity_descending_permutation(opacities: &[f32]) -> Vec<usize> {
    let probs = compute_opacity_values(opacities);
    let mut perm: Vec<usize> = (0..opacities.len()).collect();
    perm.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    perm
}

/// Reinterpret `ranks` (indices in `0..n_gaussians`, as produced by a
/// rank-based selector such as [`select_uniform_indices`] or
/// [`select_random_indices`]) as opacity ranks when `sort_by_opacity` is
/// set, mapping each rank back to the original Gaussian index via
/// [`opacity_descending_permutation`]. When `sort_by_opacity` is false,
/// `ranks` is returned unchanged (the "rank" already *is* the storage
/// index). Re-sorts ascending afterward to match every other `select_*`
/// function's convention of returning ascending original indices.
pub(super) fn select_indices_by_rank(
    ranks: Vec<usize>,
    sort_by_opacity: bool,
    opacities: &[f32],
) -> Vec<usize> {
    if !sort_by_opacity {
        return ranks;
    }
    let perm = opacity_descending_permutation(opacities);
    let mut indices: Vec<usize> = ranks.into_iter().map(|r| perm[r]).collect();
    indices.sort_unstable();
    indices
}

/// Extract rows from a flat N×M array using the given row indices.
///
/// The stride M is inferred as `source.len() / n_gaussians`. `source.len()`
/// must be an exact multiple of `n_gaussians` and every index must be
/// `< n_gaussians`, or this returns an error rather than silently
/// extracting misaligned rows (a mismatched but non-zero stride) or
/// panicking on an out-of-range slice (`idx * stride` past `source.len()`).
/// Returns `indices.len() × M` elements.
///
/// # Errors
///
/// Returns [`LodError::ArrayLengthMismatch`] if `source.len()` is not a
/// multiple of `n_gaussians`, or [`LodError::IndexOutOfRange`] if any index
/// is `>= n_gaussians`.
pub fn extract_subset(
    source: &[f32],
    n_gaussians: usize,
    indices: &[usize],
) -> Result<Vec<f32>, LodError> {
    if n_gaussians == 0 || source.is_empty() || indices.is_empty() {
        return Ok(Vec::new());
    }
    if !source.len().is_multiple_of(n_gaussians) {
        return Err(LodError::ArrayLengthMismatch {
            n_gaussians,
            field: "source".to_string(),
            actual: source.len(),
        });
    }
    let stride = source.len() / n_gaussians;
    let mut out = Vec::with_capacity(indices.len() * stride);
    for &idx in indices {
        if idx >= n_gaussians {
            return Err(LodError::IndexOutOfRange {
                index: idx,
                n_gaussians,
            });
        }
        let start = idx * stride;
        let end = start + stride;
        out.extend_from_slice(&source[start..end]);
    }
    Ok(out)
}

/// Validate that every flat array in `input` matches `input.n_gaussians`.
///
/// Called by both [`generate_lod_chain`] (eagerly, before any level is
/// generated) and [`generate_lod_level`] (so the latter is also safe to
/// call directly, bypassing the chain, instead of panicking or silently
/// extracting misaligned rows when a caller-supplied array disagrees with
/// `n_gaussians`).
fn validate_input_slices(input: &LodInputSlices<'_>) -> Result<(), LodError> {
    let n_gaussians = input.n_gaussians;
    if input.positions.len() != n_gaussians * 3 {
        return Err(LodError::ArrayLengthMismatch {
            n_gaussians,
            field: "positions".to_string(),
            actual: input.positions.len(),
        });
    }
    if input.rotations.len() != n_gaussians * 4 {
        return Err(LodError::ArrayLengthMismatch {
            n_gaussians,
            field: "rotations".to_string(),
            actual: input.rotations.len(),
        });
    }
    if input.scales.len() != n_gaussians * 3 {
        return Err(LodError::ArrayLengthMismatch {
            n_gaussians,
            field: "scales".to_string(),
            actual: input.scales.len(),
        });
    }
    if input.opacities.len() != n_gaussians {
        return Err(LodError::ArrayLengthMismatch {
            n_gaussians,
            field: "opacities".to_string(),
            actual: input.opacities.len(),
        });
    }
    if n_gaussians > 0 && !input.sh_coefficients.len().is_multiple_of(n_gaussians) {
        return Err(LodError::ArrayLengthMismatch {
            n_gaussians,
            field: "sh_coefficients".to_string(),
            actual: input.sh_coefficients.len(),
        });
    }
    Ok(())
}

/// Generate a single LOD level by selecting `target_n` Gaussians from the cloud.
///
/// # Errors
///
/// Returns [`LodError::ArrayLengthMismatch`] if any array in `input`
/// disagrees with `input.n_gaussians`, or [`LodError::InsufficientGaussians`]
/// if `target_n > input.n_gaussians`.
pub fn generate_lod_level(
    input: LodInputSlices<'_>,
    target_n: usize,
    config: &LodConfig,
    level: usize,
) -> Result<LodLevel, LodError> {
    validate_input_slices(&input)?;
    let LodInputSlices {
        n_gaussians,
        positions,
        rotations,
        scales,
        opacities,
        sh_coefficients,
    } = input;
    if target_n > n_gaussians {
        return Err(LodError::InsufficientGaussians {
            k: target_n,
            n: n_gaussians,
        });
    }

    let indices = match config.strategy {
        LodStrategy::TopOpacity => select_top_opacity_indices(n_gaussians, opacities, target_n),
        LodStrategy::Uniform => select_indices_by_rank(
            select_uniform_indices(n_gaussians, target_n),
            config.sort_by_opacity,
            opacities,
        ),
        LodStrategy::SpatialGrid => select_spatial_grid_indices(positions, target_n)?,
        LodStrategy::Random => {
            // Deterministic seed derived from level index.
            let seed = (level as u64 + 1).wrapping_mul(6_364_136_223_846_793_005u64);
            select_indices_by_rank(
                select_random_indices(n_gaussians, target_n, seed),
                config.sort_by_opacity,
                opacities,
            )
        }
    };

    let actual_n = indices.len();
    if actual_n != target_n {
        tracing::warn!(
            target_n,
            actual_n,
            strategy = ?config.strategy,
            "LOD level selection did not return the requested Gaussian count"
        );
    }
    let reduction_factor = if n_gaussians > 0 {
        actual_n as f32 / n_gaussians as f32
    } else {
        1.0
    };

    // Extract opacities directly (stride = 1).
    let selected_opacities: Vec<f32> = indices
        .iter()
        .filter_map(|&i| opacities.get(i).copied())
        .collect();

    Ok(LodLevel {
        level,
        n_gaussians: actual_n,
        reduction_factor,
        positions: extract_subset(positions, n_gaussians, &indices)?,
        rotations: extract_subset(rotations, n_gaussians, &indices)?,
        scales: extract_subset(scales, n_gaussians, &indices)?,
        opacities: selected_opacities,
        sh_coefficients: extract_subset(sh_coefficients, n_gaussians, &indices)?,
    })
}

/// Generate a full LOD chain from the original Gaussian cloud.
pub fn generate_lod_chain(
    positions: &[f32],
    rotations: &[f32],
    scales: &[f32],
    opacities: &[f32],
    sh_coefficients: &[f32],
    config: &LodConfig,
) -> Result<LodChain, LodError> {
    if positions.is_empty() {
        return Err(LodError::EmptyCloud);
    }
    if !positions.len().is_multiple_of(3) {
        return Err(LodError::InvalidPositionLength {
            len: positions.len(),
        });
    }
    let n_gaussians = positions.len() / 3;

    // Validate all array lengths eagerly (before generating any level) using
    // the same check `generate_lod_level` runs on every call, so a bad
    // rotations/scales/opacities/sh_coefficients array is reported up front
    // even for an `n_levels == 0` config that would otherwise never touch
    // `generate_lod_level` at all.
    let input = LodInputSlices {
        n_gaussians,
        positions,
        rotations,
        scales,
        opacities,
        sh_coefficients,
    };
    validate_input_slices(&input)?;

    config.validate()?;

    let mut levels = Vec::with_capacity(config.n_levels);
    for (lvl, &ratio) in config.reduction_ratios.iter().enumerate() {
        let target_n = ((n_gaussians as f32 * ratio).ceil() as usize)
            .max(1)
            .min(n_gaussians);
        let lod_level = generate_lod_level(input, target_n, config, lvl)?;
        levels.push(lod_level);
    }

    Ok(LodChain {
        original_n_gaussians: n_gaussians,
        levels,
    })
}

/// Compute statistics about an LOD chain.
pub fn compute_lod_stats(chain: &LodChain) -> LodStats {
    let n_levels = chain.levels.len();
    let level_sizes: Vec<usize> = chain.levels.iter().map(|l| l.n_gaussians).collect();
    let memory_estimates: Vec<usize> = chain
        .levels
        .iter()
        .map(|l| {
            let sh_per = l
                .sh_coefficients
                .len()
                .checked_div(l.n_gaussians)
                .unwrap_or(0);
            // 4 bytes × (3 + 4 + 3 + 1 + sh_per) × n_gaussians
            4 * (3 + 4 + 3 + 1 + sh_per) * l.n_gaussians
        })
        .collect();
    let total_memory: usize = memory_estimates.iter().sum();
    LodStats {
        n_levels,
        original_gaussians: chain.original_n_gaussians,
        level_sizes,
        memory_estimates,
        total_memory,
    }
}

/// Per-quaternion normalized-LERP of two flat N×4 quaternion arrays.
///
/// A naive component-wise LERP of two *unit* quaternions is not itself a
/// unit quaternion (the renderer's quaternion → rotation-matrix conversion
/// would then produce a scaled/skewed rotation), and blending along the
/// "long way around" the hypersphere when `dot(a, b) < 0` passes
/// arbitrarily close to the zero quaternion at `t = 0.5`, producing a
/// degenerate rotation. This corrects both: negate `b` when the dot product
/// is negative (shortest-path interpolation), LERP component-wise, then
/// renormalize.
///
/// Assumes `a.len() == b.len()` and both are a multiple of 4 (guaranteed by
/// [`merge_lod_levels`] via [`LodLevel::validate`] before this is called).
fn nlerp_quaternions(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len());
    for (qa, qb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let dot = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3];
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        let mut lerped = [0.0f32; 4];
        for i in 0..4 {
            lerped[i] = qa[i] + (sign * qb[i] - qa[i]) * t;
        }
        let norm_sq = lerped[0] * lerped[0]
            + lerped[1] * lerped[1]
            + lerped[2] * lerped[2]
            + lerped[3] * lerped[3];
        if norm_sq > f32::EPSILON {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for v in &mut lerped {
                *v *= inv_norm;
            }
        } else {
            // Degenerate (both inputs ~zero): fall back to `a` rather than
            // emit a zero/NaN quaternion.
            lerped.copy_from_slice(qa);
        }
        out.extend_from_slice(&lerped);
    }
    out
}

/// Blend two LOD levels of equal size via linear interpolation.
///
/// `weight_b = 0.0` → pure `level_a`; `weight_b = 1.0` → pure `level_b`.
/// Positions, scales, opacities and SH coefficients use plain per-element
/// LERP; rotations use quaternion NLERP (`nlerp_quaternions`) since a
/// component-wise LERP of two unit quaternions is not itself a unit
/// quaternion.
///
/// # Errors
///
/// Returns [`LodError::ArrayLengthMismatch`] if `level_a` and `level_b`
/// disagree on `n_gaussians`, if either level's own arrays are internally
/// inconsistent (see [`LodLevel::validate`]), or if their `sh_coefficients`
/// lengths differ (e.g. two levels generated with different SH degrees) —
/// the last case cannot be inferred from `n_gaussians` alone.
pub fn merge_lod_levels(
    level_a: &LodLevel,
    level_b: &LodLevel,
    weight_b: f32,
) -> Result<LodLevel, LodError> {
    if level_a.n_gaussians != level_b.n_gaussians {
        return Err(LodError::ArrayLengthMismatch {
            n_gaussians: level_a.n_gaussians,
            field: "level_b".to_string(),
            actual: level_b.n_gaussians,
        });
    }
    // `n_gaussians` matching alone is not sufficient: `level_a.validate()`
    // and `level_b.validate()` additionally guarantee positions/rotations/
    // scales/opacities each match their own `n_gaussians` (so those four
    // are then pairwise-equal for free), but `sh_coefficients` can still
    // independently satisfy "multiple of n_gaussians" on each side while
    // disagreeing with the other (e.g. differing SH degrees), which a plain
    // `zip`-based lerp would otherwise silently truncate to the shorter
    // side instead of reporting.
    level_a.validate()?;
    level_b.validate()?;
    if level_a.sh_coefficients.len() != level_b.sh_coefficients.len() {
        return Err(LodError::ArrayLengthMismatch {
            n_gaussians: level_a.n_gaussians,
            field: "sh_coefficients".to_string(),
            actual: level_b.sh_coefficients.len(),
        });
    }

    let t = weight_b; // 0 = pure a, 1 = pure b
    let lerp_vec = |a: &[f32], b: &[f32]| -> Vec<f32> {
        a.iter()
            .zip(b.iter())
            .map(|(&av, &bv)| av + (bv - av) * t)
            .collect()
    };
    let new_reduction =
        level_a.reduction_factor + (level_b.reduction_factor - level_a.reduction_factor) * t;

    Ok(LodLevel {
        level: level_a.level,
        n_gaussians: level_a.n_gaussians,
        reduction_factor: new_reduction,
        positions: lerp_vec(&level_a.positions, &level_b.positions),
        rotations: nlerp_quaternions(&level_a.rotations, &level_b.rotations, t),
        scales: lerp_vec(&level_a.scales, &level_b.scales),
        opacities: lerp_vec(&level_a.opacities, &level_b.opacities),
        sh_coefficients: lerp_vec(&level_a.sh_coefficients, &level_b.sh_coefficients),
    })
}

/// Format LOD stats as a human-readable summary string.
pub fn format_lod_stats(stats: &LodStats) -> String {
    let size_parts: Vec<String> = stats.level_sizes.iter().map(|s| s.to_string()).collect();
    let chain_str = size_parts.join(" → ");
    let total_mb = stats.total_memory as f64 / (1024.0 * 1024.0);
    format!(
        "LOD Chain [{} levels]: {} Gaussians, total: {:.1} MB",
        stats.n_levels, chain_str, total_mb
    )
}

/// Estimate memory in bytes for a single LOD level.
///
/// params_per_gaussian = 3 (pos) + 4 (rot) + 3 (scale) + 1 (opacity) + sh_per
pub fn estimate_lod_memory(n_gaussians: usize, sh_coefficients_len: usize) -> usize {
    let sh_per = sh_coefficients_len.checked_div(n_gaussians).unwrap_or(0);
    4 * (3 + 4 + 3 + 1 + sh_per) * n_gaussians
}

/// Find a geometric progression [1.0, r, r², …, r^(n-1)] such that the total
/// memory across all levels fits within `target_memory_bytes`.
///
/// Uses binary search for `r` in [0.1, 1.0].
///
/// # Errors
///
/// Returns [`LodError::MemoryBudgetExceeded`] if even the smallest ratio
/// considered by the search (`r = 0.1`) does not fit within
/// `target_memory_bytes` — level 0 is always pinned to `1.0` regardless of
/// `r`, so no chain in the search space can do better than
/// `total_memory_for_r(0.1)`, which is reported as the achievable minimum.
pub fn find_optimal_reduction_ratios(
    n_gaussians: usize,
    target_memory_bytes: usize,
    n_levels: usize,
    sh_per_gaussian: usize,
) -> Result<Vec<f32>, LodError> {
    if n_levels == 0 {
        return Ok(Vec::new());
    }
    if n_levels == 1 {
        return Ok(vec![1.0_f32]);
    }

    let bytes_per_gaussian = 4 * (3 + 4 + 3 + 1 + sh_per_gaussian);

    let total_memory_for_r = |r: f64| -> f64 {
        let mut total = 0.0f64;
        for lv in 0..n_levels {
            let ratio = r.powi(lv as i32);
            let level_n = ((n_gaussians as f64 * ratio).ceil() as usize).max(1);
            total += (bytes_per_gaussian * level_n) as f64;
        }
        total
    };

    // Binary search: find largest r in [0.1, 1.0] that fits within the budget.
    let mut lo = 0.1f64;
    let mut hi = 1.0f64;
    for _ in 0..64 {
        let mid = (lo + hi) / 2.0;
        if total_memory_for_r(mid) <= target_memory_bytes as f64 {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // If `lo` never advanced past its initial 0.1, either r=0.1 fits (in
    // which case this is simply the tightest chain found) or nothing in
    // the search space fits and `lo` stayed at 0.1 because
    // `total_memory_for_r` is non-decreasing in `r`. Distinguish the two by
    // checking the achieved total directly, rather than silently returning
    // a chain that overshoots the budget with no indication.
    let achieved_bytes = total_memory_for_r(lo);
    if achieved_bytes > target_memory_bytes as f64 {
        return Err(LodError::MemoryBudgetExceeded {
            minimum_bytes: achieved_bytes as usize,
            target_bytes: target_memory_bytes,
        });
    }

    let r = lo as f32;
    let mut ratios = Vec::with_capacity(n_levels);
    for lv in 0..n_levels {
        ratios.push(r.powi(lv as i32).clamp(0.001, 1.0));
    }
    // Force exact 1.0 for the first level.
    if let Some(first) = ratios.first_mut() {
        *first = 1.0_f32;
    }
    Ok(ratios)
}
