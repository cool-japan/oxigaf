//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    DedupConfig, DedupKeepPolicy, DedupReport, DedupResult, DedupStats, DeduplicatorError,
    DuplicateGroup, GdDeduplicateInput, GdSceneSlices, SpatialHashMap,
};

/// Compute the bucket index for integer cell coordinates using large primes.
///
/// Uses i64 arithmetic to avoid overflow, then takes unsigned absolute value
/// modulo n_buckets.
pub fn gd_hash_cell(ix: i32, iy: i32, iz: i32, n_buckets: usize) -> usize {
    let h = (ix as i64)
        .wrapping_mul(73_856_093)
        .wrapping_add((iy as i64).wrapping_mul(19_349_663))
        .wrapping_add((iz as i64).wrapping_mul(83_492_791));
    (h.unsigned_abs() as usize) % n_buckets.max(1)
}

/// Convert a world-space position to integer cell coordinates.
pub fn gd_world_to_cell(pos: [f32; 3], cell_size: f32, bounds_min: [f32; 3]) -> [i32; 3] {
    [
        ((pos[0] - bounds_min[0]) / cell_size).floor() as i32,
        ((pos[1] - bounds_min[1]) / cell_size).floor() as i32,
        ((pos[2] - bounds_min[2]) / cell_size).floor() as i32,
    ]
}

/// Return the maximum absolute value of the 3-component scale vector at index `i`.
#[inline]
fn max_scale(scales: &[f32], i: usize) -> f32 {
    let a = scales[i * 3].abs();
    let b = scales[i * 3 + 1].abs();
    let c = scales[i * 3 + 2].abs();
    a.max(b).max(c)
}

/// Check whether Gaussians `i` and `j` are near-duplicates according to `config`.
///
/// Criteria applied in order (short-circuit on first failure):
/// 1. Position L2 distance < position_threshold
/// 2. |opacity_i − opacity_j| < opacity_threshold
/// 3. Relative scale difference < scale_threshold
/// 4. DC color L2 distance < color_threshold (only when sh_channels >= 3)
pub fn gd_are_duplicates(
    i: usize,
    j: usize,
    scene: GdSceneSlices<'_>,
    config: &DedupConfig,
) -> bool {
    let GdSceneSlices {
        positions,
        opacities,
        scales,
        sh_coeffs,
        sh_channels,
    } = scene;
    // 1. Position distance.
    let dx = positions[i * 3] - positions[j * 3];
    let dy = positions[i * 3 + 1] - positions[j * 3 + 1];
    let dz = positions[i * 3 + 2] - positions[j * 3 + 2];
    let pos_dist = (dx * dx + dy * dy + dz * dz).sqrt();
    if pos_dist >= config.position_threshold {
        return false;
    }

    // 2. Opacity difference.
    if (opacities[i] - opacities[j]).abs() >= config.opacity_threshold {
        return false;
    }

    // 3. Relative scale difference.
    let si = max_scale(scales, i);
    let sj = max_scale(scales, j);
    let denom = si.max(sj).max(1e-8);
    if (si - sj).abs() / denom >= config.scale_threshold {
        return false;
    }

    // 4. DC color distance (first 3 SH coefficients).
    if sh_channels >= 3 && !sh_coeffs.is_empty() {
        let cr = sh_coeffs[i * sh_channels] - sh_coeffs[j * sh_channels];
        let cg = sh_coeffs[i * sh_channels + 1] - sh_coeffs[j * sh_channels + 1];
        let cb = sh_coeffs[i * sh_channels + 2] - sh_coeffs[j * sh_channels + 2];
        let color_dist = (cr * cr + cg * cg + cb * cb).sqrt();
        if color_dist >= config.color_threshold {
            return false;
        }
    }

    true
}

fn uf_find(parent: &mut [usize], i: usize) -> usize {
    if parent[i] != i {
        parent[i] = uf_find(parent, parent[i]);
    }
    parent[i]
}

fn uf_union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra == rb {
        return;
    }
    if rank[ra] < rank[rb] {
        parent[ra] = rb;
    } else if rank[ra] > rank[rb] {
        parent[rb] = ra;
    } else {
        parent[rb] = ra;
        rank[ra] += 1;
    }
}

/// Collect Union-Find roots into groups of size >= 2.
fn uf_to_groups(parent: &mut [usize], n: usize) -> Vec<Vec<usize>> {
    let mut root_to_group: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        let r = uf_find(parent, i);
        root_to_group.entry(r).or_default().push(i);
    }
    root_to_group
        .into_values()
        .filter(|g| g.len() >= 2)
        .collect()
}

/// Validate that `opacities`/`scales`/`sh_coeffs` are long enough for `n`
/// Gaussians before any indexed access into them. Single choke point for
/// both duplicate-detection entry points, protecting `gd_are_duplicates`'s
/// unchecked indexing from every caller (direct, or via
/// [`gd_deduplicate`]/[`gd_analyze_duplicates`]) — previously only
/// `positions` was validated here.
fn validate_dedup_attributes(
    positions: &[f32],
    opacities: &[f32],
    scales: &[f32],
    sh_coeffs: &[f32],
    sh_channels: usize,
    n: usize,
) -> Result<(), DeduplicatorError> {
    if positions.len() < n * 3 {
        return Err(DeduplicatorError::PositionLengthMismatch {
            pos: positions.len(),
            n,
        });
    }
    if opacities.len() < n {
        return Err(DeduplicatorError::AttributeLengthMismatch {
            expected: n,
            got: opacities.len(),
            count: n,
            stride: 1,
        });
    }
    if scales.len() < n * 3 {
        return Err(DeduplicatorError::AttributeLengthMismatch {
            expected: n * 3,
            got: scales.len(),
            count: n,
            stride: 3,
        });
    }
    // `gd_are_duplicates` only indexes `sh_coeffs` when `sh_channels >= 3`
    // (and non-empty, which a length check subsumes).
    if sh_channels >= 3 && sh_coeffs.len() < n * sh_channels {
        return Err(DeduplicatorError::AttributeLengthMismatch {
            expected: n * sh_channels,
            got: sh_coeffs.len(),
            count: n,
            stride: sh_channels,
        });
    }
    Ok(())
}

/// Detect near-duplicate groups using a spatial hash map (O(N) average case).
///
/// Returns groups of >= 2 indices that are mutually near-duplicates via
/// Union-Find (transitively).
pub fn gd_find_duplicates_spatial(
    positions: &[f32],
    opacities: &[f32],
    scales: &[f32],
    sh_coeffs: &[f32],
    sh_channels: usize,
    n: usize,
    config: &DedupConfig,
) -> Result<Vec<Vec<usize>>, DeduplicatorError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    validate_dedup_attributes(positions, opacities, scales, sh_coeffs, sh_channels, n)?;

    let map = SpatialHashMap::new(
        n.next_power_of_two().max(16),
        config.cell_size,
        positions,
        n,
    )?;

    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<u8> = vec![0u8; n];

    for i in 0..n {
        let pos = [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
        let candidates = map.query_neighbors(pos);
        for j in candidates {
            if j <= i {
                continue;
            }
            if gd_are_duplicates(
                i,
                j,
                GdSceneSlices {
                    positions,
                    opacities,
                    scales,
                    sh_coeffs,
                    sh_channels,
                },
                config,
            ) {
                uf_union(&mut parent, &mut rank, i, j);
            }
        }
    }

    Ok(uf_to_groups(&mut parent, n))
}

/// Detect near-duplicate groups using O(N²) pairwise comparison.
///
/// Preferred for small N or verification. Uses the same Union-Find grouping.
pub fn gd_find_duplicates_brute(
    positions: &[f32],
    opacities: &[f32],
    scales: &[f32],
    sh_coeffs: &[f32],
    sh_channels: usize,
    n: usize,
    config: &DedupConfig,
) -> Result<Vec<Vec<usize>>, DeduplicatorError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    validate_dedup_attributes(positions, opacities, scales, sh_coeffs, sh_channels, n)?;

    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<u8> = vec![0u8; n];

    for i in 0..n {
        for j in (i + 1)..n {
            if gd_are_duplicates(
                i,
                j,
                GdSceneSlices {
                    positions,
                    opacities,
                    scales,
                    sh_coeffs,
                    sh_channels,
                },
                config,
            ) {
                uf_union(&mut parent, &mut rank, i, j);
            }
        }
    }

    Ok(uf_to_groups(&mut parent, n))
}

/// Select the index to keep from a duplicate group according to `policy`.
///
/// Returns `None` if `group` is empty (there is nothing to keep). The
/// original implementation evaluated `group[0]` as `unwrap_or`'s eager
/// fallback argument even for `KeepFirst`/`KeepLast`, which panics on an
/// empty group regardless of whether the fallback was actually needed.
pub fn gd_pick_representative(
    group: &[usize],
    opacities: &[f32],
    scales: &[f32],
    policy: &DedupKeepPolicy,
) -> Option<usize> {
    let &first = group.first()?;
    let best = match policy {
        DedupKeepPolicy::KeepHighestOpacity => {
            let mut best = first;
            let mut best_val = opacities[best];
            for &idx in &group[1..] {
                if opacities[idx] > best_val {
                    best_val = opacities[idx];
                    best = idx;
                }
            }
            best
        }
        DedupKeepPolicy::KeepLargestScale => {
            let mut best = first;
            let mut best_val = max_scale(scales, best);
            for &idx in &group[1..] {
                let v = max_scale(scales, idx);
                if v > best_val {
                    best_val = v;
                    best = idx;
                }
            }
            best
        }
        DedupKeepPolicy::KeepSmallestScale => {
            let mut best = first;
            let mut best_val = max_scale(scales, best);
            for &idx in &group[1..] {
                let v = max_scale(scales, idx);
                if v < best_val {
                    best_val = v;
                    best = idx;
                }
            }
            best
        }
        DedupKeepPolicy::KeepFirst => *group.iter().min()?,
        DedupKeepPolicy::KeepLast => *group.iter().max()?,
    };
    Some(best)
}

/// Build a boolean removal mask: `true` = remove this Gaussian.
///
/// For each duplicate group, a representative is kept (mask=false) and all
/// others are marked for removal (mask=true).
pub fn gd_build_remove_mask(
    duplicate_groups: &[Vec<usize>],
    n: usize,
    opacities: &[f32],
    scales: &[f32],
    policy: &DedupKeepPolicy,
) -> Vec<bool> {
    let mut mask = vec![false; n];
    for group in duplicate_groups {
        // `None` only for an empty group, which `uf_to_groups` never
        // produces (it filters to len >= 2); skip defensively rather than
        // panicking if some other caller constructs one directly.
        let Some(keep) = gd_pick_representative(group, opacities, scales, policy) else {
            continue;
        };
        for &idx in group {
            if idx != keep {
                mask[idx] = true;
            }
        }
    }
    mask
}

/// Filter a flat array with `stride` values per Gaussian, removing entries
/// where `mask[i] == true`.
///
/// # Errors
/// Returns [`DeduplicatorError::AttributeLengthMismatch`] if
/// `data.len() != mask.len() * stride`, rather than panicking on an
/// out-of-bounds slice or silently emitting a truncated, misaligned row.
pub fn gd_apply_mask(
    data: &[f32],
    mask: &[bool],
    stride: usize,
) -> Result<Vec<f32>, DeduplicatorError> {
    let n = mask.len();
    let expected = n * stride;
    if data.len() != expected {
        return Err(DeduplicatorError::AttributeLengthMismatch {
            expected,
            got: data.len(),
            count: n,
            stride,
        });
    }
    let mut out = Vec::with_capacity(expected);
    for i in 0..n {
        if !mask[i] {
            out.extend_from_slice(&data[i * stride..i * stride + stride]);
        }
    }
    Ok(out)
}

/// Filter a flat scalar array (stride=1), removing entries where `mask[i] == true`.
///
/// # Errors
/// See [`gd_apply_mask`].
pub fn gd_apply_scalar_mask(data: &[f32], mask: &[bool]) -> Result<Vec<f32>, DeduplicatorError> {
    gd_apply_mask(data, mask, 1)
}

/// Run the full near-duplicate detection and removal pipeline.
///
/// Validates array lengths, detects duplicates (spatial or brute-force),
/// selects representatives, and filters all attribute arrays.
pub fn gd_deduplicate(
    input: GdDeduplicateInput<'_>,
    config: &DedupConfig,
) -> Result<DedupResult, DeduplicatorError> {
    let GdDeduplicateInput {
        positions,
        rotations,
        scales,
        opacities,
        sh_coefficients,
        sh_channels,
        n_gaussians,
    } = input;
    if n_gaussians == 0 {
        return Err(DeduplicatorError::EmptyScene);
    }
    if positions.len() != n_gaussians * 3 {
        return Err(DeduplicatorError::PositionLengthMismatch {
            pos: positions.len(),
            n: n_gaussians,
        });
    }

    let groups = if config.use_spatial_hash {
        gd_find_duplicates_spatial(
            positions,
            opacities,
            scales,
            sh_coefficients,
            sh_channels,
            n_gaussians,
            config,
        )?
    } else {
        gd_find_duplicates_brute(
            positions,
            opacities,
            scales,
            sh_coefficients,
            sh_channels,
            n_gaussians,
            config,
        )?
    };

    let n_groups = groups.len();
    let group_sizes: Vec<usize> = groups.iter().map(Vec::len).collect();
    let mask = gd_build_remove_mask(&groups, n_gaussians, opacities, scales, &config.keep_policy);
    let n_removed = mask.iter().filter(|&&r| r).count();
    let n_after = n_gaussians - n_removed;

    let new_positions = gd_apply_mask(positions, &mask, 3)?;
    let new_rotations = gd_apply_mask(rotations, &mask, 4)?;
    let new_scales = gd_apply_mask(scales, &mask, 3)?;
    let new_opacities = gd_apply_scalar_mask(opacities, &mask)?;
    let new_sh = if sh_channels == 0 {
        Vec::new()
    } else {
        gd_apply_mask(sh_coefficients, &mask, sh_channels)?
    };

    Ok(DedupResult {
        positions: new_positions,
        rotations: new_rotations,
        scales: new_scales,
        opacities: new_opacities,
        sh_coefficients: new_sh,
        n_before: n_gaussians,
        n_after,
        n_removed,
        n_groups,
        group_sizes,
    })
}

/// Analyze the scene for near-duplicates and compute per-group statistics.
/// Does not remove any Gaussians.
pub fn gd_analyze_duplicates(
    positions: &[f32],
    opacities: &[f32],
    scales: &[f32],
    sh_coefficients: &[f32],
    sh_channels: usize,
    n_gaussians: usize,
    config: &DedupConfig,
) -> Result<Vec<DuplicateGroup>, DeduplicatorError> {
    if n_gaussians == 0 {
        return Err(DeduplicatorError::EmptyScene);
    }

    let groups = if config.use_spatial_hash {
        gd_find_duplicates_spatial(
            positions,
            opacities,
            scales,
            sh_coefficients,
            sh_channels,
            n_gaussians,
            config,
        )?
    } else {
        gd_find_duplicates_brute(
            positions,
            opacities,
            scales,
            sh_coefficients,
            sh_channels,
            n_gaussians,
            config,
        )?
    };

    let mut result = Vec::with_capacity(groups.len());
    for group in groups {
        let k = group.len() as f32;
        let mut centroid = [0.0f32; 3];
        let mut mean_opacity = 0.0f32;
        for &idx in &group {
            centroid[0] += positions[idx * 3];
            centroid[1] += positions[idx * 3 + 1];
            centroid[2] += positions[idx * 3 + 2];
            mean_opacity += opacities[idx];
        }
        centroid[0] /= k;
        centroid[1] /= k;
        centroid[2] /= k;
        mean_opacity /= k;

        // Max pairwise spread.
        let mut max_spread = 0.0f32;
        for ii in 0..group.len() {
            for jj in (ii + 1)..group.len() {
                let a = group[ii];
                let b = group[jj];
                let dx = positions[a * 3] - positions[b * 3];
                let dy = positions[a * 3 + 1] - positions[b * 3 + 1];
                let dz = positions[a * 3 + 2] - positions[b * 3 + 2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist > max_spread {
                    max_spread = dist;
                }
            }
        }

        result.push(DuplicateGroup {
            indices: group,
            centroid,
            mean_opacity,
            max_position_spread: max_spread,
        });
    }

    Ok(result)
}

/// Compute statistics from a deduplication result.
///
/// `bytes_per_gaussian = (3 + 4 + 3 + 1 + sh_channels) * 4`
pub fn gd_compute_stats(result: &DedupResult, sh_channels: usize) -> DedupStats {
    let reduction_percent = if result.n_before == 0 {
        0.0
    } else {
        result.n_removed as f32 / result.n_before as f32 * 100.0
    };

    // Real per-group sizes (was a placeholder formula reporting total member
    // count across every group as the "max", not the largest group's size).
    let (mean_group_size, max_group_size) = if result.group_sizes.is_empty() {
        (0.0f32, 0usize)
    } else {
        let total: usize = result.group_sizes.iter().sum();
        let mean = total as f32 / result.group_sizes.len() as f32;
        let max = result.group_sizes.iter().copied().max().unwrap_or(0);
        (mean, max)
    };

    let bytes_per_gaussian = (3 + 4 + 3 + 1 + sh_channels) * 4;
    let memory_saved_bytes = result.n_removed * bytes_per_gaussian;

    DedupStats {
        n_before: result.n_before,
        n_after: result.n_after,
        reduction_percent,
        n_groups: result.n_groups,
        mean_group_size,
        max_group_size,
        memory_saved_bytes,
    }
}

/// Format deduplication statistics as a human-readable string.
pub fn gd_format_stats(stats: &DedupStats) -> String {
    format!(
        concat!(
            "Deduplication Stats:\n",
            "  Gaussians: {} -> {} (removed {}; {:.1}% reduction)\n",
            "  Duplicate groups: {}\n",
            "  Mean group size: {:.2}\n",
            "  Max group size: {}\n",
            "  Memory saved: {} bytes ({:.1} KB)"
        ),
        stats.n_before,
        stats.n_after,
        stats.n_before - stats.n_after,
        stats.reduction_percent,
        stats.n_groups,
        stats.mean_group_size,
        stats.max_group_size,
        stats.memory_saved_bytes,
        stats.memory_saved_bytes as f64 / 1024.0,
    )
}

/// Build a full deduplication report, including the top-5 largest groups.
pub fn gd_build_report(
    result: &DedupResult,
    mut groups: Vec<DuplicateGroup>,
    config: &DedupConfig,
    sh_channels: usize,
) -> DedupReport {
    // Sort groups by descending size.
    groups.sort_by_key(|g| std::cmp::Reverse(g.indices.len()));
    let largest_groups = groups.into_iter().take(5).collect();

    let stats = gd_compute_stats(result, sh_channels);
    let config_summary = gd_format_config(config);

    DedupReport {
        stats,
        largest_groups,
        config_summary,
    }
}

/// Format a full deduplication report.
pub fn gd_format_report(report: &DedupReport) -> String {
    let mut s = String::new();
    s.push_str("=== OxiGAF Deduplication Report ===\n");
    s.push_str(&gd_format_stats(&report.stats));
    s.push('\n');
    s.push_str(&report.config_summary);
    s.push('\n');
    if report.largest_groups.is_empty() {
        s.push_str("No duplicate groups found.\n");
    } else {
        s.push_str(&format!(
            "Top {} group(s) by size:\n",
            report.largest_groups.len()
        ));
        for (i, g) in report.largest_groups.iter().enumerate() {
            s.push_str(&format!(
                "  Group {}: {} Gaussians, centroid=({:.4},{:.4},{:.4}), \
                 spread={:.6}, mean_opacity={:.4}\n",
                i + 1,
                g.indices.len(),
                g.centroid[0],
                g.centroid[1],
                g.centroid[2],
                g.max_position_spread,
                g.mean_opacity,
            ));
        }
    }
    s
}

/// Format config thresholds and policy as a human-readable string.
pub fn gd_format_config(config: &DedupConfig) -> String {
    let policy_str = match &config.keep_policy {
        DedupKeepPolicy::KeepHighestOpacity => "KeepHighestOpacity",
        DedupKeepPolicy::KeepLargestScale => "KeepLargestScale",
        DedupKeepPolicy::KeepSmallestScale => "KeepSmallestScale",
        DedupKeepPolicy::KeepFirst => "KeepFirst",
        DedupKeepPolicy::KeepLast => "KeepLast",
    };
    format!(
        "DedupConfig: pos_thresh={:.4} opacity_thresh={:.4} scale_thresh={:.4} \
         color_thresh={:.4} policy={} spatial_hash={} cell_size={:.4}",
        config.position_threshold,
        config.opacity_threshold,
        config.scale_threshold,
        config.color_threshold,
        policy_str,
        config.use_spatial_hash,
        config.cell_size,
    )
}
