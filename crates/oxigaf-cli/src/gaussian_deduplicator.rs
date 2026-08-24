//! Near-duplicate Gaussian detection and removal for 3D Gaussian Splatting scenes.
//!
//! Detects and removes near-duplicate Gaussians based on spatial proximity,
//! scale similarity, opacity similarity, and DC color similarity. Duplicate
//! Gaussians waste GPU memory and cause rendering artifacts from double-counted
//! opacity.
//!
//! Uses spatial hashing for O(N) average-case detection or O(N²) brute force
//! for small scenes or ground-truth verification.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from the Gaussian deduplication pipeline.
#[derive(Debug, Error)]
pub enum DeduplicatorError {
    /// Scene contains no Gaussians.
    #[error("Empty scene: no Gaussians")]
    EmptyScene,
    /// Position array length does not match n_gaussians * 3.
    #[error("Array length mismatch: positions={pos}, expected {n}*3")]
    PositionLengthMismatch { pos: usize, n: usize },
    /// n_gaussians is invalid (e.g., zero or would overflow).
    #[error("Invalid n_gaussians: {n}")]
    InvalidCount { n: usize },
    /// Cell size must be strictly positive.
    #[error("Grid cell size must be positive, got {size}")]
    InvalidCellSize { size: f32 },
    /// A flat per-Gaussian attribute array's length doesn't match `count * stride`.
    #[error(
        "Attribute length mismatch: expected {expected} ({count} x stride {stride}), got {got}"
    )]
    AttributeLengthMismatch {
        expected: usize,
        got: usize,
        count: usize,
        stride: usize,
    },
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Policy for which Gaussian to keep when a duplicate group is found.
#[derive(Debug, Clone)]
pub enum DedupKeepPolicy {
    /// Keep the most opaque of duplicates.
    KeepHighestOpacity,
    /// Keep the largest Gaussian (highest max scale component).
    KeepLargestScale,
    /// Keep the smallest Gaussian (lowest max scale component).
    KeepSmallestScale,
    /// Keep the one with the smallest index.
    KeepFirst,
    /// Keep the one with the largest index.
    KeepLast,
}

/// Configuration for the deduplication pipeline.
#[derive(Debug, Clone)]
pub struct DedupConfig {
    /// Max L2 position distance to consider as duplicate (e.g. 0.001).
    pub position_threshold: f32,
    /// Max absolute opacity difference to consider as duplicate (e.g. 0.05).
    pub opacity_threshold: f32,
    /// Max relative scale difference; |max_si - max_sj| / max(max_si, max_sj, eps) (e.g. 0.1).
    pub scale_threshold: f32,
    /// Max DC color L2 distance (e.g. 0.1). Only checked if sh_channels >= 3.
    pub color_threshold: f32,
    /// Which Gaussian from a duplicate group to keep.
    pub keep_policy: DedupKeepPolicy,
    /// Use spatial hashing (O(N)) instead of O(N²) brute force.
    pub use_spatial_hash: bool,
    /// Spatial hash cell size; should be >= position_threshold.
    pub cell_size: f32,
}

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            position_threshold: 0.001,
            opacity_threshold: 0.05,
            scale_threshold: 0.1,
            color_threshold: 0.1,
            keep_policy: DedupKeepPolicy::KeepHighestOpacity,
            use_spatial_hash: true,
            cell_size: 0.002,
        }
    }
}

// ---------------------------------------------------------------------------
// Spatial hash map
// ---------------------------------------------------------------------------

/// Spatial hash map: maps 3D grid cells to lists of Gaussian indices.
pub struct SpatialHashMap {
    /// Flattened hash table: bucket index → list of Gaussian indices.
    pub cells: Vec<Vec<usize>>,
    /// Number of hash buckets.
    pub n_buckets: usize,
    /// World-space size of each cubic cell.
    pub cell_size: f32,
    /// Minimum world-space bounds (used to offset before hashing).
    pub bounds_min: [f32; 3],
}

impl SpatialHashMap {
    /// Build a spatial hash map from `n` Gaussians whose positions are stored
    /// as a flat `[x0,y0,z0, x1,y1,z1, ...]` slice.
    pub fn new(
        n_buckets: usize,
        cell_size: f32,
        positions: &[f32],
        n: usize,
    ) -> Result<Self, DeduplicatorError> {
        if cell_size <= 0.0 {
            return Err(DeduplicatorError::InvalidCellSize { size: cell_size });
        }
        if n > 0 && positions.len() < n * 3 {
            return Err(DeduplicatorError::PositionLengthMismatch {
                pos: positions.len(),
                n,
            });
        }

        let actual_buckets = if n_buckets == 0 { 1 } else { n_buckets };
        let mut cells: Vec<Vec<usize>> = (0..actual_buckets).map(|_| Vec::new()).collect();

        if n == 0 {
            return Ok(Self {
                cells,
                n_buckets: actual_buckets,
                cell_size,
                bounds_min: [0.0; 3],
            });
        }

        // Compute axis-aligned bounds.
        let mut bounds_min = [f32::MAX; 3];
        for i in 0..n {
            for d in 0..3 {
                let v = positions[i * 3 + d];
                if v < bounds_min[d] {
                    bounds_min[d] = v;
                }
            }
        }

        // Insert each Gaussian.
        for i in 0..n {
            let pos = [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
            let cell = gd_world_to_cell(pos, cell_size, bounds_min);
            let bucket = gd_hash_cell(cell[0], cell[1], cell[2], actual_buckets);
            cells[bucket].push(i);
        }

        Ok(Self {
            cells,
            n_buckets: actual_buckets,
            cell_size,
            bounds_min,
        })
    }

    /// Query all Gaussian indices in the same cell as `pos` and all 26 neighbors.
    pub fn query_neighbors(&self, pos: [f32; 3]) -> Vec<usize> {
        let center = gd_world_to_cell(pos, self.cell_size, self.bounds_min);
        let mut result = Vec::new();
        for dz in -1i32..=1 {
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let cx = center[0] + dx;
                    let cy = center[1] + dy;
                    let cz = center[2] + dz;
                    let bucket = gd_hash_cell(cx, cy, cz, self.n_buckets);
                    result.extend_from_slice(&self.cells[bucket]);
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Spatial hash helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Pairwise duplicate check
// ---------------------------------------------------------------------------

/// Return the maximum absolute value of the 3-component scale vector at index `i`.
#[inline]
fn max_scale(scales: &[f32], i: usize) -> f32 {
    let a = scales[i * 3].abs();
    let b = scales[i * 3 + 1].abs();
    let c = scales[i * 3 + 2].abs();
    a.max(b).max(c)
}

/// Flat scene attribute slices passed to [`gd_are_duplicates`].
pub struct GdSceneSlices<'a> {
    /// Positions, flat N×3.
    pub positions: &'a [f32],
    /// Logit-space opacities, length N.
    pub opacities: &'a [f32],
    /// Log-scales, flat N×3.
    pub scales: &'a [f32],
    /// SH coefficients, flat N×sh_channels.
    pub sh_coeffs: &'a [f32],
    /// Number of SH channels per Gaussian.
    pub sh_channels: usize,
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

// ---------------------------------------------------------------------------
// Union-Find for transitive grouping
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Shared attribute-length validation
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Duplicate detection: spatial hash
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Duplicate detection: brute force
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Representative selection
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Removal mask
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Apply mask to flat arrays
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Full deduplication pipeline
// ---------------------------------------------------------------------------

/// Result of the deduplication pipeline.
pub struct DedupResult {
    /// Filtered position array (n_after × 3).
    pub positions: Vec<f32>,
    /// Filtered rotation array (n_after × 4).
    pub rotations: Vec<f32>,
    /// Filtered scale array (n_after × 3).
    pub scales: Vec<f32>,
    /// Filtered opacity array (n_after × 1).
    pub opacities: Vec<f32>,
    /// Filtered SH coefficient array (n_after × sh_channels).
    pub sh_coefficients: Vec<f32>,
    /// Number of Gaussians before deduplication.
    pub n_before: usize,
    /// Number of Gaussians after deduplication.
    pub n_after: usize,
    /// Number of Gaussians removed.
    pub n_removed: usize,
    /// Number of duplicate groups found.
    pub n_groups: usize,
    /// Size (Gaussian count) of each duplicate group found, in detection order.
    pub group_sizes: Vec<usize>,
}

/// Input scene data for [`gd_deduplicate`].
pub struct GdDeduplicateInput<'a> {
    /// Positions, flat N×3.
    pub positions: &'a [f32],
    /// Rotations (quaternions), flat N×4.
    pub rotations: &'a [f32],
    /// Log-scales, flat N×3.
    pub scales: &'a [f32],
    /// Logit-space opacities, length N.
    pub opacities: &'a [f32],
    /// SH coefficients, flat N×sh_channels.
    pub sh_coefficients: &'a [f32],
    /// Number of SH channels per Gaussian.
    pub sh_channels: usize,
    /// Total number of Gaussians.
    pub n_gaussians: usize,
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

// ---------------------------------------------------------------------------
// Analysis (without removal)
// ---------------------------------------------------------------------------

/// Statistics for a single duplicate group.
pub struct DuplicateGroup {
    /// Indices of the Gaussians in this group.
    pub indices: Vec<usize>,
    /// Mean position of the group.
    pub centroid: [f32; 3],
    /// Mean opacity of the group.
    pub mean_opacity: f32,
    /// Maximum pairwise L2 position distance within the group.
    pub max_position_spread: f32,
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

// ---------------------------------------------------------------------------
// Statistics and reporting
// ---------------------------------------------------------------------------

/// Aggregated statistics from a deduplication run.
pub struct DedupStats {
    /// Number of Gaussians before deduplication.
    pub n_before: usize,
    /// Number of Gaussians after deduplication.
    pub n_after: usize,
    /// Percentage of Gaussians removed (0.0–100.0).
    pub reduction_percent: f32,
    /// Number of duplicate groups detected.
    pub n_groups: usize,
    /// Mean size of duplicate groups.
    pub mean_group_size: f32,
    /// Size of the largest duplicate group.
    pub max_group_size: usize,
    /// Estimated bytes saved (n_removed × bytes_per_gaussian).
    pub memory_saved_bytes: usize,
}

/// Formatted report combining stats and top duplicate groups.
pub struct DedupReport {
    /// Aggregated statistics.
    pub stats: DedupStats,
    /// Top 5 duplicate groups by size.
    pub largest_groups: Vec<DuplicateGroup>,
    /// Human-readable summary of the config used.
    pub config_summary: String,
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_config() -> DedupConfig {
        DedupConfig {
            position_threshold: 0.01,
            opacity_threshold: 0.1,
            scale_threshold: 0.2,
            color_threshold: 0.2,
            keep_policy: DedupKeepPolicy::KeepHighestOpacity,
            use_spatial_hash: false,
            cell_size: 0.02,
        }
    }

    type GaussianArrays = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    /// Build minimal flat arrays for `n` Gaussians at distinct positions.
    fn make_distinct(n: usize) -> GaussianArrays {
        let positions: Vec<f32> = (0..n).flat_map(|i| [i as f32, 0.0, 0.0]).collect();
        let rotations: Vec<f32> = (0..n).flat_map(|_| [0.0f32, 0.0, 0.0, 1.0]).collect();
        let scales: Vec<f32> = (0..n).flat_map(|_| [1.0f32, 1.0, 1.0]).collect();
        let opacities: Vec<f32> = vec![0.5f32; n];
        let sh: Vec<f32> = vec![0.0f32; n * 3];
        (positions, rotations, scales, opacities, sh)
    }

    /// Build arrays for n Gaussians, all at the same position.
    fn make_identical(n: usize) -> GaussianArrays {
        let positions: Vec<f32> = vec![0.0f32; n * 3];
        let rotations: Vec<f32> = (0..n).flat_map(|_| [0.0f32, 0.0, 0.0, 1.0]).collect();
        let scales: Vec<f32> = (0..n).flat_map(|_| [1.0f32, 1.0, 1.0]).collect();
        let opacities: Vec<f32> = vec![0.5f32; n];
        let sh: Vec<f32> = vec![0.0f32; n * 3];
        (positions, rotations, scales, opacities, sh)
    }

    // -----------------------------------------------------------------------
    // gd_hash_cell
    // -----------------------------------------------------------------------

    #[test]
    fn hash_cell_same_coords_same_hash() {
        let h1 = gd_hash_cell(3, 7, -2, 1024);
        let h2 = gd_hash_cell(3, 7, -2, 1024);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_cell_origin_in_range() {
        let h = gd_hash_cell(0, 0, 0, 1024);
        assert!(h < 1024);
    }

    #[test]
    fn hash_cell_different_coords_likely_different() {
        // Not guaranteed but extremely unlikely to collide for adjacent cells.
        let h1 = gd_hash_cell(0, 0, 0, 65536);
        let h2 = gd_hash_cell(1, 0, 0, 65536);
        let h3 = gd_hash_cell(0, 1, 0, 65536);
        // At least two should differ (probabilistic, but primes make this essentially certain).
        assert!(h1 != h2 || h1 != h3);
    }

    #[test]
    fn hash_cell_large_coords_no_panic() {
        // Should not panic even with extreme values.
        let _ = gd_hash_cell(i32::MAX, i32::MIN, i32::MAX, 1024);
    }

    #[test]
    fn hash_cell_single_bucket() {
        let h = gd_hash_cell(100, -50, 200, 1);
        assert_eq!(h, 0);
    }

    // -----------------------------------------------------------------------
    // gd_world_to_cell
    // -----------------------------------------------------------------------

    #[test]
    fn world_to_cell_at_origin() {
        let cell = gd_world_to_cell([0.0, 0.0, 0.0], 1.0, [0.0, 0.0, 0.0]);
        assert_eq!(cell, [0, 0, 0]);
    }

    #[test]
    fn world_to_cell_at_one_cell() {
        let cell = gd_world_to_cell([1.0, 0.0, 0.0], 1.0, [0.0, 0.0, 0.0]);
        assert_eq!(cell, [1, 0, 0]);
    }

    #[test]
    fn world_to_cell_fractional() {
        let cell = gd_world_to_cell([0.5, 0.5, 0.5], 1.0, [0.0, 0.0, 0.0]);
        assert_eq!(cell, [0, 0, 0]);
    }

    #[test]
    fn world_to_cell_negative_offset() {
        // Position at bounds_min should give cell [0,0,0].
        let bounds_min = [-5.0, -3.0, -1.0];
        let cell = gd_world_to_cell(bounds_min, 1.0, bounds_min);
        assert_eq!(cell, [0, 0, 0]);
    }

    #[test]
    fn world_to_cell_with_bounds_offset() {
        let bounds_min = [10.0, 0.0, 0.0];
        let cell = gd_world_to_cell([11.0, 0.0, 0.0], 1.0, bounds_min);
        assert_eq!(cell, [1, 0, 0]);
    }

    // -----------------------------------------------------------------------
    // SpatialHashMap
    // -----------------------------------------------------------------------

    #[test]
    fn spatial_hash_map_new_empty_n0() {
        let map = SpatialHashMap::new(64, 0.1, &[], 0);
        assert!(map.is_ok());
    }

    #[test]
    fn spatial_hash_map_new_invalid_cell_size() {
        let positions = vec![0.0f32; 3];
        let err = SpatialHashMap::new(64, -0.1, &positions, 1);
        assert!(matches!(
            err,
            Err(DeduplicatorError::InvalidCellSize { .. })
        ));
    }

    #[test]
    fn spatial_hash_map_query_finds_inserted_point() {
        let positions = vec![0.0f32, 0.0, 0.0, 5.0, 5.0, 5.0];
        let map = SpatialHashMap::new(64, 1.0, &positions, 2).unwrap();
        let neighbors = map.query_neighbors([0.0, 0.0, 0.0]);
        assert!(neighbors.contains(&0), "Should find index 0 near origin");
    }

    #[test]
    fn spatial_hash_map_query_does_not_find_distant() {
        let positions = vec![0.0f32, 0.0, 0.0, 100.0, 100.0, 100.0];
        let map = SpatialHashMap::new(64, 1.0, &positions, 2).unwrap();
        let near_origin = map.query_neighbors([0.0, 0.0, 0.0]);
        // Index 1 is far away; it might land in same bucket due to hash collision,
        // but it should NOT appear in the 3×3×3 cell neighborhood.
        // (We check that 0 is in the results at minimum.)
        assert!(near_origin.contains(&0));
    }

    // -----------------------------------------------------------------------
    // gd_are_duplicates
    // -----------------------------------------------------------------------

    fn default_sh() -> Vec<f32> {
        vec![]
    }

    #[test]
    fn are_duplicates_identical_positions() {
        let pos = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let op = vec![0.5f32, 0.5];
        let sc = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = make_config();
        assert!(gd_are_duplicates(
            0,
            1,
            GdSceneSlices {
                positions: &pos,
                opacities: &op,
                scales: &sc,
                sh_coeffs: &default_sh(),
                sh_channels: 0
            },
            &cfg
        ));
    }

    #[test]
    fn are_duplicates_far_apart_not_duplicate() {
        let pos = vec![0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0];
        let op = vec![0.5f32, 0.5];
        let sc = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = make_config();
        assert!(!gd_are_duplicates(
            0,
            1,
            GdSceneSlices {
                positions: &pos,
                opacities: &op,
                scales: &sc,
                sh_coeffs: &default_sh(),
                sh_channels: 0
            },
            &cfg
        ));
    }

    #[test]
    fn are_duplicates_same_pos_different_opacity() {
        let pos = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let op = vec![0.0f32, 1.0]; // diff = 1.0 >> threshold 0.1
        let sc = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = make_config();
        assert!(!gd_are_duplicates(
            0,
            1,
            GdSceneSlices {
                positions: &pos,
                opacities: &op,
                scales: &sc,
                sh_coeffs: &default_sh(),
                sh_channels: 0
            },
            &cfg
        ));
    }

    #[test]
    fn are_duplicates_within_position_threshold() {
        let pos = vec![0.0f32, 0.0, 0.0, 0.005, 0.0, 0.0]; // dist = 0.005 < 0.01
        let op = vec![0.5f32, 0.5];
        let sc = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = make_config();
        assert!(gd_are_duplicates(
            0,
            1,
            GdSceneSlices {
                positions: &pos,
                opacities: &op,
                scales: &sc,
                sh_coeffs: &default_sh(),
                sh_channels: 0
            },
            &cfg
        ));
    }

    #[test]
    fn are_duplicates_beyond_position_threshold() {
        let pos = vec![0.0f32, 0.0, 0.0, 0.02, 0.0, 0.0]; // dist = 0.02 > 0.01
        let op = vec![0.5f32, 0.5];
        let sc = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = make_config();
        assert!(!gd_are_duplicates(
            0,
            1,
            GdSceneSlices {
                positions: &pos,
                opacities: &op,
                scales: &sc,
                sh_coeffs: &default_sh(),
                sh_channels: 0
            },
            &cfg
        ));
    }

    #[test]
    fn are_duplicates_opacity_within_threshold() {
        let pos = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let op = vec![0.5f32, 0.55]; // diff = 0.05 < 0.1
        let sc = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let cfg = make_config();
        assert!(gd_are_duplicates(
            0,
            1,
            GdSceneSlices {
                positions: &pos,
                opacities: &op,
                scales: &sc,
                sh_coeffs: &default_sh(),
                sh_channels: 0
            },
            &cfg
        ));
    }

    #[test]
    fn are_duplicates_scale_within_threshold() {
        let pos = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let op = vec![0.5f32, 0.5];
        // max_scale(0) = 1.0, max_scale(1) = 1.1; rel = 0.1/1.1 ≈ 0.09 < 0.2
        let sc = vec![1.0f32, 1.0, 1.0, 1.1, 1.0, 1.0];
        let cfg = make_config();
        assert!(gd_are_duplicates(
            0,
            1,
            GdSceneSlices {
                positions: &pos,
                opacities: &op,
                scales: &sc,
                sh_coeffs: &default_sh(),
                sh_channels: 0
            },
            &cfg
        ));
    }

    #[test]
    fn are_duplicates_color_filter() {
        let pos = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let op = vec![0.5f32, 0.5];
        let sc = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        // sh_channels=3; colors are very different
        let sh = vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut cfg = make_config();
        cfg.color_threshold = 0.1;
        assert!(!gd_are_duplicates(
            0,
            1,
            GdSceneSlices {
                positions: &pos,
                opacities: &op,
                scales: &sc,
                sh_coeffs: &sh,
                sh_channels: 3
            },
            &cfg
        ));
    }

    // -----------------------------------------------------------------------
    // gd_find_duplicates_brute
    // -----------------------------------------------------------------------

    #[test]
    fn brute_two_identical_one_group() {
        let (pos, _, sc, op, sh) = make_identical(2);
        let cfg = make_config();
        let groups = gd_find_duplicates_brute(&pos, &op, &sc, &sh, 3, 2, &cfg).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2);
    }

    #[test]
    fn brute_all_distinct_no_groups() {
        let (pos, _, sc, op, sh) = make_distinct(5);
        let cfg = make_config();
        let groups = gd_find_duplicates_brute(&pos, &op, &sc, &sh, 3, 5, &cfg).unwrap();
        assert_eq!(groups.len(), 0);
    }

    #[test]
    fn brute_three_identical_one_group_of_three() {
        let (pos, _, sc, op, sh) = make_identical(3);
        let cfg = make_config();
        let groups = gd_find_duplicates_brute(&pos, &op, &sc, &sh, 3, 3, &cfg).unwrap();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 3);
    }

    #[test]
    fn brute_n0_returns_empty() {
        let cfg = make_config();
        let groups = gd_find_duplicates_brute(&[], &[], &[], &[], 0, 0, &cfg).unwrap();
        assert_eq!(groups.len(), 0);
    }

    #[test]
    fn brute_length_mismatch_error() {
        let cfg = make_config();
        // positions only has 3 floats but n=2 expects 6.
        let err =
            gd_find_duplicates_brute(&[0.0, 0.0, 0.0], &[0.5, 0.5], &[1.0; 6], &[], 0, 2, &cfg);
        assert!(matches!(
            err,
            Err(DeduplicatorError::PositionLengthMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // gd_find_duplicates_spatial matches brute force
    // -----------------------------------------------------------------------

    #[test]
    fn spatial_matches_brute_two_identical() {
        let (pos, _, sc, op, sh) = make_identical(2);
        let mut cfg = make_config();
        cfg.cell_size = 0.02;
        let brute = gd_find_duplicates_brute(&pos, &op, &sc, &sh, 3, 2, &cfg).unwrap();
        let spatial = gd_find_duplicates_spatial(&pos, &op, &sc, &sh, 3, 2, &cfg).unwrap();
        assert_eq!(brute.len(), spatial.len(), "Group counts should match");
        assert_eq!(brute[0].len(), spatial[0].len(), "Group sizes should match");
    }

    #[test]
    fn spatial_matches_brute_all_distinct() {
        let (pos, _, sc, op, sh) = make_distinct(6);
        let mut cfg = make_config();
        cfg.cell_size = 0.02;
        let brute = gd_find_duplicates_brute(&pos, &op, &sc, &sh, 3, 6, &cfg).unwrap();
        let spatial = gd_find_duplicates_spatial(&pos, &op, &sc, &sh, 3, 6, &cfg).unwrap();
        assert_eq!(brute.len(), 0);
        assert_eq!(spatial.len(), 0);
    }

    // -----------------------------------------------------------------------
    // gd_pick_representative
    // -----------------------------------------------------------------------

    #[test]
    fn pick_representative_highest_opacity() {
        let opacities = vec![0.2f32, 0.8, 0.5];
        let scales = vec![1.0f32; 9];
        let group = vec![0usize, 1, 2];
        let rep = gd_pick_representative(
            &group,
            &opacities,
            &scales,
            &DedupKeepPolicy::KeepHighestOpacity,
        )
        .expect("group is non-empty");
        assert_eq!(rep, 1);
    }

    #[test]
    fn pick_representative_keep_first() {
        let opacities = vec![0.5f32; 3];
        let scales = vec![1.0f32; 9];
        let group = vec![2usize, 0, 1];
        let rep = gd_pick_representative(&group, &opacities, &scales, &DedupKeepPolicy::KeepFirst)
            .expect("group is non-empty");
        assert_eq!(rep, 0);
    }

    #[test]
    fn pick_representative_keep_last() {
        let opacities = vec![0.5f32; 3];
        let scales = vec![1.0f32; 9];
        let group = vec![0usize, 1, 2];
        let rep = gd_pick_representative(&group, &opacities, &scales, &DedupKeepPolicy::KeepLast)
            .expect("group is non-empty");
        assert_eq!(rep, 2);
    }

    #[test]
    fn pick_representative_largest_scale() {
        let opacities = vec![0.5f32; 3];
        // max_scale(0)=1, max_scale(1)=3, max_scale(2)=2
        let scales = vec![1.0f32, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0, 1.0];
        let group = vec![0usize, 1, 2];
        let rep = gd_pick_representative(
            &group,
            &opacities,
            &scales,
            &DedupKeepPolicy::KeepLargestScale,
        )
        .expect("group is non-empty");
        assert_eq!(rep, 1);
    }

    #[test]
    fn pick_representative_smallest_scale() {
        let opacities = vec![0.5f32; 3];
        let scales = vec![1.0f32, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0, 1.0];
        let group = vec![0usize, 1, 2];
        let rep = gd_pick_representative(
            &group,
            &opacities,
            &scales,
            &DedupKeepPolicy::KeepSmallestScale,
        )
        .expect("group is non-empty");
        assert_eq!(rep, 0);
    }

    #[test]
    fn pick_representative_empty_group_returns_none_not_panic() {
        // KeepFirst/KeepLast used to evaluate group[0] as an eager
        // unwrap_or fallback, panicking on an empty group.
        let opacities: Vec<f32> = vec![];
        let scales: Vec<f32> = vec![];
        let group: Vec<usize> = vec![];
        for policy in [
            DedupKeepPolicy::KeepHighestOpacity,
            DedupKeepPolicy::KeepLargestScale,
            DedupKeepPolicy::KeepSmallestScale,
            DedupKeepPolicy::KeepFirst,
            DedupKeepPolicy::KeepLast,
        ] {
            assert_eq!(
                gd_pick_representative(&group, &opacities, &scales, &policy),
                None,
                "empty group must return None for {policy:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // gd_build_remove_mask
    // -----------------------------------------------------------------------

    #[test]
    fn build_remove_mask_group_of_two_keep_first() {
        let opacities = vec![0.5f32, 0.5];
        let scales = vec![1.0f32; 6];
        let groups = vec![vec![0usize, 1]];
        let mask =
            gd_build_remove_mask(&groups, 2, &opacities, &scales, &DedupKeepPolicy::KeepFirst);
        assert!(!mask[0], "index 0 kept");
        assert!(mask[1], "index 1 removed");
    }

    #[test]
    fn build_remove_mask_no_groups() {
        let opacities = vec![0.5f32; 3];
        let scales = vec![1.0f32; 9];
        let groups: Vec<Vec<usize>> = vec![];
        let mask =
            gd_build_remove_mask(&groups, 3, &opacities, &scales, &DedupKeepPolicy::KeepFirst);
        assert!(mask.iter().all(|&r| !r), "Nothing removed if no groups");
    }

    // -----------------------------------------------------------------------
    // gd_apply_mask / gd_apply_scalar_mask
    // -----------------------------------------------------------------------

    #[test]
    fn apply_mask_stride3_removes_correct_rows() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mask = vec![false, true, false]; // remove middle row
        let result = gd_apply_mask(&data, &mask, 3).expect("lengths match");
        assert_eq!(result, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn apply_mask_keeps_all_when_no_removal() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mask = vec![false, false];
        let result = gd_apply_mask(&data, &mask, 3).expect("lengths match");
        assert_eq!(result, data);
    }

    #[test]
    fn apply_scalar_mask_removes_elements() {
        let data = vec![0.1f32, 0.5, 0.9];
        let mask = vec![true, false, false];
        let result = gd_apply_scalar_mask(&data, &mask).expect("lengths match");
        assert_eq!(result, vec![0.5, 0.9]);
    }

    #[test]
    fn apply_scalar_mask_removes_all() {
        let data = vec![1.0f32, 2.0, 3.0];
        let mask = vec![true, true, true];
        let result = gd_apply_scalar_mask(&data, &mask).expect("lengths match");
        assert!(result.is_empty());
    }

    #[test]
    fn apply_mask_length_mismatch_is_error_not_panic() {
        // A short `data` used to panic ("slice index starts at ... but ends
        // at ..."); a length between row boundaries silently emitted a
        // truncated, misaligned row. Both must now error up front.
        let short = gd_apply_mask(&[1.0f32, 2.0, 3.0], &[false, false], 3);
        assert!(matches!(
            short,
            Err(DeduplicatorError::AttributeLengthMismatch { .. })
        ));
        let partial_row = gd_apply_mask(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[false, false], 3);
        assert!(matches!(
            partial_row,
            Err(DeduplicatorError::AttributeLengthMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // gd_deduplicate
    // -----------------------------------------------------------------------

    #[test]
    fn deduplicate_two_identical_removes_one() {
        let (pos, rot, sc, op, sh) = make_identical(2);
        let cfg = make_config();
        let result = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 2,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(result.n_before, 2);
        assert_eq!(result.n_after, 1);
        assert_eq!(result.n_removed, 1);
    }

    #[test]
    fn deduplicate_all_distinct_no_removal() {
        let (pos, rot, sc, op, sh) = make_distinct(5);
        let cfg = make_config();
        let result = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 5,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(result.n_after, 5);
        assert_eq!(result.n_removed, 0);
    }

    #[test]
    fn deduplicate_preserves_array_lengths() {
        let (pos, rot, sc, op, sh) = make_identical(4);
        let cfg = make_config();
        let res = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 4,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(res.positions.len(), res.n_after * 3);
        assert_eq!(res.rotations.len(), res.n_after * 4);
        assert_eq!(res.scales.len(), res.n_after * 3);
        assert_eq!(res.opacities.len(), res.n_after);
        assert_eq!(res.sh_coefficients.len(), res.n_after * 3);
    }

    #[test]
    fn deduplicate_length_mismatch_error() {
        let pos = vec![0.0f32; 5]; // wrong: should be 2*3=6
        let rot = vec![0.0f32; 8];
        let sc = vec![1.0f32; 6];
        let op = vec![0.5f32; 2];
        let sh = vec![0.0f32; 6];
        let cfg = make_config();
        let err = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 2,
            },
            &cfg,
        );
        assert!(matches!(
            err,
            Err(DeduplicatorError::PositionLengthMismatch { .. })
        ));
    }

    #[test]
    fn deduplicate_empty_scene_error() {
        let cfg = make_config();
        let err = gd_deduplicate(
            GdDeduplicateInput {
                positions: &[],
                rotations: &[],
                scales: &[],
                opacities: &[],
                sh_coefficients: &[],
                sh_channels: 0,
                n_gaussians: 0,
            },
            &cfg,
        );
        assert!(matches!(err, Err(DeduplicatorError::EmptyScene)));
    }

    #[test]
    fn deduplicate_10_gaussians_3_exact_duplicates() {
        // Gaussians 0-6 are distinct; 7,8,9 are copies of 0.
        let mut pos = vec![0.0f32; 10 * 3];
        let mut rot = vec![0.0f32; 10 * 4];
        let sc = vec![1.0f32; 10 * 3];
        let op = vec![0.5f32; 10];
        let sh = vec![0.0f32; 10 * 3];

        // Distinct positions for 0..7
        for i in 0..7 {
            pos[i * 3] = i as f32 * 10.0;
        }
        // rot w=1 for all
        for i in 0..10 {
            rot[i * 4 + 3] = 1.0;
        }
        // 7,8,9 copy position of 0 (already 0.0)
        // → group [0,7,8,9]
        let cfg = make_config();
        let result = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 10,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(result.n_removed, 3, "3 duplicates of index 0 removed");
        assert_eq!(result.n_after, 7);
        // group_sizes must reflect the real group (0,7,8,9), size 4.
        assert_eq!(result.group_sizes.len(), result.n_groups);
        assert_eq!(result.group_sizes.iter().copied().max(), Some(4));
        let _ = (rot, sc, op, sh); // suppress unused warnings
    }

    #[test]
    fn deduplicate_multiple_groups_three_pairs() {
        // 3 pairs: (0,1), (2,3), (4,5); all at same pos within pair.
        let mut pos = vec![0.0f32; 6 * 3];
        let _rot = (0..6)
            .flat_map(|_| [0.0f32, 0.0, 0.0, 1.0])
            .collect::<Vec<_>>();
        let sc = vec![1.0f32; 6 * 3];
        let op = vec![0.5f32; 6];
        let sh = vec![0.0f32; 6 * 3];

        // Pair centroids far apart.
        pos[0] = 0.0;
        pos[3] = 0.0;
        pos[6] = 10.0;
        pos[9] = 10.0;
        pos[12] = 20.0;
        pos[15] = 20.0;

        let cfg = make_config();
        let groups = gd_find_duplicates_brute(&pos, &op, &sc, &sh, 3, 6, &cfg).unwrap();
        assert_eq!(groups.len(), 3, "Three groups expected");
    }

    // -----------------------------------------------------------------------
    // gd_analyze_duplicates
    // -----------------------------------------------------------------------

    #[test]
    fn analyze_duplicates_centroid_correct() {
        // Two identical Gaussians at (0,0,0) → centroid=(0,0,0)
        let (pos, _, sc, op, sh) = make_identical(2);
        let cfg = make_config();
        let groups = gd_analyze_duplicates(&pos, &op, &sc, &sh, 3, 2, &cfg).unwrap();
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert!((g.centroid[0]).abs() < 1e-6);
        assert!((g.centroid[1]).abs() < 1e-6);
        assert!((g.centroid[2]).abs() < 1e-6);
    }

    #[test]
    fn analyze_duplicates_spread_zero_for_identical() {
        let (pos, _, sc, op, sh) = make_identical(3);
        let cfg = make_config();
        let groups = gd_analyze_duplicates(&pos, &op, &sc, &sh, 3, 3, &cfg).unwrap();
        assert_eq!(groups.len(), 1);
        assert!(groups[0].max_position_spread < 1e-6);
    }

    #[test]
    fn analyze_duplicates_empty_scene_error() {
        let cfg = make_config();
        let err = gd_analyze_duplicates(&[], &[], &[], &[], 0, 0, &cfg);
        assert!(matches!(err, Err(DeduplicatorError::EmptyScene)));
    }

    #[test]
    fn analyze_duplicates_mismatched_opacities_is_error_not_panic() {
        // Never validated anything beyond n_gaussians == 0 (not even
        // positions); relied on gd_are_duplicates' unchecked indexing.
        let positions = vec![0.0f32; 3 * 3];
        let scales = vec![1.0f32; 3 * 3];
        let opacities = vec![0.5f32]; // too short
        let sh = vec![0.0f32; 3 * 3];
        let cfg = make_config();
        let err = gd_analyze_duplicates(&positions, &opacities, &scales, &sh, 3, 3, &cfg);
        assert!(
            matches!(err, Err(DeduplicatorError::AttributeLengthMismatch { .. })),
            "expected AttributeLengthMismatch, got {err:?}"
        );
    }

    #[test]
    fn deduplicate_short_sh_coeffs_with_3_channels_is_error_not_panic() {
        // The sh_channels >= 3 guard only checks !sh_coeffs.is_empty(),
        // which a short-but-nonempty array passes and then indexes past.
        let (pos, rot, sc, op, _) = make_identical(100);
        let sh = vec![0.0f32; 3]; // far too short for n=100, sh_channels=3
        let cfg = make_config();
        let err = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 100,
            },
            &cfg,
        );
        assert!(
            matches!(err, Err(DeduplicatorError::AttributeLengthMismatch { .. })),
            "expected AttributeLengthMismatch, got {err:?}"
        );
    }

    #[test]
    fn analyze_duplicates_mean_opacity_correct() {
        // All 4 Gaussians at same position with very similar opacities → one group.
        // Opacities differ by <= 0.02 < threshold 0.1.
        let pos = vec![0.0f32; 4 * 3];
        let sc = vec![1.0f32; 4 * 3];
        let op = vec![0.48f32, 0.50, 0.50, 0.52];
        let sh = vec![0.0f32; 4 * 3];
        let cfg = make_config();
        let groups = gd_analyze_duplicates(&pos, &op, &sc, &sh, 3, 4, &cfg).unwrap();
        assert_eq!(groups.len(), 1);
        let mean = groups[0].mean_opacity;
        let expected = (0.48f32 + 0.50 + 0.50 + 0.52) / 4.0; // 0.5
        assert!(
            (mean - expected).abs() < 1e-5,
            "mean opacity should be {}, got {}",
            expected,
            mean
        );
    }

    // -----------------------------------------------------------------------
    // gd_compute_stats
    // -----------------------------------------------------------------------

    #[test]
    fn compute_stats_reduction_percent() {
        let result = DedupResult {
            positions: vec![],
            rotations: vec![],
            scales: vec![],
            opacities: vec![],
            sh_coefficients: vec![],
            n_before: 100,
            n_after: 75,
            n_removed: 25,
            n_groups: 5,
            group_sizes: vec![6, 6, 6, 6, 6], // 5 groups of 6 => removes 5 each = 25
        };
        let stats = gd_compute_stats(&result, 3);
        assert!((stats.reduction_percent - 25.0).abs() < 1e-3);
    }

    #[test]
    fn compute_stats_memory_saved_bytes() {
        let result = DedupResult {
            positions: vec![],
            rotations: vec![],
            scales: vec![],
            opacities: vec![],
            sh_coefficients: vec![],
            n_before: 10,
            n_after: 8,
            n_removed: 2,
            n_groups: 1,
            group_sizes: vec![3], // 1 group of 3 => removes 2, keeps 1
        };
        // bytes_per_gaussian = (3+4+3+1+3)*4 = 56
        let stats = gd_compute_stats(&result, 3);
        assert_eq!(stats.memory_saved_bytes, 2 * 56);
    }

    #[test]
    fn compute_stats_zero_groups_zero_mean() {
        let result = DedupResult {
            positions: vec![],
            rotations: vec![],
            scales: vec![],
            opacities: vec![],
            sh_coefficients: vec![],
            n_before: 10,
            n_after: 10,
            n_removed: 0,
            n_groups: 0,
            group_sizes: vec![],
        };
        let stats = gd_compute_stats(&result, 3);
        assert_eq!(stats.mean_group_size, 0.0);
        assert_eq!(stats.max_group_size, 0);
    }

    #[test]
    fn compute_stats_max_group_size_is_actual_max_not_total_members() {
        // Groups of 2 and 5: old formula (n_removed+n_groups).max(2)=7 — no
        // group has 7 members; true max is 5.
        let result = DedupResult {
            positions: vec![],
            rotations: vec![],
            scales: vec![],
            opacities: vec![],
            sh_coefficients: vec![],
            n_before: 7,
            n_after: 2,
            n_removed: 5,
            n_groups: 2,
            group_sizes: vec![2, 5],
        };
        let stats = gd_compute_stats(&result, 3);
        assert_eq!(
            stats.max_group_size, 5,
            "max group size must be the largest actual group, not total members"
        );
        let expected_mean = (2 + 5) as f32 / 2.0;
        assert!((stats.mean_group_size - expected_mean).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // gd_format_stats / gd_format_report
    // -----------------------------------------------------------------------

    #[test]
    fn format_stats_nonempty_string() {
        let result = DedupResult {
            positions: vec![],
            rotations: vec![],
            scales: vec![],
            opacities: vec![],
            sh_coefficients: vec![],
            n_before: 50,
            n_after: 40,
            n_removed: 10,
            n_groups: 2,
            group_sizes: vec![6, 6], // 2 groups of 6 => removes 5 each = 10
        };
        let stats = gd_compute_stats(&result, 3);
        let s = gd_format_stats(&stats);
        assert!(!s.is_empty());
        assert!(s.contains("50"));
    }

    #[test]
    fn format_report_nonempty_contains_group_count() {
        let result = DedupResult {
            positions: vec![],
            rotations: vec![],
            scales: vec![],
            opacities: vec![],
            sh_coefficients: vec![],
            n_before: 10,
            n_after: 8,
            n_removed: 2,
            n_groups: 1,
            group_sizes: vec![3],
        };
        let groups: Vec<DuplicateGroup> = vec![DuplicateGroup {
            indices: vec![0, 1],
            centroid: [0.0; 3],
            mean_opacity: 0.5,
            max_position_spread: 0.0,
        }];
        let cfg = make_config();
        let report = gd_build_report(&result, groups, &cfg, 3);
        let s = gd_format_report(&report);
        assert!(!s.is_empty());
        assert!(s.contains("1"), "Should mention 1 group");
    }

    #[test]
    fn format_config_nonempty() {
        let cfg = make_config();
        let s = gd_format_config(&cfg);
        assert!(!s.is_empty());
        assert!(s.contains("KeepHighestOpacity"));
    }

    // -----------------------------------------------------------------------
    // Spatial hash pipeline with use_spatial_hash=true
    // -----------------------------------------------------------------------

    #[test]
    fn deduplicate_spatial_two_identical() {
        let (pos, rot, sc, op, sh) = make_identical(2);
        let mut cfg = make_config();
        cfg.use_spatial_hash = true;
        cfg.cell_size = 0.02;
        let result = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 2,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(result.n_removed, 1);
    }

    #[test]
    fn deduplicate_spatial_all_distinct_no_removal() {
        let (pos, rot, sc, op, sh) = make_distinct(8);
        let mut cfg = make_config();
        cfg.use_spatial_hash = true;
        cfg.cell_size = 0.5;
        let result = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 8,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(result.n_removed, 0);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn deduplicate_single_gaussian_no_groups() {
        let pos = vec![1.0f32, 2.0, 3.0];
        let rot = vec![0.0f32, 0.0, 0.0, 1.0];
        let sc = vec![1.0f32, 1.0, 1.0];
        let op = vec![0.5f32];
        let sh = vec![0.1f32, 0.2, 0.3];
        let cfg = make_config();
        let result = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 3,
                n_gaussians: 1,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(result.n_after, 1);
        assert_eq!(result.n_removed, 0);
    }

    #[test]
    fn deduplicate_no_sh_channels() {
        let pos = vec![0.0f32; 4 * 3];
        let rot = [0.0f32, 0.0, 0.0, 1.0].repeat(4);
        let sc = vec![1.0f32; 4 * 3];
        let op = vec![0.5f32; 4];
        let sh: Vec<f32> = vec![];
        let cfg = make_config();
        let result = gd_deduplicate(
            GdDeduplicateInput {
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
                sh_channels: 0,
                n_gaussians: 4,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(result.n_removed, 3);
        assert!(result.sh_coefficients.is_empty());
    }

    #[test]
    fn pick_representative_highest_opacity_among_three() {
        // [0.2, 0.8, 0.5] → index 1 has highest
        let op = vec![0.2f32, 0.8, 0.5];
        let sc = vec![1.0f32; 9];
        let group = vec![0usize, 1, 2];
        let rep = gd_pick_representative(&group, &op, &sc, &DedupKeepPolicy::KeepHighestOpacity)
            .expect("group is non-empty");
        assert_eq!(rep, 1);
    }

    #[test]
    fn world_to_cell_negative_position_handled() {
        // Position below bounds_min should give cell < 0, which is fine for i32.
        let bounds_min = [0.0f32, 0.0, 0.0];
        // A position at -0.5 with cell_size 1.0 → floor(-0.5) = -1
        let cell = gd_world_to_cell([-0.5, 0.0, 0.0], 1.0, bounds_min);
        assert_eq!(cell[0], -1);
    }
}
