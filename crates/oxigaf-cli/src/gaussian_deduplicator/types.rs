//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use thiserror::Error;

use super::functions::{gd_hash_cell, gd_world_to_cell};

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
/// Input scene data for [`gd_deduplicate`](crate::gaussian_deduplicator::gd_deduplicate).
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
        let mut bounds_min = [f32::MAX; 3];
        for i in 0..n {
            for d in 0..3 {
                let v = positions[i * 3 + d];
                if v < bounds_min[d] {
                    bounds_min[d] = v;
                }
            }
        }
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
/// Result of the deduplication pipeline.
#[derive(Debug)]
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
/// Statistics for a single duplicate group.
#[derive(Debug)]
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
/// Flat scene attribute slices passed to
/// [`gd_are_duplicates`](crate::gaussian_deduplicator::gd_are_duplicates).
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
/// Formatted report combining stats and top duplicate groups.
pub struct DedupReport {
    /// Aggregated statistics.
    pub stats: DedupStats,
    /// Top 5 duplicate groups by size.
    pub largest_groups: Vec<DuplicateGroup>,
    /// Human-readable summary of the config used.
    pub config_summary: String,
}
