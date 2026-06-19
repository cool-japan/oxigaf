//! Level-of-Detail (LOD) generation for 3D Gaussian Splatting clouds.
//!
//! This module creates multiple resolution variants of a Gaussian scene
//! that can be selected based on viewing distance. Lower LOD levels have
//! fewer Gaussians but remain perceptually similar by retaining the most
//! opaque/visible Gaussians.
//!
//! # Example
//! ```rust
//! use oxigaf_cli::lod_generator::{LodConfig, LodStrategy, generate_lod_chain};
//!
//! let n = 100usize;
//! let positions: Vec<f32> = (0..n * 3).map(|i| i as f32 * 0.01).collect();
//! let rotations: Vec<f32> = (0..n * 4)
//!     .map(|i| if i % 4 == 3 { 1.0 } else { 0.0 })
//!     .collect();
//! let scales: Vec<f32> = vec![0.1f32; n * 3];
//! let opacities: Vec<f32> = vec![0.5f32; n];
//! let sh_coefficients: Vec<f32> = vec![0.0f32; n * 9];
//!
//! let config = LodConfig::default();
//! let chain = generate_lod_chain(
//!     &positions, &rotations, &scales, &opacities, &sh_coefficients, &config,
//! ).expect("LOD generation failed");
//! println!("Generated {} LOD levels", chain.levels.len());
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// LodError
// ---------------------------------------------------------------------------

/// Errors that can occur during LOD generation.
#[derive(Debug, Error)]
pub enum LodError {
    /// The Gaussian cloud has no positions.
    #[error("Empty Gaussian cloud: no positions")]
    EmptyCloud,

    /// The positions array length is not divisible by 3.
    #[error("Positions length {len} is not divisible by 3")]
    InvalidPositionLength { len: usize },

    /// An array length does not match the number of Gaussians implied by positions.
    #[error(
        "Array length mismatch: positions imply {n_gaussians} Gaussians, \
         but {field} has {actual} elements"
    )]
    ArrayLengthMismatch {
        n_gaussians: usize,
        field: String,
        actual: usize,
    },

    /// LOD level index is out of range.
    #[error("Invalid LOD level {level}: must be < n_levels ({n_levels})")]
    InvalidLodLevel { level: usize, n_levels: usize },

    /// A reduction ratio is not in the range (0, 1].
    #[error("LOD reduction ratio {ratio} out of range (0, 1)")]
    InvalidReductionRatio { ratio: f32 },

    /// Not enough Gaussians to satisfy the request.
    #[error("Requested {k} Gaussians but cloud only has {n}")]
    InsufficientGaussians { k: usize, n: usize },
}

// ---------------------------------------------------------------------------
// LodStrategy
// ---------------------------------------------------------------------------

/// Strategy for selecting Gaussians in lower LOD levels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LodStrategy {
    /// Keep highest-opacity Gaussians (simplest, best quality).
    TopOpacity,
    /// Keep uniformly spaced Gaussians (by index after sorting).
    Uniform,
    /// Keep spatially distributed Gaussians using grid sampling.
    SpatialGrid,
    /// Random selection (uses deterministic xorshift64).
    Random,
}

// ---------------------------------------------------------------------------
// LodConfig
// ---------------------------------------------------------------------------

/// Configuration for LOD generation.
#[derive(Debug, Clone)]
pub struct LodConfig {
    /// Number of LOD levels.
    pub n_levels: usize,
    /// Fraction of Gaussians per level (must be descending, each in (0, 1]).
    pub reduction_ratios: Vec<f32>,
    /// Selection strategy for lower-quality levels.
    pub strategy: LodStrategy,
    /// Sort Gaussians by opacity before selection.
    pub sort_by_opacity: bool,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            n_levels: 4,
            reduction_ratios: vec![1.0, 0.5, 0.25, 0.1],
            strategy: LodStrategy::TopOpacity,
            sort_by_opacity: true,
        }
    }
}

impl LodConfig {
    /// Validate this configuration, returning an error on the first problem found.
    pub fn validate(&self) -> Result<(), LodError> {
        if self.reduction_ratios.len() != self.n_levels {
            return Err(LodError::InvalidReductionRatio {
                ratio: self.reduction_ratios.first().copied().unwrap_or(0.0),
            });
        }
        for &ratio in &self.reduction_ratios {
            if ratio <= 0.0 || ratio > 1.0 {
                return Err(LodError::InvalidReductionRatio { ratio });
            }
        }
        // Ratios must be non-ascending.
        for i in 1..self.reduction_ratios.len() {
            if self.reduction_ratios[i] > self.reduction_ratios[i - 1] {
                return Err(LodError::InvalidReductionRatio {
                    ratio: self.reduction_ratios[i],
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LodLevel
// ---------------------------------------------------------------------------

/// A single LOD level containing a subset of the original Gaussian cloud.
#[derive(Debug, Clone)]
pub struct LodLevel {
    /// 0 = highest quality (full resolution).
    pub level: usize,
    /// Number of Gaussians at this level.
    pub n_gaussians: usize,
    /// Fraction of original (1.0 = full, 0.5 = half).
    pub reduction_factor: f32,
    /// Flat positions array: n_gaussians × 3.
    pub positions: Vec<f32>,
    /// Flat rotations array (quaternions): n_gaussians × 4.
    pub rotations: Vec<f32>,
    /// Flat log-scale array: n_gaussians × 3.
    pub scales: Vec<f32>,
    /// Logit-space opacities: n_gaussians.
    pub opacities: Vec<f32>,
    /// Spherical harmonics coefficients: n_gaussians × C.
    pub sh_coefficients: Vec<f32>,
}

impl LodLevel {
    /// Total parameters per Gaussian at this level.
    #[must_use]
    pub fn n_params_per_gaussian(&self) -> usize {
        let sh_per = self
            .sh_coefficients
            .len()
            .checked_div(self.n_gaussians)
            .unwrap_or(0);
        3 + 4 + 3 + 1 + sh_per
    }

    /// Validate that all arrays are consistent with `n_gaussians`.
    pub fn validate(&self) -> Result<(), LodError> {
        if self.positions.len() != self.n_gaussians * 3 {
            return Err(LodError::ArrayLengthMismatch {
                n_gaussians: self.n_gaussians,
                field: "positions".to_string(),
                actual: self.positions.len(),
            });
        }
        if self.rotations.len() != self.n_gaussians * 4 {
            return Err(LodError::ArrayLengthMismatch {
                n_gaussians: self.n_gaussians,
                field: "rotations".to_string(),
                actual: self.rotations.len(),
            });
        }
        if self.scales.len() != self.n_gaussians * 3 {
            return Err(LodError::ArrayLengthMismatch {
                n_gaussians: self.n_gaussians,
                field: "scales".to_string(),
                actual: self.scales.len(),
            });
        }
        if self.opacities.len() != self.n_gaussians {
            return Err(LodError::ArrayLengthMismatch {
                n_gaussians: self.n_gaussians,
                field: "opacities".to_string(),
                actual: self.opacities.len(),
            });
        }
        if self.n_gaussians > 0 && !self.sh_coefficients.len().is_multiple_of(self.n_gaussians) {
            return Err(LodError::ArrayLengthMismatch {
                n_gaussians: self.n_gaussians,
                field: "sh_coefficients".to_string(),
                actual: self.sh_coefficients.len(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LodChain
// ---------------------------------------------------------------------------

/// A chain of LOD levels from high to low quality.
#[derive(Debug, Clone)]
pub struct LodChain {
    /// Ordered from level 0 (full quality) to level N-1 (lowest quality).
    pub levels: Vec<LodLevel>,
    /// Total Gaussians in the original (level 0) cloud.
    pub original_n_gaussians: usize,
}

impl LodChain {
    /// Select the appropriate LOD level based on viewing distance.
    ///
    /// - If `distance < thresholds[0]`: level 0 (highest quality)
    /// - If `distance < thresholds[i]`: level i
    /// - If beyond all thresholds: lowest level
    pub fn select(&self, distance: f32, selector: &LodSelector) -> Result<&LodLevel, LodError> {
        if self.levels.is_empty() {
            return Err(LodError::EmptyCloud);
        }
        for (i, &threshold) in selector.thresholds.iter().enumerate() {
            if distance < threshold {
                let level_idx = i.min(self.levels.len() - 1);
                return Ok(&self.levels[level_idx]);
            }
        }
        // Beyond all thresholds → lowest quality level.
        Ok(&self.levels[self.levels.len() - 1])
    }
}

// ---------------------------------------------------------------------------
// LodSelector
// ---------------------------------------------------------------------------

/// Parameters for LOD selection based on viewing distance.
#[derive(Debug, Clone)]
pub struct LodSelector {
    /// Distance thresholds for each LOD switch.
    pub thresholds: Vec<f32>,
}

impl Default for LodSelector {
    fn default() -> Self {
        Self {
            thresholds: vec![0.5, 2.0, 5.0],
        }
    }
}

impl LodSelector {
    /// Create a selector with given distance thresholds.
    pub fn new(thresholds: Vec<f32>) -> Self {
        Self { thresholds }
    }
}

// ---------------------------------------------------------------------------
// LodStats
// ---------------------------------------------------------------------------

/// Statistics about an LOD chain.
#[derive(Debug, Clone)]
pub struct LodStats {
    /// Number of levels.
    pub n_levels: usize,
    /// Gaussian count of the original (full-quality) cloud.
    pub original_gaussians: usize,
    /// Gaussian count per level.
    pub level_sizes: Vec<usize>,
    /// Approximate byte usage per level.
    pub memory_estimates: Vec<usize>,
    /// Sum of all level memory estimates.
    pub total_memory: usize,
}

// ---------------------------------------------------------------------------
// xorshift64 (private)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// compute_opacity_values
// ---------------------------------------------------------------------------

/// Apply sigmoid to convert logit opacities to probabilities in [0, 1].
///
/// `sigmoid(x) = 1 / (1 + exp(-x))`
pub fn compute_opacity_values(opacities: &[f32]) -> Vec<f32> {
    opacities
        .iter()
        .map(|&x| 1.0_f32 / (1.0_f32 + (-x).exp()))
        .collect()
}

// ---------------------------------------------------------------------------
// select_top_opacity_indices
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// select_uniform_indices
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// select_spatial_grid_indices
// ---------------------------------------------------------------------------

/// Select at most k Gaussians distributed across a 3D spatial grid.
///
/// Divides the bounding box into ceil(k^(1/3)) cells per axis. For each
/// non-empty cell the first encountered Gaussian is selected. Returns at most
/// k indices in ascending order.
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
            selected.push(i);
        }
    }

    selected.sort_unstable();
    Ok(selected)
}

// ---------------------------------------------------------------------------
// select_random_indices
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// extract_subset
// ---------------------------------------------------------------------------

/// Extract rows from a flat N×M array using the given row indices.
///
/// The stride M is inferred as `source.len() / n_gaussians`.
/// Returns `indices.len() × M` elements.
pub fn extract_subset(source: &[f32], n_gaussians: usize, indices: &[usize]) -> Vec<f32> {
    if n_gaussians == 0 || source.is_empty() || indices.is_empty() {
        return Vec::new();
    }
    let stride = source.len() / n_gaussians;
    let mut out = Vec::with_capacity(indices.len() * stride);
    for &idx in indices {
        let start = idx * stride;
        let end = start + stride;
        out.extend_from_slice(&source[start..end.min(source.len())]);
    }
    out
}

// ---------------------------------------------------------------------------
// generate_lod_level
// ---------------------------------------------------------------------------

/// Flat scene attribute slices for [`generate_lod_level`].
pub struct LodInputSlices<'a> {
    /// Total number of Gaussians.
    pub n_gaussians: usize,
    /// Positions, flat N×3.
    pub positions: &'a [f32],
    /// Rotations (quaternions), flat N×4.
    pub rotations: &'a [f32],
    /// Log-scales, flat N×3.
    pub scales: &'a [f32],
    /// Logit-space opacities, length N.
    pub opacities: &'a [f32],
    /// SH coefficients, flat N×sh_per.
    pub sh_coefficients: &'a [f32],
}

/// Generate a single LOD level by selecting `target_n` Gaussians from the cloud.
pub fn generate_lod_level(
    input: LodInputSlices<'_>,
    target_n: usize,
    config: &LodConfig,
    level: usize,
) -> Result<LodLevel, LodError> {
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
        LodStrategy::Uniform => select_uniform_indices(n_gaussians, target_n),
        LodStrategy::SpatialGrid => select_spatial_grid_indices(positions, target_n)?,
        LodStrategy::Random => {
            // Deterministic seed derived from level index.
            let seed = (level as u64 + 1).wrapping_mul(6_364_136_223_846_793_005u64);
            select_random_indices(n_gaussians, target_n, seed)
        }
    };

    let actual_n = indices.len();
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
        positions: extract_subset(positions, n_gaussians, &indices),
        rotations: extract_subset(rotations, n_gaussians, &indices),
        scales: extract_subset(scales, n_gaussians, &indices),
        opacities: selected_opacities,
        sh_coefficients: extract_subset(sh_coefficients, n_gaussians, &indices),
    })
}

// ---------------------------------------------------------------------------
// generate_lod_chain
// ---------------------------------------------------------------------------

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

    // Validate all array lengths.
    if rotations.len() != n_gaussians * 4 {
        return Err(LodError::ArrayLengthMismatch {
            n_gaussians,
            field: "rotations".to_string(),
            actual: rotations.len(),
        });
    }
    if scales.len() != n_gaussians * 3 {
        return Err(LodError::ArrayLengthMismatch {
            n_gaussians,
            field: "scales".to_string(),
            actual: scales.len(),
        });
    }
    if opacities.len() != n_gaussians {
        return Err(LodError::ArrayLengthMismatch {
            n_gaussians,
            field: "opacities".to_string(),
            actual: opacities.len(),
        });
    }
    if !sh_coefficients.is_empty() && !sh_coefficients.len().is_multiple_of(n_gaussians) {
        return Err(LodError::ArrayLengthMismatch {
            n_gaussians,
            field: "sh_coefficients".to_string(),
            actual: sh_coefficients.len(),
        });
    }

    config.validate()?;

    let mut levels = Vec::with_capacity(config.n_levels);
    for (lvl, &ratio) in config.reduction_ratios.iter().enumerate() {
        let target_n = ((n_gaussians as f32 * ratio).ceil() as usize)
            .max(1)
            .min(n_gaussians);
        let lod_level = generate_lod_level(
            LodInputSlices {
                n_gaussians,
                positions,
                rotations,
                scales,
                opacities,
                sh_coefficients,
            },
            target_n,
            config,
            lvl,
        )?;
        levels.push(lod_level);
    }

    Ok(LodChain {
        original_n_gaussians: n_gaussians,
        levels,
    })
}

// ---------------------------------------------------------------------------
// compute_lod_stats
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// merge_lod_levels
// ---------------------------------------------------------------------------

/// Blend two LOD levels of equal size via linear interpolation.
///
/// `weight_a = 0.0` → pure `level_a`; `weight_a = 1.0` → pure `level_b`.
pub fn merge_lod_levels(
    level_a: &LodLevel,
    level_b: &LodLevel,
    weight_a: f32,
) -> Result<LodLevel, LodError> {
    if level_a.n_gaussians != level_b.n_gaussians {
        return Err(LodError::ArrayLengthMismatch {
            n_gaussians: level_a.n_gaussians,
            field: "level_b".to_string(),
            actual: level_b.n_gaussians,
        });
    }
    let t = weight_a; // 0 = pure a, 1 = pure b
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
        rotations: lerp_vec(&level_a.rotations, &level_b.rotations),
        scales: lerp_vec(&level_a.scales, &level_b.scales),
        opacities: lerp_vec(&level_a.opacities, &level_b.opacities),
        sh_coefficients: lerp_vec(&level_a.sh_coefficients, &level_b.sh_coefficients),
    })
}

// ---------------------------------------------------------------------------
// format_lod_stats
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// estimate_lod_memory
// ---------------------------------------------------------------------------

/// Estimate memory in bytes for a single LOD level.
///
/// params_per_gaussian = 3 (pos) + 4 (rot) + 3 (scale) + 1 (opacity) + sh_per
pub fn estimate_lod_memory(n_gaussians: usize, sh_coefficients_len: usize) -> usize {
    let sh_per = sh_coefficients_len.checked_div(n_gaussians).unwrap_or(0);
    4 * (3 + 4 + 3 + 1 + sh_per) * n_gaussians
}

// ---------------------------------------------------------------------------
// find_optimal_reduction_ratios
// ---------------------------------------------------------------------------

/// Find a geometric progression [1.0, r, r², …, r^(n-1)] such that the total
/// memory across all levels fits within `target_memory_bytes`.
///
/// Uses binary search for `r` in [0.1, 1.0].
pub fn find_optimal_reduction_ratios(
    n_gaussians: usize,
    target_memory_bytes: usize,
    n_levels: usize,
    sh_per_gaussian: usize,
) -> Vec<f32> {
    if n_levels == 0 {
        return Vec::new();
    }
    if n_levels == 1 {
        return vec![1.0_f32];
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

    let r = lo as f32;
    let mut ratios = Vec::with_capacity(n_levels);
    for lv in 0..n_levels {
        ratios.push(r.powi(lv as i32).clamp(0.001, 1.0));
    }
    // Force exact 1.0 for the first level.
    if let Some(first) = ratios.first_mut() {
        *first = 1.0_f32;
    }
    ratios
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Shared test utilities
    // ------------------------------------------------------------------

    type CloudArrays = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    fn make_cloud(n: usize, sh_c: usize) -> CloudArrays {
        let positions: Vec<f32> = (0..n * 3).map(|i| i as f32 * 0.01).collect();
        let rotations: Vec<f32> = (0..n * 4)
            .map(|i| if i % 4 == 3 { 1.0 } else { 0.0 })
            .collect();
        let scales: Vec<f32> = vec![-1.0f32; n * 3];
        // Logit opacities from -3.0 to 3.0 across n Gaussians.
        let opacities: Vec<f32> = (0..n)
            .map(|i| (i as f32 / (n as f32).max(1.0)) * 6.0 - 3.0)
            .collect();
        let sh_coefficients: Vec<f32> = vec![0.1f32; n * sh_c];
        (positions, rotations, scales, opacities, sh_coefficients)
    }

    fn make_level(n: usize, val: f32, level_idx: usize) -> LodLevel {
        LodLevel {
            level: level_idx,
            n_gaussians: n,
            reduction_factor: 1.0,
            positions: vec![val; n * 3],
            rotations: vec![val; n * 4],
            scales: vec![val; n * 3],
            opacities: vec![val; n],
            sh_coefficients: vec![val; n * 9],
        }
    }

    // ------------------------------------------------------------------
    // LodConfig tests
    // ------------------------------------------------------------------

    #[test]
    fn test_lod_config_default_n_levels() {
        let cfg = LodConfig::default();
        assert_eq!(cfg.n_levels, 4);
    }

    #[test]
    fn test_lod_config_default_ratios() {
        let cfg = LodConfig::default();
        assert_eq!(cfg.reduction_ratios.len(), 4);
        assert!((cfg.reduction_ratios[0] - 1.0).abs() < 1e-6);
        assert!((cfg.reduction_ratios[1] - 0.5).abs() < 1e-6);
        assert!((cfg.reduction_ratios[2] - 0.25).abs() < 1e-6);
        assert!((cfg.reduction_ratios[3] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_lod_config_default_strategy() {
        let cfg = LodConfig::default();
        assert_eq!(cfg.strategy, LodStrategy::TopOpacity);
    }

    #[test]
    fn test_lod_config_default_sort_by_opacity() {
        let cfg = LodConfig::default();
        assert!(cfg.sort_by_opacity);
    }

    #[test]
    fn test_lod_config_validate_ok() {
        assert!(LodConfig::default().validate().is_ok());
    }

    #[test]
    fn test_lod_config_validate_zero_ratio() {
        let cfg = LodConfig {
            n_levels: 2,
            reduction_ratios: vec![1.0, 0.0],
            strategy: LodStrategy::TopOpacity,
            sort_by_opacity: true,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_lod_config_validate_ratio_above_one() {
        let cfg = LodConfig {
            n_levels: 2,
            reduction_ratios: vec![1.5, 0.5],
            strategy: LodStrategy::TopOpacity,
            sort_by_opacity: true,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_lod_config_validate_ascending_ratios() {
        let cfg = LodConfig {
            n_levels: 3,
            reduction_ratios: vec![0.1, 0.5, 1.0],
            strategy: LodStrategy::TopOpacity,
            sort_by_opacity: true,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_lod_config_validate_mismatched_length() {
        let cfg = LodConfig {
            n_levels: 4,
            reduction_ratios: vec![1.0, 0.5],
            strategy: LodStrategy::TopOpacity,
            sort_by_opacity: true,
        };
        assert!(cfg.validate().is_err());
    }

    // ------------------------------------------------------------------
    // compute_opacity_values tests
    // ------------------------------------------------------------------

    #[test]
    fn test_compute_opacity_values_zero_logit() {
        let probs = compute_opacity_values(&[0.0]);
        assert!((probs[0] - 0.5).abs() < 1e-6, "sigmoid(0) must equal 0.5");
    }

    #[test]
    fn test_compute_opacity_values_large_positive() {
        let probs = compute_opacity_values(&[20.0]);
        assert!(
            probs[0] > 0.99,
            "sigmoid(20) must be close to 1.0, got {}",
            probs[0]
        );
    }

    #[test]
    fn test_compute_opacity_values_large_negative() {
        let probs = compute_opacity_values(&[-20.0]);
        assert!(
            probs[0] < 0.01,
            "sigmoid(-20) must be close to 0.0, got {}",
            probs[0]
        );
    }

    #[test]
    fn test_compute_opacity_values_preserves_monotone_order() {
        let logits = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let probs = compute_opacity_values(&logits);
        for i in 1..probs.len() {
            assert!(
                probs[i] > probs[i - 1],
                "sigmoid must be monotone increasing"
            );
        }
    }

    // ------------------------------------------------------------------
    // select_top_opacity_indices tests
    // ------------------------------------------------------------------

    #[test]
    fn test_select_top_opacity_indices_selects_highest() {
        // Logit values: index 9 is highest opacity.
        let opacities: Vec<f32> = (0..10).map(|i| i as f32 - 5.0).collect();
        let indices = select_top_opacity_indices(10, &opacities, 3);
        assert_eq!(indices.len(), 3);
        // The top 3 by logit are indices 7, 8, 9.
        assert!(indices.contains(&7) || indices.contains(&8) || indices.contains(&9));
        assert_eq!(indices.iter().filter(|&&i| i >= 7).count(), 3);
    }

    #[test]
    fn test_select_top_opacity_indices_k_equals_n() {
        let opacities = vec![1.0, 2.0, 3.0];
        let indices = select_top_opacity_indices(3, &opacities, 3);
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_select_top_opacity_indices_k_exceeds_n() {
        let opacities = vec![1.0, 2.0];
        let indices = select_top_opacity_indices(2, &opacities, 10);
        assert_eq!(indices.len(), 2);
    }

    #[test]
    fn test_select_top_opacity_indices_sorted_output() {
        let opacities: Vec<f32> = (0..20).rev().map(|i| i as f32).collect();
        let indices = select_top_opacity_indices(20, &opacities, 5);
        // Output must be in ascending index order.
        for w in indices.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    // ------------------------------------------------------------------
    // select_uniform_indices tests
    // ------------------------------------------------------------------

    #[test]
    fn test_select_uniform_indices_count() {
        let indices = select_uniform_indices(100, 10);
        assert_eq!(indices.len(), 10);
    }

    #[test]
    fn test_select_uniform_indices_includes_first() {
        let indices = select_uniform_indices(100, 5);
        assert!(indices.contains(&0));
    }

    #[test]
    fn test_select_uniform_indices_includes_last() {
        let indices = select_uniform_indices(100, 5);
        assert!(indices.contains(&99), "expected 99 in {:?}", indices);
    }

    #[test]
    fn test_select_uniform_indices_k_ge_n() {
        let indices = select_uniform_indices(5, 10);
        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_select_uniform_indices_k_zero() {
        assert!(select_uniform_indices(100, 0).is_empty());
    }

    #[test]
    fn test_select_uniform_indices_specific_known() {
        // n=10, k=5 → raw: 0,2,4,6,8 → last becomes 9 → result: 0,2,4,6,9
        let indices = select_uniform_indices(10, 5);
        assert!(indices.contains(&0), "must contain 0, got {:?}", indices);
        assert!(indices.contains(&9), "must contain 9, got {:?}", indices);
        assert_eq!(indices.len(), 5);
    }

    // ------------------------------------------------------------------
    // select_random_indices tests
    // ------------------------------------------------------------------

    #[test]
    fn test_select_random_indices_count() {
        assert_eq!(select_random_indices(100, 30, 12345).len(), 30);
    }

    #[test]
    fn test_select_random_indices_reproducible() {
        let a = select_random_indices(100, 30, 42);
        let b = select_random_indices(100, 30, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_select_random_indices_different_seeds_differ() {
        let a = select_random_indices(100, 50, 1);
        let b = select_random_indices(100, 50, 9999);
        assert_ne!(a, b);
    }

    #[test]
    fn test_select_random_indices_k_exceeds_n() {
        assert_eq!(select_random_indices(5, 20, 1).len(), 5);
    }

    #[test]
    fn test_select_random_indices_k_zero() {
        assert!(select_random_indices(100, 0, 1).is_empty());
    }

    // ------------------------------------------------------------------
    // select_spatial_grid_indices tests
    // ------------------------------------------------------------------

    #[test]
    fn test_select_spatial_grid_indices_count_limit() {
        let n = 1000usize;
        let positions: Vec<f32> = (0..n * 3).map(|i| (i as f32).sin()).collect();
        let indices = select_spatial_grid_indices(&positions, 50).expect("grid selection failed");
        assert!(indices.len() <= 50);
        assert!(!indices.is_empty());
    }

    #[test]
    fn test_select_spatial_grid_indices_empty_error() {
        assert!(select_spatial_grid_indices(&[], 10).is_err());
    }

    #[test]
    fn test_select_spatial_grid_indices_k_zero() {
        let positions = vec![0.0f32; 30];
        let indices = select_spatial_grid_indices(&positions, 0).expect("k=0 must not error");
        assert!(indices.is_empty());
    }

    #[test]
    fn test_select_spatial_grid_indices_k_ge_n() {
        let n = 5usize;
        let positions: Vec<f32> = (0..n * 3).map(|i| i as f32).collect();
        let indices = select_spatial_grid_indices(&positions, 100).expect("k>=n must not error");
        assert_eq!(indices.len(), n);
    }

    // ------------------------------------------------------------------
    // extract_subset tests
    // ------------------------------------------------------------------

    #[test]
    fn test_extract_subset_stride_3_positions() {
        // n=4 Gaussians, stride=3. Source rows: [0,1,2], [3,4,5], [6,7,8], [9,10,11].
        let source: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let out = extract_subset(&source, 4, &[0, 2]);
        assert_eq!(out.len(), 6);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 2.0);
        assert_eq!(out[3], 6.0);
        assert_eq!(out[4], 7.0);
        assert_eq!(out[5], 8.0);
    }

    #[test]
    fn test_extract_subset_stride_4_rotations() {
        // n=3 Gaussians, stride=4. Row 1 = [4,5,6,7].
        let source: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let out = extract_subset(&source, 3, &[1]);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], 4.0);
        assert_eq!(out[3], 7.0);
    }

    #[test]
    fn test_extract_subset_empty_indices() {
        let source = vec![1.0f32, 2.0, 3.0];
        assert!(extract_subset(&source, 1, &[]).is_empty());
    }

    // ------------------------------------------------------------------
    // generate_lod_level tests
    // ------------------------------------------------------------------

    #[test]
    fn test_generate_lod_level_top_opacity_count() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let config = LodConfig::default();
        let level = generate_lod_level(
            LodInputSlices {
                n_gaussians: 100,
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
            },
            50,
            &config,
            0,
        )
        .expect("generate_lod_level failed");
        assert_eq!(level.n_gaussians, 50);
    }

    #[test]
    fn test_generate_lod_level_arrays_consistent() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let config = LodConfig::default();
        let level = generate_lod_level(
            LodInputSlices {
                n_gaussians: 100,
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
            },
            50,
            &config,
            0,
        )
        .expect("generate_lod_level failed");
        assert_eq!(level.positions.len(), 50 * 3);
        assert_eq!(level.rotations.len(), 50 * 4);
        assert_eq!(level.scales.len(), 50 * 3);
        assert_eq!(level.opacities.len(), 50);
        assert_eq!(level.sh_coefficients.len(), 50 * 9);
    }

    #[test]
    fn test_generate_lod_level_level_index_stored() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let config = LodConfig::default();
        let level = generate_lod_level(
            LodInputSlices {
                n_gaussians: 100,
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
            },
            50,
            &config,
            2,
        )
        .expect("generate_lod_level failed");
        assert_eq!(level.level, 2);
    }

    #[test]
    fn test_generate_lod_level_uniform_strategy() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let config = LodConfig {
            n_levels: 2,
            reduction_ratios: vec![1.0, 0.5],
            strategy: LodStrategy::Uniform,
            sort_by_opacity: false,
        };
        let level = generate_lod_level(
            LodInputSlices {
                n_gaussians: 100,
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
            },
            40,
            &config,
            1,
        )
        .expect("uniform generate failed");
        assert!(level.n_gaussians <= 40);
    }

    #[test]
    fn test_generate_lod_level_insufficient_gaussians_error() {
        let (pos, rot, sc, op, sh) = make_cloud(10, 9);
        let config = LodConfig::default();
        let result = generate_lod_level(
            LodInputSlices {
                n_gaussians: 10,
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
            },
            20,
            &config,
            0,
        );
        assert!(matches!(
            result,
            Err(LodError::InsufficientGaussians { .. })
        ));
    }

    // ------------------------------------------------------------------
    // generate_lod_chain tests
    // ------------------------------------------------------------------

    #[test]
    fn test_generate_lod_chain_n_levels() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain generation failed");
        assert_eq!(chain.levels.len(), 4);
    }

    #[test]
    fn test_generate_lod_chain_level0_full_resolution() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain generation failed");
        assert_eq!(chain.levels[0].n_gaussians, 100);
    }

    #[test]
    fn test_generate_lod_chain_decreasing_sizes() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain generation failed");
        for i in 1..chain.levels.len() {
            assert!(
                chain.levels[i].n_gaussians <= chain.levels[i - 1].n_gaussians,
                "level {} ({}) must be ≤ level {} ({})",
                i,
                chain.levels[i].n_gaussians,
                i - 1,
                chain.levels[i - 1].n_gaussians
            );
        }
    }

    #[test]
    fn test_generate_lod_chain_empty_error() {
        let result = generate_lod_chain(&[], &[], &[], &[], &[], &LodConfig::default());
        assert!(matches!(result, Err(LodError::EmptyCloud)));
    }

    #[test]
    fn test_generate_lod_chain_invalid_position_length() {
        let result = generate_lod_chain(&[1.0, 2.0], &[], &[], &[], &[], &LodConfig::default());
        assert!(matches!(
            result,
            Err(LodError::InvalidPositionLength { .. })
        ));
    }

    #[test]
    fn test_generate_lod_chain_array_mismatch_rotations() {
        let result = generate_lod_chain(
            &[0.0, 0.0, 0.0], // n=1
            &[0.0; 8],        // should be 4, error
            &[0.0; 3],
            &[0.0; 1],
            &[],
            &LodConfig::default(),
        );
        assert!(matches!(result, Err(LodError::ArrayLengthMismatch { .. })));
    }

    #[test]
    fn test_generate_lod_chain_original_n_gaussians() {
        let (pos, rot, sc, op, sh) = make_cloud(80, 3);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        assert_eq!(chain.original_n_gaussians, 80);
    }

    // ------------------------------------------------------------------
    // compute_lod_stats tests
    // ------------------------------------------------------------------

    #[test]
    fn test_compute_lod_stats_n_levels() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        assert_eq!(compute_lod_stats(&chain).n_levels, 4);
    }

    #[test]
    fn test_compute_lod_stats_level_sizes_length() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        assert_eq!(compute_lod_stats(&chain).level_sizes.len(), 4);
    }

    #[test]
    fn test_compute_lod_stats_level0_size() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        assert_eq!(compute_lod_stats(&chain).level_sizes[0], 100);
    }

    #[test]
    fn test_compute_lod_stats_memory_level0() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        let stats = compute_lod_stats(&chain);
        // 100 × (3+4+3+1+9) × 4 = 100 × 20 × 4 = 8000
        assert_eq!(stats.memory_estimates[0], 8000);
    }

    #[test]
    fn test_compute_lod_stats_total_memory_correct() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        let stats = compute_lod_stats(&chain);
        let expected: usize = stats.memory_estimates.iter().sum();
        assert_eq!(stats.total_memory, expected);
    }

    // ------------------------------------------------------------------
    // merge_lod_levels tests
    // ------------------------------------------------------------------

    #[test]
    fn test_merge_lod_levels_weight_zero_is_pure_a() {
        let a = make_level(10, 1.0, 0);
        let b = make_level(10, 2.0, 1);
        let merged = merge_lod_levels(&a, &b, 0.0).expect("merge failed");
        assert!((merged.opacities[0] - 1.0).abs() < 1e-6);
        assert!((merged.positions[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_merge_lod_levels_weight_one_is_pure_b() {
        let a = make_level(10, 1.0, 0);
        let b = make_level(10, 2.0, 1);
        let merged = merge_lod_levels(&a, &b, 1.0).expect("merge failed");
        assert!((merged.opacities[0] - 2.0).abs() < 1e-6);
        assert!((merged.positions[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_merge_lod_levels_weight_half_midpoint() {
        let a = make_level(10, 0.0, 0);
        let b = make_level(10, 2.0, 1);
        let merged = merge_lod_levels(&a, &b, 0.5).expect("merge failed");
        assert!((merged.opacities[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_merge_lod_levels_size_mismatch_error() {
        let a = make_level(10, 1.0, 0);
        let b = make_level(5, 2.0, 1);
        assert!(merge_lod_levels(&a, &b, 0.5).is_err());
    }

    #[test]
    fn test_merge_lod_levels_preserves_level_a_index() {
        let a = make_level(10, 0.0, 3);
        let b = make_level(10, 1.0, 5);
        let merged = merge_lod_levels(&a, &b, 0.5).expect("merge failed");
        assert_eq!(merged.level, 3);
    }

    // ------------------------------------------------------------------
    // LodChain::select tests
    // ------------------------------------------------------------------

    #[test]
    fn test_lod_chain_select_distance_zero_gives_level0() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        let sel = LodSelector::default();
        let level = chain.select(0.0, &sel).expect("select failed");
        assert_eq!(level.level, 0);
    }

    #[test]
    fn test_lod_chain_select_large_distance_gives_lowest() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        let sel = LodSelector::default();
        let level = chain.select(1000.0, &sel).expect("select failed");
        assert_eq!(level.level, chain.levels.len() - 1);
    }

    #[test]
    fn test_lod_chain_select_mid_distance_gives_level1() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        // thresholds=[1.0, 3.0, 7.0]; distance=2.0 → beyond [0]=1.0, below [1]=3.0 → level 1
        let sel = LodSelector::new(vec![1.0, 3.0, 7.0]);
        let level = chain.select(2.0, &sel).expect("select failed");
        assert_eq!(level.level, 1);
    }

    // ------------------------------------------------------------------
    // format_lod_stats tests
    // ------------------------------------------------------------------

    #[test]
    fn test_format_lod_stats_is_non_empty() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        assert!(!format_lod_stats(&compute_lod_stats(&chain)).is_empty());
    }

    #[test]
    fn test_format_lod_stats_contains_lod() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        let s = format_lod_stats(&compute_lod_stats(&chain));
        assert!(s.contains("LOD"), "expected 'LOD' in '{}'", s);
    }

    #[test]
    fn test_format_lod_stats_contains_counts() {
        let stats = LodStats {
            n_levels: 3,
            original_gaussians: 500,
            level_sizes: vec![500, 250, 50],
            memory_estimates: vec![40000, 20000, 4000],
            total_memory: 64000,
        };
        let s = format_lod_stats(&stats);
        assert!(s.contains("500"));
        assert!(s.contains("3"));
    }

    // ------------------------------------------------------------------
    // estimate_lod_memory tests
    // ------------------------------------------------------------------

    #[test]
    fn test_estimate_lod_memory_with_sh() {
        // n=100, sh_len=900 → sh_per=9, params=20, bytes=4*20*100=8000
        assert_eq!(estimate_lod_memory(100, 900), 8000);
    }

    #[test]
    fn test_estimate_lod_memory_no_sh() {
        // n=10, sh_len=0, params=11, bytes=4*11*10=440
        assert_eq!(estimate_lod_memory(10, 0), 440);
    }

    #[test]
    fn test_estimate_lod_memory_zero_gaussians() {
        assert_eq!(estimate_lod_memory(0, 0), 0);
    }

    // ------------------------------------------------------------------
    // find_optimal_reduction_ratios tests
    // ------------------------------------------------------------------

    #[test]
    fn test_find_optimal_reduction_ratios_returns_n_levels() {
        let ratios = find_optimal_reduction_ratios(1000, 1_000_000, 4, 9);
        assert_eq!(ratios.len(), 4);
    }

    #[test]
    fn test_find_optimal_reduction_ratios_first_is_one() {
        let ratios = find_optimal_reduction_ratios(1000, 1_000_000, 4, 9);
        assert!(
            (ratios[0] - 1.0).abs() < 1e-6,
            "first ratio must be 1.0, got {}",
            ratios[0]
        );
    }

    #[test]
    fn test_find_optimal_reduction_ratios_non_ascending() {
        let ratios = find_optimal_reduction_ratios(500, 500_000, 4, 9);
        for i in 1..ratios.len() {
            assert!(
                ratios[i] <= ratios[i - 1] + 1e-5,
                "ratios must be non-ascending"
            );
        }
    }

    #[test]
    fn test_find_optimal_reduction_ratios_n_levels_one() {
        let ratios = find_optimal_reduction_ratios(100, 100_000, 1, 9);
        assert_eq!(ratios.len(), 1);
        assert!((ratios[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_optimal_reduction_ratios_n_levels_zero() {
        assert!(find_optimal_reduction_ratios(100, 100_000, 0, 9).is_empty());
    }

    // ------------------------------------------------------------------
    // LodError variant display tests
    // ------------------------------------------------------------------

    #[test]
    fn test_lod_error_empty_cloud_display() {
        let e = LodError::EmptyCloud;
        assert!(format!("{}", e).contains("Empty"));
    }

    #[test]
    fn test_lod_error_invalid_position_length_display() {
        let e = LodError::InvalidPositionLength { len: 7 };
        assert!(format!("{}", e).contains("7"));
    }

    #[test]
    fn test_lod_error_array_length_mismatch_display() {
        let e = LodError::ArrayLengthMismatch {
            n_gaussians: 10,
            field: "rotations".to_string(),
            actual: 30,
        };
        assert!(format!("{}", e).contains("rotations"));
    }

    #[test]
    fn test_lod_error_invalid_lod_level_display() {
        let e = LodError::InvalidLodLevel {
            level: 5,
            n_levels: 4,
        };
        assert!(format!("{}", e).contains("5"));
    }

    #[test]
    fn test_lod_error_insufficient_gaussians_display() {
        let e = LodError::InsufficientGaussians { k: 100, n: 50 };
        let s = format!("{}", e);
        assert!(s.contains("100") && s.contains("50"));
    }

    // ------------------------------------------------------------------
    // LodLevel::validate tests
    // ------------------------------------------------------------------

    #[test]
    fn test_lod_level_validate_ok() {
        assert!(make_level(10, 0.5, 0).validate().is_ok());
    }

    #[test]
    fn test_lod_level_validate_positions_extra_element() {
        let mut level = make_level(10, 0.5, 0);
        level.positions.push(0.0);
        assert!(level.validate().is_err());
    }

    #[test]
    fn test_lod_level_validate_opacities_missing_element() {
        let mut level = make_level(10, 0.5, 0);
        level.opacities.pop();
        assert!(level.validate().is_err());
    }

    // ------------------------------------------------------------------
    // LodStrategy variant tests
    // ------------------------------------------------------------------

    #[test]
    fn test_lod_strategy_all_variants_distinct() {
        assert_ne!(LodStrategy::TopOpacity, LodStrategy::Uniform);
        assert_ne!(LodStrategy::Uniform, LodStrategy::SpatialGrid);
        assert_ne!(LodStrategy::SpatialGrid, LodStrategy::Random);
        assert_ne!(LodStrategy::Random, LodStrategy::TopOpacity);
    }

    #[test]
    fn test_lod_strategy_random_chain() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let config = LodConfig {
            n_levels: 2,
            reduction_ratios: vec![1.0, 0.5],
            strategy: LodStrategy::Random,
            sort_by_opacity: false,
        };
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &config).expect("chain failed");
        assert_eq!(chain.levels[0].n_gaussians, 100);
        assert!(chain.levels[1].n_gaussians <= 50);
    }

    #[test]
    fn test_lod_strategy_spatial_grid_chain() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let config = LodConfig {
            n_levels: 2,
            reduction_ratios: vec![1.0, 0.5],
            strategy: LodStrategy::SpatialGrid,
            sort_by_opacity: false,
        };
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &config).expect("chain failed");
        assert_eq!(chain.levels.len(), 2);
    }

    // ------------------------------------------------------------------
    // LodSelector tests
    // ------------------------------------------------------------------

    #[test]
    fn test_lod_selector_default_three_thresholds() {
        let sel = LodSelector::default();
        assert_eq!(sel.thresholds.len(), 3);
    }

    #[test]
    fn test_lod_selector_default_values() {
        let sel = LodSelector::default();
        assert!((sel.thresholds[0] - 0.5).abs() < 1e-6);
        assert!((sel.thresholds[1] - 2.0).abs() < 1e-6);
        assert!((sel.thresholds[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_lod_selector_new() {
        let sel = LodSelector::new(vec![1.0, 10.0]);
        assert_eq!(sel.thresholds, vec![1.0, 10.0]);
    }

    // ------------------------------------------------------------------
    // n_params_per_gaussian tests
    // ------------------------------------------------------------------

    #[test]
    fn test_n_params_per_gaussian_with_sh9() {
        let level = make_level(10, 0.0, 0);
        // 3+4+3+1+9 = 20
        assert_eq!(level.n_params_per_gaussian(), 20);
    }

    #[test]
    fn test_n_params_per_gaussian_no_sh() {
        let mut level = make_level(10, 0.0, 0);
        level.sh_coefficients.clear();
        // 3+4+3+1+0 = 11
        assert_eq!(level.n_params_per_gaussian(), 11);
    }
}
