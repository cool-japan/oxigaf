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

    /// A row index passed to a selection/extraction function is out of range.
    #[error("Index {index} out of range: cloud has only {n_gaussians} Gaussians")]
    IndexOutOfRange { index: usize, n_gaussians: usize },

    /// No reduction-ratio chain fits the requested memory budget.
    #[error(
        "No LOD ratio chain fits within {target_bytes} bytes \
         (minimum achievable is {minimum_bytes} bytes)"
    )]
    MemoryBudgetExceeded {
        minimum_bytes: usize,
        target_bytes: usize,
    },
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
    /// Rank Gaussians by descending opacity before selection.
    ///
    /// Only affects [`LodStrategy::Uniform`] and [`LodStrategy::Random`],
    /// which otherwise pick evenly-spaced/random *storage* indices; when
    /// set, they instead pick evenly-spaced/random *opacity ranks*, so
    /// `Uniform` in particular favors more-visible Gaussians instead of an
    /// arbitrary storage order. [`LodStrategy::TopOpacity`] already ranks by
    /// opacity directly and [`LodStrategy::SpatialGrid`] ranks by 3D
    /// position, so this has no effect on either.
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
    /// Get a LOD level directly by its index (0 = highest quality).
    ///
    /// # Errors
    ///
    /// Returns [`LodError::InvalidLodLevel`] if `level >= self.levels.len()`.
    pub fn get_level(&self, level: usize) -> Result<&LodLevel, LodError> {
        self.levels.get(level).ok_or(LodError::InvalidLodLevel {
            level,
            n_levels: self.levels.len(),
        })
    }

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
        for i in 0..n_gaussians {
            if selected.len() >= k {
                break;
            }
            if !picked[i] {
                picked[i] = true;
                selected.push(i);
            }
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
// opacity_descending_permutation / select_indices_by_rank
// ---------------------------------------------------------------------------

/// Permutation of `0..opacities.len()` ordered by descending sigmoid-opacity
/// (most opaque first).
///
/// Used to let rank-based selectors ([`select_uniform_indices`],
/// [`select_random_indices`]) operate on opacity rank instead of raw
/// storage order when [`LodConfig::sort_by_opacity`] is set.
fn opacity_descending_permutation(opacities: &[f32]) -> Vec<usize> {
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
fn select_indices_by_rank(
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

// ---------------------------------------------------------------------------
// extract_subset
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// generate_lod_level
// ---------------------------------------------------------------------------

/// Flat scene attribute slices for [`generate_lod_level`].
#[derive(Debug, Clone, Copy)]
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
/// LERP; rotations use quaternion NLERP ([`nlerp_quaternions`]) since a
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
        // Regression: the grid pass alone (occupied-cell-only) used to
        // undershoot k for real point clouds (e.g. only ~25-30 of a 4×4×4
        // grid's 64 cells occupied); the top-up pass must reach exactly k.
        assert_eq!(
            indices.len(),
            50,
            "expected exactly the requested count, not a grid-occupancy-limited undershoot"
        );
    }

    #[test]
    fn test_select_spatial_grid_indices_top_up_reaches_k_when_grid_sparse() {
        // All Gaussians share one point: every one of them falls in the
        // same grid cell, so the grid pass alone can select only 1. The
        // top-up pass must still reach the full k=20.
        let n = 100usize;
        let mut positions: Vec<f32> = Vec::with_capacity(n * 3);
        for _ in 0..n {
            positions.extend_from_slice(&[1.0f32, 2.0, 3.0]);
        }
        let indices = select_spatial_grid_indices(&positions, 20).expect("grid selection failed");
        assert_eq!(indices.len(), 20);
        // No duplicate indices.
        let mut sorted = indices.clone();
        sorted.dedup();
        assert_eq!(sorted.len(), indices.len());
    }

    #[test]
    fn test_select_spatial_grid_indices_no_duplicates_general() {
        let n = 200usize;
        let positions: Vec<f32> = (0..n * 3).map(|i| (i as f32 * 0.7).cos()).collect();
        let indices = select_spatial_grid_indices(&positions, 77).expect("grid selection failed");
        assert_eq!(indices.len(), 77);
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), indices.len(), "indices must be unique");
        assert!(indices.iter().all(|&i| i < n));
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
        let out = extract_subset(&source, 4, &[0, 2]).expect("extract failed");
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
        let out = extract_subset(&source, 3, &[1]).expect("extract failed");
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], 4.0);
        assert_eq!(out[3], 7.0);
    }

    #[test]
    fn test_extract_subset_empty_indices() {
        let source = vec![1.0f32, 2.0, 3.0];
        assert!(extract_subset(&source, 1, &[])
            .expect("extract failed")
            .is_empty());
    }

    #[test]
    fn test_extract_subset_index_out_of_range_errors_instead_of_panicking() {
        // Regression: n_gaussians=4 but an index of 10 used to compute
        // `start = 10 * stride` past `source.len()`, producing a `start >
        // end` range that panics after the old `.min(source.len())` clamp
        // only clamped the end, not the start.
        let source: Vec<f32> = (0..12).map(|i| i as f32).collect(); // n=4, stride=3
        let result = extract_subset(&source, 4, &[0, 10]);
        assert!(matches!(
            result,
            Err(LodError::IndexOutOfRange { index: 10, .. })
        ));
    }

    #[test]
    fn test_extract_subset_not_multiple_of_n_gaussians_errors() {
        let source = vec![0.0f32; 10];
        let result = extract_subset(&source, 3, &[0]);
        assert!(matches!(result, Err(LodError::ArrayLengthMismatch { .. })));
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

    #[test]
    fn test_generate_lod_level_direct_call_rejects_mismatched_positions() {
        // Regression: calling `generate_lod_level` directly (bypassing
        // `generate_lod_chain`, which used to be the only validated entry
        // point) with n_gaussians=100 but a 200-element positions array
        // (a multiple of 100, but the wrong stride: 2 instead of 3) must
        // error instead of silently extracting misaligned rows.
        let n = 100usize;
        let positions = vec![0.0f32; n * 2]; // wrong: should be n * 3
        let rotations = vec![0.0f32; n * 4];
        let scales = vec![0.0f32; n * 3];
        let opacities = vec![0.0f32; n];
        let sh = vec![0.0f32; n * 9];
        let config = LodConfig::default();
        let result = generate_lod_level(
            LodInputSlices {
                n_gaussians: n,
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_coefficients: &sh,
            },
            50,
            &config,
            0,
        );
        assert!(matches!(result, Err(LodError::ArrayLengthMismatch { .. })));
    }

    #[test]
    fn test_generate_lod_level_direct_call_rejects_short_rotations() {
        // Regression: a rotations array shorter than n_gaussians*4 used to
        // reach `extract_subset` unchecked and could panic (or, before that
        // fix, `idx * stride` could exceed the array and slice out of
        // bounds). It must now be rejected up front.
        let n = 20usize;
        let positions = vec![0.0f32; n * 3];
        let rotations = vec![0.0f32; n * 4 - 1]; // one short
        let scales = vec![0.0f32; n * 3];
        let opacities = vec![0.0f32; n];
        let sh: Vec<f32> = Vec::new();
        let config = LodConfig::default();
        let result = generate_lod_level(
            LodInputSlices {
                n_gaussians: n,
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_coefficients: &sh,
            },
            10,
            &config,
            0,
        );
        assert!(matches!(result, Err(LodError::ArrayLengthMismatch { .. })));
    }

    #[test]
    fn test_generate_lod_level_uniform_sort_by_opacity_changes_selection() {
        // n=10, opacity strictly increasing with index → descending-opacity
        // permutation is exactly the reverse index order (perm[r] = 9 - r).
        let n = 10usize;
        let positions: Vec<f32> = (0..n * 3).map(|i| i as f32).collect();
        let rotations: Vec<f32> = (0..n * 4)
            .map(|i| if i % 4 == 3 { 1.0 } else { 0.0 })
            .collect();
        let scales = vec![-1.0f32; n * 3];
        let opacities: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let sh = vec![0.0f32; n];

        let unsorted_config = LodConfig {
            n_levels: 2,
            reduction_ratios: vec![1.0, 0.5],
            strategy: LodStrategy::Uniform,
            sort_by_opacity: false,
        };
        let sorted_config = LodConfig {
            sort_by_opacity: true,
            ..unsorted_config.clone()
        };

        let unsorted = generate_lod_level(
            LodInputSlices {
                n_gaussians: n,
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_coefficients: &sh,
            },
            5,
            &unsorted_config,
            1,
        )
        .expect("unsorted level");
        let sorted = generate_lod_level(
            LodInputSlices {
                n_gaussians: n,
                positions: &positions,
                rotations: &rotations,
                scales: &scales,
                opacities: &opacities,
                sh_coefficients: &sh,
            },
            5,
            &sorted_config,
            1,
        )
        .expect("sorted level");

        // Recover which original indices were kept: positions[i] == i*3.
        let kept = |level: &LodLevel| -> Vec<usize> {
            level
                .positions
                .chunks_exact(3)
                .map(|p| (p[0] / 3.0).round() as usize)
                .collect()
        };
        // select_uniform_indices(10, 5) picks ranks [0, 2, 4, 6, 9].
        assert_eq!(kept(&unsorted), vec![0, 2, 4, 6, 9]);
        // Mapped through the descending-opacity permutation (perm[r]=9-r)
        // and re-sorted ascending: {9,7,5,3,0} -> [0,3,5,7,9].
        assert_eq!(kept(&sorted), vec![0, 3, 5, 7, 9]);
    }

    #[test]
    fn test_generate_lod_level_random_sort_by_opacity_still_returns_target_n() {
        let (pos, rot, sc, op, sh) = make_cloud(50, 9);
        let config = LodConfig {
            n_levels: 2,
            reduction_ratios: vec![1.0, 0.4],
            strategy: LodStrategy::Random,
            sort_by_opacity: true,
        };
        let level = generate_lod_level(
            LodInputSlices {
                n_gaussians: 50,
                positions: &pos,
                rotations: &rot,
                scales: &sc,
                opacities: &op,
                sh_coefficients: &sh,
            },
            20,
            &config,
            1,
        )
        .expect("random+sort_by_opacity level");
        assert_eq!(level.n_gaussians, 20);
    }

    // ------------------------------------------------------------------
    // opacity_descending_permutation / select_indices_by_rank tests
    // ------------------------------------------------------------------

    #[test]
    fn test_opacity_descending_permutation_orders_highest_first() {
        let opacities: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let perm = opacity_descending_permutation(&opacities);
        assert_eq!(perm, vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn test_select_indices_by_rank_passthrough_when_disabled() {
        let opacities = vec![0.0f32; 5];
        let ranks = vec![0, 2, 4];
        let out = select_indices_by_rank(ranks.clone(), false, &opacities);
        assert_eq!(out, ranks);
    }

    #[test]
    fn test_select_indices_by_rank_maps_through_opacity_permutation() {
        let opacities: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let ranks = vec![0, 2, 4, 6, 9];
        let out = select_indices_by_rank(ranks, true, &opacities);
        assert_eq!(out, vec![0, 3, 5, 7, 9]);
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

    #[test]
    fn test_merge_lod_levels_sh_length_mismatch_error() {
        // Same n_gaussians on both sides, but a different SH degree, so the
        // mismatch is invisible to the n_gaussians check alone.
        let mut a = make_level(5, 1.0, 0);
        let mut b = make_level(5, 2.0, 0);
        a.sh_coefficients = vec![1.0; 5 * 9];
        b.sh_coefficients = vec![2.0; 5 * 3];
        let result = merge_lod_levels(&a, &b, 0.5);
        assert!(matches!(result, Err(LodError::ArrayLengthMismatch { .. })));
    }

    #[test]
    fn test_merge_lod_levels_rotations_are_unit_length() {
        let mut a = make_level(2, 0.0, 0);
        let mut b = make_level(2, 0.0, 0);
        // Two already-unit quaternions per Gaussian.
        a.rotations = vec![0.0, 0.0, 0.0, 1.0, 0.6, 0.0, 0.0, 0.8];
        b.rotations = vec![0.0, 0.0, 0.707_106_8, 0.707_106_8, 0.0, 0.6, 0.0, 0.8];
        let merged = merge_lod_levels(&a, &b, 0.5).expect("merge failed");
        for q in merged.rotations.chunks_exact(4) {
            let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "quaternion not unit length: {q:?} (norm={norm})"
            );
        }
    }

    #[test]
    fn test_merge_lod_levels_rotation_shortest_path_avoids_zero_quaternion() {
        // `b`'s quaternion is the negation of `a`'s: same rotation, opposite
        // sign (dot(a,b) = -1 < 0). A naive component-wise LERP at t=0.5
        // would average to the zero quaternion (a degenerate rotation);
        // NLERP's sign correction must instead recognize these as the same
        // rotation and return a unit quaternion equal (up to sign) to `a`.
        let mut a = make_level(1, 0.0, 0);
        let mut b = make_level(1, 0.0, 0);
        a.rotations = vec![0.0, 0.0, 0.0, 1.0];
        b.rotations = vec![0.0, 0.0, 0.0, -1.0];
        let merged = merge_lod_levels(&a, &b, 0.5).expect("merge failed");
        let q = &merged.rotations;
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "expected a unit quaternion, got {q:?} (norm={norm})"
        );
        assert!(
            q[3].abs() > 0.99,
            "expected the w component to dominate (same rotation as input), got {q:?}"
        );
    }

    #[test]
    fn test_merge_lod_levels_rotation_weight_extremes_match_inputs() {
        let mut a = make_level(1, 0.0, 0);
        let mut b = make_level(1, 0.0, 0);
        a.rotations = vec![0.6, 0.0, 0.0, 0.8];
        b.rotations = vec![0.0, 0.6, 0.0, 0.8];
        let at_a = merge_lod_levels(&a, &b, 0.0).expect("merge failed");
        assert!((at_a.rotations[0] - 0.6).abs() < 1e-5);
        assert!((at_a.rotations[3] - 0.8).abs() < 1e-5);
        let at_b = merge_lod_levels(&a, &b, 1.0).expect("merge failed");
        assert!((at_b.rotations[1] - 0.6).abs() < 1e-5);
        assert!((at_b.rotations[3] - 0.8).abs() < 1e-5);
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
    // LodChain::get_level tests
    // ------------------------------------------------------------------

    #[test]
    fn test_lod_chain_get_level_in_range() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        let level = chain.get_level(2).expect("level 2 should exist");
        assert_eq!(level.level, 2);
    }

    #[test]
    fn test_lod_chain_get_level_out_of_range_errors() {
        let (pos, rot, sc, op, sh) = make_cloud(100, 9);
        let chain = generate_lod_chain(&pos, &rot, &sc, &op, &sh, &LodConfig::default())
            .expect("chain failed");
        let result = chain.get_level(999);
        assert!(matches!(
            result,
            Err(LodError::InvalidLodLevel {
                level: 999,
                n_levels: 4
            })
        ));
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
        let ratios = find_optimal_reduction_ratios(1000, 1_000_000, 4, 9).expect("search failed");
        assert_eq!(ratios.len(), 4);
    }

    #[test]
    fn test_find_optimal_reduction_ratios_first_is_one() {
        let ratios = find_optimal_reduction_ratios(1000, 1_000_000, 4, 9).expect("search failed");
        assert!(
            (ratios[0] - 1.0).abs() < 1e-6,
            "first ratio must be 1.0, got {}",
            ratios[0]
        );
    }

    #[test]
    fn test_find_optimal_reduction_ratios_non_ascending() {
        let ratios = find_optimal_reduction_ratios(500, 500_000, 4, 9).expect("search failed");
        for i in 1..ratios.len() {
            assert!(
                ratios[i] <= ratios[i - 1] + 1e-5,
                "ratios must be non-ascending"
            );
        }
    }

    #[test]
    fn test_find_optimal_reduction_ratios_n_levels_one() {
        let ratios = find_optimal_reduction_ratios(100, 100_000, 1, 9).expect("search failed");
        assert_eq!(ratios.len(), 1);
        assert!((ratios[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_optimal_reduction_ratios_n_levels_zero() {
        assert!(find_optimal_reduction_ratios(100, 100_000, 0, 9)
            .expect("search failed")
            .is_empty());
    }

    #[test]
    fn test_find_optimal_reduction_ratios_infeasible_budget_errors() {
        // Even at r=0.1, level 0 alone (pinned to ratio 1.0) already costs
        // 1,000,000 * 80 = 80,000,000 bytes, vastly more than a 1-byte
        // budget. The old implementation returned a chain overshooting the
        // budget with no indication; it must now report the shortfall.
        let result = find_optimal_reduction_ratios(1_000_000, 1, 4, 9);
        match result {
            Err(LodError::MemoryBudgetExceeded {
                minimum_bytes,
                target_bytes,
            }) => {
                assert_eq!(target_bytes, 1);
                assert!(minimum_bytes > target_bytes);
            }
            other => panic!("expected MemoryBudgetExceeded, got {other:?}"),
        }
    }

    #[test]
    fn test_find_optimal_reduction_ratios_feasible_budget_does_not_error() {
        // Sanity check alongside the infeasible case above: a generous
        // budget must still succeed.
        let result = find_optimal_reduction_ratios(1000, 1_000_000, 4, 9);
        assert!(result.is_ok());
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
