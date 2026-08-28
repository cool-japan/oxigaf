//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use thiserror::Error;

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
/// Flat scene attribute slices for
/// [`generate_lod_level`](crate::lod_generator::generate_lod_level).
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
/// Parameters for LOD selection based on viewing distance.
#[derive(Debug, Clone)]
pub struct LodSelector {
    /// Distance thresholds for each LOD switch.
    pub thresholds: Vec<f32>,
}
impl LodSelector {
    /// Create a selector with given distance thresholds.
    pub fn new(thresholds: Vec<f32>) -> Self {
        Self { thresholds }
    }
}
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
        Ok(&self.levels[self.levels.len() - 1])
    }
}
