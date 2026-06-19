//! Gaussian filtering and querying API for 3DGS scenes.
//!
//! This module provides a flexible, composable system for filtering Gaussians
//! from a 3D Gaussian Splatting scene based on various criteria: opacity,
//! size, position, spatial region, volume, and more. It also supports
//! computing scene statistics and applying standard pruning heuristics.
//!
//! # Example
//! ```rust,no_run
//! use oxigaf_cli::filter_gaussians::{
//!     GaussianData, FilterCriterion, filter_gaussians, PruningConfig, prune_gaussians,
//! };
//!
//! let gaussians: Vec<GaussianData> = Vec::new(); // populate from scene
//! let result = filter_gaussians(
//!     &gaussians,
//!     &FilterCriterion::OpacityAbove(0.1),
//! ).expect("filter failed");
//! println!("{}", result.format_summary());
//! ```

use thiserror::Error;

// ---------------------------------------------------------------------------
// FilterError
// ---------------------------------------------------------------------------

/// Errors that can occur during Gaussian filtering operations.
#[derive(Debug, Error)]
pub enum FilterError {
    /// A threshold value is out of the valid range (e.g., opacity not in \[0,1\]).
    #[error("Invalid threshold: {0}")]
    InvalidThreshold(String),

    /// An AABB region is invalid (e.g., min > max for some axis).
    #[error("Invalid region: {0}")]
    InvalidRegion(String),

    /// The scene contains no Gaussians; statistics cannot be computed.
    #[error("Scene is empty")]
    EmptyScene,

    /// Two filter results have different lengths, so they cannot be combined.
    #[error("Length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    /// The pruning or filter configuration is invalid.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

// ---------------------------------------------------------------------------
// GaussianData
// ---------------------------------------------------------------------------

/// Flat representation of a single Gaussian's properties.
#[derive(Debug, Clone)]
pub struct GaussianData {
    /// World-space center [x, y, z].
    pub position: [f32; 3],
    /// Log-space scales [log_sx, log_sy, log_sz].
    pub log_scale: [f32; 3],
    /// Quaternion rotation [qx, qy, qz, qw].
    pub rotation: [f32; 4],
    /// Opacity in [0, 1] (sigmoid of logit).
    pub opacity: f32,
    /// DC SH color [r, g, b].
    pub color: [f32; 3],
}

impl GaussianData {
    /// Maximum scale (exp of max log_scale component).
    #[must_use]
    pub fn max_scale(&self) -> f32 {
        let max_log = self
            .log_scale
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        max_log.exp()
    }

    /// Volume approximation: product of scales = exp(log_sx + log_sy + log_sz).
    #[must_use]
    pub fn volume(&self) -> f32 {
        let sum_log = self.log_scale[0] + self.log_scale[1] + self.log_scale[2];
        sum_log.exp()
    }

    /// Aspect ratio: max_scale / min_scale, clamped to a minimum of 1.0.
    #[must_use]
    pub fn aspect_ratio(&self) -> f32 {
        let max_log = self
            .log_scale
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_log = self.log_scale.iter().cloned().fold(f32::INFINITY, f32::min);
        let ratio = (max_log - min_log).exp(); // = exp(max) / exp(min), always >= 1
        ratio.max(1.0)
    }

    /// L2 distance from the world origin.
    #[must_use]
    pub fn distance_from_origin(&self) -> f32 {
        let [x, y, z] = self.position;
        (x * x + y * y + z * z).sqrt()
    }

    /// L2 distance from an arbitrary reference point.
    #[must_use]
    pub fn distance_from(&self, point: [f32; 3]) -> f32 {
        let dx = self.position[0] - point[0];
        let dy = self.position[1] - point[1];
        let dz = self.position[2] - point[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

// ---------------------------------------------------------------------------
// FilterCriterion
// ---------------------------------------------------------------------------

/// A composable criterion for filtering Gaussians.
///
/// Criteria can be combined using `And`, `Or`, and `Not` variants to build
/// arbitrarily complex filter expressions.
#[derive(Debug, Clone)]
pub enum FilterCriterion {
    /// Keep if opacity >= min_opacity.
    OpacityAbove(f32),
    /// Keep if opacity <= max_opacity.
    OpacityBelow(f32),
    /// Keep if opacity is in [min, max].
    OpacityRange(f32, f32),
    /// Keep if max_scale >= min_size (in world units).
    SizeAbove(f32),
    /// Keep if max_scale <= max_size.
    SizeBelow(f32),
    /// Keep if max_scale in [min, max].
    SizeRange(f32, f32),
    /// Keep if inside an axis-aligned bounding box.
    InsideAabb { min: [f32; 3], max: [f32; 3] },
    /// Keep if outside an axis-aligned bounding box.
    OutsideAabb { min: [f32; 3], max: [f32; 3] },
    /// Keep if distance from center <= radius.
    InsideSphere { center: [f32; 3], radius: f32 },
    /// Keep if distance from center > radius.
    OutsideSphere { center: [f32; 3], radius: f32 },
    /// Keep if volume is in [min_vol, max_vol] (exp(Σ log_scale)).
    VolumeRange(f32, f32),
    /// Keep if aspect ratio <= max_ratio.
    MaxAspectRatio(f32),
    /// Keep if any SH color channel >= min (bright Gaussians).
    ColorBright(f32),
    /// Logical NOT of a criterion.
    Not(Box<FilterCriterion>),
    /// Logical AND of two criteria.
    And(Box<FilterCriterion>, Box<FilterCriterion>),
    /// Logical OR of two criteria.
    Or(Box<FilterCriterion>, Box<FilterCriterion>),
}

impl FilterCriterion {
    /// Test whether a single Gaussian passes this criterion.
    #[must_use]
    pub fn test(&self, g: &GaussianData) -> bool {
        match self {
            Self::OpacityAbove(threshold) => g.opacity >= *threshold,
            Self::OpacityBelow(threshold) => g.opacity <= *threshold,
            Self::OpacityRange(lo, hi) => g.opacity >= *lo && g.opacity <= *hi,

            Self::SizeAbove(min_size) => g.max_scale() >= *min_size,
            Self::SizeBelow(max_size) => g.max_scale() <= *max_size,
            Self::SizeRange(lo, hi) => {
                let s = g.max_scale();
                s >= *lo && s <= *hi
            }

            Self::InsideAabb { min, max } => {
                let [x, y, z] = g.position;
                x >= min[0]
                    && x <= max[0]
                    && y >= min[1]
                    && y <= max[1]
                    && z >= min[2]
                    && z <= max[2]
            }
            Self::OutsideAabb { min, max } => {
                let [x, y, z] = g.position;
                !(x >= min[0]
                    && x <= max[0]
                    && y >= min[1]
                    && y <= max[1]
                    && z >= min[2]
                    && z <= max[2])
            }

            Self::InsideSphere { center, radius } => g.distance_from(*center) <= *radius,
            Self::OutsideSphere { center, radius } => g.distance_from(*center) > *radius,

            Self::VolumeRange(lo, hi) => {
                let v = g.volume();
                v >= *lo && v <= *hi
            }

            Self::MaxAspectRatio(max_ratio) => g.aspect_ratio() <= *max_ratio,

            Self::ColorBright(min_brightness) => {
                g.color[0] >= *min_brightness
                    || g.color[1] >= *min_brightness
                    || g.color[2] >= *min_brightness
            }

            Self::Not(inner) => !inner.test(g),
            Self::And(a, b) => a.test(g) && b.test(g),
            Self::Or(a, b) => a.test(g) || b.test(g),
        }
    }

    /// Human-readable description of this criterion.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::OpacityAbove(t) => format!("opacity >= {t:.4}"),
            Self::OpacityBelow(t) => format!("opacity <= {t:.4}"),
            Self::OpacityRange(lo, hi) => format!("opacity in [{lo:.4}, {hi:.4}]"),
            Self::SizeAbove(t) => format!("max_scale >= {t:.4}"),
            Self::SizeBelow(t) => format!("max_scale <= {t:.4}"),
            Self::SizeRange(lo, hi) => format!("max_scale in [{lo:.4}, {hi:.4}]"),
            Self::InsideAabb { min, max } => {
                format!(
                    "inside AABB([{:.2},{:.2},{:.2}] to [{:.2},{:.2},{:.2}])",
                    min[0], min[1], min[2], max[0], max[1], max[2]
                )
            }
            Self::OutsideAabb { min, max } => {
                format!(
                    "outside AABB([{:.2},{:.2},{:.2}] to [{:.2},{:.2},{:.2}])",
                    min[0], min[1], min[2], max[0], max[1], max[2]
                )
            }
            Self::InsideSphere { center, radius } => {
                format!(
                    "inside sphere(center=[{:.2},{:.2},{:.2}], r={radius:.4})",
                    center[0], center[1], center[2]
                )
            }
            Self::OutsideSphere { center, radius } => {
                format!(
                    "outside sphere(center=[{:.2},{:.2},{:.2}], r={radius:.4})",
                    center[0], center[1], center[2]
                )
            }
            Self::VolumeRange(lo, hi) => format!("volume in [{lo:.4}, {hi:.4}]"),
            Self::MaxAspectRatio(r) => format!("aspect_ratio <= {r:.4}"),
            Self::ColorBright(t) => format!("any color channel >= {t:.4}"),
            Self::Not(inner) => format!("NOT({})", inner.description()),
            Self::And(a, b) => format!("({} AND {})", a.description(), b.description()),
            Self::Or(a, b) => format!("({} OR {})", a.description(), b.description()),
        }
    }
}

// ---------------------------------------------------------------------------
// FilterResult
// ---------------------------------------------------------------------------

/// The result of applying a filter to a set of Gaussians.
#[derive(Debug, Clone)]
pub struct FilterResult {
    /// Boolean mask: `true` means the Gaussian is kept.
    pub mask: Vec<bool>,
    /// Indices (into the original slice) of Gaussians that were kept.
    pub kept_indices: Vec<usize>,
    /// Total number of Gaussians that were tested.
    pub total: usize,
    /// Number of Gaussians kept.
    pub num_kept: usize,
    /// Number of Gaussians removed.
    pub num_removed: usize,
}

impl FilterResult {
    /// Construct a `FilterResult` from a boolean mask.
    fn from_mask(mask: Vec<bool>) -> Self {
        let total = mask.len();
        let kept_indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
            .collect();
        let num_kept = kept_indices.len();
        let num_removed = total - num_kept;
        Self {
            mask,
            kept_indices,
            total,
            num_kept,
            num_removed,
        }
    }

    /// Fraction of Gaussians that were kept (0.0 if none tested).
    #[must_use]
    pub fn keep_fraction(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.num_kept as f32 / self.total as f32
        }
    }

    /// Format a human-readable summary of this filter result.
    #[must_use]
    pub fn format_summary(&self) -> String {
        format!(
            "FilterResult: kept {}/{} ({:.1}%), removed {}",
            self.num_kept,
            self.total,
            self.keep_fraction() * 100.0,
            self.num_removed,
        )
    }

    /// Invert the filter: previously kept Gaussians are removed and vice versa.
    #[must_use]
    pub fn invert(&self) -> FilterResult {
        let new_mask: Vec<bool> = self.mask.iter().map(|&b| !b).collect();
        Self::from_mask(new_mask)
    }

    /// Combine with another result using AND (keep only if both keep).
    ///
    /// # Errors
    /// Returns [`FilterError::LengthMismatch`] if the two results have
    /// different `total` counts.
    pub fn intersect(&self, other: &FilterResult) -> Result<FilterResult, FilterError> {
        if self.total != other.total {
            return Err(FilterError::LengthMismatch {
                expected: self.total,
                actual: other.total,
            });
        }
        let new_mask: Vec<bool> = self
            .mask
            .iter()
            .zip(other.mask.iter())
            .map(|(&a, &b)| a && b)
            .collect();
        Ok(Self::from_mask(new_mask))
    }

    /// Combine with another result using OR (keep if either keeps).
    ///
    /// # Errors
    /// Returns [`FilterError::LengthMismatch`] if the two results have
    /// different `total` counts.
    pub fn union(&self, other: &FilterResult) -> Result<FilterResult, FilterError> {
        if self.total != other.total {
            return Err(FilterError::LengthMismatch {
                expected: self.total,
                actual: other.total,
            });
        }
        let new_mask: Vec<bool> = self
            .mask
            .iter()
            .zip(other.mask.iter())
            .map(|(&a, &b)| a || b)
            .collect();
        Ok(Self::from_mask(new_mask))
    }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate an AABB: every min[i] must be <= max[i].
fn validate_aabb(min: &[f32; 3], max: &[f32; 3]) -> Result<(), FilterError> {
    for axis in 0..3 {
        if min[axis] > max[axis] {
            return Err(FilterError::InvalidRegion(format!(
                "AABB axis {}: min ({}) > max ({})",
                axis, min[axis], max[axis]
            )));
        }
    }
    Ok(())
}

/// Validate that a criterion's parameters are sound, recursively.
fn validate_criterion(criterion: &FilterCriterion) -> Result<(), FilterError> {
    match criterion {
        FilterCriterion::OpacityAbove(t) | FilterCriterion::OpacityBelow(t)
            if !(*t >= 0.0 && *t <= 1.0) =>
        {
            return Err(FilterError::InvalidThreshold(format!(
                "Opacity threshold {t} is not in [0, 1]"
            )));
        }
        FilterCriterion::OpacityRange(lo, hi) => {
            if !(*lo >= 0.0 && *lo <= 1.0) {
                return Err(FilterError::InvalidThreshold(format!(
                    "Opacity range lower bound {lo} is not in [0, 1]"
                )));
            }
            if !(*hi >= 0.0 && *hi <= 1.0) {
                return Err(FilterError::InvalidThreshold(format!(
                    "Opacity range upper bound {hi} is not in [0, 1]"
                )));
            }
        }
        FilterCriterion::InsideAabb { min, max } => validate_aabb(min, max)?,
        FilterCriterion::OutsideAabb { min, max } => validate_aabb(min, max)?,
        FilterCriterion::Not(inner) => validate_criterion(inner)?,
        FilterCriterion::And(a, b) => {
            validate_criterion(a)?;
            validate_criterion(b)?;
        }
        FilterCriterion::Or(a, b) => {
            validate_criterion(a)?;
            validate_criterion(b)?;
        }
        // Size, sphere, volume, aspect, color — no domain restrictions beyond
        // what is physically meaningful; allow any f32.
        _ => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Core filtering functions
// ---------------------------------------------------------------------------

/// Apply a single filter criterion to a list of Gaussians.
///
/// Returns a [`FilterResult`] describing which Gaussians were kept.
/// An empty input is valid and produces a result with 0 kept.
///
/// # Errors
/// Returns [`FilterError`] if the criterion parameters are invalid.
pub fn filter_gaussians(
    gaussians: &[GaussianData],
    criterion: &FilterCriterion,
) -> Result<FilterResult, FilterError> {
    validate_criterion(criterion)?;
    let mask: Vec<bool> = gaussians.iter().map(|g| criterion.test(g)).collect();
    Ok(FilterResult::from_mask(mask))
}

/// Apply multiple criteria simultaneously (logical AND of all criteria).
///
/// A Gaussian is kept only if it passes every criterion in `criteria`.
/// When `criteria` is empty, all Gaussians are kept.
///
/// # Errors
/// Returns [`FilterError`] if any criterion's parameters are invalid.
pub fn filter_gaussians_multi(
    gaussians: &[GaussianData],
    criteria: &[FilterCriterion],
) -> Result<FilterResult, FilterError> {
    for criterion in criteria {
        validate_criterion(criterion)?;
    }
    let mask: Vec<bool> = gaussians
        .iter()
        .map(|g| criteria.iter().all(|c| c.test(g)))
        .collect();
    Ok(FilterResult::from_mask(mask))
}

/// Apply criteria in a sequential pipeline.
///
/// Each criterion is applied only to the Gaussians that survived all
/// previous criteria. The final mask and indices are relative to the
/// **original** input slice.
///
/// When `criteria` is empty, all Gaussians are kept.
///
/// # Errors
/// Returns [`FilterError`] if any criterion's parameters are invalid.
pub fn filter_gaussians_pipeline(
    gaussians: &[GaussianData],
    criteria: &[FilterCriterion],
) -> Result<FilterResult, FilterError> {
    for criterion in criteria {
        validate_criterion(criterion)?;
    }

    // Start with all original indices alive.
    let mut surviving: Vec<usize> = (0..gaussians.len()).collect();

    for criterion in criteria {
        // Apply criterion to the currently surviving Gaussians.
        let new_surviving: Vec<usize> = surviving
            .iter()
            .copied()
            .filter(|&orig_idx| criterion.test(&gaussians[orig_idx]))
            .collect();
        surviving = new_surviving;
    }

    // Reconstruct a full boolean mask over the original slice.
    let mut mask = vec![false; gaussians.len()];
    for orig_idx in &surviving {
        mask[*orig_idx] = true;
    }
    Ok(FilterResult::from_mask(mask))
}

// ---------------------------------------------------------------------------
// SceneStats
// ---------------------------------------------------------------------------

/// Statistics computed over an entire Gaussian scene.
#[derive(Debug, Clone)]
pub struct SceneStats {
    /// Total number of Gaussians.
    pub total_gaussians: usize,
    /// Mean opacity across all Gaussians.
    pub mean_opacity: f32,
    /// Median opacity across all Gaussians.
    pub median_opacity: f32,
    /// Mean of max_scale across all Gaussians.
    pub mean_max_scale: f32,
    /// Maximum max_scale across all Gaussians.
    pub max_max_scale: f32,
    /// Minimum max_scale across all Gaussians.
    pub min_max_scale: f32,
    /// Mean volume across all Gaussians.
    pub mean_volume: f32,
    /// Sum of volumes across all Gaussians.
    pub total_volume: f32,
    /// Per-axis minimum of all Gaussian positions.
    pub bounding_box_min: [f32; 3],
    /// Per-axis maximum of all Gaussian positions.
    pub bounding_box_max: [f32; 3],
    /// Maximum diagonal length of the scene bounding box.
    pub scene_diameter: f32,
    /// Histogram of opacities bucketed into 10 uniform bins:
    /// `bin[i]` counts Gaussians in \[i*0.1, (i+1)*0.1).
    /// The last bin \[0.9, 1.0\] is inclusive on the right.
    pub opacity_histogram: [u32; 10],
    /// Histogram of max_scale in 10 logarithmically-spaced bins
    /// between the minimum and maximum observed max_scale values.
    pub size_histogram: [u32; 10],
}

impl SceneStats {
    /// Format a human-readable summary of scene statistics.
    #[must_use]
    pub fn format_summary(&self) -> String {
        let [bmin_x, bmin_y, bmin_z] = self.bounding_box_min;
        let [bmax_x, bmax_y, bmax_z] = self.bounding_box_max;
        format!(
            "SceneStats:\n  total: {}\n  opacity: mean={:.4} median={:.4}\n  scale: mean={:.4} min={:.4} max={:.4}\n  volume: mean={:.6} total={:.6}\n  bbox: [{:.3},{:.3},{:.3}] to [{:.3},{:.3},{:.3}]\n  diameter: {:.4}",
            self.total_gaussians,
            self.mean_opacity,
            self.median_opacity,
            self.mean_max_scale,
            self.min_max_scale,
            self.max_max_scale,
            self.mean_volume,
            self.total_volume,
            bmin_x, bmin_y, bmin_z,
            bmax_x, bmax_y, bmax_z,
            self.scene_diameter,
        )
    }

    /// Count Gaussians whose opacity meets or exceeds `opacity_threshold`.
    ///
    /// Uses the opacity histogram; the returned count is approximate
    /// (it counts whole histogram bins whose lower bound >= threshold).
    #[must_use]
    pub fn num_visible(&self, opacity_threshold: f32) -> usize {
        // Bin i covers [i * 0.1, (i+1) * 0.1).
        // We want bins whose lower bound >= threshold, i.e., i * 0.1 >= threshold,
        // i.e., i >= ceil(threshold * 10).
        let first_bin = (opacity_threshold * 10.0).ceil() as i32;
        let first_bin = first_bin.clamp(0, 10) as usize;
        self.opacity_histogram[first_bin..]
            .iter()
            .map(|&c| c as usize)
            .sum()
    }
}

/// Compute [`SceneStats`] for the given list of Gaussians.
///
/// # Errors
/// Returns [`FilterError::EmptyScene`] if `gaussians` is empty.
pub fn compute_scene_stats(gaussians: &[GaussianData]) -> Result<SceneStats, FilterError> {
    if gaussians.is_empty() {
        return Err(FilterError::EmptyScene);
    }

    let n = gaussians.len();

    // --- opacities ---
    let mut opacities: Vec<f32> = gaussians.iter().map(|g| g.opacity).collect();
    let mean_opacity = opacities.iter().sum::<f32>() / n as f32;
    opacities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_opacity = if n.is_multiple_of(2) {
        (opacities[n / 2 - 1] + opacities[n / 2]) / 2.0
    } else {
        opacities[n / 2]
    };

    // --- scales ---
    let scales: Vec<f32> = gaussians.iter().map(|g| g.max_scale()).collect();
    let mean_max_scale = scales.iter().sum::<f32>() / n as f32;
    let max_max_scale = scales.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_max_scale = scales.iter().cloned().fold(f32::INFINITY, f32::min);

    // --- volumes ---
    let volumes: Vec<f32> = gaussians.iter().map(|g| g.volume()).collect();
    let total_volume = volumes.iter().sum::<f32>();
    let mean_volume = total_volume / n as f32;

    // --- bounding box ---
    let mut bbox_min = [f32::INFINITY; 3];
    let mut bbox_max = [f32::NEG_INFINITY; 3];
    for g in gaussians {
        for axis in 0..3 {
            bbox_min[axis] = bbox_min[axis].min(g.position[axis]);
            bbox_max[axis] = bbox_max[axis].max(g.position[axis]);
        }
    }

    let dx = bbox_max[0] - bbox_min[0];
    let dy = bbox_max[1] - bbox_min[1];
    let dz = bbox_max[2] - bbox_min[2];
    let scene_diameter = (dx * dx + dy * dy + dz * dz).sqrt();

    // --- opacity histogram: 10 uniform bins in [0, 1] ---
    let mut opacity_histogram = [0u32; 10];
    for &op in &opacities {
        let bin = ((op * 10.0).floor() as usize).min(9);
        opacity_histogram[bin] += 1;
    }

    // --- size histogram: 10 log-spaced bins ---
    let size_histogram = build_size_histogram(&scales, min_max_scale, max_max_scale);

    Ok(SceneStats {
        total_gaussians: n,
        mean_opacity,
        median_opacity,
        mean_max_scale,
        max_max_scale,
        min_max_scale,
        mean_volume,
        total_volume,
        bounding_box_min: bbox_min,
        bounding_box_max: bbox_max,
        scene_diameter,
        opacity_histogram,
        size_histogram,
    })
}

/// Build a 10-bin logarithmically-spaced histogram for scale values.
fn build_size_histogram(scales: &[f32], min_scale: f32, max_scale: f32) -> [u32; 10] {
    let mut histogram = [0u32; 10];

    // Guard against degenerate cases.
    let min_s = min_scale.max(f32::MIN_POSITIVE);
    let max_s = max_scale.max(f32::MIN_POSITIVE);

    if (min_s - max_s).abs() < f32::EPSILON * max_s.abs().max(1.0) {
        // All scales are identical — put everything in bin 0.
        histogram[0] = scales.len() as u32;
        return histogram;
    }

    let log_min = min_s.ln();
    let log_max = max_s.ln();
    let log_range = log_max - log_min;

    for &s in scales {
        let s_clamped = s.max(min_s);
        let log_s = s_clamped.ln();
        let frac = (log_s - log_min) / log_range;
        let bin = ((frac * 10.0).floor() as usize).min(9);
        histogram[bin] += 1;
    }

    histogram
}

// ---------------------------------------------------------------------------
// PruningConfig and prune_gaussians
// ---------------------------------------------------------------------------

/// Configuration for standard 3DGS Gaussian pruning.
#[derive(Debug, Clone)]
pub struct PruningConfig {
    /// Remove Gaussians with opacity below this threshold. Default: 0.005.
    pub min_opacity: f32,
    /// Remove Gaussians with max_scale below this size. Default: 0.0 (no pruning).
    pub min_size: f32,
    /// Remove Gaussians with max_scale above this size. Default: f32::MAX.
    pub max_size: f32,
    /// Remove Gaussians with aspect ratio above this value. Default: f32::MAX.
    pub max_aspect_ratio: f32,
    /// If `Some(n)`, among passing Gaussians, keep only the top `n` by opacity.
    pub keep_top_n: Option<usize>,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            min_opacity: 0.005,
            min_size: 0.0,
            max_size: f32::MAX,
            max_aspect_ratio: f32::MAX,
            keep_top_n: None,
        }
    }
}

/// Generate a pruning [`FilterResult`] using standard 3DGS heuristics.
///
/// The pruning applies opacity, size, and aspect-ratio criteria simultaneously.
/// If `config.keep_top_n` is set, among the surviving Gaussians, only the
/// `n` with the highest opacity are retained.
///
/// # Errors
/// Returns [`FilterError`] if the configuration is invalid.
pub fn prune_gaussians(
    gaussians: &[GaussianData],
    config: &PruningConfig,
) -> Result<FilterResult, FilterError> {
    if config.min_opacity < 0.0 || config.min_opacity > 1.0 {
        return Err(FilterError::InvalidConfig(format!(
            "min_opacity {} is not in [0, 1]",
            config.min_opacity
        )));
    }
    if config.min_size > config.max_size {
        return Err(FilterError::InvalidConfig(format!(
            "min_size ({}) > max_size ({})",
            config.min_size, config.max_size
        )));
    }

    // Build initial boolean mask from combined criteria.
    let mut mask: Vec<bool> = gaussians
        .iter()
        .map(|g| {
            let opacity_ok = g.opacity >= config.min_opacity;
            let size_ok = {
                let s = g.max_scale();
                s >= config.min_size && s <= config.max_size
            };
            let aspect_ok = g.aspect_ratio() <= config.max_aspect_ratio;
            opacity_ok && size_ok && aspect_ok
        })
        .collect();

    // If keep_top_n is set, among survivors keep only the N with highest opacity.
    if let Some(top_n) = config.keep_top_n {
        // Collect (original_index, opacity) for passing Gaussians.
        let mut survivors: Vec<(usize, f32)> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| {
                if keep {
                    Some((i, gaussians[i].opacity))
                } else {
                    None
                }
            })
            .collect();

        if survivors.len() > top_n {
            // Sort descending by opacity.
            survivors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            // Mark the tail as removed.
            let to_remove: Vec<usize> = survivors[top_n..].iter().map(|&(i, _)| i).collect();
            for i in to_remove {
                mask[i] = false;
            }
        }
    }

    Ok(FilterResult::from_mask(mask))
}

// ---------------------------------------------------------------------------
// Filter presets
// ---------------------------------------------------------------------------

/// Remove near-transparent Gaussians (opacity < 0.01).
///
/// Equivalent to `filter_gaussians(g, &OpacityAbove(0.01))`.
///
/// # Errors
/// Returns [`FilterError`] on invalid input.
pub fn filter_transparent(gaussians: &[GaussianData]) -> Result<FilterResult, FilterError> {
    filter_gaussians(gaussians, &FilterCriterion::OpacityAbove(0.01))
}

/// Keep only dominant Gaussians (opacity > 0.5).
///
/// Equivalent to `filter_gaussians(g, &OpacityAbove(0.5))`.
///
/// # Errors
/// Returns [`FilterError`] on invalid input.
pub fn filter_dominant(gaussians: &[GaussianData]) -> Result<FilterResult, FilterError> {
    filter_gaussians(gaussians, &FilterCriterion::OpacityAbove(0.5))
}

/// Remove spatial outliers: Gaussians beyond 3 standard deviations from the
/// scene centroid (computed over all Gaussian positions).
///
/// If all Gaussians share the same position (std dev ≈ 0), all are kept.
///
/// # Errors
/// Returns [`FilterError::EmptyScene`] if `gaussians` is empty.
pub fn filter_spatial_outliers(gaussians: &[GaussianData]) -> Result<FilterResult, FilterError> {
    if gaussians.is_empty() {
        return Err(FilterError::EmptyScene);
    }

    let n = gaussians.len() as f32;

    // Compute centroid.
    let mut centroid = [0.0f32; 3];
    for g in gaussians {
        centroid[0] += g.position[0];
        centroid[1] += g.position[1];
        centroid[2] += g.position[2];
    }
    centroid[0] /= n;
    centroid[1] /= n;
    centroid[2] /= n;

    // Compute distances from centroid.
    let distances: Vec<f32> = gaussians
        .iter()
        .map(|g| g.distance_from(centroid))
        .collect();

    // Compute mean and std dev of distances.
    let mean_dist = distances.iter().sum::<f32>() / n;
    let variance = distances
        .iter()
        .map(|&d| {
            let diff = d - mean_dist;
            diff * diff
        })
        .sum::<f32>()
        / n;
    let std_dev = variance.sqrt();

    // If std_dev is effectively zero, keep everything.
    if std_dev < f32::EPSILON * mean_dist.abs().max(1.0) {
        let mask = vec![true; gaussians.len()];
        return Ok(FilterResult::from_mask(mask));
    }

    let threshold = mean_dist + 3.0 * std_dev;
    let mask: Vec<bool> = distances.iter().map(|&d| d <= threshold).collect();
    Ok(FilterResult::from_mask(mask))
}

/// Remove anisotropic (highly elongated) Gaussians.
///
/// Keeps only Gaussians whose aspect ratio (max_scale / min_scale) is
/// <= `max_ratio`.
///
/// # Errors
/// Returns [`FilterError`] on invalid input.
pub fn filter_anisotropic(
    gaussians: &[GaussianData],
    max_ratio: f32,
) -> Result<FilterResult, FilterError> {
    filter_gaussians(gaussians, &FilterCriterion::MaxAspectRatio(max_ratio))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a simple isotropic Gaussian with given position, opacity,
    /// and a uniform log_scale on all three axes.
    fn make_gaussian(pos: [f32; 3], opacity: f32, log_scale: f32) -> GaussianData {
        GaussianData {
            position: pos,
            log_scale: [log_scale; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            opacity,
            color: [0.5, 0.5, 0.5],
        }
    }

    /// Helper: create a Gaussian with separate log_scale values per axis.
    fn make_anisotropic(pos: [f32; 3], opacity: f32, log_scales: [f32; 3]) -> GaussianData {
        GaussianData {
            position: pos,
            log_scale: log_scales,
            rotation: [0.0, 0.0, 0.0, 1.0],
            opacity,
            color: [0.5, 0.5, 0.5],
        }
    }

    // --- Test 1: GaussianData::max_scale ---
    #[test]
    fn test_max_scale_isotropic() {
        let g = make_gaussian([0.0; 3], 0.5, 2.0_f32.ln()); // log_scale = ln(2) → scale = 2
        let expected = 2.0_f32;
        assert!(
            (g.max_scale() - expected).abs() < 1e-5,
            "max_scale should be exp(ln(2)) = 2.0, got {}",
            g.max_scale()
        );
    }

    #[test]
    fn test_max_scale_anisotropic() {
        // log_scales: ln(1), ln(3), ln(5) → max scale = 5
        let g = make_anisotropic(
            [0.0; 3],
            0.5,
            [
                0.0_f32.ln().max(f32::NEG_INFINITY),
                3.0_f32.ln(),
                5.0_f32.ln(),
            ],
        );
        // axis 0: ln(1) = 0  → scale = 1
        // axis 1: ln(3)      → scale = 3
        // axis 2: ln(5)      → scale = 5, this is max
        assert!(
            (g.max_scale() - 5.0).abs() < 1e-5,
            "max_scale should be 5.0"
        );
    }

    // --- Test 2: GaussianData::volume ---
    #[test]
    fn test_volume_isotropic() {
        let log_s = 1.0_f32; // scale = e
        let g = make_gaussian([0.0; 3], 0.5, log_s);
        // volume = exp(3 * log_s) = e^3
        let expected = (3.0_f32 * log_s).exp();
        assert!(
            (g.volume() - expected).abs() < 1e-5,
            "volume should be e^3 = {}, got {}",
            expected,
            g.volume()
        );
    }

    // --- Test 3: GaussianData::aspect_ratio isotropic ---
    #[test]
    fn test_aspect_ratio_isotropic() {
        let g = make_gaussian([0.0; 3], 0.5, 1.0);
        assert!(
            (g.aspect_ratio() - 1.0).abs() < 1e-6,
            "isotropic Gaussian should have aspect_ratio = 1.0, got {}",
            g.aspect_ratio()
        );
    }

    // --- Test 4: FilterCriterion::OpacityAbove ---
    #[test]
    fn test_opacity_above_threshold() {
        let g_hi = make_gaussian([0.0; 3], 0.8, 0.0);
        let g_lo = make_gaussian([0.0; 3], 0.3, 0.0);
        let g_eq = make_gaussian([0.0; 3], 0.5, 0.0);
        let c = FilterCriterion::OpacityAbove(0.5);
        assert!(c.test(&g_hi), "0.8 >= 0.5 should pass");
        assert!(!c.test(&g_lo), "0.3 >= 0.5 should fail");
        assert!(c.test(&g_eq), "0.5 >= 0.5 should pass (inclusive)");
    }

    // --- Test 5: FilterCriterion::InsideAabb ---
    #[test]
    fn test_inside_aabb() {
        let c = FilterCriterion::InsideAabb {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        };
        let inside = make_gaussian([0.5, 0.5, 0.5], 0.5, 0.0);
        let outside = make_gaussian([2.0, 0.5, 0.5], 0.5, 0.0);
        let on_edge = make_gaussian([1.0, 1.0, 1.0], 0.5, 0.0);
        assert!(c.test(&inside), "center should be inside");
        assert!(!c.test(&outside), "outside x=2 should fail");
        assert!(c.test(&on_edge), "on edge should be inside (inclusive)");
    }

    // --- Test 6: FilterCriterion::InsideSphere ---
    #[test]
    fn test_inside_sphere() {
        let c = FilterCriterion::InsideSphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        };
        let inside = make_gaussian([0.5, 0.0, 0.0], 0.5, 0.0);
        let outside = make_gaussian([2.0, 0.0, 0.0], 0.5, 0.0);
        assert!(c.test(&inside), "distance 0.5 <= 1.0 should be inside");
        assert!(!c.test(&outside), "distance 2.0 > 1.0 should be outside");
    }

    // --- Test 7: FilterCriterion::Not ---
    #[test]
    fn test_not_criterion() {
        let inner = FilterCriterion::OpacityAbove(0.5);
        let not_c = FilterCriterion::Not(Box::new(inner));
        let g_hi = make_gaussian([0.0; 3], 0.8, 0.0);
        let g_lo = make_gaussian([0.0; 3], 0.2, 0.0);
        assert!(!not_c.test(&g_hi), "NOT(0.8 >= 0.5) should be false");
        assert!(not_c.test(&g_lo), "NOT(0.2 >= 0.5) should be true");
    }

    // --- Test 8: FilterCriterion::And ---
    #[test]
    fn test_and_criterion() {
        let a = FilterCriterion::OpacityAbove(0.3);
        let b = FilterCriterion::SizeBelow(5.0);
        let and_c = FilterCriterion::And(Box::new(a), Box::new(b));
        // opacity=0.5 (passes a), max_scale=exp(0)=1.0 (passes b)
        let pass = make_gaussian([0.0; 3], 0.5, 0.0);
        // opacity=0.1 (fails a), max_scale=1.0 (passes b)
        let fail_a = make_gaussian([0.0; 3], 0.1, 0.0);
        // opacity=0.5 (passes a), max_scale=exp(3)≈20.1 (fails b)
        let fail_b = make_gaussian([0.0; 3], 0.5, 3.0);
        assert!(and_c.test(&pass), "both pass → AND should pass");
        assert!(!and_c.test(&fail_a), "a fails → AND should fail");
        assert!(!and_c.test(&fail_b), "b fails → AND should fail");
    }

    // --- Test 9: FilterCriterion::Or ---
    #[test]
    fn test_or_criterion() {
        let a = FilterCriterion::OpacityAbove(0.9);
        let b = FilterCriterion::SizeBelow(0.5);
        let or_c = FilterCriterion::Or(Box::new(a), Box::new(b));
        // opacity=0.95 passes a → OR passes
        let pass_a = make_gaussian([0.0; 3], 0.95, 0.0);
        // max_scale=exp(-1)≈0.37 < 0.5 passes b → OR passes
        let pass_b = make_gaussian([0.0; 3], 0.3, -1.0);
        // opacity=0.5 fails a, max_scale=1.0 fails b
        let fail_both = make_gaussian([0.0; 3], 0.5, 0.0);
        assert!(or_c.test(&pass_a), "a passes → OR should pass");
        assert!(or_c.test(&pass_b), "b passes → OR should pass");
        assert!(!or_c.test(&fail_both), "both fail → OR should fail");
    }

    // --- Test 10: filter_gaussians empty input ---
    #[test]
    fn test_filter_gaussians_empty() {
        let result = filter_gaussians(&[], &FilterCriterion::OpacityAbove(0.5))
            .expect("empty input should not error");
        assert_eq!(result.total, 0);
        assert_eq!(result.num_kept, 0);
        assert_eq!(result.num_removed, 0);
        assert!(result.kept_indices.is_empty());
    }

    // --- Test 11: filter_gaussians all pass ---
    #[test]
    fn test_filter_gaussians_all_pass() {
        let gaussians = vec![
            make_gaussian([0.0; 3], 0.8, 0.0),
            make_gaussian([1.0, 0.0, 0.0], 0.9, 0.0),
            make_gaussian([0.0, 1.0, 0.0], 0.7, 0.0),
        ];
        let result = filter_gaussians(&gaussians, &FilterCriterion::OpacityAbove(0.5))
            .expect("should succeed");
        assert_eq!(result.total, 3);
        assert_eq!(result.num_kept, 3);
        assert_eq!(result.num_removed, 0);
        assert_eq!(result.kept_indices, vec![0, 1, 2]);
    }

    // --- Test 12: filter_gaussians mixed ---
    #[test]
    fn test_filter_gaussians_mixed() {
        let gaussians = vec![
            make_gaussian([0.0; 3], 0.8, 0.0), // kept (0.8 >= 0.5)
            make_gaussian([0.0; 3], 0.2, 0.0), // removed
            make_gaussian([0.0; 3], 0.6, 0.0), // kept
            make_gaussian([0.0; 3], 0.1, 0.0), // removed
        ];
        let result = filter_gaussians(&gaussians, &FilterCriterion::OpacityAbove(0.5))
            .expect("should succeed");
        assert_eq!(result.total, 4);
        assert_eq!(result.num_kept, 2);
        assert_eq!(result.num_removed, 2);
        assert_eq!(result.kept_indices, vec![0, 2]);
        assert_eq!(result.mask, vec![true, false, true, false]);
    }

    // --- Test 13: FilterResult::keep_fraction ---
    #[test]
    fn test_keep_fraction() {
        let gaussians = vec![
            make_gaussian([0.0; 3], 0.8, 0.0),
            make_gaussian([0.0; 3], 0.2, 0.0),
            make_gaussian([0.0; 3], 0.6, 0.0),
            make_gaussian([0.0; 3], 0.1, 0.0),
        ];
        let result = filter_gaussians(&gaussians, &FilterCriterion::OpacityAbove(0.5))
            .expect("should succeed");
        let frac = result.keep_fraction();
        assert!(
            (frac - 0.5).abs() < 1e-6,
            "keep_fraction should be 0.5, got {frac}"
        );
    }

    #[test]
    fn test_keep_fraction_empty() {
        let result =
            filter_gaussians(&[], &FilterCriterion::OpacityAbove(0.5)).expect("should succeed");
        assert_eq!(
            result.keep_fraction(),
            0.0,
            "empty → keep_fraction should be 0.0"
        );
    }

    // --- Test 14: FilterResult::invert ---
    #[test]
    fn test_filter_result_invert() {
        let gaussians = vec![
            make_gaussian([0.0; 3], 0.8, 0.0),
            make_gaussian([0.0; 3], 0.2, 0.0),
        ];
        let result = filter_gaussians(&gaussians, &FilterCriterion::OpacityAbove(0.5))
            .expect("should succeed");
        let inverted = result.invert();
        assert_eq!(inverted.total, 2);
        assert_eq!(inverted.num_kept, 1);
        assert_eq!(inverted.kept_indices, vec![1]);
        assert_eq!(inverted.mask, vec![false, true]);
    }

    // --- Test 15: FilterResult::intersect ---
    #[test]
    fn test_filter_result_intersect() {
        let gaussians = vec![
            make_gaussian([0.0, 0.0, 0.0], 0.8, 0.0),
            make_gaussian([0.5, 0.0, 0.0], 0.6, 0.0),
            make_gaussian([2.0, 0.0, 0.0], 0.8, 0.0),
        ];
        let opacity_result = filter_gaussians(&gaussians, &FilterCriterion::OpacityAbove(0.5))
            .expect("opacity filter");
        let aabb_result = filter_gaussians(
            &gaussians,
            &FilterCriterion::InsideAabb {
                min: [0.0; 3],
                max: [1.0, 1.0, 1.0],
            },
        )
        .expect("aabb filter");
        // opacity passes: [0, 1, 2]; aabb passes: [0, 1]; intersect: [0, 1]
        let intersection = opacity_result.intersect(&aabb_result).expect("intersect");
        assert_eq!(intersection.kept_indices, vec![0, 1]);
    }

    // --- Test 16: FilterResult::union ---
    #[test]
    fn test_filter_result_union() {
        let gaussians = vec![
            make_gaussian([0.0, 0.0, 0.0], 0.8, 0.0),  // passes opacity
            make_gaussian([5.0, 0.0, 0.0], 0.3, -2.0), // small scale, fails opacity
            make_gaussian([5.0, 0.0, 0.0], 0.1, 0.0),  // fails both
        ];
        let opacity_result = filter_gaussians(&gaussians, &FilterCriterion::OpacityAbove(0.5))
            .expect("opacity filter");
        // max_scale = exp(-2) ≈ 0.135
        let size_result =
            filter_gaussians(&gaussians, &FilterCriterion::SizeBelow(0.2)).expect("size filter");
        let union = opacity_result.union(&size_result).expect("union");
        // opacity passes [0]; size passes [1]; union: [0, 1]
        assert_eq!(union.kept_indices, vec![0, 1]);
    }

    // --- Test 17: filter_gaussians_multi ---
    #[test]
    fn test_filter_gaussians_multi() {
        let gaussians = vec![
            make_gaussian([0.5, 0.5, 0.5], 0.8, 0.0), // passes opacity+aabb
            make_gaussian([0.5, 0.5, 0.5], 0.2, 0.0), // fails opacity
            make_gaussian([5.0, 0.5, 0.5], 0.8, 0.0), // fails aabb
        ];
        let criteria = vec![
            FilterCriterion::OpacityAbove(0.5),
            FilterCriterion::InsideAabb {
                min: [0.0; 3],
                max: [1.0, 1.0, 1.0],
            },
        ];
        let result = filter_gaussians_multi(&gaussians, &criteria).expect("multi filter");
        assert_eq!(result.kept_indices, vec![0]);
    }

    // --- Test 18: compute_scene_stats empty ---
    #[test]
    fn test_compute_scene_stats_empty() {
        let result = compute_scene_stats(&[]);
        assert!(
            matches!(result, Err(FilterError::EmptyScene)),
            "empty input should return EmptyScene error"
        );
    }

    // --- Test 19: compute_scene_stats basic ---
    #[test]
    fn test_compute_scene_stats_basic() {
        let gaussians = vec![
            make_gaussian([0.0, 0.0, 0.0], 0.2, 0.0), // scale=1, vol=1
            make_gaussian([2.0, 0.0, 0.0], 0.8, 0.0), // scale=1, vol=1
        ];
        let stats = compute_scene_stats(&gaussians).expect("stats");
        assert_eq!(stats.total_gaussians, 2);
        assert!(
            (stats.mean_opacity - 0.5).abs() < 1e-5,
            "mean opacity should be 0.5"
        );
        assert!(
            (stats.median_opacity - 0.5).abs() < 1e-5,
            "median opacity should be 0.5"
        );
        assert!(
            (stats.mean_max_scale - 1.0).abs() < 1e-5,
            "mean scale should be 1.0"
        );
        assert!(
            (stats.max_max_scale - 1.0).abs() < 1e-5,
            "max scale should be 1.0"
        );
        assert!(
            (stats.min_max_scale - 1.0).abs() < 1e-5,
            "min scale should be 1.0"
        );
        assert!(
            (stats.scene_diameter - 2.0).abs() < 1e-5,
            "diameter should be 2.0"
        );
    }

    // --- Test 20: SceneStats::num_visible ---
    #[test]
    fn test_num_visible() {
        let gaussians = vec![
            make_gaussian([0.0; 3], 0.05, 0.0), // bin 0 [0.0, 0.1)
            make_gaussian([0.0; 3], 0.15, 0.0), // bin 1 [0.1, 0.2)
            make_gaussian([0.0; 3], 0.55, 0.0), // bin 5 [0.5, 0.6)
            make_gaussian([0.0; 3], 0.95, 0.0), // bin 9 [0.9, 1.0]
        ];
        let stats = compute_scene_stats(&gaussians).expect("stats");
        // num_visible(0.5): bins 5..=9 → 2 Gaussians
        assert_eq!(stats.num_visible(0.5), 2, "2 Gaussians above 0.5 threshold");
        // num_visible(0.0): all bins → 4
        assert_eq!(
            stats.num_visible(0.0),
            4,
            "all 4 Gaussians above 0.0 threshold"
        );
        // num_visible(1.0): only bin 10+ → 0 (nothing above exactly 1.0 threshold)
        assert_eq!(stats.num_visible(1.0), 0, "no Gaussian above 1.0 threshold");
    }

    // --- Test 21: prune_gaussians ---
    #[test]
    fn test_prune_gaussians_opacity() {
        let gaussians = vec![
            make_gaussian([0.0; 3], 0.001, 0.0), // below min_opacity
            make_gaussian([0.0; 3], 0.01, 0.0),  // above min_opacity
            make_gaussian([0.0; 3], 0.5, 0.0),   // well above
        ];
        let config = PruningConfig {
            min_opacity: 0.005,
            ..Default::default()
        };
        let result = prune_gaussians(&gaussians, &config).expect("prune");
        assert_eq!(result.num_kept, 2);
        assert_eq!(result.kept_indices, vec![1, 2]);
    }

    #[test]
    fn test_prune_gaussians_keep_top_n() {
        // 4 Gaussians all above min_opacity; keep only top 2 by opacity.
        let gaussians = vec![
            make_gaussian([0.0; 3], 0.9, 0.0),
            make_gaussian([0.0; 3], 0.3, 0.0),
            make_gaussian([0.0; 3], 0.7, 0.0),
            make_gaussian([0.0; 3], 0.1, 0.0), // below min_opacity 0.005 check (but above it)
        ];
        let config = PruningConfig {
            min_opacity: 0.005,
            keep_top_n: Some(2),
            ..Default::default()
        };
        let result = prune_gaussians(&gaussians, &config).expect("prune with top_n");
        assert_eq!(result.num_kept, 2, "should keep exactly 2");
        // Top 2 by opacity are index 0 (0.9) and index 2 (0.7).
        assert!(result.mask[0], "index 0 (opacity 0.9) should be kept");
        assert!(result.mask[2], "index 2 (opacity 0.7) should be kept");
        assert!(!result.mask[1], "index 1 (opacity 0.3) should be removed");
        assert!(!result.mask[3], "index 3 (opacity 0.1) should be removed");
    }

    // --- Test 22: filter_spatial_outliers ---
    #[test]
    fn test_filter_spatial_outliers_removes_far_gaussian() {
        // Build a tight cluster of 20 Gaussians near the origin, then add one
        // outlier that is far enough away to be beyond 3 sigma.
        // With a large cluster (n=20), the centroid is close to origin, and
        // the outlier at 5.0 is well beyond mean + 3*std_dev.
        let mut gaussians: Vec<GaussianData> = (0..20)
            .map(|i| {
                let x = (i as f32 - 10.0) * 0.01_f32; // -0.10 .. +0.09
                make_gaussian([x, 0.0, 0.0], 0.5, 0.0)
            })
            .collect();
        // Outlier at 5.0 on x-axis (well beyond 3-sigma of the cluster).
        gaussians.push(make_gaussian([5.0, 0.0, 0.0], 0.5, 0.0));

        let outlier_idx = gaussians.len() - 1;
        let result = filter_spatial_outliers(&gaussians).expect("spatial outliers");
        assert_eq!(result.total, 21);
        assert_eq!(result.num_kept, 20, "outlier should be removed");
        assert!(
            !result.mask[outlier_idx],
            "far outlier should be filtered out"
        );
    }

    // --- Test 23: filter_transparent ---
    #[test]
    fn test_filter_transparent_removes_near_zero() {
        let gaussians = vec![
            make_gaussian([0.0; 3], 0.005, 0.0), // < 0.01 → removed
            make_gaussian([0.0; 3], 0.009, 0.0), // < 0.01 → removed
            make_gaussian([0.0; 3], 0.01, 0.0),  // >= 0.01 → kept
            make_gaussian([0.0; 3], 0.5, 0.0),   // kept
        ];
        let result = filter_transparent(&gaussians).expect("filter_transparent");
        assert_eq!(result.num_kept, 2);
        assert_eq!(result.kept_indices, vec![2, 3]);
    }

    // --- Test 24: filter_anisotropic ---
    #[test]
    fn test_filter_anisotropic_removes_elongated() {
        // Isotropic: aspect_ratio = 1.0
        let isotropic = make_gaussian([0.0; 3], 0.5, 0.0);
        // Anisotropic: max_scale=exp(3), min_scale=exp(0) → ratio=exp(3)≈20
        let elongated = make_anisotropic([0.0; 3], 0.5, [0.0, 0.0, 3.0]);

        let gaussians = vec![isotropic, elongated];
        let result = filter_anisotropic(&gaussians, 5.0).expect("filter_anisotropic");
        assert_eq!(result.num_kept, 1);
        assert!(result.mask[0], "isotropic should be kept");
        assert!(!result.mask[1], "elongated should be removed");
    }

    // --- Additional test: filter_gaussians_pipeline ---
    #[test]
    fn test_filter_gaussians_pipeline_tracks_original_indices() {
        let gaussians = vec![
            make_gaussian([0.5, 0.5, 0.5], 0.8, 0.0), // idx 0: opacity ok, in AABB
            make_gaussian([0.5, 0.5, 0.5], 0.2, 0.0), // idx 1: opacity fail
            make_gaussian([5.0, 0.5, 0.5], 0.8, 0.0), // idx 2: opacity ok, AABB fail
            make_gaussian([0.5, 0.5, 0.5], 0.9, 0.0), // idx 3: opacity ok, in AABB
        ];
        let criteria = vec![
            FilterCriterion::OpacityAbove(0.5),
            FilterCriterion::InsideAabb {
                min: [0.0; 3],
                max: [1.0, 1.0, 1.0],
            },
        ];
        let result = filter_gaussians_pipeline(&gaussians, &criteria).expect("pipeline");
        // After step 1 (opacity): idx 0, 2, 3 survive.
        // After step 2 (aabb): idx 0 and 3 survive (idx 2 at x=5 fails).
        assert_eq!(result.kept_indices, vec![0, 3]);
        assert!(result.mask[0]);
        assert!(!result.mask[1]);
        assert!(!result.mask[2]);
        assert!(result.mask[3]);
    }

    // --- Additional test: invalid AABB validation ---
    #[test]
    fn test_invalid_aabb_returns_error() {
        let gaussians = vec![make_gaussian([0.0; 3], 0.5, 0.0)];
        let bad_criterion = FilterCriterion::InsideAabb {
            min: [1.0, 0.0, 0.0],
            max: [0.0, 1.0, 1.0], // min[0] > max[0]
        };
        let result = filter_gaussians(&gaussians, &bad_criterion);
        assert!(
            matches!(result, Err(FilterError::InvalidRegion(_))),
            "inverted AABB should produce InvalidRegion error"
        );
    }

    // --- Additional test: FilterResult::intersect length mismatch ---
    #[test]
    fn test_intersect_length_mismatch() {
        let a = FilterResult::from_mask(vec![true, false]);
        let b = FilterResult::from_mask(vec![true, false, true]);
        let result = a.intersect(&b);
        assert!(
            matches!(
                result,
                Err(FilterError::LengthMismatch {
                    expected: 2,
                    actual: 3
                })
            ),
            "mismatched lengths should return LengthMismatch error"
        );
    }

    // --- Additional test: description strings ---
    #[test]
    fn test_criterion_description() {
        let c = FilterCriterion::And(
            Box::new(FilterCriterion::OpacityAbove(0.1)),
            Box::new(FilterCriterion::SizeBelow(2.0)),
        );
        let desc = c.description();
        assert!(
            desc.contains("opacity"),
            "description should mention opacity"
        );
        assert!(
            desc.contains("max_scale"),
            "description should mention scale"
        );
        assert!(desc.contains("AND"), "description should mention AND");
    }

    // --- Additional test: FilterResult format_summary ---
    #[test]
    fn test_format_summary() {
        let gaussians = vec![
            make_gaussian([0.0; 3], 0.8, 0.0),
            make_gaussian([0.0; 3], 0.2, 0.0),
        ];
        let result = filter_gaussians(&gaussians, &FilterCriterion::OpacityAbove(0.5))
            .expect("should succeed");
        let summary = result.format_summary();
        assert!(summary.contains("1/2"), "summary should mention kept/total");
        assert!(
            summary.contains("50.0"),
            "summary should mention percentage"
        );
    }
}
