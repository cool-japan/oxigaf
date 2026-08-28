//! Tools to inspect a trained 3DGS Gaussian model.
//!
//! Provides spatial queries, anomaly detection, property distribution analysis,
//! and summary reporting for 3D Gaussian Splatting scenes.
//!
//! # Example
//! ```rust,no_run
//! use oxigaf_cli::model_inspector::{InspectableModel, inspect_model, format_inspection_report};
//!
//! let positions = vec![0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0];
//! let opacities = vec![0.8_f32, 0.5];
//! let scales = vec![-2.0_f32, -2.0, -2.0, -2.0, -2.0, -2.0];
//! let colors = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0];
//! let model = InspectableModel::new(positions, opacities, scales, colors)
//!     .expect("valid model");
//! let report = inspect_model(&model).expect("inspection failed");
//! println!("{}", format_inspection_report(&report));
//! ```

use rayon::prelude::*;
use thiserror::Error;

// ---------------------------------------------------------------------------
// InspectorError
// ---------------------------------------------------------------------------

/// Errors that can arise during model inspection.
#[derive(Debug, Error)]
pub enum InspectorError {
    /// The model contains no Gaussians.
    #[error("Empty model")]
    EmptyModel,

    /// An invalid query parameter was provided.
    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    /// A Gaussian index is out of bounds.
    #[error("Index out of bounds: {index} (model has {count} Gaussians)")]
    IndexOutOfBounds { index: usize, count: usize },

    /// A radius value is invalid (e.g., negative).
    #[error("Invalid radius: {0}")]
    InvalidRadius(f32),

    /// A dimension mismatch between arrays.
    #[error("Dimension error: {0}")]
    DimensionError(String),

    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// BoundingBox3d
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box in 3D.
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox3d {
    /// Minimum corner.
    pub min: [f32; 3],
    /// Maximum corner.
    pub max: [f32; 3],
}

impl BoundingBox3d {
    /// Create a new bounding box from min and max corners.
    #[must_use]
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Compute the center of the bounding box.
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Compute the size (max - min) per axis.
    #[must_use]
    pub fn size(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// Compute the diagonal (L2 length of size vector).
    #[must_use]
    pub fn diagonal(&self) -> f32 {
        let s = self.size();
        (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt()
    }

    /// Check whether a point is inside (inclusive) the bounding box.
    #[must_use]
    pub fn contains(&self, point: &[f32; 3]) -> bool {
        point[0] >= self.min[0]
            && point[0] <= self.max[0]
            && point[1] >= self.min[1]
            && point[1] <= self.max[1]
            && point[2] >= self.min[2]
            && point[2] <= self.max[2]
    }

    /// Expand this bounding box to include another box.
    #[must_use]
    pub fn expand(&self, other: &BoundingBox3d) -> BoundingBox3d {
        BoundingBox3d {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// GaussianProperties
// ---------------------------------------------------------------------------

/// Inspectable properties of a single Gaussian.
#[derive(Debug, Clone)]
pub struct GaussianProperties {
    /// Index within the model.
    pub index: usize,
    /// World-space position.
    pub position: [f32; 3],
    /// Activated opacity (sigmoid), in [0, 1].
    pub opacity: f32,
    /// Largest scale (world units, post-exp).
    pub max_scale: f32,
    /// Smallest scale (world units, post-exp).
    pub min_scale: f32,
    /// Anisotropy ratio: max_scale / (min_scale + 1e-8).
    pub anisotropy: f32,
    /// RGB color in [0, 1].
    pub color: [f32; 3],
    /// Approximate volume: product of the three scales.
    pub volume: f32,
}

// ---------------------------------------------------------------------------
// QueryResult
// ---------------------------------------------------------------------------

/// Result of a spatial or property query.
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Indices of matching Gaussians.
    pub indices: Vec<usize>,
    /// Number of matching Gaussians.
    pub count: usize,
    /// Fraction of total Gaussians that matched.
    pub fraction: f32,
}

impl QueryResult {
    fn new(indices: Vec<usize>, total: usize) -> Self {
        let count = indices.len();
        let fraction = if total == 0 {
            0.0
        } else {
            count as f32 / total as f32
        };
        Self {
            indices,
            count,
            fraction,
        }
    }
}

// ---------------------------------------------------------------------------
// InspectableModel
// ---------------------------------------------------------------------------

/// A trained 3DGS model in flat-array form, ready for inspection.
///
/// # Invariant
///
/// Every query/inspection function in this module (including
/// [`InspectableModel::activated_scale`]) assumes `positions.len() == n * 3`,
/// `scales.len() == n * 3`, `colors.len() == n * 3`, and `opacities.len() ==
/// n`. [`InspectableModel::new`] validates this; because the fields below
/// are `pub`, a struct-literal construction can still bypass that check, so
/// prefer `new` unless the caller can already guarantee the invariant holds
/// (as this module's own tests do, e.g. for a trivially-consistent empty
/// model). `#[non_exhaustive]` does not stop struct-literal construction
/// within this crate, only from an external one -- it is a defensive marker,
/// not itself a guarantee.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InspectableModel {
    /// Flat positions \[N*3\]: (x, y, z) per Gaussian.
    pub positions: Vec<f32>,
    /// Activated opacities \[N\] in \[0, 1\].
    pub opacities: Vec<f32>,
    /// Log-scale values \[N*3\]: (log_sx, log_sy, log_sz) per Gaussian.
    pub scales: Vec<f32>,
    /// DC SH color \[N*3\]: (r, g, b) per Gaussian in \[0, 1\].
    pub colors: Vec<f32>,
    /// Number of Gaussians.
    pub n: usize,
}

impl InspectableModel {
    /// Construct an `InspectableModel`, validating array dimensions.
    pub fn new(
        positions: Vec<f32>,
        opacities: Vec<f32>,
        scales: Vec<f32>,
        colors: Vec<f32>,
    ) -> Result<Self, InspectorError> {
        let n = opacities.len();

        if positions.len() != n * 3 {
            return Err(InspectorError::DimensionError(format!(
                "positions length {} != n*3 = {}",
                positions.len(),
                n * 3
            )));
        }
        if scales.len() != n * 3 {
            return Err(InspectorError::DimensionError(format!(
                "scales length {} != n*3 = {}",
                scales.len(),
                n * 3
            )));
        }
        if colors.len() != n * 3 {
            return Err(InspectorError::DimensionError(format!(
                "colors length {} != n*3 = {}",
                colors.len(),
                n * 3
            )));
        }

        Ok(Self {
            positions,
            opacities,
            scales,
            colors,
            n,
        })
    }

    /// Get the activated (exp) scale for Gaussian `i` along `axis` (0=x, 1=y, 2=z).
    ///
    /// # Errors
    ///
    /// Returns [`InspectorError::InvalidQuery`] if `axis >= 3`, or
    /// [`InspectorError::IndexOutOfBounds`] if `i` is out of range for the
    /// backing `scales` array. Bounds-checked against `self.scales.len()`
    /// directly (not `self.n`) so this cannot panic even if the invariant
    /// documented on [`InspectableModel`] has been violated via a
    /// struct-literal construction that bypassed [`InspectableModel::new`].
    pub fn activated_scale(&self, i: usize, axis: usize) -> Result<f32, InspectorError> {
        if axis >= 3 {
            return Err(InspectorError::InvalidQuery(format!(
                "axis {axis} out of range: must be < 3 (0=x, 1=y, 2=z)"
            )));
        }
        self.scales
            .get(i * 3 + axis)
            .map(|s| s.exp())
            .ok_or(InspectorError::IndexOutOfBounds {
                index: i,
                count: self.n,
            })
    }

    /// Get all inspectable properties for a single Gaussian by index.
    pub fn get(&self, index: usize) -> Result<GaussianProperties, InspectorError> {
        if index >= self.n {
            return Err(InspectorError::IndexOutOfBounds {
                index,
                count: self.n,
            });
        }

        let position = [
            self.positions[index * 3],
            self.positions[index * 3 + 1],
            self.positions[index * 3 + 2],
        ];
        let opacity = self.opacities[index];
        let sx = self.activated_scale(index, 0)?;
        let sy = self.activated_scale(index, 1)?;
        let sz = self.activated_scale(index, 2)?;

        let max_scale = sx.max(sy).max(sz);
        let min_scale = sx.min(sy).min(sz);
        let anisotropy = max_scale / (min_scale + 1e-8);
        let volume = sx * sy * sz;

        let color = [
            self.colors[index * 3],
            self.colors[index * 3 + 1],
            self.colors[index * 3 + 2],
        ];

        Ok(GaussianProperties {
            index,
            position,
            opacity,
            max_scale,
            min_scale,
            anisotropy,
            color,
            volume,
        })
    }

    /// Compute the axis-aligned bounding box of all Gaussian positions.
    pub fn bounding_box(&self) -> Result<BoundingBox3d, InspectorError> {
        if self.n == 0 {
            return Err(InspectorError::EmptyModel);
        }

        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];

        for i in 0..self.n {
            for axis in 0..3 {
                let v = self.positions[i * 3 + axis];
                if v < min[axis] {
                    min[axis] = v;
                }
                if v > max[axis] {
                    max[axis] = v;
                }
            }
        }

        Ok(BoundingBox3d::new(min, max))
    }
}

// ---------------------------------------------------------------------------
// Spatial queries
// ---------------------------------------------------------------------------

/// Find all Gaussians within `radius` of `center` (L2 distance on positions).
pub fn query_sphere(
    model: &InspectableModel,
    center: [f32; 3],
    radius: f32,
) -> Result<QueryResult, InspectorError> {
    if radius < 0.0 {
        return Err(InspectorError::InvalidRadius(radius));
    }
    let r2 = radius * radius;
    let indices: Vec<usize> = (0..model.n)
        .filter(|&i| {
            let dx = model.positions[i * 3] - center[0];
            let dy = model.positions[i * 3 + 1] - center[1];
            let dz = model.positions[i * 3 + 2] - center[2];
            dx * dx + dy * dy + dz * dz <= r2
        })
        .collect();
    Ok(QueryResult::new(indices, model.n))
}

/// Find all Gaussians whose position is inside an axis-aligned bounding box.
pub fn query_aabb(
    model: &InspectableModel,
    bbox: &BoundingBox3d,
) -> Result<QueryResult, InspectorError> {
    let indices: Vec<usize> = (0..model.n)
        .filter(|&i| {
            let pos = [
                model.positions[i * 3],
                model.positions[i * 3 + 1],
                model.positions[i * 3 + 2],
            ];
            bbox.contains(&pos)
        })
        .collect();
    Ok(QueryResult::new(indices, model.n))
}

/// Find the k nearest Gaussians to `query` by L2 distance on positions.
///
/// Returns a vec of `(index, distance)` pairs sorted by ascending distance.
/// If `k == 0` returns an empty vec. If `k >= model.n` all Gaussians are returned.
pub fn query_knn(
    model: &InspectableModel,
    query: [f32; 3],
    k: usize,
) -> Result<Vec<(usize, f32)>, InspectorError> {
    if model.n == 0 {
        return Err(InspectorError::EmptyModel);
    }
    if k == 0 {
        return Ok(Vec::new());
    }

    let mut distances: Vec<(usize, f32)> = (0..model.n)
        .map(|i| {
            let dx = model.positions[i * 3] - query[0];
            let dy = model.positions[i * 3 + 1] - query[1];
            let dz = model.positions[i * 3 + 2] - query[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            (i, dist)
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    distances.truncate(k);
    Ok(distances)
}

// ---------------------------------------------------------------------------
// Property filters
// ---------------------------------------------------------------------------

/// Find Gaussians with opacity strictly below `threshold` (likely transparent).
pub fn find_low_opacity(model: &InspectableModel, threshold: f32) -> QueryResult {
    let indices: Vec<usize> = (0..model.n)
        .filter(|&i| model.opacities[i] < threshold)
        .collect();
    QueryResult::new(indices, model.n)
}

/// Find Gaussians with anisotropy (max_scale / (min_scale + 1e-8)) above `threshold`.
///
/// A Gaussian whose scale cannot be read (see [`InspectableModel::activated_scale`];
/// only reachable for `i < model.n` if the model's invariant was violated via
/// a struct-literal bypass of [`InspectableModel::new`]) is excluded rather
/// than panicking or fabricating a value for it.
pub fn find_high_anisotropy(model: &InspectableModel, threshold: f32) -> QueryResult {
    let indices: Vec<usize> = (0..model.n)
        .filter(|&i| {
            let (Ok(sx), Ok(sy), Ok(sz)) = (
                model.activated_scale(i, 0),
                model.activated_scale(i, 1),
                model.activated_scale(i, 2),
            ) else {
                return false;
            };
            let max_s = sx.max(sy).max(sz);
            let min_s = sx.min(sy).min(sz);
            let anisotropy = max_s / (min_s + 1e-8);
            anisotropy > threshold
        })
        .collect();
    QueryResult::new(indices, model.n)
}

/// Find Gaussians with max scale (world units) above `threshold`.
///
/// See [`find_high_anisotropy`] for how an unreadable scale is handled.
pub fn find_large_gaussians(model: &InspectableModel, threshold: f32) -> QueryResult {
    let indices: Vec<usize> = (0..model.n)
        .filter(|&i| {
            let (Ok(sx), Ok(sy), Ok(sz)) = (
                model.activated_scale(i, 0),
                model.activated_scale(i, 1),
                model.activated_scale(i, 2),
            ) else {
                return false;
            };
            sx.max(sy).max(sz) > threshold
        })
        .collect();
    QueryResult::new(indices, model.n)
}

/// Find spatial outliers: Gaussians whose nearest neighbor distance exceeds `distance_threshold`.
///
/// For each Gaussian, computes the distance to its closest neighbour; if that
/// distance exceeds the threshold the Gaussian is flagged as a spatial outlier.
///
/// Uses O(n²) brute-force nearest-neighbor search for `n < 1000` (fast enough
/// that a spatial index would only add overhead), and a uniform spatial hash
/// keyed on `distance_threshold` for `n >= 1000`, mirroring the
/// `scene_optimizer::so_deduplicate_near` helper's strategy for the same
/// class of problem (not currently part of this crate's module tree; not an
/// intra-doc link for that reason). A brute-force O(n²) scan on a realistic
/// 3DGS avatar (500k-1M Gaussians) is 2.5e11-1e12 distance evaluations --
/// effectively a hang; the spatial hash reduces this to expected O(n) for
/// roughly uniformly distributed points, additionally parallelised with
/// `rayon` across Gaussians.
pub fn find_spatial_outliers(
    model: &InspectableModel,
    distance_threshold: f32,
) -> Result<QueryResult, InspectorError> {
    if model.n == 0 {
        return Err(InspectorError::EmptyModel);
    }
    if model.n == 1 {
        // A single point has no neighbour; always considered an outlier.
        return Ok(QueryResult::new(vec![0], model.n));
    }

    let n = model.n;
    let positions = &model.positions;

    // Cell size == distance_threshold: any two points within distance_threshold
    // of each other necessarily fall in the same or a face/edge/corner-adjacent
    // cell (standard uniform-grid neighbour-search argument: if |x - y| <= R
    // then |floor(x/R) - floor(y/R)| <= 1 per axis), so scanning the 3×3×3
    // neighbourhood around a point's cell is sufficient to determine whether
    // its nearest neighbour is within the threshold -- without visiting every
    // other point. Only valid for a well-defined positive cell size, so small
    // `n` (where brute force is already fast) and non-positive thresholds
    // (which `1.0 / distance_threshold` cannot handle) both use the brute-force
    // path instead, matching the original behaviour exactly for those cases.
    let is_outlier: Vec<bool> = if n < 1000 || distance_threshold <= 0.0 {
        (0..n)
            .map(|i| {
                let mut min_dist2 = f32::INFINITY;
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let dx = positions[i * 3] - positions[j * 3];
                    let dy = positions[i * 3 + 1] - positions[j * 3 + 1];
                    let dz = positions[i * 3 + 2] - positions[j * 3 + 2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 < min_dist2 {
                        min_dist2 = d2;
                    }
                }
                min_dist2.sqrt() > distance_threshold
            })
            .collect()
    } else {
        let inv_r = 1.0 / distance_threshold;
        let cell_of = |i: usize| -> (i64, i64, i64) {
            (
                (positions[i * 3] * inv_r).floor() as i64,
                (positions[i * 3 + 1] * inv_r).floor() as i64,
                (positions[i * 3 + 2] * inv_r).floor() as i64,
            )
        };

        let mut cell_map: std::collections::HashMap<(i64, i64, i64), Vec<usize>> =
            std::collections::HashMap::with_capacity(n);
        for i in 0..n {
            cell_map.entry(cell_of(i)).or_default().push(i);
        }

        (0..n)
            .into_par_iter()
            .map(|i| {
                let xi = positions[i * 3];
                let yi = positions[i * 3 + 1];
                let zi = positions[i * 3 + 2];
                let (cx, cy, cz) = cell_of(i);
                let mut min_dist2 = f32::INFINITY;

                for nx in -1i64..=1 {
                    for ny in -1i64..=1 {
                        for nz in -1i64..=1 {
                            let nc = (cx + nx, cy + ny, cz + nz);
                            let Some(neighbours) = cell_map.get(&nc) else {
                                continue;
                            };
                            for &j in neighbours {
                                if j == i {
                                    continue;
                                }
                                let dx = xi - positions[j * 3];
                                let dy = yi - positions[j * 3 + 1];
                                let dz = zi - positions[j * 3 + 2];
                                let d2 = dx * dx + dy * dy + dz * dz;
                                if d2 < min_dist2 {
                                    min_dist2 = d2;
                                }
                            }
                        }
                    }
                }

                min_dist2.sqrt() > distance_threshold
            })
            .collect()
    };

    let indices: Vec<usize> = (0..n).filter(|&i| is_outlier[i]).collect();
    Ok(QueryResult::new(indices, n))
}

// ---------------------------------------------------------------------------
// Distribution analysis
// ---------------------------------------------------------------------------

/// Compute a histogram of scalar values.
///
/// Returns `(bin_edges, counts)` where `bin_edges` has `bins + 1` entries and
/// `counts` has `bins` entries. If all values are equal, the range is expanded
/// by a small epsilon so that the histogram remains well-defined.
pub fn histogram(values: &[f32], bins: usize) -> Result<(Vec<f32>, Vec<usize>), InspectorError> {
    if bins == 0 {
        return Err(InspectorError::InvalidQuery(
            "bins must be at least 1".to_string(),
        ));
    }
    if values.is_empty() {
        return Err(InspectorError::InvalidQuery(
            "values slice is empty".to_string(),
        ));
    }

    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for &v in values {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }

    // Expand range if all values are identical to avoid division by zero.
    if (max_val - min_val).abs() < f32::EPSILON {
        max_val = min_val + 1.0;
    }

    let range = max_val - min_val;
    let mut counts = vec![0usize; bins];

    for &v in values {
        let bin_idx = ((v - min_val) / range * bins as f32) as usize;
        // Clamp to bins-1 to handle the max-value edge case.
        let bin_idx = bin_idx.min(bins - 1);
        counts[bin_idx] += 1;
    }

    let bin_width = range / bins as f32;
    let edges: Vec<f32> = (0..=bins).map(|i| min_val + i as f32 * bin_width).collect();

    Ok((edges, counts))
}

/// Compute the p-th percentile (0..=100) of a set of values using linear interpolation.
///
/// Sorts `values` in place. Returns an error if `p` is outside [0, 100] or if
/// the slice is empty.
pub fn percentile(values: &mut [f32], p: f32) -> Result<f32, InspectorError> {
    if values.is_empty() {
        return Err(InspectorError::InvalidQuery(
            "values slice is empty".to_string(),
        ));
    }
    if !(0.0..=100.0).contains(&p) {
        return Err(InspectorError::InvalidQuery(format!(
            "percentile p={p} is outside [0, 100]"
        )));
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();

    if n == 1 {
        return Ok(values[0]);
    }

    // Linear interpolation between adjacent sorted values.
    let rank = p / 100.0 * (n - 1) as f32;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f32;

    if upper >= n {
        return Ok(values[n - 1]);
    }

    Ok(values[lower] * (1.0 - frac) + values[upper] * frac)
}

// ---------------------------------------------------------------------------
// Color distribution
// ---------------------------------------------------------------------------

/// Per-channel color statistics for the model.
#[derive(Debug, Clone)]
pub struct ColorDistribution {
    /// Mean RGB.
    pub mean: [f32; 3],
    /// Standard deviation of RGB.
    pub std: [f32; 3],
    /// Index (0=R, 1=G, 2=B) of the channel with the highest mean.
    pub dominant_channel: usize,
    /// Fraction of Gaussians that are roughly grayscale:
    /// max(|r-g|, |g-b|, |r-b|) < 0.1.
    pub grayscale_fraction: f32,
}

/// Compute per-channel color statistics across all Gaussians.
pub fn analyze_color_distribution(
    model: &InspectableModel,
) -> Result<ColorDistribution, InspectorError> {
    if model.n == 0 {
        return Err(InspectorError::EmptyModel);
    }

    let mut sum = [0.0f32; 3];
    let mut sum_sq = [0.0f32; 3];
    let mut grayscale_count = 0usize;

    for i in 0..model.n {
        let r = model.colors[i * 3];
        let g = model.colors[i * 3 + 1];
        let b = model.colors[i * 3 + 2];

        sum[0] += r;
        sum[1] += g;
        sum[2] += b;
        sum_sq[0] += r * r;
        sum_sq[1] += g * g;
        sum_sq[2] += b * b;

        let rg = (r - g).abs();
        let gb = (g - b).abs();
        let rb = (r - b).abs();
        if rg.max(gb).max(rb) < 0.1 {
            grayscale_count += 1;
        }
    }

    let inv_n = 1.0 / model.n as f32;
    let mean = [sum[0] * inv_n, sum[1] * inv_n, sum[2] * inv_n];

    let std = [
        ((sum_sq[0] * inv_n - mean[0] * mean[0]).max(0.0)).sqrt(),
        ((sum_sq[1] * inv_n - mean[1] * mean[1]).max(0.0)).sqrt(),
        ((sum_sq[2] * inv_n - mean[2] * mean[2]).max(0.0)).sqrt(),
    ];

    let dominant_channel = if mean[0] >= mean[1] && mean[0] >= mean[2] {
        0
    } else if mean[1] >= mean[2] {
        1
    } else {
        2
    };

    let grayscale_fraction = grayscale_count as f32 / model.n as f32;

    Ok(ColorDistribution {
        mean,
        std,
        dominant_channel,
        grayscale_fraction,
    })
}

// ---------------------------------------------------------------------------
// Inspection report
// ---------------------------------------------------------------------------

/// Full inspection report for a model.
#[derive(Debug, Clone)]
pub struct InspectionReport {
    /// Number of Gaussians.
    pub n_gaussians: usize,
    /// Axis-aligned bounding box.
    pub bounding_box: BoundingBox3d,
    /// Mean opacity.
    pub mean_opacity: f32,
    /// Standard deviation of opacity.
    pub std_opacity: f32,
    /// Fraction of Gaussians with opacity < 0.1.
    pub transparent_fraction: f32,
    /// Mean of per-Gaussian max scales.
    pub mean_max_scale: f32,
    /// Std of per-Gaussian max scales.
    pub std_max_scale: f32,
    /// Mean anisotropy across all Gaussians.
    pub mean_anisotropy: f32,
    /// Fraction with anisotropy > 10.
    pub high_anisotropy_fraction: f32,
    /// Color distribution statistics.
    pub color_distribution: ColorDistribution,
    /// Fraction of spatial outliers (nearest-neighbour > diagonal/20).
    pub spatial_outlier_fraction: f32,
}

/// Run a full inspection of the model, computing all report fields.
///
/// The spatial outlier threshold is set to `bounding_box.diagonal() / 20.0`,
/// which provides a scene-relative measure of isolation for individual Gaussians.
pub fn inspect_model(model: &InspectableModel) -> Result<InspectionReport, InspectorError> {
    if model.n == 0 {
        return Err(InspectorError::EmptyModel);
    }

    let bounding_box = model.bounding_box()?;

    // Opacity statistics
    let mut opacity_sum = 0.0f32;
    let mut opacity_sum_sq = 0.0f32;
    let mut transparent_count = 0usize;

    // Scale and anisotropy statistics
    let mut max_scale_sum = 0.0f32;
    let mut max_scale_sum_sq = 0.0f32;
    let mut anisotropy_sum = 0.0f32;
    let mut high_anisotropy_count = 0usize;

    for i in 0..model.n {
        let op = model.opacities[i];
        opacity_sum += op;
        opacity_sum_sq += op * op;
        if op < 0.1 {
            transparent_count += 1;
        }

        let sx = model.activated_scale(i, 0)?;
        let sy = model.activated_scale(i, 1)?;
        let sz = model.activated_scale(i, 2)?;
        let max_s = sx.max(sy).max(sz);
        let min_s = sx.min(sy).min(sz);
        let anisotropy = max_s / (min_s + 1e-8);

        max_scale_sum += max_s;
        max_scale_sum_sq += max_s * max_s;
        anisotropy_sum += anisotropy;
        if anisotropy > 10.0 {
            high_anisotropy_count += 1;
        }
    }

    let inv_n = 1.0 / model.n as f32;
    let mean_opacity = opacity_sum * inv_n;
    let std_opacity = ((opacity_sum_sq * inv_n - mean_opacity * mean_opacity).max(0.0)).sqrt();
    let transparent_fraction = transparent_count as f32 * inv_n;

    let mean_max_scale = max_scale_sum * inv_n;
    let std_max_scale =
        ((max_scale_sum_sq * inv_n - mean_max_scale * mean_max_scale).max(0.0)).sqrt();
    let mean_anisotropy = anisotropy_sum * inv_n;
    let high_anisotropy_fraction = high_anisotropy_count as f32 * inv_n;

    let color_distribution = analyze_color_distribution(model)?;

    // Scene-relative outlier threshold: diagonal / 20.
    let outlier_threshold = (bounding_box.diagonal() / 20.0).max(1e-6);
    let outlier_result = find_spatial_outliers(model, outlier_threshold)?;
    let spatial_outlier_fraction = outlier_result.fraction;

    Ok(InspectionReport {
        n_gaussians: model.n,
        bounding_box,
        mean_opacity,
        std_opacity,
        transparent_fraction,
        mean_max_scale,
        std_max_scale,
        mean_anisotropy,
        high_anisotropy_fraction,
        color_distribution,
        spatial_outlier_fraction,
    })
}

/// Format an `InspectionReport` as a human-readable text summary.
pub fn format_inspection_report(report: &InspectionReport) -> String {
    let bb = &report.bounding_box;
    let size = bb.size();
    let center = bb.center();
    let cd = &report.color_distribution;
    let dominant = ["R", "G", "B"][cd.dominant_channel];

    format!(
        "=== Model Inspection Report ===\n\
         Gaussians      : {n}\n\
         \n\
         Bounding Box\n\
           Min          : ({min_x:.4}, {min_y:.4}, {min_z:.4})\n\
           Max          : ({max_x:.4}, {max_y:.4}, {max_z:.4})\n\
           Size         : ({sx:.4}, {sy:.4}, {sz:.4})\n\
           Center       : ({cx:.4}, {cy:.4}, {cz:.4})\n\
           Diagonal     : {diag:.4}\n\
         \n\
         Opacity\n\
           Mean         : {mean_op:.4}\n\
           Std          : {std_op:.4}\n\
           Transparent  : {trans:.2}% (opacity < 0.1)\n\
         \n\
         Scale\n\
           Mean max     : {mean_s:.4}\n\
           Std max      : {std_s:.4}\n\
         \n\
         Anisotropy\n\
           Mean         : {mean_anis:.4}\n\
           High (>10)   : {hi_anis:.2}%\n\
         \n\
         Color (RGB)\n\
           Mean         : ({mr:.4}, {mg:.4}, {mb:.4})\n\
           Std          : ({sr:.4}, {sg:.4}, {sb:.4})\n\
           Dominant ch  : {dominant}\n\
           Grayscale    : {gray:.2}%\n\
         \n\
         Spatial Outliers: {outlier:.2}%\n",
        n = report.n_gaussians,
        min_x = bb.min[0],
        min_y = bb.min[1],
        min_z = bb.min[2],
        max_x = bb.max[0],
        max_y = bb.max[1],
        max_z = bb.max[2],
        sx = size[0],
        sy = size[1],
        sz = size[2],
        cx = center[0],
        cy = center[1],
        cz = center[2],
        diag = bb.diagonal(),
        mean_op = report.mean_opacity,
        std_op = report.std_opacity,
        trans = report.transparent_fraction * 100.0,
        mean_s = report.mean_max_scale,
        std_s = report.std_max_scale,
        mean_anis = report.mean_anisotropy,
        hi_anis = report.high_anisotropy_fraction * 100.0,
        mr = cd.mean[0],
        mg = cd.mean[1],
        mb = cd.mean[2],
        sr = cd.std[0],
        sg = cd.std[1],
        sb = cd.std[2],
        dominant = dominant,
        gray = cd.grayscale_fraction * 100.0,
        outlier = report.spatial_outlier_fraction * 100.0,
    )
}

// ---------------------------------------------------------------------------
// CSV dump
// ---------------------------------------------------------------------------

/// Dump selected Gaussians as CSV.
///
/// The header line is: `index,x,y,z,opacity,max_scale,r,g,b`
pub fn dump_gaussians_csv(
    model: &InspectableModel,
    indices: &[usize],
) -> Result<String, InspectorError> {
    let mut out = String::from("index,x,y,z,opacity,max_scale,r,g,b\n");

    for &idx in indices {
        let props = model.get(idx)?;
        out.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            props.index,
            props.position[0],
            props.position[1],
            props.position[2],
            props.opacity,
            props.max_scale,
            props.color[0],
            props.color[1],
            props.color[2],
        ));
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Density voxel grid
// ---------------------------------------------------------------------------

/// Compute a 3D density voxel grid counting how many Gaussian centres fall in
/// each voxel.
///
/// The grid is `grid` voxels per axis. Returns `(counts, voxel_size)` where
/// `counts` is a flat `Vec<u32>` in ZYX order (index = `z * grid² + y * grid + x`)
/// and `voxel_size` is the edge length of one voxel (uniform across axes, using
/// the largest bounding-box dimension to keep voxels cubic).
///
/// Returns an error on an empty model or when `grid == 0`.
pub fn density_voxel_grid(
    model: &InspectableModel,
    grid: usize,
) -> Result<(Vec<u32>, f32), InspectorError> {
    if model.n == 0 {
        return Err(InspectorError::EmptyModel);
    }
    if grid == 0 {
        return Err(InspectorError::InvalidQuery(
            "grid must be at least 1".to_string(),
        ));
    }

    let bb = model.bounding_box()?;
    let size = bb.size();

    // Use the largest dimension to keep voxels roughly cubic.
    let max_dim = size[0].max(size[1]).max(size[2]);
    // Guard against degenerate (zero-size) scenes.
    let voxel_size = if max_dim < f32::EPSILON {
        1.0
    } else {
        max_dim / grid as f32
    };
    // Use a tiny epsilon to avoid division-by-zero in the degenerate case.
    let inv_voxel = 1.0 / voxel_size.max(f32::EPSILON);

    let total = grid * grid * grid;
    let mut counts = vec![0u32; total];

    for i in 0..model.n {
        let px = model.positions[i * 3] - bb.min[0];
        let py = model.positions[i * 3 + 1] - bb.min[1];
        let pz = model.positions[i * 3 + 2] - bb.min[2];

        let xi = ((px * inv_voxel) as usize).min(grid - 1);
        let yi = ((py * inv_voxel) as usize).min(grid - 1);
        let zi = ((pz * inv_voxel) as usize).min(grid - 1);

        let flat_idx = zi * grid * grid + yi * grid + xi;
        counts[flat_idx] += 1;
    }

    Ok((counts, voxel_size))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Create a small model with `n` Gaussians placed at (i, 0, 0) for i in 0..n.
    /// All opacities = 0.8, log_scales = (-1, -1, -1), colors = (1, 0, 0).
    fn make_line_model(n: usize) -> InspectableModel {
        let mut positions = Vec::with_capacity(n * 3);
        let mut opacities = Vec::with_capacity(n);
        let mut scales = Vec::with_capacity(n * 3);
        let mut colors = Vec::with_capacity(n * 3);

        for i in 0..n {
            positions.push(i as f32);
            positions.push(0.0);
            positions.push(0.0);
            opacities.push(0.8);
            scales.push(-1.0_f32); // exp(-1) ≈ 0.368
            scales.push(-1.0_f32);
            scales.push(-1.0_f32);
            colors.push(1.0);
            colors.push(0.0);
            colors.push(0.0);
        }

        InspectableModel::new(positions, opacities, scales, colors).expect("valid test model")
    }

    // -----------------------------------------------------------------------
    // BoundingBox3d
    // -----------------------------------------------------------------------

    #[test]
    fn test_bbox_center() {
        let bb = BoundingBox3d::new([0.0, 0.0, 0.0], [4.0, 2.0, 6.0]);
        let c = bb.center();
        assert!((c[0] - 2.0).abs() < 1e-6);
        assert!((c[1] - 1.0).abs() < 1e-6);
        assert!((c[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_size() {
        let bb = BoundingBox3d::new([1.0, 2.0, 3.0], [4.0, 7.0, 9.0]);
        let s = bb.size();
        assert!((s[0] - 3.0).abs() < 1e-6);
        assert!((s[1] - 5.0).abs() < 1e-6);
        assert!((s[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_bbox_diagonal() {
        let bb = BoundingBox3d::new([0.0; 3], [1.0, 1.0, 1.0]);
        let d = bb.diagonal();
        assert!((d - 3.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_bbox_contains_inside() {
        let bb = BoundingBox3d::new([0.0; 3], [2.0, 2.0, 2.0]);
        assert!(bb.contains(&[1.0, 1.0, 1.0]));
    }

    #[test]
    fn test_bbox_contains_boundary() {
        let bb = BoundingBox3d::new([0.0; 3], [2.0, 2.0, 2.0]);
        assert!(bb.contains(&[0.0, 0.0, 0.0]));
        assert!(bb.contains(&[2.0, 2.0, 2.0]));
    }

    #[test]
    fn test_bbox_contains_outside() {
        let bb = BoundingBox3d::new([0.0; 3], [2.0, 2.0, 2.0]);
        assert!(!bb.contains(&[3.0, 1.0, 1.0]));
        assert!(!bb.contains(&[-0.1, 1.0, 1.0]));
    }

    #[test]
    fn test_bbox_expand() {
        let bb1 = BoundingBox3d::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let bb2 = BoundingBox3d::new([-1.0, 2.0, 0.5], [3.0, 3.0, 2.0]);
        let expanded = bb1.expand(&bb2);
        assert!((expanded.min[0] - (-1.0)).abs() < 1e-6);
        assert!((expanded.min[1] - 0.0).abs() < 1e-6);
        assert!((expanded.max[0] - 3.0).abs() < 1e-6);
        assert!((expanded.max[1] - 3.0).abs() < 1e-6);
        assert!((expanded.max[2] - 2.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // InspectableModel construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_new_valid() {
        let m = InspectableModel::new(
            vec![0.0, 0.0, 0.0],
            vec![0.5],
            vec![-1.0, -1.0, -1.0],
            vec![0.8, 0.1, 0.1],
        );
        assert!(m.is_ok());
        assert_eq!(m.unwrap().n, 1);
    }

    #[test]
    fn test_model_new_positions_mismatch() {
        let m = InspectableModel::new(
            vec![0.0, 0.0], // wrong: should be 3
            vec![0.5],
            vec![-1.0, -1.0, -1.0],
            vec![0.8, 0.1, 0.1],
        );
        assert!(matches!(m, Err(InspectorError::DimensionError(_))));
    }

    #[test]
    fn test_model_new_scales_mismatch() {
        let m = InspectableModel::new(
            vec![0.0, 0.0, 0.0],
            vec![0.5],
            vec![-1.0, -1.0], // wrong: should be 3
            vec![0.8, 0.1, 0.1],
        );
        assert!(matches!(m, Err(InspectorError::DimensionError(_))));
    }

    #[test]
    fn test_model_new_colors_mismatch() {
        let m = InspectableModel::new(
            vec![0.0, 0.0, 0.0],
            vec![0.5],
            vec![-1.0, -1.0, -1.0],
            vec![0.8, 0.1], // wrong: should be 3
        );
        assert!(matches!(m, Err(InspectorError::DimensionError(_))));
    }

    // -----------------------------------------------------------------------
    // InspectableModel::get
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_get_valid() {
        let m = make_line_model(3);
        let props = m.get(1).expect("should get index 1");
        assert_eq!(props.index, 1);
        assert!((props.position[0] - 1.0).abs() < 1e-6);
        assert!((props.opacity - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_model_get_out_of_bounds() {
        let m = make_line_model(3);
        let result = m.get(5);
        assert!(matches!(
            result,
            Err(InspectorError::IndexOutOfBounds { index: 5, count: 3 })
        ));
    }

    // -----------------------------------------------------------------------
    // InspectableModel::activated_scale
    // -----------------------------------------------------------------------

    #[test]
    fn test_activated_scale_valid() {
        let m = make_line_model(3);
        // scales are all -1.0 in make_line_model, so exp(-1) for every axis.
        let sx = m.activated_scale(1, 0).expect("valid index/axis");
        assert!((sx - (-1.0f32).exp()).abs() < 1e-6);
    }

    #[test]
    fn test_activated_scale_index_out_of_bounds_errors_instead_of_panicking() {
        // Regression: this used to index `self.scales[i * 3 + axis]`
        // directly with no bounds check, panicking on an out-of-range `i`.
        let m = make_line_model(3);
        let result = m.activated_scale(100, 0);
        assert!(matches!(
            result,
            Err(InspectorError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_activated_scale_axis_out_of_bounds_errors_instead_of_panicking() {
        let m = make_line_model(3);
        let result = m.activated_scale(0, 3);
        assert!(matches!(result, Err(InspectorError::InvalidQuery(_))));
    }

    #[test]
    fn test_activated_scale_bounds_checked_against_real_scales_len_not_n() {
        // Regression: a model whose public fields were set directly (bypassing
        // `InspectableModel::new`'s validation) so that `n` overstates the
        // real `scales` array must still not panic -- the check is against
        // `scales.len()`, not the (here, lying) `n` field.
        let m = InspectableModel {
            positions: vec![0.0, 0.0, 0.0],
            opacities: vec![0.5],
            scales: vec![-1.0, -1.0, -1.0],
            colors: vec![1.0, 0.0, 0.0],
            n: 10, // lies: only 1 Gaussian's worth of data actually present
        };
        assert!(m.activated_scale(0, 0).is_ok());
        let result = m.activated_scale(5, 0);
        assert!(matches!(
            result,
            Err(InspectorError::IndexOutOfBounds { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // InspectableModel::bounding_box
    // -----------------------------------------------------------------------

    #[test]
    fn test_bounding_box_single() {
        let m = make_line_model(1);
        let bb = m.bounding_box().expect("ok");
        assert!((bb.min[0] - 0.0).abs() < 1e-6);
        assert!((bb.max[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_bounding_box_multiple() {
        let m = make_line_model(5); // positions at x=0..4
        let bb = m.bounding_box().expect("ok");
        assert!((bb.min[0] - 0.0).abs() < 1e-6);
        assert!((bb.max[0] - 4.0).abs() < 1e-6);
        assert!((bb.min[1]).abs() < 1e-6);
        assert!((bb.max[1]).abs() < 1e-6);
    }

    #[test]
    fn test_bounding_box_empty_model() {
        let m = InspectableModel {
            positions: vec![],
            opacities: vec![],
            scales: vec![],
            colors: vec![],
            n: 0,
        };
        assert!(matches!(m.bounding_box(), Err(InspectorError::EmptyModel)));
    }

    // -----------------------------------------------------------------------
    // query_sphere
    // -----------------------------------------------------------------------

    #[test]
    fn test_query_sphere_all_inside() {
        let m = make_line_model(3); // positions: (0,0,0),(1,0,0),(2,0,0)
        let result = query_sphere(&m, [1.0, 0.0, 0.0], 2.0).expect("ok");
        assert_eq!(result.count, 3);
    }

    #[test]
    fn test_query_sphere_none_inside() {
        let m = make_line_model(3);
        let result = query_sphere(&m, [100.0, 0.0, 0.0], 0.5).expect("ok");
        assert_eq!(result.count, 0);
        assert!((result.fraction - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_query_sphere_boundary() {
        let m = make_line_model(3); // (0,0,0),(1,0,0),(2,0,0)
                                    // radius = 1.0, center = (1,0,0) → all three are within or on boundary
        let result = query_sphere(&m, [1.0, 0.0, 0.0], 1.0).expect("ok");
        assert_eq!(result.count, 3);
    }

    #[test]
    fn test_query_sphere_negative_radius() {
        let m = make_line_model(2);
        let result = query_sphere(&m, [0.0, 0.0, 0.0], -1.0);
        assert!(matches!(result, Err(InspectorError::InvalidRadius(_))));
    }

    // -----------------------------------------------------------------------
    // query_aabb
    // -----------------------------------------------------------------------

    #[test]
    fn test_query_aabb_contains_all() {
        let m = make_line_model(4);
        let bbox = BoundingBox3d::new([-1.0, -1.0, -1.0], [10.0, 1.0, 1.0]);
        let result = query_aabb(&m, &bbox).expect("ok");
        assert_eq!(result.count, 4);
    }

    #[test]
    fn test_query_aabb_contains_none() {
        let m = make_line_model(4);
        let bbox = BoundingBox3d::new([50.0, 0.0, 0.0], [60.0, 1.0, 1.0]);
        let result = query_aabb(&m, &bbox).expect("ok");
        assert_eq!(result.count, 0);
    }

    #[test]
    fn test_query_aabb_partial() {
        let m = make_line_model(5); // x = 0,1,2,3,4
        let bbox = BoundingBox3d::new([1.5, -1.0, -1.0], [3.5, 1.0, 1.0]);
        let result = query_aabb(&m, &bbox).expect("ok");
        assert_eq!(result.count, 2); // x=2, x=3
    }

    // -----------------------------------------------------------------------
    // query_knn
    // -----------------------------------------------------------------------

    #[test]
    fn test_query_knn_k1() {
        let m = make_line_model(4); // positions at x=0,1,2,3
        let result = query_knn(&m, [1.6, 0.0, 0.0], 1).expect("ok");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 2); // x=2 is closest to 1.6
    }

    #[test]
    fn test_query_knn_k_greater_than_n() {
        let m = make_line_model(3);
        let result = query_knn(&m, [0.0, 0.0, 0.0], 100).expect("ok");
        assert_eq!(result.len(), 3); // clamped to n
    }

    #[test]
    fn test_query_knn_empty_model() {
        let m = InspectableModel {
            positions: vec![],
            opacities: vec![],
            scales: vec![],
            colors: vec![],
            n: 0,
        };
        assert!(matches!(
            query_knn(&m, [0.0, 0.0, 0.0], 1),
            Err(InspectorError::EmptyModel)
        ));
    }

    #[test]
    fn test_query_knn_k_zero() {
        let m = make_line_model(3);
        let result = query_knn(&m, [0.0, 0.0, 0.0], 0).expect("ok");
        assert!(result.is_empty());
    }

    #[test]
    fn test_query_knn_sorted() {
        let m = make_line_model(4);
        let result = query_knn(&m, [1.5, 0.0, 0.0], 4).expect("ok");
        // Check ascending distance order
        for w in result.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    // -----------------------------------------------------------------------
    // find_low_opacity
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_low_opacity_threshold_zero() {
        let m = make_line_model(3); // all opacity = 0.8
        let result = find_low_opacity(&m, 0.0);
        assert_eq!(result.count, 0);
    }

    #[test]
    fn test_find_low_opacity_threshold_one() {
        let m = make_line_model(3); // all opacity = 0.8 < 1.0
        let result = find_low_opacity(&m, 1.0);
        assert_eq!(result.count, 3);
    }

    #[test]
    fn test_find_low_opacity_mixed() {
        let m = InspectableModel::new(
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            vec![0.05, 0.5, 0.8],
            vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .expect("valid");
        let result = find_low_opacity(&m, 0.1);
        assert_eq!(result.count, 1);
        assert_eq!(result.indices[0], 0);
    }

    // -----------------------------------------------------------------------
    // find_high_anisotropy
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_high_anisotropy() {
        // Two Gaussians: one isotropic, one highly elongated.
        // Elongated: log_scales = (0, -10, -10) → exp(0)/exp(-10) ≈ 22026 >> 10
        let positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let opacities = vec![0.5, 0.5];
        let scales = vec![-1.0, -1.0, -1.0, 0.0, -10.0, -10.0];
        let colors = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let m = InspectableModel::new(positions, opacities, scales, colors).expect("ok");

        let result = find_high_anisotropy(&m, 10.0);
        assert_eq!(result.count, 1);
        assert_eq!(result.indices[0], 1);
    }

    // -----------------------------------------------------------------------
    // find_large_gaussians
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_large_gaussians() {
        // Gaussian 0: log_scale = -1 → exp(-1) ≈ 0.368 (small)
        // Gaussian 1: log_scale = 3  → exp(3)  ≈ 20.1  (large)
        let positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let opacities = vec![0.5, 0.5];
        let scales = vec![-1.0, -1.0, -1.0, 3.0, 3.0, 3.0];
        let colors = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let m = InspectableModel::new(positions, opacities, scales, colors).expect("ok");

        let result = find_large_gaussians(&m, 5.0);
        assert_eq!(result.count, 1);
        assert_eq!(result.indices[0], 1);
    }

    // -----------------------------------------------------------------------
    // find_spatial_outliers
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_spatial_outliers_isolated() {
        // Dense cluster at (0,0,0),(1,0,0),(2,0,0) + isolated at (100,0,0)
        let mut positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        positions.extend_from_slice(&[100.0, 0.0, 0.0]);
        let n = 4;
        let opacities = vec![0.5; n];
        let scales = vec![-1.0; n * 3];
        let colors = vec![0.5; n * 3];
        let m = InspectableModel::new(positions, opacities, scales, colors).expect("ok");

        // Threshold 5: cluster neighbours are within 1, isolated neighbour is ~98 away.
        let result = find_spatial_outliers(&m, 5.0).expect("ok");
        assert_eq!(result.count, 1);
        assert_eq!(result.indices[0], 3);
    }

    #[test]
    fn test_find_spatial_outliers_dense_cluster() {
        // All within threshold → no outliers.
        let m = make_line_model(5); // x=0..4, nearest neighbour always 1 unit
        let result = find_spatial_outliers(&m, 5.0).expect("ok");
        assert_eq!(result.count, 0);
    }

    #[test]
    fn test_find_spatial_outliers_large_n_uses_spatial_hash_path() {
        // Regression: forces n >= 1000 so this exercises the spatial-hash +
        // rayon branch (previously an O(n^2) brute-force scan regardless of
        // n, which for a realistic 500k-1M Gaussian model was effectively a
        // hang: 2.5e11-1e12 distance evaluations). A line of 1200 points 1
        // unit apart (every point has a neighbour within 1) plus one
        // isolated point far outside the line's coordinate range must find
        // exactly the isolated point as an outlier.
        let line_n = 1200;
        let m = make_line_model(line_n);
        let mut positions = m.positions.clone();
        positions.extend_from_slice(&[5000.0, 0.0, 0.0]); // far past x=0..line_n-1
        let mut opacities = m.opacities.clone();
        opacities.push(0.8);
        let mut scales = m.scales.clone();
        scales.extend_from_slice(&[-1.0, -1.0, -1.0]);
        let mut colors = m.colors.clone();
        colors.extend_from_slice(&[1.0, 0.0, 0.0]);

        let full =
            InspectableModel::new(positions, opacities, scales, colors).expect("valid model");
        assert!(full.n >= 1000, "test must exercise the n >= 1000 code path");

        let result = find_spatial_outliers(&full, 5.0).expect("ok");
        assert_eq!(result.count, 1);
        assert_eq!(result.indices[0], line_n);
    }

    #[test]
    fn test_find_spatial_outliers_large_n_all_dense_no_outliers() {
        // Companion to the above: a purely dense large-n cloud (every point
        // within threshold of a neighbour) must report zero outliers via
        // the spatial-hash path too, not just the "one isolated point" case.
        let m = make_line_model(1500); // x=0..1499, nearest neighbour always 1 unit
        assert!(m.n >= 1000, "test must exercise the n >= 1000 code path");
        let result = find_spatial_outliers(&m, 5.0).expect("ok");
        assert_eq!(result.count, 0);
    }

    #[test]
    fn test_find_spatial_outliers_empty() {
        let m = InspectableModel {
            positions: vec![],
            opacities: vec![],
            scales: vec![],
            colors: vec![],
            n: 0,
        };
        assert!(matches!(
            find_spatial_outliers(&m, 1.0),
            Err(InspectorError::EmptyModel)
        ));
    }

    // -----------------------------------------------------------------------
    // histogram
    // -----------------------------------------------------------------------

    #[test]
    fn test_histogram_uniform() {
        let values: Vec<f32> = (0..10).map(|i| i as f32).collect(); // 0..9
        let (edges, counts) = histogram(&values, 5).expect("ok");
        assert_eq!(edges.len(), 6);
        assert_eq!(counts.len(), 5);
        // Total count should equal input length
        let total: usize = counts.iter().sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_histogram_single_value() {
        let values = vec![3.0_f32; 7];
        let (edges, counts) = histogram(&values, 4).expect("ok");
        assert_eq!(edges.len(), 5);
        let total: usize = counts.iter().sum();
        assert_eq!(total, 7);
    }

    #[test]
    fn test_histogram_zero_bins_error() {
        let values = vec![1.0_f32];
        assert!(matches!(
            histogram(&values, 0),
            Err(InspectorError::InvalidQuery(_))
        ));
    }

    #[test]
    fn test_histogram_empty_error() {
        assert!(matches!(
            histogram(&[], 5),
            Err(InspectorError::InvalidQuery(_))
        ));
    }

    // -----------------------------------------------------------------------
    // percentile
    // -----------------------------------------------------------------------

    #[test]
    fn test_percentile_min() {
        let mut v: Vec<f32> = vec![3.0, 1.0, 2.0, 5.0, 4.0];
        let p0 = percentile(&mut v, 0.0).expect("ok");
        assert!((p0 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_percentile_max() {
        let mut v: Vec<f32> = vec![3.0, 1.0, 2.0, 5.0, 4.0];
        let p100 = percentile(&mut v, 100.0).expect("ok");
        assert!((p100 - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_percentile_median() {
        let mut v: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p50 = percentile(&mut v, 50.0).expect("ok");
        assert!((p50 - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_percentile_empty_error() {
        let mut v: Vec<f32> = vec![];
        assert!(matches!(
            percentile(&mut v, 50.0),
            Err(InspectorError::InvalidQuery(_))
        ));
    }

    #[test]
    fn test_percentile_out_of_range_error() {
        let mut v = vec![1.0_f32, 2.0, 3.0];
        assert!(matches!(
            percentile(&mut v, 101.0),
            Err(InspectorError::InvalidQuery(_))
        ));
        assert!(matches!(
            percentile(&mut v, -1.0),
            Err(InspectorError::InvalidQuery(_))
        ));
    }

    // -----------------------------------------------------------------------
    // analyze_color_distribution
    // -----------------------------------------------------------------------

    #[test]
    fn test_color_distribution_red_dominant() {
        let n = 4;
        let m = InspectableModel::new(
            vec![0.0; n * 3],
            vec![0.5; n],
            vec![-1.0; n * 3],
            // All fully red
            (0..n).flat_map(|_| [1.0_f32, 0.0, 0.0]).collect(),
        )
        .expect("ok");

        let cd = analyze_color_distribution(&m).expect("ok");
        assert_eq!(cd.dominant_channel, 0);
        assert!((cd.mean[0] - 1.0).abs() < 1e-5);
        assert!((cd.mean[1]).abs() < 1e-5);
        // Not grayscale
        assert!(cd.grayscale_fraction < 0.01);
    }

    #[test]
    fn test_color_distribution_grayscale() {
        let n = 3;
        let m = InspectableModel::new(
            vec![0.0; n * 3],
            vec![0.5; n],
            vec![-1.0; n * 3],
            // All (0.5, 0.5, 0.5) → grayscale
            (0..n).flat_map(|_| [0.5_f32, 0.5, 0.5]).collect(),
        )
        .expect("ok");

        let cd = analyze_color_distribution(&m).expect("ok");
        assert!((cd.grayscale_fraction - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_color_distribution_empty() {
        let m = InspectableModel {
            positions: vec![],
            opacities: vec![],
            scales: vec![],
            colors: vec![],
            n: 0,
        };
        assert!(matches!(
            analyze_color_distribution(&m),
            Err(InspectorError::EmptyModel)
        ));
    }

    // -----------------------------------------------------------------------
    // inspect_model
    // -----------------------------------------------------------------------

    #[test]
    fn test_inspect_model_valid() {
        let m = make_line_model(5);
        let report = inspect_model(&m).expect("ok");
        assert_eq!(report.n_gaussians, 5);
        assert!((report.mean_opacity - 0.8).abs() < 1e-4);
    }

    #[test]
    fn test_inspect_model_n_gaussians() {
        let m = make_line_model(10);
        let report = inspect_model(&m).expect("ok");
        assert_eq!(report.n_gaussians, 10);
    }

    #[test]
    fn test_inspect_model_transparent_fraction() {
        // 2 transparent + 2 opaque
        let m = InspectableModel::new(
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0],
            vec![0.05, 0.05, 0.9, 0.9],
            vec![-1.0; 12],
            vec![0.5; 12],
        )
        .expect("ok");

        let report = inspect_model(&m).expect("ok");
        assert!((report.transparent_fraction - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_inspect_model_empty() {
        let m = InspectableModel {
            positions: vec![],
            opacities: vec![],
            scales: vec![],
            colors: vec![],
            n: 0,
        };
        assert!(matches!(inspect_model(&m), Err(InspectorError::EmptyModel)));
    }

    // -----------------------------------------------------------------------
    // format_inspection_report
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_report_nonempty() {
        let m = make_line_model(3);
        let report = inspect_model(&m).expect("ok");
        let text = format_inspection_report(&report);
        assert!(!text.is_empty());
    }

    #[test]
    fn test_format_report_contains_key_fields() {
        let m = make_line_model(3);
        let report = inspect_model(&m).expect("ok");
        let text = format_inspection_report(&report);
        assert!(text.contains("Gaussians"));
        assert!(text.contains("Bounding Box"));
        assert!(text.contains("Opacity"));
        assert!(text.contains("Anisotropy"));
        assert!(text.contains("Color"));
    }

    // -----------------------------------------------------------------------
    // dump_gaussians_csv
    // -----------------------------------------------------------------------

    #[test]
    fn test_dump_csv_header() {
        let m = make_line_model(2);
        let csv = dump_gaussians_csv(&m, &[0, 1]).expect("ok");
        assert!(csv.starts_with("index,x,y,z,opacity,max_scale,r,g,b\n"));
    }

    #[test]
    fn test_dump_csv_row_count() {
        let m = make_line_model(5);
        let csv = dump_gaussians_csv(&m, &[0, 2, 4]).expect("ok");
        // header + 3 data rows
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn test_dump_csv_out_of_bounds() {
        let m = make_line_model(2);
        let result = dump_gaussians_csv(&m, &[0, 99]);
        assert!(matches!(
            result,
            Err(InspectorError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_dump_csv_empty_indices() {
        let m = make_line_model(3);
        let csv = dump_gaussians_csv(&m, &[]).expect("ok");
        // Only the header row
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 1);
    }

    // -----------------------------------------------------------------------
    // density_voxel_grid
    // -----------------------------------------------------------------------

    #[test]
    fn test_density_voxel_grid_counts_sum() {
        let m = make_line_model(8);
        let (counts, _voxel_size) = density_voxel_grid(&m, 4).expect("ok");
        let total: u32 = counts.iter().sum();
        assert_eq!(total, 8);
    }

    #[test]
    fn test_density_voxel_grid_output_size() {
        let m = make_line_model(4);
        let (counts, _voxel_size) = density_voxel_grid(&m, 8).expect("ok");
        assert_eq!(counts.len(), 8 * 8 * 8);
    }

    #[test]
    fn test_density_voxel_grid_single_gaussian() {
        let m = make_line_model(1);
        let (counts, _) = density_voxel_grid(&m, 4).expect("ok");
        let total: u32 = counts.iter().sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn test_density_voxel_grid_empty_error() {
        let m = InspectableModel {
            positions: vec![],
            opacities: vec![],
            scales: vec![],
            colors: vec![],
            n: 0,
        };
        assert!(matches!(
            density_voxel_grid(&m, 4),
            Err(InspectorError::EmptyModel)
        ));
    }

    #[test]
    fn test_density_voxel_grid_zero_grid_error() {
        let m = make_line_model(3);
        assert!(matches!(
            density_voxel_grid(&m, 0),
            Err(InspectorError::InvalidQuery(_))
        ));
    }
}
