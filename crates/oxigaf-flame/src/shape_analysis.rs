//! Statistical analysis tools for FLAME shape space exploration.
//!
//! Provides distance metrics, descriptive statistics, PCA via power iteration,
//! outlier detection, and shape interpolation for exploring the FLAME shape
//! parameter space.

use std::fmt;
use std::fmt::Write as FmtWrite;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during shape analysis operations.
#[derive(Debug)]
pub enum ShapeAnalysisError {
    /// Input collection is empty.
    EmptyInput,
    /// Two vectors have different lengths.
    LengthMismatch { expected: usize, got: usize },
    /// Not enough samples for the requested operation.
    InsufficientSamples { required: usize, got: usize },
    /// Requested more components than the data supports.
    InvalidComponentCount { requested: usize, max: usize },
    /// `MahalanobisDiag` metric requires `std_devs` to be provided.
    StdDevRequired,
    /// Array index exceeds valid range.
    IndexOutOfBounds { idx: usize, max: usize },
}

impl fmt::Display for ShapeAnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "input is empty"),
            Self::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            }
            Self::InsufficientSamples { required, got } => {
                write!(f, "insufficient samples: need {required}, got {got}")
            }
            Self::InvalidComponentCount { requested, max } => {
                write!(
                    f,
                    "invalid component count: requested {requested}, max {max}"
                )
            }
            Self::StdDevRequired => {
                write!(f, "std_devs required for MahalanobisDiag metric")
            }
            Self::IndexOutOfBounds { idx, max } => {
                write!(f, "index {idx} out of bounds (max {max})")
            }
        }
    }
}

impl std::error::Error for ShapeAnalysisError {}

// ---------------------------------------------------------------------------
// Xorshift64 PRNG (mirrors param_sampler.rs — no rand crate)
// ---------------------------------------------------------------------------

/// Advance xorshift64 state and return the next pseudo-random u64.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    if *state == 0 {
        *state = 0x853c_49e6_748f_ea9b;
    }
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a pseudo-random `f32` in `[0.0, 1.0)` from xorshift64 state.
#[inline]
fn rand_f32(state: &mut u64) -> f32 {
    let bits = xorshift64(state);
    let mantissa = (bits >> 41) as u32;
    let float_bits: u32 = 0x3f80_0000u32 | mantissa;
    f32::from_bits(float_bits) - 1.0_f32
}

// ---------------------------------------------------------------------------
// Distance metrics
// ---------------------------------------------------------------------------

/// Distance metric for comparing shape parameter vectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShapeDistanceMetric {
    /// L2 (Euclidean) distance in parameter space.
    L2,
    /// Cosine similarity-based distance: `1 - cos_similarity`.
    Cosine,
    /// Mahalanobis distance using a diagonal covariance (per-dim std).
    MahalanobisDiag,
}

/// Compute distance between two shape parameter vectors.
///
/// `std_devs` is only required for [`ShapeDistanceMetric::MahalanobisDiag`].
///
/// # Errors
///
/// - [`ShapeAnalysisError::EmptyInput`] if either slice is empty.
/// - [`ShapeAnalysisError::LengthMismatch`] if `a` and `b` have different lengths.
/// - [`ShapeAnalysisError::StdDevRequired`] if metric is `MahalanobisDiag` and
///   `std_devs` is `None`.
/// - [`ShapeAnalysisError::LengthMismatch`] if `std_devs` length differs from `a`.
pub fn compute_shape_distance(
    a: &[f32],
    b: &[f32],
    metric: ShapeDistanceMetric,
    std_devs: Option<&[f32]>,
) -> Result<f32, ShapeAnalysisError> {
    if a.is_empty() {
        return Err(ShapeAnalysisError::EmptyInput);
    }
    if a.len() != b.len() {
        return Err(ShapeAnalysisError::LengthMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }

    match metric {
        ShapeDistanceMetric::L2 => {
            let sum_sq: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(ai, bi)| (ai - bi).powi(2))
                .sum();
            Ok(sum_sq.sqrt())
        }
        ShapeDistanceMetric::Cosine => {
            let dot: f32 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm_a < 1e-12 || norm_b < 1e-12 {
                return Ok(1.0);
            }
            let cos_sim = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
            Ok(1.0 - cos_sim)
        }
        ShapeDistanceMetric::MahalanobisDiag => {
            let devs = std_devs.ok_or(ShapeAnalysisError::StdDevRequired)?;
            if devs.len() != a.len() {
                return Err(ShapeAnalysisError::LengthMismatch {
                    expected: a.len(),
                    got: devs.len(),
                });
            }
            let sum_sq: f32 = a
                .iter()
                .zip(b.iter())
                .zip(devs.iter())
                .map(|((ai, bi), si)| {
                    let s = si.max(1e-8_f32);
                    ((ai - bi) / s).powi(2)
                })
                .sum();
            Ok(sum_sq.sqrt())
        }
    }
}

// ---------------------------------------------------------------------------
// Shape statistics
// ---------------------------------------------------------------------------

/// Descriptive statistics for a collection of shape vectors.
#[derive(Debug, Clone)]
pub struct ShapeStatistics {
    /// Number of shape samples.
    pub num_samples: usize,
    /// Dimensionality of each shape vector.
    pub num_dims: usize,
    /// Per-dimension mean.
    pub mean: Vec<f32>,
    /// Per-dimension standard deviation (population std).
    pub std: Vec<f32>,
    /// Per-dimension minimum.
    pub min: Vec<f32>,
    /// Per-dimension maximum.
    pub max: Vec<f32>,
    /// Pearson correlation between dimensions 0 and 1 (0.0 if `num_dims < 2`).
    pub first_two_correlation: f32,
}

impl ShapeStatistics {
    /// Compute statistics from a set of shape vectors.
    ///
    /// All vectors must have the same non-zero length.
    ///
    /// # Errors
    ///
    /// - [`ShapeAnalysisError::EmptyInput`] if `shapes` is empty.
    /// - [`ShapeAnalysisError::LengthMismatch`] if any vector has a different
    ///   length than the first.
    pub fn compute(shapes: &[Vec<f32>]) -> Result<Self, ShapeAnalysisError> {
        if shapes.is_empty() {
            return Err(ShapeAnalysisError::EmptyInput);
        }
        let num_dims = shapes[0].len();
        if num_dims == 0 {
            return Err(ShapeAnalysisError::EmptyInput);
        }
        for (idx, s) in shapes.iter().enumerate() {
            if s.len() != num_dims {
                return Err(ShapeAnalysisError::LengthMismatch {
                    expected: num_dims,
                    got: s.len(),
                });
            }
            let _ = idx;
        }

        let n = shapes.len();
        let nf = n as f32;

        // Mean
        let mut mean = vec![0.0_f32; num_dims];
        for s in shapes {
            for (m, v) in mean.iter_mut().zip(s.iter()) {
                *m += v;
            }
        }
        for m in &mut mean {
            *m /= nf;
        }

        // Variance → std, min, max
        let mut variance = vec![0.0_f32; num_dims];
        let mut min_vals = shapes[0].clone();
        let mut max_vals = shapes[0].clone();

        for s in shapes {
            for d in 0..num_dims {
                let diff = s[d] - mean[d];
                variance[d] += diff * diff;
                if s[d] < min_vals[d] {
                    min_vals[d] = s[d];
                }
                if s[d] > max_vals[d] {
                    max_vals[d] = s[d];
                }
            }
        }
        let std_vals: Vec<f32> = variance.iter().map(|v| (v / nf).sqrt()).collect();

        // First-two-dims Pearson correlation
        let first_two_correlation = if num_dims >= 2 {
            let sx = std_vals[0];
            let sy = std_vals[1];
            if sx < 1e-12 || sy < 1e-12 {
                0.0
            } else {
                let cov: f32 = shapes
                    .iter()
                    .map(|s| (s[0] - mean[0]) * (s[1] - mean[1]))
                    .sum::<f32>()
                    / nf;
                cov / (sx * sy)
            }
        } else {
            0.0
        };

        Ok(Self {
            num_samples: n,
            num_dims,
            mean,
            std: std_vals,
            min: min_vals,
            max: max_vals,
            first_two_correlation,
        })
    }

    /// Format a human-readable summary of the statistics.
    #[must_use]
    pub fn format_summary(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "=== Shape Statistics ===");
        let _ = writeln!(out, "  Samples:    {}", self.num_samples);
        let _ = writeln!(out, "  Dims:       {}", self.num_dims);
        let _ = writeln!(
            out,
            "  Mean[0]:    {:.6}",
            self.mean.first().copied().unwrap_or(0.0)
        );
        let _ = writeln!(
            out,
            "  Std[0]:     {:.6}",
            self.std.first().copied().unwrap_or(0.0)
        );
        let _ = writeln!(
            out,
            "  Range[0]:   [{:.4}, {:.4}]",
            self.min.first().copied().unwrap_or(0.0),
            self.max.first().copied().unwrap_or(0.0)
        );
        if self.num_dims >= 2 {
            let _ = writeln!(out, "  Corr(0,1):  {:.4}", self.first_two_correlation);
        }
        out
    }

    /// Standardize a shape vector using the computed mean and std.
    ///
    /// Each dimension `d` becomes `(shape[d] - mean[d]) / max(std[d], 1e-8)`.
    ///
    /// # Errors
    ///
    /// - [`ShapeAnalysisError::LengthMismatch`] if `shape.len() != self.num_dims`.
    pub fn standardize(&self, shape: &[f32]) -> Result<Vec<f32>, ShapeAnalysisError> {
        if shape.len() != self.num_dims {
            return Err(ShapeAnalysisError::LengthMismatch {
                expected: self.num_dims,
                got: shape.len(),
            });
        }
        let result: Vec<f32> = shape
            .iter()
            .zip(self.mean.iter())
            .zip(self.std.iter())
            .map(|((x, m), s)| (x - m) / s.max(1e-8_f32))
            .collect();
        Ok(result)
    }

    /// Returns `true` if the shape is within `n_sigma` standard deviations of
    /// the mean for every dimension.
    ///
    /// Returns `false` if `shape.len() != num_dims`.
    #[must_use]
    pub fn is_typical(&self, shape: &[f32], n_sigma: f32) -> bool {
        if shape.len() != self.num_dims {
            return false;
        }
        shape
            .iter()
            .zip(self.mean.iter())
            .zip(self.std.iter())
            .all(|((x, m), s)| {
                let threshold = s.max(1e-8_f32) * n_sigma;
                (x - m).abs() <= threshold
            })
    }
}

// ---------------------------------------------------------------------------
// PCA via power iteration
// ---------------------------------------------------------------------------

/// PCA of a shape parameter dataset, fitted via deflated power iteration.
pub struct ShapeSpacePca {
    /// Principal components — K unit vectors, each of length `num_dims`.
    pub principal_components: Vec<Vec<f32>>,
    /// Explained variance (eigenvalue) for each PC.
    pub explained_variance: Vec<f32>,
    /// Mean of the training data used for centering.
    pub mean: Vec<f32>,
    /// Dimensionality of shape vectors.
    pub num_dims: usize,
    /// Number of principal components retained.
    pub num_components: usize,
    /// Total variance of the (centered) training data.
    pub total_variance: f32,
}

impl ShapeSpacePca {
    /// Fit PCA to a set of shape vectors using deflated power iteration.
    ///
    /// `num_components` is clamped to `min(num_components, num_dims, num_samples)`.
    ///
    /// # Errors
    ///
    /// - [`ShapeAnalysisError::EmptyInput`] if `shapes` is empty.
    /// - [`ShapeAnalysisError::LengthMismatch`] if any vector has a different length.
    /// - [`ShapeAnalysisError::InvalidComponentCount`] if `num_components` is 0.
    pub fn fit(
        shapes: &[Vec<f32>],
        num_components: usize,
        max_iter: usize,
    ) -> Result<Self, ShapeAnalysisError> {
        if shapes.is_empty() {
            return Err(ShapeAnalysisError::EmptyInput);
        }
        let num_dims = shapes[0].len();
        if num_dims == 0 {
            return Err(ShapeAnalysisError::EmptyInput);
        }
        for s in shapes.iter().skip(1) {
            if s.len() != num_dims {
                return Err(ShapeAnalysisError::LengthMismatch {
                    expected: num_dims,
                    got: s.len(),
                });
            }
        }
        if num_components == 0 {
            return Err(ShapeAnalysisError::InvalidComponentCount {
                requested: 0,
                max: num_dims.min(shapes.len()),
            });
        }

        let num_samples = shapes.len();
        let max_k = num_dims.min(num_samples);
        let k = num_components.min(max_k);

        // Compute mean
        let mut mean = vec![0.0_f32; num_dims];
        for s in shapes {
            for (m, v) in mean.iter_mut().zip(s.iter()) {
                *m += v;
            }
        }
        let nf = num_samples as f32;
        for m in &mut mean {
            *m /= nf;
        }

        // Center the data
        let mut centered: Vec<Vec<f32>> = shapes
            .iter()
            .map(|s| s.iter().zip(mean.iter()).map(|(v, m)| v - m).collect())
            .collect();

        // Total variance = mean of squared norms of centered samples
        let total_variance: f32 = centered
            .iter()
            .map(|s| s.iter().map(|x| x * x).sum::<f32>())
            .sum::<f32>()
            / nf;

        let max_iter = max_iter.max(1);

        let mut principal_components = Vec::with_capacity(k);
        let mut explained_variance = Vec::with_capacity(k);

        for comp_idx in 0..k {
            let seed = 0xdead_beef_u64.wrapping_add(comp_idx as u64 * 0x9e37_79b9_7f4a_7c15);
            let (pc, eigenval) = power_iteration_pc(&centered, num_dims, max_iter, seed);
            deflate_in_place(&mut centered, &pc);
            principal_components.push(pc);
            explained_variance.push(eigenval / nf);
        }

        Ok(Self {
            principal_components,
            explained_variance,
            mean,
            num_dims,
            num_components: k,
            total_variance,
        })
    }

    /// Project a shape vector onto the PC space (returns K coordinates).
    ///
    /// # Errors
    ///
    /// - [`ShapeAnalysisError::LengthMismatch`] if `shape.len() != num_dims`.
    pub fn project(&self, shape: &[f32]) -> Result<Vec<f32>, ShapeAnalysisError> {
        if shape.len() != self.num_dims {
            return Err(ShapeAnalysisError::LengthMismatch {
                expected: self.num_dims,
                got: shape.len(),
            });
        }
        let centered: Vec<f32> = shape
            .iter()
            .zip(self.mean.iter())
            .map(|(v, m)| v - m)
            .collect();
        let coords = self
            .principal_components
            .iter()
            .map(|pc| dot_product(&centered, pc))
            .collect();
        Ok(coords)
    }

    /// Reconstruct a shape from K PC coordinates.
    ///
    /// # Errors
    ///
    /// - [`ShapeAnalysisError::LengthMismatch`] if `pc_coords.len() != num_components`.
    pub fn reconstruct(&self, pc_coords: &[f32]) -> Result<Vec<f32>, ShapeAnalysisError> {
        if pc_coords.len() != self.num_components {
            return Err(ShapeAnalysisError::LengthMismatch {
                expected: self.num_components,
                got: pc_coords.len(),
            });
        }
        let mut result = self.mean.clone();
        for (coord, pc) in pc_coords.iter().zip(self.principal_components.iter()) {
            for (r, p) in result.iter_mut().zip(pc.iter()) {
                *r += coord * p;
            }
        }
        Ok(result)
    }

    /// L2 reconstruction error between the original shape and its PCA approximation.
    ///
    /// # Errors
    ///
    /// - [`ShapeAnalysisError::LengthMismatch`] if `shape.len() != num_dims`.
    pub fn reconstruction_error(&self, shape: &[f32]) -> Result<f32, ShapeAnalysisError> {
        let coords = self.project(shape)?;
        let reconstructed = self.reconstruct(&coords)?;
        let err: f32 = shape
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        Ok(err)
    }

    /// Fraction of total variance explained by the retained PCs.
    ///
    /// Returns `sum(explained_variance) / total_variance`, or 0.0 if
    /// `total_variance` is near-zero.
    #[must_use]
    pub fn explained_variance_ratio(&self) -> f32 {
        if self.total_variance < 1e-10 {
            return 0.0;
        }
        let captured: f32 = self.explained_variance.iter().sum();
        captured / self.total_variance
    }

    /// Generate `n_samples` shapes along PC `pc_idx` from `-n_sigma*sqrt(var)` to `+n_sigma*sqrt(var)`.
    ///
    /// # Errors
    ///
    /// - [`ShapeAnalysisError::IndexOutOfBounds`] if `pc_idx >= num_components`.
    /// - [`ShapeAnalysisError::EmptyInput`] if `n_samples == 0`.
    pub fn pc_variation_samples(
        &self,
        pc_idx: usize,
        n_samples: usize,
        n_sigma: f32,
    ) -> Result<Vec<Vec<f32>>, ShapeAnalysisError> {
        if pc_idx >= self.num_components {
            return Err(ShapeAnalysisError::IndexOutOfBounds {
                idx: pc_idx,
                max: self.num_components.saturating_sub(1),
            });
        }
        if n_samples == 0 {
            return Err(ShapeAnalysisError::EmptyInput);
        }
        let std_pc = self.explained_variance[pc_idx].max(0.0_f32).sqrt();
        let pc = &self.principal_components[pc_idx];
        let range = n_sigma * std_pc;

        let mut samples = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let t = if n_samples == 1 {
                0.0_f32
            } else {
                -1.0 + 2.0 * (i as f32) / ((n_samples - 1) as f32)
            };
            let alpha = t * range;
            let shape: Vec<f32> = self
                .mean
                .iter()
                .zip(pc.iter())
                .map(|(m, p)| m + alpha * p)
                .collect();
            samples.push(shape);
        }
        Ok(samples)
    }
}

// ---------------------------------------------------------------------------
// Power iteration helpers
// ---------------------------------------------------------------------------

/// Run one round of power iteration to find the dominant eigenvector of A^T A.
///
/// Returns `(unit_eigenvector, eigenvalue)` where eigenvalue ≈ sum over
/// samples of `(sample · v)^2` (before dividing by n; caller normalises).
fn power_iteration_pc(
    centered_data: &[Vec<f32>],
    num_dims: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<f32>, f32) {
    // Initialise v with small xorshift random perturbations around 1.0
    let mut rng_state = seed;
    let mut v: Vec<f32> = (0..num_dims)
        .map(|_| 1.0_f32 + 0.01_f32 * (rand_f32(&mut rng_state) - 0.5_f32))
        .collect();
    normalize_in_place(&mut v);

    for _ in 0..max_iter {
        // w = A^T A v  =  sum_i (sample_i · v) * sample_i
        let mut w = vec![0.0_f32; num_dims];
        for sample in centered_data {
            let proj = dot_product(sample, &v);
            for (wi, si) in w.iter_mut().zip(sample.iter()) {
                *wi += proj * si;
            }
        }
        let norm: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-12 {
            break;
        }
        for (vi, wi) in v.iter_mut().zip(w.iter()) {
            *vi = wi / norm;
        }
    }

    // Eigenvalue = sum_i (sample_i · v)^2
    let eigenvalue: f32 = centered_data
        .iter()
        .map(|s| dot_product(s, &v).powi(2))
        .sum();

    (v, eigenvalue)
}

/// Remove the projection onto `pc` from every sample (deflation).
fn deflate_in_place(data: &mut [Vec<f32>], pc: &[f32]) {
    for sample in data.iter_mut() {
        let proj = dot_product(sample, pc);
        for (si, pi) in sample.iter_mut().zip(pc.iter()) {
            *si -= proj * pi;
        }
    }
}

// ---------------------------------------------------------------------------
// Inline helpers
// ---------------------------------------------------------------------------

#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn normalize_in_place(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Outlier detection
// ---------------------------------------------------------------------------

/// Detects outlier shapes based on per-dimension deviation from training statistics.
pub struct ShapeOutlierDetector {
    /// Training statistics used for outlier scoring.
    pub stats: ShapeStatistics,
    /// Threshold in sigma units; shapes beyond this are flagged as outliers.
    pub threshold_sigma: f32,
}

impl ShapeOutlierDetector {
    /// Create a new detector from training shapes.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`ShapeStatistics::compute`].
    pub fn new(
        training_shapes: &[Vec<f32>],
        threshold_sigma: f32,
    ) -> Result<Self, ShapeAnalysisError> {
        let stats = ShapeStatistics::compute(training_shapes)?;
        Ok(Self {
            stats,
            threshold_sigma,
        })
    }

    /// Classify a shape as an outlier.
    ///
    /// Returns `(is_outlier, per_dim_deviation_in_sigma)`.
    ///
    /// A shape is an outlier if any dimension exceeds `threshold_sigma`.
    ///
    /// # Errors
    ///
    /// - [`ShapeAnalysisError::LengthMismatch`] if `shape.len() != num_dims`.
    pub fn detect(&self, shape: &[f32]) -> Result<(bool, Vec<f32>), ShapeAnalysisError> {
        if shape.len() != self.stats.num_dims {
            return Err(ShapeAnalysisError::LengthMismatch {
                expected: self.stats.num_dims,
                got: shape.len(),
            });
        }
        let deviations: Vec<f32> = shape
            .iter()
            .zip(self.stats.mean.iter())
            .zip(self.stats.std.iter())
            .map(|((x, m), s)| (x - m).abs() / s.max(1e-8_f32))
            .collect();
        let is_outlier = deviations.iter().any(|&d| d > self.threshold_sigma);
        Ok((is_outlier, deviations))
    }

    /// Return the top-`k` most deviant dimensions as `(dim_index, deviation_sigma)`.
    ///
    /// Sorted in descending order of deviation.
    ///
    /// # Errors
    ///
    /// - [`ShapeAnalysisError::LengthMismatch`] if `shape.len() != num_dims`.
    pub fn top_outlier_dims(
        &self,
        shape: &[f32],
        k: usize,
    ) -> Result<Vec<(usize, f32)>, ShapeAnalysisError> {
        let (_, deviations) = self.detect(shape)?;
        let mut indexed: Vec<(usize, f32)> = deviations.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        Ok(indexed)
    }
}

// ---------------------------------------------------------------------------
// Shape interpolation
// ---------------------------------------------------------------------------

/// Generate `n_waypoints` evenly-spaced shapes along the line from `shape_a` to `shape_b`.
///
/// The returned vector includes both endpoints (`shape_a` at index 0 and
/// `shape_b` at the last index).
///
/// # Errors
///
/// - [`ShapeAnalysisError::EmptyInput`] if either input is empty or `n_waypoints == 0`.
/// - [`ShapeAnalysisError::LengthMismatch`] if the shapes have different lengths.
pub fn shape_interpolation_path(
    shape_a: &[f32],
    shape_b: &[f32],
    n_waypoints: usize,
) -> Result<Vec<Vec<f32>>, ShapeAnalysisError> {
    if shape_a.is_empty() || shape_b.is_empty() {
        return Err(ShapeAnalysisError::EmptyInput);
    }
    if shape_a.len() != shape_b.len() {
        return Err(ShapeAnalysisError::LengthMismatch {
            expected: shape_a.len(),
            got: shape_b.len(),
        });
    }
    if n_waypoints == 0 {
        return Err(ShapeAnalysisError::EmptyInput);
    }

    let mut path = Vec::with_capacity(n_waypoints);
    for i in 0..n_waypoints {
        let t = if n_waypoints == 1 {
            0.0_f32
        } else {
            i as f32 / (n_waypoints - 1) as f32
        };
        let waypoint: Vec<f32> = shape_a
            .iter()
            .zip(shape_b.iter())
            .map(|(a, b)| a + t * (b - a))
            .collect();
        path.push(waypoint);
    }
    Ok(path)
}

/// Compute the cumulative arc length of a shape path.
///
/// Returns 0.0 for paths with fewer than 2 waypoints.
#[must_use]
pub fn path_arc_length(path: &[Vec<f32>]) -> f32 {
    if path.len() < 2 {
        return 0.0;
    }
    path.windows(2)
        .map(|w| {
            let a = &w[0];
            let b = &w[1];
            a.iter()
                .zip(b.iter())
                .map(|(ai, bi)| (ai - bi).powi(2))
                .sum::<f32>()
                .sqrt()
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Helpers -----------------------------------------------------------

    fn make_shape(vals: &[f32]) -> Vec<f32> {
        vals.to_vec()
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    // ---- Distance metrics --------------------------------------------------

    #[test]
    fn test_compute_distance_l2() {
        let a = make_shape(&[1.0, 0.0, 0.0]);
        let b = make_shape(&[0.0, 0.0, 0.0]);
        let d = compute_shape_distance(&a, &b, ShapeDistanceMetric::L2, None).unwrap();
        assert!(approx(d, 1.0), "expected 1.0, got {d}");
    }

    #[test]
    fn test_compute_distance_cosine_orthogonal() {
        // Orthogonal vectors → cosine similarity = 0 → distance = 1.0
        let a = make_shape(&[1.0, 0.0]);
        let b = make_shape(&[0.0, 1.0]);
        let d = compute_shape_distance(&a, &b, ShapeDistanceMetric::Cosine, None).unwrap();
        assert!(approx(d, 1.0), "expected 1.0, got {d}");
    }

    #[test]
    fn test_compute_distance_mahalanobis() {
        let a = make_shape(&[2.0, 0.0]);
        let b = make_shape(&[0.0, 0.0]);
        let std_devs = make_shape(&[2.0, 1.0]);
        // diff = [2,0], std = [2,1] → scaled = [1,0] → dist = 1.0
        let d = compute_shape_distance(
            &a,
            &b,
            ShapeDistanceMetric::MahalanobisDiag,
            Some(&std_devs),
        )
        .unwrap();
        assert!(approx(d, 1.0), "expected 1.0, got {d}");
    }

    #[test]
    fn test_compute_distance_mahalanobis_no_stddevs() {
        let a = make_shape(&[1.0]);
        let b = make_shape(&[0.0]);
        let err = compute_shape_distance(&a, &b, ShapeDistanceMetric::MahalanobisDiag, None);
        assert!(matches!(err, Err(ShapeAnalysisError::StdDevRequired)));
    }

    // ---- Shape statistics --------------------------------------------------

    #[test]
    fn test_shape_statistics_compute() {
        let shapes = vec![vec![0.0_f32, 0.0], vec![2.0_f32, 2.0]];
        let stats = ShapeStatistics::compute(&shapes).unwrap();
        assert_eq!(stats.num_samples, 2);
        assert_eq!(stats.num_dims, 2);
        assert!(approx(stats.mean[0], 1.0), "mean[0]={}", stats.mean[0]);
        assert!(approx(stats.mean[1], 1.0), "mean[1]={}", stats.mean[1]);
        // Population std = 1.0
        assert!(approx(stats.std[0], 1.0), "std[0]={}", stats.std[0]);
        assert!(approx(stats.min[0], 0.0));
        assert!(approx(stats.max[0], 2.0));
    }

    #[test]
    fn test_shape_statistics_standardize() {
        let shapes = vec![vec![0.0_f32, 0.0], vec![2.0_f32, 2.0]];
        let stats = ShapeStatistics::compute(&shapes).unwrap();
        // Mean = [1,1], std = [1,1]. Standardize [1.0, 1.0] → [0.0, 0.0]
        let s = stats.standardize(&[1.0, 1.0]).unwrap();
        assert!(approx(s[0], 0.0), "s[0]={}", s[0]);
        assert!(approx(s[1], 0.0), "s[1]={}", s[1]);
        // Standardize [2.0, 0.0] → [1.0, -1.0]
        let s2 = stats.standardize(&[2.0, 0.0]).unwrap();
        assert!(approx(s2[0], 1.0), "s2[0]={}", s2[0]);
        assert!(approx(s2[1], -1.0), "s2[1]={}", s2[1]);
    }

    #[test]
    fn test_shape_statistics_is_typical() {
        let shapes = vec![vec![0.0_f32, 0.0], vec![2.0_f32, 2.0]];
        let stats = ShapeStatistics::compute(&shapes).unwrap();
        // Mean = [1,1], std = [1,1]. Shape [1,1] within 2 sigma → typical
        assert!(stats.is_typical(&[1.0, 1.0], 2.0));
        // Shape [10, 1] is 9 sigma away → not typical at 2 sigma
        assert!(!stats.is_typical(&[10.0, 1.0], 2.0));
        // Wrong length → not typical
        assert!(!stats.is_typical(&[1.0], 2.0));
    }

    // ---- PCA ---------------------------------------------------------------

    fn pca_data() -> Vec<Vec<f32>> {
        // Data lies primarily along [1,0]: first PC should be close to [1,0]
        vec![
            vec![2.0_f32, 0.1],
            vec![-2.0_f32, -0.1],
            vec![1.0_f32, 0.05],
            vec![-1.0_f32, -0.05],
            vec![3.0_f32, 0.0],
            vec![-3.0_f32, 0.0],
        ]
    }

    #[test]
    fn test_pca_fit_single_component() {
        let data = pca_data();
        let pca = ShapeSpacePca::fit(&data, 1, 100).unwrap();
        assert_eq!(pca.num_components, 1);
        assert_eq!(pca.principal_components.len(), 1);
        // First PC should be close to unit vector [±1, ~0]
        let pc = &pca.principal_components[0];
        let dominant = pc[0].abs();
        assert!(dominant > 0.98, "first PC[0]={dominant}");
    }

    #[test]
    fn test_pca_project_and_reconstruct() {
        let data = pca_data();
        let pca = ShapeSpacePca::fit(&data, 2, 100).unwrap();
        let shape = vec![1.5_f32, 0.0];
        let coords = pca.project(&shape).unwrap();
        let reconstructed = pca.reconstruct(&coords).unwrap();
        // With 2 components in 2D, reconstruction should be near-perfect
        for (orig, rec) in shape.iter().zip(reconstructed.iter()) {
            assert!((orig - rec).abs() < 0.05, "orig={orig} rec={rec}");
        }
    }

    #[test]
    fn test_pca_reconstruction_error_small() {
        let data = pca_data();
        let pca = ShapeSpacePca::fit(&data, 2, 100).unwrap();
        // One of the training samples should reconstruct almost perfectly
        let err = pca.reconstruction_error(&data[0]).unwrap();
        assert!(err < 0.1, "reconstruction error={err}");
    }

    #[test]
    fn test_pca_pc_variation_samples() {
        let data = pca_data();
        let pca = ShapeSpacePca::fit(&data, 2, 100).unwrap();
        let samples = pca.pc_variation_samples(0, 5, 1.0).unwrap();
        assert_eq!(samples.len(), 5);
        // All samples should have the correct dimensionality
        for s in &samples {
            assert_eq!(s.len(), 2);
        }
        // Middle sample (index 2) should be close to mean
        let mid = &samples[2];
        for (m, v) in pca.mean.iter().zip(mid.iter()) {
            assert!((m - v).abs() < 1e-4, "mid differs from mean: m={m} v={v}");
        }
    }

    #[test]
    fn test_pca_explained_variance_ratio() {
        let data = pca_data();
        let pca = ShapeSpacePca::fit(&data, 2, 100).unwrap();
        let ratio = pca.explained_variance_ratio();
        // Data lives almost entirely in 1D, so 2 components should explain close to 100%
        assert!(ratio > 0.95, "explained variance ratio={ratio}");
    }

    // ---- Outlier detection -------------------------------------------------

    #[test]
    fn test_outlier_detector_not_outlier() {
        let training: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![i as f32 * 0.1, -(i as f32) * 0.1])
            .collect();
        let detector = ShapeOutlierDetector::new(&training, 3.0).unwrap();
        // The mean itself should not be an outlier
        let mean = detector.stats.mean.clone();
        let (is_out, _) = detector.detect(&mean).unwrap();
        assert!(!is_out, "mean should not be an outlier");
    }

    #[test]
    fn test_outlier_detector_is_outlier() {
        let training: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32 * 0.1]).collect();
        let detector = ShapeOutlierDetector::new(&training, 3.0).unwrap();
        // Shape 1000x away from typical range is definitely an outlier
        let far = vec![1000.0_f32];
        let (is_out, _) = detector.detect(&far).unwrap();
        assert!(is_out, "extreme shape should be an outlier");
    }

    #[test]
    fn test_outlier_top_dims() {
        let training: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![i as f32 * 0.1, 0.0_f32, 0.0_f32])
            .collect();
        let detector = ShapeOutlierDetector::new(&training, 3.0).unwrap();
        // Make dim 0 the biggest outlier
        let shape = vec![1000.0_f32, 0.0, 0.0];
        let top = detector.top_outlier_dims(&shape, 1).unwrap();
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, 0, "dim 0 should be the top outlier");
    }

    // ---- Interpolation -----------------------------------------------------

    #[test]
    fn test_interpolation_path_endpoints() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 1.0];
        let path = shape_interpolation_path(&a, &b, 5).unwrap();
        // First point = a
        assert!(approx(path[0][0], 0.0));
        assert!(approx(path[0][1], 0.0));
        // Last point = b
        assert!(approx(path[4][0], 1.0));
        assert!(approx(path[4][1], 1.0));
    }

    #[test]
    fn test_interpolation_path_count() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 1.0];
        let path = shape_interpolation_path(&a, &b, 7).unwrap();
        assert_eq!(path.len(), 7);
    }

    #[test]
    fn test_interpolation_midpoint() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![2.0_f32, 4.0];
        let path = shape_interpolation_path(&a, &b, 3).unwrap();
        // Middle point should be [1.0, 2.0]
        assert!(approx(path[1][0], 1.0), "mid[0]={}", path[1][0]);
        assert!(approx(path[1][1], 2.0), "mid[1]={}", path[1][1]);
    }

    #[test]
    fn test_path_arc_length() {
        // Two steps each of length 1.0 → total arc = 2.0
        let path = vec![vec![0.0_f32, 0.0], vec![1.0_f32, 0.0], vec![2.0_f32, 0.0]];
        let len = path_arc_length(&path);
        assert!(approx(len, 2.0), "arc_length={len}");
    }

    #[test]
    fn test_path_arc_length_single() {
        let path = vec![vec![0.0_f32, 1.0]];
        assert!(approx(path_arc_length(&path), 0.0));
    }
}
