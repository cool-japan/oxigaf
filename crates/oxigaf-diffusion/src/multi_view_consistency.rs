//! # Multi-View Consistency
//!
//! Tools for measuring and enforcing consistency across multiple rendered views
//! of the same 3D avatar subject. Provides pairwise distance metrics, per-pixel
//! consistency maps, weighted aggregation, and soft consistency losses for use in
//! training and inference quality evaluation.

use thiserror::Error;

// ───────────────────────────────────────────────────────────────────────────────
// Error type
// ───────────────────────────────────────────────────────────────────────────────

/// Errors produced by multi-view consistency operations.
#[derive(Debug, Error)]
pub enum ConsistencyError {
    /// Fewer than two views were provided; pairwise metrics are undefined.
    #[error("Empty view bundle: need at least 2 views")]
    TooFewViews,

    /// A view pixel buffer has an unexpected length.
    #[error("Image size mismatch: view {idx} has {actual} pixels, expected {expected}")]
    SizeMismatch {
        idx: usize,
        actual: usize,
        expected: usize,
    },

    /// Weight vector is unusable (negative weights or all-zero sum).
    #[error("Invalid weight: weights must be non-negative and sum > 0")]
    InvalidWeights,

    /// Mask length does not match the expected pixel count.
    #[error("Mask size mismatch: mask has {actual} elements, expected {expected}")]
    MaskSizeMismatch { actual: usize, expected: usize },

    /// After applying a mask or correspondence region, no valid pixels remain.
    #[error("No valid pixels in correspondence region")]
    EmptyCorrespondence,
}

// ───────────────────────────────────────────────────────────────────────────────
// ViewBundle
// ───────────────────────────────────────────────────────────────────────────────

/// A collection of images rendered from multiple viewpoints of the same subject.
///
/// Each view is stored as a flat `Vec<f32>` in row-major order with pixel values
/// in the `[0, 1]` range.  The total length of each view buffer must equal
/// `width * height * num_channels`.
pub struct ViewBundle {
    /// Pixel buffers, one per view. Each buffer has length `width * height * num_channels`.
    pub views: Vec<Vec<f32>>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Number of channels per pixel (3 = RGB, 4 = RGBA).
    pub num_channels: u32,
}

impl ViewBundle {
    /// Create an empty bundle with fixed image dimensions.
    pub fn new(width: u32, height: u32, num_channels: u32) -> Self {
        ViewBundle {
            views: Vec::new(),
            width,
            height,
            num_channels,
        }
    }

    /// Expected number of `f32` elements in each view buffer.
    fn expected_len(&self) -> usize {
        (self.width as usize) * (self.height as usize) * (self.num_channels as usize)
    }

    /// Add a pre-normalised `f32` view.  Returns an error if the buffer length
    /// does not match `width * height * num_channels`.
    pub fn add_view(&mut self, pixels: Vec<f32>) -> Result<(), ConsistencyError> {
        let expected = self.expected_len();
        if pixels.len() != expected {
            return Err(ConsistencyError::SizeMismatch {
                idx: self.views.len(),
                actual: pixels.len(),
                expected,
            });
        }
        self.views.push(pixels);
        Ok(())
    }

    /// Number of views currently stored.
    pub fn num_views(&self) -> usize {
        self.views.len()
    }

    /// Access the pixel buffer of view `idx`, or `None` if out of range.
    pub fn view(&self, idx: usize) -> Option<&[f32]> {
        self.views.get(idx).map(|v| v.as_slice())
    }

    /// Convert a `u8` RGBA/RGB buffer into the `[0, 1]` range and add it as a
    /// new view.  The byte count must equal `width * height * num_channels`.
    pub fn add_view_u8(&mut self, pixels: &[u8]) -> Result<(), ConsistencyError> {
        let expected = self.expected_len();
        if pixels.len() != expected {
            return Err(ConsistencyError::SizeMismatch {
                idx: self.views.len(),
                actual: pixels.len(),
                expected,
            });
        }
        let float_pixels: Vec<f32> = pixels.iter().map(|&b| b as f32 / 255.0).collect();
        self.views.push(float_pixels);
        Ok(())
    }

    /// Return RGB-only views.  If `num_channels == 4` the alpha channel (index 3
    /// in every pixel group) is dropped; otherwise the views are returned as-is.
    pub fn rgb_views(&self) -> Vec<Vec<f32>> {
        if self.num_channels != 4 {
            return self.views.clone();
        }
        let n_pixels = (self.width as usize) * (self.height as usize);
        self.views
            .iter()
            .map(|view| {
                let mut rgb = Vec::with_capacity(n_pixels * 3);
                for chunk in view.chunks_exact(4) {
                    rgb.push(chunk[0]);
                    rgb.push(chunk[1]);
                    rgb.push(chunk[2]);
                }
                rgb
            })
            .collect()
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// ConsistencyStats
// ───────────────────────────────────────────────────────────────────────────────

/// Summary statistics describing how consistent a set of views is with each other.
pub struct ConsistencyStats {
    /// Mean L1 distance averaged over all ordered pairs.
    pub mean_pairwise_l1: f32,
    /// Mean L2 (MSE) distance averaged over all ordered pairs.
    pub mean_pairwise_l2: f32,
    /// Mean SSIM approximation (luminance correlation mapped to `[0, 1]`).
    pub mean_pairwise_ssim: f32,
    /// Minimum pairwise consistency (worst pair; lower = less consistent).
    pub min_consistency: f32,
    /// Maximum pairwise consistency (best pair).
    pub max_consistency: f32,
    /// Overall consistency score in `[0, 1]`: 1 = perfectly consistent.
    pub consistency_score: f32,
    /// Number of unique unordered view pairs evaluated.
    pub num_view_pairs: usize,
}

// ───────────────────────────────────────────────────────────────────────────────
// CrossViewCorrelation
// ───────────────────────────────────────────────────────────────────────────────

/// Symmetric `n_views × n_views` matrix of pairwise luminance correlations.
pub struct CrossViewCorrelation {
    /// Flat row-major matrix of length `n_views * n_views`.
    pub matrix: Vec<f32>,
    /// Number of views.
    pub n_views: usize,
}

impl CrossViewCorrelation {
    /// Retrieve the correlation between views `i` and `j`.  Returns `None` if
    /// either index is out of range.
    pub fn get(&self, i: usize, j: usize) -> Option<f32> {
        if i >= self.n_views || j >= self.n_views {
            return None;
        }
        self.matrix.get(i * self.n_views + j).copied()
    }

    /// Mean correlation over all off-diagonal entries (i.e. across all pairs).
    pub fn mean_off_diagonal(&self) -> f32 {
        if self.n_views < 2 {
            return 0.0;
        }
        let n = self.n_views;
        let mut sum = 0.0_f32;
        let mut count = 0usize;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    if let Some(v) = self.get(i, j) {
                        sum += v;
                        count += 1;
                    }
                }
            }
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f32
        }
    }

    /// Minimum correlation across all off-diagonal entries.
    pub fn min_correlation(&self) -> f32 {
        let n = self.n_views;
        let mut min_val = f32::INFINITY;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    if let Some(v) = self.get(i, j) {
                        if v < min_val {
                            min_val = v;
                        }
                    }
                }
            }
        }
        if min_val.is_infinite() {
            0.0
        } else {
            min_val
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// ConsistencyLossType / ConsistencyConfig
// ───────────────────────────────────────────────────────────────────────────────

/// Which distance metric to use when computing the consistency loss.
pub enum ConsistencyLossType {
    /// Mean absolute error.
    L1,
    /// Mean squared error.
    L2,
    /// SSIM approximation (luminance correlation).
    Ssim,
    /// Weighted combination of L1 and SSIM.
    Combined { l1_weight: f32, ssim_weight: f32 },
}

/// Configuration for the consistency loss computation.
pub struct ConsistencyConfig {
    /// Which distance metric to use.
    pub loss_type: ConsistencyLossType,
    /// Relative weight of the consistency term vs the content term.
    pub consistency_weight: f32,
    /// Patch side length for patch-based losses (default: 8).
    pub patch_size: u32,
    /// Compare only the luminance channel (derived from RGB) instead of all channels.
    pub use_luminance_only: bool,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        ConsistencyConfig {
            loss_type: ConsistencyLossType::L2,
            consistency_weight: 0.1,
            patch_size: 8,
            use_luminance_only: false,
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ───────────────────────────────────────────────────────────────────────────────

/// Convert an RGB (3-channel) pixel buffer to single-channel luminance using the
/// standard BT.601 coefficients.  RGBA buffers are treated the same way (alpha is
/// ignored by operating on 3-element strides that happen to align to the first
/// three channels).
///
/// If `num_channels` is not 3 or 4 every channel is averaged uniformly instead.
fn to_luminance(pixels: &[f32], num_channels: u32) -> Vec<f32> {
    match num_channels {
        3 => pixels
            .chunks_exact(3)
            .map(|c| 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
            .collect(),
        4 => pixels
            .chunks_exact(4)
            .map(|c| 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
            .collect(),
        _ => {
            let nc = num_channels as usize;
            let inv = 1.0 / nc as f32;
            pixels
                .chunks_exact(nc)
                .map(|c| c.iter().sum::<f32>() * inv)
                .collect()
        }
    }
}

/// Validate that two slices have the same length and return their common length,
/// or emit `ConsistencyError::SizeMismatch` with idx=0 (caller may rewrite the
/// error before propagating).
fn check_same_len(a: &[f32], b: &[f32]) -> Result<usize, ConsistencyError> {
    if a.len() != b.len() {
        return Err(ConsistencyError::SizeMismatch {
            idx: 0,
            actual: b.len(),
            expected: a.len(),
        });
    }
    if a.is_empty() {
        return Err(ConsistencyError::EmptyCorrespondence);
    }
    Ok(a.len())
}

/// Validate that a bundle has at least two views.
fn require_two_views(bundle: &ViewBundle) -> Result<(), ConsistencyError> {
    if bundle.num_views() < 2 {
        Err(ConsistencyError::TooFewViews)
    } else {
        Ok(())
    }
}

// ───────────────────────────────────────────────────────────────────────────────
// Public free functions — distance metrics
// ───────────────────────────────────────────────────────────────────────────────

/// Mean absolute difference between two flat `f32` image arrays.
pub fn l1_distance(a: &[f32], b: &[f32]) -> Result<f32, ConsistencyError> {
    let n = check_same_len(a, b)?;
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    Ok(sum / n as f32)
}

/// Mean squared difference (MSE) between two flat `f32` image arrays.
pub fn l2_distance(a: &[f32], b: &[f32]) -> Result<f32, ConsistencyError> {
    let n = check_same_len(a, b)?;
    let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
    Ok(sum / n as f32)
}

/// Luminance correlation between two image arrays, used as an SSIM proxy.
///
/// Returns a value in `[-1, 1]`:
/// - `1.0`  → identical or constant difference (including the all-zeros edge case)
/// - `-1.0` → perfectly anti-correlated
///
/// Both arrays must have the same length.
pub fn luminance_correlation(a: &[f32], b: &[f32]) -> Result<f32, ConsistencyError> {
    let n = check_same_len(a, b)? as f32;

    let mean_a: f32 = a.iter().sum::<f32>() / n;
    let mean_b: f32 = b.iter().sum::<f32>() / n;

    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let da = x - mean_a;
        let db = y - mean_b;
        dot += da * db;
        norm_a += da * da;
        norm_b += db * db;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-10 {
        // Both arrays are constant — treat as identical.
        return Ok(1.0);
    }
    Ok((dot / denom).clamp(-1.0, 1.0))
}

// ───────────────────────────────────────────────────────────────────────────────
// Pairwise matrix & consistency stats
// ───────────────────────────────────────────────────────────────────────────────

/// Compute an `n × n` symmetric pairwise L2 distance matrix for all views in the
/// bundle.  Element `[i][j]` is the mean squared error between view `i` and view
/// `j` (diagonal is 0).
pub fn compute_pairwise_distances(bundle: &ViewBundle) -> Result<Vec<Vec<f32>>, ConsistencyError> {
    require_two_views(bundle)?;
    let n = bundle.num_views();
    let mut matrix = vec![vec![0.0_f32; n]; n];
    for (i, vi) in bundle.views.iter().enumerate() {
        for (j, vj) in bundle.views.iter().enumerate().skip(i + 1) {
            // Both views were validated when added — lengths are guaranteed equal.
            let d = l2_distance(vi.as_slice(), vj.as_slice())?;
            matrix[i][j] = d;
            matrix[j][i] = d;
        }
    }
    Ok(matrix)
}

/// Compute comprehensive consistency statistics for a `ViewBundle`.
///
/// Requires at least two views.
pub fn compute_consistency_stats(
    bundle: &ViewBundle,
) -> Result<ConsistencyStats, ConsistencyError> {
    require_two_views(bundle)?;
    let n = bundle.num_views();
    let num_pairs = n * (n - 1) / 2;

    let mut total_l1 = 0.0_f32;
    let mut total_l2 = 0.0_f32;
    let mut total_ssim = 0.0_f32;

    // Track consistency per pair (1 - l2 for min/max tracking).
    let mut min_cons = f32::INFINITY;
    let mut max_cons = f32::NEG_INFINITY;

    for i in 0..n {
        for j in (i + 1)..n {
            let vi = bundle.views[i].as_slice();
            let vj = bundle.views[j].as_slice();

            let d1 = l1_distance(vi, vj)?;
            let d2 = l2_distance(vi, vj)?;
            let corr = luminance_correlation(vi, vj)?;
            // Map luminance correlation from [-1,1] to [0,1]
            let ssim_approx = (corr + 1.0) * 0.5;
            let consistency = (1.0_f32 - d2).clamp(0.0, 1.0);

            total_l1 += d1;
            total_l2 += d2;
            total_ssim += ssim_approx;

            if consistency < min_cons {
                min_cons = consistency;
            }
            if consistency > max_cons {
                max_cons = consistency;
            }
        }
    }

    let inv = 1.0 / num_pairs as f32;
    let mean_l1 = total_l1 * inv;
    let mean_l2 = total_l2 * inv;
    let mean_ssim = total_ssim * inv;
    let overall_score = (1.0_f32 - mean_l2).clamp(0.0, 1.0);

    if min_cons.is_infinite() {
        min_cons = 0.0;
    }
    if max_cons.is_infinite() {
        max_cons = 0.0;
    }

    Ok(ConsistencyStats {
        mean_pairwise_l1: mean_l1,
        mean_pairwise_l2: mean_l2,
        mean_pairwise_ssim: mean_ssim,
        min_consistency: min_cons,
        max_consistency: max_cons,
        consistency_score: overall_score,
        num_view_pairs: num_pairs,
    })
}

/// Cross-view luminance-correlation matrix.
pub fn compute_cross_view_correlation(
    bundle: &ViewBundle,
) -> Result<CrossViewCorrelation, ConsistencyError> {
    require_two_views(bundle)?;
    let n = bundle.num_views();
    let nc = bundle.num_channels;

    // Convert each view to luminance for correlation.
    let lum_views: Vec<Vec<f32>> = bundle.views.iter().map(|v| to_luminance(v, nc)).collect();

    let mut matrix = vec![0.0_f32; n * n];
    for i in 0..n {
        matrix[i * n + i] = 1.0; // diagonal = self-correlation
        for j in (i + 1)..n {
            let corr = luminance_correlation(&lum_views[i], &lum_views[j])?;
            matrix[i * n + j] = corr;
            matrix[j * n + i] = corr;
        }
    }

    Ok(CrossViewCorrelation { matrix, n_views: n })
}

// ───────────────────────────────────────────────────────────────────────────────
// Consistency map
// ───────────────────────────────────────────────────────────────────────────────

/// Per-pixel standard deviation across views — a measure of inconsistency.
///
/// Returns a flat `f32` array of length `width * height * num_channels`.  Each
/// element is the normalised standard deviation (divided by `sqrt(3)` so that
/// the range for `[0, 1]` valued inputs is `[0, 1]`) for that channel position
/// across all views.
pub fn consistency_map(bundle: &ViewBundle) -> Result<Vec<f32>, ConsistencyError> {
    require_two_views(bundle)?;
    let pixel_len = bundle.expected_len();
    let n = bundle.num_views() as f32;

    // Compute per-pixel mean.
    let mut mean = vec![0.0_f32; pixel_len];
    for view in &bundle.views {
        for (m, &v) in mean.iter_mut().zip(view.iter()) {
            *m += v;
        }
    }
    for m in mean.iter_mut() {
        *m /= n;
    }

    // Compute variance.
    let mut variance = vec![0.0_f32; pixel_len];
    for view in &bundle.views {
        for (var, (&v, &m)) in variance.iter_mut().zip(view.iter().zip(mean.iter())) {
            let diff = v - m;
            *var += diff * diff;
        }
    }

    // Normalise to [0, 1] by dividing std by sqrt(3).
    let normaliser = 1.0_f32 / 3.0_f32.sqrt();
    let map: Vec<f32> = variance
        .iter()
        .map(|&var| ((var / n).sqrt() * normaliser).clamp(0.0, 1.0))
        .collect();

    Ok(map)
}

// ───────────────────────────────────────────────────────────────────────────────
// View aggregation
// ───────────────────────────────────────────────────────────────────────────────

/// Weighted mean of all views.
///
/// `weights` must have the same length as `bundle.num_views()`, be non-negative,
/// and sum to a positive value.
pub fn weighted_mean_view(
    bundle: &ViewBundle,
    weights: &[f32],
) -> Result<Vec<f32>, ConsistencyError> {
    let n = bundle.num_views();
    if n == 0 {
        return Err(ConsistencyError::TooFewViews);
    }
    if weights.len() != n {
        return Err(ConsistencyError::MaskSizeMismatch {
            actual: weights.len(),
            expected: n,
        });
    }
    if weights.iter().any(|&w| w < 0.0) {
        return Err(ConsistencyError::InvalidWeights);
    }
    let weight_sum: f32 = weights.iter().sum();
    if weight_sum <= 0.0 || !weight_sum.is_finite() {
        return Err(ConsistencyError::InvalidWeights);
    }

    let pixel_len = bundle.expected_len();
    let mut result = vec![0.0_f32; pixel_len];
    for (view, &w) in bundle.views.iter().zip(weights.iter()) {
        for (r, &v) in result.iter_mut().zip(view.iter()) {
            *r += w * v;
        }
    }
    for r in result.iter_mut() {
        *r /= weight_sum;
    }
    Ok(result)
}

/// Uniform mean of all views (equal weights).
pub fn mean_view(bundle: &ViewBundle) -> Result<Vec<f32>, ConsistencyError> {
    let n = bundle.num_views();
    if n == 0 {
        return Err(ConsistencyError::TooFewViews);
    }
    let weights = vec![1.0_f32; n];
    weighted_mean_view(bundle, &weights)
}

/// Per-pixel median across all views.
///
/// For an even number of views the two middle values are averaged.
pub fn median_view(bundle: &ViewBundle) -> Result<Vec<f32>, ConsistencyError> {
    let n = bundle.num_views();
    if n == 0 {
        return Err(ConsistencyError::TooFewViews);
    }
    let pixel_len = bundle.expected_len();
    let mut result = vec![0.0_f32; pixel_len];

    let mut scratch = Vec::with_capacity(n);
    for pos in 0..pixel_len {
        scratch.clear();
        for view in &bundle.views {
            scratch.push(view[pos]);
        }
        scratch.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = n / 2;
        result[pos] = if n % 2 == 1 {
            scratch[mid]
        } else {
            (scratch[mid - 1] + scratch[mid]) * 0.5
        };
    }
    Ok(result)
}

// ───────────────────────────────────────────────────────────────────────────────
// Consistency loss
// ───────────────────────────────────────────────────────────────────────────────

/// Consistency loss between a prediction and a set of reference views.
///
/// The prediction is compared to every view in `references` using the metric
/// specified by `config.loss_type`.  The result is the mean distance across all
/// reference views, further multiplied by `config.consistency_weight`.
pub fn consistency_loss(
    prediction: &[f32],
    references: &ViewBundle,
    config: &ConsistencyConfig,
) -> Result<f32, ConsistencyError> {
    let n = references.num_views();
    if n == 0 {
        return Err(ConsistencyError::TooFewViews);
    }

    let mut total = 0.0_f32;
    for (idx, ref_view) in references.views.iter().enumerate() {
        // Validate size.
        if ref_view.len() != prediction.len() {
            return Err(ConsistencyError::SizeMismatch {
                idx,
                actual: ref_view.len(),
                expected: prediction.len(),
            });
        }

        let dist = match &config.loss_type {
            ConsistencyLossType::L1 => l1_distance(prediction, ref_view)?,
            ConsistencyLossType::L2 => l2_distance(prediction, ref_view)?,
            ConsistencyLossType::Ssim => {
                // Convert from correlation (higher = better) to a distance (lower = better).
                let corr = luminance_correlation(prediction, ref_view)?;
                1.0 - (corr + 1.0) * 0.5
            }
            ConsistencyLossType::Combined {
                l1_weight,
                ssim_weight,
            } => {
                let d_l1 = l1_distance(prediction, ref_view)?;
                let corr = luminance_correlation(prediction, ref_view)?;
                let d_ssim = 1.0 - (corr + 1.0) * 0.5;
                l1_weight * d_l1 + ssim_weight * d_ssim
            }
        };
        total += dist;
    }

    Ok(config.consistency_weight * total / n as f32)
}

// ───────────────────────────────────────────────────────────────────────────────
// Gradient aggregation
// ───────────────────────────────────────────────────────────────────────────────

/// Aggregate per-view gradient arrays by computing their element-wise mean.
///
/// All arrays in `grad_arrays` must have identical lengths.  Returns an error if
/// the input is empty or lengths differ.
pub fn aggregate_view_gradients(grad_arrays: &[Vec<f32>]) -> Result<Vec<f32>, ConsistencyError> {
    let n = grad_arrays.len();
    if n == 0 {
        return Err(ConsistencyError::TooFewViews);
    }
    let len = grad_arrays[0].len();
    for (idx, g) in grad_arrays.iter().enumerate().skip(1) {
        if g.len() != len {
            return Err(ConsistencyError::SizeMismatch {
                idx,
                actual: g.len(),
                expected: len,
            });
        }
    }

    let inv_n = 1.0 / n as f32;
    let mut result = vec![0.0_f32; len];
    for grads in grad_arrays {
        for (r, &g) in result.iter_mut().zip(grads.iter()) {
            *r += g * inv_n;
        }
    }
    Ok(result)
}

// ───────────────────────────────────────────────────────────────────────────────
// Composite loss utilities
// ───────────────────────────────────────────────────────────────────────────────

/// Penalise a content loss with a consistency term.
///
/// `total_loss = content_loss + weight * consistency_loss`
pub fn penalized_loss(content_loss: f32, consistency_loss_val: f32, weight: f32) -> f32 {
    content_loss + weight * consistency_loss_val
}

// ───────────────────────────────────────────────────────────────────────────────
// Anchor view
// ───────────────────────────────────────────────────────────────────────────────

/// Find the "anchor" view — the view whose L2 distance to the mean view is
/// smallest.  Returns the index of that view.
pub fn find_anchor_view(bundle: &ViewBundle) -> Result<usize, ConsistencyError> {
    require_two_views(bundle)?;
    let mean = mean_view(bundle)?;

    let mut best_idx = 0usize;
    let mut best_dist = f32::INFINITY;

    for (idx, view) in bundle.views.iter().enumerate() {
        let d = l2_distance(view.as_slice(), mean.as_slice())?;
        if d < best_dist {
            best_dist = d;
            best_idx = idx;
        }
    }
    Ok(best_idx)
}

// ───────────────────────────────────────────────────────────────────────────────
// Reporting utilities
// ───────────────────────────────────────────────────────────────────────────────

/// Compute the improvement in the consistency score between two snapshots.
///
/// Positive values indicate improvement; negative values indicate regression.
pub fn consistency_improvement(before: &ConsistencyStats, after: &ConsistencyStats) -> f32 {
    after.consistency_score - before.consistency_score
}

/// Format a human-readable summary of consistency statistics.
pub fn format_consistency_report(stats: &ConsistencyStats) -> String {
    let mut s = String::new();
    s.push_str("ConsistencyReport {\n");
    s.push_str(&format!("  view_pairs:       {}\n", stats.num_view_pairs));
    s.push_str(&format!(
        "  mean_l1:          {:.6}\n",
        stats.mean_pairwise_l1
    ));
    s.push_str(&format!(
        "  mean_l2:          {:.6}\n",
        stats.mean_pairwise_l2
    ));
    s.push_str(&format!(
        "  mean_ssim:        {:.6}\n",
        stats.mean_pairwise_ssim
    ));
    s.push_str(&format!(
        "  min_consistency:  {:.6}\n",
        stats.min_consistency
    ));
    s.push_str(&format!(
        "  max_consistency:  {:.6}\n",
        stats.max_consistency
    ));
    s.push_str(&format!(
        "  consistency_score:{:.6}\n",
        stats.consistency_score
    ));
    s.push('}');
    s
}

// ───────────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Helpers ────────────────────────────────────────────────────────────────

    fn make_bundle(width: u32, height: u32, nc: u32, views: Vec<Vec<f32>>) -> ViewBundle {
        let mut b = ViewBundle::new(width, height, nc);
        for v in views {
            b.add_view(v).expect("valid view");
        }
        b
    }

    fn flat(val: f32, len: usize) -> Vec<f32> {
        vec![val; len]
    }

    fn ramp(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| i as f32 / (len as f32 - 1.0).max(1.0))
            .collect()
    }

    // ─── ViewBundle (8 tests) ────────────────────────────────────────────────

    #[test]
    fn test_view_bundle_new() {
        let b = ViewBundle::new(4, 4, 3);
        assert_eq!(b.num_views(), 0);
        assert_eq!(b.width, 4);
        assert_eq!(b.height, 4);
        assert_eq!(b.num_channels, 3);
    }

    #[test]
    fn test_view_bundle_add_view_ok() {
        let mut b = ViewBundle::new(2, 2, 3);
        let pixels = vec![0.0_f32; 12];
        assert!(b.add_view(pixels).is_ok());
        assert_eq!(b.num_views(), 1);
    }

    #[test]
    fn test_view_bundle_add_view_size_mismatch() {
        let mut b = ViewBundle::new(2, 2, 3);
        let pixels = vec![0.0_f32; 10]; // wrong length
        let err = b.add_view(pixels);
        assert!(matches!(err, Err(ConsistencyError::SizeMismatch { .. })));
    }

    #[test]
    fn test_view_bundle_view_access() {
        let mut b = ViewBundle::new(1, 1, 3);
        b.add_view(vec![0.1, 0.2, 0.3]).unwrap();
        let v = b.view(0).expect("view 0");
        assert_eq!(v.len(), 3);
        assert_eq!(b.view(99), None);
    }

    #[test]
    fn test_view_bundle_add_view_u8_ok() {
        let mut b = ViewBundle::new(1, 1, 3);
        let bytes = [255u8, 128, 0];
        assert!(b.add_view_u8(&bytes).is_ok());
        let v = b.view(0).expect("view 0");
        assert!((v[0] - 1.0).abs() < 1e-5);
        assert!((v[1] - 128.0 / 255.0).abs() < 1e-5);
        assert!(v[2].abs() < 1e-5);
    }

    #[test]
    fn test_view_bundle_add_view_u8_mismatch() {
        let mut b = ViewBundle::new(2, 2, 3);
        let bytes = vec![0u8; 5]; // wrong
        assert!(matches!(
            b.add_view_u8(&bytes),
            Err(ConsistencyError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn test_rgb_views_passthrough_for_rgb() {
        let b = make_bundle(1, 1, 3, vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]]);
        let rgb = b.rgb_views();
        assert_eq!(rgb.len(), 2);
        assert_eq!(rgb[0], vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_rgb_views_drops_alpha() {
        let b = make_bundle(
            1,
            1,
            4,
            vec![vec![0.1, 0.2, 0.3, 1.0], vec![0.4, 0.5, 0.6, 0.9]],
        );
        let rgb = b.rgb_views();
        assert_eq!(rgb.len(), 2);
        assert_eq!(rgb[0], vec![0.1, 0.2, 0.3]);
    }

    // ─── l1_distance / l2_distance (6 tests) ────────────────────────────────

    #[test]
    fn test_l1_zeros_identical() {
        let a = vec![0.0_f32; 9];
        assert!((l1_distance(&a, &a).unwrap()).abs() < 1e-7);
    }

    #[test]
    fn test_l2_zeros_identical() {
        let a = vec![0.0_f32; 9];
        assert!((l2_distance(&a, &a).unwrap()).abs() < 1e-7);
    }

    #[test]
    fn test_l1_known_value() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 1.0, 1.0];
        assert!((l1_distance(&a, &b).unwrap() - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_l2_known_value() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 1.0, 1.0];
        assert!((l2_distance(&a, &b).unwrap() - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_l1_size_mismatch() {
        let a = vec![0.0_f32; 3];
        let b = vec![0.0_f32; 4];
        assert!(matches!(
            l1_distance(&a, &b),
            Err(ConsistencyError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn test_l2_size_mismatch() {
        let a = vec![0.0_f32; 3];
        let b = vec![0.0_f32; 4];
        assert!(matches!(
            l2_distance(&a, &b),
            Err(ConsistencyError::SizeMismatch { .. })
        ));
    }

    // ─── luminance_correlation (4 tests) ────────────────────────────────────

    #[test]
    fn test_luminance_correlation_identical() {
        let a = ramp(16);
        let corr = luminance_correlation(&a, &a).unwrap();
        assert!((corr - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_luminance_correlation_all_zeros() {
        let a = vec![0.0_f32; 16];
        let b = vec![0.0_f32; 16];
        // Both constant → treated as identical.
        let corr = luminance_correlation(&a, &a).unwrap();
        assert!((corr - 1.0).abs() < 1e-5);
        let corr2 = luminance_correlation(&a, &b).unwrap();
        assert!((corr2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_luminance_correlation_opposite_signals() {
        // a = [0, 1, 0, 1, ...], b = [1, 0, 1, 0, ...]
        let n = 16;
        let a: Vec<f32> = (0..n).map(|i| (i % 2) as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| 1 - (i % 2)).map(|v| v as f32).collect();
        let corr = luminance_correlation(&a, &b).unwrap();
        assert!(corr < -0.9, "expected strong anti-correlation, got {corr}");
    }

    #[test]
    fn test_luminance_correlation_clamped_range() {
        let a = vec![0.0_f32, 1.0, 0.0];
        let b = vec![0.5_f32, 0.5, 0.5];
        let corr = luminance_correlation(&a, &b).unwrap();
        assert!((-1.0..=1.0).contains(&corr));
    }

    // ─── compute_pairwise_distances (4 tests) ────────────────────────────────

    #[test]
    fn test_pairwise_distances_2views_identical() {
        let b = make_bundle(2, 2, 3, vec![flat(0.5, 12), flat(0.5, 12)]);
        let m = compute_pairwise_distances(&b).unwrap();
        assert_eq!(m.len(), 2);
        assert!(m[0][1].abs() < 1e-7);
        assert!(m[1][0].abs() < 1e-7);
        assert!(m[0][0].abs() < 1e-7);
    }

    #[test]
    fn test_pairwise_distances_2views_different() {
        let b = make_bundle(1, 1, 1, vec![vec![0.0], vec![1.0]]);
        let m = compute_pairwise_distances(&b).unwrap();
        assert!((m[0][1] - 1.0).abs() < 1e-7);
        assert!((m[1][0] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_pairwise_distances_3views_symmetric() {
        let b = make_bundle(1, 1, 1, vec![vec![0.0], vec![0.5], vec![1.0]]);
        let m = compute_pairwise_distances(&b).unwrap();
        assert!((m[0][1] - m[1][0]).abs() < 1e-7);
        assert!((m[0][2] - m[2][0]).abs() < 1e-7);
        assert!((m[1][2] - m[2][1]).abs() < 1e-7);
    }

    #[test]
    fn test_pairwise_distances_too_few_views() {
        let b = ViewBundle::new(1, 1, 1);
        assert!(matches!(
            compute_pairwise_distances(&b),
            Err(ConsistencyError::TooFewViews)
        ));
    }

    // ─── compute_consistency_stats (5 tests) ────────────────────────────────

    #[test]
    fn test_consistency_stats_identical_views() {
        let b = make_bundle(2, 2, 3, vec![flat(0.5, 12), flat(0.5, 12)]);
        let s = compute_consistency_stats(&b).unwrap();
        assert!(s.mean_pairwise_l1 < 1e-7);
        assert!(s.mean_pairwise_l2 < 1e-7);
        assert!((s.consistency_score - 1.0).abs() < 1e-5);
        assert_eq!(s.num_view_pairs, 1);
    }

    #[test]
    fn test_consistency_stats_different_views() {
        let b = make_bundle(1, 1, 1, vec![vec![0.0], vec![1.0]]);
        let s = compute_consistency_stats(&b).unwrap();
        assert!(s.mean_pairwise_l1 > 0.5);
        assert!(s.consistency_score < 1.0);
    }

    #[test]
    fn test_consistency_stats_num_pairs_3_views() {
        let b = make_bundle(1, 1, 1, vec![vec![0.0], vec![0.5], vec![1.0]]);
        let s = compute_consistency_stats(&b).unwrap();
        assert_eq!(s.num_view_pairs, 3);
    }

    #[test]
    fn test_consistency_stats_score_in_range() {
        let b = make_bundle(1, 1, 1, vec![vec![0.0], vec![0.5]]);
        let s = compute_consistency_stats(&b).unwrap();
        assert!(s.consistency_score >= 0.0 && s.consistency_score <= 1.0);
    }

    #[test]
    fn test_consistency_stats_min_leq_max() {
        let b = make_bundle(1, 1, 1, vec![vec![0.0], vec![0.5], vec![1.0]]);
        let s = compute_consistency_stats(&b).unwrap();
        assert!(s.min_consistency <= s.max_consistency);
    }

    // ─── CrossViewCorrelation (4 tests) ─────────────────────────────────────

    #[test]
    fn test_cross_view_correlation_diagonal_is_one() {
        let b = make_bundle(4, 1, 1, vec![ramp(4), ramp(4)]);
        let c = compute_cross_view_correlation(&b).unwrap();
        let v00 = c.get(0, 0).expect("diagonal");
        let v11 = c.get(1, 1).expect("diagonal");
        assert!((v00 - 1.0).abs() < 1e-5);
        assert!((v11 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cross_view_correlation_identical_views() {
        let v = ramp(9);
        let b = make_bundle(3, 3, 1, vec![v.clone(), v.clone()]);
        let c = compute_cross_view_correlation(&b).unwrap();
        let corr = c.get(0, 1).expect("off-diagonal");
        assert!((corr - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cross_view_correlation_mean_off_diagonal() {
        let v = ramp(9);
        let b = make_bundle(3, 3, 1, vec![v.clone(), v.clone()]);
        let c = compute_cross_view_correlation(&b).unwrap();
        let mean = c.mean_off_diagonal();
        assert!((mean - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cross_view_correlation_min_out_of_range() {
        let b = make_bundle(4, 1, 1, vec![ramp(4), ramp(4)]);
        let c = compute_cross_view_correlation(&b).unwrap();
        assert_eq!(c.get(99, 0), None);
        // min_correlation should return a valid number.
        let m = c.min_correlation();
        assert!(m.is_finite());
    }

    // ─── consistency_map (4 tests) ───────────────────────────────────────────

    #[test]
    fn test_consistency_map_uniform_views_zero_std() {
        let b = make_bundle(2, 2, 3, vec![flat(0.5, 12), flat(0.5, 12)]);
        let map = consistency_map(&b).unwrap();
        for &v in &map {
            assert!(v.abs() < 1e-7, "expected zero std, got {v}");
        }
    }

    #[test]
    fn test_consistency_map_varying_views_nonzero() {
        let b = make_bundle(1, 1, 1, vec![vec![0.0], vec![1.0]]);
        let map = consistency_map(&b).unwrap();
        assert!(map[0] > 0.0);
    }

    #[test]
    fn test_consistency_map_in_unit_range() {
        let b = make_bundle(1, 1, 1, vec![vec![0.0], vec![0.5], vec![1.0]]);
        let map = consistency_map(&b).unwrap();
        for &v in &map {
            assert!((0.0..=1.0).contains(&v), "value {v} outside [0,1]");
        }
    }

    #[test]
    fn test_consistency_map_length() {
        let b = make_bundle(3, 4, 3, vec![flat(0.0, 36), flat(1.0, 36)]);
        let map = consistency_map(&b).unwrap();
        assert_eq!(map.len(), 36);
    }

    // ─── weighted_mean_view / mean_view / median_view (6 tests) ─────────────

    #[test]
    fn test_mean_view_known_result() {
        let b = make_bundle(1, 1, 1, vec![vec![0.0], vec![1.0]]);
        let m = mean_view(&b).unwrap();
        assert!((m[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_mean_view_single_nonzero_weight() {
        let b = make_bundle(1, 1, 1, vec![vec![0.2], vec![0.8]]);
        // Only weight view 1.
        let m = weighted_mean_view(&b, &[0.0, 1.0]).unwrap();
        assert!((m[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_mean_view_invalid_weights() {
        let b = make_bundle(1, 1, 1, vec![vec![0.0], vec![1.0]]);
        assert!(matches!(
            weighted_mean_view(&b, &[-1.0, 1.0]),
            Err(ConsistencyError::InvalidWeights)
        ));
        assert!(matches!(
            weighted_mean_view(&b, &[0.0, 0.0]),
            Err(ConsistencyError::InvalidWeights)
        ));
    }

    #[test]
    fn test_median_view_odd_count() {
        let b = make_bundle(1, 1, 1, vec![vec![0.0], vec![0.5], vec![1.0]]);
        let m = median_view(&b).unwrap();
        assert!((m[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_median_view_even_count() {
        let b = make_bundle(1, 1, 1, vec![vec![0.0], vec![1.0]]);
        let m = median_view(&b).unwrap();
        assert!((m[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_mean_view_three_known() {
        let b = make_bundle(1, 1, 1, vec![vec![0.0], vec![0.3], vec![0.6]]);
        let m = mean_view(&b).unwrap();
        assert!((m[0] - 0.3).abs() < 1e-6);
    }

    // ─── consistency_loss (4 tests) ──────────────────────────────────────────

    #[test]
    fn test_consistency_loss_l2_identical() {
        let b = make_bundle(1, 1, 1, vec![vec![0.5], vec![0.5]]);
        let cfg = ConsistencyConfig::default();
        let pred = vec![0.5_f32];
        let loss = consistency_loss(&pred, &b, &cfg).unwrap();
        assert!(loss.abs() < 1e-7);
    }

    #[test]
    fn test_consistency_loss_l1() {
        let b = make_bundle(1, 1, 1, vec![vec![0.0]]);
        let cfg = ConsistencyConfig {
            loss_type: ConsistencyLossType::L1,
            consistency_weight: 1.0,
            ..Default::default()
        };
        let pred = vec![1.0_f32];
        let loss = consistency_loss(&pred, &b, &cfg).unwrap();
        assert!((loss - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_consistency_loss_combined() {
        let b = make_bundle(1, 1, 1, vec![vec![0.0]]);
        let cfg = ConsistencyConfig {
            loss_type: ConsistencyLossType::Combined {
                l1_weight: 0.5,
                ssim_weight: 0.5,
            },
            consistency_weight: 1.0,
            ..Default::default()
        };
        let pred = vec![0.0_f32]; // identical to reference → all distances zero
        let loss = consistency_loss(&pred, &b, &cfg).unwrap();
        assert!(loss.abs() < 1e-6);
    }

    #[test]
    fn test_consistency_loss_ssim_type() {
        let b = make_bundle(4, 1, 1, vec![ramp(4), ramp(4)]);
        let cfg = ConsistencyConfig {
            loss_type: ConsistencyLossType::Ssim,
            consistency_weight: 1.0,
            ..Default::default()
        };
        let pred = ramp(4);
        // Identical prediction → SSIM distance = 0.
        let loss = consistency_loss(&pred, &b, &cfg).unwrap();
        assert!(loss.abs() < 1e-5);
    }

    // ─── aggregate_view_gradients (3 tests) ──────────────────────────────────

    #[test]
    fn test_aggregate_gradients_mean() {
        let grads = vec![vec![0.0_f32, 2.0], vec![2.0_f32, 0.0]];
        let agg = aggregate_view_gradients(&grads).unwrap();
        assert!((agg[0] - 1.0).abs() < 1e-6);
        assert!((agg[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_gradients_single_view() {
        let grads = vec![vec![0.5_f32, 0.7, 0.3]];
        let agg = aggregate_view_gradients(&grads).unwrap();
        assert_eq!(agg.len(), 3);
        assert!((agg[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_gradients_length_mismatch() {
        let grads = vec![vec![0.0_f32; 4], vec![0.0_f32; 3]];
        assert!(matches!(
            aggregate_view_gradients(&grads),
            Err(ConsistencyError::SizeMismatch { .. })
        ));
    }

    // ─── find_anchor_view (3 tests) ──────────────────────────────────────────

    #[test]
    fn test_find_anchor_view_returns_middle() {
        // Mean of three views: 0.0, 0.5, 1.0  →  mean = 0.5 → anchor = view 1
        let b = make_bundle(1, 1, 1, vec![vec![0.0], vec![0.5], vec![1.0]]);
        let anchor = find_anchor_view(&b).unwrap();
        assert_eq!(anchor, 1);
    }

    #[test]
    fn test_find_anchor_view_two_identical_returns_first() {
        let b = make_bundle(1, 1, 1, vec![vec![0.5], vec![0.5]]);
        // Both at same distance to mean; first one should win.
        let anchor = find_anchor_view(&b).unwrap();
        assert!(anchor < 2);
    }

    #[test]
    fn test_find_anchor_view_too_few_views() {
        let b = ViewBundle::new(1, 1, 1);
        assert!(matches!(
            find_anchor_view(&b),
            Err(ConsistencyError::TooFewViews)
        ));
    }

    // ─── penalized_loss, consistency_improvement, format_report (4 tests) ───

    #[test]
    fn test_penalized_loss_basic() {
        let total = penalized_loss(0.5, 0.2, 0.1);
        assert!((total - (0.5 + 0.02)).abs() < 1e-6);
    }

    #[test]
    fn test_consistency_improvement_positive() {
        let before = ConsistencyStats {
            mean_pairwise_l1: 0.4,
            mean_pairwise_l2: 0.4,
            mean_pairwise_ssim: 0.6,
            min_consistency: 0.5,
            max_consistency: 0.7,
            consistency_score: 0.6,
            num_view_pairs: 1,
        };
        let after = ConsistencyStats {
            mean_pairwise_l1: 0.1,
            mean_pairwise_l2: 0.1,
            mean_pairwise_ssim: 0.9,
            min_consistency: 0.8,
            max_consistency: 0.95,
            consistency_score: 0.9,
            num_view_pairs: 1,
        };
        let improvement = consistency_improvement(&before, &after);
        assert!(improvement > 0.0);
        assert!((improvement - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_consistency_improvement_negative() {
        let good = ConsistencyStats {
            mean_pairwise_l1: 0.0,
            mean_pairwise_l2: 0.0,
            mean_pairwise_ssim: 1.0,
            min_consistency: 1.0,
            max_consistency: 1.0,
            consistency_score: 1.0,
            num_view_pairs: 1,
        };
        let bad = ConsistencyStats {
            mean_pairwise_l1: 0.5,
            mean_pairwise_l2: 0.5,
            mean_pairwise_ssim: 0.5,
            min_consistency: 0.5,
            max_consistency: 0.5,
            consistency_score: 0.5,
            num_view_pairs: 1,
        };
        assert!(consistency_improvement(&good, &bad) < 0.0);
    }

    #[test]
    fn test_format_consistency_report_contains_fields() {
        let stats = ConsistencyStats {
            mean_pairwise_l1: 0.1,
            mean_pairwise_l2: 0.05,
            mean_pairwise_ssim: 0.95,
            min_consistency: 0.9,
            max_consistency: 0.99,
            consistency_score: 0.95,
            num_view_pairs: 3,
        };
        let report = format_consistency_report(&stats);
        assert!(report.contains("view_pairs"));
        assert!(report.contains("mean_l1"));
        assert!(report.contains("consistency_score"));
        assert!(report.contains('3'.to_string().as_str()));
    }
}
