//! Latent space interpolation and exploration utilities for diffusion models.
//!
//! This module provides mathematical tools for navigating the latent space of
//! variational autoencoders and diffusion models, including:
//!
//! - **Linear interpolation (LERP)**: Simple weighted blend between two latents.
//! - **Spherical linear interpolation (SLERP)**: Geodesic interpolation on the
//!   hypersphere, preserving angular structure and magnitude.
//! - **Latent arithmetic**: Analogical reasoning, weighted sums, and mean pooling.
//! - **Nearest-neighbor search**: L2-distance-based retrieval from a latent library.
//! - **2D PCA projection**: Power-iteration-based principal component analysis for
//!   visualization and exploration of latent geometry.
//!
//! ## Design Principles
//!
//! All operations are pure Rust with no external tensor library dependencies.
//! Results use `Result<T, InterpError>` — no panics, no unwraps.
//!
//! ## Example
//!
//! ```no_run
//! use oxigaf_diffusion::latent_interp::{LatentVector, InterpolationMethod, InterpolationPath};
//!
//! let a = LatentVector::new(vec![1.0, 0.0]);
//! let b = LatentVector::new(vec![0.0, 1.0]);
//! let path = InterpolationPath::new(a, b, 4, InterpolationMethod::Spherical);
//! let steps = path.all_steps().unwrap();
//! assert_eq!(steps.len(), 5); // num_steps + 1 endpoints
//! ```

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can arise during latent space interpolation operations.
#[derive(Debug)]
pub enum InterpError {
    /// One of the latent vectors has zero length (no data).
    EmptyLatent,
    /// Two latent vectors have incompatible lengths.
    LengthMismatch {
        /// Length of the first operand.
        a: usize,
        /// Length of the second operand.
        b: usize,
    },
    /// Operation requires a non-empty collection.
    EmptyCollection,
    /// Cannot normalize a zero-norm vector (ambiguous direction).
    ZeroNorm,
    /// Not enough latents for the requested operation (e.g., PCA needs ≥ 2).
    InsufficientLatents {
        /// Minimum required.
        required: usize,
        /// Actual count supplied.
        got: usize,
    },
    /// Number of weights does not match number of latents.
    WeightCountMismatch {
        /// Number of latent vectors.
        latents: usize,
        /// Number of weights supplied.
        weights: usize,
    },
    /// Step index is out of bounds for the path.
    IndexOutOfBounds {
        /// Requested index.
        idx: usize,
        /// Length of the path (exclusive upper bound).
        len: usize,
    },
}

impl fmt::Display for InterpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLatent => write!(f, "latent vector is empty"),
            Self::LengthMismatch { a, b } => {
                write!(f, "length mismatch: first has {a} elements, second has {b}")
            }
            Self::EmptyCollection => write!(f, "collection is empty"),
            Self::ZeroNorm => write!(f, "cannot normalize a zero-norm vector"),
            Self::InsufficientLatents { required, got } => {
                write!(f, "insufficient latents: need {required}, got {got}")
            }
            Self::WeightCountMismatch { latents, weights } => {
                write!(
                    f,
                    "weight count mismatch: {latents} latents but {weights} weights"
                )
            }
            Self::IndexOutOfBounds { idx, len } => {
                write!(f, "index {idx} out of bounds for path of length {len}")
            }
        }
    }
}

impl std::error::Error for InterpError {}

// ---------------------------------------------------------------------------
// LatentVector
// ---------------------------------------------------------------------------

/// A dense f32 vector representing a point in latent space.
///
/// Optionally tagged with a `[channels, height, width]` shape for structured
/// latents from convolutional VAEs. The `data` field is always stored flat.
#[derive(Debug, Clone)]
pub struct LatentVector {
    /// Flat coefficient data.
    pub data: Vec<f32>,
    /// Optional spatial shape: `[channels, height, width]`.
    pub shape: Option<[usize; 3]>,
}

impl LatentVector {
    /// Create a new unstructured latent vector.
    pub fn new(data: Vec<f32>) -> Self {
        Self { data, shape: None }
    }

    /// Create a latent vector with an associated `[channels, height, width]` shape.
    pub fn with_shape(data: Vec<f32>, shape: [usize; 3]) -> Self {
        Self {
            data,
            shape: Some(shape),
        }
    }

    /// Number of scalar elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Compute the Euclidean (L2) norm: √(Σ xᵢ²).
    pub fn norm(&self) -> f32 {
        let sum_sq: f32 = self.data.iter().map(|x| x * x).sum();
        sum_sq.sqrt()
    }

    /// Normalize in-place so that `self.norm() ≈ 1.0`.
    ///
    /// No-op when `norm < 1e-8` to avoid division-by-zero.
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n >= 1e-8 {
            let inv = 1.0 / n;
            for x in &mut self.data {
                *x *= inv;
            }
        }
    }

    /// Return a normalized copy of this vector (direction only).
    ///
    /// If the norm is below `1e-8`, returns a copy of the original data unchanged.
    pub fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize();
        copy
    }

    /// Compute the dot product with another latent.
    ///
    /// Returns [`InterpError::LengthMismatch`] when lengths differ.
    pub fn dot(&self, other: &Self) -> Result<f32, InterpError> {
        check_same_length(self, other)?;
        let val = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum();
        Ok(val)
    }

    /// Return a new vector with all coefficients scaled by `factor`.
    pub fn scale(&self, factor: f32) -> Self {
        Self {
            data: self.data.iter().map(|x| x * factor).collect(),
            shape: self.shape,
        }
    }

    /// Element-wise addition with another latent.
    pub fn add(&self, other: &Self) -> Result<Self, InterpError> {
        check_same_length(self, other)?;
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Self {
            data,
            shape: self.shape,
        })
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Self) -> Result<Self, InterpError> {
        check_same_length(self, other)?;
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Ok(Self {
            data,
            shape: self.shape,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate that two latent vectors have the same non-zero length.
fn check_same_length(a: &LatentVector, b: &LatentVector) -> Result<(), InterpError> {
    if a.is_empty() || b.is_empty() {
        return Err(InterpError::EmptyLatent);
    }
    if a.len() != b.len() {
        return Err(InterpError::LengthMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Linear interpolation
// ---------------------------------------------------------------------------

/// Linearly interpolate between two latent vectors.
///
/// `lerp(a, b, t) = a + t * (b − a)`, with `t ∈ [0, 1]`.
///
/// * `t = 0` → `a`
/// * `t = 1` → `b`
pub fn lerp(a: &LatentVector, b: &LatentVector, t: f32) -> Result<LatentVector, InterpError> {
    check_same_length(a, b)?;
    let data = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(ai, bi)| ai + t * (bi - ai))
        .collect();
    Ok(LatentVector {
        data,
        shape: a.shape,
    })
}

// ---------------------------------------------------------------------------
// Spherical linear interpolation
// ---------------------------------------------------------------------------

/// Spherically interpolate between two latent vectors (SLERP).
///
/// Travels along the great circle connecting the *directions* of `a` and
/// `b`, applying the standard slerp coefficients directly to the
/// **original, un-normalised** vectors rather than to their unit-sphere
/// projections.
///
/// ## Algorithm
///
/// 1. Compute norms; return [`InterpError::ZeroNorm`] if either is ~0.
/// 2. Normalize both vectors to unit sphere.
/// 3. `cos_θ = clamp(dot(a_n, b_n), −1, 1)`.
/// 4. If `|cos_θ| > 1 − 1e-6` (nearly collinear), fall back to [`lerp`] on
///    the originals (same magnitude blend).
/// 5. `θ = arccos(cos_θ)`.
/// 6. `slerp = (sin((1−t)θ) / sin θ) · a  +  (sin(tθ) / sin θ) · b`, using
///    the **original** un-normalised vectors `a` and `b` (not `a_n`/`b_n`).
///
///    This exactly reproduces `a` at `t = 0` and `b` at `t = 1`, and for
///    `‖a‖ == ‖b‖` it stays on the sphere of that shared radius throughout
///    (the classic slerp property). For `‖a‖ ≠ ‖b‖`, however, the
///    magnitude in between is **not** the linear blend `(1−t)‖a‖ + t‖b‖` —
///    writing `a = ‖a‖·a_n`, `b = ‖b‖·b_n`, the result is
///    `p·a_n + q·b_n` with `p = ‖a‖·sin((1−t)θ)/sinθ` and
///    `q = ‖b‖·sin(tθ)/sinθ`, whose norm is
///    `sqrt(p² + q² + 2pq·cosθ)` — equal to the linear blend only when
///    `‖a‖ == ‖b‖` (so `p + q` factors out), or when `a_n ⟂ b_n`
///    (`θ = π/2`, so the cross term vanishes) does the *squared* norm
///    happen to be a clean `p² + q²`, which still isn't `((1−t)‖a‖ +
///    t‖b‖)²` in general. If you need the magnitude to be a true linear
///    blend of `‖a‖` and `‖b‖`, rescale the direction explicitly (as
///    `crate::latent_walk::spherical_walk` does) rather than relying on
///    this function.
pub fn slerp(a: &LatentVector, b: &LatentVector, t: f32) -> Result<LatentVector, InterpError> {
    check_same_length(a, b)?;

    let norm_a = a.norm();
    let norm_b = b.norm();
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return Err(InterpError::ZeroNorm);
    }

    let a_n = a.normalized();
    let b_n = b.normalized();

    let cos_theta = a_n.dot(&b_n)?.clamp(-1.0_f32, 1.0_f32);

    // Nearly collinear or anti-collinear: fall back to lerp on originals.
    if cos_theta.abs() > 1.0 - 1e-6 {
        return lerp(a, b, t);
    }

    let theta = cos_theta.acos();
    let sin_theta = theta.sin();
    let inv_sin_theta = 1.0 / sin_theta;

    let coeff_a = ((1.0 - t) * theta).sin() * inv_sin_theta;
    let coeff_b = (t * theta).sin() * inv_sin_theta;

    let data = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(ai, bi)| coeff_a * ai + coeff_b * bi)
        .collect();

    Ok(LatentVector {
        data,
        shape: a.shape,
    })
}

// ---------------------------------------------------------------------------
// Latent arithmetic
// ---------------------------------------------------------------------------

/// Algebraic operations on collections of latent vectors.
pub struct LatentArithmetic;

impl LatentArithmetic {
    /// Analogical reasoning: `result = base + scale × (positive − negative)`.
    ///
    /// Encodes "base is to X as positive is to negative" transformations.
    pub fn analogy(
        base: &LatentVector,
        positive: &LatentVector,
        negative: &LatentVector,
        scale: f32,
    ) -> Result<LatentVector, InterpError> {
        let direction = positive.sub(negative)?;
        let scaled = direction.scale(scale);
        base.add(&scaled)
    }

    /// Compute a weighted sum: `result = Σ wᵢ · latents[i]`.
    ///
    /// Weights need not sum to 1 (no normalisation is applied).
    pub fn weighted_sum(
        latents: &[LatentVector],
        weights: &[f32],
    ) -> Result<LatentVector, InterpError> {
        if latents.is_empty() {
            return Err(InterpError::EmptyCollection);
        }
        if latents.len() != weights.len() {
            return Err(InterpError::WeightCountMismatch {
                latents: latents.len(),
                weights: weights.len(),
            });
        }
        let dim = latents[0].len();
        if dim == 0 {
            return Err(InterpError::EmptyLatent);
        }
        // Verify all same length.
        for lv in latents {
            if lv.len() != dim {
                return Err(InterpError::LengthMismatch {
                    a: dim,
                    b: lv.len(),
                });
            }
        }
        let mut result = vec![0.0_f32; dim];
        for (lv, &w) in latents.iter().zip(weights.iter()) {
            for (r, x) in result.iter_mut().zip(lv.data.iter()) {
                *r += w * x;
            }
        }
        Ok(LatentVector {
            data: result,
            shape: latents[0].shape,
        })
    }

    /// Arithmetic mean of a collection of latent vectors.
    pub fn mean(latents: &[LatentVector]) -> Result<LatentVector, InterpError> {
        if latents.is_empty() {
            return Err(InterpError::EmptyCollection);
        }
        let n = latents.len() as f32;
        let weights = vec![1.0 / n; latents.len()];
        Self::weighted_sum(latents, &weights)
    }

    /// Clamp all coefficients of `v` to `[min, max]`.
    pub fn clamp(v: &LatentVector, min: f32, max: f32) -> LatentVector {
        LatentVector {
            data: v.data.iter().map(|x| x.clamp(min, max)).collect(),
            shape: v.shape,
        }
    }
}

// ---------------------------------------------------------------------------
// Interpolation path
// ---------------------------------------------------------------------------

/// Supported interpolation methods for [`InterpolationPath`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Straight-line blend in Euclidean latent space.
    Linear,
    /// Geodesic blend on the hypersphere (SLERP).
    Spherical,
}

/// A parameterized path through latent space between two endpoints.
///
/// The path is sampled at `num_steps + 1` uniformly spaced values of `t ∈ [0, 1]`.
pub struct InterpolationPath {
    /// Starting point of the path.
    pub start: LatentVector,
    /// Ending point of the path.
    pub end: LatentVector,
    /// Interpolation method to use.
    pub method: InterpolationMethod,
    /// Number of intervals (the number of interior points + 1).
    pub num_steps: usize,
}

impl InterpolationPath {
    /// Construct a new interpolation path.
    pub fn new(
        start: LatentVector,
        end: LatentVector,
        num_steps: usize,
        method: InterpolationMethod,
    ) -> Self {
        Self {
            start,
            end,
            num_steps,
            method,
        }
    }

    /// Return the interpolated latent at step index `i`.
    ///
    /// - `i = 0` → `start`
    /// - `i = num_steps` → `end`
    /// - `i > num_steps` → [`InterpError::IndexOutOfBounds`]
    pub fn step(&self, i: usize) -> Result<LatentVector, InterpError> {
        let total = self.total_points();
        if i >= total {
            return Err(InterpError::IndexOutOfBounds { idx: i, len: total });
        }
        let t = if self.num_steps == 0 {
            0.0
        } else {
            i as f32 / self.num_steps as f32
        };
        match self.method {
            InterpolationMethod::Linear => lerp(&self.start, &self.end, t),
            InterpolationMethod::Spherical => slerp(&self.start, &self.end, t),
        }
    }

    /// Generate all `num_steps + 1` interpolated latents.
    pub fn all_steps(&self) -> Result<Vec<LatentVector>, InterpError> {
        (0..self.total_points()).map(|i| self.step(i)).collect()
    }

    /// Total number of points along the path, including both endpoints.
    pub fn total_points(&self) -> usize {
        self.num_steps + 1
    }
}

// ---------------------------------------------------------------------------
// Nearest-neighbor search
// ---------------------------------------------------------------------------

/// Find the single nearest latent to `query` in `collection` by L2 distance.
///
/// Returns `(index, distance)`, or [`InterpError::EmptyCollection`] when the
/// collection is empty.
pub fn nearest_neighbor(
    query: &LatentVector,
    collection: &[LatentVector],
) -> Result<(usize, f32), InterpError> {
    if collection.is_empty() {
        return Err(InterpError::EmptyCollection);
    }
    let mut best_idx = 0_usize;
    let mut best_dist = f32::INFINITY;

    for (i, candidate) in collection.iter().enumerate() {
        let dist = l2_distance(query, candidate)?;
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    Ok((best_idx, best_dist))
}

/// Find the `k` nearest latents to `query` in `collection` by L2 distance.
///
/// Returns a `Vec<(index, distance)>` sorted by ascending distance.  If
/// `k > collection.len()`, all elements are returned.
pub fn nearest_k_neighbors(
    query: &LatentVector,
    collection: &[LatentVector],
    k: usize,
) -> Result<Vec<(usize, f32)>, InterpError> {
    if collection.is_empty() {
        return Err(InterpError::EmptyCollection);
    }
    let mut distances: Vec<(usize, f32)> = collection
        .iter()
        .enumerate()
        .map(|(i, c)| l2_distance(query, c).map(|d| (i, d)))
        .collect::<Result<Vec<_>, InterpError>>()?;

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let take = k.min(distances.len());
    Ok(distances.into_iter().take(take).collect())
}

/// Compute L2 (Euclidean) distance between two latent vectors.
fn l2_distance(a: &LatentVector, b: &LatentVector) -> Result<f32, InterpError> {
    check_same_length(a, b)?;
    let sum_sq: f32 = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(ai, bi)| (ai - bi) * (ai - bi))
        .sum();
    Ok(sum_sq.sqrt())
}

// ---------------------------------------------------------------------------
// 2D PCA via power iteration
// ---------------------------------------------------------------------------

/// Project `latents` onto their top-2 principal components via power iteration.
///
/// ## Returns
///
/// `(projected, explained_variance)` where:
/// - `projected[i]` is the 2D coordinate of `latents[i]`.
/// - `explained_variance[j]` is the eigenvalue (variance) associated with PC `j`.
///
/// ## Errors
///
/// - [`InterpError::InsufficientLatents`] when `latents.len() < 2`.
/// - [`InterpError::EmptyLatent`] when any latent has zero length.
///
/// ## Algorithm
///
/// 1. Compute the centroid and centre all latents.
/// 2. Represent the centred matrix as an implicit operator.
/// 3. Power-iterate `v ← Aᵀ A v / ‖Aᵀ A v‖` for 20 steps to find PC1.
/// 4. Deflate: subtract the PC1 projection from each row.
/// 5. Repeat for PC2 on the deflated matrix.
/// 6. Project each centred latent onto `[PC1, PC2]`.
pub fn latent_pca_2d(latents: &[LatentVector]) -> Result<(Vec<[f32; 2]>, [f32; 2]), InterpError> {
    let n = latents.len();
    if n < 2 {
        return Err(InterpError::InsufficientLatents {
            required: 2,
            got: n,
        });
    }
    let dim = latents[0].len();
    if dim == 0 {
        return Err(InterpError::EmptyLatent);
    }
    for lv in latents {
        if lv.len() != dim {
            return Err(InterpError::LengthMismatch {
                a: dim,
                b: lv.len(),
            });
        }
    }

    // 1. Mean-centre
    let mut mean = vec![0.0_f32; dim];
    for lv in latents {
        for (m, x) in mean.iter_mut().zip(lv.data.iter()) {
            *m += x;
        }
    }
    let inv_n = 1.0 / n as f32;
    for m in &mut mean {
        *m *= inv_n;
    }
    let mut centred: Vec<Vec<f32>> = latents
        .iter()
        .map(|lv| {
            lv.data
                .iter()
                .zip(mean.iter())
                .map(|(x, m)| x - m)
                .collect()
        })
        .collect();

    // Power iteration to find the top eigenvector of the gram matrix Aᵀ A.
    // We work directly with the rows: Aᵀ A v = Aᵀ (A v).
    let power_iterate = |rows: &[Vec<f32>]| -> Vec<f32> {
        let d = rows[0].len();
        // Start with a constant vector (avoids all-zeros edge case).
        let mut v: Vec<f32> = vec![1.0 / (d as f32).sqrt(); d];
        for _ in 0..20 {
            // A v: project each row onto v → scalar per row
            let av: Vec<f32> = rows
                .iter()
                .map(|row| row.iter().zip(v.iter()).map(|(r, vi)| r * vi).sum())
                .collect();
            // Aᵀ (Av): accumulate back into dim-space
            let mut atav = vec![0.0_f32; d];
            for (row, avi) in rows.iter().zip(av.iter()) {
                for (dst, r) in atav.iter_mut().zip(row.iter()) {
                    *dst += avi * r;
                }
            }
            // Normalise
            let nrm: f32 = atav.iter().map(|x| x * x).sum::<f32>().sqrt();
            if nrm < 1e-12 {
                break;
            }
            let inv = 1.0 / nrm;
            for (vi, ai) in v.iter_mut().zip(atav.iter()) {
                *vi = ai * inv;
            }
        }
        v
    };

    // PC1
    let pc1 = power_iterate(&centred);
    let var1 = variance_along(&centred, &pc1);

    // Deflate: subtract projection onto pc1 from each centred row.
    for row in &mut centred {
        let proj: f32 = row.iter().zip(pc1.iter()).map(|(r, p)| r * p).sum();
        for (r, p) in row.iter_mut().zip(pc1.iter()) {
            *r -= proj * p;
        }
    }

    // PC2
    let pc2 = power_iterate(&centred);
    let var2 = variance_along(&centred, &pc2);

    // Restore original centred data for final projection
    let centred_orig: Vec<Vec<f32>> = latents
        .iter()
        .map(|lv| {
            lv.data
                .iter()
                .zip(mean.iter())
                .map(|(x, m)| x - m)
                .collect()
        })
        .collect();

    let projected: Vec<[f32; 2]> = centred_orig
        .iter()
        .map(|row| {
            let c1: f32 = row.iter().zip(pc1.iter()).map(|(r, p)| r * p).sum();
            let c2: f32 = row.iter().zip(pc2.iter()).map(|(r, p)| r * p).sum();
            [c1, c2]
        })
        .collect();

    Ok((projected, [var1, var2]))
}

/// Compute variance of `rows` projected onto unit vector `pc`.
fn variance_along(rows: &[Vec<f32>], pc: &[f32]) -> f32 {
    let projections: Vec<f32> = rows
        .iter()
        .map(|row| row.iter().zip(pc.iter()).map(|(r, p)| r * p).sum())
        .collect();
    let n = projections.len() as f32;
    if n < 1.0 {
        return 0.0;
    }
    let mean: f32 = projections.iter().sum::<f32>() / n;
    projections
        .iter()
        .map(|p| (p - mean) * (p - mean))
        .sum::<f32>()
        / n
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for latent space interpolation operations.
#[derive(Debug, Clone)]
pub struct LatentInterpolationConfig {
    /// Interpolation method.
    pub method: InterpolationMethod,
    /// Number of intervals between endpoints.
    pub num_steps: usize,
    /// Whether to clamp the output values.
    pub clamp_output: bool,
    /// `(min, max)` clamp range when `clamp_output` is `true`.
    pub output_clamp_range: (f32, f32),
}

impl Default for LatentInterpolationConfig {
    fn default() -> Self {
        Self {
            method: InterpolationMethod::Linear,
            num_steps: 8,
            clamp_output: false,
            output_clamp_range: (-10.0, 10.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // LatentVector basics
    // -----------------------------------------------------------------------

    #[test]
    fn test_latent_vector_norm() {
        let v = LatentVector::new(vec![3.0, 4.0]);
        let n = v.norm();
        assert!((n - 5.0).abs() < 1e-4, "expected 5.0, got {n}");
    }

    #[test]
    fn test_latent_vector_normalize() {
        let mut v = LatentVector::new(vec![3.0, 0.0, 4.0]);
        v.normalize();
        assert!(
            (v.norm() - 1.0).abs() < 1e-5,
            "norm after normalize should be ~1"
        );
        assert!((v.data[0] - 0.6).abs() < 1e-5);
        assert!((v.data[2] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_latent_vector_normalize_zero_norm() {
        // Zero-norm vector: normalize should be a no-op (no panic or error).
        let mut v = LatentVector::new(vec![0.0, 0.0]);
        v.normalize();
        assert!((v.data[0]).abs() < 1e-10);
        assert!((v.data[1]).abs() < 1e-10);
    }

    #[test]
    fn test_latent_vector_dot_product() {
        let a = LatentVector::new(vec![1.0, 2.0, 3.0]);
        let b = LatentVector::new(vec![4.0, 5.0, 6.0]);
        let d = a.dot(&b).unwrap();
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((d - 32.0).abs() < 1e-5, "expected 32, got {d}");
    }

    // -----------------------------------------------------------------------
    // LERP
    // -----------------------------------------------------------------------

    #[test]
    fn test_lerp_endpoints() {
        let a = LatentVector::new(vec![1.0, 2.0, 3.0]);
        let b = LatentVector::new(vec![4.0, 5.0, 6.0]);

        let at_zero = lerp(&a, &b, 0.0).unwrap();
        let at_one = lerp(&a, &b, 1.0).unwrap();

        for (got, exp) in at_zero.data.iter().zip(a.data.iter()) {
            assert!((got - exp).abs() < 1e-6, "t=0 should return a");
        }
        for (got, exp) in at_one.data.iter().zip(b.data.iter()) {
            assert!((got - exp).abs() < 1e-6, "t=1 should return b");
        }
    }

    #[test]
    fn test_lerp_midpoint() {
        let a = LatentVector::new(vec![0.0, 0.0]);
        let b = LatentVector::new(vec![4.0, 8.0]);
        let mid = lerp(&a, &b, 0.5).unwrap();
        assert!((mid.data[0] - 2.0).abs() < 1e-6);
        assert!((mid.data[1] - 4.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // SLERP
    // -----------------------------------------------------------------------

    #[test]
    fn test_slerp_endpoints() {
        let a = LatentVector::new(vec![1.0, 0.0]);
        let b = LatentVector::new(vec![0.0, 1.0]);

        let at_zero = slerp(&a, &b, 0.0).unwrap();
        let at_one = slerp(&a, &b, 1.0).unwrap();

        assert!(
            (at_zero.data[0] - a.data[0]).abs() < 1e-5,
            "t=0 should return a.x"
        );
        assert!(
            (at_zero.data[1] - a.data[1]).abs() < 1e-5,
            "t=0 should return a.y"
        );
        assert!(
            (at_one.data[0] - b.data[0]).abs() < 1e-5,
            "t=1 should return b.x"
        );
        assert!(
            (at_one.data[1] - b.data[1]).abs() < 1e-5,
            "t=1 should return b.y"
        );
    }

    #[test]
    fn test_slerp_midpoint_perpendicular() {
        // Two perpendicular unit vectors at 90°.  The midpoint should be at 45°.
        let a = LatentVector::new(vec![1.0, 0.0]);
        let b = LatentVector::new(vec![0.0, 1.0]);
        let mid = slerp(&a, &b, 0.5).unwrap();

        let expected = (std::f32::consts::PI / 4.0).cos(); // ≈ 0.7071
        assert!(
            (mid.data[0] - expected).abs() < 1e-4,
            "x-component at 45°: expected {expected}, got {}",
            mid.data[0]
        );
        assert!(
            (mid.data[1] - expected).abs() < 1e-4,
            "y-component at 45°: expected {expected}, got {}",
            mid.data[1]
        );
    }

    #[test]
    fn test_slerp_fallback_to_lerp_collinear() {
        // Identical vectors are collinear → slerp falls back to lerp, no error.
        let a = LatentVector::new(vec![1.0, 0.0, 0.0]);
        let b = LatentVector::new(vec![1.0, 0.0, 0.0]);
        let mid = slerp(&a, &b, 0.5).unwrap();
        assert!((mid.data[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_slerp_zero_norm_error() {
        let zero = LatentVector::new(vec![0.0, 0.0]);
        let b = LatentVector::new(vec![1.0, 0.0]);
        let result = slerp(&zero, &b, 0.5);
        assert!(matches!(result, Err(InterpError::ZeroNorm)));
    }

    // -----------------------------------------------------------------------
    // Latent arithmetic
    // -----------------------------------------------------------------------

    #[test]
    fn test_arithmetic_analogy() {
        // king - man + woman ≈ queen  (simplified 1D analogy)
        let king = LatentVector::new(vec![5.0, 5.0]);
        let man = LatentVector::new(vec![3.0, 1.0]);
        let woman = LatentVector::new(vec![1.0, 3.0]);

        let result = LatentArithmetic::analogy(&king, &woman, &man, 1.0).unwrap();
        // base + 1*(positive - negative) = king + (woman - man) = [5+1-3, 5+3-1] = [3, 7]
        assert!((result.data[0] - 3.0).abs() < 1e-5);
        assert!((result.data[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_arithmetic_weighted_sum() {
        let a = LatentVector::new(vec![1.0, 0.0]);
        let b = LatentVector::new(vec![0.0, 1.0]);
        let latents = vec![a, b];
        let weights = [0.3, 0.7];
        let result = LatentArithmetic::weighted_sum(&latents, &weights).unwrap();
        assert!((result.data[0] - 0.3).abs() < 1e-5);
        assert!((result.data[1] - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_arithmetic_mean() {
        let a = LatentVector::new(vec![1.0, 3.0]);
        let b = LatentVector::new(vec![3.0, 1.0]);
        let mean = LatentArithmetic::mean(&[a, b]).unwrap();
        assert!((mean.data[0] - 2.0).abs() < 1e-5);
        assert!((mean.data[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_arithmetic_clamp() {
        let v = LatentVector::new(vec![-20.0, 5.0, 15.0]);
        let clamped = LatentArithmetic::clamp(&v, -10.0, 10.0);
        assert!((clamped.data[0] - (-10.0)).abs() < 1e-6);
        assert!((clamped.data[1] - 5.0).abs() < 1e-6);
        assert!((clamped.data[2] - 10.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // InterpolationPath
    // -----------------------------------------------------------------------

    #[test]
    fn test_path_all_steps_count() {
        let a = LatentVector::new(vec![0.0, 0.0]);
        let b = LatentVector::new(vec![1.0, 1.0]);
        let path = InterpolationPath::new(a, b, 4, InterpolationMethod::Linear);
        let steps = path.all_steps().unwrap();
        assert_eq!(steps.len(), 5, "num_steps=4 should give 5 total points");
    }

    #[test]
    fn test_path_step_index_out_of_bounds() {
        let a = LatentVector::new(vec![0.0]);
        let b = LatentVector::new(vec![1.0]);
        let path = InterpolationPath::new(a, b, 3, InterpolationMethod::Linear);
        // total_points = 4, so index 4 is out of bounds.
        let err = path.step(4);
        assert!(matches!(
            err,
            Err(InterpError::IndexOutOfBounds { idx: 4, len: 4 })
        ));
    }

    // -----------------------------------------------------------------------
    // Nearest-neighbor search
    // -----------------------------------------------------------------------

    #[test]
    fn test_nearest_neighbor_finds_closest() {
        let collection = vec![
            LatentVector::new(vec![0.0, 0.0]),
            LatentVector::new(vec![10.0, 0.0]),
            LatentVector::new(vec![5.0, 0.0]),
        ];
        let query = LatentVector::new(vec![4.9, 0.0]);
        let (idx, dist) = nearest_neighbor(&query, &collection).unwrap();
        assert_eq!(idx, 2, "closest should be index 2 ([5.0, 0.0])");
        assert!(dist < 0.2, "distance should be ~0.1, got {dist}");
    }

    #[test]
    fn test_nearest_neighbor_empty_collection() {
        let query = LatentVector::new(vec![1.0, 2.0]);
        let result = nearest_neighbor(&query, &[]);
        assert!(matches!(result, Err(InterpError::EmptyCollection)));
    }

    #[test]
    fn test_nearest_k_neighbors() {
        let collection = vec![
            LatentVector::new(vec![0.0]),
            LatentVector::new(vec![5.0]),
            LatentVector::new(vec![2.0]),
            LatentVector::new(vec![8.0]),
        ];
        let query = LatentVector::new(vec![3.0]);
        let result = nearest_k_neighbors(&query, &collection, 2).unwrap();
        assert_eq!(result.len(), 2);
        // Distances: 3.0, 2.0, 1.0, 5.0 → sorted: (2, 1.0), (1, 2.0)
        assert_eq!(result[0].0, 2, "nearest should be index 2 (value 2.0)");
        assert_eq!(
            result[1].0, 1,
            "second nearest should be index 1 (value 5.0)"
        );
    }

    // -----------------------------------------------------------------------
    // PCA
    // -----------------------------------------------------------------------

    #[test]
    fn test_pca_2d_basic() {
        // Three latents in 3D space.
        let latents = vec![
            LatentVector::new(vec![1.0, 0.0, 0.0]),
            LatentVector::new(vec![0.0, 1.0, 0.0]),
            LatentVector::new(vec![0.0, 0.0, 1.0]),
        ];
        let (projected, ev) = latent_pca_2d(&latents).unwrap();
        assert_eq!(projected.len(), 3, "should produce one 2D point per latent");
        // Explained variances should be non-negative.
        assert!(ev[0] >= 0.0, "ev[0] should be non-negative");
        assert!(ev[1] >= 0.0, "ev[1] should be non-negative");
    }

    #[test]
    fn test_pca_2d_insufficient_error() {
        let single = vec![LatentVector::new(vec![1.0, 2.0])];
        let result = latent_pca_2d(&single);
        assert!(matches!(
            result,
            Err(InterpError::InsufficientLatents {
                required: 2,
                got: 1
            })
        ));
    }

    // -----------------------------------------------------------------------
    // Error propagation
    // -----------------------------------------------------------------------

    #[test]
    fn test_length_mismatch_error() {
        let a = LatentVector::new(vec![1.0, 2.0]);
        let b = LatentVector::new(vec![1.0, 2.0, 3.0]);
        let result = lerp(&a, &b, 0.5);
        assert!(matches!(
            result,
            Err(InterpError::LengthMismatch { a: 2, b: 3 })
        ));
    }

    #[test]
    fn test_weighted_sum_mismatch_error() {
        let latents = vec![
            LatentVector::new(vec![1.0, 2.0]),
            LatentVector::new(vec![3.0, 4.0]),
        ];
        let result = LatentArithmetic::weighted_sum(&latents, &[0.5]);
        assert!(matches!(
            result,
            Err(InterpError::WeightCountMismatch {
                latents: 2,
                weights: 1
            })
        ));
    }

    #[test]
    fn test_scale_and_add() {
        let a = LatentVector::new(vec![1.0, 2.0]);
        let b = a.scale(3.0);
        assert!((b.data[0] - 3.0).abs() < 1e-6);
        assert!((b.data[1] - 6.0).abs() < 1e-6);

        let c = a.add(&b).unwrap();
        assert!((c.data[0] - 4.0).abs() < 1e-6);
        assert!((c.data[1] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_sub() {
        let a = LatentVector::new(vec![5.0, 3.0]);
        let b = LatentVector::new(vec![2.0, 1.0]);
        let c = a.sub(&b).unwrap();
        assert!((c.data[0] - 3.0).abs() < 1e-6);
        assert!((c.data[1] - 2.0).abs() < 1e-6);
    }
}
