//! Expression transfer between FLAME parameter spaces.
//!
//! Maps facial expressions from one FLAME parameter set (source) to another
//! (target), accounting for differences in facial geometry and expression range.
//!
//! # Algorithms
//!
//! - **Direct transfer**: normalize in source space, denormalize in target space.
//! - **Scaled transfer**: direct transfer blended toward target neutral.
//! - **Style transfer**: transfer the expression *delta* (expression minus neutral)
//!   scaled by per-dimension standard deviations.
//! - **PCA-based**: power-iteration PCA of expression samples for compact representation.
//! - **Blend**: interpolate two transferred expressions.
//!
//! # Example
//!
//! ```rust
//! use oxigaf_flame::expression_transfer::{ExpressionSpace, direct_transfer};
//!
//! let space = ExpressionSpace { mean: vec![0.0; 4], std: vec![1.0; 4], dim: 4 };
//! let expr  = vec![0.1, -0.2, 0.3, 0.0];
//! let transferred = direct_transfer(&expr, &space, &space).unwrap();
//! ```

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors for expression transfer operations.
#[derive(Debug, thiserror::Error)]
pub enum ExpressionTransferError {
    /// Expression dimension mismatch.
    #[error("Expression dimension mismatch: source {src}, target {tgt}")]
    DimMismatch { src: usize, tgt: usize },

    /// Empty expression basis or database.
    #[error("Empty expression basis")]
    EmptyBasis,

    /// Transfer ratio outside `[0, 1]`.
    #[error("Transfer ratio {0} out of range [0, 1]")]
    InvalidRatio(f32),

    /// Requested PCA components exceed expression dimensionality.
    #[error("PCA components {n} exceed expression dims {max}")]
    TooManyComponents { n: usize, max: usize },

    /// Zero-variance component during normalization.
    #[error("Singular matrix in normalization (zero variance component {0})")]
    SingularNormalization(usize),

    /// Non-positive weight.
    #[error("Weight must be positive, got {0}")]
    InvalidWeight(f32),
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Minimum standard deviation used to avoid division by zero.
const MIN_STD: f32 = 1e-6_f32;

/// Check that an expression slice has the expected length.
#[inline]
fn check_len(got: usize, expected: usize) -> Result<(), ExpressionTransferError> {
    if got != expected {
        return Err(ExpressionTransferError::DimMismatch {
            src: got,
            tgt: expected,
        });
    }
    Ok(())
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a slice.
#[inline]
fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Xorshift-64 PRNG. Seed is forced nonzero.
#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Produce a pseudo-random unit vector of length `dim` using xorshift64.
fn random_unit_vector(dim: usize, seed: &mut u64) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim)
        .map(|_| {
            let bits = xorshift64(seed);
            // Map to [-1, 1) using bit manipulation
            let f = (bits >> 11) as f32 / (1u64 << 53) as f32; // [0, 1)
            f * 2.0 - 1.0
        })
        .collect();
    let n = norm(&v);
    if n > 1e-12 {
        for x in &mut v {
            *x /= n;
        }
    } else {
        // Degenerate: just use e_0
        if dim > 0 {
            v[0] = 1.0;
        }
    }
    v
}

// ---------------------------------------------------------------------------
// ExpressionSpace
// ---------------------------------------------------------------------------

/// Statistical model of expression space for a single subject.
///
/// Encapsulates the empirical mean and per-dimension standard deviation of
/// expression parameters observed across a pose set, which enables normalisation
/// across subjects with different expression ranges.
#[derive(Debug, Clone)]
pub struct ExpressionSpace {
    /// Expression parameter mean across the pose set.
    pub mean: Vec<f32>,
    /// Per-dimension standard deviation (√variance).
    pub std: Vec<f32>,
    /// Dimensionality of expression space.
    pub dim: usize,
}

impl ExpressionSpace {
    /// Compute an [`ExpressionSpace`] from a set of expression samples.
    ///
    /// # Errors
    ///
    /// Returns [`ExpressionTransferError::EmptyBasis`] when `samples` is empty.
    /// Returns [`ExpressionTransferError::DimMismatch`] if samples have inconsistent lengths.
    pub fn from_samples(samples: &[Vec<f32>]) -> Result<Self, ExpressionTransferError> {
        if samples.is_empty() {
            return Err(ExpressionTransferError::EmptyBasis);
        }

        let dim = samples[0].len();
        // Validate all samples have the same dimension.
        for (_i, s) in samples.iter().enumerate().skip(1) {
            if s.len() != dim {
                return Err(ExpressionTransferError::DimMismatch {
                    src: s.len(),
                    tgt: dim,
                });
            }
        }

        let n = samples.len() as f32;

        // Compute mean.
        let mut mean = vec![0.0_f32; dim];
        for s in samples {
            for (m, &v) in mean.iter_mut().zip(s.iter()) {
                *m += v;
            }
        }
        for m in &mut mean {
            *m /= n;
        }

        // Compute variance, then std.
        let mut variance = vec![0.0_f32; dim];
        for s in samples {
            for (var, (&v, &m)) in variance.iter_mut().zip(s.iter().zip(mean.iter())) {
                let d = v - m;
                *var += d * d;
            }
        }
        let std: Vec<f32> = variance.iter().map(|&v| (v / n).sqrt()).collect();

        Ok(Self { mean, std, dim })
    }

    /// Normalize an expression vector: `(expr - mean) / max(std, 1e-6)`.
    ///
    /// # Errors
    ///
    /// Returns [`ExpressionTransferError::DimMismatch`] when `expr.len() != self.dim`.
    pub fn normalize(&self, expr: &[f32]) -> Result<Vec<f32>, ExpressionTransferError> {
        check_len(expr.len(), self.dim)?;
        let result = expr
            .iter()
            .zip(self.mean.iter())
            .zip(self.std.iter())
            .map(|((&e, &m), &s)| (e - m) / s.max(MIN_STD))
            .collect();
        Ok(result)
    }

    /// Denormalize: `normalized * std + mean`.
    ///
    /// # Errors
    ///
    /// Returns [`ExpressionTransferError::DimMismatch`] when `normalized.len() != self.dim`.
    pub fn denormalize(&self, normalized: &[f32]) -> Result<Vec<f32>, ExpressionTransferError> {
        check_len(normalized.len(), self.dim)?;
        let result = normalized
            .iter()
            .zip(self.mean.iter())
            .zip(self.std.iter())
            .map(|((&n, &m), &s)| n * s.max(MIN_STD) + m)
            .collect();
        Ok(result)
    }

    /// Test whether an expression is "typical" (within 3σ of the mean on every dimension).
    ///
    /// # Errors
    ///
    /// Returns [`ExpressionTransferError::DimMismatch`] when `expr.len() != self.dim`.
    pub fn is_typical(&self, expr: &[f32]) -> Result<bool, ExpressionTransferError> {
        check_len(expr.len(), self.dim)?;
        let typical =
            expr.iter()
                .zip(self.mean.iter())
                .zip(self.std.iter())
                .all(|((&e, &m), &s)| {
                    let sigma = s.max(MIN_STD);
                    (e - m).abs() <= 3.0 * sigma
                });
        Ok(typical)
    }
}

// ---------------------------------------------------------------------------
// Direct / scaled transfer
// ---------------------------------------------------------------------------

/// Transfer expression from source to target space.
///
/// Algorithm:
/// 1. Normalize the source expression using `source_space`.
/// 2. Denormalize the result using `target_space`.
///
/// This maps an expression to an "equivalent" activation in the target subject's
/// expression range.
///
/// # Errors
///
/// Returns [`ExpressionTransferError::DimMismatch`] when dimensionalities differ.
pub fn direct_transfer(
    source_expr: &[f32],
    source_space: &ExpressionSpace,
    target_space: &ExpressionSpace,
) -> Result<Vec<f32>, ExpressionTransferError> {
    if source_space.dim != target_space.dim {
        return Err(ExpressionTransferError::DimMismatch {
            src: source_space.dim,
            tgt: target_space.dim,
        });
    }
    let normalized = source_space.normalize(source_expr)?;
    target_space.denormalize(&normalized)
}

/// Scaled transfer with blending toward target neutral.
///
/// ```text
/// result = target_neutral + ratio * (transferred - target_neutral)
/// ```
///
/// where `target_neutral = target_space.mean` and `transferred` is the result of
/// [`direct_transfer`].
///
/// `ratio = 0.0` → pure neutral; `ratio = 1.0` → full transferred expression.
///
/// # Errors
///
/// Returns [`ExpressionTransferError::InvalidRatio`] when `ratio ∉ [0, 1]`.
/// Returns [`ExpressionTransferError::DimMismatch`] when dimensionalities differ.
pub fn scaled_transfer(
    source_expr: &[f32],
    source_space: &ExpressionSpace,
    target_space: &ExpressionSpace,
    ratio: f32,
) -> Result<Vec<f32>, ExpressionTransferError> {
    if !(0.0..=1.0).contains(&ratio) {
        return Err(ExpressionTransferError::InvalidRatio(ratio));
    }
    let transferred = direct_transfer(source_expr, source_space, target_space)?;
    let result = transferred
        .iter()
        .zip(target_space.mean.iter())
        .map(|(&t, &m)| m + ratio * (t - m))
        .collect();
    Ok(result)
}

// ---------------------------------------------------------------------------
// PCA-based transfer
// ---------------------------------------------------------------------------

/// PCA basis for an expression space, computed via power iteration + deflation.
#[derive(Debug, Clone)]
pub struct ExpressionPcaBasis {
    /// Principal components; outer index = component, inner = expression dimension.
    /// Each row has unit L2 norm.
    pub components: Vec<Vec<f32>>,
    /// Per-component explained variance (un-normalised, in original units²).
    pub explained_variance: Vec<f32>,
    /// Expression mean (subtracted before projection).
    pub mean: Vec<f32>,
    /// Dimensionality of expression space.
    pub dim: usize,
    /// Number of components actually computed.
    pub n_components: usize,
    /// Total variance of the dataset (used for `cumulative_variance`).
    pub(crate) total_variance: f32,
}

impl ExpressionPcaBasis {
    /// Compute PCA basis from expression samples using power iteration + deflation.
    ///
    /// # Parameters
    ///
    /// - `samples`: expression vectors (all must have equal length).
    /// - `n_components`: number of principal components to extract.
    /// - `seed`: random seed for power-iteration initialisation (0 is handled gracefully).
    ///
    /// # Errors
    ///
    /// - [`ExpressionTransferError::EmptyBasis`] — no samples provided.
    /// - [`ExpressionTransferError::TooManyComponents`] — `n_components > dim`.
    pub fn fit(
        samples: &[Vec<f32>],
        n_components: usize,
        seed: u64,
    ) -> Result<Self, ExpressionTransferError> {
        if samples.is_empty() {
            return Err(ExpressionTransferError::EmptyBasis);
        }

        let dim = samples[0].len();
        if n_components > dim {
            return Err(ExpressionTransferError::TooManyComponents {
                n: n_components,
                max: dim,
            });
        }

        let n = samples.len();

        // Compute mean.
        let mut mean = vec![0.0_f32; dim];
        for s in samples {
            for (m, &v) in mean.iter_mut().zip(s.iter()) {
                *m += v;
            }
        }
        for m in &mut mean {
            *m /= n as f32;
        }

        // Build centred data matrix (N × D) stored as Vec<Vec<f32>>.
        let mut x: Vec<Vec<f32>> = samples
            .iter()
            .map(|s| s.iter().zip(mean.iter()).map(|(&v, &m)| v - m).collect())
            .collect();

        // Total Frobenius norm squared / N  (total variance).
        let total_variance: f32 = x
            .iter()
            .flat_map(|row| row.iter())
            .map(|&v| v * v)
            .sum::<f32>()
            / n as f32;

        // Helper: X * v  →  N-vector  (each row dotted with v).
        let x_times_v = |data: &[Vec<f32>], v: &[f32]| -> Vec<f32> {
            data.iter().map(|row| dot(row, v)).collect()
        };

        // Helper: X^T * w  →  D-vector.
        let xt_times_w = |data: &[Vec<f32>], w: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0_f32; dim];
            for (row, &wi) in data.iter().zip(w.iter()) {
                for (o, &rj) in out.iter_mut().zip(row.iter()) {
                    *o += rj * wi;
                }
            }
            out
        };

        // Force seed nonzero.
        let mut rng_state: u64 = seed.max(1);

        let mut components = Vec::with_capacity(n_components);
        let mut explained_variance = Vec::with_capacity(n_components);

        for _ in 0..n_components {
            // Initialise with a random unit vector.
            let mut v = random_unit_vector(dim, &mut rng_state);

            // Power iteration (30 steps).
            for _ in 0..30 {
                let xv = x_times_v(&x, &v);
                let u = xt_times_w(&x, &xv);
                let n_u = norm(&u);
                if n_u < 1e-12 {
                    // Data (after deflation) is zero in this direction.
                    break;
                }
                v = u.into_iter().map(|val| val / n_u).collect();
            }

            // Compute explained variance: ||X * v||² / N.
            let xv = x_times_v(&x, &v);
            let ev: f32 = xv.iter().map(|&val| val * val).sum::<f32>() / n as f32;
            explained_variance.push(ev);

            // Deflation: X_new = X - (X * v) * v^T
            // i.e. for each row i: x[i] -= dot(x[i], v) * v
            for row in &mut x {
                let proj = dot(row, &v);
                for (rj, &vj) in row.iter_mut().zip(v.iter()) {
                    *rj -= proj * vj;
                }
            }

            components.push(v);
        }

        Ok(Self {
            components,
            explained_variance,
            mean,
            dim,
            n_components,
            total_variance,
        })
    }

    /// Project an expression vector into PCA space: `(expr - mean) @ components^T`.
    ///
    /// Returns a vector of length `n_components`.
    ///
    /// # Errors
    ///
    /// Returns [`ExpressionTransferError::DimMismatch`] when `expr.len() != self.dim`.
    pub fn project(&self, expr: &[f32]) -> Result<Vec<f32>, ExpressionTransferError> {
        check_len(expr.len(), self.dim)?;
        let centered: Vec<f32> = expr
            .iter()
            .zip(self.mean.iter())
            .map(|(&e, &m)| e - m)
            .collect();
        let coords = self
            .components
            .iter()
            .map(|comp| dot(&centered, comp))
            .collect();
        Ok(coords)
    }

    /// Reconstruct an expression from PCA coordinates: `coords @ components + mean`.
    ///
    /// `coords` must have length `n_components`.
    ///
    /// # Errors
    ///
    /// Returns [`ExpressionTransferError::DimMismatch`] when `coords.len() != self.n_components`.
    pub fn reconstruct(&self, coords: &[f32]) -> Result<Vec<f32>, ExpressionTransferError> {
        check_len(coords.len(), self.n_components)?;
        let mut result = self.mean.clone();
        for (comp, &c) in self.components.iter().zip(coords.iter()) {
            for (r, &cj) in result.iter_mut().zip(comp.iter()) {
                *r += c * cj;
            }
        }
        Ok(result)
    }

    /// Reconstruction error: `‖original − reconstruct(project(original))‖²`.
    ///
    /// # Errors
    ///
    /// Returns [`ExpressionTransferError::DimMismatch`] when `expr.len() != self.dim`.
    pub fn reconstruction_error(&self, expr: &[f32]) -> Result<f32, ExpressionTransferError> {
        let coords = self.project(expr)?;
        let reconstructed = self.reconstruct(&coords)?;
        let error = expr
            .iter()
            .zip(reconstructed.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        Ok(error)
    }

    /// Cumulative explained variance fraction for the first `n` components.
    ///
    /// Returns 0.0 when `total_variance` is zero or `n == 0`.
    /// Clamps `n` to `n_components`.
    #[must_use]
    pub fn cumulative_variance(&self, n: usize) -> f32 {
        if self.total_variance < 1e-30 || n == 0 {
            return 0.0;
        }
        let k = n.min(self.n_components);
        let sum: f32 = self.explained_variance[..k].iter().sum();
        sum / self.total_variance
    }
}

// ---------------------------------------------------------------------------
// Style transfer
// ---------------------------------------------------------------------------

/// Transfer expression style while preserving content.
///
/// "Style" = direction of expression change relative to neutral.
/// "Content" = the neutral face.
///
/// Given `source_neutral`, `source_expr`, and `target_neutral`:
/// 1. Compute delta = `source_expr − source_neutral` (expression delta in source space).
/// 2. Scale delta component-wise by `target_space.std / max(source_space.std, 1e-6)`.
/// 3. Result = `target_neutral + strength * target_delta`.
///
/// When `strength = 0` the result is identical to `target_neutral`.
///
/// # Errors
///
/// - [`ExpressionTransferError::InvalidRatio`] when `strength ∉ [0, 1]`.
/// - [`ExpressionTransferError::DimMismatch`] when dimensionalities differ.
pub fn style_transfer(
    source_neutral: &[f32],
    source_expr: &[f32],
    target_neutral: &[f32],
    source_space: &ExpressionSpace,
    target_space: &ExpressionSpace,
    strength: f32,
) -> Result<Vec<f32>, ExpressionTransferError> {
    if !(0.0..=1.0).contains(&strength) {
        return Err(ExpressionTransferError::InvalidRatio(strength));
    }

    let dim = source_space.dim;
    if target_space.dim != dim {
        return Err(ExpressionTransferError::DimMismatch {
            src: dim,
            tgt: target_space.dim,
        });
    }
    check_len(source_neutral.len(), dim)?;
    check_len(source_expr.len(), dim)?;
    check_len(target_neutral.len(), dim)?;

    // Expression delta in source parameter space.
    let delta: Vec<f32> = source_expr
        .iter()
        .zip(source_neutral.iter())
        .map(|(&e, &n)| e - n)
        .collect();

    // Normalise delta by source std, then scale by target std.
    let target_delta: Vec<f32> = delta
        .iter()
        .zip(source_space.std.iter())
        .zip(target_space.std.iter())
        .map(|((&d, &ss), &ts)| d / ss.max(MIN_STD) * ts.max(MIN_STD))
        .collect();

    // Blend result.
    let result = target_neutral
        .iter()
        .zip(target_delta.iter())
        .map(|(&tn, &td)| tn + strength * td)
        .collect();

    Ok(result)
}

// ---------------------------------------------------------------------------
// Expression blending with transfer
// ---------------------------------------------------------------------------

/// Interpolate expressions from two sources, each transferred into a common target space.
///
/// The two source expressions are first transferred independently into `target_space`,
/// then linearly interpolated:
///
/// ```text
/// result = (1 − blend_ratio) * transfer(A→target) + blend_ratio * transfer(B→target)
/// ```
///
/// `blend_ratio = 0.0` → pure A; `blend_ratio = 1.0` → pure B.
///
/// # Errors
///
/// - [`ExpressionTransferError::InvalidRatio`] when `blend_ratio ∉ [0, 1]`.
/// - [`ExpressionTransferError::DimMismatch`] when dimensionalities differ.
pub fn blend_transferred(
    source_a: &[f32],
    space_a: &ExpressionSpace,
    source_b: &[f32],
    space_b: &ExpressionSpace,
    target_space: &ExpressionSpace,
    blend_ratio: f32,
) -> Result<Vec<f32>, ExpressionTransferError> {
    if !(0.0..=1.0).contains(&blend_ratio) {
        return Err(ExpressionTransferError::InvalidRatio(blend_ratio));
    }

    let transferred_a = direct_transfer(source_a, space_a, target_space)?;
    let transferred_b = direct_transfer(source_b, space_b, target_space)?;

    let result = transferred_a
        .iter()
        .zip(transferred_b.iter())
        .map(|(&a, &b)| (1.0 - blend_ratio) * a + blend_ratio * b)
        .collect();

    Ok(result)
}

// ---------------------------------------------------------------------------
// Expression analysis
// ---------------------------------------------------------------------------

/// Compute similarity (L2 distance normalised by dimension) between two expressions.
///
/// `similarity = sqrt(Σ(a − b)² / n)`
///
/// Returns `0.0` for identical expressions.
///
/// # Errors
///
/// Returns [`ExpressionTransferError::DimMismatch`] when `expr_a.len() != expr_b.len()`.
/// Returns [`ExpressionTransferError::EmptyBasis`] when both slices are empty.
pub fn expression_similarity(
    expr_a: &[f32],
    expr_b: &[f32],
) -> Result<f32, ExpressionTransferError> {
    if expr_a.len() != expr_b.len() {
        return Err(ExpressionTransferError::DimMismatch {
            src: expr_a.len(),
            tgt: expr_b.len(),
        });
    }
    if expr_a.is_empty() {
        return Err(ExpressionTransferError::EmptyBasis);
    }
    let n = expr_a.len() as f32;
    let sum: f32 = expr_a
        .iter()
        .zip(expr_b.iter())
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum();
    Ok((sum / n).sqrt())
}

/// Compute the "intensity" of an expression relative to neutral: `‖expr − neutral‖₂`.
///
/// When the slice lengths differ, the shorter length is used (no panic).
#[must_use]
pub fn expression_intensity(expr: &[f32], neutral: &[f32]) -> f32 {
    let n = expr.len().min(neutral.len());
    expr[..n]
        .iter()
        .zip(neutral[..n].iter())
        .map(|(&e, &neu)| (e - neu) * (e - neu))
        .sum::<f32>()
        .sqrt()
}

/// Classify expression intensity into a simple category.
///
/// | Intensity    | Category   |
/// |-------------|------------|
/// | < 0.1       | "neutral"  |
/// | 0.1 – 0.3   | "mild"     |
/// | 0.3 – 0.7   | "moderate" |
/// | ≥ 0.7       | "intense"  |
///
/// The confidence is the distance from the boundary (clamped to [0, 1]).
///
/// Returns `(category, confidence)`.
#[must_use]
pub fn classify_intensity(expr: &[f32], neutral: &[f32]) -> (&'static str, f32) {
    let intensity = expression_intensity(expr, neutral);

    if intensity < 0.1 {
        let confidence = (0.1 - intensity) / 0.1;
        ("neutral", confidence.clamp(0.0, 1.0))
    } else if intensity < 0.3 {
        let mid = 0.2_f32;
        let half_width = 0.1_f32;
        let confidence = 1.0 - ((intensity - mid) / half_width).abs();
        ("mild", confidence.clamp(0.0, 1.0))
    } else if intensity < 0.7 {
        let mid = 0.5_f32;
        let half_width = 0.2_f32;
        let confidence = 1.0 - ((intensity - mid) / half_width).abs();
        ("moderate", confidence.clamp(0.0, 1.0))
    } else {
        let confidence = (intensity - 0.7) / 0.7;
        ("intense", confidence.clamp(0.0, 1.0))
    }
}

/// Find the nearest expression in a labelled database.
///
/// Uses `expression_similarity` (normalised L2 distance) as the distance metric.
///
/// # Returns
///
/// `(index, label, distance)` for the closest database entry.
///
/// # Errors
///
/// - [`ExpressionTransferError::EmptyBasis`] when `database` is empty.
/// - [`ExpressionTransferError::DimMismatch`] when any database entry has a different length from `query`.
pub fn find_nearest(
    query: &[f32],
    database: &[Vec<f32>],
    labels: &[String],
) -> Result<(usize, String, f32), ExpressionTransferError> {
    if database.is_empty() {
        return Err(ExpressionTransferError::EmptyBasis);
    }

    let mut best_idx = 0usize;
    let mut best_dist = f32::MAX;

    for (i, entry) in database.iter().enumerate() {
        let dist = expression_similarity(query, entry)?;
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }

    let label = labels
        .get(best_idx)
        .cloned()
        .unwrap_or_else(|| best_idx.to_string());

    Ok((best_idx, label, best_dist))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn zero_samples(n: usize, dim: usize) -> Vec<Vec<f32>> {
        vec![vec![0.0_f32; dim]; n]
    }

    fn identity_space(dim: usize) -> ExpressionSpace {
        ExpressionSpace {
            mean: vec![0.0; dim],
            std: vec![1.0; dim],
            dim,
        }
    }

    fn unit_range_space(dim: usize) -> ExpressionSpace {
        ExpressionSpace {
            mean: vec![0.0; dim],
            std: vec![0.5; dim],
            dim,
        }
    }

    /// Generate samples with variance in every dimension.
    /// sample[i][j] = sin(i * (j+1) + j) — gives rank-dim data deterministically.
    fn varied_samples(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        ((i * (j + 1) + j) as f32 * std::f32::consts::FRAC_PI_4).sin()
                            * (j + 1) as f32
                            * 0.5
                    })
                    .collect()
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // ExpressionSpace tests
    // ------------------------------------------------------------------

    // Test 1: from_samples with identical samples → std = 0
    #[test]
    fn test_from_samples_identical_std_zero() {
        let samples = zero_samples(5, 4);
        let space = ExpressionSpace::from_samples(&samples).expect("should succeed");
        assert_eq!(space.dim, 4);
        for &s in &space.std {
            assert!((s - 0.0).abs() < 1e-7, "std should be zero, got {s}");
        }
        for &m in &space.mean {
            assert!((m - 0.0).abs() < 1e-7);
        }
    }

    // Test 2: from_samples empty → Err
    #[test]
    fn test_from_samples_empty() {
        let result = ExpressionSpace::from_samples(&[]);
        assert!(
            matches!(result, Err(ExpressionTransferError::EmptyBasis)),
            "expected EmptyBasis"
        );
    }

    // Test 3: normalize identity space: output = input
    #[test]
    fn test_normalize_identity_space() {
        let space = identity_space(4);
        let expr = vec![0.1, -0.2, 0.3, 0.0];
        let normalized = space.normalize(&expr).expect("normalize should succeed");
        for (n, e) in normalized.iter().zip(expr.iter()) {
            assert!((n - e).abs() < 1e-6, "expected {e}, got {n}");
        }
    }

    // Test 4: normalize dim mismatch → DimMismatch
    #[test]
    fn test_normalize_dim_mismatch() {
        let space = identity_space(4);
        let expr = vec![0.1, 0.2]; // too short
        let result = space.normalize(&expr);
        assert!(
            matches!(result, Err(ExpressionTransferError::DimMismatch { .. })),
            "expected DimMismatch"
        );
    }

    // Test 5: denormalize roundtrip with normalize (non-zero std)
    #[test]
    fn test_denormalize_roundtrip() {
        let samples: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                vec![
                    i as f32 * 0.1,
                    i as f32 * 0.2,
                    i as f32 * 0.05,
                    i as f32 * 0.3,
                ]
            })
            .collect();
        let space = ExpressionSpace::from_samples(&samples).expect("from_samples");
        let expr = vec![0.25, 0.7, 0.1, 0.55];
        let normalized = space.normalize(&expr).expect("normalize");
        let recovered = space.denormalize(&normalized).expect("denormalize");
        for (&r, &e) in recovered.iter().zip(expr.iter()) {
            assert!(
                (r - e).abs() < 1e-5,
                "roundtrip mismatch: expected {e}, got {r}"
            );
        }
    }

    // Test 6: is_typical zero expression in identity space → true
    #[test]
    fn test_is_typical_zero_in_identity() {
        let space = identity_space(4);
        let expr = vec![0.0_f32; 4];
        let typical = space.is_typical(&expr).expect("is_typical");
        assert!(typical, "zero should be typical");
    }

    // Test 7: is_typical large value → false
    #[test]
    fn test_is_typical_large_value() {
        let space = identity_space(4);
        let expr = vec![100.0_f32, 0.0, 0.0, 0.0]; // 100σ away
        let typical = space.is_typical(&expr).expect("is_typical");
        assert!(!typical, "large value should not be typical");
    }

    // ------------------------------------------------------------------
    // Direct / scaled transfer tests
    // ------------------------------------------------------------------

    // Test 8: direct_transfer with identical spaces: output = input
    #[test]
    fn test_direct_transfer_identical_spaces() {
        let space = identity_space(4);
        let expr = vec![0.1, -0.2, 0.3, 0.4];
        let result = direct_transfer(&expr, &space, &space).expect("direct_transfer");
        for (&r, &e) in result.iter().zip(expr.iter()) {
            assert!((r - e).abs() < 1e-5, "expected {e}, got {r}");
        }
    }

    // Test 9: direct_transfer dim mismatch → DimMismatch
    #[test]
    fn test_direct_transfer_dim_mismatch() {
        let src_space = identity_space(4);
        let tgt_space = identity_space(6);
        let expr = vec![0.1; 4];
        let result = direct_transfer(&expr, &src_space, &tgt_space);
        assert!(
            matches!(result, Err(ExpressionTransferError::DimMismatch { .. })),
            "expected DimMismatch"
        );
    }

    // Test 10: scaled_transfer ratio=0.0 → target neutral
    #[test]
    fn test_scaled_transfer_ratio_zero() {
        let space = identity_space(4);
        let expr = vec![0.5, -0.5, 0.3, 0.1];
        let result = scaled_transfer(&expr, &space, &space, 0.0).expect("scaled_transfer");
        // At ratio 0, result should equal target_space.mean = [0.0, ...]
        for &r in &result {
            assert!((r - 0.0).abs() < 1e-6, "expected 0.0 at ratio=0, got {r}");
        }
    }

    // Test 11: scaled_transfer ratio=1.0 → direct_transfer result
    #[test]
    fn test_scaled_transfer_ratio_one() {
        let space = unit_range_space(4);
        let space2 = identity_space(4);
        let expr = vec![0.1, -0.2, 0.3, 0.0];
        let direct = direct_transfer(&expr, &space, &space2).expect("direct_transfer");
        let scaled = scaled_transfer(&expr, &space, &space2, 1.0).expect("scaled_transfer");
        for (&s, &d) in scaled.iter().zip(direct.iter()) {
            assert!((s - d).abs() < 1e-5, "scaled@1 != direct: {s} vs {d}");
        }
    }

    // ------------------------------------------------------------------
    // PCA basis tests
    // ------------------------------------------------------------------

    // Test 12: fit with constant samples → 0 variance components
    #[test]
    fn test_pca_fit_constant_samples() {
        let samples = zero_samples(10, 4);
        let basis = ExpressionPcaBasis::fit(&samples, 2, 42).expect("fit");
        for &ev in &basis.explained_variance {
            assert!(ev < 1e-10, "expected zero variance, got {ev}");
        }
    }

    // Test 13: project then reconstruct roundtrip for n_components = dim
    #[test]
    fn test_pca_project_reconstruct_roundtrip() {
        let dim = 4;
        let samples = varied_samples(20, dim);
        let basis = ExpressionPcaBasis::fit(&samples, dim, 7).expect("fit");
        let expr = vec![0.15, 0.35, 0.55, 0.75];
        let coords = basis.project(&expr).expect("project");
        let reconstructed = basis.reconstruct(&coords).expect("reconstruct");
        let error = basis.reconstruction_error(&expr).expect("recon_error");
        // With n_components = dim, reconstruction should be near perfect.
        assert!(error < 1e-4, "reconstruction error too large: {error}");
        for (&r, &e) in reconstructed.iter().zip(expr.iter()) {
            assert!(
                (r - e).abs() < 0.01,
                "roundtrip mismatch: expected {e}, got {r}"
            );
        }
    }

    // Test 14: cumulative_variance all components → ≈ 1.0
    #[test]
    fn test_pca_cumulative_variance_full() {
        let dim = 4;
        let samples = varied_samples(20, dim);
        let basis = ExpressionPcaBasis::fit(&samples, dim, 13).expect("fit");
        let cumvar = basis.cumulative_variance(dim);
        assert!(
            (cumvar - 1.0).abs() < 0.05,
            "cumulative variance for all components should be ~1.0, got {cumvar}"
        );
    }

    // ------------------------------------------------------------------
    // Style transfer tests
    // ------------------------------------------------------------------

    // Test 15: style_transfer neutral-to-neutral → target neutral
    #[test]
    fn test_style_transfer_neutral_to_neutral() {
        let space = identity_space(4);
        let neutral = vec![0.0_f32; 4];
        let target_neutral = vec![0.1_f32, 0.2, -0.1, 0.0];
        let result = style_transfer(&neutral, &neutral, &target_neutral, &space, &space, 1.0)
            .expect("style_transfer");
        // delta = neutral - neutral = 0, so result = target_neutral + strength * 0 = target_neutral
        for (&r, &tn) in result.iter().zip(target_neutral.iter()) {
            assert!((r - tn).abs() < 1e-6, "expected {tn}, got {r}");
        }
    }

    // Test 16: style_transfer strength=0 → target neutral
    #[test]
    fn test_style_transfer_strength_zero() {
        let space = identity_space(4);
        let source_neutral = vec![0.0_f32; 4];
        let source_expr = vec![0.3_f32, -0.2, 0.5, 0.1];
        let target_neutral = vec![0.1_f32, 0.2, -0.1, 0.0];
        let result = style_transfer(
            &source_neutral,
            &source_expr,
            &target_neutral,
            &space,
            &space,
            0.0,
        )
        .expect("style_transfer");
        for (&r, &tn) in result.iter().zip(target_neutral.iter()) {
            assert!(
                (r - tn).abs() < 1e-6,
                "at strength=0, expected {tn}, got {r}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Blend transferred tests
    // ------------------------------------------------------------------

    // Test 17: blend_transferred ratio=0 → transfer from A
    #[test]
    fn test_blend_transferred_ratio_zero() {
        let space_a = identity_space(4);
        let space_b = unit_range_space(4);
        let target = identity_space(4);
        let expr_a = vec![0.1, 0.2, 0.3, 0.4];
        let expr_b = vec![0.5, 0.6, 0.7, 0.8];
        let blended = blend_transferred(&expr_a, &space_a, &expr_b, &space_b, &target, 0.0)
            .expect("blend_transferred");
        let expected = direct_transfer(&expr_a, &space_a, &target).expect("direct_transfer");
        for (&b, &e) in blended.iter().zip(expected.iter()) {
            assert!((b - e).abs() < 1e-5, "blend@0 != transfer_A: {b} vs {e}");
        }
    }

    // Test 18: blend_transferred ratio=1 → transfer from B
    #[test]
    fn test_blend_transferred_ratio_one() {
        let space_a = identity_space(4);
        let space_b = unit_range_space(4);
        let target = identity_space(4);
        let expr_a = vec![0.1, 0.2, 0.3, 0.4];
        let expr_b = vec![0.5, 0.6, 0.7, 0.8];
        let blended = blend_transferred(&expr_a, &space_a, &expr_b, &space_b, &target, 1.0)
            .expect("blend_transferred");
        let expected = direct_transfer(&expr_b, &space_b, &target).expect("direct_transfer");
        for (&b, &e) in blended.iter().zip(expected.iter()) {
            assert!((b - e).abs() < 1e-5, "blend@1 != transfer_B: {b} vs {e}");
        }
    }

    // ------------------------------------------------------------------
    // Analysis function tests
    // ------------------------------------------------------------------

    // Test 19: expression_similarity same expression → 0.0
    #[test]
    fn test_expression_similarity_same() {
        let expr = vec![0.1, 0.2, 0.3, 0.4];
        let sim = expression_similarity(&expr, &expr).expect("similarity");
        assert!(
            (sim - 0.0).abs() < 1e-7,
            "same expressions should have distance 0, got {sim}"
        );
    }

    // Test 20: expression_intensity neutral vs neutral → 0.0
    #[test]
    fn test_expression_intensity_neutral() {
        let neutral = vec![0.0_f32; 4];
        let intensity = expression_intensity(&neutral, &neutral);
        assert!(
            (intensity - 0.0).abs() < 1e-7,
            "neutral intensity should be 0, got {intensity}"
        );
    }

    // Test 21: classify_intensity neutral → ("neutral", _)
    #[test]
    fn test_classify_intensity_neutral() {
        let neutral = vec![0.0_f32; 4];
        let tiny_expr = vec![0.01_f32, 0.01, 0.01, 0.01]; // intensity << 0.1
        let (category, _confidence) = classify_intensity(&tiny_expr, &neutral);
        assert_eq!(category, "neutral", "expected neutral category");
    }

    // Test 22: find_nearest single database entry → returns it
    #[test]
    fn test_find_nearest_single_entry() {
        let query = vec![0.1, 0.2, 0.3];
        let database = vec![vec![0.1, 0.2, 0.3]];
        let labels = vec!["test".to_string()];
        let (idx, label, dist) = find_nearest(&query, &database, &labels).expect("find_nearest");
        assert_eq!(idx, 0);
        assert_eq!(label, "test");
        assert!(
            (dist - 0.0).abs() < 1e-6,
            "distance should be 0, got {dist}"
        );
    }

    // Test 23: find_nearest empty database → Err
    #[test]
    fn test_find_nearest_empty_database() {
        let query = vec![0.1, 0.2, 0.3];
        let result = find_nearest(&query, &[], &[]);
        assert!(
            matches!(result, Err(ExpressionTransferError::EmptyBasis)),
            "expected EmptyBasis for empty database"
        );
    }
}
