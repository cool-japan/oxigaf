//! CORAL (correlation alignment): feature means, centering, covariance and
//! the Frobenius-norm alignment loss.

use super::batch::DomainBatch;
use super::common::DomainAdaptationError;

// ---------------------------------------------------------------------------
// CORAL: column means
// ---------------------------------------------------------------------------

/// Compute the column (feature) means of a feature matrix \[n × d\].
///
/// If `n * d` does not match `features.len()`, `n` is clamped down to the
/// number of complete rows actually available rather than indexing out of
/// bounds.
pub fn da_feature_mean(features: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut means = vec![0.0f32; d];
    if d == 0 {
        return means;
    }
    let n = n.min(features.len() / d);
    if n == 0 {
        return means;
    }
    for i in 0..n {
        for j in 0..d {
            means[j] += features[i * d + j];
        }
    }
    let inv_n = 1.0 / n as f32;
    for m in &mut means {
        *m *= inv_n;
    }
    means
}

// ---------------------------------------------------------------------------
// CORAL: center features
// ---------------------------------------------------------------------------

/// Center a feature matrix by subtracting column means.
///
/// Returns `(centered_features, means)`.
///
/// If `n * d` does not match `features.len()`, `n` is clamped down to the
/// number of complete rows actually available (matching
/// [`da_feature_mean`]) rather than indexing out of bounds; rows beyond that
/// clamped count are copied through uncentered.
pub fn da_center_features(features: &[f32], n: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    let means = da_feature_mean(features, n, d);
    let n_clamped = features.len().checked_div(d).map_or(0, |rows| n.min(rows));
    let mut centered = features.to_vec();
    for i in 0..n_clamped {
        for j in 0..d {
            centered[i * d + j] -= means[j];
        }
    }
    (centered, means)
}

// ---------------------------------------------------------------------------
// CORAL: covariance
// ---------------------------------------------------------------------------

/// Compute the D×D sample covariance matrix of a feature matrix \[n × d\].
///
/// The matrix must already be **centered** (zero-mean per column).
/// `C = X^T X / (n - 1)` (Bessel-corrected).
///
/// Returns a `d*d` row-major flat vector.  If `n < 2`, returns an all-zero matrix.
///
/// If `n * d` does not match `features.len()`, `n` is clamped down to the
/// number of complete rows actually available rather than indexing out of
/// bounds.
pub fn da_covariance(features: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut cov = vec![0.0f32; d * d];
    if d == 0 {
        return cov;
    }
    let n = n.min(features.len() / d);
    if n < 2 {
        return cov;
    }
    let inv = 1.0 / (n - 1) as f32;
    // C[j,k] = sum_i X[i,j]*X[i,k] / (n-1)
    for i in 0..n {
        for j in 0..d {
            for k in j..d {
                let prod = features[i * d + j] * features[i * d + k] * inv;
                cov[j * d + k] += prod;
                if j != k {
                    cov[k * d + j] += prod;
                }
            }
        }
    }
    cov
}

// ---------------------------------------------------------------------------
// CORAL: Frobenius norm squared of a difference
// ---------------------------------------------------------------------------

/// Compute the squared Frobenius norm of the difference between two matrices:
/// `||A - B||²_F = sum_{i,j}(A_{ij} - B_{ij})²`.
pub fn da_frobenius_sq(a: &[f32], b: &[f32]) -> Result<f32, DomainAdaptationError> {
    if a.len() != b.len() {
        return Err(DomainAdaptationError::DimensionMismatch {
            src: a.len(),
            tgt: b.len(),
        });
    }
    let sq: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let diff = ai - bi;
            diff * diff
        })
        .sum();
    Ok(sq)
}

// ---------------------------------------------------------------------------
// CORAL: loss
// ---------------------------------------------------------------------------

/// Compute the CORAL loss between source and target distributions.
///
/// `L_CORAL = (1 / (4 D²)) * ||C_S - C_T||²_F`
///
/// Both source and target are centered before computing covariances.
pub fn da_coral_loss(batch: &DomainBatch) -> Result<f32, DomainAdaptationError> {
    if batch.n_source == 0 || batch.n_target == 0 {
        return Err(DomainAdaptationError::EmptyFeatures);
    }
    if batch.d == 0 {
        return Err(DomainAdaptationError::EmptyFeatures);
    }

    let (centered_src, _) = da_center_features(&batch.source_features, batch.n_source, batch.d);
    let (centered_tgt, _) = da_center_features(&batch.target_features, batch.n_target, batch.d);

    let cov_src = da_covariance(&centered_src, batch.n_source, batch.d);
    let cov_tgt = da_covariance(&centered_tgt, batch.n_target, batch.d);

    let frob_sq = da_frobenius_sq(&cov_src, &cov_tgt)?;
    let d_sq = (batch.d * batch.d) as f32;
    Ok(frob_sq / (4.0 * d_sq))
}
