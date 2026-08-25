//! The paired source/target [`DomainBatch`] input every adaptation loss reads.

use super::common::DomainAdaptationError;

// ---------------------------------------------------------------------------
// DomainBatch
// ---------------------------------------------------------------------------

/// A batch containing source (synthetic) and target (real) domain features.
///
/// Layout: features are stored in row-major order — element `[i, j]` is at
/// `features[i * d + j]`.
pub struct DomainBatch {
    /// Source domain feature matrix \[N_s × D\], row-major.
    pub source_features: Vec<f32>,
    /// Target domain feature matrix \[N_t × D\], row-major.
    pub target_features: Vec<f32>,
    /// Number of source samples.
    pub n_source: usize,
    /// Number of target samples.
    pub n_target: usize,
    /// Feature dimensionality.
    pub d: usize,
}

impl DomainBatch {
    /// Create a new `DomainBatch`, validating sizes.
    pub fn new(
        src: Vec<f32>,
        tgt: Vec<f32>,
        n_s: usize,
        n_t: usize,
        d: usize,
    ) -> Result<Self, DomainAdaptationError> {
        if d == 0 {
            return Err(DomainAdaptationError::EmptyFeatures);
        }
        let expected_src = n_s * d;
        let expected_tgt = n_t * d;
        if src.len() != expected_src {
            return Err(DomainAdaptationError::DimensionMismatch {
                src: src.len(),
                tgt: expected_src,
            });
        }
        if tgt.len() != expected_tgt {
            return Err(DomainAdaptationError::DimensionMismatch {
                src: tgt.len(),
                tgt: expected_tgt,
            });
        }
        Ok(Self {
            source_features: src,
            target_features: tgt,
            n_source: n_s,
            n_target: n_t,
            d,
        })
    }
}
