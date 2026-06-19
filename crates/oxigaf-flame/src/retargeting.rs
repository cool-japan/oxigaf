//! Expression retargeting between FLAME identities.
//!
//! Transfers facial expressions from one FLAME identity to another, compensating
//! for shape-dependent expression differences so the same "feeling" is conveyed
//! on the target identity regardless of differences in shape parameters (β).
//!
//! # Overview
//!
//! When two people have different face shapes, the same expression coefficient
//! vector (ψ) produces geometrically different displacements. Retargeting rescales
//! expression coefficients so the magnitude of the resulting mesh deformation is
//! comparable across identities.
//!
//! # Example
//!
//! ```rust
//! use oxigaf_flame::retargeting::{
//!     RetargetingConfig, ExpressionRetargeter, compute_expression_scale_factors,
//! };
//!
//! let source_scales = vec![1.0_f32; 10];
//! let target_scales = vec![2.0_f32; 10];
//! let scale_factors = compute_expression_scale_factors(&source_scales, &target_scales).unwrap();
//!
//! let config = RetargetingConfig::default();
//! let retargeter = ExpressionRetargeter::new(scale_factors, config).unwrap();
//!
//! let source_expr = vec![0.5_f32; 10];
//! let retargeted = retargeter.retarget(&source_expr).unwrap();
//! ```

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during expression retargeting.
#[derive(Debug, thiserror::Error)]
pub enum RetargetingError {
    /// Input slice or sequence was empty when non-empty was required.
    #[error("Empty parameters provided")]
    EmptyParams,

    /// Array length or dimension did not match the expected value.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// A configuration field had an invalid value.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Not enough expression data to perform the requested operation.
    #[error("Insufficient expression data: {0}")]
    InsufficientExpressionData(String),
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration parameters for expression retargeting.
#[derive(Debug, Clone)]
pub struct RetargetingConfig {
    /// Clip expression coefficients to this maximum absolute value before retargeting.
    ///
    /// Default: `3.0`
    pub expr_clip: f32,

    /// Multiplicative scale applied to all transferred expression coefficients.
    ///
    /// `1.0` preserves the intensity calculated from shape-based scale factors.
    /// Default: `1.0`
    pub global_scale: f32,

    /// When `true`, the sign of each source coefficient is preserved exactly.
    ///
    /// Default: `true`
    pub preserve_signs: bool,

    /// Exponential moving average factor applied to per-dimension scale factors,
    /// blending them toward 1.0. `0.0` disables smoothing; must be in `[0, 1)`.
    ///
    /// Default: `0.0`
    pub scale_smoothing: f32,
}

impl Default for RetargetingConfig {
    fn default() -> Self {
        Self {
            expr_clip: 3.0,
            global_scale: 1.0,
            preserve_signs: true,
            scale_smoothing: 0.0,
        }
    }
}

impl RetargetingConfig {
    /// Validate that all configuration fields are within their legal ranges.
    ///
    /// Returns `InvalidConfig` if any field is out of range.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn validate(&self) -> Result<(), RetargetingError> {
        if self.expr_clip <= 0.0 {
            return Err(RetargetingError::InvalidConfig(format!(
                "expr_clip must be positive, got {}",
                self.expr_clip
            )));
        }
        if self.global_scale <= 0.0 {
            return Err(RetargetingError::InvalidConfig(format!(
                "global_scale must be positive, got {}",
                self.global_scale
            )));
        }
        if !(0.0..1.0).contains(&self.scale_smoothing) {
            return Err(RetargetingError::InvalidConfig(format!(
                "scale_smoothing must be in [0, 1), got {}",
                self.scale_smoothing
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Basis scale computation
// ---------------------------------------------------------------------------

/// Compute the mean L2 displacement magnitude for each expression basis vector.
///
/// The basis is stored row-major with shape `[n_vertices * 3, n_expr_betas]`,
/// meaning element `(row, col)` is at index `row * n_expr_betas + col`.
///
/// For each expression basis `b`, this function computes the per-vertex
/// L2 norm of the 3D displacement vector and returns the mean over all vertices.
///
/// # Errors
///
/// Returns [`RetargetingError::EmptyParams`] if `n_vertices` or `n_expr_betas` is zero,
/// and [`RetargetingError::DimensionMismatch`] if `expr_basis.len()` does not equal
/// `n_vertices * 3 * n_expr_betas`.
pub fn compute_expr_basis_scales(
    expr_basis: &[f32],
    n_vertices: usize,
    n_expr_betas: usize,
) -> Result<Vec<f32>, RetargetingError> {
    if n_vertices == 0 || n_expr_betas == 0 {
        return Err(RetargetingError::EmptyParams);
    }
    let expected = n_vertices * 3 * n_expr_betas;
    if expr_basis.len() != expected {
        return Err(RetargetingError::DimensionMismatch {
            expected,
            got: expr_basis.len(),
        });
    }

    let mut scales = vec![0.0_f32; n_expr_betas];
    for b in 0..n_expr_betas {
        let mut sum_norm = 0.0_f32;
        for v in 0..n_vertices {
            let dx = expr_basis[(v * 3) * n_expr_betas + b];
            let dy = expr_basis[(v * 3 + 1) * n_expr_betas + b];
            let dz = expr_basis[(v * 3 + 2) * n_expr_betas + b];
            sum_norm += (dx * dx + dy * dy + dz * dz).sqrt();
        }
        scales[b] = sum_norm / n_vertices as f32;
    }
    Ok(scales)
}

/// Compute per-basis expression scale factors between source and target shapes.
///
/// Each scale factor is `target_scale / source_scale`, clamped to `[0.1, 10.0]`.
/// A scale of zero in the source is treated as 1.0 to avoid division by zero
/// (the corresponding expression basis has negligible effect on that identity).
///
/// # Errors
///
/// Returns [`RetargetingError::EmptyParams`] if either slice is empty, and
/// [`RetargetingError::DimensionMismatch`] if the slices have different lengths.
pub fn compute_expression_scale_factors(
    source_expr_scales: &[f32],
    target_expr_scales: &[f32],
) -> Result<Vec<f32>, RetargetingError> {
    if source_expr_scales.is_empty() {
        return Err(RetargetingError::EmptyParams);
    }
    if source_expr_scales.len() != target_expr_scales.len() {
        return Err(RetargetingError::DimensionMismatch {
            expected: source_expr_scales.len(),
            got: target_expr_scales.len(),
        });
    }

    let factors = source_expr_scales
        .iter()
        .zip(target_expr_scales.iter())
        .map(|(&src, &tgt)| {
            let denom = if src.abs() < f32::EPSILON {
                1.0_f32
            } else {
                src
            };
            (tgt / denom).clamp(0.1, 10.0)
        })
        .collect();
    Ok(factors)
}

// ---------------------------------------------------------------------------
// Core retargeting
// ---------------------------------------------------------------------------

/// Retarget expression parameters from source identity to target identity.
///
/// Steps:
/// 1. Clip each coefficient to `[-expr_clip, expr_clip]`.
/// 2. Multiply each coefficient by its per-dimension `scale_factor`.
/// 3. Apply optional EMA scale smoothing toward 1.0.
/// 4. Multiply all coefficients by `config.global_scale`.
///
/// # Errors
///
/// Returns [`RetargetingError::EmptyParams`] if any slice is empty,
/// [`RetargetingError::DimensionMismatch`] if lengths differ, and
/// [`RetargetingError::InvalidConfig`] if the config is invalid.
pub fn retarget_expression(
    source_expr: &[f32],
    scale_factors: &[f32],
    config: &RetargetingConfig,
) -> Result<Vec<f32>, RetargetingError> {
    config.validate()?;
    if source_expr.is_empty() {
        return Err(RetargetingError::EmptyParams);
    }
    if source_expr.len() != scale_factors.len() {
        return Err(RetargetingError::DimensionMismatch {
            expected: source_expr.len(),
            got: scale_factors.len(),
        });
    }

    let smoothed_scale = |raw: f32| -> f32 {
        let s = config.scale_smoothing;
        s * 1.0 + (1.0 - s) * raw
    };

    let result = source_expr
        .iter()
        .zip(scale_factors.iter())
        .map(|(&coeff, &sf)| {
            let clipped = coeff.clamp(-config.expr_clip, config.expr_clip);
            let effective_sf = smoothed_scale(sf);
            clipped * effective_sf * config.global_scale
        })
        .collect();
    Ok(result)
}

// ---------------------------------------------------------------------------
// Identity-preserving decomposition
// ---------------------------------------------------------------------------

/// Decompose expression parameters into identity-correlated and identity-independent parts.
///
/// The first `n_identity_correlated` dimensions (e.g., jaw, brow motion) tend to
/// correlate with face shape; higher-order coefficients are more identity-independent.
///
/// If `n_identity_correlated` exceeds `expr.len()`, it is clamped to `expr.len()`
/// so no error is returned.
///
/// Returns `(identity_correlated, identity_independent)`.
#[must_use]
pub fn decompose_expression(expr: &[f32], n_identity_correlated: usize) -> (Vec<f32>, Vec<f32>) {
    let split = n_identity_correlated.min(expr.len());
    let identity_corr = expr[..split].to_vec();
    let identity_indep = expr[split..].to_vec();
    (identity_corr, identity_indep)
}

/// Recompose expression parameters from their decomposed parts.
///
/// Concatenates `identity_corr` and `identity_indep` in order.
#[must_use]
pub fn recompose_expression(identity_corr: &[f32], identity_indep: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(identity_corr.len() + identity_indep.len());
    result.extend_from_slice(identity_corr);
    result.extend_from_slice(identity_indep);
    result
}

// ---------------------------------------------------------------------------
// Neutral pose correction
// ---------------------------------------------------------------------------

/// Compute a neutral correction offset.
///
/// Applies [`retarget_expression`] to `source_neutral_expr`, then negates the result.
/// Subtracting this correction from any retargeted expression removes the bias
/// introduced by the source's neutral pose, so the target's neutral looks neutral.
///
/// # Errors
///
/// Propagates errors from [`retarget_expression`].
pub fn compute_neutral_correction(
    source_neutral_expr: &[f32],
    scale_factors: &[f32],
    config: &RetargetingConfig,
) -> Result<Vec<f32>, RetargetingError> {
    let retargeted_neutral = retarget_expression(source_neutral_expr, scale_factors, config)?;
    Ok(retargeted_neutral.into_iter().map(|v| -v).collect())
}

/// Apply a neutral correction to retargeted expression coefficients.
///
/// Performs element-wise addition `retargeted[i] + correction[i]`.
///
/// # Errors
///
/// Returns [`RetargetingError::EmptyParams`] if either slice is empty and
/// [`RetargetingError::DimensionMismatch`] if their lengths differ.
pub fn apply_neutral_correction(
    retargeted: &[f32],
    correction: &[f32],
) -> Result<Vec<f32>, RetargetingError> {
    if retargeted.is_empty() {
        return Err(RetargetingError::EmptyParams);
    }
    if retargeted.len() != correction.len() {
        return Err(RetargetingError::DimensionMismatch {
            expected: retargeted.len(),
            got: correction.len(),
        });
    }
    Ok(retargeted
        .iter()
        .zip(correction.iter())
        .map(|(&r, &c)| r + c)
        .collect())
}

// ---------------------------------------------------------------------------
// Sequence retargeting
// ---------------------------------------------------------------------------

/// Retarget a sequence of expressions from source to target identity.
///
/// For each frame, calls [`retarget_expression`] and, if provided, applies
/// the neutral correction via [`apply_neutral_correction`].
///
/// An empty `source_sequence` returns an empty `Vec` without error.
///
/// # Errors
///
/// Propagates errors from [`retarget_expression`] and [`apply_neutral_correction`].
pub fn retarget_sequence(
    source_sequence: &[Vec<f32>],
    scale_factors: &[f32],
    config: &RetargetingConfig,
    neutral_correction: Option<&[f32]>,
) -> Result<Vec<Vec<f32>>, RetargetingError> {
    config.validate()?;
    let mut output = Vec::with_capacity(source_sequence.len());
    for frame in source_sequence {
        let retargeted = retarget_expression(frame, scale_factors, config)?;
        let final_frame = match neutral_correction {
            Some(correction) => apply_neutral_correction(&retargeted, correction)?,
            None => retargeted,
        };
        output.push(final_frame);
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Temporal smoothing
// ---------------------------------------------------------------------------

/// Apply exponential moving average (EMA) temporal smoothing to an expression sequence.
///
/// `output[0] = sequence[0]`
/// `output[i] = decay * output[i-1] + (1 - decay) * sequence[i]`
///
/// `decay` must be in `[0, 1)`. `0.0` disables smoothing (output equals input).
///
/// # Errors
///
/// Returns [`RetargetingError::EmptyParams`] if `sequence` is empty and
/// [`RetargetingError::InvalidConfig`] if `decay` is not in `[0, 1)`.
///
/// # Panics
///
/// Cannot panic in practice; the internal `unreachable!` path protects a
/// loop invariant that holds by construction.
pub fn smooth_expression_sequence(
    sequence: &[Vec<f32>],
    decay: f32,
) -> Result<Vec<Vec<f32>>, RetargetingError> {
    if sequence.is_empty() {
        return Err(RetargetingError::EmptyParams);
    }
    if !(0.0..1.0).contains(&decay) {
        return Err(RetargetingError::InvalidConfig(format!(
            "decay must be in [0, 1), got {decay}"
        )));
    }

    let mut output: Vec<Vec<f32>> = Vec::with_capacity(sequence.len());
    output.push(sequence[0].clone());

    for frame in &sequence[1..] {
        // Safety: output was just pushed before the loop and we push at the end of
        // each iteration, so `last()` always returns Some.
        let prev = output
            .last()
            .unwrap_or_else(|| unreachable!("output is non-empty at this point"));
        // prev and frame may have different lengths only if the caller supplied
        // inconsistent frames — we handle that gracefully by using the minimum length.
        let len = prev.len().min(frame.len());
        let smoothed: Vec<f32> = prev[..len]
            .iter()
            .zip(frame[..len].iter())
            .map(|(&p, &f)| decay * p + (1.0 - decay) * f)
            .collect();
        output.push(smoothed);
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Statistics summarising a retargeting operation over a sequence of frames.
#[derive(Debug, Clone)]
pub struct RetargetingStats {
    /// Number of frames processed.
    pub num_frames: usize,
    /// Mean of all per-dimension scale factors.
    pub mean_scale_factor: f32,
    /// Maximum per-dimension scale factor.
    pub max_scale_factor: f32,
    /// Minimum per-dimension scale factor.
    pub min_scale_factor: f32,
    /// Mean L2 magnitude of source expression vectors across frames.
    pub mean_expr_magnitude_source: f32,
    /// Mean L2 magnitude of target (retargeted) expression vectors across frames.
    pub mean_expr_magnitude_target: f32,
    /// Ratio `mean_expr_magnitude_target / mean_expr_magnitude_source`.
    pub magnitude_ratio: f32,
}

/// Compute statistics summarising how expressions changed during retargeting.
///
/// # Errors
///
/// Returns [`RetargetingError::EmptyParams`] if `scale_factors` is empty or
/// `source_sequence` is empty, and [`RetargetingError::DimensionMismatch`] if
/// `source_sequence` and `target_sequence` have different frame counts.
/// Also returns [`RetargetingError::InsufficientExpressionData`] if there are no frames.
pub fn compute_retargeting_stats(
    source_sequence: &[Vec<f32>],
    target_sequence: &[Vec<f32>],
    scale_factors: &[f32],
) -> Result<RetargetingStats, RetargetingError> {
    if scale_factors.is_empty() {
        return Err(RetargetingError::EmptyParams);
    }
    if source_sequence.is_empty() {
        return Err(RetargetingError::InsufficientExpressionData(
            "source_sequence has no frames".to_string(),
        ));
    }
    if source_sequence.len() != target_sequence.len() {
        return Err(RetargetingError::DimensionMismatch {
            expected: source_sequence.len(),
            got: target_sequence.len(),
        });
    }

    // Scale factor statistics.
    let sum_sf: f32 = scale_factors.iter().sum();
    let mean_scale_factor = sum_sf / scale_factors.len() as f32;
    let max_scale_factor = scale_factors
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_scale_factor = scale_factors.iter().copied().fold(f32::INFINITY, f32::min);

    // L2 magnitude helpers.
    let l2_magnitude = |v: &[f32]| -> f32 { v.iter().map(|&x| x * x).sum::<f32>().sqrt() };

    let num_frames = source_sequence.len();
    let mean_expr_magnitude_source =
        source_sequence.iter().map(|f| l2_magnitude(f)).sum::<f32>() / num_frames as f32;
    let mean_expr_magnitude_target =
        target_sequence.iter().map(|f| l2_magnitude(f)).sum::<f32>() / num_frames as f32;

    let magnitude_ratio = if mean_expr_magnitude_source.abs() < f32::EPSILON {
        1.0
    } else {
        mean_expr_magnitude_target / mean_expr_magnitude_source
    };

    Ok(RetargetingStats {
        num_frames,
        mean_scale_factor,
        max_scale_factor,
        min_scale_factor,
        mean_expr_magnitude_source,
        mean_expr_magnitude_target,
        magnitude_ratio,
    })
}

// ---------------------------------------------------------------------------
// Stateful retargeter
// ---------------------------------------------------------------------------

/// Stateful expression retargeter that holds precomputed scale factors and configuration.
///
/// This is the main entry point for production use. Build it once per identity pair
/// and call [`retarget`](ExpressionRetargeter::retarget) for each frame.
pub struct ExpressionRetargeter {
    scale_factors: Vec<f32>,
    config: RetargetingConfig,
    neutral_correction: Option<Vec<f32>>,
}

impl ExpressionRetargeter {
    /// Create an `ExpressionRetargeter` from precomputed scale factors.
    ///
    /// # Errors
    ///
    /// Returns [`RetargetingError::EmptyParams`] if `scale_factors` is empty, and
    /// propagates configuration validation errors.
    pub fn new(
        scale_factors: Vec<f32>,
        config: RetargetingConfig,
    ) -> Result<Self, RetargetingError> {
        config.validate()?;
        if scale_factors.is_empty() {
            return Err(RetargetingError::EmptyParams);
        }
        Ok(Self {
            scale_factors,
            config,
            neutral_correction: None,
        })
    }

    /// Create an `ExpressionRetargeter` from per-shape expression-basis scale vectors.
    ///
    /// Calls [`compute_expression_scale_factors`] internally.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`compute_expression_scale_factors`] and config validation.
    pub fn from_basis_scales(
        source_expr_scales: &[f32],
        target_expr_scales: &[f32],
        config: RetargetingConfig,
    ) -> Result<Self, RetargetingError> {
        config.validate()?;
        let scale_factors =
            compute_expression_scale_factors(source_expr_scales, target_expr_scales)?;
        Ok(Self {
            scale_factors,
            config,
            neutral_correction: None,
        })
    }

    /// Attach a neutral correction vector (see [`compute_neutral_correction`]).
    ///
    /// The correction is applied automatically in [`retarget`](Self::retarget) and
    /// [`retarget_sequence`](Self::retarget_sequence).
    #[must_use]
    pub fn with_neutral_correction(mut self, correction: Vec<f32>) -> Self {
        self.neutral_correction = Some(correction);
        self
    }

    /// Retarget a single expression frame.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`retarget_expression`] and [`apply_neutral_correction`].
    pub fn retarget(&self, source_expr: &[f32]) -> Result<Vec<f32>, RetargetingError> {
        let retargeted = retarget_expression(source_expr, &self.scale_factors, &self.config)?;
        match &self.neutral_correction {
            Some(correction) => apply_neutral_correction(&retargeted, correction),
            None => Ok(retargeted),
        }
    }

    /// Retarget a full sequence of expression frames.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`retarget_sequence`].
    pub fn retarget_sequence(
        &self,
        sequence: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, RetargetingError> {
        retarget_sequence(
            sequence,
            &self.scale_factors,
            &self.config,
            self.neutral_correction.as_deref(),
        )
    }

    /// Return the scale factors used by this retargeter.
    #[must_use]
    pub fn scale_factors(&self) -> &[f32] {
        &self.scale_factors
    }

    /// Return the effective scale for a single dimension: `global_scale * scale_factor[dim]`.
    ///
    /// Applies EMA smoothing toward 1.0 if `scale_smoothing > 0`.
    /// Returns `0.0` if `dim` is out of range.
    #[must_use]
    pub fn effective_scale(&self, dim: usize) -> f32 {
        if dim >= self.scale_factors.len() {
            return 0.0;
        }
        let raw_sf = self.scale_factors[dim];
        let s = self.config.scale_smoothing;
        let smoothed_sf = s * 1.0 + (1.0 - s) * raw_sf;
        self.config.global_scale * smoothed_sf
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    // -----------------------------------------------------------------------
    // compute_expr_basis_scales
    // -----------------------------------------------------------------------

    #[test]
    fn test_expr_basis_scales_identity_basis() -> Result<(), RetargetingError> {
        // 2 vertices, 2 expression betas, identity displacements.
        // Each column is [1,0,0, 1,0,0] => norm per vertex = 1.0, mean = 1.0
        let n_v = 2;
        let n_b = 2;
        // Row-major [n_v*3, n_b]
        // vertex 0 xyz: row 0,1,2  vertex 1 xyz: row 3,4,5
        // For basis 0 (col 0): dx=1, dy=0, dz=0 for both vertices => norm=1.0
        // For basis 1 (col 1): dx=0, dy=1, dz=0 for both vertices => norm=1.0
        let mut basis = vec![0.0_f32; n_v * 3 * n_b];
        // vertex 0, basis 0: row=0, col=0 -> index 0*2+0=0
        basis[0] = 1.0;
        // vertex 0, basis 1: row=1, col=1 -> index 1*2+1=3
        basis[3] = 1.0;
        // vertex 1, basis 0: row=3, col=0 -> index 3*2+0=6
        basis[6] = 1.0;
        // vertex 1, basis 1: row=4, col=1 -> index 4*2+1=9
        basis[9] = 1.0;
        let scales = compute_expr_basis_scales(&basis, n_v, n_b)?;
        assert_eq!(scales.len(), 2);
        assert!((scales[0] - 1.0).abs() < EPS, "scales[0]={}", scales[0]);
        assert!((scales[1] - 1.0).abs() < EPS, "scales[1]={}", scales[1]);
        Ok(())
    }

    #[test]
    fn test_expr_basis_scales_two_vertex_one_expr() -> Result<(), RetargetingError> {
        // 2 vertices, 1 expression beta.
        // vertex 0: displacement (3, 4, 0) => norm = 5.0
        // vertex 1: displacement (0, 0, 5) => norm = 5.0
        // mean = 5.0
        let n_v = 2;
        let n_b = 1;
        let mut basis = vec![0.0_f32; n_v * 3 * n_b];
        // Row-major [6, 1]; col=0 always.
        basis[0] = 3.0; // v0 dx
        basis[1] = 4.0; // v0 dy
        basis[2] = 0.0; // v0 dz
        basis[3] = 0.0; // v1 dx
        basis[4] = 0.0; // v1 dy
        basis[5] = 5.0; // v1 dz
        let scales = compute_expr_basis_scales(&basis, n_v, n_b)?;
        assert_eq!(scales.len(), 1);
        assert!((scales[0] - 5.0).abs() < EPS, "scales[0]={}", scales[0]);
        Ok(())
    }

    #[test]
    fn test_expr_basis_scales_wrong_len() {
        let result = compute_expr_basis_scales(&[1.0, 2.0], 2, 1);
        assert!(
            matches!(result, Err(RetargetingError::DimensionMismatch { .. })),
            "expected DimensionMismatch"
        );
    }

    #[test]
    fn test_expr_basis_scales_empty() {
        let result = compute_expr_basis_scales(&[], 0, 1);
        assert!(matches!(result, Err(RetargetingError::EmptyParams)));
    }

    // -----------------------------------------------------------------------
    // compute_expression_scale_factors
    // -----------------------------------------------------------------------

    #[test]
    fn test_scale_factors_equal_sources() -> Result<(), RetargetingError> {
        let src = vec![1.0_f32, 2.0, 3.0];
        let tgt = vec![1.0_f32, 2.0, 3.0];
        let factors = compute_expression_scale_factors(&src, &tgt)?;
        for f in &factors {
            assert!((f - 1.0).abs() < EPS, "expected 1.0, got {f}");
        }
        Ok(())
    }

    #[test]
    fn test_scale_factors_ratio_and_clamp() -> Result<(), RetargetingError> {
        // tgt = 2 * src => factor = 2.0, within [0.1, 10.0].
        let src = vec![1.0_f32, 0.5, 100.0];
        let tgt = vec![2.0_f32, 5.0, 0.001];
        let factors = compute_expression_scale_factors(&src, &tgt)?;
        assert!((factors[0] - 2.0).abs() < EPS, "f[0]={}", factors[0]);
        // 5.0/0.5 = 10.0 — boundary of clamp.
        assert!((factors[1] - 10.0).abs() < EPS, "f[1]={}", factors[1]);
        // 0.001/100 = 1e-5 => clamped to 0.1.
        assert!((factors[2] - 0.1).abs() < EPS, "f[2]={}", factors[2]);
        Ok(())
    }

    #[test]
    fn test_scale_factors_length_mismatch() {
        let result = compute_expression_scale_factors(&[1.0, 2.0], &[1.0]);
        assert!(matches!(
            result,
            Err(RetargetingError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_scale_factors_empty() {
        let result = compute_expression_scale_factors(&[], &[]);
        assert!(matches!(result, Err(RetargetingError::EmptyParams)));
    }

    // -----------------------------------------------------------------------
    // RetargetingConfig::validate
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_validate_valid() -> Result<(), RetargetingError> {
        RetargetingConfig::default().validate()
    }

    #[test]
    fn test_config_validate_zero_clip() {
        let cfg = RetargetingConfig {
            expr_clip: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(RetargetingError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_config_validate_negative_global_scale() {
        let cfg = RetargetingConfig {
            global_scale: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(RetargetingError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_config_validate_scale_smoothing_one() {
        let cfg = RetargetingConfig {
            scale_smoothing: 1.0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(RetargetingError::InvalidConfig(_))
        ));
    }

    // -----------------------------------------------------------------------
    // retarget_expression
    // -----------------------------------------------------------------------

    #[test]
    fn test_retarget_expression_identity() -> Result<(), RetargetingError> {
        let expr = vec![0.5_f32, -0.3, 1.0, -1.5];
        let scale_factors = vec![1.0_f32; 4];
        let cfg = RetargetingConfig {
            expr_clip: 100.0,
            global_scale: 1.0,
            ..Default::default()
        };
        let result = retarget_expression(&expr, &scale_factors, &cfg)?;
        for (a, b) in expr.iter().zip(result.iter()) {
            assert!((a - b).abs() < EPS, "a={a} b={b}");
        }
        Ok(())
    }

    #[test]
    fn test_retarget_expression_with_scaling() -> Result<(), RetargetingError> {
        let expr = vec![1.0_f32, 2.0];
        let scale_factors = vec![2.0_f32, 0.5];
        let cfg = RetargetingConfig::default();
        let result = retarget_expression(&expr, &scale_factors, &cfg)?;
        assert!((result[0] - 2.0).abs() < EPS);
        assert!((result[1] - 1.0).abs() < EPS);
        Ok(())
    }

    #[test]
    fn test_retarget_expression_with_clipping() -> Result<(), RetargetingError> {
        let expr = vec![5.0_f32, -5.0];
        let scale_factors = vec![1.0_f32, 1.0];
        let cfg = RetargetingConfig {
            expr_clip: 3.0,
            global_scale: 1.0,
            ..Default::default()
        };
        let result = retarget_expression(&expr, &scale_factors, &cfg)?;
        assert!(
            (result[0] - 3.0).abs() < EPS,
            "expected 3.0 got {}",
            result[0]
        );
        assert!(
            (result[1] - (-3.0)).abs() < EPS,
            "expected -3.0 got {}",
            result[1]
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // decompose / recompose
    // -----------------------------------------------------------------------

    #[test]
    fn test_decompose_recompose_roundtrip() {
        let expr = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let (corr, indep) = decompose_expression(&expr, 3);
        let reconstructed = recompose_expression(&corr, &indep);
        assert_eq!(reconstructed, expr);
    }

    #[test]
    fn test_decompose_n_greater_than_len_clamped() {
        let expr = vec![1.0_f32, 2.0, 3.0];
        let (corr, indep) = decompose_expression(&expr, 10); // clamped to 3
        assert_eq!(corr, vec![1.0, 2.0, 3.0]);
        assert!(indep.is_empty());
    }

    #[test]
    fn test_decompose_zero_correlated() {
        let expr = vec![1.0_f32, 2.0, 3.0];
        let (corr, indep) = decompose_expression(&expr, 0);
        assert!(corr.is_empty());
        assert_eq!(indep, expr);
    }

    // -----------------------------------------------------------------------
    // Neutral correction
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_neutral_correction() -> Result<(), RetargetingError> {
        // Neutral expr = all 1.0, scale_factors = all 1.0, clip=100.
        // retarget_expression produces all 1.0, neutral correction = all -1.0.
        let neutral = vec![1.0_f32; 4];
        let sf = vec![1.0_f32; 4];
        let cfg = RetargetingConfig {
            expr_clip: 100.0,
            global_scale: 1.0,
            ..Default::default()
        };
        let correction = compute_neutral_correction(&neutral, &sf, &cfg)?;
        for c in &correction {
            assert!((c - (-1.0)).abs() < EPS, "expected -1.0 got {c}");
        }
        Ok(())
    }

    #[test]
    fn test_apply_neutral_correction() -> Result<(), RetargetingError> {
        let retargeted = vec![2.0_f32, 3.0, 4.0];
        let correction = vec![-1.0_f32, -1.0, -1.0];
        let result = apply_neutral_correction(&retargeted, &correction)?;
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn test_apply_neutral_correction_length_mismatch() {
        let result = apply_neutral_correction(&[1.0, 2.0], &[0.0]);
        assert!(matches!(
            result,
            Err(RetargetingError::DimensionMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // Sequence retargeting
    // -----------------------------------------------------------------------

    #[test]
    fn test_retarget_sequence_empty() -> Result<(), RetargetingError> {
        let result = retarget_sequence(&[], &[1.0_f32], &RetargetingConfig::default(), None)?;
        assert!(result.is_empty());
        Ok(())
    }

    #[test]
    fn test_retarget_sequence_three_frames() -> Result<(), RetargetingError> {
        let seq = vec![vec![1.0_f32, 0.0], vec![0.5_f32, 0.5], vec![0.0_f32, 1.0]];
        let sf = vec![2.0_f32, 2.0];
        let cfg = RetargetingConfig {
            expr_clip: 10.0,
            global_scale: 1.0,
            ..Default::default()
        };
        let result = retarget_sequence(&seq, &sf, &cfg, None)?;
        assert_eq!(result.len(), 3);
        assert!((result[0][0] - 2.0).abs() < EPS);
        assert!((result[2][1] - 2.0).abs() < EPS);
        Ok(())
    }

    #[test]
    fn test_retarget_sequence_with_neutral_correction() -> Result<(), RetargetingError> {
        let seq = vec![vec![1.0_f32, 1.0]];
        let sf = vec![1.0_f32, 1.0];
        let cfg = RetargetingConfig::default();
        // correction = -0.5 per dim
        let correction = vec![-0.5_f32, -0.5];
        let result = retarget_sequence(&seq, &sf, &cfg, Some(&correction))?;
        // retarget produces [1.0, 1.0], then +correction = [0.5, 0.5]
        assert!((result[0][0] - 0.5).abs() < EPS);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Temporal smoothing
    // -----------------------------------------------------------------------

    #[test]
    fn test_smooth_expression_sequence_decay_zero() -> Result<(), RetargetingError> {
        let seq = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let result = smooth_expression_sequence(&seq, 0.0)?;
        // decay=0 => output equals input
        for (out, inp) in result.iter().zip(seq.iter()) {
            for (a, b) in out.iter().zip(inp.iter()) {
                assert!((a - b).abs() < EPS, "a={a} b={b}");
            }
        }
        Ok(())
    }

    #[test]
    fn test_smooth_expression_sequence_decay_half() -> Result<(), RetargetingError> {
        // Two frames, decay = 0.5.
        // output[0] = [1.0, 2.0]
        // output[1] = 0.5 * [1.0, 2.0] + 0.5 * [3.0, 4.0] = [2.0, 3.0]
        let seq = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]];
        let result = smooth_expression_sequence(&seq, 0.5)?;
        assert_eq!(result.len(), 2);
        assert!((result[0][0] - 1.0).abs() < EPS);
        assert!((result[0][1] - 2.0).abs() < EPS);
        assert!((result[1][0] - 2.0).abs() < EPS);
        assert!((result[1][1] - 3.0).abs() < EPS);
        Ok(())
    }

    #[test]
    fn test_smooth_expression_sequence_invalid_decay_one() {
        let seq = vec![vec![1.0_f32]];
        let result = smooth_expression_sequence(&seq, 1.0);
        assert!(matches!(result, Err(RetargetingError::InvalidConfig(_))));
    }

    #[test]
    fn test_smooth_expression_sequence_invalid_decay_gt_one() {
        let seq = vec![vec![1.0_f32]];
        let result = smooth_expression_sequence(&seq, 1.5);
        assert!(matches!(result, Err(RetargetingError::InvalidConfig(_))));
    }

    #[test]
    fn test_smooth_expression_sequence_empty() {
        let result = smooth_expression_sequence(&[], 0.5);
        assert!(matches!(result, Err(RetargetingError::EmptyParams)));
    }

    // -----------------------------------------------------------------------
    // ExpressionRetargeter
    // -----------------------------------------------------------------------

    #[test]
    fn test_expression_retargeter_new() -> Result<(), RetargetingError> {
        let sf = vec![1.5_f32, 2.0, 0.8];
        let cfg = RetargetingConfig::default();
        let retargeter = ExpressionRetargeter::new(sf.clone(), cfg)?;
        assert_eq!(retargeter.scale_factors(), &sf[..]);
        Ok(())
    }

    #[test]
    fn test_expression_retargeter_from_basis_scales() -> Result<(), RetargetingError> {
        let src = vec![1.0_f32, 2.0];
        let tgt = vec![2.0_f32, 2.0];
        let cfg = RetargetingConfig::default();
        let retargeter = ExpressionRetargeter::from_basis_scales(&src, &tgt, cfg)?;
        assert_eq!(retargeter.scale_factors().len(), 2);
        // scale factor for dim 0 = 2.0/1.0 = 2.0
        assert!((retargeter.scale_factors()[0] - 2.0).abs() < EPS);
        Ok(())
    }

    #[test]
    fn test_expression_retargeter_retarget_identity() -> Result<(), RetargetingError> {
        let sf = vec![1.0_f32; 5];
        let cfg = RetargetingConfig {
            expr_clip: 100.0,
            global_scale: 1.0,
            ..Default::default()
        };
        let retargeter = ExpressionRetargeter::new(sf, cfg)?;
        let expr = vec![0.3_f32, -0.5, 1.2, 0.0, -1.0];
        let result = retargeter.retarget(&expr)?;
        for (a, b) in expr.iter().zip(result.iter()) {
            assert!((a - b).abs() < EPS);
        }
        Ok(())
    }

    #[test]
    fn test_expression_retargeter_retarget_sequence() -> Result<(), RetargetingError> {
        let sf = vec![2.0_f32, 2.0];
        let cfg = RetargetingConfig {
            expr_clip: 10.0,
            global_scale: 1.0,
            ..Default::default()
        };
        let retargeter = ExpressionRetargeter::new(sf, cfg)?;
        let seq = vec![vec![0.5_f32, 0.5], vec![1.0, 1.0]];
        let result = retargeter.retarget_sequence(&seq)?;
        assert_eq!(result.len(), 2);
        assert!((result[0][0] - 1.0).abs() < EPS);
        assert!((result[1][0] - 2.0).abs() < EPS);
        Ok(())
    }

    #[test]
    fn test_expression_retargeter_effective_scale() -> Result<(), RetargetingError> {
        let sf = vec![2.0_f32, 3.0];
        let cfg = RetargetingConfig {
            global_scale: 0.5,
            ..Default::default()
        };
        let retargeter = ExpressionRetargeter::new(sf, cfg)?;
        assert!((retargeter.effective_scale(0) - 1.0).abs() < EPS); // 0.5 * 2.0
        assert!((retargeter.effective_scale(1) - 1.5).abs() < EPS); // 0.5 * 3.0
        assert_eq!(retargeter.effective_scale(99), 0.0); // out of range
        Ok(())
    }

    // -----------------------------------------------------------------------
    // compute_retargeting_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_retargeting_stats_basic() -> Result<(), RetargetingError> {
        let source_seq = vec![vec![3.0_f32, 4.0]]; // |v| = 5.0
        let target_seq = vec![vec![6.0_f32, 8.0]]; // |v| = 10.0
        let sf = vec![2.0_f32, 2.0];
        let stats = compute_retargeting_stats(&source_seq, &target_seq, &sf)?;
        assert_eq!(stats.num_frames, 1);
        assert!((stats.mean_scale_factor - 2.0).abs() < EPS);
        assert!((stats.mean_expr_magnitude_source - 5.0).abs() < EPS);
        assert!((stats.mean_expr_magnitude_target - 10.0).abs() < EPS);
        Ok(())
    }

    #[test]
    fn test_compute_retargeting_stats_magnitude_ratio() -> Result<(), RetargetingError> {
        let source_seq = vec![vec![1.0_f32, 0.0], vec![1.0, 0.0]]; // mean |v| = 1.0
        let target_seq = vec![vec![3.0_f32, 0.0], vec![3.0, 0.0]]; // mean |v| = 3.0
        let sf = vec![1.0_f32, 1.0];
        let stats = compute_retargeting_stats(&source_seq, &target_seq, &sf)?;
        assert!(
            (stats.magnitude_ratio - 3.0).abs() < EPS,
            "ratio={}",
            stats.magnitude_ratio
        );
        Ok(())
    }

    #[test]
    fn test_compute_retargeting_stats_min_max_scale() -> Result<(), RetargetingError> {
        let seq = vec![vec![0.0_f32]];
        let sf = vec![0.5_f32, 1.0, 3.0];
        // source/target don't need to match sf length — sf is separate
        let source_seq = vec![vec![1.0_f32, 0.0, 0.0]];
        let target_seq = vec![vec![1.0_f32, 0.0, 0.0]];
        let stats = compute_retargeting_stats(&source_seq, &target_seq, &sf)?;
        assert!((stats.min_scale_factor - 0.5).abs() < EPS);
        assert!((stats.max_scale_factor - 3.0).abs() < EPS);
        // Suppress unused warning
        drop(seq);
        Ok(())
    }

    #[test]
    fn test_compute_retargeting_stats_frame_mismatch() {
        let source_seq = vec![vec![1.0_f32]];
        let target_seq = vec![vec![1.0_f32], vec![2.0_f32]];
        let sf = vec![1.0_f32];
        let result = compute_retargeting_stats(&source_seq, &target_seq, &sf);
        assert!(matches!(
            result,
            Err(RetargetingError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_compute_retargeting_stats_empty_source() {
        let result = compute_retargeting_stats(&[], &[], &[1.0_f32]);
        assert!(matches!(
            result,
            Err(RetargetingError::InsufficientExpressionData(_))
        ));
    }
}
