//! Selective FP32 upcast utilities for numerical stability.
//!
//! This module provides pure-Rust, pure-f32 implementations of common neural
//! network operations that require higher precision than the model's nominal
//! dtype (e.g., when running in FP16 or BF16).
//!
//! ## Motivation
//!
//! Mixed-precision inference (FP16/BF16) reduces memory and improves throughput,
//! but several operations are numerically sensitive:
//!
//! - **Softmax** with large logits causes exp() overflow in FP16.
//! - **Layer normalization** variance accumulation loses precision.
//! - **Attention** score distributions collapse in FP16 for long sequences.
//!
//! The recommended approach is *selective upcast*: keep most tensors in FP16,
//! but promote specific sensitive slices to FP32 for the critical operation,
//! then demote the result back. This module implements the FP32 kernels for
//! these critical paths.
//!
//! ## Contents
//!
//! - [`AttentionPrecision`] — configuration enum for attention upcasting.
//! - [`softmax_inplace`] / [`softmax`] — log-sum-exp stable softmax.
//! - [`log_softmax`] — log-domain stable softmax.
//! - [`layer_norm`] — full layer normalization with optional affine params.
//! - [`rms_norm`] — root-mean-square normalization (RMSNorm).
//! - [`gelu`] / [`gelu_slice`] — GELU activation (tanh approximation).
//! - [`silu`] / [`silu_slice`] — SiLU / Swish activation.
//! - [`clamp_for_stability`], [`is_subnormal`], [`count_subnormal`] — diagnostics.
//! - [`NumericsError`] — error type for length/input mismatches.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Precision configuration
// ---------------------------------------------------------------------------

/// Precision mode for attention computation.
///
/// Controls when, and to what extent, attention inputs are upcast to FP32 for
/// numerical stability. This is especially relevant when the model runs in
/// FP16 or BF16.
///
/// # Default
///
/// [`AttentionPrecision::UpcastedSoftmax`] — only the softmax step is
/// promoted to FP32. This is the recommended setting for most FP16 models
/// because it eliminates exp() overflow without the full cost of upcast QKV.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionPrecision {
    /// Standard precision — use whatever the model runs in (no forced upcast).
    ///
    /// Suitable for BF16 (which has a wider dynamic range than FP16) or when
    /// the caller has already verified that overflow is not a risk.
    Standard,

    /// Upcast the softmax step to FP32 only (recommended for FP16 models).
    ///
    /// Query-key dot products are computed in the model's native precision, but
    /// the log-sum-exp and exp() operations are performed in FP32 to prevent
    /// overflow for large logit magnitudes. The resulting attention weights are
    /// then used to compute the weighted-V sum in FP32 and stored back.
    #[default]
    UpcastedSoftmax,

    /// Fully upcast Q, K, and V to FP32 before any attention computation.
    ///
    /// Safest but slowest: all arithmetic is in FP32. Use this when debugging
    /// numerical issues or when `UpcastedSoftmax` is insufficient (rare).
    FullUpcast,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by numerics operations.
#[derive(Debug, Error)]
pub enum NumericsError {
    /// Gamma or beta slice length does not match the input.
    #[error("Length mismatch: expected {expected}, got {got}")]
    LengthMismatch { expected: usize, got: usize },

    /// Operation was called with an empty input slice.
    #[error("Empty input")]
    EmptyInput,
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

/// Numerically stable in-place softmax over a flat slice.
///
/// Applies the log-sum-exp trick: subtract the row maximum before computing
/// `exp()`, preventing overflow for large logit values.
///
/// # Properties
///
/// - If `logits` is empty, this is a no-op.
/// - The output values sum to 1.0 (up to floating-point rounding).
/// - Invariant to adding a constant to all logits.
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::softmax_inplace;
///
/// let mut logits = vec![1.0_f32, 2.0, 3.0];
/// softmax_inplace(&mut logits);
/// let sum: f32 = logits.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-6);
/// ```
pub fn softmax_inplace(logits: &mut [f32]) {
    if logits.is_empty() {
        return;
    }
    // Subtract max for numerical stability (log-sum-exp trick).
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0_f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }

    // Normalize. If sum is zero (all -inf inputs), leave as-is to avoid NaN.
    if sum > 0.0 {
        for v in logits.iter_mut() {
            *v /= sum;
        }
    }
}

/// Numerically stable softmax returning a new `Vec<f32>`.
///
/// This is the owning-output variant of [`softmax_inplace`].
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::softmax;
///
/// let out = softmax(&[1.0_f32, 2.0, 3.0]);
/// let sum: f32 = out.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-6);
/// ```
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut out = logits.to_vec();
    softmax_inplace(&mut out);
    out
}

/// Numerically stable log-softmax.
///
/// Computes `logits[i] - log(sum(exp(logits - max)))` using the log-sum-exp
/// identity, which avoids catastrophic cancellation and exp() overflow.
///
/// # Properties
///
/// - `exp(log_softmax(x))` ≈ `softmax(x)` (up to floating-point rounding).
/// - A single-element input returns `0.0`.
/// - Returns an empty `Vec` for empty input.
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::log_softmax;
///
/// let lsm = log_softmax(&[1.0_f32, 2.0, 3.0]);
/// assert!((lsm[0].exp() + lsm[1].exp() + lsm[2].exp() - 1.0).abs() < 1e-5);
/// ```
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute log(sum(exp(x - max))) = log_sum_exp - max.
    let log_sum_exp: f32 = logits.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();

    logits.iter().map(|&x| (x - max) - log_sum_exp).collect()
}

// ---------------------------------------------------------------------------
// Layer normalization
// ---------------------------------------------------------------------------

/// Numerically stable layer normalization.
///
/// Normalizes `input` to zero mean and unit variance, then applies the optional
/// learned affine transformation `gamma` (scale) and `beta` (bias):
///
/// ```text
/// output[i] = (input[i] - mean) / sqrt(variance + eps) * gamma[i] + beta[i]
/// ```
///
/// # Arguments
///
/// * `input` — the raw input slice (not modified).
/// * `gamma` — optional scale parameter (must have length `input.len()` if `Some`).
/// * `beta`  — optional bias parameter (must have length `input.len()` if `Some`).
/// * `eps`   — small constant added to variance for numerical safety (e.g., `1e-5`).
///
/// # Errors
///
/// - [`NumericsError::EmptyInput`] if `input` is empty.
/// - [`NumericsError::LengthMismatch`] if `gamma` or `beta` length ≠ `input.len()`.
pub fn layer_norm(
    input: &[f32],
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
    eps: f32,
) -> Result<Vec<f32>, NumericsError> {
    if input.is_empty() {
        return Err(NumericsError::EmptyInput);
    }
    if let Some(g) = gamma {
        if g.len() != input.len() {
            return Err(NumericsError::LengthMismatch {
                expected: input.len(),
                got: g.len(),
            });
        }
    }
    if let Some(b) = beta {
        if b.len() != input.len() {
            return Err(NumericsError::LengthMismatch {
                expected: input.len(),
                got: b.len(),
            });
        }
    }

    let n = input.len() as f32;

    // Compute mean in f64 for extra precision, then cast back.
    let mean: f32 = (input.iter().map(|&x| x as f64).sum::<f64>() / n as f64) as f32;

    // Compute variance using E[(X - mean)^2].
    let var: f32 = (input
        .iter()
        .map(|&x| (x as f64 - mean as f64).powi(2))
        .sum::<f64>()
        / n as f64) as f32;

    let inv_std = 1.0 / (var + eps).sqrt();

    let out: Vec<f32> = input
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let normalized = (x - mean) * inv_std;
            let scaled = match gamma {
                Some(g) => normalized * g[i],
                None => normalized,
            };
            match beta {
                Some(b) => scaled + b[i],
                None => scaled,
            }
        })
        .collect();

    Ok(out)
}

// ---------------------------------------------------------------------------
// RMS normalization
// ---------------------------------------------------------------------------

/// Root mean square normalization (RMSNorm).
///
/// A simplified variant of layer normalization that omits mean-centering:
///
/// ```text
/// rms  = sqrt(mean(input^2) + eps)
/// output[i] = input[i] / rms * gamma[i]
/// ```
///
/// RMSNorm is used in models such as LLaMA and PaLM as a computationally
/// cheaper alternative to full layer normalization.
///
/// # Arguments
///
/// * `input` — the raw input slice.
/// * `gamma` — optional scale; must match `input.len()` when `Some`.
/// * `eps`   — stability constant (e.g., `1e-6`).
///
/// # Errors
///
/// - [`NumericsError::EmptyInput`] if `input` is empty.
/// - [`NumericsError::LengthMismatch`] if `gamma` length ≠ `input.len()`.
pub fn rms_norm(input: &[f32], gamma: Option<&[f32]>, eps: f32) -> Result<Vec<f32>, NumericsError> {
    if input.is_empty() {
        return Err(NumericsError::EmptyInput);
    }
    if let Some(g) = gamma {
        if g.len() != input.len() {
            return Err(NumericsError::LengthMismatch {
                expected: input.len(),
                got: g.len(),
            });
        }
    }

    let n = input.len() as f64;
    let mean_sq: f32 = (input.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / n) as f32;
    let rms = (mean_sq + eps).sqrt();
    let inv_rms = 1.0 / rms;

    let out: Vec<f32> = input
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let normalized = x * inv_rms;
            match gamma {
                Some(g) => normalized * g[i],
                None => normalized,
            }
        })
        .collect();

    Ok(out)
}

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

/// GELU activation (Gaussian Error Linear Unit) — tanh approximation.
///
/// Uses the fast approximation from OpenAI's GPT paper:
/// ```text
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
/// ```
///
/// This approximation is used in BERT, GPT-2, and most modern transformers.
/// For the exact erf-based GELU use a dedicated math library.
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::gelu;
///
/// assert!((gelu(0.0_f32) - 0.0).abs() < 1e-6);
/// assert!((gelu(1.0_f32) - 0.8413).abs() < 1e-3);
/// ```
#[inline]
pub fn gelu(x: f32) -> f32 {
    // sqrt(2/π) ≈ 0.7978845608
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEF: f32 = 0.044_715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + COEF * x * x * x)).tanh())
}

/// Apply [`gelu`] element-wise, returning a new `Vec<f32>`.
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::gelu_slice;
///
/// let out = gelu_slice(&[0.0_f32, 1.0, -1.0]);
/// assert!((out[0] - 0.0).abs() < 1e-6);
/// ```
pub fn gelu_slice(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| gelu(x)).collect()
}

/// SiLU activation (Sigmoid Linear Unit), also known as Swish.
///
/// ```text
/// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
/// ```
///
/// Used in MobileNet-v3, EfficientNet, and various diffusion U-Nets.
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::silu;
///
/// assert!((silu(0.0_f32) - 0.0).abs() < 1e-6);
/// // silu(1.0) = 1 / (1 + e^{-1}) ≈ 0.7311
/// assert!((silu(1.0_f32) - 0.7311).abs() < 1e-3);
/// ```
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Apply [`silu`] element-wise, returning a new `Vec<f32>`.
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::silu_slice;
///
/// let out = silu_slice(&[0.0_f32, 1.0]);
/// assert!((out[0] - 0.0).abs() < 1e-6);
/// ```
pub fn silu_slice(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| silu(x)).collect()
}

// ---------------------------------------------------------------------------
// Diagnostic utilities
// ---------------------------------------------------------------------------

/// Clamp a value to `[min, max]` for numerical safety.
///
/// This is a thin wrapper over `f32::clamp` exposed for use in pipelines
/// where callers want to document intent (stability clamping vs. algorithm
/// clamping).
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::numerics::clamp_for_stability;
///
/// assert_eq!(clamp_for_stability(2.0, -1.0, 1.0), 1.0);
/// assert_eq!(clamp_for_stability(-2.0, -1.0, 1.0), -1.0);
/// ```
#[inline]
pub fn clamp_for_stability(value: f32, min: f32, max: f32) -> f32 {
    value.clamp(min, max)
}

/// Returns `true` if `v` is a subnormal f32 value.
///
/// Subnormal values (also called denormals) are very small f32 values near
/// zero that lose precision and can cause significant performance degradation
/// on some hardware (e.g., x86 without FTZ/DAZ flags set).
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::is_subnormal;
///
/// assert!(is_subnormal(f32::MIN_POSITIVE / 2.0));
/// assert!(!is_subnormal(1.0_f32));
/// ```
#[inline]
pub fn is_subnormal(v: f32) -> bool {
    v.is_subnormal()
}

/// Count the number of subnormal f32 values in a slice.
///
/// Useful as a diagnostic during model development: a large number of
/// subnormals indicates the model may be losing precision or causing
/// performance issues.
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::count_subnormal;
///
/// let data = vec![1.0_f32, f32::MIN_POSITIVE / 2.0, 0.0, f32::MIN_POSITIVE / 4.0];
/// assert_eq!(count_subnormal(&data), 2);
/// ```
pub fn count_subnormal(data: &[f32]) -> usize {
    data.iter().filter(|&&v| v.is_subnormal()).count()
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

    /// Compare two f32 slices element-wise within tolerance.
    fn approx_eq_slice(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "slice lengths differ");
        for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (av - bv).abs() <= tol,
                "index {i}: {av} vs {bv} (tol={tol})"
            );
        }
    }

    // ------------------------------------------------------------------
    // AttentionPrecision
    // ------------------------------------------------------------------

    #[test]
    fn test_attention_precision_default_is_upcasted_softmax() {
        assert_eq!(
            AttentionPrecision::default(),
            AttentionPrecision::UpcastedSoftmax,
        );
    }

    #[test]
    fn test_attention_precision_all_variants_are_distinct() {
        assert_ne!(
            AttentionPrecision::Standard,
            AttentionPrecision::UpcastedSoftmax
        );
        assert_ne!(AttentionPrecision::Standard, AttentionPrecision::FullUpcast);
        assert_ne!(
            AttentionPrecision::UpcastedSoftmax,
            AttentionPrecision::FullUpcast
        );
    }

    // ------------------------------------------------------------------
    // softmax_inplace
    // ------------------------------------------------------------------

    #[test]
    fn test_softmax_inplace_sums_to_one() {
        let mut logits = vec![1.0_f32, 2.0, 3.0];
        softmax_inplace(&mut logits);
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn test_softmax_inplace_empty_is_noop() {
        let mut logits: Vec<f32> = Vec::new();
        softmax_inplace(&mut logits); // must not panic
        assert!(logits.is_empty());
    }

    #[test]
    fn test_softmax_inplace_large_values_no_overflow() {
        // Without max-subtract, exp(1000) overflows to inf.
        let mut logits = vec![1000.0_f32, 1001.0];
        softmax_inplace(&mut logits);
        for &v in &logits {
            assert!(v.is_finite(), "overflow detected: {v}");
        }
        let sum: f32 = logits.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn test_softmax_inplace_uniform_values() {
        let n = 5usize;
        let mut logits = vec![2.0_f32; n];
        softmax_inplace(&mut logits);
        let expected = 1.0 / n as f32;
        for (i, &v) in logits.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-6,
                "index {i}: got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_softmax_inplace_single_element_is_one() {
        let mut logits = vec![42.0_f32];
        softmax_inplace(&mut logits);
        assert!((logits[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_all_values_non_negative() {
        let mut logits = vec![-5.0_f32, -1.0, 0.0, 2.0, 10.0];
        softmax_inplace(&mut logits);
        for &v in &logits {
            assert!(v >= 0.0, "softmax output should be non-negative, got {v}");
        }
    }

    // ------------------------------------------------------------------
    // log_softmax
    // ------------------------------------------------------------------

    #[test]
    fn test_log_softmax_single_element_is_zero() {
        let lsm = log_softmax(&[3.7_f32]);
        assert!((lsm[0] - 0.0).abs() < 1e-6, "got {}", lsm[0]);
    }

    #[test]
    fn test_log_softmax_exp_recovers_softmax() {
        let logits = [1.0_f32, 2.0, 3.0, 0.5];
        let lsm = log_softmax(&logits);
        let sm = softmax(&logits);
        for (i, (l, s)) in lsm.iter().zip(sm.iter()).enumerate() {
            let recovered = l.exp();
            assert!(
                (recovered - s).abs() < 1e-5,
                "index {i}: exp(log_softmax)={recovered}, softmax={s}"
            );
        }
    }

    #[test]
    fn test_log_softmax_empty_returns_empty() {
        let lsm = log_softmax(&[]);
        assert!(lsm.is_empty());
    }

    #[test]
    fn test_log_softmax_values_are_non_positive() {
        // log(p) <= 0 for p in (0, 1].
        let logits = [1.0_f32, 2.0, 3.0];
        let lsm = log_softmax(&logits);
        for (i, &v) in lsm.iter().enumerate() {
            assert!(v <= 0.0 + 1e-6, "index {i}: log_softmax value {v} > 0");
        }
    }

    // ------------------------------------------------------------------
    // layer_norm
    // ------------------------------------------------------------------

    #[test]
    fn test_layer_norm_identity_affine_mean_and_std() {
        let input = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let out = layer_norm(&input, None, None, 1e-5).expect("layer_norm failed");
        // Mean of normalized output should be ~0.
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(mean.abs() < 1e-4, "mean={mean}");
        // Variance ~1 (std ~1).
        let var: f32 = out.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / out.len() as f32;
        assert!((var - 1.0).abs() < 1e-3, "var={var}");
    }

    #[test]
    fn test_layer_norm_with_gamma_2_beta_1() {
        let input = vec![1.0_f32, 2.0, 3.0];
        let gamma = vec![2.0_f32; 3];
        let beta = vec![1.0_f32; 3];
        let out = layer_norm(&input, Some(&gamma), Some(&beta), 1e-5).expect("layer_norm failed");
        // After scaling by 2 and shifting by 1: mean should be 1, std should be 2.
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!((mean - 1.0).abs() < 1e-3, "mean={mean}");
        let var: f32 = out.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / out.len() as f32;
        assert!((var - 4.0).abs() < 1e-2, "var={var}");
    }

    #[test]
    fn test_layer_norm_empty_returns_error() {
        let result = layer_norm(&[], None, None, 1e-5);
        assert!(matches!(result, Err(NumericsError::EmptyInput)));
    }

    #[test]
    fn test_layer_norm_mismatched_gamma_returns_error() {
        let input = vec![1.0_f32, 2.0, 3.0];
        let gamma = vec![1.0_f32; 5]; // wrong length
        let result = layer_norm(&input, Some(&gamma), None, 1e-5);
        assert!(matches!(
            result,
            Err(NumericsError::LengthMismatch {
                expected: 3,
                got: 5
            })
        ));
    }

    #[test]
    fn test_layer_norm_mismatched_beta_returns_error() {
        let input = vec![1.0_f32, 2.0, 3.0];
        let beta = vec![0.0_f32; 2]; // wrong length
        let result = layer_norm(&input, None, Some(&beta), 1e-5);
        assert!(matches!(
            result,
            Err(NumericsError::LengthMismatch {
                expected: 3,
                got: 2
            })
        ));
    }

    #[test]
    fn test_layer_norm_output_length_matches_input() {
        let input: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let out = layer_norm(&input, None, None, 1e-5).expect("layer_norm failed");
        assert_eq!(out.len(), input.len());
    }

    // ------------------------------------------------------------------
    // rms_norm
    // ------------------------------------------------------------------

    #[test]
    fn test_rms_norm_output_rms_approx_one_no_gamma() {
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = rms_norm(&input, None, 1e-6).expect("rms_norm failed");
        // After RMS normalization, rms(out) == 1.0 (when gamma=None=all-ones).
        let mean_sq: f32 = out.iter().map(|&x| x * x).sum::<f32>() / out.len() as f32;
        let rms = mean_sq.sqrt();
        assert!((rms - 1.0).abs() < 1e-4, "rms={rms}");
    }

    #[test]
    fn test_rms_norm_with_gamma_scales_output() {
        let input = vec![1.0_f32, 2.0, 3.0];
        let gamma = vec![2.0_f32; 3];
        let out_no_gamma = rms_norm(&input, None, 1e-6).expect("rms_norm failed");
        let out_gamma = rms_norm(&input, Some(&gamma), 1e-6).expect("rms_norm failed");
        // gamma=2 should double each element.
        approx_eq_slice(
            &out_gamma,
            &out_no_gamma.iter().map(|&x| x * 2.0).collect::<Vec<_>>(),
            1e-5,
        );
    }

    #[test]
    fn test_rms_norm_empty_returns_error() {
        let result = rms_norm(&[], None, 1e-6);
        assert!(matches!(result, Err(NumericsError::EmptyInput)));
    }

    #[test]
    fn test_rms_norm_mismatched_gamma_returns_error() {
        let input = vec![1.0_f32; 4];
        let gamma = vec![1.0_f32; 3]; // wrong length
        let result = rms_norm(&input, Some(&gamma), 1e-6);
        assert!(matches!(
            result,
            Err(NumericsError::LengthMismatch {
                expected: 4,
                got: 3
            })
        ));
    }

    // ------------------------------------------------------------------
    // GELU
    // ------------------------------------------------------------------

    #[test]
    fn test_gelu_zero_is_zero() {
        assert!(
            (gelu(0.0_f32) - 0.0).abs() < 1e-6,
            "gelu(0) = {}",
            gelu(0.0)
        );
    }

    #[test]
    fn test_gelu_one_approx_known_value() {
        // Tanh-approx gelu(1.0) ≈ 0.8413 (matches PyTorch behavior).
        let v = gelu(1.0_f32);
        assert!((v - 0.8413).abs() < 2e-3, "gelu(1.0)={v}");
    }

    #[test]
    fn test_gelu_negative_input_gives_small_negative_output() {
        let v = gelu(-1.0_f32);
        assert!(v < 0.0, "gelu(-1) should be negative, got {v}");
        assert!(v > -0.2, "gelu(-1) should be small negative, got {v}");
    }

    #[test]
    fn test_gelu_slice_applies_elementwise() {
        let input = vec![0.0_f32, 1.0, -1.0, 2.0];
        let out = gelu_slice(&input);
        assert_eq!(out.len(), input.len());
        for (i, (&x, &y)) in input.iter().zip(out.iter()).enumerate() {
            assert!(
                (y - gelu(x)).abs() < 1e-7,
                "index {i}: gelu({x})={} but got {y}",
                gelu(x)
            );
        }
    }

    // ------------------------------------------------------------------
    // SiLU
    // ------------------------------------------------------------------

    #[test]
    fn test_silu_zero_is_zero() {
        assert!(
            (silu(0.0_f32) - 0.0).abs() < 1e-6,
            "silu(0) = {}",
            silu(0.0)
        );
    }

    #[test]
    fn test_silu_one_approx_known_value() {
        // silu(1) = 1 / (1 + e^{-1}) ≈ 0.7311.
        let v = silu(1.0_f32);
        assert!((v - 0.7311).abs() < 1e-3, "silu(1.0)={v}");
    }

    #[test]
    fn test_silu_slice_applies_elementwise() {
        let input = vec![0.0_f32, 1.0, -2.0, 3.5];
        let out = silu_slice(&input);
        assert_eq!(out.len(), input.len());
        for (i, (&x, &y)) in input.iter().zip(out.iter()).enumerate() {
            assert!(
                (y - silu(x)).abs() < 1e-7,
                "index {i}: silu({x})={} but got {y}",
                silu(x)
            );
        }
    }

    // ------------------------------------------------------------------
    // Diagnostics
    // ------------------------------------------------------------------

    #[test]
    fn test_is_subnormal_correct() {
        // A known subnormal: f32::MIN_POSITIVE / 2.0.
        let subnormal = f32::MIN_POSITIVE / 2.0;
        assert!(is_subnormal(subnormal), "{subnormal} should be subnormal");
        assert!(!is_subnormal(1.0_f32), "1.0 should not be subnormal");
        assert!(!is_subnormal(0.0_f32), "0.0 should not be subnormal");
    }

    #[test]
    fn test_count_subnormal_counts_correctly() {
        let subnormal = f32::MIN_POSITIVE / 2.0;
        let data = vec![1.0_f32, subnormal, 0.0, subnormal, 42.0];
        assert_eq!(count_subnormal(&data), 2);
    }

    #[test]
    fn test_count_subnormal_zero_on_normal_data() {
        let data = vec![1.0_f32, 2.0, -3.5, 0.0, f32::MAX];
        assert_eq!(count_subnormal(&data), 0);
    }

    #[test]
    fn test_clamp_for_stability_clamps_correctly() {
        assert_eq!(clamp_for_stability(5.0, -1.0, 1.0), 1.0);
        assert_eq!(clamp_for_stability(-5.0, -1.0, 1.0), -1.0);
        assert_eq!(clamp_for_stability(0.5, -1.0, 1.0), 0.5);
    }
}
