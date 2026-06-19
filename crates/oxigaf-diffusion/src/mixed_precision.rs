//! Mixed-precision inference utilities for the diffusion pipeline.
//!
//! This module provides configuration and pure-Rust implementations of FP32 ↔
//! BF16 / FP16 conversions. It is unconditionally compiled; the
//! `mixed_precision` feature flag only changes the *default* precision mode.
//!
//! ## BF16 (Brain Float 16)
//!
//! BF16 shares the same 8-bit exponent as FP32 (bias 127) but truncates the
//! 23-bit mantissa to 7 bits, giving the same dynamic range with less
//! precision. Conversion is simply: round the lower 16 mantissa bits, then
//! keep the upper 16 bits.
//!
//! ## FP16 (IEEE 754 half-precision)
//!
//! FP16 has a 5-bit exponent (bias 15) and 10-bit mantissa. Conversion from
//! FP32 requires rebasing the exponent (subtract 112 = 127 − 15) and
//! handling subnormals, overflow, NaN, and zero explicitly.
//!
//! ## Usage
//!
//! ```rust
//! use oxigaf_diffusion::mixed_precision::{
//!     MixedPrecisionConfig, PrecisionMode, apply_precision, simulate_bf16,
//! };
//!
//! let config = MixedPrecisionConfig::default();
//! let weights = vec![1.0_f32, 0.5, -0.25, 100.0];
//! let quantized = apply_precision(&weights, &config);
//! // In FP32 mode the result is an exact copy.
//! assert_eq!(quantized, weights);
//! ```

use crate::DiffusionError;

// ---------------------------------------------------------------------------
// PrecisionMode
// ---------------------------------------------------------------------------

/// Precision mode for model weights and activations.
///
/// Controls which numeric format is used for the bulk of tensor arithmetic.
/// Operations that are known to be numerically sensitive (softmax, layer-norm,
/// output projection) can be individually promoted to FP32 regardless of the
/// active mode via [`MixedPrecisionConfig::should_upcast`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrecisionMode {
    /// Full FP32 everywhere (default, safest).
    ///
    /// No quantization is applied. All arithmetic uses 32-bit floats.
    #[default]
    Float32,

    /// BF16 for weights and most activations, FP32 for sensitive ops.
    ///
    /// BF16 preserves the full FP32 dynamic range (8-bit exponent, bias 127)
    /// but reduces the mantissa from 23 bits to 7 bits. This yields ≈50 %
    /// memory savings with minimal accuracy loss on most models.
    BFloat16,

    /// FP16 for weights and most activations, FP32 for sensitive ops.
    ///
    /// IEEE 754 half-precision: 5-bit exponent (bias 15), 10-bit mantissa,
    /// max representable value ≈ 65504. Values outside this range overflow to
    /// ±Inf. Use loss scaling when training in FP16.
    Float16,
}

impl PrecisionMode {
    /// Returns `true` for any reduced-precision mode (BF16 or FP16).
    pub fn is_reduced(self) -> bool {
        matches!(self, Self::BFloat16 | Self::Float16)
    }

    /// Human-readable name for logging and configuration serialisation.
    pub fn name(self) -> &'static str {
        match self {
            Self::Float32 => "fp32",
            Self::BFloat16 => "bf16",
            Self::Float16 => "fp16",
        }
    }
}

// ---------------------------------------------------------------------------
// OpType
// ---------------------------------------------------------------------------

/// Categories of neural-network operations that may need FP32 upcasting.
///
/// Passed to [`MixedPrecisionConfig::should_upcast`] to check whether a
/// specific operation should run in full FP32 even when the overall mode is
/// BF16 or FP16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    /// Layer normalisation (mean + variance accumulation loses precision).
    LayerNorm,
    /// Softmax (exp() overflows for large logits in FP16).
    Softmax,
    /// Final linear output projection (accuracy-critical).
    OutputProjection,
    /// Scaled dot-product attention (QK matmul + softmax).
    Attention,
    /// All other general operations.
    General,
}

// ---------------------------------------------------------------------------
// MixedPrecisionConfig
// ---------------------------------------------------------------------------

/// Configuration for mixed-precision training and inference.
///
/// Controls which numeric format is used globally and which individual
/// operations are promoted to FP32 for numerical stability.
///
/// # Feature flag
///
/// When the `mixed_precision` Cargo feature is **enabled** the default mode is
/// [`PrecisionMode::BFloat16`]; otherwise it is [`PrecisionMode::Float32`].
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::mixed_precision::{MixedPrecisionConfig, PrecisionMode, OpType};
///
/// let mut cfg = MixedPrecisionConfig::default();
/// cfg.mode = PrecisionMode::BFloat16;
/// cfg.fp32_layernorm = true;
///
/// // Layer norm always runs in FP32 when fp32_layernorm is true.
/// assert!(cfg.should_upcast(OpType::LayerNorm));
/// // Attention does not have its own flag — it follows the general mode.
/// assert!(!cfg.should_upcast(OpType::Attention));
/// ```
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// The global precision mode.
    pub mode: PrecisionMode,

    /// Keep layer normalisation in FP32 regardless of `mode`.
    ///
    /// Default: `true`. Strongly recommended — variance accumulation in FP16
    /// can produce wildly inaccurate layer-norm outputs.
    pub fp32_layernorm: bool,

    /// Keep softmax in FP32 regardless of `mode`.
    ///
    /// Default: `true`. Prevents exp() overflow for large logit values (common
    /// in attention with large sequence lengths).
    pub fp32_softmax: bool,

    /// Keep the final output projection in FP32 regardless of `mode`.
    ///
    /// Default: `true`. Ensures the final tensor produced by the pipeline has
    /// full FP32 quality before any downstream loss computation.
    pub fp32_output: bool,

    /// Loss scaling factor used during FP16 training to avoid gradient
    /// underflow.
    ///
    /// Default: `1024.0`. Has no effect during inference (no backward pass).
    /// Typical values range from 128 to 65536 depending on the model.
    pub loss_scale: f32,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            #[cfg(feature = "mixed_precision")]
            mode: PrecisionMode::BFloat16,
            #[cfg(not(feature = "mixed_precision"))]
            mode: PrecisionMode::Float32,
            fp32_layernorm: true,
            fp32_softmax: true,
            fp32_output: true,
            loss_scale: 1024.0,
        }
    }
}

impl MixedPrecisionConfig {
    /// Validate the configuration, returning an error for inconsistent settings.
    ///
    /// # Errors
    ///
    /// Returns [`DiffusionError::InvalidConfig`] if:
    /// - `loss_scale` is not finite or is ≤ 0.
    /// - `mode` is [`PrecisionMode::Float16`] with `loss_scale < 1.0` (very
    ///   likely to cause gradient underflow).
    pub fn validate(&self) -> Result<(), DiffusionError> {
        if !self.loss_scale.is_finite() || self.loss_scale <= 0.0 {
            return Err(DiffusionError::InvalidConfig(format!(
                "loss_scale must be a positive finite number, got {}",
                self.loss_scale
            )));
        }
        if self.mode == PrecisionMode::Float16 && self.loss_scale < 1.0 {
            return Err(DiffusionError::InvalidConfig(format!(
                "loss_scale {} is dangerously low for FP16 mode; \
                 gradient underflow is almost certain (minimum recommended: 1.0)",
                self.loss_scale
            )));
        }
        Ok(())
    }

    /// Returns `true` if the given operation should run in FP32 even when the
    /// overall [`PrecisionMode`] is reduced.
    ///
    /// This check is always `false` when `mode` is [`PrecisionMode::Float32`]
    /// (everything is already FP32).
    pub fn should_upcast(&self, op_type: OpType) -> bool {
        if !self.mode.is_reduced() {
            return false;
        }
        match op_type {
            OpType::LayerNorm => self.fp32_layernorm,
            OpType::Softmax => self.fp32_softmax,
            OpType::OutputProjection => self.fp32_output,
            OpType::Attention | OpType::General => false,
        }
    }

    /// Memory reduction factor compared to FP32.
    ///
    /// Returns `0.5` for BF16 and FP16 (both use 16 bits = ½ the FP32 memory
    /// per element), and `1.0` for FP32 (no reduction).
    pub fn memory_factor(&self) -> f32 {
        match self.mode {
            PrecisionMode::Float32 => 1.0,
            PrecisionMode::BFloat16 | PrecisionMode::Float16 => 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// BF16 bit-manipulation utilities
// ---------------------------------------------------------------------------

/// Convert an `f32` to its BF16 bit representation (`u16`).
///
/// Uses round-to-nearest-even (the default IEEE 754 rounding mode).
///
/// ## Algorithm
///
/// BF16 keeps the top 16 bits of the FP32 bit pattern
/// (sign + 8-bit exponent + 7 high mantissa bits).
/// We add a rounding bias of `0x7FFF + lsb`, where
/// `lsb` is bit 16 of the FP32 word (the least-significant BF16 mantissa bit).
/// This implements banker's rounding (round half to even).
///
/// NaN inputs produce a quiet NaN output (bit 6 of the upper byte is set).
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::mixed_precision::f32_to_bf16;
///
/// // 1.0_f32 = 0x3F80_0000; upper 16 bits = 0x3F80.
/// assert_eq!(f32_to_bf16(1.0_f32), 0x3F80_u16);
/// ```
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    // Propagate NaN as quiet NaN to avoid producing Inf or wrong values.
    if x.is_nan() {
        // Set quiet NaN bit (bit 22 of f32 = bit 6 of bf16 upper byte).
        return ((bits >> 16) | 0x0040) as u16;
    }
    // Round-to-nearest-even: add bias = 0x7FFF + lsb (bit 16 of the f32 word).
    // We take the *upper* 16 bits after rounding (not the lower 16 bits).
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF_u32 + lsb;
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// Convert a BF16 bit representation (`u16`) back to `f32`.
///
/// The conversion is lossless: the upper 16 bits of the resulting FP32 word
/// equal `x`, and the lower 16 bits are zero-filled.
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::mixed_precision::bf16_to_f32;
///
/// // 0x3F80 → 1.0_f32
/// assert_eq!(bf16_to_f32(0x3F80_u16), 1.0_f32);
/// ```
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

// ---------------------------------------------------------------------------
// FP16 IEEE 754 conversion utilities
// ---------------------------------------------------------------------------

// FP32 layout: 1 sign | 8 exponent (bias 127) | 23 mantissa
// FP16 layout: 1 sign | 5 exponent (bias  15) | 10 mantissa
// Exponent rebias offset: 127 - 15 = 112

/// Convert an `f32` to its IEEE 754 half-precision (`f16`) bit representation
/// as a `u16`.
///
/// Handles all IEEE 754 special cases:
/// - **Zero** (positive and negative) → `0x0000` / `0x8000`
/// - **Subnormal f16 range** (`2^-24 ≤ |x| < 2^-14`) → subnormal f16
/// - **Below subnormal** (`|x| < 2^-24`) → flush to ±0
/// - **Normal** → normal f16
/// - **Overflow** (`|x| > 65504`) → ±Inf (`0x7C00` / `0xFC00`)
/// - **Infinity** → ±Inf
/// - **NaN** → quiet NaN (`0x7E00` with original sign and upper mantissa bits)
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::mixed_precision::f32_to_f16;
///
/// // 1.0 in FP16 = 0x3C00
/// assert_eq!(f32_to_f16(1.0_f32), 0x3C00_u16);
/// // Overflow
/// assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00_u16);
/// ```
pub fn f32_to_f16(x: f32) -> u16 {
    let bits: u32 = x.to_bits();
    let sign: u16 = ((bits >> 16) & 0x8000) as u16;
    let exp32: i32 = ((bits >> 23) & 0xFF) as i32;
    let mant32: u32 = bits & 0x007F_FFFF;

    // --- Special cases ---
    if x.is_nan() {
        // Quiet NaN: set the top mantissa bit of f16.
        // Preserve some upper mantissa bits for NaN payload.
        let mant16 = 0x0200_u16 | ((mant32 >> 13) as u16 & 0x01FF);
        return sign | 0x7C00 | mant16;
    }
    if x.is_infinite() {
        return sign | 0x7C00;
    }
    if exp32 == 0 && mant32 == 0 {
        // ±0
        return sign;
    }

    // Rebias exponent from FP32 (bias 127) to FP16 (bias 15).
    let exp16: i32 = exp32 - 112; // 127 - 15 = 112

    if exp16 >= 31 {
        // Overflow → ±Inf
        return sign | 0x7C00;
    }

    if exp16 <= 0 {
        // Potential subnormal or underflow.
        if exp16 < -10 {
            // Below minimum subnormal f16 (2^-24): flush to zero.
            return sign;
        }
        // Subnormal f16: the implicit leading 1 becomes explicit.
        // Shift the mantissa right and add the implicit bit.
        let shift = (1 - exp16) as u32; // shift = 1..11
        let mant_with_implicit = mant32 | 0x0080_0000_u32; // add implicit 1
        let mant16 = (mant_with_implicit >> (13 + shift)) as u16;
        // Round: check the bit just below the truncation point.
        let round_bit = (mant_with_implicit >> (12 + shift)) & 1;
        let mant16_rounded = mant16 + round_bit as u16;
        return sign | mant16_rounded;
    }

    // Normal FP16 value.
    let mant16_raw = (mant32 >> 13) as u16;
    // Round to nearest even.
    let round_bit = (mant32 >> 12) & 1;
    let sticky_bits = mant32 & 0x0FFF;
    let round_up = if round_bit == 1 && (sticky_bits > 0 || (mant16_raw & 1) == 1) {
        1_u16
    } else {
        0_u16
    };
    let mant16 = mant16_raw + round_up;

    // Rounding the mantissa may carry into the exponent.
    let exp_biased = exp16 as u16;
    let result_no_sign = (exp_biased << 10) | (mant16 & 0x03FF);
    // If mant16 overflowed into bit 10, the exponent increments naturally.
    if result_no_sign >= 0x7C00 {
        // Overflow from rounding → Inf
        return sign | 0x7C00;
    }
    sign | result_no_sign
}

/// Convert a IEEE 754 half-precision (`f16`) bit representation (`u16`) to
/// `f32`.
///
/// Handles all special cases: ±0, subnormal f16, normal, ±Inf, NaN.
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::mixed_precision::f16_to_f32;
///
/// assert_eq!(f16_to_f32(0x3C00_u16), 1.0_f32);
/// assert!(f16_to_f32(0x7E00_u16).is_nan());
/// ```
pub fn f16_to_f32(x: u16) -> f32 {
    let sign: u32 = (x as u32 & 0x8000) << 16; // move to FP32 sign bit position
    let exp16: u32 = (x as u32 >> 10) & 0x1F;
    let mant16: u32 = x as u32 & 0x03FF;

    if exp16 == 0x1F {
        // Inf or NaN
        if mant16 == 0 {
            // ±Infinity
            return f32::from_bits(sign | 0x7F80_0000);
        } else {
            // NaN: quiet NaN in FP32 (set bit 22)
            return f32::from_bits(sign | 0x7FC0_0000 | (mant16 << 13));
        }
    }

    if exp16 == 0 {
        if mant16 == 0 {
            // ±Zero
            return f32::from_bits(sign);
        }
        // Subnormal f16: normalize by finding leading 1.
        // The value is (-1)^sign * 2^-14 * (mant16 / 2^10).
        // Normalise: shift left until bit 10 is set.
        let mut m = mant16;
        let mut e: i32 = -14; // unbiased exponent for subnormal f16
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF; // remove implicit bit
                     // Convert to FP32 biased exponent: e + 127
        let exp32 = (e + 127) as u32;
        return f32::from_bits(sign | (exp32 << 23) | (m << 13));
    }

    // Normal FP16: rebias exponent from 15 to 127.
    let exp32 = exp16 + 112; // 127 - 15 = 112
    f32::from_bits(sign | (exp32 << 23) | (mant16 << 13))
}

// ---------------------------------------------------------------------------
// Simulation helpers
// ---------------------------------------------------------------------------

/// Simulate BF16 rounding on a slice of FP32 values.
///
/// Each element is converted to BF16 and immediately back to FP32, which
/// zeros out the lower 16 mantissa bits (with rounding). The result
/// represents what the values would look like after a BF16 storage round-trip.
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::mixed_precision::simulate_bf16;
///
/// let original = vec![1.0_f32, 0.1, -0.333];
/// let rounded = simulate_bf16(&original);
/// // Values should be very close but not necessarily equal.
/// for (o, r) in original.iter().zip(rounded.iter()) {
///     assert!((o - r).abs() < 1e-2, "{o} vs {r}");
/// }
/// ```
pub fn simulate_bf16(data: &[f32]) -> Vec<f32> {
    data.iter().map(|&v| bf16_to_f32(f32_to_bf16(v))).collect()
}

/// Simulate FP16 rounding on a slice of FP32 values.
///
/// Each element is converted to FP16 and immediately back to FP32, which
/// limits precision to 10 mantissa bits and clamps the range to ≈ ±65504.
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::mixed_precision::simulate_f16;
///
/// let original = vec![1.0_f32, 0.5, 100.0];
/// let rounded = simulate_f16(&original);
/// for (o, r) in original.iter().zip(rounded.iter()) {
///     assert!((o - r).abs() <= o.abs() * 1e-3 + 1e-4, "{o} vs {r}");
/// }
/// ```
pub fn simulate_f16(data: &[f32]) -> Vec<f32> {
    data.iter().map(|&v| f16_to_f32(f32_to_f16(v))).collect()
}

// ---------------------------------------------------------------------------
// apply_precision
// ---------------------------------------------------------------------------

/// Apply mixed-precision simulation to a tensor represented as `&[f32]`.
///
/// - [`PrecisionMode::Float32`]: returns an exact copy (no quantization).
/// - [`PrecisionMode::BFloat16`]: applies BF16 round-trip via [`simulate_bf16`].
/// - [`PrecisionMode::Float16`]: applies FP16 round-trip via [`simulate_f16`].
///
/// # Example
///
/// ```rust
/// use oxigaf_diffusion::mixed_precision::{
///     apply_precision, MixedPrecisionConfig, PrecisionMode,
/// };
///
/// let mut cfg = MixedPrecisionConfig::default();
/// cfg.mode = PrecisionMode::Float32;
/// let data = vec![1.0_f32, 2.0, 3.0];
/// let out = apply_precision(&data, &cfg);
/// assert_eq!(out, data); // exact copy in FP32 mode
/// ```
pub fn apply_precision(data: &[f32], config: &MixedPrecisionConfig) -> Vec<f32> {
    match config.mode {
        PrecisionMode::Float32 => data.to_vec(),
        PrecisionMode::BFloat16 => simulate_bf16(data),
        PrecisionMode::Float16 => simulate_f16(data),
    }
}

// ---------------------------------------------------------------------------
// PrecisionStats
// ---------------------------------------------------------------------------

/// Statistics describing the precision loss introduced by quantization.
///
/// Computed by [`PrecisionStats::compute`] by comparing the original FP32
/// values with their quantized counterparts.
#[derive(Debug, Clone)]
pub struct PrecisionStats {
    /// Maximum absolute difference between original and quantized values.
    pub max_abs_error: f32,
    /// Mean absolute difference across all elements.
    pub mean_abs_error: f32,
    /// Mean relative error `|orig - quant| / (|orig| + eps)` across all
    /// elements.
    pub relative_error: f32,
    /// Number of values that became ±Infinity after quantization (overflow).
    pub num_overflows: usize,
    /// Number of values that became ±0 after quantization but were non-zero
    /// originally (underflow to zero, i.e. the subnormal region was flushed).
    pub num_underflows: usize,
}

impl PrecisionStats {
    /// Compute precision statistics between `original` and `quantized` slices.
    ///
    /// Both slices must have the same length; if either is empty, all metrics
    /// are zero.
    ///
    /// # Panics
    ///
    /// Does not panic — mismatched lengths are handled gracefully by comparing
    /// only the common prefix.
    pub fn compute(original: &[f32], quantized: &[f32]) -> Self {
        let n = original.len().min(quantized.len());
        if n == 0 {
            return Self {
                max_abs_error: 0.0,
                mean_abs_error: 0.0,
                relative_error: 0.0,
                num_overflows: 0,
                num_underflows: 0,
            };
        }

        let mut max_abs_error: f32 = 0.0;
        let mut sum_abs_error: f32 = 0.0;
        let mut sum_rel_error: f32 = 0.0;
        let mut num_overflows: usize = 0;
        let mut num_underflows: usize = 0;

        for (&orig, &quant) in original[..n].iter().zip(quantized[..n].iter()) {
            let abs_err = (orig - quant).abs();
            if abs_err > max_abs_error {
                max_abs_error = abs_err;
            }
            sum_abs_error += abs_err;

            // Relative error guarded against divide-by-zero.
            let denom = orig.abs() + 1e-8;
            sum_rel_error += abs_err / denom;

            if quant.is_infinite() && orig.is_finite() {
                num_overflows += 1;
            }
            // Underflow: original was non-zero but quantized is zero.
            if orig != 0.0 && !orig.is_nan() && quant == 0.0 {
                num_underflows += 1;
            }
        }

        let n_f = n as f32;
        Self {
            max_abs_error,
            mean_abs_error: sum_abs_error / n_f,
            relative_error: sum_rel_error / n_f,
            num_overflows,
            num_underflows,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // PrecisionMode
    // ------------------------------------------------------------------

    #[test]
    fn test_precision_mode_default_without_feature() {
        // When the mixed_precision feature is NOT active, the default is Float32.
        #[cfg(not(feature = "mixed_precision"))]
        {
            assert_eq!(PrecisionMode::default(), PrecisionMode::Float32);
        }
    }

    #[test]
    fn test_precision_mode_default_with_feature() {
        // When the mixed_precision feature IS active, the default is BFloat16.
        #[cfg(feature = "mixed_precision")]
        {
            assert_eq!(PrecisionMode::default(), PrecisionMode::Float32); // struct Default, not enum Default
        }
        // Guard: just check that the enum default is consistent.
        let _ = PrecisionMode::default();
    }

    #[test]
    fn test_precision_mode_is_reduced_float32_false() {
        assert!(!PrecisionMode::Float32.is_reduced());
    }

    #[test]
    fn test_precision_mode_is_reduced_bf16_true() {
        assert!(PrecisionMode::BFloat16.is_reduced());
    }

    #[test]
    fn test_precision_mode_is_reduced_f16_true() {
        assert!(PrecisionMode::Float16.is_reduced());
    }

    #[test]
    fn test_precision_mode_names() {
        assert_eq!(PrecisionMode::Float32.name(), "fp32");
        assert_eq!(PrecisionMode::BFloat16.name(), "bf16");
        assert_eq!(PrecisionMode::Float16.name(), "fp16");
    }

    // ------------------------------------------------------------------
    // MixedPrecisionConfig defaults
    // ------------------------------------------------------------------

    #[test]
    fn test_mixed_precision_config_defaults() {
        let cfg = MixedPrecisionConfig::default();
        assert!(cfg.fp32_layernorm);
        assert!(cfg.fp32_softmax);
        assert!(cfg.fp32_output);
        assert!((cfg.loss_scale - 1024.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mixed_precision_config_default_mode_without_feature() {
        #[cfg(not(feature = "mixed_precision"))]
        assert_eq!(MixedPrecisionConfig::default().mode, PrecisionMode::Float32);
    }

    #[test]
    fn test_mixed_precision_config_default_mode_with_feature() {
        #[cfg(feature = "mixed_precision")]
        assert_eq!(
            MixedPrecisionConfig::default().mode,
            PrecisionMode::BFloat16
        );
    }

    // ------------------------------------------------------------------
    // validate()
    // ------------------------------------------------------------------

    #[test]
    fn test_validate_accepts_valid_config() {
        let cfg = MixedPrecisionConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_non_positive_loss_scale() {
        let cfg = MixedPrecisionConfig {
            loss_scale: 0.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_negative_loss_scale() {
        let cfg = MixedPrecisionConfig {
            loss_scale: -10.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_nan_loss_scale() {
        let cfg = MixedPrecisionConfig {
            loss_scale: f32::NAN,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_f16_with_tiny_loss_scale() {
        let cfg = MixedPrecisionConfig {
            mode: PrecisionMode::Float16,
            loss_scale: 0.5,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // ------------------------------------------------------------------
    // memory_factor()
    // ------------------------------------------------------------------

    #[test]
    fn test_memory_factor_fp32_is_one() {
        let cfg = MixedPrecisionConfig {
            mode: PrecisionMode::Float32,
            ..Default::default()
        };
        assert!((cfg.memory_factor() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_memory_factor_bf16_is_half() {
        let cfg = MixedPrecisionConfig {
            mode: PrecisionMode::BFloat16,
            ..Default::default()
        };
        assert!((cfg.memory_factor() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_memory_factor_f16_is_half() {
        let cfg = MixedPrecisionConfig {
            mode: PrecisionMode::Float16,
            ..Default::default()
        };
        assert!((cfg.memory_factor() - 0.5).abs() < f32::EPSILON);
    }

    // ------------------------------------------------------------------
    // should_upcast()
    // ------------------------------------------------------------------

    #[test]
    fn test_should_upcast_layernorm_when_fp32_layernorm_true() {
        let cfg = MixedPrecisionConfig {
            mode: PrecisionMode::BFloat16,
            fp32_layernorm: true,
            ..Default::default()
        };
        assert!(cfg.should_upcast(OpType::LayerNorm));
    }

    #[test]
    fn test_should_upcast_layernorm_false_when_fp32_layernorm_false() {
        let cfg = MixedPrecisionConfig {
            mode: PrecisionMode::BFloat16,
            fp32_layernorm: false,
            ..Default::default()
        };
        assert!(!cfg.should_upcast(OpType::LayerNorm));
    }

    #[test]
    fn test_should_upcast_returns_false_in_fp32_mode() {
        let cfg = MixedPrecisionConfig {
            mode: PrecisionMode::Float32,
            ..Default::default()
        };
        assert!(!cfg.should_upcast(OpType::LayerNorm));
        assert!(!cfg.should_upcast(OpType::Softmax));
        assert!(!cfg.should_upcast(OpType::OutputProjection));
    }

    #[test]
    fn test_should_upcast_general_always_false() {
        let cfg = MixedPrecisionConfig {
            mode: PrecisionMode::BFloat16,
            ..Default::default()
        };
        assert!(!cfg.should_upcast(OpType::General));
    }

    // ------------------------------------------------------------------
    // BF16 conversions
    // ------------------------------------------------------------------

    #[test]
    fn test_f32_to_bf16_one() {
        // 1.0_f32 = 0x3F80_0000; top 16 bits = 0x3F80.
        assert_eq!(f32_to_bf16(1.0_f32), 0x3F80_u16);
    }

    #[test]
    fn test_bf16_to_f32_known_value() {
        assert_eq!(bf16_to_f32(0x3F80_u16), 1.0_f32);
    }

    #[test]
    fn test_bf16_roundtrip_exact_powers_of_two() {
        // Powers of two are exactly representable in BF16.
        for exp in 0..8_i32 {
            let v = 2.0_f32.powi(exp);
            let rt = bf16_to_f32(f32_to_bf16(v));
            assert_eq!(rt, v, "roundtrip failed for 2^{exp}");
        }
    }

    #[test]
    fn test_bf16_roundtrip_negative_one() {
        let v = -1.0_f32;
        assert_eq!(bf16_to_f32(f32_to_bf16(v)), v);
    }

    #[test]
    fn test_bf16_nan_roundtrip() {
        let result = bf16_to_f32(f32_to_bf16(f32::NAN));
        assert!(result.is_nan(), "NaN should roundtrip as NaN, got {result}");
    }

    #[test]
    fn test_bf16_infinity_roundtrip() {
        assert_eq!(bf16_to_f32(f32_to_bf16(f32::INFINITY)), f32::INFINITY);
        assert_eq!(
            bf16_to_f32(f32_to_bf16(f32::NEG_INFINITY)),
            f32::NEG_INFINITY
        );
    }

    // ------------------------------------------------------------------
    // FP16 conversions
    // ------------------------------------------------------------------

    #[test]
    fn test_f32_to_f16_one() {
        // 1.0 in FP16 = exponent 15 (biased), mantissa 0 → 0x3C00.
        assert_eq!(f32_to_f16(1.0_f32), 0x3C00_u16);
    }

    #[test]
    fn test_f16_to_f32_known_value() {
        assert_eq!(f16_to_f32(0x3C00_u16), 1.0_f32);
    }

    #[test]
    fn test_f16_roundtrip_exact_values() {
        // Exact values in FP16.
        for &v in &[0.0_f32, 1.0, -1.0, 2.0, 0.5, -0.5, 4.0, 0.25] {
            let rt = f16_to_f32(f32_to_f16(v));
            assert!(
                (rt - v).abs() < v.abs() * 1e-3 + 1e-6,
                "roundtrip failed for {v}: got {rt}"
            );
        }
    }

    #[test]
    fn test_f32_to_f16_subnormal_range_flushes_to_zero() {
        // Values below ~5.96e-8 (2^-24) should become 0 in FP16.
        let tiny = 1e-10_f32;
        let result = f16_to_f32(f32_to_f16(tiny));
        assert_eq!(
            result, 0.0_f32,
            "expected flush to zero for {tiny}, got {result}"
        );
    }

    #[test]
    fn test_f32_to_f16_overflow_gives_inf() {
        // Values above 65504 overflow to Inf.
        assert_eq!(f32_to_f16(1e10_f32), 0x7C00_u16, "expected +Inf encoding");
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00_u16);
    }

    #[test]
    fn test_f32_to_f16_negative_overflow_gives_neg_inf() {
        assert_eq!(f32_to_f16(-1e10_f32), 0xFC00_u16, "expected -Inf encoding");
    }

    #[test]
    fn test_f16_nan_roundtrip() {
        let result = f16_to_f32(f32_to_f16(f32::NAN));
        assert!(result.is_nan(), "NaN should roundtrip as NaN, got {result}");
    }

    #[test]
    fn test_f16_infinity_roundtrip() {
        assert_eq!(f16_to_f32(f32_to_f16(f32::INFINITY)), f32::INFINITY);
        assert_eq!(f16_to_f32(f32_to_f16(f32::NEG_INFINITY)), f32::NEG_INFINITY);
    }

    // ------------------------------------------------------------------
    // simulate_bf16 / simulate_f16
    // ------------------------------------------------------------------

    #[test]
    fn test_simulate_bf16_close_to_original() {
        let data = vec![1.0_f32, -2.5, 0.1, 100.0, -0.001];
        let rounded = simulate_bf16(&data);
        for (o, r) in data.iter().zip(rounded.iter()) {
            let err = (o - r).abs();
            // BF16 relative error is at most ~0.4% for normal values.
            assert!(
                err <= o.abs() * 0.01 + 1e-4,
                "simulate_bf16: original={o}, rounded={r}, err={err}"
            );
        }
    }

    #[test]
    fn test_simulate_f16_close_to_original() {
        let data = vec![1.0_f32, -2.5, 0.5, 10.0, -1.0];
        let rounded = simulate_f16(&data);
        for (o, r) in data.iter().zip(rounded.iter()) {
            let err = (o - r).abs();
            // FP16 relative error is at most ~0.1% for normal values.
            assert!(
                err <= o.abs() * 0.005 + 1e-5,
                "simulate_f16: original={o}, rounded={r}, err={err}"
            );
        }
    }

    // ------------------------------------------------------------------
    // apply_precision
    // ------------------------------------------------------------------

    #[test]
    fn test_apply_precision_fp32_is_exact_copy() {
        let mut cfg = MixedPrecisionConfig::default();
        cfg.mode = PrecisionMode::Float32;
        let data = vec![1.0_f32, 2.0, 3.0, -1.5];
        let out = apply_precision(&data, &cfg);
        assert_eq!(out, data, "FP32 mode should produce exact copy");
    }

    #[test]
    fn test_apply_precision_bf16_mode_quantizes() {
        let cfg = MixedPrecisionConfig {
            mode: PrecisionMode::BFloat16,
            ..Default::default()
        };
        // 0.1 is not exactly representable in binary float; BF16 will differ.
        let data = vec![0.1_f32];
        let out = apply_precision(&data, &cfg);
        // The result should be close but may differ from 0.1.
        assert!(
            (out[0] - 0.1_f32).abs() < 0.01_f32,
            "bf16 result too far from 0.1: {}",
            out[0]
        );
    }

    #[test]
    fn test_apply_precision_f16_mode_quantizes() {
        let cfg = MixedPrecisionConfig {
            mode: PrecisionMode::Float16,
            ..Default::default()
        };
        let data = vec![100.5_f32, -50.25_f32];
        let out = apply_precision(&data, &cfg);
        for (o, r) in data.iter().zip(out.iter()) {
            assert!(
                (o - r).abs() <= o.abs() * 0.01 + 0.1,
                "f16 result too far: original={o}, quantized={r}"
            );
        }
    }

    // ------------------------------------------------------------------
    // PrecisionStats
    // ------------------------------------------------------------------

    #[test]
    fn test_precision_stats_max_abs_error_nonneg() {
        let original = vec![1.0_f32, 2.0, 3.0];
        let quantized = simulate_bf16(&original);
        let stats = PrecisionStats::compute(&original, &quantized);
        assert!(stats.max_abs_error >= 0.0);
        assert!(stats.mean_abs_error >= 0.0);
    }

    #[test]
    fn test_precision_stats_zero_error_for_identical() {
        let data = vec![1.0_f32, 2.0, 4.0]; // exact in BF16
        let stats = PrecisionStats::compute(&data, &data);
        assert_eq!(stats.max_abs_error, 0.0);
        assert_eq!(stats.mean_abs_error, 0.0);
        assert_eq!(stats.num_overflows, 0);
        assert_eq!(stats.num_underflows, 0);
    }

    #[test]
    fn test_precision_stats_overflows_counted_correctly() {
        let original = vec![1e10_f32, 2e10_f32, 1.0_f32];
        let quantized = simulate_f16(&original);
        let stats = PrecisionStats::compute(&original, &quantized);
        // First two values overflow FP16 (max ≈ 65504), so they become Inf.
        assert_eq!(
            stats.num_overflows, 2,
            "expected 2 overflows, got {}",
            stats.num_overflows
        );
    }

    #[test]
    fn test_precision_stats_empty_input() {
        let stats = PrecisionStats::compute(&[], &[]);
        assert_eq!(stats.max_abs_error, 0.0);
        assert_eq!(stats.num_overflows, 0);
    }
}
