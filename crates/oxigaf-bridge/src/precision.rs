//! Precision conversion utilities
//!
//! This module handles conversion between different floating-point precisions
//! (FP32, FP16, BF16) with configurable per-layer precision.

use crate::error::{BridgeError, Result};
use half::{bf16, f16};
use std::collections::BTreeMap;

/// Floating-point precision types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// 32-bit floating point
    FP32,
    /// 16-bit floating point (IEEE 754)
    FP16,
    /// Brain Float 16 (truncated FP32)
    BF16,
}

impl Precision {
    /// Get the byte size of this precision
    pub fn byte_size(&self) -> usize {
        match self {
            Precision::FP32 => 4,
            Precision::FP16 => 2,
            Precision::BF16 => 2,
        }
    }

    /// Get the name of this precision
    pub fn name(&self) -> &'static str {
        match self {
            Precision::FP32 => "FP32",
            Precision::FP16 => "FP16",
            Precision::BF16 => "BF16",
        }
    }
}

/// Configuration for precision conversion
pub struct PrecisionConfig {
    default_precision: Precision,
    /// Patterns are stored in a `BTreeMap` purely so iteration order is
    /// deterministic (lexicographic by pattern) regardless of insertion
    /// order or the process's hash-seed -- see [`PrecisionConfig::get_layer_precision`]
    /// for the actual match-selection rule.
    layer_precisions: BTreeMap<String, Precision>,
    keep_normalization_fp32: bool,
}

impl PrecisionConfig {
    /// Create a new precision configuration with FP32 default
    pub fn new() -> Self {
        Self {
            default_precision: Precision::FP32,
            layer_precisions: BTreeMap::new(),
            keep_normalization_fp32: true,
        }
    }

    /// Set the default precision for all layers
    pub fn set_default_precision(&mut self, precision: Precision) {
        self.default_precision = precision;
    }

    /// Get the default precision
    pub fn default_precision(&self) -> Precision {
        self.default_precision
    }

    /// Set precision for a specific layer pattern
    ///
    /// # Arguments
    ///
    /// * `pattern` - Layer name pattern (e.g., "normalization", "attention")
    /// * `precision` - Precision to use for matching layers
    pub fn set_layer_precision(&mut self, pattern: impl Into<String>, precision: Precision) {
        self.layer_precisions.insert(pattern.into(), precision);
    }

    /// Get the precision for a specific layer
    ///
    /// If more than one registered pattern matches `layer_name`, the
    /// *longest* pattern wins (the most specific match); ties are broken by
    /// picking the lexicographically smallest pattern string. This is fully
    /// deterministic regardless of the order patterns were registered in,
    /// unlike a plain first-match-wins scan over a `HashMap`.
    pub fn get_layer_precision(&self, layer_name: &str) -> Precision {
        let mut best: Option<(&str, Precision)> = None;
        for (pattern, precision) in &self.layer_precisions {
            if !layer_name.contains(pattern.as_str()) {
                continue;
            }
            let is_better = match best {
                None => true,
                Some((best_pattern, _)) => {
                    pattern.len() > best_pattern.len()
                        || (pattern.len() == best_pattern.len() && pattern.as_str() < best_pattern)
                }
            };
            if is_better {
                best = Some((pattern.as_str(), *precision));
            }
        }

        if let Some((_, precision)) = best {
            return precision;
        }

        // Keep normalization layers in FP32 if configured. "norm" alone
        // covers "layernorm"/"batchnorm"/"groupnorm" (each contains "norm"
        // as a substring), so checking those separately would be redundant.
        if self.keep_normalization_fp32 && layer_name.contains("norm") {
            return Precision::FP32;
        }

        self.default_precision
    }

    /// Set whether to keep normalization layers in FP32
    pub fn set_keep_normalization_fp32(&mut self, keep: bool) {
        self.keep_normalization_fp32 = keep;
    }
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert f32 slice to f16 bytes
pub fn f32_to_f16_bytes(data: &[f32]) -> Vec<u8> {
    let f16_data: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
    f16_data.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Convert f16 bytes to f32 slice
pub fn f16_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(BridgeError::PrecisionConversion(
            "Invalid byte length for f16 data".to_string(),
        ));
    }

    let f16_data: Vec<f16> = bytes
        .chunks_exact(2)
        .map(|chunk| {
            let array: [u8; 2] = [chunk[0], chunk[1]];
            f16::from_le_bytes(array)
        })
        .collect();

    Ok(f16_data.iter().map(|x| x.to_f32()).collect())
}

/// Convert f32 slice to bf16 bytes
pub fn f32_to_bf16_bytes(data: &[f32]) -> Vec<u8> {
    let bf16_data: Vec<bf16> = data.iter().map(|&x| bf16::from_f32(x)).collect();
    bf16_data.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Convert bf16 bytes to f32 slice
pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(2) {
        return Err(BridgeError::PrecisionConversion(
            "Invalid byte length for bf16 data".to_string(),
        ));
    }

    let bf16_data: Vec<bf16> = bytes
        .chunks_exact(2)
        .map(|chunk| {
            let array: [u8; 2] = [chunk[0], chunk[1]];
            bf16::from_le_bytes(array)
        })
        .collect();

    Ok(bf16_data.iter().map(|x| x.to_f32()).collect())
}

/// Maps a `safetensors::Dtype` to the corresponding [`Precision`], if it is
/// one of the three floating-point precisions this crate converts between.
/// Returns `None` for every other dtype (integers, bool, and other float
/// widths `safetensors` supports but this crate does not convert -- those
/// are passed through by their callers unchanged instead).
pub fn float_precision_of(dtype: safetensors::Dtype) -> Option<Precision> {
    match dtype {
        safetensors::Dtype::F32 => Some(Precision::FP32),
        safetensors::Dtype::F16 => Some(Precision::FP16),
        safetensors::Dtype::BF16 => Some(Precision::BF16),
        _ => None,
    }
}

/// Maps a [`Precision`] to its corresponding `safetensors::Dtype`.
pub fn dtype_of(precision: Precision) -> safetensors::Dtype {
    match precision {
        Precision::FP32 => safetensors::Dtype::F32,
        Precision::FP16 => safetensors::Dtype::F16,
        Precision::BF16 => safetensors::Dtype::BF16,
    }
}

/// Convert f32 data to specified precision.
///
/// Returns the encoded bytes together with a count of elements that were
/// finite in `data` but became non-finite (`+/-inf`) after the conversion --
/// e.g. values whose magnitude exceeds FP16's ~65504 cap. `half::f16`/`bf16`
/// saturate silently on overflow, and the encoded bytes alone carry no trace
/// of that: a downstream NaN/Inf checker (see
/// `validation::validate_converted_checkpoint`) cannot otherwise tell such a
/// saturated value apart from one that was already `inf` in the source.
/// Callers should treat a non-zero count as a conversion warning.
pub fn convert_precision(data: &[f32], precision: Precision) -> (Vec<u8>, usize) {
    let bytes = match precision {
        Precision::FP32 => return (data.iter().flat_map(|x| x.to_le_bytes()).collect(), 0),
        Precision::FP16 => f32_to_f16_bytes(data),
        Precision::BF16 => f32_to_bf16_bytes(data),
    };
    let saturated = count_saturated(data, &bytes, precision);
    (bytes, saturated)
}

/// Counts elements that were finite in `original` but decode back to a
/// non-finite value from `encoded` at `precision`. Used by
/// [`convert_precision`] to report silent overflow saturation.
fn count_saturated(original: &[f32], encoded: &[u8], precision: Precision) -> usize {
    match bytes_to_f32(encoded, precision) {
        Ok(decoded) => original
            .iter()
            .zip(decoded.iter())
            .filter(|(&orig, &conv)| orig.is_finite() && !conv.is_finite())
            .count(),
        // Should not happen: `encoded` was just produced by this module's
        // own encoder, so it always has a length valid for `precision`.
        Err(_) => 0,
    }
}

/// Convert bytes back to f32 based on precision
pub fn bytes_to_f32(bytes: &[u8], precision: Precision) -> Result<Vec<f32>> {
    match precision {
        Precision::FP32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err(BridgeError::PrecisionConversion(
                    "Invalid byte length for f32 data".to_string(),
                ));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| {
                    let array: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    f32::from_le_bytes(array)
                })
                .collect())
        }
        Precision::FP16 => f16_bytes_to_f32(bytes),
        Precision::BF16 => bf16_bytes_to_f32(bytes),
    }
}

/// Validate round-trip conversion error.
///
/// Uses a *relative* threshold (`diff <= max_error * orig.abs().max(1.0)`)
/// rather than a fixed absolute one, since an absolute threshold is the
/// wrong metric across weights of different magnitude: a `max_error` tight
/// enough for values near zero is spuriously strict for large ones, and one
/// loose enough for large values lets small ones drift arbitrarily.
pub fn validate_conversion(original: &[f32], converted: &[f32], max_error: f32) -> Result<()> {
    if original.len() != converted.len() {
        return Err(BridgeError::Validation(format!(
            "Length mismatch: original {}, converted {}",
            original.len(),
            converted.len()
        )));
    }

    let mut max_diff = 0.0f32;
    for (i, (&orig, &conv)) in original.iter().zip(converted.iter()).enumerate() {
        let diff = (orig - conv).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        let allowed = max_error * orig.abs().max(1.0);
        if diff > allowed {
            return Err(BridgeError::Validation(format!(
                "Conversion error too large at index {}: original {}, converted {}, diff {} (allowed {})",
                i, orig, conv, diff, allowed
            )));
        }
    }

    tracing::debug!("Validation passed. Max diff: {}", max_diff);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_precision_byte_size() {
        assert_eq!(Precision::FP32.byte_size(), 4);
        assert_eq!(Precision::FP16.byte_size(), 2);
        assert_eq!(Precision::BF16.byte_size(), 2);
    }

    #[test]
    fn test_f32_to_f16_round_trip() {
        let original = vec![1.0f32, 2.5, -3.7, 0.0, 100.0];
        let bytes = f32_to_f16_bytes(&original);
        let converted =
            f16_bytes_to_f32(&bytes).expect("test: precision conversion should succeed");

        assert_eq!(original.len(), converted.len());
        for (&orig, &conv) in original.iter().zip(converted.iter()) {
            // FP16 has less precision, so allow some error
            assert_relative_eq!(orig, conv, epsilon = 0.01, max_relative = 0.01);
        }
    }

    #[test]
    fn test_f32_to_bf16_round_trip() {
        let original = vec![1.0f32, 2.5, -3.7, 0.0, 100.0];
        let bytes = f32_to_bf16_bytes(&original);
        let converted =
            bf16_bytes_to_f32(&bytes).expect("test: precision conversion should succeed");

        assert_eq!(original.len(), converted.len());
        for (&orig, &conv) in original.iter().zip(converted.iter()) {
            // BF16 has less precision in mantissa, so allow some error
            assert_relative_eq!(orig, conv, epsilon = 0.01, max_relative = 0.01);
        }
    }

    #[test]
    fn test_precision_config_default() {
        let config = PrecisionConfig::new();
        assert_eq!(config.default_precision(), Precision::FP32);

        // Normalization layers should stay FP32
        assert_eq!(
            config.get_layer_precision("model.norm.weight"),
            Precision::FP32
        );
        assert_eq!(
            config.get_layer_precision("model.layernorm.bias"),
            Precision::FP32
        );
    }

    #[test]
    fn test_precision_config_custom() {
        let mut config = PrecisionConfig::new();
        config.set_default_precision(Precision::FP16);
        config.set_layer_precision("attention", Precision::FP32);

        assert_eq!(
            config.get_layer_precision("model.layer.weight"),
            Precision::FP16
        );
        assert_eq!(
            config.get_layer_precision("model.attention.weight"),
            Precision::FP32
        );

        // Normalization should still be FP32
        assert_eq!(
            config.get_layer_precision("model.norm.weight"),
            Precision::FP32
        );
    }

    #[test]
    fn test_layer_precision_longest_match_wins_deterministically() {
        // Regression test: overlapping patterns used to resolve via HashMap
        // iteration order, which is randomized per-process. The longest
        // (most specific) match must win, and the result must be the same
        // every time regardless of registration order.
        let mut config = PrecisionConfig::new();
        config.set_default_precision(Precision::FP32);
        config.set_layer_precision("attn", Precision::FP16);
        config.set_layer_precision("attn.to_q", Precision::BF16);

        assert_eq!(
            config.get_layer_precision("down_blocks.0.attn.to_q.weight"),
            Precision::BF16,
            "the more specific pattern 'attn.to_q' should win over 'attn'"
        );
        assert_eq!(
            config.get_layer_precision("down_blocks.0.attn.to_k.weight"),
            Precision::FP16,
            "only 'attn' matches 'to_k', so it should apply"
        );

        // Same patterns, registered in the opposite order: result must be identical.
        let mut config2 = PrecisionConfig::new();
        config2.set_default_precision(Precision::FP32);
        config2.set_layer_precision("attn.to_q", Precision::BF16);
        config2.set_layer_precision("attn", Precision::FP16);
        assert_eq!(
            config2.get_layer_precision("down_blocks.0.attn.to_q.weight"),
            Precision::BF16
        );
    }

    #[test]
    fn test_convert_precision_fp32() {
        let data = vec![1.0f32, 2.0, 3.0];
        let (bytes, saturated) = convert_precision(&data, Precision::FP32);
        assert_eq!(saturated, 0);
        let converted = bytes_to_f32(&bytes, Precision::FP32)
            .expect("test: precision conversion should succeed");

        assert_eq!(data, converted);
    }

    #[test]
    fn test_convert_precision_fp16() {
        let data = vec![1.0f32, 2.0, 3.0];
        let (bytes, saturated) = convert_precision(&data, Precision::FP16);
        assert_eq!(saturated, 0);
        let converted = bytes_to_f32(&bytes, Precision::FP16)
            .expect("test: precision conversion should succeed");

        for (&orig, &conv) in data.iter().zip(converted.iter()) {
            assert_relative_eq!(orig, conv, epsilon = 0.01);
        }
    }

    #[test]
    fn test_convert_precision_fp16_reports_overflow_saturation() {
        // Regression test: values outside FP16's ~65504 range used to
        // silently become +/-inf with no way for a caller to detect it.
        let data = vec![1.0f32, 100_000.0, -200_000.0, 2.0];
        let (bytes, saturated) = convert_precision(&data, Precision::FP16);
        assert_eq!(
            saturated, 2,
            "the two out-of-range values should be flagged"
        );

        let decoded = bytes_to_f32(&bytes, Precision::FP16).expect("test: decode should succeed");
        assert!(decoded[1].is_infinite() && decoded[1].is_sign_positive());
        assert!(decoded[2].is_infinite() && decoded[2].is_sign_negative());
        assert!(decoded[0].is_finite());
        assert!(decoded[3].is_finite());
    }

    #[test]
    fn test_validate_conversion() {
        let original = vec![1.0f32, 2.0, 3.0];
        let good = vec![1.0f32, 2.0, 3.0];
        let bad = vec![1.5f32, 2.5, 3.5];

        assert!(validate_conversion(&original, &good, 1e-6).is_ok());
        assert!(validate_conversion(&original, &bad, 1e-6).is_err());
        assert!(validate_conversion(&original, &bad, 1.0).is_ok());
    }

    #[test]
    fn test_validate_conversion_uses_relative_threshold() {
        // A fixed absolute threshold is the wrong metric across weights of
        // different magnitude: the same `max_error` must scale with the
        // value being compared, not apply uniformly.
        let original = vec![1000.0f32];
        let converted = vec![1000.5f32]; // 0.05% relative error
                                         // Absolute diff (0.5) would fail a naive 0.01 absolute threshold,
                                         // but 0.05% relative error should pass a 0.001 (0.1%) relative one.
        assert!(validate_conversion(&original, &converted, 0.001).is_ok());

        let tiny_original = vec![0.001f32];
        let tiny_converted = vec![0.006f32]; // large relative error, small absolute one
        assert!(validate_conversion(&tiny_original, &tiny_converted, 0.001).is_err());
    }

    #[test]
    fn test_float_precision_of_and_dtype_of_round_trip() {
        for precision in [Precision::FP32, Precision::FP16, Precision::BF16] {
            assert_eq!(float_precision_of(dtype_of(precision)), Some(precision));
        }
        assert_eq!(float_precision_of(safetensors::Dtype::I64), None);
        assert_eq!(float_precision_of(safetensors::Dtype::BOOL), None);
    }

    #[test]
    fn test_invalid_byte_length() {
        let invalid_bytes = vec![0u8, 1, 2]; // 3 bytes, not divisible by 2
        assert!(f16_bytes_to_f32(&invalid_bytes).is_err());
        assert!(bf16_bytes_to_f32(&invalid_bytes).is_err());
    }
}
