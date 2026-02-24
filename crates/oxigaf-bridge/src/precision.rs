//! Precision conversion utilities
//!
//! This module handles conversion between different floating-point precisions
//! (FP32, FP16, BF16) with configurable per-layer precision.

use crate::error::{BridgeError, Result};
use half::{bf16, f16};
use std::collections::HashMap;

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
    layer_precisions: HashMap<String, Precision>,
    keep_normalization_fp32: bool,
}

impl PrecisionConfig {
    /// Create a new precision configuration with FP32 default
    pub fn new() -> Self {
        Self {
            default_precision: Precision::FP32,
            layer_precisions: HashMap::new(),
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
    pub fn get_layer_precision(&self, layer_name: &str) -> Precision {
        // Check if layer matches any pattern
        for (pattern, precision) in &self.layer_precisions {
            if layer_name.contains(pattern) {
                return *precision;
            }
        }

        // Keep normalization layers in FP32 if configured
        if self.keep_normalization_fp32
            && (layer_name.contains("norm")
                || layer_name.contains("layernorm")
                || layer_name.contains("batchnorm")
                || layer_name.contains("groupnorm"))
        {
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

/// Convert f32 data to specified precision
pub fn convert_precision(data: &[f32], precision: Precision) -> Vec<u8> {
    match precision {
        Precision::FP32 => data.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Precision::FP16 => f32_to_f16_bytes(data),
        Precision::BF16 => f32_to_bf16_bytes(data),
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

/// Validate round-trip conversion error
pub fn validate_conversion(original: &[f32], converted: &[f32], max_error: f32) -> Result<()> {
    if original.len() != converted.len() {
        return Err(BridgeError::Validation(format!(
            "Length mismatch: original {}, converted {}",
            original.len(),
            converted.len()
        )));
    }

    let mut max_diff = 0.0f32;
    for (_i, (&orig, &conv)) in original.iter().zip(converted.iter()).enumerate() {
        let diff = (orig - conv).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > max_error {
            return Err(BridgeError::Validation(format!(
                "Conversion error too large at index {}: original {}, converted {}, diff {}",
                _i, orig, conv, diff
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
    fn test_convert_precision_fp32() {
        let data = vec![1.0f32, 2.0, 3.0];
        let bytes = convert_precision(&data, Precision::FP32);
        let converted = bytes_to_f32(&bytes, Precision::FP32)
            .expect("test: precision conversion should succeed");

        assert_eq!(data, converted);
    }

    #[test]
    fn test_convert_precision_fp16() {
        let data = vec![1.0f32, 2.0, 3.0];
        let bytes = convert_precision(&data, Precision::FP16);
        let converted = bytes_to_f32(&bytes, Precision::FP16)
            .expect("test: precision conversion should succeed");

        for (&orig, &conv) in data.iter().zip(converted.iter()) {
            assert_relative_eq!(orig, conv, epsilon = 0.01);
        }
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
    fn test_invalid_byte_length() {
        let invalid_bytes = vec![0u8, 1, 2]; // 3 bytes, not divisible by 2
        assert!(f16_bytes_to_f32(&invalid_bytes).is_err());
        assert!(bf16_bytes_to_f32(&invalid_bytes).is_err());
    }
}
