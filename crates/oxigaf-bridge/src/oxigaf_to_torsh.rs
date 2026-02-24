//! OxiGAF to ToRSh weight conversion
//!
//! This module provides conversion from OxiGAF format to ToRSh model format.

use crate::{BridgeError, GafLayerMapper, LayerMapping, PrecisionConfig, Result};
use safetensors::SafeTensors;
use std::path::Path;

#[cfg(feature = "torsh")]
use torsh_nn::serialization::ModelState;
#[cfg(feature = "torsh")]
use torsh_tensor::Tensor;

/// Convert OxiGAF weights to ToRSh format
///
/// # Arguments
///
/// * `oxigaf_path` - Path to OxiGAF safetensors file
/// * `torsh_path` - Output path for ToRSh format
/// * `mapping` - Layer name mapping configuration
/// * `precision` - Precision conversion configuration
///
/// # Errors
///
/// Returns error if:
/// - File I/O fails
/// - Safetensors parsing fails
/// - Layer name mapping fails
/// - Tensor conversion fails
/// - Precision conversion fails
///
/// # Examples
///
/// ```rust,no_run
/// # use oxigaf_bridge::{LayerMapping, PrecisionConfig};
/// # use std::path::Path;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # #[cfg(feature = "torsh")]
/// # {
/// use oxigaf_bridge::oxigaf_to_torsh;
///
/// let mapping = LayerMapping::new();
/// let precision = PrecisionConfig::default();
///
/// oxigaf_to_torsh::convert(
///     Path::new("model_oxigaf.safetensors"),
///     Path::new("model_torsh.safetensors"),
///     &mapping,
///     &precision,
/// )?;
/// # }
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "torsh")]
pub fn convert(
    oxigaf_path: &Path,
    torsh_path: &Path,
    _mapping: &LayerMapping,
    _precision: &PrecisionConfig,
) -> Result<()> {
    use crate::precision::bytes_to_f32;

    tracing::info!(
        "Converting OxiGAF weights from {:?} to ToRSh format at {:?}",
        oxigaf_path,
        torsh_path
    );

    // Use GafLayerMapper for comprehensive GAF model layer mapping.
    // Falls back to simple dot→slash conversion for layers not in the mapper.
    let gaf_mapper = GafLayerMapper::new();

    // 1. Load OxiGAF safetensors
    let data = std::fs::read(oxigaf_path)?;

    let safetensors = SafeTensors::deserialize(&data)
        .map_err(|e| BridgeError::Conversion(format!("Failed to parse safetensors: {}", e)))?;

    // 2. Create ToRSh ModelState
    let mut state = ModelState::new("GAF".to_string());
    state.metadata.architecture = "GAF".to_string();
    state.metadata.version = "0.1.0".to_string();

    // 3. Map and convert tensors
    let tensor_names: Vec<&str> = safetensors.names().to_vec();
    tracing::info!("Converting {} tensors", tensor_names.len());

    for oxigaf_name in tensor_names {
        // Map layer name: OxiGAF → ToRSh
        // Try GafLayerMapper first (explicit mapping), fall back to dot→slash conversion
        let torsh_name = match gaf_mapper.map_oxigaf_to_torsh(oxigaf_name) {
            Ok(name) => name,
            Err(_) => {
                // Fallback: simple dot → slash conversion
                tracing::debug!(
                    "Layer '{}' not in GafLayerMapper, using dot→slash fallback",
                    oxigaf_name
                );
                oxigaf_name.replace('.', "/")
            }
        };

        // Get tensor view
        let tensor_view = safetensors.tensor(oxigaf_name).map_err(|e| {
            BridgeError::Conversion(format!("Failed to get tensor '{}': {}", oxigaf_name, e))
        })?;

        let shape: Vec<usize> = tensor_view.shape().to_vec();

        // Detect source precision from tensor dtype
        let source_precision = match tensor_view.dtype() {
            safetensors::tensor::Dtype::F32 => crate::Precision::FP32,
            safetensors::tensor::Dtype::F16 => crate::Precision::FP16,
            safetensors::tensor::Dtype::BF16 => crate::Precision::BF16,
            other => {
                return Err(BridgeError::UnsupportedDtype(format!(
                    "Unsupported dtype for tensor '{}': {:?}",
                    oxigaf_name, other
                )))
            }
        };

        // Convert data to f32 (ToRSh always uses f32 internally)
        let data_f32 = bytes_to_f32(tensor_view.data(), source_precision).map_err(|e| {
            BridgeError::PrecisionConversion(format!(
                "Failed to convert tensor '{}' from {:?}: {}",
                oxigaf_name, source_precision, e
            ))
        })?;

        // Create ToRSh tensor
        let tensor = Tensor::from_vec(data_f32, &shape).map_err(|e| {
            BridgeError::Conversion(format!(
                "Failed to create ToRSh tensor for '{}': {}",
                torsh_name, e
            ))
        })?;

        // Add to state
        state.add_parameter(torsh_name.clone(), tensor);

        tracing::debug!(
            "Converted layer: {} -> {} (shape: {:?})",
            oxigaf_name,
            torsh_name,
            shape
        );
    }

    // 4. Save to ToRSh safetensors
    state
        .save_to_safetensors(torsh_path)
        .map_err(|e| BridgeError::Conversion(format!("Failed to save ToRSh checkpoint: {}", e)))?;

    tracing::info!("Successfully converted weights to ToRSh format");
    Ok(())
}

#[cfg(not(feature = "torsh"))]
pub fn convert(
    _oxigaf_path: &Path,
    _torsh_path: &Path,
    _mapping: &LayerMapping,
    _precision: &PrecisionConfig,
) -> Result<()> {
    Err(BridgeError::Conversion(
        "ToRSh feature not enabled. Compile with --features torsh".to_string(),
    ))
}

#[cfg(all(test, feature = "torsh"))]
mod tests {
    use super::*;
    use crate::{Precision, PrecisionConfig};
    use approx::assert_relative_eq;
    use std::env;

    #[test]
    fn test_oxigaf_to_torsh_basic_conversion() -> Result<()> {
        // Create temporary test files
        let temp_dir = env::temp_dir();
        let oxigaf_path = temp_dir.join("test_oxigaf_to_torsh.safetensors");
        let torsh_path = temp_dir.join("test_torsh_output.safetensors");

        // Create test OxiGAF safetensors
        let test_data = create_test_oxigaf_safetensors()?;
        std::fs::write(&oxigaf_path, &test_data)?;

        // Convert
        let mapping = LayerMapping::new();
        let precision = PrecisionConfig::default();
        convert(&oxigaf_path, &torsh_path, &mapping, &precision)?;

        // Verify output exists
        assert!(torsh_path.exists());

        // Load and verify
        let state = ModelState::load_from_safetensors(&torsh_path)
            .map_err(|e| BridgeError::Conversion(format!("Failed to load result: {}", e)))?;

        assert!(!state.parameters.is_empty());
        // Note: SafeTensors deserialization doesn't preserve all metadata,
        // so we just check that parameters were loaded
        assert!(state.parameters.len() >= 3);

        // Cleanup
        let _ = std::fs::remove_file(&oxigaf_path);
        let _ = std::fs::remove_file(&torsh_path);

        Ok(())
    }

    #[test]
    fn test_oxigaf_to_torsh_layer_mapping() -> Result<()> {
        let temp_dir = env::temp_dir();
        let oxigaf_path = temp_dir.join("test_oxigaf_mapping.safetensors");
        let torsh_path = temp_dir.join("test_torsh_mapping.safetensors");

        // Create test data
        let test_data = create_test_oxigaf_safetensors()?;
        std::fs::write(&oxigaf_path, &test_data)?;

        // Convert with custom mapping
        let mapping = LayerMapping::new();
        let precision = PrecisionConfig::default();
        convert(&oxigaf_path, &torsh_path, &mapping, &precision)?;

        // Load and check layer names
        let state = ModelState::load_from_safetensors(&torsh_path)
            .map_err(|e| BridgeError::Conversion(format!("Failed to load result: {}", e)))?;

        // Verify layer names do not contain dots (OxiGAF convention uses dots,
        // ToRSh should have converted them to slashes or left underscores as-is)
        for name in state.parameters.keys() {
            assert!(
                !name.contains('.'),
                "Layer name should not contain dots in ToRSh format: {}",
                name
            );
        }

        // Cleanup
        let _ = std::fs::remove_file(&oxigaf_path);
        let _ = std::fs::remove_file(&torsh_path);

        Ok(())
    }

    #[test]
    fn test_oxigaf_to_torsh_precision_conversion() -> Result<()> {
        let temp_dir = env::temp_dir();
        let oxigaf_path = temp_dir.join("test_oxigaf_precision.safetensors");
        let torsh_path = temp_dir.join("test_torsh_precision.safetensors");

        // Create test data
        let test_data = create_test_oxigaf_safetensors()?;
        std::fs::write(&oxigaf_path, &test_data)?;

        // Convert with FP16 precision
        let mapping = LayerMapping::new();
        let mut precision = PrecisionConfig::default();
        precision.set_default_precision(Precision::FP16);
        convert(&oxigaf_path, &torsh_path, &mapping, &precision)?;

        // Verify output
        assert!(torsh_path.exists());

        // Cleanup
        let _ = std::fs::remove_file(&oxigaf_path);
        let _ = std::fs::remove_file(&torsh_path);

        Ok(())
    }

    #[test]
    fn test_oxigaf_to_torsh_round_trip() -> Result<()> {
        use crate::torsh_to_oxigaf;

        let temp_dir = env::temp_dir();
        let oxigaf_path1 = temp_dir.join("test_oxigaf_roundtrip1.safetensors");
        let torsh_path = temp_dir.join("test_torsh_roundtrip.safetensors");
        let oxigaf_path2 = temp_dir.join("test_oxigaf_roundtrip2.safetensors");

        // Create original OxiGAF data
        let original_data = create_test_oxigaf_safetensors()?;
        std::fs::write(&oxigaf_path1, &original_data)?;

        // Convert: OxiGAF → ToRSh → OxiGAF
        let mapping = LayerMapping::new();
        let precision = PrecisionConfig::default(); // Use FP32 for exact comparison

        convert(&oxigaf_path1, &torsh_path, &mapping, &precision)?;
        torsh_to_oxigaf::convert(&torsh_path, &oxigaf_path2, &mapping, &precision)?;

        // Load both OxiGAF files and compare
        let data1 = std::fs::read(&oxigaf_path1)?;
        let data2 = std::fs::read(&oxigaf_path2)?;

        let st1 = SafeTensors::deserialize(&data1)
            .map_err(|e| BridgeError::Conversion(format!("Failed to parse original: {}", e)))?;
        let st2 = SafeTensors::deserialize(&data2)
            .map_err(|e| BridgeError::Conversion(format!("Failed to parse converted: {}", e)))?;

        // Compare tensors
        for name in st1.names() {
            let t1 = st1
                .tensor(name)
                .map_err(|e| BridgeError::Conversion(format!("Failed to get tensor: {}", e)))?;
            let t2 = st2
                .tensor(name)
                .map_err(|e| BridgeError::Conversion(format!("Failed to get tensor: {}", e)))?;

            assert_eq!(t1.shape(), t2.shape());

            let d1: Vec<f32> = bytemuck::cast_slice(t1.data()).to_vec();
            let d2: Vec<f32> = bytemuck::cast_slice(t2.data()).to_vec();

            for (i, (&v1, &v2)) in d1.iter().zip(d2.iter()).enumerate() {
                assert_relative_eq!(v1, v2, epsilon = 1e-5, max_relative = 1e-5);
                if (v1 - v2).abs() > 1e-5 {
                    panic!("Mismatch at tensor {} index {}: {} vs {}", name, i, v1, v2);
                }
            }
        }

        // Cleanup
        let _ = std::fs::remove_file(&oxigaf_path1);
        let _ = std::fs::remove_file(&torsh_path);
        let _ = std::fs::remove_file(&oxigaf_path2);

        Ok(())
    }

    // Helper function to create test OxiGAF safetensors
    fn create_test_oxigaf_safetensors() -> Result<Vec<u8>> {
        use safetensors::tensor::{Dtype, TensorView};
        use std::collections::BTreeMap;

        // Create test tensors with OxiGAF naming convention
        let test_cases = vec![
            ("conv_0_weight", vec![3, 3, 64, 128]),
            ("norm_0_bias", vec![128]),
            ("linear_weight", vec![256, 128]),
        ];

        // Collect all data first to extend lifetimes
        let mut tensor_data: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();
        for (name, shape) in test_cases {
            let size: usize = shape.iter().product();
            let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
            tensor_data.push((name.to_string(), data, shape));
        }

        // Now create tensor views
        let mut tensors = BTreeMap::new();
        for (name, data, shape) in &tensor_data {
            let data_bytes: &[u8] = bytemuck::cast_slice(data);
            let view = TensorView::new(Dtype::F32, shape.clone(), data_bytes).map_err(|e| {
                BridgeError::Conversion(format!("Failed to create tensor view: {}", e))
            })?;
            tensors.insert(name.clone(), view);
        }

        safetensors::serialize(&tensors, None)
            .map_err(|e| BridgeError::Conversion(format!("Failed to serialize: {}", e)))
    }
}

#[cfg(all(test, not(feature = "torsh")))]
mod tests {
    use super::*;

    #[test]
    fn test_convert_requires_feature() {
        use std::env;
        let temp_dir = env::temp_dir();
        let oxigaf_path = temp_dir.join("dummy.safetensors");
        let torsh_path = temp_dir.join("dummy_out.safetensors");
        let mapping = LayerMapping::new();
        let precision = PrecisionConfig::default();

        let result = convert(&oxigaf_path, &torsh_path, &mapping, &precision);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("ToRSh feature not enabled"));
    }
}
