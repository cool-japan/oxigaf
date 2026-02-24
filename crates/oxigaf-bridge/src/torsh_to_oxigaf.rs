//! ToRSh to OxiGAF weight conversion
//!
//! This module provides conversion from ToRSh model format to OxiGAF format.

use crate::{BridgeError, GafLayerMapper, LayerMapping, PrecisionConfig, Result};
use std::path::Path;

#[cfg(feature = "torsh")]
use torsh_nn::serialization::ModelState;

/// Convert ToRSh weights to OxiGAF format
///
/// # Arguments
///
/// * `torsh_path` - Path to ToRSh safetensors file
/// * `oxigaf_path` - Output path for OxiGAF format
/// * `mapping` - Layer name mapping configuration
/// * `precision` - Precision conversion configuration
///
/// # Errors
///
/// Returns error if:
/// - File I/O fails
/// - ToRSh deserialization fails
/// - Layer name mapping fails
/// - Tensor conversion fails
/// - Safetensors serialization fails
///
/// # Examples
///
/// ```rust,no_run
/// # use oxigaf_bridge::{LayerMapping, PrecisionConfig};
/// # use std::path::Path;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # #[cfg(feature = "torsh")]
/// # {
/// use oxigaf_bridge::torsh_to_oxigaf;
///
/// let mapping = LayerMapping::new();
/// let precision = PrecisionConfig::default();
///
/// torsh_to_oxigaf::convert(
///     Path::new("model_torsh.safetensors"),
///     Path::new("model_oxigaf.safetensors"),
///     &mapping,
///     &precision,
/// )?;
/// # }
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "torsh")]
pub fn convert(
    torsh_path: &Path,
    oxigaf_path: &Path,
    _mapping: &LayerMapping,
    precision: &PrecisionConfig,
) -> Result<()> {
    use crate::precision::convert_precision;
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::BTreeMap;

    tracing::info!(
        "Converting ToRSh weights from {:?} to OxiGAF format at {:?}",
        torsh_path,
        oxigaf_path
    );

    // Use GafLayerMapper for comprehensive GAF model layer mapping.
    // Falls back to simple slash→dot conversion for layers not in the mapper.
    let gaf_mapper = GafLayerMapper::new();

    // 1. Load ToRSh ModelState
    let state = ModelState::load_from_safetensors(torsh_path)
        .map_err(|e| BridgeError::Conversion(format!("Failed to load ToRSh checkpoint: {}", e)))?;

    tracing::info!("Converting {} parameters", state.parameters.len());

    // 2. Convert tensors and map names
    // First, collect all tensor data with owned bytes
    let mut tensor_data: Vec<(String, Vec<u8>, Vec<usize>, Dtype)> = Vec::new();

    for (torsh_name, serializable_tensor) in &state.parameters {
        // Map layer name: ToRSh → OxiGAF
        // Try GafLayerMapper first (explicit mapping), fall back to slash→dot conversion
        let oxigaf_name = match gaf_mapper.map_torsh_to_oxigaf(torsh_name) {
            Ok(name) => name,
            Err(_) => {
                // Fallback: simple slash → dot conversion (VarBuilder-compatible)
                tracing::debug!(
                    "Layer '{}' not in GafLayerMapper, using slash→dot fallback",
                    torsh_name
                );
                torsh_name.replace('/', ".")
            }
        };

        // Get tensor data
        let data_f32 = &serializable_tensor.data;
        let shape = serializable_tensor.shape.clone();

        // Apply precision conversion
        let layer_precision = precision.get_layer_precision(&oxigaf_name);
        let data_bytes = convert_precision(data_f32, layer_precision);

        // Determine dtype based on precision
        let dtype = match layer_precision {
            crate::Precision::FP32 => Dtype::F32,
            crate::Precision::FP16 => Dtype::F16,
            crate::Precision::BF16 => Dtype::BF16,
        };

        tensor_data.push((oxigaf_name.clone(), data_bytes, shape, dtype));

        tracing::debug!(
            "Converted layer: {} -> {} (shape: {:?}, precision: {:?})",
            torsh_name,
            oxigaf_name,
            serializable_tensor.shape,
            layer_precision
        );
    }

    // 3. Create tensor views and serialize
    let mut tensors = BTreeMap::new();
    for (name, data_bytes, shape, dtype) in &tensor_data {
        let tensor_view = TensorView::new(*dtype, shape.clone(), data_bytes).map_err(|e| {
            BridgeError::Conversion(format!(
                "Failed to create tensor view for '{}': {}",
                name, e
            ))
        })?;
        tensors.insert(name.clone(), tensor_view);
    }

    let serialized = safetensors::serialize(&tensors, None)
        .map_err(|e| BridgeError::Conversion(format!("Failed to serialize safetensors: {}", e)))?;

    // 4. Write to file
    std::fs::write(oxigaf_path, &serialized)?;

    tracing::info!("Successfully converted weights to OxiGAF format");
    Ok(())
}

#[cfg(not(feature = "torsh"))]
pub fn convert(
    _torsh_path: &Path,
    _oxigaf_path: &Path,
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
    use safetensors::SafeTensors;
    use std::env;
    use torsh_nn::serialization::{ModelMetadata, SerializableTensor};

    #[test]
    fn test_torsh_to_oxigaf_basic_conversion() -> Result<()> {
        let temp_dir = env::temp_dir();
        let torsh_path = temp_dir.join("test_torsh_to_oxigaf.safetensors");
        let oxigaf_path = temp_dir.join("test_oxigaf_output.safetensors");

        // Create test ToRSh safetensors
        create_test_torsh_safetensors(&torsh_path)?;

        // Convert
        let mapping = LayerMapping::new();
        let precision = PrecisionConfig::default();
        convert(&torsh_path, &oxigaf_path, &mapping, &precision)?;

        // Verify output exists
        assert!(oxigaf_path.exists());

        // Load and verify
        let data = std::fs::read(&oxigaf_path)?;
        let safetensors = SafeTensors::deserialize(&data)
            .map_err(|e| BridgeError::Conversion(format!("Failed to parse result: {}", e)))?;

        assert!(!safetensors.names().is_empty());

        // Cleanup
        let _ = std::fs::remove_file(&torsh_path);
        let _ = std::fs::remove_file(&oxigaf_path);

        Ok(())
    }

    #[test]
    fn test_torsh_to_oxigaf_layer_mapping() -> Result<()> {
        let temp_dir = env::temp_dir();
        let torsh_path = temp_dir.join("test_torsh_mapping.safetensors");
        let oxigaf_path = temp_dir.join("test_oxigaf_mapping.safetensors");

        // Create test data
        create_test_torsh_safetensors(&torsh_path)?;

        // Convert
        let mapping = LayerMapping::new();
        let precision = PrecisionConfig::default();
        convert(&torsh_path, &oxigaf_path, &mapping, &precision)?;

        // Load and check layer names
        let data = std::fs::read(&oxigaf_path)?;
        let safetensors = SafeTensors::deserialize(&data)
            .map_err(|e| BridgeError::Conversion(format!("Failed to parse result: {}", e)))?;

        // Verify layer names use OxiGAF convention (underscores instead of slashes)
        for name in safetensors.names() {
            assert!(
                !name.contains('/'),
                "Layer name should not contain slashes in OxiGAF format: {}",
                name
            );
        }

        // Cleanup
        let _ = std::fs::remove_file(&torsh_path);
        let _ = std::fs::remove_file(&oxigaf_path);

        Ok(())
    }

    #[test]
    fn test_torsh_to_oxigaf_precision_conversion() -> Result<()> {
        let temp_dir = env::temp_dir();
        let torsh_path = temp_dir.join("test_torsh_precision.safetensors");
        let oxigaf_path = temp_dir.join("test_oxigaf_precision.safetensors");

        // Create test data
        create_test_torsh_safetensors(&torsh_path)?;

        // Convert with FP16 precision
        let mapping = LayerMapping::new();
        let mut precision = PrecisionConfig::default();
        precision.set_default_precision(Precision::FP16);
        convert(&torsh_path, &oxigaf_path, &mapping, &precision)?;

        // Load and verify precision
        let data = std::fs::read(&oxigaf_path)?;
        let safetensors = SafeTensors::deserialize(&data)
            .map_err(|e| BridgeError::Conversion(format!("Failed to parse result: {}", e)))?;

        // Check that tensors are in FP16 format (except normalization layers)
        for name in safetensors.names() {
            let tensor = safetensors
                .tensor(name)
                .map_err(|e| BridgeError::Conversion(format!("Failed to get tensor: {}", e)))?;

            if name.contains("norm") {
                // Normalization should stay FP32
                assert_eq!(tensor.dtype(), safetensors::tensor::Dtype::F32);
            } else {
                // Others should be FP16
                assert_eq!(tensor.dtype(), safetensors::tensor::Dtype::F16);
            }
        }

        // Cleanup
        let _ = std::fs::remove_file(&torsh_path);
        let _ = std::fs::remove_file(&oxigaf_path);

        Ok(())
    }

    #[test]
    fn test_torsh_to_oxigaf_numerical_accuracy() -> Result<()> {
        let temp_dir = env::temp_dir();
        let torsh_path = temp_dir.join("test_torsh_accuracy.safetensors");
        let oxigaf_path = temp_dir.join("test_oxigaf_accuracy.safetensors");

        // Create test data with known values
        create_test_torsh_safetensors(&torsh_path)?;

        // Convert with FP32 for exact comparison
        let mapping = LayerMapping::new();
        let precision = PrecisionConfig::default(); // FP32
        convert(&torsh_path, &oxigaf_path, &mapping, &precision)?;

        // Load ToRSh to get original values
        let torsh_state = ModelState::load_from_safetensors(&torsh_path)
            .map_err(|e| BridgeError::Conversion(format!("Failed to load ToRSh: {}", e)))?;

        // Load OxiGAF result
        let data = std::fs::read(&oxigaf_path)?;
        let safetensors = SafeTensors::deserialize(&data)
            .map_err(|e| BridgeError::Conversion(format!("Failed to parse result: {}", e)))?;

        // Compare values
        let gaf_mapper = crate::GafLayerMapper::new();
        for (torsh_name, serializable_tensor) in &torsh_state.parameters {
            // Use same mapping logic as convert(): GafLayerMapper first, fallback to slash→dot
            let oxigaf_name = match gaf_mapper.map_torsh_to_oxigaf(torsh_name) {
                Ok(name) => name,
                Err(_) => torsh_name.replace('/', "."),
            };
            let tensor = safetensors
                .tensor(&oxigaf_name)
                .map_err(|e| BridgeError::Conversion(format!("Failed to get tensor: {}", e)))?;

            let oxigaf_data: Vec<f32> = bytemuck::cast_slice(tensor.data()).to_vec();
            let torsh_data = &serializable_tensor.data;

            assert_eq!(oxigaf_data.len(), torsh_data.len());
            for (i, (&v1, &v2)) in oxigaf_data.iter().zip(torsh_data.iter()).enumerate() {
                assert_relative_eq!(v1, v2, epsilon = 1e-6, max_relative = 1e-6);
                if (v1 - v2).abs() > 1e-6 {
                    panic!(
                        "Mismatch at tensor {} index {}: {} vs {}",
                        oxigaf_name, i, v1, v2
                    );
                }
            }
        }

        // Cleanup
        let _ = std::fs::remove_file(&torsh_path);
        let _ = std::fs::remove_file(&oxigaf_path);

        Ok(())
    }

    // Helper function to create test ToRSh safetensors
    fn create_test_torsh_safetensors(path: &Path) -> Result<()> {
        let mut state = ModelState::new("GAF".to_string());
        state.metadata = ModelMetadata {
            architecture: "GAF".to_string(),
            version: "0.1.0".to_string(),
            created_at: "2026-02-11T00:00:00Z".to_string(),
            framework_version: "0.1.0".to_string(),
            tags: vec!["test".to_string()],
        };

        // Create test parameters with ToRSh naming convention (slashes)
        let test_cases = vec![
            ("conv/0/weight", vec![3, 3, 64, 128]),
            ("norm/0/bias", vec![128]),
            ("linear/weight", vec![256, 128]),
        ];

        for (name, shape) in test_cases {
            let size: usize = shape.iter().product();
            let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();

            let serializable = SerializableTensor {
                shape,
                dtype: "f32".to_string(),
                data,
                requires_grad: false,
            };

            state.parameters.insert(name.to_string(), serializable);
        }

        state.save_to_safetensors(path).map_err(|e| {
            BridgeError::Conversion(format!("Failed to save test ToRSh file: {}", e))
        })?;

        Ok(())
    }
}

#[cfg(all(test, not(feature = "torsh")))]
mod tests {
    use super::*;

    #[test]
    fn test_convert_requires_feature() {
        use std::env;
        let temp_dir = env::temp_dir();
        let torsh_path = temp_dir.join("dummy.safetensors");
        let oxigaf_path = temp_dir.join("dummy_out.safetensors");
        let mapping = LayerMapping::new();
        let precision = PrecisionConfig::default();

        let result = convert(&torsh_path, &oxigaf_path, &mapping, &precision);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("ToRSh feature not enabled"));
    }
}
