//! PyTorch to OxiGAF weight conversion
//!
//! This module handles conversion from PyTorch safetensors format to OxiGAF format.

use crate::error::{BridgeError, Result};
use crate::layer_mapping::LayerMapping;
use crate::precision::{bytes_to_f32, convert_precision, Precision, PrecisionConfig};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Convert PyTorch safetensors to OxiGAF format
///
/// # Arguments
///
/// * `pytorch_path` - Path to input PyTorch safetensors file
/// * `oxigaf_path` - Path to output OxiGAF safetensors file
/// * `layer_mapping` - Layer name mapping configuration
/// * `precision_config` - Precision conversion configuration
pub fn convert(
    pytorch_path: &Path,
    oxigaf_path: &Path,
    layer_mapping: &LayerMapping,
    precision_config: &PrecisionConfig,
) -> Result<()> {
    tracing::info!("Converting PyTorch weights to OxiGAF format");
    tracing::debug!("Input: {}", pytorch_path.display());
    tracing::debug!("Output: {}", oxigaf_path.display());

    // Load PyTorch safetensors
    let buffer = std::fs::read(pytorch_path)?;
    let pytorch_tensors = SafeTensors::deserialize(&buffer)?;

    // Convert each tensor
    let mut oxigaf_tensors: HashMap<String, (Vec<u8>, Vec<usize>, Precision)> = HashMap::new();

    for tensor_name in pytorch_tensors.names() {
        let tensor_view = pytorch_tensors
            .tensor(tensor_name)
            .map_err(|e| BridgeError::SafeTensors(e.to_string()))?;

        // Convert layer name
        let oxigaf_name = layer_mapping.pytorch_to_oxigaf(tensor_name)?;
        tracing::debug!("Converting: {} -> {}", tensor_name, oxigaf_name);

        // Get tensor data and shape
        let shape: Vec<usize> = tensor_view.shape().to_vec();
        let data_bytes = tensor_view.data();

        // Determine source precision from dtype
        let source_precision = match tensor_view.dtype() {
            safetensors::Dtype::F32 => Precision::FP32,
            safetensors::Dtype::F16 => Precision::FP16,
            safetensors::Dtype::BF16 => Precision::BF16,
            _ => {
                return Err(BridgeError::UnsupportedDtype(format!(
                    "{:?}",
                    tensor_view.dtype()
                )))
            }
        };

        // Convert to f32 first
        let f32_data = bytes_to_f32(data_bytes, source_precision)?;

        // Determine target precision for this layer
        let target_precision = precision_config.get_layer_precision(&oxigaf_name);

        // Convert to target precision
        let output_bytes = convert_precision(&f32_data, target_precision);

        oxigaf_tensors.insert(oxigaf_name, (output_bytes, shape, target_precision));
    }

    // Save as OxiGAF safetensors
    save_safetensors(oxigaf_path, oxigaf_tensors)?;

    tracing::info!(
        "Successfully converted {} tensors",
        pytorch_tensors.names().len()
    );

    Ok(())
}

/// Save tensors in safetensors format
fn save_safetensors(
    path: &Path,
    tensors: HashMap<String, (Vec<u8>, Vec<usize>, Precision)>,
) -> Result<()> {
    use safetensors::tensor::{Dtype, SafeTensorError, TensorView};

    // Prepare tensor data storage
    let mut tensor_data_storage: Vec<(String, Vec<u8>, Vec<usize>, Precision)> = Vec::new();

    for (name, (data, shape, precision)) in tensors {
        tensor_data_storage.push((name, data, shape, precision));
    }

    // Create views from stored data
    let mut tensor_views: Vec<(&str, TensorView<'_>)> = Vec::new();

    for (name, data, shape, precision) in &tensor_data_storage {
        let dtype = match precision {
            Precision::FP32 => Dtype::F32,
            Precision::FP16 => Dtype::F16,
            Precision::BF16 => Dtype::BF16,
        };

        let view = TensorView::new(dtype, shape.clone(), data)
            .map_err(|e: SafeTensorError| BridgeError::SafeTensors(e.to_string()))?;

        tensor_views.push((name.as_str(), view));
    }

    // Serialize to bytes
    let serialized = safetensors::tensor::serialize(tensor_views, None)
        .map_err(|e| BridgeError::SafeTensors(e.to_string()))?;

    // Write to file
    let mut file = File::create(path)?;
    file.write_all(&serialized)?;
    file.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_save_safetensors() {
        let temp_file = NamedTempFile::new().expect("test: temp file creation should succeed");
        let path = temp_file.path();

        let mut tensors = HashMap::new();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        let shape = vec![2, 2];

        tensors.insert("test_tensor".to_string(), (bytes, shape, Precision::FP32));

        let result = save_safetensors(path, tensors);
        assert!(result.is_ok(), "Failed to save: {:?}", result.err());

        // Verify file was created and has content
        let metadata = std::fs::metadata(path).expect("test: file operation should succeed");
        assert!(metadata.len() > 0);
    }

    #[test]
    fn test_convert_with_tempfile() {
        // This is a placeholder test - in practice, you'd create a real PyTorch safetensors file
        // For now, just test the infrastructure
        let _mapping = LayerMapping::new();
        let _config = PrecisionConfig::new();

        // Test would create temp PyTorch file, convert it, and verify output
        // Skipping actual file I/O test here
    }
}
