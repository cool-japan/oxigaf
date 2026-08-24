//! PyTorch to OxiGAF weight conversion
//!
//! This module handles conversion from PyTorch safetensors format to OxiGAF format.

use crate::error::{BridgeError, Result};
use crate::layer_mapping::{detect_prefix, LayerMapping, PREFIX_METADATA_KEY};
use crate::precision::{
    bytes_to_f32, convert_precision, dtype_of, float_precision_of, PrecisionConfig,
};
use safetensors::{Dtype, SafeTensors};
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
///
/// # Errors
///
/// Returns an error if the input file cannot be read or parsed, if two
/// distinct PyTorch tensor names map to the same OxiGAF name (a name
/// collision), or if writing the output file fails.
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
    let mut oxigaf_tensors: HashMap<String, (Vec<u8>, Vec<usize>, Dtype)> = HashMap::new();
    // Records which recognized PyTorch prefix (if any) was stripped from
    // each tensor, keyed by the resulting OxiGAF name, so
    // `oxigaf_to_pytorch::convert` can restore it exactly on the way back
    // instead of assuming one prefix for the whole checkpoint.
    let mut prefixes: HashMap<String, String> = HashMap::new();
    let mut total_saturated = 0usize;

    for tensor_name in pytorch_tensors.names() {
        let tensor_view = pytorch_tensors
            .tensor(tensor_name)
            .map_err(|e| BridgeError::SafeTensors(e.to_string()))?;

        // Convert layer name
        let oxigaf_name = layer_mapping.pytorch_to_oxigaf(tensor_name)?;
        tracing::debug!("Converting: {} -> {}", tensor_name, oxigaf_name);

        if let Some((prefix, _)) = detect_prefix(tensor_name) {
            prefixes.insert(oxigaf_name.clone(), prefix.to_string());
        }

        // Get tensor data and shape
        let shape: Vec<usize> = tensor_view.shape().to_vec();
        let data_bytes = tensor_view.data();

        let (output_bytes, output_dtype) = match float_precision_of(tensor_view.dtype()) {
            Some(source_precision) => {
                // Convert to f32 first
                let f32_data = bytes_to_f32(data_bytes, source_precision)?;

                // Determine target precision for this layer and convert to it
                let target_precision = precision_config.get_layer_precision(&oxigaf_name);
                let (bytes, saturated) = convert_precision(&f32_data, target_precision);
                if saturated > 0 {
                    tracing::warn!(
                        "{} value(s) in '{}' saturated to +/-infinity converting to {}",
                        saturated,
                        oxigaf_name,
                        target_precision.name()
                    );
                    total_saturated += saturated;
                }

                (bytes, dtype_of(target_precision))
            }
            None => {
                // Real checkpoints routinely carry non-float bookkeeping
                // tensors (e.g. `position_ids`, `num_batches_tracked`) or
                // boolean masks. These have no meaningful "precision" to
                // convert, so pass them through unchanged rather than
                // aborting the whole conversion.
                let dtype = tensor_view.dtype();
                tracing::debug!(
                    "Passing through non-float tensor '{}' unchanged (dtype: {:?})",
                    tensor_name,
                    dtype
                );
                (data_bytes.to_vec(), dtype)
            }
        };

        if oxigaf_tensors
            .insert(oxigaf_name.clone(), (output_bytes, shape, output_dtype))
            .is_some()
        {
            return Err(BridgeError::LayerMapping(format!(
                "Layer name collision: multiple PyTorch tensors map to OxiGAF name '{}'",
                oxigaf_name
            )));
        }
    }

    let converted_count = oxigaf_tensors.len();
    let metadata = prefix_metadata(&prefixes)?;

    // Save as OxiGAF safetensors
    save_safetensors(oxigaf_path, oxigaf_tensors, metadata)?;

    if total_saturated > 0 {
        tracing::warn!(
            "{} value(s) across all converted tensors saturated to +/-infinity during precision conversion",
            total_saturated
        );
    }

    tracing::info!("Successfully converted {} tensors", converted_count);

    Ok(())
}

/// Builds the `__metadata__` map used to persist stripped-prefix
/// information, if there is any to persist.
fn prefix_metadata(prefixes: &HashMap<String, String>) -> Result<Option<HashMap<String, String>>> {
    if prefixes.is_empty() {
        return Ok(None);
    }
    let encoded = serde_json::to_string(prefixes)?;
    let mut metadata = HashMap::new();
    metadata.insert(PREFIX_METADATA_KEY.to_string(), encoded);
    Ok(Some(metadata))
}

/// Save tensors in safetensors format
fn save_safetensors(
    path: &Path,
    tensors: HashMap<String, (Vec<u8>, Vec<usize>, Dtype)>,
    metadata: Option<HashMap<String, String>>,
) -> Result<()> {
    use safetensors::tensor::{SafeTensorError, TensorView};

    // Prepare tensor data storage
    let mut tensor_data_storage: Vec<(String, Vec<u8>, Vec<usize>, Dtype)> = Vec::new();

    for (name, (data, shape, dtype)) in tensors {
        tensor_data_storage.push((name, data, shape, dtype));
    }

    // Create views from stored data
    let mut tensor_views: Vec<(&str, TensorView<'_>)> = Vec::new();

    for (name, data, shape, dtype) in &tensor_data_storage {
        let view = TensorView::new(*dtype, shape.clone(), data)
            .map_err(|e: SafeTensorError| BridgeError::SafeTensors(e.to_string()))?;

        tensor_views.push((name.as_str(), view));
    }

    // Serialize to bytes
    let serialized = safetensors::tensor::serialize(tensor_views, metadata)
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
    use safetensors::tensor::TensorView;
    use std::collections::BTreeMap;
    use tempfile::NamedTempFile;

    #[test]
    fn test_save_safetensors() {
        let temp_file = NamedTempFile::new().expect("test: temp file creation should succeed");
        let path = temp_file.path();

        let mut tensors = HashMap::new();
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        let shape = vec![2, 2];

        tensors.insert("test_tensor".to_string(), (bytes, shape, Dtype::F32));

        let result = save_safetensors(path, tensors, None);
        assert!(result.is_ok(), "Failed to save: {:?}", result.err());

        // Verify file was created and has content
        let metadata = std::fs::metadata(path).expect("test: file operation should succeed");
        assert!(metadata.len() > 0);
    }

    /// Builds a small synthetic PyTorch safetensors file with a float
    /// tensor under a recognized prefix and a non-float (I64) tensor, then
    /// writes it to `path`.
    fn write_synthetic_pytorch_checkpoint(path: &Path) {
        let float_data = [1.0f32, 2.0, 3.0, 4.0];
        let float_bytes: Vec<u8> = float_data.iter().flat_map(|x| x.to_le_bytes()).collect();
        let int_data: [i64; 4] = [10, 20, 30, 40];
        let int_bytes: Vec<u8> = int_data.iter().flat_map(|x| x.to_le_bytes()).collect();

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "unet.conv_in.weight".to_string(),
            TensorView::new(Dtype::F32, vec![2, 2], &float_bytes).expect("test: tensor view"),
        );
        tensors.insert(
            "position_ids".to_string(),
            TensorView::new(Dtype::I64, vec![4], &int_bytes).expect("test: tensor view"),
        );

        let serialized =
            safetensors::serialize(&tensors, None).expect("test: serialize should succeed");
        std::fs::write(path, &serialized).expect("test: write should succeed");
    }

    #[test]
    fn test_convert_with_tempfile() -> Result<()> {
        let temp_dir = tempfile::tempdir().expect("test: failed to create temp dir");
        let pytorch_path = temp_dir.path().join("pytorch_checkpoint.safetensors");
        let oxigaf_path = temp_dir.path().join("oxigaf_checkpoint.safetensors");

        write_synthetic_pytorch_checkpoint(&pytorch_path);

        let mapping = LayerMapping::new();
        let precision = PrecisionConfig::new();
        convert(&pytorch_path, &oxigaf_path, &mapping, &precision)?;

        let data = std::fs::read(&oxigaf_path)?;
        let result =
            SafeTensors::deserialize(&data).map_err(|e| BridgeError::SafeTensors(e.to_string()))?;

        // "unet." is a recognized prefix and is stripped; the existing
        // underscore in "conv_in" is doubled before dots become underscores.
        let conv_tensor = result
            .tensor("conv__in_weight")
            .map_err(|e| BridgeError::SafeTensors(e.to_string()))?;
        assert_eq!(conv_tensor.shape(), &[2, 2]);
        assert_eq!(conv_tensor.dtype(), Dtype::F32);
        let out_floats: Vec<f32> = conv_tensor
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(out_floats, vec![1.0f32, 2.0, 3.0, 4.0]);

        // Non-float tensors have no recognized prefix and no dots, so the
        // name is unchanged, and their bytes must pass through untouched
        // rather than the conversion aborting outright.
        let int_tensor = result
            .tensor("position_ids")
            .map_err(|e| BridgeError::SafeTensors(e.to_string()))?;
        assert_eq!(int_tensor.dtype(), Dtype::I64);
        let int_data: [i64; 4] = [10, 20, 30, 40];
        let expected_bytes: Vec<u8> = int_data.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(int_tensor.data(), expected_bytes.as_slice());

        Ok(())
    }

    #[test]
    fn test_convert_rejects_colliding_names() -> Result<()> {
        // Regression test: "unet.conv.weight" and "model.conv.weight" both
        // strip their recognized prefix and land on the OxiGAF name
        // "conv_weight". Silently overwriting one tensor with the other
        // used to go undetected (and the success log even reported the
        // wrong tensor count); this must now be a hard error.
        let temp_dir = tempfile::tempdir().expect("test: failed to create temp dir");
        let pytorch_path = temp_dir.path().join("colliding.safetensors");
        let oxigaf_path = temp_dir.path().join("colliding_out.safetensors");

        let data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "unet.conv.weight".to_string(),
            TensorView::new(Dtype::F32, vec![2, 2], &bytes).expect("test: tensor view"),
        );
        tensors.insert(
            "model.conv.weight".to_string(),
            TensorView::new(Dtype::F32, vec![2, 2], &bytes).expect("test: tensor view"),
        );
        let serialized =
            safetensors::serialize(&tensors, None).expect("test: serialize should succeed");
        std::fs::write(&pytorch_path, &serialized)?;

        let mapping = LayerMapping::new();
        let precision = PrecisionConfig::new();
        let result = convert(&pytorch_path, &oxigaf_path, &mapping, &precision);
        assert!(result.is_err(), "colliding names must be rejected");

        Ok(())
    }
}
