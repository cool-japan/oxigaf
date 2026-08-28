//! OxiGAF to PyTorch weight conversion
//!
//! This module handles conversion from OxiGAF format back to PyTorch safetensors format.

use crate::error::{BridgeError, Result};
use crate::layer_mapping::{LayerMapping, PREFIX_METADATA_KEY};
use crate::precision::{
    bytes_to_f32, convert_precision, dtype_of, float_precision_of, PrecisionConfig,
};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Convert OxiGAF safetensors to PyTorch format
///
/// # Arguments
///
/// * `oxigaf_path` - Path to input OxiGAF safetensors file
/// * `pytorch_path` - Path to output PyTorch safetensors file
/// * `layer_mapping` - Layer name mapping configuration
/// * `precision_config` - Precision conversion configuration
///
/// # Errors
///
/// Returns an error if the input file cannot be read or parsed, if two
/// distinct OxiGAF tensor names map to the same PyTorch name (a name
/// collision), or if writing the output file fails.
pub fn convert(
    oxigaf_path: &Path,
    pytorch_path: &Path,
    layer_mapping: &LayerMapping,
    precision_config: &PrecisionConfig,
) -> Result<()> {
    tracing::info!("Converting OxiGAF weights to PyTorch format");
    tracing::debug!("Input: {}", oxigaf_path.display());
    tracing::debug!("Output: {}", pytorch_path.display());

    // Load OxiGAF safetensors
    let buffer = std::fs::read(oxigaf_path)?;
    let oxigaf_tensors = SafeTensors::deserialize(&buffer)?;

    // `pytorch_to_oxigaf::convert` records, per tensor, which recognized
    // PyTorch prefix (if any) it stripped, so this direction can restore
    // the exact original prefix instead of assuming a single one (e.g.
    // "unet") for the whole checkpoint. A checkpoint with no such metadata
    // (e.g. produced by some other tool) gets no prefix added back, which
    // matches how a name with no recognized prefix behaved going forward.
    let prefixes = read_prefix_metadata(&buffer)?;

    // Convert each tensor
    let mut pytorch_tensors: HashMap<String, (Vec<u8>, Vec<usize>, Dtype)> = HashMap::new();
    let mut total_saturated = 0usize;

    for tensor_name in oxigaf_tensors.names() {
        let tensor_view = oxigaf_tensors
            .tensor(tensor_name)
            .map_err(|e| BridgeError::SafeTensors(e.to_string()))?;

        // Convert layer name, restoring the original prefix if one was recorded.
        let prefix = prefixes.get(tensor_name).map(String::as_str);
        let pytorch_name = layer_mapping.oxigaf_to_pytorch(tensor_name, prefix)?;
        tracing::debug!("Converting: {} -> {}", tensor_name, pytorch_name);

        // Get tensor data and shape
        let shape: Vec<usize> = tensor_view.shape().to_vec();
        let data_bytes = tensor_view.data();

        let (output_bytes, output_dtype) = match float_precision_of(tensor_view.dtype()) {
            Some(source_precision) => {
                // Convert to f32 first
                let f32_data = bytes_to_f32(data_bytes, source_precision)?;

                // PyTorch typically uses FP32 for most operations, but
                // respect the precision config if the caller configured one.
                let target_precision = precision_config.get_layer_precision(&pytorch_name);
                let (bytes, saturated) = convert_precision(&f32_data, target_precision);
                if saturated > 0 {
                    tracing::warn!(
                        "{} value(s) in '{}' saturated to +/-infinity converting to {}",
                        saturated,
                        pytorch_name,
                        target_precision.name()
                    );
                    total_saturated += saturated;
                }

                (bytes, dtype_of(target_precision))
            }
            None => {
                // Non-float tensors (integer bookkeeping, boolean masks)
                // have no meaningful "precision" to convert; pass them
                // through unchanged rather than aborting the conversion.
                let dtype = tensor_view.dtype();
                tracing::debug!(
                    "Passing through non-float tensor '{}' unchanged (dtype: {:?})",
                    tensor_name,
                    dtype
                );
                (data_bytes.to_vec(), dtype)
            }
        };

        if pytorch_tensors
            .insert(pytorch_name.clone(), (output_bytes, shape, output_dtype))
            .is_some()
        {
            return Err(BridgeError::LayerMapping(format!(
                "Layer name collision: multiple OxiGAF tensors map to PyTorch name '{}'",
                pytorch_name
            )));
        }
    }

    let converted_count = pytorch_tensors.len();

    // Save as PyTorch safetensors
    save_safetensors(pytorch_path, pytorch_tensors)?;

    if total_saturated > 0 {
        tracing::warn!(
            "{} value(s) across all converted tensors saturated to +/-infinity during precision conversion",
            total_saturated
        );
    }

    tracing::info!("Successfully converted {} tensors", converted_count);

    Ok(())
}

/// Reads and decodes the per-tensor prefix map `pytorch_to_oxigaf::convert`
/// persists in `__metadata__` under [`PREFIX_METADATA_KEY`], if present.
///
/// Takes the raw file bytes rather than the already-deserialized
/// [`SafeTensors`]: as of safetensors 0.8 the `__metadata__` map is reachable
/// only through [`SafeTensors::read_metadata`], not from a deserialized
/// `SafeTensors` value.
///
/// Absent metadata, or metadata that fails to parse (e.g. a checkpoint from
/// an older version of this crate, or a hand-crafted/third-party file),
/// yields an empty map -- names are then reconstructed with no prefix
/// rather than the conversion failing.
fn read_prefix_metadata(buffer: &[u8]) -> Result<HashMap<String, String>> {
    let (_, metadata) = SafeTensors::read_metadata(buffer)?;
    let Some(entries) = metadata.metadata().as_ref() else {
        return Ok(HashMap::new());
    };
    let Some(encoded) = entries.get(PREFIX_METADATA_KEY) else {
        return Ok(HashMap::new());
    };
    match serde_json::from_str(encoded) {
        Ok(prefixes) => Ok(prefixes),
        Err(e) => {
            tracing::warn!(
                "Could not parse '{}' metadata ({}); converting without prefix restoration",
                PREFIX_METADATA_KEY,
                e
            );
            Ok(HashMap::new())
        }
    }
}

/// Save tensors in safetensors format
fn save_safetensors(
    path: &Path,
    tensors: HashMap<String, (Vec<u8>, Vec<usize>, Dtype)>,
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

        tensors.insert("unet.test_tensor".to_string(), (bytes, shape, Dtype::F32));

        let result = save_safetensors(path, tensors);
        assert!(result.is_ok(), "Failed to save: {:?}", result.err());

        // Verify file was created and has content
        let metadata = std::fs::metadata(path).expect("test: file operation should succeed");
        assert!(metadata.len() > 0);
    }

    /// Writes a minimal OxiGAF safetensors file (no `__metadata__`) with one
    /// float tensor, simulating a checkpoint not produced by
    /// `pytorch_to_oxigaf::convert` (e.g. `torsh_to_oxigaf::convert`, or a
    /// hand-crafted file).
    fn write_oxigaf_checkpoint_without_metadata(path: &Path) {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "conv_in.weight".to_string(),
            TensorView::new(Dtype::F32, vec![2, 2], &bytes).expect("test: tensor view"),
        );
        let serialized =
            safetensors::serialize(&tensors, None).expect("test: serialize should succeed");
        std::fs::write(path, &serialized).expect("test: write should succeed");
    }

    #[test]
    fn test_convert_defaults_to_no_prefix_when_metadata_absent() -> Result<()> {
        // Regression test: `oxigaf_to_pytorch::convert` used to hardcode a
        // "unet." prefix on every output tensor regardless of where the
        // checkpoint came from. Without prefix metadata, no prefix should
        // be added -- not a hardcoded "unet.".
        let temp_dir = tempfile::tempdir().expect("test: failed to create temp dir");
        let oxigaf_path = temp_dir.path().join("oxigaf_checkpoint.safetensors");
        let pytorch_path = temp_dir.path().join("pytorch_checkpoint.safetensors");

        write_oxigaf_checkpoint_without_metadata(&oxigaf_path);

        let mapping = LayerMapping::new();
        let precision = PrecisionConfig::new();
        convert(&oxigaf_path, &pytorch_path, &mapping, &precision)?;

        let data = std::fs::read(&pytorch_path)?;
        let result =
            SafeTensors::deserialize(&data).map_err(|e| BridgeError::SafeTensors(e.to_string()))?;

        assert!(
            result.tensor("conv_in.weight").is_ok(),
            "expected 'conv_in.weight' with no prefix; got names: {:?}",
            result.names()
        );
        assert!(result.tensor("unet.conv_in.weight").is_err());

        Ok(())
    }

    #[test]
    fn test_round_trip_preserves_each_tensors_original_prefix() -> Result<()> {
        // Regression test / round-trip: three tensors with three different
        // original "prefixes" (recognized, unrecognized, and none at all)
        // must all come back out exactly as they went in. Before the fix,
        // every tensor -- regardless of its original name -- came back
        // re-prefixed with a hardcoded "unet.".
        let temp_dir = tempfile::tempdir().expect("test: failed to create temp dir");
        let pytorch_path = temp_dir.path().join("pytorch_in.safetensors");
        let oxigaf_path = temp_dir.path().join("oxigaf_mid.safetensors");
        let pytorch_out_path = temp_dir.path().join("pytorch_out.safetensors");

        let data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();

        let names: [&str; 3] = [
            "unet.conv_in.weight",        // recognized prefix
            "vae.encoder.conv_in.weight", // unrecognized "prefix" (kept as part of the name)
            "standalone.weight",          // no prefix at all
        ];

        let mut tensors = BTreeMap::new();
        for name in names {
            tensors.insert(
                name.to_string(),
                TensorView::new(Dtype::F32, vec![2, 2], &bytes).expect("test: tensor view"),
            );
        }
        let serialized =
            safetensors::serialize(&tensors, None).expect("test: serialize should succeed");
        std::fs::write(&pytorch_path, &serialized)?;

        let mapping = LayerMapping::new();
        let precision = PrecisionConfig::new();

        crate::pytorch_to_oxigaf::convert(&pytorch_path, &oxigaf_path, &mapping, &precision)?;
        convert(&oxigaf_path, &pytorch_out_path, &mapping, &precision)?;

        let out_data = std::fs::read(&pytorch_out_path)?;
        let out = SafeTensors::deserialize(&out_data)
            .map_err(|e| BridgeError::SafeTensors(e.to_string()))?;

        let mut round_tripped: Vec<&str> = out.names().into_iter().collect();
        round_tripped.sort_unstable();
        let mut expected: Vec<&str> = names.to_vec();
        expected.sort_unstable();

        assert_eq!(
            round_tripped, expected,
            "every tensor's original prefix (including having none) must round-trip exactly"
        );

        Ok(())
    }

    #[test]
    fn test_convert_rejects_colliding_names() -> Result<()> {
        // Regression test: two OxiGAF names that both map to the same
        // PyTorch name used to silently overwrite one another; this must be
        // a hard error.
        //
        // Under the dot-preserving convention the reverse transform is the
        // identity plus an optional recorded prefix, so a collision needs
        // the prefix map to disagree with the names: "a.b" carrying a
        // recorded prefix of "unet" restores to "unet.a.b", which is
        // *also* what the literal name "unet.a.b" (no recorded prefix)
        // produces. That is reachable from a hand-edited or third-party
        // checkpoint whose `__metadata__` does not match its tensor names.
        let temp_dir = tempfile::tempdir().expect("test: failed to create temp dir");
        let oxigaf_path = temp_dir.path().join("colliding.safetensors");
        let pytorch_path = temp_dir.path().join("colliding_out.safetensors");

        let data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "a.b".to_string(),
            TensorView::new(Dtype::F32, vec![2, 2], &bytes).expect("test: tensor view"),
        );
        tensors.insert(
            "unet.a.b".to_string(),
            TensorView::new(Dtype::F32, vec![2, 2], &bytes).expect("test: tensor view"),
        );

        let mut metadata = HashMap::new();
        metadata.insert(
            PREFIX_METADATA_KEY.to_string(),
            r#"{"a.b":"unet"}"#.to_string(),
        );

        let serialized = safetensors::serialize(&tensors, Some(metadata))
            .expect("test: serialize should succeed");
        std::fs::write(&oxigaf_path, &serialized)?;

        let mapping = LayerMapping::new();
        let precision = PrecisionConfig::new();
        let result = convert(&oxigaf_path, &pytorch_path, &mapping, &precision);
        assert!(result.is_err(), "colliding names must be rejected");

        Ok(())
    }
}
