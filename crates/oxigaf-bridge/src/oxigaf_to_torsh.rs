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
/// * `mapping` - Layer name mapping configuration. Custom mappings
///   registered via [`LayerMapping::add_custom_mapping`] are consulted
///   first (looked up by the exact OxiGAF name); every other name is
///   mapped by [`GafLayerMapper`], falling back to a mechanical `.` -> `/`
///   substitution.
/// * `precision` - Precision conversion configuration. ToRSh's `Tensor`
///   type stores `f32` only, so this cannot change the *storage* dtype of
///   the output the way the OxiGAF/PyTorch directions can -- instead, a
///   layer routed to FP16/BF16 has its value rounded through that
///   precision and back before being stored, so the resulting f32 value is
///   numerically what it would be if ToRSh could store it at that
///   precision.
///
/// # Errors
///
/// Returns error if:
/// - File I/O fails
/// - Safetensors parsing fails
/// - Two distinct OxiGAF tensor names map to the same ToRSh name (a name
///   collision)
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
    mapping: &LayerMapping,
    precision: &PrecisionConfig,
) -> Result<()> {
    use crate::precision::{bytes_to_f32, convert_precision, float_precision_of};

    tracing::info!(
        "Converting OxiGAF weights from {:?} to ToRSh format at {:?}",
        oxigaf_path,
        torsh_path
    );

    // Falls back to GafLayerMapper for names `mapping` has no custom
    // override for, and finally to a mechanical dot→slash conversion.
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

    let mut total_saturated = 0usize;

    for oxigaf_name in tensor_names {
        // Map layer name: OxiGAF → ToRSh. A caller-registered custom
        // mapping wins outright; otherwise try GafLayerMapper, falling back
        // to a mechanical dot→slash conversion.
        let torsh_name = if let Some(custom) = mapping.lookup_custom(oxigaf_name) {
            custom.to_string()
        } else {
            match gaf_mapper.map_oxigaf_to_torsh(oxigaf_name) {
                Ok(name) => name,
                Err(_) => {
                    tracing::debug!(
                        "Layer '{}' not in GafLayerMapper, using dot→slash fallback",
                        oxigaf_name
                    );
                    oxigaf_name.replace('.', "/")
                }
            }
        };

        // Get tensor view
        let tensor_view = safetensors.tensor(oxigaf_name).map_err(|e| {
            BridgeError::Conversion(format!("Failed to get tensor '{}': {}", oxigaf_name, e))
        })?;

        let shape: Vec<usize> = tensor_view.shape().to_vec();

        // Detect source precision from tensor dtype
        let Some(source_precision) = float_precision_of(tensor_view.dtype()) else {
            // ToRSh's `Tensor`/`SerializableTensor` types store `f32` only
            // (see `torsh_nn::serialization::ModelState::add_parameter`), so
            // a non-float tensor (integer bookkeeping, boolean masks) has no
            // representation to convert into. Rather than aborting the
            // whole checkpoint over one such tensor -- or fabricating a
            // numeric cast whose semantics this crate doesn't control --
            // skip it with an explicit warning.
            tracing::warn!(
                "Skipping tensor '{}': ToRSh only stores f32 parameters, dtype {:?} has no representation",
                oxigaf_name,
                tensor_view.dtype()
            );
            continue;
        };

        // Convert data to f32 (ToRSh always uses f32 internally)
        let data_f32 = bytes_to_f32(tensor_view.data(), source_precision).map_err(|e| {
            BridgeError::PrecisionConversion(format!(
                "Failed to convert tensor '{}' from {:?}: {}",
                oxigaf_name, source_precision, e
            ))
        })?;

        // ToRSh cannot store anything but f32, but the *value* should still
        // reflect the configured target precision: round it through that
        // precision and back so a layer routed to FP16 numerically matches
        // what it would be if ToRSh could store FP16.
        let target_precision = precision.get_layer_precision(oxigaf_name);
        let (encoded, saturated) = convert_precision(&data_f32, target_precision);
        if saturated > 0 {
            tracing::warn!(
                "{} value(s) in '{}' would saturate to +/-infinity at {} \
                 (stored as f32 +/-infinity since ToRSh has no lower-precision storage)",
                saturated,
                oxigaf_name,
                target_precision.name()
            );
            total_saturated += saturated;
        }
        let quantized_f32 = bytes_to_f32(&encoded, target_precision).map_err(|e| {
            BridgeError::PrecisionConversion(format!(
                "Failed to decode quantized data for '{}': {}",
                oxigaf_name, e
            ))
        })?;

        // Create ToRSh tensor
        let tensor = Tensor::from_vec(quantized_f32, &shape).map_err(|e| {
            BridgeError::Conversion(format!(
                "Failed to create ToRSh tensor for '{}': {}",
                torsh_name, e
            ))
        })?;

        if state.parameters.contains_key(&torsh_name) {
            return Err(BridgeError::LayerMapping(format!(
                "Layer name collision: multiple OxiGAF tensors map to ToRSh name '{}'",
                torsh_name
            )));
        }

        // Add to state
        state.add_parameter(torsh_name.clone(), tensor);

        tracing::debug!(
            "Converted layer: {} -> {} (shape: {:?})",
            oxigaf_name,
            torsh_name,
            shape
        );
    }

    if total_saturated > 0 {
        tracing::warn!(
            "{} value(s) across all converted tensors saturated to +/-infinity during precision conversion",
            total_saturated
        );
    }

    // 4. Save to ToRSh safetensors
    state
        .save_to_safetensors(torsh_path)
        .map_err(|e| BridgeError::Conversion(format!("Failed to save ToRSh checkpoint: {}", e)))?;

    tracing::info!("Successfully converted weights to ToRSh format");
    Ok(())
}

#[cfg(all(test, feature = "torsh"))]
mod tests {
    use super::*;
    use crate::Precision;
    use approx::assert_relative_eq;

    #[test]
    fn test_oxigaf_to_torsh_basic_conversion() -> Result<()> {
        let temp_dir = tempfile::tempdir().expect("test: failed to create temp dir");
        let oxigaf_path = temp_dir.path().join("oxigaf_to_torsh.safetensors");
        let torsh_path = temp_dir.path().join("torsh_output.safetensors");

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

        Ok(())
    }

    #[test]
    fn test_oxigaf_to_torsh_layer_mapping() -> Result<()> {
        let temp_dir = tempfile::tempdir().expect("test: failed to create temp dir");
        let oxigaf_path = temp_dir.path().join("oxigaf_mapping.safetensors");
        let torsh_path = temp_dir.path().join("torsh_mapping.safetensors");

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

        Ok(())
    }

    #[test]
    fn test_oxigaf_to_torsh_honors_custom_mapping() -> Result<()> {
        // Regression test: `oxigaf_to_torsh::convert` used to ignore the
        // `mapping` argument entirely, always building its own
        // `GafLayerMapper` internally -- so `LayerMapping::add_custom_mapping`
        // had no effect on this direction.
        let temp_dir = tempfile::tempdir().expect("test: failed to create temp dir");
        let oxigaf_path = temp_dir.path().join("oxigaf_custom.safetensors");
        let torsh_path = temp_dir.path().join("torsh_custom.safetensors");

        let test_data = create_test_oxigaf_safetensors()?;
        std::fs::write(&oxigaf_path, &test_data)?;

        let mut mapping = LayerMapping::new();
        mapping.add_custom_mapping(
            "conv_0_weight".to_string(),
            "custom/renamed/path".to_string(),
        );
        let precision = PrecisionConfig::default();
        convert(&oxigaf_path, &torsh_path, &mapping, &precision)?;

        let state = ModelState::load_from_safetensors(&torsh_path)
            .map_err(|e| BridgeError::Conversion(format!("Failed to load result: {}", e)))?;

        assert!(
            state.parameters.contains_key("custom/renamed/path"),
            "custom mapping should have been honored; got keys: {:?}",
            state.parameters.keys().collect::<Vec<_>>()
        );
        assert!(!state.parameters.contains_key("conv_0_weight"));

        Ok(())
    }

    #[test]
    fn test_oxigaf_to_torsh_precision_conversion() -> Result<()> {
        let temp_dir = tempfile::tempdir().expect("test: failed to create temp dir");
        let oxigaf_path = temp_dir.path().join("oxigaf_precision.safetensors");
        let torsh_path = temp_dir.path().join("torsh_precision.safetensors");

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

        let state = ModelState::load_from_safetensors(&torsh_path)
            .map_err(|e| BridgeError::Conversion(format!("Failed to load result: {}", e)))?;

        // Regression test: `oxigaf_to_torsh::convert` used to bind the
        // precision config as `_precision` and never read it, so every
        // tensor came out bit-exact FP32 regardless of configuration. ToRSh
        // still stores f32, but a non-normalization layer's *value* must
        // now reflect FP16 rounding.
        let converted = state
            .parameters
            .get("linear_weight")
            .expect("test: linear_weight should be present");
        let original: Vec<f32> = (0..converted.data.len()).map(|i| i as f32 * 0.01).collect();
        let (expected_bytes, _) = crate::precision::convert_precision(&original, Precision::FP16);
        let expected = crate::precision::bytes_to_f32(&expected_bytes, Precision::FP16)
            .expect("test: decode should succeed");
        assert_eq!(
            converted.data, expected,
            "non-normalization layer should reflect FP16 rounding"
        );
        assert_ne!(
            converted.data, original,
            "FP16 rounding should be visibly lossy for these values"
        );

        // Normalization layers are kept at FP32 by default: bit-exact, no rounding.
        let norm = state
            .parameters
            .get("norm_0_bias")
            .expect("test: norm_0_bias should be present");
        let expected_norm: Vec<f32> = (0..norm.data.len()).map(|i| i as f32 * 0.01).collect();
        assert_eq!(norm.data, expected_norm);

        Ok(())
    }

    #[test]
    fn test_oxigaf_to_torsh_skips_non_float_tensors_with_warning() -> Result<()> {
        use safetensors::tensor::{Dtype, TensorView};
        use std::collections::BTreeMap;

        let temp_dir = tempfile::tempdir().expect("test: failed to create temp dir");
        let oxigaf_path = temp_dir.path().join("mixed_dtype.safetensors");
        let torsh_path = temp_dir.path().join("mixed_dtype_out.safetensors");

        let float_data = [1.0f32, 2.0, 3.0, 4.0];
        let float_bytes: Vec<u8> = float_data.iter().flat_map(|x| x.to_le_bytes()).collect();
        let int_data: [i64; 2] = [0, 1];
        let int_bytes: Vec<u8> = int_data.iter().flat_map(|x| x.to_le_bytes()).collect();

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "conv.weight".to_string(),
            TensorView::new(Dtype::F32, vec![2, 2], &float_bytes).expect("test: tensor view"),
        );
        tensors.insert(
            "position_ids".to_string(),
            TensorView::new(Dtype::I64, vec![2], &int_bytes).expect("test: tensor view"),
        );
        let serialized =
            safetensors::serialize(&tensors, None).expect("test: serialize should succeed");
        std::fs::write(&oxigaf_path, &serialized)?;

        let mapping = LayerMapping::new();
        let precision = PrecisionConfig::default();
        // Must not abort the whole conversion just because one tensor can't
        // be represented in ToRSh's f32-only Tensor type.
        convert(&oxigaf_path, &torsh_path, &mapping, &precision)?;

        let state = ModelState::load_from_safetensors(&torsh_path)
            .map_err(|e| BridgeError::Conversion(format!("Failed to load result: {}", e)))?;
        assert!(state.parameters.contains_key("conv/weight"));
        assert!(
            !state.parameters.contains_key("position_ids"),
            "non-float tensor should be skipped, not fabricated"
        );

        Ok(())
    }

    #[test]
    fn test_oxigaf_to_torsh_round_trip() -> Result<()> {
        use crate::torsh_to_oxigaf;

        let temp_dir = tempfile::tempdir().expect("test: failed to create temp dir");
        let oxigaf_path1 = temp_dir.path().join("oxigaf_roundtrip1.safetensors");
        let torsh_path = temp_dir.path().join("torsh_roundtrip.safetensors");
        let oxigaf_path2 = temp_dir.path().join("oxigaf_roundtrip2.safetensors");

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
