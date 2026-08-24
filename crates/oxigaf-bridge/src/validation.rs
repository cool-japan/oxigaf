//! Checkpoint validation utilities for verifying converted weights
//!
//! This module provides validation for converted GAF checkpoints to ensure they are
//! compatible with oxigaf-diffusion pipeline.

use crate::{BridgeError, Result};
use std::path::Path;

/// Validation report for a converted checkpoint
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// File exists and is readable
    pub file_exists: bool,
    /// Safetensors format is valid
    pub safetensors_valid: bool,
    /// Missing required layers
    pub missing_layers: Vec<String>,
    /// Layer names with invalid format (contains '/')
    pub invalid_names: Vec<String>,
    /// Tensors with invalid shapes
    pub invalid_shapes: Vec<(String, Vec<usize>)>,
    /// Layers containing NaN or Inf values
    pub has_nan_inf: Vec<String>,
    /// Non-critical warnings
    pub warnings: Vec<String>,
}

impl ValidationReport {
    /// Create a new empty validation report
    pub fn new() -> Self {
        Self {
            file_exists: false,
            safetensors_valid: false,
            missing_layers: Vec::new(),
            invalid_names: Vec::new(),
            invalid_shapes: Vec::new(),
            has_nan_inf: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Check if validation passed (no errors)
    pub fn is_valid(&self) -> bool {
        self.file_exists
            && self.safetensors_valid
            && self.missing_layers.is_empty()
            && self.invalid_names.is_empty()
            && self.invalid_shapes.is_empty()
            && self.has_nan_inf.is_empty()
    }

    /// Get summary of validation results
    pub fn summary(&self) -> String {
        if self.is_valid() {
            format!(
                "✓ Validation passed ({})",
                if self.warnings.is_empty() {
                    "no warnings".to_string()
                } else {
                    format!("{} warnings", self.warnings.len())
                }
            )
        } else {
            let mut errors = Vec::new();
            if !self.file_exists {
                errors.push("file not found".to_string());
            }
            if !self.safetensors_valid {
                errors.push("invalid safetensors".to_string());
            }
            if !self.missing_layers.is_empty() {
                errors.push(format!("{} missing layers", self.missing_layers.len()));
            }
            if !self.invalid_names.is_empty() {
                errors.push(format!("{} invalid names", self.invalid_names.len()));
            }
            if !self.invalid_shapes.is_empty() {
                errors.push(format!("{} invalid shapes", self.invalid_shapes.len()));
            }
            if !self.has_nan_inf.is_empty() {
                errors.push(format!("{} NaN/Inf tensors", self.has_nan_inf.len()));
            }
            format!("✗ Validation failed: {}", errors.join(", "))
        }
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate a converted checkpoint for oxigaf-diffusion compatibility
///
/// # Arguments
///
/// * `path` - Path to the converted checkpoint file
///
/// # Returns
///
/// A detailed validation report
///
/// # Errors
///
/// A missing file or a file that fails to parse as safetensors is *not*
/// reported through `Err` -- it comes back through the returned
/// [`ValidationReport`] instead (`file_exists` / `safetensors_valid` set to
/// `false`; check [`ValidationReport::is_valid`] or the individual fields).
/// This function only returns `Err` if the file exists but cannot be read
/// (e.g. a permissions error).
///
/// # Examples
///
/// ```rust,no_run
/// # use oxigaf_bridge::validation::validate_converted_checkpoint;
/// # use std::path::Path;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let report = validate_converted_checkpoint(Path::new("unet.safetensors"))?;
/// if report.is_valid() {
///     println!("✓ Checkpoint is valid");
/// } else {
///     println!("✗ Validation failed: {}", report.summary());
/// }
/// # Ok(())
/// # }
/// ```
pub fn validate_converted_checkpoint(path: &Path) -> Result<ValidationReport> {
    use safetensors::SafeTensors;

    let mut report = ValidationReport::new();

    // 1. Check file exists
    if !path.exists() {
        return Ok(report); // file_exists remains false
    }
    report.file_exists = true;

    // 2. Try to load as safetensors
    let data = std::fs::read(path)?;
    let safetensors = match SafeTensors::deserialize(&data) {
        Ok(st) => {
            report.safetensors_valid = true;
            st
        }
        Err(e) => {
            tracing::error!("Failed to parse safetensors: {}", e);
            return Ok(report);
        }
    };

    // 3. Check for required layers (minimal set for testing)
    let required_layers = get_required_unet_layers();
    for layer_name in &required_layers {
        if safetensors.tensor(layer_name).is_err() {
            report.missing_layers.push(layer_name.clone());
        }
    }

    // 4. Single pass over every tensor: name validity, shape validity, and
    // NaN/Inf, fetching each tensor once instead of three separate
    // `names()` walks each re-fetching every tensor.
    for name in safetensors.names() {
        if name.contains('/') {
            report
                .invalid_names
                .push(format!("{} (contains '/')", name));
        }

        let Ok(tensor) = safetensors.tensor(name) else {
            continue;
        };

        if !is_valid_shape(tensor.shape()) {
            report
                .invalid_shapes
                .push((name.to_string(), tensor.shape().to_vec()));
        }

        check_nan_inf(name, &tensor, &mut report);
    }

    Ok(report)
}

/// Checks `tensor` for NaN/Inf values, recording a hit in
/// `report.has_nan_inf`. Dtypes this function cannot interpret record a
/// warning instead of being silently skipped; a tensor whose byte slice is
/// not aligned or long enough for its dtype -- which `bytemuck::cast_slice`
/// would panic on -- is reported as a warning rather than aborting the
/// whole validation, matching how every other defect class here degrades
/// gracefully instead of crashing the validator on the malformed input it
/// exists to check.
fn check_nan_inf(
    name: &str,
    tensor: &safetensors::tensor::TensorView<'_>,
    report: &mut ValidationReport,
) {
    use safetensors::tensor::Dtype;

    match tensor.dtype() {
        Dtype::F32 => match bytemuck::try_cast_slice::<u8, f32>(tensor.data()) {
            Ok(data) => {
                if data.iter().any(|x| x.is_nan() || x.is_infinite()) {
                    report.has_nan_inf.push(name.to_string());
                }
            }
            Err(e) => report.warnings.push(format!(
                "Could not check {} for NaN/Inf: F32 data is unaligned or truncated ({})",
                name, e
            )),
        },
        Dtype::F16 => match bytemuck::try_cast_slice::<u8, u16>(tensor.data()) {
            Ok(data_u16) => {
                if data_u16.iter().any(|&bits| {
                    half::f16::from_bits(bits).is_nan() || half::f16::from_bits(bits).is_infinite()
                }) {
                    report.has_nan_inf.push(name.to_string());
                }
            }
            Err(e) => report.warnings.push(format!(
                "Could not check {} for NaN/Inf: F16 data is unaligned or truncated ({})",
                name, e
            )),
        },
        Dtype::BF16 => match bytemuck::try_cast_slice::<u8, u16>(tensor.data()) {
            Ok(data_u16) => {
                if data_u16.iter().any(|&bits| {
                    half::bf16::from_bits(bits).is_nan()
                        || half::bf16::from_bits(bits).is_infinite()
                }) {
                    report.has_nan_inf.push(name.to_string());
                }
            }
            Err(e) => report.warnings.push(format!(
                "Could not check {} for NaN/Inf: BF16 data is unaligned or truncated ({})",
                name, e
            )),
        },
        other => {
            // Other dtypes - skip NaN check
            report.warnings.push(format!(
                "Skipping NaN check for {} (dtype: {:?})",
                name, other
            ));
        }
    }
}

/// Get list of required U-Net layers (minimal set for validation)
fn get_required_unet_layers() -> Vec<String> {
    vec![
        // Input conv
        "conv_in.weight".to_string(),
        // Time embedding
        "time_embedding.linear_1.weight".to_string(),
        "time_embedding.linear_2.weight".to_string(),
        // At least one down block layer
        "down_blocks.0.resnets.0.norm1.weight".to_string(),
        // At least one up block layer
        "up_blocks.0.resnets.0.norm1.weight".to_string(),
        // Mid block
        "mid_block.resnets.0.norm1.weight".to_string(),
        // Output conv
        "conv_out.weight".to_string(),
    ]
}

/// Check if a tensor shape is valid
///
/// Valid shapes:
/// - Non-empty
/// - All dimensions > 0
/// - Total size < 10GB (safety check)
fn is_valid_shape(shape: &[usize]) -> bool {
    if shape.is_empty() {
        return false;
    }

    // Check all dimensions are positive
    if shape.contains(&0) {
        return false;
    }

    // 10GB / 4 bytes per f32. Computed as `u64`: the intermediate
    // `10 * 1024 * 1024 * 1024` (10,737,418,240) exceeds `u32::MAX`, so as
    // a `usize` constant this fails to const-evaluate at all on a 32-bit
    // target.
    const MAX_ELEMENTS: u64 = 10 * 1024 * 1024 * 1024 / 4;

    // Multiply with overflow checking: a malformed safetensors header
    // declaring e.g. `[usize::MAX, 2]` must be rejected outright, not
    // silently wrap to a small, spuriously "valid" product in a release
    // build (`Iterator::product` performs unchecked multiplication).
    let Some(total_elements) = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    else {
        return false;
    };

    total_elements as u64 <= MAX_ELEMENTS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_report_is_valid() {
        let mut report = ValidationReport::new();
        assert!(!report.is_valid());

        report.file_exists = true;
        report.safetensors_valid = true;
        assert!(report.is_valid());

        report.invalid_names.push("bad/name".to_string());
        assert!(!report.is_valid());
    }

    #[test]
    fn test_validation_report_summary() {
        let mut report = ValidationReport::new();
        report.file_exists = true;
        report.safetensors_valid = true;

        let summary = report.summary();
        assert!(summary.contains("✓ Validation passed"));

        report.invalid_names.push("bad/name".to_string());
        let summary = report.summary();
        assert!(summary.contains("✗ Validation failed"));
        assert!(summary.contains("1 invalid names"));
    }

    #[test]
    fn test_validate_nonexistent_file() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("nonexistent_validation_test.safetensors");

        let report = validate_converted_checkpoint(&path)
            .expect("validation should not error for nonexistent file");
        assert!(!report.file_exists);
        assert!(!report.is_valid());
    }

    #[test]
    fn test_is_valid_shape() {
        assert!(is_valid_shape(&[3, 224, 224]));
        assert!(is_valid_shape(&[1]));
        assert!(is_valid_shape(&[320, 320, 3, 3]));

        assert!(!is_valid_shape(&[])); // Empty
        assert!(!is_valid_shape(&[0, 10])); // Zero dimension
        assert!(!is_valid_shape(&[1000000, 1000000])); // Too large
    }

    #[test]
    fn test_is_valid_shape_rejects_overflowing_product_instead_of_wrapping() {
        // Regression test: `shape.iter().product::<usize>()` wraps silently
        // in a release build, so a malformed header declaring e.g.
        // `[usize::MAX, 2]` used to produce a small, spuriously "valid"
        // product. `checked_mul` must catch this instead.
        assert!(!is_valid_shape(&[usize::MAX, 2]));
        assert!(!is_valid_shape(&[usize::MAX / 2 + 1, 2]));
    }

    #[test]
    fn test_validate_converted_checkpoint_does_not_panic_on_misaligned_tensor_data() {
        // Regression test: `bytemuck::cast_slice` panics (rather than
        // returning an error) when a tensor's byte slice is not aligned for
        // its target type. `safetensors` does not guarantee dtype-aligned
        // data offsets, so a hand-crafted (or third-party) checkpoint can
        // trigger this. The validator's entire purpose is to accept
        // possibly-corrupt input and report on it, not abort the process.
        let temp_dir = tempfile::tempdir().expect("test: failed to create temp dir");
        let path = temp_dir.path().join("misaligned.safetensors");

        std::fs::write(&path, build_misaligned_f32_safetensors_bytes())
            .expect("test: write should succeed");

        let report = validate_converted_checkpoint(&path)
            .expect("validation must not panic or error on misaligned data");
        assert!(report.file_exists);
        assert!(report.safetensors_valid);
        assert!(
            !report.warnings.is_empty(),
            "misaligned tensor data should be reported as a warning, not silently ignored"
        );
    }

    /// Hand-crafts a minimal, otherwise-valid safetensors byte buffer (not
    /// produced via `safetensors::serialize`, which is free to align tensor
    /// data) containing a single F32 tensor whose data is deliberately
    /// *not* 4-byte aligned within the buffer.
    fn build_misaligned_f32_safetensors_bytes() -> Vec<u8> {
        let mut header = String::from(r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#);
        // Pad with insignificant trailing JSON whitespace until the total
        // prefix length (8 header-length bytes + header bytes) is *not* a
        // multiple of 4, so the tensor data starts at a misaligned offset.
        while (8 + header.len()) % 4 == 0 {
            header.push(' ');
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes
    }

    #[test]
    fn test_get_required_unet_layers() {
        let layers = get_required_unet_layers();
        assert!(!layers.is_empty());
        assert!(layers.contains(&"conv_in.weight".to_string()));
        assert!(layers.contains(&"conv_out.weight".to_string()));

        // All required layers should use dots, not slashes
        for layer in &layers {
            assert!(!layer.contains('/'), "Layer {} contains slash", layer);
        }
    }
}

/// Create a synthetic GAF checkpoint for testing (exposed for integration tests)
#[cfg(feature = "torsh")]
pub fn create_synthetic_gaf_checkpoint(output: &Path) -> Result<()> {
    use torsh_nn::serialization::{ModelMetadata, ModelState, SerializableTensor};

    let mut state = ModelState::new("GAF".to_string());
    state.metadata = ModelMetadata {
        architecture: "GAF".to_string(),
        version: "0.1.0".to_string(),
        created_at: "2026-02-11T00:00:00Z".to_string(),
        framework_version: "torsh-0.1.0".to_string(),
        tags: vec!["test".to_string(), "synthetic".to_string()],
    };

    // Helper to create a tensor
    let mut rng_state = 12345u64; // Simple LCG for deterministic random
    let mut next_random = || -> f32 {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng_state / 65536) % 32768) as f32 / 32768.0 * 0.02 - 0.01
    };

    let mut create_tensor = |shape: Vec<usize>| -> SerializableTensor {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|_| next_random()).collect();
        SerializableTensor {
            shape,
            dtype: "f32".to_string(),
            data,
            requires_grad: false,
        }
    };

    // Input conv
    state.parameters.insert(
        "conv_in/weight".to_string(),
        create_tensor(vec![320, 8, 3, 3]),
    );

    // Time embedding
    state.parameters.insert(
        "time_embedding/linear_1/weight".to_string(),
        create_tensor(vec![1280, 320]),
    );
    state.parameters.insert(
        "time_embedding/linear_2/weight".to_string(),
        create_tensor(vec![1280, 1280]),
    );

    // Camera embedding
    state.parameters.insert(
        "camera_embedding/linear_1/weight".to_string(),
        create_tensor(vec![1280, 16]),
    );

    // Down blocks (minimal - just first block, first resnet)
    state.parameters.insert(
        "down_blocks/0/resnets/0/norm1/weight".to_string(),
        create_tensor(vec![320]),
    );
    state.parameters.insert(
        "down_blocks/0/resnets/0/conv1/weight".to_string(),
        create_tensor(vec![320, 320, 3, 3]),
    );
    state.parameters.insert(
        "down_blocks/0/resnets/0/time_emb_proj/weight".to_string(),
        create_tensor(vec![320, 1280]),
    );
    state.parameters.insert(
        "down_blocks/0/resnets/0/norm2/weight".to_string(),
        create_tensor(vec![320]),
    );
    state.parameters.insert(
        "down_blocks/0/resnets/0/conv2/weight".to_string(),
        create_tensor(vec![320, 320, 3, 3]),
    );

    // Mid block
    state.parameters.insert(
        "mid_block/resnets/0/norm1/weight".to_string(),
        create_tensor(vec![1280]),
    );
    state.parameters.insert(
        "mid_block/resnets/0/conv1/weight".to_string(),
        create_tensor(vec![1280, 1280, 3, 3]),
    );

    // Up blocks (minimal - just first block, first resnet)
    state.parameters.insert(
        "up_blocks/0/resnets/0/norm1/weight".to_string(),
        create_tensor(vec![320]),
    );
    state.parameters.insert(
        "up_blocks/0/resnets/0/conv1/weight".to_string(),
        create_tensor(vec![320, 640, 3, 3]),
    );

    // Output conv
    state.parameters.insert(
        "conv_out/weight".to_string(),
        create_tensor(vec![4, 320, 3, 3]),
    );
    state
        .parameters
        .insert("conv_norm_out/weight".to_string(), create_tensor(vec![320]));

    // Save to file
    state.save_to_safetensors(output).map_err(|e| {
        BridgeError::Conversion(format!("Failed to save synthetic checkpoint: {}", e))
    })?;

    Ok(())
}

#[cfg(test)]
mod torsh_tests {
    use super::*;

    // `create_synthetic_gaf_checkpoint` used to be redefined here verbatim
    // (~110 lines, byte-for-byte identical to the `pub fn` above), shadowing
    // the real one for this module's tests. Any future change to the
    // fixture had to be made twice or the tests would silently drift from
    // the shipped generator; `use super::*` already brings the real one
    // into scope, so the tests below just call it directly.

    #[test]
    fn test_synthetic_checkpoint_generation() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_synthetic_gaf.safetensors");

        // Generate
        create_synthetic_gaf_checkpoint(&path)
            .expect("synthetic checkpoint creation should succeed");
        assert!(path.exists());

        // Load and verify
        let data = std::fs::read(&path).expect("should read file");
        let safetensors =
            safetensors::SafeTensors::deserialize(&data).expect("should parse safetensors");

        assert!(!safetensors.names().is_empty());
        assert!(safetensors.tensor("conv_in/weight").is_ok());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_validate_synthetic_checkpoint_before_conversion() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_validate_synthetic.safetensors");

        // Generate synthetic checkpoint
        create_synthetic_gaf_checkpoint(&path).expect("should create");

        // Validate - should have slashes (ToRSh format)
        let report = validate_converted_checkpoint(&path).expect("should validate");

        assert!(report.file_exists);
        assert!(report.safetensors_valid);

        // Should have invalid names (slashes) since this is pre-conversion
        assert!(!report.invalid_names.is_empty());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_validate_after_conversion() {
        use crate::WeightConverter;

        let temp_dir = std::env::temp_dir();
        let torsh_path = temp_dir.join("test_torsh_validate_after.safetensors");
        let oxigaf_path = temp_dir.join("test_oxigaf_validate_after.safetensors");

        // Generate and convert
        create_synthetic_gaf_checkpoint(&torsh_path).expect("should create");

        let converter = WeightConverter::new();
        converter
            .torsh_to_oxigaf(&torsh_path, &oxigaf_path)
            .expect("should convert");

        // Validate converted checkpoint
        let report = validate_converted_checkpoint(&oxigaf_path).expect("should validate");

        assert!(report.file_exists);
        assert!(report.safetensors_valid);

        // After conversion, names should be valid (dots not slashes)
        if !report.invalid_names.is_empty() {
            eprintln!("Invalid names found: {:?}", report.invalid_names);
        }
        assert!(
            report.invalid_names.is_empty(),
            "Should have no invalid names after conversion"
        );

        // Should have no NaN/Inf
        assert!(report.has_nan_inf.is_empty());

        // Cleanup
        let _ = std::fs::remove_file(&torsh_path);
        let _ = std::fs::remove_file(&oxigaf_path);
    }
}
