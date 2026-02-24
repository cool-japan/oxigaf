//! Conversion utilities for FLAME model formats.
//!
//! This module provides utilities to convert between different FLAME model formats:
//! - `.npy` files (original `NumPy` format) → `.safetensors` (modern format)
//! - Directory of `.npy` files → single `.safetensors` file
//!
//! ## Example
//!
//! ```rust,no_run
//! use oxigaf_flame::conversion::convert_npy_to_safetensors;
//! use std::path::Path;
//!
//! // Convert a directory of .npy files to a single safetensors file
//! convert_npy_to_safetensors(
//!     Path::new("flame_model/"),
//!     Path::new("flame_model.safetensors")
//! )?;
//! # Ok::<(), oxigaf_flame::FlameError>(())
//! ```

use std::collections::HashMap;
use std::path::Path;

use crate::error::FlameError;
use crate::io_safetensors::save_flame_model_safetensors;
use crate::model::FlameModel;

/// Convert a FLAME model from `.npy` format to `.safetensors` format.
///
/// This function loads a FLAME model from a directory containing `.npy` files
/// (the original `NumPy` format) and saves it as a single `.safetensors` file.
///
/// # Arguments
///
/// * `npy_dir` - Directory containing the `.npy` files:
///   - `v_template.npy`
///   - `faces.npy`
///   - `shapedirs.npy`
///   - `expressiondirs.npy`
///   - `posedirs.npy`
///   - `J_regressor.npy`
///   - `kintree_table.npy`
///   - `weights.npy`
/// * `safetensors_path` - Output path for the `.safetensors` file
///
/// # Errors
///
/// Returns error if:
/// - The `.npy` files cannot be read
/// - The `.safetensors` file cannot be written
/// - The model data is invalid
///
/// # Example
///
/// ```rust,no_run
/// use oxigaf_flame::conversion::convert_npy_to_safetensors;
/// use std::path::Path;
///
/// convert_npy_to_safetensors(
///     Path::new("flame_2020/generic_model/"),
///     Path::new("flame_2020.safetensors")
/// )?;
/// # Ok::<(), oxigaf_flame::FlameError>(())
/// ```
pub fn convert_npy_to_safetensors(
    npy_dir: &Path,
    safetensors_path: &Path,
) -> Result<(), FlameError> {
    tracing::info!(
        "Converting FLAME model from NPY ({}) to safetensors ({})",
        npy_dir.display(),
        safetensors_path.display()
    );

    // Load from .npy format
    let model = FlameModel::load(npy_dir)?;

    // Add metadata
    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), "FLAME".to_string());
    metadata.insert("version".to_string(), "2020".to_string());
    metadata.insert("converted_from".to_string(), npy_dir.display().to_string());
    metadata.insert("conversion_tool".to_string(), "oxigaf-flame".to_string());
    metadata.insert("date".to_string(), chrono::Utc::now().to_rfc3339());

    // Save to safetensors format
    save_flame_model_safetensors(&model, safetensors_path, Some(&metadata))?;

    tracing::info!("Successfully converted FLAME model to safetensors format");

    Ok(())
}

/// Convert a FLAME model from `.npy` format to `.safetensors` format with custom metadata.
///
/// This is the same as [`convert_npy_to_safetensors`] but allows you to specify
/// custom metadata to be embedded in the `.safetensors` file.
///
/// # Arguments
///
/// * `npy_dir` - Directory containing the `.npy` files
/// * `safetensors_path` - Output path for the `.safetensors` file
/// * `metadata` - Custom metadata to embed (version, source, etc.)
///
/// # Errors
///
/// Returns error if conversion fails
///
/// # Example
///
/// ```rust,no_run
/// use oxigaf_flame::conversion::convert_npy_to_safetensors_with_metadata;
/// use std::path::Path;
/// use std::collections::HashMap;
///
/// let mut metadata = HashMap::new();
/// metadata.insert("dataset".to_string(), "FLAME 2023".to_string());
/// metadata.insert("author".to_string(), "MPI".to_string());
///
/// convert_npy_to_safetensors_with_metadata(
///     Path::new("flame_model/"),
///     Path::new("flame.safetensors"),
///     &metadata
/// )?;
/// # Ok::<(), oxigaf_flame::FlameError>(())
/// ```
#[allow(clippy::implicit_hasher)]
pub fn convert_npy_to_safetensors_with_metadata(
    npy_dir: &Path,
    safetensors_path: &Path,
    metadata: &HashMap<String, String>,
) -> Result<(), FlameError> {
    tracing::info!(
        "Converting FLAME model from NPY ({}) to safetensors ({})",
        npy_dir.display(),
        safetensors_path.display()
    );

    // Load from .npy format
    let model = FlameModel::load(npy_dir)?;

    // Save to safetensors format with custom metadata
    save_flame_model_safetensors(&model, safetensors_path, Some(metadata))?;

    tracing::info!("Successfully converted FLAME model to safetensors format");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_conversion_interface() {
        // This test just validates the API compiles correctly
        // Full integration test would require real FLAME model files
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let npy_dir = temp_dir.path().join("npy");
        let safetensors_path = temp_dir.path().join("model.safetensors");

        // Create dummy directory
        fs::create_dir(&npy_dir).expect("test: dir creation should succeed");

        // Conversion will fail (no real files), but API is validated
        let result = convert_npy_to_safetensors(&npy_dir, &safetensors_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_conversion_with_metadata() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let npy_dir = temp_dir.path().join("npy");
        let safetensors_path = temp_dir.path().join("model.safetensors");

        fs::create_dir(&npy_dir).expect("test: dir creation should succeed");

        let mut metadata = HashMap::new();
        metadata.insert("test".to_string(), "value".to_string());

        // Conversion will fail (no real files), but API is validated
        let result =
            convert_npy_to_safetensors_with_metadata(&npy_dir, &safetensors_path, &metadata);
        assert!(result.is_err());
    }
}
