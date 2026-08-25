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
/// * `npy_dir` - Directory containing the `.npy` files. These names are the
///   exact strings [`crate::io::load_flame_model`] opens, so they are
///   case-sensitive on case-sensitive filesystems:
///   - `v_template.npy`
///   - `faces.npy`
///   - `shapedirs.npy`
///   - `expressiondirs.npy`
///   - `posedirs.npy`
///   - `j_regressor.npy`
///   - `kintree_table.npy`
///   - `lbs_weights.npy`
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
    use ndarray::{Array2, Array3};
    use ndarray_npy::write_npy;
    use std::fs;
    use tempfile::TempDir;

    /// Write a minimal but fully valid FLAME model into `dir`, using exactly
    /// the `.npy` filenames documented on [`convert_npy_to_safetensors`].
    ///
    /// Every array here has to satisfy `crate::io::load_flame_model`'s shape
    /// and kinematic-chain validators, so this doubles as a machine-checked
    /// statement of the documented input contract.
    fn write_documented_npy_model(dir: &Path) {
        // 4 vertices, 2 joints, 2-component shape/expression bases. The
        // loader fixes `posedirs`'s trailing dimension at `(n_joints - 1) * 9`.
        const N: usize = 4;
        const J: usize = 2;
        const K: usize = 2;

        let v_template = Array2::<f32>::from_shape_vec(
            (N, 3),
            vec![
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0,
            ],
        )
        .expect("test: (4, 3) shape for a 12-element vec");
        let faces = Array2::<i32>::from_shape_vec((2, 3), vec![0, 1, 2, 1, 3, 2])
            .expect("test: (2, 3) shape for a 6-element vec");
        let shapedirs = Array3::<f32>::zeros((N, 3, K));
        let expressiondirs = Array3::<f32>::zeros((N, 3, K));
        let posedirs = Array3::<f32>::zeros((N, 3, (J - 1) * 9));
        let j_regressor = Array2::<f32>::from_elem((J, N), 0.25);
        // Row 0 holds parent indices: root marker, then joint 1 → joint 0.
        let kintree_table = Array2::<i32>::from_shape_vec((2, J), vec![-1, 0, 0, 1])
            .expect("test: (2, 2) shape for a 4-element vec");
        let lbs_weights = Array2::<f32>::from_elem((N, J), 0.5);

        write_npy(dir.join("v_template.npy"), &v_template).expect("test: write v_template.npy");
        write_npy(dir.join("faces.npy"), &faces).expect("test: write faces.npy");
        write_npy(dir.join("shapedirs.npy"), &shapedirs).expect("test: write shapedirs.npy");
        write_npy(dir.join("expressiondirs.npy"), &expressiondirs)
            .expect("test: write expressiondirs.npy");
        write_npy(dir.join("posedirs.npy"), &posedirs).expect("test: write posedirs.npy");
        write_npy(dir.join("j_regressor.npy"), &j_regressor).expect("test: write j_regressor.npy");
        write_npy(dir.join("kintree_table.npy"), &kintree_table)
            .expect("test: write kintree_table.npy");
        write_npy(dir.join("lbs_weights.npy"), &lbs_weights).expect("test: write lbs_weights.npy");
    }

    #[test]
    fn documented_npy_filenames_are_the_ones_the_loader_opens() {
        // Regression test for a doc/loader drift: this `# Arguments` list used
        // to advertise `J_regressor.npy` and `weights.npy`, while
        // `crate::io::load_flame_model` opens `j_regressor.npy` and
        // `lbs_weights.npy`. A directory built from the documented names must
        // convert successfully.
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let npy_dir = temp_dir.path().join("npy");
        fs::create_dir(&npy_dir).expect("test: dir creation should succeed");
        write_documented_npy_model(&npy_dir);

        let safetensors_path = temp_dir.path().join("model.safetensors");
        convert_npy_to_safetensors(&npy_dir, &safetensors_path)
            .expect("test: a model written under the documented filenames must convert");

        let written = fs::metadata(&safetensors_path).expect("test: output file should exist");
        assert!(
            written.len() > 0,
            "converted safetensors file must not be empty"
        );
    }

    #[test]
    fn stale_weights_npy_filename_is_rejected() {
        // The other half of the same drift: `weights.npy` is genuinely a
        // different name from `lbs_weights.npy` (unlike the `J_regressor` /
        // `j_regressor` pair, which a case-insensitive filesystem cannot tell
        // apart), so a directory using the old documented name must fail --
        // proving the doc list was wrong rather than merely cosmetic.
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let npy_dir = temp_dir.path().join("npy");
        fs::create_dir(&npy_dir).expect("test: dir creation should succeed");
        write_documented_npy_model(&npy_dir);
        fs::rename(npy_dir.join("lbs_weights.npy"), npy_dir.join("weights.npy"))
            .expect("test: rename should succeed");

        let safetensors_path = temp_dir.path().join("model.safetensors");
        let err = convert_npy_to_safetensors(&npy_dir, &safetensors_path)
            .expect_err("test: `weights.npy` is not a filename the loader reads");
        assert!(
            matches!(&err, FlameError::NpyLoad { name, .. } if name == "lbs_weights"),
            "expected a missing-lbs_weights load error, got: {err}"
        );
    }

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
