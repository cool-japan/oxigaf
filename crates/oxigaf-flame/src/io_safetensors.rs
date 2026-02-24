//! Load and save FLAME models using the safetensors format.
//!
//! This module provides an alternative to NPY files for storing FLAME models.
//! Safetensors is a modern, efficient format widely used in ML frameworks.
//!
//! ## Advantages of safetensors
//!
//! - **Single file**: All model data in one file instead of multiple `.npy` files
//! - **Metadata support**: Store model version, source, creation date, etc.
//! - **Fast loading**: Memory-mapped access for large models
//! - **Cross-platform**: Better compatibility than pickle-based formats
//! - **Type-safe**: Built-in validation of tensor shapes and dtypes
//!
//! ## Format
//!
//! The safetensors file contains the following tensors:
//!
//! | Tensor Name      | Shape              | dtype   |
//! |------------------|--------------------|---------|
//! | `v_template`     | `[5023, 3]`        | float32 |
//! | `faces`          | `[9976, 3]`        | int32   |
//! | `shapedirs`      | `[5023, 3, 300]`   | float32 |
//! | `expressiondirs` | `[5023, 3, 100]`   | float32 |
//! | `posedirs`       | `[5023, 3, 36]`    | float32 |
//! | `j_regressor`    | `[5, 5023]`        | float32 |
//! | `kintree_table`  | `[2, 5]`           | int32   |
//! | `lbs_weights`    | `[5023, 5]`        | float32 |
//!
//! Plus optional metadata:
//! - `__metadata__`: JSON object with model info (version, source, date, etc.)

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use ndarray::{Array2, Array3};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;

use crate::error::FlameError;
use crate::model::FlameModel;

/// Load a [`FlameModel`] from a safetensors file.
///
/// # Arguments
///
/// * `path` - Path to the safetensors file
///
/// # Errors
///
/// Returns an error if:
/// - The file does not exist or cannot be read
/// - The file is not a valid safetensors file
/// - Required tensors are missing
/// - Tensor shapes do not match expected dimensions
///
/// # Example
///
/// ```rust,no_run
/// use oxigaf_flame::io_safetensors::load_flame_model_safetensors;
/// use std::path::Path;
///
/// let model = load_flame_model_safetensors(Path::new("flame_model.safetensors"))?;
/// println!("Loaded FLAME model with {} vertices", model.num_vertices());
/// # Ok::<(), oxigaf_flame::FlameError>(())
/// ```
pub fn load_flame_model_safetensors(path: &Path) -> Result<FlameModel, FlameError> {
    tracing::debug!("Loading FLAME model from safetensors: {}", path.display());

    // Read the entire file into memory
    let buffer = std::fs::read(path).map_err(|e| FlameError::IoError {
        source: e,
        path: path.to_path_buf(),
    })?;

    // Deserialize safetensors
    let tensors = SafeTensors::deserialize(&buffer).map_err(|e| FlameError::SafeTensorsLoad {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;

    // Load each tensor
    let v_template = load_tensor_f32_2d(&tensors, "v_template")?;
    let faces_i32 = load_tensor_i32_2d(&tensors, "faces")?;
    let shapedirs = load_tensor_f32_3d(&tensors, "shapedirs")?;
    let expressiondirs = load_tensor_f32_3d(&tensors, "expressiondirs")?;
    let posedirs = load_tensor_f32_3d(&tensors, "posedirs")?;
    let j_regressor = load_tensor_f32_2d(&tensors, "j_regressor")?;
    let kintree_i32 = load_tensor_i32_2d(&tensors, "kintree_table")?;
    let lbs_weights = load_tensor_f32_2d(&tensors, "lbs_weights")?;

    // Convert faces from i32 → Vec<[u32; 3]>
    let faces: Vec<[u32; 3]> = faces_i32
        .rows()
        .into_iter()
        .map(|row| [row[0] as u32, row[1] as u32, row[2] as u32])
        .collect();

    // Extract parent indices from kintree_table row 0
    let n_joints = kintree_i32.ncols();
    let parents: Vec<i32> = (0..n_joints).map(|j| kintree_i32[[0, j]]).collect();

    // Validate shapes
    let n_verts = v_template.nrows();

    tracing::info!(
        n_verts,
        n_faces = faces.len(),
        n_joints,
        n_shape = shapedirs.shape()[2],
        n_expr = expressiondirs.shape()[2],
        "FLAME model loaded from safetensors"
    );

    Ok(FlameModel {
        v_template,
        faces,
        shapedirs,
        expressiondirs,
        posedirs,
        j_regressor,
        parents,
        lbs_weights,
        n_joints,
    })
}

/// Save a [`FlameModel`] to a safetensors file.
///
/// # Arguments
///
/// * `model` - The FLAME model to save
/// * `path` - Output path for the safetensors file
/// * `metadata` - Optional metadata to include (version, source, etc.)
///
/// # Errors
///
/// Returns an error if the file cannot be written.
///
/// # Example
///
/// ```rust,no_run
/// use oxigaf_flame::{FlameModel, io_safetensors::save_flame_model_safetensors};
/// use std::path::Path;
/// use std::collections::HashMap;
///
/// # fn load_model() -> Result<FlameModel, Box<dyn std::error::Error>> {
/// #     todo!()
/// # }
/// let model = load_model()?;
///
/// let mut metadata = HashMap::new();
/// metadata.insert("version".to_string(), "1.0".to_string());
/// metadata.insert("source".to_string(), "FLAME 2020".to_string());
///
/// save_flame_model_safetensors(&model, Path::new("flame_model.safetensors"), Some(&metadata))?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(clippy::implicit_hasher)]
pub fn save_flame_model_safetensors(
    model: &FlameModel,
    path: &Path,
    metadata: Option<&HashMap<String, String>>,
) -> Result<(), FlameError> {
    tracing::debug!("Saving FLAME model to safetensors: {}", path.display());

    // Extract model data into slices
    let slices = extract_model_slices(model, path)?;

    // Create tensor views
    let tensors = create_tensor_views(model, &slices, path)?;

    // Serialize and write to file
    write_safetensors_to_file(tensors, metadata, path)?;

    tracing::info!("Successfully saved FLAME model to safetensors");

    Ok(())
}

// ---------------------------------------------------------------------------
// Helper functions for saving
// ---------------------------------------------------------------------------

/// Holds references to model data slices for serialization.
struct ModelDataSlices<'a> {
    v_template: &'a [f32],
    shapedirs: &'a [f32],
    expressiondirs: &'a [f32],
    posedirs: &'a [f32],
    j_regressor: &'a [f32],
    lbs_weights: &'a [f32],
    faces_i32: Vec<i32>,
    kintree_i32: Vec<i32>,
}

/// Extract and convert model data to slices for serialization.
fn extract_model_slices<'a>(
    model: &'a FlameModel,
    path: &Path,
) -> Result<ModelDataSlices<'a>, FlameError> {
    let v_template = model
        .v_template
        .as_slice()
        .ok_or_else(|| FlameError::SafeTensorsSave {
            path: path.to_path_buf(),
            message: "v_template is not contiguous".to_string(),
        })?;

    let shapedirs = model
        .shapedirs
        .as_slice()
        .ok_or_else(|| FlameError::SafeTensorsSave {
            path: path.to_path_buf(),
            message: "shapedirs is not contiguous".to_string(),
        })?;

    let expressiondirs =
        model
            .expressiondirs
            .as_slice()
            .ok_or_else(|| FlameError::SafeTensorsSave {
                path: path.to_path_buf(),
                message: "expressiondirs is not contiguous".to_string(),
            })?;

    let posedirs = model
        .posedirs
        .as_slice()
        .ok_or_else(|| FlameError::SafeTensorsSave {
            path: path.to_path_buf(),
            message: "posedirs is not contiguous".to_string(),
        })?;

    let j_regressor = model
        .j_regressor
        .as_slice()
        .ok_or_else(|| FlameError::SafeTensorsSave {
            path: path.to_path_buf(),
            message: "j_regressor is not contiguous".to_string(),
        })?;

    let lbs_weights = model
        .lbs_weights
        .as_slice()
        .ok_or_else(|| FlameError::SafeTensorsSave {
            path: path.to_path_buf(),
            message: "lbs_weights is not contiguous".to_string(),
        })?;

    // Convert faces to i32 array for serialization
    let faces_i32: Vec<i32> = model
        .faces
        .iter()
        .flat_map(|face| face.iter().map(|&idx| idx.cast_signed()))
        .collect();

    // Convert parents to kintree_table format
    let kintree_i32: Vec<i32> = model.parents.clone();

    Ok(ModelDataSlices {
        v_template,
        shapedirs,
        expressiondirs,
        posedirs,
        j_regressor,
        lbs_weights,
        faces_i32,
        kintree_i32,
    })
}

/// Create tensor views from model data for safetensors serialization.
#[allow(clippy::type_complexity)]
fn create_tensor_views<'a>(
    model: &FlameModel,
    slices: &'a ModelDataSlices<'a>,
    path: &Path,
) -> Result<Vec<(&'static str, TensorView<'a>)>, FlameError> {
    // Convert f32 slices to bytes
    let v_template_bytes = bytemuck::cast_slice(slices.v_template);
    let shapedirs_bytes = bytemuck::cast_slice(slices.shapedirs);
    let expressiondirs_bytes = bytemuck::cast_slice(slices.expressiondirs);
    let posedirs_bytes = bytemuck::cast_slice(slices.posedirs);
    let j_regressor_bytes = bytemuck::cast_slice(slices.j_regressor);
    let lbs_weights_bytes = bytemuck::cast_slice(slices.lbs_weights);
    let faces_bytes = bytemuck::cast_slice(&slices.faces_i32);
    let kintree_bytes = bytemuck::cast_slice(&slices.kintree_i32);

    // Create tensor views
    let tensors = vec![
        (
            "v_template",
            TensorView::new(
                Dtype::F32,
                model.v_template.shape().to_vec(),
                v_template_bytes,
            )
            .map_err(|e| FlameError::SafeTensorsSave {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?,
        ),
        (
            "faces",
            TensorView::new(Dtype::I32, vec![model.faces.len(), 3], faces_bytes).map_err(|e| {
                FlameError::SafeTensorsSave {
                    path: path.to_path_buf(),
                    message: e.to_string(),
                }
            })?,
        ),
        (
            "shapedirs",
            TensorView::new(
                Dtype::F32,
                model.shapedirs.shape().to_vec(),
                shapedirs_bytes,
            )
            .map_err(|e| FlameError::SafeTensorsSave {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?,
        ),
        (
            "expressiondirs",
            TensorView::new(
                Dtype::F32,
                model.expressiondirs.shape().to_vec(),
                expressiondirs_bytes,
            )
            .map_err(|e| FlameError::SafeTensorsSave {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?,
        ),
        (
            "posedirs",
            TensorView::new(Dtype::F32, model.posedirs.shape().to_vec(), posedirs_bytes).map_err(
                |e| FlameError::SafeTensorsSave {
                    path: path.to_path_buf(),
                    message: e.to_string(),
                },
            )?,
        ),
        (
            "j_regressor",
            TensorView::new(
                Dtype::F32,
                model.j_regressor.shape().to_vec(),
                j_regressor_bytes,
            )
            .map_err(|e| FlameError::SafeTensorsSave {
                path: path.to_path_buf(),
                message: e.to_string(),
            })?,
        ),
        (
            "kintree_table",
            TensorView::new(Dtype::I32, vec![1, model.n_joints], kintree_bytes).map_err(
                |e: safetensors::SafeTensorError| FlameError::SafeTensorsSave {
                    path: path.to_path_buf(),
                    message: e.to_string(),
                },
            )?,
        ),
        (
            "lbs_weights",
            TensorView::new(
                Dtype::F32,
                model.lbs_weights.shape().to_vec(),
                lbs_weights_bytes,
            )
            .map_err(|e: safetensors::SafeTensorError| {
                FlameError::SafeTensorsSave {
                    path: path.to_path_buf(),
                    message: e.to_string(),
                }
            })?,
        ),
    ];

    Ok(tensors)
}

/// Serialize and write safetensors data to file.
fn write_safetensors_to_file(
    tensors: Vec<(&str, TensorView<'_>)>,
    metadata: Option<&HashMap<String, String>>,
    path: &Path,
) -> Result<(), FlameError> {
    // Serialize with optional metadata
    let metadata_owned = metadata.cloned();
    let serialized = safetensors::tensor::serialize(tensors, metadata_owned).map_err(
        |e: safetensors::SafeTensorError| FlameError::SafeTensorsSave {
            path: path.to_path_buf(),
            message: e.to_string(),
        },
    )?;

    // Write to file
    let mut file = File::create(path).map_err(|e| FlameError::IoError {
        source: e,
        path: path.to_path_buf(),
    })?;
    file.write_all(&serialized)
        .map_err(|e| FlameError::IoError {
            source: e,
            path: path.to_path_buf(),
        })?;
    file.flush().map_err(|e| FlameError::IoError {
        source: e,
        path: path.to_path_buf(),
    })?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helper functions for loading
// ---------------------------------------------------------------------------

/// Load a 2D f32 tensor from safetensors
fn load_tensor_f32_2d(tensors: &SafeTensors, name: &str) -> Result<Array2<f32>, FlameError> {
    let tensor_view = tensors
        .tensor(name)
        .map_err(
            |e: safetensors::SafeTensorError| FlameError::SafeTensorsMissing {
                name: name.to_string(),
                message: e.to_string(),
            },
        )?;

    // Verify dtype
    if tensor_view.dtype() != safetensors::Dtype::F32 {
        return Err(FlameError::SafeTensorsInvalidDtype {
            name: name.to_string(),
            expected: "F32".to_string(),
            got: format!("{:?}", tensor_view.dtype()),
        });
    }

    // Get shape
    let shape = tensor_view.shape();
    if shape.len() != 2 {
        return Err(FlameError::ShapeMismatch {
            name: name.to_string(),
            expected: "2D array".to_string(),
            got: format!("{shape:?}"),
        });
    }

    // Convert bytes to f32 slice
    let data_bytes = tensor_view.data();
    let data_f32: &[f32] = bytemuck::cast_slice(data_bytes);

    // Create ndarray
    Array2::from_shape_vec((shape[0], shape[1]), data_f32.to_vec()).map_err(|e| {
        FlameError::ShapeMismatch {
            name: name.to_string(),
            expected: format!("{shape:?}"),
            got: e.to_string(),
        }
    })
}

/// Load a 3D f32 tensor from safetensors
fn load_tensor_f32_3d(tensors: &SafeTensors, name: &str) -> Result<Array3<f32>, FlameError> {
    let tensor_view = tensors
        .tensor(name)
        .map_err(
            |e: safetensors::SafeTensorError| FlameError::SafeTensorsMissing {
                name: name.to_string(),
                message: e.to_string(),
            },
        )?;

    // Verify dtype
    if tensor_view.dtype() != safetensors::Dtype::F32 {
        return Err(FlameError::SafeTensorsInvalidDtype {
            name: name.to_string(),
            expected: "F32".to_string(),
            got: format!("{:?}", tensor_view.dtype()),
        });
    }

    // Get shape
    let shape = tensor_view.shape();
    if shape.len() != 3 {
        return Err(FlameError::ShapeMismatch {
            name: name.to_string(),
            expected: "3D array".to_string(),
            got: format!("{shape:?}"),
        });
    }

    // Convert bytes to f32 slice
    let data_bytes = tensor_view.data();
    let data_f32: &[f32] = bytemuck::cast_slice(data_bytes);

    // Create ndarray
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), data_f32.to_vec()).map_err(|e| {
        FlameError::ShapeMismatch {
            name: name.to_string(),
            expected: format!("{shape:?}"),
            got: e.to_string(),
        }
    })
}

/// Load a 2D i32 tensor from safetensors
fn load_tensor_i32_2d(tensors: &SafeTensors, name: &str) -> Result<Array2<i32>, FlameError> {
    let tensor_view = tensors
        .tensor(name)
        .map_err(
            |e: safetensors::SafeTensorError| FlameError::SafeTensorsMissing {
                name: name.to_string(),
                message: e.to_string(),
            },
        )?;

    // Verify dtype
    if tensor_view.dtype() != safetensors::Dtype::I32 {
        return Err(FlameError::SafeTensorsInvalidDtype {
            name: name.to_string(),
            expected: "I32".to_string(),
            got: format!("{:?}", tensor_view.dtype()),
        });
    }

    // Get shape
    let shape = tensor_view.shape();
    if shape.len() != 2 {
        return Err(FlameError::ShapeMismatch {
            name: name.to_string(),
            expected: "2D array".to_string(),
            got: format!("{shape:?}"),
        });
    }

    // Convert bytes to i32 slice
    let data_bytes = tensor_view.data();
    let data_i32: &[i32] = bytemuck::cast_slice(data_bytes);

    // Create ndarray
    Array2::from_shape_vec((shape[0], shape[1]), data_i32.to_vec()).map_err(|e| {
        FlameError::ShapeMismatch {
            name: name.to_string(),
            expected: format!("{shape:?}"),
            got: e.to_string(),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};
    use std::collections::HashMap;
    use tempfile::TempDir;

    fn create_minimal_flame_model() -> FlameModel {
        // Create a minimal FLAME model for testing
        let n_verts = 10;
        let n_faces = 5;
        let n_joints = 5;
        let n_shape = 3;
        let n_expr = 2;
        let n_pose_dirs = 4;

        FlameModel {
            v_template: Array2::zeros((n_verts, 3)),
            faces: vec![[0, 1, 2]; n_faces],
            shapedirs: Array3::zeros((n_verts, 3, n_shape)),
            expressiondirs: Array3::zeros((n_verts, 3, n_expr)),
            posedirs: Array3::zeros((n_verts, 3, n_pose_dirs)),
            j_regressor: Array2::zeros((n_joints, n_verts)),
            parents: vec![-1, 0, 1, 2, 3],
            lbs_weights: Array2::zeros((n_verts, n_joints)),
            n_joints,
        }
    }

    #[test]
    fn test_save_and_load_round_trip() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let safetensors_path = temp_dir.path().join("test_model.safetensors");

        // Create test model
        let model = create_minimal_flame_model();

        // Save to safetensors
        save_flame_model_safetensors(&model, &safetensors_path, None)
            .expect("test: save should succeed");

        // Load back
        let loaded_model =
            load_flame_model_safetensors(&safetensors_path).expect("test: load should succeed");

        // Verify shapes match
        assert_eq!(loaded_model.v_template.shape(), model.v_template.shape());
        assert_eq!(loaded_model.faces.len(), model.faces.len());
        assert_eq!(loaded_model.shapedirs.shape(), model.shapedirs.shape());
        assert_eq!(
            loaded_model.expressiondirs.shape(),
            model.expressiondirs.shape()
        );
        assert_eq!(loaded_model.posedirs.shape(), model.posedirs.shape());
        assert_eq!(loaded_model.j_regressor.shape(), model.j_regressor.shape());
        assert_eq!(loaded_model.parents.len(), model.parents.len());
        assert_eq!(loaded_model.lbs_weights.shape(), model.lbs_weights.shape());
        assert_eq!(loaded_model.n_joints, model.n_joints);
    }

    #[test]
    fn test_metadata_preservation() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let safetensors_path = temp_dir.path().join("test_model_meta.safetensors");

        // Create test model
        let model = create_minimal_flame_model();

        // Add metadata
        let mut metadata = HashMap::new();
        metadata.insert("version".to_string(), "1.0".to_string());
        metadata.insert("source".to_string(), "test".to_string());
        metadata.insert("author".to_string(), "oxigaf-flame".to_string());

        // Save with metadata
        save_flame_model_safetensors(&model, &safetensors_path, Some(&metadata))
            .expect("test: save should succeed");

        // Verify file was created and can be loaded
        assert!(safetensors_path.exists());
        let loaded_model =
            load_flame_model_safetensors(&safetensors_path).expect("test: load should succeed");

        // Verify model structure
        assert_eq!(loaded_model.num_vertices(), model.num_vertices());
        assert_eq!(loaded_model.n_joints, model.n_joints);
    }

    #[test]
    fn test_save_with_non_contiguous_arrays() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let safetensors_path = temp_dir.path().join("test_model_slice.safetensors");

        // Create test model with potentially non-contiguous arrays
        let mut model = create_minimal_flame_model();

        // Make array contiguous by cloning (this is what as_slice requires)
        model.v_template = model.v_template.as_standard_layout().into_owned();

        // Should succeed with contiguous arrays
        let result = save_flame_model_safetensors(&model, &safetensors_path, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_missing_file() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let missing_path = temp_dir.path().join("nonexistent.safetensors");

        let result = load_flame_model_safetensors(&missing_path);
        assert!(result.is_err());

        if let Err(FlameError::IoError { source: _, path }) = result {
            assert_eq!(path, missing_path);
        } else {
            panic!("Expected IoError");
        }
    }

    #[test]
    fn test_round_trip_preserves_data() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let safetensors_path = temp_dir.path().join("test_data_preservation.safetensors");

        // Create model with specific values
        let mut model = create_minimal_flame_model();
        model.v_template[[0, 0]] = 1.5;
        model.v_template[[0, 1]] = -2.3;
        model.v_template[[1, 2]] = 0.7;
        model.shapedirs[[2, 1, 0]] = std::f32::consts::PI;
        model.parents[2] = 1;

        // Save and load
        save_flame_model_safetensors(&model, &safetensors_path, None)
            .expect("test: save should succeed");
        let loaded =
            load_flame_model_safetensors(&safetensors_path).expect("test: load should succeed");

        // Verify data preservation
        assert!((loaded.v_template[[0, 0]] - 1.5).abs() < 1e-6);
        assert!((loaded.v_template[[0, 1]] - (-2.3)).abs() < 1e-6);
        assert!((loaded.v_template[[1, 2]] - 0.7).abs() < 1e-6);
        assert!((loaded.shapedirs[[2, 1, 0]] - std::f32::consts::PI).abs() < 1e-6);
        assert_eq!(loaded.parents[2], 1);
    }

    #[test]
    fn test_faces_conversion() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let safetensors_path = temp_dir.path().join("test_faces.safetensors");

        // Create model with specific face indices
        let mut model = create_minimal_flame_model();
        model.faces = vec![[0, 1, 2], [3, 4, 5], [6, 7, 8]];

        // Save and load
        save_flame_model_safetensors(&model, &safetensors_path, None)
            .expect("test: save should succeed");
        let loaded =
            load_flame_model_safetensors(&safetensors_path).expect("test: load should succeed");

        // Verify faces preserved correctly
        assert_eq!(loaded.faces.len(), 3);
        assert_eq!(loaded.faces[0], [0, 1, 2]);
        assert_eq!(loaded.faces[1], [3, 4, 5]);
        assert_eq!(loaded.faces[2], [6, 7, 8]);
    }
}
