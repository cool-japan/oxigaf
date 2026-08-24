//! Load and save FLAME models using the safetensors format.
//!
//! This module provides an alternative to NPY files for storing FLAME models.
//! Safetensors is a modern, efficient format widely used in ML frameworks.
//!
//! ## Advantages of safetensors
//!
//! - **Single file**: All model data in one file instead of multiple `.npy` files
//! - **Metadata support**: Store model version, source, creation date, etc.
//! - **Fast loading**: Single-read, zero-copy tensor views (the whole file is
//!   read into memory once; no per-tensor file I/O). This crate does not use
//!   memory-mapping.
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

    // --- Validate shapes that don't depend on downstream conversions ---
    // Mirrors `io::load_flame_model`'s validation so a malformed or
    // untrusted .safetensors file is rejected here instead of panicking
    // later inside `forward()` (ndarray shape mismatch, out-of-range
    // kinematic-chain parent, or out-of-range face index).
    let n_verts = v_template.nrows();
    expect_shape("v_template", &[n_verts, 3], v_template.shape())?;

    // --- Extract parent indices from kintree_table row 0, and validate the
    // kinematic chain: every non-root joint must point to a strictly earlier
    // joint index, matching the traversal order
    // `FlameModel::compute_skinning_transforms` relies on. ---
    let n_joints = kintree_i32.ncols();
    let parents: Vec<i32> = (0..n_joints).map(|j| kintree_i32[[0, j]]).collect();
    validate_parents(&parents)?;

    expect_shape("j_regressor", &[n_joints, n_verts], j_regressor.shape())?;
    expect_shape("lbs_weights", &[n_verts, n_joints], lbs_weights.shape())?;

    // --- Validate blend-shape directions share the vertex/component layout
    // that `apply_blend_shapes` assumes when it slices them against `v`. ---
    expect_dir_shape("shapedirs", shapedirs.shape(), n_verts, None)?;
    expect_dir_shape("expressiondirs", expressiondirs.shape(), n_verts, None)?;
    expect_dir_shape(
        "posedirs",
        posedirs.shape(),
        n_verts,
        Some((n_joints.max(1) - 1) * 9),
    )?;

    // --- Convert faces from i32 → Vec<[u32; 3]>, rejecting indices that
    // would panic later in `Mesh::recompute_normals` or GPU upload. ---
    let faces = convert_faces(&faces_i32, n_verts)?;

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
        joint_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
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

    // Convert parents to kintree_table format: row 0 holds each joint's
    // parent index (the only row this crate's loader reads back via
    // `kintree_i32[[0, j]]`), row 1 holds the joint's own index. This
    // matches the documented `[2, n_joints]` shape and the SMPL/FLAME
    // convention of a parents row plus a self/children row -- writing only
    // `model.parents` (shape `[1, n_joints]`) contradicted the module's own
    // format table.
    let mut kintree_i32: Vec<i32> = Vec::with_capacity(model.parents.len() * 2);
    kintree_i32.extend_from_slice(&model.parents);
    kintree_i32.extend((0..model.n_joints).map(|j| j as i32));

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
            TensorView::new(Dtype::I32, vec![2, model.n_joints], kintree_bytes).map_err(
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

    // Convert bytes to f32 (see `bytes_to_f32_vec` for why this cannot use
    // `bytemuck::cast_slice` directly).
    let data_f32 = bytes_to_f32_vec(name, tensor_view.data())?;

    // Create ndarray
    Array2::from_shape_vec((shape[0], shape[1]), data_f32).map_err(|e| FlameError::ShapeMismatch {
        name: name.to_string(),
        expected: format!("{shape:?}"),
        got: e.to_string(),
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

    // Convert bytes to f32 (see `bytes_to_f32_vec` for why this cannot use
    // `bytemuck::cast_slice` directly).
    let data_f32 = bytes_to_f32_vec(name, tensor_view.data())?;

    // Create ndarray
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), data_f32).map_err(|e| {
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

    // Convert bytes to i32 (see `bytes_to_f32_vec` for why this cannot use
    // `bytemuck::cast_slice` directly).
    let data_i32 = bytes_to_i32_vec(name, tensor_view.data())?;

    // Create ndarray
    Array2::from_shape_vec((shape[0], shape[1]), data_i32).map_err(|e| FlameError::ShapeMismatch {
        name: name.to_string(),
        expected: format!("{shape:?}"),
        got: e.to_string(),
    })
}

/// Decode little-endian bytes into a `Vec<f32>` without requiring the source
/// slice to be 4-byte aligned.
///
/// `bytemuck::cast_slice::<u8, f32>` PANICS
/// (`TargetAlignmentGreaterAndInputNotAligned`) whenever the source pointer
/// is not 4-byte aligned, and also panics (`OutputSliceWouldHaveSlop`) when
/// the length is not a multiple of 4. A tensor's byte payload inside a
/// safetensors file starts at `8 + header_len + tensor_offset`; safetensors
/// does not pad the JSON header, so alignment is a file-controlled property
/// this crate cannot assume for an arbitrary (possibly untrusted or
/// corrupted) input file. Decoding byte-by-byte via `from_le_bytes` works
/// for any alignment and lets a malformed length be reported as a
/// [`FlameError`] instead of aborting the process.
fn bytes_to_f32_vec(name: &str, data_bytes: &[u8]) -> Result<Vec<f32>, FlameError> {
    if data_bytes.len() % 4 != 0 {
        return Err(FlameError::ShapeMismatch {
            name: name.to_string(),
            expected: "byte length that is a multiple of 4 (f32)".to_string(),
            got: format!("{} bytes", data_bytes.len()),
        });
    }
    Ok(data_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Decode little-endian bytes into a `Vec<i32>`. See [`bytes_to_f32_vec`]
/// for why a manual, alignment-independent decode is required here instead
/// of `bytemuck::cast_slice`.
fn bytes_to_i32_vec(name: &str, data_bytes: &[u8]) -> Result<Vec<i32>, FlameError> {
    if data_bytes.len() % 4 != 0 {
        return Err(FlameError::ShapeMismatch {
            name: name.to_string(),
            expected: "byte length that is a multiple of 4 (i32)".to_string(),
            got: format!("{} bytes", data_bytes.len()),
        });
    }
    Ok(data_bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Assert that an array has an expected shape.
///
/// Mirrors `io::expect_shape` (duplicated here since that helper is
/// private to the NPY loader and this module has its own independent
/// loading path).
fn expect_shape(name: &str, expected: &[usize], got: &[usize]) -> Result<(), FlameError> {
    if expected != got {
        return Err(FlameError::ShapeMismatch {
            name: name.to_string(),
            expected: format!("{expected:?}"),
            got: format!("{got:?}"),
        });
    }
    Ok(())
}

/// Validate a blend-shape direction array's shape against `[n_verts, 3, K]`.
///
/// The first two dimensions are always checked; the trailing component
/// count `K` is only checked when `expected_k` is `Some` (shape/expression
/// basis sizes are data-driven, but the pose-corrective basis size is fixed
/// by `n_joints`). Mirrors `io::expect_dir_shape`.
fn expect_dir_shape(
    name: &str,
    shape: &[usize],
    n_verts: usize,
    expected_k: Option<usize>,
) -> Result<(), FlameError> {
    let dims_ok = shape.len() == 3 && shape[0] == n_verts && shape[1] == 3;
    let k_ok = match expected_k {
        Some(k) => shape.len() == 3 && shape[2] == k,
        None => true,
    };
    if !dims_ok || !k_ok {
        let expected = match expected_k {
            Some(k) => format!("[{n_verts}, 3, {k}]"),
            None => format!("[{n_verts}, 3, K]"),
        };
        return Err(FlameError::ShapeMismatch {
            name: name.to_string(),
            expected,
            got: format!("{shape:?}"),
        });
    }
    Ok(())
}

/// Validate that `parents` describes a well-formed kinematic chain: the root
/// joint (index 0) carries a negative parent marker, and every other
/// joint's parent is a strictly earlier joint index.
///
/// `FlameModel::compute_skinning_transforms` builds global transforms by
/// iterating joints `0..n_joints` in order and reading `global[parent]`,
/// which is only valid (and only correct -- not merely non-panicking) when
/// `parent < j`. Mirrors `io::validate_parents`.
fn validate_parents(parents: &[i32]) -> Result<(), FlameError> {
    if parents.is_empty() {
        return Err(FlameError::InvalidParams(
            "kintree_table must describe at least one joint".to_string(),
        ));
    }
    if parents[0] >= 0 {
        return Err(FlameError::InvalidParams(format!(
            "parents[0] = {} must be negative (root joint marker)",
            parents[0]
        )));
    }
    for (j, &p) in parents.iter().enumerate().skip(1) {
        let j_i32 = j as i32;
        if p < 0 || p >= j_i32 {
            return Err(FlameError::InvalidParams(format!(
                "parents[{j}] = {p} is not a valid ancestor; expected 0 <= parents[{j}] < {j}"
            )));
        }
    }
    Ok(())
}

/// Convert raw `i32` face indices to `[u32; 3]` triangles, rejecting any
/// index that is negative or falls outside `0..n_verts`.
///
/// Without this check a negative index silently wraps to a huge `u32` under
/// `as` casting, and an out-of-range positive index is only caught much
/// later (and only as a panic) inside `Mesh::recompute_normals` or GPU
/// buffer upload. Mirrors `io::convert_faces`.
fn convert_faces(faces_i32: &Array2<i32>, n_verts: usize) -> Result<Vec<[u32; 3]>, FlameError> {
    let mut faces = Vec::with_capacity(faces_i32.nrows());
    for (row_idx, row) in faces_i32.rows().into_iter().enumerate() {
        let mut tri = [0u32; 3];
        for (c, slot) in tri.iter_mut().enumerate() {
            let v = row[c];
            if v < 0 || v as usize >= n_verts {
                return Err(FlameError::IndexOutOfBounds {
                    context: format!("faces[{row_idx}][{c}]"),
                    index: usize::try_from(v).unwrap_or(usize::MAX),
                    len: n_verts,
                });
            }
            *slot = v as u32;
        }
        faces.push(tri);
    }
    Ok(faces)
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
        // Must equal (n_joints - 1) * 9 to pass `expect_dir_shape`'s
        // posedirs check in `load_flame_model_safetensors` (pose-corrective
        // basis size is fixed by the kinematic chain, unlike shapedirs /
        // expressiondirs which are data-driven).
        let n_pose_dirs = (n_joints - 1) * 9;

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
            joint_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
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

    // -----------------------------------------------------------------------
    // Alignment-independent byte decoding (regression for the
    // `bytemuck::cast_slice` panic on unaligned/short tensor payloads).
    // -----------------------------------------------------------------------

    #[test]
    fn test_bytes_to_f32_vec_rejects_non_multiple_of_4_length() {
        let bytes = [0u8; 6]; // not a multiple of 4
        let result = bytes_to_f32_vec("test_tensor", &bytes);
        assert!(matches!(result, Err(FlameError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_bytes_to_f32_vec_handles_unaligned_slice_without_panicking() {
        // Build a buffer where the f32 payload starts at a 1-byte offset
        // from the start of the allocation, so the slice's alignment is not
        // guaranteed to be 4 -- exactly the situation a safetensors file
        // produces whenever `8 + header_len + tensor_offset` is not a
        // multiple of 4. `bytemuck::cast_slice` would panic on this input.
        let value: f32 = 1234.5;
        let mut buf = vec![0xAAu8]; // 1 padding byte to misalign what follows
        buf.extend_from_slice(&value.to_le_bytes());
        let payload = &buf[1..]; // 4 bytes, not necessarily 4-byte aligned
        let decoded = bytes_to_f32_vec("test_tensor", payload).expect("should decode");
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0] - value).abs() < 1e-3);
    }

    #[test]
    fn test_bytes_to_i32_vec_rejects_non_multiple_of_4_length() {
        let bytes = [0u8; 5];
        let result = bytes_to_i32_vec("test_tensor", &bytes);
        assert!(matches!(result, Err(FlameError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_bytes_to_i32_vec_handles_unaligned_slice_without_panicking() {
        let value: i32 = -987_654;
        let mut buf = vec![0xAAu8, 0xBBu8, 0xCCu8]; // 3 padding bytes
        buf.extend_from_slice(&value.to_le_bytes());
        let payload = &buf[3..];
        let decoded = bytes_to_i32_vec("test_tensor", payload).expect("should decode");
        assert_eq!(decoded, vec![value]);
    }

    // -----------------------------------------------------------------------
    // Shape validation helpers (regression for the "no shape validation at
    // all despite documenting it" finding).
    // -----------------------------------------------------------------------

    #[test]
    fn test_expect_shape_rejects_mismatch() {
        assert!(expect_shape("v_template", &[10, 3], &[10, 3]).is_ok());
        assert!(expect_shape("v_template", &[10, 3], &[10, 4]).is_err());
    }

    #[test]
    fn test_validate_parents_accepts_well_formed_chain() {
        assert!(validate_parents(&[-1, 0, 1, 1, 3]).is_ok());
    }

    #[test]
    fn test_validate_parents_rejects_non_negative_root() {
        let err = validate_parents(&[0, 0]).unwrap_err();
        assert!(matches!(err, FlameError::InvalidParams(_)));
    }

    #[test]
    fn test_validate_parents_rejects_forward_reference() {
        let err = validate_parents(&[-1, 2, 0]).unwrap_err();
        assert!(matches!(err, FlameError::InvalidParams(_)));
    }

    #[test]
    fn test_validate_parents_rejects_empty() {
        assert!(validate_parents(&[]).is_err());
    }

    #[test]
    fn test_convert_faces_rejects_out_of_range_index() {
        let faces_i32 = Array2::from_shape_vec((1, 3), vec![0, 1, 5])
            .expect("valid (1, 3) shape for a 3-element vec");
        let err = convert_faces(&faces_i32, 3).unwrap_err();
        assert!(matches!(err, FlameError::IndexOutOfBounds { len: 3, .. }));
    }

    #[test]
    fn test_convert_faces_rejects_negative_index() {
        let faces_i32 = Array2::from_shape_vec((1, 3), vec![0, 1, -1])
            .expect("valid (1, 3) shape for a 3-element vec");
        let err = convert_faces(&faces_i32, 3).unwrap_err();
        assert!(matches!(err, FlameError::IndexOutOfBounds { .. }));
    }

    #[test]
    fn test_expect_dir_shape_checks_leading_dims() {
        let dirs = Array3::<f32>::zeros((5, 3, 10));
        assert!(expect_dir_shape("shapedirs", dirs.shape(), 5, None).is_ok());
        assert!(expect_dir_shape("shapedirs", dirs.shape(), 6, None).is_err());
    }

    #[test]
    fn test_expect_dir_shape_checks_trailing_k_when_requested() {
        let dirs = Array3::<f32>::zeros((5, 3, 36));
        assert!(expect_dir_shape("posedirs", dirs.shape(), 5, Some(36)).is_ok());
        assert!(expect_dir_shape("posedirs", dirs.shape(), 5, Some(9)).is_err());
    }

    /// End-to-end regression: a hand-built, on-disk `.safetensors` file
    /// (bypassing `save_flame_model_safetensors`, which always writes
    /// internally-consistent shapes) with a `j_regressor` tensor that does
    /// not match `[n_joints, n_verts]` must be rejected by
    /// `load_flame_model_safetensors` with a typed error, not accepted and
    /// left to panic later inside `forward()`.
    #[test]
    fn test_load_flame_model_safetensors_rejects_mismatched_j_regressor_shape() {
        let temp_dir = TempDir::new().expect("test: temp dir creation should succeed");
        let path = temp_dir.path().join("bad_j_regressor.safetensors");

        let n_verts = 10usize;
        let n_joints = 5usize;
        let v_template = vec![0.0f32; n_verts * 3];
        let faces = vec![0i32; 5 * 3];
        let shapedirs = vec![0.0f32; n_verts * 3 * 3];
        let expressiondirs = vec![0.0f32; n_verts * 3 * 2];
        let posedirs = vec![0.0f32; n_verts * 3 * (n_joints - 1) * 9];
        // Deliberately wrong: must be [n_joints, n_verts] = [5, 10].
        let j_regressor = vec![0.0f32; n_joints * n_joints];
        let kintree: Vec<i32> = vec![-1, 0, 1, 2, 3, 0, 1, 2, 3, 4];
        let lbs_weights = vec![0.0f32; n_verts * n_joints];

        let tensors = vec![
            (
                "v_template",
                TensorView::new(
                    Dtype::F32,
                    vec![n_verts, 3],
                    bytemuck::cast_slice(&v_template),
                )
                .expect("view"),
            ),
            (
                "faces",
                TensorView::new(Dtype::I32, vec![5, 3], bytemuck::cast_slice(&faces))
                    .expect("view"),
            ),
            (
                "shapedirs",
                TensorView::new(
                    Dtype::F32,
                    vec![n_verts, 3, 3],
                    bytemuck::cast_slice(&shapedirs),
                )
                .expect("view"),
            ),
            (
                "expressiondirs",
                TensorView::new(
                    Dtype::F32,
                    vec![n_verts, 3, 2],
                    bytemuck::cast_slice(&expressiondirs),
                )
                .expect("view"),
            ),
            (
                "posedirs",
                TensorView::new(
                    Dtype::F32,
                    vec![n_verts, 3, (n_joints - 1) * 9],
                    bytemuck::cast_slice(&posedirs),
                )
                .expect("view"),
            ),
            (
                "j_regressor",
                TensorView::new(
                    Dtype::F32,
                    vec![n_joints, n_joints],
                    bytemuck::cast_slice(&j_regressor),
                )
                .expect("view"),
            ),
            (
                "kintree_table",
                TensorView::new(
                    Dtype::I32,
                    vec![2, n_joints],
                    bytemuck::cast_slice(&kintree),
                )
                .expect("view"),
            ),
            (
                "lbs_weights",
                TensorView::new(
                    Dtype::F32,
                    vec![n_verts, n_joints],
                    bytemuck::cast_slice(&lbs_weights),
                )
                .expect("view"),
            ),
        ];
        let serialized = safetensors::tensor::serialize(tensors, None).expect("serialize");
        std::fs::write(&path, serialized).expect("write");

        // `FlameModel` does not implement `Debug`, so match manually rather
        // than formatting the whole `Result` in the assertion message.
        match load_flame_model_safetensors(&path) {
            Err(FlameError::ShapeMismatch { .. }) => {}
            Err(e) => panic!(
                "expected ShapeMismatch for a malformed j_regressor, got a different error: {e}"
            ),
            Ok(_) => {
                panic!("expected ShapeMismatch for a malformed j_regressor, but load succeeded")
            }
        }
    }
}
