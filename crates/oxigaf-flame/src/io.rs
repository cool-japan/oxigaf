//! Load FLAME model data from a directory of `.npy` files.

use std::path::Path;

use ndarray::{Array2, Array3};
use ndarray_npy::read_npy;

use crate::error::FlameError;
use crate::model::FlameModel;

/// Load a [`FlameModel`] from a directory of `.npy` files.
///
/// Expected files (produced by `scripts/convert_flame.py`):
///
/// | File                | Shape              | dtype   |
/// |---------------------|--------------------|---------|
/// | `v_template.npy`    | `[5023, 3]`        | float32 |
/// | `faces.npy`         | `[9976, 3]`        | int32   |
/// | `shapedirs.npy`     | `[5023, 3, 300]`   | float32 |
/// | `expressiondirs.npy`| `[5023, 3, 100]`   | float32 |
/// | `posedirs.npy`      | `[5023, 3, 36]`    | float32 |
/// | `j_regressor.npy`   | `[5, 5023]`        | float32 |
/// | `kintree_table.npy` | `[2, 5]`           | int32   |
/// | `lbs_weights.npy`   | `[5023, 5]`        | float32 |
///
/// # Errors
///
/// Returns an error if:
/// - The directory does not exist
/// - Required `.npy` files are missing or cannot be read
/// - Array shapes do not match expected dimensions
pub fn load_flame_model(dir: &Path) -> Result<FlameModel, FlameError> {
    if !dir.is_dir() {
        return Err(FlameError::ModelDir(format!(
            "Not a directory: {}",
            dir.display()
        )));
    }

    let v_template: Array2<f32> = load_npy(dir, "v_template")?;
    let faces_i32: Array2<i32> = load_npy(dir, "faces")?;
    let shapedirs: Array3<f32> = load_npy(dir, "shapedirs")?;
    let expressiondirs: Array3<f32> = load_npy(dir, "expressiondirs")?;
    let posedirs: Array3<f32> = load_npy(dir, "posedirs")?;
    let j_regressor: Array2<f32> = load_npy(dir, "j_regressor")?;
    let kintree_i32: Array2<i32> = load_npy(dir, "kintree_table")?;
    let lbs_weights: Array2<f32> = load_npy(dir, "lbs_weights")?;

    // --- Convert faces from i32 → Vec<[u32; 3]> ---
    let faces: Vec<[u32; 3]> = faces_i32
        .rows()
        .into_iter()
        .map(|row| [row[0] as u32, row[1] as u32, row[2] as u32])
        .collect();

    // --- Extract parent indices from kintree_table row 0 ---
    let n_joints = kintree_i32.ncols();
    let parents: Vec<i32> = (0..n_joints).map(|j| kintree_i32[[0, j]]).collect();

    // --- Validate shapes ---
    let n_verts = v_template.nrows();
    expect_shape("v_template", &[n_verts, 3], v_template.shape())?;
    expect_shape("j_regressor", &[n_joints, n_verts], j_regressor.shape())?;
    expect_shape("lbs_weights", &[n_verts, n_joints], lbs_weights.shape())?;

    tracing::info!(
        n_verts,
        n_faces = faces.len(),
        n_joints,
        n_shape = shapedirs.shape()[2],
        n_expr = expressiondirs.shape()[2],
        "FLAME model loaded"
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Load a single `.npy` file from a directory.
fn load_npy<A, D>(dir: &Path, name: &str) -> Result<ndarray::Array<A, D>, FlameError>
where
    A: ndarray_npy::ReadableElement,
    D: ndarray::Dimension,
{
    let path = dir.join(format!("{name}.npy"));
    read_npy(&path).map_err(|source| FlameError::NpyLoad {
        name: name.to_string(),
        source,
    })
}

/// Assert that an array has an expected shape.
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
