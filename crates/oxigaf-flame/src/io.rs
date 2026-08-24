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
/// - Face indices reference a vertex outside `v_template`
/// - `kintree_table` describes a parent index that is not a strictly earlier
///   joint (which would make the kinematic chain ill-defined or panic during
///   skinning)
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

    // --- Validate shapes that don't depend on downstream conversions ---
    let n_verts = v_template.nrows();
    expect_shape("v_template", &[n_verts, 3], v_template.shape())?;

    // --- Extract parent indices from kintree_table row 0, and validate the
    // kinematic chain: every non-root joint must point to a strictly earlier
    // joint index, matching the traversal order `compute_skinning_transforms`
    // relies on. ---
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

/// Validate a blend-shape direction array's shape against `[n_verts, 3, K]`.
///
/// The first two dimensions are always checked; the trailing component count
/// `K` is only checked when `expected_k` is `Some` (shape/expression basis
/// sizes are data-driven, but the pose-corrective basis size is fixed by
/// `n_joints`).
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
/// joint (index 0) carries a root marker, and every other joint's parent is
/// a strictly earlier joint index.
///
/// `FlameModel::compute_skinning_transforms` builds global transforms by
/// iterating joints `0..n_joints` in order and reading `global[parent]`,
/// which is only valid (and only correct — not merely non-panicking) when
/// `parent < j`. The root itself is accepted both as the conventional `< 0`
/// sentinel and as a self-referencing `0` (some SMPL/FLAME conversion
/// pipelines emit either); `compute_skinning_transforms` handles both
/// without panicking, so both are treated as valid root markers here. Any
/// other non-negative value for `parents[0]` is rejected: it does not match
/// either recognised root convention, and — if it happens to be `>=
/// n_joints` — would index `joints`/`global` out of bounds.
fn validate_parents(parents: &[i32]) -> Result<(), FlameError> {
    if parents.is_empty() {
        return Err(FlameError::InvalidParams(
            "kintree_table must describe at least one joint".to_string(),
        ));
    }
    if parents[0] > 0 {
        return Err(FlameError::InvalidParams(format!(
            "parents[0] = {} must be a root marker (negative, or 0 for a \
             self-referencing root)",
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
/// buffer upload.
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

    #[test]
    fn validate_parents_accepts_well_formed_chain() {
        // Standard FLAME kinematic chain: root, then each joint pointing to
        // an earlier one.
        assert!(validate_parents(&[-1, 0, 1, 1, 3]).is_ok());
    }

    #[test]
    fn validate_parents_accepts_self_referencing_root() {
        // Some SMPL/FLAME conversion pipelines emit `0` rather than a
        // negative sentinel for the root joint's parent;
        // `compute_skinning_transforms` handles this without panicking, so
        // it must not be rejected here either.
        assert!(validate_parents(&[0, 0, 1]).is_ok());
    }

    #[test]
    fn validate_parents_rejects_positive_root() {
        // A root parent that is neither the conventional negative sentinel
        // nor the tolerated self-reference is not a recognised root marker
        // (and, if `>= n_joints`, would index out of bounds downstream).
        let err = validate_parents(&[5, 0]).unwrap_err();
        assert!(matches!(err, FlameError::InvalidParams(_)));
    }

    #[test]
    fn validate_parents_rejects_forward_reference() {
        // Joint 1 points at joint 2, which has not been processed yet.
        let err = validate_parents(&[-1, 2, 0]).unwrap_err();
        assert!(matches!(err, FlameError::InvalidParams(_)));
    }

    #[test]
    fn validate_parents_rejects_self_reference() {
        let err = validate_parents(&[-1, 1]).unwrap_err();
        assert!(matches!(err, FlameError::InvalidParams(_)));
    }

    #[test]
    fn validate_parents_rejects_empty() {
        assert!(validate_parents(&[]).is_err());
    }

    #[test]
    fn convert_faces_accepts_in_range_indices() {
        let faces_i32 = Array2::from_shape_vec((2, 3), vec![0, 1, 2, 2, 1, 0])
            .expect("valid (2, 3) shape for a 6-element vec");
        let faces = convert_faces(&faces_i32, 3).expect("in-range faces should convert");
        assert_eq!(faces, vec![[0, 1, 2], [2, 1, 0]]);
    }

    #[test]
    fn convert_faces_rejects_out_of_range_index() {
        let faces_i32 = Array2::from_shape_vec((1, 3), vec![0, 1, 5])
            .expect("valid (1, 3) shape for a 3-element vec");
        let err = convert_faces(&faces_i32, 3).unwrap_err();
        assert!(matches!(err, FlameError::IndexOutOfBounds { len: 3, .. }));
    }

    #[test]
    fn convert_faces_rejects_negative_index() {
        let faces_i32 = Array2::from_shape_vec((1, 3), vec![0, 1, -1])
            .expect("valid (1, 3) shape for a 3-element vec");
        let err = convert_faces(&faces_i32, 3).unwrap_err();
        assert!(matches!(err, FlameError::IndexOutOfBounds { .. }));
    }

    #[test]
    fn expect_dir_shape_checks_leading_dims() {
        let dirs = Array3::<f32>::zeros((5, 3, 10));
        assert!(expect_dir_shape("shapedirs", dirs.shape(), 5, None).is_ok());
        assert!(expect_dir_shape("shapedirs", dirs.shape(), 6, None).is_err());
    }

    #[test]
    fn expect_dir_shape_checks_trailing_k_when_requested() {
        let dirs = Array3::<f32>::zeros((5, 3, 36));
        assert!(expect_dir_shape("posedirs", dirs.shape(), 5, Some(36)).is_ok());
        assert!(expect_dir_shape("posedirs", dirs.shape(), 5, Some(9)).is_err());
    }
}
