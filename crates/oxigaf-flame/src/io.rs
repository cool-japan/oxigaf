//! Load FLAME model data from a directory of `.npy` files.

use std::path::Path;

use ndarray::{Array2, Array3};
use ndarray_npy::read_npy;

use crate::error::FlameError;
use crate::model::FlameModel;

// ---------------------------------------------------------------------------
// The `.npy` file set read by `load_flame_model`
// ---------------------------------------------------------------------------
//
// Every name below is used *twice*: once as an element of
// [`REQUIRED_NPY_FILES`], and once as the argument of the matching `load_npy`
// call inside [`load_flame_model`]. Sharing one constant between the two is
// what makes it impossible for a caller-facing "which files do I need?" list
// (e.g. `oxigaf::verify_assets`) to drift away from the names the loader
// actually opens — a drift this crate has already shipped once.

/// File name of the rest-pose vertex array, shape `[n_verts, 3]`, `float32`.
const V_TEMPLATE_NPY: &str = "v_template.npy";
/// File name of the triangle index array, shape `[n_faces, 3]`, `int32`.
const FACES_NPY: &str = "faces.npy";
/// File name of the shape blend-shape basis, shape `[n_verts, 3, K]`, `float32`.
const SHAPEDIRS_NPY: &str = "shapedirs.npy";
/// File name of the expression blend-shape basis, shape `[n_verts, 3, K]`, `float32`.
const EXPRESSIONDIRS_NPY: &str = "expressiondirs.npy";
/// File name of the pose-corrective basis, shape `[n_verts, 3, (J-1)*9]`, `float32`.
const POSEDIRS_NPY: &str = "posedirs.npy";
/// File name of the joint regressor, shape `[n_joints, n_verts]`, `float32`.
const J_REGRESSOR_NPY: &str = "j_regressor.npy";
/// File name of the kinematic tree table, shape `[2, n_joints]`, `int32`.
const KINTREE_TABLE_NPY: &str = "kintree_table.npy";
/// File name of the linear-blend-skinning weights, shape `[n_verts, n_joints]`, `float32`.
const LBS_WEIGHTS_NPY: &str = "lbs_weights.npy";

/// The exact set of `.npy` file names [`load_flame_model`] opens, in the order
/// it opens them.
///
/// This is the single source of truth for "what does a FLAME model directory
/// have to contain?". Tools that pre-flight an asset directory — such as
/// `oxigaf::verify_assets` — must iterate this constant rather than repeating
/// the names, so that adding or renaming a loader input can never leave a
/// second, stale copy of the list behind.
///
/// Names are spelled exactly as they must appear on disk (`scripts/convert_flame.py`
/// writes them), so a directory that satisfies a check against this list also
/// loads on case-sensitive filesystems.
pub const REQUIRED_NPY_FILES: &[&str] = &[
    V_TEMPLATE_NPY,
    FACES_NPY,
    SHAPEDIRS_NPY,
    EXPRESSIONDIRS_NPY,
    POSEDIRS_NPY,
    J_REGRESSOR_NPY,
    KINTREE_TABLE_NPY,
    LBS_WEIGHTS_NPY,
];

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
/// The file names are also available programmatically as
/// [`REQUIRED_NPY_FILES`]; pre-flight checks should iterate that constant
/// instead of repeating the list.
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

    let v_template: Array2<f32> = load_npy(dir, V_TEMPLATE_NPY)?;
    let faces_i32: Array2<i32> = load_npy(dir, FACES_NPY)?;
    let shapedirs: Array3<f32> = load_npy(dir, SHAPEDIRS_NPY)?;
    let expressiondirs: Array3<f32> = load_npy(dir, EXPRESSIONDIRS_NPY)?;
    let posedirs: Array3<f32> = load_npy(dir, POSEDIRS_NPY)?;
    let j_regressor: Array2<f32> = load_npy(dir, J_REGRESSOR_NPY)?;
    let kintree_i32: Array2<i32> = load_npy(dir, KINTREE_TABLE_NPY)?;
    let lbs_weights: Array2<f32> = load_npy(dir, LBS_WEIGHTS_NPY)?;

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
///
/// `file_name` is the on-disk name including the `.npy` extension — always one
/// of the [`REQUIRED_NPY_FILES`] constants, so the loader and that list cannot
/// disagree about what is opened. The reported error carries the *stem*
/// (`"lbs_weights"` rather than `"lbs_weights.npy"`), which is the array name
/// callers match on.
fn load_npy<A, D>(dir: &Path, file_name: &str) -> Result<ndarray::Array<A, D>, FlameError>
where
    A: ndarray_npy::ReadableElement,
    D: ndarray::Dimension,
{
    let path = dir.join(file_name);
    let name = file_name.strip_suffix(".npy").unwrap_or(file_name);
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
        let j_i32 = i32::try_from(j).map_err(|_| {
            FlameError::InvalidParams(format!(
                "parents has too many entries ({j}) to validate as ancestor indices"
            ))
        })?;
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

    use ndarray_npy::write_npy;
    use std::path::PathBuf;

    /// Create a fresh, empty temporary directory dedicated to one test.
    fn temp_subdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("oxigaf_flame_io_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("test: temp dir creation should succeed");
        dir
    }

    /// Write a minimal but fully valid FLAME model directory, using exactly the
    /// [`REQUIRED_NPY_FILES`] names.
    ///
    /// 4 vertices, 2 joints, a 2-component shape/expression basis. The loader
    /// fixes `posedirs`'s trailing dimension at `(n_joints - 1) * 9`.
    fn write_minimal_npy_model(dir: &Path) {
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

        write_npy(dir.join(V_TEMPLATE_NPY), &v_template).expect("test: write v_template.npy");
        write_npy(dir.join(FACES_NPY), &faces).expect("test: write faces.npy");
        write_npy(dir.join(SHAPEDIRS_NPY), &shapedirs).expect("test: write shapedirs.npy");
        write_npy(dir.join(EXPRESSIONDIRS_NPY), &expressiondirs)
            .expect("test: write expressiondirs.npy");
        write_npy(dir.join(POSEDIRS_NPY), &posedirs).expect("test: write posedirs.npy");
        write_npy(dir.join(J_REGRESSOR_NPY), &j_regressor).expect("test: write j_regressor.npy");
        write_npy(dir.join(KINTREE_TABLE_NPY), &kintree_table)
            .expect("test: write kintree_table.npy");
        write_npy(dir.join(LBS_WEIGHTS_NPY), &lbs_weights).expect("test: write lbs_weights.npy");
    }

    // -----------------------------------------------------------------------
    // REQUIRED_NPY_FILES — the shared list consumed by `oxigaf::verify_assets`
    // -----------------------------------------------------------------------

    #[test]
    fn required_npy_files_is_a_well_formed_name_set() {
        assert!(
            !REQUIRED_NPY_FILES.is_empty(),
            "the loader reads at least one file"
        );
        for name in REQUIRED_NPY_FILES {
            // The names must be spelled exactly as they appear on disk —
            // lower-case extension included, since a check against this list
            // has to hold on case-sensitive filesystems.
            assert!(
                std::path::Path::new(name)
                    .extension()
                    .is_some_and(|ext| ext == "npy"),
                "{name} must be spelled as an on-disk .npy file name"
            );
            assert_eq!(
                REQUIRED_NPY_FILES.iter().filter(|n| *n == name).count(),
                1,
                "{name} is listed more than once"
            );
        }
    }

    #[test]
    fn a_directory_of_exactly_required_npy_files_loads() {
        // Upper bound on the list: the loader needs nothing that
        // `REQUIRED_NPY_FILES` omits, so a directory built from that constant
        // alone must load. This is what makes a pre-flight check against the
        // constant meaningful rather than merely necessary.
        let dir = temp_subdir("required_set_loads");
        write_minimal_npy_model(&dir);

        let entries: Vec<String> = std::fs::read_dir(&dir)
            .expect("test: temp dir should be readable")
            .filter_map(std::result::Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            entries.len(),
            REQUIRED_NPY_FILES.len(),
            "the fixture must write exactly the required set, got {entries:?}"
        );

        let model = load_flame_model(&dir).expect("test: a complete directory must load");
        assert_eq!(model.n_joints, 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn every_required_npy_file_is_actually_opened_by_the_loader() {
        // Lower bound on the list: removing any single entry must make the
        // load fail *naming that entry*. A stale name that the loader no
        // longer opens would survive its own removal and fail this test —
        // which is exactly the drift `oxigaf::verify_assets` used to ship.
        for (index, missing) in REQUIRED_NPY_FILES.iter().enumerate() {
            let dir = temp_subdir(&format!("drop_one_{index}"));
            write_minimal_npy_model(&dir);
            std::fs::remove_file(dir.join(missing)).expect("test: fixture file should exist");

            let stem = missing.strip_suffix(".npy").unwrap_or(missing);
            let err = load_flame_model(&dir)
                .err()
                .unwrap_or_else(|| panic!("test: removing {missing} must make the load fail"));
            assert!(
                matches!(&err, FlameError::NpyLoad { name, .. } if name == stem),
                "removing {missing} must be reported as a missing {stem} array, got: {err}"
            );

            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn load_npy_reports_the_array_stem_not_the_file_name() {
        // `conversion.rs` (and any caller matching on `FlameError::NpyLoad`)
        // relies on the reported `name` being the array stem, even though the
        // loader now passes the full file name in.
        let dir = temp_subdir("npy_error_stem");
        let err = load_npy::<f32, ndarray::Ix2>(&dir, LBS_WEIGHTS_NPY)
            .expect_err("test: an absent file must fail to load");
        assert!(
            matches!(&err, FlameError::NpyLoad { name, .. } if name == "lbs_weights"),
            "got: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

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
