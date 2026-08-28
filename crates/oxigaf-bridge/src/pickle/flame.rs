//! Pure-Rust ingest of FLAME `.pkl` head models.
//!
//! This is the Rust port of `scripts/convert_flame.py`: it reads the
//! original FLAME `.pkl` (a Python 2 protocol-2 pickle of chumpy arrays and
//! SciPy sparse matrices) and writes the directory of `.npy` files
//! `oxigaf_flame::io::load_flame_model` expects.
//!
//! The Python script needed `numpy`, `scipy` and `pickle`; nothing here
//! needs a Python installation, and nothing here executes any part of the
//! pickle -- see [`super::vm`].

use super::error::{PickleError, Result};
use super::numpy::{as_array, as_array_or_sparse, NumpyArray};
use super::value::Value;
use super::vm;
use std::path::Path;

/// Number of FLAME shape (identity) blend-shape directions.
///
/// FLAME packs identity and expression directions into one `shapedirs`
/// array of `[V, 3, n_shape + n_expr]`; the first 300 columns are identity.
/// Matches the split `scripts/convert_flame.py` performed.
const FLAME_SHAPE_DIRS: usize = 300;

/// A FLAME model recovered from a `.pkl`, in the exact layout
/// `oxigaf-flame`'s loader expects.
#[derive(Debug, Clone)]
pub struct FlameModelData {
    /// Template vertices, `[V, 3]`.
    pub v_template: NamedArray,
    /// Triangle indices, `[F, 3]`, `i32`.
    pub faces: NamedArray,
    /// Identity blend-shape directions, `[V, 3, n_shape]`.
    pub shapedirs: NamedArray,
    /// Expression blend-shape directions, `[V, 3, n_expr]`.
    pub expressiondirs: NamedArray,
    /// Pose blend-shape directions, `[V, 3, 9*(J-1)]`.
    pub posedirs: NamedArray,
    /// Joint regressor, `[J, V]`, densified from SciPy sparse if needed.
    pub j_regressor: NamedArray,
    /// Kinematic tree, `[2, J]`, `i32`.
    pub kintree_table: NamedArray,
    /// Linear blend-skinning weights, `[V, J]`.
    pub lbs_weights: NamedArray,
}

/// One output array, with the base file name `oxigaf-flame` looks for.
#[derive(Debug, Clone)]
pub struct NamedArray {
    /// File stem, e.g. `v_template` (written as `v_template.npy`).
    pub name: &'static str,
    /// Dimensions, outermost first.
    pub shape: Vec<usize>,
    /// Elements, either f32 or i32 depending on the array.
    pub values: ArrayValues,
}

/// The element payload of a [`NamedArray`].
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayValues {
    /// 32-bit float elements, row-major.
    F32(Vec<f32>),
    /// 32-bit signed integer elements, row-major.
    I32(Vec<i32>),
}

impl FlameModelData {
    /// Every array in the model, in a stable order.
    pub fn arrays(&self) -> [&NamedArray; 8] {
        [
            &self.v_template,
            &self.faces,
            &self.shapedirs,
            &self.expressiondirs,
            &self.posedirs,
            &self.j_regressor,
            &self.kintree_table,
            &self.lbs_weights,
        ]
    }
}

/// Reads a FLAME `.pkl` model.
///
/// # Errors
///
/// Returns [`PickleError`] if the file cannot be read, if its pickle is
/// malformed, or if a required FLAME key is missing or has an unexpected
/// structure. Every error names the key at fault.
pub fn read_flame_model(path: &Path) -> Result<FlameModelData> {
    let bytes = std::fs::read(path)?;
    read_flame_model_bytes(&bytes)
}

/// Reads a FLAME model from an in-memory `.pkl`.
///
/// # Errors
///
/// As [`read_flame_model`].
pub fn read_flame_model_bytes(bytes: &[u8]) -> Result<FlameModelData> {
    let root = vm::load(bytes)?;
    if root.as_mapping().is_none() {
        return Err(PickleError::Structure(format!(
            "expected the FLAME model's top level to be a dict, found {root}"
        )));
    }

    let v_template_array = float_array(&root, "v_template")?;
    let n_verts = *v_template_array
        .shape
        .first()
        .ok_or_else(|| PickleError::Structure("v_template has no dimensions".to_string()))?;

    let faces_raw = int_array(&root, "f")?;
    let kintree_raw = int_array(&root, "kintree_table")?;
    let n_joints = *kintree_raw.shape.get(1).ok_or_else(|| {
        PickleError::Structure(format!(
            "kintree_table has shape {:?}, expected [2, n_joints]",
            kintree_raw.shape
        ))
    })?;

    // FLAME packs identity + expression directions into one array; split
    // them the same way `scripts/convert_flame.py` did.
    let shapedirs_full = raw_array(&root, "shapedirs")?;
    let total_dirs = *shapedirs_full.shape.get(2).ok_or_else(|| {
        PickleError::Structure(format!(
            "shapedirs has shape {:?}, expected [V, 3, n_dirs]",
            shapedirs_full.shape
        ))
    })?;
    let n_shape = FLAME_SHAPE_DIRS.min(total_dirs);
    let n_expr = total_dirs - n_shape;

    let shapedirs = slice_last_axis(&shapedirs_full, 0, n_shape, "shapedirs")?;
    let expressiondirs = if n_expr > 0 {
        slice_last_axis(&shapedirs_full, n_shape, total_dirs, "shapedirs")?
    } else {
        // Some FLAME releases store expression directions in a separate
        // key; fall back to that before emitting an empty array, matching
        // the Python script's documented behaviour but doing better where
        // the key exists.
        match root.get("expressiondirs").or_else(|| root.get("exprdirs")) {
            Some(value) => to_f32_named(&as_array_or_sparse(value)?, "expressiondirs")?,
            None => NamedArray {
                name: "expressiondirs",
                shape: vec![n_verts, 3, 0],
                values: ArrayValues::F32(Vec::new()),
            },
        }
    };

    let posedirs = float_array(&root, "posedirs")?;
    let j_regressor = float_array(&root, "J_regressor")?;
    let lbs_weights = float_array(&root, "weights")?;

    let model = FlameModelData {
        v_template: rename(v_template_array, "v_template"),
        faces: rename_int(faces_raw, "faces"),
        shapedirs: rename(shapedirs, "shapedirs"),
        expressiondirs: rename(expressiondirs, "expressiondirs"),
        posedirs: rename(posedirs, "posedirs"),
        j_regressor: rename(j_regressor, "j_regressor"),
        kintree_table: rename_int(kintree_raw, "kintree_table"),
        lbs_weights: rename(lbs_weights, "lbs_weights"),
    };

    validate(&model, n_verts, n_joints)?;
    Ok(model)
}

/// Checks the cross-array invariants `oxigaf_flame::io::load_flame_model`
/// enforces, so a bad conversion is caught here rather than surfacing much
/// later as a confusing shape error inside the renderer.
fn validate(model: &FlameModelData, n_verts: usize, n_joints: usize) -> Result<()> {
    let expect = |array: &NamedArray, expected: &[usize]| -> Result<()> {
        if array.shape != expected {
            return Err(PickleError::Structure(format!(
                "{} has shape {:?}, expected {:?}",
                array.name, array.shape, expected
            )));
        }
        Ok(())
    };

    expect(&model.v_template, &[n_verts, 3])?;
    expect(&model.j_regressor, &[n_joints, n_verts])?;
    expect(&model.lbs_weights, &[n_verts, n_joints])?;
    expect(&model.kintree_table, &[2, n_joints])?;

    for dirs in [&model.shapedirs, &model.expressiondirs, &model.posedirs] {
        let [verts, components, _] = dirs.shape[..] else {
            return Err(PickleError::Structure(format!(
                "{} has shape {:?}, expected [V, 3, n]",
                dirs.name, dirs.shape
            )));
        };
        if verts != n_verts || components != 3 {
            return Err(PickleError::Structure(format!(
                "{} has shape {:?}, expected [{}, 3, n]",
                dirs.name, dirs.shape, n_verts
            )));
        }
    }

    let ArrayValues::I32(faces) = &model.faces.values else {
        return Err(PickleError::Structure(
            "faces must be integer indices".to_string(),
        ));
    };
    if model.faces.shape.get(1) != Some(&3) {
        return Err(PickleError::Structure(format!(
            "faces has shape {:?}, expected [F, 3]",
            model.faces.shape
        )));
    }
    if let Some(&bad) = faces
        .iter()
        .find(|&&index| index < 0 || index as usize >= n_verts)
    {
        return Err(PickleError::Structure(format!(
            "faces references vertex {bad}, outside the {n_verts}-vertex template"
        )));
    }

    Ok(())
}

fn rename(mut array: NamedArray, name: &'static str) -> NamedArray {
    array.name = name;
    array
}

fn rename_int(mut array: NamedArray, name: &'static str) -> NamedArray {
    array.name = name;
    array
}

/// Reads a required key as a raw NumPy array, densifying SciPy sparse
/// storage and unwrapping chumpy nodes.
fn raw_array(root: &Value, key: &str) -> Result<NumpyArray> {
    let value = root
        .get(key)
        .ok_or_else(|| PickleError::Structure(format!("FLAME model has no '{key}' key")))?;
    as_array_or_sparse(value).map_err(|e| PickleError::Structure(format!("'{key}': {e}")))
}

/// Reads a required key as a float array, densifying SciPy sparse storage.
fn float_array(root: &Value, key: &str) -> Result<NamedArray> {
    to_f32_named(&raw_array(root, key)?, "")
}

/// Reads a required key as an integer array.
fn int_array(root: &Value, key: &str) -> Result<NamedArray> {
    let value = root
        .get(key)
        .ok_or_else(|| PickleError::Structure(format!("FLAME model has no '{key}' key")))?;
    let array = as_array(value).map_err(|e| PickleError::Structure(format!("'{key}': {e}")))?;
    Ok(NamedArray {
        name: "",
        shape: array.shape.clone(),
        values: ArrayValues::I32(array.to_i32()?),
    })
}

fn to_f32_named(array: &NumpyArray, name: &'static str) -> Result<NamedArray> {
    Ok(NamedArray {
        name,
        shape: array.shape.clone(),
        values: ArrayValues::F32(array.to_f32()?),
    })
}

/// Takes `[start, end)` along the last axis of a `[a, b, n]` array,
/// preserving row-major order.
fn slice_last_axis(array: &NumpyArray, start: usize, end: usize, key: &str) -> Result<NamedArray> {
    let [outer, middle, depth] = array.shape[..] else {
        return Err(PickleError::Structure(format!(
            "'{key}' has shape {:?}, expected three dimensions",
            array.shape
        )));
    };
    if end > depth || start > end {
        return Err(PickleError::Structure(format!(
            "'{key}': slice {start}..{end} is outside its last axis of {depth}"
        )));
    }

    let all = array.to_f32()?;
    let width = end - start;
    let mut out = Vec::with_capacity(outer * middle * width);
    for row in 0..outer * middle {
        let base = row * depth;
        out.extend_from_slice(&all[base + start..base + end]);
    }

    Ok(NamedArray {
        name: "",
        shape: vec![outer, middle, width],
        values: ArrayValues::F32(out),
    })
}

/// Writes a [`FlameModelData`] as the directory of `.npy` files
/// `oxigaf_flame::io::load_flame_model` reads.
///
/// # Errors
///
/// Returns an error if the directory cannot be created or a file cannot be
/// written.
pub fn write_npy_dir(model: &FlameModelData, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for array in model.arrays() {
        let path = dir.join(format!("{}.npy", array.name));
        std::fs::write(&path, encode_npy(array))?;
        tracing::debug!("Wrote {} {:?}", path.display(), array.shape);
    }
    Ok(())
}

/// Encodes one array in NPY format v1.0.
///
/// The format is a short ASCII header describing dtype, order and shape,
/// padded so the data begins on a 64-byte boundary, followed by raw
/// little-endian elements. Emitting it directly (rather than via
/// `ndarray-npy`) keeps this crate's dependency set unchanged and is a
/// dozen lines.
fn encode_npy(array: &NamedArray) -> Vec<u8> {
    let (descr, data): (&str, Vec<u8>) = match &array.values {
        ArrayValues::F32(values) => ("<f4", values.iter().flat_map(|v| v.to_le_bytes()).collect()),
        ArrayValues::I32(values) => ("<i4", values.iter().flat_map(|v| v.to_le_bytes()).collect()),
    };

    // NumPy renders a 1-tuple as `(n,)`; a 0-d array as `()`.
    let shape = match array.shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", array.shape[0]),
        _ => format!(
            "({})",
            array
                .shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");

    // Magic (6) + version (2) + header length (2) + header + '\n' must be a
    // multiple of 64.
    const PREFIX: usize = 10;
    while !(PREFIX + header.len() + 1).is_multiple_of(64) {
        header.push(' ');
    }
    header.push('\n');

    let mut out = Vec::with_capacity(PREFIX + header.len() + data.len());
    out.extend_from_slice(b"\x93NUMPY");
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pickle::test_support::{pickle, PickleBuilder};

    fn push_ndarray(p: &mut PickleBuilder, shape: &[usize], dtype_code: &str, raw: &[u8]) {
        p.global("numpy.core.multiarray", "_reconstruct");
        p.mark();
        p.global("numpy", "ndarray");
        p.int_tuple(&[0]);
        p.py2_str(b"b");
        p.tuple();
        p.reduce();
        p.mark();
        p.int(1);
        p.int_tuple(shape);
        p.unicode(dtype_code);
        p.bool(false);
        p.py2_str(raw);
        p.tuple();
        p.build_state();
    }

    fn f32_raw(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn i32_raw(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A tiny but structurally complete FLAME model: 4 vertices, 2 faces,
    /// 2 joints, 3 shape dirs (so the 300-column split leaves none for
    /// expression), 1 pose dir set.
    fn tiny_flame_pickle(shape_dirs: usize, expr_dirs: usize) -> Vec<u8> {
        let n_verts = 4usize;
        let n_joints = 2usize;
        let total_dirs = shape_dirs + expr_dirs;

        pickle(|p| {
            p.empty_dict();

            p.unicode("v_template");
            push_ndarray(
                p,
                &[n_verts, 3],
                "f4",
                &f32_raw(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            );
            p.setitem();

            p.unicode("f");
            push_ndarray(p, &[2, 3], "i4", &i32_raw(&[0, 1, 2, 0, 2, 3]));
            p.setitem();

            p.unicode("shapedirs");
            let dirs: Vec<f32> = (0..n_verts * 3 * total_dirs).map(|i| i as f32).collect();
            push_ndarray(p, &[n_verts, 3, total_dirs], "f4", &f32_raw(&dirs));
            p.setitem();

            p.unicode("posedirs");
            let pose_cols = 9 * (n_joints - 1);
            let pose: Vec<f32> = vec![0.25; n_verts * 3 * pose_cols];
            push_ndarray(p, &[n_verts, 3, pose_cols], "f4", &f32_raw(&pose));
            p.setitem();

            p.unicode("J_regressor");
            let reg: Vec<f32> = vec![0.5; n_joints * n_verts];
            push_ndarray(p, &[n_joints, n_verts], "f4", &f32_raw(&reg));
            p.setitem();

            p.unicode("kintree_table");
            push_ndarray(p, &[2, n_joints], "i4", &i32_raw(&[-1, 0, 0, 1]));
            p.setitem();

            p.unicode("weights");
            let w: Vec<f32> = vec![0.5; n_verts * n_joints];
            push_ndarray(p, &[n_verts, n_joints], "f4", &f32_raw(&w));
            p.setitem();
        })
    }

    #[test]
    fn test_reads_a_complete_model() {
        let bytes = tiny_flame_pickle(3, 0);
        let model = read_flame_model_bytes(&bytes).expect("test: model should read");

        assert_eq!(model.v_template.shape, vec![4, 3]);
        assert_eq!(model.faces.shape, vec![2, 3]);
        assert_eq!(model.j_regressor.shape, vec![2, 4]);
        assert_eq!(model.lbs_weights.shape, vec![4, 2]);
        assert_eq!(model.kintree_table.shape, vec![2, 2]);
        // Fewer than 300 total dirs: all of them are identity, none are
        // expression -- the same split `convert_flame.py` performed.
        assert_eq!(model.shapedirs.shape, vec![4, 3, 3]);
        assert_eq!(model.expressiondirs.shape, vec![4, 3, 0]);
    }

    #[test]
    fn test_splits_identity_and_expression_directions() {
        // 302 packed directions -> 300 identity + 2 expression, and the
        // split must preserve row-major layout rather than interleaving.
        let bytes = tiny_flame_pickle(300, 2);
        let model = read_flame_model_bytes(&bytes).expect("test: model should read");
        assert_eq!(model.shapedirs.shape, vec![4, 3, 300]);
        assert_eq!(model.expressiondirs.shape, vec![4, 3, 2]);

        let ArrayValues::F32(shape_values) = &model.shapedirs.values else {
            panic!("shapedirs must be f32");
        };
        let ArrayValues::F32(expr_values) = &model.expressiondirs.values else {
            panic!("expressiondirs must be f32");
        };
        // Source element (v, c, d) is at index (v*3 + c)*302 + d.
        assert_eq!(shape_values[0], 0.0);
        assert_eq!(shape_values[299], 299.0);
        assert_eq!(expr_values[0], 300.0);
        assert_eq!(expr_values[1], 301.0);
        // Second row starts at source index 302.
        assert_eq!(shape_values[300], 302.0);
        assert_eq!(expr_values[2], 602.0);
    }

    #[test]
    fn test_missing_key_names_itself() {
        let bytes = pickle(|p| {
            p.empty_dict();
            p.unicode("v_template");
            push_ndarray(p, &[1, 3], "f4", &f32_raw(&[0.0, 0.0, 0.0]));
            p.setitem();
        });
        let err = read_flame_model_bytes(&bytes).expect_err("missing keys must error");
        assert!(err.to_string().contains("'f'"), "got: {err}");
    }

    #[test]
    fn test_out_of_range_face_index_is_rejected() {
        // A face referencing vertex 99 in a 4-vertex model would panic
        // later inside the renderer's normal computation.
        let bytes = pickle(|p| {
            p.empty_dict();
            p.unicode("v_template");
            push_ndarray(p, &[4, 3], "f4", &f32_raw(&[0.0; 12]));
            p.setitem();
            p.unicode("f");
            push_ndarray(p, &[1, 3], "i4", &i32_raw(&[0, 1, 99]));
            p.setitem();
            p.unicode("shapedirs");
            push_ndarray(p, &[4, 3, 1], "f4", &f32_raw(&[0.0; 12]));
            p.setitem();
            p.unicode("posedirs");
            push_ndarray(p, &[4, 3, 9], "f4", &f32_raw(&[0.0; 108]));
            p.setitem();
            p.unicode("J_regressor");
            push_ndarray(p, &[2, 4], "f4", &f32_raw(&[0.0; 8]));
            p.setitem();
            p.unicode("kintree_table");
            push_ndarray(p, &[2, 2], "i4", &i32_raw(&[-1, 0, 0, 1]));
            p.setitem();
            p.unicode("weights");
            push_ndarray(p, &[4, 2], "f4", &f32_raw(&[0.0; 8]));
            p.setitem();
        });
        let err = read_flame_model_bytes(&bytes).expect_err("bad face index must error");
        assert!(err.to_string().contains("99"), "got: {err}");
    }

    #[test]
    fn test_npy_encoding_is_readable_by_ndarray_npy() {
        // The whole point of writing .npy is that `oxigaf-flame`'s loader
        // (which uses `ndarray-npy`) can read it back, so assert exactly
        // that rather than just eyeballing the header bytes.
        let array = NamedArray {
            name: "t",
            shape: vec![2, 3],
            values: ArrayValues::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        };
        let encoded = encode_npy(&array);

        assert_eq!(&encoded[..6], b"\x93NUMPY");
        // Header + prefix must be 64-byte aligned so the data is aligned.
        let header_len = u16::from_le_bytes([encoded[8], encoded[9]]) as usize;
        assert!(
            (10 + header_len).is_multiple_of(64),
            "npy data must start on a 64-byte boundary"
        );

        use ndarray_npy::ReadNpyExt;
        let read = ndarray::Array2::<f32>::read_npy(std::io::Cursor::new(&encoded))
            .expect("test: ndarray-npy should read our output");
        assert_eq!(read.shape(), &[2, 3]);
        assert_eq!(read[[1, 2]], 6.0);
    }

    #[test]
    fn test_npy_encoding_round_trips_a_1d_int_array() {
        let array = NamedArray {
            name: "t",
            shape: vec![4],
            values: ArrayValues::I32(vec![-1, 0, 1, 2]),
        };
        let encoded = encode_npy(&array);
        use ndarray_npy::ReadNpyExt;
        let read = ndarray::Array1::<i32>::read_npy(std::io::Cursor::new(&encoded))
            .expect("test: ndarray-npy should read our output");
        assert_eq!(read.to_vec(), vec![-1, 0, 1, 2]);
    }

    #[test]
    fn test_write_npy_dir_produces_every_expected_file() {
        let bytes = tiny_flame_pickle(3, 0);
        let model = read_flame_model_bytes(&bytes).expect("test: model should read");

        let dir = tempfile::tempdir().expect("test: temp dir");
        write_npy_dir(&model, dir.path()).expect("test: write should succeed");

        for name in [
            "v_template",
            "faces",
            "shapedirs",
            "expressiondirs",
            "posedirs",
            "j_regressor",
            "kintree_table",
            "lbs_weights",
        ] {
            let path = dir.path().join(format!("{name}.npy"));
            assert!(path.exists(), "{name}.npy was not written");
        }
    }

    #[test]
    fn test_shape_mismatch_between_arrays_is_rejected() {
        // A j_regressor sized for three joints while kintree_table declares
        // two would explode much later inside skinning; the converter must
        // catch it here.
        let bytes = pickle(|p| {
            p.empty_dict();
            p.unicode("v_template");
            push_ndarray(p, &[4, 3], "f4", &f32_raw(&[0.0; 12]));
            p.setitem();
            p.unicode("f");
            push_ndarray(p, &[1, 3], "i4", &i32_raw(&[0, 1, 2]));
            p.setitem();
            p.unicode("shapedirs");
            push_ndarray(p, &[4, 3, 1], "f4", &f32_raw(&[0.0; 12]));
            p.setitem();
            p.unicode("posedirs");
            push_ndarray(p, &[4, 3, 9], "f4", &f32_raw(&[0.0; 108]));
            p.setitem();
            p.unicode("J_regressor");
            push_ndarray(p, &[3, 4], "f4", &f32_raw(&[0.0; 12]));
            p.setitem();
            p.unicode("kintree_table");
            push_ndarray(p, &[2, 2], "i4", &i32_raw(&[-1, 0, 0, 1]));
            p.setitem();
            p.unicode("weights");
            push_ndarray(p, &[4, 2], "f4", &f32_raw(&[0.0; 8]));
            p.setitem();
        });
        let err = read_flame_model_bytes(&bytes).expect_err("shape mismatch must error");
        assert!(err.to_string().contains("j_regressor"), "got: {err}");
    }
}
