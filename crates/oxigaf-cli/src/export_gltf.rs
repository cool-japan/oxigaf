//! glTF 2.0 export for 3D Gaussian Splatting models.
//!
//! Outputs a `.gltf` (JSON) + `.bin` (binary buffer) file pair, with a custom
//! `OXIGAF_gaussian_splat` extension storing per-Gaussian attributes.
//!
//! # Binary Buffer Layout
//!
//! The `.bin` file contains tightly packed f32 arrays in this order:
//!
//! | Accessor    | Components | Bytes per Gaussian | Cumulative offset |
//! |-------------|------------|---------------------|-------------------|
//! | positions   | N×3        | 12                  | 0                 |
//! | rotations   | N×4        | 16                  | N×12              |
//! | scales      | N×3        | 12                  | N×28              |
//! | opacities   | N×1        |  4                  | N×40              |
//! | sh_coeffs   | N×C        |  4×C                | N×44              |
//!
//! Where `C = (sh_degree + 1)² × 3`.

use std::io::{BufWriter, Write};
use std::path::Path;

use serde_json::{json, Value};

use oxigaf::render::gaussian::GaussianModel;

use crate::error::CliError;

// ---------------------------------------------------------------------------
// glTF component type constants
// ---------------------------------------------------------------------------

/// glTF component type for 32-bit float.
const COMPONENT_TYPE_FLOAT: u32 = 5126;

// ---------------------------------------------------------------------------
// Binary buffer construction
// ---------------------------------------------------------------------------

/// Append an `f32` slice into a byte buffer, returning the byte offset at
/// which the slice was appended and its byte length.
fn append_f32_slice(buf: &mut Vec<u8>, data: &[f32]) -> (usize, usize) {
    let byte_offset = buf.len();
    let byte_length = std::mem::size_of_val(data);
    // SAFETY: f32 is Plain Old Data, converting via byte view is sound.
    buf.extend_from_slice(bytemuck::cast_slice(data));
    (byte_offset, byte_length)
}

// ---------------------------------------------------------------------------
// Public export function
// ---------------------------------------------------------------------------

/// Export a [`GaussianModel`] to a glTF 2.0 `.gltf` + `.bin` file pair.
///
/// Given `output_path` (with any extension or none), the function creates:
/// - `<stem>.gltf` — JSON glTF document
/// - `<stem>.bin`  — Binary buffer
///
/// The binary layout follows fixed per-Gaussian strides; see the module-level
/// documentation for the exact format.
pub fn export_gltf(model: &GaussianModel, output_path: &Path) -> Result<(), CliError> {
    // Derive stem and output file paths.
    let stem = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let parent = output_path.parent().unwrap_or_else(|| Path::new("."));

    let gltf_path = parent.join(format!("{stem}.gltf"));
    let bin_path = parent.join(format!("{stem}.bin"));
    let bin_filename = format!("{stem}.bin");

    // Create parent directory if needed.
    if !parent.as_os_str().is_empty() {
        std::fs::create_dir_all(parent).map_err(|e| {
            CliError::GltfExport(format!(
                "Failed to create output directory '{}': {e}",
                parent.display()
            ))
        })?;
    }

    let n = model.len();
    let sh_channels = ((model.sh_degree + 1).pow(2) * 3) as usize;

    // ----- Build binary buffer -----
    let mut bin_data: Vec<u8> = Vec::new();

    // Positions: N×3 f32
    let positions: Vec<f32> = model
        .gaussians
        .iter()
        .flat_map(|g| g.position.iter().copied())
        .collect();
    let (pos_byte_offset, _pos_byte_length) = append_f32_slice(&mut bin_data, &positions);

    // Rotations: N×4 f32 (quaternion xyzw)
    let rotations: Vec<f32> = model
        .gaussians
        .iter()
        .flat_map(|g| g.rotation.iter().copied())
        .collect();
    let (rot_byte_offset, _rot_byte_length) = append_f32_slice(&mut bin_data, &rotations);

    // Scales: N×3 f32
    let scales: Vec<f32> = model
        .gaussians
        .iter()
        .flat_map(|g| g.scale.iter().copied())
        .collect();
    let (scale_byte_offset, _scale_byte_length) = append_f32_slice(&mut bin_data, &scales);

    // Opacities: N×1 f32
    let opacities: Vec<f32> = model.gaussians.iter().map(|g| g.opacity).collect();
    let (opacity_byte_offset, _opacity_byte_length) = append_f32_slice(&mut bin_data, &opacities);

    // SH coefficients: N×C f32
    let mut sh_data = model.sh_coeffs.clone();
    sh_data.resize(n * sh_channels, 0.0_f32);
    let (sh_byte_offset, _sh_byte_length) = append_f32_slice(&mut bin_data, &sh_data);

    let total_bin_length = bin_data.len();

    // ----- Write binary file -----
    let bin_file = std::fs::File::create(&bin_path).map_err(|e| {
        CliError::GltfExport(format!(
            "Failed to create binary buffer '{}': {e}",
            bin_path.display()
        ))
    })?;
    let mut bin_writer = BufWriter::new(bin_file);
    bin_writer.write_all(&bin_data).map_err(|e| {
        CliError::GltfExport(format!(
            "Failed to write binary buffer '{}': {e}",
            bin_path.display()
        ))
    })?;
    bin_writer.flush().map_err(|e| {
        CliError::GltfExport(format!(
            "Failed to flush binary buffer '{}': {e}",
            bin_path.display()
        ))
    })?;

    // ----- Build glTF JSON -----
    // Top-level extension metadata (always present — it is custom OXIGAF
    // data, not a standard glTF construct subject to the accessor/
    // bufferView minimum-size rules below).
    let top_ext = json!({
        "gaussianCount": n,
        "shDegree": model.sh_degree
    });

    // For `n == 0` there is nothing to put in a mesh: the glTF 2.0 schema
    // requires `accessor.count >= 1` and `bufferView.byteLength >= 1`, so a
    // zero-count accessor / zero-length bufferView (what this used to
    // emit) is not a valid document, and every conforming loader rejects
    // it. An empty scene with no nodes/meshes/accessors/bufferViews/
    // buffers, by contrast, *is* valid — so that is what an empty model
    // gets instead.
    let (nodes, meshes, accessors, buffer_views, buffers, root_nodes) = if n == 0 {
        (
            Vec::<Value>::new(),
            Vec::<Value>::new(),
            Vec::<Value>::new(),
            Vec::<Value>::new(),
            Vec::<Value>::new(),
            Vec::<usize>::new(),
        )
    } else {
        // Single bufferView covering the entire binary buffer.
        let buffer_view = json!({
            "buffer": 0,
            "byteOffset": 0,
            "byteLength": total_bin_length
        });

        // Componentwise bounding box for the POSITION accessor — the glTF
        // 2.0 schema REQUIRES `min`/`max` on any accessor referenced by a
        // POSITION attribute (used by conforming loaders for frustum
        // culling / auto-framing); every other accessor here is optional.
        let mut pos_min = [f32::INFINITY; 3];
        let mut pos_max = [f32::NEG_INFINITY; 3];
        for chunk in positions.chunks_exact(3) {
            for k in 0..3 {
                pos_min[k] = pos_min[k].min(chunk[k]);
                pos_max[k] = pos_max[k].max(chunk[k]);
            }
        }

        // Accessors — each references the single bufferView with its own byteOffset.
        let accessor_positions = build_accessor(
            0,
            pos_byte_offset,
            n,
            "VEC3",
            COMPONENT_TYPE_FLOAT,
            Some((&pos_min, &pos_max)),
        );
        let accessor_rotations =
            build_accessor(0, rot_byte_offset, n, "VEC4", COMPONENT_TYPE_FLOAT, None);
        let accessor_scales =
            build_accessor(0, scale_byte_offset, n, "VEC3", COMPONENT_TYPE_FLOAT, None);
        let accessor_opacities = build_accessor(
            0,
            opacity_byte_offset,
            n,
            "SCALAR",
            COMPONENT_TYPE_FLOAT,
            None,
        );
        // sh_channels is always >= 3 (it is `(sh_degree+1)^2 * 3` with
        // `sh_degree: u32`), so `n * sh_channels >= n >= 1` here.
        let accessor_sh = build_accessor(
            0,
            sh_byte_offset,
            n * sh_channels,
            "SCALAR",
            COMPONENT_TYPE_FLOAT,
            None,
        );

        // Accessor indices.
        let acc_pos_idx = 0usize;
        let acc_rot_idx = 1usize;
        let acc_scale_idx = 2usize;
        let acc_opacity_idx = 3usize;
        let acc_sh_idx = 4usize;

        // Custom Gaussian extension on the node.
        let gaussian_extension = json!({
            "gaussianCount": n,
            "shDegree": model.sh_degree,
            "positionsAccessor": acc_pos_idx,
            "rotationsAccessor": acc_rot_idx,
            "scalesAccessor": acc_scale_idx,
            "opacitiesAccessor": acc_opacity_idx,
            "shCoefficientsAccessor": acc_sh_idx
        });

        // glTF mesh primitive uses POSITION so viewers can show a point cloud.
        let primitive = json!({
            "attributes": {
                "POSITION": acc_pos_idx
            },
            "mode": 0  // POINTS
        });

        let node = json!({
            "name": "GaussianSplat",
            "mesh": 0,
            "extensions": {
                "OXIGAF_gaussian_splat": gaussian_extension
            }
        });
        let mesh = json!({
            "name": "GaussianMesh",
            "primitives": [primitive]
        });
        let buffer = json!({
            "uri": bin_filename,
            "byteLength": total_bin_length
        });

        (
            vec![node],
            vec![mesh],
            vec![
                accessor_positions,
                accessor_rotations,
                accessor_scales,
                accessor_opacities,
                accessor_sh,
            ],
            vec![buffer_view],
            vec![buffer],
            vec![0usize],
        )
    };

    let document = json!({
        "asset": {
            "version": "2.0",
            "generator": format!("OxiGAF v{}", env!("CARGO_PKG_VERSION"))
        },
        "extensionsUsed": ["OXIGAF_gaussian_splat"],
        "scene": 0,
        "scenes": [{ "name": "GaussianScene", "nodes": root_nodes }],
        "nodes": nodes,
        "meshes": meshes,
        "buffers": buffers,
        "bufferViews": buffer_views,
        "accessors": accessors,
        "extensions": {
            "OXIGAF_gaussian_splat": top_ext
        }
    });

    // ----- Write .gltf JSON file -----
    let gltf_json = serde_json::to_string_pretty(&document)
        .map_err(|e| CliError::GltfExport(format!("Failed to serialize glTF JSON: {e}")))?;

    let gltf_file = std::fs::File::create(&gltf_path).map_err(|e| {
        CliError::GltfExport(format!(
            "Failed to create glTF file '{}': {e}",
            gltf_path.display()
        ))
    })?;
    let mut gltf_writer = BufWriter::new(gltf_file);
    gltf_writer.write_all(gltf_json.as_bytes()).map_err(|e| {
        CliError::GltfExport(format!(
            "Failed to write glTF file '{}': {e}",
            gltf_path.display()
        ))
    })?;
    gltf_writer.flush().map_err(|e| {
        CliError::GltfExport(format!(
            "Failed to flush glTF file '{}': {e}",
            gltf_path.display()
        ))
    })?;

    tracing::info!(
        "Wrote glTF 2.0: {} Gaussians (SH degree {}) → {} + {}",
        n,
        model.sh_degree,
        gltf_path.display(),
        bin_path.display(),
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Accessor builder
// ---------------------------------------------------------------------------

/// Build a glTF accessor JSON object referencing a single bufferView.
///
/// `buffer_view_idx` is the index into the `bufferViews` array.
/// `byte_offset` is the byte offset within that bufferView.
/// `min_max`, when given, populates the accessor's `min`/`max` fields —
/// REQUIRED by the glTF 2.0 schema for any accessor used as a POSITION
/// attribute, optional (and omitted here) for every other accessor.
fn build_accessor(
    buffer_view_idx: usize,
    byte_offset: usize,
    count: usize,
    accessor_type: &str,
    component_type: u32,
    min_max: Option<(&[f32; 3], &[f32; 3])>,
) -> Value {
    let mut accessor = json!({
        "bufferView": buffer_view_idx,
        "byteOffset": byte_offset,
        "componentType": component_type,
        "count": count,
        "type": accessor_type
    });
    if let Some((min, max)) = min_max {
        accessor["min"] = json!(min);
        accessor["max"] = json!(max);
    }
    accessor
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use oxigaf::render::gaussian::{GaussianAttributes, GaussianModel};

    /// Create a small GaussianModel with `n` Gaussians and given `sh_degree`.
    fn make_model(n: usize, sh_degree: u32) -> GaussianModel {
        let sh_channels = ((sh_degree + 1).pow(2) * 3) as usize;
        let gaussians: Vec<GaussianAttributes> = (0..n)
            .map(|i| {
                let f = i as f32;
                GaussianAttributes {
                    position: [f, f + 0.1, f + 0.2],
                    _pad0: 0.0,
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [0.01, 0.01, 0.01],
                    opacity: -1.0,
                }
            })
            .collect();
        let sh_coeffs = vec![0.5_f32; n * sh_channels];
        GaussianModel {
            gaussians,
            sh_coeffs,
            sh_degree,
            face_indices: vec![0u32; n],
            barycentric: vec![[1.0_f32 / 3.0; 3]; n],
            local_offsets: vec![[0.0_f32; 3]; n],
            is_rigid: vec![true; n],
        }
    }

    /// Return a unique temp directory for each test.
    fn temp_dir() -> std::path::PathBuf {
        let base = std::env::temp_dir();
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        // Use a sub-directory unique per test invocation.
        base.join(format!("oxigaf_gltf_test_{id}"))
    }

    // -----------------------------------------------------------------------
    // Test 1: export creates two files (.gltf + .bin)
    // -----------------------------------------------------------------------
    #[test]
    fn test_export_creates_two_files() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let out = dir.join("model.gltf");
        let model = make_model(10, 1);

        export_gltf(&model, &out).expect("export should succeed");

        assert!(
            dir.join("model.gltf").exists(),
            ".gltf file must be created"
        );
        assert!(dir.join("model.bin").exists(), ".bin file must be created");
    }

    // -----------------------------------------------------------------------
    // Test 2: .gltf is valid JSON
    // -----------------------------------------------------------------------
    #[test]
    fn test_gltf_is_valid_json() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let out = dir.join("output.gltf");
        let model = make_model(5, 0);

        export_gltf(&model, &out).expect("export should succeed");

        let raw = std::fs::read_to_string(dir.join("output.gltf")).expect("read gltf file");
        let parsed: Result<Value, _> = serde_json::from_str(&raw);
        assert!(parsed.is_ok(), "glTF file must be valid JSON");
    }

    // -----------------------------------------------------------------------
    // Test 3: gaussian count in metadata matches
    // -----------------------------------------------------------------------
    #[test]
    fn test_gaussian_count_in_metadata() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let n = 42usize;
        let out = dir.join("model.gltf");
        let model = make_model(n, 1);

        export_gltf(&model, &out).expect("export should succeed");

        let raw = std::fs::read_to_string(dir.join("model.gltf")).expect("read gltf file");
        let doc: Value = serde_json::from_str(&raw).expect("parse JSON");

        // Check top-level extension
        let count = doc["extensions"]["OXIGAF_gaussian_splat"]["gaussianCount"]
            .as_u64()
            .expect("gaussianCount should be a number");
        assert_eq!(count as usize, n, "gaussianCount must match model size");
    }

    // -----------------------------------------------------------------------
    // Test 4: bin file size is correct (N*44 + N*C*4 bytes)
    // -----------------------------------------------------------------------
    #[test]
    fn test_bin_file_size_correct() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let n = 20usize;
        let sh_degree = 3u32;
        let sh_channels = ((sh_degree + 1).pow(2) * 3) as usize;
        let out = dir.join("model.gltf");
        let model = make_model(n, sh_degree);

        export_gltf(&model, &out).expect("export should succeed");

        let bin_meta = std::fs::metadata(dir.join("model.bin")).expect("bin file exists");
        // positions(12) + rotations(16) + scales(12) + opacities(4) = 44 bytes/gaussian
        // sh_coeffs: n * sh_channels * 4 bytes
        let expected_bytes = n * 44 + n * sh_channels * 4;
        assert_eq!(
            bin_meta.len() as usize,
            expected_bytes,
            "bin file size must be N*44 + N*C*4"
        );
    }

    // -----------------------------------------------------------------------
    // Test 5: positions accessor byteOffset is 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_positions_accessor_byte_offset_is_zero() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let out = dir.join("model.gltf");
        let model = make_model(8, 1);

        export_gltf(&model, &out).expect("export should succeed");

        let raw = std::fs::read_to_string(dir.join("model.gltf")).expect("read gltf file");
        let doc: Value = serde_json::from_str(&raw).expect("parse JSON");
        let pos_offset = doc["accessors"][0]["byteOffset"]
            .as_u64()
            .expect("positions accessor byteOffset must be a number");
        assert_eq!(pos_offset, 0, "positions accessor byteOffset must be 0");
    }

    // -----------------------------------------------------------------------
    // Test 6: rotations accessor byteOffset is N*12
    // -----------------------------------------------------------------------
    #[test]
    fn test_rotations_accessor_byte_offset() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let n = 15usize;
        let out = dir.join("model.gltf");
        let model = make_model(n, 0);

        export_gltf(&model, &out).expect("export should succeed");

        let raw = std::fs::read_to_string(dir.join("model.gltf")).expect("read gltf file");
        let doc: Value = serde_json::from_str(&raw).expect("parse JSON");
        // accessor index 1 = rotations
        let rot_offset = doc["accessors"][1]["byteOffset"]
            .as_u64()
            .expect("rotations accessor byteOffset must be a number");
        assert_eq!(
            rot_offset as usize,
            n * 12,
            "rotations byteOffset must be N*12"
        );
    }

    // -----------------------------------------------------------------------
    // Test 7: empty model (0 Gaussians) exports without error
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_model_exports_without_error() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let out = dir.join("empty.gltf");
        let model = make_model(0, 1);

        let result = export_gltf(&model, &out);
        assert!(
            result.is_ok(),
            "empty model export should succeed: {result:?}"
        );

        assert!(dir.join("empty.gltf").exists(), ".gltf must be created");
        assert!(dir.join("empty.bin").exists(), ".bin must be created");

        // Verify .bin is empty (0 gaussians → 0 bytes)
        let bin_meta = std::fs::metadata(dir.join("empty.bin")).expect("bin file exists");
        assert_eq!(bin_meta.len(), 0, "empty model bin must have 0 bytes");
    }

    // -----------------------------------------------------------------------
    // Test 7b: empty model produces a *valid* document, not zero-count
    // accessors / a zero-length bufferView (both forbidden by the glTF 2.0
    // schema, which every conforming loader would reject).
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_model_has_no_accessors_or_buffer_views() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let out = dir.join("empty2.gltf");
        let model = make_model(0, 1);

        export_gltf(&model, &out).expect("export should succeed");

        let raw = std::fs::read_to_string(dir.join("empty2.gltf")).expect("read gltf file");
        let doc: Value = serde_json::from_str(&raw).expect("parse JSON");

        assert!(
            doc["accessors"].as_array().expect("array").is_empty(),
            "an empty model must emit zero accessors, not count:0 ones"
        );
        assert!(
            doc["bufferViews"].as_array().expect("array").is_empty(),
            "an empty model must emit zero bufferViews, not a byteLength:0 one"
        );
        assert!(
            doc["buffers"].as_array().expect("array").is_empty(),
            "an empty model must emit zero buffers, not a byteLength:0 one"
        );
        assert!(doc["meshes"].as_array().expect("array").is_empty());
        assert!(doc["nodes"].as_array().expect("array").is_empty());
        assert!(doc["scenes"][0]["nodes"]
            .as_array()
            .expect("array")
            .is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 5b: POSITION accessor carries min/max, required by the glTF 2.0
    // schema on any accessor referenced by a POSITION attribute.
    // -----------------------------------------------------------------------
    #[test]
    fn test_position_accessor_has_min_max() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let out = dir.join("model.gltf");
        let model = make_model(5, 1);

        export_gltf(&model, &out).expect("export should succeed");

        let raw = std::fs::read_to_string(dir.join("model.gltf")).expect("read gltf file");
        let doc: Value = serde_json::from_str(&raw).expect("parse JSON");

        // make_model positions are [i, i+0.1, i+0.2] for i in 0..5, so the
        // known bounding box is min=[0,0.1,0.2], max=[4,4.1,4.2].
        let min = doc["accessors"][0]["min"]
            .as_array()
            .expect("POSITION accessor must have a min array");
        let max = doc["accessors"][0]["max"]
            .as_array()
            .expect("POSITION accessor must have a max array");
        assert_eq!(min.len(), 3);
        assert_eq!(max.len(), 3);
        assert!((min[0].as_f64().unwrap() - 0.0).abs() < 1e-4);
        assert!((max[0].as_f64().unwrap() - 4.0).abs() < 1e-4);

        // Non-POSITION accessors (e.g. rotations, index 1) must not carry
        // min/max — the schema only requires it for POSITION.
        assert!(doc["accessors"][1].get("min").is_none());
    }

    // -----------------------------------------------------------------------
    // Test 8: sh_degree metadata is correct in JSON
    // -----------------------------------------------------------------------
    #[test]
    fn test_sh_degree_metadata_correct() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let sh_degree = 2u32;
        let out = dir.join("model.gltf");
        let model = make_model(5, sh_degree);

        export_gltf(&model, &out).expect("export should succeed");

        let raw = std::fs::read_to_string(dir.join("model.gltf")).expect("read gltf file");
        let doc: Value = serde_json::from_str(&raw).expect("parse JSON");

        // Check top-level extension
        let degree = doc["extensions"]["OXIGAF_gaussian_splat"]["shDegree"]
            .as_u64()
            .expect("shDegree should be a number");
        assert_eq!(
            degree as u32, sh_degree,
            "shDegree must match model sh_degree"
        );

        // Also verify via node-level extension
        let node_degree = doc["nodes"][0]["extensions"]["OXIGAF_gaussian_splat"]["shDegree"]
            .as_u64()
            .expect("node extension shDegree should be a number");
        assert_eq!(
            node_degree as u32, sh_degree,
            "node extension shDegree must match"
        );
    }
}
