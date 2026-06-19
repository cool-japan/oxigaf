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
    // Single bufferView covering the entire binary buffer.
    let buffer_view = json!({
        "buffer": 0,
        "byteOffset": 0,
        "byteLength": total_bin_length
    });

    // Accessors — each references the single bufferView with its own byteOffset.
    let accessor_positions = build_accessor(0, pos_byte_offset, n, "VEC3", COMPONENT_TYPE_FLOAT);
    let accessor_rotations = build_accessor(0, rot_byte_offset, n, "VEC4", COMPONENT_TYPE_FLOAT);
    let accessor_scales = build_accessor(0, scale_byte_offset, n, "VEC3", COMPONENT_TYPE_FLOAT);
    let accessor_opacities =
        build_accessor(0, opacity_byte_offset, n, "SCALAR", COMPONENT_TYPE_FLOAT);
    let accessor_sh = if n == 0 || sh_channels == 0 {
        build_accessor(0, sh_byte_offset, 0, "SCALAR", COMPONENT_TYPE_FLOAT)
    } else {
        build_accessor(
            0,
            sh_byte_offset,
            n * sh_channels,
            "SCALAR",
            COMPONENT_TYPE_FLOAT,
        )
    };

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

    // Top-level extension metadata.
    let top_ext = json!({
        "gaussianCount": n,
        "shDegree": model.sh_degree
    });

    // glTF mesh primitive uses POSITION so viewers can show a point cloud.
    let primitive = json!({
        "attributes": {
            "POSITION": acc_pos_idx
        },
        "mode": 0  // POINTS
    });

    let document = json!({
        "asset": {
            "version": "2.0",
            "generator": format!("OxiGAF v{}", env!("CARGO_PKG_VERSION"))
        },
        "extensionsUsed": ["OXIGAF_gaussian_splat"],
        "scene": 0,
        "scenes": [{ "name": "GaussianScene", "nodes": [0] }],
        "nodes": [{
            "name": "GaussianSplat",
            "mesh": 0,
            "extensions": {
                "OXIGAF_gaussian_splat": gaussian_extension
            }
        }],
        "meshes": [{
            "name": "GaussianMesh",
            "primitives": [primitive]
        }],
        "buffers": [{
            "uri": bin_filename,
            "byteLength": total_bin_length
        }],
        "bufferViews": [buffer_view],
        "accessors": [
            accessor_positions,
            accessor_rotations,
            accessor_scales,
            accessor_opacities,
            accessor_sh
        ],
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
fn build_accessor(
    buffer_view_idx: usize,
    byte_offset: usize,
    count: usize,
    accessor_type: &str,
    component_type: u32,
) -> Value {
    json!({
        "bufferView": buffer_view_idx,
        "byteOffset": byte_offset,
        "componentType": component_type,
        "count": count,
        "type": accessor_type
    })
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
