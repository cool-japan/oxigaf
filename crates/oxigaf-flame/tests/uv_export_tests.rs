//! Tests for UV coordinate export in OBJ and PLY formats.

use nalgebra as na;
use oxigaf_flame::{Mesh, MeshExportConfig};
use std::fs;
use std::io::Read;

/// Build a minimal triangle mesh with UV coordinates.
fn mesh_with_uv() -> Mesh {
    let vertices = vec![
        na::Point3::new(0.0f32, 0.0, 0.0),
        na::Point3::new(1.0f32, 0.0, 0.0),
        na::Point3::new(0.5f32, 1.0, 0.0),
    ];
    let faces = vec![[0u32, 1, 2]];
    let mut mesh = Mesh::new(vertices, faces);
    mesh.uv_coords = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
    mesh
}

/// Build a minimal triangle mesh WITHOUT UV coordinates.
fn mesh_without_uv() -> Mesh {
    let vertices = vec![
        na::Point3::new(0.0f32, 0.0, 0.0),
        na::Point3::new(1.0f32, 0.0, 0.0),
        na::Point3::new(0.5f32, 1.0, 0.0),
    ];
    let faces = vec![[0u32, 1, 2]];
    Mesh::new(vertices, faces)
}

#[test]
fn test_obj_export_with_uv_contains_vt_lines() {
    let mesh = mesh_with_uv();
    let dir = std::env::temp_dir();
    let path = dir.join("test_uv_export_vt.obj");

    mesh.export_obj(&path).expect("OBJ export failed");

    let content = fs::read_to_string(&path).expect("Failed to read OBJ file");

    // Must contain "vt " texture coordinate lines
    assert!(
        content.contains("vt "),
        "OBJ with UV must contain 'vt ' lines\n---\n{content}"
    );

    // Each face line must use v/vt/vn format (slash-separated with texture index)
    let face_lines: Vec<&str> = content.lines().filter(|l| l.starts_with("f ")).collect();
    assert!(!face_lines.is_empty(), "No face lines found");
    for line in &face_lines {
        // v/vt/vn has exactly two slashes per vertex group (and no double slash)
        assert!(
            !line.contains("//"),
            "Face line should use v/vt/vn, not v//vn: {line}"
        );
        assert!(
            line.contains('/'),
            "Face line must reference UV indices: {line}"
        );
    }

    // Clean up
    let _ = fs::remove_file(&path);
}

#[test]
fn test_obj_export_without_uv_has_no_vt_lines() {
    let mesh = mesh_without_uv();
    let dir = std::env::temp_dir();
    let path = dir.join("test_uv_export_no_vt.obj");

    mesh.export_obj(&path).expect("OBJ export failed");

    let content = fs::read_to_string(&path).expect("Failed to read OBJ file");

    // Must NOT contain "vt " lines
    assert!(
        !content.contains("vt "),
        "OBJ without UV must not contain 'vt ' lines"
    );

    // Face lines should use v//vn format
    let face_lines: Vec<&str> = content.lines().filter(|l| l.starts_with("f ")).collect();
    assert!(!face_lines.is_empty(), "No face lines found");
    for line in &face_lines {
        assert!(
            line.contains("//"),
            "Face line without UV should use v//vn: {line}"
        );
    }

    let _ = fs::remove_file(&path);
}

#[test]
fn test_obj_export_config_export_uv_false_suppresses_uv() {
    let mesh = mesh_with_uv();
    let dir = std::env::temp_dir();
    let path = dir.join("test_uv_export_suppressed.obj");

    let config = MeshExportConfig { export_uv: false };
    mesh.export_obj_with_config(&path, &config)
        .expect("OBJ export failed");

    let content = fs::read_to_string(&path).expect("Failed to read OBJ file");

    assert!(
        !content.contains("vt "),
        "export_uv=false must suppress 'vt ' lines"
    );

    let _ = fs::remove_file(&path);
}

#[test]
fn test_ply_export_with_uv_contains_st_properties() {
    let mesh = mesh_with_uv();
    let dir = std::env::temp_dir();
    let path = dir.join("test_uv_export_st.ply");

    mesh.export_ply(&path).expect("PLY export failed");

    // PLY header is ASCII; read whole file to check header
    let mut file = fs::File::open(&path).expect("Failed to open PLY file");
    let mut content = Vec::new();
    file.read_to_end(&mut content)
        .expect("Failed to read PLY file");

    // Find end of ASCII header
    let header_end = content
        .windows(11)
        .position(|w| w == b"end_header\n")
        .expect("PLY header not terminated");
    let header =
        std::str::from_utf8(&content[..header_end + 11]).expect("PLY header is not valid UTF-8");

    assert!(
        header.contains("property float s"),
        "PLY with UV must contain 'property float s'\n---\n{header}"
    );
    assert!(
        header.contains("property float t"),
        "PLY with UV must contain 'property float t'\n---\n{header}"
    );

    let _ = fs::remove_file(&path);
}

#[test]
fn test_ply_export_without_uv_has_no_st_properties() {
    let mesh = mesh_without_uv();
    let dir = std::env::temp_dir();
    let path = dir.join("test_uv_export_no_st.ply");

    mesh.export_ply(&path).expect("PLY export failed");

    let mut file = fs::File::open(&path).expect("Failed to open PLY file");
    let mut content = Vec::new();
    file.read_to_end(&mut content)
        .expect("Failed to read PLY file");

    let header_end = content
        .windows(11)
        .position(|w| w == b"end_header\n")
        .expect("PLY header not terminated");
    let header =
        std::str::from_utf8(&content[..header_end + 11]).expect("PLY header is not valid UTF-8");

    assert!(
        !header.contains("property float s"),
        "PLY without UV must not contain 'property float s'"
    );
    assert!(
        !header.contains("property float t"),
        "PLY without UV must not contain 'property float t'"
    );

    let _ = fs::remove_file(&path);
}

#[test]
fn test_uv_export_to_temp_file_roundtrip_line_structure() {
    let mesh = mesh_with_uv();
    let path = std::env::temp_dir().join("test_uv_roundtrip.obj");

    mesh.export_obj(&path).expect("OBJ export failed");

    let content = fs::read_to_string(&path).expect("Read failed");

    // Verify expected line counts: 3 vertices, 3 vt, 3 vn, 1 face
    let v_count = content.lines().filter(|l| l.starts_with("v ")).count();
    let vt_count = content.lines().filter(|l| l.starts_with("vt ")).count();
    let vn_count = content.lines().filter(|l| l.starts_with("vn ")).count();
    let f_count = content.lines().filter(|l| l.starts_with("f ")).count();

    assert_eq!(v_count, 3, "Expected 3 vertex lines");
    assert_eq!(vt_count, 3, "Expected 3 vt lines");
    assert_eq!(vn_count, 3, "Expected 3 vn lines");
    assert_eq!(f_count, 1, "Expected 1 face line");

    let _ = fs::remove_file(&path);
}
