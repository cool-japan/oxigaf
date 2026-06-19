//! Tests for Mesh OBJ and PLY export functionality.

use nalgebra as na;
use oxigaf_flame::Mesh;
use std::fs;
use std::io::{self, BufRead};

/// Build a minimal tetrahedron mesh (4 vertices, 4 triangular faces).
fn make_tetrahedron() -> Mesh {
    let vertices = vec![
        na::Point3::new(0.0_f32, 0.0, 0.0),
        na::Point3::new(1.0, 0.0, 0.0),
        na::Point3::new(0.5, 1.0, 0.0),
        na::Point3::new(0.5, 0.5, 1.0),
    ];
    let faces = vec![[0u32, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]];
    Mesh::new(vertices, faces)
}

// ---------------------------------------------------------------------------
// OBJ export
// ---------------------------------------------------------------------------

/// Export a tetrahedron to OBJ and verify:
/// - Correct number of `v ` lines (vertex positions)
/// - Correct number of `vn ` lines (vertex normals)
/// - Correct number of `f ` lines (faces)
/// - Faces are 1-indexed (no `f 0//…` tokens)
#[test]
fn test_obj_export() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = make_tetrahedron();
    let tmp = std::env::temp_dir().join("oxigaf_test_obj_export.obj");

    mesh.export_obj(&tmp)?;

    let file = fs::File::open(&tmp)?;
    let reader = io::BufReader::new(file);

    let mut v_count = 0usize;
    let mut vn_count = 0usize;
    let mut f_count = 0usize;
    let mut found_zero_index = false;

    for line in reader.lines() {
        let line = line?;
        if line.starts_with("v ") {
            v_count += 1;
        } else if line.starts_with("vn ") {
            vn_count += 1;
        } else if line.starts_with("f ") {
            f_count += 1;
            // Ensure no face token starts with index 0 (OBJ is 1-based)
            if line.contains("f 0//") || line.contains(" 0//") {
                found_zero_index = true;
            }
        }
    }

    assert_eq!(v_count, mesh.num_vertices(), "vertex count mismatch");
    assert_eq!(vn_count, mesh.num_vertices(), "normal count mismatch");
    assert_eq!(f_count, mesh.num_faces(), "face count mismatch");
    assert!(
        !found_zero_index,
        "OBJ faces must be 1-indexed, found 0-index token"
    );

    // Cleanup
    let _ = fs::remove_file(&tmp);
    Ok(())
}

// ---------------------------------------------------------------------------
// PLY export
// ---------------------------------------------------------------------------

/// Export a tetrahedron to binary PLY and verify:
/// - File begins with "ply\n"
/// - Header contains "end_header"
/// - File size matches: header_len + vertex_payload + face_payload
#[test]
fn test_ply_export() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = make_tetrahedron();
    let tmp = std::env::temp_dir().join("oxigaf_test_ply_export.ply");

    mesh.export_ply(&tmp)?;

    let bytes = fs::read(&tmp)?;

    // PLY signature
    assert!(
        bytes.starts_with(b"ply\n"),
        "PLY file must start with 'ply\\n'"
    );

    // Find end_header position (header ends after "end_header\n")
    let header_marker = b"end_header\n";
    let header_end_pos = bytes
        .windows(header_marker.len())
        .position(|w| w == header_marker)
        .ok_or("end_header marker not found in PLY file")?
        + header_marker.len();

    // Binary payload sizes:
    //   each vertex: 6 * 4 bytes (x, y, z, nx, ny, nz as f32)
    //   each face:   1 byte (count=3) + 3 * 4 bytes (i32 indices)
    let vertex_payload = mesh.num_vertices() * 6 * 4;
    let face_payload = mesh.num_faces() * (1 + 3 * 4);
    let expected_total = header_end_pos + vertex_payload + face_payload;

    assert_eq!(
        bytes.len(),
        expected_total,
        "PLY file size mismatch: header_end={header_end_pos} vertex_payload={vertex_payload} face_payload={face_payload}"
    );

    // Cleanup
    let _ = fs::remove_file(&tmp);
    Ok(())
}

// ---------------------------------------------------------------------------
// Round-trip vertex count
// ---------------------------------------------------------------------------

/// Export OBJ, parse it back, and verify that the number of `v ` lines equals
/// the original mesh vertex count.
#[test]
fn test_export_roundtrip_vertex_count() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = make_tetrahedron();
    let tmp = std::env::temp_dir().join("oxigaf_test_roundtrip.obj");

    mesh.export_obj(&tmp)?;

    let file = fs::File::open(&tmp)?;
    let reader = io::BufReader::new(file);

    let v_count = reader
        .lines()
        .map_while(Result::ok)
        .filter(|l| l.starts_with("v "))
        .count();

    assert_eq!(
        v_count,
        mesh.num_vertices(),
        "round-trip vertex count mismatch"
    );

    // Cleanup
    let _ = fs::remove_file(&tmp);
    Ok(())
}

// ---------------------------------------------------------------------------
// Larger mesh stress test
// ---------------------------------------------------------------------------

/// Export a mesh with many vertices/faces to check there are no panics or
/// truncation issues (e.g., 100 vertices arranged in a grid of triangles).
#[test]
fn test_obj_export_larger_mesh() -> Result<(), Box<dyn std::error::Error>> {
    // Build a 10×10 grid of vertices (100 vertices, 162 triangles)
    let mut vertices = Vec::new();
    for row in 0..10_u32 {
        for col in 0..10_u32 {
            vertices.push(na::Point3::new(col as f32, row as f32, 0.0_f32));
        }
    }

    let mut faces = Vec::new();
    for row in 0..9_u32 {
        for col in 0..9_u32 {
            let tl = row * 10 + col;
            let tr = tl + 1;
            let bl = tl + 10;
            let br = bl + 1;
            faces.push([tl, tr, bl]);
            faces.push([tr, br, bl]);
        }
    }

    let mesh = Mesh::new(vertices, faces);
    let tmp = std::env::temp_dir().join("oxigaf_test_large_obj.obj");

    mesh.export_obj(&tmp)?;

    let file = fs::File::open(&tmp)?;
    let reader = io::BufReader::new(file);
    let lines: Vec<String> = reader.lines().map_while(Result::ok).collect();

    let v_count = lines.iter().filter(|l| l.starts_with("v ")).count();
    let f_count = lines.iter().filter(|l| l.starts_with("f ")).count();

    assert_eq!(v_count, mesh.num_vertices());
    assert_eq!(f_count, mesh.num_faces());

    let _ = fs::remove_file(&tmp);
    Ok(())
}
