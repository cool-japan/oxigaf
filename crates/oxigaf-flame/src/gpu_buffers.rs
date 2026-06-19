//! GPU-ready mesh buffer export for wgpu integration.
//!
//! This module provides [`GpuMeshBuffers`] and helpers for flattening
//! [`Mesh`] data into wgpu-compatible flat byte arrays with 16-byte
//! alignment (vec4 / uvec4 layout).
//!
//! # Layout
//!
//! | Buffer   | Element type | Per-element layout     |
//! |----------|--------------|------------------------|
//! | vertices | `vec4<f32>`  | `[x, y, z, 1.0]`      |
//! | normals  | `vec4<f32>`  | `[nx, ny, nz, 0.0]`   |
//! | faces    | `uvec4<u32>` | `[v0, v1, v2, 0]`     |

use crate::{FlameError, Mesh};
use nalgebra as na;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for GPU buffer export.
#[derive(Debug, Clone)]
pub struct GpuBufferConfig {
    /// Normalize normals to unit length (default: `true`).
    pub normalize_normals: bool,

    /// Recompute normals from face geometry instead of using stored normals
    /// (default: `false`).
    pub recompute_normals: bool,

    /// Whether to include degenerate (zero-area) faces (default: `false` —
    /// skip them).
    pub include_degenerate: bool,
}

impl Default for GpuBufferConfig {
    fn default() -> Self {
        Self {
            normalize_normals: true,
            recompute_normals: false,
            include_degenerate: false,
        }
    }
}

// ---------------------------------------------------------------------------
// GpuMeshBuffers
// ---------------------------------------------------------------------------

/// GPU-ready mesh data in wgpu-compatible format.
///
/// All arrays use 16-byte alignment (vec4 / uvec4 layout) for optimal GPU
/// access.
///
/// # Buffer layouts
///
/// * `vertices` — `[V * 4]` f32, each vertex is `[x, y, z, 1.0]`.
/// * `normals`  — `[V * 4]` f32, each normal is `[nx, ny, nz, 0.0]`.
/// * `faces`    — `[F * 4]` u32, each face is `[v0, v1, v2, 0]`.
#[derive(Debug, Clone)]
pub struct GpuMeshBuffers {
    /// Vertex positions: `[V * 4]` f32 in `[x, y, z, 1.0]` order.
    ///
    /// Length: `num_vertices * 4`.
    pub vertices: Vec<f32>,

    /// Vertex normals: `[V * 4]` f32 in `[nx, ny, nz, 0.0]` order.
    ///
    /// Length: `num_vertices * 4`.
    pub normals: Vec<f32>,

    /// Triangle face indices: `[F * 4]` u32 in `[v0, v1, v2, 0]` order.
    ///
    /// Length: `num_faces * 4`.
    pub faces: Vec<u32>,

    /// Number of vertices.
    pub num_vertices: u32,

    /// Number of (non-degenerate) faces stored in the buffer.
    pub num_faces: u32,
}

impl GpuMeshBuffers {
    // -----------------------------------------------------------------------
    // Size helpers
    // -----------------------------------------------------------------------

    /// Memory size of the vertex buffer in bytes.
    #[inline]
    #[must_use]
    pub fn vertex_buffer_bytes(&self) -> usize {
        self.vertices.len() * std::mem::size_of::<f32>()
    }

    /// Memory size of the normal buffer in bytes.
    #[inline]
    #[must_use]
    pub fn normal_buffer_bytes(&self) -> usize {
        self.normals.len() * std::mem::size_of::<f32>()
    }

    /// Memory size of the face buffer in bytes.
    #[inline]
    #[must_use]
    pub fn face_buffer_bytes(&self) -> usize {
        self.faces.len() * std::mem::size_of::<u32>()
    }

    /// Total GPU memory footprint in bytes (vertices + normals + faces).
    #[inline]
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.vertex_buffer_bytes() + self.normal_buffer_bytes() + self.face_buffer_bytes()
    }

    // -----------------------------------------------------------------------
    // Raw byte accessors (for `wgpu::Queue::write_buffer`)
    // -----------------------------------------------------------------------

    /// Raw bytes for the vertex buffer.
    #[inline]
    #[must_use]
    pub fn vertex_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.vertices)
    }

    /// Raw bytes for the normal buffer.
    #[inline]
    #[must_use]
    pub fn normal_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.normals)
    }

    /// Raw bytes for the face buffer.
    #[inline]
    #[must_use]
    pub fn face_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.faces)
    }

    // -----------------------------------------------------------------------
    // Element accessors
    // -----------------------------------------------------------------------

    /// Get vertex position at `idx` as `[x, y, z]`.
    ///
    /// Returns `None` if `idx >= num_vertices`.
    #[inline]
    #[must_use]
    pub fn vertex_position(&self, idx: usize) -> Option<[f32; 3]> {
        let base = idx.checked_mul(4)?;
        if base + 3 > self.vertices.len() {
            return None;
        }
        Some([
            self.vertices[base],
            self.vertices[base + 1],
            self.vertices[base + 2],
        ])
    }

    /// Get vertex normal at `idx` as `[nx, ny, nz]`.
    ///
    /// Returns `None` if `idx >= num_vertices`.
    #[inline]
    #[must_use]
    pub fn vertex_normal(&self, idx: usize) -> Option<[f32; 3]> {
        let base = idx.checked_mul(4)?;
        if base + 3 > self.normals.len() {
            return None;
        }
        Some([
            self.normals[base],
            self.normals[base + 1],
            self.normals[base + 2],
        ])
    }

    /// Get face vertex indices at face `idx` as `[v0, v1, v2]`.
    ///
    /// Returns `None` if `idx >= num_faces`.
    #[inline]
    #[must_use]
    pub fn face_indices(&self, idx: usize) -> Option<[u32; 3]> {
        let base = idx.checked_mul(4)?;
        if base + 3 > self.faces.len() {
            return None;
        }
        Some([self.faces[base], self.faces[base + 1], self.faces[base + 2]])
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    /// Validate that all face indices are within vertex bounds.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::IndexOutOfBounds`] if any face references a
    /// vertex index `>= num_vertices`.
    pub fn validate(&self) -> Result<(), FlameError> {
        let nv = self.num_vertices as usize;
        for face_idx in 0..self.num_faces as usize {
            let base = face_idx * 4;
            for slot in 0..3_usize {
                let vi = self.faces[base + slot] as usize;
                if vi >= nv {
                    return Err(FlameError::index_out_of_bounds(
                        format!("face[{face_idx}][{slot}]"),
                        vi,
                        nv,
                    ));
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Mesh → GpuMeshBuffers
// ---------------------------------------------------------------------------

impl Mesh {
    /// Export this mesh to GPU-ready buffer format using default config.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::ShapeMismatch`] if the vertex and normal counts
    /// differ, or [`FlameError::IndexOutOfBounds`] if any face index is out of
    /// range.
    pub fn to_gpu_buffers(&self) -> Result<GpuMeshBuffers, FlameError> {
        self.to_gpu_buffers_with_config(&GpuBufferConfig::default())
    }

    /// Export this mesh to GPU-ready buffer format with explicit config.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::ShapeMismatch`] if the vertex and normal counts
    /// differ, or [`FlameError::IndexOutOfBounds`] if any face index is out of
    /// range.
    pub fn to_gpu_buffers_with_config(
        &self,
        config: &GpuBufferConfig,
    ) -> Result<GpuMeshBuffers, FlameError> {
        let num_v = self.vertices.len();

        // ------------------------------------------------------------------
        // Sanity check: normals must be present for every vertex when not
        // recomputing (we still need the same length to build the buffer).
        // ------------------------------------------------------------------
        if !config.recompute_normals && self.normals.len() != num_v {
            return Err(FlameError::ShapeMismatch {
                name: "normals".into(),
                expected: format!("{num_v}"),
                got: format!("{}", self.normals.len()),
            });
        }

        // ------------------------------------------------------------------
        // Step 1: Build flat vertex buffer — [x, y, z, 1.0] per vertex.
        // ------------------------------------------------------------------
        let mut vert_buf = Vec::with_capacity(num_v * 4);
        for v in &self.vertices {
            vert_buf.push(v.x);
            vert_buf.push(v.y);
            vert_buf.push(v.z);
            vert_buf.push(1.0_f32);
        }

        // ------------------------------------------------------------------
        // Step 2: Build normals (optionally recomputed, then normalized).
        // ------------------------------------------------------------------
        let normal_data: Vec<na::Vector3<f32>> = if config.recompute_normals {
            recompute_normals_from_faces(&self.vertices, &self.faces)
        } else {
            self.normals.clone()
        };

        let mut norm_buf = Vec::with_capacity(num_v * 4);
        for n in &normal_data {
            let (nx, ny, nz) = if config.normalize_normals {
                let len = n.norm();
                if len > 1e-10 {
                    (n.x / len, n.y / len, n.z / len)
                } else {
                    (0.0_f32, 0.0_f32, 0.0_f32)
                }
            } else {
                (n.x, n.y, n.z)
            };
            norm_buf.push(nx);
            norm_buf.push(ny);
            norm_buf.push(nz);
            norm_buf.push(0.0_f32);
        }

        // ------------------------------------------------------------------
        // Step 3: Build face buffer, optionally skipping degenerate faces.
        // ------------------------------------------------------------------
        let mut face_buf: Vec<u32> = Vec::with_capacity(self.faces.len() * 4);
        let mut kept_faces: u32 = 0;

        for face in &self.faces {
            let i0 = face[0] as usize;
            let i1 = face[1] as usize;
            let i2 = face[2] as usize;

            // Bounds check each index.
            if i0 >= num_v {
                return Err(FlameError::index_out_of_bounds("face vertex 0", i0, num_v));
            }
            if i1 >= num_v {
                return Err(FlameError::index_out_of_bounds("face vertex 1", i1, num_v));
            }
            if i2 >= num_v {
                return Err(FlameError::index_out_of_bounds("face vertex 2", i2, num_v));
            }

            // Degenerate (zero-area) check.
            if !config.include_degenerate {
                let v0 = &self.vertices[i0];
                let v1 = &self.vertices[i1];
                let v2 = &self.vertices[i2];
                let edge1 = v1 - v0;
                let edge2 = v2 - v0;
                let cross_mag = edge1.cross(&edge2).norm();
                if cross_mag < 1e-10 {
                    // Skip degenerate face.
                    continue;
                }
            }

            face_buf.push(face[0]);
            face_buf.push(face[1]);
            face_buf.push(face[2]);
            face_buf.push(0_u32); // padding
            kept_faces += 1;
        }

        Ok(GpuMeshBuffers {
            vertices: vert_buf,
            normals: norm_buf,
            faces: face_buf,
            num_vertices: num_v as u32,
            num_faces: kept_faces,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Recompute area-weighted per-vertex normals from scratch.
///
/// Mirrors `Mesh::recompute_normals` but operates on slices rather than
/// `&mut self`, so it can be called from an immutable context.
fn recompute_normals_from_faces(
    vertices: &[na::Point3<f32>],
    faces: &[[u32; 3]],
) -> Vec<na::Vector3<f32>> {
    let mut normals = vec![na::Vector3::zeros(); vertices.len()];

    for face in faces {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;

        // Guard against out-of-bounds (best-effort; validate() catches errors).
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }

        let v0 = &vertices[i0];
        let v1 = &vertices[i1];
        let v2 = &vertices[i2];

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let face_normal = edge1.cross(&edge2); // magnitude ∝ area

        normals[i0] += face_normal;
        normals[i1] += face_normal;
        normals[i2] += face_normal;
    }

    // Normalize each accumulated normal.
    for n in &mut normals {
        let len = n.norm();
        if len > 1e-10 {
            *n /= len;
        }
    }

    normals
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// A flat triangle in the XY-plane: vertices (0,0,0), (1,0,0), (0,1,0).
    fn single_triangle_mesh() -> Mesh {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0),
            na::Point3::new(1.0f32, 0.0, 0.0),
            na::Point3::new(0.0f32, 1.0, 0.0),
        ];
        let faces = vec![[0_u32, 1, 2]];
        Mesh::new(vertices, faces)
    }

    /// A unit-square mesh (2 triangles, 4 vertices).
    fn unit_square_mesh() -> Mesh {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0),
            na::Point3::new(1.0f32, 0.0, 0.0),
            na::Point3::new(0.0f32, 1.0, 0.0),
            na::Point3::new(1.0f32, 1.0, 0.0),
        ];
        let faces = vec![[0_u32, 1, 2], [1, 3, 2]];
        Mesh::new(vertices, faces)
    }

    // -----------------------------------------------------------------------
    // Basic conversion
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_gpu_buffers_single_triangle_succeeds() {
        let mesh = single_triangle_mesh();
        let buf = mesh
            .to_gpu_buffers()
            .expect("single triangle should succeed");
        assert_eq!(buf.num_vertices, 3);
        assert_eq!(buf.num_faces, 1);
    }

    #[test]
    fn test_vertex_buffer_length_equals_num_vertices_times_4() {
        let mesh = unit_square_mesh();
        let buf = mesh.to_gpu_buffers().expect("unit square");
        assert_eq!(buf.vertices.len(), buf.num_vertices as usize * 4);
    }

    #[test]
    fn test_normal_buffer_length_equals_num_vertices_times_4() {
        let mesh = unit_square_mesh();
        let buf = mesh.to_gpu_buffers().expect("unit square");
        assert_eq!(buf.normals.len(), buf.num_vertices as usize * 4);
    }

    #[test]
    fn test_face_buffer_length_equals_num_faces_times_4() {
        let mesh = unit_square_mesh();
        let buf = mesh.to_gpu_buffers().expect("unit square");
        assert_eq!(buf.faces.len(), buf.num_faces as usize * 4);
    }

    // -----------------------------------------------------------------------
    // Element accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_vertex_position_returns_first_vertex() {
        let mesh = single_triangle_mesh();
        let buf = mesh.to_gpu_buffers().expect("single triangle");
        let pos = buf.vertex_position(0).expect("vertex 0 must exist");
        // First vertex is (0,0,0)
        assert!((pos[0]).abs() < 1e-6);
        assert!((pos[1]).abs() < 1e-6);
        assert!((pos[2]).abs() < 1e-6);
    }

    #[test]
    fn test_vertex_normal_returns_normalized_normal() {
        let mesh = single_triangle_mesh();
        let buf = mesh.to_gpu_buffers().expect("single triangle");
        let norm = buf.vertex_normal(0).expect("normal 0 must exist");
        let len = (norm[0] * norm[0] + norm[1] * norm[1] + norm[2] * norm[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-5,
            "normal should be unit length, got {len}"
        );
    }

    #[test]
    fn test_face_indices_returns_correct_indices() {
        let mesh = single_triangle_mesh();
        let buf = mesh.to_gpu_buffers().expect("single triangle");
        let fi = buf.face_indices(0).expect("face 0 must exist");
        assert_eq!(fi, [0, 1, 2]);
    }

    // -----------------------------------------------------------------------
    // Padding values
    // -----------------------------------------------------------------------

    #[test]
    fn test_vertex_w_component_is_1_0() {
        let mesh = unit_square_mesh();
        let buf = mesh.to_gpu_buffers().expect("unit square");
        for v_idx in 0..buf.num_vertices as usize {
            let w = buf.vertices[v_idx * 4 + 3];
            assert!(
                (w - 1.0).abs() < 1e-7,
                "vertex[{v_idx}].w should be 1.0, got {w}"
            );
        }
    }

    #[test]
    fn test_normal_w_component_is_0_0() {
        let mesh = unit_square_mesh();
        let buf = mesh.to_gpu_buffers().expect("unit square");
        for v_idx in 0..buf.num_vertices as usize {
            let w = buf.normals[v_idx * 4 + 3];
            assert!(w.abs() < 1e-7, "normal[{v_idx}].w should be 0.0, got {w}");
        }
    }

    #[test]
    fn test_face_w_component_is_0() {
        let mesh = unit_square_mesh();
        let buf = mesh.to_gpu_buffers().expect("unit square");
        for f_idx in 0..buf.num_faces as usize {
            let w = buf.faces[f_idx * 4 + 3];
            assert_eq!(w, 0, "face[{f_idx}].w should be 0, got {w}");
        }
    }

    // -----------------------------------------------------------------------
    // Validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_passes_for_valid_mesh() {
        let mesh = unit_square_mesh();
        let buf = mesh.to_gpu_buffers().expect("unit square");
        buf.validate().expect("valid mesh should pass validation");
    }

    #[test]
    fn test_validate_fails_for_out_of_bounds_index() {
        // Build a buffer with a face pointing beyond num_vertices.
        let buf = GpuMeshBuffers {
            vertices: vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            normals: vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            faces: vec![0, 1, 99, 0], // index 99 is out of bounds (only 2 vertices)
            num_vertices: 2,
            num_faces: 1,
        };
        assert!(
            buf.validate().is_err(),
            "validation should fail for out-of-bounds face index"
        );
    }

    // -----------------------------------------------------------------------
    // Byte sizes
    // -----------------------------------------------------------------------

    #[test]
    fn test_total_bytes_equals_sum_of_three_buffers() {
        let mesh = unit_square_mesh();
        let buf = mesh.to_gpu_buffers().expect("unit square");
        assert_eq!(
            buf.total_bytes(),
            buf.vertex_buffer_bytes() + buf.normal_buffer_bytes() + buf.face_buffer_bytes()
        );
    }

    #[test]
    fn test_vertex_bytes_length_matches_float_count_times_4() {
        let mesh = unit_square_mesh();
        let buf = mesh.to_gpu_buffers().expect("unit square");
        assert_eq!(buf.vertex_bytes().len(), buf.vertices.len() * 4);
    }

    // -----------------------------------------------------------------------
    // Configuration: normalize_normals
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalize_normals_true_produces_unit_length_normals() {
        let mesh = unit_square_mesh();
        let config = GpuBufferConfig {
            normalize_normals: true,
            ..Default::default()
        };
        let buf = mesh
            .to_gpu_buffers_with_config(&config)
            .expect("unit square");
        for v_idx in 0..buf.num_vertices as usize {
            let base = v_idx * 4;
            let nx = buf.normals[base];
            let ny = buf.normals[base + 1];
            let nz = buf.normals[base + 2];
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "normal[{v_idx}] should be unit length, len={len}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Configuration: recompute_normals
    // -----------------------------------------------------------------------

    #[test]
    fn test_recompute_normals_true_produces_consistent_normals() {
        let mesh = single_triangle_mesh();
        let config = GpuBufferConfig {
            recompute_normals: true,
            ..Default::default()
        };
        let buf = mesh
            .to_gpu_buffers_with_config(&config)
            .expect("single triangle with recomputed normals");

        // All vertex normals of a flat triangle should point in the same direction.
        let n0 = buf.vertex_normal(0).expect("normal 0");
        let n1 = buf.vertex_normal(1).expect("normal 1");
        let n2 = buf.vertex_normal(2).expect("normal 2");

        // Each should be approximately (0, 0, 1) for a CCW triangle in XY-plane.
        for (i, n) in [n0, n1, n2].iter().enumerate() {
            assert!(
                n[2].abs() > 0.9,
                "normal[{i}] z-component should be ~1.0, got {}",
                n[2]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_mesh_produces_empty_buffers_and_passes_validation() {
        let mesh = Mesh::new(vec![], vec![]);
        let buf = mesh.to_gpu_buffers().expect("empty mesh");
        assert_eq!(buf.num_vertices, 0);
        assert_eq!(buf.num_faces, 0);
        assert!(buf.vertices.is_empty());
        assert!(buf.normals.is_empty());
        assert!(buf.faces.is_empty());
        buf.validate().expect("empty mesh passes validation");
    }

    #[test]
    fn test_include_degenerate_false_skips_zero_area_faces() {
        // A mesh with one real triangle and one degenerate (collinear) face.
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0), // 0
            na::Point3::new(1.0f32, 0.0, 0.0), // 1
            na::Point3::new(0.0f32, 1.0, 0.0), // 2
            na::Point3::new(0.5f32, 0.0, 0.0), // 3 — lies on edge 0-1 (degenerate)
        ];
        // Face [0,1,3] has v3 = midpoint of v0-v1, area = 0.
        let faces = vec![[0_u32, 1, 2], [0, 1, 3]];

        // Use Mesh::new which always recomputes normals.
        let mesh = Mesh {
            vertices: vertices.clone(),
            normals: {
                // Provide dummy normals (length matches vertices).
                let mut n = vec![na::Vector3::zeros(); 4];
                // Proper normal for vertex 0, 1, 2 from face [0,1,2]
                let edge1 = vertices[1] - vertices[0];
                let edge2 = vertices[2] - vertices[0];
                let fn_ = edge1.cross(&edge2).normalize();
                n[0] = fn_;
                n[1] = fn_;
                n[2] = fn_;
                n[3] = fn_; // degenerate vertex
                n
            },
            faces,
            uv_coords: Vec::new(),
        };

        let config = GpuBufferConfig {
            include_degenerate: false,
            ..Default::default()
        };
        let buf = mesh
            .to_gpu_buffers_with_config(&config)
            .expect("mesh with degenerate face");

        // Only 1 non-degenerate face should be kept.
        assert_eq!(
            buf.num_faces, 1,
            "degenerate face should be skipped, num_faces={}",
            buf.num_faces
        );
        assert_eq!(
            buf.faces.len(),
            4,
            "face buffer should hold exactly one uvec4"
        );
    }
}
