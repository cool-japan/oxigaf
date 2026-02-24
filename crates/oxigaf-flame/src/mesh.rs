//! Triangle mesh with per-vertex normals.

use nalgebra as na;

/// A triangle mesh with vertex positions and per-vertex normals.
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Vertex positions.
    pub vertices: Vec<na::Point3<f32>>,
    /// Per-vertex normals (area-weighted average of incident face normals).
    pub normals: Vec<na::Vector3<f32>>,
    /// Triangle face indices (each element is `[i0, i1, i2]`).
    pub faces: Vec<[u32; 3]>,
}

impl Mesh {
    /// Build a mesh and compute per-vertex normals.
    #[must_use]
    pub fn new(vertices: Vec<na::Point3<f32>>, faces: Vec<[u32; 3]>) -> Self {
        let mut mesh = Self {
            normals: vec![na::Vector3::zeros(); vertices.len()],
            vertices,
            faces,
        };
        mesh.recompute_normals();
        mesh
    }

    /// Recompute per-vertex normals from the current vertex positions.
    pub fn recompute_normals(&mut self) {
        // Zero out
        for n in &mut self.normals {
            *n = na::Vector3::zeros();
        }

        // Accumulate area-weighted face normals
        for face in &self.faces {
            let i0 = face[0] as usize;
            let i1 = face[1] as usize;
            let i2 = face[2] as usize;

            let v0 = &self.vertices[i0];
            let v1 = &self.vertices[i1];
            let v2 = &self.vertices[i2];

            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            // Cross product -- magnitude proportional to triangle area
            let face_normal = edge1.cross(&edge2);

            self.normals[i0] += face_normal;
            self.normals[i1] += face_normal;
            self.normals[i2] += face_normal;
        }

        // Normalize
        for n in &mut self.normals {
            let len = n.norm();
            if len > 1e-10 {
                *n /= len;
            }
        }
    }

    /// Number of vertices.
    #[inline]
    #[must_use]
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Number of triangles.
    #[inline]
    #[must_use]
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    /// Compute the area of a triangle face.
    #[must_use]
    pub fn face_area(&self, face: &[u32; 3]) -> f32 {
        let v0 = &self.vertices[face[0] as usize];
        let v1 = &self.vertices[face[1] as usize];
        let v2 = &self.vertices[face[2] as usize];
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        edge1.cross(&edge2).norm() * 0.5
    }
}
