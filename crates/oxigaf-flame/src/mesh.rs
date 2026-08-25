//! Triangle mesh with per-vertex normals.

use kiddo::{KdTree, SquaredEuclidean};
use nalgebra as na;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::FlameError;

// ---------------------------------------------------------------------------
// MeshExportConfig
// ---------------------------------------------------------------------------

/// Configuration for OBJ / PLY mesh export.
#[derive(Debug, Clone)]
pub struct MeshExportConfig {
    /// When `true` (default), UV coordinates are written to the export file
    /// if the mesh has UV data.  When `false`, UV data is omitted even if
    /// `uv_coords` is non-empty.
    pub export_uv: bool,
}

impl Default for MeshExportConfig {
    fn default() -> Self {
        Self { export_uv: true }
    }
}

/// A triangle mesh with vertex positions and per-vertex normals.
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Vertex positions.
    pub vertices: Vec<na::Point3<f32>>,
    /// Per-vertex normals (area-weighted average of incident face normals).
    pub normals: Vec<na::Vector3<f32>>,
    /// Triangle face indices (each element is `[i0, i1, i2]`).
    pub faces: Vec<[u32; 3]>,
    /// Optional UV texture coordinates, one per vertex.
    /// Length: 0 (no UVs) or `num_vertices`.
    pub uv_coords: Vec<[f32; 2]>,
}

impl Mesh {
    /// Build a mesh and compute per-vertex normals.
    #[must_use]
    pub fn new(vertices: Vec<na::Point3<f32>>, faces: Vec<[u32; 3]>) -> Self {
        let mut mesh = Self {
            normals: vec![na::Vector3::zeros(); vertices.len()],
            vertices,
            faces,
            uv_coords: Vec::new(),
        };
        mesh.recompute_normals();
        mesh
    }

    /// Recompute per-vertex normals from the current vertex positions.
    ///
    /// A face referencing a vertex index `>= self.vertices.len()` (a
    /// malformed/corrupted mesh) is skipped rather than indexed, matching
    /// the same guard `model::compute_normals_into` and
    /// `gpu_buffers::recompute_normals_from_faces` already use.
    pub fn recompute_normals(&mut self) {
        // Zero out
        for n in &mut self.normals {
            *n = na::Vector3::zeros();
        }

        let n_verts = self.vertices.len();

        // Accumulate area-weighted face normals
        for face in &self.faces {
            let i0 = face[0] as usize;
            let i1 = face[1] as usize;
            let i2 = face[2] as usize;

            if i0 >= n_verts || i1 >= n_verts || i2 >= n_verts {
                continue;
            }

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
    ///
    /// Returns `0.0` if any of `face`'s vertex indices is out of range for
    /// this mesh, rather than panicking -- a malformed/corrupted face
    /// carries no reliable area, so it is treated the same as any other
    /// degenerate (zero-area) triangle.
    #[must_use]
    pub fn face_area(&self, face: &[u32; 3]) -> f32 {
        let n_verts = self.vertices.len();
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;
        if i0 >= n_verts || i1 >= n_verts || i2 >= n_verts {
            return 0.0;
        }
        let v0 = &self.vertices[i0];
        let v1 = &self.vertices[i1];
        let v2 = &self.vertices[i2];
        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        edge1.cross(&edge2).norm() * 0.5
    }

    /// Build a mesh, validating that every face index is in range for
    /// `vertices`, before computing per-vertex normals.
    ///
    /// Unlike [`Mesh::new`] (which silently skips a malformed face when
    /// accumulating normals, leaving affected vertices with a zero/degenerate
    /// normal), this rejects the mesh outright -- useful when loading mesh
    /// data from an untrusted or external source, where corruption should be
    /// surfaced rather than silently absorbed.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::IndexOutOfBounds`] if any face references a
    /// vertex index `>= vertices.len()`.
    pub fn try_new(
        vertices: Vec<na::Point3<f32>>,
        faces: Vec<[u32; 3]>,
    ) -> Result<Self, FlameError> {
        let n_verts = vertices.len();
        for (face_idx, face) in faces.iter().enumerate() {
            for (c, &idx) in face.iter().enumerate() {
                if idx as usize >= n_verts {
                    return Err(FlameError::index_out_of_bounds(
                        format!("faces[{face_idx}][{c}]"),
                        idx as usize,
                        n_verts,
                    ));
                }
            }
        }
        Ok(Self::new(vertices, faces))
    }

    /// Export this mesh to Wavefront OBJ format using default config.
    ///
    /// Equivalent to `export_obj_with_config(path, &MeshExportConfig::default())`.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::Export`] if the file cannot be created or written.
    pub fn export_obj(&self, path: &Path) -> Result<(), FlameError> {
        self.export_obj_with_config(path, &MeshExportConfig::default())
    }

    /// Export this mesh to Wavefront OBJ format with explicit configuration.
    ///
    /// The file is written with:
    /// - Vertex positions (`v x y z`)
    /// - UV texture coordinates (`vt u v`) when `config.export_uv` is `true`
    ///   and `uv_coords` count matches vertex count
    /// - Per-vertex normals (`vn nx ny nz`)
    /// - Triangle faces:
    ///   - `f v/vt/vn v/vt/vn v/vt/vn` when UV data is present and valid
    ///   - `f v//vn v//vn v//vn` otherwise
    ///
    /// Indices are 1-based (OBJ convention). Per-vertex normals are used directly
    /// because `Mesh::new()` always computes them via `recompute_normals()`.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::Export`] if the file cannot be created or written.
    pub fn export_obj_with_config(
        &self,
        path: &Path,
        config: &MeshExportConfig,
    ) -> Result<(), FlameError> {
        let file = File::create(path).map_err(|e| {
            FlameError::export(
                "OBJ",
                format!("failed to create file '{}': {e}", path.display()),
            )
        })?;
        let mut writer = BufWriter::new(file);

        let has_uv = config.export_uv && self.uv_coords.len() == self.vertices.len();

        writeln!(writer, "# OxiGAF FLAME mesh export")
            .map_err(|e| FlameError::export("OBJ", format!("write error: {e}")))?;
        writeln!(writer, "# Vertices: {}", self.vertices.len())
            .map_err(|e| FlameError::export("OBJ", format!("write error: {e}")))?;
        writeln!(writer, "# Faces: {}", self.faces.len())
            .map_err(|e| FlameError::export("OBJ", format!("write error: {e}")))?;

        // Vertex positions
        for v in &self.vertices {
            writeln!(writer, "v {} {} {}", v.x, v.y, v.z)
                .map_err(|e| FlameError::export("OBJ", format!("write error: {e}")))?;
        }

        // UV texture coordinates (one per vertex when present)
        if has_uv {
            for uv in &self.uv_coords {
                writeln!(writer, "vt {} {}", uv[0], uv[1])
                    .map_err(|e| FlameError::export("OBJ", format!("write error: {e}")))?;
            }
        }

        // Per-vertex normals
        for n in &self.normals {
            writeln!(writer, "vn {} {} {}", n.x, n.y, n.z)
                .map_err(|e| FlameError::export("OBJ", format!("write error: {e}")))?;
        }

        // Triangle faces — OBJ is 1-indexed; vertex index == normal index == uv index
        for face in &self.faces {
            let i0 = face[0] + 1;
            let i1 = face[1] + 1;
            let i2 = face[2] + 1;
            if has_uv {
                writeln!(writer, "f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}")
                    .map_err(|e| FlameError::export("OBJ", format!("write error: {e}")))?;
            } else {
                writeln!(writer, "f {i0}//{i0} {i1}//{i1} {i2}//{i2}")
                    .map_err(|e| FlameError::export("OBJ", format!("write error: {e}")))?;
            }
        }

        writer
            .flush()
            .map_err(|e| FlameError::export("OBJ", format!("flush error: {e}")))?;

        Ok(())
    }

    /// Export this mesh to binary little-endian PLY format using default config.
    ///
    /// Equivalent to `export_ply_with_config(path, &MeshExportConfig::default())`.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::Export`] if the file cannot be created or written.
    pub fn export_ply(&self, path: &Path) -> Result<(), FlameError> {
        self.export_ply_with_config(path, &MeshExportConfig::default())
    }

    /// Export this mesh to binary little-endian PLY format with explicit configuration.
    ///
    /// The output PLY contains:
    /// - ASCII header describing element layout
    /// - Binary vertex data: `(x, y, z, nx, ny, nz)` as `f32` LE, and optionally
    ///   `(s, t)` UV coordinates when `config.export_uv` is `true` and UV count matches
    /// - Binary face data: `3` (`uchar`) followed by three `i32` LE vertex indices
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::Export`] if the file cannot be created or written.
    pub fn export_ply_with_config(
        &self,
        path: &Path,
        config: &MeshExportConfig,
    ) -> Result<(), FlameError> {
        let file = File::create(path).map_err(|e| {
            FlameError::export(
                "PLY",
                format!("failed to create file '{}': {e}", path.display()),
            )
        })?;
        let mut writer = BufWriter::new(file);

        let has_uv = config.export_uv && self.uv_coords.len() == self.vertices.len();

        // ASCII header (must end with "end_header\n")
        let uv_props = if has_uv {
            "property float s\nproperty float t\n"
        } else {
            ""
        };

        write!(
            writer,
            "ply\n\
             format binary_little_endian 1.0\n\
             comment OxiGAF FLAME mesh export\n\
             element vertex {}\n\
             property float x\n\
             property float y\n\
             property float z\n\
             property float nx\n\
             property float ny\n\
             property float nz\n\
             {uv_props}\
             element face {}\n\
             property list uchar int vertex_indices\n\
             end_header\n",
            self.vertices.len(),
            self.faces.len()
        )
        .map_err(|e| FlameError::export("PLY", format!("header write error: {e}")))?;

        // Binary vertex data
        for (i, (v, n)) in self.vertices.iter().zip(self.normals.iter()).enumerate() {
            writer
                .write_all(&v.x.to_le_bytes())
                .map_err(|e| FlameError::export("PLY", format!("vertex write error: {e}")))?;
            writer
                .write_all(&v.y.to_le_bytes())
                .map_err(|e| FlameError::export("PLY", format!("vertex write error: {e}")))?;
            writer
                .write_all(&v.z.to_le_bytes())
                .map_err(|e| FlameError::export("PLY", format!("vertex write error: {e}")))?;
            writer
                .write_all(&n.x.to_le_bytes())
                .map_err(|e| FlameError::export("PLY", format!("normal write error: {e}")))?;
            writer
                .write_all(&n.y.to_le_bytes())
                .map_err(|e| FlameError::export("PLY", format!("normal write error: {e}")))?;
            writer
                .write_all(&n.z.to_le_bytes())
                .map_err(|e| FlameError::export("PLY", format!("normal write error: {e}")))?;
            if has_uv {
                let uv = self.uv_coords[i];
                writer
                    .write_all(&uv[0].to_le_bytes())
                    .map_err(|e| FlameError::export("PLY", format!("uv write error: {e}")))?;
                writer
                    .write_all(&uv[1].to_le_bytes())
                    .map_err(|e| FlameError::export("PLY", format!("uv write error: {e}")))?;
            }
        }

        // Binary face data: uchar (3) + 3 × i32 (vertex indices)
        for face in &self.faces {
            writer
                .write_all(&[3u8])
                .map_err(|e| FlameError::export("PLY", format!("face count write error: {e}")))?;
            for &idx in face {
                let signed = idx.cast_signed();
                writer.write_all(&signed.to_le_bytes()).map_err(|e| {
                    FlameError::export("PLY", format!("face index write error: {e}"))
                })?;
            }
        }

        writer
            .flush()
            .map_err(|e| FlameError::export("PLY", format!("flush error: {e}")))?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Vertex mask (semantic region segmentation)
    // -----------------------------------------------------------------------

    /// Compute geometric vertex masks for this mesh.
    ///
    /// Classifies each vertex into a semantic [`crate::FaceRegion`] based on its
    /// position in the FLAME canonical coordinate system.
    ///
    /// The geometric thresholds are only meaningful for a mesh in the FLAME
    /// canonical pose. This method delegates to
    /// [`VertexMask::from_vertices`](crate::vertex_mask::VertexMask::from_vertices),
    /// which emits a `tracing::warn!` **on every call** for a mesh that fails
    /// the canonical-pose check. That is fine for a one-shot call, but a
    /// per-frame caller would flood the log; such callers should use
    /// [`Mesh::vertex_mask_checked`] and handle the error once instead.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra as na;
    /// # use oxigaf_flame::Mesh;
    /// let vertices = vec![
    ///     na::Point3::new(0.0f32, 0.20, 0.0), // Scalp
    ///     na::Point3::new(0.0f32, 0.05, 0.0), // Face
    /// ];
    /// let mesh = Mesh::new(vertices, vec![[0, 1, 0]]);
    /// let vm = mesh.vertex_mask();
    /// assert_eq!(vm.num_vertices, 2);
    /// ```
    #[must_use]
    pub fn vertex_mask(&self) -> crate::vertex_mask::VertexMask {
        crate::vertex_mask::VertexMask::from_vertices(&self.vertices)
    }

    /// Compute geometric vertex masks, rejecting a non-canonical mesh.
    ///
    /// Identical classification to [`Mesh::vertex_mask`], but the
    /// canonical-pose precondition is reported as an error instead of a
    /// `tracing::warn!`. Prefer this in a loop (per-frame region lookups, batch
    /// processing): the caller decides once whether to canonicalize the mesh,
    /// fall back to real FLAME region indices via
    /// [`VertexMask::from_region_map`](crate::vertex_mask::VertexMask::from_region_map),
    /// or surface the failure — instead of emitting one warning per call.
    ///
    /// # Errors
    ///
    /// Returns [`VertexMaskError::NonCanonicalPose`](crate::VertexMaskError::NonCanonicalPose)
    /// when the vertex centroid is too far from the origin or the bounding box
    /// does not have canonical FLAME head extents.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra as na;
    /// # use oxigaf_flame::{Mesh, VertexMaskError};
    /// // A head with canonical extents, centred on the origin, passes.
    /// let canonical = vec![
    ///     na::Point3::new(-0.08f32, -0.15, -0.08),
    ///     na::Point3::new(0.08f32, 0.15, 0.08),
    ///     na::Point3::new(0.0f32, 0.0, 0.05),
    /// ];
    /// let mesh = Mesh::new(canonical.clone(), vec![[0, 1, 2]]);
    /// assert!(mesh.vertex_mask_checked().is_ok());
    ///
    /// // The same head translated up by 0.2 m is reported, instead of being
    /// // classified against thresholds evaluated in the wrong frame.
    /// let translated: Vec<_> = canonical
    ///     .iter()
    ///     .map(|p| na::Point3::new(p.x, p.y + 0.2, p.z))
    ///     .collect();
    /// let posed = Mesh::new(translated, vec![[0, 1, 2]]);
    /// assert!(matches!(
    ///     posed.vertex_mask_checked(),
    ///     Err(VertexMaskError::NonCanonicalPose { .. })
    /// ));
    /// ```
    pub fn vertex_mask_checked(
        &self,
    ) -> Result<crate::vertex_mask::VertexMask, crate::vertex_mask::VertexMaskError> {
        crate::vertex_mask::VertexMask::from_vertices_checked(&self.vertices)
    }

    // -----------------------------------------------------------------------
    // KD-tree nearest-vertex queries
    // -----------------------------------------------------------------------

    /// Build a KD-tree over vertex positions for nearest-vertex queries.
    ///
    /// Returns a KD-tree indexed by vertex index (as `u64`). The tree can be
    /// reused for multiple queries, which is more efficient than calling
    /// [`nearest_vertex`](Self::nearest_vertex) repeatedly (which rebuilds the
    /// tree on each call).
    ///
    /// # Returns
    ///
    /// A `KdTree<f32, 3>` containing all vertices. Items stored in the tree
    /// are vertex indices cast to `u64`.
    ///
    /// # Example
    ///
    /// ```
    /// # use nalgebra as na;
    /// # use oxigaf_flame::Mesh;
    /// let vertices = vec![
    ///     na::Point3::new(0.0f32, 0.0, 0.0),
    ///     na::Point3::new(1.0f32, 0.0, 0.0),
    /// ];
    /// let mesh = Mesh::new(vertices, vec![[0, 1, 0]]);
    /// let tree = mesh.build_kdtree();
    /// assert_eq!(tree.size(), 2);
    /// ```
    #[must_use]
    pub fn build_kdtree(&self) -> KdTree<f32, 3> {
        let mut tree: KdTree<f32, 3> = KdTree::with_capacity(self.vertices.len());
        for (i, vertex) in self.vertices.iter().enumerate() {
            tree.add(&[vertex.x, vertex.y, vertex.z], i as u64);
        }
        tree
    }

    /// Find the nearest vertex to `point`, building a temporary KD-tree.
    ///
    /// Returns `Some((vertex_index, squared_distance))` if the mesh has at
    /// least one vertex, `None` if the mesh is empty.
    ///
    /// **Note:** This method rebuilds the KD-tree on every call. For repeated
    /// queries, build the tree once with [`build_kdtree`](Self::build_kdtree)
    /// and call [`nearest_vertex_in_tree`](Self::nearest_vertex_in_tree).
    ///
    /// # Arguments
    ///
    /// * `point` - The 3D query point as `[x, y, z]`.
    ///
    /// # Returns
    ///
    /// `Some((vertex_index, squared_distance))` or `None` for empty meshes.
    #[must_use]
    pub fn nearest_vertex(&self, point: [f32; 3]) -> Option<(u32, f32)> {
        if self.vertices.is_empty() {
            return None;
        }
        let tree = self.build_kdtree();
        let result = tree.nearest_one::<SquaredEuclidean>(&point);
        // Vertex index fits in u32 (FLAME has at most ~5023 vertices)
        #[allow(clippy::cast_possible_truncation)]
        Some((result.item as u32, result.distance))
    }

    /// Find the nearest vertex using a pre-built KD-tree.
    ///
    /// This is the efficient variant for repeated queries: build the tree once
    /// with [`build_kdtree`](Self::build_kdtree), then call this for each query.
    ///
    /// # Arguments
    ///
    /// * `tree` - A KD-tree previously built from this mesh's vertices.
    /// * `point` - The 3D query point as `[x, y, z]`.
    ///
    /// # Returns
    ///
    /// `(vertex_index, squared_distance)` from the nearest vertex.
    ///
    /// # Panics
    ///
    /// Panics if the tree is empty (no vertices were added). Check
    /// `tree.size() > 0` before calling if emptiness is possible.
    #[must_use]
    pub fn nearest_vertex_in_tree(tree: &KdTree<f32, 3>, point: [f32; 3]) -> (u32, f32) {
        let result = tree.nearest_one::<SquaredEuclidean>(&point);
        #[allow(clippy::cast_possible_truncation)]
        (result.item as u32, result.distance)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple unit-square mesh for tests.
    ///
    /// Vertices at the four corners of a unit square in the XY plane:
    /// - 0: (0, 0, 0)
    /// - 1: (1, 0, 0)
    /// - 2: (0, 1, 0)
    /// - 3: (1, 1, 0)
    fn unit_square_mesh() -> Mesh {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0), // idx 0
            na::Point3::new(1.0f32, 0.0, 0.0), // idx 1
            na::Point3::new(0.0f32, 1.0, 0.0), // idx 2
            na::Point3::new(1.0f32, 1.0, 0.0), // idx 3
        ];
        let faces = vec![[0, 1, 2], [1, 3, 2]];
        Mesh::new(vertices, faces)
    }

    // --- build_kdtree --------------------------------------------------------

    #[test]
    fn test_build_kdtree_size_matches_vertex_count() {
        let mesh = unit_square_mesh();
        let tree = mesh.build_kdtree();
        assert_eq!(tree.size(), 4, "tree should have one entry per vertex");
    }

    #[test]
    fn test_build_kdtree_empty_mesh() {
        let mesh = Mesh::new(vec![], vec![]);
        let tree = mesh.build_kdtree();
        assert_eq!(tree.size(), 0, "empty mesh produces empty tree");
    }

    #[test]
    fn test_build_kdtree_single_vertex() {
        let vertices = vec![na::Point3::new(3.0f32, 4.0, 5.0)];
        let mesh = Mesh::new(vertices, vec![]);
        let tree = mesh.build_kdtree();
        assert_eq!(tree.size(), 1);
    }

    // --- nearest_vertex (rebuilds tree each time) ----------------------------

    #[test]
    fn test_nearest_vertex_returns_none_for_empty_mesh() {
        let mesh = Mesh::new(vec![], vec![]);
        let result = mesh.nearest_vertex([0.0, 0.0, 0.0]);
        assert!(result.is_none(), "empty mesh should return None");
    }

    #[test]
    fn test_nearest_vertex_exact_hit_on_vertex() {
        let mesh = unit_square_mesh();

        // Query exactly at vertex 2: (0, 1, 0)
        let (idx, dist_sq) = mesh
            .nearest_vertex([0.0, 1.0, 0.0])
            .expect("non-empty mesh should return Some");
        assert_eq!(idx, 2, "nearest to (0,1,0) should be vertex 2");
        assert!(dist_sq < 1e-10, "distance should be ~0 for exact hit");
    }

    #[test]
    fn test_nearest_vertex_closest_corner() {
        let mesh = unit_square_mesh();

        // Query near vertex 3: (1, 1, 0) — corner that is farthest from origin
        let (idx, dist_sq) = mesh
            .nearest_vertex([0.9, 0.95, 0.0])
            .expect("non-empty mesh should return Some");
        assert_eq!(
            idx, 3,
            "nearest to (0.9, 0.95, 0) should be vertex 3 at (1,1,0)"
        );
        // Squared distance = (0.1)^2 + (0.05)^2 ≈ 0.0125
        assert!(
            (dist_sq - 0.0125_f32).abs() < 1e-5,
            "dist_sq ≈ 0.0125, got {dist_sq}"
        );
    }

    #[test]
    fn test_nearest_vertex_out_of_plane() {
        let mesh = unit_square_mesh();

        // Query above vertex 0 in Z: (0, 0, 10)
        // All vertices are in Z=0, so nearest is (0, 0, 0) at dist_sq = 100
        let (idx, dist_sq) = mesh
            .nearest_vertex([0.0, 0.0, 10.0])
            .expect("should return Some");
        assert_eq!(idx, 0, "nearest vertex in Z should be at origin");
        assert!(
            (dist_sq - 100.0_f32).abs() < 1e-4,
            "dist_sq ≈ 100, got {dist_sq}"
        );
    }

    // --- nearest_vertex_in_tree (reuse pre-built tree) -----------------------

    #[test]
    fn test_nearest_vertex_in_tree_matches_nearest_vertex() {
        let mesh = unit_square_mesh();
        let tree = mesh.build_kdtree();

        let test_points: &[[f32; 3]] = &[
            [0.1, 0.1, 0.0],
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
            [0.9, 0.9, 0.0],
            [0.5, 0.5, 0.0],
        ];

        for &pt in test_points {
            let (idx_tree, dist_tree) = Mesh::nearest_vertex_in_tree(&tree, pt);
            let (idx_rebuild, dist_rebuild) = mesh.nearest_vertex(pt).expect("non-empty mesh");

            assert_eq!(
                idx_tree, idx_rebuild,
                "pre-built tree and rebuild should agree for point {pt:?}"
            );
            assert!(
                (dist_tree - dist_rebuild).abs() < 1e-9,
                "distances should be identical"
            );
        }
    }

    #[test]
    fn test_nearest_vertex_in_tree_multiple_queries_efficiency() {
        // Build tree once and run many queries — verifies the API works correctly
        let mesh = unit_square_mesh();
        let tree = mesh.build_kdtree();

        // 100 queries against the same tree
        for i in 0..100u32 {
            let t = i as f32 / 100.0;
            let pt = [t, t, 0.0];
            let (idx, dist_sq) = Mesh::nearest_vertex_in_tree(&tree, pt);
            // The result must be a valid vertex index
            assert!(
                (idx as usize) < mesh.vertices.len(),
                "returned vertex index {idx} out of bounds"
            );
            // Squared distance must be non-negative and finite
            assert!(dist_sq.is_finite() && dist_sq >= 0.0);
        }
    }

    #[test]
    fn test_build_kdtree_large_mesh() {
        // Create a 3D grid of 5x5x5 = 125 vertices.
        // All coordinates vary so no axis has many duplicate values,
        // which keeps kiddo's bucket-split algorithm happy.
        let mut vertices = Vec::with_capacity(125);
        for z in 0..5u32 {
            for y in 0..5u32 {
                for x in 0..5u32 {
                    vertices.push(na::Point3::new(x as f32, y as f32, z as f32));
                }
            }
        }
        let mesh = Mesh::new(vertices, vec![]);
        let tree = mesh.build_kdtree();
        assert_eq!(tree.size(), 125);

        // Query near vertex (4, 3, 2): index = 2*25 + 3*5 + 4 = 50+15+4 = 69
        let (idx, dist_sq) = Mesh::nearest_vertex_in_tree(&tree, [3.9, 3.1, 2.05]);
        assert_eq!(
            idx, 69,
            "nearest to (3.9, 3.1, 2.05) should be vertex 69 at (4,3,2)"
        );
        // dist_sq = 0.1^2 + 0.1^2 + 0.05^2 = 0.01 + 0.01 + 0.0025 = 0.0225
        assert!(
            (dist_sq - 0.0225_f32).abs() < 1e-4,
            "dist_sq ≈ 0.0225, got {dist_sq}"
        );
    }

    // -----------------------------------------------------------------------
    // Bounds safety: malformed faces must not panic
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_skips_out_of_range_faces_instead_of_panicking() {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0),
            na::Point3::new(1.0f32, 0.0, 0.0),
            na::Point3::new(0.0f32, 1.0, 0.0),
        ];
        // Face references vertex index 99, which does not exist.
        let faces = vec![[0, 1, 99]];
        let mesh = Mesh::new(vertices, faces);
        // Must not panic; normals stay at zero since the only face
        // touching these vertices was skipped as malformed.
        assert_eq!(mesh.normals.len(), 3);
        for n in &mesh.normals {
            assert!(n.x.is_finite() && n.y.is_finite() && n.z.is_finite());
        }
    }

    #[test]
    fn test_recompute_normals_mixed_valid_and_invalid_faces() {
        // A valid face and an out-of-range face in the same mesh: the
        // valid face's contribution must still be computed correctly.
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0),
            na::Point3::new(1.0f32, 0.0, 0.0),
            na::Point3::new(0.0f32, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2], [0, 1, 999]];
        let mesh = Mesh::new(vertices, faces);
        // Vertex 2's normal comes only from the valid face [0,1,2], which
        // lies in the XY plane, so its normal must point along +/-Z.
        let n2 = mesh.normals[2];
        assert!(n2.x.abs() < 1e-5 && n2.y.abs() < 1e-5);
        assert!((n2.z.abs() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_face_area_out_of_range_returns_zero_instead_of_panicking() {
        let mesh = unit_square_mesh();
        let area = mesh.face_area(&[0, 1, 99]);
        assert_eq!(
            area, 0.0,
            "out-of-range face should report zero area, not panic"
        );
    }

    #[test]
    fn test_face_area_valid_face_unchanged() {
        let mesh = unit_square_mesh();
        // Triangle (0,1,2) = (0,0,0),(1,0,0),(0,1,0): area = 0.5
        let area = mesh.face_area(&[0, 1, 2]);
        assert!((area - 0.5).abs() < 1e-6, "expected area 0.5, got {area}");
    }

    #[test]
    fn test_try_new_rejects_out_of_range_face() {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0),
            na::Point3::new(1.0f32, 0.0, 0.0),
        ];
        let faces = vec![[0, 1, 5]];
        let result = Mesh::try_new(vertices, faces);
        assert!(
            result.is_err(),
            "face referencing vertex 5 with only 2 vertices should be rejected"
        );
    }

    // -----------------------------------------------------------------------
    // vertex_mask_checked: error instead of a per-call tracing::warn
    // -----------------------------------------------------------------------

    /// A three-vertex stand-in with canonical FLAME head extents, centred on
    /// the origin (mirrors `vertex_mask.rs`'s private `canonical_head`).
    fn canonical_head_mesh() -> Mesh {
        let vertices = vec![
            na::Point3::new(-0.08f32, -0.15, -0.08),
            na::Point3::new(0.08f32, 0.15, 0.08),
            na::Point3::new(0.0f32, 0.0, 0.05),
        ];
        Mesh::new(vertices, vec![[0, 1, 2]])
    }

    #[test]
    fn test_vertex_mask_checked_accepts_canonical_mesh() {
        let mesh = canonical_head_mesh();
        let mask = mesh
            .vertex_mask_checked()
            .expect("canonical-extent mesh should pass the precondition");
        assert_eq!(mask.num_vertices, mesh.num_vertices());
    }

    #[test]
    fn test_vertex_mask_checked_rejects_non_canonical_mesh() {
        // A head translated up by 0.2 m: `vertex_mask()` only warns (once per
        // call — log spam for a per-frame caller), the checked variant reports.
        let canonical = canonical_head_mesh();
        let translated: Vec<na::Point3<f32>> = canonical
            .vertices
            .iter()
            .map(|p| na::Point3::new(p.x, p.y + 0.2, p.z))
            .collect();
        let posed = Mesh::new(translated, canonical.faces.clone());

        assert!(
            matches!(
                posed.vertex_mask_checked(),
                Err(crate::VertexMaskError::NonCanonicalPose { .. })
            ),
            "translated mesh must be rejected by the checked variant"
        );
        // The infallible variant still returns a mask, so existing callers keep
        // working.
        assert_eq!(posed.vertex_mask().num_vertices, posed.num_vertices());
    }

    #[test]
    fn test_vertex_mask_checked_matches_vertex_mask_on_canonical_mesh() {
        let mesh = canonical_head_mesh();
        let checked = mesh
            .vertex_mask_checked()
            .expect("canonical-extent mesh should pass the precondition");
        let unchecked = mesh.vertex_mask();
        assert_eq!(
            checked.regions, unchecked.regions,
            "checked variant must classify identically, only the failure mode differs"
        );
    }

    #[test]
    fn test_try_new_accepts_valid_mesh() {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0),
            na::Point3::new(1.0f32, 0.0, 0.0),
            na::Point3::new(0.0f32, 1.0, 0.0),
        ];
        let faces = vec![[0, 1, 2]];
        let mesh = Mesh::try_new(vertices, faces).expect("valid faces should be accepted");
        assert_eq!(mesh.num_vertices(), 3);
        assert_eq!(mesh.num_faces(), 1);
    }
}
