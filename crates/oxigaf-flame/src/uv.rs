//! UV coordinate support for FLAME meshes.
//!
//! This module provides:
//! - [`UvAccessor`]: read-only access and barycentric interpolation of UV coordinates
//! - [`UvMeshExt`]: builder and convenience methods for [`Mesh`]
//! - [`UvChartInfo`]: statistical analysis of a mesh's UV layout

use crate::{mesh::Mesh, FlameError};

// ---------------------------------------------------------------------------
// UvAccessor
// ---------------------------------------------------------------------------

/// Read-only UV coordinate accessor for a mesh.
///
/// Guarantees that the underlying mesh has a non-empty `uv_coords` array.
/// Construct via [`UvMeshExt::uv`] or [`UvAccessor::new`].
pub struct UvAccessor<'a> {
    mesh: &'a Mesh,
}

impl<'a> UvAccessor<'a> {
    /// Create a new `UvAccessor` for `mesh`.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if the mesh has no UV coordinates.
    pub fn new(mesh: &'a Mesh) -> Result<Self, FlameError> {
        if mesh.uv_coords.is_empty() {
            return Err(FlameError::InvalidParams(
                "Mesh has no UV coordinates".into(),
            ));
        }
        Ok(Self { mesh })
    }

    /// Get the UV coordinates for a vertex by index.
    ///
    /// Returns `None` if `vertex_idx` is out of range.
    #[inline]
    #[must_use]
    pub fn vertex_uv(&self, vertex_idx: usize) -> Option<[f32; 2]> {
        self.mesh.uv_coords.get(vertex_idx).copied()
    }

    /// Interpolate UV coordinates at a point on a face using barycentric coordinates.
    ///
    /// # Arguments
    ///
    /// * `face_idx` – which triangle (index into `mesh.faces`)
    /// * `bary` – barycentric weights `[u, v, w]` that should sum to approximately 1.0
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::IndexOutOfBounds`] if `face_idx` is out of range or any
    /// face vertex index references a UV coordinate that does not exist.
    pub fn interpolate_uv(&self, face_idx: usize, bary: [f32; 3]) -> Result<[f32; 2], FlameError> {
        let face = self
            .mesh
            .faces
            .get(face_idx)
            .ok_or_else(|| FlameError::IndexOutOfBounds {
                context: "UV interpolation face index".into(),
                index: face_idx,
                len: self.mesh.faces.len(),
            })?;

        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;

        let uv0 = self
            .mesh
            .uv_coords
            .get(i0)
            .ok_or_else(|| FlameError::IndexOutOfBounds {
                context: "UV interpolation vertex 0".into(),
                index: i0,
                len: self.mesh.uv_coords.len(),
            })?;
        let uv1 = self
            .mesh
            .uv_coords
            .get(i1)
            .ok_or_else(|| FlameError::IndexOutOfBounds {
                context: "UV interpolation vertex 1".into(),
                index: i1,
                len: self.mesh.uv_coords.len(),
            })?;
        let uv2 = self
            .mesh
            .uv_coords
            .get(i2)
            .ok_or_else(|| FlameError::IndexOutOfBounds {
                context: "UV interpolation vertex 2".into(),
                index: i2,
                len: self.mesh.uv_coords.len(),
            })?;

        let [u, v, w] = bary;
        Ok([
            u * uv0[0] + v * uv1[0] + w * uv2[0],
            u * uv0[1] + v * uv1[1] + w * uv2[1],
        ])
    }

    /// Compute UV coordinates for a batch of surface samples in a single call.
    ///
    /// # Arguments
    ///
    /// * `face_indices` – one face index per sample
    /// * `barycentrics` – one `[u, v, w]` triplet per sample
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if `face_indices.len() != barycentrics.len()`.
    /// Returns [`FlameError::IndexOutOfBounds`] if any face or UV index is out of range.
    pub fn sample_uvs(
        &self,
        face_indices: &[u32],
        barycentrics: &[[f32; 3]],
    ) -> Result<Vec<[f32; 2]>, FlameError> {
        if face_indices.len() != barycentrics.len() {
            return Err(FlameError::InvalidParams(format!(
                "face_indices length {} != barycentrics length {}",
                face_indices.len(),
                barycentrics.len()
            )));
        }

        let mut result = Vec::with_capacity(face_indices.len());
        for (i, (&face_idx, &bary)) in face_indices.iter().zip(barycentrics.iter()).enumerate() {
            let uv = self
                .interpolate_uv(face_idx as usize, bary)
                .map_err(|e| match e {
                    // Preserve the structured variant (and its index/len
                    // fields) so callers can match on it; just prefix the
                    // batch index onto the context string.
                    FlameError::IndexOutOfBounds {
                        context,
                        index,
                        len,
                    } => FlameError::IndexOutOfBounds {
                        context: format!("sample {i}: {context}"),
                        index,
                        len,
                    },
                    other => other,
                })?;
            result.push(uv);
        }
        Ok(result)
    }

    /// Check whether a UV coordinate lies within the valid `[0, 1]²` range.
    #[inline]
    #[must_use]
    pub fn is_valid_uv(uv: [f32; 2]) -> bool {
        uv[0] >= 0.0 && uv[0] <= 1.0 && uv[1] >= 0.0 && uv[1] <= 1.0
    }

    /// Number of UV coordinates stored in the mesh.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.mesh.uv_coords.len()
    }

    /// Whether the UV coordinate list is empty (should not be, given `new` validates).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mesh.uv_coords.is_empty()
    }
}

// ---------------------------------------------------------------------------
// UvMeshExt
// ---------------------------------------------------------------------------

/// Extension methods for building and querying UV coordinates on a [`Mesh`].
pub trait UvMeshExt {
    /// Attach UV coordinates to the mesh, consuming it.
    ///
    /// `uvs` must have length 0 (interpreted as "no UVs") or exactly
    /// `mesh.vertices.len()`.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if `uvs.len()` is non-zero and
    /// does not equal the vertex count.
    fn with_uv_coords(self, uvs: Vec<[f32; 2]>) -> Result<Mesh, FlameError>;

    /// Whether the mesh currently has UV coordinates.
    fn has_uv_coords(&self) -> bool;

    /// Create a [`UvAccessor`], validating that UV data is present.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if the mesh has no UV coordinates.
    fn uv(&self) -> Result<UvAccessor<'_>, FlameError>;

    /// Interpolate UV at a surface point.
    ///
    /// Convenience wrapper around `uv()?.interpolate_uv(face_idx, bary)`.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if the mesh has no UV coordinates,
    /// or [`FlameError::IndexOutOfBounds`] if the face or vertex indices are out of range.
    fn interpolate_uv_at(&self, face_idx: usize, bary: [f32; 3]) -> Result<[f32; 2], FlameError>;
}

impl UvMeshExt for Mesh {
    fn with_uv_coords(mut self, uvs: Vec<[f32; 2]>) -> Result<Mesh, FlameError> {
        if !uvs.is_empty() && uvs.len() != self.vertices.len() {
            return Err(FlameError::InvalidParams(format!(
                "UV count {} != vertex count {}",
                uvs.len(),
                self.vertices.len()
            )));
        }
        self.uv_coords = uvs;
        Ok(self)
    }

    #[inline]
    fn has_uv_coords(&self) -> bool {
        !self.uv_coords.is_empty()
    }

    fn uv(&self) -> Result<UvAccessor<'_>, FlameError> {
        UvAccessor::new(self)
    }

    fn interpolate_uv_at(&self, face_idx: usize, bary: [f32; 3]) -> Result<[f32; 2], FlameError> {
        self.uv()?.interpolate_uv(face_idx, bary)
    }
}

// ---------------------------------------------------------------------------
// UvChartInfo
// ---------------------------------------------------------------------------

/// Statistical summary of a mesh's UV layout.
#[derive(Debug, Clone)]
pub struct UvChartInfo {
    /// Minimum U and V values across all UV coordinates.
    pub min_uv: [f32; 2],
    /// Maximum U and V values across all UV coordinates.
    pub max_uv: [f32; 2],
    /// Fraction of `[0, 1]²` covered by UV triangles (may exceed 1.0 for overlapping charts).
    pub coverage_fraction: f32,
    /// Number of UV coordinates outside the `[0, 1]²` range.
    pub num_out_of_range: usize,
}

impl UvChartInfo {
    /// Compute the UV chart information for a mesh.
    ///
    /// Coverage is estimated by summing the 2-D area of each UV triangle using
    /// the shoelace formula. Overlapping triangles are double-counted.
    ///
    /// # Errors
    ///
    /// Returns [`FlameError::InvalidParams`] if the mesh has no UV coordinates.
    pub fn compute(mesh: &Mesh) -> Result<Self, FlameError> {
        if mesh.uv_coords.is_empty() {
            return Err(FlameError::InvalidParams(
                "Mesh has no UV coordinates".into(),
            ));
        }

        let uvs = &mesh.uv_coords;

        // Min / max and out-of-range count
        let mut min_uv = [f32::INFINITY; 2];
        let mut max_uv = [f32::NEG_INFINITY; 2];
        let mut num_out_of_range = 0usize;

        for &uv in uvs {
            if uv[0] < min_uv[0] {
                min_uv[0] = uv[0];
            }
            if uv[1] < min_uv[1] {
                min_uv[1] = uv[1];
            }
            if uv[0] > max_uv[0] {
                max_uv[0] = uv[0];
            }
            if uv[1] > max_uv[1] {
                max_uv[1] = uv[1];
            }
            if !UvAccessor::is_valid_uv(uv) {
                num_out_of_range += 1;
            }
        }

        // Coverage: sum of 2-D triangle areas in UV space via shoelace formula.
        // Each face contributes |u1*(v2-v3) + u2*(v3-v1) + u3*(v1-v2)| * 0.5
        let mut total_uv_area = 0.0f32;
        for face in &mesh.faces {
            let i0 = face[0] as usize;
            let i1 = face[1] as usize;
            let i2 = face[2] as usize;

            // If any index is out of range, skip (shouldn't happen for a valid mesh)
            if i0 >= uvs.len() || i1 >= uvs.len() || i2 >= uvs.len() {
                continue;
            }

            let [u0, v0] = uvs[i0];
            let [u1, v1] = uvs[i1];
            let [u2, v2] = uvs[i2];

            let area = (u0 * (v1 - v2) + u1 * (v2 - v0) + u2 * (v0 - v1)).abs() * 0.5;
            total_uv_area += area;
        }

        Ok(Self {
            min_uv,
            max_uv,
            coverage_fraction: total_uv_area,
            num_out_of_range,
        })
    }

    /// Whether all UV coordinates lie within the `[0, 1]²` range.
    #[inline]
    #[must_use]
    pub fn is_normalized(&self) -> bool {
        self.min_uv[0] >= 0.0
            && self.min_uv[1] >= 0.0
            && self.max_uv[0] <= 1.0
            && self.max_uv[1] <= 1.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// A single unit triangle with matching UV coordinates.
    ///
    /// Vertices: (0,0,0), (1,0,0), (0,1,0)
    /// UVs:      [0,0],   [1,0],   [0,1]
    fn triangle_mesh_with_uvs() -> Mesh {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0),
            na::Point3::new(1.0f32, 0.0, 0.0),
            na::Point3::new(0.0f32, 1.0, 0.0),
        ];
        let faces = vec![[0u32, 1, 2]];
        let uvs: Vec<[f32; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        Mesh::new(vertices, faces)
            .with_uv_coords(uvs)
            .expect("UV count matches vertex count")
    }

    /// A unit-square mesh (4 vertices, 2 triangles) without UV coordinates.
    fn mesh_no_uvs() -> Mesh {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0),
            na::Point3::new(1.0f32, 0.0, 0.0),
            na::Point3::new(0.0f32, 1.0, 0.0),
            na::Point3::new(1.0f32, 1.0, 0.0),
        ];
        Mesh::new(vertices, vec![[0, 1, 2], [1, 3, 2]])
    }

    // -----------------------------------------------------------------------
    // has_uv_coords
    // -----------------------------------------------------------------------

    #[test]
    fn test_fresh_mesh_has_no_uv_coords() {
        let mesh = mesh_no_uvs();
        assert!(
            !mesh.has_uv_coords(),
            "Mesh::new should leave uv_coords empty"
        );
    }

    // -----------------------------------------------------------------------
    // with_uv_coords
    // -----------------------------------------------------------------------

    #[test]
    fn test_with_uv_coords_correct_count_ok() {
        let mesh = mesh_no_uvs();
        let uvs: Vec<[f32; 2]> = vec![[0.0, 0.0]; 4];
        let result = mesh.with_uv_coords(uvs);
        assert!(result.is_ok(), "same count should succeed");
        assert!(result.expect("ok").has_uv_coords());
    }

    #[test]
    fn test_with_uv_coords_wrong_count_error() {
        let mesh = mesh_no_uvs(); // 4 vertices
        let uvs: Vec<[f32; 2]> = vec![[0.0, 0.0]; 3]; // wrong count
        let result = mesh.with_uv_coords(uvs);
        assert!(result.is_err(), "mismatched count should return Err");
    }

    #[test]
    fn test_with_uv_coords_empty_clears() {
        let mesh = triangle_mesh_with_uvs(); // has UVs
        let result = mesh.with_uv_coords(Vec::new());
        assert!(result.is_ok());
        assert!(
            !result.expect("ok").has_uv_coords(),
            "empty vec should clear UVs"
        );
    }

    // -----------------------------------------------------------------------
    // uv() accessor — error on missing UVs
    // -----------------------------------------------------------------------

    #[test]
    fn test_uv_accessor_on_mesh_without_uvs_errors() {
        let mesh = mesh_no_uvs();
        let result = mesh.uv();
        assert!(result.is_err(), "uv() on mesh without UVs should fail");
    }

    // -----------------------------------------------------------------------
    // vertex_uv
    // -----------------------------------------------------------------------

    #[test]
    fn test_vertex_uv_returns_first_uv() {
        let mesh = triangle_mesh_with_uvs();
        let accessor = mesh.uv().expect("mesh has UVs");
        let uv = accessor.vertex_uv(0);
        assert_eq!(uv, Some([0.0, 0.0]), "vertex 0 UV should be [0, 0]");
    }

    #[test]
    fn test_vertex_uv_out_of_bounds_returns_none() {
        let mesh = triangle_mesh_with_uvs();
        let accessor = mesh.uv().expect("mesh has UVs");
        assert!(
            accessor.vertex_uv(999).is_none(),
            "out-of-range index should return None"
        );
    }

    // -----------------------------------------------------------------------
    // interpolate_uv
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpolate_uv_bary_1_0_0_gives_vertex_0() {
        let mesh = triangle_mesh_with_uvs();
        let accessor = mesh.uv().expect("mesh has UVs");
        let uv = accessor
            .interpolate_uv(0, [1.0, 0.0, 0.0])
            .expect("valid face");
        assert!(
            (uv[0] - 0.0).abs() < 1e-6 && (uv[1] - 0.0).abs() < 1e-6,
            "bary [1,0,0] → UV of vertex 0 = [0,0], got {uv:?}"
        );
    }

    #[test]
    fn test_interpolate_uv_bary_0_1_0_gives_vertex_1() {
        let mesh = triangle_mesh_with_uvs();
        let accessor = mesh.uv().expect("mesh has UVs");
        let uv = accessor
            .interpolate_uv(0, [0.0, 1.0, 0.0])
            .expect("valid face");
        assert!(
            (uv[0] - 1.0).abs() < 1e-6 && (uv[1] - 0.0).abs() < 1e-6,
            "bary [0,1,0] → UV of vertex 1 = [1,0], got {uv:?}"
        );
    }

    #[test]
    fn test_interpolate_uv_bary_0_0_1_gives_vertex_2() {
        let mesh = triangle_mesh_with_uvs();
        let accessor = mesh.uv().expect("mesh has UVs");
        let uv = accessor
            .interpolate_uv(0, [0.0, 0.0, 1.0])
            .expect("valid face");
        assert!(
            (uv[0] - 0.0).abs() < 1e-6 && (uv[1] - 1.0).abs() < 1e-6,
            "bary [0,0,1] → UV of vertex 2 = [0,1], got {uv:?}"
        );
    }

    #[test]
    fn test_interpolate_uv_centroid_bary() {
        // Bary [1/3, 1/3, 1/3] → centroid of UVs [0,0], [1,0], [0,1]
        // Expected: [(0+1+0)/3, (0+0+1)/3] = [1/3, 1/3]
        let mesh = triangle_mesh_with_uvs();
        let accessor = mesh.uv().expect("mesh has UVs");
        let third = 1.0f32 / 3.0;
        let uv = accessor
            .interpolate_uv(0, [third, third, third])
            .expect("valid face");
        assert!(
            (uv[0] - third).abs() < 1e-5 && (uv[1] - third).abs() < 1e-5,
            "centroid bary → [1/3, 1/3], got {uv:?}"
        );
    }

    #[test]
    fn test_interpolate_uv_bad_face_idx_errors() {
        let mesh = triangle_mesh_with_uvs();
        let accessor = mesh.uv().expect("mesh has UVs");
        let result = accessor.interpolate_uv(999, [1.0, 0.0, 0.0]);
        assert!(result.is_err(), "out-of-range face index should be Err");
    }

    // -----------------------------------------------------------------------
    // is_valid_uv
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_valid_uv_center() {
        assert!(UvAccessor::is_valid_uv([0.5, 0.5]), "[0.5, 0.5] is valid");
    }

    #[test]
    fn test_is_valid_uv_corners() {
        assert!(UvAccessor::is_valid_uv([0.0, 0.0]));
        assert!(UvAccessor::is_valid_uv([1.0, 1.0]));
        assert!(UvAccessor::is_valid_uv([0.0, 1.0]));
        assert!(UvAccessor::is_valid_uv([1.0, 0.0]));
    }

    #[test]
    fn test_is_valid_uv_negative_u() {
        assert!(
            !UvAccessor::is_valid_uv([-0.1, 0.5]),
            "negative U is invalid"
        );
    }

    #[test]
    fn test_is_valid_uv_above_one() {
        assert!(!UvAccessor::is_valid_uv([1.1, 0.5]), "U > 1.0 is invalid");
        assert!(!UvAccessor::is_valid_uv([0.5, 1.1]), "V > 1.0 is invalid");
    }

    // -----------------------------------------------------------------------
    // sample_uvs
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_uvs_matching_lengths_ok() {
        let mesh = triangle_mesh_with_uvs();
        let accessor = mesh.uv().expect("mesh has UVs");
        let face_indices = vec![0u32, 0, 0];
        let barycentrics: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let result = accessor.sample_uvs(&face_indices, &barycentrics);
        assert!(result.is_ok(), "matching lengths should succeed");
        let uvs = result.expect("ok");
        assert_eq!(uvs.len(), 3);
        assert!((uvs[0][0] - 0.0).abs() < 1e-6); // vertex 0 → [0,0]
        assert!((uvs[1][0] - 1.0).abs() < 1e-6); // vertex 1 → [1,0]
        assert!((uvs[2][1] - 1.0).abs() < 1e-6); // vertex 2 → [0,1]
    }

    #[test]
    fn test_sample_uvs_length_mismatch_errors() {
        let mesh = triangle_mesh_with_uvs();
        let accessor = mesh.uv().expect("mesh has UVs");
        let face_indices = vec![0u32, 0]; // 2
        let barycentrics: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0]]; // 1
        let result = accessor.sample_uvs(&face_indices, &barycentrics);
        assert!(result.is_err(), "mismatched lengths should fail");
    }

    #[test]
    fn test_sample_uvs_bad_face_idx_preserves_index_out_of_bounds_variant() {
        // A caller matching on `FlameError::IndexOutOfBounds { index, .. }`
        // must be able to recover the offending index; `sample_uvs` must not
        // flatten the structured error into `InvalidParams(String)`.
        let mesh = triangle_mesh_with_uvs();
        let accessor = mesh.uv().expect("mesh has UVs");
        let face_indices = vec![0u32, 999];
        let barycentrics: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let result = accessor.sample_uvs(&face_indices, &barycentrics);
        match result {
            Err(FlameError::IndexOutOfBounds { index, len, .. }) => {
                assert_eq!(index, 999, "should report the offending face index");
                assert_eq!(len, mesh.faces.len());
            }
            other => panic!("expected FlameError::IndexOutOfBounds, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // UvChartInfo
    // -----------------------------------------------------------------------

    #[test]
    fn test_uv_chart_info_compute_ok() {
        let mesh = triangle_mesh_with_uvs();
        let info = UvChartInfo::compute(&mesh);
        assert!(info.is_ok(), "should succeed for mesh with UVs");
    }

    #[test]
    fn test_uv_chart_info_on_mesh_without_uvs_errors() {
        let mesh = mesh_no_uvs();
        let result = UvChartInfo::compute(&mesh);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_normalized_true_for_unit_range_uvs() {
        let mesh = triangle_mesh_with_uvs(); // UVs in [0,1]²
        let info = UvChartInfo::compute(&mesh).expect("ok");
        assert!(info.is_normalized(), "UVs in [0,1]² should be normalized");
    }

    #[test]
    fn test_is_normalized_false_for_out_of_range_uvs() {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.0, 0.0),
            na::Point3::new(1.0f32, 0.0, 0.0),
            na::Point3::new(0.0f32, 1.0, 0.0),
        ];
        let uvs: Vec<[f32; 2]> = vec![[0.0, 0.0], [1.5, 0.0], [0.0, 1.0]]; // 1.5 > 1
        let mesh = Mesh::new(vertices, vec![[0, 1, 2]])
            .with_uv_coords(uvs)
            .expect("ok");
        let info = UvChartInfo::compute(&mesh).expect("ok");
        assert!(
            !info.is_normalized(),
            "UV with U=1.5 should not be normalized"
        );
        assert!(info.num_out_of_range > 0, "should count out-of-range UVs");
    }

    // -----------------------------------------------------------------------
    // interpolate_uv_at convenience method
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpolate_uv_at_convenience() {
        let mesh = triangle_mesh_with_uvs();
        let uv = mesh
            .interpolate_uv_at(0, [1.0, 0.0, 0.0])
            .expect("should work for mesh with UVs");
        assert!(
            (uv[0] - 0.0).abs() < 1e-6 && (uv[1] - 0.0).abs() < 1e-6,
            "convenience method should match direct accessor"
        );
    }

    #[test]
    fn test_interpolate_uv_at_on_mesh_without_uvs_errors() {
        let mesh = mesh_no_uvs();
        let result = mesh.interpolate_uv_at(0, [1.0, 0.0, 0.0]);
        assert!(result.is_err(), "should error when mesh has no UVs");
    }

    // -----------------------------------------------------------------------
    // Accessor len / is_empty
    // -----------------------------------------------------------------------

    #[test]
    fn test_accessor_len_matches_uv_count() {
        let mesh = triangle_mesh_with_uvs(); // 3 UVs
        let accessor = mesh.uv().expect("ok");
        assert_eq!(accessor.len(), 3);
        assert!(!accessor.is_empty());
    }

    // -----------------------------------------------------------------------
    // Coverage fraction sanity
    // -----------------------------------------------------------------------

    #[test]
    fn test_coverage_fraction_unit_triangle() {
        // A right-isosceles triangle with UV corners at (0,0), (1,0), (0,1)
        // has area 0.5, which is exactly 50% of [0,1]²
        let mesh = triangle_mesh_with_uvs();
        let info = UvChartInfo::compute(&mesh).expect("ok");
        assert!(
            (info.coverage_fraction - 0.5).abs() < 1e-5,
            "unit right triangle UV coverage should be 0.5, got {}",
            info.coverage_fraction
        );
    }
}
