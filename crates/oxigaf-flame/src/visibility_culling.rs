//! Per-vertex and per-face visibility culling for FLAME meshes.
//!
//! Computes which vertices and faces are visible (front-facing, inside the
//! camera frustum, not occluded) from a given camera viewpoint. Useful for
//! the training pipeline to determine which parts of a head avatar are
//! observable from each training view.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxigaf_flame::{Mesh, normal_map::Camera};
//! use oxigaf_flame::visibility_culling::{
//!     VisibilityCullerConfig, compute_vertex_visibility, compute_visibility_stats,
//!     format_visibility_stats,
//! };
//!
//! # fn example(mesh: &Mesh, camera: &Camera) -> Result<(), oxigaf_flame::visibility_culling::VisibilityError> {
//! let config = VisibilityCullerConfig::default();
//! let vis = compute_vertex_visibility(mesh, camera, &config)?;
//! let stats = compute_visibility_stats(&vis);
//! println!("{}", format_visibility_stats(&stats));
//! # Ok(()) }
//! ```

use nalgebra as na;
use thiserror::Error;

use crate::mesh::Mesh;
use crate::normal_map::Camera;

// ---------------------------------------------------------------------------
// Error Type
// ---------------------------------------------------------------------------

/// Errors produced by visibility culling functions.
#[derive(Debug, Error)]
pub enum VisibilityError {
    /// The mesh has no vertices.
    #[error("Empty mesh: no vertices")]
    EmptyMesh,

    /// The mesh has no faces.
    #[error("Empty mesh: no faces")]
    NoFaces,

    /// A face references a vertex index beyond the end of the vertex buffer.
    #[error("Vertex index {idx} out of range (n_vertices = {n})")]
    VertexIndexOutOfRange { idx: usize, n: usize },

    /// At least one camera is required for multi-view visibility.
    #[error("Camera lists are empty: cannot compute multi-view visibility")]
    NoCameras,

    /// The normal buffer length does not match the vertex count.
    #[error("Normal buffer length {normals} does not match vertex count {vertices}")]
    NormalCountMismatch { normals: usize, vertices: usize },
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for visibility computation.
#[derive(Debug, Clone)]
pub struct VisibilityCullerConfig {
    /// Dot-product threshold for backface culling.
    ///
    /// A face or vertex is considered front-facing when
    /// `dot(normal, view_dir) > backface_threshold`.  The default value `0.0`
    /// accepts exactly the 90° silhouette. Increase toward `1.0` to cull
    /// more aggressively; decrease below `0.0` to accept some back-faces.
    pub backface_threshold: f32,

    /// Extra pixel margin beyond the image boundary for frustum testing.
    ///
    /// A vertex is in-frustum when its screen-space coordinates lie within
    /// `[−margin, width+margin) × [−margin, height+margin)`. The default is
    /// `0.0` (strict image boundary).
    pub frustum_margin: f32,

    /// Whether to perform an approximate depth-occlusion test.
    ///
    /// Currently reserved; set to `false` (the default). When `true`, a
    /// future implementation will use a per-pixel depth buffer to detect
    /// self-occluded vertices.
    pub use_depth_test: bool,

    /// Depth bias used to break z-fighting ties in the depth test.
    pub depth_bias: f32,
}

impl Default for VisibilityCullerConfig {
    fn default() -> Self {
        Self {
            backface_threshold: 0.0,
            frustum_margin: 0.0,
            use_depth_test: false,
            depth_bias: 1e-4,
        }
    }
}

// ---------------------------------------------------------------------------
// Result Structures
// ---------------------------------------------------------------------------

/// Per-vertex visibility from a single camera viewpoint.
#[derive(Debug, Clone)]
pub struct VertexVisibility {
    /// `true` = vertex is visible from this camera (in-frustum AND front-facing).
    pub visible: Vec<bool>,
    /// `true` = vertex projects within the camera image plane (plus any margin).
    pub in_frustum: Vec<bool>,
    /// `true` = the per-vertex normal has a positive dot product with the view
    /// direction (i.e. faces toward the camera).
    pub front_facing: Vec<bool>,
    /// Total number of vertices.
    pub n_vertices: usize,
}

/// Per-face visibility from a single camera viewpoint.
#[derive(Debug, Clone)]
pub struct FaceVisibility {
    /// `true` = face normal points toward the camera.
    pub visible: Vec<bool>,
    /// Projected screen-space area of each face in pixels².
    ///
    /// Zero when any triangle vertex is behind the near plane.
    pub screen_area: Vec<f32>,
    /// Total number of faces.
    pub n_faces: usize,
}

/// Summary statistics about vertex visibility from a single viewpoint.
#[derive(Debug, Clone)]
pub struct VisibilityStats {
    /// Total number of vertices in the mesh.
    pub n_vertices: usize,
    /// Number of vertices marked visible (in-frustum AND front-facing).
    pub n_visible_vertices: usize,
    /// Number of vertices whose normal faces the camera.
    pub n_front_facing: usize,
    /// Number of vertices whose projection falls inside the image.
    pub n_in_frustum: usize,
    /// Fraction of vertices that are visible (`n_visible_vertices / n_vertices`).
    pub visible_fraction: f32,
    /// Fraction of vertices that are front-facing.
    pub front_facing_fraction: f32,
    /// Fraction of vertices that are in-frustum.
    pub in_frustum_fraction: f32,
}

/// Aggregate visibility across multiple camera views.
#[derive(Debug, Clone)]
pub struct MultiViewVisibility {
    /// `true` if the vertex is visible from at least one camera.
    pub any_visible: Vec<bool>,
    /// `true` if the vertex is visible from every camera.
    pub all_visible: Vec<bool>,
    /// Number of cameras from which each vertex is visible.
    pub view_count: Vec<usize>,
    /// Total number of vertices.
    pub n_vertices: usize,
    /// Total number of cameras.
    pub n_cameras: usize,
}

// ---------------------------------------------------------------------------
// Private Helpers
// ---------------------------------------------------------------------------

/// Compute the unit face normal from three vertex positions.
///
/// Uses the cross product of the two edge vectors; returns `[0, 0, 0]` for
/// degenerate (zero-area) triangles.
#[inline]
fn compute_face_normal(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> [f32; 3] {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

    // Cross product e1 × e2
    let nx = e1[1] * e2[2] - e1[2] * e2[1];
    let ny = e1[2] * e2[0] - e1[0] * e2[2];
    let nz = e1[0] * e2[1] - e1[1] * e2[0];

    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len < 1e-10 {
        [0.0, 0.0, 0.0]
    } else {
        [nx / len, ny / len, nz / len]
    }
}

/// Compute the world-space direction from a mesh vertex toward the camera origin.
///
/// The camera world position satisfies `p_cam_world = −R^T t` (where `R` and
/// `t` are the world-to-camera rotation and translation).  The returned vector
/// is normalized.  Returns `[0, 0, 0]` if the vertex coincides with the camera.
#[inline]
fn camera_direction(vertex: [f32; 3], camera: &Camera) -> [f32; 3] {
    // Camera world position: p = -R^T * t
    let rt = camera.rotation.transpose();
    let neg_t = -camera.translation;
    let cam_world = rt * neg_t;

    let dx = cam_world[0] - vertex[0];
    let dy = cam_world[1] - vertex[1];
    let dz = cam_world[2] - vertex[2];

    let len = (dx * dx + dy * dy + dz * dz).sqrt();
    if len < 1e-10 {
        [0.0, 0.0, 0.0]
    } else {
        [dx / len, dy / len, dz / len]
    }
}

/// Return `true` when `dot(face_normal, view_dir) > threshold`.
#[inline]
fn is_front_facing(face_normal: [f32; 3], view_dir: [f32; 3], threshold: f32) -> bool {
    let dot =
        face_normal[0] * view_dir[0] + face_normal[1] * view_dir[1] + face_normal[2] * view_dir[2];
    dot > threshold
}

/// Project a world-space vertex onto screen coordinates using the pinhole model.
///
/// Returns `None` if the vertex is at or behind the near clipping plane.
#[inline]
fn project_vertex(vertex: [f32; 3], camera: &Camera) -> Option<[f32; 2]> {
    let p = na::Point3::new(vertex[0], vertex[1], vertex[2]);
    let p_cam = camera.world_to_cam(&p);

    if p_cam.z <= camera.near {
        return None;
    }

    let screen_x = camera.focal_x * p_cam.x / p_cam.z + camera.cx;
    let screen_y = camera.focal_y * p_cam.y / p_cam.z + camera.cy;
    Some([screen_x, screen_y])
}

/// Test whether a screen-space position lies within the image bounds.
///
/// An additional pixel `margin` is allowed beyond each edge.
#[inline]
fn is_in_frustum(screen_pos: [f32; 2], camera: &Camera, margin: f32) -> bool {
    let w = camera.width as f32;
    let h = camera.height as f32;

    screen_pos[0] >= -margin
        && screen_pos[0] < w + margin
        && screen_pos[1] >= -margin
        && screen_pos[1] < h + margin
}

/// Compute the signed screen-space area of a triangle (shoelace formula).
///
/// Returns the **absolute** area in pixels².  Zero for degenerate triangles.
#[inline]
fn compute_face_screen_area(s0: [f32; 2], s1: [f32; 2], s2: [f32; 2]) -> f32 {
    let area = (s1[0] - s0[0]) * (s2[1] - s0[1]) - (s2[0] - s0[0]) * (s1[1] - s0[1]);
    (area * 0.5).abs()
}

// ---------------------------------------------------------------------------
// Core Public API
// ---------------------------------------------------------------------------

/// Compute per-face visibility from a single camera viewpoint.
///
/// For every face the function computes:
/// - Whether the face normal points toward the camera (front-facing test).
/// - The projected screen-space area of the triangle.
///
/// # Errors
///
/// Returns [`VisibilityError::NoFaces`] for meshes with no triangles, or
/// [`VisibilityError::VertexIndexOutOfRange`] if a face references an invalid
/// vertex index.
pub fn compute_face_visibility(
    mesh: &Mesh,
    camera: &Camera,
    config: &VisibilityCullerConfig,
) -> Result<FaceVisibility, VisibilityError> {
    if mesh.faces.is_empty() {
        return Err(VisibilityError::NoFaces);
    }

    let n_faces = mesh.faces.len();
    let n_verts = mesh.vertices.len();
    let mut visible = vec![false; n_faces];
    let mut screen_area = vec![0.0_f32; n_faces];

    for (face_idx, face) in mesh.faces.iter().enumerate() {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;

        // Validate indices
        if i0 >= n_verts {
            return Err(VisibilityError::VertexIndexOutOfRange {
                idx: i0,
                n: n_verts,
            });
        }
        if i1 >= n_verts {
            return Err(VisibilityError::VertexIndexOutOfRange {
                idx: i1,
                n: n_verts,
            });
        }
        if i2 >= n_verts {
            return Err(VisibilityError::VertexIndexOutOfRange {
                idx: i2,
                n: n_verts,
            });
        }

        let v0 = mesh.vertices[i0];
        let v1 = mesh.vertices[i1];
        let v2 = mesh.vertices[i2];

        let v0a = [v0.x, v0.y, v0.z];
        let v1a = [v1.x, v1.y, v1.z];
        let v2a = [v2.x, v2.y, v2.z];

        // Face normal from cross product of edges (world space)
        let face_normal = compute_face_normal(v0a, v1a, v2a);

        // View direction from face centroid toward camera
        let centroid = [
            (v0.x + v1.x + v2.x) / 3.0,
            (v0.y + v1.y + v2.y) / 3.0,
            (v0.z + v1.z + v2.z) / 3.0,
        ];
        let view_dir = camera_direction(centroid, camera);

        visible[face_idx] = is_front_facing(face_normal, view_dir, config.backface_threshold);

        // Projected screen-space area
        let s0 = project_vertex(v0a, camera);
        let s1 = project_vertex(v1a, camera);
        let s2 = project_vertex(v2a, camera);

        screen_area[face_idx] = match (s0, s1, s2) {
            (Some(a), Some(b), Some(c)) => compute_face_screen_area(a, b, c),
            _ => 0.0,
        };
    }

    Ok(FaceVisibility {
        visible,
        screen_area,
        n_faces,
    })
}

/// Compute per-vertex visibility from a single camera viewpoint.
///
/// A vertex is considered **visible** when it is both inside the camera frustum
/// and has a front-facing per-vertex normal.
///
/// # Errors
///
/// Returns [`VisibilityError::EmptyMesh`] for meshes with no vertices, or
/// [`VisibilityError::NormalCountMismatch`] when the normal buffer length
/// differs from the vertex count.
pub fn compute_vertex_visibility(
    mesh: &Mesh,
    camera: &Camera,
    config: &VisibilityCullerConfig,
) -> Result<VertexVisibility, VisibilityError> {
    if mesh.vertices.is_empty() {
        return Err(VisibilityError::EmptyMesh);
    }
    if mesh.normals.len() != mesh.vertices.len() {
        return Err(VisibilityError::NormalCountMismatch {
            normals: mesh.normals.len(),
            vertices: mesh.vertices.len(),
        });
    }

    let n_vertices = mesh.vertices.len();
    let mut in_frustum_buf = vec![false; n_vertices];
    let mut front_facing_buf = vec![false; n_vertices];
    let mut visible_buf = vec![false; n_vertices];

    for (i, (vertex, normal)) in mesh.vertices.iter().zip(mesh.normals.iter()).enumerate() {
        let va = [vertex.x, vertex.y, vertex.z];

        // Frustum test: project vertex and check screen bounds
        let in_f = project_vertex(va, camera)
            .is_some_and(|sp| is_in_frustum(sp, camera, config.frustum_margin));
        in_frustum_buf[i] = in_f;

        // Front-facing test using per-vertex normal
        let view_dir = camera_direction(va, camera);
        let na_arr = [normal.x, normal.y, normal.z];
        let ff = is_front_facing(na_arr, view_dir, config.backface_threshold);
        front_facing_buf[i] = ff;

        visible_buf[i] = in_f && ff;
    }

    Ok(VertexVisibility {
        visible: visible_buf,
        in_frustum: in_frustum_buf,
        front_facing: front_facing_buf,
        n_vertices,
    })
}

/// Compute summary statistics for a [`VertexVisibility`] result.
///
/// All fractions are in `[0.0, 1.0]`. For an empty mesh every fraction is
/// `0.0`.
#[must_use]
pub fn compute_visibility_stats(visibility: &VertexVisibility) -> VisibilityStats {
    let n = visibility.n_vertices;

    let n_visible = visibility.visible.iter().filter(|&&v| v).count();
    let n_front = visibility.front_facing.iter().filter(|&&v| v).count();
    let n_frustum = visibility.in_frustum.iter().filter(|&&v| v).count();

    let (vis_frac, ff_frac, inf_frac) = if n == 0 {
        (0.0, 0.0, 0.0)
    } else {
        let nf = n as f32;
        (
            n_visible as f32 / nf,
            n_front as f32 / nf,
            n_frustum as f32 / nf,
        )
    };

    VisibilityStats {
        n_vertices: n,
        n_visible_vertices: n_visible,
        n_front_facing: n_front,
        n_in_frustum: n_frustum,
        visible_fraction: vis_frac,
        front_facing_fraction: ff_frac,
        in_frustum_fraction: inf_frac,
    }
}

/// Aggregate vertex visibility across multiple camera viewpoints.
///
/// Runs [`compute_vertex_visibility`] for each camera and accumulates results.
///
/// # Errors
///
/// Returns [`VisibilityError::NoCameras`] when `cameras` is empty, or
/// propagates any error from [`compute_vertex_visibility`].
pub fn compute_multi_view_visibility(
    mesh: &Mesh,
    cameras: &[Camera],
    config: &VisibilityCullerConfig,
) -> Result<MultiViewVisibility, VisibilityError> {
    if cameras.is_empty() {
        return Err(VisibilityError::NoCameras);
    }

    let n_vertices = mesh.vertices.len();
    let n_cameras = cameras.len();

    let mut any_visible = vec![false; n_vertices];
    let mut view_count = vec![0usize; n_vertices];

    for camera in cameras {
        let vis = compute_vertex_visibility(mesh, camera, config)?;
        for (i, &v) in vis.visible.iter().enumerate() {
            if v {
                any_visible[i] = true;
                view_count[i] += 1;
            }
        }
    }

    let all_visible: Vec<bool> = view_count.iter().map(|&c| c == n_cameras).collect();

    Ok(MultiViewVisibility {
        any_visible,
        all_visible,
        view_count,
        n_vertices,
        n_cameras,
    })
}

/// Find vertices that are visible from some but not all cameras.
///
/// Returns the indices of vertices where `any_visible[i] == true` but
/// `all_visible[i] == false`.  These are "view-dependent" vertices that
/// require more care during training because they are only partially observed.
#[must_use]
pub fn find_view_dependent_vertices(multi_view: &MultiViewVisibility) -> Vec<usize> {
    multi_view
        .any_visible
        .iter()
        .zip(multi_view.all_visible.iter())
        .enumerate()
        .filter_map(|(i, (&any, &all))| if any && !all { Some(i) } else { None })
        .collect()
}

/// Compute per-camera coverage: fraction of mesh vertices visible from each camera.
///
/// Returns a `Vec<f32>` of length `cameras.len()`, where each entry is the
/// fraction of vertices visible from the corresponding camera.
///
/// # Errors
///
/// Returns [`VisibilityError::NoCameras`] when `cameras` is empty, or any
/// error from [`compute_vertex_visibility`].
pub fn compute_optimal_view_coverage(
    mesh: &Mesh,
    cameras: &[Camera],
    config: &VisibilityCullerConfig,
) -> Result<Vec<f32>, VisibilityError> {
    if cameras.is_empty() {
        return Err(VisibilityError::NoCameras);
    }

    let n_vertices = mesh.vertices.len();
    let mut coverage = Vec::with_capacity(cameras.len());

    for camera in cameras {
        let vis = compute_vertex_visibility(mesh, camera, config)?;
        let visible_count = vis.visible.iter().filter(|&&v| v).count();
        let frac = if n_vertices == 0 {
            0.0
        } else {
            visible_count as f32 / n_vertices as f32
        };
        coverage.push(frac);
    }

    Ok(coverage)
}

/// Greedily select the `k` cameras with the highest coverage fractions.
///
/// Returns camera indices sorted by coverage descending (highest coverage
/// first). If `k` exceeds the number of cameras the full list is returned.
#[must_use]
pub fn select_maximally_covering_views(coverage: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> =
        coverage.iter().enumerate().map(|(i, &c)| (i, c)).collect();

    // Sort descending by coverage; break ties by index (ascending) for determinism
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    indexed.truncate(k);
    indexed.into_iter().map(|(i, _)| i).collect()
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Format a [`VisibilityStats`] report as a human-readable one-liner.
///
/// Example output:
/// `"Visibility: 3421/5023 vertices visible (68.1%), 4200 front-facing (83.6%), 3800 in frustum (75.7%)"`
#[must_use]
pub fn format_visibility_stats(stats: &VisibilityStats) -> String {
    format!(
        "Visibility: {}/{} vertices visible ({:.1}%), {} front-facing ({:.1}%), {} in frustum ({:.1}%)",
        stats.n_visible_vertices,
        stats.n_vertices,
        stats.visible_fraction * 100.0,
        stats.n_front_facing,
        stats.front_facing_fraction * 100.0,
        stats.n_in_frustum,
        stats.in_frustum_fraction * 100.0,
    )
}

/// Format a [`MultiViewVisibility`] summary as a human-readable one-liner.
///
/// Example output:
/// `"MultiView[4 cams]: any_visible=4890 (97.4%), all_visible=1234 (24.6%), view-dependent=3656 (72.8%)"`
#[must_use]
pub fn format_multi_view_stats(mv: &MultiViewVisibility) -> String {
    let n = mv.n_vertices as f32;
    let any_count = mv.any_visible.iter().filter(|&&v| v).count();
    let all_count = mv.all_visible.iter().filter(|&&v| v).count();
    let dep_count = any_count.saturating_sub(all_count);

    let (any_pct, all_pct, dep_pct) = if mv.n_vertices == 0 {
        (0.0_f32, 0.0_f32, 0.0_f32)
    } else {
        (
            any_count as f32 / n * 100.0,
            all_count as f32 / n * 100.0,
            dep_count as f32 / n * 100.0,
        )
    };

    format!(
        "MultiView[{} cams]: any_visible={} ({:.1}%), all_visible={} ({:.1}%), view-dependent={} ({:.1}%)",
        mv.n_cameras, any_count, any_pct, all_count, all_pct, dep_count, dep_pct,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;
    use nalgebra as na;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    /// Single-triangle mesh in the XY plane with normals pointing +Z.
    ///
    /// Vertices are scaled to ±0.3 so they project well within the 256×256
    /// image when using `front_camera()` (focal=256, cx=cy=128, depth≈2).
    /// At depth 2, a vertex displaced by 0.3 in X or Y maps to screen offset
    /// 256 × 0.3 / 2 = 38.4 px from the principal point — well inside the image.
    fn simple_mesh() -> Mesh {
        let vertices = vec![
            na::Point3::new(0.0_f32, 0.0, 0.0),
            na::Point3::new(0.3_f32, 0.0, 0.0),
            na::Point3::new(0.0_f32, 0.3, 0.0),
        ];
        let normals = vec![
            na::Vector3::new(0.0_f32, 0.0, 1.0),
            na::Vector3::new(0.0_f32, 0.0, 1.0),
            na::Vector3::new(0.0_f32, 0.0, 1.0),
        ];
        let faces = vec![[0u32, 1, 2]];
        Mesh {
            vertices,
            normals,
            faces,
            uv_coords: Vec::new(),
        }
    }

    /// Camera located on the +Z side, looking toward −Z (i.e. looking at the
    /// mesh from the front, so vertex normals pointing +Z are front-facing).
    ///
    /// R is a 180° rotation around Y: maps +Z world → −Z camera.
    /// The camera world position is −R^T * t = −R * t (R symmetric here) = [0,0,2].
    fn front_camera() -> Camera {
        // 180° around Y: R = [[-1,0,0],[0,1,0],[0,0,-1]]
        let rotation = na::Matrix3::new(-1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
        Camera {
            rotation,
            translation: na::Vector3::new(0.0_f32, 0.0, 2.0),
            focal_x: 256.0,
            focal_y: 256.0,
            cx: 128.0,
            cy: 128.0,
            width: 256,
            height: 256,
            near: 0.01,
            far: 10.0,
        }
    }

    /// Camera located on the −Z side, looking toward +Z (back-facing to +Z normals).
    fn back_camera() -> Camera {
        Camera {
            rotation: na::Matrix3::identity(),
            translation: na::Vector3::new(0.0_f32, 0.0, 1.0),
            focal_x: 256.0,
            focal_y: 256.0,
            cx: 128.0,
            cy: 128.0,
            width: 256,
            height: 256,
            near: 0.01,
            far: 10.0,
        }
    }

    // -----------------------------------------------------------------------
    // compute_face_normal
    // -----------------------------------------------------------------------

    #[test]
    fn test_face_normal_xy_plane_points_z() {
        // Triangle in XY plane → normal should be +Z
        let v0 = [0.0_f32, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 1.0, 0.0];
        let n = compute_face_normal(v0, v1, v2);
        assert!(
            (n[2] - 1.0).abs() < 1e-5,
            "z component should be 1, got {}",
            n[2]
        );
        assert!(n[0].abs() < 1e-5);
        assert!(n[1].abs() < 1e-5);
    }

    #[test]
    fn test_face_normal_reversed_winding_points_neg_z() {
        let v0 = [0.0_f32, 0.0, 0.0];
        let v1 = [0.0, 1.0, 0.0]; // reversed winding
        let v2 = [1.0, 0.0, 0.0];
        let n = compute_face_normal(v0, v1, v2);
        assert!(
            (n[2] + 1.0).abs() < 1e-5,
            "z component should be -1, got {}",
            n[2]
        );
    }

    #[test]
    fn test_face_normal_degenerate_returns_zero() {
        let v0 = [0.0_f32, 0.0, 0.0];
        let v1 = [0.0, 0.0, 0.0]; // same point
        let v2 = [0.0, 0.0, 0.0];
        let n = compute_face_normal(v0, v1, v2);
        assert_eq!(n, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_face_normal_xz_plane_points_y() {
        // Triangle in XZ plane → normal should be ±Y
        let v0 = [0.0_f32, 0.0, 0.0];
        let v1 = [1.0, 0.0, 0.0];
        let v2 = [0.0, 0.0, 1.0];
        let n = compute_face_normal(v0, v1, v2);
        // e1×e2 = [1,0,0]×[0,0,1] = [0*1-0*0, 0*0-1*1, 1*0-0*0] = [0,-1,0]
        assert!(
            (n[1] + 1.0).abs() < 1e-5,
            "y component should be -1, got {}",
            n[1]
        );
    }

    #[test]
    fn test_face_normal_unit_length() {
        let v0 = [0.0_f32, 0.0, 0.0];
        let v1 = [2.0, 0.0, 0.0];
        let v2 = [0.0, 3.0, 0.0];
        let n = compute_face_normal(v0, v1, v2);
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-5,
            "normal should be unit length, len={len}"
        );
    }

    // -----------------------------------------------------------------------
    // is_front_facing
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_front_facing_aligned_normals_true() {
        // Normal and view_dir point same direction
        assert!(is_front_facing([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.0));
    }

    #[test]
    fn test_is_front_facing_opposite_normals_false() {
        assert!(!is_front_facing([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], 0.0));
    }

    #[test]
    fn test_is_front_facing_perpendicular_at_boundary() {
        // Dot = 0, threshold = 0 → NOT strictly greater
        assert!(!is_front_facing([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0));
    }

    #[test]
    fn test_is_front_facing_negative_threshold_accepts_some_backfaces() {
        // dot([1,0,0], [0.5, 0, 0]) would be 0.5 with normalized [0.5,0,0]=[1,0,0].
        // Use a glancing back-face: normal=[0,0,1], view=[0.5, 0, -0.866] (30° behind)
        // dot = 0*0.5 + 0*0 + 1*(-0.866) = -0.866
        // threshold = -0.9 → -0.866 > -0.9 → true
        let n = [0.0_f32, 0.0, 1.0];
        let v = [0.5_f32, 0.0, -0.866]; // normalized by hand (0.5^2+0.866^2≈1)
        assert!(
            is_front_facing(n, v, -0.9),
            "slightly behind should pass threshold -0.9"
        );
        // But same normal with threshold 0.0 → false (still back-facing)
        assert!(!is_front_facing(n, v, 0.0));
    }

    #[test]
    fn test_is_front_facing_high_threshold_rejects_glancing() {
        // dot = 0.1, threshold = 0.5 → false (glancing angle rejected)
        let n = [0.0_f32, 0.0, 1.0];
        let v = [0.995_f32, 0.0, 0.1]; // nearly perpendicular
        let len = (v[0] * v[0] + v[2] * v[2]).sqrt();
        let v_norm = [v[0] / len, 0.0, v[2] / len];
        assert!(!is_front_facing(n, v_norm, 0.5));
    }

    // -----------------------------------------------------------------------
    // is_in_frustum
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_in_frustum_center_is_inside() {
        let camera = front_camera();
        assert!(is_in_frustum([128.0, 128.0], &camera, 0.0));
    }

    #[test]
    fn test_is_in_frustum_outside_right_edge_false() {
        let camera = front_camera();
        assert!(!is_in_frustum([257.0, 128.0], &camera, 0.0));
    }

    #[test]
    fn test_is_in_frustum_outside_but_within_margin_true() {
        let camera = front_camera();
        // x=257, width=256 → outside by 1px; margin=2 → inside
        assert!(is_in_frustum([257.0, 128.0], &camera, 2.0));
    }

    #[test]
    fn test_is_in_frustum_negative_coords_outside() {
        let camera = front_camera();
        assert!(!is_in_frustum([-1.0, 128.0], &camera, 0.0));
    }

    #[test]
    fn test_is_in_frustum_negative_with_margin_inside() {
        let camera = front_camera();
        assert!(is_in_frustum([-1.0, 128.0], &camera, 2.0));
    }

    #[test]
    fn test_is_in_frustum_corner_pixel_inside() {
        let camera = front_camera();
        // Top-left corner pixel
        assert!(is_in_frustum([0.0, 0.0], &camera, 0.0));
    }

    // -----------------------------------------------------------------------
    // compute_face_screen_area
    // -----------------------------------------------------------------------

    #[test]
    fn test_face_screen_area_right_triangle_half() {
        // Right triangle with legs 1×1 → area = 0.5
        let s0 = [0.0_f32, 0.0];
        let s1 = [1.0, 0.0];
        let s2 = [0.0, 1.0];
        let area = compute_face_screen_area(s0, s1, s2);
        assert!((area - 0.5).abs() < 1e-5, "area should be 0.5, got {area}");
    }

    #[test]
    fn test_face_screen_area_degenerate_zero() {
        let s0 = [0.0_f32, 0.0];
        let s1 = [0.0, 0.0];
        let s2 = [0.0, 0.0];
        let area = compute_face_screen_area(s0, s1, s2);
        assert!(area.abs() < 1e-5, "degenerate triangle should have area ~0");
    }

    #[test]
    fn test_face_screen_area_unit_square_triangle() {
        // Triangle occupying half of a 2×2 square → area = 2.0
        let s0 = [0.0_f32, 0.0];
        let s1 = [2.0, 0.0];
        let s2 = [0.0, 2.0];
        let area = compute_face_screen_area(s0, s1, s2);
        assert!((area - 2.0).abs() < 1e-5, "area should be 2.0, got {area}");
    }

    #[test]
    fn test_face_screen_area_absolute_value() {
        // Reversed winding should still give positive area
        let s0 = [0.0_f32, 0.0];
        let s1 = [0.0, 1.0]; // reversed
        let s2 = [1.0, 0.0];
        let area = compute_face_screen_area(s0, s1, s2);
        assert!(area > 0.0, "area must be positive regardless of winding");
    }

    // -----------------------------------------------------------------------
    // compute_face_visibility
    // -----------------------------------------------------------------------

    #[test]
    fn test_face_visibility_empty_faces_error() {
        let mut mesh = simple_mesh();
        mesh.faces.clear();
        let config = VisibilityCullerConfig::default();
        let camera = front_camera();
        let result = compute_face_visibility(&mesh, &camera, &config);
        assert!(matches!(result, Err(VisibilityError::NoFaces)));
    }

    #[test]
    fn test_face_visibility_front_facing_visible() {
        let mesh = simple_mesh();
        let camera = front_camera();
        let config = VisibilityCullerConfig::default();
        let fv = compute_face_visibility(&mesh, &camera, &config).expect("should succeed");
        assert_eq!(fv.n_faces, 1);
        assert!(fv.visible[0], "front-facing face should be visible");
    }

    #[test]
    fn test_face_visibility_back_facing_not_visible() {
        let mesh = simple_mesh();
        let camera = back_camera(); // looking from -Z, face points +Z → back-facing
        let config = VisibilityCullerConfig::default();
        let fv = compute_face_visibility(&mesh, &camera, &config).expect("should succeed");
        assert!(!fv.visible[0], "back-facing face should not be visible");
    }

    #[test]
    fn test_face_visibility_screen_area_positive_for_visible() {
        let mesh = simple_mesh();
        let camera = front_camera();
        let config = VisibilityCullerConfig::default();
        let fv = compute_face_visibility(&mesh, &camera, &config).expect("should succeed");
        assert!(
            fv.screen_area[0] > 0.0,
            "visible face must have positive screen area"
        );
    }

    #[test]
    fn test_face_visibility_invalid_index_error() {
        let mut mesh = simple_mesh();
        mesh.faces = vec![[0, 1, 99]]; // index 99 out of range for 3 vertices
        let camera = front_camera();
        let config = VisibilityCullerConfig::default();
        let result = compute_face_visibility(&mesh, &camera, &config);
        assert!(matches!(
            result,
            Err(VisibilityError::VertexIndexOutOfRange { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // compute_vertex_visibility
    // -----------------------------------------------------------------------

    #[test]
    fn test_vertex_visibility_empty_mesh_error() {
        let mesh = Mesh {
            vertices: Vec::new(),
            normals: Vec::new(),
            faces: Vec::new(),
            uv_coords: Vec::new(),
        };
        let camera = front_camera();
        let config = VisibilityCullerConfig::default();
        let result = compute_vertex_visibility(&mesh, &camera, &config);
        assert!(matches!(result, Err(VisibilityError::EmptyMesh)));
    }

    #[test]
    fn test_vertex_visibility_normal_mismatch_error() {
        let mut mesh = simple_mesh();
        mesh.normals.pop(); // remove one normal → mismatch
        let camera = front_camera();
        let config = VisibilityCullerConfig::default();
        let result = compute_vertex_visibility(&mesh, &camera, &config);
        assert!(matches!(
            result,
            Err(VisibilityError::NormalCountMismatch { .. })
        ));
    }

    #[test]
    fn test_vertex_visibility_front_facing_camera_all_visible() {
        let mesh = simple_mesh();
        let camera = front_camera();
        let config = VisibilityCullerConfig::default();
        let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
        assert_eq!(vv.n_vertices, 3);
        // All normals point +Z, camera is on +Z side → all front-facing and in frustum
        for i in 0..3 {
            assert!(vv.front_facing[i], "vertex {i} should be front-facing");
            assert!(vv.in_frustum[i], "vertex {i} should be in-frustum");
            assert!(vv.visible[i], "vertex {i} should be visible");
        }
    }

    #[test]
    fn test_vertex_visibility_back_camera_not_front_facing() {
        let mesh = simple_mesh();
        let camera = back_camera(); // normals point +Z, camera on -Z → back-facing
        let config = VisibilityCullerConfig::default();
        let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
        for i in 0..3 {
            assert!(
                !vv.front_facing[i],
                "vertex {i} should be back-facing from this camera"
            );
            assert!(!vv.visible[i]);
        }
    }

    #[test]
    fn test_vertex_visibility_fields_consistent() {
        let mesh = simple_mesh();
        let camera = front_camera();
        let config = VisibilityCullerConfig::default();
        let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
        assert_eq!(vv.visible.len(), vv.n_vertices);
        assert_eq!(vv.in_frustum.len(), vv.n_vertices);
        assert_eq!(vv.front_facing.len(), vv.n_vertices);
        // visible must be in_frustum AND front_facing
        for i in 0..vv.n_vertices {
            assert_eq!(vv.visible[i], vv.in_frustum[i] && vv.front_facing[i]);
        }
    }

    // -----------------------------------------------------------------------
    // compute_visibility_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_visibility_stats_counts_match() {
        let mesh = simple_mesh();
        let camera = front_camera();
        let config = VisibilityCullerConfig::default();
        let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
        let stats = compute_visibility_stats(&vv);
        assert_eq!(stats.n_vertices, 3);
        assert_eq!(stats.n_visible_vertices, 3);
        assert_eq!(stats.n_front_facing, 3);
        assert_eq!(stats.n_in_frustum, 3);
    }

    #[test]
    fn test_visibility_stats_fractions_in_range() {
        let mesh = simple_mesh();
        let camera = front_camera();
        let config = VisibilityCullerConfig::default();
        let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
        let stats = compute_visibility_stats(&vv);
        assert!(stats.visible_fraction >= 0.0 && stats.visible_fraction <= 1.0);
        assert!(stats.front_facing_fraction >= 0.0 && stats.front_facing_fraction <= 1.0);
        assert!(stats.in_frustum_fraction >= 0.0 && stats.in_frustum_fraction <= 1.0);
    }

    #[test]
    fn test_visibility_stats_all_visible_fraction_one() {
        let mesh = simple_mesh();
        let camera = front_camera();
        let config = VisibilityCullerConfig::default();
        let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
        let stats = compute_visibility_stats(&vv);
        assert!((stats.visible_fraction - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_visibility_stats_none_visible() {
        let mesh = simple_mesh();
        let camera = back_camera();
        let config = VisibilityCullerConfig::default();
        let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
        let stats = compute_visibility_stats(&vv);
        assert_eq!(stats.n_visible_vertices, 0);
        assert!((stats.visible_fraction - 0.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // compute_multi_view_visibility
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_view_no_cameras_error() {
        let mesh = simple_mesh();
        let config = VisibilityCullerConfig::default();
        let result = compute_multi_view_visibility(&mesh, &[], &config);
        assert!(matches!(result, Err(VisibilityError::NoCameras)));
    }

    #[test]
    fn test_multi_view_single_camera() {
        let mesh = simple_mesh();
        let cameras = vec![front_camera()];
        let config = VisibilityCullerConfig::default();
        let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
        assert_eq!(mv.n_cameras, 1);
        assert_eq!(mv.n_vertices, 3);
        // All visible from front camera
        for i in 0..3 {
            assert!(mv.any_visible[i]);
            assert!(mv.all_visible[i]);
            assert_eq!(mv.view_count[i], 1);
        }
    }

    #[test]
    fn test_multi_view_two_cameras_any_but_not_all() {
        let mesh = simple_mesh();
        let cameras = vec![front_camera(), back_camera()];
        let config = VisibilityCullerConfig::default();
        let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
        assert_eq!(mv.n_cameras, 2);
        // Front camera sees all, back camera sees none
        // → any_visible = true for all, all_visible = false for all (back cam sees 0)
        for i in 0..3 {
            assert!(mv.any_visible[i], "vertex {i} any_visible should be true");
            assert!(!mv.all_visible[i], "vertex {i} all_visible should be false");
            assert_eq!(mv.view_count[i], 1, "vertex {i} view_count should be 1");
        }
    }

    #[test]
    fn test_multi_view_view_count_accumulates() {
        let mesh = simple_mesh();
        let cameras = vec![front_camera(), front_camera()]; // two identical cameras
        let config = VisibilityCullerConfig::default();
        let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
        for i in 0..3 {
            assert_eq!(
                mv.view_count[i], 2,
                "two identical cameras both see vertex {i}"
            );
            assert!(mv.all_visible[i]);
        }
    }

    #[test]
    fn test_multi_view_fields_lengths() {
        let mesh = simple_mesh();
        let cameras = vec![front_camera(), back_camera()];
        let config = VisibilityCullerConfig::default();
        let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
        assert_eq!(mv.any_visible.len(), 3);
        assert_eq!(mv.all_visible.len(), 3);
        assert_eq!(mv.view_count.len(), 3);
    }

    // -----------------------------------------------------------------------
    // find_view_dependent_vertices
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_view_dependent_two_cameras() {
        let mesh = simple_mesh();
        let cameras = vec![front_camera(), back_camera()];
        let config = VisibilityCullerConfig::default();
        let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
        let deps = find_view_dependent_vertices(&mv);
        // All 3 vertices are view-dependent (any=true, all=false)
        assert_eq!(deps.len(), 3);
    }

    #[test]
    fn test_find_view_dependent_all_visible_from_all_empty() {
        let mesh = simple_mesh();
        let cameras = vec![front_camera(), front_camera()];
        let config = VisibilityCullerConfig::default();
        let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
        let deps = find_view_dependent_vertices(&mv);
        // All visible from all views → no view-dependent vertices
        assert!(deps.is_empty(), "should be empty, got {} items", deps.len());
    }

    #[test]
    fn test_find_view_dependent_returns_indices_in_range() {
        let mesh = simple_mesh();
        let cameras = vec![front_camera(), back_camera()];
        let config = VisibilityCullerConfig::default();
        let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
        let deps = find_view_dependent_vertices(&mv);
        for &idx in &deps {
            assert!(idx < mv.n_vertices, "index {idx} out of range");
        }
    }

    // -----------------------------------------------------------------------
    // compute_optimal_view_coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimal_coverage_no_cameras_error() {
        let mesh = simple_mesh();
        let config = VisibilityCullerConfig::default();
        let result = compute_optimal_view_coverage(&mesh, &[], &config);
        assert!(matches!(result, Err(VisibilityError::NoCameras)));
    }

    #[test]
    fn test_optimal_coverage_front_camera_full() {
        let mesh = simple_mesh();
        let cameras = vec![front_camera()];
        let config = VisibilityCullerConfig::default();
        let cov = compute_optimal_view_coverage(&mesh, &cameras, &config).expect("should succeed");
        assert_eq!(cov.len(), 1);
        assert!(
            (cov[0] - 1.0).abs() < 1e-5,
            "all vertices visible, coverage should be 1.0"
        );
    }

    #[test]
    fn test_optimal_coverage_back_camera_zero() {
        let mesh = simple_mesh();
        let cameras = vec![back_camera()];
        let config = VisibilityCullerConfig::default();
        let cov = compute_optimal_view_coverage(&mesh, &cameras, &config).expect("should succeed");
        assert_eq!(cov.len(), 1);
        assert!(
            (cov[0] - 0.0).abs() < 1e-5,
            "no vertices visible, coverage should be 0.0"
        );
    }

    #[test]
    fn test_optimal_coverage_two_cameras_length() {
        let mesh = simple_mesh();
        let cameras = vec![front_camera(), back_camera()];
        let config = VisibilityCullerConfig::default();
        let cov = compute_optimal_view_coverage(&mesh, &cameras, &config).expect("should succeed");
        assert_eq!(cov.len(), 2);
    }

    #[test]
    fn test_optimal_coverage_values_in_01() {
        let mesh = simple_mesh();
        let cameras = vec![front_camera(), back_camera()];
        let config = VisibilityCullerConfig::default();
        let cov = compute_optimal_view_coverage(&mesh, &cameras, &config).expect("should succeed");
        for &c in &cov {
            assert!((0.0..=1.0).contains(&c), "coverage {c} out of [0,1]");
        }
    }

    // -----------------------------------------------------------------------
    // select_maximally_covering_views
    // -----------------------------------------------------------------------

    #[test]
    fn test_select_top_k_basic() {
        let coverage = vec![0.3, 0.9, 0.5, 0.7];
        let top = select_maximally_covering_views(&coverage, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0], 1, "index 1 has highest coverage 0.9");
        assert_eq!(top[1], 3, "index 3 has second-highest coverage 0.7");
    }

    #[test]
    fn test_select_top_k_exceeds_length_returns_all() {
        let coverage = vec![0.3, 0.8];
        let top = select_maximally_covering_views(&coverage, 10);
        assert_eq!(top.len(), 2, "k > n should return all cameras");
    }

    #[test]
    fn test_select_top_k_zero_returns_empty() {
        let coverage = vec![0.5, 0.9];
        let top = select_maximally_covering_views(&coverage, 0);
        assert!(top.is_empty());
    }

    #[test]
    fn test_select_top_k_single() {
        let coverage = vec![0.5];
        let top = select_maximally_covering_views(&coverage, 1);
        assert_eq!(top, vec![0]);
    }

    #[test]
    fn test_select_top_k_deterministic_tie_break() {
        // Identical coverage: tie broken by index ascending
        let coverage = vec![0.5, 0.5, 0.5];
        let top = select_maximally_covering_views(&coverage, 2);
        assert_eq!(top, vec![0, 1]);
    }

    // -----------------------------------------------------------------------
    // format_visibility_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_visibility_stats_not_empty() {
        let mesh = simple_mesh();
        let camera = front_camera();
        let config = VisibilityCullerConfig::default();
        let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
        let stats = compute_visibility_stats(&vv);
        let s = format_visibility_stats(&stats);
        assert!(!s.is_empty());
        assert!(
            s.contains("Visibility:"),
            "expected 'Visibility:' prefix in: {s}"
        );
    }

    #[test]
    fn test_format_visibility_stats_contains_numbers() {
        let mesh = simple_mesh();
        let camera = front_camera();
        let config = VisibilityCullerConfig::default();
        let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
        let stats = compute_visibility_stats(&vv);
        let s = format_visibility_stats(&stats);
        // Should contain vertex counts and fractions
        assert!(s.contains("3/3"), "should contain '3/3' in: {s}");
    }

    // -----------------------------------------------------------------------
    // format_multi_view_stats
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_multi_view_stats_not_empty() {
        let mesh = simple_mesh();
        let cameras = vec![front_camera(), back_camera()];
        let config = VisibilityCullerConfig::default();
        let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
        let s = format_multi_view_stats(&mv);
        assert!(!s.is_empty());
        assert!(
            s.contains("MultiView"),
            "expected 'MultiView' prefix in: {s}"
        );
    }

    #[test]
    fn test_format_multi_view_stats_contains_cam_count() {
        let mesh = simple_mesh();
        let cameras = vec![front_camera(), back_camera()];
        let config = VisibilityCullerConfig::default();
        let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
        let s = format_multi_view_stats(&mv);
        assert!(s.contains("2 cams"), "should mention camera count in: {s}");
    }

    // -----------------------------------------------------------------------
    // VisibilityError variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_empty_mesh_display() {
        let e = VisibilityError::EmptyMesh;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_error_no_faces_display() {
        let e = VisibilityError::NoFaces;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_error_vertex_index_out_of_range_display() {
        let e = VisibilityError::VertexIndexOutOfRange { idx: 99, n: 3 };
        let s = e.to_string();
        assert!(s.contains("99"), "should mention idx 99 in: {s}");
        assert!(s.contains('3'), "should mention n=3 in: {s}");
    }

    #[test]
    fn test_error_no_cameras_display() {
        let e = VisibilityError::NoCameras;
        assert!(!e.to_string().is_empty());
    }

    #[test]
    fn test_error_normal_count_mismatch_display() {
        let e = VisibilityError::NormalCountMismatch {
            normals: 5,
            vertices: 3,
        };
        let s = e.to_string();
        assert!(s.contains('5'), "should mention normals=5 in: {s}");
        assert!(s.contains('3'), "should mention vertices=3 in: {s}");
    }

    // -----------------------------------------------------------------------
    // VisibilityCullerConfig default values
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default_values() {
        let config = VisibilityCullerConfig::default();
        assert_eq!(config.backface_threshold, 0.0);
        assert_eq!(config.frustum_margin, 0.0);
        assert!(!config.use_depth_test);
        assert!((config.depth_bias - 1e-4).abs() < 1e-7);
    }

    // -----------------------------------------------------------------------
    // VertexVisibility and FaceVisibility field checks
    // -----------------------------------------------------------------------

    #[test]
    fn test_vertex_visibility_n_vertices_correct() {
        let mesh = simple_mesh();
        let camera = front_camera();
        let config = VisibilityCullerConfig::default();
        let vv = compute_vertex_visibility(&mesh, &camera, &config).expect("should succeed");
        assert_eq!(vv.n_vertices, mesh.vertices.len());
    }

    #[test]
    fn test_face_visibility_n_faces_correct() {
        let mesh = simple_mesh();
        let camera = front_camera();
        let config = VisibilityCullerConfig::default();
        let fv = compute_face_visibility(&mesh, &camera, &config).expect("should succeed");
        assert_eq!(fv.n_faces, mesh.faces.len());
    }

    #[test]
    fn test_face_visibility_screen_area_length() {
        let mesh = simple_mesh();
        let camera = front_camera();
        let config = VisibilityCullerConfig::default();
        let fv = compute_face_visibility(&mesh, &camera, &config).expect("should succeed");
        assert_eq!(fv.screen_area.len(), fv.n_faces);
        assert_eq!(fv.visible.len(), fv.n_faces);
    }

    // -----------------------------------------------------------------------
    // MultiViewVisibility field checks
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_view_n_cameras_correct() {
        let mesh = simple_mesh();
        let cameras = vec![front_camera(), back_camera()];
        let config = VisibilityCullerConfig::default();
        let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
        assert_eq!(mv.n_cameras, 2);
    }

    #[test]
    fn test_multi_view_n_vertices_correct() {
        let mesh = simple_mesh();
        let cameras = vec![front_camera()];
        let config = VisibilityCullerConfig::default();
        let mv = compute_multi_view_visibility(&mesh, &cameras, &config).expect("should succeed");
        assert_eq!(mv.n_vertices, 3);
    }

    // -----------------------------------------------------------------------
    // project_vertex edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_project_vertex_behind_camera_returns_none() {
        let camera = front_camera();
        // Camera is at world [0,0,2] looking toward -Z.
        // R * [0,0,3] + t = [-1,0,0]*0 + [-1,0,0]*0 + R*[0,0,3] ...
        // With R = [[-1,0,0],[0,1,0],[0,0,-1]] and t=[0,0,2]:
        // cam_pos = R*[0,0,3] + [0,0,2] = [0,0,-3] + [0,0,2] = [0,0,-1]
        // z = -1 ≤ near=0.01 → None
        let result = project_vertex([0.0, 0.0, 3.0], &camera);
        assert!(result.is_none(), "vertex behind camera should return None");
    }

    #[test]
    fn test_project_vertex_in_front_of_camera_returns_some() {
        let camera = front_camera();
        // Vertex at origin: cam_pos = R*[0,0,0]+t = [0,0,2], z=2 > near → Some
        let result = project_vertex([0.0, 0.0, 0.0], &camera);
        assert!(
            result.is_some(),
            "vertex in front of camera should project successfully"
        );
    }
}
