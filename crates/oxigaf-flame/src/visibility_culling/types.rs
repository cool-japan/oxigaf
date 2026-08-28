//! Data types for visibility culling: the error type, [`VisibilityCullerConfig`],
//! and the per-vertex/per-face/multi-view result and statistics structs.
//!
//! Split out of the former monolithic `visibility_culling.rs` to stay under
//! the workspace's 2000-line-per-file policy; see [`super::raster`] for the
//! private depth-buffer rasterizer and geometry helpers, and
//! [`super::functions`] for the public visibility algorithms.

use thiserror::Error;

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

    /// Whether to perform a depth-occlusion test.
    ///
    /// When `true`, the mesh is rasterized into a per-pixel depth buffer at the
    /// camera resolution and every front-facing, in-frustum vertex (and, for
    /// [`super::functions::compute_face_visibility`], every face centroid) is tested against it,
    /// so self-occluded geometry — the far side of the nose, an ear behind the
    /// cheek — is reported as not visible. Off by default: it costs one full
    /// rasterization pass per camera.
    pub use_depth_test: bool,

    /// Minimum depth tolerance (camera-space units) for the depth test.
    ///
    /// The effective tolerance is `max(depth_bias, largest depth step to the four
    /// neighbouring pixels)`: a slope-scaled bias, so a vertex on a steeply
    /// slanted surface is never reported as occluding itself while one genuinely
    /// behind another surface still is. Only read when `use_depth_test` is set.
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
