//! Pure-Rust SVG visualization utilities for FLAME mesh and skeleton.
//!
//! This module provides camera projection, wireframe rendering, and joint
//! overlay generation — all written to SVG text without any external SVG
//! library dependency.
//!
//! ## Example
//!
//! ```rust,no_run
//! use oxigaf_flame::visualize::{SvgCamera, WireframeOptions, render_wireframe};
//! use oxigaf_flame::Mesh;
//!
//! let camera = SvgCamera::front_view(512);
//! let options = WireframeOptions::default();
//! // mesh obtained from FlameModel::forward(...)
//! // let svg = render_wireframe(&mesh, &camera, &options)?;
//! ```

use std::collections::HashSet;
use std::fmt::Write as FmtWrite;
use std::io::Write;
use std::path::Path;

use nalgebra as na;

use crate::{FlameError, Mesh};

// ---------------------------------------------------------------------------
// Private SVG builder
// ---------------------------------------------------------------------------

/// Accumulates SVG element strings and emits a complete document.
struct SvgBuilder {
    width: u32,
    height: u32,
    background_color: String,
    elements: Vec<String>,
}

impl SvgBuilder {
    /// Create a new builder with the given canvas dimensions and background color.
    fn new(width: u32, height: u32, background_color: &str) -> Self {
        Self {
            width,
            height,
            background_color: background_color.to_owned(),
            elements: Vec::new(),
        }
    }

    /// Add a `<line>` element.
    fn add_line(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, color: &str, stroke_width: f32) {
        self.elements.push(format!(
            r#"  <line x1="{x1:.2}" y1="{y1:.2}" x2="{x2:.2}" y2="{y2:.2}" stroke="{color}" stroke-width="{stroke_width:.2}"/>"#,
        ));
    }

    /// Add a `<circle>` element.
    fn add_circle(&mut self, cx: f32, cy: f32, r: f32, fill: &str) {
        self.elements.push(format!(
            r#"  <circle cx="{cx:.2}" cy="{cy:.2}" r="{r:.2}" fill="{fill}"/>"#,
        ));
    }

    /// Add a `<text>` element. Special XML characters in `text` are escaped.
    fn add_text(&mut self, x: f32, y: f32, text: &str, color: &str, font_size: f32) {
        let escaped = xml_escape(text);
        self.elements.push(format!(
            r#"  <text x="{x:.2}" y="{y:.2}" fill="{color}" font-size="{font_size:.1}" font-family="sans-serif">{escaped}</text>"#,
        ));
    }

    /// Consume `other`'s elements into this builder.
    fn merge(&mut self, other: SvgBuilder) {
        self.elements.extend(other.elements);
    }

    /// Build and return the complete SVG document string.
    fn build(&self) -> String {
        let mut out =
            String::with_capacity(256 + self.elements.iter().map(|e| e.len() + 1).sum::<usize>());
        // write! to String never fails (infallible).
        let _ = write!(
            out,
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
            self.width, self.height, self.width, self.height
        );
        out.push('\n');
        // Background rectangle
        let _ = write!(
            out,
            r#"  <rect width="{}" height="{}" fill="{}"/>"#,
            self.width, self.height, self.background_color
        );
        out.push('\n');
        for elem in &self.elements {
            out.push_str(elem);
            out.push('\n');
        }
        out.push_str("</svg>");
        out
    }
}

/// Escape XML special characters for use in SVG text content.
fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            other => out.push(other),
        }
    }
    out
}

// ---------------------------------------------------------------------------
// SvgCamera
// ---------------------------------------------------------------------------

/// A simple perspective camera used to project 3-D points into SVG space.
///
/// The camera uses a look-at parameterisation: an eye position, a target that
/// the camera is aimed at, and an up vector.  Perspective projection maps
/// world-space points to pixel coordinates on a canvas of implicit size
/// (`2·cx × 2·cy` by convention, matching the `cx`/`cy` principal-point
/// parameters).
///
/// # Coordinate conventions
///
/// - Camera space: +X right, +Y up, **-Z forward** (right-hand rule).
/// - Screen space: +X right, +Y down (SVG convention).
#[derive(Debug, Clone)]
pub struct SvgCamera {
    /// Principal point X in pixels (typically half the image width).
    pub cx: f32,
    /// Principal point Y in pixels (typically half the image height).
    pub cy: f32,
    /// Focal length in pixels.  Larger values zoom in.
    pub focal: f32,
    /// Camera eye position in world space.
    pub eye: [f32; 3],
    /// Look-at target in world space.
    pub target: [f32; 3],
    /// World-space up vector (need not be normalised).
    pub up: [f32; 3],
}

impl SvgCamera {
    /// Construct a front-facing camera centred on a canonical FLAME head.
    ///
    /// The camera sits at `(0, 0, 0.6)` looking toward the origin, which
    /// frames a head whose vertices span roughly `[-0.1, 0.1]` in X/Y and
    /// `[0.0, 0.3]` in Z (the FLAME canonical space).
    #[must_use]
    pub fn front_view(image_size: u32) -> Self {
        let half = image_size as f32 / 2.0;
        let focal = image_size as f32 * 1.5;
        Self {
            cx: half,
            cy: half,
            focal,
            eye: [0.0, 0.0, 0.6],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
        }
    }

    /// Construct a right-side-view camera.
    #[must_use]
    pub fn side_view(image_size: u32) -> Self {
        let half = image_size as f32 / 2.0;
        let focal = image_size as f32 * 1.5;
        Self {
            cx: half,
            cy: half,
            focal,
            eye: [0.6, 0.0, 0.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
        }
    }

    /// Construct a three-quarter-view camera (45° azimuth).
    #[must_use]
    pub fn three_quarter_view(image_size: u32) -> Self {
        let half = image_size as f32 / 2.0;
        let focal = image_size as f32 * 1.5;
        let dist: f32 = 0.6;
        let angle: f32 = std::f32::consts::FRAC_PI_4; // 45°
        Self {
            cx: half,
            cy: half,
            focal,
            eye: [dist * angle.sin(), 0.0, dist * angle.cos()],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
        }
    }

    /// Build the look-at rotation matrix and translation vector.
    ///
    /// Returns `(rotation_3x3, translation_vector)` where the rotation maps
    /// world-space vectors to camera-space vectors, and `translation` is the
    /// camera origin expressed in camera space (`-R * eye`).
    ///
    /// Returns `None` when `eye` and `target` coincide, since no forward
    /// direction can be derived. A forward direction that is (anti)parallel
    /// to `up` (e.g. a straight top-down view) does NOT fail: `right_raw`
    /// falls back to an alternate world axis for the initial cross product,
    /// so the result is always a well-defined orthonormal frame rather than
    /// the NaN that `forward.cross(&up).normalize()` would otherwise produce
    /// from a zero-length cross product.
    #[must_use]
    fn view_matrix(&self) -> Option<(na::Matrix3<f32>, na::Vector3<f32>)> {
        let eye = na::Point3::from(self.eye);
        let target = na::Point3::from(self.target);
        let up = na::Vector3::from(self.up);

        let fwd_raw = target - eye;
        if fwd_raw.norm() < 1e-9 {
            return None;
        }
        let forward = fwd_raw.normalize(); // +Z_cam points toward target (we negate below)

        let mut right_raw = forward.cross(&up);
        if right_raw.norm() < 1e-6 {
            // `up` is (anti)parallel to `forward`: fall back to whichever
            // world axis is farthest from `forward` so the cross product is
            // well-conditioned (same trick as `rigid_alignment::svd_compute_u`).
            let alt = if forward.x.abs() < 0.9 {
                na::Vector3::x()
            } else {
                na::Vector3::y()
            };
            right_raw = forward.cross(&alt);
        }
        let right = right_raw.normalize();
        let up_actual = right.cross(&forward); // reorthogonalise

        // Row-major look-at: each row is a camera axis expressed in world space.
        //   row 0 = right   (+X_cam)
        //   row 1 = up_actual (+Y_cam, world Y flipped for SVG later)
        //   row 2 = -forward  (+Z_cam, conventional camera -Z is depth)
        let rot = na::Matrix3::from_rows(&[
            right.transpose(),
            up_actual.transpose(),
            (-forward).transpose(),
        ]);

        let t = rot * (-eye.coords);
        Some((rot, t))
    }

    /// Project a world-space point to SVG pixel coordinates using a
    /// precomputed view transform (see [`SvgCamera::view_matrix`]).
    ///
    /// Shared by [`SvgCamera::project`] (which computes the transform fresh
    /// every call — fine for a single ad-hoc projection) and by batch
    /// callers such as [`build_wireframe_into`] that compute the transform
    /// once and reuse it across every mesh vertex, avoiding thousands of
    /// redundant look-at reconstructions per render.
    #[must_use]
    fn project_with(
        &self,
        rot: &na::Matrix3<f32>,
        t: &na::Vector3<f32>,
        point: [f32; 3],
    ) -> Option<(f32, f32)> {
        let p_world = na::Point3::from(point);
        let p_cam = rot * p_world.coords + t;

        // Camera convention: -Z_cam is depth (toward scene). Positive cam_z
        // means the point is in front of the camera.
        let cam_z = -p_cam.z;
        if cam_z <= 0.0 {
            return None;
        }

        // Perspective division: x_screen = cx + focal * (p_cam.x / cam_z)
        // Negate Y because SVG Y increases downward but camera Y is up.
        let sx = self.cx + self.focal * (p_cam.x / cam_z);
        let sy = self.cy - self.focal * (p_cam.y / cam_z);
        Some((sx, sy))
    }

    /// Project a world-space point to SVG pixel coordinates.
    ///
    /// Returns `None` when the point is at or behind the camera plane
    /// (i.e., camera-space Z ≤ 0), or when `eye` and `target` coincide (no
    /// view direction can be derived).
    #[must_use]
    pub fn project(&self, point: [f32; 3]) -> Option<(f32, f32)> {
        let (rot, t) = self.view_matrix()?;
        self.project_with(&rot, &t, point)
    }
}

// ---------------------------------------------------------------------------
// WireframeOptions
// ---------------------------------------------------------------------------

/// Options controlling how a mesh wireframe is rendered to SVG.
#[derive(Debug, Clone)]
pub struct WireframeOptions {
    /// Canvas size in pixels (square).
    pub image_size: u32,
    /// SVG stroke colour for edges (CSS colour string, e.g. `"#333333"`).
    pub edge_color: String,
    /// Stroke width for edges in pixels.
    pub stroke_width: f32,
    /// Background fill colour (CSS colour string, e.g. `"white"`).
    pub background_color: String,
    /// Whether to draw a circle at each visible vertex position.
    pub show_vertices: bool,
    /// Radius of vertex circles in pixels.
    pub vertex_radius: f32,
    /// Fill colour of vertex circles.
    pub vertex_color: String,
    /// Skip faces whose normal points away from the camera (back-face cull).
    pub cull_backfaces: bool,
}

impl Default for WireframeOptions {
    fn default() -> Self {
        Self {
            image_size: 512,
            edge_color: "#333333".to_owned(),
            stroke_width: 0.5,
            background_color: "white".to_owned(),
            show_vertices: false,
            vertex_radius: 1.5,
            vertex_color: "#0055aa".to_owned(),
            cull_backfaces: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal projection helpers
// ---------------------------------------------------------------------------

/// Project all mesh vertices using a precomputed view transform (see
/// [`SvgCamera::view_matrix`]), returning `None` per vertex that is behind
/// the camera. Computing the transform once per render instead of once per
/// vertex avoids thousands of redundant look-at reconstructions for a
/// typical FLAME mesh (5023 vertices).
fn project_vertices_with(
    mesh: &Mesh,
    camera: &SvgCamera,
    rot: &na::Matrix3<f32>,
    t: &na::Vector3<f32>,
) -> Vec<Option<(f32, f32)>> {
    mesh.vertices
        .iter()
        .map(|v| camera.project_with(rot, t, [v.x, v.y, v.z]))
        .collect()
}

/// Back-face culling test.  Returns `true` if the face should be rendered.
///
/// A face is visible when its world-space normal has a positive dot product
/// with the vector from its centroid toward the camera eye.
fn is_front_facing(face: &[u32; 3], mesh: &Mesh, camera: &SvgCamera) -> bool {
    let i0 = face[0] as usize;
    let i1 = face[1] as usize;
    let i2 = face[2] as usize;

    let v0 = &mesh.vertices[i0];
    let v1 = &mesh.vertices[i1];
    let v2 = &mesh.vertices[i2];

    // Face normal in world space (unnormalised is fine for sign test).
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let normal = e1.cross(&e2);

    // Centroid
    let centroid = na::Point3::new(
        (v0.x + v1.x + v2.x) / 3.0,
        (v0.y + v1.y + v2.y) / 3.0,
        (v0.z + v1.z + v2.z) / 3.0,
    );

    let eye = na::Point3::from(camera.eye);
    let view_dir = eye - centroid;

    normal.dot(&view_dir) > 0.0
}

// ---------------------------------------------------------------------------
// Public rendering functions
// ---------------------------------------------------------------------------

/// Render a triangle mesh as an SVG wireframe.
///
/// The function projects every face of `mesh` through `camera`, optionally
/// culls back faces, deduplicates shared edges, and draws each visible edge
/// as an SVG `<line>`.
///
/// # Errors
///
/// Returns [`FlameError::IndexOutOfBounds`] if `mesh.faces` contains an
/// out-of-range vertex index. Returns [`FlameError::InvalidParams`] if
/// `camera`'s eye and target coincide, since no view direction can be
/// derived.
pub fn render_wireframe(
    mesh: &Mesh,
    camera: &SvgCamera,
    options: &WireframeOptions,
) -> Result<String, FlameError> {
    let size = options.image_size;
    let mut svg = SvgBuilder::new(size, size, &options.background_color);
    build_wireframe_into(mesh, camera, options, &mut svg)?;
    Ok(svg.build())
}

/// Render joint positions as an SVG overlay (circles + optional labels).
///
/// The function projects each joint through `camera` and emits a `<circle>`
/// for every joint that is in front of the camera.  When `joint_names` is
/// provided it must have the same length as `joint_positions`; a `<text>`
/// label is placed just to the right of each circle.
///
/// # Errors
///
/// Returns [`FlameError::InvalidParams`] when `joint_names` is present but
/// has a different length than `joint_positions`.
pub fn render_joints_svg(
    joint_positions: &[[f32; 3]],
    joint_names: Option<&[&str]>,
    camera: &SvgCamera,
    image_size: u32,
) -> Result<String, FlameError> {
    if let Some(names) = joint_names {
        if names.len() != joint_positions.len() {
            return Err(FlameError::InvalidParams(format!(
                "joint_names length ({}) must match joint_positions length ({})",
                names.len(),
                joint_positions.len()
            )));
        }
    }

    let mut svg = SvgBuilder::new(image_size, image_size, "none");

    let joint_color = "#ff4400";
    let label_color = "#220000";
    let radius = 4.0_f32;
    let font_size = 11.0_f32;

    for (ji, &pos) in joint_positions.iter().enumerate() {
        let Some((sx, sy)) = camera.project(pos) else {
            continue;
        };
        svg.add_circle(sx, sy, radius, joint_color);
        if let Some(names) = joint_names {
            // Safety: lengths verified equal above.
            let name = names[ji];
            svg.add_text(
                sx + radius + 2.0,
                sy + font_size * 0.35,
                name,
                label_color,
                font_size,
            );
        }
    }

    // `svg` was already built with a transparent ("none") background, so it
    // is a complete, valid standalone document as-is.
    Ok(svg.build())
}

/// Render a combined wireframe + joint overlay SVG image.
///
/// This renders the wireframe (via the same internal helper backing
/// [`render_wireframe`]) and joint circles into one shared SVG builder,
/// which is cheaper than rendering each separately and compositing the
/// resulting SVG text.
///
/// # Errors
///
/// Returns [`FlameError::IndexOutOfBounds`] if `mesh.faces` contains an
/// out-of-range vertex index. Returns [`FlameError::InvalidParams`] if
/// `camera`'s eye and target coincide, since no view direction can be
/// derived.
pub fn render_mesh_with_joints(
    mesh: &Mesh,
    joint_positions: &[[f32; 3]],
    camera: &SvgCamera,
    wireframe_options: &WireframeOptions,
) -> Result<String, FlameError> {
    let size = wireframe_options.image_size;

    // Build joint elements into their own builder, then extract elements.
    let joint_color = "#ff4400";
    let radius = 4.0_f32;
    let mut joint_builder = SvgBuilder::new(size, size, "none");
    for &pos in joint_positions {
        let Some((sx, sy)) = camera.project(pos) else {
            continue;
        };
        joint_builder.add_circle(sx, sy, radius, joint_color);
    }

    // Parse wireframe SVG: extract inner lines (cheap: just embed elements).
    // Rather than parsing SVG, we rebuild from scratch by using internal helpers.
    // Use a combined builder for the actual output.
    let mut combined = SvgBuilder::new(size, size, &wireframe_options.background_color);

    // Re-render wireframe elements into combined builder.
    build_wireframe_into(mesh, camera, wireframe_options, &mut combined)?;

    // Add joint circles on top.
    combined.merge(joint_builder);

    Ok(combined.build())
}

/// Internal helper: render wireframe elements into an existing [`SvgBuilder`].
///
/// This is the sole wireframe implementation: [`render_wireframe`] wraps it
/// with its own builder, and [`render_mesh_with_joints`] layers joint
/// circles on top of it, so any fix here (projection, culling, or the
/// degenerate-camera guard) applies identically to both public entry points
/// rather than risking the two renderers drifting apart.
fn build_wireframe_into(
    mesh: &Mesh,
    camera: &SvgCamera,
    options: &WireframeOptions,
    svg: &mut SvgBuilder,
) -> Result<(), FlameError> {
    if mesh.faces.is_empty() {
        return Ok(());
    }

    let n_verts = mesh.vertices.len();
    for face in &mesh.faces {
        for &idx in face {
            if idx as usize >= n_verts {
                return Err(FlameError::index_out_of_bounds(
                    "build_wireframe_into face index",
                    idx as usize,
                    n_verts,
                ));
            }
        }
    }

    // Compute the look-at transform once for the whole mesh (see
    // `project_vertices_with`), rather than once per vertex as a naive
    // `camera.project(...)` loop would.
    let (rot, t) = camera.view_matrix().ok_or_else(|| {
        FlameError::InvalidParams(
            "SvgCamera has a degenerate view: eye and target coincide, so no view \
             direction can be derived"
                .to_string(),
        )
    })?;
    let projected = project_vertices_with(mesh, camera, &rot, &t);

    // NOTE: face draw order does not affect the output. Every edge is drawn
    // at most once (via `drawn_edges` below) and all edges share the same
    // color/width, so whichever face's edge wins the dedup race produces a
    // byte-identical `<line>`. A painter's-algorithm depth sort used to run
    // here for no visual effect; it was removed rather than kept as dead
    // computation (it cost ~3 look-at-space transforms per face).
    let mut drawn_edges: HashSet<(u32, u32)> = HashSet::new();
    let mut visible_verts: HashSet<usize> = HashSet::new();

    for face in &mesh.faces {
        if options.cull_backfaces && !is_front_facing(face, mesh, camera) {
            continue;
        }

        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;

        // All three projected vertices must be visible (in front of camera).
        let Some(p0) = projected[i0] else { continue };
        let Some(p1) = projected[i1] else { continue };
        let Some(p2) = projected[i2] else { continue };

        let screen_pts = [(i0, p0), (i1, p1), (i2, p2)];
        let edges = [
            (face[0], face[1], p0, p1),
            (face[1], face[2], p1, p2),
            (face[2], face[0], p2, p0),
        ];

        for (a, b, pa, pb) in edges {
            let key = (a.min(b), a.max(b));
            if drawn_edges.insert(key) {
                svg.add_line(
                    pa.0,
                    pa.1,
                    pb.0,
                    pb.1,
                    &options.edge_color,
                    options.stroke_width,
                );
            }
        }

        if options.show_vertices {
            for (vi, _) in &screen_pts {
                visible_verts.insert(*vi);
            }
        }
    }

    // Draw vertex dots on top of edges.
    if options.show_vertices {
        for vi in visible_verts {
            if let Some((sx, sy)) = projected[vi] {
                svg.add_circle(sx, sy, options.vertex_radius, &options.vertex_color);
            }
        }
    }

    Ok(())
}

/// Write an SVG string to a file at `path`.
///
/// # Errors
///
/// Returns [`FlameError::Export`] if the file cannot be created or written.
pub fn save_svg(svg: &str, path: &Path) -> Result<(), FlameError> {
    let file = std::fs::File::create(path).map_err(|e| {
        FlameError::export("SVG", format!("failed to create '{}': {e}", path.display()))
    })?;
    let mut writer = std::io::BufWriter::new(file);
    writer
        .write_all(svg.as_bytes())
        .map_err(|e| FlameError::export("SVG", format!("write error: {e}")))?;
    writer
        .flush()
        .map_err(|e| FlameError::export("SVG", format!("flush error: {e}")))?;
    Ok(())
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

    /// Build a simple tetrahedron mesh facing roughly toward +Z.
    fn tetrahedron() -> Mesh {
        let vertices = vec![
            na::Point3::new(0.0f32, 0.05, 0.1),
            na::Point3::new(-0.05f32, -0.05, 0.1),
            na::Point3::new(0.05f32, -0.05, 0.1),
            na::Point3::new(0.0f32, 0.0, 0.2),
        ];
        // Faces wound counter-clockwise (normals point toward viewer for front faces).
        let faces = vec![[0u32, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
        Mesh::new(vertices, faces)
    }

    // -----------------------------------------------------------------------
    // SvgCamera tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_front_view_creates_valid_camera() {
        let cam = SvgCamera::front_view(512);
        assert_eq!(cam.cx, 256.0);
        assert_eq!(cam.cy, 256.0);
        assert!(cam.focal > 0.0, "focal must be positive");
        assert_ne!(cam.eye, cam.target, "eye and target must differ");
    }

    #[test]
    fn test_side_view_creates_valid_camera() {
        let cam = SvgCamera::side_view(512);
        assert_eq!(cam.cx, 256.0);
        assert_eq!(cam.cy, 256.0);
        assert!(cam.focal > 0.0);
    }

    #[test]
    fn test_three_quarter_view_creates_valid_camera() {
        let cam = SvgCamera::three_quarter_view(512);
        assert!(cam.focal > 0.0);
        // Eye should be offset in both X and Z.
        assert!(
            cam.eye[0].abs() > 1e-6,
            "eye X should be nonzero for 3/4 view"
        );
        assert!(
            cam.eye[2].abs() > 1e-6,
            "eye Z should be nonzero for 3/4 view"
        );
    }

    #[test]
    fn test_project_point_in_front_returns_some() {
        let cam = SvgCamera::front_view(512);
        // The camera is at (0, 0, 0.6) looking toward origin.
        // A point at (0, 0, 0.1) is between the camera and the origin: in front.
        let result = cam.project([0.0, 0.0, 0.1]);
        assert!(result.is_some(), "point in front should project to Some");
    }

    #[test]
    fn test_project_point_behind_camera_returns_none() {
        let cam = SvgCamera::front_view(512);
        // Camera eye is at Z = 0.6, looking toward -Z (toward origin).
        // A point at Z = 1.0 is BEHIND the camera.
        let result = cam.project([0.0, 0.0, 1.0]);
        assert!(result.is_none(), "point behind camera should return None");
    }

    #[test]
    fn test_project_on_axis_maps_to_principal_point() {
        let cam = SvgCamera::front_view(512);
        // The exact target is at origin; it should project near cx, cy.
        if let Some((sx, sy)) = cam.project([0.0, 0.0, 0.0]) {
            assert!((sx - cam.cx).abs() < 1.0, "on-axis X should hit cx");
            assert!((sy - cam.cy).abs() < 1.0, "on-axis Y should hit cy");
        }
    }

    #[test]
    fn test_project_degenerate_eye_equals_target_returns_none() {
        // No forward direction can be derived when eye == target; this must
        // return None, not silently produce a NaN-poisoned projection.
        let cam = SvgCamera {
            cx: 256.0,
            cy: 256.0,
            focal: 700.0,
            eye: [0.0, 0.0, 0.5],
            target: [0.0, 0.0, 0.5],
            up: [0.0, 1.0, 0.0],
        };
        assert!(
            cam.project([0.1, 0.1, 0.1]).is_none(),
            "degenerate eye==target camera must not silently project"
        );
    }

    #[test]
    fn test_project_forward_parallel_to_up_does_not_produce_nan() {
        // Top-down view: forward = target - eye = (0,-1,0), which is
        // anti-parallel to `up`. This used to make `forward.cross(&up)`
        // the zero vector, and normalizing it produced NaN for every
        // projected point.
        let cam = SvgCamera {
            cx: 256.0,
            cy: 256.0,
            focal: 700.0,
            eye: [0.0, 1.0, 0.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
        };
        let (sx, sy) = cam
            .project([0.05, 0.0, 0.0])
            .expect("degenerate `up` should recover via the fallback axis, not fail");
        assert!(sx.is_finite(), "sx must not be NaN/Inf, got {sx}");
        assert!(sy.is_finite(), "sy must not be NaN/Inf, got {sy}");
    }

    #[test]
    fn test_render_wireframe_degenerate_camera_returns_err() {
        let mesh = tetrahedron();
        let cam = SvgCamera {
            cx: 256.0,
            cy: 256.0,
            focal: 700.0,
            eye: [0.0, 0.0, 0.6],
            target: [0.0, 0.0, 0.6],
            up: [0.0, 1.0, 0.0],
        };
        let opts = WireframeOptions::default();
        let result = render_wireframe(&mesh, &cam, &opts);
        assert!(
            result.is_err(),
            "degenerate eye==target camera should return Err, not a NaN-filled SVG"
        );
    }

    // -----------------------------------------------------------------------
    // WireframeOptions tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_wireframe_options_default_values() {
        let opts = WireframeOptions::default();
        assert_eq!(opts.image_size, 512);
        assert_eq!(opts.edge_color, "#333333");
        assert!((opts.stroke_width - 0.5).abs() < f32::EPSILON);
        assert_eq!(opts.background_color, "white");
        assert!(!opts.show_vertices);
        assert!((opts.vertex_radius - 1.5).abs() < f32::EPSILON);
        assert_eq!(opts.vertex_color, "#0055aa");
        assert!(opts.cull_backfaces);
    }

    // -----------------------------------------------------------------------
    // render_wireframe tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_render_wireframe_tetrahedron_returns_ok() {
        let mesh = tetrahedron();
        let cam = SvgCamera::front_view(512);
        let opts = WireframeOptions::default();
        let result = render_wireframe(&mesh, &cam, &opts);
        assert!(result.is_ok(), "should succeed: {:?}", result.err());
    }

    #[test]
    fn test_render_wireframe_svg_starts_and_ends_correctly() {
        let mesh = tetrahedron();
        let cam = SvgCamera::front_view(512);
        let opts = WireframeOptions::default();
        let svg = render_wireframe(&mesh, &cam, &opts).expect("render should succeed");
        assert!(svg.starts_with("<svg"), "SVG must start with <svg");
        assert!(svg.ends_with("</svg>"), "SVG must end with </svg>");
    }

    #[test]
    fn test_render_wireframe_contains_line_elements() {
        let mesh = tetrahedron();
        let cam = SvgCamera::front_view(512);
        // Disable culling so we definitely get edges.
        let opts = WireframeOptions {
            cull_backfaces: false,
            ..Default::default()
        };
        let svg = render_wireframe(&mesh, &cam, &opts).expect("render should succeed");
        assert!(
            svg.contains("<line"),
            "SVG should contain at least one <line> element"
        );
    }

    #[test]
    fn test_render_wireframe_empty_mesh_returns_ok() {
        let mesh = Mesh::new(vec![], vec![]);
        let cam = SvgCamera::front_view(512);
        let opts = WireframeOptions::default();
        let result = render_wireframe(&mesh, &cam, &opts);
        assert!(result.is_ok(), "empty mesh should not error");
        let svg = result.expect("render should succeed");
        assert!(
            svg.starts_with("<svg"),
            "empty mesh SVG must still have header"
        );
        assert!(
            svg.ends_with("</svg>"),
            "empty mesh SVG must still end correctly"
        );
        assert!(!svg.contains("<line"), "empty mesh should have no edges");
    }

    #[test]
    fn test_render_wireframe_with_vertices_shown() {
        let mesh = tetrahedron();
        let cam = SvgCamera::front_view(512);
        let opts = WireframeOptions {
            show_vertices: true,
            cull_backfaces: false,
            ..Default::default()
        };
        let svg = render_wireframe(&mesh, &cam, &opts).expect("render should succeed");
        assert!(
            svg.contains("<circle"),
            "SVG should contain vertex circles when show_vertices=true"
        );
    }

    // -----------------------------------------------------------------------
    // render_joints_svg tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_render_joints_svg_returns_circles() {
        let joints: &[[f32; 3]] = &[[0.0, 0.1, 0.1], [-0.05, -0.03, 0.05], [0.05, -0.03, 0.05]];
        let cam = SvgCamera::front_view(512);
        let result = render_joints_svg(joints, None, &cam, 512);
        assert!(result.is_ok(), "should succeed: {:?}", result.err());
        let svg = result.expect("should succeed");
        assert!(
            svg.contains("<circle"),
            "SVG should contain circle elements for joints"
        );
    }

    #[test]
    fn test_render_joints_svg_with_names_includes_text() {
        let joints: &[[f32; 3]] = &[[0.0, 0.0, 0.1], [0.02, 0.0, 0.05]];
        let names: &[&str] = &["root", "neck"];
        let cam = SvgCamera::front_view(512);
        let svg = render_joints_svg(joints, Some(names), &cam, 512).expect("should succeed");
        assert!(
            svg.contains("<text"),
            "SVG should contain <text> for joint labels"
        );
        assert!(
            svg.contains("root"),
            "SVG should contain the joint name 'root'"
        );
        assert!(
            svg.contains("neck"),
            "SVG should contain the joint name 'neck'"
        );
    }

    #[test]
    fn test_render_joints_svg_mismatched_names_errors() {
        let joints: &[[f32; 3]] = &[[0.0, 0.0, 0.1], [0.0, 0.0, 0.05]];
        let names: &[&str] = &["only_one"];
        let cam = SvgCamera::front_view(512);
        let result = render_joints_svg(joints, Some(names), &cam, 512);
        assert!(result.is_err(), "mismatched names should return Err");
    }

    #[test]
    fn test_render_joints_svg_special_chars_escaped() {
        let joints: &[[f32; 3]] = &[[0.0, 0.0, 0.1]];
        let names: &[&str] = &["<root & neck>"];
        let cam = SvgCamera::front_view(512);
        let svg = render_joints_svg(joints, Some(names), &cam, 512).expect("should succeed");
        // The raw < should not appear unescaped in SVG text.
        assert!(
            !svg.contains("<root"),
            "unescaped '<' must not appear in SVG"
        );
        assert!(svg.contains("&lt;root"), "escaped '<root' must appear");
        assert!(svg.contains("&amp;"), "escaped '&' must appear");
    }

    // -----------------------------------------------------------------------
    // save_svg tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_save_svg_writes_file_matching_input() {
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"64\" height=\"64\"><rect width=\"64\" height=\"64\" fill=\"white\"/></svg>";
        let dir = std::env::temp_dir();
        let path = dir.join("oxigaf_test_save_svg.svg");
        save_svg(svg, &path).expect("save should succeed");
        let contents = std::fs::read_to_string(&path).expect("read back should succeed");
        assert_eq!(
            contents, svg,
            "file contents must match input string exactly"
        );
        // Clean up.
        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // render_mesh_with_joints tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_render_mesh_with_joints_combines_elements() {
        let mesh = tetrahedron();
        let joints: &[[f32; 3]] = &[[0.0, 0.0, 0.15]];
        let cam = SvgCamera::front_view(512);
        let opts = WireframeOptions {
            cull_backfaces: false,
            ..Default::default()
        };
        let svg = render_mesh_with_joints(&mesh, joints, &cam, &opts).expect("should succeed");
        assert!(
            svg.contains("<line"),
            "combined SVG must contain wireframe edges"
        );
        assert!(
            svg.contains("<circle"),
            "combined SVG must contain joint circles"
        );
    }

    #[test]
    fn test_render_mesh_with_joints_svg_structure() {
        let mesh = tetrahedron();
        let joints: &[[f32; 3]] = &[[0.0, 0.0, 0.15]];
        let cam = SvgCamera::front_view(512);
        let opts = WireframeOptions::default();
        let svg = render_mesh_with_joints(&mesh, joints, &cam, &opts).expect("should succeed");
        assert!(svg.starts_with("<svg"), "must start with <svg");
        assert!(svg.ends_with("</svg>"), "must end with </svg>");
    }
}
