//! CPU depth map rendering from FLAME meshes.
//!
//! Provides software z-buffer rasterization of FLAME triangle meshes into
//! depth images suitable for conditioning diffusion models on geometry.
//!
//! # Example
//!
//! ```rust,no_run
//! use oxigaf_flame::{Mesh, depth_estimation::{DepthConfig, front_depth_camera, render_depth_map}};
//!
//! # fn example(mesh: &Mesh) -> Result<(), oxigaf_flame::depth_estimation::DepthError> {
//! let config = DepthConfig::default();
//! let camera = front_depth_camera(config.width, config.height, 0.6, config.near_plane, config.far_plane);
//! let depth_map = render_depth_map(mesh, &camera, &config)?;
//! let coverage = depth_map.coverage(config.background_depth);
//! println!("Depth map coverage: {:.1}%", coverage * 100.0);
//! # Ok(()) }
//! ```

use nalgebra as na;
use thiserror::Error;

use crate::mesh::Mesh;
use crate::normal_map::Camera;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by depth estimation functions.
#[derive(Debug, Error)]
pub enum DepthError {
    /// The mesh has no faces — nothing to rasterize.
    #[error("Mesh has no faces")]
    EmptyMesh,

    /// Image dimensions are invalid (zero width or height).
    #[error("Invalid image dimensions: {width}×{height}")]
    InvalidDimensions { width: u32, height: u32 },

    /// The near/far depth range is inconsistent.
    #[error("Invalid depth range: near {near} must be < far {far}")]
    InvalidDepthRange { near: f32, far: f32 },

    /// A face references a vertex that does not exist in the mesh.
    #[error("Vertex index {idx} out of range (mesh has {n_vertices} vertices)")]
    VertexIndexOutOfRange { idx: u32, n_vertices: usize },

    /// The two depth maps have incompatible dimensions.
    #[error("Depth map size mismatch: ({aw}×{ah}) vs ({bw}×{bh})")]
    SizeMismatch { aw: u32, ah: u32, bw: u32, bh: u32 },
}

// ---------------------------------------------------------------------------
// DepthConfig
// ---------------------------------------------------------------------------

/// Configuration for depth map rendering.
#[derive(Debug, Clone)]
pub struct DepthConfig {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Near clipping plane (world units).
    pub near_plane: f32,
    /// Far clipping plane (world units).
    pub far_plane: f32,
    /// Depth value written to pixels with no geometry hit.
    ///
    /// Defaults to `far_plane`. Use `f32::INFINITY` if preferred.
    pub background_depth: f32,
}

impl Default for DepthConfig {
    fn default() -> Self {
        Self {
            width: 256,
            height: 256,
            near_plane: 0.01,
            far_plane: 5.0,
            background_depth: 5.0,
        }
    }
}

impl DepthConfig {
    /// Validate that all configuration values are sensible.
    ///
    /// # Errors
    ///
    /// Returns [`DepthError::InvalidDimensions`] if width or height is zero.
    /// Returns [`DepthError::InvalidDepthRange`] if `near_plane >= far_plane`.
    pub fn validate(&self) -> Result<(), DepthError> {
        if self.width == 0 || self.height == 0 {
            return Err(DepthError::InvalidDimensions {
                width: self.width,
                height: self.height,
            });
        }
        if self.near_plane >= self.far_plane {
            return Err(DepthError::InvalidDepthRange {
                near: self.near_plane,
                far: self.far_plane,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DepthMap
// ---------------------------------------------------------------------------

/// A per-pixel depth image produced by software rasterization.
///
/// Pixel values are depth in camera-space Z (world units). Background
/// pixels — those not covered by any triangle — are set to
/// [`DepthConfig::background_depth`].
#[derive(Debug, Clone)]
pub struct DepthMap {
    /// Per-pixel depth values in row-major order (row 0 = top of image).
    pub data: Vec<f32>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl DepthMap {
    /// Create a new depth map filled with `fill`.
    #[must_use]
    pub fn new(width: u32, height: u32, fill: f32) -> Self {
        let n = (width as usize).saturating_mul(height as usize);
        Self {
            data: vec![fill; n],
            width,
            height,
        }
    }

    /// Get the depth at pixel `(x, y)`, clamping indices to image bounds.
    #[inline]
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> f32 {
        let cx = x.min(self.width.saturating_sub(1));
        let cy = y.min(self.height.saturating_sub(1));
        let idx = cy as usize * self.width as usize + cx as usize;
        if idx < self.data.len() {
            self.data[idx]
        } else {
            0.0
        }
    }

    /// Set the depth at pixel `(x, y)`.
    ///
    /// Does nothing if `(x, y)` is out of bounds.
    #[inline]
    pub fn set_pixel(&mut self, x: u32, y: u32, depth: f32) {
        if x < self.width && y < self.height {
            let idx = y as usize * self.width as usize + x as usize;
            self.data[idx] = depth;
        }
    }

    /// Linearly remap depths into `[0, 1]` based on the observed min/max.
    ///
    /// If all pixels are at the same depth (including the all-background
    /// case), returns a zero-filled map.
    #[must_use]
    pub fn normalize(&self) -> Self {
        let min = self.data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;

        if range < 1e-12 {
            return Self::new(self.width, self.height, 0.0);
        }

        let inv_range = 1.0 / range;
        let normalized: Vec<f32> = self.data.iter().map(|&d| (d - min) * inv_range).collect();

        Self {
            data: normalized,
            width: self.width,
            height: self.height,
        }
    }

    /// Convert to 8-bit grayscale: near=255 (white), far=0 (black).
    ///
    /// Internally normalizes the map (min→1.0, max→0.0) then encodes as u8.
    #[must_use]
    pub fn to_grayscale_u8(&self) -> Vec<u8> {
        let min = self.data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;

        if range < 1e-12 {
            return vec![0u8; self.data.len()];
        }

        let inv_range = 1.0 / range;
        self.data
            .iter()
            .map(|&d| {
                // Near is white (255), far is black (0)
                let normalized = 1.0 - (d - min) * inv_range;
                (normalized.clamp(0.0, 1.0) * 255.0) as u8
            })
            .collect()
    }

    /// Convert to RGBA bytes (grayscale replicated into R=G=B, A=255).
    #[must_use]
    pub fn to_rgba_u8(&self) -> Vec<u8> {
        let gray = self.to_grayscale_u8();
        let mut rgba = Vec::with_capacity(gray.len() * 4);
        for &g in &gray {
            rgba.push(g);
            rgba.push(g);
            rgba.push(g);
            rgba.push(255u8);
        }
        rgba
    }

    /// Fraction of pixels whose depth is not equal to `background`.
    ///
    /// Uses exact float comparison (suitable when background is set from
    /// a constant such as `config.background_depth`).
    #[must_use]
    pub fn coverage(&self, background: f32) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let margin = f32::EPSILON * background.abs().max(1.0);
        let fg_count = self
            .data
            .iter()
            .filter(|&&d| (d - background).abs() > margin)
            .count();
        fg_count as f32 / self.data.len() as f32
    }
}

// ---------------------------------------------------------------------------
// DepthStats
// ---------------------------------------------------------------------------

/// Statistics computed over a [`DepthMap`].
#[derive(Debug, Clone)]
pub struct DepthStats {
    /// Minimum depth among foreground pixels.
    pub min_depth: f32,
    /// Maximum depth among foreground pixels.
    pub max_depth: f32,
    /// Mean depth over foreground pixels only.
    pub mean_depth: f32,
    /// Fraction of pixels with geometry (foreground).
    pub coverage: f32,
    /// `max_depth - min_depth`.
    pub depth_range: f32,
}

/// Compute per-map statistics, treating `background` as the sentinel value.
#[must_use]
pub fn compute_depth_stats(map: &DepthMap, background: f32) -> DepthStats {
    let mut min_d = f32::INFINITY;
    let mut max_d = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut count = 0usize;

    let bg_margin = f32::EPSILON * background.abs().max(1.0);
    for &d in &map.data {
        if (d - background).abs() > bg_margin {
            if d < min_d {
                min_d = d;
            }
            if d > max_d {
                max_d = d;
            }
            sum += f64::from(d);
            count += 1;
        }
    }

    let total = map.data.len();
    let coverage = if total > 0 {
        count as f32 / total as f32
    } else {
        0.0
    };

    if count == 0 {
        DepthStats {
            min_depth: 0.0,
            max_depth: 0.0,
            mean_depth: 0.0,
            coverage: 0.0,
            depth_range: 0.0,
        }
    } else {
        let mean = (sum / count as f64) as f32;
        DepthStats {
            min_depth: min_d,
            max_depth: max_d,
            mean_depth: mean,
            coverage,
            depth_range: max_d - min_d,
        }
    }
}

// ---------------------------------------------------------------------------
// Camera helpers
// ---------------------------------------------------------------------------

/// Build a front-facing perspective camera at the given distance.
///
/// Rotation = identity (camera looks in +Z world direction).
/// Translation = `[0, 0, distance]` so a head at world origin is at
/// camera-space Z = distance.
/// Focal length = `width * 1.5`.
///
/// `near`/`far` become the camera's own clip planes (consulted by
/// [`project_point`]); pass the same `near_plane`/`far_plane` you use for
/// the [`DepthConfig`] given to [`render_depth_map`] so the two clipping
/// stages agree.
#[must_use]
pub fn front_depth_camera(width: u32, height: u32, distance: f32, near: f32, far: f32) -> Camera {
    let focal = width as f32 * 1.5;
    Camera {
        rotation: na::Matrix3::identity(),
        translation: na::Vector3::new(0.0, 0.0, distance),
        focal_x: focal,
        focal_y: focal,
        cx: width as f32 / 2.0,
        cy: height as f32 / 2.0,
        width,
        height,
        near,
        far,
    }
}

/// Build a side-view camera looking from the subject's left (-X direction).
///
/// Rotation = -90° around Y: maps +Z world → -X camera and +X world → +Z camera.
/// With `t = [0, 0, distance]`, a head at world origin sits at camera Z = distance.
///
/// Rotation matrix (R_y(-90°)):
/// ```text
/// [ 0,  0, -1 ]
/// [ 0,  1,  0 ]
/// [ 1,  0,  0 ]
/// ```
/// Check: R * \[0,0,0\] + t = [0, 0, distance]. ✓
/// Head at world origin → cam = R * \[0,0,0\] + \[0,0,d\] = [0, 0, d]. ✓
/// Axis check: R * \[0,0,1\] = \[-1,0,0\] (+Z world → -X camera);
/// R * \[1,0,0\] = \[0,0,1\] (+X world → +Z camera).
///
/// See [`front_depth_camera`] for the meaning of `near`/`far`.
#[must_use]
pub fn side_depth_camera(width: u32, height: u32, distance: f32, near: f32, far: f32) -> Camera {
    let focal = width as f32 * 1.5;
    // R_y(-90°): rotates -90 degrees around Y axis.
    // R[row][col]: col-major storage, but na::Matrix3::new is row-major input.
    let rotation = na::Matrix3::new(0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    Camera {
        rotation,
        translation: na::Vector3::new(0.0, 0.0, distance),
        focal_x: focal,
        focal_y: focal,
        cx: width as f32 / 2.0,
        cy: height as f32 / 2.0,
        width,
        height,
        near,
        far,
    }
}

/// Build a three-quarter view camera (45° yaw looking from front-left).
///
/// Rotation = -45° around Y.
/// ```text
/// R_y(-45°):
/// [  cos45,  0, -sin45 ]     [ √2/2,  0, -√2/2 ]
/// [     0,   1,      0 ]  =  [   0,  1,     0  ]
/// [  sin45,  0,  cos45 ]     [ √2/2,  0,  √2/2 ]
/// ```
/// Check: R * \[0,0,0\] + t = \[0,0,d\]. ✓
///
/// See [`front_depth_camera`] for the meaning of `near`/`far`.
#[must_use]
pub fn three_quarter_depth_camera(
    width: u32,
    height: u32,
    distance: f32,
    near: f32,
    far: f32,
) -> Camera {
    let focal = width as f32 * 1.5;
    let c = std::f32::consts::FRAC_1_SQRT_2; // cos(45°) = sin(45°) = 1/√2
    let rotation = na::Matrix3::new(c, 0.0, -c, 0.0, 1.0, 0.0, c, 0.0, c);
    Camera {
        rotation,
        translation: na::Vector3::new(0.0, 0.0, distance),
        focal_x: focal,
        focal_y: focal,
        cx: width as f32 / 2.0,
        cy: height as f32 / 2.0,
        width,
        height,
        near,
        far,
    }
}

// ---------------------------------------------------------------------------
// project_point
// ---------------------------------------------------------------------------

/// Project a 3D world point to screen coordinates.
///
/// Returns `(screen_x, screen_y, depth_z)` where `depth_z` is Z in camera
/// space (positive = in front of camera). Returns `None` if the point is
/// at or behind the near plane.
#[inline]
#[must_use]
pub fn project_point(camera: &Camera, world_pos: [f32; 3]) -> Option<(f32, f32, f32)> {
    let p_world = na::Point3::new(world_pos[0], world_pos[1], world_pos[2]);
    let p_cam = camera.world_to_cam(&p_world);

    if p_cam.z <= camera.near {
        return None;
    }

    let sx = camera.focal_x * p_cam.x / p_cam.z + camera.cx;
    let sy = camera.focal_y * p_cam.y / p_cam.z + camera.cy;

    Some((sx, sy, p_cam.z))
}

// ---------------------------------------------------------------------------
// render_depth_map
// ---------------------------------------------------------------------------

/// Render a depth map from a FLAME mesh using software z-buffer rasterization.
///
/// # Algorithm
///
/// 1. Validate inputs.
/// 2. Initialize z-buffer to `far_plane`, output to `background_depth`.
/// 3. For each triangle face, project vertices to screen.
/// 4. Rasterize pixels within the bounding box using barycentric coordinates.
/// 5. Z-test and write depth for visible pixels.
///
/// # Errors
///
/// Returns [`DepthError::EmptyMesh`] if the mesh has no faces.
/// Returns [`DepthError::InvalidDimensions`] / [`DepthError::InvalidDepthRange`]
/// if the config is invalid.
/// Returns [`DepthError::VertexIndexOutOfRange`] if a face contains an invalid
/// vertex index.
pub fn render_depth_map(
    mesh: &Mesh,
    camera: &Camera,
    config: &DepthConfig,
) -> Result<DepthMap, DepthError> {
    config.validate()?;

    if mesh.faces.is_empty() {
        return Err(DepthError::EmptyMesh);
    }

    // Validate vertex indices upfront.
    let n_verts = mesh.vertices.len();
    for face in &mesh.faces {
        for &idx in face {
            if idx as usize >= n_verts {
                return Err(DepthError::VertexIndexOutOfRange {
                    idx,
                    n_vertices: n_verts,
                });
            }
        }
    }

    let w = config.width as usize;
    let h = config.height as usize;

    // z-buffer stores the minimum camera-space Z seen per pixel.
    let mut z_buf = vec![config.far_plane; w * h];
    // Output depth map (initialized to background).
    let mut depth_map = DepthMap::new(config.width, config.height, config.background_depth);

    let w_f = config.width as f32;
    let h_f = config.height as f32;

    for face in &mesh.faces {
        let (i0, i1, i2) = (face[0] as usize, face[1] as usize, face[2] as usize);

        let v0 = &mesh.vertices[i0];
        let v1 = &mesh.vertices[i1];
        let v2 = &mesh.vertices[i2];

        // Project to screen; skip face if any vertex is behind camera.
        let Some((x0, y0, d0)) = project_point(camera, [v0.x, v0.y, v0.z]) else {
            continue;
        };
        let Some((x1, y1, d1)) = project_point(camera, [v1.x, v1.y, v1.z]) else {
            continue;
        };
        let Some((x2, y2, d2)) = project_point(camera, [v2.x, v2.y, v2.z]) else {
            continue;
        };

        // Bounding box clamped to image.
        let bb_min_x = x0.min(x1).min(x2).max(0.0).floor() as i32;
        let bb_max_x = x0.max(x1).max(x2).min(w_f - 1.0).ceil() as i32;
        let bb_min_y = y0.min(y1).min(y2).max(0.0).floor() as i32;
        let bb_max_y = y0.max(y1).max(y2).min(h_f - 1.0).ceil() as i32;

        if bb_min_x > bb_max_x || bb_min_y > bb_max_y {
            continue;
        }

        // Precompute denominator for barycentric coordinates.
        let denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
        if denom.abs() < 1e-10 {
            // Degenerate triangle.
            continue;
        }
        let inv_denom = 1.0 / denom;

        for py in bb_min_y..=bb_max_y {
            for px in bb_min_x..=bb_max_x {
                let ppx = px as f32 + 0.5;
                let ppy = py as f32 + 0.5;

                let u = ((y1 - y2) * (ppx - x2) + (x2 - x1) * (ppy - y2)) * inv_denom;
                let v = ((y2 - y0) * (ppx - x2) + (x0 - x2) * (ppy - y2)) * inv_denom;
                let wt = 1.0 - u - v;

                if u < 0.0 || v < 0.0 || wt < 0.0 {
                    continue;
                }

                let depth = u * d0 + v * d1 + wt * d2;

                if depth <= config.near_plane {
                    continue;
                }

                let buf_idx = py as usize * w + px as usize;
                if depth < z_buf[buf_idx] {
                    z_buf[buf_idx] = depth;
                    depth_map.set_pixel(px as u32, py as u32, depth);
                }
            }
        }
    }

    Ok(depth_map)
}

// ---------------------------------------------------------------------------
// depth_to_point_cloud
// ---------------------------------------------------------------------------

/// Back-project a depth map to a 3D point cloud in world space.
///
/// Each foreground pixel (depth != `background`) is unprojected through
/// the camera model back into world coordinates.
///
/// # Projection model
///
/// `cam_x = (px - cx) * depth / focal_x`
/// `cam_y = (py - cy) * depth / focal_y`
/// `cam_z = depth`
/// `world = R^T * (cam - t)`
#[must_use]
pub fn depth_to_point_cloud(map: &DepthMap, camera: &Camera, background: f32) -> Vec<[f32; 3]> {
    let rot_t = camera.rotation.transpose();
    let mut points = Vec::new();

    for py in 0..map.height {
        for px in 0..map.width {
            let depth = map.pixel(px, py);
            let depth_margin = f32::EPSILON * background.abs().max(1.0);
            if (depth - background).abs() < depth_margin {
                continue;
            }

            let cam_x = (px as f32 + 0.5 - camera.cx) * depth / camera.focal_x;
            let cam_y = (py as f32 + 0.5 - camera.cy) * depth / camera.focal_y;
            let cam_z = depth;

            let cam_point = na::Vector3::new(cam_x, cam_y, cam_z);
            let world = rot_t * (cam_point - camera.translation);

            points.push([world.x, world.y, world.z]);
        }
    }

    points
}

// ---------------------------------------------------------------------------
// render_conditioning_depth_maps
// ---------------------------------------------------------------------------

/// Render standard front and side depth maps for diffusion model conditioning.
///
/// Uses cameras at distance `0.6` — the same default as [`Camera::default_front`].
///
/// # Errors
///
/// Propagates any error from [`render_depth_map`].
pub fn render_conditioning_depth_maps(
    mesh: &Mesh,
    config: &DepthConfig,
) -> Result<(DepthMap, DepthMap), DepthError> {
    let front_cam = front_depth_camera(
        config.width,
        config.height,
        0.6,
        config.near_plane,
        config.far_plane,
    );
    let side_cam = side_depth_camera(
        config.width,
        config.height,
        0.6,
        config.near_plane,
        config.far_plane,
    );

    let front_map = render_depth_map(mesh, &front_cam, config)?;
    let side_map = render_depth_map(mesh, &side_cam, config)?;

    Ok((front_map, side_map))
}

// ---------------------------------------------------------------------------
// blend_depth_maps
// ---------------------------------------------------------------------------

/// Linear blend of two depth maps: `result = a * alpha + b * (1 - alpha)`.
///
/// `alpha` is clamped to `[0, 1]`.
///
/// # Errors
///
/// Returns [`DepthError::SizeMismatch`] if the maps have different dimensions.
pub fn blend_depth_maps(a: &DepthMap, b: &DepthMap, alpha: f32) -> Result<DepthMap, DepthError> {
    if a.width != b.width || a.height != b.height {
        return Err(DepthError::SizeMismatch {
            aw: a.width,
            ah: a.height,
            bw: b.width,
            bh: b.height,
        });
    }

    let alpha = alpha.clamp(0.0, 1.0);
    let one_minus_alpha = 1.0 - alpha;

    let blended: Vec<f32> = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(&da, &db)| da * alpha + db * one_minus_alpha)
        .collect();

    Ok(DepthMap {
        data: blended,
        width: a.width,
        height: a.height,
    })
}

// ---------------------------------------------------------------------------
// depth_discontinuity_map
// ---------------------------------------------------------------------------

/// Compute a boolean discontinuity mask from a depth map.
///
/// A pixel is marked as a discontinuity if any 4-connected neighbor has a
/// depth difference exceeding `threshold`. Background pixels adjacent to
/// foreground are also counted as discontinuities.
///
/// The background value is inferred as any pixel equal to the most common
/// extreme value. Callers should pass the rendered `background_depth` from
/// [`DepthConfig`] to keep semantics clear; this function simply uses
/// numerical comparison — background pixels are those returned unchanged
/// after [`render_depth_map`] (i.e. equal to `config.background_depth`).
///
/// # Parameters
///
/// - `map`: The depth map to process.
/// - `threshold`: Depth difference in world units that counts as a discontinuity.
#[must_use]
pub fn depth_discontinuity_map(map: &DepthMap, threshold: f32) -> Vec<bool> {
    let w = map.width as usize;
    let h = map.height as usize;
    let n = w * h;
    let mut disc = vec![false; n];

    // 4-connected offsets: right, left, down, up (isize avoids cast_possible_wrap lint)
    let offsets: [(isize, isize); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    let w_s: isize = w.cast_signed();
    let h_s: isize = h.cast_signed();

    for py in 0..h {
        for px in 0..w {
            let idx = py * w + px;
            let d_center = map.data[idx];

            for (dx, dy) in &offsets {
                let nx: isize = px.cast_signed() + dx;
                let ny: isize = py.cast_signed() + dy;

                if nx < 0 || ny < 0 || nx >= w_s || ny >= h_s {
                    continue;
                }

                let n_idx = ny as usize * w + nx as usize;
                let d_neighbor = map.data[n_idx];

                if (d_center - d_neighbor).abs() > threshold {
                    disc[idx] = true;
                    break;
                }
            }
        }
    }

    disc
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;

    // ------------------------------------------------------------------
    // Test helpers
    // ------------------------------------------------------------------

    /// A simple unit-square mesh lying in the XY plane facing +Z.
    fn square_mesh() -> Mesh {
        let vertices = vec![
            na::Point3::new(-0.5f32, -0.5, 0.0),
            na::Point3::new(0.5f32, -0.5, 0.0),
            na::Point3::new(0.5f32, 0.5, 0.0),
            na::Point3::new(-0.5f32, 0.5, 0.0),
        ];
        // Two triangles covering the unit square.
        let faces = vec![[0u32, 1, 2], [0, 2, 3]];
        Mesh::new(vertices, faces)
    }

    /// A front-facing camera at distance 1.0, 64×64 image, whose clip
    /// planes match `default_test_config()` (see `test_camera_and_config_clip_planes_agree`).
    fn test_camera() -> Camera {
        let cfg = default_test_config();
        front_depth_camera(64, 64, 1.0, cfg.near_plane, cfg.far_plane)
    }

    fn default_test_config() -> DepthConfig {
        DepthConfig {
            width: 64,
            height: 64,
            near_plane: 0.01,
            far_plane: 5.0,
            background_depth: 5.0,
        }
    }

    // ------------------------------------------------------------------
    // DepthConfig tests
    // ------------------------------------------------------------------

    #[test]
    fn depth_config_default_values() {
        let cfg = DepthConfig::default();
        assert_eq!(cfg.width, 256);
        assert_eq!(cfg.height, 256);
        assert!((cfg.near_plane - 0.01).abs() < 1e-6);
        assert!((cfg.far_plane - 5.0).abs() < 1e-6);
        assert!((cfg.background_depth - 5.0).abs() < 1e-6);
    }

    #[test]
    fn depth_config_validate_valid() {
        let cfg = DepthConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn depth_config_validate_zero_width() {
        let cfg = DepthConfig {
            width: 0,
            height: 256,
            ..DepthConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(DepthError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn depth_config_validate_zero_height() {
        let cfg = DepthConfig {
            width: 256,
            height: 0,
            ..DepthConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(DepthError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn depth_config_validate_near_ge_far() {
        let cfg = DepthConfig {
            near_plane: 2.0,
            far_plane: 1.0,
            ..DepthConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(DepthError::InvalidDepthRange { .. })
        ));
    }

    #[test]
    fn depth_config_validate_near_equals_far() {
        let cfg = DepthConfig {
            near_plane: 1.0,
            far_plane: 1.0,
            ..DepthConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(DepthError::InvalidDepthRange { .. })
        ));
    }

    // ------------------------------------------------------------------
    // DepthMap tests
    // ------------------------------------------------------------------

    #[test]
    fn depth_map_new_fills_correctly() {
        let dm = DepthMap::new(4, 4, 3.1);
        assert_eq!(dm.data.len(), 16);
        for &v in &dm.data {
            assert!((v - 3.1).abs() < 1e-6);
        }
    }

    #[test]
    fn depth_map_pixel_bounds_clamped() {
        let dm = DepthMap::new(4, 4, 1.0);
        // Out-of-bounds access should clamp.
        let v = dm.pixel(100, 100);
        assert!((v - 1.0).abs() < 1e-6);
    }

    #[test]
    fn depth_map_set_pixel_and_retrieve() {
        let mut dm = DepthMap::new(8, 8, 5.0);
        dm.set_pixel(3, 3, 1.5);
        assert!((dm.pixel(3, 3) - 1.5).abs() < 1e-6);
        // Surrounding pixels unchanged.
        assert!((dm.pixel(2, 3) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn depth_map_set_pixel_out_of_bounds_is_noop() {
        let mut dm = DepthMap::new(4, 4, 0.0);
        dm.set_pixel(100, 100, 99.0); // Should not panic or modify anything.
        assert_eq!(dm.data.iter().filter(|&&v| v != 0.0).count(), 0);
    }

    #[test]
    fn depth_map_normalize_uniform_returns_zeros() {
        let dm = DepthMap::new(4, 4, 2.0);
        let norm = dm.normalize();
        for &v in &norm.data {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn depth_map_normalize_two_values() {
        let mut dm = DepthMap::new(2, 1, 0.0);
        dm.data[0] = 1.0;
        dm.data[1] = 3.0;
        let norm = dm.normalize();
        assert!((norm.data[0] - 0.0).abs() < 1e-6);
        assert!((norm.data[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn depth_map_to_grayscale_u8_near_white_far_black() {
        let mut dm = DepthMap::new(2, 1, 0.0);
        dm.data[0] = 1.0; // near → white
        dm.data[1] = 5.0; // far → black
        let gray = dm.to_grayscale_u8();
        assert_eq!(gray[0], 255, "near pixel should be white");
        assert_eq!(gray[1], 0, "far pixel should be black");
    }

    #[test]
    fn depth_map_to_rgba_u8_format() {
        let dm = DepthMap::new(1, 1, 2.0);
        let rgba = dm.to_rgba_u8();
        assert_eq!(rgba.len(), 4);
        assert_eq!(rgba[3], 255, "alpha channel must be 255");
        assert_eq!(rgba[0], rgba[1]);
        assert_eq!(rgba[1], rgba[2]);
    }

    #[test]
    fn depth_map_coverage_all_background() {
        let dm = DepthMap::new(4, 4, 5.0);
        assert!((dm.coverage(5.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn depth_map_coverage_partial() {
        let mut dm = DepthMap::new(4, 1, 5.0);
        dm.data[0] = 1.0;
        dm.data[1] = 2.0;
        let cov = dm.coverage(5.0);
        assert!((cov - 0.5).abs() < 1e-6, "2/4 = 0.5, got {cov}");
    }

    // ------------------------------------------------------------------
    // compute_depth_stats tests
    // ------------------------------------------------------------------

    #[test]
    fn depth_stats_all_background() {
        let dm = DepthMap::new(4, 4, 5.0);
        let stats = compute_depth_stats(&dm, 5.0);
        assert_eq!(stats.coverage, 0.0);
        assert_eq!(stats.min_depth, 0.0);
        assert_eq!(stats.max_depth, 0.0);
        assert_eq!(stats.mean_depth, 0.0);
        assert_eq!(stats.depth_range, 0.0);
    }

    #[test]
    fn depth_stats_single_foreground_pixel() {
        let mut dm = DepthMap::new(4, 4, 5.0);
        dm.data[0] = 2.0;
        let stats = compute_depth_stats(&dm, 5.0);
        assert!((stats.min_depth - 2.0).abs() < 1e-6);
        assert!((stats.max_depth - 2.0).abs() < 1e-6);
        assert!((stats.mean_depth - 2.0).abs() < 1e-6);
        assert!((stats.coverage - 1.0 / 16.0).abs() < 1e-6);
        assert!(stats.depth_range.abs() < 1e-6);
    }

    #[test]
    fn depth_stats_multiple_foreground_pixels() {
        let mut dm = DepthMap::new(1, 4, 5.0);
        dm.data[0] = 1.0;
        dm.data[1] = 2.0;
        dm.data[2] = 3.0;
        // dm.data[3] = 5.0 (background)
        let stats = compute_depth_stats(&dm, 5.0);
        assert!((stats.min_depth - 1.0).abs() < 1e-6);
        assert!((stats.max_depth - 3.0).abs() < 1e-6);
        assert!((stats.mean_depth - 2.0).abs() < 1e-4);
        assert!((stats.coverage - 0.75).abs() < 1e-6);
        assert!((stats.depth_range - 2.0).abs() < 1e-6);
    }

    #[test]
    fn depth_stats_all_foreground() {
        let mut dm = DepthMap::new(1, 3, 5.0);
        dm.data[0] = 0.5;
        dm.data[1] = 1.0;
        dm.data[2] = 1.5;
        let stats = compute_depth_stats(&dm, 5.0);
        assert!((stats.coverage - 1.0).abs() < 1e-6);
        assert!((stats.mean_depth - 1.0).abs() < 1e-5);
    }

    // ------------------------------------------------------------------
    // project_point tests
    // ------------------------------------------------------------------

    #[test]
    fn project_point_behind_camera_returns_none() {
        let cam = test_camera();
        // Point at z=0 in world, camera is at z=1 looking forward.
        // cam.z = 0 * R_identity + 1.0 = 1.0 → OK.
        // Place point very far behind: world z = 100 would be cam.z = 1 - 100 = ... no.
        // With identity rotation and translation [0,0,1]:
        //   cam.z = world.z + 1.0. Behind near = cam.z <= 0.01.
        //   Need world.z + 1.0 <= 0.01 → world.z <= -0.99.
        let result = project_point(&cam, [0.0, 0.0, -1.0]);
        assert!(result.is_none(), "point behind camera should return None");
    }

    #[test]
    fn project_point_on_axis_in_front() {
        let cam = test_camera();
        // World origin → cam = [0,0,1.0]. Should project to principal point (cx, cy).
        let result = project_point(&cam, [0.0, 0.0, 0.0]);
        assert!(result.is_some());
        let (sx, sy, depth) = result.unwrap();
        assert!((sx - cam.cx).abs() < 1e-4, "on-axis should project to cx");
        assert!((sy - cam.cy).abs() < 1e-4, "on-axis should project to cy");
        assert!((depth - 1.0).abs() < 1e-6, "depth should be 1.0");
    }

    #[test]
    fn project_point_positive_x_offset() {
        let cam = test_camera();
        // World [1, 0, 0] → cam = [1, 0, 1.0].
        // screen_x = focal_x * 1.0 / 1.0 + cx.
        let result = project_point(&cam, [1.0, 0.0, 0.0]);
        assert!(result.is_some());
        let (sx, _sy, _depth) = result.unwrap();
        let expected_x = cam.focal_x + cam.cx;
        assert!((sx - expected_x).abs() < 1e-4);
    }

    #[test]
    fn project_point_depth_equals_cam_z() {
        let cam = test_camera();
        // World [0, 0, -0.5] → cam.z = -0.5 + 1.0 = 0.5.
        let result = project_point(&cam, [0.0, 0.0, -0.5]);
        assert!(result.is_some());
        let (_, _, depth) = result.unwrap();
        assert!((depth - 0.5).abs() < 1e-6);
    }

    #[test]
    fn project_point_exactly_at_near_plane_is_none() {
        // cam.near = 0.01. World z such that cam.z == 0.01.
        // cam.z = world.z + 1.0 = 0.01 → world.z = -0.99.
        let cam = test_camera();
        let result = project_point(&cam, [0.0, 0.0, -0.99]);
        // cam.z = 0.01 == near → should be None (not strictly greater than near).
        assert!(result.is_none());
    }

    // ------------------------------------------------------------------
    // render_depth_map tests
    // ------------------------------------------------------------------

    #[test]
    fn render_depth_map_empty_mesh_error() {
        let mesh = Mesh::new(vec![], vec![]);
        let cam = test_camera();
        let cfg = default_test_config();
        let result = render_depth_map(&mesh, &cam, &cfg);
        assert!(matches!(result, Err(DepthError::EmptyMesh)));
    }

    #[test]
    fn render_depth_map_single_triangle_has_coverage() {
        // Triangle covering the center of the screen.
        let vertices = vec![
            na::Point3::new(-0.2f32, -0.2, 0.0),
            na::Point3::new(0.2f32, -0.2, 0.0),
            na::Point3::new(0.0f32, 0.2, 0.0),
        ];
        let faces = vec![[0u32, 1, 2]];
        let mesh = Mesh::new(vertices, faces);
        let cam = test_camera();
        let cfg = default_test_config();
        let dm = render_depth_map(&mesh, &cam, &cfg).expect("render should succeed");
        let cov = dm.coverage(cfg.background_depth);
        assert!(cov > 0.0, "triangle should cover some pixels");
    }

    #[test]
    fn render_depth_map_z_buffer_picks_closer_triangle() {
        // Two co-planar triangles: front at z=-0.5, back at z=-0.8 (cam space nearer).
        // With camera at [0,0,1] identity rotation:
        // z=-0.5 → cam.z=0.5, z=-0.8 → cam.z=0.2 → nearer.
        // Render back first (z=-0.8), then front (z=-0.5).
        // Front is farther in cam space (cam.z=0.5), so BACK should win for the region.
        let vertices = vec![
            // Back triangle (cam.z = 0.2) — closer to camera
            na::Point3::new(-0.3f32, -0.3, -0.8),
            na::Point3::new(0.3f32, -0.3, -0.8),
            na::Point3::new(0.0f32, 0.3, -0.8),
            // Front triangle (cam.z = 0.5) — farther from camera
            na::Point3::new(-0.3f32, -0.3, -0.5),
            na::Point3::new(0.3f32, -0.3, -0.5),
            na::Point3::new(0.0f32, 0.3, -0.5),
        ];
        let faces = vec![[3u32, 4, 5], [0, 1, 2]]; // front rendered first, then back
        let mesh = Mesh::new(vertices, faces);
        let cam = test_camera();
        let cfg = default_test_config();
        let dm = render_depth_map(&mesh, &cam, &cfg).expect("render ok");

        // Center pixel should have the nearer depth (cam.z ≈ 0.2).
        let center_depth = dm.pixel(32, 32);
        // Allow some tolerance since camera is 64x64.
        assert!(
            center_depth < 0.4,
            "center should be the nearer triangle (depth ~0.2), got {center_depth}"
        );
    }

    #[test]
    fn render_depth_map_full_quad_covers_most_of_image() {
        let mesh = square_mesh();
        let cfg = DepthConfig {
            width: 64,
            height: 64,
            near_plane: 0.01,
            far_plane: 5.0,
            background_depth: 5.0,
        };
        // Zoom in camera: large focal to make the unit square fill the frame.
        let cam = Camera {
            rotation: na::Matrix3::identity(),
            translation: na::Vector3::new(0.0, 0.0, 0.6),
            focal_x: 256.0,
            focal_y: 256.0,
            cx: 32.0,
            cy: 32.0,
            width: 64,
            height: 64,
            near: 0.01,
            far: 10.0,
        };
        let dm = render_depth_map(&mesh, &cam, &cfg).expect("render ok");
        let cov = dm.coverage(cfg.background_depth);
        assert!(
            cov > 0.5,
            "large quad should cover more than 50% of image, got {cov}"
        );
    }

    #[test]
    fn render_depth_map_invalid_config_propagates_error() {
        let mesh = square_mesh();
        let cam = test_camera();
        let cfg = DepthConfig {
            width: 0,
            height: 64,
            ..DepthConfig::default()
        };
        assert!(matches!(
            render_depth_map(&mesh, &cam, &cfg),
            Err(DepthError::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn render_depth_map_out_of_range_vertex_index() {
        // Build a mesh with 11 vertices so Mesh::new won't panic, then manually
        // replace a face to reference an out-of-range index for our validation.
        let vertices: Vec<na::Point3<f32>> = (0..11)
            .map(|i| na::Point3::new(i as f32 * 0.1, 0.0, 0.0))
            .collect();
        let faces = vec![[0u32, 1, 2]];
        let mut mesh = Mesh::new(vertices.clone(), faces);

        // Shorten the vertex list to make index 10 out-of-range (mesh has 3 now).
        mesh.vertices.truncate(3);
        mesh.normals.truncate(3);
        // Add a face that references the now-out-of-range index.
        mesh.faces.push([0u32, 1, 10]);

        let cam = test_camera();
        let cfg = default_test_config();
        assert!(matches!(
            render_depth_map(&mesh, &cam, &cfg),
            Err(DepthError::VertexIndexOutOfRange { .. })
        ));
    }

    // ------------------------------------------------------------------
    // Camera helper tests
    // ------------------------------------------------------------------

    #[test]
    fn front_depth_camera_head_at_origin_projects_to_principal_point() {
        let cam = front_depth_camera(128, 128, 0.6, 0.01, 10.0);
        // Identity rotation, t=[0,0,0.6]: cam = [0,0,0.6] for world origin.
        let result = project_point(&cam, [0.0, 0.0, 0.0]);
        assert!(result.is_some());
        let (sx, sy, depth) = result.unwrap();
        assert!((sx - 64.0).abs() < 1e-4);
        assert!((sy - 64.0).abs() < 1e-4);
        assert!((depth - 0.6).abs() < 1e-5);
    }

    #[test]
    fn side_depth_camera_head_at_origin_in_front() {
        // With side camera, world origin should be at cam.z = distance.
        let cam = side_depth_camera(128, 128, 0.6, 0.01, 10.0);
        let p_cam = cam.world_to_cam(&na::Point3::origin());
        assert!(
            p_cam.z > 0.0,
            "world origin must be in front of side camera, cam.z = {}",
            p_cam.z
        );
        assert!((p_cam.z - 0.6).abs() < 1e-5);
    }

    #[test]
    fn three_quarter_camera_head_at_origin_in_front() {
        let cam = three_quarter_depth_camera(128, 128, 0.6, 0.01, 10.0);
        let p_cam = cam.world_to_cam(&na::Point3::origin());
        assert!(
            p_cam.z > 0.0,
            "world origin must be in front of 3/4 camera, cam.z = {}",
            p_cam.z
        );
        assert!((p_cam.z - 0.6).abs() < 1e-5);
    }

    #[test]
    fn test_camera_and_config_clip_planes_agree() {
        // Regression test: the camera helpers used to hardcode
        // `near: 0.01, far: 10.0` regardless of the `DepthConfig` a caller
        // validated and passed to `render_depth_map`, so `project_point`'s
        // near clip (via `camera.near`) could silently disagree with
        // `config.near_plane`/`config.far_plane`. The camera's clip planes
        // must now match whatever config the caller supplies.
        let config = DepthConfig {
            width: 32,
            height: 32,
            near_plane: 0.25,
            far_plane: 3.0,
            background_depth: 3.0,
        };
        let front = front_depth_camera(
            config.width,
            config.height,
            1.0,
            config.near_plane,
            config.far_plane,
        );
        let side = side_depth_camera(
            config.width,
            config.height,
            1.0,
            config.near_plane,
            config.far_plane,
        );
        let three_quarter = three_quarter_depth_camera(
            config.width,
            config.height,
            1.0,
            config.near_plane,
            config.far_plane,
        );
        for cam in [&front, &side, &three_quarter] {
            assert!((cam.near - config.near_plane).abs() < 1e-6);
            assert!((cam.far - config.far_plane).abs() < 1e-6);
        }
    }

    #[test]
    fn test_side_depth_camera_axis_mapping() {
        // Regression test for the doc-comment sign error: the rotation must
        // map +Z world to -X camera and +X world to +Z camera.
        let cam = side_depth_camera(64, 64, 1.0, 0.01, 10.0);
        let plus_z_world = cam.rotation * na::Vector3::new(0.0f32, 0.0, 1.0);
        assert!(
            (plus_z_world - na::Vector3::new(-1.0, 0.0, 0.0)).norm() < 1e-6,
            "+Z world should map to -X camera, got {plus_z_world:?}"
        );
        let plus_x_world = cam.rotation * na::Vector3::new(1.0f32, 0.0, 0.0);
        assert!(
            (plus_x_world - na::Vector3::new(0.0, 0.0, 1.0)).norm() < 1e-6,
            "+X world should map to +Z camera, got {plus_x_world:?}"
        );
    }

    // ------------------------------------------------------------------
    // depth_to_point_cloud tests
    // ------------------------------------------------------------------

    #[test]
    fn depth_to_point_cloud_empty_when_all_background() {
        let dm = DepthMap::new(8, 8, 5.0);
        let cam = test_camera();
        let pts = depth_to_point_cloud(&dm, &cam, 5.0);
        assert!(pts.is_empty(), "all background → empty point cloud");
    }

    #[test]
    fn depth_to_point_cloud_center_pixel_reprojects_near_origin() {
        // Render square_mesh and back-project: result should be near the mesh.
        let mesh = square_mesh();
        let cfg = default_test_config();
        let cam = test_camera();
        let dm = render_depth_map(&mesh, &cam, &cfg).expect("render ok");
        let pts = depth_to_point_cloud(&dm, &cam, cfg.background_depth);
        // Each point should have z near 0 (mesh is at z=0 world).
        if !pts.is_empty() {
            let z_avg: f32 = pts.iter().map(|p| p[2]).sum::<f32>() / pts.len() as f32;
            assert!(
                z_avg.abs() < 0.2,
                "back-projected points should be near z=0, got avg z={z_avg}"
            );
        }
    }

    #[test]
    fn depth_to_point_cloud_count_matches_foreground_pixels() {
        let mut dm = DepthMap::new(4, 4, 5.0);
        dm.data[0] = 1.0;
        dm.data[5] = 2.0;
        dm.data[10] = 1.5;
        let cam = test_camera();
        let pts = depth_to_point_cloud(&dm, &cam, 5.0);
        assert_eq!(pts.len(), 3, "one point per foreground pixel");
    }

    // ------------------------------------------------------------------
    // render_conditioning_depth_maps tests
    // ------------------------------------------------------------------

    #[test]
    fn conditioning_maps_front_has_coverage() {
        let mesh = square_mesh();
        let cfg = default_test_config();
        let (front, _side) = render_conditioning_depth_maps(&mesh, &cfg).expect("render ok");
        assert!(
            front.coverage(cfg.background_depth) > 0.0,
            "front map must have some geometry coverage"
        );
    }

    #[test]
    fn conditioning_maps_side_has_coverage() {
        // The square mesh lies in the XY plane. From the side (looking along X),
        // it's an edge-on thin line — may have zero coverage. That's expected.
        // Instead test with a volumetric mesh.
        let vertices = vec![
            na::Point3::new(-0.3f32, -0.3, -0.3),
            na::Point3::new(0.3f32, -0.3, -0.3),
            na::Point3::new(0.3f32, 0.3, -0.3),
            na::Point3::new(-0.3f32, 0.3, 0.3),
            na::Point3::new(0.0f32, 0.0, 0.3),
        ];
        let faces = vec![
            [0u32, 1, 2],
            [0, 2, 3],
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [0, 3, 4],
        ];
        let mesh = Mesh::new(vertices, faces);
        let cfg = default_test_config();
        let (_front, side) = render_conditioning_depth_maps(&mesh, &cfg).expect("render ok");
        // Side view should also see some geometry.
        assert!(
            side.coverage(cfg.background_depth) > 0.0,
            "side map must have some geometry coverage"
        );
    }

    // ------------------------------------------------------------------
    // blend_depth_maps tests
    // ------------------------------------------------------------------

    #[test]
    fn blend_alpha_zero_returns_b() {
        let a = DepthMap::new(4, 4, 1.0);
        let b = DepthMap::new(4, 4, 3.0);
        let result = blend_depth_maps(&a, &b, 0.0).expect("blend ok");
        for &v in &result.data {
            assert!((v - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn blend_alpha_one_returns_a() {
        let a = DepthMap::new(4, 4, 1.0);
        let b = DepthMap::new(4, 4, 3.0);
        let result = blend_depth_maps(&a, &b, 1.0).expect("blend ok");
        for &v in &result.data {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn blend_alpha_half() {
        let a = DepthMap::new(4, 4, 2.0);
        let b = DepthMap::new(4, 4, 4.0);
        let result = blend_depth_maps(&a, &b, 0.5).expect("blend ok");
        for &v in &result.data {
            assert!((v - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn blend_size_mismatch_error() {
        let a = DepthMap::new(4, 4, 1.0);
        let b = DepthMap::new(8, 8, 2.0);
        assert!(matches!(
            blend_depth_maps(&a, &b, 0.5),
            Err(DepthError::SizeMismatch { .. })
        ));
    }

    // ------------------------------------------------------------------
    // depth_discontinuity_map tests
    // ------------------------------------------------------------------

    #[test]
    fn discontinuity_uniform_depth_no_edges() {
        let dm = DepthMap::new(4, 4, 2.0);
        let disc = depth_discontinuity_map(&dm, 0.1);
        assert!(
            disc.iter().all(|&d| !d),
            "uniform depth should have no discontinuities"
        );
    }

    #[test]
    fn discontinuity_step_edge_detected() {
        let mut dm = DepthMap::new(4, 1, 0.0);
        // Left two pixels at depth 1.0, right two at 5.0.
        dm.data[0] = 1.0;
        dm.data[1] = 1.0;
        dm.data[2] = 5.0;
        dm.data[3] = 5.0;
        let disc = depth_discontinuity_map(&dm, 0.5);
        // Pixels at index 1 and 2 are adjacent and differ by 4.0 > threshold.
        assert!(disc[1] || disc[2], "boundary pixels should be marked");
    }

    #[test]
    fn discontinuity_small_variation_below_threshold() {
        let mut dm = DepthMap::new(1, 4, 0.0);
        dm.data[0] = 1.0;
        dm.data[1] = 1.05;
        dm.data[2] = 1.1;
        dm.data[3] = 1.0;
        let disc = depth_discontinuity_map(&dm, 0.2);
        assert!(
            disc.iter().all(|&d| !d),
            "small variation should not trigger discontinuity"
        );
    }

    #[test]
    fn discontinuity_background_adjacent_to_foreground() {
        // 3x1: [foreground, background, background]
        let mut dm = DepthMap::new(3, 1, 0.0);
        dm.data[0] = 0.5; // foreground (cam depth)
        dm.data[1] = 5.0; // background
        dm.data[2] = 5.0; // background
        let disc = depth_discontinuity_map(&dm, 0.1);
        // Boundary between index 0 and 1 (differ by 4.5 >> threshold).
        assert!(
            disc[0] || disc[1],
            "foreground-background boundary must be discontinuity"
        );
    }
}
