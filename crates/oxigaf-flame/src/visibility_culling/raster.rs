//! Private geometry helpers and the screen-space depth-buffer rasterizer
//! used for the optional occlusion test.

use nalgebra as na;

use crate::mesh::Mesh;
use crate::normal_map::Camera;

/// Compute the unit face normal from three vertex positions.
///
/// Uses the cross product of the two edge vectors; returns `[0, 0, 0]` for
/// degenerate (zero-area) triangles.
#[inline]
pub(super) fn compute_face_normal(v0: [f32; 3], v1: [f32; 3], v2: [f32; 3]) -> [f32; 3] {
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

/// World-space position of the camera centre: `p = −Rᵀ·t`.
///
/// Constant for a given camera, so callers compute it **once** and pass it to
/// [`camera_direction`] rather than rebuilding it per vertex/face.
#[inline]
pub(super) fn camera_world_position(camera: &Camera) -> [f32; 3] {
    let p = camera.rotation.transpose() * (-camera.translation);
    [p[0], p[1], p[2]]
}

/// Compute the world-space direction from a mesh vertex toward the camera
/// origin `cam_world` (see [`camera_world_position`]).
///
/// The returned vector is normalized; returns `[0, 0, 0]` if the vertex
/// coincides with the camera.
#[inline]
pub(super) fn camera_direction(vertex: [f32; 3], cam_world: [f32; 3]) -> [f32; 3] {
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
pub(super) fn is_front_facing(face_normal: [f32; 3], view_dir: [f32; 3], threshold: f32) -> bool {
    let dot =
        face_normal[0] * view_dir[0] + face_normal[1] * view_dir[1] + face_normal[2] * view_dir[2];
    dot > threshold
}

/// Project a world-space vertex onto screen coordinates using the pinhole model.
///
/// Returns `None` if the vertex is at or behind the near clipping plane.
#[inline]
pub(super) fn project_vertex(vertex: [f32; 3], camera: &Camera) -> Option<[f32; 2]> {
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
pub(super) fn is_in_frustum(screen_pos: [f32; 2], camera: &Camera, margin: f32) -> bool {
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
pub(super) fn compute_face_screen_area(s0: [f32; 2], s1: [f32; 2], s2: [f32; 2]) -> f32 {
    let area = (s1[0] - s0[0]) * (s2[1] - s0[1]) - (s2[0] - s0[0]) * (s1[1] - s0[1]);
    (area * 0.5).abs()
}

/// Clamp a floating-point pixel coordinate into `[0, max_index]`.
#[inline]
fn clamp_pixel(value: f32, max_index: usize) -> usize {
    if !value.is_finite() {
        return 0;
    }
    value.max(0.0).min(max_index as f32) as usize
}

/// Rasterize the mesh into a screen-space depth buffer of camera-space `z`.
///
/// One `f32` per pixel, `f32::INFINITY` where no triangle covers the pixel
/// centre.  Depth is interpolated screen-linearly (`w0·z0 + w1·z1 + w2·z2`) like
/// the tile rasterizer in `normal_map` — not perspective correct, but far below
/// the depth gaps this test resolves.  Triangles with any vertex at or behind
/// the near plane are skipped.
pub(super) fn rasterize_depth_buffer(mesh: &Mesh, camera: &Camera) -> Vec<f32> {
    let width = camera.width as usize;
    let height = camera.height as usize;
    let mut depth = vec![f32::INFINITY; width * height];
    if width == 0 || height == 0 {
        return depth;
    }

    let n_verts = mesh.vertices.len();
    for face in &mesh.faces {
        let idx = [face[0] as usize, face[1] as usize, face[2] as usize];
        if idx.iter().any(|&i| i >= n_verts) {
            continue;
        }

        let cam_pts: [na::Point3<f32>; 3] = [
            camera.world_to_cam(&mesh.vertices[idx[0]]),
            camera.world_to_cam(&mesh.vertices[idx[1]]),
            camera.world_to_cam(&mesh.vertices[idx[2]]),
        ];
        if cam_pts.iter().any(|p| p.z <= camera.near) {
            continue;
        }

        let mut screen = [[0.0f32; 2]; 3];
        for (dst, p) in screen.iter_mut().zip(cam_pts.iter()) {
            *dst = [
                camera.focal_x * p.x / p.z + camera.cx,
                camera.focal_y * p.y / p.z + camera.cy,
            ];
        }

        let area = (screen[1][0] - screen[0][0]) * (screen[2][1] - screen[0][1])
            - (screen[2][0] - screen[0][0]) * (screen[1][1] - screen[0][1]);
        if area.abs() < 1e-12 || !area.is_finite() {
            continue;
        }
        let inv_area = 1.0 / area;

        let min_x = screen[0][0].min(screen[1][0]).min(screen[2][0]);
        let max_x = screen[0][0].max(screen[1][0]).max(screen[2][0]);
        let min_y = screen[0][1].min(screen[1][1]).min(screen[2][1]);
        let max_y = screen[0][1].max(screen[1][1]).max(screen[2][1]);

        let x_start = clamp_pixel(min_x.floor(), width - 1);
        let x_end = clamp_pixel(max_x.ceil(), width - 1);
        let y_start = clamp_pixel(min_y.floor(), height - 1);
        let y_end = clamp_pixel(max_y.ceil(), height - 1);

        for py in y_start..=y_end {
            let sample_y = py as f32 + 0.5;
            for px in x_start..=x_end {
                let sample_x = px as f32 + 0.5;

                let w0 = ((screen[1][0] - sample_x) * (screen[2][1] - sample_y)
                    - (screen[2][0] - sample_x) * (screen[1][1] - sample_y))
                    * inv_area;
                let w1 = ((screen[2][0] - sample_x) * (screen[0][1] - sample_y)
                    - (screen[0][0] - sample_x) * (screen[2][1] - sample_y))
                    * inv_area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }

                let z = w0 * cam_pts[0].z + w1 * cam_pts[1].z + w2 * cam_pts[2].z;
                let pixel = py * width + px;
                if z < depth[pixel] {
                    depth[pixel] = z;
                }
            }
        }
    }

    depth
}

/// Test one world-space vertex against a rasterized depth buffer.
///
/// Returns `true` when the vertex lies behind the recorded surface by more than
/// the effective tolerance `max(depth_bias, largest depth step to the four
/// neighbouring pixels)`.  Vertices outside the image, behind the near plane, or
/// on pixels no triangle covered are never reported as occluded.
pub(super) fn is_occluded(
    vertex: [f32; 3],
    camera: &Camera,
    depth: &[f32],
    depth_bias: f32,
) -> bool {
    let width = camera.width as usize;
    let height = camera.height as usize;
    if width == 0 || height == 0 {
        return false;
    }

    let p = na::Point3::new(vertex[0], vertex[1], vertex[2]);
    let p_cam = camera.world_to_cam(&p);
    if p_cam.z <= camera.near {
        return false;
    }

    let screen_x = camera.focal_x * p_cam.x / p_cam.z + camera.cx;
    let screen_y = camera.focal_y * p_cam.y / p_cam.z + camera.cy;
    if !screen_x.is_finite() || !screen_y.is_finite() || screen_x < 0.0 || screen_y < 0.0 {
        return false;
    }
    let px = screen_x as usize;
    let py = screen_y as usize;
    if px >= width || py >= height {
        return false;
    }

    let Some(&stored) = depth.get(py * width + px) else {
        return false;
    };
    if !stored.is_finite() {
        return false;
    }

    // Slope-scaled bias: how fast the surface recedes across one pixel.
    //
    // The buffer samples pixel *centres*, so a vertex may sit up to half a pixel
    // away in each axis: the sampling error is bounded by
    // `½·(|∂z/∂x| + |∂z/∂y|)`, which never exceeds the largest one-pixel step.
    // Using the full step as the tolerance therefore guarantees that a vertex
    // lying on the rasterized surface is never reported as occluding itself.
    let mut step = 0.0f32;
    for (nx, ny) in [
        (px.wrapping_sub(1), py),
        (px + 1, py),
        (px, py.wrapping_sub(1)),
        (px, py + 1),
    ] {
        if nx >= width || ny >= height {
            continue;
        }
        let neighbour = depth.get(ny * width + nx).copied().unwrap_or(f32::INFINITY);
        if neighbour.is_finite() {
            step = step.max((neighbour - stored).abs());
        }
    }

    p_cam.z > stored + depth_bias.max(step)
}
