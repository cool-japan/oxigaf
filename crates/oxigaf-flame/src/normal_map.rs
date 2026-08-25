//! CPU software rasterizer for FLAME normal maps.
//!
//! Produces an RGB image where each pixel encodes the interpolated surface
//! normal at that point: `RGB = (normal.xyz + 1) / 2 * 255`.
//!
//! ## Performance Optimizations
//!
//! This module includes several performance optimizations:
//! - SIMD-accelerated vector operations using `portable_simd`
//! - Tile-based rasterization for improved cache locality
//! - Incremental edge evaluation to reduce computation
//! - Inline hints on hot path functions
//! - Pre-computed reciprocals to avoid divisions in inner loops

use image::{Rgb, RgbImage};
use nalgebra as na;

#[cfg(all(feature = "simd", nightly))]
use std::simd::{f32x4, num::SimdFloat};

use crate::mesh::Mesh;

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

/// Simple pinhole camera for rendering.
///
/// ## Convention
///
/// World-to-camera transform is `p_cam = rotation * p_world + translation`.
/// In camera space:
/// - The camera looks down `+Z_cam` (a visible point has `p_cam.z > 0`,
///   which is exactly what the near-plane cull in
///   [`NormalMapRenderer::render`] checks).
/// - `+X_cam` points right on screen (increasing pixel `x`).
/// - `+Y_cam` points DOWN on screen (increasing pixel `y`), matching
///   [`Camera::project`]'s OpenCV-style pinhole formula
///   `u = fx·X/Z + cx`, `v = fy·Y/Z + cy` and the row-major image
///   convention where row 0 is the top of the image.
///
/// Constructing a `Camera` by struct literal must respect this convention;
/// [`Camera::default_front`] is the reference implementation.
#[derive(Debug, Clone)]
pub struct Camera {
    /// World-to-camera rotation matrix.
    pub rotation: na::Matrix3<f32>,
    /// World-to-camera translation.
    pub translation: na::Vector3<f32>,
    /// Focal length in pixels (horizontal).
    pub focal_x: f32,
    /// Focal length in pixels (vertical).
    pub focal_y: f32,
    /// Principal point x (pixels).
    pub cx: f32,
    /// Principal point y (pixels).
    pub cy: f32,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Near clipping plane.
    pub near: f32,
    /// Far clipping plane.
    pub far: f32,
}

impl Camera {
    /// Create a default front-facing camera suitable for head rendering.
    ///
    /// FLAME meshes are +Y up, +Z out of the face (toward the viewer); see
    /// the crate-level coordinate-system docs. Per the [`Camera`] convention
    /// (`+Z_cam` forward, `+Y_cam` down), the camera therefore needs
    /// `rotation = diag(1, -1, -1)`: this places the camera at world
    /// position `(0, 0, 0.6)` — in front of the face — looking back along
    /// world `-Z` toward it, with world `+Y` (up) mapping to screen "up"
    /// (decreasing pixel row) rather than being vertically mirrored.
    #[must_use]
    pub fn default_front(width: u32, height: u32) -> Self {
        let focal = width as f32 * 1.5;

        // diag(1, -1, -1): see the doc above for the derivation.
        #[rustfmt::skip]
        let rotation = na::Matrix3::new(
            1.0,  0.0,  0.0,
            0.0, -1.0,  0.0,
            0.0,  0.0, -1.0,
        );

        Self {
            rotation,
            translation: na::Vector3::new(0.0, 0.0, 0.6),
            focal_x: focal,
            focal_y: focal,
            cx: width as f32 / 2.0,
            cy: height as f32 / 2.0,
            width,
            height,
            near: 0.01,
            far: 10.0,
        }
    }

    /// Transform a world-space point to camera space.
    #[inline]
    #[must_use]
    pub fn world_to_cam(&self, p: &na::Point3<f32>) -> na::Point3<f32> {
        na::Point3::from(self.rotation * p.coords + self.translation)
    }

    /// Project a camera-space point to pixel coordinates.
    #[inline]
    #[must_use]
    pub fn project(&self, p_cam: &na::Point3<f32>) -> [f32; 2] {
        let x = self.focal_x * p_cam.x / p_cam.z + self.cx;
        let y = self.focal_y * p_cam.y / p_cam.z + self.cy;
        [x, y]
    }

    /// Project a camera-space point with pre-computed z reciprocal (optimized).
    #[inline]
    #[must_use]
    pub fn project_with_recip_z(&self, p_cam: &na::Point3<f32>, recip_z: f32) -> [f32; 2] {
        let x = self.focal_x * p_cam.x * recip_z + self.cx;
        let y = self.focal_y * p_cam.y * recip_z + self.cy;
        [x, y]
    }
}

// ---------------------------------------------------------------------------
// Helper Functions (with SIMD and non-SIMD variants)
// ---------------------------------------------------------------------------

/// Vector normalization (scalar version).
#[cfg(not(all(feature = "simd", nightly)))]
#[inline]
fn normalize_vector(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let len_sq = x * x + y * y + z * z;
    let len = len_sq.sqrt();

    if len > 1e-10 {
        let recip_len = 1.0 / len;
        (x * recip_len, y * recip_len, z * recip_len)
    } else {
        (x, y, z)
    }
}

/// SIMD-accelerated vector normalization.
#[cfg(all(feature = "simd", nightly))]
#[inline]
fn normalize_vector(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let v = f32x4::from_array([x, y, z, 0.0]);
    let len_sq = (v * v).reduce_sum();
    let len = len_sq.sqrt();

    if len > 1e-10 {
        let recip_len = 1.0 / len;
        (x * recip_len, y * recip_len, z * recip_len)
    } else {
        (x, y, z)
    }
}

/// Normal interpolation and encoding (scalar version).
#[cfg(not(all(feature = "simd", nightly)))]
#[inline]
fn interpolate_and_encode_normal(
    n0: &na::Vector3<f32>,
    n1: &na::Vector3<f32>,
    n2: &na::Vector3<f32>,
    w0: f32,
    w1: f32,
    w2: f32,
) -> [u8; 3] {
    // Interpolate
    let nx = n0.x * w0 + n1.x * w1 + n2.x * w2;
    let ny = n0.y * w0 + n1.y * w1 + n2.y * w2;
    let nz = n0.z * w0 + n1.z * w1 + n2.z * w2;

    // Normalize
    let (nx_norm, ny_norm, nz_norm) = normalize_vector(nx, ny, nz);

    // Encode: [-1, 1] -> [0, 255]
    let encode = |v: f32| ((v * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
    [encode(nx_norm), encode(ny_norm), encode(nz_norm)]
}

/// SIMD-accelerated normal interpolation and encoding.
#[cfg(all(feature = "simd", nightly))]
#[inline]
fn interpolate_and_encode_normal(
    n0: &na::Vector3<f32>,
    n1: &na::Vector3<f32>,
    n2: &na::Vector3<f32>,
    w0: f32,
    w1: f32,
    w2: f32,
) -> [u8; 3] {
    // Interpolate
    let nx = n0.x * w0 + n1.x * w1 + n2.x * w2;
    let ny = n0.y * w0 + n1.y * w1 + n2.y * w2;
    let nz = n0.z * w0 + n1.z * w1 + n2.z * w2;

    // Normalize using SIMD
    let (nx_norm, ny_norm, nz_norm) = normalize_vector(nx, ny, nz);

    // Encode: [-1, 1] -> [0, 255] using SIMD
    let normal_vec = f32x4::from_array([nx_norm, ny_norm, nz_norm, 0.0]);
    let encoded = (normal_vec * f32x4::splat(0.5) + f32x4::splat(0.5)) * f32x4::splat(255.0);
    let clamped = encoded.simd_clamp(f32x4::splat(0.0), f32x4::splat(255.0));
    let arr = clamped.to_array();

    [arr[0] as u8, arr[1] as u8, arr[2] as u8]
}

// ---------------------------------------------------------------------------
// NormalMapRenderer
// ---------------------------------------------------------------------------

/// CPU rasterizer that renders per-vertex normals of a [`Mesh`] into an image.
pub struct NormalMapRenderer;

/// Tile size for cache-friendly rasterization.
const TILE_SIZE: i32 = 16;

impl NormalMapRenderer {
    /// Render a normal map of `mesh` as seen from `camera`.
    ///
    /// Returns an RGB image where `RGB = (normal + 1) / 2 * 255`.
    ///
    /// This implementation uses several optimizations:
    /// - SIMD for barycentric coords and normal calculations
    /// - Tile-based rasterization for cache locality
    /// - Incremental edge evaluation
    /// - Pre-computed reciprocals
    #[must_use]
    pub fn render(mesh: &Mesh, camera: &Camera) -> RgbImage {
        let mut img = RgbImage::new(camera.width, camera.height);

        // A zero-dimension camera has no pixels to fill; return the (empty)
        // image immediately. Besides being the only sensible result, this
        // avoids `camera.width - 1` / `camera.height - 1` underflowing
        // (u32) further down.
        if camera.width == 0 || camera.height == 0 {
            return img;
        }

        let w = camera.width as usize;
        let h = camera.height as usize;

        // `Mesh`'s fields are all public, so a caller can construct one
        // whose `normals` doesn't match `vertices` without going through
        // `Mesh::new`'s invariants. Indexing `mesh.normals[i]` below would
        // then be unchecked; reject up front rather than risk a panic.
        if mesh.normals.len() != mesh.vertices.len() {
            tracing::warn!(
                "NormalMapRenderer::render: mesh.normals.len() ({}) != \
                 mesh.vertices.len() ({}); returning a blank image",
                mesh.normals.len(),
                mesh.vertices.len()
            );
            return img;
        }

        let mut depth_buf = vec![f32::INFINITY; w * h];

        for face in &mesh.faces {
            let (i0, i1, i2) = (face[0] as usize, face[1] as usize, face[2] as usize);
            if i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len() {
                // Face references a vertex index that doesn't exist
                // (malformed mesh topology); skip it rather than index out
                // of bounds.
                continue;
            }

            // Pre-fetch normals for cache locality
            let n0 = &mesh.normals[i0];
            let n1 = &mesh.normals[i1];
            let n2 = &mesh.normals[i2];

            // Transform to camera space
            let p0 = camera.world_to_cam(&mesh.vertices[i0]);
            let p1 = camera.world_to_cam(&mesh.vertices[i1]);
            let p2 = camera.world_to_cam(&mesh.vertices[i2]);

            // Near-plane cull
            if p0.z <= camera.near || p1.z <= camera.near || p2.z <= camera.near {
                continue;
            }

            // Pre-compute z reciprocals for projection optimization
            let recip_z0 = 1.0 / p0.z;
            let recip_z1 = 1.0 / p1.z;
            let recip_z2 = 1.0 / p2.z;

            // Project to screen using optimized projection
            let s0 = camera.project_with_recip_z(&p0, recip_z0);
            let s1 = camera.project_with_recip_z(&p1, recip_z1);
            let s2 = camera.project_with_recip_z(&p2, recip_z2);

            // Bounding box (clipped to image). `saturating_sub` is
            // belt-and-braces against a zero-dimension camera (already
            // rejected above, but this keeps the clamp itself safe even if
            // that guard is ever removed or bypassed).
            let min_x = s0[0].min(s1[0]).min(s2[0]).max(0.0).floor() as i32;
            let max_x = s0[0]
                .max(s1[0])
                .max(s2[0])
                .min(camera.width.saturating_sub(1) as f32)
                .ceil() as i32;
            let min_y = s0[1].min(s1[1]).min(s2[1]).max(0.0).floor() as i32;
            let max_y = s0[1]
                .max(s1[1])
                .max(s2[1])
                .min(camera.height.saturating_sub(1) as f32)
                .ceil() as i32;

            if min_x > max_x || min_y > max_y {
                continue;
            }

            // Triangle area (2x)
            let area = edge_fn(s0, s1, s2);
            if area.abs() < 1e-10 {
                continue;
            }

            // Pre-compute reciprocal of area to avoid divisions in inner loop
            let inv_area = 1.0 / area;

            // Tile-based rasterization for better cache locality
            let tile_min_x = (min_x / TILE_SIZE) * TILE_SIZE;
            let tile_max_x = ((max_x + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
            let tile_min_y = (min_y / TILE_SIZE) * TILE_SIZE;
            let tile_max_y = ((max_y + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;

            // Iterate over tiles
            for tile_y in (tile_min_y..tile_max_y).step_by(TILE_SIZE as usize) {
                for tile_x in (tile_min_x..tile_max_x).step_by(TILE_SIZE as usize) {
                    // Compute actual pixel bounds within this tile
                    let px_min = tile_x.max(min_x);
                    let px_max = (tile_x + TILE_SIZE).min(max_x + 1);
                    let py_min = tile_y.max(min_y);
                    let py_max = (tile_y + TILE_SIZE).min(max_y + 1);

                    // Process pixels within tile
                    Self::rasterize_tile(
                        px_min,
                        px_max,
                        py_min,
                        py_max,
                        s0,
                        s1,
                        s2,
                        inv_area,
                        recip_z0,
                        recip_z1,
                        recip_z2,
                        n0,
                        n1,
                        n2,
                        w,
                        &mut depth_buf,
                        &mut img,
                    );
                }
            }
        }

        img
    }

    /// Rasterize a single tile of pixels with incremental edge evaluation.
    ///
    /// `recip_z0`/`recip_z1`/`recip_z2` are `1/z` at each vertex (camera
    /// space); depth and the normal are both interpolated
    /// perspective-correctly from them (see the body for the derivation),
    /// rather than by affinely interpolating screen-space quantities.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn rasterize_tile(
        px_min: i32,
        px_max: i32,
        py_min: i32,
        py_max: i32,
        s0: [f32; 2],
        s1: [f32; 2],
        s2: [f32; 2],
        inv_area: f32,
        recip_z0: f32,
        recip_z1: f32,
        recip_z2: f32,
        n0: &na::Vector3<f32>,
        n1: &na::Vector3<f32>,
        n2: &na::Vector3<f32>,
        width: usize,
        depth_buf: &mut [f32],
        img: &mut RgbImage,
    ) {
        // Pre-compute edge equation coefficients for incremental evaluation
        let a01 = s0[1] - s1[1];
        let a12 = s1[1] - s2[1];
        let a20 = s2[1] - s0[1];

        for py in py_min..py_max {
            let p_y = py as f32 + 0.5;

            // Initial edge values at the start of this scanline
            let p_x_start = px_min as f32 + 0.5;
            let mut e0 = edge_fn(s1, s2, [p_x_start, p_y]);
            let mut e1 = edge_fn(s2, s0, [p_x_start, p_y]);
            let mut e2 = edge_fn(s0, s1, [p_x_start, p_y]);

            for px in px_min..px_max {
                // Normalize by the signed triangle area BEFORE the inside
                // test. `e0..e2` alone only carry the correct sign when the
                // triangle's screen-space winding happens to be positive;
                // dividing by `area` (which carries the same sign) first
                // makes the test -- and everything derived from w0..w2 --
                // independent of winding.
                let w0 = e0 * inv_area;
                let w1 = e1 * inv_area;
                let w2 = e2 * inv_area;

                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    // Perspective-correct interpolation. Under perspective
                    // projection, 1/z (unlike z itself) IS an affine
                    // function of screen position, so it is valid to
                    // interpolate it with the plain screen-space weights
                    // w0..w2:
                    let inv_z = w0 * recip_z0 + w1 * recip_z1 + w2 * recip_z2;
                    let depth = 1.0 / inv_z;
                    let idx = py as usize * width + px as usize;

                    // `depth > near` always holds here: the near-plane cull
                    // in `render` already guarantees z0,z1,z2 > near > 0,
                    // so recip_z0..2 < 1/near, so their convex combination
                    // inv_z < 1/near, so depth = 1/inv_z > near.
                    if depth < depth_buf[idx] {
                        depth_buf[idx] = depth;

                        // Perspective-correct barycentric weights for
                        // attribute interpolation (e.g. normals): scale
                        // each screen-space weight by that vertex's 1/z,
                        // then renormalize by the same `inv_z`.
                        let pw0 = w0 * recip_z0 / inv_z;
                        let pw1 = w1 * recip_z1 / inv_z;
                        let pw2 = w2 * recip_z2 / inv_z;

                        // Interpolate and encode normal (SIMD when feature enabled)
                        let rgb = interpolate_and_encode_normal(n0, n1, n2, pw0, pw1, pw2);

                        img.put_pixel(px as u32, py as u32, Rgb(rgb));
                    }
                }

                // Incremental update for next pixel in scanline
                e0 += a12;
                e1 += a20;
                e2 += a01;
            }
        }
    }
}

/// Edge function (signed area x 2) of the triangle formed by `a`, `b`, `p`.
#[inline]
fn edge_fn(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a single-triangle mesh directly (bypassing `Mesh::new`'s
    /// auto-computed normals) so tests can assign arbitrary per-vertex
    /// normals independent of the triangle's actual geometry.
    fn single_triangle_mesh(
        vertices: [na::Point3<f32>; 3],
        normals: [na::Vector3<f32>; 3],
    ) -> Mesh {
        Mesh {
            vertices: vertices.to_vec(),
            normals: normals.to_vec(),
            faces: vec![[0u32, 1, 2]],
            uv_coords: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Regression: rasterize_tile must not silently drop triangles whose
    // screen-space signed area is negative (critical bug: the inside-test
    // used raw, non-normalized edge values, so only positive-area triangles
    // were ever rasterized).
    // -----------------------------------------------------------------------

    #[test]
    fn test_rasterize_renders_negative_screen_area_triangle() {
        let camera = Camera {
            rotation: na::Matrix3::identity(),
            translation: na::Vector3::new(0.0, 0.0, 0.0),
            focal_x: 100.0,
            focal_y: 100.0,
            cx: 50.0,
            cy: 50.0,
            width: 100,
            height: 100,
            near: 0.01,
            far: 1000.0,
        };

        // Vertex order deliberately chosen so the projected triangle has
        // NEGATIVE screen-space signed area. With this camera (identity
        // rotation, zero translation, so camera space == world space):
        // v0=(-1,-1,5) -> s0=(30,30), v1=(0,1,5) -> s1=(50,70),
        // v2=(1,-1,5) -> s2=(70,30); edge_fn(s0,s1,s2) = -1600.
        let mesh = single_triangle_mesh(
            [
                na::Point3::new(-1.0f32, -1.0, 5.0),
                na::Point3::new(0.0f32, 1.0, 5.0),
                na::Point3::new(1.0f32, -1.0, 5.0),
            ],
            [na::Vector3::new(0.0f32, 0.0, 1.0); 3],
        );

        let img = NormalMapRenderer::render(&mesh, &camera);

        let has_non_background_pixel = img.pixels().any(|p| *p != Rgb([0, 0, 0]));
        assert!(
            has_non_background_pixel,
            "a triangle with negative screen-space signed area must still be rasterized, \
             not silently dropped"
        );
    }

    #[test]
    fn test_rasterize_renders_both_windings_of_the_same_triangle() {
        // The same triangle, once with each vertex order: both must
        // rasterize to the same set of covered pixels, proving the
        // inside-test is winding-independent.
        let camera = Camera {
            rotation: na::Matrix3::identity(),
            translation: na::Vector3::new(0.0, 0.0, 0.0),
            focal_x: 100.0,
            focal_y: 100.0,
            cx: 50.0,
            cy: 50.0,
            width: 100,
            height: 100,
            near: 0.01,
            far: 1000.0,
        };
        let normal = na::Vector3::new(0.0f32, 0.0, 1.0);

        let ccw = single_triangle_mesh(
            [
                na::Point3::new(-1.0f32, -1.0, 5.0),
                na::Point3::new(1.0f32, -1.0, 5.0),
                na::Point3::new(0.0f32, 1.0, 5.0),
            ],
            [normal; 3],
        );
        let cw = single_triangle_mesh(
            [
                na::Point3::new(-1.0f32, -1.0, 5.0),
                na::Point3::new(0.0f32, 1.0, 5.0),
                na::Point3::new(1.0f32, -1.0, 5.0),
            ],
            [normal; 3],
        );

        let img_ccw = NormalMapRenderer::render(&ccw, &camera);
        let img_cw = NormalMapRenderer::render(&cw, &camera);

        let coverage =
            |img: &RgbImage| -> usize { img.pixels().filter(|p| **p != Rgb([0, 0, 0])).count() };
        let n_ccw = coverage(&img_ccw);
        let n_cw = coverage(&img_cw);

        assert!(n_ccw > 0, "the positive-winding triangle must rasterize");
        assert!(n_cw > 0, "the negative-winding triangle must rasterize too");
        assert_eq!(
            n_ccw, n_cw,
            "both windings of the same triangle must cover the same pixel count"
        );
    }

    // -----------------------------------------------------------------------
    // Regression: Camera::default_front must place the camera in front of
    // the face (looking back toward it) without vertically mirroring the
    // render.
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_front_camera_orientation() {
        let camera = Camera::default_front(100, 100);

        // A point at the world origin (roughly the face) must be in front
        // of the camera: p_cam.z > 0, per the `Camera` convention doc.
        let origin_cam = camera.world_to_cam(&na::Point3::new(0.0, 0.0, 0.0));
        assert!(
            origin_cam.z > 0.0,
            "a point at the world origin must be in front of the default_front camera, got z = {}",
            origin_cam.z
        );

        // A point ABOVE the face (larger world Y, e.g. the forehead) must
        // project to a SMALLER screen-space y (higher up on screen, since
        // row 0 is the top) than a point BELOW it (smaller world Y, e.g.
        // the chin) -- otherwise the render is vertically mirrored.
        let forehead_cam = camera.world_to_cam(&na::Point3::new(0.0, 0.1, 0.0));
        let chin_cam = camera.world_to_cam(&na::Point3::new(0.0, -0.1, 0.0));
        let forehead_screen = camera.project(&forehead_cam);
        let chin_screen = camera.project(&chin_cam);
        assert!(
            forehead_screen[1] < chin_screen[1],
            "a physically higher world point must land higher on screen (smaller pixel y); \
             forehead y = {}, chin y = {}",
            forehead_screen[1],
            chin_screen[1]
        );
    }

    // -----------------------------------------------------------------------
    // Regression: depth and normal interpolation must be perspective-
    // correct, not affine in screen space.
    // -----------------------------------------------------------------------

    #[test]
    fn test_perspective_correct_normal_interpolation() {
        // Triangle with a large depth disparity: z0 = z1 = 1 (near),
        // z2 = 100 (far). World-space (x, y) for v2 is scaled up so its
        // screen footprint (divided by its large z) stays comparable to
        // v0/v1's, giving a well-conditioned screen-space triangle:
        // s0=(0,0), s1=(10,0), s2=(0,10) with focal=1, cx=cy=0.
        let camera = Camera {
            rotation: na::Matrix3::identity(),
            translation: na::Vector3::new(0.0, 0.0, 0.0),
            focal_x: 1.0,
            focal_y: 1.0,
            cx: 0.0,
            cy: 0.0,
            width: 20,
            height: 20,
            near: 0.01,
            far: 1000.0,
        };

        // n0, n1 (the two NEAR vertices) point straight at the camera
        // (+Z); n2 (the FAR vertex) points sideways (+X). At pixel (3, 3)
        // [sample point (3.5, 3.5)] the screen-space (affine) barycentric
        // weights are (0.30, 0.35, 0.35) -- a naive affine interpolation
        // would blend in 35% of the far vertex's sideways normal. The
        // perspective-correct weights are instead (~0.459, ~0.536,
        // ~0.005): the far vertex should contribute almost nothing.
        let mesh = single_triangle_mesh(
            [
                na::Point3::new(0.0f32, 0.0, 1.0),
                na::Point3::new(10.0f32, 0.0, 1.0),
                na::Point3::new(0.0f32, 1000.0, 100.0),
            ],
            [
                na::Vector3::new(0.0f32, 0.0, 1.0),
                na::Vector3::new(0.0f32, 0.0, 1.0),
                na::Vector3::new(1.0f32, 0.0, 0.0),
            ],
        );

        let img = NormalMapRenderer::render(&mesh, &camera);
        let Rgb([r, _g, b]) = *img.get_pixel(3, 3);

        // Perspective-correct expectation at this pixel: normal ~= (0.005,
        // 0, 0.995) after normalizing -> R ~= 128, B ~= 254.
        // The buggy affine interpolation would instead give normal ~=
        // (0.474, 0, 0.880) -> R ~= 187, B ~= 239.
        assert!(
            r < 145,
            "red channel should reflect the perspective-correct (small) contribution \
             of the far vertex's sideways normal, got {r} (affine interpolation gives ~187)"
        );
        assert!(
            b > 245,
            "blue channel should reflect the perspective-correct (near-unit) z \
             component, got {b} (affine interpolation gives ~239)"
        );
    }

    // -----------------------------------------------------------------------
    // Regression: a zero-dimension camera must not panic (u32 underflow in
    // the bounding-box clamp) and must return an empty image of the
    // requested (zero) dimensions.
    // -----------------------------------------------------------------------

    fn single_forward_triangle() -> Mesh {
        single_triangle_mesh(
            [
                na::Point3::new(0.0f32, 0.0, 1.0),
                na::Point3::new(1.0f32, 0.0, 1.0),
                na::Point3::new(0.0f32, 1.0, 1.0),
            ],
            [na::Vector3::new(0.0f32, 0.0, 1.0); 3],
        )
    }

    #[test]
    fn test_render_zero_width_camera_does_not_panic() {
        let camera = Camera {
            rotation: na::Matrix3::identity(),
            translation: na::Vector3::new(0.0, 0.0, 0.0),
            focal_x: 100.0,
            focal_y: 100.0,
            cx: 0.0,
            cy: 0.0,
            width: 0,
            height: 10,
            near: 0.01,
            far: 10.0,
        };

        let img = NormalMapRenderer::render(&single_forward_triangle(), &camera);
        assert_eq!(img.width(), 0);
        assert_eq!(img.height(), 10);
    }

    #[test]
    fn test_render_zero_height_camera_does_not_panic() {
        let camera = Camera {
            rotation: na::Matrix3::identity(),
            translation: na::Vector3::new(0.0, 0.0, 0.0),
            focal_x: 100.0,
            focal_y: 100.0,
            cx: 0.0,
            cy: 0.0,
            width: 10,
            height: 0,
            near: 0.01,
            far: 10.0,
        };

        let img = NormalMapRenderer::render(&single_forward_triangle(), &camera);
        assert_eq!(img.width(), 10);
        assert_eq!(img.height(), 0);
    }

    // -----------------------------------------------------------------------
    // Regression: `Mesh`'s fields are all public, so a caller can build one
    // that violates `Mesh::new`'s invariants (mismatched normals/vertices
    // lengths, out-of-range face indices). `render` must not index out of
    // bounds and panic on such a mesh.
    // -----------------------------------------------------------------------

    #[test]
    fn test_render_mismatched_normals_len_does_not_panic() {
        let camera = Camera {
            rotation: na::Matrix3::identity(),
            translation: na::Vector3::new(0.0, 0.0, 0.0),
            focal_x: 100.0,
            focal_y: 100.0,
            cx: 50.0,
            cy: 50.0,
            width: 20,
            height: 20,
            near: 0.01,
            far: 1000.0,
        };
        let mesh = Mesh {
            vertices: vec![
                na::Point3::new(0.0f32, 0.0, 1.0),
                na::Point3::new(1.0f32, 0.0, 1.0),
                na::Point3::new(0.0f32, 1.0, 1.0),
            ],
            normals: Vec::new(), // deliberately empty / mismatched
            faces: vec![[0u32, 1, 2]],
            uv_coords: Vec::new(),
        };
        let img = NormalMapRenderer::render(&mesh, &camera);
        assert_eq!(img.width(), 20);
        assert_eq!(img.height(), 20);
    }

    #[test]
    fn test_render_out_of_range_face_index_does_not_panic() {
        let camera = Camera {
            rotation: na::Matrix3::identity(),
            translation: na::Vector3::new(0.0, 0.0, 0.0),
            focal_x: 100.0,
            focal_y: 100.0,
            cx: 50.0,
            cy: 50.0,
            width: 20,
            height: 20,
            near: 0.01,
            far: 1000.0,
        };
        let mesh = Mesh {
            vertices: vec![
                na::Point3::new(0.0f32, 0.0, 1.0),
                na::Point3::new(1.0f32, 0.0, 1.0),
            ],
            normals: vec![na::Vector3::new(0.0f32, 0.0, 1.0); 2],
            // Vertex index 5 doesn't exist.
            faces: vec![[0u32, 1, 5]],
            uv_coords: Vec::new(),
        };
        let img = NormalMapRenderer::render(&mesh, &camera);
        assert_eq!(img.width(), 20);
        assert_eq!(img.height(), 20);
    }

    #[test]
    fn test_render_zero_width_and_height_camera_does_not_panic() {
        let camera = Camera {
            rotation: na::Matrix3::identity(),
            translation: na::Vector3::new(0.0, 0.0, 0.0),
            focal_x: 100.0,
            focal_y: 100.0,
            cx: 0.0,
            cy: 0.0,
            width: 0,
            height: 0,
            near: 0.01,
            far: 10.0,
        };

        let img = NormalMapRenderer::render(&single_forward_triangle(), &camera);
        assert_eq!(img.width(), 0);
        assert_eq!(img.height(), 0);
    }
}
