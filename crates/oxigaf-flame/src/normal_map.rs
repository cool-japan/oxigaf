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
    #[must_use]
    pub fn default_front(width: u32, height: u32) -> Self {
        let focal = width as f32 * 1.5;
        Self {
            rotation: na::Matrix3::identity(),
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
        let w = camera.width as usize;
        let h = camera.height as usize;
        let mut img = RgbImage::new(camera.width, camera.height);
        let mut depth_buf = vec![f32::INFINITY; w * h];

        for face in &mesh.faces {
            let (i0, i1, i2) = (face[0] as usize, face[1] as usize, face[2] as usize);

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

            // Bounding box (clipped to image)
            let min_x = s0[0].min(s1[0]).min(s2[0]).max(0.0).floor() as i32;
            let max_x = s0[0]
                .max(s1[0])
                .max(s2[0])
                .min((camera.width - 1) as f32)
                .ceil() as i32;
            let min_y = s0[1].min(s1[1]).min(s2[1]).max(0.0).floor() as i32;
            let max_y = s0[1]
                .max(s1[1])
                .max(s2[1])
                .min((camera.height - 1) as f32)
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
                        p0.z,
                        p1.z,
                        p2.z,
                        n0,
                        n1,
                        n2,
                        camera.near,
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
        z0: f32,
        z1: f32,
        z2: f32,
        n0: &na::Vector3<f32>,
        n1: &na::Vector3<f32>,
        n2: &na::Vector3<f32>,
        near: f32,
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
                // Check if point is inside triangle using edge values
                if e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0 {
                    // Compute barycentric weights
                    let w0 = e0 * inv_area;
                    let w1 = e1 * inv_area;
                    let w2 = e2 * inv_area;

                    let depth = w0 * z0 + w1 * z1 + w2 * z2;
                    let idx = py as usize * width + px as usize;

                    if depth < depth_buf[idx] && depth > near {
                        depth_buf[idx] = depth;

                        // Interpolate and encode normal (SIMD when feature enabled)
                        let rgb = interpolate_and_encode_normal(n0, n1, n2, w0, w1, w2);

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
