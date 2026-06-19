//! Screen-space normal estimation from depth maps.
//!
//! This module provides CPU-side algorithms for estimating surface normal vectors
//! from depth maps produced by a 3DGS rasterizer. Normals are critical for:
//!
//! - **Relighting**: Surface orientation for physically-based shading.
//! - **Surface quality metrics**: Smoothness, roughness, curvature.
//! - **Training supervision signals**: Normal consistency loss across views.
//!
//! # Methods
//!
//! - [`estimate_normals_cross_product`]: Physically accurate unproject + cross-product method.
//! - [`estimate_normals_sobel`]: Faster Sobel-filter approximation (no intrinsics needed).
//! - [`smooth_normals`]: Gaussian smoothing of a normal map.
//!
//! # Normal Map Encoding
//!
//! Normals are stored as unit vectors `[nx, ny, nz]` in camera space:
//! - `nx, ny`: tangential components (lateral)
//! - `nz`: depth component; positive means toward the camera
//!
//! Values are in `[-1, 1]`.  Storage is row-major, 3 floats per pixel:
//! `index = (y * width + x) * 3`.
//!
//! Invalid pixels (e.g. INFINITY depth) are stored as `[0, 0, 0]`.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by normal-estimation operations.
#[derive(Debug, Error)]
pub enum NormalError {
    /// Zero dimensions or mismatched width/height vs. slice length.
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// Depth values contain negative entries (depth must be non-negative).
    #[error("Invalid depth: {0}")]
    InvalidDepth(String),

    /// The input depth map (or normal map) is empty.
    #[error("Empty map: no pixels provided")]
    EmptyMap,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `true` when `n` represents a real normal (not the zero sentinel used
/// for invalid / infinity-depth pixels).
#[inline]
fn is_valid_normal(n: &[f32; 3]) -> bool {
    let len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    len_sq > 0.25 // length > 0.5
}

/// Normalise a 3-vector.  Returns `None` if the vector length is below `1e-8`.
#[inline]
fn normalize3(v: [f32; 3]) -> Option<[f32; 3]> {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq < 1e-16 {
        return None;
    }
    let inv = len_sq.sqrt().recip();
    Some([v[0] * inv, v[1] * inv, v[2] * inv])
}

/// Cross product of two 3-vectors.
#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two 3-vectors.
#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Subtract two 3-vectors: `a - b`.
#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

// ─────────────────────────────────────────────────────────────────────────────
// NormalMap
// ─────────────────────────────────────────────────────────────────────────────

/// Screen-space normal map.
///
/// Each pixel stores a 3-D unit normal vector in camera space.  Normals are
/// packed as `[nx, ny, nz]` with:
/// - `nx, ny` — tangential components
/// - `nz`     — depth component, positive = toward the camera
///
/// Values are in `[-1, 1]`.  Storage is row-major: the element at pixel
/// `(x, y)` starts at `normals[(y * width + x) as usize * 3]`.
///
/// Pixels that could not be estimated (e.g. INFINITY depth) carry the zero
/// sentinel `[0, 0, 0]`.
#[derive(Debug, Clone)]
pub struct NormalMap {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Flat storage: `[nx0, ny0, nz0, nx1, ny1, nz1, ...]` (row-major, 3 f32 per pixel).
    pub normals: Vec<f32>,
}

impl NormalMap {
    /// Create a new `NormalMap` with all pixels initialised to `[0, 0, 1]`
    /// (unit normal pointing toward the camera).
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width as usize) * (height as usize);
        let mut normals = vec![0.0_f32; n * 3];
        // Set nz = 1 for every pixel.
        for i in 0..n {
            normals[i * 3 + 2] = 1.0;
        }
        Self {
            width,
            height,
            normals,
        }
    }

    /// Return the normal at pixel `(x, y)`.
    ///
    /// Returns `[0.0, 0.0, 1.0]` when `(x, y)` is out of bounds.
    pub fn pixel(&self, x: u32, y: u32) -> [f32; 3] {
        if x >= self.width || y >= self.height {
            return [0.0, 0.0, 1.0];
        }
        let base = ((y as usize) * (self.width as usize) + x as usize) * 3;
        [
            self.normals[base],
            self.normals[base + 1],
            self.normals[base + 2],
        ]
    }

    /// Set the normal at pixel `(x, y)`.  Out-of-bounds writes are silently ignored.
    pub fn set_pixel(&mut self, x: u32, y: u32, normal: [f32; 3]) {
        if x >= self.width || y >= self.height {
            return;
        }
        let base = ((y as usize) * (self.width as usize) + x as usize) * 3;
        self.normals[base] = normal[0];
        self.normals[base + 1] = normal[1];
        self.normals[base + 2] = normal[2];
    }

    /// Convert to an RGB u8 image using the standard normal-map encoding:
    /// `encoded = (n * 0.5 + 0.5) * 255`.
    ///
    /// Returns a flat `Vec<u8>` with 3 bytes per pixel (row-major).
    pub fn to_rgb_u8(&self) -> Vec<u8> {
        self.normals
            .iter()
            .map(|&v| ((v * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8)
            .collect()
    }

    /// Convert to an RGB f32 image in `[0, 1]` using `(n * 0.5 + 0.5)`.
    ///
    /// Returns a flat `Vec<f32>` with 3 floats per pixel (row-major).
    pub fn to_rgb_f32(&self) -> Vec<f32> {
        self.normals.iter().map(|&v| v * 0.5 + 0.5).collect()
    }

    /// Count pixels with a valid normal (non-zero sentinel, i.e. length > 0.5).
    pub fn num_valid(&self) -> usize {
        let n_pixels = (self.width as usize) * (self.height as usize);
        (0..n_pixels)
            .filter(|&i| {
                let b = i * 3;
                let px = [self.normals[b], self.normals[b + 1], self.normals[b + 2]];
                is_valid_normal(&px)
            })
            .count()
    }

    /// Compute the mean normal across all valid pixels.
    ///
    /// Returns `[0.0, 0.0, 1.0]` if no valid pixels exist.
    pub fn mean_normal(&self) -> [f32; 3] {
        let n_pixels = (self.width as usize) * (self.height as usize);
        let mut sum = [0.0_f32; 3];
        let mut count = 0usize;
        for i in 0..n_pixels {
            let b = i * 3;
            let px = [self.normals[b], self.normals[b + 1], self.normals[b + 2]];
            if is_valid_normal(&px) {
                sum[0] += px[0];
                sum[1] += px[1];
                sum[2] += px[2];
                count += 1;
            }
        }
        if count == 0 {
            return [0.0, 0.0, 1.0];
        }
        let inv = (count as f32).recip();
        let mean_raw = [sum[0] * inv, sum[1] * inv, sum[2] * inv];
        normalize3(mean_raw).unwrap_or([0.0, 0.0, 1.0])
    }

    /// Compute per-pixel angular deviation (in radians) from a `reference` unit vector.
    ///
    /// For invalid pixels the deviation is `0.0`.
    /// Formula: `acos(clamp(dot(pixel_normal, reference), -1.0, 1.0))`.
    pub fn angular_deviation(&self, reference: [f32; 3]) -> Vec<f32> {
        self.normals
            .chunks_exact(3)
            .map(|chunk| {
                let px = [chunk[0], chunk[1], chunk[2]];
                if is_valid_normal(&px) {
                    dot3(px, reference).clamp(-1.0, 1.0).acos()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Compute the mean angular deviation (in radians) from a `reference` unit vector.
    ///
    /// Returns `0.0` if no valid pixels.
    pub fn mean_angular_deviation(&self, reference: [f32; 3]) -> f32 {
        let mut sum = 0.0_f32;
        let mut count = 0usize;
        for chunk in self.normals.chunks_exact(3) {
            let px = [chunk[0], chunk[1], chunk[2]];
            if is_valid_normal(&px) {
                sum += dot3(px, reference).clamp(-1.0, 1.0).acos();
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f32
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-product normal estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Unproject pixel `(px, py)` with depth `d` into camera-space 3-D coordinates.
///
/// Convention: positive Z points into the scene.
///
/// ```text
/// X = (px - cx) * d / fx
/// Y = (py - cy) * d / fy
/// Z = d
/// ```
#[inline]
fn unproject(px: u32, py: u32, d: f32, fx: f32, fy: f32, cx: f32, cy: f32) -> [f32; 3] {
    [(px as f32 - cx) * d / fx, (py as f32 - cy) * d / fy, d]
}

/// Camera intrinsics bundled together to reduce argument counts on inner helpers.
#[derive(Clone, Copy)]
struct CameraIntrinsics {
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
}

/// Retrieve the 3-D camera-space position for neighbour `(nx, ny)` when its
/// depth is finite, or return `None`.
#[inline]
fn neighbour_pos(
    nx: i64,
    ny: i64,
    width: u32,
    height: u32,
    depth: &[f32],
    cam: CameraIntrinsics,
) -> Option<[f32; 3]> {
    if nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64 {
        return None;
    }
    let idx = (ny as usize) * (width as usize) + nx as usize;
    let d = depth[idx];
    if !d.is_finite() || d <= 0.0 {
        return None;
    }
    Some(unproject(
        nx as u32, ny as u32, d, cam.fx, cam.fy, cam.cx, cam.cy,
    ))
}

/// Estimate surface normals from a depth map using the cross-product of finite
/// differences in camera space.
///
/// # Algorithm
///
/// For each valid pixel `(x, y)` with finite positive depth `d`:
///
/// 1. Unproject the pixel and its four cardinal neighbours into camera space:
///    `P(u, v) = [(u-cx)*d/fu, (v-cy)*d/fv, d]`
///
/// 2. Build a tangent vector along X (`dx_vec`) using central differences
///    when both neighbours are available, or a one-sided difference otherwise.
///    Similarly for Y (`dy_vec`).
///
/// 3. `normal = normalize(cross(dx_vec, dy_vec))`
///
/// 4. Orient the normal toward the camera (ensure `nz ≥ 0`).
///
/// Pixels with `INFINITY` depth receive the zero sentinel `[0, 0, 0]`.
///
/// # Errors
///
/// Returns [`NormalError::EmptyMap`] when `width == 0 || height == 0`.
/// Returns [`NormalError::InvalidDimensions`] when the slice length does not
/// equal `width * height`.
/// Returns [`NormalError::InvalidDepth`] when finite depth values are negative.
pub fn estimate_normals_cross_product(
    depth: &[f32],
    width: u32,
    height: u32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
) -> Result<NormalMap, NormalError> {
    if width == 0 || height == 0 {
        return Err(NormalError::EmptyMap);
    }
    let expected = (width as usize) * (height as usize);
    if depth.len() != expected {
        return Err(NormalError::InvalidDimensions(format!(
            "depth slice length {} does not match {}×{}={}",
            depth.len(),
            width,
            height,
            expected
        )));
    }
    // Check for negative finite depth values.
    for (i, &d) in depth.iter().enumerate() {
        if d.is_finite() && d < 0.0 {
            return Err(NormalError::InvalidDepth(format!(
                "negative depth {d} at index {i}"
            )));
        }
    }

    let cam = CameraIntrinsics { fx, fy, cx, cy };

    let mut map = NormalMap {
        width,
        height,
        normals: vec![0.0_f32; expected * 3],
    };

    for y in 0..height {
        for x in 0..width {
            let idx = (y as usize) * (width as usize) + x as usize;
            let d = depth[idx];

            if !d.is_finite() || d <= 0.0 {
                // Invalid pixel — leave as zero sentinel [0,0,0].
                continue;
            }

            let p_center = unproject(x, y, d, cam.fx, cam.fy, cam.cx, cam.cy);

            // Gather neighbours.
            let xi = x as i64;
            let yi = y as i64;
            let xp = neighbour_pos(xi + 1, yi, width, height, depth, cam);
            let xm = neighbour_pos(xi - 1, yi, width, height, depth, cam);
            let yp = neighbour_pos(xi, yi + 1, width, height, depth, cam);
            let ym = neighbour_pos(xi, yi - 1, width, height, depth, cam);

            // Tangent along X (camera right).
            let dx_vec = match (xp, xm) {
                (Some(a), Some(b)) => sub3(a, b),
                (Some(a), None) => sub3(a, p_center),
                (None, Some(b)) => sub3(p_center, b),
                (None, None) => {
                    // No X neighbours available — cannot estimate gradient.
                    map.set_pixel(x, y, [0.0, 0.0, 1.0]);
                    continue;
                }
            };

            // Tangent along Y (camera down in image space).
            let dy_vec = match (yp, ym) {
                (Some(a), Some(b)) => sub3(a, b),
                (Some(a), None) => sub3(a, p_center),
                (None, Some(b)) => sub3(p_center, b),
                (None, None) => {
                    map.set_pixel(x, y, [0.0, 0.0, 1.0]);
                    continue;
                }
            };

            let raw_normal = cross3(dx_vec, dy_vec);
            let normal = normalize3(raw_normal).unwrap_or([0.0, 0.0, 1.0]);

            // Orient toward camera (nz must be ≥ 0).
            let oriented = if normal[2] < 0.0 {
                [-normal[0], -normal[1], -normal[2]]
            } else {
                normal
            };

            map.set_pixel(x, y, oriented);
        }
    }

    Ok(map)
}

// ─────────────────────────────────────────────────────────────────────────────
// Sobel normal estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Sample depth at `(nx, ny)` for the Sobel filter.
///
/// Returns `None` if the position is out of bounds or the depth is non-finite.
#[inline]
fn sobel_sample(nx: i64, ny: i64, width: u32, height: u32, depth: &[f32]) -> Option<f32> {
    if nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64 {
        return None;
    }
    let d = depth[(ny as usize) * (width as usize) + nx as usize];
    if d.is_finite() {
        Some(d)
    } else {
        None
    }
}

/// Estimate surface normals from a depth map using Sobel-filter depth gradients.
///
/// This is a fast, camera-intrinsics-free approximation suitable when a rough
/// normal estimate is sufficient (e.g. SSAO, visualisation).
///
/// # Algorithm
///
/// 1. Convolve the depth map with the 3×3 Sobel kernels:
///    - `Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]`
///    - `Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]`
///
///    Any non-finite depth value in the 3×3 patch contributes zero (skipped).
///
/// 2. `normal = normalize([-grad_x, -grad_y, 1.0])`
///
/// Border pixels (within 1 pixel of the edge) are set to `[0, 0, 1]`.
///
/// # Errors
///
/// Returns [`NormalError::EmptyMap`] when `width == 0 || height == 0`.
/// Returns [`NormalError::InvalidDimensions`] when the slice length does not
/// equal `width * height`.
pub fn estimate_normals_sobel(
    depth: &[f32],
    width: u32,
    height: u32,
) -> Result<NormalMap, NormalError> {
    if width == 0 || height == 0 {
        return Err(NormalError::EmptyMap);
    }
    let expected = (width as usize) * (height as usize);
    if depth.len() != expected {
        return Err(NormalError::InvalidDimensions(format!(
            "depth slice length {} does not match {}×{}={}",
            depth.len(),
            width,
            height,
            expected
        )));
    }

    // Sobel kernel weights indexed as [row_offset+1][col_offset+1].
    // Gx[r][c], Gy[r][c]
    const GX: [[f32; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    const GY: [[f32; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

    let mut map = NormalMap::new(width, height);

    for y in 0..height {
        for x in 0..width {
            // Border pixels default to [0,0,1] (already set by NormalMap::new).
            if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
                continue;
            }

            let xi = x as i64;
            let yi = y as i64;

            let mut grad_x = 0.0_f32;
            let mut grad_y = 0.0_f32;

            for dr in -1i64..=1 {
                for dc in -1i64..=1 {
                    if let Some(d) = sobel_sample(xi + dc, yi + dr, width, height, depth) {
                        let r = (dr + 1) as usize;
                        let c = (dc + 1) as usize;
                        grad_x += GX[r][c] * d;
                        grad_y += GY[r][c] * d;
                    }
                }
            }

            // Check center pixel validity.
            let center_d = depth[(y as usize) * (width as usize) + x as usize];
            if !center_d.is_finite() {
                // Invalid center — leave as [0,0,1].
                continue;
            }

            let raw = [-grad_x, -grad_y, 1.0_f32];
            let normal = normalize3(raw).unwrap_or([0.0, 0.0, 1.0]);
            map.set_pixel(x, y, normal);
        }
    }

    Ok(map)
}

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian smoothing
// ─────────────────────────────────────────────────────────────────────────────

/// Build a 1-D Gaussian kernel of size `2*radius+1`, normalised to sum to 1.
fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return vec![1.0_f32];
    }
    let radius = (3.0_f32 * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(size);
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0_f32;
    for i in 0..size {
        let x = i as f32 - radius as f32;
        let val = (-x * x / two_sigma_sq).exp();
        kernel.push(val);
        sum += val;
    }
    if sum > 0.0 {
        for v in &mut kernel {
            *v /= sum;
        }
    }
    kernel
}

/// Apply separable Gaussian smoothing to a normal map.
///
/// Each of the three components `(nx, ny, nz)` is smoothed independently
/// with a 1-D Gaussian kernel applied first horizontally, then vertically.
/// After smoothing, each pixel is renormalised to unit length.
///
/// Border handling: out-of-bounds pixel accesses clamp to the nearest border pixel.
///
/// If `sigma <= 0`, the original map is returned unchanged.
pub fn smooth_normals(normal_map: &NormalMap, sigma: f32) -> NormalMap {
    if sigma <= 0.0 {
        return normal_map.clone();
    }

    let w = normal_map.width as usize;
    let h = normal_map.height as usize;
    let n_pixels = w * h;

    let kernel = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;

    // Temporary buffer for 3-channel image.
    let mut tmp = vec![0.0_f32; n_pixels * 3];

    // ── Horizontal pass ──────────────────────────────────────────────────────
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0_f32; 3];
            for (ki, &kw) in kernel.iter().enumerate() {
                let sx = (x as i64 + ki as i64 - radius as i64).clamp(0, w as i64 - 1) as usize;
                let b = (y * w + sx) * 3;
                acc[0] += kw * normal_map.normals[b];
                acc[1] += kw * normal_map.normals[b + 1];
                acc[2] += kw * normal_map.normals[b + 2];
            }
            let out_b = (y * w + x) * 3;
            tmp[out_b] = acc[0];
            tmp[out_b + 1] = acc[1];
            tmp[out_b + 2] = acc[2];
        }
    }

    // ── Vertical pass ────────────────────────────────────────────────────────
    let mut out_normals = vec![0.0_f32; n_pixels * 3];

    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0_f32; 3];
            for (ki, &kw) in kernel.iter().enumerate() {
                let sy = (y as i64 + ki as i64 - radius as i64).clamp(0, h as i64 - 1) as usize;
                let b = (sy * w + x) * 3;
                acc[0] += kw * tmp[b];
                acc[1] += kw * tmp[b + 1];
                acc[2] += kw * tmp[b + 2];
            }
            let out_b = (y * w + x) * 3;
            // Renormalise.
            let n = normalize3(acc).unwrap_or([0.0, 0.0, 1.0]);
            out_normals[out_b] = n[0];
            out_normals[out_b + 1] = n[1];
            out_normals[out_b + 2] = n[2];
        }
    }

    NormalMap {
        width: normal_map.width,
        height: normal_map.height,
        normals: out_normals,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Normal consistency loss
// ─────────────────────────────────────────────────────────────────────────────

/// Compute cross-view normal consistency loss between two normal maps.
///
/// Loss = mean over valid pixel pairs of `1 - |dot(n_a, n_b)|`.
///
/// A pixel is considered valid when its normal length exceeds 0.5 (the same
/// criterion as [`NormalMap::num_valid`]).  Returns `0.0` if no pixel pairs are
/// valid in both maps.
///
/// # Errors
///
/// Returns [`NormalError::InvalidDimensions`] when the two maps have different
/// width or height.
pub fn normal_consistency_loss(map_a: &NormalMap, map_b: &NormalMap) -> Result<f32, NormalError> {
    if map_a.width != map_b.width || map_a.height != map_b.height {
        return Err(NormalError::InvalidDimensions(format!(
            "map_a is {}×{} but map_b is {}×{}",
            map_a.width, map_a.height, map_b.width, map_b.height
        )));
    }

    let n_pixels = (map_a.width as usize) * (map_a.height as usize);
    let mut sum = 0.0_f32;
    let mut count = 0usize;

    for i in 0..n_pixels {
        let b = i * 3;
        let na = [map_a.normals[b], map_a.normals[b + 1], map_a.normals[b + 2]];
        let nb = [map_b.normals[b], map_b.normals[b + 1], map_b.normals[b + 2]];
        if is_valid_normal(&na) && is_valid_normal(&nb) {
            let d = dot3(na, nb).abs();
            sum += 1.0 - d;
            count += 1;
        }
    }

    if count == 0 {
        Ok(0.0)
    } else {
        Ok(sum / count as f32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NormalStats
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics computed from a [`NormalMap`].
#[derive(Debug, Clone)]
pub struct NormalStats {
    /// Mean unit normal of all valid pixels.
    pub mean_normal: [f32; 3],
    /// Standard deviation of angular deviations from `mean_normal` (radians).
    pub std_deviation: f32,
    /// Mean angular deviation from `mean_normal`, converted to **degrees**.
    pub roughness: f32,
    /// Number of valid pixels (normal length > 0.5).
    pub num_valid_pixels: usize,
    /// Fraction of valid pixels: `num_valid_pixels / total_pixels`.
    pub coverage: f32,
}

/// Compute surface statistics for a normal map.
///
/// Returns zeroed-out stats when the map is empty or contains no valid pixels.
pub fn compute_normal_stats(normal_map: &NormalMap) -> NormalStats {
    let total_pixels = (normal_map.width as usize) * (normal_map.height as usize);
    let mean_normal = normal_map.mean_normal();
    let num_valid_pixels = normal_map.num_valid();

    if num_valid_pixels == 0 || total_pixels == 0 {
        return NormalStats {
            mean_normal,
            std_deviation: 0.0,
            roughness: 0.0,
            num_valid_pixels: 0,
            coverage: 0.0,
        };
    }

    let devs = normal_map.angular_deviation(mean_normal);
    // Only include valid pixels in statistics.
    let valid_devs: Vec<f32> = normal_map
        .normals
        .chunks_exact(3)
        .zip(devs.iter().copied())
        .filter_map(|(chunk, dev)| {
            let px = [chunk[0], chunk[1], chunk[2]];
            if is_valid_normal(&px) {
                Some(dev)
            } else {
                None
            }
        })
        .collect();

    let mean_dev = valid_devs.iter().copied().sum::<f32>() / valid_devs.len() as f32;
    let variance = valid_devs
        .iter()
        .map(|&d| (d - mean_dev) * (d - mean_dev))
        .sum::<f32>()
        / valid_devs.len() as f32;
    let std_dev = variance.sqrt();

    NormalStats {
        mean_normal,
        std_deviation: std_dev,
        roughness: mean_dev.to_degrees(),
        num_valid_pixels,
        coverage: num_valid_pixels as f32 / total_pixels as f32,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Curvature estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate mean curvature from a normal map using the divergence of the normal
/// field.
///
/// ```text
/// κ(x, y) = div(N) / 2 ≈ (∂Nx/∂x + ∂Ny/∂y) / 2
/// ```
///
/// Each partial derivative is approximated with a central difference.  Border
/// pixels and invalid pixels receive curvature `0.0`.
///
/// Returns a flat `Vec<f32>` of length `width × height` (row-major).
pub fn estimate_curvature(normal_map: &NormalMap) -> Vec<f32> {
    let w = normal_map.width as usize;
    let h = normal_map.height as usize;
    let n_pixels = w * h;
    let mut curvature = vec![0.0_f32; n_pixels];

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let center = {
                let b = idx * 3;
                [
                    normal_map.normals[b],
                    normal_map.normals[b + 1],
                    normal_map.normals[b + 2],
                ]
            };
            if !is_valid_normal(&center) {
                continue;
            }

            // ∂Nx/∂x — central difference when both neighbours available.
            let dnx_dx = if x + 1 < w && x >= 1 {
                let right = {
                    let b = (y * w + x + 1) * 3;
                    normal_map.normals[b]
                };
                let left = {
                    let b = (y * w + x - 1) * 3;
                    normal_map.normals[b]
                };
                let nr = [
                    normal_map.normals[(y * w + x + 1) * 3],
                    normal_map.normals[(y * w + x + 1) * 3 + 1],
                    normal_map.normals[(y * w + x + 1) * 3 + 2],
                ];
                let nl = [
                    normal_map.normals[(y * w + x - 1) * 3],
                    normal_map.normals[(y * w + x - 1) * 3 + 1],
                    normal_map.normals[(y * w + x - 1) * 3 + 2],
                ];
                if is_valid_normal(&nr) && is_valid_normal(&nl) {
                    (right - left) * 0.5
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // ∂Ny/∂y — central difference when both neighbours available.
            let dny_dy = if y + 1 < h && y >= 1 {
                let ny_component_down = {
                    let b = ((y + 1) * w + x) * 3 + 1;
                    normal_map.normals[b]
                };
                let ny_component_up = {
                    let b = ((y - 1) * w + x) * 3 + 1;
                    normal_map.normals[b]
                };
                let nd = [
                    normal_map.normals[((y + 1) * w + x) * 3],
                    normal_map.normals[((y + 1) * w + x) * 3 + 1],
                    normal_map.normals[((y + 1) * w + x) * 3 + 2],
                ];
                let nu = [
                    normal_map.normals[((y - 1) * w + x) * 3],
                    normal_map.normals[((y - 1) * w + x) * 3 + 1],
                    normal_map.normals[((y - 1) * w + x) * 3 + 2],
                ];
                if is_valid_normal(&nd) && is_valid_normal(&nu) {
                    (ny_component_down - ny_component_up) * 0.5
                } else {
                    0.0
                }
            } else {
                0.0
            };

            curvature[idx] = (dnx_dx + dny_dy) * 0.5;
        }
    }

    curvature
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    fn flat_depth_map(w: u32, h: u32, d: f32) -> Vec<f32> {
        vec![d; (w * h) as usize]
    }

    // ─── NormalMap construction ───────────────────────────────────────────────

    /// Test 1: NormalMap::new initialises all pixels to [0, 0, 1].
    #[test]
    fn test_normal_map_new_initialised_to_up() {
        let map = NormalMap::new(4, 3);
        assert_eq!(map.width, 4);
        assert_eq!(map.height, 3);
        for y in 0..3 {
            for x in 0..4 {
                let p = map.pixel(x, y);
                assert!((p[0]).abs() < EPSILON, "nx should be 0");
                assert!((p[1]).abs() < EPSILON, "ny should be 0");
                assert!((p[2] - 1.0).abs() < EPSILON, "nz should be 1");
            }
        }
    }

    /// Test 2: NormalMap::pixel returns [0,0,1] for out-of-bounds access.
    #[test]
    fn test_normal_map_pixel_out_of_bounds() {
        let map = NormalMap::new(3, 3);
        let oob = map.pixel(10, 10);
        assert_eq!(oob, [0.0, 0.0, 1.0]);
    }

    /// Test 3: NormalMap::to_rgb_u8 encodes [0,0,1] as [128, 128, 255].
    #[test]
    fn test_to_rgb_u8_up_normal() {
        let map = NormalMap::new(1, 1);
        let rgb = map.to_rgb_u8();
        assert_eq!(rgb.len(), 3);
        // [0,0,1] → [0*0.5+0.5, 0*0.5+0.5, 1*0.5+0.5]*255 = [127.5, 127.5, 255]
        // Truncated to u8: 127 or 128 for the first two, 255 for z.
        assert!(
            rgb[0] >= 127 && rgb[0] <= 128,
            "r should be ~128, got {}",
            rgb[0]
        );
        assert!(
            rgb[1] >= 127 && rgb[1] <= 128,
            "g should be ~128, got {}",
            rgb[1]
        );
        assert_eq!(rgb[2], 255, "b should be 255");
    }

    /// Test 4: NormalMap::to_rgb_f32 encodes [0,0,1] as [0.5, 0.5, 1.0].
    #[test]
    fn test_to_rgb_f32_up_normal() {
        let map = NormalMap::new(1, 1);
        let rgb = map.to_rgb_f32();
        assert_eq!(rgb.len(), 3);
        assert!((rgb[0] - 0.5).abs() < EPSILON);
        assert!((rgb[1] - 0.5).abs() < EPSILON);
        assert!((rgb[2] - 1.0).abs() < EPSILON);
    }

    // ─── Sobel normal estimation ──────────────────────────────────────────────

    /// Test 5: Flat depth map → all interior normals ≈ [0, 0, 1].
    #[test]
    fn test_sobel_flat_depth() {
        let depth = flat_depth_map(8, 8, 2.5);
        let map = estimate_normals_sobel(&depth, 8, 8).unwrap();
        // Interior pixels should be [0, 0, 1]; borders default to [0,0,1] as well.
        for y in 1..7 {
            for x in 1..7 {
                let n = map.pixel(x, y);
                assert!(
                    (n[0]).abs() < EPSILON,
                    "nx@({x},{y}) should be 0, got {}",
                    n[0]
                );
                assert!(
                    (n[1]).abs() < EPSILON,
                    "ny@({x},{y}) should be 0, got {}",
                    n[1]
                );
                assert!(
                    (n[2] - 1.0).abs() < EPSILON,
                    "nz@({x},{y}) should be 1, got {}",
                    n[2]
                );
            }
        }
    }

    /// Test 6: Sobel with zero dimensions → Err(EmptyMap).
    #[test]
    fn test_sobel_empty_returns_err() {
        let result = estimate_normals_sobel(&[], 0, 0);
        assert!(matches!(result, Err(NormalError::EmptyMap)));
    }

    /// Test 7: Sobel on a depth map with a linear slope → non-trivial normals in interior.
    #[test]
    fn test_sobel_slope_gives_nontrivial_normals() {
        let w = 9u32;
        let h = 9u32;
        // Depth increases linearly along x: d = 1.0 + 0.1*x.
        let depth: Vec<f32> = (0..h)
            .flat_map(|_y| (0..w).map(|x| 1.0 + 0.1 * x as f32))
            .collect();
        let map = estimate_normals_sobel(&depth, w, h).unwrap();
        // Interior pixels should have a non-zero nx component.
        let n = map.pixel(4, 4);
        assert!(n[0].abs() > 0.01, "expected non-trivial nx, got {}", n[0]);
    }

    // ─── Cross-product normal estimation ─────────────────────────────────────

    /// Test 8: Flat depth → all interior cross-product normals ≈ [0, 0, 1].
    #[test]
    fn test_cross_product_flat_depth() {
        let depth = flat_depth_map(6, 6, 3.0);
        let map = estimate_normals_cross_product(&depth, 6, 6, 200.0, 200.0, 3.0, 3.0).unwrap();
        // Interior pixels (those with at least one X and one Y neighbour).
        for y in 1..5 {
            for x in 1..5 {
                let n = map.pixel(x, y);
                assert!((n[0]).abs() < EPSILON, "nx@({x},{y}) = {}", n[0]);
                assert!((n[1]).abs() < EPSILON, "ny@({x},{y}) = {}", n[1]);
                assert!((n[2] - 1.0).abs() < EPSILON, "nz@({x},{y}) = {}", n[2]);
            }
        }
    }

    /// Test 9: Cross-product with zero dimensions → Err(EmptyMap).
    #[test]
    fn test_cross_product_empty_returns_err() {
        let result = estimate_normals_cross_product(&[], 0, 0, 1.0, 1.0, 0.0, 0.0);
        assert!(matches!(result, Err(NormalError::EmptyMap)));
    }

    /// Test 9b: Cross-product with mismatched slice length → Err(InvalidDimensions).
    #[test]
    fn test_cross_product_mismatched_len_returns_err() {
        let depth = vec![1.0_f32; 10]; // wrong length for 4×4
        let result = estimate_normals_cross_product(&depth, 4, 4, 200.0, 200.0, 2.0, 2.0);
        assert!(matches!(result, Err(NormalError::InvalidDimensions(_))));
    }

    // ─── num_valid ────────────────────────────────────────────────────────────

    /// Test 10: After estimating from flat finite depth, all interior pixels are valid.
    #[test]
    fn test_num_valid_after_flat_estimation() {
        let depth = flat_depth_map(5, 5, 1.0);
        let map = estimate_normals_cross_product(&depth, 5, 5, 100.0, 100.0, 2.5, 2.5).unwrap();
        // All pixels in a flat map get a proper normal (at minimum [0,0,1] from fallback).
        assert!(map.num_valid() > 0);
    }

    /// Test 11: All-INFINITY depth → all normals are zero sentinel, num_valid == 0.
    #[test]
    fn test_all_infinity_depth_no_valid_normals() {
        let depth = vec![f32::INFINITY; 16];
        let map = estimate_normals_cross_product(&depth, 4, 4, 100.0, 100.0, 2.0, 2.0).unwrap();
        assert_eq!(map.num_valid(), 0);
        for y in 0..4 {
            for x in 0..4 {
                let n = map.pixel(x, y);
                assert_eq!(n, [0.0, 0.0, 0.0]);
            }
        }
    }

    // ─── smooth_normals ───────────────────────────────────────────────────────

    /// Test 12: Smoothing a flat [0,0,1] map → still [0,0,1].
    #[test]
    fn test_smooth_flat_map_unchanged() {
        let map = NormalMap::new(8, 8);
        let smoothed = smooth_normals(&map, 1.0);
        for y in 0..8 {
            for x in 0..8 {
                let n = smoothed.pixel(x, y);
                assert!(
                    (n[2] - 1.0).abs() < EPSILON,
                    "nz@({x},{y}) should be 1, got {}",
                    n[2]
                );
            }
        }
    }

    /// Test 13: Smoothing reduces angular deviation on a noisy map.
    #[test]
    fn test_smooth_reduces_angular_deviation() {
        let w = 16u32;
        let h = 16u32;
        // Create a noisy normal map by alternating two normals.
        let mut map = NormalMap::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let n = if (x + y) % 2 == 0 {
                    normalize3([0.3, 0.0, 1.0]).unwrap()
                } else {
                    normalize3([-0.3, 0.0, 1.0]).unwrap()
                };
                map.set_pixel(x, y, n);
            }
        }
        let ref_normal = [0.0_f32, 0.0, 1.0];
        let mad_before = map.mean_angular_deviation(ref_normal);
        let smoothed = smooth_normals(&map, 2.0);
        let mad_after = smoothed.mean_angular_deviation(ref_normal);
        assert!(
            mad_after < mad_before,
            "smoothing should reduce MAD: before={mad_before}, after={mad_after}"
        );
    }

    // ─── normal_consistency_loss ─────────────────────────────────────────────

    /// Test 14: Normal consistency loss of a map against itself → 0.0.
    #[test]
    fn test_consistency_loss_self() {
        let depth = flat_depth_map(5, 5, 2.0);
        let map = estimate_normals_cross_product(&depth, 5, 5, 100.0, 100.0, 2.5, 2.5).unwrap();
        let loss = normal_consistency_loss(&map, &map).unwrap();
        assert!(
            loss.abs() < EPSILON,
            "self-consistency loss should be 0, got {loss}"
        );
    }

    /// Test 15: Dimension mismatch → Err(InvalidDimensions).
    #[test]
    fn test_consistency_loss_dimension_mismatch() {
        let map_a = NormalMap::new(4, 4);
        let map_b = NormalMap::new(5, 5);
        let result = normal_consistency_loss(&map_a, &map_b);
        assert!(matches!(result, Err(NormalError::InvalidDimensions(_))));
    }

    /// Test 16: One all-zero map (all-INFINITY depth) vs one valid map → 0.0.
    #[test]
    fn test_consistency_loss_one_all_zero() {
        let depth_inf = vec![f32::INFINITY; 16];
        let depth_valid = flat_depth_map(4, 4, 1.0);
        let map_zero =
            estimate_normals_cross_product(&depth_inf, 4, 4, 100.0, 100.0, 2.0, 2.0).unwrap();
        let map_valid =
            estimate_normals_cross_product(&depth_valid, 4, 4, 100.0, 100.0, 2.0, 2.0).unwrap();
        // No pixel pairs are both valid, so loss must be 0.0.
        let loss = normal_consistency_loss(&map_zero, &map_valid).unwrap();
        assert!(
            loss.abs() < EPSILON,
            "loss with all-invalid map should be 0, got {loss}"
        );
    }

    // ─── mean_normal ─────────────────────────────────────────────────────────

    /// Test 17: Mean normal of uniform [0,0,1] map → [0,0,1].
    #[test]
    fn test_mean_normal_uniform_map() {
        let map = NormalMap::new(4, 4);
        let mean = map.mean_normal();
        assert!((mean[0]).abs() < EPSILON);
        assert!((mean[1]).abs() < EPSILON);
        assert!((mean[2] - 1.0).abs() < EPSILON);
    }

    // ─── angular_deviation ───────────────────────────────────────────────────

    /// Test 18: Angular deviation of [0,0,1] vs [0,0,1] reference → 0.0 per pixel.
    #[test]
    fn test_angular_deviation_zero() {
        let map = NormalMap::new(3, 3);
        let devs = map.angular_deviation([0.0, 0.0, 1.0]);
        for &d in &devs {
            assert!(d.abs() < EPSILON, "expected 0.0, got {d}");
        }
    }

    /// Test 19: Angular deviation of [0,0,1] vs [0,0,-1] reference → π per pixel.
    #[test]
    fn test_angular_deviation_opposite() {
        let map = NormalMap::new(2, 2);
        let devs = map.angular_deviation([0.0, 0.0, -1.0]);
        for &d in &devs {
            assert!(
                (d - std::f32::consts::PI).abs() < EPSILON,
                "expected π, got {d}"
            );
        }
    }

    // ─── compute_normal_stats ────────────────────────────────────────────────

    /// Test 20: Stats for flat normal map → roughness near 0.
    #[test]
    fn test_normal_stats_flat_map() {
        let map = NormalMap::new(8, 8);
        let stats = compute_normal_stats(&map);
        assert!(
            stats.roughness < 0.01,
            "roughness should be ~0 for flat map, got {}",
            stats.roughness
        );
        assert_eq!(stats.num_valid_pixels, 64);
        assert!((stats.coverage - 1.0).abs() < EPSILON);
    }

    // ─── estimate_curvature ───────────────────────────────────────────────────

    /// Test 21: Curvature of flat surface → values near 0.
    #[test]
    fn test_curvature_flat_surface() {
        let map = NormalMap::new(8, 8);
        let curv = estimate_curvature(&map);
        for &c in &curv {
            assert!(
                c.abs() < EPSILON,
                "curvature should be ~0 for flat surface, got {c}"
            );
        }
    }

    /// Test 22: Output size equals width × height.
    #[test]
    fn test_curvature_output_size() {
        let w = 7u32;
        let h = 5u32;
        let map = NormalMap::new(w, h);
        let curv = estimate_curvature(&map);
        assert_eq!(curv.len(), (w * h) as usize);
    }

    // ─── Additional edge-case tests ───────────────────────────────────────────

    /// Test 23: set_pixel respects bounds; out-of-bounds write is ignored.
    #[test]
    fn test_set_pixel_out_of_bounds_ignored() {
        let mut map = NormalMap::new(4, 4);
        map.set_pixel(100, 100, [1.0, 0.0, 0.0]);
        // Underlying data should be unchanged.
        for y in 0..4 {
            for x in 0..4 {
                let n = map.pixel(x, y);
                assert!((n[2] - 1.0).abs() < EPSILON);
            }
        }
    }

    /// Test 24: estimate_normals_cross_product with negative depth → Err(InvalidDepth).
    #[test]
    fn test_cross_product_negative_depth_err() {
        let depth = vec![-1.0_f32; 4];
        let result = estimate_normals_cross_product(&depth, 2, 2, 100.0, 100.0, 1.0, 1.0);
        assert!(matches!(result, Err(NormalError::InvalidDepth(_))));
    }

    /// Test 25: smooth_normals with sigma <= 0 returns clone of original.
    #[test]
    fn test_smooth_zero_sigma_returns_clone() {
        let depth = flat_depth_map(5, 5, 1.5);
        let map = estimate_normals_sobel(&depth, 5, 5).unwrap();
        let smoothed = smooth_normals(&map, 0.0);
        for y in 0..5 {
            for x in 0..5 {
                let orig = map.pixel(x, y);
                let sm = smoothed.pixel(x, y);
                assert_eq!(orig, sm);
            }
        }
    }
}
