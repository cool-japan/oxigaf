//! Panoramic rendering: equirectangular ↔ perspective conversions,
//! cube map extraction, multi-view stitching, and sphere sampling utilities.
//!
//! # Coordinate conventions
//!
//! Equirectangular images map:
//! - Longitude θ ∈ \[-π, π\] → column x (left: -X, front: +Z, right: +X, back: ±Z)
//! - Latitude  φ ∈ \[-π/2, π/2\] → row y (top: +Y, bottom: -Y)
//!
//! A direction vector **d** from (θ, φ) is:
//! ```text
//! d.x = cos(φ) * sin(θ)
//! d.y = sin(φ)
//! d.z = cos(φ) * cos(θ)
//! ```

use std::f32::consts::PI;
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by panoramic rendering operations.
#[derive(Debug, Error)]
pub enum PanoramicError {
    /// Invalid configuration parameter.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Image buffer has the wrong number of elements.
    #[error("Invalid image: {0}")]
    InvalidImage(String),

    /// Image buffer is empty (zero pixels).
    #[error("Empty image: no pixels provided")]
    EmptyImage,

    /// Dimension mismatch between supplied and expected sizes.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements.
        got: usize,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Low-level coordinate conversions
// ─────────────────────────────────────────────────────────────────────────────

/// Convert equirectangular pixel coordinates to a unit 3-D direction vector.
///
/// The centre of pixel `(px, py)` is sampled, so callers should pass
/// `px = col + 0.5`, `py = row + 0.5` for a texel-centred sample.
///
/// # Returns
///
/// `[x, y, z]` unit direction. The vector is always normalised.
pub fn equirect_to_direction(px: f32, py: f32, eq_width: usize, eq_height: usize) -> [f32; 3] {
    let theta = (px / eq_width as f32 - 0.5) * 2.0 * PI;
    let phi = (0.5 - py / eq_height as f32) * PI;
    let cos_phi = phi.cos();
    let dir = [cos_phi * theta.sin(), phi.sin(), cos_phi * theta.cos()];
    normalize(dir)
}

/// Convert a 3-D direction vector to equirectangular pixel coordinates.
///
/// # Returns
///
/// `(px, py)` where `px ∈ [0, eq_width)` and `py ∈ [0, eq_height)`.
pub fn direction_to_equirect(dir: [f32; 3], eq_width: usize, eq_height: usize) -> (f32, f32) {
    let n = normalize(dir);
    let theta = n[0].atan2(n[2]); // longitude
    let phi = n[1].clamp(-1.0, 1.0).asin(); // latitude
    let px = (theta / (2.0 * PI) + 0.5) * eq_width as f32;
    let py = (0.5 - phi / PI) * eq_height as f32;
    (px, py)
}

// ─────────────────────────────────────────────────────────────────────────────
// PanoramicCamera
// ─────────────────────────────────────────────────────────────────────────────

/// A pinhole camera used for panoramic rendering.
///
/// `rotation` is a row-major 3×3 orthonormal matrix that converts world-space
/// directions into camera-space directions.  The camera +Z axis looks *into*
/// the scene in camera space.
#[derive(Debug, Clone)]
pub struct PanoramicCamera {
    /// Camera position in world space.
    pub position: [f32; 3],
    /// Row-major 3×3 rotation matrix (world → camera).
    pub rotation: [[f32; 3]; 3],
}

impl PanoramicCamera {
    /// Construct a camera with an arbitrary pose.
    pub fn new(position: [f32; 3], rotation: [[f32; 3]; 3]) -> Self {
        Self { position, rotation }
    }

    /// Camera placed at the origin, looking along +Z, with +Y up.
    ///
    /// This is the identity pose: `rotation` is the 3×3 identity matrix.
    pub fn identity() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Build a camera at `position` looking in direction `forward`.
    ///
    /// The world up hint is `[0, 1, 0]`.  When `forward` is (anti-)parallel to
    /// world up, `[0, 0, 1]` is used as a fallback so the cross product never
    /// degenerates.
    pub fn looking_at(position: [f32; 3], forward: [f32; 3]) -> Self {
        let fwd = normalize(forward);

        // Choose an up hint that is not parallel to forward.
        let world_up = [0.0f32, 1.0, 0.0];
        let up_hint = if cross(fwd, world_up).iter().map(|v| v * v).sum::<f32>() < 1e-6 {
            [0.0f32, 0.0, 1.0]
        } else {
            world_up
        };

        let right = normalize(cross(fwd, up_hint));
        let up = normalize(cross(right, fwd));
        // rows of the rotation matrix: [right, up, forward]
        let rotation = [right, up, fwd];
        Self { position, rotation }
    }

    /// Rotate a world-space direction into camera space.
    pub fn world_to_camera_dir(&self, world_dir: [f32; 3]) -> [f32; 3] {
        mat3_vec_mul(&self.rotation, world_dir)
    }

    /// Rotate a camera-space direction into world space.
    ///
    /// Because the matrix is orthonormal the inverse is its transpose.
    pub fn camera_to_world_dir(&self, cam_dir: [f32; 3]) -> [f32; 3] {
        mat3_transpose_vec_mul(&self.rotation, cam_dir)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PerspectiveView
// ─────────────────────────────────────────────────────────────────────────────

/// A single perspective (pinhole) camera view to be warped into a panorama.
#[derive(Debug, Clone)]
pub struct PerspectiveView {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Horizontal field of view in radians.
    pub fov_x: f32,
    /// Camera configuration (pose + intrinsics basis).
    pub camera: PanoramicCamera,
}

impl PerspectiveView {
    /// Construct a perspective view.
    pub fn new(width: usize, height: usize, fov_x: f32, camera: PanoramicCamera) -> Self {
        Self {
            width,
            height,
            fov_x,
            camera,
        }
    }

    /// Vertical field of view derived from `fov_x` and the image aspect ratio.
    pub fn fov_y(&self) -> f32 {
        2.0 * ((self.fov_x / 2.0).tan() * self.height as f32 / self.width as f32).atan()
    }

    /// Project a world-space direction to image pixel coordinates.
    ///
    /// Returns `None` when the direction points behind the camera or projects
    /// outside the image boundary.
    pub fn project_direction(&self, world_dir: [f32; 3]) -> Option<(f32, f32)> {
        let cam = self.camera.world_to_camera_dir(world_dir);
        let cam_z = cam[2];
        if cam_z <= 0.0 {
            return None;
        }
        let focal_x = self.width as f32 / (2.0 * (self.fov_x / 2.0).tan());
        let focal_y = self.height as f32 / (2.0 * (self.fov_y() / 2.0).tan());
        let px = self.width as f32 / 2.0 + focal_x * cam[0] / cam_z;
        let py = self.height as f32 / 2.0 - focal_y * cam[1] / cam_z;
        if px < 0.0 || px >= self.width as f32 || py < 0.0 || py >= self.height as f32 {
            return None;
        }
        Some((px, py))
    }

    /// Unproject pixel `(px, py)` to a normalised world-space direction.
    pub fn unproject_pixel(&self, px: f32, py: f32) -> [f32; 3] {
        let focal_x = self.width as f32 / (2.0 * (self.fov_x / 2.0).tan());
        let focal_y = self.height as f32 / (2.0 * (self.fov_y() / 2.0).tan());
        let cam_x = (px - self.width as f32 / 2.0) / focal_x;
        let cam_y = (self.height as f32 / 2.0 - py) / focal_y;
        let cam_z = 1.0_f32;
        let cam_dir = normalize([cam_x, cam_y, cam_z]);
        self.camera.camera_to_world_dir(cam_dir)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Perspective → equirectangular warp
// ─────────────────────────────────────────────────────────────────────────────

/// Warp a single perspective image into an equirectangular panorama patch.
///
/// `perspective_image` must be an RGB f32 row-major slice of length
/// `view.width * view.height * 3`.  Pixels not covered by the perspective view
/// are left as `[0, 0, 0]`.
///
/// # Errors
///
/// Returns [`PanoramicError::EmptyImage`] when the slice is empty, or
/// [`PanoramicError::InvalidImage`] when its length does not match the view
/// dimensions.
pub fn perspective_to_equirect(
    perspective_image: &[f32],
    view: &PerspectiveView,
    eq_width: usize,
    eq_height: usize,
) -> Result<Vec<f32>, PanoramicError> {
    let expected = view.width * view.height * 3;
    if perspective_image.is_empty() {
        return Err(PanoramicError::EmptyImage);
    }
    if perspective_image.len() != expected {
        return Err(PanoramicError::InvalidImage(format!(
            "expected {} elements for {}×{}×3 image, got {}",
            expected,
            view.width,
            view.height,
            perspective_image.len()
        )));
    }

    let total = eq_width * eq_height * 3;
    let mut output = vec![0.0_f32; total];

    for ey in 0..eq_height {
        for ex in 0..eq_width {
            let dir = equirect_to_direction(ex as f32 + 0.5, ey as f32 + 0.5, eq_width, eq_height);
            if let Some((px, py)) = view.project_direction(dir) {
                let rgb = sample_bilinear_rgb(perspective_image, view.width, view.height, px, py);
                let out_idx = (ey * eq_width + ex) * 3;
                output[out_idx] = rgb[0];
                output[out_idx + 1] = rgb[1];
                output[out_idx + 2] = rgb[2];
            }
        }
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-view stitching
// ─────────────────────────────────────────────────────────────────────────────

/// Stitch multiple perspective views into one equirectangular panorama.
///
/// Overlapping regions are blended with equal weight (average of all
/// contributing views).  Pixels covered by no view remain black.
///
/// Each image in `views_and_images` must have length
/// `view.width * view.height * 3`.
///
/// # Errors
///
/// Propagates [`PanoramicError`] from any malformed image.
pub fn stitch_to_equirect(
    views_and_images: &[(PerspectiveView, Vec<f32>)],
    eq_width: usize,
    eq_height: usize,
) -> Result<Vec<f32>, PanoramicError> {
    let total = eq_width * eq_height * 3;
    let mut accum = vec![0.0_f32; total];
    let mut weights = vec![0.0_f32; eq_width * eq_height];

    for (view, image) in views_and_images {
        let expected = view.width * view.height * 3;
        if image.is_empty() {
            return Err(PanoramicError::EmptyImage);
        }
        if image.len() != expected {
            return Err(PanoramicError::InvalidImage(format!(
                "expected {} elements for {}×{}×3 image, got {}",
                expected,
                view.width,
                view.height,
                image.len()
            )));
        }

        for ey in 0..eq_height {
            for ex in 0..eq_width {
                let dir =
                    equirect_to_direction(ex as f32 + 0.5, ey as f32 + 0.5, eq_width, eq_height);
                if let Some((px, py)) = view.project_direction(dir) {
                    let rgb = sample_bilinear_rgb(image, view.width, view.height, px, py);
                    let out_idx = (ey * eq_width + ex) * 3;
                    accum[out_idx] += rgb[0];
                    accum[out_idx + 1] += rgb[1];
                    accum[out_idx + 2] += rgb[2];
                    weights[ey * eq_width + ex] += 1.0;
                }
            }
        }
    }

    // Normalise by total weight
    for ey in 0..eq_height {
        for ex in 0..eq_width {
            let w = weights[ey * eq_width + ex];
            if w > 0.0 {
                let idx = (ey * eq_width + ex) * 3;
                accum[idx] /= w;
                accum[idx + 1] /= w;
                accum[idx + 2] /= w;
            }
        }
    }

    Ok(accum)
}

// ─────────────────────────────────────────────────────────────────────────────
// Cube map
// ─────────────────────────────────────────────────────────────────────────────

/// Identifies one face of a cube map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeFace {
    /// +X (right)
    PosX = 0,
    /// -X (left)
    NegX = 1,
    /// +Y (top)
    PosY = 2,
    /// -Y (bottom)
    NegY = 3,
    /// +Z (front)
    PosZ = 4,
    /// -Z (back)
    NegZ = 5,
}

impl CubeFace {
    /// All six cube-map faces in enum-ordinal order.
    pub fn all() -> [CubeFace; 6] {
        [
            CubeFace::PosX,
            CubeFace::NegX,
            CubeFace::PosY,
            CubeFace::NegY,
            CubeFace::PosZ,
            CubeFace::NegZ,
        ]
    }

    /// The outward-facing axis direction for this face.
    pub fn forward_direction(&self) -> [f32; 3] {
        match self {
            CubeFace::PosX => [1.0, 0.0, 0.0],
            CubeFace::NegX => [-1.0, 0.0, 0.0],
            CubeFace::PosY => [0.0, 1.0, 0.0],
            CubeFace::NegY => [0.0, -1.0, 0.0],
            CubeFace::PosZ => [0.0, 0.0, 1.0],
            CubeFace::NegZ => [0.0, 0.0, -1.0],
        }
    }
}

/// Convert an equirectangular image to a single cube-map face.
///
/// `equirect` must be an RGB f32 row-major slice of length
/// `eq_width * eq_height * 3`.  The output face is a square of
/// `face_size * face_size * 3` RGB f32 values.
///
/// # Errors
///
/// Returns [`PanoramicError::EmptyImage`] or [`PanoramicError::InvalidImage`]
/// when the source buffer is malformed.
pub fn equirect_to_cube_face(
    equirect: &[f32],
    eq_width: usize,
    eq_height: usize,
    face: CubeFace,
    face_size: usize,
) -> Result<Vec<f32>, PanoramicError> {
    let expected = eq_width * eq_height * 3;
    if equirect.is_empty() {
        return Err(PanoramicError::EmptyImage);
    }
    if equirect.len() != expected {
        return Err(PanoramicError::InvalidImage(format!(
            "expected {} elements for {}×{}×3 equirect, got {}",
            expected,
            eq_width,
            eq_height,
            equirect.len()
        )));
    }
    if face_size == 0 {
        return Err(PanoramicError::InvalidConfig(
            "face_size must be > 0".to_string(),
        ));
    }

    let mut output = vec![0.0_f32; face_size * face_size * 3];

    for fy in 0..face_size {
        for fx in 0..face_size {
            let u = (fx as f32 + 0.5) / face_size as f32 * 2.0 - 1.0;
            let v = (fy as f32 + 0.5) / face_size as f32 * 2.0 - 1.0;

            let raw = match face {
                CubeFace::PosX => [1.0_f32, -v, -u],
                CubeFace::NegX => [-1.0_f32, -v, u],
                CubeFace::PosY => [u, 1.0_f32, v],
                CubeFace::NegY => [u, -1.0_f32, -v],
                CubeFace::PosZ => [u, -v, 1.0_f32],
                CubeFace::NegZ => [-u, -v, -1.0_f32],
            };
            let dir = normalize(raw);

            let (eq_px, eq_py) = direction_to_equirect(dir, eq_width, eq_height);
            let rgb = sample_bilinear_rgb(equirect, eq_width, eq_height, eq_px, eq_py);

            let out_idx = (fy * face_size + fx) * 3;
            output[out_idx] = rgb[0];
            output[out_idx + 1] = rgb[1];
            output[out_idx + 2] = rgb[2];
        }
    }

    Ok(output)
}

/// Convert an equirectangular image to all six cube-map faces.
///
/// Returns a `Vec<Vec<f32>>` of length 6 ordered by [`CubeFace`] discriminant.
/// Each inner `Vec<f32>` has length `face_size * face_size * 3`.
///
/// # Errors
///
/// Propagates errors from [`equirect_to_cube_face`].
pub fn equirect_to_cubemap(
    equirect: &[f32],
    eq_width: usize,
    eq_height: usize,
    face_size: usize,
) -> Result<Vec<Vec<f32>>, PanoramicError> {
    CubeFace::all()
        .iter()
        .map(|&face| equirect_to_cube_face(equirect, eq_width, eq_height, face, face_size))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Sphere-sampling utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Sample `n` viewpoints distributed near-uniformly on the unit sphere using
/// the Fibonacci lattice.
///
/// Returns a `Vec` of `(theta, phi)` pairs where
/// - `theta ∈ [-π, π]` is the longitude and
/// - `phi ∈ [-π/2, π/2]` is the latitude.
///
/// When `n == 0` an empty vec is returned.
pub fn fibonacci_sphere_views(n: usize) -> Vec<(f32, f32)> {
    if n == 0 {
        return Vec::new();
    }
    const GOLDEN_RATIO: f32 = 1.618_033_9;
    let denom = (n - 1).max(1) as f32;
    (0..n)
        .map(|i| {
            let phi = (1.0 - 2.0 * i as f32 / denom).clamp(-1.0, 1.0).asin();
            let raw_theta = 2.0 * PI * i as f32 * GOLDEN_RATIO;
            // Wrap into [-π, π]
            let theta = ((raw_theta + PI).rem_euclid(2.0 * PI)) - PI;
            (theta, phi)
        })
        .collect()
}

/// Build a [`PanoramicCamera`] positioned on a sphere of radius `distance`,
/// looking inward toward the origin.
///
/// `theta` is the longitude, `phi` is the latitude (both in radians).
pub fn camera_from_angles(theta: f32, phi: f32, distance: f32) -> PanoramicCamera {
    let x = phi.cos() * theta.sin() * distance;
    let y = phi.sin() * distance;
    let z = phi.cos() * theta.cos() * distance;
    let position = [x, y, z];
    // forward = direction from position toward origin = -position_normalized
    let forward = normalize([-x, -y, -z]);
    PanoramicCamera::looking_at(position, forward)
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics for a completed equirectangular panorama.
#[derive(Debug, Clone)]
pub struct PanoramicStats {
    /// Panoramic image width.
    pub eq_width: usize,
    /// Panoramic image height.
    pub eq_height: usize,
    /// Fraction of equirect pixels that received a non-zero value.
    pub covered_fraction: f32,
    /// Mean BT.709 luminance over covered pixels.  `0.0` when no pixels are covered.
    pub mean_luminance: f32,
    /// Number of views that were stitched.
    pub num_views: usize,
}

/// Compute statistics for a completed equirectangular panorama.
///
/// `equirect` must be an RGB f32 row-major slice of length
/// `eq_width * eq_height * 3`.
///
/// # Errors
///
/// Returns [`PanoramicError::EmptyImage`] when the slice is empty or
/// [`PanoramicError::InvalidImage`] when its length does not match the
/// specified dimensions.
pub fn compute_panoramic_stats(
    equirect: &[f32],
    eq_width: usize,
    eq_height: usize,
    num_views: usize,
) -> Result<PanoramicStats, PanoramicError> {
    let expected = eq_width * eq_height * 3;
    if equirect.is_empty() {
        return Err(PanoramicError::EmptyImage);
    }
    if equirect.len() != expected {
        return Err(PanoramicError::InvalidImage(format!(
            "expected {} elements for {}×{}×3 image, got {}",
            expected,
            eq_width,
            eq_height,
            equirect.len()
        )));
    }

    let num_pixels = eq_width * eq_height;
    let mut covered = 0u32;
    let mut lum_sum = 0.0_f32;

    for i in 0..num_pixels {
        let r = equirect[i * 3];
        let g = equirect[i * 3 + 1];
        let b = equirect[i * 3 + 2];
        // BT.709 luminance coefficients
        let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        if r != 0.0 || g != 0.0 || b != 0.0 {
            covered += 1;
            lum_sum += lum;
        }
    }

    let covered_fraction = covered as f32 / num_pixels as f32;
    let mean_luminance = if covered > 0 {
        lum_sum / covered as f32
    } else {
        0.0
    };

    Ok(PanoramicStats {
        eq_width,
        eq_height,
        covered_fraction,
        mean_luminance,
        num_views,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Normalise a 3-D vector.  Returns `[0, 0, 1]` for the zero vector so that
/// callers never receive NaN directions.
#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len2 < f32::EPSILON {
        return [0.0, 0.0, 1.0];
    }
    let inv = len2.sqrt().recip();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Cross product of two 3-D vectors.
#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Row-major 3×3 matrix × column vector.
#[inline]
fn mat3_vec_mul(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Transpose of a row-major 3×3 matrix × column vector (= Mᵀv).
#[inline]
fn mat3_transpose_vec_mul(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// Bilinear sample from an RGB f32 row-major image.
///
/// Coordinates are in pixel space (not normalised UV).  Out-of-bounds accesses
/// are clamped to the edge texel (clamp-to-edge wrapping).
#[inline]
fn sample_bilinear_rgb(image: &[f32], width: usize, height: usize, px: f32, py: f32) -> [f32; 3] {
    // Map to texel centre coordinates then split into integer + fractional part
    let tx = px - 0.5;
    let ty = py - 0.5;
    let x0 = tx.floor() as i32;
    let y0 = ty.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = tx - tx.floor();
    let fy = ty - ty.floor();

    let x0c = x0.clamp(0, width as i32 - 1) as usize;
    let y0c = y0.clamp(0, height as i32 - 1) as usize;
    let x1c = x1.clamp(0, width as i32 - 1) as usize;
    let y1c = y1.clamp(0, height as i32 - 1) as usize;

    let fetch = |row: usize, col: usize| -> [f32; 3] {
        let base = (row * width + col) * 3;
        [image[base], image[base + 1], image[base + 2]]
    };

    let c00 = fetch(y0c, x0c);
    let c10 = fetch(y0c, x1c);
    let c01 = fetch(y1c, x0c);
    let c11 = fetch(y1c, x1c);

    let lerp3 = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ]
    };

    let top = lerp3(c00, c10, fx);
    let bot = lerp3(c01, c11, fx);
    lerp3(top, bot, fy)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EQ_W: usize = 256;
    const EQ_H: usize = 128;

    fn vec_approx_eq(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps && (a[2] - b[2]).abs() < eps
    }

    // ── equirect_to_direction ────────────────────────────────────────────────

    #[test]
    fn test_equirect_to_direction_center_is_forward() {
        // Centre pixel should point along +Z (forward)
        let dir = equirect_to_direction(EQ_W as f32 / 2.0, EQ_H as f32 / 2.0, EQ_W, EQ_H);
        assert!(vec_approx_eq(dir, [0.0, 0.0, 1.0], 1e-5), "dir={dir:?}");
    }

    #[test]
    fn test_equirect_to_direction_top_is_up() {
        // py=0 → φ = π/2 → +Y
        let dir = equirect_to_direction(EQ_W as f32 / 2.0, 0.0, EQ_W, EQ_H);
        assert!(dir[1] > 0.9, "expected mostly +Y, got {dir:?}");
    }

    #[test]
    fn test_equirect_to_direction_bottom_is_down() {
        // py=EQ_H → φ = -π/2 → -Y
        let dir = equirect_to_direction(EQ_W as f32 / 2.0, EQ_H as f32, EQ_W, EQ_H);
        assert!(dir[1] < -0.9, "expected mostly -Y, got {dir:?}");
    }

    #[test]
    fn test_equirect_to_direction_right_is_positive_x() {
        // px = 3/4 * EQ_W → θ = π/2 → +X dominant
        let dir = equirect_to_direction(EQ_W as f32 * 0.75, EQ_H as f32 / 2.0, EQ_W, EQ_H);
        assert!(dir[0] > 0.9, "expected mostly +X, got {dir:?}");
    }

    #[test]
    fn test_equirect_to_direction_left_is_negative_x() {
        // px = 1/4 * EQ_W → θ = -π/2 → -X dominant
        let dir = equirect_to_direction(EQ_W as f32 * 0.25, EQ_H as f32 / 2.0, EQ_W, EQ_H);
        assert!(dir[0] < -0.9, "expected mostly -X, got {dir:?}");
    }

    // ── direction_to_equirect ────────────────────────────────────────────────

    #[test]
    fn test_direction_to_equirect_forward_is_center() {
        let (px, py) = direction_to_equirect([0.0, 0.0, 1.0], EQ_W, EQ_H);
        assert!((px - EQ_W as f32 / 2.0).abs() < 1.0, "px={px}");
        assert!((py - EQ_H as f32 / 2.0).abs() < 1.0, "py={py}");
    }

    #[test]
    fn test_direction_to_equirect_roundtrip() {
        // Test a variety of directions
        let directions = [
            [1.0_f32, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.577, 0.577, 0.577_f32],
        ];
        for orig in directions {
            let orig_n = normalize(orig);
            let (px, py) = direction_to_equirect(orig_n, EQ_W, EQ_H);
            let recovered = equirect_to_direction(px, py, EQ_W, EQ_H);
            assert!(
                vec_approx_eq(orig_n, recovered, 0.01),
                "roundtrip failed for {orig_n:?}: got {recovered:?}"
            );
        }
    }

    #[test]
    fn test_direction_to_equirect_top_is_near_zero_py() {
        let (_, py) = direction_to_equirect([0.0, 1.0, 0.0], EQ_W, EQ_H);
        assert!(py < 2.0, "top direction should map near py=0, got {py}");
    }

    // ── PanoramicCamera ──────────────────────────────────────────────────────

    #[test]
    fn test_panoramic_camera_identity() {
        let cam = PanoramicCamera::identity();
        let dir = [1.0_f32, 0.0, 0.0];
        let cam_dir = cam.world_to_camera_dir(dir);
        assert!(vec_approx_eq(cam_dir, dir, 1e-6));
    }

    #[test]
    fn test_panoramic_camera_looking_at_forward_z() {
        // looking at +Z from origin → rotation should be identity-like
        let cam = PanoramicCamera::looking_at([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        let world_z = [0.0_f32, 0.0, 1.0];
        let cam_z = cam.world_to_camera_dir(world_z);
        // After looking_at(forward=+Z): forward is stored in row 2
        assert!(
            cam_z[2] > 0.9,
            "Z component should be dominant, got {cam_z:?}"
        );
    }

    #[test]
    fn test_panoramic_camera_world_to_camera_roundtrip() {
        let cam = PanoramicCamera::looking_at([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]);
        let world_dirs: [[f32; 3]; 4] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            normalize([1.0, 1.0, 1.0]),
        ];
        for wd in world_dirs {
            let cam_d = cam.world_to_camera_dir(wd);
            let recovered = cam.camera_to_world_dir(cam_d);
            assert!(
                vec_approx_eq(wd, recovered, 1e-5),
                "roundtrip failed for {wd:?}: got {recovered:?}"
            );
        }
    }

    #[test]
    fn test_panoramic_camera_looking_at_degenerate_up() {
        // forward = [0,1,0] is parallel to world up; should not panic
        let cam = PanoramicCamera::looking_at([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let cam_fwd = cam.world_to_camera_dir([0.0, 1.0, 0.0]);
        let len =
            (cam_fwd[0] * cam_fwd[0] + cam_fwd[1] * cam_fwd[1] + cam_fwd[2] * cam_fwd[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-5,
            "direction should be unit, got len={len}"
        );
    }

    // ── PerspectiveView ──────────────────────────────────────────────────────

    #[test]
    fn test_perspective_view_fov_y_square() {
        let view = PerspectiveView::new(
            64,
            64,
            std::f32::consts::FRAC_PI_2,
            PanoramicCamera::identity(),
        );
        let fov_y = view.fov_y();
        assert!(
            (fov_y - std::f32::consts::FRAC_PI_2).abs() < 1e-5,
            "square image should have fov_y == fov_x, got {fov_y}"
        );
    }

    #[test]
    fn test_perspective_view_project_forward_is_center() {
        let w = 64;
        let h = 48;
        let view = PerspectiveView::new(w, h, 1.0, PanoramicCamera::identity());
        let result = view.project_direction([0.0, 0.0, 1.0]);
        assert!(result.is_some(), "forward should project to Some");
        let (px, py) = result.unwrap();
        assert!((px - w as f32 / 2.0).abs() < 1.0, "px={px}");
        assert!((py - h as f32 / 2.0).abs() < 1.0, "py={py}");
    }

    #[test]
    fn test_perspective_view_project_behind_is_none() {
        let view = PerspectiveView::new(64, 64, 1.0, PanoramicCamera::identity());
        let result = view.project_direction([0.0, 0.0, -1.0]);
        assert!(
            result.is_none(),
            "behind-camera direction should return None"
        );
    }

    #[test]
    fn test_perspective_view_unproject_center_is_forward() {
        let w = 64_usize;
        let h = 64_usize;
        let view = PerspectiveView::new(w, h, 1.0, PanoramicCamera::identity());
        let dir = view.unproject_pixel(w as f32 / 2.0, h as f32 / 2.0);
        assert!(
            dir[2] > 0.9,
            "center pixel should unproject to ~+Z, got {dir:?}"
        );
    }

    #[test]
    fn test_perspective_view_unproject_corner_has_nonzero_xy() {
        let w = 64_usize;
        let h = 64_usize;
        let view = PerspectiveView::new(w, h, 1.0, PanoramicCamera::identity());
        // Top-left corner
        let dir = view.unproject_pixel(0.0, 0.0);
        // x should be negative, y positive (top-left)
        assert!(dir[0] < 0.0, "top-left x should be negative, got {dir:?}");
        assert!(dir[1] > 0.0, "top-left y should be positive, got {dir:?}");
    }

    // ── perspective_to_equirect ──────────────────────────────────────────────

    #[test]
    fn test_perspective_to_equirect_empty_image_is_error() {
        let view = PerspectiveView::new(4, 4, 1.0, PanoramicCamera::identity());
        let result = perspective_to_equirect(&[], &view, 8, 4);
        assert!(matches!(result, Err(PanoramicError::EmptyImage)));
    }

    #[test]
    fn test_perspective_to_equirect_wrong_size_is_error() {
        let view = PerspectiveView::new(4, 4, 1.0, PanoramicCamera::identity());
        let bad = vec![0.5_f32; 10]; // wrong length
        let result = perspective_to_equirect(&bad, &view, 8, 4);
        assert!(matches!(result, Err(PanoramicError::InvalidImage(_))));
    }

    #[test]
    fn test_perspective_to_equirect_constant_color_covered_region() {
        // Create a small solid-red perspective image
        let w = 32_usize;
        let h = 32_usize;
        let image = [1.0_f32, 0.0, 0.0].repeat(w * h);
        let view = PerspectiveView::new(w, h, 1.0, PanoramicCamera::identity());

        let eq_w = 64_usize;
        let eq_h = 32_usize;
        let result = perspective_to_equirect(&image, &view, eq_w, eq_h);
        assert!(result.is_ok());
        let eq_img = result.unwrap();

        // The center of the panorama (front-facing pixels) should be red
        let cx = eq_w / 2;
        let cy = eq_h / 2;
        let idx = (cy * eq_w + cx) * 3;
        assert!(
            eq_img[idx] > 0.9,
            "center R should be ~1.0, got {}",
            eq_img[idx]
        );
        assert!(
            eq_img[idx + 1] < 0.1,
            "center G should be ~0.0, got {}",
            eq_img[idx + 1]
        );
        assert!(
            eq_img[idx + 2] < 0.1,
            "center B should be ~0.0, got {}",
            eq_img[idx + 2]
        );
    }

    // ── CubeFace ─────────────────────────────────────────────────────────────

    #[test]
    fn test_cube_face_all_returns_six() {
        assert_eq!(CubeFace::all().len(), 6);
    }

    #[test]
    fn test_cube_face_forward_pos_x() {
        let fwd = CubeFace::PosX.forward_direction();
        assert!(vec_approx_eq(fwd, [1.0, 0.0, 0.0], 1e-6));
    }

    #[test]
    fn test_cube_face_forward_all_unit() {
        for face in CubeFace::all() {
            let fwd = face.forward_direction();
            let len = (fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-6,
                "{face:?} forward not unit: {fwd:?}"
            );
        }
    }

    // ── equirect_to_cube_face / equirect_to_cubemap ──────────────────────────

    #[test]
    fn test_equirect_to_cube_face_solid_color_preserved() {
        // Solid green equirect image → every face should also be green
        let eq_w = 32_usize;
        let eq_h = 16_usize;
        let equirect = [0.0_f32, 1.0, 0.0].repeat(eq_w * eq_h);

        let face_size = 8_usize;
        let face = equirect_to_cube_face(&equirect, eq_w, eq_h, CubeFace::PosZ, face_size)
            .expect("should succeed");

        // All pixels should be green
        for i in 0..face_size * face_size {
            let r = face[i * 3];
            let g = face[i * 3 + 1];
            let b = face[i * 3 + 2];
            assert!(r < 0.05, "R should be ~0, got {r} at pixel {i}");
            assert!(g > 0.95, "G should be ~1, got {g} at pixel {i}");
            assert!(b < 0.05, "B should be ~0, got {b} at pixel {i}");
        }
    }

    #[test]
    fn test_equirect_to_cubemap_returns_six_faces() {
        let eq_w = 16_usize;
        let eq_h = 8_usize;
        let equirect = vec![0.5_f32; eq_w * eq_h * 3];
        let face_size = 4_usize;
        let faces = equirect_to_cubemap(&equirect, eq_w, eq_h, face_size).expect("should succeed");
        assert_eq!(faces.len(), 6);
        for (i, face) in faces.iter().enumerate() {
            assert_eq!(
                face.len(),
                face_size * face_size * 3,
                "face {i} has wrong length"
            );
        }
    }

    #[test]
    fn test_equirect_to_cube_face_empty_image_is_error() {
        let result = equirect_to_cube_face(&[], 8, 4, CubeFace::PosX, 4);
        assert!(matches!(result, Err(PanoramicError::EmptyImage)));
    }

    // ── fibonacci_sphere_views ───────────────────────────────────────────────

    #[test]
    fn test_fibonacci_sphere_views_zero() {
        let views = fibonacci_sphere_views(0);
        assert!(views.is_empty());
    }

    #[test]
    fn test_fibonacci_sphere_views_one() {
        let views = fibonacci_sphere_views(1);
        assert_eq!(views.len(), 1);
        let (theta, phi) = views[0];
        assert!(theta.is_finite());
        assert!(phi.is_finite());
    }

    #[test]
    fn test_fibonacci_sphere_views_ten_distinct() {
        let views = fibonacci_sphere_views(10);
        assert_eq!(views.len(), 10);
        // Check all angles are in valid ranges
        for (theta, phi) in &views {
            assert!(*theta >= -PI && *theta <= PI, "theta out of range: {theta}");
            assert!(
                *phi >= -PI / 2.0 && *phi <= PI / 2.0,
                "phi out of range: {phi}"
            );
        }
        // Check distinct: no two entries are identical
        for i in 0..views.len() {
            for j in (i + 1)..views.len() {
                let (t0, p0) = views[i];
                let (t1, p1) = views[j];
                let diff = (t0 - t1).abs() + (p0 - p1).abs();
                assert!(diff > 1e-4, "views {i} and {j} are too similar");
            }
        }
    }

    // ── camera_from_angles ───────────────────────────────────────────────────

    #[test]
    fn test_camera_from_angles_basic_construction() {
        let cam = camera_from_angles(0.0, 0.0, 5.0);
        // theta=0, phi=0 → position on +Z axis
        assert!(
            (cam.position[0]).abs() < 1e-5,
            "x should be 0, got {}",
            cam.position[0]
        );
        assert!(
            (cam.position[1]).abs() < 1e-5,
            "y should be 0, got {}",
            cam.position[1]
        );
        assert!(
            (cam.position[2] - 5.0).abs() < 1e-4,
            "z should be 5, got {}",
            cam.position[2]
        );
    }

    #[test]
    fn test_camera_from_angles_looks_at_origin() {
        let cam = camera_from_angles(0.5, 0.3, 3.0);
        // The forward direction in world space (row 2 of rotation applied to +Z cam) should
        // point approximately toward origin, i.e., opposite to position.
        let pos_n = normalize(cam.position);
        // Camera row 2 = forward direction (world space)
        let fwd_world = cam.camera_to_world_dir([0.0, 0.0, 1.0]);
        let dot = pos_n[0] * fwd_world[0] + pos_n[1] * fwd_world[1] + pos_n[2] * fwd_world[2];
        assert!(dot < -0.9, "camera should look toward origin, dot={dot}");
    }

    // ── stitch_to_equirect ───────────────────────────────────────────────────

    #[test]
    fn test_stitch_to_equirect_no_views_produces_black() {
        let result = stitch_to_equirect(&[], 16, 8).expect("should succeed");
        assert_eq!(result.len(), 16 * 8 * 3);
        assert!(
            result.iter().all(|&v| v == 0.0),
            "all pixels should be black"
        );
    }

    #[test]
    fn test_stitch_to_equirect_single_view_matches_perspective_to_equirect() {
        let w = 32_usize;
        let h = 32_usize;
        let image: Vec<f32> = [0.3_f32, 0.6, 0.9].repeat(w * h);
        let view = PerspectiveView::new(w, h, 1.0, PanoramicCamera::identity());
        let eq_w = 64_usize;
        let eq_h = 32_usize;

        let single = perspective_to_equirect(&image, &view, eq_w, eq_h).expect("should succeed");
        let stitched =
            stitch_to_equirect(&[(view.clone(), image)], eq_w, eq_h).expect("should succeed");

        assert_eq!(single.len(), stitched.len());
        for (i, (a, b)) in single.iter().zip(stitched.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "pixel {i}: single={a}, stitched={b}");
        }
    }

    // ── compute_panoramic_stats ──────────────────────────────────────────────

    #[test]
    fn test_compute_panoramic_stats_all_black_covered_zero() {
        let eq = vec![0.0_f32; 16 * 8 * 3];
        let stats = compute_panoramic_stats(&eq, 16, 8, 0).expect("should succeed");
        assert_eq!(stats.covered_fraction, 0.0);
        assert_eq!(stats.mean_luminance, 0.0);
    }

    #[test]
    fn test_compute_panoramic_stats_solid_white_fully_covered() {
        let eq = vec![1.0_f32; 16 * 8 * 3];
        let stats = compute_panoramic_stats(&eq, 16, 8, 2).expect("should succeed");
        assert!((stats.covered_fraction - 1.0).abs() < 1e-6);
        assert_eq!(stats.num_views, 2);
        // BT.709: 0.2126 + 0.7152 + 0.0722 ≈ 1.0
        assert!((stats.mean_luminance - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_compute_panoramic_stats_empty_is_error() {
        let result = compute_panoramic_stats(&[], 16, 8, 0);
        assert!(matches!(result, Err(PanoramicError::EmptyImage)));
    }
}
