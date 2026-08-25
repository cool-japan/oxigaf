//! CPU-side depth map generation from a 3D Gaussian Splatting scene.
//!
//! Depth maps are useful for visualisation, auxiliary supervision signals during
//! training, and depth-based compositing.
//!
//! # Overview
//!
//! - [`DepthCamera`]: Pinhole camera with pinhole projection and frustum check.
//! - [`GaussianDepthData`]: Bounding-sphere + opacity per Gaussian.
//! - [`DepthMode`]: How multiple depth samples at the same pixel are merged.
//! - [`DepthMap`]: The output raster — stores per-pixel depth (row-major).
//! - [`render_depth_map`]: Main entry point for a single camera.
//! - [`render_depth_maps`]: Convenience batch over many cameras.
//! - [`depth_to_disparity`]: Convert depth → disparity (focal / depth).
//! - [`depth_map_to_pointcloud`]: Unproject valid pixels to world-space 3-D points.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by depth-map operations.
#[derive(Debug, Error)]
pub enum DepthMapError {
    /// Invalid camera parameters.
    #[error("Invalid camera: {0}")]
    InvalidCamera(String),

    /// The Gaussian scene is empty — nothing to render.
    #[error("Empty scene: no Gaussians provided")]
    EmptyScene,

    /// Image dimensions are zero.
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// DepthCamera
// ─────────────────────────────────────────────────────────────────────────────

/// Pinhole camera used to project 3-D Gaussian centres into depth-map pixels.
///
/// The camera looks down **-Z** in camera space.  The `view_matrix_rotation`
/// is a row-major 3×3 matrix that maps **world → camera** directions:
///
/// ```text
/// cam = R * (world - position)
/// ```
#[derive(Debug, Clone)]
pub struct DepthCamera {
    /// Camera position in world space.
    pub position: [f32; 3],
    /// 3×3 rotation matrix (row-major) that rotates world-space vectors into
    /// camera space.  The camera looks down -Z in camera space.
    pub view_matrix_rotation: [f32; 9],
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Focal length along x in pixels.
    pub fx: f32,
    /// Focal length along y in pixels.
    pub fy: f32,
    /// Principal point x (column) in pixels.
    pub cx: f32,
    /// Principal point y (row) in pixels.
    pub cy: f32,
    /// Near clip plane distance (positive, in world units).
    pub near: f32,
    /// Far clip plane distance (positive, in world units).
    pub far: f32,
}

impl DepthCamera {
    /// Construct a [`DepthCamera`] from a vertical field-of-view.
    ///
    /// ```text
    /// fy = height / (2 * tan(fov_y / 2))
    /// fx = fy * (width / height)
    /// cx = width / 2,  cy = height / 2
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`DepthMapError::InvalidCamera`] when `fov_y_rad <= 0`,
    /// `width == 0`, `height == 0`, `near >= far`, or `near <= 0`.
    pub fn from_fov(
        position: [f32; 3],
        view_rotation: [f32; 9],
        width: u32,
        height: u32,
        fov_y_rad: f32,
        near: f32,
        far: f32,
    ) -> Result<Self, DepthMapError> {
        if fov_y_rad <= 0.0 || !fov_y_rad.is_finite() {
            return Err(DepthMapError::InvalidCamera(format!(
                "fov_y_rad must be positive and finite, got {fov_y_rad}"
            )));
        }
        if width == 0 {
            return Err(DepthMapError::InvalidCamera(
                "width must be non-zero".to_string(),
            ));
        }
        if height == 0 {
            return Err(DepthMapError::InvalidCamera(
                "height must be non-zero".to_string(),
            ));
        }
        if near <= 0.0 {
            return Err(DepthMapError::InvalidCamera(format!(
                "near must be positive, got {near}"
            )));
        }
        if far <= near {
            return Err(DepthMapError::InvalidCamera(format!(
                "far ({far}) must be greater than near ({near})"
            )));
        }

        let half_fov_tan = (fov_y_rad * 0.5).tan();
        let fy = height as f32 / (2.0 * half_fov_tan);
        let aspect = width as f32 / height as f32;
        let fx = fy * aspect;
        let cx = width as f32 * 0.5;
        let cy = height as f32 * 0.5;

        Ok(Self {
            position,
            view_matrix_rotation: view_rotation,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            near,
            far,
        })
    }

    /// Project a world-space point to `(pixel_x, pixel_y, depth)`.
    ///
    /// - `depth` is the **positive** depth along the camera -Z axis.
    /// - Returns `None` when the point is behind the camera (`depth <= 0`) or
    ///   the projected pixel lies outside the image rectangle.
    pub fn project(&self, world_point: [f32; 3]) -> Option<(f32, f32, f32)> {
        // Translate into camera space.
        let diff = [
            world_point[0] - self.position[0],
            world_point[1] - self.position[1],
            world_point[2] - self.position[2],
        ];

        // Apply row-major rotation: cam = R * diff
        let r = &self.view_matrix_rotation;
        let cam_x = r[0] * diff[0] + r[1] * diff[1] + r[2] * diff[2];
        let cam_y = r[3] * diff[0] + r[4] * diff[1] + r[5] * diff[2];
        let cam_z = r[6] * diff[0] + r[7] * diff[1] + r[8] * diff[2];

        // Camera looks down -Z; depth is the positive Z component.
        let depth = -cam_z;
        if depth <= 0.0 {
            return None;
        }

        // Pinhole projection (Y flipped for image coordinates).
        let px = self.fx * cam_x / depth + self.cx;
        let py = -self.fy * cam_y / depth + self.cy;

        // Bounds check: pixel must be within the image.
        let w = self.width as f32;
        let h = self.height as f32;
        if px < 0.0 || px >= w || py < 0.0 || py >= h {
            return None;
        }

        Some((px, py, depth))
    }

    /// Rough sphere-based frustum check.
    ///
    /// Returns `true` when the sphere `(center, radius)` might overlap the view
    /// frustum. This is a **conservative** (over-inclusive) broad-phase test:
    /// a `true` result does not guarantee the sphere is actually visible
    /// (false positives are expected, and cheap to filter downstream), but a
    /// `false` result must guarantee no part of the sphere can possibly be
    /// visible — false negatives are never acceptable, since they would
    /// silently drop a visible Gaussian from the depth map.
    pub fn in_frustum_approx(&self, center: [f32; 3], radius: f32) -> bool {
        // Transform sphere centre to camera space.
        let diff = [
            center[0] - self.position[0],
            center[1] - self.position[1],
            center[2] - self.position[2],
        ];
        let r = &self.view_matrix_rotation;
        let cam_x = r[0] * diff[0] + r[1] * diff[1] + r[2] * diff[2];
        let cam_y = r[3] * diff[0] + r[4] * diff[1] + r[5] * diff[2];
        let cam_z = r[6] * diff[0] + r[7] * diff[1] + r[8] * diff[2];

        let depth = -cam_z; // positive depth

        // Depth range check (with sphere radius slack).
        if depth + radius < self.near || depth - radius > self.far {
            return false;
        }
        // If the sphere is entirely in front but depth - radius <= 0, still
        // check — the front face of the sphere is at the camera.
        if depth <= 0.0 && depth + radius <= 0.0 {
            return false;
        }

        // Use the furthest visible depth for the projection — avoids division by
        // near-zero when the sphere straddles the camera.
        let ref_depth = if depth > radius {
            depth
        } else {
            depth + radius
        };
        if ref_depth <= 0.0 {
            return false;
        }

        // Project the sphere's true (tangent-line) angular radius rather than
        // its flat center-plane radius: for a sphere of radius `r` whose
        // centre is at distance `d` from the camera, the exact screen-space
        // projected radius is `f * r / sqrt(d^2 - r^2)`, which is always
        // *larger* than the naive `f * r / d` approximation. Using the naive
        // formula understates the sphere's true screen extent (worse near
        // the frustum edges) and can wrongly cull a sphere that is still
        // partially visible — exactly the false negative this function must
        // not produce. `ref_depth` is the camera-space Z depth rather than
        // the full 3-D Euclidean distance to the centre; since depth is
        // always <= the true distance, this only makes the result *more*
        // conservative (an equal-or-larger projected radius), never less.
        let denom = (ref_depth * ref_depth - radius * radius)
            .max(1e-6_f32)
            .sqrt();
        let proj_radius_x = self.fx * radius / denom;
        let proj_radius_y = self.fy * radius / denom;

        // Check if the projected centre ± radius overlaps the image
        // rectangle, testing x against `fx` and y against `fy` separately —
        // they differ for any non-square image (`fx = fy * aspect`).
        let px = self.fx * cam_x / ref_depth + self.cx;
        let py = -self.fy * cam_y / ref_depth + self.cy;

        let w = self.width as f32;
        let h = self.height as f32;

        if px + proj_radius_x < 0.0 || px - proj_radius_x >= w {
            return false;
        }
        if py + proj_radius_y < 0.0 || py - proj_radius_y >= h {
            return false;
        }

        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DepthSample
// ─────────────────────────────────────────────────────────────────────────────

/// A single depth sample contributed by one Gaussian to one pixel.
#[derive(Debug, Clone)]
pub struct DepthSample {
    /// Fractional pixel column.
    pub pixel_x: f32,
    /// Fractional pixel row.
    pub pixel_y: f32,
    /// Positive depth in camera space.
    pub depth: f32,
    /// Opacity of the contributing Gaussian (0..1).
    pub opacity: f32,
    /// Index of the Gaussian in the input slice.
    pub gaussian_index: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// GaussianDepthData
// ─────────────────────────────────────────────────────────────────────────────

/// Per-Gaussian data required for depth-map rendering.
#[derive(Debug, Clone)]
pub struct GaussianDepthData {
    /// Gaussian centre in world space.
    pub center: [f32; 3],
    /// Bounding-sphere radius in world units.
    pub radius: f32,
    /// Opacity in the range 0..1.
    pub opacity: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// DepthMode
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for combining multiple Gaussian depth contributions at one pixel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DepthMode {
    /// Z-buffer style: the nearest Gaussian centre depth wins.
    Nearest,
    /// Expected depth weighted by Gaussian opacity: Σ(α·d) / Σ(α).
    AlphaWeighted,
    /// Median of all depth samples at the pixel (up to 8 stored per pixel).
    Median,
    /// Depth of the highest-opacity Gaussian at the pixel.
    MaxOpacity,
}

// ─────────────────────────────────────────────────────────────────────────────
// DepthMapStats
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics computed over the valid (finite-depth) pixels of a [`DepthMap`].
#[derive(Debug, Clone)]
pub struct DepthMapStats {
    /// Minimum finite depth across all valid pixels.
    pub min_depth: f32,
    /// Maximum finite depth across all valid pixels.
    pub max_depth: f32,
    /// Mean depth across all valid pixels.
    pub mean_depth: f32,
    /// Standard deviation of depth across all valid pixels.
    pub std_depth: f32,
    /// Number of valid (non-INFINITY) pixels.
    pub num_valid: usize,
    /// Fraction of pixels that are valid (0..1).
    pub coverage: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// DepthMap
// ─────────────────────────────────────────────────────────────────────────────

/// A rasterised depth map produced by [`render_depth_map`].
///
/// `depths` is row-major: `depths[row * width + col]`.
/// Pixels with no Gaussian contribution store `f32::INFINITY`.
#[derive(Debug, Clone)]
pub struct DepthMap {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Depth value per pixel (row-major).  `f32::INFINITY` = no contribution.
    pub depths: Vec<f32>,
    /// Accumulated opacity per pixel (meaningful for [`DepthMode::AlphaWeighted`]).
    pub opacity_weights: Vec<f32>,
    /// Merging strategy used to produce this map.
    pub mode: DepthMode,
}

impl DepthMap {
    /// Allocate a fresh depth map — all pixels initialised to `f32::INFINITY`.
    pub fn new(width: u32, height: u32, mode: DepthMode) -> Self {
        let n = (width as usize) * (height as usize);
        Self {
            width,
            height,
            depths: vec![f32::INFINITY; n],
            opacity_weights: vec![0.0; n],
            mode,
        }
    }

    /// Return the depth at pixel `(px, py)`.
    ///
    /// Returns `f32::INFINITY` when `(px, py)` is out of bounds.
    pub fn pixel_depth(&self, px: u32, py: u32) -> f32 {
        if px >= self.width || py >= self.height {
            return f32::INFINITY;
        }
        self.depths[(py as usize) * (self.width as usize) + (px as usize)]
    }

    /// Return `Some(depth)` for a valid (finite) pixel, `None` for INFINITY.
    pub fn valid_depth(&self, px: u32, py: u32) -> Option<f32> {
        let d = self.pixel_depth(px, py);
        if d.is_finite() {
            Some(d)
        } else {
            None
        }
    }

    /// Normalise depth values to `[0, 1]` (min depth → 0, max depth → 1).
    ///
    /// `INFINITY` pixels (no contribution) are always mapped to `1.0`. When
    /// there are no valid (finite) pixels, or every valid pixel shares the
    /// same depth (a degenerate, zero-width range that cannot be divided
    /// into), every *finite* pixel is instead set to `0.0`. This includes
    /// the all-`INFINITY` case: with zero finite pixels, every output value
    /// is therefore `1.0` (not `0.0`).
    pub fn normalized(&self) -> Vec<f32> {
        let finite_iter = self.depths.iter().copied().filter(|d| d.is_finite());
        let min_d = finite_iter.clone().fold(f32::INFINITY, f32::min);
        let max_d = finite_iter.fold(f32::NEG_INFINITY, f32::max);

        let range = max_d - min_d;
        if range <= 0.0 || !range.is_finite() {
            // Either no valid pixels or all at the same depth.
            return self
                .depths
                .iter()
                .map(|d| if d.is_finite() { 0.0 } else { 1.0 })
                .collect();
        }

        self.depths
            .iter()
            .map(|d| {
                if d.is_finite() {
                    (d - min_d) / range
                } else {
                    1.0
                }
            })
            .collect()
    }

    /// Convert to an 8-bit grayscale image.
    ///
    /// Convention: **near = bright (255), far = dark (0)**, INFINITY pixels → 0.
    pub fn to_u8_image(&self) -> Vec<u8> {
        let norm = self.normalized();
        norm.iter()
            .zip(self.depths.iter())
            .map(|(n, d)| {
                if d.is_finite() {
                    // Invert so near is bright.
                    (255.0 * (1.0 - n)).clamp(0.0, 255.0) as u8
                } else {
                    0u8
                }
            })
            .collect()
    }

    /// Compute statistics over the finite-depth pixels.
    pub fn stats(&self) -> DepthMapStats {
        let valid: Vec<f32> = self
            .depths
            .iter()
            .copied()
            .filter(|d| d.is_finite())
            .collect();
        let total = (self.width as usize) * (self.height as usize);
        let num_valid = valid.len();

        if num_valid == 0 {
            return DepthMapStats {
                min_depth: 0.0,
                max_depth: 0.0,
                mean_depth: 0.0,
                std_depth: 0.0,
                num_valid: 0,
                coverage: 0.0,
            };
        }

        let min_d = valid.iter().copied().fold(f32::INFINITY, f32::min);
        let max_d = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean_d = valid.iter().copied().sum::<f32>() / num_valid as f32;
        let variance = valid
            .iter()
            .map(|d| {
                let diff = d - mean_d;
                diff * diff
            })
            .sum::<f32>()
            / num_valid as f32;
        let std_d = variance.sqrt();
        let coverage = num_valid as f32 / total as f32;

        DepthMapStats {
            min_depth: min_d,
            max_depth: max_d,
            mean_depth: mean_d,
            std_depth: std_d,
            num_valid,
            coverage,
        }
    }

    /// Count of valid (non-INFINITY) pixels.
    pub fn num_valid_pixels(&self) -> usize {
        self.depths.iter().filter(|d| d.is_finite()).count()
    }

    /// Fraction of pixels that have a valid depth (0.0 .. 1.0).
    pub fn coverage(&self) -> f32 {
        let total = (self.width as usize) * (self.height as usize);
        if total == 0 {
            return 0.0;
        }
        self.num_valid_pixels() as f32 / total as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rendering functions
// ─────────────────────────────────────────────────────────────────────────────

/// Render a depth map from a Gaussian scene using CPU rasterisation.
///
/// For each Gaussian, the centre is projected to pixel coordinates and its
/// depth is splatted to all pixels within a projected radius.  The per-pixel
/// depth is then resolved according to `mode`.
///
/// An empty `gaussians` slice is not an error — the returned map has all
/// pixels set to `f32::INFINITY`.
///
/// # Errors
///
/// - [`DepthMapError::InvalidDimensions`] when `camera.width == 0` or
///   `camera.height == 0`.
pub fn render_depth_map(
    camera: &DepthCamera,
    gaussians: &[GaussianDepthData],
    mode: DepthMode,
) -> Result<DepthMap, DepthMapError> {
    if camera.width == 0 || camera.height == 0 {
        return Err(DepthMapError::InvalidDimensions(format!(
            "camera dimensions must be non-zero, got {}×{}",
            camera.width, camera.height
        )));
    }

    let w = camera.width as usize;
    let h = camera.height as usize;
    let n_pixels = w * h;

    match mode {
        DepthMode::Nearest => render_nearest(camera, gaussians, w, h, n_pixels),
        DepthMode::AlphaWeighted => render_alpha_weighted(camera, gaussians, w, h, n_pixels),
        DepthMode::MaxOpacity => render_max_opacity(camera, gaussians, w, h, n_pixels),
        DepthMode::Median => render_median(camera, gaussians, w, h, n_pixels),
    }
}

// ── Shared helpers ───────────────────────────────────────────────────────────

/// Project a world-space bounding-sphere radius to separate x/y pixel-space
/// half-extents for rasterisation, using `camera.fx` and `camera.fy`
/// independently.
///
/// They differ for any non-square image (`DepthCamera::from_fov` sets
/// `fx = fy * aspect`), so a single shared radius would over-splat one axis
/// and under-splat the other. Both extents are floored at `1.0` pixel so a
/// Gaussian always splats to at least its own pixel.
#[inline]
fn projected_radius_px(camera: &DepthCamera, radius: f32, depth: f32) -> (f32, f32) {
    (
        (camera.fx * radius / depth).max(1.0),
        (camera.fy * radius / depth).max(1.0),
    )
}

// ── Nearest ──────────────────────────────────────────────────────────────────

fn render_nearest(
    camera: &DepthCamera,
    gaussians: &[GaussianDepthData],
    w: usize,
    h: usize,
    n_pixels: usize,
) -> Result<DepthMap, DepthMapError> {
    let mut depths = vec![f32::INFINITY; n_pixels];
    let opacity_weights = vec![0.0f32; n_pixels];

    for gaussian in gaussians {
        if !camera.in_frustum_approx(gaussian.center, gaussian.radius) {
            continue;
        }
        let (px_f, py_f, depth) = match camera.project(gaussian.center) {
            Some(v) => v,
            None => continue,
        };

        let (proj_rx, proj_ry) = projected_radius_px(camera, gaussian.radius, depth);
        splat_nearest(px_f, py_f, depth, proj_rx, proj_ry, (w, h), &mut depths);
    }

    let mut map = DepthMap::new(camera.width, camera.height, DepthMode::Nearest);
    map.depths = depths;
    map.opacity_weights = opacity_weights;
    Ok(map)
}

#[inline]
fn splat_nearest(
    px_f: f32,
    py_f: f32,
    depth: f32,
    radius_x: f32,
    radius_y: f32,
    (w, h): (usize, usize),
    depths: &mut [f32],
) {
    let x_min = (px_f - radius_x).floor() as i32;
    let x_max = (px_f + radius_x).floor() as i32;
    let y_min = (py_f - radius_y).floor() as i32;
    let y_max = (py_f + radius_y).floor() as i32;

    for iy in y_min..=y_max {
        if iy < 0 || iy >= h as i32 {
            continue;
        }
        for ix in x_min..=x_max {
            if ix < 0 || ix >= w as i32 {
                continue;
            }
            let idx = iy as usize * w + ix as usize;
            if depth < depths[idx] {
                depths[idx] = depth;
            }
        }
    }
}

// ── AlphaWeighted ────────────────────────────────────────────────────────────

fn render_alpha_weighted(
    camera: &DepthCamera,
    gaussians: &[GaussianDepthData],
    w: usize,
    h: usize,
    n_pixels: usize,
) -> Result<DepthMap, DepthMapError> {
    // Accumulate sum(alpha * depth) and sum(alpha).
    let mut sum_alpha_depth = vec![0.0f32; n_pixels];
    let mut sum_alpha = vec![0.0f32; n_pixels];

    for gaussian in gaussians {
        if !camera.in_frustum_approx(gaussian.center, gaussian.radius) {
            continue;
        }
        let (px_f, py_f, depth) = match camera.project(gaussian.center) {
            Some(v) => v,
            None => continue,
        };

        let (proj_rx, proj_ry) = projected_radius_px(camera, gaussian.radius, depth);
        let alpha = gaussian.opacity;

        let x_min = (px_f - proj_rx).floor() as i32;
        let x_max = (px_f + proj_rx).floor() as i32;
        let y_min = (py_f - proj_ry).floor() as i32;
        let y_max = (py_f + proj_ry).floor() as i32;

        for iy in y_min..=y_max {
            if iy < 0 || iy >= h as i32 {
                continue;
            }
            for ix in x_min..=x_max {
                if ix < 0 || ix >= w as i32 {
                    continue;
                }
                let idx = iy as usize * w + ix as usize;
                sum_alpha_depth[idx] += alpha * depth;
                sum_alpha[idx] += alpha;
            }
        }
    }

    // Resolve: depths[i] = sum_alpha_depth[i] / sum_alpha[i]
    let mut depths = vec![f32::INFINITY; n_pixels];
    for i in 0..n_pixels {
        if sum_alpha[i] > 0.0 {
            depths[i] = sum_alpha_depth[i] / sum_alpha[i];
        }
    }

    let mut map = DepthMap::new(camera.width, camera.height, DepthMode::AlphaWeighted);
    map.depths = depths;
    map.opacity_weights = sum_alpha;
    Ok(map)
}

// ── MaxOpacity ───────────────────────────────────────────────────────────────

fn render_max_opacity(
    camera: &DepthCamera,
    gaussians: &[GaussianDepthData],
    w: usize,
    h: usize,
    n_pixels: usize,
) -> Result<DepthMap, DepthMapError> {
    let mut depths = vec![f32::INFINITY; n_pixels];
    let mut max_opacities = vec![f32::NEG_INFINITY; n_pixels];

    for gaussian in gaussians {
        if !camera.in_frustum_approx(gaussian.center, gaussian.radius) {
            continue;
        }
        let (px_f, py_f, depth) = match camera.project(gaussian.center) {
            Some(v) => v,
            None => continue,
        };

        let (proj_rx, proj_ry) = projected_radius_px(camera, gaussian.radius, depth);
        let alpha = gaussian.opacity;

        let x_min = (px_f - proj_rx).floor() as i32;
        let x_max = (px_f + proj_rx).floor() as i32;
        let y_min = (py_f - proj_ry).floor() as i32;
        let y_max = (py_f + proj_ry).floor() as i32;

        for iy in y_min..=y_max {
            if iy < 0 || iy >= h as i32 {
                continue;
            }
            for ix in x_min..=x_max {
                if ix < 0 || ix >= w as i32 {
                    continue;
                }
                let idx = iy as usize * w + ix as usize;
                if alpha > max_opacities[idx] {
                    max_opacities[idx] = alpha;
                    depths[idx] = depth;
                }
            }
        }
    }

    let mut map = DepthMap::new(camera.width, camera.height, DepthMode::MaxOpacity);
    map.depths = depths;
    Ok(map)
}

// ── Median ───────────────────────────────────────────────────────────────────

/// Maximum number of depth samples stored per pixel in Median mode.
const MEDIAN_MAX_SAMPLES: usize = 8;

fn render_median(
    camera: &DepthCamera,
    gaussians: &[GaussianDepthData],
    w: usize,
    h: usize,
    n_pixels: usize,
) -> Result<DepthMap, DepthMapError> {
    // Flat, fixed-size per-pixel sample storage. MEDIAN_MAX_SAMPLES is a
    // compile-time constant, so a `Vec<Vec<f32>>` (one 24-byte Vec header
    // per pixel whether touched or not, plus an individual heap allocation
    // — and reallocations as it grows — the first time each pixel receives
    // a sample) is unnecessary heap traffic. `samples` holds all
    // `n_pixels * MEDIAN_MAX_SAMPLES` slots in a single allocation;
    // `counts` tracks how many of each pixel's slots are filled.
    let mut samples = vec![0.0f32; n_pixels * MEDIAN_MAX_SAMPLES];
    let mut counts = vec![0u8; n_pixels];

    for gaussian in gaussians {
        if !camera.in_frustum_approx(gaussian.center, gaussian.radius) {
            continue;
        }
        let (px_f, py_f, depth) = match camera.project(gaussian.center) {
            Some(v) => v,
            None => continue,
        };

        let (proj_rx, proj_ry) = projected_radius_px(camera, gaussian.radius, depth);

        let x_min = (px_f - proj_rx).floor() as i32;
        let x_max = (px_f + proj_rx).floor() as i32;
        let y_min = (py_f - proj_ry).floor() as i32;
        let y_max = (py_f + proj_ry).floor() as i32;

        for iy in y_min..=y_max {
            if iy < 0 || iy >= h as i32 {
                continue;
            }
            for ix in x_min..=x_max {
                if ix < 0 || ix >= w as i32 {
                    continue;
                }
                let idx = iy as usize * w + ix as usize;
                let cnt = counts[idx] as usize;
                if cnt < MEDIAN_MAX_SAMPLES {
                    samples[idx * MEDIAN_MAX_SAMPLES + cnt] = depth;
                    counts[idx] = (cnt + 1) as u8;
                }
            }
        }
    }

    // Resolve each pixel to the median of its (in-place sorted) sample prefix.
    let mut depths = vec![f32::INFINITY; n_pixels];
    for i in 0..n_pixels {
        let cnt = counts[i] as usize;
        if cnt == 0 {
            continue;
        }
        let base = i * MEDIAN_MAX_SAMPLES;
        let slot = &mut samples[base..base + cnt];
        slot.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        depths[i] = slot[cnt / 2];
    }

    let mut map = DepthMap::new(camera.width, camera.height, DepthMode::Median);
    map.depths = depths;
    Ok(map)
}

// ─────────────────────────────────────────────────────────────────────────────
// Batch rendering
// ─────────────────────────────────────────────────────────────────────────────

/// Render a depth map for each camera in `cameras`.
///
/// Returns one `Result<DepthMap, DepthMapError>` per camera.
pub fn render_depth_maps(
    cameras: &[DepthCamera],
    gaussians: &[GaussianDepthData],
    mode: DepthMode,
) -> Vec<Result<DepthMap, DepthMapError>> {
    cameras
        .iter()
        .map(|cam| render_depth_map(cam, gaussians, mode))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Disparity conversion
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a depth map to a disparity map: `disparity = focal_length / depth`.
///
/// Pixels with `INFINITY` depth (no valid sample) are mapped to `0`.
pub fn depth_to_disparity(depth_map: &DepthMap, focal_length: f32) -> Vec<f32> {
    depth_map
        .depths
        .iter()
        .map(|&d| {
            if d.is_finite() && d > 0.0 {
                focal_length / d
            } else {
                0.0
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Point cloud back-projection
// ─────────────────────────────────────────────────────────────────────────────

/// Unproject valid depth-map pixels to world-space 3-D points.
///
/// The inverse of the pinhole projection used in [`DepthCamera::project`]:
///
/// ```text
/// cam_x = (ix - cx) * depth / fx
/// cam_y = -(iy - cy) * depth / fy   (flip Y back)
/// cam_z = -depth                     (camera looks down -Z)
/// world = R^T * cam + position
/// ```
pub fn depth_map_to_pointcloud(depth_map: &DepthMap, camera: &DepthCamera) -> Vec<[f32; 3]> {
    let w = depth_map.width as usize;
    let h = depth_map.height as usize;
    let r = &camera.view_matrix_rotation;

    // Transpose of the 3×3 row-major rotation (camera→world).
    let rt = [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]];

    let mut points = Vec::new();
    for iy in 0..h {
        for ix in 0..w {
            let idx = iy * w + ix;
            let depth = depth_map.depths[idx];
            if !depth.is_finite() {
                continue;
            }

            // Back-project to camera space.
            let cam_x = (ix as f32 - camera.cx) * depth / camera.fx;
            let cam_y = -(iy as f32 - camera.cy) * depth / camera.fy;
            let cam_z = -depth;

            // Rotate to world space using R^T.
            let wx = rt[0] * cam_x + rt[1] * cam_y + rt[2] * cam_z + camera.position[0];
            let wy = rt[3] * cam_x + rt[4] * cam_y + rt[5] * cam_z + camera.position[1];
            let wz = rt[6] * cam_x + rt[7] * cam_y + rt[8] * cam_z + camera.position[2];

            points.push([wx, wy, wz]);
        }
    }

    points
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ── Helper: identity-rotation camera at z = 5 looking toward origin ───────

    fn make_camera() -> DepthCamera {
        DepthCamera::from_fov(
            [0.0, 0.0, 5.0], // camera 5 units along +Z
            [
                1.0, 0.0, 0.0, // identity rotation (world→camera)
                0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            ],
            64,
            64,
            std::f32::consts::FRAC_PI_2, // 90° FoV
            0.1,
            100.0,
        )
        .unwrap()
    }

    fn make_gaussian(z: f32, opacity: f32) -> GaussianDepthData {
        GaussianDepthData {
            center: [0.0, 0.0, z],
            radius: 0.5,
            opacity,
        }
    }

    // ── Test 1: from_fov valid → Ok ───────────────────────────────────────────
    #[test]
    fn test_from_fov_valid() {
        let cam = make_camera();
        // fy = 64 / (2 * tan(π/4)) = 64 / 2 = 32
        assert!((cam.fy - 32.0).abs() < 1e-3, "fy = {}", cam.fy);
        assert!((cam.fx - 32.0).abs() < 1e-3, "fx = {}", cam.fx);
        assert!((cam.cx - 32.0).abs() < 1e-3, "cx = {}", cam.cx);
        assert!((cam.cy - 32.0).abs() < 1e-3, "cy = {}", cam.cy);
    }

    // ── Test 2: from_fov fov = 0 → Err ───────────────────────────────────────
    #[test]
    fn test_from_fov_zero_fov_err() {
        let result = DepthCamera::from_fov(
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            64,
            64,
            0.0, // invalid
            0.1,
            100.0,
        );
        assert!(matches!(result, Err(DepthMapError::InvalidCamera(_))));
    }

    // ── Test 3: project point in front → Some ─────────────────────────────────
    #[test]
    fn test_project_in_front() {
        let cam = make_camera();
        // Camera at z=5 looks down -Z in camera space.
        // World point at z=0 → diff = [0,0,-5], cam_z = -5, depth = 5.
        let result = cam.project([0.0, 0.0, 0.0]);
        assert!(result.is_some(), "point in front should project");
        let (px, py, depth) = result.unwrap();
        // Should project to image centre.
        assert!((px - 32.0).abs() < 1e-3, "px = {px}");
        assert!((py - 32.0).abs() < 1e-3, "py = {py}");
        assert!((depth - 5.0).abs() < 1e-3, "depth = {depth}");
    }

    // ── Test 4: project point behind camera → None ────────────────────────────
    #[test]
    fn test_project_behind_camera() {
        let cam = make_camera();
        // z = 10 is behind the camera (camera at z=5, looks down -Z → behind = +Z)
        let result = cam.project([0.0, 0.0, 10.0]);
        assert!(result.is_none(), "point behind camera should return None");
    }

    // ── Test 5: projected depth is positive ───────────────────────────────────
    #[test]
    fn test_project_depth_positive() {
        let cam = make_camera();
        let (_, _, depth) = cam.project([0.0, 0.0, 0.0]).unwrap();
        assert!(depth > 0.0, "depth must be positive, got {depth}");
    }

    // ── Test 6: empty gaussians → Ok with all-INFINITY map ───────────────────
    #[test]
    fn test_render_empty_gaussians_ok() {
        let cam = make_camera();
        let result = render_depth_map(&cam, &[], DepthMode::Nearest);
        assert!(result.is_ok());
        let map = result.unwrap();
        assert_eq!(map.num_valid_pixels(), 0);
    }

    // ── Test 7: single Gaussian at scene centre → valid depth at centre ───────
    #[test]
    fn test_render_single_gaussian_centre_valid() {
        let cam = make_camera();
        let gaussians = vec![make_gaussian(0.0, 0.9)];
        let map = render_depth_map(&cam, &gaussians, DepthMode::Nearest).unwrap();
        // Centre pixel should have a finite depth.
        let d = map.valid_depth(32, 32);
        assert!(d.is_some(), "centre pixel should have valid depth");
    }

    // ── Test 8: Nearest — closer Gaussian wins ────────────────────────────────
    #[test]
    fn test_render_nearest_closer_wins() {
        let cam = make_camera();
        // Near Gaussian (depth ≈ 3) and far Gaussian (depth ≈ 5).
        // Camera at z=5; near at z=2 → depth=3, far at z=0 → depth=5.
        let gaussians = vec![
            make_gaussian(0.0, 0.5), // far, depth ≈ 5
            make_gaussian(2.0, 0.5), // near, depth ≈ 3
        ];
        let map = render_depth_map(&cam, &gaussians, DepthMode::Nearest).unwrap();
        let d = map.pixel_depth(32, 32);
        assert!(d.is_finite(), "centre pixel should have valid depth");
        // Should be close to 3, not 5.
        assert!(d < 4.0, "nearest should win, got depth={d}");
    }

    // ── Test 9: AlphaWeighted — depth is weighted by opacity ──────────────────
    #[test]
    fn test_render_alpha_weighted() {
        let cam = make_camera();
        // Two Gaussians at same pixel: depth=2 with opacity=0.9, depth=8 with opacity=0.1
        // Expected weighted depth ≈ (0.9*2 + 0.1*8) / (0.9+0.1) = (1.8+0.8)/1.0 = 2.6
        let gaussians = vec![
            GaussianDepthData {
                center: [0.0, 0.0, 3.0],
                radius: 0.5,
                opacity: 0.9,
            }, // depth 2
            GaussianDepthData {
                center: [0.0, 0.0, -3.0],
                radius: 0.5,
                opacity: 0.1,
            }, // depth 8
        ];
        let map = render_depth_map(&cam, &gaussians, DepthMode::AlphaWeighted).unwrap();
        let d = map.pixel_depth(32, 32);
        assert!(d.is_finite(), "should have valid depth");
        // High-weight depth (2) should dominate: result < 5.0
        assert!(
            d < 5.0,
            "alpha-weighted depth should be closer to high-opacity value, got {d}"
        );
    }

    // ── Test 10: MaxOpacity — most opaque Gaussian's depth returned ───────────
    #[test]
    fn test_render_max_opacity() {
        let cam = make_camera();
        // Low-opacity near Gaussian and high-opacity far Gaussian.
        let gaussians = vec![
            GaussianDepthData {
                center: [0.0, 0.0, 4.0],
                radius: 0.5,
                opacity: 0.1,
            }, // depth 1, low opacity
            GaussianDepthData {
                center: [0.0, 0.0, 0.0],
                radius: 0.5,
                opacity: 0.9,
            }, // depth 5, high opacity
        ];
        let map = render_depth_map(&cam, &gaussians, DepthMode::MaxOpacity).unwrap();
        let d = map.pixel_depth(32, 32);
        assert!(d.is_finite());
        // MaxOpacity should pick the depth=5 (high opacity) Gaussian.
        assert!(
            d > 4.0 && d < 6.0,
            "max-opacity depth should be ≈5, got {d}"
        );
    }

    // ── Test 11: num_valid_pixels: 0 for fresh map ────────────────────────────
    #[test]
    fn test_num_valid_pixels_fresh() {
        let map = DepthMap::new(8, 8, DepthMode::Nearest);
        assert_eq!(map.num_valid_pixels(), 0);
    }

    // ── Test 12: coverage: 0.0 for fresh map ─────────────────────────────────
    #[test]
    fn test_coverage_fresh() {
        let map = DepthMap::new(8, 8, DepthMode::Nearest);
        assert!((map.coverage() - 0.0).abs() < 1e-6);
    }

    // ── Test 13: normalized: all finite values in [0, 1] ─────────────────────
    #[test]
    fn test_normalized_range() {
        let cam = make_camera();
        let gaussians = vec![
            make_gaussian(0.0, 0.9), // depth ≈ 5
            make_gaussian(3.0, 0.9), // depth ≈ 2
        ];
        let map = render_depth_map(&cam, &gaussians, DepthMode::Nearest).unwrap();
        let norm = map.normalized();
        for (i, &v) in norm.iter().enumerate() {
            let d = map.depths[i];
            if d.is_finite() {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "normalized value {v} out of [0,1]"
                );
            }
        }
    }

    // ── Test 14: to_u8_image — near pixel brighter than far pixel ────────────
    #[test]
    fn test_u8_image_near_brighter_than_far() {
        // Construct a depth map with two known depths directly so normalization
        // has a non-zero range: depth=1 (near) and depth=9 (far).
        let mut map = DepthMap::new(2, 1, DepthMode::Nearest);
        map.depths[0] = 1.0; // near pixel
        map.depths[1] = 9.0; // far pixel
        let img = map.to_u8_image();
        // Near pixel should be brighter (higher u8 value).
        assert!(
            img[0] > img[1],
            "near pixel ({}) should be brighter than far pixel ({})",
            img[0],
            img[1]
        );
    }

    // ── Test 15: stats — min ≤ mean ≤ max ────────────────────────────────────
    #[test]
    fn test_stats_ordering() {
        let cam = make_camera();
        let gaussians = vec![
            make_gaussian(0.0, 0.9),
            make_gaussian(2.0, 0.9),
            make_gaussian(4.0, 0.9),
        ];
        let map = render_depth_map(&cam, &gaussians, DepthMode::Nearest).unwrap();
        let stats = map.stats();
        assert!(
            stats.min_depth <= stats.mean_depth && stats.mean_depth <= stats.max_depth,
            "min={} mean={} max={}",
            stats.min_depth,
            stats.mean_depth,
            stats.max_depth
        );
    }

    // ── Test 16: depth_to_disparity — larger depth → smaller disparity ────────
    #[test]
    fn test_disparity_inverse_depth() {
        let mut map = DepthMap::new(1, 2, DepthMode::Nearest);
        map.depths[0] = 2.0; // near
        map.depths[1] = 8.0; // far
        let disp = depth_to_disparity(&map, 32.0);
        assert!(
            disp[0] > disp[1],
            "near should have larger disparity: {} vs {}",
            disp[0],
            disp[1]
        );
    }

    // ── Test 17: depth_to_disparity — INFINITY depth → 0 ─────────────────────
    #[test]
    fn test_disparity_infinity_to_zero() {
        let map = DepthMap::new(1, 1, DepthMode::Nearest); // depths[0] = INFINITY
        let disp = depth_to_disparity(&map, 32.0);
        assert!(
            (disp[0]).abs() < 1e-6,
            "INFINITY depth should give 0 disparity, got {}",
            disp[0]
        );
    }

    // ── Test 18: depth_map_to_pointcloud — empty map → empty point cloud ──────
    #[test]
    fn test_pointcloud_empty_map() {
        let cam = make_camera();
        let map = DepthMap::new(8, 8, DepthMode::Nearest);
        let pts = depth_map_to_pointcloud(&map, &cam);
        assert!(
            pts.is_empty(),
            "empty depth map should give empty point cloud"
        );
    }

    // ── Test 19: depth_map_to_pointcloud — valid pixels near Gaussian ─────────
    #[test]
    fn test_pointcloud_near_gaussian() {
        let cam = make_camera();
        // Gaussian at world origin (0,0,0); camera at (0,0,5).
        let gaussians = vec![GaussianDepthData {
            center: [0.0, 0.0, 0.0],
            radius: 0.5,
            opacity: 0.9,
        }];
        let map = render_depth_map(&cam, &gaussians, DepthMode::Nearest).unwrap();
        let pts = depth_map_to_pointcloud(&map, &cam);
        assert!(!pts.is_empty(), "should have at least one point");
        // The centre pixel should unproject to approximately (0, 0, 0).
        // Find the point whose z-component is closest to 0.
        let best = pts.iter().min_by(|a, b| {
            let da = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
            let db = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        let p = best.unwrap();
        let dist = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        assert!(
            dist < 1.0,
            "closest point should be near origin, got dist={dist:.3}"
        );
    }

    // ── Test 20: render_depth_maps — one result per camera ───────────────────
    #[test]
    fn test_render_depth_maps_count() {
        let cam1 = make_camera();
        let cam2 = DepthCamera::from_fov(
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            32,
            32,
            std::f32::consts::FRAC_PI_4,
            0.1,
            50.0,
        )
        .unwrap();
        let cameras = vec![cam1, cam2];
        let gaussians = vec![make_gaussian(0.0, 0.9)];
        let results = render_depth_maps(&cameras, &gaussians, DepthMode::Nearest);
        assert_eq!(results.len(), 2, "should have one result per camera");
    }

    // ── Test 21: valid_depth → Some for finite, None for INFINITY ────────────
    #[test]
    fn test_valid_depth() {
        let mut map = DepthMap::new(2, 1, DepthMode::Nearest);
        map.depths[0] = 3.5;
        // depths[1] remains INFINITY
        assert_eq!(map.valid_depth(0, 0), Some(3.5));
        assert_eq!(map.valid_depth(1, 0), None);
    }

    // ── Test 22: pixel_depth out-of-bounds → INFINITY ────────────────────────
    #[test]
    fn test_pixel_depth_out_of_bounds() {
        let map = DepthMap::new(4, 4, DepthMode::Nearest);
        assert!(map.pixel_depth(100, 100).is_infinite());
        assert!(map.pixel_depth(4, 0).is_infinite());
        assert!(map.pixel_depth(0, 4).is_infinite());
    }

    // ── Test 23: in_frustum_approx must not produce false negatives ──────────
    #[test]
    fn test_in_frustum_approx_large_sphere_near_edge_not_culled() {
        // Regression test for the conservativeness bug: `in_frustum_approx`
        // must never return `false` (cull) for a sphere that genuinely still
        // overlaps the frustum. A sphere whose radius is close to its
        // distance from the camera has a true projected screen radius of
        // `f * r / sqrt(d^2 - r^2)`, far larger than the naive `f * r / d` -
        // using the naive formula wrongly culled spheres like this one, which
        // are still visible near the image edge.
        let cam = DepthCamera::from_fov(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], // identity rotation
            64,
            64,
            std::f32::consts::FRAC_PI_2, // fx = fy = 32 for a square 64×64 image
            0.01,
            1000.0,
        )
        .unwrap();

        // depth = 1.1, radius = 1.0 (camera nearly grazes the sphere
        // surface): naive proj radius ≈ 29.1px, true proj radius ≈ 69.8px.
        let radius = 1.0_f32;
        let depth = 1.1_f32;
        // Solve for a world x that projects to px = 100 (i.e. 36px past the
        // right edge of the 64px-wide image): cam_x = (px - cx) * depth / fx.
        let cam_x = (100.0 - cam.cx) * depth / cam.fx;
        let center = [cam_x, 0.0, -depth];

        assert!(
            cam.in_frustum_approx(center, radius),
            "a sphere whose true projected extent still overlaps the image \
             must not be culled by the conservative frustum check"
        );
    }

    // ── Test 24: normalized() on an all-INFINITY map ──────────────────────────
    #[test]
    fn test_normalized_all_infinity_maps_to_one() {
        // Regression test: an all-INFINITY map (no Gaussian contributed to
        // any pixel) must normalize every value to 1.0, matching the general
        // "INFINITY → 1.0" contract - not 0.0.
        let map = DepthMap::new(4, 4, DepthMode::Nearest);
        let norm = map.normalized();
        assert!(
            norm.iter().all(|&v| (v - 1.0).abs() < 1e-6),
            "all-INFINITY map should normalize to all 1.0, got {norm:?}"
        );
    }

    // ── Test 25: splat radius tracks fx/fy independently ──────────────────────
    #[test]
    fn test_render_splat_extent_respects_aspect_ratio() {
        // Regression test: the projected splat radius must use `fx` for the
        // x-extent and `fy` for the y-extent independently - using `fx` for
        // both (as the pre-fix code did) makes every splat's vertical extent
        // as wide as its horizontal one regardless of aspect ratio.
        let cam = DepthCamera::from_fov(
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            128,
            32,
            std::f32::consts::FRAC_PI_2, // fy = 16, fx = fy * (128/32) = 64
            0.1,
            100.0,
        )
        .unwrap();
        assert!((cam.fx - 64.0).abs() < 1e-3, "fx = {}", cam.fx);
        assert!((cam.fy - 16.0).abs() < 1e-3, "fy = {}", cam.fy);

        // Gaussian at world z = 1.0 → depth = 4.0, radius = 1.0:
        // proj_rx = fx*r/depth = 16px, proj_ry = fy*r/depth = 4px.
        let gaussians = vec![GaussianDepthData {
            center: [0.0, 0.0, 1.0],
            radius: 1.0,
            opacity: 0.9,
        }];
        let map = render_depth_map(&cam, &gaussians, DepthMode::Nearest).unwrap();

        let touched_rows: Vec<usize> = (0..32usize)
            .filter(|&row| {
                (0..128usize).any(|col| map.pixel_depth(col as u32, row as u32).is_finite())
            })
            .collect();
        let min_row = *touched_rows.first().expect("some row should be touched");
        let max_row = *touched_rows.last().expect("some row should be touched");
        let row_span = max_row - min_row + 1;

        // Correct behaviour splats roughly rows [12, 20] (9 rows, ≈ 2*proj_ry).
        // The pre-fix bug (using fx=64 for the y-extent too) would splat
        // rows [0, 31] - the full 32-row image height.
        assert!(
            row_span <= 14,
            "vertical splat extent should track fy (~9 rows), not fx (~32 \
             rows); got {row_span} rows spanning {min_row}..={max_row}"
        );
    }

    // ── Test 26: Median mode resolves the true median depth ──────────────────
    #[test]
    fn test_render_median_resolves_median_depth() {
        let cam = make_camera();
        // Three Gaussians whose footprints all cover the centre pixel, with
        // depths 2, 5, 8 - the centre pixel's resolved depth must be the
        // median (5) regardless of Gaussian insertion order.
        let gaussians = vec![
            make_gaussian(3.0, 0.9),  // depth = 2
            make_gaussian(0.0, 0.9),  // depth = 5
            make_gaussian(-3.0, 0.9), // depth = 8
        ];
        let map = render_depth_map(&cam, &gaussians, DepthMode::Median).unwrap();
        let d = map.pixel_depth(32, 32);
        assert!(
            (d - 5.0).abs() < 1e-3,
            "median of {{2, 5, 8}} should be 5, got {d}"
        );
    }

    // ── Test 27: Median mode caps sample count without panicking ─────────────
    #[test]
    fn test_render_median_caps_at_max_samples() {
        // More than MEDIAN_MAX_SAMPLES (8) Gaussians covering the same pixel
        // must not panic or corrupt the per-pixel sample count - the flat
        // sample buffer simply stops accepting new samples once full.
        let cam = make_camera();
        let gaussians: Vec<GaussianDepthData> = (0..12)
            .map(|i| make_gaussian(i as f32 * 0.1, 0.9))
            .collect();
        let map = render_depth_map(&cam, &gaussians, DepthMode::Median).unwrap();
        let d = map.pixel_depth(32, 32);
        assert!(d.is_finite(), "centre pixel should still resolve a depth");
    }
}
