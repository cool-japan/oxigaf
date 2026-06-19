//! CPU-side silhouette mask generation from a 3D Gaussian Splatting scene.
//!
//! Silhouettes represent the binary or soft outline/mask of an object and are
//! useful for edge supervision during training, object segmentation, background
//! removal, and shape reconstruction validation.
//!
//! # Overview
//!
//! - [`SilhouetteCamera`]: Pinhole camera with projection (no image-bounds clip).
//! - [`GaussianSilData`]: Bounding-sphere + opacity per Gaussian.
//! - [`SilhouetteMode`]: How Gaussian contributions are merged per pixel.
//! - [`SilhouetteMask`]: The output raster — per-pixel values in `[0, 1]`.
//! - [`render_silhouette`]: Main entry point for a single camera.
//! - [`render_silhouettes`]: Convenience batch over many cameras.
//! - [`render_silhouette_antialiased`]: Gaussian-kernel (soft) silhouette.
//! - [`silhouette_bce_loss`]: Binary cross-entropy between mask and target.
//! - [`silhouette_iou`]: Intersection-over-Union between two binary masks.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by silhouette operations.
#[derive(Debug, Error)]
pub enum SilhouetteError {
    /// Invalid camera parameters.
    #[error("Invalid camera: {0}")]
    InvalidCamera(String),

    /// Image dimensions are zero or mismatched.
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    /// The Gaussian scene is empty — nothing to render.
    #[error("Empty scene: no Gaussians provided")]
    EmptyScene,
}

// ─────────────────────────────────────────────────────────────────────────────
// SilhouetteCamera
// ─────────────────────────────────────────────────────────────────────────────

/// Pinhole camera used to project 3-D Gaussian centres for silhouette rendering.
///
/// The camera looks down **-Z** in camera space.  `view_rotation` is a
/// row-major 3×3 matrix that maps **world → camera** directions:
///
/// ```text
/// cam = R * (world - position)
/// ```
///
/// Unlike a depth-map camera, [`SilhouetteCamera::project`] does **not** check
/// image bounds — Gaussians whose centre is off-screen may still contribute
/// pixels via their projected radius, so bounds filtering is deferred to the
/// per-pixel loop inside [`render_silhouette`].
#[derive(Debug, Clone)]
pub struct SilhouetteCamera {
    /// Camera position in world space.
    pub position: [f32; 3],
    /// World→camera rotation (row-major 3×3).  Camera looks down -Z.
    pub view_rotation: [f32; 9],
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

impl SilhouetteCamera {
    /// Construct a [`SilhouetteCamera`] from a vertical field-of-view.
    ///
    /// ```text
    /// fy = height / (2 * tan(fov_y / 2))
    /// fx = fy * (width / height)
    /// cx = width / 2,  cy = height / 2
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`SilhouetteError::InvalidCamera`] when `fov_y_rad <= 0`,
    /// `width == 0`, `height == 0`, `near >= far`, or `near <= 0`.
    pub fn from_fov(
        position: [f32; 3],
        view_rotation: [f32; 9],
        width: u32,
        height: u32,
        fov_y_rad: f32,
        near: f32,
        far: f32,
    ) -> Result<Self, SilhouetteError> {
        if fov_y_rad <= 0.0 || !fov_y_rad.is_finite() {
            return Err(SilhouetteError::InvalidCamera(format!(
                "fov_y_rad must be positive and finite, got {fov_y_rad}"
            )));
        }
        if width == 0 {
            return Err(SilhouetteError::InvalidCamera(
                "width must be non-zero".to_string(),
            ));
        }
        if height == 0 {
            return Err(SilhouetteError::InvalidCamera(
                "height must be non-zero".to_string(),
            ));
        }
        if near <= 0.0 {
            return Err(SilhouetteError::InvalidCamera(format!(
                "near must be positive, got {near}"
            )));
        }
        if far <= near {
            return Err(SilhouetteError::InvalidCamera(format!(
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
            view_rotation,
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
    /// - Returns `None` only when the point is **behind** the camera
    ///   (`depth <= 0`).  Points outside the image rectangle are **not**
    ///   rejected here; bounds checking is done per-pixel in the splatting loop.
    pub fn project(&self, world_point: [f32; 3]) -> Option<(f32, f32, f32)> {
        let diff = [
            world_point[0] - self.position[0],
            world_point[1] - self.position[1],
            world_point[2] - self.position[2],
        ];

        // Apply row-major rotation: cam = R * diff
        let r = &self.view_rotation;
        let cam_x = r[0] * diff[0] + r[1] * diff[1] + r[2] * diff[2];
        let cam_y = r[3] * diff[0] + r[4] * diff[1] + r[5] * diff[2];
        let cam_z = r[6] * diff[0] + r[7] * diff[1] + r[8] * diff[2];

        // Camera looks down -Z; positive depth is -cam_z.
        let depth = -cam_z;
        if depth <= 0.0 {
            return None;
        }

        // Pinhole projection (Y flipped for image coordinates, matching depth_map convention).
        let px = self.fx * cam_x / depth + self.cx;
        let py = -self.fy * cam_y / depth + self.cy;

        Some((px, py, depth))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GaussianSilData
// ─────────────────────────────────────────────────────────────────────────────

/// Per-Gaussian data required for silhouette rendering.
#[derive(Debug, Clone)]
pub struct GaussianSilData {
    /// Gaussian centre in world space.
    pub center: [f32; 3],
    /// Bounding-sphere radius in world units.
    pub radius: f32,
    /// Opacity in the range 0..1.  Used for soft silhouette modes.
    pub opacity: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// SilhouetteMode
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for accumulating Gaussian contributions per pixel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SilhouetteMode {
    /// Binary: pixel is 1.0 if any Gaussian projects there, 0.0 otherwise.
    Binary,
    /// Soft: accumulated opacity per pixel (sum of alpha contributions, clamped to `[0, 1]`).
    SoftOpacity,
    /// Depth-weighted: accumulate `opacity / depth` per pixel, clamped to `[0, 1]`.
    DepthWeighted,
    /// Maximum opacity: the maximum opacity of any Gaussian projecting to the pixel.
    MaxOpacity,
}

// ─────────────────────────────────────────────────────────────────────────────
// SilhouetteMask
// ─────────────────────────────────────────────────────────────────────────────

/// Output silhouette raster.
///
/// Per-pixel values are in `[0, 1]`.  Pixels are stored row-major:
/// index `y * width + x`.
#[derive(Debug, Clone)]
pub struct SilhouetteMask {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Per-pixel values in `[0, 1]`.  Row-major: index `y * width + x`.
    pub mask: Vec<f32>,
    /// The mode used to produce this mask.
    pub mode: SilhouetteMode,
}

impl SilhouetteMask {
    /// Create a new all-zero mask.
    pub fn new(width: u32, height: u32, mode: SilhouetteMode) -> Self {
        let n = (width as usize) * (height as usize);
        Self {
            width,
            height,
            mask: vec![0.0f32; n],
            mode,
        }
    }

    /// Read the value at pixel `(x, y)`.  Returns `0.0` if out of bounds.
    pub fn pixel(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        let idx = y as usize * self.width as usize + x as usize;
        self.mask.get(idx).copied().unwrap_or(0.0)
    }

    /// Convert to binary mask: `1.0` where `value > threshold`, `0.0` elsewhere.
    pub fn binarize(&self, threshold: f32) -> Vec<f32> {
        self.mask
            .iter()
            .map(|&v| if v > threshold { 1.0f32 } else { 0.0f32 })
            .collect()
    }

    /// Convert to a `u8` image.
    ///
    /// Binary mode yields 0 or 255.  Soft modes scale linearly to `[0, 255]`.
    pub fn to_u8(&self) -> Vec<u8> {
        self.mask
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
    }

    /// Total foreground area: sum of all mask values.
    pub fn foreground_area(&self) -> f32 {
        self.mask.iter().sum()
    }

    /// Coverage fraction `[0, 1]`: `foreground_area / total_pixels`.
    ///
    /// Returns `0.0` for an empty (zero-sized) mask.
    pub fn coverage(&self) -> f32 {
        let total = self.mask.len();
        if total == 0 {
            return 0.0;
        }
        self.foreground_area() / total as f32
    }

    /// Compute the axis-aligned bounding box of foreground pixels
    /// (`value > threshold`).
    ///
    /// Returns `(min_x, min_y, max_x, max_y)` or `None` when no foreground
    /// pixels exist.
    pub fn bounding_box(&self, threshold: f32) -> Option<(u32, u32, u32, u32)> {
        let mut min_x = u32::MAX;
        let mut min_y = u32::MAX;
        let mut max_x = 0u32;
        let mut max_y = 0u32;
        let mut found = false;

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = y as usize * self.width as usize + x as usize;
                if let Some(&v) = self.mask.get(idx) {
                    if v > threshold {
                        found = true;
                        if x < min_x {
                            min_x = x;
                        }
                        if x > max_x {
                            max_x = x;
                        }
                        if y < min_y {
                            min_y = y;
                        }
                        if y > max_y {
                            max_y = y;
                        }
                    }
                }
            }
        }

        if found {
            Some((min_x, min_y, max_x, max_y))
        } else {
            None
        }
    }

    /// Erode the mask by `radius` pixels (morphological erosion with a disk SE).
    ///
    /// A pixel in the output is set to `1.0` only when all pixels in the
    /// disk of the given radius in the source are `> 0.5`.
    pub fn erode(&self, radius: u32) -> Self {
        let mut out = Self::new(self.width, self.height, self.mode);
        let r = radius as i32;
        let w = self.width as i32;
        let h = self.height as i32;

        for y in 0..h {
            for x in 0..w {
                let mut all_fg = true;
                'outer: for dy in -r..=r {
                    for dx in -r..=r {
                        if dx * dx + dy * dy > r * r {
                            continue;
                        }
                        let nx = x + dx;
                        let ny = y + dy;
                        if nx < 0 || nx >= w || ny < 0 || ny >= h {
                            // Out-of-bounds: treat as background.
                            all_fg = false;
                            break 'outer;
                        }
                        let idx = ny as usize * self.width as usize + nx as usize;
                        if self.mask.get(idx).copied().unwrap_or(0.0) <= 0.5 {
                            all_fg = false;
                            break 'outer;
                        }
                    }
                }
                let out_idx = y as usize * self.width as usize + x as usize;
                if let Some(v) = out.mask.get_mut(out_idx) {
                    *v = if all_fg { 1.0 } else { 0.0 };
                }
            }
        }
        out
    }

    /// Dilate the mask by `radius` pixels (morphological dilation with a disk SE).
    ///
    /// A pixel in the output is `1.0` when **any** pixel in the disk of the
    /// given radius in the source is `> 0.5`.
    pub fn dilate(&self, radius: u32) -> Self {
        let mut out = Self::new(self.width, self.height, self.mode);
        let r = radius as i32;
        let w = self.width as i32;
        let h = self.height as i32;

        for y in 0..h {
            for x in 0..w {
                let mut any_fg = false;
                'outer: for dy in -r..=r {
                    for dx in -r..=r {
                        if dx * dx + dy * dy > r * r {
                            continue;
                        }
                        let nx = x + dx;
                        let ny = y + dy;
                        if nx < 0 || nx >= w || ny < 0 || ny >= h {
                            continue;
                        }
                        let idx = ny as usize * self.width as usize + nx as usize;
                        if self.mask.get(idx).copied().unwrap_or(0.0) > 0.5 {
                            any_fg = true;
                            break 'outer;
                        }
                    }
                }
                let out_idx = y as usize * self.width as usize + x as usize;
                if let Some(v) = out.mask.get_mut(out_idx) {
                    *v = if any_fg { 1.0 } else { 0.0 };
                }
            }
        }
        out
    }

    /// Compute an edge map using finite differences.
    ///
    /// For each interior pixel: `edge = sqrt(dx² + dy²)` where
    /// `dx = (mask[x+1,y] - mask[x-1,y]) / 2` and
    /// `dy = (mask[x,y+1] - mask[x,y-1]) / 2`.
    ///
    /// Border pixels are set to `0.0`.
    /// The result is **not** normalized — values live in `[0, sqrt(2)/2]`
    /// for a binary input mask.
    pub fn edge_map(&self) -> Vec<f32> {
        let w = self.width as usize;
        let h = self.height as usize;
        let mut edges = vec![0.0f32; w * h];

        for y in 1..(h.saturating_sub(1)) {
            for x in 1..(w.saturating_sub(1)) {
                let get = |xi: usize, yi: usize| -> f32 {
                    self.mask.get(yi * w + xi).copied().unwrap_or(0.0)
                };
                let dx = (get(x + 1, y) - get(x - 1, y)) * 0.5;
                let dy = (get(x, y + 1) - get(x, y - 1)) * 0.5;
                let idx = y * w + x;
                if let Some(v) = edges.get_mut(idx) {
                    *v = (dx * dx + dy * dy).sqrt();
                }
            }
        }
        edges
    }

    /// Compute aggregate statistics over the mask.
    pub fn stats(&self, threshold: f32) -> SilhouetteStats {
        let total = self.mask.len();
        let mut sum = 0.0f32;
        let mut max_opacity = 0.0f32;
        let mut num_fg = 0usize;

        for &v in &self.mask {
            sum += v;
            if v > max_opacity {
                max_opacity = v;
            }
            if v > threshold {
                num_fg += 1;
            }
        }

        let mean_opacity = if total == 0 { 0.0 } else { sum / total as f32 };
        let coverage = if total == 0 {
            0.0
        } else {
            num_fg as f32 / total as f32
        };
        let bb = self.bounding_box(threshold);

        SilhouetteStats {
            coverage,
            mean_opacity,
            max_opacity,
            bounding_box: bb,
            num_foreground_pixels: num_fg,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SilhouetteStats
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics computed over a [`SilhouetteMask`].
#[derive(Debug, Clone)]
pub struct SilhouetteStats {
    /// Fraction of pixels above `threshold` in `[0, 1]`.
    pub coverage: f32,
    /// Mean mask value across all pixels.
    pub mean_opacity: f32,
    /// Maximum mask value across all pixels.
    pub max_opacity: f32,
    /// Bounding box of foreground pixels, or `None` if mask is all-zero.
    pub bounding_box: Option<(u32, u32, u32, u32)>,
    /// Number of pixels with value above `threshold`.
    pub num_foreground_pixels: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// render_silhouette (internal helpers)
// ─────────────────────────────────────────────────────────────────────────────

/// Splat a single Gaussian onto the mask using the given mode.
///
/// Steps:
/// 1. Project centre → `(px, py, depth)`.  Skip if `None` (behind camera).
/// 2. Compute projected pixel radius: `r_px = max(1.0, fx * radius / depth)`.
/// 3. Iterate over all pixels in the bounding square of the disk.
/// 4. Accept only pixels within the circle and within image bounds.
/// 5. Accumulate contribution according to `mode`.
fn splat_gaussian(
    mask: &mut [f32],
    camera: &SilhouetteCamera,
    g: &GaussianSilData,
    mode: SilhouetteMode,
) {
    let (px, py, depth) = match camera.project(g.center) {
        Some(v) => v,
        None => return,
    };

    let r_px = (camera.fx * g.radius / depth).max(1.0);
    let r_px_sq = r_px * r_px;

    let w = camera.width as i32;
    let h = camera.height as i32;
    let stride = camera.width as usize;

    let ix_min = (px - r_px).floor() as i32;
    let ix_max = (px + r_px).floor() as i32;
    let iy_min = (py - r_px).floor() as i32;
    let iy_max = (py + r_px).floor() as i32;

    for iy in iy_min..=iy_max {
        if iy < 0 || iy >= h {
            continue;
        }
        for ix in ix_min..=ix_max {
            if ix < 0 || ix >= w {
                continue;
            }
            let ddx = ix as f32 - px;
            let ddy = iy as f32 - py;
            if ddx * ddx + ddy * ddy > r_px_sq {
                continue;
            }

            let idx = iy as usize * stride + ix as usize;
            if let Some(v) = mask.get_mut(idx) {
                match mode {
                    SilhouetteMode::Binary => {
                        *v = 1.0;
                    }
                    SilhouetteMode::SoftOpacity => {
                        *v += g.opacity;
                    }
                    SilhouetteMode::DepthWeighted => {
                        *v += g.opacity / depth;
                    }
                    SilhouetteMode::MaxOpacity => {
                        if g.opacity > *v {
                            *v = g.opacity;
                        }
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public rendering API
// ─────────────────────────────────────────────────────────────────────────────

/// Render a silhouette mask from a Gaussian scene.
///
/// An empty `gaussians` slice is **not** an error; it returns an all-zero mask.
///
/// # Errors
///
/// Returns [`SilhouetteError::InvalidDimensions`] if `camera.width == 0` or
/// `camera.height == 0` (normally prevented by [`SilhouetteCamera::from_fov`],
/// but the struct is `pub` so manual construction is possible).
pub fn render_silhouette(
    camera: &SilhouetteCamera,
    gaussians: &[GaussianSilData],
    mode: SilhouetteMode,
) -> Result<SilhouetteMask, SilhouetteError> {
    if camera.width == 0 || camera.height == 0 {
        return Err(SilhouetteError::InvalidDimensions(
            "camera width and height must be non-zero".to_string(),
        ));
    }

    let mut result = SilhouetteMask::new(camera.width, camera.height, mode);

    for g in gaussians {
        splat_gaussian(&mut result.mask, camera, g, mode);
    }

    // Post-process: clamp accumulated modes to [0, 1].
    match mode {
        SilhouetteMode::Binary | SilhouetteMode::MaxOpacity => {
            // Binary is already {0, 1}; MaxOpacity is bounded by opacity ≤ 1.
        }
        SilhouetteMode::SoftOpacity | SilhouetteMode::DepthWeighted => {
            for v in &mut result.mask {
                if *v > 1.0 {
                    *v = 1.0;
                }
            }
        }
    }

    Ok(result)
}

/// Render silhouettes from multiple cameras.
///
/// Returns one `Result` per camera in the same order as `cameras`.
pub fn render_silhouettes(
    cameras: &[SilhouetteCamera],
    gaussians: &[GaussianSilData],
    mode: SilhouetteMode,
) -> Vec<Result<SilhouetteMask, SilhouetteError>> {
    cameras
        .iter()
        .map(|cam| render_silhouette(cam, gaussians, mode))
        .collect()
}

/// Render a per-pixel soft coverage mask using a Gaussian kernel.
///
/// For each Gaussian, weight contribution by:
/// `exp(-dist² / (2 * sigma²))`
/// where `sigma = projected_radius / 2`.
///
/// Results are accumulated and clamped to `[0, 1]`.
pub fn render_silhouette_antialiased(
    camera: &SilhouetteCamera,
    gaussians: &[GaussianSilData],
) -> Result<SilhouetteMask, SilhouetteError> {
    if camera.width == 0 || camera.height == 0 {
        return Err(SilhouetteError::InvalidDimensions(
            "camera width and height must be non-zero".to_string(),
        ));
    }

    let mut result = SilhouetteMask::new(camera.width, camera.height, SilhouetteMode::SoftOpacity);
    let w = camera.width as i32;
    let h = camera.height as i32;
    let stride = camera.width as usize;

    for g in gaussians {
        let (px, py, depth) = match camera.project(g.center) {
            Some(v) => v,
            None => continue,
        };

        let r_px = (camera.fx * g.radius / depth).max(1.0);
        let sigma = r_px * 0.5;
        let two_sigma_sq = 2.0 * sigma * sigma;

        let ix_min = (px - r_px).floor() as i32;
        let ix_max = (px + r_px).floor() as i32;
        let iy_min = (py - r_px).floor() as i32;
        let iy_max = (py + r_px).floor() as i32;

        for iy in iy_min..=iy_max {
            if iy < 0 || iy >= h {
                continue;
            }
            for ix in ix_min..=ix_max {
                if ix < 0 || ix >= w {
                    continue;
                }
                let ddx = ix as f32 - px;
                let ddy = iy as f32 - py;
                let dist_sq = ddx * ddx + ddy * ddy;

                // Only consider pixels within the disk (r_px).
                if dist_sq > r_px * r_px {
                    continue;
                }

                let weight = (-dist_sq / two_sigma_sq).exp();
                let contribution = g.opacity * weight;

                let idx = iy as usize * stride + ix as usize;
                if let Some(v) = result.mask.get_mut(idx) {
                    *v += contribution;
                }
            }
        }
    }

    // Clamp to [0, 1].
    for v in &mut result.mask {
        if *v > 1.0 {
            *v = 1.0;
        }
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Loss and metric functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute binary cross-entropy between a predicted soft mask and a binary target.
///
/// `Loss = -mean(t * log(p + ε) + (1 - t) * log(1 - p + ε))` where `ε = 1e-7`.
///
/// # Errors
///
/// Returns [`SilhouetteError::InvalidDimensions`] if `target.len() !=
/// mask.width * mask.height`.
pub fn silhouette_bce_loss(pred: &SilhouetteMask, target: &[f32]) -> Result<f32, SilhouetteError> {
    let expected = pred.width as usize * pred.height as usize;
    if target.len() != expected {
        return Err(SilhouetteError::InvalidDimensions(format!(
            "target length {} does not match mask size {}×{}={}",
            target.len(),
            pred.width,
            pred.height,
            expected
        )));
    }
    if expected == 0 {
        return Ok(0.0);
    }

    const EPS: f32 = 1e-7;
    let mut total = 0.0f32;

    for (p, t) in pred.mask.iter().zip(target.iter()) {
        let p_clamped = p.clamp(EPS, 1.0 - EPS);
        total -= t * (p_clamped + EPS).ln() + (1.0 - t) * (1.0 - p_clamped + EPS).ln();
    }

    Ok(total / expected as f32)
}

/// Compute Intersection over Union between two binary masks.
///
/// Both masks are thresholded at `threshold` before computing I/U.
///
/// `IoU = |A ∩ B| / |A ∪ B|`.  Returns `0.0` when the union is empty.
///
/// # Errors
///
/// Returns [`SilhouetteError::InvalidDimensions`] when `mask_a` and `mask_b`
/// have different width/height.
pub fn silhouette_iou(
    mask_a: &SilhouetteMask,
    mask_b: &SilhouetteMask,
    threshold: f32,
) -> Result<f32, SilhouetteError> {
    if mask_a.width != mask_b.width || mask_a.height != mask_b.height {
        return Err(SilhouetteError::InvalidDimensions(format!(
            "mask dimensions differ: ({}×{}) vs ({}×{})",
            mask_a.width, mask_a.height, mask_b.width, mask_b.height
        )));
    }

    let mut intersection = 0u64;
    let mut union = 0u64;

    for (a, b) in mask_a.mask.iter().zip(mask_b.mask.iter()) {
        let a_fg = *a > threshold;
        let b_fg = *b > threshold;
        if a_fg && b_fg {
            intersection += 1;
        }
        if a_fg || b_fg {
            union += 1;
        }
    }

    if union == 0 {
        Ok(0.0)
    } else {
        Ok(intersection as f32 / union as f32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a standard 64×64 camera looking down +Z (camera at (0,0,5),
    /// world origin in front of camera).
    fn make_camera(w: u32, h: u32) -> SilhouetteCamera {
        SilhouetteCamera::from_fov(
            [0.0, 0.0, 5.0],
            // Identity rotation — world-space Z aligns with camera -Z only after
            // the depth formula (`depth = -cam_z`).  With the identity rotation the
            // camera -Z axis points toward -Z world, so origin (0,0,0) is at depth 5.
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            w,
            h,
            std::f32::consts::FRAC_PI_2,
            0.1,
            100.0,
        )
        .unwrap()
    }

    fn make_centered_gaussian(opacity: f32) -> GaussianSilData {
        GaussianSilData {
            center: [0.0, 0.0, 0.0],
            radius: 0.5,
            opacity,
        }
    }

    // ── Test 1: SilhouetteCamera::from_fov valid → Ok ────────────────────────
    #[test]
    fn test_camera_from_fov_valid() {
        let cam = SilhouetteCamera::from_fov(
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            64,
            64,
            std::f32::consts::FRAC_PI_2,
            0.1,
            100.0,
        );
        assert!(cam.is_ok(), "expected Ok, got {cam:?}");
        let c = cam.unwrap();
        assert_eq!(c.width, 64);
        assert_eq!(c.height, 64);
        assert!(c.fx > 0.0);
        assert!(c.fy > 0.0);
    }

    // ── Test 2: SilhouetteCamera::from_fov zero fov → Err ───────────────────
    #[test]
    fn test_camera_from_fov_zero_fov() {
        let result = SilhouetteCamera::from_fov(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            64,
            64,
            0.0, // invalid
            0.1,
            100.0,
        );
        assert!(
            matches!(result, Err(SilhouetteError::InvalidCamera(_))),
            "expected InvalidCamera, got {result:?}"
        );
    }

    // ── Test 3: project: point in front → Some ───────────────────────────────
    #[test]
    fn test_project_in_front() {
        let cam = make_camera(64, 64);
        // World origin: in front of camera at z=0, camera at z=5.
        // depth = -(cam_z) = -(-5) = 5 > 0.
        let result = cam.project([0.0, 0.0, 0.0]);
        assert!(
            result.is_some(),
            "point in front should project successfully"
        );
        let (px, py, depth) = result.unwrap();
        // Should project near the center.
        assert!((px - 32.0).abs() < 1.0, "px should be near center: {px}");
        assert!((py - 32.0).abs() < 1.0, "py should be near center: {py}");
        assert!(depth > 0.0, "depth should be positive: {depth}");
    }

    // ── Test 4: project: point behind camera → None ──────────────────────────
    #[test]
    fn test_project_behind_camera() {
        let cam = make_camera(64, 64);
        // Point at z=10, camera at z=5 looking toward -z: cam_z = 10 - 5 = 5, depth = -5 < 0.
        let result = cam.project([0.0, 0.0, 10.0]);
        assert!(
            result.is_none(),
            "point behind camera should return None, got {result:?}"
        );
    }

    // ── Test 5: render Binary: single Gaussian at center → foreground pixels ─
    #[test]
    fn test_render_binary_single_gaussian_has_foreground() {
        let cam = make_camera(64, 64);
        let gaussians = vec![make_centered_gaussian(1.0)];
        let mask = render_silhouette(&cam, &gaussians, SilhouetteMode::Binary).unwrap();
        let area = mask.foreground_area();
        assert!(
            area > 0.0,
            "binary mask should have foreground pixels, got area={area}"
        );
        // Center pixel should be set.
        let cx = mask.width / 2;
        let cy = mask.height / 2;
        assert_eq!(
            mask.pixel(cx, cy),
            1.0,
            "center pixel should be 1.0 in binary mode"
        );
    }

    // ── Test 6: render Binary: empty gaussians → all-zero mask (not error) ───
    #[test]
    fn test_render_binary_empty_gaussians() {
        let cam = make_camera(32, 32);
        let mask = render_silhouette(&cam, &[], SilhouetteMode::Binary).unwrap();
        assert_eq!(
            mask.foreground_area(),
            0.0,
            "empty scene should give all-zero mask"
        );
    }

    // ── Test 7: SoftOpacity: opacity accumulates ─────────────────────────────
    #[test]
    fn test_render_soft_opacity_accumulates() {
        let cam = make_camera(64, 64);
        // Two overlapping Gaussians at the same position.
        let gaussians = vec![
            GaussianSilData {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
                opacity: 0.3,
            },
            GaussianSilData {
                center: [0.0, 0.0, 0.0],
                radius: 1.0,
                opacity: 0.3,
            },
        ];
        let mask_soft = render_silhouette(&cam, &gaussians, SilhouetteMode::SoftOpacity).unwrap();
        // Single Gaussian mask for comparison.
        let mask_single =
            render_silhouette(&cam, &gaussians[..1], SilhouetteMode::SoftOpacity).unwrap();

        let area_soft = mask_soft.foreground_area();
        let area_single = mask_single.foreground_area();
        assert!(
            area_soft > area_single,
            "two overlapping Gaussians should accumulate more than one: {area_soft} vs {area_single}"
        );
    }

    // ── Test 8: MaxOpacity: max value ≤ max Gaussian opacity ─────────────────
    #[test]
    fn test_render_max_opacity_bounded() {
        let cam = make_camera(64, 64);
        let max_alpha = 0.7f32;
        let gaussians = vec![
            GaussianSilData {
                center: [0.0, 0.0, 0.0],
                radius: 1.5,
                opacity: max_alpha,
            },
            GaussianSilData {
                center: [0.0, 0.0, 0.0],
                radius: 1.5,
                opacity: 0.4,
            },
        ];
        let mask = render_silhouette(&cam, &gaussians, SilhouetteMode::MaxOpacity).unwrap();
        let max_val = mask.mask.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            max_val <= max_alpha + 1e-5,
            "MaxOpacity should not exceed max Gaussian opacity {max_alpha}, got {max_val}"
        );
        assert!(
            max_val > 0.0,
            "MaxOpacity mask should have some foreground, got {max_val}"
        );
    }

    // ── Test 9: coverage: all-zero mask → 0.0 ────────────────────────────────
    #[test]
    fn test_coverage_all_zero() {
        let mask = SilhouetteMask::new(32, 32, SilhouetteMode::Binary);
        assert_eq!(
            mask.coverage(),
            0.0,
            "all-zero mask should have coverage 0.0"
        );
    }

    // ── Test 10: binarize: values above threshold → 1.0 ──────────────────────
    #[test]
    fn test_binarize_threshold() {
        let mut mask = SilhouetteMask::new(4, 1, SilhouetteMode::SoftOpacity);
        mask.mask[0] = 0.1;
        mask.mask[1] = 0.5;
        mask.mask[2] = 0.9;
        mask.mask[3] = 0.0;

        let binary = mask.binarize(0.5);
        // 0.1 ≤ 0.5 → 0, 0.5 is NOT > 0.5 → 0, 0.9 > 0.5 → 1, 0.0 → 0.
        assert_eq!(binary[0], 0.0);
        assert_eq!(binary[1], 0.0);
        assert_eq!(binary[2], 1.0);
        assert_eq!(binary[3], 0.0);
    }

    // ── Test 11: to_u8: all-zero → all-zero bytes ─────────────────────────────
    #[test]
    fn test_to_u8_all_zero() {
        let mask = SilhouetteMask::new(8, 8, SilhouetteMode::Binary);
        let bytes = mask.to_u8();
        assert!(
            bytes.iter().all(|&b| b == 0),
            "all-zero mask should give all-zero bytes"
        );
    }

    // ── Test 12: bounding_box: centered Gaussian → Some non-zero box ─────────
    #[test]
    fn test_bounding_box_centered_gaussian() {
        let cam = make_camera(64, 64);
        let gaussians = vec![make_centered_gaussian(1.0)];
        let mask = render_silhouette(&cam, &gaussians, SilhouetteMode::Binary).unwrap();
        let bb = mask.bounding_box(0.5);
        assert!(
            bb.is_some(),
            "bounding box should be Some for non-empty mask"
        );
        let (min_x, min_y, max_x, max_y) = bb.unwrap();
        let cx = mask.width / 2;
        let cy = mask.height / 2;
        assert!(min_x <= cx, "min_x {min_x} should be ≤ cx {cx}");
        assert!(max_x >= cx, "max_x {max_x} should be ≥ cx {cx}");
        assert!(min_y <= cy, "min_y {min_y} should be ≤ cy {cy}");
        assert!(max_y >= cy, "max_y {max_y} should be ≥ cy {cy}");
    }

    // ── Test 13: bounding_box: empty mask → None ─────────────────────────────
    #[test]
    fn test_bounding_box_empty() {
        let mask = SilhouetteMask::new(32, 32, SilhouetteMode::Binary);
        assert!(
            mask.bounding_box(0.5).is_none(),
            "all-zero mask should have no bounding box"
        );
    }

    // ── Test 14: erode: eroded mask smaller than original ────────────────────
    #[test]
    fn test_erode_smaller() {
        let cam = make_camera(64, 64);
        // Large Gaussian to fill a big region.
        let gaussians = vec![GaussianSilData {
            center: [0.0, 0.0, 0.0],
            radius: 3.0,
            opacity: 1.0,
        }];
        let mask = render_silhouette(&cam, &gaussians, SilhouetteMode::Binary).unwrap();
        let eroded = mask.erode(3);

        let orig_area = mask.foreground_area();
        let eroded_area = eroded.foreground_area();
        assert!(
            eroded_area <= orig_area,
            "eroded mask ({eroded_area}) should not exceed original ({orig_area})"
        );
    }

    // ── Test 15: dilate: dilated mask larger than or equal to original ────────
    #[test]
    fn test_dilate_larger_or_equal() {
        let cam = make_camera(64, 64);
        let gaussians = vec![make_centered_gaussian(1.0)];
        let mask = render_silhouette(&cam, &gaussians, SilhouetteMode::Binary).unwrap();
        let dilated = mask.dilate(3);

        let orig_area = mask.foreground_area();
        let dilated_area = dilated.foreground_area();
        assert!(
            dilated_area >= orig_area,
            "dilated mask ({dilated_area}) should be ≥ original ({orig_area})"
        );
    }

    // ── Test 16: edge_map: uniform mask has near-zero edges ──────────────────
    #[test]
    fn test_edge_map_uniform() {
        let mut mask = SilhouetteMask::new(16, 16, SilhouetteMode::Binary);
        // Fill with a uniform value — interior gradients should be zero.
        for v in &mut mask.mask {
            *v = 1.0;
        }
        let edges = mask.edge_map();
        let max_edge = edges.iter().cloned().fold(0.0f32, f32::max);
        // Interior pixels must have zero gradient; only border might differ.
        // Since borders are 0.0 and interior is uniform, gradient at border
        // pixels (which are set to 0.0 by the edge_map function) is 0. The
        // first interior row/column will have a one-sided difference though.
        // We check the absolute interior center region.
        let w = 16usize;
        // Sample a pixel at (8, 8) — should be near zero.
        let center_edge = edges[8 * w + 8];
        assert!(
            center_edge.abs() < 1e-5,
            "uniform interior should have near-zero edge, got {center_edge}"
        );
        let _ = max_edge; // tolerate non-zero at borders
    }

    // ── Test 17: silhouette_bce_loss: perfect prediction → near-zero loss ────
    #[test]
    fn test_bce_loss_perfect_prediction() {
        let mut pred = SilhouetteMask::new(4, 4, SilhouetteMode::SoftOpacity);
        // Set pred values close to target.
        for (i, v) in pred.mask.iter_mut().enumerate() {
            *v = if i < 8 { 1.0 } else { 0.0 };
        }
        let target: Vec<f32> = (0..16).map(|i| if i < 8 { 1.0 } else { 0.0 }).collect();
        let loss = silhouette_bce_loss(&pred, &target).unwrap();
        // BCE with perfect prediction is very small (only eps contribution).
        assert!(
            loss < 0.1,
            "near-perfect prediction should give small BCE loss, got {loss}"
        );
    }

    // ── Test 18: silhouette_bce_loss: size mismatch → Err ───────────────────
    #[test]
    fn test_bce_loss_size_mismatch() {
        let pred = SilhouetteMask::new(4, 4, SilhouetteMode::SoftOpacity);
        let target = vec![0.0f32; 5]; // wrong size
        let result = silhouette_bce_loss(&pred, &target);
        assert!(
            matches!(result, Err(SilhouetteError::InvalidDimensions(_))),
            "size mismatch should give InvalidDimensions, got {result:?}"
        );
    }

    // ── Test 19: silhouette_iou: identical masks → 1.0 ───────────────────────
    #[test]
    fn test_iou_identical_masks() {
        let cam = make_camera(32, 32);
        let gaussians = vec![make_centered_gaussian(1.0)];
        let mask = render_silhouette(&cam, &gaussians, SilhouetteMode::Binary).unwrap();
        // Clone and compare against itself.
        let mask2 = mask.clone();
        let iou = silhouette_iou(&mask, &mask2, 0.5).unwrap();
        assert!(
            (iou - 1.0).abs() < 1e-5,
            "identical masks should have IoU=1.0, got {iou}"
        );
    }

    // ── Test 20: silhouette_iou: non-overlapping → 0.0 ──────────────────────
    #[test]
    fn test_iou_non_overlapping() {
        let mut mask_a = SilhouetteMask::new(4, 1, SilhouetteMode::Binary);
        let mut mask_b = SilhouetteMask::new(4, 1, SilhouetteMode::Binary);
        mask_a.mask[0] = 1.0; // left pixel
        mask_b.mask[3] = 1.0; // right pixel
        let iou = silhouette_iou(&mask_a, &mask_b, 0.5).unwrap();
        assert!(
            iou.abs() < 1e-5,
            "non-overlapping masks should have IoU=0.0, got {iou}"
        );
    }

    // ── Test 21: silhouette_iou: dimension mismatch → Err ───────────────────
    #[test]
    fn test_iou_dimension_mismatch() {
        let mask_a = SilhouetteMask::new(4, 4, SilhouetteMode::Binary);
        let mask_b = SilhouetteMask::new(8, 8, SilhouetteMode::Binary);
        let result = silhouette_iou(&mask_a, &mask_b, 0.5);
        assert!(
            matches!(result, Err(SilhouetteError::InvalidDimensions(_))),
            "dimension mismatch should give InvalidDimensions, got {result:?}"
        );
    }

    // ── Test 22: render_silhouette_antialiased: center pixel high value ───────
    #[test]
    fn test_antialiased_center_pixel_high_value() {
        let cam = make_camera(64, 64);
        let gaussians = vec![make_centered_gaussian(1.0)];
        let mask = render_silhouette_antialiased(&cam, &gaussians).unwrap();
        let cx = mask.width / 2;
        let cy = mask.height / 2;
        let center_val = mask.pixel(cx, cy);
        assert!(
            center_val > 0.5,
            "antialiased center pixel should have high value for centered Gaussian, got {center_val}"
        );
    }

    // ── Test 23: SilhouetteStats: coverage and num_foreground_pixels consistent
    #[test]
    fn test_stats_consistency() {
        let cam = make_camera(64, 64);
        let gaussians = vec![make_centered_gaussian(1.0)];
        let mask = render_silhouette(&cam, &gaussians, SilhouetteMode::Binary).unwrap();
        let stats = mask.stats(0.5);

        let total_pixels = (mask.width * mask.height) as f32;
        let expected_coverage = stats.num_foreground_pixels as f32 / total_pixels;
        assert!(
            (stats.coverage - expected_coverage).abs() < 1e-5,
            "coverage should equal num_fg/total: {:.4} vs {:.4}",
            stats.coverage,
            expected_coverage
        );
        assert!(
            stats.num_foreground_pixels > 0,
            "should have some foreground pixels"
        );
        assert!(stats.max_opacity > 0.0, "max opacity should be positive");
        assert!(
            stats.bounding_box.is_some(),
            "bounding box should be Some for non-empty mask"
        );
    }
}
