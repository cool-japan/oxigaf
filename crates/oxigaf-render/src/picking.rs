//! CPU-side ray-Gaussian intersection for 3DGS scene picking/selection.
//!
//! When a user clicks on the viewport, a ray is cast from the camera through the
//! pixel, and we find which Gaussian is "hit" (nearest visible intersection).
//!
//! # Overview
//!
//! - [`Ray`]: A normalized ray (origin + direction).
//! - [`PickCamera`]: Pin-hole camera that generates rays for pixels.
//! - [`GaussianPickData`]: Per-Gaussian bounding-sphere + opacity data.
//! - [`pick_nearest`]: Returns the closest Gaussian hit by a single ray.
//! - [`pick_all`]: Returns all hit Gaussians sorted by distance.
//! - [`pick_closest_approach`]: Closest-approach fallback (no sphere required).
//! - [`pick_region`]: Batch ray-cast over a rectangular pixel region.

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the picking subsystem.
#[derive(Debug, Error)]
pub enum PickingError {
    /// The supplied ray direction was zero-length (or too close to zero).
    #[error("Invalid ray: {0}")]
    InvalidRay(String),

    /// The supplied camera parameters are invalid.
    #[error("Invalid camera: {0}")]
    InvalidCamera(String),

    /// The Gaussian scene is empty — there is nothing to pick.
    #[error("Empty scene: no Gaussians to pick")]
    EmptyScene,
}

// ─────────────────────────────────────────────────────────────────────────────
// Ray
// ─────────────────────────────────────────────────────────────────────────────

/// A ray defined by an origin and a (normalized) direction.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    /// World-space origin of the ray.
    pub origin: [f32; 3],
    /// World-space direction (always unit-length after construction).
    pub direction: [f32; 3],
}

impl Ray {
    /// Construct a ray from `origin` and `direction`.
    ///
    /// `direction` is normalized internally.  Returns [`PickingError::InvalidRay`]
    /// if the direction length is below the machine-epsilon threshold.
    pub fn new(origin: [f32; 3], direction: [f32; 3]) -> Result<Self, PickingError> {
        let len_sq =
            direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];

        if len_sq < f32::EPSILON * f32::EPSILON {
            return Err(PickingError::InvalidRay(
                "direction vector is zero-length".to_string(),
            ));
        }

        let inv_len = 1.0 / len_sq.sqrt();
        Ok(Self {
            origin,
            direction: [
                direction[0] * inv_len,
                direction[1] * inv_len,
                direction[2] * inv_len,
            ],
        })
    }

    /// Evaluate the ray at parameter `t`: `origin + t * direction`.
    #[inline]
    pub fn at(&self, t: f32) -> [f32; 3] {
        [
            self.origin[0] + t * self.direction[0],
            self.origin[1] + t * self.direction[1],
            self.origin[2] + t * self.direction[2],
        ]
    }

    /// Returns the length of the direction vector (always `1.0` for a valid ray).
    #[inline]
    pub fn direction_len(&self) -> f32 {
        let d = &self.direction;
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PickCamera
// ─────────────────────────────────────────────────────────────────────────────

/// Camera parameters used to generate picking rays.
///
/// The rotation matrix is row-major, 3×3: `rotation[i*3+j]` = R(i, j).
/// Column 0 is the right axis, column 1 is the up axis, and column 2 is the
/// *negative* forward axis (OpenGL convention: camera looks down −Z).
#[derive(Debug, Clone)]
pub struct PickCamera {
    /// Camera position in world space.
    pub position: [f32; 3],
    /// Camera rotation matrix (row-major 3×3): transforms view → world.
    pub rotation: [f32; 9],
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Horizontal field of view in radians.
    pub fov_x: f32,
}

impl PickCamera {
    /// Create a new [`PickCamera`], validating all parameters.
    ///
    /// # Errors
    ///
    /// Returns [`PickingError::InvalidCamera`] if:
    /// - `width` or `height` is 0,
    /// - `fov_x` is not in the range `(0, π)`.
    pub fn new(
        position: [f32; 3],
        rotation: [f32; 9],
        width: u32,
        height: u32,
        fov_x: f32,
    ) -> Result<Self, PickingError> {
        if width == 0 {
            return Err(PickingError::InvalidCamera(
                "image width must be > 0".to_string(),
            ));
        }
        if height == 0 {
            return Err(PickingError::InvalidCamera(
                "image height must be > 0".to_string(),
            ));
        }
        if fov_x <= 0.0 || fov_x >= std::f32::consts::PI {
            return Err(PickingError::InvalidCamera(format!(
                "fov_x must be in (0, π), got {fov_x}"
            )));
        }

        Ok(Self {
            position,
            rotation,
            width,
            height,
            fov_x,
        })
    }

    /// Generate the picking ray for pixel `(px, py)` using a pin-hole camera model.
    ///
    /// Pixel coordinates are in `[0, width) × [0, height)`.
    ///
    /// # Errors
    ///
    /// Returns [`PickingError::InvalidCamera`] if the pixel is outside the image
    /// bounds, or [`PickingError::InvalidRay`] if the computed direction is
    /// degenerate (should not happen for valid camera parameters).
    pub fn ray_for_pixel(&self, px: u32, py: u32) -> Result<Ray, PickingError> {
        if px >= self.width {
            return Err(PickingError::InvalidCamera(format!(
                "pixel x={px} out of range [0, {})",
                self.width
            )));
        }
        if py >= self.height {
            return Err(PickingError::InvalidCamera(format!(
                "pixel y={py} out of range [0, {})",
                self.height
            )));
        }

        // NDC in [-1, 1]
        let nx = (px as f32 + 0.5) / self.width as f32 * 2.0 - 1.0;
        let ny = 1.0 - (py as f32 + 0.5) / self.height as f32 * 2.0;

        let aspect = self.width as f32 / self.height as f32;
        let tan_half_fov_x = (self.fov_x * 0.5).tan();
        let tan_half_fov_y = tan_half_fov_x / aspect;

        // View-space direction (camera looks down −Z).
        let vd = [nx * tan_half_fov_x, ny * tan_half_fov_y, -1.0_f32];

        // Transform to world space: world = R * vd (row-major multiply).
        let r = &self.rotation;
        let world_dir = [
            r[0] * vd[0] + r[1] * vd[1] + r[2] * vd[2],
            r[3] * vd[0] + r[4] * vd[1] + r[5] * vd[2],
            r[6] * vd[0] + r[7] * vd[1] + r[8] * vd[2],
        ];

        Ray::new(self.position, world_dir)
    }

    /// Generate rays for every pixel in the rectangular region
    /// `[x0, x1) × [y0, y1)`.
    ///
    /// Rays are ordered row-major (y outer, x inner).
    ///
    /// # Errors
    ///
    /// Returns [`PickingError::InvalidCamera`] if the region is degenerate
    /// (x0 ≥ x1 or y0 ≥ y1, or coordinates exceed image size).
    pub fn rays_for_region(
        &self,
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
    ) -> Result<Vec<Ray>, PickingError> {
        if x0 >= x1 {
            return Err(PickingError::InvalidCamera(format!(
                "region x range is empty: x0={x0} >= x1={x1}"
            )));
        }
        if y0 >= y1 {
            return Err(PickingError::InvalidCamera(format!(
                "region y range is empty: y0={y0} >= y1={y1}"
            )));
        }
        if x1 > self.width {
            return Err(PickingError::InvalidCamera(format!(
                "region x1={x1} exceeds image width {}",
                self.width
            )));
        }
        if y1 > self.height {
            return Err(PickingError::InvalidCamera(format!(
                "region y1={y1} exceeds image height {}",
                self.height
            )));
        }

        let num_rays = ((x1 - x0) * (y1 - y0)) as usize;
        let mut rays = Vec::with_capacity(num_rays);
        for py in y0..y1 {
            for px in x0..x1 {
                rays.push(self.ray_for_pixel(px, py)?);
            }
        }
        Ok(rays)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GaussianPickData
// ─────────────────────────────────────────────────────────────────────────────

/// Per-Gaussian data required for picking.
#[derive(Debug, Clone, Copy)]
pub struct GaussianPickData {
    /// World-space center of the Gaussian.
    pub center: [f32; 3],
    /// Bounding-sphere radius (e.g. `3 × max_scale`).
    pub radius: f32,
    /// Opacity in `[0, 1]`, used to filter near-transparent Gaussians.
    pub opacity: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// PickHit
// ─────────────────────────────────────────────────────────────────────────────

/// A successful ray-Gaussian intersection result.
#[derive(Debug, Clone)]
pub struct PickHit {
    /// Index of the hit Gaussian in the input slice.
    pub index: usize,
    /// Ray parameter `t` at the sphere surface intersection (entry point).
    pub t: f32,
    /// Closest point on the ray to the Gaussian center.
    pub closest_point: [f32; 3],
    /// Perpendicular distance from the ray to the Gaussian center.
    pub distance_to_center: f32,
    /// Opacity of the hit Gaussian.
    pub opacity: f32,
}

impl PickHit {
    /// World-space position of the pick hit — identical to `closest_point`.
    #[inline]
    pub fn world_position(&self) -> [f32; 3] {
        self.closest_point
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PickConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration controlling the picking test.
#[derive(Debug, Clone)]
pub struct PickConfig {
    /// Multiplier applied to each Gaussian's bounding-sphere radius before the
    /// intersection test.  Values > 1 make the test more permissive.
    pub radius_scale: f32,
    /// Gaussians with opacity below this threshold are ignored.
    pub min_opacity: f32,
    /// Maximum perpendicular ray-to-center distance (world units) that still
    /// counts as a hit.
    pub max_hit_distance: f32,
    /// Maximum ray-parameter `t` to search (ray length limit).
    pub max_t: f32,
}

impl Default for PickConfig {
    fn default() -> Self {
        Self {
            radius_scale: 1.0,
            min_opacity: 0.1,
            max_hit_distance: f32::MAX,
            max_t: 1000.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Primitive intersection helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the closest approach of a `ray` to a world-space `point`.
///
/// Returns `(t, distance)` where:
/// - `t` is the ray parameter at the closest point (`origin + t * direction`),
/// - `distance` is the perpendicular distance from the ray to `point`.
pub fn ray_point_distance(ray: &Ray, point: [f32; 3]) -> (f32, f32) {
    // Vector from origin to point.
    let vec = [
        point[0] - ray.origin[0],
        point[1] - ray.origin[1],
        point[2] - ray.origin[2],
    ];

    // Project onto (unit) direction.
    let t = vec[0] * ray.direction[0] + vec[1] * ray.direction[1] + vec[2] * ray.direction[2];

    // Closest point on the ray (unclamped — negative t means behind origin).
    let closest = ray.at(t);

    // Perpendicular distance.
    let dx = point[0] - closest[0];
    let dy = point[1] - closest[1];
    let dz = point[2] - closest[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    (t, dist)
}

/// Analytic ray-sphere intersection.
///
/// Returns `Some(t_min)` where `t_min > 0` is the entry `t` (or the exit `t`
/// if the ray starts inside the sphere).  Returns `None` if the ray misses or
/// the sphere is entirely behind the ray origin.
pub fn ray_sphere_intersect(ray: &Ray, center: [f32; 3], radius: f32) -> Option<f32> {
    // oc = origin − center
    let oc = [
        ray.origin[0] - center[0],
        ray.origin[1] - center[1],
        ray.origin[2] - center[2],
    ];

    // a = dot(dir, dir) = 1 (normalized direction)
    let b = 2.0 * (oc[0] * ray.direction[0] + oc[1] * ray.direction[1] + oc[2] * ray.direction[2]);
    let c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - radius * radius;

    let discriminant = b * b - 4.0 * c;

    if discriminant < 0.0 {
        return None; // Ray misses the sphere.
    }

    let sqrt_disc = discriminant.sqrt();
    let t1 = (-b - sqrt_disc) * 0.5;
    let t2 = (-b + sqrt_disc) * 0.5;

    if t2 < 0.0 {
        // Both intersections are behind the origin.
        return None;
    }

    if t1 < 0.0 {
        // Origin is inside the sphere; return the exit point.
        Some(t2)
    } else {
        // Standard front-face intersection.
        Some(t1)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Picking functions
// ─────────────────────────────────────────────────────────────────────────────

/// Pick the **nearest** Gaussian hit by `ray`.
///
/// Only Gaussians with `opacity >= config.min_opacity` are considered.
/// The bounding sphere is scaled by `config.radius_scale` before the test.
/// The sphere-entry `t` must be ≤ `config.max_t`, and the perpendicular
/// ray-to-center distance must be ≤ `config.max_hit_distance`.
///
/// Returns `None` if no Gaussian is hit.
pub fn pick_nearest(
    ray: &Ray,
    gaussians: &[GaussianPickData],
    config: &PickConfig,
) -> Option<PickHit> {
    let mut best: Option<PickHit> = None;

    for (index, g) in gaussians.iter().enumerate() {
        // Step 1: opacity gate.
        if g.opacity < config.min_opacity {
            continue;
        }

        // Step 2: bounding-sphere intersection.
        let scaled_radius = g.radius * config.radius_scale;
        let sphere_t = match ray_sphere_intersect(ray, g.center, scaled_radius) {
            Some(t) => t,
            None => continue,
        };

        // Step 2b: ray-length gate (config.max_t).
        if sphere_t > config.max_t {
            continue;
        }

        // Step 3: closest approach for quality metrics.
        let (t_closest, dist) = ray_point_distance(ray, g.center);
        let closest_point = ray.at(t_closest);

        // Step 4: distance gate.
        if dist > config.max_hit_distance {
            continue;
        }

        // Step 5: keep the nearest by sphere-entry t.
        let is_nearer = best.as_ref().is_none_or(|b| sphere_t < b.t);
        if is_nearer {
            best = Some(PickHit {
                index,
                t: sphere_t,
                closest_point,
                distance_to_center: dist,
                opacity: g.opacity,
            });
        }
    }

    best
}

/// Pick **all** Gaussians hit by `ray`, sorted by `t` (nearest first).
///
/// Applies the same filters as [`pick_nearest`], including `config.max_t`.
pub fn pick_all(ray: &Ray, gaussians: &[GaussianPickData], config: &PickConfig) -> Vec<PickHit> {
    let mut hits = Vec::new();

    for (index, g) in gaussians.iter().enumerate() {
        if g.opacity < config.min_opacity {
            continue;
        }

        let scaled_radius = g.radius * config.radius_scale;
        let sphere_t = match ray_sphere_intersect(ray, g.center, scaled_radius) {
            Some(t) => t,
            None => continue,
        };

        if sphere_t > config.max_t {
            continue;
        }

        let (t_closest, dist) = ray_point_distance(ray, g.center);
        let closest_point = ray.at(t_closest);

        if dist > config.max_hit_distance {
            continue;
        }

        hits.push(PickHit {
            index,
            t: sphere_t,
            closest_point,
            distance_to_center: dist,
            opacity: g.opacity,
        });
    }

    // Sort by sphere-entry t, nearest first.
    hits.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    hits
}

/// Closest-approach pick: no sphere intersection required.
///
/// Returns the Gaussian whose **center** is geometrically closest to the ray
/// (measured as perpendicular distance), subject to:
/// - `t > 0` (Gaussian must be in front of the camera),
/// - `t < config.max_t`,
/// - perpendicular distance ≤ `config.max_hit_distance`,
/// - `opacity >= config.min_opacity`.
///
/// This is a good fallback for very small or very distant Gaussians whose
/// bounding sphere might be missed.
pub fn pick_closest_approach(
    ray: &Ray,
    gaussians: &[GaussianPickData],
    config: &PickConfig,
) -> Option<PickHit> {
    let mut best: Option<PickHit> = None;

    for (index, g) in gaussians.iter().enumerate() {
        if g.opacity < config.min_opacity {
            continue;
        }

        let (t, dist) = ray_point_distance(ray, g.center);

        if t <= 0.0 {
            continue; // Behind the camera.
        }
        if t >= config.max_t {
            continue;
        }
        if dist > config.max_hit_distance {
            continue;
        }

        let closest_point = ray.at(t);

        let is_closer = best.as_ref().is_none_or(|b| dist < b.distance_to_center);
        if is_closer {
            best = Some(PickHit {
                index,
                t,
                closest_point,
                distance_to_center: dist,
                opacity: g.opacity,
            });
        }
    }

    best
}

// ─────────────────────────────────────────────────────────────────────────────
// Region picking
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a batch pick over a rectangular pixel region.
#[derive(Debug, Clone)]
pub struct RegionPickResult {
    /// All Gaussian hits collected across all rays in the region.
    pub hits: Vec<PickHit>,
    /// Total number of Gaussians tested (rays × Gaussians per ray before any
    /// early-out).
    pub total_tested: usize,
    /// Number of rays that were cast.
    pub num_rays: usize,
}

/// Cast a ray for every pixel in `[x0, x1) × [y0, y1)` and collect all hits.
///
/// Hits from different rays may include duplicate Gaussian indices (if the same
/// Gaussian is hit by several rays).  Hits are appended in row-major ray order.
///
/// # Errors
///
/// Forwards errors from [`PickCamera::rays_for_region`].
pub fn pick_region(
    camera: &PickCamera,
    gaussians: &[GaussianPickData],
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    config: &PickConfig,
) -> Result<RegionPickResult, PickingError> {
    let rays = camera.rays_for_region(x0, y0, x1, y1)?;
    let num_rays = rays.len();
    let mut all_hits = Vec::new();
    let total_tested = num_rays * gaussians.len();

    for ray in &rays {
        let hits = pick_all(ray, gaussians, config);
        all_hits.extend(hits);
    }

    Ok(RegionPickResult {
        hits: all_hits,
        total_tested,
        num_rays,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Pick statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics about a picking operation.
#[derive(Debug, Clone)]
pub struct PickStats {
    /// Total number of Gaussians in the scene.
    pub total_gaussians: usize,
    /// Number of Gaussians skipped due to low opacity.
    pub skipped_opacity: usize,
    /// Number of recorded hits.
    pub num_hits: usize,
    /// Ray parameter `t` of the nearest hit, or `None` if there were no hits.
    pub nearest_t: Option<f32>,
    /// Mean perpendicular distance to center across all hits.
    pub mean_distance_to_center: f32,
}

/// Compute [`PickStats`] from a set of Gaussians and the hits returned by a
/// picking operation.
pub fn compute_pick_stats(gaussians: &[GaussianPickData], hits: &[PickHit]) -> PickStats {
    let total_gaussians = gaussians.len();

    // Count Gaussians that would be skipped at default min_opacity = 0.1.
    // We report unconditionally how many have opacity < 0.1.
    let skipped_opacity = gaussians
        .iter()
        .filter(|g| g.opacity < PickConfig::default().min_opacity)
        .count();

    let num_hits = hits.len();

    let nearest_t = hits.iter().map(|h| h.t).reduce(f32::min);

    let mean_distance_to_center = if num_hits == 0 {
        0.0
    } else {
        let sum: f32 = hits.iter().map(|h| h.distance_to_center).sum();
        sum / num_hits as f32
    };

    PickStats {
        total_gaussians,
        skipped_opacity,
        num_hits,
        nearest_t,
        mean_distance_to_center,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn identity_rotation() -> [f32; 9] {
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    }

    /// Camera at origin, identity rotation, 100×100, 90° FoV (looking down −Z).
    fn forward_camera() -> PickCamera {
        PickCamera::new(
            [0.0, 0.0, 0.0],
            identity_rotation(),
            100,
            100,
            std::f32::consts::FRAC_PI_2,
        )
        .unwrap()
    }

    /// A Gaussian centered at `center` on the −Z axis with radius 0.5.
    fn gaussian_on_axis(z: f32) -> GaussianPickData {
        GaussianPickData {
            center: [0.0, 0.0, z],
            radius: 0.5,
            opacity: 0.9,
        }
    }

    // ── Test 1: Ray::new normalises a valid direction ─────────────────────────
    #[test]
    fn test_ray_new_valid() {
        let ray = Ray::new([0.0, 0.0, 0.0], [3.0, 0.0, 0.0]).unwrap();
        let len = ray.direction_len();
        assert!(
            (len - 1.0).abs() < 1e-6,
            "direction should be unit-length, got {len}"
        );
        assert_eq!(ray.direction, [1.0, 0.0, 0.0]);
    }

    // ── Test 2: Ray::new rejects a zero direction ─────────────────────────────
    #[test]
    fn test_ray_new_zero_direction() {
        let result = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(result.is_err(), "zero direction should return Err");
    }

    // ── Test 3: Ray::at returns correct point along the ray ───────────────────
    #[test]
    fn test_ray_at() {
        let ray = Ray::new([1.0, 2.0, 3.0], [0.0, 0.0, 1.0]).unwrap();
        let p = ray.at(5.0);
        assert!((p[0] - 1.0).abs() < 1e-6);
        assert!((p[1] - 2.0).abs() < 1e-6);
        assert!((p[2] - 8.0).abs() < 1e-6);
    }

    // ── Test 4: ray_point_distance – point on ray has distance 0 ─────────────
    #[test]
    fn test_ray_point_distance_on_ray() {
        let ray = Ray::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap();
        let (t, dist) = ray_point_distance(&ray, [5.0, 0.0, 0.0]);
        assert!((t - 5.0).abs() < 1e-5, "t should be 5, got {t}");
        assert!(dist < 1e-5, "distance should be ~0, got {dist}");
    }

    // ── Test 5: ray_point_distance – perpendicular point ─────────────────────
    #[test]
    fn test_ray_point_distance_perpendicular() {
        // Ray along +X; point at (3, 4, 0) → closest = (3, 0, 0), dist = 4.
        let ray = Ray::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap();
        let (t, dist) = ray_point_distance(&ray, [3.0, 4.0, 0.0]);
        assert!((t - 3.0).abs() < 1e-5, "t should be 3, got {t}");
        assert!(
            (dist - 4.0).abs() < 1e-5,
            "distance should be 4, got {dist}"
        );
    }

    // ── Test 6: ray_sphere_intersect – ray hits sphere ────────────────────────
    #[test]
    fn test_ray_sphere_intersect_hit() {
        let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]).unwrap();
        // Sphere at z=−5 with radius 1.
        let t = ray_sphere_intersect(&ray, [0.0, 0.0, -5.0], 1.0);
        assert!(t.is_some(), "should hit sphere");
        let t = t.unwrap();
        assert!(t > 0.0, "t should be positive, got {t}");
        // Entry point should be at z = −4 (front of sphere).
        assert!((t - 4.0).abs() < 1e-4, "expected t≈4, got {t}");
    }

    // ── Test 7: ray_sphere_intersect – ray misses sphere ─────────────────────
    #[test]
    fn test_ray_sphere_intersect_miss() {
        let ray = Ray::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap();
        // Sphere at (0, 10, 0) with radius 1 — entirely off to the side.
        let t = ray_sphere_intersect(&ray, [0.0, 10.0, 0.0], 1.0);
        assert!(t.is_none(), "should miss sphere");
    }

    // ── Test 8: ray_sphere_intersect – ray starts inside sphere ──────────────
    #[test]
    fn test_ray_sphere_intersect_inside() {
        let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]).unwrap();
        // Large sphere centred at origin with radius 10 — we start inside.
        let t = ray_sphere_intersect(&ray, [0.0, 0.0, 0.0], 10.0);
        assert!(t.is_some(), "should return exit point");
        let t = t.unwrap();
        assert!(t > 0.0, "exit t should be positive, got {t}");
        assert!((t - 10.0).abs() < 1e-4, "expected t≈10, got {t}");
    }

    // ── Test 9: ray_sphere_intersect – sphere fully behind origin ─────────────
    #[test]
    fn test_ray_sphere_intersect_behind() {
        let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]).unwrap();
        // Sphere at +Z (behind the ray looking down −Z).
        let t = ray_sphere_intersect(&ray, [0.0, 0.0, 5.0], 1.0);
        assert!(t.is_none(), "sphere is behind origin — should return None");
    }

    // ── Test 10: PickCamera::new succeeds with valid params ───────────────────
    #[test]
    fn test_pick_camera_new_valid() {
        let cam = PickCamera::new(
            [1.0, 2.0, 3.0],
            identity_rotation(),
            800,
            600,
            std::f32::consts::FRAC_PI_4,
        );
        assert!(cam.is_ok(), "valid camera should succeed");
    }

    // ── Test 11: center pixel ray points mostly −Z ────────────────────────────
    #[test]
    fn test_ray_for_center_pixel() {
        let cam = forward_camera();
        let ray = cam.ray_for_pixel(50, 50).unwrap();
        // With identity rotation and center pixel, direction should be ≈ (0, 0, −1).
        assert!(
            ray.direction[2] < -0.9,
            "center pixel should point mostly −Z, got {:?}",
            ray.direction
        );
        // X and Y should be near zero for the center pixel.
        assert!(ray.direction[0].abs() < 0.1);
        assert!(ray.direction[1].abs() < 0.1);
    }

    // ── Test 12: left-edge pixel has negative X component ────────────────────
    #[test]
    fn test_ray_for_left_pixel() {
        let cam = forward_camera();
        // Pixel (0, 50): left edge, middle row → X component should be negative.
        let ray = cam.ray_for_pixel(0, 50).unwrap();
        assert!(
            ray.direction[0] < 0.0,
            "left pixel should have negative X, got {:?}",
            ray.direction
        );
    }

    // ── Test 13: pick_nearest finds a Gaussian directly on the ray ───────────
    #[test]
    fn test_pick_nearest_hit() {
        let cam = forward_camera();
        // Ray shooting straight down −Z.
        let ray = cam.ray_for_pixel(50, 50).unwrap();
        let gaussians = vec![gaussian_on_axis(-5.0)];
        let config = PickConfig::default();
        let hit = pick_nearest(&ray, &gaussians, &config);
        assert!(hit.is_some(), "should hit Gaussian on −Z axis");
        let hit = hit.unwrap();
        assert_eq!(hit.index, 0);
        assert!(hit.t > 0.0);
    }

    // ── Test 14: pick_nearest returns None for empty scene ────────────────────
    #[test]
    fn test_pick_nearest_empty() {
        let cam = forward_camera();
        let ray = cam.ray_for_pixel(50, 50).unwrap();
        let hit = pick_nearest(&ray, &[], &PickConfig::default());
        assert!(hit.is_none());
    }

    // ── Test 15: pick_nearest returns nearer of two Gaussians ─────────────────
    #[test]
    fn test_pick_nearest_two_gaussians() {
        let cam = forward_camera();
        let ray = cam.ray_for_pixel(50, 50).unwrap();
        let gaussians = vec![
            gaussian_on_axis(-10.0), // farther
            gaussian_on_axis(-4.0),  // nearer
        ];
        let hit = pick_nearest(&ray, &gaussians, &PickConfig::default()).unwrap();
        assert_eq!(hit.index, 1, "nearer Gaussian (index 1) should be selected");
    }

    // ── Test 16: pick_nearest skips low-opacity Gaussians ────────────────────
    #[test]
    fn test_pick_nearest_opacity_filter() {
        let cam = forward_camera();
        let ray = cam.ray_for_pixel(50, 50).unwrap();
        let gaussians = vec![GaussianPickData {
            center: [0.0, 0.0, -5.0],
            radius: 0.5,
            opacity: 0.05, // below default min_opacity = 0.1
        }];
        let hit = pick_nearest(&ray, &gaussians, &PickConfig::default());
        assert!(hit.is_none(), "low-opacity Gaussian should be skipped");
    }

    // ── Test 17: pick_all returns all hits sorted by t ────────────────────────
    #[test]
    fn test_pick_all_sorted() {
        let cam = forward_camera();
        let ray = cam.ray_for_pixel(50, 50).unwrap();
        let gaussians = vec![
            gaussian_on_axis(-10.0),
            gaussian_on_axis(-4.0),
            gaussian_on_axis(-7.0),
        ];
        let hits = pick_all(&ray, &gaussians, &PickConfig::default());
        assert_eq!(hits.len(), 3, "all three Gaussians should be hit");
        // Verify ascending t order.
        for w in hits.windows(2) {
            assert!(
                w[0].t <= w[1].t,
                "hits not sorted: t[0]={} > t[1]={}",
                w[0].t,
                w[1].t
            );
        }
        // The nearest should be at z=−4.
        assert_eq!(hits[0].index, 1, "nearest hit should be index 1 (z=−4)");
    }

    // ── max_t enforcement (pick_nearest / pick_all) ───────────────────────────

    #[test]
    fn test_pick_nearest_respects_max_t() {
        let cam = forward_camera();
        let ray = cam.ray_for_pixel(50, 50).unwrap();
        // Far beyond PickConfig::default().max_t == 1000.0.
        let gaussians = vec![gaussian_on_axis(-2000.0)];
        let hit = pick_nearest(&ray, &gaussians, &PickConfig::default());
        assert!(hit.is_none(), "hit beyond max_t should be rejected");
    }

    #[test]
    fn test_pick_nearest_max_t_allows_hit_just_inside() {
        let cam = forward_camera();
        let ray = cam.ray_for_pixel(50, 50).unwrap();
        let config = PickConfig {
            max_t: 10.0,
            ..PickConfig::default()
        };
        let gaussians = vec![gaussian_on_axis(-5.0)];
        let hit = pick_nearest(&ray, &gaussians, &config);
        assert!(hit.is_some(), "hit within max_t should still be found");
    }

    #[test]
    fn test_pick_all_respects_max_t() {
        let cam = forward_camera();
        let ray = cam.ray_for_pixel(50, 50).unwrap();
        let gaussians = vec![gaussian_on_axis(-4.0), gaussian_on_axis(-2000.0)];
        let hits = pick_all(&ray, &gaussians, &PickConfig::default());
        assert_eq!(
            hits.len(),
            1,
            "only the in-range Gaussian should be returned"
        );
        assert_eq!(hits[0].index, 0);
    }

    // ── Test 18: pick_all returns empty vec for empty scene ───────────────────
    #[test]
    fn test_pick_all_empty() {
        let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]).unwrap();
        let hits = pick_all(&ray, &[], &PickConfig::default());
        assert!(hits.is_empty());
    }

    // ── Test 19: pick_closest_approach – nearest by perpendicular dist ────────
    #[test]
    fn test_pick_closest_approach() {
        // Ray along −Z; two Gaussians slightly off-axis.
        let ray = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]).unwrap();
        let gaussians = vec![
            GaussianPickData {
                center: [3.0, 0.0, -5.0], // perpendicular dist = 3
                radius: 0.5,
                opacity: 0.9,
            },
            GaussianPickData {
                center: [1.0, 0.0, -5.0], // perpendicular dist = 1 (closer)
                radius: 0.5,
                opacity: 0.9,
            },
        ];
        let config = PickConfig {
            max_hit_distance: 10.0,
            ..Default::default()
        };
        let hit = pick_closest_approach(&ray, &gaussians, &config).unwrap();
        assert_eq!(
            hit.index, 1,
            "should pick the Gaussian with smaller perpendicular distance"
        );
    }

    // ── Test 20: pick_region with 1×1 region ─────────────────────────────────
    #[test]
    fn test_pick_region_single_pixel() {
        let cam = forward_camera();
        let gaussians = vec![gaussian_on_axis(-5.0)];
        let result = pick_region(&cam, &gaussians, 50, 50, 51, 51, &PickConfig::default()).unwrap();
        assert_eq!(result.num_rays, 1);
        assert_eq!(result.total_tested, 1); // 1 ray × 1 Gaussian
        assert!(!result.hits.is_empty(), "should hit the Gaussian");
    }

    // ── Test 21: compute_pick_stats counts are correct ────────────────────────
    #[test]
    fn test_compute_pick_stats() {
        let gaussians = vec![
            GaussianPickData {
                center: [0.0, 0.0, -5.0],
                radius: 0.5,
                opacity: 0.9,
            },
            GaussianPickData {
                center: [0.0, 0.0, -8.0],
                radius: 0.5,
                opacity: 0.05,
            }, // low opacity
        ];
        let hits = vec![PickHit {
            index: 0,
            t: 4.5,
            closest_point: [0.0, 0.0, -4.5],
            distance_to_center: 0.0,
            opacity: 0.9,
        }];
        let stats = compute_pick_stats(&gaussians, &hits);
        assert_eq!(stats.total_gaussians, 2);
        assert_eq!(stats.skipped_opacity, 1);
        assert_eq!(stats.num_hits, 1);
        assert!(stats.nearest_t.is_some());
        assert!((stats.nearest_t.unwrap() - 4.5).abs() < 1e-6);
    }

    // ── Test 22: PickHit::world_position == closest_point ────────────────────
    #[test]
    fn test_pick_hit_world_position() {
        let hit = PickHit {
            index: 0,
            t: 5.0,
            closest_point: [1.0, 2.0, 3.0],
            distance_to_center: 0.1,
            opacity: 0.8,
        };
        assert_eq!(hit.world_position(), hit.closest_point);
    }
}
