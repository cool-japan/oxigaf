//! Gaussian culling: frustum, distance, and screen-size culling for 3D Gaussian Splatting.
//!
//! Provides:
//! - `FrustumPlane` / `ViewFrustum`: perspective frustum built from camera intrinsics or a
//!   projection matrix via the Gribb-Hartmann method.
//! - `GaussianCullData`: compact per-Gaussian bounding data (world-space center + radius).
//! - `CullingConfig` / `CullingResult`: configuration and per-frame output.
//! - `cull_gaussians`: the main culling pass combining frustum, distance, and size tests.
//! - Screen-space projection helpers (`compute_screen_bounds`, `project_radius_ndc`).
//! - `CullStats`: depth/radius statistics over the visible set.

use thiserror::Error;

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors produced by culling operations.
#[derive(Debug, Error)]
pub enum CullingError {
    /// The frustum parameters are geometrically invalid (e.g., near >= far, fov <= 0).
    #[error("Invalid frustum: {0}")]
    InvalidFrustum(String),

    /// Zero Gaussians were provided (reserved; not raised by `cull_gaussians` itself).
    #[error("Empty input: no Gaussians provided")]
    EmptyInput,

    /// Camera parameters are invalid.
    #[error("Invalid camera: {0}")]
    InvalidCamera(String),
}

// ─── FrustumPlane ────────────────────────────────────────────────────────────

/// A half-space plane defined by a normalised outward normal and a signed offset.
///
/// Convention: a point P is on the *inside* (positive) side when
/// `dot(normal, P) + d > 0`.  All six planes of a `ViewFrustum` use this
/// convention, so a point is inside the frustum when all six tests pass.
#[derive(Debug, Clone, Copy)]
pub struct FrustumPlane {
    /// Normalised plane normal (magnitude 1.0).
    pub normal: [f32; 3],
    /// Signed offset: the plane equation is `dot(normal, P) + d = 0`.
    pub d: f32,
}

impl FrustumPlane {
    /// Construct a plane.  The caller is responsible for passing a unit-length normal.
    pub fn new(normal: [f32; 3], d: f32) -> Self {
        Self { normal, d }
    }

    /// Signed distance from the plane to point `p`.
    ///
    /// Positive → inside (same side as the normal), negative → outside.
    #[inline]
    pub fn signed_distance(&self, point: [f32; 3]) -> f32 {
        self.normal[0] * point[0] + self.normal[1] * point[1] + self.normal[2] * point[2] + self.d
    }
}

// ─── ViewFrustum ─────────────────────────────────────────────────────────────

/// Six half-space planes that together define the visible view frustum.
///
/// Plane order: [left, right, bottom, top, near, far].
/// A point/sphere/AABB is inside the frustum when it passes *all* six tests.
#[derive(Debug, Clone)]
pub struct ViewFrustum {
    /// [left, right, bottom, top, near, far].
    pub planes: [FrustumPlane; 6],
}

impl ViewFrustum {
    /// Build a frustum from pinhole camera intrinsics using the Gribb-Hartmann method.
    ///
    /// An OpenGL-style perspective projection matrix is constructed internally and
    /// then passed to `from_projection_matrix`.
    ///
    /// # Parameters
    /// - `fov_y_rad`: vertical field-of-view in radians (must be > 0 and < π).
    /// - `aspect`: viewport width / height (must be > 0).
    /// - `near`: near clip distance (must be > 0).
    /// - `far`: far clip distance (must be > near).
    pub fn from_perspective(
        fov_y_rad: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Result<Self, CullingError> {
        if fov_y_rad <= 0.0 || fov_y_rad >= std::f32::consts::PI {
            return Err(CullingError::InvalidFrustum(format!(
                "fov_y_rad must be in (0, π), got {}",
                fov_y_rad
            )));
        }
        if aspect <= 0.0 {
            return Err(CullingError::InvalidFrustum(format!(
                "aspect must be > 0, got {}",
                aspect
            )));
        }
        if near <= 0.0 {
            return Err(CullingError::InvalidFrustum(format!(
                "near must be > 0, got {}",
                near
            )));
        }
        if far <= near {
            return Err(CullingError::InvalidFrustum(format!(
                "far ({}) must be > near ({})",
                far, near
            )));
        }

        // Build an OpenGL-style perspective projection matrix (row-major, column vectors).
        //
        //  [f/a   0        0               0     ]
        //  [ 0    f        0               0     ]
        //  [ 0    0  (f+n)/(n-f)   2fn/(n-f)    ]
        //  [ 0    0       -1               0     ]
        //
        // where f = 1 / tan(fov_y / 2), a = aspect.
        let f = 1.0 / (fov_y_rad * 0.5).tan();
        let nf = near - far; // negative

        #[rustfmt::skip]
        let mat: [f32; 16] = [
            f / aspect, 0.0,  0.0,                     0.0,
            0.0,        f,    0.0,                     0.0,
            0.0,        0.0,  (far + near) / nf,       2.0 * far * near / nf,
            0.0,        0.0, -1.0,                     0.0,
        ];

        Self::from_projection_matrix(&mat)
    }

    /// Build a frustum from a row-major 4×4 projection matrix using the Gribb-Hartmann method.
    ///
    /// Plane extraction (rows indexed 0..3, each row has 4 elements):
    /// - Left:   row3 + row0
    /// - Right:  row3 − row0
    /// - Bottom: row3 + row1
    /// - Top:    row3 − row1
    /// - Near:   row3 + row2
    /// - Far:    row3 − row2
    ///
    /// Each raw plane `(a, b, c, d)` is normalised by dividing by `sqrt(a²+b²+c²)`.
    pub fn from_projection_matrix(mat: &[f32; 16]) -> Result<Self, CullingError> {
        // Helper: extract a row from the flat row-major array.
        let row =
            |i: usize| -> [f32; 4] { [mat[i * 4], mat[i * 4 + 1], mat[i * 4 + 2], mat[i * 4 + 3]] };

        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);

        // Raw plane coefficients: (a, b, c, d) where ax+by+cz+d >= 0 is inside.
        let raw_planes: [[f32; 4]; 6] = [
            // left
            [r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]],
            // right
            [r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]],
            // bottom
            [r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]],
            // top
            [r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]],
            // near
            [r3[0] + r2[0], r3[1] + r2[1], r3[2] + r2[2], r3[3] + r2[3]],
            // far
            [r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]],
        ];

        let mut planes = [FrustumPlane::new([0.0; 3], 0.0); 6];
        for (i, rp) in raw_planes.iter().enumerate() {
            let len = (rp[0] * rp[0] + rp[1] * rp[1] + rp[2] * rp[2]).sqrt();
            if len < 1e-10 {
                return Err(CullingError::InvalidFrustum(format!(
                    "Degenerate plane {} (zero normal length) in projection matrix",
                    i
                )));
            }
            planes[i] = FrustumPlane::new([rp[0] / len, rp[1] / len, rp[2] / len], rp[3] / len);
        }

        Ok(Self { planes })
    }

    /// Test whether a sphere (center, radius) intersects or is inside the frustum.
    ///
    /// Returns `false` only when the sphere is *fully outside* at least one plane.
    /// (Conservative: returns `true` for partial overlaps and full containment.)
    #[inline]
    pub fn intersects_sphere(&self, center: [f32; 3], radius: f32) -> bool {
        for plane in &self.planes {
            if plane.signed_distance(center) < -radius {
                return false;
            }
        }
        true
    }

    /// Test whether an AABB `[min, max]` intersects or is inside the frustum.
    ///
    /// Uses the p-vertex method: for each plane, the corner of the AABB most
    /// aligned with the plane normal is tested.  If that corner is outside, the
    /// whole box is outside that plane.
    #[inline]
    pub fn intersects_aabb(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        for plane in &self.planes {
            // The p-vertex is the corner maximally inside (most positive) for this plane.
            let px = if plane.normal[0] >= 0.0 {
                max[0]
            } else {
                min[0]
            };
            let py = if plane.normal[1] >= 0.0 {
                max[1]
            } else {
                min[1]
            };
            let pz = if plane.normal[2] >= 0.0 {
                max[2]
            } else {
                min[2]
            };

            if plane.signed_distance([px, py, pz]) < 0.0 {
                return false;
            }
        }
        true
    }
}

// ─── GaussianCullData ────────────────────────────────────────────────────────

/// Compact per-Gaussian data required for culling decisions (world space).
#[derive(Debug, Clone, Copy)]
pub struct GaussianCullData {
    /// World-space centre of the Gaussian.
    pub center: [f32; 3],
    /// Bounding sphere radius in world space (e.g. 3 × max scale).
    pub radius: f32,
}

// ─── CullingConfig ───────────────────────────────────────────────────────────

/// Configuration controlling which culling passes are active and their thresholds.
#[derive(Debug, Clone)]
pub struct CullingConfig {
    /// Multiply each Gaussian's bounding-sphere radius by this factor before culling.
    /// Values > 1 add a safety margin; values < 1 are more aggressive.
    pub radius_scale: f32,
    /// Enable view-frustum culling (requires a `ViewFrustum` to be passed).
    pub frustum_cull: bool,
    /// Enable distance-based culling (removes Gaussians farther than `max_distance`).
    pub distance_cull: bool,
    /// Maximum camera-space depth for distance culling.
    pub max_distance: f32,
    /// Minimum screen-space radius (NDC units) to render.  Gaussians projecting
    /// smaller than this are culled.  Set to 0.0 to disable.
    pub min_screen_size: f32,
}

impl Default for CullingConfig {
    fn default() -> Self {
        Self {
            radius_scale: 1.0,
            frustum_cull: true,
            distance_cull: false,
            max_distance: 100.0,
            min_screen_size: 0.0,
        }
    }
}

// ─── CullingResult ───────────────────────────────────────────────────────────

/// Per-frame culling output.
#[derive(Debug, Clone)]
pub struct CullingResult {
    /// Per-Gaussian visibility flags (same order as the input slice).
    pub visible: Vec<bool>,
    /// Indices (into the input slice) of all visible Gaussians.
    pub visible_indices: Vec<usize>,
    /// Number of Gaussians removed by frustum culling.
    pub frustum_culled: usize,
    /// Number of Gaussians removed by distance culling.
    pub distance_culled: usize,
    /// Number of Gaussians removed by screen-size culling.
    pub size_culled: usize,
    /// Total Gaussians in the input.
    pub total: usize,
    /// Number of visible Gaussians.
    pub num_visible: usize,
}

impl CullingResult {
    /// Fraction of Gaussians that survived culling (0.0 … 1.0).
    /// Returns 0.0 for empty inputs.
    pub fn visibility_ratio(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.num_visible as f32 / self.total as f32
        }
    }

    /// Human-readable one-line summary.
    pub fn format_summary(&self) -> String {
        format!(
            "visible={}/{} ({:.1}%) | frustum_culled={} distance_culled={} size_culled={}",
            self.num_visible,
            self.total,
            self.visibility_ratio() * 100.0,
            self.frustum_culled,
            self.distance_culled,
            self.size_culled,
        )
    }
}

// ─── Free functions ──────────────────────────────────────────────────────────

/// Transform a world-space point `p` by a row-major 4×4 matrix `mat`, returning
/// the resulting 3D point (the homogeneous `w` component is divided out).
///
/// The matrix is treated as transforming column vectors: `p' = M * p` where
/// the last column is the translation.
pub fn transform_point(mat: &[f32; 16], p: [f32; 3]) -> [f32; 3] {
    let x = mat[0] * p[0] + mat[1] * p[1] + mat[2] * p[2] + mat[3];
    let y = mat[4] * p[0] + mat[5] * p[1] + mat[6] * p[2] + mat[7];
    let z = mat[8] * p[0] + mat[9] * p[1] + mat[10] * p[2] + mat[11];
    let w = mat[12] * p[0] + mat[13] * p[1] + mat[14] * p[2] + mat[15];

    if w.abs() < 1e-10 {
        // Degenerate case — return as-is to avoid division by zero.
        [x, y, z]
    } else {
        [x / w, y / w, z / w]
    }
}

/// Project a world-space radius to an NDC screen-space radius.
///
/// `focal_x = fx / width` and `focal_y = fy / height` are the NDC focal lengths.
/// `depth` is the absolute camera-space Z depth of the Gaussian centre.
///
/// Returns the larger of the two projected radii (x vs y).
#[inline]
pub fn project_radius_ndc(radius: f32, depth: f32, focal_x: f32, focal_y: f32) -> f32 {
    let abs_depth = depth.abs();
    if abs_depth < 1e-10 {
        return f32::INFINITY;
    }
    let rx = radius * focal_x / abs_depth;
    let ry = radius * focal_y / abs_depth;
    rx.max(ry)
}

/// Apply culling to a list of Gaussians and return a `CullingResult`.
///
/// # Parameters
/// - `gaussians`: world-space centre + radius for every Gaussian.
/// - `view_matrix`: world → camera transform (row-major 4×4).  The camera sits
///   at the origin and looks down **−Z** in view space.
/// - `frustum`: optional view frustum.  *Required* when `config.frustum_cull` is `true`.
/// - `focal_x`, `focal_y`: NDC focal lengths (f/W and f/H respectively).
/// - `config`: culling configuration.
///
/// # Errors
/// Returns no error for empty input — an empty `CullingResult` is returned instead.
pub fn cull_gaussians(
    gaussians: &[GaussianCullData],
    view_matrix: &[f32; 16],
    frustum: Option<&ViewFrustum>,
    focal_x: f32,
    focal_y: f32,
    config: &CullingConfig,
) -> Result<CullingResult, CullingError> {
    let total = gaussians.len();

    // Fast path for empty input — return an empty result rather than an error.
    if total == 0 {
        return Ok(CullingResult {
            visible: Vec::new(),
            visible_indices: Vec::new(),
            frustum_culled: 0,
            distance_culled: 0,
            size_culled: 0,
            total: 0,
            num_visible: 0,
        });
    }

    let mut visible = vec![true; total];
    let mut frustum_culled = 0usize;
    let mut distance_culled = 0usize;
    let mut size_culled = 0usize;

    for (idx, g) in gaussians.iter().enumerate() {
        // Transform centre to view (camera) space.
        let view_center = transform_point(view_matrix, g.center);

        // Camera-space depth is |Z|.
        let depth = view_center[2].abs();

        // --- Distance culling ---
        if config.distance_cull && depth > config.max_distance {
            visible[idx] = false;
            distance_culled += 1;
            continue;
        }

        // --- Frustum culling ---
        if config.frustum_cull {
            if let Some(frust) = frustum {
                let scaled_radius = g.radius * config.radius_scale;
                if !frust.intersects_sphere(view_center, scaled_radius) {
                    visible[idx] = false;
                    frustum_culled += 1;
                    continue;
                }
            }
        }

        // --- Screen-size culling ---
        if config.min_screen_size > 0.0 {
            let screen_r =
                project_radius_ndc(g.radius * config.radius_scale, depth, focal_x, focal_y);
            if screen_r < config.min_screen_size {
                visible[idx] = false;
                size_culled += 1;
                continue;
            }
        }
    }

    let visible_indices: Vec<usize> = (0..total).filter(|&i| visible[i]).collect();
    let num_visible = visible_indices.len();

    Ok(CullingResult {
        visible,
        visible_indices,
        frustum_culled,
        distance_culled,
        size_culled,
        total,
        num_visible,
    })
}

// ─── ScreenSpaceBounds ───────────────────────────────────────────────────────

/// Axis-aligned bounding rectangle in NDC space for a single Gaussian.
#[derive(Debug, Clone, Copy)]
pub struct ScreenSpaceBounds {
    /// NDC-space projected centre of the Gaussian.
    pub center_ndc: [f32; 2],
    /// Conservative NDC-space bounding radius.
    pub radius_ndc: f32,
    /// Lower-left corner of the bounding rectangle (center_ndc - radius_ndc).
    pub min: [f32; 2],
    /// Upper-right corner of the bounding rectangle (center_ndc + radius_ndc).
    pub max: [f32; 2],
    /// Camera-space depth (positive, −Z coordinate in view space).
    pub depth: f32,
}

/// Project a Gaussian (already in view space) to a screen-space bounding rectangle.
///
/// Returns `None` when the Gaussian is *behind* the camera (`depth <= 0`).
///
/// # Parameters
/// - `view_center`: Gaussian centre in view/camera space.
/// - `world_radius`: world-space bounding radius.
/// - `focal_x`, `focal_y`: NDC focal lengths.
pub fn compute_screen_bounds(
    view_center: [f32; 3],
    world_radius: f32,
    focal_x: f32,
    focal_y: f32,
) -> Option<ScreenSpaceBounds> {
    // In view space the camera looks down −Z; depth is −Z.
    let depth = -view_center[2];
    if depth <= 0.0 {
        return None;
    }

    // Pin-hole projection: NDC = (view_x * focal_x) / depth
    let cx = view_center[0] * focal_x / depth;
    let cy = view_center[1] * focal_y / depth;

    // Conservative screen-space radius (use the larger of the two axes).
    let r = project_radius_ndc(world_radius, depth, focal_x, focal_y);

    Some(ScreenSpaceBounds {
        center_ndc: [cx, cy],
        radius_ndc: r,
        min: [cx - r, cy - r],
        max: [cx + r, cy + r],
        depth,
    })
}

// ─── CullStats ───────────────────────────────────────────────────────────────

/// Aggregate depth and radius statistics over a culled Gaussian set.
#[derive(Debug, Clone)]
pub struct CullStats {
    /// Mean camera-space depth of all Gaussians (visible and culled).
    pub mean_depth: f32,
    /// Maximum camera-space depth.
    pub max_depth: f32,
    /// Minimum camera-space depth.
    pub min_depth: f32,
    /// Mean world-space bounding radius of all Gaussians.
    pub mean_radius: f32,
    /// Fraction of Gaussians that were visible (same as `CullingResult::visibility_ratio`).
    pub visibility_ratio: f32,
}

/// Compute `CullStats` from the full Gaussian list and its culling result.
///
/// Depth is approximated as `|center[2]|` in *world* space (no view transform
/// is applied here — a view matrix is not available).  For accurate depth
/// statistics call this after the view-transform step in your pipeline.
pub fn compute_cull_stats(gaussians: &[GaussianCullData], result: &CullingResult) -> CullStats {
    if gaussians.is_empty() {
        return CullStats {
            mean_depth: 0.0,
            max_depth: 0.0,
            min_depth: 0.0,
            mean_radius: 0.0,
            visibility_ratio: 0.0,
        };
    }

    let n = gaussians.len() as f32;

    // Use |center[2]| as a proxy for depth when no view matrix is available.
    let depths: Vec<f32> = gaussians.iter().map(|g| g.center[2].abs()).collect();
    let radii: Vec<f32> = gaussians.iter().map(|g| g.radius).collect();

    let mean_depth = depths.iter().sum::<f32>() / n;
    let max_depth = depths.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_depth = depths.iter().cloned().fold(f32::INFINITY, f32::min);
    let mean_radius = radii.iter().sum::<f32>() / n;

    CullStats {
        mean_depth,
        max_depth,
        min_depth,
        mean_radius,
        visibility_ratio: result.visibility_ratio(),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──

    fn identity() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]
    }

    /// Row-major 4×4 pure translation matrix.
    fn translation(tx: f32, ty: f32, tz: f32) -> [f32; 16] {
        [
            1.0, 0.0, 0.0, tx, 0.0, 1.0, 0.0, ty, 0.0, 0.0, 1.0, tz, 0.0, 0.0, 0.0, 1.0,
        ]
    }

    /// A simple perspective frustum (60° FoV, 1:1 aspect, near=0.1, far=100).
    fn test_frustum() -> ViewFrustum {
        ViewFrustum::from_perspective(60_f32.to_radians(), 1.0, 0.1, 100.0)
            .expect("valid perspective parameters")
    }

    // ── 1. FrustumPlane signed_distance: point on positive side ──────────────

    #[test]
    fn test_frustum_plane_positive_side() {
        let plane = FrustumPlane::new([0.0, 0.0, 1.0], 0.0);
        // Point at z=1 is on the positive side.
        assert!(plane.signed_distance([0.0, 0.0, 1.0]) > 0.0);
    }

    // ── 2. FrustumPlane signed_distance: point on negative side ──────────────

    #[test]
    fn test_frustum_plane_negative_side() {
        let plane = FrustumPlane::new([0.0, 0.0, 1.0], 0.0);
        // Point at z=-1 is on the negative side.
        assert!(plane.signed_distance([0.0, 0.0, -1.0]) < 0.0);
    }

    // ── 3. ViewFrustum::from_perspective: valid params ────────────────────────

    #[test]
    fn test_from_perspective_valid() {
        let result = ViewFrustum::from_perspective(60_f32.to_radians(), 1.6, 0.1, 1000.0);
        assert!(result.is_ok(), "Expected Ok for valid parameters");
    }

    // ── 4. ViewFrustum::from_perspective: fov=0 returns Err ─────────────────

    #[test]
    fn test_from_perspective_zero_fov() {
        let result = ViewFrustum::from_perspective(0.0, 1.0, 0.1, 100.0);
        assert!(
            matches!(result, Err(CullingError::InvalidFrustum(_))),
            "Expected InvalidFrustum for fov=0"
        );
    }

    // ── 5. ViewFrustum::from_perspective: near >= far returns Err ────────────

    #[test]
    fn test_from_perspective_near_ge_far() {
        let result = ViewFrustum::from_perspective(60_f32.to_radians(), 1.0, 10.0, 5.0);
        assert!(
            matches!(result, Err(CullingError::InvalidFrustum(_))),
            "Expected InvalidFrustum for near >= far"
        );
    }

    // ── 6. intersects_sphere: sphere clearly inside → true ───────────────────

    #[test]
    fn test_sphere_inside_frustum() {
        let frustum = test_frustum();
        // Gribb-Hartmann on a projection-only matrix gives planes in VIEW space.
        // Camera looks down -Z; view-space z=-5 is well inside [near=-0.1, far=-100].
        assert!(frustum.intersects_sphere([0.0, 0.0, -5.0], 0.1));
    }

    // ── 7. intersects_sphere: sphere behind near plane → false ───────────────

    #[test]
    fn test_sphere_behind_near() {
        let frustum = test_frustum();
        // In OpenGL view space the camera looks down -Z.  A positive Z value is
        // *behind* the camera (outside the near plane on the wrong side).
        // View-space z=+5 is well behind the camera; the sphere must be culled.
        assert!(!frustum.intersects_sphere([0.0, 0.0, 5.0], 0.01));
    }

    // ── 8. intersects_sphere: sphere partially inside → true (conservative) ──

    #[test]
    fn test_sphere_partially_inside() {
        let frustum = test_frustum();
        // A sphere at (0,0,0) with a large radius straddles many planes.
        // The intersects_sphere test is conservative, so this should be true.
        assert!(frustum.intersects_sphere([0.0, 0.0, 0.0], 5.0));
    }

    // ── 9. intersects_aabb: box inside → true ────────────────────────────────

    #[test]
    fn test_aabb_inside() {
        let frustum = test_frustum();
        // Box in view space at z=[-5, -2], well inside near=0.1 and far=100.
        // x and y extents are narrow, well within the frustum cone.
        assert!(frustum.intersects_aabb([-0.5, -0.5, -5.0], [0.5, 0.5, -2.0]));
    }

    // ── 10. intersects_aabb: box outside right → false ───────────────────────

    #[test]
    fn test_aabb_outside_right() {
        let frustum = test_frustum();
        // Box far to the right in view space at z=-5.  At z=-5 the frustum half-width
        // is 5 * tan(30°) ≈ 2.89.  An x range of [500, 600] is way outside.
        assert!(!frustum.intersects_aabb([500.0, -0.5, -5.0], [600.0, 0.5, -2.0]));
    }

    // ── 11. transform_point: identity → same point ───────────────────────────

    #[test]
    fn test_transform_point_identity() {
        let p = [1.0_f32, 2.0, 3.0];
        let result = transform_point(&identity(), p);
        assert!((result[0] - p[0]).abs() < 1e-6);
        assert!((result[1] - p[1]).abs() < 1e-6);
        assert!((result[2] - p[2]).abs() < 1e-6);
    }

    // ── 12. transform_point: translation matrix ───────────────────────────────

    #[test]
    fn test_transform_point_translation() {
        let mat = translation(10.0, -5.0, 3.0);
        let result = transform_point(&mat, [1.0, 2.0, 3.0]);
        assert!((result[0] - 11.0).abs() < 1e-5, "x={}", result[0]);
        assert!((result[1] - -3.0).abs() < 1e-5, "y={}", result[1]);
        assert!((result[2] - 6.0).abs() < 1e-5, "z={}", result[2]);
    }

    // ── 13. project_radius_ndc: basic projection ──────────────────────────────

    #[test]
    fn test_project_radius_ndc() {
        // radius=1, depth=10, focal_x=focal_y=1 → screen radius = 0.1
        let r = project_radius_ndc(1.0, 10.0, 1.0, 1.0);
        assert!((r - 0.1).abs() < 1e-6, "r={}", r);
    }

    // ── 14. cull_gaussians: empty input returns empty result (not error) ──────

    #[test]
    fn test_cull_empty_input() {
        let result = cull_gaussians(&[], &identity(), None, 1.0, 1.0, &CullingConfig::default());
        let r = result.expect("empty input should return Ok, not Err");
        assert_eq!(r.total, 0);
        assert_eq!(r.num_visible, 0);
        assert!(r.visible.is_empty());
        assert!(r.visible_indices.is_empty());
    }

    // ── 15. cull_gaussians: all visible with no culling ───────────────────────

    #[test]
    fn test_cull_all_visible() {
        let gaussians = vec![
            GaussianCullData {
                center: [0.0, 0.0, -5.0],
                radius: 0.1,
            },
            GaussianCullData {
                center: [1.0, 0.0, -5.0],
                radius: 0.1,
            },
        ];
        let config = CullingConfig {
            frustum_cull: false,
            distance_cull: false,
            min_screen_size: 0.0,
            ..Default::default()
        };
        let result = cull_gaussians(&gaussians, &identity(), None, 1.0, 1.0, &config)
            .expect("should succeed");
        assert_eq!(result.num_visible, 2);
        assert!(result.visible.iter().all(|&v| v));
    }

    // ── 16. cull_gaussians: distance culling removes far Gaussians ────────────

    #[test]
    fn test_cull_distance() {
        let gaussians = vec![
            GaussianCullData {
                center: [0.0, 0.0, -3.0],
                radius: 0.1,
            }, // depth=3 → keep
            GaussianCullData {
                center: [0.0, 0.0, -200.0],
                radius: 0.1,
            }, // depth=200 → cull
        ];
        let config = CullingConfig {
            frustum_cull: false,
            distance_cull: true,
            max_distance: 50.0,
            min_screen_size: 0.0,
            ..Default::default()
        };
        let result = cull_gaussians(&gaussians, &identity(), None, 1.0, 1.0, &config)
            .expect("should succeed");
        assert_eq!(result.num_visible, 1);
        assert_eq!(result.distance_culled, 1);
        assert!(result.visible[0]);
        assert!(!result.visible[1]);
    }

    // ── 17. cull_gaussians: frustum culling removes out-of-frustum Gaussians ──

    #[test]
    fn test_cull_frustum() {
        // Build a view matrix that moves the camera: identity keeps world=view.
        // Build a standard frustum.
        let frustum = test_frustum();

        // The frustum operates in VIEW space (Gribb-Hartmann on the projection matrix).
        // View-space z=-5 is inside the frustum; x=1000 at z=-5 is way off to the side.
        let gaussians = vec![
            GaussianCullData {
                center: [0.0, 0.0, -5.0],
                radius: 0.05,
            }, // centre of view → visible
            GaussianCullData {
                center: [1000.0, 0.0, -5.0],
                radius: 0.05,
            }, // far right → culled
        ];
        let config = CullingConfig {
            frustum_cull: true,
            distance_cull: false,
            min_screen_size: 0.0,
            radius_scale: 1.0,
            ..Default::default()
        };
        let result = cull_gaussians(&gaussians, &identity(), Some(&frustum), 1.0, 1.0, &config)
            .expect("should succeed");
        // The far-off Gaussian should have been frustum-culled.
        assert_eq!(
            result.frustum_culled, 1,
            "Expected 1 frustum-culled Gaussian"
        );
    }

    // ── 18. cull_gaussians: visible_indices matches visible mask ──────────────

    #[test]
    fn test_cull_visible_indices_consistent() {
        let gaussians = vec![
            GaussianCullData {
                center: [0.0, 0.0, -3.0],
                radius: 0.1,
            },
            GaussianCullData {
                center: [0.0, 0.0, -200.0],
                radius: 0.1,
            },
            GaussianCullData {
                center: [0.0, 0.0, -5.0],
                radius: 0.1,
            },
        ];
        let config = CullingConfig {
            frustum_cull: false,
            distance_cull: true,
            max_distance: 50.0,
            min_screen_size: 0.0,
            ..Default::default()
        };
        let result = cull_gaussians(&gaussians, &identity(), None, 1.0, 1.0, &config)
            .expect("should succeed");

        // Verify visible_indices mirrors the visible mask.
        let expected_indices: Vec<usize> =
            (0..result.total).filter(|&i| result.visible[i]).collect();
        assert_eq!(result.visible_indices, expected_indices);
    }

    // ── 19. CullingResult visibility_ratio: correct fraction ─────────────────

    #[test]
    fn test_visibility_ratio() {
        let gaussians = vec![
            GaussianCullData {
                center: [0.0, 0.0, -5.0],
                radius: 0.1,
            },
            GaussianCullData {
                center: [0.0, 0.0, -200.0],
                radius: 0.1,
            },
        ];
        let config = CullingConfig {
            frustum_cull: false,
            distance_cull: true,
            max_distance: 50.0,
            min_screen_size: 0.0,
            ..Default::default()
        };
        let result = cull_gaussians(&gaussians, &identity(), None, 1.0, 1.0, &config)
            .expect("should succeed");
        let ratio = result.visibility_ratio();
        assert!((ratio - 0.5).abs() < 1e-6, "Expected 0.5, got {}", ratio);
    }

    // ── 20. compute_screen_bounds: behind camera → None ──────────────────────

    #[test]
    fn test_screen_bounds_behind_camera() {
        // In view space camera looks down -Z, so z > 0 is behind the camera.
        let result = compute_screen_bounds([0.0, 0.0, 5.0], 1.0, 1.0, 1.0);
        assert!(result.is_none(), "Expected None for point behind camera");
    }

    // ── 21. compute_screen_bounds: valid → min/max surround center ────────────

    #[test]
    fn test_screen_bounds_valid() {
        // Point at view-space (0, 0, -10), radius=1, focal=1 → depth=10, center=(0,0), r=0.1
        let result = compute_screen_bounds([0.0, 0.0, -10.0], 1.0, 1.0, 1.0);
        let bounds = result.expect("should return Some for point in front of camera");

        assert!(
            (bounds.center_ndc[0] - 0.0).abs() < 1e-6,
            "cx={}",
            bounds.center_ndc[0]
        );
        assert!(
            (bounds.center_ndc[1] - 0.0).abs() < 1e-6,
            "cy={}",
            bounds.center_ndc[1]
        );
        assert!(
            (bounds.radius_ndc - 0.1).abs() < 1e-5,
            "r={}",
            bounds.radius_ndc
        );
        assert!((bounds.min[0] - -0.1).abs() < 1e-5);
        assert!((bounds.max[0] - 0.1).abs() < 1e-5);
        assert!((bounds.depth - 10.0).abs() < 1e-5, "depth={}", bounds.depth);
    }

    // ── 22. compute_cull_stats: basic stats are reasonable ───────────────────

    #[test]
    fn test_cull_stats_basic() {
        let gaussians = vec![
            GaussianCullData {
                center: [0.0, 0.0, 3.0],
                radius: 0.5,
            },
            GaussianCullData {
                center: [0.0, 0.0, 7.0],
                radius: 1.5,
            },
        ];
        let config = CullingConfig {
            frustum_cull: false,
            distance_cull: false,
            min_screen_size: 0.0,
            ..Default::default()
        };
        let result = cull_gaussians(&gaussians, &identity(), None, 1.0, 1.0, &config)
            .expect("should succeed");
        let stats = compute_cull_stats(&gaussians, &result);

        assert!(
            (stats.mean_depth - 5.0).abs() < 1e-5,
            "mean_depth={}",
            stats.mean_depth
        );
        assert!(
            (stats.min_depth - 3.0).abs() < 1e-5,
            "min_depth={}",
            stats.min_depth
        );
        assert!(
            (stats.max_depth - 7.0).abs() < 1e-5,
            "max_depth={}",
            stats.max_depth
        );
        assert!(
            (stats.mean_radius - 1.0).abs() < 1e-5,
            "mean_radius={}",
            stats.mean_radius
        );
        assert!((stats.visibility_ratio - 1.0).abs() < 1e-6);
    }
}
