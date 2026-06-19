//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use thiserror::Error;

use super::functions::{vec3_add, vec3_cross, vec3_norm, vec3_scale, vec3_sub};

/// A ray for volumetric rendering — origin + normalised direction.
///
/// Named `VolumetricRay` to avoid collision with [`crate::picking::Ray`] which is
/// already exported from the crate root.
#[derive(Debug, Clone, Copy)]
pub struct VolumetricRay {
    /// World-space ray origin.
    pub origin: [f32; 3],
    /// World-space direction (unit vector).
    pub direction: [f32; 3],
}
impl VolumetricRay {
    /// Construct, normalising `direction`.  Uses un-normalised direction on
    /// degenerate input (very short vectors) to avoid division by zero.
    pub fn new_normalized(origin: [f32; 3], direction: [f32; 3]) -> Self {
        let len_sq =
            direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2];
        let dir = if len_sq > f32::EPSILON {
            let inv = 1.0 / len_sq.sqrt();
            [direction[0] * inv, direction[1] * inv, direction[2] * inv]
        } else {
            direction
        };
        Self {
            origin,
            direction: dir,
        }
    }
    /// Point on ray at parameter `t`: `origin + t * direction`.
    #[inline]
    pub fn at(&self, t: f32) -> [f32; 3] {
        [
            self.origin[0] + t * self.direction[0],
            self.origin[1] + t * self.direction[1],
            self.origin[2] + t * self.direction[2],
        ]
    }
}
/// Aggregated statistics over a batch of marched rays.
#[derive(Debug, Clone)]
pub struct VolumetricStats {
    /// Number of rays analysed.
    pub n_rays: usize,
    /// Mean number of integration steps per ray.
    pub mean_steps_per_ray: f32,
    /// Maximum steps taken by any single ray.
    pub max_steps_per_ray: usize,
    /// Mean alpha across all rays.
    pub mean_alpha: f32,
    /// Rays that reached `alpha > 0.99`.
    pub fully_opaque_rays: usize,
    /// Rays that never intersected the volume (`t_entry == t_exit == 0`).
    pub empty_rays: usize,
}
/// Pin-hole camera for volumetric rendering.
#[derive(Debug, Clone)]
pub struct VolumetricCamera {
    /// Camera position in world space.
    pub eye: [f32; 3],
    /// Point the camera is looking at.
    pub target: [f32; 3],
    /// Up vector hint (need not be perpendicular to the viewing direction).
    pub up: [f32; 3],
    /// Vertical field of view in radians.
    pub fov_y_rad: f32,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Near clip distance (unused in ray marching, stored for reference).
    pub near: f32,
    /// Far clip distance (unused in ray marching, stored for reference).
    pub far: f32,
}
impl VolumetricCamera {
    /// Generate a ray through pixel centre `(px, py)` (integer pixel coords).
    pub fn generate_ray(&self, px: f32, py: f32) -> VolumetricRay {
        let fwd = vec3_norm(vec3_sub(self.target, self.eye));
        let right = vec3_norm(vec3_cross(fwd, self.up));
        let up = vec3_cross(right, fwd);
        let aspect = self.width as f32 / self.height as f32;
        let half_h = (self.fov_y_rad * 0.5).tan();
        let half_w = aspect * half_h;
        let ndc_x = (px + 0.5) / self.width as f32 * 2.0 - 1.0;
        let ndc_y = 1.0 - (py + 0.5) / self.height as f32 * 2.0;
        let dir = vec3_add(
            vec3_add(fwd, vec3_scale(right, ndc_x * half_w)),
            vec3_scale(up, ndc_y * half_h),
        );
        VolumetricRay::new_normalized(self.eye, dir)
    }
    /// Default front-facing camera looking at the origin from `z = 3`.
    pub fn default_front(width: u32, height: u32) -> Self {
        Self {
            eye: [0.0, 0.0, 3.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_y_rad: std::f32::consts::FRAC_PI_4,
            width,
            height,
            near: 0.01,
            far: 100.0,
        }
    }
}
/// A uniform rectilinear grid of density (scalar) values.
///
/// Data is stored in `[iz * ny * nx + iy * nx + ix]` order (C/row-major,
/// z slowest).
#[derive(Debug, Clone)]
pub struct VolumeGrid {
    /// Flat density array, length `nx * ny * nz`.
    pub data: Vec<f32>,
    /// Number of voxels along X.
    pub nx: usize,
    /// Number of voxels along Y.
    pub ny: usize,
    /// Number of voxels along Z.
    pub nz: usize,
    /// World-space position of the min corner (voxel `[0,0,0]` centre minus
    /// half a voxel).
    pub origin: [f32; 3],
    /// Physical size of each voxel along each axis.
    pub voxel_size: [f32; 3],
}
impl VolumeGrid {
    /// Create an all-zero grid.
    pub fn new(nx: usize, ny: usize, nz: usize, origin: [f32; 3], voxel_size: [f32; 3]) -> Self {
        let n = nx.saturating_mul(ny).saturating_mul(nz);
        Self {
            data: vec![0.0_f32; n],
            nx,
            ny,
            nz,
            origin,
            voxel_size,
        }
    }
    /// Create a grid populated by evaluating `f(x, y, z)` at every voxel
    /// centre (world-space coordinates).
    pub fn from_fn(
        nx: usize,
        ny: usize,
        nz: usize,
        origin: [f32; 3],
        voxel_size: [f32; 3],
        f: impl Fn(f32, f32, f32) -> f32,
    ) -> Self {
        let mut g = Self::new(nx, ny, nz, origin, voxel_size);
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let wx = origin[0] + (ix as f32 + 0.5) * voxel_size[0];
                    let wy = origin[1] + (iy as f32 + 0.5) * voxel_size[1];
                    let wz = origin[2] + (iz as f32 + 0.5) * voxel_size[2];
                    g.data[iz * ny * nx + iy * nx + ix] = f(wx, wy, wz);
                }
            }
        }
        g
    }
    /// Convert a world-space point to continuous voxel coordinates.
    ///
    /// The returned value is centred so that `(0.5, 0.5, 0.5)` corresponds to
    /// the centre of voxel `[0, 0, 0]`.
    #[inline]
    pub fn world_to_voxel(&self, p: [f32; 3]) -> [f32; 3] {
        [
            (p[0] - self.origin[0]) / self.voxel_size[0] - 0.5,
            (p[1] - self.origin[1]) / self.voxel_size[1] - 0.5,
            (p[2] - self.origin[2]) / self.voxel_size[2] - 0.5,
        ]
    }
    /// Convert continuous voxel coordinates back to world space.
    #[inline]
    pub fn voxel_to_world(&self, vi: [f32; 3]) -> [f32; 3] {
        [
            (vi[0] + 0.5) * self.voxel_size[0] + self.origin[0],
            (vi[1] + 0.5) * self.voxel_size[1] + self.origin[1],
            (vi[2] + 0.5) * self.voxel_size[2] + self.origin[2],
        ]
    }
    /// Returns `true` if world-space point `p` is inside the grid AABB.
    #[inline]
    pub fn in_bounds(&self, p: [f32; 3]) -> bool {
        let max = [
            self.origin[0] + self.nx as f32 * self.voxel_size[0],
            self.origin[1] + self.ny as f32 * self.voxel_size[1],
            self.origin[2] + self.nz as f32 * self.voxel_size[2],
        ];
        p[0] >= self.origin[0]
            && p[1] >= self.origin[1]
            && p[2] >= self.origin[2]
            && p[0] <= max[0]
            && p[1] <= max[1]
            && p[2] <= max[2]
    }
    /// Direct integer-indexed density lookup; clamps indices to valid range.
    #[inline]
    pub fn density_at(&self, ix: usize, iy: usize, iz: usize) -> f32 {
        let ix = ix.min(self.nx.saturating_sub(1));
        let iy = iy.min(self.ny.saturating_sub(1));
        let iz = iz.min(self.nz.saturating_sub(1));
        self.data[iz * self.ny * self.nx + iy * self.nx + ix]
    }
    /// Nearest-neighbour sample at world-space position.
    pub fn sample_nearest(&self, x: f32, y: f32, z: f32) -> f32 {
        let vx = ((x - self.origin[0]) / self.voxel_size[0] - 0.5).round();
        let vy = ((y - self.origin[1]) / self.voxel_size[1] - 0.5).round();
        let vz = ((z - self.origin[2]) / self.voxel_size[2] - 0.5).round();
        let ix = vx.clamp(0.0, (self.nx as f32) - 1.0) as usize;
        let iy = vy.clamp(0.0, (self.ny as f32) - 1.0) as usize;
        let iz = vz.clamp(0.0, (self.nz as f32) - 1.0) as usize;
        self.density_at(ix, iy, iz)
    }
    /// Trilinear interpolation at world-space position, clamping at boundaries.
    pub fn sample_trilinear(&self, x: f32, y: f32, z: f32) -> f32 {
        let vx = (x - self.origin[0]) / self.voxel_size[0] - 0.5;
        let vy = (y - self.origin[1]) / self.voxel_size[1] - 0.5;
        let vz = (z - self.origin[2]) / self.voxel_size[2] - 0.5;
        let x0 = vx.floor() as i64;
        let y0 = vy.floor() as i64;
        let z0 = vz.floor() as i64;
        let fx = vx - vx.floor();
        let fy = vy - vy.floor();
        let fz = vz - vz.floor();
        let nx = self.nx as i64;
        let ny = self.ny as i64;
        let nz = self.nz as i64;
        let ci = |v: i64, max: i64| v.clamp(0, max - 1) as usize;
        let c000 = self.density_at(ci(x0, nx), ci(y0, ny), ci(z0, nz));
        let c100 = self.density_at(ci(x0 + 1, nx), ci(y0, ny), ci(z0, nz));
        let c010 = self.density_at(ci(x0, nx), ci(y0 + 1, ny), ci(z0, nz));
        let c110 = self.density_at(ci(x0 + 1, nx), ci(y0 + 1, ny), ci(z0, nz));
        let c001 = self.density_at(ci(x0, nx), ci(y0, ny), ci(z0 + 1, nz));
        let c101 = self.density_at(ci(x0 + 1, nx), ci(y0, ny), ci(z0 + 1, nz));
        let c011 = self.density_at(ci(x0, nx), ci(y0 + 1, ny), ci(z0 + 1, nz));
        let c111 = self.density_at(ci(x0 + 1, nx), ci(y0 + 1, ny), ci(z0 + 1, nz));
        let c00 = c000 * (1.0 - fx) + c100 * fx;
        let c10 = c010 * (1.0 - fx) + c110 * fx;
        let c01 = c001 * (1.0 - fx) + c101 * fx;
        let c11 = c011 * (1.0 - fx) + c111 * fx;
        let c0 = c00 * (1.0 - fy) + c10 * fy;
        let c1 = c01 * (1.0 - fy) + c11 * fy;
        c0 * (1.0 - fz) + c1 * fz
    }
    /// Central-difference gradient at world-space position (in density/unit).
    pub fn gradient(&self, x: f32, y: f32, z: f32) -> [f32; 3] {
        let h = [self.voxel_size[0], self.voxel_size[1], self.voxel_size[2]];
        let dx = (self.sample_trilinear(x + h[0], y, z) - self.sample_trilinear(x - h[0], y, z))
            / (2.0 * h[0]);
        let dy = (self.sample_trilinear(x, y + h[1], z) - self.sample_trilinear(x, y - h[1], z))
            / (2.0 * h[1]);
        let dz = (self.sample_trilinear(x, y, z + h[2]) - self.sample_trilinear(x, y, z - h[2]))
            / (2.0 * h[2]);
        [dx, dy, dz]
    }
}
/// Volumetric integration algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumetricIntegration {
    /// Front-to-back alpha compositing with early termination.
    FrontToBack,
    /// Back-to-front (Porter-Duff) compositing (no early termination).
    BackToFront,
    /// Maximum Intensity Projection — track peak density only.
    Mip,
    /// Average Intensity Projection — accumulate all samples equally.
    Avg,
}
/// Piecewise-linear transfer function mapping density → (colour, opacity).
#[derive(Debug, Clone)]
pub struct TransferFunction {
    /// Control points, guaranteed non-empty and sorted ascending by `density`.
    pub points: Vec<TransferPoint>,
}
impl TransferFunction {
    /// Construct from a list of control points.
    ///
    /// Returns an error if the list is empty or not sorted by `density`.
    pub fn new(points: Vec<TransferPoint>) -> Result<Self, VolumetricRenderError> {
        if points.is_empty() {
            return Err(VolumetricRenderError::EmptyTransferFunction);
        }
        for w in points.windows(2) {
            if w[0].density > w[1].density {
                return Err(VolumetricRenderError::UnsortedTransferFunction);
            }
        }
        Ok(Self { points })
    }
    /// Evaluate the transfer function at `density` via piecewise linear interpolation.
    /// Clamps to the first / last control point outside the range.
    pub fn evaluate(&self, density: f32) -> ([f32; 3], f32) {
        let pts = &self.points;
        if density <= pts[0].density {
            return (pts[0].color, pts[0].opacity);
        }
        if density >= pts[pts.len() - 1].density {
            let p = &pts[pts.len() - 1];
            return (p.color, p.opacity);
        }
        let mut lo = 0usize;
        let mut hi = pts.len() - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if pts[mid].density <= density {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = (density - pts[lo].density) / (pts[hi].density - pts[lo].density);
        let lerp3 = |a: [f32; 3], b: [f32; 3]| {
            [
                a[0] + t * (b[0] - a[0]),
                a[1] + t * (b[1] - a[1]),
                a[2] + t * (b[2] - a[2]),
            ]
        };
        let color = lerp3(pts[lo].color, pts[hi].color);
        let opacity = pts[lo].opacity + t * (pts[hi].opacity - pts[lo].opacity);
        (color, opacity)
    }
    /// Simple grayscale: density 0 → transparent black, `density_max` → opaque white.
    pub fn grayscale(density_max: f32) -> Self {
        Self {
            points: vec![
                TransferPoint {
                    density: 0.0,
                    color: [0.0; 3],
                    opacity: 0.0,
                },
                TransferPoint {
                    density: density_max.max(f32::EPSILON),
                    color: [1.0; 3],
                    opacity: 1.0,
                },
            ],
        }
    }
    /// Classic heat colormap: transparent-blue → cyan → green → yellow → red-opaque.
    pub fn heat(density_max: f32) -> Self {
        let d = density_max.max(f32::EPSILON);
        Self {
            points: vec![
                TransferPoint {
                    density: 0.0,
                    color: [0.0, 0.0, 1.0],
                    opacity: 0.0,
                },
                TransferPoint {
                    density: d * 0.25,
                    color: [0.0, 1.0, 1.0],
                    opacity: 0.25,
                },
                TransferPoint {
                    density: d * 0.5,
                    color: [0.0, 1.0, 0.0],
                    opacity: 0.5,
                },
                TransferPoint {
                    density: d * 0.75,
                    color: [1.0, 1.0, 0.0],
                    opacity: 0.75,
                },
                TransferPoint {
                    density: d,
                    color: [1.0, 0.0, 0.0],
                    opacity: 1.0,
                },
            ],
        }
    }
}
/// Errors produced by the volumetric rendering subsystem.
#[derive(Debug, Error)]
pub enum VolumetricRenderError {
    /// Volume grid has zero extent in at least one dimension.
    #[error("Zero-size volume: nx={nx} ny={ny} nz={nz}")]
    ZeroSizeVolume { nx: usize, ny: usize, nz: usize },
    /// Transfer function point list is empty.
    #[error("Transfer function has no points")]
    EmptyTransferFunction,
    /// Transfer function control points are not sorted by density.
    #[error("Transfer function points are not sorted by density ascending")]
    UnsortedTransferFunction,
    /// Input slice lengths are inconsistent with `n_gaussians`.
    #[error("Buffer length mismatch: expected {expected}, got {got} for '{field}'")]
    BufferLengthMismatch {
        field: &'static str,
        expected: usize,
        got: usize,
    },
    /// Invalid camera parameters.
    #[error("Invalid camera: {0}")]
    InvalidCamera(String),
}
/// A single control point in a piecewise-linear transfer function.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferPoint {
    /// Input density value (non-negative).
    pub density: f32,
    /// RGB colour output at this density.
    pub color: [f32; 3],
    /// Opacity (alpha) at this density, in `[0, 1]`.
    pub opacity: f32,
}
/// Configuration for the ray-marching integrator.
#[derive(Debug, Clone)]
pub struct VolumetricRenderConfig {
    /// World-space step size between samples.
    pub step_size: f32,
    /// Maximum number of steps per ray (safety limit).
    pub max_steps: usize,
    /// Front-to-back: stop accumulating when alpha exceeds this value.
    pub early_termination_alpha: f32,
    /// Integration algorithm.
    pub integration: VolumetricIntegration,
    /// Add sub-step jitter to reduce wood-grain banding.
    pub jitter: bool,
    /// Seed for the xorshift64 PRNG used for jitter.
    pub jitter_seed: u64,
}
/// Result of marching a single ray through the volume.
#[derive(Debug, Clone)]
pub struct RayMarchResult {
    /// Accumulated colour (RGB).
    pub color: [f32; 3],
    /// Accumulated opacity.
    pub alpha: f32,
    /// Number of integration steps taken.
    pub n_steps: usize,
    /// Ray parameter at volume entry.
    pub t_entry: f32,
    /// Ray parameter at volume exit.
    pub t_exit: f32,
}
