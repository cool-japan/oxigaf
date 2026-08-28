//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use thiserror::Error;

use super::functions::{quat_inverse, quat_mul, quat_rotate};

/// Aggregate spatial statistics for a Gaussian cloud.
#[derive(Debug, Clone)]
pub struct GeometryStats {
    /// Number of Gaussians.
    pub n_gaussians: usize,
    /// Axis-aligned bounding box.
    pub bbox: GaussianBBox,
    /// Approximate bounding sphere (Ritter's algorithm).
    pub bounding_sphere: BoundingSphere,
    /// Mean position (centroid).
    pub centroid: [f32; 3],
    /// Mean of `exp(log_scale)` across all axes and all Gaussians.
    pub mean_scale: f32,
    /// Maximum per-Gaussian scale (maximum `exp(log_scale)` over all axes and Gaussians).
    pub max_scale: f32,
    /// Minimum per-Gaussian scale (minimum `exp(log_scale)` over all axes and Gaussians).
    pub min_scale: f32,
    /// Standard deviation of positions treated as a scalar (pooled across x, y, z).
    pub std_position: f32,
}
impl GeometryStats {
    /// Human-readable summary suitable for CLI output.
    #[must_use]
    pub fn format_summary(&self) -> String {
        let c = self.bbox.center();
        let s = self.bbox.size();
        let sp = self.bounding_sphere;
        format!(
            "Gaussians : {}\n\
             BBox min  : [{:.4}, {:.4}, {:.4}]\n\
             BBox max  : [{:.4}, {:.4}, {:.4}]\n\
             BBox size : [{:.4}, {:.4}, {:.4}]  (volume {:.4})\n\
             BBox ctr  : [{:.4}, {:.4}, {:.4}]\n\
             Sphere    : center [{:.4}, {:.4}, {:.4}]  radius {:.4}\n\
             Centroid  : [{:.4}, {:.4}, {:.4}]\n\
             Scale     : mean {:.4}  min {:.4}  max {:.4}\n\
             Std pos   : {:.4}",
            self.n_gaussians,
            self.bbox.min[0],
            self.bbox.min[1],
            self.bbox.min[2],
            self.bbox.max[0],
            self.bbox.max[1],
            self.bbox.max[2],
            s[0],
            s[1],
            s[2],
            self.bbox.volume(),
            c[0],
            c[1],
            c[2],
            sp.center[0],
            sp.center[1],
            sp.center[2],
            sp.radius,
            self.centroid[0],
            self.centroid[1],
            self.centroid[2],
            self.mean_scale,
            self.min_scale,
            self.max_scale,
            self.std_position,
        )
    }
}
/// Rigid body transform: uniform scale, rotation (unit quaternion, w-last), translation.
///
/// `scale` and `rotation` only affect a Gaussian's *centre* when applied via
/// [`crate::geometry_tools::transform_positions`]/[`RigidTransform::apply_to_point`] — the
/// Gaussian's own physical extent is stored separately as a per-Gaussian
/// log-scale array, not here. Applying a transform whose `scale != 1.0` (or
/// whose `rotation` is non-identity, since orientation matters for
/// anisotropic Gaussians) to positions alone moves every centre without
/// resizing the Gaussians themselves, tearing the scene apart rather than
/// scaling it uniformly. Use [`crate::geometry_tools::transform_scales`]
/// alongside [`crate::geometry_tools::transform_positions`] (and
/// [`crate::geometry_tools::transform_rotations`]) so all arrays describing
/// the scene stay consistent with the same transform.
#[derive(Debug, Clone, Copy)]
pub struct RigidTransform {
    /// Unit quaternion `[qx, qy, qz, qw]` representing the rotation part.
    pub rotation: [f32; 4],
    /// World-space translation applied after rotation.
    pub translation: [f32; 3],
    /// Uniform scale factor (default `1.0`), applied to positions by
    /// [`RigidTransform::apply_to_point`]. Apply the same factor to a
    /// scene's per-Gaussian log-scales with
    /// [`crate::geometry_tools::transform_scales`] — see the struct-level
    /// documentation.
    pub scale: f32,
}
impl RigidTransform {
    /// Identity transform (no rotation, no translation, scale = 1).
    #[must_use]
    pub fn identity() -> Self {
        Self {
            rotation: [0.0, 0.0, 0.0, 1.0],
            translation: [0.0, 0.0, 0.0],
            scale: 1.0,
        }
    }
    /// Pure translation, no rotation.
    #[must_use]
    pub fn translation_only(t: [f32; 3]) -> Self {
        Self {
            rotation: [0.0, 0.0, 0.0, 1.0],
            translation: t,
            scale: 1.0,
        }
    }
    /// Pure uniform scale, no rotation or translation.
    #[must_use]
    pub fn from_scale(scale: f32) -> Self {
        Self {
            rotation: [0.0, 0.0, 0.0, 1.0],
            translation: [0.0, 0.0, 0.0],
            scale,
        }
    }
    /// Apply the full transform to point `p`: first scale, then rotate, then translate.
    #[must_use]
    pub fn apply_to_point(&self, p: [f32; 3]) -> [f32; 3] {
        let scaled = [p[0] * self.scale, p[1] * self.scale, p[2] * self.scale];
        let rotated = quat_rotate(self.rotation, scaled);
        [
            rotated[0] + self.translation[0],
            rotated[1] + self.translation[1],
            rotated[2] + self.translation[2],
        ]
    }
    /// Apply only the rotation to a direction vector (no scale, no translation).
    #[must_use]
    pub fn apply_to_direction(&self, d: [f32; 3]) -> [f32; 3] {
        quat_rotate(self.rotation, d)
    }
    /// Compose transforms: returns a transform equivalent to applying `self` first, then `other`.
    ///
    /// Combined translation = other.rotation(self.translation * other.scale * self.scale)  + other.translation
    /// Combined rotation    = other.rotation * self.rotation
    /// Combined scale       = self.scale * other.scale
    #[must_use]
    pub fn compose(&self, other: &RigidTransform) -> RigidTransform {
        let combined_rotation = quat_mul(other.rotation, self.rotation);
        let combined_scale = self.scale * other.scale;
        let self_t_scaled = [
            self.translation[0] * other.scale,
            self.translation[1] * other.scale,
            self.translation[2] * other.scale,
        ];
        let rotated_t = quat_rotate(other.rotation, self_t_scaled);
        let combined_translation = [
            rotated_t[0] + other.translation[0],
            rotated_t[1] + other.translation[1],
            rotated_t[2] + other.translation[2],
        ];
        RigidTransform {
            rotation: combined_rotation,
            translation: combined_translation,
            scale: combined_scale,
        }
    }
    /// Inverse: undoes the transform applied by `self`.
    #[must_use]
    pub fn inverse(&self) -> RigidTransform {
        let inv_q = quat_inverse(self.rotation);
        let inv_scale = if self.scale.abs() > f32::EPSILON {
            1.0 / self.scale
        } else {
            1.0
        };
        let neg_t = [
            -self.translation[0],
            -self.translation[1],
            -self.translation[2],
        ];
        let rotated_neg_t = quat_rotate(inv_q, neg_t);
        let inv_translation = [
            rotated_neg_t[0] * inv_scale,
            rotated_neg_t[1] * inv_scale,
            rotated_neg_t[2] * inv_scale,
        ];
        RigidTransform {
            rotation: inv_q,
            translation: inv_translation,
            scale: inv_scale,
        }
    }
}
/// Approximate bounding sphere for a Gaussian cloud.
#[derive(Debug, Clone, Copy)]
pub struct BoundingSphere {
    /// Centre of the sphere.
    pub center: [f32; 3],
    /// Radius.
    pub radius: f32,
}
impl BoundingSphere {
    /// Returns `true` if point `p` is inside (or on the boundary of) the sphere.
    #[must_use]
    pub fn contains(&self, p: [f32; 3]) -> bool {
        let dx = p[0] - self.center[0];
        let dy = p[1] - self.center[1];
        let dz = p[2] - self.center[2];
        (dx * dx + dy * dy + dz * dz) <= self.radius * self.radius
    }
    /// Returns `true` when the two spheres overlap (touching counts).
    #[must_use]
    pub fn intersects(&self, other: &BoundingSphere) -> bool {
        let dx = self.center[0] - other.center[0];
        let dy = self.center[1] - other.center[1];
        let dz = self.center[2] - other.center[2];
        let dist2 = dx * dx + dy * dy + dz * dz;
        let r_sum = self.radius + other.radius;
        dist2 <= r_sum * r_sum
    }
}
/// Errors produced by geometry operations.
#[derive(Debug, Error)]
pub enum GeometryError {
    /// No positions were provided.
    #[error("Empty point cloud: no positions")]
    EmptyCloud,
    /// Positions slice length is not divisible by 3.
    #[error("Positions length {len} is not divisible by 3")]
    InvalidPositionLength { len: usize },
    /// Rotations slice length is not divisible by 4.
    #[error("Rotations length {len} is not divisible by 4")]
    InvalidRotationLength { len: usize },
    /// Scales slice length is not divisible by 3.
    #[error("Scales length {len} is not divisible by 3")]
    InvalidScaleLength { len: usize },
    /// Different arrays do not agree on how many Gaussians there are.
    #[error("Count mismatch: {n_pos} positions vs {n_other} {name}")]
    CountMismatch {
        n_pos: usize,
        n_other: usize,
        name: String,
    },
    /// A transform was not well-formed (e.g., zero quaternion, non-positive scale).
    #[error("Invalid transform: {reason}")]
    InvalidTransform { reason: String },
}
/// Three-dimensional axis-aligned bounding box for a Gaussian cloud.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianBBox {
    /// Minimum corner `[x_min, y_min, z_min]`.
    pub min: [f32; 3],
    /// Maximum corner `[x_max, y_max, z_max]`.
    pub max: [f32; 3],
}
impl GaussianBBox {
    /// Geometric centre of the bounding box.
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }
    /// Per-axis size (`max - min`).
    #[must_use]
    pub fn size(&self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }
    /// Volume of the bounding box (`sx * sy * sz`).
    #[must_use]
    pub fn volume(&self) -> f32 {
        let s = self.size();
        s[0] * s[1] * s[2]
    }
    /// Length of the space diagonal.
    #[must_use]
    pub fn diagonal(&self) -> f32 {
        let s = self.size();
        (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt()
    }
    /// Returns `true` if point `p` is inside (or on the boundary of) the box.
    #[must_use]
    pub fn contains(&self, p: [f32; 3]) -> bool {
        p[0] >= self.min[0]
            && p[0] <= self.max[0]
            && p[1] >= self.min[1]
            && p[1] <= self.max[1]
            && p[2] >= self.min[2]
            && p[2] <= self.max[2]
    }
    /// Returns `true` when this box and `other` overlap (touching counts).
    #[must_use]
    pub fn intersects(&self, other: &GaussianBBox) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
    }
    /// Smallest bounding box that contains both `self` and `other`.
    #[must_use]
    pub fn union(&self, other: &GaussianBBox) -> GaussianBBox {
        GaussianBBox {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }
    /// Intersection of the two boxes, or `None` when they do not overlap.
    #[must_use]
    pub fn intersection(&self, other: &GaussianBBox) -> Option<GaussianBBox> {
        let imin = [
            self.min[0].max(other.min[0]),
            self.min[1].max(other.min[1]),
            self.min[2].max(other.min[2]),
        ];
        let imax = [
            self.max[0].min(other.max[0]),
            self.max[1].min(other.max[1]),
            self.max[2].min(other.max[2]),
        ];
        if imin[0] <= imax[0] && imin[1] <= imax[1] && imin[2] <= imax[2] {
            Some(GaussianBBox {
                min: imin,
                max: imax,
            })
        } else {
            None
        }
    }
    /// Expand every face outward by `margin` (or inward if negative).
    #[must_use]
    pub fn expand(&self, margin: f32) -> GaussianBBox {
        GaussianBBox {
            min: [
                self.min[0] - margin,
                self.min[1] - margin,
                self.min[2] - margin,
            ],
            max: [
                self.max[0] + margin,
                self.max[1] + margin,
                self.max[2] + margin,
            ],
        }
    }
    /// Three-dimensional intersection-over-union (IoU) with `other`.
    ///
    /// Returns 0.0 when either box has zero volume.
    #[must_use]
    pub fn iou(&self, other: &GaussianBBox) -> f32 {
        let inter_vol = self.intersection(other).map(|b| b.volume()).unwrap_or(0.0);
        if inter_vol == 0.0 {
            return 0.0;
        }
        let union_vol = self.volume() + other.volume() - inter_vol;
        if union_vol <= 0.0 {
            return 0.0;
        }
        inter_vol / union_vol
    }
}
