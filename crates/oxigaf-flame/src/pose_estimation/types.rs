//! Data types for pose estimation: the error type, 2D/3D landmark and
//! correspondence representations, the pinhole camera model, the estimated
//! [`HeadPose`], solver [`PoseConfig`], [`PitchReference`], and the
//! quaternion-smoothing [`PoseTracker`].
//!
//! Split out of the former monolithic `pose_estimation.rs` to stay under
//! the workspace's 2000-line-per-file policy; see [`super::math`] for the
//! private mat3/quaternion helpers and [`super::functions`] for the pose
//! solvers themselves.

use thiserror::Error;

use super::math::{mat3_to_quat, mat3_vec3_mul, quat_slerp, quat_to_mat3};

/// Errors that can occur during pose estimation.
#[derive(Debug, Error)]
pub enum PoseEstimationError {
    /// Not enough point correspondences were provided.
    #[error("Insufficient points: got {got}, required {required}")]
    InsufficientPoints { got: usize, required: usize },

    /// A numerical computation failed (e.g., division by near-zero value).
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// The solver configuration is invalid.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// The system matrix was singular and could not be inverted.
    #[error("Singular matrix encountered")]
    SingularMatrix,
}

// ---------------------------------------------------------------------------
// Correspondence types
// ---------------------------------------------------------------------------

/// A 2D image landmark observation.
#[derive(Debug, Clone, Copy)]
pub struct Landmark2D {
    /// Pixel column coordinate.
    pub u: f32,
    /// Pixel row coordinate.
    pub v: f32,
    /// Detection confidence in \[0, 1\].
    pub confidence: f32,
}

impl Landmark2D {
    /// Create a landmark with full confidence.
    #[must_use]
    pub fn new(u: f32, v: f32) -> Self {
        Self {
            u,
            v,
            confidence: 1.0,
        }
    }

    /// Create a landmark with explicit confidence.
    #[must_use]
    pub fn with_confidence(u: f32, v: f32, confidence: f32) -> Self {
        Self { u, v, confidence }
    }
}

/// A 3D model landmark position (in model space).
#[derive(Debug, Clone, Copy)]
pub struct Landmark3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Landmark3D {
    /// Create a 3D model landmark.
    #[must_use]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

/// A correspondence between a 3D model point and its 2D image observation.
#[derive(Debug, Clone, Copy)]
pub struct PointCorrespondence {
    pub point_3d: Landmark3D,
    pub point_2d: Landmark2D,
}

impl PointCorrespondence {
    /// Create a new correspondence.
    #[must_use]
    pub fn new(point_3d: Landmark3D, point_2d: Landmark2D) -> Self {
        Self { point_3d, point_2d }
    }

    /// Weight of this correspondence (equals the observation confidence).
    #[must_use]
    pub fn weight(&self) -> f32 {
        self.point_2d.confidence
    }
}

// ---------------------------------------------------------------------------
// Camera model
// ---------------------------------------------------------------------------

/// A simple pinhole camera for projecting 3D points to 2D image coordinates.
///
/// Named `PosePinholeCamera` to avoid collision with `fitting::PinholeCamera`.
#[derive(Debug, Clone)]
pub struct PosePinholeCamera {
    /// Focal length in pixels.
    pub focal_length: f32,
    /// Principal point x (image center, pixels).
    pub cx: f32,
    /// Principal point y (image center, pixels).
    pub cy: f32,
}

impl PosePinholeCamera {
    /// Create a pinhole camera with explicit parameters.
    #[must_use]
    pub fn new(focal_length: f32, cx: f32, cy: f32) -> Self {
        Self {
            focal_length,
            cx,
            cy,
        }
    }

    /// Construct from image dimensions, assuming square pixels and a centered
    /// principal point.  The focal length is approximated as
    /// `max(width, height)` (a common heuristic).
    #[must_use]
    pub fn from_image_size(width: usize, height: usize) -> Self {
        let focal = width.max(height) as f32;
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        Self::new(focal, cx, cy)
    }

    /// Project a 3D camera-space point `(x, y, z)` to image coordinates.
    ///
    /// Returns `None` if `z <= 0` (point is at or behind the camera).
    ///
    /// ```text
    /// u = cx + focal * x / z
    /// v = cy - focal * y / z   (image +v is down)
    /// ```
    #[must_use]
    pub fn project(&self, cam_x: f32, cam_y: f32, cam_z: f32) -> Option<(f32, f32)> {
        if cam_z <= 0.0 {
            return None;
        }
        let img_u = self.cx + self.focal_length * cam_x / cam_z;
        let img_v = self.cy - self.focal_length * cam_y / cam_z;
        Some((img_u, img_v))
    }

    /// Unproject an image point `(img_u, img_v)` at depth `depth_z` to a 3D
    /// camera-space point.
    ///
    /// ```text
    /// x = (u - cx) * z / focal
    /// y = -(v - cy) * z / focal
    /// ```
    #[must_use]
    pub fn unproject(&self, img_u: f32, img_v: f32, depth_z: f32) -> (f32, f32, f32) {
        let cam_x = (img_u - self.cx) * depth_z / self.focal_length;
        let cam_y = -(img_v - self.cy) * depth_z / self.focal_length;
        (cam_x, cam_y, depth_z)
    }
}

// ---------------------------------------------------------------------------
// Pose result
// ---------------------------------------------------------------------------

/// Estimated head pose expressed as rotation + translation.
#[derive(Debug, Clone)]
pub struct HeadPose {
    /// Row-major 3×3 rotation matrix (model-to-camera).
    pub rotation: [[f32; 3]; 3],
    /// Translation vector `[tx, ty, tz]` (model origin in camera space).
    pub translation: [f32; 3],
    /// Mean reprojection error in pixels.
    pub reprojection_error: f32,
}

impl HeadPose {
    /// Construct a pose with zero reprojection error (caller sets it later).
    #[must_use]
    pub fn new(rotation: [[f32; 3]; 3], translation: [f32; 3]) -> Self {
        Self {
            rotation,
            translation,
            reprojection_error: 0.0,
        }
    }

    /// Transform a model-space landmark into camera space:
    /// `camera = R * [x, y, z]^T + t`.
    #[must_use]
    pub fn transform(&self, point: Landmark3D) -> (f32, f32, f32) {
        let v = [point.x, point.y, point.z];
        let r = mat3_vec3_mul(&self.rotation, v);
        (
            r[0] + self.translation[0],
            r[1] + self.translation[1],
            r[2] + self.translation[2],
        )
    }

    /// Project a model-space landmark to 2D image coordinates using this pose
    /// and the given camera.
    #[must_use]
    pub fn project_point(
        &self,
        point: Landmark3D,
        camera: &PosePinholeCamera,
    ) -> Option<(f32, f32)> {
        let (cx, cy, cz) = self.transform(point);
        camera.project(cx, cy, cz)
    }

    /// Return the axis-angle representation of the rotation matrix.
    ///
    /// Uses the inverse Rodrigues formula.
    #[must_use]
    pub fn rotation_axis_angle(&self) -> [f32; 3] {
        let r = &self.rotation;
        let trace = r[0][0] + r[1][1] + r[2][2];
        let cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();

        if theta.abs() < 1e-6 {
            return [0.0, 0.0, 0.0];
        }

        let factor = 1.0 / (2.0 * theta.sin());
        let ax = (r[2][1] - r[1][2]) * factor * theta;
        let ay = (r[0][2] - r[2][0]) * factor * theta;
        let az = (r[1][0] - r[0][1]) * factor * theta;
        [ax, ay, az]
    }

    /// Extract Euler angles `[yaw, pitch, roll]` in radians from the rotation
    /// matrix.
    ///
    /// Uses the ZYX convention:
    /// ```text
    /// pitch = asin(-R[2][0])   clamped to [-π/2, π/2]
    /// yaw   = atan2(R[1][0], R[0][0])   if cos(pitch) ≠ 0
    /// roll  = atan2(R[2][1], R[2][2])   if cos(pitch) ≠ 0
    /// ```
    #[must_use]
    pub fn euler_angles(&self) -> [f32; 3] {
        let r = &self.rotation;
        let sin_pitch = (-r[2][0]).clamp(-1.0, 1.0);
        let pitch = sin_pitch.asin();
        let cos_pitch = pitch.cos();

        if cos_pitch.abs() < 1e-6 {
            // Gimbal lock: roll is undefined and folds into yaw, so it is set to 0.
            //
            // With `R = Rz(yaw)·Ry(pitch)·Rx(roll)` and `roll = 0`:
            //   pitch = +π/2 → R[1][1] = cos(yaw − roll), R[1][2] =  sin(yaw − roll)
            //   pitch = −π/2 → R[1][1] = cos(yaw + roll), R[1][2] = −sin(yaw + roll)
            // so the sine term must be negated in the −π/2 branch, otherwise the
            // recovered yaw comes out mirrored.
            let yaw = if sin_pitch > 0.0 {
                r[1][2].atan2(r[1][1])
            } else {
                (-r[1][2]).atan2(r[1][1])
            };
            return [yaw, pitch, 0.0];
        }

        let yaw = r[1][0].atan2(r[0][0]);
        let roll = r[2][1].atan2(r[2][2]);
        [yaw, pitch, roll]
    }
}

// ---------------------------------------------------------------------------
// Pose configuration
// ---------------------------------------------------------------------------

/// Configuration for the pose estimation solver.
#[derive(Debug, Clone)]
pub struct PoseConfig {
    /// Minimum number of correspondences required.
    pub min_correspondences: usize,
    /// Maximum reprojection error (pixels) for a point to be an inlier.
    pub max_reprojection_error: f32,
    /// Number of RANSAC iterations (0 = no RANSAC, use all points).
    pub ransac_iterations: usize,
    /// Minimum detection confidence to include a correspondence.
    pub min_confidence: f32,
}

impl Default for PoseConfig {
    fn default() -> Self {
        Self {
            min_correspondences: 4,
            max_reprojection_error: 10.0,
            ransac_iterations: 0,
            min_confidence: 0.5,
        }
    }
}

impl PoseConfig {
    /// Validate that the configuration is self-consistent.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn validate(&self) -> Result<(), PoseEstimationError> {
        if self.min_correspondences < 4 {
            return Err(PoseEstimationError::InvalidConfig(format!(
                "min_correspondences must be >= 4, got {}",
                self.min_correspondences
            )));
        }
        if self.max_reprojection_error <= 0.0 {
            return Err(PoseEstimationError::InvalidConfig(format!(
                "max_reprojection_error must be > 0, got {}",
                self.max_reprojection_error
            )));
        }
        if !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(PoseEstimationError::InvalidConfig(format!(
                "min_confidence must be in [0, 1], got {}",
                self.min_confidence
            )));
        }
        Ok(())
    }
}

/// Model-space reference required to turn a vertical foreshortening into an
/// actual pitch angle.
///
/// There is deliberately no `Default`: the group centroids and the
/// weak-perspective scale are properties of the caller's landmark set and
/// camera, and guessing them would silently fabricate the resulting angle.
#[derive(Debug, Clone, Copy)]
pub struct PitchReference {
    /// Model-space centroid of the upper landmark group (e.g. brow/eye centre),
    /// in the neutral (unrotated) model pose.
    pub upper_3d: Landmark3D,
    /// Model-space centroid of the lower landmark group (e.g. mouth/chin centre),
    /// in the neutral (unrotated) model pose.
    pub lower_3d: Landmark3D,
    /// Weak-perspective scale in **pixels per model unit** (`focal_length / t_z`).
    ///
    /// Available from a previous solve as
    /// `camera.focal_length / pose.translation[2]`, or from a known metric
    /// landmark distance divided by its pixel distance.
    pub scale: f32,
}

impl PitchReference {
    /// Create a pitch reference.
    #[must_use]
    pub fn new(upper_3d: Landmark3D, lower_3d: Landmark3D, scale: f32) -> Self {
        Self {
            upper_3d,
            lower_3d,
            scale,
        }
    }
}

// ---------------------------------------------------------------------------
// Pose tracker (temporal smoothing)
// ---------------------------------------------------------------------------

/// Track head pose over time using exponential moving average (EMA) smoothing.
///
/// Translation and reprojection error are smoothed with a plain EMA; the
/// rotation is smoothed on the quaternion sphere (slerp), so the tracked
/// rotation is always a proper rotation matrix.
pub struct PoseTracker {
    /// EMA decay factor.  Higher values produce smoother (lagging) output.
    pub ema_decay: f32,
    /// Smoothed rotation, stored as a unit quaternion `[w, x, y, z]`.
    current_rotation: Option<[f32; 4]>,
    current_translation: Option<[f32; 3]>,
    current_error: f32,
}

impl PoseTracker {
    /// Create a new tracker with the given EMA decay factor.
    #[must_use]
    pub fn new(ema_decay: f32) -> Self {
        Self {
            ema_decay,
            current_rotation: None,
            current_translation: None,
            current_error: 0.0,
        }
    }

    /// Incorporate a new pose estimate into the tracker.
    ///
    /// The first pose is accepted directly.  Afterwards translation and error
    /// are blended with an EMA while the rotation is interpolated on the unit
    /// quaternion sphere (`slerp` with `t = 1 − α`, shorter-arc / antipodal
    /// safe), which keeps the smoothed rotation in `SO(3)` — an element-wise
    /// matrix blend does not.
    pub fn update(&mut self, pose: &HeadPose) {
        let alpha = self.ema_decay.clamp(0.0, 1.0);
        let observed = mat3_to_quat(&pose.rotation);

        match (self.current_rotation, self.current_translation) {
            (None, _) | (_, None) => {
                // First observation: accept directly
                self.current_rotation = Some(observed);
                self.current_translation = Some(pose.translation);
                self.current_error = pose.reprojection_error;
            }
            (Some(prev_q), Some(prev_t)) => {
                // EMA for translation
                let new_t = [
                    alpha * prev_t[0] + (1.0 - alpha) * pose.translation[0],
                    alpha * prev_t[1] + (1.0 - alpha) * pose.translation[1],
                    alpha * prev_t[2] + (1.0 - alpha) * pose.translation[2],
                ];

                // Slerp for rotation: t = 1 − α moves toward the observation by
                // the same weight the EMA gives it.
                let new_q = quat_slerp(prev_q, observed, 1.0 - alpha);

                self.current_rotation = Some(new_q);
                self.current_translation = Some(new_t);
                self.current_error =
                    alpha * self.current_error + (1.0 - alpha) * pose.reprojection_error;
            }
        }
    }

    /// Return the current smoothed pose, or `None` if no updates have been
    /// received.  The `rotation` is rebuilt from the tracked quaternion, so it
    /// is always orthonormal with `det = +1`.
    #[must_use]
    pub fn current_pose(&self) -> Option<HeadPose> {
        match (self.current_rotation, self.current_translation) {
            (Some(q), Some(t)) => {
                let mut p = HeadPose::new(quat_to_mat3(q), t);
                p.reprojection_error = self.current_error;
                Some(p)
            }
            _ => None,
        }
    }

    /// Reset the tracker to its initial (no-pose) state.
    pub fn reset(&mut self) {
        self.current_rotation = None;
        self.current_translation = None;
        self.current_error = 0.0;
    }

    /// Return `true` if the tracker has received at least one pose update.
    #[must_use]
    pub fn has_pose(&self) -> bool {
        self.current_rotation.is_some()
    }
}
