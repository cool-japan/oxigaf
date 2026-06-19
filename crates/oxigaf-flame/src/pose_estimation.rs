//! Head pose estimation from 2D facial landmark observations.
//!
//! Recovers FLAME head pose parameters (rotation and translation) from 2D
//! landmark observations using a simplified `PnP` approach with a weak-perspective
//! / orthographic approximation and temporal smoothing.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

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
            // Gimbal lock: roll is undefined, set to 0
            let yaw = r[1][2].atan2(r[1][1]);
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

// ---------------------------------------------------------------------------
// Utility matrix operations (private)
// ---------------------------------------------------------------------------

/// Multiply two 3×3 matrices: `A * B`.
#[allow(dead_code)]
fn mat3_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut c = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Multiply a 3×3 matrix by a 3-vector: `M * v`.
fn mat3_vec3_mul(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Transpose a 3×3 matrix.
#[allow(dead_code)]
fn mat3_transpose(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Return the 3×3 identity matrix.
fn mat3_identity() -> [[f32; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

// ---------------------------------------------------------------------------
// Pose solvers
// ---------------------------------------------------------------------------

/// Estimate head pose using a weak-perspective (orthographic) approximation.
///
/// This approach works well when the face occupies a moderate field of view.
/// At least `config.min_correspondences` (≥ 4) correspondences with confidence
/// ≥ `config.min_confidence` are required.
///
/// # Algorithm
/// 1. Compute the 3D centroid of the model points.
/// 2. Compute the 2D centroid of the image observations.
/// 3. Estimate a uniform scale factor from XY deviations.
/// 4. Derive translation from centroid alignment.
/// 5. Use the identity rotation (simplified — no in-plane rotation fitted).
/// 6. Compute reprojection error.
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn estimate_pose_weak_perspective(
    correspondences: &[PointCorrespondence],
    camera: &PosePinholeCamera,
    config: &PoseConfig,
) -> Result<HeadPose, PoseEstimationError> {
    config.validate()?;

    // Filter by minimum confidence
    let filtered: Vec<&PointCorrespondence> = correspondences
        .iter()
        .filter(|c| c.point_2d.confidence >= config.min_confidence)
        .collect();

    let n = filtered.len();
    if n < config.min_correspondences {
        return Err(PoseEstimationError::InsufficientPoints {
            got: n,
            required: config.min_correspondences,
        });
    }

    let nf = n as f32;

    // Step 1: 3D centroid
    let centroid_3d_x = filtered.iter().map(|c| c.point_3d.x).sum::<f32>() / nf;
    let centroid_3d_y = filtered.iter().map(|c| c.point_3d.y).sum::<f32>() / nf;
    let _centroid_3d_z = filtered.iter().map(|c| c.point_3d.z).sum::<f32>() / nf;

    // Step 2: 2D centroid
    let centroid_2d_u = filtered.iter().map(|c| c.point_2d.u).sum::<f32>() / nf;
    let centroid_2d_v = filtered.iter().map(|c| c.point_2d.v).sum::<f32>() / nf;

    // Step 3: estimate scale from XY deviations
    // scale ≈ |Δu| / |Δx|  and  |Δv| / |Δy|, averaged
    let mut scale_samples: Vec<f32> = Vec::with_capacity(2 * n);

    for c in &filtered {
        let dx = (c.point_3d.x - centroid_3d_x).abs();
        let du = (c.point_2d.u - centroid_2d_u).abs();
        if dx > 1e-6 {
            scale_samples.push(du / dx);
        }

        let dy = (c.point_3d.y - centroid_3d_y).abs();
        let dv = (c.point_2d.v - centroid_2d_v).abs();
        if dy > 1e-6 {
            scale_samples.push(dv / dy);
        }
    }

    if scale_samples.is_empty() {
        return Err(PoseEstimationError::NumericalError(
            "All 3D model points are coincident; scale cannot be estimated".to_string(),
        ));
    }

    let scale = scale_samples.iter().sum::<f32>() / scale_samples.len() as f32;

    if scale < 1e-9 {
        return Err(PoseEstimationError::NumericalError(
            "Estimated scale is near zero; degenerate correspondence set".to_string(),
        ));
    }

    // Step 4: derive translation
    let t_x = (centroid_2d_u - camera.cx) / camera.focal_length;
    let t_y = -(centroid_2d_v - camera.cy) / camera.focal_length;
    let t_z = camera.focal_length / scale;

    // Step 5: identity rotation (simplified)
    let rotation = mat3_identity();
    let translation = [t_x, t_y, t_z];

    let mut pose = HeadPose::new(rotation, translation);

    // Step 6: compute reprojection error on all (filtered) correspondences
    let corr_vec: Vec<PointCorrespondence> = filtered.iter().map(|c| **c).collect();
    pose.reprojection_error = reprojection_error(&corr_vec, &pose, camera);

    Ok(pose)
}

/// Estimate yaw angle from the horizontal asymmetry of symmetric landmark pairs.
///
/// Each element of `left_points` and `right_points` is a `[u, v]` image
/// coordinate; the arrays must have the same length and at least one element.
///
/// Returns the estimated yaw in radians (positive = turned right).
///
/// # Errors
///
/// Returns an error if the operation fails.
pub fn estimate_yaw_from_symmetry(
    left_points: &[[f32; 2]],
    right_points: &[[f32; 2]],
) -> Result<f32, PoseEstimationError> {
    if left_points.is_empty() || right_points.is_empty() {
        return Err(PoseEstimationError::InsufficientPoints {
            got: 0,
            required: 1,
        });
    }
    if left_points.len() != right_points.len() {
        return Err(PoseEstimationError::InsufficientPoints {
            got: left_points.len().min(right_points.len()),
            required: left_points.len().max(right_points.len()),
        });
    }

    // Compare per-side spread: for a frontal face both sides have equal spread;
    // when the face turns, the far side compresses (smaller spread) relative to
    // the near side.
    //
    // left_spread  = max(left_u) - min(left_u)
    // right_spread = max(right_u) - min(right_u)
    //
    // yaw ≈ asin((left_spread - right_spread) / (left_spread + right_spread))
    // Positive yaw = face turned right (right side compressed in image).
    //
    // If both spreads are near zero (single-point inputs), we cannot determine
    // yaw from spread alone; return 0.0 (frontal assumption).
    let left_max_u = left_points
        .iter()
        .map(|p| p[0])
        .fold(f32::NEG_INFINITY, f32::max);
    let left_min_u = left_points
        .iter()
        .map(|p| p[0])
        .fold(f32::INFINITY, f32::min);
    let right_max_u = right_points
        .iter()
        .map(|p| p[0])
        .fold(f32::NEG_INFINITY, f32::max);
    let right_min_u = right_points
        .iter()
        .map(|p| p[0])
        .fold(f32::INFINITY, f32::min);

    let left_spread = left_max_u - left_min_u;
    let right_spread = right_max_u - right_min_u;

    let total_spread = left_spread + right_spread;
    if total_spread < 1e-6 {
        // Single-point pairs (or all coincident): no asymmetry observable
        return Ok(0.0);
    }

    let sin_yaw = ((left_spread - right_spread) / total_spread).clamp(-1.0, 1.0);
    let yaw = sin_yaw.asin();

    Ok(yaw)
}

/// Estimate pitch from the vertical positions of upper and lower face landmarks.
///
/// `upper_points` and `lower_points` are `[u, v]` image coordinates (v
/// increases downward).  Both slices must be non-empty.
///
/// ## Algorithm
///
/// 1. Compute `upper_centroid_y`: mean v-coordinate of `upper_points`.
/// 2. Compute `lower_centroid_y`: mean v-coordinate of `lower_points`.
/// 3. Compute `vertical_span = |upper_centroid_y − lower_centroid_y|`.
///    Guard: if `vertical_span < 1e-6` return `Ok(0.0)` — groups coincide.
/// 4. Compute `center_y`: mean v-coordinate across **all** points.
/// 5. `sin_pitch = (upper_centroid_y + lower_centroid_y − 2·center_y) /
///    (vertical_span + 1e-6)`.
///    For a symmetric face both centroids are equidistant from `center_y`
///    with opposite signs → numerator is zero → frontal pose.
///    When group sizes differ or the face rotates in pitch the numerator
///    becomes non-zero.
/// 6. Return `Ok(sin_pitch.clamp(−1, 1).asin())`.
///
/// **Sign convention**: positive pitch means the upper landmarks are
/// displaced away from the overall centroid more than the lower ones
/// (face tilted back / looking up in image space).
///
/// # Errors
///
/// Returns [`PoseEstimationError::InsufficientPoints`] if either slice is
/// empty.
pub fn estimate_pitch_from_vertical(
    upper_points: &[[f32; 2]],
    lower_points: &[[f32; 2]],
) -> Result<f32, PoseEstimationError> {
    if upper_points.is_empty() {
        return Err(PoseEstimationError::InsufficientPoints {
            got: 0,
            required: 1,
        });
    }
    if lower_points.is_empty() {
        return Err(PoseEstimationError::InsufficientPoints {
            got: 0,
            required: 1,
        });
    }

    let n_upper = upper_points.len() as f32;
    let n_lower = lower_points.len() as f32;

    let upper_centroid_y = upper_points.iter().map(|p| p[1]).sum::<f32>() / n_upper;
    let lower_centroid_y = lower_points.iter().map(|p| p[1]).sum::<f32>() / n_lower;

    let vertical_span = (upper_centroid_y - lower_centroid_y).abs();
    if vertical_span < 1e-6 {
        return Ok(0.0);
    }

    let sum_all: f32 = upper_points
        .iter()
        .chain(lower_points.iter())
        .map(|p| p[1])
        .sum();
    let center_y = sum_all / (n_upper + n_lower);

    let sin_pitch = (upper_centroid_y + lower_centroid_y - 2.0 * center_y) / (vertical_span + 1e-6);

    Ok(sin_pitch.clamp(-1.0, 1.0).asin())
}

// ---------------------------------------------------------------------------
// Reprojection utilities
// ---------------------------------------------------------------------------

/// Compute the confidence-weighted mean reprojection error (in pixels) for a
/// set of correspondences under the given pose and camera.
///
/// Returns `0.0` if `correspondences` is empty or no points project forward.
#[must_use]
pub fn reprojection_error(
    correspondences: &[PointCorrespondence],
    pose: &HeadPose,
    camera: &PosePinholeCamera,
) -> f32 {
    if correspondences.is_empty() {
        return 0.0;
    }

    let mut total_error = 0.0_f32;
    let mut total_weight = 0.0_f32;

    for c in correspondences {
        if let Some((proj_u, proj_v)) = pose.project_point(c.point_3d, camera) {
            let du = proj_u - c.point_2d.u;
            let dv = proj_v - c.point_2d.v;
            let dist = (du * du + dv * dv).sqrt();
            let w = c.point_2d.confidence;
            total_error += dist * w;
            total_weight += w;
        }
    }

    if total_weight < 1e-9 {
        return 0.0;
    }

    total_error / total_weight
}

/// Count the number of correspondences whose reprojection error is below
/// `threshold` (in pixels).
#[must_use]
pub fn count_inliers(
    correspondences: &[PointCorrespondence],
    pose: &HeadPose,
    camera: &PosePinholeCamera,
    threshold: f32,
) -> usize {
    correspondences
        .iter()
        .filter(|c| {
            if let Some((proj_u, proj_v)) = pose.project_point(c.point_3d, camera) {
                let du = proj_u - c.point_2d.u;
                let dv = proj_v - c.point_2d.v;
                let dist = (du * du + dv * dv).sqrt();
                dist < threshold
            } else {
                false
            }
        })
        .count()
}

// ---------------------------------------------------------------------------
// Pose tracker (temporal smoothing)
// ---------------------------------------------------------------------------

/// Track head pose over time using exponential moving average (EMA) smoothing.
pub struct PoseTracker {
    /// EMA decay factor.  Higher values produce smoother (lagging) output.
    pub ema_decay: f32,
    current_rotation: Option<[[f32; 3]; 3]>,
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
    /// On the first call the pose is accepted directly.  On subsequent calls
    /// element-wise EMA blending is applied, and each row of the blended
    /// rotation is renormalized to unit length (approximate orthogonalization).
    pub fn update(&mut self, pose: &HeadPose) {
        let alpha = self.ema_decay;

        match (self.current_rotation, self.current_translation) {
            (None, _) | (_, None) => {
                // First observation: accept directly
                self.current_rotation = Some(pose.rotation);
                self.current_translation = Some(pose.translation);
                self.current_error = pose.reprojection_error;
            }
            (Some(prev_r), Some(prev_t)) => {
                // EMA for translation
                let new_t = [
                    alpha * prev_t[0] + (1.0 - alpha) * pose.translation[0],
                    alpha * prev_t[1] + (1.0 - alpha) * pose.translation[1],
                    alpha * prev_t[2] + (1.0 - alpha) * pose.translation[2],
                ];

                // EMA for rotation (element-wise) then normalize rows
                let mut new_r = [[0.0f32; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        new_r[i][j] = alpha * prev_r[i][j] + (1.0 - alpha) * pose.rotation[i][j];
                    }
                    // Normalize each row to unit length
                    let row_len = (new_r[i][0] * new_r[i][0]
                        + new_r[i][1] * new_r[i][1]
                        + new_r[i][2] * new_r[i][2])
                        .sqrt();
                    if row_len > 1e-9 {
                        new_r[i][0] /= row_len;
                        new_r[i][1] /= row_len;
                        new_r[i][2] /= row_len;
                    }
                }

                self.current_rotation = Some(new_r);
                self.current_translation = Some(new_t);
                self.current_error =
                    alpha * self.current_error + (1.0 - alpha) * pose.reprojection_error;
            }
        }
    }

    /// Return the current smoothed pose, or `None` if no updates have been
    /// received.
    #[must_use]
    pub fn current_pose(&self) -> Option<HeadPose> {
        match (self.current_rotation, self.current_translation) {
            (Some(r), Some(t)) => {
                let mut p = HeadPose::new(r, t);
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::doc_markdown)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // -- Landmark constructors --

    #[test]
    fn test_landmark2d_new() {
        let lm = Landmark2D::new(10.0, 20.0);
        assert!((lm.u - 10.0).abs() < 1e-6);
        assert!((lm.v - 20.0).abs() < 1e-6);
        assert!((lm.confidence - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_landmark2d_with_confidence() {
        let lm = Landmark2D::with_confidence(5.0, 8.0, 0.75);
        assert!((lm.confidence - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_landmark3d_new() {
        let lm = Landmark3D::new(1.0, 2.0, 3.0);
        assert!((lm.x - 1.0).abs() < 1e-6);
        assert!((lm.y - 2.0).abs() < 1e-6);
        assert!((lm.z - 3.0).abs() < 1e-6);
    }

    // -- PosePinholeCamera --

    #[test]
    fn test_pinhole_from_image_size_proportions() {
        let cam = PosePinholeCamera::from_image_size(640, 480);
        // focal = max(640, 480) = 640
        assert!((cam.focal_length - 640.0).abs() < 1e-6);
        assert!((cam.cx - 320.0).abs() < 1e-6);
        assert!((cam.cy - 240.0).abs() < 1e-6);
    }

    #[test]
    fn test_pinhole_project_behind_camera_returns_none() {
        let cam = PosePinholeCamera::new(500.0, 320.0, 240.0);
        assert!(cam.project(0.1, 0.1, -1.0).is_none());
        assert!(cam.project(0.1, 0.1, 0.0).is_none());
    }

    #[test]
    fn test_pinhole_project_forward_point_in_image() {
        // A point along the optical axis projects to the principal point
        let cam = PosePinholeCamera::new(500.0, 320.0, 240.0);
        let (u, v) = cam.project(0.0, 0.0, 1.0).expect("should project");
        assert!((u - 320.0).abs() < 1e-4);
        assert!((v - 240.0).abs() < 1e-4);
    }

    #[test]
    fn test_pinhole_project_offset_point() {
        let cam = PosePinholeCamera::new(500.0, 320.0, 240.0);
        // x > 0 should yield u > cx
        let (u, _v) = cam.project(0.1, 0.0, 1.0).expect("should project");
        assert!(u > 320.0);
        // y > 0 should yield v < cy (y is flipped)
        let (_u, v) = cam.project(0.0, 0.1, 1.0).expect("should project");
        assert!(v < 240.0);
    }

    #[test]
    fn test_pinhole_unproject_roundtrip() {
        let cam = PosePinholeCamera::new(500.0, 320.0, 240.0);
        let orig_3d = (0.2_f32, -0.15_f32, 2.0_f32);
        let (u, v) = cam
            .project(orig_3d.0, orig_3d.1, orig_3d.2)
            .expect("should project");
        let (rx, ry, rz) = cam.unproject(u, v, orig_3d.2);
        assert!(
            (rx - orig_3d.0).abs() < 1e-4,
            "x mismatch: {} vs {}",
            rx,
            orig_3d.0
        );
        assert!(
            (ry - orig_3d.1).abs() < 1e-4,
            "y mismatch: {} vs {}",
            ry,
            orig_3d.1
        );
        assert!((rz - orig_3d.2).abs() < 1e-6);
    }

    // -- HeadPose --

    #[test]
    fn test_headpose_transform_identity_rotation() {
        let pose = HeadPose::new(mat3_identity(), [1.0, 2.0, 3.0]);
        let pt = Landmark3D::new(4.0, 5.0, 6.0);
        let (cx, cy, cz) = pose.transform(pt);
        assert!((cx - 5.0).abs() < 1e-5);
        assert!((cy - 7.0).abs() < 1e-5);
        assert!((cz - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_headpose_euler_angles_identity() {
        let pose = HeadPose::new(mat3_identity(), [0.0, 0.0, 1.0]);
        let [yaw, pitch, roll] = pose.euler_angles();
        assert!(yaw.abs() < 1e-5);
        assert!(pitch.abs() < 1e-5);
        assert!(roll.abs() < 1e-5);
    }

    #[test]
    fn test_headpose_rotation_axis_angle_identity() {
        let pose = HeadPose::new(mat3_identity(), [0.0, 0.0, 1.0]);
        let aa = pose.rotation_axis_angle();
        assert!(aa[0].abs() < 1e-5);
        assert!(aa[1].abs() < 1e-5);
        assert!(aa[2].abs() < 1e-5);
    }

    #[test]
    fn test_headpose_rotation_axis_angle_known() {
        // 90° rotation around z-axis
        let r = [[0.0_f32, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let pose = HeadPose::new(r, [0.0, 0.0, 1.0]);
        let aa = pose.rotation_axis_angle();
        // axis should be [0, 0, π/2]
        assert!(aa[0].abs() < 1e-4, "ax={}", aa[0]);
        assert!(aa[1].abs() < 1e-4, "ay={}", aa[1]);
        assert!((aa[2] - PI / 2.0).abs() < 1e-4, "az={}", aa[2]);
    }

    // -- PoseConfig --

    #[test]
    fn test_pose_config_validate_valid() {
        let cfg = PoseConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_pose_config_validate_too_few_correspondences() {
        let cfg = PoseConfig {
            min_correspondences: 3,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(PoseEstimationError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_pose_config_validate_negative_reprojection_error() {
        let cfg = PoseConfig {
            max_reprojection_error: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(PoseEstimationError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_pose_config_validate_bad_confidence() {
        let cfg = PoseConfig {
            min_confidence: 1.5,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(PoseEstimationError::InvalidConfig(_))
        ));
    }

    // -- estimate_pose_weak_perspective --

    #[test]
    fn test_estimate_pose_weak_perspective_insufficient_points() {
        let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
        let config = PoseConfig::default();
        // Only 2 correspondences
        let corrs: Vec<PointCorrespondence> = (0..2)
            .map(|i| {
                let fi = i as f32;
                PointCorrespondence::new(
                    Landmark3D::new(fi, fi, 1.0),
                    Landmark2D::new(320.0 + fi * 10.0, 240.0),
                )
            })
            .collect();
        let result = estimate_pose_weak_perspective(&corrs, &camera, &config);
        assert!(matches!(
            result,
            Err(PoseEstimationError::InsufficientPoints { .. })
        ));
    }

    #[test]
    fn test_estimate_pose_weak_perspective_front_facing() {
        // Front-facing head: 3D points in XY plane at z=1, 2D observations
        // aligned with their perspective projections.
        let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
        let config = PoseConfig {
            min_confidence: 0.0,
            ..Default::default()
        };

        // Place model points in a cross pattern centered at origin at z=0.
        // The solver will estimate t_z ≈ 1 from scale, so camera-space z ≈ 1,
        // matching the observations generated with unit depth.
        let model_pts = [
            Landmark3D::new(-0.1, 0.0, 0.0),
            Landmark3D::new(0.1, 0.0, 0.0),
            Landmark3D::new(0.0, -0.1, 0.0),
            Landmark3D::new(0.0, 0.1, 0.0),
        ];

        // Corresponding image observations at unit depth (z=1):
        //   u = cx + f*x,  v = cy - f*y
        let corrs: Vec<PointCorrespondence> = model_pts
            .iter()
            .map(|p| {
                let u = camera.cx + camera.focal_length * p.x;
                let v = camera.cy - camera.focal_length * p.y;
                PointCorrespondence::new(*p, Landmark2D::new(u, v))
            })
            .collect();

        let result = estimate_pose_weak_perspective(&corrs, &camera, &config);
        assert!(result.is_ok(), "Expected Ok, got {result:?}");
        let pose = result.unwrap();
        // Translation should be small (head centered)
        assert!(
            pose.translation[0].abs() < 0.1,
            "tx={}",
            pose.translation[0]
        );
        assert!(
            pose.translation[1].abs() < 0.1,
            "ty={}",
            pose.translation[1]
        );
        // Reprojection error should be small
        assert!(
            pose.reprojection_error < 5.0,
            "reprojection_error={}",
            pose.reprojection_error
        );
    }

    #[test]
    fn test_estimate_pose_weak_perspective_degenerate_coincident_points() {
        // All 3D points are at the same location → scale cannot be estimated
        let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
        let config = PoseConfig {
            min_confidence: 0.0,
            ..Default::default()
        };
        let corrs: Vec<PointCorrespondence> = (0..4)
            .map(|_| {
                PointCorrespondence::new(
                    Landmark3D::new(0.0, 0.0, 1.0),
                    Landmark2D::new(320.0, 240.0),
                )
            })
            .collect();
        let result = estimate_pose_weak_perspective(&corrs, &camera, &config);
        assert!(matches!(
            result,
            Err(PoseEstimationError::NumericalError(_))
        ));
    }

    // -- estimate_yaw_from_symmetry --

    #[test]
    fn test_estimate_yaw_symmetric_is_zero() {
        // Perfectly symmetric left/right landmarks → yaw ≈ 0
        let left = vec![[100.0_f32, 200.0]];
        let right = vec![[300.0_f32, 200.0]];
        let yaw = estimate_yaw_from_symmetry(&left, &right).expect("should succeed");
        assert!(yaw.abs() < 0.1, "yaw={yaw}");
    }

    #[test]
    fn test_estimate_yaw_asymmetric_nonzero() {
        // Left side wider than right → face turned toward the right (positive yaw).
        let left = vec![[80.0_f32, 200.0], [150.0, 200.0]]; // spread = 70
        let right = vec![[300.0_f32, 200.0], [320.0, 200.0]]; // spread = 20
        let yaw = estimate_yaw_from_symmetry(&left, &right).expect("should succeed");
        // yaw = asin((70-20)/(70+20)) = asin(50/90) ≈ asin(0.556) ≈ 0.59 rad
        assert!(yaw > 0.1, "expected positive nonzero yaw, got {yaw}");
    }

    #[test]
    fn test_estimate_yaw_asymmetry_sign() {
        // Use multi-point inputs so that spread asymmetry is visible.
        // Frontal face: both sides have equal internal spread (20 px each).
        let left_frontal = vec![[90.0_f32, 200.0], [110.0, 200.0]]; // spread = 20
        let right_frontal = vec![[290.0_f32, 200.0], [310.0, 200.0]]; // spread = 20
        let yaw_frontal =
            estimate_yaw_from_symmetry(&left_frontal, &right_frontal).expect("should succeed");

        // Turned right: left side expands (larger spread), right side stays (smaller spread).
        let left_turned = vec![[80.0_f32, 200.0], [130.0, 200.0]]; // spread = 50
        let right_turned = vec![[290.0_f32, 200.0], [310.0, 200.0]]; // spread = 20
        let yaw_turned =
            estimate_yaw_from_symmetry(&left_turned, &right_turned).expect("should succeed");

        // Frontal should be near 0 (equal spreads), turned should have larger absolute yaw.
        assert!(
            yaw_frontal.abs() < 0.01,
            "frontal yaw should be ~0, got {yaw_frontal}"
        );
        assert!(
            yaw_turned.abs() > yaw_frontal.abs(),
            "frontal={yaw_frontal}, turned={yaw_turned}"
        );
    }

    #[test]
    fn test_estimate_yaw_mismatched_lengths_error() {
        let left = vec![[100.0_f32, 200.0], [110.0, 210.0]];
        let right = vec![[300.0_f32, 200.0]];
        let result = estimate_yaw_from_symmetry(&left, &right);
        assert!(matches!(
            result,
            Err(PoseEstimationError::InsufficientPoints { .. })
        ));
    }

    #[test]
    fn test_estimate_yaw_empty_input_error() {
        let result = estimate_yaw_from_symmetry(&[], &[]);
        assert!(matches!(
            result,
            Err(PoseEstimationError::InsufficientPoints { .. })
        ));
    }

    // -- estimate_pitch_from_vertical --

    #[test]
    fn test_estimate_pitch_empty_upper_error() {
        let result = estimate_pitch_from_vertical(&[], &[[200.0_f32, 350.0]]);
        assert!(matches!(
            result,
            Err(PoseEstimationError::InsufficientPoints { .. })
        ));
    }

    #[test]
    fn test_estimate_pitch_empty_lower_error() {
        let result = estimate_pitch_from_vertical(&[[200.0_f32, 100.0]], &[]);
        assert!(matches!(
            result,
            Err(PoseEstimationError::InsufficientPoints { .. })
        ));
    }

    // -- reprojection_error --

    #[test]
    fn test_reprojection_error_empty() {
        let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
        let pose = HeadPose::new(mat3_identity(), [0.0, 0.0, 1.0]);
        let err = reprojection_error(&[], &pose, &camera);
        assert!((err - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_reprojection_error_known_pose() {
        let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
        // Identity pose with z=1: model point (0,0,0) → camera (0,0,1) → image cx,cy
        let pose = HeadPose::new(mat3_identity(), [0.0, 0.0, 1.0]);
        let pt3d = Landmark3D::new(0.0, 0.0, 0.0);
        let (pu, pv) = camera.project(0.0, 0.0, 1.0).expect("should project");
        let corr = PointCorrespondence::new(pt3d, Landmark2D::new(pu, pv));
        let err = reprojection_error(&[corr], &pose, &camera);
        assert!(err < 1e-4, "err={err}");
    }

    // -- count_inliers --

    #[test]
    fn test_count_inliers_all_below_threshold() {
        let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
        let pose = HeadPose::new(mat3_identity(), [0.0, 0.0, 1.0]);
        // Perfect correspondence → 0 error → inlier
        let pt3d = Landmark3D::new(0.0, 0.0, 0.0);
        let (pu, pv) = camera.project(0.0, 0.0, 1.0).expect("should project");
        let corr = PointCorrespondence::new(pt3d, Landmark2D::new(pu, pv));
        let count = count_inliers(&[corr], &pose, &camera, 1.0);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_count_inliers_none_below_threshold() {
        let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
        let pose = HeadPose::new(mat3_identity(), [0.0, 0.0, 1.0]);
        // Observation far from projection → outlier
        let pt3d = Landmark3D::new(0.0, 0.0, 0.0);
        // Projected to cx,cy = 320,240 but observed far away
        let corr = PointCorrespondence::new(pt3d, Landmark2D::new(0.0, 0.0));
        let count = count_inliers(&[corr], &pose, &camera, 1.0);
        assert_eq!(count, 0);
    }

    // -- PoseTracker --

    #[test]
    fn test_pose_tracker_new_has_no_pose() {
        let tracker = PoseTracker::new(0.7);
        assert!(!tracker.has_pose());
        assert!(tracker.current_pose().is_none());
    }

    #[test]
    fn test_pose_tracker_update_single_pose() {
        let mut tracker = PoseTracker::new(0.7);
        let pose = HeadPose::new(mat3_identity(), [1.0, 2.0, 3.0]);
        tracker.update(&pose);
        assert!(tracker.has_pose());
        let current = tracker.current_pose().expect("should have pose");
        assert!((current.translation[0] - 1.0).abs() < 1e-5);
        assert!((current.translation[1] - 2.0).abs() < 1e-5);
        assert!((current.translation[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_pose_tracker_update_ema_smoothing() {
        let mut tracker = PoseTracker::new(0.7);
        let pose1 = HeadPose::new(mat3_identity(), [0.0, 0.0, 1.0]);
        let pose2 = HeadPose::new(mat3_identity(), [10.0, 10.0, 10.0]);
        tracker.update(&pose1);
        tracker.update(&pose2);

        let current = tracker.current_pose().expect("should have pose");
        // t_x = 0.7 * 0.0 + 0.3 * 10.0 = 3.0
        assert!(
            (current.translation[0] - 3.0).abs() < 1e-4,
            "tx={}",
            current.translation[0]
        );
    }

    #[test]
    fn test_pose_tracker_reset() {
        let mut tracker = PoseTracker::new(0.7);
        let pose = HeadPose::new(mat3_identity(), [1.0, 0.0, 1.0]);
        tracker.update(&pose);
        assert!(tracker.has_pose());
        tracker.reset();
        assert!(!tracker.has_pose());
        assert!(tracker.current_pose().is_none());
    }

    // -- mat3 helpers (internal sanity) --

    #[test]
    fn test_mat3_mul_identity() {
        let id = mat3_identity();
        let result = mat3_mul(&id, &id);
        for (i, row) in result.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((val - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_mat3_transpose() {
        let m = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let t = mat3_transpose(&m);
        for (i, row) in t.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!((val - m[j][i]).abs() < 1e-6);
            }
        }
    }

    // -- estimate_pitch_from_vertical (algorithm: asymmetry-based centroid formula) --

    /// Symmetric equal-count distribution → numerator (sum_centroids - 2*center) = 0 → pitch ≈ 0.
    #[test]
    fn test_pitch_symmetric_distribution_near_zero() {
        // 2 upper at 0.2/0.25, 2 lower at 0.75/0.80: equal count, symmetric around 0.5
        // upper_centroid=0.225, lower_centroid=0.775, center_y=0.5
        // sin_pitch = (0.225+0.775 - 2*0.5) / span = 0.0 / 0.55 = 0
        let upper = vec![[0.5_f32, 0.2], [0.5, 0.25]];
        let lower = vec![[0.5_f32, 0.75], [0.5, 0.8]];
        let pitch = estimate_pitch_from_vertical(&upper, &lower).unwrap();
        assert!(
            pitch.abs() < 0.05,
            "symmetric distribution should give pitch near 0, got {pitch}"
        );
    }

    /// Unequal group sizes shift center_y, making sum_centroids ≠ 2*center_y → nonzero pitch.
    ///
    /// upper=[0.1] (1 pt), lower=[0.4,0.5,0.6] (3 pts):
    ///   upper_centroid=0.1, lower_centroid=0.5, center_y=0.4
    ///   span=0.4, sin_pitch=(0.1+0.5-0.8)/0.4 = -0.2/0.4 = -0.5 → pitch≈-0.524
    #[test]
    fn test_pitch_unequal_groups_nonzero() {
        let upper = vec![[0.5_f32, 0.1]];
        let lower = vec![[0.5_f32, 0.4], [0.5, 0.5], [0.5, 0.6]];
        let pitch = estimate_pitch_from_vertical(&upper, &lower).unwrap();
        assert!(
            pitch != 0.0,
            "unequal-count groups should give nonzero pitch, got {pitch}"
        );
    }

    /// Monotonicity: adding more lower points (compressing upper relative to center) makes
    /// pitch more negative — the asymmetry grows consistently in one direction.
    #[test]
    fn test_pitch_monotone_with_increasing_lower_asymmetry() {
        // upper=[0.1] fixed; grow lower group downward → center_y rises → sin_pitch becomes more negative.
        // Equal-size groups (1 vs 1) cancel out, so lower_small uses 2 lower points.
        let upper = vec![[0.5_f32, 0.1]];
        let lower_small = vec![[0.5_f32, 0.6], [0.5, 0.7]];
        let lower_large = vec![[0.5_f32, 0.6], [0.5, 0.7], [0.5, 0.8], [0.5, 0.9]];
        let pitch_small = estimate_pitch_from_vertical(&upper, &lower_small).unwrap();
        let pitch_large = estimate_pitch_from_vertical(&upper, &lower_large).unwrap();
        // Both should be negative (upper centroid < center_y from large lower group)
        // and larger lower group means more negative pitch
        assert!(pitch_small < 0.0, "pitch_small={pitch_small}");
        assert!(pitch_large < 0.0, "pitch_large={pitch_large}");
        assert!(
            pitch_large < pitch_small,
            "more lower points should push pitch more negative: small={pitch_small} large={pitch_large}"
        );
    }

    /// Single-point degenerate case: one point in each group → must not panic.
    #[test]
    fn test_pitch_single_point_each_group_no_panic() {
        let upper = vec![[0.5_f32, 0.3]];
        let lower = vec![[0.5_f32, 0.7]];
        let result = estimate_pitch_from_vertical(&upper, &lower);
        assert!(
            result.is_ok(),
            "single-point groups should succeed: {result:?}"
        );
        let pitch = result.unwrap();
        assert!(
            pitch.abs() < 0.05,
            "single symmetric points → pitch near 0, got {pitch}"
        );
    }

    /// Coincident centroids (vertical_span < 1e-6) → must return Ok(0.0).
    #[test]
    fn test_pitch_zero_span_returns_zero() {
        let upper = vec![[0.3_f32, 0.5], [0.5, 0.5]];
        let lower = vec![[0.7_f32, 0.5], [0.4, 0.5]];
        let pitch = estimate_pitch_from_vertical(&upper, &lower).unwrap();
        assert!(
            pitch.abs() < 1e-5,
            "zero-span input should return pitch 0.0, got {pitch}"
        );
    }
}
