//! Head pose estimation from 2D facial landmark observations.
//!
//! Recovers FLAME head pose parameters (rotation and translation) from 2D
//! landmark observations using a weak-perspective (scaled orthographic) `PnP`
//! solve with optional RANSAC robustification and quaternion-based temporal
//! smoothing.  The solver recovers the two scaled rotation rows by linear least
//! squares, projects them onto `SO(3)` and derives the translation from the
//! centroid correspondence — rotation **and** translation are estimated, not
//! assumed.

use thiserror::Error;

use crate::rigid_alignment::svd_3x3;

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

// ---------------------------------------------------------------------------
// Utility matrix operations (private)
// ---------------------------------------------------------------------------

/// Multiply two 3×3 matrices: `A * B`.
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

/// Convert a row-major flat 3×3 matrix into nested-array form.
fn mat3_from_flat(m: &[f32; 9]) -> [[f32; 3]; 3] {
    [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]]
}

/// Convert a nested-array 3×3 matrix into row-major flat form.
fn mat3_to_flat(m: &[[f32; 3]; 3]) -> [f32; 9] {
    [
        m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
    ]
}

/// Project an arbitrary 3×3 matrix onto the closest proper rotation (`SO(3)`).
///
/// Polar factor of the SVD, `R = U·Vᵀ`; `svd_3x3` guarantees `det(U) = det(V) =
/// +1`, so the result is proper.  A fully degenerate input yields the identity.
fn orthonormalize_rotation(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let (u, sv, vt) = svd_3x3(&mat3_to_flat(m));
    if sv[0] < 1e-9 {
        return mat3_identity();
    }
    mat3_mul(&mat3_from_flat(&u), &mat3_from_flat(&vt))
}

/// Moore–Penrose pseudo-inverse of a 3×3 matrix.
///
/// Singular values below `1e-5 · σ_max` are treated as zero, so coplanar model
/// points (common for facial landmarks) yield the minimum-norm solution instead
/// of a blow-up along the degenerate direction.
fn mat3_pseudo_inverse(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let (u, sv, vt) = svd_3x3(&mat3_to_flat(m));
    let tol = sv[0] * 1e-5;

    let mut s_inv = [[0.0f32; 3]; 3];
    for (i, row) in s_inv.iter_mut().enumerate() {
        if sv[i] > tol && sv[i] > 0.0 {
            row[i] = 1.0 / sv[i];
        }
    }

    let v_mat = mat3_transpose(&mat3_from_flat(&vt));
    let u_t = mat3_transpose(&mat3_from_flat(&u));
    mat3_mul(&mat3_mul(&v_mat, &s_inv), &u_t)
}

/// Euclidean norm of a 3-vector.
fn vec3_norm(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

// ---------------------------------------------------------------------------
// Quaternion helpers (private) — used for rotation-preserving smoothing
// ---------------------------------------------------------------------------

/// Normalize a quaternion `[w, x, y, z]`; returns the identity for a zero input.
fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n < 1e-12 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

/// Convert a rotation matrix into a unit quaternion `[w, x, y, z]`.
///
/// Uses Shepperd's method (largest-diagonal branch) for numerical stability.
fn mat3_to_quat(r: &[[f32; 3]; 3]) -> [f32; 4] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let q = if trace > 0.0 {
        let s = (trace + 1.0).max(1e-12).sqrt() * 2.0;
        [
            0.25 * s,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        ]
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).max(1e-12).sqrt() * 2.0;
        [
            (r[2][1] - r[1][2]) / s,
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        ]
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).max(1e-12).sqrt() * 2.0;
        [
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
        ]
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).max(1e-12).sqrt() * 2.0;
        [
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
        ]
    };
    quat_normalize(q)
}

/// Convert a unit quaternion `[w, x, y, z]` into a rotation matrix.
fn quat_to_mat3(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [w, x, y, z] = quat_normalize(q);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Spherical linear interpolation between two quaternions.
///
/// `t = 0` returns `q0`, `t = 1` returns `q1`.  Antipodal inputs are handled by
/// negating `q1` so the interpolation always takes the shorter arc.
fn quat_slerp(q0: [f32; 4], q1: [f32; 4], t: f32) -> [f32; 4] {
    let q0 = quat_normalize(q0);
    let mut q1 = quat_normalize(q1);

    let mut dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
    if dot < 0.0 {
        q1 = [-q1[0], -q1[1], -q1[2], -q1[3]];
        dot = -dot;
    }

    // Nearly parallel: slerp is numerically unstable, use normalized lerp.
    let (wa, wb) = if dot > 0.9995 {
        (1.0 - t, t)
    } else {
        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        if sin_theta.abs() < 1e-9 {
            return q0;
        }
        (
            ((1.0 - t) * theta).sin() / sin_theta,
            (t * theta).sin() / sin_theta,
        )
    };
    quat_normalize([
        wa * q0[0] + wb * q1[0],
        wa * q0[1] + wb * q1[1],
        wa * q0[2] + wb * q1[2],
        wa * q0[3] + wb * q1[3],
    ])
}

// ---------------------------------------------------------------------------
// Deterministic RNG for RANSAC sampling (private)
// ---------------------------------------------------------------------------

/// Seed for the RANSAC sampler: deterministic on purpose, so two runs over the
/// same correspondences always produce the same pose.
const RANSAC_SEED: u64 = 0x2545_F491_4F6C_DD1D;

/// Minimal `SplitMix64` generator (deterministic, dependency-free).
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform index in `[0, n)`.  Returns `0` when `n == 0`.
    fn next_index(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// Pose solvers
// ---------------------------------------------------------------------------

/// Estimate head pose using a weak-perspective (scaled orthographic) solve.
///
/// This approach works well when the face occupies a moderate field of view
/// (its depth extent is small compared to its distance from the camera).
/// At least `config.min_correspondences` (≥ 4) correspondences with confidence
/// ≥ `config.min_confidence` are required.
///
/// # Algorithm
///
/// With `x̃ᵢ` the centered model points and `(ũᵢ, ṽᵢ)` the centered
/// observations, the weak-perspective projection is `ũᵢ = s·(r₁·x̃ᵢ)` and
/// `ṽᵢ = −s·(r₂·x̃ᵢ)` (image +v points down), where `r₁`, `r₂` are the first two
/// rotation rows and `s = f / t_z`.
///
/// 1. Compute the model and image centroids.
/// 2. Solve the two 3-parameter least-squares problems for `s·r₁` and `−s·r₂`
///    (pseudo-inverse of `Σ x̃ᵢ x̃ᵢᵀ`, well behaved for coplanar landmark sets).
/// 3. Recover the scale as the mean of the two row norms.
/// 4. Complete the rotation with `r₃ = r₁ × r₂` and project onto `SO(3)` via SVD.
/// 5. Derive the translation from the centroid correspondence: `t_z = f / s`,
///    `t' = ((ū − cx)·t_z/f, −(v̄ − cy)·t_z/f, t_z)` and `t = t' − R·c₃`
///    (`c₃` = model centroid), so an off-origin model centroid is handled.
/// 6. Compute the confidence-weighted reprojection error.
///
/// When `config.ransac_iterations > 0` the solve is wrapped in RANSAC: minimal
/// subsets are sampled, scored with [`count_inliers`] against
/// `config.max_reprojection_error`, and the best consensus set is refit — in
/// which case `reprojection_error` is reported over the final inlier set.
///
/// # Errors
///
/// [`PoseEstimationError::InvalidConfig`] for an invalid configuration or
/// non-positive focal length, [`PoseEstimationError::InsufficientPoints`] when
/// too few correspondences pass the confidence filter, and
/// [`PoseEstimationError::NumericalError`] for degenerate correspondence sets.
pub fn estimate_pose_weak_perspective(
    correspondences: &[PointCorrespondence],
    camera: &PosePinholeCamera,
    config: &PoseConfig,
) -> Result<HeadPose, PoseEstimationError> {
    config.validate()?;

    // Filter by minimum confidence
    let filtered: Vec<PointCorrespondence> = correspondences
        .iter()
        .copied()
        .filter(|c| c.point_2d.confidence >= config.min_confidence)
        .collect();

    let n = filtered.len();
    if n < config.min_correspondences {
        return Err(PoseEstimationError::InsufficientPoints {
            got: n,
            required: config.min_correspondences,
        });
    }

    if config.ransac_iterations > 0 {
        solve_weak_perspective_ransac(&filtered, camera, config)
    } else {
        solve_weak_perspective(&filtered, camera)
    }
}

/// Core weak-perspective solve over **all** supplied correspondences (no
/// configuration handling, no confidence filtering).
fn solve_weak_perspective(
    points: &[PointCorrespondence],
    camera: &PosePinholeCamera,
) -> Result<HeadPose, PoseEstimationError> {
    let n = points.len();
    if n < 4 {
        return Err(PoseEstimationError::InsufficientPoints {
            got: n,
            required: 4,
        });
    }
    if camera.focal_length <= 0.0 || !camera.focal_length.is_finite() {
        return Err(PoseEstimationError::InvalidConfig(format!(
            "focal_length must be > 0, got {}",
            camera.focal_length
        )));
    }

    let nf = n as f32;

    // Step 1: centroids
    let centroid_3d = [
        points.iter().map(|c| c.point_3d.x).sum::<f32>() / nf,
        points.iter().map(|c| c.point_3d.y).sum::<f32>() / nf,
        points.iter().map(|c| c.point_3d.z).sum::<f32>() / nf,
    ];
    let centroid_2d_u = points.iter().map(|c| c.point_2d.u).sum::<f32>() / nf;
    let centroid_2d_v = points.iter().map(|c| c.point_2d.v).sum::<f32>() / nf;

    // Step 2: normal equations for the two scaled rotation rows.
    let mut normal_mat = [[0.0f32; 3]; 3];
    let mut rhs_u = [0.0f32; 3];
    let mut rhs_v = [0.0f32; 3];

    for c in points {
        let x = [
            c.point_3d.x - centroid_3d[0],
            c.point_3d.y - centroid_3d[1],
            c.point_3d.z - centroid_3d[2],
        ];
        let du = c.point_2d.u - centroid_2d_u;
        let dv = c.point_2d.v - centroid_2d_v;

        for (i, row) in normal_mat.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell += x[i] * x[j];
            }
        }
        for (i, r) in rhs_u.iter_mut().enumerate() {
            *r += du * x[i];
        }
        for (i, r) in rhs_v.iter_mut().enumerate() {
            *r += dv * x[i];
        }
    }

    let pinv = mat3_pseudo_inverse(&normal_mat);
    let row_u = mat3_vec3_mul(&pinv, rhs_u); //  s · r₁
    let row_v = mat3_vec3_mul(&pinv, rhs_v); // −s · r₂

    let norm_u = vec3_norm(row_u);
    let norm_v = vec3_norm(row_v);
    if norm_u < 1e-9 || norm_v < 1e-9 {
        return Err(PoseEstimationError::NumericalError(
            "Degenerate correspondence set; weak-perspective scale cannot be estimated".to_string(),
        ));
    }

    // Step 3: scale = mean of the two scaled-row norms
    let scale = 0.5 * (norm_u + norm_v);

    // Step 4: rotation rows, completed and re-orthonormalized
    let r1 = [row_u[0] / norm_u, row_u[1] / norm_u, row_u[2] / norm_u];
    let r2 = [-row_v[0] / norm_v, -row_v[1] / norm_v, -row_v[2] / norm_v];
    let r3 = [
        r1[1] * r2[2] - r1[2] * r2[1],
        r1[2] * r2[0] - r1[0] * r2[2],
        r1[0] * r2[1] - r1[1] * r2[0],
    ];
    let rotation = orthonormalize_rotation(&[r1, r2, r3]);

    // Step 5: translation from the centroid correspondence
    let t_z = camera.focal_length / scale;
    let t_prime = [
        (centroid_2d_u - camera.cx) * t_z / camera.focal_length,
        -(centroid_2d_v - camera.cy) * t_z / camera.focal_length,
        t_z,
    ];
    let rotated_centroid = mat3_vec3_mul(&rotation, centroid_3d);
    let translation = [
        t_prime[0] - rotated_centroid[0],
        t_prime[1] - rotated_centroid[1],
        t_prime[2] - rotated_centroid[2],
    ];

    let mut pose = HeadPose::new(rotation, translation);

    // Step 6: reprojection error over the supplied correspondences
    pose.reprojection_error = reprojection_error(points, &pose, camera);

    Ok(pose)
}

/// RANSAC wrapper around [`solve_weak_perspective`]: samples
/// `config.ransac_iterations` minimal subsets, scores each with
/// [`count_inliers`] at `config.max_reprojection_error`, then refits on the best
/// consensus set.
fn solve_weak_perspective_ransac(
    points: &[PointCorrespondence],
    camera: &PosePinholeCamera,
    config: &PoseConfig,
) -> Result<HeadPose, PoseEstimationError> {
    let n = points.len();
    let sample_size = config.min_correspondences.max(4).min(n);
    if sample_size < 4 {
        return Err(PoseEstimationError::InsufficientPoints {
            got: n,
            required: 4,
        });
    }

    let mut rng = SplitMix64::new(RANSAC_SEED);
    let mut order: Vec<usize> = (0..n).collect();
    let mut best_pose: Option<HeadPose> = None;
    let mut best_inliers = 0usize;

    for _ in 0..config.ransac_iterations {
        // Partial Fisher–Yates shuffle: the first `sample_size` entries of
        // `order` become a uniform subset drawn without replacement.
        for i in 0..sample_size {
            let j = i + rng.next_index(n - i);
            order.swap(i, j);
        }
        let subset: Vec<PointCorrespondence> =
            order[..sample_size].iter().map(|&i| points[i]).collect();

        let Ok(candidate) = solve_weak_perspective(&subset, camera) else {
            continue;
        };

        let inliers = count_inliers(points, &candidate, camera, config.max_reprojection_error);
        if inliers > best_inliers {
            best_inliers = inliers;
            best_pose = Some(candidate);
        }
    }

    // No candidate could be fitted at all (every sampled subset was degenerate):
    // fall back to a plain fit over the full correspondence set.
    let Some(mut best) = best_pose else {
        return solve_weak_perspective(points, camera);
    };

    let inlier_points: Vec<PointCorrespondence> = points
        .iter()
        .copied()
        .filter(|c| is_inlier(c, &best, camera, config.max_reprojection_error))
        .collect();

    if inlier_points.len() < config.min_correspondences.max(4) {
        // Consensus set too small to refit: keep the best sampled model but
        // report its error over every correspondence.
        let error = reprojection_error(points, &best, camera);
        best.reprojection_error = error;
        return Ok(best);
    }

    solve_weak_perspective(&inlier_points, camera)
}

/// Estimate yaw angle from the horizontal asymmetry of symmetric landmark pairs.
///
/// Each element of `left_points` and `right_points` is a `[u, v]` image
/// coordinate; the arrays must have the same length and at least one element.
/// Returns the estimated yaw in radians (positive = turned right).
///
/// Standalone heuristic for callers who only have the two landmark groups —
/// [`estimate_pose_weak_perspective`] recovers yaw as part of a full rotation
/// and needs no initial guess from here.
///
/// # Errors
///
/// [`PoseEstimationError::InsufficientPoints`] if either slice is empty or the
/// slices differ in length.
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

/// Estimate head pitch from the vertical foreshortening of two landmark groups.
///
/// `upper_points` and `lower_points` are `[u, v]` image coordinates (v
/// increases downward); only the group **centroids** are used, so the two
/// groups may have different sizes without biasing the result.
///
/// ## Geometry
///
/// Let `Δy = upper.y − lower.y`, `Δz = upper.z − lower.z` be the model-space
/// offset between the group centroids and `θ` a rotation about the camera
/// x-axis (`y' = y·cos θ − z·sin θ`).  The observed camera-space separation is
///
/// ```text
/// Δy_cam = (v_lower − v_upper) / scale = Δy·cos θ − Δz·sin θ = A·cos(θ + φ)
/// ```
///
/// with `A = hypot(Δy, Δz)`, `φ = atan2(Δz, Δy)`, so `θ = ±acos(Δy_cam/A) − φ`.
///
/// ## Return value
///
/// The two candidate pitch angles in radians, ascending.  A single
/// foreshortening measurement genuinely cannot distinguish them (they collapse
/// to `±θ` when both groups sit at the same model depth, `Δz = 0`), so both are
/// returned rather than one being picked silently.  Use
/// [`select_pitch_candidate`] with a prior (e.g. the previous frame's pitch), or
/// [`estimate_pose_weak_perspective`] on the full landmark set for a fully
/// disambiguated rotation.
///
/// # Errors
///
/// [`PoseEstimationError::InsufficientPoints`] if either slice is empty, and
/// [`PoseEstimationError::InvalidConfig`] when the reference scale is not
/// positive and finite or the two reference centroids coincide.
pub fn estimate_pitch_from_vertical(
    upper_points: &[[f32; 2]],
    lower_points: &[[f32; 2]],
    reference: &PitchReference,
) -> Result<[f32; 2], PoseEstimationError> {
    if upper_points.is_empty() || lower_points.is_empty() {
        return Err(PoseEstimationError::InsufficientPoints {
            got: upper_points.len().min(lower_points.len()),
            required: 1,
        });
    }
    if !reference.scale.is_finite() || reference.scale <= 0.0 {
        return Err(PoseEstimationError::InvalidConfig(format!(
            "PitchReference::scale must be a positive finite value, got {}",
            reference.scale
        )));
    }

    let n_upper = upper_points.len() as f32;
    let n_lower = lower_points.len() as f32;
    let upper_v = upper_points.iter().map(|p| p[1]).sum::<f32>() / n_upper;
    let lower_v = lower_points.iter().map(|p| p[1]).sum::<f32>() / n_lower;

    // Image v grows downward while camera y grows upward, hence the flip.
    let dy_cam = (lower_v - upper_v) / reference.scale;

    let dy = reference.upper_3d.y - reference.lower_3d.y;
    let dz = reference.upper_3d.z - reference.lower_3d.z;
    let amplitude = dy.hypot(dz);
    if amplitude < 1e-6 {
        return Err(PoseEstimationError::InvalidConfig(
            "PitchReference upper_3d and lower_3d coincide; pitch is unobservable".to_string(),
        ));
    }

    let phi = dz.atan2(dy);
    let base = (dy_cam / amplitude).clamp(-1.0, 1.0).acos();

    let first = base - phi;
    let second = -base - phi;
    Ok(if first <= second {
        [first, second]
    } else {
        [second, first]
    })
}

/// Pick the pitch candidate closest to `prior` (radians).
///
/// Companion to [`estimate_pitch_from_vertical`]: pass the previous frame's
/// pitch when tracking, or `0.0` to prefer the solution nearest frontal.
#[must_use]
pub fn select_pitch_candidate(candidates: [f32; 2], prior: f32) -> f32 {
    if (candidates[0] - prior).abs() <= (candidates[1] - prior).abs() {
        candidates[0]
    } else {
        candidates[1]
    }
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

/// Return `true` when the correspondence reprojects within `threshold` pixels
/// (points that do not project in front of the camera are never inliers).
fn is_inlier(
    correspondence: &PointCorrespondence,
    pose: &HeadPose,
    camera: &PosePinholeCamera,
    threshold: f32,
) -> bool {
    pose.project_point(correspondence.point_3d, camera)
        .is_some_and(|(proj_u, proj_v)| {
            let du = proj_u - correspondence.point_2d.u;
            let dv = proj_v - correspondence.point_2d.v;
            (du * du + dv * dv).sqrt() < threshold
        })
}

/// Count the number of correspondences whose reprojection error is below
/// `threshold` (in pixels).
///
/// Used by [`estimate_pose_weak_perspective`] to score RANSAC hypotheses, and
/// available to callers running their own robust loop.
#[must_use]
pub fn count_inliers(
    correspondences: &[PointCorrespondence],
    pose: &HeadPose,
    camera: &PosePinholeCamera,
    threshold: f32,
) -> usize {
    correspondences
        .iter()
        .filter(|c| is_inlier(c, pose, camera, threshold))
        .count()
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
        assert!((rx - orig_3d.0).abs() < 1e-4, "x: {rx} vs {}", orig_3d.0);
        assert!((ry - orig_3d.1).abs() < 1e-4, "y: {ry} vs {}", orig_3d.1);
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
        assert!(vec3_norm(pose.rotation_axis_angle()) < 1e-5);
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
    fn test_pose_config_validate_rejects_bad_fields() {
        let bad = [
            PoseConfig {
                min_correspondences: 3,
                ..Default::default()
            },
            PoseConfig {
                max_reprojection_error: -1.0,
                ..Default::default()
            },
            PoseConfig {
                min_confidence: 1.5,
                ..Default::default()
            },
        ];
        for cfg in &bad {
            assert!(matches!(
                cfg.validate(),
                Err(PoseEstimationError::InvalidConfig(_))
            ));
        }
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
        // Front-facing head: cross of model points at z=0 observed at unit
        // depth (u = cx + f*x, v = cy - f*y), so t_z ≈ 1 and R ≈ I.
        let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
        let config = PoseConfig {
            min_confidence: 0.0,
            ..Default::default()
        };
        let model_pts = [
            Landmark3D::new(-0.1, 0.0, 0.0),
            Landmark3D::new(0.1, 0.0, 0.0),
            Landmark3D::new(0.0, -0.1, 0.0),
            Landmark3D::new(0.0, 0.1, 0.0),
        ];
        let corrs: Vec<PointCorrespondence> = model_pts
            .iter()
            .map(|p| {
                let u = camera.cx + camera.focal_length * p.x;
                let v = camera.cy - camera.focal_length * p.y;
                PointCorrespondence::new(*p, Landmark2D::new(u, v))
            })
            .collect();

        let pose = estimate_pose_weak_perspective(&corrs, &camera, &config)
            .expect("front-facing solve should succeed");
        let (tx, ty) = (pose.translation[0], pose.translation[1]);
        assert!(tx.abs() < 0.1 && ty.abs() < 0.1, "tx={tx} ty={ty}");
        let err = pose.reprojection_error;
        assert!(err < 5.0, "reprojection_error={err}");
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
        // Frontal face: both sides have equal internal spread (20 px each).
        let left_frontal = [[90.0_f32, 200.0], [110.0, 200.0]];
        let right_frontal = [[290.0_f32, 200.0], [310.0, 200.0]];
        let yaw_frontal =
            estimate_yaw_from_symmetry(&left_frontal, &right_frontal).expect("should succeed");

        // Turned right: the left side expands (spread 50) while the right stays.
        let left_turned = [[80.0_f32, 200.0], [130.0, 200.0]];
        let yaw_turned =
            estimate_yaw_from_symmetry(&left_turned, &right_frontal).expect("should succeed");

        assert!(yaw_frontal.abs() < 0.01, "frontal yaw={yaw_frontal}");
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

    /// Upper group 0.12 above and 0.03 in front of the lower group, at 250 px
    /// per model unit.
    fn pitch_reference() -> PitchReference {
        PitchReference::new(
            Landmark3D::new(0.0, 0.06, 0.05),
            Landmark3D::new(0.0, -0.06, 0.02),
            250.0,
        )
    }

    /// Weak-perspective projection (`v = −scale·y_cam`; the constant `cy`
    /// cancels because only the difference is used) of the two reference group
    /// centroids after a pitch rotation of `theta` about the camera x-axis.
    fn project_pitched_groups(theta: f32, reference: &PitchReference) -> ([f32; 2], [f32; 2]) {
        let rot = |p: Landmark3D| p.y * theta.cos() - p.z * theta.sin();
        let upper_y = rot(reference.upper_3d);
        let lower_y = rot(reference.lower_3d);
        (
            [10.0, -reference.scale * upper_y],
            [10.0, -reference.scale * lower_y],
        )
    }

    #[test]
    fn test_estimate_pitch_empty_upper_error() {
        let result = estimate_pitch_from_vertical(&[], &[[200.0_f32, 350.0]], &pitch_reference());
        assert!(matches!(
            result,
            Err(PoseEstimationError::InsufficientPoints { .. })
        ));
    }

    #[test]
    fn test_estimate_pitch_empty_lower_error() {
        let result = estimate_pitch_from_vertical(&[[200.0_f32, 100.0]], &[], &pitch_reference());
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
    fn test_count_inliers_threshold_split() {
        let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
        let pose = HeadPose::new(mat3_identity(), [0.0, 0.0, 1.0]);
        let pt3d = Landmark3D::new(0.0, 0.0, 0.0);
        let (pu, pv) = camera.project(0.0, 0.0, 1.0).expect("should project");
        // Perfect correspondence → 0 error → inlier
        let good = PointCorrespondence::new(pt3d, Landmark2D::new(pu, pv));
        assert_eq!(count_inliers(&[good], &pose, &camera, 1.0), 1);
        // Observation far from the projection (cx, cy) → outlier
        let bad = PointCorrespondence::new(pt3d, Landmark2D::new(0.0, 0.0));
        assert_eq!(count_inliers(&[bad], &pose, &camera, 1.0), 0);
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
    fn test_mat3_mul_and_transpose() {
        let id = mat3_identity();
        let m = [[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let product = mat3_mul(&id, &m);
        let transposed = mat3_transpose(&m);
        for (i, row) in m.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!((product[i][j] - val).abs() < 1e-6, "I*M != M");
                assert!((transposed[j][i] - val).abs() < 1e-6, "transpose");
            }
        }
    }

    #[test]
    fn test_mat3_flat_roundtrip_and_pseudo_inverse() {
        let m = [[2.0_f32, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]];
        assert_eq!(mat3_from_flat(&mat3_to_flat(&m)), m);
        // Rank-2 input: the pseudo-inverse inverts the non-zero directions and
        // leaves the degenerate one at zero.
        let pinv = mat3_pseudo_inverse(&m);
        assert!((pinv[0][0] - 0.5).abs() < 1e-5, "{pinv:?}");
        assert!((pinv[1][1] - 0.25).abs() < 1e-5, "{pinv:?}");
        assert!(pinv[2][2].abs() < 1e-5, "{pinv:?}");
    }

    // -- estimate_pitch_from_vertical (geometric foreshortening solve) --

    /// Smallest distance between `theta` and either returned candidate.
    fn pitch_candidate_error(candidates: [f32; 2], theta: f32) -> f32 {
        (candidates[0] - theta)
            .abs()
            .min((candidates[1] - theta).abs())
    }

    /// Synthetically rotated landmarks must be recovered at known angles
    /// (including the frontal case), with the candidates sorted ascending.
    #[test]
    fn test_pitch_recovers_known_rotation_both_signs() {
        let reference = pitch_reference();
        for &theta in &[-0.45_f32, -0.3, -0.1, 0.0, 0.1, 0.3, 0.45] {
            let (upper, lower) = project_pitched_groups(theta, &reference);
            let cands = estimate_pitch_from_vertical(&[upper], &[lower], &reference)
                .expect("pitch solve should succeed");
            let err = pitch_candidate_error(cands, theta);
            assert!(err < 1e-3, "theta={theta}: candidates={cands:?} err={err}");
            assert!(cands[0] <= cands[1], "unsorted candidates: {cands:?}");
        }
    }

    /// Regression for the cardinality bug: the previous implementation returned
    /// exactly 0 whenever both groups had the same number of points, whatever
    /// the actual geometry, and otherwise measured only the size imbalance.
    #[test]
    fn test_pitch_uses_geometry_not_group_cardinality() {
        let reference = pitch_reference();
        let theta = 0.25_f32;
        let (upper_c, lower_c) = project_pitched_groups(theta, &reference);

        // Two points per group, spread symmetrically about each centroid.
        let upper = [[8.0_f32, upper_c[1] - 3.0], [12.0, upper_c[1] + 3.0]];
        let lower = [[8.0_f32, lower_c[1] - 5.0], [12.0, lower_c[1] + 5.0]];
        let equal = estimate_pitch_from_vertical(&upper, &lower, &reference)
            .expect("pitch solve should succeed");
        assert!(
            pitch_candidate_error(equal, theta) < 1e-3,
            "equal-size groups must still recover theta={theta}, got {equal:?}"
        );
        assert!(
            equal.iter().all(|c| c.abs() > 1e-2),
            "a rotated face must not report a zero pitch: {equal:?}"
        );

        // Same centroids, different group sizes → identical answer.
        let lower_many = [
            lower_c,
            [lower_c[0], lower_c[1] - 7.0],
            [lower_c[0], lower_c[1] + 7.0],
        ];
        let uneven = estimate_pitch_from_vertical(&[upper_c], &lower_many, &reference)
            .expect("pitch solve should succeed");
        assert!(
            (equal[0] - uneven[0]).abs() < 1e-4 && (equal[1] - uneven[1]).abs() < 1e-4,
            "group cardinality must not change the estimate: {equal:?} vs {uneven:?}"
        );
    }

    #[test]
    fn test_pitch_invalid_reference_errors() {
        let (upper, lower) = ([[0.0_f32, 0.0]], [[0.0_f32, 30.0]]);
        let up = Landmark3D::new(0.0, 0.06, 0.05);
        let low = Landmark3D::new(0.0, -0.06, 0.02);
        for bad in [
            PitchReference::new(up, low, 0.0),  // non-positive scale
            PitchReference::new(up, up, 250.0), // coincident centroids
        ] {
            assert!(matches!(
                estimate_pitch_from_vertical(&upper, &lower, &bad),
                Err(PoseEstimationError::InvalidConfig(_))
            ));
        }
    }

    #[test]
    fn test_select_pitch_candidate_prefers_prior() {
        let candidates = [-0.79_f32, 0.3];
        assert!((select_pitch_candidate(candidates, 0.35) - 0.3).abs() < 1e-6);
        assert!((select_pitch_candidate(candidates, -0.9) + 0.79).abs() < 1e-6);
    }

    // -- rotation recovery (weak-perspective solver) --

    /// Build a rotation about the model y-axis.
    fn rot_y(theta: f32) -> [[f32; 3]; 3] {
        let (s, c) = theta.sin_cos();
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    }

    /// Non-planar model point set (full-rank scatter matrix), centred at origin.
    fn spatial_model() -> [Landmark3D; 6] {
        [
            Landmark3D::new(-0.1, 0.0, 0.0),
            Landmark3D::new(0.1, 0.0, 0.0),
            Landmark3D::new(0.0, -0.1, 0.0),
            Landmark3D::new(0.0, 0.1, 0.0),
            Landmark3D::new(0.0, 0.0, 0.08),
            Landmark3D::new(0.0, 0.0, -0.08),
        ]
    }

    #[test]
    fn test_weak_perspective_recovers_rotation_not_identity() {
        let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
        let config = PoseConfig {
            min_confidence: 0.0,
            ..Default::default()
        };
        let theta = 25.0_f32.to_radians();
        let r_true = rot_y(theta);
        let t_true = [0.0_f32, 0.0, 2.0];
        let scale = camera.focal_length / t_true[2];

        // Weak-perspective observations of the rotated model.
        let corrs: Vec<PointCorrespondence> = spatial_model()
            .iter()
            .map(|p| {
                let rp = mat3_vec3_mul(&r_true, [p.x, p.y, p.z]);
                let u = camera.cx + scale * (rp[0] + t_true[0]);
                let v = camera.cy - scale * (rp[1] + t_true[1]);
                PointCorrespondence::new(*p, Landmark2D::new(u, v))
            })
            .collect();

        let pose =
            estimate_pose_weak_perspective(&corrs, &camera, &config).expect("solve should succeed");

        for (i, row) in r_true.iter().enumerate() {
            for (j, &expected) in row.iter().enumerate() {
                assert!(
                    (pose.rotation[i][j] - expected).abs() < 2e-3,
                    "R[{i}][{j}] = {} expected {expected}",
                    pose.rotation[i][j]
                );
            }
        }
        // The out-of-plane term must be non-zero — the old solver returned identity.
        let out_of_plane = pose.rotation[0][2].abs();
        assert!(
            out_of_plane > 0.3,
            "rotation still ~identity: {out_of_plane}"
        );
        let t_z = pose.translation[2];
        assert!((t_z - 2.0).abs() < 5e-3, "t_z={t_z}");
        // The observations are weak-perspective while the error uses the exact
        // pinhole projection, so a sub-pixel residual is expected.
        let err = pose.reprojection_error;
        assert!(err < 2.0, "reprojection_error={err}");
    }

    #[test]
    fn test_weak_perspective_translation_with_depth_and_offset_centroid() {
        // Model centroid at (0.05, 0, 0.05) — deliberately not the origin — and
        // a true depth of 1.95 so that t_z != 1.  All points share one depth,
        // making the perspective projection exactly weak-perspective.
        let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
        let config = PoseConfig {
            min_confidence: 0.0,
            ..Default::default()
        };
        let model = [
            Landmark3D::new(-0.05, 0.0, 0.05),
            Landmark3D::new(0.15, 0.0, 0.05),
            Landmark3D::new(0.05, -0.1, 0.05),
            Landmark3D::new(0.05, 0.1, 0.05),
        ];
        let t_true = [0.4_f32, -0.2, 1.95];

        let corrs: Vec<PointCorrespondence> = model
            .iter()
            .map(|p| {
                let cam = [p.x + t_true[0], p.y + t_true[1], p.z + t_true[2]];
                let (u, v) = camera
                    .project(cam[0], cam[1], cam[2])
                    .expect("model point must be in front of the camera");
                PointCorrespondence::new(*p, Landmark2D::new(u, v))
            })
            .collect();

        let pose =
            estimate_pose_weak_perspective(&corrs, &camera, &config).expect("solve should succeed");

        // The pre-fix formula dropped the t_z factor and the model centroid,
        // which yielded t_x = 0.2 / t_y = -0.1 here instead of 0.4 / -0.2.
        for (axis, (&got, &want)) in pose.translation.iter().zip(t_true.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "t[{axis}]={got}, expected {want}"
            );
        }
        let err = pose.reprojection_error;
        assert!(err < 1.0, "reprojection_error={err}");
    }

    // -- RANSAC --

    #[test]
    fn test_ransac_rejects_gross_outliers() {
        let camera = PosePinholeCamera::new(500.0, 320.0, 240.0);
        let t_true = [0.0_f32, 0.0, 2.0];
        let scale = camera.focal_length / t_true[2];

        // Ten clean correspondences (identity rotation) on a non-planar ring …
        let mut corrs: Vec<PointCorrespondence> = Vec::new();
        for i in 0..10 {
            let angle = i as f32 * std::f32::consts::TAU / 10.0;
            let p = Landmark3D::new(
                0.08 * angle.cos(),
                0.08 * angle.sin(),
                0.02 * (2.0 * angle).cos(),
            );
            let u = camera.cx + scale * p.x;
            let v = camera.cy - scale * p.y;
            corrs.push(PointCorrespondence::new(p, Landmark2D::new(u, v)));
        }
        // … plus three wildly wrong observations.
        for i in 0..3 {
            let p = Landmark3D::new(0.05, -0.05 + i as f32 * 0.01, 0.0);
            corrs.push(PointCorrespondence::new(
                p,
                Landmark2D::new(40.0 + i as f32 * 12.0, 430.0),
            ));
        }

        let plain = PoseConfig {
            min_confidence: 0.0,
            ..Default::default()
        };
        let robust = PoseConfig {
            min_confidence: 0.0,
            ransac_iterations: 64,
            max_reprojection_error: 5.0,
            ..Default::default()
        };

        let plain_err = estimate_pose_weak_perspective(&corrs, &camera, &plain)
            .expect("solve should succeed")
            .reprojection_error;
        let pose_robust =
            estimate_pose_weak_perspective(&corrs, &camera, &robust).expect("solve should succeed");
        let robust_err = pose_robust.reprojection_error;
        let robust_tz = pose_robust.translation[2];

        assert!(robust_err < plain_err, "RANSAC {robust_err} vs {plain_err}");
        assert!(
            (robust_tz - t_true[2]).abs() < 0.1,
            "robust t_z={robust_tz}"
        );
        // Deterministic: the sampler is seeded, so repeated runs agree exactly.
        let again =
            estimate_pose_weak_perspective(&corrs, &camera, &robust).expect("solve should succeed");
        assert!((again.reprojection_error - robust_err).abs() < 1e-6);
    }

    // -- euler gimbal lock --

    /// Build `R = Rz(yaw)·Ry(pitch)·Rx(roll)` (the convention `euler_angles` inverts).
    fn rot_zyx(yaw: f32, pitch: f32, roll: f32) -> [[f32; 3]; 3] {
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let (sr, cr) = roll.sin_cos();
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    }

    /// At both gimbal-lock poles the recovered yaw must keep its sign; the
    /// pre-fix code mirrored it at pitch = −π/2 (a +30° turn read as −30°).
    #[test]
    fn test_euler_gimbal_lock_yaw_sign_both_poles() {
        let yaw_true = 30.0_f32.to_radians();
        for &pitch_true in &[PI / 2.0, -PI / 2.0] {
            let r = rot_zyx(yaw_true, pitch_true, 0.0);
            let [yaw, pitch, roll] = HeadPose::new(r, [0.0, 0.0, 1.0]).euler_angles();
            assert!((pitch - pitch_true).abs() < 1e-3, "pitch={pitch}");
            assert!((yaw - yaw_true).abs() < 1e-3, "yaw={yaw} at {pitch_true}");
            assert!(roll.abs() < 1e-6);
        }
    }

    // -- PoseTracker rotation stays in SO(3) --

    fn mat3_det(m: &[[f32; 3]; 3]) -> f32 {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    #[test]
    fn test_pose_tracker_smoothed_rotation_is_orthonormal() {
        let mut tracker = PoseTracker::new(0.6);
        let poses = [
            HeadPose::new(mat3_identity(), [0.0, 0.0, 2.0]),
            HeadPose::new(rot_y(40.0_f32.to_radians()), [0.1, 0.0, 2.0]),
            HeadPose::new(rot_y(-35.0_f32.to_radians()), [0.0, 0.1, 2.0]),
            HeadPose::new(rot_y(80.0_f32.to_radians()), [0.0, 0.0, 2.1]),
            // The last two are >180° apart as quaternions (dot < 0), which
            // exercises the antipodal branch of the slerp.
            HeadPose::new(rot_y(170.0_f32.to_radians()), [0.0, 0.0, 2.0]),
            HeadPose::new(rot_y(-170.0_f32.to_radians()), [0.0, 0.0, 2.0]),
        ];

        for pose in &poses {
            tracker.update(pose);
            let current = tracker.current_pose().expect("tracker has a pose");
            let r = current.rotation;
            let rrt = mat3_mul(&r, &mat3_transpose(&r));
            for (i, row) in rrt.iter().enumerate() {
                for (j, &value) in row.iter().enumerate() {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!((value - expected).abs() < 1e-5, "RRᵀ[{i}][{j}]={value}");
                }
            }
            let det = mat3_det(&r);
            assert!((det - 1.0).abs() < 1e-5, "det(R)={det}");
            // Euler extraction must stay in range for a proper rotation.
            let [_yaw, pitch, _roll] = current.euler_angles();
            assert!(pitch.is_finite(), "pitch must be finite, got {pitch}");
        }
    }

    #[test]
    fn test_pose_tracker_slerp_moves_toward_observation() {
        let mut tracker = PoseTracker::new(0.5);
        tracker.update(&HeadPose::new(mat3_identity(), [0.0, 0.0, 2.0]));
        tracker.update(&HeadPose::new(
            rot_y(60.0_f32.to_radians()),
            [0.0, 0.0, 2.0],
        ));

        let smoothed = tracker.current_pose().expect("tracker has a pose");
        let aa = smoothed.rotation_axis_angle();
        let magnitude = vec3_norm(aa);
        // Half-way between 0° and 60° on the quaternion sphere.
        assert!(
            (magnitude - 30.0_f32.to_radians()).abs() < 1e-3,
            "expected ~30°, got {magnitude} rad"
        );
    }
}
