//! ArcballCamera implementation for interactive orbit controls.
//!
//! Pure-Rust, no GPU, no winit dependency. All matrices are column-major 4×4
//! arrays indexed as `mat[col * 4 + row]`.

use std::f32::consts::PI;

/// 3D position as [x, y, z].
pub type Vec3 = [f32; 3];

/// 4×4 column-major transformation matrix.
pub type Mat4 = [f32; 16];

/// Quaternion [x, y, z, w].
pub type Quat = [f32; 4];

// ---------------------------------------------------------------------------
// Vec3 helpers
// ---------------------------------------------------------------------------

/// Subtract two Vec3 values.
pub fn vec3_sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Add two Vec3 values.
pub fn vec3_add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scale a Vec3 by a scalar.
pub fn vec3_scale(a: Vec3, s: f32) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Dot product of two Vec3 values.
pub fn vec3_dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product of two Vec3 values.
pub fn vec3_cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Length of a Vec3.
pub fn vec3_length(a: Vec3) -> f32 {
    vec3_dot(a, a).sqrt()
}

/// Normalize a Vec3. Returns zero vector if input is (near-)zero.
pub fn vec3_normalize(a: Vec3) -> Vec3 {
    let len = vec3_length(a);
    if len < 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        vec3_scale(a, 1.0 / len)
    }
}

// ---------------------------------------------------------------------------
// Mat4 helpers (column-major: mat[col * 4 + row])
// ---------------------------------------------------------------------------

/// Multiply two column-major 4×4 matrices: result = a * b.
pub fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0f32;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    out
}

/// Build a look-at view matrix (column-major).
///
/// If `eye` and `center` are coincident, the forward vector degenerates to
/// zero; in that case the identity matrix is returned to avoid NaN.
pub fn look_at(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    let f = vec3_normalize(vec3_sub(center, eye)); // forward
    if vec3_length(f) < 1e-12 {
        return [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
    }
    let r = vec3_normalize(vec3_cross(f, up)); // right
    let u = vec3_cross(r, f); // camera-local up

    // Column-major layout:
    //   col 0: [r.x,  u.x, -f.x, 0]
    //   col 1: [r.y,  u.y, -f.y, 0]
    //   col 2: [r.z,  u.z, -f.z, 0]
    //   col 3: [-dot(r,eye), -dot(u,eye), dot(f,eye), 1]
    #[rustfmt::skip]
    let m = [
        // col 0
        r[0],  u[0], -f[0], 0.0,
        // col 1
        r[1],  u[1], -f[1], 0.0,
        // col 2
        r[2],  u[2], -f[2], 0.0,
        // col 3
        -vec3_dot(r, eye), -vec3_dot(u, eye), vec3_dot(f, eye), 1.0,
    ];
    m
}

/// Build a perspective projection matrix (column-major, OpenGL convention,
/// maps depth to [-1, 1]).
pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    let nf = 1.0 / (near - far);

    #[rustfmt::skip]
    let m = [
        // col 0
        f / aspect, 0.0, 0.0, 0.0,
        // col 1
        0.0, f, 0.0, 0.0,
        // col 2
        0.0, 0.0, (far + near) * nf, -1.0,
        // col 3
        0.0, 0.0, 2.0 * far * near * nf, 0.0,
    ];
    m
}

// ---------------------------------------------------------------------------
// ArcballCamera
// ---------------------------------------------------------------------------

/// Camera state for arcball orbit control.
///
/// Implements spherical-coordinate orbit around a target point with dolly
/// (zoom), pan, and look-at-bounds helpers.
#[derive(Debug, Clone, PartialEq)]
pub struct ArcballCamera {
    /// Look-at target in world space.
    pub target: Vec3,
    /// Distance from target (radius).
    pub distance: f32,
    /// Horizontal angle in radians (Y-up convention).
    pub yaw: f32,
    /// Vertical angle in radians, clamped to [-89°, 89°].
    pub pitch: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near clip plane distance.
    pub near: f32,
    /// Far clip plane distance.
    pub far: f32,
    /// Minimum zoom distance.
    pub min_distance: f32,
    /// Maximum zoom distance.
    pub max_distance: f32,
}

impl ArcballCamera {
    /// Construct a new ArcballCamera with the given orbital parameters.
    #[must_use]
    pub fn new(target: Vec3, distance: f32, yaw: f32, pitch: f32) -> Self {
        Self {
            target,
            distance,
            yaw,
            pitch,
            fov_y: PI / 4.0,
            near: 0.01,
            far: 100.0,
            min_distance: 0.1,
            max_distance: 50.0,
        }
    }

    /// Return the world-space camera position computed from spherical coords.
    #[must_use]
    pub fn position(&self) -> Vec3 {
        [
            self.target[0] + self.distance * self.pitch.cos() * self.yaw.sin(),
            self.target[1] + self.distance * self.pitch.sin(),
            self.target[2] + self.distance * self.pitch.cos() * self.yaw.cos(),
        ]
    }

    /// Orbit around the target by the given angular deltas.
    ///
    /// Yaw is unclamped. Pitch is clamped to ±89°.
    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        let max_pitch = 89.0_f32.to_radians();
        self.pitch = (self.pitch + delta_pitch).clamp(-max_pitch, max_pitch);
    }

    /// Dolly (zoom) by moving along the radial axis.
    pub fn dolly(&mut self, delta: f32) {
        self.distance = (self.distance + delta).clamp(self.min_distance, self.max_distance);
    }

    /// Pan the target in camera-local right/up directions.
    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let pos = self.position();
        let forward = vec3_normalize(vec3_sub(self.target, pos));
        let world_up: Vec3 = [0.0, 1.0, 0.0];
        let right = vec3_normalize(vec3_cross(forward, world_up));
        let up = vec3_normalize(vec3_cross(right, forward));

        // target -= right * delta_x + up * delta_y
        let pan_vec = vec3_add(vec3_scale(right, delta_x), vec3_scale(up, delta_y));
        self.target = vec3_sub(self.target, pan_vec);
    }

    /// Compute the column-major view matrix.
    #[must_use]
    pub fn view_matrix(&self) -> Mat4 {
        let world_up: Vec3 = [0.0, 1.0, 0.0];
        look_at(self.position(), self.target, world_up)
    }

    /// Compute the column-major perspective projection matrix.
    #[must_use]
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        perspective(self.fov_y, aspect, self.near, self.far)
    }

    /// Compute the combined view-projection matrix: proj * view.
    #[must_use]
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        let proj = self.projection_matrix(aspect);
        let view = self.view_matrix();
        mat4_mul(&proj, &view)
    }

    /// Reset camera to default state.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Fit the camera to a bounding box.
    ///
    /// Sets the target to the centre of the box and adjusts the distance so
    /// the entire box is visible with a 1.5× margin.
    pub fn look_at_mesh_bounds(&mut self, bounds_min: Vec3, bounds_max: Vec3) {
        // Centre of the bounding box
        self.target = [
            (bounds_min[0] + bounds_max[0]) * 0.5,
            (bounds_min[1] + bounds_max[1]) * 0.5,
            (bounds_min[2] + bounds_max[2]) * 0.5,
        ];

        // Half-diagonal of the box
        let dx = bounds_max[0] - bounds_min[0];
        let dy = bounds_max[1] - bounds_min[1];
        let dz = bounds_max[2] - bounds_min[2];
        let radius = (dx * dx + dy * dy + dz * dz).sqrt() * 0.5;

        // Fit with 1.5× margin; use fov to set proper distance
        let half_fov = self.fov_y * 0.5;
        let required = if half_fov.sin() > 1e-6 {
            radius / half_fov.sin()
        } else {
            radius * 3.0
        };
        self.distance = (required * 1.5).clamp(self.min_distance, self.max_distance);
    }
}

impl Default for ArcballCamera {
    fn default() -> Self {
        Self::new([0.0, 0.0, 0.0], 3.0, 0.0, 0.3)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // Tolerance for floating-point comparisons
    const EPS: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn vec3_approx_eq(a: Vec3, b: Vec3, eps: f32) -> bool {
        approx_eq(a[0], b[0], eps) && approx_eq(a[1], b[1], eps) && approx_eq(a[2], b[2], eps)
    }

    #[test]
    fn test_camera_default() {
        let cam = ArcballCamera::default();
        assert_eq!(cam.target, [0.0, 0.0, 0.0]);
        assert!(approx_eq(cam.distance, 3.0, EPS));
        assert!(approx_eq(cam.yaw, 0.0, EPS));
        assert!(approx_eq(cam.pitch, 0.3, EPS));
        assert!(approx_eq(cam.fov_y, PI / 4.0, EPS));
        assert!(approx_eq(cam.near, 0.01, EPS));
        assert!(approx_eq(cam.far, 100.0, EPS));
        assert!(approx_eq(cam.min_distance, 0.1, EPS));
        assert!(approx_eq(cam.max_distance, 50.0, EPS));
    }

    #[test]
    fn test_camera_position_at_front() {
        // yaw=0, pitch=0 → camera is at (0, 0, distance) looking toward origin
        let cam = ArcballCamera::new([0.0, 0.0, 0.0], 3.0, 0.0, 0.0);
        let pos = cam.position();
        assert!(vec3_approx_eq(pos, [0.0, 0.0, 3.0], EPS));
    }

    #[test]
    fn test_camera_position_at_side() {
        // yaw=PI/2, pitch=0 → camera is at (distance, 0, 0)
        let cam = ArcballCamera::new([0.0, 0.0, 0.0], 3.0, PI / 2.0, 0.0);
        let pos = cam.position();
        assert!(vec3_approx_eq(pos, [3.0, 0.0, 0.0], EPS));
    }

    #[test]
    fn test_camera_position_at_top() {
        // Attempt to orbit to pitch=PI/2; it should clamp to 89°.
        let mut cam = ArcballCamera::new([0.0, 0.0, 0.0], 3.0, 0.0, 0.0);
        cam.orbit(0.0, PI / 2.0);
        let max_pitch = 89.0_f32.to_radians();
        // pitch should be clamped
        assert!(approx_eq(cam.pitch, max_pitch, EPS));
        // y-component should equal distance * sin(89°) ≈ distance
        let pos = cam.position();
        let expected_y = 3.0 * max_pitch.sin();
        assert!(approx_eq(pos[1], expected_y, EPS));
    }

    #[test]
    fn test_orbit_clamps_pitch() {
        let mut cam = ArcballCamera::default();
        cam.orbit(0.0, PI); // attempt large positive pitch
        assert!(cam.pitch <= 89.0_f32.to_radians() + EPS);

        cam.orbit(0.0, -2.0 * PI); // attempt large negative pitch
        assert!(cam.pitch >= -89.0_f32.to_radians() - EPS);
    }

    #[test]
    fn test_orbit_wraps_yaw() {
        let mut cam = ArcballCamera::default();
        cam.orbit(3.0 * PI, 0.0);
        // Yaw is not clamped; any large value is fine
        assert!(approx_eq(cam.yaw, 3.0 * PI, EPS));
    }

    #[test]
    fn test_dolly_clamps_distance() {
        let mut cam = ArcballCamera::default();
        cam.dolly(-100.0); // try to go below min
        assert!(approx_eq(cam.distance, cam.min_distance, EPS));

        cam.dolly(1000.0); // try to exceed max
        assert!(approx_eq(cam.distance, cam.max_distance, EPS));
    }

    #[test]
    fn test_pan_moves_target() {
        let mut cam = ArcballCamera::new([0.0, 0.0, 0.0], 5.0, 0.0, 0.0);
        let original_target = cam.target;
        cam.pan(1.0, 0.0);
        // target should have moved
        assert!(!vec3_approx_eq(cam.target, original_target, EPS));
    }

    #[test]
    fn test_view_matrix_orthogonal() {
        let cam = ArcballCamera::default();
        let view = cam.view_matrix();

        // Extract rotation columns (col 0,1,2) — they should be unit vectors
        let col0 = [view[0], view[1], view[2]];
        let col1 = [view[4], view[5], view[6]];
        let col2 = [view[8], view[9], view[10]];

        let eps = 1e-3;
        assert!(approx_eq(vec3_length(col0), 1.0, eps), "col0 not unit");
        assert!(approx_eq(vec3_length(col1), 1.0, eps), "col1 not unit");
        assert!(approx_eq(vec3_length(col2), 1.0, eps), "col2 not unit");

        assert!(
            approx_eq(vec3_dot(col0, col1), 0.0, eps),
            "col0 . col1 != 0"
        );
        assert!(
            approx_eq(vec3_dot(col0, col2), 0.0, eps),
            "col0 . col2 != 0"
        );
        assert!(
            approx_eq(vec3_dot(col1, col2), 0.0, eps),
            "col1 . col2 != 0"
        );
    }

    #[test]
    fn test_projection_matrix_shape() {
        // In the OpenGL perspective matrix the near plane maps to z = -near
        // in view space, which in NDC is -1. Verify via a hand calculation:
        //   clip_z = proj[2][2] * view_z + proj[3][2]
        //   (proj[10] = (far+near)*nf, proj[14] = 2*far*near*nf, nf=1/(near-far))
        let cam = ArcballCamera::default();
        let p = cam.projection_matrix(1.0);
        let near = cam.near;
        let far = cam.far;

        // Column-major: element at (row, col) → p[col*4 + row]
        let a = p[10]; // row2, col2
        let b = p[14]; // row2, col3
        let w_near = p[11]; // row3, col2  (should be -1)

        // At z_view = -near (eye space):
        //   clip_w = -(-near) = near
        //   clip_z = a*(-near) + b
        //   ndc_z  = clip_z / clip_w
        let clip_z_near = a * (-near) + b;
        let ndc_z_near = clip_z_near / near;
        assert!(
            approx_eq(ndc_z_near, -1.0, 1e-4),
            "near plane NDC mismatch: {ndc_z_near}"
        );

        // At z_view = -far:
        //   clip_w = far
        //   clip_z = a*(-far) + b
        let clip_z_far = a * (-far) + b;
        let ndc_z_far = clip_z_far / far;
        assert!(
            approx_eq(ndc_z_far, 1.0, 1e-4),
            "far plane NDC mismatch: {ndc_z_far}"
        );

        // w-component of the projection row should be -1
        assert!(approx_eq(w_near, -1.0, 1e-6), "w_near row should be -1");
    }

    #[test]
    fn test_reset_restores_default() {
        let mut cam = ArcballCamera::default();
        cam.orbit(1.0, 0.5);
        cam.dolly(-1.0);
        cam.reset();
        let expected = ArcballCamera::default();
        assert!(vec3_approx_eq(cam.target, expected.target, EPS));
        assert!(approx_eq(cam.distance, expected.distance, EPS));
        assert!(approx_eq(cam.yaw, expected.yaw, EPS));
        assert!(approx_eq(cam.pitch, expected.pitch, EPS));
    }

    #[test]
    fn test_look_at_mesh_bounds() {
        let mut cam = ArcballCamera::default();
        cam.look_at_mesh_bounds([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        // Target should be at origin
        assert!(vec3_approx_eq(cam.target, [0.0, 0.0, 0.0], EPS));
        // Distance should be > 0 and within bounds
        assert!(cam.distance >= cam.min_distance);
        assert!(cam.distance <= cam.max_distance);
        // Should be larger than default (3.0 might stay, but at least positive)
        assert!(cam.distance > 0.0);
    }

    #[test]
    fn test_vec3_cross_product() {
        let x_axis: Vec3 = [1.0, 0.0, 0.0];
        let y_axis: Vec3 = [0.0, 1.0, 0.0];
        let result = vec3_cross(x_axis, y_axis);
        assert!(vec3_approx_eq(result, [0.0, 0.0, 1.0], EPS));

        // Anti-commutativity
        let result2 = vec3_cross(y_axis, x_axis);
        assert!(vec3_approx_eq(result2, [0.0, 0.0, -1.0], EPS));
    }

    #[test]
    fn test_vec3_normalize() {
        let v: Vec3 = [3.0, 4.0, 0.0];
        let n = vec3_normalize(v);
        assert!(approx_eq(vec3_length(n), 1.0, EPS));
        assert!(approx_eq(n[0], 0.6, EPS));
        assert!(approx_eq(n[1], 0.8, EPS));

        // Zero vector → zero vector, no panic
        let zero: Vec3 = [0.0, 0.0, 0.0];
        let nz = vec3_normalize(zero);
        assert!(vec3_approx_eq(nz, [0.0, 0.0, 0.0], EPS));
    }

    #[test]
    fn test_mat4_mul_identity() {
        let identity: Mat4 = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let cam = ArcballCamera::default();
        let view = cam.view_matrix();
        let result = mat4_mul(&view, &identity);
        for i in 0..16 {
            assert!(
                approx_eq(result[i], view[i], EPS),
                "mat4_mul identity failed at [{i}]"
            );
        }
    }
}
