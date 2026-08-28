//! Camera path animation for smooth trajectories through 3D space.
//!
//! Camera paths define smooth, interpolated trajectories through 3D space,
//! used for rendering animations or generating training views.
//!
//! # Example
//! ```rust,ignore
//! use oxigaf_render::camera_path::{turntable_path, PathInterpolation};
//!
//! // Create a turntable (orbit) path around the origin
//! let path = turntable_path(
//!     [0.0, 0.0, 0.0],  // center
//!     3.0,               // radius
//!     1.0,               // elevation
//!     64,                // keyframes
//!     std::f32::consts::FRAC_PI_4, // fov_y
//! );
//!
//! // Sample 120 cameras for animation
//! let cameras = path.to_render_cameras(120, 1920, 1080);
//! ```

use std::f32::consts::{FRAC_PI_4, PI};

use crate::rasterizer::RenderCamera;
use crate::RenderError;

// Default near/far planes matching RasterConfig::default()
const DEFAULT_NEAR: f32 = 0.01;
const DEFAULT_FAR: f32 = 100.0;

// Epsilon for floating-point comparisons in validation
const TIME_EPSILON: f32 = 0.01;

// ─── Core Types ───────────────────────────────────────────────────────────────

/// A camera keyframe with position, look-at target, and up vector.
#[derive(Debug, Clone)]
pub struct CameraKeyframe {
    /// Normalized time in [0, 1]; keyframes must be ordered non-decreasingly.
    pub time: f32,
    /// Camera position in world space.
    pub position: [f32; 3],
    /// Look-at target position in world space.
    pub target: [f32; 3],
    /// Up vector (should be a unit vector; normalized after interpolation).
    pub up: [f32; 3],
    /// Vertical field of view in radians.
    pub fov_y: f32,
}

impl CameraKeyframe {
    /// Create a keyframe with the given time, position, target, and field of view.
    /// The up vector is initialized to world-up `[0, 1, 0]`.
    #[must_use]
    pub fn new(time: f32, position: [f32; 3], target: [f32; 3], fov_y: f32) -> Self {
        Self {
            time,
            position,
            target,
            up: [0.0, 1.0, 0.0],
            fov_y,
        }
    }

    /// Create a keyframe that looks from `from` toward `to`.
    /// Uses fov_y = π/4 and up = [0, 1, 0].
    #[must_use]
    pub fn look_from_to(time: f32, from: [f32; 3], to: [f32; 3]) -> Self {
        Self {
            time,
            position: from,
            target: to,
            up: [0.0, 1.0, 0.0],
            fov_y: FRAC_PI_4,
        }
    }
}

/// Interpolation method for camera paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathInterpolation {
    /// Piecewise-linear interpolation.
    Linear,
    /// Smooth Catmull-Rom spline through all keyframes.
    CatmullRom,
    /// Smooth-step (ease-in/ease-out) with piecewise-linear underlying path.
    Ease,
}

/// A smooth camera path through keyframes.
pub struct CameraPath {
    /// Sorted list of keyframes.
    pub keyframes: Vec<CameraKeyframe>,
    /// Interpolation algorithm.
    pub interpolation: PathInterpolation,
    /// Total animation duration in seconds (for display; does not affect `sample`).
    pub total_duration_secs: f32,
}

// ─── CameraPath construction ──────────────────────────────────────────────────

impl CameraPath {
    /// Construct a camera path from keyframes with validation.
    ///
    /// Requirements:
    /// - At least 2 keyframes.
    /// - Times in [0, 1] ordered non-decreasingly (sorted internally if needed).
    /// - First time ≈ 0 (within `TIME_EPSILON`).
    /// - Last time ≥ 0.99.
    pub fn new(
        mut keyframes: Vec<CameraKeyframe>,
        interpolation: PathInterpolation,
        total_duration_secs: f32,
    ) -> Result<Self, RenderError> {
        if keyframes.len() < 2 {
            return Err(RenderError::CameraPath(format!(
                "CameraPath requires at least 2 keyframes, got {}",
                keyframes.len()
            )));
        }

        // Sort by time (stable sort preserves order of equal times)
        keyframes.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Validate time bounds
        let first_time = keyframes.first().map(|k| k.time).unwrap_or(f32::NAN);
        let last_time = keyframes.last().map(|k| k.time).unwrap_or(f32::NAN);

        if first_time.abs() > TIME_EPSILON {
            return Err(RenderError::CameraPath(format!(
                "First keyframe time must be ≈ 0.0, got {first_time}"
            )));
        }

        if last_time < 0.99 {
            return Err(RenderError::CameraPath(format!(
                "Last keyframe time must be ≥ 0.99, got {last_time}"
            )));
        }

        // Validate that times are all in [0, 1]
        for (i, kf) in keyframes.iter().enumerate() {
            if kf.time < 0.0 || kf.time > 1.0 {
                return Err(RenderError::CameraPath(format!(
                    "Keyframe {i} has time {} outside [0, 1]",
                    kf.time
                )));
            }
        }

        Ok(Self {
            keyframes,
            interpolation,
            total_duration_secs,
        })
    }

    /// Construct a camera path without validation (for internal generator functions
    /// that guarantee valid inputs).
    fn from_valid_keyframes(
        keyframes: Vec<CameraKeyframe>,
        interpolation: PathInterpolation,
        total_duration_secs: f32,
    ) -> Self {
        Self {
            keyframes,
            interpolation,
            total_duration_secs,
        }
    }

    // ─── Sampling ─────────────────────────────────────────────────────────────

    /// Sample the path at normalized time `t ∈ [0, 1]`.
    ///
    /// `t` is clamped to [0, 1] before sampling.
    #[must_use]
    pub fn sample(&self, t: f32) -> CameraKeyframe {
        let t = t.clamp(0.0, 1.0);

        // Find the segment: largest i such that keyframes[i].time <= t
        let n = self.keyframes.len();

        // Handle degenerate (already validated ≥ 2, but be defensive)
        if n == 0 {
            return CameraKeyframe::new(t, [0.0, 0.0, 0.0], [0.0, 0.0, -1.0], FRAC_PI_4);
        }
        if n == 1 {
            return self.keyframes[0].clone();
        }

        // Binary search for the left bracket
        let right_idx = self
            .keyframes
            .partition_point(|kf| kf.time <= t)
            .min(n - 1)
            .max(1);
        let left_idx = right_idx - 1;

        let kf0 = &self.keyframes[left_idx];
        let kf1 = &self.keyframes[right_idx];

        // Compute local parameter within segment
        let dt = kf1.time - kf0.time;
        let local_t = if dt.abs() < f32::EPSILON {
            0.0_f32
        } else {
            ((t - kf0.time) / dt).clamp(0.0, 1.0)
        };

        match self.interpolation {
            PathInterpolation::Linear => self.interp_linear(left_idx, local_t, t),
            PathInterpolation::CatmullRom => self.interp_catmullrom(left_idx, local_t, t),
            PathInterpolation::Ease => self.interp_ease(left_idx, local_t, t),
        }
    }

    fn interp_linear(&self, left_idx: usize, local_t: f32, global_t: f32) -> CameraKeyframe {
        let kf0 = &self.keyframes[left_idx];
        let kf1 = &self.keyframes[left_idx + 1];
        CameraKeyframe {
            time: global_t,
            position: lerp3(kf0.position, kf1.position, local_t),
            target: lerp3(kf0.target, kf1.target, local_t),
            up: normalize3(lerp3(kf0.up, kf1.up, local_t)),
            fov_y: lerp(kf0.fov_y, kf1.fov_y, local_t),
        }
    }

    fn interp_ease(&self, left_idx: usize, local_t: f32, global_t: f32) -> CameraKeyframe {
        // Smooth-step: t' = 3t² - 2t³
        let smoothed = smooth_step(local_t);
        let kf0 = &self.keyframes[left_idx];
        let kf1 = &self.keyframes[left_idx + 1];
        CameraKeyframe {
            time: global_t,
            position: lerp3(kf0.position, kf1.position, smoothed),
            target: lerp3(kf0.target, kf1.target, smoothed),
            up: normalize3(lerp3(kf0.up, kf1.up, smoothed)),
            fov_y: lerp(kf0.fov_y, kf1.fov_y, smoothed),
        }
    }

    fn interp_catmullrom(&self, left_idx: usize, local_t: f32, global_t: f32) -> CameraKeyframe {
        let n = self.keyframes.len();
        let right_idx = left_idx + 1;

        // Gather 4 control points with boundary duplication
        let p0 = if left_idx == 0 {
            self.keyframes[0].position
        } else {
            self.keyframes[left_idx - 1].position
        };
        let p1 = self.keyframes[left_idx].position;
        let p2 = self.keyframes[right_idx].position;
        let p3 = if right_idx + 1 >= n {
            self.keyframes[n - 1].position
        } else {
            self.keyframes[right_idx + 1].position
        };

        let t0 = if left_idx == 0 {
            self.keyframes[0].target
        } else {
            self.keyframes[left_idx - 1].target
        };
        let t1 = self.keyframes[left_idx].target;
        let t2 = self.keyframes[right_idx].target;
        let t3 = if right_idx + 1 >= n {
            self.keyframes[n - 1].target
        } else {
            self.keyframes[right_idx + 1].target
        };

        let u0 = if left_idx == 0 {
            self.keyframes[0].up
        } else {
            self.keyframes[left_idx - 1].up
        };
        let u1 = self.keyframes[left_idx].up;
        let u2 = self.keyframes[right_idx].up;
        let u3 = if right_idx + 1 >= n {
            self.keyframes[n - 1].up
        } else {
            self.keyframes[right_idx + 1].up
        };

        let fov0 = if left_idx == 0 {
            self.keyframes[0].fov_y
        } else {
            self.keyframes[left_idx - 1].fov_y
        };
        let fov1 = self.keyframes[left_idx].fov_y;
        let fov2 = self.keyframes[right_idx].fov_y;
        let fov3 = if right_idx + 1 >= n {
            self.keyframes[n - 1].fov_y
        } else {
            self.keyframes[right_idx + 1].fov_y
        };

        CameraKeyframe {
            time: global_t,
            position: catmullrom3(p0, p1, p2, p3, local_t),
            target: catmullrom3(t0, t1, t2, t3, local_t),
            up: normalize3(catmullrom3(u0, u1, u2, u3, local_t)),
            fov_y: catmullrom1(fov0, fov1, fov2, fov3, local_t),
        }
    }

    /// Sample the path at `n` evenly-spaced time values in [0, 1].
    #[must_use]
    pub fn sample_uniform(&self, n: usize) -> Vec<CameraKeyframe> {
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![self.sample(0.0)];
        }
        (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                self.sample(t)
            })
            .collect()
    }

    /// Generate `n` `RenderCamera`s from uniformly-spaced samples of the path.
    #[must_use]
    pub fn to_render_cameras(&self, n: usize, width: usize, height: usize) -> Vec<RenderCamera> {
        self.sample_uniform(n)
            .iter()
            .map(|kf| keyframe_to_render_camera(kf, width, height))
            .collect()
    }
}

// ─── Preset path generators ───────────────────────────────────────────────────

/// Create an orbit (turntable) camera path around `center`.
///
/// Keyframes are evenly spaced around a horizontal circle of the given
/// `radius`, at height `elevation` above `center`. All cameras look at
/// `center`. The path is a **closed loop**: keyframe 0 (time 0.0, angle 0)
/// and the last keyframe (time 1.0, angle 2π) sit at the same physical
/// position, so `sample(0.0)` and `sample(1.0)` coincide and a looped
/// playback has no jump at the wrap. This uses one of the `num_keyframes`
/// control points to close the loop, so there are `num_keyframes - 1`
/// distinct positions spaced `2π / (num_keyframes - 1)` apart.
///
/// # Arguments
/// * `center` – The point the camera orbits around.
/// * `radius` – Distance from center to camera.
/// * `elevation` – Height offset above center (Y axis).
/// * `num_keyframes` – Number of keyframes (≥ 2).
/// * `fov_y` – Vertical field of view in radians.
#[must_use]
pub fn turntable_path(
    center: [f32; 3],
    radius: f32,
    elevation: f32,
    num_keyframes: usize,
    fov_y: f32,
) -> CameraPath {
    let count = num_keyframes.max(2);
    let keyframes: Vec<CameraKeyframe> = (0..count)
        .map(|i| {
            // Angles span the full [0, 2*PI] closed interval, matching the
            // time parameterisation below exactly: angle(i)/2*PI ==
            // time(i), so angle == 0 and angle == 2*PI (the same physical
            // position) land at time 0.0 and time 1.0 respectively and the
            // orbit closes into a loop rather than leaving a gap.
            let angle = 2.0 * PI * (i as f32) / (count - 1) as f32;
            let x = center[0] + radius * angle.cos();
            let y = center[1] + elevation;
            let z = center[2] + radius * angle.sin();
            // Exact for integer-valued i / (count - 1), including 1.0 at
            // i == count - 1.
            let time = i as f32 / (count - 1) as f32;
            CameraKeyframe::new(time, [x, y, z], center, fov_y)
        })
        .collect();

    CameraPath::from_valid_keyframes(keyframes, PathInterpolation::CatmullRom, 0.0)
}

/// Create a dolly (fly-through) path from `from` to `to`, all looking at `target`.
///
/// # Arguments
/// * `from` – Start camera position.
/// * `to` – End camera position.
/// * `target` – Look-at target (constant throughout).
/// * `num_keyframes` – Number of keyframes (≥ 2).
/// * `fov_y` – Vertical field of view in radians.
#[must_use]
pub fn dolly_path(
    from: [f32; 3],
    to: [f32; 3],
    target: [f32; 3],
    num_keyframes: usize,
    fov_y: f32,
) -> CameraPath {
    let count = num_keyframes.max(2);
    let keyframes: Vec<CameraKeyframe> = (0..count)
        .map(|i| {
            let t = if i == count - 1 {
                1.0_f32
            } else {
                i as f32 / (count - 1) as f32
            };
            let pos = lerp3(from, to, t);
            CameraKeyframe::new(t, pos, target, fov_y)
        })
        .collect();

    CameraPath::from_valid_keyframes(keyframes, PathInterpolation::Linear, 0.0)
}

/// Create a spiral camera path: combined orbit and elevation change.
///
/// # Arguments
/// * `center` – The point the camera orbits around.
/// * `radius` – Distance from center to camera.
/// * `start_elevation` – Starting height above center.
/// * `end_elevation` – Ending height above center.
/// * `num_turns` – Number of full orbits.
/// * `num_keyframes` – Number of keyframes (≥ 2).
/// * `fov_y` – Vertical field of view in radians.
#[must_use]
pub fn spiral_path(
    center: [f32; 3],
    radius: f32,
    start_elevation: f32,
    end_elevation: f32,
    num_turns: f32,
    num_keyframes: usize,
    fov_y: f32,
) -> CameraPath {
    let count = num_keyframes.max(2);
    let keyframes: Vec<CameraKeyframe> = (0..count)
        .map(|i| {
            let t = if i == count - 1 {
                1.0_f32
            } else {
                i as f32 / (count - 1) as f32
            };
            let angle = 2.0 * PI * num_turns * t;
            let elevation = lerp(start_elevation, end_elevation, t);
            let x = center[0] + radius * angle.cos();
            let y = center[1] + elevation;
            let z = center[2] + radius * angle.sin();
            CameraKeyframe::new(t, [x, y, z], center, fov_y)
        })
        .collect();

    CameraPath::from_valid_keyframes(keyframes, PathInterpolation::CatmullRom, 0.0)
}

// ─── RenderCamera construction ────────────────────────────────────────────────

/// Build a `RenderCamera` from a `CameraKeyframe` and image dimensions.
///
/// Uses a look-at view matrix and a symmetric perspective projection.
/// Near/far planes default to 0.01 and 100.0 respectively.
#[must_use]
pub fn keyframe_to_render_camera(kf: &CameraKeyframe, width: usize, height: usize) -> RenderCamera {
    let eye = glam::Vec3::from(kf.position);
    let center = glam::Vec3::from(kf.target);
    let up = glam::Vec3::from(kf.up).normalize_or_zero();
    // Fall back to world-up if up is zero
    let up = if up.length_squared() < f32::EPSILON {
        glam::Vec3::Y
    } else {
        up
    };

    // Build view matrix (right-handed, camera looks toward -Z)
    let view = glam::camera::rh::view::look_at_mat4(eye, center, up);
    let view_matrix = view.to_cols_array();

    // Build perspective projection (right-handed, depth [0, 1] for wgpu)
    let aspect = width as f32 / height.max(1) as f32;
    let proj =
        glam::camera::rh::proj::directx::perspective(kf.fov_y, aspect, DEFAULT_NEAR, DEFAULT_FAR);
    let proj_matrix = proj.to_cols_array();

    // Focal lengths in pixels
    let fy = height as f32 / (2.0 * (kf.fov_y * 0.5).tan());
    let fx = fy; // square pixels assumed

    RenderCamera {
        view_matrix,
        proj_matrix,
        position: kf.position,
        focal: [fx, fy],
    }
}

// ─── Math helpers ─────────────────────────────────────────────────────────────

/// Scalar linear interpolation.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Vector linear interpolation.
#[inline]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

/// Normalize a 3-vector; returns `[0, 1, 0]` if near-zero.
#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq < f32::EPSILON {
        return [0.0, 1.0, 0.0];
    }
    let inv = 1.0 / len_sq.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Smooth-step function: 3t² - 2t³.
#[inline]
fn smooth_step(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Catmull-Rom interpolation for a scalar value.
///
/// Formula: q(t) = 0.5 * ((2*p1) + (-p0 + p2)*t + (2*p0 - 5*p1 + 4*p2 - p3)*t² + (-p0 + 3*p1 - 3*p2 + p3)*t³)
#[inline]
fn catmullrom1(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// Catmull-Rom interpolation for a 3-vector.
#[inline]
fn catmullrom3(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3], t: f32) -> [f32; 3] {
    [
        catmullrom1(p0[0], p1[0], p2[0], p3[0], t),
        catmullrom1(p0[1], p1[1], p2[1], p3[1], t),
        catmullrom1(p0[2], p1[2], p2[2], p3[2], t),
    ]
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_path(interp: PathInterpolation) -> CameraPath {
        let keyframes = vec![
            CameraKeyframe::new(0.0, [0.0, 0.0, 3.0], [0.0, 0.0, 0.0], FRAC_PI_4),
            CameraKeyframe::new(0.5, [3.0, 0.0, 0.0], [0.0, 0.0, 0.0], FRAC_PI_4),
            CameraKeyframe::new(1.0, [0.0, 0.0, -3.0], [0.0, 0.0, 0.0], FRAC_PI_4),
        ];
        CameraPath::new(keyframes, interp, 3.0).expect("valid path")
    }

    // ── Construction validation ──────────────────────────────────────────────

    #[test]
    fn test_path_new_valid() {
        let keyframes = vec![
            CameraKeyframe::new(0.0, [0.0, 0.0, 5.0], [0.0, 0.0, 0.0], FRAC_PI_4),
            CameraKeyframe::new(1.0, [5.0, 0.0, 0.0], [0.0, 0.0, 0.0], FRAC_PI_4),
        ];
        let path = CameraPath::new(keyframes, PathInterpolation::Linear, 2.0);
        assert!(path.is_ok());
        let path = path.expect("should be Ok");
        assert_eq!(path.keyframes.len(), 2);
        assert_eq!(path.interpolation, PathInterpolation::Linear);
        assert!((path.total_duration_secs - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_path_new_insufficient_keyframes_error() {
        // Zero keyframes
        let result = CameraPath::new(vec![], PathInterpolation::Linear, 1.0);
        assert!(result.is_err(), "Should fail with zero keyframes");

        // One keyframe
        let result = CameraPath::new(
            vec![CameraKeyframe::new(
                0.0,
                [0.0, 0.0, 1.0],
                [0.0; 3],
                FRAC_PI_4,
            )],
            PathInterpolation::Linear,
            1.0,
        );
        assert!(result.is_err(), "Should fail with one keyframe");
    }

    // ── Linear interpolation ─────────────────────────────────────────────────

    #[test]
    fn test_sample_linear_at_endpoints() {
        let path = make_simple_path(PathInterpolation::Linear);

        let s0 = path.sample(0.0);
        assert!((s0.position[0] - 0.0).abs() < 1e-5);
        assert!((s0.position[2] - 3.0).abs() < 1e-5);

        let s1 = path.sample(1.0);
        assert!((s1.position[0] - 0.0).abs() < 1e-5);
        assert!((s1.position[2] - (-3.0)).abs() < 1e-5);
    }

    #[test]
    fn test_sample_linear_midpoint() {
        let path = make_simple_path(PathInterpolation::Linear);

        // At t=0.5, we're exactly at the middle keyframe
        let s = path.sample(0.5);
        assert!((s.position[0] - 3.0).abs() < 1e-4);
        assert!((s.position[2] - 0.0).abs() < 1e-4);
    }

    // ── Catmull-Rom interpolation ─────────────────────────────────────────────

    #[test]
    fn test_sample_catmullrom_endpoints() {
        let path = make_simple_path(PathInterpolation::CatmullRom);

        let s0 = path.sample(0.0);
        assert!(
            (s0.position[2] - 3.0).abs() < 1e-4,
            "pos at t=0: {:?}",
            s0.position
        );

        let s1 = path.sample(1.0);
        assert!(
            (s1.position[2] - (-3.0)).abs() < 1e-4,
            "pos at t=1: {:?}",
            s1.position
        );
    }

    #[test]
    fn test_sample_catmullrom_smooth() {
        let path = make_simple_path(PathInterpolation::CatmullRom);

        // Sample many values and check continuity (adjacent samples should be close)
        let samples: Vec<[f32; 3]> = (0..=100)
            .map(|i| path.sample(i as f32 / 100.0).position)
            .collect();

        for w in samples.windows(2) {
            let dx = w[1][0] - w[0][0];
            let dy = w[1][1] - w[0][1];
            let dz = w[1][2] - w[0][2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            // Adjacent samples should not jump more than 0.2 units at this resolution
            assert!(dist < 0.2, "Large jump in Catmull-Rom path: {dist}");
        }
    }

    // ── Ease interpolation ────────────────────────────────────────────────────

    #[test]
    fn test_sample_ease_monotone() {
        // For a straight-line path, ease should be monotone in each coordinate
        let keyframes = vec![
            CameraKeyframe::new(0.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], FRAC_PI_4),
            CameraKeyframe::new(1.0, [10.0, 0.0, 0.0], [11.0, 0.0, 0.0], FRAC_PI_4),
        ];
        let path = CameraPath::new(keyframes, PathInterpolation::Ease, 1.0).expect("valid");

        let mut prev_x = f32::NEG_INFINITY;
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let s = path.sample(t);
            assert!(
                s.position[0] >= prev_x - 1e-5,
                "Ease path not monotone at t={t}: x={}, prev={prev_x}",
                s.position[0]
            );
            prev_x = s.position[0];
        }
    }

    // ── Preset generators ─────────────────────────────────────────────────────

    #[test]
    fn test_turntable_path_keyframe_count() {
        let path = turntable_path([0.0; 3], 3.0, 1.0, 16, FRAC_PI_4);
        assert_eq!(path.keyframes.len(), 16);
    }

    #[test]
    fn test_turntable_path_radius() {
        let center = [1.0, 0.5, 2.0];
        let radius = 4.0;
        let elevation = 1.5;
        let path = turntable_path(center, radius, elevation, 8, FRAC_PI_4);

        for kf in &path.keyframes {
            let dx = kf.position[0] - center[0];
            let dz = kf.position[2] - center[2];
            let actual_radius = (dx * dx + dz * dz).sqrt();
            assert!(
                (actual_radius - radius).abs() < 1e-4,
                "Expected radius {radius}, got {actual_radius}"
            );
            let actual_elevation = kf.position[1] - center[1];
            assert!(
                (actual_elevation - elevation).abs() < 1e-4,
                "Expected elevation {elevation}, got {actual_elevation}"
            );
        }
    }

    #[test]
    fn test_turntable_path_closes_loop() {
        // Regression: the angle/time parameterisation must span a full
        // revolution over t in [0, 1], not (count-1)/count of one (which
        // left a gap and made a looped playback jump at the wrap).
        let center = [0.5, 1.0, -2.0];
        let radius = 3.0;
        let elevation = 0.5;
        let path = turntable_path(center, radius, elevation, 16, FRAC_PI_4);

        let start = path.sample(0.0);
        let end = path.sample(1.0);
        for c in 0..3 {
            assert!(
                (start.position[c] - end.position[c]).abs() < 1e-3,
                "sample(0.0) and sample(1.0) should coincide (closed loop): \
                 start={:?}, end={:?}",
                start.position,
                end.position
            );
        }

        // The keyframes themselves (not just the endpoints of the spline)
        // must also span the full circle: the last keyframe's angle should
        // be a full 2*PI from the first, i.e. the same position.
        let first_kf = &path.keyframes[0];
        let last_kf = path.keyframes.last().expect("at least 2 keyframes");
        for c in 0..3 {
            assert!(
                (first_kf.position[c] - last_kf.position[c]).abs() < 1e-4,
                "first and last keyframe should be at the same position \
                 (angle 0 and angle 2*PI): first={:?}, last={:?}",
                first_kf.position,
                last_kf.position
            );
        }
    }

    #[test]
    fn test_dolly_path() {
        let from = [0.0, 0.0, 5.0];
        let to = [0.0, 0.0, -5.0];
        let target = [0.0; 3];
        let path = dolly_path(from, to, target, 5, FRAC_PI_4);

        assert_eq!(path.keyframes.len(), 5);

        // First keyframe should be at `from`
        let first = &path.keyframes[0];
        assert!((first.position[0] - from[0]).abs() < 1e-5);
        assert!((first.position[2] - from[2]).abs() < 1e-5);

        // Last keyframe should be at `to`
        let last = path.keyframes.last().expect("non-empty");
        assert!((last.position[2] - to[2]).abs() < 1e-5);

        // Intermediate at t=0.5 should be at origin in Z
        let mid = path.sample(0.5);
        assert!((mid.position[2]).abs() < 0.5);
    }

    #[test]
    fn test_spiral_path() {
        let center = [0.0; 3];
        let path = spiral_path(center, 3.0, 0.0, 5.0, 2.0, 20, FRAC_PI_4);
        assert_eq!(path.keyframes.len(), 20);

        // Check elevation range
        let first_elev = path.keyframes[0].position[1];
        let last_elev = path.keyframes.last().expect("non-empty").position[1];
        assert!((first_elev).abs() < 1e-4, "Start elevation should be 0");
        assert!((last_elev - 5.0).abs() < 1e-4, "End elevation should be 5");
    }

    // ── Uniform sampling ─────────────────────────────────────────────────────

    #[test]
    fn test_sample_uniform_count() {
        let path = make_simple_path(PathInterpolation::Linear);

        let samples = path.sample_uniform(10);
        assert_eq!(samples.len(), 10);

        // Check endpoint times
        assert!((samples[0].time - 0.0).abs() < f32::EPSILON);
        assert!((samples[9].time - 1.0).abs() < f32::EPSILON);
    }

    // ── to_render_cameras ────────────────────────────────────────────────────

    #[test]
    fn test_path_to_render_cameras_count() {
        let path = make_simple_path(PathInterpolation::Linear);
        let cameras = path.to_render_cameras(8, 640, 480);
        assert_eq!(cameras.len(), 8);
    }

    // ── Catmull-Rom bounds ────────────────────────────────────────────────────

    #[test]
    fn test_catmullrom_within_bounds() {
        // With collinear keyframes, Catmull-Rom should stay near the line
        let keyframes = vec![
            CameraKeyframe::new(0.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], FRAC_PI_4),
            CameraKeyframe::new(0.33, [3.0, 0.0, 0.0], [4.0, 0.0, 0.0], FRAC_PI_4),
            CameraKeyframe::new(0.67, [6.0, 0.0, 0.0], [7.0, 0.0, 0.0], FRAC_PI_4),
            CameraKeyframe::new(1.0, [9.0, 0.0, 0.0], [10.0, 0.0, 0.0], FRAC_PI_4),
        ];
        let path = CameraPath::new(keyframes, PathInterpolation::CatmullRom, 1.0).expect("valid");

        // Sample and ensure all positions are in a reasonable range of [0, 9]
        for i in 0..=50 {
            let t = i as f32 / 50.0;
            let s = path.sample(t);
            // For collinear points, Catmull-Rom should stay close to the line
            assert!(
                s.position[0] >= -1.0 && s.position[0] <= 10.0,
                "Position out of bounds at t={t}: {:?}",
                s.position
            );
        }
    }

    // ── CameraKeyframe helpers ────────────────────────────────────────────────

    #[test]
    fn test_keyframe_look_from_to() {
        let kf = CameraKeyframe::look_from_to(0.5, [0.0, 0.0, 5.0], [0.0, 0.0, 0.0]);
        assert!((kf.fov_y - FRAC_PI_4).abs() < f32::EPSILON);
        assert_eq!(kf.up, [0.0, 1.0, 0.0]);
        assert_eq!(kf.position, [0.0, 0.0, 5.0]);
        assert_eq!(kf.target, [0.0, 0.0, 0.0]);
        assert!((kf.time - 0.5).abs() < f32::EPSILON);
    }

    // ── Up vector normalization ────────────────────────────────────────────────

    #[test]
    fn test_up_vector_normalized_after_interpolation() {
        // Create keyframes with up vectors that aren't unit length after lerp
        let keyframes = vec![
            CameraKeyframe {
                time: 0.0,
                position: [0.0, 0.0, 5.0],
                target: [0.0; 3],
                up: [0.0, 1.0, 0.0],
                fov_y: FRAC_PI_4,
            },
            CameraKeyframe {
                time: 1.0,
                position: [5.0, 0.0, 0.0],
                target: [0.0; 3],
                up: [0.0, 1.0, 0.0],
                fov_y: FRAC_PI_4,
            },
        ];
        let path = CameraPath::new(keyframes, PathInterpolation::Linear, 1.0).expect("valid");

        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let s = path.sample(t);
            let up = s.up;
            let len = (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "Up vector not normalized at t={t}: len={len}"
            );
        }
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_time_out_of_range_clamped() {
        let path = make_simple_path(PathInterpolation::Linear);

        // t < 0 should clamp to 0
        let s_neg = path.sample(-0.5);
        let s_zero = path.sample(0.0);
        assert!((s_neg.position[0] - s_zero.position[0]).abs() < 1e-5);
        assert!((s_neg.position[2] - s_zero.position[2]).abs() < 1e-5);

        // t > 1 should clamp to 1
        let s_over = path.sample(1.5);
        let s_one = path.sample(1.0);
        assert!((s_over.position[0] - s_one.position[0]).abs() < 1e-5);
        assert!((s_over.position[2] - s_one.position[2]).abs() < 1e-5);
    }

    #[test]
    fn test_single_interval_catmullrom() {
        // Two keyframes — boundary duplication should apply on both sides
        let keyframes = vec![
            CameraKeyframe::new(0.0, [0.0, 0.0, 0.0], [0.0, 0.0, -1.0], FRAC_PI_4),
            CameraKeyframe::new(1.0, [1.0, 0.0, 0.0], [1.0, 0.0, -1.0], FRAC_PI_4),
        ];
        let path = CameraPath::new(keyframes, PathInterpolation::CatmullRom, 1.0).expect("valid");

        // Sample at midpoint — should be roughly in the middle
        let mid = path.sample(0.5);
        assert!(
            (mid.position[0] - 0.5).abs() < 0.1,
            "Midpoint should be near 0.5, got {:?}",
            mid.position
        );

        // Endpoints must match keyframes
        let s0 = path.sample(0.0);
        assert!((s0.position[0]).abs() < 1e-4);
        let s1 = path.sample(1.0);
        assert!((s1.position[0] - 1.0).abs() < 1e-4);
    }
}
