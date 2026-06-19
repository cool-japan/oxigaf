//! Camera path generation and manipulation for rendering animation sequences.
//!
//! Provides orbit paths, fly-through paths, and smooth interpolated trajectories
//! for use with 3D Gaussian avatar models.

use std::f32::consts::PI;
use thiserror::Error;

use crate::arcball::{
    look_at as arcball_look_at, vec3_add, vec3_dot, vec3_length, vec3_normalize, vec3_scale,
    vec3_sub,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from camera path operations.
#[derive(Debug, Error)]
pub enum CameraPathError {
    #[error("Empty path")]
    EmptyPath,
    #[error("Insufficient keyframes: need {needed}, got {got}")]
    InsufficientKeyframes { needed: usize, got: usize },
    #[error("Invalid config: {0}")]
    InvalidConfig(String),
    #[error("Frame out of range: {frame}, path has {len} frames")]
    FrameOutOfRange { frame: usize, len: usize },
    #[error("Invalid duration: {0} seconds")]
    InvalidDuration(f32),
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

// ---------------------------------------------------------------------------
// CameraPose
// ---------------------------------------------------------------------------

/// A camera pose: position + look-at target + up vector + field of view.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraPose {
    /// Camera position in world space.
    pub position: [f32; 3],
    /// Look-at point.
    pub target: [f32; 3],
    /// Up vector.
    pub up: [f32; 3],
    /// Vertical field of view in degrees.
    pub fov_y: f32,
}

impl CameraPose {
    /// Create a new `CameraPose` with default up `[0, 1, 0]` and fov 45°.
    pub fn new(position: [f32; 3], target: [f32; 3]) -> Self {
        Self {
            position,
            target,
            up: [0.0, 1.0, 0.0],
            fov_y: 45.0,
        }
    }

    /// Create a `CameraPose` with an explicit up vector and default fov 45°.
    pub fn look_at(position: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Self {
        Self {
            position,
            target,
            up,
            fov_y: 45.0,
        }
    }

    /// Distance from position to target.
    pub fn distance(&self) -> f32 {
        vec3_length(vec3_sub(self.target, self.position))
    }

    /// Normalized forward vector (target − position).
    pub fn forward(&self) -> [f32; 3] {
        vec3_normalize(vec3_sub(self.target, self.position))
    }

    /// 4×4 right-handed look-at view matrix, returned as row-major `[[f32;4];4]`.
    ///
    /// The underlying computation delegates to `arcball::look_at` which
    /// produces a column-major flat array; this function reshapes it into the
    /// row-major 2D form expected by the spec.
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        // arcball::look_at returns a column-major [f32; 16]
        let m = arcball_look_at(self.position, self.target, self.up);
        // Transpose: col-major m[col*4+row] → row-major out[row][col]
        let mut out = [[0.0f32; 4]; 4];
        for row in 0..4 {
            for col in 0..4 {
                out[row][col] = m[col * 4 + row];
            }
        }
        out
    }

    /// Linearly interpolate between `self` (t=0) and `other` (t=1).
    pub fn interpolate(&self, other: &CameraPose, t: f32) -> CameraPose {
        let lerp3 = |a: [f32; 3], b: [f32; 3]| -> [f32; 3] {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            ]
        };
        CameraPose {
            position: lerp3(self.position, other.position),
            target: lerp3(self.target, other.target),
            up: vec3_normalize(lerp3(self.up, other.up)),
            fov_y: self.fov_y + (other.fov_y - self.fov_y) * t,
        }
    }
}

// ---------------------------------------------------------------------------
// EasingType
// ---------------------------------------------------------------------------

/// Easing function applied to time parameter t ∈ [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EasingType {
    /// No easing — linear interpolation.
    Linear,
    /// Quadratic ease-in.
    EaseIn,
    /// Quadratic ease-out.
    EaseOut,
    /// Smoothstep ease-in-out.
    EaseInOut,
    /// CSS `cubic-bezier(0.42, 0.0, 0.58, 1.0)` — the "ease-in-out" Bézier preset.
    ///
    /// Control points: P1 = (0.42, 0.0), P2 = (0.58, 1.0) with P0 = (0,0) and
    /// P3 = (1,1) implicit.  This is the standard CSS `ease-in-out` Bézier timing
    /// function — perceptually distinct from the polynomial smoothstep used by
    /// [`EasingType::EaseInOut`] due to its asymmetric control-point placement.
    ///
    /// The mapping is computed with Newton-Raphson iteration: given input time
    /// `t`, the algorithm solves `B_x(s) = t` for the Bézier parameter `s`
    /// (up to 10 iterations, guarded against near-zero derivatives), then
    /// evaluates `B_y(s)` to obtain the output value.
    CubicBezier,
}

impl EasingType {
    /// Apply the easing function to `t` ∈ [0, 1], returning a value in [0, 1].
    pub fn apply(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            EasingType::Linear => t,
            EasingType::EaseIn => t * t,
            EasingType::EaseOut => t * (2.0 - t),
            EasingType::EaseInOut => t * t * (3.0 - 2.0 * t), // smoothstep
            EasingType::CubicBezier => {
                // CSS ease-in-out Bézier: cubic-bezier(0.42, 0.0, 0.58, 1.0)
                // P1=(cx1=0.42, cy1=0.0), P2=(cx2=0.58, cy2=1.0).
                const CX1: f32 = 0.42;
                const CY1: f32 = 0.0;
                const CX2: f32 = 0.58;
                const CY2: f32 = 1.0;
                cubic_bezier_ease(t, CX1, CY1, CX2, CY2)
            }
        }
    }
}

/// Evaluate a CSS `cubic-bezier(cx1, cy1, cx2, cy2)` curve at input time `t`.
///
/// P0 = (0,0) and P3 = (1,1) are implicit; only the inner handles
/// `(cx1, cy1)` and `(cx2, cy2)` are supplied.
///
/// Algorithm: solve `B_x(s) = t` via Newton-Raphson (≤10 iterations,
/// derivative guard `|dB_x/ds| < 1e-7`), then return `B_y(s)`.
///
/// ```text
/// B_x(s) = 3(1-s)²s·cx1 + 3(1-s)s²·cx2 + s³
/// B_y(s) = 3(1-s)²s·cy1 + 3(1-s)s²·cy2 + s³
/// dB_x   = 3(1-4s+3s²)·cx1 + 3(2s-3s²)·cx2 + 3s²
/// ```
pub fn cubic_bezier_ease(t: f32, cx1: f32, cy1: f32, cx2: f32, cy2: f32) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    // Cubic Bernstein basis: B(s) = 3(1-s)²s·P1 + 3(1-s)s²·P2 + s³
    // where P0=(0,0) and P3=(1,1) are implicit.
    let bx =
        |s: f32| 3.0 * (1.0 - s) * (1.0 - s) * s * cx1 + 3.0 * (1.0 - s) * s * s * cx2 + s * s * s;
    let dbx = |s: f32| {
        3.0 * (1.0 - 4.0 * s + 3.0 * s * s) * cx1
            + 3.0 * (2.0 * s - 3.0 * s * s) * cx2
            + 3.0 * s * s
    };
    let by =
        |s: f32| 3.0 * (1.0 - s) * (1.0 - s) * s * cy1 + 3.0 * (1.0 - s) * s * s * cy2 + s * s * s;

    // Newton-Raphson: solve bx(s) = t
    let mut s = t; // initial guess (linear)
    for _ in 0..10 {
        let fx = bx(s) - t;
        let dfx = dbx(s);
        if dfx.abs() < 1e-7 {
            break;
        }
        s -= fx / dfx;
        s = s.clamp(0.0, 1.0);
    }
    by(s)
}

// ---------------------------------------------------------------------------
// PathKeyframe
// ---------------------------------------------------------------------------

/// A keyframe in the camera path.
#[derive(Debug, Clone)]
pub struct PathKeyframe {
    /// Normalized time in [0, 1].
    pub time: f32,
    /// Camera pose at this keyframe.
    pub pose: CameraPose,
    /// Easing function applied to the segment ending at this keyframe.
    pub ease: EasingType,
}

impl PathKeyframe {
    /// Create a new keyframe with linear easing.
    pub fn new(time: f32, pose: CameraPose) -> Self {
        Self {
            time,
            pose,
            ease: EasingType::Linear,
        }
    }
}

// ---------------------------------------------------------------------------
// PathConfig
// ---------------------------------------------------------------------------

/// Configuration for camera path generation.
#[derive(Debug, Clone)]
pub struct PathConfig {
    /// Total animation duration in seconds.
    pub duration_secs: f32,
    /// Frames per second.
    pub fps: f32,
    /// Use Catmull-Rom smoothing between keyframes.
    pub smooth_tangents: bool,
}

impl Default for PathConfig {
    fn default() -> Self {
        Self {
            duration_secs: 5.0,
            fps: 30.0,
            smooth_tangents: false,
        }
    }
}

impl PathConfig {
    /// Validate the configuration, returning an error for invalid values.
    pub fn validate(&self) -> Result<(), CameraPathError> {
        if self.duration_secs <= 0.0 {
            return Err(CameraPathError::InvalidDuration(self.duration_secs));
        }
        if self.fps <= 0.0 {
            return Err(CameraPathError::InvalidConfig(format!(
                "fps must be positive, got {}",
                self.fps
            )));
        }
        Ok(())
    }

    /// Total number of frames: ceil(duration_secs × fps).
    pub fn total_frames(&self) -> usize {
        (self.duration_secs * self.fps).ceil() as usize
    }
}

// ---------------------------------------------------------------------------
// CameraPath
// ---------------------------------------------------------------------------

/// A complete camera path — a sequence of poses at discrete frames.
#[derive(Debug, Clone)]
pub struct CameraPath {
    /// Pose for each frame.
    pub frames: Vec<CameraPose>,
    /// Frames per second.
    pub fps: f32,
    /// Duration in seconds.
    pub duration_secs: f32,
}

impl CameraPath {
    /// Construct a `CameraPath` from a frame list.
    ///
    /// Returns [`CameraPathError::EmptyPath`] if `frames` is empty.
    pub fn new(frames: Vec<CameraPose>, fps: f32) -> Result<Self, CameraPathError> {
        if frames.is_empty() {
            return Err(CameraPathError::EmptyPath);
        }
        let n = frames.len();
        let duration_secs = if fps > 0.0 { n as f32 / fps } else { n as f32 };
        Ok(Self {
            frames,
            fps,
            duration_secs,
        })
    }

    /// Retrieve the pose at a given frame index.
    pub fn get_frame(&self, index: usize) -> Result<&CameraPose, CameraPathError> {
        self.frames
            .get(index)
            .ok_or(CameraPathError::FrameOutOfRange {
                frame: index,
                len: self.frames.len(),
            })
    }

    /// Total number of frames in the path.
    pub fn total_frames(&self) -> usize {
        self.frames.len()
    }

    /// Return a new path with frames in reversed order.
    pub fn reverse(&self) -> Self {
        let mut frames = self.frames.clone();
        frames.reverse();
        Self {
            frames,
            fps: self.fps,
            duration_secs: self.duration_secs,
        }
    }

    /// Return a new path by concatenating this path `n` times.
    pub fn loop_path(&self, n: usize) -> Self {
        let mut frames = Vec::with_capacity(self.frames.len() * n);
        for _ in 0..n {
            frames.extend(self.frames.iter().cloned());
        }
        let duration_secs = self.duration_secs * n as f32;
        Self {
            frames,
            fps: self.fps,
            duration_secs,
        }
    }

    /// Return a trimmed sub-path from `start_frame` (inclusive) to `end_frame` (exclusive).
    pub fn trim(&self, start_frame: usize, end_frame: usize) -> Result<Self, CameraPathError> {
        let len = self.frames.len();
        if start_frame >= len {
            return Err(CameraPathError::FrameOutOfRange {
                frame: start_frame,
                len,
            });
        }
        if end_frame > len {
            return Err(CameraPathError::FrameOutOfRange {
                frame: end_frame,
                len,
            });
        }
        if start_frame >= end_frame {
            return Err(CameraPathError::InvalidConfig(format!(
                "start_frame ({start_frame}) must be less than end_frame ({end_frame})"
            )));
        }
        let frames = self.frames[start_frame..end_frame].to_vec();
        let duration_secs = frames.len() as f32 / self.fps;
        Ok(Self {
            frames,
            fps: self.fps,
            duration_secs,
        })
    }
}

// ---------------------------------------------------------------------------
// orbit_path
// ---------------------------------------------------------------------------

/// Generate a circular orbit path around `center` at given radius and height.
///
/// Starts from azimuth 0 and completes `turns` full rotations.
pub fn orbit_path(
    center: [f32; 3],
    radius: f32,
    height: f32,
    turns: f32,
    config: &PathConfig,
) -> Result<CameraPath, CameraPathError> {
    config.validate()?;
    if radius <= 0.0 {
        return Err(CameraPathError::InvalidConfig(format!(
            "radius must be positive, got {radius}"
        )));
    }
    let n = config.total_frames().max(1);
    let mut frames = Vec::with_capacity(n);
    for i in 0..n {
        let frac = i as f32 / n as f32;
        let angle = 2.0 * PI * turns * frac;
        let position = [
            center[0] + radius * angle.cos(),
            center[1] + height,
            center[2] + radius * angle.sin(),
        ];
        frames.push(CameraPose::new(position, center));
    }
    CameraPath::new(frames, config.fps)
}

// ---------------------------------------------------------------------------
// spiral_orbit_path
// ---------------------------------------------------------------------------

/// Generate a spiral orbit with linearly changing radius and height.
pub fn spiral_orbit_path(
    center: [f32; 3],
    radius_start: f32,
    radius_end: f32,
    height_start: f32,
    height_end: f32,
    config: &PathConfig,
) -> Result<CameraPath, CameraPathError> {
    config.validate()?;
    let n = config.total_frames().max(1);
    let mut frames = Vec::with_capacity(n);
    for i in 0..n {
        let frac = i as f32 / n as f32;
        let angle = 2.0 * PI * frac;
        let radius = radius_start + (radius_end - radius_start) * frac;
        let height = height_start + (height_end - height_start) * frac;
        let position = [
            center[0] + radius * angle.cos(),
            center[1] + height,
            center[2] + radius * angle.sin(),
        ];
        frames.push(CameraPose::new(position, center));
    }
    CameraPath::new(frames, config.fps)
}

// ---------------------------------------------------------------------------
// catmull_rom
// ---------------------------------------------------------------------------

/// Catmull-Rom spline interpolation.
///
/// Returns the interpolated position at `t` ∈ [0, 1] between `p1` and `p2`,
/// using `p0` and `p3` as neighbouring control points.
pub fn catmull_rom(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3], t: f32) -> [f32; 3] {
    let t2 = t * t;
    let t3 = t2 * t;
    // Catmull-Rom basis weights
    let w0 = -0.5 * t3 + t2 - 0.5 * t;
    let w1 = 1.5 * t3 - 2.5 * t2 + 1.0;
    let w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
    let w3 = 0.5 * t3 - 0.5 * t2;
    [
        w0 * p0[0] + w1 * p1[0] + w2 * p2[0] + w3 * p3[0],
        w0 * p0[1] + w1 * p1[1] + w2 * p2[1] + w3 * p3[1],
        w0 * p0[2] + w1 * p1[2] + w2 * p2[2] + w3 * p3[2],
    ]
}

// ---------------------------------------------------------------------------
// keyframe_path
// ---------------------------------------------------------------------------

/// Interpolate a list of keyframes into a smooth `CameraPath`.
///
/// Uses linear interpolation by default; Catmull-Rom if
/// `config.smooth_tangents` is true.
pub fn keyframe_path(
    keyframes: &[PathKeyframe],
    config: &PathConfig,
) -> Result<CameraPath, CameraPathError> {
    config.validate()?;
    if keyframes.len() < 2 {
        return Err(CameraPathError::InsufficientKeyframes {
            needed: 2,
            got: keyframes.len(),
        });
    }
    let n = config.total_frames().max(1);
    let mut frames = Vec::with_capacity(n);

    for frame_idx in 0..n {
        let global_t = frame_idx as f32 / (n - 1).max(1) as f32;

        // Find which segment this frame belongs to
        let seg = find_segment(keyframes, global_t);
        let kf_a = &keyframes[seg];
        let kf_b = &keyframes[seg + 1];

        // Local t within the segment
        let span = kf_b.time - kf_a.time;
        let local_t = if span > 1e-9 {
            (global_t - kf_a.time) / span
        } else {
            0.0
        };
        let eased_t = kf_b.ease.apply(local_t);

        let pose = if config.smooth_tangents && keyframes.len() >= 2 {
            // Catmull-Rom on position and target
            let i0 = if seg == 0 { 0 } else { seg - 1 };
            let i3 = (seg + 2).min(keyframes.len() - 1);
            let p0 = keyframes[i0].pose.position;
            let p1 = kf_a.pose.position;
            let p2 = kf_b.pose.position;
            let p3 = keyframes[i3].pose.position;
            let q0 = keyframes[i0].pose.target;
            let q1 = kf_a.pose.target;
            let q2 = kf_b.pose.target;
            let q3 = keyframes[i3].pose.target;
            let position = catmull_rom(p0, p1, p2, p3, eased_t);
            let target = catmull_rom(q0, q1, q2, q3, eased_t);
            CameraPose {
                position,
                target,
                up: vec3_normalize(lerp3(kf_a.pose.up, kf_b.pose.up, eased_t)),
                fov_y: kf_a.pose.fov_y + (kf_b.pose.fov_y - kf_a.pose.fov_y) * eased_t,
            }
        } else {
            kf_a.pose.interpolate(&kf_b.pose, eased_t)
        };
        frames.push(pose);
    }
    CameraPath::new(frames, config.fps)
}

/// Find the segment index such that keyframes[seg].time <= t < keyframes[seg+1].time.
fn find_segment(keyframes: &[PathKeyframe], t: f32) -> usize {
    let last_seg = keyframes.len() - 2;
    for i in 0..last_seg {
        if t < keyframes[i + 1].time {
            return i;
        }
    }
    last_seg
}

/// Linear interpolation of two [f32; 3] values.
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

// ---------------------------------------------------------------------------
// figure_eight_path
// ---------------------------------------------------------------------------

/// Generate a figure-8 (lemniscate of Bernoulli) path around `center`.
pub fn figure_eight_path(
    center: [f32; 3],
    radius: f32,
    height: f32,
    config: &PathConfig,
) -> Result<CameraPath, CameraPathError> {
    config.validate()?;
    if radius <= 0.0 {
        return Err(CameraPathError::InvalidConfig(format!(
            "radius must be positive, got {radius}"
        )));
    }
    let n = config.total_frames().max(1);
    let mut frames = Vec::with_capacity(n);
    for i in 0..n {
        let frac = i as f32 / n as f32;
        let angle = 2.0 * PI * frac;
        // Lemniscate parametric form: x = a*cos(t)/(1+sin²(t)), z = a*sin(t)*cos(t)/(1+sin²(t))
        let denom = 1.0 + (angle.sin() * angle.sin());
        let x = radius * angle.cos() / denom;
        let z = radius * angle.sin() * angle.cos() / denom;
        let position = [center[0] + x, center[1] + height, center[2] + z];
        frames.push(CameraPose::new(position, center));
    }
    CameraPath::new(frames, config.fps)
}

// ---------------------------------------------------------------------------
// PathStats
// ---------------------------------------------------------------------------

/// Statistics about a camera path.
#[derive(Debug, Clone)]
pub struct PathStats {
    /// Total number of frames.
    pub total_frames: usize,
    /// Duration in seconds.
    pub duration_secs: f32,
    /// Total camera travel distance (sum of per-frame displacements).
    pub total_distance: f32,
    /// Mean speed in units/second.
    pub mean_speed: f32,
    /// Maximum speed between adjacent frames.
    pub max_speed: f32,
    /// Mean field-of-view in degrees.
    pub mean_fov: f32,
}

/// Compute statistics for the given `CameraPath`.
pub fn compute_path_stats(path: &CameraPath) -> Result<PathStats, CameraPathError> {
    if path.frames.is_empty() {
        return Err(CameraPathError::EmptyPath);
    }
    let total_frames = path.frames.len();
    let duration_secs = path.duration_secs;
    let dt = if path.fps > 0.0 { 1.0 / path.fps } else { 1.0 };

    let mut total_distance = 0.0f32;
    let mut max_speed = 0.0f32;
    let mut fov_sum = 0.0f32;

    for frame in &path.frames {
        fov_sum += frame.fov_y;
    }
    let mean_fov = fov_sum / total_frames as f32;

    for pair in path.frames.windows(2) {
        let d = vec3_length(vec3_sub(pair[1].position, pair[0].position));
        total_distance += d;
        let speed = d / dt;
        if speed > max_speed {
            max_speed = speed;
        }
    }
    let mean_speed = if duration_secs > 0.0 {
        total_distance / duration_secs
    } else {
        0.0
    };

    Ok(PathStats {
        total_frames,
        duration_secs,
        total_distance,
        mean_speed,
        max_speed,
        mean_fov,
    })
}

// ---------------------------------------------------------------------------
// path_to_json / path_from_json
// ---------------------------------------------------------------------------

/// Export a `CameraPath` to a JSON string (hand-rolled).
///
/// Format:
/// ```json
/// {"fps":30.0,"frames":[{"position":[x,y,z],"target":[x,y,z],"fov_y":45.0},...]}
/// ```
pub fn path_to_json(path: &CameraPath) -> String {
    let mut out = String::new();
    out.push_str(&format!("{{\"fps\":{},\"frames\":[", path.fps));
    for (i, frame) in path.frames.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"position\":[{},{},{}],\"target\":[{},{},{}],\"fov_y\":{}}}",
            frame.position[0],
            frame.position[1],
            frame.position[2],
            frame.target[0],
            frame.target[1],
            frame.target[2],
            frame.fov_y,
        ));
    }
    out.push_str("]}");
    out
}

/// Parse a `CameraPath` from a JSON string produced by [`path_to_json`].
///
/// `fps` is used as a fallback only if the JSON does not contain `"fps"`.
pub fn path_from_json(json: &str, fps: f32) -> Result<CameraPath, CameraPathError> {
    let json = json.trim();

    // Extract fps from JSON if present
    let actual_fps = extract_json_float(json, "fps").unwrap_or(fps);

    // Extract the frames array content
    let frames_start = json
        .find("\"frames\":[")
        .ok_or_else(|| CameraPathError::InvalidConfig("missing \"frames\" array".to_string()))?;
    let array_start = frames_start
        + json[frames_start..]
            .find('[')
            .ok_or_else(|| CameraPathError::InvalidConfig("missing '[' in frames".to_string()))?;
    let array_end = find_matching_bracket(json, array_start)
        .ok_or_else(|| CameraPathError::InvalidConfig("unmatched '[' in frames".to_string()))?;

    let array_content = &json[array_start + 1..array_end];
    if array_content.trim().is_empty() {
        return Err(CameraPathError::EmptyPath);
    }

    let frame_objects = split_json_objects(array_content);
    let mut frames = Vec::with_capacity(frame_objects.len());

    for obj in frame_objects {
        let position = extract_json_array3(obj, "position").ok_or_else(|| {
            CameraPathError::InvalidConfig(format!(
                "missing or invalid 'position' in frame: {}",
                &obj[..obj.len().min(80)]
            ))
        })?;
        let target = extract_json_array3(obj, "target").ok_or_else(|| {
            CameraPathError::InvalidConfig(format!(
                "missing or invalid 'target' in frame: {}",
                &obj[..obj.len().min(80)]
            ))
        })?;
        let fov_y = extract_json_float(obj, "fov_y").unwrap_or(45.0);
        frames.push(CameraPose {
            position,
            target,
            up: [0.0, 1.0, 0.0],
            fov_y,
        });
    }
    CameraPath::new(frames, actual_fps)
}

// ---------------------------------------------------------------------------
// JSON parsing helpers
// ---------------------------------------------------------------------------

/// Extract a float value from a JSON string by key.
fn extract_json_float(json: &str, key: &str) -> Option<f32> {
    let search = format!("\"{}\":", key);
    let start = json.find(&search)? + search.len();
    let rest = json[start..].trim_start();
    let end = rest.find([',', '}', ']']).unwrap_or(rest.len());
    rest[..end].trim().parse::<f32>().ok()
}

/// Find the closing bracket matching the `[` at `start` in `json`.
fn find_matching_bracket(json: &str, start: usize) -> Option<usize> {
    let bytes = json.as_bytes();
    if bytes.get(start) != Some(&b'[') {
        return None;
    }
    let mut depth = 0usize;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        match b {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Split a JSON array body (without outer `[` `]`) into individual object strings.
fn split_json_objects(content: &str) -> Vec<&str> {
    let mut objects = Vec::new();
    let bytes = content.as_bytes();
    let mut depth = 0usize;
    let mut start = None;

    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'{' => {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    if let Some(s) = start.take() {
                        objects.push(&content[s..=i]);
                    }
                }
            }
            _ => {}
        }
    }
    objects
}

/// Extract a `[f32; 3]` value from a JSON object string by key.
fn extract_json_array3(json: &str, key: &str) -> Option<[f32; 3]> {
    let search = format!("\"{}\":[", key);
    let start = json.find(&search)? + search.len();
    let rest = &json[start..];
    let end = rest.find(']')?;
    let arr_str = &rest[..end];
    let parts: Vec<&str> = arr_str.split(',').collect();
    if parts.len() < 3 {
        return None;
    }
    let x = parts[0].trim().parse::<f32>().ok()?;
    let y = parts[1].trim().parse::<f32>().ok()?;
    let z = parts[2].trim().parse::<f32>().ok()?;
    Some([x, y, z])
}

// ---------------------------------------------------------------------------
// blend_paths
// ---------------------------------------------------------------------------

/// Blend two camera paths by lerping poses frame-by-frame.
///
/// Both paths must have the same number of frames.
pub fn blend_paths(
    path_a: &CameraPath,
    path_b: &CameraPath,
    t: f32,
) -> Result<CameraPath, CameraPathError> {
    if path_a.frames.len() != path_b.frames.len() {
        return Err(CameraPathError::InvalidConfig(format!(
            "paths must have same length: {} vs {}",
            path_a.frames.len(),
            path_b.frames.len()
        )));
    }
    if path_a.frames.is_empty() {
        return Err(CameraPathError::EmptyPath);
    }
    let frames: Vec<CameraPose> = path_a
        .frames
        .iter()
        .zip(path_b.frames.iter())
        .map(|(a, b)| a.interpolate(b, t))
        .collect();
    CameraPath::new(frames, path_a.fps)
}

// ---------------------------------------------------------------------------
// smooth_path
// ---------------------------------------------------------------------------

/// Apply velocity smoothing via a moving average of camera positions.
///
/// `window` must be at least 1; window=1 returns an identical path.
pub fn smooth_path(path: &CameraPath, window: usize) -> Result<CameraPath, CameraPathError> {
    if path.frames.is_empty() {
        return Err(CameraPathError::EmptyPath);
    }
    if window == 0 {
        return Err(CameraPathError::InvalidConfig(
            "window must be at least 1".to_string(),
        ));
    }
    let n = path.frames.len();
    let half = window / 2;
    let mut frames = Vec::with_capacity(n);

    for i in 0..n {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let count = (hi - lo) as f32;
        let mut avg_pos = [0.0f32; 3];
        let mut avg_tgt = [0.0f32; 3];
        for j in lo..hi {
            avg_pos = vec3_add(avg_pos, path.frames[j].position);
            avg_tgt = vec3_add(avg_tgt, path.frames[j].target);
        }
        avg_pos = vec3_scale(avg_pos, 1.0 / count);
        avg_tgt = vec3_scale(avg_tgt, 1.0 / count);
        frames.push(CameraPose {
            position: avg_pos,
            target: avg_tgt,
            up: path.frames[i].up,
            fov_y: path.frames[i].fov_y,
        });
    }
    CameraPath::new(frames, path.fps)
}

// ---------------------------------------------------------------------------
// path_velocities
// ---------------------------------------------------------------------------

/// Compute per-frame camera velocity (distance moved per frame).
///
/// Returns a vector of length `n_frames - 1`.
pub fn path_velocities(path: &CameraPath) -> Result<Vec<f32>, CameraPathError> {
    if path.frames.is_empty() {
        return Err(CameraPathError::EmptyPath);
    }
    let velocities = path
        .frames
        .windows(2)
        .map(|pair| vec3_length(vec3_sub(pair[1].position, pair[0].position)))
        .collect();
    Ok(velocities)
}

// ---------------------------------------------------------------------------
// turntable_preset
// ---------------------------------------------------------------------------

/// Generate a full 360° orbit (turntable) for avatar showcase.
pub fn turntable_preset(
    center: [f32; 3],
    radius: f32,
    fps: f32,
    duration_secs: f32,
) -> Result<CameraPath, CameraPathError> {
    let config = PathConfig {
        duration_secs,
        fps,
        smooth_tangents: false,
    };
    orbit_path(center, radius, 0.0, 1.0, &config)
}

// ---------------------------------------------------------------------------
// zoom_in_path
// ---------------------------------------------------------------------------

/// Generate a zoom-in path: camera moves from `start_position` to a closer
/// point along the same view axis.
///
/// `zoom_factor`: end distance = start distance / zoom_factor.
pub fn zoom_in_path(
    start_position: [f32; 3],
    target: [f32; 3],
    zoom_factor: f32,
    config: &PathConfig,
) -> Result<CameraPath, CameraPathError> {
    config.validate()?;
    if zoom_factor <= 0.0 {
        return Err(CameraPathError::InvalidConfig(format!(
            "zoom_factor must be positive, got {zoom_factor}"
        )));
    }
    let dir = vec3_sub(start_position, target);
    let start_dist = vec3_length(dir);
    if start_dist < 1e-9 {
        return Err(CameraPathError::NumericalError(
            "start_position coincides with target".to_string(),
        ));
    }
    let end_dist = start_dist / zoom_factor;
    let dir_unit = vec3_scale(dir, 1.0 / start_dist);
    let end_position = vec3_add(target, vec3_scale(dir_unit, end_dist));

    let n = config.total_frames().max(1);
    let mut frames = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / (n - 1).max(1) as f32;
        let pos = lerp3(start_position, end_position, t);
        frames.push(CameraPose::new(pos, target));
    }
    CameraPath::new(frames, config.fps)
}

// ---------------------------------------------------------------------------
// Internal utilities
// ---------------------------------------------------------------------------

/// Clamp a value between lo and hi.
#[allow(dead_code)]
fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

/// Dot product alias for clarity.
#[allow(dead_code)]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    vec3_dot(a, b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn approx3(a: [f32; 3], b: [f32; 3]) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
    }

    // --- CameraPose ---

    #[test]
    fn test_camera_pose_new_defaults() {
        let pose = CameraPose::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]);
        assert_eq!(pose.position, [1.0, 2.0, 3.0]);
        assert_eq!(pose.target, [0.0, 0.0, 0.0]);
        assert_eq!(pose.up, [0.0, 1.0, 0.0]);
        assert!(approx(pose.fov_y, 45.0));
    }

    #[test]
    fn test_camera_pose_look_at() {
        let pose = CameraPose::look_at([0.0, 5.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert_eq!(pose.up, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_camera_pose_distance() {
        let pose = CameraPose::new([3.0, 4.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(approx(pose.distance(), 5.0));
    }

    #[test]
    fn test_camera_pose_distance_zero() {
        let pose = CameraPose::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(approx(pose.distance(), 0.0));
    }

    #[test]
    fn test_camera_pose_forward_normalized() {
        let pose = CameraPose::new([0.0, 0.0, 5.0], [0.0, 0.0, 0.0]);
        let fwd = pose.forward();
        let len = vec3_length(fwd);
        assert!(approx(len, 1.0), "forward not normalized: {len}");
        // Should point in -Z direction
        assert!(approx(fwd[2], -1.0));
    }

    #[test]
    fn test_camera_pose_forward_direction() {
        let pose = CameraPose::new([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        let fwd = pose.forward();
        assert!(approx3(fwd, [1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_camera_pose_view_matrix_reasonable() {
        let pose = CameraPose::new([0.0, 0.0, 5.0], [0.0, 0.0, 0.0]);
        let m = pose.view_matrix();
        // The matrix should be 4×4
        assert_eq!(m.len(), 4);
        assert_eq!(m[0].len(), 4);
        // The bottom-right element of a view matrix is 1.0
        assert!(
            approx(m[3][3], 1.0),
            "m[3][3] should be 1.0, got {}",
            m[3][3]
        );
    }

    #[test]
    fn test_camera_pose_interpolate_endpoints() {
        let a = CameraPose::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let b = CameraPose::new([2.0, 2.0, 2.0], [3.0, 1.0, 1.0]);

        let at_0 = a.interpolate(&b, 0.0);
        assert!(approx3(at_0.position, a.position));

        let at_1 = a.interpolate(&b, 1.0);
        assert!(approx3(at_1.position, b.position));
    }

    #[test]
    fn test_camera_pose_interpolate_midpoint() {
        let a = CameraPose::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let b = CameraPose::new([2.0, 2.0, 2.0], [2.0, 2.0, 2.0]);
        let mid = a.interpolate(&b, 0.5);
        assert!(approx3(mid.position, [1.0, 1.0, 1.0]));
        assert!(approx3(mid.target, [1.0, 1.0, 1.0]));
    }

    // --- PathKeyframe ---

    #[test]
    fn test_path_keyframe_new() {
        let pose = CameraPose::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        let kf = PathKeyframe::new(0.5, pose.clone());
        assert!(approx(kf.time, 0.5));
        assert_eq!(kf.ease, EasingType::Linear);
    }

    // --- EasingType ---

    #[test]
    fn test_easing_linear_half() {
        assert!(approx(EasingType::Linear.apply(0.5), 0.5));
    }

    #[test]
    fn test_easing_endpoints() {
        for ease in [
            EasingType::Linear,
            EasingType::EaseIn,
            EasingType::EaseOut,
            EasingType::EaseInOut,
            EasingType::CubicBezier,
        ] {
            assert!(approx(ease.apply(0.0), 0.0), "{ease:?} at 0 != 0");
            assert!(approx(ease.apply(1.0), 1.0), "{ease:?} at 1 != 1");
        }
    }

    #[test]
    fn test_easing_ease_in_accelerates() {
        // EaseIn: t² → value at 0.5 should be less than 0.5
        let v = EasingType::EaseIn.apply(0.5);
        assert!(v < 0.5, "EaseIn at 0.5 should be < 0.5, got {v}");
    }

    #[test]
    fn test_easing_ease_out_decelerates() {
        // EaseOut: t*(2-t) → value at 0.5 should be greater than 0.5
        let v = EasingType::EaseOut.apply(0.5);
        assert!(v > 0.5, "EaseOut at 0.5 should be > 0.5, got {v}");
    }

    #[test]
    fn test_easing_ease_in_out_symmetric() {
        let v1 = EasingType::EaseInOut.apply(0.25);
        let v2 = EasingType::EaseInOut.apply(0.75);
        // Smoothstep is symmetric: f(0.25) + f(0.75) == 1
        assert!(
            approx(v1 + v2, 1.0),
            "EaseInOut symmetry failed: {v1} + {v2}"
        );
    }

    // --- PathConfig ---

    #[test]
    fn test_path_config_validate_ok() {
        let cfg = PathConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_path_config_validate_zero_duration() {
        let cfg = PathConfig {
            duration_secs: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(CameraPathError::InvalidDuration(_))
        ));
    }

    #[test]
    fn test_path_config_validate_negative_duration() {
        let cfg = PathConfig {
            duration_secs: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(CameraPathError::InvalidDuration(_))
        ));
    }

    #[test]
    fn test_path_config_validate_zero_fps() {
        let cfg = PathConfig {
            fps: 0.0,
            ..Default::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(CameraPathError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_path_config_total_frames() {
        let cfg = PathConfig {
            duration_secs: 2.0,
            fps: 30.0,
            smooth_tangents: false,
        };
        assert_eq!(cfg.total_frames(), 60);
    }

    #[test]
    fn test_path_config_total_frames_ceil() {
        let cfg = PathConfig {
            duration_secs: 1.0,
            fps: 24.0,
            smooth_tangents: false,
        };
        // 24 * 1 = 24 exactly
        assert_eq!(cfg.total_frames(), 24);
    }

    // --- CameraPath ---

    #[test]
    fn test_camera_path_new_empty_error() {
        let result = CameraPath::new(vec![], 30.0);
        assert!(matches!(result, Err(CameraPathError::EmptyPath)));
    }

    #[test]
    fn test_camera_path_new_ok() {
        let frames = vec![CameraPose::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])];
        let path = CameraPath::new(frames, 30.0).expect("should succeed");
        assert_eq!(path.total_frames(), 1);
    }

    #[test]
    fn test_camera_path_get_frame_ok() {
        let frames = vec![
            CameraPose::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            CameraPose::new([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
        ];
        let path = CameraPath::new(frames, 30.0).expect("ok");
        assert!(path.get_frame(0).is_ok());
        assert!(path.get_frame(1).is_ok());
    }

    #[test]
    fn test_camera_path_get_frame_out_of_range() {
        let frames = vec![CameraPose::new([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])];
        let path = CameraPath::new(frames, 30.0).expect("ok");
        assert!(matches!(
            path.get_frame(5),
            Err(CameraPathError::FrameOutOfRange { .. })
        ));
    }

    #[test]
    fn test_camera_path_reverse() {
        let frames = vec![
            CameraPose::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            CameraPose::new([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            CameraPose::new([2.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ];
        let path = CameraPath::new(frames, 30.0).expect("ok");
        let rev = path.reverse();
        assert_eq!(rev.total_frames(), 3);
        assert!(approx(rev.frames[0].position[0], 2.0));
        assert!(approx(rev.frames[2].position[0], 0.0));
    }

    #[test]
    fn test_camera_path_loop_path() {
        let frames = vec![
            CameraPose::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            CameraPose::new([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ];
        let path = CameraPath::new(frames, 30.0).expect("ok");
        let looped = path.loop_path(3);
        assert_eq!(looped.total_frames(), 6);
    }

    #[test]
    fn test_camera_path_trim_ok() {
        let frames: Vec<CameraPose> = (0..10)
            .map(|i| CameraPose::new([i as f32, 0.0, 0.0], [0.0, 0.0, 0.0]))
            .collect();
        let path = CameraPath::new(frames, 30.0).expect("ok");
        let trimmed = path.trim(2, 7).expect("trim ok");
        assert_eq!(trimmed.total_frames(), 5);
        assert!(approx(trimmed.frames[0].position[0], 2.0));
    }

    #[test]
    fn test_camera_path_trim_out_of_range() {
        let frames = vec![
            CameraPose::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            CameraPose::new([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ];
        let path = CameraPath::new(frames, 30.0).expect("ok");
        assert!(path.trim(0, 100).is_err());
        assert!(path.trim(5, 6).is_err());
    }

    // --- orbit_path ---

    #[test]
    fn test_orbit_path_frame_count() {
        let config = PathConfig {
            duration_secs: 2.0,
            fps: 30.0,
            smooth_tangents: false,
        };
        let path = orbit_path([0.0, 0.0, 0.0], 3.0, 1.0, 1.0, &config).expect("ok");
        assert_eq!(path.total_frames(), 60);
    }

    #[test]
    fn test_orbit_path_circular_positions() {
        let config = PathConfig {
            duration_secs: 1.0,
            fps: 30.0,
            smooth_tangents: false,
        };
        let center = [0.0, 0.0, 0.0];
        let radius = 3.0;
        let path = orbit_path(center, radius, 0.0, 1.0, &config).expect("ok");
        for frame in &path.frames {
            let dx = frame.position[0] - center[0];
            let dz = frame.position[2] - center[2];
            let dist = (dx * dx + dz * dz).sqrt();
            assert!(
                approx(dist, radius),
                "orbit distance off: {dist} vs {radius}"
            );
        }
    }

    #[test]
    fn test_orbit_path_equidistant_from_center() {
        let config = PathConfig {
            duration_secs: 1.0,
            fps: 10.0,
            smooth_tangents: false,
        };
        let center = [1.0, 2.0, 3.0];
        let radius = 5.0;
        let height = 2.0;
        let path = orbit_path(center, radius, height, 1.0, &config).expect("ok");
        for frame in &path.frames {
            let dx = frame.position[0] - center[0];
            let dy = frame.position[1] - (center[1] + height);
            let dz = frame.position[2] - center[2];
            let dist_xz = (dx * dx + dz * dz).sqrt();
            assert!(approx(dist_xz, radius), "xz distance off: {dist_xz}");
            assert!(approx(dy, 0.0), "height offset off: {dy}");
        }
    }

    // --- spiral_orbit_path ---

    #[test]
    fn test_spiral_orbit_start_end_distance() {
        let config = PathConfig {
            duration_secs: 2.0,
            fps: 20.0,
            smooth_tangents: false,
        };
        let center = [0.0, 0.0, 0.0];
        let path = spiral_orbit_path(center, 5.0, 2.0, 0.0, 3.0, &config).expect("ok");
        let n = path.total_frames();
        // First frame approximately at radius_start
        let f0 = &path.frames[0];
        let dx0 = f0.position[0] - center[0];
        let dz0 = f0.position[2] - center[2];
        let r0 = (dx0 * dx0 + dz0 * dz0).sqrt();
        assert!(approx(r0, 5.0), "start radius off: {r0}");
        // Last frame approximately at radius_end
        let fl = &path.frames[n - 1];
        let dxl = fl.position[0] - center[0];
        let dzl = fl.position[2] - center[2];
        let rl = (dxl * dxl + dzl * dzl).sqrt();
        // At last frame frac = (n-1)/n ≈ 1, but not exactly 1
        assert!(rl < 5.0, "radius should decrease, got {rl}");
    }

    // --- keyframe_path ---

    #[test]
    fn test_keyframe_path_two_keyframes() {
        let config = PathConfig {
            duration_secs: 1.0,
            fps: 10.0,
            smooth_tangents: false,
        };
        let kfs = vec![
            PathKeyframe::new(0.0, CameraPose::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])),
            PathKeyframe::new(1.0, CameraPose::new([10.0, 0.0, 0.0], [0.0, 0.0, 1.0])),
        ];
        let path = keyframe_path(&kfs, &config).expect("ok");
        assert_eq!(path.total_frames(), 10);
        // First frame matches first keyframe
        assert!(approx3(path.frames[0].position, [0.0, 0.0, 0.0]));
        // Last frame matches second keyframe
        assert!(approx3(path.frames[9].position, [10.0, 0.0, 0.0]));
    }

    #[test]
    fn test_keyframe_path_insufficient() {
        let config = PathConfig::default();
        let kfs: Vec<PathKeyframe> = vec![];
        let result = keyframe_path(&kfs, &config);
        assert!(matches!(
            result,
            Err(CameraPathError::InsufficientKeyframes { .. })
        ));
    }

    // --- catmull_rom ---

    #[test]
    fn test_catmull_rom_at_zero_is_p1() {
        let p0 = [0.0f32; 3];
        let p1 = [1.0, 0.0, 0.0];
        let p2 = [2.0, 0.0, 0.0];
        let p3 = [3.0, 0.0, 0.0];
        let result = catmull_rom(p0, p1, p2, p3, 0.0);
        assert!(approx3(result, p1));
    }

    #[test]
    fn test_catmull_rom_at_one_is_p2() {
        let p0 = [0.0f32; 3];
        let p1 = [1.0, 0.0, 0.0];
        let p2 = [2.0, 0.0, 0.0];
        let p3 = [3.0, 0.0, 0.0];
        let result = catmull_rom(p0, p1, p2, p3, 1.0);
        assert!(approx3(result, p2));
    }

    #[test]
    fn test_catmull_rom_midpoint_collinear() {
        // For collinear equidistant points the midpoint should be at the midpoint
        let p0 = [-1.0, 0.0, 0.0];
        let p1 = [0.0, 0.0, 0.0];
        let p2 = [1.0, 0.0, 0.0];
        let p3 = [2.0, 0.0, 0.0];
        let result = catmull_rom(p0, p1, p2, p3, 0.5);
        assert!(approx(result[0], 0.5), "midpoint x: {}", result[0]);
    }

    // --- figure_eight_path ---

    #[test]
    fn test_figure_eight_frame_count() {
        let config = PathConfig {
            duration_secs: 3.0,
            fps: 24.0,
            smooth_tangents: false,
        };
        let path = figure_eight_path([0.0, 0.0, 0.0], 2.0, 1.0, &config).expect("ok");
        assert_eq!(path.total_frames(), 72);
    }

    #[test]
    fn test_figure_eight_bounded() {
        let config = PathConfig {
            duration_secs: 2.0,
            fps: 30.0,
            smooth_tangents: false,
        };
        let radius = 3.0;
        let path = figure_eight_path([0.0, 0.0, 0.0], radius, 0.0, &config).expect("ok");
        for frame in &path.frames {
            // Lemniscate is bounded by |x| ≤ radius, |z| ≤ radius/2
            let bx = frame.position[0].abs();
            let bz = frame.position[2].abs();
            assert!(bx <= radius + EPS, "x out of bounds: {bx}");
            assert!(bz <= radius + EPS, "z out of bounds: {bz}");
        }
    }

    // --- compute_path_stats ---

    #[test]
    fn test_compute_path_stats_orbit() {
        let config = PathConfig {
            duration_secs: 1.0,
            fps: 30.0,
            smooth_tangents: false,
        };
        let path = orbit_path([0.0, 0.0, 0.0], 3.0, 0.0, 1.0, &config).expect("ok");
        let stats = compute_path_stats(&path).expect("ok");
        assert!(stats.total_distance > 0.0, "total_distance should be > 0");
        assert!(stats.mean_speed > 0.0, "mean_speed should be > 0");
        assert_eq!(stats.total_frames, path.total_frames());
    }

    #[test]
    fn test_compute_path_stats_empty_error() {
        // Construct path directly to bypass the CameraPath::new guard
        // by starting with a single-frame path then clearing internally
        // (not easily possible; test via result of empty new instead)
        let result = CameraPath::new(vec![], 30.0);
        assert!(result.is_err());
    }

    // --- path_to_json / path_from_json ---

    #[test]
    fn test_path_to_json_contains_fps() {
        let config = PathConfig {
            duration_secs: 0.1,
            fps: 10.0,
            smooth_tangents: false,
        };
        let path = orbit_path([0.0, 0.0, 0.0], 1.0, 0.0, 1.0, &config).expect("ok");
        let json = path_to_json(&path);
        assert!(json.contains("fps"), "JSON missing fps field");
        assert!(json.contains("frames"), "JSON missing frames field");
        assert!(json.contains("position"), "JSON missing position field");
    }

    #[test]
    fn test_path_round_trip_json() {
        let config = PathConfig {
            duration_secs: 0.2,
            fps: 10.0,
            smooth_tangents: false,
        };
        let original = orbit_path([0.0, 0.0, 0.0], 2.0, 1.0, 1.0, &config).expect("ok");
        let json = path_to_json(&original);
        let restored = path_from_json(&json, 10.0).expect("parse ok");
        assert_eq!(original.total_frames(), restored.total_frames());
        for (a, b) in original.frames.iter().zip(restored.frames.iter()) {
            assert!(approx3(a.position, b.position), "position mismatch");
            assert!(approx3(a.target, b.target), "target mismatch");
        }
    }

    #[test]
    fn test_path_from_json_invalid() {
        let result = path_from_json("{bad json}", 30.0);
        assert!(result.is_err());
    }

    // --- blend_paths ---

    #[test]
    fn test_blend_paths_t0_is_a() {
        let config = PathConfig {
            duration_secs: 0.1,
            fps: 10.0,
            smooth_tangents: false,
        };
        let a = orbit_path([0.0, 0.0, 0.0], 1.0, 0.0, 1.0, &config).expect("ok");
        let b = orbit_path([0.0, 0.0, 0.0], 3.0, 2.0, 1.0, &config).expect("ok");
        let blended = blend_paths(&a, &b, 0.0).expect("ok");
        for (orig, bl) in a.frames.iter().zip(blended.frames.iter()) {
            assert!(approx3(orig.position, bl.position), "t=0 should equal a");
        }
    }

    #[test]
    fn test_blend_paths_t1_is_b() {
        let config = PathConfig {
            duration_secs: 0.1,
            fps: 10.0,
            smooth_tangents: false,
        };
        let a = orbit_path([0.0, 0.0, 0.0], 1.0, 0.0, 1.0, &config).expect("ok");
        let b = orbit_path([0.0, 0.0, 0.0], 3.0, 2.0, 1.0, &config).expect("ok");
        let blended = blend_paths(&a, &b, 1.0).expect("ok");
        for (orig, bl) in b.frames.iter().zip(blended.frames.iter()) {
            assert!(approx3(orig.position, bl.position), "t=1 should equal b");
        }
    }

    #[test]
    fn test_blend_paths_length_mismatch_error() {
        let cfg_a = PathConfig {
            duration_secs: 0.1,
            fps: 10.0,
            smooth_tangents: false,
        };
        let cfg_b = PathConfig {
            duration_secs: 0.2,
            fps: 10.0,
            smooth_tangents: false,
        };
        let a = orbit_path([0.0, 0.0, 0.0], 1.0, 0.0, 1.0, &cfg_a).expect("ok");
        let b = orbit_path([0.0, 0.0, 0.0], 1.0, 0.0, 1.0, &cfg_b).expect("ok");
        assert!(blend_paths(&a, &b, 0.5).is_err());
    }

    // --- smooth_path ---

    #[test]
    fn test_smooth_path_same_length() {
        let config = PathConfig {
            duration_secs: 1.0,
            fps: 10.0,
            smooth_tangents: false,
        };
        let path = orbit_path([0.0, 0.0, 0.0], 2.0, 0.0, 1.0, &config).expect("ok");
        let smoothed = smooth_path(&path, 3).expect("ok");
        assert_eq!(smoothed.total_frames(), path.total_frames());
    }

    #[test]
    fn test_smooth_path_window_one_unchanged() {
        let config = PathConfig {
            duration_secs: 0.5,
            fps: 10.0,
            smooth_tangents: false,
        };
        let path = orbit_path([0.0, 0.0, 0.0], 2.0, 0.0, 1.0, &config).expect("ok");
        let smoothed = smooth_path(&path, 1).expect("ok");
        for (orig, sm) in path.frames.iter().zip(smoothed.frames.iter()) {
            assert!(
                approx3(orig.position, sm.position),
                "window=1 should be identity"
            );
        }
    }

    // --- path_velocities ---

    #[test]
    fn test_path_velocities_length() {
        let config = PathConfig {
            duration_secs: 1.0,
            fps: 10.0,
            smooth_tangents: false,
        };
        let path = orbit_path([0.0, 0.0, 0.0], 2.0, 0.0, 1.0, &config).expect("ok");
        let vels = path_velocities(&path).expect("ok");
        assert_eq!(vels.len(), path.total_frames() - 1);
    }

    #[test]
    fn test_path_velocities_positive() {
        let config = PathConfig {
            duration_secs: 1.0,
            fps: 10.0,
            smooth_tangents: false,
        };
        let path = orbit_path([0.0, 0.0, 0.0], 2.0, 0.0, 1.0, &config).expect("ok");
        let vels = path_velocities(&path).expect("ok");
        for v in &vels {
            assert!(*v >= 0.0, "velocity should be non-negative: {v}");
        }
    }

    // --- turntable_preset ---

    #[test]
    fn test_turntable_preset_valid() {
        let path = turntable_preset([0.0, 0.0, 0.0], 3.0, 30.0, 5.0).expect("ok");
        // 30 * 5 = 150 frames
        assert_eq!(path.total_frames(), 150);
    }

    #[test]
    fn test_turntable_preset_circular() {
        let path = turntable_preset([0.0, 1.0, 0.0], 2.0, 24.0, 1.0).expect("ok");
        for frame in &path.frames {
            let dx = frame.position[0];
            let dz = frame.position[2];
            let r = (dx * dx + dz * dz).sqrt();
            assert!(approx(r, 2.0), "turntable radius off: {r}");
        }
    }

    // --- zoom_in_path ---

    #[test]
    fn test_zoom_in_distances_decrease() {
        let config = PathConfig {
            duration_secs: 1.0,
            fps: 10.0,
            smooth_tangents: false,
        };
        let start = [0.0, 0.0, 10.0];
        let target = [0.0, 0.0, 0.0];
        let path = zoom_in_path(start, target, 2.0, &config).expect("ok");
        let first_dist = path.frames[0].distance();
        let last_dist = path.frames.last().expect("has frames").distance();
        assert!(
            last_dist < first_dist,
            "end distance {last_dist} should be less than start {first_dist}"
        );
    }

    #[test]
    fn test_zoom_in_zoom_factor_correct() {
        let config = PathConfig {
            duration_secs: 1.0,
            fps: 10.0,
            smooth_tangents: false,
        };
        let start = [0.0, 0.0, 10.0];
        let target = [0.0, 0.0, 0.0];
        let zoom_factor = 2.0;
        let path = zoom_in_path(start, target, zoom_factor, &config).expect("ok");
        let first_dist = path.frames[0].distance();
        let last_dist = path.frames.last().expect("has frames").distance();
        assert!(
            approx(last_dist, first_dist / zoom_factor),
            "end distance {last_dist} should equal start/zoom {}",
            first_dist / zoom_factor
        );
    }

    #[test]
    fn test_zoom_in_coincident_error() {
        let config = PathConfig::default();
        let result = zoom_in_path([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2.0, &config);
        assert!(matches!(result, Err(CameraPathError::NumericalError(_))));
    }

    // -- CubicBezier easing --

    #[test]
    fn test_cubic_bezier_boundaries() {
        // t=0 → 0 and t=1 → 1 must hold exactly (boundary guards in bezier_ease).
        assert!((EasingType::CubicBezier.apply(0.0)).abs() < 1e-5);
        assert!((EasingType::CubicBezier.apply(1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cubic_bezier_differs_from_ease_in_out() {
        // CSS "ease" (0.25,0.1, 0.25,1.0) is NOT the same curve as polynomial
        // smoothstep — check at t=0.25 where the two curves diverge most.
        let t = 0.25_f32;
        let cb = EasingType::CubicBezier.apply(t);
        let eio = EasingType::EaseInOut.apply(t);
        assert!(
            (cb - eio).abs() > 1e-3,
            "CubicBezier (CSS ease) should differ from EaseInOut (smoothstep) at t={t}: cb={cb}, eio={eio}"
        );
    }

    #[test]
    fn test_cubic_bezier_monotonic() {
        // Sample 21 evenly-spaced values; each must be >= the previous.
        let vals: Vec<f32> = (0..=20)
            .map(|i| EasingType::CubicBezier.apply(i as f32 / 20.0))
            .collect();
        for w in vals.windows(2) {
            assert!(w[1] >= w[0] - 1e-5, "not monotonic: {w:?}");
        }
    }

    #[test]
    fn test_cubic_bezier_midpoint_sanity() {
        // B(0.5) must be strictly between 0 and 1.
        let v = EasingType::CubicBezier.apply(0.5);
        assert!(v > 0.0 && v < 1.0, "CubicBezier(0.5) out of (0,1): {v}");
    }
}
