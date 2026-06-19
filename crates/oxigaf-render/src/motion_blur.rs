//! Motion blur for 3D Gaussian Splatting rendering.
//!
//! Motion blur simulates camera or object motion during frame exposure.
//! Two approaches are provided:
//!
//! 1. **Accumulation-based** ([`accumulate_frames`]): Accumulates multiple
//!    pre-rendered sub-frame images along the camera motion path.  Each
//!    sub-frame is rendered at a slightly different camera pose and then
//!    blended with configurable weights.
//!
//! 2. **Velocity-based** ([`apply_velocity_blur`]): An image-space
//!    post-processing effect that scatters/gathers colour samples along a
//!    per-pixel 2-D velocity vector derived from consecutive frames.
//!
//! # Module overview
//!
//! ```text
//! CameraMotion          – lerp-interpolated camera pose during exposure
//! AccumulationConfig    – controls temporal super-sampling accumulation
//! VelocityField         – per-pixel screen-space velocity map
//! VelocityBlurConfig    – parameters for image-space velocity blur
//! AccumulatedBlur       – result of accumulate_frames()
//! MotionBlurStats       – diagnostic statistics
//! ```

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by motion-blur operations.
#[derive(Debug, Error)]
pub enum MotionBlurError {
    /// Configuration is invalid (e.g. num_samples = 0).
    #[error("Invalid motion blur configuration: {0}")]
    InvalidConfig(String),

    /// Image slice has the wrong number of pixels.
    #[error("Invalid image data: {0}")]
    InvalidImage(String),

    /// No frames were supplied to accumulate.
    #[error("No frames supplied for accumulation")]
    EmptyFrames,

    /// A buffer length does not match the expected dimension.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected flat length.
        expected: usize,
        /// Actual flat length.
        got: usize,
    },

    // ── image-space post-processing variants ──────────────────────────────────
    /// Image buffer is empty.
    #[error("Empty image buffer")]
    EmptyImage,

    /// Image buffer byte count does not match declared dimensions.
    #[error("Image buffer {actual} bytes does not match {width}×{height}×4 = {expected}")]
    InvalidDimensions {
        /// Actual buffer length.
        actual: usize,
        /// Expected buffer length.
        expected: usize,
        /// Declared image width.
        width: u32,
        /// Declared image height.
        height: u32,
    },

    /// Motion vector buffer length does not match pixel count × 2.
    #[error("Motion vector buffer length {mv_len} does not match image pixels {px_len}×2")]
    MotionVectorMismatch {
        /// Actual motion vector buffer length.
        mv_len: usize,
        /// Number of image pixels (width × height).
        px_len: usize,
    },

    /// Sample count is zero.
    #[error("Invalid sample count {samples}: must be >= 1")]
    InvalidSampleCount {
        /// The invalid sample count.
        samples: usize,
    },

    /// Shutter angle is out of the valid range `(0, 360]`.
    #[error("Shutter angle {angle} out of range (0, 360]")]
    InvalidShutterAngle {
        /// The invalid shutter angle.
        angle: f32,
    },

    /// Input buffer was empty.
    #[error("empty input")]
    EmptyInput,
}

// ─────────────────────────────────────────────────────────────────────────────
// Motion trajectory helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Linear interpolation factor for sub-frame `i` of `n`.
///
/// Returns a value in `[0, 1]` representing the position within the exposure
/// window. When `n == 1` the only valid index is `i = 0`, which returns `0.0`.
///
/// # Examples
///
/// ```
/// use oxigaf_render::subframe_t;
/// assert!((subframe_t(0, 1) - 0.0).abs() < 1e-6);
/// assert!((subframe_t(0, 4) - 0.0).abs() < 1e-6);
/// assert!((subframe_t(3, 4) - 1.0).abs() < 1e-6);
/// ```
pub fn subframe_t(i: usize, n: usize) -> f32 {
    i as f32 / (n.saturating_sub(1).max(1)) as f32
}

/// Linearly interpolate between two 3-D positions.
///
/// `t = 0` returns `start`; `t = 1` returns `end`.  Values outside `[0, 1]`
/// are allowed (extrapolation).
pub fn lerp_position(start: [f32; 3], end: [f32; 3], t: f32) -> [f32; 3] {
    [
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t,
        start[2] + (end[2] - start[2]) * t,
    ]
}

/// Element-wise linear interpolation between two quaternions.
///
/// **Note**: The result is *not* normalised.  Call [`normalize_quaternion`]
/// afterwards if a unit quaternion is required.  This is sufficient for small
/// angular differences between adjacent sub-frames.
pub fn lerp_quaternion(q0: [f32; 4], q1: [f32; 4], t: f32) -> [f32; 4] {
    [
        q0[0] + (q1[0] - q0[0]) * t,
        q0[1] + (q1[1] - q0[1]) * t,
        q0[2] + (q1[2] - q0[2]) * t,
        q0[3] + (q1[3] - q0[3]) * t,
    ]
}

/// Normalise a quaternion to unit length.
///
/// If the magnitude is below `1e-9`, returns the identity quaternion
/// `[0, 0, 0, 1]`.
pub fn normalize_quaternion(q: [f32; 4]) -> [f32; 4] {
    let mag_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if mag_sq < 1e-9_f32 * 1e-9_f32 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let inv = 1.0 / mag_sq.sqrt();
    [q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv]
}

// ─────────────────────────────────────────────────────────────────────────────
// CameraMotion
// ─────────────────────────────────────────────────────────────────────────────

/// Camera pose motion during one frame's exposure window.
///
/// The camera moves linearly (in position) and via nlerp (in rotation) from
/// `*_start` to `*_end` over `t ∈ [0, 1]`.
#[derive(Debug, Clone)]
pub struct CameraMotion {
    /// Start position `[x, y, z]`.
    pub position_start: [f32; 3],
    /// End position `[x, y, z]`.
    pub position_end: [f32; 3],
    /// Start rotation quaternion `[x, y, z, w]`.
    pub rotation_start: [f32; 4],
    /// End rotation quaternion `[x, y, z, w]`.
    pub rotation_end: [f32; 4],
}

impl CameraMotion {
    /// Construct a new camera motion with explicit start/end poses.
    pub fn new(
        pos_start: [f32; 3],
        pos_end: [f32; 3],
        rot_start: [f32; 4],
        rot_end: [f32; 4],
    ) -> Self {
        Self {
            position_start: pos_start,
            position_end: pos_end,
            rotation_start: rot_start,
            rotation_end: rot_end,
        }
    }

    /// Construct a stationary camera (start == end).
    pub fn stationary(position: [f32; 3], rotation: [f32; 4]) -> Self {
        Self {
            position_start: position,
            position_end: position,
            rotation_start: rotation,
            rotation_end: rotation,
        }
    }

    /// Interpolated position at sub-frame parameter `t ∈ [0, 1]`.
    pub fn position_at(&self, t: f32) -> [f32; 3] {
        lerp_position(self.position_start, self.position_end, t)
    }

    /// Interpolated, normalised rotation at sub-frame parameter `t ∈ [0, 1]`.
    pub fn rotation_at(&self, t: f32) -> [f32; 4] {
        let lerped = lerp_quaternion(self.rotation_start, self.rotation_end, t);
        normalize_quaternion(lerped)
    }

    /// Euclidean distance between start and end positions.
    pub fn translation_distance(&self) -> f32 {
        let dx = self.position_end[0] - self.position_start[0];
        let dy = self.position_end[1] - self.position_start[1];
        let dz = self.position_end[2] - self.position_start[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// `true` if both position and rotation are identical at start and end
    /// (within a tolerance of `1e-6`).
    pub fn is_stationary(&self) -> bool {
        let pos_same = (self.position_start[0] - self.position_end[0]).abs() < 1e-6
            && (self.position_start[1] - self.position_end[1]).abs() < 1e-6
            && (self.position_start[2] - self.position_end[2]).abs() < 1e-6;
        let rot_same = (self.rotation_start[0] - self.rotation_end[0]).abs() < 1e-6
            && (self.rotation_start[1] - self.rotation_end[1]).abs() < 1e-6
            && (self.rotation_start[2] - self.rotation_end[2]).abs() < 1e-6
            && (self.rotation_start[3] - self.rotation_end[3]).abs() < 1e-6;
        pos_same && rot_same
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AccumulationConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for accumulation-based motion blur.
///
/// The blur is achieved by averaging `num_samples` pre-rendered sub-frame
/// images, each captured at a different point along the camera motion path.
#[derive(Debug, Clone)]
pub struct AccumulationConfig {
    /// Number of sub-frames to accumulate.  1 = no blur; 8–32 is realistic.
    pub num_samples: usize,
    /// Fraction of the frame duration during which the shutter is open.
    /// Must be in `(0, 1]`.
    pub shutter_open: f32,
    /// `true` → equal weights for every sample.
    /// `false` → triangle (ramp) weights, peaking at the centre sub-frame.
    pub uniform_weights: bool,
}

impl Default for AccumulationConfig {
    fn default() -> Self {
        Self {
            num_samples: 8,
            shutter_open: 1.0,
            uniform_weights: true,
        }
    }
}

impl AccumulationConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// - [`MotionBlurError::InvalidConfig`] if `num_samples < 1`.
    /// - [`MotionBlurError::InvalidConfig`] if `shutter_open` is not in `(0, 1]`.
    pub fn validate(&self) -> Result<(), MotionBlurError> {
        if self.num_samples < 1 {
            return Err(MotionBlurError::InvalidConfig(
                "num_samples must be at least 1".to_string(),
            ));
        }
        if self.shutter_open <= 0.0 || self.shutter_open > 1.0 {
            return Err(MotionBlurError::InvalidConfig(format!(
                "shutter_open must be in (0, 1], got {}",
                self.shutter_open
            )));
        }
        Ok(())
    }

    /// Compute per-sample weights that sum to `1.0`.
    ///
    /// - **Uniform** (`uniform_weights = true`): every weight is `1 / n`.
    /// - **Ramp** (`uniform_weights = false`): triangle weights, peaking at
    ///   the centre sample, then normalised.
    pub fn sample_weights(&self) -> Vec<f32> {
        let n = self.num_samples.max(1);
        if self.uniform_weights || n == 1 {
            return vec![1.0 / n as f32; n];
        }

        // Triangle weights: w[i] = 1 + min(i, n - 1 - i)
        let mut weights: Vec<f32> = (0..n)
            .map(|i| (1 + i.min(n.saturating_sub(1).saturating_sub(i))) as f32)
            .collect();

        let total: f32 = weights.iter().sum();
        if total > 0.0 {
            let inv = 1.0 / total;
            for w in &mut weights {
                *w *= inv;
            }
        }
        weights
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VelocityField
// ─────────────────────────────────────────────────────────────────────────────

/// Per-pixel 2-D screen-space velocity field (pixels per frame).
///
/// Velocities are stored in row-major order as `[vx, vy, vx, vy, ...]`.
/// Positive `vx` points right; positive `vy` points down.
#[derive(Debug, Clone)]
pub struct VelocityField {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Flat velocity buffer: `len = width * height * 2`.
    /// Layout: `[vx_0, vy_0, vx_1, vy_1, ...]` in row-major order.
    pub velocities: Vec<f32>,
}

impl VelocityField {
    /// Create a zero-initialised velocity field for an image of the given size.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            velocities: vec![0.0_f32; width * height * 2],
        }
    }

    /// Get the velocity `[vx, vy]` of pixel `(x, y)`.
    ///
    /// # Errors
    ///
    /// Returns [`MotionBlurError::DimensionMismatch`] if `(x, y)` is out of
    /// bounds.
    pub fn get(&self, x: usize, y: usize) -> Result<[f32; 2], MotionBlurError> {
        if x >= self.width || y >= self.height {
            return Err(MotionBlurError::DimensionMismatch {
                expected: self.width * self.height,
                got: y * self.width + x,
            });
        }
        let idx = (y * self.width + x) * 2;
        Ok([self.velocities[idx], self.velocities[idx + 1]])
    }

    /// Set the velocity `[vx, vy]` of pixel `(x, y)`.
    ///
    /// # Errors
    ///
    /// Returns [`MotionBlurError::DimensionMismatch`] if `(x, y)` is out of
    /// bounds.
    pub fn set(&mut self, x: usize, y: usize, vx: f32, vy: f32) -> Result<(), MotionBlurError> {
        if x >= self.width || y >= self.height {
            return Err(MotionBlurError::DimensionMismatch {
                expected: self.width * self.height,
                got: y * self.width + x,
            });
        }
        let idx = (y * self.width + x) * 2;
        self.velocities[idx] = vx;
        self.velocities[idx + 1] = vy;
        Ok(())
    }

    /// Maximum velocity magnitude across all pixels.
    pub fn max_magnitude(&self) -> f32 {
        let mut max = 0.0_f32;
        for chunk in self.velocities.chunks_exact(2) {
            let mag = (chunk[0] * chunk[0] + chunk[1] * chunk[1]).sqrt();
            if mag > max {
                max = mag;
            }
        }
        max
    }

    /// Mean velocity magnitude across all pixels.
    pub fn mean_magnitude(&self) -> f32 {
        let n_pixels = self.width * self.height;
        if n_pixels == 0 {
            return 0.0;
        }
        let sum: f32 = self
            .velocities
            .chunks_exact(2)
            .map(|chunk| (chunk[0] * chunk[0] + chunk[1] * chunk[1]).sqrt())
            .sum();
        sum / n_pixels as f32
    }

    /// Compute a velocity field from two consecutive world-space position maps.
    ///
    /// Each position map stores `[x, y, z]` per pixel in row-major order, so
    /// `pos_prev.len() == pos_curr.len() == width * height * 3`.
    ///
    /// The supplied `projection` matrix is a 4 × 4 row-major matrix that maps
    /// homogeneous world coordinates to clip space.  The resulting NDC
    /// difference is scaled to pixels (`vx *= width / 2`, `vy *= height / 2`).
    ///
    /// Pixels where either clip-space `w` coordinate is near zero are left at
    /// velocity zero.
    ///
    /// # Errors
    ///
    /// - [`MotionBlurError::DimensionMismatch`] if either position map does
    ///   not have `width * height * 3` elements.
    pub fn from_position_maps(
        width: usize,
        height: usize,
        pos_prev: &[f32],
        pos_curr: &[f32],
        projection: &[[f32; 4]; 4],
    ) -> Result<Self, MotionBlurError> {
        let expected = width * height * 3;
        if pos_prev.len() != expected {
            return Err(MotionBlurError::DimensionMismatch {
                expected,
                got: pos_prev.len(),
            });
        }
        if pos_curr.len() != expected {
            return Err(MotionBlurError::DimensionMismatch {
                expected,
                got: pos_curr.len(),
            });
        }

        let n_pixels = width * height;
        let mut field = Self::new(width, height);

        for i in 0..n_pixels {
            let base3 = i * 3;

            let p_prev = [
                pos_prev[base3],
                pos_prev[base3 + 1],
                pos_prev[base3 + 2],
                1.0_f32,
            ];
            let p_curr = [
                pos_curr[base3],
                pos_curr[base3 + 1],
                pos_curr[base3 + 2],
                1.0_f32,
            ];

            let clip_prev = project_point(projection, p_prev);
            let clip_curr = project_point(projection, p_curr);

            // Skip degenerate w (behind camera or at infinity).
            const W_EPS: f32 = 1e-5;
            if clip_prev[3].abs() < W_EPS || clip_curr[3].abs() < W_EPS {
                continue;
            }

            let ndc_prev_x = clip_prev[0] / clip_prev[3];
            let ndc_prev_y = clip_prev[1] / clip_prev[3];
            let ndc_curr_x = clip_curr[0] / clip_curr[3];
            let ndc_curr_y = clip_curr[1] / clip_curr[3];

            let vx = (ndc_curr_x - ndc_prev_x) * (width as f32 / 2.0);
            let vy = (ndc_curr_y - ndc_prev_y) * (height as f32 / 2.0);

            let base2 = i * 2;
            field.velocities[base2] = vx;
            field.velocities[base2 + 1] = vy;
        }

        Ok(field)
    }
}

/// Apply a 4 × 4 row-major projection matrix to a homogeneous point.
///
/// Returns `[clip_x, clip_y, clip_z, clip_w]`.
fn project_point(m: &[[f32; 4]; 4], p: [f32; 4]) -> [f32; 4] {
    let mut out = [0.0_f32; 4];
    for (row_idx, row) in m.iter().enumerate() {
        out[row_idx] = row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + row[3] * p[3];
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// VelocityBlurConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for image-space velocity-based motion blur.
#[derive(Debug, Clone)]
pub struct VelocityBlurConfig {
    /// Number of samples taken along the velocity vector per pixel.
    pub num_samples: usize,
    /// Multiplier applied to the velocity before sampling.  `1.0` = full
    /// velocity extent is covered; values < 1.0 reduce the blur amount.
    pub velocity_scale: f32,
    /// Maximum blur radius in pixels.  Long velocity vectors are clamped to
    /// this length before sampling.
    pub max_radius: f32,
}

impl Default for VelocityBlurConfig {
    fn default() -> Self {
        Self {
            num_samples: 16,
            velocity_scale: 1.0,
            max_radius: 32.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bilinear sampler (private)
// ─────────────────────────────────────────────────────────────────────────────

/// Bilinear sample of a row-major RGB (`width × height × 3`) image.
///
/// Coordinates outside `[0, width/height)` are clamped to the edge.
///
/// Returns `[r, g, b]`.
fn bilinear_sample(image: &[f32], width: usize, height: usize, sx: f32, sy: f32) -> [f32; 3] {
    // Clamp continuous coordinates.
    let sx = sx.clamp(0.0, width as f32 - 1.0 - f32::EPSILON);
    let sy = sy.clamp(0.0, height as f32 - 1.0 - f32::EPSILON);

    let x0 = sx.floor() as usize;
    let y0 = sy.floor() as usize;

    // Neighbours — safe because we clamped `sx/sy` above so `+1` stays in bounds.
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let tx = sx - x0 as f32;
    let ty = sy - y0 as f32;

    let inv_tx = 1.0 - tx;
    let inv_ty = 1.0 - ty;

    // Weights for the four corners.
    let w00 = inv_tx * inv_ty;
    let w10 = tx * inv_ty;
    let w01 = inv_tx * ty;
    let w11 = tx * ty;

    let idx00 = (y0 * width + x0) * 3;
    let idx10 = (y0 * width + x1) * 3;
    let idx01 = (y1 * width + x0) * 3;
    let idx11 = (y1 * width + x1) * 3;

    let mut out = [0.0_f32; 3];
    for c in 0..3 {
        out[c] = w00 * image[idx00 + c]
            + w10 * image[idx10 + c]
            + w01 * image[idx01 + c]
            + w11 * image[idx11 + c];
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// apply_velocity_blur
// ─────────────────────────────────────────────────────────────────────────────

/// Apply image-space velocity-based motion blur to an RGB image.
///
/// For each pixel `(px, py)`:
/// 1. Retrieve the screen-space velocity `(vx, vy)` and scale by
///    `config.velocity_scale`.
/// 2. Clamp the vector magnitude to `config.max_radius`.
/// 3. Uniformly sample `config.num_samples` points along the vector from the
///    pixel centre to `(px + vx, py + vy)`.
/// 4. Bilinear-sample the source image at each point (clamp-to-edge).
/// 5. Average the samples into the output pixel.
///
/// # Parameters
///
/// - `image`: row-major RGB f32 buffer; `len = width * height * 3`.
/// - `velocity`: per-pixel velocity field; must match `width` and `height`.
///
/// # Errors
///
/// - [`MotionBlurError::InvalidImage`] if `image.len() != width * height * 3`.
/// - [`MotionBlurError::DimensionMismatch`] if the velocity field dimensions
///   do not match `width` / `height`.
/// - [`MotionBlurError::InvalidConfig`] if `num_samples == 0`.
pub fn apply_velocity_blur(
    image: &[f32],
    width: usize,
    height: usize,
    velocity: &VelocityField,
    config: &VelocityBlurConfig,
) -> Result<Vec<f32>, MotionBlurError> {
    let expected = width * height * 3;
    if image.len() != expected {
        return Err(MotionBlurError::InvalidImage(format!(
            "expected {} elements, got {}",
            expected,
            image.len()
        )));
    }
    if velocity.width != width || velocity.height != height {
        return Err(MotionBlurError::DimensionMismatch {
            expected: width * height,
            got: velocity.width * velocity.height,
        });
    }
    if config.num_samples == 0 {
        return Err(MotionBlurError::InvalidConfig(
            "num_samples must be at least 1".to_string(),
        ));
    }

    let n = config.num_samples;
    let inv_n = 1.0 / n as f32;
    let mut output = vec![0.0_f32; expected];

    for py in 0..height {
        for px in 0..width {
            let pixel_idx = py * width + px;
            let vel_base = pixel_idx * 2;
            let mut vx = velocity.velocities[vel_base] * config.velocity_scale;
            let mut vy = velocity.velocities[vel_base + 1] * config.velocity_scale;

            // Clamp velocity magnitude to max_radius.
            let mag = (vx * vx + vy * vy).sqrt();
            if mag > config.max_radius {
                let scale = config.max_radius / mag;
                vx *= scale;
                vy *= scale;
            }

            let px_f = px as f32;
            let py_f = py as f32;

            let mut acc = [0.0_f32; 3];
            for s in 0..n {
                // t ∈ [0, 1] across the n samples.
                let t = if n == 1 {
                    0.0
                } else {
                    s as f32 / (n - 1) as f32
                };
                let sx = px_f + vx * t;
                let sy = py_f + vy * t;
                let sample = bilinear_sample(image, width, height, sx, sy);
                acc[0] += sample[0];
                acc[1] += sample[1];
                acc[2] += sample[2];
            }

            let out_base = pixel_idx * 3;
            output[out_base] = acc[0] * inv_n;
            output[out_base + 1] = acc[1] * inv_n;
            output[out_base + 2] = acc[2] * inv_n;
        }
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// AccumulatedBlur
// ─────────────────────────────────────────────────────────────────────────────

/// Result of accumulation-based motion blur.
#[derive(Debug, Clone)]
pub struct AccumulatedBlur {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Blurred RGB image, row-major f32; `len = width * height * 3`.
    pub image: Vec<f32>,
    /// Number of sub-frames that were accumulated.
    pub num_samples: usize,
}

/// Simulate motion blur by accumulating pre-rendered sub-frame images.
///
/// Each element of `frames` is a row-major RGB f32 buffer of length
/// `width * height * 3`.  The frames are blended using the weights from
/// [`AccumulationConfig::sample_weights`].
///
/// # Errors
///
/// - [`MotionBlurError::EmptyFrames`] if `frames` is empty.
/// - [`MotionBlurError::DimensionMismatch`] if any frame has wrong length.
pub fn accumulate_frames(
    frames: &[Vec<f32>],
    width: usize,
    height: usize,
    config: &AccumulationConfig,
) -> Result<AccumulatedBlur, MotionBlurError> {
    if frames.is_empty() {
        return Err(MotionBlurError::EmptyFrames);
    }

    let expected = width * height * 3;
    for (fi, frame) in frames.iter().enumerate() {
        if frame.len() != expected {
            return Err(MotionBlurError::DimensionMismatch {
                expected,
                got: frame.len(),
            });
        }
        let _ = fi; // suppress unused-variable lint in older rustc versions
    }

    config.validate()?;

    let weights = config.sample_weights();
    // weights may have length == config.num_samples, but we only have
    // frames.len() actual frames; use the minimum.
    let n = frames.len().min(weights.len());

    let mut output = vec![0.0_f32; expected];
    let mut weight_sum = 0.0_f32;

    for i in 0..n {
        let w = weights[i];
        let frame = &frames[i];
        for (out, &src) in output.iter_mut().zip(frame.iter()) {
            *out += w * src;
        }
        weight_sum += w;
    }

    // Renormalise if we didn't use all weights (frames.len() < num_samples).
    if weight_sum > 0.0 && (weight_sum - 1.0).abs() > 1e-6 {
        let inv = 1.0 / weight_sum;
        for v in &mut output {
            *v *= inv;
        }
    }

    Ok(AccumulatedBlur {
        width,
        height,
        image: output,
        num_samples: n,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// MotionBlurStats
// ─────────────────────────────────────────────────────────────────────────────

/// Diagnostic statistics for a motion-blur operation.
#[derive(Debug, Clone)]
pub struct MotionBlurStats {
    /// Mean velocity magnitude across all pixels (pixels/frame).
    pub mean_blur_magnitude: f32,
    /// Maximum velocity magnitude across all pixels.
    pub max_blur_magnitude: f32,
    /// Fraction of pixels with velocity magnitude > 0.5 px.
    pub blurred_pixel_fraction: f32,
    /// Effective weighted sample count:
    /// `velocity_scale * num_samples` (saturated to `max_radius / mean_mag`
    /// when mean is non-zero and would exceed the cap).
    pub effective_samples: f32,
}

/// Compute motion-blur statistics from a velocity field and blur configuration.
pub fn compute_motion_stats(
    velocity: &VelocityField,
    config: &VelocityBlurConfig,
) -> MotionBlurStats {
    let n_pixels = velocity.width * velocity.height;

    if n_pixels == 0 {
        return MotionBlurStats {
            mean_blur_magnitude: 0.0,
            max_blur_magnitude: 0.0,
            blurred_pixel_fraction: 0.0,
            effective_samples: config.num_samples as f32,
        };
    }

    let mut sum_mag = 0.0_f32;
    let mut max_mag = 0.0_f32;
    let mut blurred_count = 0usize;

    for chunk in velocity.velocities.chunks_exact(2) {
        let vx = chunk[0] * config.velocity_scale;
        let vy = chunk[1] * config.velocity_scale;
        let mag = (vx * vx + vy * vy).sqrt().min(config.max_radius);
        sum_mag += mag;
        if mag > max_mag {
            max_mag = mag;
        }
        if mag > 0.5 {
            blurred_count += 1;
        }
    }

    let mean_mag = sum_mag / n_pixels as f32;
    let blurred_pixel_fraction = blurred_count as f32 / n_pixels as f32;

    // Effective samples: nominal samples times the fraction of max_radius
    // actually used.  When mean is zero all samples collapse to the same
    // point, so effective count is 1.
    let effective_samples = if mean_mag < 1e-6 {
        1.0
    } else {
        let utilisation = (mean_mag / config.max_radius).min(1.0);
        1.0 + (config.num_samples as f32 - 1.0) * utilisation
    };

    MotionBlurStats {
        mean_blur_magnitude: mean_mag,
        max_blur_magnitude: max_mag,
        blurred_pixel_fraction,
        effective_samples,
    }
}

// (The new velocity-based MB pipeline lives in the sibling crate module
// `crate::mb_pipeline`, declared and re-exported from lib.rs.)

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerance for floating-point comparisons.
    const EPS: f32 = 1e-5;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // ── subframe_t ────────────────────────────────────────────────────────────

    #[test]
    fn test_subframe_t_n1_i0() {
        // n == 1: only index 0 exists; should return 0.0
        assert!(approx(subframe_t(0, 1), 0.0), "subframe_t(0,1) should be 0");
    }

    #[test]
    fn test_subframe_t_n4_i0() {
        // First sample in n=4 → 0.0
        assert!(approx(subframe_t(0, 4), 0.0));
    }

    #[test]
    fn test_subframe_t_n4_i3() {
        // Last sample in n=4 → 1.0
        assert!(approx(subframe_t(3, 4), 1.0));
    }

    #[test]
    fn test_subframe_t_midpoint() {
        // i=1, n=3 → t = 0.5
        assert!(approx(subframe_t(1, 3), 0.5));
    }

    // ── lerp_position ─────────────────────────────────────────────────────────

    #[test]
    fn test_lerp_position_t0() {
        let start = [1.0, 2.0, 3.0];
        let end = [4.0, 5.0, 6.0];
        let r = lerp_position(start, end, 0.0);
        assert!(approx(r[0], 1.0) && approx(r[1], 2.0) && approx(r[2], 3.0));
    }

    #[test]
    fn test_lerp_position_t1() {
        let start = [1.0, 2.0, 3.0];
        let end = [4.0, 5.0, 6.0];
        let r = lerp_position(start, end, 1.0);
        assert!(approx(r[0], 4.0) && approx(r[1], 5.0) && approx(r[2], 6.0));
    }

    #[test]
    fn test_lerp_position_midpoint() {
        let start = [0.0, 0.0, 0.0];
        let end = [2.0, 4.0, 6.0];
        let r = lerp_position(start, end, 0.5);
        assert!(approx(r[0], 1.0) && approx(r[1], 2.0) && approx(r[2], 3.0));
    }

    // ── lerp_quaternion + normalize_quaternion ─────────────────────────────────

    #[test]
    fn test_lerp_quaternion_t0() {
        let q0 = [0.0, 0.0, 0.0, 1.0];
        let q1 = [1.0, 0.0, 0.0, 0.0];
        let r = lerp_quaternion(q0, q1, 0.0);
        for (a, b) in r.iter().zip(q0.iter()) {
            assert!(approx(*a, *b));
        }
    }

    #[test]
    fn test_lerp_quaternion_t1() {
        let q0 = [0.0, 0.0, 0.0, 1.0];
        let q1 = [1.0, 0.0, 0.0, 0.0];
        let r = lerp_quaternion(q0, q1, 1.0);
        for (a, b) in r.iter().zip(q1.iter()) {
            assert!(approx(*a, *b));
        }
    }

    #[test]
    fn test_normalize_quaternion_identity() {
        // [0,0,0,1] already unit length; should be unchanged.
        let q = [0.0_f32, 0.0, 0.0, 1.0];
        let r = normalize_quaternion(q);
        assert!(approx(r[3], 1.0));
    }

    #[test]
    fn test_normalize_quaternion_zero_returns_identity() {
        let q = [0.0_f32, 0.0, 0.0, 0.0];
        let r = normalize_quaternion(q);
        assert!(approx(r[0], 0.0) && approx(r[1], 0.0) && approx(r[2], 0.0) && approx(r[3], 1.0));
    }

    #[test]
    fn test_normalize_quaternion_unit_length() {
        let q = lerp_quaternion([0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], 0.5);
        let n = normalize_quaternion(q);
        let mag_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2] + n[3] * n[3];
        assert!((mag_sq - 1.0).abs() < 1e-5, "mag_sq = {}", mag_sq);
    }

    // ── CameraMotion ──────────────────────────────────────────────────────────

    #[test]
    fn test_camera_motion_stationary_is_stationary() {
        let cm = CameraMotion::stationary([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]);
        assert!(
            cm.is_stationary(),
            "Stationary camera must be is_stationary()"
        );
    }

    #[test]
    fn test_camera_motion_moving_not_stationary() {
        let cm = CameraMotion::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        );
        assert!(!cm.is_stationary(), "Moving camera must NOT be stationary");
    }

    #[test]
    fn test_camera_motion_position_at_t0_and_t1() {
        let cm = CameraMotion::new(
            [0.0, 0.0, 0.0],
            [3.0, 6.0, 9.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        );
        let p0 = cm.position_at(0.0);
        let p1 = cm.position_at(1.0);
        assert!(approx(p0[0], 0.0) && approx(p0[1], 0.0) && approx(p0[2], 0.0));
        assert!(approx(p1[0], 3.0) && approx(p1[1], 6.0) && approx(p1[2], 9.0));
    }

    #[test]
    fn test_camera_motion_position_at_midpoint() {
        let cm = CameraMotion::new(
            [0.0, 0.0, 0.0],
            [2.0, 4.0, 6.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        );
        let p = cm.position_at(0.5);
        assert!(approx(p[0], 1.0) && approx(p[1], 2.0) && approx(p[2], 3.0));
    }

    #[test]
    fn test_camera_motion_rotation_at_endpoints() {
        let r0 = [0.0_f32, 0.0, 0.0, 1.0];
        let r1 = [
            0.0_f32,
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        ];
        let cm = CameraMotion::new([0.0; 3], [0.0; 3], r0, r1);
        let at0 = cm.rotation_at(0.0);
        let at1 = cm.rotation_at(1.0);
        // Should match r0 / r1 after normalisation (r0 already unit).
        let mag0_sq = at0.iter().map(|v| v * v).sum::<f32>();
        let mag1_sq = at1.iter().map(|v| v * v).sum::<f32>();
        assert!((mag0_sq - 1.0).abs() < 1e-5, "rotation_at(0) not unit");
        assert!((mag1_sq - 1.0).abs() < 1e-5, "rotation_at(1) not unit");
    }

    #[test]
    fn test_camera_motion_translation_distance() {
        let cm = CameraMotion::new(
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        );
        let dist = cm.translation_distance();
        assert!(
            approx(dist, 5.0),
            "3-4-5 triangle: expected 5.0, got {}",
            dist
        );
    }

    // ── AccumulationConfig ────────────────────────────────────────────────────

    #[test]
    fn test_accumulation_config_validate_valid() {
        let cfg = AccumulationConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_accumulation_config_validate_zero_samples() {
        let cfg = AccumulationConfig {
            num_samples: 0,
            ..AccumulationConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(MotionBlurError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_accumulation_config_validate_bad_shutter() {
        let cfg = AccumulationConfig {
            shutter_open: 0.0,
            ..AccumulationConfig::default()
        };
        assert!(matches!(
            cfg.validate(),
            Err(MotionBlurError::InvalidConfig(_))
        ));

        let cfg2 = AccumulationConfig {
            shutter_open: 1.5,
            ..AccumulationConfig::default()
        };
        assert!(matches!(
            cfg2.validate(),
            Err(MotionBlurError::InvalidConfig(_))
        ));
    }

    #[test]
    fn test_sample_weights_uniform_sum_to_one() {
        let cfg = AccumulationConfig {
            num_samples: 8,
            uniform_weights: true,
            ..AccumulationConfig::default()
        };
        let weights = cfg.sample_weights();
        assert_eq!(weights.len(), 8);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "uniform weights sum = {}", sum);
    }

    #[test]
    fn test_sample_weights_ramp_sum_to_one() {
        let cfg = AccumulationConfig {
            num_samples: 9,
            uniform_weights: false,
            ..AccumulationConfig::default()
        };
        let weights = cfg.sample_weights();
        assert_eq!(weights.len(), 9);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "ramp weights sum = {}", sum);
    }

    #[test]
    fn test_sample_weights_ramp_peaks_at_centre() {
        let cfg = AccumulationConfig {
            num_samples: 5,
            uniform_weights: false,
            ..AccumulationConfig::default()
        };
        let weights = cfg.sample_weights();
        // Centre is index 2 for n=5.
        let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            approx(weights[2], max_w),
            "Centre weight should be maximum; weights = {:?}",
            weights
        );
    }

    // ── VelocityField ─────────────────────────────────────────────────────────

    #[test]
    fn test_velocity_field_new_all_zero() {
        let vf = VelocityField::new(4, 3);
        assert_eq!(vf.velocities.len(), 4 * 3 * 2);
        assert!(vf.velocities.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_velocity_field_get_set_roundtrip() {
        let mut vf = VelocityField::new(5, 5);
        vf.set(2, 3, 1.5, -2.5).expect("set should succeed");
        let [vx, vy] = vf.get(2, 3).expect("get should succeed");
        assert!(approx(vx, 1.5) && approx(vy, -2.5));
    }

    #[test]
    fn test_velocity_field_out_of_bounds_error() {
        let vf = VelocityField::new(4, 4);
        assert!(vf.get(4, 0).is_err());
        assert!(vf.get(0, 4).is_err());
    }

    #[test]
    fn test_velocity_field_max_magnitude() {
        let mut vf = VelocityField::new(3, 3);
        vf.set(1, 1, 3.0, 4.0).expect("set ok"); // magnitude 5
        assert!(approx(vf.max_magnitude(), 5.0));
    }

    #[test]
    fn test_velocity_field_mean_magnitude_zero() {
        let vf = VelocityField::new(4, 4);
        assert!(approx(vf.mean_magnitude(), 0.0));
    }

    #[test]
    fn test_velocity_field_mean_magnitude_nonzero() {
        let mut vf = VelocityField::new(2, 1);
        // pixel 0: (3,4) → mag 5; pixel 1: (0,0) → mag 0; mean = 2.5
        vf.set(0, 0, 3.0, 4.0).expect("set ok");
        let mean = vf.mean_magnitude();
        assert!(approx(mean, 2.5), "mean_magnitude = {}", mean);
    }

    // ── apply_velocity_blur ───────────────────────────────────────────────────

    #[test]
    fn test_apply_velocity_blur_zero_velocity_identity() {
        // Zero velocity → samples only at the pixel itself → output == input.
        let w = 4_usize;
        let h = 4_usize;
        let image: Vec<f32> = (0..w * h * 3).map(|i| i as f32 / 100.0).collect();
        let velocity = VelocityField::new(w, h); // all zero
        let config = VelocityBlurConfig {
            num_samples: 8,
            ..VelocityBlurConfig::default()
        };
        let output = apply_velocity_blur(&image, w, h, &velocity, &config)
            .expect("apply_velocity_blur failed");
        assert_eq!(output.len(), image.len());
        for (i, (&a, &b)) in image.iter().zip(output.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "pixel {} channel: expected {}, got {}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn test_apply_velocity_blur_nonzero_produces_change() {
        // Non-uniform image with horizontal velocity → output differs from input.
        let w = 8_usize;
        let h = 4_usize;
        let image: Vec<f32> = (0..w * h * 3)
            .map(|i| if (i / 3) % 2 == 0 { 1.0 } else { 0.0 })
            .collect();
        let mut velocity = VelocityField::new(w, h);
        // Set a 4-pixel horizontal velocity for all pixels.
        for y in 0..h {
            for x in 0..w {
                velocity.set(x, y, 4.0, 0.0).expect("set ok");
            }
        }
        let config = VelocityBlurConfig {
            num_samples: 4,
            velocity_scale: 1.0,
            max_radius: 32.0,
        };
        let output = apply_velocity_blur(&image, w, h, &velocity, &config)
            .expect("apply_velocity_blur failed");
        // At least some pixels should differ from the input.
        let changed = image
            .iter()
            .zip(output.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-4);
        assert!(
            changed,
            "Non-zero velocity should produce a different image"
        );
    }

    // ── accumulate_frames ─────────────────────────────────────────────────────

    #[test]
    fn test_accumulate_frames_single_frame_identity() {
        let w = 4_usize;
        let h = 3_usize;
        let image: Vec<f32> = (0..w * h * 3).map(|i| i as f32 / 50.0).collect();
        let frames = vec![image.clone()];
        let config = AccumulationConfig {
            num_samples: 1,
            ..AccumulationConfig::default()
        };
        let result = accumulate_frames(&frames, w, h, &config).expect("accumulate_frames failed");
        assert_eq!(result.image.len(), image.len());
        for (a, b) in image.iter().zip(result.image.iter()) {
            assert!((a - b).abs() < 1e-5, "single frame: got {b}, expected {a}");
        }
        assert_eq!(result.num_samples, 1);
    }

    #[test]
    fn test_accumulate_frames_two_frames_uniform_average() {
        let w = 2_usize;
        let h = 2_usize;
        let frame0 = vec![0.0_f32; w * h * 3];
        let frame1 = vec![1.0_f32; w * h * 3];
        let frames = vec![frame0, frame1];
        let config = AccumulationConfig {
            num_samples: 2,
            uniform_weights: true,
            ..AccumulationConfig::default()
        };
        let result = accumulate_frames(&frames, w, h, &config).expect("accumulate_frames failed");
        for &v in &result.image {
            assert!(
                approx(v, 0.5),
                "uniform average of 0 and 1 should be 0.5, got {v}"
            );
        }
    }

    #[test]
    fn test_accumulate_frames_empty_returns_error() {
        let frames: Vec<Vec<f32>> = vec![];
        let config = AccumulationConfig::default();
        let result = accumulate_frames(&frames, 4, 4, &config);
        assert!(matches!(result, Err(MotionBlurError::EmptyFrames)));
    }

    #[test]
    fn test_accumulate_frames_dimension_mismatch_error() {
        let w = 4_usize;
        let h = 4_usize;
        // Frame with wrong size.
        let frames = vec![vec![0.5_f32; w * h * 3 + 1]];
        let config = AccumulationConfig {
            num_samples: 1,
            ..AccumulationConfig::default()
        };
        let result = accumulate_frames(&frames, w, h, &config);
        assert!(matches!(
            result,
            Err(MotionBlurError::DimensionMismatch { .. })
        ));
    }

    // ── compute_motion_stats ──────────────────────────────────────────────────

    #[test]
    fn test_compute_motion_stats_zero_velocity() {
        let vf = VelocityField::new(4, 4);
        let config = VelocityBlurConfig::default();
        let stats = compute_motion_stats(&vf, &config);
        assert!(approx(stats.mean_blur_magnitude, 0.0));
        assert!(approx(stats.max_blur_magnitude, 0.0));
        assert!(approx(stats.blurred_pixel_fraction, 0.0));
        assert!(approx(stats.effective_samples, 1.0));
    }
}
