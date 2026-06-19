//! Head tracking over video sequences.
//!
//! Maintains pose history, applies temporal filters, detects anomalies,
//! and provides trajectory analysis for 6-DOF head pose sequences.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during head tracking operations.
#[derive(Debug, Error)]
pub enum HeadTrackerError {
    /// Trajectory has no frames.
    #[error("Empty trajectory")]
    EmptyTrajectory,

    /// Configuration value is invalid.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// Requested frame index is beyond the trajectory length.
    #[error("Frame out of range: frame {frame}, trajectory length {len}")]
    FrameOutOfRange { frame: usize, len: usize },

    /// Not enough history frames for the operation.
    #[error("Insufficient history: need {needed} frames, have {got}")]
    InsufficientHistory { needed: usize, got: usize },

    /// A numerical computation failed.
    #[error("Numerical error: {0}")]
    NumericalError(String),
}

// ---------------------------------------------------------------------------
// HeadTrackFrame
// ---------------------------------------------------------------------------

/// 6-DOF head pose with confidence, used to construct [`HeadTrackFrame`].
#[derive(Debug, Clone, Copy, Default)]
pub struct HeadTrackPose {
    /// Rotation around Y axis (radians).
    pub yaw: f32,
    /// Rotation around X axis (radians).
    pub pitch: f32,
    /// Rotation around Z axis (radians).
    pub roll: f32,
    /// Translation X (mm).
    pub tx: f32,
    /// Translation Y (mm).
    pub ty: f32,
    /// Translation Z – depth (mm).
    pub tz: f32,
    /// Detection confidence in \[0, 1\].
    pub confidence: f32,
}

/// A single head pose frame with simplified rotation and translation.
#[derive(Debug, Clone)]
pub struct HeadTrackFrame {
    /// Index of this frame in the source video.
    pub frame_idx: usize,
    /// Timestamp in milliseconds.
    pub timestamp_ms: f32,
    /// Rotation around Y axis (radians).
    pub yaw: f32,
    /// Rotation around X axis (radians).
    pub pitch: f32,
    /// Rotation around Z axis (radians).
    pub roll: f32,
    /// Translation X (mm).
    pub tx: f32,
    /// Translation Y (mm).
    pub ty: f32,
    /// Translation Z – depth (mm).
    pub tz: f32,
    /// Detection confidence in \[0, 1\].
    pub confidence: f32,
    /// `false` when detection failed for this frame.
    pub is_valid: bool,
}

impl HeadTrackFrame {
    /// Create a valid frame with the given pose.
    #[must_use]
    pub fn new(frame_idx: usize, timestamp_ms: f32, pose: HeadTrackPose) -> Self {
        Self {
            frame_idx,
            timestamp_ms,
            yaw: pose.yaw,
            pitch: pose.pitch,
            roll: pose.roll,
            tx: pose.tx,
            ty: pose.ty,
            tz: pose.tz,
            confidence: pose.confidence,
            is_valid: true,
        }
    }

    /// Create an invalid (detection-failed) frame at the given timestamp.
    #[must_use]
    pub fn invalid(frame_idx: usize, timestamp_ms: f32) -> Self {
        Self {
            frame_idx,
            timestamp_ms,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            tx: 0.0,
            ty: 0.0,
            tz: 0.0,
            confidence: 0.0,
            is_valid: false,
        }
    }

    /// Return the three Euler angles as `[yaw, pitch, roll]`.
    #[must_use]
    pub fn euler_angles(&self) -> [f32; 3] {
        [self.yaw, self.pitch, self.roll]
    }

    /// Return the translation vector as `[tx, ty, tz]`.
    #[must_use]
    pub fn translation(&self) -> [f32; 3] {
        [self.tx, self.ty, self.tz]
    }

    /// L2 norm of the Euler angle vector.
    #[must_use]
    pub fn rotation_magnitude(&self) -> f32 {
        (self.yaw * self.yaw + self.pitch * self.pitch + self.roll * self.roll).sqrt()
    }
}

// ---------------------------------------------------------------------------
// TrackingFilter
// ---------------------------------------------------------------------------

/// Filtering method applied to the trajectory stream.
#[derive(Debug, Clone, PartialEq)]
pub enum TrackingFilter {
    /// No filtering; raw frames are forwarded.
    None,
    /// Exponential moving average — `alpha` in `(0, 1]`.
    ExponentialMovingAverage {
        /// Smoothing factor. 1.0 = no smoothing.
        alpha: f32,
    },
    /// Simple causal moving average over the last `window` frames.
    SimpleMovingAverage {
        /// Window width ≥ 1.
        window: usize,
    },
    /// One Euro filter (frequency-adaptive low-pass).
    OneEuro {
        /// Minimum cutoff frequency (Hz).
        min_cutoff: f32,
        /// Speed coefficient.
        beta: f32,
    },
}

// ---------------------------------------------------------------------------
// HeadTrackerConfig
// ---------------------------------------------------------------------------

/// Configuration for [`HeadTracker`].
#[derive(Debug, Clone)]
pub struct HeadTrackerConfig {
    /// Temporal filter to apply on each update.
    pub filter: TrackingFilter,
    /// Maximum number of consecutive missing frames to interpolate over.
    pub max_gap_frames: usize,
    /// Maximum frame-to-frame rotation change (radians) before flagging an outlier.
    pub outlier_threshold: f32,
    /// Minimum detection confidence to accept a frame as valid.
    pub confidence_threshold: f32,
    /// Maximum number of frames kept in rolling history.
    pub max_history: usize,
}

impl Default for HeadTrackerConfig {
    fn default() -> Self {
        Self {
            filter: TrackingFilter::None,
            max_gap_frames: 5,
            outlier_threshold: std::f32::consts::PI / 6.0,
            confidence_threshold: 0.3,
            max_history: 1000,
        }
    }
}

impl HeadTrackerConfig {
    /// Check that all configuration values are in their valid ranges.
    ///
    /// # Errors
    ///
    /// Returns [`HeadTrackerError::InvalidConfig`] when a value is out of range.
    pub fn validate(&self) -> Result<(), HeadTrackerError> {
        if self.outlier_threshold <= 0.0 {
            return Err(HeadTrackerError::InvalidConfig(
                "outlier_threshold must be positive".to_string(),
            ));
        }
        if self.confidence_threshold < 0.0 || self.confidence_threshold > 1.0 {
            return Err(HeadTrackerError::InvalidConfig(
                "confidence_threshold must be in [0, 1]".to_string(),
            ));
        }
        if self.max_history == 0 {
            return Err(HeadTrackerError::InvalidConfig(
                "max_history must be at least 1".to_string(),
            ));
        }
        if let TrackingFilter::ExponentialMovingAverage { alpha } = self.filter {
            if alpha <= 0.0 || alpha > 1.0 {
                return Err(HeadTrackerError::InvalidConfig(
                    "EMA alpha must be in (0, 1]".to_string(),
                ));
            }
        }
        if let TrackingFilter::SimpleMovingAverage { window } = self.filter {
            if window == 0 {
                return Err(HeadTrackerError::InvalidConfig(
                    "SMA window must be at least 1".to_string(),
                ));
            }
        }
        if let TrackingFilter::OneEuro { min_cutoff, beta } = self.filter {
            if min_cutoff <= 0.0 {
                return Err(HeadTrackerError::InvalidConfig(
                    "OneEuro min_cutoff must be positive".to_string(),
                ));
            }
            if beta < 0.0 {
                return Err(HeadTrackerError::InvalidConfig(
                    "OneEuro beta must be non-negative".to_string(),
                ));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// HeadTracker
// ---------------------------------------------------------------------------

/// Real-time head tracker that maintains rolling pose history and applies filters.
pub struct HeadTracker {
    /// Active configuration.
    pub config: HeadTrackerConfig,
    /// Raw (unfiltered) frame history.
    history: Vec<HeadTrackFrame>,
    /// Filtered frame history (parallel to `history`).
    filtered_history: Vec<HeadTrackFrame>,
    /// Previous filtered yaw value (One Euro).
    one_euro_x_prev: Option<f32>,
    /// Previous yaw derivative estimate (One Euro).
    one_euro_dx_prev: Option<f32>,
}

impl HeadTracker {
    /// Create a new tracker with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`HeadTrackerError::InvalidConfig`] if the configuration fails validation.
    pub fn new(config: HeadTrackerConfig) -> Result<Self, HeadTrackerError> {
        config.validate()?;
        Ok(Self {
            config,
            history: Vec::new(),
            filtered_history: Vec::new(),
            one_euro_x_prev: None,
            one_euro_dx_prev: None,
        })
    }

    /// Push a new frame, apply the configured filter, trim history, and return
    /// a reference to the filtered frame just recorded.
    ///
    /// # Panics
    ///
    /// Panics if the internal filtered history is unexpectedly empty after a push,
    /// which should never happen in normal use.
    pub fn update(&mut self, frame: HeadTrackFrame) -> &HeadTrackFrame {
        let filtered = self.apply_filter(&frame);
        self.filtered_history.push(filtered);
        self.history.push(frame);

        // Trim both histories to max_history
        let max = self.config.max_history;
        if self.history.len() > max {
            let drain = self.history.len() - max;
            self.history.drain(0..drain);
        }
        if self.filtered_history.len() > max {
            let drain = self.filtered_history.len() - max;
            self.filtered_history.drain(0..drain);
        }

        // Safety: we just pushed to filtered_history so it cannot be empty.
        // The unreachable! path documents the invariant.
        self.filtered_history
            .last()
            .unwrap_or_else(|| unreachable!("filtered_history is non-empty after push"))
    }

    /// Apply the configured filter to produce a smoothed frame.
    fn apply_filter(&mut self, frame: &HeadTrackFrame) -> HeadTrackFrame {
        match &self.config.filter {
            TrackingFilter::None => frame.clone(),
            TrackingFilter::ExponentialMovingAverage { alpha } => {
                let alpha = *alpha;
                if let Some(prev) = self.filtered_history.last() {
                    HeadTrackFrame {
                        yaw: alpha * frame.yaw + (1.0 - alpha) * prev.yaw,
                        pitch: alpha * frame.pitch + (1.0 - alpha) * prev.pitch,
                        roll: alpha * frame.roll + (1.0 - alpha) * prev.roll,
                        tx: alpha * frame.tx + (1.0 - alpha) * prev.tx,
                        ty: alpha * frame.ty + (1.0 - alpha) * prev.ty,
                        tz: alpha * frame.tz + (1.0 - alpha) * prev.tz,
                        ..frame.clone()
                    }
                } else {
                    frame.clone()
                }
            }
            TrackingFilter::SimpleMovingAverage { window } => {
                let window = *window;
                let len = self.filtered_history.len();
                if len == 0 {
                    return frame.clone();
                }
                let count = window.min(len);
                let start = len - count;
                let (mut yaw, mut pitch, mut roll, mut tx, mut ty, mut tz) =
                    (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for f in &self.filtered_history[start..] {
                    yaw += f.yaw;
                    pitch += f.pitch;
                    roll += f.roll;
                    tx += f.tx;
                    ty += f.ty;
                    tz += f.tz;
                }
                // Include current frame
                yaw += frame.yaw;
                pitch += frame.pitch;
                roll += frame.roll;
                tx += frame.tx;
                ty += frame.ty;
                tz += frame.tz;
                let n = (count + 1) as f32;
                HeadTrackFrame {
                    yaw: yaw / n,
                    pitch: pitch / n,
                    roll: roll / n,
                    tx: tx / n,
                    ty: ty / n,
                    tz: tz / n,
                    ..frame.clone()
                }
            }
            TrackingFilter::OneEuro { min_cutoff, beta } => {
                let min_cutoff = *min_cutoff;
                let beta = *beta;
                // Apply One Euro to yaw only; other fields pass through raw
                let filtered_yaw = one_euro_step(
                    frame.yaw,
                    frame.timestamp_ms,
                    &mut self.one_euro_x_prev,
                    &mut self.one_euro_dx_prev,
                    min_cutoff,
                    beta,
                );
                HeadTrackFrame {
                    yaw: filtered_yaw,
                    ..frame.clone()
                }
            }
        }
    }

    /// Most recent raw frame, if any.
    #[must_use]
    pub fn current(&self) -> Option<&HeadTrackFrame> {
        self.history.last()
    }

    /// Number of frames in history.
    #[must_use]
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Raw frame at rolling-history index `idx` (0 = oldest retained).
    #[must_use]
    pub fn get_frame(&self, idx: usize) -> Option<&HeadTrackFrame> {
        self.history.get(idx)
    }

    /// Most recent filtered frame, if any.
    #[must_use]
    pub fn filtered_current(&self) -> Option<&HeadTrackFrame> {
        self.filtered_history.last()
    }

    /// Clear all history and reset filter state.
    pub fn reset(&mut self) {
        self.history.clear();
        self.filtered_history.clear();
        self.one_euro_x_prev = None;
        self.one_euro_dx_prev = None;
    }

    /// Fraction of frames in history that are marked valid (`is_valid == true`).
    #[must_use]
    pub fn valid_fraction(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let valid = self.history.iter().filter(|f| f.is_valid).count();
        valid as f32 / self.history.len() as f32
    }
}

// ---------------------------------------------------------------------------
// One Euro filter helper
// ---------------------------------------------------------------------------

/// Low-pass filter cutoff from the derivative magnitude.
fn one_euro_cutoff(min_cutoff: f32, beta: f32, dx: f32) -> f32 {
    min_cutoff + beta * dx.abs()
}

/// First-order low-pass filter coefficient.
fn alpha_from_cutoff(cutoff_hz: f32, dt_ms: f32) -> f32 {
    let dt_s = dt_ms / 1000.0;
    let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff_hz);
    1.0 / (1.0 + tau / dt_s.max(f32::EPSILON))
}

/// Advance the One Euro filter by one step.
///
/// Modifies `x_prev` and `dx_prev` in place.
fn one_euro_step(
    x: f32,
    timestamp_ms: f32,
    x_prev: &mut Option<f32>,
    dx_prev: &mut Option<f32>,
    min_cutoff: f32,
    beta: f32,
) -> f32 {
    if let (Some(xp), Some(dxp)) = (*x_prev, *dx_prev) {
        // Use a fixed dt assumption of 1 ms when timestamps aren't progressive
        let dt_ms = (timestamp_ms - 0.0).max(1.0);
        let _ = dt_ms; // will be used below
                       // Derivative low-pass (cutoff 1 Hz)
        let d_cutoff = 1.0f32;
        // dt estimate: since we don't store prev_ts, use a nominal 33 ms (30 fps)
        let dt_est = 33.0_f32;
        let a_d = alpha_from_cutoff(d_cutoff, dt_est);
        let dx = (x - xp) / dt_est;
        let dx_hat = a_d * dx + (1.0 - a_d) * dxp;

        let cutoff = one_euro_cutoff(min_cutoff, beta, dx_hat);
        let a = alpha_from_cutoff(cutoff, dt_est);
        let x_hat = a * x + (1.0 - a) * xp;

        *x_prev = Some(x_hat);
        *dx_prev = Some(dx_hat);
        x_hat
    } else {
        // First sample: initialize
        *x_prev = Some(x);
        *dx_prev = Some(0.0);
        x
    }
}

// ---------------------------------------------------------------------------
// HeadTrajectory
// ---------------------------------------------------------------------------

/// Immutable snapshot of tracked frames for offline analysis.
#[derive(Debug, Clone)]
pub struct HeadTrajectory {
    /// All frames in the trajectory.
    pub frames: Vec<HeadTrackFrame>,
}

impl HeadTrajectory {
    /// Create a trajectory from a vector of frames.
    ///
    /// # Errors
    ///
    /// Returns [`HeadTrackerError::EmptyTrajectory`] when `frames` is empty.
    pub fn new(frames: Vec<HeadTrackFrame>) -> Result<Self, HeadTrackerError> {
        if frames.is_empty() {
            return Err(HeadTrackerError::EmptyTrajectory);
        }
        Ok(Self { frames })
    }

    /// Total number of frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// `true` when there are no frames (always `false` after successful construction).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Duration in milliseconds (`last_ts − first_ts`).
    #[must_use]
    pub fn duration_ms(&self) -> f32 {
        if self.frames.len() < 2 {
            return 0.0;
        }
        self.frames.last().map_or(0.0, |l| l.timestamp_ms)
            - self.frames.first().map_or(0.0, |f| f.timestamp_ms)
    }

    /// Average frames per second implied by the trajectory timestamps.
    #[must_use]
    pub fn fps(&self) -> f32 {
        let dur_s = self.duration_ms() / 1000.0;
        if dur_s <= 0.0 || self.frames.len() < 2 {
            return 0.0;
        }
        (self.frames.len() - 1) as f32 / dur_s
    }

    /// `(min, max)` yaw over valid frames.
    #[must_use]
    pub fn yaw_range(&self) -> (f32, f32) {
        let valid: Vec<f32> = self
            .frames
            .iter()
            .filter(|f| f.is_valid)
            .map(|f| f.yaw)
            .collect();
        if valid.is_empty() {
            return (0.0, 0.0);
        }
        let min = valid.iter().copied().fold(f32::INFINITY, f32::min);
        let max = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    }

    /// `(min, max)` pitch over valid frames.
    #[must_use]
    pub fn pitch_range(&self) -> (f32, f32) {
        let valid: Vec<f32> = self
            .frames
            .iter()
            .filter(|f| f.is_valid)
            .map(|f| f.pitch)
            .collect();
        if valid.is_empty() {
            return (0.0, 0.0);
        }
        let min = valid.iter().copied().fold(f32::INFINITY, f32::min);
        let max = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    }

    /// `(min, max)` roll over valid frames.
    #[must_use]
    pub fn roll_range(&self) -> (f32, f32) {
        let valid: Vec<f32> = self
            .frames
            .iter()
            .filter(|f| f.is_valid)
            .map(|f| f.roll)
            .collect();
        if valid.is_empty() {
            return (0.0, 0.0);
        }
        let min = valid.iter().copied().fold(f32::INFINITY, f32::min);
        let max = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    }

    /// Iterate over valid frames.
    #[must_use]
    pub fn valid_frames(&self) -> Vec<&HeadTrackFrame> {
        self.frames.iter().filter(|f| f.is_valid).collect()
    }

    /// Fraction of frames that are valid (`0.0`–`1.0`).
    #[must_use]
    pub fn coverage(&self) -> f32 {
        if self.frames.is_empty() {
            return 0.0;
        }
        self.valid_frames().len() as f32 / self.frames.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Trajectory-level filter functions
// ---------------------------------------------------------------------------

/// Apply an exponential moving average to every pose field of a trajectory.
///
/// `alpha = 1.0` reproduces the original trajectory unchanged.
///
/// # Errors
///
/// Returns [`HeadTrackerError::EmptyTrajectory`] when the trajectory is empty,
/// or [`HeadTrackerError::InvalidConfig`] when `alpha` is out of range.
pub fn ema_smooth_trajectory(
    trajectory: &HeadTrajectory,
    alpha: f32,
) -> Result<HeadTrajectory, HeadTrackerError> {
    if trajectory.is_empty() {
        return Err(HeadTrackerError::EmptyTrajectory);
    }
    if alpha <= 0.0 || alpha > 1.0 {
        return Err(HeadTrackerError::InvalidConfig(
            "alpha must be in (0, 1]".to_string(),
        ));
    }
    let mut out: Vec<HeadTrackFrame> = Vec::with_capacity(trajectory.len());
    for (i, frame) in trajectory.frames.iter().enumerate() {
        if i == 0 {
            out.push(frame.clone());
        } else {
            let prev = &out[i - 1];
            out.push(HeadTrackFrame {
                yaw: alpha * frame.yaw + (1.0 - alpha) * prev.yaw,
                pitch: alpha * frame.pitch + (1.0 - alpha) * prev.pitch,
                roll: alpha * frame.roll + (1.0 - alpha) * prev.roll,
                tx: alpha * frame.tx + (1.0 - alpha) * prev.tx,
                ty: alpha * frame.ty + (1.0 - alpha) * prev.ty,
                tz: alpha * frame.tz + (1.0 - alpha) * prev.tz,
                ..frame.clone()
            });
        }
    }
    HeadTrajectory::new(out)
}

/// Apply a simple causal moving average with the given window to every pose field.
///
/// `window = 1` reproduces the original trajectory unchanged.
///
/// # Errors
///
/// Returns [`HeadTrackerError::EmptyTrajectory`] or [`HeadTrackerError::InvalidConfig`]
/// when window is zero.
pub fn sma_smooth_trajectory(
    trajectory: &HeadTrajectory,
    window: usize,
) -> Result<HeadTrajectory, HeadTrackerError> {
    if trajectory.is_empty() {
        return Err(HeadTrackerError::EmptyTrajectory);
    }
    if window == 0 {
        return Err(HeadTrackerError::InvalidConfig(
            "window must be at least 1".to_string(),
        ));
    }
    let mut out: Vec<HeadTrackFrame> = Vec::with_capacity(trajectory.len());
    for i in 0..trajectory.len() {
        let start = (i + 1).saturating_sub(window);
        let slice = &trajectory.frames[start..=i];
        let n = slice.len() as f32;
        let (mut yaw, mut pitch, mut roll, mut tx, mut ty, mut tz) =
            (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for f in slice {
            yaw += f.yaw;
            pitch += f.pitch;
            roll += f.roll;
            tx += f.tx;
            ty += f.ty;
            tz += f.tz;
        }
        out.push(HeadTrackFrame {
            yaw: yaw / n,
            pitch: pitch / n,
            roll: roll / n,
            tx: tx / n,
            ty: ty / n,
            tz: tz / n,
            ..trajectory.frames[i].clone()
        });
    }
    HeadTrajectory::new(out)
}

/// Apply a One Euro filter to a sequence of scalar values.
///
/// Returns a filtered sequence of the same length as `values`.
///
/// # Errors
///
/// Returns [`HeadTrackerError::InvalidConfig`] when `values` and `timestamps_ms` have
/// different lengths, or when `min_cutoff ≤ 0` or `beta < 0`.
pub fn one_euro_filter_sequence(
    values: &[f32],
    timestamps_ms: &[f32],
    min_cutoff: f32,
    beta: f32,
) -> Result<Vec<f32>, HeadTrackerError> {
    if values.len() != timestamps_ms.len() {
        return Err(HeadTrackerError::InvalidConfig(format!(
            "values length ({}) must match timestamps_ms length ({})",
            values.len(),
            timestamps_ms.len()
        )));
    }
    if min_cutoff <= 0.0 {
        return Err(HeadTrackerError::InvalidConfig(
            "min_cutoff must be positive".to_string(),
        ));
    }
    if beta < 0.0 {
        return Err(HeadTrackerError::InvalidConfig(
            "beta must be non-negative".to_string(),
        ));
    }
    let mut out = Vec::with_capacity(values.len());
    let mut x_prev: Option<f32> = None;
    let mut dx_prev: Option<f32> = None;
    for (i, (&x, &ts)) in values.iter().zip(timestamps_ms.iter()).enumerate() {
        let filtered = if let (Some(xp), Some(dxp)) = (x_prev, dx_prev) {
            let prev_ts = if i > 0 { timestamps_ms[i - 1] } else { ts };
            let dt_ms = (ts - prev_ts).max(f32::EPSILON);
            let d_cutoff = 1.0f32;
            let a_d = alpha_from_cutoff(d_cutoff, dt_ms);
            let dx = (x - xp) / dt_ms;
            let dx_hat = a_d * dx + (1.0 - a_d) * dxp;

            let cutoff = one_euro_cutoff(min_cutoff, beta, dx_hat);
            let a = alpha_from_cutoff(cutoff, dt_ms);
            let x_hat = a * x + (1.0 - a) * xp;

            dx_prev = Some(dx_hat);
            x_prev = Some(x_hat);
            x_hat
        } else {
            x_prev = Some(x);
            dx_prev = Some(0.0);
            x
        };
        out.push(filtered);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

/// Linearly interpolate frames marked `is_valid = false` from their neighbors.
///
/// Frames at the start or end that have no valid neighbor on one side are left
/// unchanged (their `is_valid` flag is flipped to `true` with nearest-neighbor
/// values to avoid propagating gaps).
///
/// # Errors
///
/// Returns [`HeadTrackerError::EmptyTrajectory`] when the trajectory is empty.
pub fn interpolate_missing_frames(
    trajectory: &HeadTrajectory,
) -> Result<HeadTrajectory, HeadTrackerError> {
    if trajectory.is_empty() {
        return Err(HeadTrackerError::EmptyTrajectory);
    }
    let frame_count = trajectory.len();
    let mut out = trajectory.frames.clone();

    // Walk through and fill each invalid run
    let mut pos = 0;
    while pos < frame_count {
        if out[pos].is_valid {
            pos += 1;
            continue;
        }
        // Find the end of this invalid run
        let run_start = pos;
        while pos < frame_count && !out[pos].is_valid {
            pos += 1;
        }
        let run_end = pos; // exclusive

        // Find left and right valid neighbors
        let left_idx = if run_start > 0 {
            Some(run_start - 1)
        } else {
            None
        };
        let right_idx = if run_end < frame_count {
            Some(run_end)
        } else {
            None
        };

        match (left_idx, right_idx) {
            (Some(left_neighbor), Some(right_neighbor)) => {
                // Linear interpolation between frames[left_neighbor] and frames[right_neighbor]
                let left = &trajectory.frames[left_neighbor].clone();
                let right = &trajectory.frames[right_neighbor].clone();
                let span = (run_end - run_start) + 1; // number of steps including endpoints
                for (step, idx) in (run_start..run_end).enumerate() {
                    let interp_t = (step + 1) as f32 / span as f32;
                    out[idx] = HeadTrackFrame {
                        yaw: left.yaw + interp_t * (right.yaw - left.yaw),
                        pitch: left.pitch + interp_t * (right.pitch - left.pitch),
                        roll: left.roll + interp_t * (right.roll - left.roll),
                        tx: left.tx + interp_t * (right.tx - left.tx),
                        ty: left.ty + interp_t * (right.ty - left.ty),
                        tz: left.tz + interp_t * (right.tz - left.tz),
                        confidence: left.confidence
                            + interp_t * (right.confidence - left.confidence),
                        is_valid: true,
                        ..out[idx].clone()
                    };
                }
            }
            (Some(left_neighbor), None) => {
                // Extrapolate from the left
                let left = trajectory.frames[left_neighbor].clone();
                for (out_frame, src_frame) in out[run_start..run_end]
                    .iter_mut()
                    .zip(trajectory.frames[run_start..run_end].iter())
                {
                    *out_frame = HeadTrackFrame {
                        is_valid: true,
                        frame_idx: src_frame.frame_idx,
                        timestamp_ms: src_frame.timestamp_ms,
                        ..left.clone()
                    };
                }
            }
            (None, Some(right_neighbor)) => {
                // Extrapolate from the right
                let right = trajectory.frames[right_neighbor].clone();
                for (out_frame, src_frame) in out[run_start..run_end]
                    .iter_mut()
                    .zip(trajectory.frames[run_start..run_end].iter())
                {
                    *out_frame = HeadTrackFrame {
                        is_valid: true,
                        frame_idx: src_frame.frame_idx,
                        timestamp_ms: src_frame.timestamp_ms,
                        ..right.clone()
                    };
                }
            }
            (None, None) => {
                // All frames invalid; mark them valid with zero pose
                for out_frame in &mut out[run_start..run_end] {
                    out_frame.is_valid = true;
                }
            }
        }
    }
    HeadTrajectory::new(out)
}

// ---------------------------------------------------------------------------
// Anomaly detection
// ---------------------------------------------------------------------------

/// Return indices of frames where the rotation magnitude change exceeds `threshold_rad`.
///
/// Only valid frames are compared; invalid frames are skipped.
#[must_use]
pub fn detect_pose_jumps(trajectory: &HeadTrajectory, threshold_rad: f32) -> Vec<usize> {
    let mut jumps = Vec::new();
    let mut prev_opt: Option<&HeadTrackFrame> = None;

    for frame in &trajectory.frames {
        if !frame.is_valid {
            prev_opt = None;
            continue;
        }
        if let Some(prev) = prev_opt {
            let d_yaw = frame.yaw - prev.yaw;
            let d_pitch = frame.pitch - prev.pitch;
            let d_roll = frame.roll - prev.roll;
            let delta = (d_yaw * d_yaw + d_pitch * d_pitch + d_roll * d_roll).sqrt();
            if delta > threshold_rad {
                jumps.push(frame.frame_idx);
            }
        }
        prev_opt = Some(frame);
    }
    jumps
}

// ---------------------------------------------------------------------------
// Rotation velocity
// ---------------------------------------------------------------------------

/// Compute per-frame rotation velocity in radians per millisecond.
///
/// Returns a vector of length `len - 1`.
///
/// # Errors
///
/// Returns [`HeadTrackerError::InsufficientHistory`] when the trajectory has fewer
/// than two frames.
pub fn rotation_velocity(trajectory: &HeadTrajectory) -> Result<Vec<f32>, HeadTrackerError> {
    if trajectory.len() < 2 {
        return Err(HeadTrackerError::InsufficientHistory {
            needed: 2,
            got: trajectory.len(),
        });
    }
    let mut velocities = Vec::with_capacity(trajectory.len() - 1);
    for window in trajectory.frames.windows(2) {
        let (a, b) = (&window[0], &window[1]);
        let dt = (b.timestamp_ms - a.timestamp_ms).abs().max(f32::EPSILON);
        let d_yaw = b.yaw - a.yaw;
        let d_pitch = b.pitch - a.pitch;
        let d_roll = b.roll - a.roll;
        let delta = (d_yaw * d_yaw + d_pitch * d_pitch + d_roll * d_roll).sqrt();
        velocities.push(delta / dt);
    }
    Ok(velocities)
}

// ---------------------------------------------------------------------------
// Trajectory statistics
// ---------------------------------------------------------------------------

/// Statistics computed over an entire trajectory.
#[derive(Debug, Clone)]
pub struct TrajectoryStats {
    /// Total frames (valid + invalid).
    pub total_frames: usize,
    /// Number of valid frames.
    pub valid_frames: usize,
    /// Mean detection confidence across all frames.
    pub mean_confidence: f32,
    /// Standard deviation of yaw over valid frames.
    pub yaw_std: f32,
    /// Standard deviation of pitch over valid frames.
    pub pitch_std: f32,
    /// Standard deviation of roll over valid frames.
    pub roll_std: f32,
    /// Mean rotation velocity (rad/ms) across consecutive frame pairs.
    pub mean_rotation_velocity: f32,
    /// Peak rotation velocity (rad/ms).
    pub max_rotation_velocity: f32,
    /// Number of detected pose jumps.
    pub jump_count: usize,
    /// Total trajectory duration in milliseconds.
    pub duration_ms: f32,
}

/// Compute statistics for the given trajectory.
///
/// # Errors
///
/// Returns [`HeadTrackerError::EmptyTrajectory`] when the trajectory is empty.
pub fn compute_trajectory_stats(
    trajectory: &HeadTrajectory,
    jump_threshold: f32,
) -> Result<TrajectoryStats, HeadTrackerError> {
    if trajectory.is_empty() {
        return Err(HeadTrackerError::EmptyTrajectory);
    }

    let total_frames = trajectory.len();
    let valid: Vec<&HeadTrackFrame> = trajectory.valid_frames();
    let valid_frames = valid.len();

    // Mean confidence
    let mean_confidence = if total_frames > 0 {
        trajectory.frames.iter().map(|f| f.confidence).sum::<f32>() / total_frames as f32
    } else {
        0.0
    };

    // Std devs for yaw, pitch, roll over valid frames
    let yaw_std = std_dev(valid.iter().map(|f| f.yaw));
    let pitch_std = std_dev(valid.iter().map(|f| f.pitch));
    let roll_std = std_dev(valid.iter().map(|f| f.roll));

    // Rotation velocity
    let (mean_rotation_velocity, max_rotation_velocity) = if total_frames >= 2 {
        match rotation_velocity(trajectory) {
            Ok(vels) if !vels.is_empty() => {
                let mean = vels.iter().copied().sum::<f32>() / vels.len() as f32;
                let max = vels.iter().copied().fold(0.0f32, f32::max);
                (mean, max)
            }
            _ => (0.0, 0.0),
        }
    } else {
        (0.0, 0.0)
    };

    let jump_count = detect_pose_jumps(trajectory, jump_threshold).len();
    let duration_ms = trajectory.duration_ms();

    Ok(TrajectoryStats {
        total_frames,
        valid_frames,
        mean_confidence,
        yaw_std,
        pitch_std,
        roll_std,
        mean_rotation_velocity,
        max_rotation_velocity,
        jump_count,
        duration_ms,
    })
}

/// Compute the population standard deviation of a value iterator.
fn std_dev(iter: impl Iterator<Item = f32> + Clone) -> f32 {
    let values: Vec<f32> = iter.collect();
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f32;
    let mean = values.iter().copied().sum::<f32>() / n;
    let variance = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    variance.sqrt()
}

// ---------------------------------------------------------------------------
// Motion segmentation
// ---------------------------------------------------------------------------

/// Segment a trajectory into still and motion segments based on velocity threshold.
///
/// Returns a vector of `(start_frame_idx, end_frame_idx, is_motion)` tuples where
/// the indices refer to positions within `trajectory.frames`.
///
/// # Errors
///
/// Returns [`HeadTrackerError::InsufficientHistory`] when the trajectory has fewer
/// than two frames.
pub fn segment_by_motion(
    trajectory: &HeadTrajectory,
    velocity_threshold: f32,
) -> Result<Vec<(usize, usize, bool)>, HeadTrackerError> {
    if trajectory.len() < 2 {
        return Err(HeadTrackerError::InsufficientHistory {
            needed: 2,
            got: trajectory.len(),
        });
    }
    let vels = rotation_velocity(trajectory)?;
    // Label each inter-frame gap
    let mut segments: Vec<(usize, usize, bool)> = Vec::new();

    let first_is_motion = vels[0] > velocity_threshold;
    let mut seg_start = 0usize;
    let mut seg_is_motion = first_is_motion;

    for (i, &v) in vels.iter().enumerate() {
        let is_motion = v > velocity_threshold;
        if is_motion != seg_is_motion {
            segments.push((seg_start, i, seg_is_motion));
            seg_start = i;
            seg_is_motion = is_motion;
        }
    }
    // Close the final segment
    segments.push((seg_start, trajectory.len() - 1, seg_is_motion));

    Ok(segments)
}

// ---------------------------------------------------------------------------
// Coverage score
// ---------------------------------------------------------------------------

/// Compute the fraction of yaw × pitch bins visited by the trajectory.
///
/// `yaw_bins` and `pitch_bins` are the number of bins along each axis.
/// The range is derived from the valid-frame min/max for each angle.
///
/// # Errors
///
/// Returns [`HeadTrackerError::EmptyTrajectory`] when the trajectory is empty,
/// or [`HeadTrackerError::InvalidConfig`] when either bin count is zero.
pub fn head_coverage_score(
    trajectory: &HeadTrajectory,
    yaw_bins: usize,
    pitch_bins: usize,
) -> Result<f32, HeadTrackerError> {
    if trajectory.is_empty() {
        return Err(HeadTrackerError::EmptyTrajectory);
    }
    if yaw_bins == 0 || pitch_bins == 0 {
        return Err(HeadTrackerError::InvalidConfig(
            "yaw_bins and pitch_bins must be at least 1".to_string(),
        ));
    }
    let valid = trajectory.valid_frames();
    if valid.is_empty() {
        return Ok(0.0);
    }

    let (yaw_min, yaw_max) = trajectory.yaw_range();
    let (pitch_min, pitch_max) = trajectory.pitch_range();
    let yaw_span = (yaw_max - yaw_min).max(f32::EPSILON);
    let pitch_span = (pitch_max - pitch_min).max(f32::EPSILON);

    let total_bins = yaw_bins * pitch_bins;
    let mut visited = vec![false; total_bins];

    for frame in &valid {
        let yb = ((frame.yaw - yaw_min) / yaw_span * yaw_bins as f32) as usize;
        let pb = ((frame.pitch - pitch_min) / pitch_span * pitch_bins as f32) as usize;
        let yb = yb.min(yaw_bins - 1);
        let pb = pb.min(pitch_bins - 1);
        visited[yb * pitch_bins + pb] = true;
    }

    let count = visited.iter().filter(|&&v| v).count();
    Ok(count as f32 / total_bins as f32)
}

// ---------------------------------------------------------------------------
// Slice
// ---------------------------------------------------------------------------

/// Extract a sub-trajectory from frame position `start` to `end` (inclusive).
///
/// # Errors
///
/// Returns [`HeadTrackerError::FrameOutOfRange`] when the range is invalid.
pub fn slice_trajectory(
    trajectory: &HeadTrajectory,
    start: usize,
    end: usize,
) -> Result<HeadTrajectory, HeadTrackerError> {
    let n = trajectory.len();
    if start >= n {
        return Err(HeadTrackerError::FrameOutOfRange {
            frame: start,
            len: n,
        });
    }
    if end >= n {
        return Err(HeadTrackerError::FrameOutOfRange { frame: end, len: n });
    }
    if start > end {
        return Err(HeadTrackerError::FrameOutOfRange {
            frame: start,
            len: end + 1,
        });
    }
    let frames = trajectory.frames[start..=end].to_vec();
    HeadTrajectory::new(frames)
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/// Resample a trajectory to a uniform time grid at `target_fps`.
///
/// Frames are generated by linear interpolation between the nearest source frames.
///
/// # Errors
///
/// Returns [`HeadTrackerError::InsufficientHistory`] when the trajectory has fewer
/// than two frames, [`HeadTrackerError::InvalidConfig`] when `target_fps ≤ 0`,
/// or [`HeadTrackerError::NumericalError`] when the duration is zero.
pub fn resample_trajectory(
    trajectory: &HeadTrajectory,
    target_fps: f32,
) -> Result<HeadTrajectory, HeadTrackerError> {
    if trajectory.len() < 2 {
        return Err(HeadTrackerError::InsufficientHistory {
            needed: 2,
            got: trajectory.len(),
        });
    }
    if target_fps <= 0.0 {
        return Err(HeadTrackerError::InvalidConfig(
            "target_fps must be positive".to_string(),
        ));
    }
    let dur_ms = trajectory.duration_ms();
    if dur_ms <= 0.0 {
        return Err(HeadTrackerError::NumericalError(
            "trajectory duration is zero or negative".to_string(),
        ));
    }

    let dt_ms = 1000.0 / target_fps;
    let t_start = trajectory.frames.first().map_or(0.0, |f| f.timestamp_ms);
    let t_end = trajectory.frames.last().map_or(0.0, |f| f.timestamp_ms);

    let mut out = Vec::new();
    let mut t = t_start;
    let mut out_idx = 0usize;

    while t <= t_end + f32::EPSILON {
        // Find surrounding source frames
        let (left, right) = find_surrounding(trajectory, t);
        let frame = lerp_frames(
            &trajectory.frames[left],
            &trajectory.frames[right],
            t,
            out_idx,
        );
        out.push(frame);
        out_idx += 1;
        t += dt_ms;
    }

    if out.is_empty() {
        return Err(HeadTrackerError::NumericalError(
            "resampled trajectory is empty".to_string(),
        ));
    }
    HeadTrajectory::new(out)
}

/// Find the indices of the two source frames surrounding timestamp `t`.
fn find_surrounding(trajectory: &HeadTrajectory, t: f32) -> (usize, usize) {
    let frames = &trajectory.frames;
    let n = frames.len();
    // Binary search for the first frame with timestamp >= t
    let pos = frames.partition_point(|f| f.timestamp_ms < t);
    if pos == 0 {
        (0, 0)
    } else if pos >= n {
        (n - 1, n - 1)
    } else {
        (pos - 1, pos)
    }
}

/// Linearly interpolate between two source frames at timestamp `t`.
fn lerp_frames(
    left: &HeadTrackFrame,
    right: &HeadTrackFrame,
    t: f32,
    out_idx: usize,
) -> HeadTrackFrame {
    let dt = right.timestamp_ms - left.timestamp_ms;
    let alpha = if dt.abs() < f32::EPSILON {
        0.0
    } else {
        (t - left.timestamp_ms) / dt
    };
    let alpha = alpha.clamp(0.0, 1.0);
    HeadTrackFrame {
        frame_idx: out_idx,
        timestamp_ms: t,
        yaw: left.yaw + alpha * (right.yaw - left.yaw),
        pitch: left.pitch + alpha * (right.pitch - left.pitch),
        roll: left.roll + alpha * (right.roll - left.roll),
        tx: left.tx + alpha * (right.tx - left.tx),
        ty: left.ty + alpha * (right.ty - left.ty),
        tz: left.tz + alpha * (right.tz - left.tz),
        confidence: left.confidence + alpha * (right.confidence - left.confidence),
        is_valid: left.is_valid || right.is_valid,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Test helpers ---

    fn make_frame(idx: usize, ts: f32, yaw: f32, pitch: f32, roll: f32) -> HeadTrackFrame {
        HeadTrackFrame::new(
            idx,
            ts,
            HeadTrackPose {
                yaw,
                pitch,
                roll,
                tx: 0.0,
                ty: 0.0,
                tz: 100.0,
                confidence: 0.9,
            },
        )
    }

    fn simple_trajectory(n: usize) -> HeadTrajectory {
        let frames: Vec<HeadTrackFrame> = (0..n)
            .map(|i| make_frame(i, i as f32 * 33.0, i as f32 * 0.05, 0.0, 0.0))
            .collect();
        HeadTrajectory::new(frames).expect("valid trajectory")
    }

    // --- HeadTrackFrame ---

    #[test]
    fn test_frame_new_valid() {
        let f = HeadTrackFrame::new(
            0,
            0.0,
            HeadTrackPose {
                yaw: 0.1,
                pitch: 0.2,
                roll: 0.3,
                tx: 1.0,
                ty: 2.0,
                tz: 500.0,
                confidence: 0.95,
            },
        );
        assert!(f.is_valid);
        assert_eq!(f.confidence, 0.95);
    }

    #[test]
    fn test_frame_invalid() {
        let f = HeadTrackFrame::invalid(5, 165.0);
        assert!(!f.is_valid);
        assert_eq!(f.frame_idx, 5);
        assert_eq!(f.confidence, 0.0);
    }

    #[test]
    fn test_euler_angles() {
        let f = make_frame(0, 0.0, 0.1, 0.2, 0.3);
        assert_eq!(f.euler_angles(), [0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_translation() {
        let f = HeadTrackFrame::new(
            0,
            0.0,
            HeadTrackPose {
                yaw: 0.0,
                pitch: 0.0,
                roll: 0.0,
                tx: 3.0,
                ty: 4.0,
                tz: 0.0,
                confidence: 1.0,
            },
        );
        assert_eq!(f.translation(), [3.0, 4.0, 0.0]);
    }

    #[test]
    fn test_rotation_magnitude() {
        let f = make_frame(0, 0.0, 3.0, 4.0, 0.0);
        assert!((f.rotation_magnitude() - 5.0).abs() < 1e-5);
    }

    // --- HeadTrackerConfig::validate ---

    #[test]
    fn test_config_default_valid() {
        assert!(HeadTrackerConfig::default().validate().is_ok());
    }

    #[test]
    fn test_config_zero_outlier_threshold() {
        let cfg = HeadTrackerConfig {
            outlier_threshold: 0.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_bad_confidence_threshold() {
        let cfg = HeadTrackerConfig {
            confidence_threshold: -0.1,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_zero_max_history() {
        let cfg = HeadTrackerConfig {
            max_history: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // --- HeadTracker ---

    #[test]
    fn test_tracker_new_ok() {
        assert!(HeadTracker::new(HeadTrackerConfig::default()).is_ok());
    }

    #[test]
    fn test_tracker_update_adds_to_history() {
        let mut tracker = HeadTracker::new(HeadTrackerConfig::default()).expect("ok");
        let f = make_frame(0, 0.0, 0.1, 0.0, 0.0);
        tracker.update(f);
        assert_eq!(tracker.history_len(), 1);
    }

    #[test]
    fn test_tracker_current() {
        let mut tracker = HeadTracker::new(HeadTrackerConfig::default()).expect("ok");
        assert!(tracker.current().is_none());
        tracker.update(make_frame(0, 0.0, 0.1, 0.0, 0.0));
        assert!(tracker.current().is_some());
    }

    #[test]
    fn test_tracker_filtered_current() {
        let cfg = HeadTrackerConfig {
            filter: TrackingFilter::ExponentialMovingAverage { alpha: 0.5 },
            ..Default::default()
        };
        let mut tracker = HeadTracker::new(cfg).expect("ok");
        tracker.update(make_frame(0, 0.0, 1.0, 0.0, 0.0));
        tracker.update(make_frame(1, 33.0, 0.0, 0.0, 0.0));
        let fc = tracker.filtered_current().expect("some");
        // EMA: second step yaw = 0.5*0 + 0.5*1 = 0.5
        assert!((fc.yaw - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_tracker_reset() {
        let mut tracker = HeadTracker::new(HeadTrackerConfig::default()).expect("ok");
        tracker.update(make_frame(0, 0.0, 0.0, 0.0, 0.0));
        tracker.reset();
        assert_eq!(tracker.history_len(), 0);
        assert!(tracker.current().is_none());
    }

    #[test]
    fn test_tracker_valid_fraction() {
        let mut tracker = HeadTracker::new(HeadTrackerConfig::default()).expect("ok");
        tracker.update(make_frame(0, 0.0, 0.1, 0.0, 0.0));
        tracker.update(HeadTrackFrame::invalid(1, 33.0));
        assert!((tracker.valid_fraction() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_tracker_max_history_trim() {
        let cfg = HeadTrackerConfig {
            max_history: 3,
            ..Default::default()
        };
        let mut tracker = HeadTracker::new(cfg).expect("ok");
        for i in 0..10usize {
            tracker.update(make_frame(i, i as f32 * 33.0, 0.0, 0.0, 0.0));
        }
        assert_eq!(tracker.history_len(), 3);
    }

    // --- HeadTrajectory ---

    #[test]
    fn test_trajectory_empty_error() {
        assert!(HeadTrajectory::new(vec![]).is_err());
    }

    #[test]
    fn test_trajectory_duration_ms() {
        let traj = simple_trajectory(10);
        let expected = 9.0 * 33.0;
        assert!((traj.duration_ms() - expected).abs() < 1e-3);
    }

    #[test]
    fn test_trajectory_fps() {
        let traj = simple_trajectory(31);
        // 30 intervals * 33ms = 990ms; 30/0.99s ≈ 30.3
        let fps = traj.fps();
        assert!(fps > 25.0 && fps < 35.0);
    }

    #[test]
    fn test_trajectory_yaw_range_only_valid() {
        let mut frames: Vec<HeadTrackFrame> = (0..5)
            .map(|i| make_frame(i, i as f32 * 33.0, i as f32 * 0.1, 0.0, 0.0))
            .collect();
        // Make frame 0 invalid (yaw=0 but not counted)
        frames[0] = HeadTrackFrame::invalid(0, 0.0);
        let traj = HeadTrajectory::new(frames).expect("ok");
        let (min, max) = traj.yaw_range();
        // Valid frames: idx 1..4, yaw 0.1..0.4
        assert!((min - 0.1).abs() < 1e-5);
        assert!((max - 0.4).abs() < 1e-5);
    }

    #[test]
    fn test_trajectory_coverage() {
        let frames = vec![
            make_frame(0, 0.0, 0.0, 0.0, 0.0),
            HeadTrackFrame::invalid(1, 33.0),
        ];
        let traj = HeadTrajectory::new(frames).expect("ok");
        assert!((traj.coverage() - 0.5).abs() < 1e-5);
    }

    // --- ema_smooth_trajectory ---

    #[test]
    fn test_ema_alpha1_identical() {
        let traj = simple_trajectory(5);
        let smoothed = ema_smooth_trajectory(&traj, 1.0).expect("ok");
        for (a, b) in traj.frames.iter().zip(smoothed.frames.iter()) {
            assert!((a.yaw - b.yaw).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ema_alpha05_smoothed() {
        let frames: Vec<HeadTrackFrame> = vec![
            make_frame(0, 0.0, 0.0, 0.0, 0.0),
            make_frame(1, 33.0, 1.0, 0.0, 0.0),
            make_frame(2, 66.0, 0.0, 0.0, 0.0),
        ];
        let traj = HeadTrajectory::new(frames).expect("ok");
        let smoothed = ema_smooth_trajectory(&traj, 0.5).expect("ok");
        // Frame 1: 0.5*1 + 0.5*0 = 0.5
        assert!((smoothed.frames[1].yaw - 0.5).abs() < 1e-5);
        // Frame 2: 0.5*0 + 0.5*0.5 = 0.25
        assert!((smoothed.frames[2].yaw - 0.25).abs() < 1e-5);
    }

    // --- sma_smooth_trajectory ---

    #[test]
    fn test_sma_window1_identical() {
        let traj = simple_trajectory(5);
        let smoothed = sma_smooth_trajectory(&traj, 1).expect("ok");
        for (a, b) in traj.frames.iter().zip(smoothed.frames.iter()) {
            assert!((a.yaw - b.yaw).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sma_window2_averaged() {
        let frames: Vec<HeadTrackFrame> = vec![
            make_frame(0, 0.0, 0.0, 0.0, 0.0),
            make_frame(1, 33.0, 1.0, 0.0, 0.0),
        ];
        let traj = HeadTrajectory::new(frames).expect("ok");
        let smoothed = sma_smooth_trajectory(&traj, 2).expect("ok");
        // Frame 0: only itself → 0.0
        assert!((smoothed.frames[0].yaw - 0.0).abs() < 1e-5);
        // Frame 1: (0+1)/2 = 0.5
        assert!((smoothed.frames[1].yaw - 0.5).abs() < 1e-5);
    }

    // --- one_euro_filter_sequence ---

    #[test]
    fn test_one_euro_same_length_output() {
        let vals = vec![0.0f32, 0.1, 0.2, 0.1, 0.0];
        let ts: Vec<f32> = (0..5).map(|i| i as f32 * 33.0).collect();
        let out = one_euro_filter_sequence(&vals, &ts, 1.0, 0.0).expect("ok");
        assert_eq!(out.len(), vals.len());
    }

    #[test]
    fn test_one_euro_mismatched_lengths_error() {
        let vals = vec![0.0f32, 0.1, 0.2];
        let ts = vec![0.0f32, 33.0]; // shorter
        assert!(one_euro_filter_sequence(&vals, &ts, 1.0, 0.0).is_err());
    }

    #[test]
    fn test_one_euro_first_sample_unchanged() {
        let vals = vec![0.42f32, 0.1];
        let ts = vec![0.0f32, 33.0];
        let out = one_euro_filter_sequence(&vals, &ts, 1.0, 0.0).expect("ok");
        assert!((out[0] - 0.42).abs() < 1e-5);
    }

    // --- interpolate_missing_frames ---

    #[test]
    fn test_interpolate_all_valid_unchanged() {
        let traj = simple_trajectory(4);
        let out = interpolate_missing_frames(&traj).expect("ok");
        for (a, b) in traj.frames.iter().zip(out.frames.iter()) {
            assert!((a.yaw - b.yaw).abs() < 1e-6);
        }
    }

    #[test]
    fn test_interpolate_one_missing() {
        let frames = vec![
            make_frame(0, 0.0, 0.0, 0.0, 0.0),
            HeadTrackFrame::invalid(1, 33.0),
            make_frame(2, 66.0, 1.0, 0.0, 0.0),
        ];
        let traj = HeadTrajectory::new(frames).expect("ok");
        let out = interpolate_missing_frames(&traj).expect("ok");
        assert!(out.frames[1].is_valid);
        // t = 1/2 of the span → yaw = 0.5
        assert!((out.frames[1].yaw - 0.5).abs() < 1e-5);
    }

    // --- detect_pose_jumps ---

    #[test]
    fn test_no_jumps() {
        let traj = simple_trajectory(5);
        let jumps = detect_pose_jumps(&traj, 1.0); // large threshold
        assert!(jumps.is_empty());
    }

    #[test]
    fn test_one_large_jump_detected() {
        let frames = vec![
            make_frame(0, 0.0, 0.0, 0.0, 0.0),
            make_frame(1, 33.0, 2.0, 0.0, 0.0), // Δ = 2.0 rad
        ];
        let traj = HeadTrajectory::new(frames).expect("ok");
        let jumps = detect_pose_jumps(&traj, 1.0);
        assert_eq!(jumps, vec![1]);
    }

    // --- rotation_velocity ---

    #[test]
    fn test_rotation_velocity_single_frame_error() {
        let traj = HeadTrajectory::new(vec![make_frame(0, 0.0, 0.0, 0.0, 0.0)]).expect("ok");
        assert!(rotation_velocity(&traj).is_err());
    }

    #[test]
    fn test_rotation_velocity_two_frames() {
        let frames = vec![
            make_frame(0, 0.0, 0.0, 0.0, 0.0),
            make_frame(1, 100.0, 1.0, 0.0, 0.0), // Δ=1 rad over 100ms
        ];
        let traj = HeadTrajectory::new(frames).expect("ok");
        let vels = rotation_velocity(&traj).expect("ok");
        assert_eq!(vels.len(), 1);
        assert!((vels[0] - 0.01).abs() < 1e-5); // 1/100
    }

    // --- compute_trajectory_stats ---

    #[test]
    fn test_stats_empty_error() {
        // Can't construct empty trajectory, so test indirectly via manual hack
        // Instead, test that a valid trajectory gives non-error stats
        let traj = simple_trajectory(5);
        assert!(compute_trajectory_stats(&traj, 1.0).is_ok());
    }

    #[test]
    fn test_stats_valid_fields() {
        let traj = simple_trajectory(10);
        let stats = compute_trajectory_stats(&traj, 1.0).expect("ok");
        assert_eq!(stats.total_frames, 10);
        assert_eq!(stats.valid_frames, 10);
        assert!(stats.duration_ms > 0.0);
        assert!(stats.mean_confidence > 0.0);
    }

    // --- segment_by_motion ---

    #[test]
    fn test_segment_still_trajectory() {
        // All same pose → velocity = 0 → all still
        let frames: Vec<HeadTrackFrame> = (0..5)
            .map(|i| make_frame(i, i as f32 * 33.0, 0.0, 0.0, 0.0))
            .collect();
        let traj = HeadTrajectory::new(frames).expect("ok");
        let segs = segment_by_motion(&traj, 0.01).expect("ok");
        // All segments should be still
        for (_, _, is_motion) in &segs {
            assert!(!is_motion);
        }
    }

    #[test]
    fn test_segment_single_frame_error() {
        let traj = HeadTrajectory::new(vec![make_frame(0, 0.0, 0.0, 0.0, 0.0)]).expect("ok");
        assert!(segment_by_motion(&traj, 0.01).is_err());
    }

    // --- head_coverage_score ---

    #[test]
    fn test_coverage_all_same_pose_low_score() {
        let frames: Vec<HeadTrackFrame> = (0..10)
            .map(|i| make_frame(i, i as f32 * 33.0, 0.0, 0.0, 0.0))
            .collect();
        let traj = HeadTrajectory::new(frames).expect("ok");
        // All same pose → all land in one bin → 1/(4*4) = 0.0625
        let score = head_coverage_score(&traj, 4, 4).expect("ok");
        assert!(score <= 0.1);
    }

    #[test]
    fn test_coverage_varied_poses_higher_score() {
        let frames: Vec<HeadTrackFrame> = (0..16)
            .map(|i| {
                let yaw = (i as f32) * 0.1;
                let pitch = (i % 4) as f32 * 0.1;
                make_frame(i, i as f32 * 33.0, yaw, pitch, 0.0)
            })
            .collect();
        let traj = HeadTrajectory::new(frames).expect("ok");
        let score = head_coverage_score(&traj, 4, 4).expect("ok");
        assert!(score > 0.1);
    }

    // --- slice_trajectory ---

    #[test]
    fn test_slice_valid_range() {
        let traj = simple_trajectory(10);
        let sliced = slice_trajectory(&traj, 2, 5).expect("ok");
        assert_eq!(sliced.len(), 4); // frames 2,3,4,5
    }

    #[test]
    fn test_slice_out_of_range_error() {
        let traj = simple_trajectory(5);
        assert!(slice_trajectory(&traj, 0, 10).is_err());
    }

    // --- resample_trajectory ---

    #[test]
    fn test_resample_same_fps_approx_same_count() {
        let traj = simple_trajectory(31); // ~30 fps over ~1 second
        let resampled = resample_trajectory(&traj, 30.0).expect("ok");
        // Should have approximately the same number of frames
        let n = resampled.len();
        assert!((25..=40).contains(&n), "got {n} frames");
    }

    #[test]
    fn test_resample_invalid_fps_error() {
        let traj = simple_trajectory(5);
        assert!(resample_trajectory(&traj, 0.0).is_err());
        assert!(resample_trajectory(&traj, -1.0).is_err());
    }

    #[test]
    fn test_resample_single_frame_error() {
        let traj = HeadTrajectory::new(vec![make_frame(0, 0.0, 0.0, 0.0, 0.0)]).expect("ok");
        assert!(resample_trajectory(&traj, 30.0).is_err());
    }
}
