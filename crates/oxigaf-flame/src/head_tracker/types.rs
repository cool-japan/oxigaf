//! Data types for head tracking: the error type, pose/frame representations,
//! filter selection, tracker configuration, the immutable [`HeadTrajectory`]
//! snapshot, and aggregate [`TrajectoryStats`].
//!
//! Split out of the former monolithic `head_tracker.rs` to stay under the
//! workspace's 2000-line-per-file policy. Pure data + accessors only; see
//! [`super::tracker`] for the live, stateful [`super::tracker::HeadTracker`] and
//! [`super::functions`] for the offline trajectory-analysis algorithms.

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

/// Configuration for [`super::tracker::HeadTracker`].
#[derive(Debug, Clone)]
pub struct HeadTrackerConfig {
    /// Temporal filter to apply on each update.
    pub filter: TrackingFilter,
    /// Maximum number of consecutive missing frames to interpolate over.
    /// Used by [`super::tracker::HeadTracker::interpolate_gaps`] / [`super::functions::interpolate_missing_frames`].
    pub max_gap_frames: usize,
    /// Maximum frame-to-frame rotation change (radians) before flagging an
    /// outlier. Enforced live in [`super::tracker::HeadTracker::update`].
    pub outlier_threshold: f32,
    /// Minimum detection confidence to accept a frame as valid. Enforced
    /// live in [`super::tracker::HeadTracker::update`].
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
