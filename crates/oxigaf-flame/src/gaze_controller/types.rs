//! Data types for the gaze controller: the error type, gaze/frame
//! representations, detected-event structs, configuration, aggregate
//! statistics, and the stateful [`GazeController`] itself.
//!
//! Split out of the former monolithic `gaze_controller.rs` to stay under
//! the workspace's 2000-line-per-file policy; see [`super::functions`] for
//! the free algorithm functions `GazeController`'s methods call into, and
//! [`super::prng`] for the xorshift64 PRNG used by blink synthesis.

use thiserror::Error;

use super::functions::{
    gz_detect_blinks, gz_detect_fixations, gz_detect_saccades, gz_synthesize_blinks,
};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by gaze controller operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum GazeControllerError {
    /// Input slice is empty when at least one element is required.
    #[error("Empty input: {0}")]
    EmptyInput(String),

    /// A vector that must be non-zero is (near-)zero.
    #[error("Zero vector: {0}")]
    ZeroVector(String),

    /// A configuration parameter is out of its valid range.
    #[error("Invalid config parameter '{name}': {reason}")]
    InvalidConfig { name: &'static str, reason: String },

    /// Requested index or count exceeds available data.
    #[error("Out of range: {0}")]
    OutOfRange(String),

    /// A numerical computation produced a non-finite result.
    #[error("Non-finite result in {0}")]
    NonFinite(String),
}

// ---------------------------------------------------------------------------
// GazeDirection
// ---------------------------------------------------------------------------

/// A gaze direction parameterised as azimuth, elevation, and vergence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GazeDirection {
    /// Horizontal angle in radians. Positive = right (from subject's viewpoint).
    pub azimuth: f32,
    /// Vertical angle in radians. Positive = up.
    pub elevation: f32,
    /// Vergence convergence distance in metres. `0.0` means optical infinity.
    pub vergence: f32,
}

impl GazeDirection {
    /// Create a new `GazeDirection`.
    #[must_use]
    pub fn new(azimuth: f32, elevation: f32, vergence: f32) -> Self {
        Self {
            azimuth,
            elevation,
            vergence,
        }
    }

    /// Primary (straight-ahead) gaze direction: all zeros.
    #[must_use]
    pub fn primary() -> Self {
        Self {
            azimuth: 0.0,
            elevation: 0.0,
            vergence: 0.0,
        }
    }

    /// Convert the azimuth/elevation to a unit Cartesian direction vector.
    /// Convention: +X right, +Y up, +Z forward (into scene).
    #[must_use]
    pub fn to_cartesian(&self) -> [f32; 3] {
        let (s_az, c_az) = self.azimuth.sin_cos();
        let (s_el, c_el) = self.elevation.sin_cos();
        [c_el * s_az, s_el, c_el * c_az]
    }
}

impl Default for GazeDirection {
    fn default() -> Self {
        Self::primary()
    }
}

// ---------------------------------------------------------------------------
// GazeFrame
// ---------------------------------------------------------------------------

/// One temporal snapshot of binocular gaze and blink state.
#[derive(Debug, Clone)]
pub struct GazeFrame {
    /// Monotonically increasing step counter.
    pub step: u64,
    /// Left-eye gaze direction.
    pub left_gaze: GazeDirection,
    /// Right-eye gaze direction.
    pub right_gaze: GazeDirection,
    /// Left-eye blink value: `0.0` = fully open, `1.0` = fully closed.
    pub blink_left: f32,
    /// Right-eye blink value: `0.0` = fully open, `1.0` = fully closed.
    pub blink_right: f32,
    /// Timestamp in milliseconds.
    pub timestamp_ms: f64,
}

impl GazeFrame {
    /// Create a frame with identical left/right gaze and no blink.
    #[must_use]
    pub fn monocular(step: u64, gaze: GazeDirection, timestamp_ms: f64) -> Self {
        Self {
            step,
            left_gaze: gaze,
            right_gaze: gaze,
            blink_left: 0.0,
            blink_right: 0.0,
            timestamp_ms,
        }
    }

    /// Average of left and right gaze azimuths and elevations.
    #[must_use]
    pub fn cyclopean_gaze(&self) -> GazeDirection {
        GazeDirection {
            azimuth: 0.5 * (self.left_gaze.azimuth + self.right_gaze.azimuth),
            elevation: 0.5 * (self.left_gaze.elevation + self.right_gaze.elevation),
            vergence: 0.5 * (self.left_gaze.vergence + self.right_gaze.vergence),
        }
    }

    /// Mean blink value across both eyes.
    #[must_use]
    pub fn mean_blink(&self) -> f32 {
        0.5 * (self.blink_left + self.blink_right)
    }
}

// ---------------------------------------------------------------------------
// Event kinds and enums
// ---------------------------------------------------------------------------

/// Classification of a detected gaze event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GazeEventKind {
    /// Rapid eye movement between fixation targets.
    Saccade,
    /// Stable, near-stationary gaze hold.
    Fixation,
    /// Eyelid closure event.
    Blink,
    /// Smooth pursuit of a moving target.
    Pursuit,
    /// Vergence movement (convergence/divergence).
    Vergence,
}

/// Phase of a blink event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlinkPhase {
    /// Eyelid is moving downward (closing).
    Closing,
    /// Eyelid is moving upward (opening).
    Opening,
    /// Eyelid is fully closed.
    Closed,
}

// ---------------------------------------------------------------------------
// Event structs
// ---------------------------------------------------------------------------

/// A detected saccadic eye movement.
#[derive(Debug, Clone)]
pub struct SaccadeEvent {
    /// Total angular amplitude in degrees of visual angle.
    pub amplitude_deg: f32,
    /// Peak angular velocity in degrees per second.
    pub peak_velocity_dps: f32,
    /// Duration of the saccade in milliseconds.
    pub duration_ms: f32,
    /// Frame step at which the saccade begins.
    pub start_step: u64,
    /// Frame step at which the saccade ends (inclusive).
    pub end_step: u64,
}

/// A detected fixation (stable gaze) event.
#[derive(Debug, Clone)]
pub struct FixationEvent {
    /// Duration of the fixation in milliseconds.
    pub duration_ms: f32,
    /// Spatial dispersion of gaze samples in degrees.
    pub dispersion_deg: f32,
    /// Mean azimuth of gaze samples during fixation.
    pub centroid_az: f32,
    /// Mean elevation of gaze samples during fixation.
    pub centroid_el: f32,
    /// Frame step at which the fixation begins.
    pub start_step: u64,
    /// Frame step at which the fixation ends (inclusive).
    pub end_step: u64,
}

/// A detected blink event.
#[derive(Debug, Clone)]
pub struct BlinkEvent {
    /// Duration of the blink in milliseconds.
    pub duration_ms: f32,
    /// Phase where [`gz_detect_blinks`] finalised this event (not at
    /// `start_step`): a completed excursion finalises at its falling edge
    /// (`Opening`, by construction); one still active at recording's end
    /// gets its phase from the trend of the last two samples instead.
    pub phase: BlinkPhase,
    /// Frame step at which the blink begins.
    pub start_step: u64,
}

// ---------------------------------------------------------------------------
// GazeControllerConfig
// ---------------------------------------------------------------------------

/// Configuration for the [`GazeController`].
#[derive(Debug, Clone)]
pub struct GazeControllerConfig {
    /// Frames per second of the input signal.
    pub fps: f32,
    /// Velocity threshold separating saccades from fixations (deg/s).
    pub saccade_velocity_threshold_dps: f32,
    /// Minimum duration of a fixation to be reported (ms).
    pub fixation_min_duration_ms: f32,
    /// Natural blink rate (blinks per minute) for synthesis.
    pub blink_rate_per_min: f32,
    /// Duration of a synthesised blink (ms).
    pub blink_duration_ms: f32,
    /// Minimum duration of a saccade to be reported (ms). Human saccades
    /// last ~20-80 ms, far shorter than a fixation, so this is a separate
    /// (much smaller) threshold from `fixation_min_duration_ms`.
    pub saccade_min_duration_ms: f32,
}

impl Default for GazeControllerConfig {
    fn default() -> Self {
        Self {
            fps: 60.0,
            saccade_velocity_threshold_dps: 30.0,
            fixation_min_duration_ms: 100.0,
            blink_rate_per_min: 15.0,
            blink_duration_ms: 150.0,
            saccade_min_duration_ms: 20.0,
        }
    }
}

// ---------------------------------------------------------------------------
// GazeStats
// ---------------------------------------------------------------------------

/// Aggregate statistics over a `GazeController`'s history.
#[derive(Debug, Clone)]
pub struct GazeStats {
    /// Total number of gaze frames in history.
    pub n_frames: usize,
    /// Number of detected saccades.
    pub n_saccades: usize,
    /// Number of detected fixations.
    pub n_fixations: usize,
    /// Number of detected blinks.
    pub n_blinks: usize,
    /// Mean fixation duration in milliseconds.
    pub mean_fixation_dur_ms: f32,
    /// Mean saccade amplitude in degrees.
    pub mean_saccade_amplitude_deg: f32,
    /// Blink rate in blinks per minute.
    pub blink_rate_per_min: f32,
    /// Mean vergence distance in metres.
    pub mean_vergence_m: f32,
}

// ---------------------------------------------------------------------------
// GazeController
// ---------------------------------------------------------------------------

/// Stateful gaze controller with ring-buffered history and event detection.
pub struct GazeController {
    /// Configuration used by this controller.
    pub config: GazeControllerConfig,
    /// Recent frame history (capped at 1 000 frames).
    pub history: Vec<GazeFrame>,
    saccades: Vec<SaccadeEvent>,
    fixations: Vec<FixationEvent>,
    blinks: Vec<BlinkEvent>,
}

/// Maximum number of frames kept in the ring buffer.
pub(super) const HISTORY_CAP: usize = 1_000;

impl GazeController {
    /// Create a new `GazeController` with the supplied configuration.
    ///
    /// # Errors
    ///
    /// Returns [`GazeControllerError::InvalidConfig`] if `fps` is non-positive.
    pub fn new(config: GazeControllerConfig) -> Result<Self, GazeControllerError> {
        if config.fps <= 0.0 {
            return Err(GazeControllerError::InvalidConfig {
                name: "fps",
                reason: format!("must be positive, got {}", config.fps),
            });
        }
        Ok(Self {
            config,
            history: Vec::new(),
            saccades: Vec::new(),
            fixations: Vec::new(),
            blinks: Vec::new(),
        })
    }

    /// Push a new `GazeFrame` onto the ring buffer.
    ///
    /// Trims the oldest frames if history would exceed `HISTORY_CAP`.
    pub fn push_frame(&mut self, frame: GazeFrame) {
        self.history.push(frame);
        if self.history.len() > HISTORY_CAP {
            let excess = self.history.len() - HISTORY_CAP;
            self.history.drain(..excess);
        }
    }

    /// Recompute all detected events from the current history.
    pub fn update_events(&mut self) {
        let fps = self.config.fps;
        let vthr = self.config.saccade_velocity_threshold_dps;
        // Distinct thresholds: saccades last ~20-80ms, far shorter than a
        // fixation, so reusing `fixation_min_duration_ms` here discarded
        // nearly every genuine saccade.
        let sacc_min = self.config.saccade_min_duration_ms;
        let fix_min = self.config.fixation_min_duration_ms;
        let blink_thr = 0.5_f32;

        self.saccades = gz_detect_saccades(&self.history, fps, vthr, sacc_min);
        self.fixations = gz_detect_fixations(&self.history, fps, vthr, fix_min);
        self.blinks = gz_detect_blinks(&self.history, fps, blink_thr);
    }

    /// Synthesise blinks over `duration_steps` frames using this
    /// controller's `blink_rate_per_min`/`blink_duration_ms`; `seed` seeds
    /// the PRNG. See [`gz_synthesize_blinks`] for the algorithm.
    #[must_use]
    pub fn synthesize_blinks(&self, duration_steps: usize, seed: u64) -> Vec<f32> {
        gz_synthesize_blinks(
            duration_steps,
            self.config.fps,
            self.config.blink_rate_per_min,
            self.config.blink_duration_ms,
            seed,
        )
    }

    /// Return the most recently detected fixation event, if any.
    #[must_use]
    pub fn current_fixation(&self) -> Option<FixationEvent> {
        self.fixations.last().cloned()
    }

    /// Return `true` if the most recent blink event is still in progress.
    #[must_use]
    pub fn is_blinking(&self) -> bool {
        let Some(last_blink) = self.blinks.last() else {
            return false;
        };
        if self.history.is_empty() {
            return false;
        }
        let last_step = self.history.last().map_or(0, |f| f.step);
        let frames_per_blink = (last_blink.duration_ms * self.config.fps / 1000.0) as u64;
        last_step < last_blink.start_step.saturating_add(frames_per_blink)
    }

    /// Compute the mean cyclopean gaze over the last `n_last` frames.
    ///
    /// # Errors
    ///
    /// Returns [`GazeControllerError::EmptyInput`] when `n_last` is 0 or
    /// the history is empty.
    pub fn mean_gaze(&self, n_last: usize) -> Result<GazeDirection, GazeControllerError> {
        if n_last == 0 {
            return Err(GazeControllerError::EmptyInput("n_last must be > 0".into()));
        }
        if self.history.is_empty() {
            return Err(GazeControllerError::EmptyInput(
                "gaze history is empty".into(),
            ));
        }
        let n = n_last.min(self.history.len());
        let slice = &self.history[self.history.len() - n..];
        let mut sum_az = 0.0_f32;
        let mut sum_el = 0.0_f32;
        let mut sum_vg = 0.0_f32;
        for f in slice {
            let c = f.cyclopean_gaze();
            sum_az += c.azimuth;
            sum_el += c.elevation;
            sum_vg += c.vergence;
        }
        let n_f = n as f32;
        Ok(GazeDirection {
            azimuth: sum_az / n_f,
            elevation: sum_el / n_f,
            vergence: sum_vg / n_f,
        })
    }

    /// Expose detected saccades (read-only).
    #[must_use]
    pub fn saccades(&self) -> &[SaccadeEvent] {
        &self.saccades
    }

    /// Expose detected fixations (read-only).
    #[must_use]
    pub fn fixations(&self) -> &[FixationEvent] {
        &self.fixations
    }

    /// Expose detected blinks (read-only).
    #[must_use]
    pub fn blink_events(&self) -> &[BlinkEvent] {
        &self.blinks
    }
}
